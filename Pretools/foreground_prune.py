"""后处理 3DGS PLY：用相机凸包剪枝低 opacity 背景。

可选用 ``--also-split`` 在 PLY 上重跑一次 ``gs_splitter``。

流水线：
    ckpt2ply.py -> (可选) foreground_prune.py [--also-split] -> viewer
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

# 复用 in-training 模块中的凸包构建与成员判定
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(_PROJECT_ROOT, "Difix3d-3dgs-demo", "examples", "gsplat"),
)
from foreground_mask import build_camera_hull, inside_hull_mask  # type: ignore


def load_camera_centers(data_dir: str) -> np.ndarray:
    """自动检测 COLMAP ``sparse/0/images.txt`` 或 ``transforms.json``，返回 ``(N,4,4)`` camtoworlds。"""
    colmap = os.path.join(data_dir, "sparse", "0", "images.txt")
    trans = os.path.join(data_dir, "transforms.json")
    if os.path.exists(colmap):
        with open(colmap) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        Ts = []
        for i in range(0, len(lines), 2):
            toks = lines[i].split()
            qw, qx, qy, qz, tx, ty, tz = map(float, toks[1:8])
            R = qvec2rot(np.array([qw, qx, qy, qz]))
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            Ts.append(c2w)
        return np.stack(Ts)
    if os.path.exists(trans):
        with open(trans) as f:
            meta = json.load(f)
        return np.array([f["transform_matrix"] for f in meta["frames"]])
    raise FileNotFoundError(
        f"No COLMAP images.txt or transforms.json in {data_dir}"
    )


def qvec2rot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ply", help="输入 3DGS PLY（ckpt2ply.py 的输出）")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--hull-opa", type=float, default=0.5,
                    help="凸包外保留 alpha 的阈值（默认 0.5）")
    ap.add_argument("--also-split", action="store_true",
                    help="凸包剪枝后，调用 gs_splitter.process_ply 再拆一次大高斯（训练后可选）")
    args = ap.parse_args()

    ply = PlyData.read(args.ply)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)

    camtoworlds = load_camera_centers(args.data_dir)
    delaunay, cam_centers = build_camera_hull(camtoworlds)
    inside = inside_hull_mask(xyz, delaunay, cam_centers)

    opa_field = next(
        (f for f in v.data.dtype.names if "opacity" in f.lower() or f == "opacity"),
        None,
    )
    if opa_field is None:
        raise RuntimeError("PLY 中找不到 opacity 字段")
    opa = np.array(v[opa_field])
    # 标准 3DGS PLY 的 opacity 字段是 logit；若是 logit，sigmoid 后再比较
    alpha = 1.0 / (1.0 + np.exp(-opa)) if opa_field == "opacity" else opa

    keep = inside | (alpha >= args.hull_opa)
    print(
        f"Hull-prune: 保留 {keep.sum()}/{len(keep)} 个高斯点 "
        f"(删除 {(1 - keep.mean()) * 100:.1f}%)"
    )

    new_v = ply["vertex"].data[keep]
    out = args.ply.replace(".ply", "_fgpruned.ply")
    PlyData([PlyElement.describe(new_v, "vertex")]).write(out)
    print(f"输出: {out}")

    if args.also_split:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gs_splitter import default_params, process_ply  # type: ignore
        split_out = process_ply(out, default_params("balanced"))
        print(f"Split done: {split_out}")


if __name__ == "__main__":
    main()
