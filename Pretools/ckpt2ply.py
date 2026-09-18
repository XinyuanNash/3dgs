import torch
import numpy as np
from plyfile import PlyData, PlyElement
import argparse

def main():
    parser = argparse.ArgumentParser(description="gsplat ckpt -> 3DGS standard PLY")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    ckpt = torch.load(args.ckpt, map_location=device)
    splats = ckpt["splats"]

    # 读取参数
    means     = splats["means"].detach().cpu().numpy()
    opacities = splats["opacities"].detach().cpu().numpy()
    quats     = splats["quats"].detach().cpu().numpy()
    scales    = splats["scales"].detach().cpu().numpy()
    sh0       = splats["sh0"].detach().cpu().numpy()
    shN       = splats["shN"].detach().cpu().numpy()

    num_gauss = means.shape[0]
    print(f"Total Gaussians: {num_gauss}")

    # 维度对齐：opacities 转为 [N,1]
    opacities = opacities.reshape(-1, 1)

    normals = np.zeros_like(means)
    f_dc = sh0.transpose(0, 2, 1).reshape(num_gauss, -1)
    f_rest = shN.transpose(0, 2, 1).reshape(num_gauss, -1)

    # 拼接所有字段
    full_attr = np.concatenate([
        means, normals, f_dc, f_rest, opacities, scales, quats
    ], axis=1)

    # 构造PLY字段名
    attr_names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(f_dc.shape[1]):
        attr_names.append(f"f_dc_{i}")
    for i in range(f_rest.shape[1]):
        attr_names.append(f"f_rest_{i}")
    attr_names.append("opacity")
    for i in range(scales.shape[1]):
        attr_names.append(f"scale_{i}")
    for i in range(quats.shape[1]):
        attr_names.append(f"rot_{i}")

    # 写出PLY
    ply_dtype = [(name, "f4") for name in attr_names]
    ply_array = np.empty(num_gauss, dtype=ply_dtype)
    ply_array[:] = list(map(tuple, full_attr))
    vertex = PlyElement.describe(ply_array, "vertex")
    PlyData([vertex]).write(args.output)
    print(f"✅ PLY saved to: {args.output}")

if __name__ == "__main__":
    main()