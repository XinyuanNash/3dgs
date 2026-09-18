"""阶段 6:undistort —— colmap image_undistorter。

输入:images/ + sparse/0/
输出:distort_free/{images/, sparse/0/}
删除:distort_free/stereo/ (不需要)

【camera model 归一化】image_undistorter 之后,如果 distort_free/sparse/cameras.txt
含有 SIMPLE_RADIAL / SIMPLE_BROWN / SIMPLE_RADIAL_FISHEYE 等带径向畸变模型,
GLOMAP / COLMAP 都会保留它们(即使 k1≈0)。3DGS 的 scene/dataset_readers.py:194
对 SIMPLE_PINHOLE / PINHOLE 之外的模型 hard-reject,直接 AssertionError 崩训练。

修复策略(2026-09-11 改:不走 model_converter):
  1. 直接 struct 解 cameras.bin → normalize → 写回 cameras.bin
  2. 同时生成 cameras.txt(imgs2poses 会读)
  3. 同样的归一化对 sparse/0/cameras.bin 也跑一遍

为什么不再用 `colmap model_converter`:
  COLMAP 4.1.1 (rig-aware) 下,image_undistorter --output_type COLMAP 会**不写出**
  cameras.bin(只写 frames.bin images.bin points3D.bin rigs.bin)。这之后
  `model_converter --output_type TXT` 没有 .bin 可读 → 内部 assertion 失败 → abort()
  → SIGABRT (exit -6)。直接读 .bin 避开这个依赖,且不依赖 COLMAP 写出特定文件。

⚠️ 已知代价:丢掉真实径向畸变 → 视频输出会有轻度 barrel / pincushion 残留。3DGS 训练
分支上没有 in-network distortion correction,这是 3DGS 唯一可行的选项。
"""
from __future__ import annotations
import shutil
import struct
from dataclasses import dataclass

from ..config import job_dir_host, to_container
from ..utils import docker_exec, StageError


# COLMAP CameraModelId 枚举(从 src/colmap/scene/camera_models.h 提取,4.x):
#   0 SIMPLE_PINHOLE  1 PINHOLE  2 SIMPLE_RADIAL  3 RADIAL
#   4 OPENCV  5 OPENCV_FISHEYE  6 FULL_OPENCV  7 FOV
#   8 SIMPLE_RADIAL_FISHEYE  9 RADIAL_FISHEYE  10 THIN_PRISM_FISHEYE
#   11 INFINIRY(4.x 新增)
_MODEL_ID_TO_NAME = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE",
    2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE", 11: "INFINIRY",
}
_NAME_TO_MODEL_ID = {v: k for k, v in _MODEL_ID_TO_NAME.items()}

# 每个 model 的 params 长度(对应 C++ CameraModelNumParams)
_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,        # f, cx, cy
    "PINHOLE": 4,               # fx, fy, cx, cy
    "SIMPLE_RADIAL": 4,         # f, cx, cy, k1
    "RADIAL": 5,                # f, cx, cy, k1, k2
    "SIMPLE_BROWN": 5,          # (legacy) f, cx, cy, k1, k2
    "BROWN": 7,                 # fx, fy, cx, cy, k1, k2, k3
    "OPENCV": 8,                # fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV_FISHEYE": 8,        # fx, fy, cx, cy, k1, k2, k3, k4
    "FULL_OPENCV": 12,          # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    "FOV": 5,                   # fx, fy, cx, cy, omega
    "SIMPLE_RADIAL_FISHEYE": 4, # f, cx, cy, k1
    "RADIAL_FISHEYE": 5,        # f, cx, cy, k1, k2
    "SIMPLE_FISHEYE": 4,        # (legacy) f, cx, cy, k1
    "FISHEYE": 8,               # (legacy) fx, fy, cx, cy, k1, k2, k3, k4
    "THIN_PRISM_FISHEYE": 12,   # fx, fy, cx, cy, k1..k6, p1, p2
    "INFINIRY": 4,              # (4.x 新) f, cx, cy, k
}
# 把带径向畸变的 camera model 归一化到 SIMPLE_PINHOLE / PINHOLE。
_PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_PINHOLE_FISHEYE"}


@dataclass
class _Camera:
    camera_id: int
    model_name: str
    width: int
    height: int
    params: tuple  # tuple[float, ...]


def _read_cameras_bin(path) -> list[_Camera]:
    """直接 struct 解 cameras.bin(对应 COLMAP WriteCamerasBinary)。"""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack_from("<Q", data, 0)[0]
    off = 8
    out: list[_Camera] = []
    for _ in range(n):
        cam_id, model_id = struct.unpack_from("<Ii", data, off); off += 8
        w, h = struct.unpack_from("<QQ", data, off); off += 16
        name = _MODEL_ID_TO_NAME.get(model_id)
        if name is None:
            raise StageError(
                "undistort",
                f"unknown CameraModelId={model_id} for camera {cam_id}",
            )
        n_params = _MODEL_NUM_PARAMS.get(name)
        if n_params is None:
            raise StageError(
                "undistort",
                f"unknown model {name!r} for camera {cam_id} (add to _MODEL_NUM_PARAMS)",
            )
        params = struct.unpack_from(f"<{n_params}d", data, off)
        off += 8 * n_params
        out.append(_Camera(cam_id, name, w, h, params))
    return out


def _write_cameras_bin(path, cameras: list[_Camera]) -> None:
    """写 COLMAP cameras.bin(WriteCamerasBinary 兼容)。"""
    buf = bytearray()
    buf += struct.pack("<Q", len(cameras))
    for cam in cameras:
        model_id = _NAME_TO_MODEL_ID.get(cam.model_name)
        if model_id is None:
            raise StageError(
                "undistort",
                f"cannot serialize unknown model {cam.model_name!r} "
                f"for camera {cam.camera_id}",
            )
        n_params = _MODEL_NUM_PARAMS[cam.model_name]
        if len(cam.params) != n_params:
            raise StageError(
                "undistort",
                f"camera {cam.camera_id} ({cam.model_name}): "
                f"expected {n_params} params, got {len(cam.params)}",
            )
        buf += struct.pack("<Ii", cam.camera_id, model_id)
        buf += struct.pack("<QQ", cam.width, cam.height)
        buf += struct.pack(f"<{n_params}d", *cam.params)
    with open(path, "wb") as f:
        f.write(buf)


def _write_cameras_txt(path, cameras: list[_Camera]) -> None:
    """写 COLMAP cameras.txt(imgs2poses / 3DGS readers 都会读)。"""
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for cam in cameras:
        param_str = " ".join(f"{p:.12g}" for p in cam.params)
        lines.append(
            f"{cam.camera_id} {cam.model_name} {cam.width} {cam.height} {param_str}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _normalize_cameras_bin(sparse_dir, log_lines: list[str]) -> int:
    """直接对 cameras.bin 做 normalize,不需要 model_converter。

    读 .bin → 把非 PINHOLE family 切到 SIMPLE_PINHOLE / PINHOLE → 写回 .bin + .txt。
    返回被改写的 camera 数。
    """
    bin_path = sparse_dir / "cameras.bin"
    if not bin_path.exists():
        return 0

    cameras = _read_cameras_bin(bin_path)
    new_cameras: list[_Camera] = []
    n_changed = 0
    for cam in cameras:
        if cam.model_name in _PINHOLE_MODELS:
            new_cameras.append(cam)
            continue
        # 切到 SIMPLE_PINHOLE: 保留前 3 个 params (f, cx, cy)
        # (SIMPLE_PINHOLE 要求 fx == fy;若不是可以走 PINHOLE,这里简化)
        keep = _MODEL_NUM_PARAMS.get(cam.model_name, 4)
        if cam.model_name == "SIMPLE_RADIAL":
            keep = 3  # f, cx, cy → SIMPLE_PINHOLE
        new_params = cam.params[:keep]
        new_model = (
            "SIMPLE_PINHOLE" if cam.model_name.startswith("SIMPLE_")
            else "PINHOLE"
        )
        new_cameras.append(_Camera(
            cam.camera_id, new_model, cam.width, cam.height, new_params,
        ))
        log_lines.append(
            f"  camera {cam.camera_id}: {cam.model_name} -> {new_model} "
            f"(dropped {len(cam.params) - keep} distortion params)"
        )
        n_changed += 1

    if n_changed == 0:
        return 0

    _write_cameras_bin(bin_path, new_cameras)
    _write_cameras_txt(sparse_dir / "cameras.txt", new_cameras)
    log_lines.append(
        f"  normalized {n_changed} cameras to SIMPLE_PINHOLE / PINHOLE "
        f"(+ wrote cameras.txt)"
    )
    return n_changed


async def run(job_id: str) -> None:
    jdir = job_dir_host(job_id)
    log_path = jdir / "logs" / "stage_08_undistort.log"
    images_dir = jdir / "images"
    sparse0 = jdir / "sparse" / "0"
    distort_free = jdir / "distort_free"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise StageError("undistort", f"images dir missing or empty: {images_dir}")
    if not (sparse0 / "cameras.bin").exists():
        raise StageError("undistort", f"sparse/0/cameras.bin missing: {sparse0}")

    distort_free.mkdir(exist_ok=True)

    container_images = to_container(images_dir)
    container_input = to_container(sparse0)
    container_output = to_container(distort_free)
    container_workdir = to_container(jdir)
    cmd = (
        f"mkdir -p {container_output} && "
        f"colmap image_undistorter --image_path {container_images} "
        f"--input_path {container_input} "
        f"--output_path {container_output} "
        f"--output_type COLMAP"
    )
    await docker_exec(cmd, workdir=container_workdir, log_path=log_path)

    out_sparse = distort_free / "sparse"
    out_cameras = out_sparse / "cameras.bin"
    if not out_cameras.exists():
        # COLMAP 较新版本把 cameras.bin / images.bin 直接写到 distort_free/sparse/
        # 较旧版本写到 distort_free/sparse/0/。容忍两种 layout,做 normalize。
        legacy_dir = out_sparse / "0"
        if legacy_dir.is_dir():
            needed_bins = ["cameras.bin", "images.bin", "points3D.bin"]
            for name in needed_bins:
                src = legacy_dir / name
                dst = out_sparse / name
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
        # COLMAP 4.1.1 rig-aware 模式可能根本不写 cameras.bin(只写
        # frames.bin images.bin points3D.bin rigs.bin)。这种情况下从
        # 输入 sparse/0/cameras.bin 拷过来再做 normalize —— image_undistorter
        # 对 PINHOLE 系相机的内参不变,丢失畸变参数在 normalize 里也会被丢掉。
        if not out_cameras.exists():
            src_cameras = sparse0 / "cameras.bin"
            if src_cameras.exists():
                shutil.copy2(str(src_cameras), str(out_cameras))
                # 同时从 log 里记录一下
                log_extra = (
                    "\n[undistort] NOTE: COLMAP 4.1.1 rig-aware output did not "
                    "write cameras.bin; copied from sparse/0/cameras.bin. "
                    "Intrinsics preserved as-is from COLMAP SfM (no undistortion "
                    "correction for non-PINHOLE models — normalization below "
                    "drops distortion params).\n"
                )
                with open(log_path, "a") as f:
                    f.write(log_extra)
            else:
                raise StageError(
                    "undistort",
                    f"distort_free/sparse/cameras.bin missing and no "
                    f"sparse/0/cameras.bin to copy: {out_sparse}",
                )

    # 删 stereo/(我们不需要 stereo depth,占空间)
    stereo = distort_free / "stereo"
    if stereo.exists():
        shutil.rmtree(stereo)

    # ===== camera model 归一化到 PINHOLE family =====
    # image_undistorter 不一定消除所有径向畸变(GLOMAP 输出尤其明显),
    # 3DGS dataset_readers.py:194 会 hard-reject 非 PINHOLE 模型。
    # 在 distort_free/sparse/ 和 sparse/0/ 两处都归一化(imgs2poses 会读 COLMAP 输出)。
    log_lines: list[str] = ["", "=== camera model 归一化(SIMPLE_RADIAL 等 → SIMPLE_PINHOLE) ==="]
    try:
        for label, target in [("distort_free/sparse", out_sparse),
                                ("sparse/0", sparse0)]:
            log_lines.append(f"--- {label} ---")
            n = _normalize_cameras_bin(target, log_lines)
            if n == 0:
                log_lines.append("  (no cameras.bin or already PINHOLE family)")
    except Exception as e:
        log_lines.append(f"camera model 归一化失败(非致命): {e!r}")
        # log_path 还没写过(image_undistorter 已写一份;但万一失败就用纯文本)
        try:
            existing = log_path.read_text()
        except FileNotFoundError:
            existing = ""
        log_path.write_text(existing + "\n".join(log_lines) + "\n")
        return

    try:
        existing = log_path.read_text()
    except FileNotFoundError:
        existing = ""
    log_path.write_text(existing + "\n".join(log_lines) + "\n")
