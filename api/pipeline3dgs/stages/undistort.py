"""阶段 6:undistort —— colmap image_undistorter。

输入:images/ + sparse/0/
输出:distort_free/{images/, sparse/0/}
删除:distort_free/stereo/ (不需要)

【camera model 归一化】image_undistorter 之后,如果 distort_free/sparse/0/cameras.txt
含有 SIMPLE_RADIAL / SIMPLE_BROWN / SIMPLE_RADIAL_FISHEYE 等带径向畸变的模型,
GLOMAP / COLMAP 都会保留它们(即使 k1≈0)。3DGS 的 scene/dataset_readers.py:194
对 SIMPLE_PINHOLE / PINHOLE 之外的模型 hard-reject,直接 AssertionError 崩训练。

修复策略:
  1. model_converter --output_type TXT  → cameras.txt
  2. SIMPLE_RADIAL → SIMPLE_PINHOLE (drop k1);其它畸变模型 → SIMPLE_PINHOLE (drop 所有
     非 fx/fy/cx/cy);PINHOLE 不动
  3. model_converter --output_type BIN  → cameras.bin
  4. 同样的归一化对 sparse/0/cameras.txt 也跑一遍(imgs2poses 会读 COLMAP 输出)

⚠️ 已知代价:丢掉真实径向畸变 → 视频输出会有轻度 barrel / pincushion 残留。3DGS 训练
分支上没有 in-network distortion correction,这是 3DGS 唯一可行的选项。
"""
from __future__ import annotations
import shutil

from ..config import job_dir_host, to_container
from ..utils import docker_exec, StageError


# 把带径向畸变的 camera model 归一化到 SIMPLE_PINHOLE。
# 保留 fx, fy, cx, cy(fx == fy 假设;若不是,后续可改写)
_PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_PINHOLE_FISHEYE"}
# model 名 → 它的 params 个数(去掉畸变只保留 fx/fy/cx/cy 需要的裁剪长度)
_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,        # f, cx, cy
    "PINHOLE": 4,               # fx, fy, cx, cy
    "SIMPLE_RADIAL": 4,         # f, cx, cy, k1
    "RADIAL": 5,                # f, cx, cy, k1, k2
    "SIMPLE_BROWN": 5,          # f, cx, cy, k1, k2
    "BROWN": 7,                 # fx, fy, cx, cy, k1, k2, k3
    "FISHEYE": 8,               # fx, fy, cx, cy, k1, k2, k3, k4
    "SIMPLE_FISHEYE": 4,        # f, cx, cy, k1
    "RADIAL_FISHEYE": 5,        # f, cx, cy, k1, k2
    "THIN_PRISM_FISHEYE": 12,   # fx, fy, cx, cy, k1..k6, p1, p2
}


def _normalize_cameras_in_dir(sparse_dir, log_lines: list[str]) -> int:
    """把 sparse_dir/cameras.{txt,bin} 里的畸变模型归一化到 SIMPLE_PINHOLE / PINHOLE。

    **in-memory 处理**,直接读 cameras.bin → 改 → 写回 cameras.bin。
    避开 `colmap model_converter`(在 frame-based reconstruction + 只有
    frames.bin/images.bin/points3D.bin 的目录上 SIGABRT 崩)。

    返回被改写的 camera 数。
    """
    cameras_bin = sparse_dir / "cameras.bin"
    if not cameras_bin.exists():
        return 0

    # COLMAP 4.x cameras.bin 格式:u64 num + 每 camera: i32 id, i32 model_id, u64 w, u64 h, f64*params
    # model_id: SIMPLE_PINHOLE=0, PINHOLE=1, SIMPLE_RADIAL=2, ...
    MODEL_ID = {
        "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
        "RADIAL": 3, "SIMPLE_BROWN": 4, "BROWN": 5,
        "FISHEYE": 6, "SIMPLE_FISHEYE": 7, "RADIAL_FISHEYE": 8,
        "THIN_PRISM_FISHEYE": 9,
    }
    MODEL_NUM_PARAMS_BIN = {
        0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 7, 6: 8, 7: 4, 8: 5, 9: 12,
    }

    import struct as _st
    data = cameras_bin.read_bytes()
    if len(data) < 8:
        return 0
    (num_cameras,) = _st.unpack_from("<Q", data, 0)
    out = bytearray()
    out += _st.pack("<Q", num_cameras)
    offset = 8
    n_changed = 0
    for _ in range(num_cameras):
        cam_id, model_id, width, height = _st.unpack_from("<iiQQ", data, offset)
        offset += 24
        n_params = MODEL_NUM_PARAMS_BIN.get(model_id, 4)
        params = _st.unpack_from(f"<{n_params}d", data, offset)
        offset += 8 * n_params

        # model_name 推断
        model_name = next(
            (n for n, i in MODEL_ID.items() if i == model_id),
            None,
        )
        if model_name is None or model_name in _PINHOLE_MODELS:
            # 不改,直接复制原 bytes
            out += _st.pack("<iiQQ", cam_id, model_id, width, height)
            out += _st.pack(f"<{n_params}d", *params)
            continue

        # SIMPLE_RADIAL (id=2) → SIMPLE_PINHOLE (id=0): 保留前 3 个 params (f, cx, cy)
        if model_name == "SIMPLE_RADIAL":
            new_model_id = 0  # SIMPLE_PINHOLE
            new_params = params[:3]
        else:
            # 其他畸变模型 → PINHOLE (id=1): 保留 fx, fy, cx, cy (前 4 个 params)
            new_model_id = 1  # PINHOLE
            new_params = params[:4]

        out += _st.pack("<iiQQ", cam_id, new_model_id, width, height)
        out += _st.pack(f"<{len(new_params)}d", *new_params)
        n_changed += 1
        log_lines.append(
            f"  camera {cam_id}: {model_name} (id={model_id}) -> "
            f"{'SIMPLE_PINHOLE' if new_model_id == 0 else 'PINHOLE'} "
            f"(id={new_model_id}) (dropped {n_params - len(new_params)} distortion params)"
        )

    if n_changed == 0:
        return 0

    # 写 cameras.bin(atomic: 临时文件 + rename)
    tmp = cameras_bin.with_suffix(".bin.tmp")
    tmp.write_bytes(bytes(out))
    tmp.replace(cameras_bin)
    # 同步写 cameras.txt(给人看,3dgs 不读但训练时方便 debug)
    cameras_txt = sparse_dir / "cameras.txt"
    txt_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {num_cameras} (normalized in-place from cameras.bin)",
        "",
    ]
    # 重新解析写好的 bin 生成 txt
    data2 = cameras_bin.read_bytes()
    (n,) = _st.unpack_from("<Q", data2, 0)
    off = 8
    inv = {v: k for k, v in MODEL_ID.items()}
    for _ in range(n):
        cam_id, model_id, width, height = _st.unpack_from("<iiQQ", data2, off)
        off += 24
        n_p = MODEL_NUM_PARAMS_BIN.get(model_id, 4)
        params = _st.unpack_from(f"<{n_p}d", data2, off)
        off += 8 * n_p
        txt_lines.append(
            " ".join([str(cam_id), inv.get(model_id, "?"), str(width), str(height),
                       *(f"{p:.10g}" for p in params)])
        )
    cameras_txt.write_text("\n".join(txt_lines) + "\n")
    log_lines.append(
        f"  normalized {n_changed} cameras to SIMPLE_PINHOLE / PINHOLE"
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
        # ===== frame-based model fallback 2026-09-11 =====
        # COLMAP 4.x mapper 默认 ba_refine_sensor_from_rig=true 时输出 frame-based
        # reconstruction(带 frames.bin / rigs.bin),image_undistorter 写出 reconstruction
        # 到 distort_free/sparse/ 时**不复制 cameras.bin**(它假定 cameras 由 frames 内联)。
        # 但 3dgs dataset_readers.py 只看 cameras.bin,所以必须从 input 复制过来。
        # cameras.bin 内含 intrinsics,与图像大小无关(undistorted images 大小同 input),
        # 复制即可,无需重新算。
        if not out_cameras.exists():
            input_cameras = sparse0 / "cameras.bin"
            if input_cameras.exists():
                shutil.copy2(str(input_cameras), str(out_cameras))
                log_lines_dbg = [f"  copied cameras.bin from {sparse0} (frame-based fallback)"]
            if not out_cameras.exists():
                raise StageError(
                    "undistort",
                    f"distort_free/sparse/cameras.bin missing: {out_sparse}",
                )

    # 删 stereo/(我们不需要 stereo depth,占空间)
    stereo = distort_free / "stereo"
    if stereo.exists():
        shutil.rmtree(stereo)

    # ===== camera model 归一化到 PINHOLE family =====
    # image_undistorter 不一定消除所有径向畸变(GLOMAP 输出尤其明显),
    # 3DGS dataset_readers.py:194 会 hard-reject 非 PINHOLE 模型。
    # **2026-09-11**:对 3 个目标全部归一化(distort_free/sparse/{cameras.bin,
    # 0/cameras.bin},sparse/0/cameras.bin),因为 COLMAP 4.x 双 layout + legacy
    # fallback 移动会把 cameras.bin 留在不同位置,3DGS 实际读的是
    # `distort_free/sparse/0/cameras.bin`(dataset_readers.py:148)。
    # 改用 in-memory cameras.bin 读写,避开 `colmap model_converter` —
    # 在 frame-based reconstruction + 没有 cameras.bin 的目录上 model_converter SIGABRT 崩。
    targets_to_normalize = [out_sparse]
    sparse0_legacy = out_sparse / "0"
    if sparse0_legacy.is_dir() and (sparse0_legacy / "cameras.bin").exists():
        targets_to_normalize.append(sparse0_legacy)
    targets_to_normalize.append(sparse0)

    log_lines: list[str] = ["", "=== camera model 归一化(SIMPLE_RADIAL 等 → SIMPLE_PINHOLE) ==="]
    try:
        for target in targets_to_normalize:
            label = target.relative_to(jdir) if target.is_relative_to(jdir) else target
            log_lines.append(f"--- {label} ---")
            n = _normalize_cameras_in_dir(target, log_lines)
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
