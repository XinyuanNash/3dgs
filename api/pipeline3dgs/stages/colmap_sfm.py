"""阶段 3-5:colmap_sfm —— 4 阶段 COLMAP(feature_extractor / exhaustive_matcher / mapper / sparse/0 检查)。

GPU:gpu_index=GPU_INDEX (默认 1),与训练共用同卡。
线程:Mapper.num_threads=4。
"""
from __future__ import annotations
import struct
import shutil

from ..config import job_dir_host, to_container, GPU_INDEX
from ..utils import docker_exec, StageError


def _colmap_prefix(gpu_index: int) -> str:
    """基础 GPU 标记(可作为环境变量或参数使用)。"""
    return f"GPU_INDEX={gpu_index}"


def _count_registered_images(model_dir) -> int:
    """读 model_dir/images.bin 头 8 字节(uint64_t 注册图数)。失败返回 -1。"""
    images_bin = model_dir / "images.bin"
    if not images_bin.exists():
        return -1
    try:
        with open(images_bin, "rb") as f:
            (n,) = struct.unpack("Q", f.read(8))
        return n
    except Exception:
        return -1


def _select_largest_model(sparse_dir):
    """从 sparse_dir/N/ 中选出注册图数最多的模型目录(返回 Path)。

    COLMAP mapper 默认 multiple_models=true,把最小模型写到 sparse/0/,
    较大的写到 sparse/1/、sparse/2/...。下游 (image_undistorter / imgs2poses)
    总读 sparse/0/,所以必须主动选最大的。
    """
    candidates = sorted(
        (
            (d, _count_registered_images(d))
            for d in sparse_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    if not candidates:
        return None, []
    return candidates[0][0], candidates


async def feature_extractor(job_id: str) -> None:
    jdir = job_dir_host(job_id)
    log_path = jdir / "logs" / "stage_04_colmap_feat.log"
    images_dir = jdir / "images"
    db_path = jdir / "database.db"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise StageError("colmap_feat", f"images dir empty or missing: {images_dir}")

    container_images = to_container(images_dir)
    container_db = to_container(db_path)
    container_workdir = to_container(jdir)
    cmd = (
        f"colmap feature_extractor --database_path {container_db} "
        f"--image_path {container_images} "
        f"--FeatureExtraction.use_gpu=1 --FeatureExtraction.gpu_index={GPU_INDEX}"
    )
    await docker_exec(cmd, workdir=container_workdir, log_path=log_path)

    if not db_path.exists() or db_path.stat().st_size < 1024 * 1024:
        raise StageError(
            "colmap_feat",
            f"database.db missing or too small (< 1 MB): {db_path}",
        )


async def exhaustive_matcher(job_id: str) -> None:
    jdir = job_dir_host(job_id)
    log_path = jdir / "logs" / "stage_05_colmap_match.log"
    db_path = jdir / "database.db"

    container_db = to_container(db_path)
    container_workdir = to_container(jdir)
    cmd = (
        f"colmap exhaustive_matcher --database_path {container_db} "
        f"--FeatureMatching.use_gpu=1 --FeatureMatching.gpu_index={GPU_INDEX}"
    )
    await docker_exec(cmd, workdir=container_workdir, log_path=log_path)

    if not db_path.exists():
        raise StageError(
            "colmap_match",
            f"database.db missing after matcher: {db_path}",
        )


async def mapper(job_id: str) -> None:
    jdir = job_dir_host(job_id)
    log_path = jdir / "logs" / "stage_06_colmap_map.log"
    db_path = jdir / "database.db"
    images_dir = jdir / "images"
    sparse_dir = jdir / "sparse"

    container_db = to_container(db_path)
    container_images = to_container(images_dir)
    container_sparse = to_container(sparse_dir)
    container_workdir = to_container(jdir)
    # Mapper flags rationale (yinjian-MyVideo_1-pipeline-2026-09-13 + 2026-09-13 v2):
    #   - ba_use_gpu=1: GPU BA (cuSolverDN Cholesky) — 2026-09-13 用户要求恢复 GPU 路径。
    #     v1 (frame_interval=50, 234 reg) 在该路径下 57min 正常退出,未观察到 Dpotrf 卡死。
    #     早期 "卡死" 描述基于误诊 (见 [[yinjian-myvideo-1-diagnosis-correction]])。
    #   - ba_gpu_index=GPU_INDEX (默认 1): 与训练共用 cuda:1。
    #   - ba_global/local_max_num_iterations=30/20: cap BA iterations per reg step
    #     防止单 local window 长时间拖死 wall-time。
    cmd = (
        f"mkdir -p {container_sparse} && "
        f"colmap mapper --database_path {container_db} "
        f"--image_path {container_images} "
        f"--output_path {container_sparse} "
        f"--Mapper.num_threads=4 "
        f"--Mapper.ba_use_gpu=1 "
        f"--Mapper.ba_gpu_index={GPU_INDEX} "
        f"--Mapper.ba_global_max_num_iterations=30 "
        f"--Mapper.ba_local_max_num_iterations=20"
    )
    await docker_exec(cmd, workdir=container_workdir, log_path=log_path)

    sparse0 = sparse_dir / "0"
    largest, candidates = _select_largest_model(sparse_dir)
    if largest is None:
        raise StageError(
            "colmap_map",
            f"mapper produced no models in {sparse_dir}",
        )
    n_largest = [c for _, c in candidates if c > 0]
    if not n_largest:
        raise StageError(
            "colmap_map",
            f"all mapper models have 0 registered images (candidates="
            f"{[(p.name, n) for p, n in candidates]}); "
            f"video/images likely have too little motion or texture for SfM",
        )
    if largest != sparse0:
        # 把最大的 symlink 到 sparse/0/,下游总读 0/
        if sparse0.exists() or sparse0.is_symlink():
            if sparse0.is_symlink() or sparse0.is_file():
                sparse0.unlink()
            else:
                shutil.rmtree(sparse0)
        sparse0.symlink_to(largest.name)

    cameras_bin = sparse0 / "cameras.bin"
    images_bin = sparse0 / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise StageError(
            "colmap_map",
            f"sparse/0/cameras.bin / images.bin missing after model selection: "
            f"{sparse0} (largest model was {largest}, "
            f"candidates={[(p.name, n) for p, n in candidates]})",
        )
