"""阶段 7:imgs2poses —— 生成 LLFF 格式 poses_bounds.npy。

调用 Pretools/imgs2poses.py。需要 PYTHONPATH=/home/Pretools 让
`from llff.poses.pose_utils import gen_poses` 工作。

由于 distort_free/sparse/0/ 已经填好,gen_poses 会打印
"Don't need to run COLMAP" 并跳过内部 COLMAP(避免重复工作)。
"""
from __future__ import annotations
import struct

from ..config import (
    CONTAINER_PRETOOLS,
    CONTAINER_PYTHON,
    job_dir_host,
    to_container,
)
from ..utils import docker_exec, StageError


def _count_registered_images(sparse0_path) -> int:
    """读 images.bin 头 8 字节(注册图数)。失败返回 -1。"""
    images_bin = sparse0_path / "images.bin"
    if not images_bin.exists():
        return -1
    try:
        with open(images_bin, "rb") as f:
            (n,) = struct.unpack("Q", f.read(8))
        return n
    except Exception:
        return -1


async def run(job_id: str) -> None:
    jdir = job_dir_host(job_id)
    distort_free = jdir / "distort_free"
    log_path = jdir / "logs" / "stage_09_imgs2poses.log"
    sparse_dir = distort_free / "sparse"
    if not (sparse_dir / "cameras.bin").exists():
        raise StageError(
            "imgs2poses",
            f"distort_free/sparse/cameras.bin missing: {sparse_dir}",
        )

    # 检查 COLMAP 注册图数 —— 如果太少(< 5 张),训练会失败,提早给出有意义的报错
    n_reg = _count_registered_images(sparse_dir)
    images_dir = distort_free / "images"
    total = sum(1 for _ in images_dir.iterdir()) if images_dir.exists() else 0
    MIN_REGISTERED = 5
    if 0 <= n_reg < MIN_REGISTERED:
        raise StageError(
            "imgs2poses",
            f"COLMAP 只注册了 {n_reg} 张图(< {MIN_REGISTERED} 下限),重建质量不足以训练。"
            f"视频可能纹理弱 / 运动不足 / 帧间重叠过多。"
            f"原始 {total} 帧中只有 {n_reg} 张被稀疏重建成功。"
            f"建议:① 重新选择帧间差异更大的片段 "
            f"② 调整 --frame_interval 让抽帧间隔更稀疏",
        )

    container_workdir = to_container(jdir)
    container_scenedir = to_container(distort_free)
    # 必须用 {CONTAINER_PYTHON} 前缀 —— /home/Pretools/imgs2poses.py 没有 +x 也没 shebang,
    # 直接 `<script>.py --args` 会被 bash 报 "Permission denied"。
    # PYTHONPATH 走 extra_env 而不是 shell 内联,避免 docker exec 拼长字符串。
    cmd = (
        f"cd {container_workdir} && "
        f"{CONTAINER_PYTHON} {CONTAINER_PRETOOLS}/imgs2poses.py "
        f"--match_type exhaustive_matcher "
        f"--scenedir {container_scenedir}"
    )
    await docker_exec(
        cmd,
        workdir=container_workdir,
        log_path=log_path,
        extra_env={"PYTHONPATH": CONTAINER_PRETOOLS},
    )

    poses_bounds = distort_free / "poses_bounds.npy"
    if not poses_bounds.exists():
        raise StageError(
            "imgs2poses",
            f"poses_bounds.npy missing: {poses_bounds}",
        )
