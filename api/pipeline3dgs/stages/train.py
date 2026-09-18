"""阶段 8:train —— ship-path CLI(verbatim)。

只替换 -s、-m、日志路径。其他参数完全不动。

最终 PLY:<model>/point_cloud/iteration_<N>/point_cloud.ply
"""
from __future__ import annotations

from ..config import (
    CONTAINER_3DGS_WORK,
    CONTAINER_PYTHON,
    GPU_INDEX,
    job_dir_host,
    to_container,
)
from ..utils import docker_exec, StageError


# ship-path CLI 模板(来自 3dgs-work/README / 实际跑通的命令)
# 8 个空格缩进只是为了让 docker exec 的多行 shell 可读,实际会被传成单行
_TRAIN_CMD_TEMPLATE = (
    "cd {job_dir} && "
    "CUDA_VISIBLE_DEVICES={gpu_index} "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "{python} {threedgs}/train.py "
    "    -s {scenedir} "
    "    -m {modeldir} "
    "    --iterations {iterations} "
    "    --eval "
    "    --test_iterations 7000 18000 {iterations} "
    "    --save_iterations {iterations} "
    "    --checkpoint_iterations {iterations} "
    "    --densification_strategy igs_plus "
    "    --igs_plus_max_cap 2200000 "
    "    --igs_plus_scale_reg_weight 0.01 "
    "    --disable_viewer "
    "    2>&1 | tee {log_path}\n"
)


def ply_path(job_id: str, iterations: int):
    """最终 PLY 的宿主机路径。"""
    return (
        job_dir_host(job_id)
        / "model"
        / "point_cloud"
        / f"iteration_{iterations}"
        / "point_cloud.ply"
    )


async def run(job_id: str, iterations: int = 30000) -> None:
    jdir = job_dir_host(job_id)
    distort_free = jdir / "distort_free"
    model_dir = jdir / "model"
    log_path = jdir / "logs" / "stage_10_train.log"

    # 防御性检查:imgs2poses 必须已经写好 poses_bounds.npy
    if not (distort_free / "poses_bounds.npy").exists():
        raise StageError(
            "train",
            "distort_free/poses_bounds.npy missing: cannot start training",
        )

    container_job = to_container(jdir)
    container_scenedir = to_container(distort_free)
    container_model = to_container(model_dir)
    container_log = to_container(log_path)
    cmd = _TRAIN_CMD_TEMPLATE.format(
        job_dir=container_job,
        gpu_index=GPU_INDEX,
        python=CONTAINER_PYTHON,
        threedgs=CONTAINER_3DGS_WORK,
        scenedir=container_scenedir,
        modeldir=container_model,
        iterations=iterations,
        log_path=container_log,
    )
    await docker_exec(
        cmd,
        workdir=container_job,
        log_path=log_path,
        # 训练可能持续 30+ 分钟,默认 docker_exec 10 分钟 timeout 不够
        # 实际 timeout 由 docker_exec 内部通过 log 写入的"心跳"判断 —— 这里透传一个大值
    )

    final_ply = ply_path(job_id, iterations)
    if not final_ply.exists() or final_ply.stat().st_size < 50 * 1024 * 1024:
        raise StageError(
            "train",
            f"final PLY missing or too small (< 50 MB): {final_ply}",
        )
