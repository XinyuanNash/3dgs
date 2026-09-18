"""FastAPI 路由 —— 所有 /api/v1 端点。"""
from __future__ import annotations
import asyncio
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from . import stages
from .config import job_dir_host, JOBS_ROOT
from .db import Db
from .models import (
    JobCreated,
    JobListResponse,
    JobStatus,
    JobStatusResponse,
    JobSummary,
    LogChunkResponse,
    StageRunInfo,
)
from .runner import AsyncPipelineRunner
from .stages.train import ply_path
from .utils import read_log_tail

router = APIRouter(prefix="/api/v1", tags=["jobs"])

# 运行时依赖 —— 通过 app.state 注入
_runner: Optional[AsyncPipelineRunner] = None
_db: Optional[Db] = None


def init(runner: AsyncPipelineRunner, db: Db) -> None:
    """由 app.py 在启动时调用,注入依赖。"""
    global _runner, _db
    _runner = runner
    _db = db


# job_id 文件名安全字符:小写字母 / 数字 / 下划线 / 连字符 / 点。 其它全替成 _
_JOB_NAME_RE = re.compile(r"[^a-z0-9_.-]+")
_JOB_NAME_MAX_LEN = 40  # 总长上限(含 timestamp 前缀后整体 < 60)


def _sanitize_job_name(name: Optional[str]) -> str:
    """清洗用户输入的 dataset name,使其可安全用于 job_id(URL/路径/字典序)。

    规则:
      - 转小写
      - 把连续非 [a-z0-9_.-] 字符替换为单个 _
      - 去掉前导 / 尾随的 _ - .
      - 截断到 _JOB_NAME_MAX_LEN 字符
      - 空结果(用户全填特殊字符)→ 返回空串(调用方回退到随机 hex)
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = _JOB_NAME_RE.sub("_", s)
    s = s.strip("_-.")
    if len(s) > _JOB_NAME_MAX_LEN:
        s = s[:_JOB_NAME_MAX_LEN].rstrip("_-.")
    return s


def _coerce_status(raw: str) -> JobStatus:
    """字符串 status → JobStatus enum;未知值回退到 FAILED 而不是抛 422。

    DB 里存的是枚举值字符串,但旧 job 行可能因为 enum 改名(例如 align_gps 已下线)
    或历史原因出现 JobStatus 不再识别的值。这种情况下返回 FAILED 让前端能渲染,
    而不是 500。
    """
    try:
        return JobStatus(raw)
    except ValueError:
        return JobStatus.FAILED


def _row_to_summary(row: dict) -> JobSummary:
    return JobSummary(
        id=row["id"],
        name=row.get("name"),
        status=_coerce_status(row["status"]),
        created_at=datetime.fromtimestamp(row["created_at"]),
        updated_at=datetime.fromtimestamp(row["updated_at"]),
        iterations=row["iterations"],
        final_psnr=row.get("final_psnr"),
        error=row.get("error_message"),
        input_kind=row["input_kind"],
        input_filename=row["input_filename"],
        image_count=row.get("image_count"),
        downsample_factor=row.get("downsample_factor", 1),
    )


def _row_to_status(row: dict, stage_runs: list[dict]) -> JobStatusResponse:
    summary = _row_to_summary(row)
    started_at = (
        datetime.fromtimestamp(row["started_at"])
        if row.get("started_at") else None
    )
    finished_at = (
        datetime.fromtimestamp(row["finished_at"])
        if row.get("finished_at") else None
    )
    duration = None
    if started_at and finished_at:
        duration = (finished_at - started_at).total_seconds()
    elif started_at:
        duration = (datetime.now() - started_at).total_seconds()

    artifacts = {}
    jdir = job_dir_host(row["id"])
    iterations = row["iterations"]
    ply = ply_path(row["id"], iterations)
    if ply.exists():
        artifacts["ply"] = str(ply.relative_to(JOBS_ROOT.parent))
    log_train = jdir / "logs" / "stage_10_train.log"
    if log_train.exists():
        artifacts["log_train"] = str(log_train.relative_to(JOBS_ROOT.parent))

    runs = [
        StageRunInfo(
            stage=r["stage"],
            started_at=datetime.fromtimestamp(r["started_at"]),
            finished_at=datetime.fromtimestamp(r["finished_at"]) if r.get("finished_at") else None,
            exit_code=r.get("exit_code"),
            log_path=r["log_path"],
        )
        for r in stage_runs
    ]
    return JobStatusResponse(
        **summary.model_dump(),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        artifacts=artifacts,
        stage_runs=runs,
    )


# ---------------- 路由 ----------------


@router.get("/health")
async def health():
    from .config import CONTAINER, GPU_INDEX
    from .utils import docker_exec
    try:
        rc = await asyncio.wait_for(
            docker_exec(
                f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {GPU_INDEX}",
                timeout=10,
            ),
            timeout=15,
        )
        container_ok = rc == 0
        gpu_free_mb = None
    except Exception:
        container_ok = False
        gpu_free_mb = None
    jobs_count = 0
    if _db is not None:
        _, jobs_count = await _db.list_jobs(limit=1)
    from .models import HealthResponse
    return HealthResponse(
        ok=container_ok,
        container_reachable=container_ok,
        gpu_free_mb=gpu_free_mb,
        jobs_count=jobs_count,
    )


@router.post("/jobs", status_code=201, response_model=JobCreated)
async def create_job(
    files: list[UploadFile] = File(..., description="图片文件夹(多个文件)或单个视频文件"),
    name: Optional[str] = Form(None),
    iterations: int = Form(30000),
    frame_interval: int = Form(13, description="视频截帧间隔(每隔 N 帧取 1 帧)。仅 video 输入生效,image_folder 输入忽略。范围 [1, 1000];默认 13 对应 ~2 帧/秒 @ 25fps"),
    downsample_factor: Optional[int] = Form(
        None,
        description="下采样倍数,作用于 frame_extract/normalize 之后、colmap 之前。"
                    "范围 {1,2,4,8};1 或 None = 不下采样(默认)。"
                    "例如 2 = 长宽各减半(像素数 1/4),"
                    "4K 视频抽帧后再 4× 下采样 ≈ 960x540,COLMAP+train 显著加速。",
    ),
    # ⚠️ align_gps Form 参数已移除(2026-09-11 用户决策):
    # 把 COLMAP 输出对齐到 GPS ENU 米制会改变训练 init scale → PSNR -4~-6 dB,
    # 且生产数据上无解。pipeline 中已无 align_gps stage;此功能下线。
    # 如确需 GPS 米制对齐,跑前手工调用 3dgs-work/scripts/align_scale_to_gps.py
):
    """提交新任务。返回 job_id 和初始状态。

    上传流式写入磁盘(1 MB chunks)—— 避免 521 MB 视频一次性
    await f.read() 撑爆进程内存。
    """
    if _db is None or _runner is None:
        raise HTTPException(503, "service not ready")

    if iterations < 100 or iterations > 100000:
        raise HTTPException(400, "iterations must be in [100, 100000]")

    if frame_interval < 1 or frame_interval > 1000:
        raise HTTPException(400, "frame_interval must be in [1, 1000]")

    # downsample_factor:None = 不下采样
    if downsample_factor is None:
        downsample_factor = 1
    if downsample_factor not in (1, 2, 4, 8):
        raise HTTPException(400, "downsample_factor must be 1, 2, 4 or 8 (or omit for 1)")

    # 生成 ULID-like ID:时间戳 + 数据集名 + 4hex 防撞
    # 例:20260904163242-my_scene-a1b2
    # 用户没填 name → 退化为旧格式(8hex)避免空段
    safe_name = _sanitize_job_name(name)
    ts = time.strftime("%Y%m%d%H%M%S")
    if safe_name:
        job_id = f"{ts}-{safe_name}-{uuid.uuid4().hex[:4]}"
    else:
        job_id = f"{ts}-{uuid.uuid4().hex[:8]}"

    # 创建任务目录
    jdir = job_dir_host(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "logs").mkdir(exist_ok=True)
    in_dir = jdir / "input"
    in_dir.mkdir(exist_ok=True)

    # 流式上传到磁盘 —— 关键:不再 await f.read() 加载完整 bytes
    # 1 MB chunks;若同名 collision,追加 _2 _3 ...
    file_paths: dict[str, Path] = {}
    total_bytes = 0
    for f in files:
        if not f.filename:
            continue
        # flatten: 用 basename 防目录注入,同时保持可读
        safe_name = Path(f.filename).name
        target = in_dir / safe_name
        counter = 1
        while target.exists():
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            target = in_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        # 1 MB chunks;同步写(UploadFile.read 是 await 的,本身就是 async)
        with target.open("wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                total_bytes += len(chunk)
        file_paths[f.filename] = target

    if not file_paths:
        # 清理空目录,免得留垃圾
        try:
            in_dir.rmdir()
            (jdir / "logs").rmdir()
            jdir.rmdir()
        except OSError:
            pass
        raise HTTPException(400, "no files uploaded")

    # 推断输入类型(只看文件名,不看 bytes)
    from .stages.inbound import classify as classify_fn
    filenames = list(file_paths.keys())
    kind = await classify_fn(filenames, "auto")
    first_filename = filenames[0]

    # 写 DB
    await _db.insert_job(
        id=job_id,
        name=name or first_filename,
        input_kind=kind,
        input_filename=first_filename,
        iterations=iterations,
        job_dir=str(jdir),
        downsample_factor=downsample_factor,
    )

    # 启动后台任务 —— 传磁盘路径,不再传 bytes
    await _runner.submit(
        job_id,
        {
            "input_kind": kind,
            "file_paths": file_paths,
            "kind_hint": "auto",
            "source_filename": first_filename,
            "iterations": iterations,
            "total_bytes": total_bytes,
            "frame_interval": frame_interval,
            "downsample_factor": downsample_factor,
            # align_gps 已下线(2026-09-11)
        },
    )

    return JobCreated(
        id=job_id,
        status=JobStatus.PENDING,
        downsample_factor=downsample_factor,
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    if _db is None:
        raise HTTPException(503, "service not ready")
    items, total = await _db.list_jobs(
        status=status.value if status else None, limit=limit, offset=offset
    )
    return JobListResponse(
        items=[_row_to_summary(r) for r in items],
        total=total,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    if _db is None:
        raise HTTPException(503, "service not ready")
    row = await _db.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"job {job_id} not found")
    runs = await _db.list_stage_runs(job_id)
    return _row_to_status(row, runs)


@router.get("/jobs/{job_id}/ply")
async def get_ply(job_id: str):
    if _db is None:
        raise HTTPException(503, "service not ready")
    row = await _db.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"job {job_id} not found")
    if row["status"] != JobStatus.SUCCEEDED.value:
        raise HTTPException(409, f"job not succeeded (status={row['status']})")
    ply = ply_path(job_id, row["iterations"])
    if not ply.exists():
        raise HTTPException(404, f"PLY not found: {ply}")
    return FileResponse(ply, filename=f"{job_id}.ply", media_type="application/octet-stream")


@router.get("/jobs/{job_id}/log", response_model=LogChunkResponse)
async def get_log(
    job_id: str,
    stage: str = Query("train", description="stage log 文件名(不含 .log)"),
    tail: int = Query(5000, ge=1, le=100000),
):
    if _db is None:
        raise HTTPException(503, "service not ready")
    row = await _db.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"job {job_id} not found")
    log_path = job_dir_host(job_id) / "logs" / f"{stage}.log"
    if not log_path.exists():
        raise HTTPException(404, f"log not found: {log_path}")
    lines, total = read_log_tail(log_path, tail)
    return LogChunkResponse(stage=stage, lines=lines, total_lines=total)


@router.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(job_id: str):
    if _db is None or _runner is None:
        raise HTTPException(503, "service not ready")
    row = await _db.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"job {job_id} not found")
    if row["status"] in (
        JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value
    ):
        raise HTTPException(409, f"job already in terminal state: {row['status']}")
    await _runner.cancel(job_id)
    row = await _db.get_job(job_id)
    runs = await _db.list_stage_runs(job_id)
    return _row_to_status(row, runs)


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, delete_files: bool = Query(True)):
    """删除任务。默认同时删除 per-job 文件夹。

    - delete_files=true:删除 <jobs_root>/<job_id>/ + DB 行
    - delete_files=false:仅删除 DB 行
    """
    if _db is None:
        raise HTTPException(503, "service not ready")
    row = await _db.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"job {job_id} not found")
    if _runner is not None and _runner.is_running(job_id):
        raise HTTPException(409, "cannot delete running job; cancel first")
    if delete_files:
        jdir = job_dir_host(job_id)
        if jdir.exists():
            shutil.rmtree(jdir)
    await _db.delete_job(job_id)
    return None
