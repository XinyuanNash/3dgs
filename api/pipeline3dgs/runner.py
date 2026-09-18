"""AsyncPipelineRunner —— 编排所有流水线阶段。

设计:
- 每个任务一个 asyncio.Task,生命周期跨请求/响应
- 每个任务一个 asyncio.Event 用于取消
- GPU_LOCK (Semaphore(1)) 串行化所有 GPU 阶段
- 阶段失败 → 标记 FAILED,文件保留,日志保留
- 服务重启时 reset_inflight_to_failed() 清理悬挂任务
"""
from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Optional

from . import stages
from .config import JOBS_ROOT
from .db import Db
from .models import JobStatus as JS
from .utils import StageError, check_disk_space

logger = logging.getLogger("pipeline3dgs.runner")

GPU_LOCK = asyncio.Semaphore(1)


class StageSpec:
    """定义一个流水线阶段。"""

    def __init__(self, status: str, coro_fn, log_name: str):
        self.status = status
        self.coro_fn = coro_fn          # async (job_id, params) -> any
        self.log_name = log_name        # logs/stage_NN_<log_name>.log

    async def run(self, job_id: str, params: dict, db: Db) -> int:
        """调用 coro_fn,写 stage_runs 行,返回 exit_code (0=OK)。"""
        from .db import Db as _Db  # 避免循环
        run_id = await db.insert_stage_run(job_id, self.status, self.log_name)
        try:
            await self.coro_fn(job_id, params)
            await db.finish_stage_run(run_id, 0)
            return 0
        except StageError as e:
            await db.finish_stage_run(run_id, 1)
            raise
        except Exception as e:
            await db.finish_stage_run(run_id, 1)
            raise StageError(self.status, f"unexpected: {e!r}") from e


# 默认流水线顺序(图片文件夹)
DEFAULT_STAGES: list[StageSpec] = [
    StageSpec(JS.INBOUND.value, lambda jid, p: stages.inbound.run(
        jid, p["file_paths"], p.get("kind_hint", "auto"), p.get("source_filename")
    ), "stage_00_inbound.log"),
    StageSpec(JS.NORMALIZING.value, lambda jid, p: stages.normalize.run(jid),
              "stage_01_normalize.log"),
    StageSpec(JS.FRAME_EXTRACTING.value,
              lambda jid, p: stages.frame_extract.run(
                  jid, p.get("frame_interval", 13), p.get("is_ts", False),
              ),
              "stage_02_frame_extract.log"),
    # 阶段 2.5:downsample —— 可选,在 colmap 前对 images/ 下采样
    # factor=1 时 stage 早返回(不动),所以默认零开销
    StageSpec(JS.DOWNSAMPLING.value,
              lambda jid, p: stages.downsample.run(
                  jid, p.get("downsample_factor", 1),
              ),
              "stage_03_downsample.log"),
    StageSpec(JS.COLMAP_FEAT.value,
              lambda jid, p: stages.colmap_sfm.feature_extractor(jid),
              "stage_04_colmap_feat.log"),
    StageSpec(JS.COLMAP_MATCH.value,
              lambda jid, p: stages.colmap_sfm.exhaustive_matcher(jid),
              "stage_05_colmap_match.log"),
    StageSpec(JS.COLMAP_MAP.value,
              lambda jid, p: stages.colmap_sfm.mapper(jid),
              "stage_06_colmap_map.log"),
    # 阶段 7 align_gps 已被彻底移除(2026-09-11 用户决策):
    # 把 COLMAP 输出对齐到 GPS ENU 米制会改变训练 init scale → PSNR -4~-6 dB,
    # 且生产数据上无解。如需 GPS 米制对齐,跑前手工调用
    # /data4/huxinyuan/3dgs/3dgs-work/scripts/align_scale_to_gps.py
    StageSpec(JS.UNDISTORT.value,
              lambda jid, p: stages.undistort.run(jid),
              "stage_08_undistort.log"),
    StageSpec(JS.IMGS2POSES.value,
              lambda jid, p: stages.imgs2poses.run(jid),
              "stage_09_imgs2poses.log"),
    StageSpec(JS.TRAINING.value,
              lambda jid, p: stages.train.run(jid, p.get("iterations", 30000)),
              "stage_10_train.log"),
]


def build_pipeline(input_kind: str) -> list[StageSpec]:
    """根据输入类型构造流水线。

    - image_folder:跳过 frame_extract(直接是图片,无需抽帧)
    - video:跳过 normalize(frame_extract 输出 images/,无需重新编号)
    """
    if input_kind == "image_folder":
        return [s for s in DEFAULT_STAGES if s.status != JS.FRAME_EXTRACTING.value]
    if input_kind == "video":
        return [s for s in DEFAULT_STAGES if s.status != JS.NORMALIZING.value]
    return DEFAULT_STAGES


class AsyncPipelineRunner:
    def __init__(self, db: Db):
        self.db = db
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancels: dict[str, asyncio.Event] = {}

    async def submit(self, job_id: str, params: dict) -> None:
        """提交任务。返回时任务已在后台运行。"""
        if job_id in self._tasks and not self._tasks[job_id].done():
            raise RuntimeError(f"job {job_id} already running")
        self._cancels[job_id] = asyncio.Event()
        task = asyncio.create_task(self._run(job_id, params))
        self._tasks[job_id] = task

    async def cancel(self, job_id: str) -> bool:
        """请求取消。返回是否成功设置 cancel_event。"""
        ev = self._cancels.get(job_id)
        if ev is None:
            return False
        ev.set()
        await self.db.request_cancel(job_id)
        return True

    def is_running(self, job_id: str) -> bool:
        t = self._tasks.get(job_id)
        return t is not None and not t.done()

    async def _run(self, job_id: str, params: dict) -> None:
        await self.db.set_started(job_id)
        pipeline = build_pipeline(params["input_kind"])
        try:
            # 磁盘预检(在 GPU 阶段前)
            check_disk_space(JOBS_ROOT)

            for spec in pipeline:
                # inbound / normalize / frame_extract / downsample 在 GPU_LOCK 外
                # —— 这些都是 CPU / 容器内 subprocess,不需要串行化
                if spec.status in (
                    JS.FRAME_EXTRACTING.value,
                    JS.NORMALIZING.value,
                    JS.INBOUND.value,
                    JS.DOWNSAMPLING.value,
                ):
                    await self.db.set_status(job_id, spec.status)
                    await spec.run(job_id, params, self.db)
                else:
                    async with GPU_LOCK:
                        await self.db.set_status(job_id, spec.status)
                        await spec.run(job_id, params, self.db)

            # 训练成功 → 提取 PSNR
            from .stages.metrics import parse_final_psnr
            from .stages.train import ply_path
            iterations = params.get("iterations", 30000)
            train_log = Path(JOBS_ROOT) / job_id / "logs" / "stage_10_train.log"
            it, psnr = parse_final_psnr(train_log)
            if psnr > 0:
                await self.db.set_final_psnr(job_id, psnr, it)

            # 验证 PLY
            ply = ply_path(job_id, iterations)
            if not ply.exists():
                raise StageError("train", f"final PLY missing: {ply}")

            await self.db.set_status(job_id, JS.SUCCEEDED.value)
        except StageError as e:
            logger.exception("job %s failed at stage %s", job_id, e.stage)
            await self.db.set_status(job_id, JS.FAILED.value)
            await self.db.set_error(job_id, e.stage, e.log_tail or str(e))
        except asyncio.CancelledError:
            await self.db.set_status(job_id, JS.CANCELLED.value)
            raise
        except Exception as e:
            logger.exception("job %s unexpected error", job_id)
            await self.db.set_status(job_id, JS.FAILED.value)
            await self.db.set_error(job_id, "runner", repr(e))
        finally:
            await self.db.set_finished(job_id)
            self._tasks.pop(job_id, None)
            self._cancels.pop(job_id, None)
