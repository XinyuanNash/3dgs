"""工具函数:exec_command(容器内 subprocess)、UID 桥接、日志、磁盘预检。

设计:
- host 模式:exec_command 内部用 `docker exec 3dgs_xy bash -lc <cmd>`
- in-container 模式:exec_command 内部用 `bash -lc <cmd>` 直接在当前进程跑
  (uvicorn 已在 3dgs_xy 容器内,所有 stage 也在这同一容器)
- 自动检测:看 HOST_PREFIX 与 CONTAINER_PREFIX(由 config.py 决定)
"""
from __future__ import annotations
import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import (
    CONTAINER,
    DOCKER_PATH,
    HOST_PREFIX,
    CONTAINER_PREFIX,
    HOST_UID,
    HOST_GID,
    DISK_FREE_MIN_GB,
)


class StageError(Exception):
    """阶段执行失败。包含 stage 名和最近日志尾部。"""

    def __init__(self, stage: str, message: str, log_tail: str = ""):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.log_tail = log_tail


def to_container(host_path: str | Path) -> str:
    """宿主机路径 → 容器内路径。in-container 模式下直接返回原路径。"""
    s = str(host_path)
    if HOST_PREFIX == CONTAINER_PREFIX:
        return s
    if not s.startswith(HOST_PREFIX):
        raise ValueError(f"path {s!r} not under HOST_PREFIX {HOST_PREFIX!r}")
    return CONTAINER_PREFIX + s[len(HOST_PREFIX):]


def disk_free_gb(path: Path) -> float:
    """返回 path 所在文件系统的可用空间(GB)。"""
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def check_disk_space(path: Path, min_gb: float = DISK_FREE_MIN_GB) -> None:
    """磁盘预检。低于 min_gb 时抛 StageError("disk_check", ...)。"""
    free = disk_free_gb(path)
    if free < min_gb:
        raise StageError(
            "disk_check",
            f"磁盘空间不足:剩余 {free:.2f} GB,需要至少 {min_gb:.2f} GB",
        )


async def _watch_cancel(proc: asyncio.subprocess.Process, cancel_event: asyncio.Event) -> None:
    """后台任务:当 cancel_event 被设置时,杀死子进程。"""
    await cancel_event.wait()
    try:
        proc.terminate()
    except ProcessLookupError:
        pass


def _is_in_container_mode() -> bool:
    """in-container 模式 = HOST_PREFIX == CONTAINER_PREFIX(已 mount 到同一视角)。"""
    return HOST_PREFIX == CONTAINER_PREFIX


async def exec_command(
    inner_cmd: str,
    *,
    workdir: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
    log_path: Optional[str | Path] = None,
    cancel_event: Optional[asyncio.Event] = None,
    timeout: Optional[float] = None,
) -> int:
    """执行 inner_cmd。

    host 模式(默认):
        docker exec -i -u UID:GID -w workdir [-e K=V ...] 3dgs_xy bash -lc <inner_cmd>
        需要 /usr/bin/docker 存在。

    in-container 模式(HOST_PREFIX==CONTAINER_PREFIX):
        bash -lc <inner_cmd> 在当前容器进程内跑
        不需要 docker 二进制。

    返回:子进程退出码(0=成功)。
    抛出:StageError(若非零退出、超时、或启动失败)。
    """
    log_f = open(log_path, "wb") if log_path else subprocess.DEVNULL

    if _is_in_container_mode():
        # ---- in-container 模式:直接 bash -lc 跑 ----
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-lc", inner_cmd,
                cwd=workdir,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as e:
            if log_f is not subprocess.DEVNULL:
                log_f.close()
            raise StageError(
                "exec_command",
                f"无法启动 bash: {e}",
            ) from e
    else:
        # ---- host 模式:docker exec 调 3dgs_xy 容器 ----
        if not Path(DOCKER_PATH).exists():
            if log_f is not subprocess.DEVNULL:
                log_f.close()
            raise StageError(
                "exec_command",
                f"docker 二进制不存在: {DOCKER_PATH}。"
                f"检查 DOCKER_PATH 环境变量或安装 docker-ce。",
            )
        cmd = [
            DOCKER_PATH, "exec", "-i",
            "-u", f"{HOST_UID}:{HOST_GID}",
            "-w", workdir or "/home",
        ]
        for k, v in (extra_env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [CONTAINER, "bash", "-lc", inner_cmd]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=log_f, stderr=subprocess.STDOUT
            )
        except FileNotFoundError as e:
            if log_f is not subprocess.DEVNULL:
                log_f.close()
            raise StageError(
                "exec_command",
                f"无法启动 docker: {e}。"
                f"检查 {DOCKER_PATH} 是否可执行,用户是否在 docker 组。",
            ) from e

    watcher = None
    if cancel_event is not None:
        watcher = asyncio.create_task(_watch_cancel(proc, cancel_event))

    try:
        if timeout is not None:
            rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
        else:
            rc = await proc.wait()
    except asyncio.TimeoutError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        if log_f is not subprocess.DEVNULL:
            log_f.close()
        raise StageError("exec_command", f"timeout after {timeout}s", _tail(log_path))
    finally:
        if watcher is not None:
            watcher.cancel()
        if log_f is not subprocess.DEVNULL:
            log_f.close()

    if rc != 0:
        raise StageError("exec_command", f"exit code {rc}", _tail(log_path))
    return rc


# 向后兼容:保留旧名,新代码用 exec_command
async def docker_exec(*args, **kwargs):
    """向后兼容的别名 → 新代码请用 exec_command。"""
    return await exec_command(*args, **kwargs)


def _tail(log_path: Optional[str | Path], n: int = 60) -> str:
    """读日志最后 n 行,用于错误诊断。"""
    if not log_path:
        return ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError:
        return ""


def read_log_tail(log_path: str | Path, n: int = 5000) -> tuple[list[str], int]:
    """读取日志最后 n 行 + 总行数。"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        return [ln.rstrip("\n") for ln in lines[-n:]], total
    except FileNotFoundError:
        return [], 0