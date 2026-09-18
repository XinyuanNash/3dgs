"""配置:路径、容器名、GPU 索引、UID 桥接。

host `/data4/huxinyuan/3dgs` ↔ container `/home` (RW bind mount)
所有路径必须在 HOST_PREFIX 下;否则容器内无法访问。

运行模式(自动检测):
- host 模式:uvenv 在宿主机跑,docker exec 调 3dgs_xy 容器
  → HOST_PREFIX=/data4/huxinyuan/3dgs, CONTAINER_PREFIX=/home
- in-container 模式:uvenv 跑在 3dgs_xy 容器内,subprocess 也在同一容器
  → HOST_PREFIX=/home, CONTAINER_PREFIX=/home (path 不需要转换)
  → 检测:`/data4/...` 不存在 + `/home/api` 存在 → 自动切到容器视角
"""
from __future__ import annotations
import os
from pathlib import Path

# --- 容器 / 用户 ---
CONTAINER = os.environ.get("DOCKER_CONTAINER", "3dgs_xy")
HOST_UID = int(os.environ.get("HOST_UID", "1022"))
HOST_GID = int(os.environ.get("HOST_GID", "1023"))
GPU_INDEX = os.environ.get("GPU_INDEX", "1")

# 绝对路径(host 模式才需要;容器内不调 docker)
DOCKER_PATH = os.environ.get("DOCKER_PATH", "/usr/bin/docker")

# --- 路径前缀 ---
# 默认值:host 视角
_DEFAULT_HOST_PREFIX = "/data4/huxinyuan/3dgs"
_DEFAULT_CONTAINER_PREFIX = "/home"


def _detect_running_mode() -> tuple[str, str]:
    """自动检测运行模式,返回 (HOST_PREFIX, CONTAINER_PREFIX)。

    - 容器内(host 默认路径是空 stub,只有 db.sqlite,真实代码在 /home):
      → HOST_PREFIX=/home, CONTAINER_PREFIX=/home
    - host 上(/data4/.../api/pipeline3dgs/config.py 真实存在):
      → HOST_PREFIX=/data4/huxinyuan/3dgs, CONTAINER_PREFIX=/home
    - 环境变量覆盖:HOST_PREFIX / CONTAINER_PREFIX
    """
    env_hp = os.environ.get("HOST_PREFIX")
    env_cp = os.environ.get("CONTAINER_PREFIX")
    if env_hp:
        return env_hp, (env_cp or _DEFAULT_CONTAINER_PREFIX)

    # 用具体文件作为标记,避免被容器内 stub 目录(/data4/.../api 里只有 db.sqlite)骗到
    host_marker = Path(_DEFAULT_HOST_PREFIX) / "api" / "pipeline3dgs" / "config.py"
    home_marker = Path("/home/api/pipeline3dgs/config.py")
    if not host_marker.exists() and home_marker.exists():
        return "/home", "/home"

    return _DEFAULT_HOST_PREFIX, _DEFAULT_CONTAINER_PREFIX


HOST_PREFIX, CONTAINER_PREFIX = _detect_running_mode()

# --- 目录 ---
API_ROOT = Path(HOST_PREFIX) / "api"
JOBS_ROOT = Path(HOST_PREFIX) / "jobs"
DB_PATH = API_ROOT / "db.sqlite"

# 确保目录存在
API_ROOT.mkdir(parents=True, exist_ok=True)
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# --- 容器内路径 ---
# 训练代码在容器内 = /home/3dgs-work,容器外(本进程若是容器内)= 同一路径
CONTAINER_3DGS_WORK = "/home/3dgs-work"
CONTAINER_PRETOOLS = "/home/Pretools"
CONTAINER_PYTHON = "/home/miniconda3/envs/3dgs/bin/python"

# --- 磁盘预检 ---
DISK_FREE_MIN_GB = float(os.environ.get("DISK_FREE_MIN_GB", "5"))

# --- ship-path CLI 常量 ---
# 与 3dgs/3dgs-work/logs/ 命名约定一致 —— 时间戳前缀
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def job_dir_host(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def to_container(host_path: str | Path) -> str:
    """把宿主路径转换成容器内路径。必须在 HOST_PREFIX 下。

    in-container 模式(HOST_PREFIX == CONTAINER_PREFIX == /home):直接返回
    原路径字符串,因为本进程已经在容器内、看到的就是容器视角。
    """
    s = str(host_path)
    if HOST_PREFIX == CONTAINER_PREFIX:
        # in-container:path 已经在容器视角,无需转换
        return s
    if not s.startswith(HOST_PREFIX):
        raise ValueError(f"path {s!r} not under HOST_PREFIX {HOST_PREFIX!r}")
    return CONTAINER_PREFIX + s[len(HOST_PREFIX):]