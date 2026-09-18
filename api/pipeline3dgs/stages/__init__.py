"""stages 子包 —— 每个模块实现一个流水线阶段。

显式 import 所有子模块,确保通过 `stages.<name>.<func>` 访问时一定找得到。
(避免依赖 Python 自动子模块 import,后者在某些缓存场景下可能失效。)
"""
from . import (  # noqa: F401
    colmap_sfm,
    downsample,
    frame_extract,
    imgs2poses,
    inbound,
    metrics,
    normalize,
    train,
    undistort,
    # align_gps 已下线(2026-09-11 用户决策):
    # 把 COLMAP 输出对齐到 GPS ENU 米制会改变训练 init scale → PSNR -4~-6 dB,
    # 且生产数据上无解。下游 stage 不会调用它。
)

__all__ = [
    "colmap_sfm",
    "downsample",
    "frame_extract",
    "imgs2poses",
    "inbound",
    "metrics",
    "normalize",
    "train",
    "undistort",
]
