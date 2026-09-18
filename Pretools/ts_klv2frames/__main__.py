"""ts_klv2frames — 顶级 CLI 入口(放在 Pretools/ 下,跟 taketheframe.py 同级)。

实际逻辑在 pipeline.py。本文件只做 argparse + 转调。
"""
from __future__ import annotations

import sys

from .pipeline import main


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
