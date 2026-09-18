"""从 train.py 日志提取最终 PSNR。

格式:`[ITER 30000] Evaluating test: L1 0.04489 PSNR 22.77941916539119`
"""
from __future__ import annotations
import re
from pathlib import Path

_PSNR_RE = re.compile(r"\[ITER (\d+)\] Evaluating test: L1 [\d.eE+-]+ PSNR ([\d.eE+-]+)")


def parse_final_psnr(log_path) -> tuple[int, float]:
    """返回 (final_iter, final_psnr)。日志缺失则返回 (0, 0.0)。"""
    p = Path(log_path)
    if not p.exists():
        return 0, 0.0
    last_iter = 0
    last_psnr = 0.0
    # 用 replace('\r', '\n') 处理 carriage-return-only progress 行(tee 在 TTY 下可能产生)
    text = p.read_text(errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        m = _PSNR_RE.search(line)
        if not m:
            continue
        it = int(m.group(1))
        psnr = float(m.group(2))
        last_iter = it
        last_psnr = psnr
    return last_iter, last_psnr
