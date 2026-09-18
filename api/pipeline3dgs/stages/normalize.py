"""阶段 1:normalize —— 把图片文件夹重新编号为 images/images_<N>.<ext>。

目的:
- 避免中文文件名
- 统一命名(images_0, images_1, ...)便于下游识别
- 按字母序排序,保证可重现映射

仅作用于 image_folder 输入。video 输入由 frame_extract 直接输出到 images/。
"""
from __future__ import annotations
import shutil
from pathlib import Path

from ..config import job_dir_host
from ..utils import StageError

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


async def run(job_id: str) -> int:
    """把 <job>/input/<原始文件> 重新编号写入 <job>/images/images_<N>.<ext>。

    返回:图片数量。
    """
    jdir = job_dir_host(job_id)
    in_dir = jdir / "input"
    out_dir = jdir / "images"
    out_dir.mkdir(exist_ok=True)

    if not in_dir.exists():
        # video 输入不应该走到这里,但容忍一下
        raise StageError("normalize", f"input directory missing: {in_dir}")

    files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise StageError("normalize", "image folder is empty or has no supported images")

    # 清空旧文件(若 normalize 重跑)
    for old in out_dir.iterdir():
        if old.is_file():
            old.unlink()

    for i, src in enumerate(files):
        ext = src.suffix.lower()
        dst = out_dir / f"images_{i}{ext}"
        shutil.copy2(src, dst)

    # 写日志
    log = jdir / "logs" / "stage_01_normalize.log"
    log.write_text(
        f"normalized {len(files)} images into {out_dir}\n"
        f"  first: {files[0].name} -> images_0{files[0].suffix.lower()}\n"
        f"  last:  {files[-1].name} -> images_{len(files)-1}{files[-1].suffix.lower()}\n"
    )

    # 更新 manifest.image_count
    manifest_path = jdir / "manifest.json"
    if manifest_path.exists():
        import json
        m = json.loads(manifest_path.read_text())
        m["image_count"] = len(files)
        manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False))

    return len(files)