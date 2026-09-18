"""阶段 2.5:downsample —— 对 images/ 下所有图做下采样(就地覆盖)。

目的:
- 让大场景数据集(比如 4K 视频抽出来的 3840x2160)能在合理时间内跑完 COLMAP + train
- 在 frame_extract / normalize 之后、colmap_sfm 之前执行
- 因子 ∈ {1, 2, 4, 8},1 表示不下采样(stage 早返回)
- JPG 保留 EXIF(含 GPS);PNG 不带 EXIF 所以 N/A
- PIL LANCZOS 重采样
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image

from ..config import job_dir_host
from ..utils import StageError

ALLOWED_FACTORS = {1, 2, 4, 8}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


async def run(job_id: str, factor: int = 1) -> int:
    """就地覆盖下采样。返回处理的图数。factor=1 时早返回(不动)。"""
    if factor not in ALLOWED_FACTORS:
        raise StageError(
            "downsample",
            f"factor must be one of {sorted(ALLOWED_FACTORS)}, got {factor}",
        )

    jdir = job_dir_host(job_id)
    images_dir = jdir / "images"
    # 与 runner.py DEFAULT_STAGES 中的 log_name 对齐:stage_03_downsample.log
    # (factor 信息已在 jobs.downsample_factor 列里)
    log_path = jdir / "logs" / "stage_03_downsample.log"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise StageError(
            "downsample",
            f"images dir missing or empty: {images_dir}。"
            f"downsample 必须在 inbound/normalize/frame_extract 之后跑。",
        )

    if factor == 1:
        log_path.write_text("factor=1, skip downsampling\n")
        return 0

    # 收集所有要处理的图
    images = sorted(p for p in images_dir.iterdir() if _is_image(p))
    if not images:
        raise StageError("downsample", f"no images in {images_dir}")

    log_lines = [
        f"downsampling {len(images)} images by 1/{factor} (LANCZOS, in-place overwrite)",
        "",
    ]
    n_done = 0
    n_failed = 0
    for i, src in enumerate(images):
        try:
            with Image.open(src) as img:
                orig_size = img.size
                orig_mode = img.mode
                # EXIF 在 PIL 里挂 img.info['exif'],PNG 不带所以 None
                exif_bytes = img.info.get("exif", None)

                new_w = max(1, orig_size[0] // factor)
                new_h = max(1, orig_size[1] // factor)
                if (new_w, new_h) == orig_size:
                    # 已经够小,跳过
                    log_lines.append(
                        f"[{i+1}/{len(images)}] SKIP {src.name} "
                        f"(already {orig_size[0]}x{orig_size[1]})"
                    )
                    continue

                # RGBA / P 等模式转 RGB 再 resize,避免 LANCZOS 在带 alpha 通道时报错
                # 但 resize 后我们写回原扩展名,所以 PNG 保留 RGBA / RGB, JPG 强制 RGB
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                ext = src.suffix.lower()
                if ext in {".jpg", ".jpeg"}:
                    # JPG 不支持 alpha —— 转 RGB
                    if resized.mode != "RGB":
                        resized = resized.convert("RGB")
                    save_kwargs = {"quality": 95, "optimize": True}
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes  # 保留 EXIF(含 GPS)
                    resized.save(src, "JPEG", **save_kwargs)
                else:
                    # PNG:保留模式(RGB / RGBA / L ...)
                    save_kwargs = {"optimize": True}
                    resized.save(src, "PNG", **save_kwargs)

                log_lines.append(
                    f"[{i+1}/{len(images)}] {src.name}: "
                    f"{orig_size[0]}x{orig_size[1]} ({orig_mode}) -> "
                    f"{new_w}x{new_h} ({resized.mode})"
                )
                n_done += 1
        except Exception as e:
            log_lines.append(f"[{i+1}/{len(images)}] FAIL {src.name}: {e!r}")
            n_failed += 1

    log_lines.append("")
    log_lines.append(f"done: {n_done} resized, {n_failed} failed")
    log_path.write_text("\n".join(log_lines) + "\n")

    if n_done == 0:
        raise StageError(
            "downsample",
            f"downsample 0/{len(images)} images, all failed。看 log。",
        )

    return n_done
