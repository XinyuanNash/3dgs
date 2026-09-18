"""阶段 2:frame_extract —— 从视频提取帧。

仅 video 输入。
- 普通视频(.mp4/.mov/.avi/.mkv/.webm):调 Pretools/taketheframe.py
  CLI:  python taketheframe.py --video_path <input.bin> --output_dir <images/> -f <N>
  输出:000001.png …(PNG,无 EXIF GPS)

- TS + KLV(.ts):调 Pretools/ts_klv2frames(ISO 13818-1 TS + MISB ST 0601 KLV)
  CLI:  cd /home/Pretools && python -m ts_klv2frames --ts <bin> --out <dir> \
                                              --frame-interval <N> --match-window-ms 500
  输出:000001.jpg …(JPG with JPEG APP1 EXIF GPS,COLMAP/PIL/OpenCV 通用可读)

切换依据:inbound.manifest.is_ts(根据上传文件扩展名自动判定)
"""
from __future__ import annotations
import json

from ..config import job_dir_host, to_container, CONTAINER_PYTHON, CONTAINER_PRETOOLS
from ..utils import docker_exec, StageError


async def run(job_id: str, frame_interval: int = 13, is_ts: bool = False) -> int:
    jdir = job_dir_host(job_id)
    input_bin = jdir / "input.bin"
    images_dir = jdir / "images"
    log_path = jdir / "logs" / "stage_02_frame_extract.log"

    if not input_bin.exists():
        raise StageError("frame_extract", f"input.bin missing: {input_bin}")

    # is_ts 默认 False;若显式未传,从 manifest.json 读(inbound 写好的)
    if not is_ts:
        manifest_path = jdir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                is_ts = bool(manifest.get("is_ts", False))
            except (json.JSONDecodeError, OSError):
                pass

    images_dir.mkdir(exist_ok=True)

    container_input = to_container(input_bin)
    container_output = to_container(images_dir)
    container_workdir = to_container(jdir)

    if is_ts:
        # TS + KLV:MISB ST 0601 GPS 元数据被注入到 JPG EXIF APP1
        # 必须 cd /home/Pretools,这样 `python -m ts_klv2frames` 才能找到包
        #
        # --klv-pid 0x0101:DJI 行业机 TS 的 KLV PID 固定为 0x0101;
        # 默认 auto-detect 对 DJI 行业机会漏,所以这里硬编码。
        # 其他厂商如需改回 auto-detect,把 0x0101 改成 0 或删掉这一行。
        cmd = (
            f"cd {CONTAINER_PRETOOLS} && "
            f"{CONTAINER_PYTHON} -m ts_klv2frames "
            f"--ts {container_input} "
            f"--out {container_output} "
            f"--frame-interval {frame_interval} "
            f"--match-window-ms 500 "
            f"--klv-pid 0x0101"
        )
    else:
        # 普通视频:taketheframe 输出 PNG(无 EXIF GPS)
        cmd = (
            f"{CONTAINER_PYTHON} {CONTAINER_PRETOOLS}/taketheframe.py "
            f"--video_path {container_input} "
            f"--output_dir {container_output} "
            f"-f {frame_interval}"
        )

    await docker_exec(
        cmd,
        workdir=container_workdir,
        log_path=log_path,
    )

    # 校验输出:TS→JPG,其他→PNG
    if is_ts:
        frames = sorted(images_dir.glob("*.jpg"))
        if not frames:
            raise StageError(
                "frame_extract",
                f"no .jpg frames produced in {images_dir}。"
                f"看 stage_02_frame_extract.log 诊断。",
            )
    else:
        frames = sorted(images_dir.glob("*.png"))
        if not frames:
            raise StageError(
                "frame_extract",
                f"no .png frames produced in {images_dir}",
            )
    return len(frames)