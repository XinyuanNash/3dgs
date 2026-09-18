#!/usr/bin/env python3
"""upload.py —— 3DGS FastAPI 服务的一键上传脚本。

用法:
    # 上传整个图片文件夹
    python upload.py /path/to/my_images/

    # 上传单个视频
    python upload.py /path/to/video.mp4

    # 指定输出名称、迭代次数
    python upload.py ./my_images/ --name scene01 --iterations 30000

    # 不轮询(只提交)
    python upload.py ./my_images/ --no-poll

    # 仅下载 PLY(已知 job_id)
    python upload.py --fetch-only JOB_ID --out model.ply

行为:
1. 检测输入类型(图片文件夹 / 视频 / 单个压缩包)
2. 若是图片文件夹 → 自动 zip 后上传
3. 若是视频 → 直接上传
4. 若是 zip → 直接上传(避免重复压缩)
6. 轮询直到 SUCCEEDED / FAILED / CANCELLED
7. 自动下载最终 PLY 到 ./<job_id>.ply
"""
from __future__ import annotations
import argparse
import io
import mimetypes
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

import requests

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ARCHIVE_EXTS = {".zip", ".tar.gz", ".tgz", ".tar"}
SKIP_NAMES = {".DS_Store", "__MACOSX"}


def detect_kind(path: Path) -> str:
    """判断输入类型: video | image_folder | archive | unknown。"""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        suffix = "".join(path.suffixes).lower()
        if any(suffix.endswith(ext) for ext in ARCHIVE_EXTS):
            return "archive"
        if path.suffix.lower() in VIDEO_EXTS:
            return "video"
        if path.suffix.lower() in IMAGE_EXTS:
            return "image_folder"  # 单张图也算"文件夹"
        return "unknown"
    # 目录:扫一遍
    has_image = False
    has_video = False
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            has_image = True
        elif ext in VIDEO_EXTS:
            has_video = True
    if has_video and not has_image:
        return "video"  # 含视频,需要走 frame_extract
    if has_image:
        return "image_folder"
    return "unknown"


def zip_folder_to_memory(folder: Path) -> tuple[bytes, str]:
    """把图片文件夹 zip 后返回 (bytes, suggested_filename)。

    - 跳过 __MACOSX/、.DS_Store、._*
    - 保留相对路径(子目录嵌套结构)
    - 中文文件名保留(UTF-8 flag)
    """
    buf = io.BytesIO()
    n_files = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(folder.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(folder)
            rel_str = str(rel).replace("\\", "/")
            # skip macOS metadata
            if any(seg in SKIP_NAMES for seg in rel.parts) or rel.name.startswith("._"):
                continue
            # write with UTF-8 filename flag
            zi = zipfile.ZipInfo(rel_str)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.flag_bits |= 0x800  # UTF-8
            with p.open("rb") as src:
                zf.writestr(zi, src.read(), compresslevel=6)
            n_files += 1
    if n_files == 0:
        raise ValueError(f"folder {folder} contains no supported images/videos")
    suggested_name = f"{folder.name or 'upload'}.zip"
    return buf.getvalue(), suggested_name


def upload(
    api_base: str,
    *,
    files: Iterable[tuple[str, bytes, str]] | None = None,
    file_path: Path | None = None,
    name: str | None = None,
    iterations: int = 30000,
) -> dict:
    """POST /api/v1/jobs。files: (filename, bytes, content_type)。"""
    url = api_base.rstrip("/") + "/jobs"
    data = {"name": name or "", "iterations": str(iterations)}
    file_payload = []
    if files:
        for fname, content, ctype in files:
            file_payload.append(("files", (fname, content, ctype)))
    elif file_path is not None:
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        file_payload.append(("files", (file_path.name, file_path.open("rb"), ctype)))
    else:
        raise ValueError("either files or file_path required")
    try:
        r = requests.post(url, data=data, files=file_payload, timeout=600)
    finally:
        if file_path is not None:
            for _, fobj, _ in file_payload:
                if hasattr(fobj, "close"):
                    fobj.close()
    if r.status_code != 201:
        raise RuntimeError(f"upload failed: HTTP {r.status_code} {r.text}")
    return r.json()


def poll_status(api_base: str, job_id: str, *, interval: int = 30, max_wait_s: int = 3600) -> dict:
    """轮询直到 SUCCEEDED / FAILED / CANCELLED。返回最终 status_response。"""
    url = f"{api_base.rstrip('/')}/jobs/{job_id}"
    deadline = time.time() + max_wait_s
    last_status = None
    while time.time() < deadline:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        info = r.json()
        status = info["status"]
        if status != last_status:
            elapsed = info.get("duration_seconds")
            elapsed_s = f"{elapsed:.0f}s" if elapsed else "?"
            print(f"  [{time.strftime('%H:%M:%S')}] status={status} elapsed={elapsed_s}", flush=True)
            last_status = status
        if status in ("succeeded", "failed", "cancelled"):
            return info
        time.sleep(interval)
    raise TimeoutError(f"job {job_id} did not finish within {max_wait_s}s")


def download_ply(api_base: str, job_id: str, out: Path) -> Path:
    url = f"{api_base.rstrip('/')}/jobs/{job_id}/ply"
    r = requests.get(url, stream=True, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"download failed: HTTP {r.status_code} {r.text[:200]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="3DGS Pipeline 一键上传",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="图片文件夹或视频文件路径(若省略,仅在 --fetch-only 模式下需要)",
    )
    parser.add_argument("--api", default=os.environ.get("API", "http://localhost:8000/api/v1"),
                        help="API base URL (default: http://localhost:8000/api/v1)")
    parser.add_argument("--name", help="任务名称(默认用文件夹名)")
    parser.add_argument("--iterations", type=int, default=30000, help="训练迭代次数 (default: 30000)")
    parser.add_argument("--no-poll", action="store_true", help="提交后不轮询")
    parser.add_argument("--max-wait", type=int, default=3600, help="最长等待秒数 (default: 3600)")
    parser.add_argument("--out", help="PLY 输出路径(默认: ./<job_id>.ply)")
    parser.add_argument("--fetch-only", metavar="JOB_ID", help="仅下载已有 job 的 PLY")
    args = parser.parse_args(argv)

    if args.fetch_only:
        out = Path(args.out) if args.out else Path(f"{args.fetch_only}.ply")
        download_ply(args.api, args.fetch_only, out)
        print(f"[OK] PLY saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0

    if not args.input:
        parser.error("input is required (or use --fetch-only)")

    in_path = Path(args.input).resolve()
    kind = detect_kind(in_path)
    print(f"[detect] {in_path} -> {kind}")

    name = args.name or in_path.stem

    if kind == "archive":
        print(f"[upload] uploading archive: {in_path.name}")
        job = upload(args.api, file_path=in_path, name=name, iterations=args.iterations)
    elif kind == "video":
        print(f"[upload] uploading video: {in_path.name}")
        job = upload(args.api, file_path=in_path, name=name, iterations=args.iterations)
    elif kind == "image_folder":
        # 整文件夹 zip 后上传
        print(f"[zip] packing folder {in_path} ...")
        zip_bytes, zip_name = zip_folder_to_memory(in_path)
        print(f"[zip] packed {len(zip_bytes)/1e6:.2f} MB")
        print(f"[upload] uploading {zip_name}")
        job = upload(args.api, files=[(zip_name, zip_bytes, "application/zip")],
                     name=name, iterations=args.iterations)
    else:
        print(f"[FAIL] cannot determine input kind: {in_path}", file=sys.stderr)
        return 1

    job_id = job["id"]
    print(f"[OK] job created: id={job_id} status={job['status']}")

    if args.no_poll:
        print(f"[info] not polling; check status with: curl {args.api}/jobs/{job_id}")
        return 0

    info = poll_status(args.api, job_id, max_wait_s=args.max_wait)
    final = info["status"]
    if final != "succeeded":
        print(f"[FAIL] job ended with status={final}")
        if info.get("error"):
            print(f"  error_stage: {info.get('error_stage', '?')}")
            err = info["error"]
            tail = err if len(err) <= 800 else err[-800:] + "...(truncated)"
            print(f"  error_message tail:\n{tail}")
        return 1

    psnr = info.get("final_psnr")
    print(f"[OK] SUCCEEDED  PSNR={psnr}" if psnr else "[OK] SUCCEEDED")
    out = Path(args.out) if args.out else Path(f"{job_id}.ply")
    print(f"[download] PLY -> {out}")
    download_ply(args.api, job_id, out)
    print(f"[OK] PLY saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())