"""阶段 0:inbound —— 文件已由 routes 层流式落盘到 <job_dir>/input/。

inbound.run 的职责现在只剩:
- 分类(image_folder | video | archive)
- 视频:把 input/<原始视频> 移到 <job_dir>/input.bin(frame_extract.py 期望此路径)
- 压缩包:解压到 <job_dir>/input/<stem>/
- 写 manifest.json + stage_00_inbound.log

为什么不在这里写盘:
- 521 MB 视频一次性 await f.read() 会 OOM
- routes.py 用 1 MB chunk 流式写到 input/ 后,inbound 直接复用磁盘文件
- 进一步节省:inbound 期间 file_paths 不再驻留内存
"""
from __future__ import annotations
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

from ..config import job_dir_host
from ..utils import StageError

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ARCHIVE_EXTS = {".zip", ".tar.gz", ".tgz", ".tar"}

# 防 zip bomb
MAX_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024   # 10 GB
MAX_EXTRACTED_FILES = 10_000

MANIFEST_NAME = "manifest.json"

# macOS metadata(应跳过)
SKIP_PATH_PREFIXES = ("__MACOSX/", ".DS_Store", "._")


def _is_archive(filename: str) -> bool:
    f = filename.lower()
    return any(f.endswith(ext) for ext in ARCHIVE_EXTS)


async def classify(filenames, hint: str = "auto") -> str:
    """根据文件名扩展名判断输入类型。
    入参可以是 dict[str, Path](取 keys)、list[str]、set[str] 之一。

    - "auto":根据上传内容自动判断
        - 单个压缩包 → archive (image_folder 的一种,会被解压后处理)
        - 单个视频文件 → video
        - 多个图片文件 → image_folder
    - 显式 hint:直接返回
    """
    if hint in ("image_folder", "video"):
        return hint
    if hint != "auto":
        raise StageError("inbound", f"unknown kind hint: {hint!r}")

    # 把各种入参归一为 list[str]
    if isinstance(filenames, dict):
        names = list(filenames.keys())
    else:
        names = list(filenames)

    if len(names) == 1:
        fname = names[0]
        if _is_archive(fname):
            return "image_folder"  # 归档会被解压再处理
        ext = Path(fname).suffix.lower()
        if ext in VIDEO_EXTS:
            return "video"
    return "image_folder"


def _should_skip(path: str) -> bool:
    """跳过 macOS 元数据、隐藏文件等。"""
    p = path.replace("\\", "/")
    if p.startswith(SKIP_PATH_PREFIXES):
        return True
    # 任意路径段以 ._ 开头(macOS resource fork)
    for seg in p.split("/"):
        if seg.startswith("._"):
            return True
    return False


def _extract_zip(archive_path: Path, dest_dir: Path) -> tuple[int, int]:
    """解压 zip。返回 (file_count, total_bytes)。"""
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        # 防 zip bomb 检查
        info_total = sum(i.file_size for i in zf.infolist())
        if info_total > MAX_UNCOMPRESSED_BYTES:
            raise StageError(
                "inbound",
                f"压缩包解压后总大小 {info_total / 1e9:.2f} GB 超过 {MAX_UNCOMPRESSED_BYTES / 1e9:.0f} GB 限制",
            )
        if len(zf.infolist()) > MAX_EXTRACTED_FILES:
            raise StageError(
                "inbound",
                f"压缩包文件数 {len(zf.infolist())} 超过 {MAX_EXTRACTED_FILES} 限制",
            )
        for info in zf.infolist():
            if _should_skip(info.filename):
                continue
            # 安全检查:阻止绝对路径 / path traversal
            target = (dest_dir / info.filename).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise StageError("inbound", f"非法路径(可能 zip slip): {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                # 流式复制(限制单文件大小)
                chunk_size = 1024 * 1024
                written_bytes = 0
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    written_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UNCOMPRESSED_BYTES:
                        raise StageError(
                            "inbound",
                            f"解压超出 {MAX_UNCOMPRESSED_BYTES / 1e9:.0f} GB 限制",
                        )
            file_count += 1
    return file_count, total_bytes


def _extract_tar(archive_path: Path, dest_dir: Path) -> tuple[int, int]:
    """解压 tar(.tar / .tar.gz / .tgz)。返回 (file_count, total_bytes)。"""
    file_count = 0
    total_bytes = 0
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        if len(members) > MAX_EXTRACTED_FILES:
            raise StageError(
                "inbound",
                f"压缩包成员数 {len(members)} 超过 {MAX_EXTRACTED_FILES} 限制",
            )
        info_total = sum(m.size for m in members if m.isfile())
        if info_total > MAX_UNCOMPRESSED_BYTES:
            raise StageError(
                "inbound",
                f"压缩包解压后总大小 {info_total / 1e9:.2f} GB 超过 {MAX_UNCOMPRESSED_BYTES / 1e9:.0f} GB 限制",
            )
        for m in members:
            if _should_skip(m.name):
                continue
            # path traversal 检查
            target = (dest_dir / m.name).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise StageError("inbound", f"非法路径(可能 tar slip): {m.name}")
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                chunk_size = 1024 * 1024
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UNCOMPRESSED_BYTES:
                        raise StageError(
                            "inbound",
                            f"解压超出 {MAX_UNCOMPRESSED_BYTES / 1e9:.0f} GB 限制",
                        )
            file_count += 1
    return file_count, total_bytes


def extract_archive(archive_path: Path, dest_dir: Path) -> dict:
    """解压压缩包到 dest_dir。返回统计信息。"""
    fname = archive_path.name.lower()
    if fname.endswith(".zip"):
        n, sz = _extract_zip(archive_path, dest_dir)
    elif fname.endswith((".tar.gz", ".tgz", ".tar")):
        n, sz = _extract_tar(archive_path, dest_dir)
    else:
        raise StageError("inbound", f"unsupported archive format: {archive_path.name}")

    return {"file_count": n, "total_bytes": sz}


def _collect_images(root: Path) -> list[Path]:
    """递归收集 root 下所有图片,按路径排序(保证可重现)。"""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


async def run(
    job_id: str,
    file_paths: dict[str, Path],
    kind_hint: str = "auto",
    source_filename: Optional[str] = None,
) -> str:
    """文件已由 routes 层流式写入 <job_dir>/input/<safe_name>。

    本阶段只做:
    - 分类
    - 视频:把 input/<video> 移动到 <job_dir>/input.bin (frame_extract 期望)
    - 压缩包:解压到 input/<stem>/,递归收集图片
    - 图片文件夹:文件已在 input/,跳过复制;记录 image_count
    - 写 manifest.json + stage_00_inbound.log

    返回:分类结果 ("video" | "image_folder")。
    """
    jdir = job_dir_host(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "logs").mkdir(exist_ok=True)

    kind = await classify(file_paths, kind_hint)
    log_path = jdir / "logs" / "stage_00_inbound.log"
    lines: list[str] = []

    if kind == "video":
        if len(file_paths) != 1:
            raise StageError("inbound", f"video upload must have exactly 1 file, got {len(file_paths)}")
        fname, src_path = next(iter(file_paths.items()))
        # frame_extract.py 硬编码 <job_dir>/input.bin;改名(同 fs 上 = rename,瞬间)
        dst = jdir / "input.bin"
        if dst.exists():
            dst.unlink()
        shutil.move(str(src_path), str(dst))
        size_mb = dst.stat().st_size / (1024 * 1024)
        is_ts = Path(fname).suffix.lower() == ".ts"
        lines.append(
            f"saved video: original_filename={fname!r} "
            f"size={size_mb:.1f} MB -> input.bin"
            f"{' (TS+KLV mode)' if is_ts else ''}"
        )
        first_filename = fname
        image_count = None
    else:
        # 图片文件夹(普通或压缩包)—— 文件已在 <job_dir>/input/
        in_dir = jdir / "input"
        image_count = 0

        # 判断是否为压缩包(单文件 + 压缩后缀)
        if len(file_paths) == 1 and _is_archive(next(iter(file_paths.keys()))):
            archive_name, archive_path = next(iter(file_paths.items()))
            lines.append(
                f"received archive: original_filename={archive_name!r} "
                f"path={archive_path} size={archive_path.stat().st_size / (1024 * 1024):.1f} MB"
            )

            # 解压到 input/<archive_stem>/
            stem = archive_name
            for ext in (".zip", ".tar.gz", ".tgz", ".tar"):
                if stem.lower().endswith(ext):
                    stem = stem[:-len(ext)]
                    break
            extract_dir = in_dir / stem
            extract_dir.mkdir(exist_ok=True)
            try:
                stats = extract_archive(archive_path, extract_dir)
            except (zipfile.BadZipFile, tarfile.TarError) as e:
                raise StageError("inbound", f"解压失败: {e!r}")
            lines.append(
                f"extracted: file_count={stats['file_count']} "
                f"total_bytes={stats['total_bytes']}"
            )

            # 递归收集所有图片
            images = _collect_images(extract_dir)
            image_count = len(images)
            lines.append(f"collected images: {image_count}")
            first_filename = archive_name
        else:
            # 普通图片文件夹:文件已在 input/,只需统计图片数
            for fname, p in file_paths.items():
                ext = Path(fname).suffix.lower()
                if ext not in IMAGE_EXTS:
                    lines.append(f"skipped non-image: {fname!r} path={p}")
                    continue
                if not p.exists():
                    lines.append(f"WARNING missing on disk: {fname!r} path={p}")
                    continue
                image_count += 1
            lines.append(f"received image_folder: image_count={image_count}")
            if image_count == 0:
                raise StageError("inbound", "image folder contains no supported images")
            first_filename = source_filename or (next(iter(file_paths.keys()), ""))

    manifest = {
        "job_id": job_id,
        "input_kind": kind,
        "source_filename": first_filename,
        "image_count": image_count,
        "is_ts": is_ts if kind == "video" else False,
    }
    (jdir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    lines.append(f"wrote manifest: {MANIFEST_NAME} (is_ts={is_ts if kind == 'video' else False})")

    log_path.write_text("\n".join(lines) + "\n")
    return kind