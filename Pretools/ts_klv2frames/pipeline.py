"""ts_klv2frames pipeline:TS → 抽帧 + KLV GPS 匹配 → PNG(带 EXIF GPSInfo)。

流程:
  1. 解 TS 包 → 找 KLV PES 流(PID auto-detect private_stream_1)
  2. 解析 KLV MISB ST 0601 → (timestamp, lat, lon, alt) 列表
  3. 用 cv2 打开 TS → cv2.CAP_PROP_POS_MSEC 抽帧(每 f 帧取 1)
  4. 对每张抽出的帧,在 KLV 时间序列上二分找最近 GPS
  5. 写入 PNG(带 EXIF GPSInfo)
  6. 严格过滤:找不到 KLV 匹配的帧直接丢

用法:
  python -m ts_klv2frames.pipeline \\
      --ts /path/to/video.ts \\
      --out /path/to/images_dir \\
      --frame-interval 13 \\
      --match-window-ms 500

输出:
  /path/to/images_dir/
      000001.png
      000002.png
      ...
      klv.csv          (timestamp_s, lat, lon, alt_m, frame_idx)
"""
from __future__ import annotations

import argparse
import bisect
import csv
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from .ts_demux import demux_klv_payload
from .klv_misb0601 import parse_klv_stream, KlvRecord
from .exif_gps import write_png_with_exif, write_jpeg_with_gps_to_file


@dataclass
class FrameSample:
    frame_idx: int        # 视频帧 index(0-based)
    timestamp_ms: float   # 视频时间戳(ms)
    image: "cv2.Mat"      # BGR ndarray


def extract_klv_series(ts_path: Path, klv_pid: int | None = None) -> list[KlvRecord]:
    """读完整 TS 文件,提取所有 KLV records(按时间排序)。"""
    records: list[KlvRecord] = []
    for payload in demux_klv_payload(ts_path, pid=klv_pid):
        for r in parse_klv_stream(payload):
            if r.timestamp_s > 0 and (r.lat is not None or r.lon is not None):
                records.append(r)
    records.sort(key=lambda r: r.timestamp_s)
    return records


def find_nearest_klv(timestamps_s: list[float], t_video_s: float,
                     window_ms: float) -> KlvRecord | None:
    """二分找视频时间戳最近的 KLV(若超出 window_ms 则返回 None)。"""
    if not timestamps_s:
        return None
    i = bisect.bisect_left(timestamps_s, t_video_s)
    candidates: list[float] = []
    if i < len(timestamps_s):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    best_idx = min(candidates, key=lambda k: abs(timestamps_s[k] - t_video_s))
    delta_ms = abs(timestamps_s[best_idx] - t_video_s) * 1000.0
    if delta_ms > window_ms:
        return None
    return best_idx


def frame_index_to_klv_index(records: list[KlvRecord], frame_indices: list[int],
                             frame_timestamps_ms: list[float], window_ms: float) -> list[int | None]:
    """返回每张抽帧对应的 KLV 索引(无匹配则为 None)。"""
    if not records:
        return [None] * len(frame_indices)
    ts_s = [r.timestamp_s for r in records]
    out: list[int | None] = []
    for ft_ms in frame_timestamps_ms:
        ft_s = ft_ms / 1000.0
        idx = find_nearest_klv(ts_s, ft_s, window_ms)
        out.append(idx)
    return out


def _iso_utc_from_klv(timestamp_s: float) -> str:
    """Unix epoch seconds → 'YYYY-MM-DDTHH:MM:SSZ'."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def run(ts_path: Path, out_dir: Path, frame_interval: int = 13,
        window_ms: float = 500.0, max_frames: int | None = None,
        keep_intermediate: bool = False, klv_pid: int | None = None) -> dict:
    """主流程。

    Returns:
        {"saved": int, "dropped_no_klv": int, "klv_total": int, "out_dir": str}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir  # PNG 直接写到 out_dir(无中间文件)

    print(f"[1/4] extracting KLV from {ts_path} ...")
    records = extract_klv_series(ts_path, klv_pid=klv_pid)
    print(f"      {len(records)} KLV records")
    if not records:
        raise RuntimeError("no KLV records found in TS — check PID or stream")

    # 把 KLV Unix epoch 时间归到"视频起始相对秒",与 cv2.CAP_PROP_POS_MSEC 同一基准
    # cv2.CAP_PROP_POS_MSEC 从视频 t=0 起;KLV 是 Unix epoch
    # 对齐:KLV[t_rel] = KLV[t_unix] - KLV[t0_unix]
    ts_offset_s = records[0].timestamp_s
    for r in records:
        r.timestamp_s = r.timestamp_s - ts_offset_s
    print(f"      KLV time base shifted by {-ts_offset_s:.3f}s (Unix → relative)")

    # 写 csv(临时)
    csv_path = out_dir / "_klv_intermediate.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_s", "lat", "lon", "alt_m", "matched_video_ms"])
        for r in records:
            w.writerow([r.extras.get("frame_idx", ""), f"{r.timestamp_s:.6f}",
                        r.lat or "", r.lon or "", r.alt if r.alt is not None else "", ""])

    print(f"[2/4] opening video ...")
    cap = cv2.VideoCapture(str(ts_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {ts_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"      fps={fps} total_frames={total_frames}")

    # 抽帧(和 jobs/data2 一致:cv2.VideoCapture 顺序读 + 每 f 帧取 1)
    print(f"[3/4] extracting frames @ interval={frame_interval} ...")
    samples: list[FrameSample] = []
    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            samples.append(FrameSample(frame_idx=i, timestamp_ms=ts_ms, image=frame))
            saved += 1
            if max_frames and saved >= max_frames:
                break
        i += 1
    cap.release()
    print(f"      {len(samples)} sampled frames")

    # 匹配 KLV
    print(f"[4/4] matching frames to KLV (window={window_ms}ms) ...")
    ts_s = [r.timestamp_s for r in records]
    matched_indices: list[int | None] = []
    for s in samples:
        idx = find_nearest_klv(ts_s, s.timestamp_ms / 1000.0, window_ms)
        matched_indices.append(idx)

    # 写 JPG + EXIF APP1(只写有 KLV 匹配的)
    # 命名:6 位 zero-padded(.jpg 后缀),COLMAP 和所有 viewer 都能读 EXIF GPS
    saved = 0
    dropped = 0
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpd:
        tmp = Path(tmpd)
        for sample, klv_idx in zip(samples, matched_indices):
            if klv_idx is None:
                dropped += 1
                continue
            rec = records[klv_idx]
            if rec.lat is None or rec.lon is None:
                dropped += 1
                continue
            saved += 1
            jpg_name = f"{saved:06d}.jpg"
            jpg_path = work_dir / jpg_name
            tmp_jpg = tmp / jpg_name
            utc_iso = _iso_utc_from_klv(rec.timestamp_s + ts_offset_s)
            # 直接写带 EXIF 的 JPG 到 out_dir(cv2 编码 + 注入 APP1 一次性完成)
            write_jpeg_with_gps_to_file(
                jpg_path, sample.image,
                lat=rec.lat, lon=rec.lon,
                alt_m=rec.alt or 0.0,
                utc_iso=utc_iso,
                quality=95,
            )

    # 更新 CSV:写入匹配的视频时间戳
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_frame_idx", "klv_timestamp_s", "lat", "lon", "alt_m"])
        for s, idx in zip(samples, matched_indices):
            if idx is None:
                continue
            r = records[idx]
            w.writerow([s.frame_idx, f"{r.timestamp_s:.6f}",
                        r.lat or "", r.lon or "",
                        r.alt if r.alt is not None else ""])

    # 清理中间文件(csv / mp4)
    if not keep_intermediate:
        csv_path.unlink(missing_ok=True)

    return {
        "saved": saved,
        "dropped_no_klv": dropped,
        "klv_total": len(records),
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TS+KLV → 抽帧 + 写 GPS EXIF")
    p.add_argument("--ts", required=True, type=Path, help="输入 TS 文件")
    p.add_argument("--out", required=True, type=Path, help="输出 PNG 目录")
    p.add_argument("--frame-interval", type=int, default=13,
                   help="抽帧间隔(默认 13,和 jobs/data2 一致)")
    p.add_argument("--match-window-ms", type=float, default=500.0,
                   help="视频帧 ↔ KLV 时间窗口(ms),超出丢帧")
    p.add_argument("--max-frames", type=int, default=None,
                   help="最多抽多少帧(debug 用)")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="保留 _klv_intermediate.csv(默认删)")
    p.add_argument("--klv-pid", type=lambda s: int(s, 0), default=None,
                   help="KLV PID(hex/dec,默认 auto-detect)")
    args = p.parse_args(argv)

    if not args.ts.exists():
        print(f"ERROR: {args.ts} not found", file=sys.stderr)
        return 2

    try:
        result = run(
            args.ts, args.out,
            frame_interval=args.frame_interval,
            window_ms=args.match_window_ms,
            max_frames=args.max_frames,
            keep_intermediate=args.keep_intermediate,
            klv_pid=args.klv_pid,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print()
    print("=== done ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
