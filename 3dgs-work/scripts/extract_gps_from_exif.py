#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_gps_from_exif.py — 把 DJI/无人机 JPEG 里的 EXIF GPS 抽成 JSON
========================================================================

用法
----
    python extract_gps_from_exif.py <images_dir> [-o gps.json] [--recursive] [--pattern "*.JPG"]

把 <images_dir> 下的所有 JPEG 的 EXIF GPS 抽成如下 JSON:
    {
      "DJI_0001.JPG": {"lat": 30.123456, "lon": 114.654321, "alt": 123.45, "time": "2026:05:18 13:01:20"},
      ...
    }

然后这个 JSON 可以喂给 align_scale_to_gps.py 的 --gps-json 参数。

注意
----
- 直读 JPEG APP1/EXIF 字节,不依赖 PIL(避免 _getexif/getexif 在某些版本上返回空)
- 自动忽略没有 EXIF / 没有 GPS IFD 的图,统计 + 列出来
- 支持 --pattern (fnmatch) + --recursive
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 复用 align_scale_to_gps 的 EXIF parser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_scale_to_gps import parse_exif_gps  # noqa: E402


def collect_jpegs(root: str, pattern: str, recursive: bool) -> List[str]:
    out = []
    if recursive:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fnmatch.fnmatch(fn, pattern):
                    out.append(os.path.join(dp, fn))
    else:
        for fn in os.listdir(root):
            if fnmatch.fnmatch(fn, pattern):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def extract_time_string(jpeg_path: str) -> Optional[str]:
    """从 EXIF DateTimeOriginal (tag 0x9003) 抽出 'YYYY:MM:DD HH:MM:SS' 字符串。"""
    try:
        with open(jpeg_path, 'rb') as f:
            data = f.read(131072)
    except (OSError, IOError):
        return None
    if data[:2] != b'\xff\xd8':
        return None
    i = 2
    tiff = None
    while i < len(data) - 4:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD9, 0xDA):
            break
        if i + 4 > len(data):
            break
        seg_len = int.from_bytes(data[i + 2:i + 4], 'big')
        if i + 2 + seg_len > len(data):
            break
        seg = data[i + 4:i + 2 + seg_len]
        if marker == 0xE1 and seg.startswith(b'Exif\x00\x00'):
            tiff = i + 4 + 6
            break
        i += 2 + seg_len
    if tiff is None or tiff + 8 > len(data):
        return None
    bo = data[tiff:tiff + 2]
    endian = 'little' if bo == b'II' else 'big'
    if tiff + 10 > len(data):
        return None
    n_ifd0 = int.from_bytes(data[tiff + 8:tiff + 10], endian)
    for e in range(min(n_ifd0, 64)):
        ent = tiff + 10 + e * 12
        if ent + 12 > len(data):
            break
        tag = int.from_bytes(data[ent:ent + 2], endian)
        if tag == 0x8769:  # ExifIFD pointer
            exif_off = tiff + int.from_bytes(data[ent + 8:ent + 12], endian)
            if exif_off + 2 > len(data):
                continue
            n = int.from_bytes(data[exif_off:exif_off + 2], endian)
            for ee in range(min(n, 64)):
                eent = exif_off + 2 + ee * 12
                if eent + 12 > len(data):
                    break
                etag = int.from_bytes(data[eent:eent + 2], endian)
                if etag == 0x9003:  # DateTimeOriginal
                    dtype = int.from_bytes(data[eent + 2:eent + 4], endian)
                    count = int.from_bytes(data[eent + 4:eent + 8], endian)
                    if dtype == 2:  # ASCII
                        if count <= 4:
                            return data[eent + 8:eent + 8 + count].decode('ascii', errors='ignore').rstrip('\x00')
                        str_off = tiff + int.from_bytes(data[eent + 8:eent + 12], endian)
                        return data[str_off:str_off + count].decode('ascii', errors='ignore').rstrip('\x00')
            return None
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Extract EXIF GPS from a folder of JPEG images to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('images_dir', help='Directory containing JPEG images')
    ap.add_argument('-o', '--output', default=None,
                    help='Output JSON path (default: <images_dir>/gps_exif.json)')
    ap.add_argument('--pattern', default='*.JPG',
                    help='Filename glob (default: *.JPG; use "*.JPG" or "*.jpg" or "*.JPEG")')
    ap.add_argument('--recursive', action='store_true',
                    help='Walk subdirectories')
    ap.add_argument('--include-missing-time', action='store_true',
                    help='Include "time": null for images without DateTimeOriginal')
    args = ap.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    if not os.path.isdir(images_dir):
        print(f"ERROR: not a directory: {images_dir}", file=sys.stderr)
        sys.exit(1)

    jpgs = collect_jpegs(images_dir, args.pattern, args.recursive)
    if not jpgs:
        print(f"ERROR: no images match pattern '{args.pattern}' in {images_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(jpgs)} images (pattern='{args.pattern}', recursive={args.recursive}) ...")

    gps_records: Dict[str, dict] = {}
    no_gps: List[str] = []
    times: Dict[str, str] = {}

    for idx, path in enumerate(jpgs):
        rel = os.path.relpath(path, images_dir)
        # 只取文件名(若非 recursive 或与 images_dir 同层),保留子目录结构
        name = rel.replace(os.sep, '/')

        gps = parse_exif_gps(path)
        if gps is None:
            no_gps.append(name)
            continue

        rec = {"lat": gps[0], "lon": gps[1], "alt": gps[2]}
        t = extract_time_string(path)
        if t:
            rec["time"] = t
            times[name] = t
        elif args.include_missing_time:
            rec["time"] = None
        gps_records[name] = rec

        if (idx + 1) % 50 == 0 or idx + 1 == len(jpgs):
            print(f"  [{idx + 1}/{len(jpgs)}] GPS found in {len(gps_records)} / {len(no_gps) + len(gps_records)} images so far",
                  file=sys.stderr)

    output_path = args.output or os.path.join(images_dir, 'gps_exif.json')
    with open(output_path, 'w') as f:
        json.dump(gps_records, f, indent=2, sort_keys=True)
    print(f"\n[OK] Wrote {len(gps_records)} GPS records to {output_path}")
    if no_gps:
        print(f"      {len(no_gps)} images have NO EXIF GPS (skipped). Sample:")
        for n in no_gps[:5]:
            print(f"        - {n}")
        if len(no_gps) > 5:
            print(f"        ... and {len(no_gps) - 5} more")
    if times:
        ts = sorted(times.values())
        print(f"      Time range: {ts[0]}  →  {ts[-1]}")


if __name__ == '__main__':
    main()
