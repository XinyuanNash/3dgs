#!/usr/bin/env python3
"""批量提交 datasets/2 下的 13 个视频到 FastAPI。

API: POST /api/v1/jobs (multipart form)
  - files: 单个 video 文件
  - name: dataset 名(用于 job_id 命名,如 "14-15")
  - iterations: 30000 (默认)

按序提交,API 的 GPU_LOCK 会让它们串行跑。
"""
import json
import sys
import time
from pathlib import Path

import urllib.request
import urllib.parse

DATASETS_DIR = Path("/data4/huxinyuan/3dgs/3dgs-work/datasets/2")
API_BASE = "http://localhost:8005/api/v1"
ITERATIONS = 30000


def submit_video(video_path: Path, name: str) -> dict:
    """POST 一个视频。返回 {'id': ..., 'status': ...} 或 raise。"""
    import http.client

    boundary = "----FormBoundary" + str(int(time.time() * 1000))
    body = []
    # name 字段
    body.append(f"--{boundary}\r\n")
    body.append('Content-Disposition: form-data; name="name"\r\n\r\n')
    body.append(f"{name}\r\n")
    # iterations 字段
    body.append(f"--{boundary}\r\n")
    body.append('Content-Disposition: form-data; name="iterations"\r\n\r\n')
    body.append(f"{ITERATIONS}\r\n")
    # files 字段(单个视频)
    body.append(f"--{boundary}\r\n")
    body.append(f'Content-Disposition: form-data; name="files"; filename="{video_path.name}"\r\n')
    body.append("Content-Type: video/mp4\r\n\r\n")
    with video_path.open("rb") as f:
        video_bytes = f.read()
    body_bytes = "".join(body).encode() + video_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{API_BASE}/jobs",
        data=body_bytes,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body_bytes)),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        return {"error": f"HTTP {e.code}: {err_body}"}


def main():
    videos = sorted(DATASETS_DIR.iterdir())
    videos = [d for d in videos if d.is_dir()]
    print(f"Found {len(videos)} video subdirs in {DATASETS_DIR}")

    results = []
    for d in videos:
        name = d.name
        # 找这个子目录里的 MP4
        mp4s = list(d.glob("*.MP4")) + list(d.glob("*.mp4"))
        if not mp4s:
            print(f"  [skip] {name}: no MP4 found")
            continue
        video = mp4s[0]
        size_mb = video.stat().st_size / 1024 / 1024
        print(f"  [submit] {name}/{video.name} ({size_mb:.1f} MB) ...")
        try:
            r = submit_video(video, name)
            if "error" in r:
                print(f"    ERROR: {r['error']}")
            else:
                print(f"    OK: job_id={r.get('id')}  status={r.get('status')}")
            results.append({"name": name, "video": str(video), "response": r})
        except Exception as e:
            print(f"    EXCEPTION: {e}")
            results.append({"name": name, "video": str(video), "exception": str(e)})

    out = Path("/tmp/datasets_2_submissions.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(results)} submissions to {out}")


if __name__ == "__main__":
    main()