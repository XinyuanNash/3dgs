#!/usr/bin/env python3
"""Sequential one-at-a-time 提交器 for datasets/2 视频。

策略:
  - 一次只能有一个 job 在跑(API GPU_LOCK 串行)
  - 跑完一个 → 提交下一个
  - 当前 job 未完成 → sleep N 分钟

状态文件: /tmp/monitor_datasets_2_state.json
  {
    "phase": "waiting_current" | "waiting_pending" | "running" | "done" | "error",
    "current_job_id": "<id>" | null,
    "current_job_name": "14-15" | null,
    "next_to_submit": "14-15",
    "submitted_order": ["14-15", "15-16", ...],   # 已 submit 的(按顺序)
    "completed": [{"name": "14-15", "job_id": "...", "status": "...", "psnr": 25.4}, ...],
    "last_check": "ISO8601"
  }
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path("/data4/huxinyuan/3dgs/api/db.sqlite")
STATE_PATH = Path("/tmp/monitor_datasets_2_state.json")
DATASETS_2 = Path("/data4/huxinyuan/3dgs/3dgs-work/datasets/2")
SUBMIT_SCRIPT = Path("/data4/huxinyuan/3dgs/api/scripts/submit_datasets_2.py")
API_BASE = "http://localhost:8005/api/v1"

INITIAL_CURRENT = "20260907170611-65-66-faec"  # 现在跑着的 65-66


def get_job(job_id: str) -> dict | None:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT id, name, status, error_stage, error_message, final_psnr FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    db.close()
    return dict(row) if row else None


def get_job_by_name(name: str) -> dict | None:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT id, name, status, final_psnr FROM jobs WHERE name = ? ORDER BY created_at DESC LIMIT 1",
        (name,),
    ).fetchone()
    db.close()
    return dict(row) if row else None


def _video_duration(name: str) -> float | None:
    """用 cv2 读视频时长(秒)。失败返回 None。"""
    try:
        import cv2
    except ImportError:
        return None
    video_dir = DATASETS_2 / name
    mp4s = list(video_dir.glob("*.MP4")) + list(video_dir.glob("*.mp4"))
    if not mp4s:
        return None
    try:
        cap = cv2.VideoCapture(str(mp4s[0]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return nframes / fps if fps > 0 else None
    except Exception:
        return None


def all_dataset_names() -> list[str]:
    """返回 datasets/2 下所有子目录名,自动跳过时长 > MAX_DURATION_SEC 的视频。"""
    SKIP_NAMES: list[str] = []  # 用户手动跳过

    def keep(name: str) -> bool:
        if name in SKIP_NAMES:
            return False
        dur = _video_duration(name)
        if dur is None:
            return True  # 时长未知 → 保留(让用户看到错误)
        return dur <= 180.0  # >3 min 跳过

    return sorted([d.name for d in DATASETS_2.iterdir() if d.is_dir() and keep(d.name)])


def submit_one(name: str) -> dict:
    """调 submit_datasets_2.py 但只提交一个 name。返回 {'id': ..., 'status': ...}。"""
    import http.client

    video_dir = DATASETS_2 / name
    mp4s = list(video_dir.glob("*.MP4")) + list(video_dir.glob("*.mp4"))
    if not mp4s:
        return {"error": f"no MP4 in {video_dir}"}
    video = mp4s[0]

    boundary = "----FormBoundary" + str(int(time.time() * 1000))
    body = []
    body.append(f"--{boundary}\r\n")
    body.append('Content-Disposition: form-data; name="name"\r\n\r\n')
    body.append(f"{name}\r\n")
    body.append(f"--{boundary}\r\n")
    body.append('Content-Disposition: form-data; name="iterations"\r\n\r\n')
    body.append("30000\r\n")
    body.append(f"--{boundary}\r\n")
    body.append(f'Content-Disposition: form-data; name="files"; filename="{video.name}"\r\n')
    body.append("Content-Type: video/mp4\r\n\r\n")
    with video.open("rb") as f:
        video_bytes = f.read()
    body_bytes = "".join(body).encode() + video_bytes + f"\r\n--{boundary}--\r\n".encode()

    import urllib.request
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
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    default = {
        "phase": "waiting_current",
        "current_job_id": INITIAL_CURRENT,
        "current_job_name": "65-66",
        "next_to_submit": all_dataset_names()[0] if all_dataset_names() else None,
        "submitted_order": [],
        "completed": [],
        "last_check": None,
    }
    # 第一次跑时把 default 立即写到磁盘,保证 cron tick 1 就有 state 可读
    STATE_PATH.write_text(json.dumps(default, indent=2, ensure_ascii=False))
    return default


def save_state(s: dict):
    s["last_check"] = datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(s, indent=2, ensure_ascii=False))


def main():
    state = load_state()
    now = datetime.now().strftime("%H:%M:%S")
    phase = state["phase"]
    print(f"[{now}] phase={phase}")
    # 每 tick 都更新 last_check + 写一份(防止 cron 间隔期间状态文件丢失)
    state["last_check"] = datetime.now().isoformat(timespec="seconds")

    if phase == "waiting_current":
        j = get_job(state["current_job_id"])
        if j is None:
            print(f"  current job {state['current_job_id']} not found → assume done")
            state["phase"] = "submit_next"
            save_state(state)
            return {"phase": state["phase"], "action": "submit_next"}
        print(f"  current {state['current_job_id']}: status={j['status']}")
        if j["status"] in ("succeeded", "failed", "cancelled"):
            state["current_job_status"] = j["status"]
            state["current_job_psnr"] = j["final_psnr"]
            # 推进:把 current job 的状态记录到 completed 列表
            if state["current_job_name"]:
                state["completed"].append({
                    "name": state["current_job_name"],
                    "job_id": state["current_job_id"],
                    "status": j["status"],
                    "psnr": j["final_psnr"],
                    "error_stage": j.get("error_stage"),
                })
            state["current_job_id"] = None
            state["current_job_name"] = None
            state["phase"] = "submit_next"
            save_state(state)
            return {"phase": state["phase"], "action": "submit_next"}
        # 还在跑 current → 保存 last_check
        save_state(state)
        return {"phase": phase, "action": "sleep", "seconds": 1800}

    if phase == "submit_next":
        # 提交 state.next_to_submit
        next_name = state.get("next_to_submit")
        if not next_name:
            state["phase"] = "done"
            save_state(state)
            return {"phase": "done", "action": "report"}
        print(f"  submitting: {next_name}")
        r = submit_one(next_name)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            state["phase"] = "error"
            state["last_error"] = r["error"]
            save_state(state)
            return {"phase": "error", "action": "report", "error": r["error"]}
        new_id = r["id"]
        print(f"  OK: {new_id} status={r['status']}")
        state["submitted_order"].append(next_name)
        state["current_job_id"] = new_id
        state["current_job_name"] = next_name
        state["phase"] = "running"
        # 计算下一个
        all_names = all_dataset_names()
        idx = all_names.index(next_name) if next_name in all_names else -1
        state["next_to_submit"] = all_names[idx + 1] if idx + 1 < len(all_names) else None
        save_state(state)
        return {"phase": state["phase"], "action": "sleep", "seconds": 1800}

    if phase == "running":
        # 监控当前 running 任务
        cur_id = state["current_job_id"]
        cur_name = state["current_job_name"]
        j = get_job(cur_id)
        if j is None:
            print(f"  current {cur_id} missing → assume deleted")
            state["current_job_id"] = None
            state["phase"] = "submit_next"
            save_state(state)
            return {"phase": state["phase"], "action": "submit_next"}
        print(f"  {cur_name} ({cur_id}): status={j['status']}")
        if j["status"] in ("succeeded", "failed", "cancelled"):
            state["completed"].append({
                "name": cur_name,
                "job_id": cur_id,
                "status": j["status"],
                "psnr": j["final_psnr"],
                "error_stage": j.get("error_stage"),
            })
            state["current_job_id"] = None
            state["current_job_name"] = None
            # 若失败 → 暂停(让用户决定是否跳过)
            if j["status"] != "succeeded":
                state["phase"] = "error"
                state["last_error"] = f"{cur_name} {j['status']}: {j.get('error_stage')} {j.get('error_message', '')[:200]}"
                save_state(state)
                return {"phase": "error", "action": "report", "error": state["last_error"]}
            # 成功 → 提交下一个
            if state.get("next_to_submit"):
                state["phase"] = "submit_next"
            else:
                state["phase"] = "done"
            save_state(state)
            return {"phase": state["phase"], "action": "submit_next" if state.get("next_to_submit") else "report"}
        # 还在跑 — 保存 last_check
        save_state(state)
        return {"phase": phase, "action": "sleep", "seconds": 1800}

    if phase in ("done", "error"):
        # 也写一次,记录 last_check
        save_state(state)
        return {"phase": phase, "action": "report", "completed": state["completed"]}

    # 兜底:未知 phase 也写一次
    save_state(state)
    return {"phase": phase, "action": "unknown"}


if __name__ == "__main__":
    r = main()
    print()
    print(json.dumps(r, ensure_ascii=False, indent=2))