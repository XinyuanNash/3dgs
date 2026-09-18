#!/usr/bin/env python3
"""监控 running job + datasets/2 进度。

状态文件: /tmp/monitor_datasets_2_state.json
  {
    "phase": "waiting_for_current" | "monitoring_queue" | "done",
    "current_job": "<id>" | null,
    "submitted": ["14-15", "15-16", ...],
    "completed": [{"name": "14-15", "job_id": "...", "status": "succeeded", "psnr": 25.4}],
    "last_check": "2026-09-07T..."
  }

每个调用(check_datasets_2.py):
  1. 检查 running job (current_job 字段);若还跑,返回 sleep N
  2. 若 current_job 完成(succeeded/failed/cancelled):
     - 若 phase="waiting_for_current" 且 phase 没切:切换到 monitoring_queue,提交 13 视频
     - 若 phase="monitoring_queue":检查 submitted 任务进度;全完 → phase="done"
  3. 返回 phase + 完成率 + sleep N
"""
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path("/data4/huxinyuan/3dgs/api/db.sqlite")
STATE_PATH = Path("/tmp/monitor_datasets_2_state.json")
DATASETS_2 = Path("/data4/huxinyuan/3dgs/3dgs-work/datasets/2")
SUBMIT_SCRIPT = Path("/data4/huxinyuan/3dgs/api/scripts/submit_datasets_2.py")

CURRENT_JOB = "20260907170611-65-66-faec"  # 当前跑着的 65-66


def get_job_status(job_id: str) -> dict | None:
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


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "phase": "waiting_for_current",
        "current_job": CURRENT_JOB,
        "submitted": [],
        "completed": [],
        "last_check": None,
    }


def save_state(state: dict) -> None:
    state["last_check"] = datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def list_pending_videos() -> list[str]:
    """datasets/2 下所有子目录名(待提交的视频名)。"""
    return sorted([d.name for d in DATASETS_2.iterdir() if d.is_dir()])


def main():
    state = load_state()
    phase = state["phase"]
    now = datetime.now().strftime("%H:%M:%S")

    print(f"[{now}] phase={phase}")

    if phase == "waiting_for_current":
        # 等当前 65-66 任务完成
        j = get_job_status(CURRENT_JOB)
        if j is None:
            print(f"  current job {CURRENT_JOB} not found in DB (already cleaned up?)")
            # 假设已成功完成,推进
            state["phase"] = "submitting"
            save_state(state)
            print("  → phase=submitting (need to run submit script)")
            return {"phase": state["phase"], "action": "submit_now"}

        print(f"  current job status: {j['status']}  psnr={j['final_psnr']}")
        if j["status"] in ("succeeded", "failed", "cancelled"):
            state["current_job_status"] = j["status"]
            state["current_job_psnr"] = j["final_psnr"]
            state["phase"] = "submitting"
            save_state(state)
            print(f"  → current job {j['status']}; phase=submitting")
            return {"phase": state["phase"], "action": "submit_now"}
        else:
            print(f"  → still running, sleep 30 min")
            return {"phase": phase, "action": "sleep", "seconds": 1800}

    if phase == "submitting":
        # 提交所有 datasets/2 视频
        videos = list_pending_videos()
        print(f"  found {len(videos)} videos in {DATASETS_2}")
        print(f"  → run: python {SUBMIT_SCRIPT}")
        # 实际提交由 monitor 主调脚本负责(check_datasets_2 只汇报)
        state["phase"] = "monitoring_queue"
        state["expected_count"] = len(videos)
        save_state(state)
        return {"phase": state["phase"], "action": "run_submit_script", "videos": videos}

    if phase == "monitoring_queue":
        # 检查每个 submitted 任务状态
        expected = state.get("expected_count", 0)
        all_videos = list_pending_videos()
        completed = []
        in_progress = []
        pending = []
        for vname in all_videos:
            j = get_job_by_name(vname)
            if j is None:
                pending.append(vname)
            elif j["status"] == "succeeded":
                completed.append({"name": vname, "job_id": j["id"], "psnr": j["final_psnr"]})
            elif j["status"] in ("failed", "cancelled"):
                completed.append({"name": vname, "job_id": j["id"], "status": j["status"], "psnr": None})
            else:
                in_progress.append({"name": vname, "job_id": j["id"], "status": j["status"]})
                pending.append(vname)
        state["completed"] = completed
        state["in_progress"] = in_progress
        save_state(state)
        print(f"  completed: {len(completed)}/{expected}")
        print(f"  in_progress: {[p['name'] for p in in_progress]}")
        print(f"  pending: {pending}")
        # 检查 active job
        if in_progress:
            return {"phase": phase, "action": "sleep", "seconds": 1800, "completed": len(completed), "expected": expected}
        else:
            # 所有 submitted 都完成?但如果 pending 数 > 0,说明还没提交
            if pending:
                return {"phase": phase, "action": "submit_pending", "pending": pending}
            state["phase"] = "done"
            save_state(state)
            return {"phase": state["phase"], "action": "report", "completed": completed}

    if phase == "done":
        return {"phase": "done", "action": "report", "completed": state.get("completed", [])}

    return {"phase": phase, "action": "unknown"}


if __name__ == "__main__":
    r = main()
    print()
    print(json.dumps(r, ensure_ascii=False, indent=2))