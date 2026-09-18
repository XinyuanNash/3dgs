"""SQLite 异步封装 (aiosqlite) + schema 初始化。"""
from __future__ import annotations
import time
from typing import Optional

import aiosqlite

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    status          TEXT NOT NULL,
    error_stage     TEXT,
    error_message   TEXT,
    input_kind      TEXT NOT NULL,
    input_filename  TEXT NOT NULL,
    image_count     INTEGER,
    iterations      INTEGER NOT NULL DEFAULT 30000,
    downsample_factor INTEGER NOT NULL DEFAULT 1,
    final_psnr      REAL,
    final_psnr_iter INTEGER,
    job_dir         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    started_at      REAL,
    finished_at     REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS stage_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    exit_code   INTEGER,
    log_path    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stage_runs_job ON stage_runs(job_id);
"""


class Db:
    """异步 SQLite 封装。所有调用必须用 await;单连接复用。"""

    def __init__(self, path: Optional[str] = None):
        self.path = str(path or DB_PATH)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        await self._migrate()

    async def _migrate(self) -> None:
        """对 2026-09-11 之前的 DB 补齐 downsample_factor 列
        (CREATE TABLE IF NOT EXISTS 不会自动加列)。
        """
        async with self.conn.execute("PRAGMA table_info(jobs)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        if "downsample_factor" not in cols:
            # DEFAULT 1 保证旧行读到 1
            await self.conn.execute(
                "ALTER TABLE jobs ADD COLUMN downsample_factor INTEGER NOT NULL DEFAULT 1"
            )
            await self.conn.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Db not connected; call connect() first")
        return self._db

    # ---------------- jobs CRUD ----------------

    async def insert_job(
        self,
        *,
        id: str,
        name: Optional[str],
        input_kind: str,
        input_filename: str,
        iterations: int,
        job_dir: str,
        downsample_factor: int = 1,
    ) -> None:
        now = time.time()
        await self.conn.execute(
            """
            INSERT INTO jobs (id, name, status, input_kind, input_filename,
                              iterations, downsample_factor, job_dir,
                              created_at, updated_at)
            VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
            """,
            (id, name, input_kind, input_filename, iterations,
             downsample_factor, job_dir, now, now),
        )
        await self.conn.commit()

    async def set_status(self, job_id: str, status: str) -> None:
        now = time.time()
        await self.conn.execute(
            "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
            (status, now, job_id),
        )
        await self.conn.commit()

    async def set_started(self, job_id: str) -> None:
        now = time.time()
        await self.conn.execute(
            "UPDATE jobs SET started_at=COALESCE(started_at, ?), updated_at=? WHERE id=?",
            (now, now, job_id),
        )
        await self.conn.commit()

    async def set_finished(self, job_id: str) -> None:
        now = time.time()
        await self.conn.execute(
            "UPDATE jobs SET finished_at=?, updated_at=? WHERE id=?",
            (now, now, job_id),
        )
        await self.conn.commit()

    async def set_error(self, job_id: str, stage: str, message: str) -> None:
        now = time.time()
        await self.conn.execute(
            "UPDATE jobs SET error_stage=?, error_message=?, updated_at=? WHERE id=?",
            (stage, message, now, job_id),
        )
        await self.conn.commit()

    async def set_image_count(self, job_id: str, n: int) -> None:
        await self.conn.execute(
            "UPDATE jobs SET image_count=? WHERE id=?",
            (n, job_id),
        )
        await self.conn.commit()

    async def set_final_psnr(self, job_id: str, psnr: float, it: int) -> None:
        now = time.time()
        await self.conn.execute(
            "UPDATE jobs SET final_psnr=?, final_psnr_iter=?, updated_at=? WHERE id=?",
            (psnr, it, now, job_id),
        )
        await self.conn.commit()

    async def request_cancel(self, job_id: str) -> None:
        now = time.time()
        await self.conn.execute(
            "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?",
            (now, job_id),
        )
        await self.conn.commit()

    async def get_job(self, job_id: str) -> Optional[dict]:
        async with self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_jobs(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        if status:
            async with self.conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status=?", (status,)
            ) as cur:
                total = (await cur.fetchone())[0]
            async with self.conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ) as cur:
                items = [dict(r) for r in await cur.fetchall()]
        else:
            async with self.conn.execute("SELECT COUNT(*) FROM jobs") as cur:
                total = (await cur.fetchone())[0]
            async with self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ) as cur:
                items = [dict(r) for r in await cur.fetchall()]
        return items, total

    async def delete_job(self, job_id: str) -> None:
        await self.conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await self.conn.commit()

    # ---------------- stage_runs ----------------

    async def insert_stage_run(self, job_id: str, stage: str, log_path: str) -> int:
        now = time.time()
        async with self.conn.execute(
            "INSERT INTO stage_runs (job_id, stage, started_at, log_path) VALUES (?, ?, ?, ?)",
            (job_id, stage, now, log_path),
        ) as cur:
            run_id = cur.lastrowid
        await self.conn.commit()
        return run_id

    async def finish_stage_run(self, run_id: int, exit_code: int) -> None:
        now = time.time()
        await self.conn.execute(
            "UPDATE stage_runs SET finished_at=?, exit_code=? WHERE id=?",
            (now, exit_code, run_id),
        )
        await self.conn.commit()

    async def list_stage_runs(self, job_id: str) -> list[dict]:
        async with self.conn.execute(
            "SELECT * FROM stage_runs WHERE job_id=? ORDER BY started_at",
            (job_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ---------------- restart recovery ----------------

    async def reset_inflight_to_failed(self) -> int:
        """服务启动时调用:把任何非终态状态标为 FAILED。返回受影响行数。"""
        now = time.time()
        cursor = await self.conn.execute(
            """
            UPDATE jobs
            SET status='failed',
                error_stage='service_restarted',
                error_message='服务在任务执行中被重启',
                finished_at=?,
                updated_at=?
            WHERE status NOT IN ('succeeded','failed','cancelled')
            """,
            (now, now),
        )
        await self.conn.commit()
        return cursor.rowcount