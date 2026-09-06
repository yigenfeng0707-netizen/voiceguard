"""任务状态存储模块 - SQLite 持久化，断点续算与幂等。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from config import get_settings

_SCHEMA_VERSION = 1

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    params_fingerprint TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    current_stage TEXT DEFAULT '',
    progress REAL DEFAULT 0.0,
    result_path TEXT DEFAULT '',
    created_at REAL DEFAULT 0,
    updated_at REAL DEFAULT 0,
    schema_version INTEGER DEFAULT 1
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_idempotent ON tasks(file_hash, params_fingerprint);
CREATE TABLE IF NOT EXISTS chunk_progress (
    task_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    artifact_path TEXT DEFAULT '',
    completed_at REAL DEFAULT 0,
    PRIMARY KEY (task_id, chunk_index),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);
"""


def _get_db() -> sqlite3.Connection:
    settings = get_settings()
    db_dir = os.path.dirname(settings.sqlite_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_SQL)
    return conn


def compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_params_fingerprint(params: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def create_task(task_id: str, file_hash: str, params_fingerprint: str) -> None:
    db = _get_db()
    now = time.time()
    db.execute(
        "INSERT OR IGNORE INTO tasks (task_id, file_hash, params_fingerprint, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, file_hash, params_fingerprint, now, now),
    )
    db.commit()
    db.close()


def update_stage(
    task_id: str, stage: str, progress: float, status: str = "processing"
) -> None:
    db = _get_db()
    db.execute(
        "UPDATE tasks SET current_stage=?, progress=?, status=?, updated_at=? WHERE task_id=?",
        (stage, progress, status, time.time(), task_id),
    )
    db.commit()
    db.close()


def update_chunk_progress(
    task_id: str, chunk_index: int, status: str, artifact_path: str = ""
) -> None:
    db = _get_db()
    db.execute(
        "INSERT OR REPLACE INTO chunk_progress (task_id, chunk_index, status, artifact_path, completed_at) VALUES (?, ?, ?, ?, ?)",
        (
            task_id,
            chunk_index,
            status,
            artifact_path,
            time.time() if status == "done" else 0,
        ),
    )
    db.commit()
    db.close()


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    db = _get_db()
    row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    db.close()
    return dict(row) if row else None


def find_completed(file_hash: str, params_fingerprint: str) -> Optional[Dict[str, Any]]:
    db = _get_db()
    row = db.execute(
        "SELECT * FROM tasks WHERE file_hash=? AND params_fingerprint=? AND status='success'",
        (file_hash, params_fingerprint),
    ).fetchone()
    db.close()
    return dict(row) if row else None


def get_in_progress_tasks() -> List[Dict[str, Any]]:
    db = _get_db()
    rows = db.execute(
        "SELECT * FROM tasks WHERE status IN ('processing', 'interrupted')"
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_chunk_progress(task_id: str) -> List[Dict[str, Any]]:
    db = _get_db()
    rows = db.execute(
        "SELECT * FROM chunk_progress WHERE task_id=? ORDER BY chunk_index", (task_id,)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def mark_success(task_id: str, result_path: str) -> None:
    db = _get_db()
    db.execute(
        "UPDATE tasks SET status='success', result_path=?, progress=1.0, updated_at=? WHERE task_id=?",
        (result_path, time.time(), task_id),
    )
    db.commit()
    db.close()
