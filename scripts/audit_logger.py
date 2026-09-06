"""审计日志模块 - JSON Lines 格式，30天滚动清理。"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from config import get_settings
from tracing import get_trace_id


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def log(
    caller: str,
    file_hash: str,
    stage: str,
    duration_ms: float,
    status: str,
    backend: str = "",
    message: str = "",
) -> None:
    settings = get_settings()
    entry: Dict[str, Any] = {
        "timestamp": time.time(),
        "trace_id": get_trace_id() or "",
        "caller": caller,
        "file_hash": file_hash,
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
        "status": status,
        "backend": backend,
        "message": message,
    }
    _ensure_dir(settings.audit_log_path)
    with open(settings.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def cleanup_old_logs() -> None:
    settings = get_settings()
    path = settings.audit_log_path
    if not os.path.exists(path):
        return
    cutoff = time.time() - settings.audit_retention_days * 86400
    kept: list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if rec.get("timestamp", 0) >= cutoff:
                    kept.append(line.strip())
            except json.JSONDecodeError:
                continue
    with open(path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
