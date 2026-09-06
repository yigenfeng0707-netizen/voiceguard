"""链路追踪模块 - 基于 contextvars 注入 trace_id。"""

from __future__ import annotations

import contextvars
import uuid
from typing import Optional

_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> Optional[str]:
    return _trace_id_var.get()


def set_trace_id(tid: str) -> None:
    _trace_id_var.set(tid)


class with_trace:
    def __init__(self, trace_id: Optional[str] = None):
        self._tid = trace_id or new_trace_id()
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        self._token = _trace_id_var.set(self._tid)
        return self._tid

    def __exit__(self, *args):
        if self._token is not None:
            _trace_id_var.reset(self._token)