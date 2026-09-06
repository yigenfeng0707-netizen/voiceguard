"""监控指标模块 - 基于 prometheus_client。"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

if _AVAILABLE:
    _task_counter = Counter(
        "voiceguard_tasks_total", "Total processing tasks", ["status"]
    )
    _stage_histogram = Histogram(
        "voiceguard_stage_duration_seconds",
        "Stage duration in seconds",
        ["stage", "backend", "status"],
    )
    _active_gauge = Gauge("voiceguard_active_tasks", "Currently active tasks")
    _memory_gauge = Gauge("voiceguard_memory_mb", "Memory usage in MB", ["type"])
    _backend_gauge = Gauge("voiceguard_backend_info", "Backend info", ["backend"])


def record_stage_duration(
    stage: str, duration_ms: float, backend: str = "", status: str = "success"
) -> None:
    if _AVAILABLE:
        _stage_histogram.labels(stage=stage, backend=backend, status=status).observe(
            duration_ms / 1000.0
        )
        _task_counter.labels(status=status).inc()


def set_resource_usage(
    memory_mb: float, vmemory_mb: float, backend: str = "cpu"
) -> None:
    if _AVAILABLE:
        _memory_gauge.labels(type="rss").set(memory_mb)
        _memory_gauge.labels(type="virtual").set(vmemory_mb)
        _backend_gauge.labels(backend=backend).set(1)


def inc_active() -> None:
    if _AVAILABLE:
        _active_gauge.inc()


def dec_active() -> None:
    if _AVAILABLE:
        _active_gauge.dec()


def get_metrics() -> bytes:
    if _AVAILABLE:
        return generate_latest()
    return b""
