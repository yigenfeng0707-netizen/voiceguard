"""配置外置模块 - 基于 pydantic-settings 加载配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    service_port: int = 8765
    cold_start_timeout: int = 60
    max_audio_minutes: int = 240
    chunk_seconds: float = 30.0
    chunk_overlap: float = 0.2
    target_sample_rate: int = 16000
    max_concurrent: int = 2
    confidence_threshold: float = 0.6
    speaker_min_switch: float = 1.0
    speaker_pause_merge: float = 3.0
    topic_max_title: int = 20
    summary_min_chars: int = 50
    audit_retention_days: int = 30
    graceful_shutdown_timeout: int = 30

    asr_model_path: str = (
        "D:/APPs/OpenVINO/demo/local-meeting-minutes/models/SenseVoiceSmall"
    )
    diarization_model_path: str = "models/pyannote-embed-ov"
    nlp_model_path: str = "D:/vg_ov/Qwen3-1.7B-ov-int4"
    summary_model_path: str = "D:/vg_ov/Qwen3-1.7B-ov-int4"

    backend_preference: str = "auto"
    quantization_default: str = "auto"
    output_dir: str = "output"
    sensitive_dict_path: str = "rules"  # VoiceGuard 规则目录
    path_whitelist: List[str] = field(default_factory=lambda: ["."])
    auth_token: Optional[str] = None

    sqlite_path: str = "data/task_store.db"
    audit_log_path: str = "data/audit.jsonl"

    # D2: 分级调度配置
    reviewer_device: str = "GPU"  # GPU / NPU / CPU
    reviewer_backend: str = "auto"  # auto / openvino / ollama
    reviewer_max_new_tokens: int = 200
    reviewer_batch_size: int = 10  # D2优化: 批量推理减少 prompt 开销
    ollama_host: str = "localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    enable_semantic_review: bool = True
    rules_dir: str = "rules"
    model_search_bases: List[str] = field(
        default_factory=lambda: ["models", r"D:\vg_ov"]
    )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        port = os.environ.get("SERVICE_PORT")
        if port:
            _settings.service_port = int(port)
        backend = os.environ.get("BACKEND_PREFERENCE")
        if backend:
            _settings.backend_preference = backend
    return _settings
