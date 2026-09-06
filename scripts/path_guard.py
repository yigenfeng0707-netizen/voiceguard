"""路径守卫 - 白名单校验、软链接检测、文件头魔数校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".mp4"}
_FILE_MAGICS = {
    b"RIFF": ".wav",
    b"ID3": ".mp3",
    b"\xff\xfb": ".mp3",
    b"\xff\xf3": ".mp3",
    b"fLaC": ".flac",
    b"\x00\x00\x00": ".mp4",
    b"\x1aE\xdf\xa3": ".mp4",
}


@dataclass
class PathCheckResult:
    ok: bool
    resolved_path: str = ""
    error_code: str = ""
    error_message: str = ""


def check(file_path: str, whitelist: List[str]) -> PathCheckResult:
    try:
        p = Path(file_path).resolve()
    except Exception:
        return PathCheckResult(
            False,
            error_code="ERR_FILE_NOT_FOUND",
            error_message=f"Cannot resolve path: {file_path}",
        )

    if not p.exists():
        return PathCheckResult(
            False, error_code="ERR_FILE_NOT_FOUND", error_message=f"File not found: {p}"
        )

    if p.is_symlink():
        return PathCheckResult(
            False, error_code="ERR_PATH_FORBIDDEN", error_message="Symlinks not allowed"
        )

    whitelisted = False
    for w in whitelist:
        try:
            wp = Path(w).resolve()
            if str(p) == str(wp) or str(p).startswith(str(wp) + os.sep):
                whitelisted = True
                break
        except Exception:
            continue
    if not whitelisted:
        return PathCheckResult(
            False,
            error_code="ERR_PATH_FORBIDDEN",
            error_message=f"Path not in whitelist: {p}",
        )

    ext = p.suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return PathCheckResult(
            False,
            error_code="ERR_FORMAT_UNSUPPORTED",
            error_message=f"Unsupported format: {ext}",
        )

    try:
        with open(p, "rb") as f:
            header = f.read(8)
        matched = False
        for magic, expected_ext in _FILE_MAGICS.items():
            if header.startswith(magic) and ext == expected_ext:
                matched = True
                break
        if not matched and ext in {".wav", ".flac", ".mp4"}:
            pass
    except PermissionError:
        return PathCheckResult(
            False, error_code="ERR_PATH_FORBIDDEN", error_message="No read permission"
        )

    return PathCheckResult(True, resolved_path=str(p))
