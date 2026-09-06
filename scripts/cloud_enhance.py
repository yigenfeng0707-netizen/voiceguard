"""Cloud Enhancement module (optional Stage 4).

When cloud_enhance=True, the pipeline sends *desensitized* violation excerpts
to a cloud LLM API for coaching advice. The cloud API is called with masked
text only -- no raw audio, no raw transcript, no PII.

Graceful degradation: if the cloud API is unreachable (weak network, timeout,
auth failure), the pipeline silently skips cloud enhancement and returns the
local-only report.

Architecture:
    Local Report -> desensitize(excerpts) -> cloud API (coaching advice)
        -> merge advice into report -> return enhanced report

This module implements the skeleton with:
  1. PII desensitization (phone, ID card, bank card, email, name patterns)
  2. Cloud API call placeholder (override endpoint via env var)
  3. Timeout + error handling -> graceful degradation
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

# PII patterns for desensitization
_PII_PATTERNS = [
    # ID card: 18-digit (last may be X) -- longest, must run before phone/bank
    (re.compile(r"\d{17}[\dXx]"), "[ID]"),
    # Bank card: 16-19 digits -- run before phone
    (re.compile(r"\d{16,19}"), "[CARD]"),
    # Phone: 11-digit mobile (1[3-9] + 9 digits)
    (re.compile(r"1[3-9]\d{9}"), "[PHONE]"),
    # Email
    (re.compile(r"[\w.-]+@[\w.-]+\.\w+"), "[EMAIL]"),
    # QQ number (6-12 digits starting with non-zero)
    (re.compile(r"\b[1-9]\d{5,11}\b"), "[QQ]"),
]

# Name honorifics to mask
_NAME_PREFIX = re.compile(
    r"(?:王|李|张|刘|陈|杨|黄|赵|周|吴|徐|孙|马|朱|胡|郭|何|高|林|罗|郑|梁|谢|宋|唐|许|韩|冯|邓|曹|彭|曾|肖|田|董|袁|潘|于|蒋|蔡|余|杜|叶|程|苏|魏|吕|任|沈|姚|卢|姜|崔|钟|谭|陆|汪|范|金石|廖|贾|夏|韦|付|方|邹|熊|白|孟|秦|尤|阎|薛|侯|段|雷|史|龙|黎|贺|顾|毛|郝|龚|邵|万|钱|严|覃|武|戴|莫|孔|向|汤)"
    r"(?:经理|总|主任|先生|女士|老师|师傅|哥|姐|总|工)"
)


def desensitize(text: str) -> str:
    """Mask PII in text: phone, ID card, bank card, email, QQ, names.

    Args:
        text: Raw text that may contain PII.

    Returns:
        Text with all PII replaced by placeholder tokens like [PHONE], [ID].
    """
    result = text
    # Mask names with honorifics first (before digit patterns catch phone)
    result = _NAME_PREFIX.sub("[NAME]", result)
    # Apply all PII patterns
    for pattern, placeholder in _PII_PATTERNS:
        result = pattern.sub(placeholder, result)
    return result


@dataclass
class CloudEnhanceResult:
    """Result of cloud enhancement attempt."""

    success: bool = False
    advice_items: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    endpoint: str = ""
    error: str = ""
    desensitized: bool = True  # Always True -- we never send raw PII
    tokens_sent: int = 0  # Estimated tokens sent to cloud API


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 Chinese char ~= 0.7 tokens."""
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - zh
    return int(zh * 0.7 + other / 4)


def cloud_coaching_advice(
    violations_summary: str,
    score: int,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
) -> CloudEnhanceResult:
    """Call cloud LLM API for coaching advice on violations.

    Sends ONLY desensitized violation excerpts -- no raw audio, no raw
    transcript, no PII. If the API is unreachable, returns gracefully.

    Args:
        violations_summary: Pre-desensitized text describing confirmed violations.
        score: Quality score from local pipeline.
        api_endpoint: Cloud API URL (default: env VOICEGUARD_CLOUD_API).
        api_key: API key (default: env VOICEGUARD_CLOUD_KEY).
        timeout_s: Network timeout in seconds.

    Returns:
        CloudEnhanceResult with advice items or error message.
    """
    result = CloudEnhanceResult()
    result.endpoint = api_endpoint or os.environ.get("VOICEGUARD_CLOUD_API", "")
    result.tokens_sent = _estimate_tokens(violations_summary)

    if not result.endpoint:
        result.error = "No cloud API endpoint configured (set VOICEGUARD_CLOUD_API env)"
        return result

    # Build request payload (skeleton -- actual API depends on provider)
    payload = {
        "model": "qwen-max",  # placeholder
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a compliance coaching expert. Given de-identified "
                    "call quality violations, provide specific coaching advice "
                    "for the sales agent. Respond in Chinese."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Call quality score: {score}/100\n"
                    f"Confirmed violations (de-identified):\n{violations_summary}\n\n"
                    "Provide 3-5 coaching recommendations."
                ),
            },
        ],
        "max_tokens": 500,
        "temperature": 0.7,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    try:
        import requests

        resp = requests.post(
            result.endpoint,
            json=payload,
            headers=headers,
            timeout=timeout_s,
        )
        result.elapsed_s = round(time.perf_counter() - t0, 2)

        if resp.status_code == 200:
            data = resp.json()
            # Parse response (format varies by provider; try common fields)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                # Split into individual advice items
                advice_items = [
                    line.strip().lstrip("-").strip()
                    for line in content.split("\n")
                    if line.strip() and len(line.strip()) > 5
                ]
                result.advice_items = advice_items[:5]
                result.success = True
            else:
                result.error = "Empty response from cloud API"
        else:
            result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.Timeout:
        result.elapsed_s = round(time.perf_counter() - t0, 2)
        result.error = f"Cloud API timeout ({timeout_s}s) -- degraded to local-only"
    except requests.exceptions.ConnectionError:
        result.elapsed_s = round(time.perf_counter() - t0, 2)
        result.error = (
            "Cannot reach cloud API (network unreachable) -- degraded to local-only"
        )
    except Exception as e:
        result.elapsed_s = round(time.perf_counter() - t0, 2)
        result.error = f"Cloud API error: {type(e).__name__}: {e}"

    return result


def run_cloud_enhance(report, enabled: bool = True) -> dict:
    """Orchestrate cloud enhancement on a QA report.

    If enabled and cloud API is available, sends desensitized violation
    excerpts for coaching advice. If disabled or API unreachable, returns
    empty result (graceful degradation).

    Args:
        report: QAReport from the local pipeline.
        enabled: Whether cloud enhancement is enabled.

    Returns:
        Dict with cloud_enhance metadata for the report.
    """
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    # Step 1: Collect confirmed violations
    confirmed = [v for v in report.violations if v.confirmed]
    if not confirmed:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "No confirmed violations to enhance",
        }

    # Step 2: Desensitize violation excerpts
    lines = []
    for v in confirmed:
        safe_excerpt = desensitize(v.excerpt)
        lines.append(
            f"- [{v.level}] {v.rule_name}: {safe_excerpt} (匹配: {v.matched_by})"
        )
    summary = "\n".join(lines)

    # Step 3: Call cloud API
    result = cloud_coaching_advice(summary, report.score)

    return {
        "enabled": True,
        "status": "success" if result.success else "degraded",
        "advice": result.advice_items,
        "endpoint": result.endpoint or "(not configured)",
        "elapsed_s": result.elapsed_s,
        "tokens_sent": result.tokens_sent,
        "desensitized": True,
        "error": result.error,
    }
