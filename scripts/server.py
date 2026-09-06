"""VoiceGuard HTTP API 服务（D4 四平台集成）。

将 run_full_pipeline() 包装为 FastAPI HTTP 接口，
供 WorkBuddy / TraeWork / 豆包办公等 Agent 平台通过 API 调用。

启动：
  python scripts/server.py                        # 默认 0.0.0.0:8765
  python scripts/server.py --host 127.0.0.1 --port 9000

接口：
  GET  /health        健康检查
  GET  /              API 信息
  POST /v1/qa         合规质检（上传音频 → 返回 JSON 报告）
  POST /v1/qa/fast    仅规则引擎（跳过模型，秒级响应）
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows GBK 控制台保护
if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in (
    "utf-8",
    "utf8",
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="VoiceGuard QA API",
    description="端侧语音合规质检 HTTP 服务",
    version="1.0.0",
)

# 模型懒加载：首次调用 /v1/qa 时初始化，避免启动时阻塞
_pipeline_ready = False
_load_error = None


def _ensure_pipeline():
    """检查 pipeline 依赖是否可用。"""
    global _pipeline_ready, _load_error
    if not _pipeline_ready:
        try:
            from pipeline import run_full_pipeline  # noqa: F401
            from report_gen import render_json  # noqa: F401

            _pipeline_ready = True
        except Exception as e:
            _load_error = str(e)


@app.get("/health")
async def health():
    """健康检查：返回服务状态和模型可用性。"""
    _ensure_pipeline()
    return {
        "status": "ok",
        "pipeline_ready": _pipeline_ready,
        "load_error": _load_error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@app.get("/")
async def api_info():
    """API 信息：列出可用接口。"""
    return {
        "service": "VoiceGuard QA API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "qa": "POST /v1/qa (audio file upload -> JSON report)",
            "qa_fast": "POST /v1/qa/fast (rule-engine only, no model)",
        },
        "supported_formats": ["wav", "mp3", "m4a", "flac", "mp4"],
    }


@app.post("/v1/qa")
async def qa(
    audio: UploadFile = File(..., description="音频文件 wav/mp3/m4a/flac"),
    packs: str = Query("all", description="规则包: all / finance / telesales"),
    device: str = Query("GPU", description="推理设备: GPU / NPU / CPU"),
):
    """合规质检：上传音频 → ASR → 规则引擎 → 语义复核 → JSON 报告。"""
    _ensure_pipeline()
    if not _pipeline_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not ready: {_load_error or 'unknown error'}",
        )

    from pipeline import run_full_pipeline
    from report_gen import render_json

    # 保存上传文件到临时路径
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="vg_qa_")
    try:
        content = await audio.read()
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)

        # 解析规则包
        pack_list = (
            None
            if packs == "all"
            else [p.strip() for p in packs.split(",") if p.strip()]
        )

        t0 = time.perf_counter()
        try:
            result = run_full_pipeline(
                tmp_path,
                packs=pack_list,
                device=device,
                no_model=False,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=f"文件不存在: {e}")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {type(e).__name__}: {e}",
            )

        elapsed = round(time.perf_counter() - t0, 2)
        report_json = render_json(result.report)

        import json

        report_dict = json.loads(report_json)
        report_dict["api_elapsed_s"] = elapsed
        return JSONResponse(content=report_dict)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/v1/qa/fast")
async def qa_fast(
    audio: UploadFile = File(..., description="音频文件 wav/mp3/m4a/flac"),
    packs: str = Query("all", description="规则包: all / finance / telesales"),
):
    """快速筛查：仅规则引擎（跳过模型语义复核），秒级响应。"""
    _ensure_pipeline()
    if not _pipeline_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not ready: {_load_error or 'unknown error'}",
        )

    from pipeline import run_full_pipeline
    from report_gen import render_json

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="vg_fast_")
    try:
        content = await audio.read()
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)

        pack_list = (
            None
            if packs == "all"
            else [p.strip() for p in packs.split(",") if p.strip()]
        )

        t0 = time.perf_counter()
        try:
            result = run_full_pipeline(
                tmp_path,
                packs=pack_list,
                device="CPU",
                no_model=True,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {type(e).__name__}: {e}",
            )

        elapsed = round(time.perf_counter() - t0, 2)
        import json

        report_dict = json.loads(render_json(result.report))
        report_dict["api_elapsed_s"] = elapsed
        return JSONResponse(content=report_dict)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description="VoiceGuard QA API Server")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址")
    ap.add_argument("--port", type=int, default=8765, help="监听端口")
    args = ap.parse_args()

    print(f"VoiceGuard QA API starting on {args.host}:{args.port}")
    print(f"  GET  /health        - 健康检查")
    print(f"  GET  /               - API 信息")
    print(f"  POST /v1/qa          - 合规质检（含模型复核）")
    print(f"  POST /v1/qa/fast     - 快速筛查（仅规则引擎）")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
