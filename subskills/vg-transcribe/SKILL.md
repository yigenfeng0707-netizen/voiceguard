---
name: vg-transcribe
description: |
  VoiceGuard ASR 转写子技能。将通话录音（wav/mp3/m4a/flac/mp4）在本地完成
  语音识别，输出带时间戳的逐字稿。基于 SenseVoiceSmall + OpenVINO IR，
  支持异构加速（NPU Encoder+CTC 双组件 / GPU / CPU 三后端自动降级）。
  录音零外发，全部在端侧完成。可被 vg-rules-check 或上层技能（如 qa-ops-daily）调用。
  Use when the user needs to transcribe call recordings locally on Intel AIPC.
---

# vg-transcribe · 端侧语音转写子技能

## 功能

把通话录音留在本机，用 SenseVoiceSmall 模型 + OpenVINO 异构加速完成语音识别。
录音不出设备，逐字稿零外发。

## 异构加速

| 组件 | 模型 | NPU | GPU | CPU |
|------|------|-----|-----|-----|
| Encoder | SenseVoiceSmall (234M) | 1.69s (RTF=0.030) | 2.08s (RTF=0.037) | 7.62s (RTF=0.135) |
| CTC Head | ctc_lo (12.8M, FP32) | 8.86ms/call | 8.19ms/call | - |

NPU 模式：Encoder + CTC Head 双组件均在 Intel AI Boost NPU 上运行，4.2x 加速。
CTC Head 需 FP32 精度（FP16 对 25055 维投影 max diff 14.1 不可用）。

## 用法

### 独立 CLI 调用（推荐，Skill 级入口）

```
python subskills/vg-transcribe/run.py <音频文件> [--device auto|cpu|gpu|npu] [--language auto] [--output seg.json]
```

输出 segments JSON 契约（`skill: vg-transcribe`），可直接被 `vg-rules-check --segments` 消费。

### Python API

```python
from transcribe import transcribe

result = transcribe(
    "call.wav",
    device="auto",       # auto / cpu / gpu / npu
    language="auto",     # auto / zh / en / ja / ko
)
# result = {"segments": [...], "text": str, "lang": str, "asr_stats": dict}
```

### 输出

```json
{
  "segments": [
    {"segment_id": 0, "text": "您好，我是...", "start_ms": 0, "end_ms": 3000, "speaker": ""}
  ],
  "text": "完整逐字稿文本",
  "lang": "zh",
  "device": "NPU",
  "asr_stats": {
    "device": "NPU",
    "ctc_device": "NPU",
    "infer_time_s": 1.69,
    "rtf": 0.030,
    "ctc_infer_s": 0.086,
    "ctc_call_count": 10,
    "ctc_avg_latency_ms": 8.86
  }
}
```

## 模型路径

- PyTorch 原始模型: `models/SenseVoiceSmall/`
- Encoder OpenVINO IR: `models/SenseVoiceSmall_ov/encoder.xml` (422MB bin)
- CTC Head FP32 IR: `models/SenseVoiceSmall_ov/ctc_head_fp32.xml` (50MB bin)

## 依赖

- funasr >= 1.4.0
- OpenVINO >= 2024.0
- torch (CPU 即可)
- soundfile, scipy

## 被调用关系

```
qa-ops-daily (上层技能)
  ├── vg-transcribe (本技能) → 输出 segments
  ├── vg-rules-check → 输入 segments → 输出 hits
  └── vg-report-gen → 输入 hits → 输出 QAReport
```

> 实证：`vg-rules-check --audio` 模式即以子进程调用本技能 CLI（`subskills/vg-transcribe/run.py`）。
