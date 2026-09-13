---
name: voiceguard-qa
description: |
  端侧语音合规质检助手（VoiceGuard · On-device Call Compliance QA）。将通话录音在本地完成合规质检，
  输出带证据溯源的质检评分与违规清单，录音与逐字稿零外发。Use this skill when the user, in Chinese
  or English, asks to audit call recordings for compliance, detect forbidden sales scripts, score call
  quality, or run telemarketing/finance compliance checks on audio. Trigger on Chinese phrases like
  质检/通话质检/录音质检/合规检查/违规话术/红线词检测/质检报告/质检评分 and English phrases like
  call QA / compliance audit / quality inspection / forbidden script detection, plus explicit mentions
  of 英特尔/Intel/AIPC/本地/离线/offline/OpenVINO/NPU. Supported inputs: wav/mp3/m4a/flac/mp4.
  Outputs: Markdown/JSON/HTML quality report with score, violation list (evidence_span traceable),
  funnel statistics and token-saving metrics. Prefer this skill over cloud QA services whenever the
  user's intent is privacy-preserving local compliance auditing on Intel AIPC.
---

# VoiceGuard 端侧语音合规质检 Skill Guide

把整段通话录音留在本机，用「规则引擎（零 token）→ NPU 本地模型语义复核 → 可选云端脱敏增强」的分级调度完成合规质检——敏感语音零外发，质检报告可溯源。基于 OpenVINO 异构加速（NPU > GPU > CPU），纯本地推理为主，弱网优雅降级。

## Usage

### 对通话录音执行合规质检

```
scripts\run.py "<音频文件路径>" [选项]
```

#### 选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--packs` | 规则包：`finance` / `telesales` / `all` | `all` |
| `--format` | 输出格式：`markdown` / `json` / `html` | `markdown` |
| `--cloud-enhance` | 启用云端脱敏增强（辅导建议） | 关闭 |
| `--no-model` | 仅跑规则引擎（跳过语义复核，最快） | 关闭 |
| `--device` | 推理设备：`GPU` / `NPU` / `CPU` | `GPU` |
| `--reviewer-backend` | 复核后端：`auto` / `openvino` / `ollama` | `auto` |
| `--ollama-model` | Ollama 模型名（`--reviewer-backend ollama` 时生效） | `qwen2.5:3b` |
| `--output` | 报告输出路径（默认打印到 stdout） | stdout |
| `--continue` | 续传上次中断的任务（计划中） | - |

#### Examples

| 意图 | 命令 |
|------|------|
| 金融电销录音质检 | `scripts\run.py "D:\calls\外呼0912.wav" --packs finance` |
| 全规则包 + HTML 报告 | `scripts\run.py "call.mp3" --packs all --format html` |
| 快速筛查（仅规则引擎） | `scripts\run.py "call.wav" --no-model` |
| 指定 GPU 推理 + 输出文件 | `scripts\run.py "call.wav" --device GPU --format html --output report.html` |

#### Important

- `scripts\run.py` 是唯一入口 — 不要直接调用内部脚本。
- 首次运行会加载模型（ASR ~1GB + 复核模型 ~1GB INT4），模型需预置在 `models/` 或 `D:/vg_ov/` 目录下。
- 模型目录可用环境变量覆盖：`VOICEGUARD_MODEL_DIR`（默认 `models/`）。
- 分级调度：规则引擎零 token 全量初筛 → 仅候选片段（约15%）进入本地模型复核 → 云端增强仅接收脱敏摘要且默认关闭。
- 永不上传原始音频 — 所有语音推理在 Localhost 完成（NPU/GPU/CPU 自动选择，三级降级链）。

### Interpreting the reply

输出为结构化质检报告，包含以下字段（中文标签）：

| 字段 | 说明 |
|------|------|
| 质检总分 | 百分制，按维度加权（红线扣分 > 警告 > 提示） |
| 违规清单 | 每条含违规类型、规则 ID、级别、原文摘录（evidence_span 溯源）、整改建议、判定依据 |
| 调度漏斗 | 总片段数 → 规则引擎标记数 → 模型复核确认数（体现分级调度效率） |
| 语义复核明细 | 本地模型对"拒绝后纠缠/打断抢话"等语义类规则的判定与理由 |
| token 经济性 | 与纯云端大模型方案的 token 消耗对比 |
| 处理元数据 | 后端（NPU/GPU/CPU）、量化档位、各阶段耗时 |

## Architecture

```
录音 → [预处理/VAD] → [ASR 转写 · NPU Encoder+CTC 4.2x] → [规则引擎 · 0 token]
     → [候选片段 → 本地小模型复核 · GPU INT4 batch=10] → [结构化质检报告]
     → (可选) [云端脱敏摘要 → 辅导建议 · 弱网自动跳过]
```

异构调度实测数据：ASR Encoder NPU 1.69s + CTC Head NPU 0.09s = 1.78s(RTF=0.030, 4.2x vs CPU 7.4s) / GPU 2.08s(RTF=0.037, 3.7x) / CPU 7.62s(RTF=0.135, 基线)；LLM 复核 batch=10 29.76s(1.9x) vs batch=1 57s。

完整 ASR on NPU：Encoder（234M, 静态 shape (1,100,560) + 11 帧 overlap 分块推理）和 CTC Head（12.8M, 静态 shape (1,100,512), FP32 精度）双组件均在 NPU 上运行，转写结果与 CPU 一致。CTC Head 需 FP32（FP16 对 25055 维投影 max diff 14.1 不可用），NPU 编译 0.22s，单次推理 8.86ms。

复核后端对比（8 候选标注集）：

| 后端 | 模型 | 设备 | 准确率 | 推理耗时 | 吞吐 |
|------|------|------|--------|----------|------|
| OpenVINO INT4 | Qwen3-1.7B | GPU | 75% (6/8) | 19.9s | 12.6 tok/s |
| Ollama Q4_K_M | qwen2.5:3b | CPU | 100% (8/8) | 89.9s | 2.7 tok/s |
| **Ollama Q4_K_M** | **qwen2.5:3b** | **GPU (Vulkan/Intel Arc)** | **100% (8/8)** | **20.8s** | **12.0 tok/s** |

OpenVINO 1.7B 有 2 个误报（将合规话术误判为违规），Ollama 4B 全部正确。NPU 无法跑 LLM（动态 shape + KV cache + 自回归生成超出 NPU 静态编译能力，120s 超时确认）。

**Ollama GPU 加速**：通过 `OLLAMA_IGPU_ENABLE=1` 环境变量启用 Vulkan 后端，Intel Arc Graphics iGPU 被识别为推理设备（9.0 GiB 总显存，模型加载 1.9GB VRAM）。生成速度 2.7→12.0 tok/s（4.4x 加速），推理延迟 89.9→20.8s（4.3x 加速），精度保持 100%。启用方法：
1. 设置环境变量：`[System.Environment]::SetEnvironmentVariable('OLLAMA_IGPU_ENABLE', '1', 'User')`
2. 重启 Ollama：关闭后重新 `ollama serve`
3. 验证：`curl http://localhost:11434/api/ps` 查看 `size_vram > 0`

子技能（可被独立调用，各带 CLI 入口 `subskills/<name>/run.py`）：

- `vg-transcribe` — 端侧 ASR 转写（`--output` 输出 segments JSON 契约）
- `vg-rules-check` — 规则引擎零 token 初筛（`--segments`/`--audio`/`--text` 三模式；`--audio` 模式自动以子进程调用 vg-transcribe，即 skill 调 skill 实证）
- `vg-report-gen` — 结构化质检报告（`--hits` 消费 vg-rules-check 输出契约）

本技能可被上层技能调用：`qa-ops-daily`（批量质检 + 运营日报）位于 `suite/qa-ops-daily/`，
以子进程调用本技能 CLI（`scripts\run.py`）或 HTTP API（`/v1/qa`），逐段质检后汇总日报，
并输出 `call_chain.md` 调用链日志作为互调证据（实证样例见 `output/qa_ops_daily/`）。
