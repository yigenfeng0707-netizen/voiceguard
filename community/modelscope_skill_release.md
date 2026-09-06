# VoiceGuard - 魔搭 Skills Center 发布说明

## 基本信息

- **技能名称**: VoiceGuard 端侧语音合规质检助手
- **作者**: 根深叶茂
- **标签**: `Intel AI PC` `OpenVINO` `NPU` `语音质检` `合规风控` `端侧AI` `零隐私泄露`
- **分类**: AI 应用 / 语音处理 / 行业方案

## 一句话描述

把通话录音留在本机，用「规则引擎零 token 初筛 → NPU/GPU 本地模型语义复核 → 可选云端脱敏增强」的分级调度完成合规质检——敏感语音零外发，API token 节省 100%。

## 技术架构

```
录音 → [ASR 转写 · NPU] → [规则引擎 · CPU · 0 token]
    → [候选片段 → 本地 LLM 复核 · GPU] → [结构化质检报告]
    → (可选) [脱敏 → 云端 API 辅导建议 → 弱网降级]
```

异构三级协同：
- **NPU (Intel AI Boost)**: ASR Encoder (234M) + CTC Head (12.8M) 双组件，4.17x 加速
- **GPU (Intel Arc)**: LLM 语义复核 (Qwen3-1.7B INT4 / Ollama Vulkan)，1.9x~4.4x 加速
- **CPU**: 规则引擎 (40 条规则，3ms，零 token)

## 核心能力

1. **40 条行业规则**：金融 20 + 电销 20，覆盖保本保息/夸大收益/索要验证码/冒充机构/饥饿营销等真实违规场景
2. **三级降级**：NPU > GPU > CPU 自动探测，无需手动配置
3. **Skill 可组合**：3 个子技能 (vg-transcribe / vg-rules-check / vg-report-gen) 可独立调用
4. **批量运营**：qa_ops_daily 脚本支持目录扫描、批量质检、运营日报生成
5. **云增强骨架**：6 种 PII 脱敏 + 云端 API 占位 + 弱网自动降级

## 安装与运行

### 依赖环境

```bash
# Python 3.10+
pip install openvino openvino-genai funasr torch fastapi uvicorn
# Ollama (可选，作为 LLM 备选后端)
# 下载安装 https://ollama.com
ollama pull qwen2.5:3b
# GPU 加速: 设置环境变量 OLLAMA_IGPU_ENABLE=1
```

### 快速开始

```bash
# 1. 仅规则引擎筛查（无需模型，零依赖）
python scripts/run.py demo/samples/demo_finance_violation.wav --no-model --format markdown

# 2. 端到端质检（NPU ASR + GPU LLM）
python scripts/run.py demo/samples/demo_finance_violation.wav --format html --output report.html --device NPU

# 3. 使用 Ollama 后端
python scripts/run.py demo/samples/demo_finance_violation.wav --format html --reviewer-backend ollama

# 4. HTTP API 服务
python scripts/server.py --port 8765
# POST /v1/qa       完整质检
# POST /v1/qa/fast  快速筛查
# GET  /health       健康检查

# 5. 批量质检 + 运营日报
python scripts/qa_ops_daily.py --audio-dir ./recordings --output daily_report.html
```

## Benchmark

| 组件 | 设备 | 耗时 | 加速比 |
|------|------|------|--------|
| ASR (Encoder+CTC) | CPU | 7.623s | 1.0x |
| ASR (Encoder+CTC) | **NPU** | **1.775s** | **4.17x** |
| ASR (Encoder) | GPU | 2.083s | 3.7x |
| LLM (29 候选) | GPU batch=1 | 57.0s | 1.0x |
| LLM (29 候选) | **GPU batch=10** | **29.76s** | **1.9x** |
| LLM (备选) | Ollama CPU | 89.9s | 2.7 tok/s |
| LLM (备选) | **Ollama GPU Vulkan** | **20.8s** | **12.0 tok/s** |

## 验证结果

- **5 场景 E2E**: 金融违规/电销违规/保险违规 + 2 合规对照，全部通过
- **94 单元测试**: 覆盖 rules_engine/reviewer/pipeline/report_gen/transcribe
- **4 平台集成**: QwenWork/WorkBuddy/TraeWork/豆包，完整模式 100% 成功
- **Token 经济**: 0 API token (100% 节省)，敏感音频零外发

## 硬件要求

- Intel Core Ultra 处理器 (CPU + Arc GPU + NPU Intel AI Boost)
- 或任意支持 OpenVINO 的 Intel 硬件 (CPU/GPU)
- 内存: >= 16GB (ASR 模型 ~500MB + LLM ~1GB)

## 技术栈

- ASR: funasr + SenseVoiceSmall + OpenVINO IR (NPU/GPU/CPU 三后端)
- LLM: Qwen3-1.7B INT4 + OpenVINO GenAI / Ollama qwen2.5:3b (Vulkan)
- 规则引擎: 自研 DSL (keyword/regex/absent/semantic 四策略)
- HTTP API: FastAPI + uvicorn
- 异构调度: NPU(ASR) + GPU(LLM) + CPU(Rules) 三级协同

## 许可证

MIT License

## 赛事

英特尔 Agentic PC Skill 大赛全国总决赛 (2026-09-22~23 · 苏州 · Intel Connection)
