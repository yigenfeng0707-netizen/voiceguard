# VoiceGuard · 端侧语音合规质检助手

> 把整段通话录音留在本机，用「规则引擎（零 token）→ NPU/GPU 本地模型语义复核 → 可选云端脱敏增强」的分级调度完成合规质检——敏感语音零外发。

**参赛作品**：英特尔 Agentic PC Skill 大赛全国总决赛（2026-09-22~23 · 苏州 · Intel Connection）

## 状态

- [x] D0 项目骨架 + OpenVINO 三后端（CPU/GPU/NPU）冒烟通过
- [x] D0 规则引擎 V1（金融 12 规则 + 电销 12 规则，3ms 零 token 初筛自测通过）
- [x] D1 规则库扩充至 40 条（金融 20 + 电销 20），覆盖保本保息/夸大收益/类存款误导/索要验证码/冒充机构/泄露信息等真实违规场景
- [x] D1 ASR 流水线验证通过（funasr/SenseVoiceSmall，CPU 基线 RTF 0.135）
- [x] D1 Qwen3-1.7B INT4 OpenVINO 转换成功，GPU 推理验证通过（~10 tok/s）
- [x] D1 19 个脚本代码审查与健壮性修复完成（12 处问题修复，py_compile 0 错误）
- [x] D1 Demo 音频制作完成（3 段：金融违规/电销违规/合规对照）
- [x] D1 E2E 端到端验证通过（ASR → 规则引擎 → 语义复核 → HTML/JSON 报告输出）
- [x] D1 94 个单元测试全部通过（rules_engine / reviewer / transcribe / report_gen，0.8s 完成）
- [x] D2-D3 分级调度编排完成：pipeline.run_full_pipeline() 实现 Stage1→Stage2→Stage3 全链路一键编排；HTML 报告增加 CSS 漏斗可视化柱状图 + 摘要卡片；config 增加分级行为配置
- [x] D2-D3 E2E 三场景验证通过（金融 17 confirmed / 电销 20 confirmed / 合规 100分 0 违规，reason 全非空）
- [x] D4 四平台集成（QwenWork/WorkBuddy/TraeWork/豆包办公）+ benchmark（三平台 100% 成功率）
- [x] D5 Demo 视频制作完成（1920x1080 30fps 73s）+ ModelScope 开发者实践文章
- [x] D6 提交包打包完成（截止 2026-09-10 24:00）
- [x] **冠军品质改造**：NPU ASR 实跑 4.5x 加速 / LLM batch 优化 1.9x 加速 / 5 段 Demo 音频 / 完整模式集成测试 100% 成功

## 目录

| 目录 | 用途 |
|------|------|
| `scripts/` | 流水线脚本（预处理/ASR/规则引擎/复核/报告/调度/HTTP API/NPU 加速） |
| `subskills/` | 3 个可独立调用子技能（vg-transcribe / vg-rules-check / vg-report-gen，各带 `run.py` CLI 入口与 JSON 数据契约） |
| `rules/` | 行业规则包（finance.json / telesales.json，可插拔） |
| `models/` | OpenVINO 量化模型 + ASR encoder IR（gitignore） |
| `agent_integrations/` | 四大 Agent 平台集成配置与验证 |
| `demo/` | 5 段自制样例音频 + 预置报告 + 一键 demo |
| `docs/` | 架构图 / 技术深度 / benchmark 报告 |
| `community/` | ModelScope 开发者实践文章 |
| `tests/` | 测试用例（94 个，覆盖核心模块） |
| `output/` | 质检报告 + 集成报告 + benchmark 数据 |

**上层技能（Skill 互调实证）**：`suite/qa-ops-daily/` — 质检运营日报技能，以独立子进程批量调用本技能 CLI（或 HTTP API）
完成当日全量质检并汇总日报 + 调用链日志（实证样例：`output/qa_ops_daily/call_chain.md`）。

## 快速开始

### 1. 自测规则引擎（零依赖）

```
cd scripts
python rules_engine.py
```

### 2. 端到端质检（需 OpenVINO + funasr 环境）

```
python scripts/run.py demo/samples/demo_finance_violation.wav --format html --output output/report.html --device GPU
```

输出：ASR 转写 → 规则引擎初筛 → Qwen3-1.7B INT4 语义复核 → HTML 质检报告

### 3. 仅规则引擎筛查（无需模型）

```
python scripts/run.py demo/samples/demo_finance_violation.wav --no-model --format markdown
```

### 4. HTTP API 服务（四平台集成）

```
python scripts/server.py --port 8765
# POST /v1/qa       完整质检（ASR + 规则 + 模型复核）
# POST /v1/qa/fast  快速筛查（仅规则引擎）
# GET  /health       健康检查
```

## 模型配置

| 组件 | 模型 | 后端 | 路径 |
|------|------|------|------|
| ASR Encoder | SenseVoiceSmall (234M) | NPU > GPU > CPU 自动降级 | `models/SenseVoiceSmall_ov/encoder.xml` (OpenVINO IR, 分块推理) |
| ASR CTC Head | SenseVoiceSmall ctc_lo (12.8M) | NPU (与 Encoder 同设备) | `models/SenseVoiceSmall_ov/ctc_head_fp32.xml` (OpenVINO IR, FP32) |
| 语义复核 | Qwen3-1.7B INT4 (~1GB) | GPU | `D:/vg_ov/Qwen3-1.7B-ov-int4` |
| 语义复核(备选) | qwen2.5:3b (4B Q4_K_M) | GPU (Vulkan) / CPU | Ollama `qwen2.5:3b`，`OLLAMA_IGPU_ENABLE=1` |
| 规则引擎 | 内置 DSL | CPU（零 token） | `rules/finance.json + telesales.json` |

## 异构加速 Benchmark

### Complete ASR on NPU (SenseVoiceSmall 234M, 56.4s 音频)

| 设备 | 前向耗时 | RTF | 编译耗时 | 加速比 | 备注 |
|------|----------|-----|----------|--------|------|
| CPU (PyTorch) | 7.623s | 0.135 | 0s | 1.0x | funasr 全序列推理基线 |
| **NPU (Intel AI Boost)** | **1.775s** | **0.030** | 4.01s | **4.2x** | Encoder(234M) + CTC Head(12.8M) 双组件均在 NPU |
| GPU (Intel Arc) | 2.083s | 0.037 | 9.45s | 3.7x | OpenVINO GPU 后端 |

NPU 双组件策略：
- **Encoder**：SANM 局部注意力（kernel=11）→ 100 帧 chunk + 11 帧 overlap，stride=78，静态 shape (1,100,560)
- **CTC Head**：Linear(512→25055) 固定投影，静态 shape (1,100,512)，**FP32 精度**（FP16 对 25055 维投影 max diff 14.1 不可用），NPU 编译 0.22s，单次推理 8.86ms

ASR on NPU 组件分解：

| 组件 | 参数量 | NPU 编译 | NPU 推理 | 调用次数 | 总推理 |
|------|--------|----------|----------|----------|--------|
| Encoder | 234M | 3.82s | 129.93ms/chunk | 13 | 1.69s |
| CTC Head | 12.8M | 0.22s | 8.86ms/call | 10 | 0.09s |

### LLM 语义复核 (Qwen3-1.7B INT4, 29 候选片段)

| 配置 | 复核耗时 | 调用次数 | 加速比 | 备注 |
|------|----------|----------|--------|------|
| v9 batch_size=1 | 57.0s | 29 次 | 1.0x | 每候选单独推理 |
| **v10 batch_size=10** | **29.76s** | 3 次 | **1.9x** | 批量推理 + 动态 max_new_tokens |

优化细节：`/no_think` 跳过思考链；`batch_size=10` 合并候选；`max_new_tokens=min(500, batch*50+20)`；`chr(60)+"/think"+chr(62)` 安全构造标签。

### Ollama GPU 加速 (Vulkan / Intel Arc)

| 配置 | 模型 | 设备 | 准确率 | 推理耗时 | 吞吐 | VRAM |
|------|------|------|--------|----------|------|------|
| Ollama (baseline) | qwen2.5:3b (4B Q4_K_M) | CPU | 100% (8/8) | 89.9s | 2.7 tok/s | 0 MB |
| **Ollama GPU** | **qwen2.5:3b (4B Q4_K_M)** | **GPU (Vulkan)** | **100% (8/8)** | **20.8s** | **12.0 tok/s** | **1840 MB** |

启用方法：设置 `OLLAMA_IGPU_ENABLE=1` 环境变量后重启 Ollama，Vulkan 后端自动识别 Intel Arc iGPU（9.0 GiB 总显存）。生成速度 4.4x 加速，推理延迟 4.3x 加速，精度保持 100%。

## 五场景 E2E 验证

| 场景 | 音频时长 | 评分 | 违规/确认 | ASR RTF | 复核耗时 |
|------|----------|------|-----------|---------|----------|
| 金融违规 | 56.4s | 0 (不合格) | 21/12 | 0.037 | 29.76s |
| 电销违规 | 53.3s | 0 (不合格) | 25/20 | 0.031 | 28.5s |
| 合规对照 | 79.3s | 100 (优秀) | 0/0 | 0.025 | 18.4s |
| **保险违规** | 65.2s | 0 (不合格) | 18/11 | 0.048 | 49.29s |
| **合规保险** | 80.3s | 100 (优秀) | 0/0 | 0.084 | 21.78s |

## 四平台完整模式集成测试

| 平台 | 接入模式 | 调用次数 | 成功率 | 平均耗时 | 平均评分 | 违规/确认 | 复核耗时 |
|------|----------|----------|--------|----------|----------|-----------|----------|
| QwenWork | 原生 Skill | - | - | - | - | - | - |
| WorkBuddy | HTTP API | 2 | 100% | 32.27s | 82.5 | 16.0/2.5 | 13.74s |
| TraeWork | HTTP API | 2 | 100% | 45.8s | 82.5 | 16.0/2.5 | 20.08s |
| 豆包办公 | HTTP API | 2 | 100% | 45.25s | 75.0 | 16.0/3.0 | 13.2s |

完整模式含 ASR(OpenVINO NPU/GPU) + 规则引擎 + 本地 LLM 语义复核(Qwen3-1.7B INT4 GPU)。6/6 调用 100% 成功。

## Token 经济性

| 方案 | API Token | 本地计算 Token | 音频外发 |
|------|-----------|---------------|----------|
| 纯云端大模型 | 450 | 0 | 是 |
| VoiceGuard | 0 | 3861 | 否 |

**API token 节省 100%**：规则引擎零 token 全量初筛，本地模型仅复核候选片段，无需调用任何外部 API。

## 测试覆盖

94 个单元测试，覆盖核心模块（rules_engine / reviewer / pipeline / report_gen / transcribe），全部通过，耗时 0.8s。

## 技术栈

- **ASR**：funasr + SenseVoiceSmall + OpenVINO IR（NPU Encoder+CTC 双组件 / GPU / CPU 三后端）
- **LLM**：Qwen3-1.7B INT4 + OpenVINO GenAI（GPU 推理）
- **规则引擎**：自研 DSL（keyword/regex/absent/semantic 四策略）
- **HTTP API**：FastAPI + uvicorn
- **异构调度**：NPU(ASR Encoder+CTC 双组件) + GPU(LLM, OpenVINO/Ollama Vulkan) + CPU(规则引擎) 三级协同
- **硬件**：Intel Core Ultra 5 125H（CPU + Arc GPU + NPU Intel AI Boost）

---

**VoiceGuard · 40 Rules | 5 Scenarios | 4 Platforms | 100% Privacy**

*Intel Agentic PC Skill Grand Finals 2026 · Suzhou*
