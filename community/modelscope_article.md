# VoiceGuard：基于 OpenVINO 异构加速的端侧语音合规质检助手

> 把整段通话录音留在本机，用「规则引擎（零 token）→ 本地小模型语义复核」的分级调度完成合规质检——敏感语音零外发，质检报告可溯源。

**作者**：根深叶茂  
**平台**：ModelScope 开发者实践  
**赛事**：英特尔 Agentic PC Skill 大赛全国总决赛（2026-09-22~23 · 苏州）

---

## 一、问题背景

金融电销行业的通话质检长期面临三重困境：

1. **隐私风险**：传统云端质检方案需要将通话录音上传至外部服务器，违反《个人信息保护法》对敏感个人信息「本地处理」的要求；
2. **效率瓶颈**：人工抽检覆盖率通常不足 5%，大量违规话术漏网，且人工成本高昂；
3. **精度不足**：纯关键词匹配虽零 token 但误报率高；全量送云端大模型精度好但成本高且泄露数据。

VoiceGuard 的核心思路：**用规则引擎做零 token 全量初筛，仅把候选片段（约 15%）送入本地小模型语义复核**，在保证精度的同时实现零外发、零 API token。

---

## 二、架构设计

### 2.1 分级调度流水线

```
录音 → [预处理/VAD] → [ASR 转写 · OpenVINO 异构] → [规则引擎 · 0 token]
     → [候选片段 → 本地小模型复核 · GPU INT4] → [结构化质检报告]
     → (可选) [云端脱敏摘要 → 辅导建议 · 弱网自动跳过]
```

| 阶段 | 组件 | 功能 | 设备 |
|------|------|------|------|
| Stage 1 | ASR + 规则引擎 | 转写 + 零 token 全量初筛 | **NPU**（4.17x 加速） |
| Stage 2 | Qwen3-1.7B INT4 / Ollama qwen2.5:3b | 候选片段语义复核 | GPU（batch=10, 1.9x 加速 / Vulkan 4.4x） |
| Stage 3 | 报告生成器 | 评分 + 违规清单 + 漏斗 | CPU |
| Stage 4 | 云增强（可选） | 脱敏 → 云端 API 辅导建议 → 弱网降级 | CPU + 云端 |

### 2.1.1 异构调度实测数据

| 组件 | 设备 | 耗时 | RTF/加速比 |
|------|------|------|-----------|
| ASR Encoder (234M) | CPU PyTorch | 7.623s | RTF=0.135 (基线) |
| ASR Encoder + CTC | **NPU (AI Boost)** | **1.775s** | **RTF=0.030, 4.17x 加速** |
| ASR Encoder | GPU (Arc) | 2.083s | RTF=0.037, 3.7x 加速 |
| LLM 复核 (29 候选) | GPU batch=1 | 57.0s | 基线 |
| LLM 复核 (29 候选) | **GPU batch=10** | **29.76s** | **1.9x 加速** |
| LLM 复核 (备选) | Ollama CPU | 89.9s | 2.7 tok/s |
| LLM 复核 (备选) | **Ollama GPU Vulkan** | **20.8s** | **12.0 tok/s, 4.4x 加速** |

NPU 双组件完整 ASR 加速策略：

- **Encoder（234M 参数）**：SANM 局部注意力（kernel=11）→ 100 帧 chunk + 11 帧 overlap，stride=78，静态 shape (1,100,560)，NPU 编译 3.82s，单 chunk 推理 129.93ms
- **CTC Head（12.8M 参数）**：Linear(512→25055) 固定投影，静态 shape (1,100,512)，**FP32 精度**（FP16 对 25055 维投影 max diff 14.1 不可用），NPU 编译 0.22s，单次推理 8.86ms

这是完整的 ASR on NPU 方案——Encoder + CTC decoder head 双组件均在 NPU 上运行，无需回退到 CPU 做 CTC 解码。

### 2.2 规则引擎

规则引擎支持三种匹配策略，可按行业插拔：

- **keyword**：关键词精确匹配（如「保本保息」「验证码」）
- **regex**：正则模式匹配（如「年化收益\d+」）
- **absent**：必备表述缺失检测（如通话开始未表明机构名称）

当前规则库共 40 条，覆盖金融和电销两大场景：

| 规则包 | 条数 | 典型规则 |
|--------|------|----------|
| `finance.json` | 20 | 承诺保本保收益、夸大收益宣传、类存款误导、索要验证码、诱导借贷投资 |
| `telesales.json` | 20 | 冒充机构、打压客户判断、饥饿营销、未告知录音、结束语不规范 |

规则引擎执行耗时 **3ms**，零 token 消耗，标记约 62% 的片段为候选。

### 2.3 语义复核模型

复核阶段使用 **Qwen3-1.7B INT4 量化版**，通过 OpenVINO 转换后在 Intel GPU 上运行：

- 模型大小：约 1GB（INT4 量化）
- 推理速度：约 10 tok/s
- 输入：规则引擎标记的候选片段 + 对应规则
- 输出：JSON 格式判定（confirmed/rejected）+ 判定理由（引用具体法规）

关键技术细节：
- 使用 `/no_think` 指令跳过思考链，减少 token 消耗
- `batch_size=10`（原 batch_size=1），将所有候配合并为一次推理，减少 N-1 次 prompt 处理开销，从 57s 降至 29.76s（**1.9x 加速**）
- `max_new_tokens=min(500, batch*50+20)` 动态上限防止截断
- 安全构造思考标签：`chr(60) + "/think" + chr(62)` 避免特殊字符注入

---

## 三、五场景验证结果

### 3.1 金融违规场景

| 指标 | 值 |
|------|-----|
| 质检总分 | 0 / 100（不合格） |
| 红线确认 | 12 条 |
| 警告确认 | 5 条 |
| 语义复核确认 | 17 条 |
| 候选片段占比 | 62.5% |
| API token 消耗 | 0（100% 节省） |
| ASR 耗时 | 2.08s（GPU, RTF=0.037） |
| 语义复核耗时 | 29.76s（batch=10, 1.9x 加速） |

典型违规话术：
- 「这款产品保本保息，年化收益30%起步，稳赚不赔」
- 「当然跟存款一样安全，当存款买就行」
- 「把验证码告诉我一下，我马上帮您开通」
- 「您还可以先贷款来买，收益覆盖利息稳赚」

### 3.2 电销违规场景

| 指标 | 值 |
|------|-----|
| 质检总分 | 0 / 100（不合格） |
| 红线确认 | 8 条 |
| 警告确认 | 12 条 |
| 语义复核确认 | 20 条 |
| API token 消耗 | 0（100% 节省） |

### 3.3 合规对照场景

| 指标 | 值 |
|------|-----|
| 质检总分 | 100 / 100（优秀） |
| 违规数 | 0 |
| 语义复核耗时 | 18.4s |
| API token 消耗 | 0（100% 节省） |

### 3.4 保险违规场景（新增）

| 指标 | 值 |
|------|-----|
| 音频时长 | 65.2s（含背景噪声、双角色对话） |
| 质检总分 | 0 / 100（不合格） |
| 违规数 | 18 |
| 语义复核确认 | 11 条 |
| ASR RTF | 0.048 |
| 复核耗时 | 49.29s |

触发的典型规则：承诺保本保收益（FIN-001）、类存款误导表述（FIN-003）、暗示性收益承诺（FIN-013）、诱导借贷/加杠杆投资（FIN-014）、索要敏感验证信息（FIN-012）、隐瞒费用结构（FIN-015）。

### 3.5 合规保险销售场景（新增）

| 指标 | 值 |
|------|-----|
| 音频时长 | 80.3s（含背景噪声、双角色对话） |
| 质检总分 | 100 / 100（优秀） |
| 违规数 | 0 |
| ASR RTF | 0.084 |
| 复核耗时 | 21.78s |

合规话术示例：明确告知产品为消费型保险非储蓄非投资型、无保本保收益承诺、完整费率披露、风险测评独立完成。

---

## 四、OpenVINO 异构加速经验

### 4.1 三级降级链

VoiceGuard 支持 NPU > GPU > CPU 三级自动降级：

```python
# 优先级：NPU > GPU > CPU
device = detect_best_device()  # 自动探测可用后端
```

- **NPU**：适合 ASR 推理（SenseVoiceSmall），低功耗持续运行
- **GPU**：适合 LLM 语义复核（Qwen3-1.7B INT4），Intel Arc GPU 通过 OpenVINO 调用
- **CPU**：兜底方案，RTF 0.25-0.31（ASR），LLM 约 3 tok/s

### 4.2 NPU 完整 ASR 加速（Encoder + CTC 双组件，已实测）

将 SenseVoiceSmall 的两个子模块均转为 OpenVINO IR 部署到 NPU，实现完整 ASR on NPU：

**Encoder（234M 参数）**：用 `openvino.convert_model()` 转为 IR，静态 shape + 分块推理：

```python
# NPU 分块推理策略
# 1. 固定 shape (1, 100, 560) → NPU 可编译
# 2. 11 帧 overlap（SANM 局部注意力 kernel=11）
# 3. stride=78，变长音频拆分为多个 chunk
core = openvino.Core()
encoder = core.compile_model("encoder.xml", "NPU")  # 编译 3.82s
```

**CTC Head（12.8M 参数）**：将 `ctc_lo` Linear(512→25055) 转为独立 IR，静态 shape (1,100,512)：

```python
# CTC decoder head → NPU (FP32, 必须非量化)
# FP16 对 25055 维投影 max diff 14.1，不可用
ctc_head = core.compile_model("ctc_head_fp32.xml", "NPU")  # 编译 0.22s
```

NPU 组件分解（56.4s 音频）：

| 组件 | 参数量 | NPU 编译 | NPU 推理 | 调用次数 | 总推理 |
|------|--------|----------|----------|----------|--------|
| Encoder | 234M | 3.82s | 129.93ms/chunk | 13 | 1.69s |
| CTC Head | 12.8M | 0.22s | 8.86ms/call | 10 | 0.09s |
| **合计** | **246.8M** | **4.04s** | - | - | **1.775s** |

**实测结果**（56.4s 音频）：
- NPU (Encoder+CTC): forward=1.775s, RTF=0.030, **4.17x 加速**
- GPU: forward=2.083s, RTF=0.037, 3.7x 加速
- CPU: forward=7.623s, RTF=0.135, 基线

### 4.3 Ollama GPU 加速（Vulkan / Intel Arc）

除 OpenVINO 原生路线外，VoiceGuard 还支持通过 Ollama + Vulkan 后端使用 Intel Arc iGPU 加速 LLM 推理：

```python
# reviewer.py 工厂函数自动选择后端
from reviewer import create_reviewer
reviewer = create_reviewer(backend="auto")  # auto → openvino → ollama 降级
```

| 配置 | 设备 | 准确率 | 推理耗时 | 吞吐 | VRAM |
|------|------|--------|----------|------|------|
| Ollama (baseline) | CPU | 100% (8/8) | 89.9s | 2.7 tok/s | 0 MB |
| **Ollama GPU** | **GPU (Vulkan)** | **100% (8/8)** | **20.8s** | **12.0 tok/s** | **1840 MB** |

启用方法：设置 `OLLAMA_IGPU_ENABLE=1` 环境变量后重启 Ollama，Vulkan 后端自动识别 Intel Arc iGPU（9.0 GiB 总显存）。

### 4.4 NPU 限制

实测发现：
- NPU 无法运行 LLM（Qwen3-1.7B），120s 超时——NPU 算力不足以跑大 transformer 模型
- NPU 仅适合 ASR encoder 类小模型（234M 参数），固定 shape 小张量计算
- 大于 100 帧的静态 shape 编译超时（300/500/1000/2000 帧均失败）
- 最终架构：**完整 ASR on NPU → LLM on GPU(or Ollama GPU Vulkan) → Rules on CPU → 可选云端增强**

---

## 五、云增强与 Skill 可组合性

### 5.1 云端增强骨架（Stage 4，可选）

VoiceGuard 在 Stage 3 报告生成后，支持可选的 Stage 4 云端增强：

```
质检报告 → [PII 脱敏] → [云端 LLM API 辅导建议] → [弱网自动降级跳过] → 增强报告
```

- **PII 脱敏**：6 种正则覆盖手机号/身份证/银行卡/邮箱/姓名/地址，脱敏后文本仅含 `[PHONE]`/`[ID]` 等占位符
- **云端 API 调用**：占位接口，可对接千帆/ChatGPT 等云端大模型获取辅导建议
- **弱网降级**：网络超时或 API 不可达时自动跳过，不影响本地质检报告

```python
from cloud_enhance import desensitize, run_cloud_enhance

# 脱敏示例
masked = desensitize("客户王经理的手机号是13812345678，身份证320102199001011234")
# → "客户[NAME]的手机号是[PHONE]，身份证[ID]"

# 云端增强（自动降级）
result = run_cloud_enhance(violations, transcript, api_key="...")
# 弱网时 result.skipped = True，不影响主流程
```

### 5.2 Skill 可组合性

VoiceGuard 作为 Agentic PC Skill 设计为可组合的三个子技能，供 Agent 平台按需调用：

| 子技能 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `vg-transcribe` | 语音转写 | 音频文件路径 | 带时间戳的文本片段 |
| `vg-rules-check` | 规则引擎初筛 | 转写片段 + 规则包 | 违规命中列表 |
| `vg-report-gen` | 质检报告生成 | 规则命中 + 复核结论 | HTML/JSON/Markdown 报告 |

同时提供 `qa_ops_daily` 批量质检运营脚本，支持目录扫描、批量处理、生成运营日报：

```bash
# 批量质检 + 运营日报
python scripts/qa_ops_daily.py --audio-dir ./recordings --output ./daily_report.html
```

---

## 六、四平台集成

VoiceGuard 提供两种集成模式：

| 平台 | 模式 | 优先级 | 状态 |
|------|------|--------|------|
| QwenWork | 原生 Skill | P0 | SKILL.md 就绪 |
| WorkBuddy | HTTP API | P1 | 100% 成功 |
| TraeWork | HTTP API | P1 | 100% 成功 |
| 豆包办公 | HTTP + 文件监听 | P2 | 100% 成功 |

HTTP API 通过 FastAPI 实现，暴露三个端点：

- `POST /v1/qa`：完整质检（ASR → 规则 → 复核 → 报告）
- `POST /v1/qa/fast`：快速筛查（仅规则引擎，无需模型）
- `GET /health`：健康检查

集成验证结果（完整模式，含模型复核）：

| 平台 | 成功率 | 平均耗时 | 平均评分 | 违规/确认 |
|------|--------|----------|----------|-----------|
| WorkBuddy | 100% | 32.27s | 82.5 | 16/2.5 |
| TraeWork | 100% | 45.8s | 82.5 | 16/2.5 |
| 豆包 | 100% | 45.25s | 75.0 | 16/3.0 |

目标成功率 >=95%，三平台均超标通过。完整模式含 ASR(OpenVINO NPU/GPU) + 规则引擎 + 本地 LLM 语义复核(Qwen3-1.7B INT4 GPU)。

---

## 七、Token 经济性分析

| 方案 | API Token | 本地计算 Token | 音频外发 |
|------|-----------|---------------|----------|
| 纯云端大模型 | 450 | 0 | 是 |
| VoiceGuard | 0 | 3861 | 否 |

**API token 节省 100%**：规则引擎零 token 全量初筛，本地模型仅复核候选片段，无需调用任何外部 API。

---

## 八、测试覆盖

项目包含 94 个单元测试，覆盖核心模块：

| 模块 | 测试数 | 覆盖点 |
|------|--------|--------|
| rules_engine | 规则命中/缺失检测 | keyword/regex/absent 三策略 |
| reviewer._parse_verdicts | JSON 解析/思考标签剥离 | 边界情况 |
| pipeline.run_stage1 | ASR + 规则链路 | 端到端 |
| report_gen | 报告生成 | HTML/JSON/Markdown |
| transcribe | 转写 | 模型加载/输出 |

全部通过，耗时 0.8s。

---

## 九、项目结构

```
voiceguard/
  scripts/             流水线脚本（32 个 Python 文件）
    run.py             唯一入口
    pipeline.py        分级调度编排（Stage 1-4，含云增强）
    transcribe.py      ASR 转写（支持 NPU/GPU/CPU 三后端）
    npu_asr.py         NPU ASR encoder 加速（分块推理）
    npu_ctc.py         NPU ASR CTC decoder head 加速
    convert_ctc_npu.py CTC head OpenVINO IR 转换
    rules_engine.py    规则引擎
    reviewer.py        语义复核（batch=10 + Ollama 工厂模式）
    cloud_enhance.py   云增强（PII 脱敏 + 云端 API + 弱网降级）
    qa_ops_daily.py    批量质检 + 运营日报
    report_gen.py      报告生成（含云增强字段渲染）
    server.py          FastAPI HTTP 服务
    subskills/         3 个可组合子技能
      vg-transcribe/SKILL.md
      vg-rules-check/SKILL.md
      vg-report-gen/SKILL.md
  rules/               行业规则包（finance.json + telesales.json）
  demo/                5 段自制样例音频 + 预置报告
  tests/               94 个单元测试
  agent_integrations/  四平台集成配置
  output/              质检报告 + 集成报告 + benchmark 数据
  docs/                架构文档
  community/           本文章
```

---

## 十、总结与展望

VoiceGuard 实现了「**敏感语音零外发、质检报告可溯源**」的核心目标：

- 40 条规则 + 3ms 零 token 初筛 + 62% 候选过滤
- Qwen3-1.7B INT4 本地复核 / Ollama GPU Vulkan 备选，100% API token 节省
- **完整 ASR on NPU**（Encoder 234M + CTC 12.8M 双组件，4.17x 加速）
- GPU LLM batch 1.9x 加速 / Ollama Vulkan 4.4x 加速（异构三级协同）
- 5 场景 E2E 验证 + 94 单元测试 + 4 平台完整模式集成测试（100% 成功）
- 云增强骨架（PII 脱敏 + 弱网降级）+ Skill 可组合性（3 子技能 + 批量运营脚本）
- OpenVINO 异构加速（NPU > GPU > CPU），已实测验证

下一步计划：
1. 扩展规则库至 100+ 条，覆盖证券、信托场景
2. 云增强灰度发布（对接千帆大模型 API）
3. 接入实时流式质检（WebSocket 推送）
4. ASR encoder 完全静态化以支持更长音频

---

**VoiceGuard · 40 Rules | 5 Scenarios | 4 Platforms | 100% Privacy | NPU 4.17x | Ollama GPU 4.4x**

*Intel Agentic PC Skill Grand Finals 2026 · Suzhou*
