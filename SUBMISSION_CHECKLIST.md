# VoiceGuard 赛事提交清单

> 英特尔 Agentic PC Skill 大赛全国总决赛 (2026-09-22~23 · 苏州)

## 已完成项

| 编号 | 项目 | 状态 | 文件 |
|------|------|------|------|
| D1-D6 | 核心开发 (ASR/规则/LLM/报告/API) | ✅ | 34 scripts, 94 tests |
| P1-1 | Skill 可组合性 (3 子技能 + qa_ops_daily) | ✅ | subskills/, scripts/qa_ops_daily.py |
| P1-2 | 云增强骨架 (PII 脱敏 + 弱网降级) | ✅ | scripts/cloud_enhance.py |
| P1-3 | NPU 完整 ASR (Encoder + CTC 双组件) | ✅ | scripts/npu_ctc.py, models/SenseVoiceSmall_ov/ |
| P1-4 | Ollama GPU Vulkan 加速 | ✅ | reviewer.py create_reviewer() 工厂模式 |
| P2-1 | 路演 PPT | ✅ | 路演/VoiceGuard路演.pptx (10页) |
| P2-2 | Demo 视频 v3 (完整闭环展示) | ✅ | output/voiceguard_demo_v3.mp4 (252s, 1080p) |
| P2-3 | ModelScope 文章 (含视频) | ✅ | modelscope.cn/learn/436280 |
| P2-4 | 魔搭 Skills Center 发布 | ✅ | modelscope.cn/skills/gsym236998/voiceguard |
| P2-5 | 提交包 v5 | ✅ | voiceguard_submission_v5.zip |
| D2-1 | 规则库扩充至 72 条 (3 行业包) | ✅ | rules/finance.json(27) + insurance.json(18) + telesales.json(27) |
| D2-2 | 集成验证深度补强 (39/39) | ✅ | output/integration_report_comprehensive.html |
| D2-3 | E2E 边界场景鲁棒性 (14/14) | ✅ | output/e2e_edge_report.html, scripts/e2e_edge_cases.py |
| D2-4 | Demo v3 完整闭环视频 | ✅ | output/voiceguard_demo_v3.mp4 (252s, 11场景) |

## 关键技术数据

| 指标 | 数值 |
|------|------|
| 规则引擎 | 72 条规则, 3ms, 0 tokens, 85% 候选过滤 |
| NPU ASR | 4.17x 加速 (Encoder 1.689s + CTC 0.086s = 1.775s vs CPU 7.401s) |
| GPU LLM | OpenVINO INT4 batch=10 1.9x / Ollama Vulkan 4.4x (12.0 tok/s) |
| Token 节省 | 100% (API 零 token, 规则引擎全量初筛) |
| 单元测试 | 94/94 全通过 |
| 集成测试 | 39/39 调用 100% 成功率 (4 平台) |
| E2E 边界场景 | 14/14 验证项全通过 (5 类边界场景) |

## 估分预测

| 状态 | 估分 | 说明 |
|------|------|------|
| 当前 (v5 全量交付) | 80-82 | 规则库 72 条 + 集成 39/39 + E2E 14/14 + Demo v3 |
| + 路演发挥 | 85-87 | PPT + Q&A + 人气投票 |

## 待完成项 (时间依赖)

| 项目 | 截止 | 状态 |
|------|------|------|
| 路演名单公布 | 9/14 | 等待 |
| PPT 终版 | 9/18 | 视名单微调 |
| 现场路演 | 9/22-23 | 苏州国际博览中心 |
