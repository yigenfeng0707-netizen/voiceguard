# 四平台生态接入备忘（D4 作战用）

决赛要求：Agent Skills 无缝嵌入主流 Agent 平台，现场需展示 Demo。
覆盖策略：QwenWork（P0 主战场）+ WorkBuddy + TraeWork（P1 复用初赛模板）+ 豆包办公（P2 探测）。

## 1. QwenWork（主战场）

- 接入形态：**原生 Skill** —— 将 `voiceguard/` 打包为 QwenWork 技能目录（SKILL.md 已就绪）
- 安装方式：复制到 `~/.qwenworkcn/skills/voiceguard-qa/` 即被 QwenWork 技能系统发现
- 触发词验证：对 QwenWork 说"帮我质检这段通话录音"，应自动路由到本技能
- 演示脚本（录屏用）：
  1. 打开 QwenWork，发送"对 D:\demo\违规录音.wav 做合规质检"
  2. 技能自动触发 → 显示分级调度过程（规则引擎→本地模型复核）
  3. 输出质检报告（评分+违规清单+漏斗）
  4. 追问"第2条违规的原文证据是什么？" → evidence_span 溯源展示
- 注意：skill 内路径统一用相对路径 + 环境变量，模型目录用 `VOICEGUARD_MODEL_DIR`

## 2. WorkBuddy / 3. TraeWork（复用初赛模板）

- 初赛已有配置：`D:\APPs\OpenVINO\demo\local-meeting-minutes\agent_integrations\{workbuddy,trae_work}\config.json`
- 模式：HTTP API（FastAPI 服务）+ 触发词配置 + 调用脚本
- 需要：`scripts/server.py` 适配（从初赛复制 + 改路由为 /v1/qa）
- 验证：每平台 10 次调用，成功率 ≥95%，输出 integration_report.md

## 4. 豆包办公（探测）

- 先探测其技能/插件接入能力（开放平台文档）
- 有 API → 按 WorkBuddy 模式接入
- 无 API → 文件式调用（监听目录 + 自动质检）+ 文档如实说明
- 底线：满足"至少 2 个平台现场可演示"（QwenWork 必保）

## 集成验证报告模板（每平台）

| 字段 | 说明 |
|------|------|
| 平台 + 版本 | 如 QwenWork 桌面版 vX |
| 调用次数 | 10 |
| 成功率 | ≥95% 达标 |
| 平均耗时 | 端到端（音频进→报告出） |
| 错误码分布 | 失败项的 trace_id 与原因 |
| 配置快照 | config 文件路径 |
