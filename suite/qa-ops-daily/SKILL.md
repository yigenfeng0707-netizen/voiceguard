---
name: qa-ops-daily
description: |
  质检运营日报技能（VoiceGuard 专家套件的上层消费者）。批量扫描当日通话录音目录，
  逐段调用 voiceguard-qa skill 完成端侧合规质检（Skill 级互调：subprocess 调用其 CLI 入口
  或 HTTP API），汇总生成质检运营日报——平均分、合格率、违规 TOP 规则、调度漏斗合计、
  token 节省合计，并输出完整调用链日志（call_chain）作为 Skill 互调证据。
  Use this skill when the user, in Chinese or English, asks to run batch QA on multiple call
  recordings, generate a daily/weekly QA operations report, or aggregate VoiceGuard results.
  Trigger on phrases like 质检日报 / 批量质检 / 运营日报 / 今日质检汇总 / batch QA /
  daily QA report / aggregate quality inspection.
---

# qa-ops-daily · 质检运营日报技能（VoiceGuard 上层技能）

## 定位

本技能是 VoiceGuard 专家套件的**上层消费者**：它自身不做质检，而是编排
`voiceguard-qa`（主编排技能）对一批录音逐一执行端侧合规质检，再汇总成运营视角的日报。

```
qa-ops-daily（本技能 · 上层编排）
  └── 调用 → voiceguard-qa（Skill 级互调：CLI / HTTP）
        ├── vg-transcribe     （端侧 ASR 转写，NPU/GPU/CPU）
        ├── vg-rules-check    （规则引擎零 token 初筛）
        └── vg-report-gen     （结构化质检报告）
```

## 用法

```
python run.py [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--audio-dir` | 录音目录 | `../../demo/samples`（voiceguard/demo/samples） |
| `--pattern` | 文件名 glob | `demo_*.wav` |
| `--full` | 完整模式（含本地模型语义复核） | 关闭（快速模式：ASR+规则） |
| `--via` | 调用方式：`cli` / `http` | `cli` |
| `--api-base` | `--via http` 时的服务地址 | `http://127.0.0.1:8765` |
| `--device` | ASR 推理设备 | `auto` |
| `--packs` | 规则包 | `all` |
| `--output-dir` | 日报输出目录 | `../../output/qa_ops_daily`（voiceguard/output/qa_ops_daily） |

### 示例

| 意图 | 命令 |
|------|------|
| 当日全量录音快速质检日报 | `python run.py` |
| 完整模式（含语义复核） | `python run.py --full` |
| 指定目录 + HTTP API 调用 | `python run.py --audio-dir D:\calls --via http` |

## 输出物

| 文件 | 内容 |
|------|------|
| `qa_ops_daily_report.html` | 运营日报（汇总卡片 + 逐通明细 + TOP 违规规则 + 调用链） |
| `qa_ops_daily_report.md` | 同内容 Markdown 版 |
| `qa_ops_daily_report.json` | 机器可读汇总（供再上层技能消费） |
| `call_chain.md` | **Skill 互调证据**：每次调用 voiceguard-qa 的完整命令、返回码、耗时 |

## 依赖

- voiceguard-qa skill（主仓库 `voiceguard/`，含可运行环境）
- 纯 Python 标准库（编排层零额外依赖）
