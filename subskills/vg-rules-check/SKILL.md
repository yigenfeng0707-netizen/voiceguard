---
name: vg-rules-check
description: |
  VoiceGuard 规则引擎子技能。对 ASR 转写的逐字稿执行零 token 规则初筛，
  输出违规命中清单（rule_id / level / excerpt / matched_by）。
  自研 DSL 支持 keyword / regex / absent / semantic 四种匹配策略，
  金融 20 条 + 电销 20 条规则，可插拔扩展。3ms 完成全量初筛，100% token 节省。
  可被 vg-report-gen 或上层技能（如 qa-ops-daily）调用。
  Use when the user needs to check call transcripts for compliance violations.
---

# vg-rules-check · 规则引擎零 token 初筛子技能

## 功能

对逐字稿执行规则引擎初筛，3ms 内完成全量扫描，零 API token 消耗。
只有规则命中的候选片段才会进入后续 LLM 语义复核（通常仅 ~15% 片段）。

## 规则策略

| 策略 | 说明 | 示例 |
|------|------|------|
| keyword | 关键词精确匹配 | "保本保息"、"稳赚不赔" |
| regex | 正则表达式匹配 | 身份证号、银行卡号模式 |
| absent | 必备语缺失检测 | 风险提示语未出现 |
| semantic | 语义模式匹配 | 承诺收益、误导性比较 |

## 规则包

| 包 | 规则数 | 覆盖场景 |
|----|--------|----------|
| finance | 20 | 保本保息/夸大收益/类存款误导/索要验证码/冒充机构/泄露信息 |
| telesales | 20 | 虚假承诺/诱导消费/隐瞒关键信息/不当催收 |

规则文件: `rules/finance.json`, `rules/telesales.json`（JSON 格式，可插拔扩展）

## 用法

### 独立 CLI 调用（推荐，Skill 级入口）

```
# 模式 A：消费 vg-transcribe 的输出（上游 skill 已跑过）
python subskills/vg-rules-check/run.py --segments seg.json [--packs finance]

# 模式 B：直接传音频 —— 自动以子进程调用 vg-transcribe skill 转写（Skill 互调实证）
python subskills/vg-rules-check/run.py --audio call.wav [--device auto]

# 模式 C：纯文本快速自测（免 ASR）
python subskills/vg-rules-check/run.py --text "这款产品保本保息"
```

输出 hits JSON 契约（`skill: vg-rules-check`，含 `called_skills` 调用链字段与透传
`segments`），可直接被 `vg-report-gen --hits` 消费。

### Python API

```python
from rules_engine import load_rule_packs, run_engine
from rules_engine import Segment

# 构造 segments（来自 vg-transcribe 输出）
segments = [
    Segment(segment_id=0, text="这款产品保本保息，年化收益30%", start_ms=0, end_ms=3000)
]

# 加载规则包
packs = load_rule_packs("rules/", pack_ids=["finance"])

# 执行初筛
result = run_engine(segments, packs)
# result.hits = [RuleHit(rule_id, rule_name, level, excerpt, matched_by, ...)]
# result.funnel = {"total_segments": 1, "segments_flagged": 1, "redline_hits": 1, ...}
# result.elapsed_ms = 3
```

## 输出

```json
{
  "hits": [
    {
      "rule_id": "FIN-001",
      "rule_name": "保本保息承诺",
      "level": "redline",
      "category": "收益承诺",
      "excerpt": "保本保息，年化收益30%",
      "matched_by": "keyword:保本保息",
      "advice": "不得向客户承诺保本保息",
      "basis": "《理财公司理财产品销售管理办法》第28条"
    }
  ],
  "funnel": {
    "total_segments": 8,
    "segments_flagged": 3,
    "redline_hits": 2,
    "warning_hits": 1,
    "semantic_tasks": 1
  },
  "elapsed_ms": 3
}
```

## 依赖

- 无外部依赖（纯 Python，re 标准库）

## 被调用关系

```
qa-ops-daily (上层技能)
  ├── vg-transcribe → 输出 segments
  ├── vg-rules-check (本技能) → 输入 segments → 输出 hits
  └── vg-report-gen → 输入 hits → 输出 QAReport
```
