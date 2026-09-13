---
name: vg-report-gen
description: |
  VoiceGuard 报告生成子技能。将规则引擎命中 + LLM 语义复核结论合并为
  结构化质检报告（评分/违规清单/调度漏斗/token 经济性/整改建议）。
  支持 Markdown / JSON / HTML 三种输出格式，HTML 含 CSS 漏斗可视化柱状图。
  可被上层技能（如 qa-ops-daily）调用。
  Use when the user needs to generate QA compliance reports from rule hits.
---

# vg-report-gen · 质检报告生成子技能

## 功能

合并规则引擎命中结果和 LLM 语义复核结论，生成结构化质检报告。
包含评分、违规清单、调度漏斗、token 经济性分析和整改建议。

## 评分算法

- 基准 100 分
- 红线确认违规：每条 -15
- 警告确认违规：每条 -5
- 提示确认违规：每条 -2
- 下限 0 分
- 等级：>=90 优秀 / 75-89 合格 / 60-74 待整改 / <60 不合格

## 用法

### 独立 CLI 调用（推荐，Skill 级入口）

```
python subskills/vg-report-gen/run.py --hits hits.json [--format html|markdown|json] [--output report.html]
```

输入 `hits.json` 为 `vg-rules-check` 的输出契约；可选 `--verdicts verdicts.json`
（语义复核结论）生成完整模式报告。

### Python API

```python
from report_gen import build_report, render_html, render_markdown, render_json

# stage1: 来自 vg-rules-check 的结果
# verdicts: 来自 LLM 语义复核的结论（可为 None = 仅规则引擎）
report = build_report(
    stage1,
    verdicts=verdicts,
    reviewer_backend="OpenVINO GPU INT4 (Qwen3-1.7B)",
    asr_backend="funasr+OpenVINO(NPU, Encoder+CTC 双NPU组件)",
    review_prompt_tokens=1200,
    review_s=29.76,
)

# 渲染
html = render_html(report)    # 含 CSS 漏斗柱状图 + 摘要卡片
md = render_markdown(report)  # Markdown 表格
json_str = render_json(report)  # JSON 序列化
```

## 输出格式

### HTML
- 摘要卡片：总分 + 红线/警告/提示确认数
- 漏斗柱状图：逐字稿片段 → 规则标记 → 红线/警告/提示 → 语义复核
- 违规清单表格（按级别着色）
- Token 经济性对比（云端估算 vs VoiceGuard 实际）
- 性能指标表（ASR/引擎/复核耗时）
- 整改建议列表
- 云端辅导建议（当 --cloud-enhance 启用时）

### Markdown
- 同 HTML 内容的 Markdown 表格版本

### JSON
- 完整 QAReport dataclass 序列化（含所有字段）

## QAReport 数据结构

```python
@dataclass
class QAReport:
    audio: str                  # 文件名
    duration_ms: int            # 时长
    lang: str                   # 语言
    score: int                  # 质检总分
    grade: str                  # 等级
    violations: List[Violation] # 违规清单
    funnel: dict                # 调度漏斗
    token_economics: dict       # token 经济性
    backends: dict              # ASR/复核后端标签
    timings: dict               # 各阶段耗时
    cloud_advice: str           # 云端辅导建议（可选）
    cloud_enhance_meta: dict    # 云增强元数据
    generated_at: str           # 生成时间
```

## 依赖

- 纯 Python 标准库（json, time, dataclasses, re）

## 被调用关系

```
qa-ops-daily (上层技能)
  ├── vg-transcribe → 输出 segments
  ├── vg-rules-check → 输出 hits
  └── vg-report-gen (本技能) → 输入 hits → 输出 QAReport
```
