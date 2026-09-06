#!/usr/bin/env python3
"""VoiceGuard Demo Video v3 - Complete closed-loop showcase.

10 scenes, Chinese narration, >=60s duration.
Shows: instruction -> decomposition -> call -> report full chain.
Includes: Rule Engine -> NPU ASR -> GPU LLM review + multi-scenario + evidence tracing.

Usage:
    python scripts/make_demo_v3.py --output output/voiceguard_demo_v3.mp4
"""

import asyncio
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

# ---- Configuration ----
W, H = 1920, 1080
FPS = 30
FONT_REGULAR = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
BG_COLOR = (15, 23, 42)
BG2_COLOR = (30, 41, 59)
ACCENT_BLUE = (59, 130, 246)
ACCENT_RED = (239, 68, 68)
ACCENT_GREEN = (34, 197, 94)
ACCENT_PURPLE = (147, 51, 234)
ACCENT_YELLOW = (234, 179, 8)
ACCENT_CYAN = (6, 182, 212)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (148, 163, 184)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
)
TMP_DIR = os.path.join(OUTPUT_DIR, "_demo_v3_tmp")
os.makedirs(TMP_DIR, exist_ok=True)


def font(size, bold=False):
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)


def draw_gradient_bg(img, draw):
    for y in range(H):
        ratio = y / H
        r = int(BG_COLOR[0] + (BG2_COLOR[0] - BG_COLOR[0]) * ratio)
        g = int(BG_COLOR[1] + (BG2_COLOR[1] - BG_COLOR[1]) * ratio)
        b = int(BG_COLOR[2] + (BG2_COLOR[2] - BG_COLOR[2]) * ratio)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


def draw_centered(draw, y, text, fnt, color=TEXT_WHITE):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w = bbox[2] - bbox[0]
    draw.text(((W - w) // 2, y), text, font=fnt, fill=color)


def draw_card(draw, x, y, w, h, label, value, value_color):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=BG2_COLOR)
    draw.rounded_rectangle([x, y, x + w, y + 36], radius=12, fill=value_color)
    draw.text((x + 16, y + 6), label, font=font(20), fill=TEXT_WHITE)
    bbox = draw.textbbox((0, 0), value, font=font(44, True))
    vw = bbox[2] - bbox[0]
    draw.text((x + (w - vw) // 2, y + 50), value, font=font(44, True), fill=value_color)


def draw_metric_bar(draw, x, y, w, h, color, label, value, fnt_label, fnt_value):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=color)
    draw.text((x + 20, y + 10), label, font=fnt_label, fill=TEXT_WHITE)
    draw.text((x + 20, y + 50), value, font=fnt_value, fill=TEXT_WHITE)


# ---- Scene renderers ----
def render_title(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 280, "VoiceGuard", font(80, True), ACCENT_BLUE)
    draw_centered(d, 380, "端侧语音合规质检助手", font(40), TEXT_WHITE)
    draw_centered(d, 440, "On-Device Voice Compliance QA", font(32), TEXT_GRAY)
    d.line([(W // 2 - 200, 510), (W // 2 + 200, 510)], fill=ACCENT_BLUE, width=4)
    draw_centered(
        d, 540, "Intel Agentic PC Skill Grand Finals 2026", font(28), ACCENT_BLUE
    )
    draw_centered(d, 580, "Intel Connection | 9.22-23 Suzhou", font(24), TEXT_GRAY)
    metrics = [
        ("72", "规则引擎", ACCENT_YELLOW),
        ("4.17x", "NPU 加速", ACCENT_GREEN),
        ("0", "API Token", ACCENT_PURPLE),
        ("39/39", "平台集成", ACCENT_CYAN),
    ]
    x_start = 320
    for i, (num, label, color) in enumerate(metrics):
        x = x_start + i * 340
        draw_metric_bar(
            d, x, 680, 300, 120, color, label, num, font(20), font(44, True)
        )
    img.save(path)


def render_problem(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 60, "痛点：电话合规质检的困境", font(48, True), TEXT_WHITE)
    items = [
        (ACCENT_RED, "云端质检泄露隐私", "录音上传云端 = 隐私泄露风险, 违反 PIPL"),
        (ACCENT_RED, "人工审核慢且贵", "人工逐条听审, 无法规模化, 成本高昂"),
        (ACCENT_YELLOW, "监管要求本地处理", "个人信息保护法、银保监要求数据主权"),
        (ACCENT_YELLOW, "纯关键词匹配误报高", "缺乏语义理解, 大量误报漏报"),
    ]
    y = 160
    for color, title, desc in items:
        d.rounded_rectangle([150, y, W - 150, y + 120], radius=10, fill=BG2_COLOR)
        d.rounded_rectangle([150, y, 170, y + 120], radius=10, fill=color)
        d.text((200, y + 18), title, font=font(34, True), fill=color)
        d.text((200, y + 65), desc, font=font(26), fill=TEXT_GRAY)
        y += 150
    draw_centered(
        d,
        860,
        "VoiceGuard: 全链路端侧处理, 零外发, 零 Token",
        font(30, True),
        ACCENT_GREEN,
    )
    img.save(path)


def render_architecture(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 50, "分级调度架构：完整闭环", font(48, True), ACCENT_BLUE)
    stages = [
        ("1", "音频输入", "wav/mp3/m4a", "坐席录音片段", ACCENT_BLUE),
        (
            "2",
            "ASR 转写",
            "SenseVoice + NPU",
            "Encoder 234M + CTC 12.8M\nNPU 4.17x 加速",
            ACCENT_GREEN,
        ),
        (
            "3",
            "规则引擎",
            "72 rules / 3 packs",
            "零 Token 全量初筛\n3ms, 覆盖金融+保险+电销",
            ACCENT_YELLOW,
        ),
        (
            "4",
            "LLM 复核",
            "Qwen3-1.7B INT4 GPU",
            "仅复核候选片段\nbatch 1.9x / Vulkan 4.4x",
            ACCENT_RED,
        ),
        (
            "5",
            "质检报告",
            "HTML + JSON",
            "评分 + 违规清单\nEvidence 溯源 + 漏斗统计",
            ACCENT_PURPLE,
        ),
    ]
    box_w, box_h = 300, 200
    gap = 30
    total_w = len(stages) * box_w + (len(stages) - 1) * gap
    x_start = (W - total_w) // 2
    box_y = 200
    for i, (num, title, device, desc, color) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        d.rounded_rectangle(
            [x, box_y, x + box_w, box_y + box_h], radius=12, fill=BG2_COLOR
        )
        d.rounded_rectangle([x, box_y, x + box_w, box_y + 40], radius=12, fill=color)
        d.text(
            (x + 16, box_y + 6),
            f"Stage {num}",
            font=font(22),
            fill=TEXT_WHITE,
        )
        d.text((x + 20, box_y + 55), title, font=font(30, True), fill=color)
        d.text(
            (x + 20, box_y + 105),
            device,
            font=font(22),
            fill=TEXT_GRAY,
        )
        for j, line in enumerate(desc.split("\n")):
            d.text(
                (x + 20, box_y + 135 + j * 28),
                line,
                font=font(22),
                fill=TEXT_WHITE,
            )
        if i < len(stages) - 1:
            ax = x + box_w
            bx = x + box_w + gap
            ay = box_y + box_h // 2
            d.line([(ax, ay), (bx, ay)], fill=TEXT_GRAY, width=3)
            d.polygon([(bx, ay), (bx - 10, ay - 7), (bx - 10, ay + 7)], fill=TEXT_GRAY)
    # Key points
    y = 480
    points = [
        "Stage 1-2: NPU 加速 ASR, 转写速度 4.17x 提升",
        "Stage 3: 规则引擎零 Token 初筛, 3ms, 识别 ~80% 候选片段",
        "Stage 4: GPU 本地模型仅复核候选, 省 Token 100%",
        "Stage 5: 结构化报告 + Evidence 溯源, 可追溯每条违规",
        "三级降级: NPU > GPU > CPU, 弱网自动 fallback",
    ]
    for pt in points:
        d.ellipse([180, y + 8, 195, y + 23], fill=ACCENT_BLUE)
        d.text((220, y), pt, font=font(26), fill=TEXT_WHITE)
        y += 55
    draw_centered(
        d,
        830,
        "录音 -> ASR(NPU) -> 规则(CPU) -> LLM(GPU) -> 报告",
        font(28),
        ACCENT_CYAN,
    )
    img.save(path)


def render_rule_engine(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(
        d, 50, "规则引擎：72 条规则, 3 个行业包", font(48, True), ACCENT_YELLOW
    )
    packs = [
        ("finance", "金融营销合规", "27 条", "19 红线 / 8 警告", ACCENT_RED),
        ("insurance", "保险销售合规", "18 条", "13 红线 / 5 警告", ACCENT_BLUE),
        (
            "telesales",
            "电销规范服务",
            "27 条",
            "13 红线 / 10 警告 / 4 提示",
            ACCENT_GREEN,
        ),
    ]
    for i, (pid, name, count, breakdown, color) in enumerate(packs):
        x = 80 + i * 590
        d.rounded_rectangle([x, 150, x + 560, 360], radius=12, fill=BG2_COLOR)
        d.rounded_rectangle([x, 150, x + 560, 190], radius=12, fill=color)
        d.text((x + 20, 158), name, font=font(28, True), fill=TEXT_WHITE)
        d.text((x + 20, 210), f"pack_id: {pid}", font=font(22), fill=TEXT_GRAY)
        d.text((x + 20, 250), count, font=font(44, True), fill=color)
        d.text((x + 200, 260), breakdown, font=font(24), fill=TEXT_WHITE)
        d.text(
            (x + 20, 320),
            "match_modes: keyword / regex / absent / semantic",
            font=font(20),
            fill=TEXT_GRAY,
        )
    # Performance
    draw_card(d, 80, 400, 260, 140, "规则总数", "72", ACCENT_YELLOW)
    draw_card(d, 360, 400, 260, 140, "引擎耗时", "5ms", ACCENT_GREEN)
    draw_card(d, 640, 400, 260, 140, "API Token", "0", ACCENT_PURPLE)
    draw_card(d, 920, 400, 260, 140, "行业包切换", "1 line", ACCENT_CYAN)
    # Evidence example
    d.text((80, 580), "Evidence 溯源示例:", font=font(30, True), fill=TEXT_WHITE)
    evidence = [
        ("FIN-001", "红线", "承诺保本保收益", '"这款产品保本保息, 年化8%, 稳赚不赔"'),
        ("INS-001", "红线", "将保险与存款混同", '"您就把这个当定期存款, 跟存银行一样"'),
        ("TEL-015", "红线", "冒充其他机构", '"我们是移动官方, 运营商通知"'),
    ]
    for i, (rid, level, name, excerpt) in enumerate(evidence):
        y = 640 + i * 65
        lvl_color = ACCENT_RED if level == "红线" else ACCENT_YELLOW
        d.rounded_rectangle([80, y, W - 80, y + 55], radius=6, fill=BG2_COLOR)
        d.text((100, y + 12), rid, font=font(22), fill=ACCENT_BLUE)
        d.rounded_rectangle([260, y + 8, 340, y + 47], radius=4, fill=lvl_color)
        d.text(
            (270, y + 12),
            level,
            font=font(20),
            fill=TEXT_WHITE,
        )
        d.text((370, y + 12), name, font=font(22), fill=TEXT_WHITE)
        d.text((680, y + 12), excerpt, font=font(22), fill=TEXT_GRAY)
    d.text(
        (80, 850),
        "规则引擎自动加载 rules/ 目录下所有 JSON 包, 支持 pack_ids 过滤按行业切换",
        font=font(22),
        fill=TEXT_GRAY,
    )
    img.save(path)


def render_finance(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 50, "场景一：金融违规通话质检", font(48, True), ACCENT_RED)
    d.rounded_rectangle([80, 140, 560, 440], radius=16, fill=(127, 29, 29))
    draw_centered(d, 170, "0", font(110, True), TEXT_WHITE)
    d.text(
        (
            300,
            300,
        ),
        "/ 100",
        font=font(36),
        fill=TEXT_GRAY,
    )
    draw_centered(
        d,
        370,
        "不合格 FAIL",
        font(32, True),
        (252, 165, 165),
    )
    draw_card(d, 600, 140, 200, 120, "红线", "12", ACCENT_RED)
    draw_card(d, 820, 140, 200, 120, "警告", "5", ACCENT_YELLOW)
    draw_card(d, 1040, 140, 200, 120, "确认", "17", ACCENT_GREEN)
    draw_card(d, 1260, 140, 200, 120, "Token", "0", ACCENT_PURPLE)
    draw_card(d, 1480, 140, 200, 120, "耗时", "3ms", ACCENT_CYAN)
    d.text(
        (80, 500), "检出违规话术 (Evidence 溯源):", font=font(30, True), fill=TEXT_WHITE
    )
    violations = [
        ("FIN-001", "承诺保本保收益", '"这款产品保本保息, 年化30%, 稳赚不赔"'),
        ("FIN-003", "类存款误导表述", '"跟存款一样安全, 当存款买就行"'),
        ("FIN-013", "暗示性收益承诺", '"放心买, 至少能赚8个点, 不会亏"'),
        ("FIN-014", "诱导借贷投资", '"您可以先贷款来买, 收益覆盖利息"'),
        ("FIN-012", "索要验证信息", '"把验证码告诉我一下"'),
        ("FIN-010", "拒绝后继续推销", '"客户:我不需要. 坐席:您先别挂..."'),
    ]
    for i, (rid, name, excerpt) in enumerate(violations):
        y = 560 + i * 55
        d.rounded_rectangle([80, y, W - 80, y + 48], radius=6, fill=BG2_COLOR)
        d.text((100, y + 10), rid, font=font(22), fill=ACCENT_BLUE)
        d.text((260, y + 10), name, font=font(22), fill=TEXT_WHITE)
        d.text((600, y + 10), excerpt, font=font(22), fill=TEXT_GRAY)
    img.save(path)


def render_insurance(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(
        d, 50, "场景二：保险违规通话质检（新增行业包）", font(46, True), ACCENT_BLUE
    )
    d.rounded_rectangle([80, 140, 560, 440], radius=16, fill=(127, 29, 29))
    draw_centered(d, 170, "15", font(100, True), TEXT_WHITE)
    d.text(
        (
            340,
            300,
        ),
        "/ 100",
        font=font(36),
        fill=TEXT_GRAY,
    )
    draw_centered(d, 370, "不合格 FAIL", font(32, True), (252, 165, 165))
    draw_card(d, 600, 140, 200, 120, "红线", "8", ACCENT_RED)
    draw_card(d, 820, 140, 200, 120, "警告", "3", ACCENT_YELLOW)
    draw_card(d, 1040, 140, 200, 120, "确认", "11", ACCENT_GREEN)
    draw_card(d, 1260, 140, 200, 120, "行业包", "insurance", ACCENT_CYAN)
    draw_card(d, 1480, 140, 200, 120, "切换", "1 line", ACCENT_PURPLE)
    d.text((80, 500), "保险行业违规检出:", font=font(30, True), fill=TEXT_WHITE)
    violations = [
        ("INS-001", "将保险与存款混同", '"您就把这个当定期存款, 跟存银行一样"'),
        ("INS-002", "承诺保险收益确定性", '"保证收益, 保本保息, 肯定能返本"'),
        ("INS-003", "夸大保障范围", '"什么都保, 什么病都赔, 出了事就赔"'),
        ("INS-005", "诱导隐瞒既往病史", '"健康告知全填否就行, 有病也不用写"'),
        ("INS-007", "诱导退保换保", '"把之前的退了, 旧的不划算, 换这个"'),
        ("INS-016", "承诺理赔速度", '"出了事一定赔, 保证理赔, 秒赔"'),
    ]
    for i, (rid, name, excerpt) in enumerate(violations):
        y = 560 + i * 55
        d.rounded_rectangle([80, y, W - 80, y + 48], radius=6, fill=BG2_COLOR)
        d.text((100, y + 10), rid, font=font(22), fill=ACCENT_BLUE)
        d.text((260, y + 10), name, font=font(22), fill=TEXT_WHITE)
        d.text((600, y + 10), excerpt, font=font(22), fill=TEXT_GRAY)
    img.save(path)


def render_compliant(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 50, "场景三：合规通话对照（对照组）", font(48, True), ACCENT_GREEN)
    d.rounded_rectangle([80, 140, 560, 440], radius=16, fill=(22, 101, 52))
    draw_centered(d, 170, "100", font(110, True), TEXT_WHITE)
    d.text(
        (
            340,
            300,
        ),
        "/ 100",
        font=font(36),
        fill=TEXT_GRAY,
    )
    draw_centered(d, 370, "合格 PASS", font(32, True), (134, 239, 172))
    draw_card(d, 600, 140, 200, 120, "违规数", "0", ACCENT_GREEN)
    draw_card(d, 820, 140, 200, 120, "复核耗时", "18s", ACCENT_BLUE)
    draw_card(d, 1040, 140, 200, 120, "Token", "0", ACCENT_PURPLE)
    draw_card(d, 1260, 140, 200, 120, "音频外发", "0", ACCENT_GREEN)
    draw_card(d, 1480, 140, 200, 120, "隐私", "100%", ACCENT_CYAN)
    d.text((80, 500), "合规要点:", font=font(30, True), fill=TEXT_WHITE)
    highlights = [
        "通话开始表明机构名称与工号身份",
        "充分揭示投资风险, 未承诺保本保收益",
        "告知通话录音事宜, 保障客户知情权",
        "尊重客户拒绝, 未继续推销或纠缠",
        "规范结束语, 感谢客户并道别",
        "费用结构透明, 未隐瞒任何收费项",
    ]
    for i, h in enumerate(highlights):
        y = 560 + i * 55
        d.rounded_rectangle([80, y, W - 80, y + 48], radius=6, fill=BG2_COLOR)
        d.ellipse([100, y + 14, 120, y + 34], fill=ACCENT_GREEN)
        d.text((140, y + 10), h, font=font(26), fill=TEXT_WHITE)
    img.save(path)


def render_benchmark(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 50, "异构加速实测：NPU + GPU 双引擎", font(48, True), ACCENT_BLUE)
    draw_centered(
        d, 120, "NPU ASR 加速 (Encoder + CTC 双组件)", font(32, True), ACCENT_GREEN
    )
    asr_data = [
        ("CPU (PyTorch)", 7.623, TEXT_GRAY, "RTF=0.135"),
        ("NPU (AI Boost)", 1.775, ACCENT_GREEN, "RTF=0.030"),
        ("GPU (Arc)", 2.083, ACCENT_BLUE, "RTF=0.037"),
    ]
    max_val = 8.0
    chart_x, chart_y, chart_h = 200, 260, 250
    for i, (label, val, color, rtf) in enumerate(asr_data):
        bar_h = int((val / max_val) * chart_h)
        x = chart_x + i * 500
        y = chart_y + chart_h - bar_h
        d.rounded_rectangle([x, y, x + 320, chart_y + chart_h], radius=8, fill=color)
        d.text((x, y - 45), f"{val:.3f}s", font=font(36, True), fill=color)
        d.text((x, chart_y + chart_h + 15), label, font=font(26, True), fill=TEXT_WHITE)
        d.text((x, chart_y + chart_h + 55), rtf, font=font(22), fill=TEXT_GRAY)
    draw_centered(
        d,
        590,
        "NPU 4.17x 加速 (CPU 7.623s -> NPU 1.775s)",
        font(28, True),
        ACCENT_GREEN,
    )
    draw_centered(d, 660, "LLM 语义复核加速对比", font(32, True), ACCENT_RED)
    llm_data = [
        ("GPU batch=1", 57.0, TEXT_GRAY, "基线"),
        ("GPU batch=10", 29.76, ACCENT_BLUE, "1.9x 加速"),
        ("Ollama CPU", 89.9, ACCENT_YELLOW, "2.7 tok/s"),
        ("Ollama GPU Vulkan", 20.8, ACCENT_GREEN, "4.4x 加速, 12.0 tok/s"),
    ]
    max_l = 95.0
    for i, (label, val, color, note) in enumerate(llm_data):
        bar_h = int((val / max_l) * 200)
        x = 200 + i * 420
        y_base = 920
        d.rounded_rectangle([x, y_base - bar_h, x + 300, y_base], radius=8, fill=color)
        d.text(
            (x - 20, y_base - bar_h - 40),
            f"{val:.1f}s",
            font=font(32, True),
            fill=color,
        )
        d.text((x - 20, y_base + 10), label, font=font(24, True), fill=TEXT_WHITE)
        d.text((x - 20, y_base + 42), note, font=font(20), fill=TEXT_GRAY)
    img.save(path)


def render_integration(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(
        d, 50, "四平台集成验证：39/39 调用 100% 成功", font(46, True), ACCENT_GREEN
    )
    d.rounded_rectangle([100, 130, W - 100, 190], radius=8, fill=BG2_COLOR)
    cols = [
        ("平台", 120),
        ("接入模式", 450),
        ("调用次数", 780),
        ("成功率", 1050),
        ("平均耗时", 1400),
    ]
    for label, x in cols:
        d.text((x, 145), label, font=font(26, True), fill=TEXT_WHITE)
    rows = [
        ("WorkBuddy", "HTTP API", "10+3=13", "100%", "8.64s / 57.02s", ACCENT_GREEN),
        ("TraeWork", "HTTP API", "10+3=13", "100%", "7.25s / 72.56s", ACCENT_GREEN),
        ("豆包办公", "HTTP API", "10+3=13", "100%", "11.63s / 58.97s", ACCENT_GREEN),
        ("QwenWork", "原生 Skill", "-", "就绪", "触发词调用", ACCENT_BLUE),
    ]
    y = 200
    for name, mode, calls, rate, latency, color in rows:
        d.rounded_rectangle([100, y, W - 100, y + 65], radius=6, fill=BG2_COLOR)
        d.text((120, y + 16), name, font=font(26, True), fill=TEXT_WHITE)
        d.text((450, y + 16), mode, font=font(26), fill=TEXT_GRAY)
        d.text((780, y + 16), calls, font=font(26), fill=TEXT_WHITE)
        d.text((1050, y + 16), rate, font=font(26, True), fill=color)
        d.text((1400, y + 16), latency, font=font(24), fill=TEXT_WHITE)
        y += 75
    draw_centered(
        d,
        540,
        "快速模式 30 次 (10/平台 x 3) + 完整模式 9 次 (3/平台 x 3) = 39 次",
        font(28),
        ACCENT_CYAN,
    )
    draw_centered(
        d,
        590,
        "FastAPI: POST /v1/qa (完整) | POST /v1/qa/fast (快速) | GET /health",
        font(24),
        TEXT_GRAY,
    )
    draw_card(d, 100, 650, 380, 140, "总调用", "39", ACCENT_GREEN)
    draw_card(d, 500, 650, 380, 140, "成功率", "100%", ACCENT_GREEN)
    draw_card(d, 900, 650, 380, 140, "错误数", "0", ACCENT_GREEN)
    draw_card(d, 1300, 650, 380, 140, "达标(>=95%)", "Pass", ACCENT_GREEN)
    img.save(path)


def render_summary(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 60, "VoiceGuard 亮点总结", font(50, True), ACCENT_BLUE)
    highlights = [
        (
            "72 条规则 3 行业包",
            "金融 27 + 保险 18 + 电销 27, 行业切换 1 行代码",
            ACCENT_YELLOW,
        ),
        ("NPU 完整 ASR", "Encoder 234M + CTC 12.8M 双组件, 4.17x 加速", ACCENT_GREEN),
        ("异构三级协同", "NPU(ASR) -> GPU(LLM) -> CPU(Rules) 自动降级", ACCENT_BLUE),
        ("100% 隐私保护", "音频零外发, PII 6 种脱敏, 弱网降级", ACCENT_PURPLE),
        ("零 API Token", "规则引擎 3ms 初筛, 本地模型复核, 100% 节省", ACCENT_CYAN),
        ("39/39 集成验证", "3 平台 x 13 调用, 100% 成功率, 0 错误", ACCENT_GREEN),
        ("Skill 可组合", "3 子技能 + 批量运营, Skill 复用与链式调用", ACCENT_YELLOW),
        ("Evidence 溯源", "每条违规可追溯至规则 ID + 匹配原文 + 音频片段", ACCENT_RED),
    ]
    for i, (title, desc, color) in enumerate(highlights):
        y = 160 + i * 85
        d.ellipse([160, y + 15, 185, y + 40], fill=color)
        d.text((210, y + 8), title, font=font(28, True), fill=color)
        d.text((210, y + 42), desc, font=font(22), fill=TEXT_GRAY)
    draw_centered(
        d,
        880,
        "VoiceGuard | 72 Rules | 3 Scenarios | 4 Platforms | 100% Privacy",
        font(28, True),
        TEXT_WHITE,
    )
    draw_centered(
        d, 930, "Intel Agentic PC Skill Grand Finals 2026 - Suzhou", font(24), TEXT_GRAY
    )
    img.save(path)


def render_closing(path):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_gradient_bg(img, d)
    draw_centered(d, 320, "VoiceGuard", font(90, True), ACCENT_BLUE)
    draw_centered(d, 430, "端侧语音合规质检助手", font(36), TEXT_WHITE)
    d.line([(W // 2 - 250, 520), (W // 2 + 250, 520)], fill=ACCENT_BLUE, width=4)
    draw_centered(
        d,
        560,
        "72 Rules | 3 Scenarios | 4 Platforms | 100% Privacy",
        font(34),
        ACCENT_BLUE,
    )
    draw_centered(
        d, 640, "NPU 4.17x | GPU 4.4x | 0 API Token | 39/39 Calls", font(30), TEXT_WHITE
    )
    draw_centered(
        d, 720, "Intel Agentic PC Skill Grand Finals 2026", font(30), TEXT_WHITE
    )
    draw_centered(d, 770, "根深叶茂 @ ModelScope", font(26), TEXT_GRAY)
    img.save(path)


# ---- TTS narration (Chinese) ----
NARRATION = [
    (
        "01_title",
        "VoiceGuard, 端侧语音合规质检助手。为英特尔 Agentic PC Skill 总决赛打造。"
        "72 条规则, NPU 4.17 倍加速, 零 API Token, 39 次平台调用全部成功。",
    ),
    (
        "02_problem",
        "传统云端质检泄露隐私, 人工审核慢且贵, 监管要求本地处理, 纯关键词匹配误报高。"
        "VoiceGuard 全链路端侧处理, 零外发, 零 Token。",
    ),
    (
        "03_arch",
        "分级调度架构, 完整闭环。"
        "第一步, 音频输入。第二步, NPU 加速 ASR 转写, 4.17 倍加速。"
        "第三步, 规则引擎零 Token 全量初筛, 3 毫秒, 覆盖金融、保险、电销三大行业。"
        "第四步, GPU 本地模型仅复核候选片段, 省 Token 百分之百。"
        "第五步, 生成结构化质检报告, 含评分、违规清单和 Evidence 溯源。",
    ),
    (
        "04_rules",
        "规则引擎, 72 条规则, 3 个行业包。"
        "金融 27 条, 保险 18 条, 电销 27 条。"
        "支持关键词、正则、缺失检测和语义提示四种匹配模式。"
        "行业包切换仅需一行代码。每条违规均可追溯到规则 ID 和匹配原文。",
    ),
    (
        "05_finance",
        "场景一, 金融违规通话。质检评分零分, 不合格。"
        "检出 12 条红线, 5 条警告, 17 条确认违规。"
        "包括承诺保本保收益、类存款误导、暗示性收益承诺、诱导借贷投资、索要验证码、拒绝后继续推销。"
        "全部零 Token, 3 毫秒完成。",
    ),
    (
        "06_insurance",
        "场景二, 保险违规通话。这是新增的保险行业规则包。"
        "评分 15 分, 不合格。检出 8 条红线, 3 条警告, 11 条确认。"
        "包括将保险与存款混同、承诺收益确定性、夸大保障范围、诱导隐瞒既往病史、诱导退保换保、承诺理赔速度。"
        "行业包切换仅一行代码, 即可从金融切换到保险。",
    ),
    (
        "07_compliant",
        "场景三, 合规通话对照组。评分 100 分, 合格。"
        "零违规。坐席表明了机构身份, 充分揭示了投资风险, 告知了通话录音事宜, 尊重客户拒绝, 规范结束语。"
        "音频零外发, 100% 隐私保护。",
    ),
    (
        "08_benchmark",
        "异构加速实测。"
        "NPU ASR: CPU 7.6 秒, NPU 1.8 秒, 4.17 倍加速。"
        "LLM 复核: GPU batch 模式 1.9 倍加速, Ollama GPU Vulkan 4.4 倍加速, 每秒 12 个 Token。"
        "三级降级: NPU 优于 GPU 优于 CPU, 弱网自动 fallback。",
    ),
    (
        "09_integration",
        "四平台集成验证。39 次调用, 100% 成功率, 0 错误。"
        "快速模式 30 次, 每平台 10 次。完整模式 9 次, 每平台 3 次。"
        "WorkBuddy、TraeWork、豆包办公均通过 HTTP API 接入, QwenWork 使用原生 Skill 模式。"
        "目标 95%, 全部超标。",
    ),
    (
        "10_summary",
        "VoiceGuard 亮点总结。"
        "72 条规则 3 行业包, NPU 完整 ASR, 异构三级协同, 100% 隐私保护, 零 API Token, "
        "39 次集成验证全通过, Skill 可组合, Evidence 溯源。",
    ),
    (
        "11_closing",
        "VoiceGuard, 端侧语音合规质检助手。"
        "英特尔 Agentic PC Skill 总决赛 2026, 苏州。"
        "根深叶茂。",
    ),
]

SCENE_DURATIONS = [6, 6, 9, 8, 8, 8, 6, 8, 7, 6, 5]


async def generate_tts():
    import edge_tts

    audio_files = []
    for name, text in NARRATION:
        out = os.path.join(TMP_DIR, f"{name}.mp3")
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(
                    text, voice="zh-CN-XiaoxiaoNeural", rate="-8%"
                )
                await communicate.save(out)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  TTS retry {attempt + 1} for {name}: {e}")
                    await asyncio.sleep(1)
                else:
                    raise
        # Get actual audio duration
        probe = subprocess.run(
            ["ffmpeg", "-i", out, "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        audio_files.append(out)
        print(f"  TTS: {name} done")
    return audio_files


def get_audio_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True,
        text=True,
        timeout=10,
    )
    import json

    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def build_video(scene_images, audio_files, output_path):
    clips = []
    for i, (img, audio, planned_dur) in enumerate(
        zip(scene_images, audio_files, SCENE_DURATIONS)
    ):
        audio_dur = get_audio_duration(audio)
        dur = max(audio_dur + 0.5, planned_dur)
        clip = os.path.join(TMP_DIR, f"clip_{i:02d}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                img,
                "-i",
                audio,
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "fps=30,scale=1920:1080",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-t",
                f"{dur:.2f}",
                "-shortest",
                clip,
            ],
            check=True,
            capture_output=True,
        )
        clips.append(clip)
        print(f"  Clip {i + 1}: {dur:.1f}s")

    concat_file = os.path.join(TMP_DIR, "concat.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for clip in clips:
            abs_path = clip.replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    print(f"  Final video: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate VoiceGuard demo video v3")
    parser.add_argument(
        "--output", default=os.path.join(OUTPUT_DIR, "voiceguard_demo_v3.mp4")
    )
    args = parser.parse_args()

    print("=== VoiceGuard Demo Video v3 ===")

    print("[1/3] Rendering scene images...")
    renderers = [
        render_title,
        render_problem,
        render_architecture,
        render_rule_engine,
        render_finance,
        render_insurance,
        render_compliant,
        render_benchmark,
        render_integration,
        render_summary,
        render_closing,
    ]
    scene_images = []
    for i, renderer in enumerate(renderers):
        path = os.path.join(TMP_DIR, f"scene_{i + 1:02d}.png")
        renderer(path)
        scene_images.append(path)
        print(f"  Scene {i + 1}: {renderer.__name__}")
    print(f"  {len(scene_images)} scenes rendered")

    print("[2/3] Generating TTS narration...")
    audio_files = asyncio.run(generate_tts())

    print("[3/3] Composing final video...")
    build_video(scene_images, audio_files, args.output)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    total_dur = sum(SCENE_DURATIONS)
    print(f"\nDone! Output: {args.output} ({size_mb:.1f} MB, ~{total_dur}s)")

    import shutil

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Temp files cleaned.")


if __name__ == "__main__":
    main()
