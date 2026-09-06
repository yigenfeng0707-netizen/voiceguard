#!/usr/bin/env python3
"""Generate VoiceGuard real-demo video frames from actual NPU run data."""

import os, math
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30
FONT = "C:/Windows/Fonts/msyh.ttc"
OUT = "D:/APPs/Intel苏州线下比赛/voiceguard/output/demo_frames"
os.makedirs(OUT, exist_ok=True)


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT, size)
    except:
        return ImageFont.load_default()


# Colors
BG = (15, 23, 42)  # dark navy
BG2 = (30, 41, 59)  # lighter navy
WHITE = (255, 255, 255)
GRAY = (148, 163, 184)
BLUE = (59, 130, 246)
GREEN = (34, 197, 94)
RED = (239, 68, 68)
YELLOW = (234, 179, 8)
PURPLE = (147, 51, 234)
CYAN = (6, 182, 212)


def draw_bg(img, draw):
    """Gradient background."""
    for y in range(H):
        ratio = y / H
        r = int(BG[0] + (BG2[0] - BG[0]) * ratio)
        g = int(BG[1] + (BG2[1] - BG[1]) * ratio)
        b = int(BG[2] + (BG2[2] - BG[2]) * ratio)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


def draw_text(draw, x, y, text, fnt, color=WHITE):
    draw.text((x, y), text, font=fnt, fill=color)


def draw_centered(draw, y, text, fnt, color=WHITE):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w = bbox[2] - bbox[0]
    draw.text(((W - w) // 2, y), text, font=fnt, fill=color)


def draw_bar(draw, x, y, w, h, color, label, value, fnt_label, fnt_value):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=color)
    draw_text(draw, x + 20, y + 10, label, fnt_label, WHITE)
    draw_text(draw, x + 20, y + 50, value, fnt_value, WHITE)


def make_frame(name, draw_fn):
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    draw_bg(img, draw)
    draw_fn(draw)
    img.save(os.path.join(OUT, f"{name}.png"))
    return img


# ===== Scene 1: Title (frames 0-89, 3s) =====
f_title = font(72, True)
f_sub = font(36)
f_small = font(24)


def scene1(draw):
    draw_centered(draw, 350, "VoiceGuard", f_title, BLUE)
    draw_centered(draw, 450, "端侧语音合规质检助手", f_sub, WHITE)
    draw_centered(draw, 530, "Intel Agentic PC Skill Grand Finals 2026", f_small, GRAY)
    # Hardware badge
    draw_centered(
        draw,
        620,
        "Intel Core Ultra 5 125H  |  CPU + Arc GPU + NPU AI Boost",
        f_small,
        CYAN,
    )
    # Key metrics
    metrics = [
        ("NPU 4.17x", "ASR 加速", BLUE),
        ("100%", "隐私保护", GREEN),
        ("0 tokens", "API 节省", PURPLE),
        ("40 rules", "规则引擎", YELLOW),
    ]
    x_start = 360
    for i, (num, label, color) in enumerate(metrics):
        x = x_start + i * 350
        draw_bar(draw, x, 720, 280, 120, color, label, num, font(20), font(40, True))


for i in range(90):
    make_frame(f"s1_{i:04d}", scene1)

# ===== Scene 2: Architecture (frames 90-209, 4s) =====
f_h2 = font(42, True)
f_body = font(28)
f_code = font(24)


def scene2(draw):
    draw_centered(draw, 80, "分级调度架构", f_h2, BLUE)
    stages = [
        (
            "Stage 1",
            "ASR + 规则引擎",
            "NPU",
            "转写 + 零 token 全量初筛",
            "4.17x 加速",
            BLUE,
        ),
        (
            "Stage 2",
            "LLM 语义复核",
            "GPU",
            "Qwen3-1.7B INT4 / Ollama Vulkan",
            "1.9x ~ 4.4x",
            GREEN,
        ),
        ("Stage 3", "报告生成", "CPU", "评分 + 违规清单 + 漏斗", "17ms", YELLOW),
        (
            "Stage 4",
            "云增强 (可选)",
            "CPU+Cloud",
            "脱敏 -> 云端 API -> 弱网降级",
            "0 tokens",
            PURPLE,
        ),
    ]
    for i, (stage, title, device, desc, perf, color) in enumerate(stages):
        y = 180 + i * 180
        draw.rounded_rectangle([200, y, 1720, y + 150], radius=12, fill=BG2)
        draw.rounded_rectangle([200, y, 280, y + 150], radius=12, fill=color)
        draw_text(draw, 300, y + 20, stage, font(28, True), color)
        draw_text(draw, 300, y + 60, title, font(32, True), WHITE)
        draw_text(draw, 700, y + 20, f"Device: {device}", f_body, GRAY)
        draw_text(draw, 700, y + 60, desc, f_body, WHITE)
        draw_text(draw, 1300, y + 40, perf, font(36, True), color)

    # Arrow
    draw_centered(
        draw,
        920,
        "录音 -> ASR(NPU) -> 规则引擎(CPU) -> LLM复核(GPU) -> 质检报告",
        font(26),
        CYAN,
    )


for i in range(120):
    make_frame(f"s2_{i:04d}", scene2)


# ===== Scene 3: NPU Benchmark (frames 210-329, 4s) =====
def scene3(draw):
    draw_centered(draw, 80, "NPU 完整 ASR 加速 (Encoder + CTC 双组件)", f_h2, BLUE)

    # Bar chart
    data = [
        ("CPU (PyTorch)", 7.623, GRAY, "RTF=0.135"),
        ("NPU (AI Boost)", 1.775, GREEN, "RTF=0.030"),
        ("GPU (Arc)", 2.083, BLUE, "RTF=0.037"),
    ]
    max_val = 8.0
    chart_x = 300
    chart_y = 220
    chart_w = 1320
    chart_h = 450

    for i, (label, val, color, rtf) in enumerate(data):
        bar_h = int((val / max_val) * chart_h)
        bar_w = 300
        x = chart_x + 50 + i * 420
        y = chart_y + chart_h - bar_h
        draw.rounded_rectangle(
            [x, y, x + bar_w, chart_y + chart_h], radius=8, fill=color
        )
        draw_text(draw, x - 20, y - 50, f"{val:.3f}s", font(36, True), color)
        draw_text(draw, x - 20, chart_y + chart_h + 20, label, font(28, True), WHITE)
        draw_text(draw, x - 20, chart_y + chart_h + 60, rtf, font(24), GRAY)

    # NPU components breakdown
    draw_text(
        draw,
        300,
        800,
        "NPU 组件: Encoder(234M) 3.82s编译/129.93ms每chunk  |  CTC Head(12.8M) 0.22s编译/8.86ms每次",
        font(24),
        CYAN,
    )
    draw_text(
        draw,
        300,
        840,
        "加速比: 4.17x (CPU 7.623s -> NPU 1.775s)",
        font(28, True),
        GREEN,
    )


for i in range(120):
    make_frame(f"s3_{i:04d}", scene3)


# ===== Scene 4: Rule Engine Results (frames 330-449, 4s) =====
def scene4(draw):
    draw_centered(draw, 80, "规则引擎实测结果 (金融违规场景)", f_h2, BLUE)

    # Summary cards
    cards = [
        ("0/100", "质检总分", "不合格", RED),
        ("8", "红线确认", "Redline", RED),
        ("7", "警告确认", "Warning", YELLOW),
        ("2", "提示确认", "Notice", CYAN),
        ("17ms", "引擎耗时", "0 tokens", GREEN),
    ]
    for i, (num, label, sub, color) in enumerate(cards):
        x = 100 + i * 360
        draw.rounded_rectangle([x, 180, x + 320, 320], radius=12, fill=BG2)
        draw.rounded_rectangle([x, 180, x + 320, 220], radius=12, fill=color)
        draw_text(draw, x + 20, 185, label, font(22), WHITE)
        draw_text(draw, x + 20, 235, num, font(48, True), color)
        draw_text(draw, x + 20, 295, sub, font(20), GRAY)

    # Violation table
    draw_text(draw, 100, 370, "典型违规话术检出:", font(28, True), WHITE)
    violations = [
        (
            "FIN-001",
            "红线",
            "承诺保本保收益",
            "这款产品保本保息, 年化收益30%, 稳赚不赔",
        ),
        ("FIN-003", "红线", "类存款误导表述", "当然跟存款一样安全, 当存款买就行"),
        ("FIN-013", "红线", "暗示性收益承诺", "至少能赚8个点, 不会亏的"),
        ("FIN-014", "红线", "诱导借贷投资", "您还可以先贷款来买, 收益覆盖利息稳赚"),
        ("FIN-012", "红线", "索要验证信息", "把验证码告诉我一下, 我马上帮您开通"),
        ("FIN-005", "警告", "饥饿营销施压", "最后一小时名额马上抢光"),
    ]
    for i, (rid, level, name, excerpt) in enumerate(violations):
        y = 420 + i * 55
        lvl_color = RED if level == "红线" else YELLOW
        draw.rounded_rectangle([100, y, 1820, y + 48], radius=6, fill=BG2)
        draw_text(draw, 120, y + 10, rid, font(22), BLUE)
        draw.rounded_rectangle([280, y + 5, 360, y + 43], radius=4, fill=lvl_color)
        draw_text(draw, 290, y + 10, level, font(20), WHITE)
        draw_text(draw, 390, y + 10, name, font(22), WHITE)
        draw_text(draw, 700, y + 10, excerpt, font(22), GRAY)

    draw_text(
        draw,
        100,
        780,
        "漏斗: 5片段 -> 4标记(80%) -> 17违规 -> 9语义复核任务",
        font(26),
        CYAN,
    )
    draw_text(
        draw,
        100,
        820,
        "Token: 云端估算438 -> VoiceGuard 0 (100%节省)  |  音频零外发",
        font(26),
        GREEN,
    )


for i in range(120):
    make_frame(f"s4_{i:04d}", scene4)


# ===== Scene 5: LLM Benchmark (frames 450-569, 4s) =====
def scene5(draw):
    draw_centered(draw, 80, "LLM 语义复核加速对比", f_h2, BLUE)

    configs = [
        ("GPU batch=1", 57.0, GRAY, "29次调用, 基线"),
        ("GPU batch=10", 29.76, BLUE, "3次调用, 1.9x加速"),
        ("Ollama CPU", 89.9, YELLOW, "2.7 tok/s"),
        ("Ollama GPU Vulkan", 20.8, GREEN, "12.0 tok/s, 4.4x加速"),
    ]
    max_val = 95.0
    for i, (label, val, color, note) in enumerate(configs):
        bar_h = int((val / max_val) * 500)
        x = 200 + i * 420
        y_base = 680
        draw.rounded_rectangle(
            [x, y_base - bar_h, x + 300, y_base], radius=8, fill=color
        )
        draw_text(
            draw, x - 30, y_base - bar_h - 50, f"{val:.1f}s", font(36, True), color
        )
        draw_text(draw, x - 30, y_base + 20, label, font(26, True), WHITE)
        draw_text(draw, x - 30, y_base + 60, note, font(22), GRAY)

    draw_text(
        draw,
        200,
        800,
        "Ollama GPU: OLLAMA_IGPU_ENABLE=1 -> Vulkan -> Intel Arc iGPU (9.0 GiB)",
        font(24),
        CYAN,
    )
    draw_text(
        draw,
        200,
        840,
        "准确率: 100% (8/8)  |  VRAM: 1840 MB  |  吞吐: 12.0 tok/s",
        font(24),
        GREEN,
    )


for i in range(120):
    make_frame(f"s5_{i:04d}", scene5)


# ===== Scene 6: Platform Integration (frames 570-689, 4s) =====
def scene6(draw):
    draw_centered(draw, 80, "四平台完整模式集成", f_h2, BLUE)

    platforms = [
        ("QwenWork", "原生 Skill", "SKILL.md 就绪", "P0", BLUE),
        ("WorkBuddy", "HTTP API", "100% 成功 / 32.3s", "P1", GREEN),
        ("TraeWork", "HTTP API", "100% 成功 / 45.8s", "P1", GREEN),
        ("豆包办公", "HTTP+监听", "100% 成功 / 45.3s", "P2", GREEN),
    ]
    for i, (name, mode, result, prio, color) in enumerate(platforms):
        y = 200 + i * 160
        draw.rounded_rectangle([200, y, 1720, y + 140], radius=12, fill=BG2)
        draw.rounded_rectangle([200, y, 400, y + 140], radius=12, fill=color)
        draw_text(draw, 240, y + 40, name, font(36, True), WHITE)
        draw_text(draw, 240, y + 85, prio, font(24), WHITE)
        draw_text(draw, 460, y + 30, f"接入模式: {mode}", font(28), WHITE)
        draw_text(draw, 460, y + 75, result, font(28), color)
        draw_text(
            draw, 460, y + 110, "完整模式: ASR(NPU) + Rules + LLM(GPU)", font(22), GRAY
        )

    draw_centered(
        draw, 880, "6/6 调用 100% 成功  |  目标 >= 95% 全部超标", font(28, True), GREEN
    )
    draw_centered(
        draw,
        930,
        "HTTP API: POST /v1/qa (完整)  |  POST /v1/qa/fast (快速)  |  GET /health",
        font(24),
        CYAN,
    )


for i in range(120):
    make_frame(f"s6_{i:04d}", scene6)


# ===== Scene 7: Summary (frames 690-809, 4s) =====
def scene7(draw):
    draw_centered(draw, 100, "VoiceGuard 亮点总结", f_h2, BLUE)

    highlights = [
        ("完整 ASR on NPU", "Encoder(234M) + CTC(12.8M) 双组件, 4.17x 加速", GREEN),
        ("异构三级协同", "NPU(ASR) -> GPU(LLM) -> CPU(Rules) 自动降级", BLUE),
        ("100% 隐私保护", "敏感音频零外发, PII 6种脱敏, 弱网降级", PURPLE),
        ("零 API Token", "规则引擎 3ms 初筛, 本地模型复核, 100% 节省", CYAN),
        (
            "Skill 可组合",
            "3 子技能(vg-transcribe/rules-check/report-gen) + 批量运营",
            YELLOW,
        ),
        ("94 单元测试", "覆盖 rules/reviewer/pipeline/report/transcribe, 0.8s", RED),
        ("5 场景验证", "金融/电销/保险 + 2 合规对照, E2E 全通过", GREEN),
        ("4 平台集成", "QwenWork/WorkBuddy/TraeWork/豆包, 100% 成功", BLUE),
    ]
    for i, (title, desc, color) in enumerate(highlights):
        y = 200 + i * 85
        draw.ellipse([220, y + 15, 250, y + 45], fill=color)
        draw_text(draw, 280, y + 10, title, font(28, True), color)
        draw_text(draw, 280, y + 45, desc, font(24), GRAY)

    draw_centered(
        draw,
        900,
        "VoiceGuard | 40 Rules | 5 Scenarios | 4 Platforms | 100% Privacy | NPU 4.17x",
        font(28, True),
        WHITE,
    )
    draw_centered(
        draw, 950, "Intel Agentic PC Skill Grand Finals 2026 - Suzhou", font(24), GRAY
    )


for i in range(120):
    make_frame(f"s7_{i:04d}", scene7)

total_frames = 90 + 120 + 120 + 120 + 120 + 120 + 120
print(f"Total frames: {total_frames}")
print(f"Duration: {total_frames / FPS:.1f}s")
print(f"Output dir: {OUT}")
