"""VoiceGuard Demo Video Generator.

Renders scene images with Pillow, generates TTS narration with edge-tts,
and composes the final MP4 with ffmpeg.

Usage:
    python scripts/make_demo_video.py --output output/voiceguard_demo.mp4
"""

import asyncio
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

# ---- Configuration ----
W, H = 1920, 1080
FONT_REGULAR = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
BG_COLOR = (15, 23, 42)  # dark navy
ACCENT_BLUE = (37, 99, 235)
ACCENT_RED = (220, 38, 38)
ACCENT_GREEN = (22, 163, 74)
ACCENT_PURPLE = (124, 58, 237)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (156, 163, 175)
CARD_BG = (30, 41, 59)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
)
TMP_DIR = os.path.join(OUTPUT_DIR, "_video_tmp")
os.makedirs(TMP_DIR, exist_ok=True)


def font(size, bold=False):
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)


def draw_centered_text(draw, text, y, fnt, fill=TEXT_WHITE, x_offset=0):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2 + x_offset
    draw.text((x, y), text, font=fnt, fill=fill)
    return bbox[3] - bbox[1]


def draw_card(draw, x, y, w, h, label, value, value_color, fnt_label, fnt_value):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=CARD_BG)
    vy = y + (h - fnt_value.size - fnt_label.size - 10) // 2
    draw_centered_text(
        draw, value, vy, fnt_value, fill=value_color, x_offset=x - (W // 2)
    )
    draw_centered_text(
        draw,
        label,
        vy + fnt_value.size + 8,
        fnt_label,
        fill=TEXT_GRAY,
        x_offset=x - (W // 2),
    )


# ---- Scene renderers ----
def render_title(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(d, "VoiceGuard", 320, font(96, True), TEXT_WHITE)
    draw_centered_text(
        d, "End-Device Voice Compliance QA Assistant", 440, font(40), TEXT_GRAY
    )
    draw_centered_text(
        d, "On-Device Call Compliance Quality Audit", 500, font(36), TEXT_GRAY
    )
    # accent line
    d.line([(W // 2 - 200, 580), (W // 2 + 200, 580)], fill=ACCENT_BLUE, width=4)
    draw_centered_text(
        d, "Intel Agentic PC Skill Grand Finals", 620, font(32), ACCENT_BLUE
    )
    draw_centered_text(d, "2026.09 Suzhou - Intel Connection", 670, font(28), TEXT_GRAY)
    img.save(path)


def render_problem(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(
        d, "Pain Point: Call Recording Compliance", 80, font(52, True), TEXT_WHITE
    )
    items = [
        ("Cloud QA leaks sensitive audio", "Recordings sent to cloud = privacy risk"),
        ("Human audit is slow & costly", "Manual review cannot scale"),
        (
            "Regulations demand local processing",
            "PIPL, banking rules require data sovereignty",
        ),
        ("Small models lack accuracy", "Pure keyword match = false positives"),
    ]
    y = 200
    for title, desc in items:
        d.rounded_rectangle([200, y, W - 200, y + 120], radius=10, fill=CARD_BG)
        d.text((240, y + 18), title, font=font(36, True), fill=ACCENT_RED)
        d.text((240, y + 65), desc, font=font(28), fill=TEXT_GRAY)
        y += 140
    img.save(path)


def render_architecture(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(
        d, "Tiered Dispatch Architecture", 60, font(52, True), TEXT_WHITE
    )
    # Pipeline boxes
    boxes = [
        (120, "Audio Input", "wav/mp3/m4a", ACCENT_BLUE),
        (410, "ASR Engine", "SenseVoiceSmall\nCPU/NPU", ACCENT_PURPLE),
        (700, "Rule Engine", "40 rules\n0 token", ACCENT_GREEN),
        (990, "Semantic Review", "Qwen3-1.7B INT4\nGPU", ACCENT_RED),
        (1280, "QA Report", "HTML/JSON\nScore+Evidence", ACCENT_BLUE),
    ]
    box_w, box_h = 250, 160
    box_y = 280
    for x, title, sub, color in boxes:
        d.rounded_rectangle([x, box_y, x + box_w, box_y + box_h], radius=12, fill=color)
        lines = title.split("\n")
        ty = box_y + 25
        d.text((x + 20, ty), title, font=font(28, True), fill=TEXT_WHITE)
        for line in sub.split("\n"):
            ty += 40
            d.text((x + 20, ty), line, font=font(22), fill=TEXT_WHITE)
    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + box_w
        x2 = boxes[i + 1][0]
        ay = box_y + box_h // 2
        d.line([(x1, ay), (x2, ay)], fill=TEXT_GRAY, width=3)
        d.polygon([(x2, ay), (x2 - 12, ay - 8), (x2 - 12, ay + 8)], fill=TEXT_GRAY)
    # Funnel labels
    draw_centered_text(
        d,
        "Stage 1: ASR + Rule Engine (0 token)  ->  Stage 2: Local Model Review  ->  Stage 3: Report",
        520,
        font(30),
        ACCENT_GREEN,
    )
    # Key points
    y = 600
    points = [
        "Rule engine: zero-token full scan, 3ms, identifies ~62% candidate segments",
        "Local model reviews only candidates: Qwen3-1.7B INT4 on Intel GPU, ~1GB",
        "Cloud enhancement optional, receives only sanitized summaries, disabled by default",
        "Three-level degradation: NPU > GPU > CPU, graceful fallback on weak networks",
    ]
    for pt in points:
        d.ellipse([200, y + 8, 215, y + 23], fill=ACCENT_BLUE)
        d.text((240, y), pt, font=font(28), fill=TEXT_WHITE)
        y += 65
    img.save(path)


def render_finance(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(
        d, "Scenario 1: Finance Violation", 50, font(52, True), TEXT_WHITE
    )
    # Score card
    d.rounded_rectangle([100, 150, 560, 450], radius=16, fill=(127, 29, 29))
    draw_centered_text(d, "0", 180, font(120, True), TEXT_WHITE, x_offset=100 - W // 2)
    draw_centered_text(d, "/ 100", 310, font(40), TEXT_GRAY, x_offset=100 - W // 2)
    draw_centered_text(
        d, "FAIL", 360, font(36, True), (252, 165, 165), x_offset=100 - W // 2
    )
    # Summary cards
    draw_card(
        d, 100, 480, 220, 130, "Redline", "12", ACCENT_RED, font(24), font(48, True)
    )
    draw_card(
        d, 340, 480, 220, 130, "Warning", "5", (217, 119, 6), font(24), font(48, True)
    )
    draw_card(
        d, 580, 480, 220, 130, "Confirmed", "17", ACCENT_GREEN, font(24), font(48, True)
    )
    draw_card(
        d,
        820,
        480,
        220,
        130,
        "Token Saved",
        "100%",
        ACCENT_BLUE,
        font(24),
        font(48, True),
    )
    # Violation list
    d.text((100, 640), "Key Violations Detected:", font=font(32, True), fill=TEXT_WHITE)
    violations = [
        "FIN-001: Promised principal protection + 30% return",
        "FIN-003: Compared product to bank deposit",
        "FIN-012: Asked for verification code",
        "FIN-013: Guaranteed minimum 8% profit",
        "FIN-014: Suggested borrowing money to invest",
        "FIN-010: Continued selling after customer refused",
    ]
    y = 690
    for v in violations:
        d.ellipse([100, y + 6, 115, y + 21], fill=ACCENT_RED)
        d.text((130, y), v, font=font(26), fill=TEXT_WHITE)
        y += 50
    img.save(path)


def render_telesales(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(
        d, "Scenario 2: Telesales Violation", 50, font(52, True), TEXT_WHITE
    )
    # Score
    d.rounded_rectangle([100, 150, 560, 450], radius=16, fill=(127, 29, 29))
    draw_centered_text(d, "0", 180, font(120, True), TEXT_WHITE, x_offset=100 - W // 2)
    draw_centered_text(d, "/ 100", 310, font(40), TEXT_GRAY, x_offset=100 - W // 2)
    draw_centered_text(
        d, "FAIL", 360, font(36, True), (252, 165, 165), x_offset=100 - W // 2
    )
    # Cards
    draw_card(
        d, 100, 480, 220, 130, "Redline", "8", ACCENT_RED, font(24), font(48, True)
    )
    draw_card(
        d, 340, 480, 220, 130, "Warning", "12", (217, 119, 6), font(24), font(48, True)
    )
    draw_card(
        d, 580, 480, 220, 130, "Confirmed", "20", ACCENT_GREEN, font(24), font(48, True)
    )
    draw_card(
        d,
        820,
        480,
        220,
        130,
        "Token Saved",
        "100%",
        ACCENT_BLUE,
        font(24),
        font(48, True),
    )
    # List
    d.text((100, 640), "Key Violations Detected:", font=font(32, True), fill=TEXT_WHITE)
    violations = [
        "TEL-001: Impersonated government agency",
        "TEL-013: Failed to identify organization",
        "TEL-017: Suppressed customer judgment",
        "TEL-006: Did not notify call recording",
        "TEL-019: Made unauthorized promises",
        "TEL-020: Used manipulative language",
    ]
    y = 690
    for v in violations:
        d.ellipse([100, y + 6, 115, y + 21], fill=ACCENT_RED)
        d.text((130, y), v, font=font(26), fill=TEXT_WHITE)
        y += 50
    img.save(path)


def render_compliant(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(
        d, "Scenario 3: Compliant Call (Control)", 50, font(52, True), TEXT_WHITE
    )
    # Score
    d.rounded_rectangle([100, 150, 560, 450], radius=16, fill=(22, 101, 52))
    draw_centered_text(
        d, "100", 180, font(120, True), TEXT_WHITE, x_offset=100 - W // 2
    )
    draw_centered_text(d, "/ 100", 310, font(40), TEXT_GRAY, x_offset=100 - W // 2)
    draw_centered_text(
        d, "PASS", 360, font(36, True), (134, 239, 172), x_offset=100 - W // 2
    )
    # Cards
    draw_card(
        d, 100, 480, 220, 130, "Violations", "0", ACCENT_GREEN, font(24), font(48, True)
    )
    draw_card(
        d,
        340,
        480,
        220,
        130,
        "Review Time",
        "18s",
        ACCENT_BLUE,
        font(24),
        font(48, True),
    )
    draw_card(
        d,
        580,
        480,
        220,
        130,
        "Token Saved",
        "100%",
        ACCENT_GREEN,
        font(24),
        font(48, True),
    )
    draw_card(
        d,
        820,
        480,
        220,
        130,
        "Audio Leaked",
        "0",
        ACCENT_GREEN,
        font(24),
        font(48, True),
    )
    # Compliance highlights
    d.text((100, 640), "Compliance Highlights:", font=font(32, True), fill=TEXT_WHITE)
    highlights = [
        "Identified organization at call start",
        "Disclosed investment risks appropriately",
        "Notified customer about call recording",
        "No false promises or guaranteed returns",
        "Respected customer refusal, did not push",
        "Proper closing with thanks and goodbye",
    ]
    y = 690
    for h in highlights:
        d.ellipse([100, y + 6, 115, y + 21], fill=ACCENT_GREEN)
        d.text((130, y), h, font=font(26), fill=TEXT_WHITE)
        y += 50
    img.save(path)


def render_token(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(d, "Token Economy & Privacy", 60, font(52, True), TEXT_WHITE)
    # Comparison
    d.rounded_rectangle([150, 180, 860, 480], radius=16, fill=(127, 29, 29))
    draw_centered_text(
        d, "Cloud QA", 210, font(40, True), (252, 165, 165), x_offset=150 - W // 2
    )
    draw_centered_text(
        d, "450 tokens", 270, font(64, True), TEXT_WHITE, x_offset=150 - W // 2
    )
    draw_centered_text(
        d,
        "Full transcript uploaded",
        360,
        font(28),
        (252, 165, 165),
        x_offset=150 - W // 2,
    )
    draw_centered_text(
        d,
        "Audio sent to external server",
        400,
        font(28),
        (252, 165, 165),
        x_offset=150 - W // 2,
    )
    d.rounded_rectangle([1060, 180, 1770, 480], radius=16, fill=(22, 101, 52))
    draw_centered_text(
        d, "VoiceGuard", 210, font(40, True), (134, 239, 172), x_offset=1060 - W // 2
    )
    draw_centered_text(
        d, "0 tokens", 270, font(64, True), TEXT_WHITE, x_offset=1060 - W // 2
    )
    draw_centered_text(
        d,
        "Local computation: 3861 tokens",
        360,
        font(28),
        (134, 239, 172),
        x_offset=1060 - W // 2,
    )
    draw_centered_text(
        d,
        "Audio never leaves the device",
        400,
        font(28),
        (134, 239, 172),
        x_offset=1060 - W // 2,
    )
    # Big number
    draw_centered_text(d, "100% API Token Saving", 560, font(56, True), ACCENT_GREEN)
    # Points
    y = 660
    points = [
        "Rule engine: zero-token full scan (3ms)",
        "Local model reviews only ~62% candidate segments",
        "Cloud enhancement receives sanitized summaries only",
        "Sensitive voice data stays on device, fully compliant with PIPL",
    ]
    for pt in points:
        d.ellipse([200, y + 6, 215, y + 21], fill=ACCENT_GREEN)
        d.text((240, y), pt, font=font(28), fill=TEXT_WHITE)
        y += 55
    img.save(path)


def render_integration(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(
        d, "Four-Platform Integration Benchmark", 60, font(52, True), TEXT_WHITE
    )
    # Table header
    d.rounded_rectangle([100, 180, W - 100, 250], radius=8, fill=CARD_BG)
    cols = [
        ("Platform", 120),
        ("Mode", 450),
        ("Success Rate", 750),
        ("Avg Latency", 1100),
        ("Priority", 1450),
    ]
    for label, x in cols:
        d.text((x, 195), label, font=font(28, True), fill=TEXT_WHITE)
    # Rows
    rows = [
        ("QwenWork", "Native Skill", "N/A (ready)", "Direct", "P0"),
        ("WorkBuddy", "HTTP API", "100%", "22.65s", "P1"),
        ("TraeWork", "HTTP API", "100%", "6.94s", "P1"),
        ("Doubao", "HTTP + Watch", "100%", "10.85s", "P2"),
    ]
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_GREEN, ACCENT_GREEN]
    y = 260
    for (name, mode, rate, latency, prio), color in zip(rows, colors):
        d.rounded_rectangle([100, y, W - 100, y + 70], radius=6, fill=(30, 41, 59))
        d.text((120, y + 18), name, font=font(26, True), fill=TEXT_WHITE)
        d.text((450, y + 18), mode, font=font(26), fill=TEXT_GRAY)
        d.text((750, y + 18), rate, font=font(26, True), fill=color)
        d.text((1100, y + 18), latency, font=font(26), fill=TEXT_WHITE)
        d.text((1450, y + 18), prio, font=font(26), fill=TEXT_GRAY)
        y += 80
    # Summary
    draw_centered_text(
        d,
        "All 3 HTTP platforms: 100% success rate (target: >=95%)",
        620,
        font(36, True),
        ACCENT_GREEN,
    )
    draw_centered_text(
        d,
        "First call cold start ~54s (ASR model load), subsequent calls ~7s",
        680,
        font(28),
        TEXT_GRAY,
    )
    draw_centered_text(
        d,
        "FastAPI server: POST /v1/qa + POST /v1/qa/fast + GET /health",
        740,
        font(28),
        TEXT_GRAY,
    )
    img.save(path)


def render_closing(path):
    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_centered_text(d, "VoiceGuard", 300, font(96, True), TEXT_WHITE)
    draw_centered_text(d, "End-Device Voice Compliance QA", 430, font(40), TEXT_GRAY)
    d.line([(W // 2 - 200, 520), (W // 2 + 200, 520)], fill=ACCENT_BLUE, width=4)
    draw_centered_text(
        d,
        "40 Rules | 3 Scenarios | 4 Platforms | 100% Privacy",
        560,
        font(36),
        ACCENT_BLUE,
    )
    draw_centered_text(
        d, "Intel Agentic PC Skill Grand Finals 2026", 660, font(32), TEXT_WHITE
    )
    draw_centered_text(
        d, "RootDeepLeafLuxuriant @ ModelScope", 710, font(28), TEXT_GRAY
    )
    img.save(path)


# ---- TTS narration ----
NARRATION = [
    (
        "01_title",
        "VoiceGuard, a privacy-preserving voice compliance quality audit assistant. Built for Intel Agentic PC Skill Grand Finals.",
    ),
    (
        "02_problem",
        "Traditional cloud-based QA leaks sensitive audio. Human review is slow and costly. Regulations demand local processing. Pure keyword matching lacks accuracy.",
    ),
    (
        "03_arch",
        "VoiceGuard uses tiered dispatch. Stage one: ASR transcription and rule engine full scan with zero tokens. Stage two: local Qwen3 model reviews only candidate segments. Stage three: structured quality report. All processing stays on device.",
    ),
    (
        "04_finance",
        "Scenario one: finance violation call. Score zero out of one hundred. Twelve redline violations, five warnings, seventeen confirmed. The salesperson promised principal protection, compared to bank deposits, and asked for verification codes.",
    ),
    (
        "05_telesales",
        "Scenario two: telesales violation call. Score zero. Eight redline and twelve warning violations, twenty confirmed. The caller impersonated a government agency, suppressed customer judgment, and failed to identify their organization.",
    ),
    (
        "06_compliant",
        "Scenario three: compliant call control group. Score one hundred. Zero violations. Proper risk disclosure, recording notification, and respectful handling of customer refusal.",
    ),
    (
        "07_token",
        "Token economy: cloud QA sends four hundred fifty tokens with full transcript. VoiceGuard uses zero API tokens. One hundred percent token saving. Audio never leaves the device. Fully compliant with personal information protection law.",
    ),
    (
        "08_integration",
        "Four-platform integration benchmark. WorkBuddy, TraeWork, and Doubao all achieved one hundred percent success rate via HTTP API. QwenWork uses native skill mode. Target was ninety-five percent. All platforms exceeded expectations.",
    ),
    (
        "09_closing",
        "VoiceGuard. Forty rules, three scenarios, four platforms, one hundred percent privacy. Intel Agentic PC Skill Grand Finals 2026.",
    ),
]

SCENE_DURATIONS = [5, 7, 9, 10, 9, 7, 8, 8, 5]


async def generate_tts():
    import edge_tts

    audio_files = []
    for name, text in NARRATION:
        out = os.path.join(TMP_DIR, f"{name}.mp3")
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural", rate="-5%")
        await communicate.save(out)
        audio_files.append(out)
        print(f"  TTS: {name} done")
    return audio_files


def build_video(scene_images, audio_files, output_path):
    """Build final video with ffmpeg."""
    # Step 1: Create video from images with durations
    concat_file = os.path.join(TMP_DIR, "concat.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for img, dur in zip(scene_images, SCENE_DURATIONS):
            f.write(f"file '{img}'\n")
            f.write(f"duration {dur}\n")
        # Repeat last frame
        f.write(f"file '{scene_images[-1]}'\n")

    silent_video = os.path.join(TMP_DIR, "silent.mp4")
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
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "fps=30,scale=1920:1080",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            silent_video,
        ],
        check=True,
        capture_output=True,
    )
    print("  Silent video created")

    # Step 2: Concatenate audio files
    audio_concat = os.path.join(TMP_DIR, "narration.mp3")
    audio_list = os.path.join(TMP_DIR, "audio_list.txt")
    with open(audio_list, "w", encoding="utf-8") as f:
        for af in audio_files:
            abs_path = af.replace("\\", "/")
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
            audio_list,
            "-c",
            "copy",
            audio_concat,
        ],
        check=True,
        capture_output=True,
    )
    print("  Narration audio concatenated")

    # Step 3: Mux video + audio
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            silent_video,
            "-i",
            audio_concat,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    print(f"  Final video: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate VoiceGuard demo video")
    parser.add_argument(
        "--output", default=os.path.join(OUTPUT_DIR, "voiceguard_demo.mp4")
    )
    args = parser.parse_args()

    print("=== VoiceGuard Demo Video Generator ===")

    # Step 1: Render scene images
    print("[1/3] Rendering scene images...")
    renderers = [
        render_title,
        render_problem,
        render_architecture,
        render_finance,
        render_telesales,
        render_compliant,
        render_token,
        render_integration,
        render_closing,
    ]
    scene_images = []
    for i, renderer in enumerate(renderers):
        path = os.path.join(TMP_DIR, f"scene_{i + 1:02d}.png")
        renderer(path)
        scene_images.append(path)
        print(f"  Scene {i + 1}: {renderer.__name__}")
    print(f"  {len(scene_images)} scenes rendered")

    # Step 2: Generate TTS
    print("[2/3] Generating TTS narration...")
    audio_files = asyncio.run(generate_tts())

    # Step 3: Build video
    print("[3/3] Composing final video...")
    build_video(scene_images, audio_files, args.output)

    # Get file size
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone! Output: {args.output} ({size_mb:.1f} MB)")

    # Cleanup
    import shutil

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Temp files cleaned.")


if __name__ == "__main__":
    main()
