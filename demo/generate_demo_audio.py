"""Generate 3 demo call recordings for VoiceGuard E2E testing.

Segment 1: Finance violation call (multiple FIN-xxx rules triggered)
Segment 2: Telesales violation call (multiple TEL-xxx rules triggered)
Segment 3: Compliant call (no violations, control group)

Uses edge-tts with two voices to simulate agent/customer dialogue.
Output: WAV files in demo/samples/
"""

import asyncio
import os
import struct
import wave
from pathlib import Path

import edge_tts

AGENT_VOICE = "zh-CN-XiaoxiaoNeural"
CUSTOMER_VOICE = "zh-CN-YunxiNeural"
SILENCE_MS = 400  # pause between speakers

OUTPUT_DIR = Path(__file__).parent / "samples"
TMP_DIR = Path(__file__).parent / "_tts_tmp"


async def _tts_one(
    text: str, voice: str, out_path: Path, rate: str = "+0%", retries: int = 3
):
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(out_path))
            if out_path.stat().st_size > 100:
                return
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)
                continue
            raise


async def _generate_dialogue(lines: list, out_wav: Path):
    """lines: list of (role, text) where role='agent' or 'customer'."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    for i, (role, text) in enumerate(lines):
        voice = AGENT_VOICE if role == "agent" else CUSTOMER_VOICE
        clip = TMP_DIR / f"{out_wav.stem}_{i:03d}.mp3"
        await _tts_one(text, voice, clip)
        clip_paths.append(clip)

    # Build silence wav
    silence_path = TMP_DIR / "silence.wav"
    _make_silence_wav(silence_path, SILENCE_MS)

    # Concatenate: clip0 -> silence -> clip1 -> silence -> ...
    parts = []
    for i, clip in enumerate(clip_paths):
        if i > 0:
            parts.append(str(silence_path))
        parts.append(str(clip))

    concat_txt = TMP_DIR / f"{out_wav.stem}_concat.txt"
    concat_txt.write_text("\n".join(f"file '{p}'" for p in parts), encoding="utf-8")

    import subprocess

    tmp_wav = TMP_DIR / f"{out_wav.stem}_raw.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(tmp_wav),
        ],
        capture_output=True,
        timeout=120,
    )
    import shutil

    shutil.copy(str(tmp_wav), str(out_wav))


def _make_silence_wav(path: Path, duration_ms: int):
    sample_rate = 16000
    num_samples = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))


# ──────────────────────────────────────────────
# Dialogue scripts
# ──────────────────────────────────────────────

FINANCE_VIOLATION = [
    (
        "agent",
        "喂您好，我是鑫源财富的王经理，工号8807。今天给您推荐一款特别好的理财产品。",
    ),
    ("customer", "什么产品啊？"),
    (
        "agent",
        "这款产品保本保息，年化收益30%起步，稳赚不赔的。成立以来年年正收益，从没亏过。",
    ),
    ("customer", "真这么好？"),
    (
        "agent",
        "那当然，跟存款一样安全，当存款买就行。今天不买就没了，最后一小时，名额马上抢光。",
    ),
    ("customer", "可是我风险承受能力比较低。"),
    (
        "agent",
        "没事，风险测评那里您选激进型就行，我帮您填。您放心买，至少能赚8个点，不会亏的。",
    ),
    ("customer", "那我考虑考虑。"),
    (
        "agent",
        "不用考虑了，别去比了。您还可以先贷款来买，收益覆盖利息稳赚。另外把验证码告诉我一下，我马上帮您开通。",
    ),
]

TELESALES_VIOLATION = [
    ("agent", "喂，给您推荐一个套餐，非常划算。"),
    ("customer", "你们是哪家公司的？"),
    ("agent", "我们是移动的，运营商通知，您的套餐可以免费升级。"),
    ("customer", "我现在不方便接电话，在睡觉。"),
    ("agent", "就耽误您一分钟。这个套餐先免费试用，到期自动续费，不用管它。"),
    ("customer", "我不需要，你们怎么又打来了，昨天刚打过。"),
    (
        "agent",
        "你必须买，你肯定要办，没有别的选择。别去比了，不用查了，信我就对了。你邻居也买了，你同事也办了。",
    ),
    ("customer", "我不买了，太贵了。"),
    ("agent", "买不起就别问。不买你会后悔的，后果自负。"),
    ("customer", "你态度怎么这样？"),
    ("agent", "有病吧，爱买不买。我是国家认证高级理财师，CFA持证，你听我的就行。"),
]

COMPLIANT_CALL = [
    (
        "agent",
        "您好，这里是恒信财富的客户服务中心，我是客户经理李明，工号3321。本次通话可能被记录，用于服务质量监控，请问您可以继续吗？",
    ),
    ("customer", "可以的。"),
    (
        "agent",
        "给您介绍一款稳健型理财产品，过往年化收益在3%到6%之间波动，存在亏损可能。产品为非保本浮动收益型，不承诺保本。",
    ),
    ("customer", "风险大吗？"),
    (
        "agent",
        "该产品风险等级为R2中低风险，需要您完成风险测评后，风险等级匹配才能购买。这是费率说明，管理费年化0.6%，无申购费，赎回费持有满30天免收。",
    ),
    ("customer", "我风险测评是保守型。"),
    (
        "agent",
        "保守型适合R1到R2产品，这款R2产品与您的风险承受能力匹配。但投资有风险，请您谨慎决策。如果您需要了解其他产品也可以对比看看。",
    ),
    ("customer", "好的，我想再考虑一下。"),
    (
        "agent",
        "好的，完全理解。您可以随时联系我们，产品募集期到本周五，您可结合自身情况合理安排。感谢您的接听，祝您生活愉快，再见。",
    ),
]


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scripts = [
        ("demo_finance_violation.wav", FINANCE_VIOLATION),
        ("demo_telesales_violation.wav", TELESALES_VIOLATION),
        ("demo_compliant_call.wav", COMPLIANT_CALL),
    ]

    for name, lines in scripts:
        out = OUTPUT_DIR / name
        print(f"Generating {name} ({len(lines)} lines)...")
        await _generate_dialogue(lines, out)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"  -> {out} ({size_mb:.1f} MB)")

    # Cleanup temp
    import shutil

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    print("Done. All demo audio generated.")


if __name__ == "__main__":
    asyncio.run(main())
