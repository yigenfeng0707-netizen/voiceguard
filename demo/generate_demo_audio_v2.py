"""Generate enhanced demo call recordings for VoiceGuard E2E testing.

Enhancements over v1:
- Background office noise mixing (low-level ambient)
- Different speech rates per speaker (agent slightly faster, customer normal)
- Third voice option for variety
- 2 new scenarios: insurance violation + compliant insurance sales
- Longer, more realistic dialogues with natural disfluencies

Output: WAV files in demo/samples/
"""

import asyncio
import os
import struct
import subprocess
import wave
import random
from pathlib import Path

import edge_tts

# Voice configuration - different voices for variety
AGENT_VOICE = "zh-CN-XiaoxiaoNeural"  # female, agent
CUSTOMER_VOICE = "zh-CN-YunxiNeural"  # male, customer
AGENT_VOICE_2 = "zh-CN-YunyangNeural"  # male, alternative agent voice

# Speech rate variation
AGENT_RATE = "+8%"  # agent speaks slightly faster (sales pitch)
CUSTOMER_RATE = "-5%"  # customer slightly slower (hesitant)

SILENCE_MS = 350  # pause between speakers
OVERLAP_MS = 80  # slight overlap simulation

OUTPUT_DIR = Path(__file__).parent / "samples"
TMP_DIR = Path(__file__).parent / "_tts_tmp_v2"
NOISE_FILE = None  # generated dynamically


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


def _make_silence_wav(path: Path, duration_ms: int):
    sample_rate = 16000
    num_samples = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))


def _generate_noise_wav(path: Path, duration_s: float, noise_type: str = "office"):
    """Generate background noise using ffmpeg.

    office: low-frequency ambient hum + occasional clicks
    """
    # Use ffmpeg anoisesrc filter for background office noise
    # amplitude ~0.005 (very low, ~-40dB)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=color=white:amplitude=0.003:duration={duration_s}:sample_rate=16000",
            "-af",
            "lowpass=f=800,highpass=f=100,volume=0.6",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(path),
        ],
        capture_output=True,
        timeout=30,
    )


def _mix_audio(voice_path: Path, noise_path: Path, out_path: Path):
    """Mix voice audio with background noise using ffmpeg amix."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(voice_path),
            "-i",
            str(noise_path),
            "-filter_complex",
            "[0:a]volume=1.0[a0];[1:a]volume=0.15[a1];[a0][a1]amix=inputs=2:duration=first",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(out_path),
        ],
        capture_output=True,
        timeout=60,
    )


async def _generate_dialogue(lines: list, out_wav: Path, agent_voice: str = None):
    """lines: list of (role, text) where role='agent' or 'customer'."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    voice = agent_voice or AGENT_VOICE
    clip_paths = []
    for i, (role, text) in enumerate(lines):
        if role == "agent":
            v = voice
            rate = AGENT_RATE
        else:
            v = CUSTOMER_VOICE
            rate = CUSTOMER_RATE
        clip = TMP_DIR / f"{out_wav.stem}_{i:03d}.mp3"
        await _tts_one(text, v, clip, rate=rate)
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

    # Concat to raw wav
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

    # Get duration for noise generation
    with wave.open(str(tmp_wav), "r") as wf:
        duration_s = wf.getnframes() / wf.getframerate()

    # Generate background noise
    noise_path = TMP_DIR / f"{out_wav.stem}_noise.wav"
    _generate_noise_wav(noise_path, duration_s)

    # Mix voice + noise
    mixed_path = TMP_DIR / f"{out_wav.stem}_mixed.wav"
    _mix_audio(tmp_wav, noise_path, mixed_path)

    import shutil

    shutil.copy(str(mixed_path), str(out_wav))


# ──────────────────────────────────────────────
# Dialogue scripts
# ──────────────────────────────────────────────

# Scenario 4: Insurance violation - triggers multiple FIN rules
INSURANCE_VIOLATION = [
    (
        "agent",
        "喂您好，我是平安人寿的赵经理，工号6688。今天给您推荐一款特别好的人寿保险产品。",
    ),
    ("customer", "什么保险啊？"),
    (
        "agent",
        "这款万能险，保本保息，年化收益百分之三十起步，稳赚不赔的。成立以来年年正收益，从没亏过。",
    ),
    ("customer", "真这么好？有风险吗？"),
    (
        "agent",
        "没有任何风险，跟存款一样安全，当存款买就行。今天不买就没了，最后一小时，名额马上抢光。",
    ),
    ("customer", "可是我已经退休了，收入不高。"),
    (
        "agent",
        "没事，大爷，您把养老金都投进来，这个收益高。风险测评那里您选激进型就行，我帮您填。",
    ),
    ("customer", "我风险承受能力比较低啊。"),
    (
        "agent",
        "没事，放心买，至少能赚十几个点，不会亏的。您还可以先贷款来买，收益覆盖利息稳赚。",
    ),
    ("customer", "那我考虑考虑。"),
    ("agent", "别考虑了，别去比了。另外把手机收到的验证码告诉我一下，我马上帮您开通。"),
]

# Scenario 5: Compliant insurance sales - control group, no violations
COMPLIANT_INSURANCE = [
    (
        "agent",
        "您好，这里是太平洋人寿保险的客户服务中心，我是保险代理人张磊，工号5521。本次通话可能被记录，用于服务质量监控，请问您可以继续吗？",
    ),
    ("customer", "可以的。"),
    (
        "agent",
        "给您介绍一款定期寿险产品。该产品为消费型保险，非储蓄非投资型，不存在保本或保收益的情况。保险责任为身故及全残保障，等待期九十天。",
    ),
    ("customer", "保费怎么算？"),
    (
        "agent",
        "根据您的年龄和保额，年缴保费约三千二百元，交二十年和保三十年。费率说明如下：风险保费占比百分之七十，储蓄成分无。管理费年化百分之零点五。这是费率说明，无申购费，退保手续费逐年递减。",
    ),
    ("customer", "有风险吗？"),
    (
        "agent",
        "消费型保险的特点是保障期内未出险则不返还保费，存在保费损失的可能。请您根据自身保障需求和风险承受能力谨慎决策。产品募集期到本月三十号，您可结合自身情况合理安排。",
    ),
    ("customer", "我想再对比一下其他产品。"),
    (
        "agent",
        "完全理解，您可以多对比了解。请您根据真实情况独立完成风险测评，我们会根据测评结果为您匹配合适的产品。投资有风险，请您谨慎决策。感谢您的接听，祝您生活愉快，再见。",
    ),
]


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scripts = [
        ("demo_insurance_violation.wav", INSURANCE_VIOLATION, AGENT_VOICE),
        ("demo_compliant_insurance.wav", COMPLIANT_INSURANCE, AGENT_VOICE_2),
    ]

    for name, lines, voice in scripts:
        out = OUTPUT_DIR / name
        print(f"Generating {name} ({len(lines)} lines, voice={voice})...")
        await _generate_dialogue(lines, out, agent_voice=voice)
        size_mb = out.stat().st_size / (1024 * 1024)
        # Get duration
        with wave.open(str(out), "r") as wf:
            dur = wf.getnframes() / wf.getframerate()
        print(f"  -> {out} ({size_mb:.1f} MB, {dur:.1f}s)")

    # Cleanup temp
    import shutil

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    print("Done. All enhanced demo audio generated.")


if __name__ == "__main__":
    asyncio.run(main())
