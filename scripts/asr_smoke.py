"""ASR 冒烟测试：funasr + SenseVoiceSmall 本地转写（D1 验证用）。

验证目标：样例音频 → 带时间戳逐字稿（为 rules_engine 供料）。
后续：导出 OpenVINO 版本上 NPU（D2 任务），本脚本先确保正确性基线。
"""
import json
import os
import sys
import time

MODEL_DIR = r"D:\APPs\OpenVINO\demo\local-meeting-minutes\models\SenseVoiceSmall"
SAMPLES = r"D:\APPs\OpenVINO\demo\local-meeting-minutes\demo\samples"


def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SAMPLES, "single_zh.wav")
    t0 = time.perf_counter()
    from funasr import AutoModel
    model = AutoModel(model=MODEL_DIR, device="cpu", disable_update=True)
    t_load = time.perf_counter() - t0
    print(f"[load] {t_load:.1f}s")

    t1 = time.perf_counter()
    res = model.generate(input=audio, language="auto", use_itn=True, batch_size_s=300)
    t_infer = time.perf_counter() - t1

    out = []
    for item in res:
        out.append({"text": item.get("text", ""), "timestamp": item.get("timestamp", [])[:8]})
    print(json.dumps({
        "audio": os.path.basename(audio),
        "load_s": round(t_load, 1),
        "infer_s": round(t_infer, 2),
        "segments": out,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
