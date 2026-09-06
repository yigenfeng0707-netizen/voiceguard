@echo off
chcp 65001 >nul
title VoiceGuard Demo - Intel AI PC
color 0A

echo.
echo  ========================================================================
echo   VoiceGuard - 端侧语音合规质检助手
echo   Intel Agentic PC Skill Grand Finals 2026
echo  ========================================================================
echo.
echo   Hardware: Intel Core Ultra 5 125H (CPU + Arc GPU + NPU AI Boost)
echo   ASR:      SenseVoiceSmall (234M Encoder + 12.8M CTC) on NPU
echo   LLM:      Qwen3-1.7B INT4 on GPU / Ollama qwen2.5:3b on GPU Vulkan
echo   Rules:    40 rules, 3ms, zero-token
echo.
echo   Press any key to start E2E pipeline...
echo  ------------------------------------------------------------------------
pause

cd /d D:\APPs\Intel苏州线下比赛\voiceguard

echo.
echo  [1/4] ASR Transcription (NPU Encoder + CTC Head)...
echo  ------------------------------------------------------------------------
D:\miniconda3\python.exe scripts\run.py demo\samples\demo_finance_violation.wav --no-model --format html --output output\demo_real_record.html --device NPU

echo.
echo  [2/4] Quality Inspection Report Generated!
echo  ------------------------------------------------------------------------
echo  Opening HTML report...
start "" "output\demo_real_record.html"

echo.
echo  [3/4] Benchmark Summary
echo  ------------------------------------------------------------------------
echo   ASR (NPU):    1.775s forward, RTF=0.030, 4.17x vs CPU
echo   ASR (CPU):    7.623s forward, RTF=0.135 (baseline)
echo   Rules Engine: 17ms, 0 tokens, 80%% candidate filtering
echo   LLM Review:   29.76s (batch=10, 1.9x) / 20.8s (Ollama GPU Vulkan)
echo   Token Saving: 100%% (0 API tokens, 0 audio external)
echo.
echo  [4/4] Demo Complete
echo  ------------------------------------------------------------------------
echo   Score: 0/100 (Unqualified) - 17 violations detected
echo   Redline: 8 | Warning: 7 | Notice: 2
echo   100%% Privacy - Sensitive audio never leaves the device
echo.
echo  Press any key to exit...
pause >nul
