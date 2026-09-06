# VoiceGuard 进度日志

## D0 · 2026-09-04（周五）晚 — 完成 ✅

### 环境确认
- Python 3.13.13（miniconda3）/ OpenVINO 2026.3.0 / optimum-intel 2.1.0 / funasr 1.4.1 / nncf 3.3.0 — 全部就绪
- 设备冒烟通过：**CPU（Core Ultra 5 125H）+ GPU（Arc iGPU）+ NPU（Intel AI Boost）三后端全部被 OpenVINO 识别**

### D0 产出
| 交付物 | 状态 |
|--------|------|
| 项目骨架 + 12 个初赛运行时模块移植 | ✅ |
| rules/finance.json（12条）+ rules/telesales.json（12条） | ✅ |
| scripts/rules_engine.py 零token引擎（自测：5片段→红线6/警告3/提示2，3ms） | ✅ |
| SKILL.md / README / requirements / .gitignore | ✅ |

---

## D0 加时赛（22:00–00:30）— 用户确认选题后开工

### 模型与转换（关键路径）— ✅ 已突破
- ✅ Qwen3-4B 下载完成（7.6GB）/ Qwen3-1.7B 下载完成（3.8GB）— 魔搭 ID 无 -Instruct 后缀
- ✅ **Qwen3-1.7B INT4 转换成功**（`D:\vg_ov\Qwen3-1.7B-ov-int4`，1.04GB，openvino-genai 可直接加载）
- ❌ 4B 转换仍失败：`save_model` badbit（UTF-8 环境下复现，确认是尺寸相关问题而非编码问题）→ D1 攻坚
- ✅ **GPU 推理验证通过**：Arc iGPU 加载 14.5s，no_think 模式推理 **2.0s**，正反用例判定全对：
  - 违规样本（保本保息/稳赚不赔）→ confirmed: true + 理由
  - 合规样本（存款利率2%客观表述）→ confirmed: false
- ⚠️ **NPU 探测结论**：LLM 可编译加载（145s）但生成时 `ZE_RESULT_ERROR_DEVICE_LOST`（设备挂起）——Meteor Lake NPU 暂不适合 LLM 生成。**最终架构分工：NPU→ASR（D2），GPU→语义复核（已通），CPU→兜底**

### ASR 链路 — ✅ 验证通过
- SenseVoiceSmall + funasr 1.4.1：真实语音（4分钟英文旁白）转写 3223 字成功，语言/情感/事件标记正常
- 初赛样例音频是 NumPy 合成信号（无真实人声），`<|nospeech|>` 是正确判定——**参赛 demo 音频必须自录/真实 TTS**
- transcribe.py：funasr → Segment 列表（分句修复：按句边界切分+过短合并+过长硬切）

### 管线模块（全部完成）
| 模块 | 状态 |
|------|------|
| scripts/transcribe.py | ✅ ASR 前端（分句已修复） |
| scripts/rules_engine.py | ✅ 第一级零token筛查（3ms/5片段） |
| scripts/pipeline.py | ✅ Stage1 编排，真实音频全链路跑通 |
| scripts/reviewer.py | ✅ **第二级复核器（openvino-genai LLMPipeline，GPU INT4，/no_think，解析三格式兼容），真实推理通过** |
| scripts/report_gen.py | ✅ 第三级报告（评分/漏斗/违规清单/token经济性），合成数据 21/100 测试通过 |
| scripts/run.py | ✅ 统一入口（SKILL.md 契约），UTF-8 控制台保护，模型自动发现 |
| agent_integrations/INTEGRATION_NOTES.md | ✅ 四平台接入作战备忘 |

### 踩坑记录（备战资产）
1. **GBK 控制台**：Windows cmd 下跑 optimum-cli / 含✅输出必须 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`（或程序内强制 UTF-8 stdout），否则 rich/print 崩溃掩盖真实错误
2. **optimum-cli 参数**：`--weight-format int4 --sym --group-size 128`（不接受 int4_sym_g128）；本地目录必须显式 `--task text-generation-with-past`
3. **OpenVINO save_model badbit**：与模型尺寸相关（1.7B 过 / 4B 挂），D1 按假设清单攻坚
4. **optimum-intel 2.1.0 与 transformers 5.x 不兼容**（`get_experts_implementation` 缺失）→ LLM 推理一律走 openvino-genai，不用 OVModelForCausalLM.generate
5. **Qwen3 思考模式**：默认 <think> 会吃光 token，必须 `/no_think`（推理从 21.5s → 2.0s）
6. **NPU LLM 生成不可用**（DEVICE_LOST），NPU 定位给 ASR
7. **魔搭搜索**：`PUT https://modelscope.cn/api/v1/dolphin/models`

---

## D1 · 9/5（周六）作战清单（按优先级）

1. 【攻坚】4B INT4 转换：假设验证序列 ①`--weight-format fp32` 导出测试（验证是否 fp16 压缩写盘问题）②pip 试 openvino 2025.x/nightly ③分段导出。目标：4B 可用（主打），1.7B 已是保底
2. Demo 音频制作：TTS 合成违规话术 3 段（金融版/电销版/对照组）+ 用户自录 1 段，带授权声明（D2 硬需求，提前做）
3. SenseVoice → OpenVINO 导出（ONNX 中转调研），目标 ASR 上 NPU —— 异构调度叙事的另一半
4. run.py 中文违规音频 E2E 全链路（等 Demo 音频）
5. 规则库按用户审核意见修订
6. 开始 QwenWork 技能安装联调（skill 复制到 ~/.qwenworkcn/skills/voiceguard-qa/）
