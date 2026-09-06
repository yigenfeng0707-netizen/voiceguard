# VoiceGuard 赛事提交清单

> 英特尔 Agentic PC Skill 大赛全国总决赛 (2026-09-22~23 · 苏州)

## 已完成项

| 编号 | 项目 | 状态 | 文件 |
|------|------|------|------|
| D1-D6 | 核心开发 (ASR/规则/LLM/报告/API) | ✅ | 33 scripts, 94 tests |
| P1-1 | Skill 可组合性 (3 子技能 + qa_ops_daily) | ✅ | subskills/, scripts/qa_ops_daily.py |
| P1-2 | 云增强骨架 (PII 脱敏 + 弱网降级) | ✅ | scripts/cloud_enhance.py |
| P1-3 | NPU 完整 ASR (Encoder + CTC 双组件) | ✅ | scripts/npu_ctc.py, models/SenseVoiceSmall_ov/ |
| P1-4 | Ollama GPU Vulkan 加速 | ✅ | reviewer.py create_reviewer() 工厂模式 |
| P2-1 | 路演 PPT | ✅ | 路演/VoiceGuard路演.pptx (10页) |
| P2-2 | 真机 Demo 视频 | ✅ | output/voiceguard_real_demo_v2.mp4 (27s, 1080p) |
| P2-3 | ModelScope 文章 (更新版) | ✅ | community/modelscope_article.md (含 NPU CTC/Ollama/云增强/Skill可组合性) |
| P2-4 | 魔搭 Skills Center 发布说明 | ✅ | community/modelscope_skill_release.md |
| P2-5 | 提交包 v4 | ✅ | voiceguard_submission_v4.zip (87文件, 2.2MB, 含最新代码+CHECKLIST) |

## 待用户操作项 (P0)

以下操作需要用户账号和平台权限，DuMate 无法代为执行：

### 1. 魔搭 Skills Center 发布
- **平台**: https://www.modelscope.cn
- **操作**: 
  1. 登录账号「根深叶茂」
  2. 进入 Skills Center -> 发布技能
  3. 填写技能信息（参考 `community/modelscope_skill_release.md`）
  4. 添加标签: `Intel AI PC` `OpenVINO` `NPU` `语音质检` `合规风控`
  5. 上传提交包 `voiceguard_submission_v4.zip`
- **截止**: 路演前 (2026-09-18)

### 2. ModelScope 开发者实践文章发布
- **平台**: https://www.modelscope.cn
- **操作**:
  1. 登录账号「根深叶茂」
  2. 发布文章 -> 粘贴 `community/modelscope_article.md` 内容
  3. 添加标签: `Intel AI PC` `OpenVINO` `端侧AI`
  4. 附带 Demo 视频 `output/voiceguard_real_demo_v2.mp4`
- **截止**: 路演前 (2026-09-18)

### 3. 比赛报名 / 作品提交
- **平台**: Intel Connection 赛事官网
- **操作**:
  1. 完成赛事报名（如尚未报名）
  2. 上传提交包 `voiceguard_submission_v4.zip`
  3. 确认提交内容完整
- **截止**: 2026-09-10 24:00（如已过期需确认延期）

### 4. 路演准备
- **材料**: 路演/VoiceGuard路演.pptx (10页)
- **Demo**: output/voiceguard_real_demo_v2.mp4
- **时间**: 2026-09-22~23 苏州

## 估分预测

| 状态 | 估分 | 说明 |
|------|------|------|
| 当前 (代码+测试+Demo+PPT) | 73-76 | P1 全部完成, P2 全部完成 |
| + 魔搭 + ModelScope 发布 | 78-80 | P0 发布完成 |
| + 比赛报名提交 | 80-82 | P0 全部完成 |
| + 路演发挥 | 85-87 | 全部补齐 |

## 关键技术数据

- NPU ASR: 4.17x 加速 (Encoder 234M + CTC 12.8M 双组件, RTF=0.030)
- GPU LLM: 1.9x 加速 (batch=10, 29.76s) / Ollama Vulkan 4.4x (20.8s, 12.0 tok/s)
- 规则引擎: 3ms, 0 tokens, 80% 候选过滤
- 94 单元测试全通过, 5 场景 E2E 全通过, 4 平台 100% 成功
