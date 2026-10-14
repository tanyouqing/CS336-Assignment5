# 作业覆盖与验证状态

本表记录代码覆盖，不把静态解析当成测试通过。当前 Windows 实现阶段：
Python AST/TOML/JSON 静态检查已执行；Linux pytest、模型训练、CUDA 兼容性和
GPU 显存测试尚未执行。所有数值正确性与性能结论以之后的真实日志为准。
Ruff检查与格式检查通过。当前Windows的WSL不可用，未安装Linux环境来绕过该限制。

| 内容 | 代码入口 | 验收证据 / 当前状态 |
|---|---|---|
| 主作业 7 个接口 | core.py、tests/adapters.py | 原始 test_grpo.py，待 Linux 验证 |
| 三种 prompting 基线 | cli eval --task rl | 输出回答、奖励类别、吞吐量，待运行 |
| 标准 GRPO / 四个 on-policy 变体 | cli rl --variant ... | loss、熵、梯度范数、奖励，待运行 |
| 四种 off-policy 变体 | cli rl --importance ... | 固定旧 log-probs、clip fraction，待运行 |
| 自定义估计器 | cli rl --variant loo | 仅 baseline 改为 leave-one-out，待运行 |
| 学习率、prompt、多种子实验 | cli matrix | 清单覆盖，未启动 |
| Packed SFT 与 batching | data.py、tests/adapters.py | 原始 test_data.py，待验证 |
| MMLU/GSM8K 解析 | data.py | 原始 test_metrics.py，待验证 |
| SFT / DPO 训练与缓存 | cli sft、cli dpo | 小模型恢复、缓存一致性，待验证 |
| DPO per-instance loss | dpo.py、tests/adapters.py | 原始 test_dpo.py，待验证 |
| 四项补充评估 | cli eval | MMLU/GSM8K评分；Alpaca/Safety 保存生成 |
| 安全裁判 | cli judge-safety | 接口完成；解析失败独立计数，自动运行暂缓 |
| 原版 AlpacaEval 裁判 | 仓库既有 annotator 配置 | 需额外 70B 推理资源，暂缓 |
| 提交打包 | test_and_make_submission.sh | 测试失败停止，排除实验输出和缓存 |
| 实验报告 | REPORT_TEMPLATE.md | 数字、曲线、错误案例留待实测填写 |

## 精简配置与讲义差异

- RL 保留原 OLMo 1B，默认单模型 HF 采样，使用 1024/256 train/val、G=4、
  64条回答、短训练与主要单种子；讲义默认 6400/1024、G=8、256条回答、
  200批次和4种子。`rl_original.json` 与 `matrix --profile original` 保留原规模。
- 单卡 SFT/DPO 保留 Llama 3.1 8B，但采用 QLoRA；讲义训练完整参数。
  固定 SFT 2048样本、HH每来源最多512训练样本，使用 adapter 学习率。
- DPO reference 是冻结 SFT checkpoint 的缓存，并非原始 base model。
  微批为一个 pair；缓存复用避免第二份8B模型驻留。
- 补充评估使用固定缩减 MMLU/GSM8K/Alpaca 子集，Safety全量。
  Base/SFT/DPO 使用相同 seed 选择同一批样本，不混用主作业的 GSM8K parser。
- 自动70B裁判暂缓；没有 winrate/safe-proportion 实测时保持 null。
- 不保证短实验达到25%准确率；只报告实测值。
- 单种子曲线不足以说明方差或统计显著性；原版4种子配置应另行运行。

## 尚需运行的检查

`bash scripts/check_linux.sh`：原始测试和额外不变量测试。
随后执行30分钟测速，再按预算执行训练。未经执行的检查不得改成“已通过”。

租卡之前准备课程模型、SFT train/validation 的本地路径；代码不替你申请账号、
租用硬件或获取课程共享盘权限。

新增单卡extra未能在当前离线环境更新uv.lock：旧gpu extra引用的FlashAttention
wheel不在缓存。Linux优先按RUNBOOK使用pip单卡安装；之后有网络时再更新锁。
