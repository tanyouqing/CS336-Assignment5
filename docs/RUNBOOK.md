# 单卡运行手册

以下命令在 Linux 租卡机器执行，工作目录为 assignment5-alignment。
Windows 开发阶段不运行 pytest、训练或 GPU smoke tests。

## 1. 环境与路径

准备 Python 3.12、可用的 NVIDIA 驱动、24–32GB 显存，以及本地课程模型和数据。
模型目录和数据目录通过参数传入；不要把 access token 写进配置或日志。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按 PyTorch 官方安装选择器安装与租卡环境兼容的 CUDA 版本。
python -m pip install -e '.[single-gpu]'
bash scripts/check_linux.sh
```

默认 SDPA，不要求 FlashAttention/vLLM/Modal。50系显卡必须实测
Torch/CUDA/bitsandbytes 组合；安装成功不等于量化 forward/backward 可运行。
`uv.lock` 若尚未同步新增 extra，应在 Linux 用 `uv lock` 刷新；不要使用
旧锁的 `--frozen` 来声称已经安装单卡依赖。测速会保存环境 freeze。

## 2. 先测速，不启动整套实验

```sh
python -m cs336_alignment.cli benchmark \
  --model /models/OLMo-2-0425-1B \
  --sft-model /models/Meta-Llama-3.1-8B \
  --sft-data /data/course/train.jsonl.gz \
  --sft-val-data /data/course/test.jsonl.gz \
  --gpu-hours 0.5 --output runs/pilot
```

总计算上限30分钟；串行子进程有外部 watchdog。网络下载时间单独记录。
查看 `benchmark_report.json`、各阶段 `console.log` 和 `metrics.jsonl`。
第一个更新视为预热；没有足够成功更新时吞吐量为 null，不编造估计。
如果模型/数据未给齐，只测具备资源的流程。
报告包含25批次RL和精简SFT/DPO的近似费用推算，按小批次线性外推，
不包含全部验证、哈希和存盘成本，不能当成实际账单或固定性能保证。

出现 OOM：先保持算法和上下文不变，确认 microbatch=1、generation batch=1，
再检查 checkpointing 和 attention 后端。仍不够时，用新配置减少长度；
不要把不同长度的结果直接当成公平算法对比。只用失败日志确认问题，
不会循环自动尝试更多付费任务。

## 3. RL

```sh
# 先检查10个rollout batch；运行到50步时，从last恢复。
python -m cs336_alignment.cli rl --config experiments/rl_single_gpu.json \
  --model /models/OLMo-2-0425-1B --steps 10 --output runs/grpo
python -m cs336_alignment.cli rl --config experiments/rl_single_gpu.json \
  --model /models/OLMo-2-0425-1B --steps 50 --resume runs/grpo/last --output runs/grpo

# 一个采样批次做32次更新，group优势仍在完整batch上计算。
python -m cs336_alignment.cli rl --variant grpo --importance gspo \
  --train-batch-size 2 --microbatch-size 1 --cliprange 0.0003 \
  --steps 25 --output runs/gspo

python -m cs336_alignment.cli rl --variant dr_grpo --steps 25 --output runs/dr-grpo
python -m cs336_alignment.cli rl --variant loo --steps 25 --output runs/loo
```

`--steps` 是总目标 rollout batch 数，不是恢复后再增加的数量。
`--max-seconds` 在安全边界合作停止；单次训练 update 和 checkpoint 写入
可能超出该时间。严格上限使用 benchmark 或 matrix 的 watchdog。
硬超时中断时选择含 `COMPLETE.json` 的 checkpoint；不完整目录不可恢复。

`--backend vllm` 是原版双卡路线，需要额外 gpu dependencies、训练 GPU
和不同编号的 inference GPU；默认HF路线不调用 NCCL 同步。

## 4. SFT → DPO

```sh
python -m cs336_alignment.cli sft --config experiments/sft_single_gpu.json \
  --model /models/Meta-Llama-3.1-8B \
  --train-data /data/course/train.jsonl.gz --val-data /data/course/test.jsonl.gz

python -m cs336_alignment.cli dpo --config experiments/dpo_single_gpu.json \
  --model /models/Meta-Llama-3.1-8B --adapter runs/sft/last

python -m cs336_alignment.cli dpo --config experiments/dpo_single_gpu.json \
  --model /models/Meta-Llama-3.1-8B --adapter runs/sft/last --resume runs/dpo/last
```

SFT checkpoint 是 adapter，不是8B完整模型；重加载时仍需要相同 base model。
DPO 的 `--adapter` 永远指初始 SFT checkpoint；`--resume` 指更新过的 DPO
checkpoint。这两个参数的含义不同。

DPO 首先预计算冻结reference；训练与验证样本全部评分后才开始训练。
不完整reference缓存可继续补齐，fingerprint不一致时明确拒绝复用。
reference缓存包括完整模型/adapter文件摘要和实际token摘要；摘要生成也会耗时。
本地模型文件特别大时优先保留缓存，避免频繁重启导致重复哈希。

日志里的 implicit reward accuracy 使用
`(logπ(chosen)-logπ(rejected)) - (logπref(chosen)-logπref(rejected)) > 0`，
不是裸policy概率比较。报告中必须写清定义。

训练超时会保存last；非有限loss或失败更新不会覆写有效checkpoint。
resume校验训练语义、数据选择和优化器；不要从不可信来源加载training_state.pt。
允许RL增加总steps，但不允许静默更换数据、优化器、学习率或epoch数量。

## 5. 评估

```sh
python -m cs336_alignment.cli eval --task rl --prompt r1_zero \
  --model /models/OLMo-2-0425-1B --output runs/rl-baseline
python -m cs336_alignment.cli eval --task mmlu --data data/mmlu \
  --model /models/Meta-Llama-3.1-8B --qlora --stage base --output runs/base-mmlu
python -m cs336_alignment.cli eval --task gsm8k --stage sft \
  --model /models/Meta-Llama-3.1-8B --adapter runs/sft/last --qlora --output runs/sft-gsm8k
python -m cs336_alignment.cli eval --task alpaca --stage dpo --limit 100 \
  --data data/alpaca_eval/alpaca_eval_gpt4_turbo.json \
  --model /models/Meta-Llama-3.1-8B --adapter runs/dpo/last --qlora --output runs/dpo-alpaca
python -m cs336_alignment.cli eval --task safety --stage dpo --limit 0 \
  --data data/simple_safety_tests/simple_safety_tests.csv \
  --model /models/Meta-Llama-3.1-8B --adapter runs/dpo/last --qlora --output runs/dpo-safety
```

MMLU/GSM8K生成时使用greedy decoding；主RL评估使用temperature=1。
保持各模型相同seed、limit、max tokens。Base使用system wrapper；SFT/DPO使用
Alpaca wrapper。Base/SFT/DPO 输出保留误例、解析失败和吞吐量。

Alpaca输出为JSON数组，Safety输出为JSONL。两者未判分时status=pending，
winrate/safe比例保持null。需要原版AlpacaEval时，使用仓库现有70B annotator
配置配合 `alpaca_outputs.json`；这需要额外资源，不由matrix自动执行。
`judge-safety` 提供可选本地裁判入口，其分数必须注明裁判模型及解析失败数。

## 6. 清单、预算、图表

```sh
# 只生成清单，不运行。先确认模型、SFT和HH路径。
python -m cs336_alignment.cli matrix --output runs/plan \
  --model /models/OLMo-2-0425-1B --sft-model /models/Meta-Llama-3.1-8B \
  --sft-data /data/course/train.jsonl.gz --sft-val-data /data/course/test.jsonl.gz

# 只有明确给出总GPU小时上限才执行。按priority串行，保留失败/partial状态。
python -m cs336_alignment.cli matrix --output runs/plan \
  --manifest runs/plan/manifest.json --execute --budget-gpu-hours 2

python -m cs336_alignment.cli summarize --runs runs --output output/summary
```

预算包含推理、训练、验证及checkpoint子进程耗时，不包含租卡创建流程。
清单中未启动的 estimated_gpu_hours 保持null；用pilot实测估算后再安排预算。
不会自动租卡，不会提交外部收费评估请求。

汇总CSV和PNG只来自真实metrics；每个种子的曲线单独呈现。
断点恢复写入同目录时日志可能包含重复step，分析时以最后记录为准。
`data_seed`固定数据子集，`seed`控制模型、采样和训练随机性；
比较不同训练seed时保持data_seed不变，避免同时改变验证集。
不要把旧配置输出目录用于新的实验；生成清单后修改路径要另建清单。

## 7. 磁盘与提交

RL全参数checkpoint包含模型与AdamW状态，可能远大于纯权重；定期checkpoint
频率可以用 `--checkpoint-every` 控制。SFT/DPO只保存adapter与adapter优化器。
模型、训练日志与结果默认不进Git；不要把runs或reference缓存塞进code.zip。
更新后的打包脚本先运行两份作业与额外测试，测试失败即停止；
同时排除runs、output和.cache，避免提交日志、缓存和大模型。

最后补全 REPORT_TEMPLATE.md；没有真实实验的项目继续写“未运行”，不要估造
准确率、70B裁判分数、显存占用或训练耗时。
