# Assignment 5 实验报告模板

状态：实验未运行。所有占位数字保持空白。

## 环境和范围

GPU / 显存：待填写。Python / CUDA / PyTorch / PEFT / bitsandbytes：待填写。
完整软件版本来自config和environment-freeze.txt。
模型、tokenizer、adapter、训练/验证划分、随机种子和总GPU耗时：待填写。

声明单卡精简实验与原版差异，包括QLoRA、数据规模、种子数量和裁判暂缓。

## 主作业

### 三种 prompting

| prompt | 正确 | 格式正确但判错 | 格式错误 | 总reward | 吞吐量 |
|---|---|---|---|---|---|
| question_only | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| r1_zero | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| three-shot | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

人工检查各类别回答；格式正确但判错至少观察10例。解析误判数量与具体例子：待填写。
三种prompt行为差异与支持案例：待填写。

### 标准 GRPO 与消融

先给出短训练正确性证据。附loss、grad norm、entropy、train/val total和format
reward、val平均回答长度、训练前后回答例子。多种子图保留每次运行，不只画均值。
25%最终平均准确率是否达到：尚未实测，不声称达到。

学习率扫描与prompt消融：待填写。控制参数与种子数：待填写。
GRPO/GRPO_constant/Dr.GRPO/RFT/MaxRL比较：待填写。
Offpolicy naive/noclip/clip/GSPO比较，含clip fraction和wall clock：待填写。

### 自定义 leave-one-out

唯一改动是Dr.GRPO的baseline。给出组大小有限时的推导，说明与group-mean
版本的缩放关系，实验不预设更好。结果与局限：待填写。

### 理论题

依次完成：baseline_calcs、think_about_length_normalization、think_about_rft、
derive_difficulty_reweightings、think_about_advantage_normalization、
derive_surrogate_objectives、think_about_importance_reweighting。

## 补充作业

SFT随机10例的任务与质量分析：待填写。HH各类随机案例与偏好分析：待填写。
SFT训练设置、learning curve、验证loss、保存checkpoint：待填写。
DPO冻结reference、cache fingerprint、beta、loss、隐式reward accuracy：待填写。

| stage | MMLU | GSM8K | Alpaca winrate | Safety比例 |
|---|---|---|---|---|
| base | 未运行 | 未运行 | 裁判暂缓 | 裁判暂缓 |
| SFT | 未运行 | 未运行 | 裁判暂缓 | 裁判暂缓 |
| DPO | 未运行 | 未运行 | 裁判暂缓 | 裁判暂缓 |

附生成吞吐量、解析失败、各任务10个误例分析；未自动判分的任务只做人工观察。
Red-teaming：记录三种潜在误用、测试过程、投入时间和结果；不将其等同于完整安全评估。
Alignment tax与自动裁判局限：待填写。

## 验证证据

Linux原始测试：待执行。额外不变量测试：待执行。checkpoint恢复：待执行。
单卡forward/backward、峰值显存、30分钟pilot：待执行。
