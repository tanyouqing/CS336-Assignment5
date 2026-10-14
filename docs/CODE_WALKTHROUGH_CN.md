# Assignment 5 代码导读：用 PyTorch 串起后训练 pipeline

本篇从**当前项目如何运行**出发：一个命令进入哪个函数、数据怎样变成 tensor、谁负责 forward/backward、参数何时更新、模型怎样保存与继续训练。算法概念只在解释相关代码时引入。

这是一份仓库导读，不是另一套独立示例工程。标为“源码节选”的片段来自当前文件；标为“流程展开”的片段为便于阅读省略了日志、检查及异常处理，不能直接替代原函数。路径均相对于 `assignment5-alignment/`。旧版[概念教程](POST_TRAINING_TUTORIAL_CN.md)保留作为补充。

**验证边界**：当前教程经过源码核对；本文不声称训练已实测成功。Linux原始测试、数值smoke与GPU训练仍待执行，HTML中的执行步进是教学模拟，不会加载模型。

## 1. 文件结构：先分清入口、计算与基础设施

```text
assignment5-alignment/
├── cs336_alignment/
│   ├── cli.py                  # 命令行、配置合并、任务分发
│   ├── runtime.py              # 加载模型、生成、日志、保存与恢复
│   ├── data.py                 # JSONL、抽样、划分、packing、数据解析
│   ├── core.py                 # 主RL的tensor计算与一次optimizer更新
│   ├── rl.py                   # 主RL外层循环：采样→奖励→训练→验证
│   ├── dpo.py                  # 偏好样本token化、序列评分、DPO loss
│   ├── supplement.py           # SFT/DPO训练循环、reference缓存与验证
│   ├── evaluation.py           # 生成评估、答案解析、指标与裁判接口
│   ├── experiments.py          # 限时pilot、实验清单、日志汇总
│   ├── drgrpo_grader.py         # 数学奖励函数
│   ├── prompts/                # 主作业的三种prompt
│   ├── prompts_safety/         # 补充任务与Alpaca模板
│   ├── vllm_utils.py           # 可选双卡推理及权重同步
│   ├── checkpoint.py           # 原有加载工具；新CLI主要使用runtime.py
│   └── modal_utils*.py         # 原课程部署工具；本地单卡主线无需调用
├── tests/
│   ├── adapters.py             # 保留作业接口，转接到实现
│   ├── test_grpo.py            # 原始主作业测试
│   ├── test_data.py            # 原始packed数据测试
│   ├── test_dpo.py             # 原始DPO测试
│   ├── test_metrics.py         # 原始解析测试
│   ├── test_invariants.py      # 累积、mask、reference、恢复等性质
│   └── test_training_smoke.py  # 小模型SFT恢复与DPO流程
├── experiments/*.json         # 精简及原版运行配置
├── scripts/check_linux.sh     # Linux验证入口
├── docs/                      # 本教程、运行手册、报告模板
└── runs/                      # 运行后产生；默认不纳入Git
```

阅读时把模块分成三层：

| 层 | 模块 | 负责什么 | 不负责什么 |
|---|---|---|---|
| 编排层 | cli、rl、supplement、experiments | 先做什么、做多少次、何时验证 | 不在CLI里写token级数学 |
| 计算层 | core、dpo、data | tensor、mask、loss、样本结构 | 不决定整套实验预算 |
| 运行层 | runtime、evaluation | 模型对象生命周期、I/O、生成与指标 | 不改变算法baseline |

主RL与SFT/DPO**共用模型加载与log-prob评分基础设施**，但并不硬合成同一个万能训练循环。RL的数据在循环中生成；SFT与DPO数据是离线固定的；它们需要不同的状态与更新语义。

建议首先同时打开 [cli.py](../cs336_alignment/cli.py)、[rl.py](../cs336_alignment/rl.py)、[supplement.py](../cs336_alignment/supplement.py)，先找函数调用关系，再向下读计算细节。

`tests/adapters.py`只是课程测试与实现之间的薄转接层，主CLI不从它进入训练。主作业7个接口都接到core：tokenize、log-probs、rollout奖励、分组优势、policy loss、microbatch聚合、train step；补充的packed dataset/batch迭代/MMLU解析/GSM8K解析接到data，单样本DPO loss接到dpo。这样保留课程签名，同时正式训练可以增加缓存与外层编排。

## 2. 从一条命令追到训练函数

示例命令在Linux执行，路径是待替换的占位符：

```sh
python -m cs336_alignment.cli rl \
  --config experiments/rl_single_gpu.json \
  --model /models/OLMo-2-0425-1B \
  --train-data /data/gsm8k/train.jsonl \
  --val-data /data/gsm8k/test.jsonl \
  --steps 10 --output runs/rl-first
```

`python -m` 把包中的 `cli.py` 作为入口，执行底部 `main()`。它不开始重新构建Transformer，后训练使用已有预训练模型。

### 2.1 config为什么不会被argparse默认值覆盖

`main()` 的合并顺序是：**COMMON → 任务默认值 → JSON → 显式命令行参数**。`argparse.SUPPRESS` 让未提供的参数不出现在解析结果中。

源码节选：

```python
config = COMMON | specific
if config_path:
    supplied = json.loads(open(config_path, encoding="utf-8").read())
    config.update(supplied)
config.update(arguments)
```

`|` 合并字典，右侧覆盖同名key；`update()` 原地修改。这样JSON中的 `steps=50` 不会被一个未显式填写的CLI默认值意外改掉，但 `--steps 10` 会覆盖JSON。

`main()` 还检查未知配置key、正数参数、variant、prompt等。配置通过后，才按任务导入相应模块：

```text
rl          → train_rl(config)
sft / dpo   → config['kind']=command → train_supplement(config)
eval        → evaluate(config)
benchmark   → benchmark(config)
matrix      → matrix(config)
summarize   → summarize(config)
```

把任务导入放进分支，能减少未使用流程的依赖干扰；vLLM也只在对应backend分支导入。

### 2.2 三个batch参数各管哪段代码

| 参数 | 使用位置 | 默认主RL含义 |
|---|---|---|
| rollout_batch_size=64 | 外层采样 | 本轮生成64个回答 |
| train_batch_size=64 | 外层切训练片段 | 一次optimizer更新使用64个回答 |
| microbatch_size=1 | 内层forward/backward | 每次前向放1个回答 |
| generation_batch_size=1 | generate循环 | 同时生成1个回答 |
| group_size=4 | prompt重复、奖励reshape | 每题生成4个回答 |

主RL的 `steps` 是rollout轮数；SFT/DPO的 `steps` 是optimizer更新次数上限。不要把同名配置误认为完全相同的计数单位。

## 3. runtime.load_model：已有模型如何变成可训练对象

阅读 [runtime.py 的 load_model](../cs336_alignment/runtime.py)。它返回两个核心对象：

```python
model, tokenizer = load_model(...)
```

tokenizer负责字符串与token id；model是PyTorch `nn.Module` 风格的对象，包含 `parameters()`、`train()`、`eval()` 和可微 `forward()`。HF提供模型架构/预训练权重、tokenizer、generate；本项目自己组织数据、计算loss、累积梯度、优化与保存，不使用Trainer。

### 3.1 主作业：普通BF16模型

源码节选：

```python
options = {
    "torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
    "attn_implementation": attention,
}
model = AutoModelForCausalLM.from_pretrained(model_path, **options)
if not qlora:
    model.to(device)
```

`**options` 把字典展开成关键字参数。`from_pretrained` 创建并加载权重，`.to(device)` 把参数搬到GPU。主RL优化器接收 `model.parameters()`，默认进行全参数训练。

### 3.2 补充作业：量化base + 可训练adapter

QLoRA分支先配置4-bit NF4、double quantization与BF16计算，再 `prepare_model_for_kbit_training()`。接着有两条分支：

```text
有adapter路径 → PeftModel.from_pretrained(..., is_trainable=train)
无adapter且QLoRA训练 → LoraConfig(r=16, alpha=32, dropout=0,
                               target_modules='all-linear') → get_peft_model
```

base权重被冻结，adapter参数可训练。`build_optimizer()`只选择 `p.requires_grad` 为真的参数。因此“同样调用backward”并不意味着优化的参数集合相同。

模型adapter不会代替基础模型架构和权重；重加载时需要base路径。DPO的 `--adapter` 指SFT初始化/reference，而 `--resume` 指已训练的DPO状态。

### 3.3 开始训练前统一设置

开启non-reentrant activation checkpointing；把 `nn.Dropout.p` 设为0；`model.train()`；训练 `use_cache=False`。这些设置使评分较稳定并控制显存。

`train()`与`eval()`只改变某些层的行为，**不负责开关autograd**。是否建立计算图由 `no_grad()` 和tensor/参数的 `requires_grad` 决定。冻结base仍允许激活路径传导到adapter，不能给policy的整个训练forward加no_grad。

## 4. 所有循环共用的评分器：字符串→labels→log_probs

现在进入 [core.py](../cs336_alignment/core.py)。先理解评分基础，再回到训练循环。

### 4.1 tokenize_prompt_and_output输出合同

输入是两个等长字符串列表，输出一个tensor字典：

```text
input_ids      LongTensor [N,T]
labels         LongTensor [N,T]
response_mask  BoolTensor [N,T]
```

它分别encode prompt/response、拼接、右补齐，再进行移位：

```python
return {
    "input_ids": ids[:, :-1],
    "labels": ids[:, 1:],
    "response_mask": mask[:, 1:],
}
```

若 `[P0,P1,R0,R1]` 是完整序列，input是 `[P0,P1,R0]`，labels是 `[P1,R0,R1]`，mask是 `[0,1,1]`。labels不是再送给模型的另一段上下文，而是用于从模型输出分布中挑出待评分token。

<!-- LAB:tokens -->

接口不自动添加BOS/EOS。PAD若复用EOS id，mask仍由长度/边界构造，不能通过token值推断。评分中右padding位于真实前缀后方，因果注意力使真实位置看不到后方padding；loss显式排除这些位置。生成的左padding则通过attention mask处理。

### 4.2 get_response_log_probs怎样调用PyTorch

源码中的关键路径：

```python
logits = model(input_ids).logits                  # [N,T,V]
logp = F.log_softmax(block.float(), dim=-1)       # [N,C,V]
selected = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
```

`C`是T的一个分块，默认最大128。`model(input_ids)`调用模型forward；因果注意力保证每个位置只能看到此前内容。`log_softmax`变成log概率；`gather`沿词表轴挑labels对应的分数，输出 `[N,C]`，最终 `torch.cat(...,dim=1)` 合并为 `[N,T]`。

这里没有把labels交给HF自动算loss；项目后面要自行做response mask、RL权重或DPO序列求和。

有梯度时，分块评分通过 `checkpoint(score_block,...,use_reentrant=False)`，让反向时重算FP32 softmax；否则仅把forward写成循环，仍可能保留所有分块softmax图。**模型的完整logits仍已生成**，此实现没有消除 `[N,T,V]` 的全部内存。

训练forward的评分是可微tensor；用于日志的熵在no_grad中计算。保存日志时才 `detach()/item()`，若先把loss变成Python float就无法backward。

## 5. train_rl外层：一次rollout如何产生训练材料

阅读 [rl.py 的 train_rl](../cs336_alignment/rl.py)。这个函数初始化model、AdamW、log、训练/验证样本、variant与deadline。检查训练/验证题目不交叠，并在resume时核对数据指纹。

主循环的前半段可展开为：

```python
# 流程展开：省略检查、vLLM和日志
rows = random.choices(training, k=64 // 4)
prompts = [template.format(question=r["question"]) for r in rows for _ in range(4)]
answers = [gsm8k_answer(r) for r in rows for _ in range(4)]
responses = completions(prompts)
rewards, reward_stats = compute_rollout_rewards(reward_fn, responses, answers)
advantages, _ = compute_group_normalized_rewards(
    rewards, group_size, baseline, advantage_normalizer=normalizer
)
```

`random.choices`是有放回抽样；并非每轮不重复地走完dataset。先抽16道题，再每题重复4次，因此顺序是 `题0的4个回答、题1的4个回答……`。后续reshape依赖这个顺序。

### 5.1 generate期间没有训练图

`runtime.generate` 用 `@torch.no_grad()`，记录此前的mode/cache/padding；切 `eval`、左padding和 `use_cache=True`，再调用 `model.generate()`。默认数学RL temperature1、top_p1、top_k0；取生成id的prompt宽度之后部分，decode为字符串。

`finally`恢复此前状态，即使超时/失败也不能让下一次训练误用生成设置。默认单卡HF在**同一个model对象上交替生成和训练**，不是驻留第二份推理模型。

当前训练重新tokenize生成字符串来评分，old/new都比较这些重新编码的token；并非直接保存生成时原始token id。decode/re-encode若不完全可逆，重新评分序列可能与原采样id存在差异，这是需要实测关注的边界。

### 5.2 grader只是普通函数

`math_reward()`选择 `drgrpo_grader.py` 的grader，返回字典中 `reward/format_reward/answer_reward`。`compute_rollout_rewards` 把总奖励收集成CPU FloatTensor `[64]`，不把文本判断放进autograd。

`compute_group_normalized_rewards` 做 `[64]→[16,4]`，按每题求baseline/normalizer，再flatten回 `[64]`。完整组上的优势必须先算好，再传给内层训练。

<!-- LAB:advantages -->

`VARIANTS`把算法差别表示为 `(baseline, normalizer, aggregation)`。这避免为每个变体复制训练loop。`importance`另控制概率比与裁剪；variant叫grpo不意味着importance一定启用了PPO裁剪。

### 5.3 为什么先缓存old，再做任何更新

只要importance不是none，就将完整rollout token化，再调用 `old_scores()`：no_grad分microbatch评分、转CPU、cat回 `[64,T_global]`。缓存是在本轮更新前计算的，后面每个train batch共享同一旧策略语义。

切局部batch时，它的padding宽度可能小于全局宽度，因此使用：

```python
old_local = old[start:end, : local["input_ids"].shape[1]]
```

这里裁掉的是全局补齐的尾部列；不能随意重排样本或token，否则old不再与labels对应。默认train64产生1次更新；off-policy train2产生32次更新，旧分数与完整组优势在这32次中固定。

## 6. core.grpo_train_step内层：真正改变参数的地方

外层调用 `grpo_train_step(..., precomputed_advantages=advantages[start:end])`，它负责**一次optimizer更新**，不是完整epoch。

### 6.1 先准备固定权重与活跃索引

函数取得当前batch回答数n、奖励统计、token tensors。外层预计算优势已提供时直接使用；没有提供时才自行求组优势。它仍重算该局部batch奖励供日志使用，并不重建完整分组优势。

源码节选：

```python
optimizer.zero_grad(set_to_none=True)
active = advantages.nonzero().flatten()
for start in range(0, len(active), size):
    indices = active[start : start + size]
    inputs = batch["input_ids"][indices].to(device)
    labels = batch["labels"][indices].to(device)
    mask = batch["response_mask"][indices].to(device)
```

`nonzero()`选出优势不为零的回答；`flatten()`得到一维索引。CPU上的数据切片后再 `.to(device)`，把GPU驻留规模控制在microbatch。零优势的loss/梯度贡献本来为零，跳过只为了节省评分；原n仍用于归一化。

### 6.2 forward→loss→backward

源码节选：

```python
scored = get_response_log_probs(model, inputs, labels, True)
per_token, info = compute_policy_gradient_loss(...)
loss = aggregate_loss_across_microbatch(
    per_token, mask, loss_normalization, normalization_constant
)
if loss_normalization == "sequence":
    loss = loss * (len(indices) / n)
loss.backward()
loss_sum += loss.detach()
```

shape依次为：`input/labels [m,T] → logits [m,T,V] → log_probs [m,T] → per_token [m,T] → loss []`。`[]`表示标量tensor。`backward()`累加到参数的 `.grad`，此时还没有修改参数。

无importance时：

```python
advantages = raw_rewards_or_advantages.detach().reshape(-1, 1).to(policy_log_probs)
return -advantages * policy_log_probs, {}
```

`.to(policy_log_probs)`匹配评分tensor的device/dtype；`[m,1]`通过broadcast乘每个回答的T个位置。优势固定，梯度沿log-probs返回模型。

### 6.3 聚合分母为什么属于完整训练batch

sequence先对每回答的有效token求平均，再对microbatch求平均；乘 `m/n` 使多个microbatch相加等价于原batch平均。constant直接对mask后的loss求和再除 `train_batch_size×max_tokens`，不能再次除累积次数。

例如64条只有16条活跃，microbatch1：sequence每条仍乘1/64，而不是1/16。这样过滤不会改变目标。

### 6.4 最后才step

```python
if len(active):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

先做全部microbatch backward，再裁剪完整梯度并step一次。全零优势时跳过step。真实代码还检查loss与梯度范数是否有限，失败则清梯度并抛异常，不把坏更新当正常状态保存。

`set_to_none=True`把grad设为None，区别于写满零tensor；下一次backward重新累积。日志 `gradient_norm`记录clip函数返回的裁剪前范数，不能把它误认为裁剪后的范数。

## 7. 在同一循环中切换noclip、GRPO、GSPO

阅读 `compute_policy_gradient_loss()`，关注的不是新循环，而是 `[m,T]` loss如何变化。

```text
none：  −A × current_log_prob
noclip：−A × exp(current_log_prob − detached_old_log_prob)
grpo：  −min(A×ratio, A×clamp(ratio,1−eps,1+eps))
gspo：  先对response内log-ratio求均值，再exp为序列ratio，使用同样surrogate
```

old和优势都detach；current不detach。noclip的exp本身就有梯度，所以不再额外乘log-prob。

GSPO使用response mask排除prompt/padding，序列比为 `[m,1]`，再扩展到 `[m,T]` 供同一个聚合函数使用。把它与constant聚合组合会引入长度权重，不应当作与原序列平均版本完全一样。

<!-- LAB:clip -->

默认on-policy `importance=none`；若在同一初始模型上缓存old，第一次forward的比率为1，但之后更新的batch会偏离。这里是复用本轮rollout的切片，不是每个切片再训练32个epoch。

RL外层的 `step += 1`在完整rollout的全部片段更新结束后执行。因此日志要同时看 `step` 与 `optimizer_steps`：off-policy默认每rollout是32次optimizer更新，on-policy为1次；同样25个rollout未必同样更新成本。

## 8. SFT的数据对象：Dataset并不直接等于batch

阅读 [data.py 的 PackedSFTDataset](../cs336_alignment/data.py) 与 [supplement.py 的SFT准备分支](../cs336_alignment/supplement.py)。数据记录的合同是 `{'prompt': str, 'response': str}`。

流程是：读原始JSONL → 去重/按prompt隔离验证 → 固定抽样 → 保存selection → train/val分别packing。

`PackedSFTDataset`继承PyTorch Dataset，只有两项关键协议：`len(dataset)`与 `dataset[index]`。初始化时对Alpaca全文+EOS编码，拼成CPU LongTensor一维token流。

源码节选：

```python
def __len__(self):
    return max(0, (len(self.tokens) - 1) // self.seq_length)

def __getitem__(self, index):
    start = index * self.seq_length
    return {
        "input_ids": self.tokens[start : start + self.seq_length],
        "labels": self.tokens[start + 1 : start + self.seq_length + 1],
    }
```

一个item输出两个 `[512]` 张量。相邻块从token流连续取样，labels多偏移1；不足完整context的token尾巴被舍弃，但最后不足有效batch的完整块仍参与训练。

课程接口 `iterate_batches()`返回 `DataLoader(..., drop_last=False)`，满足通用批量接口。**当前正式SFT循环并没有调用它**，而是手动构造索引和stack，为了明确维护epoch/cursor恢复状态。

本作业packed SFT优化完整文本，包括prompt与模板；没有response mask。EOS作为文档边界，不会自动让attention在文档间隔离。初始化采用tokenizer默认special-token行为，与主RL不插BOS的接口不同。

## 9. SFT训练循环：从packed item到一次AdamW更新

`train_supplement(config)`在 `kind='sft'` 下准备packed dataset，按真实块数计算计划更新量：

```python
total_steps = math.ceil(len(training) / config["train_batch_size"]) * config["epochs"]
optimizer, scheduler = build_optimizer(model, config, total_steps)
```

`build_optimizer`过滤trainable参数，创建AdamW（或配置指定的RMSprop），再用 `LambdaLR` 构造约3%warmup与cosine schedule。这里计划基于全epoch步数；`--steps`只限制执行数量，并未把计划调度长度改成该上限。

### 9.1 order与cursor保证批次可重建

源码节选：

```python
order = list(range(len(training)))
random.Random(config["seed"] + epoch).shuffle(order)
indices = order[cursor : cursor + config["train_batch_size"]]
```

每个epoch用固定seed+epoch重建顺序，再按cursor切当前batch。packed文档流自身在创建时按seed打乱；这里打乱的是已packing块的索引，而不是每epoch重新packing全文。

### 9.2 stack、移到GPU、loss和梯度

```python
# 源码节选：selected由当前microbatch索引获得
batch = {
    key: torch.stack([r[key] for r in selected])
    for key in ("input_ids", "labels")
}
loss = sft_loss(model, batch) * (len(selected) / len(indices))
loss.backward()
```

`torch.stack`把m个 `[512]` 加一个batch维变为 `[m,512]`；不是沿序列拼接成 `[m×512]`。`sft_loss`内部把input/labels移到model device，再调用共用评分器：

```python
return -get_response_log_probs(
    model, batch["input_ids"].to(device), batch["labels"].to(device)
)["log_probs"].mean()
```

数学上等于这些固定长块的next-token交叉熵。每块长度相同，所以按块数加权microbatch平均也等价于整体token平均。如果以后改成变长块，就需要重新审查这个权重。

### 9.3 最后不足batch怎么处理

有效batch32、最后只剩5块时，indices长度是5；microbatch2会切2、2、1，分别乘2/5、2/5、1/5。不是仍除32，也不是每个microbatch都乘1/3。

最后执行裁剪trainable梯度 → optimizer.step → scheduler.step → zero_grad → cursor增加实际样本数 → step增加1。日志的learning_rate取scheduler.step之后值，即通常是下一更新将使用的学习率。

SFT的step单位是optimizer更新；默认2048是原始文档数，packed块数取决于文本长度，不能从2048直接推断2048/32步。

## 10. DPO的准备阶段：先把reference变成固定数据

SFT和DPO共用 `train_supplement` 的状态管理，但DPO准备分支完全不同。

```text
load_model(base + 原SFT adapter)
  → load_hh()：过滤单轮、相同prompt、去重、train/val分离
  → reference_cache(train + val)
  → 若resume：释放原SFT模型，加载DPO resume adapter
  → build_optimizer → restore_state → 开始DPO更新
```

HH loader输出 `{'prompt','chosen','rejected','source'}`。过滤多轮文本后只接受同一个Human prompt的两个回答；原始chosen/rejected整段对话不能直接当两个response。

### 10.1 reference_cache不是第三个训练器

它先调用 `reference_identity()`：绑定base/adapter文件摘要、tokenizer信息、实际token与mask、样本、最大长度等。若已有cache指纹不匹配，报错而非静默复用。

每条样本no_grad评分后存为普通Python数值：

```text
reference.json
  fingerprint: …
  metadata: …
  scores:
    row_hash:
      chosen: 一个序列log-prob
      rejected: 一个序列log-prob
```

每32条flush一次，finally也flush，用临时文件replace缓存；部分完成可以补齐。训练与验证样本都要缓存，但验证只评分、不backward。

### 10.2 初始化、reference与resume是三个角色

首次DPO直接用已评分的SFT模型开始更新，因此初始policy=reference。resume时先加载原SFT补缓存，再 `del model`、`empty_cache()`，最后加载DPO状态。`empty_cache()`释放空闲缓存，不会自动释放仍被引用的活跃tensor。

原SFT reference在整个训练中固定；不能把resume后的policy当reference，也不能关闭adapter用base替代SFT。缓存让训练只驻留一个量化模型，不需同时保留两个8B forward对象。

## 11. DPO训练：一对回答如何合成可微loss

阅读 [dpo.py](../cs336_alignment/dpo.py)。正式循环调用三个函数：`preference_tokens → score_preference → cached_dpo_loss`。

### 11.1 构造chosen/rejected两个batch字典

`preference_tokens`分别构造Alpaca prompt prefix与response tokens，二者共享完全相同的prefix id。默认最大长度512，prefix最多保留约一半的末尾，response保留开头并补EOS。

```python
# 源码节选：位于每个chosen/rejected分支
ids = prefix + output
result[key] = {
    "input_ids": torch.tensor(ids[:-1]).unsqueeze(0),
    "labels": torch.tensor(ids[1:]).unsqueeze(0),
    "mask": torch.tensor([False] * (len(prefix) - 1) + [True] * len(output)).unsqueeze(0),
}
```

`unsqueeze(0)`给单个序列增加batch维，变为 `[1,T_side]`；chosen与rejected长度可以不同，无需一起padding。mask同样对齐labels，第一个response token在最后一个prefix输入位置被预测。

截断prompt尾部可能丢掉开头指令信息，截断response可能改变偏好含义；这是精简配置的取舍，不应当作无损预处理。

### 11.2 score_preference：同一个policy做两次forward

```python
score = get_response_log_probs(model, input_ids, labels)["log_probs"]
result[key] = (score * mask).sum()
```

chosen/rejected各得到一个标量tensor；它们都保留计算图，随后共同影响共享模型参数。只累加回答部分（含EOS），不做长度平均。

### 11.3 cached_dpo_loss：reference没有计算图

源码节选：

```python
ref_margin = torch.as_tensor(reference["chosen"] - reference["rejected"], device=device)
margin = (scores["chosen"] - scores["rejected"]) - ref_margin
return -F.logsigmoid(beta * margin), margin.detach()
```

缓存是Python数值，转成tensor不自动具有reference梯度。返回的loss保留policy图；第二个返回值detach用于指标。因此不需要为缓存reference维护一个optimizer。

<!-- LAB:dpo -->

### 11.4 DPO的真实microbatch行为

当前训练分支**始终一次处理一对偏好样本**；不是读取 `microbatch_size` 把多对padding到一起。每对做chosen/rejected两次forward，然后：

```python
loss = loss / len(indices)
loss.backward()
loss_sum += float(loss.detach())
```

循环完有效batch16对再裁剪、step、scheduler.step。最后不足16时用实际 `len(indices)`。改变 `--microbatch-size`目前不会让DPO多对并行；若希望更高吞吐，需要另行实现pair collator与一致的mask/聚合。

### 11.5 作业测试接口为什么看起来不同

`tests/adapters.py`中的DPO接口转接 `compute_per_instance_dpo_loss`：传入policy与reference对象，对完整Alpaca文本+EOS编码并求和，reference包在no_grad中。这满足单样本接口路径，但正式单卡训练使用response-only缓存路径。

若chosen/rejected具有相同token prefix，全文prompt项能在各自差值中抵消；整体编码与分开编码的边界token可能不同，不能未经检查就宣称这两条路径完全逐位相同。**正式缓存与在线policy训练使用相同的preference_tokens定义**，这是缓存正确性的关键。

## 12. 评估、日志与保存：训练循环的后半段

RL每完整rollout记录train事件，按配置保存rollout样本、周期checkpoint、validation生成。SFT/DPO在optimizer更新后记录train，周期验证并选择best；结束保存last。

### 12.1 训练内验证不等于benchmark生成评估

| 函数 | 实际做什么 | 选择best依据 |
|---|---|---|
| validate_sft | teacher-forcing验证packed块平均NLL | 更低validation_loss |
| validate_dpo | 验证偏好loss与相对reference margin | 更高implicit_reward_accuracy |
| evaluate_rl | 生成数学回答后grader评分 | 主RL周期流程没有同样的best选择逻辑 |
| evaluate | 加载指定checkpoint生成MMLU/GSM8K等 | 保存结果，按任务解析 |

`validate_sft/dpo`使用eval/no_grad并在finally恢复mode。DPO implicit reward accuracy看 `margin>0`，不是 `policy_chosen>policy_rejected`；初始化等于reference时严格大于0不成立，不能把这个起点值误认为模型完全不会选择chosen。

补充生成评估的base用zero-shot wrapper；SFT/DPO用Alpaca wrapper。MMLU解析字母、GSM8K解析数值；Alpaca/安全集未自动裁判时只存输出且标未判分，不虚构指标。

### 12.2 RunLog如何工作

`RunLog`写config.json，在metrics.jsonl追加事件；每条记录有event、elapsed_seconds和任务指标，可选WandB。CUDA峰值通过 `max_memory_allocated/reserved`记录，是自最近reset以来的累计峰值，不是每step独立瞬时显存。

RL重点看采样耗时、更新耗时、rollout_reward、active_responses、token_entropy、clip_fraction。活跃样本过滤后的熵不代表完整rollout，零活跃时0只是占位。SFT/DPO关注loss、验证指标、gradient_norm、learning_rate与实际batch_examples。

## 13. checkpoint与resume：要恢复的是训练状态

`runtime.save_checkpoint`先移除完成标记，依次保存模型/adapter、tokenizer、训练状态与配置，最后写 `COMPLETE.json`。训练state包含：

```text
step + optimizer.state_dict + config
Python / NumPy / Torch / CUDA RNG
extra:
  RL → data_fingerprint
  SFT/DPO → epoch、cursor、best、scheduler.state_dict、data_fingerprint
            reference_fingerprint（DPO）
```

完整目录不是一次原子事务；marker用于拒绝读取未完成状态。`training_state.pt`使用pickle恢复复杂RNG对象，只加载可信的本地checkpoint。

**model权重加载与optimizer恢复是两步**。主RL先 `load_model(resume or model)`，再 `restore_state`；adapter训练先加载base+对应adapter，再创建optimizer和载入其state。scheduler随后从extra恢复。

SFT/DPO以seed+epoch重建order，再从cursor继续；当前dataset在固定seed下重建并核对数据指纹。RL在完整rollout边界保存step与随机状态。改变数据、学习率等训练语义会被拒绝；提高主RL总steps属于允许操作。

异常路径有区别：超时通常保存可恢复的完整边界；非有限loss/失败更新不会用可能部分修改的policy覆盖有效checkpoint。`--max-seconds`是合作式时间限制，不能硬中断一个耗时forward/step或保存；pilot/matrix另用进程watchdog。

## 14. 如何亲手跑通并检查这条pipeline

下面是教学执行顺序，所有模型/训练命令在Linux运行。完整环境要求与实测状态见 [RUNBOOK.md](RUNBOOK.md)。

### 14.1 先验证计算，再租卡跑pilot

```sh
bash scripts/check_linux.sh
```

原始四组测试检查课程接口；额外invariants检查累积分母、零优势过滤、GSPO mask、完整分组、DPO固定reference/cache和checkpoint恢复；smoke用小模型检查真实参数更新以及恢复与不中断流程的一致性。测试文件存在不等于测试已通过，当前尚无Linux执行证据。

### 14.2 三条路线最短命令链

```sh
# RL：steps是rollout轮数
python -m cs336_alignment.cli rl --config experiments/rl_single_gpu.json \
  --model /models/OLMo-2-0425-1B --steps 10 --output runs/rl-first

# SFT：先用少量optimizer更新检查流程；JSON中的课程路径须填写
python -m cs336_alignment.cli sft --config experiments/sft_single_gpu.json \
  --model /models/Meta-Llama-3.1-8B \
  --train-data /data/course/train.jsonl.gz --val-data /data/course/test.jsonl.gz \
  --steps 3 --output runs/sft-first

# DPO：首次包含reference评分，不只有3次更新的成本
python -m cs336_alignment.cli dpo --config experiments/dpo_single_gpu.json \
  --model /models/Meta-Llama-3.1-8B --adapter runs/sft-first/last \
  --train-data /data/hh --steps 3 --output runs/dpo-first

# 生成评估：base与adapter都要正确指定
python -m cs336_alignment.cli eval --task gsm8k --stage dpo --qlora \
  --model /models/Meta-Llama-3.1-8B --adapter runs/dpo-first/last \
  --data /data/gsm8k/test.jsonl --limit 32 --output runs/dpo-eval
```

上面的3步SFT模型只用于流程学习，不足以视为正式SFT基线。正式训练前按计划执行总计算上限30分钟pilot；矩阵命令先生成清单，不在预算未明确时一键全部训练。

### 14.3 观察这些文件，而不是只看“程序没报错”

| 输出 | 应查看什么 |
|---|---|
| config.json、metrics.jsonl的start | 实际模型、batch、attention、量化、软件版本 |
| selection/split_manifest | 固定样本是否符合计划，训练验证是否分离 |
| rollouts-*.jsonl | prompt、回答、标准答案、reward、A是否一致 |
| train事件 | loss有限、更新次数单位、有效样本数、耗时 |
| validation事件及生成样本 | loss与真实能力是否一致，解析失败有哪些 |
| last/COMPLETE.json、training_state.pt | 最近可恢复状态，step/cursor是否正确 |
| reference.json | 原SFT身份、样本与token配置是否匹配 |

### 14.4 三个最有价值的源码断点

1. `grpo_train_step`评分后：看inputs/labels/mask的shape与前几个位置，log_probs是否有grad_fn，A/old是否固定。
2. `loss.backward()`之后、step之前：看trainable参数grad是否非空且有限；QLoRA base不应更新，adapter应获得梯度。
3. `optimizer.step()`之后：检查参数确实变化，随后保存/重载再比较相同输入输出。loss下降不能替代这个验证。

先降低microbatch/生成并发控制显存，再审查logits、激活与KV cache；更改context/算法需另存配置。24–32GB是否够用以实测为准。本仓库没有训练critic/reward model，也没有分布式DDP循环，不应在阅读时把不存在的组件补进执行图。

## 15. 执行步进与进一步修改的路线

配套HTML可以按“下一步”沿RL、SFT、DPO调用链行进，观察数据对象、梯度状态与更新边界。它只展示上述真实函数组织的教学状态，不执行Python或产生训练证据。

<!-- LAB:pipeline -->

如果你想亲手接管代码，按以下顺序做小改动，每次只改一个层面：

| 想做什么 | 首先改哪里 | 必须保持什么 |
|---|---|---|
| 增加主RL prompt | prompts文件、cli允许列表、grader选择 | 生成与评估模板一致 |
| 增加优势估计器 | core分组函数、rl.VARIANTS | 完整组计算后切片、无梯度权重 |
| 更改RL聚合 | aggregate_loss_across_microbatch、train_step权重 | 多microbatch等价、过滤保留分母 |
| 改SFT模板 | alpaca_sft.prompt与data.sft_text | 重新packing、重新核对token与评估wrapper |
| 增大DPO并行 | pair collator、score_preference、supplement循环 | chosen/rejected mask正确、参考缓存同定义 |
| 更改DPO截断 | preference_tokens | reference指纹/缓存更新，说明偏好变化 |
| 增加评估任务 | evaluation的加载、prompt与解析分支 | 原始输出可审查，失败不可伪造为正确 |

<details><summary>练习1：指出一次RL更新里哪些tensor要有梯度。</summary>

current log_probs、policy loss与最终标量loss要保留图。奖励、优势、old_scores不求梯度。input_ids/labels是整数索引；不是需要优化的连续参数。
</details>

<details><summary>练习2：SFT的Dataset、microbatch、有效batch分别在哪里形成？</summary>

Dataset在data.PackedSFTDataset；当前有效batch由order[cursor:cursor+train_batch_size]形成；microbatch由selected切片并torch.stack组成，多个backward后只step一次。
</details>

<details><summary>练习3：改DPO microbatch_size为4，会自动四对并行吗？</summary>

不会。当前DPO分支按index逐对处理，每对chosen/rejected各forward一次。需要实现新的pair batching路径才会改变并行度。
</details>

<details><summary>练习4：恢复DPO时adapter与resume应该分别指向哪里？</summary>

adapter指原始SFT checkpoint，供初始化与固定reference；resume指已更新DPO checkpoint。先用原SFT核对缓存，再加载DPO继续优化。
</details>

<details><summary>练习5：为什么loss日志要detach，而实际loss不能先float？</summary>

日志不需要保留计算图，否则可能拖住显存。backward必须在可微tensor上执行；float/item会把数值与autograd图断开。
</details>

<details><summary>练习6：为off-policy设置train_batch2，25个steps究竟是多少次更新？</summary>

默认rollout64切为32个片段，每片段一次更新；25个完整rollout计划为800次optimizer调用，零优势片段可能跳过实际step。
</details>

<details><summary>练习7：如何证明缓存reference与在线reference一致？</summary>

用相同冻结SFT、tokenizer、样本与preference_tokens在线评分，再比较缓存chosen/rejected分数；同时确认无梯度、改变token配置会拒绝旧缓存。这需要数值测试而非静态阅读。
</details>

<details><summary>练习8：从第一轮日志定位程序究竟在采样、缓存还是更新。</summary>

RL看sampling_seconds/training_seconds；DPO首先reference_scoring，随后dataset/train；SFT直接准备dataset后train。结合elapsed与原始console，不把首次模型加载/缓存时间当稳定更新吞吐。
</details>

读完后，建议自己画出 `命令 → config → model/data → forward → loss → backward → step → validate/save`，并为每个箭头标注具体文件与函数。这比记住算法名更接近独立维护一条后训练pipeline。

页面由 `node scripts/build_tutorial.cjs --implementation` 从本Markdown生成，可通过MARKED_MODULE指定marked包目录。旧教程与新教程互不覆盖。
