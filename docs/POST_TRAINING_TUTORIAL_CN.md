# Assignment 5：从数据到梯度的后训练实现教程

这份教程解释本仓库主作业与补充作业的实际实现。目标是让你能沿着数据流读代码、检查一次更新是否正确，并判断单卡实验的结果能说明什么。

**阅读约定**：公式中的例子是教学计算，不是真实训练结果。当前完成的是实现与静态检查；Linux 数值测试、真实 GPU 吞吐、准确率与显存仍需验证。已有运行命令见 [RUNBOOK.md](RUNBOOK.md)，覆盖情况见 [REQUIREMENTS_STATUS.md](REQUIREMENTS_STATUS.md)。

## 1. 先建立整体地图

两份作业是两条学习路线，并不是必须顺次训练同一个模型。

```text
主作业：OLMo 1B → 生成数学答案 → 可验证奖励 → GRPO 等策略梯度更新
补充：Llama 8B → 指令数据 SFT → 人类偏好 DPO → 能力与安全评估
```

| 阶段 | 数据中有什么 | 优化信号 | 本仓库位置 |
|---|---|---|---|
| 预训练 | 连续文本 | 预测下一个 token | Assignment 1 的背景 |
| SFT | 指令与示范回答 | 示范 token 的交叉熵；本作业要求完整文本 loss | `data.py`、`supplement.py` |
| 可验证奖励 RL | 题目、标准答案，回答由当前模型产生 | 答案/格式奖励转为策略梯度 | `rl.py`、`core.py` |
| DPO | 同一 prompt 的 chosen/rejected | 相对固定 SFT 参考策略，提高偏好回答的相对概率 | `dpo.py`、`supplement.py` |

SFT 学“示范怎么写”；数学 RL 学“哪些自己生成的答案得到奖励”；DPO 学“两个候选中哪一个更被偏好”。奖励和偏好只是训练信号，不自动等于所有场景中的真实性、安全性或用户满意度。

传统 RLHF 常包含偏好数据训练奖励模型、再用 PPO 优化策略。本补充作业直接用 DPO 优化离线偏好，不实现这条完整 PPO-RLHF 链。主作业使用可验证数学奖励，也没有学习一个奖励模型。

**建议读法**：先读 2–5 节弄清 token 与梯度，再读 6–8 节理解算法差异，最后读 9–14 节理解工程、资源与验证。打开配套 HTML 可以调整例子参数。

## 2. 所有目标的共同底座：token 的条件概率

语言模型输出 `logits[N,T,V]`：N 是样本数，T 是输入长度，V 是词表大小。logits 是未归一化分数，不是概率。

对词表维做 `log_softmax`，得到每个候选 token 的 log-probability。用真实的下一个 token 索引 `gather`，便得到 `log_probs[N,T]`。这里的“真实”在 SFT 中来自示范，在 RL 中来自刚刚采样的回答。

```python
logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
selected = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
```

`unsqueeze(-1)` 把 `[N,T]` 变成 `[N,T,1]`，使每个时间位置只选择一个词表元素；`squeeze(-1)` 恢复 `[N,T]`。`.float()` 让归一化计算使用 FP32，避免低精度造成不必要的数值误差。

自回归模型的回答概率是条件概率的乘积：

```text
p(y|x) = ∏t p(yt | x, y<t)
log p(y|x) = Σt log p(yt | x, y<t)
```

例如两个 token 概率为 0.5、0.2，回答概率为 0.1，序列 log-prob 为 `log(0.5)+log(0.2)=log(0.1)`。DPO 使用这样的**求和**，不能随意换成 token 平均；平均会改变优化目标。

训练评分使用 teacher forcing：给模型完整的已知序列，用因果注意力让位置 t 只看到此前内容，然后并行评分每个 next token。生成则必须逐 token 采样。两者的模型相同，执行方式不同。

### 熵不是答案正确率

分布熵 `H=-Σv pv log pv` 衡量一个位置的预测分布有多分散，不是所选 token 的 `-log p`，也不是数学题正确率。低熵可能意味着自信，也可能意味着错误模式固化。

`get_response_log_probs` 的熵用于日志，在 `no_grad` 下计算，当前没有熵正则项。经过零优势过滤后，训练日志的熵只覆盖实际评分的活跃样本；没有活跃样本时的 0 是日志占位，不能解读为模型分布熵为 0。

## 3. 最容易出错的一步：移位与 mask

假设 prompt 是 `[P0,P1]`，回答是 `[R0,R1,EOS]`：

| 位置 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| input_ids | P0 | P1 | R0 | R1 |
| labels | P1 | R0 | R1 | EOS |
| response_mask | 0 | 1 | 1 | 1 |

**在最后一个 prompt 输入位置 P1，就要预测第一个回答 R0。** mask 对齐 labels，不是未移位的 input_ids。否则容易漏掉回答首 token 或把 prompt token 计入 RL loss。

实际代码先拼接 `prompt_ids + response_ids`，再使用 `ids[:,:-1]` 与 `ids[:,1:]`，response mask 同样取 `mask[:,1:]`。主作业接口不额外插入 BOS/EOS，以上 EOS 只用于说明“若序列中包含它时”的对齐。

<!-- LAB:tokens -->

三种不同的 mask 要分清：

1. 因果 mask：当前位置不能看到未来 token。
2. padding/attention mask：哪些位置只是补齐，不应作为有效上下文。
3. loss/response mask：哪些 next-token 预测要计入优化目标。

主作业评分采用右 padding，真实 token 前缀不会看到其右边的 padding；loss 用明确构造的 response mask 排除补齐位置。生成采用左 padding，并传递 tokenizer 生成的 attention mask。即使 PAD 和 EOS 共用 id，也不能只靠“token id 是否等于 EOS”判断一个位置是不是 padding。

分开编码 prompt 和 response，有助于固定边界，但并不保证与一次性编码拼接字符串得到完全相同的 token；某些 tokenizer 会跨文本边界合并字符。这也影响 DPO 接口与训练路径的比较，见第 8 节。

## 4. SFT 与策略梯度：为什么符号决定更新方向

SFT 的单 token loss 是 `-log pθ(target|context)`。最小化它会增加目标 token 的概率。注意，优化器执行的是梯度下降，所以 loss 前面的负号不能漏掉。

RL 目标则是提高期望奖励：

```text
J(θ) = Ey~πθ [r(y)]
∇J = Ey~πθ [r(y) ∇log πθ(y)]
```

推导来自 `∇p = p∇log p`，不需要奖励函数对模型可微。因此可以对生成的文本用普通 Python 判答案，获得数值奖励，再通过模型重新评分这些 token 来建立梯度图。

换成最小化形式：`L = -A × log πθ(y|x)`。A 为正时增加该回答概率；A 为负时降低。主作业按回答给一个 A，再广播到回答 token：`A[N] → A[N,1] → token_loss[N,T]`。

```text
采样文本（无梯度） → 判答案得到 r（无梯度） → 计算 A（固定数值）
                                                   ↓
重新前向评分（有梯度） → log_probs → loss → backward → 参数更新
```

`detach()` 表示把数值当作固定权重；`no_grad()` 表示整个评分过程不建梯度图。当前策略评分必须保留梯度；旧策略分数、参考策略分数、奖励和优势都不应成为反向传播对象。

给整个回答相同优势，是一种粗粒度归因：它并不知道推理中具体哪一步正确。不能把“token 都收到了梯度”解释为奖励已经准确定位了每个推理步骤。

## 5. 分组优势：同一道题的多个回答互相比

记 B 为题目数、G 为每题回答数、N=B×G。默认一次 rollout 有 64 个回答，G=4，所以是 16 道题，**不是 64 道题**。

将奖励 `[N]` reshape 成 `[B,G]`，在每一行内统计：

```text
mean_g = (Σi ri)/G
std_g = sqrt(Σi (ri−mean_g)²/(G−1))
GRPO advantage_i = (ri−mean_g)/(std_g+1e−6)
```

代码用 PyTorch 默认的样本标准差，分母是 G−1。对于 `[1,0,0,1]`，均值 0.5、标准差约 0.57735，优势约 `[0.866,-0.866,-0.866,0.866]`。

如果一组全对或全错，减均值后都为 0，该组没有相对区分信号。epsilon 避免除零，但不能凭空制造信号。监控活跃样本数、奖励均值和奖励分布，才能区别“已学会”和“全部失败”。

<!-- LAB:advantages -->

### 为什么必须先算完整组，再切训练 minibatch

off-policy 默认把 64 个回答分成 32 个训练 batch，每批 2 个回答。若在这 2 个回答里重新求组均值，就改变了 baseline，甚至拆散同一道题的组。当前实现先在完整 rollout 上计算全部 A，随后切片训练。

### 自定义 LOO baseline 的实际含义

```text
b_−i = (Σj≠i rj)/(G−1)
Ai_LOO = ri−b_−i = G/(G−1) × (ri−mean_g)
```

G=4 时，LOO 是减组均值优势的 4/3 倍。在本实现中 LOO 和 Dr. GRPO 都不除标准差、都用常数归一化，所以差别主要是优势缩放，而非新的排序信号。AdamW、梯度裁剪等让“缩放优势”不必严格等价于“缩放学习率”。不要预设它会更好。

## 6. 两层归一化与梯度累积

奖励归一化改变 A；loss 归一化改变不同长度、不同 microbatch 的权重。二者不是一回事。

记 M 是 response mask、li 是回答有效长度。序列平均：

```text
L_sequence = (1/N) Σi [Σt Mit Lit / li]
```

每个回答整体权重相同。常数归一化：

```text
L_constant = Σi,t Mit Lit / (N × Lmax)
```

默认 Lmax=512。较长回答包含更多 token 项，所以在相同每 token loss 下贡献更大。它消除了每条回答 `1/li` 的长度依赖，但不意味着所有长度具有同样权重。

| rl.py 中 variant | baseline | 优势归一化 | loss 归一化 |
|---|---|---|---|
| grpo | 组均值 | 标准差 | 序列平均 |
| grpo_constant | 组均值 | 标准差 | 常数 |
| dr_grpo | 组均值 | 无 | 常数 |
| rft | 无 | 无 | 常数 |
| maxrl | 组均值 | 组奖励均值加 epsilon | 常数 |
| loo | leave-one-out | 无 | 常数 |

variant 与 importance_reweighting 是两组配置：前者控制优势/聚合，后者控制 none、noclip、grpo、gspo 的策略 loss。名字相似，不能只看一个字段推断完整算法。

### 累积梯度必须还原同一个目标

显存放不下 N 个回答时，可以多次 forward/backward 后只做一次 optimizer.step。序列平均的每个 microbatch loss 应乘 `microbatch样本数 / 原训练batch样本数`。

例如 5 个样本切成 2、2、1，权重是 0.4、0.4、0.2，而不是三个 batch 都除以 3。常数归一化的各片段直接使用同一完整分母，再相加；不能再额外除累积次数。

过滤零优势样本只减少计算，**必须保留过滤前的 N**。若 64 个回答只留下 16 个，还把分母改成 16，就把更新放大 4 倍。全零时跳过 optimizer.step，避免只由优化器状态造成参数变化。

顺序是：清梯度 → 多次 backward → 一次梯度范数裁剪 → 一次 step → 清梯度。梯度范数裁剪约束本次梯度大小，与下面 PPO 的概率比裁剪完全不同。

## 7. 旧策略、重要性比与 PPO/GSPO 裁剪

采样策略生成固定回答后，连续更新会让训练策略与它不一致。重要性比比较同一个 token 在当前与采样时策略下的概率：

```text
ρit = exp(log πθ(yt|context) − log πold(yt|context))
```

旧 log-probs 必须在任何更新前记录，整个 rollout 的多次更新中固定不变。它不是上一 microbatch 的当前分数，也不是 DPO 的 SFT reference。

| 路径 | token 级最小化目标 | 说明 |
|---|---|---|
| none / naive | `−A log πθ` | 不校正旧数据分布 |
| noclip | `−A ρ` | 比率本身已有梯度，不再乘 log-prob |
| grpo | `−min(Aρ, A clip(ρ,1−ε,1+ε))` | token 级 PPO surrogate |
| gspo | 同样的 clipped surrogate，ρ 改为序列比 | 使用回答内平均 log-ratio |

ε=0.2、A=+1、ρ=1.4：目标收益取 min(1.4,1.2)=1.2，继续增大比率不再改善这个 surrogate。A=−1、ρ=0.6：取 min(−0.6,−0.8)=−0.8，继续压低也不再改善它。

相反，A=+1、ρ=0.6 仍有梯度；A=−1、ρ=1.4 也仍有梯度。**超出区间不一定意味着梯度被截断**，还取决于优势符号。裁剪也不是硬性保证最终概率比或 KL 永不超界。

<!-- LAB:clip -->

GSPO 使用：

```text
ρi_seq = exp(Σt Mit (new_logpit−old_logpit) / Σt Mit)
```

这是 token 概率比的几何平均，不是整个序列概率比的直接乘积；padding 不得进入平均。序列比扩展到各 token 后，配合序列平均得到序列级 loss。若搭配 constant 聚合，则又引入长度权重，不能称为完全相同的原版 GSPO。

默认 off-policy 设置 `rollout64 / train_batch2 = 32` 次更新，microbatch=1，所以每次更新累积两条回答。这里的 32 是把 rollout 分成 32 个训练 batch，不是每个 2 样本 batch 都再重复 32 个 epoch。较多更新不保证更省时间或效果更好，必须同时观察采样耗时、更新耗时和验证指标。

## 8. DPO：比较的是“相对参考策略的偏好差”

一条样本是 `(x, y_chosen, y_rejected)`。chosen 是偏好标签，并不意味着这条回答在任何指标上都更优。

```text
dθ = log πθ(chosen|x) − log πθ(rejected|x)
dref = log πref(chosen|x) − log πref(rejected|x)
m = dθ−dref
L_DPO = −log sigmoid(βm)
```

例子：策略分数 chosen=−4、rejected=−6，参考为 −5、−6，得到 dθ=2、dref=1、m=1。β=0.1 时 loss 约 0.6444。策略等于参考时 m=0，loss≈0.6931。

用 `F.logsigmoid` 避免先算 sigmoid 再取 log 的不稳定。导数 `dL/dm=−β sigmoid(−βm)`：正 margin 已很大时梯度变小；β 改变对 margin 的缩放，不能简单认为“β 越大总是学得越好”。DPO 推导中的 β 与 KL 正则有联系，但代码并没有显式计算一个 KL penalty。

<!-- LAB:dpo -->

本实现不除回答长度，EOS 也参与回答分数。长序列 log-prob 的和通常更负，这使长度与偏好可能相互作用；改为均值会改变 DPO 定义，不能作为无代价的“修复”。

### SFT reference 与 RL old policy 的区别

| 对象 | 来源 | 固定多久 | 是否求梯度 |
|---|---|---|---|
| RL old policy 分数 | 当前 rollout 生成时的策略 | 该 rollout 的全部更新 | 否 |
| DPO reference 分数 | 冻结的 SFT checkpoint | 整个 DPO 训练 | 否 |
| 当前 policy 分数 | 正在更新的模型 | 每次 forward 都重新算 | 是 |

DPO 从 SFT adapter 初始化。关闭 adapter 后得到的是 base model，**不是 SFT reference**。参考策略若跟着 DPO 更新，m 的意义就变了。

### 为什么可以先缓存 reference

固定数据、固定 reference、固定 tokenization 下，reference 分数每次相同。先加载 SFT 模型，用 eval/no_grad 评分并缓存，再只让 DPO policy 驻留显存，省掉第二个 8B 模型。

缓存绑定模型/adapter 权重指纹、tokenizer、prompt、token 序列、截断与相关配置。更换 SFT checkpoint、模板、最大长度、样本或 tokenizer 必须重算。恢复 DPO 时仍从原始 SFT checkpoint 检查/补齐缓存，然后再加载恢复的 policy。

**两个接口路径的区别**：测试用 `compute_per_instance_dpo_loss` 对完整 Alpaca 文本+EOS 评分；训练用 `preference_tokens/score_preference` 明确固定共享 prompt token，只累加 response mask。只有 chosen/rejected 的 prompt token 前缀完全相同时，完整序列中的 prompt 分数才会在差值中抵消。分开与整体编码可能产生边界差异，因此不能未经 tokenizer 检查就声称两条路径逐位完全一致。缓存与在线训练使用同一套训练 token 序列。

## 9. Packed SFT：把文档装满上下文

补充作业的 SFT 先把指令与回答组织成 Alpaca 文本，末尾加文档 EOS，再将多个文档 token 串接成一条流，切成固定 context 的训练片段。

```text
文档 A + EOS | 文档 B + EOS | 文档 C + EOS
             ↓ 串接、切块、next-token 移位
input[context] → labels[context]
```

收益是减少 padding、提高有效 token 比例。代价是一个块可能包含多个文档，文档也可能跨块。

本作业采用**完整文本语言模型 loss**，指令、模板和回答都计入 SFT loss；它不同于常见的 response-only instruction tuning。代码没有文档级 block-diagonal attention 或跨文档 position reset，因此 EOS 提供边界标记，但并不阻断后一篇文档注意到前一篇。不要把 EOS 理解为自动清空 attention。

若总 token 数为 K、context 为 C，可用完整块数量约为 `floor((K−1)/C)`；剩余不足一个 context 的尾部不进入当前 packed dataset。它与“最后不足一个 batch 的已成块样本仍应处理”是两件事。

数据必须先按原始样本划分 train/validation，再分别 tokenization 和 packing，否则同一文档或高度重合 prompt 可能同时出现在两边。偏好数据也按原始 prompt 保持训练与验证不重叠。默认抽样2048条 SFT 样本是文档数，不等于2048个 packed块。

## 10. QLoRA 与单卡显存：省的是哪部分

主作业 1B 模型默认全参数 BF16 训练；补充 8B 默认 4-bit NF4 QLoRA。两条路线保存、优化器和显存需求不同。

LoRA 对线性层使用：

```text
W_eff = W_frozen + (alpha/r) B A
W: [d_out,d_in]   A: [r,d_in]   B: [d_out,r]
```

r=16、alpha=32，所以缩放为 2。4096×4096 的一层，完整矩阵约1678万参数，LoRA 两矩阵共131072参数。示例只解释单层数量，不能直接当成整个模型训练参数比例。

QLoRA 把适用的基础权重量化保存，仍用 BF16 进行相应计算，并只优化 adapter。NF4 与 double quantization 用于减少存储。默认目标 `all-linear` 由 PEFT 对支持层处理，并不代表模型所有张量都是4bit。官方实现说明见 [PEFT 量化文档](https://huggingface.co/docs/peft/developer_guides/quantization)。

**冻结参数不等于关闭前向梯度**：基础权重不更新，但激活路径仍需传梯度给 LoRA。对整个 policy forward 加 no_grad 会导致 adapter 无法学习。

显存至少包含：基础权重、adapter、梯度、优化器状态、训练激活、logits、采样 KV cache、临时计算张量。`8B×0.5 byte≈4GB` 只是粗略的量化权重理想下界，不是总显存预测；量化元数据与部分非量化层也要空间。当前用普通 AdamW，不是自动启用 paged optimizer。

### 三种 checkpoint/cache 不要混淆

| 名称 | 保存什么 | 作用 |
|---|---|---|
| activation checkpointing | 只留部分前向激活，反向时重算 | 用计算换训练显存 |
| disk checkpoint | 权重/adapter、优化器、游标与 RNG | 恢复训练 |
| KV cache | 生成时过去 token 的 key/value | 加速自回归推理 |

训练关闭 KV cache；生成启用 KV cache。activation checkpointing 采用 non-reentrant 方式，具体机制见 [PyTorch 官方说明](https://docs.pytorch.org/docs/stable/checkpoint.html)。

`get_response_log_probs` 沿 T 分块做 FP32 softmax，并对分块评分 checkpoint 重算；否则仅分块但保留全部 softmax 反向图，仍可能累积很大的显存。原始 `[N,T,V]` logits 仍然存在，不能宣称评分已完全流式化。

BF16 的指数范围比 FP16 大，但不保证不会出现非有限 loss。降低 microbatch、生成并发与评估 batch 通常先于缩短上下文；后者会改变任务，应保存为另一配置再比较。40/50 系只是产品系列，是否支持当前 CUDA、PyTorch、bitsandbytes 后端必须在租用机器实测。

## 11. 一次更新、保存与恢复到底保存了什么

训练模型会关闭 dropout，减少同一序列评分的随机差异；生成会切 eval/no_grad，结束后恢复训练状态。HF 单卡方案让同一个模型交替采样和更新，省去同时驻留训练模型与独立 vLLM 模型的成本。原版双卡 vLLM 路径需要同步新权重。

SFT 默认有效 batch32、microbatch1；DPO 有效 batch16、microbatch1。有效 batch 表示一次 optimizer.step 汇总多少样本/块，不表示 GPU 同时放这么多。

SFT/DPO 使用带约3% warmup 的 cosine 调度；adapter 初始学习率分别2e−4、1e−5。它们是精简实验起点，不是全参数最优设置。主 RL 用 AdamW、lr1e−5、betas(0.9,0.95)、weight_decay0；不要把三条路线的学习率直接互换。

恢复训练不仅要 model weights，还需 optimizer state、scheduler、训练游标、随机状态、数据指纹与配置。只加载 adapter 更适合推理或新一轮训练，不等于从中断点继续。

运行状态记录 Python/NumPy/Torch/CUDA RNG；`data_seed` 控制固定抽样，`seed` 控制训练随机性。固定数据、换训练 seed 能减少数据子集差异的干扰，但仍不保证不同 GPU/kernel 上逐位可复现。

checkpoint 完成后才写 `COMPLETE.json`，防止把不完整目录当成可恢复状态；整个目录写入并不是一次原子事务。失败时保留先前有效 checkpoint。中途已发生但尚未保存的更新可能丢失，因此应区分“最近保存的完整状态”与“报错前最后一次 forward”。

## 12. 评估与预算：如何避免错误结论

主 RL 用数学 grader，分别报告 answer、format 与总奖励。补充 MMLU 解析选项字母，GSM8K 解析最后数值；解析器是评估协议的一部分，格式失误与知识错误应结合原始生成案例检查。

主 RL 默认随机采样 temperature1；补充评估默认 greedy temperature0。不同温度、prompt 或输出上限会影响分数，比较时要同时记录这些设置。训练 loss 更低并不保证验证准确率更高。

AlpacaEval 和 SimpleSafetyTests 当前可保存生成结果；自动裁判尚未完成时标记未判分，不填胜率/安全率。裁判返回非标准 true/false 是解析失败，不默认为安全。MMLU256/GSM8K256 的抽样结果也不能等同于全量 benchmark。

### 首轮资源清单

| 实验 | 默认规模 | 学习目的 |
|---|---|---|
| 标准 RL | 1024训练、256验证，先10后50 rollout batch | 验证采样→奖励→更新 |
| 算法与消融 | 各25 rollout batch、seed0 | 在相同预算内观察差异 |
| SFT | 2048文档、context512、一epoch | 观察指令模仿与packing |
| DPO | HH四类各最多512合格训练、另留200验证 | 观察相对偏好更新 |
| 补充生成评估 | Alpaca100，安全集全量生成 | 保留待裁判材料 |

30分钟 pilot 覆盖采样、RL更新、SFT、reference评分、DPO少量步骤，模型下载另记。调度只在明确预算下启动矩阵。吞吐需扣除预热影响；训练时间应拆成生成、更新、评估、缓存与保存，而非只用 optimizer.step 时间估算全部成本。

目前没有真实曲线、准确率或显存数据。25%数学门槛是待测目标，不能通过代码静态检查证明；单种子差异只能作为观察，不能据此断言算法优越性。

## 13. 代码阅读地图与验证证据

以下路径相对于 assignment5-alignment；除 `tests/adapters.py` 外，表中的模块均位于 `cs336_alignment/` 目录：

| 文件 | 阅读时要回答的问题 |
|---|---|
| `tests/adapters.py` | 12个作业接口接到哪一个实现？ |
| `core.py` | labels/mask 如何对齐？A/old 分数在哪里 detach？分母是什么？ |
| `rl.py` | rollout分组、旧分数与训练切片的先后顺序？ |
| `data.py` | 数据是否先划分？模板、EOS、packing及解析器怎么定义？ |
| `dpo.py` | chosen/rejected的response分数如何求和？reference在哪里固定？ |
| `supplement.py` | reference缓存、优化器与resume如何配合？ |
| `runtime.py` | train/generate模式切换、QLoRA加载、checkpoint有哪些状态？ |
| `evaluation.py` | 分数由什么生成和解析规则得出？哪些尚未判分？ |
| `experiments.py` | pilot如何限时？清单怎样估时与汇总真实日志？ |
| `cli.py` | JSON配置与命令行如何合并？ |

主作业7个接口：tokenize、log-probs、rollout rewards、group advantages、policy loss、microbatch聚合、train step。补充5个接口：packed dataset、batch迭代、MMLU解析、GSM8K解析、per-instance DPO loss。

**建议追踪顺序**：先拿一个prompt+两个token回答走完 `tokenize → score → loss`，再拿4个奖励走优势与聚合，最后进入 `train_rl` 看完整循环。不要一开始同时阅读所有参数解析与日志代码。

原始测试检查接口和数学行为；补充 invariant tests 检查累积、过滤、GSPO mask、完整分组、DPO reference/cache和恢复；小模型 smoke 验证参数更新与恢复。它们已经写入仓库，**尚未在Linux动态执行**。Windows仅做静态检查，静态通过不等于这些数值性质已被验证。

在Linux按 RUNBOOK 执行检查。遇到失败先区分环境导入失败、数值断言失败和GPU资源失败。不得修改预期快照来迁就错误实现。

## 14. 自测：能回答这些问题才算真正读懂

<details><summary>1. prompt有两个token，第一回答token在哪个input位置被预测？</summary>

在最后一个prompt位置。labels相对input左移一位，response mask对齐labels。
</details>

<details><summary>2. 奖励不可微，模型怎么收到梯度？</summary>

奖励作为固定优势权重，重新前向评分得到可微log-probs；梯度通过策略概率进入模型。
</details>

<details><summary>3. 全组奖励相同，是否应该用epsilon制造更新？</summary>

不应该。减均值后优势为零，epsilon只处理数值除零。应查看组奖励与活跃率。
</details>

<details><summary>4. 过滤零优势后为何不能改归一化分母？</summary>

过滤仅删除零贡献项。改变分母会改变原训练目标和有效更新尺度。
</details>

<details><summary>5. noclip为何不再乘一个log-prob？</summary>

exp(new−old)自身对当前log-prob有梯度。再乘会得到不同目标及额外导数项。
</details>

<details><summary>6. ρ超出[0.8,1.2]是否必然无梯度？</summary>

否。正优势抑制过大的比率，负优势抑制过小的比率；反方向仍有梯度。
</details>

<details><summary>7. 关闭DPO adapter可以得到reference吗？</summary>

不可以。reference是SFT模型，关闭adapter得到base。需要原SFT adapter或其固定分数缓存。
</details>

<details><summary>8. DPO margin为正，是否说明chosen绝对概率必然增加？</summary>

不能这样推断。margin比较相对参考的chosen/rejected差；共享参数与多个样本更新会耦合概率变化。
</details>

<details><summary>9. QLoRA冻结base后能给整个模型加no_grad吗？</summary>

不能。训练adapter仍需要经过基础层的可微激活路径。
</details>

<details><summary>10. EOS会阻断packed文档之间的attention吗？</summary>

当前不会。它只是token边界标记，代码没有文档级attention隔离。
</details>

<details><summary>11. 保存权重为什么不足以恢复训练？</summary>

还需优化器、调度器、游标、随机状态及一致的数据/配置，否则后续更新路径不同。
</details>

<details><summary>12. 静态检查通过、训练loss下降能证明算法正确且更好吗？</summary>

不能。还需数值测试、更新/恢复smoke、固定协议验证和预算匹配的实验；算法比较还需考虑多种子不确定性。
</details>

## 15. 延伸材料与出处

本教程的具体行为以本仓库代码和两份本地讲义为准。下面的原始材料帮助理解方法来源，不能替代本实现的验证。

- [主作业讲义](../cs336_spring2026_assignment5_alignment.pdf)、[补充讲义](../cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)：课程目标、接口与原版实验要求。
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)：分组相对优化的原始论文。
- [DPO](https://arxiv.org/abs/2305.18290)：从偏好数据直接优化策略的理论来源。
- [PEFT量化指南](https://huggingface.co/docs/peft/developer_guides/quantization)：量化模型上的adapter训练配置。
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)：重计算与non-reentrant实现的官方说明。

配套HTML由此Markdown生成，包含教学交互，不依赖外网或模型。修改正文后运行 `scripts/build_tutorial.cjs` 重新生成；该脚本需要Node与marked，具体依赖位置可通过 `MARKED_MODULE` 指定。
