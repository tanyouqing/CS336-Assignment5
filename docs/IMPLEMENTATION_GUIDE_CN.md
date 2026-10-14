# 从代码理解大模型后训练

本文配合本仓库 Spring 2026 作业代码阅读。先理解数据流，再对照函数。
当前实现尚待 Linux 数值测试和单卡实测；本文描述设计，不把设计当成实验结论。

## 1. 预训练之后，究竟改变了什么

预训练通常拿已有文本做 next-token prediction；SFT 拿指令和示范回答继续做
next-token prediction；数学 RL 只有问题、最终答案与验证器，回答由当前模型
自己生成；DPO 拿同一指令的两个回答和偏好标签，调整它们的相对概率。

| 阶段 | 数据 | 模型学习的信号 |
|---|---|---|
| 预训练 | 原始文档 | 真实下一个token |
| SFT | prompt + 示范response | 示范token的概率 |
| 数学RL | question + 可验证答案 | 自己生成的response是否答对 |
| DPO | prompt + chosen/rejected | 哪个回答更被偏好 |

主作业和补充作业是两个模型分支：OLMo 1B做数学RL，Llama 8B做SFT→DPO。
不是先训练Llama的DPO再把它接到OLMo的GRPO上。

## 2. 张量形状：从字符串到梯度

令B为问题数、G为每题回答数、N=B×G为回答数、T为移位后的padding长度、V为词表大小。

```text
B个问题 → 每题G条回答 → N个(prompt,response)
                     → input_ids/labels/response_mask [N,T]
input_ids → causal LM → logits [N,T,V]
labels    → gather   → log_probs [N,T]
grader               → rewards [N]
分组归一化            → advantages [N]
advantage × log-prob → per-token loss [N,T]
response_mask + aggregation → scalar → backward → optimizer.step
```

`labels`是input token序列右移一位后的预测目标。mask对齐labels，例如：
prompt是[a,b]、response是[c,d]，input=[a,b,c]、labels=[b,c,d]，mask=[0,1,1]。
首个response token由最后一个prompt token位置预测，所以mask不能照抄input区域。

主RL的prompt与response分别tokenize，不自动加BOS/EOS，再直接拼接。
补充SFT使用完整Alpaca文档加EOS，遵循fixture里的默认tokenizer特殊token规则。
这两套语义不能因为“共用tokenizer”就合并为一个规则。

## 3. 七个RL组件

### tokenize_prompt_and_output

按回答最长长度padding，返回三个[N,T]张量；只在labels的回答位置标True。
短回答的尾部padding不能参与loss。这里使用右padding，causal attention保证
前面的真实token不会看到后面的padding；生成时则使用左padding。

### get_response_log_probs

对每个位置的V维logits做log_softmax，再按label的token id gather：

$$\ell_{nt}=\log\pi_\theta(y_t\mid x,y_{<t}).$$

gather并不是求全词表平均，而是拿模型分配给“实际生成token”的概率。
返回值仍覆盖prompt与padding位置，后续用mask排除。

熵是整个next-token分布的熵：

$$H=-\sum_v p_v\log p_v.$$

熵作为日志不回传梯度，不等于额外添加entropy bonus。
评分按时间维分块，并checkpoint softmax计算；仅分块但保存所有softmax
反向图仍会占用整段FP32内存，所以反向时重算这些块。
模型的完整logits仍会驻留，microbatch=1不是可省略的限制。

### compute_rollout_rewards

给grader的ground truth是GSM8K的`####`之后内容，不是整条推理过程。
两种grader分别处理R1标签和boxed答案。训练reward只奖励正确答案；
format_reward记录格式遵循情况，不额外鼓励“会写标签但答错”。

### compute_group_normalized_rewards

先把rewards从[N]恢复为[B,G]，再按组计算。
标准GRPO使用：

$$A_{ij}=\frac{r_{ij}-\mu_i}{s_i+\epsilon}.$$

实现按讲义要求使用默认torch.std的样本标准差，即分母G−1；
讲义推导中的总体标准差写法与实现要求不同。
同组全对或全错时，group-mean优势为零，梯度没有贡献。
这不等于模型“全部学会了”：全错组也没有可区分的正样本。

### compute_policy_gradient_loss

On-policy逐token代理loss是：

$$L_{it}=-A_i\log\pi_\theta(y_t\mid x,y_{<t}).$$

负号把希望增大奖励的gradient ascent转为PyTorch的gradient descent。
reward/advantage视为采样后的常量，不对grader或组统计回传。
这个loss值本身不是稳定可比较的评估指标，不能要求它像交叉熵一样下降。

### aggregate_loss_across_microbatch

Sequence版本先在每条回答的有效token内平均，再在回答之间平均。
Constant版本把有效token loss求和再除以固定Z。

Sequence对应：

$$L=\frac1N\sum_i\frac1{L_i}\sum_t M_{it}L_{it}.$$

Constant对应：

$$L=\frac1Z\sum_{i,t}M_{it}L_{it},\qquad Z=N L_{\max}.$$

因此两者的微批缩放不同：sequence微批平均乘n_micro/N；
constant每个微批已经除完整Z，累加即可，不能再除累积次数。
筛掉零优势回答时也不能把N/Z改成剩余回答数。

### grpo_train_step

整批计算reward和advantage，然后筛出非零贡献的回答，逐微批反向传播。
最后只做一次梯度裁剪和optimizer step，并清空梯度为None。
熵和clip fraction默认统计实际评分的非零优势回答token；它们不是全量采样
分布的无偏估计。全量reward使用rollout_reward日志判断。

## 4. On-policy算法矩阵

| 方法 | baseline | advantage normalizer | token aggregation |
|---|---|---|---|
| GRPO | group mean | std | sequence |
| GRPO_constant | group mean | std | constant |
| Dr.GRPO | group mean | none | constant |
| RFT | none | none | constant |
| MaxRL | group mean | mean | constant |

RFT只训练正确回答；错误回答为0权重。Dr.GRPO会降低同组相对较差回答的
概率。MaxRL用group mean归一化，本作业的constant版本是受控变体，
不能直接说完全复现原论文。

新增loo版本只替换Dr.GRPO baseline：

$$b_{-j}=\frac{\sum_{k\ne j}r_k}{G-1},\quad
A_j^{LOO}=r_j-b_{-j}=\frac G{G-1}(r_j-\mu).$$

其他独立回答的均值不依赖当前回答，所以避免group mean包含自身的有限G
缩放；但在同一采样组上它也只是整体乘G/(G−1)。AdamW、epsilon和梯度裁剪
会影响这个缩放的实际效果，不应把结果提升预先解释成全新探索能力。

## 5. Off-policy：复用采样，但不能重新定义它的来源

一批64条回答，每个train batch只有2条，意味着32次optimizer更新。
采样的policy在第一个更新后已经过时。所有old log-probs都必须在更新之前
计算一次并固定，组优势也必须在完整64条回答上计算。

$$w_t=\exp(\ell_t^{new}-\ell_t^{old}).$$

noclip的loss是−A×w，不是−A×w×logp；对w微分已经产生logp梯度。
GRPO clipped objective：

$$L_t=-\min(Aw_t,A\operatorname{clip}(w_t,1-\epsilon,1+\epsilon)).$$

裁剪是否生效取决于优势正负，不能简单把所有区间外token都置零。
日志clip fraction指被此目标截断梯度的有效token比例。

GSPO使用回答有效token的几何平均权重：

$$s=\exp\left(\frac1L\sum_t M_t(\ell_t^{new}-\ell_t^{old})\right).$$

在log域平均，避免先连乘。prompt和padding不能进入平均。
序列loss展开成[N,T]便于共享聚合；平均后仍只有一次s的贡献。
默认GSPO和标准sequence-normalized GRPO对比；把它和constant变体组合会改变
长度权重，应单独说明，不能直接冒充原GSPO。

## 6. SFT与packing

把课程prompt/response格式化为Alpaca文档，末尾加EOS，把文档token串拼接。
每个训练输入取连续m个token，对应标签取后移一个token的m个token。
保留完整输入/标签长度，最后不足m+1个可用token的残段丢弃。

本补充作业要求整段语言模型loss，prompt token也在其中。其他实践里常见的
response-only SFT不是这里默认数据集和fixture的语义。

先按原始样本划分train/validation，再packing。把已经拼接的token流随机切
成训练与验证容易让同一示范跨两边，产生泄漏。本实现还排除重复prompt。

每个微批loss按该微批真实样本数/当前有效batch样本数加权，最后不足batch
时也如此。optimizer只更新需要梯度的参数。

## 7. QLoRA节省什么，又不节省什么

Base模型以4-bit NF4存储，计算采用BF16；冻结base，只训练小LoRA矩阵。
线性层更新可写为：

$$W'=W+\frac\alpha r BA.$$

rank r=16，alpha=32，adapter dropout=0。梯度、optimizer状态主要跟LoRA
参数数目相关，显著减少训练状态；仍然需要base权重、激活、logits和KV cache。
因此“8B×4bit约4GB”只是权重的粗估，不是完整显存需求。

Gradient checkpointing通过反向重算节省激活；它消耗额外算力。
SFT/DPO学习率是adapter的设置，不应照抄full-finetuning默认值后声称相同训练。
保存checkpoint时只保存adapter、tokenizer和恢复状态，仍需要原base模型。

## 8. DPO：从偏好差到梯度

每条数据包含同一个prompt和chosen/rejected回答。
先评分两条回答的序列log-prob，固定SFT reference评分：

$$m=(\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))
 -(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x)).$$

$$L_{DPO}=-\log\sigma(\beta m).$$

policy要比reference更偏好chosen。beta控制DPO目标尺度与偏离reference的
关系，不是一个直接可读的“准确率温度”。使用logsigmoid避免先sigmoid再log。

原始测试入口对两个完整格式化字符串评分；相同prompt的概率在差值里抵消。
单卡训练入口显式只累计response token，这样reference缓存可直接复用。
训练中的截断规则是共享prompt保留末尾不超过半个context，回答保留开头并
保留EOS；双方使用同一prompt token。缓存绑定实际token，避免模板变了还复用。

最容易错的是：关闭SFT adapter会回到原base模型，而不是SFT reference。
本实现先冻结SFT checkpoint评分，缓存chosen/rejected log-probs，然后才更新。
resume DPO时仍用初始SFT生成/验证缓存，不能用更新后的DPO模型作为reference。

## 9. 实验、恢复和成本

RL的step是rollout batch；off-policy每个step里有32个optimizer steps。
同一横轴“step”下比较训练速度可能误导；同时检查wall clock、生成token数
和optimizer_steps。SFT/DPO的step则是optimizer更新。

断点保存模型/adapter、optimizer、scheduler、数据位置、RNG和配置。
checkpoint目录只有出现COMPLETE.json才是可恢复状态；硬超时产生的半成品
不用于恢复。发生非有限梯度时不把坏模型写到last覆盖已有有效状态。

租卡测速必须验证实际forward/backward、量化、512上下文、采样和保存。
第一个step排除预热；两次稳态数据仍不足以作精确成本承诺，推断值明确标为估计。
严格时间上限由外部子进程watchdog执行，max_seconds本身只是合作式检查。

MMLU/GSM8K固定子集可以看学习趋势，单种子不能估计训练方差。
自动裁判没有运行就不产生winrate或安全比例；人工样本观察也不等于整体安全。
报告区分“测试通过”“训练跑通”“性能提升”和“满足讲义门槛”四件事。
