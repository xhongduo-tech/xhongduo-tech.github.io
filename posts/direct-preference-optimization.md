## 核心结论

DPO，Direct Preference Optimization，直译是“直接偏好优化”，意思是不用先单独训练奖励模型，再跑一套强化学习，而是直接用“人类更喜欢哪个回答”来更新语言模型。

它的关键结论只有一句：**在带 KL 约束的 RLHF 目标下，最优策略和参考模型之间的对数概率比，本身就等价于一个隐式奖励函数。** 这让“奖励建模 + PPO”可以改写成一个更简单的二分类训练问题。

具体做法是：给定同一个输入 $x$，以及一对回答 $(y_w, y_l)$，其中 $y_w$ 是优选回答，$y_l$ 是劣选回答，DPO要求当前策略 $\pi_\theta$ 相比参考模型 $\pi_{ref}$，在 $y_w$ 上更“加分”，在 $y_l$ 上更“不加分”。损失函数写成：

$$
\mathcal{L}_{\text{DPO}}
=
-\mathbb{E}_{(x,y_w,y_l)}
\log \sigma \left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
$$

其中 $\sigma$ 是 Sigmoid 函数，也就是把一个实数压到 $(0,1)$ 区间的函数；$\beta$ 是 KL 约束强度，直观上就是“允许策略偏离参考模型多少”。

对初学者最重要的理解是：**DPO不是直接让模型绝对偏爱某个答案，而是让模型“相对参考模型”更偏爱优选答案。** 这样做的好处是训练更稳，工程管线更短，不需要 PPO 里常见的价值网络、奖励模型和复杂采样循环。

---

## 问题定义与边界

DPO解决的问题是：当我们只有“回答 A 比回答 B 更好”这类偏好数据时，怎样把这种监督信号稳定地转成语言模型参数更新。

这里的基本数据单位是偏好三元组：

| 字段 | 含义 | 例子 |
|---|---|---|
| $x$ | 输入提示词，也就是用户问题 | “解释什么是哈希表” |
| $y_w$ | winner，优选回答，也就是被偏好标注为更好的回答 | 定义准确、结构清晰的答案 |
| $y_l$ | loser，劣选回答，也就是较差回答 | 定义含糊、遗漏复杂度的答案 |

如果只有这种“二选一偏好”，DPO非常合适；如果只有单个答案的打分，比如 1 到 7 分，DPO就不是最自然的建模方式，因为它天然是成对比较。

它和 PPO 这一类 RLHF 流程的边界差异可以先看表：

| 方法 | 训练数据 | 是否需要奖励模型 | 是否需要在线采样 | 工程复杂度 |
|---|---|---|---|---|
| DPO | 偏好对 $(x,y_w,y_l)$ | 不需要显式奖励模型 | 通常不需要 | 低 |
| PPO-based RLHF | 偏好数据先训练 reward model，再采样训练策略 | 需要 | 需要 | 高 |
| 纯 SFT | 单条监督答案 | 不需要 | 不需要 | 很低 |

DPO的适用边界有三条。

第一，**必须能算参考模型概率**。也就是你要有一个固定的 $\pi_{ref}$，通常来自 SFT 后的基座模型或其冻结副本。

第二，**数据最好和模型实际生成分布不要差太远**。这里的“分布”可以先理解为“回答风格、长度、主题范围”。如果偏好数据全是短问答，但模型真实上线要写长篇分析，训练就容易不稳。

第三，**偏好标注质量必须高**。DPO对“哪条更好”的判断非常敏感，因为它直接把这个判断当作训练目标，而不是先经过奖励模型平滑一次。

玩具例子可以这样看：输入是“2+2等于几？”  
- $y_w$：“2+2=4。”  
- $y_l$：“答案可能是 5。”  

这时 DPO 不需要知道“4 分比 1 分高多少”，只需要知道前者胜过后者，就可以更新模型。

---

## 核心机制与推导

先从 RLHF 的 KL 正则化目标出发：

$$
\max_{\pi}
\mathbb{E}_{x, y \sim \pi(\cdot|x)}[r(x,y)]
-
\beta \, \mathrm{KL}(\pi(\cdot|x)\|\pi_{ref}(\cdot|x))
$$

这里的 KL，Kullback-Leibler divergence，叫“相对熵”，白话解释是“两个概率分布差了多少”。它的作用是防止策略为了追奖励跑得太远。

把目标按每个输入 $x$ 展开，可以得到最优策略满足：

$$
\pi^*(y|x)
\propto
\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

对上式取对数并整理：

$$
r(x,y)
=
\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + C(x)
$$

其中 $C(x)$ 是只和输入有关的归一化常数。这个式子就是 DPO 的核心。它说明：**奖励不必显式训练出来，只要看策略相对参考模型的概率比，就已经隐含了奖励。**

接下来把人类偏好建模成 Bradley-Terry 模型。Bradley-Terry 是一个“成对比较谁赢”的概率模型，白话解释是：给两个候选项各打一个隐藏分数，分数高的更容易赢。写成：

$$
P(y_w \succ y_l \mid x)
=
\frac{\exp(r(x,y_w))}{\exp(r(x,y_w)) + \exp(r(x,y_l))}
=
\sigma(r(x,y_w)-r(x,y_l))
$$

把前面的隐式奖励代进去：

$$
P(y_w \succ y_l \mid x)
=
\sigma \left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
$$

再对偏好数据做极大似然估计，就得到 DPO 损失。到这里，推导链条是：

KL 正则化 RLHF目标  
$\rightarrow$ 最优策略可写成参考模型乘指数奖励  
$\rightarrow$ 奖励等价于策略与参考模型的对数概率比  
$\rightarrow$ Bradley-Terry 成对偏好概率  
$\rightarrow$ Sigmoid 二分类损失

玩具数值例子如下。设 $\beta = 0.2$，参考模型和当前策略在两个回答上的条件概率为：

| 回答 | $\pi_{ref}$ | $\pi_\theta$ |
|---|---:|---:|
| $y_w$ | 0.4 | 0.6 |
| $y_l$ | 0.2 | 0.1 |

则差值项为：

$$
0.2 \left[
\log\frac{0.6}{0.4}
-
\log\frac{0.1}{0.2}
\right]
=
0.2(\log 1.5 - \log 0.5)
\approx 0.2196
$$

于是：

$$
\sigma(0.2196) \approx 0.555
$$

这表示当前策略已经比参考模型更偏向优选答案，但偏向程度还不强，所以还会继续被优化。

真实工程例子可以看客服问答。输入是“退款流程是什么？”  
- 优选回答：步骤完整、条件明确、引用最新政策。  
- 劣选回答：只说“联系客服处理”，没有边界条件。  

DPO更新的不是一句“优选回答得 9 分”，而是让模型在这类问题上，相对于参考模型，更稳定地提高第一类回答的生成概率。

---

## 代码实现

工程上，DPO最大的优势是可以直接复用 SFT 的训练框架。你只需要把单条样本改成成对样本，并同时前向 policy 和 reference。

一个最小可运行的 Python 例子如下，它演示 DPO 的核心计算，不依赖深度学习框架：

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dpo_loss(policy_win, policy_lose, ref_win, ref_lose, beta=0.2):
    diff = beta * ((math.log(policy_win) - math.log(ref_win)) -
                   (math.log(policy_lose) - math.log(ref_lose)))
    loss = -math.log(sigmoid(diff))
    return diff, loss

diff, loss = dpo_loss(
    policy_win=0.6,
    policy_lose=0.1,
    ref_win=0.4,
    ref_lose=0.2,
    beta=0.2
)

assert round(diff, 4) == 0.2197
assert 0.58 < loss < 0.60
print(diff, loss)
```

如果换成训练时更常见的“对数概率”写法，伪代码通常只有四步：

```python
# 1. 读入偏好三元组
batch = {
    "prompt": x,
    "chosen": y_w,
    "rejected": y_l,
}

# 2. policy 和 reference 分别前向，得到序列 logprob
pi_logp_w = policy.logprob(x, y_w)
pi_logp_l = policy.logprob(x, y_l)
ref_logp_w = ref.logprob(x, y_w)
ref_logp_l = ref.logprob(x, y_l)

# 3. 计算 DPO margin
diff = beta * ((pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l))

# 4. 计算损失并反向传播
loss = -log_sigmoid(diff).mean()
loss.backward()
optimizer.step()
```

这里的 `logprob` 是“模型给整段目标序列分配的对数概率”，白话讲就是“模型有多愿意生成这段回答”。

真实工程里通常还有两个实现细节。

第一，使用同一个 tokenizer 和 prompt 模板，保证 policy 与 reference 的概率可比。否则哪怕文本一样，token 切分不同，损失也会失真。

第二，只对回答部分计 loss，不对 prompt 计 loss。因为我们关心的是模型对候选回答的偏好，不是提示词本身的似然。

如果你已经有一个 SFT 训练器，那么接入 DPO 往往只需要改三件事：

| 模块 | SFT 做法 | DPO 做法 |
|---|---|---|
| 数据集 | $(x, y)$ | $(x, y_w, y_l)$ |
| 前向 | 只跑 policy | 同时跑 policy 和 frozen reference |
| 损失 | token-level CE | pairwise DPO loss |

---

## 工程权衡与常见坑

DPO的主要收益是“少系统、少超参、少不稳定源”。

它比 PPO 简单，因为没有单独的 reward model、value model、优势函数估计和在线采样循环。对很多团队来说，这不只是代码量减少，而是**可复现性和调参成本显著下降**。

但它也有明确代价。最大的问题不是公式，而是数据。

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 偏好数据质量差 | 训练后回答风格混乱 | 标注标准不一致 | 统一标注规范，做双标注一致性检查 |
| policy 与数据分布差太远 | loss 波动大，甚至性能下降 | 离线偏好数据无法覆盖当前策略行为 | 先做高质量 SFT，对齐初始分布 |
| $\beta$ 太小 | 模型发散、胡乱迎合偏好 | KL 约束过弱 | 增大 $\beta$，加强对参考模型的约束 |
| $\beta$ 太大 | 几乎学不动 | 过分贴近 reference | 减小 $\beta$，观察 win-rate 与 KL |
| 长短回答偏差 | 模型偏向短答或模板化答复 | 序列总 logprob 容易受长度影响 | 做长度均衡，或使用长度归一化策略 |
| 模板不一致 | 同一答案概率不可比 | prompt format 改变了条件分布 | 固定 chat template 与 tokenizer |

一个常见误区是：**DPO 不等于“只要有偏好对就一定稳定”。**  
如果偏好数据全部来自短回复，而你的模型要处理长链路推理或长文总结，那么 policy 在这些长样本上的概率分布可能和 reference 严重错位。结果就是某些 batch 的 `logprob` 差值非常极端，训练会出现反向推进，甚至让模型失去原有能力。

真实工程例子：一个客服模型在偏好数据中主要学“简短礼貌回复”，上线后却要处理“退款规则说明”“物流异常解释”“条款争议澄清”这类长答案。此时 DPO 可能把“简短、看起来礼貌”错误当成“整体更好”，导致模型过度缩短回答。解决方案通常不是继续硬调学习率，而是补齐长答案偏好对，并按任务类型分桶采样。

另一个坑是参考模型选得不对。如果 reference 太弱，KL 约束本身就没有足够质量；如果 reference 和 policy 初始化不一致，训练目标会引入额外噪声。工程上更常见的做法是：**用同一个 SFT checkpoint 初始化 policy 和 reference，reference 全程冻结。**

---

## 替代方案与适用边界

DPO不是要替代所有对齐方法，而是适合一类很明确的场景：**你已经有较好的 SFT 模型，也有质量不错的偏好对，并且希望用最短路径做偏好对齐。**

如果问题不满足这个条件，就要考虑别的方法。

| 方法 | 需要什么数据 | 是否显式奖励模型 | 是否需要采样 | 适合什么场景 |
|---|---|---|---|---|
| DPO | 偏好对 | 否 | 通常否 | 简化 RLHF 管线，快速做偏好对齐 |
| PPO-based RLHF | 偏好数据或可训练 reward 数据 | 是 | 是 | 需要在线探索、复杂约束、成熟 RL 管线 |
| GRPO | 分组比较或规则/奖励信号 | 通常是 | 是 | 推理类任务、可构造组内相对奖励 |
| 纯 SFT | 标准答案 | 否 | 否 | 基础能力学习、格式跟随 |

有些任务天然不适合 DPO。

第一类是**只有标量奖励，没有成对偏好**。例如每个答案给 1 到 7 分，这时显式 reward model 往往更自然，因为分数里有细粒度强弱信息，直接硬转成 winner/loser 会丢信息。

第二类是**强依赖在线探索**。例如代码生成里，模型需要不断尝试新解，再用单元测试结果作为奖励。这类任务更接近强化学习原型，PPO、GRPO 或其他 policy gradient 方法更合适。

第三类是**需要复杂多目标约束**。比如既要帮助性，也要安全性，还要格式稳定性，并且三者权重需要动态调度。DPO不是不能做，但表达能力会比显式 reward 组合更弱。

可以用一句话概括边界：  
**DPO擅长把“人更喜欢 A 而不是 B”这类监督信号，低成本地转成稳定训练；一旦信号变成连续分数、环境反馈或在线搜索，DPO就不再是最省事的方案。**

---

## 参考资料

1. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. 论文原文，给出了 DPO 的核心推导和损失函数定义。  
2. Hugging Face, *Guide to RL Post-Training for LLMs: PPO, DPO, GRPO*. 工程综述文章，适合建立 DPO 与 PPO、GRPO 的方法对比。  
3. RLHFlow, *Alignment Guidebook*. 对 KL 正则化目标、Bradley-Terry 偏好建模和 DPO 训练细节有较清楚的整理。  
4. Aminer 相关课程讲义，RLHF PPO & DPO 部分。适合补充 offline 偏好数据、分布偏移和训练稳定性问题。  
5. 一些开源训练框架中的 DPOTrainer 实现，例如 Hugging Face TRL。适合直接看工程接口、batch 组织方式和 logprob 计算细节。
