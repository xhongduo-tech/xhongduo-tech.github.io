## 核心结论

离线 DPO 和在线 DPO 的核心差别，不在损失函数名字，而在偏好数据是不是跟着当前策略一起变化。

DPO，Direct Preference Optimization，直译是“直接偏好优化”，意思是不用先显式训练奖励模型、再跑强化学习，而是直接用“回答 A 胜过回答 B”的偏好对来更新策略。离线 DPO 使用一批提前收集好的固定偏好对；在线 DPO 则在训练过程中用当前模型实时生成回答，再立刻标注优劣并继续更新。

结论可以压缩成一句话：如果模型已经偏离旧数据分布，离线 DPO 的训练信号会越来越旧；在线 DPO 因为不断收集当前策略上的新偏好，能更接近 on-policy，也就是“按当前策略分布学习”的状态，因此更容易持续改进。

一个新手可用的直观类比是：离线 DPO 像产品团队只拿 1 月份的一次问卷做全年迭代；在线 DPO 像每天都收集最新用户投票再更新排序逻辑。前者便宜，但会逐渐失真；后者更贵，但反馈闭环更短。

下面这个表可以先建立全局认识：

| 对比维度 | 离线 DPO | 在线 DPO |
| --- | --- | --- |
| 数据来源 | 预先标注好的固定偏好对 | 当前策略采样后实时标注 |
| 分布特性 | 静态，容易与当前模型脱节 | 动态，能覆盖当前高概率输出 |
| 训练反馈 | 更像“拿旧样本修今天的模型” | 更像“边生成边纠偏” |
| 工程复杂度 | 低，流程简单 | 高，需要采样、打分、缓存、更新闭环 |
| 典型风险 | 分布偏移、过时偏好 | 采样成本高、奖励噪声大、系统复杂 |

SimPO 可以看作 DPO 的一个变体。它的关键设计是去掉参考模型。参考模型就是训练时拿来做“相对比较基线”的旧策略。标准 DPO 要比较当前策略和参考模型之间的对数概率差；SimPO 直接把当前策略对整条回答的长度归一化 log 概率当成隐式奖励，再通过 margin 拉开优劣回答的距离。这样做的直接收益是工程更轻、显存更省，但训练稳定性更依赖采样质量和超参数设置。

Iterative DPO，即迭代式 DPO，可以理解为“在线思想的批处理版本”：不是每个 step 都立刻重新采样，而是一轮训练后重新用新策略生成数据、重新构造偏好对、再进入下一轮。它通常是工程上比纯在线更容易落地的折中方案。

---

## 问题定义与边界

这类方法要解决的问题，本质上是“偏好学习中的分布错位”。

偏好数据指的是同一个输入 $x$ 下，两条回答 $y_w$ 和 $y_l$，其中 $y_w$ 被认为优于 $y_l$。这里的 $w$ 是 winner，表示胜者；$l$ 是 loser，表示败者。DPO 假设这些偏好对能告诉模型“什么回答更符合目标”，然后把这种比较信号转成可微的训练目标。

标准写法通常基于 Bradley-Terry 模型。Bradley-Terry 是一种“成对比较概率模型”，白话解释就是：如果两个候选项各自有一个隐含分数，那么分数差越大，前者胜出的概率越高。对应到 DPO，可写成：

$$
p^*(y_w \succ y_l \mid x) =
\sigma\left(
\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right)
$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数，$\beta$ 是温度或缩放系数，控制偏好差异被放大的程度；$\pi_\theta$ 是当前策略；$\pi_{\text{ref}}$ 是参考策略。

问题在于：这个公式本身没有错，但数据分布可能过时。

如果偏好对都来自旧模型、旧用户或旧任务分布，那么训练时真正被优化的是“让当前模型在旧支持集上更偏向 winner”。支持集可以理解为“数据实际覆盖到的区域”。当模型更新后，它开始频繁生成一些旧数据里几乎没有的新回答模式，这些模式就拿不到足够训练信号。结果是：模型在推理时跑到新区域，训练却还停留在旧区域。

一个玩具例子可以说明这个边界。

假设只有一个 prompt：“解释 Python 列表推导式”。  
1 月份收集的偏好数据偏爱简短答案，因为当时用户只要一句定义。  
3 月份产品人群变化，新用户更偏爱“定义 + 语法 + 示例 + 常见坑”的结构化回答。

如果你继续只用 1 月份的离线偏好集训练，那么模型即使已经学会生成结构化长回答，训练仍可能反复把概率往“更短、更像旧分布”的方向推。不是因为新回答一定差，而是因为旧数据根本没覆盖这种回答风格。

因此，这篇文章讨论的边界很明确：

1. 重点讨论成对偏好优化，即 DPO、在线 DPO、SimPO、Iterative DPO。
2. 不展开 PPO 等完整 RLHF 管线，只在必要时提及对比。
3. 讨论重点是“数据收集策略”和“训练分布关系”，不是基础语言模型预训练。
4. 在线不一定意味着每个 token 都在线更新，很多工程系统是“小批量在线”或“轮次在线”。

---

## 核心机制与推导

标准 DPO 的目标函数通常写为：

$$
\mathcal{L}_{\mathrm{DPO}}
=
-\mathbb{E}_{(x,y_w,y_l)}
\left[
\log \sigma \left(
\beta \left(
\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right)
\right)
\right]
$$

它做的事很直接：如果 winner 在当前策略相对参考模型的优势还不够大，就继续把 winner 的相对概率推高，把 loser 的相对概率压低。

这里有两个关键点。

第一，DPO 优化的是“相对优势”，不是绝对概率。  
如果当前策略和参考模型在 winner 上都高，在 loser 上都低，未必有足够梯度；真正重要的是 winner 相对 loser 的优势有没有拉开。

第二，DPO 默认数据已经足够代表目标分布。  
这正是离线 DPO 的瓶颈。因为一旦策略更新，$\pi_\theta$ 发生变化，模型最常生成的回答也跟着变化，但训练集不变。

在线 DPO 的机制就是把数据分布也纳入训练循环。一个典型回路是：

1. 从 prompt 池采样输入。
2. 用当前策略为每个 prompt 生成多条候选回答。
3. 用人工标注、规则打分或模型评审生成偏好。
4. 选出 winner / loser 组成新偏好对。
5. 立刻或分批执行 DPO 更新。
6. 用更新后的策略进入下一轮采样。

这个流程更接近 on-policy。on-policy 的白话解释是“训练时使用当前策略自己生成的数据”。它的好处是偏好标签总是贴在当前模型真实会输出的区域上，而不是贴在很久以前的输出区域上。

再看一个简单数值例子。

同一个 prompt 下，当前策略采样出 3 条回答，规则奖励分别为：

| 回答 | 奖励 |
| --- | --- |
| A | 0.8 |
| B | 0.5 |
| C | 0.2 |

最简单的构造方式是取 A 为 winner，C 为 loser，形成一对 $(A, C)$。如果每轮都这样做，偏好对会随当前策略演化。比如训练几轮后，模型不再生成特别差的 C，而是生成更接近 A 的新候选，那么后续训练面对的是更细粒度、更接近真实决策边界的比较。

这就是在线或迭代式 DPO 比离线 DPO 更有效的原因：不是损失函数突然变了，而是训练信号始终贴着当前策略分布。

SimPO 的变化更激进。它把参考模型去掉，定义：

$$
r_{\text{SimPO}}(x,y)=\frac{\beta}{|y|}\sum_{i=1}^{|y|}\log \pi_\theta(y_i \mid x, y_{<i})
$$

这里 $|y|$ 是回答长度，长度归一化是为了避免“只因回答更长、总 log 概率更低或更高”带来的偏差。于是损失可以写成：

$$
\mathcal{L}_{\mathrm{SimPO}}
=
-\log \sigma \left(
r_{\text{SimPO}}(x,y_w) - r_{\text{SimPO}}(x,y_l) - \gamma
\right)
$$

其中 $\gamma$ 是 margin，表示希望 winner 至少领先 loser 一个明确间隔。margin 的白话解释是“不是只要求赢，而是要求赢得足够明显”。

SimPO 的优点是：

1. 不需要维护参考模型，计算和显存成本更低。
2. 梯度直接作用于当前策略，不再通过相对参考的差值间接表达。

但它也引入一个风险：因为奖励完全来自当前策略自身概率，如果采样不稳定，或者模型学会某些“高似然但不一定高质量”的模式，就可能把这种偏差直接放大。因此 SimPO 常常需要更谨慎地设置温度、margin，以及更稳定的采样策略。

Iterative DPO 则是另一个很实用的工程思路。它不一定完全实时，而是按轮次执行：

1. 用当前策略在一批任务上生成候选。
2. 构造偏好对。
3. 训练若干步。
4. 用新模型再次生成。
5. 重复上述过程。

这相当于把“在线 DPO 的闭环”离散化成多轮重采样。对于算力和标注系统不够强的团队，这通常比真正的流式在线系统更容易上线。

真实工程例子可以看数学推理任务。比如对一道数学题，同一个模型一次采样 8 个解答，利用规则校验器判断最终答案是否正确，再把正确且更简洁的解答作为 winner，把错误或冗长的解答作为 loser。经过多轮迭代后，模型会逐渐把概率质量集中到“能解对且表达稳定”的解题轨迹上。这类流程在需要规则验证器的任务上尤其有效，因为偏好信号可以自动构造，不完全依赖人工标注。

---

## 代码实现

下面先用一个最小可运行的 Python 脚本演示 DPO 和在线数据构造的核心逻辑。这个例子不是训练真实大模型，而是用标量 log 概率模拟“winner 应该比 loser 更占优”的更新方向。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=1.0):
    margin = beta * ((pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l))
    return -math.log(sigmoid(margin))

def simpo_loss(avg_logp_w, avg_logp_l, beta=1.0, gamma=0.5):
    margin = beta * (avg_logp_w - avg_logp_l) - gamma
    return -math.log(sigmoid(margin))

# 一个玩具偏好对：当前策略更偏向 loser，需要被纠正
pi_logp_w = -2.0
pi_logp_l = -1.2
ref_logp_w = -2.4
ref_logp_l = -1.5

loss_before = dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=2.0)

# 模拟一次“朝正确方向”的更新：提高 winner 概率，降低 loser 概率
pi_logp_w_new = -1.5
pi_logp_l_new = -1.8

loss_after = dpo_loss(pi_logp_w_new, pi_logp_l_new, ref_logp_w, ref_logp_l, beta=2.0)
assert loss_after < loss_before

# SimPO 也应满足：winner 的平均 log 概率更高时，loss 更小
simpo_bad = simpo_loss(avg_logp_w=-1.8, avg_logp_l=-1.2, beta=1.0, gamma=0.2)
simpo_good = simpo_loss(avg_logp_w=-1.0, avg_logp_l=-1.6, beta=1.0, gamma=0.2)
assert simpo_good < simpo_bad

# 在线构造偏好对：从多条响应里选最好和最差
responses = [
    {"text": "A", "reward": 0.8},
    {"text": "B", "reward": 0.5},
    {"text": "C", "reward": 0.2},
]
winner = max(responses, key=lambda x: x["reward"])
loser = min(responses, key=lambda x: x["reward"])
assert winner["text"] == "A"
assert loser["text"] == "C"

print("all tests passed")
```

这个脚本体现了三件事：

1. DPO 关心的是 winner 和 loser 的相对 margin。
2. SimPO 去掉参考模型后，直接比较当前策略上的归一化 log 概率。
3. 在线 DPO 的关键不是“神秘新损失”，而是“每轮用当前策略重新构造偏好对”。

如果把它放大成真实训练流水线，伪代码通常长这样：

```python
for prompt in prompt_batch:
    responses = current_policy.sample(prompt, num_samples=n)
    scores = reward_fn(prompt, responses)  # 人工、规则或评审模型
    winner = responses[argmax(scores)]
    loser = responses[argmin(scores)]

    loss = dpo_or_simpo_loss(prompt, winner, loser, current_policy, reference_model)
    loss.backward()
    optimizer.step()
```

离线 DPO 和在线 DPO 的代码差异，主要不是 `loss.backward()` 这一行，而是 `winner/loser` 从哪里来。

离线 DPO 一般是：

1. 先读固定的 `(prompt, chosen, rejected)` 数据集。
2. 按 batch 喂给模型。
3. 计算 DPO loss。
4. 训练结束。

在线或迭代 DPO 一般是：

1. 准备 prompt 池。
2. 当前策略采样多条回复。
3. 打分或标注。
4. 选 pair。
5. 更新模型。
6. 把新模型继续用于后续采样。

真实工程里还会多出几个必要模块：

| 模块 | 作用 | 典型实现 |
| --- | --- | --- |
| Sampler | 生成候选回答 | 温度采样、top-p、best-of-n |
| Judge | 判断回答优劣 | 人工标注、规则校验器、奖励模型、LLM-as-a-judge |
| Buffer | 缓存新偏好对 | 滑动窗口、优先队列、分轮数据集 |
| Trainer | 执行 DPO/SimPO 更新 | 单机微调或分布式训练 |
| Reference 管理 | 决定参考模型是否刷新 | 固定 reference、周期更新、EMA |

对于零基础读者，最重要的认知是：在线 DPO 不是单独一条公式，而是一条数据闭环。没有稳定的数据生成和标注链路，只写出 DPO loss 并不等于完成在线训练系统。

---

## 工程权衡与常见坑

在线 DPO 的收益来自更贴近当前策略，但代价也非常明确：系统更复杂，而且每个环节都会引入噪声。

最常见的第一个坑，是把“旧数据质量高”误认为“旧数据永远够用”。  
离线偏好集即使最初质量很好，也只能代表当时的模型输出分布。随着策略变化，模型会在新区域生成更多候选，而这些区域没有标注，训练就会出现明显的 off-policy 偏差。off-policy 的白话解释是“你在用不是当前策略产生的数据训练当前策略”。

第二个坑，是在在线 DPO 里总是取 best-of-n 和 worst-of-n 的极端 pair。  
这听起来合理，但很容易过拟合异常值。比如一组候选里，最高分回答只是因为“写得特别长、碰巧触发奖励规则”，最低分回答则是采样噪声产生的灾难样本。若每轮都只训练这两个极端，模型会学到“如何刷分”，而不是学到稳定的偏好规律。

可以用分布感知 pairing 缓解。分布感知 pairing 的意思是：不是永远选最大和最小，而是在奖励分布的不同位置构造样本，比如从高于均值 $\mu$ 的回答里选 winner，从低于 $\mu-\sigma$ 的回答里选 loser。这样做通常比只抓极端值更稳。

第三个坑，是 SimPO 的“无参考模型”被误解成“无代价简化”。  
SimPO 省掉了 reference，但并不意味着调参更容易。恰恰相反，因为它直接把当前策略的平均 log 概率当作奖励基础，如果采样温度太高、回答长度分布不稳定、或 margin 太小，都可能导致训练信号非常抖。工程上常见的症状是验证集波动大、奖励提升但人工偏好不升，或者模型开始偏好模板化输出。

第四个坑，是在线系统的吞吐瓶颈。  
真正的成本通常不在梯度更新，而在采样和打分。尤其当 judge 是更大的 LLM 或人工标注时，系统会受制于标注延迟。很多团队最后采用的不是完全实时在线，而是“每 6 小时或每天一轮”的 Iterative DPO。

下面的表可以作为排障清单：

| 陷阱 | 症状 | 规避 |
| --- | --- | --- |
| 静态偏好集 | 新模型输出与训练样本风格脱节 | 加入 on-policy 采样或周期刷新数据 |
| 极端配对过多 | 验证集提升不稳，回答越来越模板化 | 使用分布感知 pairing，不只取 max/min |
| SimPO 不稳定 | loss 波动大，风格突然塌缩 | 调整 margin、温度、长度归一化与采样策略 |
| 奖励黑客化 | 模型学会刷规则分而非提升真实质量 | 混合人工评审、规则校验和多评审一致性 |
| 在线链路过重 | 采样和打分延迟高，训练难持续 | 用 Iterative DPO 做轮次式更新 |

一个具体工程例子是客服问答系统。假设团队先用历史人工偏好数据做离线 DPO，初期效果提升明显。但上线后，用户问题从“功能介绍”逐渐转向“故障排查”和“退款边界”。这时模型实际生成的回答类型变了，旧偏好集里却主要是营销型回答。结果是离线 DPO 继续强化旧风格，线上满意度反而下降。比较合理的做法是：保留高质量离线偏好集作为基础，再加入近一周真实请求上的在线或半在线偏好数据，按比例混合训练。

---

## 替代方案与适用边界

纯离线 DPO 和纯在线 DPO 并不是非此即彼。实际工程里更常见的是混合方案。

第一类折中是 InCo-DPO 一类的 on-policy / off-policy 混合方法。  
思路很直接：离线数据往往质量更高、标注更稳定，但分布旧；在线数据更贴近当前策略，但噪声更大、成本更高。混合训练就是在“质量”和“分布一致性”之间找平衡。这类方法通常适合已有一批不错的人工偏好集、同时又能承担有限在线采样成本的团队。

第二类替代是 DPO-Shift 一类关注概率分布位移的方案。  
它的重点不是简单地把 winner 概率做大，而是更显式地控制 chosen probability 的分布变化，避免 likelihood displacement。likelihood displacement 可以白话理解为“模型为了满足偏好目标，把原本合理的概率结构挤歪了”。这类方法适合对生成分布形状更敏感的场景，比如需要兼顾安全性、覆盖率和稳定性的生产系统。

第三类替代是 SimPO。  
它最适合的场景是：你希望保留 DPO 的直接偏好优化思路，但不想长期维护参考模型的显存和工程开销。不过前提是采样和训练足够稳，否则无参考模型未必比标准 DPO 更省心。

第四类替代是 Iterative DPO。  
如果团队没有能力做真正流式在线系统，但可以接受按天或按轮次迭代数据，这往往是最现实的方案。它特别适合“可自动打分”的任务，例如代码、数学、格式化抽取、规则问答。

可以用一个总结表收尾：

| 方法 | 优势 | 限制/边界 |
| --- | --- | --- |
| 离线 DPO | 最容易落地，适合已有高质量偏好集 | 分布会过时，长期迭代能力弱 |
| 在线 DPO | 数据始终贴近当前策略，减轻分布偏移 | 系统复杂，采样和标注成本高 |
| SimPO | 无参考模型，训练更轻 | 更依赖稳定采样和超参数 |
| Iterative DPO | 比纯在线简单，仍能逐轮刷新数据 | 更新频率低于真正在线 |
| 混合方案（如 InCo-DPO） | 兼顾旧数据质量和新数据覆盖 | 采样比例和数据权重更难调 |

因此，适用边界可以简单概括：

1. 如果你只有一批高质量人工偏好集、任务分布变化也不快，离线 DPO 足够实用。
2. 如果模型会快速进入旧数据没覆盖的新输出区域，在线 DPO 或 Iterative DPO 更合适。
3. 如果显存和系统复杂度是主要约束，可以考虑 SimPO，但要留出更多稳定性验证。
4. 如果任务可以自动打分，迭代式在线偏好构造通常性价比最高。

---

## 参考资料

- Online Direct Preference Optimization 相关综述与条目：<https://www.emergentmind.com/topics/online-direct-preference-optimization-odpo>
- Direct Preference Optimization 机制与损失推导：<https://www.emergentmind.com/articles/direct-preference-optimization-dpo>
- SimPO 论文条目：<https://arxiv.gg/abs/2405.14734>
- RLHFlow 在线/迭代 DPO 实践与代码：<https://github.com/RLHFlow/Online-DPO-R1>
- Hugging Face 对偏好优化训练的课程材料：<https://huggingface.co/learn/smol-course/en/unit2/2>
- InCo-DPO 条目：<https://paperswithcode.com/paper/inco-dpo-balancing-distribution-shift-and>
- DPO-Shift 条目：<https://paperswithcode.com/paper/dpo-shift-shifting-the-distribution-of-direct>
