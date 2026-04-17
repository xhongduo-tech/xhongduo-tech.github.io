## 核心结论

数据混合策略的核心，不是“哪个语料桶更大就多喂一点”，而是“先定义模型想学成什么样，再按目标能力去分配训练批次”。这里的“域”指一类来源或任务相近的数据，比如网页、书籍、代码；“混合权重”指训练时从各域抽样的比例。

如果把训练预算看成固定的 $T$ 个 token，那么混合策略本质上是在决定：这 $T$ 个 token 里，多少给通用知识，多少给代码，多少给高质量长文本。这样做的原因很直接：模型不会自动理解“代码更重要”或“书籍质量更高”，它只会对自己反复见到的分布做拟合。

一个直接结论是：高质量小语料可以被刻意放大，但不能无限放大。因为一旦某个小域被重复太多次，训练信号会从“学习规律”变成“记住样本”。所以混合策略同时是能力分配问题，也是重复率控制问题。

玩具例子可以先这样理解：你有三桶数据，网页 7000 万 token、书籍 1000 万 token、代码 200 万 token。你不想让模型只会“像网页一样说话”，于是规定每 100 个训练样本里，70 个来自网页、20 个来自书籍、10 个来自代码。这个配方表达的是能力意图，而不是尊重原始数据体量。

真实工程里，大模型预训练几乎都依赖这种思路。公开配方常见的做法是让 web 占大头，用来维持覆盖面；让 code 占一个不小但受控的比例，用来增强形式化表达和长链条约束；再留一部分给 books 或百科，补足结构化长文本能力。最终模型表现，往往更像混合配方的结果，而不是原始爬虫总量的结果。

---

## 问题定义与边界

问题可以形式化为：

给定总训练预算 $T$、每个数据域可用 token 数量 $N_d$、每个域的目标训练权重 $w_d$，如何设计抽样机制，使模型在预算内达到目标能力，同时避免小域被过度重复。

这里：
- $d$ 表示某个数据域
- $N_d$ 表示该域可用 token 总量
- $w_d$ 表示该域在训练中的目标占比，满足 $\sum_d w_d = 1$
- $T$ 表示总训练 token 预算

每个域的平均重复率可以写成：

$$
\text{repeats}_d = \frac{w_d \cdot T}{N_d}
$$

它表示“该域中每个 token 在整个训练中平均会被看到多少次”。这是数据混合最关键的监控量。

很多初学者会把它理解反：以为数据越多越该占更高比例。实际上，数据多只说明“可选空间大”，不说明“训练目标重要”。如果你的目标是提升代码补全能力，那么即使代码数据只占原始语料的一小部分，它在训练中也常常需要被显著上调。

下面看一个最小数字例子。假设总预算是 $1\text{T}$ token，书籍域有 $50\text{B}$ token，目标权重是 $15\%$，那么：

$$
\text{repeats}_{books} = \frac{0.15 \times 1\text{T}}{50\text{B}} = 3
$$

这表示平均每个书籍 token 会被看到 3 次。这个数不是天然好或坏，但它已经给工程判断提供了边界：如果某个小域被推到 8 次、10 次甚至更高，你就要怀疑模型是否开始记忆数据细节，而不是学习可迁移规律。

下表能更直观看出“原始体量”和“目标权重”不是一回事：

| 域 | 原始体量 | 目标权重 | 总预算 | 平均重复率 |
|---|---:|---:|---:|---:|
| Web | 900B | 70% | 1T | 0.78 |
| Books | 50B | 15% | 1T | 3.00 |
| Code | 20B | 8% | 1T | 4.00 |
| Wiki/参考资料 | 30B | 7% | 1T | 2.33 |

这个表说明两件事。第一，代码虽然只占较小体量，但为了能力目标可以被明显过采样。第二，小域一旦被抬权重，重复率会上升得很快，因此混合策略的边界不是“想要什么就多喂什么”，而是“想要什么，但不能把重复率推到失控”。

---

## 核心机制与推导

固定混合最基础的机制，是让每一步训练先按概率向量 $w$ 选择域，再从该域中抽样样本。也就是说，批次分布服从目标权重，而不是原始语料自然分布。

如果有 $k$ 个域，权重向量为：

$$
w = (w_1, w_2, \dots, w_k), \quad \sum_{d=1}^{k} w_d = 1
$$

那么单步训练可以写成：
1. 从多项分布 $\text{Multinomial}(w)$ 采样域
2. 在被选中的域内采样文档或 token
3. 拼成 batch，执行梯度更新

这样做为什么成立？因为 SGD 看到的梯度期望，本质上由训练样本分布决定。你改变采样概率，就是在改变模型长期接收到的平均梯度方向。数据混合策略不是后处理技巧，而是直接作用于优化目标的输入分布。

进一步地，自适应混合会根据各域的学习状态动态调整 $w_d$。一个常见思路是比较 proxy model 和 reference model 在各域上的损失差，也就是 excess loss。这里“proxy model”可以理解为当前正在训练、规模较小或更新更频繁的观察模型；“reference model”是一个基线，用来衡量某个域到底是不是还“学得不够”。

定义：

$$
\Delta_d = \ell_d^{proxy} - \ell_d^{ref}
$$

其中：
- $\ell_d^{proxy}$ 是 proxy 在域 $d$ 上的 loss
- $\ell_d^{ref}$ 是 reference 在域 $d$ 上的 loss
- $\Delta_d > 0$ 表示 proxy 在该域落后更多，说明该域可能值得加权

然后使用指数梯度更新：

$$
\tilde{w}_d = w_d \cdot \exp(\eta \Delta_d)
$$

再做归一化：

$$
w_d' = \frac{\tilde{w}_d}{\sum_j \tilde{w}_j}
$$

这里的 $\eta$ 是学习率，控制权重调整有多激进。

直觉上，这套推导表达的是：如果代码域的 loss 明显高于参考基线，就提高代码的采样权重；如果某个域已经学得很好，权重自然会被相对压低。它不是拍脑袋调比例，而是让“模型目前缺什么能力”进入采样决策。

可以把更新过程压缩成一个伪流程表：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 1 | 当前权重 $w$ | 域采样结果 | 决定本轮 batch 来自哪些域 |
| 2 | 各域 batch | 各域 loss | 观察训练状态 |
| 3 | $\ell^{proxy}, \ell^{ref}$ | $\Delta_d$ | 估计哪些域更欠学 |
| 4 | $w_d \cdot \exp(\eta \Delta_d)$ | 新权重 $w'$ | 增加落后域的采样概率 |
| 5 | 归一化、平滑 | 稳定后的 $w'$ | 避免剧烈震荡 |

玩具例子如下。假设当前三域权重分别是 web 0.7、books 0.2、code 0.1；测得 excess loss 为 web 0.02、books 0.01、code 0.20。因为 code 明显更差，更新后 code 权重会放大，下一阶段训练就会“多看代码”。这就是“按模型短板回补分布”的机制。

---

## 代码实现

工程实现通常分成两层：
1. 静态或动态地维护权重向量 $w$
2. 用 data loader 按 $w$ 抽样，并持续监控实际重复率与 loss

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示“重复率计算”和“自适应更新”的核心逻辑。

```python
from math import exp

def compute_repeats(domain_tokens, weights, total_budget):
    repeats = {}
    for name, n_tokens in domain_tokens.items():
        repeats[name] = weights[name] * total_budget / n_tokens
    return repeats

def update_weights(weights, delta_loss, eta=1.0):
    raw = {}
    for name, w in weights.items():
        raw[name] = w * exp(eta * delta_loss[name])
    z = sum(raw.values())
    return {name: value / z for name, value in raw.items()}

domain_tokens = {
    "web": 900,
    "books": 50,
    "code": 20,
}

weights = {
    "web": 0.70,
    "books": 0.20,
    "code": 0.10,
}

total_budget = 1000

repeats = compute_repeats(domain_tokens, weights, total_budget)
assert round(repeats["web"], 2) == 0.78
assert round(repeats["books"], 2) == 4.00
assert round(repeats["code"], 2) == 5.00

delta_loss = {
    "web": 0.02,
    "books": 0.01,
    "code": 0.20,
}

new_weights = update_weights(weights, delta_loss, eta=1.0)

assert abs(sum(new_weights.values()) - 1.0) < 1e-9
assert new_weights["code"] > weights["code"]
assert new_weights["web"] < weights["web"]

print("repeats =", repeats)
print("new_weights =", new_weights)
```

如果把它接到真实训练系统里，最外层伪代码通常长这样：

```python
weights = init_weights()

for step in range(train_steps):
    domain = sample_domain(weights)      # 按权重采样域
    batch = sample_batch(domain)         # 在该域中抽样样本
    loss = train_one_step(batch)

    if step % eval_interval == 0:
        delta = measure_proxy_minus_ref_loss()
        weights = update_weights(weights, delta, eta=eta)
        weights = smooth(weights)        # 平滑，避免骤变
        log_metrics(weights)
```

一个真实工程例子是通用大模型预训练。假设你维护一个 70B 级别模型，希望兼顾聊天、知识问答和代码生成。系统常见做法不是把所有抓来的数据混在一起，而是先把数据分为 web、books、code、reference 四个桶，再分别做清洗、去重、质量评分和抽样。训练期间监控的重点不是“今天读了多少 TB”，而是下面这些字段：

| 域 | 目标 weight | 实际采样占比 | 平均 repeats | 验证集 loss | 是否告警 |
|---|---:|---:|---:|---:|---|
| Web | 0.70 | 0.69 | 0.80 | 1.92 | 否 |
| Books | 0.15 | 0.16 | 3.10 | 1.78 | 否 |
| Code | 0.08 | 0.09 | 4.20 | 1.65 | 是 |
| Ref | 0.07 | 0.06 | 2.05 | 1.83 | 否 |

如果 code 的重复率持续高于阈值，同时验证 loss 不再下降，那就不是“继续加代码权重”，而是要回头检查：数据是否太小、去重是否不足、是否该补充新代码源，或者是否把代码训练放到后续专项阶段更合适。

---

## 工程权衡与常见坑

数据混合不是数学上算出一组比例就结束，真正困难在工程权衡。

第一类坑是“小语料高权重”。高质量小域确实值得放大，但它的副作用是重复率会陡增。重复率太高后，梯度看起来很稳定，实际上信息增量已经变小。模型会更擅长复述见过的模式，却不一定更会泛化。

第二类坑是“动态调度过快”。如果你把 books 权重从 5% 一次性提到 30%，优化器里的动量还停留在旧分布上，训练会出现明显震荡。很多损失上升，不是因为新数据差，而是因为分布切换过猛。更稳妥的做法通常是分阶段插值，比如 5% -> 15% -> 25%。

第三类坑是“目标权重和实际采样不一致”。表面上配置写的是 8% code，但如果 loader、分片、长度截断、失败重试机制有偏差，最终 token 级占比可能完全不是 8%。所以监控必须看实际消费 token，而不是只看配置文件。

第四类坑是“按文档数混，不按 token 数混”。长文档和短文档长度差异极大，如果只按样本条数配比，你以为在做 1:1 混合，实际 token 占比可能是 5:1。训练优化看到的是 token，不是文档条数。

下面是常见坑位和应对方式：

| 坑位 | 现象 | 根因 | 对策 |
|---|---|---|---|
| 高权重小语料 | 验证集提升停滞，训练集继续变好 | 重复率过高，开始记忆 | 设 repeats 上限，补充新数据源 |
| 频繁调度 | loss 震荡、收敛变慢 | 分布变化快于优化器适应 | 线性插值、EMA 平滑、降低 $\eta$ |
| 采样误差 | 配方和实际不一致 | loader 统计口径错 | 记录 token 级消费占比 |
| 文档长度偏差 | 某域被隐性放大 | 按样本数而非 token 数混合 | 所有配比统一换算到 token |
| 质量分层失真 | 高权重域效果反而下降 | 域内低质样本过多 | 域内先排序，再做分层抽样 |

很多团队会加入温度退火。这里“温度”可以理解为把极端权重拉回中间的平滑参数。一个简单伪代码是：

```python
def soften(weights, temperature=1.2):
    raw = {k: v ** (1.0 / temperature) for k, v in weights.items()}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}
```

当 temperature 大于 1 时，过高的权重会被压一压，能减少训练分布突然偏向单一小域的风险。

---

## 替代方案与适用边界

固定比例混合是最容易落地的方案，优点是稳定、可解释、好复现；缺点是对训练状态不敏感，容易在某些阶段喂多了、某些阶段喂少了。

自适应混合，例如基于 excess loss 的 DoReMi 类方法，更适合“不同域难度变化快”或“你不确定最优比例”的场景。它能根据模型短板回补，但前提是你有可靠的验证与评估闭环，否则更新信号会被噪声带偏。

再往前一步，还有文档级或样本级的动态加权，比如 SampleMix。这类方法不只看域，还看单个样本的质量分数和多样性分数。这里“多样性”可以理解为它是否补充了训练集中稀缺的模式；“质量分数”则衡量样本是否干净、信息密度是否高。

一个简化写法是：

$$
w_i = \exp\left(\frac{\alpha \hat{d}_i + (1-\alpha)\hat{q}_i}{\tau}\right)
$$

其中：
- $\hat{d}_i$ 是样本 $i$ 的多样性标准化分数
- $\hat{q}_i$ 是质量标准化分数
- $\alpha$ 控制两者权重
- $\tau$ 是温度，控制分布是否尖锐

然后再按这些 $w_i$ 去决定样本复制次数或采样概率。它更细粒度，但工程成本更高，因为你要维护样本级打分、更新和索引。

下面做一个对比：

| 策略 | 核心思想 | 收敛稳定性 | 算力/系统成本 | 适合场景 |
|---|---|---|---|---|
| 固定比例混合 | 预先定义各域权重 | 高 | 低 | 大规模预训练基线、强复现需求 |
| DoReMi 类自适应 | 按域级 loss 差动态调权 | 中 | 中 | 不确定最优配方、多域能力补短板 |
| SampleMix 多指标 | 按样本级质量与多样性调权 | 中到低 | 高 | 数据稀疏、异构强、需细粒度筛选 |

适用边界也要说清。第一，数据混合主要解决“训练分布怎么配”，不解决“原始数据本身质量差”的问题。脏数据再精妙地混，也不会自动变成好数据。第二，如果你的任务非常单一，比如只做垂直代码补全，小而纯的数据集往往比复杂混合更有效。第三，在多模态或强监督任务里，“混合策略”可能不再是域比例问题，而是损失权重、模态对齐和课程学习问题，这时就不能把文本预训练的经验直接照搬。

---

## 参考资料

- Michael Brenndoerfer, *Data Mixing: Domain Proportions, Quality Weighting, Optimal Mixing*, Language AI Handbook, 2026. https://mbrenndoerfer.com/writing/data-mixing-domain-proportions-quality-weighting-optimal
- EmergentMind, *Adaptive Mixing Training Strategies*, 2025. https://www.emergentmind.com/topics/mixing-training-strategy
- EmergentMind, *Data Mixture Learning: Theory & Methods*, 2025. https://www.emergentmind.com/topics/data-mixture-learning
