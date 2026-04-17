## 核心结论

投机采样（Speculative Decoding）是一种**不改变目标模型输出分布**的推理加速方法。它的做法不是让大模型一次生成更多 token，而是让一个更快的小模型先写出一段“草稿”，再由大模型一次性验证这段草稿里哪些 token 可以直接接受。

这里的“草稿模型”是指参数更少、单步前向更快的模型；“目标模型”是指最终负责保证质量和分布正确性的原始大模型。两者配合后，目标模型不再每次只确认 1 个 token，而是经常能一次确认一个前缀，因此减少串行前向次数。

它解决的是自回归生成的核心瓶颈：**大模型逐 token 解码时，单步计算量不大，但每一步都要重复读权重、调度 kernel、访问 KV Cache，整体更像 memory-bound，而不是 compute-bound。**“memory-bound”可以直白理解为：速度主要卡在搬运数据，不是卡在算力不够。

一个直观流程可以写成两行：

`draft:  x_t -> y1 y2 y3 y4 y5`  
`target: x_t -> [验 y1][验 y2][验 y3][验 y4][验 y5] -> 接受最长前缀`

如果草稿模型和目标模型分布足够接近，且草稿延迟显著更低，那么在保持质量不变的前提下，吞吐通常能明显提升。Leviathan 2023 一类实验里，常见条件是 $\alpha \approx 0.53 \sim 0.75$、草稿长度 $k=5\sim7$，吞吐可到约 2.3 到 3.4 倍。

---

## 问题定义与边界

问题先说清楚：自回归生成一次只输出一个 token。token 就是模型处理的最小文本单位，可以理解为“词片”。这意味着即使一句话后面 5 个 token 很容易预测，目标模型也必须走 5 次串行前向。

这带来三个后果：

| 项目 | 含义 | 对性能的影响 |
|---|---|---|
| 目标模型限制 | 每步只生成 1 个 token | 串行依赖强，端到端延迟高 |
| 草稿模型角色 | 快速提出未来 $k$ 个 token 候选 | 把“多步猜测”提前做掉 |
| 希望获得的度量 | 吞吐、首 token 后延迟、acceptance rate | 衡量是否真的比纯目标快 |

“acceptance rate”就是接受率，直白解释是：草稿里有多少 token 被目标模型认可。

边界也必须明确：

1. 目标不是“近似正确”，而是**保持目标模型原始分布不变**。
2. 草稿模型必须足够快，否则验证前节省的时间会被草稿生成成本吃掉。
3. 草稿模型和目标模型最好共享或高度兼容 tokenizer。tokenizer 是把文本切成 token 的规则；规则不一致会导致验证逻辑失效。
4. 该方法更适合长文本、高吞吐推理；如果只关心单个极短请求的首 token 时延，收益可能有限。

玩具例子：  
假设目标模型要输出“今天 上海 下雨 了 吗”。传统解码要连续跑 5 次目标前向。投机采样里，小模型先猜出这 5 个 token，大模型一次性检查，若前 4 个都对、只在第 5 个分歧，那么本轮就直接确认前 4 个，下一轮从第 5 个继续。原本 4 次串行确认，被压缩成了 1 次目标验证。

真实工程例子：  
机器翻译或摘要服务通常是批量请求、平均输出长度较长。此时 GPU 更容易被权重读写和 KV Cache 访问拖住，而不是纯算力打满。引入一个同 tokenizer 的小草稿模型后，目标模型可以把多个未来位置一起验证，从而提高整体 token/s。

---

## 核心机制与推导

核心机制分两步：

1. 草稿模型按自己的分布连续采样 $k$ 个 token。
2. 目标模型对这 $k$ 个位置做一次并行验证，并按校正规则决定接受多长的前缀。

设第 $i$ 个草稿 token 为 $x_i$，则它的接受概率写成：

$$
\alpha_i = \min\left(1,\frac{p_{\text{target}}(x_i \mid x_{<i})}{p_{\text{draft}}(x_i \mid x_{<i})}\right)
$$

这里的意思很直接：  
如果目标模型对这个 token 的概率不低于草稿模型，就全额接受；如果更低，就按比例接受。这个设计保证被接受的部分恰好对应两个分布的重叠区域，因此最终输出分布仍等于目标模型分布。

若把平均接受率近似记成常数 $\alpha$，草稿长度为 $k$，则一轮中**平均被确认的 token 数**可近似写成：

$$
\tau = \sum_{i=0}^{k}\alpha^i = \frac{1-\alpha^{k+1}}{1-\alpha}
$$

这里 $\tau$ 可以理解为“一轮到底推进了多少 token”。当 $\alpha$ 越接近 1，$\tau$ 就越接近 $k+1$。

若草稿模型每生成 1 个 token 的相对成本是 $c$，即草稿总成本约为 $kc$ 个“目标前向单位”，那么理论加速可写成：

$$
\text{speedup} \approx \frac{\tau}{1+kc}
$$

这比“只看接受率”更实用，因为它同时考虑了草稿本身并非免费。

再看一个最小数值例子。设：

- $\alpha = 0.6$
- $k = 5$
- $c = 0.1$

则：

$$
\tau = \frac{1-0.6^6}{1-0.6} \approx 2.38
$$

单轮相对成本为：

$$
1 + kc = 1 + 5 \times 0.1 = 1.5
$$

所以加速约为：

$$
\frac{2.38}{1.5} \approx 1.59
$$

这说明即使接受率只有 0.6，只要草稿足够便宜，仍然有接近 60% 的速度提升。

还可以顺手解释一个常见直觉：为什么有人会写出大约 $1/(1-\alpha^k)$ 的收益表达？因为当草稿长度固定、且系统经常能一轮接受完整草稿时，串行中断概率大致受 $\alpha^k$ 控制。这种写法适合做粗略直觉，但工程上更常用的是 $\tau/(1+kc)$，因为它把草稿开销也算进去了。

---

## 代码实现

实现上最关键的是三件事：

1. 草稿模型连续生成 $k$ 个 token，并保存对应条件概率。
2. 目标模型对“原上下文 + 草稿前缀”做一次前向，拿到每个位置的 logits。
3. 从左到右做接受测试，遇到第一个拒绝位置就停止，并从目标分布做校正采样。

下面给一个可运行的 Python 玩具实现。它不是完整 Transformer 推理代码，但完整体现了“草稿生成 + 并行验证 + 最长前缀接受”的逻辑。

```python
from math import prod

def expected_accepted(alpha: float, k: int) -> float:
    assert 0.0 <= alpha <= 1.0
    assert k >= 1
    if alpha == 1.0:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def speedup(alpha: float, k: int, c: float) -> float:
    assert 0.0 <= c
    tau = expected_accepted(alpha, k)
    return tau / (1 + k * c)

def accept_prefix(draft_tokens, p_draft, p_target):
    """
    draft_tokens: 草稿 token 列表
    p_draft[i]: 草稿模型给 draft_tokens[i] 的条件概率
    p_target[i]: 目标模型给同一 token 的条件概率
    返回可接受前缀长度
    """
    assert len(draft_tokens) == len(p_draft) == len(p_target)
    accepted = 0
    for pd, pt in zip(p_draft, p_target):
        alpha_i = min(1.0, pt / pd)
        # 玩具实现里，用“alpha_i 是否等于 1”模拟必然接受；
        # 真实系统这里是按 alpha_i 做随机接受/拒绝。
        if alpha_i < 1.0:
            break
        accepted += 1
    return accepted

# 数值例子
tau = expected_accepted(0.6, 5)
sp = speedup(0.6, 5, 0.1)

assert round(tau, 2) == 2.38
assert round(sp, 2) == 1.59

# 前缀接受例子
draft = ["今天", "上海", "下雨", "了", "吗"]
p_d = [0.9, 0.8, 0.7, 0.8, 0.6]
p_t = [0.95, 0.85, 0.72, 0.50, 0.4]

# 前三个 token 目标概率 >= 草稿概率，第四个开始不是
assert accept_prefix(draft, p_d, p_t) == 3
```

真实系统中的伪代码通常更接近下面这个流程：

```python
while not stop:
    draft_tokens, draft_probs = draft.generate_k(context, k)
    target_logits = target.forward_once(context, draft_tokens)

    accepted = 0
    for i in range(k):
        token = draft_tokens[i]
        p_d = draft_probs[i][token]
        p_t = softmax(target_logits[i])[token]
        alpha = min(1.0, p_t / p_d)

        if random_uniform() <= alpha:
            accepted += 1
        else:
            repaired = sample_from_corrected_distribution(target_logits[i], draft_probs[i])
            output(repaired)
            context.append(repaired)
            break

    output(draft_tokens[:accepted])
    context.extend(draft_tokens[:accepted])

    if accepted == k:
        extra = sample_from_target(target_logits[k])
        output(extra)
        context.append(extra)
```

真实工程例子：  
在一个在线摘要服务里，目标模型是 30B 级别，草稿模型是同语料蒸馏出来的 1B 级别。请求进入后，草稿模型先跑出 6 个 token；目标模型用一次前向验证 6 个位置。服务侧同时记录每轮 acceptance rate、平均 accepted prefix、草稿延迟占比，并根据这些指标动态把 $k$ 从 4 调到 8。这样做的重点不是“永远用最大 $k$”，而是让收益稳定高于草稿额外成本。

---

## 工程权衡与常见坑

投机采样不是“白送的加速”。它很依赖草稿质量、延迟占比和任务分布。

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| $\alpha$ 过低 | 大量草稿被拒，吞吐下降 | 用蒸馏后的草稿模型，确保分布接近 |
| tokenizer 不一致 | token 边界不同，无法逐位验证 | 草稿与目标尽量共用 tokenizer |
| 草稿不够快 | $c$ 偏大，理论收益被抵消 | 控制草稿规模，优先低延迟模型 |
| $k$ 过大 | 草稿多写了但大多不被接受 | 动态调节 $k$，按 acceptance rate 自适应 |
| 忽略采样策略差异 | 温度、top-p 不一致导致分布漂移 | 草稿和目标的解码设置要对齐 |
| 只看 token/s | 单请求时延可能没降反升 | 同时监控 p50/p95 latency 与吞吐 |

一个常见误区是只看“草稿越大越准，所以越好”。这不对。草稿更大通常意味着 $c$ 变大。如果 $\alpha$ 的提升不足以覆盖额外成本，总体反而变慢。工程上要盯的是整体比值 $\tau/(1+kc)$，不是单独盯 $\alpha$。

另一个坑是任务分布漂移。比如代码补全、数学推导、开放式创作的 token 分布差异很大。摘要任务上训练出来的草稿模型，换到代码生成上，$\alpha$ 可能明显下降。原因很简单：草稿学到的是某个分布的近似，不是所有任务上的统一近似。

再举一个真实坑：  
如果直接拿一个随机初始化后简单微调的小模型做摘要草稿，$\alpha$ 可能低于 0.4。表面看草稿生成很快，但每轮只确认 1 到 2 个 token，且频繁触发拒绝后的修正采样，最终比纯目标解码更慢。后来改成同 tokenizer、同语料蒸馏，并把温度和 top-p 对齐后，acceptance rate 才回到可用范围。

---

## 替代方案与适用边界

投机采样优化的是“跨步串行依赖”，不是所有推理瓶颈。很多时候，先做更便宜的优化更合适。

| 方案 | 主要解决什么 | 实现难度 | 典型收益边界 | 更适合什么场景 |
|---|---|---|---|---|
| Speculative Decoding | 多步串行解码慢 | 中到高 | 取决于 $\alpha,k,c$ | 长输出、高吞吐服务 |
| KV Cache | 避免重复计算历史注意力 | 低到中 | 几乎是基础配置 | 所有自回归推理 |
| 量化 | 降低权重访存和显存占用 | 中 | 常见 1.2x 到数倍 | 显存紧张、单步慢 |
| 连续批处理 | 提高 GPU 利用率 | 中到高 | 服务吞吐明显提升 | 多请求并发服务 |
| 多卡流水线/张量并行 | 扩展大模型部署能力 | 高 | 依硬件拓扑而定 | 超大模型部署 |
| 剪枝/MoE | 减少有效计算量 | 高 | 依模型结构而定 | 训练和部署联动优化 |

适用边界可以概括成三条：

1. 如果你的主要问题是**单步前向太慢**，先试量化、内核优化、KV Cache。
2. 如果你的主要问题是**大量长文本请求导致串行步数太多**，投机采样更值得上。
3. 如果你拿不到一个足够快且足够像目标模型的草稿模型，就不要硬上。此时复杂度增加了，收益未必成立。

一句简单判断标准：  
如果你只想降低单 token latency，先做量化和缓存；如果你已经把单步做得差不多了，而整体仍被“一个 token 一个 token 地走”拖住，再考虑 speculative 架构。

---

## 参考资料

| 名称 | 内容聚焦 | URL |
|---|---|---|
| Leviathan et al., 2023, Fast Inference from Transformers via Speculative Decoding | 原始论文，给出算法与分布不变证明 | https://arxiv.org/abs/2211.17192 |
| BentoML Handbook: Speculative Decoding | 工程流程、草稿/目标协同机制 | https://bentoml.com/llm/inference-optimization/speculative-decoding |
| Phil Krav: Speculative Decoding | 接受率、$\tau$、speedup 的公式推导 | https://philkrav.com/posts/speculative/ |
| Data Processing Club: Speculative Decoding | 对 Leviathan 实验结果的工程化解读 | https://data-processing.club/speculative/ |
| Machine Learning Mastery: Guide to Speculative Decoding | 面向实践者的直观解释和常见问题 | https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-speculative-decoding/ |
| Lxyuan 博客：Speculative Decoding LLM Inference | 接受概率公式与流程拆解 | https://lxyuan0420.github.io/posts/speculative-decoding-llm-inference |
