## 核心结论

Parallel Attention-FFN 的核心改动很简单：把同一层里的 Attention 和 FFN 从“前后串行”改成“同输入并行”，再一起加回残差路径。这里的 Attention 指注意力模块，白话解释是“让每个 token 按相关性去看别的 token”；FFN 指前馈网络，白话解释是“对每个位置单独做一次更强的非线性变换”。

标准 pre-norm Transformer 常见写法可以简化成：

$$
h = x + A(N(x))
$$

$$
y = h + F(N(h))
$$

其中 $N(\cdot)$ 是归一化，$A(\cdot)$ 是 Attention，$F(\cdot)$ 是 FFN。Parallel Attention-FFN 改成：

$$
y = x + A(N(x)) + F(N(x))
$$

这意味着 FFN 不再等待 Attention 的输出，而是和 Attention 同时读取同一个输入 $x$。它减少的是“层内串行依赖深度”，不是自动减少总计算量。真正收益通常来自三点：关键路径更短、硬件更容易并行执行、实现上更容易做算子融合。

可以把两者的依赖关系直接看成：

```text
串行：x -> Attention -> FFN -> y
并行：x -> Attention --\
         -> FFN --------> y
```

如果只记一句话，应记住：Parallel Attention-FFN 改的是数据依赖图，不是把 Attention 或 FFN 的数学定义本身换掉。

---

## 问题定义与边界

问题先定义清楚。Transformer 层通常包含两类主要计算：

| 模块 | 作用 | 是否依赖上下文 |
|------|------|----------------|
| Attention | 让每个位置聚合其他位置的信息 | 是 |
| FFN | 对每个位置做非线性特征变换 | 否，按位置独立 |

标准层结构里，FFN 往往接在 Attention 后面，因此形成一条更长的串行链。对硬件来说，串行链越长，可同时发出的工作越少，流水线更难填满。并行结构的想法不是“把模型任务改了”，而是把同一层内的连接方式改了。

下面这张表能把边界说清：

| 维度 | 串行 Transformer | Parallel Attention-FFN |
|------|------------------|------------------------|
| 依赖关系 | Attention 先，FFN 后 | Attention 和 FFN 并行 |
| 输入来源 | FFN 依赖 Attention 输出 | 两者都直接读同一输入 |
| 改动对象 | 层内连接方式 | 层内连接方式 |
| 是否必然更省 FLOPs | 否 | 否 |
| 是否等同于序列并行 | 否 | 否 |
| 是否等同于张量并行 | 否 | 否 |

这里的 FLOPs 是浮点运算次数，白话解释是“理论上要做多少乘加计算”。Parallel Attention-FFN 通常不把 Attention 和 FFN 变小，所以总 FLOPs 往往没有本质下降。它主要优化的是“怎么排这些计算”，不是“把这些计算删掉”。

玩具例子可以先用一句最直白的话理解：如果 FFN 必须等 Attention 先出结果，GPU 就只能先做一半工作；如果两者都直接看同一个输入，就能同时发两段工作，再在末尾合并。

它也有明确边界。它不是序列并行。序列并行是把 token 维度切开，分给不同设备处理；它不是张量并行。张量并行是把大矩阵切开，分散到多卡；它也不是“同算力下参数必然更多”。参数能否增大，取决于你是否把省下来的墙钟时间、调度开销或层数预算重新投入到模型规模上。

---

## 核心机制与推导

先统一符号。设：

- $x$：层输入
- $N(\cdot)$：LayerNorm 或 RMSNorm，白话解释是“把激活尺度调整到更稳定的范围”
- $A(\cdot)$：Attention
- $F(\cdot)$：FFN

那么标准串行 pre-norm 块可以写成：

$$
h = x + A(N(x))
$$

$$
y = h + F(N(h))
$$

Parallel Attention-FFN 则写成：

$$
y = x + A(N(x)) + F(N(x))
$$

这个式子说明两件事。

第一，Attention 和 FFN 的输入一样，都是 $N(x)$。  
第二，残差只在最后统一汇合。这里的残差，白话解释是“把原始输入直接加回输出，减轻深层训练困难”。

为什么这会改变依赖深度？因为在串行结构里，$F$ 的输入是 $N(h)$，而 $h$ 依赖 $A(N(x))$，所以 FFN 必须等待 Attention 完成。在并行结构里，$F$ 直接读取 $N(x)$，不再等待中间结果，依赖图被压扁了。

也存在 post-norm 风格写法：

$$
y = N(x + A(x) + F(x))
$$

两类公式的语义差别在于归一化放在哪里。pre-norm 更常见于大模型训练，因为训练稳定性通常更好；post-norm 更接近早期 Transformer 结构，但深层训练时更容易不稳定。无论哪一种，Parallel 的本质约束都不变：$A$ 和 $F$ 必须先并行算，不能把 $F$ 的输入接成 $A$ 的输出，否则就退化回串行。

看一个最小数值例子。先忽略归一化，设：

$$
x = [1, 2]
$$

$$
A(x) = [0.3, -0.1]
$$

$$
F(x) = [0.2, 0.4]
$$

并行时：

$$
y = x + A(x) + F(x) = [1.5, 2.3]
$$

这就是“玩具例子”。如果你改成串行，FFN 看到的输入不再是原始 $x$，而是某个已经叠加了 Attention 结果的中间状态，因此输出一般不会还是 $[0.2, 0.4]$。两种结构不是简单的重排括号，而是不同的数据流。

更进一步，从表达能力角度看，并行结构并不是把 FFN 变弱了。它只是把“先混合上下文，再做逐位置变换”改成“同一输入上同时做上下文混合和逐位置变换，再合并”。因此它改变的是层内组合顺序，而不是删掉其中一类能力。

---

## 代码实现

代码层面最重要的点只有一个：同一个输入 `x` 同时送进两个分支。下面是一个可运行的最小 Python 版本，用列表模拟并行块的行为：

```python
def add(a, b):
    return [x + y for x, y in zip(a, b)]

def attention(x):
    # 玩具 attention：返回一个固定偏移，真实模型里它来自 QK^T 和 V 的计算
    return [0.3, -0.1]

def ffn(x):
    # 玩具 FFN：返回另一个固定偏移
    return [0.2, 0.4]

def parallel_block(x):
    return add(add(x, attention(x)), ffn(x))

y = parallel_block([1.0, 2.0])
assert y == [1.5, 2.3]
```

对应到 PyTorch，结构通常长这样：

```python
import torch
import torch.nn as nn

class ParallelBlock(nn.Module):
    def __init__(self, hidden_size, attn, ffn):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = attn
        self.ffn = ffn

    def forward(self, x):
        h = self.norm(x)
        attn_out = self.attn(h)
        ffn_out = self.ffn(h)
        return x + attn_out + ffn_out
```

把它和串行写法对照，会更清楚：

```text
串行：
h = norm(x)
h = x + attn(h)
y = h + ffn(norm(h))

并行：
h = norm(x)
y = x + attn(h) + ffn(h)
```

真实工程例子可以看 `T5-large` 这一类 encoder-decoder 模型。工程上如果你把 attention 投影和 FFN 投影改成同层并行，就有机会减少层内等待，并让两边的大矩阵乘更容易统一调度或融合。公开工程讨论中，曾报告在 `T5-large` 上使用 parallel attention 后训练速度可提升约 `23%`，同时收敛表现保持不变。这里应把这个数字理解为“特定实现和硬件配置下的工程结果”，而不是结构本身的固定收益。

实现时至少检查四件事：

```text
1. pre-norm 还是 post-norm
2. 残差是在末尾一次相加，还是中间分两次相加
3. state dict 键名和张量形状是否兼容旧 checkpoint
4. 推理路径、缓存逻辑、并行 kernel 是否同步更新
```

如果只改前向公式，不改权重迁移和缓存逻辑，常见结果不是“效果略差”，而是直接无法加载旧权重或推理结果错误。

---

## 工程权衡与常见坑

工程上应优先看“吞吐、显存、兼容性”三项，而不是只看公式是否更短。

| 项目 | 可能收益 | 可能代价 |
|------|----------|----------|
| 训练速度 | 关键路径更短，吞吐更高 | 提升幅度依赖 kernel 和调度 |
| 硬件利用率 | 更容易并行发射算子 | 小模型上可能被调度开销抵消 |
| 显存 | 不一定改善 | 两个分支同时保留激活，峰值可能更高 |
| 权重迁移 | 可设计兼容方案 | norm 位置不一致时容易失败 |

常见坑主要有四类。

第一，把它和别的“并行”混淆。Parallel Attention-FFN 是层内依赖重排，不是分布式并行策略。你不能因为模型用了并行 Attention-FFN，就推断它一定用了序列并行或张量并行。

第二，默认它一定更省算力。这通常不成立。很多情况下 FLOPs 差不多，变的是墙钟时间和硬件利用率。墙钟时间，白话解释是“实际训练一轮花了多久”。

第三，只看训练 loss，不做下游验证。结构改变后，即便预训练 loss 接近，也可能在某些下游任务、长上下文行为或稳定性上出现差异。尤其是小模型，调度收益不明显时，结构改动带来的分布变化反而更值得检查。

第四，忽略归一化位置。pre-norm 和 post-norm 的差异不是代码风格问题，而是训练动力学问题。你如果把一个串行 pre-norm 模型随意改成并行 post-norm，再试图直接迁移权重，失败几乎是预期结果。

一个经常被忽略的点是显存峰值。并行结构让 Attention 分支和 FFN 分支更可能在相近时间段内保留激活，如果实现没有做重计算、融合或及时释放，中大型模型上显存可能先成为瓶颈，而不是算力。

---

## 替代方案与适用边界

Parallel Attention-FFN 适合什么场景？结论很明确：当模型瓶颈主要在层内调度和串行依赖，而不是某个单独算子本身时，它更有价值。

如果你的模型很小，或者实现本身已经把串行块优化得很好，那么理论上更短的关键路径不一定换来明显收益。因为调度开销、访存布局、kernel 启动成本，都可能把结构上的优势吃掉。

可以把它和几种替代方案放在一起看：

| 方案 | 目标 | 适用边界 |
|------|------|----------|
| Parallel Attention-FFN | 缩短层内依赖 | 关注训练吞吐、可改前向图 |
| 序列并行 | 切 token 维度 | 长序列、显存受限 |
| 张量并行 | 切参数矩阵 | 大模型多卡训练 |
| 保持串行但做融合 | 降低工程风险 | 兼容性优先、已有 checkpoint 多 |

这几类方案解决的问题并不相同。Parallel Attention-FFN 改的是计算图；序列并行、张量并行改的是并行维度；继续串行但做 kernel 融合，改的是实现层面的执行效率。不能把它们当成互相替代的同一件事。

还要补一条边界：公开资料提到 PaLM 使用 parallel layers，这可以作为结构实践的直接参考；但像 Grok-1 这类公开仓库，如果 README 只给出模型总参数、层数、上下文长度，而没有明确写出层结构，就不能把“采用 Parallel Attention-FFN”当作已核实事实。工程判断必须分清“论文明确写了”和“外部推测”。

---

## 参考资料

- 论文：PaLM: Scaling Language Modeling with Pathways  
  https://arxiv.org/abs/2204.02311

- 论文：Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design  
  https://arxiv.org/abs/2305.13297

- 工程讨论：PyTorch issue #95210, Add parallel attention layers and Multi-Query Attention (MQA) from PaLM to the fast path for transformers  
  https://github.com/pytorch/pytorch/issues/95210

- 公开仓库：Grok-1  
  https://github.com/xai-org/grok-1

- 背景论文：Attention Is All You Need  
  https://arxiv.org/abs/1706.03762

本文以论文和工程讨论为主，仓库资料只作辅助，不单独作为“某结构已被采用”的证据。
