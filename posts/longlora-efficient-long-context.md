## 核心结论

LongLoRA 的核心不是重新发明一种新的长上下文模型，而是把“长上下文微调太贵”这个工程问题拆成两部分处理：

1. 参数更新部分继续用 LoRA。LoRA 是“低秩适配”，白话解释就是：不改整块大权重，只外挂一小块可训练增量参数。
2. 注意力计算部分在训练时换成 Shift Short Attention，也常写成 S²-Attention。它的白话解释是：不再让所有 token 全局两两互看，而是先分块，再让部分注意力头做错位平移，用更便宜的局部计算近似获得跨块信息流动。

因此，LongLoRA 解决的是“如何低成本把已有大模型微调到更长上下文”，而不是“如何让推理阶段永远使用一种新注意力结构”。它的一个关键点是：训练时可以改计算图来省显存和算力，推理时通常仍回到原模型的标准 attention 形式。

可以把它概括成下面这个组合：

$$
\text{LongLoRA} = \text{LoRA-style parameter update} + \text{S}^2\text{-Attention for training}
$$

如果目标是把一个原本只适合 4K 上下文的模型，较低成本扩到 32K、64K 甚至更长做微调，LongLoRA 是很有代表性的方案。如果目标是从底层设计一个原生超长推理模型，它就不是唯一解，甚至可能不是最合适的解。

---

## 问题定义与边界

先定义问题。为什么长上下文微调困难？核心不是“序列变长”这四个字本身，而是序列变长后，标准自注意力的计算和显存开销上升得太快。

Transformer 的自注意力里，每个 token 都要和其他 token 建立相关性。若序列长度为 $n$，注意力矩阵大小大致是 $n \times n$，因此主要复杂度会带有 $O(n^2)$ 的量级。这里的“量级”白话讲就是：长度翻几倍，成本不只是线性涨，而是近似按平方涨。

举个直接的数量级例子：

| 序列长度 | 注意力元素数量近似 | 相对 4K 的倍数 |
|---|---:|---:|
| 4K | $4096^2 \approx 1.68 \times 10^7$ | 1 |
| 8K | $8192^2 \approx 6.71 \times 10^7$ | 4 |
| 32K | $32768^2 \approx 1.07 \times 10^9$ | 64 |
| 100K | $100000^2 = 10^{10}$ | 约 596 |

这还只是单层、单次前向里注意力矩阵的量级感知，真实训练还叠加了多层、batch、反向传播、中间激活保存，因此显存和训练时间都会迅速失控。

所以 LongLoRA 处理的不是一般的“模型微调”，而是更窄的一类问题：

- 已有基座模型通常是短上下文或中等上下文。
- 你想在有限资源下，把它微调到能处理更长输入。
- 你能接受训练阶段对 attention 做结构性改造。
- 你更关心“训练成本降下来”，而不是“推理时也换一种全新体系”。

它的边界也要说清楚：

| 方案 | 计算成本 | 训练方式 | 适用场景 |
|---|---|---|---|
| 普通短上下文微调 | 低到中 | 标准全局 attention | 4K 附近任务、常规 SFT |
| 直接长上下文微调 | 很高 | 标准全局 attention 直接拉长 | 资源充足、追求最直接训练 |
| LongLoRA | 中到较低 | LoRA + 训练时 S²-Attention | 想低成本扩长上下文 |

因此，LongLoRA 不是通用推理加速器，也不是所有任务都必须使用的技术。如果任务本身只依赖 2K 到 4K 输入，比如普通客服问答、短代码补全、单轮指令跟随，那么引入长上下文微调很可能收益有限。

---

## 核心机制与推导

LongLoRA 的机制可以拆成两层：LoRA 的低秩更新，以及 S²-Attention 的训练时局部错位注意力。

先看 LoRA。标准线性层权重记作 $W$，LoRA 不直接更新 $W$，而是学习一个低秩增量 $\Delta W$：

$$
W' = W + \Delta W,\qquad \Delta W = BA
$$

其中：

- $W$ 是冻结的原始权重。
- $A$ 和 $B$ 是新增的小矩阵。
- “低秩”意思是中间维度 $r$ 很小，通常远小于原权重维度。

如果原权重是 $d \times k$，直接全量训练要更新 $dk$ 个参数；LoRA 只更新 $dr + rk$ 个参数。当 $r \ll d,k$ 时，参数量和优化状态都会显著下降。这就是它“参数高效”的来源。

但只靠 LoRA 还不够。因为在长上下文训练中，真正爆炸的是 attention 的序列维度成本，不是参数量本身。于是 LongLoRA 把第二刀砍在 attention 上。

设输入序列为：

$$
X = [x_1, x_2, \dots, x_n]
$$

标准全局 attention 相当于让任意 $x_i$ 都能和任意 $x_j$ 计算相关性。S²-Attention 改成三步：

1. 把序列按 block 大小 $g$ 切分。
2. 对部分 attention heads，把序列平移 $g/2$ 个位置。
3. 只在每个 block 内做局部 attention，最后再移回原顺序。

形式上可以写成：

$$
\tilde X = \operatorname{Roll}(X, -g/2)
$$

$$
Y = \operatorname{Unshift}\big(\operatorname{BlockAttention}(\tilde X)\big)
$$

这里的 `Roll` 可以理解为循环平移，`BlockAttention` 可以理解为“每块内部自注意力”。

为什么这个结构有效？看一个玩具例子。

原序列是：

$$
[A,B,C,D,E,F,G,H]
$$

设 block 大小 $g=4$。不平移时分块为：

| block | token |
|---|---|
| 1 | A B C D |
| 2 | E F G H |

如果一组注意力头把序列左移 2 位，得到：

$$
[C,D,E,F,G,H,A,B]
$$

再按 4 个一块切：

| shifted block | token |
|---|---|
| 1 | C D E F |
| 2 | G H A B |

这时 `E` 和 `F` 不再只与原来 block 2 的成员局部交互，它们也会进入包含 `C,D` 的局部块中。换句话说，跨 block 的邻接关系通过“错位分组”被建立了。

这不是完整全局 attention，因为 `A` 不会在同一步里直接看见所有远端 token；但它也不是彼此完全隔离的局部窗口，因为不同头的分块边界不同，信息可以逐层、逐头传播。工程上这是一种典型折中：牺牲一部分精确的全局两两连接，换取更低成本的近似全局信息流动。

从复杂度直觉看，如果每个 block 长度是 $g$，共有 $n/g$ 个 block，那么局部 attention 的主要计算近似变成：

$$
\frac{n}{g}\cdot g^2 = ng
$$

相较于全局 attention 的 $n^2$，当 $g \ll n$ 时差距非常大。比如从 $n=32768$、$g=2048$ 的量级看，核心开销已从“和全序列平方相关”变成“和序列长度乘一个固定块长相关”。

真实工程例子是长文档问答。假设输入是：

- 一份 120 页合同
- 若干补充协议
- 用户提问：“第 3 方违约后的通知期限和赔偿上限分别是什么？”

如果模型只能看 4K 上下文，它可能只能截取局部章节，容易丢掉前后引用关系。如果直接用全局 32K attention 做微调，成本又非常高。LongLoRA 的价值就在这里：它让训练时更便宜，但仍能逼着模型学习“答案依赖远处片段”的模式。

---

## 代码实现

实现层面通常不是从零写一个模型，而是在已有 Transformer attention 模块上做替换，并只训练 LoRA 适配器及少量必要参数。

下面先给一个最小可运行的 Python 玩具实现。它不是真实训练代码，但能把“shift -> block attention -> unshift”的流程跑通。

```python
from typing import List

def roll_left(xs: List[str], k: int) -> List[str]:
    k %= len(xs)
    return xs[k:] + xs[:k]

def roll_right(xs: List[str], k: int) -> List[str]:
    k %= len(xs)
    return xs[-k:] + xs[:-k]

def blocks(xs: List[str], block_size: int) -> List[List[str]]:
    assert len(xs) % block_size == 0, "toy example requires divisible length"
    return [xs[i:i + block_size] for i in range(0, len(xs), block_size)]

def shift_short_partition(xs: List[str], block_size: int) -> List[List[str]]:
    assert block_size % 2 == 0, "block size should be even for half-block shift"
    shifted = roll_left(xs, block_size // 2)
    return blocks(shifted, block_size)

def unshift_from_partition(partitioned: List[List[str]], block_size: int) -> List[str]:
    flat = [x for block in partitioned for x in block]
    return roll_right(flat, block_size // 2)

tokens = ["A", "B", "C", "D", "E", "F", "G", "H"]
parts = shift_short_partition(tokens, 4)

assert parts == [["C", "D", "E", "F"], ["G", "H", "A", "B"]]

restored = unshift_from_partition(parts, 4)
assert restored == tokens

# 证明 E 落入了和 C、D 同一个 shifted block
assert "E" in parts[0] and "C" in parts[0] and "D" in parts[0]

print("ok")
```

真实训练中，流程通常更接近下面这个伪代码：

```python
def forward(hidden_states):
    q, k, v = project_qkv(hidden_states)

    q1, q2 = split_heads(q)
    k1, k2 = split_heads(k)
    v1, v2 = split_heads(v)

    # 一半 heads 不平移
    out1 = block_local_attention(q1, k1, v1, block_size=g)

    # 一半 heads 先 shift，再做局部 attention，最后 unshift
    q2 = shift(q2, step=g // 2)
    k2 = shift(k2, step=g // 2)
    v2 = shift(v2, step=g // 2)

    out2 = block_local_attention(q2, k2, v2, block_size=g)
    out2 = unshift(out2, step=g // 2)

    out = merge_heads(out1, out2)
    out = lora_output_projection(out)
    return out
```

训练参数配置上，常见开关可以概括成：

| 模块 | 常见做法 | 原因 |
|---|---|---|
| Attention/MLP 主权重 | 冻结 | 保持参数高效 |
| Q/K/V/O 投影上的 LoRA | 训练 | 主要适配入口 |
| Embedding | 常常设为可训练 | 长上下文扩展时更稳 |
| LayerNorm / RMSNorm | 常常设为可训练 | 帮助分布重新校准 |
| Attention 计算图 | 训练时改成 S²-Attention | 降低长序列成本 |

这里有一个容易误解的点：LongLoRA 不是“只加 LoRA 就完成了”。论文和开源实现都强调，长上下文扩展时仅靠 LoRA 往往不够稳，embedding 和 normalization 层适度参与训练，效果更可靠。

---

## 工程权衡与常见坑

LongLoRA 省的是训练成本，不是白拿性能。它有明确代价。

第一，注意力实现复杂度上升。你不再只是调用标准 flash attention 或框架默认实现，而是要引入 block 切分、head 分组、shift/unshift、mask 处理，以及和并行训练逻辑的兼容。

第二，数据要求更高。长上下文训练不是把文本拼长就行，样本必须真的依赖远距离信息。否则模型只是在昂贵地学习“长输入里仍然回答局部问题”。

第三，训练和推理语义容易被混淆。LongLoRA 的核心优化发生在训练图，而不是说线上推理就该永远使用同样的 shifted 局部注意力。

常见坑可以直接列成表：

| 坑位 | 表现 | 规避方式 |
|---|---|---|
| 只挂 LoRA，不动 embedding/norm | 长度扩展后 loss 波动大，效果不稳 | 允许 embedding 与 norm 一起训练 |
| 只喂超长样本 | 长文档任务提升，但短问答、短指令能力回退 | 混入短样本，保持原任务分布 |
| 把训练 trick 当成推理 trick | 线上推理结构和预期不一致，延迟或质量异常 | 明确区分“训练省成本”和“推理部署结构” |
| block size 选得过小 | 成本低了，但跨段建模明显不够 | 根据任务依赖范围调大 block |
| 任务本身不需要长依赖 | 训练成本增加，但收益很小 | 先验证样本是否真的依赖远距离证据 |
| 数据只是机械拼接 | 模型学不到真正检索和整合能力 | 让答案必须跨段、跨页、跨章节抽取 |

一个真实工程坑是数据配比。比如你做企业知识库问答，训练集全部是 32K 长文档样本，看起来很“对题”，但最后模型可能在“总结这封短邮件”“改写一句话”这类普通短任务上明显退化。原因不是 LongLoRA 本身有问题，而是训练分布被你推得过于单一。官方实践里混入短 QA，就是为了避免这种回退。

---

## 替代方案与适用边界

LongLoRA 不是唯一方案。做长上下文通常至少有四类思路：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| LongLoRA | 微调成本低，适合在现有模型上扩长 | 训练图更复杂，不是原生长模型 | 预算有限，想把已有模型从 4K 拉到 32K/64K |
| RoPE 外推/缩放 | 改动小，常可直接在位置编码层处理 | 能力提升有限，容易在超长处失真 | 想低改动试探性扩窗 |
| 原生长上下文模型 | 结构和数据从一开始就为长序列设计 | 训练与获取成本高 | 从头做产品级长上下文能力 |
| 稀疏注意力/线性注意力 | 理论上更省，适合极长序列 | 实现、稳定性和效果取舍复杂 | 序列非常长，且能接受结构变化 |
| 分段检索/RAG | 不必让模型一次看完整文档 | 依赖召回质量，跨段推理有上限 | 知识问答、文档检索类任务 |

怎么选，关键看你的目标是什么。

如果你的目标是：“我已经有一个现成的 LLaMA 系模型，现在想低成本把上下文从 4K 扩到 32K，主要做长文档问答或摘要。”那么 LongLoRA 很合适，因为它兼顾参数效率和训练成本。

如果你的目标是：“我要做一个原生支持超长推理、推理阶段也强依赖远距离精确建模的模型。”那么原生长上下文训练、稀疏注意力结构，甚至任务级检索增强，都可能比 LongLoRA 更合适。

如果你的目标只是：“偶尔希望多塞一点上下文进去，不想改训练图。”那么位置编码外推类方法往往是更低门槛的第一步。

所以 LongLoRA 的适用边界可以压缩成一句话：它最适合“已有模型、预算有限、训练期可改结构、目标是扩长上下文微调能力”的场景。

---

## 参考资料

1. [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
2. [MIT Han Lab: LongLoRA Project Page](https://hanlab.mit.edu/projects/longlora)
3. [LongLoRA Official Code Repository](https://github.com/dvlab-research/LongLoRA)
4. [LongQLoRA Repository](https://github.com/yangjianxin1/LongQLoRA)
5. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
