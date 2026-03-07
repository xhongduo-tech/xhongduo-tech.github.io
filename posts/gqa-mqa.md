## 核心结论

GQA（Grouped-Query Attention，分组查询注意力）本质上是在“每个 Q 头都配一套 K/V”与“所有 Q 头共用一套 K/V”之间取中间值。这里的 Q 头指查询头，也就是负责“发起匹配”的注意力子空间；K/V 头指键和值头，分别负责“被匹配的索引”和“被取回的内容”。

标准多头注意力 MHA（Multi-Head Attention，多头注意力）有 $H$ 个 Q 头，也有 $H$ 个 K 头和 $H$ 个 V 头，因此 KV Cache 规模与 $H$ 成正比。MQA（Multi-Query Attention，多查询共享单键值注意力）把所有 Q 头压到同一组 K/V，上下文缓存最省，但表达能力下降更明显。GQA 用 $G$ 组 K/V 服务 $H$ 个 Q 头，每组负责 $H/G$ 个 Q 头，于是把 KV Cache 直接缩到原来的：

$$
\text{KV Cache 比例}=\frac{G}{H}
$$

这就是它在工程上重要的原因。推理阶段真正吃显存的往往不是参数本身，而是长上下文带来的 KV Cache。只要把可缓存的 K/V 头数从 $H$ 变成 $G$，显存和带宽压力就同步下降。

一个新手可直接记住的例子：

| 配置 | Q 头数 $H$ | KV 头数 | 比例 |
|---|---:|---:|---:|
| MHA | 64 | 64 | 1 |
| GQA | 64 | 8 | 1/8 |
| MQA | 64 | 1 | 1/64 |

如果一个模型有 64 个 query 头，改成 8 组 KV，那么每 8 个 query 头共享同一对 K/V，KV Cache 只剩原来的 1/8。LLaMA-2 70B 采用的就是这类思路：64 个 Q 头、8 个 KV 头，缓存压力约缩小 8 倍。

组级 K/V 的一个常见近似构造方式是对组内原始头做平均：

$$
K_g=\frac{1}{n_g}\sum_{i \in \mathcal{G}_g}K_i,\qquad
V_g=\frac{1}{n_g}\sum_{i \in \mathcal{G}_g}V_i
$$

这里 $\mathcal{G}_g$ 表示第 $g$ 组包含的头集合，$n_g$ 是该组头数。这个公式的含义很直接：一组里的多个 K/V 头被压缩成一个“组代表”。

---

## 问题定义与边界

问题先定义清楚：我们讨论的是**自回归推理阶段**的注意力优化，不是训练吞吐优化。自回归推理指模型每生成一个新 token，都要读取之前所有 token 的 K/V 缓存。上下文一长，读缓存的成本会快速变大。

标准 MHA 的问题不在“算不出结果”，而在“缓存太贵”。如果 batch 大小为 $B$，序列长度为 $L$，每头维度为 $d_h$，则单层 KV Cache 的量级可写成：

$$
\text{Cache}_{\text{MHA}} \propto 2 \cdot H \cdot d_h \cdot B \cdot L
$$

前面的 2 是因为要同时存 K 和 V。这里“量级”意思是先看增长趋势，不关心字节单位细节。GQA 把式子里的 $H$ 换成 $G$：

$$
\text{Cache}_{\text{GQA}} \propto 2 \cdot G \cdot d_h \cdot B \cdot L
$$

因此它解决的是一个很具体的边界问题：当推理瓶颈来自 KV Cache 显存和带宽，而不是来自前向矩阵乘法本身时，减少 KV 头数是高收益操作。

三种方案的边界可以直接对比：

| 方案 | Q 头 | K/V 头 | KV Cache | 性能影响 | 典型场景 |
|---|---|---|---|---|---|
| MHA | $H$ | $H$ | 最大 | 最小 | 训练、显存充足推理 |
| GQA | $H$ | $G$ | 降到 $G/H$ | 通常较小 | 长上下文推理、部署 |
| MQA | $H$ | 1 | 最小 | 更容易下降 | 极限显存场景 |

玩具例子可以用“64 个 Q 头分 8 组”来理解。原来每个头有自己独立的 K/V，现在每 8 个头共用一对 K/V。你损失了一部分“每头独立建模”的自由度，但换来的是 KV 件数从 64 变成 8。这种交换在 8k、32k、100k 长上下文下非常值钱。

真实工程边界也要说清楚。GQA 不是“所有模型都该改”。如果你的场景是短上下文、小 batch、显存充足，那么 KV Cache 不是主要矛盾，GQA 价值有限。相反，如果你做的是大模型长上下文在线服务，单卡或少卡部署，经常被 cache 撑满，那么 GQA 往往比再抠一点 kernel 优化更直接。

---

## 核心机制与推导

先从标准 MHA 写起。单个头的注意力计算为：

$$
\text{Attn}(Q_i,K_i,V_i)=\text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_h}}\right)V_i
$$

这里 $Q_i$ 是第 $i$ 个查询头的查询向量，$K_i$ 是键向量，$V_i$ 是值向量。白话说，$Q$ 负责“我要找什么”，$K$ 负责“每个历史 token 提供什么索引”，$V$ 负责“真正取回什么内容”。

MHA 的特点是每个头都各算各的：

$$
\{(Q_1,K_1,V_1),\dots,(Q_H,K_H,V_H)\}
$$

MQA 的做法最极端：所有 $Q_i$ 都共享同一个 $K,V$。GQA 则把头分组。设总 Q 头数为 $H$，KV 组数为 $G$，每组头数为：

$$
n_g = \frac{H}{G}
$$

若按等分组，则第 $g$ 组的 K/V 可由组内头做平均得到：

$$
K_g=\frac{1}{n_g}\sum_{i \in \mathcal{G}_g}K_i,\qquad
V_g=\frac{1}{n_g}\sum_{i \in \mathcal{G}_g}V_i
$$

然后组内每个 Q 头都去和同一个 $K_g,V_g$ 做注意力：

$$
O_i=\text{softmax}\left(\frac{Q_iK_g^\top}{\sqrt{d_h}}\right)V_g,\qquad i\in \mathcal{G}_g
$$

注意这里常见误区是“Q 也必须合并”。不是。GQA 的核心是**合并 K/V，不合并 Q**。因为推理时真正需要缓存的是历史 token 的 K 和 V，而当前步的 Q 只为本步计算服务，不需要跨步缓存。

### 玩具例子

假设：

- $H=64$
- $G=8$
- 每头维度 $d_h=128$
- batch $B=1$
- 上下文长度 $L=8192$

那么单层 KV 元素数：

$$
\text{MHA}: 2 \cdot 64 \cdot 128 \cdot 8192
$$

$$
\text{GQA}: 2 \cdot 8 \cdot 128 \cdot 8192
$$

两者比值为：

$$
\frac{2\cdot 8\cdot 128\cdot 8192}{2\cdot 64\cdot 128\cdot 8192}=\frac{1}{8}
$$

也就是说，GQA 只保留 MHA 的 12.5% KV 缓存。这个结论和模型层数无关，层数只会把总量一起放大。

### 真实工程例子

LLaMA-2 70B 使用的是“64 个 Q 头配 8 个 KV 头”的 GQA 结构。工程上这意味着：模型仍然保留较强的多 query 表达能力，但每层每个 token 只需要缓存 8 组 K 和 8 组 V，而不是 64 组。对长上下文推理来说，这种差距会从“几个 GB”变成“几十 GB”。

为什么不是直接上 MQA？因为 MQA 把所有 Q 头都绑到同一组 K/V，缓存继续更小，但头之间看到的“历史索引视角”过于一致，精度更容易掉。GQA 本质上是给你一个连续旋钮：从 $G=H$ 的 MHA 一路降到 $G=1$ 的 MQA，中间任何点都能选。

---

## 代码实现

实现层面可以分成两件事：

1. 模型结构原生就是 GQA，此时直接产生 $H$ 个 Q 头和 $G$ 个 K/V 头。
2. 已有 MHA 模型要改成 GQA，此时需要把原来的 K/V 头按组聚合，常见近似是均值合并。

下面给一个可运行的 Python 玩具实现，只演示“按组平均 K/V 并计算 GQA 输出”的核心逻辑。代码不依赖深度学习框架，目的是把张量形状看清楚。

```python
import math
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def gqa_attention(Q, K, V, num_kv_heads):
    """
    Q: [H, Tq, D]
    K: [H, Tk, D]  # 这里假设输入是原始 MHA 的 K
    V: [H, Tk, D]  # 这里假设输入是原始 MHA 的 V
    返回:
      output: [H, Tq, D]
      K_grouped: [G, Tk, D]
      V_grouped: [G, Tk, D]
    """
    H, Tq, D = Q.shape
    Hk, Tk, Dk = K.shape
    assert H == Hk
    assert D == Dk
    assert H % num_kv_heads == 0

    group_size = H // num_kv_heads

    # 按组对 K/V 做平均，得到 G 组缓存
    K_grouped = K.reshape(num_kv_heads, group_size, Tk, D).mean(axis=1)
    V_grouped = V.reshape(num_kv_heads, group_size, Tk, D).mean(axis=1)

    output = np.zeros_like(Q)

    for q_head in range(H):
        g = q_head // group_size
        scores = Q[q_head] @ K_grouped[g].T / math.sqrt(D)   # [Tq, Tk]
        probs = softmax(scores, axis=-1)                     # [Tq, Tk]
        output[q_head] = probs @ V_grouped[g]                # [Tq, D]

    return output, K_grouped, V_grouped

# 玩具数据：64 个 Q 头，压成 8 个 KV 头
H = 64
G = 8
Tq = 2
Tk = 5
D = 16

rng = np.random.default_rng(0)
Q = rng.normal(size=(H, Tq, D))
K = rng.normal(size=(H, Tk, D))
V = rng.normal(size=(H, Tk, D))

out, K_g, V_g = gqa_attention(Q, K, V, num_kv_heads=G)

assert out.shape == (H, Tq, D)
assert K_g.shape == (G, Tk, D)
assert V_g.shape == (G, Tk, D)
assert np.allclose(K_g[0], K[:8].mean(axis=0))
assert np.allclose(V_g[7], V[56:64].mean(axis=0))

# KV 头数从 64 变成 8，缓存比例是 1/8
assert K_g.shape[0] / K.shape[0] == 1 / 8
```

如果你是在 PyTorch 或推理框架里实现，核心改动通常有两处：

| 模块 | MHA | GQA |
|---|---|---|
| 投影层输出 | `Q,K,V` 都是 `H` 个头 | `Q` 是 `H` 个头，`K,V` 是 `G` 个头 |
| KV Cache | 缓存 `H` 组 K/V | 只缓存 `G` 组 K/V |

如果是把已有 MHA checkpoint 转成 GQA，常见流程是：

1. 先确定分组，比如 `64 -> 8`。
2. 把原来 64 个 K 头按顺序或按对齐策略分成 8 组。
3. 每组求平均，得到新的 8 个 K 头。
4. V 头做同样处理。
5. Q 投影保持不变。
6. 推理缓存逻辑改成只存 8 组 K/V。

这里要强调一句：**“无需从零重训练”不等于“完全无代价”**。均值合并是一种可工作的初始化方式，适合快速迁移；若要尽量保精度，通常还会做少量继续训练或先做头对齐。

---

## 工程权衡与常见坑

GQA 的好处很集中，坏处也很集中。它不是免费午餐，而是把“表达自由度”换成“缓存效率”。

最重要的权衡如下：

| 现象 | 原因 | 处理建议 |
|---|---|---|
| 精度下降 1% 到 3% | $G$ 太小，过多头共享同一组 K/V | 优先选 `G=8` 或 `G=16`，不要一上来压到 1 |
| 转换后效果不稳定 | 原始 K/V 头功能差异大，直接平均破坏结构 | 先做头对齐，再合并 |
| 吞吐没明显提升 | 当前瓶颈不在 KV 读写，而在别处 | 先 profile，再决定是否改 GQA |
| 长上下文仍爆显存 | 总层数大、序列太长、batch 太大 | GQA 结合分页缓存、量化缓存一起用 |
| 实现出错 | Q 头和 KV 头映射关系写错 | 明确 `group_size = H // G`，统一 reshape 规则 |

一个新手常见误区是“既然 MQA 最省，那就直接把 $G=1$”。这在数学上没错，在工程上常常太激进。原因是 MQA 把所有 query 头都压到同一组 K/V，相当于让不同头共享同一个“历史检索视角”。如果模型原本依赖多个头从不同模式检索上下文，MQA 会更容易损伤这部分能力。

更现实的策略通常是：

1. 先试 GQA，`G` 取 8 或 16。
2. 观察长上下文任务、指令跟随和基准集指标。
3. 只有在显存压力仍不够低时，再考虑继续减小 `G`。

再说一个真实工程坑：把 MHA checkpoint 转 GQA 时，组内头并不一定“天然相似”。如果你直接按顺序平均，等于默认第 1 到第 8 个头是一个自然组，但这未必成立。更稳妥的做法是先做头对齐，让功能接近的 K/V 头进入同一组，再做融合。相关工作会用正交对齐、稀疏 mask 或 $L_0$ 约束来自动找组，这样通常比生硬平均更保精度。

---

## 替代方案与适用边界

如果只看“减少 KV Cache”，GQA 不是唯一方案，但它是最容易落地的一类结构改动。选择时可以按下面这张表判断。

| 方案 | 适合什么 | 不适合什么 |
|---|---|---|
| MHA | 训练阶段、追求最强表达、显存和带宽充足 | 超长上下文单卡部署 |
| GQA | 长上下文推理、在线服务、单卡或少卡部署 | 极短上下文且缓存不是瓶颈 |
| MQA | 显存极紧、吞吐优先、可接受轻微性能下降 | 高保真任务、复杂长文理解 |
| 其他缓存优化 | 需要与现有结构兼容，如 KV 量化、分页缓存 | 想直接减少头数时效果有限 |

可以把选择流程简化成下面几步：

1. 先问自己：瓶颈是不是 KV Cache 显存或带宽？
2. 如果不是，优先看 kernel、并行策略、batch 调度，不要先动注意力结构。
3. 如果是，而且你希望尽量少伤精度，优先选 GQA。
4. 如果显存极端紧张，可以考虑 MQA，但要接受更高性能回退风险。
5. 如果已有 MHA checkpoint，不想从零训练，先做 MHA → GQA 转换，再决定是否继续微调。

给一个面向部署的经验化建议：

| 约束 | 更合理的选择 |
|---|---|
| 训练新模型，资源充足 | MHA 或原生 GQA |
| 需要 100k 上下文，单卡 A100 80G | GQA，常见起点是 `G=8` |
| 显存压到极限，任务允许小幅退化 | MQA |
| 已有 MHA 权重，希望快速上线 | MHA 转 GQA，组均值初始化，再做轻量校准 |

最后再强调边界：GQA 优化的是**推理时的 K/V 头数**，不是把注意力计算复杂度整体变成线性，也不是替代所有长上下文技术。它常常需要和 KV Cache 量化、分页缓存、连续批处理一起使用，才能把真实服务成本压下来。

---

## 参考资料

- Ainslie et al., 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。对应本文的架构定义、MHA/MQA/GQA 关系、从 MHA checkpoint 迁移到 GQA 的基本思路。
- Shazeer, 2019, *Fast Transformer Decoding: One Write-Head is All You Need*。对应 MQA 的原始思路，即用单组 K/V 降低解码期缓存与带宽成本。
- Emergent Mind, *Grouped-Query Attention (GQA)* 主题综述。对应本文中的组均值公式、KV Cache 比例公式与工程背景总结。
- Jin et al., 2025, *Align Attention Heads for Grouped-Query Attention with Sparse Fusion / L0-style pruning*。对应本文中“先对齐头、再做组融合”的工程化迁移思路。
- Meta LLaMA-2 / LLaMA-3 相关公开资料与社区解析。对应本文中的真实工程例子，即 70B 级模型采用 GQA 来降低长上下文推理缓存压力。
