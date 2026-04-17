## 核心结论

DeltaNet 是一种基于 **delta rule** 的序列建模结构：它不再把每个 token 的信息无差别累加到记忆里，而是先用当前记忆做预测，再根据预测误差修正记忆。

它的核心结论可以压缩成一句话：

> DeltaNet 把线性注意力中的“外积累加写入”改成“误差驱动写入”，从而在保持线性时间推理的同时，提高长上下文检索、覆盖修正和 in-context learning 能力。

这里的 **delta rule** 可以先按白话理解：模型先预测一个值，再看预测和真实目标差多少，然后只按这个差值更新状态。

普通线性注意力像“每来一条信息就往本子里抄一遍”。DeltaNet 更像“先检查本子里对这个 key 已经写了什么，再把错的地方改掉”。所以它更适合反复查同一类信息、逐步修正记忆、处理持续流入上下文的任务。

| 机制 | 记忆写入方式 | 状态大小 | 推理复杂度 | 主要问题 |
|---|---|---:|---:|---|
| 标准 Transformer 注意力 | 保存历史 KV，再按 query 全量匹配 | 随长度增长 | $O(n^2)$ 或带 KV cache 的逐步增长 | 长上下文显存压力大 |
| 普通线性注意力 | 每步做 $v_t k_t^T$ 外积累加 | 固定 | $O(n)$ | 容易无差别叠加，覆盖能力弱 |
| DeltaNet | 根据 $S_{t-1}k_t - v_t$ 的误差修正 | 固定 | $O(n)$ | 实现更复杂，对稳定性敏感 |

总览流程如下：

```text
输入 token
  ↓
生成 k_t, v_t, q_t
  ↓
用旧状态预测: S_{t-1} k_t
  ↓
计算误差: S_{t-1} k_t - v_t
  ↓
按 β_t 修正状态: S_t
  ↓
用 q_t 读出输出: o_t = S_t q_t
```

DeltaNet 值得单独讲，是因为它不是线性注意力的简单改写。它改变了“记忆如何被更新”这个核心机制。

---

## 问题定义与边界

DeltaNet 所在的问题空间是 **序列建模**。序列建模就是模型按顺序处理一串 token，并在每一步产生表示或预测结果。语言模型、代码补全、长文档问答都属于这个范围。

标准 Transformer 的注意力需要比较 query 和历史 key。长度为 $n$ 时，全量注意力通常涉及 $n \times n$ 的交互。推理阶段可以用 KV cache 缓解重复计算，但缓存仍然随上下文长度增长。

线性注意力试图解决这个问题：把注意力写成可递推状态，让模型只维护一个固定大小的状态矩阵。DeltaNet 继承了这个方向，但补上一个短板：普通线性注意力主要是累加，DeltaNet 加入预测误差修正。

| 项目 | 定义 |
|---|---|
| 任务 | 按顺序处理 token，并根据历史状态产生输出 |
| 输入长度 | 可以很长，目标是让推理开销随长度线性增长 |
| 状态形式 | 固定大小矩阵 $S_t \in \mathbb{R}^{d_v \times d_k}$ |
| 每步输入 | key $k_t$、value $v_t$、query $q_t$ |
| 每步输出 | $o_t = S_t q_t$ |
| 推理复杂度 | 对序列长度为 $O(n)$，不需要保存全部历史 KV |

基础记号如下：

| 记号 | 形状 | 白话解释 |
|---|---:|---|
| $k_t$ | $\mathbb{R}^{d_k}$ | 当前 token 的 key，用来定位要写入或读取的记忆方向 |
| $v_t$ | $\mathbb{R}^{d_v}$ | 当前 token 的 value，要写入记忆的内容 |
| $q_t$ | $\mathbb{R}^{d_k}$ | 当前 token 的 query，用来从状态里读出信息 |
| $S_t$ | $\mathbb{R}^{d_v \times d_k}$ | 到第 $t$ 步为止的压缩记忆状态 |
| $o_t$ | $\mathbb{R}^{d_v}$ | 当前步从状态中读出的输出 |

新手版理解：如果你要连续读一长段文档并随时回答问题，标准 Transformer 往往要保存越来越长的 KV cache；DeltaNet 只维护固定大小的状态矩阵，更像一个可反复修正的工作记忆。

边界也必须说清楚。DeltaNet 不是标准 Transformer 的完全替代品，也不是参数优化器里的 delta rule。这里的 delta rule 是状态更新规则，不是在训练时直接更新模型参数的优化算法。

| 适合 | 不适合 |
|---|---|
| 长上下文流式推理 | 很短、对速度不敏感的小任务 |
| 需要固定状态开销的部署场景 | 强依赖任意 token-to-token 精细对齐的任务 |
| 反复检索、持续更新记忆 | 需要完整保留全部历史细节的场景 |
| 低显存长序列服务 | 已经被标准 Transformer 高效解决的短序列建模 |

---

## 核心机制与推导

先看普通线性注意力的基线形式。它维护一个状态矩阵：

$$
S_t = S_{t-1} + v_t k_t^T
$$

这里的 $v_t k_t^T$ 是 **外积**。外积可以理解为：用 value 的每个维度和 key 的每个维度两两相乘，得到一个矩阵，然后加到状态里。

输出通常写成：

$$
o_t = S_t q_t
$$

这条路径的问题是：所有新信息都被直接加进去。它没有先问“当前状态对这个 key 已经知道什么”，也没有根据预测误差做修正。

DeltaNet 的更新式是：

$$
S_t = S_{t-1} - \beta_t (S_{t-1}k_t - v_t)k_t^T
$$

其中 $\beta_t \in [0,1]$ 是更新强度。$\beta_t$ 可以理解为“这一步改记忆时下手多重”。

这个式子里的每一项都有明确含义：

| 项 | 含义 |
|---|---|
| $S_{t-1}k_t$ | 用旧状态对当前 key 读出的预测值 |
| $v_t$ | 当前真实要写入的 value |
| $S_{t-1}k_t - v_t$ | 预测误差 |
| $\beta_t$ | 修正幅度 |
| $(S_{t-1}k_t - v_t)k_t^T$ | 沿当前 key 方向修正状态 |

等价展开形式是：

$$
S_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T
$$

这个等价形式能看出两件事。第一，旧状态不是完整保留，而是会在 $k_t$ 对应方向上被调整。第二，新 value 不是简单加进去，而是带着 $\beta_t$ 写入。

推导流程如下：

```text
旧状态 S_{t-1}
  ↓
当前 key: k_t
  ↓
预测当前 value: \hat{v}_t = S_{t-1} k_t
  ↓
计算误差: e_t = \hat{v}_t - v_t
  ↓
沿 key 方向修正: S_t = S_{t-1} - β_t e_t k_t^T
```

玩具例子用标量最容易看清楚。设 $S_0=0$，$k_1=1$，$v_1=2$，$\beta_1=0.5$，$q_1=1$：

$$
S_1 = 0 - 0.5(0 \times 1 - 2)\times 1 = 1
$$

所以：

$$
o_1 = S_1q_1 = 1
$$

第二步仍然是 $k_2=1$，但目标值变成 $v_2=4$：

$$
S_2 = 1 - 0.5(1 \times 1 - 4)\times 1 = 2.5
$$

它没有把 4 直接加到原来的 1 上变成 5，而是发现当前预测为 1，目标是 4，只按误差方向修正。这个行为就是 DeltaNet 和普通外积累加的关键差异。

真实工程例子是长文档问答或代码补全的流式推理。用户不断输入新上下文，模型需要持续更新状态并回答问题。标准 Transformer 的 KV cache 会随上下文增长；DeltaNet 维护固定大小状态 $S_t$，更适合低显存、长会话、检索密集的在线服务。

训练时还有一个关键点：推理可以逐 token 串行更新，但训练如果也完全串行，会浪费 GPU 并行能力。论文将递推重参数化为 Householder/WY 形式，用 **chunkwise 并行** 训练。chunkwise 并行就是把序列切成块，块内用矩阵形式并行处理，块之间再递推传递状态。

Householder/WY 可以先按直观理解：它是一类把连续线性变换组织成矩阵乘法的重写方式，目标不是改变 DeltaNet 的含义，而是让 GPU 更容易并行计算。

---

## 代码实现

实现时要区分两条路径：推理路径和训练路径。

推理路径可以一步一步递推，因为在线服务本来就是 token 逐步到达。训练路径通常不应该写成纯 Python 循环，因为那会让 GPU 长时间等串行依赖，吞吐很差。

最小伪代码如下：

```text
初始化 S = 0

for each token t:
    k_t, v_t, q_t, beta_t = projection(x_t)

    pred = S @ k_t
    err = pred - v_t

    S = S - beta_t * outer(err, k_t)
    o_t = S @ q_t

    return o_t, S
```

状态张量形状如下：

| 张量 | 单头形状 | batch 后常见形状 | 说明 |
|---|---:|---:|---|
| $k_t$ | $(d_k,)$ | $(B, H, d_k)$ | key |
| $v_t$ | $(d_v,)$ | $(B, H, d_v)$ | value |
| $q_t$ | $(d_k,)$ | $(B, H, d_k)$ | query |
| $S_t$ | $(d_v, d_k)$ | $(B, H, d_v, d_k)$ | 每层每头状态 |
| $\beta_t$ | 标量 | $(B, H, 1)$ 或 $(B,H)$ | 更新强度 |

下面是一个可运行的 Python 玩具实现，展示标量和向量场景。它不是高性能训练实现，但能准确表达 DeltaNet 的状态更新含义。

```python
import numpy as np

def deltanet_step(S, k, v, q, beta):
    """
    S: (d_v, d_k)
    k: (d_k,)
    v: (d_v,)
    q: (d_k,)
    beta: scalar in [0, 1]
    """
    pred = S @ k
    err = pred - v
    S_new = S - beta * np.outer(err, k)
    out = S_new @ q
    return S_new, out

# 玩具例子：标量形式
S = np.array([[0.0]])
k = np.array([1.0])
v = np.array([2.0])
q = np.array([1.0])

S, o = deltanet_step(S, k, v, q, beta=0.5)
assert np.allclose(S, [[1.0]])
assert np.allclose(o, [1.0])

v2 = np.array([4.0])
S, o = deltanet_step(S, k, v2, q, beta=0.5)
assert np.allclose(S, [[2.5]])
assert np.allclose(o, [2.5])

# 向量例子：d_v=2, d_k=3
S = np.zeros((2, 3))
k = np.array([1.0, 0.0, 0.0])
v = np.array([3.0, -1.0])
q = np.array([1.0, 0.0, 0.0])

S, o = deltanet_step(S, k, v, q, beta=1.0)
assert np.allclose(S[:, 0], v)
assert np.allclose(o, v)
```

训练与推理的流程差异如下：

| 路径 | 典型做法 | 优点 | 风险 |
|---|---|---|---|
| 推理 | token-by-token 更新 $S_t$ | 状态固定，适合流式服务 | 需要正确维护每层每头状态 |
| 训练 | chunkwise 并行计算 | 利用 GPU 并行，提高吞吐 | 实现复杂，需处理块间状态传递 |
| 朴素训练 | Python 循环逐 token 更新 | 容易理解 | 慢，难以用于真实模型训练 |

工程接口通常会拆成两层：一个低层 kernel 负责高效扫描或分块计算，一个上层模块负责生成 $q,k,v,\beta$，再调用 DeltaNet 核心算子。

```python
class DeltaNetLayer:
    def forward_train(self, x, chunk_size=64):
        # 训练路径：通常调用高效 chunkwise kernel
        q, k, v, beta = self.project(x)
        return chunkwise_delta_rule(q, k, v, beta, chunk_size)

    def forward_infer(self, x_t, state):
        # 推理路径：单步更新状态
        q_t, k_t, v_t, beta_t = self.project_one(x_t)
        new_state, out_t = delta_step(state, q_t, k_t, v_t, beta_t)
        return out_t, new_state
```

这段接口只是示意。真实工程里会考虑多头、batch、层归一化、残差连接、门控、混合精度和 CUDA/Triton kernel。

---

## 工程权衡与常见坑

DeltaNet 的优势来自固定状态和误差修正，但代价是实现复杂度更高。它不像标准注意力那样直接调用成熟 attention kernel 就能得到稳定性能。

最重要的稳定性变量是 $\beta_t$。如果 $\beta_t$ 过大，就像每次改笔记都太用力，原来正确的内容也会被覆盖；如果 $\beta_t$ 太小，又像修改太保守，模型学得慢、记不住新信息。

| 风险 | 表现 | 对策 |
|---|---|---|
| $\beta_t$ 过大 | 状态震荡，旧记忆被破坏 | 用 sigmoid 限制范围，配合初始化控制 |
| key 未归一化 | 更新幅度随 key 范数失控 | 对 key 做归一化或使用稳定参数化 |
| 朴素串行训练 | GPU 利用率低，训练极慢 | 使用 chunkwise 或 flash-linear-attention 实现 |
| 状态形状写错 | 多头状态混在一起，输出异常 | 明确 $(B,H,d_v,d_k)$ 布局 |
| 混合精度溢出 | loss 不稳定，状态出现 NaN | 对状态更新使用更稳的精度策略 |
| 误认为能替代所有注意力 | 任务效果不稳定 | 按任务比较，而不是按概念替换 |

稳定性检查清单：

| 检查项 | 判断标准 |
|---|---|
| $\beta_t$ 范围 | 是否被限制在合理区间，例如 $[0,1]$ |
| key 范数 | 是否存在极端大值 |
| 状态范数 | 是否随序列长度持续爆炸 |
| 输出分布 | 是否在长序列后明显漂移 |
| chunk 结果 | chunkwise 训练是否和串行参考实现近似一致 |
| 推理状态 | batch 内不同样本的状态是否被正确隔离 |

性能与显存权衡如下：

| 方案 | 显存 | 速度 | 实现难度 | 说明 |
|---|---:|---:|---:|---|
| 标准注意力 | 高 | 短序列快，长序列贵 | 低 | 生态成熟 |
| KV cache 推理 | 随长度增长 | 单步较快 | 中 | 长会话显存压力仍在 |
| 朴素 DeltaNet | 低 | 训练慢 | 低 | 适合理解，不适合生产训练 |
| 高效 DeltaNet | 低 | 可竞争 | 高 | 需要 chunkwise/kernel 支持 |

最小调参建议：

| 参数 | 优先级 | 建议 |
|---|---:|---|
| $\beta_t$ | 高 | 先保证范围受控，再调表达能力 |
| key 归一化 | 高 | 优先检查 key 范数和状态范数 |
| chunk 大小 | 中 | 在吞吐和数值一致性之间折中 |
| 精度选择 | 中 | 长序列状态更新可考虑更稳的累积精度 |
| 初始状态 | 中 | 通常从零状态开始，注意 batch 重置 |

常见误解有四个。

第一，把 DeltaNet 和优化器里的 delta rule 混为一谈。这里更新的是序列处理过程中的状态矩阵 $S_t$，不是直接更新模型参数。

第二，以为它只是“线性注意力换个名字”。普通线性注意力是外积累加，DeltaNet 是误差修正写入，两者的记忆行为不同。

第三，训练时直接写逐 token 循环。这样可以验证公式，但不能代表论文中的高效训练方式。

第四，误判它能完全取代 Transformer。DeltaNet 是长序列建模的一条路线，不是所有任务上的默认最优解。

---

## 替代方案与适用边界

DeltaNet 应该和标准 Transformer、线性注意力、状态空间模型放在同一个问题框架下比较：它们都在回答“如何在长序列里维护和读取历史信息”。

| 方法 | 历史信息形式 | 长序列成本 | 表达特点 | 适用边界 |
|---|---|---:|---|---|
| Transformer | 显式 KV 或全量注意力矩阵 | 高 | token-to-token 交互强 | 短到中等长度、需要精细对齐 |
| 线性注意力 | 固定状态矩阵 | 低 | 高效，但覆盖修正较弱 | 长序列、高吞吐场景 |
| DeltaNet | 误差修正的固定状态矩阵 | 低 | 更强在线更新和覆盖能力 | 长上下文、流式推理、反复检索 |
| SSM / Mamba 类方法 | 状态空间递推 | 低 | 强序列扫描归纳偏置 | 流式、长序列、连续动态建模 |

**状态空间模型**，简称 SSM，可以白话理解为：模型维护一个随时间递推的内部状态，用状态变化来吸收历史信息。Mamba 是这一路线中代表性很强的方法。

任务适配表如下：

| 任务 | DeltaNet 适配度 | 原因 |
|---|---:|---|
| 长文档问答 | 高 | 固定状态开销适合长输入，误差修正有利于检索 |
| 流式推理 | 高 | token 到达后可在线更新状态 |
| 代码补全 | 中到高 | 长上下文和重复模式明显，但仍需实际评测 |
| 短文本分类 | 中低 | 标准 Transformer 更直接，工程成本更低 |
| 精细对齐任务 | 中低 | 显式注意力可能更稳定 |
| 小模型教学实现 | 中 | 公式清晰，但高效训练不简单 |

新手版理解：如果你要做短文本分类，标准 Transformer 可能更直接；如果你要做长会话、流式问答、代码补全这类持续写入记忆的任务，DeltaNet 的固定状态更有优势。

一句话选型规则：

> 当任务主要瓶颈是长上下文状态维护和在线更新时，DeltaNet 值得评估；当任务依赖短序列内强全局交互时，标准 Transformer 仍然是更稳的起点。

它和线性注意力的差异在于写入规则。线性注意力问的是“如何把注意力改写成可累积状态”；DeltaNet 进一步问“这个状态应该怎样被修正，而不是只被累加”。

它和 Mamba 类方法的关系是同属线性时间序列建模路线，但归纳偏置不同。Mamba 更偏状态空间递推，DeltaNet 更偏 key-value 记忆和误差驱动更新。

什么时候不该用 DeltaNet：数据短、部署资源充足、已有 Transformer 基线足够强、团队缺少高效 kernel 维护能力，或者任务明显依赖完整历史 token 的细粒度匹配。这些情况下，DeltaNet 的理论优势未必能抵消工程成本。

---

## 参考资料

| 阅读顺序 | 资料 | 用途 |
|---:|---|---|
| 1 | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://huggingface.co/papers/2406.06484) | 看论文摘要、核心贡献和实验方向 |
| 2 | [DeltaNet Explained (Part I)](https://sustcsonglin.github.io/blog/2024/deltanet-1/) | 看作者解释，理解 delta rule 如何进入线性 Transformer |
| 3 | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | 看主流工程实现和高效算子接口 |
| 4 | [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) | 看后续扩展和门控 DeltaNet 实现 |

理论理解优先看论文和作者博客。工程落地优先看 `fla-org/flash-linear-attention`，因为高效 DeltaNet 不能只靠朴素 Python 循环实现。想了解后续变体，再看 `NVlabs/GatedDeltaNet`。
