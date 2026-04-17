## 核心结论

Self-Attention 的任务，是让序列中的每个位置在生成自己的新表示时，都能访问同一序列里的全部位置。这里的“位置”可以理解为一个 token，也就是一句话中一个词、一个子词，或者代码中的一个符号。

它的核心公式是：

$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中，$Q$ 是查询，白话解释是“我现在想找什么信息”；$K$ 是键，白话解释是“我能被怎样匹配”；$V$ 是值，白话解释是“如果你关注我，我提供什么内容”。

与早期 Seq2Seq 注意力的关键区别是：Self-Attention 里 $Q,K,V$ 都来自同一条输入序列；而 cross-attention 中，查询和键值来自不同序列。

$\sqrt{d_k}$ 缩放不是细节，而是稳定训练的必要条件。若 $q_i,k_i$ 独立同分布且方差为 1，则点积 $q\cdot k=\sum_{i=1}^{d_k} q_i k_i$ 的方差近似为 $d_k$。维度越大，分数越容易过大，softmax 越容易饱和，梯度越接近 0。除以 $\sqrt{d_k}$ 后，分数尺度被拉回稳定区间。

---

## 问题定义与边界

问题可以表述为：给定长度为 $n$ 的序列 $X\in\mathbb{R}^{n\times d_{model}}$，如何让第 $i$ 个位置在更新表示时，不只看自己，还能有选择地聚合其他位置的信息？

这类需求在语言和代码里都很常见。

玩具例子：句子只有两个词，“猫”“跳”。“跳”这个位置如果只看自己，只知道“跳”是一个动作；如果它同时能看“猫”，就能形成“谁在跳”的更完整表示。Self-Attention 的作用，就是让“跳”对“猫”分配一定权重，再把“猫”的信息融合进来。

真实工程例子：在代码补全里，模型生成 `return total / count` 时，`count` 的含义可能在几十个 token 之前定义。Self-Attention 允许当前位置直接与远处的定义位置建立依赖，而不是像纯 RNN 那样必须沿时间步逐层传递。

它的边界也必须说清楚：

| 机制 | Q 来自哪里 | K/V 来自哪里 | 解决的问题 |
|---|---|---|---|
| Self-Attention | 当前序列 | 当前序列 | 序列内部依赖建模 |
| Cross-Attention | 目标序列 | 源序列 | 跨序列信息对齐 |
| 普通前馈层 | 当前 token | 不涉及 | 单位置非线性变换 |

因此，Self-Attention 只负责“同一序列内部的信息路由”。如果任务本质上需要“一个序列去查询另一个序列”，例如机器翻译中 decoder 查 encoder 输出，就不是纯自注意力的边界，而是 cross-attention 的边界。

另一个边界是复杂度。标准 Self-Attention 需要构造 $n\times n$ 的注意力矩阵，时间和显存复杂度都与 $O(n^2)$ 相关。短序列效果好，超长序列会变贵。

---

## 核心机制与推导

先看输入。设序列输入为：

$$
X\in\mathbb{R}^{n\times d_{model}}
$$

每一行对应一个位置的向量表示。Self-Attention 不直接拿 $X$ 去做匹配，而是先做三次线性投影：

$$
Q=XW^Q,\quad K=XW^K,\quad V=XW^V
$$

其中：

- $W^Q\in\mathbb{R}^{d_{model}\times d_k}$
- $W^K\in\mathbb{R}^{d_{model}\times d_k}$
- $W^V\in\mathbb{R}^{d_{model}\times d_v}$

为什么要拆成三组矩阵？因为“我拿什么特征去询问别人”和“我拿什么特征让别人匹配我”以及“我真正输出什么内容”是三件不同的事。如果直接把同一个向量同时当查询、键、值，表达能力会被绑死。

设第 $i$ 个位置的查询向量是 $q_i$，第 $j$ 个位置的键向量是 $k_j$。两者点积：

$$
s_{ij}=q_i\cdot k_j
$$

可以看成“位置 $i$ 对位置 $j$ 的原始相关性分数”。把所有位置两两计算，就得到分数矩阵：

$$
S = QK^\top \in \mathbb{R}^{n\times n}
$$

这里第 $i$ 行表示“第 $i$ 个查询看向所有键”的结果，第 $j$ 列表示“所有查询看向第 $j$ 个键”的结果。

然后做缩放：

$$
\tilde{S}=\frac{QK^\top}{\sqrt{d_k}}
$$

再对每一行做 softmax，也就是 row-wise softmax：

$$
A_{ij}=\frac{e^{\tilde{S}_{ij}}}{\sum_{t=1}^{n} e^{\tilde{S}_{it}}}
$$

得到注意力权重矩阵 $A\in\mathbb{R}^{n\times n}$。每一行加起来等于 1，表示“当前位置把多少注意力预算分给各个位置”。

最后用这些权重对值向量加权求和：

$$
O=AV
$$

其中 $O\in\mathbb{R}^{n\times d_v}$，第 $i$ 行可展开为：

$$
o_i=\sum_{j=1}^{n} A_{ij}v_j
$$

这句话很关键：第 $i$ 个位置的新表示，不再只是自己的旧向量，而是整条序列所有值向量的加权和。

下面给一个玩具例子。序列有两个位置，“猫”和“跳”。假设某一步计算后，针对“跳”的查询分数是：

$$
[\text{score to 猫},\ \text{score to 跳}] = [1.0, 1.4]
$$

做 softmax 后可能得到：

$$
[0.40,\ 0.60]
$$

这表示“跳”这个位置更新自己时，40% 关注“猫”，60% 关注自己。于是它的新表示就不是单独的“跳”，而是带有“猫在跳”上下文的表示。

从统计角度看缩放项也能推出来。若 $q_i,k_i$ 独立、均值 0、方差 1，则：

$$
q\cdot k=\sum_{t=1}^{d_k} q_t k_t
$$

由于每项方差近似为 1，和的方差近似为 $d_k$。所以 $d_k$ 越大，未缩放点积的波动越大。softmax 输入过大时会接近 one-hot，导致梯度很小，训练不稳定。除以 $\sqrt{d_k}$ 后，方差被压回常数量级。

---

## 代码实现

下面用纯 Python 和 `numpy` 写一个单头 scaled dot-product attention。代码是可运行的，并带有 `assert` 检查形状与权重归一化。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # 数值稳定
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(dk)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

# 玩具输入：2 个 token，每个 token 2 维
X = np.array([
    [1.0, 0.0],   # 猫
    [0.0, 1.0],   # 跳
])

# 为了演示，手工指定投影矩阵
WQ = np.array([
    [1.0, 0.5],
    [0.2, 1.0],
])

WK = np.array([
    [1.0, 0.1],
    [0.3, 1.0],
])

WV = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])

Q = X @ WQ
K = X @ WK
V = X @ WV

output, weights = scaled_dot_product_attention(Q, K, V)

# 每一行是一个查询对所有位置的注意力分布
assert weights.shape == (2, 2)
assert np.allclose(weights.sum(axis=-1), np.array([1.0, 1.0]))

# 输出形状应与 V 的最后一维一致
assert output.shape == (2, 2)

# 权重应为非负
assert np.all(weights >= 0)

print("Q=\n", Q)
print("K=\n", K)
print("V=\n", V)
print("weights=\n", weights)
print("output=\n", output)
```

这段代码对应的步骤只有四步：

| 步骤 | 代码 | 含义 |
|---|---|---|
| 1 | `scores = (Q @ K.T) / sqrt(dk)` | 计算查询和键的相似度 |
| 2 | `scores + mask` | 可选屏蔽，例如因果掩码 |
| 3 | `weights = softmax(scores)` | 把分数变成归一化权重 |
| 4 | `output = weights @ V` | 用权重对值做加权求和 |

真实工程里通常还会加 mask。最常见的是因果 mask，也就是生成第 $t$ 个 token 时不能偷看未来位置。实现方式通常是给未来位置加上一个极小值，如 $-10^9$，让 softmax 后这些位置权重接近 0。

在框架里，单头版本通常被包在多头注意力里。多头的含义是把通道维拆成多个子空间，每个头学习一种关系，比如指代关系、语法关系、局部邻近关系。单头已经能说明机制，多头只是并行重复这套过程再拼接。

---

## 工程权衡与常见坑

Self-Attention 的理论公式很短，但工程里经常出问题，主要集中在数值稳定、形状管理和复杂度三个方面。

| 常见坑 | 直接后果 | 排查方式 |
|---|---|---|
| 忘记除以 $\sqrt{d_k}$ | softmax 饱和，梯度过小 | 看 logits 范围是否随维度增大明显变大 |
| Q/K/V 共用同一投影 | 表达能力下降 | 比较独立投影与共享投影的验证集损失 |
| `softmax` 维度写错 | 权重语义错误 | 确认是沿最后一维，即“对每个查询看所有键” |
| mask 加在 softmax 后 | 被屏蔽位置仍可能泄漏 | 必须在 softmax 前加极小值 |
| 形状不匹配 | 运行时报错或隐式广播错误 | 明确检查 `(n, d_k) @ (d_k, n)` |
| 长序列直接全量注意力 | 显存爆炸 | 评估 $n^2$ 级别成本 |

先说最重要的缩放问题。很多初学者会觉得公式里多一个 $\sqrt{d_k}$ 很像“经验参数”，其实不是。它有明确统计意义。没有这个缩放，随着头维度增加，分数分布会越来越尖，训练早期尤其容易出现某一列几乎独占全部概率的情况。

再说独立投影。若直接令 $Q=K=V=X$，模型会退化成“用同一种表示既做匹配又做内容输出”。这不是不能工作，而是能力受限。查询需要强调“我要找什么”，键需要强调“我适合被谁找到”，值需要强调“我能提供什么信息”，三者目标不同。

真实工程例子：在代码大模型里，函数调用参数往往依赖前文定义。比如前面出现 `config.max_len`，后面出现 `truncate(tokens, max_len)`。如果模型能在某些头里专门学习“变量引用关系”，那么当前位置就会把较高权重分给相关定义位置。若投影过于简单或头数太少，这种关系会更难学出来。

另一个常见误区是把注意力矩阵解释成“因果证明”。注意力权重高，只能说明当前层当前头在这一时刻更依赖某些位置，不等于模型决策完全由这些位置决定。因为后面还有残差连接、前馈层、多层叠加和多头混合。

最后是复杂度。标准 Self-Attention 对长度为 $n$ 的序列要算 $n^2$ 个相关性分数。序列从 1k 扩到 32k，不只是慢一点，而是矩阵规模平方增长。工程上必须结合窗口注意力、分块、KV cache、FlashAttention 一类实现优化。

---

## 替代方案与适用边界

Self-Attention 不是唯一方案，也不是所有任务都必须用全量自注意力。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| Self-Attention | 语言建模、编码器表征 | 全局建模、并行性强 | $O(n^2)$ 成本高 |
| Cross-Attention | 翻译、检索增强生成 | 能做跨序列对齐 | 需要两路输入 |
| 卷积 | 局部模式强的序列 | 高效、局部归纳偏置强 | 长距离依赖弱 |
| RNN/LSTM | 流式处理、低资源场景 | 顺序状态自然 | 并行性差 |
| 稀疏/窗口注意力 | 超长文本 | 降低成本 | 丢失部分全局交互 |

Self-Attention 适合“同一序列内部每个位置都可能彼此相关”的场景，例如语言建模、文本分类、代码建模。Cross-Attention 适合“我当前在一个序列中生成内容，但需要查询另一个序列”的场景，例如机器翻译中 decoder 查 encoder，或者 RAG 中生成端查检索结果。

卷积和 RNN 不是过时，而是边界不同。卷积自带局部平移不变性，适合局部模式稳定的任务；RNN 天然适合流式输入，因为它不需要一次拿到整段序列。若系统要求低延迟、低显存、边输入边输出，它们仍然有价值。

因此，选择机制时不要只问“哪个更强”，而要问三件事：

1. 依赖是序列内部还是跨序列。
2. 序列长度是否允许 $O(n^2)$。
3. 任务更需要全局交互，还是更需要局部归纳偏置与流式能力。

---

## 参考资料

1. Vaswani et al. *Attention Is All You Need*. 提出 scaled dot-product attention 与 Transformer 主体结构。  
2. Wikipedia: *Attention (machine learning)*。适合快速确认标准公式与术语。  
3. Wikipedia: *Transformer (deep learning)*。适合建立 self-attention、cross-attention、多头结构的整体图景。  
4. Purdue 相关讲义 *Attention from Scratch*。对 Q、K、V 投影动机的解释比较直接。  
5. 一些 toy example 教程。适合先手算 $QK^\top$、softmax、再乘 $V$，把矩阵流程走通。
