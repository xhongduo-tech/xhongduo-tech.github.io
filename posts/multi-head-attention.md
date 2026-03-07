## 核心结论

多头注意力（Multi-Head Attention）是 Transformer 中的核心模块。它不是“把注意力算很多遍”这么简单，而是把输入表示的总维度 $d_{\text{model}}$ 拆成 $h$ 个更小的子空间，在每个子空间里各自学习一套查询、键、值投影，再并行计算注意力，最后把结果拼接回来。

公式写成：

$$
\text{head}_i=\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

它的价值在于：同一层里，不同 head 可以从不同视角看同一串 token。直白地说，就是“同一篇文章由多个人同时做批注”，有人盯语法关系，有人盯关键词，有人盯指代链，最后把这些批注合并成一个更完整的上下文表示。

一个常见误解是“多头一定比单头参数更多”。在标准设置下，若 $d_k=d_v=d_{\text{model}}/h$，总参数量仍然是：

$$
3d_{\text{model}}^2 + d_{\text{model}}^2 = 4d_{\text{model}}^2
$$

也就是 Q、K、V、O 四个投影矩阵的总和，并不会因为 head 变多而线性暴涨。

| 对比项 | 单头注意力 | 多头注意力 |
|---|---|---|
| 观察视角 | 单一投影空间 | 多个独立投影空间 |
| 并行关系建模 | 弱 | 强 |
| 对句法/语义/共指的分工 | 容易混在一起 | 更容易分头学习 |
| 输出信息丰富度 | 较低 | 较高 |

玩具例子可以这样理解：黑板上同时画“语法树”“关键词云”“代词指代链”，每个人只负责一项，最后再合并讲解。这就是多头把同一序列映射到多个视角后再拼接的效果。

---

## 问题定义与边界

问题本身很明确：Transformer 如何在一次自注意力里，同时捕捉多种 token 之间的依赖关系？

这里的 token 可以理解为“文本切分后的基本单位”，例如一个词、子词或符号。注意力的目标，是让当前位置在读入整句时，不只看局部邻居，而是按相关性从全序列里取信息。

多头注意力解决的是“同一轮里多视角建模”的问题，而不是解决所有长序列效率问题。它的边界主要由下面几个量决定：

| 符号 | 含义 | 常见关系 |
|---|---|---|
| $d_{\text{model}}$ | 模型主维度，即每个 token 的表示长度 | 例如 512、768 |
| $h$ | head 数量，即并行注意力分支数 | 例如 8、12 |
| $d_k$ | 每个 head 中 Q/K 的维度 | 常设为 $d_{\text{model}}/h$ |
| $d_v$ | 每个 head 中 V 的维度 | 常设为 $d_{\text{model}}/h$ |

标准设置下：

$$
d_k=d_v=\frac{d_{\text{model}}}{h}
$$

每个 head 都看同一输入序列，但有自己独立的参数：

$$
W_i^Q,W_i^K,W_i^V \in \mathbb{R}^{d_{\text{model}}\times d_k}
$$

输出投影矩阵通常是：

$$
W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}
$$

新手可把它理解成：所有学生都读同一篇文章，但有人专注分词边界，有人专注情感极性，有人专注代词“他/她/它”到底指谁。最后把每个人的笔记拼到一起，得到更完整的答案。

这里也要说明边界条件：

1. 多头不意味着每个头都一定学到“有意义且不同”的功能，训练后可能出现冗余头。
2. 多头不改变注意力对序列长度的二次复杂度本质，$QK^\top$ 仍然是 $O(n^2)$ 级别。
3. 多头适合“同一序列存在多种依赖”的任务；如果依赖结构很单一，收益可能有限。

---

## 核心机制与推导

先看单个 head。注意力（Attention）可以理解为“按相关性加权汇总信息”。Q 是 Query，白话解释是“当前 token 想找什么”；K 是 Key，白话解释是“每个 token 暴露给别人的索引”；V 是 Value，白话解释是“真正被取走的内容”。

单头缩放点积注意力公式是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中 $\sqrt{d_k}$ 的作用是缩放，防止内积值过大后 softmax 过于尖锐。

多头的做法不是直接在原始空间算一次大注意力，而是先投影：

$$
Q_i=QW_i^Q,\quad K_i=KW_i^K,\quad V_i=VW_i^V
$$

再分别计算：

$$
\text{head}_i=\text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right)V_i
$$

最后拼接：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

下面用一个最小玩具例子说明维度变化。设：

- 序列长度为 $n$
- $d_{\text{model}}=8$
- $h=2$
- 所以 $d_k=d_v=4$

则输入：

$$
X \in \mathbb{R}^{n\times 8}
$$

每个 head 的投影矩阵为：

$$
W_i^Q,W_i^K,W_i^V \in \mathbb{R}^{8\times 4}
$$

于是：

- 输入 $X$ 经过投影，得到每个 head 的 $Q_i,K_i,V_i \in \mathbb{R}^{n\times 4}$
- 每个 head 内部计算注意力，输出 $\text{head}_i \in \mathbb{R}^{n\times 4}$
- 两个 head 拼接后得到 $\mathbb{R}^{n\times 8}$
- 再乘输出矩阵 $W^O\in\mathbb{R}^{8\times 8}$，回到模型主维度

| 张量/矩阵 | 形状 |
|---|---|
| $X$ | $n\times 8$ |
| $W_i^Q,W_i^K,W_i^V$ | $8\times 4$ |
| $Q_i,K_i,V_i$ | $n\times 4$ |
| $\text{head}_i$ | $n\times 4$ |
| $\text{Concat}(\text{head}_1,\text{head}_2)$ | $n\times 8$ |
| $W^O$ | $8\times 8$ |

参数量推导也很直接。每个 head 有 3 个投影矩阵，总参数量是：

$$
h \cdot 3(d_{\text{model}}\times d_k)
$$

当 $d_k=d_{\text{model}}/h$ 时：

$$
h \cdot 3d_{\text{model}}\frac{d_{\text{model}}}{h}=3d_{\text{model}}^2
$$

输出投影矩阵参数量为：

$$
(hd_v)\times d_{\text{model}}=d_{\text{model}}^2
$$

总量就是：

$$
3d_{\text{model}}^2+d_{\text{model}}^2=4d_{\text{model}}^2
$$

真实工程例子是 BERT。BERT Base 的每层 encoder 使用 12 个 head。训练后，不同头往往会偏向不同语言现象，例如某些头更关注相邻词或短程搭配，某些头更容易盯住代词与先行词，某些头更偏句子分隔符或结构边界。这不是“手工规定”的，而是模型通过训练学出来的分工。

---

## 代码实现

实现时通常不会真的写一个 Python `for` 循环逐头慢慢算，而是把 head 维一起并进 batch 维做张量运算。但先看逻辑版伪代码更容易理解：

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention(q, k, v):
    dk = q.shape[-1]
    scores = q @ k.T / np.sqrt(dk)
    weights = softmax(scores, axis=-1)
    return weights @ v

def multi_head_attention(X, W_q, W_k, W_v, W_o):
    heads = []
    h = len(W_q)
    for i in range(h):
        q = X @ W_q[i]
        k = X @ W_k[i]
        v = X @ W_v[i]
        heads.append(attention(q, k, v))
    multi_head = np.concatenate(heads, axis=-1) @ W_o
    return multi_head

# toy example: d_model=8, h=2, d_k=d_v=4
np.random.seed(0)
n = 3
d_model = 8
h = 2
d_k = d_model // h

X = np.random.randn(n, d_model)
W_q = [np.random.randn(d_model, d_k) for _ in range(h)]
W_k = [np.random.randn(d_model, d_k) for _ in range(h)]
W_v = [np.random.randn(d_model, d_k) for _ in range(h)]
W_o = np.random.randn(d_model, d_model)

Y = multi_head_attention(X, W_q, W_k, W_v, W_o)

assert Y.shape == (n, d_model)
assert d_k == 4
assert sum(w.shape[0] * w.shape[1] for w in W_q + W_k + W_v) + W_o.size == 4 * d_model * d_model
```

上面每一行对应的机制是：

- `X @ W_q[i]`：把原始表示投影到第 `i` 个 head 的查询子空间。
- `X @ W_k[i]`：投影到键子空间。
- `X @ W_v[i]`：投影到值子空间。
- `attention(q, k, v)`：在该子空间里独立计算注意力。
- `concatenate`：把所有 head 的输出按最后一维拼接。
- `@ W_o`：重新混合各头信息，回到统一的 `d_model` 空间。

| 变量 | 形状 | 说明 |
|---|---|---|
| `X` | `(n, d_model)` | 输入序列表示 |
| `W_q[i]` | `(d_model, d_k)` | 第 i 个 head 的 Q 投影 |
| `W_k[i]` | `(d_model, d_k)` | 第 i 个 head 的 K 投影 |
| `W_v[i]` | `(d_model, d_k)` | 第 i 个 head 的 V 投影 |
| `attention(q, k, v)` 返回值 | `(n, d_k)` | 单 head 输出 |
| `W_o` | `(h*d_k, d_model)` | 拼接后输出投影 |

工程实现中通常会把参数整理成形如 `(d_model, h, d_k)` 或一次性线性层输出 `(n, h, d_k)` 的形式，这样能用矩阵并行而不是 Python 循环，速度更高。

---

## 工程权衡与常见坑

多头注意力提升了表达能力，但没有免费午餐。

第一类问题是复杂度。注意力分数矩阵来自 $QK^\top$，若序列长度为 $n$，每个 head 都要形成一个 $n\times n$ 的相关性矩阵，所以整体瓶颈仍然是二次复杂度。真实工程里，处理 8K token 长文档时，即使每个 head 的维度不大，注意力矩阵本身也会占掉大量显存。

| 方案 | 时间复杂度 | 空间复杂度 | 适用场景 |
|---|---|---|---|
| 密集 attention | $O(n^2 d)$ | $O(n^2)$ | 中短序列，追求完整依赖 |
| 局部/稀疏 attention | 低于密集 | 低于密集 | 长序列，允许只看部分上下文 |

第二类问题是 head 冗余。head 数太多时，多个头可能学出相似的注意图。表面看是“12 个头很强”，实际上有效视角可能没那么多。这时可以做头剪枝，白话解释就是“把贡献很小的头删掉”，降低推理成本。

第三类问题是维度切分。若 $h$ 太大，则单个 head 的 $d_k$ 太小，表达能力会变弱。例如固定 $d_{\text{model}}=256$，若硬拆成 32 个头，则每头只有 8 维，可能不足以承载复杂关系。

长序列场景下，一个常见改法是只让每个 token 关注局部窗口。伪代码如下：

```python
def build_local_mask(n, window):
    mask = np.full((n, n), -1e9)
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        mask[i, left:right] = 0.0
    return mask
```

这个 mask 的意思是：窗口外的位置分数全部加上极小值，softmax 后几乎变成 0。这样能控制显存，但代价是丢掉远程依赖。

---

## 替代方案与适用边界

如果问题不是“需要更多视角”，而是“序列太长、算不动”，那么标准多头注意力未必是最优解。

一种替代是头共享。白话解释是“多个 head 共享部分投影参数”，减少参数与访存开销。适合资源受限部署，但表达多样性会下降。

一种替代是头剪枝。它不改变训练时结构，而是在训练后删掉不重要的头，适合已经有成熟模型、希望降低推理成本的场景。

更典型的替代是稀疏或分块注意力。它的核心不是减少 head，而是减少每个 token 真正可见的位置集合。可写成：

$$
\text{Attention}(Q,K,V;M)=\text{softmax}\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)V
$$

其中 $M$ 是 mask。若位置 $(i,j)$ 不允许连接，则 $M_{ij}$ 取一个极小值。

超长文本摘要就是典型真实工程例子。标准全局多头注意力要比较所有 token 两两关系，代价太高。常见替代是“局部 window + 全局 summary token”：普通 token 只看附近窗口，少量全局 token 负责跨段聚合。这样保留部分全局信息，同时把 $n^2$ 成本压下去。

| 方案 | 核心思路 | 优点 | 适用边界 |
|---|---|---|---|
| 头共享 | 多个 head 共享部分参数 | 更省参数 | 设备受限、可接受精度下降 |
| 头剪枝 | 删除冗余头 | 降低推理成本 | 已训练模型优化 |
| 稀疏 attention | 只连部分位置 | 适合长序列 | 需要设计连接模式 |
| Linformer | 低秩投影近似注意力 | 降低复杂度 | 可接受近似误差 |
| Reformer | 用局部敏感哈希近似匹配 | 更长序列可行 | 实现复杂，训练细节多 |

因此，标准多头注意力最适合的边界是：序列长度中等、任务确实需要多种依赖关系、并且硬件足以承担密集注意力成本。超过这个边界，重点通常不再是“加多少头”，而是“如何减少连接密度”。

---

## 参考资料

1. Vaswani et al. *Attention Is All You Need*：Multi-Head Attention 的原始定义与 Transformer 整体架构。  
2. Harvard NLP, *The Annotated Transformer*：对多头公式、维度与参数预算有直观推导。  
3. SuperML 多头注意力教程：用可视化方式解释不同 head 如何关注不同关系。  
4. Stanford, *What Does BERT Look At?*：分析 BERT 各注意力头在语法、指代等现象上的行为。
