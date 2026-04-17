## 核心结论

Key 投影矩阵 $W_K$ 的作用，是把输入表示映射到“可被匹配的特征空间”。白话说，Key 不是原始 token 本身，而是 token 经过线性变换后，对外声明“我能提供哪些信息”的向量。对应地，Query 投影矩阵 $W_Q$ 把输入映射到“检索需求空间”，表示“我在找什么”。

Transformer 中最核心的打分过程是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中 $Q=XW_Q,\ K=XW_K,\ V=XW_V$。$QK^\top$ 的每个元素，都是一个 Query 与一个 Key 的匹配分数。

Key 投影矩阵的设计考量可以压缩成三点：

1. $W_K$ 和 $W_Q$ 是否独立，决定注意力是否天然具有方向性。
2. $W_K$ 的输出维度和初始化方差，直接影响分数分布是否稳定。
3. 在参数受限场景下，共享投影可以省参数，但会牺牲一部分“谁看谁”的表达能力。

如果令 $W_Q=W_K=W$，则打分核退化为对称形式：

$$
A_{\text{sym}}(x_i,x_j)=(Wx_i)(Wx_j)^\top
$$

这意味着从打分公式本身看，样本对 $(i,j)$ 与 $(j,i)$ 的兼容度相同，模型更像是在建模“相似性”，而不是“检索关系”。如果任务需要保留方向性，可以在共享投影后加入一个可学习矩阵 $S$：

$$
A_{\text{pair}}(x_i,x_j)=(Wx_i)S(Wx_j)^\top
$$

这样参数仍比独立的 $W_Q,W_K$ 少，但已经恢复了非对称建模能力。

---

## 问题定义与边界

本文讨论的是自注意力或交叉注意力中，Key 投影矩阵 $W_K$ 的结构设计，不讨论位置编码、Value 压缩、FlashAttention 这类实现优化。

问题定义很明确：给定输入矩阵 $X\in\mathbb{R}^{n\times d_{\text{model}}}$，如何设计 $W_K$，使得注意力分数既能表达有效匹配，又能在训练初期保持稳定。

这里有两个边界条件必须先说清：

| 条件 | 含义 | 是否必须 |
|---|---|---|
| $d_q=d_k$ | Query 和 Key 必须在同一维空间做点积 | 必须 |
| $d_v$ 可不同于 $d_k$ | Value 是被加权汇总的内容向量 | 可以不同 |
| $W_Q,W_K$ 可独立 | 更强方向性，更高参数量 | 常见默认 |
| $W_Q=W_K$ 可共享 | 更省参数，但更接近对称核 | 特定场景可用 |

缩放因子 $\sqrt{d_k}$ 不是装饰项。若 $q,k$ 的各维独立、均值为 0、方差为 1，则未缩放点积的方差近似为：

$$
\mathrm{Var}(q\cdot k)=d_k
$$

缩放后：

$$
\mathrm{Var}\left(\frac{q\cdot k}{\sqrt{d_k}}\right)=O(1)
$$

这保证 logits 在维度变大时不会无限放大，softmax 不会过早进入饱和区。白话说，缩放是在给 softmax“限速”。

玩具例子先看一个最小版本。设 3 个 token 的输入为：

$$
X=
\begin{bmatrix}
1&0&1&0\\
0&1&1&0\\
1&1&0&0
\end{bmatrix}
,\quad
W_Q=W_K=W_V=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
0&0
\end{bmatrix}
$$

则：

$$
Q=K=V=XW=
\begin{bmatrix}
2&1\\
1&2\\
1&1
\end{bmatrix}
$$

第一个 token 对三个 token 的原始打分为：

$$
[2,1]\cdot
\begin{bmatrix}
2&1\\
1&2\\
1&1
\end{bmatrix}^{\top}
=
[5,4,3]
$$

缩放后变成：

$$
\left[\frac{5}{\sqrt{2}},\frac{4}{\sqrt{2}},\frac{3}{\sqrt{2}}\right]
$$

再经过 softmax，得到对三个 token 的注意力权重。这个过程展示了三件事：先投影，再点积，再缩放归一化，缺一不可。

---

## 核心机制与推导

注意力本质上是在构造一个两两匹配矩阵。对输入 $X$ 而言：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

于是分数矩阵为：

$$
S=\frac{QK^\top}{\sqrt{d_k}}=\frac{XW_QW_K^\top X^\top}{\sqrt{d_k}}
$$

这个式子很关键。因为它说明注意力分数不只取决于输入 $X$，还取决于中间双线性核 $W_QW_K^\top$。白话说，真正决定“怎么比较两个 token”的，不是 token 本身，而是这两个投影矩阵共同定义的比较规则。

如果 $W_Q$ 与 $W_K$ 独立，那么 $W_QW_K^\top$ 一般不是对称矩阵，于是有：

$$
s_{ij}\neq s_{ji}
$$

这就是注意力的非对称性来源。它能表达“我向你提问”和“你向我提问”不是同一件事。

如果令 $W_Q=W_K=W$，则：

$$
S=\frac{XWW^\top X^\top}{\sqrt{d_k}}
$$

由于 $WW^\top$ 对称，所以分数矩阵在 softmax 之前是对称的。注意，这并不意味着最终每一行 softmax 后完全一样，因为 softmax 是按行归一化的；但它意味着底层兼容度来自一个对称核，更偏向“相互接近”而非“单向查询”。

这时若想恢复方向感，可以插入可学习矩阵 $S_p$：

$$
\tilde{S}=ZW_SZ^\top,\quad Z=XW
$$

若 $S_p$ 不是对称矩阵，则一般有：

$$
\tilde{s}_{ij}\neq \tilde{s}_{ji}
$$

因此 pairwise attention 的本质，是在共享特征抽取器 $W$ 的前提下，用额外的小矩阵重新引入方向性。

下面把三种方案放在一起：

| 方案 | 打分形式 | 参数规模 | 方向性 | 适合场景 |
|---|---|---|---|---|
| 原始独立投影 | $(x_iW_Q)(x_jW_K)^\top$ | 高 | 强 | 通用 NLP、问答、跨模态检索 |
| 对称共享投影 | $(x_iW)(x_jW)^\top$ | 低 | 弱 | 语义相似性强、资源紧张 |
| Pairwise 共享投影 | $(x_iW)S(x_jW)^\top$ | 中 | 中到强 | 想省参数但仍需方向性 |

继续用上面的玩具例子。如果把第一个 token 理解成“我想找一个名词特征”，第二个 token 理解成“我能提供名词相关线索”，那么独立 $W_Q,W_K$ 时，模型可以学到“问”和“答”是不同子空间。若强行共享成同一个 $W$，那么“我在找什么”和“我能提供什么”被压到同一投影里，适合做相似匹配，不适合做角色区分。

真实工程例子可以看问答系统。用户问题中的词，往往承担 Query 角色；候选上下文中的词，承担 Key 角色。此时“法国的首都”去看“巴黎”是合理的，但“巴黎”去看“法国的首都”不是同样的任务。如果 $W_Q=W_K$，模型会更倾向对称语义对齐；若使用独立投影或共享投影加 $S$，则更容易保留“查询指向答案”的方向性。

---

## 代码实现

下面用一个可直接运行的 Python 例子，把标准注意力、对称共享投影、pairwise 共享投影放在一起。代码只依赖 `numpy`。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention_independent(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return scores, weights, out

def attention_symmetric(X, W, Wv):
    Z = X @ W
    V = X @ Wv
    scores = Z @ Z.T / np.sqrt(Z.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return scores, weights, out

def attention_pairwise(X, W, S, Wv):
    Z = X @ W
    V = X @ Wv
    scores = Z @ S @ Z.T / np.sqrt(Z.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return scores, weights, out

# 玩具输入：3 个 token，每个 token 4 维
X = np.array([
    [1., 0., 1., 0.],
    [0., 1., 1., 0.],
    [1., 1., 0., 0.],
])

W = np.array([
    [1., 0.],
    [0., 1.],
    [1., 1.],
    [0., 0.],
])

Wq = np.array([
    [1.0, 0.0],
    [0.2, 1.0],
    [0.8, 0.6],
    [0.0, 0.0],
])

Wk = np.array([
    [0.9, 0.1],
    [0.0, 1.1],
    [1.0, 0.7],
    [0.0, 0.0],
])

Wv = W.copy()
S = np.array([
    [1.0, 0.8],
    [-0.3, 1.2],
])

scores_i, weights_i, out_i = attention_independent(X, Wq, Wk, Wv)
scores_s, weights_s, out_s = attention_symmetric(X, W, Wv)
scores_p, weights_p, out_p = attention_pairwise(X, W, S, Wv)

# 断言 1：softmax 后每一行和为 1
assert np.allclose(weights_i.sum(axis=-1), 1.0)
assert np.allclose(weights_s.sum(axis=-1), 1.0)
assert np.allclose(weights_p.sum(axis=-1), 1.0)

# 断言 2：对称共享投影的打分矩阵是对称的
assert np.allclose(scores_s, scores_s.T)

# 断言 3：pairwise 如果 S 非对称，通常能打破对称性
assert not np.allclose(scores_p, scores_p.T)

print("independent scores:\n", np.round(scores_i, 3))
print("symmetric scores:\n", np.round(scores_s, 3))
print("pairwise scores:\n", np.round(scores_p, 3))
print("symmetric weights row0:", np.round(weights_s[0], 3))
```

这段代码对应的实现顺序是：

| 步骤 | 标准独立投影 | 对称共享 | Pairwise |
|---|---|---|---|
| 1 | $Q=XW_Q,\ K=XW_K$ | $Z=XW$ | $Z=XW$ |
| 2 | 算 $QK^\top$ | 算 $ZZ^\top$ | 算 $ZS Z^\top$ |
| 3 | 除以 $\sqrt{d_k}$ | 除以 $\sqrt{d_k}$ | 除以 $\sqrt{d_k}$ |
| 4 | 行 softmax | 行 softmax | 行 softmax |
| 5 | 乘 $V$ | 乘 $V$ | 乘 $V$ |

真实工程里，一般不会只写这一层，还会接输出投影、残差连接和 LayerNorm。Pre-LN 架构里，LayerNorm 会放在注意力前面，目的是先把输入分布稳定住，再进入投影层。白话说，先把电压调平，再送进精密仪器。

---

## 工程权衡与常见坑

实际设计 $W_K$ 时，最常见的误判不是“公式不会写”，而是忽略了表达能力和训练稳定性的冲突。

先看权衡：

| 设计点 | 好处 | 代价 |
|---|---|---|
| 独立 $W_Q,W_K$ | 方向性强，表达丰富 | 参数更多，显存更高 |
| 共享 $W_Q=W_K$ | 参数少，实现简单 | 注意力更接近对称相似度 |
| 共享后加 $S$ | 兼顾省参数与方向性 | 多一层双线性打分，略增复杂度 |

第一个常见坑，是“对称化扭曲”。

很多人看到 $W_Q$ 和 $W_K$ 输入都来自同一个 $X$，就会直接共享权重，觉得只是少一个矩阵。但共享后，模型更难区分“提问方”和“供给方”。这在分类检索、问答、工具调用、跨模态文本引导视觉定位里都可能出问题。修正方法不是一刀切地禁止共享，而是看任务是否需要方向性。若需要，可加 $S$、相对位置偏置、角色偏置，或者干脆保留独立投影。

第二个常见坑，是初始化方差过大导致注意力熵塌缩。熵塌缩的白话解释是：softmax 一开始就把几乎所有概率压到一两个 token 上，模型“只盯一个地方看”，训练会变得尖锐、不稳定。

如果 $W_Q,W_K$ 初始化过大，则 $QK^\top$ 的值域也会大，虽然有 $\sqrt{d_k}$ 缩放，但仍可能让 logits 过大。经验上可采用 SmallInit 思路，把标准差设为：

$$
\mathrm{std}=\sqrt{\frac{2}{5d}}
$$

这里的 $d$ 可取输入或投影维度的同阶量级。它的作用不是提升表达力，而是让初始分数留在 softmax 的可训练区间。

对应的坑与修正如下：

| Pitfall | 现象 | Fix |
|---|---|---|
| 共享投影后方向性不足 | A 看 B 与 B 看 A 区分不明显 | 加 $S$，或恢复独立 $W_Q,W_K$ |
| 初始化方差过大 | 注意力过早尖锐，loss 抖动 | 用 SmallInit |
| 缩放遗漏或错误 | logits 随维度膨胀 | 使用 $\sqrt{d_k}$ |
| 后归一化不稳 | 深层训练易爆炸或退化 | 采用 Pre-LN |
| 多头维度过小 | Key 空间表达贫弱 | 提高 head_dim 或增加 heads |

给一个新手最容易遇到的例子。假设你手写了一个小 Transformer，把 $W_K$ 初始化成标准正态 $\mathcal{N}(0,1)$，head_dim 又设成 64。虽然代码能跑，但初始 logits 往往已经很大，softmax 接近 one-hot，结果是梯度集中、训练初期非常不稳。把初始化改成 $\sqrt{2/(5d)}$ 后，注意力分布通常会更平滑，模型能先学大结构，再学尖锐对齐。

---

## 替代方案与适用边界

如果只从“如何设计 Key 投影矩阵”出发，至少有三类替代路线。

| 方案 | 适用边界 | 优势 | 风险 |
|---|---|---|---|
| Symmetric | 输入双方语义接近，强调共享表示 | 参数省、实现简单 | 难表达方向性 |
| Pairwise | 想共享特征抽取，但还需要非对称关系 | 折中较好 | 设计与调参更复杂 |
| Low-rank/Nyström/DCT | 长序列，主要瓶颈在 $QK^\top$ 成本 | 降低算力和显存 | 细粒度依赖可能丢失 |

Symmetric attention 适合什么场景？典型是两边信息结构近似、目标偏向相似性融合，而不是查询应答。多模态图像融合就是一个真实工程例子：若红外特征和可见光特征都在做“同一场景的互补表达”，共享投影就相当于共享特征提取器，可以减少参数和显存占用。

但如果你把同样的结构直接搬到问答系统、检索增强生成、指令跟随代理里，风险就会上来。因为这些任务依赖方向性。“问题找答案”和“答案找问题”不是同一映射。此时更合理的方案通常是：

1. 保留独立 $W_Q,W_K$。
2. 或者共享底层 $W$，在 head 内加入 $S$。
3. 再配合位置偏置或角色嵌入，明确区分来源。

低秩近似则是另一条线。它不是直接修改 $W_K$ 的语义，而是近似整个注意力核，解决长序列的二次复杂度问题。这类方法在超长上下文里很重要，但它解决的是“算不起”，不是“投影该怎么表达”。所以只有当序列长度已经成为主瓶颈时，才应该优先考虑它。

可以把选择标准压缩成一句话：如果你要的是“谁和谁像”，共享投影可行；如果你要的是“谁向谁取信息”，独立投影或 pairwise 更稳妥。

---

## 参考资料

- Scaled Dot-Product Self-Attention 概览：<https://www.emergentmind.com/topics/scaled-dot-product-self-attention>
- Pre-LayerNorm Transformer 与 SmallInit：<https://www.emergentmind.com/topics/pre-layernorm-transformer>
- Transformer 页面公式与维度约束：<https://en.wikipedia.org/wiki/Transformer_(deep_learning)>
- Query/Key/Value 角色解释：<https://artificial-intelligence-wiki.com/ai-development/model-architecture-design/attention-mechanisms-explained/>
- Self-Attention 基础示例：<https://medium.com/@latifurrafi/understanding-self-attention-in-transformers-from-basics-to-mastery-489b1b3b8e16>
- Q/K/V 直观说明：<https://medium.com/@wasowski.jarek/self-attention-query-key-value-transformer-b7266afbe730>
- Transformer cheat sheet：<https://csinva.io/notes/cheat_sheets/transformers_cheat_sheet.html>
- Self-Attention from Scratch 示例：<https://codingowen.github.io/projects/self_attention/>
- Key 投影定义与最小示例：<https://hackmd.io/@e41406/BJogIAwZC>
- MixFuse 论文信息：<https://www.sciencedirect.com/science/article/abs/pii/S0957417424022942>
