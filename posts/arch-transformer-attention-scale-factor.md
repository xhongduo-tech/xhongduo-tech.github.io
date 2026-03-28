## 核心结论

注意力机制里的尺度缩放，指的是在 softmax 前把打分矩阵除以 $\sqrt{d_k}$。这里的 $d_k$ 是 key 向量维度，也常等于单个 attention head 的 `head_dim`。公式是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

它不是“经验调参”，而是一个直接来自方差分析的数值稳定性设计。若 $Q,K$ 的各维分量独立同分布、均值为 0、方差为 1，则未缩放点积 $q\cdot k$ 的方差是 $d_k$。维度越大，logits 越大。logits 是 softmax 前的原始分数。原始分数过大时，softmax 会接近 one-hot，意思是“几乎只选一个位置”，其梯度接近 0，训练会变得不稳定甚至学不动。

一个新手版直觉是：64 维向量做点积，结果的典型尺度大约会放大到 $\sqrt{64}=8$ 倍；再送进 softmax，相当于先开了 8 倍放大镜，分数差异被指数函数继续放大，注意力就会变得过尖。除以 8 以后，分数回到可学习区间，注意力分布更平滑，梯度也更容易传播。

---

## 问题定义与边界

问题很具体：为什么注意力分数不是直接用 $QK^\top$，而是要用 $QK^\top/\sqrt{d_k}$？

先把边界说清楚。

1. 这里讨论的是点积注意力，不是加性注意力。
2. 缩放发生在 softmax 之前，只做一次。
3. 缩放维度应该是单个 head 的 $d_k$，不是整个模型维度 `d_model`。
4. 这个设计主要解决的是大维度点积导致的 softmax 饱和问题。饱和是指输出已经极端接近 0 或 1，小变化几乎带不动梯度。

若假设每一维都满足：

$$
\mathbb{E}[q_i]=\mathbb{E}[k_i]=0,\quad \mathrm{Var}(q_i)=\mathrm{Var}(k_i)=1
$$

那么有：

$$
q\cdot k=\sum_{i=1}^{d_k} q_i k_i,\qquad \mathrm{Var}(q\cdot k)=d_k
$$

因此缩放后：

$$
\mathrm{Var}\!\left(\frac{q\cdot k}{\sqrt{d_k}}\right)=\frac{1}{d_k}\mathrm{Var}(q\cdot k)=1
$$

这意味着 attention logits 的统计尺度不再随维度增长。

下面这张表先给结论对比。

| 项目 | 不缩放 | 除以 $\sqrt{d_k}$ |
|---|---|---|
| logits 方差 | 随 $d_k$ 增长 | 约为常数 1 |
| softmax 分布 | 容易过尖、接近 one-hot | 更平滑 |
| 梯度规模 | 容易变小 | 更稳定 |
| 对 `head_dim` 变化的敏感性 | 高 | 低 |
| 训练表现 | 容易震荡、难调学习率 | 更容易收敛 |

玩具例子可以直接看 $d_k=64$。若每维是标准分布，未缩放点积的标准差约为 $\sqrt{64}=8$。这不是说每个分数都等于 8，而是说它们常常落在一个“对 softmax 来说已经偏大”的区间里。softmax 的指数会把这类差异继续放大，导致某个位置拿走几乎全部注意力权重。

---

## 核心机制与推导

核心推导只有两步：先看点积的方差，再看 softmax 的梯度。

### 1. 为什么点积方差是 $d_k$

对单个位置的 query 和 key，点积是：

$$
q\cdot k=\sum_{i=1}^{d_k} q_i k_i
$$

在独立假设下：

$$
\mathbb{E}[q_i k_i]=\mathbb{E}[q_i]\mathbb{E}[k_i]=0
$$

又因为 $\mathrm{Var}(q_i)=\mathrm{Var}(k_i)=1$，有：

$$
\mathrm{Var}(q_i k_i)=\mathbb{E}[q_i^2k_i^2]-0
= \mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=1
$$

于是

$$
\mathrm{Var}(q\cdot k)
= \sum_{i=1}^{d_k}\mathrm{Var}(q_i k_i)
= d_k
$$

所以未缩放时，维度每增大一倍，点积的标准差按 $\sqrt{d_k}$ 增大。这正是问题来源。

### 2. 为什么 softmax 会失去梯度

softmax 把 logits $z$ 变成概率：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

它的雅可比矩阵是：

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
$$

对角项就是常见的：

$$
\frac{\partial p_i}{\partial z_i}=p_i(1-p_i)
$$

这条式子很重要。若某一项 $p_i\approx 1$，则 $p_i(1-p_i)\approx 0$；若某一项 $p_i\approx 0$，则也接近 0。也就是说，当 softmax 已经非常尖锐时，几乎所有位置的梯度都会变小。

看一个玩具例子：

- 未缩放 logits：$[10,12,8]$
- 若 $d_k=64$，缩放后 logits：$[1.25,1.5,1.0]$

对应的 softmax 与近似梯度如下。

| logits | softmax 输出 | 对角近似梯度 $p(1-p)$ |
|---|---|---|
| $[10,12,8]$ | $[0.117, 0.867, 0.016]$ | $[0.103, 0.115, 0.016]$ |
| $[1.25,1.5,1.0]$ | $[0.326, 0.419, 0.254]$ | $[0.220, 0.243, 0.190]$ |

未缩放时，最大项已经明显主导；缩放后，三个位置都还能分到可学习的梯度。这里不要求梯度“越大越好”，而是要求它不要过早进入几乎为零的饱和区。

对初学者来说，可以把它理解成：softmax 不是线性函数，而是一个会放大差异的竞争器。logits 本身已经因为维度增大而变大，再经过指数，就容易变成“赢家通吃”。

真实工程例子是大语言模型训练。假设 `head_dim=128`，使用 BF16。BF16 是一种低精度浮点格式，指数范围够用，但尾数精度较低，更怕前向分数分布太极端。若遗漏了 $\sqrt{d_k}$ 缩放，某些头会很快输出极尖的注意力图，训练中表现为 loss 抖动、梯度在不同 step 间波动很大、不同头利用率不均。有缩放时，注意力权重更平滑，反向传播更稳定。

---

## 代码实现

最小可运行例子先用纯 Python 演示。它验证两件事：

1. 点积方差确实接近 $d_k$
2. 除以 $\sqrt{d_k}$ 后，方差回到接近 1

```python
import math
import random

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def variance(xs):
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)

random.seed(0)
d_k = 64
num_samples = 5000

raw_scores = []
scaled_scores = []

for _ in range(num_samples):
    q = [random.gauss(0, 1) for _ in range(d_k)]
    k = [random.gauss(0, 1) for _ in range(d_k)]
    s = dot(q, k)
    raw_scores.append(s)
    scaled_scores.append(s / math.sqrt(d_k))

raw_var = variance(raw_scores)
scaled_var = variance(scaled_scores)

print("raw_var =", raw_var)
print("scaled_var =", scaled_var)

assert 45 < raw_var < 85
assert 0.7 < scaled_var < 1.3
```

如果你手写 attention，关键代码只有一行：

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, V)
```

如果用 PyTorch 的现成接口，`torch.nn.functional.scaled_dot_product_attention` 默认就会使用 `1 / sqrt(head_dim)` 作为缩放因子：

```python
import math
import torch
import torch.nn.functional as F

B, H, T, D = 2, 4, 8, 64
Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
V = torch.randn(B, H, T, D)

out = F.scaled_dot_product_attention(Q, K, V)

manual_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1 / math.sqrt(D))
manual_attn = torch.softmax(manual_scores, dim=-1)
manual_out = manual_attn @ V

assert out.shape == (B, H, T, D)
```

这里有一个工程细节：PyTorch 接口里的 `scale` 是乘法因子，所以若手写是“除以 $\sqrt{d_k}$”，在接口实现里通常表现为“乘以 $1/\sqrt{d_k}$”。数学上等价。

---

## 工程权衡与常见坑

最常见的问题不是“不知道这个公式”，而是代码里写错位置、写错维度，或者调试时没观察 score 分布。

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 忘记缩放 | softmax 极尖，训练抖动 | logits 方差随 $d_k$ 增长 | 显式写 `1 / sqrt(head_dim)` |
| 用错维度 | 表现忽好忽坏 | 用成 `d_model` 而非 `head_dim` | 检查 `Q.size(-1)` |
| 在 softmax 后才缩放 | 基本无效 | 问题发生在 softmax 输入端 | 缩放必须在 softmax 前 |
| 调试只看 loss | 很难定位 | loss 不直接暴露注意力数值问题 | 打印 `scores.mean()` 和 `scores.std()` |
| 低精度训练不看分布 | 容易误判为优化器问题 | 实际是注意力分数过激 | 联合看 logits、梯度、学习率 |

一个直接可用的调试片段：

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
print("scores mean:", scores.mean().item())
print("scores std :", scores.std().item())
```

经验上，如果 `scores.std()` 明显大于 1 且 softmax 后长期接近 one-hot，就要先检查缩放是否遗漏或维度是否写错。

真实工程里还会碰到一个误区：看到训练不稳定，就先加梯度裁剪、调学习率、改 warmup。它们有时能缓解症状，但如果 attention scale 本身就错了，这些方法只是“压住后果”，没有修正根因。

---

## 替代方案与适用边界

$\frac{1}{\sqrt{d_k}}$ 是默认方案，但不是唯一能工作的方案。关键不是“必须是这个魔法数字”，而是要让 logits 保持在 softmax 的有效工作区间。

| 方案 | 形式 | 优点 | 风险 | 适用场景 |
|---|---|---|---|---|
| 固定 scale | $\frac{1}{\sqrt{d_k}}$ | 简单、稳定、标准做法 | 几乎无额外自由度 | 大多数 Transformer |
| 可学习 scale | $\alpha \cdot \frac{1}{\sqrt{d_k}}$ | 允许模型微调温度 | $\alpha$ 失控会重新饱和 | 有充分监控的研究实验 |
| 自适应 temperature | $\frac{QK^\top}{\tau(x)}$ | 可按样本动态调节 | 实现复杂，训练更难稳 | 特殊任务或论文探索 |
| 改用加性注意力 | 非点积形式 | 对大维度更稳 | 计算慢，难并行优化 | 小模型或特定结构 |

若要做 learnable scale，常见写法是：

$$
\mathrm{scores} = \alpha \cdot \frac{QK^\top}{\sqrt{d_k}}
$$

这里 $\alpha$ 是可学习参数。它可行，但初值通常仍应设为 1，也就是整体初始 scale 仍是 $1/\sqrt{d_k}$。否则模型一开始就可能落到 softmax 饱和区。

适用边界也要说明白：

1. 这个推导依赖独立同分布、单位方差等近似假设，真实训练中不会严格成立。
2. 但结论依然成立，因为它抓住的是“点积尺度会随维度增大”这个主导趋势。
3. 即使使用 FlashAttention、混合精度、KV cache，这个 scale 仍然通常保留，只是实现位置可能被融合进 kernel。
4. 若模型结构不是标准点积 attention，这个结论不能机械照搬。

---

## 参考资料

| 来源 | 作用 | 关键内容 |
|---|---|---|
| [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/01/attention.html) | 公式与原始推导 | 给出 $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$，并说明未缩放时大 $d_k$ 会让 softmax 梯度极小 |
| [PyTorch `scaled_dot_product_attention` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) | 工程实现参考 | 说明默认 `scale_factor = 1 / sqrt(query.size(-1))`，对应单个 head 的最后一维 |
| [Imad Dabbura: The Transformer Architecture](https://imaddabbura.github.io/posts/nlp/Transformer-Architecture-Explained.html) | 直观解释方差与梯度 | 解释为什么未缩放会导致 softmax 接近 one-hot，缩放后更利于梯度流动 |
| [AI Wiki: Scaled Dot-Product Attention](https://artificial-intelligence-wiki.com/natural-language-processing/transformer-architecture-and-attention/scaled-dot-product-attention/) | 数值例子与饱和直觉 | 用大维度下的分数范围说明 softmax 饱和与近似 one-hot 的风险 |
| [ApXML: Scaled Dot-Product Attention](https://apxml.com/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts/scaled-dot-product-attention) | 入门向补充阅读 | 适合快速建立“为什么要 scale”的整体直觉 |
| [HogoNext: Debug Transformer Attention Issues](https://hogonext.com/how-to-debug-transformer-attention-issues/) | 调试实践补充 | 关注 attention 分布、数值范围与训练异常的排查方法 |
