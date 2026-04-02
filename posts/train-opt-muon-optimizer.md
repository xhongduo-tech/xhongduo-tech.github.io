## 核心结论

Muon 是一种面向矩阵参数的优化器。这里的“矩阵参数”指形状像 $W\in\mathbb{R}^{h\times w}$ 的二维权重，例如 Transformer 里的线性层权重。它不直接拿原始梯度更新权重，而是先做两件事：

1. 用动量把最近几步梯度累积成更稳定的方向。
2. 用 Newton-Schulz 迭代把这个方向“正交化”。正交化可以先理解成：把不同方向上的拉伸强弱抹平，让更新更像“只保留方向、不保留大小失衡”。

最终更新可以写成：

$$
W_t = W_{t-1} - \eta \left(O_t + \lambda W_{t-1}\right)
$$

其中 $O_t$ 是正交化后的更新方向，$\eta$ 是学习率，$\lambda$ 是权重衰减系数。

直觉上，传统 SGD 或 AdamW 很容易让少数大奇异值方向主导更新。奇异值可以先理解成“矩阵在某些方向上放大或缩小的强度”。Muon 的目标就是把这种不均衡压平，让每个方向推进得更平均。

一个面向新手的玩具描述是：如果把线性层权重看成一张二维表，普通优化器会在表里某几个格子上施加更大的力；Muon 则先把这股力整理成“各方向长度更一致”的推力，再按统一步长推动参数。它不是逐元素修补，而是按整个矩阵的几何结构更新。

Muon 不是 AdamW 的简单替代，而更像“线性层专用更新规则”。实际工程里，推荐做法通常是：

- 线性层矩阵参数用 Muon
- embedding、LayerNorm、bias 等非 2D 或不适合正交化的参数继续用 AdamW

这也是为什么很多实测会写成 “Muon + AdamW hybrid”。

| 维度 | Muon | AdamW |
|---|---|---|
| 关注对象 | 整个矩阵的方向结构 | 每个元素独立缩放 |
| 方向均衡性 | 强，显式压平奇异值差异 | 弱，可能被大方向主导 |
| 低精度稳定性 | 通常较好，尤其 bfloat16 训练 | 已成熟，但矩阵几何未被显式利用 |
| 适用范围 | 主要是 2D 线性层权重 | 几乎所有参数 |

---

## 问题定义与边界

问题的核心不是“如何把梯度变小”，而是“如何让矩阵更新更符合矩阵本身的几何结构”。传统优化器大多按元素处理参数，即把权重矩阵当成很多独立数字。这种做法简单，但会忽略一个事实：线性层真正作用在输入上的方式，是整个矩阵共同决定的。

设某层梯度是：

$$
G=\operatorname{diag}(3,1)=
\begin{bmatrix}
3 & 0\\
0 & 1
\end{bmatrix}
$$

如果直接用 SGD 更新，那么第一方向的步长会是第二方向的 3 倍。也就是说，训练会优先沿着“强方向”移动，而弱方向推进较慢。若这个不均衡长期存在，优化过程就容易被少数方向绑架。

Muon 的出发点是：对矩阵而言，重要的不只是元素大小，还包括谱结构。这里的“谱结构”可以先理解成“这个矩阵在不同方向上的作用强弱分布”。因此，Muon 不直接使用 $G$，而是把动量后的矩阵做正交化，使更新更接近“每个方向长度一致”。

但它有明确边界。

| 参数类型 | 是否适合 Muon | 原因 |
|---|---|---|
| 线性层权重 `W[h, w]` | 是 | 标准二维矩阵，适合做正交化 |
| 卷积核 | 视实现而定 | 常需 reshape，且几何含义不总是稳定 |
| Embedding | 否，通常交给 AdamW | 行向量语义特殊，不适合直接套矩阵正交化 |
| LayerNorm 权重 | 否 | 通常是一维缩放参数 |
| Bias | 否 | 一维参数，没有矩阵谱结构 |

因此，Muon 解决的是“二维权重矩阵如何更均衡地更新”，不是“所有参数统一优化”的问题。只要参数不是标准 2D 矩阵，或者正交化成本不值得，通常就不该强行用 Muon。

还有一个边界很关键：Muon 主要改善的是更新方向的几何形状，不等于它自动解决学习率设置、权重衰减、梯度爆炸等全部训练问题。它仍然需要和 warmup、权重衰减、混合精度策略一起配合。

---

## 核心机制与推导

Muon 的机制可以拆成四步：动量累积、Nesterov 预测、Newton-Schulz 正交化、尺度归一。

第一步是动量。动量可以理解成“对过去梯度做指数加权平均”，作用是减少抖动，让更新方向更稳定：

$$
m_t = \mu m_{t-1} + g_t
$$

其中 $g_t$ 是当前梯度，$\mu$ 是动量系数。

第二步可选 Nesterov。Nesterov 可以先理解成“提前看一步的动量修正”，常写成：

$$
\tilde g_t = g_t + \mu m_t
$$

于是，真正送入正交化模块的不是原始梯度，而是更稳定的 $\tilde g_t$。

第三步是核心：对 $\tilde g_t$ 做 Newton-Schulz 迭代，逼近它的极分解中的正交因子。极分解可以先理解成：任意矩阵大致可以拆成“纯方向旋转部分”和“拉伸缩放部分”。Muon 想保留前者，削弱后者。

若记正交化结果为：

$$
O_t = \operatorname{NS}(\tilde g_t)
$$

那么 $O_t$ 会接近一个奇异值都接近 1 的矩阵。这里“奇异值都接近 1”意味着不同方向被压到统一尺度，不再有某个方向特别大。

一个玩具例子最直观。假设：

$$
\tilde g_t=
\begin{bmatrix}
3 & 0\\
0 & 1
\end{bmatrix}
$$

它的两个奇异值分别是 $3$ 和 $1$。普通 SGD 直接用它更新时，第一方向推进是第二方向的 3 倍。Muon 经过 Newton-Schulz 若逼近到一个正交矩阵或近正交矩阵，则更新矩阵的奇异值会更接近：

$$
(1,1)
$$

也就是“方向保留，尺度压平”。

可以用一张表看这个变化：

| 阶段 | 矩阵示意 | 奇异值示意 | 含义 |
|---|---|---|---|
| 原始梯度 | $\operatorname{diag}(3,1)$ | $(3,1)$ | 第一方向主导 |
| 动量后 | 仍可能不均衡 | 例如 $(2.8,1.2)$ | 抖动减少，但失衡仍在 |
| NS 正交化后 | 近正交矩阵 | 约 $(1,1)$ | 每个方向更新量更平均 |

第四步是尺度归一。很多实现会按 $\sqrt{\max(h,w)}$ 做额外归一化，使 Muon 的更新量级更容易和 AdamW 的学习率体系对齐。直观理解是：矩阵越大，未经处理的整体更新范数通常也越大，所以需要一个和形状相关的缩放因子，避免“大矩阵天然迈更大步”。

于是最终更新写成：

$$
W_t = W_{t-1} - \eta \left(\frac{O_t}{\sqrt{\max(h,w)}} + \lambda W_{t-1}\right)
$$

如果不写归一项，也可以把它吸收到学习率里，但工程上单独写出来更清楚。

真实工程例子可以看大模型微调。比如一个 Qwen3-4B 级别的 Transformer，注意力投影层和 MLP 层里有大量二维线性权重。这些层的更新最容易受矩阵谱结构影响。若对这些层用 Muon，对 embedding、LayerNorm、bias 继续用 AdamW，通常会比“全部参数都用 AdamW”更快进入低损失区间，且在 bfloat16 下更稳。这类收益主要来自：线性层是大模型计算主体，也是最适合矩阵级更新规则的部分。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是高性能训练代码，但足够说明 Muon 的核心步骤：动量更新、Newton-Schulz 迭代、形状归一、权重更新。

```python
import numpy as np

def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
    # 先按谱范数近似做缩放，避免迭代数值发散
    norm = np.linalg.norm(G, ord=2)
    X = G / (norm + eps)

    # 一个简单的 Newton-Schulz 风格迭代：
    # X <- 1.5 X - 0.5 X X^T X
    # 目标是让 X^T X 接近 I
    for _ in range(steps):
        X = 1.5 * X - 0.5 * X @ X.T @ X
    return X

def muon_update(W, grad, momentum, lr, weight_decay, mu=0.95, ns_steps=5):
    m = mu * momentum + grad
    # Nesterov 形式
    g_tilde = grad + mu * m

    O = newton_schulz_orthogonalize(g_tilde, steps=ns_steps)
    scale = np.sqrt(max(W.shape))
    W_new = W - lr * (O / scale + weight_decay * W)
    return W_new, m, O

# 玩具例子：梯度在两个方向上强弱明显不同
W = np.array([[1.0, 0.0],
              [0.0, 1.0]], dtype=np.float64)
grad = np.array([[3.0, 0.0],
                 [0.0, 1.0]], dtype=np.float64)
momentum = np.zeros_like(W)

W_new, momentum, O = muon_update(
    W=W,
    grad=grad,
    momentum=momentum,
    lr=0.1,
    weight_decay=0.01,
    mu=0.95,
    ns_steps=5
)

# O 应接近正交：O^T O ≈ I
I = O.T @ O
assert np.allclose(I, np.eye(2), atol=1e-2), I

# 更新后矩阵形状不变
assert W_new.shape == W.shape

# 说明确实发生了更新
assert np.linalg.norm(W_new - W) > 0
```

上面这段代码有三个要点。

第一，Muon 更新的输入应该是矩阵，而不是任意形状张量。第二，Newton-Schulz 只做少量迭代，工程里常见经验值是 5 步。第三，学习率不一定要重新发明，很多实现通过形状归一后可以沿用 AdamW 量级。

伪代码可以写得更接近训练器：

```python
for step in training_steps:
    for W in linear_layers:
        m[W] = beta * m[W] + grad(W)

        # momentum warmup: beta 从 0.85 逐步升到 0.95
        beta = schedule_beta(step)

        g_tilde = grad(W) + beta * m[W]
        O = newton_schulz(g_tilde, steps=5)
        O = O / sqrt(max(W.height, W.width))

        W -= lr * (O + weight_decay * W)

    for p in other_params:  # embedding / LN / bias
        adamw_update(p)
```

推荐把参数分组明确写出来，而不是在同一个优化器里隐式判断。这样更容易排查问题，也更符合 Muon 的使用边界。

一个常见超参数起点如下：

| 超参数 | 建议起点 | 说明 |
|---|---|---|
| momentum $\mu$ | 0.95 | 常见默认值 |
| momentum warmup | 0.85 $\rightarrow$ 0.95 | 训练初期更稳 |
| NS 步数 | 5 | 常见 sweet spot |
| 学习率 | 参考 AdamW 基线 | 配合形状归一复用 |
| 归一因子 | $\sqrt{\max(h,w)}$ | 对齐不同矩阵尺寸 |
| 权重衰减 | 与 AdamW 同量级起步 | 不必完全重调一套 |

---

## 工程权衡与常见坑

Muon 的收益并不是“免费”的。它把更新从逐元素操作提升到矩阵操作，因此要多做几次矩阵乘法。对大线性层来说，这个开销可能是值得的；对小模型、轻量任务或者不规则参数，则不一定。

第一个常见坑是把所有参数都交给 Muon。这样做通常是错的。Embedding、LayerNorm、bias 没有稳定的二维谱结构，强行正交化往往没有理论收益，反而让训练更难调。推荐流程很简单：

1. 识别所有线性层二维权重
2. 这些参数用 Muon
3. 其余参数继续用 AdamW

第二个常见坑是省略 momentum warmup。动量一开始就设得很高，可能让动量缓存过快积累，导致正交化前的矩阵范数突变，训练前几百步特别容易不稳。实践里，先从 0.85 之类的值升到 0.95，通常更平滑。

第三个坑是 Newton-Schulz 步数乱改。步数太少，正交化不充分；步数太多，计算增加，还可能在低精度下引入额外数值问题。5 步常被视为折中点，不是数学上唯一正确，而是工程上成本和效果平衡较好。

第四个坑是没做形状归一。不同线性层尺寸差异很大，如果不按 $\sqrt{\max(h,w)}$ 或等价规则缩放，大矩阵层和小矩阵层的更新量级会错位，导致你以为“Muon 不稳”，实际上是不同比例的层在抢学习率。

| 常见坑 | 表现 | 规避策略 |
|---|---|---|
| 所有参数都用 Muon | LN、bias、embedding 训练异常 | 仅对 2D 线性层启用 Muon |
| 省略 momentum warmup | 前期 loss 抖动大 | 从 0.85 升到 0.95 |
| NS 步数过多或过少 | 不收敛或开销过大 | 先固定 5 步 |
| 未做形状归一 | 不同层更新量级失衡 | 按 $\sqrt{\max(h,w)}$ 缩放 |
| 直接照搬 AdamW 参数组 | 混用逻辑混乱 | 显式拆分 Muon 组与 AdamW 组 |

真实工程里，一个常见案例是 Qwen3-4B 微调：所有 attention/MLP 线性层用 Muon，embedding、LayerNorm、bias 用 AdamW，同时做 momentum warmup，例如从 0.85 升到 0.95。这样的 hybrid 配置往往比全 AdamW 更快下降，也更不容易在低精度训练时出现梯度异常。这里的重点不是“Muon 单独完胜”，而是“把合适的层交给合适的优化规则”。

---

## 替代方案与适用边界

如果你的模型不大、训练预算有限，或者训练框架对自定义矩阵正交化支持一般，那么 AdamW 仍然是更稳妥的默认选项。AdamW 的优势是成熟、通用、实现广，几乎任何参数都能处理。Muon 的优势则更集中：它在大规模 Transformer 的线性层上，能更直接利用矩阵结构，让更新方向更均衡。

和 SGD 比，Muon 的差异尤其明显。SGD 可以很快说明问题本质，但它对矩阵谱结构几乎没有显式处理，通常更依赖学习率与调度技巧。AdamW 则用逐元素二阶统计近似改善稳定性，但它仍然是“按元素缩放”，不是“按矩阵方向重整”。

| 方案 | 收敛表现 | 稳定性 | 适用层 | 额外算力 |
|---|---|---|---|---|
| Muon | 大型线性层上常有优势 | 对低精度较友好 | 主要是 2D 矩阵 | 有正交化开销 |
| AdamW | 通用强基线 | 很成熟 | 几乎所有参数 | 中等 |
| SGD/Nesterov SGD | 简单直接 | 对超参数更敏感 | 通用，但大模型常不占优 | 低 |

适用场景可以这样看：

- 适合 Muon：大规模 Transformer 微调、预训练；线性层占主要计算；支持高效矩阵乘法；愿意做参数分组。
- 不太适合 Muon：小模型快速实验；参数大多不是标准二维矩阵；硬件对额外矩阵乘法不友好；训练框架难以插入自定义优化步骤。
- 更实际的折中：以 AdamW 为默认，仅把核心线性层切到 Muon。

一个真实边界案例是 Moonshot 的 Moonlight 3B/16B 预训练报告：在大规模 token 训练中，Muon 显示出较好的算力效率，有报告称在 5.7T token 规模下可用更少 FLOPs 达到相近损失。这类结果说明 Muon 的潜力主要在“大模型 + 长训练 + 线性层占主导”的组合里。反过来说，如果你只是训练一个小分类器，这种优化器切换很可能得不偿失。

因此，正确结论不是“以后都该用 Muon”，而是：当模型的主要难点已经变成矩阵级优化效率时，Muon 值得进入候选列表；否则，AdamW 仍然是默认基线。

---

## 参考资料

1. Tri Dao, Gram Newton-Schulz 解析  
   说明 Muon 背后的极分解与 Newton-Schulz 直觉。  
   https://tridao.me/blog/2026/gram-newton-schulz/

2. Jose David Baena, Muon Optimizer Explained  
   面向实现与直觉的解释，强调矩阵几何结构而非逐元素更新。  
   https://josedavidbaena.com/blog/nanochat/muon-optimizer-explained

3. `mlx_optimizers.Muon` 文档  
   给出公式、参数形式与实现接口，适合理解更新式。  
   https://stockeh.github.io/mlx-optimizers/build/html/_autosummary/mlx_optimizers.Muon.html

4. Hugging Face 社区文章，Muon + AdamW 实测  
   总结混合使用 Muon 与 AdamW 的经验，包括 warmup 与参数分组。  
   https://huggingface.co/blog/KingNish/optimizer-part1

5. Moonshot / Moonlight 相关预训练报告  
   用大规模预训练结果说明 Muon 在特定模型与算力条件下的收益边界。  
   可从 arXiv 与相关技术解读继续追踪原始实验设置。

6. Muon 8-bit 量化相关 OpenReview 材料  
   适合进一步阅读 Muon 在更低精度与更高效率方向上的扩展。  
   可在 OpenReview 检索 “Muon 8-bit” 相关论文条目。
