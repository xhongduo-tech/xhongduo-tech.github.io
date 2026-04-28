## 核心结论

Muon 是一种面向**矩阵参数**的优化器。矩阵参数就是形状像 `out_dim × in_dim` 的权重表，例如 Transformer 里的线性层权重。它先按 SGD/Nesterov 的方式累积动量，再把这个**动量矩阵**做一次近似正交化，然后再更新参数。

它和 AdamW 的关键差别，不是“每个元素怎么缩放”，而是“把整块更新矩阵当成几何对象处理”。直白说，AdamW 更像逐元素调步长；Muon 更像先把更新方向压成一组互不冗余的主方向，再整体前进。

| 对比项 | SGD / AdamW | Muon |
|---|---|---|
| 处理对象 | 本质上按向量或逐元素处理 | 明确利用矩阵结构 |
| 关键操作 | 动量、逐元素缩放、权重衰减 | 动量后再做正交化 |
| 主要目标 | 稳定下降、适配噪声梯度 | 减少病态方向中的冗余更新 |
| 更适合 | 通用场景 | 2D hidden weights，尤其大模型主干层 |
| 不建议直接用于 | 无特殊限制 | embedding、LM head、bias、LayerNorm、1D 参数 |

新手版可以这样理解：普通动量像“沿着当前坡度继续走”；Muon 像“先把你累积出来的走路方向校正得更直，再迈步”。它不是保证“相邻两步时间上彼此正交”，而是把**当前这一步的更新矩阵**变成最近的半正交矩阵，减少内部重复分量。

真实工程里，常见做法不是“全模型只用一个优化器”，而是分工：`hidden weights` 用 Muon，`embedding / LM head / bias / LayerNorm` 继续用 AdamW。这一点不是装饰，而是稳定训练的前提之一。

---

## 问题定义与边界

Muon 要解决的问题，是普通梯度或动量在**病态条件**下容易出现低效更新。病态条件可以简单理解为：不同方向的曲率差很多，有些方向很陡，有些方向很平。此时优化轨迹容易抖动、折返，等价于把很多步长浪费在重复或冗余方向上。

| 现象 | 后果 | Muon 解决方式 |
|---|---|---|
| 更新矩阵奇异值差异很大 | 主方向过强，次方向长期被压制 | 把更新投影到最近半正交矩阵 |
| 病态方向上来回震荡 | 有效步长下降，收敛慢 | 弱化尺度差异，保留方向结构 |
| 矩阵参数被当成向量处理 | 忽略行列耦合关系 | 显式利用矩阵几何结构 |

术语先做最小定义：

| 术语 | 最小定义 |
|---|---|
| 动量 | 把过去梯度做指数滑动平均，形成更稳定的更新方向 |
| 正交化 | 把矩阵变成“方向彼此不重复”的形式 |
| 矩阵参数 | 形状天然是 2D 的权重，如线性层和可展平卷积核 |

玩具例子：如果一个 $2 \times 2$ 更新矩阵在某一条主方向上很大、另一条方向很小，那么普通动量会继续保留这种尺度不平衡；Muon 会尽量把这一步更新压到“主方向互不重叠”的形式，再执行更新。

边界同样重要。Muon 不是“全参数通用优化器”，而是更像给主干矩阵权重定制的专用方法。

| 适合 | 不适合 |
|---|---|
| Transformer 的 attention / MLP 线性层权重 | bias |
| 2D hidden weights | LayerNorm / RMSNorm 标量或向量参数 |
| 可展平为矩阵的卷积核 | embedding |
| 大模型预训练的主干层 | 输出头 `LM head` 或 final classifier |

反例要明确：`embedding` 和 `bias` 虽然也能写成张量，但它们的优化行为与主干隐藏层不同。官方实现和文档都建议这些参数继续走 AdamW，而不是直接套 Muon。

---

## 核心机制与推导

Muon 的机制链路只有三步：

1. 先计算梯度 $g_t$
2. 再累积动量 $m_t$
3. 对动量矩阵做正交化，得到 $u_t$，再更新参数

核心公式是：

$$
m_t = \beta m_{t-1} + (1-\beta) g_t
$$

$$
u_t = \mathrm{Ortho}(m_t)
$$

$$
W_{t+1} = W_t - \eta u_t
$$

其中，$\beta$ 是动量系数，白话就是“过去方向保留多少”；$\eta$ 是学习率；$\mathrm{Ortho}(\cdot)$ 表示把矩阵投影到最近的半正交矩阵。

理想化情况下，如果

$$
m_t = U \Sigma V^\top
$$

那么正交化结果是

$$
u_t = UV^\top
$$

这里的含义很直接：保留左右奇异向量，也就是主要方向结构；把奇异值 $\Sigma$ 的尺度信息压平。于是更新不再被某几个大奇异值主导。

这也解释了一个容易误解的点：Muon 不是简单“归一化梯度范数”，它处理的是**矩阵的谱结构**。谱结构可以白话理解为“这块矩阵的主方向和伸缩强弱”。

玩具例子可以直接算：

$$
m = \begin{bmatrix}3 & 0 \\ 0 & 1\end{bmatrix}
$$

它的 SVD 很简单，就是 $U=I,\Sigma=\mathrm{diag}(3,1),V=I$，所以

$$
\mathrm{Ortho}(m)=UV^\top=I
$$

如果当前参数矩阵是

$$
W=\begin{bmatrix}2 & 1 \\ 0 & 4\end{bmatrix}, \quad \eta=0.1
$$

那么更新后

$$
W' = W - 0.1I
=
\begin{bmatrix}
1.9 & 1 \\
0 & 3.9
\end{bmatrix}
$$

关键不在于数值本身，而在于原本 $[3,1]$ 的尺度差被压成了单位正交方向。也就是：大方向不再无限挤占小方向的步长预算。

工程上通常不会每步做精确 SVD，因为太慢。实际实现更常用 **Newton-Schulz 迭代**，它是一类矩阵迭代法，可以高效逼近正交化结果。官方 Muon 实现使用的是多项式型 Newton-Schulz 近似，而不是精确极分解。

简化后的步骤表如下：

| 步骤 | 作用 |
|---|---|
| 1. 取动量矩阵 $m_t$ | 得到更稳定的原始更新 |
| 2. 按 Frobenius 范数缩放 | 把奇异值压到适合迭代的范围 |
| 3. 运行若干步 Newton-Schulz | 逼近 $UV^\top$ |
| 4. 必要时转置回原形状 | 保持与参数矩阵一致 |
| 5. 用结果更新参数 | 完成 Muon 更新 |

“理想投影”和“实际实现”必须区分开：

| 项 | 理想投影 | 实际实现 |
|---|---|---|
| 数学形式 | $U\Sigma V^\top \rightarrow UV^\top$ | 几步 Newton-Schulz 迭代近似 |
| 计算代价 | 高，SVD 昂贵 | 低，适合 GPU matmul |
| 精确性 | 高 | 近似，但通常足够好 |
| 工程价值 | 便于理解 | 便于训练大模型 |

真实工程例子：训练 Transformer 预训练模型时，attention 的 `W_q/W_k/W_v/W_o` 和 MLP 的投影矩阵通常是最典型的 Muon 目标。它们参数量大、结构明确、主导训练算力消耗，也是最可能从矩阵几何优化中获益的部分。

---

## 代码实现

理解代码时抓住一句话就够了：**先动量，再正交化，再更新**。

最小伪代码：

```python
m = beta * m + (1 - beta) * g
u = orthogonalize(m)   # usually Newton-Schulz approximation
W -= lr * u
```

下面给一个可运行的 Python 例子。它同时展示了“理想 SVD 正交化”和“Newton-Schulz 近似”的最小实现，并带有 `assert`。

```python
import numpy as np

def ortho_svd(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt

def newton_schulz5(m: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    # 官方博客给出的 quintic 系数
    a, b, c = 3.4445, -4.7750, 2.0315
    x = m.astype(np.float64).copy()
    x = x / (np.linalg.norm(x) + eps)

    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x

    if transposed:
        x = x.T
    return x

def muon_step(W, g, m_prev, lr=0.1, beta=0.95, use_exact=False):
    m = beta * m_prev + (1.0 - beta) * g
    u = ortho_svd(m) if use_exact else newton_schulz5(m, steps=5)
    W_next = W - lr * u
    return W_next, m, u

# 玩具例子
m = np.array([[3.0, 0.0],
              [0.0, 1.0]])
u_exact = ortho_svd(m)
assert np.allclose(u_exact, np.eye(2), atol=1e-8)

u_ns = newton_schulz5(m, steps=5)
assert np.allclose(u_ns, np.eye(2), atol=1e-2)

W = np.array([[2.0, 1.0],
              [0.0, 4.0]])
g = m.copy()
m_prev = np.zeros_like(g)

W_next, m_new, u = muon_step(W, g, m_prev, lr=0.1, beta=0.0, use_exact=True)
assert np.allclose(W_next, np.array([[1.9, 1.0], [0.0, 3.9]]), atol=1e-8)

print("Muon toy example passed.")
```

实际训练里，更重要的是**参数分组**，而不是把上面的线代代码手抄一遍。

| 参数组 | 推荐优化器 | 说明 |
|---|---|---|
| 2D hidden weights | Muon | attention / MLP 主干矩阵 |
| embedding | AdamW | 官方明确不建议直接用 Muon |
| LM head / final classifier | AdamW | 经验上更稳 |
| bias | AdamW | 1D 参数 |
| LayerNorm / RMSNorm | AdamW | 标量或向量参数 |

如果是卷积层，4D 卷积核通常按矩阵处理：把它展平成

$$
[\text{out\_channels}, \text{in\_channels} \times k_h \times k_w]
$$

再做正交化。这个规则本质上是在把卷积看成一次矩阵乘法的权重表。

关键实现参数通常有这些：

| 参数 | 作用 | 常见含义 |
|---|---|---|
| `ns_steps` | Newton-Schulz 迭代步数 | 越大越接近理想正交化，但更慢 |
| `momentum_beta` | 动量系数 | 过去梯度保留比例 |
| `weight_decay` | 权重衰减 | 控制参数规模，通常与学习率联调 |
| `lr` | 学习率 | 不能直接照搬 AdamW |
| `nesterov` | 是否用 Nesterov 动量 | 官方实现常默认开启或推荐使用 |

---

## 工程权衡与常见坑

Muon 的工程价值很明确，但代价也明确。

第一类权衡是**稳定性 vs 精确性**。理论里最漂亮的是精确 SVD 或极分解；工程里真正跑的是近似 Newton-Schulz。近似让它足够快，能上 GPU tensor cores，但也意味着 `ns_steps`、数值精度和缩放策略都会影响训练稳定性。

第二类权衡是**专用性 vs 通用性**。Muon 对 2D hidden weights 很有针对性，但这也意味着它不是“AdamW 全面替代者”。如果你把所有参数无差别切给 Muon，常见结果不是更快，而是更不稳。

常见坑可以直接列出来：

| 常见坑 | 说明 |
|---|---|
| 只适合 2D 参数 | 1D 参数直接用 Muon 通常不合理 |
| 把近似实现当精确 SVD | 理论公式和训练代码不是一回事 |
| 学习率照搬 AdamW | Muon 的有效步长语义不同 |
| 4D 卷积核不展平 | 会导致维度或语义不一致 |
| 忽略 embedding / output head 特例 | 这些层往往更适合 AdamW |
| 混合精度下不检查数值行为 | 近似迭代对精度策略敏感 |

排错时，优先检查这份清单：

1. `Muon` 是否只用于 2D hidden weights  
2. `embedding / LM head / bias / norm` 是否仍走 AdamW  
3. `ns_steps` 是否过少或过多  
4. `lr` 是否直接复用了 AdamW 的配置  
5. `weight_decay` 是否与 Muon 的步长一起调过  
6. 混合精度下矩阵乘法精度是否足够稳定  
7. QKV 是否按实现需要拆分为独立矩阵处理  

症状排查表：

| 症状 | 可能原因 | 修复建议 |
|---|---|---|
| loss 前期震荡明显 | `lr` 过大，或 `ns_steps` 太激进 | 先降 `lr`，再检查 `ns_steps` |
| hidden layers 稳定，embedding 发散 | 错把 embedding 放进 Muon | 把 embedding 切回 AdamW |
| 收敛速度没有提升 | 模型并非矩阵主导，或参数分组不对 | 先核对适用层，再做对照实验 |
| 训练吞吐下降过多 | 迭代步数过多，或实现未优化 matmul | 减少 `ns_steps`，检查核函数实现 |
| 理论推导正确但实验不稳 | 近似误差与精度设置影响实际表现 | 联调 `ns_steps`、精度、学习率 |

新手最容易踩的坑，是把 Muon 理解成“更高级的 AdamW”，然后试图全模型替换。这个方向通常不对。Muon 更接近“给矩阵主干层准备的专用几何更新器”。

---

## 替代方案与适用边界

如果你的任务不是大模型预训练，或者主参数并不以 2D hidden weights 为核心，那么 Muon 未必是默认首选。更成熟的替代方案仍然很多。

| 优化器 | 核心机制 | 优点 | 局限 |
|---|---|---|---|
| Muon | 动量后对矩阵更新做正交化 | 利用矩阵结构，适合主干矩阵层 | 专用性强，不适合全参数统一替换 |
| AdamW | 一阶动量 + 二阶尺度估计 + decoupled decay | 通用、稳、生态成熟 | 不显式利用矩阵几何 |
| SGD + Momentum | 简单动量更新 | 成本低、解释直接 | 对病态问题更易 zigzag |
| Lion | 符号化动量更新 | 状态更轻、实现简单 | 仍不做矩阵正交化 |
| Adafactor | 因子化二阶近似 | 节省显存，适合大模型 | 目标是省内存，不是正交化更新 |

维度适用边界也可以直接看：

| 参数类型 | Muon 是否优先 |
|---|---|
| 2D 矩阵 | 是 |
| 1D 参数 | 否 |
| embedding | 否 |
| output head | 通常否 |

可以把判断流程压成一句：

1. 是否是矩阵主导的主干权重  
2. 是否在做中大型模型训练，尤其预训练  
3. 是否愿意接受专用参数分组和额外调参成本  

如果三个问题里前两个都不是“是”，那通常优先 AdamW 或 SGD+Momentum 更合理。

场景判断可以总结为：

| 场景 | 更推荐 |
|---|---|
| 大模型 Transformer 主干预训练 | 优先考虑 Muon + AdamW 混合 |
| 中小模型通用训练 | 先用 AdamW |
| 非矩阵主导任务 | 先用 AdamW 或 SGD+Momentum |
| 需要极简稳定基线 | AdamW / SGD+Momentum |
| 显存紧张但想保留自适应特性 | Adafactor |

所以，Muon 的正确定位不是“新的默认优化器”，而是“当主干参数天然是矩阵，且你愿意做参数分工与调参时，一个可能更快的专用优化器”。

---

## 参考资料

1. [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)（定义）
2. [KellerJordan/Muon GitHub 实现](https://github.com/KellerJordan/Muon)（实现）
3. [NVIDIA NeMo Emerging Optimizers: Muon](https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html)（实现）
4. [Convergence of Muon with Newton-Schulz](https://openreview.net/forum?id=lJSfxtLpLm)（理论分析）
5. [On the Convergence Analysis of Muon](https://openreview.net/forum?id=4nH4CulGaP)（理论分析）
