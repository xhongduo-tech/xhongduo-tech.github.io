## 核心结论

Adafactor 是一种面向大矩阵参数的低内存自适应优化器。自适应优化器的意思是：它会根据历史梯度大小，给不同参数分配不同步长。它最核心的价值不是“收敛一定比 Adam 更快”，而是“在尽量保留 Adam 类方法效果的前提下，把优化器状态内存显著压低”。

如果参数是一个 $m \times n$ 的矩阵，Adam 通常需要为每个元素保存完整的二阶矩状态，状态规模是 $O(mn)$。Adafactor 不保存整张矩阵，而是只保存“每一行的统计量”和“每一列的统计量”，状态规模变成 $O(m+n)$。当矩阵是方阵时，这就是从 $O(n^2)$ 降到 $O(n)$。

这件事的工程意义很直接：当模型里有很多超大权重矩阵时，真正先爆掉的常常不是参数本身，而是优化器状态。Adafactor 解决的是这个问题，所以它特别适合 TPU 上的大规模预训练，也被 T5、PaLM 这类模型采用。

| 优化器 | 状态规模 | 特点 |
|---|---:|---|
| Adam / AdamW | $O(mn)$ | 稳定、通用、状态大 |
| Adafactor | $O(m+n)$ | 省内存、近似更强、实现更讲究 |

一句话判断：如果训练瓶颈是优化器状态内存，Adafactor 值得优先考虑；如果内存不是瓶颈，AdamW 往往更省心。

---

## 问题定义与边界

先定义问题。优化器状态，就是优化器为每个参数额外保存的历史信息。以 Adam 为例，它通常要保存一阶矩和二阶矩，也就是“历史平均梯度”和“历史平均平方梯度”。这让 Adam 在大多数任务上很好用，但代价是状态开销很大。

对零基础读者，一个直接理解方式是：模型参数像账本里的“当前余额”，优化器状态像“历史流水”。大模型训练时，内存不只装参数，还要装这套历史流水。于是你会发现，训练大模型时，参数只是成本的一部分，优化器状态同样可能成为主要成本。

Adafactor 的边界也必须说清楚。它主要优化的是“状态内存”，不是所有训练指标都无条件更优。也就是说，它是一个工程折中方案，不是 AdamW 的全面替代品。

| 问题 | Adam 的表现 | Adafactor 的应对 |
|---|---|---|
| 状态内存太大 | 每个参数都要存历史信息 | 用行/列统计量近似完整二阶矩 |
| TPU 上训练大模型 | 容易受 HBM 内存限制 | 更容易把大矩阵塞进同样预算 |
| 小模型或常规任务 | 状态开销未必是瓶颈 | 收益可能有限 |

适合 Adafactor 的典型场景：

- 大 Transformer
- 长序列训练
- TPU 或显存/内存非常紧张的环境
- 参数里有大量二维矩阵，例如 embedding、attention projection、FFN 权重

不太适合的场景：

- 模型不大，优化器状态不是瓶颈
- 更重视调参直觉和生态一致性
- 需要和大量现成 AdamW 配方直接兼容

---

## 核心机制与推导

先看 Adam 想做什么。Adam 的二阶矩本质上是在估计每个参数位置“过去梯度平方通常有多大”。如果某个位置历史上梯度一直很大，就把更新缩小；如果一直很小，就把更新放大。这个“按历史尺度缩放梯度”的过程叫预条件化，可以理解为“先把不同方向的梯度单位拉平，再更新”。

对于矩阵参数 $W \in \mathbb{R}^{m \times n}$，记当前梯度为 $G_t$。Adam 会近似维护一个和 $W$ 同形状的矩阵 $V_t$，其中每个元素都对应一个位置的平方梯度历史。这很贵，因为它要存满 $m \times n$ 个数。

Adafactor 的关键观察是：对于很多大矩阵，没必要把每个格子都精确记下来，记录“这一行整体梯度多大”和“这一列整体梯度多大”，往往已经足够提供一个可用的缩放尺度。于是它改成维护：

$$
r_t(i) = \beta_t r_{t-1}(i) + (1-\beta_t)\operatorname{mean}_j(G_t(i,j)^2)
$$

$$
c_t(j) = \beta_t c_{t-1}(j) + (1-\beta_t)\operatorname{mean}_i(G_t(i,j)^2)
$$

这里的 $r_t$ 是行统计量，白话说就是“每一行最近平均有多激烈”；$c_t$ 是列统计量，白话说就是“每一列最近平均有多激烈”。

然后用行因子和列因子重建一个近似二阶矩：

$$
\hat V_t(i,j) = \frac{r_t(i)\,c_t(j)}{\operatorname{mean}(r_t)}
$$

最后再用它去缩放当前梯度：

$$
U_t = \frac{G_t}{\sqrt{\hat V_t + \varepsilon_1}}
$$

很多实现还会做 update clipping，也就是更新裁剪。裁剪的意思是：如果本次更新整体过大，就统一压一下，防止尖峰更新把训练打崩：

$$
\bar U_t = \frac{U_t}{\max(1,\operatorname{RMS}(U_t)/d)}
$$

其中 $\operatorname{RMS}$ 是均方根，可以理解为“整体平均幅度”，$d$ 是裁剪阈值。最终参数更新通常写成：

$$
W_t = W_{t-1} - \alpha_t \bar U_t
$$

这里的 $\alpha_t$ 是学习率。有些 Adafactor 实现不是直接用固定学习率，而是再乘一个参数尺度，比如和 $\operatorname{RMS}(W)$ 有关，目的是让更新量更贴合参数本身量级。

先看一个玩具例子。假设：

$$
G=\begin{bmatrix}
2 & 1\\
0 & 3
\end{bmatrix},\quad
G^2=\begin{bmatrix}
4 & 1\\
0 & 9
\end{bmatrix}
$$

行均值是 $r=[2.5,\,4.5]$，列均值是 $c=[2,\,5]$，并且 $\operatorname{mean}(r)=3.5$。于是重建得到：

$$
\hat V \approx
\begin{bmatrix}
1.43 & 3.57\\
2.57 & 6.43
\end{bmatrix}
$$

这个结果和原始 $G^2$ 不一样。它不是在“恢复真值”，而是在“用低成本给出一个足够合理的缩放表”。这就是 Adafactor 的本质：牺牲精确度，换状态内存。

再看真实工程例子。训练 T5 这类 Transformer 时，最大的参数往往是词嵌入矩阵和前馈层矩阵。比如一个很大的 embedding 矩阵，形状可能是“词表大小 × 隐藏维度”。如果用 AdamW，除了参数本身，还要额外保存同样大小的优化器状态，整体开销很容易逼近甚至超过模型参数本身。Adafactor 把这类二维大矩阵的状态压成“行向量 + 列向量”，所以在 TPU 这种强调大批量、长时间稳定训练的场景里很有吸引力。

---

## 代码实现

真正实现 Adafactor 时，最容易出错的不是公式本身，而是“哪些参数做分解、哪些参数不做分解”。通常规则是：二维参数做 factored second moment，意思是“分解式二阶矩”；一维参数和标量参数一般退回完整二阶矩，因为它们没有自然的“行列结构”。

下面是一个可运行的最小 Python 版本，只演示核心思想，不包含偏置校正、权重衰减、完整学习率调度等工程细节：

```python
import math

def mean(xs):
    return sum(xs) / len(xs)

def rms_matrix(mat):
    flat = [x for row in mat for x in row]
    return math.sqrt(sum(x * x for x in flat) / len(flat))

def adafactor_factored_update(param, grad, beta2=0.9, eps=1e-30, clip_threshold=1.0, lr=0.1):
    m = len(param)
    n = len(param[0])
    assert m == len(grad) and n == len(grad[0])

    grad_sq = [[g * g for g in row] for row in grad]

    row_stat = [mean(row) for row in grad_sq]
    col_stat = [mean([grad_sq[i][j] for i in range(m)]) for j in range(n)]
    row_mean = mean(row_stat)

    v_hat = [
        [(row_stat[i] * col_stat[j]) / (row_mean + eps) for j in range(n)]
        for i in range(m)
    ]

    update = [
        [grad[i][j] / math.sqrt(v_hat[i][j] + eps) for j in range(n)]
        for i in range(m)
    ]

    update_rms = rms_matrix(update)
    scale = max(1.0, update_rms / clip_threshold)
    update = [[u / scale for u in row] for row in update]

    new_param = [
        [param[i][j] - lr * update[i][j] for j in range(n)]
        for i in range(m)
    ]
    return new_param, v_hat, update

param = [[1.0, 1.0], [1.0, 1.0]]
grad = [[2.0, 1.0], [0.0, 3.0]]

new_param, v_hat, update = adafactor_factored_update(param, grad)

assert len(v_hat) == 2 and len(v_hat[0]) == 2
assert v_hat[0][0] > 0 and v_hat[1][1] > 0
assert new_param[0][0] < param[0][0]
assert new_param[1][1] < param[1][1]
assert abs(update[1][0]) < 1e-12

print("v_hat =", v_hat)
print("new_param =", new_param)
```

这个例子里，`v_hat` 是重建后的近似二阶矩，`update` 是按它缩放并裁剪后的更新。你可以把它理解成一个“教学版本”。

实现时建议优先检查下面几件事：

| 实现点 | 说明 |
|---|---|
| 参数维度分支 | `2D` 常做分解，`1D`/标量常走完整状态 |
| `clip_threshold` | 用于抑制更新尖峰，不能随手删 |
| `eps` | 防止除零，不同库默认值不同 |
| 学习率设计 | 有的实现使用相对步长和参数缩放，不是裸 `lr` |

如果阅读框架源码，顺序建议是：先看状态结构，再看 factored / unfactored 分支，最后看学习率和 clipping 怎么组合。

---

## 工程权衡与常见坑

Adafactor 的工程优势很明确：省内存。它的工程代价也同样明确：近似更强，行为更依赖实现细节，调参经验不如 AdamW 通用。

最常见的误区是“把它当成一个低配版 Adam，超参照搬就行”。这通常不成立。原因在于 Adafactor 的有效行为不仅由基础公式决定，还受相对步长、参数尺度缩放、是否使用一阶矩、update clipping 等细节影响。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 直接照搬 AdamW 学习率 | 训练抖动或收敛变慢 | 按具体实现重新设定 `lr` 与 warmup |
| 忽略 update clipping | 个别 step 更新尖峰 | 保留合理 `clip_threshold` |
| 所有参数都强行做分解 | 维度不匹配或行为异常 | 仅对合适的二维参数使用 |
| 忽略框架实现差异 | 复现结果对不上 | 对照具体文档与源码 |
| 误以为更省内存就一定更好 | 小模型上收益不明显 | 先判断是否真有内存瓶颈 |

还有一个很实际的问题：不同实现并不完全一样。比如有些库支持一阶矩，有些默认关闭；有些库用 `mean`，有些在细节上等价于 `sum` 加归一化；有些实现对高维张量会按最后两维分解，有些则有专门规则。论文只给出思想，工程结果往往取决于实现。

所以在真实项目里，正确姿势不是背公式，而是先回答三个问题：

1. 你缺的是不是优化器状态内存？
2. 你的训练配方是否已经和某个实现深度绑定？
3. 你是否有时间重新验证超参与稳定性？

如果这三个问题里，前两个答案分别是“是”和“没有强绑定”，Adafactor 的收益通常比较明确。

---

## 替代方案与适用边界

把 Adafactor 放回优化器全景里看，它不是“最好”的优化器，而是“在状态内存受限时很有价值”的优化器。

| 优化器 | 状态开销 | 训练体验 | 典型适用场景 |
|---|---|---|---|
| SGD / Momentum | 低 | 简单，但自适应弱 | 经典视觉任务、强配方场景 |
| AdamW | 高 | 通用、稳定、生态成熟 | 大多数深度学习任务 |
| Adafactor | 低 | 需要更仔细配置 | 大模型、TPU、内存敏感训练 |

和 SGD 比，Adafactor 仍然保留了自适应缩放能力，所以在 Transformer 这类参数尺度差异很大的模型里更实用。和 AdamW 比，Adafactor 最大差异不是“更新方向完全不同”，而是“状态信息更粗糙”。这种粗糙会损失一部分精细度，但换来明显的内存收益。

因此它的适用边界可以概括为：

- 优先选 Adafactor：超大矩阵很多，TPU 训练，长序列预训练，优化器状态是硬瓶颈
- 优先选 AdamW：中小模型，常规 GPU 训练，更重视稳定复用现成经验
- 优先选 SGD：任务和配方已经长期验证，且不依赖 Adam 类自适应性质

对初级工程师，一个最稳妥的结论是：Adafactor 首先是“资源优化工具”，其次才是“优化算法选择”。如果你没有碰到状态内存问题，不必为了“看起来更高级”而先上它。

---

## 参考资料

1. [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
2. [PyTorch 官方文档 `torch.optim.Adafactor`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adafactor.html)
3. [T5X 源码 `t5x/adafactor.py`](https://raw.githubusercontent.com/google-research/t5x/main/t5x/adafactor.py)
4. [T5 代码仓库 README](https://github.com/google-research/text-to-text-transfer-transformer)
5. [T5 仓库讨论：T5 微调使用 Adafactor](https://github.com/google-research/text-to-text-transfer-transformer/issues/230)
