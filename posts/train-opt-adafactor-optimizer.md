## 核心结论

Adafactor 是一种**自适应优化器**，也就是会按参数和梯度的统计量自动调整更新步长的优化算法。它解决的核心问题不是“收敛更快”，而是“**在大模型训练里，把优化器状态占用的显存压下来**”。

它和 Adam/AdamW 的本质区别有三点：

1. Adam 会给每个参数保存完整的一阶矩和二阶矩，Adafactor 默认**不保存一阶矩**，直接省掉一部分状态。
2. 对于矩阵参数，Adam 要保存和参数同形状的二阶矩 $v_t$；Adafactor 只保存**行统计**和**列统计**，再把它们组合成对 $v_t$ 的近似。
3. 它常用**相对学习率**，也就是学习率不直接手工指定，而是和当前参数尺度绑定：
   $$
   \alpha_t=\max(\epsilon_2,\mathrm{RMS}(\theta_{t-1}))\cdot \rho_t
   $$

结论可以直接记成一句话：**Adafactor 用“少存状态”换“更低显存”，在超大矩阵参数很多的模型里尤其划算。**

对于形状为 $m \times n$ 的矩阵参数，Adam 保存二阶矩需要 $O(mn)$ 空间，Adafactor 近似后只需 $O(m+n)$。这不是常数级优化，而是量级变化。Transformer 里的线性层、注意力投影层、Embedding 都有大量矩阵，因此收益明显。

---

## 问题定义与边界

先定义问题。训练神经网络时，参数更新不是简单做
$$
\theta_t=\theta_{t-1}-\eta g_t
$$
因为不同参数的梯度尺度差异很大。Adam 这类算法会维护梯度的历史统计，让每个参数按自己的尺度更新，更稳定。但代价是显存占用高。

如果参数总量是 $N$，Adam/AdamW 通常要额外保存：

| 优化器 | 主要状态 | 额外状态量级 | 常见每参数状态字节 |
|---|---|---:|---:|
| SGD | 无或仅动量 | $O(N)$ 或更少 | 0B 或 4B |
| AdamW | 一阶矩 + 二阶矩 | $O(N)$ | 8B |
| Adafactor | 仅二阶矩近似，且矩阵做分解 | 向量级或 $O(N)$ 混合 | 约 4B+ |
| 8-bit Adam | 一阶矩 + 二阶矩，但量化存储 | $O(N)$ | 约 2B |

这里的“二阶矩”可以白话理解成：**梯度平方的滑动平均，用来估计某个参数最近波动有多大**。波动越大，更新通常越要保守。

Adafactor 的边界也要说清楚：

- 它主要对**矩阵参数**有明显收益，因为矩阵的二阶矩才可以做行列分解。
- 对偏置、LayerNorm 标量、小向量参数，通常仍按普通向量方式保存统计，省不了太多。
- 它的目标是**内存效率优先**，不是绝对收敛速度优先。
- 在梯度噪声很大、训练很不稳定时，没有一阶矩会让它更依赖 clip 和 warmup。

一个玩具例子可以帮助建立直觉。假设某层权重是一个 $3\times3$ 矩阵。Adam 会给 9 个元素都保存独立的二阶矩。Adafactor 不这么做，它只记：

- 每一行总体有多“抖”
- 每一列总体有多“抖”

也就是从“记录 9 个格子的历史”变成“记录 3 个行数值 + 3 个列数值”，然后再估算每个格子的尺度。

如果矩阵很小，这个优化没意义；如果矩阵是 $4096\times4096$，意义就非常大。

---

## 核心机制与推导

设某个矩阵参数为 $W\in\mathbb{R}^{m\times n}$，当前梯度为 $G_t$。传统 Adam 会维护与 $W$ 同形状的二阶矩：
$$
V_t=\beta_{2,t}V_{t-1}+(1-\beta_{2,t})(G_t\odot G_t)
$$

Adafactor 的关键改写是：不直接保存 $V_t$，而是只保存它的行均值和列均值。记
$$
R_t(i)=\beta_{2,t}R_{t-1}(i)+(1-\beta_{2,t})\frac{1}{n}\sum_{j=1}^n G_t(i,j)^2
$$
$$
C_t(j)=\beta_{2,t}C_{t-1}(j)+(1-\beta_{2,t})\frac{1}{m}\sum_{i=1}^m G_t(i,j)^2
$$

然后用它们重建每个位置的近似二阶矩：
$$
\hat V_t(i,j)\approx \frac{R_t(i)\,C_t(j)}{\frac{1}{m}\sum_{i=1}^m R_t(i)}
$$

这一步里的“近似”可以白话理解成：**假设某个元素的波动大小，主要由它所在行和所在列共同决定。**

接着做自适应缩放：
$$
U_t(i,j)=\frac{G_t(i,j)}{\sqrt{\hat V_t(i,j)}+\epsilon_1}
$$

再做更新裁剪。这里的“裁剪”意思是：**如果整次更新过大，就按比例整体缩小，避免突然跳飞**。
$$
\bar U_t=\frac{U_t}{\max\left(1,\frac{\mathrm{RMS}(U_t)}{d}\right)}
$$

最后不是直接乘固定学习率，而是乘相对学习率：
$$
\alpha_t=\max(\epsilon_2,\mathrm{RMS}(\theta_{t-1}))\cdot \rho_t
$$
其中常见设定是
$$
\rho_t=\min(10^{-2},1/\sqrt{t})
$$

于是更新式是：
$$
\theta_t=\theta_{t-1}-\alpha_t\bar U_t
$$

这个流程有两个后果：

- 参数本身数值大，允许的绝对步长也会更大。
- 参数本身数值小，步长也会自动缩小，减少尺度不匹配。

### 玩具例子

设一个 $3\times3$ 梯度矩阵为
$$
G=
\begin{bmatrix}
1 & 2 & 1\\
2 & 4 & 2\\
1 & 2 & 1
\end{bmatrix}
$$

它的平方为
$$
G^2=
\begin{bmatrix}
1 & 4 & 1\\
4 & 16 & 4\\
1 & 4 & 1
\end{bmatrix}
$$

行均值是 $[2,8,2]$，列均值也是 $[2,8,2]$。这说明中间那一行和中间那一列波动最大，因此中心元素的二阶矩估计也最大。二阶矩越大，分母越大，对应位置更新越保守。这正是自适应优化器要做的事。

---

## 代码实现

下面给一个可运行的 Python 版本，只演示 Adafactor 对矩阵参数的核心更新逻辑。代码不是工业实现，但足够对应上面的公式。

```python
import math

def rms_matrix(x):
    total = sum(v * v for row in x for v in row)
    count = len(x) * len(x[0])
    return math.sqrt(total / count)

def row_mean_square(g):
    return [sum(v * v for v in row) / len(row) for row in g]

def col_mean_square(g):
    m, n = len(g), len(g[0])
    return [sum(g[i][j] * g[i][j] for i in range(m)) / m for j in range(n)]

def adafactor_step(param, grad, beta2=0.0, eps1=1e-30, eps2=1e-3, d=1.0, rho_t=1e-2):
    m, n = len(param), len(param[0])
    r = row_mean_square(grad)
    c = col_mean_square(grad)

    mean_r = sum(r) / len(r)
    vhat = [[(r[i] * c[j]) / mean_r for j in range(n)] for i in range(m)]

    u = [[grad[i][j] / (math.sqrt(vhat[i][j]) + eps1) for j in range(n)] for i in range(m)]
    u_rms = rms_matrix(u)
    scale = max(1.0, u_rms / d)
    u_bar = [[u[i][j] / scale for j in range(n)] for i in range(m)]

    alpha_t = max(eps2, rms_matrix(param)) * rho_t
    new_param = [[param[i][j] - alpha_t * u_bar[i][j] for j in range(n)] for i in range(m)]
    return new_param, r, c, vhat, alpha_t

param = [
    [1.0, 0.5, -0.5],
    [0.2, -1.0, 0.3],
    [0.1, 0.4, -0.2],
]

grad = [
    [1.0, 2.0, 1.0],
    [2.0, 4.0, 2.0],
    [1.0, 2.0, 1.0],
]

new_param, r, c, vhat, alpha_t = adafactor_step(param, grad)

assert len(r) == 3
assert len(c) == 3
assert round(r[1], 6) == 8.0
assert round(c[1], 6) == 8.0
assert alpha_t > 0
assert new_param[1][1] != param[1][1]
```

这段代码体现了四个关键点：

1. 不保存完整矩阵二阶矩，只算 `r` 和 `c`。
2. 用 `vhat` 近似恢复每个元素的尺度。
3. 用 `u_bar` 做 clip。
4. 用参数 RMS 计算实际步长 `alpha_t`。

真实工程里通常直接使用框架实现，而不是手写。以 PyTorch 或 Transformers 为例，Adafactor 常见配置是开启 `relative_step` 和 `scale_parameter`。这意味着你传进去的 `lr` 更像“上限”或调度因子，而不是 AdamW 里那种固定全局学习率。

### 真实工程例子

在 T5-3B、T5-11B 这类大模型微调里，显存经常不是耗在前向本身，而是耗在优化器状态上。假设模型参数量是 30 亿：

- AdamW 需要完整一阶矩和二阶矩，状态大致按 8B/参数估算，仅状态就约 24GB。
- Adafactor 不保存一阶矩，矩阵二阶矩又改成行列统计，状态可降到略高于 4B/参数，约 12GB 左右。

这节省出来的显存通常有三种用途：

| 节省的显存可以换成什么 | 直接收益 |
|---|---|
| 更大的 batch size | 梯度估计更稳定，吞吐更高 |
| 更长的序列长度 | 适合长文本任务 |
| 更多 checkpoint 或激活缓存 | 降低 OOM 风险，提高训练灵活性 |

所以在超大模型上，Adafactor 经常不是“更优雅的优化器”，而是“**否则根本放不下**”时的现实方案。

---

## 工程权衡与常见坑

Adafactor 的优势很明确，但工程上不能只看省显存。

第一，**没有一阶矩**。一阶矩可以理解成“梯度方向的惯性”，也就是常说的动量。没有它，优化器对噪声更敏感，很多任务上收敛会比 AdamW 稍慢，或者需要更谨慎的 warmup。

第二，**相对学习率可能不稳定**。因为
$$
\alpha_t=\max(\epsilon_2,\mathrm{RMS}(\theta_{t-1}))\rho_t
$$
把参数 RMS 直接带进了步长。如果参数尺度本身变化快，或者梯度方差很大，实际更新可能突然放大。这也是为什么工程上常配：

- `clip_threshold ≈ 1.0`
- warmup
- 动态 $\beta_{2,t}$，例如 $1-\frac{1}{t^c}$

第三，**不是所有参数都同样受益**。Embedding 和大线性层收益明显，但很多小向量参数不明显。因此整个模型的总节省量，取决于“矩阵参数占比”而不是只看总参数量。

第四，**不要把它当成 AdamW 的平替去无脑换**。如果你的训练资源充足，目标是最稳妥的收敛，AdamW 往往更省心。Adafactor 更像“显存受限下的折中最优”。

常见坑可以直接列成表：

| 坑 | 现象 | 处理方式 |
|---|---|---|
| 把 `lr` 当普通固定学习率调 | 训练行为和预期不一致 | 明确是否开启 `relative_step` |
| 没开 clip | 某些 step 更新突然爆大 | 设 `clip_threshold≈1.0` |
| 直接照搬 AdamW 超参 | 前期震荡或收敛慢 | 加 warmup，重新看 decay 策略 |
| 任务本身噪声大 | loss 抖动明显 | 优先保留更稳的 AdamW 或改用 8-bit Adam |
| 以为所有层都能省很多 | 实际省显存不如预期 | 先分析模型里大矩阵占比 |

---

## 替代方案与适用边界

如果只看“省显存”，Adafactor 不是唯一方案。

**AdamW** 适合资源充足、想要稳定收敛的情况。它保留完整动量和二阶矩，理论和实践都最成熟，缺点就是状态重。

**8-bit Adam** 适合“我还想保留 Adam 的行为，但状态得压缩”。它不是做矩阵分解，而是把状态量化成低精度，所以仍然保留完整结构，只是存得更省。

**Adafactor** 适合“矩阵参数很多、显存非常紧、可以接受一点收敛代价”的场景。典型就是大语言模型或 T5 家族微调。

可以用一句话判断：

| 方案 | 更适合什么情况 |
|---|---|
| AdamW | 资源够，稳定性优先 |
| Adafactor | 显存瓶颈明显，矩阵参数占比高 |
| 8-bit Adam | 想保留 Adam 行为，同时压缩状态 |

再给一个实际决策规则：

- 单卡训练已经接近 OOM，先看 Adafactor。
- 训练不稳定但还想省内存，先试 8-bit Adam。
- 训练预算够，优先 AdamW。
- 如果你已经用了 gradient checkpointing、混合精度、冻结部分层，但还是放不下，Adafactor 的优先级会明显上升。

它的适用边界也要记住：**Adafactor 擅长解决“内存问题”，不保证解决“优化问题”。**

---

## 参考资料

- Noam Shazeer, Mitchell Stern. *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*. ICML 2018. https://proceedings.mlr.press/v80/shazeer18a.html
- Cornell Optimization Wiki. *Adafactor*. https://optimization.cbe.cornell.edu/index.php?title=Adafactor
- Hugging Face Documentation. *Efficient Training on a Single GPU*. https://huggingface.co/docs/transformers/v4.35.2/en/perf_train_gpu_one
- PyTorch Documentation. *torch.optim.Adafactor*. https://docs.pytorch.org/docs/stable/generated/torch.optim.Adafactor.html
- Emergent Mind. *Adafactor Algorithm*. https://www.emergentmind.com/topics/adafactor-algorithm
