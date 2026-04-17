## 核心结论

AdamW、Adafactor、SOAP、Lion 都属于 Adam 生态。它们不改变“用一阶动量平滑梯度”的主线，而是在四个不同痛点上做修改：

| 优化器 | 主要改动 | 解决的核心问题 | 更适合的场景 |
|---|---|---|---|
| AdamW | 将 weight decay 与梯度更新解耦 | 正则项被 Adam 的自适应缩放污染 | 绝大多数训练与微调 |
| Adafactor | 用行/列因子近似二阶矩 | Adam 状态内存过大 | 大模型、显存/内存紧张 |
| SOAP | 把 Shampoo 的 Kronecker 预条件引入 Adam 框架 | 仅靠逐元素自适应无法利用矩阵曲率 | 大 batch 预训练 |
| Lion | 用 momentum 的符号而不是幅度更新 | 步长过大、状态和更新过重 | 微调、内存敏感训练 |

AdamW 最重要的结论是：在 Adam 中，L2 正则化不再等价于 weight decay。原因很简单，Adam 会按坐标对梯度做缩放，而如果你把 $\lambda w_t$ 直接并入梯度，正则项也会被同样缩放，含义就变了。AdamW 用下面的公式把两者重新分开：

$$
w_{t+1} = w_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda w_t
$$

其中 $\hat m_t,\hat v_t$ 是偏差校正后的一阶、二阶统计；偏差校正就是“把刚开始统计量偏小的问题补回来”。

玩具例子先看清这个区别。设当前参数 $w_t=1$，原始梯度 $g_t=2$，学习率 $\eta=10^{-3}$，$\lambda=0.1$，并假设此时 Adam 的二阶项满足 $\sqrt{\hat v_t}=2$。

如果把 L2 正则直接并进梯度，那么有效梯度变成 $2+0.1\times1=2.1$，更新量是：

$$
\eta\frac{2.1}{2}=1.05\times 10^{-3}
$$

这里的“衰减”被二阶统计一起缩放了。

而 AdamW 的流程是“先按原始梯度做 Adam，再单独做衰减”：

1. Adam 部分：$\eta\times \frac{2}{2}=1.0\times10^{-3}$
2. Weight decay 部分：$\eta\lambda w_t = 10^{-3}\times0.1\times1 = 1.0\times10^{-4}$

总更新是 $1.1\times10^{-3}$，并且第二部分的含义始终是“按当前参数值做线性收缩”，不会被自适应缩放污染。对新手来说，可以把它理解为：梯度负责“往损失更小的方向走”，weight decay 负责“把参数往 0 拉回去”，这两条力在 AdamW 里是分开的。

真实工程里，四种方法通常不是谁“绝对更先进”，而是谁更匹配约束：默认先用 AdamW；内存不够时考虑 Adafactor；大批量预训练、希望更强曲率信息时考虑 SOAP；微调或状态预算紧时考虑 Lion。

---

## 问题定义与边界

本文讨论的边界很明确：只看 Adam 这一类自适应优化器在 Transformer 训练中的变体，不展开 SGD、SAM、采样类优化器，也不讨论分布式系统实现细节。

问题来自三个常见瓶颈：

| 现象 | 真正瓶颈 | 对应方向 |
|---|---|---|
| 正则强度不稳定 | $\lambda w$ 混进梯度后被 Adam 缩放 | AdamW |
| 显存或优化器状态爆炸 | Adam 需要为每个参数存 $m,v$ 两份状态 | Adafactor |
| 大 batch 下收敛轮数偏多 | 逐元素二阶统计不够，缺少矩阵结构信息 | SOAP |
| 微调时步长过猛、震荡 | 更新幅度过于依赖数值大小 | Lion |

以一个简化的 Transformer 线性层 $W\in\mathbb{R}^{d_{out}\times d_{in}}$ 为例。Adam 会维护与 $W$ 同形状的一阶矩 $m_t$ 和二阶矩 $v_t$。如果我们做传统 L2 正则，会把 $\lambda W$ 加进梯度：

$$
g'_t = g_t + \lambda W_t
$$

然后再更新 $m_t,v_t$。问题是，$g_t$ 与 $\lambda W_t$ 的角色不同。前者来自当前 batch 的损失曲面，后者来自参数范数惩罚。它们被合并后，共同进入动量累计和逐坐标缩放，导致你已经很难说“$\lambda=0.01$ 到底意味着多强的衰减”。

在 large-batch 和 mixed precision 训练中，这个问题更明显。原因不是公式失效，而是数值范围、梯度裁剪、动态 loss scale 都会改变梯度分布；一旦正则项混进同一条路径，它的有效步长也会一起漂移。

第二个瓶颈是内存。Adam 对每个参数都要存：

- 参数本身
- 一阶矩 $m$
- 二阶矩 $v$

如果参数是 FP16/BF16，而状态常常还要用 FP32 存储，那么优化器状态的开销会显著高于模型权重本身。对大模型来说，这直接限制 batch size 和上下文长度。

第三个瓶颈是曲率利用不足。这里的“曲率”可以白话理解为“不同方向的损失面弯曲程度不同”。Adam 的二阶矩是逐元素的，它知道某个坐标最近波动大不大，但不知道一个矩阵中行和列之间的耦合结构。SOAP 试图把这部分结构信息拉回来。

---

## 核心机制与推导

### 1. AdamW：把正则从梯度路径拿出来

Adam 的标准统计是：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

偏差校正后：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

如果采用“L2 并入梯度”的写法，那么进入统计的是 $g_t+\lambda w_t$。这意味着：

- 一阶矩会记住正则项
- 二阶矩也会记住正则项
- 正则项会被 $\sqrt{\hat v_t}+\epsilon$ 缩放

因此它不再等价于参数层面的线性衰减。

AdamW 改成两步：

1. 用原始梯度 $g_t$ 更新 Adam 统计
2. 对参数直接做 $-\eta\lambda w_t$

所以它不是“数学小修补”，而是把两个控制旋钮拆开：学习率控制损失下降，weight decay 控制参数收缩。

### 2. Adafactor：用行列统计近似全量二阶矩

Adafactor 要解决的是：矩阵参数的二阶矩 $V_t$ 太大，没必要完整保存。设某层梯度矩阵是 $G_t\in \mathbb{R}^{m\times n}$，元素平方为 $G_t^2$。Adam 会存一个同形状矩阵 $V_t$，而 Adafactor 只存行和列两个累加器：

$$
R_t = \beta_2 R_{t-1} + (1-\beta_2)(G_t^2 \mathbf{1}_n)
$$

$$
C_t = \beta_2 C_{t-1} + (1-\beta_2)(\mathbf{1}_m^\top G_t^2)
$$

这里：

- $R_t$ 是每一行平方梯度的累积
- $C_t$ 是每一列平方梯度的累积

它们可以组合出一个近似的二阶矩：

$$
\hat V_t \approx \frac{R_t C_t}{\mathbf{1}_m^\top R_t \, C_t \mathbf{1}_n}
$$

直观理解是：如果一个矩阵的变化大致可分解为“行强度 × 列强度”，那就没必要保存每个元素的独立历史，只记两组边缘统计即可。

玩具例子：设某层梯度平方矩阵是

$$
G_t^2=
\begin{bmatrix}
4 & 16 \\
1 & 4
\end{bmatrix}
$$

那么行和列和分别是：

$$
R_t \propto
\begin{bmatrix}
20\\
5
\end{bmatrix},\qquad
C_t \propto
\begin{bmatrix}
5 & 20
\end{bmatrix}
$$

两者外积后得到：

$$
R_t C_t \propto
\begin{bmatrix}
100 & 400\\
25 & 100
\end{bmatrix}
$$

归一化后，结构上仍能恢复“第二列更大、第一行更大”的趋势。它不是精确还原，但常常足够做自适应缩放。这就是“低秩因子估计”的核心。

Adafactor 还常配合两个工程技巧：

- update clipping：限制单步更新范数，防止早期爆炸
- $\beta_2$ warmup：让二阶统计从不稳定阶段逐步进入平稳阶段

### 3. SOAP：先旋转到更合适的坐标系，再做 Adam

SOAP 可以理解为“把 Shampoo 的矩阵预条件思想嫁接到 Adam 上”。Shampoo 的核心思想是：对矩阵参数，不只按元素缩放，而是估计左右两个方向上的结构统计。

对梯度矩阵 $G_t$，可构造左右二阶统计：

$$
L_t \approx \mathbb{E}[G_t G_t^\top],\qquad
R_t \approx \mathbb{E}[G_t^\top G_t]
$$

再做特征分解：

$$
L_t = Q_L \Lambda_L Q_L^\top,\qquad
R_t = Q_R \Lambda_R Q_R^\top
$$

这里 $Q_L,Q_R$ 可以理解为“更适合看这个矩阵变化的坐标轴”。SOAP 不直接在原空间里做 Adam，而是先把梯度旋转到该特征空间：

$$
\tilde G_t = Q_L^\top G_t Q_R
$$

然后在 $\tilde G_t$ 上做 Adam 式的一阶、二阶统计和更新，最后再旋回原空间。它的优势是：在新坐标系里，不同方向更接近解耦，逐元素 Adam 的效果更接近真正的二阶预条件。

可以把流程写成一个步骤序列：

| 步骤 | 操作 | 作用 |
|---|---|---|
| 1 | 累积 $L_t,R_t$ | 估计矩阵左右方向结构 |
| 2 | 周期性分解得到 $Q_L,Q_R$ | 找到更好的旋转基 |
| 3 | 计算 $\tilde G_t=Q_L^\top G_t Q_R$ | 转到特征空间 |
| 4 | 在 $\tilde G_t$ 上执行 Adam | 保留 Adam 的稳定性 |
| 5 | 把更新旋回原空间 | 应用于原参数 |

它比 AdamW 更贵，因为特征分解和 Kronecker 结构更新都要算；但在大 batch 预训练下，这类额外开销有时能换来更少训练步数。

### 4. Lion：只用方向，不太关心幅度

Lion 的名字常被理解为“只保留动量，再用 sign 更新”。这里 sign 就是“只看正负号，不看具体大小”。

一个简化写法是：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$

$$
u_t=\beta_2 m_t+(1-\beta_2)g_t
$$

$$
w_{t+1}=w_t-\eta\cdot \mathrm{sign}(u_t)
$$

其中 $\mathrm{sign}(\cdot)$ 返回每个元素的符号。白话解释是：Lion 不是问“这一步走多远最合适”，而是先问“该往哪边走”，然后统一给出步长 $\eta$。

这带来两个效果：

- 状态更省，不需要二阶矩
- 更新更硬，容易得到更强的隐式正则

但缺点也直接：因为幅度信息被抹平，学习率往往要比 AdamW 更小，否则会过冲。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它不是生产级优化器，只用于展示四类更新的核心差异。

```python
import math

def adamw_step(w, g, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, wd=0.1, eps=1e-8):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g * g)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    # 先 Adam 更新，再独立做 weight decay
    w = w - lr * m_hat / (math.sqrt(v_hat) + eps) - lr * wd * w
    return w, m, v

def adafactor_step_matrix(G, R, C, beta2=0.999):
    rows = len(G)
    cols = len(G[0])
    row_sums = [sum(x * x for x in row) for row in G]
    col_sums = [sum(G[i][j] * G[i][j] for i in range(rows)) for j in range(cols)]

    R = [beta2 * r + (1 - beta2) * rs for r, rs in zip(R, row_sums)]
    C = [beta2 * c + (1 - beta2) * cs for c, cs in zip(C, col_sums)]

    denom = sum(R) * sum(C)
    V = [[(R[i] * C[j]) / denom if denom > 0 else 0.0 for j in range(cols)] for i in range(rows)]
    return R, C, V

def lion_step(w, g, m, lr=1e-4, beta1=0.9, beta2=0.99, wd=0.1):
    # 常见工程实现也会配 decoupled weight decay
    update = beta2 * m + (1 - beta2) * g
    sign = 1.0 if update > 0 else (-1.0 if update < 0 else 0.0)
    w = w * (1 - lr * wd) - lr * sign
    m = beta1 * m + (1 - beta1) * g
    return w, m

# AdamW 玩具例子
w, g, m, v = 1.0, 2.0, 0.0, 4.0
w2, m2, v2 = adamw_step(w, g, m, v, t=1, lr=1e-3, wd=0.1)
assert w2 < w
assert m2 > 0 and v2 > 0

# Adafactor 只存行/列累加器
G = [[2.0, 4.0], [1.0, 2.0]]
R, C, V = adafactor_step_matrix(G, [0.0, 0.0], [0.0, 0.0], beta2=0.0)
assert len(R) == 2 and len(C) == 2
assert V[0][1] > V[1][0]  # 第二列更大、第一行更大

# Lion 只看方向
w3, m3 = lion_step(1.0, 0.5, 0.0, lr=1e-4)
assert w3 < 1.0
assert m3 > 0
```

如果写成统一伪代码，可以更容易看差异：

```python
if optimizer == "adamw":
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g * g)
    w = w - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * w

elif optimizer == "adafactor":
    # 对矩阵参数，只保留行/列二阶累加器
    R = update_row_accumulator(G)
    C = update_col_accumulator(G)
    V = factorized_second_moment(R, C)
    U = clip(G / sqrt(V + eps))
    w = w - lr * U

elif optimizer == "soap":
    # 周期性更新 Kronecker 特征基，再在旋转空间里做 Adam
    QL, QR = update_eigen_bases_if_needed(G)
    G_tilde = transpose(QL) @ G @ QR
    m, v = adam_stats(G_tilde)
    U_tilde = m_hat / (sqrt(v_hat) + eps)
    w = w - lr * (QL @ U_tilde @ transpose(QR))

elif optimizer == "lion":
    # 最终更新只用 sign，幅度信息不直接进入步长
    u = beta2 * m + (1 - beta2) * g
    w = w * (1 - lr * wd) - lr * sign(u)
    m = beta1 * m + (1 - beta1) * g
```

实现时有几个细节几乎总会遇到：

- AdamW 和 Lion 通常都应把 `bias`、`LayerNorm.weight`、有时还包括 `embedding norm` 排除出 weight decay。
- Adafactor 常见稳定化配置包括 `clip_threshold`、相对步长、以及随时间变化的二阶衰减。
- SOAP 不会每一步都更新特征分解，通常按固定频率更新，否则代价过高。
- Lion 常需要比 AdamW 小 3 到 10 倍的学习率，同时适当增大 weight decay。

---

## 工程权衡与常见坑

真实工程例子最典型的是大规模 Transformer 训练。T5、PaLM 这一类模型参数矩阵很大，优化器状态经常比前向激活更先成为瓶颈。Adafactor 的价值就在这里：如果某层是矩阵参数，它只需要保存行、列两个累加器，而不是完整的二阶矩矩阵，所以在 TPU 或显存紧张场景中，可以显著降低状态开销，还能保留 full-precision checkpoint。

SOAP 的真实价值则出现在大 batch 预训练。比如百万级 token 每步的训练中，单纯 AdamW 有时需要更多步才能把同样的数据预算转成有效损失下降；SOAP 通过周期性引入 Kronecker 结构信息，常能提高样本效率。但它不是白送收益，你要为特征分解和更复杂的调度买单。

Lion 的工程位置更像“有意识的偏置选择”。它不是通用替代 AdamW 的默认答案，而是在显存敏感、微调、或者希望更强正则时值得尝试。因为 sign 更新会放大有效步长，所以最常见的问题不是“不收敛”，而是“震荡”。

| 问题/坑 | 为什么会发生 | 规避方法 |
|---|---|---|
| bias 和 LayerNorm 也被 weight decay | 框架参数分组默认不细分 | 手动参数分组排除 |
| 以为 Adam 中 L2 等于 weight decay | 只在 SGD 类更新里才近似等价 | 在 Adam 系列中使用 AdamW 式 decoupled decay |
| Adafactor 早期训练不稳 | 因子近似 + 大步长 + 二阶统计未稳定 | 开启 update clipping、$\beta_2$ warmup、相对步长 |
| SOAP 开销过大 | 特征分解频繁，矩阵大 | 降低 basis 更新频率，和 batch size 联动调参 |
| Lion 直接沿用 AdamW 学习率 | sign 更新的有效步长更大 | 学习率先降 3 到 10 倍，再重新搜 weight decay |
| 只看单步 loss 判断优化器优劣 | 不同优化器早期动态差异很大 | 用固定 token budget 或 wall-clock 做对比 |

一个很常见的误区是：看到 Adafactor 省内存，就直接拿来替换所有 AdamW。这个决策往往过早。因为当模型并不大、单机 GPU 也放得下时，AdamW 的调参经验最成熟，收益最高的通常不是“换优化器”，而是把学习率、warmup、梯度裁剪、weight decay 分组先调对。

---

## 替代方案与适用边界

如果没有明确瓶颈，AdamW 仍然是默认选择。原因不是它最先进，而是它的行为最可解释、最稳定、经验最丰富。

| 场景 | AdamW | Adafactor | SOAP | Lion |
|---|---|---|---|---|
| 小模型、单机 GPU | 很适合 | 一般不必优先 | 通常不值当 | 可试，但不是默认 |
| 大模型、内存紧 | 可用但状态重 | 最适合 | 可选 | 可选 |
| 超大 batch 预训练 | 可用 | 可用 | 更有潜力 | 需谨慎调参 |
| 曲率结构重要的矩阵层 | 一般 | 一般 | 更适合 | 不擅长 |
| 微调任务 | 默认首选 | 较少使用 | 过重 | 常值得尝试 |
| 调参成本敏感 | 最低 | 中等 | 最高 | 中等 |

可以把选择规则压缩成三句：

1. 小模型、普通训练、单机 GPU：优先 AdamW。
2. 大规模 Transformer、状态内存是硬约束：优先考虑 Adafactor。
3. 超大 batch 预训练且愿意承担更复杂实现与调参：再考虑 SOAP。
4. 微调、状态预算紧、接受更小学习率搜索：可以尝试 Lion。

还要强调一个边界：这些变体都不是“换了就更快收敛”。优化器的收益高度依赖模型结构、batch size、序列长度、学习率计划和梯度裁剪。尤其在中小规模任务中，训练稳定性的主要决定因素常常还是数据、初始化和 lr schedule，而不是优化器名字本身。

---

## 参考资料

1. Loshchilov, I., Hutter, F. “Decoupled Weight Decay Regularization”, 2019。AdamW 原始论文，核心贡献是证明在 Adam 中应将 weight decay 与梯度更新解耦。
2. Shazeer, N., Stern, M. “Adafactor: Adaptive Learning Rates with Sublinear Memory Cost”, 2018。Adafactor 原始论文，核心贡献是用行列因子近似二阶矩，显著降低状态内存。
3. “SOAP: Improving and Stabilizing Shampoo using Adam”, 2024。SOAP 论文，核心贡献是把 Shampoo 的 Kronecker 预条件与 Adam 的稳定更新结合起来。
4. Chen, X. et al. “Symbolic Discovery of Optimization Algorithms”, 2023。Lion 来源论文，核心贡献是提出基于 momentum 符号的更新规则。
5. TensorTonic 关于 adaptive optimizers 的解析文章。适合从工程视角理解 AdamW、Adafactor、SOAP、Lion 的差异与适用场景。
6. Emergent Mind 对 Adafactor、SOAP、Lion 论文的整理页。适合快速查看论文摘要、实验结论与相关引用。
7. TensorFlow / 相关框架中的 Lion 工程文档。适合查默认超参数、参数分组和实际实现细节。
