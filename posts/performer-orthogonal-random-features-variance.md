## 核心结论

Performer 的 FAVOR+ 本质上是在近似 softmax attention 对应的核函数，而不是改写 attention 的定义。它利用
$$
\exp(q^\top k)=\mathbb E\big[\phi^+(q)^\top \phi^+(k)\big]
$$
把原本需要显式构造的 $L\times L$ 注意力矩阵，改写成几次线性矩阵乘法。因此，单头 attention 的主计算量可以从
$$
O(L^2d)
$$
降到
$$
O(Lmd),
$$
其中 $L$ 是序列长度，$d$ 是每个头的特征维度，$m$ 是随机特征数。对应的空间复杂度也从
$$
O(L^2+Ld)
$$
降到
$$
O(Lm+Ld+md).
$$

FAVOR+ 的两个关键点不能拆开理解：

1. `positive random features`：正值随机特征。特征映射后的每一维都非负，这保证 attention 分母始终为正，归一化更稳定。
2. `orthogonal random features`：正交随机特征。把原本独立采样的随机方向改成尽量互相正交，从而减少重复方向带来的采样噪声。

这里必须纠正一个常见误读：论文严格证明的是“正交化会降低核估计的均方误差，并给出更强的大偏差尾概率界”，而不是一句模糊的“方差从 $O(1/m)$ 直接变成 $O(e^{-m/d})$”。更准确的说法如下。

| 对象 | i.i.d. 高斯特征 | 正交特征 |
|---|---:|---:|
| 单点估计的基础误差阶 | 一般随 $m$ 以 $O(1/m)$ 下降 | 仍随 $m$ 下降，但常数更低，MSE 有显式改进 |
| 大偏差尾概率 | 界更松 | 论文给出更强的指数型上界 |
| 同样 $m$ 下的工程误差 | 往往更大 | 往往明显更小 |
| 极端坏样本出现概率 | 更高 | 更低 |

如果用一句不失真的白话概括：普通 softmax attention 是“所有 query 和 key 两两打分”；FAVOR+ 是“先把每个 token 压到一个随机特征空间，再利用矩阵乘法结合律，避免构造整张 $L\times L$ 打分表”。

真实工程里，这种改写最有价值的场景是长序列。原论文在 $L=4096,d=16$ 的实验中展示：当随机特征数 $m=64$ 时，正交特征的近似误差已经明显低于 i.i.d. 特征；继续增大 $m$，收益会逐渐逼近 float32 的数值精度上限，而不是无限线性改善。也就是说，随机特征数不是越大越好，而是存在“统计误差已经很小，数值误差开始主导”的拐点。

---

## 问题定义与边界

标准 self-attention 的核心开销来自注意力矩阵
$$
A=\exp(QK^\top),
$$
其中 $Q,K\in\mathbb R^{L\times d}$，所以 $A\in\mathbb R^{L\times L}$。这里的 $L$ 是序列长度，也就是 token 数。一旦 $L$ 变大，时间和显存都会按平方增长。

以单头 attention 为例，标准写法是
$$
\mathrm{Att}(Q,K,V)=D^{-1}\exp(QK^\top)V,
$$
其中
$$
D=\operatorname{diag}\big(\exp(QK^\top)\mathbf 1\big).
$$
这表示：先为每个 query 和所有 key 计算相似度，再对每一行做归一化，最后加权求和得到输出。

对应的复杂度对比如下。

| 方法 | 主要计算 | 时间复杂度 | 空间复杂度 |
|---|---|---|---|
| 标准 softmax attention | 显式算 $QK^\top$ 和 $AV$ | $O(L^2d)$ | $O(L^2+Ld)$ |
| FAVOR+ | 先算 $Q'=\phi(Q),K'=\phi(K)$，再做线性乘法 | $O(Lmd)$ | $O(Lm+Ld+md)$ |

这里的边界条件必须说清楚，否则很容易把 FAVOR+ 理解错：

1. FAVOR+ 近似的是 softmax kernel，不是任意 attention 机制。
2. 它追求的是“核函数估计正确且复杂度线性化”，不是逐项精确恢复每个 attention 权重。
3. 当 $m$ 太小、输入范数太大、随机矩阵抽样不稳定时，核估计误差会沿着整层传播。
4. 论文对 ORF 的严格结果先在单块 $m\le d$ 的场景下给出；工程上若 $m>d$，通常按 block 构造，即每 $d$ 个方向组成一块、块内正交、块间独立。

很多新手第一次读 Performer 会卡在一个概念上：它到底是在“近似 softmax”，还是在“换掉 softmax”？答案是前者。  
更准确地说，softmax attention 里真正麻烦的对象是
$$
\exp(q_i^\top k_j),
$$
而 FAVOR+ 做的是把这个量看成一个核函数值，再用显式随机特征去近似它。于是“先两两算分数”变成了“先映射，再内积”。

一个非常小的例子能帮助建立直觉。假设只有 3 个 token：

| 做法 | 需要显式计算的量 |
|---|---|
| 标准 attention | 9 个两两分数：$q_i^\top k_j$ |
| FAVOR+ | 3 个 query 特征、3 个 key 特征，再做聚合 |

当 $L=3$ 时差别不大；当 $L=4096$ 时，这个改写才真正体现价值。

---

## 核心机制与推导

softmax kernel 定义为
$$
\mathrm{SM}(x,y)=\exp(x^\top y).
$$
Performer 的关键想法是：如果能构造一个随机映射 $\phi(x)\in\mathbb R^m$，使得
$$
\mathbb E[\phi(x)^\top\phi(y)]=\exp(x^\top y),
$$
那么就可以用
$$
\phi(q_i)^\top\phi(k_j)
$$
来近似原来的 softmax 打分。

### 1. 正值随机特征的定义

Performer 使用的正值随机特征为
$$
\phi^+(x)=\frac{1}{\sqrt m}\exp\!\left(-\frac{\|x\|^2}{2}\right)
\begin{bmatrix}
\exp(\omega_1^\top x)\\
\vdots\\
\exp(\omega_m^\top x)
\end{bmatrix},
\qquad \omega_i\sim \mathcal N(0,I_d).
$$

这个式子包含三层信息：

| 部分 | 作用 |
|---|---|
| $\omega_i^\top x$ | 沿随机方向投影 |
| $\exp(\omega_i^\top x)$ | 把投影送入指数空间，匹配 softmax kernel 的指数形式 |
| $\exp(-\|x\|^2/2)$ | 做配平，使期望恰好还原 $\exp(x^\top y)$ |

### 2. 为什么它的期望等于 softmax kernel

对任意 $\omega\sim\mathcal N(0,I_d)$，高斯矩母函数告诉我们
$$
\mathbb E[\exp(\omega^\top u)] = \exp\!\left(\frac{\|u\|^2}{2}\right).
$$
令 $u=x+y$，则
$$
\mathbb E[\exp(\omega^\top x)\exp(\omega^\top y)]
=
\mathbb E[\exp(\omega^\top(x+y))]
=
\exp\!\left(\frac{\|x+y\|^2}{2}\right).
$$
再乘上两个前置衰减因子：
$$
\exp\!\left(-\frac{\|x\|^2}{2}\right)
\exp\!\left(-\frac{\|y\|^2}{2}\right)
\exp\!\left(\frac{\|x+y\|^2}{2}\right)
=
\exp(x^\top y).
$$
因此
$$
\mathbb E[\phi^+(x)^\top \phi^+(y)] = \exp(x^\top y).
$$

这一步是全文最关键的数学点。它说明 FAVOR+ 不是经验性技巧，而是一个有严格期望匹配的核估计器。  
新手可以把它理解成：虽然我们没有直接算 $\exp(x^\top y)$，但平均意义下，我们算出来的东西就是它。

### 3. attention 如何从二次改写成线性

设
$$
Q'=\phi(Q)\in\mathbb R^{L\times m},\qquad K'=\phi(K)\in\mathbb R^{L\times m}.
$$
则 attention 输出可近似写成
$$
\widehat{\mathrm{Att}}(Q,K,V)
=
\operatorname{diag}\!\big(Q'(K'^\top \mathbf 1)\big)^{-1}
Q'(K'^\top V).
$$

把这个式子拆开看更直观。对第 $i$ 个 query，
$$
\hat o_i
=
\frac{\sum_{j=1}^L \big(\phi(q_i)^\top\phi(k_j)\big)v_j}
{\sum_{j=1}^L \phi(q_i)^\top\phi(k_j)}.
$$
再利用内积和求和可交换：
$$
\hat o_i
=
\frac{\phi(q_i)^\top \left(\sum_{j=1}^L \phi(k_j)v_j^\top\right)}
{\phi(q_i)^\top \left(\sum_{j=1}^L \phi(k_j)\right)}.
$$

于是可以先把所有 key 和 value 聚合：
$$
S=\sum_{j=1}^L \phi(k_j)v_j^\top\in\mathbb R^{m\times d_v},
\qquad
r=\sum_{j=1}^L \phi(k_j)\in\mathbb R^m.
$$
然后每个 query 只要再做两次内积：
$$
\hat o_i=\frac{\phi(q_i)^\top S}{\phi(q_i)^\top r}.
$$

复杂度下降的根源就在这里：不再构造 $L\times L$ 的注意力矩阵，而是把“对所有 key 求和”提前聚合。

### 4. 为什么“正值”很重要

因为 $\phi^+(x)$ 的每一维都非负，所以
$$
z_i=\phi(q_i)^\top\sum_{j=1}^L\phi(k_j) > 0.
$$
这意味着归一化分母天然为正。  
如果用一般会出现负值的随机特征，那么分母可能发生严重抵消，甚至接近 0，训练时就容易数值爆炸。

这件事可以用一个简单对比理解：

| 特征类型 | 可能出现负值 | 分母稳定性 |
|---|---|---|
| 正值特征 $\phi^+$ | 否 | 更稳 |
| 一般三角特征 | 是 | 更容易出现抵消和异常放大 |

### 5. 为什么正交化能降低误差

如果 $\omega_1,\dots,\omega_m$ 是 i.i.d. 高斯采样，那么有限个方向里经常会出现“抽到很多相近方向”的情况。  
这不会破坏无偏性，但会浪费特征预算，因为多个方向在重复测量相近的信息。

正交化后的直觉是：既然总共只能抽 $m$ 个方向，就让它们尽量覆盖不同方向，而不是扎堆。

| 采样方式 | 方向分布 | 结果 |
|---|---|---|
| i.i.d. 高斯 | 可能重复、可能聚团 | 估计噪声更大 |
| ORF | 更分散、更均匀 | 同样 $m$ 下误差更小 |

论文中的两个理论结论需要分开记忆：

| 理论结果 | 严格含义 |
|---|---|
| Theorem 2 | ORF 对正值 softmax 特征的 MSE 给出严格改进 |
| Theorem 3 | 对 regularized softmax，ORF 给出更强的指数型尾界 |

因此工程里常说“ORF 收敛更快”，其严格含义不是某个统一误差阶突然改变，而是：

1. 平均误差更小。
2. MSE 更低。
3. 大偏差事件更少。

### 6. 一个新手友好的直觉总结

把这一节压缩成一句可操作的话：

- `positive` 解决的是“分母稳不稳”。
- `orthogonal` 解决的是“同样的随机特征数，误差能不能更小”。

前者偏数值稳定性，后者偏统计效率。两者合起来，才是 FAVOR+ 的完整设计。

---

## 代码实现

下面给出一个可直接运行的 NumPy 玩具实现，演示四件事：

1. 构造 i.i.d. 高斯正值特征；
2. 构造 block-ORF 正交高斯特征；
3. 比较它们对 $\exp(q^\top k)$ 的近似误差；
4. 用同一套特征实现单头 linear attention。

这段代码只依赖 `numpy`，可以直接保存为 `performer_orf_demo.py` 运行。

```python
import math
import numpy as np


def positive_features(x, omega):
    """
    x: [d]
    omega: [m, d]
    return: [m]
    """
    x = np.asarray(x, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    scale = math.exp(-0.5 * float(x @ x)) / math.sqrt(omega.shape[0])

    # 做一个简单的数值保护，避免 exp 直接上溢。
    proj = omega @ x
    proj = np.clip(proj, -60.0, 60.0)
    return scale * np.exp(proj)


def positive_feature_matrix(X, omega):
    """
    X: [L, d]
    omega: [m, d]
    return: [L, m]
    """
    X = np.asarray(X, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    norms = np.sum(X * X, axis=1, keepdims=True)  # [L, 1]
    proj = X @ omega.T                            # [L, m]
    proj = np.clip(proj, -60.0, 60.0)

    return np.exp(-0.5 * norms) * np.exp(proj) / math.sqrt(omega.shape[0])


def iid_gaussian_matrix(m, d, rng):
    return rng.normal(size=(m, d))


def orthogonal_gaussian_block(block_size, d, rng):
    """
    生成一块近似高斯分布尺度的正交方向。
    常见做法是 QR 后再乘以高斯范数，尽量匹配高斯行向量长度分布。
    """
    g = rng.normal(size=(d, d))
    q, _ = np.linalg.qr(g)

    # 采样 d 维高斯向量的范数，用来给每一行做缩放。
    radii = np.sqrt(rng.chisquare(df=d, size=d))
    block = (radii[:, None] * q)[:block_size, :]
    return block


def orthogonal_gaussian_matrix(m, d, rng):
    """
    block-ORF:
    - 当 m <= d 时，退化成单块正交。
    - 当 m > d 时，按每 d 行一块构造，块间独立。
    """
    blocks = []
    remaining = m
    while remaining > 0:
        block_size = min(d, remaining)
        blocks.append(orthogonal_gaussian_block(block_size, d, rng))
        remaining -= block_size
    return np.vstack(blocks)


def estimate_softmax_kernel(q, k, omega):
    phi_q = positive_features(q, omega)
    phi_k = positive_features(k, omega)
    return float(phi_q @ phi_k)


def exact_softmax_kernel(q, k):
    return math.exp(float(np.asarray(q) @ np.asarray(k)))


def single_head_favor_attention(Q, K, V, omega, eps=1e-12):
    """
    Q: [Lq, d]
    K: [Lk, d]
    V: [Lk, dv]
    omega: [m, d]
    return: [Lq, dv]
    """
    Qp = positive_feature_matrix(Q, omega)  # [Lq, m]
    Kp = positive_feature_matrix(K, omega)  # [Lk, m]

    kv = Kp.T @ V                           # [m, dv]
    z = Kp.sum(axis=0)                      # [m]
    denom = Qp @ z                          # [Lq]
    denom = np.maximum(denom, eps)          # 防止极端情况下过小

    out = (Qp @ kv) / denom[:, None]
    return out


def single_head_exact_attention(Q, K, V):
    scores = np.exp(Q @ K.T)
    probs = scores / scores.sum(axis=1, keepdims=True)
    return probs @ V


def run_kernel_demo():
    d = 8
    m = 16
    seeds = 200

    q = np.array([0.5, -0.2, 0.1, 0.7, -0.4, 0.3, 0.2, -0.1], dtype=np.float64)
    k = np.array([0.3, 0.1, -0.6, 0.4, -0.2, 0.5, -0.1, 0.2], dtype=np.float64)

    truth = exact_softmax_kernel(q, k)
    iid_abs_err = []
    orf_abs_err = []

    for seed in range(seeds):
        rng_iid = np.random.default_rng(seed)
        rng_orf = np.random.default_rng(seed)

        omega_iid = iid_gaussian_matrix(m, d, rng_iid)
        omega_orf = orthogonal_gaussian_matrix(m, d, rng_orf)

        est_iid = estimate_softmax_kernel(q, k, omega_iid)
        est_orf = estimate_softmax_kernel(q, k, omega_orf)

        iid_abs_err.append(abs(est_iid - truth))
        orf_abs_err.append(abs(est_orf - truth))

    print("=== Kernel Approximation Demo ===")
    print(f"truth                 = {truth:.8f}")
    print(f"mean_abs_error_iid    = {np.mean(iid_abs_err):.8f}")
    print(f"mean_abs_error_orf    = {np.mean(orf_abs_err):.8f}")
    print(f"median_abs_error_iid  = {np.median(iid_abs_err):.8f}")
    print(f"median_abs_error_orf  = {np.median(orf_abs_err):.8f}")

    # 玩具实验中 ORF 通常更优，但不做过强断言。
    assert truth > 0.0
    assert np.isfinite(np.mean(iid_abs_err))
    assert np.isfinite(np.mean(orf_abs_err))


def run_attention_demo():
    rng = np.random.default_rng(42)

    L = 6
    d = 8
    dv = 4
    m = 24

    Q = rng.normal(size=(L, d))
    K = rng.normal(size=(L, d))
    V = rng.normal(size=(L, dv))

    omega_iid = iid_gaussian_matrix(m, d, np.random.default_rng(1))
    omega_orf = orthogonal_gaussian_matrix(m, d, np.random.default_rng(1))

    exact = single_head_exact_attention(Q, K, V)
    out_iid = single_head_favor_attention(Q, K, V, omega_iid)
    out_orf = single_head_favor_attention(Q, K, V, omega_orf)

    err_iid = np.linalg.norm(out_iid - exact) / np.linalg.norm(exact)
    err_orf = np.linalg.norm(out_orf - exact) / np.linalg.norm(exact)

    print("\n=== Single-Head Attention Demo ===")
    print(f"relative_error_iid = {err_iid:.8f}")
    print(f"relative_error_orf = {err_orf:.8f}")

    assert np.all(np.isfinite(out_iid))
    assert np.all(np.isfinite(out_orf))


if __name__ == "__main__":
    run_kernel_demo()
    run_attention_demo()
```

### 代码里值得注意的三点

| 位置 | 作用 | 为什么需要 |
|---|---|---|
| `np.clip(proj, -60, 60)` | 限制指数输入范围 | 防止示例里直接溢出 |
| `block-ORF` | 支持 $m>d$ | 更符合真实工程写法 |
| `denom = np.maximum(denom, eps)` | 分母保护 | 防止极小值导致数值放大 |

### 对应的线性 attention 公式

上面代码中的核心三步，和论文中的线性化推导一一对应：

```python
Qp = phi(Q)            # [L, m]
Kp = phi(K)            # [L, m]

KV = Kp.T @ V          # [m, dv]
Z = Kp.sum(axis=0)     # [m]

out = (Qp @ KV) / (Qp @ Z)[:, None]
```

这段式子看起来不像 softmax，但本质结构没有变：

1. 先根据 query 和 key 产生权重。
2. 再对 value 做加权求和。
3. 最后做归一化。

只是原来“先构造完整权重矩阵再乘 $V$”，现在变成了“先把 key 端信息压缩聚合，再让每个 query 去读取”。

### 一个更贴近工程的例子

以长时序故障预测为例，假设有几千步传感器数据。标准 attention 需要显式比较每个时间步与所有其他时间步；FAVOR+ 则把每个时间步映射成随机特征后，直接对整段历史做聚合。这种场景下：

- 序列长，$L^2$ 成本高；
- 长依赖重要，不能只看局部窗口；
- attention 的语义仍然有价值，不想直接换成别的核。

这正是 FAVOR+ 比较有代表性的落点。

---

## 工程权衡与常见坑

FAVOR+ 的理论很漂亮，但工程里真正踩坑的地方，基本集中在“数值稳定性”和“近似误差”两个方向。

### 1. 把“random features”误解成“随便采样都行”

这是最常见的误区。  
对 softmax kernel 而言，不是任何随机特征都适合。原论文明确指出，一些带正负号震荡的特征在 softmax 小值区域会产生非常大的方差，结果可能是：

- 分母接近 0；
- 某些样本权重异常放大；
- 训练中出现 `NaN` 或梯度爆炸。

所以 FAVOR+ 强调的是 `positive random features`，不是一般意义上的随机特征。

### 2. 把“指数型尾界”误写成“误差整体指数下降”

这也是常见误读。  
正确表述是：论文给出了更强的尾概率界，也就是“大误差事件发生的概率”下降更快。  
这不等于所有误差指标都能直接写成指数衰减，更不等于“把 $m$ 加倍，误差一定按某个固定指数缩小”。

用表格区分最不容易错：

| 量 | ORF 的改进 |
|---|---|
| 单点估计的无偏性 | 仍成立 |
| MSE | 更小 |
| 大偏差概率 | 更强的指数型上界 |
| 每次实验的误差曲线 | 通常更好，但不保证完美单调 |

### 3. 忽略 block-ORF 的实现前提

原理论先写在 $m\le d$ 的情形，因为这时一块里最多只能放 $d$ 个两两正交方向。  
工程里如果想取更大的 $m$，标准做法不是放弃正交，而是：

1. 每 $d$ 个方向构成一个 block。
2. 块内做 QR 正交化。
3. 块间独立采样。

这就是为什么很多实现里会看到“分块生成正交矩阵”。

### 4. 忽略输入范数导致指数上溢

正值随机特征里有
$$
\exp(\omega^\top x).
$$
如果 $\|x\|$ 很大，或者某次采样正好让 $\omega^\top x$ 取到很大正值，就可能直接上溢。

一个粗略但实用的量级判断是：
$$
\omega^\top x \sim \mathcal N(0,\|x\|^2).
$$
也就是说，$\|x\|$ 越大，指数项越容易失控。

工程上常见对策包括：

| 问题 | 现象 | 对策 |
|---|---|---|
| 输入范数过大 | `exp` 上溢、输出异常大 | 预缩放、归一化、clip |
| 混合精度过激 | 半精度下数值更脆弱 | 关键路径保留 FP32 |
| 分母过小 | 输出抖动大 | `eps` 保护、检查特征实现 |

### 5. 忽略 redraw 随机矩阵的价值

随机特征有一个很现实的问题：理论上期望正确，不代表每一次采样都抽得好。  
某些 seed 下，某个 attention head 可能刚好抽到质量较差的方向集合，导致这一轮训练噪声偏大。这就是工程里常说的“抽卡效应”。

原论文实验也指出，周期性 redraw 随机矩阵可以进一步提升稳定性，因为它能把一次不理想采样的影响在训练过程中平均掉。

### 6. 一个简化版排错清单

| 症状 | 更可能的原因 | 优先检查 |
|---|---|---|
| loss 突然 `NaN` | 特征不是正值、分母不稳、指数上溢 | 特征公式与数值范围 |
| 误差始终很大 | $m$ 太小、未用 ORF、输入尺度异常 | $m$、block-ORF、输入归一化 |
| 某些 batch 特别差 | 随机矩阵采样不佳 | redraw 策略 |
| 理论正常但速度没提升 | 序列不够长或实现不高效 | $L$ 的规模与算子实现 |

可以把这一节压缩成一句工程判断：  
FAVOR+ 的收益来自“线性复杂度 + 较稳近似”，但前提是特征映射实现正确、随机矩阵构造合理、数值保护做足。

---

## 替代方案与适用边界

FAVOR+ 不是唯一的线性或近线性 attention 方案。实际选型时，不能只看复杂度，还要看“你到底要保留什么”。

### 1. 与常见方案的对比

| 方法 | 核心思想 | 复杂度 | 稳定性特点 | 适用边界 |
|---|---|---|---|---|
| Performer / FAVOR+ | 用正值随机特征近似 softmax | $O(Lmd)$ | 分母恒正，ORF 降低大偏差 | 长序列，且希望尽量保留 softmax 语义 |
| Linear Transformer | 直接换核函数 | 线性 | 实现简单，但不是 softmax 的近似 | 可以接受注意力语义改变 |
| Nyströmformer | 用 landmark 做低秩近似 | 近线性 | 依赖低秩结构与采样质量 | 注意力矩阵近似低秩时 |
| Reformer | 用 LSH 做稀疏路由 | 次二次 | 对哈希质量敏感 | 注意力稀疏结构明显 |
| FlashAttention | 精确 attention 的 IO 优化 | 仍是二次 | 数值稳定、工程成熟 | 中等长度序列，瓶颈主要是显存或带宽 |

### 2. 如何做实际判断

如果目标是：

- “我想保留 softmax attention 的行为，但序列太长，$L^2$ 顶不住”，优先考虑 Performer。
- “我不在乎是否还是 softmax，只想要一个简单线性 attention”，可以看 Linear Transformer 一类方法。
- “序列不算特别长，只是标准 attention 太吃显存”，FlashAttention 往往更直接，因为它没有近似误差。
- “注意力天然接近低秩或稀疏”，Nyströmformer、Reformer 这类方法可能更合适。

### 3. FAVOR+ 的真正适用边界

FAVOR+ 最适合的不是所有 Transformer，而是这类任务：

| 任务特征 | FAVOR+ 是否占优 |
|---|---|
| 序列极长，$L$ 达到几千甚至更高 | 通常是 |
| 仍希望保留 softmax 的全局交互语义 | 通常是 |
| 注意力不是主要瓶颈 | 未必 |
| 序列只有几百，且已有高效精确 attention | 往往不明显 |
| 任务对近似误差很敏感 | 需要谨慎评估 |

Performer-KAN 这类工作说明了 FAVOR+ 的一个典型应用面：超长时序信号。  
在工业传感器、长文档、蛋白质序列这类问题里，序列长度很大、全局依赖重要、显式 $L^2$ attention 又成本过高，此时 FAVOR+ 的价值最明显。

反过来说，如果任务只有几百个 token，或者训练瓶颈主要在前馈层、数据加载、优化器状态，而不在 attention，那么引入随机特征近似未必值得。

---

## 参考资料

1. Choromanski, Likhosherstov, Dohan, et al. *Rethinking Attention with Performers*. ICLR 2021.  
   重点：FAVOR+ 原始论文，给出了正值随机特征、ORF 的 MSE 改善、regularized softmax 的尾界，以及长序列误差实验。  
   链接：https://openreview.net/pdf?id=Ua6zuk0WRH

2. Rahimi, Recht. *Random Features for Large-Scale Kernel Machines*. NeurIPS 2007.  
   重点：随机特征近似核函数的经典起点。Performer 属于“把隐式核变成显式特征”的这一条技术路线。  
   链接：https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

3. Yu, Suresh, Choromanski, et al. *Orthogonal Random Features*. NeurIPS 2016.  
   重点：解释为什么正交化能降低核估计误差，是理解 Performer 中 ORF 设计的重要前置文献。  
   链接：https://papers.nips.cc/paper_files/paper/2016/file/53adaf494dc89ef7196d73636eb2451b-Paper.pdf

4. Wang, Chen, Zhang, et al. *Performer-KAN-Based Failure Prediction for IGBT with BO-CEEMDAN*. 2025.  
   重点：展示 FAVOR+ 在长时序工业信号上的实际落地方式，说明线性 attention 在工程侧的典型价值。  
   链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC12195581/

5. Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML 2020.  
   重点：代表“直接换核函数”的线性 attention 路线，适合与 Performer 的“近似 softmax”思路做对照。  
   链接：http://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf

6. Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.  
   重点：说明“高效 attention”不一定靠近似，也可以通过 IO 优化保留精确 softmax attention。  
   链接：https://arxiv.org/pdf/2205.14135
