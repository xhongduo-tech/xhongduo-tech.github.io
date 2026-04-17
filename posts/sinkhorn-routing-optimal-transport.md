## 核心结论

Sinkhorn 路由可以把 MoE 里的“token 该去哪个 expert”改写成一个带约束的最优传输问题。最优传输的直观含义是：在总量守恒的前提下，不是让每个 token 各自做局部最优，而是让整个 batch 一起求一个全局更优的分配矩阵。这里被“搬运”的不是货物，而是每个 token 总量为 1 的路由概率质量。

它和常见的 Top-k Softmax 路由的根本区别在于约束写入的位置不同。Softmax 路由先看每个 token 对各 expert 的局部分数，选出 top-k，再用 auxiliary loss 或 load balancing loss 在训练中“事后纠偏”；Sinkhorn 路由则把“每个 token 的概率必须分完”和“每个 expert 平均应该接到多少 token”直接写进优化约束里，在约束满足的前提下最大化整体匹配质量。论文里的熵正则化目标可以写成：

$$
\hat\Pi = \arg\max_{\Pi>0}\{\langle\Pi,C\rangle-\xi\langle\Pi,\log\Pi\rangle\},\quad
\Pi\mathbf{1}_n=\mathbf{1}_m,\quad
\Pi^{\!\top}\mathbf{1}_m=\frac{m}{n}\mathbf{1}_n
$$

这里：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $\Pi \in \mathbb{R}^{m\times n}$ | 传输矩阵 | 第 $i$ 个 token 分给第 $j$ 个 expert 的权重 |
| $C \in \mathbb{R}^{m\times n}$ | 兼容性矩阵 | token 和 expert 的匹配分数，越大越适合 |
| $\xi > 0$ | 熵正则系数 | 控制解是更尖锐还是更平滑 |
| $\langle \Pi, C\rangle$ | 内积项 | 尽量把更多质量分给高分匹配 |
| $-\xi\langle\Pi,\log\Pi\rangle$ | 熵正则项 | 防止解过早塌成极端硬分配 |

SSR（Selective Sinkhorn Routing）的关键并不是“训练全程都跑 Sinkhorn”，而是训练时以概率 $p$ 在 Softmax 路由和 Sinkhorn 路由之间切换。Softmax 路由负责保留门控网络对真实兼容性的学习能力，Sinkhorn 路由负责在少量训练步中强制 batch 内负载均衡。这样既能减少 expert collapse，又把完整 Sinkhorn 的额外计算成本压到很低。

如果把问题缩到最小例子，两个 token 对应两个 expert，SSR 的含义就是：大多数训练步仍然让模型按普通 Softmax 学习“谁更适合谁”，少数训练步把这一小批 token 的分配交给 Sinkhorn，在行归一化和列归一化之后，强制两个 expert 都接到差不多的总质量。这个“低频强约束纠偏”就是 SSR 的核心价值。

---

## 问题定义与边界

MoE 的路由问题，本质上是在做一个受约束的稀疏分配。MoE 是 Mixture of Experts，可以理解为“模型内部有很多专家子网络，但每个 token 只激活其中少数几个”。这样一来，总参数量可以很大，但每次前向传播不需要把所有参数都跑一遍。

设一个 batch 里有 $m$ 个 token、$n$ 个 expert。我们要学习一个矩阵 $\Pi\in\mathbb{R}^{m\times n}$，其中 $\Pi_{ij}$ 表示第 $i$ 个 token 分配给第 $j$ 个 expert 的权重。这个矩阵通常满足下面三类约束：

| 约束 | 数学形式 | 含义 | 实践影响 |
| --- | --- | --- | --- |
| 正性 | $\Pi > 0$ | 每个元素都为正 | 熵项有定义，优化过程更平滑 |
| 行边际 | $\Pi\mathbf{1}_n=\mathbf{1}_m$ | 每个 token 的总权重为 1 | 每个 token 的路由概率必须完整分配 |
| 列边际 | $\Pi^\top\mathbf{1}_m=\frac{m}{n}\mathbf{1}_n$ | 每个 expert 平均接收 $m/n$ 单位质量 | batch 内负载被直接拉平 |

这里“边际”这个术语容易让新手卡住。它的意思其实很简单：

- 行和固定：看每个 token 那一行，所有 expert 权重加起来必须等于 1。
- 列和固定：看每个 expert 那一列，所有 token 分给它的总质量必须接近设定目标。

因此，Sinkhorn 路由并不是“每个 token 单独找最优 expert”，而是“整个 batch 联合协商，谁多给一点、谁少给一点”，最后得到一个同时满足行约束和列约束的矩阵。

这个边界要看清楚，否则很容易误用。

第一，列边际约束描述的是“batch 级平均均衡”，不是“所有输入、所有时间、所有阶段都必须均衡”。它的主要价值在训练期：避免少数 expert 长期吃满流量，其他 expert 长期收不到梯度。到了推理期，如果仍对单个样本或小批量强行施加列边际约束，就会把本来应该体现输入差异的专家专精拉向平均分配，损伤表达能力。

第二，Sinkhorn 求出来的通常是稠密矩阵 $\hat\Pi$，但 MoE 真正执行时仍然往往只保留每行 top-k 个 expert。也就是说，训练时先通过全局约束得到“谁整体更该分到多少”的稠密分配，再把它截成稀疏路由权重。最终真正计算的还是少数 expert，而不是让所有 expert 都参与前向。

第三，列边际约束是“平均容量约束”，不是“硬容量上限”。如果某个工程实现还带 capacity factor、token dropping 或者 overflow 逻辑，那是另一层工程机制。Sinkhorn 先解决的是“负载倾斜”问题，不是一次性解决所有路由和调度问题。

为了更直观，可以把 Softmax 路由和 Sinkhorn 路由的决策方式并排看：

| 维度 | Top-k Softmax | Sinkhorn |
| --- | --- | --- |
| 决策粒度 | 每个 token 独立决策 | 整个 batch 联合决策 |
| 均衡方式 | 依赖额外辅助损失 | 列边际约束直接写进目标 |
| 输出形态 | 天然是行归一化分布 | 近似双随机或带目标列和的矩阵 |
| 训练成本 | 低 | 更高，需要迭代缩放 |
| 推理适配 | 直接可用 | 通常不直接用于推理期均衡 |

---

## 核心机制与推导

### 1. 为什么是最优传输

最优传输问题通常有两个对象：供给端和需求端。供给端有多少质量，需求端需要接收多少质量，中间要找一个整体最优的搬运方案。把这个语言翻译到 MoE 路由里：

- 每个 token 提供 1 单位概率质量。
- 所有 token 一共提供 $m$ 单位质量。
- 每个 expert 目标上平均接收 $m/n$ 单位质量。
- 在所有满足这些约束的分配矩阵里，优先把更多质量分给高兼容性的 token-expert 配对。

所以，Sinkhorn 路由和普通 gating 的区别，不只是“多了一个算法”，而是目标函数的结构变了。Softmax 更像是“每个 token 自己投票”；最优传输更像是“整个 batch 做一次受约束的全局排班”。

目标函数里的两项各自承担不同作用：

$$
\langle\Pi,C\rangle
$$

这一项希望把质量尽量放在高分位置上。如果没有别的约束，它会鼓励解尽量偏向分数最高的几个 expert。

$$
-\xi\langle\Pi,\log\Pi\rangle
$$

这一项是熵正则。它的作用不是提升分数，而是让分配不要过早塌成非常尖锐的硬选择。这样做有三个现实好处：

| 作用 | 解释 |
| --- | --- |
| 数值稳定 | 没有熵正则时，最优传输更容易退化成难优化的硬匹配 |
| 可微性更好 | 稠密正值矩阵比 0-1 选择更容易反向传播 |
| 可用 Sinkhorn 高效求解 | 熵正则化 OT 可以转成矩阵缩放问题 |

$\xi$ 的含义也要明确：

| $\xi$ 大小 | 解的形态 | 优点 | 风险 |
| --- | --- | --- | --- |
| 小 | 更尖锐，接近硬匹配 | 更像最终 top-k 选择 | 容易数值不稳定，可能溢出 |
| 大 | 更平滑，更分散 | 更稳定、更容易收敛 | 可能过度平均，弱化最优 expert |

因此，Sinkhorn 不是在“精确模拟最终 top-k”，而是在训练时找到一个兼顾匹配质量、均衡约束和数值稳定性的中间分配。

### 2. Sinkhorn 怎么逼近双随机矩阵

有了熵正则化目标，问题可以转写成对一个正矩阵做交替缩放。先定义核矩阵：

$$
K=\exp(C/\xi)
$$

这里指数不是装饰，而是来自熵正则化最优传输的解析形式。分数越高，$K_{ij}$ 越大；分数越低，$K_{ij}$ 越小。接下来引入两个缩放向量 $u\in\mathbb{R}^m_{>0}$ 和 $v\in\mathbb{R}^n_{>0}$，让最终矩阵写成：

$$
\hat\Pi = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)
$$

问题就变成：如何选 $u,v$，使得 $\hat\Pi$ 的行和等于 $r$，列和等于 $s$。在本文设定里：

$$
r=\mathbf{1}_m,\qquad s=\frac{m}{n}\mathbf{1}_n
$$

Sinkhorn-Knopp 算法做的事非常直接：先修列，再修行，反复交替。

$$
v^{(t)} = s \oslash (K^\top u^{(t-1)}),\qquad
u^{(t)} = r \oslash (K v^{(t)})
$$

其中 $\oslash$ 表示按元素相除。

这两个更新式的意义分别是：

- 如果当前列和不对，就调整 $v$，把每一列整体缩放到目标总量附近。
- 如果当前行和不对，就调整 $u$，把每一行整体缩放到目标总量附近。
- 不断交替后，矩阵会越来越接近同时满足行约束和列约束。

这个过程常被说成“逼近双随机矩阵”，更准确地说，在本文设定下它逼近的是“行和为 1、列和为 $m/n$”的矩阵；只有当 $m=n$ 且列目标也等于 1 时，才是标准双随机矩阵。

从计算角度看，每轮迭代主要包含两次矩阵向量乘：

- 一次 $K^\top u$
- 一次 $Kv$

所以成本和迭代轮数几乎线性相关。轮数越多，边际误差越小；轮数越少，开销越低，但均衡越差。

| 迭代轮次 | 典型效果 | 适用场景 |
| --- | --- | --- |
| 5 | 常常只能粗略纠偏 | 只想做很轻量的平衡校正 |
| 20 | 多数小规模实验已能明显改善列失衡 | 常见折中点 |
| 50 | 边际误差通常更小 | 更重视稳定均衡 |
| 100 | 接近论文里常见上限设置 | 研究型实验或小模型验证 |

新手常见误解是把 Sinkhorn 看成“排序算法”或“选 top-k 的替代品”。它实际上不是排序，而是一个带目标边际的矩阵重标定过程。排序解决的是“谁最大”，Sinkhorn 解决的是“在整体约束下，谁该分多少”。

### 3. 玩具例子

先看一个最小可算例子。设有 2 个 token、2 个 expert，兼容性矩阵为：

$$
C=\begin{bmatrix}
0.73 & 0.27\\
0.27 & 0.73
\end{bmatrix}
$$

它表示：

- token 1 更适合 expert 1
- token 2 更适合 expert 2
- 但两者都不是绝对排他，只是对角线位置更优

如果直接对每一行做 Softmax，两行都会偏向对角线位置；如果再做 top-1，结果也会是对角选中。但 Sinkhorn 多做了一步：它不仅关心“每行谁更大”，还关心“每列总共拿到了多少质量”。

以 $\xi=0.5$ 为例，先算核矩阵：

$$
K=\exp(C/\xi)=
\exp\!\left(
\begin{bmatrix}
1.46 & 0.54\\
0.54 & 1.46
\end{bmatrix}
\right)
\approx
\begin{bmatrix}
4.305 & 1.716\\
1.716 & 4.305
\end{bmatrix}
$$

由于这个例子本身是对称的，迭代后会得到一个同样对称的分配矩阵。若迭代充分收敛，可得到近似：

$$
\hat\Pi \approx
\begin{bmatrix}
0.715 & 0.285\\
0.285 & 0.715
\end{bmatrix}
$$

检查它的边际：

$$
\hat\Pi\mathbf{1}_2 \approx
\begin{bmatrix}
1\\
1
\end{bmatrix},
\qquad
\hat\Pi^\top\mathbf{1}_2 \approx
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

这说明两件事同时成立：

- 每个 token 的权重都分完了。
- 每个 expert 接收到的总质量也相等。

再做每行 top-1，就得到最终稀疏路由：

- token 1 送往 expert 1
- token 2 送往 expert 2

这个例子之所以重要，不是因为结果“看起来和直觉一致”，而是因为它说明了一个事实：Sinkhorn 并不会破坏合理匹配，它做的是在合理匹配基础上，把 batch 级负载一起校正。

如果把例子改坏一点，例如两行都强烈偏向同一个 expert：

$$
C=\begin{bmatrix}
0.95 & 0.05\\
0.90 & 0.10
\end{bmatrix}
$$

那么纯 Softmax 更容易把两个 token 都压给第一个 expert；而 Sinkhorn 会因为列约束存在，被迫把一部分质量重新分给第二个 expert。这样做会牺牲一部分局部匹配分数，但换来更好的负载均衡和更健康的 expert 训练信号。

### 4. 为什么 top-k 后还合理

实际 MoE 通常不执行稠密路由，而是只保留每个 token 最相关的 $k$ 个 expert。因此，一个自然问题是：既然 Sinkhorn 先求的是稠密矩阵 $\hat\Pi$，为什么后面再做 top-k 不会把前面的优化全毁掉？

论文采用的做法是：先选出第 $i$ 行最大的 $k$ 个位置 $\{i_1,\dots,i_k\}$，再在这些位置上重新归一化：

$$
\alpha_{i,i_r}=\frac{\hat\Pi_{i,i_r}}{\sum_{j=1}^{k}\hat\Pi_{i,i_j}}
$$

其中 $\alpha_i$ 是第 $i$ 个 token 最终真正执行的稀疏路由权重。

这个式子的意义不是“拍脑袋重缩放”，而是有一个信息论上的解释：在“只保留 $k$ 个位置、且这些位置权重和为 1”的约束下，上式对应于尽量保留原始 $\hat\Pi_i$ 的信息，也就是尽量小地改变原分布。常见的写法是最小化某种与 KL 散度有关的偏离。

KL 散度如果不熟，可以暂时把它理解成“两个分布差得有多远”。那么 top-k 后再归一化的目标就是：虽然必须做稀疏化，但要尽量让稀疏后的分布仍然贴近原始 Sinkhorn 解。

这背后的工程逻辑是：

| 步骤 | 目的 |
| --- | --- |
| 稠密 Sinkhorn | 在 batch 级别实现受约束的全局均衡 |
| top-k 截断 | 回到稀疏计算，控制真正前向成本 |
| 重新归一化 | 保证最终执行权重仍是合法概率分布 |

因此，Sinkhorn 路由不是和稀疏 MoE 对立，而是给稀疏 MoE 提供一个更均衡的“稠密前解”，再把它安全地压回到 top-k 执行形式。

---

## 代码实现

下面给一个可以直接运行的 Python 示例。它做三件事：

1. 实现稳定版 Sinkhorn 迭代
2. 把稠密矩阵做 top-k 稀疏化
3. 模拟 SSR 在 Softmax 路由和 Sinkhorn 路由之间按概率切换

代码只依赖 `numpy`，复制到本地 `python3` 环境即可运行。

```python
import numpy as np


def row_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def sinkhorn_transport(scores, xi=0.5, max_iter=100, tol=1e-6, eps=1e-12):
    """
    Compute an entropic OT plan with row target = 1 and column target = m / n.

    Args:
        scores: shape [m, n], compatibility matrix C. Larger means better match.
        xi: entropy regularization coefficient.
        max_iter: maximum number of Sinkhorn iterations.
        tol: stopping threshold on marginal error.
        eps: small constant to avoid division by zero.

    Returns:
        Pi: shape [m, n], dense transport plan.
        stats: dict with iteration count and marginal errors.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D matrix")

    m, n = scores.shape
    if m == 0 or n == 0:
        raise ValueError("scores must be non-empty")

    # Stabilize exponentiation. Subtracting the global max does not change the
    # final normalized solution after Sinkhorn scaling, but reduces overflow risk.
    scaled = (scores - np.max(scores)) / max(xi, eps)
    K = np.exp(scaled)
    K = np.maximum(K, eps)

    r = np.ones(m, dtype=np.float64)
    s = np.full(n, m / n, dtype=np.float64)

    u = np.ones(m, dtype=np.float64)
    v = np.ones(n, dtype=np.float64)

    row_err = np.inf
    col_err = np.inf

    for step in range(1, max_iter + 1):
        Kv = K @ v
        Kv = np.maximum(Kv, eps)
        u = r / Kv

        KTu = K.T @ u
        KTu = np.maximum(KTu, eps)
        v = s / KTu

        Pi = (u[:, None] * K) * v[None, :]
        row_err = np.max(np.abs(Pi.sum(axis=1) - r))
        col_err = np.max(np.abs(Pi.sum(axis=0) - s))

        if max(row_err, col_err) < tol:
            break

    stats = {
        "iterations": step,
        "row_err": float(row_err),
        "col_err": float(col_err),
    }
    return Pi, stats


def topk_and_renorm(Pi, k=1):
    """
    Keep top-k entries per row and renormalize each row to sum to 1.
    """
    Pi = np.asarray(Pi, dtype=np.float64)
    m, n = Pi.shape
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= number of experts")

    alpha = np.zeros_like(Pi)
    top_idx = np.argpartition(-Pi, kth=k - 1, axis=1)[:, :k]

    for i in range(m):
        cols = top_idx[i]
        vals = Pi[i, cols]
        vals_sum = np.sum(vals)
        if vals_sum <= 0:
            raise ValueError("row has non-positive top-k sum")
        alpha[i, cols] = vals / vals_sum

    return alpha


def hard_top1_softmax(scores):
    probs = row_softmax(scores)
    top = np.argmax(probs, axis=1)
    alpha = np.zeros_like(probs)
    alpha[np.arange(scores.shape[0]), top] = 1.0
    return alpha, probs


def ssr_route(scores, p=0.01, xi=0.5, top_k=1, max_iter=50, tol=1e-6, rng=None):
    """
    Selective Sinkhorn Routing:
    with probability p, use Sinkhorn routing; otherwise use standard softmax top-k.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if rng.random() < p:
        Pi, stats = sinkhorn_transport(scores, xi=xi, max_iter=max_iter, tol=tol)
        alpha = topk_and_renorm(Pi, k=top_k)
        return {
            "mode": "sinkhorn",
            "Pi": Pi,
            "alpha": alpha,
            "stats": stats,
        }

    alpha, probs = hard_top1_softmax(scores)
    return {
        "mode": "softmax",
        "Pi": probs,
        "alpha": alpha,
        "stats": None,
    }


def print_matrix(name, x):
    print(f"{name} =")
    print(np.array2string(x, precision=4, suppress_small=True))
    print()


def main():
    # Example 1: symmetric case, easy to inspect
    scores = np.array([
        [0.73, 0.27],
        [0.27, 0.73],
    ], dtype=np.float64)

    Pi, stats = sinkhorn_transport(scores, xi=0.5, max_iter=100, tol=1e-8)
    alpha = topk_and_renorm(Pi, k=1)

    print("=== Example 1: balanced preference ===")
    print_matrix("scores", scores)
    print_matrix("Pi", Pi)
    print_matrix("alpha", alpha)
    print("stats =", stats)
    print("row sums =", Pi.sum(axis=1))
    print("col sums =", Pi.sum(axis=0))
    print()

    assert np.allclose(Pi.sum(axis=1), np.ones(2), atol=1e-6)
    assert np.allclose(Pi.sum(axis=0), np.ones(2), atol=1e-6)
    assert np.array_equal(np.argmax(alpha, axis=1), np.array([0, 1]))
    assert np.allclose(alpha.sum(axis=1), np.ones(2), atol=1e-8)

    # Example 2: both tokens prefer expert 0; Sinkhorn redistributes load
    skewed_scores = np.array([
        [0.95, 0.05],
        [0.90, 0.10],
    ], dtype=np.float64)

    Pi2, stats2 = sinkhorn_transport(skewed_scores, xi=0.5, max_iter=100, tol=1e-8)
    alpha2 = topk_and_renorm(Pi2, k=1)

    print("=== Example 2: skewed preference ===")
    print_matrix("scores", skewed_scores)
    print_matrix("softmax probs", row_softmax(skewed_scores))
    print_matrix("Pi", Pi2)
    print_matrix("alpha", alpha2)
    print("stats =", stats2)
    print("row sums =", Pi2.sum(axis=1))
    print("col sums =", Pi2.sum(axis=0))
    print()

    assert np.allclose(Pi2.sum(axis=1), np.ones(2), atol=1e-6)
    assert np.allclose(Pi2.sum(axis=0), np.ones(2), atol=1e-6)

    # SSR demo: force Sinkhorn branch
    result = ssr_route(scores, p=1.0, xi=0.5, top_k=1, max_iter=100, tol=1e-8)
    assert result["mode"] == "sinkhorn"
    assert np.allclose(result["alpha"].sum(axis=1), np.ones(scores.shape[0]), atol=1e-8)

    # SSR demo: force Softmax branch
    result = ssr_route(scores, p=0.0, xi=0.5, top_k=1)
    assert result["mode"] == "softmax"
    assert np.allclose(result["alpha"].sum(axis=1), np.ones(scores.shape[0]), atol=1e-8)

    print("All checks passed.")


if __name__ == "__main__":
    main()
```

这段代码和很多“看起来能跑”的伪实现有几个关键差别：

| 问题 | 常见错误写法 | 这里的处理 |
| --- | --- | --- |
| 数值溢出 | 直接对大分数做 `exp(scores / xi)` | 先减去全局最大值，再指数化 |
| 更新顺序不清楚 | 每轮内混用旧 `u,v` 导致难复现 | 固定按 `u -> v -> Pi` 的顺序更新 |
| 列目标遗漏 | 默认按标准双随机处理 | 明确使用列目标 $m/n$ |
| top-k 后不归一化 | 截断后直接拿来算 | 重新归一化为合法分布 |
| 只给一个例子 | 看不出负载均衡效果 | 同时给平衡例子和失衡例子 |

如果把它嵌入训练循环，可以写成下面这种更接近工程实际的伪代码：

```python
# gating_scores: [num_tokens, num_experts]
# top_k: number of active experts per token

if random() < p:
    # Low-frequency correction branch
    Pi, _ = sinkhorn_transport(
        gating_scores,
        xi=xi,
        max_iter=max_iter,
        tol=delta,
    )
    alpha = topk_and_renorm(Pi, k=top_k)
else:
    # Standard training branch
    probs = row_softmax(gating_scores + gaussian_noise)
    alpha = topk_and_renorm(probs, k=top_k)
```

这里要特别注意一个概念区分：`scores`、`probs`、`Pi` 不是同一个东西。

| 名称 | 含义 | 是否已经满足行和为 1 | 是否已经满足列目标 |
| --- | --- | --- | --- |
| `scores` | 门控网络原始打分 | 否 | 否 |
| `probs` | 对每行做 Softmax 后的概率 | 是 | 否 |
| `Pi` | Sinkhorn 后的传输矩阵 | 是 | 是，近似满足 |

也就是说，SSR 不是在 Softmax 的结果上做一个简单平滑，而是把“局部概率”进一步提升成“满足全局约束的路由方案”。

论文里的实际训练策略也说明了这一点：它不是每一步都跑 Sinkhorn，而是只在很小比例的训练步触发，例如大约 0.1% 到 1%。这等价于把 Sinkhorn 视作一种低频但强力的负载校正器，而不是默认主路径。

可以作为起点的参数范围如下：

| 参数 | 典型取值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `p` | `0.001` 到 `0.01` | 触发 Sinkhorn 的概率 | 先从 `0.005` 左右试 |
| `xi` | `0.5` 或 `1.0` | 熵正则强度 | 新手优先从 `1.0` 起步 |
| `delta` | `1e-4` 到 `1e-6` | 停止阈值 | 误差不敏感时可放宽 |
| `max_iter` | `20` 到 `100` | 最大迭代轮数 | 小模型先用 `20~50` |
| `alpha_noise` | `0.1` 到 `4` | 门控噪声强度 | 防止早期 expert 垄断 |

---

## 工程权衡与常见坑

Sinkhorn 路由最大的优点，是把负载均衡从“额外正则项希望它发生”变成“约束层面要求它发生”。最大的代价，则是每次触发都要做多轮矩阵缩放，因此训练成本会上升。

这类方法最值得看的不是“理论上更优”，而是“在真实训练里贵多少、值不值”。论文给出的结果很有代表性：完整 Sinkhorn-based SMoE 相对 vanilla SMoE 会明显增加训练时间，而 SSR 只在极少数步触发 Sinkhorn，因此几乎保留了 Softmax 路由的主路径效率，同时获得更稳的 expert 负载。

从工程角度，最重要的不是记住具体百分比，而是理解这三个结论：

1. 完整 Sinkhorn 比纯 Softmax 贵得多。
2. SSR 的开销主要由触发概率 $p$ 决定。
3. 对大模型来说，低频强校正通常比全程 OT 更现实。

常见问题可以直接总结成表：

| 问题 | 现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 推理阶段继续做 Sinkhorn 均衡 | 输出变“平均”，expert 专精下降 | 列边际约束把输入差异抹平了 | 推理期只用普通路由，不做 batch 均衡 |
| `xi` 太小导致 NaN 或 inf | 指数爆炸、训练不稳 | $\exp(C/\xi)$ 对小 $\xi$ 非常敏感 | 从 `0.5` 或 `1.0` 起步，必要时裁剪分数 |
| `p` 太小几乎不生效 | expert 仍长期失衡 | 纠偏频率不足 | 至少保证有可观察的触发次数 |
| `p` 太大拖慢训练 | 吞吐明显下降 | Sinkhorn 分支太频繁 | 先从千分级或百分级小概率试 |
| 迭代轮数太少 | 列和误差大，均衡差 | 未充分收敛 | 联合调 `max_iter` 和 `tol` |
| 没有门控噪声 | 少数 expert 很早锁死优势 | 早期偏置自我强化 | 训练期加 Gaussian noise |
| top-k 后忘记归一化 | 权重和不等于 1 | 稠密矩阵被截断后未修正 | 始终重新归一化 |
| 把 batch 均衡理解成全局最优 | 离线分析好看，在线推理变差 | 训练目标和推理目标混淆 | 明确区分 train routing 与 inference routing |

给新手一个更实用的判断标准：

| 观察现象 | 更可能该调什么 |
| --- | --- |
| expert 使用极不均匀 | 增大 `p`，或加大噪声 |
| 训练突然不稳、出现溢出 | 增大 `xi`，缩小 score 范围 |
| Sinkhorn 太慢 | 降低 `p`，减少 `max_iter` |
| top-k 后效果差 | 检查稀疏化和归一化是否正确 |
| 训练期好、推理期差 | 检查是否错误地把均衡约束搬到推理 |

参数上，一个保守起点通常是：

- `p >= 0.001`
- `xi >= 0.5`
- `delta = 1e-4`
- `max_iter = 20~50`

错误用法通常是：训练期和推理期完全沿用同一套 Sinkhorn 均衡逻辑，希望“既强平衡又强表达”。这往往做不到。正确用法更接近分工协作：

- 训练期：让 SSR 偶尔触发 Sinkhorn，修正 expert 负载
- 推理期：回到确定性的 Softmax / top-k 路由，保留 expert 专精

---

## 替代方案与适用边界

如果把主流思路粗略分成三类，它们分别代表三种不同取舍：

| 方案 | 代表思路 | 优点 | 代价 |
| --- | --- | --- | --- |
| Vanilla Softmax | 局部打分 + top-k | 最简单、最快、最易部署 | 容易出现 expert collapse |
| 完整 Sinkhorn | 全程最优传输均衡 | 负载最强约束 | 训练显著变慢 |
| SSR | 低频 Sinkhorn + 高频 Softmax | 效率和均衡折中较好 | 需要额外切换逻辑和参数调节 |

如果结合论文里给出的困惑度或 BPC 结果，整体趋势也比较清晰：

| 方案 | PPL / BPC | 训练开销 | 结论 |
| --- | --- | --- | --- |
| Vanilla Softmax | 基线水平 | 最低 | 适合作为默认起点 |
| 完整 Sinkhorn | 不一定优于基线 | 很高 | 更像研究型方案 |
| SSR-S w/ noise | 结果通常更稳、更优 | 很低 | 更接近可落地工程方案 |

这背后的原因并不神秘。

纯 Softmax 的优势，在于它完全尊重门控网络的局部分数，因此学习路径简单、梯度直接、实现代价低。但它的问题也同样直接：一旦某些 expert 在训练早期吃到优势，后续就更容易继续吃到更多 token，形成正反馈，最后出现专家崩溃或专家闲置。

完整 Sinkhorn 的优势，是把这种失衡问题在优化层面直接压住。它不是“希望 expert 更均匀”，而是“要求 expert 更均匀”。但它的缺点也来自同一处：每次都做全局约束求解太贵，而且这种 batch 级均衡如果被误带到推理期，还会削弱 expert 的输入条件化专精。

SSR 的价值恰好在中间。它承认两件事同时成立：

- Softmax 对学习真实兼容性是必要的。
- 仅靠 Softmax 又很容易让负载失衡。

因此 SSR 不试图让某一边完全取代另一边，而是把 Sinkhorn 变成一个低频、强制、用于纠偏的训练插件。这个思路对大型 MoE 尤其合理，因为大模型通常训练周期长、专家数多、轻微失衡会被长期放大，而全程运行最优传输又很难承受。

可以把适用边界概括得更直白一些：

| 场景 | 更推荐 |
| --- | --- |
| 训练预算非常紧，先求简单可跑 | Vanilla Softmax |
| 研究负载均衡本身，希望验证最强 OT 约束 | 完整 Sinkhorn |
| 既关心训练效率，又想缓解 expert collapse | SSR |
| 推理期要求稳定且简单 | Softmax / top-k 推理 |
| 训练期 expert 明显失衡 | 增加 SSR 触发频率 |

因此，Selective Sinkhorn 更像是一个工程上可接受的折中，而不是“理论最优方案的缩水版”。它的重点不在于把最优传输做满，而在于只在最有价值的地方付出这笔成本。

---

## 参考资料

1. Duc Anh Nguyen, Huu Binh Ta, Nhuan Le Duc, Tan M. Nguyen, Toan Tran. *Selective Sinkhorn Routing for Improved Sparse Mixture of Experts*. 原始论文，包含 OT 目标、Algorithm 1、理论命题与实验结果。https://arxiv.org/abs/2511.08972
2. 论文 PDF 版本。适合直接查公式、算法与实验表格。https://arxiv.org/pdf/2511.08972
3. Moonlight 评述：*Selective Sinkhorn Routing for Improved Sparse Mixture of Experts*。偏工程视角的结构化解读。https://www.themoonlight.io/review/selective-sinkhorn-routing-for-improved-sparse-mixture-of-experts
4. AIModels.fyi 解读：*Selective Sinkhorn Routing for Improved Sparse Mixture of Experts*。适合快速浏览问题背景、方法和结果。https://www.aimodels.fyi/papers/arxiv/selective-sinkhorn-routing-improved-sparse-mixture-experts
5. Marco Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. 熵正则最优传输和 Sinkhorn 迭代的经典参考，适合补最优传输基础。https://arxiv.org/abs/1306.0895
