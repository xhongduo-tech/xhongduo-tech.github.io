## 核心结论

线性注意力里的因果 Mask，本质上可以改写成“前缀状态的递推更新”。这句话的意思是：在第 $t$ 个位置，不需要重新和全部历史 token 逐个做交互，只要保留到第 $t$ 步为止的两个累计状态，再结合当前 query，就能得到当前位置的输出。

对去掉 softmax、并把相似度写成核分解形式的因果注意力，若先对 query 和 key 做特征映射 $\phi(\cdot)$，则历史可以压缩成两个状态：

$$
S_t = S_{t-1} + \phi(k_t) v_t^\top,\qquad
z_t = z_{t-1} + \phi(k_t)
$$

其中：

- $S_t \in \mathbb{R}^{d_\phi \times d_v}$：状态矩阵，累计了“历史 key 特征对 value 的写入”。
- $z_t \in \mathbb{R}^{d_\phi}$：归一化向量，累计了“历史 key 特征的总量”。
- $d_\phi$：特征映射后的维度，不一定等于原始 $d_k$。
- $d_v$：value 维度。

输出写成：

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
$$

这件事直接带来三个结论：

1. 因果线性注意力在推理时等价于一个 RNN 风格的状态更新器。
2. 每一步只需要维护固定大小状态，不需要随着上下文长度增长的完整 KV cache。
3. 训练时不必真的逐步串行执行；因为 $S_t,z_t$ 都是前缀和，可以用并行 scan 或 prefix sum 一次性算完整段序列。

最直接的直觉是：标准注意力保存的是“整段历史明细”；因果线性注意力保存的是“历史的压缩统计量”。新 token 到来时，不再回看全部历史，而是读这个统计状态。

---

## 问题定义与边界

先把问题说清楚。因果注意力中的“因果”，指当前位置只能访问自己和过去的位置，不能访问未来；“Mask”就是一个下三角约束，它把所有未来位置屏蔽掉。

标准 softmax 因果注意力在第 $t$ 步可写成：

$$
o_t = \sum_{i \le t} \frac{\exp(q_t^\top k_i)}{\sum_{j \le t}\exp(q_t^\top k_j)} v_i
$$

这里：

- $q_t$ 是当前位置的 query；
- $k_i, v_i$ 是第 $i$ 个历史位置的 key 和 value；
- 分子表示当前位置对第 $i$ 个历史位置的权重；
- 分母是归一化项，保证所有权重加起来为 $1$。

这个公式的表达力没有问题，问题主要出在推理阶段的存储和带宽。

当模型已经生成到第 $t$ 个 token，要生成第 $t+1$ 个 token 时，通常还得保留前面所有的 $k_1,\dots,k_t$ 和 $v_1,\dots,v_t$。因此：

- 序列越长，KV cache 越大；
- 显存占用会增长；
- 每步读取历史时的带宽压力也会增长。

线性注意力要解决的不是“取消按内容读取历史”，而是把“读取整段历史”改成“读取一个固定维度的状态”。

可以先用一张表把边界对齐：

| 项目 | 传统因果 attention | 因果线性 attention |
|---|---|---|
| 历史表示 | 全量 KV cache | 固定大小状态 $(S_t,z_t)$ |
| 单步推理存储 | 随 $t$ 增长 | 与 $t$ 无关 |
| 单步推理计算 | 与全部历史交互 | 状态更新 + 状态读取 |
| 归一化 | softmax | 显式分母 $\phi(q_t)^\top z_t$ |
| 是否天然写成递推 | 否 | 是 |

这件事成立，有三个前提。

第一，任务必须是因果场景。  
典型例子是自回归语言模型、流式语音识别、在线推荐日志建模。当前位置只能看过去，这时“前缀递推”才和任务约束完全一致。双向编码也可以设计类似形式，但最自然的递推解释是在因果设置下。

第二，注意力核必须能拆开。  
标准 softmax 用的是 $\exp(q^\top k)$，而线性注意力通常要求相似度能写成：

$$
\mathrm{score}(q,k)=\phi(q)^\top \phi(k)
$$

只有写成这种“query 一部分、key 一部分”的乘积形式，才能把历史项提前累计进状态，等到 query 来时再读取。

第三，固定状态不是免费压缩。  
状态维度 $d_\phi$ 决定你能保留多少历史信息。维度太小，状态容量不足，模型可能无法保留复杂的长程依赖。换句话说，线性注意力不是“无损压缩全部历史”，而是“用固定大小状态表示某种历史统计”。

对新手，一个简单类比是：

- 标准注意力像保存所有交易明细，然后每次都查整本账；
- 线性注意力像持续维护总账和若干统计量，查询时直接读统计结果。

这个类比只帮助理解“压缩状态”，不能替代正式定义。正式上，保存的是 $\sum \phi(k_i)v_i^\top$ 和 $\sum \phi(k_i)$ 这两类前缀量。

工程里最典型的应用是长上下文流式生成。假设上下文从 4k 扩展到 128k：

- 标准 KV cache 的容量和读取流量会持续增大；
- 线性注意力只维护固定状态，单步存储不会随上下文增长。

这就是它在低延迟、流式、超长上下文设置下持续被讨论的原因。

---

## 核心机制与推导

推导从一个关键替换开始：把原本依赖 $q_t^\top k_i$ 的相似度，改写成核分解形式：

$$
\mathrm{score}(q_t,k_i)=\phi(q_t)^\top \phi(k_i)
$$

于是因果输出写成：

$$
o_t
=
\frac{\sum_{i\le t} \phi(q_t)^\top \phi(k_i)\, v_i}
{\sum_{i\le t} \phi(q_t)^\top \phi(k_i)}
$$

这一步还只是把 softmax 换成了可分解的核形式，重点在于下一步的提取。

因为对固定的时刻 $t$ 来说，$\phi(q_t)$ 与求和下标 $i$ 无关，所以它可以从求和内部提出：

$$
o_t
=
\frac{
\phi(q_t)^\top \left(\sum_{i\le t}\phi(k_i)v_i^\top\right)
}{
\phi(q_t)^\top \left(\sum_{i\le t}\phi(k_i)\right)
}
$$

现在可以自然定义两个前缀状态：

$$
S_t := \sum_{i\le t}\phi(k_i)v_i^\top,\qquad
z_t := \sum_{i\le t}\phi(k_i)
$$

代回去得到：

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
$$

到这里，公式已经从“访问全部历史”变成“读取当前状态”。接着再看为什么它天然是递推的。

由于 $S_t$ 和 $z_t$ 都是前缀和，所以有：

$$
S_t = S_{t-1} + \phi(k_t)v_t^\top,\qquad
z_t = z_{t-1} + \phi(k_t)
$$

这一步就是全文最核心的地方。原本因果 Mask 的含义是“只对 $i \le t$ 的历史求和”；现在这个限制已经被吸收进前缀状态里了，因为：

- $S_t$ 只累计到第 $t$ 步；
- $z_t$ 也只累计到第 $t$ 步；
- 所以读取 $S_t,z_t$ 时，天然不会访问未来。

也就是说，因果 Mask 不再需要显式写成一个下三角矩阵，而是变成了“状态只沿时间向前累积”的递推约束。

### 为什么需要两个状态，而不是一个

很多新手第一次看公式时会问：为什么只保存 $S_t$ 不够？

原因是：

- $S_t$ 负责分子，累计的是“key 特征写入 value 的结果”；
- $z_t$ 负责分母，累计的是“所有历史 key 特征的总权重”。

如果只保留 $S_t$，输出会退化成未归一化的加权和：

$$
o_t' = \phi(q_t)^\top S_t
$$

这时输出尺度会随着历史长度增长而漂移，不再对应归一化注意力。保留 $z_t$ 的目的，就是恢复“归一化读取”的语义。

### 维度检查

这一步很重要，能避免很多实现错误。

设：

- $\phi(q_t), \phi(k_t) \in \mathbb{R}^{d_\phi}$
- $v_t \in \mathbb{R}^{d_v}$

则：

- $\phi(k_t)v_t^\top \in \mathbb{R}^{d_\phi \times d_v}$
- 所以 $S_t \in \mathbb{R}^{d_\phi \times d_v}$
- $\phi(q_t)^\top S_t \in \mathbb{R}^{d_v}$
- $\phi(q_t)^\top z_t \in \mathbb{R}$

因此：

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
\in \mathbb{R}^{d_v}
$$

维度完全闭合。

### 玩具例子

下面给一个完整的手算例子。取特征维度 $d_\phi=2$，并令 $\phi(x)=x$，即恒等映射。设初始状态为：

$$
S_0=\begin{bmatrix}0&0\\0&0\end{bmatrix},\qquad
z_0=\begin{bmatrix}0\\0\end{bmatrix}
$$

第 1 步输入：

$$
k_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
v_1=\begin{bmatrix}0\\1\end{bmatrix},\quad
q_1=\begin{bmatrix}1\\1\end{bmatrix}
$$

先更新状态。由于

$$
k_1v_1^\top
=
\begin{bmatrix}1\\0\end{bmatrix}
\begin{bmatrix}0&1\end{bmatrix}
=
\begin{bmatrix}0&1\\0&0\end{bmatrix}
$$

所以：

$$
S_1=S_0+k_1v_1^\top
=
\begin{bmatrix}0&1\\0&0\end{bmatrix}
$$

同时：

$$
z_1=z_0+k_1=\begin{bmatrix}1\\0\end{bmatrix}
$$

再计算输出：

$$
o_1=\frac{q_1^\top S_1}{q_1^\top z_1}
=
\frac{
[1,1]
\begin{bmatrix}0&1\\0&0\end{bmatrix}
}{
[1,1]
\begin{bmatrix}1\\0\end{bmatrix}
}
=
\frac{[0,1]}{1}
=
[0,1]
$$

这一步的解释是：历史里只有一个 value，因此输出自然就是它自己。

第 2 步取：

$$
k_2=\begin{bmatrix}0\\1\end{bmatrix},\quad
v_2=\begin{bmatrix}1\\0\end{bmatrix}
$$

则新的外积为：

$$
k_2v_2^\top
=
\begin{bmatrix}0\\1\end{bmatrix}
\begin{bmatrix}1&0\end{bmatrix}
=
\begin{bmatrix}0&0\\1&0\end{bmatrix}
$$

于是：

$$
S_2=S_1+k_2v_2^\top
=
\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad
z_2=\begin{bmatrix}1\\1\end{bmatrix}
$$

现在假设第 2 步的 query 取为：

$$
q_2=\begin{bmatrix}2\\1\end{bmatrix}
$$

则：

$$
q_2^\top S_2
=
[2,1]
\begin{bmatrix}0&1\\1&0\end{bmatrix}
=
[1,2]
$$

$$
q_2^\top z_2
=
[2,1]
\begin{bmatrix}1\\1\end{bmatrix}
=
3
$$

所以输出为：

$$
o_2=\frac{[1,2]}{3}=\left[\frac13,\frac23\right]
$$

这个结果说明两件事：

1. 输出是由当前 query 对历史状态进行读取得到的；
2. 读取时已经不需要显式访问 $\{(k_1,v_1),(k_2,v_2)\}$ 的原始列表。

### 复杂度从哪里变化

设：

- 特征维度为 $d_\phi$；
- value 维度为 $d_v$；
- 当前长度为 $t$。

那么在单步推理中：

| 方法 | 单步主要计算 | 单步存储 |
|---|---|---|
| 标准因果注意力 | 与全部历史 $t$ 个 key/value 交互 | $O(td_k + td_v)$ |
| 因果线性注意力 | 更新 $S_t,z_t$ 并读取一次状态 | $O(d_\phi d_v)$ |

更具体地说：

- 更新 $S_t$ 需要一次外积，成本约为 $O(d_\phi d_v)$；
- 更新 $z_t$ 需要一次向量加法，成本约为 $O(d_\phi)$；
- 读取输出 $\phi(q_t)^\top S_t$ 也是 $O(d_\phi d_v)$；
- 分母 $\phi(q_t)^\top z_t$ 是 $O(d_\phi)$。

因此单步主成本通常记为：

$$
O(d_\phi d_v)
$$

如果常见情况下 $d_\phi \approx d_v \approx d$，那么也常写作：

$$
O(d^2)
$$

所以它和 RNN 的相似点不是“结构长得像”，而是“都维护固定大小隐状态，并按时间递推”。

---

## 代码实现

下面给一个最小但完整、可以直接运行的 Python 示例。它做四件事：

1. 定义一个简单的非负特征映射 $\phi$；
2. 用递推形式计算因果线性注意力；
3. 用直接定义式逐项求和做对照；
4. 验证两者数值一致。

```python
import numpy as np


def phi(x: np.ndarray) -> np.ndarray:
    """
    一个简单的非负特征映射。
    这里用 elu + 1 的平滑变体，避免分母容易出现负值或过小。
    """
    return np.where(x > 0.0, x + 1.0, np.exp(x))


def linear_attention_recurrent(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    eps: float = 1e-8,
):
    """
    因果线性注意力的递推实现。

    Q: [T, d]
    K: [T, d]
    V: [T, dv]
    返回:
        O: [T, dv]
        S_hist: [T, d, dv]
        z_hist: [T, d]
    """
    T, d = K.shape
    dv = V.shape[1]

    S = np.zeros((d, dv), dtype=np.float64)
    z = np.zeros(d, dtype=np.float64)

    O = np.zeros((T, dv), dtype=np.float64)
    S_hist = np.zeros((T, d, dv), dtype=np.float64)
    z_hist = np.zeros((T, d), dtype=np.float64)

    for t in range(T):
        kt = phi(K[t])
        qt = phi(Q[t])

        S += np.outer(kt, V[t])
        z += kt

        numer = qt @ S          # shape: [dv]
        denom = qt @ z + eps    # scalar

        O[t] = numer / denom
        S_hist[t] = S
        z_hist[t] = z

    return O, S_hist, z_hist


def linear_attention_direct(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    eps: float = 1e-8,
):
    """
    按定义直接计算:
        o_t = sum_{i<=t} (phi(q_t)^T phi(k_i)) v_i / sum_{i<=t} phi(q_t)^T phi(k_i)
    """
    T, _ = K.shape
    dv = V.shape[1]
    O = np.zeros((T, dv), dtype=np.float64)

    for t in range(T):
        qt = phi(Q[t])
        numer = np.zeros(dv, dtype=np.float64)
        denom = 0.0

        for i in range(t + 1):
            ki = phi(K[i])
            w = qt @ ki
            numer += w * V[i]
            denom += w

        O[t] = numer / (denom + eps)

    return O


def main():
    Q = np.array([
        [1.0, 1.0],
        [0.5, 2.0],
        [2.0, 0.1],
    ], dtype=np.float64)

    K = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float64)

    V = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 3.0],
    ], dtype=np.float64)

    o_rec, S_hist, z_hist = linear_attention_recurrent(Q, K, V)
    o_dir = linear_attention_direct(Q, K, V)

    print("recurrent output:")
    print(o_rec)
    print()

    print("direct output:")
    print(o_dir)
    print()

    print("allclose:", np.allclose(o_rec, o_dir, atol=1e-8))
    print()

    print("final S_T:")
    print(S_hist[-1])
    print()

    print("final z_T:")
    print(z_hist[-1])


if __name__ == "__main__":
    main()
```

如果运行正常，`allclose` 应输出 `True`。这说明“递推形式”和“直接按因果定义逐项求和”在数值上是一致的。

上面代码里几个实现点值得单独说明。

### 1. 为什么先更新状态，再读输出

在本文的定义里：

$$
S_t=\sum_{i\le t}\phi(k_i)v_i^\top,\qquad z_t=\sum_{i\le t}\phi(k_i)
$$

也就是说，第 $t$ 个位置可以看到自己，所以必须先把第 $t$ 个 token 的 $(k_t,v_t)$ 写入状态，再用 $q_t$ 读取。  
如果你的任务定义成“当前位置不能看自己，只能看过去”，那就应先读后写，但那已经不是标准自回归 attention 的常见定义了。

### 2. 为什么分母要加 $\varepsilon$

因为分母是：

$$
\phi(q_t)^\top z_t
$$

如果它非常小，数值会不稳定。实现里通常都会加一个很小的常数：

$$
\phi(q_t)^\top z_t + \varepsilon
$$

这不是在改变核心机制，而是在防止浮点除法爆炸。

### 3. 推理时真正需要保存什么

在线推理时，不需要保存整段 `S_hist` 和 `z_hist`。  
那只是为了教学和调试才保留的。部署时通常只保留当前的：

- `S`
- `z`

每来一个新 token，做一次更新，然后立刻输出。

推理伪代码可以压缩成：

```python
kt = phi(k_t)
qt = phi(q_t)

S += outer(kt, v_t)
z += kt
o_t = (qt @ S) / (qt @ z + eps)
```

### 4. 训练为什么仍然可以并行

虽然推理形式看起来像 RNN，但训练时不必真的用 Python `for` 循环逐步跑完整个序列。因为：

$$
\Delta S_t=\phi(k_t)v_t^\top,\qquad
\Delta z_t=\phi(k_t)
$$

而：

$$
S_t=\sum_{i\le t}\Delta S_i,\qquad
z_t=\sum_{i\le t}\Delta z_i
$$

这两个量都是前缀和。前缀和的特点是可以用并行 scan 实现，因此在 GPU 上仍然可以把整段序列并行化处理。  
这就是“推理像 RNN，训练仍可并行”的具体原因，不是口号，而是由前缀和结构直接决定的。

### 5. 一个批量版本的张量写法

为了让新手更容易把单序列代码和深度学习框架里的 batch 版本对应起来，可以把张量维度写清楚。设：

- $Q,K \in \mathbb{R}^{B \times T \times d_\phi}$
- $V \in \mathbb{R}^{B \times T \times d_v}$

则每个位置的增量是：

$$
\Delta S_{b,t} = \phi(K_{b,t}) V_{b,t}^\top \in \mathbb{R}^{d_\phi \times d_v}
$$

把所有时间步的增量堆叠后，就可以沿时间维做 prefix scan，得到每个 $t$ 对应的 $S_t,z_t$。  
这一步在概念上和 RNN 很像，但在训练实现上更接近“结构化前缀累计”。

---

## 工程权衡与常见坑

线性注意力的递推形式很简洁，但真正落地时，问题通常不在公式本身，而在数值稳定性、状态容量和硬件实现。

先看一张总表：

| 选择 | 稳定性 | 训练表现 | 常见问题 |
|---|---|---|---|
| 保留分母 $\phi(q_t)^\top z_t$ | 更接近归一化注意力 | 通常更合理 | 分母过小会导致数值爆炸 |
| 去掉分母，仅用 $\phi(q_t)^\top S_t$ | 实现简单 | 常偏离目标机制 | 输出尺度随历史长度漂移 |
| $\phi$ 输出非负 | 分母更可控 | 通常更稳定 | 表达能力受特征形式约束 |
| $\phi$ 可正可负 | 更灵活 | 训练更敏感 | 分母可能相互抵消接近 0 |

下面把常见坑拆开说。

### 坑 1：把“线性注意力”误解成“直接删 softmax”

很多实现上的错误都从这里开始。  
线性注意力不是简单地把 softmax 去掉，变成：

$$
o_t = \sum_{i\le t}(q_t^\top k_i)v_i
$$

如果这样写，既没有归一化，也没有把相似度改写成可递推的核分解。它只是一个未归一化的加权和，输出尺度和序列长度强相关，通常不是想要的目标。

真正关键的是两步：

1. 把相似度写成 $\phi(q)^\top \phi(k)$；
2. 显式保留归一化项 $\phi(q_t)^\top z_t$。

### 坑 2：分母存在，但没有做数值保护

如果 $\phi(q_t)^\top z_t$ 很小，那么：

- 输出会突然变大；
- 梯度也可能异常放大；
- 半精度训练时更容易出问题。

常见处理包括：

| 方法 | 作用 | 代价 |
|---|---|---|
| 加 $\varepsilon$ | 防止除零或过小分母 | 只能缓解，不能根治 |
| 选非负 $\phi$ | 降低分母抵消风险 | 限制特征映射形式 |
| 用更高精度累计 $S,z$ | 减少数值误差 | 增加显存或算力 |
| 对状态做缩放/归一化 | 控制幅度增长 | 增加实现复杂度 |

对新手而言，最容易忽略的是：即使公式正确，低精度累计也可能导致结果明显偏移。尤其在长序列下，$S_t,z_t$ 都是不断累加的，误差会逐步积累。

### 坑 3：误以为固定状态等于无损记忆

这是概念层面最常见的误解。

标准注意力保留的是全部历史明细，因此理论上可以对任意位置做精确内容匹配。  
线性注意力保留的是固定维度状态，因此它更像“把历史压成若干统计量”。

这意味着：

- 若任务只需要某种整体性、平滑性、低秩式的历史读取，固定状态可能足够；
- 若任务强依赖精确位置、稀疏检索或复杂的 pairwise 结构，固定状态可能不够。

可以用一个极端例子理解。  
如果历史里有两个非常接近、但必须严格区分的长程线索，而状态维度又很小，那么它们可能被压进相似的统计方向，导致读取时混淆。

### 坑 4：只看渐近复杂度，不看硬件常数

理论上从“和历史长度 $t$ 交互”变成“和固定状态交互”，看起来一定更快。但工程上并不是所有长度区间都如此。

真实速度取决于：

- 上下文长度是否已经足够大；
- $d_\phi$ 和 $d_v$ 是否偏大；
- 外积和状态读取是否能高效映射到硬件；
- 原系统是否已经对 KV cache 做了很强的 kernel 优化。

因此更准确的说法不是“线性注意力一定更快”，而是：

- 当上下文足够长，且 KV cache 成为瓶颈时，固定状态的优势更容易体现；
- 当上下文较短、算子实现不成熟或状态维度较大时，理论优势未必直接变成端到端吞吐优势。

### 坑 5：忽略训练和推理的优化目标并不相同

线性注意力常被同时宣传为“训练可并行”和“推理存储固定”。这两点都对，但优化对象不同：

- 训练更关心整段序列的并行吞吐、显存和反向传播；
- 推理更关心单步延迟、缓存大小和在线更新成本。

同一个公式，在训练和推理阶段的最佳实现路径可能完全不同。  
如果把训练版 scan 实现直接搬到流式推理里，或者反过来把推理版 Python 循环当成训练实现，性能都会很差。

### 坑 6：忘记多头场景下状态是“每个头各自维护”

在真实 Transformer 里，attention 通常是多头的。  
这意味着不是维护一个全局 $S,z$，而是每个头都有自己的状态：

$$
S_t^{(h)},\ z_t^{(h)}
$$

若头数为 $H$，则总状态成本通常是：

- $H \times d_\phi \times d_v$ for $S$
- $H \times d_\phi$ for $z$

因此“固定状态”不等于“状态很小”，而是“不随上下文长度增长”。  
如果头数多、每头维度大，总状态仍然可能很可观，只是它的规模不再依赖序列长度。

---

## 替代方案与适用边界

线性注意力不是唯一一种把序列处理改写成固定状态递推的方法。更大的背景是：很多看起来不同的序列模型，本质上都在回答同一个问题，即“如何用有限状态表示历史，并支持对当前输入做条件化读取”。

先把几个视角放进一张表里：

| 视角 | 线性注意力 | SSM / SSD 视角 |
|---|---|---|
| 历史记忆 | $S_t,z_t$ | 隐状态 $h_t$ |
| 更新方式 | 外积累加或带衰减递推 | 状态转移 + 输入注入 |
| 输出方式 | query 读取状态 | 读出矩阵读取状态 |
| 并行训练 | scan / prefix sum | 结构化矩阵或 scan |
| 低延迟推理 | 天然支持 | 天然支持 |

### 和标准注意力的区别

标准 softmax 注意力的核心特征是“显式两两交互再归一化”。  
线性注意力则要求相似度可分解，因此它更像：

1. 先把历史写入一个有限状态；
2. 再用当前 query 读这个状态。

两者都叫 attention，但计算路径不同。  
所以更准确的理解是：线性注意力不是“把 attention 近似成 RNN”，而是“在可分解核前提下，attention 本来就能重写成递推形式”。

### 和 RNN 的关系

它和 RNN 的共同点是都维护固定大小状态。  
差异在于读取方式：

- 传统 RNN 常由隐状态直接产生输出；
- 线性注意力则保留了“query 条件化读取历史”的结构。

也就是说，同样是状态模型，线性注意力不是盲目把所有历史混成一个向量，而是允许当前位置的 query 决定如何读这个状态。这一点是它比普通 RNN 更接近 attention 的地方。

### 和 SSM / SSD 的关系

状态空间模型（SSM）也用递推形式处理序列，典型写法是：

$$
h_t = A_t h_{t-1} + B_t x_t,\qquad
y_t = C_t h_t
$$

这里：

- $h_t$ 是隐状态；
- $A_t$ 是状态转移；
- $B_t$ 把当前输入写进状态；
- $C_t$ 从状态中读出输出。

在线性注意力里，$\phi(k_t)v_t^\top$ 可以看作一种“写入状态”的低秩更新，而 $\phi(q_t)$ 则扮演读取器。  
在 SSD 这类统一视角下，一部分结构化 SSM 与带特定 Mask 的线性注意力，本质上实现的是同一类序列变换，只是：

- 一种写成显式递推；
- 另一种写成结构化 attention 或 masked matrix form。

这也是为什么近年的一些工作会把 attention 和 SSM 放到同一框架下讨论。

### 什么场景适合

适合的情况通常有三类：

1. 自回归生成或流式处理，对单步延迟和缓存大小敏感。
2. 上下文很长，KV cache 的显存和带宽已经成为系统瓶颈。
3. 任务允许把历史压缩成有限维统计，而不是要求精确保留所有 pairwise 关系。

典型例子包括：

- 长上下文文本生成；
- 在线语音流处理；
- 实时日志/事件序列检测；
- 边生成边消费的服务端推理系统。

### 什么场景要谨慎

不适合或者至少要谨慎评估的情况也很明确：

1. 你需要尽量贴近标准 softmax 注意力的精确行为。
2. 任务强依赖稀疏、精确、远距离的定位式检索。
3. 状态维度预算有限，但任务本身的信息密度很高。
4. 当前系统已经对 KV cache 做了高度优化，且上下文长度没有大到让缓存真正成为瓶颈。

因此，线性注意力不是“全面替代 attention”的通用答案，而是某类约束下的有效改写。

### 一个更准确的结论

这一节最该记住的不是“线性注意力像 RNN”，而是下面这句话：

在可分解核的条件下，因果 attention 可以重写成固定状态递推；因此它同时具备 attention 的条件读取结构和状态模型的流式更新形式。

这比单纯说“更省缓存”更准确，也更接近它的数学本质。

---

## 参考资料

1. Katharopoulos et al., 2020, *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*  
   重点：经典线性注意力论文，直接给出将因果线性注意力写成递推形式的方法，是 $S_t,z_t$ 写法的代表性来源。  
   链接：https://arxiv.org/abs/2006.16236

2. LION Blog, 2025, *Part I - Full Linear Attention*  
   重点：系统讲解全线性注意力、因果递推形式以及 masked form 与 recurrent form 的对应关系。  
   链接：https://lions-epfl.github.io/2025/lion-part1-model/

3. Victor Fiz, 2025, *Memory in Transformers (1): Linear Attention*  
   重点：从“记忆容量应按信息量而非上下文长度增长”的角度解释线性注意力，并讨论归一化与数值稳定性。  
   链接：https://victorfiz.com/blog/2025/02/02/memory-in-transformers-1.html

4. Emergent Mind, 2026, *Structured State-Space Duality (SSD)*  
   重点：总结 SSM 与 semiseparable masked attention 的统一视角，帮助理解线性注意力与状态空间模型的关系。  
   链接：https://www.emergentmind.com/topics/structured-state-space-duality-ssd

5. Hu et al., 2025, *On Structured State-Space Duality*  
   重点：形式化推广 SSD，讨论对角 SSM 与特定 masked attention 结构的等价条件，并指出 softmax attention 不属于该结构族。  
   链接：https://www.emergentmind.com/papers/2510.04944
