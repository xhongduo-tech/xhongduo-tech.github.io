## 核心结论

RoPE，Rotary Position Embedding，中文常译为“旋转位置编码”。它的核心做法不是把一个“位置向量”直接加到 token 表示上，而是把向量的每一对维度看成一个二维平面中的点，再按照位置对应的角度做旋转。

可以先记住两个结论。

第一，RoPE 把位置信息直接写进了 Query 和 Key 的几何关系里，因此注意力分数天然携带相对位置信息。更准确地说，如果第 $m$ 个位置的 Query 记为 $f_q(x_m,m)$，第 $n$ 个位置的 Key 记为 $f_k(x_n,n)$，那么它们的内积只依赖位置差 $m-n$，而不依赖单独的 $m,n$。这正是 Transformer 建模“谁离谁更近、谁和谁相隔更远”时最有用的性质。

第二，RoPE 使用的是旋转矩阵。旋转矩阵只改变方向，不改变长度，因此向量范数保持不变。范数就是向量长度。长度不变的直接好处是：位置编码不会把向量越转越大，也不会越转越小，这对长序列推理和 KV-cache 尤其重要。

先看三类常见方案在结论层面的差异：

| 方案 | 位置信息注入方式 | 注意力是否天然依赖相对位置 | 是否保持范数 | 对 KV-cache 是否友好 |
|---|---|---:|---:|---:|
| 绝对位置编码加法 | 将位置向量加到输入上 | 否 | 不保证 | 中 |
| 相对位置 bias | 在 attention score 上加偏置 | 是 | 与输入几何无关 | 中 |
| RoPE | 对 Q/K 做按位置旋转 | 是 | 是 | 高 |

RoPE 的一个紧凑写法是：

$$
f(x_m,m)=x_m \cdot e^{im\theta_i}
$$

这里把每一对维度看成一个复数。复数可以理解成“二维向量的另一种写法”：实部对应一个轴，虚部对应另一个轴。这样一来，二维旋转就可以写成乘上一个单位复数 $e^{im\theta_i}$。

最终在 attention 中会出现：

$$
\langle f_q(x_m,m), f_k(x_n,n)\rangle
= \sum_i \Re\left(q_i\overline{k_i}\,e^{i(m-n)\theta_i}\right)
$$

这条式子的关键信息只有一句话：位置只以差值 $m-n$ 的形式进入点积。

一个适合新手的直观理解是：把每两个维度看成一支二维小箭头。位置是 5，就把箭头转到第 5 个角度；位置是 8，就把箭头转到第 8 个角度。两支箭头做点积时，真正影响结果的是它们相差多少角度，因此注意力天然更关心“相距多远”。

再看一个极简例子。假设某一对维度原本是向右的箭头，位置 2 旋转 $2\theta$，位置 5 旋转 $5\theta$。两者相似度受影响的不是“2”与“5”这两个编号本身，而是角度差 $3\theta$。这就是“相对位置”被编码进注意力的最直观含义。

---

## 问题定义与边界

Transformer 的自注意力机制本身并不知道顺序。它擅长比较“内容是否相关”，但如果不给位置信息，它并不知道谁在前、谁在后。因此，“猫追狗”和“狗追猫”可能会在某些层面显得过于接近，因为模型看到的是一组内容向量，而不是带有明确顺序的序列。

所以问题不只是“让模型知道第几个 token 在哪里”，而是更进一步：

$$
\text{我们希望 attention score 本身就能感知相对距离。}
$$

例如，第 5 个 token 去看第 8 个 token，模型通常更关心“它们相差 3”，而不是“5”这个编号和“8”这个编号本身分别意味着什么”。

RoPE 的问题边界可以概括为：

| 问题 | RoPE 是否直接解决 | 说明 |
|---|---:|---|
| 表达序列顺序 | 是 | 通过位置相关旋转编码顺序 |
| 让注意力依赖相对距离 | 是 | 点积项中只保留 $m-n$ |
| 保持向量长度稳定 | 是 | 旋转矩阵正交，范数不变 |
| 直接修改 Value 语义 | 否 | RoPE 通常不作用于 V |
| 无限长度外推 | 否 | 能改善长上下文行为，但不代表无限泛化 |
| 自动解决长文本退化 | 否 | 长上下文仍受训练长度、频率设计、数据分布影响 |

对第 $m$ 个位置的输入向量 $x_m$，RoPE 不是做

$$
x_m + p_m
$$

而是定义

$$
f_q(x_m,m)=R_m x_m,\qquad f_k(x_n,n)=R_n x_n
$$

其中 $R_m$ 是由位置 $m$ 决定的分块旋转矩阵。

如果改用复数写法，可以写成：

$$
f(x_m,m)=x_m\cdot e^{im\theta_i}
$$

这里的 $\theta_i$ 是第 $i$ 组二维子空间的频率，标准形式通常写成：

$$
\theta_i=\frac{1}{10000^{2i/d}}
\qquad (i=0,1,\dots,d/2-1)
$$

这表示不同维度对位置的敏感程度不同。低编号维度变化更快，高编号维度变化更慢。可以把它理解成一组不同频率的“位置时钟”：有的时钟转得快，擅长表达近距离变化；有的时钟转得慢，擅长表达远距离模式。

一个最小边界例子如下：

- 绝对位置编码会分别给位置 5 和位置 8 附加两个不同的编码。
- RoPE 则把位置 5 和位置 8 映射成两个不同旋转角，并在点积时自动体现“它们相差 3”。

这就是 RoPE 的边界：它不是给位置打标签，而是把位置差写进相似度的几何结构。

---

## 核心机制与推导

RoPE 的核心机制是把 $d$ 维向量拆成 $d/2$ 个二维子空间，每个二维子空间独立旋转。

设某一对维度是 $(x_{2i},x_{2i+1})$，写成复数就是：

$$
z_i=x_{2i}+ix_{2i+1}
$$

对位置 $m$ 进行编码后：

$$
z_i' = z_i \, e^{im\theta_i}
$$

这表示原本的二维向量 $(x_{2i},x_{2i+1})$ 被旋转了 $m\theta_i$。

如果写成实数矩阵形式，就是更熟悉的二维旋转矩阵：

$$
\begin{bmatrix}
x'_{2i}\\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}
$$

展开后可得：

$$
x'_{2i}=x_{2i}\cos(m\theta_i)-x_{2i+1}\sin(m\theta_i)
$$

$$
x'_{2i+1}=x_{2i}\sin(m\theta_i)+x_{2i+1}\cos(m\theta_i)
$$

这就是工程实现里最常见的两行公式。

### 1. 为什么范数不变

旋转矩阵满足：

$$
R_m^\top R_m = I
$$

因此对于任意向量 $x$，

$$
\|R_m x\|_2^2 = (R_m x)^\top (R_m x)=x^\top R_m^\top R_m x=x^\top x=\|x\|_2^2
$$

也就是说：

$$
\|R_m x\|_2=\|x\|_2
$$

这个性质不是“数学上好看而已”，而是直接影响工程稳定性。位置变大时，如果编码方式会持续放大或压缩向量，那么长序列下的注意力分数分布就会被位置本身扭曲。RoPE 避免了这一点。

### 2. 为什么点积只依赖相对位置

设某一对维度中，Query 对应复数 $q_i$，Key 对应复数 $k_i$，位置分别为 $m,n$。旋转后：

$$
q_i' = q_i e^{im\theta_i},\qquad
k_i' = k_i e^{in\theta_i}
$$

二维点积可以写成复数形式的实部：

$$
\langle q_i',k_i'\rangle = \Re(q_i'\overline{k_i'})
$$

代入后得到：

$$
\Re\left(q_i e^{im\theta_i}\overline{k_i e^{in\theta_i}}\right)
=
\Re\left(q_i\overline{k_i}e^{i(m-n)\theta_i}\right)
$$

注意这里的 $\overline{k_i}$ 表示复共轭，即虚部取负号。这样写的好处是，可以把二维点积压缩成一个非常紧凑的式子。

对所有维度对求和，就有：

$$
\langle f_q(x_m,m),f_k(x_n,n)\rangle
=
\sum_i \Re\left(q_i\overline{k_i}e^{i(m-n)\theta_i}\right)
$$

关键点到这里已经完全清楚：点积中的位置信息只通过 $m-n$ 出现。

### 3. 用实数形式再看一遍

对新手来说，上面的复数推导可能有点抽象。可以只看二维实数情形。

令：

$$
q=[q_0,q_1],\qquad k=[k_0,k_1]
$$

经过位置 $m,n$ 的旋转后，分别得到：

$$
q' = R_m q,\qquad k' = R_n k
$$

则有：

$$
(q')^\top k' = q^\top R_m^\top R_n k
$$

由于二维旋转矩阵满足：

$$
R_m^\top = R_{-m},\qquad R_{-m}R_n = R_{n-m}
$$

所以：

$$
(q')^\top k' = q^\top R_{n-m}k
$$

这说明旋转后的点积，本质上等价于“让一个向量相对另一个向量旋转了 $(n-m)\theta$ 后再做点积”。因此决定相似度的不是各自绝对位置，而是相对位移。

### 4. 玩具例子

设 $d=2$，只有一组二维向量，并取 $\theta_0=1$。令：

$$
q=[1,0],\quad k=[0,1],\quad m=1,\quad n=3
$$

则旋转后：

$$
q'=[\cos 1,\ \sin 1]
$$

$$
k'=[-\sin 3,\ \cos 3]
$$

点积为：

$$
\langle q',k'\rangle
=
\cos1(-\sin3)+\sin1\cos3
=
\sin(1-3)
$$

也就是：

$$
\langle q',k'\rangle = \sin(m-n)
$$

注意这里出现的是 $m-n$，而不是单独的 $m$ 和 $n$。

再做一组位置替换，例如改成 $(m,n)=(2,4)$。由于位置差仍然是 $-2$，点积也保持不变。这就是“只依赖相对位置差”的直接数值体现。

### 5. 多频率结构为什么有必要

如果所有维度都用同一个频率，那么模型只能在一个固定尺度上感知位置差。RoPE 使用一组频率：

| 维度对 | 复数表示 | 频率 $\theta_i$ | 位置 $m$ 的旋转角 |
|---|---|---|---|
| $(0,1)$ | $x_0+ix_1$ | $10000^{-0/d}$ | $m\theta_0$ |
| $(2,3)$ | $x_2+ix_3$ | $10000^{-2/d}$ | $m\theta_1$ |
| $(4,5)$ | $x_4+ix_5$ | $10000^{-4/d}$ | $m\theta_2$ |
| $\dots$ | $\dots$ | $\dots$ | $\dots$ |

这相当于模型同时拥有多组不同尺度的位置参照系。高频维度善于区分邻近位置，低频维度更适合表达更远的距离关系。多频率叠加后，模型才能同时处理局部顺序和全局跨度。

---

## 代码实现

工程实现里通常不会真的把张量转成复数再计算，而是直接使用 `cos/sin` 表对偶数维和奇数维做成对旋转。原因主要有三点：

1. 更容易和现有张量布局兼容。
2. 更适合批量计算。
3. 便于和多头注意力、KV-cache 结合。

下面给出一个可直接运行的 Python 示例，演示四件事：

1. 如何构造频率和 `cos/sin` 缓存。
2. 如何对单个向量应用 RoPE。
3. 如何对批量张量应用 RoPE。
4. 如何验证“相同位置差得到相同点积”和“范数保持不变”。

```python
import numpy as np


def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0):
    """
    返回:
        cos, sin: shape [seq_len, dim // 2]
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2

    pair_idx = np.arange(half, dtype=np.float64)
    theta = 1.0 / (base ** (2.0 * pair_idx / dim))   # [half]

    positions = np.arange(seq_len, dtype=np.float64)[:, None]   # [seq_len, 1]
    angles = positions * theta[None, :]                         # [seq_len, half]

    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin


def apply_rope_1d(x: np.ndarray, cos: np.ndarray, sin: np.ndarray):
    """
    x:   [dim]
    cos: [dim // 2]
    sin: [dim // 2]
    """
    assert x.ndim == 1, "x must be 1D"
    dim = x.shape[0]
    assert dim % 2 == 0, "dim must be even"

    x_even = x[0::2]
    x_odd = x[1::2]

    out = np.empty_like(x, dtype=np.float64)
    out[0::2] = x_even * cos - x_odd * sin
    out[1::2] = x_even * sin + x_odd * cos
    return out


def apply_rope_batch(x: np.ndarray, cos: np.ndarray, sin: np.ndarray):
    """
    x:   [seq_len, dim]
    cos: [seq_len, dim // 2]
    sin: [seq_len, dim // 2]
    """
    assert x.ndim == 2, "x must be [seq_len, dim]"
    seq_len, dim = x.shape
    assert dim % 2 == 0, "dim must be even"
    assert cos.shape == (seq_len, dim // 2)
    assert sin.shape == (seq_len, dim // 2)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    out = np.empty_like(x, dtype=np.float64)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def main():
    # 1) 验证相同相对位置差 => 相同点积
    q = np.array([1.0, 0.0])
    k = np.array([0.0, 1.0])

    cos, sin = build_rope_cache(seq_len=8, dim=2)

    q_m1 = apply_rope_1d(q, cos[1], sin[1])
    k_n3 = apply_rope_1d(k, cos[3], sin[3])
    score_13 = float(q_m1 @ k_n3)

    q_m2 = apply_rope_1d(q, cos[2], sin[2])
    k_n4 = apply_rope_1d(k, cos[4], sin[4])
    score_24 = float(q_m2 @ k_n4)

    assert np.allclose(score_13, score_24, atol=1e-6)

    # 2) 验证范数保持不变
    x = np.array([0.3, -0.7, 1.2, 0.5], dtype=np.float64)
    cos4, sin4 = build_rope_cache(seq_len=5, dim=4)
    x_rot = apply_rope_1d(x, cos4[3], sin4[3])

    assert np.allclose(np.linalg.norm(x), np.linalg.norm(x_rot), atol=1e-6)

    # 3) 演示批量旋转
    X = np.array([
        [1.0, 0.0, 0.5, -0.5],
        [0.2, 0.8, -1.0, 2.0],
        [0.0, 1.0, 1.5, 0.3],
    ], dtype=np.float64)

    cos_b, sin_b = build_rope_cache(seq_len=3, dim=4)
    X_rot = apply_rope_batch(X, cos_b, sin_b)

    print("score_13 =", score_13)
    print("score_24 =", score_24)
    print("norm(x) =", np.linalg.norm(x))
    print("norm(x_rot) =", np.linalg.norm(x_rot))
    print("X_rot =")
    print(X_rot)


if __name__ == "__main__":
    main()
```

这段代码可以直接运行。预期现象有两个：

- `score_13` 与 `score_24` 基本相等，因为两组位置差都等于 2。
- `norm(x)` 与 `norm(x_rot)` 基本相等，因为旋转不改变范数。

### 训练阶段如何使用

训练时通常会为整段序列一次性构造 `cos/sin` 表，然后把每个位置上的 Query 和 Key 批量旋转。简化后的逻辑是：

```python
cos, sin = build_rope_cache(seq_len, head_dim)

q = apply_rope_batch(q, cos, sin)
k = apply_rope_batch(k, cos, sin)
```

如果是多头注意力，真实张量形状往往是：

$$
[\text{batch},\ \text{heads},\ \text{seq\_len},\ \text{head\_dim}]
$$

此时做法通常是沿着最后一维按 `(0,1),(2,3),\dots` 配对，再把 `cos/sin` 广播到 batch 和 heads 维度上。

### 推理阶段为什么特别适合 RoPE

推理时会用到 KV-cache。KV-cache 的含义是：历史 token 的 Key 和 Value 已经算过，就直接缓存起来，后续生成新 token 时不再重算历史部分。

RoPE 对推理友好的原因是：

- 它只作用于 Q 和 K，不作用于 V。
- 历史 Key 在被缓存时已经绑定了自己的绝对位置。
- 生成新 token 时，只需要给“当前这个新位置”的 Query 和 Key 使用正确角度即可。

训练与推理的差异可以总结为：

| 阶段 | Q 是否旋转 | K 是否旋转 | V 是否旋转 | 是否缓存 | 角度如何取得 |
|---|---:|---:|---:|---:|---|
| 训练 | 是 | 是 | 否 | 一般不需要 | 按整段序列预生成 |
| 推理首轮 | 是 | 是 | 否 | 是 | 按 prompt 长度生成 |
| 推理续写 | 是 | 新 token 的 K 是 | 否 | 复用旧 K/V | 按真实绝对位置继续取 |

举一个实际场景。假设 prompt 长度已经是 4096，模型现在要生成第 4097 个 token。此时新 token 使用的旋转角必须对应位置 4096。如果错误地从位置 0 重新取角度，那么模型就会把这个 token 误认为接近序列开头，导致后续 attention 的相对位置关系整体错位。

这也是为什么许多线上问题并不是“RoPE 数学错了”，而是“推理时位置偏移没对齐”。

---

## 工程权衡与常见坑

RoPE 的工程优势很明确，但真正落地时，问题也高度集中。大多数 bug 都不是出在公式推导，而是出在实现细节、缓存逻辑和超参数一致性上。

第一，只对 Q/K 应用，不对 V 应用。  
Value 是被注意力权重加权汇总的内容载体。如果对 V 也做旋转，相当于把“内容本身”按位置扭曲。这样即便 attention 权重算对了，读出的内容也会被位置污染。

第二，频率基数 `base`、头维 `head_dim` 和维度配对方式必须一致。  
RoPE 不是一个“松散可替换”的装饰层。只要训练和推理在其中任何一项上不一致，位置角度就会解释错位。

第三，KV-cache 必须与绝对位置同步。  
缓存不是只存张量就够了，还要保证续写时新 token 的位置编号接在历史长度后面，而不是重新从 0 开始。

常见坑可以系统化地看：

| 常见坑 | 直接后果 | 为什么会出错 | 规避策略 |
|---|---|---|---|
| 对 V 也做 RoPE | 输出内容被位置扭曲 | Value 承载的是内容，不是相似度几何 | 只旋转 Q/K |
| 训练和推理 `base` 不一致 | 注意力模式失配 | 相同位置被映射成不同角度 | 固化配置，加载模型时严格校验 |
| 头维拆分顺序写错 | 配对错误，结果异常 | `(0,1)` 被错配成 `(0,2)` 等 | 明确按偶数/奇数维成对旋转 |
| cache 位置偏移错误 | 续写时注意力错乱 | 新 token 角度没有接续历史位置 | 使用累计 token 数作为位置索引 |
| 长上下文直接硬外推 | 远距离行为退化 | 高频分量超出训练分布 | 结合插值、缩放、NTK-aware、YaRN 等扩展方法 |
| 不同实现的张量布局混用 | 推理结果偏差甚至崩坏 | `[B,H,S,D]` 与 `[B,S,H,D]` 广播规则不同 | 明确 shape，逐层断言 |
| 混淆 `dim` 和 `head_dim` | 频率构造错误 | RoPE 通常作用在单头维度上 | 用 `head_dim` 生成频率 |

### 为什么“范数保持”在工程上重要

数学上，原因已经很清楚：

$$
R^\top R = I
$$

工程上，它意味着历史 Key 的尺度不会因为位置越来越大而系统性漂移。这样在计算

$$
\text{score}(m,n)=\frac{q_m^\top k_n}{\sqrt{d}}
$$

时，分数的变化主要来自内容相似度和相对位置，而不是“编码本身把向量搞大了或搞小了”。

这并不等于“用了 RoPE 就一定长文本更准”。真正的长上下文效果还取决于训练长度、数据覆盖、模型容量、频率缩放策略等因素。RoPE 解决的是“位置编码引入几何失真”这类问题，而不是所有长上下文问题。

### 一个排错思路

当模型在长 prompt 续写时突然明显退化，可以优先检查这三项：

1. 新 token 的位置编号是否接续历史长度。
2. `base`、`head_dim`、配对顺序是否与训练时一致。
3. 旋转是否只施加在 Q/K，而没有误施加到 V。

很多线上 bug 用这三条就能快速定位。

---

## 替代方案与适用边界

RoPE 不是唯一方案。最常见的替代思路有两类：绝对位置编码，以及相对位置 bias。

### 1. 绝对位置编码

绝对位置编码通常直接加到输入上：

$$
x_m' = x_m + p_m
$$

它的优点是实现简单，几乎不需要改动 attention 结构。缺点是：attention score 中的相对位置信息不是显式构造出来的，而是让模型自己从绝对位置中学出来。因此在长长度外推时，它往往更吃力。

### 2. 相对位置 bias

相对位置 bias 的典型形式是：

$$
\text{score}(m,n)=\frac{q_m^\top k_n}{\sqrt{d}} + b_{m-n}
$$

这里的 $b_{m-n}$ 可以理解为一张“距离查表”或一个距离函数。它不改变 Q/K 的几何结构，而是直接在打分阶段加一个与相对距离相关的偏置。

这种方法的优点是控制直接、解释清楚。缺点是它把相对位置信息放在 score 层，而不是特征层。

### 3. RoPE 的位置

RoPE 对应的是：

$$
\text{score}(m,n)=\frac{\langle R_m q_m,\ R_n k_n\rangle}{\sqrt{d}}
$$

它不是在输入层加位置，也不是在 score 层加偏置，而是把位置写进了 Query 和 Key 的几何关系。

三类方案对比如下：

| 方案 | 外推能力 | KV-cache 兼容 | 实现成本 | 对注意力的作用位置 | 优势 | 局限 |
|---|---:|---:|---:|---|---|---|
| 绝对位置编码 | 较弱 | 一般 | 低 | 输入层 | 实现简单 | 相对位置不是显式结构 |
| 相对位置 bias | 中 | 中 | 中 | score 层 | 距离偏置可控 | 需要额外 bias 逻辑 |
| RoPE | 通常更好 | 高 | 中 | Q/K 几何层 | 相对位置自然进入点积，适合缓存 | 长度外推仍有限，需要额外扩展策略 |

适用边界也需要说清楚。

如果你的场景是自回归生成、长上下文推理、需要稳定使用 KV-cache，RoPE 通常是更自然的选择。它的几何结构和缓存机制天然兼容。

如果你更关心“距离偏置是否可控、可解释”，例如某些 encoder 场景或需要显式建模距离桶的任务，那么 relative bias 也很稳妥。

如果你只追求最少代码改动，绝对位置编码仍然可以工作，但它在训练长度之外的泛化通常不如 RoPE。这里的“外推”指的是：训练时没见过这么长的序列，测试时却要求处理更长上下文。

一句工程判断可以直接记住：

- 想要 KV-cache 兼容且对长序列更友好，优先考虑 RoPE。
- 想在 score 上显式控制距离偏置，relative bias 更直接。
- 只想快速实现一个基础版本，绝对位置编码最省事，但上限通常更低。

---

## 参考资料

下面给出更完整、可直接追溯的参考资料。阅读时建议优先看原始论文和主流实现，因为很多二手文章会省略关键前提，例如“RoPE 作用在单头维度上”或“续写时位置索引必须连续”。

| 来源 | 类型 | 主要贡献 | 更适合看什么 |
|---|---|---|---|
| Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* | 原始论文 | 首次系统提出 RoPE，给出旋转定义、相对位置性质与实验结果 | 数学定义、动机、正式结论 |
| LLaMA / Qwen / GPT-NeoX 等开源实现中的 RoPE 代码 | 工程实现 | 展示 `cos/sin` cache、张量布局、多头广播和 KV-cache 细节 | 真正如何写代码 |
| 多篇 RoPE 推导笔记与技术博客 | 二次讲解 | 把复数形式、旋转矩阵形式和相对位置性质串起来 | 入门理解与公式复现 |
| 长上下文扩展方法，如 NTK-aware scaling、Position Interpolation、YaRN | 工程扩展 | 讨论如何在保留 RoPE 结构的前提下改善长度外推 | 长上下文实践 |
| Transformer 与 attention 基础教材/论文 | 基础背景 | 帮助理解为什么“位置信息必须进入 QK 交互” | 新手补基础 |

建议阅读顺序如下：

1. 先看一篇偏直观的讲解，建立“二维旋转 + 相对位置”的基本图像。
2. 再看 RoFormer 原始论文，确认正式定义和推导。
3. 然后对照一个主流开源模型实现，理解 `cos/sin` cache、张量 shape 和 KV-cache 偏移。
4. 最后再看长上下文扩展方法，理解为什么“原始 RoPE 很强，但不是无限长度的万能解”。

如果只想抓住最重要的三篇材料，可以优先看：

1. RoFormer 原始论文。
2. 一个主流开源模型中的 `apply_rotary_pos_emb` 实现。
3. 一篇专门讨论 RoPE 长上下文扩展的工程文章或论文。
