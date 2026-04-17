## 核心结论

Performer 的核心不是“把 attention 做快一点”，而是把 softmax attention 的计算顺序彻底改写。

标准 attention 先显式构造 $L \times L$ 的权重矩阵，再对值向量加权：

$$
\mathrm{Att}(Q,K,V)=\mathrm{softmax}(QK^\top)V
$$

把 softmax 展开成“指数核 + 行归一化”后，可写成：

$$
\mathrm{Att}(Q,K,V)=
\mathrm{diag}\!\bigl(\exp(QK^\top)\mathbf{1}_L\bigr)^{-1}\exp(QK^\top)V
$$

Performer 用 FAVOR+（Fast Attention Via positive Orthogonal Random features）把 softmax 对应的指数核近似成两个低秩矩阵的乘积：

$$
e^{q^\top k}\approx \phi(q)^\top \phi(k)
$$

于是 attention 可改写为：

$$
\widehat{\mathrm{Att}}(Q,K,V)=D^{-1}\phi(Q)\bigl(\phi(K)^\top V\bigr)
$$

其中归一化项为：

$$
D=\mathrm{diag}\bigl(\phi(Q)(\phi(K)^\top \mathbf{1}_L)\bigr)
$$

这样做后，不再需要构造 $L \times L$ 的注意力矩阵，复杂度从 $O(L^2 d)$ 下降到 $O(Lmd)$。这里的 $m$ 是随机特征维度，意思是“用多少个随机方向去近似原来的核函数”。

最重要的直觉是两种计算顺序不同：

```text
传统 Attention
Q,K -> QK^T -> softmax -> 权重矩阵 A(LxL) -> A V

Performer / FAVOR+
Q,K -> φ(Q), φ(K)
φ(K)^T V -> 先得到“值向量摘要”(m x d_v)
φ(Q) · 摘要 -> 再按查询重组输出
```

可以把它理解成：

- 传统 attention：先给每个 query 生成一整行权重，再取值。
- FAVOR+：先把所有 key 对应的 value 压成一个小摘要，再让每个 query 去读取这个摘要。

这就是“先生成 $L^2$ 权重”与“先总结后打分”的本质区别。

| 方法 | 核心中间结果 | 时间复杂度 | 空间瓶颈 |
|---|---|---:|---:|
| 标准 softmax attention | $QK^\top \in \mathbb{R}^{L \times L}$ | $O(L^2 d)$ | $O(L^2)$ |
| Performer FAVOR+ | $\phi(K)^\top V \in \mathbb{R}^{m \times d_v}$ | $O(Lmd)$ | $O(md)$ 或 $O(Lm)$ |

---

## 问题定义与边界

问题本身很具体：softmax attention 的表达能力强，但其代价是二次复杂度。序列长度为 $L$ 时，哪怕单个 token 的特征维度 $d$ 不大，只要 $L$ 变到 8k、16k、32k，$QK^\top$ 的计算和存储都会变成瓶颈。

标准形式是：

$$
\mathrm{Att}(Q,K,V)=\mathrm{diag}\bigl(\exp(QK^\top)\mathbf{1}_L\bigr)^{-1}\exp(QK^\top)V
$$

这里把 softmax 写成“指数再归一化”的形式，是为了看清 Performer 近似的对象其实不是 softmax 本身，而是指数核 $e^{q^\top k}$。

FAVOR+ 的目标不是完全改变注意力语义，而是保留“基于 query-key 相似度给 value 加权”的模式，同时避免显式构造 $L \times L$ 矩阵。它近似的是：

$$
\exp(QK^\top)\approx \phi(Q)\phi(K)^\top
$$

于是：

$$
\widehat{\mathrm{Att}}(Q,K,V)
=
\mathrm{diag}\bigl(\phi(Q)(\phi(K)^\top \mathbf{1}_L)\bigr)^{-1}
\phi(Q)(\phi(K)^\top V)
$$

这件事成立有明确边界，不是“任意压缩都可以”：

| 约束 | 含义 | 不满足时的后果 |
|---|---|---|
| $\phi(x)$ 必须为正值特征 | 每个分量非负，便于做 softmax 风格归一化 | 分母可能为负或接近 0，输出不稳定 |
| $\mathbb{E}[\phi(q)^\top\phi(k)]=e^{q^\top k}$ | 期望一致，也叫无偏 | 系统性偏差累积 |
| 随机方向最好正交化 | 降低估计方差 | 小 $m$ 时误差大、训练抖动 |
| $m$ 不能过小 | 近似精度依赖特征维度 | 长序列时误差扩散 |
| 输入尺度要受控 | softmax 通常带 $1/\sqrt d$ 缩放 | 指数过大，数值容易溢出 |
| 需要定期 redraw | 重新采样随机特征，避免长期偏置固化 | 某些头长期学到坏近似 |

一个新手容易混淆的点是：Performer 并不是把 $L^2$ 变成 $m^2$，而是把“对每个 query 分别和全部 key 计算权重”改成“先对全部 key 做一次 $m$ 维摘要”。因此真正被压缩的是中间表示的形状。

```text
传统:
QK^T = (L x d)(d x L) -> (L x L)

FAVOR+:
φ(K)^T V = (m x L)(L x d_v) -> (m x d_v)
再用 φ(Q) = (L x m) 去读
```

如果业务非常依赖精确的注意力权重矩阵，比如要可解释地查看“第 17 个 token 主要看了谁”，那么 FAVOR+ 不是最直接的方案，因为它本质上已经不显式保存那张 $L \times L$ 权重表了。

再给一个量级比较。假设单头 $d=d_v=64$：

| 序列长度 $L$ | 标准 attention 权重矩阵元素数 | 若 $m=256$，FAVOR+ 主要中间量元素数 |
|---:|---:|---:|
| 2,048 | 4,194,304 | 524,288 |
| 8,192 | 67,108,864 | 2,097,152 |
| 32,768 | 1,073,741,824 | 8,388,608 |

这里比较的是“中间表示规模”，不是完整模型总 FLOPs，但已经足够说明为什么长序列下二者差距会迅速拉开。

---

## 核心机制与推导

先定义随机特征映射。随机特征可以理解成“用随机投影把核函数改写成内积”的技巧。FAVOR+ 使用的是正值映射：

$$
\phi(x)=\frac{1}{\sqrt{m}}
\left[
\exp(\omega_1^\top x-\|x\|^2/2),
\dots,
\exp(\omega_m^\top x-\|x\|^2/2)
\right]
$$

其中 $\omega_i \sim \mathcal{N}(0,I)$，并通常做正交化。白话说，就是先把向量投影到 $m$ 个随机方向，再过指数函数，保证结果全为正。

### 为什么它能近似 softmax 核

关键是高斯随机变量的矩母函数：

$$
\mathbb{E}_{\omega\sim \mathcal{N}(0,I)}[\exp(\omega^\top x)]
=
\exp\left(\frac{\|x\|^2}{2}\right)
$$

令 $x=q+k$，则：

$$
\mathbb{E}[\exp(\omega^\top(q+k))]
=
\exp\left(\frac{\|q+k\|^2}{2}\right)
$$

把两边同时乘上 $\exp(-\|q\|^2/2)\exp(-\|k\|^2/2)$，得到：

$$
\mathbb{E}\bigl[\exp(\omega^\top q-\|q\|^2/2)\exp(\omega^\top k-\|k\|^2/2)\bigr]
=
\exp(q^\top k)
$$

因此：

$$
\mathbb{E}[\phi(q)^\top \phi(k)] = e^{q^\top k}
$$

这说明 $\phi(q)^\top \phi(k)$ 是指数核的无偏估计。

### 从单个 query 推到矩阵形式

标准 attention 对第 $i$ 个 query 的输出是：

$$
y_i=
\frac{\sum_{j=1}^L e^{q_i^\top k_j} v_j}
{\sum_{j=1}^L e^{q_i^\top k_j}}
$$

用随机特征替换后：

$$
y_i \approx
\frac{\sum_{j=1}^L \phi(q_i)^\top \phi(k_j) v_j}
{\sum_{j=1}^L \phi(q_i)^\top \phi(k_j)}
$$

因为 $\phi(q_i)$ 与求和下标 $j$ 无关，可以提到求和外面：

$$
y_i \approx
\frac{
\phi(q_i)^\top \left(\sum_{j=1}^L \phi(k_j) v_j^\top\right)
}{
\phi(q_i)^\top \left(\sum_{j=1}^L \phi(k_j)\right)
}
$$

这里出现了两个关键中间量：

$$
S=\sum_{j=1}^L \phi(k_j) v_j^\top \in \mathbb{R}^{m \times d_v}
$$

$$
z=\sum_{j=1}^L \phi(k_j) \in \mathbb{R}^{m}
$$

于是：

$$
y_i \approx \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}
$$

把所有 query 一次写成矩阵形式，就是：

$$
\widehat{\mathrm{Att}}(Q,K,V)=D^{-1}\phi(Q)(\phi(K)^\top V)
$$

其中：

$$
D=\mathrm{diag}\bigl(\phi(Q)(\phi(K)^\top \mathbf{1}_L)\bigr)
$$

这里的 $D$ 可以理解成“每一行自己的归一化分母”。标准 attention 的分母来自一整行 $L$ 个相似度，Performer 的分母来自一个长度为 $m$ 的随机特征摘要。

### 玩具例子

取 $m=2$，随机方向先不考虑正交细节，设：

$$
\omega_1=(1,0),\quad \omega_2=(0,1)
$$

查询、键、值分别为：

$$
q=(1,0),\quad k=(0,1),\quad v=(2,3)
$$

因为 $\|q\|^2=\|k\|^2=1$，所以：

$$
\phi(q)=\frac{1}{\sqrt{2}}(e^{1-1/2}, e^{0-1/2})
=\frac{1}{\sqrt{2}}(e^{0.5}, e^{-0.5})
$$

$$
\phi(k)=\frac{1}{\sqrt{2}}(e^{-0.5}, e^{0.5})
$$

两者内积为：

$$
\phi(q)^\top \phi(k)
=
\frac{1}{2}(e^0+e^0)=1
$$

而真实核值是：

$$
e^{q^\top k}=e^0=1
$$

这个最小例子刚好对齐。它说明一件事：随机特征空间里的内积，确实可以逼近原来指数相似度。

如果现在有多个键值对，Performer 先计算：

$$
S=\phi(K)^\top V
$$

这个 $S$ 就是“值向量摘要”。后续每个 query 只需要与摘要做一次乘法，而不是重新扫一遍所有 key。

| 步骤 | 公式 | 结果形状 |
|---|---|---|
| 特征映射 | $\phi(Q), \phi(K)$ | $L \times m$ |
| 值摘要 | $\phi(K)^\top V$ | $m \times d_v$ |
| 分子 | $\phi(Q)(\phi(K)^\top V)$ | $L \times d_v$ |
| 分母 | $\phi(Q)(\phi(K)^\top \mathbf{1})$ | $L \times 1$ |
| 归一化输出 | $D^{-1}\cdot$分子 | $L \times d_v$ |

### 为什么正交随机特征更稳

如果 $\omega_1,\dots,\omega_m$ 完全独立，高概率会出现“多个方向差不多”的情况，相当于把采样预算浪费在重复区域。正交化的作用是让这些方向尽量分散，降低估计方差。

这不会改变无偏性目标：

$$
\mathbb{E}[\phi(q)^\top\phi(k)] = e^{q^\top k}
$$

但会改善有限样本时的误差分布。对工程来说，含义就是：同样的 $m$，正交随机特征通常比普通高斯采样更稳，训练抖动更小。

---

## 代码实现

下面给一个可运行的简化 Python 实现。它展示的不是完整训练版 Performer，而是 FAVOR+ 的核心计算路径：先算 $\phi(K)^\top V$，再右乘 $\phi(Q)$，最后用分母归一化。

代码只依赖 Python 标准库，直接用 `python3` 即可运行。

```python
import math
import random


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def qr_orthogonal_rows(d, rng):
    # 用 Gram-Schmidt 生成 d x d 正交矩阵，再把每一行当作一个随机方向
    cols = [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(d)]
    qcols = []
    for col in cols:
        v = col[:]
        for q in qcols:
            dot = sum(vi * qi for vi, qi in zip(v, q))
            for i in range(d):
                v[i] -= dot * q[i]
        norm = math.sqrt(sum(vi * vi for vi in v))
        if norm < 1e-12:
            return qr_orthogonal_rows(d, rng)
        qcols.append([vi / norm for vi in v])
    return transpose(qcols)


def orthogonal_random_matrix(m, d, seed=0):
    rng = random.Random(seed)
    rows = []
    while len(rows) < m:
        rows.extend(qr_orthogonal_rows(d, rng))
    return rows[:m]


def positive_random_features(x, omega):
    # x: [L, d], omega: [m, d] -> [L, m]
    m = len(omega)
    out = []
    for row in x:
        sq_norm_half = 0.5 * sum(v * v for v in row)
        feats = []
        for w in omega:
            dot = sum(a * b for a, b in zip(row, w))
            feats.append(math.exp(dot - sq_norm_half) / math.sqrt(m))
        out.append(feats)
    return out


def rowwise_softmax(scores):
    out = []
    for row in scores:
        shift = max(row)
        exps = [math.exp(v - shift) for v in row]
        denom = sum(exps)
        out.append([v / denom for v in exps])
    return out


def softmax_attention(q, k, v):
    # 标准 attention，带 1/sqrt(d) 缩放
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    scores = [[scale * sum(a * b for a, b in zip(qi, kj)) for kj in k] for qi in q]
    weights = rowwise_softmax(scores)
    return matmul(weights, v)


def favor_attention(q, k, v, omega, eps=1e-9):
    # 为了匹配 softmax(QK^T / sqrt(d))，把 q 和 k 都乘上 d^(-1/4)
    d = len(q[0])
    scale = d ** -0.25
    q_scaled = [[scale * x for x in row] for row in q]
    k_scaled = [[scale * x for x in row] for row in k]

    q_phi = positive_random_features(q_scaled, omega)  # [L, m]
    k_phi = positive_random_features(k_scaled, omega)  # [L, m]

    kv_summary = matmul(transpose(k_phi), v)           # [m, d_v]
    k_sum = [[sum(col)] for col in transpose(k_phi)]   # [m, 1]

    numerator = matmul(q_phi, kv_summary)              # [L, d_v]
    denominator = matmul(q_phi, k_sum)                # [L, 1]

    out = []
    for num_row, den_row in zip(numerator, denominator):
        den = den_row[0] + eps
        out.append([x / den for x in num_row])
    return out


def max_abs_diff(a, b):
    diff = 0.0
    for ra, rb in zip(a, b):
        for xa, xb in zip(ra, rb):
            diff = max(diff, abs(xa - xb))
    return diff


if __name__ == "__main__":
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[2.0, 1.0], [0.0, 3.0]]

    omega = orthogonal_random_matrix(m=256, d=2, seed=42)

    exact = softmax_attention(Q, K, V)
    approx = favor_attention(Q, K, V, omega)

    print("exact =")
    for row in exact:
        print([round(x, 6) for x in row])

    print("approx =")
    for row in approx:
        print([round(x, 6) for x in row])

    err = max_abs_diff(exact, approx)
    print("max_abs_err =", round(err, 6))

    assert len(exact) == len(approx) == 2
    assert len(exact[0]) == len(approx[0]) == 2
    assert err < 0.35, err
```

一组实际输出如下：

```text
exact =
[1.339523, 1.660477]
[0.660477, 2.339523]

approx =
[1.102546, 1.897454]
[0.833118, 2.166882]

max_abs_err = 0.236977
```

这个实现有几个关键点：

| 模块 | 作用 | 维度 |
|---|---|---|
| `orthogonal_random_matrix` | 生成近似正交随机方向 | $m \times d$ |
| `positive_random_features` | 计算正值映射 $\phi(x)$ | $L \times m$ |
| `kv_summary` | 汇总所有 key 对 value 的贡献 | $m \times d_v$ |
| `k_sum` | 汇总所有 key 的归一化分母贡献 | $m \times 1$ |
| `numerator / denominator` | 对每个 query 产出近似 attention 输出 | $L \times d_v$ |

原理上最容易写错的是分母。正确形式不是把 `k_phi.T` 和 `q_phi` 直接做全矩阵乘完再取和，而是先算：

$$
z=\phi(K)^\top \mathbf{1}_L=\sum_{j=1}^L \phi(k_j)
$$

再算：

$$
\phi(Q)z
$$

这才对应每个 query 自己的归一化项。

如果把它嵌到真实模型中，通常还会做三件事：

1. 对 $Q,K$ 加入标准的 $1/\sqrt d$ 缩放。
2. 对指数输入做数值稳定处理，避免溢出。
3. 定期 redraw 随机矩阵 $\omega$，避免固定随机基底导致误差模式长期固化。

### 真实工程例子

在长上下文文档模型、语音模型或 DNA / 蛋白序列模型里，$d$ 往往固定，而 $L$ 会从几千涨到几万。此时标准 attention 的代价随 $L^2$ 上涨，Performer 的主要代价则变成 $L \cdot m \cdot d$。

如果原本单头 $d=64$、序列长度 $L=16384$，取 $m=256$，则：

- 标准 attention 需要显式处理 $16384 \times 16384$ 的相似度矩阵。
- FAVOR+ 只需要维护 $L \times m$ 的特征映射，以及 $m \times d_v$ 的摘要。

工程上常见做法是：

1. 从 $m=2d$ 或 $m=4d$ 起试。
2. 观察训练损失抖动、验证集 perplexity、BLEU、WER 或下游任务指标。
3. 如果误差偏大，再提高 $m$，或者缩短 redraw 周期。

这类场景里，用户通常不关心“精确注意力矩阵长什么样”，而更关心“全局依赖能不能保住，同时显存和吞吐能不能接受”。这是 FAVOR+ 最适合的区域。

---

## 工程权衡与常见坑

FAVOR+ 好用，但它不是“把公式改一下就行”。

第一类坑是特征映射取错。很多人第一次接触随机特征，会想到 $\cos/\sin$ 形式，因为它常见于 RBF 核近似。但 softmax attention 需要的是正值归一化结构。若特征可以取负，分母

$$
D_i=\phi(q_i)^\top \sum_{j=1}^L \phi(k_j)
$$

就可能非常小、为负、甚至数值上接近 0，最后导致输出爆炸或 NaN。FAVOR+ 强调 positive random features，原因就在这里。

第二类坑是只采样高斯方向，不做正交化。普通随机特征的方差通常按 $O(1/m)$ 下降，也就是说，想把误差压下去，需要很大 $m$。正交随机特征的好处是不同方向更“分散”，减少重复采样相似方向，同样的 $m$ 下更稳定。

第三类坑是 $m$ 选得太激进。线性 attention 的优势来自不再出现 $L^2$，但如果把 $m$ 压得远小于所需精度，误差会直接变成模型能力损失，而不是简单的数值噪声。

第四类坑是忽略 softmax 的缩放。真实 attention 不是 $e^{q^\top k}$，而是：

$$
e^{q^\top k / \sqrt d}
$$

实现时通常把 $q$ 和 $k$ 都乘上 $d^{-1/4}$，这样两者内积就自动变成 $q^\top k / \sqrt d$。如果忘了这一步，近似对象就已经错了。

第五类坑是只看吞吐，不看指标。线性复杂度不等于无损替代。序列越长、任务越依赖细粒度对齐，近似误差越可能暴露。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 用 $\cos/\sin$ 替代正值特征 | 分母不稳，可能 NaN | 使用 $\exp(\omega^\top x-\|x\|^2/2)$ |
| 不做正交化 | 方差大，小 $m$ 误差明显 | 用 QR、Hadamard 或 Gram-Schmidt |
| $m$ 太小 | 近似过粗，精度掉得快 | 从 $2d$ 或 $4d$ 起调 |
| 忘记 $1/\sqrt d$ 缩放 | 近似目标错位 | 对 $q,k$ 乘 $d^{-1/4}$ |
| 长期固定一组 $\omega$ | 误差模式固化 | periodic redraw |
| 忽略分母数值稳定 | 除零或梯度爆炸 | 加 $\varepsilon$，控制输入尺度 |

periodic redraw 的位置可以简单理解为：

```text
训练若干 step
-> 重新采样/正交化 ω
-> 后续继续用新的 φ
```

它不是每一步都必须重采样，而是周期性刷新。太频繁会增加噪声，太稀疏又可能让某组随机方向长期主导误差。

一个真实工程上的排查顺序通常是：

1. 先检查 $m$ 是否过小。
2. 再检查是否真的用了正交随机特征。
3. 再检查缩放、分母 $\varepsilon$、redraw 周期和混合精度设置。
4. 最后再判断是不是任务本身就不适合纯随机特征近似。

---

## 替代方案与适用边界

FAVOR+ 不是唯一的长序列方案。它属于“保持全局交互，但把 softmax 核线性化”的路线。其他路线常见的还有稀疏 attention、低秩投影和局部-全局混合方案。

| 方案 | 复杂度 | 是否保留全局交互 | 核心思路 | 适用场景 |
|---|---:|---|---|---|
| 标准 attention | $O(L^2 d)$ | 是 | 精确计算所有两两关系 | 中短序列、要求精确 |
| FAVOR+ / Performer | $O(Lmd)$ | 是 | 随机特征近似 softmax 核 | 长序列、仍需全局依赖 |
| Linformer | 近似 $O(Lkd)$ | 部分 | 对序列维做低秩投影 | 注意力矩阵低秩明显时 |
| Sparse attention | 依结构而定 | 常为局部+少量全局 | 只算部分边 | 已知局部结构明显 |
| Hybrid local+linear | 混合 | 是 | 局部精确，全局近似 | 既要局部细节又要长程 |

本质区别可以写成两类。

稀疏方法假设：

$$
A_{ij}=0 \quad \text{对大量 } (i,j)
$$

也就是“很多位置根本不看”。

FAVOR+ 假设：

$$
e^{q^\top k}\approx \phi(q)^\top \phi(k)
$$

也就是“大家仍然全局互相看，但相似度可被低维随机特征近似”。

所以它们适用的边界不同：

- 如果任务天然是局部结构主导，比如固定窗口语音建模、局部 patch 建模，sparse 或 local attention 很自然。
- 如果任务需要任意位置的全局交互，但又无法承受 $L^2$，FAVOR+ 更合适。
- 如果你必须精确恢复注意力权重图，或者误差容忍度极低，标准 attention 或 hybrid 方案更稳。

给新手一句最直接的判断标准：

- 需要“所有 token 都可能彼此相关”，选 FAVOR+ 这类全局线性化方案。
- 需要“只关注附近或指定模式”，选 sparse 类方案。
- 需要“局部精确 + 全局可扩展”，选 hybrid。

再补一句工程判断：如果你已经知道任务的先验结构很强，先用结构化方法；如果你没有明确稀疏先验，但必须保留全局交互，再考虑 FAVOR+。

---

## 参考资料

| 资料 | 重点 | 说明 |
|---|---|---|
| [Choromanski et al., *Rethinking Attention with Performers* (Google Research / ICLR 2021)](https://research.google/pubs/rethinking-attention-with-performers/) | FAVOR+ 原理、正值正交随机特征、线性 attention 推导、误差界 | 主参考。先看摘要和方法总览，再看 softmax kernel 近似与理论部分。 |
| [arXiv: *Rethinking Attention with Performers*](https://arxiv.org/abs/2009.14794) | 完整公式、定理、附录细节 | 适合逐条核对文中公式，尤其是正值特征构造、缩放方式和因果版本。 |
| [Rahimi & Recht, *Random Features for Large-Scale Kernel Machines* (NeurIPS 2007)](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) | 随机特征近似核函数的基本思想 | 不是 Performer 论文，但能帮助理解“为什么核函数可以改写成低维内积近似”。 |

Choromanski 等人的论文是主参考，因为核心结论都来自这里：softmax 核的随机特征分解、正值映射、正交随机特征带来的方差改进，以及 Performer 的矩阵重排形式。

Rahimi 与 Recht 的论文不是 softmax attention 论文，但它提供了随机特征方法的基本框架。对初学者来说，先接受“核函数可以用随机内积来估计”，再回来看 FAVOR+，会更容易理解 Performer 为什么成立。

如果只打算读一篇，优先读 Performer 原论文；如果读到“随机特征为什么能近似核”这一层卡住，再回看随机特征基础论文。
