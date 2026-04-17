## 核心结论

线性变换可以理解为“保持加法和数乘结构不变的映射”，白话说，就是它不会破坏向量空间里的线性关系。矩阵不是线性变换本身，而是线性变换在某一组基下的坐标写法。设 $T:V\to V$ 是同一个线性变换，基 $\beta$ 下的矩阵是 $B=[T]_\beta$，基 $\beta'$ 下的矩阵是 $B'=[T]_{\beta'}$，那么二者满足

$$
B' = P^{-1}BP
$$

其中 $P$ 是过渡矩阵，列向量是“新基向量在旧基下的坐标”。这条公式的意思不是“变换变了”，而是“观察坐标系变了”。

玩具例子先给结论。令

$$
B=\begin{bmatrix}2&0\\0&1\end{bmatrix},\quad
P=\begin{bmatrix}1&1\\0&1\end{bmatrix}
$$

则

$$
P^{-1}=\begin{bmatrix}1&-1\\0&1\end{bmatrix},\quad
B'=P^{-1}BP=\begin{bmatrix}2&1\\0&1\end{bmatrix}
$$

在标准基里，这个变换看起来像“沿 $x$ 方向放大 2 倍，沿 $y$ 方向不变”；在斜基里，同一个变换会出现非对角项，但本质没变。

| 性质 | $B$ | $B'$ | 是否相同 |
|---|---:|---:|---|
| 秩 rank | 2 | 2 | 是 |
| 迹 trace | 3 | 3 | 是 |
| 行列式 determinant | 2 | 2 | 是 |
| 特征值 eigenvalues | 2, 1 | 2, 1 | 是 |

迹可以理解为主对角线之和，白话说，它常被用来刻画线性变换整体缩放趋势的一部分。行列式可以理解为体积缩放因子，白话说，它告诉你面积或体积被整体放大或翻转了多少。相似矩阵共享这些不变量，因此它们只是“同一变换的不同坐标版本”。

---

## 问题定义与边界

问题的核心不是“矩阵怎么换”，而是“同一个线性变换在不同基下如何表示”。

设 $V=\mathbb{R}^2$。标准基记为

$$
\beta = \{e_1=(1,0), e_2=(0,1)\}
$$

再选一组新基

$$
\beta' = \{v_1=(1,0), v_2=(1,1)\}
$$

那么过渡矩阵

$$
P=\begin{bmatrix}1&1\\0&1\end{bmatrix}
$$

因为它的第一列是 $v_1$ 在旧基 $\beta$ 下的坐标，第二列是 $v_2$ 在旧基 $\beta$ 下的坐标。

这时，一个向量若在新基下坐标是 $[x]_{\beta'}$，那么它在旧基下坐标是

$$
[x]_\beta = P[x]_{\beta'}
$$

反过来，

$$
[x]_{\beta'} = P^{-1}[x]_\beta
$$

可以把它理解成“先把新坐标翻译成旧坐标，再用旧规则做变换，再翻译回新坐标”。

这里的边界非常重要：

1. 只有当 $T:V\to V$，也就是输入空间和输出空间是同一个空间时，才谈相似变换 $B'=P^{-1}BP$。
2. 如果 $T:V\to W$ 且 $\dim V\neq \dim W$，矩阵一般不是方阵，这时不能用相似变换，只能用双基变换。
3. 相似描述的是“同一个算子在同一空间的不同坐标表示”；等价变换描述的是“不同输入基和输出基下的矩阵重写”，两者不是一回事。

一个容易混淆的点是：换基不是改动向量本身，而是改动“记录向量的尺子”。把 $x$ 轴、$y$ 轴换成斜轴之后，几何箭头还在那里，变的是坐标数字。

---

## 核心机制与推导

推导只要抓住三步：

1. 新基坐标先映到旧基坐标：$[x]_\beta=P[x]_{\beta'}$
2. 在线性变换 $T$ 下，用旧基矩阵 $B$ 作用：$[T(x)]_\beta=B[x]_\beta$
3. 再映回新基坐标：$[T(x)]_{\beta'}=P^{-1}[T(x)]_\beta$

连起来就是

$$
[T(x)]_{\beta'} = P^{-1}BP[x]_{\beta'}
$$

因此

$$
B'=[T]_{\beta'}=P^{-1}BP
$$

这条公式本质上是在说：先换坐标，再做旧矩阵定义的变换，最后换回坐标。

用前面的玩具例子展开一次：

$$
B=\begin{bmatrix}2&0\\0&1\end{bmatrix},\quad
P=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad
P^{-1}=\begin{bmatrix}1&-1\\0&1\end{bmatrix}
$$

先算

$$
BP=
\begin{bmatrix}2&0\\0&1\end{bmatrix}
\begin{bmatrix}1&1\\0&1\end{bmatrix}
=
\begin{bmatrix}2&2\\0&1\end{bmatrix}
$$

再算

$$
B'=P^{-1}BP=
\begin{bmatrix}1&-1\\0&1\end{bmatrix}
\begin{bmatrix}2&2\\0&1\end{bmatrix}
=
\begin{bmatrix}2&1\\0&1\end{bmatrix}
$$

原基和新基的过程可以写成表：

| 步骤 | 原基 $\beta$ 视角 | 新基 $\beta'$ 视角 |
|---|---|---|
| 输入坐标 | $[x]_\beta$ | $[x]_{\beta'}$ |
| 坐标转换 | 不需要 | $[x]_\beta=P[x]_{\beta'}$ |
| 应用变换 | $B[x]_\beta$ | $BP[x]_{\beta'}$ |
| 转回新基 | 不需要 | $P^{-1}BP[x]_{\beta'}$ |

接着看秩-零化度定理：

$$
\dim \ker T + \dim \operatorname{Im} T = \dim V
$$

核 $\ker T$ 可以理解为“被变换压到零向量的所有输入”；像 $\operatorname{Im} T$ 可以理解为“变换真正能到达的输出集合”。白话说，一个变换如果把很多方向压扁成 0，那它能保留下来的独立输出方向就会变少。

对上面的 $B=\mathrm{diag}(2,1)$，因为它可逆，所以

$$
\ker T=\{0\},\quad \dim\ker T=0
$$

同时满秩，

$$
\dim\operatorname{Im} T=2
$$

于是 $0+2=2=\dim \mathbb{R}^2$。换成 $B'$ 后，这两个维度不变，因为相似矩阵表示同一个线性变换，只是坐标系不同。

---

## 代码实现

工程里最直接的实现就是：给定旧基矩阵表示 $B$ 和过渡矩阵 $P$，先求逆，再做 $P^{-1}BP$。关键不是代码长短，而是维度和可逆性检查。

下面给一个可运行的 `python` 例子。它不依赖第三方库，直接验证相似变换、迹、行列式和核维数的基本结论。

```python
from fractions import Fraction

def matmul(A, B):
    rows, cols, inner = len(A), len(B[0]), len(B)
    return [
        [sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]

def det2(M):
    return M[0][0]*M[1][1] - M[0][1]*M[1][0]

def inv2(M):
    d = det2(M)
    assert d != 0, "P must be invertible"
    return [
        [Fraction(M[1][1], d), Fraction(-M[0][1], d)],
        [Fraction(-M[1][0], d), Fraction(M[0][0], d)],
    ]

def trace(M):
    return sum(M[i][i] for i in range(len(M)))

def rank2(M):
    if det2(M) != 0:
        return 2
    if all(M[i][j] == 0 for i in range(2) for j in range(2)):
        return 0
    return 1

def nullity2(M):
    return 2 - rank2(M)

def change_of_basis(B, P):
    Pinv = inv2(P)
    return matmul(matmul(Pinv, B), P)

B = [
    [2, 0],
    [0, 1],
]
P = [
    [1, 1],
    [0, 1],
]

Bp = change_of_basis(B, P)

assert Bp == [
    [Fraction(2, 1), Fraction(1, 1)],
    [Fraction(0, 1), Fraction(1, 1)],
]

assert trace(B) == trace(Bp) == 3
assert det2(B) == det2(Bp) == 2
assert rank2(B) == rank2(Bp) == 2
assert nullity2(B) + rank2(B) == 2
assert nullity2(Bp) + rank2(Bp) == 2

print("all checks passed")
```

如果写成 JavaScript/TypeScript 风格，核心接口通常就是：

```ts
function changeOfBasis(B: Matrix, P: Matrix): Matrix {
  const Pinv = invert(P); // 先校验 P 可逆
  return multiply(multiply(Pinv, B), P);
}
```

真实工程例子可以看 Transformer 的多头注意力。注意力里的 $W^Q, W^K, W^V$ 是线性投影矩阵，投影可以理解为“把原始表示映射到某个子空间”。如果输入表示 $x\in\mathbb{R}^{d_{model}}$，那么某个头上的查询向量是

$$
q_i = xW_i^Q
$$

键和值同理。多个头分别在不同子空间做计算，本质上就是多个线性变换并行工作。若这些子空间选得更接近正交基，头与头之间的功能更容易分离，因为不同方向上的信息耦合更弱。这里“正交”可以理解为“彼此垂直、互不干扰”的坐标方向。

---

## 工程权衡与常见坑

相似变换和等价变换最容易被混用。相似变换要求方阵，表示同一个空间上的同一个线性变换。等价变换允许输入基和输出基分别变化，适合非方阵。

| 对比项 | 相似变换 | 等价变换 |
|---|---|---|
| 公式 | $B' = P^{-1}BP$ | $\tilde B = P^{-1}BQ$ |
| 适用矩阵 | 方阵 | 任意 $m\times n$ |
| 变化对象 | 同一空间的基 | 输入基和输出基分别变化 |
| 保持性质 | 特征值、迹、行列式、秩 | 一般保持秩，不保持特征值等 |

常见坑主要有四类。

第一，把非方阵误写成 $P^{-1}BP$。如果 $B$ 是 $m\times n$ 且 $m\neq n$，这个式子通常连维度都不合法。即使维度碰巧能乘，也不表示“同一变换的换基”。

第二，把“矩阵长得不一样”误以为“模型行为变了”。在线性代数里，很多差异只是坐标描述差异。真正要追踪的是不变量，例如秩、特征值、迹、行列式。

第三，忽略 $P$ 的可逆性。过渡矩阵必须由一组基组成，因此列向量必须线性无关。白话说，新尺子不能有冗余方向，否则无法唯一表示向量。

第四，在多头注意力里忽略维度配平。若模型总维度是 $d_{model}$，头数是 $h$，通常每头维度 $d_k=d_v=d_{model}/h$。如果某个头的 $W_i^Q$ 或 $W_i^V$ 输出维度不一致，拼接时就会失败，因为各头输出不能沿同一规则堆叠回去。

工程上建议直接写断言，而不是依赖运行时报错。例如：

- 断言 `P` 可逆
- 断言 `B` 与 `P` 维度匹配
- 多头注意力中断言所有头的输出维度一致
- 若做数值计算，优先使用正交化后的基，减少病态矩阵带来的误差放大

---

## 替代方案与适用边界

当矩阵不是方阵时，应使用双基变换：

$$
\tilde B = P^{-1}BQ
$$

这里 $Q$ 负责输入空间基的变化，$P$ 负责输出空间基的变化。注意左右两边的矩阵不再是同一件事的“对称换基”，因为输入和输出空间本来就可能不同。

设

$$
B\in\mathbb{R}^{m\times n}
$$

则维度关系是：

| 矩阵 | 维度 | 作用 |
|---|---|---|
| $B$ | $m\times n$ | 从输入空间到输出空间 |
| $Q$ | $n\times n$ | 输入空间换基 |
| $P$ | $m\times m$ | 输出空间换基 |
| $P^{-1}BQ$ | $m\times n$ | 新双基下的表示 |

这在真实工程里很常见。例如 embedding 层把词向量从 $\mathbb{R}^{d_{token}}$ 映射到 $\mathbb{R}^{d_{model}}$，这是非方阵映射，不应硬套相似变换。

在 Transformer 场景下，还有两类常见替代思路：

1. 选取正交基或近似正交的投影方向。优点是头之间更容易解耦，数值更稳定；缺点是会限制参数自由度。
2. 用低秩分解近似投影矩阵，例如把 $W^Q$ 拆成两个更小矩阵。低秩可以理解为“用更少的独立方向近似原始变换”。优点是节省参数和算力；缺点是表达能力可能下降。

所以边界要记清楚：

- 研究同一线性算子在不同坐标下的形式，用相似变换。
- 研究不同输入输出空间中的矩阵重写，用双基等价。
- 研究模型压缩或子空间结构时，可以引入正交约束、低秩分解或块结构，而不必执着于纯粹的相似变换框架。

---

## 参考资料

| 来源标题 | 主题 | 适合解决的问题 | 链接 |
|---|---|---|---|
| Change of basis | 换基、过渡矩阵 | 理解 $P$ 的列向量含义，以及为什么会出现 $P^{-1}BP$ | https://en.wikipedia.org/wiki/Change_of_basis |
| Matrix similarity | 相似矩阵 | 理解相似矩阵为何共享特征值、迹、行列式等不变量 | https://en.wikipedia.org/wiki/Matrix_similarity |
| Rank-nullity theorem | 秩-零化度定理 | 理解核与像的维数平衡，以及为什么 $\dim\ker T + \dim\operatorname{Im}T = \dim V$ | https://en.wikipedia.org/wiki/Rank%E2%80%93nullity_theorem |
| Matrix equivalence | 矩阵等价 | 区分非方阵的双基变换与方阵的相似变换 | https://en.wikipedia.org/wiki/Matrix_equivalence |
| Attention Is All You Need | 多头注意力 | 理解 Q/K/V 投影为什么本质上是线性变换 | https://en.wikipedia.org/wiki/Attention_Is_All_You_Need |
| Attention Is All You Need Paper Summary | 注意力机制解读 | 用工程视角理解多头子空间分解与维度设计 | https://kingy.ai/blog/attention-is-all-you-need-paper-summary/ |

建议阅读顺序是：先看 Change of basis，再看 Matrix similarity 和 Rank-nullity theorem，最后回到 Transformer 的 Q/K/V 投影理解“线性变换如何进入真实模型”。
