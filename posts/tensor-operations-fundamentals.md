## 核心结论

张量可以先理解成“多轴数组”，也就是比一维数组、二维表再多几个方向的数据容器。标量是零阶张量，只有一个数；向量是一阶张量，像一排数；矩阵是二阶张量，像一个表格；更高阶张量只是把“轴”的数量继续增加。

从数学上看，张量运算最核心的只有三类：

1. 张量积，也叫外积，作用是把两个张量拼成更高阶结构，阶数相加。
2. 张量收缩，也叫缩并，作用是对某些重复索引求和，阶数通常减少 2。
3. 张量重排，也叫置换或转置，作用是不改数据值，只改轴的顺序。

很多看起来复杂的运算，其实都能拆成这三类。例如矩阵乘法本质上就是一次收缩：
$$
C_{ik}=\sum_j A_{ij}B_{jk}
$$

Einstein 求和约定可以理解成“一旦某个索引字母重复出现，就默认沿这个轴求和”，所以它把长公式压缩成更短的索引表达式。工程实现里，PyTorch、TensorFlow 并不是在“发明新数学”，而是在高效执行这些轴操作。

如果只记一个判断标准，可以记这一条：看运算后轴数怎么变。轴数变多，多半是张量积；轴数变少，多半是收缩；轴数不变但顺序变了，多半是置换。

---

## 问题定义与边界

本文讨论的是**密集张量**。密集的意思是“绝大多数位置都有值，而且按连续数组存储”，这也是 NumPy、PyTorch、TensorFlow 默认处理的对象。本文不讨论微分几何中的协变、逆变，也不讨论张量场、稀疏张量、符号张量。

对初学者，更实用的定义是：一个张量由三部分共同决定其计算行为。

| 概念 | 白话解释 | 例子 |
| --- | --- | --- |
| `shape` | 每个轴有多长 | `(2, 3, 4)` 表示 3 个轴，长度分别为 2、3、4 |
| `stride` | 在内存里沿某个轴走一步要跳几个元素 | `(12, 4, 1)` 表示最后一轴相邻元素挨着放 |
| `storage` | 真正存放数据的一段连续内存 | 可以被多个视图共享 |

这里的 `stride` 容易抽象。可以把它理解成：如果逻辑索引沿某个轴加 1，底层一维内存偏移量要加多少。于是，同样的 `shape`，只要 `stride` 不同，逻辑上看起来一样，物理布局就可能不同。

先看一个二维例子。假设有矩阵：
$$
A=
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix}
$$

它的 `shape` 是 `(2, 3)`。如果按行连续存储，那么底层可以看成：
$$
[1,2,3,4,5,6]
$$

此时 `stride=(3,1)`，意思是：

- 行索引加 1，要跨过 3 个元素
- 列索引加 1，只跨过 1 个元素

所以元素 $A_{1,2}$ 的内存偏移量是：
$$
1 \cdot 3 + 2 \cdot 1 = 5
$$

如果从 0 开始计数，偏移量 5 对应的值就是 6。

下面是图像张量里常见的两个布局。`N` 是 batch 数，`C` 是通道数，`H` 是高度，`W` 是宽度。

| 维度顺序 | 常见 shape | 默认 stride 示例 | 物理含义 |
| --- | --- | --- | --- |
| `NCHW` | `(1, 3, 224, 224)` | `(150528, 50176, 224, 1)` | 通道在前，PyTorch 传统默认布局 |
| `NHWC` | `(1, 224, 224, 3)` 或 channels-last 视图 | `(150528, 672, 3, 1)` | 通道在后，更利于部分卷积实现 |

很多初学者会把“维度顺序”和“数据内容”混为一谈。需要区分两件事：

| 你看到的东西 | 它回答的问题 |
| --- | --- |
| `shape` | 这个张量有几个轴，每个轴多长 |
| `stride` | 这些轴在内存里怎么走 |
| 索引记号 `x[i,j,k]` | 这个位置对应哪一个元素 |
| 运算公式 | 这些轴如何配对、保留或消掉 |

这也是本文的边界：我们重点解释**轴、索引、shape、stride、收缩、置换**如何统一成一个框架，而不是展开所有张量理论分支。

---

## 核心机制与推导

先从最小例子开始。

### 1. 标量、向量、矩阵为什么能统一

设有一个标量 $a$，一个向量 $v_i$，一个矩阵 $M_{ij}$。它们的差别不是“本质不同”，而是索引个数不同：

- $a$ 没有索引，零阶
- $v_i$ 有一个索引，一阶
- $M_{ij}$ 有两个索引，二阶

因此，张量本质上是“带多个索引的数值对象”。

如果再往上加一个索引，例如 $T_{ijk}$，它就成了三阶张量。可以把它理解成“一叠矩阵”，但这个比喻只适合帮助建立直觉，不是严格定义。严格定义仍然是：它有三个独立索引，每个索引沿一个轴取值。

下面这个表更容易建立统一视角：

| 对象 | 索引形式 | 阶数 | 常见 `shape` 例子 |
| --- | --- | --- | --- |
| 标量 | $a$ | 0 | `()` |
| 向量 | $v_i$ | 1 | `(3,)` |
| 矩阵 | $M_{ij}$ | 2 | `(2, 3)` |
| 三阶张量 | $T_{ijk}$ | 3 | `(2, 3, 4)` |

初学时最重要的一点是：**阶数不是“数据有多大”，而是“有多少个独立轴”**。`shape=(1000,)` 的向量依然是一阶张量；`shape=(2,2,2)` 的张量虽然元素只有 8 个，但它是三阶张量。

### 2. 张量积：把轴拼起来

张量积可以理解成“所有元素两两配对”。如果 $S$ 是一个二阶张量，$T$ 是一个一阶张量，那么：
$$
(S\otimes T)^{ijk}=S^{ij}T^k
$$

这里左边多了一个新索引 $k$，所以阶数从 $2+1$ 变成 3。

玩具例子：

设
$$
u=\begin{bmatrix}1\\2\end{bmatrix},\quad
v=\begin{bmatrix}3\\4\\5\end{bmatrix}
$$

则外积 $u\otimes v$ 是一个 $2\times 3$ 的矩阵：
$$
u\otimes v=
\begin{bmatrix}
1\cdot 3 & 1\cdot 4 & 1\cdot 5\\
2\cdot 3 & 2\cdot 4 & 2\cdot 5
\end{bmatrix}
=
\begin{bmatrix}
3 & 4 & 5\\
6 & 8 & 10
\end{bmatrix}
$$

从“轴数变化”看，一阶和一阶做张量积，得到二阶。

再看一个三阶结果的例子。设矩阵
$$
A=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
\quad
w=
\begin{bmatrix}
10\\20\\30
\end{bmatrix}
$$

那么 $A \otimes w$ 的 `shape` 是 `(2, 2, 3)`，并且
$$
(A\otimes w)_{ijk}=A_{ij}w_k
$$

这说明张量积的本质不是“只会得到矩阵”，而是把输入张量的所有轴直接拼接起来。

张量积有两个常见用途：

| 用途 | 直观解释 |
| --- | --- |
| 构造高阶特征 | 把两个对象的组合关系全部展开 |
| 表达秩一结构 | 用几个向量的外积构造简单高阶张量 |

如果只看公式，张量积最容易识别的特征是：**没有求和，只有索引并列出现**。

### 3. 张量收缩：把某个轴求和消掉

收缩可以理解成“找到重复索引，对那个索引遍历求和”。最典型就是矩阵乘法：
$$
C_{ik}=\sum_j A_{ij}B_{jk}
$$

这里 $j$ 是中间索引。它在式子右边重复出现，于是沿着 $j$ 求和，结果左边只剩 $i,k$ 两个索引。

玩具例子：

$$
A=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},\quad
B=
\begin{bmatrix}
5 & 6\\
7 & 8
\end{bmatrix}
$$

则
$$
C_{11}=1\cdot5+2\cdot7=19,\quad
C_{12}=1\cdot6+2\cdot8=22
$$
$$
C_{21}=3\cdot5+4\cdot7=43,\quad
C_{22}=3\cdot6+4\cdot8=50
$$

所以
$$
C=
\begin{bmatrix}
19 & 22\\
43 & 50
\end{bmatrix}
$$

这就是“每行乘对应列再累加”的严格索引写法。

收缩不只发生在矩阵乘法里。下面几个例子都属于收缩：

| 运算 | 索引形式 | 输入阶数 | 输出阶数 |
| --- | --- | --- | --- |
| 向量点积 | $s=a_i b_i$ | 1 和 1 | 0 |
| 矩阵乘向量 | $y_i=A_{ij}x_j$ | 2 和 1 | 1 |
| 矩阵乘法 | $C_{ik}=A_{ij}B_{jk}$ | 2 和 2 | 2 |

可以看到，重复索引每出现一对，通常就会消去一个求和轴，因此总阶数会下降 2。

还可以把“求迹”看成收缩。对于方阵 $A$：
$$
\mathrm{tr}(A)=A_{ii}=\sum_i A_{ii}
$$

它把矩阵的两个轴收缩到一个标量。这也是为什么“对角线求和”与“收缩”是同一类操作。

初学时最容易出错的点是：**逐元素乘法不是收缩**。例如：
$$
D_{ij}=A_{ij}B_{ij}
$$

如果输出里保留了 $i,j$，那它只是对应位置相乘；如果写成
$$
A_{ij}B_{ij}
$$
而没有输出索引，那么这才表示沿 $i,j$ 全部求和，结果是标量。

### 4. Einstein 求和约定：省略求和号

Einstein 求和约定可以白话理解成“重复字母自动求和”。于是前面的矩阵乘法可以写成：
$$
C_{ik}=A_{ij}B_{jk}
$$

没有显式写 $\sum_j$，但因为 $j$ 重复了，所以默认沿 $j$ 求和。

它的好处是统一。比如：

- 向量点积：$s=a_i b_i$
- 矩阵乘法：$C_{ik}=A_{ij}B_{jk}$
- 批量矩阵乘法：$C_{bik}=A_{bij}B_{bjk}$

索引字母本身就说明了轴如何匹配。

对初学者，最实用的读法是下面这个四步法：

| 步骤 | 你要做什么 |
| --- | --- |
| 1 | 先看每个输入分别有哪些索引 |
| 2 | 找出右边重复出现的索引 |
| 3 | 重复索引表示要求和 |
| 4 | 左边保留下来的索引，决定输出 `shape` |

例如：
$$
y_i=A_{ij}x_j
$$

可以这样读：

- `A` 有两个轴 `i,j`
- `x` 有一个轴 `j`
- `j` 重复，所以对 `j` 求和
- 输出只保留 `i`，因此 `y` 是向量

再看一个批量例子：
$$
C_{bik}=A_{bij}B_{bjk}
$$

这里 `b` 没被消掉，说明 batch 轴只是对齐保留，不参与收缩；真正被收缩的是 `j`。所以输出仍然有 `b,i,k` 三个轴。

程序里的 `einsum` 只是把这种索引写法字符串化。例如：

| 数学写法 | `einsum` 写法 |
| --- | --- |
| $s=a_i b_i$ | `'i,i->'` |
| $y_i=A_{ij}x_j$ | `'ij,j->i'` |
| $C_{ik}=A_{ij}B_{jk}$ | `'ij,jk->ik'` |
| $C_{bik}=A_{bij}B_{bjk}$ | `'bij,bjk->bik'` |

因此，学会 Einstein 记号的关键不是背字符串，而是先能从索引推出输出轴。

### 5. 置换与转置：只改看法，不改值

转置可以理解成“交换轴的顺序”。例如矩阵转置：
$$
B_{ji}=A_{ij}
$$

在工程实现里，这往往不需要复制数据，只需要改 `stride`。访问某个元素的底层位置可以写成：
$$
\text{offset}=\sum_{m=1}^{n} \text{index}_m \cdot \text{stride}_m
$$

这个公式说明：逻辑索引如何映射到内存，由 `stride` 决定。转置的本质，通常就是调整这组步长。

先看二维情况。若矩阵 `A` 的 `shape=(2,3)`，`stride=(3,1)`，那么转置后的 `A^T`：

- `shape` 变成 `(3,2)`
- `stride` 变成 `(1,3)`

这表示原来“沿列移动一步”的方式，变成了“沿行移动一步”的方式。

更一般地，若三阶张量 $T_{ijk}$ 做置换，变成 $S_{kij}$，那只是把轴顺序从 `(i,j,k)` 改成 `(k,i,j)`。元素值本身没有变，变化的是你如何解释这些轴。

可以把三类核心运算放在一张表里统一看：

| 运算 | 索引特征 | 阶数变化 | 是否求和 | 是否改值 |
| --- | --- | --- | --- | --- |
| 张量积 | 索引直接并列 | 增加 | 否 | 会生成新值 |
| 收缩 | 有重复索引 | 减少 | 是 | 会生成新值 |
| 置换 | 索引顺序调整 | 不变 | 否 | 否 |

这张表基本就是全文的压缩版。遇到一个新运算时，先问自己三件事：

1. 有没有重复索引。
2. 输出保留哪些轴。
3. 轴顺序有没有变化。

大多数张量公式都可以这样拆开。

---

## 代码实现

先用纯 Python 写一个最小实现，把“外积”和“收缩”落到代码上。

```python
def shape_of_2d(x):
    if not x or not x[0]:
        raise ValueError("matrix must be non-empty")
    cols = len(x[0])
    if any(len(row) != cols for row in x):
        raise ValueError("matrix rows must have equal length")
    return len(x), cols


def outer_product(u, v):
    return [[ui * vj for vj in v] for ui in u]


def matmul(a, b):
    rows_a, cols_a = shape_of_2d(a)
    rows_b, cols_b = shape_of_2d(b)
    if cols_a != rows_b:
        raise ValueError(
            f"incompatible shapes: ({rows_a}, {cols_a}) and ({rows_b}, {cols_b})"
        )

    out = []
    for i in range(rows_a):
        row = []
        for k in range(cols_b):
            s = 0
            for j in range(cols_a):
                s += a[i][j] * b[j][k]
            row.append(s)
        out.append(row)
    return out


def transpose(a):
    rows, cols = shape_of_2d(a)
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


if __name__ == "__main__":
    u = [1, 2]
    v = [3, 4, 5]
    assert outer_product(u, v) == [[3, 4, 5], [6, 8, 10]]

    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    c = matmul(a, b)
    assert c == [[19, 22], [43, 50]]
    assert transpose(a) == [[1, 3], [2, 4]]

    print("outer_product(u, v) =", outer_product(u, v))
    print("matmul(a, b) =", c)
    print("transpose(a) =", transpose(a))
```

这段代码分别对应三类基本动作：

| 代码 | 对应概念 |
| --- | --- |
| `outer_product(u, v)` | 张量积，保留两个输入的轴 |
| `matmul(a, b)` | 收缩，对中间索引 `j` 求和 |
| `transpose(a)` | 置换，交换两个轴 |

这里有一个关键观察。`matmul` 的三层循环不是随便写出来的，它正好对应公式：
$$
C_{ik}=\sum_j A_{ij}B_{jk}
$$

- 外层 `i`：枚举行
- 中层 `k`：枚举列
- 内层 `j`：做收缩求和

如果换成 `einsum` 风格，矩阵乘法就是：

```python
# 逻辑等价于 'ij,jk->ik'
```

真实工程里，框架会把这种索引关系映射成高性能内核。

再给一个纯 NumPy 版本，便于从“手写循环”过渡到“库函数表达”：

```python
import numpy as np

u = np.array([1, 2])
v = np.array([3, 4, 5])
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

outer = np.einsum("i,j->ij", u, v)
matmul_result = np.einsum("ij,jk->ik", a, b)
transposed = np.transpose(a)

assert np.array_equal(outer, np.array([[3, 4, 5], [6, 8, 10]]))
assert np.array_equal(matmul_result, np.array([[19, 22], [43, 50]]))
assert np.array_equal(transposed, np.array([[1, 3], [2, 4]]))
```

这个版本的价值在于：你可以直接把数学索引抄进代码。对新手来说，它通常比先写一串 `reshape`、`permute` 更容易检查。

### PyTorch：关注布局和 stride

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

x = torch.randn(1, 3, 8, 8)
print("x.shape =", x.shape)
print("x.stride() =", x.stride())
print("x.is_contiguous() =", x.is_contiguous())

x_cl = x.to(memory_format=torch.channels_last)
print("x_cl.shape =", x_cl.shape)
print("x_cl.stride() =", x_cl.stride())
print("x_cl.is_contiguous(channels_last) =",
      x_cl.is_contiguous(memory_format=torch.channels_last))

w = torch.randn(4, 3, 3, 3)

y = F.conv2d(x, w, padding=1)
y_cl = F.conv2d(x_cl, w, padding=1)

assert y.shape == y_cl.shape
assert torch.allclose(y, y_cl, atol=1e-5)

p = x.permute(0, 2, 3, 1)
print("p.shape =", p.shape)
print("p.stride() =", p.stride())
print("p.is_contiguous() =", p.is_contiguous())

pc = p.contiguous()
print("pc.is_contiguous() =", pc.is_contiguous())
```

这里有两个关键点：

1. `shape` 不变，不代表物理布局不变。`x` 和 `x_cl` 的数值语义相同，但 `stride` 不同。
2. `permute` 往往只改视图，不复制数据，所以很容易得到非连续张量。

真实工程例子：在卷积神经网络里，如果后端对 channels-last 优化较好，那么 `NHWC` 或等价的 channels-last 布局常常能提高缓存命中率和吞吐。原因不是数学变了，而是卷积核访问通道维时更连续。

为了把 `stride` 看得更具体，可以观察一个小例子：

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x)
print("shape:", x.shape)
print("stride:", x.stride())

xt = x.t()
print(xt)
print("shape:", xt.shape)
print("stride:", xt.stride())

assert x[1, 2].item() == xt[2, 1].item()
```

这个例子说明：转置前后的同一个值仍然能访问到，但访问路径变了。

### TensorFlow：关注索引表达式

```python
import tensorflow as tf

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[5., 6.], [7., 8.]])

c = tf.einsum("ij,jk->ik", a, b)
expected = tf.constant([[19., 22.], [43., 50.]])
tf.debugging.assert_near(c, expected)

x = tf.constant([1., 2.])
y = tf.constant([3., 4., 5.])
outer = tf.einsum("i,j->ij", x, y)
outer_expected = tf.constant([[3., 4., 5.], [6., 8., 10.]])
tf.debugging.assert_near(outer, outer_expected)

print("matmul result:\n", c.numpy())
print("outer result:\n", outer.numpy())
```

`'ij,jk->ik'` 这串字符串就是 Einstein 记号的程序化版本：

- `ij` 表示第一个输入的两个轴
- `jk` 表示第二个输入的两个轴
- `->ik` 表示输出保留 `i` 和 `k`
- `j` 没出现在输出里，因此被收缩求和

更一般的批量写法是：

```python
# tf.einsum('...ij,...jk->...ik', s, t)
```

`...` 可以理解成“前面还有若干公共批次轴，我不想一个个写出来”。当模型里有很多高阶收缩时，`einsum` 比手工 `reshape + transpose + matmul` 更接近数学定义，也更不容易写错轴。

对新手来说，写 `einsum` 时可以先做一个检查表：

| 检查项 | 你要确认什么 |
| --- | --- |
| 输入轴 | 每个输入的索引是否和 `shape` 对得上 |
| 收缩轴 | 哪些字母重复，哪些轴会被求和 |
| 输出轴 | 输出中保留哪些字母，顺序是否正确 |
| 广播轴 | 有没有依赖 `...` 或长度为 1 的轴 |

只要把这四项写清楚，大多数 `einsum` 错误都能提前发现。

---

## 工程权衡与常见坑

张量数学本身不难，真正容易出错的是“数学对象”和“物理布局”被混在一起。

| 场景 | 潜在问题 | 缓解方法 |
| --- | --- | --- |
| `permute` 或 `transpose` 后直接算 | 结果逻辑正确，但可能变成非连续视图，后续算子隐式拷贝 | 在性能敏感路径检查 `is_contiguous()`，必要时调用 `.contiguous()` |
| NCHW 与 NHWC 混用 | 每层都在做 layout 转换，吞吐下降 | 尽早统一整条算子链的布局 |
| `einsum` 索引写错 | 维度不匹配或错误广播 | 先手写索引含义，再检查输出轴是否符合预期 |
| 高阶收缩过大 | 中间张量爆内存 | 调整收缩顺序，或用框架优化路径 |

一个典型坑是“非 contiguous 张量”。`contiguous` 可以白话理解成“当前视图是否按预期连续排布，能不能直接高效顺序读取”。

例如在 PyTorch 中：

```python
import torch

x = torch.randn(2, 3, 4, 5)
y = x.permute(0, 2, 3, 1)

assert not y.is_contiguous()

z = y.contiguous()
assert z.is_contiguous()
assert z.shape == y.shape
assert torch.allclose(z, y)
```

`permute` 只是改轴顺序，往往不复制数据，所以 `y` 可能只是一个“改了看法的视图”。这在数学上没问题，但如果后续算子要求更规则的内存布局，框架可能在你看不见的地方做一次拷贝。性能分析时，这类隐式拷贝经常是吞吐下降的原因。

真实工程里，图像模型最常见的权衡如下：

1. 如果模型主干是 CNN，且后端对 channels-last 支持成熟，可以统一采用该布局。
2. 如果模型中间夹杂很多老算子、第三方算子或自定义算子，混用布局的转换成本可能抵消收益。
3. 如果重点是开发可读性而不是极致性能，先保证轴语义清晰，再做布局优化更稳妥。

还有一个常见坑是把“收缩”和“逐元素乘法”混淆。`A_{ij}B_{ij}` 会对重复索引求和；如果你想保留每个位置的乘积，应该写逐元素乘法，而不是隐含 Einstein 收缩。

下面用一张表把几个容易混淆的写法放在一起：

| 写法 | 含义 | 输出类型 |
| --- | --- | --- |
| `A * B` | 逐元素乘法 | 与输入同形状 |
| `A @ B` | 矩阵乘法，包含收缩 | 新矩阵 |
| `einsum('ij,ij->ij', A, B)` | 显式逐元素乘法 | 新矩阵 |
| `einsum('ij,ij->', A, B)` | 对两个轴都收缩 | 标量 |

另一个常见问题是广播。比如 `shape=(2,3)` 和 `shape=(3,)` 可以做逐元素运算，是因为后者会沿前面的轴扩展；但广播不是收缩，也不是置换，它只是框架在计算前对轴进行对齐。很多 bug 都来自“你以为在做矩阵乘法，实际写成了广播逐元素乘法”。

因此，工程里判断一个张量表达式是否安全，至少要同时检查三件事：

1. 数学语义对不对。
2. 轴顺序对不对。
3. 布局代价是否可接受。

---

## 替代方案与适用边界

不是所有张量问题都该直接写底层索引操作。工程上通常有三层选择。

| 方案 | 适合场景 | 优点 | 边界 |
| --- | --- | --- | --- |
| 明确调用 `matmul`、`conv` 等专用算子 | 常见线代和深度学习算子 | 最稳定，后端优化最好 | 表达复杂高阶收缩不够直观 |
| 使用 `einsum` | 轴关系复杂、公式驱动的实现 | 与数学表达一致，少写 `reshape/transpose` | 大规模收缩需要注意路径优化 |
| 使用张量分解 | 参数量或存储量太大 | 可降维、压缩模型 | 需要额外误差分析和训练策略 |

### 1. `einsum` 不是万能键

`einsum` 在这些场景尤其合适：

- 你能清楚写出索引关系
- 运算本质是收缩、外积、批量线代
- 你想减少手工变形张量的代码

但如果只是普通矩阵乘法、卷积、线性层，直接用专用 API 往往更清晰，也更容易吃到框架的专门优化。

一个简单判断标准是：

| 如果你的目标是 | 优先选择 |
| --- | --- |
| 常见算子、性能优先 | `matmul`、`conv`、`linear` |
| 公式很清楚、轴关系复杂 | `einsum` |
| 研究型原型、需要快速验证数学式 | `einsum` 或 NumPy/PyTorch 原型代码 |

### 2. PyTorch 与 TensorFlow 的侧重点不同

PyTorch 更直接暴露布局细节，例如 `stride`、`memory_format`，适合你需要主动控制性能路径的场景。TensorFlow 更强调图级优化，例如 `tf.einsum`、`tf.function(jit_compile=True)`、XLA 融合，适合把一串张量操作交给编译器统一处理。

可以粗略这样理解：

- PyTorch 偏“我能看见物理布局，并主动调它”
- TensorFlow 偏“我描述计算关系，让编译器优化它”

这不是说两者只能做一类事，而是默认思路不同。对新手来说，学习顺序通常可以这样安排：

1. 先用 PyTorch 或 NumPy 建立 `shape/stride/permute` 直觉。
2. 再用 `einsum` 练习把公式翻成程序。
3. 最后再关心图编译、融合、布局调优。

这样不容易在还没搞清楚轴语义时，就过早陷入性能术语。

### 3. 张量分解：不是基础运算，但常用于降维

当张量非常大时，仅靠调整 `stride` 或改写 `einsum` 不够，问题变成“数据本身太大”。这时会用到张量分解。

- CP 分解：把高阶张量拆成若干个秩一张量之和。秩一可以白话理解成“由几个向量外积得到的最简单高阶结构”。
- Tucker 分解：把原张量拆成一个更小的核心张量和多组因子矩阵。

它们常用于：

- 模型压缩
- 特征降维
- 多模态推荐系统
- 时空数据分析

可以把它们和基础运算的关系理解成：

| 层次 | 你在解决什么问题 |
| --- | --- |
| 张量积、收缩、置换 | 一个张量怎么计算 |
| 布局、stride、memory format | 一个张量怎么存得更适合计算 |
| 张量分解 | 一个太大的张量怎么用更小结构近似 |

但这已经超出“基础运算”层面。初学者应先把“张量积、收缩、置换、布局”学扎实，再理解分解更自然。

---

## 参考资料

- Wikipedia: Tensor  
  https://en.wikipedia.org/wiki/Tensor
- Wikipedia: Tensor contraction  
  https://en.wikipedia.org/wiki/Tensor_contraction
- Wikipedia: Einstein notation  
  https://en.wikipedia.org/wiki/Einstein_notation
- PyTorch Docs: Storage  
  https://docs.pytorch.org/docs/stable/storage.html
- PyTorch Docs: Tensor attributes / memory format  
  https://docs.pytorch.org/docs/stable/tensor_attributes.html
- PyTorch Docs: `torch.einsum`  
  https://docs.pytorch.org/docs/stable/generated/torch.einsum.html
- PyTorch Blog: Accelerating PyTorch Vision Models with Channels Last on CPU  
  https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/
- PyTorch Blog: Tensor Memory Format Matters  
  https://pytorch.org/blog/tensor-memory-format-matters/
- TensorFlow Docs: `tf.einsum`  
  https://www.tensorflow.org/api_docs/python/tf/einsum
- TensorFlow Guide: GPU performance analysis / XLA  
  https://www.tensorflow.org/guide/gpu_performance_analysis
- NumPy Docs: `numpy.einsum`  
  https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
