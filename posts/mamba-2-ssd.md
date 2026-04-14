## 核心结论

Mamba-2 的 SSD Theory（Structured State Space Duality，结构化状态空间对偶）给出一个统一视角：**同一个选择性状态空间模型，可以既写成递归的 SSM，也写成带结构掩码的 attention-like 矩阵乘法**。这不是“两个近似方法”，而是同一个线性变换的两种计算顺序。

递归形式是：

$$
h_t = a_t h_{t-1} + B_t x_t,\qquad y_t = C_t^\top h_t
$$

这里的“状态”可以理解为模型手里一直带着的一份压缩记忆，随着时间步不断更新。  
attention 形式是：

$$
Y = MX,\qquad M = L \circ (CB^\top)
$$

其中 $\circ$ 是逐元素乘法，$L$ 是因果掩码，控制“当前位置只能使用过去的信息”。

这件事重要，不是因为公式更漂亮，而是因为它直接改变了实现路径：

| 视角 | 计算方式 | 硬件友好性 | 典型优势 |
|---|---|---|---|
| SSM recurrence | 按时间步递推 | 串行状态更新友好 | 推理时状态缓存自然 |
| Structured attention | 显式或分块矩阵乘 | Tensor Core / matmul 友好 | 训练和长序列吞吐更高 |

对新手最直观的理解是：  
“每一步都做一次 `旧记忆 * 衰减 + 新输入写入`”，和“先把所有时间步之间的影响关系写成一个下三角矩阵，再一次性乘输入序列”，**结果相同，只是算的顺序不同**。

---

## 问题定义与边界

先说问题本身。

长序列建模里，Transformer 的注意力表达能力强，但标准实现是 $O(T^2)$；状态空间模型能把计算压到线性复杂度，但很多实现更像递归机，未必充分利用现代 GPU 最擅长的矩阵乘法。SSD 要解决的是：

**能不能在不丢掉 SSM 长程建模能力和因果性的前提下，把它改写成 attention-friendly 的结构，让同一层既能递归算，也能按矩阵块并行算？**

答案是可以，但有明确边界。

### 1. 不是任意 SSM 都能直接这样改写

关键限制是 $A_t$ 的结构。  
在 Mamba-2 的 SSD 里，$A_t$ 不是一般矩阵，也不是逐维不同的对角矩阵，而是 **scalar-times-identity**，即“标量乘单位阵”。白话说：**这个时间步的遗忘强度，对该头里所有状态维度都一样**。所以常写成一个标量 $a_t$。

如果放松成一般矩阵，很多漂亮的分解就消失了；如果放松成普通对角结构，仍然是 SSM，但不再是这里要讲的最核心 SSD 子类。

### 2. 必须保留因果性

因果性就是“第 $t$ 个 token 不能偷看未来 token”。  
在矩阵形式里，这对应 $M$ 必须是下三角；在 SSD 里，这由结构矩阵 $L$ 保证。

### 3. 半可分结构是核心，不是装饰

“半可分矩阵”（semiseparable matrix）可以先理解成：**下三角区域里很多子块都不是满秩的，而是可分解、可共享因子的**。这让它既能表达长程依赖，又能做结构化加速。

下面这个对比能把边界说清楚：

| 项目 | SSM 递归视角 | Structured attention 视角 |
|---|---|---|
| 输入 | 序列 $x_1,\dots,x_T$ | 矩阵 $X\in \mathbb{R}^{T\times P}$ |
| 中间量 | 状态 $h_t$ | 混合矩阵 $M\in \mathbb{R}^{T\times T}$ |
| 因果性 | 由递推顺序天然保证 | 由下三角掩码 $L$ 保证 |
| 结构约束 | $A_t=a_t I$ | $L$ 是 1-semiseparable |
| 适合计算 | scan / recurrence | matmul / block matmul |

---

## 核心机制与推导

### 1. 从递归展开到矩阵

先看最简单的一维玩具例子，设状态维度 $P=1$，并且：

- $a_1 = 0.8$
- $B_1=B_2=1$
- $C_1=C_2=1$
- 输入 $x=[1,2]$

递推是：

$$
h_1 = 0.8\cdot 0 + 1\cdot 1 = 1
$$

$$
h_2 = 0.8\cdot 1 + 1\cdot 2 = 2.8
$$

所以输出是：

$$
y_1 = 1,\qquad y_2 = 2.8
$$

如果把它展开成矩阵，就得到：

$$
M =
\begin{bmatrix}
1 & 0 \\
0.8 & 1
\end{bmatrix}
$$

于是

$$
y = Mx =
\begin{bmatrix}
1 & 0 \\
0.8 & 1
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
=
\begin{bmatrix}
1\\
2.8
\end{bmatrix}
$$

结果完全一致。

这个例子说明：**递归不是另一种模型，只是矩阵乘法没有被显式写出来。**

### 2. 一般形式为什么成立

对一般的 SSM，展开后可写成：

$$
M_{ij} =
\begin{cases}
0, & i<j \\
C_i^\top A_{i:j}^\times B_j, & i\ge j
\end{cases}
$$

其中 $A_{i:j}^\times$ 表示从第 $j+1$ 步到第 $i$ 步的递推系数乘积。  
如果 $A_t$ 是标量 $a_t$，那么它可以从双线性项里直接提出去：

$$
C_i^\top A_{i:j}^\times B_j
=
A_{i:j}^\times \cdot (C_i^\top B_j)
$$

于是矩阵 $M$ 被拆成两部分：

- 一部分是只由时间衰减构成的下三角矩阵 $L$
- 一部分是内容相关的双线性项 $CB^\top$

所以：

$$
M = L \circ (CB^\top)
$$

这就是 SSD 的 attention-like 形式。

### 3. 1-SS 掩码是什么

SSD 里，$L$ 不是普通 causal mask，而是 **1-semiseparable mask**。它长这样：

$$
L=
\begin{bmatrix}
1 & 0 & 0 & \cdots \\
a_1 & 1 & 0 & \cdots \\
a_2a_1 & a_2 & 1 & \cdots \\
\vdots & \vdots & \ddots & \ddots
\end{bmatrix}
$$

白话解释：  
第 $i$ 行第 $j$ 列的值，表示“第 $j$ 个输入对第 $i$ 个输出还能保留多少影响”，本质上是中间所有遗忘系数的连乘。

所以：

- $a_t$ 大，历史保留更久
- $a_t$ 小，历史更快衰减
- 对角线上永远是 1，因为当前位置直接看到当前位置输入

### 4. 为什么这能统一 attention 与 SSM

从 SSM 看，是“先做状态递推，再读出输出”。  
从 attention 看，是“先构造 token 间影响矩阵，再乘输入”。

二者只是张量 contraction 的 reduction order 不同。  
白话说：**先按时间累计，还是先把时间关系摊平成矩阵，数学上是同一个线性算子。**

### 5. 真实工程例子

假设你在做一个 32k 长度上下文的代码补全模型。

如果用纯递归实现 SSD 层：

- 每个 token 都要做状态更新
- 很难让大部分计算落到 Tensor Core
- 序列越长，串行部分越显眼

如果改成 SSD 的分块 attention-like 算法：

- 块内关系可直接用 batched matmul
- 块间状态只需要在更短的 chunk 序列上扫描
- 大头算力被转成矩阵乘法

这就是 Mamba-2 在系统实现里能获得明显吞吐提升的原因。提升不是来自“少算了”，而是来自**把相同计算重排成更符合 GPU 硬件的形式**。

---

## 代码实现

下面先给一个最小可运行版本，验证“递归”和“矩阵形式”等价。

```python
import math

def ssm_recurrence(x, a, b, c):
    """
    x: 输入序列, shape [T]
    a: 递推系数, shape [T]
    b: 写入系数, shape [T]
    c: 读出系数, shape [T]
    """
    h = 0.0
    y = []
    for t in range(len(x)):
        h = a[t] * h + b[t] * x[t]
        y.append(c[t] * h)
    return y

def build_ssd_matrix(a, b, c):
    """
    构造 M = L ∘ (C B^T)
    这里是一维情形，方便验证。
    """
    T = len(a)
    M = [[0.0 for _ in range(T)] for _ in range(T)]
    for i in range(T):
        for j in range(i + 1):
            decay = 1.0
            for k in range(j + 1, i + 1):
                decay *= a[k]
            M[i][j] = decay * c[i] * b[j]
    return M

def matvec(M, x):
    return [sum(M[i][j] * x[j] for j in range(len(x))) for i in range(len(M))]

# 玩具例子
x = [1.0, 2.0]
a = [0.0, 0.8]   # a[0]不会真正用到前一状态，可填任意占位
b = [1.0, 1.0]
c = [1.0, 1.0]

y_rec = ssm_recurrence(x, a, b, c)
M = build_ssd_matrix(a, b, c)
y_mat = matvec(M, x)

assert abs(y_rec[0] - 1.0) < 1e-8
assert abs(y_rec[1] - 2.8) < 1e-8
assert all(abs(u - v) < 1e-8 for u, v in zip(y_rec, y_mat))

print("recurrence:", y_rec)
print("matrix:", y_mat)
```

如果你运行它，会得到相同输出。这就是 SSD 理论最核心的最小证明。

接着看工程化一点的伪实现。真正的 Mamba-2 不会直接把整个 $T\times T$ 矩阵全部物化，而是分块：

```python
def ssd_chunked_pseudocode(X, A, B, C, Q):
    """
    X: [T, P]
    A: [T]          每步一个标量衰减
    B, C: [T, N]    写入/读出因子
    Q: block size
    """
    T = len(X)
    Y = zeros_like(X)

    for block_start in range(0, T, Q):
        block_end = min(block_start + Q, T)

        # 1. 块内 attention-like 计算
        # 构造局部 1-SS mask，然后做 batched matmul
        Y_local = intra_chunk_matmul(
            X[block_start:block_end],
            A[block_start:block_end],
            B[block_start:block_end],
            C[block_start:block_end],
        )

        # 2. 计算该块结束时的 chunk state
        chunk_state = summarize_final_state(
            X[block_start:block_end],
            A[block_start:block_end],
            B[block_start:block_end],
        )

        # 3. 在 chunk 级别传递状态
        # 这里只在更短的序列上做 scan
        propagated_state = pass_state_across_chunks(chunk_state)

        # 4. 把块初始状态转成该块输出贡献
        Y_state = state_to_output(
            propagated_state,
            A[block_start:block_end],
            C[block_start:block_end],
        )

        Y[block_start:block_end] = Y_local + Y_state

    return Y
```

这里最关键的不是具体 API，而是结构：

| 步骤 | 含义 | 主要计算特征 |
|---|---|---|
| 1 | 块内输出 | 小块 attention-like matmul |
| 2 | 块内最终状态 | batched matmul |
| 3 | 块间传状态 | 短序列 scan |
| 4 | 状态转输出 | batched matmul |

所以 SSD 算法的思想不是“完全抛弃递归”，而是**把必须递归的部分缩到 chunk 级别，把大部分工作交给 matmul**。

---

## 工程权衡与常见坑

### 1. 最大前提是结构不能乱

最常见误用是：  
“既然是 attention-like 矩阵，那我直接学一个任意下三角矩阵不就行了？”

不行。  
一旦 $L$ 是任意随机下三角矩阵，它通常不再是 1-SS 结构，也就不再对应那个简单的标量 SSM 递推。你失去的不是一点优化技巧，而是**整个对偶理论成立的条件**。

### 2. 数值稳定性比公式更难

$L$ 的元素本质上是很多个 $a_t$ 的连乘。  
如果 $a_t \approx 0.9$，序列很长时就会变得非常小；如果参数化不当，也可能爆掉。直接做 `cumprod` 很容易数值不稳。

常见处理是：

- 用 log-space，把连乘变连加
- 用 segment sum 而不是直接做乘积比值
- 对参数化后的 $a_t$ 做裁剪或稳定变换

### 3. 不要误以为“矩阵形式一定更快”

如果序列很短、batch 很小、硬件对大 matmul 不敏感，显式构造块矩阵未必占优。SSD 的收益主要在：

- 长序列
- 多头
- GPU/加速器对 matmul 吞吐极高
- 训练或批量推理

### 4. 常见坑总结

| 误用 | 影响 | 规避方式 |
|---|---|---|
| 把 $L$ 当任意可学习下三角矩阵 | 失去 1-SS 结构，无法与递归严格对齐 | 用由 $a_t$ 构造的 1-SS mask |
| 忘记因果掩码 | 当前 token 泄露未来信息 | 强制下三角结构 |
| 直接用 `cumprod` 构造长程衰减 | 数值下溢或精度差 | 改用 log-space / segsum |
| 全量物化大矩阵 | 显存和带宽压力大 | 分块、共享因子、chunk 传状态 |
| 认为 SSD 完全替代 scan | 忽略块间状态传递 | 保留 chunk 级 scan |

---

## 替代方案与适用边界

SSD 不是“所有序列模型的终点”，它是一个很强、但有结构约束的子类。

### 1. Mamba-1 / S6：表达更自由

Mamba-1 的核心 selective SSM 使用对角 $A$，不是标量 $A$。  
白话说：**每个状态维度可以有不同的遗忘速度**。这通常更灵活，但也更难直接改写成 SSD 那样漂亮的 structured attention。

适合场景：

- 更在意状态动态的表达能力
- 不强求把大部分算子转成 Tensor Core 友好的 matmul
- 接受更偏 scan 的实现

### 2. SSD：结构更强，系统更友好

SSD 把 $A$ 进一步收紧成标量，因此牺牲一部分自由度，换来：

- 与 1-SS mask 的严格对应
- 块分解算法更自然
- 更容易借用 Transformer 生态里的并行和系统优化

适合场景：

- 长序列训练
- 希望充分利用 GPU matmul
- 希望在 SSM 与 attention 之间切换计算视角

### 3. 其它 structured masked attention

更广义地看，SSD 还是 structured masked attention 的一个特例。  
也就是说，理论上你可以把 $L$ 换成其它结构矩阵，比如某些 Toeplitz 或频域结构，来注入不同归纳偏置。代价是：**一旦换掉结构，原来那套 SSM 对偶和系统优化未必还能原样保留。**

下面给一个选择表：

| 方法 | 结构条件 | 优点 | 典型适用场景 |
|---|---|---|---|
| Mamba-1 / S6 | 对角 $A$ | 状态动态更丰富 | 更强调表达力的 SSM |
| Mamba-2 / SSD | 标量 $A$，1-SS mask | 易做块化 matmul，加速明显 | 长序列、高吞吐训练与推理 |
| 一般 structured attention | 自定义结构掩码 $L$ | 研究空间大 | 探索新归纳偏置 |
| 标准 Transformer attention | 一般 causal mask + softmax | 生态成熟，表达强 | 通用基线与成熟工程栈 |

结论可以压缩成一句话：

**如果你的目标是“更强状态建模”，优先看更一般的 SSM；如果你的目标是“把 SSM 重排成更适合 Tensor Core 的形式”，SSD 是更直接的路线。**

---

## 参考资料

1. Tri Dao, “State Space Duality (Mamba-2) Part II - The Theory”, 2024.  
   https://tridao.me/blog/2024/mamba2-part2-theory/

2. Tri Dao, “State Space Duality (Mamba-2) Part I - The Model”, 2024.  
   https://tridao.me/blog/2024/mamba2-part1-model/

3. Tri Dao, “State Space Duality (Mamba-2) Part III - The Algorithm”, 2024.  
   https://tridao.me/blog/2024/mamba2-part3-algorithm/

4. Princeton Language and Intelligence, “Mamba-2: Algorithms and Systems”, 2024.  
   https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems
