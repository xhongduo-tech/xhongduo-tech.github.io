## 核心结论

Mamba-2 的关键贡献，不是“又设计了一种新层”，而是给出了一个统一视角：当状态空间模型（SSM，白话说就是“用一个隐藏状态递推地记忆过去输入的模型”）里的状态转移矩阵满足 $A=\alpha I$ 时，递归更新和因果线性注意力可以写成同一个输入输出映射。

更具体地说，离散 SSM
$$
x_{t+1}=A x_t + B u_t,\qquad y_t = C x_t + D u_t
$$
展开后，输出对历史输入的核为
$$
M_{t,s}=C A^{t-s-1} B,\qquad s<t.
$$
如果 $A=\alpha I$，那么
$$
M_{t,s}=\alpha^{\,t-s-1}(CB).
$$
这意味着它可以拆成“因果衰减项”和“低秩内容项”的乘积，本质上就是一个带因果 mask 的半可分矩阵乘法，形式上与因果线性注意力一致。

这件事的工程价值很直接：同一个数学对象，既可以按 SSM 的方式做递推 scan，也可以按 attention 的方式做矩阵乘法 matmul。SSD（Structured State-Space Duality，白话说就是“结构化状态空间对偶”）算法正是利用这个等价关系，把块内计算改写成大矩阵乘法，把块间信息传递保留为短递推，从而让 Mamba-2 在现代 GPU 上明显快于 Mamba-1。

一个新手版理解是：原来看起来必须“从左到右一步一步扫”的递归，其实在特定结构下，等价于“把过去所有位置一次性乘进来”的上三角矩阵乘法。结果不变，实现方式变了，硬件利用率也就变了。

---

## 问题定义与边界

这里要解决的问题不是“所有 SSM 都等价于 attention”，而是一个更精确的问题：

当 SSM 的状态转移矩阵满足 $A=\alpha I$，也就是每个状态维度都按同一个标量缩放时，它的因果卷积核能否写成一种可分的、适合矩阵乘法的结构？

答案是可以，而且这个结构正好是 1-半可分矩阵。半可分（semiseparable，白话说就是“矩阵的非对角部分能用很低秩表示”）意味着它既保留因果性，又允许高效实现。

边界也要说清楚：

| 维度 | SSM 递推视角 | SSD/线性注意力视角 |
| --- | --- | --- |
| 输入增益 | $B$ 把输入写入状态 | 类似 key/value 投影中的“写入强度” |
| 状态缩放 | $A=\alpha I$ 控制历史衰减 | 因果 mask 中的 $\alpha^{t-s}$ |
| 输出读出 | $C$ 从状态中读信息 | 类似 query 与状态内容的匹配读出 |
| 直接通路 | $D u_t$ 直接加到输出 | 相当于当前位置的跳连项 |

这个结论只在特定结构下最干净：
1. 需要 $A$ 具有标量乘单位阵的形式，或者至少能被很好近似成这种形式。
2. 需要序列足够长、硬件足够偏向 matmul，矩阵化实现的优势才明显。
3. 需要处理好数值稳定性，否则理论等价不代表训练稳定。

玩具例子先看最小版本。设状态递推为
$$
x_{t+1}=0.5x_t+u_t,\qquad y_t=x_t,
$$
且 $x_0=0$。输入序列为 $u=[2,4,8]$，则递推结果为
$$
x_1=2,\quad x_2=5,\quad x_3=10.5.
$$
如果换成矩阵形式，就是
$$
\begin{bmatrix}
y_1\\y_2\\y_3
\end{bmatrix}
=
\begin{bmatrix}
1&0&0\\
0.5&1&0\\
0.25&0.5&1
\end{bmatrix}
\begin{bmatrix}
2\\4\\8
\end{bmatrix}
=
\begin{bmatrix}
2\\5\\10.5
\end{bmatrix}.
$$
这已经展示出“递推”和“因果 mask 乘法”是同一个映射。

---

## 核心机制与推导

从离散 SSM 开始：
$$
x_{t+1}=A x_t + B u_t,\qquad y_t=Cx_t.
$$
把递推展开：
$$
x_t=A^{t-1}x_1+\sum_{s=1}^{t-1}A^{t-s-1}Bu_s.
$$
若忽略初始状态或把它并入边界项，输出可写为
$$
y_t=\sum_{s=1}^{t-1} C A^{t-s-1} B\,u_s.
$$
因此核函数为
$$
M_{t,s}=C A^{t-s-1} B.
$$

现在加上 Mamba-2 讨论的关键约束：
$$
A=\alpha I.
$$
因为单位阵的幂还是单位阵，所以
$$
A^{t-s-1}=(\alpha I)^{t-s-1}=\alpha^{t-s-1}I.
$$
代回去得到
$$
M_{t,s}=\alpha^{t-s-1}(CB).
$$

这一步是 SSD 的核心。它说明这个核有两部分：

1. $\alpha^{t-s-1}$ 只依赖相对位置 $t-s$，控制历史衰减。
2. $CB$ 与状态内容相关，但不再依赖时间差的矩阵幂。

于是可以写成
$$
M = L \circ K,
$$
其中 $\circ$ 是 Hadamard 积，也就是逐元素乘法。这里
$$
L_{t,s}=
\begin{cases}
\alpha^{t-s-1}, & s<t\\
0, & s\ge t
\end{cases}
$$
是因果衰减 mask，$K$ 是由内容项构成的低秩核。在 attention 写法里，这对应
$$
Y=(L\circ QK^\top)V.
$$

为什么说 $L$ 是 1-半可分？直观上看，每一行都是上一行按常数 $\alpha$ 缩放后再右移一格。例如 $\alpha=0.5$ 时，长度为 4 的下三角部分是
$$
\begin{bmatrix}
1&0&0&0\\
0.5&1&0&0\\
0.25&0.5&1&0\\
0.125&0.25&0.5&1
\end{bmatrix}.
$$
它的严格下三角部分满足“每一列是几何衰减、每一行由前一行缩放得到”的结构，因此可以用很低维的生成向量表示，而不必显式存整个矩阵。这就是半可分结构的本质。

玩具例子继续展开。设 $\alpha=0.5$，$B=C=1$，输入 $u=[1,2,3]$。则
$$
M=
\begin{bmatrix}
1&0&0\\
0.5&1&0\\
0.25&0.5&1
\end{bmatrix},
\qquad
Mu=
\begin{bmatrix}
1\\2.5\\4.25
\end{bmatrix}.
$$
如果按递推算：
$$
x_1=1,\quad x_2=0.5\cdot 1+2=2.5,\quad x_3=0.5\cdot 2.5+3=4.25.
$$
两者完全一致。

真实工程例子是长上下文语言模型。假设输入长度是 16K，如果仍用传统 scan，全序列每一步都依赖上一步，GPU 很难把大部分算力用在 tensor core 上。SSD 的做法是把序列分成长度为 $Q$ 的 chunk：
- chunk 内依赖可写成半可分矩阵乘法，交给 matmul；
- chunk 与 chunk 之间只保留压缩后的状态传递，交给短 scan。

这样做并没有改变数学映射，但把“长串行”变成了“短串行 + 大并行”。

---

## 代码实现

下面先给一个可运行的 Python 例子，验证“递推形式”和“半可分矩阵乘法形式”在 $A=\alpha I$ 时得到同样结果。

```python
import numpy as np

def ssm_scan(u, alpha, B=1.0, C=1.0):
    x = 0.0
    ys = []
    for ut in u:
        x = alpha * x + B * ut
        ys.append(C * x)
    return np.array(ys, dtype=float)

def semiseparable_matmul(u, alpha, B=1.0, C=1.0):
    n = len(u)
    M = np.zeros((n, n), dtype=float)
    for t in range(n):
        for s in range(t + 1):
            M[t, s] = (alpha ** (t - s)) * (C * B)
    return M @ np.array(u, dtype=float)

u = np.array([1.0, 2.0, 3.0, 4.0])
y_scan = ssm_scan(u, alpha=0.5)
y_matmul = semiseparable_matmul(u, alpha=0.5)

assert np.allclose(y_scan, y_matmul)
assert np.allclose(y_scan, np.array([1.0, 2.5, 4.25, 6.125]))

print(y_scan)
```

这个例子里：
- `ssm_scan` 是标准递推；
- `semiseparable_matmul` 直接构造了因果半可分矩阵；
- `assert` 验证两种实现输出一致。

如果继续向工程实现靠近，重点不是显式构造整个 $M$，而是分块。下面是与 SSD 思路一致的抽象伪代码：

```python
def ssd_layer(seq, chunk_size, state):
    outputs = []
    for chunk in split(seq, chunk_size):
        block_kernel = compute_semiseparable_kernel(chunk)
        local_out = matmul(block_kernel, chunk)   # 步骤1/2：块内并行
        state, carry_out = recursive_scan(state, chunk)  # 步骤3：块间递推
        y_chunk = merge(local_out, carry_out)
        outputs.append(project_out(y_chunk))      # 步骤4：输出投影
    return concat(outputs), state
```

这里的设计原则是：
- 大部分 FLOPs 尽量留给 `matmul`；
- `recursive_scan` 只处理块间摘要状态，不在全序列上串行展开；
- `chunk_size=Q` 是核心调参项。

一个简单经验表如下：

| chunk size $Q$ | 优点 | 风险 |
| --- | --- | --- |
| 小 | 块间状态更容易控制，cache 压力小 | matmul 不够大，吞吐上不去 |
| 中 | 通常是最佳平衡点 | 需要针对硬件调参 |
| 大 | tensor core 利用率高 | 块内显存和数值范围压力增加 |

真实工程里，Mamba-2 的性能优势就来自这种分工：块内走矩阵乘法，块间走短递推。它不是取消递推，而是把递推缩到更适合串行的那一小部分。

---

## 工程权衡与常见坑

理论上最容易犯的错，是以为“既然有 $\alpha^{t-s}$，那我直接 `cumprod` 构造整个衰减矩阵就行”。这在长序列上通常不稳。

原因很简单。若 $\alpha=0.5$，则
$$
0.5^{512}\approx 7.46\times 10^{-155}.
$$
在更长序列、混合精度训练、带梯度传播的场景里，这种数会快速接近下溢区间。结果不是“只是变小一点”，而是大面积归零、梯度消失，进一步触发归一化或除法相关路径的不稳定，最后出现 NaN。

下面是两类实现的对比：

| 实现方式 | 做法 | 数值后果 | 工程结论 |
| --- | --- | --- | --- |
| 不稳定实现 | 直接 `cumprod` 或显式幂次构造长 mask | 长序列下趋零、下溢、梯度差 | 不适合训练主路径 |
| 稳定实现 | `segsum`、log-space、块矩阵重写 | 范围可控，便于混合精度 | 适合训练和高性能实现 |

这里的 `segsum` 可以理解成“分段求和再指数化”，白话说就是先在更稳定的加法域里累计，再回到乘法域，避免很多极小数直接连乘。

下面给一个最小数值观察：

```python
import numpy as np

alpha = 0.5
v_32 = np.float32(alpha) ** 128
v_16 = np.float16(alpha) ** 32

assert v_32 > 0.0
assert v_16 == 0.0  # 半精度下很快下溢

print(v_32, v_16)
```

这不是说“0.5 不好”，而是说显式幂次在低精度下太脆弱。SSD 的正确方向，是避免把长距离衰减直接物化成一整块稠密矩阵。

另一个常见坑是块大小 $Q$ 的选择。$Q$ 不是越大越好：
- 太小，矩阵乘法规模不够，GPU 跑不满；
- 太大，块内中间张量膨胀，cache miss 和显存占用上升；
- 训练时还会放大数值范围波动，尤其在低精度下更明显。

一个实用流程通常是：
1. 先固定模型维度与状态维度。
2. 在目标硬件上扫描几个候选 $Q$。
3. 同时观察吞吐、显存峰值、loss 是否稳定。
4. 不只看 tokens/s，也看是否出现 NaN 或 loss 抖动。

---

## 替代方案与适用边界

SSD 不是“统一天下”的答案，它有明确边界。

第一类边界是结构边界。如果 $A\neq \alpha I$，例如不同状态维度有不同特征值，或者 $A$ 是更一般的非对角矩阵，那么
$$
A^{t-s-1}
$$
不再能简化成一个纯标量衰减项。此时核函数未必还能写成干净的 1-半可分结构，low-rank 分解也可能不稳定或近似误差过大。这种情况下，更稳妥的做法通常是：
- 回到原始 SSM scan；
- 或使用更通用的线性注意力 / 多头注意力。

第二类边界是硬件边界。SSD 的优势依赖高效 matmul。如果是在纯 CPU、边缘设备、或序列本来就很短的场景，复杂分块的收益可能覆盖不了实现成本。此时直接 scan 反而更简单。

第三类边界是任务边界。如果任务特别依赖灵活的内容寻址，而不是固定形式的状态衰减，那么标准 attention 仍然更自然，因为它的权重是内容驱动的，而不是主要由结构化递推决定。

可以用一个简单决策表概括：

| 硬件 | 序列长度 | 矩阵结构 | 推荐方案 |
| --- | --- | --- | --- |
| GPU，tensor core 强 | 长序列 | $A\approx \alpha I$ 或可低秩化 | SSD 分块 matmul + 短 scan |
| GPU/CPU 都可 | 中短序列 | 一般结构 | 原始 SSM scan |
| GPU 强 | 长序列 | 内容交互复杂，结构不规则 | 注意力或线性注意力 |
| CPU 为主 | 短序列 | 任意 | 简单 scan，优先实现复杂度低 |

所以，SSD 更准确的定位不是“替代 attention”，而是“在一类有结构的 SSM 上，把递推和 attention 放进同一计算框架”。

---

## 参考资料

1. Dao, T., Gu, A. 《Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality》, ICML 2024。适合看数学定义、对偶性证明和统一形式化表达。  
2. Emergent Mind, “Structured State-Space Duality”。适合初学者快速建立直觉，理解为什么递推系统与 masked attention 可以视为同一个矩阵映射。  
3. Princeton Language and Intelligence, “Mamba-2: Algorithms and Systems”。适合看工程实现，尤其是 chunk、matmul/scan 分工、数值稳定性和 benchmark 讨论。
