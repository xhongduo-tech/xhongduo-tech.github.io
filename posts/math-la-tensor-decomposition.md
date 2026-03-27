## 核心结论

张量可以先看成“多维数组”。矩阵是 2 阶张量，彩色图片常写成 3 阶张量，卷积层权重常写成 4 阶张量。所谓“阶”，就是这个数组有几个独立维度。

张量分解的目标，不是把数学写复杂，而是用更少参数近似表达原始结构。最常见的两类方法是：

| 方法 | 表达形式 | 直观含义 | 主要特点 |
| --- | --- | --- | --- |
| CP 分解 | $T\approx\sum_{r=1}^R a^{(r)}\otimes b^{(r)}\otimes c^{(r)}$ | 用若干个“秩一张量”相加恢复整体 | 参数少，但秩选择敏感 |
| Tucker 分解 | $T\approx G\times_1 U_1\times_2 U_2\times_3 U_3$ | 每个维度先找低维子空间，再由小核张量组合 | 更灵活，通常更稳 |

这里的“外积”可以白话理解为：把多个向量按不同维度拼成一个多维模式；“mode-n 乘积”可以理解为：沿第 $n$ 个维度做线性变换。

一个关键关系是：Tucker 是更通用的框架。如果核心张量 $G$ 只在“对角线方向”有非零值，Tucker 就会退化成 CP。因此可以把 CP 看成 Tucker 的特殊情况。

在工程里，张量分解最有价值的用途是模型压缩。比如卷积层权重张量
$$
W\in\mathbb{R}^{C_{out}\times C_{in}\times k\times k}
$$
做 Tucker-2 分解后，常能压缩 80% 以上参数，同时保持可接受精度。Transformer 里的多头注意力虽然通常不直接叫“张量分解”，但本质上也在处理形如 $[B,H,T,d]$ 的张量，并通过 `einsum` 执行张量收缩。

---

## 问题定义与边界

问题可以表述为：给定一个高维张量，能否找到一个低秩结构，使存储量更小、计算更快，同时误差可控。

这里“低秩”不是只对矩阵说的。到了张量场景，秩不再只有一种定义。CP 秩表示需要多少个秩一张量才能近似原张量；Tucker 秩表示每个模上分别保留多少个主方向。

以卷积层为例，4 维权重张量写成
$$
W\in\mathbb{R}^{C_{out}\times C_{in}\times k\times k}.
$$
这四个维度分别表示输出通道、输入通道、高、宽。对新手而言，可以把它理解成“很多个卷积核按输入通道和输出通道堆叠起来”。

但不是每个维度都同等适合压缩。实践里通常更优先压缩通道维，而不是空间维。原因很直接：当 $k=3$ 或 $k=5$ 时，空间维本来就小，继续压缩收益有限；而 $C_{in},C_{out}$ 往往很大，才是参数主要来源。

| 模式 | 物理含义 | 常见规模 | 是否优先压缩 | 原因 |
| --- | --- | --- | --- | --- |
| mode-1 | 输出通道 $C_{out}$ | 大 | 是 | 决定卷积核数量，参数占比高 |
| mode-2 | 输入通道 $C_{in}$ | 大 | 是 | 决定每个核接收的信息宽度 |
| mode-3 | 空间高度 $k$ | 小 | 通常否 | 3x3、5x5 时收益有限 |
| mode-4 | 空间宽度 $k$ | 小 | 通常否 | 与高度同理 |

这就是为什么很多 CNN 压缩方法会采用 Tucker-2，只压缩输入通道和输出通道两个模。

边界也必须讲清楚：

1. 张量分解不是无损压缩。除非原张量本身低秩，否则一定有近似误差。
2. 参数减少不等于延迟一定下降。硬件是否擅长执行分解后的算子，影响很大。
3. 秩不能拍脑袋选。秩过低会损失表达能力，秩过高则可能压不动，甚至参数反增。

一个简单判断标准是：如果某层通道很小、原本已经很轻，分解往往意义不大；如果某层是大通道卷积或大投影矩阵，分解价值通常更高。

---

## 核心机制与推导

先从“展开”开始。mode-n 展开，就是把第 $n$ 个模保留下来作为矩阵的行，把其他模压成列。白话说，就是从某一个维度观察整个张量，把多维结构临时摊平成二维。

对 3 阶张量 $T\in\mathbb{R}^{I_1\times I_2\times I_3}$，其 mode-1 展开可记为
$$
T_{(1)}\in\mathbb{R}^{I_1\times (I_2I_3)}.
$$
这样做的意义是：很多线性代数工具先在矩阵上定义，展开后就能用 SVD、特征值分解等成熟方法分析张量结构。

### 玩具例子

考虑一个最小例子：
$$
T\in\mathbb{R}^{2\times2\times1},\quad
T(:,:,1)=
\begin{bmatrix}
1 & 2\\
3 & 6
\end{bmatrix}.
$$

这个矩阵两列线性相关，所以本质上是秩 1。它可以写成
$$
T=[1,3]^T\otimes[1,2]^T\otimes[1].
$$

这就是 CP 秩为 1 的情形。这里“秩一张量”可以理解为：每个维度只有一个方向，三者相乘后形成完整结构。

如果用 Tucker 表示，也可以取
$$
G\in\mathbb{R}^{1\times1\times1},\quad G=[1],
$$
再令
$$
U_1=[1,3]^T,\quad U_2=[1,2]^T,\quad U_3=[1].
$$
则同样能重构原张量。这个例子说明：当数据本身结构简单时，低秩表示可以几乎不损失信息。

### CP 分解的机制

CP 分解把张量写成若干个秩一张量的和：
$$
T\approx\sum_{r=1}^R a^{(r)}\otimes b^{(r)}\otimes c^{(r)}.
$$

如果把所有向量收集成因子矩阵：
$$
A=[a^{(1)},\dots,a^{(R)}],\;
B=[b^{(1)},\dots,b^{(R)}],\;
C=[c^{(1)},\dots,c^{(R)}],
$$
那么 CP 的本质就是：找到 $R$ 组跨模方向，使它们叠加后尽量接近原张量。

优点是参数量很省。对 3 阶张量，原始参数量是 $I_1I_2I_3$，CP 近似后约为 $R(I_1+I_2+I_3)$。但缺点也明显：$R$ 很难选，优化问题也更不稳定。

### Tucker 分解的机制

Tucker 分解把张量写成：
$$
T\approx G\times_1 U_1\times_2 U_2\times_3 U_3.
$$

这里核心张量 $G$ 可以理解为“小型交互中心”，它描述各模低维方向之间如何组合；$U_i$ 表示第 $i$ 个模的基方向，也就是“压缩后的坐标轴”。

如果用矩阵观点看，Tucker 先在每个模上做降维，再用核心张量保留多维耦合关系。相比 CP，它允许不同维度取不同秩：
$$
G\in\mathbb{R}^{R_1\times R_2\times R_3}.
$$

这就是为什么 Tucker 更灵活。比如图像任务里，通道维可以压得更激进，空间维则保守一些。

### 卷积层里的 Tucker-2 推导

对卷积核张量
$$
W\in\mathbb{R}^{C_{out}\times C_{in}\times k\times k},
$$
常见的 Tucker-2 只压缩通道维，不压缩空间维：
$$
W(t,s,i,j)=\sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2} w_1(r_1,s)\,U(r_2,r_1,i,j)\,w_3(t,r_2).
$$

这个式子可以直接对应成三层卷积：

1. 第一层 `1x1` 卷积，把输入通道 $C_{in}$ 压到 $R_1$。
2. 第二层 `k x k` 卷积，在低维通道空间里做主要空间计算。
3. 第三层 `1x1` 卷积，把 $R_2$ 升回输出通道 $C_{out}$。

所以 Tucker-2 不是抽象数学，它等价于一个可直接部署的网络结构替换。

### Transformer 里的张量收缩

多头注意力里，输入常先从 $[B,T,D]$ 映射为查询、键、值，再 reshape 成 $[B,H,T,d]$。这里：

- $B$ 是 batch，大意是一批样本
- $H$ 是头数，大意是并行子空间数
- $T$ 是序列长度
- $d$ 是每个头的通道维

常见实现会用 `einsum`，例如：
```python
Q = torch.einsum("btd,hdk->bhtk", x, w_q)
```

`einsum` 可以白话理解为“按索引名字写收缩规则”。它不是逐个头 for-loop 地算，而是一次性对整个张量执行并行收缩。这也是现代深度学习框架高效实现注意力的核心手段之一。

---

## 代码实现

下面用一个最小可运行例子展示：如何构造秩 1 张量、如何重建，以及卷积 Tucker-2 的参数量估算。

```python
import numpy as np

def cp_rank1_reconstruct(a, b, c):
    # 外积得到 3 阶张量
    return np.einsum("i,j,k->ijk", a, b, c)

def mode_n_unfold(X, mode):
    # 把第 mode 个维度移到最前，再展平成矩阵
    order = [mode] + [i for i in range(X.ndim) if i != mode]
    transposed = np.transpose(X, order)
    return transposed.reshape(X.shape[mode], -1)

def tucker_reconstruct(core, factors):
    X = core
    for mode, U in enumerate(factors):
        X = np.tensordot(U, X, axes=(1, mode))
        # tensordot 后新维度被放到最前，转回原顺序
        axes = list(range(1, mode + 1)) + [0] + list(range(mode + 1, X.ndim))
        X = np.transpose(X, axes)
    return X

# 玩具例子：T(:,:,1) = [[1,2],[3,6]]
a = np.array([1.0, 3.0])
b = np.array([1.0, 2.0])
c = np.array([1.0])

T = cp_rank1_reconstruct(a, b, c)
expected = np.array([[[1.0], [2.0]],
                     [[3.0], [6.0]]])

assert T.shape == (2, 2, 1)
assert np.allclose(T, expected)

# mode-0 展开后应为 2 x 2
unfolded = mode_n_unfold(T, 0)
assert unfolded.shape == (2, 2)

# Tucker 形式重建同一个张量
core = np.array([[[1.0]]])
U1 = np.array([[1.0], [3.0]])
U2 = np.array([[1.0], [2.0]])
U3 = np.array([[1.0]])

T2 = tucker_reconstruct(core, [U1, U2, U3])
assert np.allclose(T2, expected)

def conv_tucker2_param_count(c_out, c_in, k, r1, r2):
    original = c_out * c_in * k * k
    decomposed = r1 * c_in + r2 * r1 * k * k + c_out * r2
    return original, decomposed

orig, decomp = conv_tucker2_param_count(256, 256, 3, 64, 64)
assert orig == 256 * 256 * 9
assert decomp == 64 * 256 + 64 * 64 * 9 + 256 * 64
assert decomp < orig
```

如果把它迁移到深度学习框架，卷积层的 Tucker-2 结构通常可以写成下面这种伪代码：

```python
import torch
import torch.nn as nn

class Tucker2Conv(nn.Module):
    def __init__(self, c_in, c_out, rank_in, rank_out, k=3):
        super().__init__()
        self.reduce = nn.Conv2d(c_in, rank_in, kernel_size=1, bias=False)
        self.core = nn.Conv2d(rank_in, rank_out, kernel_size=k, padding=k // 2, bias=False)
        self.expand = nn.Conv2d(rank_out, c_out, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.reduce(x)
        x = self.core(x)
        x = self.expand(x)
        return x
```

多头注意力中的张量收缩，也可以用最小代码写清楚：

```python
# x: [B, T, D]
# w_q: [H, D, d]
q = torch.einsum("btd,hdk->bhtk", x, w_q)
# q: [B, H, T, d]
```

真实工程例子是 CNN 压缩。假设原始卷积为 `256 -> 256, 3x3`，参数量是：
$$
256\times256\times3\times3=589{,}824.
$$
若取 $R_1=R_2=64$，Tucker-2 后参数量变成：
$$
64\times256 + 64\times64\times3\times3 + 256\times64 = 69{,}632.
$$
压缩比例约为 88.2%。这就是“分解能省很多参数”的直接来源。

---

## 工程权衡与常见坑

张量分解在论文里看起来往往很顺，但落到工程里，主要问题集中在秩选择、算子支持和微调稳定性。

| 维度 | 优点 | 风险 |
| --- | --- | --- |
| 参数量 | 可显著下降 | 秩选不好会压不动 |
| FLOPs | 理论上可减少 | 实际延迟依赖硬件内核 |
| 精度 | 合理 rank 下损失可控 | rank 过低会明显退化 |
| 部署 | 可转成标准卷积/矩阵乘 | 算子链变长后调度成本上升 |

最常见的坑有四类。

第一，rank 过低。  
这会导致表达能力不足。表现通常是训练后精度明显掉，尤其在分类边界复杂或特征耦合强的层上更明显。

第二，rank 过高。  
这时核心张量和因子矩阵加起来可能并不比原层小多少，甚至更大。Tucker 尤其要注意这一点，因为它有核心张量开销。

第三，只分解不微调。  
绝大多数情况下，直接把预训练权重分解后替换进模型，精度会下降。原因不复杂：分解是近似，不是完全等价。通常需要全模型微调，把误差重新吸收回去。

第四，硬件不友好。  
新手很容易以为“参数少了就一定更快”，这是错误的。比如某些 CPU 或移动端对标准 `3x3` 卷积优化很好，但对多层 `1x1 + 3x3 + 1x1` 的调度并不一定占优；如果 group 卷积或小 batch 表现差，实际延迟可能反而上升。

下面这个表格可以直接帮助判断 rank 调整方向：

| 情况 | 现象 | 处理建议 |
| --- | --- | --- |
| rank 过低 | 精度明显下降 | 先增大通道秩，再看是否恢复 |
| rank 过高 | 压缩率不明显 | 优先缩减通道维秩 |
| 分解后训练震荡 | 微调不稳定 | 降低学习率，分层微调 |
| 理论 FLOPs 降了但更慢 | 延迟未改善 | 做真实 profile，不只看参数量 |

自动选 rank 是更稳的路线。常见方法是 VBMF、基于能量保留的截断、或者以延迟预算为约束的搜索。一个实用流程是：

1. 先对候选层做敏感度分析。
2. 对每层的展开矩阵估计有效秩。
3. 先分解高开销层，再整体微调。
4. 最后以真实设备延迟而不是 FLOPs 作为验收标准。

---

## 替代方案与适用边界

CP 和 Tucker 不是唯一方案。工程上通常要和 TT 分解、矩阵低秩分解、量化一起比较。

| 方法 | 压缩率 | 实现复杂度 | 适合对象 | 推理平台友好度 |
| --- | --- | --- | --- | --- |
| CP | 高 | 中 | 高阶张量，追求极限压缩 | 一般 |
| Tucker | 高 | 中 | CNN 权重张量，按模调秩 | 较好 |
| TT | 很高 | 高 | 更高阶张量或超大权重 | 一般 |
| 矩阵低秩 | 中 | 低 | 线性层、投影层 | 很好 |
| 量化 | 高 | 中 | 全模型普适 | 依赖硬件支持 |

它们的适用边界可以概括成三条。

第一，如果目标层天然是高阶张量，且各模语义明确，Tucker 往往是更稳的起点。卷积层就是典型例子，因为输入通道、输出通道、空间维度天然分开。

第二，如果你只处理线性层或注意力投影矩阵，先试矩阵低秩分解通常更直接，因为实现更简单、框架支持也更成熟。

第三，如果部署平台对低比特计算支持很好，量化往往比张量分解更容易拿到真实速度收益。张量分解更像“结构压缩”，量化更像“数值压缩”，两者也可以叠加。

一个真实工程边界案例是视觉问答模型。像 DecomVQANet 这类工作，会联合使用 CP、Tucker 和 TT 去压缩 CNN 与序列模块，整体参数可减少约 80%，而性能只小幅下降。这说明：在多模块模型中，往往不是“单一分解法通杀”，而是按层类型混合选型。

因此可以给出一个朴素决策规则：

- 卷积通道很大，且平台擅长 `1x1` 卷积：优先 Tucker-2。
- 需要极限压缩，能接受更复杂实现：考虑 CP 或 TT。
- 主要瓶颈是线性层：先做矩阵低秩。
- 主要瓶颈是部署吞吐：优先验证量化。

---

## 参考资料

- Little Book of Matrix and Tensor Algebra: 说明张量、mode-n 展开、CP 与 Tucker 的基础定义，适合作为入门数学参考。
- PMC 上关于张量分解与 HOSVD 的综述/教程：解释 Tucker 分解、核张量、正交因子矩阵，以及不同秩定义的关系。
- MDPI 2023 关于卷积层张量分解的论文：给出卷积核的 CP/Tucker 表达，并说明 Tucker-2 可转写成 `1x1 -> kxk -> 1x1` 结构。
- Kim 等关于 CNN 压缩的工作：较早系统展示了对卷积层做 Tucker-2 分解并配合微调的工程路径。
- DecomVQANet 相关论文：展示在视觉问答模型中混合使用 CP/Tucker/TT 的压缩效果与精度折中。
- HuggingFace 与 TensorRT-LLM 中的 `einsum`/注意力实现示例：说明多头注意力本质上是张量收缩，而不是简单的逐头串行计算。
