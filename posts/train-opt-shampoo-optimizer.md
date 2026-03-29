## 核心结论

Shampoo 是一种面向矩阵或张量参数的全矩阵预条件优化器。预条件的白话解释是：先根据历史梯度的形状，给不同方向做不同缩放，再决定这一步往哪里走。它不是像 AdamW 那样只给每个参数一个标量缩放，而是分别在“行空间”和“列空间”上学习梯度几何结构，所以更接近二阶方法。

对矩阵参数 $W\in\mathbb{R}^{m\times n}$，Shampoo 维护两个统计量：
$$
L_t=\sum_{s=1}^t G_sG_s^\top,\qquad
R_t=\sum_{s=1}^t G_s^\top G_s
$$
其中 $G_t=\nabla_W\ell_t$ 是当前梯度。更新写成：
$$
W_{t+1}=W_t-\eta\,L_t^{-1/4}G_tR_t^{-1/4}
$$
这里的逆四次根可以理解为“把梯度在行和列两个方向上都标准化”。

它的优势不是“每步更便宜”，而是“同样训练目标下，方向更好”。这也是为什么它常被视为比 Adam/AdaGrad 更强的结构感知自适应方法。代价也很直接：要维护矩阵统计量，还要周期性求矩阵逆根，单次代价接近 $O(m^3)+O(n^3)$，大矩阵上必须工程化处理。

一个最小玩具例子能直接说明它的作用。若
$$
G=\operatorname{diag}(2,4)
$$
则
$$
L=GG^\top=\operatorname{diag}(4,16),\qquad
R=G^\top G=\operatorname{diag}(4,16)
$$
于是
$$
L^{-1/4}=\operatorname{diag}(1/\sqrt2,1/2),\quad
R^{-1/4}=\operatorname{diag}(1/\sqrt2,1/2)
$$
所以
$$
L^{-1/4}GR^{-1/4}=\operatorname{diag}(1,1)
$$
原本两个方向的梯度幅度是 2 和 4，经过 Shampoo 后变成同尺度更新。它修正的不是单点大小，而是整个矩阵的几何结构。

下表先给一个直观对比：

| 优化器 | 预条件形式 | 捕捉结构 | 单步额外代价 | 典型适用场景 |
|---|---|---|---|---|
| SGD | 无 | 否 | 低 | 基线训练、强正则场景 |
| AdamW | 对角缩放 | 仅单参数尺度 | 低 | 通用默认选择 |
| AdaGrad | 对角累计平方梯度 | 仅单参数尺度 | 低 | 稀疏特征、凸优化 |
| Shampoo | 行/列矩阵预条件 | 是 | 高 | 大层矩阵、追求更快收敛 |
| Distributed Shampoo | 分片后的 Shampoo | 是 | 中到高 | 多 GPU / 大模型训练 |

---

## 问题定义与边界

问题本质是：当参数本身有二维或更高维结构时，只做逐元素缩放会丢失方向之间的相关性。相关性的白话解释是：某一行的变化，常常会和某一列的变化一起出现，它们不是互相独立的。

以全连接层权重 $W\in\mathbb{R}^{m\times n}$ 为例，梯度 $G$ 也是一个矩阵。AdamW 会给 $G_{ij}$ 各自一个学习率修正，但它不会回答一个更重要的问题：如果第 3 行和第 7 行总是联动波动，或者某几列形成一个强耦合子空间，更新方向应该如何整体旋转和压缩？

Shampoo 的边界定义得很清楚：

| 维度形态 | 是否适合直接用 Shampoo | 主要瓶颈 | 常见处理 |
|---|---|---|---|
| 小矩阵，如 $64\times64$ | 适合 | 额外实现复杂度 | 可直接做特征分解 |
| 中矩阵，如 $1024\times4096$ | 视资源而定 | 逆根计算贵 | 分块、低频更新 |
| 超大矩阵，如 LLM MLP / Attention 投影 | 不适合原始做法 | 计算和显存都重 | Distributed Shampoo、量化、分片 |
| 向量参数或很小 embedding | 收益有限 | 结构信息少 | AdamW/Adafactor 常更划算 |

对白话版本可以这样理解：把一层梯度看成一张表，Shampoo 不只看表里每个格子有多大，还看整行和整列怎样一起抖动，然后把更新方向重新拉伸到更合理的尺度，避免某几行或某几列单独爆发。

它只解决“矩阵级别的二阶近似预条件”问题，不解决所有训练稳定性问题。学习率、权重衰减、梯度裁剪、batch size 和混合精度，仍然会直接影响结果。

---

## 核心机制与推导

Shampoo 可以看成“全矩阵 AdaGrad 的可分解近似”。可分解的白话解释是：本来要维护一个巨大协方差矩阵，现在拆成行统计和列统计两个较小矩阵。

对当前梯度 $G_t\in\mathbb{R}^{m\times n}$，定义：
$$
L_t=L_{t-1}+G_tG_t^\top,\qquad
R_t=R_{t-1}+G_t^\top G_t
$$
其中：

- $L_t\in\mathbb{R}^{m\times m}$ 描述行方向累计波动
- $R_t\in\mathbb{R}^{n\times n}$ 描述列方向累计波动

如果把完整矩阵展平成向量，理想二阶预条件会非常大。Shampoo 用 Kronecker 结构近似它：
$$
\mathcal{P}_t \approx L_t^{1/2}\otimes R_t^{1/2}
$$
因此其逆预条件对应到矩阵形式，就是左右各乘一个逆四次根：
$$
\widetilde{G}_t=L_t^{-1/4}G_tR_t^{-1/4}
$$

为什么是 $-1/4$ 而不是 $-1/2$？因为左右各作用一次，组合后等价于对 Kronecker 结构整体施加一次逆平方根。直观看，左乘负责修正“行方向曲率”，右乘负责修正“列方向曲率”。

继续看前面的玩具例子：
$$
G=\begin{bmatrix}
2&0\\
0&4
\end{bmatrix}
$$
SGD 会直接沿着 $\operatorname{diag}(2,4)$ 更新，第二个方向步子是第一个方向的两倍。Shampoo 则把它转成 $\operatorname{diag}(1,1)$，说明它认为两个方向都只是“同样陡”，只是原坐标尺度不同。

如果 $G$ 不是对角矩阵，而是
$$
G=\begin{bmatrix}
2&2\\
0&4
\end{bmatrix}
$$
那 $L_t$ 和 $R_t$ 一般都不是对角的。非对角的意思是方向之间相关。此时特征分解
$$
L_t=Q_L\Lambda_L Q_L^\top,\qquad
R_t=Q_R\Lambda_R Q_R^\top
$$
就给出一套新的正交基。正交基的白话解释是：先换到一组互相独立的新坐标轴，再按每个轴的难易程度缩放。于是
$$
L_t^{-1/4}=Q_L\Lambda_L^{-1/4}Q_L^\top,\qquad
R_t^{-1/4}=Q_R\Lambda_R^{-1/4}Q_R^\top
$$
这一步既包含缩放，也包含旋转。

从工程角度看，完整流程通常是：

1. 每步累积 $L_t,R_t$
2. 每隔 $f$ 步才重新计算一次 $L_t^{-1/4},R_t^{-1/4}$
3. 中间步直接复用缓存的逆根矩阵
4. 用缓存后的预条件梯度做参数更新

这里的 $f$ 就是预条件更新频率。$f$ 越大，单步便宜，但方向会滞后；$f$ 越小，方向更新更准，但代价更高。

---

## 代码实现

下面先给一个可运行的最小 Python 版本，展示玩具例子中的核心计算。代码只依赖 `numpy`，并用 `assert` 检查结果。

```python
import numpy as np

def inv_quarter_root(mat, eps=1e-12):
    # 对称正定矩阵的逆四次根：Q diag(lambda^(-1/4)) Q^T
    vals, vecs = np.linalg.eigh(mat + eps * np.eye(mat.shape[0]))
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals ** (-0.25)) @ vecs.T

G = np.diag([2.0, 4.0])
L = G @ G.T
R = G.T @ G

L_inv_q = inv_quarter_root(L)
R_inv_q = inv_quarter_root(R)
precond_grad = L_inv_q @ G @ R_inv_q

assert np.allclose(precond_grad, np.eye(2), atol=1e-6)

eta = 0.1
W = np.zeros((2, 2))
W_next = W - eta * precond_grad

assert np.allclose(W_next, -0.1 * np.eye(2), atol=1e-6)
print(precond_grad)
print(W_next)
```

如果把它扩展成训练循环，核心伪代码通常是：

```python
state.L += grad @ grad.T
state.R += grad.T @ grad

if step % precondition_frequency == 0:
    state.inv_root_L = matrix_inverse_quarter_root(state.L + eps * I)
    state.inv_root_R = matrix_inverse_quarter_root(state.R + eps * I)

precond_grad = state.inv_root_L @ grad @ state.inv_root_R

if use_momentum:
    state.m = beta1 * state.m + (1 - beta1) * precond_grad
    update = state.m
else:
    update = precond_grad

param -= lr * update
```

缓存对象和更新频率通常这样设计：

| 缓存对象 | 作用 | 更新频率 |
|---|---|---|
| `L` | 行统计矩阵 | 每步 |
| `R` | 列统计矩阵 | 每步 |
| `inv_root_L` | 行方向逆四次根 | 每 `f` 步 |
| `inv_root_R` | 列方向逆四次根 | 每 `f` 步 |
| `momentum` | 平滑更新方向 | 每步 |

真实工程例子是大模型或 ResNet-50 训练。Distributed Shampoo 的做法不是把所有预条件器都复制到每张卡，而是把不同参数块对应的 $L/R$ 分到不同设备上，局部算完搜索方向后，再通过 `AllGather` 汇总每层的更新方向。简化后的分布式伪代码类似：

```python
# 每个设备持有一部分 preconditioner block
local_L += local_grad @ local_grad.T
local_R += local_grad.T @ local_grad

if step % f == 0:
    local_inv_root_L = inv_quarter_root(local_L + eps * I)
    local_inv_root_R = inv_quarter_root(local_R + eps * I)

local_search_dir = local_inv_root_L @ local_grad @ local_inv_root_R

# 再把各设备局部搜索方向聚合成完整更新
global_search_dir = all_gather(local_search_dir)
param -= lr * global_search_dir
```

Shi 等人在 2023 年的 Distributed Shampoo 论文里描述了基于 PyTorch DTensor 的实现，核心工程点就是“分片预条件器 + 每步 AllGather 搜索方向 + 低频更新逆根”。这让 Shampoo 从“论文里可行”变成“多 GPU 可用”。

---

## 工程权衡与常见坑

Shampoo 的主要矛盾只有一个：方向质量更高，但预条件器太贵。

第一类坑是逆根计算频率。若每步都对大矩阵做特征分解或矩阵根，开销会迅速主导训练时间。实践中常把逆根更新频率设为 20 到 100 步，靠缓存摊销成本。Distributed Shampoo 论文报告其多 GPU 实现相对常规对角自适应方法的单步 wall-clock 降低幅度可控制在约 10% 以内；很多工程总结会把 `f≈50` 的 ImageNet/ResNet-50 场景概括为“额外 wall-clock 开销很小，但达到相近精度所需轮数更少”。这里应理解为工程上可接受，而不是免费。

第二类坑是数值稳定性。若 $L_t$ 或 $R_t$ 接近奇异，逆四次根会不稳定。奇异的白话解释是：矩阵里有些方向几乎没有信息，求逆时会把噪声放大。常见补救是：
$$
L_t \leftarrow L_t + \varepsilon I,\qquad
R_t \leftarrow R_t + \varepsilon I
$$
并结合动量、梯度裁剪、warmup 一起使用。

第三类坑是通信。分布式 Shampoo 往往不是算不过来，而是同步不过来。尤其当参数块很多、设备数很多、batch 又不够大时，`AllGather` 会吃掉本来节省下来的收敛收益。

常见陷阱与对策如下：

| 陷阱 | 现象 | 对策 |
|---|---|---|
| `f=1` 每步更新逆根 | 训练明显变慢 | 提高到 20-100 步 |
| 未加 $\varepsilon I$ | 出现 NaN 或方向抖动 | 对 $L/R$ 加对角正则 |
| 参数矩阵过大 | 显存爆炸、特征分解过慢 | 分块 Shampoo |
| 分布式同步过密 | GPU 利用率低 | 合理分片、增大 batch、减少同步频率 |
| 直接照搬 AdamW 超参 | 收敛异常 | 重新调学习率、动量、weight decay |
| 小模型也强上 Shampoo | 训练不划算 | 回退到 AdamW |

一个很实际的经验是：Shampoo 不是“默认替代 AdamW”，而是“当你愿意为更好的方向质量付工程成本时”的选择。对小模型、小数据集、单卡训练，它常常不划算。对高维矩阵层很多、训练周期很长的大模型，它更可能把额外预处理成本赚回来。

---

## 替代方案与适用边界

如果资源足够，Shampoo 的优势来自结构感知预条件；如果资源紧，就要考虑近似版本。

| 方法 | 额外状态 | 方向质量 | 计算/通信成本 | 适用边界 |
|---|---|---|---|---|
| AdamW | 低 | 中 | 低 | 单机默认方案、通用训练 |
| Adafactor | 很低 | 中 | 低 | 超大模型、内存受限 |
| Shampoo | 高 | 高 | 高 | 中大矩阵、可接受额外代价 |
| Distributed Shampoo | 高 | 高 | 中到高 | 多机多卡、大规模预训练 |
| SOAP | 中 | 高 | 中 | 想保留 Shampoo 方向质量，同时简化调参 |
| 4-bit Shampoo | 更低 | 中到高 | 中 | 显存紧张但仍想用二阶预条件 |

适用边界可以直接记成三句话：

1. 当层的行列维度都较大，比如超过千级，且训练足够长，Shampoo 更有机会体现优势。
2. 当 GPU 间带宽足够、batch 较大、允许引入预条件更新 schedule 时，Distributed Shampoo 才能稳定发挥。
3. 当模型较小、训练预算紧、工程复杂度敏感时，AdamW 往往是更省资源的选择。

SOAP 的定位很重要。它不是原始 Shampoo 的简单实现技巧，而是把 Adam 风格的二阶矩更新放到 Shampoo 预条件器的特征空间里。根据 2024 年论文，SOAP 在大 batch 语言模型预训练里，相对 AdamW 在迭代数和 wall-clock 上都能显著下降，并且比直接低频更新 Shampoo 更稳。

4-bit Shampoo 则是另一条路线：不是改变算法结构，而是压缩状态存储。它试图把“Shampoo 太占内存”这个问题单独解决。若你的瓶颈主要是显存而不是通信，这比直接换回 AdamW 更值得考虑。

所以实际选择可以很粗暴：

- 默认通用训练：AdamW
- 显存极紧：Adafactor
- 想要二阶结构但能接受复杂实现：Shampoo
- 多卡大模型：Distributed Shampoo
- 想保留 Shampoo 优势并降低调参与频率敏感性：SOAP
- 二阶优化但内存受限：4-bit Shampoo

---

## 参考资料

- [Gupta, Koren, Singer. Shampoo: Preconditioned Stochastic Tensor Optimization. arXiv:1802.09568](https://arxiv.org/abs/1802.09568)  
  原始论文，给出 Shampoo 的核心公式、张量推广和理论基础。

- [Shi et al. A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale. arXiv:2309.06497](https://arxiv.org/abs/2309.06497)  
  工程实现论文，重点是 DTensor 分片、AllGather 搜索方向、多 GPU 训练开销控制。

- [Vyas et al. SOAP: Improving and Stabilizing Shampoo using Adam. arXiv:2409.11321](https://arxiv.org/abs/2409.11321)  
  讨论 Shampoo 与 Adam/Adafactor 的关系，并提出更稳定、频率更友好的替代方案。

- [Wang et al. 4-bit Shampoo for Memory-Efficient Network Training. arXiv:2405.18144](https://arxiv.org/abs/2405.18144)  
  关注状态量化，目标是在尽量保留 Shampoo 效果的同时降低内存成本。

- [Emergent Mind: Shampoo Optimizer](https://www.emergentmind.com/topics/shampoo-optimizer)  
  面向读者的综述入口，适合快速回顾公式与几何直觉。

- [Emergent Mind: Distributed Shampoo](https://www.emergentmind.com/topics/distributed-shampoo)  
  汇总分布式实现、实验结论和工程折中，适合与原论文交叉阅读。
