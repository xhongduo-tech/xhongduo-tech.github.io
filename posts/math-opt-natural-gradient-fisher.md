## 核心结论

自然梯度解决的不是“梯度算错了”，而是“距离量错了”。普通梯度默认参数空间用欧氏距离，也就是把参数改动的长度定义为 $\|\Delta\theta\|_2$。这个定义只关心参数数值变了多少，不关心模型输出分布变了多少。对概率模型来说，这个距离通常不公平，因为不同参数方向对预测的敏感度完全不同。

自然梯度的做法是把局部度量换成 Fisher 信息矩阵。Fisher 信息矩阵可以理解为“参数轻微变化时，模型输出分布有多敏感”。它的定义是：

$$
F(\theta)=\mathbb{E}\big[\nabla_\theta \log p(x|\theta)\nabla_\theta \log p(x|\theta)^\top\big]
$$

于是更新不再是直接沿着普通梯度 $g=\nabla L(\theta)$ 走，而是先用 $F(\theta)^{-1}$ 对梯度做一次几何校正：

$$
\Delta\theta=-\eta F(\theta)^{-1} g
$$

这一步的含义是：在“分布变化”这个更合理的尺度下，寻找最速下降方向。它的直接结果有两个。

第一，更新方向对参数化方式更稳定。参数重写、缩放、重参数化之后，普通梯度可能完全变形，但自然梯度在 KL 几何下表达的是同一类分布移动。

第二，不同维度的步长会自动匹配预测敏感度。玩具例子里，普通梯度是 `[-0.3, -0.6]`，如果 Fisher 近似为 $\mathrm{diag}(0.4,0.8)$，那么自然梯度是：

$$
F^{-1}g=
\begin{bmatrix}
1/0.4 & 0 \\
0 & 1/0.8
\end{bmatrix}
\begin{bmatrix}
-0.3\\
-0.6
\end{bmatrix}
=
\begin{bmatrix}
-0.75\\
-0.75
\end{bmatrix}
$$

可以看到，两维原本数值不同，但经过 Fisher 校正后，更新变成“在预测空间里同等重要”的改动，而不是机械地让第二维走得更大。

| 对比项 | 普通梯度 | 自然梯度 |
| --- | --- | --- |
| 默认长度定义 | 参数空间欧氏长度 $\|\Delta\theta\|_2$ | 分布空间局部 KL 长度 $\Delta\theta^\top F \Delta\theta$ |
| 关心什么 | 参数改了多少 | 预测分布改了多少 |
| 是否受参数化影响 | 强 | 弱，局部上具有参数化不变性 |
| 典型问题 | 某些方向过冲，某些方向走不动 | 更新更贴合模型真实敏感度 |

---

## 问题定义与边界

问题的核心不是“梯度下降无效”，而是“普通梯度把所有参数方向看成同样贵”。这在简单线性模型里影响还有限，但在深度网络、概率模型、策略网络里会很快变成优化瓶颈。

先看一个抽象过程：

参数空间中的一步 $\Delta\theta$  
$\downarrow$ 通过模型映射  
输出分布 $p(\cdot|\theta)$ 发生变化  
$\downarrow$ 用 KL 距离衡量变化大小  
得到这一步在“预测意义上”到底有多大

这里的 KL 距离，即 Kullback-Leibler divergence，可以白话理解为“两个概率分布差了多少”。如果 $\Delta\theta$ 很小，那么：

$$
\mathrm{KL}\big(p(\cdot|\theta)\|p(\cdot|\theta+\Delta\theta)\big)
\approx
\frac{1}{2}\Delta\theta^\top F(\theta)\Delta\theta
$$

这说明 Fisher 信息矩阵不是凭空定义出来的，它正是 KL 在局部二阶展开里的曲率矩阵。也就是说，Fisher 给出的不是一个拍脑袋的预条件器，而是概率分布几何本身。

为什么欧氏距离不够？因为“参数变化 0.01”在不同模型结构里意义完全不同。

一个典型对比是线性层和卷积层。假设两个参数张量都做同样大小的欧氏更新，表面看长度一样，但它们影响到的激活范围、共享方式、输出统计特征都可能完全不同。卷积核的一个参数会在空间维度重复使用，而线性层参数通常只影响单次乘加。于是相同步长在参数空间里一样长，在输出分布空间里却可能差很多。

所以，问题定义可以严格写成：

在最小化损失 $L(\theta)$ 时，我们希望找到一个更新方向，使得在“允许输出分布只发生有限变化”的条件下，损失下降最快。

这比“在固定欧氏半径内下降最快”更贴近概率模型本质。形式上就是：

$$
\min_{\Delta\theta} g^\top \Delta\theta
\quad
\text{s.t.}
\quad
\frac{1}{2}\Delta\theta^\top F(\theta)\Delta\theta \le \epsilon
$$

这里 $g=\nabla L(\theta)$。约束的意思是：不要让一步更新把模型分布改得太离谱。这个边界很重要，因为自然梯度只在局部 KL 二阶近似成立时可靠。步子太大，Fisher 的局部几何解释就会失真。

因此它的适用边界也要说清楚：

| 条件 | 结论 |
| --- | --- |
| 模型是概率模型或可写成条件分布 | Fisher 和 KL 几何最自然 |
| 只关注局部小步更新 | 二阶近似可靠 |
| 参数规模中小，或能接受近似 | 可实际求解 |
| 超大模型且无法做曲率近似 | 需要退化为近似方法或一阶方法 |

---

## 核心机制与推导

Fisher 信息矩阵有几种等价视角。

第一种视角，它是 score function 的二阶矩。score function 指 $\nabla_\theta \log p(x|\theta)$，白话讲就是“样本对参数的瞬时拉扯方向”。

$$
F(\theta)=\mathbb{E}[s(x,\theta)s(x,\theta)^\top], \quad s(x,\theta)=\nabla_\theta \log p(x|\theta)
$$

第二种视角，在正则条件下，它等于负对数似然 Hessian 的期望：

$$
F(\theta) = - \mathbb{E}\big[\nabla_\theta^2 \log p(x|\theta)\big]
$$

这说明 Fisher 描述的是局部曲率。曲率大，代表这一方向对分布更敏感；曲率小，代表这一方向更“平”。

现在推导自然梯度。设当前损失梯度为 $g$，我们想在固定 KL 半径内让损失下降最多，于是求解：

$$
\min_{\Delta\theta} g^\top \Delta\theta
\quad \text{s.t.} \quad
\frac{1}{2}\Delta\theta^\top F \Delta\theta \le \epsilon
$$

写拉格朗日函数：

$$
\mathcal{L}(\Delta\theta,\lambda)=g^\top\Delta\theta+\lambda\left(\frac{1}{2}\Delta\theta^\top F\Delta\theta-\epsilon\right)
$$

对 $\Delta\theta$ 求导并令其为零：

$$
g+\lambda F\Delta\theta = 0
$$

得到：

$$
\Delta\theta = -\frac{1}{\lambda}F^{-1}g
$$

把 $\eta=\frac{1}{\lambda}$ 看成学习率，就得到自然梯度更新：

$$
\Delta\theta=-\eta F^{-1}g
$$

这一步可以理解成“先把普通梯度投到 KL 曲率定义的正交坐标里，再决定走多远”。普通梯度只回答“哪个参数数值下降最快”，自然梯度回答“哪个分布变化方向下降最快”。

### 玩具例子：二维逻辑回归

假设单样本逻辑回归输入 $x=[1,2]$，当前普通梯度为：

$$
g=
\begin{bmatrix}
-0.3\\
-0.6
\end{bmatrix}
$$

经验 Fisher 粗略近似成对角矩阵：

$$
F\approx
\begin{bmatrix}
0.4 & 0\\
0 & 0.8
\end{bmatrix}
$$

那么自然梯度方向是：

$$
F^{-1}g=
\begin{bmatrix}
2.5 & 0\\
0 & 1.25
\end{bmatrix}
\begin{bmatrix}
-0.3\\
-0.6
\end{bmatrix}
=
\begin{bmatrix}
-0.75\\
-0.75
\end{bmatrix}
$$

这说明第二维虽然普通梯度更大，但它本身也更敏感，所以要被更多抑制。自然梯度不是平均化参数，而是在平均化“分布影响”。

### TRPO 为什么会出现自然梯度

TRPO 是 Trust Region Policy Optimization，可以白话理解为“每次只允许策略分布改一小圈，再在这圈里尽量提升回报”。它把强化学习目标近似成线性项，把策略变化限制成 KL 约束：

$$
\max_{\Delta\theta} g^\top \Delta\theta
\quad \text{s.t.} \quad
\mathrm{KL}(\pi_{\text{old}}\|\pi_{\text{new}})\le \delta
$$

局部展开后：

$$
\max_{\Delta\theta} g^\top \Delta\theta
\quad \text{s.t.} \quad
\frac{1}{2}\Delta\theta^\top F \Delta\theta \le \delta
$$

解出来就是：

$$
\Delta\theta \propto F^{-1}g
$$

而步长由约束半径 $\delta$ 决定，常写成：

$$
\Delta\theta =
\sqrt{\frac{2\delta}{g^\top F^{-1}g}}\,F^{-1}g
$$

这里的 $g^\top F^{-1}g$ 是在 Fisher 度量下的梯度长度。TRPO 的关键不是“用了复杂 RL 技巧”，而是它明确把优化问题写成“在 KL 信赖域里找最优方向”，自然策略梯度就是这个二次子问题的一阶解。

流程可以压缩成四步：

1. 算普通梯度 $g$
2. 用 Fisher 定义局部 KL 曲率
3. 解线性系统 $Fv=g$
4. 取 $v=F^{-1}g$ 作为更新方向，并按 KL 半径缩放

---

## 代码实现

工程里几乎不会显式构造完整 Fisher 矩阵。原因很直接：如果参数数是 $p$，那么 Fisher 是 $p\times p$，存储就是 $O(p^2)$，求逆是 $O(p^3)$。深度模型很快就不可承受。

实际做法通常是只实现 Fisher-vector product，也就是给一个向量 $v$，高效计算 $Fv$。有了这个算子，就能用共轭梯度法近似求解：

$$
F x = g
$$

求出的 $x$ 就近似是 $F^{-1}g$。

下面先给一个可运行的最小版 Python 例子。它不依赖自动微分，直接演示“普通梯度”和“自然梯度”在二维对角 Fisher 下的差异。

```python
import numpy as np

def natural_gradient_step(grad, fisher, lr):
    grad = np.asarray(grad, dtype=float)
    fisher = np.asarray(fisher, dtype=float)
    nat_grad = np.linalg.solve(fisher, grad)
    new_theta = -lr * nat_grad
    return nat_grad, new_theta

g = np.array([-0.3, -0.6])
F = np.array([[0.4, 0.0],
              [0.0, 0.8]])

nat_g, step = natural_gradient_step(g, F, lr=1.0)

assert np.allclose(nat_g, [-0.75, -0.75])
assert np.allclose(F @ nat_g, g)
assert step.shape == (2,)

print("natural gradient:", nat_g)
print("update step:", step)
```

这个例子能跑通，但它还不是工程实现。工程里更常见的是下面这种结构：

```python
def conjugate_gradient(fisher_vec, b, iterations=10, tol=1e-10):
    x = 0 * b
    r = b.copy()
    p = r.copy()
    rs_old = r @ r

    for _ in range(iterations):
        Ap = fisher_vec(p)
        alpha = rs_old / (p @ Ap + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if rs_new < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

g = compute_gradient(model, loss)

def fisher_vec(v):
    return empirical_fisher_vector_product(model, data, v)

natural_grad = conjugate_gradient(fisher_vec, g, iterations=10)
theta = theta - eta * natural_grad
```

这里 `empirical_fisher_vector_product` 的思想是：

$$
Fv = \mathbb{E}\left[s(s^\top v)\right]
$$

不需要显式构造 $F$，只需要对每个样本求 score $s=\nabla \log p$，先做内积 $s^\top v$，再乘回 $s$，最后取平均。

### 真实工程例子：Transformer 预训练

在 Transformer 预训练里，参数量极大，直接构造 Fisher 完全不可行，内存会先爆掉，随后求逆时间也不可接受。可行方案通常是 K-FAC。K-FAC 是 Kronecker-Factored Approximate Curvature，可以白话理解为“把每层的大曲率矩阵拆成两个更小的矩阵相乘近似”。

对全连接层，K-FAC 把层的 Fisher 块近似为：

$$
F_l \approx A_l \otimes G_l
$$

其中 $A_l$ 是输入激活的二阶统计，$G_l$ 是输出梯度的二阶统计，$\otimes$ 是 Kronecker 积。这样原本巨大的矩阵逆可以拆成两个小矩阵逆，复杂度大幅下降。

| 方法 | 是否显式构造完整 F | 主要代价 | 优点 | 适合场景 |
| --- | --- | --- | --- | --- |
| 共轭梯度 + Fv | 否 | 每轮要多次算 `Fv` | 实现贴近理论，适合 TRPO | 中等规模模型、策略优化 |
| K-FAC | 否，按层近似 | 维护激活/梯度统计 | 可扩展到大模型，收敛步数少 | 大型监督训练、预训练 |
| 直接求逆 | 是 | 内存和时间都极高 | 理论最直接 | 只适合玩具模型 |

---

## 工程权衡与常见坑

自然梯度的问题从来不是“公式错”，而是“代价太高”。最大的工程矛盾是：它比一阶方法更懂几何，但几何信息本身很贵。

第一类问题是 Fisher 欠定。样本少、参数多时，经验 Fisher 往往秩不足，矩阵不可逆或者病态。病态的意思是数值上极不稳定，微小误差会被放大。这时通常要加 Tikhonov 阻尼：

$$
F_{\text{damped}} = F + \lambda I
$$

这里 $\lambda I$ 的作用是把所有特征值往上抬，避免某些方向接近零而导致求逆爆炸。阻尼太小，系统不稳定；阻尼太大，又会退化回普通梯度。

第二类问题是估计噪声。Fisher 是期望矩阵，mini-batch 估计会抖动，尤其在强化学习和大模型训练前期更明显。常见办法是做滑动平均：

$$
\hat{F}_t = \beta \hat{F}_{t-1} + (1-\beta)F_t
$$

这样得到的不是瞬时 Fisher，而是平滑统计量，数值更稳定。

第三类问题是结构近似带来的偏差。K-FAC 默认每层块结构和 Kronecker 可分解成立，但真实网络常有残差连接、权重共享、LayerNorm、混合精度、张量并行等复杂因素。近似一旦和实际计算图错位，优化器会“以为”自己走在自然梯度方向上，实际上已经偏离。

### 真实工程例子：大模型训练中的 K-FAC 落地

假设在 Transformer 训练里直接尝试 Fisher 逆，你很快会遇到三个问题：

1. 参数数以亿计，完整矩阵根本放不下
2. 即使分块后能放下，求逆也慢得不可接受
3. 混合精度下小特征值方向容易数值崩溃

所以落地流程通常变成：

`estimate F` → `apply damping` → `solve` → `update`

更具体一点：

1. 对每个线性层收集输入激活统计 $A_l$
2. 对每个线性层收集输出梯度统计 $G_l$
3. 用滑动平均更新这两个统计
4. 给每个块加阻尼，例如 $A_l+\lambda I,\; G_l+\lambda I$
5. 用近似逆构造预条件方向
6. 再配合全局学习率、梯度裁剪、loss scaling 做稳定更新

实践里，K-FAC 可能让训练步数明显减少，文献中在现代架构上常见到 20% 到 40% 的 wall-clock 加速区间。但代价也很明确：实现复杂、需要框架钩子、额外通信和统计同步，且对层类型支持有要求。

| 近似方法 | 内存 | 偏差 | 最适合 |
| --- | --- | --- | --- |
| 全 Fisher | 极高 | 最小 | 研究级小模型 |
| 块对角 Fisher | 高 | 中 | 中型网络实验 |
| K-FAC | 中 | 中等，依赖层近似 | 大型前馈/Transformer |
| 对角近似 | 低 | 较大 | 快速试验、资源受限 |

常见坑还包括：

| 坑 | 表现 | 解决思路 |
| --- | --- | --- |
| 阻尼太小 | 训练震荡、共轭梯度发散 | 增大 $\lambda$，做线搜索 |
| 阻尼太大 | 退化成保守一阶法 | 分层调阻尼，不要全局写死 |
| Fisher 统计过旧 | 后期方向滞后 | 缩短更新周期或降低平滑系数 |
| BatchNorm/共享参数未对齐 | 近似块失真 | 显式处理共享结构或避开不支持层 |

---

## 替代方案与适用边界

如果无法承担 Fisher 近似的成本，最常见替代是 Adam 或 RMSProp。它们也会对不同维度做缩放，但缩放依据是梯度平方的历史均值，而不是 KL 几何。白话说，它们知道“哪个维度梯度大”，但不知道“哪个维度让输出分布变化更大”。

这就是为什么 Adam 不是自然梯度。Adam 的典型更新是：

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

其中 $\hat{v}_t$ 是梯度二阶矩估计，不等于 Fisher，也不直接对应 KL 局部曲率。它是一种实用的自适应步长方法，不是分布空间里的最速下降。

另一些替代包括 GGN 和不同形式的近似自然梯度。GGN 即 Generalized Gauss-Newton，可以白话理解为“对某些损失和网络结构，取一个更容易算的曲率近似”。它和 Fisher 在某些条件下关系很近，但不总是等价。

在图像分类里，经常看到这样的取舍：

| 方案 | KL 几何感知 | 计算复杂度 | 实现复杂度 | 适用条件 |
| --- | --- | --- | --- | --- |
| SGD/动量 | 无 | 低 | 低 | 基线训练、调参充足 |
| Adam/RMSProp | 弱，不是 KL 度量 | 低到中 | 低 | 大多数工业训练 |
| 共轭梯度自然梯度 | 强 | 中到高 | 中 | RL、策略优化、可算 Fv |
| K-FAC | 中到强，取决于近似 | 中 | 高 | 大模型训练、重视收敛效率 |
| GGN/其他二阶近似 | 中 | 中 | 中到高 | 特定损失、特定架构 |

简单总结它们的边界：

如果你要的是最低实现成本和稳健默认值，Adam 往往更合适。  
如果你要的是明确的 KL 约束和策略稳定性，TRPO/自然策略梯度更合适。  
如果你训练的是大模型，且优化效率足够重要到值得维护复杂优化器，K-FAC 才有现实意义。  
如果只是小模型教学或验证概念，直接构造 Fisher 反而最清楚。

---

## 参考资料

- Natural Gradient Descent | AI Under the Hood  
  用于理解 Fisher 信息矩阵、KL 二阶近似和自然梯度定义。

- Trust Region Policy Optimization · Depth First Learning  
  用于理解 TRPO 为什么会导出自然策略梯度，以及 KL 信赖域的直观含义。

- Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures  
  用于理解 K-FAC 如何把大规模曲率近似做成可落地工程方案。

- Amari, S. Natural Gradient Works Efficiently in Learning  
  自然梯度的经典来源，适合回到原始定义和几何视角。

- Martens, J. New Insights and Perspectives on the Natural Gradient Method  
  用于理解自然梯度和 Hessian、GGN、阻尼之间的关系。

- Schulman et al. Trust Region Policy Optimization  
  TRPO 原始论文，适合看约束优化形式和强化学习场景下的推导细节。
