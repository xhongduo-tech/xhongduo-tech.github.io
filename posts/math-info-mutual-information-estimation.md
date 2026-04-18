## 核心结论

互信息 `I(X;Y)` 衡量两个随机变量 `X` 和 `Y` 的依赖强度。随机变量是“结果不固定、只能用概率描述的量”，例如传感器读数、用户点击、模型隐藏向量。互信息的新手版解释是：知道 `X` 之后，`Y` 的不确定性减少了多少。

互信息最常用的定义是：

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
$$

其中 `H(X)` 是熵，表示离散变量 `X` 的不确定性；`H(X,Y)` 是联合熵，表示同时观察 `X` 和 `Y` 的不确定性。对连续变量，熵通常写成微分熵：

$$
I(X;Y)=h(X)+h(Y)-h(X,Y)
$$

更本质的写法是：

$$
I(X;Y)=D_{KL}(p(x,y)\Vert p(x)p(y))
$$

`KL 散度` 是两个概率分布之间差异的度量。这里的意思是：如果联合分布 `p(x,y)` 和独立分布 `p(x)p(y)` 差得越多，`X` 和 `Y` 的依赖越强。

玩具例子：若 `X,Y` 是零均值二维高斯变量，相关系数 `ρ=0.6`，真实互信息为：

$$
I(X;Y)=-\frac{1}{2}\log(1-\rho^2)=-\frac{1}{2}\log(0.64)=0.2231\ \text{nats}
$$

`nats` 是以自然对数为底时的信息单位。换成 `bits` 需要除以 `log(2)`，约为 `0.322 bits`。这个例子说明：互信息不是只看线性回归效果，而是在量化整体依赖结构。

| 情况 | 联合分布与独立分布 | 互信息 | 含义 |
|---|---:|---:|---|
| `X` 与 `Y` 独立 | `p(x,y)=p(x)p(y)` | `0` | 知道 `X` 不减少 `Y` 的不确定性 |
| `X` 与 `Y` 有依赖 | `p(x,y)≠p(x)p(y)` | `>0` | 知道 `X` 后，`Y` 更容易判断 |
| 依赖更强 | 差异更大 | 更大 | 变量之间共享更多信息 |

实际估计里，方法本质上分三类：基于近邻的非参数估计、基于神经网络的下界估计、基于离散化或分类代理的估计。它们不是“越复杂越好”，而是适配不同数据形态。

---

## 问题定义与边界

本文讨论的是从样本估计互信息。样本是从真实分布中观测到的一组数据点，例如：

$$
\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}
$$

问题是：真实的 `p(x,y)`、`p(x)`、`p(y)` 不可见，只能用这 `N` 个样本近似。对离散变量，可以数频次估计概率；对连续变量，不能直接数“某个值出现几次”，因为两个浮点数完全相同的概率通常接近 `0`。

这就是连续变量互信息估计的核心挑战：公式清楚，但分布不可见；分布要估，估计过程又会引入偏差和方差。偏差是系统性误差，方差是不同样本下结果波动的程度。

真实工程例子：工业设备有三路连续传感器数据：振动、温度、转速。工程目标不是先精确恢复三者的概率密度，而是判断：

$$
I(\text{振动};\text{转速})
$$

和

$$
I(\text{温度};\text{转速})
$$

哪个更大，是否足以支持特征筛选、去冗余或滞后对齐。若振动与转速互信息明显高，说明转速变化能解释一部分振动变化；若异常发生后互信息下降，说明原本稳定的耦合关系被破坏。

| 场景 | 可用方法 | 主要风险 | 是否推荐分箱 |
|---|---|---|---|
| 离散变量 | 频次统计、插件估计 | 样本少时低频类别不稳定 | 可以 |
| 低维连续变量 | KSG、k-NN、核密度、粗分箱 | 尺度、样本量、边界效应 | 只适合粗分析 |
| 高维连续变量 | MINE、JS 估计器、降维后 KSG | 高维稀疏、训练不稳定、下界偏差 | 不推荐 |
| 深度学习表示 | MINE、Jensen-Shannon 代理目标 | 不能直接当真实 MI 读 | 不推荐 |

本文重点放在连续变量和高维样本场景。最简单的离散计数型问题不是重点，因为它的难点主要是样本量和类别稀疏，而不是连续密度估计。

---

## 核心机制与推导

最朴素的方法是分箱。分箱是把连续值切成若干区间，再把每个区间当成离散类别。例如把温度切成 `[0,10)`、`[10,20)`、`[20,30)`。这样就能数频次，然后代入离散互信息公式。

问题在于，分箱结果强烈依赖 bin 宽。bin 太宽，细节被抹掉，互信息偏小；bin 太窄，很多格子为空，估计方差变大。二维或高维时，格子数还会指数增长，这就是维度灾难：维度增加后，样本在空间里变得极度稀疏。

KSG 是常用的 k-NN 互信息估计器。k-NN 是“看最近的 `k` 个邻居”的方法。KSG 的核心不是先估出完整密度，而是利用样本之间的距离近似局部概率质量。

KSG 的直觉是：如果 `X` 和 `Y` 强依赖，那么在联合空间 `(x,y)` 里靠得很近的点，在边缘空间 `x` 和 `y` 里也会呈现特定计数结构。常见形式为：

$$
\hat{I}_{KSG}
=
\psi(k)+\psi(N)
-
\frac{1}{N}\sum_i
[
\psi(n_x(i)+1)+\psi(n_y(i)+1)
]
$$

其中 `ψ` 是 digamma 函数，可以理解为和 `log` 相关的平滑计数函数；`N` 是样本数；`k` 是近邻数；`n_x(i)` 和 `n_y(i)` 是第 `i` 个样本在边缘空间里的邻居计数。

单个样本点的 KSG 走查如下。给定样本 `i=(x_i,y_i)`：

| 步骤 | 操作 | 得到什么 |
|---|---|---|
| 1 | 在联合空间 `(X,Y)` 找第 `k` 个近邻 | 半径 `ε_i` |
| 2 | 回到 `X` 空间统计 `|x_j-x_i|<ε_i` 的点数 | `n_x(i)` |
| 3 | 回到 `Y` 空间统计 `|y_j-y_i|<ε_i` 的点数 | `n_y(i)` |
| 4 | 把所有样本的计数代入公式 | `Ĩ_KSG` |

这个过程数的不是“某个精确值出现几次”，而是“局部邻域里有多少样本”。它绕开了直接估计完整密度的问题。

MINE 是 Mutual Information Neural Estimation，意思是用神经网络估计互信息。神经网络 critic `T_θ(x,y)` 是一个打分函数，输入一对样本，输出一个实数分数。MINE 基于 Donsker-Varadhan 表示，把互信息写成 KL 散度的可优化下界：

$$
\hat{I}_{DV}
=
\mathbb{E}_{p(x,y)}[T_\theta]
-
\log \mathbb{E}_{p(x)p(y)}[\exp(T_\theta)]
$$

第一项来自真配对样本 `(x,y)`，第二项来自独立配对样本。独立配对通常通过打乱 batch 中的 `y` 得到，例如 `(x_i,y_{\pi(i)})`。训练目标是让真配对得分高，打乱配对在指数平均意义下受控。

Jensen-Shannon 估计器常用于表示学习。它把“真配对”和“打乱配对”当成二分类问题：真配对为正样本，打乱配对为负样本。常见目标为：

$$
\hat{I}_{JS}
\approx
\mathbb{E}_{joint}[-sp(-T_\theta)]
-
\mathbb{E}_{prod}[sp(T_\theta)]
+
const
$$

其中 `sp` 是 `softplus`：

$$
sp(z)=\log(1+\exp(z))
$$

JS 目标通常比 DV 更稳定，但它更像互信息下界或代理目标，不适合直接当作可校准的绝对互信息值。

| 方法 | 输入 | 核心思想 | 输出性质 | 优点 | 缺点 |
|---|---|---|---|---|---|
| 分箱 | 连续值切成类别 | 用离散频次近似概率 | 粗略 MI | 简单直观 | 对 bin 宽敏感 |
| KSG/k-NN | 样本点与距离 | 联合空间找近邻，边缘空间计数 | 数值估计 | 中低维连续变量好用 | 尺度敏感，高维变差 |
| MINE/DV | 真配对与打乱配对 | 神经网络拟合 KL 下界 | 可优化下界 | 可接入深度学习 | 训练震荡，偏差明显 |
| JS | 正负样本对 | 二分类代理目标 | 稳定下界或相对指标 | 训练更稳 | 易饱和，不宜读绝对值 |

---

## 代码实现

工程实现应从流程出发：先标准化，再选择估计器，再解释单位。标准化是把特征减均值、除标准差，避免距离被量纲支配。例如温度范围是 `0~100`，振动范围是 `0~1`，不标准化会让近邻主要由温度决定。

下面代码给出一个可运行的二维高斯玩具例子，用 KSG 估计互信息，并用解析公式做断言。它做的不是直接求密度，而是用样本间距离间接估计依赖强度。

```python
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

def standardize(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean(axis=0, keepdims=True)) / a.std(axis=0, keepdims=True)

def ksg_mi(x, y, k=5):
    x = standardize(np.asarray(x).reshape(len(x), -1))
    y = standardize(np.asarray(y).reshape(len(y), -1))
    n = len(x)

    xy = np.hstack([x, y])

    # Chebyshev 距离对应 KSG 常见实现里的 max-norm。
    nn_joint = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1)
    nn_joint.fit(xy)
    distances, _ = nn_joint.kneighbors(xy)
    eps = np.nextafter(distances[:, k], 0.0)

    nx = np.empty(n, dtype=int)
    ny = np.empty(n, dtype=int)

    nn_x = NearestNeighbors(metric="chebyshev")
    nn_y = NearestNeighbors(metric="chebyshev")
    nn_x.fit(x)
    nn_y.fit(y)

    for i in range(n):
        nx[i] = len(nn_x.radius_neighbors([x[i]], radius=eps[i], return_distance=False)[0]) - 1
        ny[i] = len(nn_y.radius_neighbors([y[i]], radius=eps[i], return_distance=False)[0]) - 1

    return digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))

rng = np.random.default_rng(7)
rho = 0.6
cov = np.array([[1.0, rho], [rho, 1.0]])
samples = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=3000)
x = samples[:, 0]
y = samples[:, 1]

estimated = ksg_mi(x, y, k=5)
truth = -0.5 * np.log(1 - rho ** 2)

assert truth > 0.22 and truth < 0.23
assert abs(estimated - truth) < 0.08, (estimated, truth)

print(round(estimated, 4), round(truth, 4))
```

MINE 和 JS 的训练流程可以用伪代码理解。关键是构造两类样本：联合样本来自真实配对 `(x_i,y_i)`，边缘乘积分布样本来自打乱配对 `(x_i,y_{\pi(i)})`。

```python
# Python 风格伪代码：展示训练结构，不依赖具体深度学习框架。

for x_batch, y_batch in dataloader:
    y_shuffle = shuffle(y_batch)

    joint_score = critic(x_batch, y_batch)
    prod_score = critic(x_batch, y_shuffle)

    mine_loss = -(
        mean(joint_score)
        - log(mean(exp(prod_score)))
    )

    js_loss = -(
        mean(-softplus(-joint_score))
        - mean(softplus(prod_score))
    )

    # 使用其中一个 loss 更新 critic 参数。
    optimizer.zero_grad()
    loss = js_loss
    loss.backward()
    optimizer.step()
```

| 估计器 | 输入张量形状 | 关键超参 | 输出单位 | 是否可微 |
|---|---|---|---|---|
| KSG | `X: [N, dx]`, `Y: [N, dy]` | `k`、距离度量 | 通常为 nats | 否 |
| MINE/DV | batch 真配对与打乱配对 | batch size、critic、学习率 | nats 下界 | 是 |
| JS | batch 真配对与打乱配对 | 负样本构造、critic、学习率 | 代理下界 | 是 |
| 分箱 | 离散类别或 bin 编号 | bin 数、平滑项 | nats 或 bits | 否 |

---

## 工程权衡与常见坑

互信息估计最重要的问题不是“公式会不会背”，而是“这个数在当前数据上是否可信”。估计误差通常不是单纯随机噪声，而是由方法选择、样本量、尺度、维度和超参共同决定。

KSG 的 `k` 控制偏差-方差权衡。`k` 太小，只看很局部的邻域，方差大；`k` 太大，邻域变宽，局部结构被抹平，偏差大。例如同一批传感器数据，可能出现：

| `k` | 估计结果 `I(振动;转速)` | 解释 |
|---:|---:|---|
| 3 | `0.41 nats` | 局部敏感，波动可能较大 |
| 10 | `0.35 nats` | 通常更稳，可作为主参考 |
| 30 | `0.26 nats` | 平滑更强，可能低估局部依赖 |

这不表示三个结果互相矛盾，而是说明估计器在不同平滑强度下看到了不同尺度的依赖结构。工程上应固定预处理和 `k`，再比较不同变量对，而不是混用不同配置。

MINE 的常见问题是训练曲线乱跳。比如 batch size 只有 `32`，critic 又很深，`exp(T_θ)` 的均值会被少数极端分数支配，导致 DV 下界一会儿很大、一会儿崩掉。实际训练中常用更大 batch、梯度裁剪、滑动平均估计分母，并监控正负样本分数分布。

| 坑点 | 现象 | 原因 | 推荐动作 |
|---|---|---|---|
| 分箱法 | bin 数一变，MI 大幅变化 | 连续变量被粗暴离散化 | 只做粗分析；报告 bin 设置 |
| KSG 尺度敏感 | 某个量纲大的特征主导距离 | 距离计算受尺度影响 | 先标准化；检查异常值 |
| KSG 高维失效 | 结果接近 0 或波动很大 | 高维空间样本稀疏 | 降维、增大样本、分组估计 |
| MINE 分母估计偏差 | 曲线乱跳或虚高 | `exp(T)` 方差大 | 增大 batch、滑动平均、限制 critic |
| JS 饱和 | loss 稳定但区分不出差异 | 分类任务过易或过难 | 改负样本策略；只做相对比较 |
| 小样本比较 | 变量排名不稳定 | 采样误差大 | bootstrap 置信区间 |

一个实用判断是：如果换随机种子、换 `k`、换子样本后排序完全变了，当前互信息估计不适合支撑强结论。此时应先增加样本、降低维度，或把目标从“估绝对值”改成“同一配置下比较相对变化”。

---

## 替代方案与适用边界

不同方法的选择应围绕数据类型、样本量、维度和目标。估计绝对 MI 与比较相对变化是两类任务。KSG 更偏向数值估计；MINE 和 JS 更常用于表示学习、特征选择或训练过程中的相对监控。

| 数据类型 | 样本量 | 维度 | 是否需要可微 | 是否需要绝对值 | 推荐方法 |
|---|---:|---:|---|---|---|
| 两个离散变量 | 中等 | 低 | 否 | 是 | 频次统计 |
| 连续传感器 | `200~5000` | 低 | 否 | 尽量需要 | KSG |
| 连续高维向量 | 大 | 高 | 否 | 不强求 | 降维后 KSG 或相对比较 |
| 神经网络表示 | 大 | 高 | 是 | 不强求 | MINE 或 JS |
| 图像、文本表示 | 大 | 高 | 是 | 通常不需要 | JS、InfoNCE、对比学习目标 |
| 结论 | 先选最简单可用的方法 | 再升级复杂估计器 | 保持配置一致 | 区分绝对值和相对值 | 避免只看单次数字 |

选择题式场景如下。

若只有 `200` 条样本，并且是低维连续传感器数据，优先使用 `KSG`，同时做标准化和 bootstrap。bootstrap 是反复重采样估计结果波动的方法，用来判断变量排序是否稳定。

若要在神经网络中端到端最大化表示 `Z` 和输入 `X` 的相关性，优先使用 `MINE` 或 `JS`。这里的目标不是得到一个严格可校准的互信息数值，而是提供一个可反向传播的训练信号。

若只是粗略统计两个离散变量关系，例如“用户等级”和“是否点击”，频次统计就足够。若把连续变量强行分箱，结果只能作为探索性参考，不能单独作为严肃结论。

还有一些替代目标也常见。`InfoNCE` 是对比学习中常用的互信息下界，适合大量负样本和表示学习；皮尔逊相关系数适合线性关系快速筛查；距离相关和 HSIC 可用于非线性依赖检验。它们不等价于互信息，但在工程上经常更稳定。

结论是：先明确目标，再选估计器。要一个中低维连续变量的数值估计，先试 KSG；要深度模型里的训练目标，考虑 MINE、JS 或 InfoNCE；要快速探索离散关系，用频次统计即可。

---

## 参考资料

- Kraskov, Stögbauer, Grassberger, *Estimating mutual information*, Phys. Rev. E, 2004. 用于理解 KSG 近邻计数机制。
- Belghazi et al., *MINE: Mutual Information Neural Estimation*, ICML 2018. 用于理解 DV 下界和神经网络互信息估计。
- Hjelm et al., *Learning deep representations by mutual information estimation and maximization*, ICLR 2019. 用于理解 Deep InfoMax 和 JSD 目标在表示学习中的用法。
- DuaneNielsen/DeepInfomaxPytorch. 用于参考 Deep InfoMax 中 JSD 风格目标如何落地到代码。
- Cover and Thomas, *Elements of Information Theory*. 用于查阅互信息、KL 散度、熵和二元高斯互信息公式的基础推导。
