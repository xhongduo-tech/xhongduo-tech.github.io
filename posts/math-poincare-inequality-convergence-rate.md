## 核心结论

Poincaré 不等式的核心结论是：在平衡分布 $\mu$ 下，一个函数整体偏离均值的程度，可以被它的局部变化能量控制。常见形式是

$$
\mathrm{Var}_\mu(f)\le C_P\,\mathcal E(f,f)
$$

其中 $\mathrm{Var}_\mu(f)$ 是方差，表示函数值围绕平均值波动的大小；$\mathcal E(f,f)$ 是 Dirichlet 能量，表示函数在空间中局部变化的强度；$C_P$ 是 Poincaré 常数。

谱隙是 Poincaré 常数倒数：

$$
\lambda_{\mathrm{gap}}=C_P^{-1}
$$

对马尔可夫链来说，谱隙把“每一步能削弱多少波动”转成“分布接近平衡分布的速度”。谱隙越大，链越快忘掉初始状态。

| 概念 | 控制对象 | 结论 |
|---|---|---|
| Poincaré 不等式 | 方差 | 局部能量越能控制全局波动 |
| 谱隙 | $L^2$ 收敛速率 | 谱隙越大，混合越快 |
| log-Sobolev 不等式 | 熵 | 比方差更强的信息层面控制 |

玩具例子：如果一条线上的相邻点函数值都差不多，那么这个函数不可能一边长期很大、另一边长期很小。马尔可夫链例子：如果链每一步都能有效抹平这种差异，那么从任意初始状态出发都会更快接近平衡分布。

---

## 问题定义与边界

先统一对象。函数 $f$ 是定义在状态空间上的实值函数，可以理解为“每个状态对应一个数”。平衡分布 $\mu$ 是马尔可夫链长期运行后稳定下来的分布。$\mu f$ 表示在 $\mu$ 下对 $f$ 取平均：

$$
\mu f=\int f\,d\mu
$$

方差定义为

$$
\mathrm{Var}_\mu(f)=\mu[(f-\mu f)^2]
$$

它衡量 $f$ 相对自身均值的整体波动。

能量在不同场景下写法不同。连续空间里，常见写法是

$$
\mathcal E(f,f)=\int |\nabla f|^2\,d\mu
$$

其中梯度 $\nabla f$ 表示函数在空间中的变化率。离散马尔可夫链里，常见写法是

$$
\mathcal E(f,f)=\langle f,(I-P)f\rangle_\mu
$$

其中 $P$ 是转移矩阵，表示一步转移概率；$I-P$ 衡量一步转移对函数波动的削弱。

| 场景 | 平衡分布 | 能量写法 | 常见用途 |
|---|---|---|---|
| 连续扩散 | $\mu$ | $\int |\nabla f|^2d\mu$ | 解析估计、偏微分方程 |
| 离散马尔可夫链 | $\mu$ | $\langle f,(I-P)f\rangle_\mu$ | 混合速度分析 |
| MCMC 采样 | 后验分布 $\pi$ | 由转移核决定 | 判断采样效率 |

边界需要讲清楚。Poincaré 不等式主要给 $L^2$ 意义下的收敛速度。$L^2$ 距离是均方意义下的距离，强调平方误差。总变差距离 TV 是两个分布最大概率差异的度量，更接近“采样分布看起来是否已经接近平衡”。只知道谱隙，通常不能直接断言 TV 距离一定下降很快，还需要初始分布、最小平稳概率、热启动条件或其他比较工具。

---

## 核心机制与推导

谱隙的变分形式是

$$
\lambda_{\mathrm{gap}}
=\inf_{f\not\equiv const}
\frac{\mathcal E(f,f)}{\mathrm{Var}_\mu(f)}
$$

这句话的意思是：在所有非常数函数里，寻找“能量 / 方差”的最小值。这个最小值越大，说明任何明显的全局波动都必须付出较大的局部变化能量，链就更容易把波动压下去。

机制链条可以写成：

$$
\text{局部变化被转移削弱}
\Rightarrow
\text{Dirichlet 能量足够大}
\Rightarrow
\text{方差下降}
\Rightarrow
\text{分布更快接近平衡}
$$

连续时间马尔可夫半群 $P_t$ 是“运行时间 $t$ 后对函数的平均作用”。如果 $P_t f$ 表示函数 $f$ 被链演化后的结果，那么在可逆情形下有

$$
\frac{d}{dt}\mathrm{Var}_\mu(P_t f)
=
-2\mathcal E(P_t f,P_t f)
$$

再由 Poincaré 不等式得到

$$
\mathcal E(P_t f,P_t f)
\ge
\lambda_{\mathrm{gap}}\mathrm{Var}_\mu(P_t f)
$$

于是

$$
\frac{d}{dt}\mathrm{Var}_\mu(P_t f)
\le
-2\lambda_{\mathrm{gap}}\mathrm{Var}_\mu(P_t f)
$$

解这个微分不等式，得到方差指数衰减：

$$
\mathrm{Var}_\mu(P_t f)
\le
e^{-2\lambda_{\mathrm{gap}}t}\mathrm{Var}_\mu(f)
$$

离散时间中，如果转移矩阵 $P$ 的非平凡最大特征值绝对值为 $\rho$，常见收敛因子类似 $\rho^t$。对可逆链，谱隙常写成 $1-\lambda_2$，其中 $\lambda_2$ 是第二大特征值。这里要注意：连续时间的 $e^{-\lambda t}$ 和离散时间的 $(1-\lambda)^t$ 是两套记号，不能直接混用。

log-Sobolev 不等式是更强的推广，形式为

$$
\mathrm{Ent}_\mu(f^2)\le 2C_{LS}\mathcal E(f,f)
$$

熵 $\mathrm{Ent}_\mu$ 衡量的是信息偏离，不只是平方波动。通常 log-Sobolev 能推出更强的收敛结论，但常数更难估计。

---

## 代码实现

下面用两状态对称链做最小例子：

$$
P=
\begin{pmatrix}
0.9 & 0.1\\
0.1 & 0.9
\end{pmatrix},
\qquad
\mu=(1/2,1/2)
$$

取 $f=(1,0)$。直接计算：

$$
\mu f=1/2,\qquad
\mathrm{Var}_\mu(f)=1/4
$$

能量为

$$
\mathcal E(f,f)=\langle f,(I-P)f\rangle_\mu=0.05
$$

所以

$$
\frac{\mathrm{Var}_\mu(f)}{\mathcal E(f,f)}=5
$$

对应谱隙为

$$
\lambda_{\mathrm{gap}}=0.2
$$

可运行代码如下：

```python
import numpy as np

def stationary_distribution(P):
    w, v = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(w - 1.0))
    mu = np.real(v[:, idx])
    mu = mu / mu.sum()
    return mu

def dirichlet_energy(P, mu, f):
    I = np.eye(P.shape[0])
    return float(np.dot(mu * f, (I - P) @ f))

def variance(mu, f):
    mean = float(np.dot(mu, f))
    var = float(np.dot(mu, (f - mean) ** 2))
    return mean, var

P = np.array([
    [0.9, 0.1],
    [0.1, 0.9],
])

f = np.array([1.0, 0.0])

mu = stationary_distribution(P)
mean, var = variance(mu, f)
energy = dirichlet_energy(P, mu, f)

eigenvalues = np.linalg.eigvals(P)
nontrivial = sorted(np.real(eigenvalues), reverse=True)[1]
gap = 1.0 - nontrivial

print("mu =", mu)
print("mu f =", mean)
print("Var =", var)
print("E =", energy)
print("Var / E =", var / energy)
print("gap =", gap)

assert np.allclose(mu, np.array([0.5, 0.5]))
assert abs(mean - 0.5) < 1e-12
assert abs(var - 0.25) < 1e-12
assert abs(energy - 0.05) < 1e-12
assert abs(var / energy - 5.0) < 1e-12
assert abs(gap - 0.2) < 1e-12
```

| 量 | 数值 |
|---|---:|
| $\mu f$ | $0.5$ |
| $\mathrm{Var}_\mu(f)$ | $0.25$ |
| $\mathcal E(f,f)$ | $0.05$ |
| $\mathrm{Var}/\mathcal E$ | $5$ |
| $\lambda_{\mathrm{gap}}$ | $0.2$ |

真实工程例子：在贝叶斯后验采样中，Metropolis-Hastings 或 Gibbs sampler 的目标是从复杂后验分布 $\pi(\theta\mid x)$ 中采样。如果谱隙很小，链可能长时间停留在某个局部区域，样本看起来数量很多，但有效样本数很低。工程上会用自相关时间、有效样本数、trace plot、多个链的诊断结果来间接判断混合质量。

---

## 工程权衡与常见坑

Poincaré 不等式适合回答“平方波动下降得多快”。它不直接回答“总变差距离是否已经足够小”，也不直接回答“样本是否已经覆盖所有重要模式”。工程上把它当作收敛分析工具，而不是万能诊断指标。

| 误区 | 正确理解 |
|---|---|
| Poincaré = log-Sobolev | 前者控制方差，后者控制熵 |
| 谱隙大就一定 TV 快 | 还需要把 $L^2$ 控制转成 TV 控制 |
| 离散与连续常数可直接对比 | 一个常用 $(1-\lambda)^t$，一个常用 $e^{-\lambda t}$ |
| 可逆链与非可逆链同样处理 | 非可逆链常要伪谱隙等工具 |
| 链一直在移动就说明混合好 | 移动频繁不等于有效样本多 |

工程诊断中常用的量如下：

| 诊断指标 | 含义 | 作用 |
|---|---|---|
| 谱隙 | $L^2$ 混合速度 | 理论比较算法快慢 |
| 自相关时间 | 样本之间依赖的持续时间 | 估计有效样本数 |
| burn-in | 初始偏差消退阶段 | 决定前多少步丢弃 |
| 有效样本数 | 独立样本等价数量 | 判断估计是否稳定 |

常见坑之一是只看接受率。MCMC 中接受率合适不代表混合好。随机游走提议步长很小时，接受率可能很高，但每步移动很小，自相关时间很长。另一个坑是只看单条链的轨迹。如果目标分布有多个模式，单条链可能在一个模式内看起来稳定，却长期没有跨到其他模式。

---

## 替代方案与适用边界

当目标只是控制方差衰减，Poincaré 不等式直接、清晰、常数含义明确。当目标变成熵衰减、浓缩不等式、强混合估计，log-Sobolev 不等式通常更合适。当链非可逆、存在强方向流或状态空间结构复杂时，单纯看普通谱隙可能不够。

| 方法 | 控制对象 | 优点 | 适用边界 |
|---|---|---|---|
| Poincaré | 方差 | 简单、直接 | 主要给 $L^2$ 速率 |
| log-Sobolev | 熵 | 更强，可给更细收敛 | 常数更难估计 |
| 伪谱隙 | 非可逆链 | 适合非对称过程 | 解释和计算更复杂 |
| 耦合方法 | 总变差 | 直观、可构造 | 依赖具体模型 |
| 路径法 | 瓶颈与流量 | 适合有限状态空间 | 路径设计影响结论 |

何时不用 Poincaré 作为主工具：

| 场景 | 更合适的工具 |
|---|---|
| 目标是 TV 距离 | 耦合、Doeblin 条件、比较定理 |
| 目标是熵衰减 | log-Sobolev 不等式 |
| 链明显非可逆 | 伪谱隙、加性反对称分解 |
| 状态空间有强瓶颈 | conductance、Cheeger 不等式 |
| 高维采样强相关 | 重参数化、阻塞更新、预条件 |

真实工程中，改进采样器经常不是直接“增大谱隙”这个词，而是通过重参数化、阻塞更新、预条件矩阵、Hamiltonian Monte Carlo、改进提议分布来减少随机游走行为。它们的共同目标是让链更快削弱偏差信号，提高有效混合速度。

---

## 参考资料

| 顺序 | 目的 | 资料 |
|---|---|---|
| 1 | 建立 Poincaré、谱隙、Dirichlet 能量框架 | Bakry, Gentil, Ledoux, *Analysis and Geometry of Markov Diffusion Operators*, Springer |
| 2 | 学习马尔可夫链混合时间与谱方法 | Levin, Peres, Wilmer, *Markov Chains and Mixing Times*, AMS |
| 3 | 理解 log-Sobolev 在随机游走模型中的改进形式 | Goel, *Modified logarithmic Sobolev inequalities for some models of random walk*, 2004 |
| 4 | 了解最快混合链与几何界 | Olesker-Taylor, Zanetti, *Geometric bounds on the fastest mixing Markov chain*, 2024 |

- Bakry, Gentil, Ledoux, *Analysis and Geometry of Markov Diffusion Operators*, Springer: <https://link.springer.com/book/10.1007/978-3-319-00227-9>
- Levin, Peres, Wilmer, *Markov Chains and Mixing Times*, AMS: <https://bookstore.ams.org/mbk-107>
- Goel, *Modified logarithmic Sobolev inequalities for some models of random walk*, Stochastic Processes and their Applications, 2004: <https://www.sciencedirect.com/science/article/pii/S0304414904000912>
- Olesker-Taylor, Zanetti, *Geometric bounds on the fastest mixing Markov chain*, PTRF, 2024: <https://link.springer.com/article/10.1007/s00440-023-01257-x>

不同文献的 Poincaré 常数、Dirichlet 型、连续时间生成元可能差一个 $2$ 或符号约定。比较结论前，先统一 $\mathcal E(f,f)$、$\lambda_{\mathrm{gap}}$ 和半群衰减公式。
