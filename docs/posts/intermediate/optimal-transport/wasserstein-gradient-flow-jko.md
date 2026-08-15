---
title: Wasserstein 梯度流与 JKO 格式
date: 2026-08-07
---

# Wasserstein 梯度流与 JKO 格式

<div class="epigraph">
<p>把 Fokker–Planck 方程看作熵在 Wasserstein 距离下的梯度流，我们就得到了一个统一而优美的变分图景。</p>
<footer>—— 里夏尔 · 若尔当（Richard Jordan）、戴维 · 金德勒雷尔（David Kinderlehrer）与费利克斯 · 奥托（Felix Otto），《The Variational Formulation of the Fokker–Planck Equation》（1998，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优传输理论 ｜ Santambrogio《Optimal Transport for Applied Mathematicians》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从 Wasserstein 梯度流开始

普通微积分告诉我们：一个函数 $F$ 在欧氏空间里的最速下降，是沿着负梯度方向运动。**如果函数的定义域不是欧氏空间，而是"所有概率测度"组成的空间 $\mathcal{P}_2$，并且我们在这个空间上用 $W_2$ 来量距离，那么"最速下降"就变成了一条概率测度的演化轨迹**——这称为 **Wasserstein 梯度流（Wasserstein gradient flow）**。1998 年，若尔当、金德勒雷尔与奥托（JKO）发现：许多著名的偏微分方程——热方程、Fokker–Planck 方程、多孔介质方程——其实都是同一个东西：**某个泛函在 $W_2$ 下的梯度流**。这个观察把"PDE 分析"与"最优传输"焊在了一起，也给出了一种稳定且只依赖 Wasserstein 距离的数值格式，即 **JKO 格式**。<span class="marginnote">JKO 论文的标题本身就是纲领：把 Fokker–Planck 方程（一个线性抛物型 PDE）重新理解为熵的 $W_2$ 梯度流（一个变分问题）。这个"用变分法解 PDE"的视角，后来成了生成模型、采样算法与平均场博弈的共同语言。</span>

## 1 测度空间上的最速下降

先建立类比。在欧氏空间 $\mathbb{R}^d$ 中，梯度流 $x'(t) = -\nabla F(x(t))$ 的隐式欧拉格式是

$$
x_{n+1} = \arg\min_{x} \; \Big( \frac{1}{2\tau}\|x - x_n\|^2 + F(x) \Big)
$$

这等价于"靠近 $x_n$ 的同时尽量压低 $F$"——$\tau$ 是步长。把 $\| \cdot \|$ 换成 $W_2$、把 $x$ 换成概率测度 $\mu$，就得到 **JKO 格式**：<span class="marginnote">隐式欧拉之所以用 $\arg\min$ 而非显式 $x_{n+1} = x_n - \tau\nabla F(x_n)$，是为了稳定与无条件收敛。JKO 继承了这一优点：它天然无条件稳定，即使 $F$ 不光滑也不怕。</span>

这个类比要成立，关键的一步是把 $W_2$ 当"黎曼度量"用：$W_2$ 的平方对应切向量长度的平方，于是 $\frac{1}{2\tau}W_2^2(\mu_n,\mu)$ 扮演"位移的平方"的角色。这也是为什么 JKO 里用 $W_2^2$ 而非 $W_2$——平方项才配得上"动能"的物理直觉。对学过第一级《理论力学》的读者，把 $\mu$ 换成位置 $x$、$W_2$ 换成欧氏距离，整条推理就回到熟知的隐式欧拉格式：**每一步都是"动能最小 + 势能最小"的折中**。

## 2 公式解析：JKO 格式

**JKO 格式**：给定初始测度 $\mu_0$ 与泛函 $\mathcal{F}$，迭代

$$
\mu_{n+1} = \arg\min_{\mu \in \mathcal{P}_2(\mathbb{R}^d)} \; \frac{1}{2\tau} W_2^2(\mu_n, \mu) + \mathcal{F}(\mu)
$$

当 $\tau \to 0$ 时，$\mu_n$ 逼近一条测度值的轨迹 $\mu_t$，它满足 **Wasserstein 梯度流方程**

$$
\partial_t \mu = \nabla \cdot \Big( \mu \, \nabla \frac{\delta \mathcal{F}}{\delta \mu} \Big)
$$

其中 $\frac{\delta \mathcal{F}}{\delta \mu}$ 是 $\mathcal{F}$ 的**一阶变分导数**。拆成三步读：

- **第一步，读懂 JKO 的两项**：第一项 $\frac{1}{2\tau}W_2^2(\mu_n,\mu)$ 惩罚"离上一帧太远"（惯性项），第二项 $\mathcal{F}(\mu)$ 惩罚"势能太高"（驱动力项）。**每一步都在"贴近上一步"与"降低泛函"之间做最优折中**——这与隐式欧拉格式同构。
- **第二步，读懂梯度流方程**：$\partial_t \mu = \nabla \cdot (\mu \nabla \delta\mathcal{F}/\delta\mu)$ 是"连续性方程"：质量以速度场 $v = -\nabla(\delta\mathcal{F}/\delta\mu)$ 流动。**速度场是泛函导数的负梯度**——这就是"梯度流"三个字的含义。
- **第三步，读懂为什么是 JKO**：JKO 的意义在于，它**用最优化定义演化**——即使不知道 $\mathcal{F}$ 的导数、即使泛函不光滑，只要每一步能算 $W_2$ 与 $\arg\min$，就能得到演化。这就把"解 PDE"变成"反复做最优传输"。

**辨析｜易错点：** $\arg\min$ 里 $W_2$ 取的是"平方距离"还是"距离"差别巨大。JKO 用的是 $\frac{1}{2\tau}W_2^2$，因为只有平方项才与梯度流方程匹配（对应 $L^2$ 型度量）；用 $W_1$ 会得到完全不同的、涉及总变差的正则化动力学。<span class="marginnote">这也提醒我们：$W_2$ 在这里不只是"一种距离"，它是度量微积分里的"黎曼结构"。平方项给出了光滑的切空间几何，这是 Otto 微积分（见第 4 节）能成立的根本原因。</span>

## 3 熵的梯度流就是 Fokker–Planck

JKO 最著名的应用是把熵当泛函。取 **Boltzmann 熵**

$$
\mathcal{F}(\mu) = \int \rho \log \rho \, dx, \qquad \mu = \rho\, dx
$$

其一阶变分导数是 $\delta\mathcal{F}/\delta\mu = \log\rho + 1$，梯度为 $\nabla(\log\rho) = \nabla\rho/\rho$。代入梯度流方程：

$$
\partial_t \rho = \nabla \cdot \Big( \rho \cdot \frac{\nabla\rho}{\rho} \Big) = \Delta \rho
$$

**熵的 $W_2$ 梯度流就是热方程**。<span class="marginnote">这个结论优美得惊人：热传导（一个线性 PDE）竟然是熵极大化（一个纯变分原理）在 Wasserstein 几何下的最速下降。类似的，加上外部势 $V$ 后 $\mathcal{F} = \int \rho\log\rho + \int V\rho$，梯度流变成 Fokker–Planck 方程 $\partial_t\rho = \Delta\rho + \nabla\cdot(\rho \nabla V)$——正是 Langevin 动力学在分布层面的写法，采样算法与扩散模型都从这里来。</span>

## 4 Otto 微积分与更多例子

JKO 论文之后，奥托提出了一种"形式化黎曼几何"：把 $\mathcal{P}_2$ 看作无穷维黎曼流形，$W_2$ 是其测地距离，则泛函的梯度、Hessian 都可以形式化地定义——这套工具称为 **Otto 微积分（Otto calculus）**。它的价值在于：许多物理演化方程可以被**读出**为某个泛函的梯度流，从而获得单调量（能量耗散）、稳态（泛函极小点）等结构信息。

常见的梯度流对应表：<span class="marginnote">梯度流视角的威力在于"统一"：粒子系统、动力学与 PDE 三个层次（粒子 Langevin / 分布 Fokker–Planck / 场论）都能从同一个变分原理推出。这也是第一级《理论力学》里最小作用量思想在概率世界的复活。</span>

| 泛函 $\mathcal{F}(\mu)$ | 梯度流方程 | 现象 |
| --- | --- | --- |
| $\int \rho\log\rho$ | $\partial_t\rho = \Delta\rho$ | 热扩散 |
| $\int \rho\log\rho + \int V\rho$ | $\partial_t\rho = \Delta\rho + \nabla\cdot(\rho\nabla V)$ | Fokker–Planck |
| $\frac{1}{m-1}\int \rho^m$ | $\partial_t\rho = \Delta(\rho^m)$ | 多孔介质 |
| $\int V\rho + W*\rho$ 项 | 聚集–扩散方程 | 群体行为 |

Otto 微积分还提供**二阶结构**（Wasserstein Hessian），用来判定稳态的稳定性：凸泛函的梯度流保证收敛到唯一稳态，非凸泛函则可能停在不同局部极小。对采样算法而言，这解释了为什么"熵 + 凸势能"的 Langevin 采样保证收敛到目标分布，而一般非凸目标只能保证收敛到某个稳态——**单调量与凸性，是梯度流理论的灵魂**。

Otto 微积分也把最优传输与**信息几何**（第二级《信息几何》里的 Fisher 度量）摆成鲜明对照：Wasserstein 几何与 Fisher 几何是测度空间上两种最自然的黎曼结构，前者适合"质量搬运"、后者适合"信息推断"。理解这个对照，就理解了分布空间的现代全景。

以这一对照收束本章：最优传输理论在 21 世纪的回响，几乎都建立在这两个简单等式上——**热方程 = 熵的 $W_2$ 梯度流**、**Langevin 采样 = 热方程加漂移**。把它们记住，你就在分布的世界里有了自己的坐标系。

## 5 粒子视角：把分布梯度流变成常微分方程

梯度流方程 $\partial_t\mu = \nabla \cdot (\mu \nabla \delta\mathcal{F}/\delta\mu)$ 是"分布层面"的演化。工程上更常用的是**粒子视角**：用一个测度的随机样本 $\{x_i(t)\}_{i=1}^{N}$ 来逼近 $\mu_t$，每个粒子沿同一个速度场运动：

$$
\dot{x}_i(t) = -\,\nabla \frac{\delta\mathcal{F}}{\delta\mu}\big(x_i(t)\big)
$$

这来自"连续性方程 + 质点沿流线走"的事实：$\partial_t\mu + \nabla\cdot(\mu v) = 0$ 等价于质点以速度 $v$ 运动。**分布方程与粒子方程是同一枚硬币的两面**。<span class="marginnote">举例：熵 $+\,$ 势能泛函 $\mathcal{F} = \int \rho\log\rho + \int V\rho$ 的粒子方程是 $\dot{x}_i = -\nabla V(x_i)$——再加一个布朗运动扰动就得到 Langevin 方程，采样与扩散模型的粒子基础全在这里。这是本专题与"从极限到大模型"主线最直接的交点：大模型的扩散采样器，本质上就是 JKO/梯度流思想的离散化。</span>

把三种视角并排，能看清 JKO 在整条链路里的位置：

| 视角 | 数学对象 | 一句话 |
| --- | --- | --- |
| PDE | $\partial_t\mu = \nabla\cdot(\mu\nabla\delta\mathcal{F}/\delta\mu)$ | 分布的演化方程 |
| 变分（JKO） | $\mu_{n+1} = \arg\min \frac{1}{2\tau}W_2^2(\mu_n,\mu) + \mathcal{F}(\mu)$ | 用最优传输定义演化 |
| 粒子 | $\dot{x}_i = -\nabla\delta\mathcal{F}/\delta\mu(x_i)$ | 用有限质点模拟演化 |

三者互为因果：粒子方程是分布方程的"表示"，JKO 是分布方程的"求解器"，分布方程是前两者的"连续极限"。理解其中任何一个，都能推出另外两个——这是最优传输理论难得的结构之美。

**JKO 格式的更广身世**：若尔当–金德勒雷尔–奥托的贡献不是发明 $\arg\min$ 迭代（那是 De Giorgi 的 **minimizing movement** 框架，可上溯到隐式欧拉），而是证明了**在 $W_2$ 度量下、取熵泛函时，minimizing movement 恰好收敛到 Fokker–Planck 方程**。度量选 $W_2$ 而非别的，才是关键——这再次回到第 4 篇的结论：$W_2$ 是"几何"的。

## 6 小结

- **Wasserstein 梯度流**：在 $(\mathcal{P}_2, W_2)$ 上沿泛函 $\mathcal{F}$ 的最速下降，服从方程 $\partial_t\mu = \nabla\cdot(\mu\nabla \delta\mathcal{F}/\delta\mu)$。
- **JKO 格式**：$\mu_{n+1} = \arg\min \frac{1}{2\tau}W_2^2(\mu_n,\mu) + \mathcal{F}(\mu)$，用"反复做最优传输"逼近梯度流。
- **热方程与 Fokker–Planck 都是熵的梯度流**，这给出采样、扩散模型的变分基础。
- **Otto 微积分**把 $\mathcal{P}_2$ 看作无穷维黎曼流形，形式化定义梯度与 Hessian，与信息几何的 Fisher 度量形成鲜明对照。
- **三视角一体**：PDE / 变分（JKO）/ 粒子三种描述互为因果——粒子是表示，JKO 是求解器，PDE 是连续极限。
- **扩散模型接口**：采样与扩散模型的去噪过程，本质上是 JKO / 梯度流思想的离散化。

在下一节，我们将暂时放下"连续演化"，回到"大规模计算"：熵正则化加上 Sinkhorn 的交替缩放，将给出一个能在 GPU 上处理百万级直方图的实用算法。
