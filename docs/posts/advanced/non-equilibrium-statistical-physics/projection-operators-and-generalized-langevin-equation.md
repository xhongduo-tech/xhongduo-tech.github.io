---
title: 投影算符与广义朗之万方程
date: 2026-08-07
---

# 投影算符与广义朗之万方程

<div class="epigraph">
<p>整体大于部分之和。</p>
<footer>—— 亚里士多德（Aristotle，《形而上学》）</footer>
</div>

<div class="article-byline">
<p>第四级 · 非平衡统计物理 ｜ Zwanzig《Nonequilibrium Statistical Mechanics》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从投影算符开始

Green-Kubo 公式给出了输运系数的精确表达式，但它没有回答一个更根本的问题：**为什么宏观变量的演化可以用关联函数表达？** 玻尔兹曼方程靠分子混沌假设，RTA 靠单弛豫时间——它们都是「有效模型」。能否从刘维尔方程**不丢信息**地、系统地把「慢变量」的方程推导出来？

**Mori-Zwanzig 投影算符理论**（1960 年代）给出了答案：它把相空间「投影」到感兴趣的慢变量上，把快变量全部吸进「随机力」与「记忆函数」。结果是一个**广义朗之万方程**——朗之万方程（第2篇）的精确版，不再假设白噪声与瞬时摩擦，而是带记忆与色噪声。它是现代非平衡统计物理最强大的形式工具。

## 1 相关变量与投影

设我们关心的慢变量集合为 $A = (A_1, \dots, A_n)$——比如守恒密度（质量、动量、能量）或布朗粒子的速度。它们服从刘维尔方程 $\dot A = i\mathcal{L}A$。

定义**投影算符（projection operator）** $\mathcal{P}$：它把任意相函数投影到由 $A$ 张成的「相关子空间」上。对经典统计，最常用的 Mori 投影：

$$
\mathcal{P}X = \langle X, A\rangle \cdot \langle A, A\rangle^{-1}\cdot A
$$

其中内积 $\langle X, Y\rangle = \langle X^* Y\rangle_{eq}$ 是平衡平均。$\mathcal{P}$ 提取「$X$ 中与 $A$ 线性相关的部分」，其余部分是**正交补** $\mathcal{Q} = 1 - \mathcal{P}$。<span class="marginnote">投影算符是「把无穷维相空间压到有限个慢变量上」的数学装置。$\mathcal{P}$ 作用在任意函数上，只保留它「沿 $A$ 方向」的分量；$\mathcal{Q}$ 则取「与 $A$ 正交的剩余」。整个 Mori-Zwanzig 理论，就是在 $\mathcal{P}$ 与 $\mathcal{Q}$ 两个子空间之间做分解。</span>

## 2 广义朗之万方程

对慢变量 $A$，用标准的算符恒等式（Dyson 分解）把演化拆成三部分，得到**广义朗之万方程（generalized Langevin equation，GLE）**：

$$
\boxed{\dot A_i(t) = i\Omega_{ij}\,A_j(t) - \int_0^t K_{ij}(\tau)\,A_j(t-\tau)\,d\tau + f_i(t)}
$$

三项各司其职：

- **$i\Omega_{ij}$**：**频率矩阵**。它是投影到相关子空间内的确定性部分——「振荡/进动项」，对应可逆的弹道运动（如速度的零频运动、等离子体的集体振荡）。
- **$-\int_0^t K_{ij}(\tau)A_j(t-\tau)d\tau$**：**记忆项**。核 $K_{ij}(\tau)$ 叫**记忆函数（memory function）**，描述「系统现在受到的阻力依赖它过去的历史」——这正是「摩擦」的精确版。
- **$f_i(t)$**：**随机力**。它是初始条件正交于 $A$ 的分量，$f(t) = e^{i\mathcal{Q}\mathcal{L}t}\,\mathcal{Q}i\mathcal{L}A$，满足 $\mathcal{P}f = 0$（与相关变量正交）。<span class="marginnote">随机力不是「人为假设的噪声」，而是<strong>从刘维尔动力学精确推导出来的剩余项</strong>：它是「初始时刻与慢变量无关的那部分快自由度」演化到 $t$ 时刻的表现。这个定义式 $f(t) = e^{i\mathcal{Q}\mathcal{L}t}\mathcal{Q}i\mathcal{L}A$ 是 Mori-Zwanzig 的引擎——$\mathcal{Q}$ 把演化限制在正交子空间，所以 $f(t)$ 永远正交于相关变量。</span>

**辨析｜易错点：** 记忆项里 $K(\tau)A(t-\tau)$ 的积分是**历史卷积**——现在受到的「摩擦」不是由当前状态决定，而是由过去所有时刻的状态加权决定。这就是「非马尔可夫性」：GLE 是带记忆的方程。只有当记忆函数衰减得比所有可观测量都快（$K(\tau)\to 0$ 快）时，积分退化为 $A(t)$，才回到无记忆的朗之万方程。

## 3 公式解析：记忆函数

记忆函数 $K_{ij}(t)$ 的定义与随机力紧密相连：

$$
K_{ij}(t) = \langle f_i(t), f_j\rangle\,\langle A, A\rangle^{-1}
$$

- **$f_i(t)$**：随机力（上面定义的 $\mathcal{Q}$ 子空间演化）。记忆函数是随机力与**初始随机力**的内积——它度量「噪声对自己的记忆」。
- **$\langle A,A\rangle^{-1}$**：相关变量的协方差逆。它把随机力关联「归一化」成有正确量纲的核。
- **与涨落-耗散的关系**：$\langle f_i(t)f_j(0)\rangle = K_{ik}(t)\langle A_kA_j\rangle$。**记忆函数 = 随机力关联函数（归一化后）**。这正是涨落-耗散定理在 GLE 中的精确形态：记忆（耗散）与随机力（涨落）由同一个对象 $K(t)$ 给出。
- **物理解读**：记忆函数是「摩擦的时间结构」。白噪声极限 $K(t) = 2\gamma\delta(t)$ 给出摩擦 $-\gamma A(t)$；有限宽 $K(t)$ 给出带记忆的摩擦——系统「记得」自己过去的运动，阻力不再即时。

## 4 从 GLE 到 Green-Kubo

GLE 最深刻的应用：**从它可以直接推出 Green-Kubo 公式**，不需要任何额外假设。对 $A = \dot x$（速度）的 GLE 做拉普拉斯变换，解出速度关联函数的拉普拉斯变换：

$$
\tilde C(s) = \frac{\langle v^2\rangle}{s + \tilde K(s)}
$$

其中 $\tilde K(s)$ 是记忆函数的拉普拉斯变换。零频极限（$s\to 0$）：

$$
D = \int_0^\infty C(t)dt = \tilde C(0) = \frac{\langle v^2\rangle}{\tilde K(0)} = \frac{k_BT/m}{\int_0^\infty K(t)dt}
$$

结合记忆函数与随机力关联的关系，这正是 $D = \int_0^\infty\langle v(0)v(t)\rangle dt$——**Green-Kubo 公式被 Mori-Zwanzig 严格重建**。<span class="marginnote">更妙的是，从 GLE 还能导出<strong>分数阶</strong>或<strong>多指数</strong>的关联函数：若 $K(t) = \sum_n c_n e^{-t/\tau_n}$（多个弛豫模式），则 $C(t)$ 是多个指数/振荡的叠加——这解释了为什么真实系统的关联函数几乎从不干净地单指数衰减。GLE 给了我们「任意记忆结构」的普适语言。</span>

## 5 GLE 与朗之万方程的回归

GLE 是朗之万方程的**精确版**，两者关系可以总结成一张对照表：

| 特征 | 朗之万方程（第2篇） | 广义朗之万方程 |
| --- | --- | --- |
| 摩擦 | 常数 $\gamma$ | 记忆函数 $K(t)$ |
| 噪声 | 白噪声 $\langle\eta(t)\eta(t')\rangle\propto\delta(t-t')$ | 色噪声 $\langle f(t)f(0)\rangle \propto K(t)$ |
| 性质 | 马尔可夫（无记忆） | 非马尔可夫（带记忆） |
| 来源 | 唯象假设 | 刘维尔方程投影（精确） |

当记忆函数衰减极快（$K(t)\approx 2\gamma\delta(t)$）时，GLE 退化为朗之万方程。**朗之万方程不是近似于 GLE，而是 GLE 在「记忆瞬时」极限下的特例**——这也是为什么第2篇的朗之万方法如此成功：许多系统的记忆确实很短。

**辨析｜易错点：** GLE 的「随机力」与朗之万的「白噪声」有本质区别——GLE 的 $f(t)$ 由初始条件精确决定（$f(t) = e^{i\mathcal{Q}\mathcal{L}t}\mathcal{Q}i\mathcal{L}A$），不是概率性的输入，而是**确定性但看起来随机**的轨迹。它的「随机性」来自我们放弃了对 $\mathcal{Q}$ 子空间的精确知识。这一区分对理解「有效随机性」的统计本质至关重要。

## 6 例：从 GLE 到阻尼谐振子

把 GLE 用到最标准的模型——**阻尼谐振子**，可以看清每个构件的作用。设相关变量 $A = (x, p)$（位置与动量），GLE 分解出：

$$
\dot x = \frac{p}{m}, \qquad \dot p = -m\omega_0^2 x - \int_0^t K(t-\tau)\,p(\tau)\,d\tau + f(t)
$$

- **频率矩阵** $i\Omega$：给出无阻尼振荡频率 $\omega_0$——「弹道/可逆」部分。
- **记忆函数** $K(t)$：给出摩擦的时间结构。白噪声极限 $K(t) = 2\gamma\delta(t)$ 时，记忆项退化为 $\gamma p(t)$，回到经典阻尼谐振子。
- **随机力** $f(t)$：正交于 $(x,p)$ 的库自由度，满足 $\langle f(t)f(0)\rangle \propto K(t)$。

**关键验证——均分定理**：GLE 的稳态解必须满足 $\langle p^2/2m\rangle = k_BT/2$。把 GLE 的拉普拉斯变换解出 $\langle p^2\rangle$，要求它等于 $mk_BT$ 就自动给出记忆函数与随机力关联的涨落-耗散关系——**GLE 自洽地维持平衡统计**。

把记忆函数取不同形式，可以得到丰富的行为：

| 记忆函数 | 关联函数行为 | 物理系统 |
| --- | --- | --- |
| $2\gamma\delta(t)$ | 单指数衰减 | 简单液体、布朗运动 |
| $K_0 e^{-t/\tau_K}$ | 双指数/欠阻尼振荡 | 粘弹性介质、玻璃 |
| $\sim t^{-\alpha}$ | 幂律尾巴、次扩散 | 聚合物、胞内动力学 |

**辨析｜易错点：** GLE 的「频率矩阵」不是可观测频率——它由 $\mathcal{P}$ 子空间内的动力学决定，而真实振荡频率由记忆函数与频率矩阵共同决定。初学者把 $\Omega$ 误当作「谐振频率」会漏掉记忆函数的频移。从关联函数反推记忆函数（通过拉普拉斯变换 $\tilde K(s) = \langle v^2\rangle/\tilde C(s) - s$）是标准的诊断方法，也常被用来检验模型的记忆结构。

## 7 小结

- **投影算符** $\mathcal{P}$ 把相空间压到慢变量子空间，$\mathcal{Q}=1-\mathcal{P}$ 取正交补。
- **广义朗之万方程** $\dot A = i\Omega A - \int_0^t K(\tau)A(t-\tau)d\tau + f(t)$ 从刘维尔方程精确导出，含频率矩阵、记忆项与随机力三部分。
- **记忆函数** $K(t) = \langle f(t), f\rangle\langle A,A\rangle^{-1}$