---
title: 二次量子化（产生湮灭算符、Fock 空间、场算符）
date: 2026-08-07
---

# 二次量子化（产生湮灭算符、Fock 空间、场算符）

<div class="epigraph">
<p>物理定律必须用「粒子不可分辨」的语言来写。二次量子化让这一要求变成自动的。</p>
<footer>—— 理乍得 · 费曼（Richard P. Feynman）论量子力学的全同粒子（转述）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics*, Ch. 1 ｜ 2026-08-07</p>
</div>

## 为什么从二次量子化开始

多体物理的麻烦不在于「方程难解」，而在于**描述方式本身**：$N$ 个全同粒子的波函数 $\Psi(\mathbf{r}_1,\dots,\mathbf{r}_N)$ 必须满足置换对称性或反对称性，这个约束随着 $N$ 的增长让计算彻底失去可操作性。<span class="marginnote"><strong>全同粒子</strong>：电子之间、氢原子之间没有个体标记。交换两个全同粒子不产生新的物理态——波函数只能整体乘以相位，玻色子为 $+1$，费米子为 $-1$。</span>我们在本专题第 1 篇已经用占据数表象处理过一遍这条主线；第 5 篇作为系统复习，把**产生湮灭算符、Fock 空间、场算符**这三个支柱重新搭一遍，为后面全部章节（Green 函数、Feynman 图、Bogoliubov 变换、BCS）提供一份可以随时查证的「语言手册」。

## 1 产生与湮灭算符：多体语言的字母表

**产生算符（creation operator）** $a_\lambda^\dagger$ 在单粒子态 $\varphi_\lambda$ 上添加一个粒子，**湮灭算符（annihilation operator）** $a_\lambda$ 从该态上取走一个粒子。$\lambda$ 是一组完备量子数，例如动量 $\mathbf{k}$ 与自旋 $\sigma$。数算符（number operator）测出占据数：

$$\hat{n}_\lambda = a_\lambda^\dagger a_\lambda$$

对玻色子，$a_\lambda^\dagger$ 连续作用可以堆叠任意多个粒子；对费米子，泡利原理要求一个态上最多一个粒子，这一约束由代数结构自动给出。<span class="marginnote">为什么叫「二次量子化」？位置与动量在第一层量子化里已是算符，现在连「粒子数」都被提升为算符——这是第二次量子化。名字是历史的，内容是实用主义：它把对称化从波函数手工操作变成算符代数的必然结果。</span>

## 2 Fock 空间：粒子数可变的剧场

单粒子 Hilbert 空间容纳一个粒子的所有态；**Fock 空间（Fock space）**则把粒子数为 $0,1,2,\dots$ 的全体多体态并排放进来：

$$\mathcal{F} = \bigoplus_{N=0}^{\infty} \mathcal{H}^{(N)}$$

其中 $\mathcal{H}^{(N)}$ 是 $N$ 粒子对称（玻色）或反对称（费米）子空间，$\mathcal{H}^{(0)}$ 由真空态 $|0\rangle$ 张成。Fock 空间的优越性在于：它允许粒子数改变，因此是描述产生、湮灭、散射、凝聚的统一舞台——这正是多体理论真正需要「场」的原因。

从真空出发，任何占据数态都可写成产生算符的积：

$$|n_1,n_2,\dots\rangle = \prod_\lambda \frac{(a_\lambda^\dagger)^{n_\lambda}}{\sqrt{n_\lambda!}}\,|0\rangle$$

玻色子的 $\sqrt{n_\lambda!}$ 来自 $N!$ 个等价排序的归一化；费米子的 $n_\lambda \in \{0,1\}$ 使阶乘项退化为 1。

## 3 对易与反对易：统计性质内建于代数

两套代数关系划出了玻色子与费米子的分水岭：

**玻色子**：$[a_\lambda, a_\mu^\dagger] = \delta_{\lambda\mu}$，其余对易子为零。
**费米子**：$\{c_\lambda, c_\mu^\dagger\} = \delta_{\lambda\mu}$，其余反对易子为零。

**重点：费米子的反对易关系自动实现泡利不相容**。因为 $c_\lambda^\dagger c_\lambda^\dagger = -c_\lambda^\dagger c_\lambda^\dagger$，唯一可能的结果是 $c_\lambda^\dagger c_\lambda^\dagger = 0$——往已占据态再放一个费米子恒得零。对称性与反对称性也内建其中：交换两个产生算符，玻色子不变号，费米子变号。<span class="marginnote"><strong>辨析</strong>：$a_\lambda a_\lambda^\dagger = a_\lambda^\dagger a_\lambda + 1$（玻色）与 $\{c_\lambda,c_\lambda^\dagger\}=1$ 只差一个符号，却导出截然不同的宏观统计：玻色凝聚与泡利不相容。这是整门多体理论里「一个符号改变世界」的最早例子。</span>

## 4 场算符：从离散指标到连续空间

占据数表象用离散指标 $\lambda$ 标记态；把指标「换成」空间点 $\mathbf{r}$，就得到**场算符（field operator）**：

$$\hat{\psi}(\mathbf{r}) = \sum_\lambda \varphi_\lambda(\mathbf{r})\,a_\lambda, \qquad \hat{\psi}^\dagger(\mathbf{r}) = \sum_\lambda \varphi_\lambda^*(\mathbf{r})\,a_\lambda^\dagger$$

$\hat{\psi}^\dagger(\mathbf{r})$ 在 $\mathbf{r}$ 处产生一个粒子，$\hat{\psi}(\mathbf{r})$ 在 $\mathbf{r}$ 处湮灭一个粒子。注意它们与一次量子化里的波函数同名而异质：$\psi(\mathbf{r})$ 是**数**，$\hat{\psi}(\mathbf{r})$ 是**算符**。密度算符与总粒子数算符分别是：

$$\hat{n}(\mathbf{r}) = \hat{\psi}^\dagger(\mathbf{r})\hat{\psi}(\mathbf{r}), \qquad \hat{N} = \int\! d\mathbf{r}\,\hat{\psi}^\dagger(\mathbf{r})\hat{\psi}(\mathbf{r})$$

## 5 公式解析：哈密顿量翻译成算符语言

二次量子化的实战价值在于把相互作用哈密顿量写成场算符的积。以含库仑相互作用的电子气为例：

$$
\hat{H} = \sum_{\mathbf{k}\sigma} \varepsilon_{\mathbf{k}}\,c_{\mathbf{k}\sigma}^\dagger c_{\mathbf{k}\sigma} + \frac{1}{2V}\sum_{\mathbf{k}\mathbf{k}'\mathbf{q}\atop\sigma\sigma'} v_{\mathbf{q}}\,c_{\mathbf{k}+\mathbf{q},\sigma}^\dagger c_{\mathbf{k}'-\mathbf{q},\sigma'}^\dagger c_{\mathbf{k}',\sigma'} c_{\mathbf{k},\sigma}
$$

逐项拆解：

- **第一项（动能）**：$\varepsilon_{\mathbf{k}}=\hbar^2 k^2/2m$，$c^\dagger c$ 是数算符，累加每个被占据态的动能。无相互作用时的基态是「填满费米球」，费米动量 $k_F$ 与密度 $n$ 的关系是 $n = k_F^3/3\pi^2$。
- **第二项（两体散射）**：算符从右往左读——先湮灭两个电子，再产生两个电子，即交换动量 $\mathbf{q}$ 的散射事件；$v_{\mathbf{q}} = 4\pi e^2/q^2$ 是库仑势的 Fourier 分量（三维）。
- **系数 $1/2V$**：$1/2$ 避免把同一个散射过程数两次，$1/V$ 来自动量求和与实空间体积的换算。
- **顺序**：产生算符一律排在湮灭算符左边（正规序），保证真空期望为零，为 Wick 定理铺路。

**重点：二次量子化不引入新物理，只是换了一种更适配全同粒子的数学**。它把「对称化约束」从波函数层搬到算符代数层，代价是一次性学会产生/湮灭算符的规则，收益是 $10^{23}$ 粒子也能形式化处理。

## 6 二次量子化的三大经典应用

掌握算符代数后，立刻能欣赏三个标志性结果，它们全部建立在产生/湮灭算符之上：

**声子场**：晶格振动位移场 $u(\mathbf{r}) = \sum_{\mathbf{k}} \sqrt{\hbar/2NM\omega_{\mathbf{k}}}(a_{\mathbf{k}}+a_{-\mathbf{k}}^\dagger)e^{i\mathbf{k}\cdot\mathbf{r}}$，其中 $a_{\mathbf{k}}^\dagger$ 产生动量为 $\mathbf{k}$ 的声子。连续场被量子化成可计数的准粒子——这正是文小刚「从声子的起源到光子和电子的起源」的出发点。

**Bogoliubov 变换**：对有相互作用的玻色系统，二次量子化的哈密顿量含 $a a$ 与 $a^\dagger a^\dagger$ 项，通过混合产生与湮灭算符的线性变换 $b_{\mathbf{k}} = u_{\mathbf{k}}a_{\mathbf{k}} + v_{\mathbf{k}}a_{-\mathbf{k}}^\dagger$ 对角化。这个技巧在第 3 篇玻色凝聚、第 5 篇 BCS 超导中会反复出现。

**局域相互作用**：δ 型两体相互作用在坐标表象写成 $V \sum_{i<j}\delta(\mathbf{r}_i-\mathbf{r}_j)$，在二次量子化语言里化为 $\frac{V}{2}\int d\mathbf{r}\,\hat{\psi}^\dagger\hat{\psi}^\dagger\hat{\psi}\hat{\psi}$——Hubbard 模型的相互作用项正是它的格点版本。

## 7 三种视角的统一与辨析

| 视角 | 基本对象 | 优势 | 典型场合 |
| --- | --- | --- | --- |
| 坐标表象 | $\Psi(\mathbf{r}_1,\dots,\mathbf{r}_N)$ | 直观、贴近薛定谔方程 | 原子、少体系统 |
| 占据数表象 | $\|n_1,n_2,\dots\rangle$ | 全同粒子内建、粒子数可变 | 凝聚、超导、磁性 |
| 场算符 | $\hat{\psi}(\mathbf{r})$ | 连接粒子与场、连续化 | Green 函数、路径积分、相对论推广 |

## 8 小结

- **产生/湮灭算符**是二次量子化的字母表：$a_\lambda^\dagger$ 加粒子、$a_\lambda$ 取粒子，数算符 $\hat{n}_\lambda=a_\lambda^\dagger a_\lambda$ 测占据数。
- **Fock 空间**是粒子数可变的多体态全集，$|n_1,n_2,\dots\rangle$ 由产生算符作用真空构造，玻色因子 $\sqrt{n_\lambda!}$ 保证归一。
- **玻色对易、费米反对易**自动实现泡利不相容与交换符号——统计性质不再靠手工对称化。
- **场算符** $\hat{\psi}(\mathbf{r})=\sum_\lambda\varphi_\lambda(\mathbf{r})a_\lambda$ 把态指标换成空间点，波函数升格为算符。
- 二次量子化的**哈密顿量**由单粒子项加两体项构成，产生算符排在左边（正规序），系数 $1/2$ 防重复计数。
- 这套语言是后面 Green 函数、Feynman 图、平均场理论的共同地基，也是本专题与量子场论衔接的接口。

## 9 公式速查：一页纸复习

| 对象 | 表达式 | 一句话要点 |
| --- | --- | --- |
| 占据数态 | $\|n_1,n_2,\dots\rangle$ | Fock 空间基 |
| 数算符 | $\hat{n}_\lambda=a_\lambda^\dagger a_\lambda$ | 测占据数 |
| 玻色对易 | $[a_\lambda,a_\mu^\dagger]=\delta_{\lambda\mu}$ | 对称统计，可堆叠 |
| 费米反对易 | $\{c_\lambda,c_\mu^\dagger\}=\delta_{\lambda\mu}$ | 泡利不相容内建 |
| 场算符 | $\hat{\psi}(\mathbf{r})=\sum_\lambda\varphi_\lambda(\mathbf{r})a_\lambda$ | 波函数变算符 |
| 密度算符 | $\hat{n}(\mathbf{r})=\hat{\psi}^\dagger\hat{\psi}$ | 实空间粒子密度 |

**易错复盘**：其一，$a^\dagger a$ 与 $aa^\dagger$ 相差单位算符（玻色）或 $(1-2n)$（费米）——顺序不能随意换；其二，费米产生算符作用已满态得零；其三，场算符是算符不是波函数。这些点在后续 Feynman 图的正负号里会反复找上门。

**知识连线**：场算符是第 1 篇《二次量子化》的深化版本，直接喂给第 2 篇《有限温度形式》的虚时场论与第 5 篇《费曼图与微扰论》的 Wick 收缩。「把连续场离散成可增减的单元」与「把连续表示离散成 token 序列」同构——这是本篇与「从极限到大模型」主线最早的一处连接。

在下一节，我们将把二次量子化搬到**有限温度**：引入虚时演化与密度矩阵，在巨正则系综里重新表述多体理论。
