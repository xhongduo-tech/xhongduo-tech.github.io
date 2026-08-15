---
title: Hartree-Fock 近似
date: 2026-08-07
---

# Hartree-Fock 近似

<div class="epigraph">
<p>把多体问题化约为单体问题的艺术，是量子化学的第一课。</p>
<footer>—— 约翰 · 斯莱特（John C. Slater）</footer>
</div>

<div class="article-byline">
<p>第四级 · 第一性原理计算与电子结构理论 ｜ R. M. Martin《Electronic Structure》第5、8章 ｜ 2026-08-07</p>
</div>

## 为什么从 Hartree-Fock 开始

「第一性原理」意味着不引入实验参数，直接从薛定谔方程出发求解材料的一切性质。但一个真实的固体含有 $10^{23}$ 量级的电子，它们之间两两库仑排斥——把这样一个**多体问题**直接丢进薛定谔方程，即使穷尽今天的超级计算机也无解。<span class="marginnote">多体（many-body）：电子数远大于二、相互作用不可忽略的量子体系。多体问题的「难」，本质上是波函数维度随电子数指数爆炸。</span>

**Hartree-Fock 近似**是第一次严肃地处理这个困境的尝试：它把「每个电子都感受到其他电子的瞬时作用」这种牵一发而动全身的描述，替换成「每个电子在一个平均场中独立运动」的图像。这正是「从极限到大模型」这条主线的又一个枢纽——**把无穷退化为平均，把多体退化为单体**。今天的密度泛函理论、GW 方法、量子蒙特卡洛，全都从这个近似出发，只是用不同方式修补它留下的遗憾。

## 1 多电子薛定谔方程与玻恩-奥本海默近似

一个由 $N$ 个电子与若干原子核组成的体系，其非相对论薛定谔方程写为：

$$
\hat{H}\Psi = E\Psi, \qquad
\hat{H} = \hat{T}_e + \hat{V}_{ee} + \hat{V}_{eN} + \hat{T}_N + \hat{V}_{NN}
$$

其中 $\hat{T}_e$ 是电子动能，$\hat{V}_{ee}$ 是电子-电子库仑排斥，$\hat{V}_{eN}$ 是电子-核吸引，$\hat{T}_N$ 与 $\hat{V}_{NN}$ 是核的动能与核间排斥。原子核质量比电子大三到五个数量级，运动慢得多，于是先冻结核坐标、只解电子问题，再把核当作在电子能量面上缓慢移动的经典粒子——这就是**玻恩-奥本海默近似（Born-Oppenheimer approximation）**。<span class="marginnote">玻恩-奥本海默近似是把「化学」翻译成「电子结构问题」的接口：有了它，势能面、分子几何优化、反应路径这些概念才成立。它失效的情形（如光子诱导的解离、Jahn-Teller 锥形交叉）本身就是当前研究热点。</span>

在玻恩-奥本海默近似下，电子部分的哈密顿量简化为：

$$
\hat{H}_e = \sum_i \left[ -\frac{1}{2}\nabla_i^2 - \sum_A \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|} \right] + \sum_{i<j}\frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}
$$

（采用原子单位，$\hbar = m_e = e = 1$。）这里已经可以看到「好项」与「坏项」的分野：方括号里是单体项，每个电子只跟固定的核作用；而 $\sum_{i<j}1/|\mathbf{r}_i-\mathbf{r}_j|$ 是两体项，它让电子的命运彼此纠缠——**多体困难的全部根源，就在这一项里**。

## 2 单行列式假设与反对称性

处理两体项的第一种直觉来自平均场：每个电子不再关心其他电子在每一时刻的位置，只感受它们的**平均电荷云**。若波函数写成单电子轨道的乘积

$$
\Psi(\mathbf{r}_1,\ldots,\mathbf{r}_N) = \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2)\cdots\phi_N(\mathbf{r}_N)
$$

就叫 **Hartree 近似**。它有个致命缺陷：电子是费米子，交换任意两个电子的坐标，波函数必须反号；而乘积形式的波函数交换后不变号，直接违背**泡利不相容原理**。<span class="marginnote">反对称性（antisymmetry）是费米子的出生证明：两个自旋相同的电子不能占据同一轨道，正是它支撑了元素周期表的结构，也决定了固体中费米面的存在。</span>

修正的方法是把波函数写成 **Slater 行列式**——用单电子轨道 $\phi_i$ 构造一个 $N\times N$ 行列式：

$$
\Psi = \frac{1}{\sqrt{N!}}
\begin{vmatrix}
\phi_1(\mathbf{x}_1) & \phi_2(\mathbf{x}_1) & \cdots & \phi_N(\mathbf{x}_1) \\
\phi_1(\mathbf{x}_2) & \phi_2(\mathbf{x}_2) & \cdots & \phi_N(\mathbf{x}_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(\mathbf{x}_N) & \phi_2(\mathbf{x}_N) & \cdots & \phi_N(\mathbf{x}_N)
\end{vmatrix}
$$

其中 $\mathbf{x}_i$ 同时包含空间坐标与自旋坐标。交换两行，行列式反号——反对称性自动满足；两列相同则行列式为零——两个电子不能进同一轨道，泡利原理自动满足。<span class="marginnote">Slater 行列式的优雅在于：它把「反对称」这条费米子律法内建进数学结构，而不是外加约束。行列式的展开恰好覆盖了所有「每个轨道一个电子、电子两两不同」的排布方式。</span>

## 3 从 Hartree 方程到 Hartree-Fock 方程

把 Slater 行列式代入电子哈密顿量求期望值，用变分原理极小化能量，就得到 **Hartree-Fock 方程**：

$$
\hat{f}_i\,\phi_i(\mathbf{x}) = \varepsilon_i\,\phi_i(\mathbf{x})
$$

这里的 $\hat{f}_i$ 是 **Fock 算符**，它不再是普适的哈密顿量，而是依赖待求解轨道的有效单体算符——这就是**自洽场（self-consistent field, SCF）**一词的由来：解出来的轨道反过来定义算符，算符再决定新轨道，如此循环直到收敛。

Fock 算符具体写为：

$$
\hat{f}_i = -\frac{1}{2}\nabla^2 - \sum_A\frac{Z_A}{|\mathbf{r}-\mathbf{R}_A|} + v^{\mathrm{H}}(\mathbf{r}) + \hat{v}^{\mathrm{X}}_i
$$

其中 **Hartree 势** $v^{\mathrm{H}}(\mathbf{r})$ 描述电子感受到的平均库仑排斥：

$$
v^{\mathrm{H}}(\mathbf{r}) = \int \frac{n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}\mathbf{r}', \qquad n(\mathbf{r}) = \sum_i |\phi_i(\mathbf{r})|^2
$$

而 $\hat{v}^{\mathrm{X}}_i$ 是**交换算符**，它来自反对称性，是 Hartree-Fock 区别于 Hartree 近似的唯一新增物，也是下一节公式解析的主角。

## 4 公式解析：交换项的物理与数学

**交换能是 Hartree-Fock 的灵魂，也是最容易被误解的项。** 先看它长什么样：

$$
E^{\mathrm{X}} = -\frac{1}{2}\sum_{ij}\int\!\int \frac{\phi_i^*(\mathbf{x}_1)\phi_j^*(\mathbf{x}_2)\,\phi_j(\mathbf{x}_1)\phi_i(\mathbf{x}_2)}{|\mathbf{r}_1-\mathbf{r}_2|}\,\mathrm{d}\mathbf{x}_1\,\mathrm{d}\mathbf{x}_2
$$

分三步拆解这条式子：

- **第一步，看清它的形态**：把分母上方的积分核与 Hartree 项 $n(\mathbf{r}_1)n(\mathbf{r}_2) = \sum_{ij}|\phi_i(\mathbf{r}_1)|^2|\phi_j(\mathbf{r}_2)|^2$ 对比——Hartree 项把 $i$ 电子的密度与 $j$ 电子的密度相乘，而交换项把 $i$ 与 $j$ 的轨道**交叉**相乘：$\phi_i^*(\mathbf{x}_1)\phi_j(\mathbf{x}_1)$ 在前一个坐标，$\phi_j^*(\mathbf{x}_2)\phi_i(\mathbf{x}_2)$ 在后一个坐标。
- **第二步，理解符号**：交换项前带负号，且 $i=j$ 的自作用项与 Hartree 项中 $i=j$ 的部分恰好相消——**Hartree 势让每个电子「与自己」的静电场相互作用，这是虚假的自作用（self-interaction）；交换项把这份自作用精确抵消掉**。这是 Hartree-Fock 的一个了不起的副作用。
- **第三步，看见物理图像**：反对称波函数迫使同自旋电子彼此疏远，每个电子周围形成一个「交换空穴」（exchange hole）——少了一份同自旋电子的密度。交换能正是电子与其交换空穴之间的库仑吸引。**把交换项理解为「同自旋电子的动态避让」**，你就抓住了它的全部直觉。

**辨析｜易错点：** 交换项只作用于**同自旋**电子对。自旋相反的电子之间不存在交换相互作用，它们仍然被 Hartree 势「平均地」排斥——这正是 Hartree-Fock 最大的短板：它完全忽略自旋反平行电子之间的动态关联。

## 5 Hartree-Fock 的能量与迭代

把 Fock 算符解出的 $\phi_i$ 代回能量期望值，总能量可写成轨道之和减去双计数修正：

$$
E = \sum_i \varepsilon_i - \frac{1}{2}\int\!\!\int \frac{n(\mathbf{r})n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}\mathbf{r}\,\mathrm{d}\mathbf{r}' - E^{\mathrm{X}}
$$

这给出一个实用的判据：**SCF 收敛后，轨道能 $\varepsilon_i$ 之和并不等于总能量**——Hartree 项被数了两遍，必须扣回去。这个「减双计数」的技巧在所有平均场方法（含 DFT）中都会反复出现。

**SCF 迭代的流程**是：猜一组初始轨道 $\rightarrow$ 构造 $n(\mathbf{r})$ 与 Fock 算符 $\rightarrow$ 解本征方程得新轨道 $\rightarrow$ 检查能量是否变化小于阈值 $\rightarrow$ 未收敛则用新轨道回到第二步。Hartree-Fock 计算量标度为 $O(N^4)$（$N$ 为基函数个数），因为交换项的四个轨道指标都要求和——这比后文要讲的密度泛函理论贵一个数量级，也是它难以直接用于大体系的原因。

## 6 Hartree-Fock 的成就与极限

Hartree-Fock 在量子化学里立下了两个里程碑：其一，**它给出了「电子结构」这个概念的严格起点**，分子轨道、能级、电离能这些语言都由它奠定；其二，**它对原子与分子基态几何、键长、振动的描述在定性上常常正确**，是许多量化软件的默认起步。

但它有两个系统性的缺陷，理解它们才能理解后来一切方法的动机：

**关联能缺失。** 把精确能量与 Hartree-Fock 能量之差定义为**关联能（correlation energy）**：

$$
E_{\mathrm{corr}} = E_{\mathrm{exact}} - E_{\mathrm{HF}}
$$

它始终为负，数值上只占总能的一小部分（约 1%），却决定了化学反应能否发生、分子能否解离——**Hartree-Fock 连氢分子的解离都描述不好**。<span class="marginnote">关联能之「小」是能量尺度上的小，之「重」是化学上的重：1% 的能量差就足以决定一个键能否断裂。所以后文的 DFT 与波函数方法，本质上都是「如何在可控代价下找回关联能」的竞赛。</span>

**单行列式太僵硬。** 一个 Slater 行列式无法同时描述两个竞争性的电子排布（如键断裂时两个原子各自成对的两种构型）。要修正，就得做多组态展开——把多个 Slater 行列式线性叠加，这引出 CISD、CCSD 等**组态相互作用与耦合簇**方法，计算代价进一步爆炸。

## 7 小结

- **Hartree-Fock 近似**把多体薛定谔方程化约为单电子方程，代价是只保留平均场与交换作用，丢掉动态关联。
- **Slater 行列式**内建反对称性，自动满足泡利原理；交换项抵消 Hartree 自作用，并产生同自旋电子间的「交换空穴」。
- Fock 算符依赖待解轨道，必须通过 **SCF 自洽迭代**求解，计算量约 $O(N^4)$。
- **关联能** $E_{\mathrm{exact}}-E_{\mathrm{HF}}$