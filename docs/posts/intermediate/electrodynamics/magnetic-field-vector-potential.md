---
title: 磁场与矢势
date: 2026-08-07
---

# 磁场与矢势

<div class="epigraph">
<p>无源之场必为某矢量场的旋度——磁场的势，是矢势。</p>
<footer>—— 赫尔曼 · 冯 · 亥姆霍兹（Hermann von Helmholtz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第三章 §3.2 ｜ 2026-08-07</p>
</div>

## 为什么磁场需要「矢势」

静电学里，无旋的电场 $\nabla\times\mathbf{E} = 0$ 引出标量势 $\varphi$，把三个分量的场压缩成一个标量。磁场 $\mathbf{B}$ 不是无旋的（$\nabla\times\mathbf{B} = \mu_0\mathbf{J} \neq 0$），不能写成本征标量势的梯度——但它是**无源**的（$\nabla\cdot\mathbf{B} = 0$），这对应另一条数学定理：无源场必是某矢量场的旋度。于是存在**磁矢势（vector potential）$\mathbf{A}$**，满足

$$\mathbf{B} = \nabla\times\mathbf{A}$$

矢势用三个分量换来磁场三个分量的自动满足 $\nabla\cdot\mathbf{B} = 0$，看似没省力——但它在计算、对称性、以及量子力学（阿哈罗诺夫-玻姆效应）中都是不可替代的。<span class="marginnote">矢势的重要性在经典层面被低估，直到 1959 年阿哈罗诺夫-玻姆实验证明：电子经过无磁场的区域（$\mathbf{B} = 0$ 但 $\mathbf{A} \neq 0$）仍会受 $\mathbf{A}$ 影响产生干涉条纹移动——矢势不只是数学工具，它是物理实体。这个发现让「$\mathbf{A}$ 只是辅助量」的旧观念被彻底改写。</span>

## 1 矢势的存在性与规范自由度

由 $\nabla\cdot\mathbf{B} = 0$ 与「旋度的散度恒为零」，至少存在一个 $\mathbf{A}$ 使 $\mathbf{B} = \nabla\times\mathbf{A}$。但 $\mathbf{A}$ 远非唯一：若把 $\mathbf{A}$ 换成

$$\mathbf{A}' = \mathbf{A} + \nabla\psi$$

（$\psi$ 是任意标量函数），则 $\nabla\times\mathbf{A}' = \nabla\times\mathbf{A} = \mathbf{B}$——因为梯度的旋度恒为零。这称为**规范变换（gauge transformation）**，矢势的多余自由度称为**规范自由度**。<span class="marginnote">规范自由度不是缺陷，而是资源：我们可以挑最方便的 $\mathbf{A}$ 来用。静磁场最常用的选择是<strong>库仑规范</strong> $\nabla\cdot\mathbf{A} = 0$，它把 $\mathbf{A}$ 的散度固定，让计算更简洁。这个「用自由度简化计算」的思想，在电磁波与相对论章节还会以洛伦兹规范的形式出现。</span>

## 2 矢势的微分方程与积分公式

把 $\mathbf{B} = \nabla\times\mathbf{A}$ 代入安培环路定理 $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$，利用矢量恒等式 $\nabla\times(\nabla\times\mathbf{A}) = \nabla(\nabla\cdot\mathbf{A}) - \nabla^2\mathbf{A}$，并取库仑规范 $\nabla\cdot\mathbf{A} = 0$，得

$$\nabla^2\mathbf{A} = -\mu_0\mathbf{J}$$

——**矢势满足矢量泊松方程**，每个分量都形如 $\nabla^2 A_i = -\mu_0 J_i$，与标量泊松方程 $\nabla^2\varphi = -\rho/\varepsilon_0$ 完全同构。因此可以直接写积分解：

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}V'$$

对照静电的 $\varphi = \frac{1}{4\pi\varepsilon_0}\int \frac{\rho}{r}\,\mathrm{d}V'$——**矢势的每个分量就是「把电荷密度换成相应电流密度分量」的结果**。<span class="marginnote">这个「逐分量类比」极其实用：会解静电场的人，把 $\rho \to J_i$、$1/\varepsilon_0 \to \mu_0$ 代换，就能得到磁场的矢势。唯一要注意的是 $\mathbf{A}$ 是矢量，积分要分三个分量做。</span>

**典型结果**：无限长直导线（电流 $I$）的矢势 $\mathbf{A} = \dfrac{\mu_0 I}{2\pi}\ln\dfrac{r_0}{r}\hat{\mathbf{z}}$（方向沿电流，对数发散，故需参考点 $r_0$）；圆电流轴线上的矢势方向沿圆周切线。取旋度即可验证 $\mathbf{B}$ 的已知结果。

## 3 公式解析：从 $\mathbf{B}$ 的方程到 $\mathbf{A}$ 的泊松方程

这条推导是本节的核心，也是「为什么要取库仑规范」的最好说明：

- **第一步，代入恒等式**：$\nabla\times\mathbf{B} = \mu_0\mathbf{J}$ 换成 $\nabla\times(\nabla\times\mathbf{A}) = \mu_0\mathbf{J}$。左边展开为 $\nabla(\nabla\cdot\mathbf{A}) - \nabla^2\mathbf{A}$。
- **第二步，利用规范自由度**：$\nabla(\nabla\cdot\mathbf{A})$ 这一项的存在让方程既耦合又复杂。但规范变换让我们自由选择 $\nabla\cdot\mathbf{A}$——取 $\nabla\cdot\mathbf{A} = 0$（库仑规范），这一项恰好消失，方程退化为三个独立的标量泊松方程 $\nabla^2 A_i = -\mu_0 J_i$。<span class="marginnote">这里有个细微但重要的点：库仑规范下 $\mathbf{A}$ 是否总能选得让 $\nabla\cdot\mathbf{A} = 0$？答案是能——对任意 $\mathbf{A}$，解泊松方程 $\nabla^2\psi = -\nabla\cdot\mathbf{A}$ 找到 $\psi$，则 $\mathbf{A}' = \mathbf{A} + \nabla\psi$ 自动满足 $\nabla\cdot\mathbf{A}' = 0$。规范自由度总够用。</span>
- **第三步，解泊松方程**：矢量泊松方程的分量形式与标量完全相同，直接用静电的格林函数解逐分量写出，即得积分公式。整个过程说明：**矢势的价值在于把「矢量场的旋度方程」变成「三个标量场的高斯型方程」**——把「难」的数学翻译成「已会」的数学。

**辨析｜易错点：** $\mathbf{A}$ 的积分公式只在**库仑规范**下成立。用其他规范（如洛伦兹规范）时，$\mathbf{A}$ 满足的方程与积分表达式会多出附加项。公式与规范是绑定的，混用规范会导致错误。另外，由 $\mathbf{B} = \nabla\times\mathbf{A}$ 反推 $\mathbf{A}$ 需要附加规范条件，否则解不唯一——「给定 $\mathbf{B}$ 无法唯一确定 $\mathbf{A}$」是规范自由度的另一面。

## 4 磁场边界条件与矢势的连续性

磁场的界面条件（由麦克斯韦方程组的积分形式推得）：

- **法向分量连续**：$B_{1n} = B_{2n}$（无磁单极子的直接推论）。
- **切向分量（无面电流时）**：$H_{2t} - H_{1t} = 0$；有自由面电流 $\mathbf{K}$ 时，$\hat{\mathbf{n}}\times(\mathbf{H}_2 - \mathbf{H}_1) = \mathbf{K}$。

用矢势表述，界面条件化为：$\mathbf{A}$ 连续（切向分量连续），且法向导数的跳跃由面电流决定。<span class="marginnote">边界条件在求解「介质 + 电流」的磁场问题时是必备工具：先求出各区域 $\mathbf{A}$ 的一般解，再用界面条件定系数，与静电边值问题的流程完全平行。这正是「电-磁类比」的又一次体现。</span>

## 5 矢势的更深层意义

- **能量表述**：磁场能量可用矢势写出 $W_m = \frac{1}{2}\int \mathbf{J}\cdot\mathbf{A}\,\mathrm{d}V$——与静电 $W_e = \frac{1}{2}\int\rho\varphi\,\mathrm{d}V$ 完美类比（见《磁场能量与磁化》）。
- **规范理论之始**：矢势的规范自由度在量子电动力学中发展为「规范场」的整套思想——电磁力、弱力、强力都从「规范对称性」中诞生。你在电动力学里学的「$\mathbf{A} \to \mathbf{A} + \nabla\psi$」，是理解整个粒子物理标准模型的第一块砖。<span class="marginnote">麦克斯韦方程组本身具有规范对称性，而「要求拉格朗日量具有局域规范对称性」直接逼迫我们引入矢量场——这就是现代粒子物理「从对称性推出相互作用」的核心方法论，它正是从这里开始的。</span>

## 6 矢势的完整例题与规范检验

矢势不像 $\mathbf{E}$、$\mathbf{B}$ 那样有直观，必须靠计算建立手感。用一个完整例子把「求 $\mathbf{A}$ → 取旋度得 $\mathbf{B}$」走通，再演示规范自由度的实际作用。

**例题：有限长直导线的矢势与磁场。** 沿 $z$ 轴从 $z_1$ 到 $z_2$ 的直导线通电流 $I$，场点距导线 $r$。用积分公式（库仑规范）：

$$A_z = \frac{\mu_0 I}{4\pi}\int_{z_1}^{z_2}\frac{\mathrm{d}z'}{\sqrt{r^2 + (z-z')^2}} = \frac{\mu_0 I}{4\pi}\ln\frac{z-z_1+\sqrt{r^2+(z-z_1)^2}}{z-z_2+\sqrt{r^2+(z-z_2)^2}}$$

取旋度 $B_r = -\partial A_z/\partial z$、$B_z = 0$，得到与毕奥-萨伐尔定律完全一致的磁场。**这个例子展示的流程——「积分求 $\mathbf{A}$，再取旋度」——是矢势法最标准的操作。**

**规范自由度的实际检验**：把上面的 $\mathbf{A}$ 加上任意标量场的梯度 $\nabla\psi$，比如 $\psi = \lambda z$，则 $\mathbf{A}' = \mathbf{A} + \lambda\hat{\mathbf{z}}$。计算 $\nabla\times\mathbf{A}' = \nabla\times\mathbf{A}$——磁场不变。**「加梯度不改变磁场」不是抽象定理，你可以亲手验证：任何常数轴向平移（$\lambda\hat{\mathbf{z}}$）或任意 $\psi$ 的梯度，都改变不了 $\mathbf{B}$。**

**为什么矢势仍是必需的：** 磁场由 $\nabla\times\mathbf{A}$ 给出时，安培定律 $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$ 变成对 $\mathbf{A}$ 的二阶方程，在时变问题与量子效应中，$\mathbf{A}$ 是比 $\mathbf{B}$ 更基本的量。阿哈罗诺夫-玻姆效应里，电子在 $\mathbf{B} = 0$ 的区域仍因 $\mathbf{A} \neq 0$ 而改变干涉相位——**如果你还认为 $\mathbf{A}$ 只是「计算工具」，这个实验会彻底推翻你的观念。**

**辨析｜易错点：** ① 库仑规范的积分公式只对「电流分布有限」的情形成立；无限长导线的 $\mathbf{A}$ 用该式发散，需要像静电势那样设定参考点。② 取旋度时 $\nabla\times(\nabla\psi) = 0$ 恒成立，但**不同规范下的 $\mathbf{A}$ 在界面上的连续性不同**，用界面条件定系数前要先固定规范。③ 由 $\mathbf{B}$ 反求 $\mathbf{A}$ 没有唯一解——必须先定规范，再谈「求」。

**矢势的规范固定三选一**：静磁场常用**库仑规范** $\nabla\cdot\mathbf{A} = 0$（简化泊松方程）；时变电磁场常用**洛伦兹规范** $\nabla\cdot\mathbf{A} + \dfrac{1}{c^2}\dfrac{\partial\varphi}{\partial t} = 0$（把推迟势方程化成对称的四维形式）；量子场论里常用**辐射规范**（库仑规范 + 无源条件）。**选择哪种规范，取决于「想让哪个方程最简洁」**——规范自由度的存在，正是为了让你总能挑到最顺手的那把尺子。

**阿哈罗诺夫-玻姆效应的定量图像**：电子双缝实验中，在双缝之后放置一根细长螺线管（管内 $\mathbf{B}\neq 0$、管外 $\mathbf{B} = 0$ 但 $\mathbf{A} \neq 0$）。电子分两路绕过螺线管，两路的相位差为 $\Delta\phi = \dfrac{e}{\hbar}\oint\mathbf{A}\cdot\mathrm{d}\mathbf{l} = \dfrac{e\Phi}{\hbar}$——**磁通 $\Phi$ 直接进入干涉条纹**。尽管电子从未进入 $\mathbf{B}\neq 0$ 的区域，条纹却随磁通移动。这个 1959 年的实验说明：**电磁场的基本描述量是势 $\mathbf{A},\varphi$，而不是场 $\mathbf{E},\mathbf{B}$**——矢势不是「计算辅助」，而是物理本身。

**由矢势算磁通的捷径**：穿过曲面 $S$ 的磁通可以用矢势的环量表示：

$$
\Phi = \int_S \mathbf{B}\cdot\mathrm{d}\mathbf{S} = \int_S (\nabla\times\mathbf{A})\cdot\mathrm{d}\mathbf{S} = \oint_{\partial S} \mathbf{A}\cdot\mathrm{d}\mathbf{l}
$$

**磁通 = 矢势沿边界曲线的环量**。这个「斯托克斯定理」的改写非常有用：算磁通不必做面积分，只要沿边界走一圈积分 $\mathbf{A}$ 就行。在电磁感应（法拉第定律）与阿哈罗诺夫-玻姆效应中，$\oint\mathbf{A}\cdot\mathrm{d}\mathbf{l}$ 直接决定感应电动势与相位差——**矢势的环量是比「面积分磁通」更基本的量**。

**辨析｜易错点：** ① 由 $\mathbf{B}$ 求 $\mathbf{A}$ 不唯一，必须给定规范；但**磁通（或 $\oint\mathbf{A}\cdot\mathrm{d}\mathbf{l}$）是规范不变的**——任意 $\mathbf{A}\to\mathbf{A}+\nabla\psi$ 不改变环量（$\oint\nabla\psi\cdot\mathrm{d}\mathbf{l} = 0$）。这个「环量规范不变」的性质，保证了物理量不依赖你选的规范。② 斯托克斯定理要求曲面可定向、边界光滑，理想化模型里这些条件通常自动满足。③ 无限长导线的 $\mathbf{A}$ 含对数项发散，算环量时用有限长度或参考点来处理。

**矢势与标势的四维统一**：在相对论框架下，标势 $\varphi$ 与矢势 $\mathbf{A}$ 组合成**四维势** $A^\mu = (\varphi/c, \mathbf{A})$。电磁场张量正是四维势的四维旋度：$F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu$。**「电场磁场互相转化」的本质，是四维势的分量在洛伦兹变换下混合**——$\varphi$ 与 $\mathbf{A}$ 本是一体的两面，正如 $\mathbf{E}$ 与 $\mathbf{B}$。规范变换 $A^\mu \to A^\mu + \partial^\mu\psi$ 在四维语言里同样成立，且自动保持 $F^{\mu\nu}$ 不变。**把 $\mathbf{A}$ 与 $\varphi$ 统一成 $A^\mu$，是通往相对论电磁学（第六章）的桥梁。**

**对称性与规范自由度的深层意义**：为什么「加一个梯度不改变物理」？因为这对应电磁理论的**规范对称性**——物理规律在变换 $A^\mu \to A^\mu + \partial^\mu\psi$ 下不变。量子电动力学（QED）中，这个对称性被提升为「局域规范对称性」，并由此唯一地推出电磁相互作用的形式。**你在这里学到的「$\mathbf{A}$ 可以任意加梯度」，是二十世纪粒子物理最深刻的原理之一——规范对称性——在经典层面的第一次亮相。**

**由 $\mathbf{A}$ 求 $\mathbf{B}$ 的一个数值检验**：无限长直导线的矢势 $\mathbf{A} = \dfrac{\mu_0 I}{2\pi}\ln\dfrac{r_0}{r}\hat{\mathbf{z}}$。取旋度时用柱坐标公式：$B_\phi = -\dfrac{\partial A_z}{\partial r} = \dfrac{\mu_0 I}{2\pi r}$——**与安培环路定理的结果一致**。这个检验的价值在于确认「$\mathbf{A}$ 的对数形式」取旋度后给出正确的 $1/r$ 磁场，而不是发散——对数发散在求导后消失，**「$\mathbf{A}$ 可以发散，只要 $\mathbf{B}$ 有限」**是矢势与标势的一个隐蔽差别。

**辨析｜易错点：** ① 柱坐标旋度公式里有额外项（如 $B_\phi$ 含 $1/r$ 因子），用直角坐标公式硬套柱对称问题是常见错误。② 无限长导线取 $\mathbf{A} \propto \ln(r_0/r)$ 时参考点 $r_0$ 必须存在，否则 $\mathbf{A}$ 无定义——**「势需要一个参考点」在静电标势与磁矢势中同样成立**。③ 矢势的叠加遵循矢量叠加：多个电流源的 $\mathbf{A}$ 直接相加，取旋度后得到总 $\mathbf{B}$。

**矢势与磁通量的规范不变性对照**：虽然 $\mathbf{A}$ 本身依赖规范，但两个物理量是规范不变的——磁场 $\mathbf{B} = \nabla\times\mathbf{A}$ 与磁通 $\Phi = \oint\mathbf{A}\cdot\mathrm{d}\mathbf{l}$。**「势依赖规范、可观测量不依赖规范」是规范理论的基本信条**：任何物理预言都必须从规范不变的量读出。这个原则在量子场论中升华为「可观测算符必须规范不变」，是理解标准模型的一条主线。

**本节在电动力学中的位置**：矢势把「磁场无源」转化为「存在 $\mathbf{A}$」，与「电场无旋 → 存在 $\varphi$」形成对偶。**这两条势的存在性定理是整门课的方法论地基**：静电场靠 $\varphi$ 简化为标量方程，静磁场靠 $\mathbf{A}$ 简化为矢量方程，时变场靠 $(\varphi,\mathbf{A})$ 统一成四维势。后续的磁标势、磁多极、以及辐射理论，全部从这里出发。

## 7 小结

- **磁矢势** $\mathbf{B} = \nabla\times\mathbf{A}$，存在性由无源条件保证。
- **规范自由度**：$\mathbf{A} \to \mathbf{A} + \nabla\psi$ 不改变 $\mathbf{B}$；静磁场常用**库仑规范** $\nabla\cdot\mathbf{A} = 0$。
- 库仑规范下 $\mathbf{A}$ 满足**矢量泊松方程** $\nabla^2\mathbf{A} = -\mu_0\mathbf{J}$，积分解逐分量与静电同构。
- 磁场边界条件：$B_n$ 连续、$H_t$ 连续（无面电流时）。
- 矢势在能量、规范场论与量子效应（Aharonov-Bohm）中扮演深层角色。

在下一节，我们问一个大胆的问题：既然电场有无旋区域的标势，磁场有没有「无电流区域的标势」？——**磁标势与磁多极展开**。
