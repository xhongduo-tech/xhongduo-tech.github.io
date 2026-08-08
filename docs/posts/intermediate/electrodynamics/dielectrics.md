---
title: 电介质中的静电场
date: 2026-08-07
---

# 电介质中的静电场

<div class="epigraph">
<p>物质并不是旁观者——它在电场中被极化，又以极化电荷反哺电场。</p>
<footer>—— 整理自彼得 · 德拜（Peter Debye）的偶极子思想</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第二章 §2.6 ｜ 2026-08-07</p>
</div>

## 为什么真空中的静电场还不够

到目前为止，所有场方程都是在真空中写的。但工程与自然的绝大多数场景里都有**电介质（dielectric）**——绝缘体、玻璃、水、生物组织。电介质中虽然没有自由电荷流动，但电场会让分子尺度上的电荷发生微小的重新分布，产生**极化**。极化的结果是在材料内部与表面出现**束缚电荷**，它们反过来成为新的电场源。要让麦克斯韦方程组在有物质时继续好用，需要引入新的场量 $\mathbf{D}$ 与材料参数 $\varepsilon$。<span class="marginnote">电介质的价值远超「绝缘」：电容器靠它提高容值、手机屏幕靠它的介电响应、生物膜靠它维持电位差。介电常数 $\varepsilon$ 是材料最基本的电学指纹之一。</span>

## 1 极化的微观机制

电场作用下，电介质分子发生三种极化：

**电子极化（electronic polarization）**：原子外层电子云相对原子核微小偏移，原子被拉成瞬时偶极子。所有物质都有，响应最快。<span class="marginnote">电子极化大约在 $10^{-15}\ \mathrm{s}$ 内完成——光频场也能驱动它。这就是为什么玻璃的折射率与它的介电常数相关（麦克斯韦关系 $n \approx \sqrt{\varepsilon_r}$）。</span>
- **离子极化（ionic polarization）**：正负离子在晶格中相对位移。离子晶体（如 NaCl）的特征，响应在红外频段。
- **取向极化（orientational polarization）**：固有偶极矩的分子（如水 $\ce{H2O}$）在电场下转动排列。响应最慢（微波频段），且受热运动对抗——升温削弱取向极化。

**极化强度（polarization）$\mathbf{P}$**：单位体积内的电偶极矩，$\mathbf{P} = \sum \mathbf{p}_i / \Delta V$，单位 $\mathrm{C/m^2}$。它把微观的偶极子分布打包成宏观连续量。

## 2 束缚电荷与极化电荷

极化使介质内部出现**束缚电荷密度**，表面出现**束缚面电荷密度**，它们与极化强度 $\mathbf{P}$ 的关系为

$$\rho_b = -\nabla\cdot\mathbf{P}, \qquad \sigma_b = \mathbf{P}\cdot\hat{\mathbf{n}}$$

其中 $\hat{\mathbf{n}}$ 是介质表面外法向。<span class="marginnote">推导路径：一个体积元 $V$ 的偶极矩 $\mathbf{P}$ 贡献的势 $\varphi = \frac{1}{4\pi\varepsilon_0}\int \mathbf{P}\cdot\nabla'\frac{1}{r}\,\mathrm{d}V'$，分部积分后恰好整理成「束缚电荷的势」——$\rho_b = -\nabla\cdot\mathbf{P}$ 就是从这里跑出来的。这个推导在郭硕鸿教材里是必做练习。</span>

**物理图像**：平行板电容器中插入电介质，介质两表面出现束缚面电荷——靠近正极板的表面带负束缚电荷，靠近负极板的表面带正束缚电荷。这些束缚电荷部分屏蔽了自由电荷的电场，使介质中场强减弱为 $\mathbf{E} = \mathbf{E}_0/\varepsilon_r$（$\varepsilon_r$ 为相对介电常数）。

**辨析｜易错点：** 束缚电荷与自由电荷在「产生电场」上完全等价——都是高斯定理的源。区别在于：自由电荷可以移动（导体中流动、被迁移），束缚电荷被束缚在分子上不可移动。$\nabla\cdot\mathbf{E} = (\rho_f + \rho_b)/\varepsilon_0$ 中必须把两种电荷都算上，而 $\mathbf{D}$ 的引入就是为了「只数自由电荷」也能写出简洁的高斯定理。

## 3 电位移矢量与本构关系

为了回避束缚电荷的复杂性，引入**电位移矢量（electric displacement）$\mathbf{D}$**：

$$\mathbf{D} = \varepsilon_0\mathbf{E} + \mathbf{P}$$

由 $\nabla\cdot\mathbf{E} = (\rho_f + \rho_b)/\varepsilon_0$ 与 $\rho_b = -\nabla\cdot\mathbf{P}$ 可推出

$$\nabla\cdot\mathbf{D} = \rho_f$$

**高斯定理在介质中变成「只数自由电荷」**——束缚电荷被 $\mathbf{P}$ 吸收进 $\mathbf{D}$ 的定义里了。这是引入 $\mathbf{D}$ 的唯一目的，也是它的价值所在。

对**线性各向同性**介质，极化与场成正比：$\mathbf{P} = \varepsilon_0\chi_e\mathbf{E}$，其中 $\chi_e$ 是电极化率。于是

$$\mathbf{D} = \varepsilon_0(1 + \chi_e)\mathbf{E} = \varepsilon_0\varepsilon_r\mathbf{E} = \varepsilon\mathbf{E}$$

其中 $\varepsilon_r = 1 + \chi_e$ 是相对介电常数，$\varepsilon = \varepsilon_r\varepsilon_0$ 是介电常数。**本构关系（constitutive relation）** $\mathbf{D} = \varepsilon\mathbf{E}$ 是介质中的关键一环。<span class="marginnote">各向异性晶体中 $\mathbf{D}$ 与 $\mathbf{E}$ 不再同向，$\varepsilon$ 变为张量 $\varepsilon_{ij}$；铁电材料中 $\mathbf{P}$ 与 $\mathbf{E}$ 的关系是非线性的、有滞回。线性各向同性只是最常用的一层近似，但电动力学课程几乎都止步于此。</span>

**典型值**：真空气（$\varepsilon_r = 1.0006$）、变压器油（$\approx 2.2$）、玻璃（$\approx 5$）、水（$\approx 80$，静态）、高介电陶瓷（$10^3$ 量级）。

## 4 公式解析：为什么 $\mathbf{D} = \varepsilon_0\mathbf{E} + \mathbf{P}$

这条定义看起来像循环论证（$\mathbf{D}$ 用 $\mathbf{E}$ 和 $\mathbf{P}$ 定义），它的深层含义值得拆解：

**第一步，从总电荷到两种电荷**：真空高斯定理 $\nabla\cdot\mathbf{E} = \rho/\varepsilon_0$ 中，$\rho$ 包含自由与束缚两种电荷。束缚电荷 $\rho_b = -\nabla\cdot\mathbf{P}$，代入后 $\nabla\cdot\mathbf{E} = (\rho_f - \nabla\cdot\mathbf{P})/\varepsilon_0$。
**第二步，移项合并**：把 $\nabla\cdot\mathbf{P}$ 移到左边：$\nabla\cdot(\varepsilon_0\mathbf{E} + \mathbf{P}) = \rho_f$。括号里这个新量正是 $\mathbf{D}$ 的定义。
**第三步，读出哲学**：$\mathbf{D}$ 不是「又一个新的场」，而是「把束缚电荷效应并入之后的电场」——它让我们在介质中沿用「电荷是源」的直觉，只不过数的是**自由电荷**。反过来，$\mathbf{P}$ 是「物质对电场的响应」，$\varepsilon_0\mathbf{E}$ 是「真空中的场」，两者相加得到「物质中看得见的场源方程」。<span class="marginnote">注意 $\mathbf{D}$ 的通量性质：$\oint\mathbf{D}\cdot\mathrm{d}\mathbf{S} = Q_{f,\text{内}}$ 对任意闭合面成立。但它不像 $\mathbf{E}$ 那样有简单的库仑源——介质中 $\mathbf{D}$ 的旋度 $\nabla\times\mathbf{D} = \nabla\times\mathbf{P}$ 一般不为零。所以「$\mathbf{D}$ 有源无旋」是错的，$\mathbf{D}$ 只是「源的计算方便」。</span>

**辨析｜易错点：** 许多教材说「$\mathbf{D}$ 的源是自由电荷」，但这不意味着 $\mathbf{D}$ 就是自由电荷直接产生的场——在介质界面附近，$\mathbf{D}$ 的分布受束缚电荷影响（因为 $\mathbf{P}$ 也在界面跃变）。「$\mathbf{D}$ 只看自由电荷」只在**积分形式**的意义上正确（通量等于面内自由电荷），**逐点**意义上 $\mathbf{D}$ 仍受束缚电荷约束。区分「通量」与「场值」，是介质章节最重要的辨析。

## 5 介质中的边界条件与能量

在两种介质界面两侧，连接条件为：

- $\varphi$ 连续（$\mathbf{E}$ 切向分量连续）；
- 无自由面电荷时 $\varepsilon_1\dfrac{\partial\varphi_1}{\partial n} = \varepsilon_2\dfrac{\partial\varphi_2}{\partial n}$（$\mathbf{D}$ 法向分量连续）。

**介质中静电场能量密度**推广为 $w = \frac{1}{2}\mathbf{E}\cdot\mathbf{D} = \frac{1}{2}\varepsilon E^2$。对比真空情形，介质中能量多了极化项——极化过程存储的能量与电场能量之和。

**辨析｜易错点：** $\mathbf{D}$ 的法向连续依赖「界面无自由面电荷」。若界面上有自由面电荷 $\sigma_f$，则 $D_{2n} - D_{1n} = \sigma_f$。导体-介质界面属于这一类：导体表面必有自由电荷，$\mathbf{D}$ 的法向分量恰等于 $\sigma_f$——这正是「介质中导体表面电荷密度 = $\mathbf{D}$ 的法向分量」的由来。

## 6 电介质中的完整例题：平行板电容器

把本章的全部概念（极化、束缚电荷、$\mathbf{D}$、边界条件）放进一个最经典的装置里串一遍。

**问题**：平行板电容器极板面积 $A$、间距 $d$，两极板分别带自由电荷 $\pm Q_f$。板间一半填充介电常数 $\varepsilon$ 的电介质（与极板平行、厚度恰为 $d$），求电容。

**第一步，用 $\mathbf{D}$ 的高斯定理。** 取一圆柱高斯面，一底面在极板内、另一底面在介质中，$\oint\mathbf{D}\cdot\mathrm{d}\mathbf{S} = Q_f$ 给出 $D = \sigma_f = Q_f/A$——**$\mathbf{D}$ 只由自由电荷决定，与介质无关**。

**第二步，由本构关系求场。** 介质中 $\mathbf{E} = D/\varepsilon = \dfrac{Q_f}{\varepsilon A}$；若未填介质的区域（真空）在场强 $E_0 = D/\varepsilon_0$。注意**板间电场在两种介质中不同**，因为 $D$ 连续而 $\varepsilon$ 不同。

**第三步，求电势差与电容。** 板间电压 $V = E\,d = \dfrac{Q_f d}{\varepsilon A}$，故

$$C = \frac{Q_f}{V} = \frac{\varepsilon A}{d}$$

——**填满介质的电容是真空情形的 $\varepsilon_r$ 倍**。这解释了电容器为什么都要装介质：同样的极板与间距，介电常数翻倍，容值翻倍，储能 $\frac{1}{2}CV^2$ 也翻倍。

**第四步，检验束缚电荷。** 介质表面的束缚面电荷 $\sigma_b = \mathbf{P}\cdot\hat{\mathbf{n}} = \varepsilon_0\chi_e E$。束缚电荷部分抵消自由电荷的场，所以介质中场比真空中弱 $\varepsilon_r$ 倍。**「介质中电场减弱」与「电容增大」是同一件事的两面**：场弱了，达到同样电压所需的电荷就多了。

**辨析｜易错点：** ① $\mathbf{D} = \varepsilon\mathbf{E}$ 只对线性各向同性介质成立，各向异性时 $\mathbf{D}$ 与 $\mathbf{E}$ 不同向。② 电容器「插介质容值变大」的前提是**接电源恒压**或**恒电荷**两种情况下都能推出来，但能量的变化方向不同（恒 $V$ 时电池供能，恒 $Q$ 时能量减少）——用能量讨论前先看清约束。③ 介质中的束缚电荷不能自由移动，把它当成「可导走」的自由电荷是完全错误的。

**介质击穿与安全**：每种电介质都有**击穿场强**——超过该场强，介质失去绝缘性（电子雪崩、电弧击穿）。空气约 $3\ \mathrm{MV/m}$，变压器油约 $20\ \mathrm{MV/m}$，云母可达 $100\ \mathrm{MV/m}$。**「耐压」的本质是「在给定间距下，场强不超过击穿阈值」**：$V_{\max} = E_{\text{击穿}}\cdot d$。这就是为什么高压设备的绝缘距离要足够大、为什么要用高击穿场强的绝缘材料。

**介电常数与光速、折射率的联系**：麦克斯韦关系 $n \approx \sqrt{\varepsilon_r\mu_r}$（非磁性介质 $\mu_r = 1$ 时 $n = \sqrt{\varepsilon_r}$）把**介电常数与折射率**直接挂钩——光在介质中的折射，本质是电磁波与介电常数的相互作用。水 $\varepsilon_r \approx 80$（静态）却折射率仅 $1.33$，差异来自水的取向极化在光频跟不上振荡（色散）——**「静态介电常数」与「光频介电常数」是两回事**，这个区别在光学与微波工程里至关重要。

**电介质极化与宏观测量的联系**：介电常数不是理论参数，而是可测量的材料指纹。测量手段包括：电容法（测平行板电容反推 $\varepsilon$）、传输线法（测介质填充波导的传播常数）、谐振腔微扰法（测腔体频移）。**「材料在电场下的响应」被压缩进一个 $\varepsilon$，而这一个数就决定它在电容、波导、天线中的全部行为**——这就是介电常数如此重要的原因。

**辨析｜易错点：** ① $\mathbf{D}$ 的高斯定理只对「静电场」严格成立，时变场中 $\nabla\cdot\mathbf{D} = \rho_f$ 仍成立但 $\mathbf{D}$ 的物理解释更复杂（含位移电流）。② 铁电材料的 $\mathbf{P}$–$\mathbf{E}$ 关系有磁滞（类似铁磁），「线性各向同性」假设不适用。③ 电介质的极化是「束缚电荷」的再分布，与导体的「自由电荷」流动是两种完全不同的响应——导体内部场为零，介质内部场不为零。

## 7 小结

- **极化**三种机制：电子、离子、取向极化；宏观量为极化强度 $\mathbf{P}$。
- **束缚电荷** $\rho_b = -\nabla\cdot\mathbf{P}$、$\sigma_b = \mathbf{P}\cdot\hat{\mathbf{n}}$，与自由电荷同为高斯源。
- **电位移矢量** $\mathbf{D} = \varepsilon_0\mathbf{E} + \mathbf{P}$，介质高斯定理 $\nabla\cdot\mathbf{D} = \rho_f$。
- 线性各向同性介质 $\mathbf{D} = \varepsilon\mathbf{E}$，$\varepsilon_r = 1 + \chi_e$。
- 边界条件：$\mathbf{E}$ 切向连续、$\mathbf{D}$ 法向连续（无自由面电荷）；能量密度 $w = \frac{1}{2}\mathbf{E}\cdot\mathbf{D}$。

至此静电篇收束。下一节我们将开启**静磁场与稳恒电流**：电流如何持续存在、磁场如何用矢势描述——**稳恒电流与欧姆定律**。
