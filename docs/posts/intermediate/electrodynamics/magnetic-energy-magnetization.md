---
title: 磁场能量与磁化
date: 2026-08-07
---

# 磁场能量与磁化

<div class="epigraph">
<p>磁场把能量储存在它自身与物质的磁化状态里——电感是它的容器，磁化是物质的回应。</p>
<footer>—— 约瑟夫 · 亨利（Joseph Henry）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第三章 §3.4 ｜ 2026-08-07</p>
</div>

## 为什么磁能公式比电能公式「脏」一点

静电能量 $W_e = \frac{\varepsilon_0}{2}\int E^2\,\mathrm{d}V$ 干净利落；磁场能量 $W_m = \frac{1}{2}\int\mathbf{B}\cdot\mathbf{H}\,\mathrm{d}V$ 形式对称，但含有一个微妙的疑问：磁能到底是「储存在场里」还是「储存在电流回路里」？答案藏在建立磁场的过程中——为了让电流增长，电源必须克服感应电动势做功，这份功的一部分化为磁场能量。而物质被磁场**磁化**后，磁能还与材料的磁化曲线（$\mathbf{B}$–$\mathbf{H}$ 关系）有关，于是出现「线性磁介质」与「非线性磁介质（铁磁）」两种截然不同的能量行为。<span class="marginnote">回忆静电的能量建立过程：搬移电荷时克服已存在的电场做功，能量储存在电场。磁场不同——电流建立时没有「已存在的磁荷」可搬，能量来自电磁感应：电路自感反抗电流变化，电源做的功被「锁进」磁场。这个差异是理解 $W = \frac{1}{2}LI^2$ 物理来源的钥匙。</span>

## 1 磁场能量与电感

对单回路，电流从 0 增到 $I$，磁通从 0 增到 $\Phi = LI$（$L$ 为自感）。电源克服自感电动势 $\mathcal{E} = -L\,\mathrm{d}I/\mathrm{d}t$ 做的功为

$$W_m = \int_0^I L I'\,\mathrm{d}I' = \frac{1}{2}L I^2$$

对多回路系统，能量含互能项：

$$W_m = \frac{1}{2}\sum_{i,j} L_{ij}I_i I_j$$

其中 $L_{ii}$ 是自感、$L_{ij} = L_{ji}$ 是互感。<span class="marginnote">自感是「电流回路自己的磁通对自己形成的阻碍」的度量，互感是「一个回路对另一个回路的感应」。互感对称 $L_{12} = L_{21}$ 对应「互易定理」——1 对 2 的感应等于 2 对 1 的感应，它同样是自伴性的体现。变压器、无线充电都是互感的应用。</span>

**用矢势表达能量**：由 $W_m = \frac{1}{2}\int\mathbf{A}\cdot\mathbf{J}\,\mathrm{d}V$（对照静电 $W_e = \frac{1}{2}\int\varphi\rho\,\mathrm{d}V$），可把能量改写为对全空间的积分：

$$W_m = \frac{1}{2}\int \mathbf{B}\cdot\mathbf{H}\,\mathrm{d}V$$

**磁场能量密度** $w_m = \frac{1}{2}\mathbf{B}\cdot\mathbf{H}$，真空中 $w_m = \dfrac{B^2}{2\mu_0}$。这个「场能密度」形式再次说明：磁场能量储存在场中，而非集中在电流线上。<span class="marginnote">两式的等价性依赖分部积分与边界项消失：$\frac{1}{2}\int\mathbf{A}\cdot\mathbf{J}\,\mathrm{d}V$ 通过 $\mathbf{J} = \nabla\times\mathbf{H}/\mu_0$ 与 $\int\mathbf{A}\cdot(\nabla\times\mathbf{H})\,\mathrm{d}V = \int\mathbf{H}\cdot(\nabla\times\mathbf{A})\,\mathrm{d}V$（差一个边界项）化为 $\frac{1}{2}\int\mathbf{H}\cdot\mathbf{B}\,\mathrm{d}V$。这个「把能量从源改写为场」的手势，与静电完全平行。</span>

**典型值**：螺线管电感 $L = \mu_0 n^2 S l$（$n$ 单位长度匝数，$S$ 截面积，$l$ 长度），能量 $\frac{1}{2}LI^2 = \frac{B^2}{2\mu_0}\cdot Sl$——恰为「能量密度 × 体积」。

## 2 磁化强度与磁介质分类

**磁化强度（magnetization）$\mathbf{M}$**：单位体积内的磁偶极矩。磁场强度定义

$$\mathbf{H} = \frac{\mathbf{B}}{\mu_0} - \mathbf{M}$$

或 $\mathbf{B} = \mu_0(\mathbf{H} + \mathbf{M})$。把 $\mathbf{H}$ 定义为「外场与磁化共同作用后、排除了 $\mu_0$ 因子的场量」，安培环路定理在介质中改写为

$$\nabla\times\mathbf{H} = \mathbf{J}_f$$

——**只数自由电流**，束缚（磁化）电流被 $\mathbf{M}$ 吸收。这与电介质中 $\nabla\cdot\mathbf{D} = \rho_f$ 只数自由电荷完全平行。<span class="marginnote">磁化电流密度 $\mathbf{J}_b = \nabla\times\mathbf{M}$，表面磁化电流密度 $\mathbf{K}_b = \mathbf{M}\times\hat{\mathbf{n}}$——与电介质束缚电荷 $\rho_b = -\nabla\cdot\mathbf{P}$、$\sigma_b = \mathbf{P}\cdot\hat{\mathbf{n}}$ 一一对应。这组「电-磁平行」可以列成一张对照表背熟。</span>

**磁介质三类**（线性近似下 $\mathbf{M} = \chi_m\mathbf{H}$，$\chi_m$ 磁化率）：

| 类型 | $\chi_m$ | $\mu_r = 1 + \chi_m$ | 实例 | 物理机制 |
| --- | --- | --- | --- | --- |
| 抗磁质 | 负，$|\chi_m| \sim 10^{-5}$ | $\mu_r \lesssim 1$ | 铋、铜、水 | 轨道感生磁矩反抗外场 |
| 顺磁质 | 正，$\chi_m \sim 10^{-5}$ | $\mu_r \gtrsim 1$ | 铝、氧 | 固有磁矩取向排列（受热扰动对抗） |
| 铁磁质 | 正且巨大（非线性） | $\mu_r \gg 1$ | 铁、钴、镍 | 磁畴协同排列，可自发磁化 |

<span class="marginnote">抗磁性在所有物质中都存在（朗道抗磁性是量子效应），只是通常被顺磁/铁磁掩盖；水的 $\mu_r \approx 0.999991$，几乎不可测，但超导体的完美抗磁性（迈斯纳效应）让 $\mu_r = 0$，是抗磁的极端形态。材料磁性的复杂性源于「轨道磁矩、自旋磁矩、热涨落、量子交换作用」的多方竞争。</span>

## 3 铁磁性、磁滞与磁能损耗

铁磁质不遵守线性关系，$\mathbf{B}$–$\mathbf{H}$ 曲线呈**磁滞回线（hysteresis loop）**：磁化后撤去外场，材料保留剩余磁化（剩磁 $B_r$），需要反向场才退磁（矫顽力 $H_c$）。<span class="marginnote">磁滞的来源是磁畴的不可逆翻转：畴壁移动要克服钉扎作用，能量以热的形式耗散。每次循环磁滞回线包围的面积 = 每周期损耗的能量密度。软磁材料（回线窄，用于变压器铁芯）要剩磁小、损耗低；硬磁材料（回线宽，用于永磁体）要矫顽力大、保持磁化。</span>

**磁化过程中的能量**：对非线性磁介质，磁能密度不是 $\frac{1}{2}\mathbf{B}\cdot\mathbf{H}$，而是沿磁化曲线累积的积分：

$$w_m = \int_0^{B} \mathbf{H}\cdot\mathrm{d}\mathbf{B}$$

**磁滞损耗（hysteresis loss）**：沿磁滞回线循环一圈，净损耗正比于回线面积。这正是变压器铁芯要选用低损耗硅钢片、且采用交变磁场时铁芯会发热的原因——变压器发热的一部分来自磁滞损耗（另一部分是涡流损耗，见《导体中的电磁波与趋肤效应》）。

**辨析｜易错点：** 铁磁质的 $\mu_r$ 不是常数，随场强变化且具历史依赖性（磁化历史决定当前状态）。把「$\mathbf{B} = \mu\mathbf{H}$ 恒成立」套到铁磁材料上会出错——磁滞回线本身就说明 $\mathbf{B}$ 不是 $\mathbf{H}$ 的单值函数。严格说，铁磁问题只能给定回线做图解法或数值处理。

## 4 公式解析：$\mathbf{H}$ 的定义为什么是 $\frac{\mathbf{B}}{\mu_0} - \mathbf{M}$

这条定义比 $\mathbf{D} = \varepsilon_0\mathbf{E} + \mathbf{P}$ 看起来更别扭（少一个 $\mu_0$ 因子、多一个减号），拆开看：

- **第一步，从库仑磁荷类比**：若存在磁单极子，磁场也会有「磁荷密度」做源。虽然磁荷不存在，但 $\mathbf{M}$ 的作用在数学上等价于一套「假想磁荷分布」——磁化强度在边界上的法向分量就是「束缚磁荷」的类比。
- **第二步，减号的意义**：$\mathbf{H} = \frac{\mathbf{B}}{\mu_0} - \mathbf{M}$ 中，$\mathbf{B}/\mu_0$ 是「总场」（含磁化贡献），$\mathbf{M}$ 是「物质的磁化」，两者之差 $\mathbf{H}$ 就是「扣除物质贡献后、由自由电流产生的部分」。它满足 $\nabla\times\mathbf{H} = \mathbf{J}_f$，所以**$\mathbf{H}$ 的源是自由电流**，如同 $\mathbf{D}$ 的源是自由电荷。<span class="marginnote">为什么是 $\mathbf{B}/\mu_0$ 而不是 $\mathbf{B}$？纯粹是单位制约定：SI 中 $\mathbf{B}$ 与 $\mu_0\mathbf{H}$ 同量纲，$\mathbf{M}$ 与 $\mathbf{H}$ 同量纲。不同单位制（高斯制）里这条定义要换写法——单位制是电磁学最磨人的地方之一，但物理图像（扣除物质贡献后归因于自由电流）不变。</span>
- **第三步，对比 $\mathbf{D}$**：$\mathbf{D} = \varepsilon_0\mathbf{E} + \mathbf{P}$ 是**加**（极化增强位移），$\mathbf{H} = \frac{\mathbf{B}}{\mu_0} - \mathbf{M}$ 是**减**。为何不对称？因为电荷是「发散型源」（电场线从电荷发出），而磁偶极矩是「涡旋型源」（磁化电流绕圈）。磁化的效果是**削弱** $\mathbf{B}/\mu_0$ 中的「磁性贡献」，所以是减号。记住「电加磁减」，就抓住了 $\mathbf{D}$ 与 $\mathbf{H}$ 定义的全部差异。

## 5 磁场的边界条件

磁介质界面的连接条件（由 $\nabla\cdot\mathbf{B} = 0$ 与 $\nabla\times\mathbf{H} = \mathbf{J}_f$ 推得）：

- **法向**：$B_{1n} = B_{2n}$——$\mathbf{B}$ 的法向分量连续（无磁荷）。
- **切向**：无自由面电流时 $H_{1t} = H_{2t}$——$\mathbf{H}$ 的切向分量连续；有面电流 $\mathbf{K}$ 时 $\hat{\mathbf{n}}\times(\mathbf{H}_2 - \mathbf{H}_1) = \mathbf{K}$。

**辨析｜易错点：** 铁磁体表面 $\mathbf{B}$ 法向连续、$\mathbf{H}$ 切向连续，但 $\mathbf{H}$ 的法向与 $\mathbf{B}$ 的切向都会发生跃变。由于铁磁质 $\mu_r$ 巨大，同样的 $\mathbf{H}$ 对应很大的 $\mathbf{B}$——这让铁芯能「集中磁通」，是电机与变压器的核心原理。把「$\mathbf{B}$ 连续」误当作「$\mathbf{B}$ 处处连续」会漏掉铁芯的聚磁效应。

## 6 磁路的完整例题：电磁铁气隙

把磁场能量、磁化、边界条件放进电磁铁的设计问题里，看它们如何协同工作。

**问题**：铁芯（相对磁导率 $\mu_r$ 很大）弯成环状，截面积 $A$，平均周长 $l$，开有宽度 $g$ 的气隙。铁芯上绕 $N$ 匝线圈通电流 $I$，求气隙中的磁场。

**第一步，用安培环路定理。** 沿磁路取积分 $\oint\mathbf{H}\cdot\mathrm{d}\mathbf{l} = NI$。铁芯内 $H_{\text{铁}} l + H_{\text{气}} g = NI$。**注意 $\mathbf{H}$ 在铁与气隙中不同，但穿过界面的磁通连续（$B_n$ 连续）**：$B_{\text{铁}} = B_{\text{气}} = B$。

**第二步，用本构关系。** 铁芯内 $H_{\text{铁}} = B/\mu$（$\mu = \mu_r\mu_0$），气隙内 $H_{\text{气}} = B/\mu_0$。代入环路方程：

$$B\left(\frac{l}{\mu} + \frac{g}{\mu_0}\right) = NI \quad\Longrightarrow\quad B = \frac{NI}{l/\mu + g/\mu_0}$$

**第三步，读出磁路本质。** 定义磁阻 $R_m = \dfrac{l}{\mu A}$、气隙磁阻 $R_g = \dfrac{g}{\mu_0 A}$，则 $B = \dfrac{NI}{A(R_m + R_g)}$，磁通 $\Phi = BA = \dfrac{NI}{R_m + R_g}$——**「磁通 = 磁动势 / 磁阻」与「电流 = 电动势 / 电阻」完全同构**。磁路分析就是把磁场问题翻译成「磁动势、磁阻、磁通」三个电路量。

**第四步，看气隙的主导作用**：因为 $\mu_r \gg 1$，铁芯磁阻 $R_m$ 通常远小于气隙磁阻 $R_g$——**气隙是磁路的「瓶颈」，决定总磁阻**。这也是为什么磁悬浮、电磁铁、变压器都极力减小气隙：气隙每增加一点，气隙磁场就显著下降。

**辨析｜易错点：** ① 铁芯内 $\mathbf{B}$ 与 $\mathbf{H}$ 的关系是非线性的（磁滞回线），实际设计要用 $B$–$H$ 曲线插值，而不是常数 $\mu_r$。② 气隙处磁力线会向外「鼓出」（边缘效应），有效气隙面积大于铁芯截面积，精确计算要乘边缘因子。③ 磁动势 $NI$ 是安培匝数，**不是**磁场强度——两者单位相同（安培）但物理不同，混写是磁路分析最常见的错误。

**顺磁质与抗磁质的能量对比**：顺磁质（$\chi_m > 0$）放入磁场中，磁化方向与外场一致，磁能 $W = -\frac{1}{2}\mu_0\chi_m H^2V$ 为负——**物质进入磁场是「降能」，被拉进高场区**；抗磁质（$\chi_m < 0$）磁化反向，能量为正，被推出高场区。这个「能量最小化」判据统一解释了顺磁/抗磁物质的受力方向，也是磁悬浮（超导体完全抗磁）的定性来源。**「物质往能量低处走」的普适原则，在磁性上同样成立。**

## 7 小结

- **磁场能量** $W_m = \frac{1}{2}LI^2 = \frac{1}{2}\int\mathbf{A}\cdot\mathbf{J}\,\mathrm{d}V = \frac{1}{2}\int\mathbf{B}\cdot\mathbf{H}\,\mathrm{d}V$，真空中 $w_m = B^2/2\mu_0$。
- **磁化强度** $\mathbf{M}$；$\mathbf{H} = \mathbf{B}/\mu_0 - \mathbf{M}$，$\nabla\times\mathbf{H} = \mathbf{J}_f$。
- 磁介质三类：**抗磁**（$\chi_m<0$）、**顺磁**（$\chi_m>0$ 小）、**铁磁**（非线性、磁滞回线）。
- 磁滞损耗正比回线面积；非线性介质能量 $w_m = \int\mathbf{H}\cdot\mathrm{d}\mathbf{B}$。
- 边界条件：$B_n$ 连续、$H_t$ 连续（无面电流时）。

至此静磁篇收束。下一节我们迎来最波澜壮阔的一章：磁场与电场开始互相激发、脱离源独立传播——**时谐电磁波**。
