---
title: 恒定磁场与电感
date: 2026-08-07
---

# 恒定磁场与电感

<div class="epigraph">
<p>实在概念的这次变革，是自牛顿时代以来物理学所经历的最为深刻和富有成果的一次。</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein）</footer>
</div>

<div class="article-byline">
<p>第六级 · 电磁场与电磁波（工程电磁场） ｜ David K. Cheng《Field and Wave Electromagnetics》第2版 §6-1～§6-7 ｜ 2026-08-07</p>
</div>

## 为什么从恒定磁场开始

1820 年奥斯特偶然发现，通电导线旁的磁针会偏转——电流不仅发热，还会"制造"磁。<span class="marginnote">此前的磁学只研究天然磁石；奥斯特的发现第一次把磁与电荷的运动联系起来，催生了安培的"电流生磁"、法拉第的"磁生电"，最终在麦克斯韦手里合成为完整的电磁理论。本专题第 4 篇就会见到那个合流点。</span>静电场有"电荷"这个源，恒定磁场的源是**恒定电流**。二者最根本的差别是：电场线从正电荷发出、止于负电荷；而磁力线**永远闭合**——没有"磁荷"（磁单极子）。本篇对标 David K. Cheng《Field and Wave Electromagnetics》第 6 章，讲恒定磁场的计算工具，以及它最重要的工程产物——**电感**。

## 1 从奥斯特到毕奥-萨伐尔定律

给定电流分布求磁场，基本公式是**毕奥-萨伐尔定律（Biot-Savart law）**：电流元 $I\,\mathrm{d}\mathbf{l}$ 在场点产生的磁场强度

$$\mathrm{d}\mathbf{H} = \frac{I\,\mathrm{d}\mathbf{l}\times\hat{\mathbf{R}}}{4\pi R^2}$$

其中 $\mathbf{R}$ 是电流元到场点的位置矢量。**磁场强度（magnetic field intensity）** $\mathbf{H}$ 单位 A/m，磁感应强度 $\mathbf{B} = \mu\mathbf{H}$（$\mu$ 是磁导率，真空 $\mu_0 = 4\pi\times10^{-7}\ \mathrm{H/m}$），单位特斯拉（T）。<span class="marginnote">$\mathrm{d}\mathbf{l}\times\hat{\mathbf{R}}$ 里的叉积意味着：电流元正上方的场为零、侧面场最强——磁场围绕电流线"旋转"。这一几何事实决定了几乎所有磁场的形状。</span>

两个必须背下来的经典结果：

- **无限长直导线的磁场**：距导线 $\rho$ 处 $H_\phi = I/(2\pi\rho)$，场线是以导线为轴的同心圆。
- **圆形电流环轴线上的磁场**：半径为 $a$ 的圆环，轴线上距环心 $z$ 处 $H_z = Ia^2/\bigl[2(a^2+z^2)^{3/2}\bigr]$。

**辨析｜易错点：** 毕奥-萨伐尔定律求的是"电流元"的贡献，方向由叉积右手定则决定；实际电流线是闭合的，积分时不能丢方向分量。常见错误是把 $\hat{\mathbf{R}}$ 方向写反、或者对非闭合"假想电流段"积分——稳恒电流必须构成回路才有物理意义。

## 2 安培环路定律与它的用法

与高斯定律对称，磁场有**安培环路定律（Ampère's circuital law）**：磁场强度沿闭合回路的环量等于回路所包围的总电流。

$$\oint_C \mathbf{H}\cdot\mathrm{d}\mathbf{l} = I_{\mathrm{enc}}, \qquad \nabla\times\mathbf{H} = \mathbf{J}$$

同样，只有**高度对称**的电流分布（无限长直导线、螺线管、环形线圈、同轴线）才能用积分形式直接解出 $\mathbf{H}$。<span class="marginnote">注意高斯定律的源是电荷、安培定律的源是电流——一个是散度方程，一个是旋度方程。第 1 篇提醒过"哪个带散、哪个带旋"，这里就是第一次正面相遇。</span>

螺线管与环形线圈是最常用的结果：无限长密绕螺线管内部 $B = \mu_0 nI$（$n$ 为单位长度匝数），外部近似为零；环形线圈（环形磁芯上绕 $N$ 匝，通电流 $I$）内 $B = \mu NI/(2\pi\rho)$。

**辨析｜易错点：** 安培环路定律只能处理"场沿回路处处可求"的对称情形；对任意形状线圈，$\oint\mathbf{H}\cdot\mathrm{d}\mathbf{l}=I$ 仍然成立，但积分左侧无从先化简，无法直接给出 $\mathbf{H}$。把"定律成立"与"能用它求出场"混为一谈，是静态场学习中最常见的误区。

把静电与静磁并排看，整个静态场就成了一面镜子：

| 量 | 静电场 | 恒定磁场 |
| --- | --- | --- |
| 源 | 电荷（标量） | 电流（矢量） |
| 场 | $\mathbf{E}$、$\mathbf{D}$ | $\mathbf{H}$、$\mathbf{B}$ |
| 散度方程 | $\nabla\cdot\mathbf{D}=\rho$（有源） | $\nabla\cdot\mathbf{B}=0$（无源） |
| 旋度方程 | $\nabla\times\mathbf{E}=0$（无旋） | $\nabla\times\mathbf{H}=\mathbf{J}$（有旋） |
| 位函数 | 标量位 $\varphi$ | 矢量位 $\mathbf{A}$ |
| 储能 | $w_e=\frac{1}{2}\varepsilon E^2$ | $w_m=\frac{1}{2}\mu H^2$ |

这张表是理解 Maxwell 方程组的最佳跳板：时变后，旋度方程两边都要补上时间变化项，对称被打破又重新缝合。

## 3 磁矢位：无源场的势

因为 $\nabla\cdot\mathbf{B} = 0$（磁力线闭合、无磁单极），$\mathbf{B}$ 可以写成另一个场的旋度：

$$\mathbf{B} = \nabla\times\mathbf{A}$$

$\mathbf{A}$ 叫**磁矢位（magnetic vector potential）**，单位 Wb/m（韦伯每米）。<span class="marginnote">这与静电场 $\mathbf{E}=-\nabla\varphi$ 完全平行：无旋场有标量位，无源场有矢量位。"位"的引入把场方程降阶——求标量 $\varphi$ 或矢量 $\mathbf{A}$ 比直接解矢量场 $\mathbf{B}$ 简单，这在数值方法里尤其重要。</span>

$\mathbf{A}$ 不是唯一的：给 $\mathbf{A}$ 加上任意一个无旋场 $\nabla\chi$，旋度不变。为消除这种任意性，工程上常取**库仑规范** $\nabla\cdot\mathbf{A}=0$。在库仑规范下，矢量泊松方程 $\nabla^2\mathbf{A} = -\mu\mathbf{J}$ 把磁场的求解变成三个标量泊松方程——与静电场的标量泊松方程完全同构。

**辨析｜易错点：** $\mathbf{A}$ 的方向由电流方向决定（平行于 $\mathbf{J}$ 方向），$\mathbf{B}$ 却垂直于它。求 $\mathbf{B}$ 时容易直接把 $\mathbf{A}$ 当作 $\mathbf{B}$，忘了还要"取旋度"、而且旋度会引入垂直于 $\mathbf{A}$ 的分量。

## 4 磁通、磁链与电感

穿过曲面的磁场总量叫**磁通（magnetic flux）**：$\Phi = \int_S \mathbf{B}\cdot\mathrm{d}\mathbf{S}$，单位韦伯（Wb）。多匝线圈把每匝穿过的磁通累加，得**磁链（flux linkage）** $\lambda = N\Phi$。

**电感（inductance）** 定义为磁链与电流之比 $L = \lambda/I$，单位亨利（H）。它和电容一样，只由线圈的几何与介质决定，是"这个结构储存磁场能力"的固有属性。<span class="marginnote">电感与电容在第六级《电路分析基础》里是三大无源元件之二：电感 $L$ 的电压电流关系 $v = L\,\mathrm{d}i/\mathrm{d}t$，正是法拉第电磁感应定律的电路翻译。这里学的是从场论算出 $L$，电路课里则把 $L$ 当已知参数。</span>

两个线圈之间还有**互感（mutual inductance）** $M = \lambda_{12}/I_2$：线圈 2 的电流在线圈 1 中产生的磁链与 $I_2$ 之比。$M_{12}=M_{21}$，这是互易性（reciprocity）的体现。<span class="marginnote">互感是变压器的心脏：原边电流变化，通过 $M$ 在副边感应出电动势，能量就这样"穿过"磁场耦合到另一回路，没有电的直接连接。电力系统、无线充电、开关电源全都建立在这个 $M$ 上。</span>

## 5 磁场能量与磁路

恒定磁场存储能量，能量密度

$$w_m = \frac{1}{2}\mathbf{B}\cdot\mathbf{H} = \frac{1}{2}\mu H^2$$

总磁能 $W_m = \int_V \frac{1}{2}\mathbf{B}\cdot\mathbf{H}\,\mathrm{d}v$，与电感的储能公式 $W_m = \frac{1}{2}LI^2$ 等价。**两种算 $L$ 的路子由此殊途同归**：先求 $\mathbf{B}$ 算磁链，或先算能量再反解 $L$。

对铁磁材料构成的磁路（变压器、电机），工程上常用**磁路模型**：把磁通当作"电流"，磁动势 $NI$ 当作"电压"，磁阻 $\mathcal{R} = l/(\mu S)$ 当作"电阻"，安培环路定律就写成 $\Phi = NI/\mathcal{R}$——与欧姆定律 $I = V/R$ 完全同构。<span class="marginnote">磁路的静电类比（$B\leftrightarrow J$、$\mu\leftrightarrow\sigma$、$\Lambda=1/\mathcal{R}\leftrightarrow G$）与上一节的"静电类比"是同一条思想：不同物理、同一套拓扑方程。电机学、变压器设计里到处是这张类比表。</span>

## 6 磁介质的边界条件

磁介质分界面两侧，场量满足：

磁感应强度的法向分量连续：$\hat{\mathbf{n}}\cdot(\mathbf{B}_1-\mathbf{B}_2)=0$；
磁场强度的切向分量差等于自由面电流密度：$\hat{\mathbf{n}}\times(\mathbf{H}_1-\mathbf{H}_2)=\mathbf{K}_s$。

前者来自"无磁单极"，后者来自安培环路定律。<span class="marginnote">铁磁材料（$\mu\gg\mu_0$）内部磁力线几乎平行于界面"钻"进材料——这就是为什么变压器铁芯能把磁通"约束"在芯内，形成低磁阻回路。</span>

**辨析｜易错点：** 无自由面电流时，$\mathbf{H}$ 切向连续而 $\mathbf{B}$ 法向连续；$\mathbf{B}$ 的切向与 $\mathbf{H}$ 的法向都会"跳变"。与静电场完全对偶——那边是 $\mathbf{E}$ 切向连续、$\mathbf{D}$ 法向连续；这边是 $\mathbf{H}$ 切向连续、$\mathbf{B}$ 法向连续。把两套边界条件混记，是考试和工程里最普遍的失误，建议对照着背。

## 7 公式解析：同轴线的单位长度电感

用对称积分求一个工程界天天使用的量——**同轴线单位长度电感**。内导体半径 $a$、外导体半径 $b$（视为薄壳），内导体通电流 $I$，外导体回流 $-I$。

- **第一步，用安培环路定律求场**：在 $a\lt \rho<b$ 的介质区，回路包围电流 $I$，由对称性 $H_\phi\cdot 2\pi\rho = I$，得 $H_\phi = I/(2\pi\rho)$，$B_\phi = \mu I/(2\pi\rho)$。
- **第二步，算磁通**：取长为 $l$ 的一段，穿过单位长纵向截面的磁通 $\Phi' = \int_a^b \frac{\mu I}{2\pi\rho}\,\mathrm{d}\rho = \frac{\mu I}{2\pi}\ln\frac{b}{a}$。
- **第三步，定义比**：单位长度电感 $L' = \Phi'/I$，于是

$$L' = \frac{\mu}{2\pi}\ln\frac{b}{a}\ \ \mathrm{H/m}$$

**三个要点**：电感只由几何（$a,b$）与介质（$\mu$）决定；与电流大小无关；对数是同轴几何的签名——就像点电荷的 $1/R$ 与线电荷的 $\ln$ 一样，场的空间维数决定了位函数的形状。<span class="marginnote">这个结果直接支撑同轴电缆的特性阻抗计算（第 7 篇传输线理论里 $Z_0=\sqrt{L'/C'}$ 中的 $L'$ 就是它），所以今天这一步是为导行电磁波铺的路。</span>

## 8 小结

- 恒定磁场的源是**恒定电流**；$\nabla\cdot\mathbf{B}=0$，磁力线永远闭合，无磁单极。
- **毕奥-萨伐尔定律**是磁场的积分公式，**安培环路定律** $\oint\mathbf{H}\cdot\mathrm{d}\mathbf{l}=I_{\mathrm{enc}}$ 在对称分布下直接解场。
- 无源场可写**磁矢位** $\mathbf{B}=\nabla\times\mathbf{A}$，库仑规范下满足矢量泊松方程。
- **电感** $L=\lambda/I$ 只由几何与介质决定；磁能 $w_m=\frac{1}{2}\mu H^2$ 与 $W_m=\frac{1}{2}LI^2$ 互为表里。
- **磁路模型**把磁通比作电流、磁动势比作电压，工程上简化铁芯设计。
- **边界条件**：$\mathbf{B}$ 法向连续、$\mathbf{H}$