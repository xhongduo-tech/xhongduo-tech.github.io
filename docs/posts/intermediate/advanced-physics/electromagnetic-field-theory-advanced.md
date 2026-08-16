---
title: 电磁场理论深化（麦克斯韦方程组、势、边界值问题）
date: 2026-08-07
---

# 电磁场理论深化（麦克斯韦方程组、势、边界值问题）

<div class="epigraph">
<p>电磁场，就是包含并环绕着带电或带磁物体的那一部分空间。</p>
<footer>—— 詹姆斯 · 克拉克 · 麦克斯韦（James Clerk Maxwell），《电磁场的动力学理论》，1865</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 郭硕鸿《电动力学》第一～三章 ｜ 2026-08-07</p>
</div>

## 为什么从电磁场理论深化开始

在「第七篇 四大力学入门 · 第二十一章 电动力学」里，我们已经见过麦克斯韦方程组的积分形式与微分形式、以及电磁场的边值关系。本专题进入「第1篇」深挖课程，这一节要把电磁场理论的**骨架**一次搭完整：为什么四个方程是「一个整体」、电磁势和规范自由度是怎么回事、静电场边值问题有哪些标准解法。这套内容是电磁学的理论化完成形态——它把「第四篇 电磁学」里零散的法拉第定律、安培环路定理、高斯定理组织成一个自洽的动力学系统，也是后续电磁波、辐射与波导几篇的出发点。

## 1 麦克斯韦方程组：电磁学的统一图景

麦克斯韦方程组（微分形式）用四条方程把电磁场的一切局部行为收紧为四条微分约束：

$$
\nabla\cdot\boldsymbol{E} = \frac{\rho}{\varepsilon_0}, \qquad
\nabla\times\boldsymbol{E} = -\frac{\partial\boldsymbol{B}}{\partial t}, \qquad
\nabla\cdot\boldsymbol{B} = 0, \qquad
\nabla\times\boldsymbol{B} = \mu_0\boldsymbol{J} + \mu_0\varepsilon_0\frac{\partial\boldsymbol{E}}{\partial t}
$$

**重点：四条方程不是四个独立的经验定律，而是一套自洽的动力学系统——电荷守恒律 $\nabla\cdot\boldsymbol{J} + \partial\rho/\partial t = 0$ 可以从它们推导出来，而不是外加的假设。** 其中最关键的一步是麦克斯韦 1861 年在电磁感应、安培环路定律基础上补上的**位移电流项** $\mu_0\varepsilon_0\,\partial\boldsymbol{E}/\partial t$：没有它，方程组不自洽；有了它，电磁波的存在从数学上被强制导出。<span class="marginnote"><strong>历史时刻</strong>：麦克斯韦 1865 年的《电磁场的动力学理论》由 20 个方程组成，经过赫兹、亥维赛的矢量改写，才浓缩成今天这四条。1879 年麦克斯韦去世，直到 1888 年赫兹才用实验证实电磁波——「电磁场」作为独立的物理实在，在理论提出二十三年后才得到验证。</span>

方程之间的成对结构值得强调：$\nabla\cdot\boldsymbol{B}=0$ 保证磁感应强度可写成旋度（无源场），$\nabla\times\boldsymbol{E} = -\partial\boldsymbol{B}/\partial t$ 使电场在时变磁场下不再无旋——这正是「磁生电、电生磁」互为因果、自洽传播的数学体现。

## 2 电磁势与规范变换

利用「旋度的散度为零、梯度的旋度为零」两个恒等式，可以把方程组降阶。**电磁势（potentials）**：由于 $\nabla\cdot\boldsymbol{B} = 0$，存在矢量势 $\boldsymbol{A}$ 使 $\boldsymbol{B} = \nabla\times\boldsymbol{A}$；代入法拉第定律得 $\nabla\times(\boldsymbol{E} + \partial\boldsymbol{A}/\partial t) = 0$，故存在标量势 $\varphi$ 使

$$
\boldsymbol{E} = -\nabla\varphi - \frac{\partial\boldsymbol{A}}{\partial t}, \qquad \boldsymbol{B} = \nabla\times\boldsymbol{A}
$$

**关键认识：电磁势不是唯一的——它们定义了一个规范自由度。** 对任意标量场 $\chi$，作**规范变换（gauge transformation）**

$$
\boldsymbol{A}' = \boldsymbol{A} + \nabla\chi, \qquad \varphi' = \varphi - \frac{\partial\chi}{\partial t}
$$

电磁场 $\boldsymbol{E}$、$\boldsymbol{B}$ 完全不变。物理量必须是规范不变的。<span class="marginnote"><strong>为什么要选规范</strong>：规范自由度就像坐标系自由度——不改变物理，只改变计算繁简。库仑规范 $\nabla\cdot\boldsymbol{A}=0$ 适合静电场与近静态问题；洛伦兹规范 $\nabla\cdot\boldsymbol{A} + \frac{1}{c^2}\frac{\partial\varphi}{\partial t}=0$ 让 $\varphi$、$\boldsymbol{A}$ 分别满足波动方程，是处理辐射与相对论问题的首选。见《电磁场的矢势、标势与规范变换》。</span>

在洛伦兹规范下，势满足达朗贝尔方程 $\Box\varphi = -\rho/\varepsilon_0$、$\Box\boldsymbol{A} = -\mu_0\boldsymbol{J}$（$\Box = \nabla^2 - \frac{1}{c^2}\frac{\partial^2}{\partial t^2}$）——四个耦合方程解耦成四个独立的波动方程，这是下一节电磁波与辐射的出发点。

## 3 边值问题：唯一性定理与镜像法

求解静电场，最常见的情形是「给边界条件，求场分布」——这就是**边值问题（boundary-value problem）**。它比「由电荷直接算场」难得多，但有一套漂亮的解。

**唯一性定理（uniqueness theorem）**：给定导体表面电势（第一类边界）或表面电荷（第二类边界）后，区域内的电场唯一确定。<span class="marginnote"><strong>唯一性定理为什么是解题武器</strong>：只要「蒙」出一个满足边界条件的解，它一定就是真解——于是可以大胆猜测解的形式、用镜像电荷等技巧试探，而不必担心猜错。这是镜像法成立的合法性依据，也是数值方法（有限元）收敛性的理论保证。见《静电场的唯一性定理与镜像法》。</span>

**镜像法（method of images）**：用区域外的「虚设电荷」替代导体边界对内部场的影响。一个经典例子：离无限大接地导体平面距离 $d$ 处有一点电荷 $q$，它在导体内部的场，等价于在导体平面另一侧对称位置放一个 $-q$ 的虚电荷所产生的场。由此可算导体表面感应电荷密度

$$
\sigma(x,y) = -\frac{q d}{2\pi (x^2 + y^2 + d^2)^{3/2}}
$$

**重点：镜像法把「导体边界的无穷多感应电荷」替换成「区域外一个（或几个）虚电荷」——边界条件被精确满足，而计算量骤降。** 导体球的镜像、两平行导线的镜像，是同一套思想的标准推演。<span class="marginnote"><strong>数值算例（接地导体球）</strong>：点电荷 $q$ 距接地导体球（半径 $R$）球心 $a$，虚电荷 $-q'$ 放在距球心 $b = R^2/a$ 处，$q' = (R/a)q$——位置与电量由「球面电势为零」唯一确定。这一结果在静电除尘、尖端放电等工程问题里反复出现。</span>

## 4 公式解析：分离变量法求边值问题

当边界与坐标面重合时，**分离变量法（separation of variables）**是标准解法。以「半径为 $R$ 的接地导体球，置于均匀外电场 $\boldsymbol{E}_0$ 中」为例：边界上 $\varphi|_{r=R} = 0$，无穷远 $\varphi \to -E_0 r\cos\theta$。

$$
\varphi(r,\theta) = \sum_{l=0}^{\infty}\left(A_l r^l + \frac{B_l}{r^{l+1}}\right)P_l(\cos\theta)
$$

- **第一步，选坐标系**：球坐标下问题最自然，拉普拉斯方程 $\nabla^2\varphi = 0$ 的通解写成上式——径向部分 $r^l$ 与 $r^{-(l+1)}$ 两项，角向部分为勒让德多项式 $P_l(\cos\theta)$。
- **第二步，用无穷远条件**：$r\to\infty$ 时 $\varphi \to -E_0 r\cos\theta = -E_0 r P_1(\cos\theta)$，故只有 $l=1$ 项，$A_1 = -E_0$，其余 $A_l = 0$。
- **第三步，用球面条件**：$\varphi(R,\theta) = -E_0 R\cos\theta + \frac{B_1}{R^2}\cos\theta = 0$，得 $B_1 = E_0 R^3$。
- **第四步，读结果**：$\varphi = -E_0 r\cos\theta + E_0 R^3\frac{\cos\theta}{r^2}$。第一项是外场，第二项是球上感生偶极子（偶极矩 $p = 4\pi\varepsilon_0 E_0 R^3$）的势——**均匀场中的导体球等效于一个感生电偶极子**。<span class="marginnote"><strong>为什么勒让德多项式登场</strong>：分离变量要求「角向函数乘径向函数」在拉普拉斯方程下解耦，唯一自治的角向函数族就是勒让德多项式——它们正交、完备，任何角分布都能展开成它们的级数。这一展开在量子力学（球谐函数）、地球物理（位场反演）里都是基本语言。</span>

## 5 电磁场的能量与动量：场本身就是「东西」

麦克斯韦方程组的深刻之处在于：电场与磁场携带能量与动量，场不是力在空中的「投影」，而是独立的物理实在。能量密度与能流密度：

$$
u = \frac{1}{2}\varepsilon_0 E^2 + \frac{1}{2\mu_0}B^2, \qquad
\boldsymbol{S} = \frac{1}{\mu_0}\boldsymbol{E}\times\boldsymbol{B} \quad(\text{坡印廷矢量})
$$

能量守恒由**坡印廷定理**给出：$\frac{\partial u}{\partial t} + \nabla\cdot\boldsymbol{S} = -\boldsymbol{J}\cdot\boldsymbol{E}$——左边是「场能量减少 + 流出」，右边是「对带电体做功」。「场有能量、有动量、能传播」这个观念，是爱因斯坦质能关系与光的辐射压实验的思想前提。<span class="marginnote"><strong>数值算例（太阳辐射压）</strong>：太阳在地球处的能流密度约 $1.36\times10^3\ \mathrm{W/m^2}$（太阳常数），对应的辐射压约 $4.5\times10^{-6}\ \mathrm{Pa}$——虽然微小，却是「光帆」航天器推进的唯一动力来源。场动量概念的工程化身，见《电磁场的能量、动量与坡印廷矢量》。</span>

## 6 数值算例：从解析到数值的现代电磁计算

边值问题的解析解法（镜像法、分离变量）只有在几何足够简单时才可行。真实工程中的电磁场——微带线、天线罩、芯片互连、雷达散射——几乎都必须数值求解。三大主流方法在此做一次对照：

| 方法 | 基本思想 | 典型应用 | 与本节的关系 |
| --- | --- | --- | --- |
| 有限差分时域（FDTD） | 对麦克斯韦方程组在时空网格上直接离散 | 天线、超材料、电磁兼容 | 直接离散四条方程 |
| 有限元法（FEM） | 对拉普拉斯/波动方程的变分形式离散 | 静电场、涡流、波导本征模 | 分离变量法的「一般化」 |
| 矩量法（MoM） | 把积分方程化为线性方程组 | 细导线天线、散射 | 格林函数 + 唯一性定理 |

**重点：解析解法给出「理解」，数值解法给出「答案」——两者靠唯一性定理与收敛性分析连接。** 电磁计算的迅猛发展（从 1960 年代的矩量法论文到今天 AI 加速求解器）并没有推翻本节任何结论，反而让「边值问题」成为设计现代电子系统的常规操作。<span class="marginnote"><strong>与「从极限到大模型」主线的连接</strong>：深度学习里求解偏微分方程的物理信息神经网络（PINN），正是把边值问题改写为「损失函数 = 方程残差 + 边界残差」的优化问题；而天线阵的波束赋形、芯片互连的信号完整性分析，都直接调用本节镜像法与分离变量法的经典解。</span>

## 7 术语速查表

| 术语 | 公式 | 要点 |
| --- | --- | --- |
| 麦克斯韦方程组 | 四条微分方程 | 自洽 + 推出电荷守恒 |
| 电磁势 | $\boldsymbol{B}=\nabla\times\boldsymbol{A}$，$\boldsymbol{E}=-\nabla\varphi-\partial\boldsymbol{A}/\partial t$ | 非唯一 |
| 规范变换 | $\boldsymbol{A}\to\boldsymbol{A}+\nabla\chi$ | $\boldsymbol{E}$、$\boldsymbol{B}$ 不变 |
| 洛伦兹规范 | $\nabla\cdot\boldsymbol{A}+\frac{1}{c^2}\partial\varphi/\partial t=0$ | 势满足波动方程 |
| 唯一性定理 | 边界定场 | 镜像法的合法性 |
| 镜像法 | 虚电荷替代边界 | 感应电荷分布直接算出 |
| 分离变量法 | $\sum(A_l r^l + B_l r^{-l-1})P_l(\cos\theta)$ | 勒让德多项式展开 |
| 坡印廷矢量 | $\boldsymbol{S}=\boldsymbol{E}\times\boldsymbol{B}/\mu_0$ | 能量流动的方向 |

## 8 小结

- **麦克斯韦方程组**是电磁学的动力学核心，位移电流的加入使其自洽，并预言电磁波。
- **电磁势** $\varphi$、$\boldsymbol{A}$ 存在但不唯一，**规范变换**不改变物理；洛伦兹规范使势满足达朗贝尔方程。
- **边值问题**的两大解法：**镜像法**（虚电荷替代边界）与**分离变量法**（坐标面正交时），唯一性定理为两者背书。
- **电磁场携带能量与动量**：能量密度、坡印廷矢量、辐射压，使「场」成为独立物理实在。
- 本节的势、规范与边值语言，是《电磁波与辐射》《平面电磁波传播》等后续篇章的共同工具箱。

在下一节，我们将让麦克斯韦方程组「动起来」——研究电磁波如何产生、如何携带能量与动量、如何被束缚进波导——**电磁波与辐射**。
