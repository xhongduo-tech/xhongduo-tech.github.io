---
title: 矢量分析与场论基础
date: 2026-08-07
---

# 矢量分析与场论基础

<div class="epigraph">
<p>宇宙这部伟大的书，是用数学语言写成的，其字母是三角形、圆以及其他几何图形。</p>
<footer>—— 伽利略（Galileo Galilei）</footer>
</div>

<div class="article-byline">
<p>第六级 · 电磁场与电磁波（工程电磁场） ｜ David K. Cheng《Field and Wave Electromagnetics》第2版 §2-1～§2-7 ｜ 2026-08-07</p>
</div>

## 为什么从矢量分析开始

电磁场与电磁波这门课的全部对象——电场 $\mathbf{E}$、磁感应强度 $\mathbf{B}$、电流密度 $\mathbf{J}$——都是**随空间位置变化、又带方向的量**，数学上叫**矢量场（vector field）**。<span class="marginnote">与标量场相对：温度场、电势场只给一个数 $\varphi(x,y,z)$；矢量场在每个点给一个矢量。两者的处理工具不同，这就是本章要建立的"场论"。</span>要描述电场怎么"流"、磁场怎么"旋"，光有坐标不够，还需要三件武器：**梯度**、**散度**、**旋度**。本专题对标 David K. Cheng《Field and Wave Electromagnetics》（辅以谢处方、饶克谨《电磁场与电磁波》），就从这三件武器讲起——它们是 Maxwell 方程组逐项的几何含义，也是后面静电场、恒磁场、电磁波一切公式的语法。

## 1 矢量与矢量场：电磁学的第一语言

**矢量（vector）**：既有大小又有方向的量，用黑体或箭头记号 $\mathbf{A}$ 表示，大小为 $A = |\mathbf{A}|$。在直角坐标系里写成分量形式

$$\mathbf{A} = A_x \hat{\mathbf{x}} + A_y \hat{\mathbf{y}} + A_z \hat{\mathbf{z}}$$

其中 $\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}}$ 是三个方向的**单位矢量**。<span class="marginnote">电场强度 $\mathbf{E}$ 的单位是伏/米（V/m），磁场强度 $\mathbf{H}$ 的单位是安/米（A/m），磁感应强度 $\mathbf{B}$ 的单位是特斯拉（T）。这些量后面逐章出场，今天先把"带方向的量"这个直觉立住。</span>

两个最基本的矢量运算贯穿全课：

**标量积（点积）**：$\mathbf{A} \cdot \mathbf{B} = AB\cos\theta$，结果为标量，度量两矢量的"同向程度"。直角坐标下 $\mathbf{A} \cdot \mathbf{B} = A_xB_x + A_yB_y + A_zB_z$。
**矢量积（叉积）**：$\mathbf{A} \times \mathbf{B} = AB\sin\theta\,\hat{\mathbf{n}}$，结果仍是矢量，大小 $AB\sin\theta$ 是平行四边形的面积，方向由**右手定则**决定，$\hat{\mathbf{n}}$ 为垂直于 $\mathbf{A},\mathbf{B}$ 所在平面的单位矢量。<span class="marginnote">叉积直接给出面积元的方向：后面求磁通 $\Phi = \int_S \mathbf{B}\cdot\mathrm{d}\mathbf{S}$ 时，面元矢量 $\mathrm{d}\mathbf{S} = \mathrm{d}\mathbf{S}\,\hat{\mathbf{n}}$ 的方向就是靠它定义的。</span>

空间里还有一个"常量级"的矢量——**位置矢量（position vector）**：从原点指向场点 $P(x,y,z)$ 的矢量 $\mathbf{R} = x\hat{\mathbf{x}} + y\hat{\mathbf{y}} + z\hat{\mathbf{z}}$。两点之间的相对位置 $\mathbf{R} - \mathbf{R}'$ 贯穿全场理论：库仑定律里，源电荷在 $\mathbf{R}'$ 处，场点电荷在 $\mathbf{R}$ 处，两电荷间的力方向就由 $\mathbf{R} - \mathbf{R}'$ 决定。三个矢量 $\mathbf{A},\mathbf{B},\mathbf{C}$ 还有两个高阶组合：**标量三重积** $\mathbf{A}\cdot(\mathbf{B}\times\mathbf{C})$（结果是平行六面体体积）与**矢量三重积** $\mathbf{A}\times(\mathbf{B}\times\mathbf{C})$（后面推导电动力学的双叉积恒等式 $\nabla\times(\nabla\times\mathbf{E})$ 时会反复出现）。

**辨析｜易错点：** 点积与叉积的一个高频陷阱是"点积结果是标量、叉积结果是矢量"，有人把 $\mathbf{A}\cdot\mathbf{B}=|\mathbf{A}||\mathbf{B}|\cos\theta$ 与 $|\mathbf{A}\times\mathbf{B}|=|\mathbf{A}||\mathbf{B}|\sin\theta$ 的大小混为一谈。记住一句话：**点积回答"有多同向"，叉积回答"有多垂直且朝向哪边"**。

## 2 三种正交坐标系

处理不同形状的问题要选不同坐标系。工程电磁场主要用三种**正交坐标系**：直角坐标、圆柱坐标、球坐标。

| 坐标系 | 变量 | 体积元 $\mathrm{d}v$ | 适用场景 |
| --- | --- | --- | --- |
| 直角 | $x, y, z$ | $\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z$ | 平面波、矩形波导、平行板 |
| 圆柱 | $\rho, \phi, z$ | $\rho\,\mathrm{d}\rho\,\mathrm{d}\phi\,\mathrm{d}z$ | 同轴线、长直导线、圆波导 |
| 球 | $r, \theta, \phi$ | $r^2\sin\theta\,\mathrm{d}r\,\mathrm{d}\theta\,\mathrm{d}\phi$ | 点电荷、偶极子、天线辐射 |

选择坐标系的原则不是"哪个好看"，而是**让等坐标面与问题边界重合**。点电荷的等势面是球面，用球坐标时边界条件自动简化；长直导线的场绕轴旋转对称，用圆柱坐标积分会少一层负担。<span class="marginnote">这个"坐标系贴着边界走"的直觉，到第 10 篇《准静态场与数值方法》仍有用：网格剖分时也是让单元贴合几何边界。</span>

三个坐标系之间可以换算，例如球坐标到直角坐标 $x = r\sin\theta\cos\phi,\ y = r\sin\theta\sin\phi,\ z = r\cos\theta$。<strong>关键不是背公式，而是记住面积元与体积元里多出的因子（圆柱里那个 $\rho$、球坐标里的 $r^2\sin\theta$）来自坐标线弯曲造成的"弧长伸长"</strong>。

还有一个隐蔽的坑：在圆柱与球坐标里，单位矢量不是常矢量。直角坐标的单位矢量 $\hat{\mathbf{x}}$ 到处朝同一个方向，而圆柱坐标的 $\hat{\boldsymbol{\rho}}$ 随 $\phi$ 转动、球坐标的 $\hat{\mathbf{r}}$ 随 $\theta,\phi$ 转动。因此，对坐标单位矢量求导时会出现交叉项——这正是后面推导 $\nabla\times\mathbf{B}$ 在柱坐标、球坐标表达式时要格外小心的原因。<span class="marginnote">许多同学背了公式却不理解"圆柱坐标里 $\partial\hat{\boldsymbol{\rho}}/\partial\phi = \hat{\boldsymbol{\phi}}$"这种关系，结果在运算里丢掉方向项。建议把三个坐标系下 $\nabla$ 的表达式抄一遍对照，就明白它们是从同一套矢量恒等式来的。</span>

## 3 标量场的梯度

对标量场 $\varphi(x,y,z)$，三个偏导拼成一个矢量，叫**梯度（gradient）**：

$$\nabla\varphi = \frac{\partial \varphi}{\partial x}\hat{\mathbf{x}} + \frac{\partial \varphi}{\partial y}\hat{\mathbf{y}} + \frac{\partial \varphi}{\partial z}\hat{\mathbf{z}}$$

$\nabla$（读作 nabla）是矢量微分算符。梯度的物理意义：**指向 $\varphi$ 增大最快的方向，大小等于该方向的方向导数**。等高线/等势面密的地方，梯度就大。<span class="marginnote">电势的梯度 $-\nabla\varphi$ 就是电场：电荷在电势场里"顺坡下滑"，力的方向永远垂直于等势面。梯度的这个几何直觉，是静电场的势函数理论的出发点。</span>

**辨析｜易错点：** 梯度作用在标量上给出矢量；反过来，对矢量场做 $\nabla\cdot$（散度）得到标量，做 $\nabla\times$（旋度）得到矢量。$\nabla$ 三种用法结果类型各不相同，初学最常见的错就是"梯度算出来的量方向忘了写"。

## 4 矢量场的散度与高斯散度定理

对矢量场 $\mathbf{A}$，定义**散度（divergence）**

$$\nabla\cdot\mathbf{A} = \frac{\partial A_x}{\partial x} + \frac{\partial A_y}{\partial y} + \frac{\partial A_z}{\partial z}$$

散度度量"场在这一点的净流出程度"。把场想象成水流：散度为正的点是**源**（水从那里涌出），为负的点是**汇**（水被吸走），为零说明"来多少走多少"。流过闭曲面的总通量与曲面内源的强弱，由**高斯散度定理**联系起来：

$$\oint_S \mathbf{A}\cdot\mathrm{d}\mathbf{S} = \int_V (\nabla\cdot\mathbf{A})\,\mathrm{d}v$$

它把一个面积分换成一个体积分。<span class="marginnote">静电学里 $\nabla\cdot\mathbf{E} = \rho/\varepsilon_0$（Maxwell 第一方程）说的正是：电荷是电场的源，电场从正电荷发出、终止于负电荷。高斯定理是"通量 = 源总量"这句话的数学形式。</span>

## 5 矢量场的旋度与斯托克斯定理

对矢量场 $\mathbf{A}$，定义**旋度（curl）**

$$\nabla\times\mathbf{A} = \begin{vmatrix} \hat{\mathbf{x}} & \hat{\mathbf{y}} & \hat{\mathbf{z}} \\ \partial_x & \partial_y & \partial_z \\ A_x & A_y & A_z \end{vmatrix}$$

行列式展开即是各分量的偏导组合。旋度度量场的"旋转程度"：把一个桨轮放进流场，桨轮转得越猛，该点旋度越大，旋度方向沿转轴（右手定则）。绕闭合回路的**环量（circulation）** 与面内旋度由**斯托克斯定理**联系：

$$\oint_C \mathbf{A}\cdot\mathrm{d}\mathbf{l} = \int_S (\nabla\times\mathbf{A})\cdot\mathrm{d}\mathbf{S}$$

**辨析｜易错点：** 散度为零的场叫**无源场（螺线管场）**，旋度为零的场叫**无旋场（保守场）**。静电场无旋（$\nabla\times\mathbf{E}=0$），所以可以写成某势函数的梯度；恒定磁场无源（$\nabla\cdot\mathbf{B}=0$），所以磁力线总是闭合的、没有"磁单极"。这组对应是本章最重要的一张对照表，务必分清"哪个场带旋、哪个场带散"。

## 6 公式解析：拉普拉斯算符与亥姆霍兹定理

把"梯度之散度"作用在标量 $\varphi$ 上，得到**拉普拉斯算子（Laplacian）**：

$$\nabla^2\varphi = \nabla\cdot(\nabla\varphi) = \frac{\partial^2\varphi}{\partial x^2} + \frac{\partial^2\varphi}{\partial y^2} + \frac{\partial^2\varphi}{\partial z^2}$$

对这条公式做三步拆解：

- **第一步，看清复合结构**：先对 $\varphi$ 取梯度（得矢量），再对这个矢量取散度（得标量）。$\nabla^2$ 是一个二阶标量算子，作用结果仍是标量。
- **第二步，物理含义**：$\nabla^2\varphi = 0$ 是**拉普拉斯方程**，$\nabla^2\varphi = -f$ 是**泊松方程**。无源区电势满足拉普拉斯方程——这正是后面静电场边值问题求解的核心方程。
- **第三步，为什么重要**：给定矢量场的散度与旋度，再加边界条件，场就被唯一确定。这条**亥姆霍兹定理**保证：工程上"测出散度和旋度就能重建整个场"，也是数值方法（第 10 篇）的数学依据。

做电磁场推导时，下面几条矢量恒等式比任何技巧都常用，建议当作四则运算一样熟练：

- $\nabla\cdot(\nabla\times\mathbf{A}) = 0$——任何场旋度的散度恒为零，这是"磁力线闭合、无磁单极"的代数根源。
- $\nabla\times(\nabla\varphi) = 0$——任何标量梯度的旋度恒为零，这是"静电场无旋、可写电势"的代数根源。
- $\nabla\times(\nabla\times\mathbf{A}) = \nabla(\nabla\cdot\mathbf{A}) - \nabla^2\mathbf{A}$——**矢量拉普拉斯恒等式**，导出电磁波波动方程的关键一步，第 4 篇求解 Maxwell 方程组时会原样用到。

<div class="epigraph">
<p>……无论场的散度与旋度取什么值，只要它们在整个区域内已知，且场在无穷远处趋于零，这个场就是唯一确定的。</p>
<footer>—— 赫尔曼 · 冯 · 亥姆霍兹（Hermann von Helmholtz）</footer>
</div>

## 7 小结

- **矢量与矢量场**是电磁学的语言：点积度量同向、叉积给出面积元方向。
- 三种**正交坐标系**（直角、圆柱、球）按边界形状选用，注意面积元/体积元的"弧长因子"。
- **梯度** $\nabla\varphi$ 指向标量场增长最快的方向；**散度** $\nabla\cdot\mathbf{A}$ 度量源的强弱；**旋度** $\nabla\times\mathbf{A}$ 度量旋转强弱。
- **高斯散度定理**把通量与源总量相连，**斯托克斯定理**把环量与旋度相连。
- **亥姆霍兹定理**：场的散度与旋度加边界条件唯一确定场，是后续全部电磁场求解的基石。
- **矢量恒等式**是推导的引擎：$\nabla\cdot(\nabla\times\mathbf{A})=0$、$\nabla\times(\nabla\varphi)=0$、$\nabla\times(\nabla\times\mathbf{A})=\nabla(\nabla\cdot\mathbf{A})-\nabla^2\mathbf{A}$，前两条对应"无磁单极"与"静电无旋"，第三条导出波动方程。

在下一节，我们带着这三件武器进入第一种具体场——**静电场**：它无旋、有源，于是可以用电势 $\varphi$