---
title: 麦克斯韦方程组的建立
date: 2026-08-07
---

# 麦克斯韦方程组的建立

<div class="epigraph">
<p>光，就是一种以波的形式在电磁场中传播的电磁扰动。</p>
<footer>—— 詹姆斯 · 克拉克 · 麦克斯韦（James Clerk Maxwell, 1865）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第一章 §1.6 ｜ 2026-08-07</p>
</div>

## 为什么安培环路定理必须被修正

1861 年的电磁学看似完备：库仑定律、安培环路定理、法拉第定律、磁单极子缺失——四条经验定律各安其位。但麦克斯韦敏锐地发现一个**逻辑裂缝**：安培环路定理 $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$ 与电荷守恒定律互相矛盾。因为任何旋度的散度恒为零，取安培定理两边的散度得 $\nabla\cdot\mathbf{J} = 0$——电流无散。但连续性方程 $\nabla\cdot\mathbf{J} = -\partial\rho/\partial t$ 说，电荷密度随时间变化的地方，电流必须有散度。两者冲突。<span class="marginnote">这正是电容器充电的困境：电容器两极板之间没有传导电流，却有随时间变化的电场。安培环路定理用在包围极板的回路上，会得出「磁场有无穷多种互相矛盾的答案」——取决于你选哪张曲面穿过回路。这不是小瑕疵，而是理论在时变情形下的根本失稳。</span>

## 1 位移电流

麦克斯韦的补救方法是补上缺失的项。利用高斯定理 $\nabla\cdot\mathbf{D} = \rho$（$\mathbf{D}$ 为电位移矢量，见《电介质中的静电场》），连续性方程改写为

$$\nabla\cdot\mathbf{J} + \frac{\partial}{\partial t}(\nabla\cdot\mathbf{D}) = \nabla\cdot\left(\mathbf{J} + \frac{\partial\mathbf{D}}{\partial t}\right) = 0$$

因此「$\mathbf{J} + \partial\mathbf{D}/\partial t$」是无散的。麦克斯韦把第二项定义为**位移电流密度**：

$$\mathbf{J}_d = \frac{\partial \mathbf{D}}{\partial t}$$

它虽非真实电荷流动，却在产生磁场方面与传导电流等价。修正后的安培环路定理为

$$\nabla\times\mathbf{H} = \mathbf{J} + \frac{\partial\mathbf{D}}{\partial t}$$

其中 $\mathbf{H}$ 是磁场强度（与 $\mathbf{B}$ 的关系由本构方程 $\mathbf{B} = \mu\mathbf{H}$ 给出）。<span class="marginnote">位移电流的物理本质是「变化的电场」：真空中它是 $\varepsilon_0\partial\mathbf{E}/\partial t$，电介质中还包含极化电荷的运动。它让电场与磁场第一次实现「互相激发」——变化的电场产生磁场，变化的磁场又产生电场，电磁波因此成为可能。</span>

**辨析｜易错点：** 位移电流不是「电荷的流动」，它只是「电位移矢量的时间变化率」，在真空中也照样存在。它之所以叫「电流」，仅仅是因为在产生磁场这一效果上与传导电流等价。把它误解为真实的电荷流，是初学者最常见的错误。在电容器极板间，磁场完全由位移电流产生——赫兹用这个事实证明了位移电流的物理实在性。

## 2 麦克斯韦方程组

补全位移电流后，电磁学的大厦落成。**麦克斯韦方程组（Maxwell's equations）** 的微分形式：

$$\nabla\cdot\mathbf{D} = \rho \qquad \nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}$$

$$\nabla\cdot\mathbf{B} = 0 \qquad \nabla\times\mathbf{H} = \mathbf{J} + \frac{\partial\mathbf{D}}{\partial t}$$

积分形式则更为直观：

| 方程 | 积分形式 | 物理含义 |
| --- | --- | --- |
| 高斯定律 | $\oint_S \mathbf{D}\cdot\mathrm{d}\mathbf{S} = \iiint_V \rho\,\mathrm{d}V$ | 电荷是电场的散度源 |
| 高斯磁定律 | $\oint_S \mathbf{B}\cdot\mathrm{d}\mathbf{S} = 0$ | 无磁单极子 |
| 法拉第定律 | $\oint_L \mathbf{E}\cdot\mathrm{d}\mathbf{l} = -\dfrac{\mathrm{d}}{\mathrm{d}t}\int_S\mathbf{B}\cdot\mathrm{d}\mathbf{S}$ | 变化磁场产生电场 |
| 安培-麦克斯韦定律 | $\oint_L \mathbf{H}\cdot\mathrm{d}\mathbf{l} = \iint_S \left(\mathbf{J} + \dfrac{\partial\mathbf{D}}{\partial t}\right)\cdot\mathrm{d}\mathbf{S}$ | 电流与变化电场产生磁场 |

方程组要闭环，还需要**本构关系（constitutive relations）**把场量连起来，外加**洛伦兹力**：

$$\mathbf{D} = \varepsilon\mathbf{E}, \qquad \mathbf{B} = \mu\mathbf{H}, \qquad \mathbf{F} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B})$$

<span class="marginnote">真空中 $\mathbf{D} = \varepsilon_0\mathbf{E}$、$\mathbf{B} = \mu_0\mathbf{H}$。麦克斯韦方程组 + 洛伦兹力 + 牛顿方程，构成了经典电磁学的全部——所有电磁现象，从电荷吸引到电磁波传播，都是这组方程的解。</span>

**麦克斯韦方程组的美学意义**：四条方程把电与磁统一成一个自洽整体，且三条来源经验、一条来源逻辑自洽性（位移电流）。更关键的是，它揭示了 **电场与磁场不再是两个独立实体，而是同一个电磁场的两个方面**——在不同参考系里互相转化。

## 3 电磁波的存在性：麦克斯韦的惊人预言

在真空中（$\rho = 0,\ \mathbf{J} = 0$），对麦克斯韦方程组取旋度可导出电磁波方程：

$$\nabla^2\mathbf{E} = \mu_0\varepsilon_0 \frac{\partial^2\mathbf{E}}{\partial t^2}, \qquad \nabla^2\mathbf{B} = \mu_0\varepsilon_0 \frac{\partial^2\mathbf{B}}{\partial t^2}$$

这是标准的**波动方程**，波速

$$c = \frac{1}{\sqrt{\mu_0\varepsilon_0}} \approx 2.998 \times 10^8\ \mathrm{m/s}$$

与实测光速一致。麦克斯韦据此断言：**光是电磁波**。这一预言 1888 年被赫兹实验证实，是整个 19 世纪物理学最辉煌的成果。<span class="marginnote">推导要点：对 $\nabla\times\mathbf{E} = -\partial\mathbf{B}/\partial t$ 取旋度，用恒等式 $\nabla\times(\nabla\times\mathbf{E}) = \nabla(\nabla\cdot\mathbf{E}) - \nabla^2\mathbf{E}$ 与 $\nabla\cdot\mathbf{E}=0$ 消去散度项，再代入 $\nabla\times\mathbf{H} = \partial\mathbf{D}/\partial t$ 与 $\mathbf{B}=\mu_0\mathbf{H}$。这个 5 行推导值得读者亲手做一遍——它把电场与磁场「你中有我、我中有你」地锁成波。</span>

## 4 公式解析：位移电流如何救活安培定理

核心步骤是把「无散条件」化为「存在性」，拆成三步：

**第一步，取散度暴露矛盾**：对安培定理 $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$ 两边取散度，左边为零（旋度无散），右边 $\propto \nabla\cdot\mathbf{J}$。于是方程要求 $\nabla\cdot\mathbf{J} = 0$，与连续性方程矛盾——除非 $\rho$ 恒定。安培定理只在静磁情形成立。
**第二步，利用高斯定理改写**：$\nabla\cdot\mathbf{D} = \rho$，于是 $\partial\rho/\partial t = \nabla\cdot(\partial\mathbf{D}/\partial t)$。连续性方程变成 $\nabla\cdot(\mathbf{J} + \partial\mathbf{D}/\partial t) = 0$——括号里这个量自动无散。
**第三步，替换无散源**：因为旋度场的散度恒为零，而物理上磁场只由无散的量产生，把 $\mathbf{J}$ 换成「$\mathbf{J} + \partial\mathbf{D}/\partial t$」后，安培定理与电荷守恒不再冲突。麦克斯韦的洞见在于：**当数学结构自相矛盾时，问题不在数学，而在物理模型漏掉了一项**。<span class="marginnote">这类「由逻辑自洽性发现新物理」的例子在理论物理中反复出现：狄拉克方程、中微子（泡利）、正电子（狄拉克）都是先由方程的自洽性预言，后被实验证实。麦克斯韦位移电流是这一方法论的第一个伟大胜利。</span>

## 5 从方程组看电磁学的统一图景

把四条方程摆在一起，能看到三个层面的统一：

- **电与磁统一**：电场可由电荷（高斯）或变化磁场（法拉第）产生；磁场可由电流（安培）或变化电场（位移电流）产生。
- **源与场统一**：方程组左侧是场的微分（散度、旋度），右侧是源（电荷、电流）。「源产生场」是局域关系——场在每一点由该点的源决定，不存在超距作用。
- **电磁与光学统一**：波速等于光速，光学成为电磁学的特例。

**辨析｜易错点：** 初学者常误以为麦克斯韦方程组「四选一就能推出全部」。事实上四条方程缺一不可：去掉位移电流，电磁波不复存在；去掉法拉第定律，磁场不再能产生电场；去掉高斯磁定律，磁单极子被允许。四条方程 + 本构关系 + 洛伦兹力才是完整的电磁理论。

## 6 麦克斯韦方程组的对称性与未解之谜

四条方程摆在一起，还藏着几层「不对称」，每层都通向 20 世纪物理学的新篇章。

**不对称一：电有源、磁无源。** 方程里没有磁荷项，电场有散度源而磁场没有。若存在磁单极子，方程组会变得完全对称（并自动解释电荷量子化），但实验至今未找到。**这个「缺失的对称性」是当代物理的悬案之一。**

**不对称二：时变项的不配对。** 法拉第定律右侧是 $-\partial\mathbf{B}/\partial t$，安培-麦克斯韦定律右侧是 $\partial\mathbf{D}/\partial t$，形式上配对。但静电场可以有独立于磁场的解（点电荷静止），静磁场却不能独立于电场——**因为「电荷」存在而「磁荷」不存在**。若引入磁荷，电磁场会获得全新的自由度（电-磁对偶），这是理论物理反复试探的方向。

**不对称三：方程是线性的。** 麦克斯韦方程组对场是线性的，但电荷密度 $\rho$ 与电流密度 $\mathbf{J}$ 是场的函数（电荷受洛伦兹力运动），整体上「场 + 电荷」的自洽系统是**非线性**的。非线性带来孤子、混沌等丰富现象，也意味着「电荷运动 + 场演化」的完整问题往往需要数值求解。

**方程组的边界：** 麦克斯韦方程组在什么情形失效？——微观尺度上量子效应接管（原子中的电子不按经典辐射塌缩）；强场下真空中会出现正负电子对产生（施温格极限，场强约 $1.3\times10^{18}\ \mathrm{V/m}$）。**经典电磁学的适用范围是「宏观 + 弱场」，越过边界就要量子电动力学登场。**

**辨析｜易错点：** 常有人说「麦克斯韦方程组包含全部电磁学」。严格说是「经典电磁学在宏观尺度上由麦克斯韦方程组 + 洛伦兹力 + 本构关系描述」，量子与强场效应不在其中。把「方程组完备」与「理论完备」区分开，才不会被教材的豪言误导。

**规范条件的预告**：麦克斯韦方程组用 $\mathbf{E}$、$\mathbf{B}$ 描述时是四条；用 $\varphi$、$\mathbf{A}$ 描述时，四条方程压缩成两条（关于 $\varphi$ 与 $\mathbf{A}$ 的波动方程），但多了规范自由度。常用的**洛伦兹规范** $\nabla\cdot\mathbf{A} + \dfrac{1}{c^2}\dfrac{\partial\varphi}{\partial t} = 0$ 把两条方程解耦成对称的推迟势方程——**这就是「推迟势」形式的麦克斯韦方程组**，也是下一章辐射理论（李纳-维谢尔势、偶极辐射）的出发点。规范的选择将在《磁场与矢势》与相对论章节里反复出现。

**麦克斯韦方程组的历史地位**：1873 年麦克斯韦《电磁通论》出版，统一了电、磁、光三大领域。费曼曾评价：如果文明只剩一条信息传给后人，应该选「物质由原子构成」；而物理学史上最伟大的公式集，麦克斯韦方程组当之无愧。**它也是「从极限到大模型」这条主线上，数学物理方法第一次在真实物理中全面兑现的范本**——分离变量、格林函数、张量分析，全都在这里有了用武之地。

## 7 小结

- **位移电流** $\mathbf{J}_d = \partial\mathbf{D}/\partial t$：为兼容电荷守恒而引入，是时变电磁场的枢纽。
- **麦克斯韦方程组**四条：高斯定律、高斯磁定律、法拉第定律、安培-麦克斯韦定律，微分与积分形式互为表里。
- **本构关系** $\mathbf{D}=\varepsilon\mathbf{E}$、$\mathbf{B}=\mu\mathbf{H}$ 与**洛伦兹力**补全闭环。
- 真空中导出**波动方程**，波速 $c = 1/\sqrt{\mu_0\varepsilon_0}$——光即电磁波。
- 方程组揭示了电、磁、光三者的统一，是经典物理的巅峰。

在下一节，我们将追问：电磁场既是物理实体，它就该有能量与动量——电磁场如何储存能量、传递动量？——**电磁场的能量与动量**。
