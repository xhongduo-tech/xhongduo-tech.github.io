---
title: 分析力学入门（拉格朗日方程、广义坐标、约束）
date: 2026-08-07
---

# 分析力学入门（拉格朗日方程、广义坐标、约束）

<div class="epigraph">
<p>这部著作不会出现任何图——唯有代数运算，从始至终。</p>
<footer>—— 约瑟夫-路易 · 拉格朗日（Joseph-Louis Lagrange），《分析力学》（Mécanique Analytique）前言，1788</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 周衍柏《理论力学》分析力学部分 ｜ 2026-08-07</p>
</div>

## 为什么从分析力学入门开始

在「第七篇 四大力学入门」里，我们已经见过拉格朗日方程的「操作版」：选广义坐标、写 $L = T - V$、代入 $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot q_j} - \frac{\partial L}{\partial q_j} = 0$。那是用结果说话。而本专题进入「第1篇」深挖课程，第一站就要把分析力学的**根基**一次讲透：约束为什么必须分类、广义坐标凭什么成立、达朗贝尔原理与虚功原理如何把牛顿方程「几何化」。这些概念是哈密顿力学、乃至量子场论与广义相对论的共同地基——现代物理的每一套基本理论，都写成「拉格朗日量 + 变分」的形式。先立住根基，后面三篇（哈密顿、电磁场、量子）才能站在同一套语言上说话。

## 1 约束与自由度

牛顿力学里，我们为每个质点列受力方程，约束（绳、杆、轨道、铰链）以**约束力的形式**进入方程。分析力学换了一个视角：把约束当成**先验的几何条件**，在列方程之前就把它消掉。

**约束（constraint）**：限制质点系中各质点位置（与速度）的几何条件或运动学条件。对 $N$ 个质点、$3N$ 个坐标的系统，约束方程可以写成

$$
f_\alpha(\boldsymbol{r}_1, \dots, \boldsymbol{r}_N, \dot{\boldsymbol{r}}_1, \dots, \dot{\boldsymbol{r}}_N, t) = 0, \qquad \alpha = 1, 2, \dots, r
$$

按是否含速度、是否显含时间，约束分成两类正交的维度：

**完整约束（holonomic constraint）**：约束方程不含速度（或可以积分成不含速度的形式），只限制位置，如「绳长固定」$x^2 + y^2 = l^2$。**非完整约束（nonholonomic constraint）**：约束方程不可积地含速度，如球在粗糙面上的纯滚动、冰刀只能沿刀锋方向滑行。<span class="marginnote"><strong>为什么「完整」这么重要</strong>：完整约束可以直接代入坐标、把坐标数降下来；非完整约束则只能以微分形式限制虚位移，无法整体消去坐标——这就是后面拉格朗日乘子法要处理的对象。轮式机器人、自行车都属于典型的非完整系统。</span>

**自由度（degrees of freedom）**：确定系统位形所需的独立坐标个数，$s = 3N - r$（$r$ 个独立完整约束）。约束越强、自由度越小：自由粒子 $s=3$，单摆 $s=1$，刚体 $s=6$。<span class="marginnote">刚体的 6 个自由度：3 个平动（质心位置）+ 3 个转动（欧拉角）。「自由度」是分析力学最重要的数——方程个数、广义坐标个数、守恒量个数都等于它。</span>

## 2 广义坐标

有了自由度，就能定义一套「任何方便的量」来充当坐标。**广义坐标（generalized coordinates）**：足以完全确定系统位形的、相互独立的一组参量 $q_1, q_2, \dots, q_s$，个数等于自由度。它不一定是长度，可以是角度、弧长、面积，甚至电荷量——只要「够用且独立」即可。

每个广义坐标 $q_j$ 对应一个**广义速度** $\dot q_j$ 与一个**广义力** $Q_j$。位置矢量用广义坐标重写：

$$
\boldsymbol{r}_i = \boldsymbol{r}_i(q_1, q_2, \dots, q_s, t), \qquad
\dot{\boldsymbol{r}}_i = \sum_{j=1}^{s} \frac{\partial \boldsymbol{r}_i}{\partial q_j}\,\dot q_j + \frac{\partial \boldsymbol{r}_i}{\partial t}
$$

**重点：广义坐标把「坐标系选择」从解题技巧升格为理论的基本自由度——物理不依赖坐标，方程的形式也不依赖坐标。** 用极坐标写行星轨道、用角坐标写复摆，方程形式完全相同。<span class="marginnote">在狭义相对论与广义相对论里，「坐标本来就是任意的标号」这一观念成为第一原理；而在量子场论里，场 $\phi(x)$ 本身就是「无穷多个广义坐标」。广义坐标是把经典力学与现代物理连成一线的第一个概念。</span>

**数值算例（双摆）**：两个共面摆锤组成的双摆，位形由两个角度 $(\theta_1, \theta_2)$ 完全确定，$s=2$。用直角坐标需要 4 个坐标加 2 条约束方程；用广义坐标 $(\theta_1,\theta_2)$ 则 2 个独立坐标一步到位——这就是「选坐标就是选自由度」的活例。

## 3 虚位移与虚功原理

分析力学把「力」的视角换成「功」的视角。**虚位移（virtual displacement）$\delta\boldsymbol{r}_i$**：在某一瞬时、保持时间 $t$ 不变、约束不破坏的前提下，质点系各质点假想发生的**任意微小位移**。「虚」在于它不依赖于时间的流逝，与真实位移 $\mathrm{d}\boldsymbol{r}_i$ 有本质区别。<span class="marginnote"><strong>虚位移 vs 真实位移</strong>：真实位移 $\mathrm{d}\boldsymbol{r}_i$ 需要时间 $dt$，且由动力学决定；虚位移 $\delta\boldsymbol{r}_i$ 是「冻结时间」后由约束许可的试探位移。对定常约束，真实位移 $\mathrm{d}\boldsymbol{r}_i$ 是虚位移之一；对非定常约束则不是。</span>

**理想约束（ideal constraint）**：约束反力在任意虚位移上做的虚功之和为零，$\sum_i \boldsymbol{N}_i \cdot \delta\boldsymbol{r}_i = 0$。光滑接触面、光滑铰链、无质量的刚性杆、不可伸长的绳都是理想约束——它们不做功，于是能从方程里整体剔除。

在理想约束下，对静力平衡的质点系，把牛顿定律投影到虚位移方向、再求和，得到**虚功原理（principle of virtual work）**：

$$
\sum_i \boldsymbol{F}_i \cdot \delta\boldsymbol{r}_i = 0
$$

**重点：虚功原理把「平衡」翻译成「主动力对任意虚位移不做功」——约束力被理想约束条件自动消去。** 这比列力矩平衡方程更一般：约束越多、结构越复杂，虚功原理越显威力。<span class="marginnote">虚功原理在结构力学里直接演化成「虚位移原理（求力）」与「虚力原理（求位移）」两条互补工具，而有限元方法的刚度矩阵正是从虚功方程出发导出的——分析力学离工程计算比想象的近。</span>

## 4 达朗贝尔原理：动力学问题的「静力学化」

静止的平衡问题有虚功原理，运动问题怎么办？达朗贝尔（Jean le Rond d'Alembert）在 1743 年给出天才的一步：把「惯性力 $-m_i\boldsymbol{a}_i$」当作一个额外的力，动力学方程 $\boldsymbol{F}_i + \boldsymbol{N}_i - m_i\boldsymbol{a}_i = 0$ 形式上就变成了「平衡」方程。再对虚位移求和，利用理想约束消去约束力，得到**达朗贝尔原理（d'Alembert's principle）**：

$$
\sum_i (\boldsymbol{F}_i - m_i\boldsymbol{a}_i)\cdot\delta\boldsymbol{r}_i = 0
$$

**重点：达朗贝尔原理是「虚功原理 + 惯性力」——把动力学问题改写为每一瞬时都成立的虚功平衡。** 它并不神奇到免去受力分析，但它把「力」换成了「能量与几何」，为拉格朗日方程铺平了最后一级台阶。<span class="marginnote"><strong>谁是达朗贝尔</strong>：法国启蒙数学家，《百科全书》主编之一，1743 年在《动力学论》（Traité de dynamique）中首次给出这一原理。他的「惯性力」观念后来成为非惯性系里惯性力（离心力、科里奥利力）概念的直接源头，见前文《非惯性系与惯性力》。</span>

## 5 公式解析：从达朗贝尔原理到拉格朗日方程

把虚位移展开到广义坐标、把各项整理成对广义坐标的导数，就能从达朗贝尔原理严格推出拉格朗日方程。这一推导是分析力学的「加冕礼」，值得逐行拆解：

$$
\sum_i (\boldsymbol{F}_i - m_i\boldsymbol{a}_i)\cdot\delta\boldsymbol{r}_i = 0 \quad\Longrightarrow\quad \frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot q_j} - \frac{\partial L}{\partial q_j} = 0
$$

- **第一步，展开虚位移**：$\delta\boldsymbol{r}_i = \sum_j \frac{\partial\boldsymbol{r}_i}{\partial q_j}\delta q_j$（约束已把 $3N$ 维位置压到 $s$ 维流形上，虚位移只沿约束许可方向）。
- **第二步，处理主动力项**：$\sum_i \boldsymbol{F}_i \cdot \delta\boldsymbol{r}_i = \sum_j Q_j\,\delta q_j$，定义**广义力** $Q_j = \sum_i \boldsymbol{F}_i \cdot \frac{\partial\boldsymbol{r}_i}{\partial q_j}$。当主动力有势 $V$ 时，$Q_j = -\frac{\partial V}{\partial q_j}$。
- **第三步，处理惯性力项**：利用恒等式 $\sum_i m_i\boldsymbol{a}_i\cdot\delta\boldsymbol{r}_i = \sum_j \left[\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial T}{\partial\dot q_j} - \frac{\partial T}{\partial q_j}\right]\delta q_j$，其中 $T$ 是动能——这是推导里最巧妙的一步，把加速度转化为动能的导数。
- **第四步，令各 $\delta q_j$ 独立变分**：$s$ 个虚位移相互独立，系数必须分别为零，于是得到 $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial T}{\partial\dot q_j} - \frac{\partial T}{\partial q_j} = Q_j$；引入 $L = T - V$ 且 $Q_j = -\partial V/\partial q_j$，即得拉格朗日方程。

**辨析｜易错点：** 推导中「$\delta q_j$ 相互独立」这一条件只在**完整约束**下严格成立——约束若非完整，系数不能逐项归零，必须用拉格朗日乘子法修正。所以「拉格朗日方程的标准形式 = 完整约束 + 主动力有势」两个前提缺一不可。

## 6 拉格朗日乘子法与约束力

现实问题往往需要知道约束力（绳的张力、轨道的支持力）。拉格朗日乘子法（Lagrange multiplier method）把它找回来：把约束 $f_\alpha(q,t)=0$ 乘上待定乘子 $\lambda_\alpha$ 加入拉格朗日量，

$$
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot q_j} - \frac{\partial L}{\partial q_j} = \sum_\alpha \lambda_\alpha \frac{\partial f_\alpha}{\partial q_j}
$$

**重点：乘子 $\lambda_\alpha$ 的物理意义就是广义约束力——多一个约束，就多一个乘子，也多一条方程。** 求出的 $\lambda_\alpha$ 乘上对应方向的几何因子，就是真实约束力。<span class="marginnote"><strong>数值算例</strong>：研究摆长 $l$、质量 $m$ 的单摆，把约束 $r=l$ 显式保留，拉格朗日乘子法给出张力 $T = ml\dot\theta^2 + mg\cos\theta$——正是牛顿法受力分析的结果。乘子法把「隐藏的约束力」重新显形。</span>在约束优化（拉格朗日对偶）、有限元约束处理、以及统计力学里「在平均能量固定下最大化熵」（正则系综的推导）中，同一套乘子思想反复出现——它是最通用的「带约束变分」工具。

## 7 核心对比表：牛顿力学与分析力学

两套框架回答同一批力学问题，但语言与工具完全不同。把并排对照看，才能体会分析力学「换语言」的价值：

| 维度 | 牛顿力学 | 分析力学（拉格朗日） |
| --- | --- | --- |
| 基本对象 | 力 $\boldsymbol{F}$、加速度 $\boldsymbol{a}$ | 广义坐标 $q_j$、动能 $T$、势能 $V$ |
| 核心方程 | $m\boldsymbol{a} = \sum\boldsymbol{F}$（矢量方程） | $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot q_j}-\frac{\partial L}{\partial q_j}=0$（标量方程） |
| 约束处理 | 约束力显式出现，需受力分析 | 约束被坐标吸收，约束力不出现 |
| 坐标选择 | 依赖笛卡尔坐标惯性系 | 任意广义坐标，形式不变 |
| 每质点 | 每质点 3 条方程 | 每自由度 1 条方程 |
| 守恒律 | 解方程后验证 | 循环坐标一眼看出（诺特定理） |
| 现代延伸 | 经典刚体、多体动力学 | 量子场论、广义相对论、规范理论 |

**重点：牛顿力学的方程个数 = $3N$，分析力学的方程个数 = 自由度 $s$；前者解「具体力」，后者解「系统结构」。** 对自由度数远小于 $3N$ 的约束系统（双摆、机械臂、航天器），分析力学省掉的方程数以指数计——这正是机器人运动学与多体系统仿真全部改用拉格朗日或哈密顿表述的原因。

**数值算例（复摆线性化）**：平面双摆在大约平衡位置附近作小振动，拉格朗日方程自动给出两个耦合的线性常微分方程，其行列式条件（久期方程）直接给出两个本征频率 $\omega^2 = (g/l)(3\pm\sqrt{3})$。用牛顿法要同时处理 4 个坐标与 2 个约束；用拉格朗日法只写两个角度、两条方程——自由度 $s=2$ 的全部威力在此显形。<span class="marginnote">这个频率结果与耦合振子的简正模分析（见前文《简谐振动的合成》）本质相同：拉格朗日方程把「求振动频率」统一成「解矩阵本征问题」，为后面量子力学的本征值问题埋下同一颗种子。</span>

## 8 小结

- **约束**分完整（只限位置）与非完整（限速度、不可积）；**自由度** $s = 3N - r$，是分析力学一切「个数」的总闸。
- **广义坐标**是任意「够用且独立」的参量，个数等于自由度；物理规律与坐标选择无关。
- **虚位移**是冻结时间后约束许可的试探位移；**理想约束**虚功为零；**虚功原理** $\sum_i\boldsymbol{F}_i\cdot\delta\boldsymbol{r}_i=0$ 刻画静力平衡。
- **达朗贝尔原理**把「惯性力」并入虚功平衡，实现动力学的「静力学化」。
- **拉格朗日方程** $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot q_j} - \frac{\partial L}{\partial q_j}=0$ 由达朗贝尔原理严格推出，前提是完整约束 + 主动力有势。
- **拉格朗日乘子法**把约束力找回来：乘子 $\lambda_\alpha$ 即广义约束力。

在下一节，我们将从拉格朗日框架升入哈密顿的表述：引入相空间与正则方程，看看力学如何变成「几何学」——**哈密顿力学与正则变换**。
