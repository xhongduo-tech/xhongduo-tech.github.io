---
title: 软物质流变学
date: 2026-08-07
---

# 软物质流变学

<div class="epigraph">
<p>流变学的任务是回答一个问题：物质在外力下如何流动与变形？而对软物质来说，答案常常是——看你怎么碰它。</p>
<footer>—— 土井正男（Masao Doi），*Soft Matter Physics\*</footer>
</div>

<div class="article-byline">
<p>第四级 · 软物质物理 ｜ Masao Doi, *Soft Matter Physics\*, Ch.9 ｜ 2026-08-07</p>
</div>

## 为什么从流变学开始

前几篇我们分别研究了链、颗粒、液晶、自组装与玻璃。可无论哪种软物质，最终都要过一道工程关：**它在被搅拌、涂抹、挤压、泵送时表现如何？** 牙膏要挤得动、又要立在牙刷上；面团要拉得开、又不能瘫成糊；乳液要稠得挂壁、又要稀得倒得出。这门研究「流动与变形」的学科叫**流变学（rheology）**，而软物质流变学最迷人的地方是它的「两面性」：**同一块材料，你慢吞吞地碰它，它像液体；你猛地一下砸它，它像固体**。这一篇先把「黏」和「弹」两个理想端立起来，再用一个弹簧和阻尼器搭出黏弹性，最后给你两把测量软物质的标准工具：**动态模量**与**剪切变稀**。

## 1 流变学要描述什么

任何材料受外力都有两种基本响应：

- **弹性（elasticity）**：形变与应力成比例，撤力即恢复，能量被**储存**——理想弹性体是弹簧。
- **黏性（viscosity）**：应力与形变速率成比例，能量被**耗散**——理想黏性体是阻尼器（活塞在油缸里）。

流变学关心的就是这两者的比例与配合。对理想黏性流体，牛顿定律写为

$$
\sigma = \eta\,\dot\gamma
$$

其中 $\sigma$ 是剪切应力、$\dot\gamma$ 是剪切速率（$1/\mathrm{s}$）、$\eta$ 是**黏度（viscosity）**。水的 $\eta \approx 10^{-3}\,\mathrm{Pa\!\cdot\!s}$，蜂蜜约 10 Pa·s，聚合物熔体可达 $10^3{-}10^6$ Pa·s。

判断「像液体还是像固体」需要一个无量纲数——**德博拉数（Deborah number）**

$$
\mathrm{De} = \frac{\tau}{t_{\mathrm{obs}}}
$$

即材料的内禀弛豫时间 $\tau$ 与观察时间 $t_{\mathrm{obs}}$ 之比。<span class="marginnote">德博拉（Deborah）典出《圣经·士师记》：「大山踊跃如公羊，小山跳舞如羊羔」——大山都像流体一样舞动，只要你的时间尺度足够长。$\mathrm{De} \gg 1$ 时材料显得像固体（来不及弛豫），$\mathrm{De} \ll 1$ 时它彻底像液体。<strong>软与硬的分别，常常只是你多快去看它。</strong></span>这句话是整个流变学的哲学起点。

## 2 应力、应变与应变率

先把三个量定死。取一块夹在平行板之间的材料，下板固定、上板以速度 $v$ 平移，板间距 $d$：

- **剪切应变** $\gamma = \Delta x / d$：上板位移除以板距，无量纲。
- **剪切应变率** $\dot\gamma = \mathrm{d}\gamma/\mathrm{d}t = v/d$：单位时间的应变，单位 $1/\mathrm{s}$。
- **剪切应力** $\sigma = F/A$：上板受到的切向力除以面积，单位 Pa。

三个量之间的关系就是材料的**本构方程（constitutive equation）**——流变学研究的核心对象。理想弹簧 $\sigma = G\gamma$（$G$ 为剪切模量），理想牛顿流体 $\sigma = \eta\dot\gamma$。软物质的本构方程几乎总是二者的混合，且常常还依赖于 $\dot\gamma$ 本身的大小。

## 3 线性黏弹性：Maxwell 与 Kelvin–Voigt 模型

把弹簧与阻尼器用最朴素的方式拼起来，就得到黏弹性的两个基本模型：

**Maxwell 模型**：弹簧与阻尼器**串联**。本构方程

$$
\sigma + \tau \frac{\mathrm{d}\sigma}{\mathrm{d}t} = \eta\,\dot\gamma, \qquad \tau = \frac{\eta}{G}
$$

$\tau$ 是**麦克斯韦弛豫时间**。物理图景：瞬间敲它，弹簧先响应（像固体）；等久了，阻尼器慢慢滑移（像液体）。它描述应力弛豫——拉住材料不动（$\dot\gamma = 0$），应力按 $e^{-t/\tau}$ 衰减。聚合物熔体、蜂蜜都接近 Maxwell 行为。

**Kelvin–Voigt 模型**：弹簧与阻尼器**并联**。本构方程 $\sigma = G\gamma + \eta\dot\gamma$。物理图景：材料被拉到一个新位置后，不会完全松弛到零，而是被弹簧拽回——描述蠕变与延迟恢复，凝胶、黏土更像它。<span class="marginnote">如何区分一个黏弹体是 Maxwell 型还是 Kelvin–Voigt 型？做两个实验：应变固定看应力是否弛豫到零（Maxwell 会），应力固定看应变是否趋于有限值（Kelvin–Voigt 会）。软物质里大多数体系是多个 Maxwell/Kelvin 单元并联（广义 Maxwell 模型），用一个弛豫时间谱描述。</span>

**辨析｜易错点：** 别把「黏度」与「稠」混为一谈。黏度是材料属性（$\eta = \sigma/\dot\gamma$），「稠不稠」是你挤它时的体感。更易错的是：**很多软物质的黏度根本不是常数**——它随剪切速率变化。这正是下一节要处理的非线性效应。

## 4 公式解析：动态模量与损耗角

流变学最标准的一把尺子，是给材料施加正弦应变 $\gamma(t) = \gamma_0 \sin(\omega t)$，测量应力响应。对线性黏弹性体，应力不是同相的正弦，而是

$$
\sigma(t) = \gamma_0 \big[\, G'(\omega)\sin(\omega t) + G''(\omega)\cos(\omega t)\, \big]
$$

- **第一步，认识两个模量**：$G'$ 是**储能模量（storage modulus）**，与应变同相，代表弹性、能存能放；$G''$ 是**损耗模量（loss modulus）**，与应变差 90° 相位，代表黏性、耗散能量。两者的比值 $\tan\delta = G''/G'$ 叫**损耗角正切**。
- **第二步，代入 Maxwell 模型**：把 $G'$、$G''$ 从 $\sigma = G\gamma + \eta\dot\gamma$（即 Maxwell 的频域形式 $G^* = i\omega\eta G/(G + i\omega\eta)$）中解出来，得

$$
G'(\omega) = G\,\frac{(\omega\tau)^2}{1 + (\omega\tau)^2}, \qquad
G''(\omega) = G\,\frac{\omega\tau}{1 + (\omega\tau)^2}
$$

- **第三步，读极限**：低频（$\omega\tau \ll 1$）时 $G' \propto \omega^2 \to 0$、$G'' \propto \omega$——材料表现为**黏性流体**；高频（$\omega\tau \gg 1$）时 $G' \to G$、$G'' \to 0$——表现为**弹性固体**。这正是「砸它像固体、放它像液体」的定量版。
- **第四步，实验判据**：实际测量时只需看 $G'$ 与 $G''$ 的大小与交点。**凝胶**：$G'$ 在很宽频率上大于 $G''$（弹性主导），且几乎与 $\omega$ 无关；**溶液**：$G'$ 与 $G''$ 在 $\omega = 1/\tau$ 附近交叉。一测便知是「弹」还是「黏」，这是质检、配方、生物组织表征的每日工具。

## 5 剪切变稀、屈服应力与缠结

真实软物质的黏度几乎总是**随剪切速率变化**，这已超出线性黏弹性：

**剪切变稀（shear thinning）**：聚合物溶液、乳液、牙膏的黏度随 $\dot\gamma$ 增大而下降，经验上 $\eta \propto \dot\gamma^{\,n-1}$（$n \lt  1$）。直觉：剪切把分子链拉伸定向、把絮凝团拆散，流动阻力变小。番茄酱「越晃越稀」就是典型。<span class="marginnote">少数体系反而<strong>剪切增稠</strong>（$\eta$ 随 $\dot\gamma$ 上升）：浓玉米淀粉浆越搅越硬，甚至能「跑」起来。机理是颗粒在剪切下被迫形成「水合固体团块」——这是浓悬浮液在阻塞线附近的临界响应，把第7篇的阻塞物理和流变学直接连在一起。</span>

**屈服应力（yield stress）**：某些材料（牙膏、发胶、混凝土浆）存在一个临界应力 $\sigma_y$：应力低于 $\sigma_y$ 时几乎不流动（像固体），超过后才开始流动。这类**屈服应力流体（Bingham 流体）**让牙膏「挤得动、立得住」。

聚合物熔体还有专属的**缠结**动力学：链太长时会互相打结，运动只能像蛇一样沿着「管子」蠕动——这就是 de Gennes 与 Doi–Edwards 的**蛇行（reptation）模型**。它预言熔体的最长弛豫时间 $\tau \propto N^3$，黏度 $\eta \propto N^3$——分子量翻倍，黏度涨约 8 倍，与实验惊人吻合。<span class="marginnote">缠结意味着聚合物的黏度对分子量极其敏感：$\eta \propto N^{3}$（分子量 < $M_e$ 时为 $\propto N$）。工业上通过控制分子量分布来调配塑料与橡胶的加工性能，正是吃这条幂律的红利。管模型也给第7篇的玻璃化一个分子图像：链在管里蛇行的时间，就是黏度发散的时间。</span>

### 数值一瞥：读一张 $G'$、$G''$ 曲线

用一组具体数字把第4节的结论钉死。取一个 Maxwell 材料：$G = 100\,\mathrm{Pa}$、$\eta = 10\,\mathrm{Pa\!\cdot\!s}$，则弛豫时间 $\tau = \eta/G = 0.1\,\mathrm{s}$。代入 $G'(\omega)$、$G''(\omega)$：

| 频率 | $\omega\tau$ | $G'$ | $G''$ | 读法 |
| --- | --- | --- | --- | --- |
| $\omega = 0.1$ rad/s（慢搅） | 0.01 | ≈ 0.01 Pa | ≈ 1 Pa | $G'' \gg G'$，像液体 |
| $\omega = 10$ rad/s（快搅） | 1 | ≈ 50 Pa | ≈ 50 Pa | 交叉点，黏弹并存 |
| $\omega = 100$ rad/s（猛砸） | 10 | ≈ 99 Pa | ≈ 10 Pa | $G' \gg G''$，像固体 |

同一块材料，**慢搅是水、快砸是橡皮**——这张表就是第1节德博拉数在实验室里的化身。工业质检里「扫一条 $G'$–$G''$ 频率谱」之所以能替代「用手摸」，正是因为它把「像不像固体」量化成了一对数字。

### 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| 流变学 | rheology | 研究材料在外力下流动与变形的学科 |
| 应力 | stress | 单位面积上的力，剪切应力 $\sigma = F/A$ |
| 应变 | strain | 单位厚度的位移，$\gamma = \Delta x/d$ |
| 应变率 | strain rate | 应变的速率，$\dot\gamma = v/d$ |
| 黏度 | viscosity | 应力与应变率之比，$\eta = \sigma/\dot\gamma$ |
| 剪切模量 | shear modulus | 应力与应变之比，$\sigma = G\gamma$ |
| 德博拉数 | Deborah number | 弛豫时间与观察时间之比 $\mathrm{De} = \tau/t_{\mathrm{obs}}$ |
| 黏弹性 | viscoelasticity | 同时具有黏性与弹性响应的材料行为 |
| 储能模量 | storage modulus | 与应变同相的模量 $G'$，代表弹性储能 |
| 损耗模量 | loss modulus | 与应变差相位的模量 $G''$，代表黏性耗散 |
| 剪切变稀 | shear thinning | 黏度随剪切速率增大而下降，$\eta \propto \dot\gamma^{n-1}$ |
| 屈服应力 | yield stress | 超过后才开始流动的临界应力 $\sigma_y$ |

## 6 小结

- 流变学用**应力、应变、应变率**与**本构方程**描述「流动与变形」；德博拉数 $\mathrm{De} = \tau/t_{\mathrm{obs}}$ 判定像液体还是像固体。
- 理想端：弹性 $\sigma = G\gamma$（弹簧，储能量）、黏性 $\sigma = \eta\dot\gamma$（阻尼器，耗能量）。
- **Maxwell 模型**（串联）描述应力弛豫；**Kelvin–Voigt 模型**（并联）描述蠕变恢复；真实材料用弛豫时间谱。
- **动态模量** $G'(\omega)$、$G''(\omega)$：$G'$ 储能有、$G''$ 耗能有，低频黏性、高频弹性；$\tan\delta = G''/G'$。
- 非线性三件套：**剪切变稀**（$\eta \propto \dot\gamma^{n-1}$）、**剪切增稠**、**屈服应力** $\sigma_y$。
- 聚合物缠结：**reptation 管模型**预言 $\tau \propto N^3$、$\eta \propto N^3$