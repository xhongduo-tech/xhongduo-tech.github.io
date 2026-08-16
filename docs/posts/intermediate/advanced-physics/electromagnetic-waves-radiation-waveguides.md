---
title: 电磁波与辐射（坡印廷矢量、偶极辐射、波导）
date: 2026-08-07
---

# 电磁波与辐射（坡印廷矢量、偶极辐射、波导）

<div class="epigraph">
<p>自牛顿以来，物理学中最深刻、最富有成效的变革，是麦克斯韦的电磁场理论。</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein），《麦克斯韦对物理实在概念发展的影响》，1931</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 郭硕鸿《电动力学》第四～五章 ｜ 2026-08-07</p>
</div>

## 为什么从电磁波与辐射开始

上一节搭好了麦克斯韦方程组的骨架，这一节让这个体系真正「活起来」：电磁波从方程组里被预言出来、被赫兹用实验证实、被无线电报、雷达、光纤、手机变成了人类文明的日常基础设施。但「波会在空中传播」只是故事的一半——**波从哪里来**（辐射）、**波被装进什么容器**（波导）才是本节的主角。电偶极辐射给出「电荷加速 → 能量以波的形式离场」的机制，波导给出「让波沿确定的通道前进」的工程约束。这节之后，电磁学就从「场的静态理论」彻底过渡到「场的动力学理论」。

## 1 平面电磁波：方程组的自洽解

在无源区（$\rho=0,\ \boldsymbol{J}=0$），对麦克斯韦方程组取旋度并整理，电场与磁场分别满足**波动方程**：

$$
\nabla^2\boldsymbol{E} = \frac{1}{c^2}\frac{\partial^2\boldsymbol{E}}{\partial t^2}, \qquad
\nabla^2\boldsymbol{B} = \frac{1}{c^2}\frac{\partial^2\boldsymbol{B}}{\partial t^2}, \qquad
c = \frac{1}{\sqrt{\mu_0\varepsilon_0}}
$$

**重点：波动方程的存在不是假设而是推论——只要麦克斯韦方程组成立，电磁扰动就必然以速度 $c$ 传播。** $c = 1/\sqrt{\mu_0\varepsilon_0}$ 由真空的两个常数算出，数值约 $2.998\times10^8\ \mathrm{m/s}$，与光速吻合——麦克斯韦由此断言「光是电磁波」。<span class="marginnote"><strong>实验里程碑</strong>：1887 年赫兹用火花放电产生的振荡电流在空间中激发出电磁波，用环形检波器在约一米外探测到，并测出波长与反射、折射、偏振等性质——「光是电磁波」从预言变成事实，直接开启了无线电时代。今天 5G 通信的毫米波、Wi-Fi 的 2.4 GHz 频段，都是这段历史的直系后代。</span>

平面简谐波的解形式为 $\boldsymbol{E} = \boldsymbol{E}_0 e^{i(\boldsymbol{k}\cdot\boldsymbol{r} - \omega t)}$，其中波矢 $k = \omega/c$，**波是横波**（$\boldsymbol{k}\cdot\boldsymbol{E}_0 = 0$），且 $\boldsymbol{E}$、$\boldsymbol{B}$、$\boldsymbol{k}$ 三者互相垂直，$|\boldsymbol{E}| = c|\boldsymbol{B}|$。<span class="marginnote"><strong>横波的直接后果</strong>：电场在传播方向上没有分量，这是「偏振」现象的基础——见《光的偏振与马吕斯定律》。而导体中的波则会因为趋肤效应（穿透深度 $\delta = \sqrt{2/(\mu_0\sigma\omega)}$）被急剧衰减，这正是电磁屏蔽的原理。</span>

## 2 坡印廷矢量：能量随波旅行

电磁波把能量从一个地方搬到另一个地方。单位时间垂直通过单位面积的能量由**坡印廷矢量（Poynting vector）**给出：

$$
\boldsymbol{S} = \frac{1}{\mu_0}\boldsymbol{E}\times\boldsymbol{B}, \qquad
\langle S\rangle = \frac{1}{2}\sqrt{\frac{\varepsilon_0}{\mu_0}}\,E_0^2
$$

**重点：坡印廷矢量的方向就是能量传播的方向，其大小是能流密度——对电磁波，平均能流正比于场强的平方。** 这一点与量子力学「光子能量正比于频率」并不矛盾：经典能流是大量光子的统计平均。辐射压 $P = \langle S\rangle/c$ 是坡印廷矢量在动量层面的双胞胎，动量密度 $\boldsymbol{g} = \boldsymbol{S}/c^2$ 则把「电磁场也有动量」说成定量语言——光帆推进、彗星尾的辐射压形变都源于此。<span class="marginnote"><strong>数值算例（Wi-Fi 信号）</strong>：手机基站发射功率约 40 W，传到 100 米处的能流密度约 $3\times10^{-4}\ \mathrm{W/m^2}$，场强约 $0.3\ \mathrm{V/m}$——经过 1/距离平方的衰减。反平方衰减是辐射（球面波）的指纹，也是通信系统预算的出发点。</span>

## 3 电偶极辐射：波从哪里来

静止电荷不辐射，匀速运动电荷不辐射，**加速运动的电荷才辐射**。最简单的辐射源是**振荡电偶极子（oscillating electric dipole）**：一个电荷 $q$ 沿固定轴以频率 $\omega$ 做简谐振动，等效于电偶极矩 $\boldsymbol{p}(t) = \boldsymbol{p}_0\cos\omega t$ 的振荡。其辐射场（远区）为

$$
\boldsymbol{E}(r,\theta,t) = \frac{\mu_0 p_0 \omega^2\sin\theta}{4\pi r}\cos(\omega t - kr)\,\hat{\boldsymbol{\theta}}, \qquad
\boldsymbol{B} = \frac{1}{c}\,\hat{\boldsymbol{r}}\times\boldsymbol{E}
$$

**重点：辐射场按 $1/r$ 衰减（球面波），随 $\omega^2$ 增大（高频更易辐射），且具有角分布 $\sin\theta$（沿偶极子轴向为零、赤道面最大）。** 这些特征是天线设计、无线电传播、乃至原子发光（原子中的电子在能级间跃迁时等效于振荡偶极子）的共同基础。

把天线尺寸 $l$ 与波长比较可以分两种情形：$l \ll \lambda$ 的是「电偶极近似」（本节公式适用），$l \sim \lambda$ 的短振子/半波天线则要考虑电流沿导体的分布——但所有情形下，辐射的方向性都源于「多单元辐射场在空间的相干叠加」，这与光学里「多缝衍射的角分布」是同一个数学结构（见《光栅衍射与 X 射线衍射》）。<span class="marginnote"><strong>为什么 $1/r$ 而非 $1/r^2$</strong>：库仑场与静磁场的感应场按 $1/r^2$、$1/r^3$ 衰减，只有加速电荷激发的辐射场按 $1/r$ 衰减——这意味着辐射场在远距离「胜出」，其余都被压制。能流密度 $S \propto 1/r^2$，对球面积分得总辐射功率为常数——能量守恒要求辐射场必须衰减得比感应场慢。</span>

## 4 公式解析：偶极辐射功率与角分布

辐射带走的能量由坡印廷矢量对包围源的球面积分得到。把 $E_\theta$ 代入 $\boldsymbol{S} = \boldsymbol{E}\times\boldsymbol{B}/\mu_0$，得辐射功率：

$$
P(t) = \oint \boldsymbol{S}\cdot\mathrm{d}\boldsymbol{A} = \frac{\mu_0 p_0^2\omega^4}{6\pi c}\cos^2\omega t, \qquad
\langle P\rangle = \frac{\mu_0 p_0^2\omega^4}{12\pi c}
$$

- **第一步，写角分布**：能流密度 $S(r,\theta) = \frac{1}{\mu_0 c}E_\theta^2 = \frac{\mu_0 p_0^2\omega^4\sin^2\theta}{16\pi^2 c r^2}\cos^2(\omega t - kr)$——角分布正比 $\sin^2\theta$。
- **第二步，对立体角积分**：$\oint \sin^2\theta\,\mathrm{d}\Omega = \frac{8\pi}{3}$，其中 $\mathrm{d}\Omega = \sin\theta\,\mathrm{d}\theta\,\mathrm{d}\phi$ 是立体角元。
- **第三步，代入**：$P(t) = \frac{\mu_0 p_0^2\omega^4}{16\pi^2 c}\cdot\frac{8\pi}{3}\cos^2\omega t = \frac{\mu_0 p_0^2\omega^4}{6\pi c}\cos^2\omega t$。
- **第四步，时间平均**：$\langle\cos^2\omega t\rangle = 1/2$，得 $\langle P\rangle = \frac{\mu_0 p_0^2\omega^4}{12\pi c}$——**辐射功率正比于 $\omega^4$**，即「高频更容易辐射」。<span class="marginnote"><strong>$\omega^4$ 的现实意义</strong>：这正是「为什么收音机用长波、手机用微波」的物理原因之一——同样的振荡，频率越高辐射越强、天线可以越小。高频辐射器（X 射线管、同步辐射光源）正是靠让电子猛烈加速来产生强辐射。</span>

## 5 波导与谐振腔：把波关进容器

辐射波在自由空间扩散，但很多应用要把波「关」起来、沿特定路径输送——这就是**波导（waveguide）**与**谐振腔（cavity）**。

**波导**：中空的金属管，电磁波在其中沿管轴传播，但横向被边界约束。关键概念是**截止频率（cutoff frequency）**：对矩形波导（宽 $a$、高 $b$），模式 $\mathrm{TE}_{mn}$ 的截止条件为 $f_c = \frac{c}{2}\sqrt{(m/a)^2 + (n/b)^2}$。频率低于截止频率的模式无法传播——波导是「高通滤波器」。<span class="marginnote"><strong>为什么需要波导</strong>：在微波频段（GHz 以上），普通同轴电缆损耗急剧增大，而空腔金属波导损耗低、功率容量大。雷达、卫星通信的馈线系统、粒子加速器（把微波能量注入加速腔）都离不开波导。光纤则是「介质波导」——靠全反射把光约束在芯层里，见《全反射》与《光学仪器》。</span>

**谐振腔**：把波导两端封闭，形成驻波，就得到谐振腔——微波段的「LC 谐振电路」。其谐振频率由尺寸决定，品质因数 $Q$ 远高于集总元件电路。<span class="marginnote"><strong>数值算例（微波炉）</strong>：家用微波炉工作在 2.45 GHz，腔体尺寸约 30 cm 量级，正是让波长 $\lambda = c/f \approx 12.2$ cm 的驻波能在腔内多模谐振的尺度。粒子的回旋加速器、激光器的光学腔，也是同一「驻波共振」思想的不同实现。</span>

## 6 数值算例：偶极天线与自由空间传播

把偶极辐射公式落到工程上，天线设计的一切都从这里生长。**半波偶极天线（half-wave dipole）**是标准例子：长度约 $\lambda/2$ 的金属杆，两端电流为零、中心电流最大，其辐射场角分布接近 $\sin\theta$，方向性系数 $D \approx 1.64$。

- **第一步，定频率与尺寸**：FM 广播 100 MHz，波长 $\lambda = c/f \approx 3$ m，半波偶极天线长约 1.5 m。
- **第二步，算功率密度**：发射功率 1 kW 时，赤道面 10 km 处能流密度 $S = P D/(4\pi r^2) \approx 1.64/(4\pi\cdot10^8) \approx 1.3\times10^{-9}\ \mathrm{W/m^2}$——接收天线只能捕获其中很小一部分。
- **第三步，算接收电压**：接收天线（半波偶极，有效面积 $A_e = \lambda^2 D/(4\pi) \approx 1.18\ \mathrm{m^2}$）收到的功率约 $1.5\times10^{-9}\ \mathrm{W}$，转换为几微伏的信号电压——正是收音机高增益放大的对象。
- **第四步，体会**：整个链路预算（发射 → 自由空间衰减 → 接收）就是「偶极辐射 + 坡印廷矢量 + 有效面积」三个公式的连锁。5G 的波束赋形（相控阵）本质上是多个偶极元的辐射相干叠加，让 $\sin\theta$ 角分布被「整形」成指向特定用户的窄波束。

**三种「波载体」的对照**，把本节的图景收拢：

| 载体 | 约束机制 | 传播方式 | 频段与用途 |
| --- | --- | --- | --- |
| 自由空间 | 无约束 | 球面波扩散，$1/r$ 衰减 | 广播、雷达、卫星通信 |
| 金属波导 | 边界反射 | 离散模式，有截止频率 | 微波馈线、加速器 |
| 光纤 | 全反射 | 单/多模导波 | 光通信主干网 |

**重点：辐射讲「波如何离开源」，波导与光纤讲「波如何被约束住」——两者一放一收，共同构成电磁波的应用全景。** 现代通信网络正是「自由空间 + 波导 + 光纤」三种载体协同工作的结果。

## 7 术语速查表

| 术语 | 公式 | 要点 |
| --- | --- | --- |
| 波动方程 | $\nabla^2\boldsymbol{E} = \frac{1}{c^2}\partial^2\boldsymbol{E}/\partial t^2$ | $c = 1/\sqrt{\mu_0\varepsilon_0}$ |
| 平面简谐波 | $\boldsymbol{E}_0 e^{i(\boldsymbol{k}\cdot\boldsymbol{r}-\omega t)}$ | 横波，$\boldsymbol{k}\perp\boldsymbol{E}$ |
| 坡印廷矢量 | $\boldsymbol{S}=\boldsymbol{E}\times\boldsymbol{B}/\mu_0$ | 能量流方向与大小 |
| 辐射角分布 | $S\propto\sin^2\theta$ | 偶极轴向为零 |
| 辐射功率 | $\langle P\rangle = \mu_0 p_0^2\omega^4/(12\pi c)$ | 正比 $\omega^4$ |
| 截止频率 | $f_c = \frac{c}{2}\sqrt{(m/a)^2+(n/b)^2}$ | 波导高通特性 |

## 8 小结

- **电磁波是麦克斯韦方程组的必然推论**，速度 $c = 1/\sqrt{\mu_0\varepsilon_0}$ 与光速一致——「光是电磁波」由赫兹实验证实。
- 电磁波是**横波**，$\boldsymbol{E}\perp\boldsymbol{B}\perp\boldsymbol{k}$，$|\boldsymbol{E}| = c|\boldsymbol{B}|$。
- **坡印廷矢量** $\boldsymbol{S}$ 描述能量流动；辐射压 $P = \langle S\rangle/c$ 是场的动量表现。
- **加速电荷辐射**：振荡偶极子的辐射场按 $1/r$ 衰减、正比 $\omega^2$、角分布 $\sin^2\theta$，功率正比 $\omega^4$。
- **波导**有截止频率（高通特性），**谐振腔**靠驻波储存能量——两者是微波与光学工程的基础。
- 链路预算（发射 → 自由空间衰减 → 接收）是「偶极辐射 + 坡印廷矢量 + 有效面积」三公式的连锁，天线阵用相干叠加把 $\sin\theta$ 角分布整形为定向波束。
- 电磁波的能量与动量（坡印廷矢量、动量密度、辐射压）证明「场」是独立物理实在，为爱因斯坦质能关系与量子化（光子）铺路。

在下一节，我们将离开经典场论，进入现代物理的深处——用量子力学的语言重新描述世界：**量子力学公理体系**。
