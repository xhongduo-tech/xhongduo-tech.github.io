---
title: 麦克斯韦方程组与守恒律（位移电流、坡印廷定理、电磁张量）
date: 2026-08-07
---

# 麦克斯韦方程组与守恒律（位移电流、坡印廷定理、电磁张量）

<div class="epigraph">
<p>有了这些方程，从那时起，电学与磁学的发展便由数学的光芒照亮了。</p>
<footer>—— 海因里希 · 赫兹（Heinrich Hertz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第一章，Griffiths《电动力学导论》第 7–8 章 ｜ 2026-08-07</p>
</div>

## 为什么从麦克斯韦方程组开始

前四节像是搭积木：库仑定律、高斯定理、安培环路定理、法拉第定律，各自成形却互不联络。麦克斯韦在 1861–1865 年做的，是把四块积木拼成一座桥——他先发现安培环路定理与电荷守恒矛盾，于是补上**位移电流**；拼好后的四方程在无源区自然导出一个波动方程，波速恰是光速。**他由此断言：光就是电磁波。**这是物理学史上第一次有人把一个宏观实验定律与另一个宏观实验定律焊接成统一的场论，也是「从极限到大模型」这条主线上，理论物理第一次达到「用四条方程概括一个学科」的巅峰。<span class="marginnote">四方程不是「四件事」，而是「一个电磁场的四种约束」：两条源方程（电荷/无磁荷）决定场的散度，两条场方程（感应/位移电流）决定场的旋度。散度管「源」，旋度管「传播」。</span>

## 1 位移电流：麦克斯韦的补丁

安培环路定理 $\nabla\times\mathbf{B}=\mu_0\mathbf{J}$ 有一个隐藏矛盾：对两边取散度，左边恒为零，右边却是 $\mu_0\nabla\cdot\mathbf{J}$。由电荷守恒 $\nabla\cdot\mathbf{J}=-\partial\rho/\partial t$，只有当 $\partial\rho/\partial t=0$（静场）时等式才成立——**非稳态下安培环路定理与电荷守恒打架**。

麦克斯韦的解法是给 $\mathbf{J}$ 加一项「位移电流」：

$$\nabla\times\mathbf{B} = \mu_0\left(\mathbf{J} + \frac{\partial\mathbf{D}}{\partial t}\right)$$

把 $\partial\mathbf{D}/\partial t$ 看作一种「电流」，对两边取散度：$\nabla\cdot(\partial\mathbf{D}/\partial t) = \partial\rho/\partial t$（由 $\nabla\cdot\mathbf{D}=\rho$），正好抵消极了 $\nabla\cdot\mathbf{J}$。矛盾消解。<span class="marginnote">最直观的检验是给平行板电容充电：导线里电流流入，板间却没有电荷流动，但板间电场在增强——「变化的电场」就是板间的位移电流。位移电流虽然不移动电荷，照样产生磁场，所以电磁波能在真空里传播。</span>

**辨析｜易错点：** 位移电流 $\partial\mathbf{D}/\partial t$ 不是真的电流，它不迁移电荷、不产热；但在「产生磁场」这件事上它与传导电流平权。真空中的位移电流正是「变电场生磁场」那句口诀的数学化身。

麦克斯韦本人把这个想法推向顶点的论文，是 1865 年那篇著名的《电磁场的动力学理论》（A Dynamical Theory of the Electromagnetic Field）。他在文中明确写道：「光本身是一种以波的形式传播的电磁扰动，按电磁学定律穿过场。」——**这句话把光学整个吞并进了电磁学**。而他用以承载这一切的「以太」，几十年后被迈克耳孙—莫雷实验否定，取而代之的是爱因斯坦的狭义相对论——麦克斯韦方程组非但没有倒下，反而成了相对论诞生的第一块基石。

## 2 麦克斯韦方程组：四方程的完整表述

合并所有成果，得到真空中（含源）的**麦克斯韦方程组（Maxwell's equations）**：

$$\nabla\cdot\mathbf{E} = \frac{\rho}{\varepsilon_0}, \qquad \nabla\cdot\mathbf{B} = 0, \qquad \nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}, \qquad \nabla\times\mathbf{B} = \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial\mathbf{E}}{\partial t}$$

在介质中改用 $\mathbf{D}$、$\mathbf{H}$：$\nabla\cdot\mathbf{D}=\rho_f$，$\nabla\cdot\mathbf{B}=0$，$\nabla\times\mathbf{E}=-\partial\mathbf{B}/\partial t$，$\nabla\times\mathbf{H}=\mathbf{J}_f+\partial\mathbf{D}/\partial t$。<span class="marginnote">四条方程的顺序值得背熟：第一、二条管「源」（有电荷、无磁荷），第三、四条管「变化」（感应、位移电流）。前两条静态、后两条动态；前两条定「多少」，后两条定「怎么变」。</span>

四条方程各有积分形式与微分形式，合起来是「通量/环量 × 源/变化」的完整矩阵：

| 名称 | 微分形式 | 积分形式 | 含义 |
| --- | --- | --- | --- |
| 高斯定律 | $\nabla\cdot\mathbf{E}=\rho/\varepsilon_0$ | $\oint\mathbf{E}\cdot\mathrm{d}\mathbf{S}=Q/\varepsilon_0$ | 电荷是电场的源 |
| 无磁荷 | $\nabla\cdot\mathbf{B}=0$ | $\oint\mathbf{B}\cdot\mathrm{d}\mathbf{S}=0$ | 无磁单极子 |
| 法拉第定律 | $\nabla\times\mathbf{E}=-\partial\mathbf{B}/\partial t$ | $\oint\mathbf{E}\cdot\mathrm{d}\mathbf{l}=-\mathrm{d}\Phi_B/\mathrm{d}t$ | 变磁场生电场 |
| 安培—麦克斯韦 | $\nabla\times\mathbf{B}=\mu_0(\mathbf{J}+\varepsilon_0\partial\mathbf{E}/\partial t)$ | $\oint\mathbf{B}\cdot\mathrm{d}\mathbf{l}=\mu_0(I+\varepsilon_0\,\mathrm{d}\Phi_E/\mathrm{d}t)$ | 电流与变电场生磁场 |

这张表值得反复看：微分形式说「局部、点上」，积分形式说「整体、围线/围面」，两者由高斯散度定理与斯托克斯定理互连。**做题时，对称性高用积分形式，局部行为问微分形式。**

从方程组能直接推出电磁波：在无源区（$\rho=0,\mathbf{J}=0$）对第四式再取旋度，用矢量恒等式 $\nabla\times\nabla\times\mathbf{E}=\nabla(\nabla\cdot\mathbf{E})-\nabla^2\mathbf{E}$，代入第一式得**波动方程**：

$$\nabla^2\mathbf{E} = \mu_0\varepsilon_0\frac{\partial^2\mathbf{E}}{\partial t^2}, \qquad v = \frac{1}{\sqrt{\mu_0\varepsilon_0}} = 3.00\times10^8\ \mathrm{m/s}$$

波速恰好等于光速 $c$——**「光是一种电磁波」这个断言，纯粹从理论推出来，再被实验证实**。赫兹在 1888 年用火花隙放电产生了第一道人为电磁波并测出波长，为麦克斯韦方程画上最后的句号。

## 3 坡印廷定理：能量如何在空间流动

电磁场携带能量，能量守恒的局域形式是**坡印廷定理（Poynting's theorem）**。从麦克斯韦方程组出发，用 $\mathbf{E}\cdot(\nabla\times\mathbf{H})$ 减去 $\mathbf{H}\cdot(\nabla\times\mathbf{E})$，结合矢量恒等式与能量密度 $u=\frac12(\mathbf{E}\cdot\mathbf{D}+\mathbf{B}\cdot\mathbf{H})$，得到：

$$\frac{\partial u}{\partial t} + \nabla\cdot\mathbf{S} = -\mathbf{J}\cdot\mathbf{E}, \qquad \mathbf{S} = \mathbf{E}\times\mathbf{H}$$

**坡印廷矢量（Poynting vector）** $\mathbf{S}$ 是单位时间垂直穿过单位面积的能量流，方向就是能量流动的方向——对电磁波，它沿传播方向。对平面电磁波，时间平均的能量流密度 $\langle S\rangle = \frac12\varepsilon_0 c E_0^2$——它正比于振幅平方，这个关系在光学里就是「光强正比于 $E^2$」，也是天线辐射功率计算的出发点。<span class="marginnote">坡印廷矢量是 1884 年坡印廷（J. H. Poynting）与赫维赛德（Oliver Heaviside）几乎同时导出的。它的意义在于：能量不是在「源和受体」之间瞬间传递，而是<strong>穿过中间的空间</strong>——这正是「场」的本体论地位：场不只是方便的计算工具，它承载能量与动量。</span>

右边的 $-\mathbf{J}\cdot\mathbf{E}$ 是场对电荷做功的功率密度：正号表示电荷从场吸收能量（如电阻发热），负号表示电荷向场释放能量（如电池对外做功）。

**一个经典算例**：直流传输的同轴线缆（内导体半径 $a$、外导体半径 $b$，电流 $I$、内外电压 $V$）。内部电磁场：磁场绕内导体 $B=\mu_0I/(2\pi r)$，径向电场 $E=V/(r\ln(b/a))$，坡印廷矢量 $\mathbf{S}=\mathbf{E}\times\mathbf{H}$ 沿轴向。对横截面求 $\mathbf{S}$ 的面积分，恰好得到输运功率 $P=VI$——**能量不是「沿着导线内壁流」的，而是穿过导线的绝缘介质空间流动的**。这个结果彻底颠覆「能量从导线里传输」的直觉，是坡印廷矢量最具教育意义的一课。

## 4 动量守恒与电磁张量

电磁场不仅有能量，还有**动量**：动量密度

$$\mathbf{g} = \varepsilon_0\mu_0\mathbf{S} = \frac{\mathbf{S}}{c^2}$$

光照射物体表面会传递动量，这就是**光压（radiation pressure）**。太阳光在完全吸收表面上产生的压强约 $4.7\times10^{-6}\ \mathrm{Pa}$——微弱，却是太阳帆航天器（如 Breakthrough Starshot 设想的光帆）的驱动力，也是「彗尾总是背向太阳」的原因之一。若表面完全反射，光压加倍（光子动量翻转），这正是一面理想镜面比黑面多受一倍的压强来源——**反射、吸收对光压的差别，是光帆设计中「选材料」的物理依据**。<span class="marginnote">光压在宏观上微弱，在微观上决定原子运动：激光冷却把原子速度降到接近零、再「落」成玻色—爱因斯坦凝聚（BEC）——1997 年诺贝尔奖给了激光冷却与捕获原子。光子的动量 $p=E/c$ 正是爱因斯坦从「光量子」假设推出来的。</span>

动量的完整守恒律涉及**麦克斯韦应力张量（Maxwell stress tensor）** $\mathbf{\overline{T}}$：场对体积内物体的总力等于应力张量沿表面的积分，$F_i = \oint \sum_j T_{ij}\,\mathrm{d}S_j$。电磁场的能量、动量、应力三者构成「场的力学」——电场力、磁压、光压全是这一套的实例。

电磁场的**角动量**同样守恒，而且藏着一个著名的「费曼盘悖论」（Feynman's disk paradox）：一个不带电的绝缘圆盘上挂着通电螺线管，当螺线管断电、磁场消失时，圆盘竟然会转动起来——外表看「没有外力」，动量守恒好像被打破。答案是：磁场消失前的静电能与磁场能中寄存着角动量，断电后它被释放成机械转动。**电磁场不是旁观者，它是实实在在拥有能量、动量、角动量的物理实体**——这正是把场当作「物质」而非「计算工具」的底气。

## 5 公式解析：从四方程到波动方程

波动方程是从四方程「约化」出来的，拆开看每步在干什么。

- **第一步，取旋度**：对法拉第定律 $\nabla\times\mathbf{E}=-\partial\mathbf{B}/\partial t$ 再取旋度：$\nabla\times(\nabla\times\mathbf{E}) = -\partial(\nabla\times\mathbf{B})/\partial t$。
- **第二步，代入位移电流**：无源区 $\nabla\times\mathbf{B}=\mu_0\varepsilon_0\partial\mathbf{E}/\partial t$，代入得 $\nabla\times\nabla\times\mathbf{E} = -\mu_0\varepsilon_0\partial^2\mathbf{E}/\partial t^2$。
- **第三步，用矢量恒等式**：$\nabla\times\nabla\times\mathbf{E} = \nabla(\nabla\cdot\mathbf{E}) - \nabla^2\mathbf{E}$，无源区 $\nabla\cdot\mathbf{E}=0$，得 $\nabla^2\mathbf{E} = \mu_0\varepsilon_0\partial^2\mathbf{E}/\partial t^2$。<span class="marginnote">这里每一步都是「把场的空间变化翻译成时间变化」：散度管源、旋度管传播、矢量恒等式连接两者。到《平面电磁波》一篇，这个波动方程的解会被写成行波 $\mathbf{E}(\mathbf{r},t)=\mathbf{E}_0 e^{i(\mathbf{k}\cdot\mathbf{r}-\omega t)}$。</span>
- **第四步，读出波速**：标准波动方程 $\nabla^2 f = \dfrac{1}{v^2}\dfrac{\partial^2 f}{\partial t^2}$ 给出 $v = 1/\sqrt{\mu_0\varepsilon_0}$。

**关键直觉**：麦克斯韦方程组里没有任何「波」的字眼，波动方程却自动跳出来——这说明波不是方程的「额外解」，而是方程组的必然。**位移电流与法拉第定律这两条「时间项」只要同时存在，电磁波就无法避免。**

值得顺带记住的是：$c = 1/\sqrt{\mu_0\varepsilon_0}$ 里的两个常数在 SI 中都被「人为定义」了（$\mu_0 = 4\pi\times10^{-7}$ 精确，$\varepsilon_0$ 由 $c$ 推出），所以 $c$ 的出现不是巧合，而是这套单位制与真空的「电磁本性」之间的契约。更深一层，精细结构常数 $\alpha = e^2/(4\pi\varepsilon_0\hbar c) \approx 1/137$ 把电荷、作用量子与光速捆在一起，至今是物理学最神秘的「纯数字」之一。

## 6 从方程到现代物理

麦克斯韦方程组的意义远超「电学工具」，它奠定了整个现代物理的几条路线：

- **光与光学**：折射、反射、偏振、色散全部纳入电磁框架，催生了光学工程与微波技术（见后文《平面电磁波》与《电磁波在界面的行为》）。
- **狭义相对论**：四方程的洛伦兹不变性引导爱因斯坦 1905 年建立狭义相对论；电磁场可改写为协变形式——**电磁场张量** $F^{\mu\nu}$ 把 $\mathbf{E}$ 与 $\mathbf{B}$ 拼成一个 4×4 反对称矩阵，统一描述电场磁场在不同参考系间的变换。这条线在《狭义相对论与电动力学协变形式》一节展开。
- **规范场论**：位移电流与电荷守恒咬合的「规范结构」，在量子层面升级为 U(1) 规范理论；弱力与强力的规范理论（SU(2)、SU(3)）沿同一逻辑建立——**整个粒子物理标准模型是麦克斯韦方程组的嫡系后代**。<span class="marginnote">「对称性 → 守恒律 → 相互作用」这条链（诺特定理 + 规范原理）是 20 世纪物理的主干。从麦克斯韦到电弱统一（格拉肖、温伯格、萨拉姆，1979 年诺奖），再到 QCD 与标准模型，一线贯穿。</span>

**辨析｜易错点：** 麦克斯韦方程组是**线性**的——这是「叠加原理」在场论里的最高形式，也是电磁波、天线阵、干涉的基础。但「线性」只对真空与线性介质成立；强场下介质非线性（如等离子体、非线性晶体），方程组本身被修正，出现孤子、谐波等非线性现象——那是《非线性光学》与《等离子体物理》的领域。

麦克斯韦方程组统一出的电磁波谱，从极低频到伽马射线跨越约 20 个数量级：电力线工频 50 Hz（波长 6000 km）、调频广播约 100 MHz（波长 3 m）、Wi-Fi 2.4 GHz、可见光约 $5\times10^{14}$ Hz（波长约 600 nm）、X 射线与伽马射线在 $10^{18}$ Hz 以上。**「频率不同，物理相同」**——从收音机到 CT 扫描，都是同一条波动方程的不同频段，这是麦克斯韦方程组送给工程世界最慷慨的礼物。

工程世界对这套方程的「消费」也早已制度化：天线设计靠坡印廷矢量的辐射积分，微波电路靠边界条件解波动方程，光纤通信靠色散与偏振管理，电磁兼容（EMC）靠场与屏蔽的数值模拟——**每一行工程计算，本质上都是在求麦克斯韦方程组在某组边界条件下的一个具体解**。这就是为什么本专题要把接下来的篇幅全部投给「解方程」：平面波、界面、波导、辐射。

## 7 小结

- **位移电流** $\partial\mathbf{D}/\partial t$ 是麦克斯韦为调和「安培环路定理 vs 电荷守恒」补上的项，真空里照样产生磁场。
- **麦克斯韦方程组**四条：两条源方程（散度）、两条场方程（旋度），静态与动态各司其职。
- 无源区自动推出**波动方程**，波速 $c=1/\sqrt{\mu_0\varepsilon_0}$——光就是电磁波。
- **坡印廷定理**给出能量守恒：$u$ 的变化率 + $\mathbf{S}=\mathbf{E}\times\mathbf{H}$ 的散度 = 场的做功功率。
- 电磁场有**动量密度** $\mathbf{g}=\mathbf{S}/c^2$，光压是太阳帆、彗尾、激光冷却的物理根源。
- 四方程引出现代物理三条路线：光学、狭义相对论（$F^{\mu\nu}$ 张量）、规范场论（标准模型）。
- 电磁场拥有能量、动量、角动量，场不只是工具而是物理实体（同轴电缆能量在介质中流动、费曼盘悖论）。

在下一节，我们将把波动方程真正「解开」：**平面电磁波**——它如何传播、如何偏振、进入色散介质后相速与群速如何分裂——以及为什么一切无线电与光通信都建立在这组行波之上。
