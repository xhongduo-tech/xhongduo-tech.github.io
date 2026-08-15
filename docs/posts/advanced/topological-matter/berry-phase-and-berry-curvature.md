---
title: Berry 相位与 Berry 曲率
date: 2026-08-07
---

# Berry 相位与 Berry 曲率

<div class="epigraph">
<p>相位是量子力学的货币，而几何相位是这笔货币里最出乎意料的一笔财富。</p>
<footer>—— 迈克尔 · 贝里（Michael Berry，1984 年论文《量子绝热定理中的相位因子》开场白意译）</footer>
</div>

<div class="article-byline">
<p>第四级 · 拓扑物态与拓扑绝缘体 ｜ Bernevig &amp; Hughes, <em>Topological Insulators and Topological Superconductors</em>, §1.2 ｜ 2026-08-07</p>
</div>

## 为什么从 Berry 相位开始

上一节我们见到了 TKNN 公式里那个神秘的量——Berry 曲率 $\Omega(k)$，以及它在布里渊区上的积分（陈数）。但当时我们直接把它当「能带结构的曲率」用了，没问它从哪来。这一节补上这块基石：**Berry 相位**是量子系统在参数空间中缓慢绕行一圈后，波函数积累的一个「额外的」相位——它不来自动力学（能量 × 时间），而来自波函数的**几何**，因此又叫**几何相位（geometric phase）**。<span class="marginnote">「几何相位」这个名字点出了本质：它只依赖参数空间中走过的闭合路径的几何形状，与走这条路花了多长时间无关。这跟「动力学相位 $e^{-iEt/\hbar}$」恰好相反——后者完全由时间决定。一条闭合路径的几何形状，在数学上就是拓扑学研究的对象。</span>

Berry 相位的重要性怎么强调都不过分：它连接了量子力学与微分几何，是陈数、拓扑绝缘体、量子霍尔效应、反常霍尔效应的共同源头，甚至解释了分子光谱里的锥形交叉、光学里的 Pancharatnam 相位。这一节，我们从绝热定理出发，把 Berry 相位和 Berry 曲率一步步推出来。

## 1 绝热定理：慢到极致的演化

先回顾一条量子力学的基本定理。**绝热定理（adiabatic theorem）**：若哈密顿量 $H(\mathbf{R}(t))$ 随时间变化得足够慢，且初始时刻系统处在瞬时本征态 $\lvert n(\mathbf{R}(0)) \rangle$，则任意时刻系统都停留在同一个瞬时本征态 $\lvert n(\mathbf{R}(t)) \rangle$ 上——最多只差一个相位。<span class="marginnote">这里的 $\mathbf{R}$ 是「参数矢量」，可以是外磁场方向、晶格动量 $k$、分子的核坐标，或任何哈密顿量赖以改变的外部参量。绝热条件的具体判据是 $|\langle m|\dot H|n\rangle| \ll |E_m - E_n|^2/\hbar$：能级越接近、哈密顿量变得越快，绝热越容易失效。</span>

绝热演化保证「不跳能级」，但波函数到底带走了什么相位？设演化后

$$\lvert \psi(t) \rangle = \exp\!\left[-\frac{i}{\hbar}\int_0^t E_n(\mathbf{R}(t'))\,\mathrm{d}t' \right]\, e^{i\gamma_n(t)} \,\lvert n(\mathbf{R}(t)) \rangle$$

第一个指数是**动力学相位**，由能量对时间的积分给出。而 $e^{i\gamma_n}$ 是「多出来的」部分——它恰好是 Berry 相位。注意一个看似矛盾的事实：既然绝热定理说「停留在本征态」，为什么还会有非平凡的相位？因为**本征态本身带着参数依赖**：$\lvert n(\mathbf{R}) \rangle$ 在参数空间中绕一圈后，可以「转」出一个相位，就像指南针沿地球表面绕一圈回到原地后，指北针相对出发时的方向发生了偏转。

## 2 推导 Berry 相位

把薛定谔方程 $i\hbar \partial_t \lvert \psi \rangle = H \lvert \psi \rangle$ 代入上面的绝热解，并对时间求导，把 $\partial_t$ 作用在参数依赖的本征态上。关键的一步是投影：左边乘 $\langle n(\mathbf{R}) \rvert$，利用本征态的正交归一，得到相位满足

$$\dot{\gamma}_n = i \langle n(\mathbf{R}) \rvert \nabla_{\mathbf{R}} \lvert n(\mathbf{R}) \rangle \cdot \dot{\mathbf{R}}$$

对时间积分，闭合路径 $\mathcal{C}$ 一圈：

$$\gamma_n = i \oint_{\mathcal{C}} \langle n(\mathbf{R}) \rvert \nabla_{\mathbf{R}} \lvert n(\mathbf{R}) \rangle \cdot \mathrm{d}\mathbf{R} \equiv \oint_{\mathcal{C}} \mathbf{A}_n(\mathbf{R}) \cdot \mathrm{d}\mathbf{R}$$

这里定义了**Berry 联络（Berry connection）**

$$\mathbf{A}_n(\mathbf{R}) = i \langle n(\mathbf{R}) \rvert \nabla_{\mathbf{R}} \lvert n(\mathbf{R}) \rangle$$

注意 Berry 联络是一个实数矢量（$i$ 的因子保证实性），而 $\gamma_n$ 是可观测的物理相位。<span class="marginnote">联络（connection）是微分几何术语：它告诉你在流形上「平行移动」一个矢量时，坐标基怎么变。Berry 联络就是「波函数的相位如何随参数变化」的联络。这也解释了为什么要用 $\nabla_{\mathbf{R}}$ 而不是普通导数——本征态在参数空间不同点之间没有天然的「同一个」参考系。</span>

## 3 Berry 曲率：联络的旋度

Berry 联络 $\mathbf{A}_n$ 依赖规范（本征态乘以任意相位因子 $e^{i\phi(\mathbf{R})}$ 会改变 $\mathbf{A}_n$），但它的**旋度是规范不变的**：

$$\mathbf{\Omega}_n(\mathbf{R}) = \nabla_{\mathbf{R}} \times \mathbf{A}_n(\mathbf{R})$$

这个量叫 **Berry 曲率（Berry curvature）**。它与电磁场中「矢势的旋度是磁场」的结构完全同构：联络 $\mathbf{A}$ 像矢势，曲率 $\mathbf{\Omega}$ 像磁场。<span class="marginnote">这个类比不是巧合。U(1) 规范理论（电磁学）与 Berry 联络共享同一个数学结构：联络的曲率是规范不变的，且其「磁通」穿过闭合曲面必为 $2\pi$ 的整数倍。陈数之所以是整数，根源就在于此——布里渊区环面上 Berry 曲率的总通量被量子化为整数。</span>

Berry 曲率还有一个更实用的形式，把求和显式写出：

$$\mathbf{\Omega}_n(\mathbf{R}) = i \sum_{m \neq n} \frac{\langle n \rvert \nabla_{\mathbf{R}} H \rvert m \rangle \times \langle m \rvert \nabla_{\mathbf{R}} H \rvert n \rangle}{(E_n - E_m)^2}$$

**辨析｜易错点：** 这个求和式里，分母是能级差的平方。它揭示了一个关键事实：**Berry 曲率在能级简并点附近发散**。当两个能级在参数空间中某个点交叉（$E_m = E_n$），该点成为 Berry 曲率的「源」——在二维问题里，这个源是 Dirac 锥（锥形交叉）；在三维问题里，它是磁单极子。后面讲外尔半金属时，外尔点正是 Berry 曲率的单极子源。初学者常误以为 Berry 曲率只是「小修正」，实际上它在简并点附近可以任意大，这正是拓扑相变发生的场所。

## 4 公式解析：经典例子——自旋在旋转磁场中

最经典的 Berry 相位计算来自一个自旋 $1/2$ 系统：粒子自旋 $\mathbf{S}$，哈密顿量 $H = -\mathbf{B}(t) \cdot \mathbf{S}$，磁场方向 $\hat{\mathbf{n}}$ 缓慢绕行，在单位球面上扫出立体角 $\Omega$（磁场大小保持不变）。<span class="marginnote">这个例子常被称为「磁单极子里的自旋」：磁场方向在参数空间（单位球面）上绕行，就像参数空间里有个磁单极子（Berry 曲率源）位于球心。对自旋 $s$ 的粒子，Berry 相位 $\gamma = -s\Omega$，其中 $\Omega$ 是磁场方向扫过的立体角。</span>

推导分三步：

- **第一步，求瞬时本征态**：取沿 $\hat{\mathbf{n}} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ 方向的自旋本征态 $\lvert \uparrow_{\mathbf{n}} \rangle$，它是一个两分量旋量，显式写出来含有球坐标角度。
- **第二步，算 Berry 联络**：对 $\lvert \uparrow_{\mathbf{n}} \rangle$ 求梯度，得 $\mathbf{A} = i \langle \uparrow \rvert \nabla \rvert \uparrow \rangle$。直接计算会发现它正比于「方位角方向的单位矢量除以极角」，像一个绕着北极的漩涡。
- **第三步，积分得到立体角**：把联络沿闭合路径积分，$\gamma = \oint \mathbf{A} \cdot \mathrm{d}\mathbf{l} = \int_{\text{球面片}} \mathbf{\Omega}\cdot \mathrm{d}\mathbf{S} = -\tfrac12 \Omega$，其中 $\mathbf{\Omega} = \hat{\mathbf{n}}/(2|\mathbf{n}|^2)$ 是 Berry 曲率，它像点电荷场一样从「简并点」发出——这个简并点在 $\mathbf{B}=0$ 处。

这个结果给出一个漂亮的几何结论：**Berry 相位等于磁场方向扫过的立体角的一半**。当磁场方向扫过整个球面（$\Omega = 4\pi$），自旋 $1/2$ 的 Berry 相位是 $2\pi$ 的整数倍——波函数回到自身，这正对应「自旋绕着参数空间完整走一圈」的拓扑约束。

## 5 布洛赫态中的 Berry 相位：能带的几何

回到凝聚态物理。布洛赫定理告诉我们，周期势场中的电子波函数写成

$$\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r})$$

这里晶格动量 $\mathbf{k}$ 就是参数，布洛赫周期部分 $\lvert u_{n\mathbf{k}} \rangle$ 在动量空间（布里渊区）上演化。于是上面整套绝热几何可以直接搬过来：

$$\gamma_n = i \oint_{\mathcal{C}} \langle u_{n\mathbf{k}} \rvert \nabla_{\mathbf{k}} \lvert u_{n\mathbf{k}} \rangle \cdot \mathrm{d}\mathbf{k}$$

在动量空间中，Berry 曲率 $\Omega_n(\mathbf{k})$ 是「动量空间的磁场」，它在整个布里渊区上的通量给出陈数。**这就是上一节 TKNN 公式里 $C$ 的完整出身**：不是人为发明的积分，而是波函数几何结构在环面（布里渊区）上的必然产物。<span class="marginnote">对二维系统，布里渊区是环面 $\mathbb{T}^2$；Berry 曲率积分除以 $2\pi$ 得到陈数 $C$。对一维系统，能带是闭合的圈，Berry 相位退化为绕圈一圈的相位，对应「Zak 相位」——下一节 SSH 模型的核心工具。</span>

**辨析｜易错点：** 很多人以为 Berry 相位只对「闭合路径」有意义。确实，Berry 相位本身是闭合路径上的积分（规范不变）；但 Berry **曲率**是逐点定义的局域量，不依赖任何路径。两者的关系就像「磁通」与「磁场」：你可以谈论某个位置处的磁场（曲率），但只有积分成磁通（相位/陈数）才是规范不变的物理量。混淆「联络（规范依赖）」与「曲率（规范不变）」是最常见的误区。

### 一维的退化情形：Zak 相位

当系统只有一维（如聚合物链、光晶格中的一维能带），动量空间是圆周 $S^1$，Berry 相位退化为沿整条一维布里渊区绕一圈的相位，称为 **Zak 相位（Zak phase）**：

$$\gamma_n^{\text{Zak}} = i \int_{-\pi/a}^{\pi/a} \langle u_{nk} \rvert \partial_k \lvert u_{nk} \rangle \, \mathrm{d}k$$

Zak 相位是下一节 SSH 模型的核心主角：它在拓扑相（非平庸）取值 $\pi$，在平庸相取值 $0$。注意一维没有「环面」，所以 Zak 相位不是陈数，取值可以不是 $2\pi$ 的整数倍——它是 $\mathbb{Z}_2$ 型（只分 $0$ 与 $\pi$ 两类）的拓扑量。<span class="marginnote">Zak 相位的二值性来自一个约束：一维绝缘体在「幺正规范」下，Zak 相位被钉在 $0$ 或 $\pi$。这为「对称性保护的拓扑相」提供了一个最简模型——没有磁场、没有陈数，也能有拓扑，只要时间反演或手征对称在适当维度上保护它。</span>

### 为什么拓扑物态「需要」Berry 曲率

回到整数量子霍尔效应：如果 Berry 曲率恒为零，会怎样？由反常速度公式 $\dot{\mathbf{r}} = \nabla_k E_n - \dot{\mathbf{k}} \times \mathbf{\Omega}_n$，横向速度项消失，电子在电场下只有纵向漂移——就没有量子霍尔电导了。**非零 Berry 曲率是「横向无耗散输运」的根源**。这一句话把 Berry 几何、陈数、量子霍尔、反常霍尔、拓扑绝缘体全部串在了同一条线上。

## 6 Berry 相位的实验观测

Berry 相位不是纸上谈兵，它有清晰的实验足迹。最早的一类实验用**中子干涉仪**：中子束被劈成两束，一束经过缓慢旋转的磁场、另一束作为参考，两束重新汇合时的干涉条纹位移恰好对应 Berry 相位。因为中子干涉仪能直接读出相位差，这是最「干净」的测量方式。<span class="marginnote">1986 年，Bitter 与 Dubbers 用极化的中子束在旋转磁场中直接测到了 $\gamma = -\tfrac12 \Omega$，与理论精确符合。这是几何相位第一次在实验中被「肉眼看见」，比 Berry 的论文发表晚了两年。</span>

此后观测手段不断翻新：

**核磁共振（NMR）**：核自旋在射频场中演化，用脉冲序列实现参数空间的闭合回路，读出几何相位——测量精度高、可控性强。
**光子系统**：光的偏振态（Poincaré 球）充当自旋，用波片组合实现偏振方向的绕行，测出 Pancharatnam–Berry 相位。光子实验把 Berry 相位带进了经典光学，也启发了后来「拓扑光子学」这一整个分支。
**冷原子与光晶格**：超冷原子在人工规范场中演化，可以直接「画」出动量空间里的 Berry 曲率分布，甚至直接测量陈数。<span class="marginnote">2013 年左右，慕尼黑、苏黎世等小组在光晶格冷原子实验中，通过测量原子波包的异常速度（$\dot{\mathbf{r}} = \nabla_k E - \dot{\mathbf{k}} \times \mathbf{\Omega}$）直接映射出 Berry 曲率的空间分布。这里的 $\Omega$ 项正是「反常速度」，它让 Berry 曲率从「藏在波函数里的几何」变成了「可直接测量的动力学量」。</span>

这套「可测量性」至关重要：拓扑物态之所以能进实验室、进应用，前提是拓扑不变量背后藏着可观测的动力学后果——Berry 曲率通过反常速度直接改写电子的运动方程。

## 7 小结

- **Berry 相位**：绝热演化中波函数绕参数空间闭合路径一圈后，除动力学相位外额外积累的相位 $\gamma_n = \oint \mathbf{A}_n \cdot \mathrm{d}\mathbf{R}$，依赖路径的几何而非时间。
- **Berry 联络与曲率**：联络 $\mathbf{A}_n = i\langle n\rvert\nabla\rvert n\rangle$ 规范依赖，曲率 $\mathbf{\Omega}_n = \nabla \times \mathbf{A}_n$ 规范不变；曲率在能级简并点附近发散。
- **自旋例子**：自旋 $s$ 在旋转磁场中绕行立体角 $\Omega$，Berry 相位 $\gamma = -s\Omega$；自旋 $1/2$ 绕整球为 $2\pi$。
- **与电磁学的同构**：联络 ↔ 矢势，曲率 ↔ 磁场，闭合曲面通量量子化 ↔ 陈数为整数。
- **在凝聚态中的角色**：晶格动量 $\mathbf{k}$ 作为参数，Berry 曲率在布里渊区上的总通量给出陈数；它通过反常速度 $\dot{\mathbf{r}} = \nabla_{\mathbf{k}} E - \dot{\mathbf{k}} \times \mathbf{\Omega}$ 直接改写电子的运动方程，是量子霍尔、反常霍尔与拓扑绝缘体共享的源头。
- **一维退化**：Zak 相位在拓扑相取 $\pi$、在平庸相取 $0$，说明「没有陈数也能有拓扑」——只要对称性在低维提供保护，这就是对称性保护拓扑态的最小种子。
- **实验可测**：中子干涉、NMR、光子偏振与冷原子反常速度等路径都直接测到了 Berry 相位与曲率，拓扑不变量因此从来不是「纸上几何」。

在下一节，我们将第一次亲手「造」一个拓扑物态：沿 SSH 链看绕数如何在 $v/w$ 扫过 $1$ 时发生跳变、边界态如何随之生灭——**拓扑相变与边界态**。