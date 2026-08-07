---
title: 外微分算子
date: 2026-08-07
---

# 外微分算子

<div class="epigraph">
<p>外微分把梯度、旋度、散度统一成一个算子——微积分的「求导」被一次性地发明了三次。</p>
<footer>—— 亨利 · 嘉当（Élie Cartan）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§7.4 ｜ 2026-08-07</p>
</div>

## 为什么从外微分开始

微分形式有了乘法（外积），现在给它「求导」——**外微分（exterior derivative）**

$$
d: \Omega^k(M) \longrightarrow \Omega^{k+1}(M)
$$

它把 $k$-形式映成 $(k+1)$-形式。外微分的惊人之处：**梯度、旋度、散度——向量微积分的三大微分算子——全是外微分在不同阶数下的化身。** 有了 $d$，微积分里零零散散的微分公式被统一成一个算子。<span class="marginnote">$d$ 是 Cartan 在 1899 年系统化的算子。它统一了三个「看似不同」的对象：0-形式的 $d$ 是梯度（$df$）、1-形式的 $d$ 是旋度（$d\omega$ 对应 $\nabla\times$）、2-形式的 $d$ 是散度（$d\eta$ 对应 $\nabla\cdot$）。「一次发明，三次使用」——向量微积分的三大算子原来是同一个算子的三个分身。</span>

## 1 外微分的定义

**定义（外微分）**：对 $k$-形式 $\omega$，外微分 $d\omega$ 是 $(k+1)$-形式，坐标下定义为

- **0-形式（函数）**：$df = \sum_i \frac{\partial f}{\partial x^i}\,dx^i$——就是全微分。
- **$k$-形式**：若 $\omega = \sum_I \omega_I\,dx^I$（$I = (i_1<\cdots<i_k)$），则
  $$
  d\omega = \sum_I d\omega_I \wedge dx^I = \sum_I \sum_j \frac{\partial \omega_I}{\partial x^j}\,dx^j \wedge dx^I
  $$

**重点：外微分 = 「对系数求偏导 + 用外积接上 $dx^j$」。** 它是坐标相关的定义，但**结果是坐标无关的**（可验证）——这正是「$d$ 良定义」的含义。<span class="marginnote">外微分的坐标无关性需要验证：换坐标下 $d\omega$ 相同。这个验证依赖「$d$ 与坐标变换兼容」（$d(f^*\omega) = f^*(d\omega)$，下一节拉回的交换性）。「坐标下定义、坐标无关」是流形上算子定义的典型模式——先局部定义，再验证拼合一致。</span>

## 2 外微分的核心性质

外微分满足两条决定性的性质：

1. **线性**：$d(\omega + \eta) = d\omega + d\eta$。
2. **Leibniz（graded）**：$d(\omega\wedge\eta) = d\omega\wedge\eta + (-1)^{\deg\omega}\,\omega\wedge d\eta$。
3. **幂零性（最关键）**：$d^2 = 0$，即
   $$
   d(d\omega) = 0 \qquad \forall\ \omega
   $$

**重点：$d^2 = 0$ 是外微分最深刻的性质。** 它说「求导两次得零」——任何形式的微分再微分必为零。这不是偶然，而是「混合偏导可交换」的直接后果（$d^2f$ 中的 $dx^i\wedge dx^j$ 与 $dx^j\wedge dx^i$ 抵消）。**$d^2 = 0$ 是「精确形式必闭」的代数根源，也是 de Rham 上同调（微分形式的拓扑理论）的地基。**<span class="marginnote">$d^2 = 0$ 在向量微积分里对应「旋度的梯度为零」（$\nabla\times\nabla f = 0$）与「散度的旋度为零」（$\nabla\cdot\nabla\times\mathbf{F}=0$）——两条「显然」的恒等式，本质都是 $d^2=0$。物理里「磁场无源」（$\nabla\cdot B=0$）就是「$F$ 闭」（$dF=0$）——$d^2=0$ 是电磁场理论的代数支柱。</span>

## 3 公式解析：为什么 $d^2 = 0$

把「两次求导为零」逐层拆解：

- **第一步，对函数**：$d(df) = d\Big(\sum_i \partial_i f\,dx^i\Big) = \sum_{i,j} \partial_j\partial_i f\,dx^j\wedge dx^i$。
- **第二步，对称偏导 + 反对称外积**：$\partial_j\partial_i f = \partial_i\partial_j f$（混合偏导可交换），但 $dx^j\wedge dx^i = -dx^i\wedge dx^j$（反对称）。于是 $(i,j)$ 项与 $(j,i)$ 项互为相反数，两两抵消：
  $$
  \sum_{i,j} \partial_j\partial_i f\,dx^j\wedge dx^i = 0
  $$
- **第三步，对任意形式**：$d(d\omega)$ 由「对系数求两次导」加上「外积交换的符号」——同类抵消，全部为零。**$d^2 = 0$ 是「混合偏导对称 × 外积反对称」的必然结果。**

**辨析｜易错点：** $d^2 = 0$ 是**逐点**成立的恒等式，但它并不意味着「闭形式都是精确形式」。$d\omega = 0$（闭）只能保证「局部」存在 $\eta$ 使 $\omega = d\eta$（Poincaré 引理），**整体上未必**——闭而不精确的形式正是上同调研究的对象。**「$d^2=0$」是「局部精确」的保证，不是「整体精确」的保证。**

## 4 三大算子的统一

外微分把向量微积分的三大微分算子统一：

| 算子 | 形式版本 | 经典向量版本 |
| --- | --- | --- |
| 梯度 | $d f$（0-形式→1-形式） | $\nabla f$ |
| 旋度 | $d\omega$（1-形式→2-形式） | $\nabla\times\mathbf{F}$ |
| 散度 | $d\eta$（2-形式→3-形式） | $\nabla\cdot\mathbf{F}$ |

**重点：$d$ 的一个算子，同时是梯度、旋度、散度。** 微积分教材里的三章内容，在外微分语言里是同一个算子的三个维度。这个统一让 Stokes 定理（第七篇）变得极简：$\int_\Omega d\omega = \int_{\partial\Omega}\omega$ 一个公式盖住牛顿-莱布尼茨、Green、Gauss、Stokes 四大公式。<span class="marginnote">记忆「$d$ 升级」：0-形式（函数）$d$ 成 1-形式（梯度），1-形式 $d$ 成 2-形式（旋度），2-形式 $d$ 成 3-形式（散度）。维度用完即止——三维空间里 3-形式再 $d$ 就是 0（没有 4-形式）。这个「阶梯」是向量微积分三大算子最清晰的图像。</span>

## 5 闭形式与精确形式

外微分引出微分形式的一对核心概念：

**定义（闭形式 / 精确形式）**：
- $\omega$ 是**闭的（closed）**，如果 $d\omega = 0$。
- $\omega$ 是**精确的（exact）**，如果存在 $\eta$ 使 $\omega = d\eta$。

**定理（$d^2=0$ 的推论）**：精确形式必闭（$d(d\eta) = 0$）。**闭形式未必精确**（整体）。两者的「差」用 **de Rham 上同调** 度量：

$$
H^k_{\text{dR}}(M) = \frac{\ker(d: \Omega^k\to\Omega^{k+1})}{\operatorname{im}(d: \Omega^{k-1}\to\Omega^k)} = \frac{\text{闭形式}}{\text{精确形式}}
$$

**重点：$H^k_{\text{dR}}(M)$ 是微分形式的「拓扑指纹」——闭形式中「真正不精确」的部分。** de Rham 定理：$H^k_{\text{dR}}(M) \cong$ 奇异上同调——**微分形式能「看见」流形的拓扑（洞的个数）**。这与 Gauss-Bonnet 精神相通：几何/分析对象编码拓扑。<span class="marginnote">de Rham 上同调是「微分形式版本的洞计数」：第 $k$ 个洞对应 $H^k_{\text{dR}}$ 的维数（Betti 数）。$S^2$ 上 $H^2 = \mathbb{R}$（球面的「洞」是二维的），环面 $T^2$ 有 $H^1 = \mathbb{R}^2$（两个洞）。拓扑数据分析（TDA）的持久同调，正是离散版 de Rham 上同调。</span>

### 例：旋度与散度都是外微分

在 $\mathbb{R}^3$ 里把外微分与向量微积分精确对应。设 1-形式 $\omega = P\,dx + Q\,dy + R\,dz$：

$$
d\omega = \Big(\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}\Big)dy\wedge dz + \Big(\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}\Big)dz\wedge dx + \Big(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\Big)dx\wedge dy
$$

**三个系数正是 $\nabla\times\mathbf{F}$ 的三个分量——$d\omega$ = 旋度。** 类似地，2-形式 $\eta = A\,dy\wedge dz + B\,dz\wedge dx + C\,dx\wedge dy$ 的 $d\eta = (\partial_x A + \partial_y B + \partial_z C)\,dx\wedge dy\wedge dz$——系数正是散度 $\nabla\cdot\mathbf{F}$。

**重点：$d$ 在 1-形式上是旋度、在 2-形式上是散度——一个算子自动「生成」向量微积分的全部微分算子。** 「旋度的散度为零」「梯度的旋度为零」在微分形式里都是同一个 $d^2 = 0$。记住了 $d^2=0$，就记住了向量微积分的两大「显然」恒等式。

### $d^2 = 0$ 与「精确形式」

$d^2 = 0$ 的直接推论是「精确形式必闭」：若 $\omega = d\eta$，则 $d\omega = d^2\eta = 0$。但**反过来说「闭形式必精确」只在局部成立**（Poincaré 引理），整体上闭而不精确的形式正是 de Rham 上同调的元素。

**重点：「精确 ⟹ 闭，闭 ⇏ 精确（整体）」——这个「非等价」缺口就是上同调。** 球面的面积形式闭（$d(K\,dA)=0$）但不精确（积分 $4\pi \neq 0$）。「闭而不精确 = 拓扑的洞」——$d^2=0$ 定义了「哪些形式是精确的」，上同调量化「哪些闭形式不是精确的」。一个恒等式，开启一门拓扑理论。

### 外微分与「方向」的直觉

$d\omega$ 度量「$\omega$ 沿所有方向的变化率的反对称组合」。对 1-形式 $\omega$，$d\omega(v,w)$ 是「沿 $v$ 与 $w$ 张成的小平行四边形的环量密度」——正是旋度的几何意义。

**「$d\omega$ = 环量密度 / 通量密度」——外微分把「变化」组织成「可积分的密度」。** 这就是为什么 Stokes 定理成立：内部的「密度积分」等于边界的「总量」。「$d$ 造密度，$\int$ 算总量」——外微分与积分的配对是微积分统一的基础。

## 6 小结

- **外微分** $d: \Omega^k\to\Omega^{k+1}$：对系数求偏导 + 外积接 $dx^j$。
- 性质：线性、graded Leibniz、**幂零 $d^2 = 0$**。
- $d^2=0$ 的根源：混合偏导对称 × 外积反对称。
- 三大算子统一：$d$ = 梯度（0→1）、旋度（1→2）、散度（2→3）。
- **闭形式**（$d\omega=0$）与**精确形式**（$\omega = d\eta$）；de Rham 上同调 = 闭/精确——微分形式看见拓扑。

在下一节，我们研究微分形式在映射下的「搬运」：**拉回**——如何把 $N$ 上的形式搬到 $M$ 上。
