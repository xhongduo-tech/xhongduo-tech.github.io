---
title: Lagrangian 子流形与锥
date: 2026-08-07
---

# Lagrangian 子流形与锥

<div class="epigraph">
<p>整部量子力学可以被重新表述为关于 Lagrangian 子流形之间的交集的理论。</p>
<footer>—— 艾伦 · 温斯坦（Alan Weinstein）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第3章；Cannas 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 Lagrangian 子流形开始

辛流形上最重要的子流形是那些「在辛形式下完全隐形」的：**Lagrangian 子流形**。它们维数取到最大（$n$ 维），却让辛形式在其上处处为零——位置空间、相空间里的「不动点集」、可积系统的纤维，全是 Lagrangian 子流形。如果说辛形式是「面积探测器」，Lagrangian 子流形就是「探测器读数恒为零的探针」。它们是哈密顿力学与辛几何的接口：哈密顿流的平衡点、Arnold 猜测、Floer 同调的边界条件，全部建立在 Lagrangian 子流形之上。这一篇还要引入一个几何对象——**Lagrangian 锥**（Maslov 锥），它是度量「两个 Lagrangian 子空间如何相交」的工具，也是 Maslov 指标的舞台。<span class="marginnote">温斯坦那句名言不是修辞：几何量子化里，经典态对应 Lagrangian 子流形，量子态对应其上的平坦线丛。到第3篇《几何量子化》你会再见到它。</span>

## 1 Lagrangian 子流形的定义

**迷向（isotropic）子流形**：$L \subset M$ 满足 $i^*\omega = 0$，即 $\omega_p(u, v) = 0$ 对所有 $u, v \in T_p L$、所有 $p \in L$。在 $L$ 上辛形式「恒为零」。

**Lagrangian 子流形**：维数为 $\frac{1}{2}\dim M = n$ 的迷向子流形。

为什么「$n$ 维」特殊？因为 $T_p L$ 若是迷向子空间，它的维数最多 $n$；取到最大值 $n$ 就是 **Lagrangian 子空间**（线性版本，见第1篇）。所以 Lagrangian 子流形就是「每点切空间都是 Lagrangian 子空间」的 $n$ 维子流形——**迷向到极限**。<span class="marginnote">维数上界是辛线性代数的事实：若 $L$ 迷向且 $\dim L &gt; n$，取 $u \in T_pL$ 与正交补，非退化性会让 $T_pL$ 与 $T_pL^{\perp_\omega}$ 相交非空，矛盾。所以 $n$ 是迷向子空间的「天花板」。</span>

类比排序（按辛正交补 $L^{\perp_\omega} = \{v : \omega(v,u)=0,\ \forall u \in L\}$）：

| 子空间类型 | 条件 | 维数 |
| --- | --- | --- |
| 迷向 isotropic | $L \subset L^{\perp_\omega}$ | $\le n$ |
| **Lagrangian** | $L = L^{\perp_\omega}$ | $= n$ |
| 余迷向 coisotropic | $L \supset L^{\perp_\omega}$ | $\ge n$ |
| 辛 symplectic | $L \cap L^{\perp_\omega} = \{0\}$ | 任意（$L$ 上 $\omega$ 非退化） |

**Lagrangian 是「自对偶」的子空间**：它等于自己的辛正交补。这个「自正交」性质让 Lagrangian 子流形的一切配对问题都变成「看它们如何相交」。

## 2 三个基本例子

**例1（位置空间）**：余切丛 $T^*Q$ 的**零截面** $Q_0 = \{(q, 0)\}$ 是 Lagrangian。因为 $\lambda = \sum p_i dq_i$ 在 $p = 0$ 处为零，$\omega_{\mathrm{can}} = -d\lambda$ 在 $Q_0$ 上为零。位置空间本身就是相空间的 Lagrangian 子流形。

**例2（闭 1-形式的图）**：对任意闭 1-形式 $\alpha$（$d\alpha = 0$），它的图

$$
\Gamma_\alpha = \{ (q, \alpha_q) : q \in Q \} \subset T^*Q
$$

是 Lagrangian。验证只需一步：$i^*\lambda = \alpha$（拉回刘维尔 1-形式），故 $i^*\omega = -d\alpha = 0$。**闭性条件 $d\alpha = 0$ 精确对应 Lagrangian 条件。** 这是后面公式解析的主角。

**例3（圆与环面）**：$S^2$ 上的赤道 $S^1$（面积形式的测地线）是 Lagrangian；$T^2 = S^1 \times S^1$ 上，两个因子 $S^1 \times \{*\}$、$\{*\} \times S^1$ 都是 Lagrangian。<span class="marginnote">在 $T^2$ 上，任意斜率为有理数的闭曲线都是 Lagrangian；斜率为无理数的稠密曲线则不是（它不闭）。「哪些闭曲线是 Lagrangian」在环面上与数论纠缠，这是低维辛几何的魅力。</span>

## 3 Lagrangian 锥与 Maslov 指标

Lagrangian 子空间全体构成 **Lagrangian 格拉斯曼流形**

$$
\Lambda(n) = \{ L \subset \mathbb{R}^{2n} : L \text{ 是 Lagrangian 子空间} \}
$$

固定一个参考 $L_0 = \mathbb{R}^n \times \{0\}$。**Maslov 锥（Maslov cone）** $\Sigma \subset \Lambda(n)$ 定义为所有与 $L_0$ **不正交**（即 $L \cap L_0 \neq \{0\}$）的 Lagrangian 子空间的集合。<span class="marginnote">为什么叫「锥」？因为 $\Lambda(n) = \mathrm{U}(n)/\mathrm{O}(n)$，而 $\Sigma$ 在其中是「奇异轨道」的并，呈锥状分层。它把 Lagrangian 子空间分成「与 $L_0$ 横截」与「不横截」两类，是交点数理论的边界对象。</span>

**Maslov 指标（Maslov index）** 测量一条 $L_0$-横截的 Lagrangian 路径「穿过 Maslov 锥多少次（带符号）」。对闭曲线 $\gamma: S^1 \to \Lambda(n)$，Maslov 指标 $\mu(\gamma) \in \mathbb{Z}$ 是 $\gamma$ 与 $\Sigma$ 的（带符号）交点数。它是 $H^1(\Lambda(n); \mathbb{Z}) \cong \mathbb{Z}$ 的生成元在 $\gamma$ 上的取值，是辛几何里少数「整数不变量」之一。<span class="marginnote">Maslov 指标在变分法里测量 Jacobi 场的共轭点个数，在量子化里给出半形式修正（$\hbar/2$ 的相移）。到第3篇《几何量子化》你会发现 WKB 近似里那个神秘的 $e^{i\pi/4}$ 因子就是 Maslov 指标的产物。</span>

## 4 公式解析：图的 Lagrangian 条件

**核心公式：**

$$
\Gamma_\alpha = \{ (q, \alpha_q) : q \in Q \} \subset T^*Q \text{ 是 Lagrangian} \iff d\alpha = 0
$$

这是「闭性 ⟺ Lagrangian」的精确翻译，三步拆解：

- **第一步，拉回刘维尔 1-形式**：$i: \Gamma_\alpha \to T^*Q$ 是包含，$s_\alpha: Q \to T^*Q$ 是截面 $q \mapsto \alpha_q$。则 $i^*\lambda = s_\alpha^*\lambda$。而 $\lambda = \sum p_i dq_i$，在截面 $p = \alpha(q)$ 上取值 $\alpha = \sum \alpha_i(q) dq_i$。所以 $i^*\lambda = \alpha$。
- **第二步，对拉回的辛形式求值**：$i^*\omega_{\mathrm{can}} = i^*(-d\lambda) = -d(i^*\lambda) = -d\alpha$。这里用到了「拉回与外微分可交换」：$i^* d = d i^*$。
- **第三步，判读**：$i^*\omega = 0$ 当且仅当 $d\alpha = 0$。所以 **$\Gamma_\alpha$ 是 Lagrangian ⟺ $\alpha$ 是闭 1-形式**。

**直觉总结：** 「图的切空间」与「水平方向」的夹角由 $d\alpha$ 度量；$d\alpha = 0$ 意味着图在每一点都与水平方向「辛正交」——这正是 Lagrangian 的几何含义。而 $d\alpha \neq 0$ 时，图是**辛**子流形（$\alpha$ 是辛形式，退化为辛叶），不是 Lagrangian——这是「退化」与「非退化」的分界。

**辨析｜易错点：** 不要以为「余切丛里任何 $n$ 维子流形都是 Lagrangian」。反例：图 $\Gamma_\alpha$ 对非闭的 $\alpha$ 就不是。特别地 $T^*Q$ 里「倾斜」的子流形一般不是 Lagrangian，只有「满足 $d\alpha = 0$ 的图」与「垂直纤维 $T_q^*Q$」等才是。垂直纤维 $T_q^*Q$ 也是 Lagrangian（$p$ 方向：$dq_i$ 在纤维上为零）。

## 5 Weinstein 邻域定理

Lagrangian 子流形最重要的结构定理是：

**Weinstein 邻域定理**：设 $L \subset (M, \omega)$ 是 Lagrangian 子流形。则存在 $L$ 在 $M$ 中的开邻域 $U$ 与 $L$ 在 $T^*L$ 中零截面的开邻域 $V$，以及辛同胚

$$
\varphi: (U, \omega) \longrightarrow (V, \omega_{\mathrm{can}})
$$

把 $L$ 映到零截面。换句话说，**每个 Lagrangian 子流形都在自己的余切丛里有个标准邻域**。<span class="marginnote">这是 Darboux 定理的推广：Darboux 说「点」附近是 $\mathbb{R}^{2n}$，Weinstein 说「Lagrangian 子流形」附近是 $T^*L$。证明同样用 Moser 技巧，把「$L$ 邻域上的辛形式」沿同痕拉回标准型。</span>

**推论**：两个 Lagrangian 子流形 $L_1, L_2$ 若都经过同一点 $p$ 且切空间相同，则它们有辛同胚的标准邻域——**Lagrangian 子流形的「局部模型」只有一种：余切丛的零截面**。这再一次印证辛几何「局部无差异、差异在整体」的主题：Lagrangian 子流形之间的差异（交点数、相对同调类、Floer 同调）全部是整体量。

**应用**：Gromov 的非压缩定理、Arnold 猜想、以及 Lagrangian Floer 同调都从这里出发——Weinstein 邻域把「两个 Lagrangian 的相交」局部化为「余切丛里两个图/截面的相交」，后者可以用 1-形式语言处理。

## 6 Lagrangian 子流形的位移与钳制

Lagrangian 子流形有个反直觉的性质：**它们往往「钳」在流形里，很难被哈密顿流移开**。

**位移能量（displacement energy）**：对 Lagrangian $L$，定义

$$
e(L) = \inf\{ d_H(\mathrm{id}, \phi) : \phi(L) \cap L = \emptyset \}
$$

即「把 $L$ 与自身完全移开所需的 Hofer 能量」。$e(L) > 0$ 的 Lagrangian 叫**不可位移（non-displaceable）**——用任意小能量的哈密顿流都移不走它。<span class="marginnote">这与「容量」是姊妹概念：容量是「球的嵌入障碍」，位移能量是「Lagrangian 的移动障碍」。$S^1 \subset \mathbb{C}$（赤道圆）是不可位移的——因为 $e(S^1) = \pi$（正数）。而 $\mathbb{R}^{2n}$ 里的非紧 Lagrangian（如直线）可被平移移开，$e = 0$。</span>

**Clifford 挠（Clifford torus）**：$T^n_{\mathrm{Cl}} = S^1 \times \cdots \times S^1 \subset \mathbb{C}^n$（每个坐标的赤道圆之积）是 $\mathbb{C}^n$ 的 Lagrangian 子环面。Cho 证明它不可位移：$e(T^n_{\mathrm{Cl}}) > 0$。**不可位移性由 Lagrangian Floer 同调（第4篇）检测**：$HF(L, L) \neq 0$（良定义时）⟹ $e(L) > 0$。这是「同调不变量 ⟹ 动力系统性质」的又一例。

**与温斯坦引言的呼应**：量子力学里，经典态是 Lagrangian 子流形；量子态间能否跃迁（交点数）由「它们是否可位移」决定——不可位移的 Lagrangian 对应「稳定的量子态」。温斯坦那句「量子力学是关于 Lagrangian 相交的理论」在此获得可操作的判据。

## 7 小结

- **Lagrangian 子流形**：$n$ 维迷向子流形，即 $i^*\omega = 0$ 且维数取到最大；等价于 $T_pL = T_pL^{\perp_\omega}$（自辛正交）。
- **谱系**：迷向 $\subset$ Lagrangian $\subset$ 余迷向；Lagrangian 是「自对偶」的特殊位置。
- **例子**：余切丛零截面、闭 1-形式的图、球面赤道、环面因子；**$d\alpha = 0$ ⟺ $\Gamma_\alpha$ 是 Lagrangian**。
- **Lagrangian 格拉斯曼流形 $\Lambda(n) = \mathrm{U}(n)/\mathrm{O}(n)$ 与 Maslov 锥**：度量横截性的边界对象，Maslov 指标是整数同调类。
- **Weinstein 邻域定理**：每个 Lagrangian 子流形在 $T^*L$