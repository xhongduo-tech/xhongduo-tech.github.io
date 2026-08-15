---
title: SU(2)：自旋与同位旋
date: 2026-08-07
---

# SU(2)：自旋与同位旋

<div class="epigraph">
<p>这种经典上无法描述的「二值性」，是电子内禀的量子力学性质。</p>
<footer>—— 沃尔夫冈 · 泡利（Wolfgang Pauli, <i>On the Connection between the Completion of the Electron Groups in an Atom with the Complex Structure of Spectra</i>, 1925）</footer>
</div>

<div class="article-byline">
<p>第二级 · 物理学中的群论 ｜ Howard Georgi, <i>Lie Algebras in Particle Physics</i> 第1–2章 ｜ 2026-08-07</p>
</div>

## 为什么从 SU(2) 开始

上一节的 $SO(3)$ 讲得很顺：整数 $l$ 的不可约表示给出轨道角动量。但它漏了一整类最重要的量子数——**半整数**。银原子通过非均匀磁场会分裂成两束（斯特恩-盖拉赫实验），电子的磁矩只有两种取值；氢原子光谱的精细结构也要求电子携带一个「半个」的角动量。这个凭空多出来的量子自由度就是**自旋（spin）**。自旋不能用 $SO(3)$ 的表示描述——它需要更大的群 $SU(2)$。本节还会看到，同一套数学被粒子物理借去描述**同位旋（isospin）**：质子与中子的「对称性」、介子的分类，用的都是 $SU(2)$。<span class="marginnote">$SU(2)$ 是理解「为什么物理世界有旋量」的关键。它把第 5 篇的 $SO(3)$ 扩展为双覆盖，也让「自旋」这种没有经典对应的自由度有了精确的数学位置。学完本节，角动量群论就完整了——$j$ 取整数或半整数，全部来自 $SU(2)$。</span>

## 1 自旋：电子内禀的「二值性」

1922 年，斯特恩与盖拉赫让银原子束穿过非均匀磁场，发现束流分裂为**两束**而非连续的许多束。这意味着电子的磁矩（从而角动量）在磁场方向只能取两个分立值——这与轨道角动量给出的奇数个取值（$2l+1$）矛盾。泡利在 1925 年把这种性质称为「经典上无法描述的二值性」，并给出**泡利不相容原理**：每个量子态至多容纳一个电子，需要一个新的量子数区分「同一轨道里的两个电子」。

这个新量子数就是**自旋量子数 $s = \tfrac12$**。电子的自旋态是**二分量旋量（spinor）**：

$$
\left| \chi \right\rangle = \begin{pmatrix} a \\ b \end{pmatrix}, \qquad S_z = \frac{\hbar}{2} \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

$S_z$ 的本征态是 $\left| \uparrow \right\rangle = (1, 0)^T$ 与 $\left| \downarrow \right\rangle = (0, 1)^T$，本征值 $\pm \hbar/2$。<span class="marginnote">自旋没有经典对应物：它不是一个电子「绕自己转」产生的角动量（那需要表面速度超过光速）。它就是电子内禀的自由度——一个二分量复矢量。把 $S_z$ 写成 $2\times2$ 矩阵，暗示了背后是 $SU(2)$ 的二维表示：这就是「旋量」一词的群论含义。</span>

## 2 SU(2)：结构与泡利矩阵

**$SU(2)$ 是所有行列式为 $1$ 的 $2\times2$ 幺正矩阵构成的群**：

$$
SU(2) = \left\{ U \in M_2(\mathbb{C}) \;\middle|\; U^\dagger U = I,\; \det U = 1 \right\}
$$

$SU(2)$ 是**三参数**连续群（幺正条件给 4 个实条件、行列式条件再给 1 个，$8 - 4 - 1 = 3$）。它的生成元是**泡利矩阵（Pauli matrices）**的一半：

$$
\sigma_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_2 = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_3 = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

泡利矩阵是厄米、无迹、且 $\sigma_i \sigma_j = \delta_{ij} + i\varepsilon_{ijk}\sigma_k$。令 $T_i = \sigma_i/2$，则

$$
[T_i, T_j] = i \varepsilon_{ijk} T_k
$$

——与第 5 篇 $\mathfrak{so}(3)$ 的**对易关系完全相同**。<span class="marginnote">这是本节最深刻的观察：$\mathfrak{su}(2)$ 与 $\mathfrak{so}(3)$ 是同构的李代数。两个不同的群共享同一套李代数，意味着它们在「无穷小」层面是同一个东西——差别只藏在整体的拓扑里。</span>

## 3 双重覆盖：SU(2) 与 SO(3) 的关系

两个群共享李代数，那么群本身有什么关系？答案藏在一个漂亮的映射里。定义 $U \in SU(2)$ 到 $SO(3)$ 的映射：对任意三维矢量 $\vec{x} = x_i \sigma_i$（泡利矩阵张成的无迹厄米矩阵空间），要求

$$
U^\dagger (\vec{x} \cdot \vec{\sigma}) U = (\vec{x}' \cdot \vec{\sigma}), \qquad \vec{x}' = R \vec{x}
$$

每个 $U$ 给出一个保持 $\det$（从而保持 $|\vec{x}|$）的线性变换 $R$，即 $R \in SO(3)$。这个映射 $\rho: SU(2) \to SO(3)$ 是**同态**，且**恰好二对一**：$U$ 与 $-U$ 映到同一个 $R$。

$$
\rho(U) = \rho(-U), \qquad \ker \rho = \{I, -I\} = \mathbb{Z}_2
$$

$SU(2)$ 因此是 $SO(3)$ 的**双覆盖（double cover）**。<span class="marginnote">二对一的来源很直观：绕 $\hat{n}$ 轴转 $\theta$ 对应 $U = e^{-i\theta\,\hat{n}\cdot\vec{\sigma}/2}$，而转 $2\pi$（$U = e^{-i\pi \hat{n}\cdot\vec{\sigma}} = -I$）在 $SU(2)$ 里不是恒等，只有转 $4\pi$ 才是。$SO(3)$ 的「一圈」在 $SU(2)$ 里是「半圈」——所以 $SU(2)$ 更「大」、更基本。</span>

**辨析｜易错点：**「$SU(2)$ 是 $SO(3)$ 的双覆盖」不等于「$SU(2)$ 和 $SO(3)$ 一一对应」。对每个空间旋转 $R$，存在两个 $SU(2)$ 元素 $\pm U$。物理上这意味着：绕任意轴转 $360°$，自旋 $\tfrac12$ 的态会乘上 $-1$——这个「负号」在干涉实验里真实可见（中子干涉实验已验证），是自旋为半整数粒子的标志。

## 4 公式解析：泡利矩阵与 SU(2)→SO(3) 映射

把上面的映射写成显式公式，是本节最重要的计算：

$$
e^{-i\theta\,\hat{n}\cdot\vec{\sigma}/2} \; \vec{\sigma} \; e^{+i\theta\,\hat{n}\cdot\vec{\sigma}/2} = R(\hat{n}, \theta)\, \vec{\sigma}
$$

逐项拆解：

- **第一步，识别指数**：$U = e^{-i\theta\,\hat{n}\cdot\vec{\sigma}/2}$ 是 $SU(2)$ 的一般元素（三个参数 $\theta, \hat{n}$）。指数里的 $\vec{\sigma}/2$ 是生成元，正如 $SO(3)$ 里 $e^{-i\theta\hat{n}\cdot\vec{J}}$。
- **第二步，看清变换对象**：等号左边，泡利矢量 $\vec{\sigma} = (\sigma_1, \sigma_2, \sigma_3)$ 被 $U$ 夹着做相似变换。由于 $U^\dagger = U^{-1}$，这是「用 $U$ 把泡利矩阵旋转一下」。
- **第三步，读出右边的矩阵**：$R(\hat{n}, \theta) \vec{\sigma}$ 表示 $R$ 的三行作为系数线性组合 $\sigma_1, \sigma_2, \sigma_3$——它自动是 $2\times2$ 厄米无迹矩阵，且保持 $|\vec{x}|$ 不变。因此 $R$ 是正交矩阵且 $\det R = +1$，即 $R \in SO(3)$。
- **第四步，验证二对一**：把 $\theta$ 换成 $\theta + 2\pi$，$U \to e^{-i(\theta+2\pi)\hat{n}\cdot\vec{\sigma}/2} = -U$，但夹着 $\vec{\sigma}$ 的相似变换不变（$(-U)\sigma(-U)^\dagger = U\sigma U^\dagger$），所以 $\rho(-U) = \rho(U)$。一个 $R$ 对应两个 $U$。

这个公式的物理后果是：**费米子的波函数在 $2\pi$ 旋转下变号，玻色子不变号。** 自旋为半整数的粒子属于 $SU(2)$ 的双值表示（真双值表示），整数自旋属于 $SO(3)$ 的普通表示——这是泡利不相容原理与「自旋-统计」联系的群论背景。

## 5 自旋表示与角动量合成

$SU(2)$ 的不可约表示由**半整数或整数 $j = 0, \tfrac12, 1, \tfrac32, \dots$** 标记，维数 $2j+1$。这比 $SO(3)$ 多出的正是半整数。$j = \tfrac12$ 是二分量旋量表示，$j = 1$ 是三维矢量表示。

把两个自旋 $\tfrac12$ 合起来，是原子物理的日常操作。张量积表示分解为

$$
\frac12 \otimes \frac12 = 1 \oplus 0
$$

即「两个自旋 $\tfrac12$」分解为**三重态（triplet，$j=1$）**与**单重态（singlet，$j=0$）**。三重的三个态是 $\left|\uparrow\uparrow\right\rangle$、$\frac{1}{\sqrt2}(\left|\uparrow\downarrow\right\rangle + \left|\downarrow\uparrow\right\rangle)$、$\left|\downarrow\downarrow\right\rangle$，单重态是 $\frac{1}{\sqrt2}(\left|\uparrow\downarrow\right\rangle - \left|\downarrow\uparrow\right\rangle)$。<span class="marginnote">这些组合系数叫<strong>克莱布施-戈登系数（Clebsch–Gordan coefficients）</strong>。单重态对交换反对称、三重态对称——泡利原理要求两个电子整体波函数反对称，于是「空间波函数对称 → 自旋单重态」，「空间反对称 → 自旋三重态」。氦原子能级里正氦（三重态）与仲氦（单重态）的差别，根源就在 $SU(2)$ 的这个分解。</span>

更一般的张量积 $j_1 \otimes j_2 = (j_1 + j_2) \oplus \cdots \oplus |j_1 - j_2|$，每次减 $1$。这条「角动量合成规则」不是经验法则，而是 $SU(2)$ 张量积表示的标准分解——所有原子光谱的精细结构、超精细结构都由此组织。

## 6 同位旋：借来的 SU(2)

同一个 $SU(2)$，被海森伯在 1932 年借去描述核物理：质子与中子的质量几乎相同（差约 $0.1\%$），且在强相互作用下行为对称——若忽略电磁作用与质量差，质子和中子只是同一个核子的两个「自旋态」。于是定义**同位旋（isospin）**：

$$
N = \begin{pmatrix} p \\ n \end{pmatrix}, \qquad I_3 = \frac12 \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

质子带 $I_3 = +\tfrac12$，中子带 $I_3 = -\tfrac12$。<span class="marginnote">同位旋是「把自旋的语言借来命名新的对称性」：数学结构与 $SU(2)$ 完全一样，但物理含义不同——不是空间旋转，而是「质子里交换中子的抽象旋转」。它启发了杨-米尔斯理论里「局域对称性」的思路，是现代规范理论的直接前身。</span>

同位旋的威力在于预言新的粒子：核子（$I = \tfrac12$）散射中交换的介子应组成 $I = 1$ 的三重态——这正是 $\pi^+$、$\pi^0$、$\pi^-$ 三个π介子。电荷与同位旋第三分量和重子数的关系为 $Q = I_3 + \tfrac{B}{2}$（盖尔曼-西岛关系的前身）。「强相互作用的对称群不是更小而是更大」，同位旋把核物理从「一堆粒子」整理成「几组多重态」。

## 7 小结

- **自旋**是电子内禀的二分量自由度，$s = \tfrac12$；斯特恩-盖拉赫实验与泡利不相容原理确立了它的存在。
- **$SU(2)$** 是 $2\times2$ 幺模幺正矩阵群，生成元为 $\sigma_i/2$，李代数 $\mathfrak{su}(2) \cong \mathfrak{so}(3)$。
- **双覆盖**：映射 $U \mapsto R$ 二对一，$\ker\rho = \{I, -I\}$；绕任意轴转 $2\pi$，旋量乘 $-1$。
- **不可约表示**由 $j = 0, \tfrac12, 1, \tfrac32, \dots$ 标记，维数 $2j+1$；$\tfrac12 \otimes \tfrac12 = 1 \oplus 0$ 给出三重态/单重态与克莱布施-戈登系数。
- **同位旋**把 $SU(2)$ 借给核物理：$(p, n)$ 构成二重态，$\pi^\pm, \pi^0$ 构成三重态；电荷满足 $Q = I_3 + B/2$。
- $SU(2)$ 的表示论与克莱布施-戈登分解，是原子光谱、核能级与强子多重态的共同代数骨架。

在下一节，我们将把 $SU(2)$ 推广到 $SU(3)$