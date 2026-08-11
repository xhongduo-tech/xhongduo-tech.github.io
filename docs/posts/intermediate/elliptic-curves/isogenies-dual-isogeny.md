---
title: 同源与对偶同源
date: 2026-08-11
---

# 同源与对偶同源

<div class="epigraph">
<p>两个有限群之间的有限覆盖，是一切算术几何的呼吸节奏。</p>
<footer>—— 约翰 · 泰特（John Tate）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 椭圆曲线与模形式 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从同源开始

椭圆曲线的内部对称有两个层面：**自同构**（第 9 篇的 $\mathrm{End}(E)$）与**同源**（finite-to-finite 的覆盖映射）。同源是更基本的结构——自同态环正是由「从 $E$ 到 $E$ 的同源」构成的，而「乘以 $n$」$[n]$ 是最显眼的同源。同源理论是连接第 5 篇（Frobenius 是同源）、第 12 篇（Hecke 算子来自同源图）与第 13 篇（Galois 表示）的枢纽。<span class="marginnote">同源的几何直觉：它是「两个紧致黎曼面的有限覆盖」。对环面 $E(\mathbb{C}) \cong \mathbb{C}/\Lambda$，同源就是「把格 $\Lambda$ 映到它的一个子格」的乘法 $z \mapsto \alpha z$（$\alpha\Lambda \subseteq \Lambda'$）。「次数」= 覆盖的叶数 = $[\Lambda' : \alpha\Lambda]$。Silverman §III.4 是这条线的标准教材。</span>本节主题：同源的定义与次数、对偶同源的构造与关键性质、Weil 配对。

## 1 同源：有限覆盖的算术面孔

### 定义

设 $E_1, E_2$ 是 $k$ 上的椭圆曲线。一个**同源（isogeny）**是态射

$$\phi: E_1 \to E_2, \qquad \phi(O_1) = O_2$$

（保持无穷远点、保持群律的非常数态射。）**次数（degree）** $\deg\phi$ 是「覆盖的叶数」，在代数闭域上 $\#\ker\phi = \deg\phi$（可分情形）。

**重点：$\phi$ 是群同态且核有限**，于是 $E_1 \cong E_2$ 当且仅当存在次数 1 的同源（即同构）。**同源建立了「同构类之间的有限覆盖」**——它把曲线的世界连成一张网。<span class="marginnote">「可分解 / 不可分解」：同源 $\phi$ 的核给出「子群结构」，而子群与同源一一对应（「同源 = 核子群」定理）：对每个有限子群 $G \subseteq E$，存在唯一（到同构）的同源 $\phi: E \to E/G$，核恰为 $G$。于是「找同源」=「找有限子群」——这是有限域密码学（子群攻击、同源密码 SIDH）的全部根据。</span>

### 例：乘以 $n$ 与 Frobenius

- $[n]: E \to E$，次数 $n^2$，核 $E[n] \cong (\mathbb{Z}/n)^2$（特征不整除 $n$ 时）。**$[n]$ 是同源的「基准」，一切同源都以它为参照。**
- **Frobenius** $\varphi_q: E \to E^{(q)}$（第 5 篇），次数 $q$，核是「几何 Frobenius 的固定点」。**$E^{(q)}$ 是「坐标全部 $q$ 次方」的曲线**，$\varphi_q$ 把 $E$ 覆盖到它。

**辨析｜易错点：** 同源不要求「目标曲线 = 源曲线」——$E_1 \to E_2$ 可以完全不同构。但**若 $\phi: E_1 \to E_2$ 与 $\psi: E_1 \to E_2$ 同构则次数相同**；「次数」是「覆盖深度」的不变量，不是「核」的完整信息——核的大小一样（$\deg$），但核的**结构**（$\mathbb{Z}/p$ 还是 $(\mathbb{Z}/p)^2$ 之类）不同。**「次数」与「核结构」是两个不同的刻度**，初学者常混为一谈。

## 2 对偶同源：同源的「逆」的影子

### 定义与存在性

对每个非零同源 $\phi: E_1 \to E_2$（次数 $m$），存在唯一同源

$$\hat{\phi}: E_2 \to E_1, \qquad \phi \circ \hat{\phi} = [m]_{E_2}, \qquad \hat{\phi} \circ \phi = [m]_{E_1}$$

称为 $\phi$ 的**对偶同源（dual isogeny）**。它「几乎」是 $\phi$ 的逆——但因为 $\phi$ 是有限覆盖，它的「逆」会「乘以次数」：**对偶同源是「除以 $\phi$ 时留下的余数 $[m]$」**。<span class="marginnote">为什么对偶一定存在？用除子/格的语言：对 $\phi$ 的核 $G$，$E_1 \to E_2 = E_1/G$，而「除以 $G$ 的代数」自然给出「把 $E_2$ 拉回 $E_1$」的映射——它的核是 $\phi(E_1[m])$。构造是「内禀」的，不依赖于具体坐标。对环面 $E(\mathbb{C}) = \mathbb{C}/\Lambda$：$\phi$ 是 $\alpha$，则 $\hat{\phi}$ 是「$\overline{\alpha}/\alpha \cdot \alpha$」的某种伴随——在复解析层面对偶同源就是「伴随算子」。</span>

### 关键性质

对偶同源满足一组「范数式」性质：

$\deg \hat{\phi} = \deg \phi$（次数守恒）；
- $\widehat{\hat{\phi}} = \phi$（对偶的对偶是自己）；
- $\widehat{\phi + \psi} = \hat{\phi} + \hat{\psi}$，$\widehat{\phi \circ \psi} = \hat{\psi} \circ \hat{\phi}$（加性与反序性）；
- $\widehat{[n]} = [n]$，且 $\hat{\phi}\circ\phi = [\deg\phi]$——**对偶是「从右看同源」的操作**。

**重点：次数乘法公式** $\deg(\psi \circ \phi) = \deg\psi \cdot \deg\phi$，且 $\deg\phi = \#\ker\phi$ 是「乘法性的范数」。这套性质使「同源环」成为「有限维代数」：自同态环 $\mathrm{End}(E)$ 上有一个「范数」$\deg$，其极化为迹 $\mathrm{tr}(\phi) = \phi + \hat{\phi} \in \mathbb{Z}$。<span class="marginnote">对 $\phi = [n]$：$\deg[n] = n^2$，$\mathrm{tr}[n] = 2n$。对 Frobenius：$\deg\varphi_q = q$，$\mathrm{tr}\,\varphi_q = t = q+1-\#E(\mathbb{F}_q)$——<strong>第 5 篇的迹 $t$ 正是 $\varphi_q$ 的「迹」</strong>。把「迹」视作「特征多项式 $x^2 - tx + q$ 的系数」，「Hasse 界」就变成「该二次多项式判别式 ≤ 0」——算术的界在此与代数的正定性合一。</span>

## 3 Weil 配对：探测同源核的指纹

### 定义

对 $m$-挠元 $P, Q \in E[m]$，**Weil 配对（Weil pairing）** 给出

$$e_m: E[m] \times E[m] \to \mu_m \subset k^*$$

满足双线性、交错性（$e_m(P,P) = 1$）、非退化性（$e_m(P,Q) = 1 \ \forall Q \Rightarrow P = O$）。<span class="marginnote">构造的直观：取 $Q$ 的「预像」$Q'$（$\phi_Q(Q')=Q$ 的某种除子平移），定义 $e_m(P,Q) =$ 「$P$ 平移 $Q'$ 时除子变换的比值」——它度量「$P$ 与 $Q$ 的『交缠』」。用格语言（$E(\mathbb{C})=\mathbb{C}/\Lambda$）：$e_m(a/m, b/m) = e^{2\pi i \cdot \text{（一个 2×2 行列式）/}m}$——<strong>Weil 配对是格上「交比」的有限化</strong>。</span>

### 对偶同源下的变换

设 $\phi: E_1 \to E_2$，次数 $m$，则

$$e_{m}\big(\phi(P), \phi(Q)\big) = e_{m^{2}}(P, Q)^{\deg\phi} \qquad \text{（适当阶的配对）}$$

更常用的形式：对 $\phi$ 与对偶 $\hat\phi$ 有 $\phi \circ \hat\phi = [\deg\phi]$，配对把「对偶」翻译成「转置」——**Weil 配对是「对偶同源」的对偶配对的化身**。<span class="marginnote">这解释了配对为何在密码学里如此重要：它把「$E[m]$ 上的离散对数」约化到「$\mu_m$ 上的离散对数」（另一条曲线的配对），是「配对密码学」的基础；同时「自守形式 ↔ 表示」的证明也反复借用「配对的转置性」。</span>

## 4 公式解析：$\phi \circ \hat{\phi} = [\deg\phi]$ 的三个读法

对偶同源定义式是全章的「锚」，拆成三层读。

$$
\phi \circ \hat{\phi} = [\deg \phi], \qquad \hat{\phi} \circ \phi = [\deg \phi]
$$

- **第一步，代数读法**：$\hat\phi$ 是「从右到左的伴随」。把 $\mathrm{End}(E)$ 看成一个「环」，$\phi \mapsto \hat\phi$ 是一个「对合 + 反序」的映射（类似于矩阵的共轭转置），而「$\phi\hat\phi = \deg\phi \cdot 1$」是说 **$\deg\phi$ 是「$\phi$ 的范数」**——$\hat\phi$ 的存在使「范数」有了「相伴元素」。
- **第二步，几何读法**：$\phi$ 是 $E_1 \to E_2$ 的 $m$ 叶覆盖，$\hat\phi$ 是「沿同一个图走回来」的 $m$ 叶覆盖，两次走完恰好是「乘以 $m$」——**覆盖复合后回到出发点并绕了 $m$ 圈**。这把「有限覆盖的拓扑」翻译成「乘法的代数」。
- **第三步，算术读法**：$E$ 在有限域上时，取 $\phi = \varphi_q$：$\hat{\varphi}_q \circ \varphi_q = [q]$，且 $\mathrm{tr}\,\varphi_q = \varphi_q + \hat{\varphi}_q = t$。于是「$t^2 - 4q \leq 0$」=「$(\varphi_q + \hat\varphi_q)^2 \leq 4\varphi_q\hat\varphi_q$」——**Hasse 界变成「迹的 Cauchy-Schwarz」**，这是一条纯代数的证明路径。
- **第四步，操作读法**：给定核 $G$（$E \to E/G$），对偶同源的核是「$\phi(E[m])$」——**「对偶」把「核」反射成「目标里的核」**。这是计算「同源图」时反复用到的公式，也是 SIDH 类密码的数学基础。

## 5 同源理论的现代角色

- **同源图与密码**：固定一个「超奇异」曲线族，$p$-同源构成「同源图」。基于同源的密码（SIDH、CSIDH）的安全假设是「在同源图中找路径」的困难——它不同于离散对数，是目前「抗量子」的候选之一。<span class="marginnote">同源密码是「后量子密码」里的新星：它把安全性建立在「在指数级大小的同源图中找路径」上，而非分解或离散对数。SIDH 曾因「配对攻击」被破（2022），CSIDH 等仍在演化——「同源 = 有限覆盖」的古老理论，在量子时代焕发新生。</span>
- **同源与模曲线**：$X_0(N)$ 的点 = 「椭圆曲线 + 一个 $N$-同源图」；Hecke 算子（第 12 篇）就是「沿同源图求和」。**同源图是模曲线算术的心脏**。
- **同源与 L-函数**：两曲线同源 ⇔ 它们的 $a_p$ 序列「几乎相同」（相差有限多素数）——**同源保持 L-函数**。这让「同源类」成为模性理论的基本单元。

### 补充：自同态环上的 Cayley-Hamilton

对偶同源给每个自同态 $\phi$ 配齐了「特征多项式」：

$$\phi^2 - \mathrm{tr}(\phi)\,\phi + \deg(\phi) = 0$$

（即 $\phi^2 - (\phi + \hat\phi)\phi + \phi\hat\phi = 0$，恒等地成立。）两个例子说明它的分量：

**CM 例子**：$E: y^2 = x^3 + x$，$\mathrm{End}(E) = \mathbb{Z}[i]$。取 $\phi = i$（解析侧 $z \mapsto iz$）：$\deg i = 1$，$\mathrm{tr}\,i = 0$，于是 $i^2 + 1 = 0$——**Cayley-Hamilton 说的正是「$i^2 = -1$」**。

**Frobenius 例子**：$\phi = \varphi_q$（第 5 篇），$\deg = q$，$\mathrm{tr} = t$，于是 $\varphi_q^2 - t\varphi_q + q = 0$。取判别式：$t^2 - 4q \leq 0$，即 **Hasse 界**——「迹的有界」是「特征多项式判别式非正」的纯代数推论。

**一句话**：一旦知道 $\mathrm{tr}$ 与 $\deg$ 两个数，自同态 $\phi$ 的「代数身份」就被完全固定。算术里「特征多项式」之所以无处不在，正因为它把「几何对象（同源）」压缩成「两个整数」。

## 6 小结

- **同源**是保持群律、核有限的非常数态射；次数 = 核的大小 = 覆盖叶数。
- **对偶同源** $\hat\phi$ 满足 $\phi\circ\hat\phi = \hat\phi\circ\phi = [\deg\phi]$，是「有限覆盖的伴随」。
- **次数是范数**：$\deg(\psi\phi) = \deg\psi\deg\phi$，迹 $\mathrm{tr}\phi = \phi + \hat\phi$；Hasse 界是迹的 Cauchy-Schwarz。
- **Weil 配对** $e_m$ 双线性、非退化，把「对偶」翻译成「转置」，支撑配对密码与模性证明。
- 同源图是现代**后量子密码**与模曲线算术的共同舞台。

在下一节，我们回到复解析的温柔乡：**椭圆曲线上的复结构（格与一致化）**——为什么每条椭圆曲线都是环面 $\mathbb{C}/\Lambda$，而 Weierstrass $\wp$-函数正是这条等价关系的显式公式。
