---
title: 格点几何与 Minkowski 基本定理
date: 2026-08-07
---

# 格点几何与 Minkowski 基本定理

<div class="epigraph">
<p>上帝创造了整数，其余一切皆出自人手。</p>
<footer>—— 利奥波德 · 克罗内克（Leopold Kronecker）</footer>
</div>

<div class="article-byline">
<p>第二级 · 离散与凸几何 ｜ Matoušek《离散几何讲义》第4章 ｜ 2026-08-07</p>
</div>

## 为什么从格开始

上一篇的多面体顶点可以在任意位置；这一篇我们强迫顶点落在**整数格点**上。把 $\mathbb{Z}^d$ 摆在空间里，多面体与整数点之间的「摩擦」产生了整个**几何数论（geometry of numbers）**：一个凸体体积要多大，才能保证它肚子里一定有一个非零格点？这个由 Minkowski 在 1890 年代建立的问题，把数论（整数）与几何（凸体体积）焊接在一起。它的一次次应用——两平方和、四平方和、Dirichlet 逼近——全都出自同一条体积论证。下一节的 Ehrhart 多项式还会把「数格点」升级成一套多项式理论，而今天这篇就是它的地基。

## 1 格：整数坐标的离散骨架

**格（lattice）**：$\Lambda \subseteq \mathbb{R}^d$ 是格，如果存在线性无关向量 $b_1, \dots, b_d$ 使

$$
\Lambda = \left\{ \sum_{i=1}^{d} z_i b_i : z_i \in \mathbb{Z} \right\}
$$

也就是基向量的一切整数线性组合。标准格 $\mathbb{Z}^d$ 的基是标准基 $e_1, \dots, e_d$；高斯整数 $a + bi$（$a, b \in \mathbb{Z}$）对应平面格 $\mathbb{Z}^2$。<span class="marginnote">格不是「一堆整数点」那么随便：它是<strong>线性无关的整数系数组合</strong>。斜着的基向量照样生成一个格，比如 $\{(m, n) + m(0.5, 1) : m,n \in \mathbb{Z}\}$ 之类——只要基向量线性无关。</span>

同一格有无数个基，它们通过**单模变换**（行列式为 $\pm 1$ 的整数矩阵）互相转化。<span class="marginnote">「同一格、多个基」是格论里最容易踩的坑：$e_1, e_2$ 与 $e_1, e_1+e_2$ 生成同一个 $\mathbb{Z}^2$，因为变换矩阵 $\begin{psmallmatrix}1&1\\0&1\end{psmallmatrix}$ 的行列式是 1。</span>格的本质不变量不是基，而是下面要定义的行列式。

**子格与指数**：若 $\Lambda' \subseteq \Lambda$ 都是 $d$-维格，则 $\Lambda'$ 是 $\Lambda$ 的**子格（sublattice）**，比值 $\det \Lambda' / \det \Lambda$ 是正整数，称为**指数（index）**，等于商群 $\Lambda / \Lambda'$ 的阶。例如 $2\mathbb{Z}^2 \subseteq \mathbb{Z}^2$ 的指数是 $4$——标准格被细化后每个基本域缩成原来的 $1/4$。<span class="marginnote">子格理论是「格的算术」：一个格在什么条件下包含某个子格？指数是多少？这类问题在格基密码（NTRU、G6K 攻击）与晶体学（倒格、布拉维格子）里是命脉。</span>后面证两平方和时构造的「模 $p$ 格」就是一个指数 $p$ 的子格。

## 2 基本域与行列式：格有多大

**基本域（fundamental domain）**：给定基 $b_1, \dots, b_d$，**基本平行体（fundamental parallelepiped）**

$$
\Pi = \left\{ \sum_{i=1}^{d} t_i b_i : 0 \le t_i \lt  1 \right\}
$$

是「每个格点恰好分到一个拷贝」的周期单元：$\mathbb{R}^d = \bigcup_{\lambda \in \Lambda} (\lambda + \Pi)$，且各拷贝互不重叠。它像墙纸的单元花纹——平移格点后正好铺满平面。<span class="marginnote">「平移 $\Pi$ 覆盖空间、彼此不交」是格的<strong>平铺（tiling）</strong>性质。这个想法在《填充与覆盖》里会放大成一般凸体的平铺理论：格本身就是一个完美填充 + 完美覆盖。</span>

**行列式（determinant / covolume）**：$\det \Lambda = |\det(b_1, \dots, b_d)|$，即基本平行体的 $d$-维体积。它不依赖基的选择（单模变换不改变行列式绝对值），是格的**固有体积单位**——一格的「平均每点占多少体积」：

$$
\det \Lambda = \mathrm{vol}(\Pi) = \frac{1}{\text{密度}}
$$

格点密度 = $1/\det\Lambda$。$\mathbb{Z}^d$ 的行列式是 1，每单位体积恰好 1 个格点。

**辨析｜易错点：** 行列式与「点之间的距离」不是一回事。$\mathbb{Z}^2$ 里两点最短距离是 1，行列式也是 1；但把基向量拉长、缩短，最短距离可以很小而行列式不变。格有三个独立的量：最短向量长度 $\lambda_1$（Minkowski 第一定理）、覆盖半径 $\mu$、行列式 $\det\Lambda$——初学者最常把三者混为一谈。**这三个量构成格的「体检报告」**：$\lambda_1$ 管「塞得下多小的凸体」，$\mu$ 管「盖得多大的空间」，$\det$ 管「平均密度」。格基密码的安全性分析，正是盯着这三者做文章。

**Minkowski 第二定理**：若 $\lambda_1 \le \lambda_2 \le \cdots \le \lambda_d$ 是格的**逐次极小（successive minima）**——第 $i$ 个是「$i$ 个线性无关的最短向量」的长度——则第一定理只是 $\lambda_1$ 的单向量版本，而第二定理把 $d$ 个极小与行列式绑定：

$$
\frac{2^d}{d!}\, \det\Lambda \le \lambda_1 \lambda_2 \cdots \lambda_d \cdot \mathrm{vol}(B_2^d) \le 2^d \det\Lambda
$$

它说明「$\lambda_1 \cdots \lambda_d$ 与 $\det\Lambda$ 互相钉住」——**格的算术量（行列式）与几何量（逐次极小）不可能同时过小**，这条「测不准原理」是几何数论与格基约化（LLL）的基石。<span class="marginnote">逐次极小把「一个最短向量」升级为「一列最短向量」。LLL 算法求的正是 $\lambda_1$ 的近似；而「积 $\lambda_1\cdots\lambda_d$」的下界保证格基不能太「斜」——这是格密码里最难解的一类问题的根源。</span>

## 3 Minkowski 基本定理：体积的保证

**定理（Minkowski 凸体定理，第一基本定理）**：设 $\Lambda \subseteq \mathbb{R}^d$ 是格，$C \subseteq \mathbb{R}^d$ 是**中心对称凸体**（$x \in C \Rightarrow -x \in C$）。若

$$
\mathrm{vol}(C) > 2^d \det \Lambda
$$

则 $C$ 内含一个**非零格点**。若 $C$ 紧且 $\mathrm{vol}(C) \ge 2^d \det\Lambda$，同样含非零格点。

这条定理是「用体积换格点」的典型：只要凸体对称且够大，它就无法避开格点。它把两个看起来无关的量——体积与格点存在性——用一条不等式绑死。<span class="marginnote">中心的「2 倍」从哪来？证明里把 $C$ 切成两半 $C/2 = \{x/2 : x \in C\}$，$C$ 的体积要大到 $C/2$ 的体积超过 $\det\Lambda$，好让 Blichfeldt 引理起效。这个「除以 2」正是中心对称性的用途。</span>

证明分两步，中间站着一个更一般也更常用的事实：

**Blichfeldt 引理**：设 $S \subseteq \mathbb{R}^d$ 可测，$\mathrm{vol}(S) > \det \Lambda$，则存在两个不同点 $x, y \in S$ 使 $x - y \in \Lambda$。

直觉：把 $S$ 按基本域分块后平移进 $\Pi$，总测度超过 $\Pi$ 的测度，于是必有两块重叠——重叠处对应 $S$ 中相差一个格向量的两个点。**鸽子笼原理从计数搬到了体积**，这是它最漂亮的一次升维。

## 4 公式解析：为什么 2^d 是临界值

把 Minkowski 定理的证明拆成四步，看清楚 $2^d$ 这个数字的来历：

$$
\mathrm{vol}(C) > 2^d \det \Lambda \quad\Longrightarrow\quad C \cap (\Lambda \setminus \{0\}) \neq \emptyset
$$

- **第一步，做一半体积的凸体**：考虑 $C' = \frac{1}{2} C = \{ x : 2x \in C \}$。因为 $C$ 中心对称，$C'$ 也是中心对称凸体，且 $\mathrm{vol}(C') = \mathrm{vol}(C)/2^d > \det\Lambda$。
- **第二步，对 $C'$ 用 Blichfeldt**：$\mathrm{vol}(C') > \det\Lambda$，于是存在 $x \neq y \in C'$ 使 $x - y = \lambda \in \Lambda$。此时 $\lambda \neq 0$ 是个非零格点。
- **第三步，把差拉回 $C$**：$x, y \in C'$ 即 $2x, 2y \in C$。中心对称性给出 $-2y \in C$；凸性给出中点 $\frac{2x + (-2y)}{2} = x - y = \lambda \in C$。
- **第四步，检查结论**：$\lambda$ 是非零格点且落在 $C$ 里，证毕。整个论证只用了三条性质：体积、对称、凸——缺一不可。若 $C$ 不对称，第三步的中点论证直接失效；若体积不达标，第一步的缩放就不足以触发 Blichfeldt。

关键数字 $2^d$ 完全来自「对称 + 取半 + 中点」这一串动作：取半损失 $2^d$ 倍体积，中点在凸体里捡回差分。**这是一个体积——拓扑——算术的连锁反应，每一步都不可删。**

## 5 应用：从两平方和到 Dirichlet 逼近

Minkowski 定理的威力在应用中才能看清。

**两平方和定理（部分情形）**：每个形如 $4k+1$ 的素数 $p$ 都能写成 $x^2 + y^2$。几何证明：考虑格 $\Lambda = \{(a, b) \in \mathbb{Z}^2 : b \equiv ma \pmod p\}$（$m$ 满足 $m^2 \equiv -1 \pmod p$），它的行列式是 $p$。取以原点为中心、面积大于 $4p$ 的圆盘 $C$（例如半径 $\sqrt{2p/\pi}$ 以上的盘）。由 Minkowski 定理，$C$ 含非零格点 $(a,b)$，它满足 $a^2 + b^2 \lt  2p$ 且 $a^2 + b^2 \equiv a^2(1+m^2) \equiv 0 \pmod p$，于是 $a^2 + b^2$ 是 $p$ 的倍数又小于 $2p$，只能等于 $p$。<span class="marginnote">费马在 1640 年断言「$4k+1$ 型素数可表为两平方和」却未给出完整证明，欧拉 1749 年才证明。而 Minkowski 用一条体积不等式就把它降为作业——「存在性问题交给几何，等式的精确值再代数化」。这种「先证存在、再套等式」的两段式，在解析数论里是家常便饭。</span>

**Dirichlet 逼近定理**：对任意实数 $\alpha$ 与正整数 $N$，存在整数 $p, q$（$1 \le q \le N$）使

$$
\left| \alpha - \frac{p}{q} \right| \lt  \frac{1}{qN} \le \frac{1}{q^2}
$$

它说任意实数都能被分母不超过 $N$ 的有理数逼近到误差 $1/(qN)$ 以内。用格子看：考虑格 $\{(q, p) : p, q \in \mathbb{Z}\}$ 的斜格与一个瘦长的中心对称平行四边形——Minkowski 定理保证斜格有一个非零向量落在瘦长体内，向量给出 $(q, p)$，比值 $p/q$ 就是所求逼近。<span class="marginnote">Dirichlet 逼近是连分数理论的入口，也是「格点几何是数论的手电筒」的最佳例证：一个纯分析命题，被 Minkowski 的体积论证照得通体透明。它还与后面《与计算几何的联系》中「用格做几何哈希」有隐秘的关联。</span>

**四平方和定理（Lagrange）**：每个非负整数都能写成四个整数的平方和 $n = a^2 + b^2 + c^2 + d^2$。用 $d=4$ 的 Minkowski 定理可以给出统一证明：取格 $\Lambda = \{(a,b,c,d) \in \mathbb{Z}^4 : b \equiv ma \pmod n \text{ 等四条同余} \}$，构造行列式 $n^2$ 的格与一个中心对称凸体，体积条件让 Minkowski 定理保证存在非零格点 $v$ 满足 $v^\top v = a^2+b^2+c^2+d^2 = n$。**两平方、四平方共用同一条体积论证**——Minkowski 定理是「平方和定理家族」的统一武器，也是「格点几何是数论手电筒」的第二个聚光点。

## 6 小结

- **格**：线性无关基向量的整数系数组合；单模变换不改格，只换基。
- **行列式 / 协体积**：基本平行体体积，$\det\Lambda$ 是格的固有体积单位，密度 $= 1/\det\Lambda$。
- **Blichfeldt 引理**：测度超过 $\det\Lambda$ 的集合必含差为格向量的两点——体积版鸽子笼。
- **Minkowski 基本定理**：中心对称凸体体积 $> 2^d \det\Lambda$ 必含非零格点；证明 = 取半 + Blichfeldt + 对称中点。
- **两平方和**：素数 $p$（$4k+1$ 型）经圆盘 + 模 $p$ 格直接证出 $p = x^2 + y^2$。
- **Dirichlet 逼近**：任意实数可被有理数逼近到 $1/q^2$；格点几何给出存在性证明。

在下一节，我们把「凸体里有没有格点」升级为「凸体里到底有多少格点」：当多面体的边长倍 $n$，格点数 $L_P(n)$ 居然是个关于 $n$