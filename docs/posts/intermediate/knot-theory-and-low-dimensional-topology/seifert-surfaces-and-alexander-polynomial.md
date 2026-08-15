
---

title: Seifert 曲面与 Alexander 多项式

date: 2026-08-07

---



# Seifert 曲面与 Alexander 多项式



<div class="epigraph">

<p>给一个结一个曲面，它便不再只是一个缠结，而是一个三维世界的边界。</p>

<footer>—— 本文作者按</footer>

</div>



<div class="article-byline">

<p>第二级 · 纽结理论与低维拓扑 ｜ Lickorish《An Introduction to Knot Theory》第2、6章 · Adams《The Knot Book》第4章 ｜ 2026-08-07</p>

</div>



## 为什么从「曲面」开始



一条打了结的绳子是一条一维闭曲线，但它总可以是**某个曲面的边界**——像一条闭合的拉链是一条布带的边界。任何结都能「张成」一个紧致的可定向曲面，这类曲面称为**Seifert 曲面**。这个看似简单的几何事实，是纽结理论第一个强大不变量——**Alexander 多项式**——的入口：从 Seifert 曲面读出「缠绕矩阵」，再取行列式，就得到多项式。



Herbert Seifert 在 1934 年给出算法（现在叫 **Seifert 算法**）：任何结图都能机械地构造出一个可定向曲面。James Alexander 在 1928 年定义了同名的多项式，但它真正「活」起来是在 Seifert 曲面理论出现之后——两者结合，纽结理论第一次有了「曲面 → 矩阵 → 多项式」这条可计算的流水线。<span class="marginnote">Alexander 多项式是第一个被发现的结多项式（1928），比 Jones 多项式早了半个多世纪。它平凡地不能区分镜像（$\Delta_K(t) = \Delta_{K^*}(t)$，$K^*$ 为镜像），却能证明大量结非平凡——1930 年代它就是判定「三叶结真的打结了」的最早代数工具。</span>



## 1 Seifert 曲面与 Seifert 算法



**Seifert 曲面（Seifert surface）**：以定向结 $K$ 为边界的、紧致、可定向、嵌入 $\mathbb{R}^3$ 的曲面 $F$，满足 $\partial F = K$。



Seifert 算法从结图出发构造曲面，分三步：



1. **拆交叉**：在每个交叉点把「交叉」替换为「接通」（切断交叉、按定向接通两侧），得到若干条**不相交的闭曲线**（Seifert 圈）。

2. **嵌圆盘**：给每条 Seifert 圈「填」一个平盘（圆盘），让它们落在不同高度，互不相交。

3. **连半扭带**：在原交叉处，用**半扭带**（half-twisted band）把上下两个圆盘连起来，恢复原来的交叉结构。



<svg viewBox="0 0 380 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="ttl desc" font-family="sans-serif">

<title id="ttl">Seifert 算法：交叉 → 接通 → 圆盘 + 半扭带</title>

<desc id="desc">从左到右：一个交叉被切成两条圈，每个圈填圆盘，再在半扭带处连回。</desc>

<rect x="0" y="0" width="380" height="150" fill="#faf9f5"/>

<text x="190" y="20" text-anchor="middle" font-size="14" fill="#333">Seifert 算法的三步</text>

<!-- 交叉 -->

<text x="50" y="42" text-anchor="middle" font-size="13" fill="#c0392b">交叉</text>

<path d="M 15 120 L 60 60" stroke="#333" stroke-width="3" fill="none"/>

<path d="M 15 60 L 60 120" stroke="#333" stroke-width="3" fill="none"/>

<!-- 接通 -->

<text x="150" y="42" text-anchor="middle" font-size="13" fill="#c0392b">接通</text>

<path d="M 110 70 Q 150 70 150 100" stroke="#333" stroke-width="3" fill="none"/>

<path d="M 110 100 Q 150 100 150 70" stroke="#333" stroke-width="3" fill="none"/>

<!-- 圆盘 + 带 -->

<text x="275" y="42" text-anchor="middle" font-size="13" fill="#c0392b">圆盘＋半扭带</text>

<ellipse cx="230" cy="95" rx="26" ry="14" fill="#1f6f8b" opacity="0.35" stroke="#1f6f8b" stroke-width="2"/>

<ellipse cx="320" cy="75" rx="26" ry="14" fill="#1f6f8b" opacity="0.35" stroke="#1f6f8b" stroke-width="2"/>

<path d="M 256 90 Q 290 60 320 70" stroke="#333" stroke-width="3" fill="none"/>

<path d="M 256 100 Q 290 110 320 80" stroke="#333" stroke-width="3" fill="none"/>

</svg>



Seifert 算法的输出是个**可定向曲面**：因为每条半扭带都「正确拧向」保持整体定向。这个构造保证了**每个结都有 Seifert 曲面**——这是本专题最关键的存在性定理之一。



## 2 亏格：曲面的「复杂度」



可定向曲面按**亏格（genus）**分类：球面亏格 0，环面亏格 1，双环面亏格 2，依此类推。



**结的亏格（knot genus）**：$g(K)$ = 以 $K$ 为边界的 Seifert 曲面中亏格的最小值。



平凡结的亏格为 0（它张成一个圆盘）。

三叶结亏格为 1（它张成环面带一个洞的曲面）。

- 亏格是**不变量**，因为「最小亏格」不依赖具体 Seifert 曲面的选择。



**亏格与连通和**：$g(K_1 \# K_2) = g(K_1) + g(K_2)$——连通和把亏格「相加」，这与「多项式相乘」相互印证。<span class="marginnote">亏格是「结有多复杂」的几何度量：它衡量「要多少洞才能把结张成曲面」。Seifert 算法给出的曲面不一定是亏格最小的——Seifert 曲面之间的极小化问题（「这个曲面还能不能更简单」）是结理论中与「平凡结识别」同样深的问题。</span>



## 3 Seifert 矩阵与 Alexander 多项式



从 Seifert 曲面构造矩阵：设 $F$ 是亏格为 $g$ 的 Seifert 曲面，取 $2g$ 条生成 $\partial F$ 的曲线 $a_1, \ldots, a_{2g}$（它们组成 $H_1(F)$ 的基）。把每条曲线沿 $F$ 的「法向正侧」稍微推离曲面，得到 $a_i^+$。定义 **Seifert 矩阵（Seifert matrix）** $V$：



$$

V_{ij} = \operatorname{lk}(a_i, a_j^+),

$$



即第 $i$ 条曲线与第 $j$ 条曲线的（提升后的）环绕数。$V$ 是 $2g \times 2g$ 的整矩阵，依赖于曲面与基的选取，但它派生的多项式却与选取无关：



**Alexander 多项式**：



$$

\Delta_K(t) = \det\left(V - t V^{\mathsf{T}}\right),

$$



（精确到相差 $\pm t^k$ 的因子，即按惯例归一化）。这里 $V^{\mathsf{T}}$ 是 $V$ 的转置，$\det$ 是行列式。



**辨析｜Alexander 多项式为什么「差个因子」**：$\det(V - tV^{\mathsf{T}})$ 依赖基与曲面选取，可能相差 $t$ 的幂（以及符号）。约定取「对称规范」$\Delta_K(t) = \Delta_K(t^{-1})$ 归一化。所以「Alexander 多项式」本质上是一个**规范类**——两个多项式若相差 $\pm t^k$ 倍，视为同一个 Alexander 多项式。



## 4 公式解析：三叶结的 Alexander 多项式



用 Seifert 矩阵算三叶结 $3_1$。三叶结的 Seifert 曲面是亏格 1 的曲面（带一个洞），$H_1(F)$ 有两条生成曲线 $a, b$。经计算，Seifert 矩阵（在适当基下）为



$$

V = \begin{pmatrix} -1 & 1 \\ 0 & -1 \end{pmatrix}.

$$



于是



$$

V - t V^{\mathsf{T}} = \begin{pmatrix} -1 & 1 \\ 0 & -1 \end{pmatrix} - t \begin{pmatrix} -1 & 0 \\ 1 & -1 \end{pmatrix} = \begin{pmatrix} -1 + t & 1 \\ -t & -1 + t \end{pmatrix}.

$$



取行列式：



$$

\Delta_{3_1}(t) = (-1 + t)^2 + t = t^2 - 2t + 1 + t = t^2 - t + 1.

$$



- **第一步，读懂矩阵元**：$V_{ab} = \operatorname{lk}(a, b^+)$ 记录了「把 $a$ 推离曲面、与 $b$ 环绕了几次」。矩阵元是**整数**，携带的是缠绕信息。

- **第二步，为什么是 $V - tV^{\mathsf{T}}$**：这个组合把「$a$ 与 $b^+$ 的缠绕」与「$a^+$ 与 $b$ 的缠绕」合并，抵消掉基选取造成的差异。

- **第三步，归一化**：$t^2 - t + 1$ 在 $t \leftrightarrow t^{-1}$ 下不变（乘以 $t^{-2}$ 后对称），是规范形式。三叶结 Alexander 多项式 = $t^2 - t + 1$，这是所有教材都会算的「第一次」。<span class="marginnote">注意到 $\Delta_{3_1}(1) = 1$。这是一般规律：$\Delta_K(1) = \pm 1$ 对任何结成立（代入 $t=1$，$\det(V - V^{\mathsf{T}})$ 是交替矩阵的行列式，恒为 $\pm 1$）。这个「$t=1$ 处的归一化」是 Alexander 多项式最容易被遗忘的性质，也是它区别于 Jones 多项式（$V_K(1) = 1$ 强制）的记号习惯。</span>



## 5 Alexander 多项式的性质



- **对称性**：$\Delta_K(t) = \Delta_K(t^{-1})$（相差 $\pm t^k$），所以镜像结的 Alexander 多项式相同——**不能区分镜像**。

- **$t = 1$ 取值**：$\Delta_K(1) = \pm 1$（结的情况）。

- **连通和相乘**：$\Delta_{K_1 \# K_2} = \Delta_{K_1} \Delta_{K_2}$。

- **平凡结**：$\Delta_{0_1}(t) = 1$。

- **链环推广**：对 $\mu$ 分量链环，Alexander 多项式推广为 $\mu \times \mu$ 矩阵的行列式之比。



**易错点｜Alexander 多项式 ≠ 完全分类器**：八字结 $4_1$ 的 Alexander 多项式为 $t^2 - 3t + 1$，而 $5_2$ 结的是 $2t^2 - 3t + 2$，可区分；但存在不同结共享同一 Alexander 多项式（如 $8_8$ 与 $8_{9}$）。Alexander 多项式分辨力有限——它区分不了镜像，也区分不了某些不同结。这正是半个世纪后 Jones 多项式登场的动力。



## 6 小结



- **Seifert 曲面**是以结为边界的可定向曲面；**Seifert 算法**从结图机械构造之。

- **结的亏格** $g(K)$ 是「张成结所需的最少洞数」，是不变量，对连通和相加。

- **Seifert 矩阵** $V_{ij} = \operatorname{lk}(a_i, a_j^+)$ 编码缠绕；**Alexander 多项式** $\Delta_K(t) = \det(V - tV^{\mathsf{T}})$。

- 三叶结 $\Delta = t^2 - t + 1$；Alexander 多项式对称（不分镜像）、$t=1$ 处取值 $\pm 1$——所以 Alexander 区分不了镜像，需要更精细的工具。
- Seifert 曲面把「打结」翻译成「曲面的缠绕」，是 Alexander 多项式三种定义（曲面、覆盖、skein）里最「几何」的一种。

在下一节，我们把 Alexander 的工程换成一条优雅的递归——**Conway 多项式与 skein 关系**：一条「切换一个交叉」的线性关系从平凡结递归算遍所有结。