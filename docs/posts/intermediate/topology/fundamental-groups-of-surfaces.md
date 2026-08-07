---
title: 曲面的基本群：从边字读出生成元与关系
date: 2026-08-07
---

# 曲面的基本群：从边字读出生成元与关系

<div class="epigraph">
<p>数学中真正的神秘的不是它的有用性，而是它居然有用。</p>
<footer>—— 尤金 · 维格纳（Eugene Wigner）</footer>
</div>

<div class="article-byline">
<p>第二级 · 拓扑学 ｜ 尤承业《基础拓扑学讲义》第八章 ｜ 2026-08-07</p>
</div>

## 为什么从曲面基本群开始

分类曲面除了靠可定向性与 $\chi$，还有一个更深的结构——**基本群**。基本群是曲面的「代数指纹」：它记录曲面上所有回路彼此是否等价。对环面，$\pi_1$ 是交换群 $\mathbb{Z}^2$；对射影平面，$\pi_1$ 是二阶群 $\mathbb{Z}/2$；对亏格 $g$ 曲面，$\pi_1$ 是一大类非交换群。计算它们，既是对第六篇 Van Kampen 定理的实战检验，也是第九篇拓扑应用的预备队。<span class="marginnote">基本群还是区分「同伦等价但不拓扑同胚」的灵敏工具：环面与 Klein 瓶 $\chi$ 相同、可定向性相反，基本群也不一样（$\mathbb{Z}^2$ 交换 vs 非交换），是区分二者的第二道保险。</span>

本课用**边字直读**与 **Van Kampen 定理**两条路线交叉计算曲面的基本群。

## 1 从边字直读基本群

把曲面粘成一个多边形后，多边形的中心是单连通的，所有回路都可以「压」到边界上的边字。这给出一个机械法则：

<strong>边字表示法：若曲面 $M$ 由边字 $W$ 表示（字母为 $a_1, \dots, a_k$），则</strong>

$$
\pi_1(M) \cong \langle a_1, \dots, a_k \mid W = 1 \rangle
$$

其中 $W = 1$ 是把边字当作一个字（letter word）等于单位元。这个呈现（presentation）的合法性，来自「把整个多边形收缩到基点」与 Van Kampen 定理的组合论证。

例如环面 $T^2$ 边字 $a b a^{-1} b^{-1}$，直接读出

$$
\pi_1(T^2) \cong \langle a, b \mid a b a^{-1} b^{-1} = 1 \rangle
$$

关系 $aba^{-1}b^{-1} = 1$ 即 $ab = ba$，所以 $a, b$ **交换**，群是自由交换群：

$$
\pi_1(T^2) \cong \mathbb{Z} \oplus \mathbb{Z} \cong \mathbb{Z}^2
$$

## 2 亏格 g 曲面：一族非交换群

亏格 $g$ 曲面 $M_g$ 的边字是 $\prod_{i=1}^g a_i b_i a_i^{-1} b_i^{-1}$，于是

$$
\pi_1(M_g) \cong \bigl\langle a_1, b_1, \dots, a_g, b_g \;\big|\; [a_1, b_1] [a_2, b_2] \cdots [a_g, b_g] = 1 \bigr\rangle
$$

其中 $[a, b] = aba^{-1}b^{-1}$ 是**换位子（commutator）**。

- $g = 0$：没有生成元与关系，$\pi_1(S^2) = \{1\}$——球面单连通。
- $g = 1$：一个关系 $[a,b] = 1$，即交换，$\pi_1(T^2) = \mathbb{Z}^2$。
- $g \ge 2$：群**非交换**且非平凡。它由 $2g$ 个生成元、一个关系给出，是「表面群」（surface group）的经典例子。<span class="marginnote">表面群 $\pi_1(M_g)$（$g\ge2$）是群论里的名角：它是双曲曲面上的离散群，与代数几何的模空间、Teichmüller 理论紧密相连。第二级之后若走几何方向，会在《黎曼曲面》《双曲几何》里再见它。</span>

## 3 不可定向曲面：射影平面与 Klein 瓶

射影平面 $\mathbb{RP}^2$ 边字 $a a$：

$$
\pi_1(\mathbb{RP}^2) \cong \langle a \mid a^2 = 1 \rangle \cong \mathbb{Z} / 2\mathbb{Z}
$$

只绕「$a$」这一条边一圈，再绕一圈就抵消——所以射影平面上有一条「绕两圈才解开」的回路，基本群是二阶循环群。

Klein 瓶 $K$ 边字 $a b a b^{-1}$：

$$
\pi_1(K) \cong \langle a, b \mid a b a b^{-1} = 1 \rangle
$$

化简：$abab^{-1} = 1 \Rightarrow ab = ba^{-1}$，即 $bab^{-1} = a^{-1}$。这是一个**半直积**（semi-direct product）$\mathbb{Z} \rtimes \mathbb{Z}$：$b$ 共轭地把 $a$ 变成 $a^{-1}$。它非交换，与 $\pi_1(T^2) = \mathbb{Z}^2$ 形成鲜明对照。

**辨析｜易错点：** $\pi_1(K)$ 不是 $\mathbb{Z}^2$，尽管 $\chi(K) = \chi(T^2) = 0$。$\mathbb{Z}^2$ 交换，而 $ab \ne ba$（因为 $ab = ba^{-1} \ne ba$ 除非 $a^2 = 1$）。这提醒我们：**Euler 示性数与基本群是互补的不变量，各管一方面**。<span class="marginnote">半直积 $\mathbb{Z}\rtimes\mathbb{Z}$ 与直积 $\mathbb{Z}\times\mathbb{Z}$ 的差别，正是 Klein 瓶与环面的差别——拓扑上的「扭转」对应代数上的「非平凡共轭作用」。</span>

## 4 用 Van Kampen 定理交叉验证

另一种计算路线是分解曲面。以环面为例：环面 $T^2$ 可看作两个「带手柄的开圆盘」沿一个圆周粘合。更简单的验证是射影平面：

$\mathbb{RP}^2 = M$（Möbius 带）$\cup_{\partial} D^2$。Möbius 带形变收缩到它的中缝 $S^1$，所以 $\pi_1(M) \cong \pi_1(S^1) \cong \mathbb{Z}$（生成元 $a$）。沿边界圆把 $D^2$ 粘上，Van Kampen 定理给出：$D^2$ 的包含映射把 $a$ 送进 $a^2$（Möbius 带边界绕中缝两圈），于是商掉 $a^2 = 1$：

$$
\pi_1(\mathbb{RP}^2) \cong \langle a \mid a^2 = 1 \rangle
$$

与边字直读结果完全一致。这套「分解—分别算—粘合归并」正是 Van Kampen 定理的标准流程，也是第九篇里反复使用的计算范式。

## 5 公式解析：环面基本群的「两维自由度」

把最重要的计算——环面——展开成逐步解析：

$$
\pi_1(T^2) = \langle a, b \mid ab = ba \rangle \cong \mathbb{Z} \oplus \mathbb{Z}
$$

- **生成元 $a, b$**：$a$ 是「绕环面一圈的赤道」回路，$b$ 是「绕环面的子午圈」回路。它们互相独立，像平面上的两个方向。
- **关系 $ab = ba$**：先绕赤道再绕子午圈，与先绕子午圈再绕赤道，得到的回路同伦（在环面上可以把一圈在另一圈上「滑过去」）。所以两个生成元交换。
- **$\mathbb{Z} \oplus \mathbb{Z}$**：交换群由两个独立无限循环生成，同构于平面格点 $\mathbb{Z}^2$。元素 $(m, n)$ 表示「绕 $a$ 共 $m$ 圈、绕 $b$ 共 $n$ 圈」。
- **几何直觉**：环面的回路空间（的同伦类）就是一个二维格——两个洞的自由度各给一个整数圈数，与万有覆盖 $\mathbb{R}^2 \to T^2$ 的平移群 $\mathbb{Z}^2$ 精确对应（呼应第七篇覆盖空间）。
- **对照 Klein 瓶**：同样的生成元，关系换成 $ab = ba^{-1}$，交换性被破坏——这就是「扭转」与「不扭转」在代数上的分水岭。

## 6 曲面基本群一览表

把本课全部计算结果汇总成表，方便查用：

| 曲面 | 表示 | $\pi_1$ | 交换？ | $\chi$ |
| --- | --- | --- | --- | --- |
| $S^2$ | $aa^{-1}$ | $\{1\}$ | — | 2 |
| $T^2$ | $aba^{-1}b^{-1}$ | $\mathbb{Z}^2$ | 是 | 0 |
| $M_g$（$g\ge2$） | $\prod [a_i,b_i]$ | $\langle a_i,b_i \mid \prod[a_i,b_i]=1\rangle$ | 否 | $2-2g$ |
| $\mathbb{RP}^2$ | $aa$ | $\mathbb{Z}/2$ | 是 | 1 |
| $K$ | $abab^{-1}$ | $\mathbb{Z}\rtimes\mathbb{Z}$ | 否 | 0 |

注意：$\mathbb{RP}^2$ 的 $\pi_1 = \mathbb{Z}/2$ 交换（二阶群平凡交换），但它是「绕两圈才消」的有限群，与 $\mathbb{Z}$ 完全不同。这张表也是第九篇「用基本群证明不动点定理」的查表工具：只要知道 $\pi_1(S^1)=\mathbb{Z}$、$\pi_1(D^2)=\{1\}$，就能复现全部矛盾论证。

## 7 小结

- 边字直读：$\pi_1(M) \cong \langle \text{字母} \mid \text{边字} = 1\rangle$。
- $\pi_1(S^2) = \{1\}$，$\pi_1(T^2) = \mathbb{Z}^2$，$\pi_1(M_g)$（$g\ge2$）= 表面群 $\langle a_i, b_i \mid \prod[a_i, b_i] = 1 \rangle$。
- $\pi_1(\mathbb{RP}^2) \cong \mathbb{Z}/2$，$\pi_1(K) \cong \mathbb{Z} \rtimes \mathbb{Z}$（非交换）。
- **Euler 示性数 + 可定向性 + 基本群**构成区分曲面的三重保险。
- Van Kampen 定理的「分解—分别算—归并」是验证边字结果的标准工具。

在下一课，我们迎来第八篇的压轴：**紧曲面分类定理**——只需一个清单，穷尽所有紧连通曲面。
