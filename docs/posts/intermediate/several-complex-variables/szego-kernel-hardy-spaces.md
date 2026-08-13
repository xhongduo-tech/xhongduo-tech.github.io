---
title: Szegő 核、Hardy 空间与边界值
date: 2026-08-07
---

# Szegő 核、Hardy 空间与边界值

<div class="epigraph">
<p>Hardy 空间是全纯函数在边界上的影子——它们把无界的函数变成有界的边值，把内蕴的解析变成外显的测量。</p>
<footer>—— 仿 加博尔 · 塞格（Gábor Szegő），《正交多项式与再生核》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Krantz 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从 Hardy 空间开始

Bergman 空间（第 22 篇）用**区域内部**的 $L^2$ 范数定义。但全纯函数真正的「个性」往往藏在**边界**上——单复变的 Hardy 空间 $H^p$ 告诉我们：圆盘上的有界全纯函数有「几乎处处」的边界值，且边界值的 $L^p$ 范数决定函数的一切。多复变把这一整套搬到**强伪凸域**上：**Hardy 空间** $H^p(D)$ 用边界范数定义，**Szegő 核**是它的再生核（对应 $p=2$），**Szegő 投影**把 $L^2(\partial D)$ 投到 Hardy 空间。这是连接「内部解析」与「边界测量」的桥梁，也是 CR 理论（下一组）的直接入口。<span class="marginnote">为什么 Hardy 而非 Bergman 是「边界理论」的主角？因为 Hardy 范数只依赖边界值：$\|f\|_{H^p} = \sup_{\varepsilon} \|f\|_{L^p(\partial D_\varepsilon)}$（内逼近边界）。这让 Hardy 空间天然适合研究「函数在边界附近的极限行为」——而边界行为正是 CR 结构与延拓理论的起点。</span>

## 1 Hardy 空间的定义

设 $D \Subset \mathbb{C}^n$ 有光滑边界。对 $1 \le p \le \infty$，定义

$$
H^p(D) = \left\{ f \in \mathcal O(D) : \sup_{0\lt \varepsilon\ll1} \int_{\partial D_\varepsilon} |f|^p \, d\sigma_\varepsilon \lt  \infty \right\}
$$

其中 $D_\varepsilon = \{ \rho \lt  -\varepsilon \}$（向内部平移的平行区域），$d\sigma_\varepsilon$ 是 $D_\varepsilon$ 边界上的曲面测度。范数 $\|f\|_{H^p}$ 取上确界。

**边界值存在定理**：对 $f \in H^p(D)$（$1 \le p \lt  \infty$），存在边界函数 $f^* \in L^p(\partial D, d\sigma)$，使 $f \to f^*$ 在 $L^p$ 意义下（沿内逼近），且 $f$ 由 $f^*$ 唯一决定（Fatou 型定理在强伪凸域上的推广）。<span class="marginnote">$n=1$ 时，$H^p(\mathbb D)$ 的经典理论（Fatou、径向极限）完全推广：边界值 $f^*(e^{i\theta})$ 几乎处处存在。$n \geq 2$ 时（强伪凸域），Korányi–Stein 用<strong>非切向极限</strong>沿复切方向建立边界值；一般伪凸域上则没有这么干净的理论——边界光滑性与伪凸强度再次成为分水岭。</span>

## 2 Szegő 核与 Szegő 投影

考虑 $p = 2$。$H^2(D)$ 是 $L^2(\partial D, d\sigma)$ 的闭子空间（通过边值嵌入）。**Szegő 投影** $S$：$L^2(\partial D) \to H^2(D)$ 的正交投影。它的**再生核**即 **Szegő 核** $S(z, \zeta)$：

$$
f(z) = \int_{\partial D} f^*(\zeta)\, S(z,\zeta) \, d\sigma(\zeta), \qquad z \in D
$$

$S(z,\zeta)$ 对 $z$ 全纯、对 $\zeta$ 是边界函数（$\bar\zeta$ 方向一般）。<span class="marginnote">Szegő 核与 Bergman 核的区别：Bergman 核在<strong>内部</strong>积分（Lebesgue 测度），Szegő 核在<strong>边界</strong>积分（曲面测度）。强伪凸域上两者都 $C^\infty$ 直到边界、都有边界奇性，但奇性指数不同：Bergman 对角奇性 $\sim \delta^{-(n+1)}$，Szegő $\sim \delta^{-n}$（$\delta$ = 到边界距离）。它们由「$\bar\partial$-Neumann 算子」与「Tangential CR 算子」分别控制。</span>

## 3 Szegő 核与 CR 结构的关系

Szegő 投影和核与 $\bar\partial$ 理论、CR 理论三重交织：

**（1）与 $\bar\partial$-Neumann 的联系**：$S = I - \bar\partial^*_b N_b \bar\partial_b$（边界版本），其中 $\bar\partial_b$ 是 **CR 结构算子**，$N_b$ 是边界上的 $\bar\partial_b$-Neumann 解算子。这预示了下一组的 CR 理论。

**（2）与 CR 函数的关系**：$H^2(D)$ 的边界值空间正是边界 $\partial D$ 上满足 Cauchy–Riemann 方程（CR 方程）的 $L^2$ 函数空间。**Hardy 空间边值 = 平方可积 CR 函数**——这是「解析内部」与「CR 边界」的精确接缝。<span class="marginnote">这个等式是多复变边界理论的枢纽：一边是 $H^p$（内部解析的边界投影），一边是 CR 函数（边界上的内蕴方程）。下一组我们会从边界这一侧重新出发——不再有「内部」，只有超曲面 $\partial D$ 与它携带的 CR 结构。</span>

**（3）边界值的延拓**：若 $f^* \in H^2$ 的边值满足更强的正则性（如 CR 函数可延拓），则 $f$ 能越过边界——这通向下一组末篇的「解析延拓与连续性原理」。

## 4 公式解析：Szegő 核的再生公式

$$
f(z) = \int_{\partial D} f^*(\zeta)\, S(z,\zeta)\, d\sigma(\zeta)
$$

- **第一步，与 Bergman 再生的区别**：Bergman：$f(z) = \int_D f(w) K(z,w) dw$（内部积分，Lebesgue）；Szegő：$f(z) = \int_{\partial D} f^* S\, d\sigma$（边界积分，曲面测度）。**同一个再生思想，不同的积分域与测度**。
- **第二步，Riesz 表示的重演**：$H^2 \subset L^2(\partial D)$，求值泛函 $f \mapsto f(z)$ 连续（由 Hardy 范数 + 边界正则性），Riesz 给出 $S(\cdot, z) \in H^2$，令 $S(z,\zeta) = \overline{S(\zeta, z)}$ 即得再生公式。
- **第三步，对角值与边界奇性**：$S(z,z) = \sum_\nu |s_\nu(z)|^2 \sim \mathrm{dist}(z,\partial D)^{-n}$（强伪凸），比 Bergman 的 $-(n+1)$ 弱一次——因为边界测度比内部测度「少一维」，范数的缩放不同。**指数之差是内部与边界维数差的回声**。

## 5 辨析与延伸：Hardy 空间的五个要点

**辨析 1：Hardy 范数依赖边界值，Bergman 范数依赖内部值**。$H^p$ 的范数是边界 $L^p$ 范数的上确界（内逼近）；$A^2$ 的范数是内部 $L^2$。**Hardy 空间是「边界理论」，Bergman 空间是「内部理论」**——这决定了它们在不同问题中的分工。<span class="marginnote">对 $n\geq2$ 的强伪凸域，$f\in H^p$ 有非切向边界值（Korányi–Stein 定理），但边界值只在复切方向「丰满」——这是 CR 结构与 Hardy 理论交织的深层原因。</span>

**辨析 2：$p$ 的范围**。$H^p$ 对 $1\leq p\leq\infty$ 定义；$p=2$ 时是 Hilbert 空间（有核）；$p\neq2$ 时是 Banach 空间（无再生核）。**Szegő 核只属于 $p=2$**——这是 Hilbert 空间结构的特权。

**辨析 3：边界值 ≠ 连续延拓**。$f\in H^p$ 有 $L^p$ 边界值 $f^*$，但 $f$ 未必连续延拓到边界。边界值是在 $L^p$ 意义下的极限，不是逐点值。**「几乎处处」的边界值是 Hardy 理论的常态**。

**辨析 4：Hardy 空间与 CR 函数的等同**。$H^2(D)$ 的边值空间 = $\partial D$ 上平方可积的 CR 函数空间。这个等同是「内部解析」与「边界 CR」的接缝——也是第 5 组 CR 理论的入口。

**误区清单**：

- **误区 1**：以为「Hardy 范数是边界逐点值」。
  正解：是 $L^p$ 极限意义下的边界值。
- **误区 2**：以为「Szegő 核对所有 $p$ 都有」。
  正解：只有 $p=2$（Hilbert 空间）有再生核。
- **误区 3**：以为「Hardy = Bergman」。
  正解：一个测边界、一个测内部。
- **误区 4**：以为「Hardy 空间只在单位圆盘有意义」。
  正解：强伪凸域上有完整理论（Korányi–Stein）。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| Hardy 空间 | Hardy space | $H^p(D)$ |
| Szegő 核 | Szegő kernel | 边界再生核 |
| Szegő 投影 | Szegő projection | $L^2(\partial D)\to H^2$ |
| 非切向极限 | nontangential limit | 边界值方式 |
| 边值函数 | boundary function | $f^*$ |
| CR 函数 | CR function | $\bar\partial_b f=0$ |

## 6 历史注记与知识树

**历史**：Hardy 空间由 Hardy（1915）在单位圆盘上引入；Szegő 给出再生核与投影。多复变推广主要由 Stein 学派完成：Korányi 与 Stein 建立强伪凸域的边界值理论（非切向极限），Folland–Stein 发展 Heisenberg 群上的 Hardy 空间。至今 $H^p$ 理论仍是调和分析与 CR 几何的交汇点。

**知识树**：

- 向后：Bergman 核（本组第 22 篇）、Bochner–Martinelli（本组第 23 篇）。
- 向前：CR 结构与 CR 函数（第 5 组）——Hardy 边值 = 平方可积 CR 函数。
- 横向：调和分析的 $H^p$ 理论（第三级《调和分析》）——实与复两个世界的平行理论。

**一句话记忆**：Hardy = 边界范数 + $L^p$ 边界值；Szegő = 边界再生核（$p=2$）；边值空间 = CR 函数空间——内部解析与边界 CR 在此合流。

## 7 小结

- **Hardy 空间** $H^p(D)$：边界范数定义的全纯函数空间；有 $L^p$ 边界值（$p\lt ∞$）。
- **Szegő 投影与核**：$L^2(\partial D) \to H^2$ 的正交投影及其再生核；边值空间 = 平方可积 CR 函数。
- **三重联系**：与 $\bar\partial_b$-Neumann、CR 函数、边界延拓相连。
- **Bergman vs Szegő**：内部积分 vs 边界积分；对角奇性 $-(n+1)$ vs $-n$。
- **$H^2$ 边值 = 平方可积 CR 函数**：内部解析与边界 CR 的精确接缝。
- **非切向极限**：$H^p$ 边界值在强伪凸域上由 Korányi–Stein 定理给出。
- **$p=2$ 专属**：再生核（Szegő 核）只属于 Hilbert 空间情形。
- **对角奇性 vs 内部奇性**：Szegő $\sim \delta^{-n}$，Bergman $\sim \delta^{-(n+1)}$