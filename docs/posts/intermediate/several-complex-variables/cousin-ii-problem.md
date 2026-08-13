---
title: Cousin II 问题与乘法问题
date: 2026-08-07
---

# Cousin II 问题与乘法问题

<div class="epigraph">
<p>给出一套零点与极点，问它们是否恰好是某个亚纯函数的零点与极点——这是乘法版本的创世纪。</p>
<footer>—— 仿 亨利 · 嘉当（Henri Cartan），《解析函数与亚纯函数》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第4章；Krantz 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从 Cousin II 问题开始

Cousin I（上一节）解决「给定主部找函数」，用的是**加法**结构。Cousin II 问的是它的**乘法**孪生：给定区域上**局部定义的非零全纯函数**（每点附近指定一组零点与极点），能否乘出一个整体亚纯函数，恰好以这些点为零点与极点？这个问题在代数几何与复几何中无比重要——**除子（divisor）**的理论、线丛的存在性、以及「什么条件下能构造出想要的零点集」都归结于此。<span class="marginnote">「乘法」之名来自零点的乘积结构：$f \sim g$ 若 $f/g$ 全纯且不消失。Cousin II 的障碍住在 $\mathcal O^*$（乘法群层）的上同调 $H^1(D, \mathcal O^*)$，而不是 $\mathcal O$ 的 $H^1$——这正是它与 Cousin I 的本质差别。</span>

## 1 问题陈述：乘法 Cousin 问题

设 $D \subset \mathbb{C}^n$。给定开覆盖 $\{U_\alpha\}$ 与一族**处处非零**的全纯函数 $f_\alpha \in \mathcal O^*(U_\alpha)$，满足相容性：

$$
\frac{f_\alpha}{f_\beta} \in \mathcal O^*(U_\alpha \cap U_\beta)
$$

**Cousin II 问题**：是否存在 $D$ 上的整体亚纯函数 $m$（不恒为零），使 $m/f_\alpha$ 在 $U_\alpha$ 上全纯且不消失？

直觉：$f_\alpha$ 描述「局部零点/极点结构」；相容性要求两套局部数据在重叠区**比例上是可逆全纯**的。问能否把它们乘成整体函数。<span class="marginnote">注意 Cousin II 比 Cousin I 严格得多。在 $n=1$，Cousin II 对应 Weierstrass 因式分解定理（任意离散零点集是一个全纯函数的零点集）；在 $n \geq 2$，一般区域上 Cousin II 未必可解——零点集不再是任意的离散点，而是<strong>超曲面</strong>（复余维 1 的解析子簇），且必须满足拓扑条件（§3）。</span>

## 2 层论翻译：障碍在 $H^1(D, \mathcal O^*)$

对乘法群层 $\mathcal O^*$（非零全纯函数，乘法），有指数层序列：

$$
0 \to \mathbb Z \xrightarrow{\;2\pi i\cdot\;} \mathcal O \xrightarrow{\;\exp\;} \mathcal O^* \to 1
$$

（$\exp$ 的核是 $\mathbb Z \cong 2\pi i\mathbb Z$——$e^{2\pi i k}=1$）。取长正合列：

$$
\mathcal O(D) \xrightarrow{\exp} \mathcal O^*(D) \to H^1(D, \mathbb Z) \to H^1(D, \mathcal O) \to H^1(D, \mathcal O^*) \to H^2(D, \mathbb Z) \to \cdots
$$

**核心结论**：Cousin II 的局部数据 $f_\alpha$ 给出 $H^1(\mathcal O^*)$ 中的类；整体解存在 ⟺ 该类为零。由长正合列，$H^1(D, \mathcal O^*) = 0$ 的充分条件是 $H^1(D,\mathbb Z) = 0$ 且 $H^1(D,\mathcal O)=0$ 且 $H^2(D,\mathbb Z)=0$。<span class="marginnote">注意这里出现了 <strong>$\mathbb Z$ 的整值上同调</strong>——Cousin II 的障碍不仅来自分析（$\mathcal O$），还来自<strong>拓扑</strong>（$\mathbb Z$）。直观：乘法数据「绕圈」后回到自身，缠绕次数（整数）无法被 $\exp$ 消除时，就产生障碍。这是多复变中分析-拓扑耦合的第一次显形。</span>

## 3 何时可解：全纯凸域 ≠ 自动可解

这是本专题最重要的「警示」之一：**Cousin II 比 Cousin I 难得多，全纯凸域不再自动保证可解**。

**反例**：取 $D = \mathbb{C}^2 \setminus \{0\}$（去原点的 $\mathbb{C}^2$）。它是全纯凸域吗？是——但 $H^1(D, \mathcal O^*) \neq 0$。事实上，$D$ 上可以存在「没有全局非零全纯函数能实现」的除子：例如 $z_1 = 0$ 与 $z_2 = 0$ 两条超平面……（严格说，Hartogs 现象保证 $\mathcal O(\mathbb{C}^2\setminus\{0\}) = \mathcal O(\mathbb{C}^2)$，但 $\mathcal O^*$ 上同调非零）。<span class="marginnote">$H^1(D, \mathcal O^*) \neq 0$ 的几何原因：$D$ 的基本群 $\pi_1(D) = \mathbb Z$（绕原点一圈），$H^1(D,\mathbb Z) = \mathbb Z$。指数序列的长正合列中，这个 $\mathbb Z$ 会「流进」$H^1(\mathcal O^*)$，产生障碍。<strong>拓扑的洞卡住了乘法的解</strong>。</span>

**可解性判据**：若 $D$ 是全纯凸域且 $H^2(D, \mathbb Z) = 0$，则 $H^1(D, \mathcal O^*) = 0$（由长正合列：$H^1(\mathcal O)=0$ 且 $H^2(\mathbb Z)=0$，且 $H^1(\mathbb Z)$ 通常通过可缩性消失）。特别地：

- 凸域、多圆柱、多球：$H^2(\mathbb Z)=0$，Cousin II 可解。
- 有「环状洞」的区域：$H^2(\mathbb Z) \neq 0$，Cousin II 可能不可解。

## 4 公式解析：指数层序列与障碍的传递

$$
0 \to \mathbb Z \hookrightarrow \mathcal O \xrightarrow{e^{2\pi i(\cdot)}} \mathcal O^* \to 1
$$

- **第一步，核的计算**：$\exp$ 把 $g$ 送到 $e^{2\pi i g}$。若 $e^{2\pi i g} = 1$，则 $g \in \mathbb Z$（$g$ 是整值常函数）。所以 $\ker\exp = \mathbb Z$（嵌入为常函数）。满性：局部每个非零全纯函数都有对数（因为单连通小块上 $\log$ 可定义），所以 $\mathcal O \to \mathcal O^*$ 局部满——序列正合。
- **第二步，长正合列的传递机制**：给定 Cousin II 数据 $f_\alpha$，若整体解存在，$f_\alpha$ 是某 $\exp(g)$ 的限制，类为零。若 $H^1(\mathcal O)=0$，则「类为零」的障碍只剩下「$\log$ 定义出的 $g_\alpha$ 粘合时的整值偏差」——这个偏差精确落在 $H^2(\mathbb Z)$。所以 **$H^2(\mathbb Z) = 0$ 是「拓扑洞」不存在的保证**，Cousin II 可解。
- **第三步，对比 Cousin I**：Cousin I 只看 $H^1(\mathcal O)$（纯分析）；Cousin II 要看 $H^1(\mathcal O^*)$，它混合了分析（$H^1(\mathcal O)$）与拓扑（$H^1, H^2(\mathbb Z)$）。**乘法结构把拓扑带了进来**——这是二者最深刻的差别。

## 5 辨析与延伸：Cousin II 的五个要点

**辨析 1：乘法 vs 加法的本质差别**。Cousin I 用 $\mathcal O$（加法群），Cousin II 用 $\mathcal O^*$（乘法群）。$\mathcal O^*$ 的上同调比 $\mathcal O$ 复杂：$H^1(\mathcal O^*)$ 混入了整值上同调 $H^1(\mathbb Z)$ 与 $H^2(\mathbb Z)$。**乘法结构「看见」拓扑，加法结构看不见**——这是两问题最深的差别。<span class="marginnote">几何直观：乘法数据可以「绕圈」（沿环 $e^{2\pi i t}$ 转一圈回到自身），绕圈次数是整数；若区域有非平凡的洞（$H^1(\mathbb Z)\neq0$），绕圈障碍无法被 $\exp$ 消掉，Cousin II 不可解。</span>

**辨析 2：零点集必须是超曲面**。$n \geq 2$ 时，全纯函数 $f$ 的零点集 $V(f)$ 是**超曲面**（复余维 1），不可能是孤立点（Hartogs 现象！）。所以 Cousin II 的数据必须是「超曲面型零点/极点」——孤立点数据在多复变里根本不可能对应全纯函数。

**辨析 3：Weierstrass 因式分解的多维形态**。单复变的 Weierstrass 因式分解（任意离散零点集）在多复变中变成「Cousin II + 主理想化」问题。而 $n \geq 2$ 时情况更复杂：零点集虽为超曲面，但「主因子分解」不总是存在——这与 $\mathcal O_p$ 是否为 UFD 有关（是：Weierstrass 预备定理保证）。

**辨析 4：$H^2(\mathbb Z)$ 的角色**。Cousin II 可解的充分条件之一是 $H^2(D,\mathbb Z)=0$（没有「二维洞」）。这解释了为什么 $\mathbb{C}^2\setminus\{0\}$（有 $H^1(\mathbb Z)=\mathbb Z$，但 $H^2=0$）上 Cousin II……实际反例更微妙，需具体计算。**记住原则：乘法问题的障碍是「拓扑 + 分析」混合体**。

**误区清单**：

- **误区 1**：以为「Cousin II 与 Cousin I 一样好解」。
  正解：Cousin II 难得多，涉及 $\mathcal O^*$ 与整值上同调。
- **误区 2**：以为「全纯凸域上 Cousin II 自动可解」。
  正解：还需 $H^2(D,\mathbb Z)=0$ 等拓扑条件。
- **误区 3**：以为「零点集可以是离散点」。
  正解：$n\ge2$ 时零点集是超曲面，离散点不可能（Hartogs）。
- **误区 4**：以为「$\mathcal O^*$ 上同调 = $\mathcal O$ 上同调」。
  正解：指数层序列连接两者，但 $\mathcal O^*$ 多出整值部分。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 乘法问题 | multiplicative Cousin problem | Cousin II |
| 除子 | divisor | 零点/极点形式和 |
| 指数层序列 | exponential sheaf sequence | $\mathbb Z\to\mathcal O\to\mathcal O^*$ |
| 整值上同调 | integral cohomology | $H^q(D,\mathbb Z)$ |
| 超曲面 | hypersurface | 复余维 1 子簇 |
| 主除子 | principal divisor | 函数定义之除子 |

## 6 历史注记与知识树

**历史**：Cousin II 的完整解决晚于 Cousin I。关键工具是指数层序列（Cartan–Serre，1950s）：$0 \to \mathbb Z \to \mathcal O \to \mathcal O^* \to 1$ 把乘法问题归约为整值上同调 + $\mathcal O$-上同调。这一序列后来成为代数几何中「除子与线丛」的支柱，其影响远超多复变本身。

**知识树**：

- 向后：指数层序列、$\mathcal O^*$ 上同调（本组第 14 篇）。
- 向前：线丛与除子（代数几何）、Dolbeault 上同调（本组第 17 篇）。
- 横向：代数拓扑的整值上同调（第三级《代数拓扑》）——乘法问题把拓扑拉进分析。

**一句话记忆**：Cousin II = 乘法零点拼合，障碍 = $H^1(D,\mathcal O^*)$；$\mathcal O^*$ 混入整值上同调，故「拓扑的洞」能卡死分析的解。

## 7 小结

- **Cousin II 问题**：局部零点/极点数据能否乘成整体亚纯函数；障碍在 $H^1(D, \mathcal O^*)$。
- **指数层序列**：$0 \to \mathbb Z \to \mathcal O \to \mathcal O^* \to 1$ 连接分析与拓扑。
- **关键差别**：Cousin II 涉及整值上同调 $H^q(D,\mathbb Z)$；拓扑的洞（如 $\mathbb{C}^2\setminus\{0\}$）会造成不可解障碍。
- **判据**：全纯凸域 + $H^2(D,\mathbb Z)=0$ ⇒ Cousin II 可解。

在下一节，我们把 $\mathcal O$-上同调与 $\bar\partial$