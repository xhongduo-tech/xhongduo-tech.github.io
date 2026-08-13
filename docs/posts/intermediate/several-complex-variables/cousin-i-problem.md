---
title: Cousin I 问题与加法问题
date: 2026-08-07
---

# Cousin I 问题与加法问题

<div class="epigraph">
<p>给定每个点附近的主子部，能否把它织成整个区域的亚纯函数？——这是 Mittag-Leffler 命题的多复变灵魂。</p>
<footer>—— 仿 戈斯塔 · 米塔-列夫勒（Gösta Mittag-Leffler）与 皮埃尔 · 库桑（Pierre Cousin）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第4章；史济怀 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从 Cousin I 问题开始

单复变的 **Mittag-Leffler 定理** 说：在区域 $D$ 上预先指定任意离散极点的**主子部**（主项），总能找到一个亚纯函数恰好以这些为主部。多复变把这个命题整体化：给定 $D$ 上**局部定义的亚纯函数**（每点附近是一个主子部），能否拼成一个**全局亚纯函数**？这就是 **Cousin I 问题（加法问题）**。它之所以重要，是因为它把「构造有指定奇点的函数」这个分析任务，**彻底翻译成了层上同调的语言**——$H^1(D, \mathcal O) = 0$ 与否。<span class="marginnote">「加法」的名字来自叠加原理：主子部之间是<strong>加法</strong>关系（$f \sim g$ 若 $f-g$ 全纯）。Cousin 姊妹问题成对出现：I 是加法（主部），II 是乘法（零点，下一节）。加法的障碍住在 $H^1(\mathcal O)$，乘法的障碍住在 $H^1(\mathcal O^*)$。</span>

## 1 问题陈述：加法 Cousin 问题

设 $D \subset \mathbb{C}^n$ 是区域。给定 $D$ 的一个开覆盖 $\{ U_\alpha \}$ 与一族亚纯函数 $m_\alpha$（在 $U_\alpha$ 上），满足**相容性**：

$$
m_\alpha - m_\beta \in \mathcal O(U_\alpha \cap U_\beta) \qquad (\text{在交集上差全纯})
$$

**Cousin I 问题**：是否存在 $D$ 上的整体亚纯函数 $m$，使 $m - m_\alpha$ 在每个 $U_\alpha$ 上全纯？

直观：$m_\alpha$ 给出「局部主部」，相容性保证局部主部在重叠区**一致地**符合（只差全纯部分——这无关紧要，因为全纯部分不影响主部）。问的是：能否把这套局部主部「缝合」成整体主部。<span class="marginnote">为什么 $m_\alpha - m_\beta$ 全纯就够？因为亚纯函数的「主部」定义到差一个全纯函数为止；重叠区两种主部差全纯，意味着它们描述了同一套奇点结构。这正是层论里 $\mathcal M / \mathcal O$（亚纯模全纯）商的自然表述——Cousin I 的障碍正是 $H^1(D, \mathcal O)$。</span>

## 2 层论翻译：为什么障碍是 $H^1(D, \mathcal O)$

把问题装进层：设 $\mathcal M$ 为亚纯函数层，$\mathcal O \subset \mathcal M$。主子部数据 $m_\alpha$ 定义的是商层 $\mathcal M/\mathcal O$ 的整体截面。由长正合列（指数……不，加法序列）：

$$
0 \to \mathcal O \to \mathcal M \to \mathcal M/\mathcal O \to 0
$$

取上同调得：

$$
0 \to \mathcal O(D) \to \mathcal M(D) \to (\mathcal M/\mathcal O)(D) \xrightarrow{\;\delta\;} H^1(D, \mathcal O) \to H^1(D, \mathcal M) \to \cdots
$$

**核心结论**：

> **Cousin I 问题可解（对任意主子部数据） ⟺ $H^1(D, \mathcal O) = 0$。**

理由：给的主子部数据是 $(\mathcal M/\mathcal O)(D)$ 的元素，整体解存在 ⟺ 它落在 $\mathcal M(D)$ 的像里 ⟺ 边界映射 $\delta$ 把它送到 $H^1$ 的像是零。若 $H^1 = 0$，则 $\delta$ 恒为零，一切数据可解。<span class="marginnote">这是一个极其漂亮的「分析问题 ⟺ 上同调消失」的例子。而单复变的 Mittag-Leffler 定理就是「$H^1(D, \mathcal O) = 0$」在 $n=1$ 时的结论——因为单复变中任意区域是全纯域，而全纯域的 $\mathcal O$-上同调在第一层为零（Dolbeault 引理）。</span>

## 3 何时 $H^1(D, \mathcal O) = 0$

现在问题变成：**什么样的 $D$ 有 $H^1(D, \mathcal O) = 0$？** 答案与全纯域理论完美衔接：

**定理（Cartan–Serre / Hörmander）**：若 $D \subset \mathbb{C}^n$ 是**全纯凸域**（等价地伪凸），则

$$
H^q(D, \mathcal O) = 0 \qquad \forall q \geq 1
$$

**推论**：全纯凸域上 **Cousin I 问题恒可解**——这是 Oka–Weil 逼近 + 层上同调的结合。$H^q = 0$（$q \geq 1$）的证明路径：先证 Dolbeault 上同调 $H^{0,q}_{\bar\partial}(D) = 0$（用 $\bar\partial$ 方程可解，Hörmander L² 方法），再由 Dolbeault 定理 $H^q(D, \mathcal O) \cong H^{0,q}_{\bar\partial}(D)$。<span class="marginnote">这条链把「伪凸」与「上同调消失」完全打通：伪凸 ⇒ $\bar\partial$ 可解 ⇒ Dolbeault 上同调消失 ⇒ $\mathcal O$-上同调消失 ⇒ Cousin I 可解。几乎整本 Hörmander 前六章都在这条链上。下一节（Dolbeault 上同调与 $\bar\partial$ 方程）将补上缺失的一环。</span>

## 4 公式解析：长正合列中的边界映射 $\delta$

$$
(\mathcal M/\mathcal O)(D) \xrightarrow{\; \delta \;} H^1(D, \mathcal O), \qquad [m_\alpha] \longmapsto \left[ f_{\alpha\beta} = m_\beta - m_\alpha \right]
$$

- **第一步，读出 $\delta$ 的定义**：给定主子部数据 $m_\alpha$（重叠区差全纯），定义 $f_{\alpha\beta} = m_\beta - m_\alpha$（或 $m_\alpha - m_\beta$，符号约定不一）。它满足上闭链条件：$f_{\alpha\beta} - f_{\alpha\gamma} + f_{\beta\gamma} = 0$（三交上 $m$ 的差可消）。所以 $f$ 是 $H^1(\mathcal U, \mathcal O)$ 的上闭链。
- **第二步，为什么 $\delta[m_\alpha] = 0$ 等价于可解**：若存在整体 $m$ 使 $m - m_\alpha$ 全纯，则 $f_{\alpha\beta} = m_\beta - m_\alpha = (m_\beta - m) - (m_\alpha - m)$ 是两个全纯函数之差——恰好是上边缘（$f_{\alpha\beta} = g_\beta - g_\alpha$，取 $g_\alpha = m - m_\alpha$）。反之若 $f$ 是上边缘 $g_\beta - g_\alpha$，则 $m_\alpha + g_\alpha$ 拼成整体亚纯函数。所以「$\delta$ 送零」⟺「上闭链是上边缘」⟺「可解」。
- **第三步，整体图景**：Cousin I 的障碍「住在」$H^1$。若 $H^1 = 0$，所有障碍消失——这正是 §3 定理在分析上的回响。

## 5 辨析与延伸：Cousin I 的五个要点

**辨析 1：Cousin I 是 Mittag-Leffler 的多维推广**。单复变的 Mittag-Leffler 定理允许任意离散极点集；多复变的 Cousin I 允许任意「局部主部」。$n=1$ 时两者重合，且一切区域可解（因为 $H^1(D,\mathcal O)=0$）；$n \geq 2$ 时可解性依赖区域的伪凸性。<span class="marginnote">所以 Cousin I 把单复变「免费」的构造问题，变成了多复变「收费」的存在性问题——费用由区域几何决定。</span>

**辨析 2：主子部 vs 主部**。主子部（principal part）是亚纯函数在奇点附近的负幂部分；Cousin I 给的是「局部主子部」$m_\alpha$。它们只定义到「差一个全纯函数」，所以相容性条件是差全纯。**「模全纯」是贯穿 Cousin 理论的隐藏商结构**。

**辨析 3：加法结构的来源**。$m_\alpha - m_\beta$ 全纯是加法相容性。为什么是加法？因为主部叠加（Mittag-Leffler 的核心理念）是「把奇点部分相加」。**Cousin I 是「加法问题」，Cousin II 是「乘法问题」**——一个对应 $\mathcal O$，一个对应 $\mathcal O^*$。

**辨析 4：$H^1=0$ 的「充分必要」性**。Cousin I 恒可解 ⟺ $H^1(D,\mathcal O)=0$。这是「分析问题 = 上同调问题」的精确表达。理解这一点，比记住定理本身更重要：**每个 Cousin 型问题都对应一个层的上同调**。

**误区清单**：

- **误区 1**：以为「Cousin I 与 Mittag-Leffler 无关」。
  正解：Cousin I 正是 Mittag-Leffler 的多维推广。
- **误区 2**：以为「主部必须用极点」。
  正解：主部是负幂部分，可由任意局部亚纯函数给出。
- **误区 3**：以为「全纯凸域上 Cousin I 需额外条件」。
  正解：$H^q(D,\mathcal O)=0$（$q\ge1$）自动成立，无需额外条件。
- **误区 4**：以为「可解性只依赖区域大小」。
  正解：依赖区域的**伪凸性**（几何），不是大小。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 主部 | principal part | 负幂部分 |
| 亚纯函数 | meromorphic function | 局部可写成分式 |
| 加法问题 | additive Cousin problem | Cousin I |
| 模全纯 | modulo holomorphic | 差全纯等价 |
| 上同调障碍 | cohomology obstruction | $H^1$ 元素 |
| 边界映射 | connecting map | 长正合列中 $\delta$ |

## 6 历史注记与知识树

**历史**：Cousin 1895 年提出他的两个问题；单复变情形早已由 Mittag-Leffler（1884）解决。多复变的完整解答依赖层论：Oka 与 Cartan 在 1940s 证明「$H^1(D,\mathcal O)=0$ 在伪凸域成立」，从而 Cousin I 恒可解。这一成果是「分析问题层论化」的首次大规模胜利。

**知识树**：

- 向后：层上同调与长正合列（本组第 14 篇）。
- 向前：Cousin II（本组第 16 篇）、Dolbeault 上同调（本组第 17 篇）。
- 横向：亚纯函数与除子理论（代数几何）。

**一句话记忆**：Cousin I = 加法主部拼合，障碍 = $H^1(D,\mathcal O)$；伪凸域上 $H^1=0$，恒可解。

## 7 小结

- **Cousin I 问题**：局部主部数据能否拼成整体亚纯函数；「加法」源于主部的加法叠加。
- **层论翻译**：障碍 = $H^1(D, \mathcal O)$；$H^1 = 0$ ⟺ 恒可解。
- **全纯凸域**：$H^q(D, \mathcal O) = 0$（$q \geq 1$）⇒ Cousin I 恒可解。
- **证明链**：伪凸 ⇒ $\bar\partial$ 可解 ⇒ Dolbeault 上同调消失 ⇒ $H^q(\mathcal O) = 0$。

在下一节，我们转向 Cousin 问题的另一半：**Cousin II 问题（乘法问题）**——给定零点，找整体全纯函数，它的障碍住在乘法群层 $H^1(\mathcal O^*)$