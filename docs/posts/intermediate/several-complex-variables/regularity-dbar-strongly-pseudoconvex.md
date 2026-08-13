---
title: 强伪凸域上 ∂̄ 方程的正则性
date: 2026-08-07
---

# 强伪凸域上 ∂̄ 方程的正则性

<div class="epigraph">
<p>存在性只是门票，正则性才是演出——解的光滑程度决定一切定理的可用性。</p>
<footer>—— 仿 约瑟夫 · 科恩（Joseph J. Kohn），《∂̄ 方程与边界正则性》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第6章；史济怀 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从正则性开始

Hörmander L² 估计给出的是 **$L^2$ 解**——存在性好得惊人，但光滑性没保证。很多场合（Bergman 核、CR 理论、积分表示）需要**光滑解**。何时能把 $L^2$ 解升级成 $C^\infty$ 解？答案取决于区域的边界几何：**强伪凸域**上，$\bar\partial$ 方程有「椭圆型」的正则性理论，解与数据同阶光滑；一般伪凸域则不行——这是「强」与「弱」伪凸最实质的分野。<span class="marginnote">Kohn 在 1960 年代系统地建立了强伪凸域上的 $\bar\partial$-Neumann 问题：通过椭圆估计 + 次椭圆估计，证明 $H^{0,q}_{\bar\partial}$ 中每个上同调类有光滑代表元，且解算子保持 Sobolev 光滑性。这是 Hörmander 理论向边界理论的延伸。</span>

## 1 内部正则性：容易的一半

先看**内部**（远离边界）正则性。对内部的光滑 $g$，$L^2$ 解 $u$ 是否内部光滑？

**是。** 原因是 $\bar\partial$ 在内部是**椭圆**算子的组成部分：$\bar\partial\bar\partial^* + \bar\partial^*\bar\partial = \square$ 是（加权的）**Laplacian 型算子**，椭圆。椭圆算子解的经典正则性定理（Weyl 引理）保证：**若 $g$ 在 $D$ 的内部光滑，则解 $u$ 在内部光滑**。<span class="marginnote">具体地：$\bar\partial u = g$ 且 $g$ 光滑 ⇒ $u$ 满足 $\square u = \bar\partial^* g$（一个椭圆方程），由椭圆正则性 $u$ 光滑。所以「内部正则性」几乎是免费的——真正的困难全在<strong>边界</strong>上。</span>

所以正则性的全部问题浓缩为：**解在边界附近的行为**。

## 2 边界正则性与 $\bar\partial$-Neumann 问题

为了控制边界行为，需要给 $\bar\partial$ 方程配**边界条件**。经典的框架是 **$\bar\partial$-Neumann 问题**：

$$
\begin{cases}
\bar\partial u = g, \\
\bar\partial^* u = 0, \\
u \;\text{满足边界条件：}\; u \perp \text{切向部分}
\end{cases}
$$

这里 $\bar\partial^*$ 的共轭在带边流形上取，边界条件（Dirichlet 型的「$\bar\partial^*$ 边界条件」）保证 $u$ 在边界上的值被恰当约束。<span class="marginnote">$\bar\partial$-Neumann 问题是「古典 Neumann 问题（$\Delta u = f$，$\partial u/\partial n = 0$）」的复变翻版。Kohn 的方法：先把问题化为 $\square = \bar\partial\bar\partial^* + \bar\partial^*\bar\partial$ 在「$\bar\partial$-闭形式」子空间上的可解性，再用<strong>次椭圆估计</strong>处理边界。</span>

**Kohn 次椭圆估计**：对强伪凸域 $D$ 的边界点，有

$$
\|u\|^2_{s+1/2} \;\leq\; C_s \left( \|\bar\partial u\|^2_s + \|\bar\partial^* u\|^2_s + \|u\|^2_{-N} \right), \qquad \forall s \geq 0
$$

其中 $\|\cdot\|_s$ 是 Sobolev 范数。**关键：正则性增益是 $1/2$ 阶**——这是「次椭圆」：不是全阶（+1）增益，而是半阶。半阶增益足以证明无限光滑性（叠代），且是强伪凸边界独有的特征。<span class="marginnote">为什么是 $1/2$？强伪凸性使 Levi 形式正定，边界「竖」起了 $n-1$ 个复切方向；在这些方向上方程是椭圆的，在法向上退化。半阶损失来自法向。若 Levi 形式有零特征值（一般伪凸），损失更多，甚至可能只有任意小正阶——这就是强伪凸与一般伪凸在正则性上的天壤之别。</span>

## 3 强伪凸 ⇒ 正则解：主定理

**Kohn 定理**：设 $D$ 是具 $C^\infty$ 边界的**强伪凸**域。则对任意 $q \geq 1$，$\bar\partial$-Neumann 问题有解算子 $N_q$（**$\bar\partial$-Neumann 解算子 / 格林算子**），使得：

$\bar\partial N_q = I$ 在 $\bar\partial$-闭的 $(0,q)$-形式上（即 $\bar\partial N_q g = g$）；
$N_q$ 保持光滑性：$g$ 光滑 ⟹ $N_q g$ 光滑，且连续映射于 Sobolev 空间 $H^s \to H^{s+1/2}$。

**推论**：强伪凸域上，$\bar\partial u = g$（$g$ 光滑、$\bar\partial g = 0$）有**光滑**解 $u$。<span class="marginnote">这比 Hörmander 的 $L^2$ 存在性强得多：不仅是存在，还保光滑。Bergman 核（下节）的 $C^\infty$ 光滑性、CR 函数的延拓、积分表示的边界值理论，全都依赖这条「保光滑」性质。Kohn 的方法（1963）也是后来次椭圆算子和 Kohn 雪花理论的开端。</span>

## 4 公式解析：次椭圆估计的增益

$$
\|u\|^2_{s+1/2} \;\lesssim\; \|\bar\partial u\|^2_s + \|\bar\partial^* u\|^2_s + \|u\|^2_{-N}
$$

- **第一步，认识 Sobolev 范数**：$\|u\|_s$ 度量「$u$ 有 $s$ 阶导数在 $L^2$」；$s$ 越大越光滑。$+1/2$ 表示**半阶提升**：数据有 $s$ 阶导数，解有 $s+1/2$ 阶。
- **第二步，为什么这足以推出 $C^\infty$**：叠代：$u \in H^{s+1/2}$ 后，代入右端让 $g$ 的数据提升，得到 $u \in H^{s+1}$，再 $H^{s+3/2}$，…… 无限提升至任意阶——Sobolev 嵌入定理给出 $C^\infty$。**半阶提升在叠代下「滚雪球」成全阶**。
- **第三步，强伪凸的角色**：估计的常数 $C_s$ 依赖边界 Levi 形式正定（有下界 $\delta>0$）。若 Levi 形式只是半正定（一般伪凸），右端要多加「边界邻域内的低阶项」，提升可能降到任意小——正则性大幅减弱。**强伪凸性 = 正则性的「安全气囊」**。

## 5 辨析与延伸：正则性的五个要点

**辨析 1：内部正则性是「免费」的，边界正则性是「付费」的**。$\bar\partial$ 在内部是椭圆（$\square$ 是 Laplacian），Weyl 引理免费给出内部光滑性；边界上算子退化（法向方向非椭圆），必须靠次椭圆估计「买回」半阶。**「内部免费、边界付费」是 PDE 正则性理论的普遍规律**。<span class="marginnote">为什么法向退化？因为 $\bar\partial$ 沿实法向方向没有复结构——那里算子不再是椭圆。强伪凸性（Levi 正定）恰好把复切方向的椭圆性「锚住」，次椭圆估计由此而来。</span>

**辨析 2：$1/2$ 阶增益的意义**。次椭圆估计 $\|u\|_{s+1/2} \lesssim \|\bar\partial u\|_s + \|\bar\partial^*u\|_s$ 只给半阶提升。半阶看似少，但叠代后滚成全阶：$s \to s+\frac12 \to s+1 \to \cdots \to \infty$。**半阶是「种子」，叠代是「成长」**。

**辨析 3：强伪凸 vs 一般伪凸的正则性鸿沟**。一般伪凸域上 $\bar\partial$ 方程只有 $L^2$ 存在性，解的边界行为不可控；强伪凸域上有完整 Sobolev 正则性。这条鸿沟解释了为什么 Bergman 核、Szegő 核的显式性质只在强伪凸域上成立。

**辨析 4：$\bar\partial$-Neumann 算子是什么**。$N$ 是 $\square = \bar\partial\bar\partial^* + \bar\partial^*\bar\partial$ 在合适边界条件下的逆（格林算子）。$N$ 的正则性直接翻译成 $\bar\partial$ 方程解的正则性。**$N$ 是「正则性的总开关」**——$N$ 保光滑，则一切保光滑。

**误区清单**：

- **误区 1**：以为「$L^2$ 解自动光滑」。
  正解：内部自动光滑；边界光滑性依赖强伪凸。
- **误区 2**：以为「次椭圆 = 椭圆」。
  正解：次椭圆只给半阶增益，比椭圆弱。
- **误区 3**：以为「一般伪凸域也有 Kohn 定理」。
  正解：Kohn 定理需要强伪凸。
- **误区 4**：以为「正则性估计是纯技术」。
  正解：它是 Bergman/Szegő 核光滑性的根基。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 正则性 | regularity | 解的光滑程度 |
| Sobolev 范数 | Sobolev norm | $\|u\|_s$ |
| 次椭圆 | subelliptic | 半阶增益 |
| $\bar\partial$-Neumann | $\bar\partial$-Neumann | 带边界的 $\square$ 问题 |
| Kohn Laplacian | Kohn Laplacian | $\square_b$ |
| 椭圆正则性 | elliptic regularity | Weyl 引理 |

## 6 历史注记与知识树

**历史**：Morrey（1958）用 $\bar\partial$-Neumann 思想处理复椭圆方程；Kohn（1963–64）建立强伪凸域上次椭圆估计与解算子；Hörmander 的 $L^2$ 框架与 Kohn 的次椭圆框架互补，共同奠定边界正则性理论。后来 Catlin 等研究一般伪凸域的正则性指数，揭示「有限型」条件的精细作用。

**知识树**：

- 向后：Hörmander $L^2$ 估计（本组第 20 篇）、强伪凸域（第 2 组）。
- 向前：Bergman 核（本组第 22 篇）——核的正则性由 $N$ 的正则性决定。
- 横向：椭圆 PDE 的正则性理论（第二级《偏微分方程》）。

**一句话记忆**：内部免费光滑、边界付费半阶；强伪凸 ⟹ 次椭圆 ⟹ $N$ 保光滑 ⟹ 一切核与解保光滑。

## 7 小结

- **内部正则性**：$\bar\partial$ 在内部椭圆，$g$ 光滑 ⟹ 解内部光滑——免费。
- **边界正则性**：需 $\bar\partial$-Neumann 问题 + **次椭圆估计**，增益 $1/2$ 阶。
- **Kohn 定理**：强伪凸域上 $\bar\partial$ 方程有保光滑的解算子（$\bar\partial$-Neumann 算子）。
- **强 vs 一般伪凸**：$1/2$ 阶增益是强伪凸的专利；一般伪凸只保证 $L^2$ 存在性。

在下一节，我们用正则性理论构造多复变最重要的积分核：**Bergman 核**——从 $\bar\partial$ 方程的 $L^2$