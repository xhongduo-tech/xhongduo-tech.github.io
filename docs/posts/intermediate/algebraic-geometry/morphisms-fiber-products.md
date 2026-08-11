---
title: 概形的态射与纤维积
date: 2026-08-11
---

# 概形的态射与纤维积

<div class="epigraph">
<p>如果概形理论只有一条定理，那它应该是"纤维积存在"。</p>
<footer>—— 由亚历山大 · 格罗滕迪克（Alexander Grothendieck）思想转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从态射继续

上一节我们把"几何对象"升级成了概形。现在要给概形之间装上"合法映射"——**态射**。态射在簇理论里是"局部多项式映射"，在概形理论里被重新表述为**局部环化空间之间的连续映射**：拉回函数环的同时保持"在点处消失"的结构。

本节真正的主角是**纤维积（fiber product）**。它是代数几何里最"基建"的操作：给定 $f: X \to S$ 与 $g: Y \to S$，纤维积 $X \times_S Y$ 是"在 $S$ 上同时兼容 $X$ 和 $Y$ 的最大对象"。为什么重要？因为**几乎所有现代代数几何都以"在基 $S$ 上"为默认语境**：曲线族是"在基上的态射"，一族曲线的"纤维"（$S$ 中的每个点拉到的一根曲线）正是用纤维积定义的。在算术几何里，"改变基"（如把 $\mathbb{Q}$ 上的方程换成 $\mathbb{Z}$ 上、再换成 $\mathbb{F}_p$ 上）就是用基变换完成的。本节是理解第 7 篇（分离/真态射）、第 13 篇（曲线族）、第 16 篇（GAGA）的共同脚手架。

## 1 态射：把环拉回来

**核心概念：概形的态射（morphism of schemes）**：局部环化空间的态射 $f: X \to Y$ 由一对 $(f, f^\#)$ 组成：连续映射 $f: X \to Y$，加上**结构层拉回** $f^\#: \mathcal{O}_Y \to f_* \mathcal{O}_X$（每个开集 $V \subseteq Y$ 上的环同态 $\mathcal{O}_Y(V) \to \mathcal{O}_X(f^{-1}V)$），并且要求在每点保持局部结构：诱导的茎映射

$$f^\#_P: \mathcal{O}_{Y, f(P)} \longrightarrow \mathcal{O}_{X, P}$$

把 $Y$ 在 $f(P)$ 处的极大理想映到 $X$ 在 $P$ 处的极大理想内。<span class="marginnote">最后一条「局部同态」不是多余条件：它保证"在点处取零值的函数"被拉回后仍在取零值——即<strong>函数的值保持良定</strong>。没有它，态射可能把"函数在点 $f(P)$ 的值"拉回成"在 $P$ 处非零"，破坏几何直觉。</span>

对**仿射概形**，态射理论变成干净的环同态理论：<span class="marginnote">这是整个概形理论最甜美的第一定理：几何的态射 = 代数的环同态，方向反转（反变等价）。一切"概形的态射"问题，局部地都化归为"交换环的环同态"问题。</span>

**重点：$\operatorname{Spec}$ 是反变等价。** 对任意环同态 $\varphi: B \to A$，它诱导态射

$$\operatorname{Spec} \varphi: \operatorname{Spec} A \longrightarrow \operatorname{Spec} B, \qquad \mathfrak{p} \longmapsto \varphi^{-1}(\mathfrak{p})$$

且**每个**态射 $\operatorname{Spec} A \to \operatorname{Spec} B$ 都来自唯一的环同态 $B \to A$。于是

$$\{\text{态射 } \operatorname{Spec} A \to \operatorname{Spec} B\} \longleftrightarrow \{\text{环同态 } B \to A\}$$

例子：环同态 $k[x] \hookrightarrow k[x, y]$ 对应态射 $\mathbb{A}^2_k \to \mathbb{A}^1_k$（投影到 $x$ 轴）；环同态 $k \to k[x]$ 对应 $\mathbb{A}^1_k \to \operatorname{Spec} k$——"把平面放到一个点上"。

## 2 纤维积：拉回方块

**核心概念：纤维积（fiber product）**：给定态射 $f: X \to S$ 与 $g: Y \to S$，**纤维积** $X \times_S Y$ 是"在 $S$ 上的相容拉回"：它是满足下述**拉回方块**的概形

$$\begin{array}{ccc}
X \times_S Y & \xrightarrow{p_2} & Y \\
\Big\downarrow p_1 & & \Big\downarrow g \\
X & \xrightarrow{f} & S
\end{array}$$

并且具有**泛性质**：对任意概形 $T$ 与态射 $u: T \to X$、$v: T \to Y$ 使 $f \circ u = g \circ v$，存在唯一态射 $T \to X \times_S Y$ 使两个三角形交换。泛性质保证 $X \times_S Y$（如果存在）在同构意义下唯一。<span class="marginnote">泛性质是范畴论里"最好的对象"的标准表达：它不是"我们手工构造的一个概形"，而是"一切与 $X, Y, S$ 兼容者都唯一穿过它"的那个对象。同构意义下唯一——所以不同构造殊途同归。</span>

**存在性定理：** 概形的纤维积**总是存在**。仿射情形特别简单：

$$\operatorname{Spec} A \times_{\operatorname{Spec} C} \operatorname{Spec} B = \operatorname{Spec} (A \otimes_C B)$$

即"纤维积 = 张量积的谱"。一般情形通过把 $X, Y$ 切成仿射图、在图上做张量积、再胶合而得到。<span class="marginnote">几何的"拉回方块"在仿射图上一律翻译成代数的"张量积"。例：$\mathbb{A}^1 \times_{\operatorname{Spec} k} \mathbb{A}^1 = \operatorname{Spec} k[x] \otimes_k k[y] = \operatorname{Spec} k[x, y] = \mathbb{A}^2$——平面是两个 $\mathbb{A}^1$ 在 $k$ 上的纤维积，正如平面是两条线的乘积。</span>

**核心概念：纤维（fiber）**：态射 $f: X \to S$ 在点 $s \in S$ 上的**纤维**是

$$X_s = X \times_S \operatorname{Spec} \kappa(s)$$

其中 $\kappa(s) = \mathcal{O}_{S,s}/\mathfrak{m}_s$ 是 $s$ 处的留数域。<span class="marginnote">对闭点 $s$，$\kappa(s) = k$，$X_s$ 就是"$S$ 中这个点对应的那个几何对象"。对 $S = \operatorname{Spec} \mathbb{Z}$ 的算术情形，$s = (p)$ 时 $\kappa(s) = \mathbb{F}_p$，$X_s$ 是"模 $p$ 化简"——基变换让一族曲线在每一个素数特征"重获新生"。</span>纤维积的存在性说明：**"点"本身也是基**，所以"一族对象 + 每个纤维"这个图像被纳入单一概念。

## 3 基变换与几何性质

**核心概念：基变换（base change）**：给定 $f: X \to S$ 与任意态射 $S' \to S$，定义

$$X' = X \times_S S'$$

称 $f': X' \to S'$ 是 $f$ 沿 $S' \to S$ 的**基变换**，$f'$ 是 $f$ 的**拉回**。<span class="marginnote">基变换就是"把整个图景搬到一个新的基上"。欧氏比喻：把曲线族从实数域搬到复数域、从 $\mathbb{Q}$ 搬到 $\mathbb{F}_p$、或者限制到一个子空间——统统是基变换的特例。它是算术几何的第一动词。</span>

**重点：良好的几何性质沿基变换保持。** 例如：若 $f$ 是"有限型、分离、真、平坦"等性质之一，则任何基变换 $f'$ 保持同一性质。这使"证明在某个标准基上成立 → 对任意基成立"成为常规论证。<span class="marginnote">例如想研究"一般纤维的几何"（在 $S$ 的一般点上），可以先把基变到"某个点上"，把问题化到"一个点上的几何"，性质不丢失。这类论证在算术几何与模空间理论里反复出现。</span>

**辨析｜易错点：** 纤维积不是"乘积"的简单推广。$X \times_S Y$ 只有在 $S = \operatorname{Spec} k$ 时才退化为"通常乘积" $X \times_k Y$；一般的 $S$ 上，"兼容于 $S$"的约束会让维数下降。初学者常见错误：把 $X \times_S Y$ 想成"两个簇的直接乘积"。事实上 $X \times_S Y \to X \times_k Y$ 是"对角嵌入限制到 $S$"——直觉应该换成"在 $S$ 上求交"。

## 4 纤维的维数：几何直觉的检验

纤维积的泛性质加上维数理论，给出一句非常有用的直觉命题：设 $f: X \to Y$ 是有限型态射（可设想为"一族簇"），则

$$\dim X_y \ge \dim X - \dim Y$$

其中 $X_y$ 是 $y$ 处的纤维。<span class="marginnote">对 $f: \mathbb{A}^2 \to \mathbb{A}^1$（投影），$y$ 处纤维是一根直线，$\dim X_y = 1$，恰取下等号。若 $f$ 是平坦的，则等号处处成立——"纤维维数恒定"，这是"平坦"概念的意义之一。</span>这句话把"纤维积的几何"和"维数的连续性"挂钩，是"一族几何对象的尺寸如何随参数变化"的代数表述——在模空间理论（第 13 篇曲线的模）里，它决定"曲线族在退化点是否坍缩"。

## 5 公式解析：仿射纤维积 = 张量积

$$
\operatorname{Spec} A \times_{\operatorname{Spec} C} \operatorname{Spec} B = \operatorname{Spec} \left( A \otimes_C B \right)
$$

分三步拆解：

- **第一步，为什么张量积**：张量积 $A \otimes_C B$ 满足"从 $A$ 和 $B$ 的 $C$-线性信息生成最小可交换代数"的泛性质——正是"在 $C$ 上兼容 $A$ 与 $B$ 的最小对象"的代数版。$\operatorname{Spec}$ 把它送回去，得到"在 $S = \operatorname{Spec} C$ 上兼容 $X, Y$ 的最小概形"。<span class="marginnote">张量积 = "联立约束下的自由乘积"。例：$k[x] \otimes_k k[y] = k[x, y]$（无约束的自由乘积 = 多元多项式环）；若 $B = k[x]/(x^2)$，则 $A \otimes_k B = A[x]/(x^2)$——幂零元被张量积继承，纤维积保留了"无穷小结构"。</span>
- **第二步，为什么点对应素理想**：$\operatorname{Spec}(A \otimes_C B)$ 的点是 $A \otimes_C B$ 的素理想，它们恰是"同时兼容 $A$ 与 $B$ 的素理想对"的代数化身。这与"纤维 = 在点上的拉回"完全一致。
- **第三步，回到几何直觉**：对 $C = k$、$A = k[x]$、$B = k[y]$，得 $k[x,y]$，即 $\mathbb{A}^2$。**平面 = 两条直线的纤维积**。这个最小例子背后是整套操作逻辑：任何"在基上联立"的几何问题，都可以翻译成环的张量积问题，代数解决后再翻译回来。

一句话直觉：**纤维积 = "在基上求交"；仿射情形 = "环的张量积"**。基变换则让"换基"成为一条可随时执行的命令，把一族对象的所有"切片"统一在一个对象里。

## 6 小结

- **态射**：连续映射 + 结构层拉回 + 局部同态保持；仿射情形 ⟺ 环同态（反变等价）。
- **纤维积** $X \times_S Y$：在 $S$ 上兼容的最大对象，由拉回方块与泛性质唯一确定，**总是存在**。
- **仿射公式**：$\operatorname{Spec} A \times_{\operatorname{Spec} C} \operatorname{Spec} B = \operatorname{Spec}(A \otimes_C B)$。
- **纤维** $X_s$：把"点"当基，"一族对象的切片"精确化为基变换的特例。
- **基变换**：$X' = X \times_S S'$；有限型、分离、真、平坦等性质沿基变换保持。

在下一节，我们讨论两个"分离性"概念：**分离态射与真态射**——用对角态射和判别准则（valuative criterion）区分"像流形"的概形与"像紧空间"的概形。
