---
title: C\*-代数 K 理论初步
date: 2026-08-07
---

# C\*-代数 K 理论初步

<div class="epigraph">
<p>让优秀的人像奴隶一般耗在计算的苦役里，是不配的——若用机器，本可安然解脱。</p>
<footer>—— 戈特弗里德 · 莱布尼茨（Gottfried Wilhelm Leibniz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Murphy《C\*-Algebras and Operator Theory》第5章 ｜ 2026-08-07</p>
</div>

## 为什么从 K 理论开始

第 17 篇的 Elliott 定理已经让 $K_0$ 出场，却只把它当黑箱。这一节要打开它：**K 理论**是 1950 年代 Grothendieck 在代数几何中发明的「投影/向量丛的分类学」，1960 年代被 Atiyah–Hirzebruch 移植到拓扑，1980 年代由 Kasparov、Pimsner–Voiculescu 等全面建立于 C\* 代数。它的口号是：**把「有多少个本质不同的投影/酉元」变成一个阿贝尔群**。

K 理论为什么是 C\* 代数分类的胜负手？因为它是**同伦不变、保直和、对理想与商有六项正合列**的函子——它能「数清」投影的稳定等价类（$K_0$）与酉元的同伦类（$K_1$），而这些恰好是 C\* 代数最深层的离散不变量。理解 K 理论，就握住了分类理论（第 26 篇）、指标理论（第 5 篇的回响）与非交换几何的公共引擎。

## 1 从投影到 K₀：造群的 Grothendieck 技巧

**稳定等价（stable equivalence）**：投影 $p,q\in M_\infty(A)=$（$A$ 的矩阵扩张 $\bigcup M_n(A)$）称为**稳定等价**，若存在 $v$ 使 $p=v^*v$、$q=vv^*$（即 $p\sim_0q$：加零块后 Murray–von Neumann 等价）。等价类记 $[p]$。

**半群**：$[p]+[q]=[\mathrm{diag}(p,q)]$（直和）给出交换半群 $V(A)$。

**K₀ 群（K$_0$-group）**：$K_0(A)$ 是 $V(A)$ 的 **Grothendieck 群**：对每个 $[p]-[q]$ 形式地引入「负元」，$K_0(A)=\{[p]-[q]:p,q\in M_\infty(A)\}/$（等价关系）。<span class="marginnote">Grothendieck 的招数：把「只能加不能减」的半群，强行补出负元变成群——就像自然数到整数的扩充。$K_0(A)$ 的加法是直和，负元是形式差。AF 代数的 $K_0$ 是「有序群」（第 17 篇），一般 C\* 代数的 $K_0$ 是「有单位元的群」。</span>

**基本例子**：
$K_0(\mathbb{C})=\mathbb{Z}$（$[p]=\mathrm{rank}\,p$）；
$K_0(M_n)=\mathbb{Z}$，$K_0(\mathcal{K})=\mathbb{Z}$；
- $K_0(C(X))=$ 复向量丛的稳定同构类群 $K^0(X)$（向量丛 ↔ 投影的连续族）；
- $K_0(C_0(\mathbb{R}^2))=\mathbb{Z}$（Bott 元素的预演）；
- $K_0(A_\theta)=\mathbb{Z}+\theta\mathbb{Z}$（第 15 篇的 Rieffel 投影）。

## 2 K₁：酉元的同伦类

**K₁ 群（K$_1$-group）**：$K_1(A)=\mathrm{GL}_\infty(A)/\mathrm{GL}_\infty(A)_0$，其中 $\mathrm{GL}_\infty(A)$ 是可逆矩阵的稳定群，$\mathrm{GL}_\infty(A)_0$ 是含单位元的连通分支。等价地，$K_1(A)=U_\infty(A)/U_\infty(A)_0$（酉元模同伦）。

**命题（K₁ 的两种口味）**：$K_1(A)$ 度量「本质不同的酉元」：$[u]$ 是 $u$ 的同伦类。对 $A=C(\mathbb{T})$，$K_1(A)=\mathbb{Z}$，同构由 $\mathrm{wind}(u,0)$（卷绕数）给出——**第 14 篇的 Toeplitz 指标 $\operatorname{ind}(T_f)=-\mathrm{wind}(f)$ 正是 $K_1$ 的语言**。<span class="marginnote">K 理论把第 5 篇的 Fredholm 指标、第 14 篇的卷绕数统一收编：$\operatorname{ind}:K_1(\mathcal{Q})\to\mathbb{Z}$ 就是「指标同态」。指标不再是孤立的整数，而是 K 理论同态——「指标 = K 理论中的计算」这一观念，让 Atiyah–Singer 指标定理成为 K 理论的推论。</span>

**悬浮（suspension）**：$SA=C_0(\mathbb{R},A)$（在一点消失的 $A$ 值函数）。**Bott 周期**（定理）：

$$K_0(A) \cong K_1(SA), \qquad K_0(S^2A)\cong K_0(A),\quad K_1(S^2A)\cong K_1(A).$$

K 理论是 **2-周期的**：$K_0,K_1$ 两个群就够，悬浮两次回到自己。Bott 元素 $\beta\in K_0(C_0(\mathbb{R}^2))$ 是这一周期性的种子。

## 3 六项正合列：K 理论的引擎

**定理（短正合列的 K 理论）**：设 $0\to I\to A\to A/I\to0$ 是 C\* 代数的短正合列（$I$ 闭理想），则 K 理论给出**六项正合列**：

$$
\begin{CD}
K_0(I) @>>> K_0(A) @>>> K_0(A/I)\\
@AAA @. @VVV\\
K_1(A/I) @\lt \lt \lt  K_1(A) @\lt \lt \lt  K_1(I)
\end{CD}
$$

（对角线由 **指数映射** $\mathrm{ind}:K_1(A/I)\to K_0(I)$ 与**边界映射**连接。）<span class="marginnote">六项正合列是 K 理论的「牛顿定律」：给定一个理想扩张 $0\to I\to A\to A/I\to0$，三个 K 群互相锁定，边界映射记录「商里的酉元何时源于原代数的投影」的信息。第 14 篇 Toeplitz 的 $0\to\mathcal{K}\to\mathcal{T}\to C(\mathbb{T})\to0$ 给出六项列：$K_0(\mathcal{T})=\mathbb{Z}$，$K_1(\mathcal{T})=0$，指标同态从 $K_1(C(\mathbb{T}))=\mathbb{Z}$ 打到 $K_0(\mathcal{K})=\mathbb{Z}$。</span>

**应用（指标映射）**：Toeplitz 正合列里，$K_1(C(\mathbb{T}))\to K_0(\mathcal{K})$ 的指数映射正是 $f\mapsto\operatorname{ind}(T_f)$——**Fredholm 指标 = K 理论的边界映射**。这就是第 5 篇「指标 = 拓扑量」的代数证明。

## 4 公式解析：$K_0(A)$ 与六项正合列

$$
K_0(A)=\mathrm{Gr}\bigl(\mathrm{Proj}(M_\infty(A))/\!\sim_0\bigr), \qquad 
0\to I\to A\to A/I\to0 \ \Rightarrow\ \text{六项正合列}
$$

- **第一步，看 $K_0$ 的构成**：先把投影（模稳定等价）收集成半群 $V(A)$（加法 = 直和），再用 Grothendieck 技巧补负元得 $K_0(A)$。**$K_0$ 数的是「稳定意义下投影的个数（带符号）」**。
- **第二步，看矩阵扩张 $M_\infty$**：$p\in M_n(A)$ 允许把投影「放大」到任意阶矩阵。稳定等价 $p\sim_0q$ 允许「加零块」——$K_0$ 只看「本质投影」：加了一堆零块的投影算同一个类。这消除「投影嵌在不同阶矩阵」的尴尬。
- **第三步，看 $K_1$ 与指标**：$K_1$ 数「酉元的同伦类」。六项正合列的指数映射 $K_1(A/I)\to K_0(I)$ 把「商代数里一个酉元」提升为「理想里一个投影」——正合性保证「能提升的恰是核为 0 的」。对 Toeplitz，这就是 $\operatorname{ind}(T_f)=-\mathrm{wind}(f)$ 的群论形态。
- **第四步，看为什么它能分类**：$K_0,K_1$ 对同伦不变、对张量积有 Künneth 公式、对理想扩张有六项列——一个「做代数运算不丢信息」的函子。配合迹（第 10、22 篇），就构成 Elliott 不变量（第 17 篇预告），撑起第 26 篇的整个分类大厦。

## 5 K 理论的用武之地

**应用 1（分类理论）**：$K_0$（带正锥与单位元）完备分类 AF 代数（第 17 篇 Elliott）；$(K_0,K_1,\text{迹})$ 分类 $\mathcal{O}_n$ 的稳定同类（Kirchberg–Phillips）。**K 理论是分类的唯一通用不变量**。<span class="marginnote">Elliott 纲领的信条：一个单 C\* 代数的同构类，由 $K_0$（有序）、$K_1$、迹单形（及配对）完全决定。$A_\theta$ 的 $K_0=\mathbb{Z}+\theta\mathbb{Z}$、$\mathcal{O}_n$ 的 $K_0=\mathbb{Z}/(n-1)$、AF 的 $K_0=\lim\mathbb{Z}^k$——K 理论把「代数」翻译成「离散数论对象」，分类变成「数论对象分类」。</span>

**应用 2（指标理论）**：Atiyah–Singer 指标定理的现代证明即「用 K 理论定义解析指标 + 用拓扑 K 理论算它」；Toeplitz、椭圆算子、Dirac 算子的指标全是 K 理论的边界映射。**非交换几何（Connes）** 里，K 理论 + 循环上同调（Chern 特征）是「积分」的非交换替代。

**应用 3（动力系统与物理）**：交叉积的 Pimsner–Voiculescu 六项列（第 20 篇）算 $K_*(C(X)\rtimes G)$；量子霍尔效应的电导 = Chern 数 = $K_0$ 配对的整数（第 15 篇的回响）。K 理论成为「量子拓扑不变量」的工厂。

**辨析｜易错点：**$K_0(A)$ 是「**稳定**等价」而非「等价的」投影类——$p$ 与 $p\oplus 0_n$ 是同一个 $K_0$ 类。因此 $K_0$ 看不见「有限维投影的绝对秩」，只看见「模零块的稳定秩」。另一个易错点：$K_1$ 用的是**同伦**而非代数和——$u$ 与 $u\oplus 1$ 同伦，所以 $K_1$ 也是「稳定」的。理解 K 理论的「稳定性」，是正确解读它的前提。

## 6 例：K 理论计算精选

把 K 理论在几个关键例子上算出来，抽象定义立刻有「数值感」。

**$K_0(\mathbb{C})=\mathbb{Z}$**：投影 = 有限秩投影，等价类由秩决定。$[p]=\mathrm{rank}\,p$。

**$K_0(C(\mathbb{T}))=\mathbb{Z}$**：$\mathbb{T}$ 上的复向量丛只有直线丛（秩一）+ 平凡丛的稳定类——$K^0(\mathbb{T})=\mathbb{Z}$。$K_1(C(\mathbb{T}))=\mathbb{Z}$（卷绕数）。

**$K_0(C_0(\mathbb{R}^2))=\mathbb{Z}$**：Bott 元素 $\beta$ 生成。$C_0(\mathbb{R}^2)$「像」一个「单位球」——它的 $K_0$ 非平凡。

**$K_0(A_\theta)=\mathbb{Z}+\theta\mathbb{Z}$**（第 15 篇）：Rieffel 投影的迹值填满稠密子群。$A_\theta\cong A_{\theta'}$ 当且仅当「$\theta'=\pm\theta$ 模 $\mathbb{Z}$ 或取倒数」。

**$K_0(\mathcal{O}_n)=\mathbb{Z}/(n-1)\mathbb{Z}$，$K_1(\mathcal{O}_n)=0$**（第 18 篇）：纯无限单代数的 K 群可以很小。

**$K_0(\mathcal{T})=\mathbb{Z}$，$K_1(\mathcal{T})=0$**（Toeplitz）：六项正合列算出——指标映射 $K_1(C(\mathbb{T}))\to K_0(\mathcal{K})$ 是「满射」。

**一句话总结**：$K_0$ 数投影、$K_1$ 数酉元；从 $\mathbb{Z}$、$\mathbb{Z}/(n-1)$ 到 $\mathbb{Z}+\theta\mathbb{Z}$，K 理论把代数的「离散身份」一网打尽。

## 7 延伸：Bott 周期

Bott 周期是 K 理论最神秘也最深刻的性质。

**Bott 元素**：$\beta\in K_0(C_0(\mathbb{R}^2))$ 是生成元（$[\beta]=1$）。$C_0(\mathbb{R}^2)$ 是「二维悬浮」的化身——$\mathbb{R}^2$ 的一点紧化是 $S^2$，$K_0(S^2)$ 有非平凡元素。

**Bott 周期性**：$K_0(A)\cong K_0(A\otimes C_0(\mathbb{R}^2))$，$K_1(A)\cong K_1(A\otimes C_0(\mathbb{R}^2))$——张量上「二维圆盘」不改变 K 理论。所以 $K_0,K_1$ 两个群足够，2-周期。

**证明的思想**：用「同伦 + 张量积」把 $C_0(\mathbb{R}^2)$ 的 Bott 元素「吸收」进 K 群——本质是「$\mathbb{R}^2$ 的复结构给出一个规范投影」。拓扑 K 理论里，Bott 周期由「$S^2$ 的复向量丛」驱动。

**与 Fredholm 指标**：Bott 周期保证指标理论「2-周期」——椭圆算子的指标只依赖「符号的 K 类模 2」。

**意义**：K 理论是「2-周期」的周期性理论，而周期性与「能谱的稳定性」直接相关——拓扑学、分析、物理在这里共享同一个「2」。

**一句话总结**：Bott 周期 = 「$\mathbb{R}^2$ 不改变 K 理论」——它让 K 理论成为 2-周期的伟大理论。

## 8 延伸：K 理论与指标/分类

K 理论的用武之地，集中在指标理论与分类理论两处。

**指标 = K 理论同态**：Atiyah–Singer 指标定理的现代形态：解析指标 $\mathrm{ind}:K(T^*M)\to\mathbb{Z}$ 等于拓扑指标。Toeplitz（第 14 篇）、Fredholm（第 5 篇）、椭圆算子全被 K 理论收编。

**Elliott 不变量**：分类（第 26 篇）用 $(K_0,K_0^+,[1],K_1,T(A),\rho_A)$。K 理论是分类的「第一不变量」。

**K 理论 vs 迹**：$K_0$ 给「离散形状」，迹给「连续测度」，配对 $\rho_A$ 焊接两者。Toms 反例（第 26 篇）说明「只靠 $K$ 群 + 迹」在非 Z-稳定时会漏信息——K 理论是必要条件，配 Z-稳定性才是充分。

**非交换几何**：Connes 用 $K$ 理论 + 循环上同调（Chern 特征）定义「非交换积分」。$A_\theta$ 的 $K_0=\mathbb{Z}+\theta\mathbb{Z}$ 是量子霍尔电导的代数根源（第 15 篇）。

**一句话总结**：K 理论是「分类的骨架、指标的舞台、非交换几何的积分器」——C\* 代数理论最锋利的不变量。

## 9 小结

- **$K_0(A)$**：投影（模稳定等价）的 Grothendieck 群；$K_0(\mathbb{C})=\mathbb{Z}$，$K_0(C(X))=K^0(X)$，$K_0(A_\theta)=\mathbb{Z}+\theta\mathbb{Z}$。
- **$K_1(A)$**：酉元模同伦的群；$K_1(C(\mathbb{T}))=\mathbb{Z}$ 由卷绕数给出。
- **Bott 周期**：$K_0(S^2A)\cong K_0(A)$，$K_1(S^2A)\cong K_1(A)$——K 理论 2-周期。
- **六项正合列**：$0\to I\to A\to A/I\to0$ 给出 $K_0,K_1$