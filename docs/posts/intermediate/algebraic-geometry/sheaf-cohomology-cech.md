---
title: 层上同调与 Čech 上同调
date: 2026-08-07
---

# 层上同调与 Čech 上同调

<div class="epigraph">
<p>同调代数不是抽象的抽象，而是精确的度量——度量"正合性被破坏的程度"。</p>
<footer>—— 由亚历山大 · 格罗滕迪克（Alexander Grothendieck）思想转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数几何 ｜ Hartshorne, Algebraic Geometry (GTM 52) Ch. III §2-4 ｜ 2026-08-07</p>
</div>

## 为什么从层上同调继续

前面我们反复遇到一个"遗憾"：整体截面函子 $\Gamma(X, \cdot)$ 只左正合。短正合列 $0 \to \mathcal{F}' \to \mathcal{F} \to \mathcal{F}'' \to 0$ 取截面后变成 $0 \to \Gamma(X,\mathcal{F}') \to \Gamma(X,\mathcal{F}) \to \Gamma(X,\mathcal{F}'')$——最后的满射不一定能保住。例如 $\mathbb{P}^1$ 上 $0 \to \mathcal{O}(-2) \to \mathcal{O}(-1) \to \mathcal{O}(-1)/\mathcal{O}(-2) \to 0$ 取截面后满射失败，正是这个"缺口"记录了不变量。

**层上同调**就是把这个缺口系统化：定义 $H^i(X, \mathcal{F})$ 使"取截面的满射缺口"成为 $H^1$，而"长正合列"把整条链的信息拉直。它是连接"局部可解"与"全局可解"的桥梁——正如微分几何里"局部可解 ⟹ 全局可解"需要 Hodge 理论，代数几何里一切全局命题（Riemann-Roch、Serre 对偶、消没定理）都从上同调出发。本节同时给出**导出函子上同调**（抽象定义）与 **Čech 上同调**（可计算定义），并证明两者一致，最后算出射影空间 $H^i(\mathbb{P}^n, \mathcal{O}(d))$ 的全部答案。

## 1 缺口：为什么需要上同调

回顾第 8 篇的模层。整体截面函子 $\Gamma(X, \cdot): \mathcal{F} \mapsto \mathcal{F}(X)$ 是左正合但**不是**右正合。经典例：$X = \mathbb{P}^1_k$ 上的正合列

$$0 \longrightarrow \mathcal{O}(-2) \longrightarrow \mathcal{O}(-1) \longrightarrow k_P \longrightarrow 0$$

（$k_P$ 是点 $P$ 的 skyscraper 层）取整体截面：$H^0(\mathbb{P}^1, \mathcal{O}(-2)) = 0$、$H^0(\mathbb{P}^1, \mathcal{O}(-1)) = 0$、$H^0(\mathbb{P}^1, k_P) = k$，于是 $0 \to 0 \to 0 \to k$ 的满射没了。<span class="marginnote">为什么 $\Gamma$ 不右正合？因为"整体截面"要求定义在<strong>整个</strong> $X$ 上，而"局部上有定义"的数据不一定能胶合成整体数据（第 5 篇层公理的"胶合"条件）。上同调 $H^1$ 恰好度量"胶合失败"：$H^1(X, \mathcal{F}) = 0$ 大致表示"$\mathcal{F}$ 的任意局部数据都能拼成整体数据"。</span>

**重点：长正合列（long exact sequence）。** 对短正合列 $0 \to \mathcal{F}' \to \mathcal{F} \to \mathcal{F}'' \to 0$，存在**连接同态**与长正合列

$$0 \longrightarrow H^0(X, \mathcal{F}') \longrightarrow H^0(X, \mathcal{F}) \longrightarrow H^0(X, \mathcal{F}'') \longrightarrow H^1(X, \mathcal{F}') \longrightarrow H^1(X, \mathcal{F}) \longrightarrow \cdots$$

每一个"缺口"（$H^0$ 层的满射失败处）由下一个 $H^1$ 精确捕获。<span class="marginnote">长正合列是同调代数的第一条命脉：它把"局部信息是否足够"的每个缺口翻译成上层上同调的一个元素。例：$H^1(X, \mathcal{F}') = 0$ 时，$H^0$ 层的满射恢复——这就是"$\mathcal{F}'$ 无上同调 ⟹ 全局截面正合"的消没原理。</span>

## 2 导出函子上同调：抽象定义

**核心概念：内射层（injective sheaf）**：模层 $\mathcal{I}$ 称为内射的，如果 $\operatorname{Hom}(-, \mathcal{I})$ 是正合函子（"接收一切扩展"）。每个模层都嵌进一个内射层，从而可以取**内射消解**

$$0 \longrightarrow \mathcal{F} \longrightarrow \mathcal{I}^0 \longrightarrow \mathcal{I}^1 \longrightarrow \cdots$$

**核心概念：层上同调（sheaf cohomology）**：对概形 $X$ 与模层 $\mathcal{F}$，定义

$$H^i(X, \mathcal{F}) = R^i \Gamma(X, \mathcal{F}) = \frac{\ker\big(\Gamma(X, \mathcal{I}^i) \to \Gamma(X, \mathcal{I}^{i+1})\big)}{\operatorname{im}\big(\Gamma(X, \mathcal{I}^{i-1}) \to \Gamma(X, \mathcal{I}^i)\big)}$$

即"取整体截面后的第 $i$ 个导出函子"。$H^0(X, \mathcal{F}) = \mathcal{F}(X)$，且短正合列自动给出长正合列。<span class="marginnote">内射消解是"把层塞进最好的能接受一切扩展的层"的标准做法，与拓扑学里奇异上同调用"自由 Abel 群消解"、群上同调用"自由消解"是同一套同调代数机器。$H^i$ 不依赖消解的选择（消解之间拟同构）。</span>

这个定义抽象但难算。真正可计算的武器是 Čech 上同调。

## 3 Čech 上同调：可计算定义

**核心概念：Čech 上同调（Čech cohomology）**：取 $X$ 的开覆盖 $\mathfrak{U} = \{U_i\}$，定义 Čech 复形：$i$-上链是"在 $i+1$ 个开集交集上"的截面

$$C^i(\mathfrak{U}, \mathcal{F}) = \prod_{i_0 < \cdots < i_i} \mathcal{F}(U_{i_0} \cap \cdots \cap U_{i_i})$$

微分 $d^i$ 由"交替求差 + 限制"给出，则

$$\check{H}{}^i(\mathfrak{U}, \mathcal{F}) = H^i(C^\bullet(\mathfrak{U}, \mathcal{F}))$$

**重点：有限开覆盖 ⟹ Čech 与导出函子上同调一致。** 若 $\mathfrak{U}$ 是有限开覆盖且对任意 $i_0 < \cdots < i_p$ 有 $H^q(U_{i_0} \cap \cdots \cap U_{i_p}, \mathcal{F}) = 0$（$q \ge 1$，即交叠处无上同调），则

$$\check{H}{}^i(\mathfrak{U}, \mathcal{F}) \cong H^i(X, \mathcal{F})$$

对拟射影簇的有限覆盖（用仿射图覆盖，仿射图间交叠是仿射的，由下述仿射消没定理 $H^q = 0$）该条件自动满足。<span class="marginnote">Čech 上同调的价值：<strong>它把上同调化成了"矩阵计算"</strong>——查交叠处截面、作交替差、取商。$H^0$ = 满足"处处相容"的整体截面；$H^1$ = "局部相容但整体配不平"的障碍；$H^i$ 依此类推。这是本节最可用的计算工具。</span>

**重点：仿射概形上同调消没（定理，Serre）。** 设 $X = \operatorname{Spec} A$ 仿射，$\mathcal{F}$ 是拟凝聚层，则

$$H^i(X, \mathcal{F}) = 0 \quad (i \ge 1)$$

这一句把第 5、8 篇的"仿射 = 代数"推向极致：**仿射概形没有高阶上同调**——一切全局障碍都由代数直接解决。<span class="marginnote">几何直觉：仿射概形"没有洞"，所以"局部数据拼不起来"的障碍不存在。这与"$\mathbb{R}^n$ 的开凸集没有 $H^1$"是同构的。它使"把问题化到仿射开集"成为几乎所有上同调计算的第一步。</span>

## 4 射影空间上同调：全部答案

仿射消没给出万能开局的工具，配合长正合列与归纳，射影空间的全部上同调可以一步到位：

**重点：$\mathbb{P}^n_k$ 的上同调。** 对 $d \in \mathbb{Z}$：

$$
H^i(\mathbb{P}^n, \mathcal{O}(d)) =
\begin{cases}
k[x_0, \dots, x_n]_d & i = 0,\ d \ge 0 \\
0 & 0 < i < n \text{ 或 } (i = 0, d < 0) \\
k[x_0, \dots, x_n]_{-d-n-1}^* & i = n,\ d \le -n-1 \\
0 & i = n,\ d > -n-1
\end{cases}
$$

特别地：$H^0(\mathbb{P}^n, \mathcal{O}(d))$ = 次数 $d$ 的齐次多项式（$d \ge 0$）；$H^n(\mathbb{P}^n, \mathcal{O}(-n-1)) = k$（一维！）；其余全为零。<span class="marginnote">当 $n=1$ 时：$H^1(\mathbb{P}^1, \mathcal{O}(d)) = 0$ 对 $d \ge -1$，$H^1(\mathbb{P}^1, \mathcal{O}(-2)) = k$。这就是 Riemann-Roch 在 $\mathbb{P}^1$ 上的全部同调内容。这条表是第 11、12 篇计算 Serre 对偶与 Riemann-Roch 的算盘。</span>

**重点：有限性定理。** 设 $X$ 是 $k$ 上射影概形，$\mathcal{F}$ 凝聚，则每个 $H^i(X, \mathcal{F})$ 是**有限维 $k$-向量空间**且当 $i > \dim X$ 时 $H^i = 0$。<span class="marginnote">"射影 + 凝聚 ⟹ 有限维上同调"是本篇最深刻的整体结论之一：它保证了"截面数"是可以计数的整数——Riemann-Roch 的 $\ell(D) = \dim H^0$ 因此有意义。证明的关键是"真态射的推出保持凝聚"（第 7、8 篇）+ 归纳维数。</span>

## 5 公式解析：长正合列与消没

$$
0 \to H^0(\mathcal{F}') \to H^0(\mathcal{F}) \to H^0(\mathcal{F}'') \to H^1(\mathcal{F}') \to \cdots, \qquad H^i(\text{仿射}, \mathcal{F}) = 0\ (i \ge 1)
$$

分三步拆解：

- **第一步，长正合列从哪来**：导出函子的标准性质——短正合列 $0 \to \mathcal{F}' \to \mathcal{F} \to \mathcal{F}'' \to 0$ 先取内射消解得到"层的复形短正合列"，再取 $\Gamma$ 后得到"向量空间的复形短正合列"，对其做同调即得长正合列。连接同态 $\delta: H^0(\mathcal{F}'') \to H^1(\mathcal{F}')$ 把"满射缺口"抬升成"$H^1$ 元素"。<span class="marginnote">蛇引理（snake lemma）在层论里的化身：$\operatorname{coker}$ 的亏空被 $\ker$ 的下一个同调项接管，链条如此不断延伸。上同调把"逐层的亏空"串成一条无限链，这就是长正合列。</span>
- **第二步，为什么仿射消没**：$\operatorname{Spec} A$ 上的拟凝聚层 $\mathcal{F} = \widetilde{M}$。内射消解可以取成"由 $\widetilde{M}$ 生成的拟凝聚内射层"（利用 $A$-模内射消解），而 $\Gamma(\operatorname{Spec} A, \widetilde{M}) = M$ 对仿射情形**正合**——取截面不再丢失信息，故 $R^i \Gamma = 0$（$i \ge 1$）。
- **第三步，怎么用**：把任意概形覆盖成仿射开集 $\{U_i\}$，用 Čech 复形计算；交叠处（也是仿射的）无高阶上同调，Čech 上同调 = 导出函子上同调。**上同调计算 = 选仿射覆盖 + 算 Čech 复形**，这条流水线是整章的标准动作。

一句话直觉：**上同调是"正合性亏空的精密账本"**；仿射概形没有欠账（消没），射影空间的全本账目已被算出，其余一切由"覆盖 + Čech 复形 + 长正合列"逐层结算。

## 6 速查表：$\mathbb{P}^1$ 上的上同调

| 层 | $H^0$ | $H^1$ | 备注 |
| --- | --- | --- | --- |
| $\mathcal{O}(-1)$ | 0 | 0 | 两处都零 |
| $\mathcal{O}$ | $k$ | 0 | 常数函数 |
| $\mathcal{O}(1)$ | 2 维 | 0 | 线性式 $x_0, x_1$ |
| $\mathcal{O}(-2)$ | 0 | $k$ | 唯一的非平凡 $H^1$ |

**数值算例：用 Čech 复形算 $H^1(\mathbb{P}^1, \mathcal{O}(-2))$。** 取覆盖 $\mathfrak{U} = \{U_0, U_1\}$（$U_i = \{x_i \neq 0\}$）。$C^0 = \mathcal{O}(-2)(U_0) \oplus \mathcal{O}(-2)(U_1)$，$C^1 = \mathcal{O}(-2)(U_0 \cap U_1)$，微分 $d: (f_0, f_1) \mapsto f_0|_{U_0\cap U_1} - f_1|_{U_0\cap U_1}$。$U_0 \cap U_1 = \operatorname{Spec} k[t, t^{-1}]$ 上的截面形如 $t^{-2} \cdot k[t,t^{-1}]$（带 $1/x_0^2$ 的规范权），而 $U_i$ 上的截面分别只含"非负幂/非正幂"一侧——相减后商掉可补回的部分，剩下的正是 $H^1 \cong k$。<span class="marginnote">这就是"仿射覆盖 + Čech 复形"把抽象上同调变成具体计算的示范：不需要内射消解，只需要在两张仿射图与它们的交上做减法和取商。$\mathbb{P}^1$ 上 $\mathcal{O}(-2)$ 的 $H^1$ 之所以是 $k$，是因为 $1/(x_0 x_1)$ 这个截面局部良定、却无法由两侧的整体截面之差补回。</span>

一句话：**$\mathbb{P}^1$ 的 Čech 账本**——$H^0$ 数"处处相容的整体截面"，$H^1$ 数"局部相容但整体配不平"的障碍，$H^1(\mathbb{P}^1, \mathcal{O}(-2)) = k$ 是其中最纯粹的一笔。

## 7 小结

- **缺口**：$\Gamma$ 左正合不右正合；短正合列给出**长正合列**，每个缺口由下一个 $H^1$ 捕获。
- **导出函子上同调** $H^i = R^i\Gamma$：用内射消解定义，抽象、不变、自动长正合。
- **Čech 上同调**：用开覆盖的交叠截面作交替差计算，有限仿射覆盖下与导出函子上同调一致。
- **仿射消没**：$H^i(\operatorname{Spec} A, \mathcal{F}) = 0\ (i \ge 1)$——仿射无洞。
- **射影空间表**：$H^0(\mathbb{P}^n, \mathcal{O}(d))$ = 齐次多项式、$H^n(\mathbb{P}^n, \mathcal{O}(-n-1)) = k$、其余为零；射影 + 凝聚 ⟹ 有限维且高次为零。

在下一节，我们迎来第一个大定理：**Serre 对偶定理**——把上同调与对偶（线性泛函）挂钩，用"典范层"把 $H^i$ 与 $H^{n-i}$ 配对。
