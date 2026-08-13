---
title: Gelfand-Naimark 定理
date: 2026-08-07
---

# Gelfand-Naimark 定理

<div class="epigraph">
<p>数学家是在定理之间寻找类比的人；更好的数学家是在证明之间看到类比的人；最好的数学家是能在理论之间察觉类比的人。</p>
<footer>—— 斯特凡 · 巴拿赫（Stefan Banach）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Davidson《C\*-Algebras by Example》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Gelfand-Naimark 定理开始

第 7 篇我们把算子「抽象化」成 C\* 代数，第 8 篇给交换情形做完了「再具体化」：交换 C\* 代数 ≅ $C(X)$。那么**非交换**的情形呢？抽象 C\* 代数还能不能被「认回去」成具体算子？

**Gelfand–Naimark 定理**给出石破天惊的回答：**能**。每个 C\* 代数都 $\ast$-同构于某个 $B(\mathcal{H})$ 的闭 $\ast$-子代数。这意味着「抽象 C\* 代数」与「具体算子代数」是两个完全等价的世界——抽象不是失去具体性，而是把具体性推到更普适的高度。<span class="marginnote">这条定理（连同它的证明工具 GNS 构造）把 C\* 代数理论从「Banach 代数的一个漂亮分支」升级为「算子理论本身」。它让后来的一切分类、表示、von Neumann 代数理论都有了统一舞台。</span>

## 1 表示：把代数放到算子世界

**表示（representation）**：C\* 代数 $A$ 到 $B(\mathcal{H})$ 的 $\ast$-同态 $\pi:A\to B(\mathcal{H})$（保对合、保乘法、线性）。如果 $\ker\pi=\{0\}$，称 $\pi$ 是**忠实的（faithful）**；如果 $\pi$ 保单位（$\pi(1)=I$），称**非退化**或**单位表示**。

**辨析｜易错点：**$\pi$ 只是 $\ast$-同态，**不自动保范**。C\* 代数里，$\ast$-同态自动满足 $\|\pi(a)\|\le\|a\|$（范数压缩），但要 $\|\pi(a)\|=\|a\|$ 必须 $\pi$ 忠实。Gelfand–Naimark 定理的全部功夫，就是造出一个**忠实**表示，使压缩变成等距。

**例（交换情形的表示）**：对 $A=C(X)$，求值映射 $\pi_x(f)=f(x)$ 不是表示（值域是 $\mathbb{C}$，太小）；但**对角表示** $\pi(f)=\mathrm{diag}(f(x_n))$ 把函数送到 $\bigoplus_n\mathbb{C}\cong\ell^2$ 上的乘法算子——这正是 Gelfand 变换的「表示形态」。

## 2 态：造表示所需的原料

证明的核心工具是**态（state）**——先于下一节的完整理论，这里给出最小定义：**态**是 $A$ 上满足 $\varphi(a^*a)\ge0$、$\varphi(1)=1$ 的正线性泛函。

关键思想（**GNS 构造**，第 11 篇完整展开）：给定态 $\varphi$，在 $A$ 上定义半内积 $\langle a,b\rangle_\varphi=\varphi(b^*a)$，商掉零空间再完备化，得到 Hilbert 空间 $\mathcal{H}_\varphi$；$A$ 通过左乘法作用上去，得到表示 $\pi_\varphi:A\to B(\mathcal{H}_\varphi)$，且有循环向量 $\xi_\varphi$ 使 $\varphi(a)=\langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle$。<span class="marginnote">GNS 构造是「从态到表示」的算法：每个态造出一个带循环向量的表示，而表示又反过来产生态（$\langle\pi(\cdot)\xi,\xi\rangle$）。这种「态 ⟷ 表示」的双向通道，是 C\* 代数理论最漂亮的循环论证之一。</span>

**存在性**：$A$ 上态一定存在（对含幺代数用 Hahn–Banach 延拓正泛函，或对 $\lambda\mapsto\lambda$ 这个单位元上的泛函延拓），且对每个 $a\neq0$，存在态 $\varphi$ 使 $\varphi(a^*a)>0$——也就是说**态把每个非零正元素都「看见」**。

## 3 定理：忠实表示一定存在

**定理（Gelfand–Naimark）**：设 $A$ 是 C\* 代数。则存在 Hilbert 空间 $\mathcal{H}$ 与忠实表示 $\pi:A\to B(\mathcal{H})$，且 $\pi$ 可取的等距的（$\|\pi(a)\|=\|a\|$）。

构造（通用表示 universal representation）：

取 $S(A)$ 为 $A$ 上全体态的集合，令

$$\mathcal{H}_u=\bigoplus_{\varphi\in S(A)}\mathcal{H}_\varphi, \qquad \pi_u=\bigoplus_{\varphi\in S(A)}\pi_\varphi.$$

$A$ 逐坐标作用到直和上。**忠实性**来自态的分离性：若 $\pi_u(a)=0$，则对所有 $\varphi$ 与 $b$，$\pi_\varphi(b)\pi_\varphi(a)\xi_\varphi=0$，特别取 $b$ 与 $\xi_\varphi$ 整理得 $\varphi(a^*a)=0$ 对所有 $\varphi$，由「态分离正元素」得 $a^*a=0$，故 $a=0$。<span class="marginnote">通用表示 $\pi_u$ 是「最大的」表示：任何循环表示都「嵌」在其中，任何忠实表示都能由它压缩得到。把 $A$ 塞进 $\mathcal{H}_u$，就是给抽象代数装上了一个最宽敞的舞台，让所有表示同时上演。</span>

**推论（C\* 代数 = 算子代数）**：每个 C\* 代数都等距 $\ast$-同构于某个 $B(\mathcal{H})$ 的闭 $\ast$-子代数。从此「抽象」与「具体」不再有边界：研究 C\* 代数，就是研究 Hilbert 空间上的闭算子代数。

## 4 公式解析：GNS 内积公式

$$
\langle a, b\rangle_\varphi = \varphi(b^*a), \qquad \varphi(a) = \langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle
$$

- **第一步，看半内积**：$\langle a,b\rangle_\varphi=\varphi(b^*a)$ 把「代数元素」配成内积。正性（$\langle a,a\rangle\ge0$）来自态的正性：$\varphi(a^*a)\ge0$。若 $\langle a,a\rangle=0$ 而 $a\neq0$，就违反了「态分离正元素」，所以在商空间上内积是正定的。
- **第二步，看左乘变成算子**：$a$ 的左乘 $L_a:b\mapsto ab$ 穿过内积 $\langle ab,c\rangle=\varphi(c^*ab)=\langle b,a^*c\rangle$，正好满足 $L_a^*=L_{a^*}$——左乘法在 GNS 内积下自动「伴随即对合」，这就是 $\pi_\varphi(a)=L_a$ 成立的根源。
- **第三步，看循环向量**：$\xi_\varphi=1$（单位元所在的等价类），满足 $\varphi(a)=\langle a,1\rangle_\varphi=\langle\pi_\varphi(a)1,1\rangle_\varphi$。一个态被一个表示和一个向量精确复制——**态就是「向量期望值」**。
- **第四步，看为什么这保证忠实**：对任意 $a\neq0$，取态 $\varphi$ 使 $\varphi(a^*a)>0$，则 $\|\pi_u(a)\xi_\varphi\|^2=\varphi(a^*a)>0$，故 $\pi_u(a)\neq0$。每个非零元都被某个「坐标」看见，直和表示自然忠实。

## 5 定理的意义：理论从此一体

Gelfand–Naimark 定理带来的不只是保证存在，而是一整套世界观：

**表示理论成为中心**：要理解 $A$，就研究它的所有表示。单点表示不够就看所有表示；第 11 篇 GNS 构造将把「态 ↔ 表示」系统化，第 10 篇先给态的完整理论。

**谱与范数全部保真**：因为存在等距表示，抽象谱、抽象范数就是具体算子谱、具体算子范数。第 7 篇的谱稳定性、第 4 篇的函数演算，在表示下都不变形。<span class="marginnote">「存在忠实表示」意味着：任何在算子世界里成立的定理，若只用代数的语言书写，就自动对一切 C\* 代数成立。反之亦然。抽象与具体，从此互为镜像。</span>

**分离情形（separating）**：若 $A$ 可分（作为 Banach 空间），上面的直和可以取成可数直和，$\mathcal{H}_u$ 也可分。工程与物理中遇到的 C\* 代数几乎都可分，这让「可数多表示就足够」成为日常默认。

**辨析｜易错点：**Gelfand–Naimark 保证的是「存在某个 Hilbert 空间上的忠实表示」，但表示空间**不唯一**，表示也**不唯一**。同一个抽象 $A$ 可以有千千万万个表示，它们共享谱与范数，却在「循环性、可分性、生成元行为」上各不相同。研究「表示等价类」正是第 10–11 篇的主题，也是 von Neumann 代数理论（第 21 篇）与分类理论（第 26 篇）的前线。

## 6 例：表示论的第一次实战

把表示与 GNS 放到具体例子里，看它们如何「工作」。

**有限维表示**：$A=M_n$，表示 $\pi:M_n\to M_m$ 都是「直和」：$\pi(X)=\mathrm{diag}(X,\dots,X)$ 的压缩。不可约表示只有一个：恒等表示 $M_n\to M_n$。表示论在有限维退化为「计数直和」，仍然有效。

**求值表示**：$A=C(X)$，每个 $x\in X$ 给出「一维表示」$\mathrm{ev}_x(f)=f(x)$。交换 C\* 代数的不可约表示 = 特征 = 点——Gelfand 变换（第 8 篇）的表示论形态。

**移位表示的 Toeplitz 预览**：$A=C(\mathbb{T})$ 的表示 $\pi(f)=T_f$（Toeplitz 算子，第 14 篇）不是一维的——它把函数「抬」到 $H^2$ 上。表示论让交换代数的「非交换实现」成为可能。

**GNS 的有限维例子**：$A=M_2$，态 $\varphi(X)=\mathrm{Tr}(\rho X)$（$\rho$ 密度矩阵）。GNS 表示就是「$M_2$ 按 $\rho$ 的支撑做压缩」，循环向量对应 $\rho^{1/2}$。

**为什么「表示」是理解代数的窗口**：同一抽象代数在不同表示里「长不同面孔」；表示论的任务就是给这些面孔分类。GNS 说：态决定面孔，面孔（循环表示）反过来也决定态。

## 7 延伸：忠实表示与表示的分类

「存在忠实表示」只是起点——表示论真正的问题是「有多少表示、怎么分类」。

**表示等价**：$\pi_1,\pi_2$ 酉等价（存在酉 $U$ 使 $U\pi_1=\pi_2U$）视为同一。分类 = 描述酉等价类。

**不可约表示 = 原子**：每个表示都可以「分解」成不可约表示的（直接积分，第 24 篇）。研究「原子」（不可约表示）是表示论的首要任务。

**纯态 ⟺ 不可约**（第 11 篇将系统讲）：$GNS(\varphi)$ 不可约 ⟺ $\varphi$ 纯态。于是「不可约表示的分类」=「纯态的分类」——分析对象与代数对象再次等同。

**双连续（faithful）表示**：忠实表示「不丢信息」。可分 C\* 代数有忠实态，从而单个 GNS 表示就忠实——「一个表示演全场」在可分情形成立。

**原理想的角色**：$\ker\pi$ 是原始理想（第 12 篇）；表示的分类 ↔ 原始理想的分类。$\mathrm{Prim}(A)$（带 Jacobson 拓扑）是表示论的「点空间」。

## 8 延伸：GNS 构造的物理意义

GNS 构造不只是数学机器，它是量子理论的语言。

**真空态与真空表示**：量子场论里，真空态 $\omega_0$ 的 GNS 表示 $(\pi_0,\mathcal{H}_0,\xi_0)$ 给出「物理 Hilbert 空间」。不同真空（不等价表示）给出不同理论——表示等价性 = 物理等价性。

**可观测量的代数**：物理可观测量构成 C\* 代数 $\mathcal{A}$；态 $\omega$ 给出期望值。GNS 说：每个态都有一个「实现它的 Hilbert 空间」——测量理论从此有了数学舞台。

**超选择规则**：$\pi_\omega(\mathcal{A})''$ 的交换子非平凡时，Hilbert 空间分成「扇区」，不同扇区之间没有可观察的相对相位。第 24 篇约化理论把这些扇区分解成因子。

**时间演化**：哈密顿量 $H$ 作用在 $\mathcal{H}_\omega$ 上，$e^{itH}$ 是酉演化。GNS 把「态」与「演化」都装进同一个 Hilbert 空间——量子动力学的代数框架。

**一句话总结**：GNS 构造是「态 → 空间 → 表示」的流水线：物理上它把「测量期望值」升格为「整个 Hilbert 空间的物理」。

## 9 小结

- **表示**：$A\to B(\mathcal{H})$ 的 $\ast$-同态；忠实表示是单射，$\ast$-同态自动压缩范数，忠实才等距。
- **态**：正范数泛函（$\varphi(a^*a)\ge0,\varphi(1)=1$），是造表示的原料；态分离正元素。
- **GNS 构造**：每个态 $\varphi$ 造出 Hilbert 空间 $\mathcal{H}_\varphi$、表示 $\pi_\varphi$、循环向量 $\xi_\varphi$，且 $\varphi(a)=\langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle$。
- **Gelfand–Naimark 定理**：通用表示 $\pi_u=\bigoplus_\varphi\pi_\varphi$ 是忠实的，每个 C\* 代数等距同构于某个 $B(\mathcal{H})$ 的闭 $\ast$