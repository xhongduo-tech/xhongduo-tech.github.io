---
title: 谱定理
date: 2026-08-07
---

# 谱定理

<div class="epigraph">
<p>上帝用美丽的数学创造了世界。</p>
<footer>—— 保罗 · 狄拉克（Paul Dirac）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Davidson《C\*-Algebras by Example》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从谱定理开始

线性代数最深刻的定理是：**实对称矩阵可以正交对角化**。到了无穷维，这句话升级为谱定理——它是整个算子理论皇冠上的明珠。第 4 篇的连续函数演算已经能对正规算子计算 $f(T)$，但只对连续函数有效；谱定理把函数演算的刀刃磨到极致：**对所有有界 Borel 函数 $f$ 都能定义 $f(T)$**，从而把 $T$ 分解成「沿谱测度积分」的形态。

谱定理同时是量子力学的数学基石：可观测量的谱分解、投影测量的 Born 规则、以及 von Neumann 代数理论中「交换子代数结构」的证明，全部依赖它。这一节我们讲三种互相等价的谱定理表述，以及它们如何统一成一台功能强大的机器。

## 1 谱定理：三种同样深刻的说辞

**形态一（乘法算子）**：正规算子 $T$ 酉等价于某个 $L^2(\mu)$ 上的乘法算子 $M_f$，$M_f g=fg$。<span class="marginnote">这是「对角化」的字面推广：正规算子 =（换坐标系后）乘一个函数。有限维里「乘函数」就是「乘对角矩阵」，谱 $\sigma(T)$ 对应函数 $f$ 的值域，特征向量对应 $L^2$ 的点质量。</span>

**形态二（Borel 函数演算）**：存在唯一映射 $f\mapsto f(T)$，把**有界 Borel 函数** $f$ 送到算子，它延拓连续函数演算，且满足谱映射定理 $\sigma(f(T))=f(\sigma(T))$。

**形态三（谱测度）**：存在唯一的**投影值测度（projection-valued measure）** $E$，使

$$T = \int_{\sigma(T)} \lambda\, dE(\lambda).$$

三种说辞等价：乘法形态给出直观，Borel 演算给出操作，谱测度给出分解。三者由「交换 C\* 代数的表示」统一（下节看）。

## 2 投影值测度：把算子切成投影

**谱测度 / 投影值测度**：映射 $E:\mathrm{Borel}(\sigma(T))\to\operatorname{Proj}(\mathcal{H})$（每个 Borel 集对应一个正交投影），满足：

1. $E(\emptyset)=0$，$E(\sigma(T))=I$；
2. 可数可加性：互不相交的 $S_n$ 有 $E(\cup S_n)=\sum_n E(S_n)$（强算子收敛）；
3. 乘性：$E(S\cap T)=E(S)E(T)$（特别地，$E(S),E(T)$ 交换）。

**直觉**：$E(S)$ 是「把 $\mathcal{H}$ 切到 $T$ 的谱落在 $S$ 内的那一块」的投影。<span class="marginnote">投影值测度把「$\sigma(T)$ 这个频率轴」变成「一系列互相正交的闭子空间」。$E(\{\lambda\})$ 非零时，$\lambda$ 就是特征值，$E(\{\lambda\})\mathcal{H}$ 就是特征空间；连续谱对应「没有单点质量、却有连续分布」的投影测度。</span>

**谱积分**：$\int f\,dE$ 定义为「逐步函数 → 极限」：若 $s=\sum\alpha_i\chi_{S_i}$ 是简单函数，则 $\int s\,dE=\sum\alpha_iE(S_i)$；对一般有界 Borel $f$ 用一致逼近取极限。积分满足 $\|\int f\,dE\|\le\|f\|_\infty$，且 $(\int f\,dE)^*=\int\overline f\,dE$。

## 3 Borel 函数演算：刀刃的极限

谱定理给出的 Borel 函数演算 $f\mapsto f(T)=\int f\,dE$ 拥有全部理想性质：

**定理（Borel 演算）**：对正规算子 $T$，$f\mapsto f(T)$ 是 $B_\infty(\sigma(T))\to B(\mathcal{H})$ 的 $\ast$-同态（把逐点收敛的有界序列映为强收敛的算子序列），且：

**谱映射**：$\sigma(f(T))=f(\sigma(T))$（$f$ 连续时；对一般 Borel $f$，$\sigma(f(T))\subset\overline{f(\sigma(T))}$）；
**交换子不变**：任何与 $T$ 交换的算子也与 $f(T)$ 交换——**函数演算不增大交换子**。<span class="marginnote">最后一条是通往 von Neumann 代数的桥：它保证由 $T$ 生成的弱闭代数与连续函数演算生成的代数有相同交换子。第 21 篇双交换子定理的核心，就是「谱测度的像张成 $T$ 的 von Neumann 代数」。</span>

**例子（自伴算子）**：$T$ 自伴时，$E$ 集中在 $\mathbb{R}$ 上，$T=\int_{\mathbb{R}}\lambda\,dE(\lambda)$ 是**实值积分**——这就是「自伴 = 实谱」的最强形态。

**例子（酉算子）**：$U$ 酉时，$E$ 集中在单位圆 $\mathbb{T}$ 上，$U=\int_{\mathbb{T}}z\,dE(z)$。谱定理把「酉算子的旋转」分解成「沿圆各点的投影旋转」。

## 4 公式解析：$T=\int\lambda\,dE(\lambda)$

$$
T = \int_{\sigma(T)} \lambda\, dE(\lambda)
$$

- **第一步，看右端**：$\lambda$ 是标量，$dE(\lambda)$ 是投影值测度。积分把「每个谱点分配一个投影」组合起来。对有限维正规矩阵，$E(\{\lambda\})=$ 特征空间投影，积分退化为 $\sum\lambda P_\lambda$——与对角化完全一致。
- **第二步，看为什么不只是连续演算**：连续演算只能处理 $f\in C(\sigma(T))$，特征函数 $\chi_S$（$S$ 是 Borel 集）一般不在 $C(\sigma(T))$ 里。Borel 演算把 $f$ 推广到可测函数，于是 $\chi_S(T)=E(S)$——**投影直接由特征函数得到**。这为「把 Hilbert 空间切成谱块」提供了全部工具。
- **第三步，看内积形态**：对 $x,y\in\mathcal{H}$，$\langle Tx,y\rangle=\int\lambda\,d\mu_{x,y}(\lambda)$，其中 $\mu_{x,y}(S)=\langle E(S)x,y\rangle$ 是复值测度。取 $x=y=\xi$ 且 $\|\xi\|=1$，$\mu_\xi(S)=\langle E(S)\xi,\xi\rangle$ 是**概率测度**——量子力学里「在集合 $S$ 内测得结果的概率」正是它，Born 规则在此显形。
- **第四步，看统一性**：三条说辞在同一条链上：Gelfand 变换把 $C^*(T,1)$ 变成 $C(\sigma(T))$，Riesz–Markov 定理把 $C(\sigma(T))$ 上的正泛函变成测度，测度再张成投影值测度。**谱定理 = Gelfand 变换 + Riesz–Markov**，是交换理论在单个元素上的完整爆发。

## 5 谱定理的武力展示

**平方根与极分解**：$T\ge0$ 时 $T^{1/2}=\int\lambda^{1/2}dE(\lambda)$ 唯一；一般算子有极分解 $T=U|T|$，其中 $U$ 是部分等距。这些在连续演算里已见，Borel 演算让它们更完整（$f(t)=|t|$ 在 $C(\sigma(T))$ 里已经够用）。

**谱子空间（spectral subspaces）**：$E([a,b])\mathcal{H}$ 是「$T$ 的谱在 $[a,b]$ 内的部分」。这使算子可以「按频率滤波」——量子测量、信号处理、以及第 23 篇因子理论里维数函数的构造都靠它。<span class="marginnote">谱子空间把「投影」与「区间」对应起来，让微积分的直觉（$[a,b]$ 上的积分）直接作用于算子。第 23 篇 Type I 因子的维数函数 $\dim(E([0,t])\mathcal{H})$ 正是这种滤波的连续版。</span>

**交换子的视角（von Neumann 预告）**：设 $T$ 自伴，$\mathcal{M}=W^*(T)$ 是 $T$ 生成的 von Neumann 代数。谱定理 + 双交换子定理（第 21 篇）给出：$\mathcal{M}$ 恰好是 $\{f(T):f\in B_\infty(\sigma(T))\}$ 的强闭包——**von Neumann 代数的全部元素都是 $T$ 的 Borel 函数**。这是「对角化」在 von Neumann 层面的终极形态，也是 Type I 因子分类的出发点。

**辨析｜易错点：**连续演算与 Borel 演算的**谱映射有区别**：对连续 $f$，$\sigma(f(T))=f(\sigma(T))$ 精确成立；对一般 Borel $f$（如特征函数），只能保证 $\sigma(f(T))\subset\overline{f(\sigma(T))}$。特征函数 $\chi_S$ 的像 $E(S)$ 的谱是 $\{0,1\}$，与 $\{0,1\}$ 吻合，但若 $f$ 取「跨谱跳跃」的值，像的谱可能比 $f(\sigma(T))$ 小。**连续性不是免费的**——这正是为何谱测度的支集（support）是闭集，而一般 Borel 函数的信息会「收缩」。

## 6 例：谱定理在不同算子上的展开

同一个谱定理，在不同算子身上长成不同但统一的形态。

**自伴对角算子**：$T e_n=\lambda_n e_n$（$\lambda_n\in\mathbb{R}$）。$E(S)=\text{「$\lambda_n\in S$ 对应的特征空间投影」}$，$T=\int\lambda\,dE(\lambda)=\sum\lambda_nP_n$。特征值 = 谱测度的原子。

**乘法算子 $M_t$ 于 $L^2[0,1]$**：$E([a,b])=\chi_{[a,b]}(t)$ 的乘法，$M_t=\int_{[0,1]}t\,dE(t)$。连续谱没有原子，谱测度「连续分布」——投影 $E([a,b])$ 是「把函数截断在 $[a,b]$ 上」。

**酉算子 $U$**：谱在单位圆上，$E(S)$ 集中 $S\subset\mathbb{T}$，$U=\int_{\mathbb{T}}z\,dE(z)$。酉算子的「谱分解」= 沿圆的旋转分解——量子力学时间演化的数学。

**紧自伴算子**：$E(\{\lambda_n\})=P_n$（特征投影），$T=\sum\lambda_nP_n$（纯原子谱测度）。谱定理在紧算子上的形态是「可数原子」——谱测度退化为求和。

**有限维矩阵**：$E(\{\lambda\})=$ 特征空间投影，$T=\sum\lambda P_\lambda$。谱定理的最古老祖先——线性代数的谱分解定理。

**一句话总结**：谱定理 = 「谱测度把谱点变成投影」——原子谱对应特征值，连续谱对应连续投影分布，两者统一于 $T=\int\lambda\,dE(\lambda)$。

## 7 延伸：谱测度与物理测量

谱测度不只是数学对象，它是量子测量理论的核心。

**Born 规则**：对态 $\xi$（$\|\xi\|=1$）与可观测量 $T$，测量 $T$ 得到「值落在 $S$」的概率是 $\langle E(S)\xi,\xi\rangle$。谱测度把「可测量的概率」装进数学。

**投影测量**：$E(S)$ 是「问『$T$ 的值是否在 $S$ 里』」的投影。测量后系统「塌缩」到 $E(S)\mathcal{H}$——量子测量公设的代数版本。

**期望值**：$\langle T\rangle_\xi=\int\lambda\,d\langle E(\lambda)\xi,\xi\rangle=\int\lambda\,d\mu_\xi(\lambda)$。谱定理把「期望值」化成「关于谱测度的积分」。

**方差与不确定性**：$(\Delta T)^2=\int(\lambda-\langle T\rangle)^2d\mu_\xi(\lambda)$。位置与动量的谱测度不对易，给出 Heisenberg 不等式——谱测度是它的几何舞台。

**POVM 的推广**：一般测量用正算子值测度（POVM）$F(S)\ge0$ 描述，是谱测度的推广。谱测度（投影值）是「无噪声测量」的理想化，POVM 是现实测量的标准。

## 8 延伸：谱定理的证明脉络

谱定理是「许多路线通向同一真理」的典范——三条主要证明路径。

**路线一（函数演算 + 扩张）**：先在 $C(\sigma(T))$ 上定义 $f(T)$（第 4 篇），再用 Riesz–Markov 把「$f\mapsto\langle f(T)x,y\rangle$」变成复测度 $\mu_{x,y}$，最后张成投影值测度。这是最「分析」的路线。

**路线二（Gelfand 变换）**：$C^*(T,1)\cong C(\sigma(T))$（第 8 篇），特征空间 = 谱，Gelfand 变换把 $T$ 变成「谱上的恒等函数」，再取表示。这是最「代数」的路线。

**路线三（乘法算子实现）**：直接构造乘法算子表示：取 $L^2(\sigma(T),\mu)$ 使 $T$ 酉等价于乘 $\lambda$。这是「对角化」字面意义的实现。

**三条路线的统一**：都靠「谱 = 值域」这条字典。谱定理是 Gelfand 变换在单个正规算子上的「全套爆发」，也是整座 C\* 代数理论对「对角化」问题的总回答。

**后续**：谱定理 → 交换 von Neumann 代数 $\cong L^\infty$（第 21 篇）→ 约化理论（第 24 篇）。一条线从单算子通到整个 von Neumann 世界。

**一句话总结**：谱定理证明的每一条路，都是「把算子还原成乘法」的一次尝试——它们殊途同归，因为「正规算子本质是乘法」这条真理只有一种。

## 9 小结

- **谱定理三形态**：乘法算子 / Borel 函数演算 / 谱测度积分，三者等价，统一于 Gelfand + Riesz–Markov。
- **投影值测度** $E$：Borel 集 → 正交投影，可数可加、乘性；$E(S)\mathcal{H}$ 是谱落在 $S$ 的谱子空间。
- **Borel 演算**：$f\mapsto f(T)=\int f\,dE$ 延拓连续演算，保谱映射（连续时精确）且不增大交换子。
- **$T=\int\lambda\,dE(\lambda)$**：自伴是实积分，酉是圆积分，有限维退化为 $\sum\lambda P_\lambda$。
- **量子力学**：$\langle E(S)\xi,\xi\rangle$ 是投影测量概率，Born 规则在此显形。
- **通往 von Neumann**：$W^*(T)=\{f(T)\}$