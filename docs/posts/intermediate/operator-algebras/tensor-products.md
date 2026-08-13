---
title: 张量积
date: 2026-08-07
---

# 张量积

<div class="epigraph">
<p>理论研究的主要目标之一，是找到那个让主题显得最为简明的观察角度。</p>
<footer>—— 乔赛亚 · 威拉德 · 吉布斯（Josiah Willard Gibbs）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Davidson《C\*-Algebras by Example》第13章 ｜ 2026-08-07</p>
</div>

## 为什么从张量积开始

吉布斯这句话正好描述张量积的使命：**把多线性问题变成线性问题**。一个双线性映射 $B(x,y)$ 又依赖两个变量、又依赖非线性，难缠；但若先造一个「张量空间」$X\otimes Y$，把 $(x,y)$ 打包成基本张量 $x\otimes y$，$B$ 就变成一个线性映射 $\widetilde B:X\otimes Y\to Z$。复杂度没有消失，但被**收编进了空间结构**——这是整个数学最成功的「封装」之一。

在算子理论里，张量积承担着双重任务：Hilbert 空间的张量积是**量子力学复合系统**的语言（两个粒子的态空间是各自态空间的张量积）；C\* 代数的张量积则是**构造新代数**的工厂（$C(X)\otimes C(Y)=C(X\times Y)$、Toeplitz 代数的推广、以及整类可核代数的定义）。这一节我们从「线性代数的张量积」一路走到「C\* 代数的张量积及其范数之谜」。

## 1 Hilbert 空间的张量积

**代数张量积**：$\mathcal{H}\odot\mathcal{K}=\mathrm{span}\{x\otimes y:x\in\mathcal{H},y\in\mathcal{K}\}$，配双线性关系 $(x_1+x_2)\otimes y=x_1\otimes y+x_2\otimes y$ 等。**内积**由 $\langle x_1\otimes y_1,x_2\otimes y_2\rangle=\langle x_1,x_2\rangle\langle y_1,y_2\rangle$ 线性扩展定义。

**完备化**：$\mathcal{H}\otimes\mathcal{K}=\overline{\mathcal{H}\odot\mathcal{K}}$ 是 Hilbert 空间。若 $\{e_i\}$、$\{f_j\}$ 是标准正交基，则 $\{e_i\otimes f_j\}$ 是 $\mathcal{H}\otimes\mathcal{K}$ 的标准正交基。<span class="marginnote">基的直积给出维数相乘：$\dim(\mathcal{H}\otimes\mathcal{K})=\dim\mathcal{H}\cdot\dim\mathcal{K}$。量子力学里「两个自旋 1/2 粒子」：$\mathbb{C}^2\otimes\mathbb{C}^2\cong\mathbb{C}^4$，其中纠缠态正是「不能写成 $x\otimes y$ 形式」的张量——张量积空间比「乘积态」大得多，多出来的部分就是纠缠。</span>

**两个招牌同构**：
$L^2(X\times Y)\cong L^2(X)\otimes L^2(Y)$（乘积测度的分解）；
$\mathcal{H}\otimes\mathcal{K}\cong$ **Hilbert–Schmidt 算子** $L^2(\mathcal{H},\mathcal{K})$（第 6 篇），$x\otimes y\mapsto$ 秩一算子 $z\mapsto\langle z,\overline y\rangle x$。

**例（复合系统）**：两粒子态空间 = 单粒子态空间的张量积。纠缠（entanglement）是 $\mathcal{H}\otimes\mathcal{K}$ 中不可分解的向量——**张量积空间把「复合」从「并排」升级为「耦合」**，这是量子信息理论的几何基础。

## 2 C\*-代数的张量积：范数的麻烦

对 C\* 代数 $A,B$，代数张量积 $A\odot B$（有限和 $\sum a_i\otimes b_i$）是 $\ast$-代数（乘法 $(a\otimes b)(a'\otimes b')=aa'\otimes bb'$，对合 $(a\otimes b)^*=a^*\otimes b^*$），但要成为 C\* 代数，需要**一个满足 C\* 恒等式的范数**。问题来了：

**命题（范数不唯一）**：$A\odot B$ 上一般有**不止一个** C\* 范数。<span class="marginnote">这是张量积理论的核心难题：不同于 Hilbert 空间（内积唯一决定范数），C\* 代数的张量积范数依赖「用哪个表示去张」。$B(\mathcal{H})\otimes B(\mathcal{H})$ 上就同时存在最小范数与最大范数——数学里罕见的「范数歧义」，直接催生了可核性（nuclearity）概念。</span>

**最小（空间）张量积** $A\otimes_{\min}B$：取 $A,B$ 的忠实表示 $\pi_A,\pi_B$，在 Hilbert 张量积 $\mathcal{H}_A\otimes\mathcal{H}_B$ 上令 $\pi_A(a)\otimes\pi_B(b)$ 作用，用算子范数完备化。它不依赖表示的选择（因 C\* 恒等式强制规范），是「最小的」C\* 张量积范数。

**最大张量积** $A\otimes_{\max}B$：取所有「兼容表示」的范数的上确界，是「最大的」C\* 范数。最小与最大一般不相等。

**例（交换情形）**：$C(X)\otimes C(Y)\cong C(X\times Y)$，且最小 = 最大 = 唯一。交换 C\* 代数的张量积自动「无歧义」——**交换世界没有张量积的烦恼**。

## 3 可核性：张量积的「唯一范数」奖

**可核 C\* 代数（nuclear C\* algebra）**：对所有 C\* 代数 $B$，最小范数 = 最大范数（$A\otimes_{\min}B=A\otimes_{\max}B$ 对一切 $B$）。

**定理（可核代数的王国）**：以下代数都可核：
$C(X)$、$C_0(X)$、有限维代数 $M_n$、**紧算子** $\mathcal{K}(\mathcal{H})$；
**AF 代数**（第 17 篇）、**Cuntz 代数** $\mathcal{O}_n$（第 18 篇）、Toeplitz 代数 $\mathcal{T}$；
- **无理旋转代数** $A_\theta$ 与所有**可均群的群 C\* 代数** $C^*(G)$。

而 $B(\mathcal{H})$（$\mathcal{H}$ 无穷维）**不可核**。<span class="marginnote">可核性是 C\* 代数理论最重要的「良性质」之一：它等价于「近似有限维性质」的种种版本（完全正逼近性质 CPAP、Amenability in operator algebra sense）。可核代数拥有稳定良好的张量积、张量积 K 理论（Künneth 公式）也最干净。AF、$\mathcal{O}_n$、$A_\theta$ 全都可核，而 $B(\mathcal{H})$ 不是——<strong>可核性把「好代数」与「巨型代数」精确划开</strong>。</span>

**辨析｜易错点：**「可核」与「近似有限维」**不是**同一概念，但密切相关：每个可分的可核 C\* 代数都具有**完全正逼近性质（CPAP）**（由有限秩完全正映射逼近恒等）。把「可核」误当成「AF」是常见错误——$\mathcal{O}_n$ 可核却纯无限、绝非 AF。可核是「逼近性质」层面的事，与「有限维逼近的程度」是两把尺子。

## 4 公式解析：$\|a\otimes b\|_{\min}=\|a\|\,\|b\|$

$$
A\otimes_{\min}B = \overline{\pi_A(A)\otimes\pi_B(B)}^{\,B(\mathcal{H}_A\otimes\mathcal{H}_B)}, \qquad \|a\otimes b\|_{\min} = \|a\|\,\|b\|
$$

- **第一步，看定义**：在 $\mathcal{H}_A\otimes\mathcal{H}_B$ 上，$a\otimes b$ 作用为 $\pi_A(a)\otimes\pi_B(b)$（张量积算子）。取算子范数再完备化，得到 $A\otimes_{\min}B$。
- **第二步，看范数为什么是乘积**：$\|\pi_A(a)\otimes\pi_B(b)\|=\|\pi_A(a)\|\,\|\pi_B(b)\|=\|a\|\|b\|$（张量积算子的范数 = 因子范数之积，第 1 节 Hilbert 张量积的直接结论）。基本张量的范数是乘积，一般张量 $\sum a_i\otimes b_i$ 的范数则是「尽量拉开」后的算子范数。
- **第三步，看为什么与表示无关**：对 $a\otimes b$，$\|\pi(a)\otimes\pi(b)\|=\|a\|\|b\|$ 由 C\* 恒等式固定（两边都是 $r(a)r(b)$ 或直接算子范数），于是「最小范数」完全由 $A,B$ 自身决定，不挑表示。
- **第四步，看最小 vs 最大**：最小范数对应「取算子范数」（最省），最大范数对应「让所有表示都相容」（最费）。两者一致当且仅当 $A$ 可核。**可核性 = 张量积范数唯一 = 不存在「表示歧义」**——这是张量积理论给 C\* 代数世界留下的最重要的一道分水岭。

## 5 张量积的用武之地

**量子力学与量子信息**：复合系统态空间 $=\mathcal{H}_1\otimes\mathcal{H}_2$；纠缠、Bell 不等式、量子纠缠的数学描述全在张量积语言里。密度矩阵的张量积 $=\rho_1\otimes\rho_2$ 对应独立系统，非张量积的态就是关联/纠缠。

**K 理论与 Bott 周期**：张量积使 K 理论成为「环」：$K(A\otimes B)$ 有乘积结构；Bott 周期 $K_0(A)\cong K_0(A\otimes C_0(\mathbb{R}^2))$ 正是「张量积 $C_0(\mathbb{R}^2)$ 不改变 K 群」——第 25 篇 K 理论的六大项正合列与乘积都靠张量积。（第 25 篇预告。）

**构造新代数**：$A_\theta$ 与 $A_{\theta'}$ 的张量积、群 C\* 代数的张量积 $C^*(G\times H)\cong C^*(G)\otimes C^*(H)$（可均时最小=最大）——张量积是「复合结构」的代数配方。<span class="marginnote">群 C\* 代数的张量积恒等式把「直积群」翻译成「张量积代数」：$C^*(G\times H)=C^*(G)\otimes C^*(H)$。这使张量积成为「从基本块搭复杂系统」的标准积木，与非交换几何中「乘积空间」的概念一一对应。</span>

**辨析｜易错点：**代数张量积 $A\odot B$ 与完备化后的 $A\otimes_{\min}B$ 要分清：前者是稠密子代数（还没完备，还不是 C\* 代数），后者才是 C\* 代数。所有「无限和」只在完备化后才存在；在 $A\odot B$ 里只能取有限和。写张量积时先问：完备了吗？

## 6 例：张量积的数值与结构

用具体例子把「张量积」的各个侧面摸清。

**$\mathbb{C}^2\otimes\mathbb{C}^2$**：基 $\{e_1\otimes e_1,e_1\otimes e_2,e_2\otimes e_1,e_2\otimes e_2\}$，维数 $4$。Bell 态 $\frac1{\sqrt2}(e_1\otimes e_1+e_2\otimes e_2)$ 不能写成 $x\otimes y$——纠缠。

**$L^2(\mathbb{R})\otimes L^2(\mathbb{R})=L^2(\mathbb{R}^2)$**：$f\otimes g$ 对应 $f(x)g(y)$。张量积 = 「分离变量的函数」；一般函数（如 $e^{-xy}$）不是张量——张量积空间远大于「分离变量函数集」。

**$\mathcal{H}\otimes\mathcal{K}\cong$ Hilbert–Schmidt 算子**：$x\otimes y\mapsto$ 秩一算子。张量积空间与「算子空间」同构——第 6 篇的 $L^2(\mathcal{H})$ 正是这个同构。

**$C(\mathbb{T})\otimes C(\mathbb{T})=C(\mathbb{T}^2)$**：两个圆上的函数 = 环面上的函数。张量积 = 「乘积空间」的函数代数——这是交换情形最直观的例子。

**$M_n\otimes M_m\cong M_{nm}$**：矩阵的张量积 = 克罗内克积。有限维 C\* 代数的张量积仍是矩阵块。

**一句话总结**：张量积 = 「复合系统」——空间相乘、函数分离、算子张开；纠缠与分离变量是它的一体两面。

## 7 延伸：最小与最大张量积

最小与最大张量积的区分是 C\* 张量积理论的核心张力。

**最小（空间）张量积 $\otimes_{\min}$**：用「空间表示」（在 $\mathcal{H}_A\otimes\mathcal{H}_B$ 上）定义范数。对 $C(X)\otimes C(Y)$，它给出 $C(X\times Y)$——「最小的」范数，也是最「几何」的。

**最大张量积 $\otimes_{\max}$**：用「所有兼容表示」的范数上确界定义。一般大于最小——「最大的」范数，也是最「代数」的。

**何时相等**：$A$ 可核（第 19 篇 §3）⟺ $\otimes_{\min}=\otimes_{\max}$ 对所有 $B$。$B(\mathcal{H})$ 不可核——$B(\mathcal{H})\otimes_{\min}B(\mathcal{H})\neq B(\mathcal{H})\otimes_{\max}B(\mathcal{H})$。

**物理意义**：量子场论里，张量积范数的选择对应「代数独立 vs 统计相关」。W\*-代数（von Neumann）的张量积范数问题（Connes）至今深刻。

**K 理论的 Künneth 公式**：对可核代数，$K_*(A\otimes B)$ 由 $K_*(A),K_*(B)$ 的「张量积」算出——张量积在 K 理论里的行为需要可核性。

**一句话总结**：最小与最大张量积 = 「空间」与「代数」两种范数直觉的碰撞；可核性 = 它们和解的条件。

## 8 延伸：张量积与量子信息

张量积是量子信息的母语，几个核心概念都从它长出。

**复合系统**：两个系统的联合态空间 = 张量积。独立系统 = 张量积态 $\rho_1\otimes\rho_2$；纠缠态 = 非张量积的态。

**纠缠的量化**：$\rho$ 纠缠 ⟺ 不能写成 $\sum p_k\rho_k^{(1)}\otimes\rho_k^{(2)}$（可分离）。部分转置判据（PPT）：$\rho^{T_2}\ge0$ 是「非纠缠」的必要条件（$2\times2$ 与 $2\times3$ 时充分）。

**量子纠缠与 Bell 不等式**：Bell 态违背经典关联不等式——「非张量积的关联」在实验上可检验。张量积结构是「非局域性」的数学前提。

**量子信道**：信道 = 完全正映射 $\Phi:M_n\to M_m$；其「张量积扩张」$\Phi\otimes\mathrm{id}_k$ 保持正性。完全正性与张量积直接相关——量子信息的核心工具。

**纠缠熵**：$S(\rho_A)$ 由约化密度矩阵（部分迹 $\mathrm{Tr}_B$）定义——部分迹正是张量积的「取迹另一半」。张量积 + 部分迹 = 量子熵的语言。

**一句话总结**：张量积是「复合量子系统」的空间；纠缠、信道、熵全部由它定义——量子信息是张量积理论的「应用学科」。

## 9 小结

- **Hilbert 张量积**：$x\otimes y$ 双线性打包，基相乘、维数相乘；$L^2(X\times Y)=L^2(X)\otimes L^2(Y)$。
- **纠缠**：$\mathcal{H}\otimes\mathcal{K}$ 中不可分解的向量——复合系统比乘积态多出的自由度。
- **C\* 张量积**：范数不唯一；最小（空间）与最大张量积一般不同；$C(X)\otimes C(Y)=C(X\times Y)$。
- **可核性**：最小 = 最大对所有 $B$ 成立；$C(X)$、$\mathcal{K}$、AF、$\mathcal{O}_n$、$A_\theta$ 可核，$B(\mathcal{H})$ 不可核。
- **$\|a\otimes b\|_{\min}=\|a\|\|b\|$**：基本张量范数 = 乘积，最小范数与表示无关。
- **教训**：代数张量积与完备化是两回事；可核 ≠ AF。

在下一节，我们把「群作用」加到任意 C\* 代数上——**交叉积与动力系统 C\* 代数**：由动力系统 $(A,G,\alpha)$ 造出 $A\rtimes_\alpha G$