---
title: Toeplitz 算子与 Toeplitz 代数
date: 2026-08-07
---

# Toeplitz 算子与 Toeplitz 代数

<div class="epigraph">
<p>完美的纯粹数学，其可应用性是它真实性的一个自然结果。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Davidson《C\*-Algebras by Example》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从 Toeplitz 代数开始

前两篇把抽象理论讲得天花乱坠，现在该「验货」了。**Toeplitz 代数**是第一件最完美的样品：它由**单侧移位算子**这一个算子生成，却同时包含了紧算子、连续函数代数、以及一条精确到极致的短正合列。它把第 5 篇的 Fredholm 指标、第 8 篇的 Gelfand 变换、第 12 篇的理想与商，全部拧进一个具体的、可计算的例子里。

为什么 Toeplitz 代数值得单开一篇？因为它是**所有后续例子的样板**。无理旋转代数、群 C\* 代数、交叉积，本质上都在重复它的配方：取一个「好」的生成元（或一群生成元），观察它生成的 C\* 代数，用一个「符号映射」读它的商结构，再用指标理论数它的亏量。理解 Toeplitz 代数，就是理解整套 C\* 代数构造法的第一步。

## 1 Hardy 空间与单侧移位

**Hardy 空间** $H^2$：单位圆盘上满足 $\sup_{r\lt 1}\int_0^{2\pi}|f(re^{it})|^2dt\lt \infty$ 的解析函数空间。它有标准正交基 $\{z^n\}_{n\ge0}$（或 $\{e_n\}_{n\ge0}$），故 $H^2\cong\ell^2(\mathbb{N}_0)$。<span class="marginnote">$H^2$ 的妙处在于：它既是「圆盘上解析函数」的优雅舞台，又和 $\ell^2(\mathbb{N})$ 有显式等距——前者给几何直觉（Hardy 类、边界值），后者给代数直觉（移位算子）。物理/工程里 $H^2$ 还出现在因果滤波器的频率响应里。</span>

**单侧移位（unilateral shift）**：$S:H^2\to H^2$，$(Sf)(z)=zf(z)$，即 $Se_n=e_{n+1}$。

**关键性质**：
- $S^*S=I$（等距），$SS^*=I-P_0$（$P_0$ 是向 $e_0$ 的投影）；
- $\sigma(S)=\overline{\mathbb{D}}$（第 3 篇见过：边界是连续谱，内部是剩余谱）；
- $S$ **不正规**——它是「非正规算子也有函数演算」反例的活标本。

## 2 Toeplitz 算子：把符号嵌进 H²

对 $f\in L^\infty(\mathbb{T})$，定义**乘法算子** $M_f$ 于 $L^2(\mathbb{T})$，其 **Toeplitz 算子**是它在 $H^2$ 上的「压缩」：

$$T_f = P\, M_f\big|_{H^2},$$

其中 $P:L^2(\mathbb{T})\to H^2$ 是正交投影（把 Fourier 级数「砍掉负频率」）。$f$ 叫**符号（symbol）**。

**矩阵形态**：在基 $\{z^n\}$ 下，$T_f$ 的矩阵是 $(\widehat f(m-n))_{m,n\ge0}$——**Toeplitz 矩阵**（沿对角线常数）。这是「Toeplitz 算子」一词的来源，也是它与 Toeplitz 矩阵论、预测理论交汇的接口。<span class="marginnote">「砍掉负频率」的投影 $P$ 是 Toeplitz 理论的心脏：$T_f$ 先把信号乘上 $f$，再把负频率砍掉。这个「乘再截断」的过程在信号处理里叫加窗/滤波，Toeplitz 算子正是它的算子化。</span>

**基本性质**：$T_f$ 有界且 $\|T_f\|=\|f\|_\infty$；$T_f^*=T_{\overline f}$；$T_{fg}\neq T_fT_g$（一般不等，因截断破坏乘法）——**符号乘法不保**，这正是 Toeplitz 代数非交换性的来源。

## 3 Toeplitz 代数：一个生成元的全部子嗣

**Toeplitz 代数** $\mathcal{T}=C^*(S)$ 是由单侧移位生成的 C\* 代数（也等于 $\{T_f:f\in C(\mathbb{T})\}$ 的生成代数）。

**定理（Toeplitz 代数的结构）**：
1. 紧算子都在里面：$\mathcal{K}(H^2)\subset\mathcal{T}$；
2. **符号映射**（symbol map）：存在满 $\ast$-同态 $\sigma:\mathcal{T}\to C(\mathbb{T})$，$\sigma(T_f)=f$，$\ker\sigma=\mathcal{K}(H^2)$；
3. 因此有**短正合列**：

$$0 \longrightarrow \mathcal{K}(H^2) \longrightarrow \mathcal{T} \xrightarrow{\ \sigma\ } C(\mathbb{T}) \longrightarrow 0.$$<span class="marginnote">这条短正合列是整套理论的「宪法」：$\mathcal{T}$ 的每个元素 $X$ 都等于「一个 Toeplitz 算子 + 一个紧算子」，即 $X=T_{\sigma(X)}+K$。商掉紧算子后剩下的恰好是「圆上的连续函数」。Toeplitz 代数是 Calkin 代数思想的最纯粹样本。</span>

**推论（Fredholm 性判据）**：$T_f$ 是 Fredholm 的当且仅当 $f$ 处处非零（$0\notin f(\mathbb{T})$），此时

$$\operatorname{ind}(T_f) = -\operatorname{wind}(f, 0),$$

指标等于符号绕原点的**卷绕数（winding number）**。分析指标 = 拓扑指标——这是 Atiyah–Singer 指标定理的最早、最具体的小型版本。

## 4 公式解析：$\operatorname{ind}(T_f)=-\operatorname{wind}(f,0)$

$$
\operatorname{ind}(T_f) = \dim\ker T_f - \dim\mathrm{coker}\,T_f = -\frac{1}{2\pi}\int_{\mathbb{T}} d\arg f = -\mathrm{wind}(f,0)
$$

- **第一步，看左端**：$\operatorname{ind}(T_f)$ 是第 5 篇的 Fredholm 指标——方程 $T_fg=h$ 的「解数减缺解数」。
- **第二步，看右端**：$\mathrm{wind}(f,0)$ 是 $f$ 的像曲线绕原点转的圈数，由 $\arg f$ 的增量除以 $2\pi$ 计算。它是纯拓扑量（同伦不变），只数「圈数」，不数具体形状。
- **第三步，看等式为什么成立**：对 $f(z)=z^n$，$T_z=S$（移位），$\operatorname{ind}(S)=-1$，而 $\mathrm{wind}(z^n,0)=n$，等式给出 $-1=-n$ 当 $n=1$；一般 $n$ 用 $S^n$ 验证。任意 $f$ 分解为「单项式 × 非零同伦扰动」，指标的同伦不变性 + 卷绕数的同伦不变性把二者锁定。
- **第四步，看意义**：一个**分析对象**（解空间的维数差）等于一个**拓扑对象**（卷绕圈数）。这解释了为什么指标在巨大形变下不变——拓扑量本就不怕连续形变。整个指标理论（第 25 篇 K 理论里的指标映射）都在这条等式的延长线上。

## 5 Toeplitz 代数教会我们的

**例 1（不可约与理想）**：$\mathcal{T}$ 包含紧算子，故不是简单代数；它的理想格由 $\mathcal{K}$ 与正合列完全描述。**任何含非零紧算子的 C\* 代数都「肥胖」**——紧算子是理想的种子。

**例 2（本质谱）**：$T_f$ 的谱由「本质谱 + 卷绕信息」拼成：$\sigma_{\mathrm{ess}}(T_f)=f(\mathbb{T})$（符号的值域），而完整谱还要加上「指标非零处填满的洞」。物理上，本质谱对紧扰动稳定——这正对应量子力学中连续谱的稳定性。<span class="marginnote">「谱 = 本质谱 + 指标填充」是 Toeplitz 理论的招牌结论：$f$ 绕原点转一圈时，$T_f$ 的谱覆盖整条「洞」里的圆盘。这个「指标填洞」现象在第 25 篇 K 理论里会以更抽象的形式再现。</span>

**例 3（通往非交换几何）**：$\mathcal{T}$ 是**可核（nuclear）**C\* 代数（第 19 篇张量积会再遇），且是 **扩展（extension）**$0\to\mathcal{K}\to\mathcal{T}\to C(\mathbb{T})\to0$ 的典范——第 25 篇 K 理论里，这类扩展的分类由指标映射控制。

**辨析｜易错点：**不要以为 $T_f$ 的范数等于 $f$ 的「$H^2$ 上范数」的平凡事——$\|T_f\|=\|f\|_\infty$ 需要证明（不是显然的 $H^2$ 压缩保持范数）。反之，$T_fT_g\neq T_{fg}$ 是常态而非例外：Toeplitz 算子只「近似乘法」。把 $T_{fg}$ 与 $T_fT_g$ 混为一谈，就会错失正合列中「紧算子修正」的全部精妙。

## 6 例：Toeplitz 算子谱的完整计算

把 $T_f$ 的谱、本质谱与指标完整算一遍，Toeplitz 理论就「活了」。

**$T_z=S$（移位）**：$\sigma(S)=\overline{\mathbb{D}}$，$\sigma_{\mathrm{ess}}(S)=\mathbb{T}$（$S$ 的「本质谱」是单位圆，因为 $S$ 模紧算子是酉）。指标 $\operatorname{ind}(S)=-1$。

**$T_{\overline z}=S^*$**：$\sigma(S^*)=\overline{\mathbb{D}}$ 同（伴随谱共轭），但 $\operatorname{ind}(S^*)=+1$。指标「差一个符号」——左右移位的镜像关系。

**实值符号 $f$（如 $f(e^{it})=t/\pi$）**：$T_f$ 自伴，$\sigma(T_f)$ 是实区间 $[\min f,\max f]$。谱 = 符号的值域，与「$f$ 在圆上的摆动」完全一致。

**指标非零的填充**：$f$ 绕原点转一圈（$\mathrm{wind}=1$），$\sigma(T_f)$ 填满整个单位圆盘——「指标非零处谱被填充」。这是 Toeplitz 谱理论最迷人的现象。

**本质谱 = 值域**：$\sigma_{\mathrm{ess}}(T_f)=f(\mathbb{T})$ 对一切 $f$。紧扰动不改本质谱，故「本质谱」完全由符号决定——符号是 $T_f$ 的「无穷远行为」。

**一句话总结**：$T_f$ 的谱 = 「$f$ 的值域 + 指标填洞」，本质谱 = 值域，指标 = 卷绕数——三个量全由符号 $f$ 读出。

## 7 延伸：Toeplitz 代数与 K 理论

Toeplitz 代数是 K 理论（第 25 篇）最好的入门例子。

**短正合列**：$0\to\mathcal{K}\to\mathcal{T}\to C(\mathbb{T})\to0$ 是「扩展」（extension）的典范。K 理论对它的反应是六项正合列。

**指数映射**：$K_1(C(\mathbb{T}))\to K_0(\mathcal{K})$ 由 $f\mapsto\operatorname{ind}(T_f)$ 给出。$K_1(C(\mathbb{T}))=\mathbb{Z}$（卷绕数），$K_0(\mathcal{K})=\mathbb{Z}$（Fredholm 指标）——指标映射把「卷绕」变成「指标」。

**Toeplitz 代数的 K 群**：$K_0(\mathcal{T})=\mathbb{Z}$，$K_1(\mathcal{T})=0$（六项列算出）。$\mathcal{T}$ 的 K 理论「平凡」，但它的扩展结构非平凡——指标映射捕捉的正是这「非平凡」。

**扩展的分类**：$0\to\mathcal{K}\to E\to C(\mathbb{T})\to0$ 的等价类由指标映射的「相」参数化。Toeplitz 扩展是「半平凡」扩展的原型——Brown–Douglas–Fillmore 理论（BDC）的起点。

**一句话总结**：Toeplitz 代数把「指标 = 卷绕数」收编进 K 理论的六项正合列——扩展理论由此起步。

## 8 延伸：从 Toeplitz 到更一般的扩展

Toeplitz 只是一个开始——它的配方可以推广到任何理想扩展。

**一般扩展**：$0\to\mathcal{K}\to E\to A\to0$（$E$ 含紧算子，商是 $A$）。Toeplitz 是 $A=C(\mathbb{T})$ 的情形；$A=C(X)$ 一般情形的分类由 BDF 理论（Brown–Douglas–Fillmore）完成。

**指标不变量的推广**：对 $A=C(X)$，指标映射 $K_1(C(X))\to K_0(\mathcal{K})=\mathbb{Z}$ 不再是「整数」，而是「$X$ 上的连续指标函数」——更丰富的几何信息。

**与 Cuntz 代数的关系**：Cuntz 代数 $\mathcal{O}_n$（第 18 篇）满足 $0\to\mathcal{K}\to E_n\to\mathcal{O}_n\otimes\mathcal{K}\to0$ 之类的扩展结构——Toeplitz 的配方在纯无限世界再现。

**与非交换几何**：扩展 $0\to\mathcal{K}\to E\to A\to0$ 对应「非交换空间 $A$ 的紧化」——Connes 的非交换几何用扩展描述「带边界的非交换流形」。

**一句话总结**：Toeplitz 代数是「紧算子扩展」的种子——从它长出 BDF 理论、Cuntz 扩展与非交换几何的「带边空间」。

## 9 小结

- **Hardy 空间** $H^2$ 与单侧移位 $S$：等距但不酉，$\sigma(S)=\overline{\mathbb{D}}$。
- **Toeplitz 算子** $T_f=P M_f|_H^2$：乘符号再砍负频率；矩阵是 Toeplitz 矩阵；$\|T_f\|=\|f\|_\infty$。
- **Toeplitz 代数** $\mathcal{T}=C^*(S)$：含紧算子，符号映射 $\sigma:\mathcal{T}\to C(\mathbb{T})$，$\ker\sigma=\mathcal{K}$。
- **短正合列** $0\to\mathcal{K}\to\mathcal{T}\to C(\mathbb{T})\to0$ 是整套理论的骨架。
- **指标定理**：$T_f$ Fredholm ⟺ $0\notin f(\mathbb{T})$，$\operatorname{ind}(T_f)=-\mathrm{wind}(f,0)$。
- **教训**：非正规单生成算子照样长出丰富的 C\* 代数；商结构 + 指标是读它的两只眼睛。

在下一节，我们用两个「非交换生成元」替换一个移位——**无理旋转代数与非交换环面**：一个由两个酉元 $U,V$ 以 $VU=e^{2\pi i\theta}UV$