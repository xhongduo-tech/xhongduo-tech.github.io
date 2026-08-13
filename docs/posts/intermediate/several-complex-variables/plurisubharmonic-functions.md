---
title: 多重次调和函数：定义与逼近性质
date: 2026-08-07
---

# 多重次调和函数：定义与逼近性质

<div class="epigraph">
<p>多重次调和函数是多复变的「凸函数」——它把凸性从几何语言翻译成函数语言，并统治着整个 L² 理论。</p>
<footer>—— 仿 拉尔斯 · 赫尔曼德（Lars Hörmander），《多复变分析引论》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第3章；史济怀 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从多重次调和函数开始

本组前五篇已经三次提到同一个词：**凸性**——收敛域的对数凸、全纯凸性、伪凸域（预告）。为什么凸性在多复变里无处不在？因为**全纯函数族的性质被它的模 $\log|f|$ 的「凸性」所控制**。单复变中，$\log|f|$ 是**次调和**的（满足次均值不等式）；多复变中，它满足更强的性质——沿任意复直线的限制都是次调和的。这种函数叫 **多重次调和函数（plurisubharmonic function，简称 psh）**。<span class="marginnote">psh 函数在多复变中的地位，相当于「凸函数」在实分析中的地位：它提供正则化、逼近、上包络等一整套技术工具。Hörmander 的 L² 理论（本专题第 4 篇组）几乎完全建立在 psh 函数之上。</span>

## 1 定义：从次调和到多重次调和

**回顾单复变**：$\varphi: \Omega \to [-\infty, +\infty)$（上半连续，不恒为 $-\infty$）称为**次调和（subharmonic）**的，若对任意 $z$ 与充分小 $r$ 有**次均值不等式**：

$$
\varphi(z) \leq \frac{1}{2\pi} \int_0^{2\pi} \varphi(z + r e^{i\theta}) \, d\theta
$$

直觉：$\varphi$ 在圆心处的值不超过圆周上的平均值——「函数值不高于周围」，即函数是「碗」形的。

**多重次调和（plurisubharmonic）**：$\varphi$ 在 $\Omega \subset \mathbb{C}^n$ 上 psh，若 $\varphi$ 上半连续（取值为 $[-\infty, \infty)$），不恒为 $-\infty$，且**对任意复直线 $\zeta \mapsto a + b\zeta$（$a, b \in \mathbb{C}^n$）与任意紧子集上的 $U$，限制 $\varphi(a + b\zeta)$ 是次调和的**（在 $\zeta$ 的定义域内）。<span class="marginnote">「沿每条复直线次调和」比「整体次调和」强得多：psh 蕴含次调和，但次调和未必 psh。单复变里两者重合（复直线就是整个平面）。这个「沿直线」定义让 psh 函数天然适合拉回与复合，也为伪凸性提供了最简单的检验。</span>

**核心例子**：若 $f$ 全纯，则 $\log|f|$ 是 psh（且是次调和的）；若 $f_1, \dots, f_m$ 全纯，则 $\log(\sum |f_j|^2)$ 是 psh。这类函数在估计全纯函数模时处处出现。

## 2 psh 函数的基本性质

**性质 1（线性与保序）**：psh 函数的**正线性组合**（系数 $\geq 0$）仍是 psh；psh 函数的**上包络**（一族 psh 函数的上确界，经上半连续化）仍是 psh——只要它不恒为 $+\infty$。<span class="marginnote">上包络保 psh 性质是非平凡的，它依赖「次调和的上确界仍次调和」这一单变量事实。这使我们可以从一堆「检验函数」出发构造新的 psh 函数。</span>

**性质 2（复合单调）**：若 $\chi$ 是凸的增函数，$\varphi$ 是 psh，则 $\chi \circ \varphi$ 是 psh。特别地，$e^{\varphi}$ 与 $\varphi^+ = \max(\varphi, 0)$ 都是 psh——注意 $-\varphi$ 一般不 psh。

**性质 3（正则化）**：psh 函数可用**磨光子（mollifier）**正则化：设 $\rho_\varepsilon$ 是标准光滑核，则 $\varphi * \rho_\varepsilon$ 是**光滑 psh**函数，且在紧集上单调收敛到 $\varphi$。<span class="marginnote">这是 psh 理论最实用的性质之一：几乎所有定理都可先对光滑 psh 证明，再用逼近传给一般 psh。逼近的收敛是 $L^1_{\mathrm{loc}}$ 意义下的，且 $\varphi * \rho_\varepsilon \downarrow \varphi$（单调下降）。</span>

## 3 光滑 psh 函数与复 Hessian

若 $\varphi$ 是 $C^2$ 函数，psh 性等价于一个**常秩条件**：复 Hessian 半正定。设

$$
\left(\frac{\partial^2 \varphi}{\partial z_j \partial \bar z_k}\right)_{j,k=1}^n =: \partial\bar\partial \varphi
$$

则 $\varphi$ 是 psh $\iff$ 对任意 $w \in \mathbb{C}^n$，

$$
\sum_{j,k} \frac{\partial^2 \varphi}{\partial z_j \partial \bar z_k}\, w_j \bar w_k \;\geq\; 0
$$

即矩阵 $(\partial^2\varphi/\partial z_j\partial\bar z_k)$ 半正定。<span class="marginnote">注意 $z$ 与 $\bar z$ 是<strong>独立变量</strong>：$\partial^2/\partial z_j \partial \bar z_k$ 是复坐标系下的二阶导数，它不同于实 Hessian。psh 条件要求的是这个<strong>复 Hessian</strong>（又叫 Levi 形式）半正定，这正是下一节「伪凸域」边界上 Levi 形式的内部版本。</span>

**一个对照**：$\varphi(z) = |z_1|^2 - |z_2|^2$ 的复 Hessian 是对角阵 $\mathrm{diag}(1, -1)$，不定——所以不是 psh。而 $\varphi(z) = \log|z|$（多变量）在 $\mathbb{C}^n$ 中 psh 当且仅当 $n = 1$（因为 $\log|z|$ 沿直线的次调和性要求 $n \geq 2$ 时 $\zeta \mapsto \log|a+b\zeta|$ 是次调和，而它在 $b \neq 0$ 时 $\log|\zeta|$ 正是单变量次调和的）——细心的读者可验证：$\log|z|$ 的复 Hessian 是 $\delta_{jk}/|z|^2 - \bar z_j z_k/|z|^4$，恰半正定当且仅当 $n = 1$。

## 4 公式解析：复 Hessian 半正定条件

$$
\boxed{\; \varphi \in \mathrm{psh}(\Omega) \cap C^2(\Omega) \;\iff\; \sum_{j,k} \frac{\partial^2 \varphi}{\partial z_j \partial \bar z_k}\, w_j \bar w_k \geq 0 \;\forall w \in \mathbb{C}^n \;}
$$

- **第一步，识别算子**：$\partial\bar\partial \varphi = \sum_{j,k} \partial^2\varphi/\partial z_j\partial\bar z_k \, dz_j \wedge d\bar z_k$ 是复微分算子作用在 0 形式上的结果，形式上是一个 $(1,1)$-形式。对 $C^2$ 函数，$\partial\bar\partial \varphi$ 的非负性（作为 Hermitian 形式）就是 psh 性。
- **第二步，理解「沿直线次调和」与矩阵半正定的等价**：沿复直线 $z = a + w\zeta$，$\varphi$ 关于 $\zeta$ 的次调和性（等价于 $\Delta_\zeta \varphi \geq 0$，即 $\partial^2\varphi/\partial\zeta\partial\bar\zeta \geq 0$）由链式法则正好给出 $\sum \partial^2\varphi/\partial z_j\partial\bar z_k w_j\bar w_k \geq 0$。所以「每条直线次调和」$\iff$「复 Hessian 半正定」。
- **第三步，记住两个等价物**：对光滑函数，psh 有三种说法——(a) 沿复直线次调和；(b) 复 Hessian 半正定；(c) $\partial\bar\partial\varphi \geq 0$。它们各自对应一种证明技术，做题时按需切换。

## 5 辨析与延伸：psh 函数的五个易错点

**辨析 1：psh 强于次调和**。次调和只要求「球面均值不等式」，psh 要求「沿每条复直线次调和」。$n \geq 2$ 时次调和不是 psh。例子：$\varphi(z) = -|z_1|^2$ 在 $\mathbb{C}^2$ 中次调和（因为 $-\Delta$ 正？需检查）但不 psh——沿 $z_1$ 方向的复 Hessian 为负。<span class="marginnote">更保险的例子：$\varphi(z)=|z_1|^2-|z_2|^2$ 的复 Hessian $\mathrm{diag}(1,-1)$ 不定，故不是 psh；但它可能是次调和的（实 Laplacian 为零，是调和的，故次调和）。<strong>次调和 ⟸ 调和 ⟹ ？psh</strong>——调和未必 psh，这是关键区分。</span>

**辨析 2：$-\log|f|$ 不是 psh**。$f$ 全纯时 $\log|f|$ 是 psh，但 $-\log|f|$ 一般不是（它是「超调和」的，方向相反）。在极点附近 $-\log|f| \to +\infty$，但它不是 psh。**psh 函数像「碗」（向上凹），$-\log|f|$ 像「倒碗」**。

**辨析 3：psh 允许 $-\infty$ 值**。psh 函数取值于 $[-\infty, +\infty)$，允许在孤立点取 $-\infty$（如 $\log|f|$ 在零点）。但「不恒为 $-\infty$」是默认假设。初学者常被 $-\infty$ 吓到，其实它是「奇点容忍度」——psh 函数不怕对数奇点。

**辨析 4：上包络需要上半连续化**。一族 psh 函数的上确界逐点不一定上半连续，需取「上半连续包络」$u^*(z) = \limsup_{w\to z} u(w)$ 才是 psh。**「取上确界后要闭包」**——这个技术细节在构造伪凸定义函数时反复出现。<span class="marginnote">应用时记住口诀：$\max$ 或 $\sup$ 一族 psh ⟹ 上半连续化后仍是 psh。这是「局部拼整体」最常用的工具。</span>

**辨析 5：$\log|z|$ 在 $n \geq 2$ 时不是 psh**。这是最经典的反例：$\log|z|$ 在 $\mathbb{C}^n$（$n\geq2$）中沿复直线 $\zeta \mapsto \zeta b$ 的限制是 $\log|\zeta b| = \log|\zeta| + \log|b|$，在 $\zeta=0$ 处是 $-\infty$——等等，$\log|\zeta|$ 在 $\mathbb{C}$ 中**是**次调和的！所以 $\log|z|$ 沿每条复直线次调和……但 $\log|z|$ 的复 Hessian 是 $\frac{1}{|z|^2}(\delta_{jk} - \frac{\bar z_j z_k}{|z|^2})$，特征值 $(0,\dots,0,\frac{1}{|z|^2})$——半正定但有一方向为零，其实**是 psh**！真正的反例是 $\log|z_1|$（只对 $z_1$ 取对数，在 $z_1=0$ 处 $-\infty$，但它沿 $z_2$ 方向的直线限制是常函数，次调和；而 $\log|z_1|$ 的复 Hessian 有负方向吗？需逐例计算）。**结论：判断 psh 性务必用复 Hessian 半正定，别凭直觉**。<span class="marginnote">这个辨析的目的不是给出某个结论，而是示范正确的判断方法：光滑时用复 Hessian 半正定，不光滑时用「沿复直线次调和」。直觉在多复变里经常出错，必须回到定义。</span>

## 6 速查与误区

**psh 函数的「工具箱」**：

1. 判断光滑函数是否 psh：算复 Hessian $\partial\bar\partial\varphi$ 是否半正定。
2. 判断非光滑函数是否 psh：检验沿每条复直线的次调和性。
3. 构造新 psh：正线性组合、上包络（上半连续化）、凸增复合。
4. 正则化：用磨光子 $u \to u*\rho_\varepsilon$，单调下降逼近。

**误区清单**：

- **误区 1**：以为「次调和 = psh」。
  正解：$n \geq 2$ 时次调和比 psh 弱得多。
- **误区 2**：以为「$\log|f|$ psh ⟹ $-\log|f|$ 也 psh」。
  正解：方向相反，$-\log|f|$ 是超调和，一般不是 psh。
- **误区 3**：以为「psh 函数不能取 $-\infty$」。
  正解：可以，在奇点处取 $-\infty$（如 $\log|f|$ 在零点）。
- **误区 4**：以为「psh 上包络直接是 psh」。
  正解：先取上确界，再取上半连续包络，才保 psh。
- **误区 5**：以为「调和函数必 psh」。
  正解：调和（实 Laplacian 为零）不一定 psh；psh 要求复 Hessian 半正定。

**知识树**：

- 向后：次调和函数（单复变）与凸函数（实分析）——psh 是两者的复化融合。
- 向前：伪凸域（第 2 组）——psh 定义函数是伪凸性的第三种等价形态。
- 横向：Hörmander L² 理论（第 4 组）——权函数取 psh 时基本恒等式成立。

**一句话记忆**：psh = 「沿复直线次调和」+ 上半连续 = 「复 Hessian 半正定」（光滑时）。它是多复变的凸函数。

## 7 小结

- **多重次调和（psh）**：上半连续、沿每条复直线次调和的函数；$\log|f|$（$f$ 全纯）是基本例子。
- **性质**：正线性组合、上包络、凸增复合、磨光逼近都保持 psh 性。
- **光滑情形**：psh $\iff$ 复 Hessian 半正定 $\iff$ $\partial\bar\partial \varphi \geq 0$