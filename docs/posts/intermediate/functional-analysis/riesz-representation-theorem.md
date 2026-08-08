---
title: Riesz 表示定理及其应用
date: 2026-08-07
---

# Riesz 表示定理及其应用

<div class="epigraph">
<p>在 Hilbert 空间里，每一个线性泛函都是一次内积——这也许是分析学中最优美的表示定理。</p>
<footer>—— 弗里杰什 · 里斯（Frigyes Riesz），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§4.10 ｜ 2026-08-07</p>
</div>

## 为什么这是第四章的巅峰

回顾第三章：对偶空间 $X^*$ 是「全体连续线性泛函」的空间。对一般 Banach 空间，$X^*$ 是神秘莫测的对象（$c_0$ 的对偶是 $l^1$，但 $X^*$ 本身还能更复杂）。可 Hilbert 空间给出了一个惊人的答案：**每个连续线性泛函 $f$ 都长成「固定一个向量 $y$，做内积 $\langle \cdot, y\rangle$」的样子。** 于是 $H^*$ 与 $H$ 一一对应——**Hilbert 空间的对偶就是它自己**。这就是 **Riesz 表示定理（Riesz representation theorem）**，它把「泛函」还原成「向量」，让对偶理论、弱收敛、伴随算子全部变得无比清晰。<span class="marginnote">「表示」的含义：用具体的「代表元」$y$ 完全描述抽象泛函 $f$。$f(x) = \langle x, y\rangle$ 意味着「$f$ 就是『与 $y$ 做内积』这个操作」。Riesz 表示定理在有限维就是「线性泛函 = 行向量」，但它对无穷维也成立——这是 Hilbert 空间独有的福利。</span>

## 1 定理陈述

**定理（Riesz 表示定理）：设 $H$ 是 Hilbert 空间。则对每个连续线性泛函 $f \in H^*$，存在唯一的 $y \in H$，使**

$$
f(x) = \langle x, y\rangle, \qquad \forall x \in H
$$

**且 $\|f\| = \|y\|$。** 反之，每个 $y \in H$ 都通过 $f_y(x) = \langle x, y\rangle$ 给出一个连续线性泛函（Cauchy-Schwarz 保证 $\|f_y\| \le \|y\|$，取 $x = y$ 得等号）。于是

$$
H^* \cong H, \qquad f \longleftrightarrow y
$$

**核心要点：$H^*$ 与 $H$ 是同一个空间（反线性同构）**。泛函的问题全部可以翻译成向量的语言。

## 2 证明：用正交分解构造代表元

证明优雅且只用正交分解定理：

- **第一步，若 $f = 0$**：取 $y = 0$ 即可。
- **第二步，找「方向」**：$f \neq 0$ 时，$\ker f$ 是 $H$ 的**真闭**子空间（核是闭的，因为 $f$ 连续；核不可能是全空间）。由正交分解定理，存在非零 $z \perp \ker f$。
- **第三步，构造代表元**：令 $y = \frac{\overline{f(z)}}{\|z\|^2} z$（复空间需共轭）。对任意 $x \in H$，把 $x$ 正交分解为 $x = m + \lambda z$（$m \in \ker f$、$\lambda z$ 沿 $z$ 方向），则 $f(x) = f(m) + f(\lambda z) = \lambda f(z)$。另一方面 $\langle x, y\rangle = \lambda\langle z, y\rangle = \lambda \frac{f(z)}{\|z\|^2}\|z\|^2 = \lambda f(z)$。两者相等。
- **第四步，唯一性与范数**：唯一性由「$\langle x, y_1\rangle = \langle x, y_2\rangle$ 对一切 $x$ ⟹ $y_1 = y_2$」给出；$\|f\| = \|y\|$ 由 $|f(x)| \le \|x\|\|y\|$ 与 $f(y) = \|y\|^2$ 联合得到。<span class="marginnote">证明的引擎是<strong>正交分解</strong>：$H = \ker f \oplus (\ker f)^\perp$。代表元 $y$ 就住在 $(\ker f)^\perp$ 这条「垂直方向」上——泛函 $f$ 由它的核完全决定，而核的正交补只给出一维自由度。这从几何上解释了「为什么表示元只有一个向量」。</span>

## 3 公式解析：构造代表元的那一步

把 $y = \frac{\overline{f(z)}}{\|z\|^2}z$ 的合理性拆开：

**第一步，$z$ 的角色**：$z \perp \ker f$ 且 $f(z) \neq 0$。$z$ 是「$f$ 的方向」——$f$ 在垂直于核的方向上取值。
**第二步，$y$ 沿 $z$**：$y = \alpha z$，其中 $\alpha = \overline{f(z)}/\|z\|^2$。于是 $\langle z, y\rangle = \alpha\|z\|^2 = f(z)$。
**第三步，推广到全空间**：任意 $x$ 分解为 $m + \lambda z$（$m\in\ker f$）。$f(x) = \lambda f(z)$；$\langle x,y\rangle = \langle m,y\rangle + \lambda\langle z,y\rangle = 0 + \lambda f(z)$（$m \perp y$ 因 $y$ 沿 $z$ 方向而 $m\in\ker f \perp z$）。相等。

**关键**：代表元 $y$ 的构造完全被「$y \perp \ker f$ 且 $\langle z, y\rangle = f(z)$」两条性质锁定——**表示定理说：泛函在垂直于核的方向上取值，且这个方向唯一**。

## 4 应用一：对偶空间即自身

Riesz 表示定理的第一个推论已经给出：$H^* \cong H$。这意味着：

**弱收敛的定义简化**：$x_n \rightharpoonup x$（弱收敛，第七章）定义为「对一切 $f \in H^*$，$f(x_n) \to f(x)$」。在 Hilbert 空间里，这等价于「对一切 $y \in H$，$\langle x_n, y\rangle \to \langle x, y\rangle$」——**不需要知道 $H^*$ 长什么样，直接对向量做内积即可**。<span class="marginnote">Riesz 表示定理让 Hilbert 空间的弱收敛理论「免构造」：别的空间要费力描述 $X^*$，Hilbert 空间用 $y$ 直接代替 $f$。这也是为什么 Hilbert 空间是「最温柔的无穷维空间」——对偶不添乱。</span>

**例（$L^2$）**：$L^2[a,b]$ 上的泛函 $f \mapsto \int_a^b f(t)\overline{g(t)}\,dt$（$g \in L^2$）正是 Riesz 表示定理中的 $f_g$。**「$L^2$ 的对偶是 $L^2$」是 $L^p$ 对偶理论（$p=2$ 特例）与 Riesz 表示定理的统一表述。**

## 5 应用二：伴随算子的存在性

对 $T \in \mathcal{B}(H_1, H_2)$，固定 $y \in H_2$，映射 $x \mapsto \langle Tx, y\rangle$ 是 $H_1$ 上的连续线性泛函（$|\langle Tx,y\rangle| \le \|T\|\|x\|\|y\|$）。由 Riesz 表示定理，存在唯一 $T^* y \in H_1$ 使

$$
\langle T x, y\rangle_{H_2} = \langle x, T^* y\rangle_{H_1}
$$

**$T^*$ 就是 $T$ 的伴随算子（adjoint operator）**。它的存在性完全由 Riesz 表示定理保证——这正是下一节「伴随算子」的铺垫。<span class="marginnote">有限维里 $T^*$ 对应共轭转置 $A^* = \overline{A^T}$。Riesz 表示定理让「转置」在无穷维存在且唯一。自伴算子 $T = T^*$、正规算子 $TT^* = T^*T$ 都是下一节与第九章谱理论的主角。</span>

## 6 应用三：变分法与弱解（预告）

变分法的核心事实：泛函的极小点满足「Euler-Lagrange 方程」，而这方程常常是「对一切测试函数做内积」——即弱解定义。Riesz 表示定理把「方程」还原为「找代表元」。第十章我们将看到：椭圆型偏微分方程的弱解存在性（Lax-Milgram 定理）正是 Riesz 表示定理的推广——用「共轭双线性形式」代替内积，结论依然成立。<span class="marginnote">Lax-Milgram 定理说：若 $a(\cdot,\cdot)$ 是连续强制的共轭双线性形式，则对每个 $f$ 存在唯一 $u$ 使 $a(u,v) = f(v)$ 对一切 $v$。它把 Riesz 表示定理推广到「非对称内积」，是偏微分方程弱解理论的基石——第十章变分法一节会展开。</span>

## 7 表示定理在 L^2 中的具体形态

Riesz 表示定理在 $L^2$ 中的形态最常用，值得单独拆开看。

**定理（$L^2$ 版本）**：$L^2[a,b]$ 上的每个连续线性泛函 $f$ 都唯一地由某个 $g \in L^2$ 表示：

$$
f(\varphi) = \int_a^b \varphi(t)\,\overline{g(t)}\, dt = \langle \varphi, g\rangle, \qquad \|f\| = \|g\|_2
$$

这与第三章的 $(L^p)^* = L^q$（$p = q = 2$）完全一致——Riesz 表示定理是对偶理论的 Hilbert 版本。

**例一（求值泛函的局限）**：$\delta_{t_0}(\varphi) = \varphi(t_0)$ 在 $C[0,1]$ 上是连续线性泛函，但它**不能**写成「与某个 $L^2$ 函数做内积」——因为 $L^2$ 函数无法「抓住单点的值」。这解释了为什么 $C$ 的对偶是测度而不是函数，也预告了分布论（广义函数）的登场：$\delta$ 要放进 $L^2$ 的对偶之外。

**例二（傅里叶系数算子）**：$\varphi \mapsto \langle \varphi, e^{int}\rangle$ 是「取第 $n$ 个傅里叶系数」的泛函，由 $g = e^{int}$ 表示。帕塞瓦尔等式 $\sum |\langle f, e_n\rangle|^2 = \|f\|^2$ 正是「傅里叶系数算子保范」的表述——它把 $L^2$ 等距嵌入 $l^2$。

**应用（Galerkin 方法的原型）**：解方程 $Au = b$（$A$ 自伴正定）时，把问题限制在有限维子空间 $V_N$ 上，用 Riesz 表示定理把「弱形式 $\langle Au, v\rangle = \langle b, v\rangle$」翻译成有限维线性方程组。这正是有限元方法的第一性原理——表示定理保证「变分形式」与「算子形式」不丢信息。

**核心要点：Riesz 表示定理让「泛函」与「向量」可以互换**。在 $L^2$ 里，这意味着「线性观测」与「与某个函数做内积」是同一件事——这是从傅里叶分析到量子力学一切「配对」语言的基础。

## 8 例题精讲：找表示元的三个计算

**例题一：$\mathbb{R}^2$ 中的表示元**。

- 泛函 $f(x_1,x_2) = 2x_1 - x_2$，要找 $y$ 使 $f(x) = \langle x, y\rangle$。
- $y = (2, -1)$：$\langle x, y\rangle = 2x_1 - x_2$。
- $\|f\| = \|y\| = \sqrt5$。表示元就是「系数组成的向量」。

**例题二：$L^2$ 中的表示元**。

- $f(\varphi) = \int_0^1 t\varphi(t)\,dt$，表示元 $g(t) = t$。
- $\|f\| = \|g\|_2 = \sqrt{1/3}$。
- 积分泛函的表示元 = 被积核。

**例题三：用正交分解找表示元**。

- $f \neq 0$ 时，$y = \overline{f(z)} z/\|z\|^2$，其中 $z \perp \ker f$。
- $\ker f$ 是超平面，$z$ 是它的法向。
- 表示元 = 沿法向方向的向量，长度由 $f(z)$ 决定。

**核心要点**：找表示元的三条路——「直接读系数」「识别核」「用正交分解」——本质都是「$y \perp \ker f$ 且 $\langle z,y\rangle = f(z)$」。

**辨析｜易错点：** 复空间表示元带共轭：$f(x) = \langle x, y\rangle$，$y$ 的系数是 $f$ 的系数取共轭。


## 9 小结

- **Riesz 表示定理**：每个 $f \in H^*$ 唯一地是 $f(x) = \langle x, y\rangle$，且 $\|f\| = \|y\|$；$H^* \cong H$。
- **证明引擎**：正交分解 $H = \ker f \oplus (\ker f)^\perp$，代表元 $y$ 沿垂直方向唯一确定。
- **应用一**：$H^* = H$，弱收敛写成内积版（第七章的基础）。
- **应用二**：伴随算子 $T^*$ 的存在性与唯一性由 Riesz 保证。
- **应用三**：Lax-Milgram 把表示定理推广到弱解理论（第十章伏笔）。

在下一节，我们正式研究**伴随算子的定义与性质**——把「转置」推广到无穷维，并考察自伴、正规算子的基本结构。
