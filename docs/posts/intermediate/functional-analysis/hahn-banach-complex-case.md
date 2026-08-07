---
title: 复线性空间情形的延拓
date: 2026-08-07
---

# 复线性空间情形的延拓

<div class="epigraph">
<p>复泛函不外乎是实泛函的「实部」加上了它的「虚部」——延拓问题因此可以化归。</p>
<footer>—— 斯特凡 · 巴拿赫（Stefan Banach），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§6.2 ｜ 2026-08-07</p>
</div>

## 为什么要单独处理复情形

复泛函分析（全纯函数、算子谱）离不开复线性泛函。但上一节的 Hahn-Banach 延拓定理是为实空间证明的——它的关键不等式 $f(x) \le p(x)$ 依赖「实值」的次序（$\le$ 对复数没有意义）。复线性泛函取复数值，无法直接与实值 $p$ 比较。解决之道是**把复问题化归为实问题**：复线性泛函由它的实部决定（$f(x) = \operatorname{Re}f(x) + i\operatorname{Im}f(x)$），而实部是实线性泛函。先对实部延拓，再「拼回」复延拓。<span class="marginnote">这个「化归」是泛函分析处理复结构的通用模板：<strong>复空间的问题先拆成实部/虚部，用实理论解决，再拼回去</strong>。下一节「赋范空间上的保范延拓」也沿用同一策略——先证实版本，再推复版本。</span>

## 1 复线性泛函 = 实部 + 虚部

设 $X$ 是复线性空间，$f : X \to \mathbb{C}$ 是复线性泛函。令 $u = \operatorname{Re} f$（实部），则：

- $u$ 是**实线性泛函**（把 $X$ 当作实线性空间看，$u: X \to \mathbb{R}$ 线性）；
- **$f$ 由 $u$ 完全决定**：$f(x) = u(x) - i\,u(ix)$。

**推导**：$f(x) = u(x) + iv(x)$，其中 $v(x) = \operatorname{Im} f(x)$。由复线性，$f(ix) = if(x) = i(u(x) + iv(x)) = -v(x) + iu(x)$，故 $u(ix) = \operatorname{Re} f(ix) = -v(x)$，即 $v(x) = -u(ix)$。于是

$$
f(x) = u(x) - i\,u(ix)
$$

**核心要点：复线性泛函与其实部一一对应**——给定实线性泛函 $u$，上式自动给出复线性 $f$（可验证 $f(ix) = if(x)$）。这是「复问题化归实问题」的代数基础。<span class="marginnote">这条公式是「结构定理」：<strong>复线性 = 实线性 + 与「乘 $i$」相容</strong>。泛函理论里，研究复泛函 $f$ 等价于研究实泛函 $u = \operatorname{Re}f$——差一个「乘 $i$ 再取实部」的变换。这也是为什么 Hahn-Banach 的复版本证明只有两行：其余全是实版本的搬运。</span>

## 2 复次线性泛函与半范数

复情形用什么代替「次线性泛函 $p$」？答案是**半范数（seminorm）** $p$：

$$
p(x + y) \le p(x) + p(y), \qquad p(\alpha x) = |\alpha|\, p(x)
$$

注意与实次线性泛函的差别：齐次性用绝对值 $|\alpha|$（对复 $\alpha$），且自动 $p \ge 0$（$0 = p(0) \le 2p(x)$）。范数、范数的一半等典型半范数。

**复 Hahn-Banach（半范数版）**：设 $p$ 是复线性空间 $X$ 上的半范数，$M$ 是子空间，$f: M \to \mathbb{C}$ 是满足 $|f(x)| \le p(x)$（$x \in M$）的复线性泛函。则存在复线性延拓 $\tilde f : X \to \mathbb{C}$ 使 $| \tilde f(x)| \le p(x)$ 对一切 $x$。<span class="marginnote">复版本的控制条件是「绝对值」$|\tilde f(x)| \le p(x)$，不是「单侧」$f(x) \le p(x)$——因为复数的值不能与实数 $p$ 直接比较大小。这个「绝对值控制」是复版本的签名，下一节保范延拓也用它。</span>

## 3 证明：化归为实版本

证明只有两步，把「化归」的威力展示得淋漓尽致：

- **第一步，实化**：把 $X$ 当实空间，$u = \operatorname{Re} f$ 是实线性泛函，且 $u(x) \le |f(x)| \le p(x)$——$u$ 满足实 Hahn-Banach 的条件（$p$ 是实次线性泛函，因为 $p(\alpha x) = |\alpha|p(x)$，对 $\alpha \ge 0$ 是 $=\alpha p(x)$）。由实版本，存在实线性延拓 $\tilde u : X \to \mathbb{R}$ 使 $\tilde u(x) \le p(x)$。
- **第二步，复化**：定义 $\tilde f(x) = \tilde u(x) - i\,\tilde u(ix)$。由第 1 节，$\tilde f$ 复线性且在 $M$ 上等于 $f$。还需验证控制 $|\tilde f(x)| \le p(x)$：

对任意 $x$，令 $\tilde f(x) = |\tilde f(x)| e^{i\theta}$，则

$$
|\tilde f(x)| = e^{-i\theta}\tilde f(x) = \tilde f(e^{-i\theta}x)
$$

这是实数（因为 $\tilde f(e^{-i\theta}x) = |\tilde f(x)| \in \mathbb{R}$），故 $\tilde f(e^{-i\theta}x) = \tilde u(e^{-i\theta}x)$。于是

$$
|\tilde f(x)| = \tilde u(e^{-i\theta}x) \le p(e^{-i\theta}x) = |e^{-i\theta}|\, p(x) = p(x)
$$

证毕。<span class="marginnote">最后一步「旋转到实轴」是复分析的标准技巧：<strong>先把复数的模变成「旋转后的实值」，再用实控制</strong>。$|\tilde f(x)| = \tilde f(e^{-i\theta}x)$ 这一招，在复分析里处理「模长与线性」的关系时反复出现。</span>

## 4 公式解析：复化拼回的验证

把「$\tilde f$ 复线性 + 控制」的验证拆开：

$$
\tilde f(x) = \tilde u(x) - i\,\tilde u(ix)
$$

- **第一步，复线性**：需要验证 $\tilde f(ix) = i\tilde f(x)$。计算 $\tilde f(ix) = \tilde u(ix) - i\tilde u(i(ix)) = \tilde u(ix) - i\tilde u(-x) = \tilde u(ix) + i\tilde u(x)$（实线性），而 $i\tilde f(x) = i\tilde u(x) - i^2\tilde u(ix) = i\tilde u(x) + \tilde u(ix)$——两者相等。实线性由 $\tilde u$ 实线性 + 此式保证。
- **第二步，控制**：对 $x$ 取 $\theta$ 使 $e^{-i\theta}\tilde f(x) = |\tilde f(x)| \in \mathbb{R}$。则 $|\tilde f(x)| = \tilde f(e^{-i\theta}x)$（复线性）$= \tilde u(e^{-i\theta}x)$（因实值）。由实控制 $\tilde u \le p$ 及 $p$ 齐次性得 $\le p(x)$。
- **第三步，在 $M$ 上一致**：$x \in M$ 时 $\tilde u(x) = u(x)$，故 $\tilde f(x) = u(x) - iu(ix) = f(x)$。

**关键**：复化的全部工作就是「验证 $\tilde f(ix) = i\tilde f(x)$」与「旋转到实轴后用实控制」——**实版本的延拓能力原封不动地传递到复版本**。

## 5 为什么复版本重要

复 Hahn-Banach 在三个地方至关重要：

- **复 Banach 空间的对偶理论**：$C(X)$（复值连续函数）、$L^p$（复值）的泛函表示都依赖复延拓。
- **谱理论**：复 Banach 代数、算子谱是复分析的对象，谱点分离需要复泛函。
- **复插值理论**：Riesz-Thorin 插值定理（复插值）建立在复 Hahn-Banach 之上。<span class="marginnote">一个常见的误解是「复空间理论只是实空间的『加个 $i$』」——其实不然：<strong>复结构带来全新的现象（如复分析的全纯性、谱的复平面几何），但延拓这步确实可以「实→复」免费升级</strong>。Hahn-Banach 恰好是「免费」的那一类。</span>

## 6 常见误区与反例汇总

**误区一：以为复 Hahn-Banach 的条件与实版相同**。复版本用半范数 $p$ 与「绝对值控制」$|f(x)| \le p(x)$；实版本用次线性泛函与「单侧控制」$f(x) \le p(x)$。两者不等价。

**误区二：忘记「$f(x) = u(x) - iu(ix)$」只在复线性下成立**。这条公式是复线性泛函的结构定理；实线性泛函 $u$ 不一定满足它。验证 $\tilde f$ 复线性时，核心是检查 $\tilde f(ix) = i\tilde f(x)$。

**误区三：把「半范数」当「范数」**。半范数允许 $p(x) = 0$ 而 $x \neq 0$。Hahn-Banach 对半范数成立，但「$p(x) = 0 \Rightarrow x = 0$」不是前提。

**一个关键例子**：$C[0,1]$ 上 $p(f) = \max|f|$（范数），$\delta_{t_0}$ 受 $p$ 控制（$|f(t_0)| \le \max|f|$）。复 Hahn-Banach 保证 $\delta_{t_0}$ 可保控延拓到更大的空间。

**例（为什么需要复版本）**：研究复 Banach 代数（如 $C(X)$ 复值连续函数）时，谱点的分离需要复值泛函；实版本只能给出实值泛函，不够用。谱理论（第九章）离开复 Hahn-Banach 寸步难行。

**核心要点：复版本 = 实版本 + 旋转技巧**。证明里的「旋转到实轴」$|\tilde f(x)| = \tilde f(e^{-i\theta}x)$ 是复化的钥匙。

## 7 例题精讲：复延拓的练习

**练习一：实部决定复泛函**。

- $f(x) = u(x) - iu(ix)$，$u = \operatorname{Re}f$。
- 例：$f(z) = (1+i)z$（$\mathbb{C}$ 上），$u(z) = \operatorname{Re}((1+i)z)$。
- $u(iz) = \operatorname{Re}((1+i)iz) = \operatorname{Re}((-1+i)z)$，验证 $f(z) = u(z) - iu(iz)$。

**练习二：旋转到实轴的技巧**。

- $|\tilde f(x)| = \tilde f(e^{-i\theta}x)$，其中 $\theta = \arg\tilde f(x)$。
- 左边是实数，故 $\tilde f(e^{-i\theta}x) = \tilde u(e^{-i\theta}x)$（实部）。
- 用实控制 $\tilde u \le p$：$|\tilde f(x)| = \tilde u(e^{-i\theta}x) \le p(e^{-i\theta}x) = p(x)$。

**练习三：半范数的齐次性**。

- $p(\alpha x) = |\alpha|p(x)$ 对复 $\alpha$。
- 旋转 $e^{-i\theta}$ 的模为 1，$p(e^{-i\theta}x) = p(x)$。
- 「旋转不改变范数」让实控制直接可用。

**核心要点**：复延拓 = 实延拓 + 旋转技巧；半范数的模齐次性是旋转合法性的来源。

**辨析｜易错点：** 验证 $\tilde f$ 复线性时必须检查 $\tilde f(ix) = i\tilde f(x)$——这是复线性与实线性的分水岭。


## 8 小结

- **复泛函由其实部决定**：$f(x) = u(x) - iu(ix)$；实线性 + 乘 $i$ 相容 = 复线性。
- **半范数**：$p(\alpha x) = |\alpha|p(x)$ 替代次线性泛函；复版本控制条件是 $|\tilde f(x)| \le p(x)$。
- **化归证明**：实化（实 Hahn-Banach 延拓实部）→ 复化（用公式拼回）+ 旋转到实轴验证控制。
- **价值**：复对偶理论、谱理论、复插值都依赖它；「实 → 复」在延拓这一步免费。
- **签名技巧**：$|\tilde f(x)| = \tilde f(e^{-i\theta}x)$——旋转到实轴再用实控制。

在下一节，我们回到赋范空间——**赋范空间上有界线性泛函的保范延拓**：Hahn-Banach 最常用的形态，保证泛函的范数在延拓中不增大。
