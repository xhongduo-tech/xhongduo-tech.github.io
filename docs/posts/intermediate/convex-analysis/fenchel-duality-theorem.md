---
title: Fenchel 对偶定理
date: 2026-08-07
---

# Fenchel 对偶定理

<div class="epigraph">
<p>在一粒沙中看见世界，在一朵野花中看见天堂。</p>
<footer>—— 威廉 · 布莱克（William Blake）</footer>
</div>

<div class="article-byline">
<p>第二级 · 凸分析 ｜ Rockafellar《Convex Analysis》第31章 ｜ 2026-08-07</p>
</div>

## 为什么从 Fenchel 对偶开始

Lagrange 对偶从约束出发；
**Fenchel 对偶**则从**函数结构**出发：
当一个目标天然是「两个凸函数之和」$f(x) + g(Ax)$ 时（如「损失 + 正则」、LASSO 的 $\|Ax - b\|^2 + \lambda \|x\|_1$），
对偶问题可以直接用共轭函数与伴随算子 $A^T$ 写出来。
它揭示了对偶性的代数本质——**求对偶就是求共轭 + 转置**——并把凸分析第4篇的所有工具（共轭、双共轭、Fenchel–Young）收进一条定理。
<span class="marginnote">Fenchel 对偶是「对偶 = 共轭」哲学的最纯粹体现：$f + g \circ A$ 的对偶是 $f^* \circ A^T + g^*$。
所有凸优化（LP、QP、SDP、Lasso）的对偶都能写成这种形式——理解了它，
就理解了「对偶问题为什么长那样」。
</span>

## 1 原始问题形式

**Fenchel 对偶的原始问题**：

$$\min_x \big( f(x) + g(Ax) \big)$$

其中 $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$、$g: \mathbb{R}^m \to \mathbb{R} \cup \{+\infty\}$ 是正常凸函数，$A \in \mathbb{R}^{m \times n}$ 是线性算子。<span class="marginnote">这个形式包罗万象：取 $g = \delta_{\{b\}}$（点约束），得到 $\min f(x)$ 满足 $Ax = b$；取 $g = \delta_C$（集合约束），得到带约束问题；取 $f$ 为损失、$g = \lambda \|\cdot\|_1$，得到正则化问题。<strong>「两个函数 + 一个线性算子」是凸优化的通用语法</strong>。</span>

原问题最优值记 $p^* = \inf_x [f(x) + g(Ax)]$。它与 Lagrange 对偶等价：令 $g = \delta_{\{(u,v)\colon u \le 0, v = 0\}}$ 或引入约束即可互推。

## 2 Fenchel 对偶问题

**Fenchel 对偶问题**：

$$\max_{\nu \in \mathbb{R}^m} \big( -f^*(A^T \nu) - g^*(-\nu) \big)$$

对偶最优值记 $d^*$。
<span class="marginnote">记忆口诀：
<strong>对偶 = 原函数的共轭，
线性算子转置（$A \mapsto A^T$），
第二个变量变号（$g^*(-\nu)$）</strong>。
它完全由 $f^*$、$g^*$ 与 $A^T$ 决定——不需要任何约束清单，
这正是 Fenchel 对偶「纯代数」的优雅之处。
</span>

**弱对偶（Fenchel–Young 直接给出）**：
对任意 $x, \nu$，

$$f(x) + g(Ax) \ge \langle x, A^T \nu \rangle - f^*(A^T\nu) + \langle Ax, -\nu \rangle - g^*(-\nu) = -f^*(A^T\nu) - g^*(-\nu)$$

中间两项 $\langle x, A^T\nu \rangle$ 与 $\langle Ax, -\nu\rangle$ 相互抵消（因为 $\langle x, A^T\nu\rangle = \langle Ax, \nu\rangle$），所以原始值 ≥ 对偶值，$d^* \le p^*$。

## 3 强对偶与约束规格

**Fenchel 对偶定理**：设 $f, g$ 正常凸函数，$A$ 线性算子。若

$$0 \in \operatorname{ri}\big(\operatorname{dom} g - A\, \operatorname{dom} f\big)$$

则强对偶成立：$d^* = p^*$，且对偶最优解可达到（存在 $\nu^*$ 取到 $d^*$）。
<span class="marginnote">这个约束规格的本质与 Slater 条件同构：它要求 $\operatorname{dom} g$ 与 $A \operatorname{dom} f$ 的差包含 $0$ 于相对内部——即两个定义域「在合适的位置相遇」。
翻译成 Lagrange 语言正是 Slater；
翻译成多面体语言正是 LP 强对偶的非退化条件。
</span>

**重点：** 强对偶条件只涉及**相对内部**——$f, g$ 的定义域不必相交，
只要它们的像「足够贴近」即可。
这比逐个约束验证 Slater 更全局，
也是 Fenchel 对偶适合做**结构分析**的原因。

**例子（LP 的 Fenchel 对偶）**：
$\min c^T x$ 满足 $Ax = b$、$x \ge 0$ 可写成 $f = \delta_{\{x \ge 0\}} + c^T x$、$g = \delta_{\{b\}}$。
算共轭：$f^*(y) = \delta_{\{y \le c\}}$（$x \ge 0$ 的对偶约束）、$g^*(\nu) = b^T \nu$。
对偶问题是 $\max_\nu b^T \nu$ 满足 $A^T \nu \le c$——正是教科书里的 LP 对偶。
<span class="marginnote"><strong>LP 对偶不是另一套理论，而是 Fenchel 对偶在指示函数下的特例</strong>。
看到「$A^T\nu \le c$」对偶约束的来源了吗？
它来自 $\operatorname{dom} f^* = \{y \mid y - c \le 0\}$——原始的非负约束 $x \ge 0$ 在对偶里变成了系数上界约束。
</span>

## 4 公式解析：从 Fenchel–Young 推对偶

从原始问题到对偶问题的「推导」其实只有两步：

- **第一步，Fenchel–Young 各自放缩**：对任意 $x, \nu$，$f(x) \ge \langle x, A^T\nu \rangle - f^*(A^T\nu)$（取 $y = A^T\nu$），且 $g(Ax) \ge \langle Ax, -\nu \rangle - g^*(-\nu)$（取 $y = -\nu$）。
- **第二步，相加**：$f(x) + g(Ax) \ge \langle x, A^T\nu\rangle + \langle Ax, -\nu\rangle - f^*(A^T\nu) - g^*(-\nu)$。
- **第三步，伴随抵消**：$\langle x, A^T\nu\rangle = \langle Ax, \nu\rangle = -\langle Ax, -\nu\rangle$，两项正好抵消，得 $f(x) + g(Ax) \ge -f^*(A^T\nu) - g^*(-\nu)$。
- **第四步，两端各取极值**：左边 $\inf_x$、右边 $\sup_\nu$，得 $p^* \ge d^*$（弱对偶）；在约束规格下反向也成立（强对偶），且 $\nu^*$ 存在。

**这条推导的优雅在于：对偶问题「自己长出来」**——不需要猜，
只需对两个函数分别做 Fenchel–Young，
让中间项靠伴随运算消掉。

## 5 Fenchel 对偶的应用实例

把 Fenchel 对偶用到两个具体问题上，
看它对偶问题如何「自己长出来」。

**例 1：LASSO。**
原始问题 $\min_x \frac{1}{2}\|Ax - b\|_2^2 + \lambda \|x\|_1$。
取 $f(x) = \frac{1}{2}\|Ax - b\|_2^2$（但 Fenchel 形式要写成 $f_1(x) + g(Ax)$……更标准的拆法：
令 $f(x) = \lambda\|x\|_1$，
$g(Ax) = \frac{1}{2}\|Ax - b\|^2$）。
实际上 LASSO 最常见的 Fenchel 对偶是：

$$\min_x \frac{1}{2}\|x\|^2 + \lambda \|x\|_1 \quad\text{的}\quad \max_{\|z\|_\infty \le \lambda} -\frac{1}{2}\|z - b\|^2$$

把原始损失与 $\ell_1$ 正则的角色交换，对偶问题变成一个**带箱约束的光滑二次问题**——比原始的非光滑问题更好解。<span class="marginnote">这正是「对偶比原始好解」的典型：原始有 $\|x\|_1$（不可微），对偶只有 $\|z\|_\infty \le \lambda$（一个简单箱约束），目标光滑二次。近端/对偶算法（如 FISTA 的对偶形式、ADMM）常选对偶侧求解，就是看中对偶的结构更干净。</span>

**例 2：约束最小化。** $\min f(x)$ 满足 $Ax = b$。写成 $f(x) + \delta_{\{b\}}(Ax)$（$g = \delta_{\{b\}}$）。算 $g^*(\nu) = b^T \nu$，Fenchel 对偶为

$$\max_\nu \; -f^*(-A^T\nu) - b^T \nu$$

这正是 Lagrange 对偶问题——**Fenchel 对偶与 Lagrange 对偶在「约束写成指示函数」时完全重合**。
两种对偶不是两套理论，而是同一枚硬币的两面：
Lagrange 从约束写，Fenchel 从函数写。

**为什么伴随算子 $A^T$ 总在对偶里出现。**
因为在 $\langle x, A^T\nu\rangle = \langle Ax, \nu\rangle$ 这一步（伴随运算），
内积结构把「算子」转置到对偶变量上。
**对偶问题里的 $A^T$ 不是「记号巧合」，
而是内积的伴随**——这也是为什么对偶问题总比原问题「维度翻转」：
原问题在 $x$ 空间（$\mathbb{R}^n$），
对偶在 $\nu$ 空间（$\mathbb{R}^m$）。
<span class="marginnote">「$A$ 变 $A^T$」在对偶理论里无处不在：LP 对偶的约束从 $Ax \le b$ 变 $A^T\nu \ge c$，Fenchel 对偶从 $f + g \circ A$ 变 $f^* \circ A^T + g^*$。
记住这条规律，对偶问题基本不用「推导」，
直接「写」出来。
</span>

**辨析｜易错点：** Fenchel 对偶的第二个变量带负号（$g^*(-\nu)$），
且 $f$ 与 $g$ 的角色不对称——$A$ 只作用在 $g$ 上，
$A^T$ 只作用在 $f^*$ 上。
**把 $f$ 与 $g$ 调换、或把 $\nu$ 的符号写错，
会得到错误的对偶。**
写完后用「约束最小化」的例子验证一遍符号。

## 6 小结

- **原始形式**：$\min f(x) + g(Ax)$，覆盖 LP、QP、Lasso、约束问题。
- **Fenchel 对偶**：$\max_\nu \big(-f^*(A^T\nu) - g^*(-\nu)\big)$——**共轭 + 转置 + 变号**。
- 弱对偶由 **Fenchel–Young 直接相加**给出，中间项靠 $\langle x, A^T\nu\rangle$ 与 $\langle Ax, -\nu\rangle$ 抵消。
- **强对偶**：$0 \in \operatorname{ri}(\operatorname{dom} g - A \operatorname{dom} f)$ 时成立，等价于 Slater 条件的全局形态。
- LP 对偶是 Fenchel 对偶在指示函数下的特例——「$A^T\nu \le c$」来自 $\operatorname{dom} f^*$