---
title: 次微分的运算规则：和、复合与逐点极大
date: 2026-08-07
---

# 次微分的运算规则：和、复合与逐点极大

<div class="epigraph">
<p>我们必须知道，我们必将知道。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 凸分析 ｜ Rockafellar《Convex Analysis》第23章 ｜ 2026-08-07</p>
</div>

## 为什么从次微分运算开始

上一节定义了次梯度，但要真正用起来，必须能**计算**复杂函数的次微分。
就像普通微积分有和、积、链式法则，次微分也有一套运算规则——只是每一条都带一个「约束条件」（constraint qualification），
保证等式而非只是包含。
这套规则是「可分解凸规划」在微分层的镜像：
CVXPY 判定凸性靠函数运算规则，而求解器的次梯度、近端步骤则靠次微分运算规则。
<span class="marginnote">机器学习的损失几乎都是「和 + 复合 + 最大」的拼装：$\ell$ 正则 + 数据项之和、$f(Ax+b)$ 型复合、合页损失的逐点最大。
能对它们逐层求次微分，就等于有了求「任意凸损失次梯度」的自动机。
</span>

## 1 和与数乘

**和规则**：设 $f_1, f_2$ 是正常凸函数，
则

$$\partial(f_1 + f_2)(x) \supseteq \partial f_1(x) + \partial f_2(x)$$

等号成立的充分条件是**约束规格**：存在一点 $x_0 \in \operatorname{dom} f_1 \cap \operatorname{dom} f_2$ 使 $f_1, f_2$ 至少一个在该点连续（更常用的是相对内部相交：$\operatorname{ri}(\operatorname{dom} f_1) \cap \operatorname{ri}(\operatorname{dom} f_2) \ne \emptyset$）。<span class="marginnote">「约束规格」听起来吓人，本质是要求「两个函数的有效域不要纠缠得太病态」。对定义在全空间的连续凸函数（机器学习里几乎都是），它自动满足，等号无条件成立——$L1 + L2$ 正则 + 数据项的次梯度就是逐项次梯度之和。</span>

**数乘规则**：对 $\lambda > 0$，$\partial(\lambda f)(x) = \lambda\, \partial f(x)$；对 $\lambda = 0$，$\partial(0 \cdot f)(x) = \{0\}$ 在 $x \in \operatorname{dom} f$ 时成立。负数乘会把凸变凹，规则失效——这再次呼应第10节「权重必须非负」。

**辨析｜易错点：** 包含方向 $\supseteq$ 总是成立（逐项取次梯度再相加，仍是和的次梯度）；**等式需要约束规格**。若两个函数定义域「错开」到病态（如指示函数 $\delta_A + \delta_B$ 且 $A, B$ 相对内部不相交），可能只有包含、没有等式。工程上：**看到 $\partial(f+g) = \partial f + \partial g$，先确认相对内部相交。**

## 2 仿射复合与链式

**仿射复合规则**：设 $f$ 凸，$g(x) = f(Ax + b)$，则

$$\partial g(x) = A^T\, \partial f(Ax + b)$$

这对应普通微积分的链式法则：$g'(x) = A^T f'(Ax+b)$。
<span class="marginnote">证明的核心是把次梯度不等式两边同时「代回」$Ax+b$：$f(y) \ge f(Ax+b) + g_0^T (y - (Ax+b))$ 中令 $y = Ax' + b$ 即得 $g(x') \ge g(x) + (A^T g_0)^T (x' - x)$。
线性变换的伴随 $A^T$ 在次梯度世界里就是链式法则的「转置」。
</span>

**一般的链式法则** $f_2(f_1(x))$ 需要更细的条件（$f_2$ 非降、或 $f_1$ 在关键点不退化），
规则与第10节凸复合的方向约束配对：**外层凸非降、内层凸 ⇒ 复合凸**，
此时次微分可以逐层传递。
对非单调复合，次微分会出现「套着集合的集合」，
工程上几乎不用。

## 3 逐点极大

**逐点极大规则**：设 $f(x) = \max_{i=1,\dots,m} f_i(x)$，
各 $f_i$ 凸，则

$$\partial f(x) = \operatorname{conv}\left\{ \bigcup_{i \in I(x)} \partial f_i(x) \right\}, \qquad I(x) = \{ i \mid f_i(x) = f(x) \}$$

即在 $x$ 处**取到最大值的那些指标**的次微分取凸包。<span class="marginnote">这个公式太漂亮了：合页损失 $f(x) = \max\{0, 1 - y\, w^T x\}$ 的次梯度，在「未激活」（$1 - y w^T x \lt  0$）时是 $\{0\}$，在「激活」（$> 0$）时是 $\{-y x\}$，在「边界」（$= 0$）时是 $\{-\theta y x : \theta \in [0,1]\}$ 的凸包区间——SVM 对偶与次梯度实现全赖于此。</span>

**辨析｜易错点：** 凸包取在**次微分集合**上，不是取在次梯度的**端点上**再单独处理——必须把 $I(x)$ 中每个活跃函数的整个次微分集合拿进来，再做凸包。若某个活跃函数在 $x$ 处有「一整段」次微分（如 $|x|$ 在 $0$），凸包会把这段也包进来。

## 4 公式解析：逐点极大的次微分

用 $f(x) = \max\{f_1(x), f_2(x)\}$ 两个函数演示，假设 $x$ 处两者都取到最大（$I(x) = \{1, 2\}$）。

- **第一步，一个方向**：任取 $g = \theta g_1 + (1-\theta) g_2$，其中 $g_i \in \partial f_i(x)$。对任意 $y$，$f_i(y) \ge f_i(x) + g_i^T(y-x) = f(x) + g_i^T(y-x)$。
- **第二步，凸组合放缩**：两边取权重 $\theta$ 与 $1-\theta$ 相加，得 $f(y) \ge f(x) + g^T(y-x)$——$g$ 是 $f$ 的次梯度。所以凸包 ⊆ 次微分。
- **第三步，反方向**：取 $g \in \partial f(x)$，沿方向 $d$ 用方向导数公式 $f'(x;d) = \max_{i\in I(x)} f_i'(x;d) = \max_i \sup_{g_i \in \partial f_i(x)} g_i^T d$，把「最大值的方向导数」与「次梯度的上确界」对起来，反推出 $g$ 属于凸包。
- **第四步，多函数**：$m > 2$ 时同样的论证对任意有限个成立；无穷族的情形（$\sup$）则要取弱闭凸包，超出本节范围。

**这条规则把「不可微的最大运算」翻译成了「可计算的凸包」**——它让合页损失、分片线性目标这类「处处有尖」的函数的次梯度变得触手可及。

## 5 次微分计算的实战演练

把运算规则用到具体组合上，一次看清每条规则怎么配合：

**练习 1：L1 正则最小二乘。** $F(x) = \|Ax - b\|_2^2 + \lambda \|x\|_1$。第一项可微，$\partial(\|Ax-b\|^2)(x) = \{2A^T(Ax - b)\}$；第二项用 $\ell_1$ 的次微分（各分量 $\partial |x_i|$）。由和规则（$F$ 定义域全空间，约束规格自动满足）：

$$\partial F(x) = 2A^T(Ax - b) + \lambda\, \partial\|x\|_1$$

把 $\partial\|x\|_1$ 的 $[-1,1]$ 分量写进去，
就得到 LASSO 的次梯度条件——**稀疏解的「悬空分量」正是从这里来的**。

**练习 2：合页损失 + L2（SVM 原问题）。**
$F(w) = \frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i w^T x_i)$。
逐点最大规则的结合：先对每个样本算 $\partial \max(0, 1 - y_i w^T x_i)$（仿射复合 + 逐点最大），
再求和（非负加权和）。
结果是一个「只依赖活跃样本」的次梯度——这正是 SVM 次梯度法「只更新支撑向量」的算法基础。
<span class="marginnote">SVM 的次梯度更新可以写成 $w \leftarrow w - \alpha(w - C \sum_{i \in \text{活跃}} y_i x_i)$——只有合页损失非零的样本（在间隔内或错分）贡献梯度。
这就是「稀疏更新」：大部分样本的梯度为 $0$。
</span>

**练习 3：逐点最大求次微分。**
$f(x) = \max\{x_1^2, x_2^2\}$。
在 $x = (1, 1)$ 处，两个函数都取到 $1$（$I(x) = \{1, 2\}$），
$\partial f = \operatorname{conv}\{(2, 0), (0, 2)\}$——连接两点的线段。
在 $x = (2, 1)$ 处只有第一个活跃，
$\partial f = \{(4, 0)\}$——单点。
**活跃指标集合 $I(x)$ 决定次微分是「单点」还是「凸多面体」。**

**辨析｜易错点：** 约束规格（constraint qualification）不是装饰。
练习 1 的等号成立是因为定义域是全空间；
若换成 $\delta_A + \delta_B$ 且 $A, B$ 相对内部不相交（如两条交叉线段），
$\partial(f+g)$ 可能严格大于 $\partial f + \partial g$。
<span class="marginnote">病态例子：$A = \{0\}$（单点），$B = \mathbb{R}$，$\delta_A + \delta_B = \delta_{\{0\}}$，$\partial(\delta_A + \delta_B)(0) = \mathbb{R}$（法锥是全空间），而 $\partial \delta_A(0) + \partial \delta_B(0) = \mathbb{R} + \{0\} = \mathbb{R}$——竟然还相等；
真正失衡的例子需要相对内部错开。
<strong>实用建议：先检查定义域「开」或「相对内部相交」，再放心用等号。
</strong></span>

**工程速查。** 常用次微分：$|x|$ 的 $[-1,1]$、$\|x\|_1$ 的分量符号、$\delta_C$ 的法锥 $N_C$、$\max_i f_i$ 的活跃凸包。
**遇到复杂函数，先拆块、再逐层套规则，
最后核对约束规格——三步走完，次梯度到手。**

## 6 小结

- **和规则**：$\partial(f_1+f_2)(x) \supseteq \partial f_1(x) + \partial f_2(x)$，等号需约束规格（相对内部相交）。
- **仿射复合**：$\partial(f(Ax+b))(x) = A^T \partial f(Ax+b)$，链式法则的转置版本。
- **逐点极大**：$\partial(\max_i f_i)(x) = \operatorname{conv}\bigcup_{i \in I(x)} \partial f_i(x)$，凸包取在活跃指标上。
- 数乘规则只对**非负**权重成立，负权翻转凸性。
- 每条运算规则的等式都有约束规格，工程使用前先确认定义域「规整」。

在下一节，我们把次梯度用到刀口上——**极小化问题的次梯度最优性条件**，
证明「$0 \in \partial f(x)$