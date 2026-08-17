---
title: 共轭函数与对偶（Fenchel 共轭）
date: 2026-08-07
---

# 共轭函数与对偶（Fenchel 共轭）

<div class="epigraph">
<p>在数学里，你并不是理解了事物，你只是习惯了它们。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 凸分析 ｜ Rockafellar《Convex Analysis》第12章；Boyd《Convex Optimization》§3.3 ｜ 2026-08-07</p>
</div>

## 为什么从共轭函数开始

对偶性是凸分析献给优化最深的一件礼物，而它的原料就是**共轭函数（conjugate function）**。给定一个凸函数 $f$，Fenchel 共轭把它变换成另一个函数 $f^*$——$f^*$ 不记录「在哪个点函数值是多少」，而是记录「在哪个斜率方向，$f$ 离那条直线有多远」。<span class="marginnote">可以这样理解共轭：$f$ 用「点–值」说话，$f^*$ 用「斜率–截距」说话。<strong>共轭就是把函数从点坐标换到切线坐标</strong>——凸函数被它的一簇支撑线完全决定，所以两种说法等值。几何上 $f^*$ 的 $-1$ 倍……直接想复杂了，就从 Fenchel–Young 不等式入手最顺。</span>这一篇是第4篇共轭与对偶的综述，把 Fenchel 共轭、Fenchel–Young、双共轭与 Lagrange 对偶装进同一个框架，为第5篇的 KKT 与第8篇的对偶算法做铺垫。

共轭也是统计力学与信息论的常客：指数族分布、KL 散度、Legendre 变换（经典力学的哈密顿量 = 拉格朗日量的 Legendre 变换）都是共轭的近亲。理解了 $f^*$，你等于同时拿到了物理学家与统计学家手里的同一把钥匙。

## 1 Fenchel 共轭：函数的下方「对偶坐标」

**共轭函数（conjugate function）**：对任意函数 $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$，其 Fenchel 共轭是

$$f^*(y) = \sup_{x \in \operatorname{dom} f} \big( \langle x, y \rangle - f(x) \big)$$

$f^*$ 取的是「直线 $\langle x, y \rangle$ 减去 $f(x)$ 的盈余上确界」。$f$ 不必凸也能定义 $f^*$（只要上确界有意义），但**只有凸闭函数 $f^*$ 才有资格「对偶回去」**。<span class="marginnote">$f^*$ 的名字常令人困惑：它不是「共轭复数」那种共轭。更形象的翻译是「上镜函数（support function 的推广）」——$\operatorname{dom} f^*$ 是「使直线 $\langle x,y\rangle$ 与 $f$ 的上图有支撑关系的斜率 $y$ 的集合」。</span>

**重点：** 共轭自动是凸函数——一族仿射函数的上确界永远是凸的（无论 $f$ 是否凸）。这个「凸性免费」的观察，是共轭在优化里如此好用的第一原因。

**计算示例**：$f(x) = |x|$，则 $f^*(y) = \sup_x (xy - |x|)$。$|y| \le 1$ 时上确界为 0（取 $x=0$），$|y| > 1$ 时无界发散到 $+\infty$——所以 $f^*(y) = \delta_{\{|y| \le 1\}}$，一个指示函数。**范数的共轭是单位球指示函数**；$\ell_1$ 的共轭是 $\ell_\infty$ 球指示函数，$\ell_2$ 的共轭是 $\ell_2$ 球指示函数。这条「对偶范数」规律贯穿稀疏优化。

把最常用的几组共轭列成对照表，计算时直接查：

| 函数 $f(x)$ | 共轭 $f^*(y)$ | 备注 |
| --- | --- | --- |
| $|x|$ | $\delta_{\{|y| \le 1\}}$ | 对偶范数规律 |
| $\|x\|_1$ | $\delta_{\{\|y\|_\infty \le 1\}}$ | 稀疏优化 |
| $\|x\|_2$ | $\delta_{\{\|y\|_2 \le 1\}}$ | 自对偶范数 |
| $\tfrac12\|x\|_2^2$ | $\tfrac12\|y\|_2^2$ | 自对偶 |
| $e^x$ | $y \ln y - y$（$y>0$） | 指数与熵相连 |
| $-\log x$（$x>0$） | $-1 - \log(-y)$（$y<0$） | 凹函数的共轭为无穷 |

这张表是「计算共轭」的速查手册，前四行都来自「范数 ↔ 对偶范数指示函数」这条统一规律。

**把 $\ell_1$ 的共轭推导到底**：$f(x) = \|x\|_1 = \sum_i |x_i|$。代入定义，$\langle x, y\rangle - f(x) = \sum_i (x_i y_i - |x_i|)$ 逐分量可分离，于是

$$f^*(y) = \sum_i \sup_{x_i \in \mathbb{R}} (x_i y_i - |x_i|)$$

对每个分量单独看：若 $|y_i| > 1$，取 $x_i \to +\infty$（当 $y_i > 0$）使 $x_i y_i - |x_i| = x_i(y_i - 1) \to +\infty$，上确界发散；若 $|y_i| \le 1$，由 $x_i y_i \le |x_i|$ 得上确界为 0（取 $x_i = 0$）。因此

$$f^*(y) = \begin{cases} 0 & \|y\|_\infty \le 1 \\ +\infty & \text{otherwise} \end{cases} = \delta_{\{\|y\|_\infty \le 1\}}$$

**这就是「范数的共轭是对偶范数单位球指示函数」的逐分量验证**——$\ell_1$ 的对偶范数是 $\ell_\infty$，两个方向都被这条推导同时说明。以后遇到范数的共轭，不必重算，直接查对偶范数。

## 2 Fenchel–Young 不等式与二次共轭

**Fenchel–Young 不等式**：对任意 $x, y$，

$$f(x) + f^*(y) \ge \langle x, y \rangle$$

由共轭定义直接给出：$f^*(y) \ge \langle x,y \rangle - f(x)$ 对每个 $x$ 成立，移项即得。<span class="marginnote">这是凸分析里最廉价也最常用的一把不等式：它不要求凸性、不要求光滑、不要求任何条件，只要共轭定义有意义。第4篇的弱对偶 $d^* \le p^*$ 就是 Fenchel–Young 相加再取极值的直接结果——弱对偶本质上是「两把 Fenchel–Young」。</span>

**验证一把 Fenchel–Young**：取 $f(x) = \tfrac12 x^2$，则 $f^*(y) = \tfrac12 y^2$，不等式写作 $\tfrac12 x^2 + \tfrac12 y^2 \ge xy$——正是「均值 ≥ 乘积」的经典形式 $xy \le \tfrac12(x^2 + y^2)$。等号在 $x = y$ 时取到。**Fenchel–Young 的等号条件 $y \in \partial f(x)$**，是第5篇次梯度定义的另一面：次梯度就是「让 Fenchel–Young 取等」的斜率。

**二次共轭（biconjugate）**：$f^{**} = (f^*)^*$。一般只有 $f^{**} \le f$；当 $f$ 是**正常闭凸函数**（正常、凸、下半连续）时，

$$f^{**} = f$$

这就是**Fenchel–Moreau 定理**：闭凸函数等于它二次共轭，即 $f$ 被它的一族支撑线完全重建。它回答了对偶性的第一个核心问题——「共轭转一圈，信息不丢失，当且仅当 $f$ 闭凸」。<span class="marginnote">Fenchel–Moreau 是对偶理论的总开关：<strong>闭凸函数的二次共轭等于自身</strong>，于是「原始函数」与「对偶函数」互为镜像，极小化 $f$ 可以等价地改写成对 $f^*$ 做某个运算——这就是「对偶问题」的诞生。若 $f$ 不是闭凸，二次共轭退化为 $f$ 的闭凸包，共轭自动「凸化+取闭包」，信息被有损地还原成闭凸壳。</span>

**重点：** 共轭把「取闭凸化」与「取对偶」合成一步：$f^{**}$ 总是 $f$ 的闭凸包。对非凸 $f$，$f^{**}$ 是它最接近的凸松弛——这解释了为什么「共轭=凸松弛」这一思路贯穿整数规划的 Lagrangian 松弛。

## 3 共轭与对偶：Lagrange 对偶的共轭视角

**Lagrange 对偶**的第4篇构造，用共轭语言可以重写得更干净。考虑

$$\min_x f_0(x) \quad \text{s.t.} \quad f_i(x) \le 0,\ i = 1,\dots,m$$

Lagrange 函数 $L(x, \lambda) = f_0(x) + \sum_i \lambda_i f_i(x)$（$\lambda \ge 0$）。对偶函数是

$$g(\lambda) = \inf_x L(x, \lambda) = -f_0^*(-A^T \lambda)$$

其中 $A$ 是把 $f_1,\dots,f_m$ 打包的算子，$f_0^*$ 是 $f_0$ 的共轭。<span class="marginnote">这条式子的要点：<strong>对偶函数就是「共轭在某个斜率处的值再取负」</strong>——共轭在这里把「对 $x$ 的极小化」变成「对某个斜率参数的函数求值」。这正是第4篇 Fenchel 对偶定理（$f + g \circ A$ 的对偶是 $f^* \circ A^T + g^*$）的特例。</span>对偶问题的目标变成 $g(\lambda)$（一个凹函数，共轭与负号的组合），对偶变量 $\lambda \ge 0$ 是约束的影子价格。

**对偶间隙的几何**：$p^* = \inf_x f_0(x)$、$d^* = \sup_{\lambda \ge 0} g(\lambda)$。由 Fenchel–Young 恒有 $d^* \le p^*$（弱对偶）；在 Slater 条件（存在严格可行点）下强对偶 $d^* = p^*$。用共轭看：**强对偶 ⟺ $f_0$ 与其约束拉格朗日化的「下包络」重合**——这仍是 Fenchel–Moreau「信息不丢失」在约束情形下的版本。

**算一个共轭视角的 Lagrange 对偶**：$\min_x \tfrac12\|x\|_2^2$ s.t. $Ax = b$（投影到仿射流形）。Lagrange $L(x, \nu) = \tfrac12\|x\|^2 + \nu^T(Ax - b)$。对偶函数

$$g(\nu) = \inf_x \tfrac12\|x\|^2 + \nu^TAx - \nu^Tb = -\tfrac12\|A^T\nu\|^2 - \nu^Tb$$

推导只有两行：驻点 $\nabla_x = x + A^T\nu = 0$ 得 $x = -A^T\nu$，代入即得；用共轭公式 $g(\nu) = -f_0^*(-A^T\nu) - \nu^Tb$ 与 $f_0^*(y) = \tfrac12\|y\|^2$ 也得到同一结果——两条路殊途同归，正是共轭「把极小化变成函数求值」的现场演示。对偶问题 $\max_\nu g(\nu)$ 是无约束凹极大化，解出 $\nu^*$ 后原解 $x^* = -A^T\nu^*$。**这个例子的教学意义**：共轭把「带约束的原始问题」翻译成「无约束的对偶问题」，两个问题在同一张图上互为镜像。

**辨析｜易错点：** 共轭的符号与定义域要盯紧。$f^*$ 的自变量是「斜率」$y$，不是点；$\langle x,y\rangle$ 用的是内积不是逐点乘积（对一般内积空间要换成 $\langle \cdot, \cdot \rangle$）。计算共轭时最常见的错误是把 $f$ 的 $\operatorname{dom}$ 与 $f^*$ 的 $\operatorname{dom}$ 弄混——$f^*$ 的有限域是「$f$ 的支撑斜率的集合」，尺寸维度与 $x$ 相同但含义不同。

## 4 公式解析：共轭的几何解释与计算

把共轭拆成「支撑直线」的语言，计算就变得机械：

$$f^*(y) = \sup_x \big( \langle x, y \rangle - f(x) \big)$$

- **第一步，固定斜率 $y$**：画一族平行直线 $l(x) = \langle x, y \rangle - c$（$c$ 是截距）。每个 $c$ 对应一条斜率为 $y$ 的直线。
- **第二步，找「托住」的截距**：$\sup_x (\langle x,y\rangle - f(x))$ 就是「使直线 $\langle x,y\rangle - c$ 整体落在 $f(x)$ 下方」的最大截距 $c$ 的相反数再取……整理：$f^*(y) = \sup_x (\langle x,y\rangle - f(x))$，这正是「斜率为 $y$ 的支撑线下方的最大截距」。
- **第三步，光滑情形**：若 $f$ 可微且上确界在内部取到，驻点条件 $\nabla f(x) = y$，此时 $f^*(y) = \langle x, y \rangle - f(x)$，$x$ 是满足 $\nabla f(x) = y$ 的点——**共轭在光滑凸函数上是 Legendre 变换**。
- **第四步，算一个标准例**：$f(x) = \frac{1}{2} x^2$。$\nabla f(x) = x = y$，于是 $f^*(y) = \frac{1}{2} y^2$——二次函数自对偶（up to 结构）。$f(x) = e^x$：$\nabla f = e^x = y$，$x = \ln y$，$f^*(y) = y\ln y - y$（$y > 0$）——指数与熵由此相连。

**这条链路的要点**：<span class="marginnote">光滑凸函数的共轭 = Legendre 变换 = 在「斜率坐标」下重写函数。遇到「计算 $f^*$」的题，先试驻点条件，再回退到定义做上确界，最后用 Fenchel–Young 验证符号——三步循环，几乎不会有错。</span>共轭不是抽象魔法，而是「切线的对偶坐标」。

**共轭的一个深度应用：KL 散度的变分形式**。负熵的共轭是 $\log$ 配分函数，由此信息论里的变分下界（ELBO）本质上是一次共轭运算——「对偶坐标」的观点把统计推断翻译成了优化。这一点在「从极限到大模型」的贝叶斯模型与概率论课程里会再次相遇，届时你会认出这把钥匙。

### 术语速查：共轭与对偶的名词对照

| 术语 | 一句话定义 | 出处 |
| --- | --- | --- |
| Fenchel 共轭 | $f^*(y) = \sup_x(\langle x,y\rangle - f(x))$，斜率坐标下的函数重写 | 本篇 |
| 对偶范数 | $\|y\|_* = \sup_{\|x\| \le 1} \langle x,y\rangle$，范数的共轭是指示函数 | 本篇 |
| Fenchel–Young | $f(x) + f^*(y) \ge \langle x, y \rangle$，等号当 $y \in \partial f(x)$ | 本篇 |
| 二次共轭 | $f^{**} = (f^*)^*$，闭凸时等于 $f$，一般等于闭凸包 | 本篇 |
| Fenchel–Moreau | $f$ 正常闭凸 ⟹ $f^{**} = f$；否则 $f^{**} = \operatorname{cl}\operatorname{conv} f$ | 本篇 |
| Legendre 变换 | 光滑凸函数的共轭，「点坐标」到「斜率坐标」 | 本篇 |
| Lagrange 对偶函数 | $g(\lambda) = -f_0^*(-A^T\lambda)$，共轭在斜率处的取值 | 本篇 / 第4篇 |
| 弱对偶 / 强对偶 | $d^* \le p^*$ 恒成立；Slater 下 $d^* = p^*$ | 第4篇 |

## 5 小结

- **Fenchel 共轭**：$f^*(y) = \sup_x (\langle x,y\rangle - f(x))$——函数在「斜率坐标」下的重写，永远凸。
- **Fenchel–Young**：$f(x) + f^*(y) \ge \langle x, y \rangle$，无条件成立，弱对偶由此而来；等号条件 $y \in \partial f(x)$。
- **Fenchel–Moreau**：$f$ 闭凸 ⟹ $f^{**} = f$；一般 $f^{**} = \operatorname{cl}\operatorname{conv} f$——共轭 = 闭凸化。
- **共轭 ↔ 对偶**：Lagrange 对偶函数 $g(\lambda) = -f_0^*(-A^T\lambda)$；对偶间隙为零 ⇔ 强对偶（Slater）。
- 范数的共轭是对偶范数单位球的指示函数；光滑凸函数的共轭是 Legendre 变换。
- 共轭是统计变分下界、经典力学哈密顿量、信息论熵的语言，远不止优化一隅。

在下一节，我们将让对偶性从「函数之间」落到「集合之间」——**次微分**：次梯度如何充当「不可微点的导数」，Moreau–Rockafellar 理论又如何把运算规则写进次微分。
