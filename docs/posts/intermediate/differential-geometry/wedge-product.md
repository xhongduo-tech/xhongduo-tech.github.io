---
title: 外积
date: 2026-08-07
---

# 外积

<div class="epigraph">
<p>外积是微分形式的乘法：把低阶形式乘成高阶形式，每一次乘法都记住方向的顺序。</p>
<footer>—— 赫尔曼 · 格拉斯曼（Hermann Grassmann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§7.3 ｜ 2026-08-07</p>
</div>

## 为什么从外积开始

微分形式是逐点的交替形式场。现在给它装上「乘法」——**外积（wedge product）**：$k$-形式 $\omega$ 与 $\ell$-形式 $\eta$ 乘成 $(k+\ell)$-形式 $\omega\wedge\eta$。

为什么需要外积？因为微分形式不是孤立个体，它们需要组合：$dx$ 与 $dy$ 要乘出面积元 $dx\wedge dy$，$dx$ 与 $dy\wedge dz$ 要乘出体积元 $dx\wedge dy\wedge dz$。外积就是「把形式并起来、同时记住反对称」的运算——它是微分形式从「逐点张量」变成「代数结构」的关键一步。<span class="marginnote">外积是格拉斯曼（1844）发明的「外代数」中的乘法。它把「体积」概念从「长方形面积」推广成「$k$ 维平行体的带符号体积」——$dx\wedge dy$ 是面积、$dx\wedge dy\wedge dz$ 是体积、更高阶是「高维体积」。外积让「$k$-维体积」成为统一的多线性对象。</span>

## 1 外积的定义（逐点）

**定义（外积）**：设 $\omega$ 是 $k$-形式、$\eta$ 是 $\ell$-形式。它们的外积是 $(k+\ell)$-形式，在每点 $p$ 定义为

$$
(\omega\wedge\eta)_p(v_1,\dots,v_{k+\ell}) = \frac{1}{k!\,\ell!}\sum_{\sigma\in S_{k+\ell}} \operatorname{sgn}(\sigma)\; \omega_p(v_{\sigma(1)},\dots,v_{\sigma(k)})\,\eta_p(v_{\sigma(k+1)},\dots,v_{\sigma(k+\ell)})
$$

（对所有置换求和，带置换符号——「反对称化」两个张量的乘积。）

**重点：外积 = 「张量积 + 反对称化」。** 它是逐点的交替形式的乘法，输出 $k+\ell$ 阶交替形式。坐标下，外积是「分量交替乘积 + 符号」的组合。

## 2 外积的运算律

外积满足三条关键运算律：

1. **双线性**：$(\omega_1+\omega_2)\wedge\eta = \omega_1\wedge\eta + \omega_2\wedge\eta$，$(a\omega)\wedge\eta = a(\omega\wedge\eta)$。
2. **结合律**：$(\omega\wedge\eta)\wedge\theta = \omega\wedge(\eta\wedge\theta)$。
3. **反交换（graded commutativity）**：
   $$
   \omega\wedge\eta = (-1)^{k\ell}\,\eta\wedge\omega
   $$
   其中 $k = \deg\omega$、$\ell = \deg\eta$。

**重点：外积是「带符号交换」的——交换两个形式，出现 $(-1)^{k\ell}$ 因子。** 特别地：

- $k$ 或 $\ell$ 为偶数：$\omega\wedge\eta = \eta\wedge\omega$（交换不变号）。
- $k, \ell$ 都奇数：$\omega\wedge\eta = -\eta\wedge\omega$（交换变号）。
- **$k$ 奇数：$\omega\wedge\omega = 0$**（自身外积为零）。<span class="marginnote">「反交换」是外积与普通乘法的本质区别：普通乘法 $xy=yx$，外积 $dx\wedge dy = -dy\wedge dx$。「$dx\wedge dx = 0$」让外积自动处理「退化」（重复的坐标方向没有体积）——这是微积分里「$dx\,dx$ 没有意义」的代数化解决。</span>

### 记忆口诀

「偶交换不变号，奇交换变号」——判断外积交换是否变号，看两个因子的次数奇偶性。**两个奇数因子交换才变号。**

## 3 公式解析：为什么 $dx\wedge dy = -dy\wedge dx$

用最基础的一步建立外积的手感：

- **第一步，1-形式的反交换**：$\omega, \eta$ 都是 1-形式（$k=\ell=1$，都奇数），由反交换律
  $$
  \omega\wedge\eta = (-1)^{1\cdot 1}\eta\wedge\omega = -\eta\wedge\omega
  $$
  即 $dx\wedge dy = -dy\wedge dx$。
- **第二步，几何含义**：$dx\wedge dy$ 是「带符号面积」：$(dx\wedge dy)(v,w) = \det\begin{pmatrix}dx(v)&dx(w)\\dy(v)&dy(w)\end{pmatrix}$。交换输入（$v\leftrightarrow w$）变号——行列式交换两列变号。**外积的反对称 = 行列式的反对称。**
- **第三步，$dx\wedge dx = 0$**：由 $dx\wedge dx = -dx\wedge dx$ 得 $=0$——「重复方向的体积为零」。<span class="marginnote">微积分里「$dx\,dx$ 不能写」的尴尬，在外积里自动解决：$dx\wedge dx = 0$。而「换元时 $dx\,dy$ 变成 $J\,du\,dv$」的 Jacobi 因子，正是 $dx\wedge dy$ 的反对称性（行列式）的直接体现——你早已在使用外积，只是不知道它的名字。</span>

## 4 例子：三维空间的外积运算

在 $\mathbb{R}^3$ 里做外积运算：

- **1-形式乘 1-形式得 2-形式**：
  $$
  (P\,dx + Q\,dy + R\,dz)\wedge(dx) = -P\,dx\wedge dx + Q\,dy\wedge dx + R\,dz\wedge dx = -Q\,dx\wedge dy - R\,dx\wedge dz
  $$
  （用到 $dy\wedge dx = -dx\wedge dy$、$dx\wedge dx = 0$。）
- **1-形式乘 2-形式得 3-形式**：$dx\wedge(dy\wedge dz) = dx\wedge dy\wedge dz$（体积形式）。
- **3-形式乘任何形式 = 0**：在三维里没有 4-形式（$k>n$ 自动为零）。

**重点：外积逐步「升级」——1 乘 1 得 2，1 乘 2 得 3，3 乘任何都归零。** 维数上限 $n$ 自动截断。

## 5 外积的应用：体积与坐标无关

外积的核心应用是构造**坐标无关的体积元**：

- **面积元**：$dx\wedge dy$ 在换坐标 $(x,y)\to(u,v)$ 下
  $$
  dx\wedge dy = \frac{\partial(x,y)}{\partial(u,v)}\,du\wedge dv
  $$
  Jacobi 行列式自然出现——**外积自动给出换元公式**。
- **体积形式**：$dV = dx\wedge dy\wedge dz$，任意坐标下带 $\sqrt{\det g}$（黎曼体积形式，第八篇）。
- **定向**：$\omega\wedge\omega$ 的符号记录定向——外积自动处理「正反定向」。

**辨析｜易错点：** 别把外积 $dx\wedge dy$ 与普通乘积 $dx\,dy$ 混为一谈。$dx\wedge dy$ 是交替形式（反对称），$dx\,dy$ 是普通乘积（对称）——它们只在「$dx\,dy$ 不存在的严格意义」上有交集。写微分形式时一律用 $\wedge$。<span class="marginnote">工程与物理里，外积无处不在：电磁场 $F = E\wedge dt + B$（$B$ 是磁 2-形式）、流体力学里的涡量 2-形式、辛几何里的辛形式 $dp\wedge dq$。Hamilton 力学的「面积不变性」（Liouville 定理）本质是 $dp\wedge dq$ 在流下的不变性——外积是经典力学的几何语言。</span>

### 例：外积的「面积」读法

把外积看成「带符号面积」，直觉最清晰。设 $v = (v_1, v_2)$、$w = (w_1, w_2)$ 是平面向量，则

$$
(dx\wedge dy)(v, w) = \begin{vmatrix} v_1 & w_1 \\ v_2 & w_2 \end{vmatrix} = v_1 w_2 - v_2 w_1
$$

**这正是 $v, w$ 张成的平行四边形的带符号面积**——交换 $v, w$ 变号（面积反号），$v = w$ 时为零（无面积）。

**重点：外积就是「逐点的带符号体积」——$k$ 个向量张成的 $k$ 维平行体的体积。** 这个读法让外积的一切性质（反交换、自身为零、Jacobi 换元）都有了「体积」直觉：交换顺序体积反号、重复方向无体积、坐标变换缩放体积（乘行列式）。「外积 = 体积的代数」是微分形式理论的灵魂。

### 外积与「叉积」的关系

在 $\mathbb{R}^3$ 里，外积与向量叉积有精确对应：1-形式 $\omega = P\,dx+Q\,dy+R\,dz$ 对应向量 $\mathbf{F} = (P,Q,R)$，则

$$
\omega\wedge(\text{另一个1-形式}) \longleftrightarrow \mathbf{F}\times\mathbf{G}
$$

**叉积的反对称性（$\mathbf{F}\times\mathbf{G} = -\mathbf{G}\times\mathbf{F}$）正是外积的反交换。** 外积是叉积在任意维、任意阶的推广——2-形式（$dx\wedge dy$ 等）对应叉积的结果向量。「外积 = 叉积的维数无关版」——从三维叉积到外代数，一个思想贯穿。这解释了为什么「外积」在几何里如此自然：它就是「体积 + 方向」的代数。

## 6 小结

- **外积** $\wedge$：逐点的交替形式乘法；$k$-形式 $\wedge$ $\ell$-形式 $= (k+\ell)$-形式。
- 运算律：双线性、结合律、**反交换** $\omega\wedge\eta = (-1)^{k\ell}\eta\wedge\omega$。
- $dx\wedge dy = -dy\wedge dx$、$dx\wedge dx = 0$——行列式与退化体积的代数化。
- 换元：$dx\wedge dy = \frac{\partial(x,y)}{\partial(u,v)}du\wedge dv$——Jacobi 自动出现。
- 应用：面积元、体积形式、电磁场、辛几何、Liouville 定理。

在下一节，我们研究微分形式的「求导」：**外微分算子**——把 $k$-形式映成 $(k+1)$-形式，微积分基本定理的高维心脏。
