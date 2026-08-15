---
title: 重积分与累次积分：Fubini 定理与 Tonelli 定理
date: 2026-08-07
---

# 重积分与累次积分：Fubini 定理与 Tonelli 定理

<div class="epigraph">
<p>计算高维体积，不必同时看所有方向——先沿一个方向切片，再沿另一个方向累积，答案不差分毫。</p>
<footer>—— 圭多 · 富比尼（Guido Fubini）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》§5.9 ｜ 2026-08-07</p>
</div>

## 为什么从 Fubini 定理开始

多变量积分的第一问是：**重积分 $\iint f$ 能不能写成累次积分 $\int\left(\int f\,dy\right)dx$？** 直观上当然可以——把二维面积「先竖切再横切」。但在 Lebesgue 框架里，这个「显然」需要证明，且**需要条件**：非负（Tonelli）或可积（Fubini）。没有条件时，累次积分可能不等于重积分，甚至两个累次积分彼此不等。

Fubini 与 Tonelli 是重积分理论的**宪法**：它使「先积 $y$ 再积 $x$」与「先积 $x$ 再积 $y$」合法化，是概率论中「联合分布 → 边际分布」、统计中「边缘化」、以及一切多变量计算的根基。学懂它，就掌握了高维积分全部实践的一半。<span class="marginnote">Tonelli 定理（非负可测函数）是「无条件版」：非负函数的重积分 = 任一累次积分（允许 $+\infty$）。Fubini 定理（可积函数）是「结论版」：先证非负，再对正负部用 Tonelli。<strong>「先 Tonelli 后 Fubini」是标准路径</strong>——非负性是无限交换的安全阀。</span>

## 1 记号与截面（回顾）

设 $f(x,y)$ 定义在 $\mathbb{R}^{p+q}=\mathbb{R}^p\times\mathbb{R}^q$ 上。对固定的 $x\in\mathbb{R}^p$，$y\mapsto f(x,y)$ 是 $f$ 沿 $y$ 方向的**截面函数**。前面《乘积空间的测度》一节已证：$f$ 可测时，对 a.e. $x$，截面函数 $y\mapsto f(x,y)$ 可测（可测函数的截面可测）。

**重积分**：$\int_{\mathbb{R}^{p+q}}f\,dm_{p+q}$（在乘积空间上的一次积分）。
**累次积分**：$\int_{\mathbb{R}^p}\left(\int_{\mathbb{R}^q}f(x,y)\,dm_q(y)\right)dm_p(x)$（先沿 $y$ 积，再把结果沿 $x$ 积）。

**重点：重积分与累次积分的差别是「一次积」vs「两次积」。** 重积分对 $(x,y)$ 整体求体积；累次积分先求「竖截面面积函数」$x\mapsto\int f(x,y)dy$，再求这条曲线的面积。**Fubini 定理断言：在合适条件下，两者相等。**

## 2 Tonelli 定理

**定理（Tonelli）**：设 $f:\mathbb{R}^{p+q}\to[0,+\infty]$ 非负可测。则

- 对 a.e. $x$，$y\mapsto f(x,y)$ 可测（截面可测）；
- 函数 $x\mapsto\int_{\mathbb{R}^q}f(x,y)\,dm_q(y)$ 可测（非负值）；
- 重积分等于累次积分：

$$\int_{\mathbb{R}^{p+q}}f\,dm_{p+q}=\int_{\mathbb{R}^p}\left(\int_{\mathbb{R}^q}f(x,y)\,dm_q(y)\right)dm_p(x)$$

**证明**（四阶段标准路径）：

- **指示函数**：$f=\chi_E$（$E$ 可测），重积分 $=m_{p+q}(E)$；累次积分 $=\int m_q(E_x)dx$（$E_x$ 是截面）。而乘积测度定义正是「截面测度沿 $x$ 积分」——由乘积测度构造，等式成立。
- **简单函数**：线性组合，线性性传递。
- **非负可测**：$\varphi_k\uparrow f$（逼近定理），对每个 $\varphi_k$ 成立，令 $k\to\infty$——左边由单调收敛，右边由「逐 $x$ 单调收敛 + 外层单调收敛」交换，得等式。<span class="marginnote">第四阶段的「两层单调收敛交换」：$\int_p\lim_k(\cdot)=\lim_k\int_p(\cdot)$ 与 $\lim_k\int_q\varphi_k(x,\cdot)=\int_q\lim_k\varphi_k(x,\cdot)$ 都由 Levi 定理免费提供。<strong>非负性让两层极限交换零风险</strong>——这是 Tonelli「无条件」的实质。</span>

**推论（累次积分次序无关）**：非负可测函数，$\int_p\int_q=\int_q\int_p$（都等于重积分）——**换序自由**。

## 3 Fubini 定理

**定理（Fubini）**：设 $f\in L^1(\mathbb{R}^{p+q})$（即 $\int|f|<\infty$）。则

- 对 a.e. $x$，$y\mapsto f(x,y)\in L^1(\mathbb{R}^q)$；
- $x\mapsto\int_q f(x,y)dy$ 是 $L^1(\mathbb{R}^p)$ 函数；
- 重积分等于累次积分：

$$\int_{\mathbb{R}^{p+q}}f\,dm_{p+q}=\int_{\mathbb{R}^p}\left(\int_{\mathbb{R}^q}f(x,y)\,dm_q(y)\right)dm_p(x)$$

**证明**：$f=f^+-f^-$，对 $f^+$、$f^-$ 用 Tonelli（非负），线性相减。由 $\int|f|<\infty$，两个累次积分有限（Tonelli 保证），减法合法。

**辨析｜易错点：Fubini 的适用前提是「$\int|f|<\infty$」，不是「某个累次积分收敛」。** 反例（Sierpiński–Fubini 反例）：$f(x,y)=\tfrac{x^2-y^2}{(x^2+y^2)^2}$ 在单位正方形上的重积分不存在，但两个累次积分都存在且不等（$\int_0^1\int_0^1=\tfrac\pi4$，$\int_0^1\int_0^1=-\tfrac\pi4$）。**「先积 $x$」与「先积 $y$」给出不同答案**——这正是 Lebesgue 积分拒绝条件收敛的原因：它要保证换序恒等。<span class="marginnote">$f(x,y)=\tfrac{x^2-y^2}{(x^2+y^2)^2}$ 的重积分 $\int_0^1\int_0^1$ 绝对发散（$\int|f|=\infty$），因此 Fubini 不适用，累次积分可以各说各话。<strong>「重积分存在」与「累次积分存在且相等」之间的鸿沟，只能靠 $\int|f|<\infty$ 填平</strong>。</span>

## 4 公式解析：Tonelli 的「两层极限」交换

Tonelli 证明最后一步的交换结构：

$$\int_p\int_q\lim_k\varphi_k=\int_p\lim_k\int_q\varphi_k=\lim_k\int_p\int_q\varphi_k=\lim_k\int_{p+q}\varphi_k=\int_{p+q}f$$

- **第一步，读「内层交换」**：$\int_q\lim_k\varphi_k(x,\cdot)=\lim_k\int_q\varphi_k(x,\cdot)$——对**每个固定的 $x$**，非负函数沿 $y$ 的单调收敛（Levi），交换合法。
- **第二步，读「外层交换」**：$\int_p\lim_k g_k=\lim_k\int_pg_k$，其中 $g_k(x)=\int_q\varphi_k(x,\cdot)$——$g_k$ 是「逐 $x$ 的内层结果」，非负递增（$\varphi_k\uparrow$），外层 Levi 再交换。
- **第三步，读「两次交换的总账」**：两个 Levi 接力，把「$\lim_k$ 与 $\int_p\int_q$」的交换归结为「与 $\int_q$」和「与 $\int_p$」的两次独立交换。**每一层都由单调性担保**——非负递增，永远可交换。

**「先内层后外层」的 Levi 接力**，是 Tonelli 证明的标准结构，也是「非负性」在积分交换中的全部作用。

## 6 数值演练与 Fubini 速查

**算例一（重积分 = 累次积分的数值验证）**：$\int_{[0,1]^2}xy\,dm_2$。累次：$\int_0^1(\int_0^1xy\,dy)dx=\int_0^1x\cdot\tfrac12dx=\tfrac14$；重积分 $\int_0^1\int_0^1xy\,dxdy=\tfrac12\cdot\tfrac12=\tfrac14$。**换序后答案不变**——$xy\ge0$ 可积，Tonelli/Fubini 都适用。

**算例二（条件收敛的换序陷阱）**：$f(x,y)=\tfrac{x^2-y^2}{(x^2+y^2)^2}$ 于 $[0,1]^2$。$\int_0^1\int_0^1f\,dxdy=\tfrac\pi4$ 而 $\int_0^1\int_0^1f\,dy dx=-\tfrac\pi4$——**两个累次积分符号相反**。根因：$\int|f|=\infty$（重积分绝对发散），Fubini 前提不满足。

**对照表：Tonelli vs Fubini**

| 定理 | 条件 | 结论 |
| --- | --- | --- |
| Tonelli | $f\ge0$ 可测 | 重积分 = 任一累次积分（含 $+\infty$） |
| Fubini | $f\in L^1$ | 重积分 = 累次积分，截面 a.e. 可积 |
| 反例 | $\int|f|=\infty$ | 换序可能失败 |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| 截面 $f(x,\cdot)$ | 固定 $x$ 沿 $y$ 的函数 |
| $m_{p+q}$ | 乘积空间测度 |
| 边缘化 | 联合 → 边际（概率） |
| $L^1$ | 绝对可积 |

**辨析｜易错点：Tonelli 允许 $+\infty$，Fubini 要求有限。** 非负函数重积分可为 $+\infty$（Tonelli 仍成立）；但 Fubini 的结论（两个累次积分都有限）依赖 $\int|f|<\infty$。**「非负」给免费换序，「可积」给有限结论。**

### 三步记住「先 Tonelli 后 Fubini」

- **非负**：$f\ge0$ → 用 Tonelli，换序自由。
- **可积**：$\int|f|<\infty$ → 拆正负部，用 Tonelli。
- **线性**：相减得 Fubini 结论。

**延伸（与概率论连接）**：$E[g(X,Y)]=\int\int g\,dF_{XY}$ 的「联合→边际」正是 Fubini——先对 $y$ 积分得到条件期望的再积分。**边缘化、重期望法则 $E[X]=E[E[X\mid Y]]$ 都是 Fubini 的概率措辞。**

**一道收束练习**：用 Tonelli 证明 $\int_0^\infty\tfrac{e^{-x}-e^{-ax}}{x}dx=\ln a$（把 $\tfrac1x=\int_0^\infty e^{-tx}dt$ 代入，交换积分次序）——它展示 Tonelli 在「换序 + 参数积分」中的威力。

## 7 小结

- **Tonelli**：非负可测函数，重积分 = 任一累次积分，换序自由，允许 $+\infty$。
- **Fubini**：$f\in L^1$，重积分 = 累次积分；先 Tonelli 后拆正负。
- **前提本质**：$\int|f|<\infty$ 是换序安全的充分条件；条件收敛的累次积分可各说各话。
- **反例**：$\tfrac{x^2-y^2}{(x^2+y^2)^2}$ 两个累次积分不等（$\pm\pi/4$）。
- **应用**：概率论边缘化、统计中的联合→边际、多变量计算的宪法。
- **数值**：$\int_{[0,1]^2}xy=\tfrac14$ 换序不变；条件收敛反例 $\pm\pi/4$。

在下一节，我们回到几何直觉：**积分的几何意义**——可测函数的下方图形与积分的关系。
