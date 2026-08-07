---
title: 基本解（三维 r⁻¹ 与二维 ln r）
date: 2026-08-08
---

# 基本解（三维 r⁻¹ 与二维 ln r）

<div class="epigraph">
<p>点源的响应，是全场响应的积木。</p>
<footer>—— 基本解与格林函数的哲学</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第三章 ｜ 2026-08-08</p>
</div>

## 为什么从基本解开始

泊松方程 $\Delta u = -\rho$ 在「源」任意分布时怎么解？答案是：**先解一个点的源，再把所有点的贡献积分起来**。这个「一个点的源」的响应就是**基本解（fundamental solution）**。它有两个显眼的名字：三维的 $1/(4\pi r)$ 叫**牛顿势**（Newtonian potential），二维的 $-\frac{1}{2\pi}\ln r$ 叫**对数势**。这一节定义基本解、验证它确实是点源的响应、并引出它在各维度的统一结构——它是格林函数方法、源像法、以及第九篇 δ 函数理论的枢纽。

## 1 基本解的定义

**基本解**：满足

$$
\Delta \Gamma = -\delta
$$

的分布 $\Gamma(x)$ 称为拉普拉斯算子的基本解，其中 $\delta$ 是集中在原点的 Dirac δ 函数（第九篇严格化）。

**「基本」二字的含义：它是「一个点源」的泊松方程的解。** 有了它，泊松方程 $\Delta u = -\rho$ 的解就是卷积

$$
u(x) = \int \Gamma(x - y)\,\rho(y)\,dy
$$

因为 $\Delta u = \int \Delta\Gamma(x-y)\rho(y)dy = \int (-\delta(x-y))\rho(y)dy = -\rho(x)$。<span class="marginnote">这就是「点源响应 × 源密度 的叠加」：把源密度看成无数点源的连续和，每个点源在 $x$ 处贡献 $\Gamma(x-y)\rho(y)$，积分即得全场响应。这个「基本解卷积」模式是线性 PDE 理论最通用的求解框架，第九篇会为它建立严格的广义函数基础。</span>

## 2 三维基本解：$1/(4\pi r)$

**三维基本解**

$$
\Gamma(x) = \frac{1}{4\pi r}, \qquad r = |x| = \sqrt{x^2 + y^2 + z^2}
$$

验证它满足 $\Delta\Gamma = -\delta$：

- **第一步，$r \ne 0$ 处验证调和。** 由球对称，$\Delta\Gamma = \frac{1}{r^2}\frac{\partial}{\partial r}\big(r^2\Gamma'\big)$。代入 $\Gamma = \frac{1}{4\pi r}$：$r^2\Gamma' = -\frac{1}{4\pi}$，导数为零，故 $\Delta\Gamma = 0$。
- **第二步，原点处测「源强度」。** 用 Gauss 定理：$\int_{|x|=\epsilon}\nabla\Gamma\cdot\boldsymbol{n}\,dS = \int_{|x|=\epsilon}\frac{\partial\Gamma}{\partial r}dS = -\frac{1}{4\pi\epsilon^2}\cdot 4\pi\epsilon^2 = -1$。
- **第三步，识别为 δ。** 分布的意义下，「$r\ne0$ 处为零 + 总积分 $-1$」正是 $-\delta$ 的特征。故 $\Delta\Gamma = -\delta$。

**系数 $1/(4\pi)$ 的意义：让单位球面的总通量归一化为 1。** 这是「归一化」的选择——基本解乘以任何常数仍是基本解，但 $1/(4\pi)$ 让源强度恰好等于 1，与泊松方程右端 $\rho$ 的系数匹配。<span class="marginnote">对比物理：点电荷 $Q$ 的电势是 $\frac{Q}{4\pi\varepsilon_0}\frac{1}{r}$，其中的 $1/(4\pi)$ 来自立体角归一化。基本解 $1/(4\pi r)$ 正是「源强度为 1 的点电荷」的电势（无量纲化后）。</span>

## 3 二维基本解：$-\frac{1}{2\pi}\ln r$

二维情形的调和分析中，$1/r$ 不再特殊——二维的「牛顿势」是**对数势**

$$
\Gamma(x) = -\frac{1}{2\pi}\ln r, \qquad r = \sqrt{x^2 + y^2}
$$

验证方法与三维逐字平行：

- **第一步，$r \ne 0$ 处调和。** 二维极坐标下 $\Delta = \frac{1}{r}\frac{\partial}{\partial r}\big(r\frac{\partial}{\partial r}\big)$。对 $-\frac{1}{2\pi}\ln r$：$r\Gamma' = -\frac{1}{2\pi}$ 为常数，导数为零，故 $\Delta\Gamma = 0$。
- **第二步，原点处测源强度。** 在圆周 $|x| = \epsilon$ 上 $\frac{\partial\Gamma}{\partial r} = -\frac{1}{2\pi\epsilon}$，环积分 $\oint\frac{\partial\Gamma}{\partial r}ds = -\frac{1}{2\pi\epsilon}\cdot 2\pi\epsilon = -1$。
- **第三步，结论。** $\Delta\Gamma = -\delta$，二维基本解成立。

**二维与三维的关键差别：$1/r$ 在二维不是基本解**（$\Delta(1/r)$ 在二维不给 δ 源），对数势才是。为什么？因为二维的「点源」其实是**线源**的截面——一根无限长带电线的场在横截面上就是对数势。<span class="marginnote">维数改变基本解的形状，这是拉普拉斯算子各向同性 + 奇点阶数匹配的必然结果。更一般的 $n$ 维基本解是 $\Gamma \propto r^{2-n}$（$n \ge 3$），对数势只在 $n=2$ 出现；$n \ge 3$ 时 $r^{2-n}$ 的衰减与维度对应，$n=2$ 时 $r^0 = 1$ 发散，极限给出 $\ln r$。</span>

## 4 公式解析：为什么是 $r^{2-n}$ 与 $\ln r$

把「基本解长什么样」统一成一条规律。假设 $\Gamma = \Phi(r)$ 球对称，在 $r\ne0$ 处调和。$n$ 维极坐标拉普拉斯作用于径向函数：

$$
\Delta\Phi = \Phi'' + \frac{n-1}{r}\Phi' = 0
$$

- **第一步，解径向方程。** 令 $v = \Phi'$，则 $v'/v = -(n-1)/r$，解得 $\Phi'(r) = C\,r^{-(n-1)}$。
- **第二步，积分。** $n \ne 2$ 时 $\Phi = \frac{C}{2-n}r^{2-n} + C'$；$n = 2$ 时 $\Phi = C\ln r + C'$。**取负对数/负幂**并归一化，得到
  $$ n \ge 3:\ \Gamma = \frac{1}{n(2-n)\omega_n}r^{2-n}, \qquad n = 2:\ \Gamma = -\frac{1}{2\pi}\ln r $$
  其中 $\omega_n$ 是 $n$ 维单位球面积（$\omega_3 = 4\pi$，$\omega_2 = 2\pi$）。
- **第三步，读归一化。** 系数 $1/((2-n)\omega_n)$ 让源强度归一：$\oint\frac{\partial\Gamma}{\partial r}dS = 1$（带符号）。这保证 $\Delta\Gamma = -\delta$ 在任何维数成立。
- **第四步，结论。** **基本解 = 「径向调和 + 源强度归一」的唯一选择。** $n$ 维球的几何（面积 $\omega_n$）决定了系数，维数决定了幂次。

**「径向调和」约束了函数形状，「源强度归一」固定了系数——基本解由这两条唯一确定。**

## 5 基本解的意义

基本解是整个位势理论的原子，它的用途四通八达：

1. **泊松方程通解**：$\Delta u = -\rho$ 的特解是 $\Gamma * \rho$（卷积）。
2. **格林函数的构件**：格林函数 = 基本解 + 调和修正（镜像法/边界修正），下一节起全程使用。
3. **牛顿引力/静电**：$1/r$ 就是点质量的引力势，对数势是线电荷的电势——基本解直接是物理势。
4. **随机游走/布朗运动**：基本解是布朗运动的格林函数，$r^{2-n}$ 的幂次对应随机游走的返回性质（二维常返、三维暂态）。

**辨析｜易错点：** 基本解不是「在原点补值的调和函数」——它在原点发散、不满足拉普拉斯方程，只在分布意义下满足 $\Delta\Gamma = -\delta$。写「$\Gamma$ 在 $r>0$ 调和」要带着「去心区域」的限定。另外符号约定常让人困惑：有的书定义 $\Delta\Gamma = \delta$（无负号），相应泊松方程写 $\Delta u = \rho$——**只要全书约定自洽即可，但跨书对照时务必先核对符号**。<span class="marginnote">第九篇会用广义函数严格化「$\Delta\Gamma = -\delta$」这句话：届时 $\Gamma$ 的奇点在分布意义下求导，δ 源自动出现，无需手动「挖球取极限」。基本解是第九篇概念最自然的引子。</span>

## 6 小结

- 基本解满足 $\Delta\Gamma = -\delta$，是「一个点源」的泊松方程解。
- 三维基本解 $1/(4\pi r)$（牛顿势），二维 $-\frac{1}{2\pi}\ln r$（对数势）。
- 验证两步骤：$r\ne0$ 处调和 + 小球/小圆通量归一化为 1。
- $n$ 维统一：$\Gamma \propto r^{2-n}$（$n\ge3$），$n=2$ 为 $\ln r$，系数由单位球面积归一化。
- 基本解是泊松方程通解、格林函数、物理势与随机分析共同的基础。

在下一节，我们用基本解与格林公式导出调和函数的积分表达式。
