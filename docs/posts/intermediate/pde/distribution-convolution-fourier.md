---
title: 广义函数的卷积与傅里叶变换
date: 2026-08-07
---

# 广义函数的卷积与傅里叶变换

<div class="epigraph">
<p>卷积给 δ 一个身份证：任何函数与 δ 卷积，都原样返回自己。</p>
<footer>—— 广义函数卷积与变换</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第八章 ｜ 2026-08-07</p>
</div>

## 为什么从卷积与变换开始

本专题反复出现「解 = 初值 * 传播子」「$\mathcal{F}[\delta] = 1$」这类运算，但 δ 不是普通函数，这些式子此前只是「碰巧能用」。这一节在广义函数框架下把它们严格化：定义广义函数的**卷积**（δ 是卷积的单位元）与**傅里叶变换**（δ 变换为 1、常数变换为 $2\pi\delta$）。严格化之后，「基本解 = 算子符号的逆变换」这个思想浮出水面——它是理解线性 PDE 的终极视角。

## 1 广义函数与测试函数的卷积

**广义函数 $T$ 与测试函数 $\varphi$ 的卷积**：

$$
(T * \varphi)(x) = \langle T(y),\ \varphi(x - y)\rangle
$$

右端是「$T$ 作用在平移后的测试函数上」——良定义，且**结果是光滑函数**（$\varphi$ 光滑 + 求导可穿过）。

**卷积的转移性质**：$(T * \varphi)' = T' * \varphi = T * \varphi'$——**求导可以移到任意一个因子**。这条性质让「卷积微分」像普通乘积一样自由。

**δ 是卷积的单位元**：

$$
\delta * \varphi = \varphi, \qquad \delta * T = T
$$

**验证**：$(\delta * \varphi)(x) = \langle\delta(y),\varphi(x-y)\rangle = \varphi(x)$ ✓。**δ 在卷积下的角色 = 1 在乘法下的角色**——这就是「基本解」理论的代数基础。

## 2 广义函数间的卷积

两个广义函数 $T * S$ 的卷积需要条件（例如一个支集紧，或两者都是适定的「好」分布）：

$$
\langle T * S,\ \varphi\rangle = \langle T(x),\ \langle S(y),\ \varphi(x+y)\rangle\rangle
$$

**只要内层作用的结果是好测试函数，卷积就良定义。** 常见充分条件：$S$ 支集紧。

**卷积结合律与交换律在适当条件下成立**：$T * S = S * T$、$(T*S)*R = T*(S*R)$。**δ 仍是单位元**：$T * \delta = T$。<span class="marginnote">「卷积代数」的框架让 PDE 求解变成「找卷积逆元」：$L u = f$ 若算子 $L$ 有基本解 $E$（$LE = \delta$），则 $u = E * f$ 是解（因为 $L(E*f) = (LE)*f = \delta*f = f$）。<strong>基本解 = 微分算子的「卷积逆元」</strong>——这个代数视角统一了热核、泊松核、格林函数的所有实例。</span>

## 3 缓增广义函数与傅里叶变换

普通广义函数 $\mathcal{D}'$ 对傅里叶变换「太宽」——$\mathcal{F}$ 需要快速衰减的函数类。引入**Schwartz 空间** $\mathcal{S}$（无穷光滑 + 与所有多项式乘积都快速衰减）与**缓增广义函数** $\mathcal{S}'$（$\mathcal{S}$ 上的连续线性泛函）。

**缓增广义函数的傅里叶变换**（转移原则）：

$$
\langle \mathcal{F}[T],\ \varphi\rangle = \langle T,\ \mathcal{F}[\varphi]\rangle, \qquad \varphi \in \mathcal{S}
$$

**「变换的作用」被转移成「对变换后的测试函数作用」。** 由于 $\mathcal{F}:\mathcal{S}\to\mathcal{S}$ 是同构，右端良定义且 $\mathcal{F}$ 在 $\mathcal{S}'$ 上是双射——**每个缓增广义函数都有傅里叶变换，且变换是可逆的**。

## 4 公式解析：两个里程碑等式

**等式一：$\mathcal{F}[\delta] = 1$**

- **第一步，用定义。** $\langle\mathcal{F}[\delta],\varphi\rangle = \langle\delta,\mathcal{F}[\varphi]\rangle = \mathcal{F}[\varphi](0)$。
- **第二步，算变换在 0。** $\mathcal{F}[\varphi](0) = \int\varphi(x)e^{-i\cdot0x}dx = \int\varphi(x)dx = \langle 1,\varphi\rangle$。
- **第三步，结论。** $\langle\mathcal{F}[\delta],\varphi\rangle = \langle 1,\varphi\rangle$，故 $\mathcal{F}[\delta] = 1$——**δ 的变换是常数 1**。

**等式二：$\mathcal{F}[1] = 2\pi\,\delta$**

- **第一步，对称论证。** 由互逆性 $\mathcal{F}^{-1}[\delta] = \frac{1}{2\pi}$？不——小心常数。用 $\mathcal{F}[\delta]=1$ 与互逆公式 $\mathcal{F}^{-1}\mathcal{F} = \text{id}$ 反推：$\mathcal{F}^{-1}[1] = \delta$，即 $\mathcal{F}[\delta] = 1$ 是同一枚硬币。更直接：$\langle\mathcal{F}[1],\varphi\rangle = \langle1,\mathcal{F}[\varphi]\rangle = \int\mathcal{F}[\varphi](\xi)d\xi = 2\pi\varphi(0)$（傅里叶积分的逆变换公式在 $\xi=0$）$= \langle2\pi\delta,\varphi\rangle$。
- **第二步，结论。** $\mathcal{F}[1] = 2\pi\delta$。

**这两个等式把「周期函数」（常数可看作零频率）的谱说成「集中在零频率的 δ」**——第七篇预告的「周期函数的变换是 δ 组合」在这里落地：$e^{i\omega_0x}$ 的变换是 $2\pi\delta(\omega - \omega_0)$，常数是特例 $\omega_0 = 0$。<span class="marginnote">「常数的傅里叶变换 = $2\pi\delta$」把级数与变换两大体系彻底统一：周期函数用离散谱（δ 峰在整数频率处），非周期函数用连续谱。广义函数是这场统一的黏合剂——没有 $\mathcal{S}'$，$\mathcal{F}[1]$ 根本无意义。</span>

## 5 基本解与算子符号

傅里叶变换让「微分算子」与「多项式」对应：$\mathcal{F}[\partial_x^k T] = (i\omega)^kT$（变换的微分性质在广义意义下成立）。于是

**常系数微分算子 $L = \sum a_k\partial_x^k$ 的「符号」是多项式 $P(\omega) = \sum a_k(i\omega)^k$**，且

$$
\mathcal{F}[Lu] = P(\omega)\,\mathcal{F}[u]
$$

**求基本解** $E$（$LE = \delta$）：两边变换得 $P(\omega)\mathcal{F}[E] = \mathcal{F}[\delta] = 1$，故

$$
\mathcal{F}[E](\omega) = \frac{1}{P(\omega)} \quad\Longrightarrow\quad E = \mathcal{F}^{-1}\Big[\frac{1}{P}\Big]
$$

**基本解的变换是算子符号的倒数。** 热方程 $L = \partial_t - a^2\partial_{xx}$（对空间变换）符号 $P = -a^2\omega^2$（含时间参数），$1/P$ 的逆变换正是热核 $G$——**「热核 = 符号倒数的逆变换」把第七篇的谱方法与第九篇的基本解理论焊成一体**。

**辨析｜易错点：** 符号倒数的逆变换是「形式基本解」，其严格存在需要 $1/P$ 属于合适的函数/广义函数类；$P$ 的零点（算子的「特征值」）对应变换的极点，需要留数理论处理（复变方法）。**「形式推导 + 严格化」是基本解理论的标准节奏**——先由符号倒数猜出基本解，再用广义函数验证 $LE = \delta$。

## 6 数值算例：阶跃函数与符号函数的变换

- **第一步，阶跃的导数。** $H'(x) = \delta$，两边变换：$i\omega\,\mathcal{F}[H] = 1$，故 $\mathcal{F}[H](\omega) = \mathrm{p.v.}\,\frac{1}{i\omega} + \pi\delta(\omega)$（主值 + δ 修正）。
- **第二步，符号函数。** $\mathrm{sgn}(x) = 2H(x) - 1$，$\mathcal{F}[\mathrm{sgn}] = \frac{2}{i\omega}$（主值意义）。
- **第三步，核对反变换。** $\mathcal{F}^{-1}\big[\frac{2}{i\omega}\big] = \mathrm{sgn}(x)$——逆变换公式在广义意义下成立。
- **第四步，读结构。** 这些「不绝对可积」函数的变换都含主值或 δ——**广义函数把「粗糙函数」也送进频域**，且结果自动带「正则化」记号。

**数值例子的意义**：阶跃、符号函数在经典意义下没有傅里叶变换，广义函数框架下却干净利落——「$1/\omega$ 型奇点用主值、零点用 δ」是这类变换的标准结构。

## 7 卷积与变换在 PDE 的统一作用

把本专题的所有「解 = 卷积」收拢：

- **热传导**：$u = \varphi * G$（热核 = 基本解）。
- **波动**：$u = $ 初值在特征锥上的卷积（延迟格林函数）。
- **拉普拉斯**：$u = \Gamma * \rho$（基本解卷积源）。
- **统一视角**：$Lu = f$ 的解 $u = E * f$（$E$ 是基本解）——**「解线性 PDE = 与基本解卷积」是全部线性理论的终局**。

**辨析｜易错点：** 卷积要求「支集 / 衰减条件」——两个「太胖」的广义函数卷积可能无定义。判别方法是看「内层作用后是否仍是好测试函数」：若内层给的是「好函数」，卷积良定义。**「卷积的可定义性」比「函数的可定义性」更微妙**，是分布理论最后的边界。<span class="marginnote">从「从极限到大模型」的视角，「解 = 卷积」是「线性系统 + 脉冲响应」的最纯粹表达——信号处理、控制系统、PDE 全部共享「输出 = 输入卷积冲激响应」这一结构。理解「基本解 = 冲激响应」，就理解了线性科学的一半。</span>

## 8 小结

- 广义函数与测试函数卷积是光滑函数，δ 是卷积单位元。
- 广义函数间卷积需条件（如支集紧），$LE = \delta$ 使 $u = E*f$ 解 $Lu = f$。
- 缓增广义函数 $\mathcal{S}'$ 上傅里叶变换是双射。
- $\mathcal{F}[\delta] = 1$、$\mathcal{F}[1] = 2\pi\delta$——级数与变换统一的黏合剂。
- 基本解的变换 = 算子符号的倒数：$E = \mathcal{F}^{-1}[1/P]$。

在下一节，我们引入索伯列夫（Sobolev）空间初步。
