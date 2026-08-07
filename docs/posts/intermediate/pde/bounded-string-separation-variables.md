---
title: 有界弦的初边值问题：分离变量法
date: 2026-08-08
---

# 有界弦的初边值问题：分离变量法

<div class="epigraph">
<p>一根两端固定的弦，只会以整数倍的基频振动。</p>
<footer>—— 泛音列（harmonic series）的由来</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第四章 ｜ 2026-08-08</p>
</div>

## 为什么从分离变量法开始

达朗贝尔公式处理无限弦，但真实的琴弦两端固定，长度有限。有界弦问题的标准解法是**分离变量法（method of separation of variables）**——这是整个数学物理方程课程最重要的一套技术。它的思想看似简单：假设解能写成「空间因子 × 时间因子」，把 PDE 拆成两个 ODE；但它的副产品极其深刻——**本征值问题**与**傅里叶级数**。学完这一节，你不仅会解弦振动，更拿到了热传导、拉普拉斯、特殊函数（第八篇）全部问题的通用钥匙。

## 1 有界弦的初边值问题

两端固定的弦，长度 $L$，振动满足

$$
u_{tt} = a^2 u_{xx}, \qquad 0 < x < L,\ t > 0
$$

$$
u(0,t) = 0,\qquad u(L,t) = 0 \quad\text{（固定端边界）}
$$

$$
u(x,0) = \varphi(x),\qquad u_t(x,0) = \psi(x) \quad\text{（初始条件）}
$$

这是个**初边值问题**：既有初始条件（对 $t$），又有边界条件（对 $x$）。<span class="marginnote">对比无限弦柯西问题只有初始条件、没有边界；有界弦把「反射」变成「驻波」——两列行波在边界来回反射并叠加，最终呈现为固定的振动模态（驻波）。分离变量法正是直接去寻找这些驻波模态。</span>

## 2 分离变量：把 PDE 拆成两个 ODE

设解有形如

$$
u(x,t) = X(x)\,T(t)
$$

的非零解，代入方程：

$$
X\,T'' = a^2 X''\,T \quad\Longrightarrow\quad \frac{T''}{a^2 T} = \frac{X''}{X}
$$

左边只依赖 $t$，右边只依赖 $x$，两边必须同为常数，记为 $-\lambda$：

$$
X'' + \lambda X = 0, \qquad T'' + a^2\lambda T = 0
$$

**分离常数 $\lambda$ 不是自由参数，它被边界条件「挑选」。** 边界条件 $u(0,t) = X(0)T(t) = 0$ 给出 $X(0) = 0$，同理 $X(L) = 0$。于是空间因子满足

$$
X'' + \lambda X = 0, \qquad X(0) = X(L) = 0
$$

## 3 公式解析：本征值问题的求解

求解 $X'' + \lambda X = 0$、$X(0) = X(L) = 0$：

- **第一步，试 $\lambda$ 的符号。** 若 $\lambda \le 0$，通解是指数函数或线性函数，边界条件强制 $X \equiv 0$——只有零解，无意义。**故必须有 $\lambda > 0$。**
- **第二步，写通解。** $X(x) = A\cos(\sqrt\lambda\, x) + B\sin(\sqrt\lambda\, x)$。
- **第三步，套边界。** $X(0) = A = 0$；$X(L) = B\sin(\sqrt\lambda\, L) = 0$。非零解要求 $\sin(\sqrt\lambda\, L) = 0$，即
  $$ \sqrt\lambda\, L = n\pi \quad\Longrightarrow\quad \lambda_n = \left(\frac{n\pi}{L}\right)^2, \qquad n = 1, 2, 3, \dots $$
- **第四步，得本征函数。** 对每个 $n$，$X_n(x) = \sin\frac{n\pi x}{L}$（取 $B=1$）。

$\lambda_n$ 称为**本征值**，$X_n$ 称为**本征函数**。它们是弦的**固有振动模态**——两端固定的弦只能以 $n\pi/L$ 这样的离散频率振动，这就是「泛音列」的数学由来。

时间因子满足 $T_n'' + a^2\lambda_n T_n = 0$，解为

$$
T_n(t) = A_n\cos\frac{an\pi t}{L} + B_n\sin\frac{an\pi t}{L}
$$

于是对每个 $n$，$u_n = X_n T_n$ 是一个**驻波解**，频率 $\omega_n = \frac{an\pi}{L}$（基频 $\omega_1 = a\pi/L$）。

## 4 叠加原理与傅里叶级数

方程线性齐次，叠加原理保证任意有限个 $u_n$ 的和仍是解。把全部模态叠加，猜测

$$
u(x,t) = \sum_{n=1}^{\infty}\left(A_n\cos\frac{an\pi t}{L} + B_n\sin\frac{an\pi t}{L}\right)\sin\frac{n\pi x}{L}
$$

用初始条件定系数。令 $t = 0$：

$$
\varphi(x) = \sum_{n=1}^{\infty} A_n \sin\frac{n\pi x}{L}, \qquad \psi(x) = \sum_{n=1}^{\infty} B_n\frac{an\pi}{L}\sin\frac{n\pi x}{L}
$$

**这里出现了一个重大事件：任意的初始位移被写成 $\sin$ 函数的无穷级数——傅里叶正弦级数。** 利用本征函数的正交性

$$
\int_0^L \sin\frac{n\pi x}{L}\sin\frac{m\pi x}{L}\,dx = \begin{cases} \frac{L}{2}, & n = m \\ 0, & n \neq m \end{cases}
$$

两边乘 $\sin\frac{m\pi x}{L}$ 再积分，得

$$
A_n = \frac{2}{L}\int_0^L \varphi(x)\sin\frac{n\pi x}{L}\,dx, \qquad B_n = \frac{2}{an\pi}\int_0^L \psi(x)\sin\frac{n\pi x}{L}\,dx
$$

**分离变量法把「解 PDE」归结为「算傅里叶系数」——而傅里叶级数是否收敛、能否代表任意初值，是第五篇《傅里叶级数解的收敛性》要严格回答的问题。**

## 5 驻波的物理图像

有界弦的解是驻波叠加，与无限弦的行波解有本质区别：

| 性质 | 无限弦 | 有界弦 |
| --- | --- | --- |
| 解的结构 | 行波 $F(x+at) + G(x-at)$ | 驻波 $\sum \sin\frac{n\pi x}{L}\cdot\text{时间因子}$ |
| 频率 | 连续谱 | 离散谱 $\omega_n = an\pi/L$ |
| 波节 | 无 | $x = kL/n$ 处波节 |

**辨析｜易错点：** 分离变量法的假设 $u = XT$ 只是**猜测**，不是「所有解都这样」。真正的逻辑是：先找一族形如 $XT$ 的**特解**（驻波模态），再用叠加原理把初值展开成这些特解的线性组合。把「分离变量假设」误解为「解必然可分离」，会漏掉一般解的构造逻辑。收敛性定理保证：只要初值适当光滑，这个叠加确实给出解。

弦的振动 = 基频 + 二倍频 + 三倍频……的叠加。琴弦被拨后，各模态按各自的频率振动，它们的时间因子不同——这就是音色（谐波含量）的数学基础。<span class="marginnote">不同乐器弹同一个音高，基频相同，但泛音列的强度分布不同，所以音色不同。分离变量法给出的模态分解，正是「音色 = 傅里叶系数分布」的雏形，与第七篇傅里叶分析的信号视角完全同构。</span>

## 6 小结

- 有界弦初边值问题用分离变量 $u = XT$ 化为两个 ODE。
- 本征值问题 $X'' + \lambda X = 0$、$X(0)=X(L)=0$ 给出离散谱 $\lambda_n = (n\pi/L)^2$ 与本征函数 $\sin(n\pi x/L)$。
- 解是驻波叠加，频率 $\omega_n = an\pi/L$，对应泛音列。
- 叠加原理 + 正交性把初值展开成傅里叶正弦级数，系数由积分给出。
- 分离变量法是热传导、拉普拉斯、特殊函数问题的通用模板。

在下一节，我们处理非齐次波动方程——杜阿梅尔（Duhamel）原理。
