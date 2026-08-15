---
title: 泊松括号与正则变换初步
date: 2026-08-07
---

# 泊松括号与正则变换初步

<div class="epigraph">
<p>泊松括号是哈密顿力学的「代数语法」——它让守恒律变得机械，也预言了量子力学的对易子。</p>
<footer>—— 西梅翁 · 德尼 · 泊松（Siméon Denis Poisson），1809</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 周衍柏《理论力学》分析力学部分 ｜ 2026-08-07</p>
</div>

## 为什么从泊松括号开始

哈密顿正则方程描述了运动，但分析力学还需要一套「代数」来处理守恒量与坐标变换。**泊松括号（Poisson bracket）**是两个相空间函数的运算，它让「某量是否守恒」变成「它与哈密顿量的泊松括号是否为零」——一个纯代数的判据。而**正则变换（canonical transformation）**让我们能在保持哈密顿方程形式的前提下自由更换坐标——是求解复杂系统的利器。这两个概念在经典力学与量子力学之间有直接的对应（对易子），是通往量子力学的桥梁。

## 1 泊松括号的定义

两个相空间函数 $f(q, p)$、$g(q, p)$ 的**泊松括号（Poisson bracket）**：

$$\{f, g\} = \sum_j\left(\frac{\partial f}{\partial q_j}\frac{\partial g}{\partial p_j} - \frac{\partial f}{\partial p_j}\frac{\partial g}{\partial q_j}\right)$$

**基本泊松括号**（正则变量的自对易关系）：

$$\{q_j, q_k\} = 0, \qquad \{p_j, p_k\} = 0, \qquad \{q_j, p_k\} = \delta_{jk}$$

**重点：基本泊松括号 $\{q_j, p_k\} = \delta_{jk}$ 是哈密顿力学的「代数学公理」。** 它把「坐标与动量是共轭对」这一事实编码成代数关系——$q$ 与自己的动量泊松括号为 1，与其他量无关。<span class="marginnote">泊松括号的性质：反对称（$\{f,g\} = -\{g,f\}$）、双线性、满足雅可比恒等式（$\{f,\{g,h\}\} + \{g,\{h,f\}\} + \{h,\{f,g\}\} = 0$）——它使相空间函数构成一个「李代数」。这些代数性质与量子力学对易子的性质完全同构——这是「经典 → 量子」对应的关键。</span>

## 2 泊松括号与运动方程

函数 $f(q, p, t)$ 沿运动轨迹的时间变化率：

$$\frac{\mathrm{d}f}{\mathrm{d}t} = \{f, H\} + \frac{\partial f}{\partial t}$$

（用正则方程代入可证。）于是：

- **守恒律判据**：若 $f$ 不显含时间且 $\{f, H\} = 0$，则 $f$ 守恒。

**重点：$f$ 守恒 ⟺ $\{f, H\} = 0$（$f$ 不显含 $t$ 时）——守恒律变成泊松括号运算。** 例：$\{H, H\} = 0$，所以能量守恒；若 $H$ 不含某坐标 $q_j$，则 $\{p_j, H\} = -\partial H/\partial q_j = 0$，动量守恒。正则方程本身也可写成泊松括号形式：$\dot{q}_j = \{q_j, H\}$、$\dot{p}_j = \{p_j, H\}$。

**辨析｜易错点：**泊松括号的求值顺序：$\{f, g\}$ 先对 $f$ 求 $q$ 偏导乘 $g$ 对 $p$ 偏导，减去 $f$ 对 $p$ 偏导乘 $g$ 对 $q$ 偏导。符号别错（第二项是减号）。计算 $\{f, H\}$ 时，把 $H$ 当第二个参数。

## 3 公式解析：泊松括号判守恒

验证角动量分量 $L_z = xp_y - yp_x$ 在中心力场（$H = \frac{p^2}{2m} + V(r)$）中守恒。

$$
\{L_z, H\} = \left\{\sum_i\epsilon_{zik}x_ip_k, H\right\}
$$

- **第一步，展开泊松括号**：$\{L_z, H\} = \sum_i\left(\frac{\partial L_z}{\partial q_i}\frac{\partial H}{\partial p_i} - \frac{\partial L_z}{\partial p_i}\frac{\partial H}{\partial q_i}\right)$，$q_i = (x, y, z)$。
- **第二步，算偏导**：$\frac{\partial L_z}{\partial x} = p_y$，$\frac{\partial L_z}{\partial y} = -p_x$，$\frac{\partial L_z}{\partial z} = 0$；$\frac{\partial L_z}{\partial p_x} = -y$，$\frac{\partial L_z}{\partial p_y} = x$；$\frac{\partial H}{\partial p_i} = p_i/m$，$\frac{\partial H}{\partial q_i} = \frac{\partial V}{\partial q_i}$。
- **第三步，代入**：$\{L_z, H\} = p_y\frac{p_x}{m} + (-p_x)\frac{p_y}{m} - \left[(-y)\frac{\partial V}{\partial x} + x\frac{\partial V}{\partial y}\right] = y\frac{\partial V}{\partial x} - x\frac{\partial V}{\partial y}$。
- **第四步，中心力场**：$V = V(r)$，$\frac{\partial V}{\partial x} = V'\frac{x}{r}$、$\frac{\partial V}{\partial y} = V'\frac{y}{r}$，于是 $yV'\frac{x}{r} - xV'\frac{y}{r} = 0$——$\{L_z, H\} = 0$，角动量守恒。

**重点：泊松括号让守恒律变成「机械化」的代数运算——算一次 $\{f, H\}$，就知道 $f$ 守不守恒。** 无需猜、无需积分，纯求导运算。

## 4 正则变换初步

**正则变换（canonical transformation）**：从 $(q, p)$ 到新坐标 $(Q, P)$ 的变换，使哈密顿方程的形式保持不变（新哈密顿量 $K(Q, P, t)$ 仍满足正则方程）。

**生成函数（generating function）**：正则变换由生成函数产生，如 $F_1(q, Q, t)$：

$$p_j = \frac{\partial F_1}{\partial q_j}, \qquad P_j = -\frac{\partial F_1}{\partial Q_j}, \qquad K = H + \frac{\partial F_1}{\partial t}$$

**重点：正则变换保持哈密顿方程形式不变——坐标变换的自由度被极大扩展，可以选择让哈密顿量「简化」的坐标。** 理想的正则变换把 $K$ 化得尽可能简单（如所有坐标循环、$K$ 不含 $Q$），使方程立即积分。这个「找生成函数解方程」的思想是求解哈密顿系统的高级工具。<span class="marginnote">「正则变换的威力」：如果能找到让 $K$ 不含某个 $Q_j$ 的变换，对应的 $P_j$ 就是常数——问题降维。若 $K$ 不含时间，能量守恒。终极版本是「哈密顿-雅可比方程」：找变换使 $K = 0$，运动方程化为「全部守恒」——这是通往量子力学的薛定谔方程的另一条路（作用量作为生成函数）。</span>

## 5 泊松括号与量子力学

**对应原理（经典 → 量子）**：泊松括号对应量子力学中的对易子：

$$\{f, g\} \;\longrightarrow\; \frac{1}{i\hbar}[\hat{f}, \hat{g}]$$

**基本对应**：$\{q, p\} = 1 \;\longrightarrow\; [\hat{q}, \hat{p}] = i\hbar$——正则量子化的代数表述。

**重点：泊松括号是量子对易子的经典对应——$\{q, p\} = 1$ 在量子中变成 $[\hat{q}, \hat{p}] = i\hbar$。** 这个对应是「经典力学 → 量子力学」的数学桥梁：经典泊松括号的代数结构（反对称、雅可比恒等式）在量子力学中原样保留（对易子），只是乘上 $i\hbar$。第 116 节预告的「正则量子化」在此落实。<span class="marginnote">「从泊松到对易」是量子化的纲领：物理量 $f$ 变成算符 $\hat{f}$，泊松括号变对易子。这也解释了不确定关系：$[\hat{q}, \hat{p}] = i\hbar \neq 0$ 意味着 $q$、$p$ 不能同时有确定值——不确定关系（第 101 节）的代数根源。分析力学最抽象的角落，恰恰是量子力学最基础的出发点。</span>

## 6 数值算例：基本泊松括号的计算

直接计算 $\{x, p_x\}$、$\{x, y\}$ 与 $\{x, p_y\}$，验证基本关系。

$$

\{x, p_x\} = \frac{\partial x}{\partial x}\frac{\partial p_x}{\partial p_x} - \frac{\partial x}{\partial p_x}\frac{\partial p_x}{\partial x} = 1\times1 - 0\times0 = 1, \qquad \{x, y\} = 0, \qquad \{x, p_y\} = 0

$$

- **第一步，展开 $\{x, p_x\}$**：$\frac{\partial x}{\partial x} = 1$、$\frac{\partial p_x}{\partial p_x} = 1$、$\frac{\partial x}{\partial p_x} = 0$、$\frac{\partial p_x}{\partial x} = 0$，得 $\{x, p_x\} = 1$。
- **第二步，算 $\{x, y\}$**：$x$ 与 $y$ 是不同坐标，所有偏导交叉为零，$\{x, y\} = 0$。
- **第三步，算 $\{x, p_y\}$**：$x$ 与 $p_y$ 是不匹配的坐标-动量对，$\{x, p_y\} = 0$。
- **第四步，解读**：只有匹配的坐标-动量对（$q_j$ 与自己的 $p_j$）泊松括号为 1，其余全为零——$\{q_j, p_k\} = \delta_{jk}$ 正是这些结果的浓缩。这是哈密顿力学「共轭对」的代数指纹。<span class="marginnote">「$\delta_{jk}$ 的意义」：克罗内克符号 $\delta_{jk}$ 在 $j = k$ 时为 1、否则为 0。泊松括号 $\{q_j, p_k\} = \delta_{jk}$ 说：每个坐标专属自己的动量，跨坐标的括号全为零。这套代数在量子力学中变成 $[\hat{q}_j, \hat{p}_k] = i\hbar\delta_{jk}$——同一结构的量子版本。</span>

## 7 正则变换的判定与辛结构

一个变换 $(q, p) \to (Q, P)$ 是正则变换的**判定条件**：新变量的基本泊松括号不变。

$$

\{Q_j, Q_k\} = 0, \qquad \{P_j, P_k\} = 0, \qquad \{Q_j, P_k\} = \delta_{jk}

$$

- **第一步，读判定**：正则变换保持基本泊松括号——这是「形式不变」的代数表述。
- **第二步，例：恒等变换**：$Q = q$、$P = p$，显然满足——平凡的生成函数 $F_1 = qQ$。
- **第三步，例：交换变换**：$Q = p$、$P = -q$（交换坐标动量并反号），验证 $\{Q, P\} = \{p, -q\} = 1$——正则。
- **第四步，体会辛结构**：相空间是「辛流形」，正则变换是保持辛形式的变换——「面积守恒」在相空间的推广（刘维尔定理：相体积守恒）。<span class="marginnote">「辛结构与量子化」：正则变换保持泊松括号（辛形式），这让相空间的体积不变——刘维尔定理（统计物理的相空间守恒）由此而来。更深刻的是，量子力学的正则量子化要求保持泊松括号结构——所以 $[\hat{Q},\hat{P}] = i\hbar$ 也必须在任何量子化的坐标下成立。辛几何是现代理论物理（几何量子化、弦论）的通用语言，这一节的初步概念是它的入口。</span>

**辨析｜易错点：**不是所有坐标变换都是正则变换——只有保持基本泊松括号的才是。判别时逐项验证 $\{Q_j, P_k\} = \delta_{jk}$，不要只看新旧坐标的对应关系。写生成函数 $F_1(q,Q,t)$ 时，$p = \partial F_1/\partial q$、$P = -\partial F_1/\partial Q$——注意 $P$ 前是负号，方向别错。

## 8 术语速查表

| 术语 | 公式 | 要点 |
| --- | --- | --- |
| 泊松括号 | $\{f,g\} = \sum(\partial_q f\,\partial_p g - \partial_p f\,\partial_q g)$ | 相空间代数 |
| 基本括号 | $\{q_j,p_k\} = \delta_{jk}$ | 共轭对的指纹 |
| 守恒判据 | $\{f,H\} = 0$ | 不显含 t 时 |
| 正则变换 | 保持泊松括号 | 辛结构 |
| 生成函数 | $F_1(q,Q,t)$ | 构造正则变换 |
| 对易子 | $[\hat q,\hat p] = i\hbar$ | 泊松的量子版 |

泊松括号是哈密顿力学的「代数语法」：$\{f,H\} = 0$ 判守恒、正则变换保辛结构、对易子是其量子对应。它把运动方程、守恒律、坐标变换全部变成代数运算——也把经典力学与量子力学用同一个数学骨架连起来。下一节我们进入第二十一章《电动力学》，从**麦克斯韦方程组的微分形式**开始。

## 9 小结

- **泊松括号**：$\{f, g\} = \sum(\frac{\partial f}{\partial q}\frac{\partial g}{\partial p} - \frac{\partial f}{\partial p}\frac{\partial g}{\partial q})$；基本关系 $\{q_j, p_k\} = \delta_{jk}$。
- **守恒判据**：$\{f, H\} = 0$（$f$ 不显含 $t$）⟹ $f$ 守恒；运动方程 $\dot{f} = \{f, H\} + \partial f/\partial t$。
- **正则变换**：保持哈密顿方程形式的坐标变换，由生成函数产生——选择让 $K$ 简化的坐标。
- 泊松括号 ⟹ 对易子（$[f,g] = i\hbar\{f,g\}$）——经典到量子的桥梁；$[\hat{q},\hat{p}] = i\hbar$ 是不确定关系的代数根源。

在下一节，我们进入**第二十一章《电动力学》**，从**麦克斯韦方程组的微分形式**开始。
