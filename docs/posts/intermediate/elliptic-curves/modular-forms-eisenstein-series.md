---
title: 模形式与 Eisenstein 级数
date: 2026-08-11
---

# 模形式与 Eisenstein 级数

<div class="epigraph">
<p>模形式是一类如此特殊、又如此自然的函数，以至于上帝把数论中最好的部分都写在了它们的系数里。</p>
<footer>—— 马丁 · 艾希勒（Martin Eichler）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 椭圆曲线与模形式 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从模形式开始

这一节是第二篇的真正入口：我们登上**模形式**的舞台。椭圆曲线在每一个素数处吐出一个数 $a_p$（点个数），而模形式在每个尖点处吐出一串系数（Fourier 展开）——模性定理（第 17 篇）断言这两串数**完全一致**。但在那之前，先把模形式自身的血肉长出来：定义、Eisenstein 级数、权与尖点形式。<span class="marginnote">模形式的历史贯穿整个 19—20 世纪：从 Jacobi 的 theta 函数，到 Hecke 的算子理论，再到 Wiles 证明费马大定理时「模形式与椭圆曲线一一对应」的最终形态。Serre《算术课程》第 VII 章是这个领域的标准入门，我们跟随它。</span>本节的主题：权 $k$ 的模形式的定义、Eisenstein 级数 $G_k$ 的构造、以及「系数为什么恰好是数论里的那些数」。

## 1 上半平面与 $SL_2(\mathbb{Z})$ 的作用

### 舞台：上半平面

记上半平面 $\mathfrak{h} = \{\tau \in \mathbb{C} : \mathrm{Im}\,\tau > 0\}$。它的「对称群」是模群

$$SL_2(\mathbb{Z}) = \left\{ \begin{pmatrix} a & b \\ c & d \end{pmatrix} : a,b,c,d \in \mathbb{Z},\ ad - bc = 1 \right\}$$

通过分式线性变换作用：$\gamma = \begin{pmatrix} a&b\\c&d\end{pmatrix}$ 把 $\tau$ 送到 $\gamma\tau = \frac{a\tau + b}{c\tau + d}$。<span class="marginnote">为什么这个群与椭圆曲线相关？因为 $E(\mathbb{C}) \cong \mathbb{C}/\Lambda$，$\Lambda = \mathbb{Z} + \mathbb{Z}\tau$，而「换基」$\Lambda \mapsto \Lambda'$ 恰好由 $SL_2(\mathbb{Z})$ 实现。模群是格等价类的对称群——<strong>椭圆曲线在复解析侧的对称，就是模群</strong>。</span>

### 权与不变性

**权为 $k$ 的模形式（modular form of weight $k$）** 是满足下列条件的全纯函数 $f: \mathfrak{h} \to \mathbb{C}$：

1. **模性**：$f\left(\frac{a\tau + b}{c\tau + d}\right) = (c\tau + d)^k f(\tau)$ 对所有 $\gamma \in SL_2(\mathbb{Z})$；
2. **尖点条件**：$f$ 在 $\infty$（及各尖点）处的 Fourier 展开 $f(\tau) = \sum_{n \geq 0} a_n q^n$ 只有非负幂项（$q = e^{2\pi i\tau}$）。

若还要求 $a_0 = 0$，则 $f$ 是**尖点形式（cusp form）**。<span class="marginnote">模性条件中的因子 $(c\tau+d)^k$ 是「自动因子」：它保证「$f(\tau)d\tau^{k/2}$ 是不变量」。权 $k$ 像「曲率维度」——变换时函数被「扭曲 $k$ 次」。条件 2（尖点处可展开且无负幂）是「在无穷远点无极点」的全纯性声明，它是「模形式是紧致对象上的全纯微分」这句话的坐标化。</span>

### 为什么系数是整数

对权 $k$ 的模形式，若其 Fourier 系数 $a_n$ 全为整数，则 $f$ 是「整模形式」。**Eisenstein 级数的系数正是著名的数论函数**（除数函数），这让模形式成为「生成函数」的工厂。

## 2 Eisenstein 级数：模形式的「种子」

### 定义

对权 $k \geq 4$（偶数），定义 **Eisenstein 级数**

$$G_k(\tau) = \sum_{\substack{(m,n) \in \mathbb{Z}^2 \\ (m,n) \neq (0,0)}} \frac{1}{(m\tau + n)^k}$$

它是权 $k$ 的模形式，也是**一切模形式的「加法基础」**：权 $k$ 的模形式空间维数有限，而 $G_k$ 往往是空间的「骨架」。

**重点：$k \geq 4$ 时级数绝对收敛**（因为 $\sum 1/|m\tau+n|^k$ 当 $k > 2$ 收敛）。$k = 2$ 时级数不绝对收敛（条件收敛给出「准模形式」，权 2 是特殊的、有缺陷的），$k$ 为奇数时恒为零（$m,n$ 取反相消）。<span class="marginnote">「权 2 的特殊性」是模形式理论里的著名暗礁：$G_2$ 不是模形式（差一个常数项），它是 quasi-modular。它对应到「椭圆曲线对模曲线的某些退化」，在微分方程与 Hecke 理论里反复出现。知道「$k=2$ 有鬼」比知道「$k\ge4$ 正常」更重要。</span>

### Fourier 展开与系数

记 $q = e^{2\pi i\tau}$，Eisenstein 级数有干净展开：

$$G_k(\tau) = 2\zeta(k) + 2\cdot\frac{(2\pi i)^k}{(k-1)!} \sum_{n \geq 1} \sigma_{k-1}(n)\, q^n$$

其中 $\zeta$ 是 Riemann zeta 函数，$\sigma_{k-1}(n) = \sum_{d \mid n} d^{k-1}$ 是**除数函数**。系数全部由除数函数生成——**模形式的系数是「数论信号的母函数」**。

## 3 模形式空间：有限维与小维数

### 空间结构

记 $M_k = \{\text{权 } k \text{ 的模形式}\}$，$S_k = \{\text{尖点形式}\}$。它们是有限维 $\mathbb{C}$-向量空间，维数公式：

$$\dim M_k = \begin{cases} \left\lfloor \frac{k}{12} \right\rfloor & k \equiv 2 \pmod{12} \\[4pt] \left\lfloor \frac{k}{12} \right\rfloor + 1 & \text{其他（} k \equiv 0, 4, 6, 8, 10 \text{）} \end{cases} \qquad (k \text{ 偶数})$$

**直观：权越大，自由度越多，但「12 个一次」地增长**——这个「12」来自模曲线的亏格公式（$g = \lfloor k/12 \rfloor$ 相关的 Riemann-Roch 计算），是「几何决定维数」的又一例。<span class="marginnote">为什么是 12？因为模曲线 $X(1)$ 的亏格为 0，而「典范除子」的度是 $-12$——Riemann-Roch 里那个「$1-g$」在这里变成「12 的周期性」。这个 12 与「$j$-不变量的常数项 744」「$e^{\pi\sqrt{163}}$ 逼近整数」同源：都是「模群算术地嵌入」的结果。</span>

### 低权全貌

- 权 4：$M_4 = \mathbb{C}\cdot G_4$，一维。
- 权 6：$M_6 = \mathbb{C}\cdot G_6$，一维。
- 权 8：$M_8 = \mathbb{C}\cdot G_8 = \mathbb{C}\cdot G_4^2$——**$G_8$ 与 $G_4^2$ 成比例**（维数 1，只能线性相关）。这是「不同 Eisenstein 级数之间存在代数关系」的最小例子，通向 $j$-不变量（$j = 1728\,G_4^3/\Delta$）。
- 权 12：$M_{12} = \mathbb{C}\cdot G_{12} \oplus \mathbb{C}\cdot \Delta$，维数 2；$\Delta$ 是**唯一的权 12 尖点形式**，其 $q$-展开从 $q$ 开始：$\Delta = q \prod_{n \geq 1}(1-q^n)^{24}$（Ramanujan 的 $\tau$ 函数承载的正是这个）。

**辨析｜易错点：** 权 $k$ 为**奇数**时，模形式空间维数为 0（除非退化），因为 $f \mapsto (-1)^k f$ 的矛盾。初学常把「权 8 与权 4 的平方比例」当成「两件独立的事」——**维数公式告诉你空间只有一个方向，任何两个权 8 模形式必成比例**。用维数说话，比逐一验证省力得多。

## 4 公式解析：Eisenstein 级数如何变成除数函数

把「二重求和」变成「除数函数」，是模形式理论里最具代表性的计算。拆成四步。

$$
G_k(\tau) = 2\zeta(k) + 2\cdot\frac{(2\pi i)^k}{(k-1)!} \sum_{n\geq 1} \sigma_{k-1}(n)\, q^n
$$

- **第一步，分离常数项**：$(m,n) = (0,0)$ 项单独拿掉后，$G_k$ 的常数项是 $2\zeta(k)$——两个非零的 $n$ 向 $m$ 求和给出 $2\zeta(k)$（因为 $(m,n)$ 与 $(-m,-n)$ 成对）。
- **第二步，化到单重求和**：对固定 $N = m\tau + n$ 的不同格点重新分组。利用「泊松求和」或「分解为对 $n$ 的求和」，把「对格点求和」变成「对 $n$ 求和」：
$$G_k(\tau) = 2\zeta(k) + 2\sum_{n \geq 1} \sum_{d \mid n} d^{k-1} q^{n}$$
- **第三步，辨认除数函数**：内层 $\sum_{d \mid n} d^{k-1} = \sigma_{k-1}(n)$。于是系数序列就是 $\sigma_{k-1}(n)$——**一个纯算术的「数因子」函数**。模形式与数论在此第一次「对表」。
- **第四步，范数校正**：系数前还挂着一个「常数 $2(2\pi i)^k/(k-1)!$」，它来自「把 $\csc$/Bernoulli 展开嵌进求和」的规范化。约去后得到的**规范 Eisenstein 级数** $E_k = G_k/(2\zeta(k))$ 的系数是整数：
$$E_4 = 1 + 240\sum_{n\geq1}\sigma_3(n)q^n, \qquad E_6 = 1 - 504\sum_{n\geq1}\sigma_5(n)q^n$$
正是这些整系数让「模形式 ↔ 数论」的对应精确到「每个素数」。

## 5 从 Eisenstein 到整个空间

- **乘积与代数结构**：$M_k$ 由 $G_4, G_6$ 的单项式张成（$M = \mathbb{C}[G_4, G_6]$）——模形式环是一个自由多项式环。这是「一切模形式都由两个 Eisenstein 种子长出来」的代数表述。
- **尖点形式**：$S_k = \Delta \cdot M_{k-12}$，$\Delta$ 是「唯一权 12 尖点形式」。它像「分母」：乘上 $\Delta$ 把「非尖点」变成「尖点」。Ramanujan 的 $\tau(n)$（$\Delta$ 的系数）满足乘性：$\tau(mn) = \tau(m)\tau(n)$ 当 $\gcd(m,n)=1$——**这是一切 Hecke 理论（第 12 篇）的起点**。
- **与前文的联系**：$j = 1728\,G_4^3/\Delta$ 把「两个 Eisenstein 种子」和「唯一尖点形式」合成一个模函数——$j$-不变量是模函数（权 0）而非模形式，第 11 篇将正式登场。

### 补充：权 2 的例外——准模形式 $G_2$

$k = 2$ 是 Eisenstein 级数公式的「边界」：$\sum 1/(m\tau+n)^2$ 不绝对收敛，只能条件收敛，且 $G_2$ 不是模形式——它是**准模形式（quasi-modular form）**：

$$G_2\!\left(\frac{a\tau+b}{c\tau+d}\right) = (c\tau+d)^2 G_2(\tau) - 2\pi i\,c\,(c\tau+d)$$

差一项「$2\pi ic(c\tau+d)$」。它的 Fourier 展开（注意符号）：

$$G_2(\tau) = 2\zeta(2) - 8\pi^2 \sum_{n\geq 1} \sigma_1(n)\, q^n = \frac{\pi^2}{3} - 8\pi^2(q + 3q^2 + 4q^3 + 7q^4 + \cdots)$$

**为什么这个「例外」如此重要**：微分算子 $q\frac{d}{dq}$ 作用在权 $k$ 模形式上得到的不是模形式（权变成 $k+2$，但「谱」被 $G_2$ 污染）——Ramanujan 的著名恒等式

$$q\frac{d}{dq}E_4 = \frac{E_2 E_4 - E_6}{3}, \qquad q\frac{d}{dq}E_6 = \frac{E_2 E_6 - E_4^2}{2}$$

正是「用 $G_2$（即 $E_2$）修正微分运算」的雏形。它在第 12 篇 Hecke 算子、以及椭圆曲线对模曲线「推前」的理论里反复出现——**权 2 的「失败」其实是「成功」的另一面。**

## 6 小结

- **模形式**：上半平面全纯、按 $(c\tau+d)^k$ 变换、尖点处无负幂的函数；系数是它的「算术签名」。
- **Eisenstein 级数** $G_k = \sum_{(m,n)} (m\tau+n)^{-k}$ 是模形式的种子，$k\ge 4$ 绝对收敛。
- **系数 = 除数函数**：$G_k$ 的 $q$-展开系数是 $\sigma_{k-1}(n)$，规范后为整数。
- 空间维数 $\dim M_k = \lfloor k/12 \rfloor + 1$（$k \not\equiv 2$），**12 来自模曲线的几何**。
- 环结构 $M = \mathbb{C}[G_4, G_6]$，尖点形式由 $\Delta$ 乘出；$\Delta$ 的系数是 Ramanujan $\tau$ 函数——**Hecke 理论的伏笔**。

在下一节，我们把「权重」降到 0、把 Eisenstein 与 $\Delta$ 组装起来：**j-不变量与模曲线**——为什么 $j$ 是「格的身份号」，而模曲线 $X_0(N)$ 正是椭圆曲线带 level 结构的「分类空间」。
