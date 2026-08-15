---
title: 模运算与扩展欧几里得算法：线性同余方程求解
date: 2026-08-07
---

# 模运算与扩展欧几里得算法：线性同余方程求解

<div class="epigraph">
<p>欧几里得算法只告诉你公约数是多少；扩展版本还告诉你——怎么把这个公约数「组合」出来。</p>
<footer>—— 托马斯 · 科尔曼 等（Thomas H. Cormen）《算法导论》</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法设计与分析 ｜ 《算法导论》（CLRS）第 31.2–31.4 节 ｜ 2026-08-07</p>
</div>

## 为什么从扩展欧几里得开始

普通欧几里得求 $\gcd(a,b)$；**扩展欧几里得（EXTENDED-EUCLID）**更进一步，求出**贝祖系数**——使 $\gcd(a,b)$ 被表示为 $a, b$ 的整数线性组合。<span class="marginnote">这不仅是理论精致：贝祖系数直接给出<strong>模逆元</strong>（RSA 解密密钥的核心），并解决<strong>线性同余方程</strong> $ax \equiv b \pmod m$。从「求最大公约数」到「求逆元」，扩展欧几里得是密码学计算里被调用最多的原语之一。</span>

这一课讲模运算基础、扩展欧几里得的递归结构、以及它如何解线性同余方程。

## 1 模运算基础

**模（modulo）**：$a \bmod n$ = $a$ 除以 $n$ 的余数（$0 \le a \bmod n < n$）。

**同余（congruence）**：$a \equiv b \pmod n$ ⟺ $n | (a-b)$——「$a$ 与 $b$ 模 $n$ 同余」。同余是等价关系，且对加乘保持：

$a \equiv b, c \equiv d \Rightarrow a+c \equiv b+d,\; ac \equiv bd \pmod n$。

**模 $n$ 的乘法逆元**：$a^{-1}$ 使 $a \cdot a^{-1} \equiv 1 \pmod n$。**存在性定理**：$a$ 模 $n$ 有逆元 ⟺ $\gcd(a,n) = 1$。<span class="marginnote">「互素才有逆元」是模运算的分水岭：$2$ 在模 $4$ 下无逆元（$2x \equiv 1 \pmod 4$ 无解，因为左边恒偶），而 $3$ 有逆元（$3 \cdot 3 = 9 \equiv 1$）。逆元存在性由 GCD 判定，<strong>求逆元本身</strong>则由扩展欧几里得完成——判定与求解一步到位。</span>

## 2 扩展欧几里得算法

**EXTENDED-EUCLID(a, b)** 返回三元组 $(d, x, y)$，使

$$d = \gcd(a, b) = ax + by$$

```text
EXTENDED-EUCLID(a, b):
    if b = 0:
        return (a, 1, 0)                  // a = a·1 + 0·0
    (d, x′, y′) ← EXTENDED-EUCLID(b, a mod b)
    return (d, y′, x′ − ⌊a/b⌋ · y′)       // 系数回溯更新
```

**递归结构**：底层 $\gcd(a,0) = a = a\cdot 1 + 0\cdot 0$；回溯时利用「$a \bmod b = a - \lfloor a/b\rfloor b$」把系数从 $(b, a \bmod b)$ 的组合改写成 $(a, b)$ 的组合。<span class="marginnote">回溯步的代数：若 $d = bx' + (a \bmod b)y' = bx' + (a - \lfloor a/b\rfloor b)y' = ay' + b(x' - \lfloor a/b\rfloor y')$——所以新系数是 $(x, y) = (y', x' - \lfloor a/b\rfloor y')$。这个「系数更新」是扩展欧几里得的全部机关：它把「较小对的贝祖系数」逐层传回「较大对」。</span>

## 3 公式解析：模逆元的计算

求 $a$ 模 $n$ 的逆元（$\gcd(a,n) = 1$）：

$$\text{EXTENDED-EUCLID}(a, n) = (1, x, y), \qquad ax + ny = 1 \Rightarrow ax \equiv 1 \pmod n \Rightarrow x \equiv a^{-1}$$

- **第一步，跑扩展欧几里得**：得到 $(d, x, y)$，$d = 1$（互素）说明逆元存在。
- **第二步，读出逆元**：$ax + ny = 1$ 两边模 $n$：$ax \equiv 1 \pmod n$——$x$ 就是逆元。
- **第三步，规范化**：$x$ 可能是负数，取 $[0, n)$ 得到 $[0, n)$ 内的逆元。

**要点**：求逆元 = 「贝祖系数里的 $x$」+「模 $n$ 归正」。扩展欧几里得一次调用同时给出「逆元存在性」与「逆元本身」。<span class="marginnote">RSA 解密需要 $d \equiv e^{-1} \pmod{\varphi(n)}$——正是这里算的模逆元。扩展欧几里得因此是 RSA 密钥生成的固定步骤。相比「试除法找逆元」，它 $O(\log n)$ 秒级完成，是「算法效率支撑密码系统」的活例。</span>

## 4 线性同余方程

**线性同余方程** $ax \equiv b \pmod n$ 的求解：

- **无解**：$\gcd(a, n) \nmid b$。
- **$d = \gcd(a, n)$ 个解**（当 $d | b$）：先解 $a' x \equiv b' \pmod{n'}$（$a' = a/d$，$b' = b/d$，$n' = n/d$，此时 $\gcd(a', n') = 1$，有唯一解 $x_0$），则原方程的解为

$$x \equiv x_0 + k \cdot \frac{n}{d} \pmod n, \qquad k = 0, 1, \dots, d-1$$

<span class="marginnote">直觉：$ax \equiv b$ 的解数由「$a$ 与 $n$ 的公约数 $d$」决定——$d$ 越大，「$a$ 模 $n$ 的周期」越短，解的个数越多。先约分到互素情形解出 $x_0$，再按 $n/d$ 的步长生成全部解。这个「先约分再解」是处理同余方程的通用套路。</span>

**例**：$14x \equiv 30 \pmod{100}$。$\gcd(14, 100) = 2$，$2 | 30$，有 2 个解。约分得 $7x \equiv 15 \pmod{50}$，扩展欧几里得解出 $x_0$，再 $+ 50$ 得第二个解。

**辨析｜易错点：** 解个数是 $d = \gcd(a,n)$，**不是 $n$**。且「无解判定」$d \nmid b$ 要最先做——不满足就直接报无解，别硬算。逆元是「$b = 1$ 且 $d = 1$」的特例。

## 5 数值算例：完整跑一遍

**求 $11$ 模 $26$ 的逆元**（仿射密码的经典步骤）。递归调用链及每一步返回：

| 调用 | 商 $q = \lfloor a/b \rfloor$ | 返回 $(d, x, y)$ | 验证 $ax + by = d$ |
| --- | --- | --- | --- |
| EXT-E(1, 0) | — | (1, 1, 0) | $1\cdot1 + 0\cdot0 = 1$ |
| EXT-E(3, 1) | 3 | (1, 0, 1) | $3\cdot0 + 1\cdot1 = 1$ |
| EXT-E(4, 3) | 1 | (1, 1, $-1$) | $4\cdot1 + 3\cdot(-1) = 1$ |
| EXT-E(11, 4) | 2 | (1, $-1$, 3) | $11\cdot(-1) + 4\cdot3 = 1$ |
| EXT-E(26, 11) | 2 | (1, 3, $-7$) | $26\cdot3 + 11\cdot(-7) = 1$ |

最后一行 $26 \cdot 3 + 11 \cdot (-7) = 1$ 两边模 $26$：$11 \cdot (-7) \equiv 1$，所以 $11^{-1} \equiv -7 \equiv 19 \pmod{26}$——验证 $11 \times 19 = 209 = 8 \times 26 + 1$。注意逆元取的是 $y$ 系数（$b$ 模 $a$），因为这里 $b = 11$ 才是要逆的元素。<span class="marginnote">这个例子展示了「回溯更新」的每一步：返回时 $x = y'$（旧对的第二系数）、$y = x' - q\cdot y'$（旧对第一系数减商倍）。底层 EXT-E(1,0) 的 (1,1,0) 像种子一样逐层上翻，系数在每一层被 $q$ 调整。手工验一遍比背公式更能建立直觉。</span>

**再解一个线性同余方程**：$14x \equiv 30 \pmod{100}$。

- 第一步：$d = \gcd(14, 100) = 2$，且 $2 \mid 30$，故有 $d = 2$ 个解。
- 第二步：约分得 $7x \equiv 15 \pmod{50}$（$\gcd(7, 50) = 1$）。
- 第三步：$7^{-1} \equiv 43 \pmod{50}$（因 $7\times 43 = 301 = 6\times50 + 1$），故 $x_0 \equiv 15 \times 43 = 645 \equiv 45 \pmod{50}$。
- 第四步：$d = 2$ 个解为 $x \equiv 45$ 与 $45 + 50 = 95 \pmod{100}$。验证：$14 \times 45 = 630 \equiv 30$，$14 \times 95 = 1330 \equiv 30 \pmod{100}$。

## 6 小结

- **模运算**：$a \bmod n$、同余 $a \equiv b$、模逆元存在 ⟺ 互素。
- **扩展欧几里得**：返回 $(d, x, y)$ 使 $ax + by = \gcd(a,b)$；回溯更新系数。
- **模逆元**：$\text{EXTENDED-EUCLID}(a, n)$ 的 $x$ 即 $a^{-1} \bmod n$——RSA 解密密钥的核心。
- **线性同余方程** $ax \equiv b \pmod n$：$d \nmid b$ 无解；否则 $d$ 个解（约分 + 步长 $n/d$）。
- 数值算例：$11^{-1} \equiv 19 \pmod{26}$；$14x \equiv 30 \pmod{100}$ 的两解 $45, 95$。
- 复杂度 $O(\log n)$；密码学里最常被调用的数论原语之一。


**速查表**：

| 概念 | 含义 |
| --- | --- |
| 贝祖系数 $(x,y)$ | 使 $ax+by=\gcd(a,b)$ 的整数组合 |
| 回溯更新 | $x=y'$、$y=x'-\lfloor a/b\rfloor y'$ |
| 模逆元 | $ax \equiv 1 \pmod n$ 的解 $x$，存在 ⟺ $\gcd(a,n)=1$ |
| 同余方程解数 | $\gcd(a,n)=d$ 时 $d$ 个解（若 $d \mid b$） |

**辨析｜易错点：** 求逆元时注意取「$b$ 的系数」还是「$a$ 的系数」——EXT-E(a,n) 的 $y$ 系数对应 $n$，$x$ 系数对应 $a$。规范取模到 $[0,n)$ 得到标准逆元；负数逆元要加 $n$ 归正。

**数值快照**：EXT-E(26,11) 的递归链五层返回 $(1,3,-7)$，$26\cdot3+11\cdot(-7)=1$ 两边模 26 得 $11^{-1}\equiv -7\equiv 19\pmod{26}$——一次调用同时给出「逆元存在」与「逆元本身」。

在下一课，我们组合数论工具——**中国剩余定理与模幂运算**：把大模数拆成小模数，以及高效计算 $a^b \bmod m$。
