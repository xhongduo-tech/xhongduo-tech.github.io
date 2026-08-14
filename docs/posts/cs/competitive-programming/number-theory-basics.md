---
title: 数论基础：素数、GCD 与快速幂
date: 2026-08-07
---

# 数论基础：素数、GCD 与快速幂

<div class="epigraph">
<p>数学是科学的皇后，数论是数学的皇后。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss）</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法竞赛与编程实践 ｜ 刘汝佳《算法竞赛入门经典》第10章 ｜ 2026-08-07</p>
</div>

## 为什么从数论基础开始

整数里的问题，从小学数学一路延伸到密码学与公钥加密。竞赛中的**数论（number theory）** 不是「深奥的数学」，而是**几个高频工具**：判断素数、分解因数、求最大公约数、快速幂取模、模运算。它们是组合数学、概率题、密码题的公共地基——几乎所有「大数据下的整数问题」最后都会落到这几样工具上。这一章我们把这些工具的原理与实现一次打通，特别是**欧几里得算法**与**快速幂**这两个「把朴素做法加速一个量级」的典范。

## 1 素数判定：从试除到筛法

**素数（prime number）** 是大于 1 且只有 1 与自身两个因数的整数。判断单个数是否为素数，朴素做法试除所有小于它的数；但只需试到 $\sqrt{n}$——因为若 $n = a \cdot b$ 且 $a \le b$，则 $a \le \sqrt{n}$：

```cpp
bool isPrime(long long n) {
    if (n < 2) return false;
    for (long long i = 2; i * i <= n; i++)
        if (n % i == 0) return false;
    return true;
}
```

**核心概念：试除上界 $\sqrt{n}$。** 因子成对出现，较小的那个必不超过 $\sqrt{n}$——所以只要试到 $\sqrt{n}$ 就够了。单次判定 $O(\sqrt{n})$，对 $n \le 10^{12}$ 量级是可行的。

**筛法（sieve）**：要一次性筛出 $[2, N]$ 内全部素数，用**埃拉托斯特尼筛法（Eratosthenes sieve）**——从 2 开始，把每个素数的倍数全部标记为合数：

```cpp
vector<int> isPrime(N + 1, 1);
isPrime[0] = isPrime[1] = 0;
for (int i = 2; i * i <= N; i++)
    if (isPrime[i])
        for (int j = i * i; j <= N; j += i)
            isPrime[j] = 0;
```

<span class="marginnote">筛法复杂度 $O(N \log\log N)$，接近线性。从 `i * i` 开始标记（而不是 `2*i`）省去大量重复。更快的<strong>线性筛</strong>（欧拉筛）让每个合数只被最小质因子筛一次，复杂度严格 $O(N)$，配合「同时求最小质因子」还能做质因数分解。</span>

**辨析｜易错点：** 筛法的 `i * i` 在 `i` 较大时可能溢出 `int`——把循环变量与乘法都提升到 `long long`。另外，`for (int j = i*i; ...)` 的起点若写成 `2*i` 也没错只是慢，写成 `i` 则会错误地把素数本身标记成合数。

## 2 最大公约数：欧几里得算法

**最大公约数（greatest common divisor, GCD）** 用**辗转相除法（欧几里得算法）** 求解，我们在《函数与递归》里见过它的递归形态：

$$
\gcd(a, b) = \gcd(b, a \bmod b), \qquad \gcd(a, 0) = a
$$

```cpp
long long gcd(long long a, long long b) {
    return b == 0 ? a : gcd(b, a % b);
}
long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b;      // 注意先除后乘防溢出
}
```

**核心概念：余数严格缩小保证终止。** $a \bmod b \lt  b$，所以每步的第二个参数严格变小，递归必然终止。<span class="marginnote">最小公倍数 `lcm(a, b) = a * b / gcd(a, b)`——但要<strong>先除后乘</strong>，因为 `a * b` 可能溢出而 `a / gcd * b` 不会。这个「先除后乘防溢出」的细节，在模运算题里尤其重要。</span>

**进阶：扩展欧几里得（extended Euclid）**。不仅能求 $\gcd(a,b)$，还能求出一组整数解 $x, y$ 满足 $ax + by = \gcd(a,b)$——它是**乘法逆元**、一次同余方程、乃至 RSA 密钥生成的基石。方程有整数解的条件（$ax + by = c$ 有解 ⟺ $\gcd(a,b) \mid c$）由此而来。

## 3 快速幂：在 O(log n) 里算 a^n

朴素算 $a^n$ 连乘 $n$ 次是 $O(n)$；$n$ 到 $10^9$ 就不可行。**快速幂（fast exponentiation）** 把幂拆成二进制，每步平方：

$$
a^n = \begin{cases} (a^{n/2})^2 & n \text{ 为偶数} \\ (a^{(n-1)/2})^2 \cdot a & n \text{ 为奇数} \end{cases}
$$

```cpp
long long qpow(long long a, long long n, long long mod) {
    long long res = 1 % mod;
    while (n > 0) {
        if (n & 1) res = res * a % mod;   // 当前位是 1，乘上 a^{2^k}
        a = a * a % mod;                  // a 自乘，准备下一位
        n >>= 1;
    }
    return res;
}
```

- **第一步，拆幂**：把 $n$ 写成二进制，如 $13 = 1101_2$，则 $a^{13} = a^8 \cdot a^4 \cdot a^1$。
- **第二步，自乘**：每次 `a = a*a`，得到 $a^{2^k}$ 序列。
- **第三步，按位乘**：$n$ 的当前最低位是 1 就乘进结果，然后右移一位。
- **第四步，复杂度**：循环次数等于 $n$ 的二进制位数，$O(\log n)$。

**重点：快速幂是「分治」的又一化身。** 把幂次对半拆，正是我们在《分治算法》里学的递归结构；这里用迭代 + 位运算实现，避免了递归的栈开销，也天然支持取模。

## 4 公式解析：模运算的分配律

竞赛里的整数几乎都带模——`mod` 一个固定大素数（常用 $10^9+7$、$998244353$）。模运算有漂亮的分配律：

$$
(a + b) \bmod M = (a \bmod M + b \bmod M) \bmod M
$$

$$
(a \times b) \bmod M = ((a \bmod M) \times (b \bmod M)) \bmod M
$$

- **第一步，加与乘**：加法与乘法可以「先取模再算」，结果不变——这是所有「边算边取模」做法的依据。
- **第二步，注意除法**：**除法没有这种性质**。$(a / b) \bmod M \ne (a \bmod M / b \bmod M) \bmod M$——模意义下的除法要用**逆元**：$a \cdot b^{-1} \bmod M$。
- **第三步，逆元怎么来**：当 $M$ 是素数且 $\gcd(b, M) = 1$ 时，$b^{-1} \equiv b^{M-2} \pmod M$（费马小定理）——用快速幂在 $O(\log M)$ 内求出。

**辨析｜易错点：** 模运算里「减出负数」要先加回模：`(a - b) % M` 在 `a < b` 时为负，写成 `((a - b) % M + M) % M` 归一。<span class="marginnote">「取模防溢出」的典型场景：$10^9+7$ 的平方约 $10^{18}$，正好落在 `long long`（约 $9.2\times10^{18}$）范围内——所以模运算题几乎一律用 `long long` 运算，并在每一步乘法后立刻取模。费马小定理求逆元的写法 `qpow(b, M-2, M)` 是模算术题的「万能钥匙」。</span>

## 5 数论工具包一览

| 工具 | 作用 | 复杂度 | 关键公式 |
| --- | --- | --- | --- |
| 试除判素 | 单个数素性 | $O(\sqrt{n})$ | 试到 $\sqrt{n}$ |
| 埃氏筛 | 筛出 $[2,N]$ 全部素数 | $O(N\log\log N)$ | 标记素数倍数 |
| 欧几里得 | $\gcd$ / $\text{lcm}$ | $O(\log\min(a,b))$ | $\gcd(a,b)=\gcd(b,a\bmod b)$ |
| 扩展欧几里得 | $ax+by=\gcd$ 的解 | $O(\log)$ | 求逆元、同余方程 |
| 快速幂 | $a^n \bmod M$ | $O(\log n)$ | 二进制拆幂 |

**核心概念：数论工具是「组合拳」**。素数筛出素数表、质因数分解用它试除、逆元靠扩展欧几里得或费马小定理 + 快速幂、组合数取模靠预处理阶乘 + 逆元——工具链一环扣一环。遇到数论题，先识别需要哪个工具，再调用。

## 6 小结

- 素数判定试到 $\sqrt{n}$；埃氏筛 $O(N\log\log N)$ 筛区间素数，线性筛更稳。
- 欧几里得 $\gcd(a,b)=\gcd(b,a\bmod b)$，$O(\log)$；lcm 先除后乘防溢出。
- 扩展欧几里得求 $ax+by=\gcd$ 的解，是逆元与同余方程的基石。
- 快速幂二进制拆幂，$O(\log n)$，同时支持边乘边取模。
- 模运算加法乘法可分配，除法用逆元；费马小定理 $b^{-1}=b^{M-2}$