---
title: 中国剩余定理（Chinese Remainder Theorem）与模幂运算
date: 2026-08-07
---

# 中国剩余定理（Chinese Remainder Theorem）与模幂运算

<div class="epigraph">
<p>大问题拆成小问题，再按各自约束拼回来——中国剩余定理是「分治」在数论里的化身。</p>
<footer>—— 《孙子算经》，约公元 3 世纪</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法设计与分析 ｜ 《算法导论》（CLRS）第 31.5–31.6 节 ｜ 2026-08-07</p>
</div>

## 为什么从中国剩余定理开始

「今有物不知其数，三三数之剩二，五五数之剩三，七七数之剩二，问物几何？」——一千七百年前的问题，给出的是**中国剩余定理（CRT）**：把「模一个大数」的同余系统拆成「模几个互素小模数」的同余系统，分别求解再拼回。<span class="marginnote">CRT 的现代价值远超「韩信点兵」：它让「模大数」的运算可以「分而治之」——把模 $n$ 拆成互素的因子模 $n_i$，各自算、再组合，速度与并行度都大幅提升。RSA 的 CRT 加速解密、秘密共享方案都建立在它上面。</span>

这一课讲 CRT、它的「拼回公式」，以及另一个核心工具：**模幂运算**（快速幂）。

## 1 中国剩余定理

**定理（CRT）**：设 $n_1, n_2, \dots, n_k$ 两两互素，$n = n_1 n_2 \cdots n_k$。则同余系统

$$x \equiv a_i \pmod{n_i}, \qquad i = 1, \dots, k$$

在模 $n$ 下有**唯一解**。且存在一一对应：$x \bmod n \leftrightarrow (x \bmod n_1, \dots, x \bmod n_k)$。

**构造解**：对每个 $i$，设 $m_i = n/n_i$（去掉 $n_i$ 的乘积），$m_i$ 与 $n_i$ 互素，故有逆元 $c_i = m_i^{-1} \pmod{n_i}$。令

$$x = \sum_{i=1}^{k} a_i m_i c_i \pmod n$$

<span class="marginnote">构造公式的直觉：$a_i m_i c_i$ 这一项「只负责第 $i$ 个同余条件」——因为 $m_i$ 被 $n_i$ 之外的每个 $n_j$ 整除（$m_i$ 含 $n_j$ 因子），所以这一项模 $n_j$（$j \ne i$）为 0；而模 $n_i$ 时 $m_i c_i \equiv 1$，这一项 ≡ $a_i$。于是每一项只激活自己的约束，求和满足全部约束。这就是「拼回」的机制。</span>

## 2 公式解析：CRT 解的唯一性与构造

**唯一性**：若 $x, y$ 都满足所有同余，则 $x - y$ 被每个 $n_i$ 整除；因 $n_i$ 两两互素，$x-y$ 被 $n = \prod n_i$ 整除——$x \equiv y \pmod n$。

**构造正确性**：对第 $i$ 个约束，

$$x \equiv \sum_{j=1}^{k} a_j m_j c_j \equiv a_i \cdot \underbrace{m_i c_i}_{\equiv 1} + \sum_{j \ne i} a_j \underbrace{m_j c_j}_{\equiv 0} \equiv a_i \pmod{n_i}$$

- **第一步，第 $i$ 项**：$m_i c_i \equiv 1 \pmod{n_i}$（$c_i$ 是逆元）——该项贡献 $a_i$。
- **第二步，其他项**：$j \ne i$ 时 $n_i | m_j$，$m_j \equiv 0 \pmod{n_i}$——该项贡献 0。
- **第三步，求和**：$x \equiv a_i$，第 $i$ 个约束满足；对每个 $i$ 成立——全部约束满足。

**要点**：构造公式的精髓是「**每一项只激活自己的约束**」——$m_j$ 的因子结构让「别家约束」自动为 0。这是「分治拼回」的代数机制。<span class="marginnote">CRT 的计算流程：① 对每个 $i$ 算 $m_i$、$c_i$（扩展欧几里得求逆元）；② 算 $a_i m_i c_i$ 并求和取模。全部子问题都是「小模数」上的运算——这正是「大数模运算拆小」的价值：每个 $n_i$ 比 $n$ 小得多，运算更快、甚至可并行（RSA-CRT 把解密提速近 4 倍）。</span>

## 3 模幂运算：快速幂

**模幂（modular exponentiation）**：算 $a^b \bmod n$。朴素「乘 $b$ 次」是 $O(b)$——$b$ 大时不可行。**快速幂（exponentiation by squaring）** 用二进制展开把指数降到 $O(\log b)$：

$$a^b = a^{b_0 + 2b_1 + 4b_2 + \cdots} = \prod_{b_i = 1} a^{2^i}$$

```
MODULAR-EXPONENTIATION(a, b, n)
  c = 0;  d = 1
  for i from high bit down to 0 of b
    c = 2c;  d = (d*d) mod n
    if b_i == 1
      c = c + 1;  d = (d*a) mod n
  return d
```

每比特一次平方 + 至多一次乘——$O(\log b)$ 次模乘。<span class="marginnote">快速幂是「二进制展开把乘法次数从线性压到对数」的经典。RSA 的加密/解密 $c = m^e \bmod n$ 全靠它——否则指数 $e$（通常 $2^{16}+1$ 或更大）让直接幂乘完全不可行。这也是「算法设计直接支撑密码系统规模」的例子：没有 $O(\log b)$，RSA 无法实用。</span>

## 4 公式解析：快速幂为什么是 $O(\log b)$

设 $b$ 有 $\ell = \lfloor \log_2 b \rfloor + 1$ 个二进制位：

$$T(b) = \ell \cdot O(1) = O(\log b)$$

- **第一步，二进制展开**：$b = \sum b_i 2^i$，$b_i \in \{0,1\}$——共 $\ell$ 位。
- **第二步，逐位累乘**：对每位做一次平方（$d = d^2$），若 $b_i = 1$ 再乘一次 $a$——常数次模乘。
- **第三步，总次数**：$\ell = O(\log b)$ 次——**指数从 $b$ 压到 $\log b$**。

**要点**：快速幂是「分治」在指数上的应用（二分平方），与归并排序的「每层减半」同构。**「把线性扫描换成二分/二进制分解」是降低复杂度的通用杠杆**（对比二分查找、LIS 优化）。<span class="marginnote">CRT 与快速幂常配合使用（RSA-CRT）：先用 CRT 把模 $n$ 拆成 $p, q$ 两个小模数，各算快速幂，再拼回——每个快速幂的指数与模数都变小，速度翻倍。理解两者的组合，就理解了 RSA 加速的全部秘密。</span>

## 5 小结

- **CRT**：两两互素模数下，同余系统模 $n$ 有唯一解；构造公式 $x = \sum a_i m_i c_i$。
- 每一项「只激活自己的约束」（$m_j$ 的因子结构让别家为 0）——分治拼回的机制。
- **模幂 / 快速幂**：二进制展开 + 逐位平方，$O(\log b)$ 次模乘。
- CRT 拆小模数 + 快速幂快速计算 = RSA-CRT 加速解密的核心。
- 应用：RSA、秘密共享、大数模运算并行化。

在下一课，我们把数论工具推向实际——**RSA 公钥密码体制**：密钥如何生成、加密解密如何工作、为什么它安全。
