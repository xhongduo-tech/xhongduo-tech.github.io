---
title: 欧几里得与扩展欧几里得
date: 2026-09-08
section: cs
---

# 欧几里得与扩展欧几里得

<div class="epigraph">
<p>$\gcd(a,b)=\gcd(b,a\bmod b)$；回溯系数得 $ax+by=\gcd$。$\gcd(a,n)=1$ 时 $x$ 就是模 $n$ 的逆。</p>
<footer>—— 据 Euclid, Elements 卷 VII；Knuth, TAOCP 卷 2 整理</footer>
</div>

上一课[模运算](/cs/modular-arithmetic) 留下 $ax\equiv 1\pmod n$。缺口是**欧几里得算法**与扩展形式。不重写剩余类。复杂度：除法步数 $O(\log\min(a,b))$（最坏 Fibonacci）。

## 问题

$\gcd$ 与最小正线性组合重合：存在 $x,y$，$ax+by=\gcd(a,b)$（Bézout）。扩展欧几里得在递归同时带回 $x,y$。$\gcd(a,n)=1\iff a$ 在 $\mathbb{Z}/n\mathbb{Z}$ 可逆，$a^{-1}\equiv x\pmod n$。$\gcd\gt 1$ 则 $ax\equiv 1$ 无解；$ax\equiv b$ 有解当 $\gcd\mid b$。

二进制 gcd、Lehmer 加速是实现，本课标准除法版。

### 不是「约分分数」的小学课

对象是任意大整数，密码里几百比特。算法要停在 $\log$ 步，不能试除到 $\sqrt a$ 来求 gcd。

<span class="marginnote">欧几里得原本对线段。Knuth 分析步数与 Fibonacci 最坏。本课不证 Lamé。RSA 解密指数 $d$ 用扩展欧几里得求 $e$ 对 $\varphi(n)$ 的逆，后课欧拉。</span>

## 方法

手算 $\gcd(240,46)$ 并列 $x,y$。写出 $a^{-1}\bmod n$ 当且仅当互素。点名：求逆失败即发现与 $n$ 不互素——RSA 模上这几乎是分解，实践中信息位随机不会碰到。

```mermaid
flowchart TD
  AB["a, b"] --> EUC["辗转相除"]
  EUC --> G["gcd"]
  EUC --> BEZ["Bézout x, y"]
  BEZ --> INV["gcd=1 ⇒ 模逆"]
```

<span class="marginnote">数字实例：$240=5\times46+10$，$46=4\times10+6$，$10=1\times6+4$，$6=1\times4+2$，$4=2\times2+0$，停，得 $\gcd=2$。因为 $2\neq1$，240 对模 46 没有乘法逆——「求逆失败」就是这样在几步之内被发现的，不用试除。</span>

## 机制

有了逆，乘法群 $(\mathbb{Z}/n\mathbb{Z})^\times$ 才真的是群（下一课欧拉函数数它的阶）。高斯消元在模 $p$ 上需要每步求逆。扩展欧几里得也是格基、RSA CRT 实现的组件。

```mermaid
flowchart TD
  S1["240 = 5×46 + 10"] --> S2["46 = 4×10 + 6"]
  S2 --> S3["10 = 1×6 + 4"]
  S3 --> S4["6 = 1×4 + 2"]
  S4 --> S5["4 = 2×2 + 0，余数为 0"]
  S5 --> G2["除数 2 就是 gcd"]
  G2 --> B2["回代各步得 Bézout 系数"]
```

<span class="marginnote">直觉类比：辗转相除像用两把长短不同的尺子互相量——每次量不完剩下的那截，就是下一次的新尺子；哪一步恰好量尽，最后那把尺子的长度就是 gcd。余数一轮比一轮小、又非负，所以必然在有限步内停机，这就是对数步数的来源。</span>

不要用费马小定理 $a^{p-2}$ 当本课主算法——那要 $p$ 素数且更贵。

Lamé：最坏步数由连续 Fibonacci 达到。二进制 gcd 用移位，硬件友好。模逆失败返回 $\gcd\neq 1$：RSA 随机消息碰到 $p$ 的概率可忽略。线性方程 $ax\equiv b\pmod n$ 有 $\gcd$ 个解当 $\gcd\mid b$，解之间差 $n/\gcd$。


## 边界

本课不引入连分数攻击，不证素性。后课默认：互素则可逆，逆由扩展欧几里得。下一课费马与欧拉，给指数运算减指数。

Bézout 系数就是模逆（当 gcd 为 1）。算法对数步，适合密码整数。费马 $a^{p-2}$ 也能求逆，但更贵且要素数；本课主算法是扩展欧几里得。

<span class="marginnote">常见误区：初学者以为 $ax\equiv b\pmod n$ 要么无解要么唯一——其实当 $\gcd\mid b$ 时恰有 $\gcd$ 个解，相邻解差 $n/\gcd$。例如 $2x\equiv4\pmod6$ 有 $x=2,5$ 两个解；而 $2x\equiv1\pmod4$ 因 $\gcd=2\nmid1$ 无解。</span>

## 小结

- 欧几里得算 gcd；扩展给出 Bézout 系数。
- $\gcd(a,n)=1\iff$ 模逆存在。
- 步数对数，适合大整数。
- 出处：Euclid；Knuth, TAOCP 卷 2。
