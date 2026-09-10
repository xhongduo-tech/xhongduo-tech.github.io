---
title: NTT
date: 2026-09-08
section: cs
---

# NTT

<div class="epigraph">
<p>模 $p$ 下若存在 $n$ 次单位根，DFT 的加减乘全部在整数进行：卷积精确，适合大整数与多项式取模。</p>
<footer>—— 据 CLRS 第 30、31 章；数论变换见 Nussbaumer 与数论教材整理</footer>
</div>

上一课[FFT](/cs/fft)在复数上 $O(n\log n)$，浮点有误差。缺口是数论变换（NTT）：模素数 $p=k\cdot n+1$，原根 $g$，$n$ 次单位根 $\omega=g^k$。不重写蝶形结构。后课多项式求逆仍用 NTT 乘。接[有限域](/cs/finite-field-gf2n)的乘法直觉，不重写域公理。

## 问题

要 $\omega^n\equiv 1\pmod p$，且对真因子 $d\mid n$，$\omega^{n/d}\not\equiv 1$。则 Cooley–Tukey 原样，除 $n$ 改模逆。常用 $p=998244353=119\cdot 2^{23}+1$ 等。多模 + CRT 拼回大整数（Schönhage–Strassen 一类精确乘的实践版）。

缺口是模意义单位根存在条件，不是新蝶形。

### 不是任意模都能 NTT

$n$ 要整除 $p-1$。合数模没有域，不能随便当 NTT。长度不是 2 的幂可用 CRT 拼或其它原根阶。不要把 $10^9+7$ 当总能做长度为 $2^{20}$ 的 NTT——它不整除。

<span class="marginnote">NTT 是 DFT 在环 $\mathbb{Z}/p\mathbb{Z}$。CLRS 30.8 讨论数论。后课求逆：牛顿迭代 + NTT 乘。</span>

## 方法

选 $p$、$n\mid p-1$、原根。预处理 $\omega$ 幂。蝶形全模 $p$。逆变换乘 $n^{-1}$。多素数再 CRT。

```mermaid
flowchart TD
  P["p = k n + 1"] --> W["n 次单位根"]
  W --> NTT["模 p 蝶形"]
  NTT --> CONV["精确卷积 mod p"]
```

卷积长度与 $p$ 同时约束。

## 机制

单位根的几何和在模 $p$ 同样正交，故 IDFT 还原。与 FFT 比：无舍入，有模范围。与 Karatsuba：NTT 对很长多项式渐近更好，阈值仍实验。快速幂算 $\omega$ 与 $n^{-1}$。

## 边界

本课不写全部常用模列表当作业。不写 p-adic、不写 Schönhage–Strassen 的完整复杂度陈述（可点名 $O(n\log n\log\log n)$ 级）。后课默认：精确多项式乘用 NTT。下一课多项式求逆与除法。

## 小结

- 存在单位根的模上，DFT 全精确。
- $n\mid p-1$；常用 Fermat 素数型模。
- 大整数：多模 CRT。
- 出处：CLRS 第 30 章；NTT 为 DFT 的模算术形式。
