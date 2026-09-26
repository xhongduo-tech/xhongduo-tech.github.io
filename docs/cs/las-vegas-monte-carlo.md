---
title: Las Vegas 与 Monte Carlo
date: 2026-09-08
section: cs
---

# Las Vegas 与 Monte Carlo

<div class="epigraph">
<p>Las Vegas 永远正确、时间随机；Monte Carlo 限时、以小概率错。重复与双边错误把 RP/BPP 与期望多项式分开。</p>
<footer>—— 据 Motwani and Raghavan, Randomized Algorithms, 1995；[随机化算法直觉](/cs/randomized-algo)；CLRS 第 5 章整理</footer>
</div>

上一课[Karger 最小割](/cs/karger-min-cut)是典型 Monte Carlo：可能报更大的割。主干已分两类。本课缺口是**钉术语**并接上 Karger、rho、指纹：何时可变成 Las Vegas（有验证器）。不重写指示器。后课 Freivalds 指纹。

## 问题

Las Vegas：输出正确，如随机快排、随机增量几何（期望 $O(n\log n)$）。可中止重来，期望时间有界。Monte Carlo：Karger、Miller–Rabin（单侧）、指纹相等（碰撞则可能错）。单侧错误可重复压概率。有确定性多项式验证（NP 证书或最小割再检查）则 MC 可改 LV：错就重抽，直到验证过。

<span class="marginnote">术语翻译：单侧错误就是「只会错一边」。Miller–Rabin 说「这是合数」时给出的是铁证、绝不会错；只有说「这是素数」时才可能看走眼。所以错了的那一侧可以靠重复运行把概率指数压小；双侧错误（BPP）两边都可能错，就得靠多数投票。</span>

缺口是分类，不是新算法。

### 期望多项式不是最坏多项式

LV 最坏可很长。MC 最坏时间固定。不要把「平均输入」与「算法内硬币」混。

<span class="marginnote">常见误区：初学者容易把 LV 的「期望时间」当成「输入平均不坏」。实际上硬币是算法自己掷的：随机快排对任何一个固定输入——包括最坏的已排序数组——期望运行时间都是 $O(n \log n)$，期望是对算法的随机性取的，与输入分布无关。</span>

<span class="marginnote">Motwani–Raghavan。RP：单侧错；BPP 双侧。本课算法，复杂度类点名。后课 Freivalds 是 MC 矩阵乘检验。</span>

## 方法

写算法时声明：错从哪来、能否验证、重复几次。Karger：可用流验证全局割是否等于报的值（对无向）。指纹：碰撞无法本地验证相等。

```mermaid
flowchart TD
  LV["永远对，时间随机"] --> EX["期望多项式"]
  MC["限时，可错"] --> REP["重复压失败概率"]
```

错误概率对最坏输入取，不是平均实例。

## 机制

重复 $k$ 次独立，失败 $(1-p)^k$。双侧错误要 Chernoff 或多数票。与近似比：随机近似给期望比或高概率比，后课。与在线：在线没有「重复整个输入」。

<span class="marginnote">数字实例：单次错误概率 $p=\frac{1}{3}$ 时，重复 10 次独立运行、全错的概率只有 $(\frac{1}{3})^{10} \approx 1.7\times 10^{-5}$；重复 20 次就降到 $3\times 10^{-10}$。每多跑一次几乎白拿一个数量级——这是随机算法最划算的地方。</span>

<span class="marginnote">直觉类比：MC 改 LV 的条件可以想象成「对答案」。考试时先蒙一个答案（MC 运行），如果后面有标准答案可查（验证器），蒙错了就重蒙，最终交卷的一定对（LV）；如果没法对答案（指纹相等没有本地验证），就只能多蒙几遍靠概率压错误。</span>

```mermaid
flowchart TD
  S["运行一次 MC 算法"] --> V{"有没有本地验证器?"}
  V -->|"有: 如割值可用流复算"| C{"验证通过?"}
  C -->|"通过"| OUT["输出: 必正确, 已是 LV"]
  C -->|"失败"| R["重抽硬币, 再跑一次"]
  R --> S
  V -->|"没有: 如指纹相等"| ERR["无法本地纠错, 只能重复压概率"]
```

这张图回答的问题：MC 何时能改写成 LV。关键分叉是「有无本地验证器」——有则失败重抽的循环把错误彻底清零（只花时间不冒错），没有则只能靠独立重复把失败概率从 $p$ 压到 $(1-p)^k$。

## 边界

本课不证 $P=BPP?$。不写全部 RP 完全问题。后课默认：先分 LV/MC 再分析。下一课指纹与 Freivalds。

## 小结

- LV 正确；MC 可错可限时。
- 有验证器则 MC 可改 LV。
- Karger 是 MC；快排是 LV。
- 出处：Motwani and Raghavan, 1995；CLRS。
