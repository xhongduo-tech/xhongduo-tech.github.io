---
title: Pollard rho
date: 2026-09-08
section: cs
---

# Pollard rho

<div class="epigraph">
<p>伪随机迭代在模 $n$ 上走，Floyd 判圈；$\gcd(|x-y|,n)$ 给出因子，期望 $\tilde O(n^{1/4})$ 级。</p>
<footer>—— 据 Pollard, A Monte Carlo Method for Factorization, 1975；CLRS 第 31.9 节整理</footer>
</div>

上一课[BSGS](/cs/bsgs)在阶已知的群里根号搜离散对数；本课对象换成合数 $n$，目标是求非平凡因子。试除要 $O(\sqrt n)$，$n$ 上到几十位就不可行。缺口是 Pollard rho：把生日悖论搬到因子 $p$ 的环上找碰撞。本课不重写 Miller–Rabin 素性；后课 Lucas 换回组合数。

## 问题

$n=pq$ 时，迭代 $f(x)=x^2+c\bmod n$ 在模 $p$ 的投影里进入一个短环，撞环步数只有 $O(\sqrt p)$。我们看不见 $p$，但 Floyd 龟兔（一个走一步、一个走两步）无需知道环长就能找到模 $p$ 同余的相遇点 $x_i\equiv x_j$；此时 $\gcd(|x_i-x_j|,n)$ 撞出既非 $1$ 也非 $n$ 的因子。期望步数 $O(\sqrt p)$，对最小素因子即 $O(n^{1/4})$ 量级；gcd 失败（恰得 $n$）就换参数 $c$ 重来。

Brent 变体重排 gcd 时机，省掉大量 $f$ 求值；Pollard $p-1$ 是另一条路（赌 $p-1$ 光滑），点名即可。

### 不是素性测试

rho 假定 $n$ 是合数：$n$ 为素数时迭代只在模 $n$ 自身的环上走，gcd 恒为 $1$ 或 $n$，永远撞不出因子。所以流程是先 Miller–Rabin 确认合数，再跑 rho；不要对素数跑 rho 充当分解。

<span class="marginnote">Pollard 1975。CLRS 31.9。二次筛、NFS 分解大整数，本课 rho 适合 60–80 位因子量级直觉。后课 Lucas 定理。</span>

## 方法

操作序列：随机选 $x_0$ 与 $c$；推进 Floyd 循环；每步（或按 Brent 的批量节拍）算一次 gcd；拿到因子 $d$ 后对 $d$ 与 $n/d$ 递归分解。动手前先查完全平方——有平方因子先开方，免得两支相同因子空转。

```mermaid
flowchart TD
  F["x←x²+c mod n"] --> FLOYD["龟兔"]
  FLOYD --> G["gcd(|x-y|, n)"]
  G --> FAC["非平凡因子"]
```

离散对数也有 rho 版本，用陪集上的随机函数游走找碰撞，点名即可。

## 机制

机制是生日悖论在模 $p$ 上起作用：$O(\sqrt p)$ 个值就有相当概率出现一对模 $p$ 同余，碰撞差 $x_i-x_j$ 是 $p$ 的倍数而 $n$ 整体不是——gcd 一筛就是因子。Floyd 的妙处在空间 $O(1)$：不存历史，用两倍速指针替代哈希表。与 BSGS 对照鲜明：求对数要哈希表存 $\sqrt q$ 项，分解只需 gcd。rho 每次输出的因子都经 gcd 验证，不会给错答案；Las Vegas 与 Monte Carlo 的严格分类后课再钉。

## 边界

本课不写数域筛 NFS，也不展开 Lenstra 的椭圆曲线分解 ECM；量级边界在 marginnote：rho 适合中等因子，大整数要二次筛与 NFS。后课默认：中等因子用 rho。下一课 Lucas 定理。

## 小结

- $f$ 迭代 + Floyd + gcd 出因子。
- 期望依赖最小素因子的根号。
- 先确认合数。
- 出处：Pollard, 1975；CLRS 第 31.9 节。
