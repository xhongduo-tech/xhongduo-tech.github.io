---
title: 主定理
date: 2026-09-08
section: cs
---

# 主定理

<div class="epigraph">
<p>形如 $T(n)=aT(n/b)+f(n)$ 的递归，其渐近由 $f$ 与 $n^{\log_b a}$ 谁主谁从决定。</p>
<footer>—— 据 Bentley, Haken and Saxe, A General Method for Solving Divide-and-Conquer Recurrences, 1980；CLRS 第 4 章整理</footer>
</div>

上一课[循环不变式](/cs/loop-invariant)给了迭代过程的正确性骨架。[渐近记号](/cs/asymptotic-notation)早已能写 $\Theta$。本课不重证大 $O$，也不把递归树画成百科。缺口是：后课分治会留下 $T(n)=aT(n/b)+f(n)$，循环不变式推不出这根式子的阶。本课只钉主定理的三种情形，把分治范式本身留给下一课。

## 问题

归并、二分、Strassen 乘法都把规模 $n$ 切成 $a$ 块、每块约 $n/b$，合并花 $f(n)$。递归树的层数是 $\Theta(\log_b n)$，叶子是 $n^{\log_b a}$。总代价是根上的 $f$、中间层、叶子三者之一主导。没有比较 $f$ 与 $n^{\log_b a}$ 的规则，每道题都要重画树。

缺口因此不是「递归是什么」，而是一条够用的判定：$f$ 多项式地小于、等于、或大于叶子阶时，$T(n)$ 分别是 $\Theta(n^{\log_b a})$、$\Theta(n^{\log_b a}\log n)$、$\Theta(f(n))$。规律性条件（对某个 $c\lt 1$ 有 $a f(n/b)\le c f(n)$）管第三种，本课点名不展开证明。

### 主定理不是所有递归

$T(n)=T(n-1)+\Theta(n)$ 不是等分，主定理不适用；那是等差求和。$T(n)=T(\lfloor n/2\rfloor)+T(\lceil n/2\rceil)+\Theta(n)$ 要地板函数引理，CLRS 用替换法补。本课默认 $n/b$ 写法已经光滑。

<span class="marginnote">Akra–Bazzi 处理不等分；本课不启用。Bentley–Haken–Saxe 的递归树是同一套几何级数比较。后课只引用三种情形的结论。</span>

## 方法

先算临界指数 $\log_b a$。把 $f(n)$ 与 $n^{\log_b a}$ 比多项式因子：若 $f=O(n^{\log_b a-\varepsilon})$ 对某 $\varepsilon\gt 0$，叶子赢；若 $f=\Theta(n^{\log_b a})$，每层同阶，乘 $\log n$；若 $f$ 更大且规律，根上的 $f$ 赢。

<span class="marginnote">数字实例：归并排序 $T(n)=2T(n/2)+n$，先算 $\log_2 2=1$；$f(n)=n=n^1$ 与叶子同阶，落情形二，直接得 $\Theta(n\log n)$。拿到递归先代一次这个比值，就知道查哪一行，不必真画树。</span>

```mermaid
flowchart TD
  REC["T(n)=a T(n/b)+f(n)"] --> CMP["比较 f 与 n^{log_b a}"]
  CMP --> C1["情形一：叶子主导"]
  CMP --> C2["情形二：层数乘等阶"]
  CMP --> C3["情形三：合并主导"]
```

替换法仍是后盾：猜对阶再归纳。主定理只是常见等分的快捷方式。本课不把主定理证完；树的层和几何比在 CLRS 4.5–4.6。

## 机制

有了三种情形，后课写归并 $T(n)=2T(n/2)+\Theta(n)$ 直接落在情形二，$\Theta(n\log n)$；二分查找 $T(n)=T(n/2)+\Theta(1)$ 是情形二的退化，$\Theta(\log n)$。正确性仍走递归假设或循环不变式，本课只管代价。

判断落在哪种情形之后，可以再问一句：递归树上各层的代价随深度怎么变？这决定总账记在哪一层。

```mermaid
flowchart TD
  ROOT["根：一份 f(n)"] --> MID["中间层：a^j 份 f(n/b^j)"]
  MID --> LEAF["底层：约 n^(log_b a) 片叶子"]
  ROOT --> C3["合并主导：总和约 f(n)"]
  MID --> C2["每层同阶：总和乘 log n"]
  LEAF --> C1["叶子主导：总和约 n^(log_b a)"]
```

<span class="marginnote">直觉类比：把递归树想成一座金字塔——顶层是根上的合并费 $f(n)$，底层是 $n^{\log_b a}$ 片叶子。哪一层体积最大，总造价就记哪层的账；各层一样厚时，多出来的 $\log n$ 层数就是那个乘法因子。</span>

不要把 $\log$ 底写进 $\Theta$：底是常数。也不要把 $a$、$b$ 当输入规模；它们是算法切法的参数。

<span class="marginnote">常见误区：初学者见到 $\log_2 n$、$\log_{10} n$ 想换算底数。由换底公式 $\log_b n=\log n/\log b$，底只差常数倍，在 $\Theta$ 里被吞掉；真正改变阶的是 $a$ 与 $b$ 的组合，比如 $a$ 翻一倍可能把阶抬高一整档。</span>

## 边界

主定理不处理减而治之、不处理期望递归、不处理最坏与平均混写。快排的期望是[后课](/cs/quicksort-expected)另开的指示器分析。不规则 $f$（在临界附近振荡）可以掉出三种情形，那时退回递归树或 Akra–Bazzi。

后课默认：看到等分递归，先套主定理再谈实现。分治作为设计法，下一课才命名。

## 小结

- $T(n)=aT(n/b)+f(n)$ 由 $f$ 与 $n^{\log_b a}$ 的多项式比较分三种情形。
- 不是所有递归都能套；减一、不等分要另法。
- 本课只给阶，不给分治的正确性模板。
- 出处：Bentley, Haken and Saxe, 1980；CLRS 第 4 章。
