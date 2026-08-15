---
title: Dirichlet 特征与等差数列中的素数
date: 2026-08-07
---

# Dirichlet 特征与等差数列中的素数

<div class="epigraph">
<p>美是首要的检验：丑陋的数学在这个世界上没有永恒的位置。</p>
<footer>—— 戈弗雷 · 哈罗德 · 哈代（G. H. Hardy，《一个数学家的辩白》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 解析数论 ｜ Apostol《Introduction to Analytic Number Theory》Ch. 6-7 ｜ 2026-08-07</p>
</div>

## 为什么素数会「按余类分配」

从素数定理我们知道素数整体有多密，但更精细的问题是：素数在**不同的余数类**里怎么分布？例如模 4 的素数，为什么形如 $4n+1$ 与 $4n+3$ 的各占一半？更一般地，模 3 时，$3n+1$ 与 $3n+2$ 也各约一半。欧拉用初等方法证明了「无穷多个形如 $4n+1$ 的素数」和「无穷多个 $4n+3$ 的素数」，但模 $q$ 的一般情形卡了两千年。

1837 年 Dirichlet 用一个全新的对象攻下了它：**Dirichlet 特征**。这是「把等差数列里的素数问题翻译成 L-函数问题」的第一把钥匙，也是后续一整座大厦（类数公式、圆法、大筛法）的地基。

## 1 从「筛掉坏模」到特征

**等差数列中的素数定理（Dirichlet 定理）**：设 $(a, q) = 1$，则存在无穷多个素数 $p \equiv a \pmod q$。

关键在于：**必须要求 $(a,q)=1$**。若 $\gcd(a,q) = d > 1$，那么任何 $p \equiv a \pmod q$ 都能被 $d$ 整除，不可能有多于一个这样的素数。<span class="marginnote"><strong>辨析｜易错点：</strong> Dirichlet 定理不是「模 $q$ 的每个余类都有无穷多素数」——要求互素是本质的。$p \equiv 2 \pmod 4$ 只有 $p = 2$ 一个，正是因为 $(2,4) \ne 1$。这也说明了「互素」在数论里几乎永远是好东西。</span>

Dirichlet 的洞察：与其逐个处理余类 $a$，不如**把「$n$ 属于余类 $a$」这个条件分解成若干「乘法函数」的线性组合**——这些乘法函数就是特征。这个想法与 Fourier 分析同构：Fourier 把周期函数拆成不同频率的叠加，特征则把「模 $q$ 的周期函数」拆成「对乘法友好的基本波」。

## 2 Dirichlet 特征的定义

**Dirichlet 特征（模 $q$）**：一个函数 $\chi: \mathbb{Z} \to \mathbb{C}$，满足

1. **周期**：$\chi(n+q) = \chi(n)$；
2. **乘法性**：$\chi(mn) = \chi(m)\chi(n)$；
3. $\chi(n) = 0$ 当且仅当 $(n, q) > 1$；且 $\chi(1) = 1$。

把定义域限制到 $\left(\mathbb{Z}/q\mathbb{Z}\right)^{\times}$（模 $q$ 的既约剩余类群，第二级《抽象代数》已见过），$\chi$ 就是该群到 $\mathbb{C}^{\times}$ 的群同态。模 $q$ 的所有特征构成一个群（按逐点相乘），叫**特征群**，与既约剩余类群同构，故共有 $\varphi(q)$ 个特征。

**主特征（principal character）** $\chi_0$：对 $(n,q)=1$ 恒等于 1，否则为 0。它相当于「全 1 信号」，对应余类群里的单位元。其他特征叫**非主特征**。

### 例：模 4 的两个特征与 $L(1,\chi_1)=\pi/4$

抽象定义要落到手上才可信。模 $q=4$ 的既约剩余类群只有 $\{1,3\}$，故恰有 $\varphi(4)=2$ 个特征：

| $n$ | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| $\chi_0$ | 1 | 0 | 1 | 0 |
| $\chi_1$ | 1 | 0 | -1 | 0 |

$\chi_1$ 是唯一的非主实特征，而它的 $L$ 函数正是读者早就熟悉的**交错调和级数**：

$$
L(1, \chi_1) = 1 - \frac13 + \frac15 - \frac17 + \cdots = \frac{\pi}{4}
$$

这条级数显然不收敛于 $0$——于是「非主特征 $L(1,\chi)\ne0$」在模 4 的情形下一眼可见，Dirichlet 定理「$4n+1$ 与 $4n+3$ 型素数都无穷多」也被这条古老级数免费证明。<span class="marginnote">$L(1,\chi_1)=\pi/4$ 是类数公式的最简特例：它把一个超越常数 $\pi$ 与「模 4 有两个素数类」联系起来。模 3 的同类恒等式 $1-\frac12+\frac14-\frac15+\cdots=\frac{\pi}{3\sqrt3}$ 也在第 5 节「$L(1,\chi)\ne0$ ⟺ 均匀分布」那里继续上演。</span>

## 3 正交关系：把余类拆成特征

特征群的威力来自它的**正交关系**：

$$
\sum_{a \bmod q} \chi(a) = 0 \quad (\chi \ne \chi_0), \qquad
\sum_{\chi} \chi(a) = 0 \quad (a \ne 1)
$$

以及最关键的**反演公式**：

$$
\frac{1}{\varphi(q)} \sum_{\chi} \overline{\chi(a)}\, \chi(n) =
\begin{cases}
1, & n \equiv a \pmod q,\\
0, & \text{否则}
\end{cases}
\qquad \text{当 } (a,q)=1
$$

**重点：这就是「把『$n$ 落在余类 $a$』写成特征的线性组合」**——分析工具从此可以作用在单个余类上，而不是只对「全体互素类」说话。它是 Fourier 反演在数论里的孪生兄弟。

## 4 公式解析：指示函数如何拆成特征和

把上面的反演公式拆开看，它到底在做什么：

- **第一步，认结构**：$\chi$ 是 $\left(\mathbb{Z}/q\mathbb{Z}\right)^\times$ 到 $\mathbb{C}^\times$ 的群同态。由于单位根的幂可张出该群的一切「频率」，$\sum_\chi \overline{\chi(a)}\chi(n)$ 就是「在频率域里测 $n$ 与 $a$ 的相关性」。
- **第二步，验证**：若 $n \equiv a$，则 $\chi(n) = \chi(a)$，和式 $= \sum_\chi |\chi(a)|^2 = \varphi(q)$；若 $n \not\equiv a$，则 $n a^{-1} \ne 1$，而有限群上的非平凡特征和为零，和式 $= 0$。
- **第三步，归一化**：除以 $\varphi(q)$，得到一个取值 0/1 的指示函数——**一条「$n$ 在不在余类 $a$ 里」的开关**。

把这条开关插进素数计数的求和里，$\pi(x; q, a) = \sum_{p \le x, p \equiv a} 1$ 就能写成

$$
\pi(x; q, a) = \frac{1}{\varphi(q)} \sum_{\chi} \overline{\chi(a)} \sum_{p \le x} \chi(p)
$$

**辨析｜易错点：** 反演公式要求 $(a,q)=1$。若 $a$ 与 $q$ 不互素，公式根本不适用——这也再次呼应第 1 节的「互素是命门」。<span class="marginnote">这个「频率分解」的观点也是 L-函数与 Dirichlet 卷积之外理解特征的最好抓手。在第三级《信息论与傅里叶方法》里你会看到，紧群的表示论与傅里叶分析本就是同一套语言，特征是它最古老也最漂亮的案例。</span>

## 5 L-函数与 Dirichlet 定理的证明骨架

现在让分析进场。对每个特征 $\chi$ 定义 **Dirichlet L-函数**：

$$
L(s, \chi) = \sum_{n=1}^{\infty} \frac{\chi(n)}{n^s} = \prod_{p} \left(1 - \frac{\chi(p)}{p^s}\right)^{-1}, \qquad \sigma > 1
$$

由于 $\chi$ 是乘法函数，欧拉乘积自动成立。对 $L$ 取对数：

$$
\log L(s, \chi) = \sum_p \sum_{k\ge1} \frac{\chi(p^k)}{k\, p^{ks}}
$$

**Dirichlet 定理的骨架**：若 $L(1, \chi) \neq 0$ 对一切非主特征 $\chi$ 成立，则 $\sum_{p \equiv a} \frac{1}{p}$ 发散（即无穷多个这样的素数）。证明是漂亮的级数处理：把反演公式代入 $\sum_p \chi(p)/p$，$\chi = \chi_0$ 的项给出发散主项，非主特征项给出 $\log L(1,\chi)$，只要这些对数**不趋于 $-\infty$** 即可——这正是 $L(1,\chi) \neq 0$。<span class="marginnote">历史注脚：对复特征（$\chi$ 取值不是实数），$L(1,\chi) \ne 0$ 容易证；难点全在<strong>实特征</strong>（取值 $\pm 1$）上——若 $L(1,\chi)=0$，则 $\prod L(1,\chi)$ 的乘积论证会直接矛盾。这个「实特征最麻烦」的伏笔，在第七篇《零区域与 Deuring–Heilbronn 现象》会发展成 Siegal 零点理论。</span>

更精细地，Dirichlet 之后的分析把 PNT 推广到每个余类：

$$
\pi(x; q, a) \sim \frac{1}{\varphi(q)} \frac{x}{\log x} \qquad ((a,q)=1)
$$

**重点：素数不仅无穷多，而且在每个互素余类里「按比例 $\frac1{\varphi(q)}$ 均匀分布」**——这就是素数在等差数列中的均匀性定理，也是大筛法（第十一篇）要不断改进它的起点。

## 6 小结

- **Dirichlet 定理**：$(a,q)=1$ 时有无穷多个素数 $p \equiv a \pmod q$；更精确地，素数在每个互素余类中按 $\frac1{\varphi(q)}$ 均匀分布。
- **Dirichlet 特征**：模 $q$ 既约剩余类群到 $\mathbb{C}^\times$ 的群同态，共 $\varphi(q)$ 个；主特征 $\chi_0$ 对应「全 1」。
- **正交关系**把「$n$ 落在余类 $a$」写成特征的线性组合，是 Fourier 反演的数论版，也是 L-函数公式化的前提。
- **L-函数** $L(s,\chi) = \sum \chi(n)n^{-s} = \prod_p (1 - \chi(p)p^{-s})^{-1}$；Dirichlet 定理等价于非主特征 $L(1,\chi) \ne 0$。
- **实特征最难**：$L(1,\chi)=0$ 的唯一隐患来自实特征，这一伏笔通向 Siegel 零点。

在下一节，我们会仔细打量特征最锋利的一把刀——**Gauss 和**：为什么它的大小恰是 $\sqrt{q}$，以及它如何给出二次互反律的解析证明。
