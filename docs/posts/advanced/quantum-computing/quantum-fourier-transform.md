---
title: 量子傅里叶变换（QFT）及其实现
date: 2026-08-07
---

# 量子傅里叶变换（QFT）及其实现

<div class="epigraph">
<p>傅里叶变换是数学的通用语言；量子傅里叶变换则是量子算法的心脏。</p>
<footer>—— 基塔耶夫（Alexei Kitaev）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§5.1 ｜ 2026-08-07</p>
</div>

## 为什么从 QFT 开始

上一节末尾我们留下一条谱系：Hadamard 是 $2\times2$ 的傅里叶变换，而 **QFT（quantum Fourier transform）** 是它到 $N$ 维的推广。Shor 算法、相位估计、以及很多量子算法都以 QFT 为核心引擎。<span class="marginnote">经典 FFT 把 $N$ 个数的离散傅里叶变换做到 $O(N\log N)$ 时间；量子版本的神奇之处是：作用在 $n = \log N$ 个比特上，用 $O(n^2) = O(\log^2 N)$ 个门完成同样的「变换」，对<strong>指数多个振幅同时运算</strong>。本节把它的定义、线路与复杂度一次讲清。</span>理解 QFT 只需要线性代数 + 一点群论直觉，它是整个量子算法篇的「分水岭」——从 QFT 之后，算法开始变得「真刀真枪」。

## 1 QFT 的定义

设 $N = 2^n$。**量子傅里叶变换**是 $\mathbb{C}^{N}$ 上的酉变换，定义在计算基上为

$$
QFT_N \lvert j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i \, jk / N} \lvert k\rangle, \qquad j = 0, 1, \dots, N-1
$$

其中 $j$、$k$ 是把 $n$ 比特二进制串当成的整数。<span class="marginnote">对比经典 DFT：$X_k = \frac{1}{\sqrt{N}}\sum_j x_j e^{2\pi i jk/N}$。经典 DFT 输入一组数、输出一组数；QFT 输入一个态、输出一个态，而「输入态在计算基上的 $N$ 个振幅」与「输出态的振幅」之间正是 DFT 关系——QFT 是在「振幅的向量空间」上做 DFT。</span>它是酉变换（系数矩阵是酉矩阵），故可逆，逆变换是

$$
QFT_N^{-1}\lvert k\rangle = \frac{1}{\sqrt{N}}\sum_j e^{-2\pi i jk/N}\lvert j\rangle
$$

## 2 有效线路：为什么只要 $O(n^2)$ 个门

关键结论：**QFT 可以只用 $O(n^2)$ 个基本门实现**，而不是 $O(N)$。线路由两类门构成：

- **Hadamard 门**：每个比特一个，共 $n$ 个。
- **受控相位门**：$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^k} \end{pmatrix}$ 及其受控版本，共 $\frac{n(n-1)}{2}$ 个。

线路上，对第 $j$ 个比特：先 $H$，然后依次对 $j+1, j+2, \dots, n$ 施加受控 $R_2, R_3, \dots$，最后对前 $n/2$ 个比特做「反转」（swap）。<span class="marginnote">门数合计 $\frac{n(n-1)}{2} + n = O(n^2)$，深度 $O(n)$（前面《线路的深度、宽度与复杂度》已算过）。若允许近似，门数还能压到 $O(n\log n)$（用截断的旋转门）——工程上 Shor 的线路常这样做。</span>

## 3 公式解析：QFT 的二进制分解

定义式 $e^{2\pi i jk/2^n}$ 看似笨重，但把 $j$、$k$ 写成二进制后，它惊人地变简单。设 $j = j_1 j_2 \dots j_n$（$j_1$ 最高位），$0.j_{l} \dots j_n$ 表示二进制小数 $\frac{j_l}{2} + \cdots + \frac{j_n}{2^{n-l+1}}$。则

$$
QFT_N\lvert j\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{l=1}^{n} \left( \lvert0\rangle + e^{2\pi i \, 0.j_{n-l+1}\cdots j_n} \lvert1\rangle \right)
$$

- **第一步，拆开相位**：$e^{2\pi i jk/2^n} = \prod_{l} e^{2\pi i j k_l / 2^{n-l+1}}$，把相位按 $k$ 的每个比特拆成因子。
- **第二步，合并到各比特**：每个 $k_l$ 只影响第 $l$ 个输出比特，于是张量积拆开：第 $l$ 个比特的系数是 $e^{2\pi i j \cdot 2^{l-1}/2^n}$。
- **第三步，二进制小数**：$j \cdot 2^{l-1}/2^n = 0.j_{n-l+1}\cdots j_n$（把 $j$ 的末 $l$ 位看成二进制小数）。<span class="marginnote">这条分解式是 QFT 线路的蓝图：<strong>每个输出比特只依赖输入 $j$ 的一个「后缀」</strong>。受控相位门 $R_k$ 的作用，就是把「后续比特的后缀」逐步叠进当前比特的相位——复杂度因此从「$N$ 维矩阵乘法」塌缩成「$n^2$ 个门」。</span>

## 4 QFT 的两大性质：周期性提取与谱集中

**性质一（周期检测）**：若输入是周期为 $r$ 的均匀叠加 $\frac{1}{\sqrt{M}}\sum_{m=0}^{M-1}\lvert j_0 + mr\rangle$（$M = N/r$），则 QFT 输出集中在 $\frac{N}{r}$ 的整数倍的频率上。这是 Shor 相位估计的基础。

**性质二（单位根的正交性）**：$\frac{1}{N}\sum_{j=0}^{N-1} e^{2\pi i (k-k')j/N} = \delta_{kk'}$。这让 QFT 的逆变换精确、无损——QFT 是酉变换的直接推论。<span class="marginnote">这两条性质一「实用」、一「保证正确」。周期检测把「找周期」翻译成「找谱峰」，而正交性保证测量能可靠读出谱峰位置——两者合起来，QFT 成为「相位→频率」的翻译机。</span>

**辨析｜易错点：** QFT **不是**把「$N$ 个数」快速求和的算法——它作用在**态的振幅**上，不直接作用在经典数据上。把 QFT 当成「量子数据库的 FFT」是常见误解。它真正的角色是「基变换」：把「位置基」变到「频率基」，让周期性结构在频率基下显露出来。

## 5 QFT 的应用版图

QFT 是量子算法库的核心组件：

**相位估计（phase estimation）**：用 QFT 把特征相位「读出来」，是 Shor、量子化学（VQE 的解析后处理）、量子模拟的共同子程序。
**Shor 算法**：用 QFT 找模幂函数的周期，从而分解大数。
**隐藏子群问题（HSP）**：Abel 群上的 HSP 全部以 QFT 为核心——Simon 是 $\mathbb{Z}_2^n$ 的特例，Shor 是 $\mathbb{Z}_N$ 的特例。<span class="marginnote">更深刻的事实：Abel 隐藏子群问题几乎「等于」QFT 的应用；而求解非 Abel 隐藏子群（如对称群上的图同构）仍是开放问题——QFT 家族的能力边界，正是量子算法的前沿之一。</span>

## 6 小结

- **QFT 定义**：$QFT_N\lvert j\rangle = \frac{1}{\sqrt N}\sum_k e^{2\pi i jk/N}\lvert k\rangle$，是振幅空间上的 DFT，酉可逆。
- **线路**：$n$ 个 $H$ + $\frac{n(n-1)}{2}$ 个受控相位，共 $O(n^2)$ 门、$O(n)$ 深度。
- **二进制分解**：每个输出比特只依赖输入后缀 $0.j_{n-l+1}\cdots j_n$，复杂度因此塌缩。
- 两大性质：**周期检测**（谱集中）+ **正交性**（逆变换精确）。
- 应用：相位估计、Shor、Abel 隐藏子群问题。

在下一节，我们用 QFT 搭建最通用的量子算法子程序——**相位估计（phase estimation）**，它把「酉算符的特征相位」精确读出。
