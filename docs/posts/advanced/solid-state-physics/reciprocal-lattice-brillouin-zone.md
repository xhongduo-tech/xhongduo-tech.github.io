---
title: 倒格子与布里渊区
date: 2026-08-07
---

# 倒格子与布里渊区

<div class="epigraph">
<p>倒格子不是数学的虚构，而是晶体的傅里叶灵魂。</p>
<footer>—— 据 L. 布里渊（Léon Brillouin，《Wave Propagation in Periodic Structures》）</footer>
</div>

<div class="article-byline">
<p>第四级 · 固体物理 ｜ Kittel 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从倒格子开始

晶体是周期结构，而**周期性结构的天然语言是傅里叶分析**。实空间的晶体我们看得见；但对物理学家来说，真正好用的是一个「影子空间」——**倒空间（reciprocal space）**。在这里，晶体的格点变成一组离散的波矢（倒格矢），晶体的「密度起伏」变成一个个傅里叶分量。衍射实验测的、能带理论算的、声子色散画出来的，统统住在倒空间里。**布里渊区（Brillouin zone）**就是倒空间里划出来的基本单元：一块晶体的一切波（电子波、声子）行为，都被折叠进这一个区域来讨论。

从主线看，这一步是把「从极限到大模型」里傅里叶分析的直觉移植到固体：采样定理里的 Nyquist 频率对应布里渊区边界，混叠对应能带折叠，离散傅里叶变换对应倒格子。**你在信号处理里学过的所有「频域」知识，到这里会逐一找到物理解释。**<span class="marginnote">数字音频采样率的一半就是 Nyquist 频率，高于它的成分会被「混叠」回低频；电子在晶体里的波函数也是如此：<strong>波矢 k 超出第一布里渊区的部分，会被周期性折叠回区内</strong>。两者是同一个定理的两个名字。</span>

## 1 周期性函数的傅里叶语言

设晶体的某物理量（如电子密度）是周期函数 $n(\mathbf{r})$，满足 $n(\mathbf{r}+\mathbf{R}) = n(\mathbf{r})$ 对一切布拉伐格矢 $\mathbf{R}$ 成立。与一维周期函数一样，它可以展开成傅里叶级数：

$$n(\mathbf{r}) = \sum_{\mathbf{K}} n_{\mathbf{K}}\, e^{i\mathbf{K}\cdot\mathbf{r}}$$

这里的求和指标 $\mathbf{K}$ 是一组离散波矢。要让 $n(\mathbf{r})$ 保持平移周期性，必须对每个格矢 $\mathbf{R}$ 都有：

$$e^{i\mathbf{K}\cdot\mathbf{R}} = 1 \qquad (\text{对一切 }\mathbf{R} \in \text{布拉伐格子})$$

满足这一条件的全部 $\mathbf{K}$ 构成的集合，就叫**倒格子（reciprocal lattice）**。它是实空间格子在傅里叶意义下的「镜像」。

![实空间格子（左）与倒格子（右）：倒格点间距为 2π/a，第一布里渊区是倒空间的 Wigner-Seitz 原胞](/images/solid-state-physics/reciprocal-lattice-brillouin-zone-1.svg)

## 2 倒格基矢：实空间与倒空间的乘法表

如何具体构造倒格子？设实空间基矢为 $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$，体积 $V_c = |\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)|$。**倒基矢（reciprocal basis vectors）**由下式定义：

$$\mathbf{b}_1 = 2\pi\,\frac{\mathbf{a}_2 \times \mathbf{a}_3}{V_c}, \qquad
\mathbf{b}_2 = 2\pi\,\frac{\mathbf{a}_3 \times \mathbf{a}_1}{V_c}, \qquad
\mathbf{b}_3 = 2\pi\,\frac{\mathbf{a}_1 \times \mathbf{a}_2}{V_c}$$

关键性质是正倒基矢的**正交归一**关系：

$$\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\,\delta_{ij}$$

任何一个倒格矢都能写成整数组合 $\mathbf{K} = h\mathbf{b}_1 + k\mathbf{b}_2 + l\mathbf{b}_3$。注意验证：对任意格矢 $\mathbf{R} = n_1\mathbf{a}_1+n_2\mathbf{a}_2+n_3\mathbf{a}_3$，$\mathbf{K}\cdot\mathbf{R} = 2\pi(hn_1 + kn_2 + ln_3)$ 确为 $2\pi$ 的整数倍，即 $e^{i\mathbf{K}\cdot\mathbf{R}}=1$ 自动成立。<span class="marginnote">「正交归一」里的 $2\pi$ 因子是物理惯例：定义 $\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\delta_{ij}$ 则 $\mathbf{k}$（波矢）与 $\mathbf{r}$ 的乘积直接给出相位 $\mathbf{k}\cdot\mathbf{r}$；若用纯数学傅里叶变换的 $1$ 惯例，则要处处多挂一个 $2\pi$。全书统一用 $2\pi$ 惯例。</span>

## 3 倒格子与原格子的对偶：谁是谁的影子

倒格子的形状与原格子严格互反，且互为对偶：**实空间 bcc 的倒格子是 fcc，实空间 fcc 的倒格子是 bcc，实空间 sc 的倒格子还是 sc。**

以简单立方为例：晶胞边长 $a$，则倒基矢长度 $|\mathbf{b}| = 2\pi/a$，倒格子仍是立方格子、边长 $2\pi/a$。对 bcc，原胞基矢常取 $\mathbf{a}_1=\tfrac{a}{2}(\hat{y}+\hat{z}-\hat{x})$ 等；代入倒基矢公式得到一组 fcc 型的基矢，其立方晶胞边长为 $4\pi/a$。<span class="marginnote">bcc ↔ fcc 互偶这一事实，让「实空间格子的对称性」自动映射成「倒空间格子的对称性」——后面讲<strong>衍射消光</strong>（第4篇）与<strong>费米面形状</strong>（第9章）时，都要靠这张对偶表。</span>

倒空间与实空间的尺度成反比：晶胞越大，倒格子越密。这正是一维采样定理的推广——**实空间周期越大，频域分辨率越细**。

## 4 布里渊区：倒空间的原胞

**第一布里渊区（first Brillouin zone）**：倒空间里以某倒格点为原点，用所有倒格矢连线的中垂面围出来的最小区域——也就是**倒格子的 Wigner–Seitz 原胞**。它内含的波矢覆盖了一个完整周期，因此**晶体的所有本征态都可以用第一布里渊区内的 $\mathbf{k}$ 来标记**。<span class="marginnote">Wigner–Seitz 原胞在上一讲与 Voronoi 图同源；这里布里渊区就是倒空间的 Voronoi 单元。<strong>实空间讲「离哪个格点最近」，倒空间讲「属于哪个 k 的原胞」</strong>，几何完全同构。</span>

- 简单立方：第一布里渊区是边长 $2\pi/a$ 的立方体，边界在 $k_x=\pm\pi/a$ 处。
- fcc：截角八面体（含 12 个面）；bcc：菱形十二面体。这两种立体图是能带文献里的常客。

布里渊区边界对应布拉格反射条件：$2\mathbf{k}\cdot\mathbf{G} + G^2 = 0$（$G$ 为倒格矢）。当波矢落在边界上，电子波被周期性势场强反射，这正是下一讲能隙的起源。

## 5 公式解析：倒基矢公式怎么来的

倒基矢公式是全章的引擎，拆成三步理解：

$$
\mathbf{b}_1 = 2\pi\,\frac{\mathbf{a}_2 \times \mathbf{a}_3}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)}
$$

- **第一步，分母是原胞体积**：$V_c = \mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)$。除以体积保证量纲为「长度⁻¹」，即倒空间是波矢的国度。
- **第二步，分子是另外两个基矢的叉积**：$\mathbf{a}_2 \times \mathbf{a}_3$ 垂直于 $\mathbf{a}_2$、$\mathbf{a}_3$ 所在的平面，于是 $\mathbf{b}_1$ 垂直于该平面，自动保证 $\mathbf{b}_1\cdot\mathbf{a}_2 = \mathbf{b}_1\cdot\mathbf{a}_3 = 0$。
- **第三步，系数 $2\pi$ 校准相位**：把整个式子点乘 $\mathbf{a}_1$，得 $\mathbf{b}_1 \cdot \mathbf{a}_1 = 2\pi V_c/V_c = 2\pi$，恰好是正交归一条件。

循环置换 $\mathbf{a}$ 的下标即可得到 $\mathbf{b}_2,\mathbf{b}_3$。整条公式的本质是：**「垂直平面」的叉积给出正交性，「除以体积」给出归一化，「乘以 $2\pi$」给出相位校准。**

### 手算验证：bcc 的倒格子为什么是 fcc

用倒基矢公式亲手推一次。取 bcc 惯用原胞基矢（晶胞边长 $a$）：

$$\mathbf{a}_1 = \frac{a}{2}(\hat{y}+\hat{z}-\hat{x}),\quad \mathbf{a}_2 = \frac{a}{2}(\hat{x}+\hat{z}-\hat{y}),\quad \mathbf{a}_3 = \frac{a}{2}(\hat{x}+\hat{y}-\hat{z})$$

原胞体积 $V_c = a^3/2$。先算 $\mathbf{a}_2\times\mathbf{a}_3$，再代入 $\mathbf{b}_1 = 2\pi(\mathbf{a}_2\times\mathbf{a}_3)/V_c$，得到：

$$\mathbf{b}_1 = \frac{2\pi}{a}(\hat{y}+\hat{z}),\quad \mathbf{b}_2 = \frac{2\pi}{a}(\hat{x}+\hat{z}),\quad \mathbf{b}_3 = \frac{2\pi}{a}(\hat{x}+\hat{y})$$

这正是面心立方晶胞的对角矢量——**三个倒基矢互成 60°、长度 $4\pi/a$，构成 fcc 倒格子**。反过来把 $\mathbf{a},\mathbf{b}$ 互换，fcc 的倒格子就是 bcc。这组互偶关系在衍射指标化（第 4 篇）与费米面计算中反复使用。

### 一维的布里渊区：最简单的算例

一维晶格最直观：格点间距 $a$，倒格点间距 $2\pi/a$，第一布里渊区是区间 $[-\pi/a, +\pi/a]$。区边界 $k = \pm\pi/a$ 处满足布拉格条件 $2k\cdot G = G^2$（$G = 2\pi/a$）。**所有一维能带图、声子色散图都画在这个区间上**——它是理解「折叠」「能隙」的最小实验室。

### 倒空间的物理直觉：为什么费米面要画在倒空间

既然能带、声子、衍射都住在倒空间，那倒空间到底「好在哪」？三个理由：

- **平移对称性被自动消化**：实空间里无穷多原胞，倒空间里只有第一布里渊区一个周期单元——问题规模从无穷降到有限。
- **波矢是守恒量**：布洛赫电子、声子都以 $\mathbf{k}$ 为量子数，散射规律（准动量守恒）在倒空间里写成简单的加法。
- **衍射几何直接可读**：倒格点位置就是布拉格反射方向，无需逐面计算。

一句话：**实空间看「原子怎么排」，倒空间看「波怎么传」**。整个固体物理——从能带到声子到输运——都是这两幅图的来回切换。

## 6 辨析｜易错点：倒空间的常见误解

- **倒格矢的量纲不是长度**：$|\mathbf{b}| \sim 2\pi/a$，单位是「每长度」——它是波矢，不是另一种长度的格子。
- **布里渊区边界 ≠ 倒格点**：边界是中垂面，边界上每个点都满足布拉格条件；倒格点只是这些中垂面的「生成中心」。
- **k 是第一布里渊区里的点，G 是倒格矢**：电子波矢 $\mathbf{k}$ 与倒格矢 $\mathbf{G}$ 都住在倒空间，但 $\mathbf{k}$ 遍历第一布里渊区（连续取值），$\mathbf{G}$ 是离散的倒格点。两者相加：任何 $\mathbf{k}'$ 都可写成 $\mathbf{k} + \mathbf{G}$ 归入第一布里渊区——这就是「能带折叠」的数学。
- **bcc 与 fcc 互偶不要记反**：实空间 bcc → 倒空间 fcc；实空间 fcc → 倒空间 bcc。

## 7 小结

- 周期函数展开成傅里叶级数，波矢 $\mathbf{K}$ 满足 $e^{i\mathbf{K}\cdot\mathbf{R}}=1$，构成**倒格子**。
- 倒基矢 $\mathbf{b}_i$ 满足 $\mathbf{a}_i\cdot\mathbf{b}_j = 2\pi\delta_{ij}$，由「叉积正交 + 除以体积归一 + $2\pi$ 校准」构造。
- 倒格子对偶表：**sc↔sc，bcc↔fcc，fcc↔bcc**；实空间晶胞越大，倒格子越密。
- **第一布里渊区 = 倒空间的 Wigner–Seitz 原胞**；一切晶体本征态用区内 $\mathbf{k}$