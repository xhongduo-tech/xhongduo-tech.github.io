---
title: 有限温度形式（虚时演化、密度矩阵、巨正则形式）
date: 2026-08-07
---

# 有限温度形式（虚时演化、密度矩阵、巨正则形式）

<div class="epigraph">
<p>温度不是一个标量参数，而是一条通往量子力学的复时间轴——绕虚轴一圈，热力学就长出来了。</p>
<footer>—— 菲利普 · 安德森（Philip W. Anderson）论虚时场论（转述）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics*, Ch. 3 ｜ 2026-08-07</p>
</div>

## 为什么从有限温度形式开始

前面几篇都在零温或基态的框架里工作：Green 函数描写基态上粒子的传播，Feynman 图展开从真空态出发。<span class="marginnote"><strong>零温 vs 有限温度</strong>：实验上几乎所有凝聚态现象都发生在有限温度，$T=0$ 只是理想极限。超导临界温度、磁转变温度、费米简并温度都是有限温度物理。</span>可真实的固体有温度，电子在 $10^{2}$ K 下的热激发不可忽略。要处理这些，需要把二次量子化与统计力学合并：这就是**有限温度形式（finite-temperature formalism）**。核心思想是引入**虚时** $\tau = it$，把配分函数写成沿虚时间轴的「演化」——温度与时间在数学上被统一起来，这是 Matsubara 技术与解析延拓的出发点。

## 1 密度矩阵与巨正则系综

有限温度系统的信息全部编码在**密度矩阵（density matrix）**里。巨正则系综（grand canonical ensemble）允许粒子数涨落，配分函数为：

$$\mathcal{Z} = \mathrm{Tr}\, e^{-\beta(\hat{H}-\mu\hat{N})}, \qquad \beta = \frac{1}{k_B T}$$

其中 $\mu$ 是化学势，$\hat{N}$ 是粒子数算符，$\mathrm{Tr}$ 对 Fock 空间的全体态求迹。所有平衡热力学量都可以从 $\mathcal{Z}$ 推出：Helmholtz 自由能 $\Omega = -k_B T \ln \mathcal{Z}$，平均粒子数 $\langle N\rangle = -\partial\Omega/\partial\mu$，内能 $E = \partial(\beta\Omega)/\partial\beta$。<span class="marginnote"><strong>为什么用巨正则</strong>：多体计算里产生/湮灭算符天然改变粒子数，巨正则系综让粒子数成为可变的动态变量，与二次量子化无缝衔接。正则系综要求 $N$ 固定，操作上反而别扭。</span>

巨正则系综的平均定义为 $\langle \hat{O}\rangle = \mathrm{Tr}(\hat{\rho}\hat{O})$，其中密度算符

$$\hat{\rho} = \frac{1}{\mathcal{Z}}e^{-\beta(\hat{H}-\mu\hat{N})}$$

## 2 虚时演化：把温度变成时间

量子力学的实时间演化由 $e^{-i\hat{H}t/\hbar}$ 给出。如果做**维克旋转（Wick rotation）** $t \to -i\tau$，即把时间轴转到虚方向，演化算符就变成

$$e^{-\hat{H}\tau/\hbar} = e^{-\beta\hat{H}}$$

只要把 $\tau$ 取到「虚时区间」 $[0,\beta]$ 的尽头 $\tau = \beta$，密度算符就登场了。这正是把 $\hat{\rho}$ 读作「沿虚时间演化了 $\beta$ 的算符」的由来。<span class="marginnote"><strong>物理直觉</strong>：实时间演化保持概率守恒，是幺正的；虚时间演化让高能态指数衰减，非幺正。所以虚时演化天然是「低温投影」——温度越低，留在基态的权重越大。T=0 极限下配分函数退化为基态投影，零温 Green 函数作为特例被包含进来。</span>

**重点：虚时不是一个数学噱头，而是有限温度场论的地基。** 所有有限温度 Green 函数都定义在虚时轴上，配分函数就是虚时演化算符的求迹。

## 3 虚时算符与虚时时序

把二次量子化的产生/湮灭算符搬到虚时，定义

$$a_\lambda(\tau) = e^{\tau(\hat{H}-\mu\hat{N})} a_\lambda e^{-\tau(\hat{H}-\mu\hat{N})}$$

对玻色子，沿虚时的「时序」定义要求 $\tau$ 只能取值在 $[0,\beta]$ 区间内，且玻色函数按 $\beta$ 周期、费米函数按 $2\beta$ 周期。这个周期性是有限温度形式最重要的结构——它导致 Matsubara 频率的离散化，是第 4 篇《松原格林函数》的主角。

## 4 配分函数与虚时路径积分

虚时视角让配分函数有了路径积分形式。对玻色系统，把 $[0,\beta]$ 切成 $M$ 片，插入相干态完备集，取 $M\to\infty$ 极限：

$$\mathcal{Z} = \int \mathcal{D}\bar\psi\mathcal{D}\psi\; e^{-S_E[\bar\psi,\psi]}$$

其中作用量（虚时 / 欧氏作用量）

$$S_E = \int_0^\beta d\tau \left[\sum_\lambda \bar\psi_\lambda \partial_\tau \psi_\lambda + H(\bar\psi,\psi) - \mu N(\bar\psi,\psi)\right]$$

**重点：虚时路径积分让「配分函数」与「场论」合二为一。** 有限温度多体问题从此可以像零温一样做微扰展开，只是把实时间换成虚时间、把频率求和换成离散的 Matsubara 频率。

## 5 公式解析：为什么是 $e^{-\beta(\hat{H}-\mu\hat{N})}$

把巨正则密度算符逐项拆解：

$$
\hat{\rho} = \frac{1}{\mathcal{Z}} e^{-\beta(\hat{H}-\mu\hat{N})}
$$

- **指数里是 $\hat{H} - \mu\hat{N}$，不是 $\hat{H}$**：在巨正则系综里，粒子数 $N$ 不是守恒的约束而是可调的变量，$\mu$ 是它的拉格朗日乘子。加一个粒子要付出能量 $\mu$，所以「有效能量」是 $E - \mu N$。
- **$\beta = 1/k_B T$**：温度的倒数，单位是能量。它把「温度」翻译成「虚时长度」——温度越低，虚时区间 $[0,\beta]$ 越长，可以装下的虚时过程越多，这正对应低温下量子涨落更重要。
- **$\mathrm{Tr}$ 在 Fock 空间上**：对玻色/费米的所有占据数态求和。在占据数基底下，$e^{-\beta(\hat{H}-\mu\hat{N})}$ 是对角占优的，这是多数有限温度计算的实际入口。

从 $\hat{\rho}$ 出发，单粒子量 $\langle a_\lambda^\dagger a_\lambda\rangle$ 就是玻色分布（对玻色子）

$$\langle \hat{n}_\lambda\rangle = \frac{1}{e^{\beta(\varepsilon_\lambda-\mu)} - 1}$$

费米子则是 $\langle c_\lambda^\dagger c_\lambda\rangle = [e^{\beta(\varepsilon_\lambda-\mu)}+1]^{-1}$。一次简单的求迹，量子统计的全部内容就自动流出来——这正是有限温度形式的力量。

## 6 具体例子：一个费米子单态

为了让虚时框架落地，看一个最简单的非平凡系统：单个费米子能级 $\varepsilon$，自旋简并但不考虑相互作用。巨正则配分函数对 $n=0,1$ 两个占据态求和：

$$\mathcal{Z} = 1 + 2e^{-\beta(\varepsilon-\mu)}$$

拆解一下：$1$ 是空态贡献，$e^{-\beta(\varepsilon-\mu)}$ 是一个自旋方向的单占据贡献，因自旋简并出现因子 $2$。平均占据数为

$$\langle n\rangle = \frac{2e^{-\beta(\varepsilon-\mu)}}{1 + 2e^{-\beta(\varepsilon-\mu)}}$$

这正是费米分布函数的二能级形式。这个例子的价值在于：它用三个步骤展示了有限温度形式的全部套路——**写出 $\hat{H}-\mu\hat{N}$ 的本征谱、对全部态求迹、从配分函数取对数求热力学量**。任何有限温度计算，无论多复杂，结构都逃不出这三步。

进一步，若想求虚时 Green 函数，只需把算符期望值 $e^{-\beta\hat{H}}$ 里的演化拆成区间段：$G(\tau) = \langle a(\tau)a^\dagger\rangle$ 在 $\tau\in(0,\beta)$ 上由占据数 $\langle n\rangle$ 直接给出。区间边界上的周期性（玻色周期 $\beta$、费米反周期 $2\beta$）会自动把虚时计算约束到「每周期只有有限个独立量」——这就是下一节谱分析、以及再下一节 Matsubara 求和的伏笔。

## 7 从有限温度回到零温：解析延拓预览

有限温度计算的最后一步通常是把虚时结果「搬回」实频率，这一步叫**解析延拓（analytic continuation）**。粗略地讲，Matsubara 频率 $\omega_n$（离散的）与实频率 $\omega$（连续的）由同一个解析函数的两个边界连接：$G(\omega_n)$ 在 $i\omega_n\to\omega+i0^+$ 处延拓即得推迟 Green 函数。这条技术主线在本专题第 4 篇《松原格林函数》展开，这里只需记住一句话：**有限温度理论是零温理论的自然推广，虚时区间越长（温度越低），两者越接近。**

## 8 小结

- **密度矩阵与巨正则系综**：配分函数 $\mathcal{Z}=\mathrm{Tr}\,e^{-\beta(\hat{H}-\mu\hat{N})}$ 编码全部热力学信息。
- **虚时演化**：Wick 旋转 $t\to-i\tau$ 把密度算符读作沿虚时 $[0,\beta]$ 演化 $\beta$ 的算符，温度与时间统一。
- **虚时算符**满足玻色 $\beta$-周期、费米 $2\beta$-反周期的边界条件，导致 Matsubara 频率离散化。
- **虚时路径积分**把配分函数写成欧氏作用量的路径积分，有限温度微扰论由此成立。
- 单粒子占据数由玻色/费米分布给出，化学势 $\mu$ 是巨正则系综的核心拉格朗日乘子。
- **解析延拓** $i\omega_n\to\omega+i0^+$ 把有限温度结果搬回实频率，衔接推迟 Green 函数。

## 9 公式速查：一页纸复习

| 对象 | 表达式 | 一句话要点 |
| --- | --- | --- |
| 配分函数 | $\mathcal{Z}=\mathrm{Tr}\,e^{-\beta(\hat{H}-\mu\hat{N})}$ | 有限温度全部信息 |
| 密度算符 | $\hat{\rho}=e^{-\beta(\hat{H}-\mu\hat{N})}/\mathcal{Z}$ | 虚时演化 $\beta$ |
| 虚时演化 | $e^{-\hat{H}\tau/\hbar}$ | Wick 旋转 $t\to-i\tau$ |
| 玻色占据 | $\frac{1}{e^{\beta(\varepsilon-\mu)}-1}$ | 可任意堆积 |
| 费米占据 | $\frac{1}{e^{\beta(\varepsilon-\mu)}+1}$ | 泡利限制 |
| 自由能 | $\Omega=-k_BT\ln\mathcal{Z}$ | 热力学入口 |

**易错复盘**：虚时 $\tau$ 取值只能落在 $[0,\beta]$，超出区间要用周期（玻色）或反周期（费米）延拓；$e^{-\beta(\hat{H}-\mu\hat{N})}$ 里 $\mu$ 不能漏；$\beta$ 的单位是能量的倒数。把这三处盯住，有限温度计算的基本框架就站稳了。

**知识连线**：本篇把第 1 篇《二次量子化》的语言搬进统计力学，是第 4 篇《松原格林函数》与第 5 篇《费曼图与微扰论》在有限温度下的直接前提；它也是从「从极限到大模型」主线理解温度的入口——温度在量子多体理论里不是外部参数，而是虚时区间的长度。

在下一节，我们将给零温 Green 函数做一次系统的谱分析：从 Lehmann 表示到谱函数与解析性质。
