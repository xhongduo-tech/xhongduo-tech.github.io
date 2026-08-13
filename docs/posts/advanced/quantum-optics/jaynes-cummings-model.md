---
title: Jaynes-Cummings 模型
date: 2026-08-07
---

# Jaynes-Cummings 模型

<div class="epigraph">
<p>一个二能级原子，一个光学模式，再无其他——却容纳了量子光学一半的深邃。</p>
<footer>—— 埃德温·杰恩斯（Edwin T. Jaynes）与弗雷德·卡明斯（Fred W. Cummings），1963 年</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ M. O. Scully & M. S. Zubairy, Quantum Optics 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从 Jaynes-Cummings 模型开始

半经典理论把光当经典波，解释不了自发辐射；
要理解光与物质在**单光子水平**的相互作用，
需要把场也量子化。**Jaynes-Cummings（JC）模型**只保留一个二能级原子 
+ 一个单模腔场，是可精确求解的**全量子模型**。它是腔 QED 的数学心脏：
真空 Rabi 振荡、崩塌-复活、Fock 态制备、量子纠缠，全都写在 JC 
模型的解里。它还是量子信息理论中「光-物质接口」的标准模型——几乎所有光量子存储、
量子网络的方案都从 JC 
出发。<span class="marginnote">JC 
模型是「从极限到大模型」课程里少有的「完全可解 + 
完全量子」的模型——它的每一阶激发都能精确计算，
这使它成为教科书与量子模拟的双重主角。</span>

## 1 哈密顿量：原子 + 场 + 相互作用

JC 模型哈密顿量由三项组成（$\hbar = 1$）：

$$\hat{H} = \underbrace{\omega_0 \hat{\sigma}_z}_{\text{原子}} + \underbrace{\omega \hat{a}^\dagger\hat{a}}_{\text{场}} + \underbrace{g(\hat{\sigma}_+\hat{a} + \hat{\sigma}_-\hat{a}^\dagger)}_{\text{相互作用（RWA）}}$$

其中 $\hat{\sigma}_+ = |e\rangle\langle g|$ 
是原子的升算符，$\hat{\sigma}_- = |g\rangle\langle e|$ 
是降算符，$g$ 
是**单光子耦合强度**（$g = d_{eg}\sqrt{\omega_0/(2\hbar\epsilon_0 V)}$，
正比于偶极矩、反比于腔体积的平方根）。

相互作用项的两部分各有名字：

- $\hat{\sigma}_+\hat{a}$：原子吸收一个光子、跃迁到激发态（**共旋**项）；
- $\hat{\sigma}_-\hat{a}^\dagger$：原子释放一个光子、回到基态（**共旋**项）。

在旋转波近似下，
反旋项 $\hat{\sigma}_+\hat{a}^\dagger$、$\hat{\sigma}_-\hat{a}$ 
被丢弃。<span class="marginnote">RWA 在此的物理意义：
只有「原子下、光子上」和「原子上、
光子下」的通道被保留——总激发数 $\hat{N} = \hat{a}^\dagger\hat{a} + \hat{\sigma}_z$ 
成为守恒量，JC 模型因此可分解成一个个独立的小希尔伯特空间。</span>

## 2 激发数守恒与两能级子空间

**重点：总激发数 $\hat{N} = \hat{a}^\dagger\hat{a} + \hat{\sigma}_z$ 是守恒量**（$[\hat{N}, \hat{H}] = 0$）。
每个「激发数 = n」的子空间只含两个态：

$$|e, n-1\rangle, \qquad |g, n\rangle$$

$n = 0$ 时只有 $|g, 0\rangle$（基态 + 真空），
能量 $-\omega_0/2$，不参与动力学。在 $n \geq 1$ 子空间，
哈密顿量约化为 $2\times2$ 矩阵：

$$\hat{H}_n = \begin{pmatrix} \omega(n-\tfrac{1}{2}) + \frac{\Delta}{2} & g\sqrt{n} \\ g\sqrt{n} & \omega(n-\tfrac{1}{2}) - \frac{\Delta}{2} \end{pmatrix}$$

其中 $\Delta = \omega_0 - \omega$ 是原子-场失谐。
注意**耦合强度被 $\sqrt{n}$ 放大**：$g_n = g\sqrt{n}$——光子数越多，
相互作用越强，
这是玻色增强的量子版本。<span class="marginnote">$g\sqrt{n}$ 
与半经典的拉比频率 $\Omega$ 呼应：
经典强场对应大 $n$，$\Omega \leftrightarrow 2g\sqrt{n}$。
JC 模型在 $n \to \infty$ 
极限「还原」了半经典结果——这是经典-量子对应的又一个例证。</span>

## 3 真空 Rabi 振荡与崩塌-复活

对角化 $2\times2$ 
矩阵得到**缀饰态（dressed states）**能量

$$E_{n\pm} = \omega\left(n - \tfrac{1}{2}\right) \pm \frac{1}{2}\sqrt{\Delta^2 + 4g^2 n}$$

设初始原子在激发态、场在 Fock 
态 $|n\rangle$（即 $|e, n\rangle$），激发概率的时间演化

$$P_e(t) = \cos^2\left(\frac{t}{2}\sqrt{\Delta^2 + 4g^2 n}\right)$$

共振时（$\Delta = 0$）这是频率 $\Omega_n = 2g\sqrt{n}$ 
的振荡。当初始场是相干态（光子数按泊松分布）时，$P_e(t)$ 要对 $n$ 
加权求和：

$$P_e(t) = \sum_n P(n)\cos^2(g\sqrt{n}\,t)$$

不同 $n$ 的振荡频率不同，
它们**相干叠加后先相消（崩塌）、随后又相位对齐（复活）**——这就是著名的**崩塌-复活（collapse and revival）**现象。<span class="marginnote">崩塌-复活是纯量子效应的显影：
若场是经典波（单一频率），只有干净的 Rabi 振荡；
振荡的「崩塌」恰恰来自场的光子数叠加——这是量子光场「离散性」最直观的证据。</span>

## 4 公式解析：缀饰态能量 $E_{n\pm} = \omega(n-\tfrac{1}{2}) \pm \tfrac{1}{2}\sqrt{\Delta^2 + 4g^2n}$

拆成三步：

**第一步，对角化 $2\times2$ 哈密顿量**：矩阵本征值公式 $\lambda = \tfrac{1}{2}\mathrm{Tr} \pm \tfrac{1}{2}\sqrt{(\mathrm{Tr}/2)^2 - \det}$，代入即得 $E_{n\pm}$。
**第二步，读出物理**：$\omega(n-\frac{1}{2})$ 是「$n$ 个光子 + 原子」的裸能量基线；$\pm\tfrac{1}{2}\sqrt{\Delta^2+4g^2n}$ 是相互作用把裸态劈裂的幅度。共振时劈裂 $2g\sqrt{n}$——**真空 Rabi 劈裂**（$n=1$ 时为 $2g$），这正是《真空 Rabi 劈裂》专篇的出发点。
- **第三步，大 $n$ 极限**：$\sqrt{\Delta^2+4g^2n} \approx 2g\sqrt{n}$（$\Delta$ 小时），劈裂随 $\sqrt{n}$ 增长——与半经典 $\Omega = 2g\sqrt{n}$ 一致。JC 模型「包含」半经典理论作为 $n \gg 1$ 极限。

## 5 JC 模型作为量子信息工作台

JC 模型不只是理论玩具，它是**量子技术的基础设施**：

- **原子-光子纠缠**：相互作用 $\hat{\sigma}_+\hat{a} + \hat{\sigma}_-\hat{a}^\dagger$ 把原子与场耦合，演化会生成 $\sim (|e,0\rangle + |g,1\rangle)$ 型纠缠态——这是光-原子量子网络的原始资源。
- **单光子确定性发射**：原子在腔内，把 $\pi$ 脉冲后的激发原子耦合到腔模，可确定性地发射单光子。
- **量子门**：原子与腔模的态交换（swap）、几何相位门都在 JC 框架内设计。
- **Fock 态制备**：通过受控原子探测（见《腔场态操控与 Fock 态制备》专篇）可在腔内制备 $|n\rangle$ 态，直接观测 JC 模型的 $\sqrt{n}$ 阶梯。

**辨析｜易错点：** JC 模型的 RWA 
在**超强耦合**（$g \sim \omega$）下失效，
此时必须保留反旋项（量子 Rabi 模型），系统会出现 
Bloch-Siegert 位移等新效应。不要把 JC 
模型的预言硬套进超强耦合实验——那正是近年「超强耦合量子电动力学」新领域的入口。<span class="marginnote">超强耦合领域把「RWA 
失效」当资源：反旋项允许奇偶宇称混合、
产生新的光子统计——这是量子光学仍在活跃扩张的前沿。</span>

## 6 小结

- JC 模型 = 单模腔场 + 二能级原子 + RWA 相互作用，**可精确求解**。
- 总激发数守恒 → 每个 $n$ 子空间是 $2\times2$ 系统，耦合 $g\sqrt{n}$。
- 缀饰态劈裂 $\sqrt{\Delta^2+4g^2n}$：真空 Rabi 劈裂（$2g$