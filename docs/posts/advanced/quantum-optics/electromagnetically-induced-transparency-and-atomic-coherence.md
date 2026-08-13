---
title: 电磁感应透明与原子相干
date: 2026-08-07
---

# 电磁感应透明与原子相干

<div class="epigraph">
<p>一束光可以引导另一束光穿过本来会吸收它的介质——像一扇由光打开的透明之门。</p>
<footer>—— 斯蒂芬·哈里斯（Stephen E. Harris），电磁感应透明的发现者</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ M. O. Scully & M. S. Zubairy, Quantum Optics 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从电磁感应透明开始

二能级原子对共振光是强烈吸收的。但给原子再加一个能级、
再用一束强光「控制」，
原本吸收光的介质会在共振频率处**变得透明**——这就是**电磁感应透明（EIT）**。
它看似反直觉，根子却在**原子相干**：控制场把原子制备到一个暗态，
使探测光的吸收路径被量子干涉相消。EIT 不仅带来无吸收的慢光、光存储，
还是量子记忆与量子信息处理的关键技术。<span class="marginnote">EIT 
与第三级《原子物理》的 Autler-Townes 劈裂同源，但本质不同：
EIT 依赖暗态的存在，是一种「无粒子布居转移」的相干效应。</span>

## 1 Λ 型三能级系统与暗态

考虑 Λ 型三能级原子：基态 $|1\rangle$、
亚稳态 $|2\rangle$、激发态 $|3\rangle$。
探测场 $\Omega_p$ 
驱动 $|1\rangle \leftrightarrow |3\rangle$，
控制场 $\Omega_c$ 
驱动 $|2\rangle \leftrightarrow |3\rangle$，
且 $|3\rangle$ 的自发辐射率 $\Gamma$ 
很大（激发态寿命短）。<span class="marginnote">「Λ 
型」的名字来自能级图：两个低能态像希腊字母 Λ 的两条腿，激发态在顶。
两个低能态之间的相干至关重要——但两者之间没有直接跃迁。</span>

在双光子共振（$\Delta_1 = \Delta_2$）条件下，
三个态中有一个**本征态完全不含激发态 $|3\rangle$**：

$$|D\rangle = \frac{\Omega_c|1\rangle - \Omega_p|2\rangle}{\sqrt{\Omega_p^2 + \Omega_c^2}}$$

**这就是暗态（dark state）**。它的两个性质决定了 EIT 
的一切：

- 不含 $|3\rangle$，因此**不被激发态的自发辐射影响**（不发光、不吸收）；
- 是 $\hat{H}_{\mathrm{int}}$ 的零本征态，**动力学冻结**（不随时间演化）。

**重点：探测场被原子「不吸收地穿过去」，因为原子被制备在了与探测场解耦的暗态里。** 
光子没有真的被吸收——它们与原子一起「滑」进了暗态并原样穿出。<span class="marginnote">暗态概念也出现在受激拉曼绝热通道（STIRAP）里：
通过绝热改变 $\Omega_p/\Omega_c$ 的比值，
可以把原子从 $|1\rangle$ 
平滑地搬运到 $|2\rangle$ 
而不经过 $|3\rangle$——无损耗的量子态转移。</span>

## 2 线性响应：探测光为什么透明

用线性响应理论计算探测光的极化率 $\chi(\omega_p)$，在 
EIT 条件下（双光子共振、控制场强于拉比速率）其虚部（吸收）为

$$\mathrm{Im}\,\chi(\omega_p) \propto \frac{\gamma_2}{(\omega_p - \omega_{31})^2/\Omega_c^2 + \gamma_2^2}$$

吸收谱线宽由 **$\gamma_2$（两个基态间的退相率）** 决定，
而不是激发态寿命 $1/\Gamma$。
由于 $\gamma_2 \ll \Gamma$（亚稳态相干寿命长），
吸收被极度压制——在共振处形成一个**透明窗口**。<span class="marginnote">正常二能级原子吸收线宽 $\sim \Gamma$（几十 
MHz）；EIT 
透明窗口线宽可做到 $\sim \gamma_2$（kHz 甚至更窄）。
透明窗口越窄，折射率变化越陡，慢光效果越强。</span>

同时，透明窗口伴随**异常陡峭的正常色散**（$dn/d\omega$ 
巨大），导致群速度极慢：

$$v_g = \frac{c}{1 + \frac{n\omega_p}{2}\frac{dn}{d\omega}} \approx \frac{c}{1 + \text{巨大项}}$$

实验上群速度可慢至 $17$ m/s，甚至**降到零**（光存储）。

## 3 慢光与光存储

EIT 的群速度减慢不只是好玩——它实现了**光信息的时间缓冲**：


- **慢光**：脉冲在介质中速度降低上千倍，可延长光在延迟线中的驻留时间；
- **光存储（停光）**：脉冲进入介质后**绝热关掉控制场**，光被「冻结」为暗态自旋波（$\Omega_p|1\rangle - \Omega_c|2\rangle$ 的集体激发）；控制场重新打开时，光**原样释放**。存储时间由基态相干寿命 $\gamma_2^{-1}$ 决定。

这套「光写-存储-光读」正是量子记忆（quantum 
memory）的骨干方案。2001 年哈佛小组实现 17 m/s 慢光，
2005 年实现光停止与再释放，是 EIT 
应用的两座里程碑。<span class="marginnote">EIT 
光存储保存的不只是光强，
而是光场的量子态——这是它与经典慢光（如光纤延迟线）的本质区别，
也是量子网络需要它的原因。</span>

## 4 公式解析：暗态 $|D\rangle = \dfrac{\Omega_c|1\rangle - \Omega_p|2\rangle}{\sqrt{\Omega_p^2+\Omega_c^2}}$

这条式子定义了 EIT 的「工作态」，拆成三步：

**第一步，验证不含 $|3\rangle$**：把 $\hat{H}_{\mathrm{int}}$（耦合 $|1\rangle,|3\rangle$ 与 $|2\rangle,|3\rangle$）作用在 $|D\rangle$ 上。由于 $|D\rangle$ 只含 $|1\rangle, |2\rangle$，相互作用只能把这两个分量「抬」到 $|3\rangle$，两路贡献分别为 $\Omega_p\cdot\Omega_c$ 与 $\Omega_c\cdot(-\Omega_p)$——正好相消，净结果为零。
**第二步，权重比例**：$\Omega_c$ 越大，暗态越偏 $|1\rangle$；$\Omega_p$ 越大，越偏 $|2\rangle$。改变场强比值，就改变了暗态在基态子空间的方向——这是 STIRAP 绝热搬运的几何基础。
- **第三步，探测光视角**：探测光只「看见」$|1\rangle \to |3\rangle$ 通道。当原子处于暗态，$|1\rangle$ 分量与 $|3\rangle$ 的耦合被 $|2\rangle$ 分量的反相耦合抵消——从探测光眼里，原子「消失了」。

## 5 EIT 的现代图景

EIT 已从原子物理的奇观成长为量子技术的工具箱：

- **量子记忆**：冷原子系综中的 EIT 光存储是量子中继器（quantum repeater）的核心器件；
- **量子门**：Rydberg EIT 实现光子-光子非线性相互作用，用于光子量子逻辑门；
- **灵敏探测**：EIT 介质对磁场、电场的敏感响应用于磁力计、电场计；
- **光速操控**：慢光、停光、光速前移（fast light）在光通信与量子信息处理中各有用途。

**辨析｜易错点：** EIT 
的透明窗口不是「功率烧孔」（population hole 
burning）或饱和吸收——那两者靠布居转移，会消耗原子；EIT 
靠量子干涉，原子始终留在暗态、布居几乎不变。判断标准：EIT 
窗口中心是**完美透明**（无吸收），且伴随反常色散；
饱和吸收则始终有残余吸收。<span class="marginnote">EIT 
的物理内核——「量子干涉制造暗态」——与相干布居俘获（CPT）、
无反转激光（LWI）是同一棵树的三个果实，它们都写在 Scully & 
Zubairy 第7章。</span>

## 6 小结

- EIT：Λ 型三能级 + 控制场制备**暗态**，探测场无吸收穿过。
- 暗态不含激发态，因此无自发辐射、无吸收、动力学冻结。
- 透明窗口线宽 $\sim \gamma_2$