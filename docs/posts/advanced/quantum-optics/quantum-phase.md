---
title: 量子相位
date: 2026-08-07
---

# 量子相位

<div class="epigraph">
<p>光子的相位，就像圆的弧长——它几乎无处不在，却没有人见过它的算符。</p>
<footer>—— 迪特里希·斯特鲁普（Dietrich Strou），物理学界对相位问题的长期咏叹</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ D. F. Walls & G. J. Milburn, Quantum Optics 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从量子相位开始

相位是光学最日常的概念：干涉条纹、透镜成像、激光相干性全都依赖相位。
但一到量子层面，
相位立刻变成难题——**不存在一个完美定义的光子相位算符**。1927 
年狄拉克就试图构造它，结果发现数学上处处是坑。
这个问题困扰物理学家半个世纪，直到 Pegg-Barnett 
与多种实用方案出现才获得满意解答。理解「相位为什么难定义」，
等于理解量子力学中「互补性」的极限形态；它也是理解零差探测、
量子相位估计与引力波干涉仪精度极限的关键。<span class="marginnote">本节的互补性讨论是第二级《量子力学》「不确定关系」一章在光场上的具体化，
也是第五级《量子信息》相位估计协议的理论源头。</span>

## 1 相位问题的由来

经典电场写成 $E(t) = E_0\cos(\omega t + \phi)$，$\phi$ 
是相位。量子化后场算符

$$\hat{E} \propto \hat{a}e^{-i\omega t} + \hat{a}^\dagger e^{i\omega t}$$

要定义相位算符 $\hat{\phi}$，
最自然的尝试是把它与数算符配对成「角动量式」的对易关系。
狄拉克猜：$[\hat{N}, \hat{\phi}] = i$，
由此推出数-相不确定关系

$$\Delta N\,\Delta\phi \geq \frac{1}{2}$$

但立即撞墙：若 $\Delta N$ 有限，
则 $\Delta\phi \to \infty$，
而相位应该在 $[0, 2\pi)$ 
上有界——一个有界算符与无界数算符不可能满足正则对易关系。这是 
Susskind-Glogower（1964）明确指出的核心困难。<span class="marginnote">直觉：$\Delta\phi$ 
有上界（至多 $2\pi$），对易关系却要求它与 $\Delta N$ 
的乘积无下界，两者冲突。这就是「相位算符不存在」的雏形。</span>

## 2 Susskind-Glogower 算符与 Pegg-Barnett 方案

**Susskind-Glogower（SG）指数算符**把相位「编进」算符的指数：

$$\hat{E} = \frac{1}{\sqrt{\hat{N}+1}}\,\hat{a}, \qquad \hat{E}^\dagger = \hat{a}^\dagger \frac{1}{\sqrt{\hat{N}+1}}$$

它们满足 $\hat{E}\hat{E}^\dagger = 1$ 
但 $\hat{E}^\dagger\hat{E} = 1 - |0\rangle\langle 0|$——**单位算符被真空态破坏**，
这就是相位算符「不完美」的代数根源。

**Pegg-Barnett（1988）方案**用有限维截断规避问题：
先把空间截到 $s+1$ 维，定义

$$|\theta_m\rangle = \frac{1}{\sqrt{s+1}}\sum_{n=0}^{s} e^{in\theta_m}|n\rangle, \qquad \theta_m = \theta_0 + \frac{2\pi m}{s+1}$$

相位态 $|\theta_m\rangle$ 
是「相位本征态」的有限维近似，
相位算符 $\hat{\phi}_\theta = \sum_m \theta_m|\theta_m\rangle\langle\theta_m|$。
物理量先在这套有限维空间中计算，最后取 $s \to \infty$ 
极限。有限维修正项往往以 $1/(s+1)$ 消失，
但真空态附近的修正会残留——这正是相位问题的精髓：**真空中没有相位**。<span class="marginnote">Pegg-Barnett 
方案对「相位分布」的计算最为自然：给定态 $|\psi\rangle$，
相位分布 $P(\theta) = |\langle\theta|\psi\rangle|^2$，
有限维修正取极限后给出光滑分布。</span>

## 3 数-相不确定关系与相位分布

有了可用的相位表示，就能给「相位涨落」一个定义。
对相干态 $|\alpha\rangle$（$\alpha = |\alpha|e^{i\theta_0}$），
相位分布

$$P(\theta) = \frac{1}{2\pi}\left|1 + 2\sum_{n=1}^{\infty}\frac{|\alpha|^n}{\sqrt{(n+1)!}}e^{in(\theta-\theta_0)}\right|^2$$

强场极限（$|\alpha| \gg 1$）下，$P(\theta)$ 
收敛为以 $\theta_0$ 为中心、
宽度 $\Delta\theta \approx \frac{1}{2|\alpha|} = \frac{1}{2\sqrt{\bar{n}}}$ 
的高斯型分布。于是数-相不确定关系在强场下成为

$$\Delta N\,\Delta\theta \approx \sqrt{\bar{n}}\cdot\frac{1}{2\sqrt{\bar{n}}} = \frac{1}{2}$$

**重点：相干态达到数-相不确定关系的下限**——它既是最小不确定态，
也在相位意义上最优。
这条关系是干涉测量精度极限（标准量子极限 $\propto 1/\sqrt{N}$）的量子源头。<span class="marginnote">若改用压缩态、
Fock 态等，还能进一步压低相位涨落（超分辨率），
这是《压缩态》与《量子增强测量》一脉相承的故事。</span>

## 4 公式解析：相干态的相位分布 $P(\theta)$

这条式子看起来吓人，但拆开只有三步：

**第一步，相位态的内积**：$P(\theta) = |\langle\theta|\psi\rangle|^2$ 是相位本征态投影的概率密度。把相干态展开 $\sum_n c_n|n\rangle$ 代入，相位态里每个 $|n\rangle$ 带 $e^{in\theta_m}$，于是内积成了 $\sum_n c_n e^{in\theta}$——一个关于 $\theta$ 的傅里叶级数。
**第二步，模方展开**：$|\sum_n c_n e^{in\theta}|^2 = \sum_{n,m}c_n c_m^* e^{i(n-m)\theta}$，对角项给出常数 $1/(2\pi)$ 项，交叉项 $n \neq m$ 给出随 $|\alpha|$ 增长的峰。
- **第三步，强场高斯化**：$|\alpha|$ 很大时，$c_n$ 集中在 $n \approx |\alpha|^2$ 附近且展宽 $\sqrt{\bar{n}}$，傅里叶级数近似成高斯包络，中心 $\theta_0$，宽度 $1/(2\sqrt{\bar{n}})$。这正是「激光相位极稳定」的数学表达——$\bar{n}$ 越大，相位越尖。

## 5 相位测量的实验面貌

理论上的相位分布如何落地？两条主流路径：

- **干涉测量**：马赫-曾德尔干涉仪中，两臂相对相位改变 $\phi$ 把光子分配到两个输出口，统计光子计数即可估计 $\phi$。精度受限于标准量子极限 $\delta\phi \geq 1/\sqrt{N}$，其中 $N$ 是总光子数。
- **零差探测**：把信号场与本地振荡器在分束器上混合，扫描本地振荡相位可测出场的正交分量分布——这是《零差探测与量子态断层成像》专篇的方法，也是实际提取相位分布 $P(\theta)$ 的标准工具。

**辨析｜易错点：** 
不要把「测量到的相位分布」与「相位算符本征值」混为一谈。
严格的相位算符（Pegg-Barnett 极限）是一个理想化对象；
实验上测到的永远是某个测量策略（干涉仪、
零差机）下的相位估计分布。**相位不是「藏在态里的数」，而是「测量协议的函数」**——这是量子测量理论的核心教训。<span class="marginnote">这个「可观测量由测量协议定义」的观点，
与第九级《量子信息》中「测量即信息提取」的视角完全一致。</span>

## 6 小结

- 相位问题的根源：**有界相位与无界光子数无法满足正则对易关系**。
- Susskind-Glogower 指数算符用 $\hat{E} = \hat{a}/\sqrt{\hat{N}+1}$ 部分解决，但真空态破坏单位性。
- Pegg-Barnett 用有限维截断 + 取极限构造严格的相位算符。
- 数-相不确定关系 $\Delta N\Delta\phi \geq 1/2$；相干态达到下限，相位分布宽度 $\propto 1/\sqrt{\bar{n}}$