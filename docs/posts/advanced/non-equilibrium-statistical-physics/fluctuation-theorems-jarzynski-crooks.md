---
title: 涨落定理（Jarzynski 等式、Crooks 定理、热力学不确定性关系）
date: 2026-08-07
---

# 涨落定理（Jarzynski 等式、Crooks 定理、热力学不确定性关系）

<div class="epigraph">
<p>须知原子必须微微偏斜——恰如卢克莱修所吟咏的原子倾斜，那不可察觉的偏离。</p>
<footer>—— 卢克莱修（Lucretius），《物性论》（De Rerum Natura）卷二</footer>
</div>

<div class="article-byline">
<p>第四级 · 非平衡统计物理 ｜ Zwanzig《Nonequilibrium Statistical Mechanics》第10章延伸；Jarzynski 1997、Crooks 1999、Seifert 2012 原始文献 ｜ 2026-08-07</p>
</div>

## 为什么从涨落定理继续

热力学第二定律说：任何过程的熵增 ≥ 0，等号只在可逆过程取到。但这句话只说平均值。**单次过程可能暂时「违背」第二定律**——一小团气体自发地变得更有序、热量短暂地从冷端流向热端，概率虽小却不为零。涨落定理精确回答：「违背第二定律」有多大的概率？

1997 年，克里斯托弗·贾津斯基（Christopher Jarzynski）证明了一个震惊物理界的等式：

$$
\langle e^{-\beta W}\rangle = e^{-\beta \Delta F}
$$

它把任意（甚至极快、极不可逆）过程的功 $W$ 的指数平均，与平衡自由能差 $\Delta F$ 精确相连。1999 年加文·克鲁克斯（Gavin Crooks）给出更一般的版本。此后涨落定理成为非平衡统计物理最活跃的前沿，并直接催生了**随机热力学**这一分支（下一讲）。<span class="marginnote">涨落定理的惊人之处：等号两边的量差着「一个世界」。左边是无数条瞬时非平衡轨迹上的平均，右边是纯平衡热力学的自由能差。它相当于说——即使每一次拉橡皮筋、折叠蛋白都充满涨落，把它们全部取指数平均，竟能精确还原本该只由可逆过程定义的热力学量。</span>

## 1 为什么第二定律是「平均定律」

先建立一个精确的观念。热力学第二定律 $\Delta S_{tot} \ge 0$ 说的是**系综平均**：$S_{tot} = \ln P(W)$ 中的期望值非负。但对有限小系统、短时间的单次过程，熵增是随机变量，可正可负。

小系统是大自然的常态：**生物分子（蛋白、DNA）、胶体粒子、纳米器件**都只有极少的自由度，热涨落的比例巨大。宏观系统的涨落相对量级为 $1/\sqrt{N}$（$N$ 是粒子数），宏观不可见；微观系统则处处可见。涨落定理把「第二定律在微观尺度如何修正」写成了精确的分布级陈述。<span class="marginnote">对 $N$ 个粒子的系统，熵产生 $S_{tot}$ 的相对涨落 $\propto 1/\sqrt{N}$。宏观一杯水 $N\sim 10^{25}$，$S_{tot}\ge 0$ 几乎必然；单个蛋白质 $N\sim 10^3$，涨落显著。这解释了为什么「违背第二定律」只在微观可见——不是定律失效，而是概率结构不同。</span>

## 2 Jarzynski 等式

考虑一个由参数 $\lambda(t)$ 驱动的系统，从平衡态出发，从 $\lambda(0)$ 到 $\lambda(\tau)$ 快速（任意速率）改变参数。沿轨迹测得的功 $W$ 是随机变量，其系综平均 $\langle W\rangle \ge \Delta F$（这个不等式本身是「功的平均 ≥ 自由能差」）。Jarzynski 等式的惊人之处在于把不等式升级为精确等式：

$$
\langle e^{-\beta W}\rangle = e^{-\beta \Delta F},\qquad \beta = \frac{1}{k_B T}
$$

**由 Jensen 不等式 $\langle e^{-x}\rangle \ge e^{-\langle x\rangle}$，立即恢复 $\langle W\rangle \ge \Delta F$。** Jarzynski 等式比第二定律更强：它不仅断言平均功至少是 $\Delta F$，还精确刻画了功涨落的整个分布。有趣的是，等式的指数权重让「少数极其小功的轨迹」贡献巨大——即使大部分轨迹耗散严重，只要存在一条近乎可逆的轨迹，$\Delta F$ 也能被恢复。这也解释了为何实验需要采集海量轨迹：极端涨落事件携带的信息量不成比例地大。<span class="marginnote">Jarzynski 1997 年发表于《Physical Review Letters》的论文《Nonequilibrium Equality for Free Energy Differences》是这个领域的引爆点。它最直接的应用是「功测量估自由能」：即便过程不可逆，只要记录足够多轨迹的功分布，$e^{-\beta\Delta F} = \langle e^{-\beta W}\rangle$ 就能给出平衡自由能——光学镊子拉伸 DNA 测弹性自由能正是这么做的。</span>

## 3 Crooks 涨落定理

Crooks 定理把 Jarzynski 等式推广为「正过程与反过程的功分布之比」：

$$
\frac{P_F(+W)}{P_R(-W)} = e^{\beta(W - \Delta F)}
$$

其中 $P_F(W)$ 是正向过程（$\lambda(0)\to\lambda(\tau)$）做功能量为 $W$ 的概率，$P_R(-W)$ 是反向过程（$\lambda(\tau)\to\lambda(0)$，从对应平衡态出发）释放能量 $W$ 的概率。

**物理内涵**：正向过程「逆热力学」地做了大功 $W > \Delta F$ 的概率，与反向过程「顺势」做了功 $W - \Delta F$ 的概率之比，精确等于玻尔兹曼因子 $e^{\beta(W-\Delta F)}$。**反过程越是容易实现 $W - \Delta F$ 的功，正过程实现 $W$ 就越难——比值给出精确的指数关系。**

Crooks 定理还有一个实用的推论：正向与反向功分布的交叉点恰好位于 $W = \Delta F$——因此只需找到两分布相交处，即可直接读出自由能差，无需任何积分。这就是实验上最常用的「交叉点估计法」，也是其直觉可操作性的体现。<span class="marginnote">把 Crooks 定理两端对 $W$ 积分可得 Jarzynski 等式，所以 Crooks 定理是更强版本。它还给了第二定律一个分布级的表述：$P(W-\Delta F>0)$ 的负功事件（单次「违背」）不是零概率，其概率由 $e^{-\beta(W-\Delta F)}$ 压制——违反越严重，越不可能，且压制是指数的。</span>

## 4 公式解析：从细致平衡到 Jarzynski 等式

Jarzynski 等式的推导揭示了它与微观可逆性的深刻联系：

$$
\langle e^{-\beta W}\rangle = e^{-\beta \Delta F}
$$

- **第一步**：从系综出发，沿轨迹的初态权重是玻尔兹曼因子 $e^{-\beta H_0}$。轨迹的生成概率 ∝ 各时刻处于相应状态的玻尔兹曼权重。
- **第二步**：关键重排——把「演化算符」与「玻尔兹曼权重」交换位置。利用哈密顿量的时间依赖，把 $e^{-\beta W}$ 吸收进轨迹概率的归一化。
- **第三步**：整理后，$\langle e^{-\beta W}\rangle = Z(\tau)/Z(0)$，其中 $Z(t)$ 是参数 $\lambda(t)$ 对应哈密顿量的配分函数。
- **第四步**：由配分函数关系 $F = -k_B T\ln Z$，得 $\langle e^{-\beta W}\rangle = e^{-\beta[F(\tau)-F(0)]} = e^{-\beta\Delta F}$。

**重点：推导只用了「初态平衡 + 微观可逆动力学」，没有对驱动速率做任何假设。** 这就是 Jarzynski 等式对任意不可逆过程都成立的原因——它把平衡统计的微观可逆性，精确投影到非平衡轨迹的平均上。

## 5 热力学不确定性关系

涨落定理家族的最新成员是 2015 年前后出现的**热力学不确定性关系（Thermodynamic Uncertainty Relation, TUR）**。它把「精度、耗散、时间」三者绑定：

$$
\frac{\mathrm{Var}(J)}{\langle J\rangle^2} \cdot \frac{\sigma_{tot}}{k_B} \ge \frac{2}{\tau}
$$

其中 $J$ 是某个流（如粒子流、化学产率），$\mathrm{Var}(J)$ 是其方差，$\sigma_{tot}$ 是总熵产生率，$\tau$ 是观测时间。

**物理内涵：想精确测量一个流的平均值（即减小相对涨落），必须以更大的耗散为代价；给定耗散与时间，流的精度有绝对上限。**<span class="marginnote">TUR 是 2015 年由 Barato &amp; Seifert（《Phys. Rev. Lett.》）与 Gingrich 等人分别建立的。它把涨落定理家族从「描述涨落」推进到「约束涨落」——在分子马达效率、纳米开关精度、生物传感灵敏度研究中成为定量设计工具：要多少能量才能把测量做到某个精度，TUR 给出硬性下界。</span>

**应用领域**：分子马达（驱动蛋白）、单分子酶动力学、DNA 测序的能耗-精度权衡、纳米电子器件噪声。TUR 回答了「生命机器为何要耗能」的定量版本：无耗散的精确微观测量不存在。

**计算方法的支点**：Crooks 定理在分子模拟中有直接应用——**Bennett acceptance ratio（BAR）**方法利用正反过程功分布的比值估计自由能差，是分子模拟中估算自由能的标准工具（Bennett 1976，其现代形式即 Crooks 涨落定理）。实际做法是分别采样正向与反向功分布，用最大似然拟合得到 $\Delta F$——比直接对 $\langle W\rangle$ 取平均收敛快得多，这正是「用涨落分布反推热力学量」策略在计算物理里的果实。

## 6 涨落定理家族一览

涨落定理不止 Jarzynski 等式与 Crooks 定理，而是一个不断扩展的家族。把主要成员放在一张表里对照：

| 定理 | 表述 | 成立条件 | 首次提出 |
| --- | --- | --- | --- |
| 非平衡功关系 | $\langle e^{-\beta W}\rangle = e^{-\beta\Delta F}$ | 初态平衡、任意驱动 | Jarzynski 1997 |
| Crooks 涨落定理 | $P_F(W)/P_R(-W) = e^{\beta(W-\Delta F)}$ | 正反过程配对 | Crooks 1999 |
| 详细涨落定理 | $P(S_{tot})/P(-S_{tot}) = e^{\beta S_{tot}}$ | 马尔可夫动力学 | Kurchan 1998、Lebowitz &amp; Spohn 1999 |
| 积分涨落定理 | $\langle e^{-\beta S_{tot}}\rangle = 1$ | 马尔可夫动力学 | Seifert 2005 |
| 热力学不确定性关系 | $\frac{\mathrm{Var}(J)}{\langle J\rangle^2}\frac{\sigma_{tot}}{k_B}\ge\frac{2}{\tau}$ | 稳态马尔可夫 | Barato &amp; Seifert 2015、Gingrich 等 |

几个关键观察：

- **Crooks 定理蕴含 Jarzynski 等式**：把 Crooks 定理两端对功积分即得 Jarzynski 等式，因此 Crooks 定理是更强的版本。
- **积分涨落定理** $\langle e^{-\beta S_{tot}}\rangle = 1$：把「第二定律的分布级表述」推广到任意马尔可夫过程的单轨迹熵产生，是家族中最一般的成员。
- **详细涨落定理**：正负熵产生的概率比精确为 $e^{S_{tot}}$，是 Crooks 定理在熵产生变量上的化身。
- **实验验证**：除光学镊子拉 RNA，涨落定理还在光镊驱动的布朗粒子（Wang 等 2002，《Nature》）、流体扰动下的胶体微粒（Blickle &amp; Bechinger 2012）以及电子热噪声涨落中得到精确验证，误差在实验精度内。

**重点：涨落定理家族不是「第二定律的推翻」，而是「第二定律的精化」。** 第二定律是家族所有成员在「取平均」意义下的共同推论；家族成员则各自描述了涨落分布的精确形状。<span class="marginnote">涨落定理的数学根源与玻尔兹曼的 H 定理一脉相承：都是「微观可逆 + 平衡初态」在统计平均下的投影。区别在于 H 定理给出不等式，涨落定理给出等式——后者是前者在分布级的精确化，正对应本专题从「熵增」到「涨落」的深化路线。</span>

## 7 小结

- **第二定律是平均定律**：单次过程可「违背」，概率由 $e^{-\beta S_{tot}}$ 指数压制，宏观不可见。
- **Jarzynski 等式**：$\langle e^{-\beta W}\rangle = e^{-\beta\Delta F}$，任意不可逆过程的功分布精确还原自由能差。
- **Crooks 定理**：$P_F(W)/P_R(-W) = e^{\beta(W-\Delta F)}$，正反过程功分布之比是精确的指数。
- **推导核心**：初态平衡 + 微观可逆性，对驱动速率无假设。
- **热力学不确定性关系**：$\frac{\mathrm{Var}(J)}{\langle J\rangle^2}\frac{\sigma_{tot}}{k_B}\ge\frac{2}{\tau}$，精度与耗散的硬性权衡。
- **应用**：BAR 自由能计算、分子马达效率分析、纳米器件能耗设计。

在下一节，我们把涨落定理用于「小系统做功」的完整图景：随机热力学——单分子功分布、信息热力学与麦克斯韦妖。
