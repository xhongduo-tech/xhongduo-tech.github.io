---
title: 配分函数
date: 2026-08-07
---

# 配分函数

<div class="epigraph">
<p>热力学的定律，就其本质而言，不过是概率定律的推论。</p>
<footer>—— 约西亚 · 威拉德 · 吉布斯（Josiah Willard Gibbs）</footer>
</div>

<div class="article-byline">
<p>第二级 · 物理化学 ｜ 傅献彩《物理化学》第3章 §3.4–3.5 · Atkins《Physical Chemistry》Ch.16 ｜ 2026-08-07</p>
</div>

## 为什么从配分函数开始

上一节的玻尔兹曼分布里，分母 $q = \sum_j g_j e^{-\varepsilon_j/kT}$ 看似只是归一化常数，实则暗藏玄机。**配分函数（partition function）$q$ 是统计热力学的「总账本」**：它把系统全部能级信息压缩进一个函数，所有宏观热力学量——内能、熵、自由能、压力——都可以从它及其导数读出。<span class="marginnote">Gibbs 发明了系综与配分函数，Boltzmann 给出了 $S = k\ln\Omega$。配分函数把「微观求和」与「宏观量」之间的每一道门都打通：$\ln q$ 对温度、体积、粒子数的偏导，分别给出能量、压力、化学势。统计力学的方法论可以浓缩成一句口诀：算 $q$，然后微分。</span>

对化学家来说，配分函数的价值是**可计算的**：分子的平动、转动、振动、电子能级可由光谱与量子力学求出，代入 $q$ 再微分，就得到理想气体的热容、熵、平衡常数——从光谱数据预测热力学性质，是统计热力学最了不起的实用成就。

## 1 配分函数的定义

**配分函数（partition function）** 定义为对所有量子态的玻尔兹曼因子求和：

$$q = \sum_i g_i\, e^{-\varepsilon_i / kT} = \sum_\text{states} e^{-\varepsilon/kT}$$

它的名字来自「分配」：$g_i e^{-\varepsilon_i/kT}/q$ 正是粒子在能级 $i$ 上的分布概率。<span class="marginnote">配分函数是 $T$ 的函数，也隐含着体积（平动能级依赖容器尺寸）。它的量纲为 1，但不同运动的 $q$ 数值量级悬殊：平动配分函数动辄 $10^{30}$，振动配分函数接近 1，转动配分函数数百——这个量级差别本身就说明了各自由度的激发程度。</span>

**重点：$q$ 是「有效状态数」的加权平均。** 若所有能级简并且能量为零，$q = g_0$ 就是基态量子态数；能量越高，权重越小。$q$ 越大，说明系统「有效可达」的量子态越多，对应的熵与热容也越大。

## 2 配分函数的分解：独立运动的乘积

分子的运动近似可分解为相互独立的平动、转动、振动、电子运动，总能量为各项之和：

$$\varepsilon = \varepsilon_\text{tr} + \varepsilon_\text{rot} + \varepsilon_\text{vib} + \varepsilon_\text{el}$$

因为 $e^{-\varepsilon/kT} = e^{-\varepsilon_\text{tr}/kT} \cdot e^{-\varepsilon_\text{rot}/kT} \cdots$ 是乘积，求和可拆为连乘：

$$q = q_\text{tr} \cdot q_\text{rot} \cdot q_\text{vib} \cdot q_\text{el}$$

**重点：独立自由度对应的配分函数相乘，而不是相加。** 这是玻尔兹曼因子的指数结构带来的简化——把复杂分子的求和分解成几个简单运动的分别求和，再相乘。对多原子分子，还要把振动按简正模式继续分解为 $3N-6$ 个一维谐振子之积。<span class="marginnote">「独立运动相乘」对应「独立自由度能量相加」。量子力学告诉我们分子能级确实可近似写为平动+转动+振动+电子之和，光谱学正是按这个框架解析谱线的——转动能级间隔很小（毫米波/远红外），振动间隔较大（红外），电子间隔最大（紫外可见）。</span>

## 3 各运动项的配分函数

各运动配分函数的形式如下（$h$ 普朗克常数，$\Theta$ 特征温度）：

- **平动**（一维势箱求和推广到三维）：$q_\text{tr} = \left(\dfrac{2\pi mkT}{h^2}\right)^{3/2} V$，与体积成正比，量级巨大。
- **转动**（双原子分子）：$q_\text{rot} = \dfrac{T}{\sigma\Theta_\text{rot}}$，其中 $\Theta_\text{rot} = h^2/(8\pi^2 Ik)$，$I$ 是转动惯量，$\sigma$ 是对称数。
- **振动**（谐振子，基态能量取 $\frac{1}{2}h\nu$）：$q_\text{vib} = \dfrac{e^{-\Theta_\text{vib}/2T}}{1 - e^{-\Theta_\text{vib}/T}}$，$\Theta_\text{vib} = h\nu/k$。
- **电子**：电子能级间隔大，通常 $q_\text{el} = g_0 e^{-\varepsilon_0/kT}$ 只取基态简并度（$g_0$），激发态可忽略。

<span class="marginnote"><strong>特征温度（characteristic temperature）$\Theta = \Delta\varepsilon/k$</strong> 把能级间隔换算成温度标尺：$T \gg \Theta$ 时该自由度充分激发、贡献经典热容；$T \ll \Theta$ 时被冻结。水的转动 $\Theta_\text{rot} \sim 20\ \text{K}$、振动 $\Theta_\text{vib} \sim 2300\ \text{K}$——室温下转动全激发、振动基本冻结，水蒸气热容因此介于两者之间。</span>

**辨析｜易错点：** 振动配分函数的能量零点有两种约定：取基态为零或取阱底为零，两者差一个常数因子 $e^{-\Theta_\text{vib}/2T}$。这个常数不影响熵与热容（取对数再微分后抵消），但影响内能与自由能的绝对值。使用公式前先弄清能量零点约定。

## 4 公式解析：平动配分函数

$$q_\text{tr} = \left(\frac{2\pi mkT}{h^2}\right)^{3/2} V$$

**这是统计力学里最常被计算的配分函数，也揭示了量子极限如何回归经典。** 四步拆解：

- **第一步，看一维的起源**：一维势箱中粒子的能级 $\varepsilon_n = n^2 h^2/(8mL^2)$ 是量子化的。当 $kT \gg h^2/(8mL^2)$（室温下大箱子显然满足），能级间距远小于 $kT$，求和可换成积分 $\sum e^{-\varepsilon/kT} \to \int_0^\infty e^{-n^2 h^2/(8mL^2 kT)}\,\mathrm{d}n$，积分给出 $q_\text{tr,1D} = \sqrt{2\pi mkT}\,L/h$。
- **第二步，推广到三维**：三个方向独立，$q_\text{tr} = q_x q_y q_z$，$L_x L_y L_z = V$，得到上式。立方根号内是热德布罗意波长 $\Lambda = h/\sqrt{2\pi mkT}$ 的倒数，故 $q_\text{tr} = V/\Lambda^3$。
- **第三步，看数量级**：室温下 $\Lambda \sim 0.1\ \text{nm}$ 量级，$q_\text{tr} = V/\Lambda^3 \sim 10^{30}$——平动量子态极其密集，宏观上表现为连续能谱，这就是经典气体「貌似连续」的原因。
- **第四步，看温度与体积依赖**：$q_\text{tr} \propto T^{3/2} V$。代入平均能量公式 $\bar\varepsilon = kT^2(\partial\ln q/\partial T)_V = \frac{3}{2}kT$，恰好回归麦克斯韦的 $\frac{3}{2}kT$ 平动能量——量子求和与经典极限在此完美闭合。

## 5 由配分函数读出热力学量

**重点：$\ln q$ 及其导数携带全部热力学信息。** 对定域粒子系统（可区分粒子，如晶体中的原子），核心公式为：

$$\begin{aligned}
U &= NkT^2\left(\frac{\partial\ln q}{\partial T}\right)_{N,V} \\
S &= Nk\ln q + \frac{U}{T} \\
A &= -NkT\ln q
\end{aligned}$$

对非定域系统（不可区分的气体分子），须除以 $N!$：$A = -NkT\ln(q/N) - NkT$（斯特林近似 $\ln N! \approx N\ln N - N$）。<span class="marginnote">为什么气体要除以 $N!$？因为全同分子交换不产生新微观状态，而玻尔兹曼计数把可区分粒子算多了 $N!$ 倍。不除 $N!$ 会出现两个物理错误：熵不广延（加倍系统熵不翻倍）和混合熵悖论（吉布斯佯谬）。除以 $N!$ 的修正也顺便修正了化学势与平衡常数，使气体反应的统计计算与实验一致。</span>

有了 $U$、$S$、$A$，其他量依次导出：$G = A + pV$，$p = -(\partial A/\partial V)_T = NkT/V$（理想气体状态方程从统计力学重新长出来），$\mu = (\partial A/\partial n)_T = -RT\ln(q/N)$（化学势与配分函数挂钩）。

## 6 配分函数计算实例

把配分函数的公式落到一个具体分子上，体会「求和 → 微分 → 热力学量」的完整套路。以 $25\,^\circ\text{C}$ 的 $\ce{HCl}$ 气体为例（刚性转子-谐振子近似）：

**平动项**：$q_\text{tr} = \dfrac{(2\pi mkT)^{3/2}}{h^3}V$。$\ce{HCl}$ 质量 $m = 36.5/6.02\times10^{23}\ \text{g}$，算得 $q_\text{tr} \approx 2.7\times10^{30}$（对 $V = 1\ \text{m}^3$）——平动量子态密集到几乎连续。

**转动项**：$q_\text{rot} = \dfrac{8\pi^2IkT}{\sigma h^2}$。$\ce{HCl}$ 的转动惯量 $I \approx 2.7\times10^{-47}\ \text{kg·m}^2$，$\sigma = 1$（异核），得 $q_\text{rot} \approx 19.6$——约 20 个转动量子态被热激发。

**振动项**：$q_\text{vib} = \dfrac{e^{-\Theta_\text{vib}/2T}}{1 - e^{-\Theta_\text{vib}/T}}$。$\ce{HCl}$ 振动频率 $\nu \approx 8.97\times10^{13}\ \text{s}^{-1}$，$\Theta_\text{vib} = h\nu/k \approx 4300\ \text{K}$，室温下 $q_\text{vib} \approx 0.0068$——振动几乎全在基态（零点能占主导）。

**内能分配**：$\bar\varepsilon = kT^2\partial\ln q/\partial T = \frac{3}{2}kT + kT + kT\left(\frac{\Theta_\text{vib}/2T}{1-e^{-\Theta_\text{vib}/T}} + \frac{\Theta_\text{vib}/T}{e^{\Theta_\text{vib}/T}-1}\right)$。数值：平动 $\frac{3}{2}kT$、转动 $kT$、振动接近零点能——**平动、转动、振动对能量的贡献依激发程度分化**，这正是热容的统计来源。

**辨析｜易错点：** 三个配分函数的量级悬殊（$10^{30}$ vs $20$ vs $0.007$）不是「重要性排序」——能量与熵看的是 $\ln q$ 对温度/体积的导数，而非 $q$ 本身。平动 $q$ 虽大，其熵贡献却与转动熵同量级，因为 $S \propto \ln q + \cdots$。**用 $q$ 的绝对值比大小会误导，要比较的是各运动对 $\ln q$ 的贡献。**

一个把配分函数与宏观量连接起来的数值体验。取 $\ce{HCl}$ 转动配分函数 $q_\text{rot} \approx 19.6$（$25\,^\circ\text{C}$，前节算得）。由转动内能公式：

$$\bar\varepsilon_\text{rot} = kT^2\frac{\partial\ln q_\text{rot}}{\partial T} = kT^2\cdot\frac{1}{T} = kT$$

**转动内能恰为 $kT$（双原子分子 2 个转动自由度 × $\frac{1}{2}kT$）**——经典均分的结果从配分函数自动涌现。再看转动熵：$S_\text{rot} = Nk[\ln(T/\sigma\Theta_\text{rot}) + 1] = R[\ln 19.6 + 1] = 8.314\times(2.98+1) \approx 33\ \text{J·mol}^{-1}\text{K}^{-1}$。

**这条计算的深层含义**：配分函数 $q$ 是一个数，但它的**对数对温度、体积的导数**承载了全部热力学量——$q$ 本身（19.6）与熵（33 J/mol/K）之间隔着「$\ln$ + 微分」的桥梁。**这就是「配分函数是统计热力学的总账本」的实操含义**：算 $q$ 只是第一步，会从 $q$ 读出各种宏观量才是掌握。

**辨析｜易错点：** 转动配分函数公式 $q_\text{rot} = T/\sigma\Theta_\text{rot}$ 要求 $T \gg \Theta_\text{rot}$（高温极限）。对 $\Theta_\text{rot} > T$ 的分子（如 $\ce{H2}$，$\Theta_\text{rot} \approx 85\ \text{K}$），低温下必须保留转动能级的离散求和，不能直接套公式——「先验 $T/\Theta$ 再决定用积分还是求和」是配分函数计算的纪律。

把配分函数放回统计热力学的核心。**它是「微观求和」与「宏观量」之间的总账本**——$q$ 本身只是一个数，但 $\ln q$ 及其对温度、体积、粒子数的导数给出了内能、熵、压力、化学势的全部热力学量。独立运动相乘（$q = q_\text{tr}q_\text{rot}q_\text{vib}q_\text{el}$）的分解，让复杂分子能级求和变成几个简单运动的分别处理。

**重点：配分函数是统计热力学的「计算方法论」**。从玻尔兹曼分布出发定义 $q$，从 $q$ 微分出宏观量，代入具体分子能级得到数值——这套「三步走」贯穿《理想气体统计》《晶体统计》《统计热力学应用》全部后续章节。掌握配分函数，等于掌握了从分子光谱到热力学性质的标准翻译流程，这也是统计热力学区别于「背公式」的关键：**一切宏观热力学都可以从 $q$ 重新长出来**。

一个补充的辨析：配分函数的零点选择（振动基态取零还是阱底取零）影响 $U$、$A$ 的绝对值，但不影响熵与热容——**计算相对量（熵变、热容、平衡常数）时零点约定可忽略，计算绝对量时必须注明**。

## 7 小结

- **配分函数** $q = \sum_i g_i e^{-\varepsilon_i/kT}$ 是「有效状态数」，全部热力学量都由它及其导数给出。
- 独立运动对应配分函数**相乘**：$q = q_\text{tr} q_\text{rot} q_\text{vib} q_\text{el}$。
- 平动 $q_\text{tr} = (2\pi mkT/h^2)^{3/2}V$，量级 $10^{30}$，量子求和回归经典 $\frac{3}{2}kT$。
- 特征温度 $\Theta = \Delta\varepsilon/k$ 决定自由度激发与否。
- 定域系统 $A = -NkT\ln q$；**非定域气体须除以 $N!$**（吉布斯佯谬修正），由此导出状态方程与化学势。

在下一节，我们将把「由配分函数读热力学量」系统地展开——每一项宏观量（内能、熵、自由能、化学势）的统计本质分别是什么，这正是**热力学量的统计解释**。
