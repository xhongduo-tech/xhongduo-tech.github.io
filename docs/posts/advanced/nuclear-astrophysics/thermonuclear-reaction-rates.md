---
title: 热核反应率
date: 2026-08-07
---

# 热核反应率

<div class="epigraph">
<p>对元素起源的探索，是我一生科学工作的主线。</p>
<footer>—— 威廉·A·福勒（William A. Fowler），《追寻元素的起源》，1983 年诺贝尔物理学奖演讲</footer>
</div>

<div class="article-byline">
<p>第四级 · 核天体物理 ｜ Iliadis《Nuclear Physics of Stars》Ch.3 · Clayton《Principles of Stellar Evolution and Nucleosynthesis》Ch.4 ｜ 2026-08-07</p>
</div>

## 为什么从反应率开始

上一节的截面回答"一次碰撞在给定能量下成不成立"，但恒星内部是 $10^7$ K 的热气体，粒子速度各不相同，分布服从麦克斯韦-玻尔兹曼律。恒星真正需要的量是**热核反应率（thermonuclear reaction rate）**：单位体积单位时间内某反应发生多少次。它是连接微观核物理与宏观恒星结构的中枢——把反应率代入恒星结构方程，才能得到温度、密度与核合成的自洽解，这正是《恒星结构与演化》专题里 pp 链与 CNO 循环模型的核心输入。本节我们把反应率公式拆开，区分非共振、窄共振与宽共振三种情形，并认识核天体物理界的"标准答案集"——NACRE 数据库。

## 1 反应率的定义与平均

两种粒子 $1$、$2$ 发生反应，数密度分别为 $N_1$、$N_2$。单位体积单位时间内的反应次数为

$$r_{12} = \frac{N_1 N_2}{1+\delta_{12}}\,\langle\sigma v\rangle$$

其中 $\delta_{12}$ 是克罗内克符号（同种粒子时避免重复计数），$\langle\sigma v\rangle$ 是**反应率参数（reactivity）**——对质心系相对速度 $v$ 求平均：

$$\langle\sigma v\rangle = \int_0^\infty \sigma(v)\,v\,\phi(v)\,dv$$

$\phi(v)$ 是相对速度分布。对非简并等离子体，$\phi(v)$ 取麦克斯韦-玻尔兹曼分布，于是可把积分改写为对质心能量 $E=\frac{1}{2}\mu v^2$ 的形式：

$$\langle\sigma v\rangle = \left(\frac{8}{\pi\mu}\right)^{1/2}\frac{1}{(kT)^{3/2}}\int_0^\infty \sigma(E)\,E\,\exp\!\left(-\frac{E}{kT}\right)dE$$

<span class="marginnote">若粒子相对速度不服从麦克斯韦分布——例如存在高能非热尾巴或简并——反应率要改用相应分布重算。太阳核心非简并，麦克斯韦分布成立；而第 6 篇的硅燃烧区已近简并，处理上要更小心。</span>

## 2 非共振反应率：Gamow 窗口积分

把第 2 篇的截面公式 $\sigma(E)=S(E)\,e^{-2\pi\eta}/E$ 代入积分，得

$$\langle\sigma v\rangle \propto \int_0^\infty S(E)\exp\!\left(-\frac{E}{kT}-\frac{b}{\sqrt{E}}\right)dE$$

被积函数在 Gamow 峰 $E_0=(bkT/2)^{2/3}$ 处取极大。对平滑的 $S(E)$，可把 $S(E)$ 提到积分号外，算出解析近似：

$$\langle\sigma v\rangle \approx \frac{(8/\pi\mu)^{1/2}}{(kT)^{3/2}}\,S(E_0)\,\Delta E\,\exp\!\left(-\frac{3E_0}{kT}\right),\qquad \Delta E = 4\sqrt{\frac{E_0 kT}{3}}$$

<span class="marginnote">$\Delta E$ 是 Gamow 窗口的有效半宽。对 $\ce{^1H+^1H}$ 在太阳中心约 1.5 keV——只有整个麦克斯韦谱的一小段在"干活"，其余粒子太慢或太少。这个窗口概念是第 2 篇公式解析的延续。</span>

这条近似公式是整篇反应率理论的"工作马"：只要在窗口能量 $E_0$ 处取一个 S 因子值，就够算出量级正确的反应率。它也直接暴露了误差来源——$S(E_0)$ 若在外推中带 ±30% 误差，反应率就带同样量级的误差，而 $E_0$ 的误差会指数放大。核天体物理实验的全部苦心，就是要把 $S(E_0)$ 的误差压到最小。

**温度敏感性由此而来**：$E_0\propto T^{2/3}$，因而反应率大体按 $\exp(-c/T^{1/3})$ 变化，通常写作 $\langle\sigma v\rangle\propto T^{n}$ 的幂律。pp 反应 $n\approx4$，而 CNO 环的 $\ce{^14N(p,\gamma)^{15}O}$ $n\approx15$—20。温度每升一点，CNO 就比 pp 放能更快——这正是主序星高温一侧 CNO 取代 pp 的原因。

一个量级直觉：太阳中心的 pp 反应率参数 $\langle\sigma v\rangle_{\mathrm{pp}}\approx10^{-49}$ cm³/s，配合质子数密度 $N_p\approx6\times10^{25}$ cm⁻³，得 pp 反应率约 $10^{19}$ cm⁻³ s⁻¹——每个质子平均要等待约 100 亿年才反应一次。**恒星燃烧所以缓慢而持久，根源在于弱相互作用起跳的反应截面极小**；这份"慢"恰恰是恒星能稳定发光几十亿年的原因。

**辨析｜易错点：** 初学者常把 "Gamow 峰能量 $E_0$" 与 "粒子最可几能量" 混为一谈。前者是麦克斯韦分布与隧穿因子乘积的峰，即**对反应贡献最大的能量**；后者才是速度分布自身的峰值 $E\approx kT$。$E_0$ 恒大于 $kT$——粒子必须"偏快"才能有效反应，二者之差随库仑势垒 $Z_1 Z_2$ 增大而增大。检查自己有没有理解：为什么 $\ce{^12C + ^12C}$ 的 $E_0$ 比 $\ce{^1H + ^1H}$ 高得多？答案是 $Z_1 Z_2$ 从 1 涨到 36，隧穿更难，Gamow 峰被迫推向高能。

## 3 窄共振反应率：Breit–Wigner 替换

当入射能量落在某个复合核共振能级 $E_R$ 附近，且共振宽度 $\Gamma\ll\Delta E$（窄共振）时，截面由 Breit–Wigner 公式主导，积分可以解析完成。结果为

$$\langle\sigma v\rangle = \left(\frac{2\pi}{\mu kT}\right)^{3/2}\,\hbar^2\,\left(\omega\gamma\right)\,\exp\!\left(-\frac{E_R}{kT}\right)$$

其中共振强度为

$$\omega\gamma = \frac{2J+1}{(2J_1+1)(2J_2+1)}\,\frac{\Gamma_{\mathrm{in}}\Gamma_{\mathrm{out}}}{\Gamma}$$

<span class="marginnote">$\omega\gamma$ 中 $\omega$ 是自旋统计因子，$J,J_1,J_2$ 是共振态与两粒子的自旋；$\Gamma_{\mathrm{in}}$、$\Gamma_{\mathrm{out}}$ 是入射道与出射道宽度，$\Gamma=\sum_i\Gamma_i$ 为总宽度。共振强度可直接由实验给出，是核天体物理数据表的基本条目。</span>

**窄共振的关键特征是与细节无关**：只要知道 $E_R$ 与 $\omega\gamma$，无需知道共振形状就能算反应率；而共振是否落在 Gamow 窗口内，对反应率的量级有决定性影响。

把三种情形的算法并列，差异一目了然：

| 情形 | 判据 | 反应率来源 | 温度依赖 |
| --- | --- | --- | --- |
| 非共振（直接俘获） | 窗口内无强共振 | $S(E_0)\,\Delta E\,\exp(-3E_0/kT)$ | $T^{n}$，$n$ 较小 |
| 窄共振 | $\Gamma\ll\Delta E$ 且 $E_R$ 在窗口 | $\omega\gamma\,\exp(-E_R/kT)$ | 强指数，$T$ 高时饱和 |
| 宽共振 | $\Gamma\gg\Delta E$ | 对 Breit–Wigner 逐点积分 | 接近非共振行为 |

这张表是阅读反应率数据文件的"速查表"：看到一个共振条目，先判断它是窄还是宽，就知道该用哪个公式、对温度有多敏感。

## 4 公式解析：$\ce{^12C(\alpha,\gamma)^{16}O}$ 的窄共振之争

这个反应是氦燃烧的"胜负手"，我们第 5 篇还会讲它的天体物理后果。这里只看它怎么被算进反应率：

- **第一步，找共振**：$^{16}\mathrm{O}$ 的低激发态中，$E_R\approx2424$ keV 附近的 $1^-$ 态是亚阈共振（略低于阈值），其作用经外推进入恒星能区；恒星能区的有效贡献主要来自亚阈态与直接俘获的干涉。
- **第二步，组装 $\omega\gamma$**：入射道是 $\alpha+^{12}\mathrm{C}$，出射道是 $\gamma$。$\Gamma_{\alpha}\ll\Gamma_{\gamma}$ 时 $\omega\gamma\approx\omega\Gamma_\alpha$，于是反应率由 $\alpha$ 宽度主导，而 $\Gamma_\alpha$ 又依赖 $\alpha$ 在库仑势垒中的隧穿。
- **第三步，带入窄共振公式**：在氦燃烧温度 $T\approx2\times10^8$ K，$E_R\gg kT$ 处共振大多不在窗口正中，需要逐共振求和再加直接俘获项——这就是为什么 $\ce{^12C(\alpha,\gamma)^{16}O}$ 的恒星反应率至今仍有 10% 以上不确定度，成为核天体物理最大单项误差源之一。<span class="marginnote">这个反应的后果深远：它决定恒星最终留下的是 $\ce{^12C}$ 为主还是 $\ce{^{16}O}$ 为主。我们的身体里碳氧之比，源头就在这个反应率里。</span>

## 5 宽共振与电子屏蔽：两条修正

不是所有共振都"窄"：

- **宽共振（broad resonance）**：$\Gamma$ 与 Gamow 窗口相当或更宽时，不能再用 $\exp(-E_R/kT)$ 近似，必须在窗口内对 Breit–Wigner 形状逐点积分。$\Gamma\gg\Delta E$ 时共振对整个窗口"平均贡献"，其反应率更像非共振情形。<span class="marginnote">Iliadis Ch.3 明确给出判据：$|\Gamma|\lesssim\Delta E$ 用窄共振公式，$|\Gamma|\gg\Delta E$ 用逐点积分；介于其间需要数值处理。</span>
- **电子屏蔽（electron screening）**：恒星等离子体中的自由电子与背景离子会部分屏蔽靶核的库仑势，等效降低势垒、提高有效能量处的截面。屏蔽因子 $f\approx\exp(Z_1 Z_2 e^2 / (kT\,R_{\mathrm{Debye}}))$ 在高密度、低温和高 $Z_1 Z_2$ 时不可忽略。<span class="marginnote">太阳中心的屏蔽效应仅让 pp 反应率增大约百分之几，但对高 $Z$ 反应可达数倍；实验室里靶内束缚电子的屏蔽（"增强因子"）与恒星等离子体屏蔽是两回事，实验外推时必须区分——这也是 LUNA 等地下实验室研究的对象之一。</span>

## 6 NACRE 与反应率数据库

理论公式需要大量实测输入。1999 年发布的 **NACRE 数据库（Nuclear Astrophysics Compilation of Reaction Rates）** 整理了约 86 个对恒星核合成最重要的带电粒子反应的反应率推荐值、上下限与适用温度范围，是研究者的标准起点。<span class="marginnote">后继者有 REACLIB（美国）、JINA REACLIB、STARLIB 等更大规模的数据集，覆盖面从 pp 链一直延伸到 r 过程上万条反应；第 12 篇讲反应网络时我们会用到它们。</span>

对每个反应，NACRE 给出的都是"推荐值 + 低/高误差带"：低端和高端分别对应不同的物理假设（如亚阈共振强度、干涉符号的取舍）。用误差带跑一遍恒星模型，得到的就是**反应率不确定度对元素产率的影响**——现代核天体物理几乎每条重要结论都要附上这样的误差条。

怎么读一张 NACRE 数据页，值得给初学者几条实用提示：

- **适用温度范围**：每个反应只对一段 $T_9$（以 $10^9$ K 为单位的温度）有效，超出范围要用别的公式或数据源。
- **单位约定**：带电粒子反应率常用 cm³ mol⁻¹ s⁻¹（摩尔量纲），换算到 $\langle\sigma v\rangle$（cm³ s⁻¹）要乘阿伏伽德罗常数相关因子，照抄公式前先核对单位。
- **误差不对称**：许多反应率的误差带上下不对称，取自共振参数的保守上下限，不能简单当作对称高斯误差处理。

<span class="marginnote">$T_9$ 是核天体物理常用温度单位：$T_9=1$ 表示 $10^9$ K。氢燃烧在 $T_9\approx0.015$，氦燃烧 $T_9\approx0.1$，硅燃烧与爆炸燃烧进入 $T_9\gtrsim3$。记住这几个量级，读恒星模型论文就能快速定位"现在烧到哪一步"。</span>

## 7 小结

反应率理论将在后续各篇反复出现，先记住各燃烧阶段的温度坐标：氢燃烧 $T_9\approx0.015$、氦燃烧 $T_9\approx0.1$、碳燃烧 $T_9\approx0.6$、氖/氧燃烧 $T_9\approx1$–2、硅燃烧与爆炸燃烧 $T_9\gtrsim3$。随着温度上升，Gamow 峰移向高能、库仑势垒更重的反应才被"点燃"，这就是恒星逐级烧到铁峰的动力学原因。

- **反应率** $r=\dfrac{N_1 N_2}{1+\delta_{12}}\langle\sigma v\rangle$ 是恒星核合成的宏观速率。
- **非共振**情形由 Gamow 窗口积分给出，$\langle\sigma v\rangle\propto T^n$，$n$ 越大对温度越敏感。
- **窄共振**由 $E_R$ 与 $\omega\gamma$ 唯一决定；**宽共振**需逐点积分。
- **电子屏蔽**与**电子简并**会修正标准麦克斯韦分布，高密度区不可忽略。
- **NACRE / REACLIB / STARLIB** 提供带误差带的反应率数据集，是模型的输入标准。
- 实际恒星模型中还要同时解**逆反应**：正逆反应率通过细致平衡联系，高温高密度下逆反应（光致离解、$(\gamma,\alpha)$ 等）不再可忽略，这正是第 6 篇核统计平衡的伏笔。

在下一节，我们把武器对准恒星的第一场燃烧——**氢燃烧**：pp 链如何用弱相互作用起跳、CNO 循环如何靠温度优势接管，以及每一轮燃烧如何伴随中微子把能量悄悄带走。值得注意的是，氢燃烧的总能量释放（每合成一个 $^4\mathrm{He}$ 约 26.7 MeV）几乎与具体路径无关——差别只在于温度门槛与中微子损耗，这正是第 4 篇的开场。
