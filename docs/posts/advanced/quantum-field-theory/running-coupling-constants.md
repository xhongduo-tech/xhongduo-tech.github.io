---
title: 跑动耦合常数
date: 2026-08-07
---

# 跑动耦合常数

<div class="epigraph">
<p>耦合常数不是常数——它是能量尺度的函数，而这个函数就是理论的全体。</p>
<footer>—— 自标度物理传统（为本文所作）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §12.2, §16.7 ｜ 2026-08-07</p>
</div>

## 为什么「常数」会跑

上一节的真空极化图给出了第一个「电荷随尺度变」的物理图像。
本节把它上升为普遍规律：**耦合常数的跑动（running）**——从 β 函数的单圈解、到 QED 的屏蔽行为与 Landau 极点、再到实验如何在 $m_Z$ 能标测出 $\alpha(m_Z) \approx 1/128$。
这是重正化群最「可触摸」的产出：它把「1/137」从一个神秘的常数，变成一条随能量变化的曲线上的一个点。<span class="marginnote">「精细结构常数 $= 1/137$」只是 $\alpha$ 在零动量/低能（电子质量附近）的数值。在 $Z$ 玻色子质量 $m_Z \approx 91$ GeV，$\alpha \approx 1/128$——这不是矛盾，是同一理论在不同尺度的读数。</span>

## 1 有效电荷与跑动的一般解

把 β 函数 ${\beta(e) = \frac{e^3}{12\pi^2}}$ 代入跑动方程，解得 QED 的**有效精细结构常数**：

$$\bar\alpha(\mu) = \frac{\alpha(\mu_0)}{1 - \frac{2\alpha(\mu_0)}{3\pi}\log\frac{\mu}{\mu_0}}$$

这里 $\bar\alpha = e^2/4\pi$，$\mu_0$ 是参考标度（通常取电子质量或零动量）。
这个解的**形状**对任何「β > 0」的理论都一样：分母随 $\mu$ 增大而减小，$\bar\alpha$ 增大。<span class="marginnote">单圈跑动的普适形状：$\bar g^2(\mu) = \frac{g^2(\mu_0)}{1 - b_0 g^2(\mu_0)\log(\mu/\mu_0)}$，$b_0$ 由理论决定。三个理论的 $b_0$：QED $> 0$、$\phi^4 > 0$、QCD $< 0$（无夸克贡献时）。<strong>符号决定跑向。</strong></span>

## 2 三种性格：屏蔽、加强、反屏蔽

跑动的方向由**真空的极化性质**决定：

- **QED（β > 0，屏蔽）**：虚正反电子对围绕电荷形成**屏蔽云**。距离越近，看到的裸电荷越大；能量越高（探测越深），$\alpha$ 越大。低能 $\alpha \approx 1/137$，$m_Z$ 处 $\alpha \approx 1/128$。
- **$\phi^4$（β > 0）**：无规范结构，圈图直接加强耦合。高能耦合爆炸 → Landau 极点。
- **QCD（β < 0，反屏蔽，见下节）**：胶子自耦合（非阿贝尔规范场）产生**反屏蔽**，高能耦合变弱——渐近自由。这是强相互作用的"性格反转"。<span class="marginnote">QCD 的 β 负号来自胶子（规范场）的自耦合——光子没有自耦合，所以 QED 是屏蔽；胶子自身带「色荷」，虚胶子云<strong>反屏蔽</strong>色荷，让距离越近色荷越小。<strong>反屏蔽是非阿贝尔规范场独有的签名。</strong></span>

跑动耦合的意义在于**让微扰论有了适用范围**：微扰展开的好坏由 $\bar\alpha(\mu)$ 决定，而不是某固定值。QCD 低能（$\mu \sim 1$ GeV）$\alpha_s \sim 1$，微扰失效（夸克禁闭）；高能（$\mu \sim 100$ GeV）$\alpha_s \sim 0.1$，微扰可靠。

## 3 实验测定：α 在哪几个标度被测量

跑动耦合不是纸面数学——它是被实验精确测定的：

| 能标 | 观测量 | 测定值 |
| --- | --- | --- |
| $q^2 \approx 0$（低能） | 电子 $g-2$、氢原子精细结构 | $\alpha \approx 1/137.036$ |
| $q^2 = m_\tau^2$ | $\tau$ 轻子衰变中的强子截面 | $\alpha_s(m_\tau) \approx 0.33$ |
| $q^2 = m_Z^2$ | $Z$ 衰变率、喷注结构 | $\alpha(m_Z) \approx 1/128$，$\alpha_s(m_Z) \approx 0.118$ |

这些测量值沿着同一条 RG 演化曲线互相印证——**不同标度测得的 $\alpha$ 值落在 β 函数预言的同一条跑动曲线上，是重正化群最重要的实验验证**。<span class="marginnote">强耦合 $\alpha_s$ 的跑动是「多个实验点拼一条曲线」的典范：从 $\alpha_s(m_\tau) \approx 0.33$ 到 $\alpha_s(m_Z) \approx 0.118$，比值精确符合 QCD 单圈（含次领头阶）β 函数。这既是渐进自由的证据，也是 QCD 作为正确理论的证据。</span>

## 4 公式解析：单圈跑动解

**跑动方程的解 = β 函数积分 + 一条边界条件。** 拆解四步：

$$
\bar\alpha(\mu) = \frac{\alpha(\mu_0)}{1 - \frac{2\alpha(\mu_0)}{3\pi}\log\frac{\mu}{\mu_0}}
$$

- **第一步，积分 β 方程**：$\frac{d\bar\alpha}{d\log\mu} = \frac{2\bar\alpha^2}{3\pi}$（用 $\alpha = e^2/4\pi$ 重写 $\beta(e) = e^3/12\pi^2$）。分离变量：$\frac{d\bar\alpha}{\bar\alpha^2} = \frac{2}{3\pi}d\log\mu$。
- **第二步，定积分**：从 $\mu_0$ 积到 $\mu$：$-\frac{1}{\bar\alpha} + \frac{1}{\alpha(\mu_0)} = \frac{2}{3\pi}\log\frac{\mu}{\mu_0}$。
- **第三步，反解**：移项得 $\bar\alpha = \frac{\alpha(\mu_0)}{1 - \frac{2\alpha(\mu_0)}{3\pi}\log(\mu/\mu_0)}$。这是 β 函数一阶微分的精确解（单圈精度）。
- **第四步，读 Landau 极点**：当分母为零，$\mu = \mu_0\exp\left(\frac{3\pi}{2\alpha(\mu_0)}\right)$。对 $\alpha \approx 1/137$，这个标度高达 $\sim 10^{286}$ GeV——远超过普朗克标度，说明 QED 的有效性延伸到极远，但理论上仍不是完整的紫外理论。

## 5 辨析｜易错点

- **$\alpha = 1/137$ 不是「错的」**：它在低能是对的。问「$\alpha$ 是多少」必须同时问「在哪个能标」。实验报告的 $\alpha^{-1} = 137.036$ 是低能极限值。<span class="marginnote"><strong>$\alpha_s$ vs $\alpha$ 别混</strong>：$\alpha$ 是 QED 的（电磁），$\alpha_s$ 是 QCD 的（强）。$\alpha_s(m_Z) \approx 0.118$ 而 $\alpha(m_Z) \approx 1/128 \approx 0.0078$——强耦合比电磁耦合大一个多数量级，这正是「强核力」叫「强」的原因。</span>
- **Landau 极点不是「QED 的预言」**：它是「QED 作为独立理论不自洽」的判据。实际宇宙里 QED 在高能会与其他相互作用统一（电弱统一），所以 $\alpha$ 跑到 $1/128$ 就会被新物理接管——不是跑到 $\infty$。
- **跑动耦合 ≠ 物理量随能量真的「变强」**：耦合常数是「参数」，随能标变的是「有效耦合」（吸收了大 log 重求和后的展开参数）。可测物理量（截面）对 $\mu$ 不变，只是用不同能标的参数展开微扰而已。

## 6 延伸：跑动耦合与「有效理论」世界观

跑动耦合把「一个常数」升级成「一条曲线」，这个转变背后是现代物理的**有效理论世界观**：

- **每个能标一套参数**：低能我们写 $\mathcal{L}_{\text{eff}}$，参数是低能值；高能（如果有新物理）换一套。跑动曲线把「不同能标的参数」连起来，是有效理论之间的「汇率表」。
- **大对数重求和**：单圈跑动解 $\bar\alpha(\mu)$ 实际上把 $\alpha\log(\mu/\mu_0)$ 这类「大对数」重求和了。不跑动而用固定 $\alpha$ 时，微扰展开里 $\log(\mu/\mu_0)$ 会破坏收敛；跑了之后，这些对数被吸收进参数，展开才可靠。
- **Landau 极点 = 理论失效的预告**：$\phi^4$ 与 QED 的 Landau 极点不是物理事件，而是「这个理论在某个标度之前一定需要新物理」的判据。标准模型里 Higgs 自耦合 $\lambda$ 的跑动甚至暗示真空在极高能可能不稳——这仍是活跃研究。

回到主线：**「参数随标度跑」不是误差，而是理论的动力学内容**。就像大模型里的「学习率随步数衰减」不是 bug 而是训练策略——标度依赖常常是系统最重要的信息载体。

### 自测清单

- [ ] 能解 QED 单圈跑动方程并写出 $\bar\alpha(\mu)$。
- [ ] 能说出屏蔽/反屏蔽的物理图像。
- [ ] 能记住 $\alpha \approx 1/137$、$\alpha(m_Z) \approx 1/128$、$\alpha_s(m_Z) \approx 0.118$。
- [ ] 能解释 Landau 极点的意义。

<span class="marginnote">把「耦合常数」改成「耦合函数」——<strong>常数是近似，函数是真相</strong>。这一小步改观，就是整个跑动物理的起点。</span>

### 延伸阅读指引

- 进阶推导：P&S §12.2 的 $\beta$ 函数计算、§16.7 的 QCD 单圈 $\beta$；想深入「跑动耦合为何吸收大对数」可读 §12.1 的 Wilson 视角。
- 实验来源：PDG（Particle Data Group）的 Review of Particle Physics 每年更新 $\alpha_s$ 世界平均值；对比 $\alpha_s(m_\tau)$ 与 $\alpha_s(m_Z)$ 的跑动验证。
- 联系主线：跑动耦合是「有效理论」世界观的入口，与《凝聚态物理》里的重整化群、以及机器学习里的「学习率标度」是同一思想的三处投影。

把这四个数刻进记忆：$\alpha \approx 1/137$、$\alpha(m_Z) \approx 1/128$、$\alpha_s(m_Z) \approx 0.118$、$\Lambda_{\text{QCD}} \approx 200$ MeV——它们是一张「耦合曲线」上的四个锚点。

### 本节记忆锚点

- 单圈解形状：$\bar\alpha(\mu) = \alpha(\mu_0)/\left(1 - b_0\alpha(\mu_0)\log\frac{\mu}{\mu_0}\right)$，$b_0$ 的符号决定性格。
- 四个数字：$\alpha \approx 1/137$、$\alpha(m_Z) \approx 1/128$、$\alpha_s(m_Z) \approx 0.118$、$\Lambda_{\text{QCD}} \approx 200$ MeV。
- 屏蔽/反屏蔽：费米子圈屏蔽（QED）、胶子圈反屏蔽（QCD）。
- Landau 极点：QED 的失效标度极高，QCD 无此问题（β < 0）。
- 实测方法：用 $\alpha_s$ 的多个能标测定值拼一条 RG 曲线，是最直观的「跑动即物理」证据。
- 换算提醒：论文里报 $\alpha_s(m_Z) = 0.118$ 默认是 $\overline{\text{MS}}$、$\mu = m_Z$，换算前先核对方案。
- 交叉引用：与《粒子物理》《凝聚态》的重整化群章节对照。

## 7 小结

- 单圈跑动解 $\bar\alpha(\mu) = \frac{\alpha(\mu_0)}{1 - \frac{2\alpha(\mu_0)}{3\pi}\log(\mu/\mu_0)}$，QED β > 0。
- **QED 屏蔽**：虚对屏蔽电荷，$\alpha$ 随能量增大（$1/137 \to 1/128$）。
- **QCD 反屏蔽**：胶子自耦合使 $\alpha_s$ 随能量减小（渐近自由，下节）。
- 实验在 $m_\tau, m_Z$ 等多标度测定 $\alpha, \alpha_s$，落在同一条 RG 曲线上。
- Landau 极点是理论失效标度；物理上有新物理在更高能标介入。

在下一节，我们终于迎向 QCD 最震撼的发现——**渐进自由**：为什么强相互作用的耦合在高能反而变弱，这个「反直觉」如何由非阿贝尔规范场的胶子自耦合造成，并换来 2004 年诺贝尔物理学奖。


