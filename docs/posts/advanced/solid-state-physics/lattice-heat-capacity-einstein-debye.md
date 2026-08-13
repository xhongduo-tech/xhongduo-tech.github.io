---
title: 晶格热容：爱因斯坦与德拜模型
date: 2026-08-07
---

# 晶格热容：爱因斯坦与德拜模型

<div class="epigraph">
<p>比热不是热学的一个注脚，而是声子谱的望远镜。</p>
<footer>—— 据 P. 德拜（Peter Debye）</footer>
</div>

<div class="article-byline">
<p>第四级 · 固体物理 ｜ Kittel 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从晶格热容开始

把一块固体加热，热量花到哪里去了？经典图像里，能量均分定理说每个振动自由度贡献 $\tfrac12 k_BT$，于是每摩尔固体应有热容 $3R$（杜隆-珀替定律）。但实验很快打脸：金刚石在室温下热容远小于 $3R$，且温度越低热容越小、趋近于零。**量子声子论给出了正确解释**——而这正是上一讲声子理论的第一场实战。

从主线看，热容问题演示了「**色散关系 → 态密度 → 可观测量**」的标准流水线：先知道振动模式怎么分布（态密度 $g(\omega)$），再叠加玻色统计的平均能量，最后求导得到热容。这条流水线在第 8 章载流子统计、磁性理论里还会原样重演。

## 1 热容的统计定义与声子图像

晶格内能是所有声子模式能量之和（零点能不随温度变，可忽略）：

$$U = \sum_{\mathbf{k},s} \hbar\omega_s(\mathbf{k})\, \bar n(\mathbf{k},s), \qquad \bar n = \frac{1}{e^{\hbar\omega/k_BT}-1}$$

热容

$$C_V = \left(\frac{\partial U}{\partial T}\right)_V = \sum_{\mathbf{k},s} \hbar\omega \frac{\partial \bar n}{\partial T}$$

把求和换成对频率的积分 $\sum_{\mathbf{k},s} \to \int_0^{\omega_{\max}} g(\omega)\,d\omega$，关键在于**态密度（density of states, DOS）** $g(\omega)$——单位频率区间内的振动模式数。$g(\omega)$ 的形状由色散关系决定，两个模型的分歧就在此处。

## 2 爱因斯坦模型：单频近似

爱因斯坦（1907）做了一个大胆简化：**所有 $3N$ 个振动模式共用同一个频率 $\omega_E$**。于是

$$C_V = 3N k_B \left(\frac{\Theta_E}{T}\right)^2 \frac{e^{\Theta_E/T}}{(e^{\Theta_E/T}-1)^2}, \qquad \Theta_E = \frac{\hbar\omega_E}{k_B}$$

$\Theta_E$ 叫**爱因斯坦温度**。这个模型抓住了量子化带来的「低温冻结」：$T\ll\Theta_E$ 时 $e^{\Theta_E/T}$ 巨大，热容指数式衰减；$T\gg\Theta_E$ 时回到 $3Nk_B$（杜隆-珀替）。<span class="marginnote">爱因斯坦模型的失败暴露在低温：实验测得热容按 $T^3$ 衰减，而单频模型给的是指数衰减。原因直观——<strong>真实晶体有低能长波声子（$\omega\to0$），它们极容易被激发，而单频模型低估了低频模式</strong>。</span>

## 3 德拜模型：连续声学谱

德拜（1912）的改进：**用声学支的线性色散 $\omega = v_s k$ 代替单频**，并设定一个截止频率 $\omega_D$（德拜频率），使总模式数恰好为 $3N$。对三维各向同性介质，声子态密度为

$$g(\omega) = \begin{cases} \dfrac{9N\,\omega^2}{\omega_D^3}, & \omega \le \omega_D \\[6pt] 0, & \omega > \omega_D \end{cases}$$

$$N_{\text{modes}} = \int_0^{\omega_D} g(\omega)\,d\omega = 3N$$

于是热容积分：

$$C_V = 9N k_B \left(\frac{T}{\Theta_D}\right)^3 \int_0^{\Theta_D/T} \frac{x^4 e^x}{(e^x-1)^2}\,dx, \qquad \Theta_D = \frac{\hbar\omega_D}{k_B}$$

**德拜温度 $\Theta_D$** 是每种材料的特征参数：金刚石约 2230 K，铜约 343 K，铅约 105 K。$\Theta_D$ 越大，说明声子「又硬又快」，低温冻结来得越早。

## 4 态密度：能装多少振动模式

德拜态密度 $g(\omega)\propto \omega^2$ 是三维各向同性色散的天然结果。从倒空间计数推导：体积 $V$ 内波矢空间态密度为 $V/(2\pi)^3$，在等频面 $S$ 上的模式数为

$$g(\omega) = \frac{V}{(2\pi)^3}\int_S \frac{dS}{|\nabla_{\mathbf{k}}\omega|}$$

$|\nabla_{\mathbf{k}}\omega| = v_g$ 是群速度。对线性色散 $\omega=vk$，等频面是球面，$dS/v = 4\pi k^2/v$，代入即得 $g \propto \omega^2$。<span class="marginnote">态密度公式里的 $1/|\nabla\omega|$ 在色散极值处发散——这就是 <strong>van Hove 奇点</strong>：光学支顶部、声学支边界处态密度出现尖峰。实验测量热容、中子散射谱里的特征峰，很多都能对应到 van Hove 奇点。</span>

## 5 公式解析：德拜 T³ 定律

低温极限 $T \ll \Theta_D$ 时积分上限 $\Theta_D/T \to \infty$，热容简化为：

$$C_V = 9N k_B \left(\frac{T}{\Theta_D}\right)^3 \int_0^{\infty} \frac{x^4 e^x}{(e^x-1)^2}\,dx = \frac{12\pi^4}{5}N k_B \left(\frac{T}{\Theta_D}\right)^3$$

拆三步：

- **第一步，留积分**：$\int_0^\infty x^4 e^x/(e^x-1)^2\,dx = 4\pi^4/15$ 是标准积分（可展开为黎曼 $\zeta$ 函数 $\zeta(4)$）。它只来自低能玻色统计，与材料无关。
- **第二步，看幂次**：$C_V \propto T^3$。物理根源是 $g(\omega)\propto\omega^2$（低频模式数目随频率平方增长）乘以每个模式的玻色能量贡献。
- **第三步，定系数**：$12\pi^4/5 \approx 233$。这条 **德拜 $T^3$ 定律**是固体物理最著名的定量预言之一，与金刚石、硅、惰性元素晶体的低温比热测量吻合极好。

## 6 两模型对比与实验

| 维度 | 爱因斯坦模型 | 德拜模型 |
| --- | --- | --- |
| 频率假设 | 单一 $\omega_E$ | 线性色散 $\omega=vk$，截止 $\omega_D$ |
| 态密度 | 单个 $\delta$ 峰 | $g \propto \omega^2$（截止） |
| 低温热容 | 指数衰减 $e^{-\Theta_E/T}$ | **$T^3$ 定律** |
| 高温热容 | $3Nk_B$ | $3Nk_B$ |
| 适用场景 | 光学支（高频、近平坦） | 声学支（低频、线性） |

真实晶体同时有声学支与光学支：德拜模型管声学支，爱因斯坦模型管光学支，两者叠加即可。实验上用低温比热拟合 $\Theta_D$，用中子散射直接测色散，两条路线互相印证。<span class="marginnote">一个材料的完整热容曲线是一条信息量很大的「指纹」：低温 $T^3$ 段给出声速，爱因斯坦特征峰给出光学支频率。<strong>测比热 ≈ 间接做声子谱学</strong>。</span>

### 数值例子：德拜温度与声速

德拜温度 $\Theta_D = \hbar\omega_D/k_B$ 可以从声速直接估算。对各向同性介质，德拜频率由「模式总数 = $3N$」决定：

$$\omega_D = v_s\left(6\pi^2\frac{N}{V}\right)^{1/3}$$

对铜（$v_s \approx 4700$ m/s，原子密度 $8.5\times10^{28}$ m⁻³）：

$$\omega_D \approx 4700\times(6\pi^2\times8.5\times10^{28})^{1/3} \approx 4.5\times10^{13}\ \text{rad/s}, \qquad \Theta_D = \frac{\hbar\omega_D}{k_B} \approx 340\ \text{K}$$

实测铜 $\Theta_D \approx 343$ K——**只靠声速和原子密度，就能预报德拜温度**。金刚石声速快（约 $1.2\times10^4$ m/s），$\Theta_D$ 高达约 2230 K，这就是它室温下热容远低于杜隆-珀替值的直接原因。

### 德拜积分的数值对照

德拜热容公式中的积分 $D(\Theta_D/T)$ 有标准数值表：

| $\Theta_D/T$ | $C_V/3Nk_B$ | 近似区间 |
| --- | --- | --- |
| 0（高温） | 1.000 | 杜隆-珀替 |
| 1 | 0.952 | 过渡 |
| 3 | 0.583 | 中温 |
| 10 | 0.076 | 接近 $T^3$ |
| 20 | 0.0098 | $T^3$ 定律 |

表格显示：$\Theta_D/T \gtrsim 10$ 时 $T^3$ 定律误差已很小；$\Theta_D/T \lt  1$ 时回到经典值。**「高温」「低温」都是相对 $\Theta_D$ 而言的**——金刚石的室温就是它的低温（$T/\Theta_D \approx 0.13$）。

## 7 辨析｜易错点：热容的几个陷阱

- **$C_V$ 与 $C_p$ 不同**：实验通常测 $C_p$，两者相差 $C_p - C_V = 9\alpha^2 B T V$（$\alpha$ 热膨胀系数）。德拜公式给的是 $C_V$，对照实验要修正。
- **杜隆-珀替是高温极限，不是普适定律**：室温下金刚石早已偏离 $3R$，因为 $\Theta_D \approx 2230$ K 远高于室温。
- **$T^3$ 定律只在 $T \ll \Theta_D$ 成立**：温度稍高就要算完整的德拜积分，别滥用低温近似。
- **电子对热容也有贡献**：金属里自由电子贡献 $\gamma T$（第 6 章），低温时 $\gamma T$ 会盖过 $T^3$——测比热时的第一道「去背景」工序。

## 8 小结

- 热容 = 声子模式平均能量对温度的导数；**色散 → 态密度 $g(\omega)$ → $C_V$** 是标准流水线。
- **爱因斯坦模型**：单频近似，热容指数衰减，适合光学支。
- **德拜模型**：线性色散 + 截止频率 $\omega_D$，态密度 $g\propto\omega^2$；低温给出**德拜 $T^3$ 定律** $C_V = \tfrac{12\pi^4}{5}Nk_B(T/\Theta_D)^3$，高温回到杜隆-珀替 $3Nk_B$。
- **德拜温度 $\Theta_D$