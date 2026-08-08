---
title: 热容的德拜与爱因斯坦模型
date: 2026-08-07
---

# 热容的德拜与爱因斯坦模型

<div class="epigraph">
<p>固体热容的低温行为，是量子力学在宏观世界的第一个指纹。</p>
<footer>—— 彼得 · 德拜（Peter Debye）</footer>
</div>

<div class="article-byline">
<p>第四级 · 凝聚态物理 ｜ 黄昆《固体物理学》第3章、Ashcroft &amp; Mermin Ch.23–24 ｜ 2026-08-07</p>
</div>

## 为什么从热容开始

把一块金属加热，温度升高一度需要吸收多少热量——这个量就是**热容（heat capacity）**。
它看似简单，却藏着量子力学的第一场胜利。
19 世纪末，经典物理预言所有固体的热容都是常数（杜隆-珀替定律），实验却在低温下发现热容**随温度下降**，室温下也偏离常数值。<span class="marginnote">量子论的第一个成功应用，不是原子光谱，而是固体热容：1907 年爱因斯坦用单频谐振子解释热容骤降，1912 年德拜用连续声子谱给出低温 $T^3$ 定律。
<strong>黑体辐射、光电效应、固体热容三件事联手埋葬了经典物理</strong>。</span>

热容是声子气体能量对温度的响应，是声子态密度的「积分探测器」——测热容，等于间接测态密度。

## 1 经典极限：杜隆-珀替定律

能量均分定理说：每个自由度平均携带 $k_B T$ 的能量（动能加势能各一半）。
$N$ 个原子的晶体有 $3N$ 个振动自由度，总能量：

$$U = 3N k_B T$$

热容 $C_V = \partial U/\partial T = 3Nk_B$，即**每摩尔 $3R \approx 25$ J/(mol·K)**（$R$ 是气体常数）。
这被称为**杜隆-珀替定律（Dulong–Petit law）**，室温下对大多数固体近似成立。

**但实验给出的不是常数**：低温下 $C_V$ 随 $T$ 迅速下降，$T \to 0$ 时 $C_V \to 0$。经典能量均分给出与温度无关的常数——**它彻底失败了**。原因是能量均分是经典极限 $k_BT \gg \hbar\omega$，低温下声子能量 $k_BT \ll \hbar\omega$，能量不能连续取，必须量子化。

## 2 爱因斯坦模型

**爱因斯坦模型（Einstein model）**：假设所有 $3N$ 个振动模式都有**同一个频率** $\omega_E$，即声子谱只有一根「刺」：$g(\omega) = 3N\delta(\omega - \omega_E)$。

每个模式是量子谐振子，占据数 $\langle n\rangle = 1/(e^{\hbar\omega_E/k_BT} - 1)$，总能量：

$$U = 3N \frac{\hbar\omega_E}{e^{\hbar\omega_E/k_BT} - 1}$$

由此得到爱因斯坦热容：

$$C_V = 3Nk_B \left(\frac{\theta_E}{T}\right)^2 \frac{e^{\theta_E/T}}{(e^{\theta_E/T} - 1)^2}, \qquad \theta_E = \frac{\hbar\omega_E}{k_B}$$

- 高温 $T \gg \theta_E$：$C_V \to 3Nk_B$，回归杜隆-珀替；
- 低温 $T \ll \theta_E$：$C_V \propto e^{-\theta_E/T}$，指数衰减。

**辨析｜易错点：**爱因斯坦模型低温给出**指数**衰减，而实验是 $T^3$。原因是它只用一个频率，忽略了长波声学声子——真实固体在低温下主要激发的是低频长波声子，而爱因斯坦模型把它们统统忽略。**模型定性正确（热容下降），定量错误（指数 vs 幂律）**。<span class="marginnote">爱因斯坦模型的价值：它第一个说明<strong>量子化能解释热容骤降</strong>。今天它仍是描述光学支（频率集中）与缺陷/两能级系统的标准工具——用它描述光学支、德拜模型描述声学支，是实际固体的标配「拼盘」。</span>

## 3 德拜模型

**德拜模型（Debye model）** 改用真实声子谱的近似：把声学支当成**各向同性的线性色散** $\omega = v_s q$，声子谱连续分布，并截断在**德拜频率** $\omega_D$ 处，使总模式数保持 $3N$：

$$\int_0^{\omega_D} g(\omega)\, d\omega = 3N, \qquad g(\omega) = \frac{3V}{2\pi^2 v_s^3}\omega^2 \;\;(\text{三维})$$

截断条件定出**德拜温度** $\theta_D = \hbar\omega_D/k_B$。
它唯一地由声速（或弹性常数）与原子密度决定，是每个材料的特征温度。
总能量：

$$U = \int_0^{\omega_D} \frac{\hbar\omega}{e^{\hbar\omega/k_BT} - 1}\, g(\omega)\, d\omega$$

德拜热容：

$$C_V = 9Nk_B \left(\frac{T}{\theta_D}\right)^3 \int_0^{\theta_D/T} \frac{x^4 e^x}{(e^x - 1)^2}\, dx$$

### 低温极限：德拜 $T^3$ 定律

当 $T \ll \theta_D$，积分上限 $\theta_D/T \to \infty$，定积分收敛为常数 $\frac{4\pi^4}{15}$：

$$C_V = \frac{12\pi^4}{5} Nk_B \left(\frac{T}{\theta_D}\right)^3 \propto T^3$$

**德拜 $T^3$ 定律**是凝聚态物理最著名的定量预言之一：低温热容只激发长波声子，而长波声子态密度 $g \propto \omega^2$，能被激发的模式数 $\propto T^3$。<span class="marginnote">直觉：低温下只有 $\hbar\omega \lesssim k_BT$ 的低频声子能被激发，它们占据 $g(\omega) \propto \omega^2 \propto T^2$ 的态密度段，每个携带能量 $\sim k_BT$，总能量 $\sim T^2 \cdot T = T^3$——<strong>三份 $T$ 各来一处：态密度 $T^2$ + 单声子能量 $T$</strong>。</span>

## 4 公式解析：德拜低温热容 $T^3$ 定律

$$C_V = \frac{12\pi^4}{5}Nk_B \left(\frac{T}{\theta_D}\right)^3$$

- **第一步，写积分**：$C_V = \partial U/\partial T$，其中 $U = \int_0^{\omega_D} \frac{\hbar\omega}{e^{\hbar\omega/k_BT}-1} \frac{3V}{2\pi^2 v_s^3}\omega^2 d\omega$。
- **第二步，无量纲化**：令 $x = \hbar\omega/k_BT$，则 $d\omega = (k_BT/\hbar) dx$，$\omega^2 \propto x^2 T^2$，态密度带来 $\omega^2 d\omega \propto T^3 x^2 dx$——**$T^3$ 的源头已经现身**。
- **第三步，低温截断**：$T \ll \theta_D$ 时上限 $\omega_D \to \infty$，积分 $\int_0^\infty \frac{x^4 e^x}{(e^x-1)^2} dx = \frac{4\pi^4}{15}$ 是已知常数。
- **第四步，求导得热容**：$U \propto T^4 \cdot \frac{1}{T} $? 注意 $\hbar\omega$ 项提供因子 $k_BT$，合并得到 $U \propto T^4$，对 $T$ 求导给出 $C_V \propto T^3$，系数正好是 $\frac{12\pi^4}{5}Nk_B/\theta_D^3$。

**低温热容是声子谱的「低频探针」**——测出 $T^3$ 的系数，就得到德拜温度，进而得到声速与弹性常数。

## 5 模型对比与实验检验

| 模型 | 声子谱假设 | 低温行为 | 描述对象 |
| --- | --- | --- | --- |
| 杜隆-珀替 | 能量连续（经典） | 常数 $3Nk_B$ | 高温极限 |
| 爱因斯坦 | 单频率 $\omega_E$ | $e^{-\theta_E/T}$ | 光学支、缺陷 |
| 德拜 | $\omega = v_sq$ 截断于 $\omega_D$ | $T^3$ | 声学支（大多数固体的热容主体） |

**辨析｜易错点：**① 德拜模型用**一个** $\theta_D$ 描述整个声学支，真实晶体有纵波与横波两支、速度不同且各向异性，德拜温度只是「平均」参数——同一材料从热容、电导、弹性测出的 $\theta_D$ 会有差异。② 金属的低温热容还有**电子贡献** $C_e = \gamma T$（正比于 $T$，不是 $T^3$），测量时要把 $C_V = \gamma T + \beta T^3$ 一起拟合——$C/T$ 对 $T^2$ 作图得直线，截距给 $\gamma$、斜率给 $\beta$。**只按 $T^3$ 拟合金属热容，会漏掉电子项**。<span class="marginnote">电子比热 $C_e = \gamma T$ 是费米气体的「量子尾巴」：只有费米面附近 $\sim k_BT$ 宽度的电子能吸收热量。测 $\gamma$ 可以直接得到费米能级处的态密度——这是把热容当探针的一个经典应用，见「自由电子气与索末菲模型」一节。</span>

### 算例：从德拜温度验声速

铜的德拜温度 $\Theta_D \approx 343$ K。
由 $\hbar\omega_D = k_B\Theta_D$ 与 $\omega_D = v_s(6\pi^2 n)^{1/3}$（$n$ 是原子数密度 $8.5\times10^{28}$ m⁻³）反推声速：

$$v_s = \frac{k_B\Theta_D}{\hbar(6\pi^2 n)^{1/3}} \approx \frac{1.38\times10^{-23}\times343}{1.05\times10^{-34} \times (5.04\times10^{30})^{1/3}} \approx \frac{4.73\times10^{-21}}{1.05\times10^{-34}\times1.71\times10^{10}} \approx 2.6\times10^3\ \text{m/s}$$

**与实测平均声速约 3.8 km/s 同量级**。德拜温度是「声速的另一种写法」——从热容测出 $\Theta_D$，等于测出声速；反之从弹性常数算声速，能预言热容曲线。**热容、声速、弹性常数三件套，由 $\Theta_D$ 一根线串起来**。

### 算例：电子比热何时超过声子比热

金属总比热 $C_V = \gamma T + \beta T^3$。对铜：$\gamma \approx 0.7$ mJ/(mol·K²)，$\beta$ 对应 $\Theta_D \approx 343$ K。电子项与声子项相等时：

$$\gamma T = \beta T^3 \Rightarrow T^2 = \frac{\gamma}{\beta}$$

$\beta = \frac{12\pi^4 N_A k_B}{5\Theta_D^3} \approx \frac{12\pi^4\times8.31}{5\times(343)^3} \approx 4.8\times10^{-5}$ J/(mol·K⁴) $\approx 0.048$ mJ/(mol·K⁴)。于是：

$$T^2 = \frac{0.7}{0.048} \approx 14.6 \Rightarrow T \approx 3.8\ \text{K}$$

**低于约 4 K，电子比热占主导；高于它，声子比热主导**。这就是「低温测 $\gamma$、高温测 $\beta$」的经验法则：$C/T$ 对 $T^2$ 作图，低温段直线（电子项），高温段抛物线（声子项）。**画一张图，两个物理量（费米面态密度 + 德拜温度）同时到手**。

## 6 小结

- **杜隆-珀替定律** $C_V = 3Nk_B$：经典能量均分，高温极限成立。
- **爱因斯坦模型**：单频率声子谱，低温 $C_V \propto e^{-\theta_E/T}$，定性对、定量错。
- **德拜模型**：线性声学色散 + 截断 $\omega_D$，低温给出 $C_V \propto T^3$，与实验吻合。
- **德拜温度** $\theta_D = \hbar\omega_D/k_B$ 是材料特征参数，由声速与原子密度决定。
- 金属低温热容 = 声子 $T^3$ + 电子 $\gamma T$ 两项，需同时拟合。

在下一节，我们将从「热容（热储存）」转向「热输运」——看声子如何把热量从热端搬到冷端，即**热膨胀与热导率**。
