---
title: 平面电磁波的传播及其在介质界面的反射与折射
date: 2026-08-07
---

# 平面电磁波的传播及其在介质界面的反射与折射

<div class="epigraph">
<p>一束平面电磁波在介质中奔行，在界面上分道扬镳——反射多少、透射多少、相位变不变，全由介质与入射角决定。</p>
<footer>—— 电动力学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 郭硕鸿《电动力学》第三章 ｜ 2026-08-07</p>
</div>

## 为什么从平面波传播开始

麦克斯韦方程组在无源区域化出波动方程，其最基本的解就是**平面电磁波（plane electromagnetic wave）**。理解平面波在均匀介质中的传播（波速、色散、衰减）与在界面的反射折射（菲涅耳公式），是光学、通信、雷达、光纤的全部物理基础。这一节研究平面波的传播性质，并用边值关系（第 120 节）推导反射与折射的规律。

## 1 均匀介质中的平面电磁波

在无源、均匀、各向同性介质中，电磁场满足波动方程，平面波解：

$$\boldsymbol{E} = \boldsymbol{E}_0e^{i(\boldsymbol{k}\cdot\boldsymbol{r} - \omega t)}, \qquad \boldsymbol{B} = \boldsymbol{B}_0e^{i(\boldsymbol{k}\cdot\boldsymbol{r} - \omega t)}$$

- **波矢** $\boldsymbol{k}$：指向传播方向，$k = |\boldsymbol{k}| = \omega/v$；
- **波速**：$v = \frac{1}{\sqrt{\mu\varepsilon}} = \frac{c}{n}$，折射率 $n = \sqrt{\mu_r\varepsilon_r} \approx \sqrt{\varepsilon_r}$（非磁性介质）；
- **横波条件**：$\boldsymbol{k}\cdot\boldsymbol{E} = 0$、$\boldsymbol{k}\cdot\boldsymbol{B} = 0$（电场、磁场垂直传播方向）；
- **关系**：$\boldsymbol{B} = \frac{1}{v}\hat{\boldsymbol{k}}\times\boldsymbol{E}$（$\boldsymbol{E}$、$\boldsymbol{B}$ 垂直且 $\boldsymbol{E}\times\boldsymbol{B}$ 沿传播方向）。

**重点：平面电磁波是横波，$\boldsymbol{E} \perp \boldsymbol{B} \perp \boldsymbol{k}$，波速 $v = c/n$。** 电场、磁场与波矢两两垂直（右手系）；波速由介质折射率决定（$n = \sqrt{\varepsilon_r}$，非磁性）。这些性质第 70 节已有定性结论，这里给出严格矢量形式。

**辨析｜易错点：**$\boldsymbol{B}$ 与 $\boldsymbol{E}$ 的关系 $\boldsymbol{B} = \frac{1}{v}\hat{\boldsymbol{k}}\times\boldsymbol{E}$——$\boldsymbol{B}$ 垂直 $\boldsymbol{E}$、垂直传播方向，且三者满足右手定则（$\boldsymbol{E}\times\boldsymbol{B}$ 沿 $\boldsymbol{k}$）。相位因子 $e^{i(\boldsymbol{k}\cdot\boldsymbol{r}-\omega t)}$ 中「$\boldsymbol{k}\cdot\boldsymbol{r}$」随传播方向取正——注意 $\boldsymbol{k}$ 的方向约定。

## 2 色散与衰减

**色散（dispersion）**：折射率（波速）随频率变化。介质的 $\varepsilon(\omega)$、$\mu(\omega)$ 与频率有关，导致：

- 不同频率的波在介质中速度不同——光脉冲展宽（光纤色散）、白光色散成彩虹；
- **相速度** $v_p = \omega/k$ 与**群速度** $v_g = \mathrm{d}\omega/\mathrm{d}k$ 不同——信号以群速度传播。

**导电介质中的衰减**：导体中 $\boldsymbol{j} = \sigma\boldsymbol{E}$（欧姆定律），波矢变复数，波幅按 $e^{-\alpha x}$ 衰减（**趋肤效应**）：

$$\alpha \approx \sqrt{\frac{\omega\mu\sigma}{2}}$$

**重点：电磁波在介质中有色散（速度依赖频率）、在导体中有衰减（趋肤效应）。** 光纤通信的带宽限制来自色散（脉冲展宽）；导体中波迅速衰减（趋肤深度 $\delta = 1/\alpha$）——高频电流只在导体表面流动。<span class="marginnote">「趋肤效应」：导体中高频电磁波只穿透很浅（趋肤深度 $\delta \propto 1/\sqrt{\omega\sigma\mu}$）——高频电流集中在导体表面。工程影响：高频导线用多股细线（增大表面积）、射频屏蔽用薄铜皮、电磁炉靠趋肤效应在锅表面加热。传导电流密度 $\boldsymbol{j} = \sigma\boldsymbol{E}$ 与位移电流的竞争，决定了波在导体中是传播还是衰减。</span>

## 3 界面上的反射与折射

平面波入射到两种介质界面，由边值关系（$\boldsymbol{E}$、$\boldsymbol{H}$ 切向连续）得到：

**运动学（方向）**：

- 反射定律：$\theta_i = \theta_r$；
- 折射定律（斯涅耳）：$n_1\sin\theta_i = n_2\sin\theta_t$。

**动力学（振幅）——菲涅耳公式**：

对电场垂直于入射面的 s 偏振：

$$r_s = \frac{E_{r0}}{E_{i0}} = \frac{n_1\cos\theta_i - n_2\cos\theta_t}{n_1\cos\theta_i + n_2\cos\theta_t}$$

对电场平行于入射面的 p 偏振：

$$r_p = \frac{n_2\cos\theta_i - n_1\cos\theta_t}{n_2\cos\theta_i + n_1\cos\theta_t}$$

**重点：反射/折射定律（方向）与菲涅耳公式（振幅）都由边界条件推出——反射系数依赖偏振与入射角。** s 与 p 偏振的反射系数不同，导致布儒斯特角（$r_p = 0$，反射光完全偏振）、全反射（$n_1 > n_2$ 且超临界角）。<span class="marginnote">「菲涅耳公式的推论」：① 垂直入射（$\theta_i = 0$）：$r = \frac{n_1 - n_2}{n_1 + n_2}$——由折射率差定反射；② 布儒斯特角：$\tan\theta_B = n_2/n_1$ 时 $r_p = 0$（第 87 节）；③ 全反射：$n_1 > n_2$、$\theta_i > \theta_c$ 时 $|r| = 1$（第 73 节）。几何光学的全部定律 + 波动光学的偏振规律，都是菲涅耳公式的特例。</span>

## 4 公式解析：垂直入射的反射系数

光从空气（$n_1 = 1$）垂直入射到玻璃（$n_2 = 1.5$）表面。求反射系数与反射率。

$$
r = \frac{n_1 - n_2}{n_1 + n_2} = \frac{1 - 1.5}{1 + 1.5} = -0.2, \qquad R = r^2 = 0.04
$$

- **第一步，写垂直入射反射系数**：$r = (n_1 - n_2)/(n_1 + n_2)$（垂直入射时 s、p 相同）。
- **第二步，代入**：$r = -0.2$——负号表示反射光相位反转（半波损失）。
- **第三步，算反射率**：$R = |r|^2 = 0.04 = 4\%$——只有 4% 的能量被反射，96% 透射。
- **第四步，解读**：这就是为什么镜头/窗户看起来「透明」（反射仅 4%）；也解释了增透膜（镀膜降低反射）与棱镜反射镜（镀银提高反射）的需求。$n$ 差越大反射越强（金刚石 $n = 2.42$，$R \approx 17\%$）。

**辨析｜易错点：**反射率 $R = |r|^2$（能量），反射系数 $r$ 是振幅比——两者关系 $R = r^2$（垂直入射、无吸收）。$r$ 可以为负（相位反转），$R$ 恒为正。菲涅耳公式的 s/p 偏振定义随教材可能不同（有的取入射面为参考），用「电场垂直/平行入射面」确定即可。

## 5 平面波传播的应用

- **光纤**：全反射导光（第 73 节）+ 色散管理（第 85 节）；
- **抗反射/增反膜**：用菲涅耳系数设计多层膜（干涉滤光片）；
- **雷达散射**：目标对电磁波的反射（RCS）；
- **隐身技术**：吸波材料（衰减）+ 阻抗匹配（减少反射）；
- **地球物理与遥感**：电磁波在介质界面反射用于探测（探地雷达）。

<span class="marginnote">「反射折射的工程意义」：从镜头镀膜（控制反射率）到光纤通信（全反射导光）、从雷达（目标反射）到遥感（地表反射率判读），平面波在界面的行为是这些技术的共同物理。菲涅耳公式 + 边值关系，把「一束光打在水面上」的日常现象量化成了可设计的工程参数。</span>

## 6 小结

- **平面波**：$\boldsymbol{E} = \boldsymbol{E}_0e^{i(\boldsymbol{k}\cdot\boldsymbol{r}-\omega t)}$；横波（$\boldsymbol{E}\perp\boldsymbol{B}\perp\boldsymbol{k}$）；波速 $v = c/n$；$\boldsymbol{B} = \frac{1}{v}\hat{\boldsymbol{k}}\times\boldsymbol{E}$。
- **色散**：$n(\omega)$，相速 vs 群速；光纤色散限制带宽。
- **导体衰减（趋肤效应）**：波幅 $e^{-\alpha x}$，趋肤深度 $\delta = 1/\alpha \propto 1/\sqrt{\omega\mu\sigma}$。
- **界面**：反射/折射定律 + 菲涅耳公式（s、p 偏振反射系数）——布儒斯特角、全反射都是其推论。
- 垂直入射：$r = (n_1-n_2)/(n_1+n_2)$，$R = r^2$（玻璃约 4%）。

在下一节，我们研究辐射问题的解——**推迟势**。
