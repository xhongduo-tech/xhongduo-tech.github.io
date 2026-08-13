---
title: 波动方程与均匀平面波解
date: 2026-08-07
---

# 波动方程与均匀平面波解

<div class="epigraph">
<p>在任何媒质中，电磁扰动都按照波动方程传播；而最简单的解，就是平面波——它是理解一切复杂波动的第一块积木。</p>
<footer>—— 朱利叶斯 · 亚当斯 · 斯特拉顿（J. A. Stratton, *Electromagnetic Theory\*, 第6章）</footer>
</div>

<div class="article-byline">
<p>第四级 · 无线电物理 ｜ J. A. Stratton, *Electromagnetic Theory\*, 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从均匀平面波开始

天线的辐射场在远处会渐渐「长平」成平面波；波导里的导波可以看成平面波在壁间来回反射的叠加；激光、雷达、射电望远镜接收的远场信号，在局部都近似平面波。**均匀平面波（uniform plane wave）**是电磁波最单纯、也是最重要的解：等相位面是平面，面上各点场强处处相同。它像力学里的质点、热学里的理想气体——是一个「不存在的理想化」，但一切真实波动都要回到它来度量。<span class="marginnote">「远处看是平面波」背后是波的局域平面性：任何光滑波前的曲面，在足够小的邻域内都可用它的切平面逼近。这就是为什么分析天线远场、地物散射时，先问「局部的波前长什么样」。</span>

上一章我们已把麦克斯韦方程组化简为时谐代数方程，并预告了波动方程。这一章正式求解它，得到平面波解的完整形态：场结构、传播常数、波阻抗、色散与相速/群速。

## 1 从麦克斯韦方程组到波动方程

在无源、线性、均匀、各向同性媒质中，对法拉第定律取旋度，代入安培–麦克斯韦定律，利用矢量恒等式 $\nabla\times\nabla\times\mathbf{E} = \nabla(\nabla\cdot\mathbf{E}) - \nabla^2\mathbf{E}$ 且 $\nabla\cdot\mathbf{E}=0$，得**矢量亥姆霍兹方程（vector Helmholtz equation）**：

$$
\nabla^2 \mathbf{E} + k^2 \mathbf{E} = 0, \qquad k = \omega\sqrt{\mu\varepsilon}
$$

$k$ 叫**波数（wavenumber）**，单位为 rad/m；真空中的波数 $k_0 = \omega\sqrt{\mu_0\varepsilon_0} = \omega/c$。对 $\mathbf{H}$ 也有同样的方程。**推导的核心在于**：求导一次消元，时间二阶导数被 $-\omega^2$ 代替，空间二阶算子就是 $\nabla^2$。<span class="marginnote">亥姆霍兹方程是时谐电磁场所有问题的母方程。波导、谐振腔、散射——第2、3、5篇的分析全是「解亥姆霍兹方程 + 施加边界条件」的不同变奏。</span>

## 2 均匀平面波解

设波沿 $z$ 传播，等相位面为 $z = \mathrm{const}$ 平面，面上场均匀，故 $\partial/\partial x = \partial/\partial y = 0$。亥姆霍兹方程退化为标量常微分方程：

$$
\frac{d^2 E_x}{dz^2} + k^2 E_x = 0
$$

通解为 $E_x(z) = E^+ e^{-jkz} + E^- e^{jkz}$，分别对应沿 $+z$ 与 $-z$ 传播的波。取 $+z$ 传播的那一支，电场与磁场为：

$$
E_x(z) = E_0 e^{-jkz}, \qquad H_y(z) = \frac{E_0}{\eta} e^{-jkz}
$$

**这就是均匀平面波：电场 $\mathbf{E}$ 与磁场 $\mathbf{H}$ 相互垂直，且都垂直于传播方向 $z$——横电磁（TEM）波。** 瞬时形式 $E_x = E_0\cos(\omega t - kz)$，等相位面以相速 $v_p = \omega/k = 1/\sqrt{\mu\varepsilon}$ 前进，真空即为 $c$。<span class="marginnote">「TEM」这个名字会反复出现：传输线主模、同轴电缆、自由空间辐射波全是 TEM；而波导里的模式不是 TEM——这是下一章导波理论的关键区分，先在这里记住「平面波是 TEM」。</span>

## 3 三个方向、两种手性

把波矢量 $\mathbf{k}$ 放进坐标，可以看清平面波的结构：$\mathbf{E}$、$\mathbf{H}$、$\mathbf{k}$ 三者构成右手系，且满足

$$
\mathbf{H} = \frac{1}{\omega\mu}\,\mathbf{k} \times \mathbf{E}, \qquad \mathbf{E} = -\frac{1}{\omega\varepsilon}\,\mathbf{k} \times \mathbf{H}
$$

一个记住右手定则的抓手：$\mathbf{k}$ 指向传播方向，$\mathbf{E}$ 由 $\mathbf{H}$ 绕 $\mathbf{k}$ 右手旋转得到。<span class="marginnote">这套「$\mathbf{k} \times$」关系在数值电磁学里是判断场分量正负号的标准工具，也是后面推导坡印廷矢量 $\mathbf{S} = \mathbf{E}\times\mathbf{H}$ 方向（沿 $\mathbf{k}$）的基础。</span>

波的极化（第5篇第5章专题）就是研究 $\mathbf{E}$ 矢量端点的轨迹；而「$\mathbf{E}$ 与 $\mathbf{H}$ 同相、幅度比恒为 $\eta$」则是无损耗媒质平面波的标志。

## 4 有损耗媒质中的平面波

把媒质参数换成复介电常数 $\varepsilon_c = \varepsilon' - j\varepsilon''$，波数也变成复数：

$$
k_c = \omega\sqrt{\mu\varepsilon_c} = \beta - j\alpha
$$

场解变为 $E_x = E_0 e^{-\alpha z} e^{-j\beta z}$：$e^{-\alpha z}$ 是衰减，$e^{-j\beta z}$ 是相位。定义**趋肤深度（skin depth）**：

$$
\delta_s = \frac{1}{\alpha}
$$

**辨析｜易错点：** 区分三组量的次序是关键——无损时 $\alpha = 0$、$\beta = k$；良导体中 $\alpha \approx \beta \approx \sqrt{\omega\mu\sigma/2}$，两者几乎相等；而低损耗介质中 $\alpha$ 很小、$\beta \approx \omega\sqrt{\mu\varepsilon}$。不少同学把趋肤深度与波长混为一谈：波长是 $\lambda = 2\pi/\beta$（相位重复的距离），趋肤深度是幅度衰减到 $1/e$ 的距离（$1/\alpha$），两者物理意义完全不同。在铜中 1 GHz 时 $\delta_s \approx 2.1\ \mu$m，而 $\lambda$ 仍有 30 cm 量级——差五个数量级。<span class="marginnote">正是这个微米级的趋肤深度，决定了射频 PCB 的铜箔厚度（约 35 μm）为何足够、微波炉为何能加热食物内部（穿透深度厘米量级）以及金属屏蔽为何有效。把 $\delta_s = \sqrt{2/(\omega\mu\sigma)}$ 背下来，就同时背下了这三件工程事实。</span>

良导体中的波阻抗为 $Z_s = (1+j)\sqrt{\omega\mu/(2\sigma)} = (1+j)R_s$，实部 $R_s$ 即表面电阻。金属表面单位面积的损耗功率为：

$$
P_{\mathrm{loss}} = \frac{1}{2} R_s |\mathbf{J}_s|^2
$$

这是估算传输线、波导、谐振腔导体损耗的基础公式。

## 5 色散、相速与群速

若波数 $k$ 与频率 $\omega$ 的关系为非线性（即 $\varepsilon$、$\mu$ 随频率变），媒质就是**色散的（dispersive）**。两种速度的区分由此而来：

$$
v_p = \frac{\omega}{k} \quad \text{（相速，等相位面的速度）}, \qquad v_g = \frac{d\omega}{dk} \quad \text{（群速，能量与信号的速度）}
$$

真空无色散，$v_p = v_g = c$。在色散媒质中二者不同；在某些波导中群速还可能小于相速的倒数关系出现「快波/慢波」。<span class="marginnote">通信信号的信息承载在波包的包络上，包络以群速前进，所以<strong>信号的传播速度是群速而非相速</strong>。电离层短波传播、光纤色散补偿，全是 $v_g$ 与 $v_p$ 的博弈，第4篇传播章节会回来细算。</span>

**辨析｜易错点：** 相速可以超过光速（如某些波导模式），但群速严格小于 $c$，因果性由群速守护。把「相速超光速」误当作信息超光速，是教科书中经典的概念陷阱——它只说明等相位面跑得快，不携带信息。

## 6 公式解析：$E_x = E_0 e^{-jkz}$ 的每一步

把平面波解的物理内容逐层剥开：

- **第一步，$e^{-jkz}$ 里的「负号」**：写成 $\cos(\omega t - kz)$，固定相位 $\omega t - kz = C$，求 $dz/dt = \omega/k > 0$——波向 $+z$ 传播。若换成正号 $e^{+jkz}$，则波向 $-z$ 传播。符号直接决定传播方向，这是读解的起点。
- **第二步，波数 $k$ 的量纲与角色**：$k = \omega\sqrt{\mu\varepsilon}$，rad/m。相位 $kz$ 是「走过的距离换算成相位」。波长 $\lambda = 2\pi/k$：相位每走 $2\pi$ 就是一个波长。
- **第三步，磁场从哪来**：由法拉第定律 $\nabla\times\mathbf{E} = -j\omega\mu\mathbf{H}$，代入 $\mathbf{E} = \hat{\mathbf{x}}E_0 e^{-jkz}$，旋度只有 $\partial/\partial z$ 分量贡献，得 $\mathbf{H} = \hat{\mathbf{y}}(k/\omega\mu)E_0 e^{-jkz} = \hat{\mathbf{y}}(E_0/\eta)e^{-jkz}$。**磁场不是独立假设的，它是电场经麦克斯韦方程组的强制伴侣**——这也是为什么描述平面波只需给定 $\mathbf{E}$，$\mathbf{H}$ 自动跟随。
- **第四步，能量流动**：坡印廷矢量 $\mathbf{S} = \mathbf{E}\times\mathbf{H}$ 沿 $\hat{\mathbf{z}}$，平均功率密度 $\frac{1}{2}|E_0|^2/\eta$。能量沿 $+z$ 流动，与波的传播方向一致——波把能量从源送向远方，这正是天线辐射的本质。

## 7 平面波的工程现实：波束、准平面与近似

理想均匀平面波是无穷大、等幅、单一方向的——真实世界没有这样的波，但工程处处在用「准平面」近似：

**天线远场 = 局部平面波**（第3篇已见）：距离足够远时，球面波前的曲率可以忽略，任意小区域内的波近似为平面波。天线测量的远场判据 $2D^2/\lambda$，就是「球面波前曲率小到可以当平面波」的量化。

**波束与有限口径**：真实波束（喇叭、贴片、阵列）在传播中会扩散（衍射），其截面不是均匀的而是有横向分布的——严格说它们是**波束解**（如高斯波束），但局部仍可展开成平面波的叠加（角谱方法）。**「平面波展开」是连接理想与现实的桥**：把任意波场分解成平面波的积分，分析与数值都从这里起跳。

**数值仿真的吸收边界**：FDTD、FEM 等仿真器的计算域边界需要「吸收」，让波不反射回来——理想匹配层（PML）模拟的就是「平面波无反射地走出去」的边界。**仿真器里那块看不见的 PML，是平面波「无反射传播」概念最忠实的工程化身**。

**解析｜为什么先学理想化**：平面波是「教科书物理」的典型——真实世界没有，但它是一切分析的第一阶近似。**掌握「什么时候能当平面波」，比记住平面波本身更值钱**：天线远场、雷达回波、光波传播，判断的起点都是这一问。

## 8 小结

- 无源均匀媒质中，时谐场满足**矢量亥姆霍兹方程** $\nabla^2\mathbf{E} + k^2\mathbf{E} = 0$，$k = \omega\sqrt{\mu\varepsilon}$。
- **均匀平面波**是 TEM 波：$\mathbf{E}\perp\mathbf{H}\perp\mathbf{k}$，三者右手系；$\mathbf{E}$ 与 $\mathbf{H}$ 同相、幅度比 $\eta$。
- 有损媒质中波数复数化，**衰减常数** $\alpha$ 与**相位常数** $\beta$ 分离；**趋肤深度** $\delta_s = 1/\alpha$ 是衰减到 $1/e$ 的距离。
- 色散媒质中**相速** $v_p = \omega/k$ 与**群速** $v_g = d\omega/dk$ 不同，信号与能量按群速传播。
- 金属表面以**表面阻抗** $Z_s = (1+j)R_s$ 统一刻画，损耗功率正比于 $R_s|\mathbf{J}_s|^2$