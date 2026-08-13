---
title: 时谐场与复矢量方法
date: 2026-08-07
---

# 时谐场与复矢量方法

<div class="epigraph">
<p>一切科学的伟大发现，几乎都是把习以为常的事物当作问题重新审视的结果。</p>
<footer>—— 斯坦因梅茨（Charles Proteus Steinmetz，复数的工程师语言之父）</footer>
</div>

<div class="article-byline">
<p>第四级 · 无线电物理 ｜ C. A. Balanis, *Advanced Engineering Electromagnetics\*, 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从时谐场开始

麦克斯韦方程组对时间求导、对空间求导纠缠在一起，直接求解很难。但无线电系统几乎都工作在**单一频率**上：载波是 $f_0$，本地振荡器是 $f_{\mathrm{LO}}$，天线的设计频率是 $f_0$。既然激励是单频正弦，响应也是同一频率的正弦——**场对时间的依赖可以「算出来」，不必当作独立变量**。这就是时谐（time-harmonic）假设的价值：它把一组偏微分方程变成一组代数方程，把「每时每刻的场」压缩成「一个复振幅」。<span class="marginnote">一个系统只有工作在线性区，时谐方法才严格成立。非线性器件（混频器、功率放大器）会把单频变成多频——那正是第4篇射频前端章节要处理的「非线性」例外。</span>

这套「把正弦时间依赖抽出来」的数学，就是电气与射频工程里无处不在的**相量方法（phasor method）**，在电磁场语境下叫**复矢量（complex vector）**。学完本章，你将拥有阅读几乎所有射频与微波文献的语言。

## 1 从瞬时场到复矢量

一个沿 $z$ 轴极化、沿 $z$ 方向传播的正弦电场，瞬时形式为：

$$
\mathcal{E}_x(z, t) = E_0 \cos(\omega t - \beta z + \phi_0)
$$

其中 $\omega = 2\pi f$ 是角频率，$\beta$ 是相位常数，$\phi_0$ 是初相。欧拉公式 $e^{j\theta} = \cos\theta + j\sin\theta$ 允许我们把它写成：

$$
\mathcal{E}_x(z, t) = \mathrm{Re}\big[ E_x(z)\, e^{j\omega t} \big], \qquad E_x(z) = E_0 e^{-j\beta z + j\phi_0}
$$

抽掉 $e^{j\omega t}$ 后剩下的 $E_x(z)$ 就是**相量（phasor）**，它是一个只依赖空间的复数量。<span class="marginnote">记号约定：瞬时量常用花体（$\mathcal{E}$）或小写，相量用大写。读文献时先分清这套记号，后面所有推导都不再含糊。</span>把三个分量都这样处理，就得到复矢量 $\mathbf{E} = \hat{\mathbf{x}} E_x + \hat{\mathbf{y}} E_y + \hat{\mathbf{z}} E_z$，其中每个 $E_i$ 都是复数。

**复矢量的幅度与相位**：$|E_x|$ 是峰值幅度，$\arg E_x$ 是初相。相比瞬时形式，复矢量把「振幅、相位、空间分布」三项信息一次性打包，而把「时间」显式地交给 $e^{j\omega t}$。

## 2 对时间求导变成乘法

时谐方法最大的实惠在于微积分退化。设 $\mathcal{F}$ 是时谐量，则：

$$
\frac{\partial \mathcal{F}}{\partial t} = \mathrm{Re}\big[ j\omega\, F\, e^{j\omega t} \big], \qquad
\int \mathcal{F}\, dt = \mathrm{Re}\big[ \tfrac{1}{j\omega}\, F\, e^{j\omega t} \big]
$$

对时间求导变成乘以 $j\omega$，积分变成除以 $j\omega$。这相当于傅里叶变换在单频点上的退化形式：时域卷积定理、微分定理都坍缩成代数操作。<span class="marginnote">把「求导 = 乘 $j\omega$」记牢，是理解电感电容阻抗公式的关键：$Z_L = j\omega L$、$Z_C = 1/(j\omega C)$ 全都来自这个替换。</span>

于是麦克斯韦方程组在无源均匀媒质中变成：

$$
\nabla \times \mathbf{E} = -j\omega\mu\mathbf{H}, \qquad \nabla \times \mathbf{H} = (\sigma + j\omega\varepsilon)\mathbf{E}, \qquad \nabla\cdot\mathbf{E} = 0, \qquad \nabla\cdot\mathbf{H} = 0
$$

注意第二式把传导电流与位移电流合并成一项 $(\sigma + j\omega\varepsilon)\mathbf{E}$——**复介电常数**的影子已经出现。

## 3 复介电常数与复磁导率

把安培–麦克斯韦定律改写，定义**复介电常数（complex permittivity）**：

$$
\nabla \times \mathbf{H} = j\omega\varepsilon_c \mathbf{E}, \qquad \varepsilon_c = \varepsilon' - j\varepsilon'' = \varepsilon - j\frac{\sigma}{\omega}
$$

其中 $\varepsilon'' = \sigma/\omega$ 项把介质损耗吸收进介电常数的虚部。定义**损耗正切（loss tangent）**：

$$
\tan\delta = \frac{\varepsilon''}{\varepsilon'}
$$

**辨析｜易错点：** 教材里「低损耗介质」「良导体」的分界，正是看 $\tan\delta$ 与 1 的关系。$\tan\delta \ll 1$ 时介质近似无损；$\tan\delta \gg 1$ 时位移电流可忽略，媒质行为像导体。很多同学只记「金属 $\sigma$ 很大」，却忘了判断标准是 $\sigma/(\omega\varepsilon)$ 这个**无量纲比**，而不是 $\sigma$ 本身——水在直流下是绝缘体、在微波下却是强吸收介质，正是这个比在起作用。<span class="marginnote">金属中 $\sigma/(\omega\varepsilon) \gg 1$，故电磁波只能进入趋肤深度 $\delta_s = \sqrt{2/(\omega\mu\sigma)}$；在微波频段这个深度是微米量级，所以波导内壁镀银、天线用薄金属层就够——这是工程上「薄」与「厚」的物理标尺。</span>

类似地定义复磁导率 $\mu_c = \mu' - j\mu''$，$\tan\delta_m = \mu''/\mu'$。损耗的来源于是被统一表述：介电损耗与磁损耗各用一个损耗正切刻画。

## 4 复传播常数与衰减

在无源均匀媒质中，电磁波沿 $z$ 传播的相量解为 $e^{-\gamma z}$，其中**传播常数（propagation constant）**为：

$$
\gamma = \alpha + j\beta = j\omega\sqrt{\mu_c \varepsilon_c}
$$

实部 $\alpha$ 是**衰减常数**（Np/m），虚部 $\beta$ 是**相位常数**（rad/m）。<span class="marginnote">无损时 $\alpha = 0$，$\beta = \omega\sqrt{\mu\varepsilon}$；有损时波幅随距离按 $e^{-\alpha z}$ 指数衰减，$\alpha$ 的倒数 $1/\alpha$ 表征波穿透媒质的特征距离。</span>一个物理量（如功率）若写 $e^{-2\alpha z}$，衰减就常以 dB 计量：每单位长度的 dB 衰减为 $8.686\alpha$ dB/m，因为 $20\log_{10}e = 8.686$。

把 $1/\sqrt{\mu\varepsilon}$ 记作波速，在低损耗近似下可推出 $\alpha \approx \frac{\omega}{2}\sqrt{\mu\varepsilon}\,\tan\delta$，$\beta \approx \omega\sqrt{\mu\varepsilon}(1 + \frac{1}{8}\tan^2\delta)$——衰减正比于损耗正切，相位常数几乎不受损耗影响。这些近似式是设计低损耗传输线、估算介质衰减的出发点。

## 5 波阻抗与复坡印廷矢量

在时谐框架下，波的电场与磁场之比定义一个与坐标无关的常数——**波阻抗（wave impedance）**：

$$
Z_w = \frac{E_x}{H_y} = \sqrt{\frac{\mu_c}{\varepsilon_c}}, \qquad \text{真空中 } \eta_0 = \sqrt{\frac{\mu_0}{\varepsilon_0}} \approx 377\ \Omega
$$

377 Ω 这个数字在无线电物理里随处可见：自由空间的波阻抗，也是偶极子天线辐射电阻的理论参照。在低损耗媒质中 $Z_w$ 为实数；在高损耗（金属）中 $Z_w$ 的实部与虚部相等，这就是金属表面的**表面阻抗（surface impedance）** $Z_s = (1+j)\sqrt{\omega\mu/(2\sigma)}$ 的由来。<span class="marginnote">表面阻抗把「金属内部发生了什么」压缩成一个复边界条件 $E_t = Z_s\, J_s$。射频工程师用它估算金属损耗：功率损耗密度 $P_{\mathrm{loss}} = \frac{1}{2} R_s |\mathbf{J}_s|^2$，$R_s$ 越小、导电性越好。</span>

时谐场的**复坡印廷矢量（complex Poynting vector）**定义为：

$$
\mathbf{S} = \frac{1}{2}\,\mathbf{E} \times \mathbf{H}^*
$$

它同时携带**实功率**（时间平均能流）与**无功功率**（储能往返振荡）两个信息：$\frac{1}{2}\mathrm{Re}[\mathbf{E}\times\mathbf{H}^*]$ 是时间平均功率流密度，$\frac{1}{2}\mathrm{Im}[\mathbf{E}\times\mathbf{H}^*]$ 与电、磁储能之差相关。天线分析里，辐射功率从复坡印廷矢量的实部读出，无功功率则体现在天线输入阻抗的电抗部分——这是第3篇天线参数的基础。

## 6 公式解析：一个量同时含幅度与相位

取平面波的电场相量 $E_x(z) = E_0 e^{-j\beta z + j\phi_0}$ 与磁场相量 $H_y(z) = \dfrac{E_0}{\eta} e^{-j\beta z + j\phi_0}$，其中 $\eta$ 是波阻抗。分三步看懂它携带的全部信息：

- **第一步，分离幅度与相位**：$E_0$ 是峰值幅度，$e^{j\phi_0}$ 是 $z=0$ 处的初相，$e^{-j\beta z}$ 是随位置落后的相位——三者各司其职，没有混在一个数字里。
- **第二步，回到瞬时值**：取 $\mathrm{Re}[E_x e^{j\omega t}]$，得 $E_0\cos(\omega t - \beta z + \phi_0)$。$z$ 越大相位越落后，说明波沿 $+z$ 传播；固定一个等相位面 $\omega t - \beta z = \mathrm{const}$，得 $v_p = dz/dt = \omega/\beta$，即**相速度**。
- **第三步，比较 E 与 H**：两者相位完全相同，说明 $E$ 与 $H$ 同相；幅度之比 $E_0/H_0 = \eta$，在真空中恰为 377 Ω。E 与 H 同相、垂直、且都与传播方向正交——这是均匀平面波最标准的特征，也是理解极化（下一章）与坡印廷矢量（再下一章）的参照系。

**时谐方法的核心一句话**：把时间的偏导换成 $j\omega$、把瞬时量换成复振幅，麦克斯韦方程组就从微分方程降维成代数方程，而幅度与相位全部保留在复矢量里。**辨析｜易错点：** 复矢量本身不是「真实存在的场」，它是瞬时场的傅里叶单频分量；求瞬时值必须补回 $e^{j\omega t}$ 并取实部。直接把相量的实部当瞬时场，是初学者最常见的错误。

## 7 小结

- **时谐假设**：单频正弦激励下，场对时间依赖由 $e^{j\omega t}$ 给出，可抽出来单算。
- **复矢量**把幅度、相位、空间分布打包，$\partial/\partial t \to j\omega$，微分方程变代数方程。
- **复介电常数** $\varepsilon_c = \varepsilon - j\sigma/\omega$ 吸收损耗；**损耗正切** $\tan\delta$ 划分低损耗介质与良导体。
- **传播常数** $\gamma = \alpha + j\beta$：实部衰减、虚部相位；dB 与 Np 的换算系数为 8.686。
- **波阻抗** $\eta = \sqrt{\mu_c/\varepsilon_c}$