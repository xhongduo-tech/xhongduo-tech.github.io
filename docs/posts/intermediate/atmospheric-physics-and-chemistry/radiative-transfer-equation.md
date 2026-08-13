---
title: 大气辐射传输方程
date: 2026-08-07
---


# 大气辐射传输方程

<div class="epigraph">
<p>宇宙不仅比我们想象的更奇异，而且比我们能够想象的更奇异。</p>
<footer>—— 阿瑟 · 爱丁顿（Arthur Eddington）</footer>
</div>

<div class="article-byline">
<p>第二级 · 大气物理学与大气化学 ｜ 盛裴轩《大气物理学》第5章；Salby《Physics of the Atmosphere and Climate》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从辐射传输方程开始

上一课的黑体辐射给出了「物质能发射多少辐射」的理想答案，但真实大气远不是黑体：
一束太阳光穿过大气，会被氧气、臭氧、水汽选择性吸收，会被云和气溶胶散射，
还会被沿途的大气层自身发射的红外辐射「加厚」。**辐射传输方程（radiative 
transfer equation, RTE）** 就是追踪一束辐射在吸收、发射、
散射并存的介质中如何变化的守恒方程——它是辐射平衡、
温室效应定量计算与卫星遥感的共同母体。
<span class="marginnote">
爱丁顿在本世纪之交研究恒星大气时系统化了辐射传输理论，他引入的 Eddington 
近似至今仍是简化求解的标准工具。从恒星到地球大气，辐射传输方程是同一条方程——
天文与气象在此共享方法。</span>

## 1 比尔-布格-朗伯定律：纯吸收的衰减

先不考虑发射与散射，只看**吸收**。强度为 $I_\lambda$ 
的辐射穿过厚度 $\mathrm{d}s$ 的大气，
被吸收的份额正比于路径长度与吸收物质的密度 $\rho_a$：

$$
\mathrm{d}I_\lambda = -k_\lambda \rho_a I_\lambda\,\mathrm{d}s
$$

其中 $k_\lambda$ 是**质量吸收系数**（m²/kg），
$\rho_a\,\mathrm{d}s$ 是路径上吸收物质的质量。
积分得**比尔-布格-朗伯定律（Beer–Bouguer–Lambert law）
**：

$$
I_\lambda = I_{\lambda,0}\, e^{-\tau_\lambda}, \qquad \tau_\lambda = \int_0^s k_\lambda\rho_a\,\mathrm{d}s'
$$

**光学厚度（optical depth）** $\tau_\lambda$ 
是无量纲量，度量「这段大气总共挡住了多少辐射」：$\tau = 0$ 完全透明，
$\tau = 1$ 衰减到 $1/e$，$\tau \gg 1$ 光学厚。
<span class="marginnote">
光学厚度是辐射传输里最常用的「距离单位」。一个直观参照：垂直看向晴空，
可见光波段整层大气的光学厚度约 0.1–0.3（以散射为主）；
红外吸收带里则可达几到几十。光学厚度比物理距离更有意义，因为它直接决定衰减比例。
</span>

## 2 发射项与源函数

只衰减而不补充是不完整的：沿途大气自身也在发射辐射。发射的增量正比于局地黑体辐射 
$B_\lambda(T)$ 与吸收系数——由基尔霍夫定律，发射率等于吸收率，
于是源项写作 
$+k_\lambda\rho_a B_\lambda(T)\,\mathrm{d}s$
。把衰减与发射合起来，得到**一般形式的辐射传输方程**：

$$
\frac{\mathrm{d}I_\lambda}{k_\lambda\rho_a\,\mathrm{d}s} = -I_\lambda + B_\lambda(T)
$$

定义**源函数（source function）** 
$J_\lambda = B_\lambda(T)$（纯吸收发射情形），方程写成

$$
\frac{\mathrm{d}I_\lambda}{\mathrm{d}\tau_\lambda} = I_\lambda - J_\lambda
$$

这条一阶常微分方程的物理意义非常干净：**辐射沿路径的变化，等于「已有辐射」
与「局地源」之差**。光学厚处 
$I_\lambda \to J_\lambda = B_\lambda(T)$，
辐射趋于局地黑体辐射——这正是恒星内部与深厚云层内辐射趋于各向同性的原因。
<span class="marginnote">一个经典推论：
从外太空看一团均匀的发射云，如果光学厚度很大，你看到的不是云的「里面」，
而是光学厚度约 1 的那一层（即「看见的深度」）。卫星测云顶温度、测海洋表面温度，
用的都是这个「有效发射层」的概念。</span>

## 3 散射与多次散射

实际大气里，辐射还会被空气分子、云滴、气溶胶**散射**——
光子改变方向而不损失能量。散射把问题变复杂，因为同一方向的光既会被散射走，
也会从其他方向散射进来。定义**单次散射反照率（single-scattering 
albedo）**：

$$
\tilde{\omega}_0 = \frac{\sigma_\text{sca}}{\sigma_\text{ext}} = \frac{\text{散射截面}}{\text{吸收截面}+\text{散射截面}}
$$

$\tilde{\omega}_0 = 1$ 表示纯散射（晴空分子散射近似如此），
$\tilde{\omega}_0 = 0$ 表示纯吸收。引入散射后，源函数变为

$$
J_\lambda = (1-\tilde\omega_0)B_\lambda(T) + \frac{\tilde\omega_0}{4\pi}\int_{4\pi} I_\lambda(\Omega')\,P(\Omega',\Omega)\,\mathrm{d}\Omega'
$$

第一项是发射，第二项是把所有方向散射进来的辐射按**相函数（phase 
function）** $P(\Omega',\Omega)$ 汇总——这使 
RTE 变成包含散射耦合的积分微分方程，解析解只存在于极简情形，
数值求解（离散纵标法、Monte Carlo）成为工程标准。
<span class="marginnote">散射是「蓝天为什么蓝」的直接答案：
瑞利散射强度反比于波长四次方，短波蓝光被散射得远比红光多，于是天空呈现蓝色；
而云朵因米散射对波长不敏感而呈白色。到《太阳辐射的吸收与散射》一课我们再展开。
</span>

## 4 公式解析：简化的解析解

对许多大气问题，散射可暂时忽略（$\tilde\omega_0 = 0$），RTE 
退化为一条可严格积分的一阶线性方程。以平面平行大气垂直坐标 $z$（向上为正）为例，
设天顶角余弦 $\mu = \cos\theta$，则

$$
\mu\,\frac{\mathrm{d}I_\lambda(z,\mu)}{\mathrm{d}z} = k_\lambda\rho_a\left[I_\lambda(z,\mu) - B_\lambda(T(z))\right]
$$

分三步求解：

- **第一步，写成积分因子形式**：令 $I' = I_\lambda e^{-\tau_\lambda/\mu}$，方程化为 $\dfrac{\mathrm{d}I'}{\mathrm{d}z} = -\dfrac{k_\lambda\rho_a}{\mu} B_\lambda\, e^{-\tau_\lambda/\mu}$。
- **第二步，沿路径积分**：从大气顶（$\tau=0$）到高度 $z$（$\tau = \tau_\lambda$），利用边界条件 $I_\lambda(\tau=0)=0$（无向下入射），得

$$
I_\lambda(\tau_\lambda) = \int_0^{\tau_\lambda} B_\lambda(T(\tau'))\, e^{-(\tau_\lambda - \tau')/\mu}\,\frac{\mathrm{d}\tau'}{\mu}
$$

- **第三步，读出物理结构**：**到达高度 $\tau_\lambda$ 的辐射 = 每一层发射的 $B_\lambda$ 乘以上方大气的透过率，再逐层叠加**。这就是「发射-衰减级联」：每一层既贡献自己的发射，又被其上大气削弱。

这个解说明了一个深刻的道理：**你看到的红外辐射，主要来自光学厚度约 1 
的那一层**。从卫星往下看，探测到的长波辐射大致来自「看见深度」对应的那一层大气，
其温度就是所谓**亮温（brightness temperature）**。
卫星温度探测器正是靠测量不同吸收强度谱线的亮温，反演不同高度的温度廓线——
遥感物理的整个基础都在这一条式子里。
<span class="marginnote">这也是温室效应的机制核心：
地表长波辐射被大气吸收后，发射面被抬高到更冷的高空。冷层发射的辐射能量少，
逃逸到太空的净长波减少，地表被迫升温来恢复平衡——辐射传输方程把「温室」
翻译成了数学。</span>

## 5 小结

- **比尔-布格-朗伯定律** $I = I_0 e^{-\tau}$ 描述纯吸收的指数衰减，光学厚度 $\tau$ 是核心无量纲量。
- 计入局地发射后，**辐射传输方程** $\dfrac{\mathrm{d}I}{\mathrm{d}\tau} = I - J$ 成为守恒方程，其解是「发射-衰减级联」。
- **单次散射反照率** $\tilde\omega_0$