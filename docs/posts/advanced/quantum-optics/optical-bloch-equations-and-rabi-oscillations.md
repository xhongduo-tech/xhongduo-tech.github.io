---
title: 光学 Bloch 方程与 Rabi 振荡
date: 2026-08-07
---

# 光学 Bloch 方程与 Rabi 振荡

<div class="epigraph">
<p>原子的量子态像一个陀螺——它在布洛赫球上进动，在驰豫中倒下。</p>
<footer>—— 阿伦·布洛赫（Felix Bloch），1946 年核磁共振奠基之作的直觉</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ R. Loudon, The Quantum Theory of Light 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从光学 Bloch 方程开始

上一节我们有了有效哈密顿量，能写薛定谔方程，但它只描述**相干**演化，
且只对纯态适用。真实原子还受**驰豫**支配：自发辐射让布居漏回基态，
碰撞与相位噪声抹掉相干性。要把两者统一，
需要一套能容纳混合态与耗散的方程——这就是**光学 Bloch 方程（OBE）**。
它是量子光学里最「工程化」的工具：核磁共振、激光冷却、量子存储、
光学时钟全都在解这套方程。理解它，
等于获得描述「原子在光场中」的通用语言。<span class="marginnote">这套方程脱胎于核磁共振（NMR）的 
Bloch 方程——量子光学直接借用了同一套数学骨架，
只是把「磁矩进动」换成了「光学偶极矩」。</span>

## 1 密度矩阵与布洛赫矢量

二能级原子的任意态由密度算符 $\rho$ 描述。用泡利矩阵展开，可写成

$$\rho = \frac{1}{2}\left(\mathbb{1} + \vec{u}\cdot\vec{\sigma}\right)$$

其中实矢量 $\vec{u} = (u, v, w)$ 
叫**布洛赫矢量**。分量对应物理量：

- $w = \rho_{ee} - \rho_{gg}$：**布居反转**（$w = 1$ 全在激发态，$w = -1$ 全在基态）；
- $u, v$：**相干**（偶极矩的实部/虚部，$u + iv = 2\rho_{ge}$）。

**布洛赫球**：单位球面上的每个点对应一个纯态，球内点对应混合态；
北极是激发态，南极是基态，赤道是等权叠加态。原子的动力学 = 
布洛赫矢量在球上的运动。<span class="marginnote">布洛赫球与第五级《量子计算》里的单比特量子门图像完全同构：$X$ 
门绕 $x$ 轴转 $\pi$，$H$ 
门是一串绕轴旋转——光学原子就是一台天然的单比特量子计算机。</span>

## 2 光学 Bloch 方程的推导

在旋转波近似 + 偶极近似下，
把有效哈密顿量代入 $\dot{\rho} = -\frac{i}{\hbar}[\hat{H}_{\mathrm{eff}}, \rho]$，
再加唯象驰豫项，得到光学 Bloch 方程：

$$\dot{u} = -\gamma_2 u + \Delta v$$
$$\dot{v} = -\gamma_2 v - \Delta u + \Omega w$$
$$\dot{w} = -\gamma_1(w + 1) - \Omega v$$

三个参数定义：

- $\Omega$：拉比频率（场驱动强度）；
- $\Delta = \omega - \omega_0$：失谐；
- $\gamma_1 = 1/T_1$：**纵向驰豫率**（布居反转向平衡恢复，来自自发辐射，$\gamma_1 = A_{21}$）；
- $\gamma_2 = 1/T_2$：**横向驰豫率**（相干 $u,v$ 的衰减，$\gamma_2 \geq \gamma_1/2$，额外来自退相过程）。<span class="marginnote">$T_2$ 和 $T_1$ 的关系 $T_2 \leq 2T_1$：自发辐射本身就把相干带走了（$\gamma_2 \geq \gamma_1/2$），退相过程只会让 $T_2$ 更短。记住「$T_2$ 永远不慢于 $T_1$」。</span>

**重点：这组方程线性且定常，是「可解析求解」的幸运儿。** 
几乎所有的共振荧光、饱和吸收、相干布居俘获现象都是 OBE 的解。

## 3 稳态解：饱和与共振荧光谱

设 $\dot{u} = \dot{v} = \dot{w} = 0$，
解出稳态布居反转与相干：

$$w_{\mathrm{ss}} = -\frac{1 + \Delta^2/\gamma_2^2}{1 + \Delta^2/\gamma_2^2 + \Omega^2/(\gamma_1\gamma_2)}, \qquad \rho_{ee}^{\mathrm{ss}} = \frac{1+w_{\mathrm{ss}}}{2}$$

定义**饱和参数** $s = \frac{\Omega^2/2}{\gamma_1\gamma_2(1+\Delta^2/\gamma_2^2)}$，
则 $\rho_{ee}^{\mathrm{ss}} = \frac{s}{2(1+s)}$。
物理图像：

- 弱场（$s \ll 1$）：$\rho_{ee} \propto s \propto I$，吸收率正比于光强（线性区）。
- 强场（$s \gg 1$）：$\rho_{ee} \to 1/2$——**饱和**。无论场多强，激发布居最多半对半，多余光子只能被散射（共振荧光）。

共振荧光的稳态光子散射率 
= $\gamma_1\rho_{ee}^{\mathrm{ss}}$，
在强场饱和极限下为 $\gamma_1/2$——这就是「二能级原子的散射截面在饱和时减半」的量子根源。<span class="marginnote">散射截面饱和是激光冷却减速原子的关键：
光强越大减速越快，但饱和限制了最大散射率，
冷却速度存在上限。</span>

## 4 公式解析：稳态反转 $w_{\mathrm{ss}} = -\dfrac{1+\Delta^2/\gamma_2^2}{1+\Delta^2/\gamma_2^2+\Omega^2/(\gamma_1\gamma_2)}$

这条式子把「驱动 vs 驰豫」的竞争写得明明白白，拆成三步：

- **第一步，无场极限**：$\Omega = 0$ 时 $w_{\mathrm{ss}} = -1$，原子回到热平衡基态。分母的第二项 $\Omega^2/(\gamma_1\gamma_2)$ 是「场把原子推离平衡」的强度。
- **第二步，失谐的作用**：分子分母都含 $\Delta^2/\gamma_2^2$。失谐大时，分子分母同增，比值趋向 $1$——但 $w_{\mathrm{ss}}$ 的整体幅度被分母额外项压低，所以失谐削弱饱和。共振时 $\Delta = 0$，饱和最有效。
- **第三步，饱和参数重读**：把 $w_{\mathrm{ss}} = -1/(1+2s)$ 记法——这正是 Lorentzian 型饱和曲线：半饱和强度 $I_{\mathrm{sat}} = \frac{\epsilon_0 c\hbar^2\gamma_1\gamma_2}{2|d_{eg}|^2}$。记住「在 $I = I_{\mathrm{sat}}$ 处，激发布居达到其极限值的一半」。

## 5 Rabi 振荡与阻尼

不含驰豫（$\gamma_1 = \gamma_2 = 0$）时，
OBE 退化为上一节的拉比振荡。含驰豫且共振时，激发概率

$$P_e(t) = \frac{\Omega^2}{2\Omega^2 + \gamma_1\gamma_2}\left[1 - e^{-(\gamma_1+\gamma_2)t/2}\cos(\Omega' t)\right]$$

其中 $\Omega' = \sqrt{\Omega^2 - (\gamma_1-\gamma_2)^2/4}$ 
是阻尼拉比频率。振荡被指数包络压制，
最终衰减到稳态值 $P_e^{\infty} = \frac{\Omega^2/2}{\Omega^2 + \gamma_1\gamma_2}$。**这就是「从拉比振荡到稳态饱和」的完整故事线**：
初期相干振荡，后期耗散主导。

**辨析｜易错点：** 
当驰豫率与拉比频率可比（$\gamma \sim \Omega$）时，
系统进入**过阻尼**区，振荡消失，只有单调驰豫。
不要以为「有驱动就一定有振荡」——只有 $\Omega > \gamma$ 
的强驱动区才有清晰可见的 Rabi 振荡。这个判据也是腔 QED 
强耦合（$g > \gamma$）判据的孪生兄弟。<span class="marginnote">把 $\Omega$ 
换成腔耦合 $g$、$\gamma$ 换成腔泄漏 $\kappa$，
同样的数学就变成《腔 QED 
与强耦合条件》的判据——方程是普适的。</span>

## 6 小结

- 布洛赫矢量 $(u,v,w)$ 完整描述二能级原子：$w$ 是布居反转，$u+iv$ 是相干。
- 光学 Bloch 方程 = 相干驱动 + 唯象驰豫（$T_1$ 纵向、$T_2$ 横向）。
- 稳态饱和：$\rho_{ee}^{\mathrm{ss}} \to 1/2$（强场），散射率 $\to \gamma_1/2$