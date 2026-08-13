---
title: 光与原子相互作用的半经典理论
date: 2026-08-07
---

# 光与原子相互作用的半经典理论

<div class="epigraph">
<p>把原子量子化而把光当作经典波——这看似不彻底，却解释了激光、拉比振荡和半数的现代光学。</p>
<footer>—— 罗伊·格劳伯（Roy J. Glauber）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ R. Loudon, The Quantum Theory of Light 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从半经典理论开始

处理「光 + 原子」的相互作用，
有两条进路。**全量子理论**把光也量子化（Jaynes-Cummings 
模型，见本专题专篇）；**半经典理论**只把原子量子化，
光场仍当作经典的电磁波 $E(t)$。出人意料的是，
半经典理论能解释**受激吸收、受激辐射、拉比振荡、光学布洛赫方程、饱和吸收**等一系列核心现象——只有自发辐射和真空涨落它解释不了。
先学半经典，是为了把「原子对经典场的响应」这一半先搞透，
再补上「场的量子涨落」另一半。<span class="marginnote">这条「先半经典、
后全量子」的教学路线，与第二级《量子力学》里「先氢原子、
后微扰论」的渐进式推进一脉相承。</span>

## 1 偶极近似与相互作用哈密顿量

原子在光场中感受到的主要是**电偶极相互作用**。
设原子的电偶极矩算符 $\hat{\mathbf{d}} = -e\hat{\mathbf{r}}$，
经典光场 $\mathbf{E}(t) = \mathbf{E}_0\cos(\omega t)$，
则相互作用哈密顿量为

$$\hat{H}_{\mathrm{int}} = -\hat{\mathbf{d}}\cdot\mathbf{E}(t)$$

在偶极近似里，
我们忽略了光场随空间的变化（$\vec{k}\cdot\vec{r} \ll 1$，
因为原子尺度远小于光波长）。
对于二能级原子（基态 $|g\rangle$、
激发态 $|e\rangle$，
能量差 $\hbar\omega_0$），相互作用在基矢下的矩阵元为

$$\hat{H}_{\mathrm{int}} = -\hbar\Omega\cos(\omega t)\,\hat{\sigma}_x, \qquad \hbar\Omega = \mathbf{d}_{eg}\cdot\mathbf{E}_0$$

其中 $\Omega$ 
叫**拉比频率（Rabi frequency）**，$\hat{\sigma}_x = |e\rangle\langle g| + |g\rangle\langle e|$ 
是跃迁算符。<span class="marginnote">偶极近似成立的条件 $\lambda \gg a_0$（波长远大于玻尔半径）对可见光（$\lambda \sim 500$ 
nm）与原子（$a_0 \sim 0.1$ nm）显然成立；
但这一近似在微波腔里也开始失效——那是腔 QED 
需要更精细处理的起点。</span>

## 2 旋转波近似与有效哈密顿量

把 $\cos(\omega t) = \frac{1}{2}(e^{i\omega t} + e^{-i\omega t})$ 
代入，并换到相互作用绘景，会出现四类项，
其中两类振荡频率 $\omega + \omega_0$（反旋项，
快速振荡），两类 $\omega - \omega_0$（共旋项，
近共振时慢速）。**旋转波近似（RWA）** 丢弃快速振荡项，
理由是它们在时间平均中相互抵消。于是得到（在旋转坐标里）

$$\hat{H}_{\mathrm{eff}} = \frac{\hbar\Delta}{2}\hat{\sigma}_z - \frac{\hbar\Omega}{2}\hat{\sigma}_x, \qquad \Delta = \omega - \omega_0$$

$\Delta$ 是**失谐（detuning）**。
这个 $2\times2$ 
有效哈密顿量完全决定了共振附近原子的动力学——它是光学布洛赫方程的出发点。<span class="marginnote">RWA 
成立条件 $|\Delta| \ll \omega + \omega_0$，
即近共振且场不强。强场、
超强耦合时反旋项会显形（Bloch-Siegert 位移），这是腔 
QED 强耦合区的重要修正。</span>

## 3 拉比振荡：共振驱动下的原子翻转

在共振 $\Delta = 0$ 时，
有效哈密顿量为 $\hat{H}_{\mathrm{eff}} = -\frac{\hbar\Omega}{2}\hat{\sigma}_x$。
初始在基态的原子时间演化

$$|\psi(t)\rangle = \cos\left(\frac{\Omega t}{2}\right)|g\rangle - i\sin\left(\frac{\Omega t}{2}\right)|e\rangle$$

于是激发概率

$$P_e(t) = |\langle e|\psi(t)\rangle|^2 = \sin^2\left(\frac{\Omega t}{2}\right)$$

原子在基态与激发态之间以角频率 $\Omega$ 
正弦振荡——这就是**拉比振荡**。<span class="marginnote">拉比振荡是量子控制的基础：$\Omega t = \pi$ 
的脉冲（$\pi$ 
脉冲）把原子完全翻转到激发态，$\Omega t = \pi/2$ 
的脉冲（$\pi/2$ 
脉冲）制备等权重叠加态——这是量子计算的「门」操作原型。</span>

**重点：拉比频率 $\Omega$ 正比于场振幅 $E_0$ 与偶极矩 $d_{eg}$。** 
场越强、跃迁偶极矩越大，翻转越快。
这与爱因斯坦理论里「跃迁速率正比于光强（$E_0^2$）」不同——后者是**时间平均**的结果，
而拉比振荡是**相干**的短期行为；一旦引入驰豫，两者就统一了。

## 4 公式解析：激发概率 $P_e(t) = \sin^2(\Omega t/2)$

这条式子看似简单，却承载着全部共振动力学的信息，拆成三步：

**第一步，解薛定谔方程**：$\hat{H} = -\hbar\Omega\hat{\sigma}_x/2$ 的本征态是 $\frac{1}{\sqrt{2}}(|g\rangle \mp i|e\rangle)$，能量 $\pm\hbar\Omega/2$。初始态 $|g\rangle$ 是这两个本征态的等权叠加，因此时间演化是两个频率的差拍——差频恰好是 $\Omega$。
**第二步，物理意义**：$\sin^2(\Omega t/2)$ 的周期 $T = 2\pi/\Omega$ 是「原子完成一个完整翻转周期」的时间。$t = \pi/\Omega$ 处 $P_e = 1$（完全激发），$t = 2\pi/\Omega$ 处回到 $P_e = 0$。
- **第三步，失谐修正**：$\Delta \neq 0$ 时，$P_e(t) = \frac{\Omega^2}{\Omega^2 + \Delta^2}\sin^2\left(\frac{\sqrt{\Omega^2+\Delta^2}}{2}t\right)$。有效拉比频率变为 $\Omega' = \sqrt{\Omega^2 + \Delta^2}$，且最大激发概率降至 $\Omega^2/(\Omega^2+\Delta^2)$——远离共振，翻转效率降低。这就是「共振泵浦」为何要锁频的原因。

## 5 半经典理论的边界

半经典理论辉煌却有限。它解释不了三个纯量子现象：

- **自发辐射**：真空中没有经典场（$\mathbf{E}_0 = 0$），半经典框架给不出跃迁速率，但原子确实自发发光——必须有场的零点涨落参与。
- **Lamb 位移**：原子能级的微小移动，来自真空涨落对电子自能修正。
- **光子反聚束**：半经典光场加经典探测理论给不出 $g^{(2)}(0) \lt  1$ 的预测。

这三个「失败」恰好勾勒出量子化的必要性：**经典场是相干态在平均场极限下的投影，真空涨落才是量子光学的灵魂**。
下一节我们就进入全量子框架，
看自发辐射如何在量子化场中获得解释。<span class="marginnote">半经典「够用又不够用」的双重身份，
是理解量子光学边界的最佳例证——它解释了受激过程，
却把自发过程留给了真空。</span>

## 6 小结

- 半经典模型：原子量子化 + 场经典化，哈密顿量 $\hat{H}_{\mathrm{int}} = -\hat{\mathbf{d}}\cdot\mathbf{E}(t)$。
- 偶极近似 + 旋转波近似给出有效哈密顿量 $\hat{H}_{\mathrm{eff}} = \frac{\hbar\Delta}{2}\hat{\sigma}_z - \frac{\hbar\Omega}{2}\hat{\sigma}_x$。
- 共振拉比振荡：$P_e(t) = \sin^2(\Omega t/2)$