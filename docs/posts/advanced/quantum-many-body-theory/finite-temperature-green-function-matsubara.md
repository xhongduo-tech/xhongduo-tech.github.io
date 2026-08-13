---
title: 有限温度Green函数与Matsubara求和
date: 2026-08-07
---

# 有限温度Green函数与Matsubara求和

<div class="epigraph">
<p>真实的实验几乎都在有限温度下进行。把零温 Green 函数方法搬进热平衡，Matsubara 在 1955 年给出了最优雅的方案：让时间绕道虚轴走一圈，微扰理论的一切工具原封不动地继续工作。</p>
<footer>—— G. D. Mahan（*Many-Particle Physics\*）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 3 ｜ 2026-08-07</p>
</div>

## 为什么需要有限温度 Green 函数

前几篇的 Green 函数技术全部建立在**零温基态**之上：真空态 $|0\rangle$ 的期望值、时序乘积、Wick 定理。但现实中的实验几乎都在有限温度下进行——金属在室温、超导体在低温但非零温、冷原子实验从 $\mu\text{K}$ 到 $n\text{K}$。有限温度下，体系不再处于单一基态，而是按玻尔兹曼权重 $\exp(-\beta H)$（$\beta = 1/k_BT$）处于**混合态（thermal ensemble）**。

困难在于：零温技巧依赖「真空态的时序期望值」，而热平衡是统计混合。Matsubara 的洞见是：**把时间轴转到虚轴，让热权重 $\exp(-\beta H)$ 天然变成一个「虚时演化算符」**，于是有限温度的配分函数可以写成虚时路径上的时序乘积——Wick 定理、Feynman 图、Dyson 方程全部照搬。<span class="marginnote">这方法被称为<strong>松原（Matsubara）虚时形式</strong>。它把「温度」这个听起来很统计的概念，翻译成一个「虚时间长度 $\beta$」的几何概念——温度越高，虚时周期越短。这是物理学家用几何消解复杂性的又一次胜利。</span>

## 1 虚时与温度的联系

关键观察从配分函数出发：

$$Z = \text{Tr}\big[e^{-\beta H}\big] = \sum_n e^{-\beta E_n}$$

在虚时 $\tau = it$ 下，演化算符 $e^{-iHt}$ 变成 $e^{-H\tau}$。于是热权重 $e^{-\beta H}$ 恰好是「把体系在虚时间 $\tau$ 上演化 $\beta$ 这么久」。由此，有限温度平均可以用虚时序乘积写成：

$$\langle \mathcal{T}_\tau[\hat{A}(\tau)\hat{B}(\tau')]\rangle \equiv \frac{\text{Tr}\big[e^{-\beta H}\mathcal{T}_\tau[\hat{A}(\tau)\hat{B}(\tau')]\big]}{\text{Tr}[e^{-\beta H}]}$$

其中虚时间演化定义为 $\hat{A}(\tau) = e^{H\tau}\hat{A}e^{-H\tau}$，$\mathcal{T}_\tau$ 是虚时序算符。**重点：虚时 Green 函数在 $\tau$ 方向具有周期性（或反周期性）**——对玻色子 $\mathcal{G}(\tau+\beta) = \mathcal{G}(\tau)$，对费米子 $\mathcal{G}(\tau+\beta) = -\mathcal{G}(\tau)$。这个边界条件将决定后面的离散频率结构。<span class="marginnote">周期 vs 反周期的根源：把 $\hat{A}(\tau+\beta)$ 通过 $e^{-\beta H}$ 挪到 $\hat{B}$ 另一边时，费米子交换一次产生 $-1$。这个看似几何的边界条件，最终决定了费米子与玻色子求和时频率取值的不同——一切又回到统计性质。</span>

## 2 松原频率与频率求和

虚时方向有界（$0 \le \tau \le \beta$），所以虚时 Green 函数的 Fourier 级数是**离散**的：

$$\mathcal{G}(i\omega_n) = \int_0^\beta d\tau\, e^{i\omega_n \tau}\,\mathcal{G}(\tau)$$

由周期性边界条件，频率 $\omega_n$ 只能取分立值——这就是**松原频率（Matsubara frequency）**：

- 费米子（反周期）：$\omega_n = (2n+1)\pi k_BT / \hbar$，奇频率，$n \in \mathbb{Z}$。
- 玻色子（周期）：$\omega_n = 2n\pi k_BT / \hbar$，偶频率。

**重点：费米子与玻色子的松原频率错开了一个单位**——奇偶分立，互不重合。这使得「温度 $T$」通过频率间距进入一切公式：$T \to 0$ 时频率变密，松原求和回到零温的积分。**松原求和（Matsubara sum）**指处理形如 $\frac{1}{\beta}\sum_{i\omega_n} f(i\omega_n)$ 的离散求和，标准做法是把它换成对复平面留数的围道积分，最终结果总可以写成「费米函数 $f(\omega)=1/(e^{\beta\omega}+1)$ 或玻色函数 $n_B(\omega)=1/(e^{\beta\omega}-1)$ 乘解析函数」的形式。<span class="marginnote">频率求和的标配套路：把 $i\omega_n$ 看成复函数 $\frac{1}{\beta}\sum_n f(i\omega_n) = \oint \frac{dz}{2\pi i}\, f(z)\, n_{B/F}(z)$。围道积分后，结果由 $f$ 的极点贡献，费米子用 $n_F$、玻色子用 $n_B$——统计因子就这么自然出现。</span>

## 3 有限温度 Green 函数的谱表示

有限温度下的 Green 函数同样有谱表示，只是把零温的真空期望替换为热平均。对松原函数，谱表示写为：

$$\mathcal{G}(\mathbf{k}, i\omega_n) = \int_{-\infty}^{\infty} d\omega'\, \frac{A(\mathbf{k},\omega')}{i\omega_n - \omega'}$$

谱函数 $A(\mathbf{k},\omega)$ 的定义与零温一致（$-\text{Im}\,G^R/\pi$），求和规则也相同。区别在于：有限温度下粒子占据由**费米函数** $n_F(\omega) = 1/(e^{\beta(\omega-\mu)}+1)$ 给出，占据数公式变为：

$$\langle n_{\mathbf{k}}\rangle = \int_{-\infty}^{\infty} d\omega\, A(\mathbf{k},\omega)\, n_F(\omega)$$

零温的 $\theta(\mu-\omega)$ 正是 $T\to0$ 时 $n_F(\omega)$ 的极限——温度只是把费米面的锐利台阶「磨圆」了。<span class="marginnote">这条「零温 θ 函数 → 有限温费米函数」的替换是普适的：它对应统计力学里「基态占据」到「热占据」的过渡。对玻色子，对应的因子是玻色函数 $n_B(\omega)$；两者在化学势与发散行为上的差异（玻色凝聚！）将在本专题第 3 篇大放异彩。</span>

**辨析｜易错点：** 初学者最容易在两点上翻车。其一，**松原频率是离散虚频率，不是实频率**——$i\omega_n$ 是纯虚数，只有做完解析延拓 $i\omega_n \to \omega + i\eta$ 才能得到推迟函数的实频率极限；其二，**有限温度的 Feynman 图规则里，内部线的频率求和是 $\frac{1}{\beta}\sum_{i\omega_n}$ 而不是积分**——频率离散化导致每个内圈多一个求和，这是有限温计算量比零温大的根本原因。

## 4 公式解析：单环的松原求和

用最简的例子把整套技术走一遍：计算一个玻色子环的配对涨落（后面 RPA 的雏形）：

$$
\Pi_0(i\nu_m) = \frac{1}{\beta}\sum_{i\omega_n} G_0(i\omega_n)\, G_0(i\omega_n + i\nu_m)
$$

- **第一步，代入自由传播子**：$G_0(i\omega_n) = 1/(i\omega_n - \xi_{\mathbf{k}})$，其中 $\xi_{\mathbf{k}} = \varepsilon_{\mathbf{k}} - \mu$。第二个因子 $1/(i\omega_n+i\nu_m - \xi_{\mathbf{k}+\mathbf{q}})$。
- **第二步，化为围道积分**：费米子求和（$\omega_n$ 为费米松原频率）换成 $\oint \frac{dz}{2\pi i}\, \frac{1}{z-\xi_{\mathbf{k}}}\,\frac{1}{z+i\nu_m-\xi_{\mathbf{k}+\mathbf{q}}}\, n_F(z)$。
- **第三步，取留数**：被积函数在 $z=\xi_{\mathbf{k}}$ 与 $z = \xi_{\mathbf{k}+\mathbf{q}} - i\nu_m$ 有极点，取两个留数并相加：
  $$\Pi_0(i\nu_m) = \frac{n_F(\xi_{\mathbf{k}}) - n_F(\xi_{\mathbf{k}+\mathbf{q}})}{i\nu_m - (\xi_{\mathbf{k}+\mathbf{q}} - \xi_{\mathbf{k}})}$$
- **第四步，读出物理**：分子上的费米函数差 $n_F(\xi_{\mathbf{k}}) - n_F(\xi_{\mathbf{k}+\mathbf{q}})$ 正是「泡利不相容允许的粒子-空穴对激发」的相空间因子；分母是粒子-空穴对的能量差。这个表达式就是 Lindhard 响应函数的有限温版本，它的虚部给出粒子的能量可被体系吸收的条件——这正是后面介电函数与等离子体激元（本专题第 2 篇）的核心输入。

**重点：松原求和把「温度」变成「统计因子」，把「频率求和」变成「留数求和」。** 所有温度依赖最后都收进费米/玻色函数，这是有限温 Green 函数技术能像零温一样系统推进的原因。

## 5 从有限温度到「从极限到大模型」

有限温度 Green 函数在精神上极接近机器学习里的**温度采样与退火**：温度控制分布的「锐度」，$T\to0$ 取最概然（基态），高温取均匀（混沌）。大模型解码里的「temperature」参数、模拟退火算法、以及扩散模型里噪声强度的调度，都是同一族思想——用「温度」这个旋钮在「确定/随机」之间平滑过渡。<span class="marginnote">更深的联系在统计学习理论：玻尔兹曼机就是直接用玻尔兹曼分布 $\exp(-\beta E)$ 作生成模型的。想让模型「敢于探索」，就升高温度；想让答案「收敛」，就退火降温——与 Matsubara 形式里 $\beta$ 控制频率间距是同一个参数。</span>

对多体物理自身而言，有限温度 Green 函数是通向**超导转变温度 $T_c$、Kondo 效应、量子临界现象**等一切「温度驱动的相变」的必经之路——BCS 超导的 $T_c$ 公式就是松原求和的直接产物（本专题第 3 篇）。

## 6 小结

- 有限温度体系处于热混合态，需用配分函数 $Z=\text{Tr}\,e^{-\beta H}$ 描述；**虚时** $\tau=it$ 把温度变成虚时间长度 $\beta$。
- **虚时 Green 函数**满足周期（玻色子）/反周期（费米子）边界条件，导致频率离散化。
- **松原频率**：费米子 $\omega_n=(2n+1)\pi k_BT/\hbar$，玻色子 $\omega_n=2n\pi k_BT/\hbar$，奇偶分立。
- **松原求和**通过围道积分化为留数求和，温度依赖全部收进费米/玻色统计因子。
- 谱表示在有限温度下仍成立，占据数由费米函数 $n_F(\omega)$ 给出，$T\to0$ 回到 $\theta(\mu-\omega)$。
- 解析延拓 $i\omega_n \to \omega+i\eta$ 是连接松原函数与推迟函数的必要步骤。

在下一节，我们将回到实频率世界，讨论最有实验味道的一对对象：**推迟 Green 函数与 Kubo 公式**——它告诉你如何用 Green 函数计算线性响应（电导、磁化率、电导率），把「微观关联」翻译成「宏观输运系数」。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| 虚时 Green 函数 | $\mathcal{G}(\tau) = -\langle\mathcal{T}_\tau[\psi(\tau)\psi^\dagger(0)]\rangle$ | 虚时 $\tau=it$，温度进几何 |
| 松原频率 | 费米子 $(2n+1)\pi k_BT/\hbar$，玻色子 $2n\pi k_BT/\hbar$ | 奇偶分立，互不重合 |
| 谱表示 | $\mathcal{G}(i\omega_n) = \int d\omega'\, A(\omega')/(i\omega_n-\omega')$ | 松原函数由谱函数完全决定 |
| 占据数 | $\langle n_{\mathbf{k}}\rangle = \int d\omega\, A(\mathbf{k},\omega)\,n_F(\omega)$ | $T\to0$ 回到 $\theta(\mu-\omega)$ |
| 单环求和 | $\Pi_0(i\nu_m) = \frac{n_F(\xi)-n_F(\xi')}{i\nu_m-(\xi'-\xi)}$ | 粒子-空穴对激发的相空间 |

**易错复盘**：两点最容易翻车。其一，松原频率是纯虚数——$i\omega_n$ 必须解析延拓 $i\omega_n\to\omega+i\eta$ 才能得到实频率推迟函数；其二，有限温图规则的内部频率求和是 $\frac{1}{\beta}\sum_{i\omega_n}$ 而非积分——频率离散化让每个内圈多一个求和，这是有限温计算量增大的根源。

**知识连线**：本篇把第 1 篇的零温 Green 函数技术整体搬到热平衡，是第 2 篇 Kubo 公式（推迟函数）在有限温的计算工具；松原求和直接产出第 3 篇 BCS 的 $T_c$ 公式与第 3 篇 Eliashberg 方程。「零温 $\theta$ 函数 → 有限温 $n_F/n_B$」的替换，是「从基态到热平衡」的通用桥梁。

**延伸思考**：为什么费米子松原频率是奇频率、玻色子是偶频率？提示：从虚时 Green 函数的反周期/周期边界条件出发。$T\to0$ 时松原频率间距趋于零，求和应回到积分——请验证单环求和的 $T\to0$ 极限。


**实践与辨析**：一道综合题：计算玻色子单环极化 $\Pi_0(i\nu_m)$ 并说明它的物理意义。提示：把费米子求和换成围道积分，取两个极点留数，得到 $[n_F(\xi)-n_F(\xi')]/(i\nu_m-(\xi'-\xi))$。易错提醒：求和因子是 $1/\beta$ 且频率是离散松原频率，不能直接换成 $\int d\omega/2\pi$——除非 $T\to0$。

**延伸阅读**：Matsubara 形式与频率求和见 Mahan《Many-Particle Physics》第 3 章；解析延拓的技巧与陷阱见该书第 3 章末节。


**延伸阅读**：松原形式与频率求和见 Mahan《Many-Particle Physics》第 3 章；有限温 Green 函数在超导与磁性中的应用贯穿该书第 10 章；解析延拓技巧的严格讨论见 Abrikosov, Gorkov & Dzyaloshinskii《Methods of Quantum Field Theory in Statistical Physics》。


**关键术语**：虚时 $\tau$、松原频率 $\omega_n$