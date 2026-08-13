---
title: 单粒子Green函数与谱表示
date: 2026-08-07
---

# 单粒子Green函数与谱表示

<div class="epigraph">
<p>Green 函数是量子多体理论的计算中枢：它把「一个粒子从时空点 $(\mathbf{r}_1,t_1)$ 传播到 $(\mathbf{r}_2,t_2)$ 的振幅」包装成一个函数，而几乎所有可观测量——能量、密度、输运系数——都能从它身上读出来。</p>
<footer>—— G. D. Mahan（*Many-Particle Physics\*）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 2 ｜ 2026-08-07</p>
</div>

## 为什么从 Green 函数开始

上一节的二次量子化给了我们描述多体体系的算符语言，但只是「描述」还不够，我们真正要回答的是动力学问题：一个电子在固体里如何运动？它带着多大能量、以多快的速度在格点间跳跃？它与其他电子、声子发生相互作用后，还会不会像一个「干净的粒子」那样行为？

答案藏在**传播子（propagator）**里。量子力学的路径积分思想告诉我们：粒子从一点到另一点的「振幅」，就是传播子——它集齐了所有可能的中间过程。Green 函数就是传播子在多体体系里的严格推广。**只要算出了 Green 函数，粒子的能量谱、寿命、密度分布就全都到手了**；它在时间域的傅里叶变换甚至直接对应角分辨光电子谱（ARPES）的实验读数。<span class="marginnote">ARPES 实验测到的光电子强度谱 $I(\mathbf{k},\omega)$，在理论上正比于单粒子谱函数 $A(\mathbf{k},\omega)$。这使 Green 函数不仅是理论工具，还是直接可被实验检验的物理量。</span>

## 1 三类 Green 函数：定义与物理图像

在温度为零的相互作用体系中，最常用的是**时序 Green 函数（time-ordered Green function）**，定义为场算符的时序乘积的真空期望值：

$$G(\mathbf{r},t;\mathbf{r}',t') = -i\langle 0 |\, \mathcal{T}\big[\hat{\psi}(\mathbf{r},t)\hat{\psi}^\dagger(\mathbf{r}',t')\big]\, |0\rangle$$

其中 $\mathcal{T}$ 是**时序算符（time-ordering operator）**：把算符按时间从晚到早重新排列，且费米子每次交换次序要贡献一个负号：

$$\mathcal{T}\big[\hat{\psi}(t)\hat{\psi}^\dagger(t')\big] = \begin{cases} \hat{\psi}(t)\hat{\psi}^\dagger(t'), & t > t' \\ -\hat{\psi}^\dagger(t')\hat{\psi}(t), & t \lt  t' \end{cases}$$

物理图像非常直观：当 $t > t'$ 时，函数描述「先在 $t'$ 时刻、$\mathbf{r}'$ 处产生一个粒子，传播到 $t$ 时刻、$\mathbf{r}$ 处湮灭」——正比于电子从 $\mathbf{r}'$ 传到 $\mathbf{r}$ 的振幅；当 $t \lt  t'$ 时，描述的是空穴反向传播。**一个函数同时囊括了粒子和空穴两种传播**，这是多体 Green 函数与单粒子波函数本质不同的一点。<span class="marginnote">费米子时序交换要加负号，这个负号看似琐碎，却贯穿整个 Feynman 图技术：费米子闭合圈每圈贡献一个 $-1$，最终影响所有输运与自能计算的符号。</span>

除了时序函数，还有更「物理」的两个成员：**推迟 Green 函数** $G^R$（retarded）与**超前 Green 函数** $G^A$（advanced）。它们定义为：

$$G^R(\mathbf{r},t;\mathbf{r}',t') = -i\theta(t-t')\langle \{\hat{\psi}(\mathbf{r},t),\hat{\psi}^\dagger(\mathbf{r}',t')\}\rangle$$

其中 $\theta(t-t')$ 是阶跃函数，$\{\cdot,\cdot\}$ 是反对易子。推迟函数只在 $t>t'$ 时非零，天然因果——「原因永远在结果之前」。它是线性响应理论（Kubo 公式，见本专题后续篇目）与输运计算的常客，也是我们最终取物理极限时最方便的对象。

## 2 动量与频率空间的 Green 函数

平移不变体系下，Green 函数只依赖坐标差 $\mathbf{r}-\mathbf{r}'$，作 Fourier 变换后得到动量空间形式。对自由电子气，结果是一个令人安心的简洁式：

$$G^{(0)}(\mathbf{k},\omega) = \frac{1}{\omega - \varepsilon_{\mathbf{k}} + i\eta\,\text{sgn}(\omega - \mu)}$$

这里 $i\eta$ 是一个无穷小正数（$\eta \to 0^+$），它的作用是为积分提供正确的绕极方向。**Green 函数在复 $\omega$ 平面上有一个位于 $\omega = \varepsilon_{\mathbf{k}}$ 的单极点**——这个极点的位置就是单粒子激发能量。<span class="marginnote">$i\eta$ 的符号决定极点在实轴上方还是下方，进而决定积分围道的取法。推迟函数取 $G^R \propto 1/(\omega - \varepsilon + i\eta)$（极点下移），超前函数取 $i\eta \to -i\eta$。这条「无穷小虚部」是 Green 函数技术中最易错也最关键的地方。</span>

**辨析｜易错点：** 初学者常问「为什么自由 Green 函数里要人为加一个 $i\eta$？它不改变物理吗？」答案是否定的——它不改变物理，但它决定了数学：没有 $i\eta$，$\omega=\varepsilon_{\mathbf{k}}$ 处的极点正好落在积分路径上，Fourier 变换不收敛；加了 $i\eta$，极点被推向复平面，积分才能用留数定理完成，且取 $\eta\to0^+$ 后物理回到实数轴上。它是一个「正则化技巧」，而非物理输入。

## 3 谱表示：把 Green 函数交给谱函数

Green 函数最深刻的性质是它**完全由谱函数决定**。所谓**谱函数（spectral function）**$A(\mathbf{k},\omega)$ 定义为推迟函数虚部的负号除以 $\pi$：

$$A(\mathbf{k},\omega) = -\frac{1}{\pi}\,\text{Im}\, G^R(\mathbf{k},\omega)$$

谱函数是正定的（$A \geq 0$），且满足**求和规则（sum rule）**：

$$\int_{-\infty}^{\infty} d\omega\, A(\mathbf{k},\omega) = 1$$

物理上，$A(\mathbf{k},\omega)$ 度量「动量为 $\mathbf{k}$ 时，体系以多大权重在能量 $\omega$ 上拥有单粒子激发」。对自由电子气，$A^{(0)}(\mathbf{k},\omega) = \delta(\omega - \varepsilon_{\mathbf{k}})$——谱函数是一根打在 $\varepsilon_{\mathbf{k}}$ 上的 δ 函数尖峰，代表一个无限寿命的准粒子。<span class="marginnote">当相互作用开启后，谱函数从 δ 尖峰展宽成 Lorentzian 峰：峰位给出准粒子能量，峰宽给出准粒子寿命（衰减率 $\Gamma = 1/\tau$），峰的谱权重被重正化因子 $Z$ 压低。这套「峰位-峰宽-权重」的解读是理解费米液体（本专题第 2 篇）的钥匙。</span>

有了谱函数，整个 Green 函数可以重建为（这就是**谱表示 / Lehmann 表示**）：

$$G^R(\mathbf{k},\omega) = \int_{-\infty}^{\infty} d\omega'\, \frac{A(\mathbf{k},\omega')}{\omega - \omega' + i\eta}$$

这条式子说明：Green 函数无非是谱函数与分母 $1/(\omega-\omega'+i\eta)$ 的卷积。所有「相互作用如何改变单粒子物理」的信息，都被压缩进了谱函数 $A(\mathbf{k},\omega)$ 这一个正函数里。

## 4 公式解析：自由电子 Green 函数的谱表示

把自由电子的推迟 Green 函数 $G^{R(0)}(\mathbf{k},\omega) = 1/(\omega - \varepsilon_{\mathbf{k}} + i\eta)$ 代进谱表示，可以完整验证整条逻辑链：

- **第一步，写出谱函数**：$A^{(0)}(\mathbf{k},\omega) = -\frac{1}{\pi}\text{Im}\,\frac{1}{\omega-\varepsilon_{\mathbf{k}}+i\eta}$。利用公式 $\text{Im}\,\frac{1}{x+i\eta} = -\pi\delta(x)$，得 $A^{(0)}(\mathbf{k},\omega) = \delta(\omega-\varepsilon_{\mathbf{k}})$。
- **第二步，代入谱表示**：$G^{R(0)}(\mathbf{k},\omega) = \int d\omega'\, \frac{\delta(\omega'-\varepsilon_{\mathbf{k}})}{\omega - \omega' + i\eta} = \frac{1}{\omega - \varepsilon_{\mathbf{k}} + i\eta}$——δ 函数把积分收缩成单一项，回到起点。闭环自洽。
- **第三步，读出物理**：求和规则 $\int d\omega\, \delta(\omega-\varepsilon_{\mathbf{k}}) = 1$ 自动满足；谱函数只有一个频率有重量，说明自由电子是「单色」的——能量确定、寿命无限。

**重点：谱函数是「多体态有多少单粒子成分」的定量答案。** 它的三个特征——峰位（能量）、峰宽（寿命）、总权重（占据数）——分别对应实验中可测的色散关系、散射率与费米面跳跃。这也是为什么几乎所有凝聚态谱学实验（光电子谱、中子散射、隧穿谱）最后都要翻译成谱函数的语言。

## 5 占据数与谱函数

谱函数还编码了体系处于热平衡时的粒子分布。零温下，**占据数**由谱函数在负频率一侧的积分给出：

$$\langle \hat{n}_{\mathbf{k}\sigma}\rangle = \int_{-\infty}^{\mu} d\omega\, A(\mathbf{k},\omega)$$

对自由电子气，$\mu$（化学势）以下的 δ 峰被完整积分，$\langle n_{\mathbf{k}\sigma}\rangle = 1$（费米面内）或 $0$（费米面外）——这就是零温费米子分布 $n_{\mathbf{k}} = \theta(\mu - \varepsilon_{\mathbf{k}})$ 的谱函数表述。相互作用把 δ 峰展宽后，占据数不再是 0 或 1，而会在费米面附近出现**连续过渡**，这正对应动量分布函数在费米面处跳跃从 1 降到 $Z \lt  1$——费米液体理论里费米面跳跃 $Z$ 的核心概念（后文详述）。

<span class="marginnote">谱函数对频率的积分与化学势 $\mu$ 的联系，是零温与有限温度公式的「交接点」：到有限温度（本专题 Matsubara 篇）时，$\theta(\mu-\omega)$ 会被费米函数 $f(\omega)=1/(e^{\beta(\omega-\mu)}+1)$ 取代，同一套框架无缝延拓。</span>

## 6 小结

- **时序 Green 函数** $G = -i\langle \mathcal{T}\psi\psi^\dagger\rangle$ 同时描述粒子与空穴的传播；费米子时序交换贡献负号。
- **推迟/超前 Green 函数**满足因果律，是输运与响应计算的常客；两者由无穷小虚部 $i\eta$ 的符号区分。
- 自由电子 Green 函数 $G^{(0)}(\mathbf{k},\omega) = 1/(\omega-\varepsilon_{\mathbf{k}}+i\eta)$ 有一个单极点，极点位置即单粒子能量。
- **谱函数** $A(\mathbf{k},\omega) = -\text{Im}\,G^R/\pi$ 正定且求和规则 $\int d\omega\, A = 1$；峰位、峰宽、权重分别给出能量、寿命与准粒子权重。
- **谱表示** $G^R = \int d\omega'\, A(\omega')/(\omega-\omega'+i\eta)$ 把 Green 函数完全还原成谱函数，是理解相互作用体系的万能框架。
- 占据数 $\langle n\rangle = \int_{-\infty}^\mu d\omega\, A(\mathbf{k},\omega)$，谱函数的展宽对应费米面跳跃 $Z$ 的压低。

在下一节，我们将给 Green 函数装上「时间演化」的引擎：**相互作用绘景与 Wick 定理**——它告诉你如何把相互作用项系统地展开，并把复杂的时序乘积化简成一组对易子的收缩，这是微扰理论与 Feynman 图的开场。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| 时序 Green 函数 | $G = -i\langle\mathcal{T}[\psi\psi^\dagger]\rangle$ | 粒子与空穴传播同时编码 |
| 自由 Green 函数 | $G^{(0)}(\mathbf{k},\omega) = 1/(\omega-\varepsilon_{\mathbf{k}}+i\eta)$ | 单极点，位置即单粒子能量 |
| 谱函数 | $A(\mathbf{k},\omega) = -\text{Im}\,G^R/\pi$ | 正定，$\int d\omega\,A=1$ |
| 谱表示 | $G^R = \int d\omega'\, A(\omega')/(\omega-\omega'+i\eta)$ | Green 函数由谱函数完全决定 |
| 占据数 | $\langle n\rangle = \int_{-\infty}^{\mu}d\omega\,A(\mathbf{k},\omega)$ | 谱函数携带统计信息 |

**易错复盘**：三点要盯住。其一，$i\eta$ 的符号决定推迟/超前——它是正则化技巧而非物理输入，但符号错了物理就错；其二，时序函数同时含粒子与空穴传播（费米子带符号），与单粒子波函数本质不同；其三，谱函数的三个特征（峰位、峰宽、权重）分别对应能量、寿命、占据——ARPES 直接测它。

**知识连线**：Green 函数是第 1 篇二次量子化的「动力学延伸」；谱表示与第 2 篇 Kubo 公式（推迟函数）直接相连，占据数公式连接有限温度（第 2 篇 Matsubara）。它是后续所有微扰、Dyson、响应计算的共同起点。

**实践与辨析**：为什么 $\int d\omega\,A(\mathbf{k},\omega)=1$ 总成立？提示：谱函数是正定概率密度，求和规则来自对易子的完整关系。相互作用为什么把 δ 峰展宽成 Lorentzian？提示：自能虚部 $\text{Im}\Sigma$ 给出寿命 $\tau=\hbar/|\Gamma|$。易错提醒：展宽后峰面积 $Z\lt 1$