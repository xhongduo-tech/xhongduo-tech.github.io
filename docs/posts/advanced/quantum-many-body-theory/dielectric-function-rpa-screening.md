---
title: 介电函数与RPA屏蔽
date: 2026-08-07
---

# 介电函数与RPA屏蔽

<div class="epigraph">
<p>电子的存在改变真空的性质：一个放在金属里的裸电荷，会被电子云包裹、再包裹，最终它的电场被大幅衰减。介电函数就是对这个「自组织屏蔽」的定量描述。</p>
<footer>—— G. D. Mahan（*Many-Particle Physics\*）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 5 ｜ 2026-08-07</p>
</div>

## 为什么需要介电函数

上一节的均匀电子气里，电子通过长程库仑势 $V(\mathbf{q}) = 4\pi e^2/q^2$ 相互作用。但真实的金属中，这个长程势几乎从未「裸奔」过：一个外来的电荷（或一个电子）放进电子气，会立即排斥周围的电子、吸引周围的正背景，在它周围形成一个**极化云**，从而把它的有效电场大大削弱。这个现象叫**屏蔽（screening）**，它的定量表述就是**介电函数（dielectric function）**。

介电函数是理解金属导电、杂质散射、超导配对乃至整个凝聚态相互作用层次的核心：**一切长程库仑相互作用在真实材料里都要除以介电函数**。而计算介电函数的标准工具，就是**随机相位近似（Random Phase Approximation, RPA）**——它把极化过程近似成无穷多个「粒子-空穴对」独立激发的叠加。这一篇我们把 Lindhard 函数、RPA 求和与三种屏蔽极限讲透。<span class="marginnote">介电函数最早来自经典电动力学（电位移 $D=\varepsilon E$），但这里的 $\varepsilon(\mathbf{q},\omega)$ 是<strong>动量与频率依赖</strong>的微观介电函数——它不再是常数，而是编码了体系所有极化模式（粒子-空穴对、等离子体激元）的响应。凝聚态里「介电」二字的含义，远超中学物理的「介电常数」。</span>

## 1 介电函数的定义与极化

考虑电子气中一个**试探电荷**（外加电荷密度 $\rho_{\text{ext}}$）。体系的响应是形成**感应电荷密度** $\rho_{\text{ind}}$，总电荷密度 $\rho_{\text{tot}} = \rho_{\text{ext}} + \rho_{\text{ind}}$。**介电函数**定义为外加势与总势之比：

$$\varepsilon(\mathbf{q},\omega) = \frac{V_{\text{ext}}(\mathbf{q},\omega)}{V_{\text{tot}}(\mathbf{q},\omega)} = 1 - V(\mathbf{q})\,\chi(\mathbf{q},\omega)$$

其中 $\chi(\mathbf{q},\omega)$ 是**密度-密度响应函数（极化率）**，$V(\mathbf{q}) = 4\pi e^2/q^2$ 是裸库仑势。**有效相互作用**因此被屏蔽为：

$$V_{\text{eff}}(\mathbf{q},\omega) = \frac{V(\mathbf{q})}{\varepsilon(\mathbf{q},\omega)}$$

**重点：屏蔽的物理 = 一个电子感受到的相互作用不是裸库仑势，而是被 $\varepsilon$ 削弱的有效势。** 在静态极限（$\omega\to0$）与长波极限（$\mathbf{q}\to0$），$\varepsilon$ 通常远大于 1，库仑长程尾巴被切成短程——这就是金属里电子能「近似自由」的原因之一。<span class="marginnote">介电函数有两条等效的理解路径：线性响应里它来自密度-密度关联 $\chi = -i\langle[\rho,\rho]\rangle$（Kubo 公式，本专题前篇）；路径积分里它来自把库仑作用做 Hubbard-Stratonovich 变换后积掉密度涨落。两条路殊途同归，都是 $\varepsilon = 1 - V\chi$。</span>

## 2 Lindhard 函数：单圈极化

最低阶的极化来自一个**粒子-空穴对激发**，对应图中一个电子圈（一个「泡」）。这个单圈极化率叫 **Lindhard 函数**：

$$\chi_0(\mathbf{q},i\omega_n) = \frac{2}{\beta V}\sum_{\mathbf{k},n} G_0(\mathbf{k},i\omega_n)\, G_0(\mathbf{k}+\mathbf{q}, i\omega_n+i\nu_m)$$

这正是在《有限温度 Green 函数与 Matsubara 求和》一篇里我们亲手算过的松原求和！它的解析延拓结果（实频率）为：

$$\chi_0(\mathbf{q},\omega) = \int \frac{d^3k}{(2\pi)^3}\frac{n_F(\xi_{\mathbf{k}})-n_F(\xi_{\mathbf{k}+\mathbf{q}})}{\omega - (\xi_{\mathbf{k}+\mathbf{q}}-\xi_{\mathbf{k}}) + i\eta}$$

**重点：Lindhard 函数度量「体系以多大的效率把外加电荷转变成粒子-空穴对激发」。** 它的大小由两项决定：分子上的费米函数差 $n_F(\xi_{\mathbf{k}})-n_F(\xi_{\mathbf{k}+\mathbf{q}})$（泡利原理允许的激发相空间）与分母上的能量差（激发的能量代价）。它同时包含**静态极限**（$\omega=0$，屏蔽）与**动态结构**（$\omega\neq0$，激发与吸收）。<span class="marginnote">Lindhard 函数是凝聚态里最重要的单圈对象之一，它同时出现在介电函数、声子自能、超导 Eliashberg 函数、磁化率等几乎所有输运与响应的计算里。会算 Lindhard 函数，就等于会算半部多体理论。</span>

## 3 RPA：把泡串起来

单个泡只描述了「独立粒子-空穴对」的激发。真实体系里，感应密度本身又会引发新的极化——泡可以**串成链**。**随机相位近似（RPA）**假设：所有高阶极化图都可以近似为「泡与泡通过库仑线首尾相接」的无穷链（即只保留不含交叉的泡泡链）：

$$\chi_{\text{RPA}}(\mathbf{q},\omega) = \frac{\chi_0(\mathbf{q},\omega)}{1 - V(\mathbf{q})\,\chi_0(\mathbf{q},\omega)}$$

这是又一条**几何级数**（与 Dyson 方程的求和同理）。代回介电函数定义：

$$\varepsilon_{\text{RPA}}(\mathbf{q},\omega) = 1 - V(\mathbf{q})\,\chi_0(\mathbf{q},\omega)$$

**重点：RPA 介电函数 = 裸库仑势的几何级数求和。** 名字「随机相位」来自 Bohm-Pines 的历史：当系统里大量粒子-空穴对的相位随机时，只有「同相位」的集体贡献（即泡链）保留下来，交叉图被忽略。RPA 在弱耦合与高密度极限（$r_s\ll1$）下是**系统性的首阶修正**，也是电子气关联能（Gell-Mann–Brueckner）与等离子体激元理论的基础。<span class="marginnote">RPA 的历史：Bohm 与 Pines 1953 年用正则变换处理电子集体运动，提出「等离子体激元 + 屏蔽」的图像；随后 Gell-Mann 与 Brueckner 1957 年证明 RPA 恰好求和了高密度下最重要的发散图。RPA 名字里的「随机相位」即源于此——它常被中文译为「无规相近似」。</span>

## 4 三种屏蔽极限

RPA 介电函数在三个极限下给出不同物理：

**Thomas-Fermi 屏蔽（静态、长波 $\mathbf{q}\to0$）**：$\omega=0$、$q\to0$ 时，$\varepsilon(q,0) \approx 1 + q_{TF}^2/q^2$，其中 **Thomas-Fermi 波矢**：

$$q_{TF}^2 = 4\pi e^2 N(E_F) = \frac{4}{\pi}\frac{k_F}{a_B}$$

于是有效势 $V_{\text{eff}}(q) = 4\pi e^2/(q^2+q_{TF}^2)$，Fourier 变换回实空间得到 **Yukawa 型屏蔽势**：

$$V_{\text{eff}}(r) = \frac{e^2}{r}\,e^{-q_{TF}r}$$

**重点：长程库仑 $1/r$ 被削成短程 Yukawa 势，屏蔽长度 $\lambda_{TF} = 1/q_{TF}$。** 这就是为什么杂质在金属里的散射截面有限、电子气能形成正常金属——裸库仑的发散被屏蔽后一切变得可控。<span class="marginnote">Thomas-Fermi 屏蔽的物理很朴素：静态外场下，电子只需「重新分布」就能把外加电场挡在屏蔽长度之外——这正是金属是「导体」的本质：内部的电场被自由电子的重排完全抵消。</span>

**Friedel 振荡（静态、有限 $q$）**：$q\sim 2k_F$ 处，Lindhard 函数有奇性（费米面处的锐利边界），导致屏蔽势在实空间长距离衰减为振荡尾巴：

$$V_{\text{eff}}(r) \sim \frac{\cos(2k_F r)}{r^3}$$

这是费米面「刚性」的直接后果——电子气无法完全屏蔽 $2k_F$ 尺度的扰动，留下 RKKY 型振荡。它在磁性（间接交换作用）与扫描隧道显微镜的 Friedel 振荡成像里都可观测。

**动态屏蔽与等离子体激元（$\omega\neq0$）**：介电函数的零点 $\varepsilon(\mathbf{q},\omega)=0$ 对应集体振荡——**等离子体激元（plasmon）**，这是下一节的专门主题。

**辨析｜易错点：** 初学者常把 Thomas-Fermi 屏蔽当成唯一屏蔽，忘了两件事：其一，**屏蔽是动量依赖的**——对 $q\ll q_{TF}$ 的分量屏蔽极强，对 $q\gg q_{TF}$ 的分量几乎不屏蔽；其二，**动态屏蔽与静态屏蔽完全不同**——高频外场下电子来不及重排，$\varepsilon$ 趋于 1，电子又「暴露」出裸库仑相互作用（这正是超导配对需要的！）。「屏蔽」不是一个数，而是一个函数 $\varepsilon(\mathbf{q},\omega)$。

## 5 公式解析：Lindhard 静态极限

把 Lindhard 函数在零温静态极限（$\omega=0, T=0$）算出来，得到经典结果：

$$\chi_0(\mathbf{q},0) = -N(E_F)\,g\Big(\frac{q}{2k_F}\Big)$$

其中 $N(E_F) = mk_F/\pi^2\hbar^2$ 是费米面态密度，形状因子为：

$$g(x) = \frac{1}{2}\Big[1 + \frac{1-x^2}{2x}\ln\Big|\frac{1+x}{1-x}\Big|\Big]$$

- **第一步，零温费米函数**：$n_F(\xi_{\mathbf{k}}) = \theta(k_F-k)$，分子变成「动量在费米面内」的指示函数，把积分限制在 $k<k_F$ 的球内。
- **第二步，角度积分**：对 $k$ 方向积分后，能量差 $\xi_{\mathbf{k}+\mathbf{q}}-\xi_{\mathbf{k}} \approx \hbar^2\mathbf{q}\cdot\mathbf{k}/m$，积出对数结构——对数奇性来自费米面边界 $|\mathbf{k}|=k_F$ 的锐利。
- **第三步，读形状**：$x\to0$ 时 $g(0)=1$，$\chi_0 = -N(E_F)$（Thomas-Fermi 结果）；$x\to\infty$ 时 $g\to 1/(3x^2)$，$\chi_0\to 0$（短波长下电子气几乎不响应）；$x=1$ 处 $g(x)$ 有对数奇性（Friedel 振荡的根源）。

**重点：整个 Lindhard 函数的形状由费米面这一个对象决定。** 费米面有多锐利，响应的结构就有多丰富——这是「费米面是金属物理的心脏」这句话的数学体现。磁化率、声子软化、超导配对都从这条曲线读出。

## 6 屏蔽与「从极限到大模型」

屏蔽的思想——**「单个对象感受到的相互作用，被它所在的环境集体重正化」**——是「从极限到大模型」里反复出现的自组织原理。在大模型训练里，「梯度屏蔽」与「归一化层」做的事本质上就是调节每个参数的「有效学习率」；batch normalization 在训练集上的统计重正化，几乎就是「用环境均值修正个体」的工程版屏蔽。<span class="marginnote">更贴切的对应在<strong>对抗训练与注意力机制</strong>：注意力权重是 token 之间「有效相互作用」的屏蔽函数——相关度高的 token 对的相互作用被放大，不相关的被压低。Transformer 的 softmax 就是一个「非线性介电函数」，把原始的内积势「屏蔽」成归一化的有效权重。想深入见第四级《大模型原理》。</span>

对多体理论自身，介电函数是理解金属、半导体、等离子体与超导的枢纽：下一节我们将专门研究 $\varepsilon(\mathbf{q},\omega)$ 的零点——**等离子体激元与集体激发**——看看电子的集体运动如何以量子的方式改变整个体系。

## 7 小结

- **介电函数** $\varepsilon(\mathbf{q},\omega) = 1 - V(\mathbf{q})\chi(\mathbf{q},\omega)$ 度量电子气对外场的屏蔽能力；有效势 $= V/\varepsilon$。
- **Lindhard 函数** $\chi_0$ 是单圈极化，度量粒子-空穴对的激发效率；它的计算即前篇的松原求和。
- **RPA** 把泡以库仑线串成几何级数：$\chi_{\text{RPA}} = \chi_0/(1-V\chi_0)$，对应高密度极限的系统性首阶修正。
- 三种极限：**Thomas-Fermi 屏蔽**（$q\to0$，Yukawa 势）、**Friedel 振荡**（$q\sim2k_F$）、**动态屏蔽**（高频不屏蔽）。
- Lindhard 函数的形状由**费米面**决定：静态极限 $\chi_0(0) = -N(E_F)g(q/2k_F)$，对数奇性给出 Friedel 振荡。
- 屏蔽是动量与频率依赖的函数，不是常数；高频下电子暴露裸相互作用（超导配对的前提）。

在下一节，我们将研究介电函数的零点——**等离子体激元与集体激发**——电子气的集体振荡如何成为体系的基本激发模式，以及这背后「元激发」概念如何统一描述声子、磁振子与等离激元。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| 介电函数 | $\varepsilon = 1 - V\chi$ | 屏蔽能力 |
| 有效势 | $V_{\text{eff}} = V/\varepsilon$ | 裸库仑被削弱 |
| Lindhard 函数 | $\chi_0 = \int\frac{n_F(\xi)-n_F(\xi')}{\omega-(\xi'-\xi)}$ | 单圈极化，粒子-空穴对 |
| RPA | $\chi_{\text{RPA}} = \chi_0/(1-V\chi_0)$ | 泡链几何级数求和 |
| Thomas-Fermi | $V_{\text{eff}}(r) = \frac{e^2}{r}e^{-q_{TF}r}$ | 长程库仑变短程 Yukawa |
| Friedel 振荡 | $V_{\text{eff}} \sim \cos(2k_Fr)/r^3$ | 费米面锐利性的尾巴 |

**易错复盘**：两点要盯住。其一，屏蔽是动量与频率依赖的函数，不是常数——高频外场下电子来不及重排，$\varepsilon\to1$，电子又暴露裸库仑相互作用（超导配对的前提）；其二，静态屏蔽（Thomas-Fermi）与动态屏蔽完全不同——$\omega\to0$ 强屏蔽、$\omega\to\infty$ 不屏蔽。

**知识连线**：Lindhard 函数是第 1 篇松原求和的直接产物；RPA 的几何级数与第 2 篇 Dyson 方程同构；等离子体激元（第 2 篇）正是介电函数零点。「环境集体重正化个体相互作用」与注意力机制的 softmax 屏蔽、batch norm 的统计重正化同构——本篇与「从极限到大模型」的连接。

**实践与辨析**：为什么金属能屏蔽静电场而真空不能？提示：自由电子重排把内部电场抵消，$q_{TF}^2\neq0$。为什么 $2k_F$ 处的扰动无法被屏蔽？提示：费米面的刚性边界导致 Lindhard 函数在 $q=2k_F$ 有奇性，留下振荡尾巴。易错提醒：RPA 是高密度极限（$r_s\ll1$