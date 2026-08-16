---
title: 零温格林函数（谱表示、Lehmann 表示、解析性质）
date: 2026-08-07
---

# 零温格林函数（谱表示、Lehmann 表示、解析性质）

<div class="epigraph">
<p>Green 函数把「粒子如何从一个时空点走到另一个」变成了一个可微扰展开的函数——整个多体理论就建立在这个转译上。</p>
<footer>—— 朱利安 · 施温格（Julian Schwinger）论传播子（转述）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics*, Ch. 2 ｜ 2026-08-07</p>
</div>

## 为什么从零温 Green 函数开始

前面第 1 篇给了我们二次量子化语言，第 2 篇把它搬进有限温度。但多数多体计算（微扰展开、自能、准粒子）最先是在零温基态上建立的，其核心对象就是**单粒子 Green 函数**。<span class="marginnote"><strong>为什么零温也值得单开一篇</strong>：零温公式没有 Matsubara 频率的离散化，实频率 $\omega$ 连续，谱结构（极点、割线）最直接地显现。理解了零温谱表示，有限温度只是把它离散化并加上占居因子。</span>本篇系统梳理零温 Green 函数的三个层面：**定义、谱表示（Lehmann）、解析性质**。三者合起来回答了「Green 函数到底是什么函数」——它是一根带着全部谱信息的复频率函数。

## 1 零温 Green 函数的定义

在 $T=0$ 基态 $|0\rangle$ 上定义**时序 Green 函数（time-ordered Green function）**：

$$G(\mathbf{k}, t) = -i\langle 0|\,T\,[c_{\mathbf{k}}(t)c_{\mathbf{k}}^\dagger(0)]\,|0\rangle$$

其中 $T$ 是时间排序算符：$t>0$ 时先 $c^\dagger$ 后 $c$（先产生粒子再传播），$t<0$ 时顺序相反。Heisenberg 绘景算符

$$c_{\mathbf{k}}(t) = e^{i\hat{H}t}c_{\mathbf{k}}e^{-i\hat{H}t}$$

**Green 函数回答的问题是**：「在 $t=0$ 时刻往基态里放入一个动量为 $\mathbf{k}$ 的粒子，到时刻 $t$ 它还在这个态的振幅是多少？」振幅大，说明该态是好的准粒子态；振幅随时间衰减，说明粒子与相互作用纠缠，寿命有限。<span class="marginnote"><strong>物理直觉</strong>：$G(\mathbf{k},t)$ 振幅 $e^{-i\varepsilon_{\mathbf{k}}t}$ 对应自由传播；衰减与能级移动则全部来自相互作用。可以说 Green 函数是「粒子传播的放大镜」。</span>

## 2 谱表示：把 Green 函数写成频率函数

对时间做 Fourier 变换 $G(\mathbf{k},\omega)=\int_{-\infty}^{\infty}dt\,e^{i\omega t}G(\mathbf{k},t)$，得到频率空间的 Green 函数。用一组完整本征态 $\{|\Psi_n\rangle\}$（含激发态）展开后，可以证明它取如下形式：

$$G(\mathbf{k},\omega) = \int_{-\infty}^{\infty} d\omega'\,\frac{A(\mathbf{k},\omega')}{\omega - \omega' + i\eta\,\mathrm{sgn}(\omega')}$$

其中 $\eta\to 0^+$，$A(\mathbf{k},\omega)$ 是**谱函数（spectral function）**。这就是**谱表示（spectral representation）**：它把任意 Green 函数还原成「一个加权积分」，权重就是谱函数。谱函数满足归一化

$$\int_{-\infty}^{\infty}\frac{d\omega}{2\pi}\,A(\mathbf{k},\omega) = 1$$

这一条要求保证了 Green 函数的高频渐近行为 $G \sim 1/\omega$，是检验近似计算是否自洽的硬标准。

## 3 Lehmann 表示：显式的谱函数

把基态与激发态的矩阵元写出来，就得到**Lehmann 表示（Lehmann representation）**的显式形式。谱函数

$$A(\mathbf{k},\omega) = \sum_n \Bigl[|\langle \Psi_n|c_{\mathbf{k}}^\dagger|0\rangle|^2\, 2\pi\delta(\omega - E_n + E_0) + |\langle \Psi_n|c_{\mathbf{k}}|0\rangle|^2\, 2\pi\delta(\omega + E_n - E_0)\Bigr]$$

两项分别对应「加入一个粒子」与「移走一个粒子」的激发谱：前者只可能发生在 $\omega>0$（$E_n>E_0$），后者在 $\omega<0$（移走粒子使系统能量升高）。<span class="marginnote"><strong>Lehmann 表示的妙处</strong>：谱函数 $A$ 直接就是角分辨光电子谱（ARPES）实验能测到的量——$\omega<0$ 部分是移走电子谱（角分辨光电子谱，ARPES），$\omega>0$ 部分是逆光电子谱（IPES）。Green 函数的全部「可测内容」都浓缩在谱函数里。</span>

**重点：谱函数 $A(\mathbf{k},\omega)$ 是 Green 函数的「实体」，频率分解是外壳。** 知道 $A$ 就知道 $G$，反之亦然；而 $A$ 与实验直接相关，这让 Green 函数从纯形式对象变成了可证伪的物理预言。

## 4 解析性质：推迟与超前 Green 函数

谱表示揭示了 Green 函数在复频率平面的结构。对自由费米气体，$A(\mathbf{k},\omega)=2\pi\delta(\omega-\varepsilon_{\mathbf{k}})$，代入谱表示：

$$G(\mathbf{k},\omega) = \frac{1}{\omega - \varepsilon_{\mathbf{k}} + i\eta\,\mathrm{sgn}(\varepsilon_{\mathbf{k}})}$$

当 $\varepsilon_{\mathbf{k}}>0$ 时极点在 $\omega=\varepsilon_{\mathbf{k}}-i\eta$（下半平面）；当 $\varepsilon_{\mathbf{k}}<0$ 时极点在 $\omega=\varepsilon_{\mathbf{k}}+i\eta$（上半平面）。于是：**Green 函数在复平面的极点是准粒子能量，割线是连续谱，解析延拓把时序函数与推迟/超前函数连接起来**。<span class="marginnote"><strong>推迟函数</strong>$G^R(\omega)$ 在 $\omega$ 上半平面解析，它的实部与虚部由 Kramers–Kronig 关系互相决定。解析性不是技术细节，而是「因果性」的体现：推迟函数描述响应必须在扰动之后，等价于它在上半平面无极点。</span>

## 5 公式解析：Lehmann 表示里发生了什么

把谱表示与 Lehmann 表示合起来读一遍，拆成三步：

$$
G(\mathbf{k},\omega) = \int_{-\infty}^{\infty} \frac{d\omega'}{2\pi}\,\frac{A(\mathbf{k},\omega')}{\omega-\omega'+i\eta\,\mathrm{sgn}(\omega')}
$$

- **第一步，识别 $A$ 是激发谱的密度**：Lehmann 表示把 $A$ 写成两串 $\delta$ 函数，一串在 $\omega>0$（加一个粒子，权重 $|\langle\Psi_n|c^\dagger|0\rangle|^2$），一串在 $\omega<0$（移走一个粒子，权重 $|\langle\Psi_n|c|0\rangle|^2$）。权重平方正是「基态到该激发态跃迁的振幅」。
- **第二步，理解 $i\eta\,\mathrm{sgn}(\omega')$ 的作用**：它把积分路径的极点半推入复平面，保证因果性与收敛。$\eta\to0^+$ 的记号表示极限顺序不能随意交换——这是解析延拓里最容易出错的地方。
- **第三步，读出可观测信息**：$A$ 的峰位给出准粒子色散 $\varepsilon_{\mathbf{k}}$，峰宽给出准粒子寿命/衰减率，峰的相对权重给出剩余权重（quasiparticle residue）$Z_{\mathbf{k}}$。$Z_{\mathbf{k}}\to0$ 意味着准粒子图像崩溃，这在强关联系统里是「非费米液体」的信号（本专题第 7 篇《费米液体理论》会回来用 $Z$ 说话）。

## 6 从 Green 函数到可观测量

Green 函数不是终点，它通过几个恒等式直接连到实验：

**态密度**：$N(\omega) = \int \frac{d^3k}{(2\pi)^3} A(\mathbf{k},\omega)$。谱函数对动量积分即局域态密度。

**基态能量**：$E_0 = -\frac{i}{2}\int \frac{d\omega}{2\pi}\sum_{\mathbf{k}} \left[1 + \frac{\partial}{\partial\omega}\ln G\right] G$ 类的关系把基态能量写成 Green 函数的积分，这是「从 Green 函数计算一切热力学量」的开端。

**动量分布**：$\langle n_{\mathbf{k}}\rangle = \int_{-\infty}^{0}\frac{d\omega}{2\pi} A(\mathbf{k},\omega)$。粒子数分布就是谱函数在负频区（移走粒子区）的积分——费米液体里这个量在 $k_F$ 处有陡峭跳变，跳变高度正是 $Z_{\mathbf{k}}$。

## 7 具体例子：一个相互作用气体的谱函数

把谱表示用在「自由气体加一个被猝灭的杂态」上，能看清权重转移的物理。设系统有两个单粒子态，分别能量 $\varepsilon_0$（原占据）与 $\varepsilon_1$（空），外加让两者混合的耦合 $V$。对角化后谱函数 $A(\omega)$ 由两个峰组成：

$$A(\omega) = 2\pi\Bigl[u^2\delta(\omega-E_+) + v^2\delta(\omega-E_-)\Bigr]$$

其中 $E_\pm$ 是混合后的本征能量，$u^2+v^2=1$ 是归一化。这个简单例子演示了 Lehmann 表示的所有要点：

- **权重守恒**：$u^2+v^2=1$ 正是谱函数归一化的体现——总权重不变，但峰的位置与高度被耦合 $V$ 重新分配。
- **峰位移动**：$E_\pm$ 相对裸能量 $\varepsilon_{0,1}$ 发生偏移，这就是「能级移动」（能量重整化）的最小模型。
- **权重转移**：如果把 $v^2$ 看作从主峰「借走」的权重，它对应的就是准粒子剩余权重 $Z=u^2<1$。$V$ 越大，$Z$ 越小，准粒子越「不像自由电子」。

把这个二能级模型外推到连续系统，就是费米液体里「谱权重从费米面附近转移到高能区」的图像——$Z_{\mathbf{k}}$ 变小、背景权重变大的全部骨架，早在这一页的小模型里就出现了。

## 8 小结

- 零温**时序 Green 函数** $G(\mathbf{k},t)=-i\langle0|Tc_\mathbf{k}(t)c_\mathbf{k}^\dagger|0\rangle$ 描写「放一个粒子、看它传播」的振幅。
- **谱表示**把 $G(\omega)$ 写成谱函数 $A$ 的加权积分，$A$ 满足 $\int d\omega A = 2\pi$ 的归一化。
- **Lehmann 表示**给出 $A$ 的显式：正频（加粒子）与负频（移粒子）两串 $\delta$ 峰，权重是跃迁振幅平方。
- **解析性质**：$G(\omega)$ 的极点是准粒子能量，$i\eta\,\mathrm{sgn}(\omega')$ 把极点推入复平面，推迟函数在上半平面解析，因果性内建其中。
- **谱函数 $A$ 直接对应 ARPES/IPES 实验**，是 Green 函数理论可证伪性的关键。
- 准粒子剩余权重 $Z_{\mathbf{k}}$ 由 $A$ 的峰结构读出，$Z\to0$ 预告准粒子图像的崩溃。

## 9 公式速查：一页纸复习

| 对象 | 表达式 | 一句话要点 |
| --- | --- | --- |
| 时序 Green | $-i\langle0\|Tc_\mathbf{k}(t)c_\mathbf{k}^\dagger\|0\rangle$ | 粒子传播振幅 |
| 谱表示 | $\int\frac{d\omega'}{2\pi}\frac{A}{\omega-\omega'+i\eta\,\mathrm{sgn}(\omega')}$ | $G$ 是 $A$ 的加权积分 |
| Lehmann | $A=\sum_n(\|\langle\Psi_n\|c^\dagger\|0\rangle\|^2 2\pi\delta(\cdots)+\cdots)$ | 加/移粒子两串峰 |
| 自由气体 | $G=1/(\omega-\varepsilon_\mathbf{k}+i\eta\,\mathrm{sgn})$ | 单极点在复平面 |
| 态密度 | $N(\omega)=\int\frac{d^3k}{(2\pi)^3}A$ | $A$ 对动量积分 |
| 动量分布 | $\langle n_\mathbf{k}\rangle=\int_{-\infty}^0\frac{d\omega}{2\pi}A$ | 负频积分 |

**易错复盘**：一是 $i\eta\,\mathrm{sgn}(\omega')$ 里的 $\mathrm{sgn}$ 不能省，否则因果性错乱；二是谱函数归一化必须自洽验证；三是 $Z_{\mathbf{k}}$ 从 $\delta$ 峰相对权重读出，连续谱背景不算。这些都将在后续 Dyson 方程与自能计算里反复用到。

**知识连线**：本篇是第 1 篇《二次量子化》的直接延伸，为第 3 篇《微扰展开与 Feynman 图》、第 4 篇《Dyson 方程与自能》提供谱分析地基；它与本专题后续的费米液体、超导都共享同一套 Lehmann 语言。谱函数是「实验可测的信息密度」，这与大模型里「信息如何编码与提取」的问题在精神上相通——微观世界的可观测信息，总是藏在某个加权函数里。

在下一节，我们将把零温 Green 函数搬到虚时与离散频率：系统学习松原 Green 函数、频率求和与解析延拓。
