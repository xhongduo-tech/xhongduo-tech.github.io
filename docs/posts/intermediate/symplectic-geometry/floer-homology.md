---
title: Floer 同调与 Arnol'd 猜想
date: 2026-08-07
---

# Floer 同调与 Arnol'd 猜想

<div class="epigraph">
<p>Floer 同调是辛几何的量子化：把曲线的计数变成同调群，把周期轨道变成链复形。</p>
<footer>—— 安德烈亚斯 · 弗洛尔（Andreas Floer, 1988）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从 Floer 同调开始

哈密顿系统的一个古老问题是：**一个哈密顿同胚至少有多少个不动点？** Arnol'd 猜想给出一个「拓扑下界」：不动点数不少于流形贝蒂数之和。这个问题在一般流形上悬置了数十年，直到 **Floer 同调**（Floer 1988）把它解决。Floer 的核心洞见是：把哈密顿同胚的不动点看成**无穷维空间（环路空间）上的 Morse 理论**——作用泛函的临界点恰是周期轨道，梯度流线的计数定义了一个新的同调群，**这个同调群同构于流形的普通同调**，于是「数临界点」变成「算贝蒂数」。这一篇介绍作用泛函、Floer 方程、链复形，以及 Arnol'd 猜想的证明逻辑。<span class="marginnote">在课程地图上：Floer 同调是第3篇伪全纯曲线理论的「全局化」——它把「数曲线」升级成「定义同调」。下一篇 Lagrangian Floer 同调、Gromov-Witten 不变量、谱不变量全是它的子孙。</span>

## 1 Arnol'd 猜想

**Arnol'd 猜想**（1965）：设 $(M, \omega)$ 是紧辛流形，$\phi \in \mathrm{Ham}(M,\omega)$ 是哈密顿同胚。若所有不动点是**非退化的**（$d\phi_p$ 无单位特征值），则

$$
\#\mathrm{Fix}(\phi) \ge \sum_{k} \dim H^k(M; \mathbb{F}_2)
$$

若不动点退化，用「临界点计数」替换：$\#\mathrm{crit} \ge \sum \dim H^k$。<span class="marginnote">类比 Morse 理论：紧流形上的每个非退化函数至少有「贝蒂数和」个临界点（Morse 不等式）。Arnol'd 猜想的哲学是：<strong>哈密顿同胚的不动点就是「无穷维流形上的函数」的临界点</strong>，所以 Morse 理论应适用——Floer 正是把它做成了。</span>

**例**：$M = T^{2n}$（环面），$\sum \dim H^k = 2^{2n}$，猜想说每个哈密顿同胚至少有 $2^{2n}$ 个不动点——这是「每一维都该有贡献」的直观。

## 2 作用泛函与周期轨道

考虑哈密顿量 $H$（周期 1 的含时函数），其流 $\phi_H^t$。**1-周期轨道**是满足 $\gamma(t+1) = \gamma(t)$ 的轨道，即 $\phi_H^1$ 的不动点经过的点。

取环路空间

$$
\mathcal{L}M = \{ \gamma: S^1 \to M : \gamma \text{ 光滑} \}
$$

**作用泛函（action functional）**：

$$
\mathcal{A}_H(\gamma) = -\int_{D^2} \bar\gamma^*\omega + \int_0^1 H(t, \gamma(t)) dt
$$

其中 $\bar\gamma: D^2 \to M$ 是 $\gamma$ 的延拓（需 $[\omega]$ 在 $\pi_2$ 上消失或引入 Novikov 环）。<span class="marginnote">第一项「辛面积」需要延拓 $\bar\gamma$ 才良定义；不同延拓差一个「球面面积」$\int_{S^2}\omega$，用 Novikov 环（形式幂级数）处理。对单调辛流形或 Calabi-Yau 流形，这个障碍消失或可控——这是 Floer 理论的技术前提。</span>

**Floer 的关键观察**：$\mathcal{A}_H$ 的**临界点**恰是 $H$ 的 1-周期轨道（此时 $d\mathcal{A}_H(\gamma) = 0 \iff \dot\gamma = X_H(\gamma)$）。所以「数不动点」=「数 $\mathcal{A}_H$ 的临界点」——Morse 理论的设定齐了。但 $\mathcal{L}M$ 是无穷维的，普通 Morse 理论不够，需要 Floer 的「无穷维 Morse 理论」。

## 3 Floer 方程与链复形

**Floer 方程（梯度流线方程）**：$\mathcal{A}_H$ 关于某个 $L^2$ 度量的「梯度流线」满足

$$
\partial_s u + J(u)\big( \partial_t u - X_H(u) \big) = 0
$$

其中 $u: \mathbb{R} \times S^1 \to M$，$(s, t)$ 分别是时间与圈参数。<span class="marginnote">这是「伪全纯曲线方程」的含时版本：$X_H = 0$ 时退化为 $\bar\partial u = 0$。Floer 的伟大简化在于证明了「Floer 方程的解空间在有界能量下紧致（无冒泡或冒泡可控）」——这是 Gromov 紧致性在 $S^1$-参数化的移植。</span>

**Floer 链复形**：

$$
CF_*^H = \bigoplus_{\gamma \in \mathrm{Per}_1(H)} \mathbb{Z}_2 \langle \gamma \rangle
$$

生成元是 $H$ 的 1-周期轨道（非退化），系数取 $\mathbb{Z}_2$ 回避定向。**Floer 微分** $\partial$ 计数「连接轨迹」：

$$
\partial \gamma = \sum_{\gamma'} n(\gamma, \gamma') \gamma'
$$

$n(\gamma, \gamma')$ 是「从 $\gamma$ 到 $\gamma'$ 的、指标差 1 的 Floer 方程解的模数（模平移，模 2 计数）」。**Floer 定理**：$\partial^2 = 0$（紧致性 + 模空间光滑性），故有同调

$$
HF_*^H = \ker\partial / \operatorname{im}\partial
$$

**谱不变量（spectral invariant）**：$\mathcal{A}_H$ 的临界值给出 $HF$ 上的滤波与谱不变量 $c_\sigma(H)$——下一篇的伏笔。

## 4 公式解析：作用泛函的临界点

**核心公式（临界点 ⟺ 周期轨道）：**

$$
d\mathcal{A}_H(\gamma) = 0 \iff \dot\gamma(t) = X_H(t, \gamma(t))
$$

拆解：

- **第一步，取变分**：设 $\gamma_\varepsilon$ 是 $\gamma$ 的变分（$\frac{d}{d\varepsilon}\gamma_\varepsilon = V$，$V$ 是沿环路的向量场），计算 $\frac{d}{d\varepsilon}\mathcal{A}_H(\gamma_\varepsilon)$。面积项给出 $\int \omega(\dot\gamma, V)dt$ 型项（用 Stokes），哈密顿项给出 $\int dH(V)dt$。
- **第二步，合并**：$d\mathcal{A}_H(V) = \int_0^1 \big( \omega(\dot\gamma, V) + dH(V) \big) dt = \int_0^1 \omega\big( \dot\gamma - X_H, V \big) dt$。这里用了 $\omega(X_H, V) = dH(V)$（哈密顿向量场定义）。
- **第三步，判零**：$d\mathcal{A}_H = 0$ 对所有 $V$ ⟺ $\omega(\dot\gamma - X_H, V) = 0$ 对所有 $V$ ⟺（$\omega$ 非退化）$\dot\gamma = X_H$。
- **第四步，结论**：**临界点正是哈密顿方程的解，即周期轨道**。非退化不动点对应非退化临界点（Hessian 可逆），Morse 指标 = 轨道指数（Conley-Zehnder）。

**直觉总结：** 作用泛函把「解微分方程（周期轨道）」重写成「找泛函临界点」。这听起来只是改写，却让 Morse 理论的整个机器（临界点计数、梯度流、同调）可以上场——**Floer 同调 = 无穷维 Morse 同调**。

## 5 证明 Arnol'd 猜想

Floer 证明 Arnol'd 猜想的逻辑是四条腿：

1. **Floer 同调有限维生成**：$HF_*^H$ 由周期轨道生成，所以「生成元数 = 不动点数」的下界来自「$HF$ 的贝蒂数 ≤ 生成元数」（Morse 不等式）。
2. **$HF$ 同构于普通同调**：Floer 证明（对单调流形或 $H^1 = 0$ 等情形）$HF_*^H \cong H_*(M; \mathbb{Z}_2)$。**同调群不依赖 $H$**——它只依赖 $M$。
3. **同伦不变**：Floer 同调在同痕下不变（Floer 同构 / continuation map），所以不依赖具体 $H$。
4. **合并**：$\#\mathrm{Fix}(\phi) \ge \sum \dim HF_k = \sum \dim H_k(M)$。**Arnol'd 猜想得证。**<span class="marginnote">关键一步是「$HF \cong H_*(M)$」：选一个「极小的」哈密顿量（如非常小的扰动），其周期轨道恰是常值轨道（对应 $M$ 的点），Floer 复形退化成一个方向导数的复形——恰好是 Morse 复形（对扰动函数的梯度流）。于是 $HF \cong H_*(M)$ 是「同调不变性 + 特殊哈密顿量计算」的组合。</span>

**后续推广**：Arnol'd 猜想对更一般流形由不同机制证明（monotone 用 Floer，一般情形用 Piunikhin-Salamon-Schwarz 同构、或利用量子同调与谱不变量）。它成为「动力系统下界」与「拓扑不变量」相互作用的模范。

**辨析｜易错点：** Floer 同调**依赖选择**（$H$、$J$、系数、延拓），但这些选择给出**同构**的同调群——同调类不依赖，链复形依赖。初学者常把「链复形」与「同调群」混为一谈：**链复形与 $H$、$J$ 有关，同调群只与 $M$ 有关**（在同伦范畴内）。另外「不动点计数」只在非退化情形是「数点」，退化情形用「临界点计数」（小扰动后的 Morse 计数）。

## 6 小结

- **Arnol'd 猜想**：哈密顿同胚不动点下界 = 贝蒂数之和；等价于「无穷维 Morse 不等式」。
- **作用泛函** $\mathcal{A}_H$：环路空间上的函数，**临界点 = 周期轨道**。
- **Floer 方程**：$\partial_s u + J(\partial_t u - X_H) = 0$，是伪全纯曲线方程的含时版。
- **Floer 链复形**：周期轨道生成，微分数连接轨迹；**Floer 同调** = 无穷维 Morse 同调。
- **$HF \cong H_*(M)$**：极小哈密顿量退化到普通 Morse 复形——Arnol'd 猜想由此得证。
- **选择无关性**：链复形依赖 $H, J$ 与延拓，但同调群只依赖 $M$ 的辛形变类——Floer 同调是真正的不变量。
- **谱不变量伏笔**：$\mathcal{A}_H$ 的临界值给 $HF$ 加滤波，产生谱不变量 $c_\sigma(H)$——下一篇《哈密顿流与谱不变量》的主角。

在下一节，我们将顺着「临界值 = 谱」这条线索，研究**哈密顿流与谱不变量**：把 Floer 同调的滤波信息提炼成一族数值不变量，用它刻画哈密顿同胚的能量与辛嵌入的刚性。Floer 同调在这里从「数不动点」升级为「给能量定标尺」。