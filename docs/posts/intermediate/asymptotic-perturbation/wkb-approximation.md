---
title: WKB 近似及其在量子力学中的应用
date: 2026-08-07
---

# WKB 近似及其在量子力学中的应用

<div class="epigraph">
<p>波在非均匀介质中穿行时，仿佛每一小段都把自己当成平面波。</p>
<footer>—— 改写自 Carl M. Bender & Steven A. Orszag 对 WKB 思想的评述</footer>
</div>

<div class="article-byline">
<p>第二级 · 渐近分析与摄动方法 ｜ Bender &amp; Orszag §10.1–10.4 ｜ 2026-08-07</p>
</div>

## 为什么从 WKB 近似开始

整个专题的摄动工具即将在**波动方程**上会师。当波（声波、电磁波、量子概率波）在**非均匀介质**中传播，介质的性质随空间缓变时，严格求解几乎不可能，但有一个极其有效的近似：把波写成「**局部平面波**」——相位随路径积分累积，振幅随介质的「阻抗」变化。这就是 **WKB 近似**（Wentzel–Kramers–Brillouin，1926）。在量子力学里，它就是 $\hbar\to0$ 的半经典近似，连接了薛定谔方程与经典力学。<span class="marginnote">WKB 独立发现者包括 Jeffreys、Wentzel、Kramers、Brillouin；它早期也被称为 JWKB。它的数学本质是：$\varepsilon^2 y'' + Q(x)y = 0$ 型方程在 $\varepsilon\to0$ 时的渐近解——正是奇异摄动的地盘。</span>

## 1 从薛定谔方程到 WKB 方程

一维定态薛定谔方程（第二级《量子力学》的常客）：

$$
-\frac{\hbar^2}{2m}\,\psi''(x) + V(x)\,\psi(x) = E\,\psi(x)
$$

改写为

$$
\varepsilon^2\,\psi''(x) + p(x)^2\,\psi(x) = 0, \qquad
\varepsilon = \hbar, \qquad
p(x) = \sqrt{2m\,[E - V(x)]}
$$

这里 $p(x)$ 是**局部动量**（经典动量的位置依赖版本）。$\varepsilon=\hbar$ 扮演小参数——半经典极限就是 $\hbar\to0$。<span class="marginnote">$\varepsilon^2$ 乘在最高阶导数上，这是标准的奇异摄动形态。$E>V(x)$ 处 $p$ 实（振荡区，经典可达）；$E<V(x)$ 处 $p$ 虚（指数区，经典禁区）。正负转折点 $p=0$ 把两个区域分开。</span>

**WKB 的 ansatz**：把 $\psi$ 写成指数形式，相位按 $\varepsilon$ 展开：

$$
\psi(x) = \exp\left[\frac{i}{\varepsilon}\left(S_0(x) + \varepsilon S_1(x) + \varepsilon^2 S_2(x) + \cdots\right)\right]
$$

代入方程，按 $\varepsilon$ 的幂次匹配。

## 2 逐阶求解：相位与振幅

把 ansatz 代入 $\varepsilon^2\psi'' + p^2\psi = 0$，需要计算 $\psi''$。记 $\Phi = S_0 + \varepsilon S_1 + \cdots$，$\psi = e^{i\Phi/\varepsilon}$：

$$
\psi' = \frac{i}{\varepsilon}\Phi'\,\psi, \qquad
\psi'' = \left(\frac{i}{\varepsilon}\Phi'' - \frac{1}{\varepsilon^2}(\Phi')^2\right)\psi
$$

于是方程除以 $\psi$ 后变为

$$
\varepsilon^2\left[\frac{i}{\varepsilon}\Phi'' - \frac{1}{\varepsilon^2}(\Phi')^2\right] + p^2
= i\varepsilon\,\Phi'' - (\Phi')^2 + p^2 = 0
$$

展开 $\Phi = S_0 + \varepsilon S_1 + \varepsilon^2 S_2 + \cdots$：

- **$O(1)$**：$(S_0')^2 = p^2 \Rightarrow S_0 = \pm\int^x p(s)\,ds$。**相位是动量的积分**——这就是「局部平面波」的相位。
- **$O(\varepsilon)$**：$i S_0'' - 2S_0' S_1' = 0 \Rightarrow S_1 = \frac{i}{2}\ln|S_0'| + \text{const}$。**振幅由相位的一阶导数决定**——即振幅 $\sim 1/\sqrt{p}$。

于是零阶 WKB 近似为

$$
\psi_{\text{WKB}}(x) \sim \frac{1}{\sqrt{p(x)}}\left[C_+ e^{\frac{i}{\hbar}\int^x p\,ds} + C_- e^{-\frac{i}{\hbar}\int^x p\,ds}\right]
$$

**在经典可达区（$E>V$）**，这是两个反向行波的叠加；**在经典禁区（$E<V$）**，$p = i\kappa$，指数变为

$$
\psi \sim \frac{1}{\sqrt{\kappa(x)}}\left[A\, e^{-\frac{1}{\hbar}\int^x \kappa\,ds} + B\, e^{+\frac{1}{\hbar}\int^x \kappa\,ds}\right]
$$

一个指数衰减、一个指数增长。<span class="marginnote">$1/\sqrt{p}$ 正是「振幅随介质阻抗变化」的数学形式：粒子越慢（$p$ 小），波函数越集中在那个区域——与概率密度的物理直觉吻合。这不只是数学技巧，它有守恒流 $\sim |\psi|^2 p = $ 常数的物理内容。</span>

## 3 转折点与连接公式

$p(x)=0$ 的**转折点（turning point）** 处，$1/\sqrt{p}$ 发散，WKB 解失效。物理上，经典粒子在转折点**折返**，量子粒子则**隧穿**。要跨过转折点，需在转折点附近解精确的模型方程。

设转折点在 $x=a$，$V(x)-E \approx V'(a)(x-a)$ 线性近似，则局部动量 $p^2 \propto (x-a)$。方程变成

$$
\frac{d^2\psi}{dz^2} - z\,\psi = 0, \qquad z = \left[\frac{2m|V'(a)|}{\hbar^2}\right]^{1/3}(x-a)
$$

这是 **Airy 方程** $Ai'' - z\,Ai = 0$。<span class="marginnote">Airy 函数又出场了——第 5 篇驻相法里它是焦散的主角，这里是转折点的主角。同一个特殊函数，在波的「合并奇点」处反复出现，因为它是最简单的「二阶导 = 线性势」方程。</span>

Airy 函数把 WKB 的两段接起来。对**线性势转折点**，从禁区到可达区（$x<a$ 禁区，$x>a$ 可达区）的**连接公式（connection formula）** 为

$$
\frac{1}{2\sqrt{\kappa}}\,e^{-\frac{1}{\hbar}\int_x^a \kappa\,ds}
\;\longrightarrow\;
\frac{1}{\sqrt{p}}\,\cos\left(\frac{1}{\hbar}\int_a^x p\,ds - \frac{\pi}{4}\right)
$$

**禁区衰减的指数，在可达区变成相位为 $-\pi/4$ 的余弦**。这个 $\pi/4$ 与第 5 篇驻相法的 $e^{\pm i\pi/4}$ 遥相呼应——它们来自同一个高斯积分的相位因子。<span class="marginnote">连接公式的符号选取（$+\pi/4$ 还是 $-\pi/4$）是 WKB 应用里最容易错的地方：从左（禁）到右（可）跨转需 $-\pi/4$，反向则用 $+\pi/4$。$Ai$ 函数的渐近式 $\mathrm{Ai}(z)\sim \pi^{-1/2}z^{-1/4}\cos(\frac23 z^{3/2}-\frac\pi4)$ 是最可靠的锚点。</span>

## 4 公式解析：束缚态量子化条件

WKB 最有名的应用是给出**束缚态能量量子化**。把粒子关在势阱 $V(x)$ 里，两侧各有一个转折点 $a$（左）与 $b$（右）。连接公式要求：从 $a$ 传出的余弦在 $b$ 处反射后，回到 $a$ 时相位自洽。

经典力学里，一个周期的相空间面积为 $\oint p\,dx = \int_a^b p\,dx \times 2$。量子化条件给出

$$
\oint p(x)\,dx = \left(n + \frac{1}{2}\right)\,2\pi\hbar, \qquad n = 0,1,2,\dots
$$

即

$$
\int_a^b \sqrt{2m\,[E_n - V(x)]}\; dx = \left(n + \frac{1}{2}\right)\pi\hbar
$$

- **第一步，相位自洽**：波从 $a$ 传到 $b$，相位累积 $\frac{1}{\hbar}\int_a^b p\,dx$；每次转折点反射引入 $-\pi/2$ 相位（来自 $-\pi/4$ 的两端）。往返一周总相位增量 $= 2\times\frac{1}{\hbar}\int_a^b p\,dx - \pi$，必须等于 $2\pi n$ 的整数倍。
- **第二步，解出 $E_n$**：对每个 $n$，上式是关于 $E_n$ 的方程。对谐振子 $V=\frac12 m\omega^2 x^2$，积分给出 $E_n = (n+\frac12)\hbar\omega$——**与精确解完全一致**！
- **第三步，看 $+\frac12$ 的来历**：它来自两个转折点各自的 $-\pi/4$ 相位，物理上就是量子零点能。<span class="marginnote">对谐振子 WKB 恰好精确，因为它的转折点之间是线性势、Airy 近似零误差。对一般势阱，$E_n = (n+\frac12)\hbar\omega$ 型公式给出首阶，更高阶修正靠连接公式的高阶项。这是半经典量子化的统一框架，比玻尔的圆轨道量子化深刻得多。</span>

## 5 辨析｜易错点：WKB 的边界

- **转折点处直接套公式**：$1/\sqrt{p}$ 在转折点发散，WKB 解在转折点 $O(\hbar^{1/3})$ 邻域内**无效**。要跨转折点必须用 Airy 匹配，不能硬代。
- **禁区里保留增长项**：禁区通解含 $e^{+}$ 与 $e^{-}$ 两支。无界势阱内必须**丢弃增长支**；两侧都有关键修正，丢弃错误的一支会让量子化条件差出 $\pi/4$。
- **连接公式方向搞反**：$-\pi/4$ 与 $+\pi/4$ 取决于跨越方向。拿不准就回到 Airy 函数渐近式核对。
- **二阶 WKB 就够？**：$S_0,S_1$ 只是零阶；势变化剧烈或 $\hbar$ 不够小时，需 $S_2$ 项（含 $p''$）或干脆数值求解。**先检查 $|\hbar p'/p^2|\ll 1$**（绝热条件）。
- **多个转折点**：双阱、势垒隧穿等要逐段连接，每段一个 Airy 匹配，公式链变长——但思路不变。

## 6 小结

- **WKB**：$\psi = e^{\frac{i}{\hbar}(S_0+\hbar S_1+\cdots)}$，$S_0=\pm\int p\,dx$ 给相位，$S_1=\frac i2\ln|S_0'|$ 给振幅 $1/\sqrt{p}$。
- 经典可达区是**行波叠加**，经典禁区是**指数衰减/增长**；**转折点** $p=0$ 处用 Airy 函数连接。
- **连接公式**：禁区指数 → 可达区 $\cos(\cdots-\pi/4)$，相位因子与驻相法同源。
- **束缚态量子化**：$\oint p\,dx = (n+\frac12)2\pi\hbar$