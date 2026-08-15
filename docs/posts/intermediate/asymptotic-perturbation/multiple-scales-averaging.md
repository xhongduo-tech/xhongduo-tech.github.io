---
title: 多重尺度方法与平均法
date: 2026-08-07
---

# 多重尺度方法与平均法

<div class="epigraph">
<p>缓慢的变化不是噪声，而是被快时间藏起来的慢时间。</p>
<footer>—— 改写自 Ali H. Nayfeh《Perturbation Methods》对多尺度法的评述</footer>
</div>

<div class="article-byline">
<p>第二级 · 渐近分析与摄动方法 ｜ Hinch §5–6 ｜ 2026-08-07</p>
</div>

## 为什么从多重尺度方法开始

第 6 篇的正则摄动在阻尼振子上栽了跟头：一阶解里冒出久期项 $t\sin t$，$t\sim 1/\varepsilon$ 时整个展开失效。真正的解 $e^{-\varepsilon t}\sin(\sqrt{1-\varepsilon^2}t)$ 告诉我们，阻尼带来的**振幅衰减**发生在慢时间尺度 $T=\varepsilon t$ 上，频率微移也发生在慢尺度上。**多重尺度方法（method of multiple scales）** 正是为慢变调制而生的：把快时间 $t$ 与慢时间 $T=\varepsilon t$ 当作两个**相互独立**的自变量，让振幅与相位在慢时间上慢慢演化。<span class="marginnote">直觉：钟摆的周期是快尺度 $O(1)$，而空气阻力让摆幅减半需要慢尺度 $O(1/\varepsilon)$。把两个尺度分开看，快尺度算振荡、慢尺度算包络，久期项自然被「摊平」到慢时间上。</span>

## 1 久期项的祸根：共振的数学面目

回顾阻尼振子 $\ddot{y} + 2\varepsilon\dot{y} + y = 0$。正则摄动展开 $y = y_0 + \varepsilon y_1 + \cdots$：

$$
y_0 = A\cos t + B\sin t, \qquad
\ddot{y}_1 + y_1 = -2\dot{y}_0 = 2A\sin t - 2B\cos t
$$

右端是齐次方程 $\ddot{y}_1 + y_1 = 0$ 的**解模态**（$\sin t, \cos t$），于是 $y_1$ 必含 $t\cos t$、$t\sin t$ 这类久期项。<span class="marginnote">久期项的判据：右端若是零阶解的模态，一阶解就含「时间 × 模态」。这是共振，不是巧合——激振频率正好等于固有频率。</span>

为什么久期项是灾难？因为 $\varepsilon y_1 \sim \varepsilon t$，当 $t = O(1/\varepsilon)$ 时它与 $y_0=O(1)$ 同量级，**展开的「小修正」变得不比零阶小**。正则摄动声称 $\varepsilon y_1 \ll y_0$，而长时间演化把这个前提破坏殆尽。

多重尺度法的应对：**不把 $\varepsilon$ 当固定的数，而把时间本身分解成多个尺度**。

## 2 多重尺度的思想：时间坐标变成两个

引入慢时间

$$
T = \varepsilon t
$$

把 $y$ 视为两个变量的函数 $y(t, T)$，两个变量在求导时**独立**：

$$
\frac{d}{dt} = \partial_t + \varepsilon\,\partial_T
$$

于是

$$
\dot{y} = y_t + \varepsilon y_T, \qquad
\ddot{y} = y_{tt} + 2\varepsilon\, y_{tT} + \varepsilon^2 y_{TT}
$$

把 $t$ 与 $T$ 当独立变量，代价是「多出来」的交叉项 $2\varepsilon y_{tT}$——而这正是久期项的猎物：**我们将在 $y_1$ 的方程里要求「久期项系数为零」，从而确定慢时间的演化规律**。<span class="marginnote">这套操作叫「摊开尺度再逐个消奇性」：快变量求振荡、慢变量求演化。每一个 $O(\varepsilon^n)$ 方程右端的共振模态，都给出一个慢演化的常微分方程——这是多重尺度法的引擎。</span>

## 3 多重尺度求解阻尼振子

对 $\ddot{y} + 2\varepsilon\dot{y} + y = 0$ 完整跑一遍。

**零阶**：$y_{0,tt} + y_0 = 0$，通解（$A,B$ 依赖 $T$）：

$$
y_0 = A(T)\cos t + B(T)\sin t
$$

改用振幅-相位形式 $y_0 = R(T)\cos(t - \phi(T))$。

**一阶**：$y_{1,tt} + y_1 = -2y_{0,tT} - 2y_{0,t}$。代入 $y_0 = R\cos(t-\phi)$，$\theta = t - \phi$：

$$
y_{0,t} = -R\sin\theta, \qquad
y_{0,tT} = -R'\sin\theta + R\phi'\cos\theta
$$

右端化为

$$
-2y_{0,tT} - 2y_{0,t}
= (2R' + 2R)\sin\theta - 2R\phi'\cos\theta
$$

其中 $R' = dR/dT$，$\phi' = d\phi/dT$。

**消久期**：$\sin\theta$ 与 $\cos\theta$ 都是 $y_1$ 的齐次模态，右端这两个模态必须清零：

$$
2R' + 2R = 0 \;\Rightarrow\; R(T) = R_0 e^{-T}, \qquad
\phi' = 0 \;\Rightarrow\; \phi \text{ 为常数}
$$

于是

$$
y \approx R_0\, e^{-\varepsilon t}\cos(t - \phi_0)
$$

**振幅按 $e^{-\varepsilon t}$ 指数衰减，频率保持 $1$**——与精确解 $e^{-\varepsilon t}\sin(\sqrt{1-\varepsilon^2}\,t)$ 在 $O(\varepsilon)$ 内完全一致。<span class="marginnote">对比第 6 篇正则摄动的惨状：那里只得到 $y\approx \sin t - \varepsilon t\sin t$，振幅随时间「线性涨破」；多重尺度把慢演化从快振荡里剥离，振幅正确地指数衰减。这就是两个时间尺度的胜利。</span>

## 4 公式解析：Duffing 振子的频率漂移

多重尺度最漂亮的战果之一是**非线性频率漂移**。考虑无阻尼非线性振子

$$
\ddot{y} + y + \varepsilon\, y^3 = 0
$$

- **第一步，零阶**：$y_0 = R\cos\theta$，$\theta = t - \phi(T)$，$R,\phi$ 依赖 $T$。
- **第二步，一阶方程**：$y_{1,tt} + y_1 = -2y_{0,tT} - y_0^3$。立方项展开：

$$
y_0^3 = R^3\cos^3\theta = R^3\left(\frac{3}{4}\cos\theta + \frac{1}{4}\cos 3\theta\right)
$$

- **第三步，分离共振模态**：右端含 $\sin\theta$、$\cos\theta$、$\cos 3\theta$。其中 $\sin\theta$、$\cos\theta$ 是齐次模态（须消去），$\cos 3\theta$ 不是（可保留，驱动三倍频响应）。消去条件：

$$
R' = 0, \qquad 2R\phi' + \frac{3}{4}R^3 = 0
$$

- **第四步，读出物理**：振幅 $R$ 守恒（无阻尼），但相位演化

$$
\phi(T) = -\frac{3}{8}R^2\, T = -\frac{3}{8}\varepsilon R^2\, t
$$

有效频率为 $\Omega = 1 - \dfrac{3}{8}\varepsilon R^2$。<span class="marginnote"><strong>频率依赖振幅</strong>——摆得越凶，周期越长。这是非线性振动的标志性效应，也是引力波、微机电谐振器、分子振动光谱里的实测效应。多重尺度把它从「久期项」里干净地提炼出来。</span>

**为什么 $R'=0$？** 因为右端 $\sin\theta$ 的系数是 $2R'$，无阻尼时没有耗散源，振幅自然守恒——这与阻尼例子里 $R'=-R$ 的来源完全平行：**每个久期项对应一个守恒/演化律**。

## 5 平均法：克里洛夫-博戈留波夫

当方程写成「简谐振子 + 小扰动」的标准形式

$$
\ddot{x} + x = \varepsilon\, f(x,\dot{x})
$$

**平均法（method of averaging）**，即克里洛夫-博戈留波夫（Krylov–Bogolyubov）方法，直接对振幅与相位做**一个周期上的平均**。设

$$
x = a(t)\cos\theta, \qquad \dot{x} = -a(t)\sin\theta, \qquad \theta = t + \psi(t)
$$

把 $a,\psi$ 视为慢变量，代入方程并**在一个周期上平均**（忽略快振荡项），得到

$$
\frac{da}{dt} = -\frac{\varepsilon}{2\pi}\int_0^{2\pi} f(a\cos\theta, -a\sin\theta)\,\sin\theta\, d\theta
$$

$$
\frac{d\psi}{dt} = \frac{\varepsilon}{2\pi a}\int_0^{2\pi} f(a\cos\theta, -a\sin\theta)\,\cos\theta\, d\theta
$$

以 **van der Pol 方程** $\ddot{x} + x = \varepsilon(1-x^2)\dot{x}$ 为例，$f = (1-x^2)\dot{x} = -(1-a^2\cos^2\theta)a\sin\theta$：

$$
\frac{da}{dt} = \frac{\varepsilon a}{2\pi}\int_0^{2\pi}(1-a^2\cos^2\theta)\sin^2\theta\, d\theta
= \frac{\varepsilon a}{2}\left(1 - \frac{a^2}{4}\right)
$$

非平凡不动点 $a = 2$——**van der Pol 极限环的振幅**，平均法一句话给出。<span class="marginnote">平均法与多重尺度在 $O(\varepsilon)$ 阶上给出相同结果：前者对一个周期「洗掉」快变量，后者用慢时间显式摊开快慢。选哪个是口味问题，物理结论一致——这正是它们常被并列讲授的原因。</span>

## 6 辨析｜易错点：多尺度的陷阱

- **慢时间定义不唯一**：$T=\varepsilon t$ 是 $O(\varepsilon)$ 尺度；若演化发生在 $O(\varepsilon^{1/2})$（如弱非线性耦合），得换 $T=\varepsilon^{1/2}t$。**先估计慢演化的时间量级，再定 $T$**。
- **两个快模态都要查**：$\sin\theta$ 与 $\cos\theta$ 是独立的齐次模态，**各自**都要消去。漏掉一个，展开照常算下去，但结果缺了相位的演化。
- **平均法别平均 $a$ 本身**：$a$ 里含快振荡分量 $O(\varepsilon)$，直接对 $x$ 平均会丢信息；先写 $x=a\cos\theta$ 再对 $\theta$ 平均才是正解。
- **共振与近共振**：若扰动频率接近固有频率（$\ddot{x} + x = \varepsilon\cos(\omega t)$，$\omega\approx 1$），主共振使振幅线性增长，多重尺度要在 $T$ 上显式引入失谐量 $\omega - 1$。**近共振不是小问题，是慢时间的另一个来源**。
- **多尺度解只在慢时间有限区间有效**：$T=O(1)$ 内精确；$T\to\infty$ 时需更高阶或直接数值。别指望一个展开管到底。

## 7 小结

- **多重尺度法**：令 $T=\varepsilon t$ 为独立慢变量，逐阶消去久期项，得到振幅/相位的慢演化方程。
- 阻尼振子：$R'=-R$ 给出指数衰减；Duffing 振子：$\phi'=-\tfrac38 R^2$ 给出振幅依赖的频率漂移。
- **平均法**（Krylov–Bogolyubov）对一个周期平均，直接给出 $a,\psi$ 的慢方程；van der Pol 极限环 $a=2$ 信手拈来。
- 久期项不是错误，而是**慢演化的线索**——每种共振模态都编码一条演化律。

在下一节，我们进入这门学科的收官应用：当波在**非均匀介质**中传播、$\varepsilon$