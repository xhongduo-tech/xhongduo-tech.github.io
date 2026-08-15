---
title: Weierstrass 逼近定理及其证明
date: 2026-08-07
---

# Weierstrass 逼近定理及其证明

<div class="epigraph">
<p>一个定义在闭区间上的连续函数，可以在这个区间上被多项式一致逼近到任意指定的精度。</p>
<footer>—— 卡尔 · 魏尔斯特拉斯（Karl Weierstrass, 1885）</footer>
</div>

<div class="article-byline">
<p>第二级 · 函数逼近论 ｜ E. Ward Cheney, Introduction to Approximation Theory, §1.2 ｜ 2026-08-07</p>
</div>

## 为什么从 Weierstrass 定理开始

上一篇搭好了赋范空间的框架，把逼近问题写成了「在子空间里找最近点」。但一个根本问题悬而未决：**候选集里有没有足够多的好东西？** 如果多项式只能表示平滑得像直线的东西，那「逼近任意连续函数」就是空谈。Weierstrass 定理一劳永逸地回答了这件事：多项式在 $C[a,b]$ 中是稠密的。它不只是逼近论的第一个大定理，更在某种意义上定义了整门学科的问题域——因为有了它，「能用多项式逼近」才从奢望变成了定理，后续一切最佳逼近、插值、正交展开的研究才有了意义。同时，这个定理也是第二级《数学分析》一致收敛理论在函数空间视角下的自然延伸——「收敛」不再只是数列的性质，而是整个函数空间的拓扑性质。

## 1 定理陈述与归一化

**Weierstrass 逼近定理（Weierstrass approximation theorem）**：设 $f$ 在闭区间 $[a,b]$ 上连续，则对任意 $\varepsilon > 0$，存在一个多项式 $p$，使得

$$
|f(x) - p(x)| \lt  \varepsilon, \qquad \forall x \in [a,b]
$$

用范数写就是 $\|f - p\|_{\infty} \lt  \varepsilon$。这等价于说：**多项式全体在 $C[a,b]$ 中稠密**。<span class="marginnote">「稠密」是拓扑语言：子集 $V$ 稠密当且仅当空间中每个元素都能被 $V$ 中的点以任意精度逼近。Weierstrass 定理的实质就是「多项式在连续函数里稠密」这一句话。</span>

讨论证明前先做一个**归一化（normalization）**：任何 $[a,b]$ 都可以通过仿射变换 $t = (x-a)/(b-a)$ 映到 $[0,1]$，多项式经仿射变换后仍是多项式，所以只需证明 $[0,1]$ 上的情形。这个「把区间拉回标准位置」的手法在本专题会反复出现。

## 2 Bernstein 多项式：概率视角的构造

1895 年，伯恩斯坦（Sergei Bernstein）给出了一个漂亮得惊人的证明。对 $f \in C[0,1]$，定义**第 $n$ 个 Bernstein 多项式**：

$$
B_n(f)(x) = \sum_{k=0}^{n} f\!\left(\frac{k}{n}\right) \binom{n}{k} x^k (1-x)^{n-k}
$$

初看这是「把函数值 $f(k/n)$ 按二项分布的权重加权平均」。这个形式不是凭空掉下来的：做 $n$ 次独立试验，每次成功的概率是 $x$，则成功次数 $S_n$ 服从二项分布 $\mathrm{Bin}(n, x)$，而

$$
\mathbb{P}(S_n = k) = \binom{n}{k} x^k (1-x)^{n-k}
$$

于是 $B_n(f)(x) = \mathbb{E}[f(S_n/n)]$——**在 $x$ 处取 $f$ 关于成功频率 $S_n/n$ 的期望**。<span class="marginnote">概率论的「大数定律」说：试验次数越多，成功频率 $S_n/n$ 越接近成功概率 $x$。Bernstein 的洞察是把「频率趋近概率」翻译成「多项式的值趋近 $f(x)$」——概率直觉直接变成了一个构造性的证明。</span>

## 3 证明的三个步骤

证明分三步，核心是「先证 $B_n(f)(x)$ 与 $f(x)$ 的差可以被方差控制，再用一致连续性把它压到 $\varepsilon$ 以内」。

**第一步，把差写成期望。** 由 $\sum_{k=0}^n \binom{n}{k} x^k(1-x)^{n-k} = 1$，有

$$
B_n(f)(x) - f(x) = \mathbb{E}\!\left[f\!\left(\frac{S_n}{n}\right) - f(x)\right]
$$

**第二步，控制「频率离概率有多远」。** 二项分布方差是 $\mathrm{Var}(S_n) = n x (1-x)$，所以

$$
\mathbb{E}\!\left[\left(\frac{S_n}{n} - x\right)^2\right] = \frac{x(1-x)}{n} \le \frac{1}{4n}
$$

这是整个证明的量纲来源：$S_n/n$ 到 $x$ 的均方距离以 $1/n$ 的速度趋于零。

**第三步，用一致连续性收尾。** $f$ 在 $[0,1]$ 上一致连续：给定 $\varepsilon>0$，存在 $\delta > 0$ 使 $|s-t|\lt \delta$ 蕴含 $|f(s)-f(t)|\lt \varepsilon/2$。把样本空间按 $|S_n/n - x|$ 是否小于 $\delta$ 分成两段：近段用一致连续性，远段用有界性 $\|f\|_\infty$ 与第二步的均方界，最终得到

$$
|B_n(f)(x) - f(x)| \le \frac{\varepsilon}{2} + \frac{2\|f\|_\infty}{\delta^2} \cdot \frac{1}{4n}
$$

对充分大的 $n$，第二项也小于 $\varepsilon/2$。$B_n(f)$ 逐点且一致地收敛到 $f$。这就是定理的全部。

## 4 公式解析：Bernstein 算子的收敛速度

**Bernstein 证明的美在于：收敛速度被概率方差直接钉死。** 关键量是均方偏差

$$
\mathbb{E}\!\left[\left(\frac{S_n}{n} - x\right)^2\right] = \frac{x(1-x)}{n}
$$

拆解它的三步：

- **第一步，$x(1-x)$ 是「边缘风险」**：当 $x$ 靠近端点 $0$ 或 $1$ 时，$x(1-x)$ 很小——频率本来就几乎确定，方差天然小；在 $x = 1/2$ 时达到最大 $1/4$。这个因子让收敛「在中间慢、在两端快」。
- **第二步，$1/n$ 是「样本量的回报」**：方差随试验次数线性衰减。这正是大数定律的定量版本——频率以 $1/\sqrt{n}$ 的尺度摇摆，平方后是 $1/n$。
- **第三步，它如何翻译成逼近误差**：均匀连续函数在尺度 $\delta$ 内至多抖 $\varepsilon/2$，而落进尺度 $\delta$ 之外的概率至多 $\mathrm{Var}/\delta^2$（Chebyshev 不等式）。两者一夹，误差就破了 $\varepsilon$。

值得记在心里的结论：**Bernstein 多项式的逼近速度至多是 $O(1/n)$ 量级**，实际对平滑函数往往更慢。它优美、构造性、普适，但在数值上收敛缓慢——这为后面「为什么需要最佳逼近与 Chebyshev 多项式」埋下了伏笔。

### 一个可算的例子：$f(x) = x^2$

用 Bernstein 多项式验证收敛是极好的练手。二项分布 $\mathrm{Bin}(n,x)$ 的两个矩是已知的：

$$
\mathbb{E}\!\left[\frac{S_n}{n}\right] = x, \qquad
\mathbb{E}\!\left[\left(\frac{S_n}{n}\right)^2\right] = x^2 + \frac{x(1-x)}{n}
$$

于是对 $f(t) = t^2$，由定义 $B_n(f)(x) = \mathbb{E}[f(S_n/n)]$ 直接得到**显式公式**：

$$
B_n(x^2) = \mathbb{E}\!\left[\left(\frac{S_n}{n}\right)^2\right] = x^2 + \frac{x(1-x)}{n}
$$

这个例子处处可爱：

- **误差精确已知**：$\|B_n(x^2) - x^2\|_\infty = \max_{x\in[0,1]} \frac{x(1-x)}{n} = \frac{1}{4n}$，恰在 $x = 1/2$ 处取到。概率方差一步到位，无需任何高阶导数估计。
- **低次多项式被保持**：$B_n(1) = 1$、$B_n(x) = x$——Bernstein 算子精确保持不超过一次的多项式。
- **概率直觉分毫不差**：误差项就是方差 $x(1-x)/n$。这个「矩即误差」的现象在 Bernstein 理论里是系统性的——对二次多项式如此，对高阶函数则要动用更精细的概率估计。

这个例子最大的教学价值，是把「概率直觉」落成了「显式公式」：你亲眼看到方差如何变成逼近误差，而不是停在「由大数定律保证收敛」的定性层面。

## 5 推广与边界

Weierstrass 定理之后，边界问题立刻浮现：**连续是充分条件，还是必要条件？** 答案是——不能放松太多。多项式本身连续，一致收敛保持连续性，所以被多项式一致逼近的函数必须连续；反过来定理说连续就够了。因此**「存在多项式一致逼近」与「函数连续」在 $C[a,b]$ 上是同一件事**。

定理还有两个重要推广：

- **Stone–Weierstrass 定理**：把「多项式」换成更一般的代数（含常数、分离点、封闭于共轭），在紧 Hausdorff 空间上仍有稠密性。这是 Weierstrass 定理在抽象空间中的最终形态，也是后续泛函分析课程的经典内容。
- **Weierstrass 第二定理**：周期连续函数可被三角多项式一致逼近。它把理论从「代数多项式」推进到「三角多项式」，是 Fourier 逼近（本专题第 7 篇）的理论基石。

同时要划清一条边界：**插值多项式在等距节点上未必收敛**——Runge 现象会制造越来越大的振荡（第 5 篇细讲）。所以「存在稠密的多项式」与「随便插值都能逼近」是两回事：前者是存在性定理，后者要选节点、选方法，这正是整个逼近论应用层面的核心张力。

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| Bernstein 多项式 | Bernstein polynomial | $B_n(f)(x)=\sum_{k=0}^n f(k/n)\binom{n}{k}x^k(1-x)^{n-k}$ |
| 稠密 | dense | 子集的闭包等于全空间，元素可被任意逼近 |
| 一致收敛 | uniform convergence | $\|f_n - f\|_\infty \to 0$，误差与点位置无关 |
| 一致连续 | uniformly continuous | 存在与 $x$ 无关的 $\delta$ 控制偏差 |
| 二项分布 | binomial distribution | $n$ 次独立成功概率 $x$ 的试验的成功次数 |
| 大数定律 | law of large numbers | 试验次数增多时频率收敛到概率 |
| Stone–Weierstrass 定理 | Stone–Weierstrass theorem | Weierstrass 定理在紧 Hausdorff 空间的推广 |
| Runge 现象 | Runge phenomenon | 等距高次插值在边界附近振荡发散 |
| 归一化 | normalization | 仿射变换把一般区间映到 $[0,1]$ |
| 三角多项式 | trigonometric polynomial | $\frac{a_0}{2}+\sum_{k=1}^n(a_k\cos kx+b_k\sin kx)$ |

## 7 小结

- Weierstrass 定理：$[0,1]$ 上连续函数可被多项式一致逼近到任意精度，即**多项式在 $C[a,b]$ 中稠密**。
- Bernstein 多项式 $B_n(f)(x) = \mathbb{E}[f(S_n/n)]$ 给出了**概率论式的构造性证明**：成功频率 $S_n/n$ 依大数定律趋近 $x$。
- 收敛的核心是均方偏差 $\mathbb{E}[(S_n/n - x)^2] = x(1-x)/n \to 0$，配合一致连续性把误差夹到 $\varepsilon$ 以内。
- Bernstein 逼近速度约 $O(1/n)$