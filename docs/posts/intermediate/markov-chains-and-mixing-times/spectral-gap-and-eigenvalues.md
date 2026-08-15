---
title: 谱隙与特征值方法
date: 2026-08-07
---

# 谱隙与特征值方法

<div class="epigraph">
<p>凡物皆有节奏，特征值就是节奏的频率。</p>
<footer>—— 戴维 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 马尔可夫链与混合时间 ｜ Levin, Peres &amp; Wilmer《Markov Chains and Mixing Times》Ch. 12 ｜ 2026-08-07</p>
</div>

## 为什么从谱隙与特征值方法开始

耦合方法给出混合时间上界，但它依赖「设计巧妙的耦合」，而设计往往需要灵感。**谱方法**换一条路：把转移矩阵当线性算子，用它的**特征值**直接读混合速度。对可逆链，$P$ 关于平稳分布是对称算子，谱分解给出干净的公式——$P^n$ 的非平凡特征值都以 $\lambda^n$ 衰减，最慢的模式由**第二大特征值**决定。于是「混合多快」浓缩成一个数：**谱隙（spectral gap）**。这个方法的好处是：它是纯代数计算，不需要概率灵感；它给出的是**上界**，且往往与真实混合时间只差对数因子。<span class="marginnote">谱方法与耦合方法互为补充：耦合更直观、谱更机械；耦合的上界在「几何上远」的链上更紧，谱的上界在「代数上慢」的链上更省力。现代混合时间研究常两者并用。</span>

## 1 可逆链的谱表示

设 $P$ 关于平稳分布 $\pi$ 可逆（细致平衡成立）。在带权内积

$$
\langle f, g \rangle_\pi = \sum_{x \in \Omega} f(x) g(x) \pi(x)
$$

下，$P$ 是自伴算子：$\langle f, Pg \rangle_\pi = \langle Pf, g \rangle_\pi$。因此 $P$ 有**实特征值**，可以按大小排列：

$$
1 = \lambda_0 \geq \lambda_1 \geq \cdots \geq \lambda_{N-1} \geq -1
$$

其中 $N = |\Omega|$，$\lambda_0 = 1$ 对应常数函数（平稳分布），其余特征值对应「在 $\pi$ 下均值为 0」的模式。<span class="marginnote">特征值 1 总是出现，因为 $P \mathbf{1} = \mathbf{1}$（行和为 1）。$P$ 的自伴性保证特征向量正交、特征值实数——这是普通非可逆链没有的奢侈品。</span>

谱表示的核心结论：对任意函数 $f$，

$$
P^n f = \mathbb{E}[f(X_n) \mid X_0 = \cdot] = \sum_{i=0}^{N-1} \lambda_i^n \langle f, \phi_i \rangle_\pi \phi_i
$$

其中 $\{\phi_i\}$ 是 $L^2(\pi)$ 的标准正交特征函数基。每一项 $\lambda_i^n$ 都在以几何速率衰减，**最慢的项是第二大特征值** $\lambda_1$。

## 2 松弛时间与谱隙

**谱隙（spectral gap）**定义为

$$
\gamma = 1 - \lambda_1
$$

即特征值 1 与第二大特征值之间的间距。**松弛时间（relaxation time）** 是其倒数：

$$
t_{\mathrm{rel}} = \frac{1}{\gamma} = \frac{1}{1 - \lambda_1}
$$

名字的由来：$\lambda_1^n = (1-\gamma)^n \approx e^{-\gamma n}$，所以**每个「模式」以速率 $\gamma$ 指数衰减，衰减到 $1/e$ 需要 $t_{\mathrm{rel}}$ 步**。<span class="marginnote">直观上，$t_{\mathrm{rel}}$ 是「链忘掉初始状态特征值」的特征时间。它不直接等于混合时间，但对可逆链两者只差对数因子：$t_{\mathrm{mix}} \geq \tfrac12 t_{\mathrm{rel}}$ 且 $t_{\mathrm{mix}} \leq t_{\mathrm{rel}} \log(4/\pi_{\min})$（当 $\pi_{\min} > 0$）。</span>

松弛时间与混合时间的关系是谱方法的支柱：

$$
\frac{1}{2}\, t_{\mathrm{rel}} \leq t_{\mathrm{mix}} \leq t_{\mathrm{rel}} \log\left(\frac{4}{\pi_{\min}}\right)
$$

**下界**来自「最慢模式的存活」：取特征函数 $\phi_1$ 对应的事件，其概率差以 $\lambda_1^n$ 衰减，故需要至少 $\tfrac12 t_{\mathrm{rel}}$ 步。**上界**来自 $L^2$ 到全变差的转化：$\lVert \mu P^n - \pi \rVert_{\mathrm{TV}}^2 \leq \tfrac{1}{4\pi_{\min}} \lVert \mu_0 - \pi \rVert^2_\pi \, e^{-2\gamma n}$，再解出 $n$。

## 3 公式解析：$L^2$ 混合时间上界的推导

写出关键链：

$$
\lVert \mu_0 P^n - \pi \rVert_{\mathrm{TV}}^2 \leq \frac{\lVert \mu_0 - \pi \rVert_\pi^2}{4\pi_{\min}} \, e^{-2\gamma n}
$$

逐项拆解这条估计的来源：

- **第一步，全变差到 $L^2(\pi)$**：对任意概率 $\mu$，利用 Cauchy–Schwarz，

$$
\lVert \mu - \pi \rVert_{\mathrm{TV}}^2 = \tfrac14 \left(\sum_x |\mu(x)-\pi(x)|\right)^2 \leq \frac{1}{4}\left(\sum_x \frac{|\mu(x)-\pi(x)|^2}{\pi(x)}\right)\left(\sum_x \pi(x)\right) = \frac{\lVert\mu-\pi\rVert_\pi^2}{4}
$$

但 $\mu$ 可能在某处远小于 $\pi$，所以用 $\pi_{\min} = \min_x \pi(x)$ 收紧：$\mu(x)-\pi(x)$ 的支撑受限，额外因子 $1/\pi_{\min}$ 出现。
- **第二步，谱衰减 $L^2$ 范数**：因为 $P$ 自伴且非平凡特征值都被 $\lambda_1$ 控制，

$$
\lVert P^n f - \pi f \rVert_\pi^2 = \sum_{i \geq 1} \lambda_i^{2n} \langle f, \phi_i\rangle_\pi^2 \leq \lambda_1^{2n} \lVert f \rVert_\pi^2 = e^{-2\gamma n} \lVert f \rVert_\pi^2
$$

- **第三步，把 $f = d\mu_0/d\pi - 1$**：$\lVert f \rVert_\pi = \lVert \mu_0 - \pi \rVert_\pi$，代入第一步，得到上面整条估计。
- **第四步，解出 $n$**：令右边 $\leq \varepsilon^2$，得 $n \geq \tfrac12 t_{\mathrm{rel}} \log(\lVert\mu_0-\pi\rVert_\pi^2 / (4\pi_{\min}\varepsilon^2))$，整理即得上界公式。

这条推导的核心是**自伴性 → 正交分解 → 逐模式衰减**，三步走可复用到任何可逆链的谱分析。

## 4 Dirichlet 形式与变分公式

特征值本身往往不可显式计算，但可以用**变分公式**把它夹出来。定义 Dirichlet 形式

$$
\mathcal{E}(f, f) = \frac{1}{2}\sum_{x,y} \pi(x) P(x,y) (f(x) - f(y))^2
$$

它是「函数 $f$ 在链的随机一跳下的平均平方变化」的一半。谱隙的**变分刻画（Rayleigh 商）**：

$$
\gamma = \min_{f : \operatorname{Var}_\pi(f) \neq 0} \frac{\mathcal{E}(f, f)}{\operatorname{Var}_\pi(f)}
$$

其中 $\operatorname{Var}_\pi(f) = \tfrac12\sum_{x,y}\pi(x)\pi(y)(f(x)-f(y))^2$。<span class="marginnote">这个公式把「找最小特征值」变成「在函数空间里找一个比值最小的 $f$」——为分析留出了巨大空间：取一个试探函数 $f$ 代入，就得到谱隙的一个上界；这比直接对角化 $P$ 容易得多。</span>

对随机游走在图上的链，Dirichlet 形式有漂亮的翻译：若 $\pi(x) = \deg(x)/(2|E|)$，则

$$
\mathcal{E}(f, f) = \frac{1}{2|E|} \sum_{\{x,y\} \in E} (f(x) - f(y))^2
$$

谱隙衡量「图上振荡模式的最小耗散」——这连接了**瓶颈比（bottleneck ratio）**与 Cheeger 不等式，是谱隙下界的标准工具（第 7 篇会用到类似技术）。

## 5 例子：$n$-环上的简单随机游走

状态为 $\mathbb{Z}_n$（模 $n$ 的环），每步以 $1/2$ 概率 $\pm 1$。转移矩阵是循环矩阵，特征向量是傅里叶基 $f_k(x) = e^{2\pi i kx/n}$，特征值为

$$
\lambda_k = \frac{e^{2\pi i k/n} + e^{-2\pi i k/n}}{2} = \cos\left(\frac{2\pi k}{n}\right)
$$

第二大特征值 $\lambda_1 = \cos(2\pi/n) \approx 1 - \frac{2\pi^2}{n^2}$，故谱隙

$$
\gamma = 1 - \cos\frac{2\pi}{n} \approx \frac{2\pi^2}{n^2}
$$

松弛时间 $t_{\mathrm{rel}} \approx n^2/(2\pi^2)$。<span class="marginnote">$n^2$ 的标度是扩散的签名：随机游走走 $t$ 步的典型位移约 $\sqrt{t}$，要覆盖整个 $n$ 环需要 $t \sim n^2$。谱隙直接读出这个二次标度——这是谱方法最漂亮的时刻：代数特征值对应几何直觉。</span>混合时间上界给出 $t_{\mathrm{mix}} \leq t_{\mathrm{rel}}\log(4/\pi_{\min}) \approx C n^2 \log n$，与真实量级一致。

## 6 辨析｜易错点：负特征值与奇偶性

**辨析｜易错点：** 谱隙只关心 $\lambda_1$（第二大），但**最小特征值 $\lambda_{N-1}$ 也会拖慢混合**。若 $\lambda_{N-1} \approx -1$，则 $P^n$ 在奇偶步之间振荡，链可能「几乎周期」。典型例子：无自环的二分图随机游走，特征值恰为 $\{\pm 1\}$ 之外还有 $-1$，导致混合需要「先懒惰化再分析」。判断混合时间时必须检查 $|\lambda_{N-1}|$ 而非只盯 $\lambda_1$。

**辨析｜易错点：** 谱方法只对**可逆**链给出干净的 $L^2$ 理论。不可逆链的特征值可能是复数，$P^n$ 的范数衰减不能用实数特征值排序直接读。此时要么先验证可逆性，要么改用耦合/瓶颈等不依赖谱对称性的方法。

## 7 小结

- 可逆链的 $P$ 在 $L^2(\pi)$ 中**自伴**，谱分解给出 $P^n f = \sum_i \lambda_i^n \langle f,\phi_i\rangle_\pi \phi_i$。
- **谱隙** $\gamma = 1-\lambda_1$ 与**松弛时间** $t_{\mathrm{rel}} = 1/\gamma$ 刻画最慢模式的衰减速率。
- 混合时间被松弛时间夹住：$\tfrac12 t_{\mathrm{rel}} \leq t_{\mathrm{mix}} \leq t_{\mathrm{rel}} \log(4/\pi_{\min})$。
- **Dirichlet 形式**给出谱隙的变分公式，可用试探函数求上界，并连接瓶颈比。
- $n$-环随机游走的谱隙 $\approx 2\pi^2/n^2$，直接读出扩散的 $n^2$