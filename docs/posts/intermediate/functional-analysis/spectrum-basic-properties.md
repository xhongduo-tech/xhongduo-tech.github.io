---
title: 有界线性算子谱的基本性质：非空性与紧性
date: 2026-08-07
---

# 有界线性算子谱的基本性质：非空性与紧性

<div class="epigraph">
<p>谱永远非空、永远紧——这是有界算子世界的基本律法。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§9.3 ｜ 2026-08-07</p>
</div>

## 为什么谱的「形状」是基本问题

有了谱的定义，第一个基本问题：**谱长什么样？** 两个惊人的普遍答案：**谱永远非空**（每个有界算子都有谱点），且**谱是有界闭集**（紧集）。前者说明「谱」不是可有可无的装饰——每个算子都有谱；后者说明谱被限制在「以 $\|T\|$ 为半径的圆盘」里，且包含聚点。这两条性质是所有谱理论（谱半径公式、紧算子谱、自伴谱）的共同地基。<span class="marginnote">非空性的意义非同小可：<strong>有限维矩阵「特征多项式必有根」（代数基本定理）的无穷维推广，就是「谱必非空」</strong>。若无界算子除外，每个有界算子都「有谱」——这个事实支撑着整个谱理论的存在性。</span>

## 1 谱是有界的

**定理（谱的有界性）**：设 $T \in \mathcal{B}(X)$，则 $\sigma(T) \subset \overline{B}(0, \|T\|)$，即

$$
|\lambda| > \|T\| \Rightarrow \lambda \in \rho(T)
$$

**证明（诺伊曼级数）**：对 $|\lambda| > \|T\|$，

$$
\lambda I - T = \lambda\left(I - \frac{T}{\lambda}\right), \qquad \left\|\frac{T}{\lambda}\right\| < 1
$$

由诺伊曼级数（第三章），$I - T/\lambda$ 可逆，故 $\lambda I - T$ 可逆，$\lambda \in \rho(T)$。<span class="marginnote">这个证明是「诺伊曼级数」的又一次登场：<strong>谱点一定在 $\\|T\\|$ 半径内，因为半径外 $T/\\lambda$ 是压缩</strong>。它还给出预解式的级数表示 $R_\\lambda = \\sum_{n=0}^\\infty T^n/\\lambda^{n+1}$——谱半径公式（下节）的起点。</span>

**例**：$\|T\| = 1$ 的算子（如移位 $S$）谱落在闭单位盘内：$\sigma(S) = \overline{D}$。

## 2 谱是闭的

**定理（谱的闭性）**：$\rho(T)$ 是开集，从而 $\sigma(T)$ 是闭集。

**证明**：设 $\lambda_0 \in \rho(T)$，要证 $\lambda_0$ 附近仍属于 $\rho(T)$。对 $|\lambda - \lambda_0|$ 小：

$$
\lambda I - T = (\lambda_0 I - T) + (\lambda - \lambda_0) I = (\lambda_0 I - T)\big(I + (\lambda - \lambda_0) R_{\lambda_0}\big)
$$

取 $|\lambda - \lambda_0| < 1/\|R_{\lambda_0}\|$，则 $\|(\lambda - \lambda_0)R_{\lambda_0}\| < 1$，由诺伊曼级数第二因子可逆，故 $\lambda I - T$ 可逆。<span class="marginnote">证明再次用诺伊曼级数——这次是「在预解点附近展开」：<strong>只要 $\\lambda_0$ 是正则点，它周围一小片都是正则点（预解集开）</strong>。同时这个论证还说明 $\\lambda \\mapsto R_\\lambda$ 在 $\\rho(T)$ 上是（局部）算子值解析函数——谱的闭性与预解式的解析性密不可分。</span>

**核心要点：$\sigma(T)$ 是紧集**——闭 + 有界（$|\lambda| \le \|T\|$）。谱被限制在一个「以原点为中心、半径 $\|T\|$」的紧圆盘里。

## 3 谱是非空的

**定理（谱的非空性）**：设 $X \neq \{0\}$ 是复 Banach 空间，$T \in \mathcal{B}(X)$。则 $\sigma(T) \neq \emptyset$。

**证明（用预解式的解析性 + Liouville）**：

- **第一步**：假设 $\sigma(T) = \emptyset$，则 $R_\lambda$ 对一切 $\lambda \in \mathbb{C}$ 有定义。
- **第二步（解析性）**：$R_\lambda$ 是整函数（全平面解析）——由预解式方程与诺伊曼展开可证 $R_\lambda$ 的强解析性。
- **第三步（有界性）**：$|\lambda| > 2\|T\|$ 时 $\|R_\lambda\| \le \frac{1}{|\lambda| - \|T\|} \to 0$（$\lambda \to \infty$）——$R_\lambda$ 在无穷远趋于 0。
- **第四步（Liouville）**：整函数 + 有界（有界性由局部一致 + 无穷远趋于 0 给出）⟹ 常数；但常数且趋于 0 ⟹ $R_\lambda \equiv 0$——矛盾（$R_\lambda$ 是算子，不可能恒为零）。<span class="marginnote">这个证明是复分析进入泛函分析的典范：<strong>用 Liouville 定理（有界整函数是常数）反证谱非空</strong>。它的深刻含义：谱的非空性本质上依赖复数的代数完备性——实 Banach 空间的算子可以没有实谱（旋转矩阵的谱是复的）。</span>

**辨析｜易错点：** 谱的非空性**依赖复空间**。在实 Banach 空间上，旋转矩阵 $\begin{pmatrix}0 & -1 \\ 1 & 0\end{pmatrix}$ 没有实特征值、实谱为空——只有取复化后谱才非空。所以「谱非空」是复 Banach 空间理论。

## 4 公式解析：预解式在无穷远的行为

谱非空证明里最精妙的是「$R_\lambda \to 0$」这一步：

$$
\|R_\lambda\| \le \frac{1}{|\lambda| - \|T\|}, \qquad |\lambda| > \|T\|
$$

- **第一步，展开**：$\lambda I - T = \lambda(I - T/\lambda)$，$\|T/\lambda\| < 1$。
- **第二步，级数**：$R_\lambda = \frac{1}{\lambda}\sum_{n=0}^\infty \frac{T^n}{\lambda^n}$。
- **第三步，估计**：$\|R_\lambda\| \le \frac{1}{|\lambda|}\sum \frac{\|T\|^n}{|\lambda|^n} = \frac{1}{|\lambda| - \|T\|}$。
- **第四步，极限**：$|\lambda| \to \infty$ 时右边 $\to 0$。

**关键**：$R_\lambda$ 在无穷远「像 $\frac1\lambda$ 一样趋于 0」——这个衰减是有界性的来源，配合 Liouville 反证谱非空。**谱非空的深层原因，是「预解式在无穷远必须消失」与「解析函数不能无端消失」之间的矛盾**。

## 5 例题精讲：谱性质的应用

**例题一：$T^n$ 的谱**。

- $\sigma(T^n) = \{\lambda^n : \lambda \in \sigma(T)\}$（谱映射定理）。
- $\sigma(T)$ 非空 ⟹ $\sigma(T^n)$ 非空。
- 谱的紧性：$\sigma(T^n)$ 也紧（连续像）。

**例题二：$\|T^n\|$ 与谱半径**。

- 谱半径 $r(T) = \sup_{\lambda \in \sigma(T)}|\lambda| \le \|T\|$。
- 谱半径公式（下节）：$r(T) = \lim \|T^n\|^{1/n}$。
- 谱的紧性保证 sup 可达（$r(T)$ 是最大模）。

**例题三：$T$ 可逆 ⟺ $0 \notin \sigma(T)$**。

- $0 \in \rho(T)$ ⟺ $T$ 可逆（定义）。
- 谱非空性说明「$T$ 无谱」不可能；$T$ 可逆与否看 0 是否在谱里。
- 紧算子 $K$：$0 \in \sigma(K)$ 恒成立（$K$ 不可逆于无穷维），这是紧算子的标志。

**核心要点**：谱性质的三个应用——幂的谱、谱半径、可逆性——都是「非空 + 紧」两条基本律法的推论。

**辨析｜易错点：** 谱半径 $r(T) \le \|T\|$，但可以严格小于（拟幂零算子 $V$：$r(V) = 0$ 而 $\|V\| > 0$）。「谱半径 ≤ 范数」只是上界，谱半径公式给出精确值。

## 6 常见误区与辨析

**误区一：以为谱半径等于范数**。

- 只对自伴算子成立。
- 拟幂零算子 $r(T) = 0$ 而 $\|T\| > 0$。

**误区二：忘记谱非空依赖复空间**。

- 实空间算子可无实谱。
- 复化后谱非空。

**误区三：把「$0 \in \sigma(T)$」当「$T$ 无界」**。

- $0 \in \sigma(T)$ 只是「$T$ 不可逆」。
- 紧算子在无穷维总有 $0 \in \sigma(T)$。

**核心要点：谱非空 + 紧是有界算子谱的基本律法**——一切谱理论的共同地基。


## 7 小结

- **有界**：$|\lambda| > \|T\| \Rightarrow \lambda \in \rho(T)$（诺伊曼级数）。
- **闭**：$\rho(T)$ 开，$\sigma(T)$ 闭——谱是紧集。
- **非空**：$\sigma(T) \neq \emptyset$（Liouville 反证，依赖复空间）。
- **预解式**：$R_\lambda$ 在 $\rho(T)$ 解析、在无穷远趋于 0。
- **推论**：谱映射、谱半径 $\le \|T\|$、$T$ 可逆 ⟺ $0 \notin \sigma(T)$。
- **定位**：谱的「非空 + 紧」是所有谱理论（下节的谱半径公式）的共同地基。

在下一节，我们证明谱半径公式——**Gelfand 定理**：谱半径由范数的幂次根极限精确给出。
