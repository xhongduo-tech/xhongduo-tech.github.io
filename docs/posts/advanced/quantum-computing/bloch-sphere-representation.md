---
title: 布洛赫球（Bloch sphere）表示
date: 2026-08-07
---

# 布洛赫球（Bloch sphere）表示

<div class="epigraph">
<p>数学语言在物理定律表述中的不可思议的有效性，是上天赐予的礼物，我们既不理解，也不配拥有。</p>
<footer>—— 尤金 · 维格纳（Eugene Wigner）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§1.2 量子比特 ｜ 2026-08-07</p>
</div>

## 为什么从布洛赫球开始

上一篇《量子比特》把单个量子比特写成 $|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$，并预告了这颗球。今天我们把几何彻底立起来：**一个量子比特的全部纯态，恰好就是单位球面（半径 1 的球面）上的每一个点**。这颗球叫**布洛赫球（Bloch sphere）**。

布洛赫球是量子计算里最值钱的一张图。它把抽象的复振幅、相位、测量概率、甚至量子门，全部翻译成看得见摸得着的几何：态是点，测量是「把点压到一根轴上」，门是「把球转一下」。以后读到 Grover 的振幅放大、变分算法的参数化线路，脑子里浮现的都是这颗球在转。维格纳说的「数学语言的有效性」，在这里有个具体的注脚——**一个二维复向量空间，去掉不可观测的整体相位后，剩下的自由度恰好是二维实球面**，几何与物理严丝合缝地对上了。

## 1 参数化：两个实数，一颗球

先复习上一篇的结论。去掉全局相位后，任意单量子比特纯态可以唯一写成

$$
|\psi\rangle = \cos\frac{\theta}{2}\,|0\rangle + e^{i\phi}\sin\frac{\theta}{2}\,|1\rangle, \qquad 0 \leq \theta \leq \pi, \quad 0 \leq \phi < 2\pi
$$

把 $\theta, \phi$ 当作**球坐标**：$\theta$ 是极角（从 $z$ 轴正方向量起），$\phi$ 是方位角（绕 $z$ 轴的转角）。把它们翻译成直角坐标：

$$
x = \sin\theta\cos\phi, \qquad y = \sin\theta\sin\phi, \qquad z = \cos\theta
$$

**重点：** $x^2 + y^2 + z^2 = \sin^2\theta + \cos^2\theta = 1$，所以每个态对应**单位球面上的一点**；反过来，球面上每个点都能解出唯一的一组 $(\theta, \phi)$，从而唯一确定一个态。**点与态是一一对应的**。

为什么极角是 $\theta/2$ 而不是 $\theta$？这是初学最容易卡住的地方。**辨析｜易错点：** 球面上的极角 $\theta$ 与叠加系数里的 $\theta/2$ 是两个不同的量。$\theta = 0$ 时 $\cos 0 = 1$，态是 $|0\rangle$，落在**北极**；$\theta = \pi$ 时 $\sin\frac{\pi}{2} = 1$，态是 $|1\rangle$，落在**南极**。若直接用 $\theta$ 当极角，$|1\rangle$ 会被送到赤道而不是南极——正是这个 $\frac12$ 把「比例的连续变化」压缩成球面纬度，让 $|0\rangle$ 与 $|1\rangle$ 成为对跖点。<span class="marginnote">用 $\theta/2$ 的深层原因是：态矢量绕布洛赫矢量转 $2\pi$ 才回到自身（旋量转一圈变负号），而球面转 $\pi$ 已是对跖点。这个「转两圈才还原」的现象与自旋 $\tfrac12$ 的 $720^\circ$ 对称性同源，到第九篇《超导量子比特》谈物理实现时会再遇到。</span>

## 2 布洛赫球上的地标

给球面装上坐标系，六个最重要的态正好落在三根轴上：

| 态 | 表达式 | $(\theta, \phi)$ | 布洛赫矢量 $(x,y,z)$ |
| --- | --- | --- | --- |
| $|0\rangle$ | $\binom{1}{0}$ | $(0, \cdot)$ | $(0, 0, 1)$ 北极 |
| $|1\rangle$ | $\binom{0}{1}$ | $(\pi, \cdot)$ | $(0, 0, -1)$ 南极 |
| $|+\rangle$ | $\frac{|0\rangle+|1\rangle}{\sqrt2}$ | $(\frac{\pi}{2}, 0)$ | $(1, 0, 0)$ |
| $|-\rangle$ | $\frac{|0\rangle-|1\rangle}{\sqrt2}$ | $(\frac{\pi}{2}, \pi)$ | $(-1, 0, 0)$ |
| $|+i\rangle$ | $\frac{|0\rangle+i|1\rangle}{\sqrt2}$ | $(\frac{\pi}{2}, \frac{\pi}{2})$ | $(0, 1, 0)$ |
 | $\frac{|0\rangle-i|1\rangle}{\sqrt2}$ | $(\frac{\pi}{2}, \frac{3\pi}{2})$ | $(0, -1, 0)$ |

三根轴各自代表一组测量基：$z$ 轴两端是计算基 $\{|0\rangle, |1\rangle\}$，$x$ 轴两端是 $\{|+\rangle, |-\rangle\}$，$y$ 轴两端是 $\{|+i\rangle, |-i\rangle\}$。**态在某一根轴上的「投影」，恰好是它在该基下测量时的概率差**——例如 $z$ 轴分量 $z = p(0) - p(1)$，$x$ 轴分量 $x = p(+) - p(-)$。这个「投影 = 可观测量的期望值」的读法，是下一节《测量与基的选择》的钥匙，我们先在这里埋下。<span class="marginnote">$|+i\rangle$、$|-i\rangle$ 这两个态常常被忽略，因为它们的测量概率与 $|+\rangle$、$|-\rangle$ 在计算基下一模一样，区别藏在相位里。要真正看见 $y$ 轴，必须在 $Y$ 基下测量——这也提醒我们：只看一种基，布洛赫球的「纵深」是看不见的。</span>

![布洛赫球上的坐标轴、地标与布洛赫矢量](/images/quantum-computing/bloch-sphere-representation-1.svg)

## 3 布洛赫矢量与密度算符：从球面到球体

纯态都在球面上。但单量子比特更一般的状态是**混合态**（见第一篇《密度算符》），它不在球面上，而在**球体内部**。把两者统一起来的工具是密度算符。

记 $\boldsymbol{\sigma} = (X, Y, Z)$ 为 Pauli 算符向量。**任意单量子比特密度算符都可以唯一地写成**

$$
\rho = \frac{I + \boldsymbol{r}\cdot\boldsymbol{\sigma}}{2} = \frac12\begin{pmatrix} 1 + z & x - iy \\ x + iy & 1 - z \end{pmatrix}
$$

其中 $\boldsymbol{r} = (x, y, z)$ 叫**布洛赫矢量（Bloch vector）**。纯态满足 $|\boldsymbol{r}| = 1$（在球面上），混合态满足 $|\boldsymbol{r}| < 1$（在球体内部），完全混合态 $\rho = I/2$ 对应球心 $\boldsymbol{r} = \boldsymbol{0}$。

**重点：** 布洛赫矢量还有一个等价读法——它是三个 Pauli 算符的期望值向量：

$$
\boldsymbol{r} = \big(\langle X\rangle, \langle Y\rangle, \langle Z\rangle\big), \qquad \langle\sigma_i\rangle = \mathrm{Tr}(\rho\,\sigma_i)
$$

这句话的分量含义极其深刻：**想要知道一个量子比特的布洛赫矢量，不需要任何玄学，只需要制备很多份相同的态，分别在三根轴上测量，数频率**。测量（下一节的主题）在几何上就是把球面上的点「压」到某根轴上，期望值就是压出来的读数。于是布洛赫球把「态」这个抽象对象，变成了**可以实验探测的三维坐标**。<span class="marginnote">密度算符是线性代数里「迹为正一」的矩阵；布洛赫矢量把它参数化成「球内的一个点」。这套「几何化」的思路在量子机器学习里被直接复用：第十一篇的「角度编码」就是把经典数据 $x$ 映射到布洛赫球上的旋转角，让数据本身变成球面上的点，再用干涉来区分。</span>

## 4 单比特门：球面上的旋转

布洛赫球最漂亮的用法是：**所有单比特量子门，都是球面上的旋转**。设 $\hat{n}$ 是单位方向向量，绕 $\hat{n}$ 轴转 $\theta$ 角的旋转算符是

$$
R_{\hat{n}}(\theta) = e^{-i\theta\,\hat{n}\cdot\boldsymbol{\sigma}/2} = \cos\frac{\theta}{2}\,I - i\sin\frac{\theta}{2}\,(\hat{n}\cdot\boldsymbol{\sigma})
$$

（第二个等号由 Pauli 算符平方等于 $I$ 推出，见第一篇《厄米算符与幺正算符》。）几个重要的门都是它的特例：

- **$X$ 门** = 绕 $x$ 轴转 $\pi$：$|0\rangle \leftrightarrow |1\rangle$，赤道上 $|+\rangle, |-\rangle$ 不动。这就是经典「比特翻转」。
- **$Z$ 门** = 绕 $z$ 轴转 $\pi$：$|0\rangle$ 不动，$|1\rangle$ 变号；$|+\rangle \leftrightarrow |-\rangle$。它只改相对相位，不改计算基概率。
- **Hadamard 门 $H$** = 绕 $(\hat{x}+\hat{z})/\sqrt2$ 轴转 $\pi$：把北极送到赤道、把 $|0\rangle$ 变成 $|+\rangle$。
- **相位门 $S$、$T$** = 绕 $z$ 轴分别转 $\pi/2$ 与 $\pi/4$，是后续第三篇《相位门》的主角。

**辨析｜易错点：** 「门 = 旋转」不等于「所有旋转都是 $X,Y,Z$ 这种简单门」。任意单比特幺正门确实对应球面上的某个旋转（欧拉定理），但反过来写成一个旋转轴加一个角度，往往需要组合好几个基本门。这是第三篇《旋转门与任意单比特门分解》的内容——现在只需记住**几何图景**：门的复合 = 球的连续转动，转错了角度，态就会偏到不该去的地方。这也是后面变分算法里「参数就是旋转角」的由来。<span class="marginnote">旋转算符 $R_{\hat n}(\theta)$ 在数学上生成三维旋转群 SO(3)；而量子态被转 $2\pi$ 回到负号、转 $4\pi$ 才还原，对应的是 SU(2) 到 SO(3) 的 $2:1$ 覆盖——这是第二级《抽象代数》里群论与量子力学交界的经典例子。</span>

## 5 公式解析：从密度算符到布洛赫矢量

把上面的核心公式 $\rho = (I + \boldsymbol{r}\cdot\boldsymbol{\sigma})/2$ 完整推导一遍，看清楚 $\boldsymbol{r}$ 的三个分量是怎么从 $\theta, \phi$ 里冒出来的。

**第一步，写出纯态的密度算符。** 对 $|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$，有

$$
\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} \cos^2\frac{\theta}{2} & \cos\frac{\theta}{2}\sin\frac{\theta}{2}\,e^{-i\phi} \\ \cos\frac{\theta}{2}\sin\frac{\theta}{2}\,e^{i\phi} & \sin^2\frac{\theta}{2} \end{pmatrix}
$$

**第二步，把公式右端展开。** 代入 $X = \begin{pmatrix}0&1\\1&0\end{pmatrix},\ Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix},\ Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$：

$$
\frac{I + \boldsymbol{r}\cdot\boldsymbol{\sigma}}{2} = \frac12\begin{pmatrix} 1 + z & x - iy \\ x + iy & 1 - z \end{pmatrix}
$$

**第三步，逐项比对，解出 $\boldsymbol{r}$。** 对角元相等：

$$
\frac{1+z}{2} = \cos^2\frac{\theta}{2} = \frac{1 + \cos\theta}{2} \ \Longrightarrow\ z = \cos\theta
$$

非对角元相等（取实部与虚部）：

$$
\frac{x + iy}{2} = \cos\frac{\theta}{2}\sin\frac{\theta}{2}\,e^{i\phi} = \frac{\sin\theta}{2}\,e^{i\phi}
\ \Longrightarrow\ x = \sin\theta\cos\phi, \quad y = \sin\theta\sin\phi
$$

于是 $\boldsymbol{r} = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)$，恰好是第一节的球坐标翻译——**推导闭合**。

**第四步，验证期望值读法。** 用迹的性质 $\mathrm{Tr}(I) = 2,\ \mathrm{Tr}(\sigma_i\sigma_j) = 2\delta_{ij}$：

$$
\mathrm{Tr}(\rho\,\sigma_k) = \frac12\mathrm{Tr}\Big((I + \boldsymbol{r}\cdot\boldsymbol{\sigma})\sigma_k\Big) = \frac12\Big(\mathrm{Tr}(\sigma_k) + \sum_i r_i\,\mathrm{Tr}(\sigma_i\sigma_k)\Big) = r_k
$$

（中间用了 $\mathrm{Tr}(\sigma_k) = 0$。）所以**布洛赫矢量 = 期望值向量**，公式与实验读数对上了。

## 6 小结

- 单量子比特纯态 $\leftrightarrow$ **单位球面**上一点：$\boldsymbol{r} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$；$|0\rangle$ 在北、$|1\rangle$ 在南、$\{|\pm\rangle\}$ 在 $x$ 轴、$\{|\pm i\rangle\}$ 在 $y$ 轴。
- 混合态落在**球体内部**：$\rho = (I + \boldsymbol{r}\cdot\boldsymbol{\sigma})/2$，$|\boldsymbol{r}| < 1$；球心是完全混合态 $I/2$。
- 布洛赫矢量可用实验读出：**$\boldsymbol{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$**，在三个基下测量取期望即可。
- 单比特门 = **球面旋转**：$X$ 是绕 $x$ 轴转 $\pi$，$Z$ 是绕 $z$ 轴转 $\pi$，$H$ 是绕 $(\hat{x}+\hat{z})/\sqrt2$ 轴转 $\pi$，旋转算符 $R_{\hat n}(\theta) = e^{-i\theta\hat n\cdot\boldsymbol{\sigma}/2}$。
- 球只对**单个**量子比特成立：两个及以上量子比特的态空间维数随 $n$ 指数增长，任何球面都装不下——那是纠缠的世界。

在下一节，我们让测量正式登场：单比特量子态的**测量与基的选择**——你会看到「在布洛赫球上把点压到某根轴上」这个动作如何变成数学，以及为什么「选择哪一组基」是量子信息里一个需要慎重对待的权力。
