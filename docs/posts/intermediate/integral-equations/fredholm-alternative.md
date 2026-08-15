---
title: Fredholm 择一定理
date: 2026-08-07
---

# Fredholm 择一定理

<div class="epigraph">
<p>要么方程对一切右端都恰有一解，要么齐次方程有多解而右端被附加上界——二者必居其一，没有中间态。</p>
<footer>—— 埃里克 · 伊瓦尔 · 弗雷德霍姆（Erik Ivar Fredholm）</footer>
</div>

<div class="article-byline">
<p>第二级 · 积分方程 ｜ R. Kress《Linear Integral Equations》 第四章 ｜ 2026-08-07</p>
</div>

## 为什么是「择一」

在退化核那一节，我们看见解的命运完全由行列式 $D(\lambda)$ 主宰：$D \neq 0$ 则唯一可解，$D = 0$ 则要么无解要么多解，且可解性由伴随方程把关。Fredholm 的惊人之举，是把这套**有限维直觉**推广到了任意连续核：他造出一个无穷阶行列式 $D(\lambda)$（后人称 **Fredholm 行列式**），并证明它对 $\lambda$ 是**整函数**——一个无穷阶的「多项式」，除了离散的零点外处处不为零。<span class="marginnote">Fredholm 1903 年的论文把核写成矩形网格上的有限阶行列式，再令网格加密取极限，得到整函数 $D(\lambda)$ 与一阶子式 $D(x,t;\lambda)$。这个「先离散、再取极限」的做法，比泛函分析里的 Riesz 理论早了近二十年，是分析学的杰作。</span>

**Fredholm 择一定理（Fredholm alternative）**：对第二类方程

$$y(x) = f(x) + \lambda \int_{a}^{b} K(x,t)\, y(t)\, dt$$

对每个固定的 $\lambda$，下列两种情形**有且仅有一种**成立：

- **第一择一**：齐次方程 $y = \lambda K y$ 只有零解，此时非齐次方程对**任意** $f$ 有**唯一**解，且解连续依赖 $f$。
- **第二择一**：齐次方程有 $n \ge 1$ 个线性无关的非零解（$n$ 有限），此时非齐次方程**并非对所有 $f$ 可解**；它可解当且仅当 $f$ 满足恰好 $n$ 个**正交条件**（与伴随齐次方程的解正交）。

没有第三种可能：不存在「齐次方程有非零解、而非齐次方程却对所有 $f$ 有唯一解」的 $\lambda$。这就是「择一」二字的全部含义。

从物理的角度读它：把 $f$ 想成「外加驱动」，齐次解想成「系统的固有模式」。择一说的是——**驱动要么唯一地推动系统，要么撞上某个固有模式的「共振频率」，此时驱动必须避开固有模式的形状（与之正交）系统才有响应**。这种「共振条件」的语言，与《常微分方程》里参数共振、与后续《偏微分方程》里特征值问题的可解性条件，是同一个物理直觉的三种写法。

数学上「择一」的证明最终依赖 Riesz 理论：$I - \lambda K$ 是恒等算子与紧算子之差，其核空间有限维、值域闭、指标为零——**这三条合起来，就是「择一」在无穷维成立的完整理由**。

## 1 有限维的影子：矩阵的 Fredholm 择一

先看 $n \times n$ 矩阵方程 $\boldsymbol{y} = \boldsymbol{f} + \lambda \boldsymbol{A}\boldsymbol{y}$，即 $(\boldsymbol{I} - \lambda\boldsymbol{A})\boldsymbol{y} = \boldsymbol{f}$。线性代数的基本事实是：

- 若 $\det(\boldsymbol{I} - \lambda\boldsymbol{A}) \neq 0$，方程对一切 $\boldsymbol{f}$ 有唯一解。
- 若行列式为 0，齐次方程有 $n \ge 1$ 个线性无关解；此时 $\boldsymbol{f}$ 必须与齐次**转置**方程的所有解正交，方程才可解。

后者由**秩-零化度定理**（rank–nullity）保证：矩阵与其转置有相同的秩，所以「列空间的正交补」的维数恰好等于「左零空间的维数」。<span class="marginnote">择一在有限维是线性代数教科书第一学期的内容；Fredholm 把它搬进无穷维，关键在于积分算子与矩阵共享两个本质属性：<strong>紧性</strong>（有界集被映成相对紧集）与<strong>相伴性</strong>（积分算子有自然的伴随算子）。这两条正是 Riesz 后来抽象出的 Riesz 理论。</span>

**Fredholm 择一定理的全部内容，就是这句话的无穷维版本。** 积分算子 $K$ 是紧算子，$I - \lambda K$ 是「恒等算子加紧算子」，这类算子的谱理论由 Riesz 理论完整刻画：核空间有限维、值域闭、指标为零（核维数等于余核维数）。

## 2 第一择一：解算子与预解核

第一择一时，算子 $I - \lambda K$ 有界可逆。解可显式写成

$$y(x) = f(x) + \lambda \int_{a}^{b} \Gamma(x,t;\lambda)\, f(t)\, dt$$

其中 **$\Gamma$ 是预解核（resolvent kernel）**。Fredholm 给出它的级数构造：

$$\Gamma(x,t;\lambda) = \frac{D(x,t;\lambda)}{D(\lambda)}$$

分母是 Fredholm 行列式 $D(\lambda) = \sum_{m=0}^\infty \frac{(-1)^m}{m!} A_m$，分子 $D(x,t;\lambda)$ 是「一阶子式」的整函数级数。

这个比值形式的威力在上一节已经见过：它把 Neumann 级数从「小 $\lambda$ 才收敛」升级为「除特征值外处处解析」。**特征值正是预解核的极点，也是 $D(\lambda)$ 的零点**——这印证了退化核情形的结论：解对 $\lambda$ 的奇异性完全由特征值决定。

第一择一还隐含一个工程上至关重要的性质：**解连续依赖数据 $f$**。若 $f$ 只是微微扰动，$\Gamma$ 是有界函数，解的变化量被 $|\lambda|\int|\Gamma|\,\cdot$ 控制住——这正是「良定问题」要求的第三条：存在、唯一、稳定。反观下一课会遇到的**第一类方程**，解对 $f$ 的依赖普遍不连续，第一择一的这层「稳定性红利」在第二类方程身上才存在。<span class="marginnote">从第二级《泛函分析》的视角，$\lambda \mapsto (I - \lambda K)^{-1}$ 是预解式的亚纯延拓，极点集就是谱。这里我们不引入 Banach 空间的语言，只保留「分母整函数、零点即特征值」的核心图景。</span>

## 3 第二择一：正交条件与伴随方程

第二择一时，设齐次方程 $y = \lambda K y$ 的线性无关解为 $y_1, \dots, y_n$。同时考虑**伴随齐次方程**

$$\psi(x) = \lambda \int_{a}^{b} K(t,x)\, \psi(t)\, dt$$

其中核被**转置**：$K(x,t) \to K(t,x)$。设其线性无关解为 $\psi_1, \dots, \psi_n$——维度与 $y_k$ 相同，同为 $n$（这是择一定理的核心结论之一）。<span class="marginnote">核的转置对应伴随算子：$\langle K\psi, y\rangle = \langle \psi, K^* y\rangle$，其中 $K^*\psi = \int K(t,x)\psi(t)dt$。当核<strong>对称</strong>时，$K = K^*$，两套特征函数重合——这就是下一课 Hilbert–Schmidt 理论的入场券。</span>

**Fredholm 第二定理**：非齐次方程可解，当且仅当

$$\int_{a}^{b} f(t)\, \psi_k(t)\, dt = 0, \qquad k = 1, \dots, n$$

即自由项 $f$ 与伴随齐次方程**每一个**解正交。$n$ 个条件对应 $n$ 个维度的障碍；条件满足时解存在但**不唯一**（任意加上齐次方程的解仍为解），解集是一个 $n$ 维仿射空间。

**辨析｜易错点：** 正交条件用的是**伴随方程** $\psi$ 的解，不是原齐次方程的解 $y_k$。核不对称时两者完全不同。很多初学者用 $\int f y_k = 0$ 去检验，在非对称核下会得到错误结论。只有对称核下 $y_k = \psi_k$，两个检验才等价。

**第二择一的完整例子**：延续上一节的 $[0,1]$ 上核 $K(x,t) = xt$ 的方程，此时特征值 $\lambda = 3$（$n=1$）。非齐次方程 $y = f + 3\int_0^1 xt\, y(t)\,dt$ 是否可解？伴随齐次方程 $\psi(x) = 3\int_0^1 tx\, \psi(t)\,dt$ 的解是 $\psi(x) = cx$（与 $y(x) = cx$ 相同，因为核对称），于是正交条件为 $\int_0^1 t\, f(t)\,dt = 0$。取 $f(x) = x$，检验：$\int_0^1 t\cdot t\,dt = 1/3 \neq 0$，不可解——确实解不出来；取 $f(x) = x - 1/2$，检验：$\int_0^1 t(t - 1/2)dt = 1/3 - 1/4 \neq 0$，仍不可解；只有取 $f(x) = x^2 - 1/4$（它在 $L^2$ 中与 $x$ 正交）才可解，且解不唯一。这个例子把抽象的「择一」落成了可验算的算术。

## 4 公式解析：正交条件从哪里来

把「$f$ 要满足 $n$ 个正交条件」这个结论追溯到能量恒等式，看它的每一步：

$$
\langle f, \psi_k\rangle = \int_{a}^{b} f(t)\, \psi_k(t)\, dt = 0, \qquad k = 1, \dots, n
$$

- **第一步，两边对 $\psi_k$ 取内积**：设 $y = f + \lambda K y$，两边与 $\psi_k$ 做 $L^2$ 内积，得 $\langle y, \psi_k\rangle = \langle f, \psi_k\rangle + \lambda\langle K y, \psi_k\rangle$。
- **第二步，把积分算子转手**：把 $K$ 从 $y$ 身上挪到 $\psi_k$ 身上——利用伴随定义 $\langle K y, \psi_k\rangle = \langle y, K^*\psi_k\rangle = \langle y, \lambda\psi_k\rangle$，这里第二步用了 $\psi_k = \lambda K^*\psi_k$ 这个伴随特征方程。
- **第三步，两项相消**：于是 $\langle y, \psi_k\rangle = \langle f, \psi_k\rangle + \lambda \cdot \langle y, \lambda^{-1}\psi_k\rangle$ 中，左右两边的 $\langle y, \psi_k\rangle$ 相互抵消（注意 $\lambda K^*\psi_k = \psi_k$），得到 $0 = \langle f, \psi_k\rangle$。**解 $y$ 从方程中彻底消失，剩下的约束只落在 $f$ 身上。**
- **第四步，反方向论证**：若这 $n$ 个正交条件全满足，则 $f$ 落在算子值域里，方程可解。**「可解 ⟺ 与伴随核正交」**，就是 Fredholm 第二定理的全部内容——它也是《线性代数》里「$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$ 可解 ⟺ $\boldsymbol{b}$ 与 $\boldsymbol{A}^\top$ 的零空间正交」的无穷维翻版。

这套「内积移手」的技巧值得单独记住：**凡是想从「解满足的等式」推出「数据必须满足的约束」，就取内积、把算子转手、让解相消**。它在第二级《泛函分析》的伴随算子理论、在数学物理方法的 Green 函数方法里反复出现，是分析学的标准动作。

## 5 Fredholm 三定理与谱的整体图景

Fredholm 1903 年的理论通常被拆成三个定理，把择一定理逐层夯实：

- **Fredholm 第一定理**：预解核是 $\lambda$ 的**亚纯函数**，即 $\Gamma(x,t;\lambda) = D(x,t;\lambda)/D(\lambda)$，其中 $D$ 是整函数（Fredholm 行列式），$D(x,t;\lambda)$ 也是整函数。$D(\lambda)$ 的零点集就是特征值集。
- **Fredholm 第二定理**：若 $\lambda_0$ 使 $D(\lambda_0) = 0$，则齐次方程 $y = \lambda_0 K y$ 与伴随齐次方程 $\psi = \lambda_0 K^* \psi$ **各有同样多个**线性无关解（设为 $n$ 个）。
- **Fredholm 第三定理**：非齐次方程在 $\lambda_0$ 处可解，当且仅当 $f$ 与伴随齐次方程的这 $n$ 个解全部正交。

三定理合起来，就给出积分算子谱的完整画像：

**特征值集合是离散的。** 整函数 $D(\lambda)$ 的零点没有有限聚点，因此特征值在复平面内至多可数，且任何有界区域内只有有限多个。这与有限维矩阵「特征值至多 $n$ 个」是同一精神的无穷维延续，却与微分算子（如 $-d^2/dx^2$ 的谱可能连续）形成鲜明对照。<span class="marginnote">「积分算子的谱离散、微分算子的谱可能连续」是分析数学的核心景观：积分是光滑化算子，天然把信息压缩成离散模式；微分是锐化算子，保留连续谱。理解这点，后续学第二级《泛函分析》的谱理论时会顺畅得多。</span>

**指标为零。** 在特征值处，「无解的障碍方向数」恰好等于「解的自由度方向数」，都是 $n$。用线性代数的语言说，核空间与余核空间同维，Fredholm 算子的**指标** $\dim \ker(I - \lambda K) - \dim \operatorname{coker}(I - \lambda K) = 0$。这是 Fredholm 算子区别于一般有界算子的关键不变量，也是「择一」在更深层的根源：障碍与自由度总是成对出现，从不单飞。

历史地看，Fredholm 的这些结论早于一般算子的谱理论。三十年后 Riesz 在 Banach 空间里重述它们，抽象出「紧算子的 Riesz 理论」，Fredholm 行列式退居后台，指标与零化度成为主角。**但谱的离散性、择一的成对性这两条本质，从矩阵到积分算子再到一般紧算子，始终没变**——它正是下一节对称核理论的出发点。

## 6 小结

- **Fredholm 择一定理**：对每个 $\lambda$，要么齐次方程只有零解且非齐次方程对一切 $f$ 唯一可解，要么齐次方程有 $n$ 个线性无关解且非齐次方程受 $n$ 个正交条件约束——**二者必居其一**。
- 第一择一时解可写成**预解核**形式 $y = f + \lambda\int\Gamma f$，其中 $\Gamma = D(x,t;\lambda)/D(\lambda)$，$D$ 是整函数 **Fredholm 行列式**。
- 第二择一时**伴随齐次方程**同样有 $n$ 个解，可解条件是与它们全部正交；满足条件后解不唯一。
- 正交条件来自**内积移手**：$\langle K y,\psi\rangle = \langle y, K^*\psi\rangle$，解的约束借此全部转嫁给自由项 $f$