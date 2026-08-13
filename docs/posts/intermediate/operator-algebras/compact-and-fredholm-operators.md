---
title: 紧算子与 Fredholm 算子
date: 2026-08-07
---

# 紧算子与 Fredholm 算子

<div class="epigraph">
<p>数学没有种族与地域的边界，对于数学，文化世界是一个国家。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Davidson《C\*-Algebras by Example》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从紧算子开始

无穷维空间里，矩阵的全部好处都丢失了：单位球不再紧，特征值可能消失，谱也不再离散。可是有一类算子**顽强地保留了有限维的所有美德**——它们叫**紧算子**。积分算子、Hilbert–Schmidt 算子、有限秩算子都在这面旗帜下集合，它们的谱几乎就像矩阵的谱：非零特征值离散、每个有限重、唯一聚点是 0。

紧算子的「近有限维」不是缺陷而是金矿：因为紧，所以能逼近；因为能逼近，Fredholm 才发明了用「模掉紧算子」来测量一般算子的方式——**指标（index）**。指标理论是分析学史上最深刻的洞见之一：一个整数（$\dim\ker T-\dim\mathrm{coker}\,T$）在同伦与紧扰动下纹丝不动，却决定了微分方程、拓扑与几何里无数存在性问题。这一篇就讲这两个主角：紧算子与 Fredholm 算子。

## 1 紧算子：有限维算子的极限

**紧算子（compact operator）**：$K\in B(\mathcal{H})$ 把有界集映为相对紧集；等价地，$K$ 把单位球映为紧集，等价于每个有界序列 $\{x_n\}$ 的像 $\{Kx_n\}$ 都有收敛子列。

三个例子构成紧算子家族的主干：

**有限秩算子**：$\dim\mathrm{im}\,K\lt \infty$。它的值域是有限维子空间，其闭包仍是紧集（有限维有界闭集紧）。

**积分算子**：$Kf(x)=\int_0^1 k(x,y)f(y)\,dy$，其中核 $k\in L^2([0,1]^2)$。由 $L^2$ 可积核定义的积分算子是紧的（靠 $k$ 被有限秩核逼近）。<span class="marginnote">积分方程理论是紧算子概念的源头：Fredholm 在 1903 年研究「积分方程何时有解」时，发现解的性质只由核 $k$ 的近似有限维结构决定。今天我们把他的观察总结成一句话：<strong>核积分算子都是紧的，紧算子都有 Fredholm 式的好谱</strong>。</span>

**对角线趋于零的对角算子**：$K e_n = \lambda_n e_n$，$\lambda_n\to 0$。这是紧算子的「标准型」雏形，也是理解紧算子谱的脚手架。

**辨析｜易错点：**恒等算子 $I$ 在无穷维**不是**紧的（单位球被原样映到自身，而无穷维单位球不紧）。于是「紧算子 + 非紧算子」可以很接近范数，却永远差一口气——这正是 Calkin 代数 $\mathcal{Q}(\mathcal{H})=B(\mathcal{H})/K(\mathcal{H})$ 中非零元素存在的根源。

## 2 紧算子的谱：几乎像矩阵一样

**定理（紧算子谱定理）**：设 $K$ 是 Hilbert 空间上的紧算子。则 $\sigma(K)\setminus\{0\}$ 由至多可数多个**特征值**组成，每个非零特征值的代数重数有限，且唯一可能的聚点是 $0$。<span class="marginnote">这条定理是「紧 ⇒ 近有限维」的最强体现：紧算子的非零谱完全由特征值构成，连续谱与剩余谱只剩 $0$ 这一种可能。Fredholm 选择定理/方法（Fredholm alternative）说：齐次方程 $Kx=\lambda x$（$\lambda\neq0$）要么只有零解、要么有有限维解空间——与有限维线性方程组的替代律完全平行。</span>

**Fredholm 二择一（Fredholm alternative）**：对紧算子 $K$ 与 $\lambda\neq0$，下列命题二选一：

$$(\lambda I-K)\ \text{可逆} \quad \text{或} \quad \ker(\lambda I-K)\neq\{0\}.$$

即：非零谱点若存在，必是特征值；方程 $(\lambda I-K)x=y$ 对每个 $y$ 可解，当且仅当相应的齐次方程只有零解。这是「行列式是否为零」在无穷维的全面替代。

**例子（Volterra 算子）**：$(Vf)(x)=\int_0^x f(t)\,dt$ 在 $L^2[0,1]$ 上紧且**幂零**：$\sigma(V)=\{0\}$，一个特征值都没有。它提醒我们紧算子可以谱为单点 $\{0\}$ 却不为零算子。

## 3 紧算子构成闭理想

设 $\mathcal{K}(\mathcal{H})$ 为全体紧算子。

**定理**：$\mathcal{K}(\mathcal{H})$ 是 $B(\mathcal{H})$ 中的**闭 $\ast$-理想**：对 $T\in B(\mathcal{H})$、$K\in\mathcal{K}(\mathcal{H})$，$TK,KT\in\mathcal{K}(\mathcal{H})$；$K^*\in\mathcal{K}(\mathcal{H})$；且有限秩算子在 $\mathcal{K}(\mathcal{H})$ 中稠密（Hilbert 空间上）。<span class="marginnote">「理想」意味着紧算子在任意一侧乘任意有界算子仍是紧的；「闭」意味着有限秩的极限还是紧的。于是商代数 $\mathcal{Q}(\mathcal{H})=B(\mathcal{H})/\mathcal{K}(\mathcal{H})$（Calkin 代数）良定义，而 Fredholm 算子恰是 Calkin 代数里的可逆元——这为下一节的指标定理给出最优雅的框架。</span>

**推论（Calkin 代数）**：$\pi:T\mapsto T+\mathcal{K}$ 是 $B(\mathcal{H})\to\mathcal{Q}(\mathcal{H})$ 的商映射，$\mathcal{Q}$ 是 C\*-代数。$T$ 为 Fredholm 当且仅当 $\pi(T)$ 在 $\mathcal{Q}$ 中可逆。

## 4 公式解析：Fredholm 指标

**Fredholm 算子（Fredholm operator）**：$T\in B(\mathcal{H})$ 满足 $\dim\ker T\lt \infty$ 且 $\mathrm{im}\,T$ 闭且余维有限（$\dim(\mathcal{H}/\mathrm{im}\,T)\lt \infty$）。它的**指标（index）**为

$$
\operatorname{ind}(T) = \dim\ker T - \dim\mathrm{coker}\,T, \qquad \mathrm{coker}\,T = \mathcal{H}/\mathrm{im}\,T.
$$

- **第一步，看两个维度**：$\dim\ker T$ 度量「解空间的大小」（方程 $Tx=0$ 有多少线性无关解），$\dim\mathrm{coker}\,T$ 度量「像空间缺了多少」（方程 $Tx=y$ 对多少 $y$ 无解）。两者都有限，指标才是整数。
- **第二步，看差**：指标不是笼统的「病态程度」，而是**定向的亏量**。对有限维方阵，$\dim\ker=\dim\mathrm{coker}$（秩-零化度定理），指标恒为 0。指标非零，是无穷维特有的「解的不平衡」。
- **第三步，为什么这个整数不朽**：两个稳定性定理——
  - **紧扰动不变**：$K$ 紧时，$\operatorname{ind}(T+K)=\operatorname{ind}(T)$；
  - **同伦不变**：$T_t$ 是 Fredholm 算子的连续道路时，$\operatorname{ind}(T_t)$ 为常数。<span class="marginnote">指标在巨大形变下纹丝不动，却在「解与缺解之间」传递精确的计数。这正是 Atiyah–Singer 指标定理（把分析指标等于拓扑指标）的胚胎形态：一个纯粹分析的对象，由纯粹拓扑的信息决定。</span>

**例子（单侧移位）**：右移位 $S e_n=e_{n+1}$ 满足 $\ker S=0$、$\mathrm{coker}\,S=\mathbb{C}$（缺少 $e_0$ 方向的像），故 $\operatorname{ind}(S)=-1$。左移位 $S^*$ 的指标为 $+1$。它们是「指标非零」的最小标本。

## 5 用指标看世界：Fredholm 理论的两翼

**Atkinson 定理**：$T$ 是 Fredholm 的，当且仅当 $\pi(T)$ 在 Calkin 代数中可逆，当且仅当存在 $S\in B(\mathcal{H})$ 使 $ST-I$ 与 $TS-I$ 都是紧算子。「模掉紧算子后的可逆」是 Fredholm 的现代定义，它把 Fredholm 理论完全纳入 C\*-代数框架——后面第 25 篇 K 理论里，Fredholm 算子将再次作为「基本元素」登场。<span class="marginnote">紧算子的「可忽略性」被 Atkinson 定理提升为定义：<strong>Fredholm 算子 = 模掉可忽略项后的可逆元</strong>。这种「商代数里看可逆」的眼光，会一路带到第 25 篇的 K 理论，那里可逆元/投影的等价类将被升级为 K 群。</span>

**应用（微分方程的 Fredholm 理论）**：椭圆算子（如 $\Delta$）在有界区域上配合边界条件后，常常是 Fredholm 的；其指标等于拓扑量（Euler 特征数），这解释了大范围解的结构与边界几何如何互相锁定。分析、几何、拓扑在指标这面旗子下第一次握手。

**辨析｜易错点：**Fredholm 算子要求 $\mathrm{im}\,T$ **闭**。单射且像稠密但像不闭的算子（如「谱压缩」的某些乘法算子）不是 Fredholm——「像闭」与「像稠密」是两回事，漏掉闭性条件，指标公式就失去意义。判断像闭性的标准工具是下面这条：$T$ 有闭像当且仅当 $\inf\{\|Tx\|:\|x\|=1,\,x\perp\ker T\}>0$。

## 6 例：紧算子的谱长什么样

紧算子的谱「几乎像矩阵」——用几个具体例子，把这条定理的每一条都落到实处。

**对角算子（$\lambda_n\to0$）**：$K e_n=\lambda_n e_n$，$\lambda_n\to0$。非零谱 = $\{\lambda_n\}$（特征值），唯一聚点是 0。这是「紧算子谱」的标准形态：离散特征值堆积向 0，绝不远离 0。

**Volterra 算子**：$(Vf)(x)=\int_0^x f(t)\,dt$。$V$ 紧、幂零，$\sigma(V)=\{0\}$——一个特征值都没有，但 $V\neq0$。它提醒我们：紧算子的谱可以「退化」到只剩 0，算子本身却充满信息。

**紧自伴算子（谱定理的紧版）**：$K=K^*$ 紧。则存在标准正交基使 $K=\sum \lambda_n\langle\cdot,e_n\rangle e_n$（$\lambda_n\to0$ 实数）——紧自伴算子**完全对角化**。这是第 13 篇谱定理在紧算子上的「免费」版本。

**Fredholm 二择一的例子**：解积分方程 $(\lambda I-K)f=g$（$K$ 紧，$\lambda\neq0$）：要么对每个 $g$ 有唯一解（$\lambda\notin\sigma_p(K)$），要么齐次方程有有限维非零解空间。这就是「非零谱点若存在必为特征值」的解题形态。

**为什么「非零谱 = 特征值」**：紧算子把单位球映成相对紧集，$(K-\lambda I)$（$\lambda\neq0$）在「退化方向」上有限维。直觉：紧算子在无穷维里「只有有限维的野性」，野性之外全是指标化的。

## 7 延伸：Calkin 代数——商掉紧算子看世界

Calkin 代数 $\mathcal{Q}=B(\mathcal{H})/\mathcal{K}(\mathcal{H})$ 是「模掉可忽略项」的哲学在算子世界的实现。

**本质谱（essential spectrum）**：$\sigma_{\mathrm{ess}}(T)=\sigma(\pi(T))$（$\pi$ 是商映射）。本质谱对紧扰动不变：$\sigma_{\mathrm{ess}}(T+K)=\sigma_{\mathrm{ess}}(T)$。物理上，「本质谱 = 对任何紧修正都稳定的谱」。

**本质范数**：$\|\pi(T)\|=\operatorname{dist}(T,\mathcal{K})$——$T$ 到紧算子集的距离。它度量「$T$ 有多少「不可被紧算子逼近」的成分」。

**Fredholm = Calkin 可逆**（Atkinson 定理）：$\pi(T)$ 可逆 ⟺ $T$ Fredholm。这一句把 Fredholm 理论从「维数条件」翻译成「商代数里的可逆性」，优雅而深刻。

**指标在 Calkin 里**：$\operatorname{ind}(T)=\operatorname{ind}(\pi(T))$ 良定义（紧扰动不变）。指标是定义在 $\mathcal{Q}$ 上的量，第 25 篇 K 理论将在 $\mathcal{Q}$ 上重新发现它。

**与 von Neumann 的关联**：$\mathcal{K}$ 是 $B(\mathcal{H})$ 的最小非零闭理想（可分离时）。理想理论（第 12 篇）说：商掉最小理想，剩下「本质」；von Neumann 代数（第 21 篇）则取弱闭包，抓住「极限」。两条路互补。

## 8 延伸：指标理论的谱系

$\operatorname{ind}(T)=\dim\ker-\dim\mathrm{coker}$ 是分析史上最「不变量」的整数之一，它后面站着一整条谱系。

**卷绕数预告**：对 $f\in C(\mathbb{T})$，$T_f$ 的指标 $=-\mathrm{wind}(f,0)$（第 14 篇 Toeplitz）。分析量等于拓扑量——这是指标谱系的第一层。

**Fredholm 指标 = K 理论映射**：第 25 篇里，$\operatorname{ind}:K_1(\mathcal{Q})\to K_0(\mathcal{K})=\mathbb{Z}$ 是六项正合列里的指数映射。指标不再是孤例，而是 K 理论结构的一部分。

**Atiyah–Singer 展望**：椭圆微分算子（如 $\Delta$）在紧流形上是 Fredholm 的，其指标 = 拓扑指数（Chern 类积分）。Atiyah–Singer 指标定理把这套谱系推向顶峰——分析指标永远等于拓扑指标。

**应用（偏微分方程）**：椭圆边值问题 Fredholm 性保证「解空间的维数差」有限；指标非零时，方程「解与缺解不平衡」，这是拓扑障碍的直接读数。

**一句话总结**：指标是「解数减缺解数」，但它本质上是拓扑量——从卷绕数到 K 理论到 Atiyah–Singer，一条线贯通。

## 9 小结

- **紧算子**把有界集映为相对紧集；有限秩算子、$L^2$ 核积分算子、$\lambda_n\to0$ 的对角算子都是紧的，紧算子构成闭 $\ast$-理想。
- **紧算子谱**：非零谱全是有限重特征值，唯一聚点是 $0$；Fredholm 二择一描述了非零谱点的行为。
- **Calkin 代数** $\mathcal{Q}=B/\mathcal{K}$：Fredholm 算子 = $\mathcal{Q}$ 中的可逆元（Atkinson 定理）。
- **指标** $\operatorname{ind}=\dim\ker-\dim\mathrm{coker}$ 是整数，紧扰动与同伦不变——分析对象被拓扑信息决定。
- **像闭性是 Fredholm 的隐含前提**，判断靠 $\inf\|Tx\|$