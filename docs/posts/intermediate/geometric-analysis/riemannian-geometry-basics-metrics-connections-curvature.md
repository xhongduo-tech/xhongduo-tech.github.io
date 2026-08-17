---
title: 黎曼几何基础复习（度量、联络、曲率张量与 Bianchi 恒等式）
date: 2026-08-07
---

# 黎曼几何基础复习（度量、联络、曲率张量与 Bianchi 恒等式）

<div class="epigraph">
<p>「在这篇论文中，我要建立一个关于度量的完全一般的概念，单个的具体情形只是它的特殊例子而已。」</p>
<footer>—— 伯恩哈德 · 黎曼（Bernhard Riemann）《关于几何学的基本假设》（Über die Hypothesen, welche der Geometrie zu Grunde liegen, 1854）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Jost《Riemannian Geometry and Geometric Analysis》基础章 ｜ Peter Li《Geometric Analysis》Ch. 1 ｜ 2026-08-07</p>
</div>

## 为什么从黎曼几何复习开始

几何分析这个词，是把「几何」与「分析」焊在一起的一门学科：它用偏微分方程的武器去研究流形，又用曲率的几何约束去指挥方程的解。要进入这门学科，第一道门槛是黎曼几何的基础语言——度量、联络、曲率。**它们不是预备知识，而是几何分析的一切算子、一切估计、一切定理所依附的坐标系。**

从本博客的学习主线看，你已经在第二级《微分几何》里见过曲面上的第一基本形式与曲率，也学过向量微积分里的梯度、散度、旋度。这一篇把那些概念从「嵌入在欧氏空间中的曲面」升级为「内在定义的流形」，并补上三个此后每一篇都要用的工具：度量张量 $g$、Levi-Civita 联络 $\nabla$、黎曼曲率张量 $R$。它们与线性代数中的张量、PDE 中的算子一道，构成几何分析的语法。

<span class="marginnote">黎曼 1854 年的就职演说只有短短十几页，却奠定了现代微分几何与广义相对论的度量观。他提出：曲率不必来自「弯曲进去的环境空间」，而可以完全由流形内部的弧长微元 $ds^2 = g_{ij}dx^i dx^j$ 决定——这就是内蕴几何。</span>

## 1 黎曼度量：如何量长度、角度与体积

**黎曼度量（Riemannian metric）**是光滑流形 $M$ 上每个切空间 $T_pM$ 的一个正定内积 $g_p$，随点光滑变化。在局部坐标 $\{x^i\}$ 下写成分量 $g_{ij} = g(\partial_i, \partial_j)$，弧长微元为

$$ds^2 = g_{ij} \, dx^i dx^j$$

有了度量，一切度量量随之而来：向量 $v,w$ 的夹角由 $g(v,w)$ 给出；曲线 $\gamma$ 的长度是 $\int \sqrt{g(\dot\gamma,\dot\gamma)}\,dt$；体积元为 $dV = \sqrt{\det g}\, dx^1\cdots dx^n$。<span class="marginnote">「度量 = 给每个切空间配一个正定内积」这句话的分量式是 $g_{ij}$ 对称、正定、随点光滑。球面上取 $g = r^2(d\theta^2 + \sin^2\theta\,d\varphi^2)$，平面取 $g = dx^2+dy^2$，闵氏时空是负定的伪度量 $g = -c^2dt^2+dx^2+dy^2+dz^2$——后者正是广义相对论的舞台，见第二级《微分几何》与第四级《广义相对论》。</span>

**为什么这比「嵌入曲面」更一般**：高斯的《论曲面的一般研究》（1827）里，曲面的第一基本形式 $E\,du^2+2F\,dudv+G\,dv^2$ 就是 $g_{ij}$ 的雏形；但黎曼让 $g$ 成为流形自身的结构，不需要任何外嵌。这正是现代几何「内蕴」哲学的起点，也是几何分析里一切全局定理（而非局部坐标计算）的立足点。

## 2 Levi-Civita 联络：沿切向量求导的规则

有了度量，才能定义一个「好」的求导规则——**Levi-Civita 联络（Levi-Civita connection）**：它是满足下面三条性质的唯一联络 $\nabla$：

**度量相容（metric-compatible）**：$\nabla_X g = 0$，即平行移动保内积；
**无挠（torsion-free）**：$\nabla_X Y - \nabla_Y X = [X,Y]$；
**线性与 Leibniz**：$\nabla_X(fY) = (Xf)Y + f\nabla_X Y$。

在局部坐标下，联络由**Christoffel 符号（Christoffel symbols）** $\Gamma^k_{ij}$ 确定：$\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\partial_k$，且有显式公式

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl}\big(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\big)$$

联络把「方向导数」从欧氏空间移植到流形：没有它，向量场之间不能比较、不能求导，平行移动、测地线、曲率都无从谈起。<span class="marginnote">用曲面上「投影回切平面」定义协变导数（陈维桓《微分几何》§8 的做法）得到的就是嵌入诱导的 Levi-Civita 联络；而流形上的联络纯靠三条公理定义，不依赖环境空间。两者的差别——公理化 vs 投影构造——是内蕴几何与子流形几何的分水岭。</span>

## 3 黎曼曲率张量与 Bianchi 恒等式

**黎曼曲率张量（Riemann curvature tensor）**量化「曲率」：对向量场 $X,Y,Z$ 定义

$$R(X,Y)Z = \nabla_X\nabla_Y Z - \nabla_Y\nabla_X Z - \nabla_{[X,Y]}Z$$

在坐标下记 $R(\partial_i,\partial_j)\partial_k = R^{\,l}_{\;kij}\partial_l$。它的直觉是：把 $Z$ 沿一个「平行四边形」平行移动一圈后，回来的 $Z$ 与原来之差正比于 $R$——**曲率测量的是平行移动的不可交换性**。曲率张量满足关键的对称性与两组**Bianchi 恒等式（Bianchi identities）**：

$$R_{ijkl} = -R_{jikl} = -R_{ijlk}, \qquad R_{ijkl} = R_{klij}$$

$$R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0 \quad\text{（第一 Bianchi）}$$

第一 Bianchi 恒等式相当于「三个向量的环绕和为零」，它刻画了为什么曲率是一个「三阶反对称、两两反对称」的张量；第二 Bianchi 恒等式则是曲率的「微分恒等式」，后面会看到它直接给出 Einstein 场方程的自动守恒。

## 4 公式解析：Bianchi 第二恒等式

第二 Bianchi 恒等式的分量形式是**几何分析里最重要的计算工具之一**：

$$\nabla_l R^{\,m}_{\;ijk} + \nabla_i R^{\,m}_{\;jlk} + \nabla_j R^{\,m}_{\;lik} = 0$$

逐项拆解：

- **第一步，看懂指标**：$R^{\,m}_{\;ijk}$ 是曲率张量的分量，四个指标 $i,j,k,l,m$ 各不相同；$\nabla_l$ 是沿 $\partial_l$ 的协变导数。
- **第二步，看懂循环**：等号左边是对指标 $(l,i,j)$ 作**循环求和**——$(l,i,j) \to (i,j,l) \to (j,l,i)$，也就是轮流把求导指标放第一位，其余两个跟随。这个模式与第一 Bianchi 恒等式里对 $(X,Y,Z)$ 的循环完全同构。
- **第三步，为什么成立**：联络是无挠的，这使 $\nabla_X\nabla_Y - \nabla_Y\nabla_X$ 只差一个 $R$ 项；把三个这样的「换序差」相加，括号内的项两两抵消，只剩零。它是一个纯代数的恒等式，对任何黎曼流形都对。

**重点：第二 Bianchi 恒等式经一次缩并（contract）得到 Einstein 场方程的自动满足。** 令 $i=l$ 求和（用度量把 $m$ 拉下来）即得 $\nabla_j R_{jk} - \frac{1}{2}\nabla_k R = 0$，写成 $\nabla^k\big(R_{jk} - \tfrac12 R g_{jk}\big) = 0$。这解释了为什么广义相对论里物质守恒 $\nabla^k T_{jk}=0$ 是几何自动保证的——它就是 Bianchi 恒等式的化身。<span class="marginnote">Einstein 场方程 $G_{jk} = 8\pi T_{jk}$ 中，爱因斯坦张量 $G = Ric - \tfrac12 R g$ 正是被构造为「协变散度恒为零」的张量，目的就是与 Bianchi 恒等式相容。这一层的因果链：Bianchi（几何恒等式）→ 爱因斯坦张量守恒 → 物质守恒。</span>

## 5 缩并：Ricci 张量与标量曲率

曲率张量太大（$n$ 维有 $\frac{n^2(n^2-1)}{12}$ 个独立分量），几何分析更常用它的缩并产物——**Ricci 张量（Ricci tensor）**与**标量曲率（scalar curvature）**：

$$R_{jk} = R^{\,i}_{\;jik}, \qquad R = g^{jk}R_{jk}$$

Ricci 张量有直观的几何意义：**沿一个方向 $v$ 的 Ricci 曲率 $\operatorname{Ric}(v,v)$ 度量的是「以 $v$ 为方向的截面在近旁收缩（或膨胀）的速度」**。标量曲率则把全部截面曲率再平均一次，是最粗的曲率不变量。下表给出各层级曲率的角色：

| 对象 | 定义 | 几何意义 | 几何分析中的用途 |
| --- | --- | --- | --- |
| 截面曲率 $K(\sigma)$ | $\frac{R(X,Y,Y,X)}{g(X,X)g(Y,Y)-g(X,Y)^2}$ | 平面的弯曲程度 | Rauch、Toponogov 比较定理 |
| Ricci 张量 $\operatorname{Ric}$ | 截面曲率的方向平均 | 体积沿方向的收缩率 | Bonnet–Myers、Li–Yau 估计 |
| 标量曲率 $R$ | $\operatorname{Ric}$ 的迹 | 体积元整体收缩率 | Yamabe 问题、正质量定理 |
| 曲率张量 $R$ | 平行移动不可交换 | 全部局部几何 | 各向同性、Ricci 流曲率演化 |

**辨析｜易错点：** 截面曲率决定 Ricci 与标量曲率，但反过来不成立——三个 $n=2$ 曲面上曲率只有一个数，而 $n \ge 4$ 时截面曲率含的信息远比 Ricci 多。<span class="marginnote">在 2 维，Ricci 张量与度量成比例（$\operatorname{Ric} = \tfrac12 R g$），曲率信息退化为一个标量——这正是第一级《微分几何》里「高斯曲率」的情形。进入 3 维，Ricci 张量才拥有「独立的形状」，这是 Ricci 流在三维能大放异彩的代数基础。</span>

**高斯绝妙定理（Theorema Egregium）**：截面曲率是内蕴不变量——任意两个局部等距的流形在同一点、同一平面的截面曲率相等。高斯 1827 年以曲面证明它：曲面上的高斯曲率完全由第一基本形式（即度量 $g$）决定，与外嵌无关。<span class="marginnote">「绝妙」（egregium）正是高斯的感叹：从二维曲面的计算中，他意识到曲率是「曲面自己的」而非「环境空间的」。黎曼把它推广到任意维，几何分析的全部工作都建立在「曲率是内蕴量」之上。</span>

**平直判别（flatness criterion）**：黎曼曲率张量恒为零 $R\equiv 0$ 当且仅当流形局部等距于欧氏空间（处处可找局部坐标使 $g_{ij}=\delta_{ij}$）。这是「曲率为零 ⇔ 平直」的严格表述，也是三级曲率信息的最强区分点：$R=0$ 是「全平直」，$\operatorname{Ric}=0$ 是「Einstein 真空」，标量曲率 $R_{scalar}=0$ 是「零标量曲率」——三者层级不同，切勿混淆。

### 5.1 记号与约定速查

几何分析文献的记号各有门派，本专题统一采用以下约定，读外部文献时务必先核对符号：

| 记号 | 本专题含义 | 常见歧义 |
| --- | --- | --- |
| $\Delta$ | 正的 Laplace–Beltrami：$-\operatorname{div}\nabla$ | 几何文献常取负号 |
| $\nabla$ | Levi-Civita 联络 | 亦指梯度（视上下文） |
| $\operatorname{Ric}$ | Ricci 张量 | 与 $\operatorname{Ric}(v,v)$ 标量区分 |
| $R$ | 标量曲率 | 与曲率张量 $R$ 冲突，后者常写作 $\operatorname{Riem}$ |

Einstein 求和约定默认使用：重复的上下指标自动求和。练习读写张量分量时，建议先核对「哪些指标是几何的、哪些是分析学的」，这正是几何分析初学者的第一个思维习惯。

**一个贯穿全局的结论**：度量 $g$、联络 $\nabla$、曲率 $R$ 是「一个流形上最基础的几何数据」，而 Bianchi 恒等式是它们之间唯一的微分约束。之后的每一篇——测地线的第二变分、比较定理、Laplace 算子、热核的 Li–Yau 估计——都将反复用到本篇的记号与恒等式。

## 6 小结

- **黎曼度量** $g$：每个切空间的正定内积，分量 $g_{ij}$，给出长度、角度与体积元 $dV = \sqrt{\det g}\,dx$。
- **Levi-Civita 联络**：唯一满足度量相容 + 无挠的求导规则，分量是 Christoffel 符号 $\Gamma^k_{ij}$。
- **曲率张量** $R(X,Y)Z = \nabla_X\nabla_Y Z - \nabla_Y\nabla_X Z - \nabla_{[X,Y]}Z$，测量平行移动的不可交换性。
- **两组 Bianchi 恒等式**：第一是代数恒等式，第二是微分恒等式；缩并后者即得爱因斯坦张量散度为零。
- **缩并层级**：$R \to \operatorname{Ric} \to R_{scalar}$，由细到粗，在几何分析中承担不同任务。

在下一节，我们将让「度量」做它最自然的事——沿最短路径运动。这就是**测地线与变分法**：从能量泛函的临界点出发，用指数映射与 Jacobi 场研究测地线的稳定与聚焦。
