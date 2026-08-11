---
title: 等距浸入与第二基本形式
date: 2026-08-11
---

# 等距浸入与第二基本形式

<div class="epigraph">
<p>曲面究竟在空间中如何弯曲，是一回事；它自身内部如何丈量，是另一回事。绝妙定理说：后者完全由前者决定。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss），《关于曲面的一般研究》（Disquisitiones generales circa superficies curvas，1827）（意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 黎曼几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从等距浸入开始

到目前为止，我们一直用「内蕴」的眼光看流形：只有度量，不问它躺在哪里。但现实中流形常常以「嵌在大空间里」的面貌出现——球面是 $\mathbb{R}^3$ 里的球壳，曲面是 $\mathbb{R}^3$ 里的纸面，而深度学习的嵌入层把离散对象泡进欧氏向量空间。这些「外蕴」身份不是摆设：**子流形在环境空间里的弯曲方式，与它自己的内蕴度量，是两套数据，而 Gauss 的绝妙定理揭示了它们之间那条惊人的纽带。**

本节研究：给定等距浸入 $f: M^n \to \bar M^{n+k}$，如何把环境空间的导数分解成「切向部分」与「法向部分」——前者给出 $M$ 自身的联络，后者给出**第二基本形式**；再由这一分解推出 **Gauss 方程、Codazzi 方程**，最终抵达 Gauss 的绝妙定理（Theorema Egregium）：内蕴曲率被外蕴弯曲完全决定，却**不依赖**外蕴弯曲方式。

这条路线连接「从极限到大模型」主线：流形假设把数据看成低维子流形，而**流形学习本质上是在恢复第二基本形式及其曲率**；计算机图形学的曲面重建、深度网络嵌入的「各向异性」也与这套外蕴-内蕴语言同构。<span class="marginnote">先给一个意象：圆柱面在 $\mathbb{R}^3$ 里是弯的（外蕴），但把它剪开摊平，内部距离不变（内蕴平直）。圆柱面的 Gauss 曲率为零——外蕴弯、内蕴平，两者的分歧正是绝妙定理要精确化的话题。</span>

## 1 等距浸入与分解公式

**核心概念**：**等距浸入（isometric immersion）**：光滑映射 $f: M^n \to \bar M^{n+k}$，使得切映射 $df_p: T_pM \to T_{f(p)}\bar M$ 处处是单射，且对 $X,Y \in T_pM$ 有

$$
\langle df_p X, df_p Y\rangle_{\bar M} = \langle X, Y\rangle_M
$$

即「把 $M$ 的度量原样带进 $\bar M$」。若 $f$ 还是单射且像集是拓扑子空间，则称为**嵌入（embedding）**。<span class="marginnote">「浸入」允许像集自交（如 $S^1$ 的 8 字形浸入 $\mathbb{R}^2$），「嵌入」不允许。经典嵌入定理（Nash 定理）断言任何黎曼流形都能等距嵌入到高维欧氏空间——这说明内蕴几何并不缺乏外蕴表现，但外蕴表现永远只是内蕴几何的「影子」。</span>

在等距浸入下，$M$ 的切向量同时是 $\bar M$ 的切向量，且**沿 $M$ 的 Levi-Civita 联络等于环境联络的切向投影**。于是对环境向量场做导数，得到分解：

**核心结论（Gauss 公式）**：设 $\bar\nabla$ 是 $\bar M$ 的联络，$\nabla$ 是 $M$ 的联络，则对 $X,Y \in \mathfrak{X}(M)$：

$$
\bar\nabla_XY = \nabla_XY + \mathrm{II}(X,Y)
$$

其中 $\nabla_XY \in TM$ 是切向分量，而 $\mathrm{II}(X,Y)$ 是 $TM$ 的**法向正交补** $T^\perp M$ 中的向量。<span class="marginnote">直觉：在环境空间里求导，结果像一支箭；把它「拆开」——落在流形切面上的部分（内蕴导数）加上垂直穿出流形的部分（第二基本形式）。流形越弯，穿出的部分越大。</span>

## 2 第二基本形式与形状算子

**核心概念**：由 Gauss 公式定义的**第二基本形式（second fundamental form）** 是 $M$ 上取值于法丛 $T^\perp M$ 的对称双线性映射

$$
\mathrm{II}: TM \times TM \longrightarrow T^\perp M
$$

其对称性 $\mathrm{II}(X,Y) = \mathrm{II}(Y,X)$ 由联络无挠推出。<span class="marginnote">与第一基本形式（即度量）形成对照：第一基本形式管「内蕴丈量」，第二基本形式管「外蕴弯曲」。两者合起来才完整刻画子流形在环境中的几何。</span>

对单位法向量 $\eta$，把 $\mathrm{II}$ 投影到 $\eta$ 方向得到标量型：

$$
\langle \mathrm{II}(X,Y), \eta\rangle = \langle S_\eta X, Y\rangle
$$

其中 $S_\eta: TM \to TM$ 是**形状算子（shape operator / Weingarten 映射）**，它是自伴算子，特征值即主曲率。

- 平面曲线/曲面情形：$M$ 是 $\mathbb{R}^3$ 中曲面时，形状算子的特征值 $\kappa_1, \kappa_2$ 是**主曲率**，其积 $\kappa_1\kappa_2$ 是 **Gauss 曲率**，平均 $(\kappa_1+\kappa_2)/2$ 是**平均曲率**（极小曲面要求平均曲率为零）。
- 球面 $S^2 \subset \mathbb{R}^3$：第二基本形式处处非零（主曲率都为 $1$）；
- 圆柱面：一个主曲率为零，另一个非零，Gauss 曲率为零——外蕴弯而内蕴平。

## 3 Gauss 方程与 Codazzi 方程

把 Gauss 公式代入曲率张量的定义，逐项展开并分离切向与法向，得到两条「可积性条件」：

**Gauss 方程**（切向部分）：

$$
\langle \bar R(X,Y)Z, W\rangle = \langle R(X,Y)Z, W\rangle - \langle \mathrm{II}(X,W), \mathrm{II}(Y,Z)\rangle + \langle \mathrm{II}(X,Z), \mathrm{II}(Y,W)\rangle
$$

它把环境曲率、内蕴曲率、第二基本形式联系起来。

**Codazzi 方程**（法向部分）：

$$
\left(\bar R(X,Y)Z\right)^\perp = (\bar\nabla_X \mathrm{II})(Y,Z) - (\bar\nabla_Y \mathrm{II})(X,Z)
$$

它说第二基本形式的「协变导数」与环境曲率的法向部分相互约束。

**重点：Gauss-Codazzi 是「子流形能否存在」的可积性条件。** 反过来，若给定度量与满足 Gauss-Codazzi 的第二基本形式，局部必存在等距浸入（Fundamental Theorem of Submanifolds）——这与「向量场积分存在当且仅当可积性条件成立」同构，是微分几何与 PDE 的交界。

## 4 绝妙定理：内蕴曲率不看外蕴弯曲

**定理（Gauss 的绝妙定理，Theorema Egregium）**：对 $\mathbb{R}^3$ 中曲面，**Gauss 曲率 $K$ 只由第一基本形式决定**，与它在空间中如何弯曲（第二基本形式）无关。<span class="marginnote">「Egregium」意为「卓越/惊人」。Gauss 当时称「这一定理特别出色」——因为他的曲面论前辈总把曲面放在空间里研究，而绝妙定理第一次宣告：曲面自身的内部丈量，足以算出自己的 Gauss 曲率。</span>

从 Gauss 方程看得很清楚：环境空间 $\mathbb{R}^3$ 是平直的，$\bar R = 0$，于是取 $X,Y$ 为 $TM$ 的单位正交基 $e_1,e_2$：

$$
0 = \langle R(e_1,e_2)e_2, e_1\rangle + \langle \mathrm{II}(e_1,e_1), \mathrm{II}(e_2,e_2)\rangle - \langle \mathrm{II}(e_1,e_2), \mathrm{II}(e_2,e_1)\rangle
$$

即

$$
K = \kappa_1\kappa_2 = \langle \mathrm{II}(e_1,e_1), \mathrm{II}(e_2,e_2)\rangle - \langle \mathrm{II}(e_1,e_2), \mathrm{II}(e_2,e_1)\rangle
$$

**右侧全部由内蕴量（$R$）与外蕴量（$\mathrm{II}$）组成，而等式保证它们的组合不再依赖嵌入方式。** 换言之：无论把曲面折成圆柱还是卷成锥面（保距变形），只要第一基本形式不变，Gauss 曲率就恒等——**一个可以摊平的地图永远没有 Gauss 曲率**。地球仪无法无损展开成平面地图，正是因为球面的 $K>0$ 是内蕴的，撕不破也摊不平。

**辨析｜易错点：绝妙定理只保护「Gauss 曲率」，不保护平均曲率。** 圆柱面可以保距地摊平（$K=0$ 不变），但平均曲率从非零变到零——平均曲率不是内蕴量。**内蕴的对象只有度量及由度量导出的量**（曲率张量、体积、测地线、截面曲率），形状算子这类依赖法向的玩意不在其中。

## 5 公式解析：Gauss 公式

$$
\bar\nabla_XY = \nabla_XY + \mathrm{II}(X,Y)
$$

四步拆解：

- **第一步，读成分**：等号右边第一项是切向分量（落在 $TM$ 内），第二项是法向分量（落在 $T^\perp M$ 内）。这是把环境导数「正交投影分解」——切向投影即内蕴联络，法向投影即第二基本形式。

- **第二步，为何切向投影正是 $M$ 的联络**：因为等距浸入把度量原样带入，Levi-Civita 联络由度量唯一决定，所以 $M$ 的联络必然是环境联络在切空间上的投影；投影后再投影等于自身，满足联络的全部公理。

- **第三步，$\mathrm{II}$ 为什么对称**：无挠性给出 $\bar\nabla_XY - \bar\nabla_YX = [X,Y] \in TM$；切向投影相减后 $[X,Y]$ 被消去，只剩法向部分，故 $\mathrm{II}(X,Y) = \mathrm{II}(Y,X)$。**对称性是「无挠」在外蕴层面的回响。**

- **第四步，降维到曲面**：$M^2 \subset \mathbb{R}^3$ 时取单位法 $\eta$，$\mathrm{II}(X,Y) = \langle S_\eta X, Y\rangle\,\eta$，于是 Gauss 公式退化为经典的「曲面第二基本形式」：$\bar\nabla_XY = \nabla_XY + h(X,Y)\,\eta$，主曲率、Gauss 曲率、绝妙定理全部由此展开。

## 6 小结

- **等距浸入** $f: M^n \to \bar M^{n+k}$：度量被原样带入；$M$ 的联络是环境联络的切向投影。
- **Gauss 公式** $\bar\nabla_XY = \nabla_XY + \mathrm{II}(X,Y)$：导数分解为切向（内蕴）+ 法向（外蕴）。
- **第二基本形式** $\mathrm{II}$：对称双线性、法向取值；投影到法向量得形状算子，特征值为主曲率。
- **Gauss 方程 / Codazzi 方程**：曲率的内外关联，也是浸入存在的可积性条件。
- **绝妙定理**：$\mathbb{R}^3$ 中曲面的 Gauss 曲率内蕴，不依赖嵌入方式；平均曲率则不是内蕴量。

在下一节，我们把视线从「局部的流形」拉回「整体」：研究**完备性与 Hopf-Rinow 定理**——测地线能走多远，决定了两点之间是否一定有最短路径。
