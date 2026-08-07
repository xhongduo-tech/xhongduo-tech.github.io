---
title: 切向量：导子观点与曲线观点
date: 2026-08-07
---

# 切向量：导子观点与曲线观点

<div class="epigraph">
<p>一个切向量不再是一个箭头，而是一种「对函数求方向导数」的指令——这是现代微分几何的转折点。</p>
<footer>—— 昂利 · 嘉当（Henri Cartan）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§6.4 ｜ 2026-08-07</p>
</div>

## 为什么从切向量开始

曲面上，切向量是「过点的曲线速度」——直观、几何。但流形**没有环境空间**，没有「箭头」可画。于是需要一个内在的切向量定义——**不依赖嵌入、只依赖流形自身**。

流形上的切向量有两种经典定义：

- **曲线观点（curve viewpoint）**：过 $p$ 的曲线的速度等价类——推广「速度向量」。
- **导子观点（derivation viewpoint）**：满足 Leibniz 法则的线性算子 $v: C^\infty(M) \to \mathbb{R}$——把切向量看成「方向导数」。

**两者完全等价**——这是现代微分几何的美妙转折：**切向量从「几何箭头」变成「代数算子」**。这个抽象不仅让切向量在无环境空间的流形上存在，还让「切空间」成为线性代数可以全权处理的对象。<span class="marginnote">「切向量 = 方向导数」是微分几何最深刻的观念转变之一。在曲面论里，切向量是嵌入空间里的箭头；在流形论里，切向量是一个算子：输入函数、输出「沿某个方向的导数」。箭头只是算子的「化身」——我们通常用箭头画算子，但算子是本体。</span>

## 1 导子观点：切向量 = 方向导数

**定义（导子，derivation）**：流形 $M$ 在 $p$ 处的一个**切向量**是一个线性映射

$$
v: C^\infty(M) \longrightarrow \mathbb{R}
$$

满足两条：

1. **线性**：$v(af + bg) = a\,v(f) + b\,v(g)$。
2. **Leibniz 法则**：$v(fg) = f(p)\,v(g) + v(f)\,g(p)$。

这样的 $v$ 称为 $p$ 处的**导子**。全体导子构成**切空间** $T_pM$。

**重点：切向量 = 满足 Leibniz 法则的线性算子。** 它把每个光滑函数 $f$ 送到一个数——「$f$ 沿这个方向的瞬时变化率」。Leibniz 法则正是「乘积求导法则」的抽象，它把「求方向导数」从「具体方向」中解放出来。<span class="marginnote">直觉：如果 $v$ 是「沿曲线 $\alpha$ 求导」算子 $v(f) = \frac{d}{dt}\big|_{0} f(\alpha(t))$，它天然满足线性与 Leibniz（乘积法则）。所以「方向导数」是最典型的导子。反过来，任何导子都「是」某个方向导数——这就是两观点等价的实质。</span>

### 坐标基导子

在坐标卡 $(x^1,\dots,x^n)$ 下，**偏导算子**

$$
\frac{\partial}{\partial x^i}\Big|_p: f \longmapsto \frac{\partial (f\circ\varphi^{-1})}{\partial x^i}\Big|_{\varphi(p)}
$$

是导子。它们构成 $T_pM$ 的一组基——**坐标基（coordinate basis）**：

$$
T_pM = \operatorname{span}\Big\{\frac{\partial}{\partial x^1}\Big|_p, \dots, \frac{\partial}{\partial x^n}\Big|_p\Big\}
$$

**切空间的维数 = 流形维数 $n$。** 任意切向量 $v = \sum_i v^i \frac{\partial}{\partial x^i}\big|_p$，$v^i$ 是它的坐标分量。<span class="marginnote">记号说明：$\partial/\partial x^i$ 不是真的「除」，而是「对第 $i$ 个坐标求偏导」的算子符号。$v(f) = \sum_i v^i \partial f/\partial x^i$ 就是「沿 $v$ 方向的方向导数」——与第一级《数学分析》的方向导数公式完全一致，只是换成了算子语言。</span>

## 2 曲线观点：切向量 = 速度等价类

**定义（曲线观点）**：$p$ 处的切向量是过 $p$ 的光滑曲线 $\alpha$（$\alpha(0)=p$）的**速度向量**的等价类：$\alpha \sim \beta$ 若它们在 $p$ 处「一阶相同」——对任意坐标卡，$\frac{d}{dt}\big|_0 \varphi(\alpha(t)) = \frac{d}{dt}\big|_0 \varphi(\beta(t))$。

**等价类 $[\alpha]$ 就是切向量**——「过 $p$ 且在该点以某个速度方向运动的曲线族」。

### 两种观点的对应

给定曲线 $\alpha$（$\alpha(0) = p$），定义导子

$$
[\alpha] \longmapsto v_\alpha, \qquad v_\alpha(f) = \frac{d}{dt}\Big|_0 f(\alpha(t))
$$

这个对应是双射：**每个速度类给出一个导子，每个导子来自某个速度类**。<span class="marginnote">对应关系不依赖坐标：$f(\alpha(t))$ 在坐标下是 $f\circ\varphi^{-1}(\varphi\circ\alpha(t))$，链式法则说明 $v_\alpha(f)$ 由 $\alpha$ 的坐标速度唯一决定。所以「曲线观点」与「导子观点」是同一事物的两种语法——几何的（曲线）与代数的（算子）。</span>

## 3 公式解析：为什么坐标基导子线性无关

验证 $\{\partial/\partial x^i\}$ 张成且无关，是理解切空间维数的关键：

- **第一步，作用在坐标函数上**：记坐标函数 $x^j: M\to\mathbb{R}$（把点映到第 $j$ 个坐标）。偏导算子作用：
  $$
  \frac{\partial}{\partial x^i}\big|_p (x^j) = \delta_i^j = \begin{cases}1 & i=j\\ 0 & i\neq j\end{cases}
  $$
  就像平面里 $\partial_x$ 作用在 $x$ 上得 1、作用在 $y$ 上得 0。
- **第二步，线性无关**：若 $\sum_i c^i \frac{\partial}{\partial x^i}\big|_p = 0$（零算子），作用在 $x^j$ 上得 $\sum_i c^i\delta_i^j = c^j = 0$——每个 $c^j = 0$。**坐标基导子线性无关。**
- **第三步，张成性**：任意导子 $v$ 在 $p$ 附近可展开为 $\sum_i v(x^i)\,\partial/\partial x^i$（用 Taylor 展开 + Leibniz 法则），$v(x^i)$ 正是 $v$ 的第 $i$ 个分量。**所以 $\{\partial/\partial x^i\}$ 是基，$\dim T_pM = n$。**

**重点：切空间的维数 = 流形维数，坐标基由偏导算子给出。** 这个结论把「切空间是线性空间」落到实处——全部线性代数（基、矩阵、特征值）立即可以用于切空间。<span class="marginnote">「$T_pM$ 是 $n$ 维向量空间」是流形理论的基础事实：每一点都自带一个 $n$ 维线性空间（切空间），流形是「处处贴着 $n$ 维线性空间的弯曲空间」。这个「纤维化」结构（每点一个向量空间）是向量丛思想的起点。</span>

## 4 坐标变换下切向量的行为

切向量的分量随坐标变换如何变？设两个坐标卡 $x$ 与 $\bar x$，同一向量 $v$ 在两个基下分别为 $v^i$ 与 $\bar v^i$。由链式法则：

$$
v = \sum_i v^i\frac{\partial}{\partial x^i} = \sum_j \bar v^j\frac{\partial}{\partial \bar x^j}, \qquad
\bar v^j = \sum_i v^i \frac{\partial \bar x^j}{\partial x^i}
$$

**重点：切向量分量按「Jacobi 矩阵」变换——这是「逆变向量」的定义性行为。** 切向量（切空间元素）在坐标变换下用逆矩阵（$\partial\bar x/\partial x$）变换，故称**逆变（contravariant）**。这个「上标分量」的变换律是张量分析的起点（第七篇）。<span class="marginnote">对比：余切向量（对偶空间元素，如梯度）按「相反方向」变换（用 $\partial x/\partial\bar x$），称<strong>协变（covariant）</strong>。上标（逆变）下标（协变）的命名与变换律，在黎曼几何张量演算里是日常语言。记住「切向量逆变、梯度协变」即可。</span>

## 5 切向量的现代意义

切向量从「箭头」到「算子」的转变，带来的远不止抽象：

- **普适性**：流形上没有环境空间，箭头画不出来，但算子处处存在。
- **线性化**：切空间是流形的「局部线性化」——一切非线性对象都在切空间上线性化。
- **向量丛**：把所有 $T_pM$ 捆成**切丛 $TM$**——流形的「速度空间」，向量场的定义域。
- **深度学习**：参数流形上，梯度、动量、自然梯度都是切向量；「沿切向量更新」是流形优化的基础（第九篇）。<span class="marginnote">在黎曼优化里，参数空间是流形，目标函数是光滑函数，梯度是切向量——但「切向量」现在不画箭头，而是「对目标函数求方向导数」的算子。深度学习框架里「梯度」本质上就是这个导子算子。从箭头到算子，几何与优化的语言从此统一。</span>

### 例：$\mathbb{R}^n$ 上的导子就是方向导数

在 $\mathbb{R}^n$ 上验证「导子 = 方向导数」。设 $v = (v^1, \dots, v^n)$ 是普通向量，定义算子

$$
v(f) = \sum_i v^i \frac{\partial f}{\partial x^i}\Big|_p = D_v f(p)
$$

- **线性**：导数线性 ✓。
- **Leibniz**：$D_v(fg) = (D_v f)g + f(D_v g)$（乘积法则）✓。
- **坐标基**：$\partial/\partial x^i$ 对应基向量 $e_i$——$e_i(f) = \partial f/\partial x^i$。

**重点：$\mathbb{R}^n$ 里的切向量（箭头）与导子（方向导数）是同一回事——箭头 $v$ 对应的导子就是「沿 $v$ 的方向导数」。** 这个例子确认「导子观点」没有丢掉几何直觉：箭头还在，只是换了个「身份」（算子）。流形上箭头画不了，但导子处处可定义——「代数化让几何普适」。

## 6 小结

- **导子观点**：切向量 = 满足线性 + Leibniz 法则的算子 $v: C^\infty(M)\to\mathbb{R}$。
- **曲线观点**：切向量 = 过 $p$ 的曲线的速度等价类；两观点等价（双射）。
- **坐标基**：$\{\partial/\partial x^i\}$ 是 $T_pM$ 的基，$\dim T_pM = \dim M$。
- 分量变换：$v$ 逆变（按 $\partial\bar x/\partial x$ 变换）；梯度类协变。
- 意义：无环境空间的普适切向量、局部线性化、切丛、流形优化。

在下一节，我们把切空间扩展成「全体切空间的集合」：**切空间与余切空间**——从单点切空间走向切丛与对偶空间。
