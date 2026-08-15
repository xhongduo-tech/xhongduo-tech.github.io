---
title: 退化核方程
date: 2026-08-07
---

# 退化核方程

<div class="epigraph">
<p>当核只由有限个「积之和」拼成，积分方程就露馅了——它不过是一副披着积分外衣的线性方程组。</p>
<footer>—— 埃里克 · 伊瓦尔 · 弗雷德霍姆（Erik Ivar Fredholm）</footer>
</div>

<div class="article-byline">
<p>第二级 · 积分方程 ｜ R. Kress《Linear Integral Equations》 第二章 §2.4 ｜ 2026-08-07</p>
</div>

## 为什么从退化核开始

上一节 Neumann 级数有一个尴尬的限制：收敛半径 $1/[M(b-a)]$ 通常小于 $\lambda$ 的真实可解范围。更糟的是，当 $\lambda$ 恰好是特征值时，级数发散，而解却可能仍然存在（只是不唯一）。Fredholm 的洞察在于：**有一类核，方程可以被彻底解出来，并且暴露解对 $\lambda$ 的全部依赖方式**——那就是退化核（degenerate kernel）。

**退化核（degenerate kernel）**：能写成有限项「$x$ 的函数乘 $t$ 的函数」之和的核，即

$$K(x,t) = \sum_{j=1}^{n} a_j(x)\, b_j(t)$$

对这类核，积分方程不再「藏得住」未知函数：积分把 $b_j(t)$ 与 $y(t)$ 的乘积扫过区间，留下的只是 $n$ 个**数字**——于是积分方程坍缩成 $n$ 元线性方程组。<span class="marginnote">退化核并不是个可有可无的玩具：连续核可以用多项式或三角函数均匀逼近，而有限项积之和恰好是「可分离」逼近的代数版本。用退化核去逼近一般核，正是 Fredholm 证明一般理论的核心战术——他先在退化核上把结论证完，再用逼近过渡到连续核。</span>学退化核，是在给整套 Fredholm 理论打地基。

## 1 代入：把方程变成方程组

设 $K(x,t) = \sum_{j=1}^{n} a_j(x) b_j(t)$，代入第二类非齐次方程：

$$y(x) = f(x) + \lambda \sum_{j=1}^{n} a_j(x) \int_{a}^{b} b_j(t)\, y(t)\, dt$$

定义 $n$ 个未知常数

$$c_j := \int_{a}^{b} b_j(t)\, y(t)\, dt, \qquad j = 1, \dots, n$$

于是解的结构一目了然：

$$y(x) = f(x) + \lambda \sum_{j=1}^{n} c_j\, a_j(x)$$

**未知函数 $y$ 被压缩成「$f$ 加上有限个 $a_j$ 的线性组合」**——整个无穷维自由度只剩 $n$ 个待定系数。剩下的工作就是把这 $n$ 个 $c_j$ 定出来。

## 2 回代：导出 $n$ 阶线性方程组

把 $y$ 的表达式代回 $c_j$ 的定义，两边乘以 $b_i(x)$ 并积分：

$$c_i = \int_{a}^{b} b_i(x)\, f(x)\, dx + \lambda \sum_{j=1}^{n} c_j \int_{a}^{b} b_i(x)\, a_j(x)\, dx$$

记

$$f_i = \int_{a}^{b} b_i(x)\, f(x)\, dx, \qquad a_{ij} = \int_{a}^{b} b_i(x)\, a_j(x)\, dx$$

就得到标准的线性方程组：

$$c_i - \lambda \sum_{j=1}^{n} a_{ij}\, c_j = f_i, \qquad i = 1, \dots, n$$

写成矩阵形式 $(\boldsymbol{I} - \lambda \boldsymbol{A})\, \boldsymbol{c} = \boldsymbol{f}$。于是**积分方程的求解彻底变成线性代数**：矩阵 $\boldsymbol{A}$ 的维度等于核的秩 $n$，与区间的无穷维结构无关。<span class="marginnote">这里的 $(\boldsymbol{I} - \lambda \boldsymbol{A})$ 与算子的 $(I - \lambda K)$ 是同构的缩影：矩阵特征值问题 $(I - \lambda A)c = 0$ 的非零解对应积分方程的特征值。退化核把无穷维谱理论「降维」成了有限维谱理论。</span>

## 3 非零行列式：Cramer 法则与有理分式解

线性方程组可解与否，由特征行列式

$$D(\lambda) = \det(\boldsymbol{I} - \lambda \boldsymbol{A})$$

决定。

**当 $D(\lambda) \neq 0$ 时**，方程组有唯一解。由 Cramer 法则，$c_j = D_j(\lambda)/D(\lambda)$，其中 $D_j$ 是把第 $j$ 列换成 $\boldsymbol{f}$ 得到的行列式。代回解的结构：

$$y(x) = f(x) + \lambda \sum_{j=1}^{n} \frac{D_j(\lambda)}{D(\lambda)}\, a_j(x)$$

把求和整理成统一的核，就得到**预解核（resolvent kernel）**的显式：

$$y(x) = f(x) + \lambda \int_{a}^{b} \Gamma(x,t;\lambda)\, f(t)\, dt, \qquad \Gamma = \frac{\sum_{j} a_j(x)\,\Delta_j(t,\lambda)}{D(\lambda)}$$

$\Gamma$ 是 $\lambda$ 的**有理函数**——分子是多项式，分母是 $D(\lambda)$。<span class="marginnote">对比上一节的 Neumann 级数：级数是 $\lambda$ 的无穷幂级数，只在 $|\lambda|<R$ 收敛；而退化核的预解核是分式，除了使 $D(\lambda)=0$ 的有限个点外处处有定义。Fredholm 的伟大发现正是：<strong>一般核的预解核也长这样，分母是整函数 $D(\lambda)$，分子是「一阶子式级数」</strong>——这就把整个理论从「小 $\lambda$ 局部」提升到「除特征值外全平面」。</span>

**手算小例**：取区间 $[0,1]$，核 $K(x,t) = xt$（即 $a_1 = x$，$b_1 = t$，$n=1$），方程 $y(x) = x^2 + \lambda\int_0^1 xt\, y(t)\,dt$。设 $c_1 = \int_0^1 t\,y(t)\,dt$，则 $y = x^2 + \lambda c_1 x$。回代得 $c_1 = \int_0^1 t(t^2 + \lambda c_1 t)\,dt = 1/4 + \lambda c_1/3$，解得 $c_1 = \tfrac{1}{4}(1 - \lambda/3)^{-1}$。于是 $y(x) = x^2 + \tfrac{\lambda x}{4(1-\lambda/3)}$——分母 $D(\lambda) = 1 - \lambda/3$ 恰是 $1\times 1$ 矩阵 $\boldsymbol{I} - \lambda\boldsymbol{A}$ 的行列式，特征值 $\lambda = 3$ 一目了然。

这个例子虽小，却展示了整套逻辑：**定义 $c_j$ → 重写解 → 回代封闭 → 读行列式**。$n$ 增大时每一步都照搬，只是矩阵变稠密而已。

## 4 行列式为零：齐次解与相容条件

**当 $D(\lambda) = 0$ 时**，矩阵 $\boldsymbol{I} - \lambda\boldsymbol{A}$ 奇异。此时方程组要么无解，要么解不唯一——这取决于 $\boldsymbol{f}$ 是否落在列空间里。

具体而言，设 $\boldsymbol{c}^0$ 是齐次方程组 $(\boldsymbol{I} - \lambda\boldsymbol{A})\boldsymbol{c} = \boldsymbol{0}$ 的非零解，则对应的

$$y_0(x) = \lambda \sum_{j=1}^{n} c_j^0\, a_j(x)$$

就是齐次积分方程 $y = \lambda K y$ 的非零解，即**特征函数**。而非齐次方程可解，当且仅当自由项满足**相容条件**：

$$\sum_{i=1}^{n} f_i\, d_i = 0$$

其中 $\boldsymbol{d}$ 是转置方程组 $(\boldsymbol{I} - \lambda \boldsymbol{A}^\top)\boldsymbol{d} = \boldsymbol{0}$ 的解。

**辨析｜易错点：** 这里出现了一个关键的非对称现象——**特征函数对应左特征向量还是右特征向量**。齐次积分方程 $y = \lambda Ky$ 的非零解对应矩阵的**右**特征向量；而相容条件里出现的 $\boldsymbol{d}$ 来自**转置**矩阵，对应**左**特征向量。核 $K(x,t)$ 的转置 $K(t,x)$ 引出的是「伴随方程」，它的特征函数才是检验非齐次方程可解性的标尺。这两套特征函数在非对称核下完全不同，混淆它们是最常见的错误。

## 5 公式解析：从积分方程到矩阵方程

把最核心的「代入→回代」链条单拎出来，看它每一步在做什么：

$$
y(x) = f(x) + \lambda \sum_{j=1}^{n} a_j(x)\, \underbrace{\int_{a}^{b} b_j(t)\, y(t)\, dt}_{c_j}
\;\Longrightarrow\;
c_i = f_i + \lambda \sum_{j=1}^{n} a_{ij}\, c_j
$$

- **第一步，定义 $c_j$**：积分 $\int_a^b b_j(t)y(t)\,dt$ 与 $x$ 无关，是「核在 $t$ 方向与未知函数的内积」。它把 $y$ 的无穷信息压缩成 $n$ 个数——这就是退化核「降维」的机制。
- **第二步，改写解的形式**：既然每个 $c_j$ 都是常数，$y(x) = f(x) + \lambda\sum_j c_j a_j(x)$。注意 $y$ 自己不再出现在任何积分号下，于是「未知函数」变成了「$n$ 个未知常数」。
- **第三步，两边同乘 $b_i(x)$ 再积分**：这是关键的一招。对 $y = f + \lambda\sum_j c_j a_j$ 两边乘 $b_i(x)$ 并积分，左边得到 $c_i$（正是 $c_j$ 的定义），右边分别算出 $f_i$ 与 $a_{ij}$。**同一个定义被用两次：一次用来压缩 $y$，一次用来封闭方程组。**
- **第四步，判断可解性**：矩阵 $(\boldsymbol{I} - \lambda\boldsymbol{A})$ 是否可逆，就是积分算子 $I - \lambda K$ 是否可逆的精确镜像。$D(\lambda) = 0$ 的点是特征值；可解性条件由伴随（转置）方程把关。

## 6 小结

- **退化核** $K(x,t) = \sum_{j=1}^n a_j(x)b_j(t)$ 把积分方程**降维**成 $n$ 元线性方程组 $(\boldsymbol{I} - \lambda\boldsymbol{A})\boldsymbol{c} = \boldsymbol{f}$。
- 解形如 $y(x) = f(x) + \lambda\sum_j c_j a_j(x)$，未知的只是 $n$ 个常数 $c_j = \int b_j(t)y(t)\,dt$。
- 行列式 $D(\lambda) \neq 0$ 时解唯一，且可写成**预解核** $y = f + \lambda\int \Gamma f$，$\Gamma$ 是 $\lambda$ 的有理函数——Neumann 级数的收敛半径问题被一举消除。
- $D(\lambda) = 0$ 时齐次方程有**特征函数**，非齐次方程可解当且仅当 $\boldsymbol{f}$ 与伴随方程的解正交（**相容条件** $\sum f_i d_i = 0$）。
- 特征函数来自矩阵的**右**特征向量，相容条件把关的是**左**特征向量——对称核下两者重合，非对称核下必须区分。
- 退化核的意义远超自身：Fredholm 正是先在退化核上把结论证完、再用连续核逼近，才把整个理论推广到一般核——它是整套 Fredholm 理论的脚手架。

在下一节，我们将把退化核里「行列式 $D(\lambda)$ 主宰一切」的直觉，提升为对任意连续核都成立的定理——这就是 **Fredholm 择一定理**。