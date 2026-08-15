---
title: 对称核的展开定理
date: 2026-08-07
---

# 对称核的展开定理

<div class="epigraph">
<p>一个对称核，就是一张谱表：它的特征函数把自身展开成无限级数，而解方程只是在这张表上做加法。</p>
<footer>—— 詹姆斯 · 默塞尔（James Mercer）</footer>
</div>

<div class="article-byline">
<p>第二级 · 积分方程 ｜ R. Kress《Linear Integral Equations》 第八章 ｜ 2026-08-07</p>
</div>

## 为什么要把核本身展开

上一节我们得到算子的展开 $(Kf)(x) = \sum_n \lambda_n\langle f,\varphi_n\rangle\varphi_n(x)$，它说的是「算子作用在 $f$ 上」的分解。但还有一个更漂亮的问题悬而未决：**核本身 $K(x,t)$ 能不能写成特征函数的级数？** 若能，则不仅 $(Kf)$ 可展开，连积分方程的**预解核**、解的一切性质都能在特征基下一览无余。<span class="marginnote">这相当于把 $K$ 看成「特征函数的无穷维矩阵」，$K(x,t)$ 就是它的「$(x,t)$ 元」。对称矩阵可对角化的论断「$\boldsymbol{A} = \boldsymbol{\Phi}\boldsymbol{\Lambda}\boldsymbol{\Phi}^\top$」，在这里变成了核的级数展开——本节就是要把这条有限维等式在无穷维完全兑现。</span>

**对称核的展开定理（expansion theorem）**：设 $K(x,t)$ 是对称连续核，$\{\varphi_n\}$ 是它的（正交归一）特征函数系，$\{\lambda_n\}$ 是相应特征值，则核有展开

$$K(x,t) = \sum_{n=1}^{\infty} \lambda_n\, \varphi_n(x)\, \varphi_n(t)$$

其中收敛在 $L^2$ 意义下成立。若核还满足**正定性**（对任意连续 $f$ 有 $\int_a^b\int_a^b K(x,t)f(x)f(t)\,dx\,dt \ge 0$），则收敛还是**绝对且一致**的——这就是 **Mercer 定理**。

## 1 展开定理的内容与条件

展开定理分两个层次：

**层次一（平方收敛）**：对任意对称连续核，级数 $\sum_n \lambda_n\varphi_n(x)\varphi_n(t)$ 在 $L^2(a,b)\times L^2(a,b)$ 中收敛到 $K(x,t)$。这个层次的收敛由 Hilbert–Schmidt 理论直接保证，不需要额外假设。

两层收敛的差别，用一张对照表可以钉死：

| | 平方收敛（一般对称核） | 一致收敛（Mercer，正定核） |
| --- | --- | --- |
| 前提 | 对称、连续 | 对称、连续、正定 |
| 特征值符号 | 可正可负 | 全部 $\ge 0$ |
| 收敛方式 | $L^2\times L^2$ 均方 | 绝对且一致 |
| 逐点代入 | 不可靠 | 可靠 |
| 典型例子 | $\sin(x+t)$ 类核 | 高斯核、$\min(x,t)$ |

**层次二（一致收敛，Mercer）**：若核**正定**——即对所有有限点集 $\{x_i\}$ 与所有实数 $c_i$，$\sum_{i,j} K(x_i,x_j)c_i c_j \ge 0$——则特征值全非负，且级数在 $[a,b]\times[a,b]$ 上**绝对且一致收敛**。<span class="marginnote">正定核正是「矩阵半正定」的连续版本：把点集 $\{x_i\}$ 想成矩阵的行列指标，$K(x_i,x_j)$ 就是半正定矩阵的元。协方差核（如高斯核 $e^{-(x-t)^2}$）是正定核的典型，所以它在统计与机器学习里无处不在——这也是本博客第九级《机器学习》里核方法（kernel methods）的数学根子。</span>

**辨析｜易错点：** 一般对称核的展开只保证**均方收敛**，不能逐点断言；只有当核正定时才升级为一致收敛。用「展开后逐点代入」去处理非正定核，是常见错误——可能在个别点上错得离谱。

## 2 展开定理的证明骨架

为何对称核能被自己的特征函数展开？证明的核心是**有限秩逼近**。取前 $N$ 个特征值，构造截断核

$$K_N(x,t) = \sum_{n=1}^{N} \lambda_n\, \varphi_n(x)\, \varphi_n(t)$$

则 $K_N$ 与 $K$ 的差是一个「高频残余」。关键是证明

$$\int_a^b\int_a^b \big|K(x,t) - K_N(x,t)\big|^2\, dx\, dt \;\xrightarrow[N\to\infty]{}\; 0$$

这一步用到 Fredholm 行列式的整函数性质（或紧算子的近似数值域），思路是：若残余核不趋于 0，它就还有非零特征值，而这些「新特征值」对应的特征函数正交于前 $N$ 个——这与 $\{\lambda_n\}$ 按绝对值递减的排列相矛盾。**「特征函数系已经把核张满」这个结论，本质上与 Fourier 级数的完备性是同一个论证。**<span class="marginnote">这也解释了为什么「特征值只能聚于 0」如此关键：残余核的谱就是 $\{\lambda_n\}_{n>N}$，它随 $N$ 增大而萎缩到 0，正是「$|\lambda_n|\to 0$」保证残余核在 $L^2$ 意义下趋于 0。</span>

注意这个论证没有用到 Fredholm 行列式的具体形式，只用到了「非零特征值至多可数、无有限聚点」这两条紧性性质。因此**同样的结论可以原封不动移植到任意紧自伴算子**——这正是谱定理的一般性所在，也解释了为什么本节理论在第二级《泛函分析》里会以更抽象、却同样漂亮的形式重现。

## 3 公式解析：核展开如何变成解方程的操作

把 Mercer 展开代进第二类方程，是本节最有用的动作。设 $\lambda$ 不是特征值（即 $\lambda \neq 1/\lambda_n$），把 $y = f + \lambda K y$ 的解写成

$$
y(x) = f(x) + \lambda \sum_{n=1}^{\infty} \frac{\lambda_n\, \langle f,\varphi_n\rangle}{1 - \lambda\lambda_n}\, \varphi_n(x)
$$

- **第一步，把 $y$ 在特征基下展开**：$y(x) = f(x) + \lambda\int K(x,t)y(t)dt$。把 $K(x,t)$ 换成级数 $\sum_n\lambda_n\varphi_n(x)\varphi_n(t)$，积分逐项进行，出现 $\langle y,\varphi_n\rangle$。
- **第二步，写出 $y$ 的系数**：设 $y = f + \sum_n c_n\varphi_n(x)$（注意 $f$ 自己也要按 $\{\varphi_n\}$ 展开，除非 $f$ 恰好正交于某个特征方向），两边与 $\varphi_n$ 取内积，得 $c_n = \langle f,\varphi_n\rangle + \lambda\lambda_n\, c_n$，即 $c_n = \langle f,\varphi_n\rangle/(1-\lambda\lambda_n)$。
- **第三步，回代得到显式解**：把 $c_n$ 代回，就是上面那条公式。**每个方向独立求解、互不干扰**——这就是「对角化」带来的全部便利。
- **第四步，观察预解核的谱表示**：把公式重写成 $y = f + \lambda\int\Gamma(x,t;\lambda)f(t)dt$，可读出

$$\Gamma(x,t;\lambda) = \sum_{n=1}^{\infty} \frac{\varphi_n(x)\, \varphi_n(t)}{1 - \lambda\lambda_n}$$

**预解核 $\Gamma$ 的特征展开是本节最重要的产出**：它把 Fredholm 行列式的有理分式，升级成特征方向上一目了然的单项分式之和。

## 4 特征展开的完备性与应用

展开定理还蕴含一条深刻的**完备性**结论：如果对称核没有特征函数（即 $\lambda_n$ 全不存在），则 $K \equiv 0$。更一般地，特征函数系 $\{\varphi_n\}$ **张满 $K$ 的值域**——任何形如 $Kf$ 的函数都能被特征级数展开。

这条完备性直接服务三大应用：

**应用一（解方程）**：如上节，第二类方程在非特征值时解有显式谱表示；在特征值时择一条件 $\langle f,\varphi_m\rangle = 0$ 也可以逐方向读出。

**应用二（逼近与截断）**：因为 $|\lambda_n|\to 0$，取前 $N$ 项就能得到高精度近似解——这为数值方法（本专题最后一课）提供了理论依据。<span class="marginnote">现代「谱方法」（spectral methods）解积分方程，本质上就是在特征基里截断求和。对称核的谱方法有近乎最优的收敛性，因为截断误差正比于被砍掉的 $\lambda_n$，而它们衰减极快。</span>

**应用三（正定核的机器翻译）**：Mercer 展开把正定核写成「特征函数的外积和」，意味着**核函数等价于把数据映射到特征空间后的内积**——这正是第九级《机器学习》核技巧（kernel trick）的数学基石：$K(x,t) = \langle \Phi(x),\Phi(t)\rangle$，$\Phi(x) = (\sqrt{\lambda_1}\varphi_1(x), \sqrt{\lambda_2}\varphi_2(x), \dots)$。

**退化核情形的一致性**：若核本身退化（有限秩 $n$），展开自动停在 $N = n$ 项，展开定理退化成第三课的有理分式公式——这正是「退化核理论是一般理论的有限维影子」的又一佐证。

## 5 例：用展开定理解 $\min(x,t)$ 方程

把上一节算出的谱拿来实战。核 $K(x,t) = \min(x,t)$，特征值 $\lambda_n = (n-\tfrac12)^2\pi^2$，特征函数 $\varphi_n(x) = \sqrt{2}\sin((n-\tfrac12)\pi x)$。解方程

$$y(x) = 1 + \lambda \int_{0}^{1} \min(x,t)\, y(t)\, dt$$

**第一步，算 $f(x) = 1$ 的投影系数**：

$$\langle 1,\varphi_n\rangle = \int_0^1 \sqrt{2}\sin\!\left((n-\tfrac12)\pi x\right)\, dx = \frac{\sqrt{2}}{(n-\tfrac12)\pi}$$

这里用了 $\cos((n-\tfrac12)\pi) = 0$，积分一次即得。

**第二步，套谱解公式**：

$$y(x) = 1 + \lambda \sum_{n=1}^{\infty} \frac{\lambda_n\, \langle 1,\varphi_n\rangle}{1 - \lambda\lambda_n}\, \varphi_n(x) = 1 + \lambda \sum_{n=1}^{\infty} \frac{2\,(n-\tfrac12)\pi\, \sin((n-\tfrac12)\pi x)}{1 - \lambda (n-\tfrac12)^2\pi^2}$$

**第三步，验证边界情形**：$\lambda = 0$ 时级数消失，$y(x) = 1$，正确（此时方程就是 $y = 1$）；$\lambda$ 很小时级数首项近似 $\lambda\cdot 2(n-\tfrac12)\pi\sin(\cdots)$，量级符合 Neumann 级数的首阶。**公式自动体现了「分母在 $\lambda = 1/\lambda_n$ 处爆炸」的择一结构**——这正是对称核展开把抽象理论变成可算算术的范例。

如果想验证数值，取 $\lambda = 1$、$x = 1/2$，前几项的和就能逼近真实解；截断误差由第一个被砍掉的项控制，而它的 $\lambda_n$ 已经很小。**谱方法的高效在此显形**。

## 6 小结

- **对称核展开定理**：$K(x,t) = \sum_n \lambda_n\varphi_n(x)\varphi_n(t)$，$L^2$ 收敛；核正定时（Mercer 定理）收敛为**绝对且一致**。
- 证明靠**有限秩截断**与「特征值只能聚于 0」：残余核的谱萎缩到 0，逼近必然成功。
- 第二类方程在特征基下**逐方向求解**，解与**预解核**都有显式谱表示 $\Gamma = \sum_n \varphi_n\varphi_n/(1-\lambda\lambda_n)$。
- **完备性**：特征函数系张满 $K$ 的值域；无特征函数则核恒为 0。
- 应用覆盖**解方程、谱方法数值逼近、机器学习核技巧**三条线。
- 退化核自动退化为有限和，与第三课的有理分式公式一致；一般紧自伴算子共享同一套论证。

在下一节，我们回到非对称的世界，专门处理有着「因果结构」的 **Volterra 方程**——它不需要特征值理论，预解核通过更简单的递推就能算出来，且解对所有 $\lambda$