---
title: Hilbert–Schmidt 理论
date: 2026-08-07
---

# Hilbert–Schmidt 理论

<div class="epigraph">
<p>对称算子拥有配得上它的谱：特征值实数、特征函数正交，就像一块水晶把自己的晶面按对称性打磨得整整齐齐。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 积分方程 ｜ R. Kress《Linear Integral Equations》 第七章 ｜ 2026-08-07</p>
</div>

## 为什么需要对称核

Fredholm 择一定理对任意连续核都成立，但它的「择一」只告诉我们在特征值处会出事，却没告诉特征值**分布得多整齐**。现在我们要给核加一条金贵的假设：

$$K(x,t) = \overline{K(t,x)}$$

实值情形下就是 $K(x,t) = K(t,x)$，称为**对称核（symmetric kernel）**；复值情形称为 **Hermite 对称核**。这条假设看似温柔，却让整个谱理论从「一般紧算子的混沌谱」跃迁到「几乎等于线性代数的光谱理论」。<span class="marginnote">对称核在物理中太常见了：Green 函数几乎总是对称的——「在 $x$ 处放一个点源对 $t$ 处的影响」等于「在 $t$ 处放一个点源对 $x$ 处的影响」，这是互易性（reciprocity）的数学化身。热传导、弹性力学、电动力学里大量积分方程的核都是对称的。</span>

**Hilbert–Schmidt 理论**研究的正是这类算子：它证明对称积分算子的谱具有「实特征值 + 正交特征函数 + 完备展开」三位一体的结构，并把这种结构写成可操作的级数公式。学完这一节，积分方程将首次拥有**类似对角化矩阵**的完整图景。

## 1 自伴算子与紧算子

先为「为什么对称核这么特殊」铺两块基石。

**自伴（self-adjoint）**：在 $L^2(a,b)$ 内积 $\langle u, v\rangle = \int_a^b u(t)\overline{v(t)}\,dt$ 下，对称核诱导的算子满足

$$\langle Ku, v\rangle = \langle u, Kv\rangle$$

这是「核对称 ⟺ 算子自伴」的直接推论，把转置核的抽象概念落成了内积等式。<span class="marginnote">自伴算子的谱恒为实数，这是《线性代数》里「实对称矩阵特征值都是实数」在无穷维的翻版。证明只需一行：$\lambda\langle \varphi,\varphi\rangle = \langle K\varphi,\varphi\rangle = \langle \varphi,K\varphi\rangle = \overline{\lambda}\langle\varphi,\varphi\rangle$。</span>

**紧（compact）**：带连续核的积分算子把 $L^2$ 中的有界集映成相对紧集——这就是上一节提到的「紧性」。紧性带来两个决定性的后果：特征值（除了 0）**只能有有限重数、且至多可数、无有限聚点**。这两条（自伴 + 紧）正是泛函分析谱定理的全部入场券。

## 2 特征值的三条基本性质

在自伴 + 紧的假设下，对称积分算子的特征值 ${\lambda_n}$ 满足：

**性质一（特征值为实数）**：设 $\varphi$ 是特征函数，$K\varphi = \lambda\varphi$，两边与 $\varphi$ 取内积：

$$\lambda = \frac{\langle K\varphi, \varphi\rangle}{\langle\varphi, \varphi\rangle}$$

分子是实数（自伴性保证 $\langle K\varphi,\varphi\rangle = \overline{\langle K\varphi,\varphi\rangle}$），分母是正实数，故 $\lambda \in \mathbb{R}$。

**性质二（异值特征函数正交）**：设 $K\varphi = \lambda\varphi$、$K\psi = \mu\psi$，且 $\lambda \neq \mu$。则

$$\lambda\langle\varphi,\psi\rangle = \langle K\varphi,\psi\rangle = \langle\varphi,K\psi\rangle = \mu\langle\varphi,\psi\rangle$$

故 $(\lambda - \mu)\langle\varphi,\psi\rangle = 0$，由 $\lambda \neq \mu$ 得 $\langle\varphi,\psi\rangle = 0$。**不同特征值对应的特征函数彼此正交**——同一特征值内部则用 Gram–Schmidt 正交化补齐。

**性质三（特征值仅聚于 0）**：紧算子的特征值序列若无穷，则 $|\lambda_n| \to 0$。直观地说，紧算子在高频方向上「压得越来越扁」，特征值也随之萎缩到 0。<span class="marginnote">对比微分算子：$-\dfrac{d^2}{dx^2}$ 在 $[0,\pi]$ 上的特征值是 $n^2 \to \infty$。积分算子的特征值聚于 0、微分算子的特征值趋于无穷，两者互为对偶——这正是「积分是微分的逆」在谱层面的回声。</span>

## 3 Hilbert–Schmidt 定理：算子的对角化

核心定理给出特征展开：**设 $K$ 是对称连续核，$\{\varphi_n\}$ 是全部特征函数（已正交归一），$\{\lambda_n\}$ 是相应特征值（非零），则对任意 $f \in L^2(a,b)$**

$$(Kf)(x) = \sum_{n=1}^{\infty} \lambda_n\, \langle f, \varphi_n\rangle\, \varphi_n(x)$$

级数在 $L^2$ 范数下收敛。<span class="marginnote">这就是无穷维版的「对称矩阵对角化」：$\boldsymbol{K} = \boldsymbol{\Phi}\boldsymbol{\Lambda}\boldsymbol{\Phi}^*$。左边的积分算子 $K$ 被拆成「投影到每个特征方向、乘以特征值、再叠回来」，与 $K(x,t) = \sum \lambda_n\varphi_n(x)\overline{\varphi_n(t)}$ 的核展开是同一件事的两种写法。</span>

这个定理的威力在于：**$K$ 的像空间被特征函数完全控制**。只要 $f$ 落在 $K$ 的像里，它的分解就是「有限个基函数的加权和」，高频部分被 $\lambda_n \to 0$ 天然截断。这为下一节的核展开定理铺好了路。

在较弱的条件下（核只要平方可积而非连续），$K$ 是所谓的 **Hilbert–Schmidt 算子**，其特征值还满足**平方可和**条件 $\sum_n |\lambda_n|^2 \lt  \infty$。这个「特征值平方和有限」是 $L^2$ 核算子的标志性约束——它保证特征级数在「均方」意义下处处好用，也是数值谱方法的理论基础。连续核的情形自动包含在内，因为连续核必有界、更平方可积。

## 4 公式解析：展开式里的每一个符号

把核心公式逐项拆开：

$$
(Kf)(x) = \sum_{n=1}^{\infty} \lambda_n\, \langle f, \varphi_n\rangle\, \varphi_n(x)
$$

- **第一步，看 $\langle f, \varphi_n\rangle$**：这是 $f$ 在特征方向 $\varphi_n$ 上的投影系数，等价于傅里叶系数——但它用的是**算子的特征基**，而非正弦基。对一般对称核，这组基由核自己「长出来」。
- **第二步，看 $\lambda_n$**：每个投影系数被特征值加权。$\lambda_n$ 大，说明该方向被算子放大；$\lambda_n \to 0$，说明高频方向被压制。**正是这一层加权让 $Kf$ 比 $f$ 更光滑**——积分的「光滑化」性质在频谱上就是「高频乘上小系数」。
- **第三步，看求和次序**：级数按 $|\lambda_1| \ge |\lambda_2| \ge \cdots$ 排列。紧性保证 $|\lambda_n|\to 0$，所以求和收敛很快；这也是数值上取「前 $N$ 项截断」就能高精度逼近的根据。
- **第四步，看两边的一致性**：把核展开 $K(x,t) = \sum \lambda_n \varphi_n(x)\overline{\varphi_n(t)}$ 代入 $(Kf)(x) = \int_a^b K(x,t)f(t)dt$，逐项积分恰好得到右边——**核展开与算子展开互为表里**。

## 5 从 Hilbert–Schmidt 到谱分解：解方程的一步到位

有了展开定理，第二类方程 $y = f + \lambda K y$ 在 $\lambda$ 非特征值时的解，可以像解对角矩阵一样写出来。设 $\lambda \neq 1/\lambda_n$（注意这里的特征值约定），把 $y$ 在特征基下展开，系数逐个解出：

$$y(x) = f(x) + \lambda \sum_{n=1}^{\infty} \frac{\lambda_n \langle f,\varphi_n\rangle}{1 - \lambda\lambda_n}\, \varphi_n(x)$$

这个公式是整节理论的高潮：**解被分解成特征方向上的一个个「标量方程」**，每个方向独立求解，再叠加。<span class="marginnote">对照有限维：对称矩阵 $I - \lambda A$ 的逆在特征基下就是 $\sum_n (1-\lambda\lambda_n)^{-1} \varphi_n\varphi_n^*$。这里 $1/(1-\lambda\lambda_n)$ 正是「第一择一」中预解核在特征方向的谱表示，Fredholm 行列式 $D(\lambda) = \prod(1-\lambda\lambda_n)$ 也由此自然出现。</span>

当 $\lambda = 1/\lambda_m$ 时，第 $m$ 个方向的分母爆炸，解要么不存在（除非 $\langle f,\varphi_m\rangle = 0$），要么不唯一——这正是 Fredholm 择一在对称核下的显式形态，比抽象的正交条件更直观。

## 6 例：$K(x,t) = \min(x,t)$ 的完整谱

理论要落地，最好的例子是 $[0,1]$ 上的核 $K(x,t) = \min(x,t)$——它是 $-\dfrac{d^2}{dx^2}$ 的 Green 函数，本身就是一个「积分是微分逆运算」的活标本。考虑齐次方程

$$y(x) = \lambda \int_{0}^{1} \min(x,t)\, y(t)\, dt$$

**第一步，把积分方程翻译成微分方程**。把积分拆成 $t \lt  x$ 与 $t > x$ 两段：$y(x) = \lambda\left[\int_0^x t\,y(t)\,dt + x\int_x^1 y(t)\,dt\right]$。对 $x$ 求导，第一段贡献 $x\,y(x)$，第二段贡献 $\int_x^1 y(t)\,dt - x\,y(x)$，两者相加恰剩 $\lambda\int_x^1 y(t)\,dt$；再求一次导，得

$$y''(x) = -\lambda\, y(x)$$

**第二步，读边界条件**。从 $y(0) = 0$（因 $\min(0,t) = 0$）与 $y'(1) = 0$（因 $y'(1) = \lambda\int_1^1 \cdots = 0$）。

**第三步，解这个特征值问题**。$y'' + \lambda y = 0$ 配合 $y(0)=0$ 给 $y = c\sin(\sqrt{\lambda}\,x)$；$y'(1) = 0$ 给 $\cos\sqrt{\lambda} = 0$，于是

$$\lambda_n = \left(n - \tfrac12\right)^2\pi^2, \qquad \varphi_n(x) = \sqrt{2}\,\sin\!\left(\left(n - \tfrac12\right)\pi x\right), \qquad n = 1,2,\dots$$

**第四步，验证理论承诺的性质**：$\lambda_n$ 全是正的实数；$\varphi_n$ 在 $[0,1]$ 上两两正交（正弦族的经典正交性）。这个例子完整展示了对称核谱理论从「假设」到「可算的谱」的全程——而且特征值序列 $\propto n^2$ 发散，正是「积分算子之逆（微分算子）特征值趋于无穷」的印证。

**辨析｜易错点：** 不同教材对特征值有两种约定：一种写 $Ky = \lambda y$（本课采用），一种写 $Ky = \mu y$ 后再设 $\lambda = 1/\mu$。两套记号的 $\lambda_n$ 互为倒数，张冠李戴会算出差 $1/\lambda_n$ 的荒谬结果。解题前先确认方程写的是 $y = \lambda Ky$ 还是 $\lambda y = Ky$。

## 7 小结

- **对称核** $K(x,t) = \overline{K(t,x)}$ 诱导**自伴**算子，配合积分算子的**紧性**，构成谱理论的全部前提。
- 特征值三条性质：**实数**、**异值特征函数正交**、**仅聚于 0**。
- **Hilbert–Schmidt 定理**：$(Kf)(x) = \sum_n \lambda_n\langle f,\varphi_n\rangle\varphi_n(x)$，是「对称矩阵对角化」的无穷维版本。
- 核展开 $K(x,t) = \sum_n \lambda_n\varphi_n(x)\overline{\varphi_n(t)}$ 与算子展开**互为表里**。
- 第二类方程在特征基下逐方向求解，解有显式的谱表示 $y = f + \lambda\sum_n \frac{\lambda_n\langle f,\varphi_n\rangle}{1-\lambda\lambda_n}\varphi_n$；$\lambda = 1/\lambda_m$ 时第 $m$ 个方向分母爆炸，正是择一的显式形态。
- $K(x,t) = \min(x,t)$ 的完整例证表明：积分方程的谱可以整体翻译成微分方程的特征值问题，且特征值序列 $\propto n^2$ 发散——积分、微分互为逆在谱层面的回声。

在下一节，我们把这个「对角化」贯彻到底：用对称核的特征展开直接表达核与解，得到**对称核的展开定理**——谱理论的收官之作。