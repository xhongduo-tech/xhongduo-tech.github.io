---
title: 哈密顿向量场与 Poisson 括号
date: 2026-08-07
---

# 哈密顿向量场与 Poisson 括号

<div class="epigraph">
<p>把辛结构交给函数，函数就会动起来；函数与函数之间的代数，就是物理学的语法。</p>
<footer>—— 苏菲 · 热尔曼 精神续写（原句出自让-玛丽·苏里奥）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ Cannas 第8章；McDuff & Salamon 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从哈密顿向量场开始

辛结构不是静止的几何对象——它的全部意义在于**让函数产生运动**。给定一个光滑函数 $H$（能量），辛形式 $\omega$ 通过非退化性把它变成唯一一个向量场 $X_H$，积分这个向量场就是物理系统的演化。这就是哈密顿力学用几何语言重述的样子：**能量函数 + 辛结构 = 动力学方程**。更重要的是，函数之间的 **Poisson 括号**把「函数的代数」与「向量场的李代数」焊在一起——物理里的守恒量、对称性、可积性，全是这个括号的语言。这一篇是理解哈密顿流、可积系统、乃至几何量子化（「把 Poisson 代数变成算子代数」）的枢纽。<span class="marginnote">从课程地图看：上一篇讲「静态子流形」，这一篇让结构动起来。下一节《哈密顿流与辛同痕》继续这条线的整体版本。</span>

## 1 哈密顿向量场

**哈密顿向量场（Hamiltonian vector field）**：设 $(M, \omega)$ 是辛流形，$H \in C^\infty(M)$。定义向量场 $X_H$ 为满足

$$
\iota_{X_H} \omega = dH
$$

的唯一向量场，即 $\omega(X_H, \cdot) = dH(\cdot)$，或对一切 $v \in TM$：

$$
\omega(X_H, v) = dH(v)
$$

**为什么唯一存在？** 因为 $\omega$ 非退化，映射 $v \mapsto \iota_v\omega = \omega(v, \cdot)$ 是 $TM \to T^*M$ 的逐点同构——这正是第1篇的核心事实。**微分 $dH$ 通过这个同构被翻成向量场 $X_H$。** 这个操作叫作「把 1-形式变回向量场」，是 Moser 技巧里那一步的翻版。<span class="marginnote">在标准坐标 $(\mathbb{R}^{2n}, \omega_0 = \sum dq_i \wedge dp_i)$ 下，方程 $\iota_X\omega_0 = dH$ 展开为：$X = \sum_i \left( X^{q_i} \partial_{q_i} + X^{p_i} \partial_{p_i} \right)$，$\iota_X\omega_0 = \sum_i (X^{q_i} dp_i - X^{p_i} dq_i)$，对比 $dH = \sum (\partial_{q_i}H\, dq_i + \partial_{p_i}H\, dp_i)$ 得 $X^{q_i} = \partial_{p_i}H$、$X^{p_i} = -\partial_{q_i}H$。</span>

**标准坐标下的哈密顿方程：**

$$
\dot{q}_i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}
$$

这就是《理论力学》里的哈密顿方程——在这里它只是「$X_H$ 的积分曲线」的坐标写法。辛几何把「写下运动方程」变成「取一个函数、翻转微分」，不再需要任何坐标系。

## 2 例子与性质

**例1（谐振子）**：$H = \frac{1}{2}(p^2 + q^2)$ 在 $\mathbb{R}^2$ 上，$X_H = p\partial_q - q\partial_p$，积分曲线是圆 $q(t) = q_0\cos t + p_0\sin t$。能量的水平集就是运动轨道——$H$ 沿着 $X_H$ 恒为常数。

**例2（自由粒子）**：$H = p^2/2m$，$X_H = (p/m)\partial_q$，积分曲线是直线 $q(t) = q_0 + p_0 t/m$。

**核心性质：$H$ 沿 $X_H$ 是常数（能量守恒）**：

$$
\mathcal{L}_{X_H} H = dH(X_H) = \omega(X_H, X_H) = 0
$$

最后一步因为 $\omega$ 反对称。**所以哈密顿向量场把 $H$ 的水平集保持住**——几何上，$X_H$ 永远沿「等能量面」滑动，这就是能量守恒的微分几何证明。

**哈密顿向量场的李括号保持哈密顿性**：若 $X_f, X_g$ 是哈密顿向量场，则 $[X_f, X_g]$ 也是哈密顿向量场，其哈密顿函数是 $-{f, g}$（见下节）。所以哈密顿向量场构成李代数，且映射 $f \mapsto X_f$ 是一个「李代数反同态」——**函数到向量场：括号保持，符号翻转**。

## 3 Poisson 括号

**Poisson 括号（Poisson bracket）**：对 $f, g \in C^\infty(M)$，

$$
\{f, g\} := \omega(X_f, X_g) = dg(X_f) = \mathcal{L}_{X_f} g
$$

**几何含义**：$\{f, g\}$ 度量「沿 $f$ 的哈密顿流，$g$ 的变化率」。$X_f(g)$ 是 $g$ 沿 $X_f$ 的方向导数——所以 $\{f, g\}$ 同时是「$g$ 沿 $f$ 的流的变化率」和「$\omega$ 对两个哈密顿向量场的配对」。<span class="marginnote">在标准坐标下 $\{f, g\} = \sum_i \left( \partial_{q_i} f\, \partial_{p_i} g - \partial_{p_i} f\, \partial_{q_i} g \right)$。注意对易关系 $\{q_i, p_j\} = \delta_{ij}$——这就是量子力学里 $[\hat{q}_i, \hat{p}_j] = i\hbar\delta_{ij}$ 的经典原型。</span>

**Poisson 括号的四大性质**：对任意 $f, g, h$ 与常数 $\lambda$，

1. **反对称**：$\{f, g\} = -\{g, f\}$；
2. **双线性**：$\{f, \lambda g + h\} = \lambda\{f, g\} + \{f, h\}$；
3. **Jacobi 恒等式**：$\{f, \{g, h\}\} + \{g, \{h, f\}\} + \{h, \{f, g\}\} = 0$；
4. **莱布尼茨法则**：$\{f, gh\} = g\{f, h\} + h\{f, g\}$。

前三条来自「$f \mapsto X_f$ 是李代数反同态 + $\omega$ 闭」。第四条来自「$X_f$ 是导子」（方向导数）。

满足 1–3 的结构叫**李代数结构**，再加 4（莱布尼茨）就构成**泊松代数（Poisson algebra）**：$C^\infty(M)$ 既是结合代数（普通乘法）又是李代数（括号），且两者通过莱布尼茨法则相容。<span class="marginnote">这就是量子力学的代数蓝图：经典可观测量构成 Poisson 代数，量子化把它变成算子代数，Poisson 括号变成对易子 $[A,B] = AB - BA$ 除以 $i\hbar$。第3篇《几何量子化》将把这个对应做成严格构造。</span>

**辨析｜易错点：** Jacobi 恒等式不是自动的。若随便给一个反对称双线性括号，Jacobi 恒等式一般不成立。它之所以在这里成立，是因为 $\omega$ 的**闭性**：$d\omega = 0$ 是 Jacobi 恒等式的几何根源。这再次呼应第2篇——「为什么辛形式要求闭」的答案之一就是「为了让函数代数构成泊松代数」。

## 4 公式解析：$\{f,g\}$ 的三种写法

**核心公式：**

$$
\{f, g\} = \omega(X_f, X_g) = dg(X_f) = \mathcal{L}_{X_f} g = X_f(g)
$$

四种写法说的是同一件事，拆解：

- **第一种（几何）**：$\omega(X_f, X_g)$——把两个哈密顿向量场放进辛形式里配对。这最直接地使用了 $\omega$。
- **第二种（微分）**：$dg(X_f)$——因为 $\iota_{X_f}\omega = df$，$\omega(X_f, X_g) = -\omega(X_g, X_f) = -df(X_g)$。等等，符号要注意。$\omega(X_f, X_g) = (\iota_{X_f}\omega)(X_g) = df(X_g)$? 不对：$\iota_{X_f}\omega(X_g) = \omega(X_f, X_g)$，而 $\iota_{X_f}\omega = df$，所以 $\omega(X_f, X_g) = df(X_g) = X_g(f)$。但另一方面 $dg(X_f) = X_f(g)$。这两个差符号：$X_f(g) = -X_g(f)$? 因为 $\{f,g\}$ 反对称。让我核对。

$\omega(X_f, X_g)$：
- $= df(X_g)$（用 $\iota_{X_f}\omega = df$，作用在 $X_g$ 上）
- $= X_g(f)$
- $= -X_f(g)$（反对称）

而 $dg(X_f) = X_f(g) = -\omega(X_f, X_g)$。嗯，那 $\{f,g\} = \omega(X_f,X_g)$ 还是 $dg(X_f)$？

标准定义：$\{f,g\} = \omega(X_f, X_g)$。也等于 $X_f(g) = dg(X_f)$ 吗？

$X_f(g) = dg(X_f)$。而 $dg = \iota_{X_g}\omega$，所以 $dg(X_f) = \omega(X_g, X_f) = -\omega(X_f, X_g)$。

矛盾！让我仔细核对符号约定。

标准参考（Cannas da Silva, Lectures on Symplectic Geometry, Def 3.5.6 附近）：定义 $\{f,g\} = \omega(X_f, X_g)$。然后 $X_f(g) = dg(X_f) = \omega(X_g, X_f) = -\omega(X_f, X_g) = -\{f,g\}$。Hmm，这样 $\{f,g\} = -X_f(g)$？

不对。让我重查。Cannas 3.5: "Given $f \in C^\infty(M)$, the Hamiltonian vector field $X_f$ is defined by $\iota_{X_f}\omega = df$. The Poisson bracket of $f, g \in C^\infty(M)$ is $\{f, g\} = \omega(X_f, X_g)$."

然后她指出：$\{f, g\} = \omega(X_f, X_g) = -dH_g(X_f)$? 让我用她的话。实际上：$\{f,g\} = \omega(X_f, X_g) = X_f \cdot g = -X_g \cdot f$? 

验证：$\omega(X_f, X_g)$，由于 $\iota_{X_f}\omega = df$，所以 $\omega(X_f, X_g) = df(X_g) = X_g(f)$。这是正的 $X_g(f)$。

由于反对称，$\omega(X_f, X_g) = -\omega(X_g, X_f) = -X_f(g)$（用 $\iota_{X_g}\omega = dg$ 得 $\omega(X_g, X_f) = dg(X_f) = X_f(g)$）。

所以 $\{f,g\} = \omega(X_f, X_g) = X_g(f) = -X_f(g)$。

Hmm，所以 $\{f,g\} = X_g(f) = -X_f(g)$。注意下标方向！$\{f,g\}$ 是「$f$ 沿 $X_g$ 的变化」？还是「$g$ 沿 $X_f$」？这里 $X_g(f) = df(X_g)$ 是「$f$ 沿 $X_g$ 的变化率」。

等等，这与我之前写的不一致。让我用具体例子验证。

$\mathbb{R}^2$，$f = q$，$g = p$。$X_f = X_q$：$\iota_X\omega_0 = dq$。$\omega_0 = dq \wedge dp$，$\iota_{\partial_p}\omega_0 = \iota_{\partial_p}(dq \wedge dp) = dq$。所以 $X_q = \partial_p$。类似 $X_p = -\partial_q$。

$\{q, p\} = \omega_0(X_q, X_p) = \omega_0(\partial_p, -\partial_q) = -(dq \wedge dp)(\partial_p, \partial_q) = -[dq(\partial_p)dp(\partial_q) - dp(\partial_p)dq(\partial_q)] = -[0\cdot 0 - 1\cdot 1] = 1$。✓ 好，$\{q,p\} = 1$。

$X_q(p) = \partial_p(p) = 1$。所以 $\{q,p\} = X_q(p)$。也即 $\{f,g\} = X_f(g)$！等下，$X_f(g)$ 其中 $f=q, g=p$：$X_q(p) = 1$ ✓。

所以 $\{f,g\} = X_f(g) = dg(X_f)$。这与 $\omega(X_f, X_g) = df(X_g) = X_g(f)$ 一致吗？$\{q,p\} = X_q(p) = 1$，而 $X_p(q) = -\partial_q(q) = -1 = -\omega(X_q, X_p)$... 

$\omega(X_q, X_p) = \omega(\partial_p, -\partial_q)$。$\omega_0(\partial_p, -\partial_q) = dq\wedge dp(\partial_p, -\partial_q) = dq(\partial_p)dp(-\partial_q) - dp(\partial_p)dq(-\partial_q) = 0 - 1\cdot(-1) = 1$。✓ $\omega(X_q, X_p) = 1$。

所以 $\{f,g\} = \omega(X_f, X_g) = X_f(g) = dg(X_f)$，同时 $= -X_g(f) = -df(X_g)$? 检查：$X_f(g) = dg(X_f) = 1$（上面）。$-df(X_g)$? $df(X_g) = dq(X_p) = dq(-\partial_q) = -1$，取负得 $1$ ✓。所以 $\{f,g\} = X_f(g) = -X_g(f)$。

好，结论：**$\{f,g\} = \omega(X_f, X_g) = X_f(g) = dg(X_f) = -X_g(f) = -df(X_g)$。**

我之前草稿里写 $\{f,g\} = dg(X_f) = \mathcal{L}_{X_f}g = X_f(g)$ 是对的！而「$g$ 沿 $f$ 的流的变化率」的说法也对应 $X_f(g)$。好，我原稿没问题。

那我第3节写 $\{f, g\} := \omega(X_f, X_g) = dg(X_f) = \mathcal{L}_{X_f} g$ 正确。✓

好，继续。第4节四种写法：$\omega(X_f, X_g) = dg(X_f) = \mathcal{L}_{X_f}g = X_f(g)$。全等。✓

那我说「$\{f,g\}$ 度量『沿 $f$ 的哈密顿流，$g$ 的变化率』」正确（$X_f(g)$ 是 $g$ 沿 $X_f$ 的变化率）。

好。现在公式解析第4节我就按这个写，不需要修正已有内容——但我需要写这个文件。让我写。

再验证一下 Jacobi 恒等式与闭性：$\{f, \{g,h\}\} = X_f(\{g,h\})$... Jacobi 等价于 $[X_f, X_g] = -X_{\{f,g\}}$ 且 $\omega$ 闭。标准。✓

## 5 Poisson 流形与 Casimir

Poisson 括号可以脱离辛结构独立定义：**Poisson 流形（Poisson manifold）**是带泊松括号 $\{\cdot,\cdot\}$ 的流形，不必非退化。辛流形是「处处非退化的 Poisson 流形」；一般 Poisson 流形在退化方向上会有分层。

**Casimir 函数**：与一切函数 Poisson 对易的函数 $C$：$\{C, f\} = 0$ 对所有 $f$。在辛流形上，Casimir 只有常数（因为 $\{C, f\} = X_C(f) = 0$ 对所有 $f$ 推出 $X_C = 0$，非退化推出 $dC = 0$）。但在非退化分层的 Poisson 流形上，Casimir 是「每一层上的常数」，刻画层的几何。<span class="marginnote">经典例子：$\mathbb{R}^3$ 上的叉积括号 $\{f,g\}(x) = x \cdot (\nabla f \times \nabla g)$ 给出 Poisson 结构，球面 $|x| = r$ 是辛叶，$C = |x|^2$ 是 Casimir。刚体运动的角动量守恒正是 $C$ 沿流不变。</span>

## 6 小结

- **哈密顿向量场 $X_H$**：$\iota_{X_H}\omega = dH$ 唯一确定；标准坐标下就是哈密顿方程 $\dot{q} = \partial_pH$、$\dot{p} = -\partial_qH$。
- **能量守恒**：$X_H(H) = \omega(X_H, X_H) = 0$，$H$ 沿流是常数。
- **Poisson 括号** $\{f,g\} = \omega(X_f, X_g) = X_f(g)$：满足反对称、双线性、**Jacobi（来自 $d\omega = 0$）**、莱布尼茨。
- **泊松代数**：$C^\infty(M)$