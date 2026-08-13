---
title: 余切丛与机械辛结构
date: 2026-08-07
---

# 余切丛与机械辛结构

<div class="epigraph">
<p>自然界的全部奥秘都藏在相空间里，而相空间就是余切丛。</p>
<footer>—— 让-马里 · 苏里奥（Jean-Marie Souriau，法语转述）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第3章；Cannas 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从余切丛开始

上一篇给出了辛流形的抽象定义，但还没有一个「取之不尽」的例子来源。余切丛就是辛流形最重要、最自然、也是**唯一一个对任何流形都存在**的辛流形。为什么？因为任何力学系统的构型空间 $Q$（一个流形）一旦给定，它的**相空间**就该是 $T^*Q$——位置 $q$ 是 $Q$ 的点，动量 $p$ 是余切向量。余切丛上有一个与所有坐标选择无关的**典范辛结构**，它从「微分」这个操作本身生长出来，不需要任何额外数据。这一篇我们构造这个结构，并证明它在平凡情形下就是标准辛形式——这为下一节的 Darboux 定理埋下伏笔。<span class="marginnote">在课程地图上，这一篇是《理论力学》相空间讲法的几何化：那里的正则坐标 $(p, q)$ 与正则变换，在这里就是余切坐标与辛同胚。</span>

## 1 余切丛与刘维尔 1-形式

设 $Q$ 是 $n$ 维光滑流形，记 $M = T^*Q$。$M$ 上的点记作 $(q, p)$，其中 $q \in Q$，$p \in T_q^* Q$ 是 $q$ 处的余切向量。$M$ 是 $2n$ 维流形，自然投影

$$
\pi: T^*Q \longrightarrow Q, \qquad (q, p) \mapsto q
$$

**刘维尔 1-形式（Liouville 1-form）** $\lambda \in \Omega^1(T^*Q)$ 定义如下：对 $(q,p) \in T^*Q$ 与切向量 $v \in T_{(q,p)}(T^*Q)$，

$$
\lambda_{(q,p)}(v) := p\big( d\pi_{(q,p)}(v) \big)
$$

也就是说：先把 $v$ 投影到 $Q$ 上的切向量 $d\pi(v)$，再让余向量 $p$ 作用上去。<span class="marginnote">这个定义初看绕，但意义清晰：$\lambda$ 在点 $(q,p)$ 处只「看」$p$ 这个余向量，并把它作用在 $v$ 的投影上。它是「坐标无关」的——没有选任何坐标系就写出来了，这是「典范」一词的含义。</span>

在局部坐标 $(q_1, \dots, q_n, p_1, \dots, p_n)$ 下，$\pi$ 是投影，$d\pi(v)$ 的分量是 $v$ 的前 $n$ 个分量，而 $p = \sum p_i dq_i$ 作用上去得 $\sum p_i v_{q_i}$。于是

$$
\lambda = \sum_{i=1}^{n} p_i \, dq_i
$$

这就是你熟悉的「$p\, dq$」。**注意它不含 $dp$ 项**——这是整个构造的机关。

## 2 典范辛形式

**典范辛形式（canonical symplectic form）**：

$$
\omega_{\mathrm{can}} := -d\lambda
$$

在局部坐标下

$$
\omega_{\mathrm{can}} = -d\left( \sum_i p_i dq_i \right) = \sum_{i=1}^{n} dq_i \wedge dp_i
$$

验证它是辛形式：闭性 $d\omega_{\mathrm{can}} = -d^2\lambda = 0$ 自动成立（外微分二次为零）；非退化性来自局部坐标下它是标准形式 $\sum dq_i \wedge dp_i$——这正是上一篇的标准辛形式 $\omega_0$。<span class="marginnote">符号约定要小心：不同教材把 $\omega_{\mathrm{can}}$ 写成 $-d\lambda$ 或 $+d\lambda$。这里取 $-d\lambda$ 得到 $\sum dq_i \wedge dp_i$，与「位置-动量」的标准配对一致；若取 $+d\lambda$ 则得到 $\sum dp_i \wedge dq_i$，二者差一个整体符号，是同构的辛结构。</span>

**关键观察：** 余切丛的辛结构完全是「微分出来的」，不需要度量、不需要任何几何选择。任何流形 $Q$ 的余切丛都自动是辛流形。这就是为什么余切丛是辛几何的「标准入口」——也解释了为什么辛几何与哈密顿力学、变分法联系如此紧密。

## 3 切丛是「位置-速度」，余切丛是「位置-动量」

为什么是余切丛而不是切丛？对比一下：

**切丛 $TQ$**：元素是 $(q, \dot{q})$，即位置与速度。它上面的几何结构需要额外输入——通常来自一个拉格朗日函数 $L$（动能减势能），结构依赖于 $L$。
**余切丛 $T^*Q$**：元素是 $(q, p)$，即位置与动量。它自带典范辛结构，不依赖任何额外数据。

两者的桥梁是**勒让德变换（Legendre transform）**：在超正则情形下，$p_i = \partial L / \partial \dot{q}_i$ 给出 $TQ \to T^*Q$ 的微分同胚，把拉格朗日方程变成哈密顿方程。辛几何选择的舞台是余切丛，因为它把「结构从何而来」的问题一次性解决：**辛结构就是相空间的固有几何**。<span class="marginnote">对理解大模型的读者，一个类比：切丛像「模型参数 + 梯度」的坐标表示，余切丛像「参数 + 对偶变量」——对偶空间的结构（比如正则化、对偶问题）往往比原始空间更本质。辛几何正是「站在对偶这一边」的几何。</span>

## 4 公式解析：从 $\lambda$ 到 $\omega_{\mathrm{can}}$

**核心公式：**

$$
\omega_{\mathrm{can}} = -d\lambda = \sum_{i=1}^{n} dq_i \wedge dp_i
$$

三步拆解：

- **第一步，先算 $d\lambda$**：$\lambda = \sum_i p_i dq_i$，对它取外微分，按莱布尼茨法则 $d(p_i dq_i) = dp_i \wedge dq_i$（$p_i$ 与 $dq_i$ 的楔积次序照抄）。于是 $d\lambda = \sum_i dp_i \wedge dq_i$。
- **第二步，换次序**：$dp_i \wedge dq_i = -dq_i \wedge dp_i$，所以 $-d\lambda = \sum_i dq_i \wedge dp_i$。这一步的符号来自反对称性，也正是「为什么取 $-d$ 才得到漂亮形式」的原因。
- **第三步，看成标准形式**：$dq_i \wedge dp_i$ 与上一篇的标准形式 $\omega_0 = \sum dx_i \wedge dy_i$ 结构完全相同——只是把坐标改名成 $(q_i, p_i)$。所以局部上，余切丛的辛结构就是标准辛形式。

**物理直觉：** $\sum dq_i \wedge dp_i$ 度量的是相空间中的**定向面积元**。相体积 $\int dq_1 dp_1 \cdots dq_n dp_n$（Liouville 定理说它在哈密顿流下不变）正是 $n!$ 分之一乘上 $\omega^n$ 的积分。这一条公式把「微分几何的对象」和「统计力学的相体积」焊在了一起。

## 5 正则坐标与正则变换

局部坐标 $(q_i, p_i)$ 使得 $\omega_{\mathrm{can}} = \sum dq_i \wedge dp_i$ 的，叫**正则坐标（canonical coordinates）**。**正则变换（canonical transformation）**就是保持 $\omega_{\mathrm{can}}$ 的微分同胚——也就是余切丛上的辛同胚。

经典力学教材里的「生成函数」理论在这里获得几何翻译：一个辛同胚由母函数 $S(q, Q)$ 生成，因为

$$
p \, dq - P \, dQ = dS
$$

左边两个 1-形式的差等于某个函数的全微分，这正是「拉回保持辛形式」的势形式说法。<span class="marginnote">生成函数的四种类型（$S(q,Q)$、$S(q,P)$、$S(p,Q)$、$S(p,P)$）对应辛同胚与坐标变换的不同参数化。它们在量子力学的 WKB 近似与路径积分里也扮演角色，那里正则变换对应么正变换。</span>

**辨析｜易错点：** 余切丛的辛结构 $\omega_{\mathrm{can}} = \sum dq_i \wedge dp_i$ 看起来「平凡」，容易让人以为所有辛流形都长这样。这是错的——Darboux 定理说的是**局部**每个辛流形都长这样，但**全局**可以拧成各种拓扑（比如 $S^2$、$T^4$ 等不能写成任何 $T^*Q$ 的紧辛流形）。余切丛的辛结构是「典范但非平凡」的起点，而不是全部。

## 6 刘维尔 1-形式的几何意义

$\lambda$ 不只是「$\omega_{\mathrm{can}}$ 的势」，它本身携带几何：$\lambda$ 是 $T^*Q$ 上**唯一（在某种意义下）最自然的 1-形式**，满足对任何 1-形式 $\alpha \in \Omega^1(Q)$（看作截面 $s_\alpha: Q \to T^*Q$）有

$$
s_\alpha^* \lambda = \alpha
$$

**回拉还原**：把 $\lambda$ 沿任意 1-形式「拉回」到 $Q$，正好还原该 1-形式。这是 $\lambda$ 的**泛性质**——它「生成」所有 1-形式，正如 $\omega_{\mathrm{can}}$「生成」辛结构。<span class="marginnote">泛性质的语言：$\lambda \in \Omega^1(T^*Q)$ 是「万有的」，因为任何 $\alpha \in \Omega^1(Q)$ 都是 $s_\alpha^*\lambda$。在范畴论里，$T^*Q$ 是「余切函子」的对象，$\lambda$ 是它的自然变换——这个观点在几何量子化与几何表示论里反复出现。</span>

**为什么 $\lambda$ 不含 $dp$ 项**：$\lambda = \sum p_i dq_i$ 只「看」$Q$ 方向。若加 $dp$ 项，泛性质 $s_\alpha^*\lambda = \alpha$ 会被破坏（拉回后多出 $\alpha$ 的导数项）。**$\lambda$ 的「无 $dp$」是它成为「势」的充分条件**——这与量子力学里「$p dq$ 是相积分的被积函数」直接相关（路径积分 $\oint pdq$ 的几何本质）。

**与辛势的关系**：$\omega_{\mathrm{can}} = -d\lambda$ 说明辛形式是「$\lambda$ 的曲率」的负值——$\lambda$ 是 $T^*Q$ 上联络的势（在平凡线束上）。这为几何量子化（第3篇）埋线：前量子线束的联络势正是 $-\lambda$，整性条件 $[\omega] \in H^2(M;\mathbb{Z})$ 在余切丛上自动满足（$H^2(T^*Q) = 0$）。

## 7 小结

- **余切丛 $T^*Q$** 对任何流形 $Q$ 都有**典范辛结构** $\omega_{\mathrm{can}} = -d\lambda$，$\lambda = \sum p_i dq_i$ 是刘维尔 1-形式。
- **刘维尔 1-形式**坐标无关地定义为「先投影、再用余向量作用」，不含 $dp$ 项。
- **闭性自动成立**（$d^2 = 0$），局部上 $\omega_{\mathrm{can}} = \sum dq_i \wedge dp_i$ 就是标准辛形式。
- **切丛 vs 余切丛**：切丛装速度、结构依赖拉格朗日量；余切丛装动量、结构自足——这是辛几何选余切丛为舞台的原因。
- **正则坐标与正则变换**是辛坐标与辛同胚在力学中的名字；相体积保持（Liouville）是 $\omega^n/n!$