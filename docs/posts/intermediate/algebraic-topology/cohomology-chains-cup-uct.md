---
title: 上同调：上链、杯积与万有系数定理
date: 2026-08-07
---

# 上同调：上链、杯积与万有系数定理

<div class="epigraph">
<p>上同调是同调的镜像：箭头反转，洞变成「绕洞的函子」。</p>
<footer>—— 代数拓扑学习者的概括（出处佚名）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第3.1、3.2章 ｜ 2026-08-07</p>
</div>

## 为什么从复习上同调开始

前面六节复习了同调的家族：单纯、奇异、正合列、胞腔。本节复习它的**镜像——上同调（cohomology）**。名字只差一个字，性质却翻了个身：链变成**上链**、箭头**反转**、同态由「推出」变「拉回」。多出的回报是两件同调没有的宝贝——**杯积**（把上同调类乘起来，构成环）与**万有系数定理**（把同调与上同调用 $\mathrm{Ext}$ 精确对接）。<span class="marginnote">为什么要学上同调？因为「映射把函数拉回来」是几何里更自然的操作（拉回微分形式、拉回度量），而上同调正是「拉回」的天然载体。从代数角度，同调是协变函子、上同调是反变函子，两者配合才构成完整的代数拓扑机器。</span>

复习的路线：上链与上同调群 → 反变函子性 → 杯积与上同调环 → 万有系数定理把三件套收口。

## 1 复习：上链复形与上同调群

把奇异链复形「转置」：定义**上链群（cochain group）**

$$C^n(X; G) = \operatorname{Hom}(C_n(X), G)$$

即「$n$-链到系数群 $G$ 的群同态」，元素叫 $n$-**上链（cochain）**。边界算子 $\partial$ 诱导**上边界算子（coboundary）**

$$\delta^n : C^n \to C^{n+1}, \qquad (\delta \varphi)(\sigma) = \varphi(\partial \sigma)$$

即「先取链的边界、再读上链的值」。$\delta^2 = 0$ 由 $\partial^2 = 0$ 自动继承，得到**上链复形**：

$$\cdots \to C^{n-1} \xrightarrow{\delta} C^n \xrightarrow{\delta} C^{n+1} \to \cdots$$

**上同调群（cohomology group）**：

$$H^n(X; G) = \frac{\ker \delta^n}{\operatorname{im}\delta^{n-1}} = \frac{\text{上循环}}{\text{上边界}}$$

**核心观察：方向相反。** 同调的箭头向下（$\partial$ 降维），上同调的箭头向上（$\delta$ 升维）；同调的循环「围住洞」，上同调的循环「探测洞」——上循环是「洞上的计分器」。<span class="marginnote">「探测」而非「围住」，是理解上同调的钥匙：一个上循环 $\varphi$ 对每个循环给出一个数（在 $G$ 中），洞越多、可测的函数越多。这正与微分形式的语言吻合：积分就是「上链对上链的配对」。</span>

## 2 复习：反变函子性——拉回

连续映射 $f : X \to Y$ 把奇异单形 $\sigma$ 送到 $f\circ\sigma$，于是把上链「拉回」：

$$f^\# : C^n(Y) \to C^n(X), \qquad f^\#(\varphi) = \varphi \circ f$$

注意方向：**映射向前、上链向后**——$f^\#$ 是反变的。它保边界（$\delta f^\# = f^\#\delta$），诱导上同调同态

$$f^* : H^n(Y) \to H^n(X)$$

且 $(g\circ f)^* = f^* \circ g^*$。**同伦不变性照旧：$f \simeq g \Rightarrow f^* = g^*$。** 于是 $H^n$ 是反变同伦不变量。<span class="marginnote">「拉回」在几何里无处不在：微分形式的拉回 $f^*\omega$、度量/联络的拉回、示性类的拉回——上同调的 $f^*$ 是这一切的拓扑原型。第 12 篇复习的 Stiefel–Whitney 类的自然性（$f^* w_i = w_i(f^*E)$）正是 $f^*$ 的实战。</span>

**例：$S^1$ 的上同调。** $H^0(S^1) = H^1(S^1) = \mathbb{Z}$，与同调维数相同（这里无挠）。$z \mapsto z^k$ 诱导 $H^1(S^1) \to H^1(S^1)$ 是「乘 $k$」——**映射度在 $f^*$ 上同样可读**。

## 3 复习：杯积与上同调环

同调群只是群，上同调群却能**乘起来**——这是上同调压倒同调的真正原因。

**杯积（cup product）** $\smile$：对 $\varphi \in H^k(X)$、$\psi \in H^l(X)$，定义 $\varphi \smile \psi \in H^{k+l}(X)$：

$$(\varphi \smile \psi)(\sigma) = \varphi(\sigma|_{[v_0,\dots,v_k]}) \cdot \psi(\sigma|_{[v_k,\dots,v_{k+l}]})$$

即「把 $(k+l)$-单形切成前 $k$ 面与后 $l$ 面，分别让两个上循环去读，再相乘」。杯积是**分级交换（graded commutative）**的：

$$\varphi \smile \psi = (-1)^{kl}\, \psi \smile \varphi$$

**上同调环（cohomology ring）**：$H^*(X; R) = \bigoplus_n H^n(X; R)$ 配杯积，构成**分级环**。<span class="marginnote">分级交换是「交换性」的分数版本：两个同维度（$kl$ 奇）元素交换会变号。$\mathbb{RP}^n$ 的 $H^* = \mathbb{Z}[\alpha]/(\alpha^{n+1})$（$\deg\alpha = 1$）中 $\alpha^2$ 的变号，正体现了「扭」的代数印记。</span>

**例：上同调环一瞥。** $H^*(S^n) = \mathbb{Z}[\alpha]/(\alpha^2)$（$\deg\alpha = n$）；$H^*(T^2) = \mathbb{Z}[a,b]/(a^2, b^2)$（$\deg a = \deg b = 1$），其中 $ab$ 是「横竖各绕一圈」的 2-类——**杯积把「横竖两个独立循环」的交叉信息编码成一个 2-维类**。杯积比直和更精细：$S^2 \times S^2$ 与 $\mathbb{CP}^2 \# \mathbb{CP}^2$ 的同调群一样，上同调环却不同（前者无三次幂、后者 $\alpha^2 \neq 0$）——**同调分不出、上同调环分得出**。

## 4 复习：万有系数定理——同调与上同调的桥梁

上同调不是独立的新信息：它由同调 + 系数群决定，这就是**万有系数定理（universal coefficient theorem，UCT）**。

**上同调万有系数定理**：对链复形 $C$ 与阿贝尔群 $G$，

$$0 \to \operatorname{Ext}^1(H_{n-1}(C), G) \to H^n(C; G) \to \operatorname{Hom}(H_n(C), G) \to 0$$

此短正合序列**分裂**（但不自然），故 $H^n(X; G) \cong \operatorname{Hom}(H_n(X), G) \oplus \operatorname{Ext}^1(H_{n-1}(X), G)$。<span class="marginnote">$\mathrm{Ext}$ 项负责「挠对挠」：当 $H_{n-1}$ 有 $\mathbb{Z}_m$ 挠、$G$ 取 $\mathbb{Z}_m$ 时，$\mathrm{Ext}$ 贡献对偶的 $\mathbb{Z}_m$。没有 Ext，挠信息会在「对偶」时丢失。</span>

**两个立即读出**：

- 系数取域 $F$（如 $\mathbb{Q}, \mathbb{Z}_p$）时 $\mathrm{Ext} = 0$，$H^n(X; F) \cong \operatorname{Hom}(H_n(X;F), F)$——上同调就是同调的**向量空间对偶**。
- 整数系数的挠：$H_n(X) = \mathbb{Z}^r \oplus \bigoplus \mathbb{Z}_{m_j}$ 时，$H^n(X) \cong \mathbb{Z}^r \oplus \bigoplus \mathbb{Z}_{m_j}$（$\mathrm{Ext}(\mathbb{Z}_{m},\mathbb{Z}) = \mathbb{Z}_m$）。**$\mathbb{RP}^2$：$H_1 = \mathbb{Z}_2$，$H^2(\mathbb{RP}^2) = \mathrm{Ext}(\mathbb{Z}_2, \mathbb{Z}) = \mathbb{Z}_2$**——上同调「知道」同调的挠，还把它升了一维。

## 5 公式解析：杯积的定义

$$(\varphi \smile \psi)(\sigma) = \varphi(\sigma|_{[v_0,\dots,v_k]}) \cdot \psi(\sigma|_{[v_k,\dots,v_{k+l}]})$$

- **第一步，切单形**：$(k+l)$-单形 $\sigma$ 切成「前 $k$ 面」$[v_0,\dots,v_k]$ 与「后 $l$ 面」$[v_k,\dots,v_{k+l}]$——两片共享顶点 $v_k$。
- **第二步，分别读数**：$\varphi$ 读前片、$\psi$ 读后片，各得 $G$ 中元素。
- **第三步，相乘取积**：两个读数在 $G$ 中相乘，得到 $k+l$ 维上循环对 $\sigma$ 的值。**「先切、再读、后乘」三步，就是杯积的全部。**

一句话：**杯积把「两个独立方向的探测」乘成一个更高维的探测；分级交换律让「方向顺序」在代数里显形。**

## 6 核心对比：同调 vs 上同调

| 性质 | 同调 $H_n$ | 上同调 $H^n$ |
| --- | --- | --- |
| 函子方向 | 协变 $f_*$ | 反变 $f^*$（拉回） |
| 维度方向 | $\partial$ 降维 | $\delta$ 升维 |
| 几何角色 | 循环「围住」洞 | 上循环「探测」洞 |
| 附加结构 | 群（一般无乘法） | 环（杯积） |
| 与系数的关系 | 万有系数定理（$\mathrm{Tor}$） | 万有系数定理（$\mathrm{Ext}$） |

**辨析｜易错点：**

- **$\delta = \varphi \circ \partial$ 的方向**：上边界算子是「先取链边界再读值」——写成 $(\delta\varphi)(\sigma) = \varphi(\partial\sigma)$，别把 $\partial$ 与 $\delta$ 的先后弄反。
- **杯积是「切前片、后片」，不是「全链」**：$\varphi \smile \psi$ 作用在 $(k+l)$-链上，两个上循环各自只读一半；「全读」不是杯积。
- **分级交换有变号**：$\varphi \smile \psi = (-1)^{kl} \psi \smile \varphi$；遗忘 $(-1)^{kl}$ 会让上同调环计算全错。
- **UCT 分裂不自然**：$H^n \cong \mathrm{Hom}(H_n,G) \oplus \mathrm{Ext}(H_{n-1},G)$ 的分裂不自然（依赖基选择）；「对偶地读出」只在域系数下才是字面的向量空间对偶。

## 7 小结

- **上链与上同调**：$C^n = \operatorname{Hom}(C_n, G)$，$\delta = \cdot \circ \partial$，$H^n = \ker\delta/\operatorname{im}\delta$——箭头反转的镜像。
- **反变函子**：$f^* : H^n(Y) \to H^n(X)$ 拉回；同伦不变，$f \simeq g \Rightarrow f^* = g^*$。
- **杯积**：切单形、分别读、相乘；分级交换 $(-1)^{kl}$；$H^*(X)$ 是分级环。
- **万有系数定理**：$0 \to \mathrm{Ext}(H_{n-1},G) \to H^n \to \operatorname{Hom}(H_n,G) \to 0$ 分裂；域系数下上同调 = 同调的向量对偶。
- 复习口诀：**反变拉回、上链升维、杯积成环、Ext 管挠**。
- 与课程的连接：上同调是《微分拓扑》中 de Rham 上同调（微分形式的闭/恰当）的拓扑骨架；杯积与上同调环是第 10 篇复习（Poincaré 对偶）与第 12 篇复习（示性类）的运算平台。

在下一节，我们复习上同调在流形上的最大奖赏——**Poincaré 对偶**：定向闭流形的 $H^n$ 与 $H_{m-n}$ 如何互为镜像。
