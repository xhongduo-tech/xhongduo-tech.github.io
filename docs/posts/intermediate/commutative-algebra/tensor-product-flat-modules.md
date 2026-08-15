---
title: 张量积与平坦模
date: 2026-08-07
---

# 张量积与平坦模

<div class="epigraph">
<p>张量积把「双线性」的一生压缩成一个线性对象；平坦性则是它「不破坏正合」的品格。</p>
<footer>—— 交换代数课堂传统（H. Cartan 与 S. Eilenberg 之后）</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Atiyah–Macdonald Ch. 2-3 ｜ 2026-08-07</p>
</div>

## 为什么从张量积开始

第一级《线性代数》里你已见过向量空间的张量积：把双线性映射「改写」成线性映射。模论的张量积是它的直接推广，但这里有一个深刻的新现象——**张量积不是精确的**。它保持右正合，却可能撕破左正合。那些「不撕破」的模，就是**平坦模**。局部化（第1篇《局部化》）是平坦模最重要的例子，而「平坦 = 纤维连续变化」是代数几何里「族」概念的基础。<span class="marginnote">「平坦」（flat）一词来自代数几何的「平坦族」（flat family）：一个簇的纤维随参数连续变化、维数稳定不爆点，就叫平坦。Serre 1950 年代把「平坦」从几何直觉提炼成「张量积正合」的代数定义，从此「族」变得可算。</span>

这一篇把张量积、平坦性、忠实平坦性一次讲清，并把局部化、基变换、纤维三个最常用的操作统统收编到「张量积」这把钥匙之下。

## 1 张量积：万有的双线性

**张量积（tensor product）**：$A$-模 $M, N$ 的张量积 $M \otimes_A N$ 是满足如下**万有性质**的 $A$-模：对任何 $A$-模 $P$ 与双线性映射 $f: M \times N \to P$，存在唯一线性映射 $\bar{f}: M \otimes_A N \to P$ 使下图交换

$$M \times N \xrightarrow{\otimes} M \otimes_A N \xrightarrow{\bar{f}} P, \qquad f = \bar{f} \circ \otimes.$$

构造：以自由模 $A^{(M \times N)}$ 为基，模去关系（双线性条件的全部零元）即可。纯张量记作 $m \otimes n$。<span class="marginnote">直觉：「$M \otimes N$ 是把 $M$ 与 $N$ 的『乘积』线性化」。第一级线性代数里 $V \otimes W$ 的维数是 $\dim V \cdot \dim W$；模论里没有维数，但「$\mathbb{Z}/2 \otimes \mathbb{Z}/3 = 0$」这类算术依然惊人地精确。</span>

基本算例：
- $\mathbb{Z}/m \otimes_{\mathbb{Z}} \mathbb{Z}/n = \mathbb{Z}/\gcd(m,n)$；
- $A/I \otimes_A M = M/IM$（**基变换**：取商就「模掉 $I$」）；
- $S^{-1}A \otimes_A M = S^{-1}M$（**局部化 = 张量积**，见第1篇）。

**核心对照表：张量积的算术**

| 对象 | 结果 | 直觉 |
| --- | --- | --- |
| $\mathbb{Z}/m \otimes_{\mathbb{Z}} \mathbb{Z}/n$ | $\mathbb{Z}/\gcd(m,n)$ | 两个周期的公共部分 |
| $A/I \otimes_A M$ | $M/IM$ | 取商 = 模掉 $I$ |
| $S^{-1}A \otimes_A M$ | $S^{-1}M$ | 局部化 = 张量积 |
| $k[x] \otimes_k k[y]$ | $k[x,y]$ | 多项式环的张量 = 多变量 |
| $k \otimes_k k$ | $k$ | 域的张量还是域 |
| $M \otimes_A N$（$N$ 平坦） | 保持正合 | 平坦 = 不撕断 |

这串算例的共同点是：**张量积把「结构」组合起来，但组合的代价是放弃「分解信息」**——$\gcd$ 就是两个周期「取公共部分」的结果。

## 2 张量积的正合性：只保一半

对短正合列 $0 \to M' \to M \to M'' \to 0$，张量积后一般**不**再正合。准确说：

**重点：$-\otimes_A N$ 是右正合函子，但不一定左正合。** 对 $M' \to M \to M'' \to 0$ 正合，则

$$M' \otimes N \to M \otimes N \to M'' \otimes N \to 0$$

正合；但「$0 \to M'$」这一段可能被拉断。经典例子：$0 \to \mathbb{Z} \xrightarrow{\cdot 2} \mathbb{Z}$ 正合，张量 $\mathbb{Z}/2$ 后变成 $\mathbb{Z}/2 \xrightarrow{\cdot 2} \mathbb{Z}/2$，$\cdot 2 = 0$ 非单射——左正合丢失。<span class="marginnote">丢失的信息由 $\operatorname{Tor}_1^A(N, \cdot)$ 记录：张量积的正合性「缺掉一段」，正是一段高阶同调在计数。$\operatorname{Tor}$ 与张量积形影不离，就像 $\operatorname{Ext}$ 与 $\operatorname{Hom}$。</span>

把「撕断」的例算到底：$0 \to \mathbb{Z} \xrightarrow{\cdot 2} \mathbb{Z} \to \mathbb{Z}/2 \to 0$ 正合，张量 $\mathbb{Z}/2$ 后中间映射变零映射，单射段丢成核 $\mathbb{Z}/2$——丢失的那段由 $\operatorname{Tor}_1^{\mathbb{Z}}(\mathbb{Z}/2, \mathbb{Z}/2) = \mathbb{Z}/2$ 精确补齐。**$\operatorname{Tor}$ 的使命，就是把「张量积丢掉的正合段」原样记账**；平坦模正是「账目恒为零」的模。

**辨析｜易错点：** 张量积对两个变量都右正合、都未必左正合，**对称**。不要以为「$N$ 在右边所以更安全」——$\cdot 2$ 的例子里 $N = \mathbb{Z}/2$ 在右边，照样撕断。判断张量积是否丢正合，只看「$N$ 是否平坦」这一个属性。

## 3 平坦模：正合性的守护者

**平坦模（flat module）**：$A$-模 $M$ 称为平坦的，若 $M \otimes_A -$ 是正合函子（即同时左正合）。

等价条件（Noether 环境中）：
对一切 $N$ 有 $\operatorname{Tor}_1^A(M, N) = 0$；
- $M/I M \to$ 之类的保持单射成立（**Bourbaki 判别法**：$I \otimes M \to M$ 对一切有限生成理想 $I$ 是单射）。

**忠实平坦（faithfully flat）**：$M \otimes N = 0 \Rightarrow N = 0$（张量积「不把非零模杀成零」）。忠实平坦还推出「$M \otimes N' \to M \otimes N$ 单射 ⇒ $N' \to N$ 单射」这类可逆性——平坦 + 忠实，让张量积几乎成为「可逆的操作」。

标准例子：
- **自由模、投射模都平坦**（自由模显然，投射是自由的和）。
- **局部化是平坦的**：$S^{-1}A$ 是平坦 $A$-模（第1篇《局部化》已见正合性保持）。
- $M = k[x]/(x)$ 在 $k[x]$ 上**不**平坦：$0 \to (x) \to k[x] \to M \to 0$ 正合，张量 $M$ 得 $(x)\otimes M \to M \to M\otimes M \to 0$；因 $(x) \cong k[x]$ 自由，$(x)\otimes M \cong M$，而映射 $M \to M$ 是零映射，故 $\operatorname{Tor}_1(M,M) = M \neq 0$——$x$ 这个零因子把「平坦」杀死。

**重点：平坦 = 张量积「不撕断」；忠实平坦 = 张量积「可逆」。** 局部化、自由模是每天都要用的平坦模；几何上「平坦族」保证纤维维数连续，而「忠实平坦下降」是「从基变换回推性质」的总开关。<span class="marginnote">例：$A \to B$ 忠实平坦且 $B$ Noether ⇒ $A$ Noether；「下降」问题（descent）——在 $B$ 上成立的命题什么时候能在 $A$ 上成立——完全由忠实平坦性统治。Grothendieck 把它发展为一大套下降理论。</span>

忠实平坦的「可逆性」有一个极具威力的用法：**忠实平坦下降（descent）**。若 $A \to B$ 忠实平坦且「$B$ 上某性质成立」，很多情形下能回推「$A$ 上也成立」——例如 $B$ Noether 且忠实平坦 ⇒ $A$ Noether。Grothendieck 把「下降」发展成整套语言：先到更大的环上去验证，再「降」回原环，成为现代代数几何的底层设施。

## 4 平坦性与代数几何：纤维的观点

设 $A \to B$ 是环同态，$\mathfrak{p} \in \operatorname{Spec} A$ 对应**纤维环（fiber ring）**

$$B \otimes_A \kappa(\mathfrak{p}), \qquad \kappa(\mathfrak{p}) = A_{\mathfrak{p}}/\mathfrak{p}A_{\mathfrak{p}}.$$

**重点：$B$ 在 $A$ 上平坦，直观地意味着「$\operatorname{Spec} B \to \operatorname{Spec} A$ 的纤维维数处处连续」。** 一个标准结论：平坦 + 有限型 ⇒ 纤维维数是上半连续的（且局部为常数当 $A$ 正规时）。「平坦」把「族」从直观升格为可判定的代数性质。<span class="marginnote">对照：$k[x,y]/(xy)$ 投影到 $k[x]$ 的纤维在 $x \neq 0$ 处是点、在 $x=0$ 处突然变成直线——纤维维数「跳变」，环同态不平坦。几何直觉在这里与代数定义完全吻合。</span>

**公式解析：局部化 = 张量积**。把第1篇的核心公式重写成张量积语言：

$$S^{-1}M \;\cong\; M \otimes_A S^{-1}A.$$

- **第一步，两个构造相遇**：$M \otimes_A S^{-1}A$ 的纯张量 $m \otimes \tfrac{a}{s} = m \cdot \tfrac{a}{s}$，可当作 $M$ 的「带分母元素」；映射 $m \otimes \tfrac{a}{s} \mapsto \tfrac{am}{s}$ 与反方向 $\tfrac{m}{s} \mapsto m \otimes \tfrac1s$ 互逆。
- **第二步，为什么平坦**：$S^{-1}(-)$ 正合（第1篇），而它等于 $- \otimes S^{-1}A$，故 $S^{-1}A$ 平坦。
- **第三步，统一的钥匙**：局部化、商模（基变换）、纤维都是「张量积某个模」——**张量积是环论里唯一的「换基」操作**。

**辨析｜易错点：** 平坦 ≠ 自由。$S^{-1}A$（如 $\mathbb{Z}_{(p)}$）一般不是自由 $\mathbb{Z}$-模，却平坦；反过来自由模必平坦。用「Tor₁ = 0」或 Bourbaki 判别法判断，别用「有没有基」。

把三个「换基」操作并排看：局部化 $S^{-1}M = M \otimes S^{-1}A$、商模 $M/IM = M \otimes A/I$、纤维 $B \otimes_A \kappa(\mathfrak{p})$——全部是「张量积某个模」。**张量积是环论里唯一的「换基」运算**：换环、取商、看纤维，共用一把钥匙，这也正是它在本专题出场率最高的原因。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| 张量积 $M \otimes_A N$ | 双线性万有的线性化 |
| 右正合 | $-\otimes N$ 保持右端正合 |
| 平坦模 | 张量积正合，$\operatorname{Tor}_1 = 0$ |
| 忠实平坦 | 张量积不杀非零模，可回推性质 |
| Bourbaki 判别法 | 检查 $I \otimes M \to M$ 单射 |
| 纤维环 | $B \otimes_A \kappa(\mathfrak{p})$ |

## 5 小结

- **张量积** $M \otimes_A N$ 由双线性万有性质决定；$\mathbb{Z}/m \otimes \mathbb{Z}/n = \mathbb{Z}/\gcd(m,n)$、$A/I \otimes M = M/IM$。
- **$-\otimes N$ 只右正合**，左正合可能撕断（$\mathbb{Z} \xrightarrow{\cdot 2} \mathbb{Z}$ 反例），缺段由 $\operatorname{Tor}_1$ 记录。
- **平坦** = 张量积正合（Tor₁ = 0）；**忠实平坦** = 张量积不杀非零模、几乎可逆。
- 局部化、自由/投射模都平坦；平坦族的纤维维数连续。

在下一节，张量积成为整扩张的地基：**整元、整闭包、Going-up 与 Going-down**——「有限生成模块」与「环的扩张」如何互相翻译，维数又如何被扩张保持。
