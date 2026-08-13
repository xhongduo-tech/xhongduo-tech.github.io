---
title: 二次量子化：产生湮灭算符与多粒子态
date: 2026-08-07
---

# 二次量子化：产生湮灭算符与多粒子态

<div class="epigraph">
<p>多体问题的困难不在于粒子多，而在于我们找不到一个简洁的语言来同时表达「哪个粒子在哪儿」与「粒子是不可分辨的」这两个事实。二次量子化就是为这个难题发明的语言。</p>
<footer>—— 文小刚（Xiao-Gang Wen）《量子多体理论》</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 1 ｜ 2026-08-07</p>
</div>

## 为什么从二次量子化开始

量子力学第一课教我们把单粒子波函数 $\psi(\mathbf{r})$ 当作基本对象，把 $|\psi|^2$ 当作概率密度。可一旦面对固体里的 $10^{23}$ 个电子，这条路立刻走不通：全同粒子的波函数必须满足对称性或反对称性，而任何「这个电子在 1 号位、那个电子在 2 号位」的写法都预设了电子可以被编号——这正是全同性原理所禁止的。<span class="marginnote"><strong>全同性原理</strong>：同类微观粒子不可分辨。交换两个粒子只会让波函数乘上相位因子 $e^{i\theta}$；玻色子取 $\theta=0$（对称），费米子取 $\theta=\pi$（反对称）。</span>

**二次量子化（second quantization）**：把「粒子数」也变成算符的量子化方案。它不再追问每个粒子的坐标，而是追问每个单粒子态被占据了几个粒子——用占据数 $n_\lambda$ 描述整个体系。这套语言以「产生」与「湮灭」算符为基本构件，天然内置了全同性原理，是多体理论一切后续工具（Green 函数、Feynman 图、平均场）的共同地基。学好这一篇，等于拿到了从量子力学通往整个多体物理的钥匙。

## 1 从一次量子化到占据数表象

一次量子化（first quantization）里，$N$ 个粒子的波函数是 $\Psi(\mathbf{r}_1,\dots,\mathbf{r}_N)$，玻色子要求它交换任意两坐标不变，费米子要求交换时变号。这个写法在 $N$ 小的时候还可用，$N$ 稍大就指数爆炸——更致命的是，它把「粒子编号」硬塞进了描述里，而编号在物理上是多余的。

**占据数表象（occupation-number representation）**换了个角度：设单粒子态为 $\{\varphi_\lambda\}$（$\lambda$ 是一组量子数的指标，例如动量 $\mathbf{k}$ 与自旋 $\sigma$），多体态不再记录「谁在哪儿」，而是记录**每个单粒子态被占了多少个粒子**：

$$|n_1, n_2, n_3, \dots\rangle$$

其中 $n_\lambda$ 是态 $\varphi_\lambda$ 上的占据数。玻色子允许 $n_\lambda = 0,1,2,\dots$ 任意非负整数；费米子受泡利不相容原理限制，只能取 $0$ 或 $1$。<span class="marginnote">泡利不相容原理在占据数表象里是「自动」的：费米子的反对易关系保证 $c_\lambda^\dagger c_\lambda^\dagger = 0$，即一个态上再放第二个粒子会恒等于零。</span>

所有可能的占据数组 $\{n_\lambda\}$ 张成的空间称为**Fock 空间（Fock space）**。注意它与单粒子 Hilbert 空间不同：Fock 空间是粒子数可变的全体多体态之和。

## 2 产生与湮灭算符

要在占据数表象里做代数，需要两个基本算符：

**湮灭算符** $a_\lambda$：把态 $\varphi_\lambda$ 上的一个粒子拿走，$n_\lambda \to n_\lambda - 1$。
**产生算符** $a_\lambda^\dagger$：把一个粒子放进态 $\varphi_\lambda$，$n_\lambda \to n_\lambda + 1$。

真空态 $|0\rangle$ 定义为所有 $n_\lambda = 0$ 的态，任何态都可以由真空连续作用产生算符得到：

$$|n_1, n_2, \dots\rangle = \prod_\lambda \frac{(a_\lambda^\dagger)^{n_\lambda}}{\sqrt{n_\lambda!}}\,|0\rangle$$

其中 $\sqrt{n_\lambda!}$ 是玻色子的归一化因子。粒子数的测量由**数算符（number operator）**完成：

$$\hat{n}_\lambda = a_\lambda^\dagger a_\lambda$$

它的本征值就是占据数 $n_\lambda$。<span class="marginnote">数算符的出现揭示了「二次量子化」名字的由来：位置与动量在普通量子力学里已被量子化一次，现在连「粒子数」也被量子化成算符——这是第二次量子化，故名二次量子化。</span>

**辨析｜易错点：** 初学者常把 $a^\dagger a$ 与 $a a^\dagger$ 当成一回事。在玻色子情形两者相差单位算符：$[a_\lambda, a_\lambda^\dagger] = 1$，所以 $a_\lambda a_\lambda^\dagger = a_\lambda^\dagger a_\lambda + 1$。把真空的 $n_\lambda = 0$ 代入可验证：$a_\lambda a_\lambda^\dagger|0\rangle = |0\rangle$ 而 $a_\lambda^\dagger a_\lambda |0\rangle = 0$——顺序反了，物理完全不同。

## 3 对易关系：玻色子与费米子的分水岭

产生与湮灭算符之间的代数关系，完全决定了粒子的统计性质：

**玻色子**满足**对易关系（commutation relations）**：

$$[a_\lambda, a_{\mu}^\dagger] = \delta_{\lambda\mu}, \qquad [a_\lambda, a_\mu] = [a_\lambda^\dagger, a_\mu^\dagger] = 0$$

**费米子**满足**反对易关系（anticommutation relations）**：

$$\{c_\lambda, c_\mu^\dagger\} = \delta_{\lambda\mu}, \qquad \{c_\lambda, c_\mu\} = \{c_\lambda^\dagger, c_\mu^\dagger\} = 0$$

其中 $\{A,B\} = AB + BA$。一条小小的符号差异，带来巨大的物理后果：反对易关系自动给出泡利不相容（$c_\lambda^\dagger c_\lambda^\dagger = 0$），也保证费米子波函数交换变号。正是这套代数，让二次量子化**免去了每次手工对称化/反对称化的麻烦**——统计性质已经内建在算符代数里。

**重点：产生算符作用于一个已满的费米子态给出零，作用于空态给出费米子；湮灭算符反之。** 这是费米子系统所有计算的出发点，也是后面 Wick 定理与 Feynman 图规则里「费米子圈带来负号」的根源。

## 4 场算符：把「态指标」换成「空间点」

占据数表象用离散指标 $\lambda$ 标记单粒子态。但很多问题里我们关心的是空间上的粒子密度分布，于是引入**场算符（field operator）**：

$$\hat{\psi}(\mathbf{r}) = \sum_\lambda \varphi_\lambda(\mathbf{r})\, a_\lambda, \qquad \hat{\psi}^\dagger(\mathbf{r}) = \sum_\lambda \varphi_\lambda^*(\mathbf{r})\, a_\lambda^\dagger$$

场算符的意义很漂亮：$\hat{\psi}^\dagger(\mathbf{r})$ 在 $\mathbf{r}$ 处产生一个粒子，$\hat{\psi}(\mathbf{r})$ 在 $\mathbf{r}$ 处湮灭一个粒子。它们不再是单粒子波函数，而是**算符**——这正是「二次量子化」里「第二次数子化」的直观：把原来地位显赫的波函数 $\psi(\mathbf{r})$ 变成了算符 $\hat{\psi}(\mathbf{r})$。粒子密度算符与总粒子数算符可写成：

$$\hat{n}(\mathbf{r}) = \hat{\psi}^\dagger(\mathbf{r})\hat{\psi}(\mathbf{r}), \qquad \hat{N} = \int d\mathbf{r}\, \hat{\psi}^\dagger(\mathbf{r})\hat{\psi}(\mathbf{r})$$

<span class="marginnote">场算符的引入把「粒子数守恒」表述为 $\hat{N}$ 与哈密顿量对易。后续许多理论（如 Bogoliubov 变换、BCS 超导）会处理粒子数不守恒的情形，那时正是场算符语言大显身手的地方。</span>

## 5 公式解析：哈密顿量的二次量子化形式

二次量子化的最大价值在于把相互作用哈密顿量翻译成算符语言。以含两体相互作用的电子气为例：

$$
\hat{H} = \sum_{\mathbf{k}\sigma} \varepsilon_{\mathbf{k}}\, c_{\mathbf{k}\sigma}^\dagger c_{\mathbf{k}\sigma} + \frac{1}{2}\sum_{\mathbf{k}\mathbf{k}'\mathbf{q}\atop \sigma\sigma'} V_{\mathbf{q}}\, c_{\mathbf{k}+\mathbf{q},\sigma}^\dagger c_{\mathbf{k}'-\mathbf{q},\sigma'}^\dagger c_{\mathbf{k}',\sigma'} c_{\mathbf{k},\sigma}
$$

这条式子是多体物理的「见面礼」，拆解如下：

- **第一项，单粒子（动能）部分**：$\varepsilon_{\mathbf{k}}$ 是动量为 $\mathbf{k}$ 的单粒子能量（对自由电子气即 $\hbar^2 k^2/2m$）。$c_{\mathbf{k}\sigma}^\dagger c_{\mathbf{k}\sigma}$ 就是数算符 $\hat{n}_{\mathbf{k}\sigma}$，统计在态 $(\mathbf{k},\sigma)$ 上的粒子数。整项把「每个粒子的动能」翻译成「按态累加的能量」。
- **第二项，两体相互作用**：$V_{\mathbf{q}}$ 是传递动量 $\mathbf{q}$ 的相互作用强度（如库仑势的 Fourier 分量 $4\pi e^2/q^2$）。算符序列从右往左读：先在 $\mathbf{k}$ 湮灭一个 $\sigma$ 自旋电子，在 $\mathbf{k}'$ 湮灭一个 $\sigma'$ 电子，再在 $\mathbf{k}'-\mathbf{q}$ 产生一个 $\sigma'$ 电子、在 $\mathbf{k}+\mathbf{q}$ 产生一个 $\sigma$ 电子——即两个电子交换动量 $\mathbf{q}$ 后散射到新态。
- **系数 $1/2$**：两体相互作用的配对（$\mathbf{k},\mathbf{k}'$）交换一次会给出同一个物理过程，除以 2 避免重复计数。
- **费米子算符顺序**：产生算符都放在湮灭算符左边（所谓「正规序」），这保证真空期望值为零，也为后面的 Wick 收缩铺平道路。

**重点：二次量子化哈密顿量不是一个新物理模型，而是同一物理的另一种数学表达。** 它没有改变单粒子量子力学的任何内容，只是把「全同粒子 + 对称化」这个约束，从波函数层面搬到了算符代数层面——而这正是让多体计算得以规模化的关键一跃。

## 6 二次量子化与经典极限的桥梁

二次量子化并非纯形式游戏，它直接通向宏观物理。经典场（如晶格振动里的原子位移场）可以被量子化为声子场，方法就是这一篇的场算符：位移 $u(\mathbf{r}) = \sum_{\mathbf{k}} \sqrt{\hbar/2NM\omega_{\mathbf{k}}}(a_{\mathbf{k}} + a_{-\mathbf{k}}^\dagger) e^{i\mathbf{k}\cdot\mathbf{r}}$。这里 $a_{\mathbf{k}}^\dagger$ 产生一个动量为 $\mathbf{k}$、能量 $\hbar\omega_{\mathbf{k}}$ 的声子。<span class="marginnote">这正是文小刚《量子多体理论》开篇的著名视角：<strong>声子是振动场的量子</strong>。晶格振动的集体模式被「第二次量子化」后，就变成了一堆可数、可产生的粒子——从连续场到粒子的这条道路，是全书「从声子的起源到光子、电子的起源」的主线。</span>

这也是本篇与「从极限到大模型」主线的连接点：宏观世界的极限（经典场论、流体、弹性波）在微观尺度上呈现为量子化的准粒子；而反过来，大模型里「把连续表示离散成 token 序列」的做法，在精神上与「把连续场离散成占据数」同构——把无限维的连续对象，编码成一组离散的、可增减的单元。

## 7 小结

- **二次量子化**以占据数表象替代坐标表象，用 Fock 空间描述粒子数可变的多体体系，全同性原理内建在算符代数中。
- 核心构件是**产生算符 $a^\dagger$ 与湮灭算符 $a$**，数算符 $\hat{n}=a^\dagger a$ 测量占据数。
- **玻色子对易、费米子反对易**，这一符号差异自动实现泡利不相容与交换符号，是统计性质的源头。
- **场算符** $\hat{\psi}(\mathbf{r})=\sum_\lambda \varphi_\lambda(\mathbf{r})a_\lambda$ 把态指标换成空间点，是连接粒子语言与场语言的桥梁。
- 二次量子化的**哈密顿量**把单粒子动能与两体相互作用翻译成算符代数，系数 $1/2$ 与正规序是易错点。
- 声子、光子等准粒子都是经典场二次量子化的产物，这构成了多体理论统一描述的起点。

在下一节，我们将进入本专题最核心的工具之一：**单粒子 Green 函数**——它把「粒子如何从一点传播到另一点」编码成一个可计算的函数，并成为后续所有 Feynman 图展开的舞台。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| 占据数态 | $|n_1,n_2,\dots\rangle$ | Fock 空间基 |
| 玻色对易 | $[a_\lambda,a_\mu^\dagger]=\delta_{\lambda\mu}$ | 对称统计 |
| 费米反对易 | $\{c_\lambda,c_\mu^\dagger\}=\delta_{\lambda\mu}$ | 反对称统计，泡利原理内建 |
| 场算符 | $\hat{\psi}(\mathbf{r}) = \sum_\lambda\varphi_\lambda(\mathbf{r})a_\lambda$ | 波函数变算符 |
| 数算符 | $\hat{n}_\lambda = a_\lambda^\dagger a_\lambda$ | 测量占据数 |
| 两体哈密顿量 | $\frac{1}{2}\sum V_\mathbf{q}c^\dagger c^\dagger cc$ | 产生在左、湮灭在右，系数 $1/2$ |

**易错复盘**：三点要盯住。其一，$a^\dagger a$ 与 $aa^\dagger$ 不同——玻色子相差单位算符，费米子相差 $(1-2n)$；其二，费米子产生算符作用在已满态给出零（泡利），作用在空态给出费米子——这个规则是后续一切负号与 Wick 定理的根源；其三，二次量子化不改变物理，只改变数学——它把「全同粒子 + 对称化」从波函数层搬到算符代数层。

**知识连线**：二次量子化是全专题的「语言基础」——场算符进入 Green 函数（第 1 篇）、声子与极化子（第 2 篇）、Bogoliubov 与 BCS（第 3 篇）、Hubbard 与磁性（第 4 篇）全部用它。「把连续场离散成可增减的单元」与「把连续表示离散成 token 序列」同构——本篇与「从极限到大模型」主线的最早连接。

**实践与辨析**：为什么系数 $1/2$ 出现在两体相互作用前？提示：配对 $(\mathbf{k},\mathbf{k}')$ 交换一次给出同一物理过程，除以 2 避免重复计数。为什么费米子产生算符都放左边（正规序）？提示：保证真空期望值为零，为 Wick 收缩铺路。易错提醒：场算符是算符不是波函数——$\hat{\psi}(\mathbf{r})$ 在 $\mathbf{r}$