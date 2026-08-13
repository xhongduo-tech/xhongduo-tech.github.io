---
title: Hilbert-Schmidt 算子与迹类算子
date: 2026-08-07
---

# Hilbert-Schmidt 算子与迹类算子

<div class="epigraph">
<p>数学在自然科学中那种不可思议的有效性，是近乎神秘的天赐礼物。</p>
<footer>—— 尤金 · 维格纳（Eugene Wigner）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Murphy《C\*-Algebras and Operator Theory》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Hilbert-Schmidt 与迹类算子开始

上一节的紧算子有一个缺憾：**算子范数太粗糙**。两个紧算子可以范数上无限接近，内部结构却天差地别。我们需要更精细的标尺，把「有多大」度量到分量级别。Hilbert–Schmidt 范数与迹范数正是这样的尺子——它们把算子当成「无穷矩阵」逐项量大小，就像用 Frobenius 范数度量矩阵、用 $\ell^1$ 范数度量数列。

这两类算子同时是量子力学的天然语言：物理态的期望值都写成迹 $\mathrm{Tr}(\rho A)$，可观测量的平均就是「迹类算子 $\rho$ 与有界算子 $A$ 的配对」。更深刻的是，**迹类算子是 $B(\mathcal{H})$ 的预对偶**——理解迹类算子，就是为第 22 篇 von Neumann 代数的超弱拓扑与正常态铺路。这篇是「算子准备篇」的收官之作。

## 1 用正交基给算子「称重」

设 $\{e_n\}$ 是 $\mathcal{H}$ 的一组标准正交基，$T\in B(\mathcal{H})$。把 $T$ 想成无穷矩阵 $(t_{mn})$，$t_{mn}=\langle Te_n,e_m\rangle$。

**Hilbert–Schmidt 范数（Hilbert–Schmidt norm）**：

$$\|T\|_2 = \left(\sum_{n}\|Te_n\|^2\right)^{1/2} = \left(\sum_{m,n}|t_{mn}|^2\right)^{1/2}.$$

**辨析｜易错点：**$\|T\|_2$ 可能为 $\infty$（例如恒等算子在无穷维上 $\sum\|e_n\|^2=\infty$）。所以「Hilbert–Schmidt 算子」不是所有算子，而是**满足 $\|T\|_2\lt \infty$ 的那一批**。另一个易错点：$\|T\|_2$ 的定义**不依赖正交基的选择**——换一组基，$t_{mn}$ 全体平方和不变（Parseval 恒等式的算子版本）。<span class="marginnote">基无关性使 $\|T\|_2$ 成为真正内蕴的量。验证它需要算两次 Parseval：$\sum_n\|Te_n\|^2=\sum_{m,n}|\langle Te_n,e_m\rangle|^2$，左边与基无关是因为它是 $\|T\|_2$ 定义，右边是矩阵元素的平方和，二者互为翻译。</span>

**范数关系**：$\|T\|\le\|T\|_2$，且 $\|T\|_2\le\|T\|_1$（迹范数，见下节）。三把尺子由粗到细：算子范数 ≤ Hilbert–Schmidt 范数 ≤ 迹范数。

## 2 Hilbert–Schmidt 算子：算子里的 $L^2$

记 $L^2(\mathcal{H})=\{T\in B(\mathcal{H}):\|T\|_2\lt \infty\}$。

**定理**：$L^2(\mathcal{H})$ 在 Hilbert–Schmidt 范数下是 **Hilbert 空间**，内积为 $\langle T,S\rangle_2=\mathrm{Tr}(S^*T)$；且它是 $B(\mathcal{H})$ 中的**双边理想**（乘任意有界算子保持 $\|T\|_2\lt \infty$），并嵌入 $\mathcal{K}(\mathcal{H})$。<span class="marginnote">「算子里的 $L^2$」这个类比非常到位：就像 $L^2$ 函数是「平方可积」的函数，Hilbert–Schmidt 算子是「平方可积」的算子；也像 $L^2$ 有内积、$L^2\subset$ 局部可积，Hilbert–Schmidt 算子有内积、$L^2(\mathcal{H})\subset\mathcal{K}(\mathcal{H})$。</span>

**关键例（积分算子）**：设 $k\in L^2(X\times Y,\mu\times\nu)$，则 $K:L^2(Y)\to L^2(X)$，$(Kf)(x)=\int_Y k(x,y)f(y)\,d\nu(y)$ 是 Hilbert–Schmidt 的，且 $\|K\|_2=\|k\|_{L^2}$。反之，每个 Hilbert–Schmidt 算子都来自某个 $L^2$ 核——**Hilbert–Schmidt 算子与平方可积核一一对应**。这使「核方法」在偏微分方程、量子力学与机器学习核技巧中畅通无阻。

## 3 迹类算子：算子的绝对值可求和

对算子 $T$，设 $|T|=(T^*T)^{1/2}$（第 4 篇的绝对值）。定义奇异值 $s_n(T)$ 为 $|T|$ 的特征值（按重数排列）。

**迹类算子（trace-class operator）**：$\|T\|_1=\sum_n s_n(T)\lt \infty$，即 $|T|\in L^1(\mathcal{H})$。其**迹（trace）**定义为

$$\mathrm{Tr}(T) = \sum_n \langle Te_n, e_n\rangle,$$

其中 $\{e_n\}$ 是任意标准正交基；级数绝对收敛且与基无关。<span class="marginnote">定义中用 $|T|$ 而非 $T$ 本身，是为了保证迹的良定义：$\mathrm{Tr}(T)$ 的级数可能条件收敛甚至发散，但 $\sum\langle|T|e_n,e_n\rangle\lt \infty$ 时，$\sum\langle Te_n,e_n\rangle$ 绝对收敛且基无关。数学家引入 $|T|$，本质上是给「取模」这件事在算子世界找代言人。</span>

**例子**：对角线算子 $\mathrm{diag}(\lambda_1,\lambda_2,\dots)$ 是迹类的当且仅当 $\sum|\lambda_n|\lt \infty$，此时 $\mathrm{Tr}=\sum\lambda_n$。Hilbert–Schmidt 但不迹类的例子：$\mathrm{diag}(1,1/2,1/3,\dots)$（$\sum s_n^2\lt ∞$ 但 $\sum s_n=∞$）。

## 4 公式解析：三个范数的递进

$$
\|T\| \ \le\ \|T\|_2 = \Big(\sum_n \|Te_n\|^2\Big)^{1/2}\ \le\ \|T\|_1 = \sum_n s_n(T)
$$

- **第一步，看最右的 $\|T\|_1$**：奇异值 $s_n(T)$ 是 $|T|$ 的特征值，$\sum s_n$ 是「把 $T$ 的伸缩程度逐维加起来」。这对应矩阵的 $\ell^1$ 直觉：迹范数对「所有方向的伸缩」一视同仁地求和，所以它是三把尺中最细的。
- **第二步，看中间的 $\|T\|_2$**：$s_n(T)^2$ 加起来再开方——这是奇异值的 $\ell^2$ 范数。因为 $s_n\ge0$ 且平方和 ≤ 和的平方，$\|T\|_2\le\|T\|_1$ 恒成立。
- **第三步，看最左的 $\|T\|$**：算子范数只取「最大的那个方向」：$\|T\|=s_1(T)=\max s_n(T)$。于是一个方向的伸缩就决定了算子范数，而 Hilbert–Schmidt 与迹范数则要求**所有方向**都好。这三条不等式把「粗-中-细」的度量层次钉死。
- **第四步，一条恒等式**：$\|T\|_2^2=\mathrm{Tr}(T^*T)$，把 Hilbert–Schmidt 范数翻译成内积的迹——这解释了为什么 $L^2(\mathcal{H})$ 是 Hilbert 空间，也预告了迹在量子信息里的角色（$\mathrm{Tr}(\rho^2)\le1$ 刻画纯态）。

**迹的两个关键性质**：
- **正性**：$T\ge0 \Rightarrow \mathrm{Tr}(T)\ge0$，且 $\mathrm{Tr}(T)=0\Rightarrow T=0$。
- **循环性**：$A\in B(\mathcal{H}),\,B\in L^1(\mathcal{H})$ 时 $\mathrm{Tr}(AB)=\mathrm{Tr}(BA)$。<span class="marginnote">循环性 $Tr(AB)=Tr(BA)$ 是有限维「迹的循环不变性」的直接遗产，量子力学中把它当作 $\mathrm{Tr}(\rho A)=\mathrm{Tr}(A\rho)$ 来用。注意它只在「一方迹类、另一方有界」时无条件成立；两个一般有界算子的循环性在无穷维不成立（$AB$ 可能连迹类都不是）。</span>

## 5 迹类算子与对偶：通往 von Neumann 代数

迹类算子的最深价值藏在对偶理论里。设 $L^1(\mathcal{H})$ 为全体迹类算子。

**定理**：映射 $\varphi\mapsto \mathrm{Tr}(\varphi\,\cdot)$ 给出等距同构

$$L^1(\mathcal{H})^* \cong B(\mathcal{H}), \qquad B(\mathcal{H})_* \cong L^1(\mathcal{H}).$$

即：**$B(\mathcal{H})$ 的预对偶恰是迹类算子**。每个有界线性泛函 $\omega$ 对应唯一的迹类算子 $\varphi_\omega$，使 $\omega(T)=\mathrm{Tr}(\varphi_\omega T)$，且 $\|\omega\|=\|\varphi_\omega\|_1$。<span class="marginnote">这句话是理解 von Neumann 代数一切拓扑的钥匙：$B(\mathcal{H})$ 上「由迹类算子诱导的弱-$\ast$ 拓扑」（即超弱拓扑）之所以好，正因为它的对偶空间就是迹类算子——一个可分、具体、能计算的对象。第 22 篇的超弱拓扑与正常态，全部从这里长出来。</span>

**应用（量子态）**：密度矩阵 $\rho$ 是正迹类算子且 $\mathrm{Tr}(\rho)=1$；物理可观测量 $A\in B(\mathcal{H})$ 的期望值是 $\langle A\rangle_\rho=\mathrm{Tr}(\rho A)$。迹类算子的正性、循环性与预对偶地位，使量子力学的全部概率结构得以在算子上展开——这是维格纳「不可思议的有效性」的一个最具体的注脚。

**辨析｜易错点：**迹类算子集 $L^1(\mathcal{H})$ 在迹范数下完备（是 Banach 空间），但在算子范数下**不闭**——它在 $\mathcal{K}(\mathcal{H})$ 中稠密。所以「取迹」不能随意扩展到紧算子：$\mathrm{diag}(1,1/2,1/3,\dots)$ 是紧的、Hilbert–Schmidt 的，却无迹。看到「迹」字，先问一句：迹类吗？

## 6 例：三种范数的数值体验

用同一个对角算子，把三种范数的差别「算」出来。

**对角算子 $\mathrm{diag}(\lambda_n)$**：$\|T\|=\sup|\lambda_n|$，$\|T\|_2=(\sum|\lambda_n|^2)^{1/2}$，$\|T\|_1=\sum|\lambda_n|$。

**HS 但不迹类**：$\lambda_n=1/n$。$\sum|\lambda_n|^2=\pi^2/6\lt \infty$ 而 $\sum|\lambda_n|=\infty$——所以 $\mathrm{diag}(1,1/2,1/3,\dots)$ 是 Hilbert–Schmidt 的、紧的，却不是迹类的。它精确展示了「$\|T\|_2\le\|T\|_1$ 的反例」。

**迹类但不…**：$\lambda_n=1/n^2$。$\sum 1/n^2\lt \infty$，是迹类，$\mathrm{Tr}=\sum 1/n^2=\pi^2/6$。这个例子也说明：迹可以是「收敛级数的和」，不必是有限和。

**有限维直觉**：在 $M_n$ 里三种范数等价（都控制矩阵的大小），但界依赖 $n$。无穷维里它们彻底分开——这正是「无穷维需要更细的标尺」的活证据。

**积分算子对照**：$L^2$ 核 $k$ 给 HS 算子 $\|K\|_2=\|k\|_2$；若核「好」（如连续）则通常可迹。核方法（机器学习）里，「核的好坏」常由「它属于哪个算子类」判定。

**一句话总结**：三种范数 = 奇异值的三种统计量（最大、$\ell^2$、$\ell^1$），用对角算子一算，层次分明。

## 7 延伸：迹类算子的物理角色

迹类算子不是分析的边角料，它是量子力学的「状态空间」。

**密度矩阵**：物理态 = 正迹类算子 $\rho$，$\mathrm{Tr}\,\rho=1$。期望值 $\langle A\rangle_\rho=\mathrm{Tr}(\rho A)$。量子信息的全部计算都在这套框架里。

**纯态与混合态**：$\rho=|\xi\rangle\langle\xi|$（秩一投影）是纯态，$\mathrm{Tr}(\rho^2)=1$；一般混合态 $\mathrm{Tr}(\rho^2)\lt 1$。$\mathrm{Tr}(\rho^2)$ 是「纯度」的标尺。

**纠缠的判定**：两体系统 $\rho\in L^1(\mathcal{H}_1\otimes\mathcal{H}_2)$；$\rho$ 可分离当且仅当 $\rho=\sum p_k\rho_k^{(1)}\otimes\rho_k^{(2)}$。张量积（第 19 篇）+ 迹类算子（本篇）是纠缠的数学容器。

**熵**：von Neumann 熵 $S(\rho)=-\mathrm{Tr}(\rho\ln\rho)$。谱分解 $\rho=\sum\lambda_k|\xi_k\rangle\langle\xi_k|$ 后，$S(\rho)=-\sum\lambda_k\ln\lambda_k$——与 Shannon 熵同构。

**预对偶预告**：$B(\mathcal{H})_*=L^1(\mathcal{H})$ 意味着「物理态 = $B(\mathcal{H})$ 上的正常线性泛函」（第 22 篇）。迹类算子是 von Neumann 代数世界的第一公民。

## 8 延伸：从迹类算子到 von Neumann 预对偶

预对偶 $B(\mathcal{H})_*\cong L^1(\mathcal{H})$ 是整座 von Neumann 代数理论的支点，值得多走一步。

**弱-$\ast$ 拓扑**：$B(\mathcal{H})$ 上由 $L^1(\mathcal{H})$ 诱导的弱-$\ast$ 拓扑，正是第 22 篇的超弱拓扑。$T_\alpha\to T$（超弱）⟺ $\mathrm{Tr}(\rho T_\alpha)\to\mathrm{Tr}(\rho T)$ 对所有迹类 $\rho$。

**单位球紧**：Alaoglu 定理给 von Neumann 代数的单位球「弱-$\ast$ 紧」。紧性是「球内取极限」的通行证——分析里最常用的工具。

**正常态 = 密度矩阵**：$(\mathcal{M}_*)^+_1$（正迹类算子范数 1）正是正常态空间（第 22 篇）。正常态保上确界、由密度矩阵给出，物理与数学在此完全吻合。

**为什么预对偶优于对偶**：$B(\mathcal{H})^*$（全体有界泛函）太大、太野；$B(\mathcal{H})_*$（正常泛函）恰到好处。von Neumann 代数选择预对偶，等于选择「只听见物理上可实现的那部分线性泛函」。

**一句话总结**：迹类算子不只是「一类算子」——它是 von Neumann 代数的对偶空间，是「正常」与「物理」的定义。

## 9 小结

- **Hilbert–Schmidt 范数** $\|T\|_2=(\sum\|Te_n\|^2)^{1/2}$ 与基无关，$L^2(\mathcal{H})$ 是 Hilbert 空间、双边理想、且 $\subset\mathcal{K}(\mathcal{H})$。
- **迹类算子** $\|T\|_1=\sum s_n(T)\lt \infty$，迹 $\mathrm{Tr}(T)=\sum\langle Te_n,e_n\rangle$ 绝对收敛、基无关。
- **范数递进** $\|T\|\le\|T\|_2\le\|T\|_1$，对应「最大方向 / 平方和 / 全和」三重视角。
- **迹的正性与循环性**：$\mathrm{Tr}(AB)=\mathrm{Tr}(BA)$ 在「一方迹类」时成立。
- **预对偶** $B(\mathcal{H})_*\cong L^1(\mathcal{H})$：迹类算子是 $B(\mathcal{H})$