---
title: 超弱拓扑与正常态
date: 2026-08-07
---

# 超弱拓扑与正常态

<div class="epigraph">
<p>数学家，就像画家或诗人一样，是模式的制造者。</p>
<footer>—— 戈弗雷 · 哈罗德 · 哈代（Godfrey Harold Hardy）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Kadison & Ringrose《Fundamentals of the Theory of Operator Algebras》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从超弱拓扑开始

第 21 篇的双交换子定理把「弱闭」立为 von Neumann 代数的定义，但「弱闭」里的「弱」到底有几种？$B(\mathcal{H})$ 上其实住着一整族拓扑——弱、强、$\ast$-强、超弱、超强，它们收敛的速度各不相同，却在「单位球上」全部一致。von Neumann 代数的分析性格，正是由这些拓扑中的**超弱拓扑**与它的对偶——**正常态**——决定的。

这一节要回答三个问题：$B(\mathcal{H})$ 上有哪些自然的算子拓扑、为什么超弱拓扑是「最正确」的一个（它有预对偶）、以及什么样的态是「好」的（正常态，保上确界）。答案将引出一个贯穿性事实：**von Neumann 代数的全部结构（第 23 篇类型、第 24 篇约化）都写在它的预对偶与正常态上**。

## 1 B(H) 上的算子拓扑家族

给定 $\mathcal{H}$，$B(\mathcal{H})$ 上定义五种拓扑：

| 拓扑 | 收敛 $T_\alpha\to T$ 的条件 | 直觉 |
| --- | --- | --- |
| 范数 | $\|T_\alpha-T\|\to0$ | 最强，逐点一致 |
| 强 SOT | $T_\alpha x\to Tx$ 对每个 $x$ | 逐点收敛 |
| 弱 WOT | $\langle T_\alpha x,y\rangle\to\langle Tx,y\rangle$ 对每个 $x,y$ | 配对收敛 |
| 超强 UST | $\|(T_\alpha-T)x\|\to0$，且 $T_\alpha^*x\to T^*x$ | 强 + 伴随 |
| 超弱 UWT | $\mathrm{Tr}(\rho T_\alpha)\to\mathrm{Tr}(\rho T)$ 对每个迹类 $\rho$ | 预对偶配对 |

**辨析｜易错点：**强拓扑**不**等价于「$\ast$-强」：SOT 收敛推不出 $T_\alpha^*\to T^*$（对合在 SOT 下不连续）。弱拓扑与超弱拓扑在单位球上一致，但整体上超弱更细。**乘法在 SOT 与 WOT 下都不连续**（$(A_\alpha,B_\alpha)\mapsto A_\alpha B_\alpha$ 整体不连续），却「在单侧固定时连续」——这个微妙性是 von Neumann 代数里大量论证的背景音。<span class="marginnote">「乘法不连续」看似是缺陷，实则是财富：von Neumann 代数的许多结构定理恰恰需要「当 $A_\alpha\to A$ 且 $B$ 固定时 $A_\alpha B\to AB$」这种「半连续」性质。第 23 篇因子理论里，这种半连续性是正常态与迹理论能展开的前提。</span>

## 2 预对偶：超弱拓扑的灵魂

第 6 篇已经埋下伏笔：$B(\mathcal{H})$ 的预对偶是迹类算子。

**定理（预对偶）**：映射 $\rho\mapsto\omega_\rho$（$\omega_\rho(T)=\mathrm{Tr}(\rho T)$）给出等距同构

$$B(\mathcal{H})_* \cong L^1(\mathcal{H}),$$

且 $B(\mathcal{H})$ 上的超弱拓扑恰是「$L^1(\mathcal{H})$ 作为对偶空间」诱导的 **弱-$\ast$ 拓扑**。于是：

- 超弱连续线性泛函 = 迹类算子给出的泛函（$\omega_\rho$ 形）；
- von Neumann 代数 $\mathcal{M}\subset B(\mathcal{H})$ 的**预对偶** $\mathcal{M}_*=L^1(\mathcal{H})/\mathcal{M}_\perp$ 使得 $\mathcal{M}\cong(\mathcal{M}_*)^*$。<span class="marginnote">「有预对偶」是 von Neumann 代数的抽象定义（$W^*$-代数的标志）：$\mathcal{M}=(\mathcal{M}_*)^*$。这意味着 von Neumann 代数天然是<strong>某个 Banach 空间的对偶</strong>，从而自动带有弱-$\ast$ 拓扑与弱-$\ast$ 紧单位球（Alaoglu）。预对偶是它比一般 C\* 代数「分析上更强」的根源。</span>

**推论（球紧）**：$\mathcal{M}$ 的单位球在超弱拓扑下紧（Alaoglu 定理），这使「在单位球内取极限」在 von Neumann 世界里永远可行——不存在「极限跑出球外」的意外。

## 3 正常态：保上确界的态

**正常态（normal state）**：$\mathcal{M}$ 上的态 $\varphi$，对任意递增网 $a_\alpha\nearrow a$（$a_\alpha\le a_\beta\le\cdots$，$a=\sup a_\alpha$）满足

$$\varphi(\sup_\alpha a_\alpha) = \sup_\alpha \varphi(a_\alpha).$$

正常态「与上确界交换」——它看见一切的「极限」。「不保上确界」的态叫**奇异态（singular state）**。

**定理（正常态的刻画）**：$\mathcal{M}$ 上的态 $\varphi$ 正常 ⟺ 存在正迹类算子 $\rho$（密度矩阵，$\mathrm{Tr}\,\rho=1$）使 $\varphi(T)=\mathrm{Tr}(\rho T)$ ⟺ $\varphi$ 在 $\mathcal{M}$ 的单位球上超弱连续。<span class="marginnote">量子力学的「密度矩阵」正是正常态：任意混合态 $\rho$ 给出的期望 $\mathrm{Tr}(\rho A)$ 是正常态。奇异态（如 $\ell^\infty$ 上沿自由超滤子的「极限」）存在但「看不见」——它们对应物理上不可实现的「无限远处」的测量，也解释了为何量子理论只认正常态。</span>

**辨析｜易错点：**正常态**不是**「超弱连续态」的同义反复——需要定理才能等价；「正常」是序性质（保 sup），「超弱连续」是拓扑性质，两者的等价是 von Neumann 代数特有的（一般 C\* 代数上不成立）。另一个易错点：正常态只能由**迹类**密度矩阵给出；一般 C\* 代数上的态（第 10 篇）大多不是正常态——**正常态是 von Neumann 世界特有的精贵品**。

## 4 公式解析：$\varphi(T)=\mathrm{Tr}(\rho T)$

$$
\varphi(T) = \mathrm{Tr}(\rho T), \qquad \rho\ge0,\ \mathrm{Tr}(\rho)=1
$$

- **第一步，看 $\rho$**：$\rho$ 是正迹类算子（第 6 篇），$\mathrm{Tr}\,\rho=1$ 归一化。它扮演「概率密度」：把 $\rho$ 想成 $\mathrm{diag}(\lambda_1,\lambda_2,\dots)$（$\lambda_i\ge0$，$\sum\lambda_i=1$）。
- **第二步，看配对**：$\mathrm{Tr}(\rho T)=\sum_i\langle Te_i,\rho e_i\rangle$ 对正交基展开。对 $\rho=|\xi\rangle\langle\xi|$（秩一投影，纯态），退化为 $\langle T\xi,\xi\rangle$——回到第 10 篇的向量态。
- **第三步，看为什么正常**：若 $a_\alpha\nearrow a$，则 $\mathrm{Tr}(\rho a_\alpha)\nearrow\mathrm{Tr}(\rho a)$ 由单调收敛定理（对可数情形）或超弱连续性（一般情形）给出——密度矩阵态自动保 sup。
- **第四步，看它如何区分正常与奇异**：正常态由迹类算子给出，奇异态则「逃出」$L^1$（只能靠非主超滤子等非构造对象）。物理上用正常态，数学上却必须理解奇异态为何存在——它们对应 $L^\infty$ 的「原子之外的尾巴」。

## 5 Kaplansky 密度定理与 von Neumann 代数的分析

**Kaplansky 密度定理**：设 $\mathcal{A}\subset B(\mathcal{H})$ 是含幺 $\ast$-子代数，$\mathcal{M}=\overline{\mathcal{A}}^{\,\mathrm{SOT}}$。则 $\mathcal{M}$ 的**单位球**包含于 $\mathcal{A}$ 的单位球在 SOT 下的闭包——且对自伴部分、正部分、酉部分分别成立。<span class="marginnote">Kaplansky 密度定理是「弱闭代数仍可被稠密子代数在<strong>球内</strong>逼近」的精确陈述。它保证：从稠密子代数出发，可以在不放大范数的情况下逼近任何 von Neumann 代数的元素——第 21 篇双交换子定理、以及此后一切「逼近式」构造（张量积、交叉积的 von Neumann 版本）都靠它。</span>

**应用（双交换子定理的完整证明）**：第 21 篇的 $T\in\mathcal{M}''\Rightarrow T\in\overline{\mathcal{M}}^{\mathrm{SOT}}$ 只是第一步；Kaplansky 密度定理把「强闭包」升级为「球内强逼近」，配合 $\mathcal{M}$ 含幺完成全部证明。拓扑与代数的缝合，Kaplansky 是那根针线。

**应用（von Neumann 代数的表示不依赖性）**：预对偶 $\mathcal{M}_*$ 与正常态空间 $(\mathcal{M}_*)^+_1$ 不依赖「$\mathcal{M}$ 具体住在哪个 $\mathcal{H}$ 上」——它们是**内蕴**的。于是第 23 篇的类型分类、第 24 篇的约化理论都可以脱离表示来讲，这正是 $W^*$-代数「抽象化」的底气。

**辨析｜易错点：**Kaplansky 定理对**含幺** $\ast$-子代数成立，且逼近在**球内**进行。若去掉含幺条件或要求在范数下逼近，结论崩塌。初学者常误以为「弱闭 = 范数闭包里加极限」，其实弱闭远大于范数闭——$B(\mathcal{H})$ 的弱闭包可包含范数闭包外的海量算子，单位球内的逼近正是唯一「温和」的通道。

## 6 例：拓扑的收敛差异

五种拓扑的差别，用一个收敛序列的例子就看清。

**强收敛但范数不收敛**：$P_n$（向 $\mathrm{span}\{e_1,\dots,e_n\}$ 的投影）。$P_n\to I$（强收敛：$\|P_nx-x\|\to0$），但 $\|P_n-I\|=1$ 不收敛到 0。强拓扑比范数拓扑「松」。

**弱收敛但不强收敛**：$e_n$（基向量，视为秩一投影 $|e_n\rangle\langle e_n|$）。$|e_n\rangle\langle e_n|\to0$（弱：$\langle e_n,x\rangle\langle e_n,y\rangle\to0$），但不强收敛（$\||e_n\rangle\langle e_n|e_1\|$ 不趋 0）。弱拓扑比强拓扑更「松」。

**$\ast$-强 vs 强**：$T_n$ 使 $T_nx\to Tx$ 但 $T_n^*x\not\to T^*x$——对合在 SOT 下不连续的例子。$\ast$-强拓扑「记住」伴随。

**超弱 vs 弱**：在单位球上一致，整体上超弱更细。$\mathrm{Tr}(\rho T_\alpha)\to\mathrm{Tr}(\rho T)$ 对所有迹类 $\rho$——比「逐对向量」的弱收敛要求更全面。

**乘法不连续**：$S_n\to S$、$T_n\to T$（强）不一定 $S_nT_n\to ST$——但 $S_nT\to ST$ 与 $ST_n\to ST$ 分别成立。「乘法半连续」是 von Neumann 论证的常态。

**一句话总结**：强/弱/超弱拓扑逐级变「松」，单位球上统一；「乘法半连续」是它们共同的性格。

## 7 延伸：正常态与物理

正常态不只是数学对象，它是物理态的精确刻画。

**密度矩阵 = 物理态**：正常态 $\varphi(T)=\mathrm{Tr}(\rho T)$ 正是量子力学的密度矩阵态。所有物理可实现的态都是正常态——奇异态物理上不可实现。

**为何「保 sup」**：正常态保上确界意味着「测量与极限交换」——对递增可观测序列 $A_\alpha\nearrow A$，期望值也递增到 $\varphi(A)$。这是「可观测量的极限仍然可观测」的保证。

**时间演化与正常态**：$\alpha_t$（哈密顿量生成的自同构）保持正常态（$e^{itH}\rho e^{-itH}$ 仍是密度矩阵）。正常态构成物理演化不变的态空间。

**KMS 态**：热力学平衡态 = KMS 态（满足 KMS 条件的正常态）。量子统计力学的平衡概念 = von Neumann 代数里的 KMS 态——正常态理论的深度应用。

**超选择再谈**：正常态对中心投影「兼容」：不同扇区的叠加态不是正常态（或说，正常态在扇区内）。超选择规则由「正常态」概念自动编码。

**一句话总结**：正常态 = 密度矩阵 = 物理态；「保 sup + 由迹类给出」让物理的「极限」概念在代数里有了精确形式。

## 8 延伸：预对偶的意义

$\mathcal{M}_*=(\mathcal{M}_*)^*$ 是 von Neumann 代数区别于一般 C\* 代数的「分析基因」。

**对偶与预对偶**：$\mathcal{M}$ 有对偶 $\mathcal{M}^*$（全体有界泛函）也有预对偶 $\mathcal{M}_*$（正常泛函）。预对偶是「恰到好处」的：$\mathcal{M}=(\mathcal{M}_*)^*$ 使 $\mathcal{M}$ 自动带弱-$\ast$ 拓扑。

**唯一性**：预对偶在等距同构下唯一——von Neumann 代数的拓扑不依赖「它住在哪个 $\mathcal{H}$」。这是 $W^*$-代数「抽象化」的基石（第 21 篇）。

**正常泛函的分类**：$\mathcal{M}_*$ 由迹类算子的「商」给出（$\mathcal{M}_*\cong L^1(\mathcal{H})/\mathcal{M}_\perp$）。正常态 = 预对偶的正锥范数 1 元素。

**为什么 $B(\mathcal{H})$ 有预对偶而一般 C\* 代数没有**：$B(\mathcal{H})=L^1(\mathcal{H})^*$，因为迹类算子 $L^1$ 的存在（第 6 篇）。预对偶是「迹」的化身——没有迹就没有 von Neumann 代数。

**在分类里**：$\mathcal{M}_*$ 的序结构与 $\mathcal{M}$ 的中心分解（第 24 篇）一一对应。预对偶是分类的「底层数据」。

**一句话总结**：预对偶让 von Neumann 代数「自带拓扑」——它是「迹类算子」的代数化身，也是 von Neumann 代数区别于 C\* 代数的分水岭。

## 9 小结

- **算子拓扑族**：范数、SOT、WOT、UST、UWT；SOT 下 $\ast$ 不连续，乘法只在单侧固定时连续。
- **预对偶** $B(\mathcal{H})_*\cong L^1(\mathcal{H})$：超弱拓扑 = 弱-$\ast$ 拓扑；von Neumann 代数有预对偶、单位球超弱紧。
- **正常态**：保上确界的态 ⟺ 密度矩阵态 $\mathrm{Tr}(\rho\,\cdot)$ ⟺ 单位球上超弱连续；奇异态存在但物理上不可实现。
- **$\varphi(T)=\mathrm{Tr}(\rho T)$