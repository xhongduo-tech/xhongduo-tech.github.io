---
title: 万有系数定理：同调与上同调的桥梁
date: 2026-08-07
---

# 万有系数定理：同调与上同调的桥梁

<div class="epigraph">
<p>同调群本身已经知道关于它的一切——只要你会用 Tor 和 Ext 去问。</p>
<footer>—— 诺曼 · 斯廷罗德（Norman Steenrod）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第3.1章；Munkres 第10、11章 ｜ 2026-08-07</p>
</div>

## 为什么从万有系数定理开始

上一篇的 Künneth 公式里，$\operatorname{Tor}$ 第一次登场。它来自一个更基础的问题：**换系数。** 我们一路算的都是整数系数
$H_n(X) = H_n(X;\mathbb{Z})$，但 Eilenberg–Steenrod 公理篇提到，系数群 $G$
可以任意选：$\mathbb{Z}$、$\mathbb{Z}_p$、$\mathbb{Q}$……那么 $H_n(X; G)$ 到底是什么？

直觉上，$H_n(X;\mathbb{Z})$ 已经包含了「空间的洞」的全部信息，换成 $G$ 系数只是「换一种计数」。**万有系数定理（universal
coefficient theorem）**把这个直觉精确化：$H_n(X;G)$ 完全由 $H_n(X;\mathbb{Z})$ 与 $G$
决定，公式是一条短正合序列，修正项正是 $\operatorname{Tor}$。它对上同调也有一个对偶版本，用
**$\operatorname{Ext}$** 函子：$H^n(X;G)$ 完全由 $H_{n-1}(X)$、$H_n(X)$ 与 $G$ 决定。

**为什么「万有」**：因为定理对**一切**系数群 $G$ 同时成立，是「换系数」这一操作的总说明书。它同时是通往第 4
篇「上同调」的正式入口——上同调不是「同调的镜像」，而是「同调的某种函数」（$\operatorname{Hom}$）的结果，二者由万有系数定理精确挂钩。<span class="marginnote">从「洞怎么数」到「洞上的函数怎么数」：同调记录「有哪些循环」，上同调记录「循环上的标量值」。万有系数定理说，给定整系数同调，这两个问题完全由代数决定——这是代数拓扑里「几何信息会沉淀为代数信息」的又一次证明。</span>


换系数这件事，乍看只是「换个数法」，为什么值得一整条定理？因为**系数选择直接决定你看到多少信息**。$\mathbb{Z}$
系数能看到挠，$\mathbb{Q}$ 系数只看得到秩，$\mathbb{Z}_p$ 系数专捕
$p$-挠——同一个空间，不同系数下呈现不同侧影。而万有系数定理保证这一切不是随意的：**给定整系数同调，任意系数的同调都能「算」出来**，不需要重算链复形。它是「换系数」操作的总说明书，也是通往第
4 篇上同调的正式入口。

## 1 准备：张量积、Hom、Tor 与 Ext

四个函子，各自扮演一个角色。设 $A, B$ 是阿贝尔群。

- **张量积 $A \otimes B$**：由双线性对 $(a,b)$ 生成的群，把「组合」记录成「配对」。$\mathbb{Z} \otimes B \cong B$，$\mathbb{Z}_m \otimes \mathbb{Z}_n = \mathbb{Z}_{\gcd(m,n)}$。
- **Hom 群 $\operatorname{Hom}(A, B)$**：$A \to B$ 的群同态全体，记录「$A$ 的每一个生成元要送到 $B$ 的哪个元素」。$\operatorname{Hom}(\mathbb{Z}, B) \cong B$，$\operatorname{Hom}(\mathbb{Z}_m, \mathbb{Z}_n) \cong \mathbb{Z}_{\gcd(m,n)}$。
- **$\operatorname{Tor}(A,B)$**：张量积的导出函子，度量「张量积丢掉的信息」。$\operatorname{Tor}(\mathbb{Z}_m,\mathbb{Z}_n) = \mathbb{Z}_{\gcd(m,n)}$。
- **$\operatorname{Ext}(A,B)$**：Hom 的导出函子，度量「扩张的信息」。$\operatorname{Ext}(\mathbb{Z}_m,\mathbb{Z}_n) = \mathbb{Z}_{\gcd(m,n)}$，$\operatorname{Ext}(\mathbb{Z}, B) = 0$。

**辨析｜易错点：** Tor 与 Ext 的输入输出容易记混。一句口诀：**Tor 管「组合」的误差，Ext
管「映射」的误差**——同调版本（用张量积）配 Tor，上同调版本（用 Hom）配 Ext。别把 $\operatorname{Tor}$
安到上同调公式、把 $\operatorname{Ext}$ 安到同调公式上。

## 2 同调万有系数定理

**定理（同调万有系数定理）。** 对任意空间 $X$ 与阿贝尔群 $G$，存在分裂的短正合序列

$$0 \longrightarrow H_n(X) \otimes G \longrightarrow H_n(X; G) \longrightarrow \operatorname{Tor}\big(H_{n-1}(X),\ G\big) \longrightarrow 0$$

其中 $H_n(X) = H_n(X;\mathbb{Z})$。**分裂**，故同构（不典范）：

$$H_n(X; G) \cong \big(H_n(X) \otimes G\big) \oplus \operatorname{Tor}\big(H_{n-1}(X),\ G\big)$$

**三个典型读数**：

- $G = \mathbb{Q}$：$H_n(X;\mathbb{Q}) = H_n(X) \otimes \mathbb{Q}$——**把挠全部磨平，只留秩**。这就是为什么「有理同调」只关心每个维度的 Betti 数。
- $G = \mathbb{Z}_p$：$H_n(X;\mathbb{Z}_p)$ 同时看见秩与 $p$-挠，是模 $p$ 系数的常用版本。
- $H_{n-1}(X)$ 无挠时：Tor 项消失，$H_n(X;G) = H_n(X) \otimes G$，换系数就是「张量上去」。

**例**：$\mathbb{RP}^2$，$H_1 = \mathbb{Z}_2$。$H_1(\mathbb{RP}^2;\mathbb{Z}_2) =
(\mathbb{Z}_2 \otimes \mathbb{Z}_2) \oplus \operatorname{Tor}(H_0,
\mathbb{Z}_2) = \mathbb{Z}_2 \oplus
\operatorname{Tor}(\mathbb{Z},\mathbb{Z}_2) = \mathbb{Z}_2$。<span class="marginnote">这里 Tor 项恰好为零；但 $H_2(\mathbb{RP}^2;\mathbb{Z}_2)$
就非平凡：$H_2(\mathbb{RP}^2;\mathbb{Z}_2) = (H_2 \otimes \mathbb{Z}_2) \oplus
\operatorname{Tor}(H_1, \mathbb{Z}_2) = 0 \oplus
\operatorname{Tor}(\mathbb{Z}_2,\mathbb{Z}_2) = \mathbb{Z}_2$——模 2 同调「看见」了
$\mathbb{RP}^2$ 的 2-维洞，而整系数同调看不见。模 2 系数是研究挠空间的利器。</span>

## 3 上同调与万有系数定理（上同调版）

第 4 篇会正式定义上同调；这里先给出它的万有系数定理，因为它揭示了上同调的本质。

**定理（上同调万有系数定理）。** 对任意空间 $X$ 与阿贝尔群 $G$，存在分裂的短正合序列

$$0 \longrightarrow \operatorname{Ext}\big(H_{n-1}(X),\ G\big) \longrightarrow H^n(X; G) \longrightarrow \operatorname{Hom}\big(H_n(X),\ G\big) \longrightarrow 0$$

**这个定理的哲学意义**：$H^n(X;G)$ **几乎就是** $\operatorname{Hom}(H_n(X),
G)$——上同调是「同调群上的线性函数」的集合，只差一个 $\operatorname{Ext}$ 修正项。<span class="marginnote">「上同调 = 同调的 Hom」是理解上同调的第一直觉：$H^n(X;\mathbb{Z})$ 中的每个类给 $H_n$
的每个类赋一个整数（即「积分」）。第 4 篇的杯积、Poincaré
对偶都建立在这个直觉上——上同调之所以能「相乘」，正是因为它本质是「函数环」。</span>

**关键推论**：**上同调由同调完全决定**（给定
$G$）。这似乎让上同调「多余」？不——上同调有同调没有的**乘法结构**（杯积），那是「函数相乘」天然带来的，同调自己看不见。第 4
篇将以杯积为中心展开。

## 4 公式解析：上同调万有系数定理

$$0 \to \operatorname{Ext}\big(H_{n-1}(X),\ G\big) \to H^n(X; G) \xrightarrow{\;h\;} \operatorname{Hom}\big(H_n(X),\ G\big) \to 0$$

- **第一步，主项 $\operatorname{Hom}(H_n(X), G)$**：$H^n(X;G)$ 里的类给每个 $n$-维同调类赋一个 $G$-值——映射 $h$ 是「求值」：把上同调类 $\varphi$ 送到「在 $H_n$ 上的求值函数」。**主项说：上同调类基本就是同调类上的函数。**
- **第二步，修正项 $\operatorname{Ext}(H_{n-1}(X), G)$**：低一维的挠会造成「不能由求值函数看出」的上同调类，这正是 $\operatorname{Ext}$ 记录的。**维度差 1 与同调版本中 Tor 的维度差 1 同源**。
- **第三步，分裂**：同构 $H^n(X;G) \cong \operatorname{Ext}(H_{n-1}(X),G) \oplus \operatorname{Hom}(H_n(X),G)$（不典范）。**一旦 $H_{n-1}$ 无挠，上同调就纯粹是 $\operatorname{Hom}(H_n, G)$。**

**辨析｜易错点：** 上同调版的索引是 $H_{n-1}$ 配 $\operatorname{Ext}$、$H_n$ 配
$\operatorname{Hom}$，与同调版的 $H_n \otimes G$、$H_{n-1}$ 配 $\operatorname{Tor}$
刚好「错开又对称」。对照记忆：**同调版「张量主项在上、Tor 修正在下维」，上同调版「Hom 主项在上、Ext 修正在下维」。**


**例：用万有系数定理算 $H^n(\mathbb{RP}^2;\mathbb{Z})$。** 已知 $H_0 = \mathbb{Z}$，$H_1 =
\mathbb{Z}_2$，$H_2 = 0$。上同调 UCT：$H^n \cong \operatorname{Ext}(H_{n-1},
\mathbb{Z}) \oplus \operatorname{Hom}(H_n, \mathbb{Z})$。$H^0 =
\operatorname{Hom}(\mathbb{Z},\mathbb{Z}) = \mathbb{Z}$；$H^1 =
\operatorname{Ext}(\mathbb{Z}_2,\mathbb{Z}) \oplus
\operatorname{Hom}(\mathbb{Z}_2,\mathbb{Z}) = \mathbb{Z}_2 \oplus 0 =
\mathbb{Z}_2$（因为 $\operatorname{Ext}(\mathbb{Z}_m, \mathbb{Z}) =
\mathbb{Z}_m$，而 $\mathbb{Z}_m \to \mathbb{Z}$ 只有零同态）；$H^2 =
\operatorname{Ext}(\mathbb{Z}_2,\mathbb{Z}) =
\mathbb{Z}_2$。**注意**：$H^2(\mathbb{RP}^2) = \mathbb{Z}_2$ 但 $H_2(\mathbb{RP}^2)
= 0$——上同调比同调「多」了一个维度信息，这正是 $\operatorname{Ext}$ 的贡献，也是杯积能存在的代数前提。

**Tor 与 Ext 的统一来源**：两者都可以从「取自由分解再应用函子」得到——$H_n(X;G)$ 的 Tor 项来自「张量掉
$G$」，$H^n(X;G)$ 的 Ext 项来自「Hom 进 $G$」。这套「自由分解 +
导出函子」是第二级《同调代数》的主线；此处你只需记住一个计算口诀：**$\operatorname{Tor}$
管「乘法的误差」，$\operatorname{Ext}$ 管「映射的误差」，两者都把维度压低一维出现在公式里。**

**一个选择直觉**：当 $G = \mathbb{Q}$，$\operatorname{Tor}(-, \mathbb{Q}) =
\operatorname{Ext}(-, \mathbb{Q}) = 0$（$\mathbb{Q}$
无挠且可除），公式化为纯直和。**「有理系数天下太平」**——没有挠，一切公式都变干净，代价是丢掉全部挠信息。很多几何定理先证有理版本（省事），再升级到
$\mathbb{Z}$ 版本（费事但完整），这条「由易到难」的路线会反复出现。

## 5 小结

- **同调万有系数定理**：$0 \to H_n(X) \otimes G \to H_n(X;G) \to \operatorname{Tor}(H_{n-1}(X), G) \to 0$，分裂。
- **上同调万有系数定理**：$0 \to \operatorname{Ext}(H_{n-1}(X),G) \to H^n(X;G) \to \operatorname{Hom}(H_n(X), G) \to 0$，分裂。
- **口诀**：张量积配 Tor，Hom 配 Ext；修正项都在低一维。
- **推论**：上同调由同调决定；$G=\mathbb{Q}$ 磨平挠，$G=\mathbb{Z}_p$ 捕捉 $p$