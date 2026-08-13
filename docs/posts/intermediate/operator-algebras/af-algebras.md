---
title: AF 代数
date: 2026-08-07
---

# AF 代数

<div class="epigraph">
<p>我不能创造的东西，我就不理解。</p>
<footer>—— 理查德 · 费曼（Richard Feynman）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Davidson《C\*-Algebras by Example》第10章 ｜ 2026-08-07</p>
</div>

## 为什么从 AF 代数开始

费曼说「不能创造就不理解」。AF 代数正是这句话的数学化身：它**由有限维 C\* 代数拼成**，每一块 $M_{n_1}\oplus\cdots\oplus M_{n_k}$ 都是「可以亲手造出来」的对象，而 AF 代数就是它们的**归纳极限**。逼近有限维（Approximately Finite-dimensional），故得名 **AF 代数**。

AF 代数的魅力在于「可造性」：它的投影可以显式构造、迹可以显式积分、理想可以显式列出。而它又不平凡：无理旋转代数的某些约化、群代数 $C^*_r(\mathbb{F}_2)$ 的某些子代数、以及整个 UHF 家族都藏身其中。更重要的是，**AF 代数是历史上第一个被 Elliott 不变量完备分类的 C\* 代数类**——分类理论的序幕从这里拉开。

## 1 有限维 C\*-代数：一切的积木

**定理（Wedderburn 结构定理）**：有限维 C\* 代数 $A$ 同构于矩阵代数的直和

$$A \cong M_{n_1}(\mathbb{C}) \oplus M_{n_2}(\mathbb{C}) \oplus \cdots \oplus M_{n_k}(\mathbb{C}).$$

这一条是有限维世界的地基：**每个有限维 C\* 代数都是一摞矩阵块的直和**。块的数量 $k$ 与尺寸 $(n_1,\dots,n_k)$ 完整刻画了 $A$ 的同构类。<span class="marginnote">Wedderburn 定理把「有限维抽象代数」全部化成「具体矩阵块」。$M_n$ 是简单代数（第 12 篇），直和让它们互相独立地并排放置。投影、迹、态在这套「块状」图景下都可显式写出：每个 $M_{n_i}$ 有唯一的迹 $\frac1{n_i}\mathrm{Tr}$。</span>

**投影的等价**：$M_n$ 中两个投影 $p,q$ 等价（$p\sim q$，Murray–von Neumann 意义：存在部分等距 $v$，$v^*v=p$，$vv^*=q$）当且仅当 $\mathrm{rank}\,p=\mathrm{rank}\,q$。于是「投影的等价类」按秩计数，而秩就是「维数」。这为 $K_0$ 群提供了原始模型。

## 2 AF 代数：归纳极限

**AF 代数（approximately finite-dimensional C\*-algebra）**：存在有限维 C\* 代数链

$$A_1 \hookrightarrow A_2 \hookrightarrow A_3 \hookrightarrow \cdots$$

（含单位嵌入），使 $A=\overline{\bigcup_n A_n}$，即 $A$ 是 $\{A_n\}$ 的归纳极限。等价刻画：对每个有限集 $\{a_1,\dots,a_k\}\subset A$ 与 $\varepsilon>0$，存在有限维子代数 $F\subset A$ 与 $b_i\in F$ 使 $\|a_i-b_i\|\lt \varepsilon$——**有限维子代数在 $A$ 中稠密**。

**例子（UHF 代数）**：取 $A_n=M_{2^n}$ 且 $M_{2^n}\hookrightarrow M_{2^{n+1}}$ 沿对角嵌入 $a\mapsto\begin{pmatrix}a&0\\0&a\end{pmatrix}$，极限 $A=M_{2^\infty}$ 是 **UHF 代数**（uniformly hyperfinite），又称 CAR 代数（第 18 篇的主角）。UHF 代数完全由「超因子」 $d$（$2,2^\infty$ 的「超积」型记号）决定，$M_{2^\infty}$ 的 $K_0\cong\mathbb{Z}[1/2]$。

**例子（Ries–AF）**：无理旋转代数 $A_\theta$ 的约化…不，$A_\theta$ 本身不是 AF；但 $C^*_r(\mathbb{F}_2)$ 的某些子代数、以及 Cuntz 代数的「对角」部分是 AF。**AF 代数不交换**（如 $M_{2^\infty}$ 非交换），却依然完全可控。

**命题（AF 的理想结构）**：AF 代数的闭理想是良序的（按包含构成全序）——这是「可造性」的又一表现：AF 代数的理想格被 Bratteli 图完全读出。<span class="marginnote">理想良序是 AF 世界的标志性事实：比如 $M_{2^\infty}$ 的闭理想只有 $0$ 与自身（简单），而一般的 AF 代数的理想链可以很长，每条链都由图的「边收缩」决定。非交换且理想结构清晰，AF 代数因此成为测试理想理论的理想试验田。</span>

## 3 Bratteli 图：逼近的骨架

有限维链 $A_1\hookrightarrow A_2\hookrightarrow\cdots$ 的嵌入信息可以画成一张图：**Bratteli 图**。顶点分成 $n=1,2,\dots$ 层，第 $n$ 层的顶点对应 $A_n$ 的矩阵块 $M_{k}$；从第 $n$ 层的块 $M_{a}$ 到第 $n+1$ 层的块 $M_{b}$ 的边数，等于嵌入 $M_a\hookrightarrow M_b$ 中「把 $a$ 维块放大 $m$ 倍」的重数 $m$（满足 $b=a\cdot m$ 对每个块）。

**例**：$M_{2^\infty}$ 的 Bratteli 图是单顶点、每层一条重数 2 的边——「一直加倍」。

**命题（图的威力）**：Bratteli 图（连同迹态在其上诱导的测度）完全决定 AF 代数 $A$ 的结构；两个 AF 代数同构当且仅当它们的 Bratteli 图「同构」（在同尾意义下）。<span class="marginnote">Bratteli 图把「无穷维代数的构造」画成「无穷层的有限图」。Elliott 证明 $K_0$ 上的序结构足以读出图，从而把「图分类」提升为「代数不变量分类」。今天 Bratteli 图仍是构造 AF 代数的标准语言——从图出发造代数，从代数出发读图。</span>

## 4 公式解析：$K_0(A)=\lim_{\longrightarrow}K_0(A_n)$

$$
K_0(A) = \varinjlim K_0(A_n), \qquad K_0(M_n)\cong\mathbb{Z}
$$

- **第一步，看 $K_0(M_n)\cong\mathbb{Z}$**：$K_0$ 群把「投影等价类」收集成群（第 25 篇完整定义）。$M_n$ 的投影等价类由秩决定：$p\mapsto\mathrm{rank}\,p$ 给同构 $\mathbb{Z}$。对直和 $A_1\oplus\cdots\oplus A_k$，$K_0$ 就是 $\mathbb{Z}^k$，每个坐标记一个块的秩。
- **第二步，看归纳极限**：$K_0$ 是**保归纳极限**的函子：$A=\lim A_n$ 时，$K_0(A)=\lim K_0(A_n)$。嵌入 $A_n\hookrightarrow A_{n+1}$ 诱导 $K_0(A_n)\to K_0(A_{n+1})$（秩怎么被放大），极限把这些「秩的缩放」无穷次叠加。
- **第三步，看正锥**：$K_0(A)^+$ 是「真的投影」对应的半群（正元素 = 非负秩），$A$ 含幺时 $(K_0(A),K_0(A)^+,[1_A])$ 一起给出**有序 $K_0$**。对 $M_{2^\infty}$：$K_0\cong\mathbb{Z}[1/2]$，正锥是非负 dyadic 有理数，单位元 $[1]=1$。
- **第四步，看为什么它分类**：Elliott 定理（下节）说两个 AF 代数同构当且仅当它们的有序 $K_0$（连同单位元与迹单形）同构。K 理论不再是「不变量之一」，而是**完备不变量**——分类的圣杯在 AF 世界首次到手。

## 5 Elliott 定理：分类的序幕

**Elliott 定理（AF 分类，1976）**：设 $A,B$ 是含幺 AF 代数。则

$$A\cong B \iff (K_0(A), K_0(A)^+, [1_A]) \cong (K_0(B), K_0(B)^+, [1_B])$$

作为有序阿贝尔群（带单位元）。更一般地，迹态空间 $T(A)$（第 10 篇）与 $K_0$ 配合，给出**Elliott 不变量** $(K_0,K_0^+,[1],T(A))$。<span class="marginnote">Elliott 定理是 C\* 分类理论的「莱布尼茨时刻」：一个代数被它的 K 理论 + 迹完全决定。它宣告了一条计划——<strong>Elliott 纲领</strong>：用 $(K_0,K_0^+,[1],\text{迹单形})$ 分类所有可分的单 C\* 代数。这条纲领统治了随后四十年，直到第 26 篇的 Jiang–Su 代数与 Z 稳定性才划出精确边界。</span>

**Murray–von Neumann 等价是引擎**：分类的证明依靠「投影的 Murray–von Neumann 等价 + 有序 $K_0$ 的态射提升为 $\ast$-同构」的构造性步骤——AF 代数的可造性使「从不变量造同构」成为可能。这是「构造即理解」的最佳示范。

**辨析｜易错点：**不要以为「AF = 交换」。$M_{2^\infty}$ 非交换（矩阵块会越嵌越大），但它仍是 AF。**AF 刻画的是「被有限维逼近」而非「交换性」**。另一个易错点：$K_0$ 若不带**正锥与单位元**，信息不足（$K_0\cong\mathbb{Z}$ 的代数可以五花八门）；**有序结构才是分类的胜负手**。

## 6 例：从 Bratteli 图造 AF 代数

Bratteli 图不是画着玩的——从图出发可以直接「造」出 AF 代数。

**图例一（UHF $2^\infty$）**：单顶点、每层一条重数 2 的边。对应的代数链 $M_2\hookrightarrow M_4\hookrightarrow M_8\hookrightarrow\cdots$（对角加倍），极限 $M_{2^\infty}$。$K_0=\mathbb{Z}[1/2]$。

**图例二（UHF $3^\infty$）**：重数 3 的边。$M_{3^n}$ 链，极限 $K_0=\mathbb{Z}[1/3]$。$M_{2^\infty}\not\cong M_{3^\infty}$——$K_0$ 不同（$1/2$ 与 $1/3$ 的「分母集」不同）。

**图例三（乘积 $M_{2^\infty}\otimes M_{3^\infty}$）**：超因子 $d=6$ 的 UHF，$K_0=\mathbb{Z}[1/6]$。张量积在图上对应「重数相乘」。

**图例四（两个顶点的图）**：第 $n$ 层两个顶点、交叉嵌入 $M_2\oplus M_2\to M_4$。极限不是 UHF，而是「带理想结构」的 AF 代数——理想由图的「子图」读出。

**图与迹**：图的「边界测度」给出迹态。$M_{2^\infty}$ 的唯一迹态来自图的「均匀测度」。

**一句话总结**：Bratteli 图是 AF 代数的「基因图谱」——顶点是矩阵块，边是嵌入，极限代数的一切（理想、迹、K 群）都由图读出。

## 7 延伸：UHF 代数的超因子

UHF 代数是最规整的一类 AF 代数，它的「超因子」理论值得单独说说。

**定义**：$A$ 是 UHF 若它是 $M_{k_1}\hookrightarrow M_{k_2}\hookrightarrow\cdots$ 的极限（$k_n\mid k_{n+1}$）。「uniformly hyperfinite」——一致超有限。

**超因子（supernatural number）**：$d$ 是「可数个素数的可数幂」之积（形式对象）。$M_{d}$ 记 UHF 代数，$K_0(M_d)=\mathbb{Z}[1/d]$。

**Glimm 定理**：$M_d\cong M_{d'}$ ⟺ $d=d'$。UHF 代数由超因子完全分类——这是 Elliott 分类（第 17 篇 §5）在 UHF 上的最简形态。

**$M_{2^\infty}$ 的特殊地位**：它与 CAR 代数（第 18 篇）同构，是费米子 Fock 空间的代数——凝聚态、量子场论里无处不在。

**UHF 的迹与态**：$M_d$ 有唯一迹态（来自各 $M_{k_n}$ 的归一化迹的极限）。$M_d$ 简单、有唯一迹——「温和」到极致。

**一句话总结**：UHF 代数 = 超因子决定的 AF 代数；$K_0=\mathbb{Z}[1/d]$ 是它们的「身份证号」。

## 8 延伸：Elliott 纲领的起点

Elliott 定理（有序 $K_0$ 分类 AF）不是孤立结果，它是整个分类纲领的源头。

**为什么从 AF 开始**：AF 代数是「可造」的——投影显式、迹显式、Bratteli 图显式。可造性是分类可证的前提：「从不变量造同构」在 AF 世界里可以一步步实现。

**从不变量到同构**：Elliott 定理的证明 = 把有序 $K_0$ 的同构提升为 $\ast$-同构：先对应投影（K 理论），再对应生成元（迹/稠密），最后用范数控制完成。这就是后来一切分类定理的「标准剧本」。

**不变量包含什么**：$(K_0,K_0^+,[1])$ 对 AF 足够；一般情形还要 $K_1$、迹单形、配对（第 26 篇）。AF 是「不变量最少」的幸运儿。

**纲领的扩展**：Elliott 1976 年的猜想把「有序 $K_0$」换成「完整 Elliott 不变量」，试图分类所有可分单可核代数。AF 是第一批被收编的——此后四十年，$A_\theta$、$\mathcal{O}_n$、Z-稳定代数逐一入列。

**一句话总结**：Elliott 定理是分类理论的「第一次登月」——它证明「不变量 = 身份」可以是真的，从而开启了整个分类纲领。

## 9 小结

- **Wedderburn**：有限维 C\* 代数 = 矩阵块直和 $M_{n_1}\oplus\cdots\oplus M_{n_k}$。
- **AF 代数**：有限维子代数稠密的 C\* 代数（归纳极限）；$M_{2^\infty}$、CAR 代数是旗舰例。
- **Bratteli 图**：把有限维逼近画成图，理想、迹、结构全部可读。
- **$K_0(A)=\lim K_0(A_n)$**：保归纳极限，$M_n$ 的 $K_0=\mathbb{Z}$，正锥 + 单位元给出有序 $K_0$。
- **Elliott 定理**：有序 $K_0$ 完备分类 AF 代数——分类纲领由此开启。
- **教训**：AF ≠ 交换；$K_0$ 必须带正锥与单位元才有分类力。

在下一节，我们看两个「极端简单」的 AF 亲戚——**Cuntz 代数与 CAR 代数**：前者由等距生成、纯无限且单；后者是费米子 Fock 空间的代数、UHF $2^\infty$