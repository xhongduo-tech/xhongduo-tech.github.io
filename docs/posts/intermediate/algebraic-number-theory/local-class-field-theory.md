---
title: 局部类域论
date: 2026-08-11
---

# 局部类域论

<div class="epigraph">
<p>科学家研究自然并非因为它有用，而是因为他从中得到乐趣；而他从中得到乐趣，是因为它是美的。</p>
<footer>—— 儒勒 · 昂利 · 庞加莱（Henri Poincaré，Le savant n'étudie pas la nature parce qu'elle est utile）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从局部类域论开始

前几节的舞台是**全局**：数域 $K$ 连同它所有的素数。但现代数论的基本信条是「**局部决定全局**」——把所有注意力集中在单个素理想上（完备化成 $\mathbb{Q}_p$ 那样的局部域），把问题「降维」再求解。**局部类域论**回答的问题是：局部域 $K$（如 $\mathbb{Q}_p$ 的有限扩张）的阿贝尔扩张，能否被 $K^\times$ 的简单数据完全分类？答案是**能**，而且分类器极其漂亮：**有限阿贝尔扩张 ↔ 范数子群**，中间的桥梁是 Artin 映射。这是整个类域论（全球与局部）的发动机。

## 1 局部域与范数子群

**局部域（local field）**：完备的离散赋值域，且剩余类域有限。标准例子：$\mathbb{Q}_p$ 的有限扩张 $K$（特征 $0$），或 $\mathbb{F}_q((t))$ 的有限扩张（特征 $p$）。<span class="marginnote">局部域拥有完整的工具包：赋值环 $\mathcal{O}_K$、极大理想 $\mathfrak{p}$、剩余类域 $\mathbb{F}_q$、单位群 $\mathcal{O}_K^\times$。相较全局数域，它们「每个素数各安其位」，结构干净得多——这是它们适合做阿贝尔扩张分类的原因。</span>

对有限扩张 $L/K$，定义**范数映射** $\mathrm{N}_{L/K}: L^\times \to K^\times$。其像

$$
\mathrm{N}_{L/K}(L^\times) \subseteq K^\times
$$

称为 $L$ 的**范数子群（norm group）**。它是 $K^\times$ 中**开、有限指数**的子群。

**核心例子（无分歧扩张）**：$L/K$ 无分歧、次数 $n$（$[L:K] = n$，剩余类域扩张）。此时 $L^\times$ 的元素写成 $\varpi_L^m \cdot u$（$\varpi$ 为素元），范数落在 $K^\times$ 的 $\pi^m \cdot \mathrm{N}(u)$。范数子群恰好是

$$
\mathrm{N}_{L/K}(L^\times) = \langle \pi \rangle^{n} \cdot \mathcal{O}_K^\times
$$

其中 $\pi$ 是 $K$ 的素元。<span class="marginnote">看这个例子的结构：无分歧扩张的范数子群只「压缩」素元部分（$\pi$ 的幂次变 $n$ 倍），单位群 $\mathcal{O}_K^\times$ 则全部保留。商 $K^\times / \mathrm{N}(L^\times) \cong \mathbb{Z}/n$ 由素元的像生成——它正好同构于 Galois 群 $\mathrm{Gal}(L/K) = \mathbb{Z}/n$（Frobenius 生成）。这已经预告了 Artin 映射。</span>

**范数子群为什么是「开」子群**：$\mathrm{N}_{L/K}$ 是连续映照，$\mathrm{N}_{L/K}(L^\times)$ 是局部紧拓扑下的开子群——**正是「开」保证了商群 $K^\times/\mathrm{N}$ 有限**，互反映射的定义域才可控、才配得上「分类」二字。

## 2 局部互反律：范数子群分类阿贝尔扩张

**定理（局部类域论 / 局部互反律）：** 设 $L/K$ 是局部域的**有限阿贝尔**扩张。存在一个典范的群同构——**Artin / 互反映射**

$$
\mathrm{rec}: \frac{K^\times}{\mathrm{N}_{L/K}(L^\times)} \;\longrightarrow\; \mathrm{Gal}(L/K)
$$

把 $K^\times$ 中素元（uniformizer）$\pi$ 的像映到 Frobenius 自同构 $\mathrm{Fr}_{L/K}$。反过来，**每个开有限指数的子群 $H \subseteq K^\times$ 都是某个有限阿贝尔扩张的范数子群**。<span class="marginnote">这两个方向合成完全分类：$L \mapsto \mathrm{N}(L^\times)$ 把阿贝尔扩张嵌入 $K^\times$ 的「开有限指数子群」布尔格；$\mathrm{rec}$ 给出商群到 Galois 群的同构。局部阿贝尔扩张的分类被<strong>彻底归结为 $K^\times$ 的算术</strong>——这是「类域」这个名称的由来：$K$ 的「类」（阿贝尔扩张类）由「域」$K^\times$ 决定。</span>

**关键精确序列**（局部互反律的标准形式）：

$$
1 \longrightarrow \mathrm{N}_{L/K}(L^\times) \longrightarrow K^\times \xrightarrow{\;\mathrm{rec}\;} \mathrm{Gal}(L/K) \longrightarrow 1
$$

这条精确序列把「范数子群是核」表述得最彻底：**$K^\times$ 去掉范数部分，剩下的就是 Galois 群**。让 $L$ 跑遍所有有限阿贝尔扩张取极限，得到局部互反律的整体形态：$K^\times$ 的 profinite 完备化与 $\mathrm{Gal}(K^{\mathrm{ab}}/K)$ 同构——局部类域论的完全形态。

**Fr 与 Artin 映射**：无分歧时 $\mathrm{rec}(\pi \bmod \mathrm{N}) = \mathrm{Fr}_{L/K}$（剩余类域的 Frobenius）；分歧部分由惯性子群吸收。于是局部互反律把「算术生成元 $\pi$」和「拓扑生成元 $\mathrm{Fr}$」配对——**同一个生成元在算术与 Galois 两侧互相呼应**。

## 3 应用：Hilbert 符号与 Kummer 理论

局部类域论的第一个直用是 **Hilbert 符号**：对 $K^\times$ 中 $a, b$，定义

$$
(a, b)_K \in \{\pm 1\}, \qquad (a, b)_K = \mathrm{rec}(b)\big(\sqrt{a}\big) / \sqrt{a}
$$

把「$b$ 的互反作用在 $\sqrt{a}$ 上的符号」提取出来，是二次互反律、四元代数分类与局部中「$\mathbb{Q}_2$ 上 $-1$ 的平方性」等问题的手持计算器。<span class="marginnote">对 $K = \mathbb{Q}_p$，$(a,b)_{\mathbb{Q}_p}$ 是双线性、非退化的配对，还与全局二次型理论（四元代数、Hasse 不变量）逐点相扣。它是「符号法」在代数数论里最普及的遗产之一。</span>

**Kummer 理论**：当 $K$ 含 $n$ 次本原单位根时，$K^\times/(K^\times)^n$ 与 $K$ 的 $n$ 次阿贝尔扩张（Kummer 扩张 $K(\sqrt[n]{a})$）一一对应——它是「范数子群分类」在可解情形下的显式化，也是 Abel 扩张理论的骨架。类域论把它推广到一般情形，但「$K^\times$ 的商控制扩张」的思想脉络始终未变。<span class="marginnote">符号法的更深处：$(a,b)_K$ 可推广为 $n$ 次互反符号 $(\frac{a,b}{K})_n \in \mu_n$——Kummer 配对的局部版，把二次情形的 $\{\pm1\}$ 推广到任意单位根，是类域论通向 Langlands 的阶梯之一。</span>

## 4 公式解析：局部互反律的核心精确序列

$$
1 \longrightarrow \mathrm{N}_{L/K}(L^\times) \longrightarrow K^\times \xrightarrow{\mathrm{rec}} \mathrm{Gal}(L/K) \longrightarrow 1
$$

- **第一步，认两边**：中间是乘法群 $K^\times$（局部域的可逆元全体），右侧是阿贝尔 Galois 群。前者是「算术」的化身，后者是「对称」的化身——互反律就是声称两者通婚。
- **第二步，认核**：左侧的范数子群被完全映射为平凡——$K^\times$ 中「$L$ 的范数」的部分，正是 Galois 群下的**不动点**。交换群的 Galois 理论里「范数不动」与「正规扩张」互为因果。
- **第三步，认箭头**：$\mathrm{rec}$ 把素元映到 Frobenius。它的**唯一性**（不依赖选择）来自「$\pi$ 生成剩余类域」这一算术事实——**生成元的对应是典范的，这使整个同构成为同构而非任意配对**。
- **第四步，全部阿贝尔扩张**：让 $L$ 跑遍 $K$ 的有限阿贝尔扩张，得到 $K^\times$ 的「类」与 $K^{\mathrm{ab}}$ 的 Galois 群的对应——局部类域论的完整形态。

（互反映射的**典范性**是关键：$K^\times$ 的素元被强制送到 Frobenius，没有自由选择的余地——这正是同构「是唯一的一个」而非「存在某一个」的精确含义。）

## 5 从局部到全局：预告

局部类域论的价值一半在自身，一半在它给**全局**类域论（下一节）供能：全局的 Artin 互反律把「所有素理想上的局部 Artin 映射」拼成一条**idèle** 层面的映射。局部配对 $(a,b)_K$、局部范数子群、局部 Frobenius——是全局 Artin 符号在每一点的「切片」；而 Hasse 局部—整体原则保证某些全局问题（如 $ax^2 + by^2 = c$ 是否局部处处可解）可以逐点检查。**idèle 化的预告**：把每个素理想的局部互反律取「限制直积」拼接，就得到 idèle 类群上的全局 Artin 映射——下一节的全局版本正是所有 $\mathrm{rec}_{\mathfrak p}$、所有 $(a,b)_{\mathfrak p}$ 的逐点合成。<span class="marginnote">类域论的地图因此清晰：<strong>局部类域论</strong> = 每个 $\mathfrak{p}$ 的完备化上的阿贝尔扩张分类；<strong>全局类域论</strong> = 把所有点的数据打包成 idèle 类群，再得出一条全球性的互反律。下一节我们登上全球的舞台。</span>

**辨析｜易错点：** 局部互反律的映射方向（$K^\times$ → Galois 群）是「范数群为核」的版本；千万不要与 Artin 符号在全局的「素理想 → Frobenius」方向混淆。另外**范数子群必须是开的**——「开有限指数」是分类的完整性条件（拓扑与代数并存，缺一不可）。

## 6 实例：$\mathbb{Q}_p$ 上的阿贝尔扩张

**例 1（无分歧）**：$L = \mathbb{Q}_5(\sqrt{2})$。$2$ 模 $5$ 非平方（平方类 $\{1,4\}$），Hensel 引理排除 $\sqrt2 \in \mathbb{Q}_5$，故 $L$ 是次数 $2$ 的无分歧扩张，范数子群

$$
\mathrm{N}_{L/\mathbb{Q}_5}(L^\times) = \langle 5 \rangle^2 \cdot \mathbb{Z}_5^\times, \qquad \mathbb{Q}_5^\times / \mathrm{N} \cong \mathbb{Z}/2
$$

互反映射 $\mathrm{rec}(5) = \mathrm{Fr}$——Frobenius 恰好生成 Galois 群。

**例 2（分歧）**：$L = \mathbb{Q}_5(\sqrt{5})$。$v_5(\sqrt5) = \tfrac12$，素元 $\varpi = \sqrt5$，$e = 2$ 分歧。范数子群仍给出 $\mathbb{Q}_5^\times/\mathrm{N} \cong \mathbb{Z}/2$，但这次 $\mathrm{rec}$ 把素元映到**惯性群内的非平凡元素**而非 Frobenius——分歧部分的行为由 Hilbert 符号捕捉。

**例 3（Hilbert 符号）**：$(a, b)_{\mathbb{Q}_5} = 1 \iff a \in \mathrm{N}(\mathbb{Q}_5(\sqrt{b})^\times)$。取 $a = 2, b = -1$：$-1$ 模 $5$ 非平方，$\mathbb{Q}_5(\sqrt{-1})$ 是次数 $2$ 扩张；$x^2 + y^2 = 2$ 在 $\mathbb{Q}_5$ 有解（$1^2 + 1^2 = 2$），故 $(2, -1)_{\mathbb{Q}_5} = 1$。

**对照（$\mathbb{Q}_3$ 上的同一题）**：$x^2 = 2$ 在 $\mathbb{Q}_3$ 可解吗？$2$ 模 $3$ 非平方（平方类 $\{1\}$），Hensel 排除，故 $\sqrt2 \notin \mathbb{Q}_3$，$\mathbb{Q}_3(\sqrt2)$ 同样是次数 $2$ 的无分歧扩张——$3$-adic 与 $5$-adic 的「同一道题」行为一致，这正是局部世界的高度规律性。<span class="marginnote">这套「解方程判定符号」的操作，正是四元代数、二次型理论里「局部-整体」逐点检查的标准姿势：每个素数算一个 $(a,b)_p$，再拼出全局结论。</span>

**辨析｜易错点：** 无分歧与分歧扩张的范数子群**都能**给出 $\mathbb{Z}/2$ 的商——**「范数子群 ↔ 阿贝尔扩张」的一一对应不直接区分分歧与否**，区别在 $\mathrm{rec}$ 把素元映到哪里（Frobenius 还是惯性元）。「范数子群分类」是类域论本体；「是哪种扩张」要靠 $\mathrm{rec}$ 的像去读。

## 7 小结

- **局部域**：$\mathbb{Q}_p$ 或 $\mathbb{F}_q((t))$ 的有限扩张；工具是赋值环、素元、单位群、剩余类域。
- **范数子群** $\mathrm{N}_{L/K}(L^\times) \subseteq K^\times$：开有限指数子群；无分歧情形 $= \langle \pi\rangle^n \mathcal{O}_K^\times$。
- **局部互反律**：$\mathrm{rec}: K^\times / \mathrm{N}(L^\times) \xrightarrow{\sim} \mathrm{Gal}(L/K)$，$\pi \mapsto \mathrm{Fr}$；阿贝尔扩张 ↔ 范数子群完全一一对应。
- 应用：Hilbert 符号 $(a,b)_K$ 判别二次扩域与四元代数；Kummer 理论用 $K^\times/(K^\times)^n$ 分类 $n$ 次扩张。
- 全局互反律 = 各点局部互反律的打包：下一节进入全局舞台。

在下一节，我们将把局部 Artin 映射沿所有素理想拼装成一条全球的互反律——**Artin 互反律**，它把阿贝尔扩张的 Galois 群与「理想类群」的普遍版（idèle 类群）接通，二次互反律只是它最小的一格投影。
