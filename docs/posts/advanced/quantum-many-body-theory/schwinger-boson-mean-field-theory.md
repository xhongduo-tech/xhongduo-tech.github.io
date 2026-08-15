---
title: Schwinger玻色子平均场理论
date: 2026-08-07
---

# Schwinger玻色子平均场理论

<div class="epigraph">
<p>当磁性序被量子涨落摧毁、自旋波发散时，我们需要一个「不预设任何序」的自旋表示——把自旋写成两个玻色子，让磁序与无序在这个统一的语言里自由竞争。Schwinger 玻色子就是这样的语言。</p>
<footer>—— A. Auerbach（*Interacting Electrons and Quantum Magnetism\*）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ A. Auerbach, *Interacting Electrons and Quantum Magnetism\*, Ch. 18 ｜ 2026-08-07</p>
</div>

## 为什么需要 Schwinger 玻色子

自旋波理论（Holstein-Primakoff）从**有序的基态**出发——它假设 Néel 序存在，然后算涨落。NLSM 处理「序的临界」但仍以序参量为场。但如果体系**根本没有磁序**（量子自旋液体），这两种语言都从错误的起点出发。

**Schwinger 玻色子表示（Schwinger boson representation）**换了一个思路：不预设任何序，把自旋直接写成两个玻色子。这个表示的美妙之处在于——**磁有序与自旋液体可以在同一个框架里竞争**：平均场解开出的「价键凝聚」给出 Néel 序，不凝聚则给出自旋液体。它把「序 vs 无序」从「选起点」变成「看平均场解的长相」。<span class="marginnote">Schwinger 玻色子源于 Schwinger 的角动量理论（1952 年）：用两个谐振子表示角动量。Arovas-Auerbach 1988 年把它用于反铁磁 Heisenberg 模型，得到自旋液体的平均场图像；Read-Sachdev 随后发展了它的规范结构。它是理解高温超导 RVB（共振价键）理论的核心工具之一。</span>

## 1 Schwinger 玻色子表示

对自旋 $S$，定义两个玻色子 $b_\uparrow, b_\downarrow$，自旋算符表示为：

$$\mathbf{S}_i = \frac{1}{2}\,b^\dagger_{i\alpha}\,\boldsymbol{\sigma}_{\alpha\beta}\,b_{i\beta}$$

即 $S^z_i = \frac{1}{2}(b^\dagger_{i\uparrow}b_{i\uparrow} - b^\dagger_{i\downarrow}b_{i\downarrow})$，$S^+_i = b^\dagger_{i\uparrow}b_{i\downarrow}$，$S^-_i = b^\dagger_{i\downarrow}b_{i\uparrow}$。为保证自旋大小为 $S$，需要**局域约束**：

$$\sum_\sigma b^\dagger_{i\sigma}b_{i\sigma} = 2S$$

**重点：Schwinger 玻色子的核心是「约束」——每个格点的玻色子总数固定为 $2S$。** 这个约束是局域的（每格点一个），把「自旋」这个 $2S+1$ 维对象编码进两个玻色子的占据数。物理上：$S=1/2$ 时每个格点恰有 1 个玻色子（$b^\dagger_\uparrow|0\rangle$ = 自旋上，$b^\dagger_\downarrow|0\rangle$ = 自旋下）。<span class="marginnote">与 Holstein-Primakoff 的对比：HP 变换围绕<strong>一个选定方向</strong>线性化（适合有序），Schwinger 玻色子是<strong>方向中性的</strong>（两个玻色子地位平等），所以它「不知道」任何序的方向——这就是它能处理无序相的原因。代价是必须硬处理约束，而约束正是平均场的难点。</span>

## 2 平均场分解

把 Schwinger 玻色子代入 Heisenberg 哈密顿量。交换项 $\mathbf{S}_i\cdot\mathbf{S}_j$ 用恒等式分解成「配对」通道：

$$\mathbf{S}_i\cdot\mathbf{S}_j = -\frac{1}{2}\Big(\underbrace{b^\dagger_{i\alpha}b_{i\beta}b^\dagger_{j\beta}b_{j\alpha}}_{\text{密度-密度}} - \underbrace{b^\dagger_{i\alpha}b^\dagger_{j\beta}b_{i\beta}b_{j\alpha}}_{\text{配对}}\Big)$$

定义两个**平均场序参量**：

$$\chi_{ij} = \langle b^\dagger_{i\alpha}b_{j\alpha}\rangle \quad(\text{凝聚/hopping 通道}), \qquad Q_{ij} = \langle b_{i\alpha}b_{j\alpha}\rangle \quad(\text{配对通道})$$

用 $\chi$ 与 $Q$ 做 Hubbard-Stratonovich 分解，得到平均场哈密顿量：

$$H_{\text{MF}} = \sum_{\mathbf{k}\alpha}\varepsilon_{\mathbf{k}}\,b^\dagger_{\mathbf{k}\alpha}b_{\mathbf{k}\alpha} + \frac{1}{2}\sum_{\mathbf{k}}\big(\Delta_{\mathbf{k}}b^\dagger_{\mathbf{k}\uparrow}b^\dagger_{-\mathbf{k}\downarrow} + \text{h.c.}\big) + \sum_i\lambda_i\Big(\sum_\sigma b^\dagger_{i\sigma}b_{i\sigma} - 2S\Big)$$

其中 $\lambda_i$ 是约束的拉格朗日乘子（化学势），$\varepsilon_{\mathbf{k}}$、$\Delta_{\mathbf{k}}$ 由 $\chi,Q$ 的 Fourier 变换给出。<span class="marginnote">$\lambda_i$ 的角色至关重要：它把「每个格点 $2S$ 个玻色子」的硬约束变成平均场里的软约束（化学势）。约束的严格处理需要规范理论（Read-Sachdev），平均场近似先把它软化——代价是可能丢失「规范涨落」的物理（如 deconfined 自旋子）。</span>

**重点：平均场把自旋问题变成「配对玻色子问题」——与 BCS 的配对方程结构同构，只是玻色子配对（动 $U(1)$ 对称）。** 两种通道 $\chi$（凝聚）与 $Q$（配对）竞争，谁占优决定磁性的类型。

## 3 平均场的两个解：有序与无序

解平均场方程（自洽确定 $\chi,Q,\lambda$），得到两类解：

**凝聚解（condensed）**：当 $\min_{\mathbf{k}}\varepsilon_{\mathbf{k}} = \lambda$ 时，$\mathbf{k}=0$ 处玻色子**凝聚**（BEC！），$\chi \neq 0$ 宏观占据。玻色凝聚意味着每个格点有确定的相对相位——**这正是 Néel 序**：交错的相位对应交错磁化。磁振子 = 凝聚体上的涨落。

**非凝聚解（uncondensed）**：当 $\varepsilon_{\mathbf{k}} > \lambda$ 恒成立，玻色子不凝聚，$\chi=0$，体系没有长程磁序——**量子自旋液体**。激发是能隙化的（或无能隙的，取决于 $\Delta_{\mathbf{k}}$ 结构）**自旋子（spinon）**——分数化的 $S=1/2$ 激发（玻色子本身，而非磁振子的 $S=1$）。

**重点：Schwinger 玻色子平均场把「磁有序 vs 自旋液体」变成了「玻色子凝聚 vs 不凝聚」——一个 BEC 判据决定了磁性类型。** 这套框架天然包含了两种可能性，而不像自旋波那样只从有序出发。凝聚解在长波极限等价于 NLSM/自旋波（可以证明：凝聚体的相位涨落 → Goldstone 磁振子），非凝聚解则给自旋液体以具体实现。<span class="marginnote">凝聚的物理：玻色子凝聚 = 格点间出现长程相位相干 = 自旋方向被锁定 = 磁有序。所以「磁有序」在 Schwinger 玻色子语言里只是「玻色凝聚的副产品」——这与 BEC 是超流的序参量完全平行。而自旋液体的自旋子激发携带 $S=1/2$（两个自旋子组成一个 $S=1$ 磁振子），是「分数化」的先声（本专题第 5 篇《分数量子霍尔效应与分数化》）。</span>

## 4 公式解析：自旋子谱与能隙

把平均场的准粒子谱算出来，看能隙如何出现：

- **第一步，对角化配对哈密顿量**：含 $b^\dagger b$ 与 $b^\dagger b^\dagger$ 项的二次型，用 Bogoliubov 变换（与 BCS、玻色凝聚同款）对角化，得到**自旋子谱**：
  $$\omega_{\mathbf{k}} = \sqrt{\varepsilon_{\mathbf{k}}^2 - |\Delta_{\mathbf{k}}|^2}$$
- **第二步，约束确定化学势**：$\lambda$ 由 $\sum_{\mathbf{k}}\langle b^\dagger_{\mathbf{k}\sigma}b_{\mathbf{k}\sigma}\rangle = N(2S)$ 自洽确定——总玻色子数固定在 $2SN$。
- **第三步，判凝聚**：若 $\varepsilon_{\mathbf{k}=0} = \lambda$，最低模软化到零，玻色子凝聚（Néel 序）；若 $\varepsilon_{\mathbf{k}}>\lambda$ 对所有 $\mathbf{k}$ 成立，最低模有正能隙 $\Delta_{\text{gap}} = \min_{\mathbf{k}}\omega_{\mathbf{k}}>0$，自旋子有能隙（gapped 自旋液体）。
- **第四步，物理**：能隙化自旋液体的基态是**短程价键态（RVB 型）**——每对相邻格点的自旋组成局域单态（价键），价键的量子涨落构成液体。激发自旋子打破一个价键，携带 $S=1/2$ 传播。

**重点：自旋子能隙 = 自旋液体「价键强度」的度量；凝聚 = 磁序的信号。** 平均场解从能隙（自旋液体）到零模（Néel 序）的过渡，就是量子相变。这个「能隙化 vs 凝聚」的判据，是 Schwinger 玻色子平均场最重要的输出。

**辨析｜易错点：** 初学者常把「Schwinger 玻色子自旋液体」误当成「弱磁性」或「涨落很大的 Néel 序」。自旋液体是**本质上无磁序**的相——不是「序被涨落模糊」，而是「序参量精确为零」。区分：Néel 相有交错磁化（中子散射有布拉格峰），自旋液体没有磁布拉格峰、但有**自旋子连续谱**（非弹性中子散射的连续背景）——实验上正是靠这些指纹区分。

## 5 Schwinger 玻色子与「从极限到大模型」

Schwinger 玻色子理论的方法论启示：「**不预设答案的表示 + 平均场竞争**」——它不先假设体系有序还是无序，而是选一个「中性的语言」（两个玻色子），让物理自己通过平均场方程的竞争性解来「表态」。这在机器学习里对应「**架构中立 + 数据决定**」的理念：好的表示（架构/表征）不预设任务的答案，而是让优化过程自己找到结构。<span class="marginnote">更深的类比：Schwinger 玻色子的「约束」是硬信息（自旋大小固定）——类似机器学习里「约束优化」（如权重归一化、注意力 softmax 的归一化）；而「平均场解开约束」对应「用 soft 约束/正则化近似硬约束」。自旋液体的「自旋子分数化」也提示：<strong>体系的元激发可能不是「基本单元」本身，而是它的分数化组合</strong>——模型的「能力单元」（如某个功能模块）可能由更基本的「分数化组件」构成。可参考第四级《大模型原理》。</span>

对多体理论自身，Schwinger 玻色子是探索量子自旋液体与高温超导 RVB 的核心工具——下一节，我们把 Hubbard 模型在「掺杂 + 强关联」下推到 t-J 模型：**t-J 模型与高温超导**。

## 6 数值算例：从平均场解读出磁序

把 Schwinger 玻色子平均场放到正方格子 $S=1/2$ Heisenberg 模型上，看数怎么走。取 $J=1$（能量单位），自洽求解 $\chi$、$Q$ 与 $\lambda$。

- **凝聚判据**：$S=1/2$ 时约束 $\sum b^\dagger b = 1$（每格点恰一个玻色子）。平均场自洽解给出 $\varepsilon_{\mathbf{k}}$ 的最低点位置取决于配对结构 $Q$——配对使色散在 $\mathbf{k}=(\pi,\pi)$ 或 $(0,0)$ 处软化。
- **能隙化自旋液体**：当解满足 $\Delta_{\mathbf{k}}\neq0$ 且 $\varepsilon_{\mathbf{k}}>\lambda$ 时，最低自旋子模 $\omega_{\mathbf{k}}$ 有正能隙——基态是 gapped 短程价键液体，激发携带 $S=1/2$。这类解出现在阻挫格子（三角、笼目）上，是自旋液体的具体实现。
- **凝聚到 Néel 序**：若自洽解让 $\varepsilon_{\mathbf{k}=0}=\lambda$，玻色子在 $\mathbf{k}=0$ 凝聚，交错相位给出 Néel 序，磁振子能谱 $\omega_{\mathbf{k}}\propto|\mathbf{k}|$（线性、Goldstone）——与自旋波理论一致。

**重点：一个平均场，两种结局——凝聚即磁有序，不凝聚即自旋液体。** 具体走哪条路由格子的几何（阻挫程度）与自旋大小 $S$ 决定：$S=1/2$ 正方格子反铁磁有 Néel 序但量子涨落强，阻挫格子（三角、Kagome）则倾向自旋液体。<span class="marginnote">数值对照：DMRG 与精确对角化表明 Kagome 格子 $S=1/2$ Heisenberg 基态很可能是自旋液体（自 1990s 起持续争议），而正方格子明确有 Néel 序，交错磁化 $m_s\approx0.31$，被量子涨落从经典值 $1/2$ 显著压低。Schwinger 玻色子平均场定性复现了这个图景——这正是它作为「无序相语言」的价值。</span>

**从算例到直觉**：凝聚与非凝聚的差别，本质是「玻色子是否多到可以共享一个相位」。凝聚后每个格点的自旋方向锁定（磁有序），不凝聚时自旋方向随机漂移（液体）——「能不能凝聚」就是磁性类型的判据。

## 7 小结

- **Schwinger 玻色子表示**把自旋写成两个玻色子（$\mathbf{S}=\frac{1}{2}b^\dagger\boldsymbol{\sigma}b$），靠局域约束 $\sum b^\dagger b = 2S$ 锁住自旋大小。
- 它是**方向中性的**（不预设磁序），与 Holstein-Primakoff（围绕有序态）互补。
- 平均场用两个通道（凝聚 $\chi$、配对 $Q$）分解 Heisenberg 交换项，配合拉格朗日乘子处理约束。
- **凝聚解 → Néel 序**（玻色子 BEC = 磁有序）；**非凝聚解 → 自旋液体**（能隙化自旋子激发）。
- 自旋子谱 $\omega_{\mathbf{k}}=\sqrt{\varepsilon_{\mathbf{k}}^2-|\Delta_{\mathbf{k}}|^2}$：最低模软化 → 凝聚；正能隙 → gapped 自旋液体（RVB 型基态）。
- 自旋子携带 $S=1/2$，是「分数化」的先声；平均场把「序 vs 无序」变成 BEC 判据。

在下一节，我们把强关联推向掺杂：**t-J 模型与高温超导**——当 Mott 绝缘体掺入空穴，束缚的自旋对如何可能变成配对的电荷，以及这如何是高温超导的候选机制。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| Schwinger 表示 | $\mathbf{S}_i = \frac{1}{2}b^\dagger_{i\alpha}\boldsymbol{\sigma}_{\alpha\beta}b_{i\beta}$ | 自旋写成两个玻色子，方向中性 |
| 局域约束 | $\sum_\sigma b^\dagger_{i\sigma}b_{i\sigma} = 2S$ | 玻色子总数固定，锁住自旋大小 |
| 交换项分解 | $\mathbf{S}_i\cdot\mathbf{S}_j = -\frac{1}{2}(\text{密度-密度} - \text{配对})$ | 凝聚通道 $\chi$ 与配对通道 $Q$ |
| 自旋子谱 | $\omega_{\mathbf{k}} = \sqrt{\varepsilon_{\mathbf{k}}^2 - |\Delta_{\mathbf{k}}|^2}$ | Bogoliubov 型，最低模软化即凝聚 |
| 判据 | 凝聚 → Néel 序；不凝聚 → 自旋液体 | 一个 BEC 判据决定磁性类型 |

**易错复盘**：两点要盯住。其一，Schwinger 玻色子平均场的约束处理是「软」的（化学势 $\lambda$），严格的硬约束需要规范理论——平均场的局限在于丢失规范涨落；其二，「非凝聚解」不是「弱磁性」而是「本质上无磁序」的自旋液体——序参量精确为零，激发是分数化自旋子。

**知识连线**：本篇与第 3 篇 Bogoliubov 理论同构（玻色化 + Bogoliubov 对角化），区别是凝聚体是磁序而非粒子凝聚；它又是第 4 篇 Heisenberg 模型的「无序相工具」，与自旋波（有序相）互补。自旋子携带 $S=1/2$，是第 4 篇《量子自旋液体与拓扑序》分数化概念的微观实现。

**延伸思考**：为什么 Schwinger 玻色子表示能同时容纳磁序与自旋液体，而 Holstein-Primakoff 只能从有序出发？提示：前者方向中性、不预设序；后者围绕选定方向线性化。约束 $\sum b^\dagger b = 2S$ 在 $S=1/2$ 时变成什么？提示：每格点恰好一个玻色子。


**实践与辨析**：一道综合题：写出 $S=1/2$ 时 Schwinger 玻色子的约束，并说明在平均场里它是如何被软化的（化学势 $\lambda$）。提示：约束 $\sum b^\dagger b = 2S$ 在 $S=1/2$ 时变成「每格点恰一个玻色子」；平均场把硬约束换成 $\sum\lambda_i(\sum b^\dagger b - 2S)$，由自洽条件确定 $\lambda$。再想一步：若配对通道 $Q=0$（只剩凝聚通道 $\chi$），平均场会退化到什么图像？提示：退化为经典 Néel 的图像，丢失自旋液体。

**自查清单**：学完本篇，你能不加参考地说清吗——
- Schwinger 玻色子表示与 Holstein-Primakoff 的本质差别（方向中性 vs 预设有序）。
- 为什么「玻色子凝聚 = 磁有序」是一个自洽的判据，而不只是一句口号。
- 自旋子能隙 $\Delta_{\text{gap}}$ 与自旋液体「价键强度」的关系。
- 凝聚解在长波极限如何等价于 NLSM/自旋波（相位涨落 → Goldstone 磁振子）。