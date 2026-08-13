---
title: 力迫关系、力迫定理与布尔值模型
date: 2026-08-07
---

# 力迫关系、力迫定理与布尔值模型

<div class="epigraph">
<p>力迫关系是预言：在 $V$ 里就写好一份「$G$ 会怎样判定每个命题」的剧本，而力迫定理保证剧本一定成真。</p>
<footer>—— 达纳 · 斯科特（Dana Scott）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第14章；Kunen 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从力迫关系开始

上一节我们看到，通用滤 $G$ 造出扩张 $V[G]$，并用「分叉稠密集」证明 $G$ 携带 $V$ 里没有的新对象。但还没回答：**$V[G]$ 里到底哪些命题为真？** 答案藏在 **力迫关系（forcing relation）$\Vdash$** 里：$p \Vdash \varphi$ 意为「任何包含 $p$ 的通用滤都使得 $V[G] \vDash \varphi$」。力迫定理则把这条「预言」变成精确的等价：**$V[G] \vDash \varphi$ 当且仅当存在 $p \in G$ 使 $p \Vdash \varphi$**。<span class="marginnote">力迫关系的意义：它把「$G$ 是什么」完全预先编码进 $V$ 里——$V[G]$ 里的真值由 $V$ 里的条件「强迫」决定，而 $V$ 无法预言的只是「哪个 $G$ 被选」。这就是 Cohen「$V$ 里的机器 + 一个通用滤 = $V[G]$ 的一切」的精髓。</span>

今天把力迫关系定义清楚，证明力迫定理的两半，再给一个「代数版」——**布尔值模型 $V^{\mathbb{B}}$**：把力迫翻译成布尔代数的运算，让「真值」变成一个布尔值而非只有 0/1。布尔值模型在力迫的元数学里更优雅，也是理解力迫「可计算性」的捷径。

## 1 力迫关系：把「$G$ 的判决」提前写进 $V$

在 $V$ 里递归定义关系 $p \Vdash \varphi$，其中 $\varphi$ 是关于 $\mathbb{P}$-名字（见下文）的一阶公式。核心条款：

$$
p \Vdash \sigma \in \tau \iff \{q \le p : \exists (\rho, r) \in \tau, \; q \le r \wedge q \Vdash \sigma = \rho\} \text{ 稠密在 } p \text{ 之下}
$$

$$
p \Vdash \sigma = \tau \iff \forall (\rho, r) \in \tau, \forall q \le p \cap r: \; q \Vdash \rho \in \sigma \quad \text{且对称地}
$$

直觉：$p$ 强迫「$\sigma \in \tau$」当且仅当「在 $p$ 的所有强化里，都能找到 $\tau$ 的元素 $\rho$ 使 $\sigma = \rho$」。整个定义是一阶的、在 $V$ 内部完成的——**$\Vdash$ 是 $V$ 里的可定义关系**，不需要碰 $G$。

**$\mathbb{P}$-名字（names）**：$V[G]$ 的元素在 $V$ 里都有「图纸」，叫名字。$\dot{x}$ 是 $G$ 的名字（$\dot{G} = \{(\check p, p) : p \in \mathbb{P}\}$，其中 $\check p$ 是 $p$ 的「典型名字」）。$V[G]$ 由「所有名字在 $G$ 下的赋值」构成：

$$
\mathrm{val}(\tau, G) = \{\mathrm{val}(\sigma, G) : \exists p \in G, \; (\sigma, p) \in \tau\}
$$

<span class="marginnote">名字是力迫的「蓝图层」：每个 $V[G]$ 元素都有一个（一般多个）名字，$G$ 决定最终装配哪张图纸。名字递归定义于 $V$，因此「$V[G]$ 里的集合」其实都是「$V$ 里可预言的组合」——只是组装权在 $G$ 手里。</span>

## 2 力迫定理：预言的兑现

**力迫定理（Forcing Theorem）**：设 $G$ 是 $V$ 上的通用滤，$\varphi$ 是带名字参数 $\tau_1,\dots,\tau_n$ 的一阶公式。则

$$
V[G] \vDash \varphi(\mathrm{val}(\tau_1,G), \dots, \mathrm{val}(\tau_n,G)) \iff \exists p \in G: \; p \Vdash \varphi(\tau_1,\dots,\tau_n)
$$

证明分两个方向，核心都是「真值通过稠密集翻译」：

- **（$\Leftarrow$）**：若 $p \in G$ 且 $p \Vdash \varphi$，则定义 $D = \{q : q \Vdash \varphi\}$。$D$ 是稠密的（由 $\Vdash$ 的定义，任何条件都可细化成「决定 $\varphi$」的条件），故 $G \cap D \neq \emptyset$；而 $G$ 里每个条件都强迫 $\varphi$，于是 $V[G] \vDash \varphi$。
- **（$\Rightarrow$）**：若 $V[G] \vDash \varphi$，则 $D = \{q : q \Vdash \varphi \text{ 或 } q \Vdash \lnot \varphi\}$ 稠密；$G$ 必碰 $D$ 中的某个「决定 $\varphi$ 真值」的条件。由 $V[G] \vDash \varphi$，该条件必是 $q \Vdash \varphi$（否则与真值矛盾）。<span class="marginnote">力迫定理就是「$V[G] \vDash$」与「$V$ 里的 $\Vdash$」之间的翻译机：它把「扩张里的真理」化约为「$V$ 里可检查的力迫关系」。因此「$V[G] \vDash \mathrm{ZFC}$」可逐条公理验证——每条公理的力迫版本都可由 $V$ 里的推理完成。</span>

**要点**：力迫定理是「$V[G] \vDash \mathrm{ZFC}$」的通行证——它把「扩张里公理成立」翻译成「$V$ 里每个公理被所有充分细化的条件强迫」，而后者是 $V$ 内可证的（且 ccc 保证不新增序数，从而基数不塌）。

## 3 布尔值模型：力迫的代数化

**布尔值模型（Boolean-valued model）$V^{\mathbb{B}}$**：固定一个完备布尔代数 $\mathbb{B}$，定义 $V^{\mathbb{B}}$ 为所有「以 $\mathbb{B}$ 为赋值域的名字」的类，每个命题 $\varphi$ 在 $V^{\mathbb{B}}$ 里取布尔值 $\llbracket \varphi \rrbracket \in \mathbb{B}$，递归定义：

$$
\llbracket \sigma \in \tau \rrbracket = \bigvee_{(\rho, r) \in \tau} (r \wedge \llbracket \sigma = \rho \rrbracket), \qquad
\llbracket \sigma = \tau \rrbracket = \bigwedge_{(\rho, r) \in \tau} (r \Rightarrow \llbracket \rho \in \sigma \rrbracket) \wedge \text{对称}
$$

于是每个命题有一个「真值度」而非简单的对/错。<span class="marginnote">$V^{\mathbb{B}}$ 的核心定理：$\llbracket \varphi \rrbracket = \mathbf{1}$（恒真）当且仅当 $V \vDash \mathrm{ZFC}$ 且 $\mathbb{B}$ 是完备布尔代数——即「在布尔值宇宙里 ZFC 恒真」。当 $\mathbb{B}$ 是「力迫偏序的完成」时，$V^{\mathbb{B}}$ 与「$V[G]$ 的全体（随 $G$ 变）」一一对应，布尔值模型是力迫的代数骨架。</span>

**布尔值模型的价值**：力迫常被「挑一个通用滤」模糊化——$G$ 的存在依赖「$V$ 里稠密集的可数多」等条件。布尔值模型则完全取消 $G$：$V^{\mathbb{B}}$ 里一切都是确定的布尔值，真值从不依赖选择。**（且）**力迫定理的布尔值版本：$V[G] \vDash \varphi \iff \llbracket \varphi \rrbracket$ 在 $G$ 的对应超滤里取值 $\mathbf{1}$——通用滤只是「把布尔值投影回 0/1」的坐标。

## 4 公式解析：$\Vdash$ 与 $\llbracket\cdot\rrbracket$ 如何互通

把力迫关系与布尔值连成一条换算链，拆开每一步：

$$
p \Vdash \varphi \iff p \le \llbracket \varphi \rrbracket
$$

- **$\llbracket \varphi \rrbracket$**：命题 $\varphi$ 的布尔真值——「$\varphi$ 为真的最大条件」。它是 $V^{\mathbb{B}}$ 里递归算出来的元素。
- **$p \le \llbracket \varphi \rrbracket$**：「条件 $p$ 被包含在 $\varphi$ 的真值里」——$p$ 是 $\varphi$ 真值的「一部分」。这是力迫关系的布尔翻译。
- **$\iff$**：两边定义等价——$p \Vdash \varphi$ 当且仅当「$p$ 所在的每个通用滤都让 $\varphi$ 真」，等价于「$\varphi$ 的布尔真值 $\ge p$」。力迫定理正是这条等式在 $V[G]$ 层面的兑现。
- **特例**：$V[G] \vDash \varphi \iff \exists p \in G, \; p \le \llbracket \varphi \rrbracket$——因为 $G$ 是超滤，$p \le \llbracket\varphi\rrbracket$ 与 $p \in G$ 合起来即「$\llbracket \varphi \rrbracket \in G$ 对应的超滤」。

**辨析｜易错点：** $\llbracket \varphi \rrbracket$ 是 $\mathbb{B}$ 的元素，不是「概率」——它不能相加、不满足可数可加，只是「真值所在的格点」。初学者易把布尔值模型误当成概率模型；实际上它对应的是「条件集合」，不是「测度」。

## 6 动手推导：名字与赋值的一步换算

把「$V[G]$ 里到底有哪些对象」用名字语言算一遍，理解「图纸 → 装配」的机制。

- **第一步，最简名字**：对 $V$ 里的集合 $x$，它的「典型名字」$\check x$ 递归定义：$\check x = \{(\check y, p) : y \in x, p \in \mathbb{P}\}$——「无论 $G$ 是什么，$\check x$ 都被装配成 $x$」。$\mathrm{val}(\check x, G) = x$ 对一切 $G$。
- **第二步，$G$ 的名字**：$\dot G = \{(\check p, p) : p \in \mathbb{P}\}$。赋值 $\mathrm{val}(\dot G, G) = \{p \in \mathbb{P} : \exists q \in G, \text{条件使 } \check p = \mathrm{val}(\check p, G)\} = G$——$\dot G$ 在 $G$ 下装配成 $G$ 本身。
- **第三步，新对象的名字**：Cohen 实数 $x_G$ 的名字是 $\dot x = \{(\check n, p) : n \in \mathrm{dom}(p), p(n) = 1\}$（加合适的序对编码）。$\mathrm{val}(\dot x, G) = \{n : \exists p \in G, p(n) = 1\}$——即「$G$ 里条件在 $n$ 处给 1 的那些 $n$」。
- **第四步，要点**：$V[G] = \{\mathrm{val}(\tau, G) : \tau \text{ 是名字}\}$——扩张里的每个元素都有一张名字图纸，$G$ 决定装配方式。**「$V$ 里预知的图纸 + 一个通用滤 = $V[G]$ 的一切」**，这就是力迫的哲学。

**辨析｜易错点：** 名字 $\tau$ 是 $V$ 里的集合，但 $\mathrm{val}(\tau, G)$ 通常不在 $V$ 里——名字是「图纸」，不是「成品」。初学者常把「$\tau \in V$」误读成「$\mathrm{val}(\tau,G) \in V$」；后者仅在 $\tau$ 是「$\check x$ 型」（不依赖 $G$）时成立。

### 更进一步：布尔值模型为何是力迫的「代数骨架」

布尔值模型 $V^{\mathbb{B}}$ 的价值，在于它把力迫从「挑一个通用滤」的偶然性里解放出来。$V[G]$ 依赖 $G$ 的选择，而 $V^{\mathbb{B}}$ 的每个命题有一个确定的布尔值 $\llbracket \varphi \rrbracket$——**不依赖任何滤**。两种视角的转换规则：

1. **给布尔值 → 还原为力迫**：对任意超滤 $H$（$\mathbb{B}$ 上的），定义 $V^{\mathbb{B}}/H$ 为「按 $H$ 投影布尔值」的商——每个 $\llbracket\varphi\rrbracket$ 投影成 0/1，得到普通模型。通用滤对应的正是「正则超滤」。
2. **给力迫 → 嵌入布尔值**：任意偏序 $\mathbb{P}$ 的「正则开域完备化」$\mathrm{RO}(\mathbb{P})$ 是完备布尔代数，$V[G]$ 与 $V^{\mathrm{RO}(\mathbb{P})}/H$ 同构。

**要点**：布尔值模型是「力迫的元理论」——它让「$V[G] \vDash \varphi$」变成「$\llbracket \varphi \rrbracket = \mathbf{1}$」这一纯粹的代数事实，摆脱了「通用滤存在」的前提。这使力迫定理的证明更干净，也使「力迫与 ZFC 一致」的元数学论证更直接——因为 $V^{\mathbb{B}}$ 本身不依赖任何外部对象。

### 补充：力迫定理的「两个方向」别记反

力迫定理经常被初学者记混方向。给一个记忆锚：

- **$p \Vdash \varphi$ 的意思是「预言」**：任何含 $p$ 的通用滤都让 $\varphi$ 真——这是从条件到扩张的「下行」。
- **$V[G] \vDash \varphi$ 的意思是「兑现」**：$\varphi$ 确实真——这是从扩张到条件的「上行」，断言「存在 $p \in G$ 已预言」。
- 力迫定理把两者缝合成等价：**下行可证 ⟺ 上行兑现**。

**辨析｜易错点：** 别把「$V[G] \vDash \varphi$ 当且仅当存在 $p \in G$，$p \Vdash \varphi$」错记成「当且仅当所有 $p \in G$」。通用滤里可能有「未预言 $\varphi$」的条件（它们预言了 $\lnot\varphi$ 的反面分支）——「存在一个预言者」才是力迫定理的准确形态。

## 9 小结

- **力迫关系 $\Vdash$**：$V$ 内可定义的关系，$p \Vdash \varphi$ 意为「含 $p$ 的通用滤都使 $\varphi$ 真」；递归条款定义 $\in$ 与 $=$。
- **$\mathbb{P}$-名字**：$V[G]$ 元素在 $V$ 里的图纸；$\mathrm{val}(\tau,G)$ 是 $G$ 下的装配。
- **力迫定理**：$V[G] \vDash \varphi \iff \exists p\in G, p\Vdash\varphi$；由此 $V[G] \vDash \mathrm{ZFC}$ 逐条可证。
- **布尔值模型 $V^{\mathbb{B}}$**：命题取布尔值 $\llbracket\varphi\rrbracket$；$\llbracket \varphi \rrbracket = \mathbf{1}$ 时在 $V^{\mathbb{B}}$ 中恒真。
- **互通**：$p \Vdash \varphi \iff p \le \llbracket \varphi \rrbracket$；通用滤把布尔值投影回 0/1。

在下一节，我们用力迫收割最著名的果实：连续统假设的独立性、选择公理的独立性，以及 Suslin 假设的独立性——Cohen 的 $\aleph_2$