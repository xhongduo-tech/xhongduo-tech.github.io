---
title: Kan 扩张
date: 2026-08-11
---

# Kan 扩张

<div class="epigraph">
<p>一切概念都是 Kan 扩张。</p>
<footer>—— 桑德斯 · 麦克莱恩（Saunders Mac Lane）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 范畴论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Kan 扩张开始

从上节 Yoneda 引理我们知道「对象由关系决定」，也知道「每个预层都由可表预层拼成」。现在把这个问题推进到函子层面：**给定一个函子 $F: \mathbf{C} \to \mathbf{E}$ 与一个「换坐标系」的函子 $K: \mathbf{C} \to \mathbf{D}$，能否把 $F$ 沿着 $K$「扩张」成一个定义在 $\mathbf{D}$ 上的函子 $\mathrm{Lan}_K F$？** 这就是**Kan 扩张（Kan extension）**。Mac Lane 那句著名的「一切概念都是 Kan 扩张」在字面上并非夸张：极限、余极限、伴随、Yoneda 嵌入、层的张量积，全都可以统一写成 Kan 扩张的形式。<span class="marginnote">对机器学习读者，Kan 扩张像极了一次「跨任务迁移」：在源任务（$\mathbf{C}$）上训练好的模型 $F$，沿着「任务之间关系」$K$ 推到目标任务（$\mathbf{D}$）上去。</span>

## 1 Kan 扩张的定义

设 $K: \mathbf{C} \to \mathbf{D}$、$F: \mathbf{C} \to \mathbf{E}$。**左 Kan 扩张（left Kan extension）** $\mathrm{Lan}_K F: \mathbf{D} \to \mathbf{E}$ 是「从 $F$ 出发、经 $K$ 到达的最泛函子」：它配有自然变换 $\eta: F \Rightarrow \mathrm{Lan}_K F \circ K$，且对任意函子 $G: \mathbf{D} \to \mathbf{E}$ 与任意自然变换 $\alpha: F \Rightarrow G K$，存在**唯一**自然变换 $\beta: \mathrm{Lan}_K F \Rightarrow G$ 使下图交换：

$$\alpha = \beta K \circ \eta$$

把方向都反过来（自然变换从 $G K \Rightarrow F$），得到**右 Kan 扩张（right Kan extension）** $\mathrm{Ran}_K F$。<span class="marginnote">直观上，$\mathrm{Lan}_K F$ 回答「如果只知道 $F$ 在 $\mathbf{C}$ 上的取值，那么它在整个 $\mathbf{D}$ 上『最合理』的取值是什么」——剩下的所有自由度由普适性质约束掉。</span>当 $K$ 是 $\mathbf{C}$ 的某种嵌入时，Kan 扩张就是「在保证 $\mathbf{C}$ 上取值不变的前提下做最经济的外推」。

## 2 实例：极限、伴随都是 Kan 扩张

**余极限是 Kan 扩张**：把常值函子 $\Delta$ 沿 $F$ 扩张。具体地，对图 $F: \mathbf{J} \to \mathbf{C}$，余极限 $\mathrm{colim}\, F = \mathrm{Lan}_F(1_{\mathbf{J}})(\ast)$——在单点范畴上取值，即「沿 $F$ 的右 Kan 扩张作用于唯一对象」。
- **极限同理**：$\lim F = \mathrm{Ran}_F(1_{\mathbf{J}})(\ast)$。
- **伴随是 Kan 扩张**：$F \dashv G$ 当且仅当 $\mathrm{Lan}_G(1) = F$ 且 $\mathrm{Ran}_G(1) = F$（恰当地），或等价地 $F$ 是 $G$ 沿恒等函子的 Kan 扩张。**伴随就是「沿恒等的 Kan 扩张」**——Mac Lane 名言的又一注脚。<span class="marginnote">这样看来，前六节的全部内容——极限、余极限、伴随、自由/遗忘——都被 Kan 扩张一句话重写。这也是为什么 Riehl 把 Kan 扩张放在「一切皆是……」的位置。</span>

## 3 公式解析：逐点公式

当 $\mathbf{E}$ 有余极限时，左 Kan 扩张可以「逐点」地算出来：

$$
(\mathrm{Lan}_K F)(d) = \mathrm{colim}_{(c, f) \in (K \downarrow d)} F c
$$

其中 $K \downarrow d$ 是**逗号范畴（comma category）**：对象是「$K c \xrightarrow{f} d$」的配对 $(c, f)$。右 Kan 扩张对称地为对偶的极限：

$$
(\mathrm{Ran}_K F)(d) = \lim_{(d \xrightarrow{f} K c)} F c
$$

- **第一步，确定指标**：对固定的 $d$，逗号范畴收集「所有从 $K c$ 到 $d$ 的态射 $f$」——每个 $f$ 给出一份「数据」$F c$。
- **第二步，怎么拼**：$\mathrm{Lan}_K F$ 在 $d$ 处的值，是所有这份数据在 $F$ 下像的**余极限**——「把每条通往 $d$ 的路径的贡献全部粘起来」。
- **第三步，方向**：左用余极限（拼）、右用极限（取交集式约束）——与「左保余极限、右保极限」完全同向，可互相印证。
- **第四步，直觉**：迁移学习里 $d$ 是目标任务中的一个点，$K c \to d$ 是「源任务点 $c$ 与 $d$ 的关系」，$(\mathrm{Lan}_K F)(d)$ 把所有这些关系下的 $F c$ 取并——**相似源点的输出拼成目标点的输出**。

**辨析｜易错点：** 逐点公式需要 $\mathbf{E}$ 有余/极限，否则 Kan 扩张可能不存在或算不出。另注意逗号范畴的方向：左扩张用 $K \downarrow d$（$K c \to d$），右扩张用 $d \downarrow K$（$d \to K c$）——方向搞反会得到错误构造。

## 4 密集性、神经与右 Kan 扩张

- **密度定理（density theorem）**：Yoneda 嵌入 $y: \mathbf{C} \to \mathbf{Set}^{\mathbf{C}^{\mathrm{op}}}$ 的 Kan 扩张给出「每个预层 = 可表预层的余极限」，即 $\mathrm{Lan}_y(y) \cong 1$。这是「关系决定本质」在函子层面的终极形态。<span class="marginnote">层论中「层化」本身也是一个 Kan 扩张；把预层放在「稠密嵌入」下游，Kan 扩张自动完成「局部一致性 → 全局拼合」。</span>
**神经与实现（nerve–realization）**：几何实现 $\lvert - \rvert: \mathbf{Set}^{\Delta^{\mathrm{op}}} \to \mathbf{Top}$ 是「标准单形嵌入」$\Delta \hookrightarrow \mathbf{Top}$ 的左 Kan 扩张；单形神经是它的右伴随。拓扑与组合之间的这座桥，纯靠 Kan 扩张搭起。
- **程序语言**：自由幺半群、列表函子、`fold` 的普适性，都能看成某个 Kan 扩张的存在性。<span class="marginnote">对做 LLM 的读者，词表嵌入训练出的「关系矩阵」若沿「子词→词」的函子做右 Kan 扩张，就得到一种无需重训的词汇外推——这是 Kan 扩张思想在表示学习中的直接应用。</span>

## 5 小结

- **左 Kan 扩张** $\mathrm{Lan}_K F$：沿 $K$ 外推 $F$ 的最泛函子，满足普适性质 $\alpha = \beta K \circ \eta$；**右 Kan 扩张**为其对偶。
- **一切概念都是 Kan 扩张**：极限、余极限、伴随、Yoneda 嵌入、密度定理、神经—实现全部统一。
- **逐点公式**：$(\mathrm{Lan}_K F)(d) = \mathrm{colim}_{K c \to d} F c$，$(\mathrm{Ran}_K F)(d) = \lim_{d \to K c} F c$。
- 逗号范畴与方向是关键：左扩张沿 $K \downarrow d$ 取余极限。
- 密度定理：每个预层都是可表预层的余极限。

在最后一节，我们把「沿嵌入外推」这套思想放到拓扑学与逻辑学的交汇处：**拓扑斯与层**——用范畴的语言让「局部一致性推全局」变得精确而可计算。
