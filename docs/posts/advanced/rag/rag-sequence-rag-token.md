---
title: RAG-Sequence 与 RAG-Token
date: 2026-08-11
---

# RAG-Sequence 与 RAG-Token

<div class="epigraph">
<p>整个旅程使用同一份地图，还是每一步都重新看地图？这取决于你要去哪里。</p>
<footer>—— 对两种边缘化策略的直觉（本文作者）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · RAG 与检索增强 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从两种生成方式开始

第 1 篇给出了 RAG 的心脏公式 $p(y \mid x) = \sum_z p(z\mid x) p(y \mid x, z)$，但留下一个悬而未决的问题：这个「对文档求和」到底发生在那一步？答案不同，模型的行为就完全不同。RAG 论文提出了两个具体实例——**RAG-Sequence** 与 **RAG-Token**。它们共享同一套检索器与生成器，唯一的差别是**求和的位置**：一个让整段回答共用同一个文档，另一个让每个词都独立选文档。这个看似微小的设计差异，是理解「检索增强如何与自回归生成耦合」的钥匙。

## 1 同一条公式的两个求和位置

回看 RAG 的核心目标：答案 $y$ 是生成器逐 token 产出的序列，而每个 token 的生成都可能受益于不同的文档。于是有两种极端策略：

- **RAG-Sequence（整句共用）**：先选定一个文档 $z$，整段回答都在这个文档条件下生成，最后对所有文档求和。适合「答案应出自单一来源」的任务。
- **RAG-Token（逐词换档）**：每个 token 生成时都可以换一个新的文档分布。适合「答案的每一部分可能来自不同来源」的任务。<span class="marginnote">直觉类比：RAG-Sequence 像选好一本书再读完整个章节；RAG-Token 像每读一个词都可能翻到另一本书的对应页。前者稳定连贯，后者更灵活、更耗算力。</span>

## 2 RAG-Sequence：整句共用一个文档

RAG-Sequence 的生成过程是：先用检索分布选文档，再在该文档条件下自回归生成整句。

$$
p_{\text{seq}}(y \mid x) = \sum_{z \in \text{top-}k} p_\eta(z \mid x) \cdot p_\theta(y \mid x, z)
$$

其中 $p_\theta(y \mid x, z) = \prod_t p_\theta(y_t \mid x, z, y_{<t})$。意思是：**给定查询和所选文档，整个序列按普通自回归方式生成**，求和只发生在整句层面。<span class="marginnote">实现细节：为了让「每个文档各生成一条完整答案再加权」的朴素做法不至于慢 $k$ 倍，论文采用近似——让 BART 的编码器只对拼接的文档输入编码一次，再在每个解码步对不同的文档表示计算不同的分布。</span>这样的好处是答案内部一致性有保障：整段话都「用同一个证据说话」。

## 3 RAG-Token：逐 token 选文档

RAG-Token 把求和挪到了每个 token 内部。每生成一个词，都把 Top-k 文档重新「考虑一遍」，加权后再采样下一个词：

$$
p_{\text{tok}}(y \mid x) = \prod_{t} \sum_{z \in \text{top-}k} p_\eta(z \mid x) \cdot p_\theta(y_t \mid x, z, y_{<t})
$$

对比 RAG-Sequence 的「整句一层求和」，这里**每一层 token 都有一层求和**。代价是每个解码步都要对 k 个文档各算一次分布，计算量乘以 k；收益是生成到中途可以「换证据」——比如回答第一句用文档 A 的事实，第二句转向文档 B 的事实。<span class="marginnote">直觉：多文档拼凑型答案（「这个项目由 A 发起、由 B 资助」）天然契合 RAG-Token；单一来源型答案（「这首诗的作者是谁」）则适合 RAG-Sequence。</span>

## 4 公式解析：把两个求和放到位

把两种策略并排写出来，差异一目了然——区别只在「求和号 $\sum_z$ 的位置」：

$$
p_{\text{seq}}(y \mid x) = \sum_{z} \left[ p(z\mid x) \cdot \prod_t p(y_t \mid x, z, y_{<t}) \right]
$$

$$
p_{\text{tok}}(y \mid x) = \prod_{t} \left[ \sum_{z} p(z\mid x) \cdot p(y_t \mid x, z, y_{<t}) \right]
$$

- **第一步，看括号的位置**：RAG-Sequence 的 $\sum_z$ 括住了整个序列乘积；RAG-Token 的 $\sum_z$ 只括住单个 token 的分布。位置决定「文档选择的粒度」。
- **第二步，看乘积的顺序**：RAG-Sequence 先选文档再生成整句（先 $\sum$ 后 $\prod$）；RAG-Token 逐词生成、逐词求和（先 $\prod$ 后 $\sum$）。
- **第三步，看语义**：RAG-Sequence 是一个「文档混合高斯」式生成；RAG-Token 是逐 token 的文档混合，允许序列内部切换证据源。<span class="marginnote">从概率角度看，前者假设整个序列 $y$ 由单一隐变量 $z$ 驱动，后者假设每个 $y_t$ 的隐变量独立。隐变量独立性假设不同，模型的表达力就不同。</span>

这个「求和与乘积谁在里、谁在外」的差别，正是第 1 篇边缘化公式的全部歧义所在——论文正是用这两个极端实例把歧义消解成可训练的模型。

## 5 实验结果：证据切换真的有用

RAG 论文在开放域问答、事实核查等任务上比较了两者，结论值得记住：**在绝大多数知识密集型任务上，RAG-Token 优于 RAG-Sequence**。例如在 Natural Questions 上，RAG-Token 的 EM 分数更高。原因正是开放域答案常常「拼接」多个证据：问题本身拆开看，每个部分可能指向不同段落。<span class="marginnote">但这不等于 RAG-Token 永远更好：它在每个解码步都要遍历 k 个文档，推理成本显著上升；当任务本身强调单一来源、单一证据时，RAG-Sequence 的稳定与便宜反而划算。</span>两者与「纯参数化生成」（不检索）相比都大幅领先，这构成了 RAG 论文的核心实验证据：**检索增强不是锦上添花，而是知识密集型任务上的必要机制**。

## 6 小结

- RAG-Sequence 与 RAG-Token 共享同一套检索器与生成器，差别仅在**对文档求和的粒度**：整句一次 vs 逐 token 一次。
- 公式层面是「**乘积与求和的嵌套顺序**」不同，背后的概率假设是「序列共享一个隐文档」还是「每个 token 独立选文档」。
- 实验上 **RAG-Token 在多数知识密集型任务上更好**，代价是逐 token 遍历 k 个文档的推理开销。
- 选型标准：**答案来源单一选 RAG-Sequence，多证据拼接选 RAG-Token**。

在下一节，我们突破「单次检索」的假设：当问题需要多步推理、多个证据链式衔接时，一次 Top-k 检索就不够了——这就是**检索增强多跳推理**。
