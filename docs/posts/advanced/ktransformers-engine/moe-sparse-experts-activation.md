---
title: MoE 稀疏专家架构与激活原理
date: 2026-08-07
---

# MoE 稀疏专家架构与激活原理

<div class="epigraph">
<p>人多的路好走，专精的人走得远。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第四级 · ktransformers（消费级 MoE 推理引擎） ｜ Hongtao Chen et al., "KTransformers" SOSP 2025 §1 ｜ 2026-08-07</p>
</div>

## 为什么从稀疏专家讲起

上一站我们在《大模型原理》与《MoE 混合专家架构》里认识了**稀疏专家（sparse experts）**：模型不再是「一个巨大的前馈网络」，而是几百个互相竞争的小网络。今天要写的 ktransformers，全部设计都建立在一个事实上——**超大 MoE 模型虽然参数多，但每个 token 只激活其中一小部分**。这正是「消费级显卡也能跑满血 DeepSeek-V3」的第一性前提。<span class="marginnote">参数总量与激活量是两个完全不同的数字。ktransformers 的整套思路可以浓缩成一句话：<strong>把「永远用不到的那部分权重」藏进便宜的大内存里。</strong></span>不把这一层算清楚，后面所有「为什么 CPU 能算」「为什么能省显存」都无从谈起。

## 1 从稠密到稀疏：MoE 的设计动机

经典 Transformer 的每一个 Transformer 块里，前馈网络（FFN）是**稠密（dense）**的：所有参数对每个 token 都要算一遍。参数量一大，推理开销就与总参数量线性挂钩——而训练一个几万亿参数的全稠密模型，算力成本几乎是天文数字。

**混合专家（Mixture of Experts，MoE）**把这条路改道：不训练一个巨型稠密网络，而是训练**一组专家（experts）**，每个专家是一个较小的 FFN；外加一个**路由网络（router / gating）**，由它来决定每个 token 该交给哪几个专家处理。<span class="marginnote">MoE 的「条件计算」（conditional computation）思想可追溯到 1991 年 Jacobs 等人的工作；GShard、Switch Transformer 让它在现代大模型里复活。这一系谱在第四级《MoE 混合专家架构》里有完整梳理。</span>

**条件计算**是理解 MoE 的关键词：网络的计算路径**依赖于输入**。不同的 token 走不同的专家，「不相关的参数就不计算」——这就是「稀疏」二字的来历。模型变大了，但单次推理只触碰参数总集的一小片，计算量被「解耦」出参数量之外。

「解耦」是这里最值得停留的词：**在稠密模型里，参数量与计算量严格绑定**——70B 模型每个 token 就要算 70B 参数；而 MoE 打破了这条绑定——671B 模型每 token 只算 37B。这个「打破」意味着：**你可以把模型做得很大（能力更强），而不必付出同等的计算代价（推理更便宜）**。这是 MoE 在「规模」与「成本」之间解出的一个漂亮折中，也是它成为当代超大模型主流架构的根本原因。

## 2 DeepSeekMoE 的组成：共享专家 + 路由专家

ktransformers 的看家模型 DeepSeek-V3 用的正是 DeepSeekMoE 结构。**每个 MoE 层由两部分专家组成**：

**共享专家（shared expert）**：每个 token 必算，负责「几乎所有 token 都需要的通用知识」，提供稳定的基础能力。DeepSeek-V3 每层有 1 个共享专家。

**路由专家（routed experts）**：数量庞大、各司其职，负责「按需分发的专项知识」。DeepSeek-V3 每层有 **256 个路由专家**，而每个 token 只在其中选 1 个来算（top-1 路由）。

于是每层每 token 实际计算的 FFN 专家只有 **1 个共享专家 + 1 个路由专家 = 2 个**。路由选择靠一个轻量门控网络：

对比「2/257」这个比例（256 路由专家里选 1，加 1 共享，共 257 个专家里激活 2 个），稀疏度达到约 99.2%——**每层每 token 有 255 个专家完全不被计算**。这个「99.2% 不计算」就是一切性能优化的空间：不需要的权重，可以安心地放到慢速存储里吃灰。MoE 的稀疏性，本质上是一张「可安全忽略」的清单。

$$
\text{expert}_t = \arg\max_{e \in \{1,\dots,256\}} \big(\text{softmax}(W_g\, x_t)\big)_e
$$

这行式子说的是：把 token 的表示 $x_t$ 经过门控投影 $W_g$ 打分，取分数最高的那一个专家。<span class="marginnote">DeepSeek-V3 的 top-1 比 V2 的 top-8 更极端——稀疏到极致，也给「只把少数专家放上 GPU」的异构方案腾出了空间。激活粒度越细，ktransformers 越游刃有余。</span>

## 3 激活的稀疏性：671B 与 37B 的落差

DeepSeek-V3 的总参数量是 **671B**，但每个 token 实际激活的参数只有约 **37B**。这个落差从哪来？把 61 层 Transformer 拆开看：

- **Attention 部分**（含 MLA 的投影）与**共享专家**：稠密，每 token 必算。
- **256 个路由专家**：稀疏，每 token 只碰 1 个。

671B 参数里，路由专家占了约 **500B+**，也就是七成半以上的权重是「靠运气才被摸到」的。**激活率（activation ratio）**——激活参数量与总参数量的比值——决定了稀疏性的强度：

$$
\text{activation ratio} = \frac{\text{37B 激活参数}}{\text{671B 总参数}} \approx 5.5\%
$$

平均下来每个参数只有约 5.5% 的 token 会用到它。

5.5% 这个数字也给出了「异构可行」的边界：若激活率高达 50%（每 token 要碰一半参数），那么「冷专家」几乎不存在，卸载的收益会大幅缩水。**MoE 的稀疏程度，直接决定了异构推理的收益上限**——激活率越低，能安全放慢速存储的权重越多，CPU/DRAM 的「接单量」越大。这也是为什么 ktransformers 最适配的是 DeepSeek-V3 这种「高稀疏」模型（每 token 只激活 37B/671B），而不是激活率偏高的中等 MoE。<span class="marginnote">5.5% 是「平均」，真实的分布远非均匀——少数「热专家」被高频命中、多数「冷专家」几乎无人问津，这正是下一篇《专家激活幂律分布》的主角。那个分布才是 ktransformers 一切放置策略的根。</span>

## 4 公式解析：稀疏激活比例与推理成本

把「稀疏性」翻译成可比较的数字，是理解异构推理的第一步。设模型总参数为 $P$，单 token 激活参数为 $P_{\text{act}}$，则**单 token 的计算量（FLOPs）**约正比于 $P_{\text{act}}$，而**权重存储量**正比于 $P$。两者之比即激活率：

$$
\text{FLOPs per token} \approx 2 \times P_{\text{act}}, \qquad \text{weights} \approx P
$$

对 671B 模型，逐项拆解：

- **第一步，激活**：$P_{\text{act}} \approx 37\text{B}$，所以推理时真正要做的矩阵乘，规模只与 37B 相当。
- **第二步，存储**：$P \approx 671\text{B}$