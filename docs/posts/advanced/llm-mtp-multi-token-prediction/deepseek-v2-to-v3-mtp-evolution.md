---
title: 从 DeepSeek-V2 到 V3 的 MTP 演进
date: 2026-08-07
---

# 从 DeepSeek-V2 到 V3 的 MTP 演进

<div class="epigraph">
<p>效率不是一次革命，而是一连串正确的叠加。</p>
<footer>—— 作者自注，概括 DeepSeek 系列架构的演进哲学</footer>
</div>

<div class="article-byline">
<p>第四级 · MTP 多 Token 预测 ｜ DeepSeek-AI, "DeepSeek-V2/V3 Technical Reports" ｜ 2026-08-07</p>
</div>

## 为什么从版本演进切入

DeepSeek-V3 的 MTP 不是凭空出现的：它坐落在 V2 打下的地基上。**V2 解决「单步推理怎么变快」——靠 MLA 压缩 KV Cache、靠 MoE 稀疏激活省计算；V3 才把 MTP 加进来，解决「训练表征怎么更聪明、解码怎么更进一步」**。把这条演进线梳理清楚，你就能看到：为什么 MTP 恰好出现在 V3，而不是更早——因为它的推理红利依赖前两样架构的铺垫。

## 1 地基：DeepSeek-V2 的 MLA 与 MoE

DeepSeek-V2 的核心是两个效率创新，它们与 MTP 的关系是「铺垫」而非「包含」：

- **MLA（Multi-head Latent Attention，多头潜在注意力）**：把 KV Cache 压缩到低维潜在空间，使 KV 显存占用降低约一个数量级。<span class="marginnote">MLA 让推理的访存瓶颈从「KV Cache」一侧松绑——长上下文不再线性吞噬显存带宽。这为 V3 的投机解码腾出了「可以快速读 KV」的硬件余量，MTP 草稿模型才跑得动。</span>
- **DeepSeekMoE**：细粒度专家 + 共享专家，每个 token 只激活一小部分参数，把训练与推理的 FLOPs 大幅压低。

**V2 的范式依然是标准下一 token 预测**——它在技术报告里只字未提多步预测。V2 的贡献集中在「每一步怎么算得更省」，而不是「一次算几步」。这一点很重要：**MTP 的「多步」视角是 V3 才引入的新维度。**

## 2 转折：V3 把「多步监督」写进训练目标

DeepSeek-V3 的技术报告在训练目标章节首次加入 MTP：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{MTP}}, \qquad \lambda = 0.3$$

V3 的 MTP 选择了一个保守而务实的配置——**深度 K=1，即只额外预测 1 个 token**（一个 MTP 模块）。这个「只多一步」的选择值得咀嚼：<span class="marginnote">Gloeckle 在中小模型上把 K 推到 4 以换取代码与数学增益；DeepSeek-V3 是 671B 的 MoE 巨模型，训练成本极其昂贵，K=1 既避免了远期预测的梯度噪声，又保留了 MTP 的两大红利（训练辅助 + 推理复用）——<strong>模型规模不同，最优 K 不同</strong>。</span>

与 Gloeckle 方案的另一个分野是：V3 的 MTP 模块是**级联的厚模块**（完整 Transformer 块，吃「上层隐藏 + 上一 token 真实嵌入」），而非并联的薄头。这让 MTP 模块在训练后**天然具备「吃一个 token 猜一个 token」的自回归能力**——这正是推理复用所要求的接口。

## 3 落地：MTP 从「辅助目标」到「推理加速器」

V3 对 MTP 的最大贡献，是把 Gloeckle 式「训练完就丢掉的辅助头」变成了**推理期可复用的草稿模型**：

**训练阶段**：MTP 模块与主干联合训练，损失权重 $\lambda=0.3$。它的表征红利已固化进主干。
**推理阶段**：不再丢弃 MTP 头，而是让 MTP 模块作为**草稿模型（draft model）**，一次推测下一 token，主干用单次前向验证并接受或拒绝——这就是投机解码，DeepSeek 用它免去了**训练一个独立小模型当草稿**的额外成本。
- **收益**：官方技术报告与公开评测普遍显示，MTP 投机解码显著提升了生成吞吐，常被引用的加速数据在约 1.8× 量级。<span class="marginnote">「草稿模型不用额外训练」是 V3 这条路的杀手锏：标准投机解码需要一个比主模型小得多、又足够强的草稿模型，训练它本身就是一笔开销。MTP 模块共享主干训练、随主干白得，边际成本趋近于零。</span>

从 V2 到 V3，一条清晰的能力叠加线浮现出来：**MLA 省显存 → MoE 省计算 → MTP 省「草稿训练」并把训练红利二次变现为推理加速。**

## 4 公式解析：为什么 MTP 模块能直接当草稿模型

MTP 模块的自回归能力来自它的输入构造，回顾第1篇《MTP 训练目标与损失函数设计》的融合公式：

$$
h_t^{(1)} = \text{MTP\_Module}^{(1)}\!\left(\text{RMSNorm}(h_t) + \text{RMSNorm}\!\left(\text{Emb}(x_{t})\right)\right)
$$

- **第一步，看输入**：模块吃「主干当前表示 $h_t$ + 当前 token $x_t$ 的嵌入」。
- **第二步，看输出**：$h_t^{(1)}$ 经输出头给出 $p(x_{t+1})$——**它天生就是「给定 $x_t$ 预测 $x_{t+1}$」的函数**。
- **第三步，看复用**：推理时，主干算出 $h_t$ 后，让 MTP 模块扮演「下一步预测器」；主干的最后线性层扮演「验证器」。**两件套都是训练时已有的组件，无需任何新参数。**
- **第四步，看关键假设**：这一复用成立的前提是训练时 MTP 模块确实学会了「输入 token → 预测下一 token」的映射，而这正是 teacher forcing 的融合输入训练出来的。

## 5 演进对照表

| 维度 | DeepSeek-V2 | DeepSeek-V3 |
| --- | --- | --- |
| 预测范式 | 标准下一 token | 下一 token + MTP（K=1） |
| 推理效率来源 | MLA + MoE | MLA + MoE + MTP 投机解码 |
| MTP 作用 | 无 | 训练辅助目标（λ=0.3）+ 推理草稿模型 |
| 草稿模型 | 无 | MTP 模块复用，免额外训练 |
| 加速逻辑 | 每步算得更省 | 一步验证多个 token |

## 6 小结

- DeepSeek-V2 是**效率地基**：MLA 压缩 KV Cache、MoE 稀疏激活，但仍是单 Token 范式。
- DeepSeek-V3 **首次引入 MTP**：$K=1$、$\lambda=0.3$