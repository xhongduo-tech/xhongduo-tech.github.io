---
title: MTP 训练目标与损失函数设计
date: 2026-08-07
---

# MTP 训练目标与损失函数设计

<div class="epigraph">
<p>定义一个好目标，胜过堆一百个技巧——损失函数就是模型眼中的世界。</p>
<footer>—— 作者自注，取材于 DeepSeek-V3 技术报告 MTP 章节的工程哲学</footer>
</div>

<div class="article-byline">
<p>第四级 · MTP 多 Token 预测 ｜ DeepSeek-AI, "DeepSeek-V3 Technical Report" §MTP ｜ 2026-08-07</p>
</div>

## 为什么从损失函数开始

上一节给出了 MTP 的抽象定义「一个输入预测 K 个未来 token」，但抽象定义到工业级实现之间隔着一层工程选择：**每个预测头怎么接？梯度怎么流？各路损失怎么加权？** DeepSeek-V3 的 MTP 设计是当前最完整的答案样本——它既是训练辅助目标，又是推理投机解码的载体。这一节只谈训练侧：**MTP 目标函数长什么样、损失如何加权、与主损失如何合并**。

## 1 训练目标的总框架：主损失 + MTP 辅助损失

DeepSeek-V3 的总体训练损失是**两项之和**：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{MTP}}
$$

- $\mathcal{L}_{\text{CE}}$ 是标准的下一个 token 交叉熵（主任务，保证语言建模能力不退步）。
- $\mathcal{L}_{\text{MTP}}$ 是 K 个 MTP 模块的额外预测损失（辅助任务，提供更丰富的远期监督）。
- $\lambda$ 是**MTP 损失权重**，DeepSeek-V3 取 $\lambda = 0.3$。

<span class="marginnote">为什么不是直接换成纯 MTP 目标？因为主任务与辅助任务是<strong>阶梯关系</strong>：主任务保证「能流畅生成」，MTP 任务负责「提前规划」。把 MTP 作为带权重的辅助损失，能让模型在<strong>不牺牲主线能力</strong>的前提下获得额外表征红利——这个「主+辅」的加法结构贯穿整节。</span>

**关键点：MTP 损失在训练时每个 token 位置都计算，且各模块损失平均后并入总损失，从而主干的梯度来自两路信号。**

## 2 MTP 模块内部的损失构造：递进头

DeepSeek-V3 的每个 MTP 模块 $i$ 预测的是 $x_{t+i}$（比当前位置远 $i$ 步的 token）。第 $i$ 个模块的输入不是原始隐藏状态，而是**「上层隐藏 + 前一个目标 token 的嵌入」的融合**：

$$
h_t^{(i)} = \text{MTP\_Module}^{(i)}\!\left(\text{RMSNorm}\!\left(h_t^{(i-1)}\right) + \text{RMSNorm}\!\left(\text{Emb}\!\left(x_{t+i-1}\right)\right)\right)
$$

**这个融合是 DeepSeek-V3 MTP 的精髓**：预测 $x_{t+i}$ 时，把「当前步的表示 $h_t^{(i-1)}$」与「上一步已见到的 token $x_{t+i-1}$ 的真实嵌入」相加，再喂给一个完整的 Transformer 块（自注意力 + FFN）。<span class="marginnote">对比 Gloeckle 式的并联头：DeepSeek 让每个头<strong>级联</strong>，因此第 2 个头能看到「第 1 个头正在预测的 token 的真实值」——训练时用真实 token，推理时用推测 token。级联带来更强的表达力，代价是多一层 Transformer 块的计算。</span>

第 $i$ 个模块的预测头是一个**线性输出层**（前接 RMSNorm）：

$$
p^{(i)} = \text{Softmax}\!\left(\text{Out\_Linear}\!\left(\text{RMSNorm}\!\left(h_t^{(i)}\right)\right)\right)
$$

每个模块有**独立**的嵌入矩阵与输出头，但**共享主干**的绝大部分参数——这就是「逐层深度集成」在损失层面的体现，详见第2篇《MTP 模块的逐层深度集成》。

## 3 损失聚合：对所有位置、所有模块平均

MTP 模块一共 K 个（DeepSeek-V3 取 $K=1$，即额外预测一个 token），每个模块损失为标准的交叉熵，按位置与模块双重平均：

$$
\mathcal{L}_{\text{MTP}} = \frac{1}{K} \sum_{i=1}^{K} \left[ -\frac{1}{T} \sum_{t=1}^{T} \log p^{(i)}_{t,\; x_{t+i}} \right]
$$

- **第一步，看内层**：对序列里每个位置 $t$，把模块 $i$ 预测到的正确 token $x_{t+i}$ 的概率取负对数——这是标准的交叉熵。
- **第二步，看外层**：对 K 个模块平均。每个模块的「难度」不同：$i=1$ 最容易（距离最近），$i$ 越大越难（远期不确定累积）。**平均而非求和**，保证 MTP 损失的量级与 K 无关，$\lambda$ 才能稳定调参。
- **第三步，看合并**：$K$ 个模块共享同一个 $\lambda$，但 DeepSeek 在实现上把各模块损失直接平均，避免了「K 越大总损失越大」的漂移。

<span class="marginnote">一个容易踩的坑：<strong>不要</strong>把 MTP 各模块损失直接求和再乘 $\lambda$——那样 K 从 1 增到 4 时，辅助损失量级翻 4 倍，主损失被稀释，需要重新调 $\lambda$。平均化让 $\lambda$ 与 K 解耦，是「可扩展的损失设计」。</span>

## 4 公式解析：为什么用真实 token 嵌入而非预测 token

训练时第 $i$ 个模块吃的是**上一步真实 token 的嵌入** $\text{Emb}(x_{t+i-1})$。这条设计的收益可以拆三步：

- **第一步，信息最大化**：真实 token 是最强的条件信号。预测「远未来」时，知道「紧邻未来」的真实值，模型可以**分步推理**——先想紧邻，再想远邻。
- **第二步，与推理对齐的代价**：推理阶段真实值不存在，只能把前一个模块的**预测**喂进来（投机解码时的近似）。这就是 **teacher forcing 与 inference mismatch** 的经典张力：训练用真实值更高效，推理用预测值有累积误差。
- **第三步，工程折中**：DeepSeek-V3 之所以敢用这个结构，是因为它同时把 MTP 模块在推理时复用作**草稿模型**——训练时的 teacher forcing 让 MTP 模块学会「吃一个 token，吐一个 token」，正好满足投机解码的接口需求。**训练目标的每一步设计，都在为推理复用铺路。**

## 5 小结

- 总体损失 = **主 CE 损失 + $\lambda \cdot \mathcal{L}_{\text{MTP}}$**，DeepSeek-V3 取 $\lambda = 0.3$。
- 每个 MTP 模块预测**远 $i$ 步**的 token，输入为**上层隐藏 + 上一步真实 token 嵌入**的融合。
- 模块内部是**完整的 Transformer 块 + RMSNorm + 线性输出头**，嵌入矩阵与输出头独立。
- MTP 损失**按模块与位置双重平均**，使 $\lambda$