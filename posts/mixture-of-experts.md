## 混合专家模型（Mixture of Experts）的核心思想

**混合专家模型（Mixture of Experts，MoE）**是一种稀疏条件计算（Sparse Conditional Computation）架构：模型拥有大量参数，但每次推理只激活其中一小部分。

**密集模型（Dense Model）**的推理方式是每个 token 经过全部参数。一个 1750 亿参数的密集模型，每生成一个 token 就需要全部 1750 亿参数参与计算，推理成本与参数量成正比。

MoE 打破了这个约束：用参数空间换计算时间，让模型在保持大参数容量的同时，将单次推理的计算量控制在较低水平。

---

## MoE 在 Transformer 中的位置

标准 Transformer 由注意力层（Attention）和前馈网络层（FFN）交替构成：

```
[Attention] → [FFN] → [Attention] → [FFN] → ...
```

MoE 将其中部分或全部 FFN 层替换为 MoE 层：

```
[Attention] → [MoE Layer] → [Attention] → [MoE Layer] → ...
```

每个 MoE 层的内部结构：

```
                      ┌── Expert 1 (FFN) ──┐
输入 token x → Router ─┼── Expert 2 (FFN) ──┼─→ 加权求和 → 输出
                      └── ... Expert N ────┘
                        (只激活 Top-K 个)
```

MoE 层包含两个核心组件：

- **专家（Experts）**：$N$ 个并行的 FFN 子网络，各自处理不同类型的输入
- **路由器（Router / Gating Network）**：一个轻量级线性层，为每个 token 打分并选出最合适的 $K$ 个专家

Attention 层不做 MoE 处理，所有 token 共享同一套 Attention 参数，只有 FFN 层被稀疏化。

---

## 路由机制的数学细节

以下推演设 **4 个专家（$N=4$）、每次激活前 2 名（$K=2$）**。

### 第一步：带噪声的打分

路由器包含一个可学习的打分矩阵 $W_g$。每个 token 输入后与 $W_g$ 相乘，得到 $N$ 个专家的基础分。为防止路由崩溃（Router Collapse），在基础分上叠加随机噪声：

$$H(x)_i = (x \cdot W_g)_i + \varepsilon_i \cdot \text{Softplus}\!\left((x \cdot W_{noise})_i\right), \qquad \varepsilon_i \sim \mathcal{N}(0,1)$$

| 项 | 含义 |
|---|---|
| $x \cdot W_g$ | 基础分，路由器通过训练学习的偏好 |
| $\varepsilon_i \sim \mathcal{N}(0,1)$ | 每次随机采样的正态噪声 |
| $\text{Softplus}(\cdot)$ | 控制噪声幅度的可学习缩放器（$\text{Softplus}(z) = \ln(1+e^z) > 0$） |

噪声的作用：训练早期，若某个专家偶然获得更多梯度更新而变强，路由器会越来越偏爱它，触发马太效应式正反馈，最终大部分专家几乎不参与计算。这就是**路由崩溃（Router Collapse）**。噪声打破了这种固化。

推演示例——加噪声后的得分 $H(x)$：

```
专家 1：8.0 分
专家 2：2.0 分
专家 3：1.0 分
专家 4：7.0 分
→ 分数数组：[8.0, 2.0, 1.0, 7.0]
```

### 第二步：Top-K 筛选

$K=2$，只保留前 2 名，落选者分数置为 $-\infty$：

$$\text{KeepTopK}(v,k)_i = \begin{cases} v_i & \text{if } v_i \text{ is in top-}k \text{ of } v \\ -\infty & \text{otherwise} \end{cases}$$

专家 1（8.0）和专家 4（7.0）胜出，数组变为：

```
[8.0, -∞, -∞, 7.0]
```

### 第三步：Softmax 转换为权重

将分数转换为和为 1 的权重分布：

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), K))$$

$e^{-\infty} = 0$，被淘汰专家的权重自动归零。

$$\text{专家 1 权重} = \frac{e^{8.0}}{e^{8.0} + e^{7.0}} \approx 0.73$$

$$\text{专家 4 权重} = \frac{e^{7.0}}{e^{8.0} + e^{7.0}} \approx 0.27$$

最终权重向量：$G(x) = [0.73,\ 0,\ 0,\ 0.27]$

### MoE 层的完整输出

$$\text{MoE}(x) = \sum_{i=1}^{N} G_i(x) \cdot E_i(x)$$

代入数字：

$$\text{MoE}(x) = 0.73 \cdot E_1(x) + 0 \cdot E_2(x) + 0 \cdot E_3(x) + 0.27 \cdot E_4(x)$$

$G_2 = G_3 = 0$ 意味着专家 2 和专家 3 无需运算——乘数为零，直接跳过，零 FLOP 消耗。

**稀疏激活的数学本质：算力因结构而省，而非靠近似或剪枝。** 模型拥有 $N$ 个专家的知识容量，每次推理只花 $K$ 个专家的计算成本。

---

## 负载均衡：对抗路由崩溃

### 辅助损失（Auxiliary Loss）

Shazeer et al.（2017）引入辅助损失强制负载均衡：

$$\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

| 符号 | 含义 |
|---|---|
| $f_i$ | 专家 $i$ 实际处理的 token 比例（不可微） |
| $P_i$ | 路由器对专家 $i$ 的平均 softmax 概率（可微） |
| $\alpha$ | 均衡强度超参数（通常 $10^{-2}$ 到 $10^{-3}$） |

当 $f_i$ 和 $P_i$ 都趋近 $1/N$（均匀分布）时，$\sum f_i \cdot P_i$ 取最小值。该损失提供梯度信号，鼓励路由器均匀使用所有专家。

总训练损失：

$$\mathcal{L} = \mathcal{L}_{main} + \mathcal{L}_{aux}$$

### 容量因子（Capacity Factor）

每个专家设置 token 处理上限：

$$C = \left\lfloor \frac{T \cdot k}{N} \right\rfloor \times \text{capacity\_factor}$$

其中 $T$ 为 batch 内 token 总数，$k$ 为每个 token 激活的专家数。capacity_factor 通常设为 1.0 至 1.5。超出上限的溢出 token 跳过该专家，直接通过残差连接传递。

---

## 关键论文演进

### 1991 — MoE 概念提出

Jacobs et al. 提出 MoE 框架，用于让不同子模型各自负责数据空间的不同区域。这一前深度学习时代的理论工作奠定了核心思想：**根据输入内容选择子模型**。

### 2017 — 稀疏门控 MoE 的大规模验证

Shazeer et al.（Google Brain）发表 [*Outrageously Large Neural Networks*](https://arxiv.org/abs/1701.06538)，将 MoE 层嵌入 LSTM 语言模型，实现约 137B 参数的语言模型。

核心贡献是 **Noisy Top-K Gating**——上文推演的"加噪声 → Top-K 筛选 → Softmax"三步机制：

$$H(x)_i = (x \cdot W_g)_i + \varepsilon_i \cdot \text{Softplus}\!\left((x \cdot W_{noise})_i\right), \qquad \varepsilon_i \sim \mathcal{N}(0, 1)$$

噪声在训练初期提供探索性，防止路由器过早固化，奠定了后续所有 MoE 训练的基础范式。

### 2020 — GShard：首个千亿级 MoE

Lepikhin et al.（Google）将 MoE 应用于神经机器翻译，实现 **600B 参数**模型（激活约 48B）。

GShard 的工程贡献：

- **专家并行（Expert Parallelism）**：不同专家放在不同 TPU 核心
- **Local Group Dispatching**：将 batch 内 token 分组，确保每组内跨专家负载均衡
- **随机路由（Random Routing）**：首选专家满载时，以一定概率随机选第二候选专家

### 2021 — Switch Transformer：Top-1 路由的验证

Fedus et al.（Google）提出将 Top-K 中 $K$ 设为 1——每个 token 只路由到一个专家。

简化带来的收益：

- 路由计算和反向传播更简单（梯度只流向一个专家）
- All-to-All 通信量减半
- 训练更稳定

Switch Transformer 达到 **1.6T 总参数**，在等计算预算下比 T5-XXL 快 7 倍。

| 模型 | 总参数 | 激活参数/层 | 专家数 | K |
|---|---|---|---|---|
| T5-XXL | 11B | 11B（全部）| — | — |
| Switch-C | 1571B | ~7B | 2048 | 1 |

### 2023 — Mixtral 8x7B：开源 MoE 标杆

Mistral AI 发布第一个被广泛使用的开源 MoE 模型。

架构参数：

- 8 个专家，每次激活 Top-2
- 总参数约 46.7B，激活参数约 **12.9B**
- 在多个基准上超越 LLaMA 2 70B

Mixtral 量化了 MoE 的实际收益：**以约 1/3 的激活算力，获得接近 2 倍总参数量密集模型的性能**。

### 2024 — DeepSeek-MoE：细粒度专家分割

DeepSeek 团队提出两项关键创新。

**细粒度专家分割（Fine-Grained Expert Segmentation）**

传统 MoE 中每个专家是一个完整的 FFN（隐层维度为 $d_{ff}$）。DeepSeek-MoE 将每个专家的隐层维度缩小为 $d_{ff}/m$，同时把专家数增加到 $mN$，激活 $mk$ 个专家。总 FLOP 不变，但路由器的组合粒度更细——$mk$ 个小专家的任意组合远比 $k$ 个大专家的组合丰富，减少了不同 token 被迫共用同一专家的知识冗余。

**共享专家隔离（Shared Expert Isolation）**

在路由专家之外，另设 $K_s$ 个**始终激活的共享专家**，处理各 token 共同需要的通用知识（基础语法、通用语义等）。路由专家则专注于特化知识。

$$\text{DeepSeekMoE}(x) = \underbrace{\sum_{i=1}^{K_s} E_i^{shared}(x)}_{\text{共享专家（始终激活）}} + \underbrace{\sum_{j \in \text{TopK}} G_j(x) \cdot E_j^{routed}(x)}_{\text{路由专家（稀疏激活）}}$$

这解决了传统 MoE 的一个深层问题：路由专家被迫同时存储特化知识和通用知识，形成冗余。共享专家卸载通用知识后，路由专家的表达能力更纯粹。

DeepSeek-V3（2024 年末）将此架构扩展到 **671B 总参数、激活仅 37B**，在多个任务上达到或超越 GPT-4o 的水平。

---

## 工程挑战：专家并行与通信瓶颈

### 密集模型的标准并行策略

- **张量并行（Tensor Parallelism）**：将单个矩阵按行/列切割到多 GPU，每张 GPU 持有矩阵的一个切片
- **流水线并行（Pipeline Parallelism）**：不同 Transformer 层分配到不同 GPU 组，形成计算流水线

### 专家并行（Expert Parallelism）

MoE 天然支持将不同专家放在不同 GPU 上。以 8 个专家分布在 8 张 GPU 为例：

```
GPU 0: Expert 0, 8, 16, ...
GPU 1: Expert 1, 9, 17, ...
...
GPU 7: Expert 7, 15, 23, ...
```

**核心瓶颈**在于 **All-to-All 通信**：每个 token 需路由到对应专家所在 GPU 处理，处理完再收回。每个 GPU 都需要向其他所有 GPU 发送和接收数据。

### 通信开销量化

设 batch 内 token 总数为 $T$，hidden_size 为 $d$，激活专家数为 $k$，GPU 数为 $P$：

$$\text{单次 All-to-All 数据量} \approx T \cdot d \cdot k \cdot \left(1 - \frac{1}{P}\right) \text{ 个元素}$$

前向传播需要 2 次 All-to-All（分发 token + 收集结果），反向传播同样 2 次。随着 GPU 数和模型规模增大，通信开销可能超过计算本身。

这是 MoE 在小 batch 低延迟场景下不如密集模型的根本原因：All-to-All 有固定开销，batch 小时无法摊薄。MoE 的优势在高吞吐批量推理场景，通信开销被大 batch 均摊。

---

## 现代 MoE 模型对比

| 模型 | 机构 | 总参数 | 激活参数 | 专家数 | K |
|---|---|---|---|---|---|
| Switch-C | Google | 1571B | ~7B/层 | 2048 | 1 |
| GLaM | Google | 1200B | 96B | 64 | 2 |
| Mixtral 8x7B | Mistral AI | 46.7B | 12.9B | 8 | 2 |
| Mixtral 8x22B | Mistral AI | 141B | 39B | 8 | 2 |
| Grok-1 | xAI | 314B | ~86B | 8 | 2 |
| DeepSeek-V2 | DeepSeek | 236B | 21B | 160 | 6 |
| DeepSeek-V3 | DeepSeek | 671B | 37B | 256 | 8 |

GPT-4 和 Gemini 1.5 Pro 被广泛认为采用了 MoE 架构，但官方均未正式披露细节。趋势明显：**专家数越来越多，单专家越来越小，激活比例越来越低**——DeepSeek-MoE 细粒度分割思路的延伸。

---

## 局限与未来方向

### 当前局限

**内存消耗**

全部专家参数必须常驻显存（或 CPU 内存/NVMe），即便每次推理只激活少数专家。Mixtral 8x7B 的 46.7B 参数以 fp16 存储需约 93 GB 显存，DeepSeek-V3 的 671B 参数需数十张 H100。

**训练不稳定性**

辅助损失权重 $\alpha$ 敏感：太小则路由崩溃，太大则干扰主任务学习。负载均衡与模型性能之间始终存在张力，稳定训练大规模 MoE 依赖大量工程经验和超参数搜索。

**小 batch 推理效率低**

All-to-All 通信的固定开销在小 batch 场景下无法摊薄，单请求延迟可能高于同等激活参数量的密集模型。延迟敏感场景需要额外工程手段（投机解码、专家预取等）。

### 未来方向

**专家卸载（Expert Offloading）**

将暂时不使用的专家卸载到 CPU 内存或 NVMe SSD，激活时按需加载回显存。llama.cpp 等推理框架已在探索这一方向，目标是让消费级硬件运行 100B+ 参数的 MoE 模型。

**细粒度路由演进**

现有路由是 token 级别——同一 token 的所有特征维度整体路由到同一批专家。可能的演进方向：

- **特征维度路由**：同一 token 的不同特征维度路由到不同专家
- **自适应深度路由**：某些 token 跳过某些 MoE 层（动态深度）

**专家语义特化分析**

研究发现训练完成的 MoE 中，专家会自发展现语义偏好——某些倾向处理数字计算，某些倾向处理代码语法，某些倾向处理多语言输入。这为可解释 MoE 提供了基础，也引出一个问题：能否主动设计专家分工，而非依赖自发涌现。

---

## 总结

MoE 是一种稀疏条件计算范式：输入内容决定计算路径，而非所有参数平等参与每次计算。这使模型参数容量的增长与推理计算成本的增长解耦。

从 1991 年的学术构想，到 2024 年驱动 GPT-4、Gemini 1.5 Pro、DeepSeek-V3 等系统的核心架构，MoE 走过了三十年。内存消耗、通信开销、训练不稳定性都是真实代价，但在"以最少的每次推理 FLOP 撬动最多的模型知识容量"这个维度上，MoE 目前没有替代方案。

---

*参考文献*

- Jacobs et al., 1991. *Adaptive Mixtures of Local Experts.*
- Shazeer et al., 2017. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* [arxiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Lepikhin et al., 2020. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* [arxiv:2006.16668](https://arxiv.org/abs/2006.16668)
- Fedus et al., 2021. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arxiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Jiang et al., 2024. *Mixtral of Experts.* [arxiv:2401.04088](https://arxiv.org/abs/2401.04088)
- DeepSeek-AI, 2024. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* [arxiv:2401.06066](https://arxiv.org/abs/2401.06066)
- DeepSeek-AI, 2024. *DeepSeek-V3 Technical Report.* [arxiv:2412.19437](https://arxiv.org/abs/2412.19437)
