## 引言：规模即能力的代价

大语言模型有一个简单粗暴的核心洞见：**规模即能力（Scale is all you need）**。

从 GPT-2 的 15 亿参数，到 GPT-3 的 1750 亿，再到当前最强模型的万亿级参数，模型能力随规模增长呈现出惊人的涌现（Emergence）特性。但这里隐藏着一个巨大的物理约束：**计算量与参数量成正比**。

一个拥有 1750 亿参数的**密集模型（Dense Model）**，推理时每生成一个词（token），都需要全部 1750 亿个参数参与计算。这意味着：

- 推理成本高昂（算力、内存、电力）
- 单卡装不下，需要多机多卡并行
- 训练成本动辄数百万美元

**混合专家模型（Mixture of Experts，MoE）**正是为了打破这个约束而生：让模型**拥有更多参数，但每次推理只激活其中一小部分**。

这种以参数空间换计算时间的设计哲学，学术上称为：**稀疏条件计算（Sparse Conditional Computation）**。

---

## 从直觉开始：分诊台与专科医生

在进入复杂的数学与代码之前，我们先来建立直觉。

想象一家大型综合医院。

**密集模型（Dense Model）的工作方式**，相当于医院里的每一位专科医生——心脏科、神经外科、皮肤科、消化科——都参与诊断你的感冒。每位医生都看诊、都给意见、都参与决策。结果固然面面俱到，但极度浪费医疗资源。

**MoE 的工作方式**，相当于在医院入口设一个**分诊台（Router，路由器）**。

1. 当你（输入数据）走进医院时，分诊护士（Router）先看一眼你的症状
2. 然后迅速将你精准分配给 2-3 位最相关的专科医生（Experts，专家）
3. 其他没被点名的医生照常在岗，但不参与你此次诊断。

这揭示了 MoE 的两个核心组件：

+ **专家（Experts）**：一批并行存在的子网络，各自专门化处理不同类型的输入
+ **路由器（Router / Gating Network）**：决定把当前任务（token）分配给哪些专家的机制

通过这种方式，每次推理只激活 K 个专家（通常 $K=1$ 或 $K=2$），模型因此可以拥有远超其激活参数量的总知识库。

---

## MoE 的基本结构

有了直观印象，我们来看看在主流的大语言模型中，MoE 究竟长什么样。

在标准的 Transformer 模型中，网络是由注意力层（Attention）和前馈网络层（FFN）交替构成的：

```
[Attention] → [FFN] → [Attention] → [FFN] → ...
```

MoE 将其中部分（或全部）FFN 层替换为 MoE 层：

```
[Attention] → [MoE Layer] → [Attention] → [MoE Layer] → ...
```

每个 MoE 层的内部结构（每个独立的 FNN 前加了一个统一的“路由器”）：

```
                      ┌── Expert 1 (FFN) ──┐
输入 token x → Router ─┼── Expert 2 (FFN) ──┼─→ 加权求和 → 输出
                      └── ... Expert N ────┘
                        (只激活 Top-K 个)
```

Router 是一个轻量级线性层，负责为每个 token 打分并选出最合适的 K 个专家。值得注意的是，**Attention 层不做 MoE 处理**——所有 token 共享同一套 Attention 参数，只有 FFN 层被稀疏化。

---

## 数学深挖：路由器的「选秀打分」

宏观直觉有了，结构也清楚了。下面带你看路由器在后台究竟怎么运作——不跳过公式，但每一步都配具体数字跑一遍。

> 为了讲清楚，我们设 **4 个专家（$N=4$）、每次只激活前 2 名（$K=2$）**。

### 第一步：基础打分 + 加点"运气"

路由器里有一个**可学习的打分矩阵 $W_g$**，每个词（token）进来时，路由器把它和 $W_g$ 相乘，给 $N$ 个专家各打一个基础分。

但只靠基础分有个问题：训练早期，如果某个专家偶然多拿几次梯度就变得更强，路由器就越来越偏爱它——"强者恒强"的正反馈会导致大量专家几乎永远被冷落，最终模型退化成只用几个专家的低效结构，也就是**路由崩溃（Router Collapse）**。

解决方案：**加噪声**。每次打分时，给每个专家叠加一点随机"运气分"，打破固化：

$$H(x)_i = (x \cdot W_g)_i + \varepsilon_i \cdot \text{Softplus}\!\left((x \cdot W_{noise})_i\right), \qquad \varepsilon_i \sim \mathcal{N}(0,1)$$

| 项 | 含义 |
|---|---|
| $x \cdot W_g$ | 基础分，路由器通过训练学习出来的偏好 |
| $\varepsilon_i \sim \mathcal{N}(0,1)$ | 每次随机采样的正态噪声，引入"运气成分" |
| $\text{Softplus}(\cdot)$ | 控制噪声幅度的缩放器，同样可学习（$\text{Softplus}(z) = \ln(1+e^z) > 0$） |

**推演示例**：4 个专家加上噪声后的最终得分 $H(x)$：

```
专家 1：8.0 分
专家 2：2.0 分
专家 3：1.0 分
专家 4：7.0 分
→ 分数数组：[8.0, 2.0, 1.0, 7.0]
```

### 第二步：残酷淘汰赛（KeepTopK）

$K=2$，只保留前 2 名，落选者分数直接打入负无穷大 $-\infty$——意味着彻底出局：

$$\text{KeepTopK}(v,k)_i = \begin{cases} v_i & \text{if } v_i \text{ is in top-}k \text{ of } v \\ -\infty & \text{otherwise} \end{cases}$$

**推演示例**：专家 1（8.0）和专家 4（7.0）胜出，数组变为：

```
[8.0, -∞, -∞, 7.0]
```

### 第三步：分数变权重（Softmax）

原始分数还不能直接用，需要转换成"加起来等于 1"的权重比例，这就是 **Softmax** 的作用——以自然常数 $e$ 为底、得分为指数，再求各自占总和的比例：

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), K))$$

Softmax 有一个关键性质：$e^{-\infty} = 0$，被淘汰专家的权重自动归零，不需要任何额外处理。

**推演示例**：

$$\text{专家 1 权重} = \frac{e^{8.0}}{e^{8.0} + e^{-\infty} + e^{-\infty} + e^{7.0}} = \frac{e^{8.0}}{e^{8.0} + e^{7.0}} \approx 0.73$$

$$\text{专家 4 权重} = \frac{e^{7.0}}{e^{8.0} + e^{7.0}} \approx 0.27$$

最终权重向量：$G(x) = [0.73,\ 0,\ 0,\ 0.27]$

### 闭环：稀疏激活的数学本质

MoE 层的完整输出公式：

$$\text{MoE}(x) = \sum_{i=1}^{N} G_i(x) \cdot E_i(x)$$

代入数字：

$$\text{MoE}(x) = 0.73 \cdot E_1(x) + 0 \cdot E_2(x) + 0 \cdot E_3(x) + 0.27 \cdot E_4(x)$$

$G_2 = G_3 = 0$ 意味着计算机**根本不需要运行专家 2 和专家 3**——乘数是零，无论它们输出什么都不影响结果，所以直接跳过，零 FLOP 消耗。

**这就是稀疏激活的数学本质：算力因结构而省，而非靠近似或剪枝。** 模型拥有 $N$ 个专家的知识容量，但每次推理只花 $K$ 个专家的计算成本。

---

## 负载均衡：训练的暗礁

这里有一个致命陷阱：**路由器崩溃（Router Collapse）**。

如果不加干预，路由器会很快学到"把所有 token 都发给同一个或少数几个专家"。因为一旦某个专家偶然得到更多梯度更新，它就变得更强，路由器就更倾向于选它，进而触发**马太效应式正反馈**：越强越被选，越被选越强。最终，大部分专家几乎从不参与计算，模型退化为一个低效的密集模型。

### 辅助损失（Auxiliary Loss）

Shazeer et al.（2017）引入**辅助损失**来强制负载均衡：

$$\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中：
- $f_i$ = 专家 $i$ 实际处理的 token 比例（真实分配量，不可微）
- $P_i$ = 路由器对专家 $i$ 的平均 softmax 概率（可微的意愿分配量）
- $\alpha$ 为超参数，控制均衡强度（通常设为 $10^{-2}$ 到 $10^{-3}$）

当 $f_i$ 和 $P_i$ 都趋近 $1/N$（均匀分布）时，$\sum f_i \cdot P_i$ 取得最小值。这个损失提供梯度信号，鼓励路由器均匀使用所有专家。

总训练损失为：

$$\mathcal{L} = \mathcal{L}_{main} + \mathcal{L}_{aux}$$

### 容量因子（Capacity Factor）

工程上的另一手段是设置每个专家的 **token 处理上限**：

$$C = \left\lfloor \frac{T \cdot k}{N} \right\rfloor \times \text{capacity\_factor}$$

其中 $T$ 为 batch 内 token 总数，$k$ 为每个 token 激活的专家数。capacity_factor 通常设为 1.0 至 1.5。

超出 $C$ 的溢出 token 会跳过该专家，直接通过残差连接传递。这防止某个专家成为瓶颈，但溢出的 token 得不到 FFN 层的充分处理。

---

## 关键论文演进

### 1991 — 起源

Jacobs et al. 在 1991 年提出 MoE 框架，用于让不同子模型各自负责数据空间的不同区域。这时还是前深度学习时代，MoE 是纯理论研究。概念的核心——**根据输入内容选择子模型**——在三十年后被证明是大规模 AI 的基石之一。

### 2017 — 稀疏门控 MoE 的大规模首次验证

Shazeer et al.（Google Brain）发表 [*Outrageously Large Neural Networks*](https://arxiv.org/abs/1701.06538)，将 MoE 层嵌入 LSTM 语言模型，实现当时参数量最大的语言模型（约 137B 参数）。

核心贡献：提出 **Noisy Top-K Gating**，也就是本文第二章逐步推演的那套"加噪声→淘汰赛→Softmax"三步机制。在路由分数上叠加可学习幅度的正态噪声：

$$H(x)_i = (x \cdot W_g)_i + \varepsilon_i \cdot \text{Softplus}\!\left((x \cdot W_{noise})_i\right), \qquad \varepsilon_i \sim \mathcal{N}(0, 1)$$

噪声在训练初期提供探索性，防止路由器过早固化到少数专家——对抗路由崩溃的第一个实践手段，也奠定了之后所有 MoE 训练的基础范式。

### 2020 — GShard：首个千亿级 MoE

Lepikhin et al.（Google）将 MoE 应用于神经机器翻译，实现 **600B 参数**模型（激活约 48B），是 MoE 真正大规模工程落地的第一步。

GShard 的工程贡献：
- **Expert Parallelism（专家并行）**：不同专家放在不同的 TPU 核心
- **Local Group Dispatching**：将 batch 内 token 分组，确保每组内的跨专家负载均衡
- **随机路由（Random Routing）**：当首选专家满载时，以一定概率随机选第二候选专家，而非直接溢出

### 2021 — Switch Transformer：极简主义的万亿参数

Fedus et al.（Google）提出一个大胆简化：**Top-K 中 K=1**。

> "每个 token 只需要一个专家。"

简化的好处：
- 路由计算和反向传播更简单（梯度只流向一个专家）
- All-to-All 通信量减半
- 训练更稳定

Switch Transformer 达到 **1.6T 总参数**，在等计算预算下比 T5-XXL 快 7 倍。这证明 Top-1 路由是可行的——不需要总是 Top-2。

| 模型 | 总参数 | 激活参数/层 | 专家数 | K |
|---|---|---|---|---|
| T5-XXL | 11B | 11B（全部）| — | — |
| Switch-C | 1571B | ~7B | 2048 | 1 |

### 2023 — Mixtral 8x7B：开源 MoE 的标杆

Mistral AI 将 MoE 带入开源社区，Mixtral 8x7B 成为第一个被广泛使用的开源 MoE 模型。

架构细节：
- 8 个专家，每次激活 Top-2
- 总参数 ~46.7B，但激活参数仅 **~12.9B**
- 在多个基准测试上超越 LLaMA 2 70B（后者总参数是前者约 1.5 倍）

Mixtral 清晰地量化了 MoE 的实际收益：**以约 1/3 的激活算力，获得接近 2× 总参数量密集模型的性能**。这让 MoE 的价值主张第一次对整个开源社区变得清晰。

### 2024 — DeepSeek-MoE：细粒度专家分割

DeepSeek 团队提出两项关键创新，显著提升了 MoE 的参数利用效率。

#### 创新 1：细粒度专家分割（Fine-Grained Expert Segmentation）

传统 MoE 中每个专家是一个完整的 FFN（隐层维度为 $d_{ff}$）。DeepSeek-MoE 将每个专家的隐层维度缩小为 $d_{ff}/m$，同时把专家数增加到 $mN$，并激活 $mk$ 个专家。总 FLOP 不变，但路由器有了更细的组合粒度——$mk$ 个小专家的任意组合，远比 $k$ 个大专家的组合更丰富，大幅减少了不同 token 被迫共用同一专家的知识冗余。

#### 创新 2：共享专家隔离（Shared Expert Isolation）

在所有路由专家之外，另设 $K_s$ 个**始终激活的共享专家**，专门处理各 token 共同需要的通用知识（基础语法、通用语义等）。其余路由专家则专注于特化知识。

$$\text{DeepSeekMoE}(x) = \underbrace{\sum_{i=1}^{K_s} E_i^{shared}(x)}_{\text{共享专家（始终激活）}} + \underbrace{\sum_{j \in \text{TopK}} G_j(x) \cdot E_j^{routed}(x)}_{\text{路由专家（稀疏激活）}}$$

这解决了传统 MoE 的一个深层问题：普通路由专家被迫同时存储特化知识和通用知识，形成冗余。共享专家卸载通用知识后，路由专家的表达能力更纯粹。

DeepSeek-V3（2024 年末）将这一架构推到 **671B 总参数、激活仅 37B**，在多个任务上达到或超越 GPT-4o 的水平。

---

## 工程挑战：专家并行与通信瓶颈

MoE 的参数规模要求独特的分布式并行策略，也带来了密集模型没有的工程挑战。

### 密集模型的标准并行策略

- **张量并行（Tensor Parallelism）**：把单个矩阵按行/列切割到多 GPU，每张 GPU 持有矩阵的一个切片
- **流水线并行（Pipeline Parallelism）**：不同 Transformer 层分配到不同 GPU 组，形成计算流水线

### MoE 新增：专家并行（Expert Parallelism）

MoE 天然可以将**不同专家放在不同 GPU** 上。以 8 个专家分布在 8 张 GPU 为例：

```
GPU 0: Expert 0, 8, 16, ...
GPU 1: Expert 1, 9, 17, ...
...
GPU 7: Expert 7, 15, 23, ...
```

每个专家只在自己所在的 GPU 上存储和计算，参数内存得以分散。

**核心瓶颈**：一个 batch 中，每个 token 需要被路由到对应专家所在的 GPU 上处理，处理完再收回。这需要 **All-to-All 通信**——集合通信中开销最大的操作之一，因为每个 GPU 都需要向其他所有 GPU 发送和接收数据。

### 通信开销的量化

设 batch 内 token 总数为 $T$，hidden_size 为 $d$，激活专家数为 $k$，GPU 数为 $P$：

$$\text{单次 All-to-All 数据量} \approx T \cdot d \cdot k \cdot \left(1 - \frac{1}{P}\right) \text{ 个元素}$$

前向传播需要 **2 次 All-to-All**（分发 token + 收集结果），反向传播还需要 2 次。随着 GPU 数 $P$ 和模型规模增大，通信开销可能超过计算本身。

这是 MoE 在**小 batch 低延迟**场景下不如密集模型的根本原因：All-to-All 有固定开销，batch 小时无法摊薄。MoE 的真正优势在**高吞吐批量推理**场景，此时通信开销被大 batch 均摊到可接受的范围。

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

GPT-4 和 Gemini 1.5 Pro 被广泛认为采用了 MoE 架构，但 OpenAI 和 Google 均未正式披露细节。从这张表可以看到一个趋势：**专家数越来越多，单专家越来越小，激活比例越来越低**——正是 DeepSeek-MoE 细粒度分割思路的延伸。

---

## MoE 的局限与未来

### 当前局限

**内存消耗**

MoE 的全部专家参数必须常驻显存（或 CPU 内存/NVMe）——即便每次推理只激活少数专家。Mixtral 8x7B 的 46.7B 参数以 fp16 存储需要约 93 GB 显存，DeepSeek-V3 的 671B 参数则需要数十张 H100 才能完整加载。内存墙是 MoE 部署的第一道门槛。

**训练不稳定性**

辅助损失的权重 $\alpha$ 很敏感：太小则路由崩溃，太大则辅助损失干扰主任务的学习。负载均衡与模型性能之间始终存在张力。稳定训练大规模 MoE 依赖大量工程经验和超参数搜索，这是密集模型不需要面对的额外复杂度。

**小 batch 推理效率低**

如前文分析，All-to-All 通信的固定开销在小 batch 场景下无法摊薄，单请求延迟可能高于同等激活参数量的密集模型。MoE 是一种为吞吐量优化的架构，在延迟敏感场景需要额外工程手段（如投机解码、专家预取）。

### 未来方向

**Expert Offloading（专家卸载）**

将暂时不使用的专家卸载到 CPU 内存或 NVMe SSD，激活时再按需加载回显存。llama.cpp 等推理框架已在探索这一方向，目标是让消费级硬件也能运行 100B+ 参数级别的 MoE 模型。

**细粒度路由演进**

现有路由是 **token 级别**的——同一个 token 的所有特征维度被整体路由到同一批专家。未来可能出现：
- **特征维度路由**：同一 token 的不同特征维度路由到不同专家
- **自适应深度路由**：某些 token 可以跳过某些 MoE 层（动态深度）

**专家语义特化分析**

研究发现，训练完成的 MoE 中专家会自发地展现出语义偏好——某些专家倾向于处理数字计算，某些倾向于处理代码语法，某些倾向于处理多语言混合输入。这为**可解释 MoE** 提供了基础，也引出了一个更深的问题：能否主动设计专家的分工，而非让其自发涌现？

---

## 总结

MoE 的本质是一种**稀疏条件计算（Sparse Conditional Computation）**范式：

> 输入内容决定计算路径，而非所有参数平等参与每次计算。

这个思想让模型参数容量的增长与推理计算成本的增长解耦——这是目前在实践中真正大规模 work 的参数高效化方案。

从 1991 年的学术构想，到 2024 年驱动 GPT-4、Gemini 1.5 Pro、DeepSeek-V3 等最强 AI 系统的核心架构，MoE 走过了三十年。它不是银弹：内存消耗、通信开销、训练不稳定性都是真实的代价。但在"以最少的每次推理 FLOP 撬动最多的模型知识容量"这个维度上，目前没有比它更好的答案。

---

*参考文献*

- Jacobs et al., 1991. *Adaptive Mixtures of Local Experts.*
- Shazeer et al., 2017. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* [arxiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Lepikhin et al., 2020. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* [arxiv:2006.16668](https://arxiv.org/abs/2006.16668)
- Fedus et al., 2021. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arxiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Jiang et al., 2024. *Mixtral of Experts.* [arxiv:2401.04088](https://arxiv.org/abs/2401.04088)
- DeepSeek-AI, 2024. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* [arxiv:2401.06066](https://arxiv.org/abs/2401.06066)
- DeepSeek-AI, 2024. *DeepSeek-V3 Technical Report.* [arxiv:2412.19437](https://arxiv.org/abs/2412.19437)
