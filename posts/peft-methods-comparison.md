## 核心结论

PEFT，Parameter-Efficient Fine-Tuning，中文可叫“参数高效微调”，意思是冻结大模型主干，只训练极少数新增参数，让这些小参数承载任务适应能力。对初级工程师来说，可以先记住一句话：

Prompt Tuning 像在输入前面贴一段可学习的“软提示”；Prefix Tuning 像给每一层注意力都加一段可学习的“前置上下文”；LoRA 和 Adapter 则不是改输入，而是往模型内部插入可训练的小结构，只训练这些小结构，不训练整网。

这四类方法的共同目标，是把“显存、训练成本、存储成本、任务切换成本”压到远低于全参数微调的水平。它们的主要差异，在于可训练参数放在哪里、更新信号能影响多深、性能上限有多高。

从经验上看，常见 Pareto 前沿，也就是“参数越少通常越省，但性能通常也越低”的最优折中边界，大致是：

Prompt < Prefix < LoRA/Adapter < Full Fine-Tuning

这里的“小于”不是绝对优劣，而是“参数量和性能能力整体递增”。如果你只允许动极少参数，Prompt/Prefix 更轻；如果你希望更稳定地逼近全量微调，LoRA/Adapter 往往更常用；如果你有足够数据、显存和部署空间，全参数微调仍然通常是最终性能上限。

下面这个表可以先建立直觉。数值是 7B 量级模型的典型级别，用来说明量级关系，而不是通用固定答案。

| 方法 | 参数位置 | 典型可训练参数占比 | 典型参数量级 | 性能级别 |
|---|---|---:|---:|---|
| Prompt Tuning | 输入侧软提示 | 约 0.0059% | 0.41M | 最轻，性能通常最低 |
| Prefix Tuning | 每层 attention 前缀 | 约 0.0749% | 5.24M | 比 Prompt 强，但低样本波动较大 |
| LoRA | 线性层低秩增量 | 约 0.24% | 16.78M | 常见最优折中 |
| Adapter | 层内瓶颈模块 | 约 0.48% | 33.55M | 稳定，参数略高 |
| Full FT | 全部权重 | 100% | 全模型 | 上限最高，成本最高 |

可以把它理解成一条“参数量 vs 准确率”的 Pareto 曲线示意图：越往右参数越多，通常越能逼近全量微调；越往左越省资源，但对任务和数据更敏感。

---

## 问题定义与边界

这篇文章讨论的问题不是“怎么让模型学会所有新知识”，而是更具体的工程问题：

在保持大模型主干冻结的前提下，如何只训练少量参数，让模型学会某个具体任务或领域信号，并在有限显存、有限样本、频繁切换任务的约束下，取得尽可能高的效果。

这里的“冻结”，就是原始权重不更新；“主干”，就是 Transformer 里的大部分层和参数；“任务信号”，就是分类、抽取、问答、风格控制这类目标所需的额外偏好或边界。

一个真实工程例子是金融分类。假设你要在 LLaMA-2 7B 上做 200 类交易分类。如果用全参数微调，训练资源可能接近 140GB 显存级别；如果改成 LoRA，设 $r=16,\alpha=32$，在 5 万样本上训练，显存可以下降到 20-30GB，同时维持约 95% 的基线性能。这个变化的本质不是“模型更强了”，而是“你只训练一小块可调结构，所以优化和存储都轻很多”。

这类问题的边界限制通常有四个：

- 显存边界：你能不能放下训练状态、梯度和优化器。
- 模型大小边界：7B、13B、70B 对可行方案影响很大。
- 样本数量边界：100 条、500 条、5 万条，最优方法会明显不同。
- 部署与切换边界：你是否需要一个底座模型同时挂很多任务插件，快速切换。

还有一个容易忽略的边界：任务目标类型。PEFT 对分类、抽取、指令跟随通常都有效，但“有效”不等于“相同方式最优”。短文本分类和长上下文问答，对 Prefix 或 Prompt 的依赖方式并不一样。

玩具例子可以帮助理解。假设你有一个通用语言模型，现在要让它只学会“把影评判成正面或负面”。全参数微调等于把整个模型都拿去改；PEFT 则是告诉模型：“原来的语言能力别动，我只给你加一小段可调开关，让你在这个任务上偏向正确决策。”

所以，PEFT 解决的是“在不重写整台机器的情况下，只换几个旋钮去适配任务”的问题。

---

## 核心机制与推导

先从最常用的 LoRA 讲起。LoRA，Low-Rank Adaptation，中文可叫“低秩适配”。“低秩”这句话对白话一点就是：本来你可以随意修改一个大矩阵，现在你只允许通过两个更小的矩阵组合去修改它，所以更新空间被压缩了。

设原始线性层权重为 $W \in \mathbb{R}^{d \times k}$，原始前向是：

$$
y = Wx
$$

LoRA 不直接训练 $W$，而是引入一个增量：

$$
\Delta W = BA
$$

其中：

$$
A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}, \quad r \ll \min(d,k)
$$

于是前向变成：

$$
y = (W + \Delta W)x = (W + BA)x
$$

训练时只更新 $A,B$，冻结 $W$。如果把损失函数记为 $\mathcal{L}$，那么对增量矩阵的梯度先传到 $\Delta W$，再传到 $A,B$：

$$
\frac{\partial \mathcal{L}}{\partial A} = B^T \frac{\partial \mathcal{L}}{\partial \Delta W}, \qquad
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial \Delta W} A^T
$$

这说明优化器真正更新的是低秩分解里的两个小矩阵，而不是原始大矩阵。新手可以把它理解成：LoRA 是在原始权重上贴两层“薄板”，只调薄板，不碰钢筋骨架。

Prefix Tuning 的机制不同。Prefix，前缀，就是在每一层注意力里额外加一段可训练向量，常见做法是拼接到 key/value 前面。设原始 attention 的 key/value 是 $K_l,V_l$，第 $l$ 层的前缀向量是 $P_l^K,P_l^V$，则可以写成：

$$
K_l' = [P_l^K; K_l], \qquad V_l' = [P_l^V; V_l]
$$

这里的“拼接”意思是：不是替换原始上下文，而是在它前面多放一段可学习的虚拟上下文。模型在做注意力时，会把这段前缀也当成可参考信息。

Prompt Tuning 更轻。它通常只在输入嵌入前面添加一段可训练软提示，不像 Prefix 那样深入每一层。直观上，它对模型内部行为的干预最弱，所以参数最少，但性能和稳定性也更依赖任务是否适合“靠输入提示解决”。

Adapter 则是在层内插入一个小瓶颈网络。所谓“瓶颈”，就是先把高维压低，再升回去。例如原始隐藏状态 $h \in \mathbb{R}^d$，Adapter 做：

$$
h' = h + W_{\text{up}} \sigma(W_{\text{down}} h)
$$

其中 $W_{\text{down}}: d \to m$，$W_{\text{up}}: m \to d$，且 $m \ll d$。它本质上是在每层挂一个小残差支路，主路冻结，只训练这条小支路。

如果用文字描述四者的简化图示，可以这样看：

- Prompt：在最前面加可训练 token。
- Prefix：在每层 attention 前面加可训练 key/value。
- LoRA：在原有线性层旁边加低秩增量支路。
- Adapter：在层内部插一个小瓶颈残差块。

为什么 LoRA 和 Adapter 常比 Prompt/Prefix 更稳？原因在于它们直接作用在内部表示变换上，优化空间更接近“改模型的算子”；而 Prompt/Prefix 更像“给模型额外上下文”，表达能力受底座模型原有习惯约束更强。

但这不意味着 Prefix/Prompt 没价值。它们的优势是轻，特别适合任务切换频繁、极端受限部署、或者样本很少且你希望最小改动时使用。

---

## 代码实现

下面先给一个可运行的玩具 Python 例子，用 NumPy 模拟 LoRA 的核心计算。它不是完整训练脚本，但足够说明“冻结原始权重，只更新低秩增量”的结构。

```python
import numpy as np

np.random.seed(0)

d_in, d_out, r = 4, 3, 2
x = np.array([[1.0], [2.0], [-1.0], [0.5]])

# 冻结的原始权重
W = np.random.randn(d_out, d_in)

# 可训练的 LoRA 参数
A = np.random.randn(r, d_in) * 0.01
B = np.random.randn(d_out, r) * 0.01

def lora_forward(W, A, B, x):
    delta_W = B @ A
    return (W + delta_W) @ x

y_base = W @ x
y_lora = lora_forward(W, A, B, x)

# LoRA 输出应等于原始输出 + 增量输出
delta_y = (B @ A) @ x
assert np.allclose(y_lora, y_base + delta_y)

# 冻结 W，只改变 A/B 时输出会变化
A2 = A + 0.1
y_lora_2 = lora_forward(W, A2, B, x)
assert not np.allclose(y_lora, y_lora_2)

print("LoRA toy example passed.")
```

如果换成接近 PyTorch 的伪代码，四类方法的实现核心分别如下。

LoRA 结构：

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r, alpha):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(r, base_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        base_out = self.base(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scale * delta
```

Adapter 结构：

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.act = nn.ReLU()

    def forward(self, h):
        return h + self.up(self.act(self.down(h)))
```

Prefix Tuning 结构：

```python
class PrefixBlock(nn.Module):
    def __init__(self, num_layers, prefix_len, num_heads, head_dim):
        super().__init__()
        self.prefix_k = nn.Parameter(torch.randn(num_layers, prefix_len, num_heads, head_dim))
        self.prefix_v = nn.Parameter(torch.randn(num_layers, prefix_len, num_heads, head_dim))

    def get_prefix(self, layer_id, batch_size):
        pk = self.prefix_k[layer_id].unsqueeze(0).expand(batch_size, -1, -1, -1)
        pv = self.prefix_v[layer_id].unsqueeze(0).expand(batch_size, -1, -1, -1)
        return pk, pv
```

Prompt Tuning 结构：

```python
class PromptEmbedding(nn.Module):
    def __init__(self, prompt_len, hidden_size):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_len, hidden_size) * 0.02)

    def forward(self, token_embeds, batch_size):
        p = self.prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([p, token_embeds], dim=1)
```

训练流程的共同点也很固定：

```python
for p in model.parameters():
    p.requires_grad = False

# 插入 LoRA / Adapter / Prefix / Prompt 模块
attach_peft_modules(model)

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=lr)

assert len(trainable) > 0
assert all(p.requires_grad for p in trainable)
```

真实工程里，最关键的不是“代码能不能跑”，而是三件事：

- 你是否真的冻结了底座参数。
- 优化器里是否只放了新增参数。
- 推理时是否正确加载了底座模型和对应 PEFT 模块。

如果这三步有一步错，实验结果会完全失真。

---

## 工程权衡与常见坑

PEFT 的核心权衡，不是“能不能替代全量微调”，而是“在当前资源边界下，哪种方法最划算”。

先看低样本问题。样本少于 1K 时，模型学到的不是稳定规律，而可能只是数据集局部噪声。参数越少不一定越稳，因为极小参数空间虽然不容易过拟合整网，但也可能把有限训练信号压进一个过于狭窄的位置，结果是表达不够或写入噪声过重。

一个常见经验是：500 样本下，Prefix 可能只有约 71% 准确率，而 LoRA 可以到约 82%。原因通常不是 Prefix 一定更差，而是它更依赖前缀向量是否恰好学到对所有层都有效的上下文模式，这在低样本时很难。

下表总结常见风险。

| 方法 | 低样本典型风险 | 超参数敏感点 | 常见对策 |
|---|---|---|---|
| Prompt | 表达能力不足，任务迁移弱 | prompt 长度、初始化 | 用更强底座模型，增加 prompt 长度，做模板筛选 |
| Prefix | 容易写入噪声，层间共享难优化 | prefix 长度、层数、学习率 | 降低学习率，减少前缀长度，先做小集验证 |
| LoRA | rank 太小欠表达，学习率太大易退化 | rank、alpha、lr、target modules | 从 `r=8/16` 起扫参，加权重衰减或数据增强 |
| Adapter | 参数略多，推理路径更长 | bottleneck 大小、插入位置 | 只插关键层，控制瓶颈维度 |
| Full FT | 显存高、过拟合和遗忘更明显 | 全局 lr、训练轮数 | 更强正则、更大数据、更高成本训练 |

LoRA 的一个典型坑是 rank 太小。rank 可以理解成“更新空间的宽度”。如果 $r$ 太小，$\Delta W=BA$ 的表达能力不够，模型只能做很有限的修正；但如果 $r$ 盲目加大，虽然更强，却会抬高显存和存储。对 7B 级模型，很多任务从 `r=8` 或 `r=16` 开始试更合理。

另一个坑是学习率。PEFT 模块小，很多人会误以为“参数少，可以把学习率开很大”。实际往往相反：新增参数虽然少，但梯度会直接影响输出分布，过大的学习率很容易让训练前几百步就偏离稳定区域。

还要注意“灾难性遗忘”。这句话的白话意思是：模型在适应新任务时，把原本有用的能力破坏掉了。即便底座参数冻结，LoRA 或 Adapter 也可能通过增量路径覆盖原有行为。在 100-150 样本这种极低资源设置下，这种现象会更明显，通常需要更强正则、早停或数据增强。

最后是部署问题。Prompt/Prefix 常常更轻，但 Prefix 需要在每层 attention 参与计算；Adapter 会引入额外层内模块；LoRA 在推理时可以合并权重，也可以保留增量形式。不同框架的推理开销差别不只看参数量，还要看实现方式。

---

## 替代方案与适用边界

如果只问一句“默认先选什么”，在大多数今天的工程实践里，LoRA 往往是默认起点。原因很简单：它在参数量、性能、实现复杂度、生态支持之间，通常是最均衡的。

但选型不能脱离边界。下面这个决策矩阵更实用。

| 方法 | 样本数 | 显存约束 | 切换频率 | 性能优先级 | 适用判断 |
|---|---|---|---|---|---|
| Prompt | 极少，几十到几百 | 极紧 | 很高 | 中低 | 先试，适合超轻量实验 |
| Prefix | 少到中等 | 紧 | 高 | 中 | 适合想增强上下文控制但能接受调参成本 |
| LoRA | 中到大，几百到数万 | 中等 | 高 | 高 | 大多数任务的默认工程选择 |
| Adapter | 中到大 | 中等偏宽 | 中高 | 高 | 适合希望结构清晰、模块独立的场景 |
| Full FT | 大样本 | 宽松 | 低 | 最高 | 当资源充足且追求上限时使用 |

可以把它转成几个具体判断：

- 如果样本极少、只是做快速适配或验证任务可行性，Prompt/Prefix 可以先试。
- 如果你有几百到几万样本，目标是稳定逼近全参效果，LoRA/Adapter 更合适。
- 如果你要在一个底座模型上频繁切换很多下游任务，PEFT 比 Full FT 更适合管理。
- 如果任务要求极致精度，而且显存、训练时间、部署体积都不是主要矛盾，全参数微调仍然通常是性能上限。

还有混合方案。比如 LoRA + Prompt：前者改内部映射，后者补输入侧任务提示；或者 PEFT 先做监督微调，再接强化学习阶段。这些方案的共同点不是“更高级”，而是把适配信号拆到多个位置，提高表达能力。

但对初级工程师来说，实践顺序最好是：

1. 先用 LoRA 建立基线。
2. 如果显存再紧，就试 Prompt/Prefix。
3. 如果任务对结构控制很敏感，或团队更偏好模块化插层，再看 Adapter。
4. 只有在确定 PEFT 已接近瓶颈、且资源允许时，再考虑 Full FT。

---

## 参考资料

1. Michael Brenndoerfer, “PEFT Comparison: Choosing the Right Fine-Tuning Method”  
   链接：https://mbrenndoerfer.com/writing/peft-comparison-lora-qlora-adapters-selection-guide?utm_source=openai  
   关键词：参数量级、Pareto 视角、7B 模型对比表。本文中的 Prompt/Prefix/LoRA/Adapter 参数量级表主要依据这篇资料的整理口径。

2. Data Annotation Tech, “Parameter-Efficient Fine-Tuning and the Data Quality Gap”  
   链接：https://www.dataannotation.tech/developers/parameter-efficient-fine-tuning?utm_source=openai  
   关键词：低样本表现、数据质量、Prefix/Prompt 稳定性。本文中关于 500 样本下 Prefix 与 LoRA 的表现差异、低样本噪声风险，主要参考这篇。

3. SandGarden, “The Art of Efficient AI Adaptation with PEFT”  
   链接：https://www.sandgarden.com/learn/parameter-efficient-fine-tuning-peft?utm_source=openai  
   关键词：PEFT 总体定义、工程权衡、参数效率。本文关于 PEFT 的总体定义和资源收益框架，主要参考这篇。

4. Next.gr, “Fine-Tuning LLMs with LoRA”  
   链接：https://next.gr/ai/large-language-models/fine-tuning-llms-with-lora?utm_source=openai  
   关键词：LoRA 公式、低秩分解、梯度路径。本文中 $\Delta W = BA$ 及其梯度推导形式参考这篇的讲解口径。

5. Amine SmartFlow AI, “Fine-Tuning Explained: PEFT, LoRA, Adapters, Which Choice for Your Case?”  
   链接：https://aminesmartflowai.com/fine-tuning-explained-peft-lora-adapters-which-choice-for-your-case/?utm_source=openai  
   关键词：真实工程案例、金融分类、显存收益。本文中的金融科技交易分类工程例子主要依据这篇的案例描述。
