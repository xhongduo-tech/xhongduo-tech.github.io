## 核心结论

UL2 的核心不是“发明了一个更强的单一预训练目标”，而是把几类原本分开的目标放进同一套训练框架里。它用 Mixture-of-Denoisers，简称 MoD，即“多种去噪器混合训练”，让同一个 Transformer 同时学三种能力：

1. R-Denoiser：接近 T5 的短 span corruption，也就是“挖掉一小段再补回去”。
2. S-Denoiser：接近 Prefix LM，也就是“给前缀，让模型顺着往后写”。
3. X-Denoiser：更极端的大跨度去噪，也就是“输入保留很少，要求模型重建更大范围文本”。

UL2 还加了模式选择 token。token 是“模型输入中的特殊标记”。训练和推理时，输入前面放 `[NLU]`、`[NLG]`、`[S2S]` 这类前缀，模型就会偏向调用对应的“读写状态”。这意味着同一个参数集合里，同时保留理解、补全、生成三类行为，而不是为不同任务分别训练不同模型。

论文里的关键发现也很重要：Few-shot 能力提升，主要来自任务和目标的多样性，而不是某个目标形式本身天然更神。换句话说，UL2 的价值不是“prefix LM 一定更强”或“span corruption 一定更强”，而是把这些训练信号统一起来，让模型不在能力分布上偏科。

从结果看，UL2 20B 在 FLAN 指令微调后，论文与后续公开资料显示其综合任务表现优于同规模 T5 变体，并在 MMLU 等知识与推理基准上达到很强水平。Flan-UL2 20B 的模型卡给出的 MMLU 为 55.7%，略高于 Flan-T5-XXL 11B 的 55.1%。这说明 UL2 checkpoint 作为后续 instruction tuning 的基础模型，确实有工程价值。

---

## 问题定义与边界

UL2 要解决的问题很具体：一个预训练目标，往往只能把模型推向某一侧。

纯 Prefix LM 更像“从左到右续写”，对开放生成友好，但不天然适合双向理解。双向理解是“同时利用左右文恢复缺失内容”的能力。纯 span corruption 则更像“完形填空”，对表示学习和条件生成友好，但在 few-shot 续写、长程生成风格上不一定最佳。

所以 UL2 的问题定义不是“设计一个新 mask 策略”，而是：

$$
\text{如何让一个模型在同一训练阶段中同时接触理解型、生成型、重建型目标，并在推理时可显式切换。}
$$

它的边界也要说清楚。

第一，UL2 仍然是序列到序列框架，本质上依赖 encoder-decoder 架构，而不是把 decoder-only 模型直接改个损失就完事。第二，UL2 统一的是“预训练目标范式”，不是替代所有后训练流程。要做聊天、问答、指令遵循，后面仍然常常需要 FLAN 这类 instruction tuning。第三，UL2 解决的是“综合能力分布”问题，不保证在某个单一生成任务上必然超过专门为该任务设计的模型。

一个玩具例子可以看清这个边界。

假设句子是：

`猫坐在窗台上看雨。`

- R 模式：挖掉“窗台上”，让模型补回来。
- S 模式：保留“猫坐在”，要求模型继续写后半句。
- X 模式：只保留“猫……雨”，中间大量删掉，让模型重建整句。

这三种训练信号显然不一样。第一种偏局部恢复，第二种偏顺序生成，第三种偏粗粒度语义重建。UL2 的目标就是把这三类信号统一到一套训练中，而不是从中选一个当唯一真理。

---

## 核心机制与推导

UL2 把单个去噪器写成一个参数化形式：

$$
\text{SpanCorrupt}(\mu, r, n)
$$

其中：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $\mu$ | 平均 span 长度 | 每次连续挖掉多长 |
| $r$ | corruption rate | 总共挖掉多少比例的 token |
| $n$ | span 数 | 一条样本里大致挖几段 |

这三个量决定“输入被破坏成什么样”。UL2 不是固定一个配置，而是混合 7 个配置。论文 Table 1 的核心可以概括为下表：

| 组别 | 典型配置 | 角色 | 大意 | 权重趋势 |
|---|---|---|---|---|
| R | $\mu=3,\ r=0.15$ | 短 span 去噪 | 接近 T5 式补空 | 较高 |
| R | $\mu=8,\ r=0.15$ | 中短 span 去噪 | 稍长的局部恢复 | 较高 |
| S | Prefix LM 形式 | 前缀生成 | 给前半段，预测后半段 | 约 20% |
| X | 大 span、较高破坏率 | 极端去噪 | 保留少量上下文做重建 | 较高 |
| X | 更少 span、单段更长 | 粗粒度重建 | 更接近长段恢复 | 较高 |
| X | 更高 corruption | 极端压缩输入 | 提升跨范围建模 | 中等 |
| X | 变体配置 | 补足多样性 | 防止目标单一化 | 中等 |

这里最重要的不是死记每个数字，而是理解三组角色分工：

- R 负责局部知识补全。
- S 负责顺序续写。
- X 负责在信息很少时做大范围恢复。

训练目标是这些去噪器损失的加权和。交叉熵是“预测分布与真实 token 之间的标准分类损失”。形式上可以写成：

$$
\mathcal{L} = \sum_{k=1}^{K} w_k \cdot \mathbb{E}_{(x,y)\sim D_k}\big[-\log p_\theta(y\mid \tilde{x}_k)\big]
$$

其中：

- $K$ 是去噪器配置数。
- $w_k$ 是第 $k$ 个配置的采样权重。
- $\tilde{x}_k$ 是按第 $k$ 种规则破坏后的输入。
- $y$ 是要恢复或生成的目标序列。

模式 token 则是“显式路由器”。路由器是“告诉模型当前该走哪类行为分布的信号”。例如：

- `[NLU]` 更接近理解导向。
- `[NLG]` 更接近生成导向。
- `[S2S]` 更接近一般序列到序列任务。

这件事看起来像 prompt engineering，但本质更强，因为模型在预训练阶段就已经学会“看到这个 token，该调用哪种损失下形成的行为模式”。

新手最容易误解的一点是：UL2 不是同时算三种 loss 再在推理时随便挑一个输出，而是在训练数据层面先采样一种 denoiser，把样本改写后送入同一个模型。也就是说，统一点在“参数共享 + 目标混合 + 显式模式选择”，不是多头模型，也不是专家并行。

---

## 代码实现

下面用一个最小可运行的 Python 例子说明 UL2 风格的数据管线。它不是完整训练器，但包含三件关键事：模式采样、样本破坏、模式 token 前缀。

```python
import random

MODES = {
    "R": {"token": "[NLU]", "mu": 3, "r": 0.15, "kind": "span"},
    "S": {"token": "[NLG]", "kind": "prefix"},
    "X": {"token": "[S2S]", "mu": 12, "r": 0.5, "kind": "extreme_span"},
}

WEIGHTS = {"R": 0.4, "S": 0.2, "X": 0.4}

def sample_mode(weights):
    modes = list(weights.keys())
    probs = list(weights.values())
    return random.choices(modes, probs, k=1)[0]

def r_denoise(tokens, mu=3, r=0.15):
    n_mask = max(1, int(len(tokens) * r))
    start = min(len(tokens) - 1, 1)
    end = min(len(tokens), start + mu, start + n_mask)
    corrupted = tokens[:start] + ["<extra_id_0>"] + tokens[end:]
    target = tokens[start:end]
    return corrupted, target

def s_denoise(tokens):
    pivot = max(1, len(tokens) // 2)
    corrupted = tokens[:pivot]
    target = tokens[pivot:]
    return corrupted, target

def x_denoise(tokens, mu=12, r=0.5):
    keep = max(1, len(tokens) - int(len(tokens) * r))
    left = max(1, keep // 2)
    right = len(tokens) - (keep - left)
    corrupted = tokens[:left] + ["<extra_id_0>"] + tokens[right:]
    target = tokens[left:right]
    return corrupted, target

def build_ul2_example(text, forced_mode=None):
    tokens = text.split()
    mode = forced_mode or sample_mode(WEIGHTS)
    spec = MODES[mode]

    if mode == "R":
        corrupted, target = r_denoise(tokens, spec["mu"], spec["r"])
    elif mode == "S":
        corrupted, target = s_denoise(tokens)
    else:
        corrupted, target = x_denoise(tokens, spec["mu"], spec["r"])

    model_input = [spec["token"]] + corrupted
    return {
        "mode": mode,
        "input_tokens": model_input,
        "target_tokens": target,
    }

toy = "UL2 unifies denoising prefix language modeling and extreme reconstruction"
example_r = build_ul2_example(toy, forced_mode="R")
example_s = build_ul2_example(toy, forced_mode="S")
example_x = build_ul2_example(toy, forced_mode="X")

assert example_r["input_tokens"][0] == "[NLU]"
assert example_s["input_tokens"][0] == "[NLG]"
assert example_x["input_tokens"][0] == "[S2S]"
assert len(example_r["target_tokens"]) >= 1
assert len(example_s["target_tokens"]) >= 1
assert len(example_x["target_tokens"]) >= 1
```

这个例子里：

- `build_ul2_example` 模拟“先采模式，再改写样本”。
- `input_tokens` 前面加模式 token，模拟训练和推理的一致接口。
- `target_tokens` 是要恢复或继续生成的部分。
- `assert` 用来验证最基本的行为是否成立。

如果把它映射到真实训练工程，流程通常是：

| 步骤 | 玩具例子做法 | 真实工程做法 |
|---|---|---|
| 模式采样 | `sample_mode` | 按预设权重在 data loader 中采样 |
| 输入改写 | Python 函数直接切 token | 先分词，再生成 sentinel/span mask |
| 标签构造 | 返回 `target_tokens` | 拼成 decoder labels，忽略 pad |
| 模型前缀 | 直接加 `[NLU]/[NLG]/[S2S]` | 加入词表并参与训练 |
| 优化目标 | 省略 logits | 标准 cross-entropy |

真实工程例子是 Flan-UL2。它不是重新设计架构，而是在 UL2 checkpoint 上继续做大规模 instruction tuning，把“多目标预训练获得的综合能力”转成“更好的人类指令遵循能力”。因此它能作为 20B 级别的通用推理基座，在问答、分类、解释、few-shot prompt 等场景里直接工作。

---

## 工程权衡与常见坑

UL2 最容易被错误实现的地方，不在模型结构，而在数据比例。

论文的一个关键结论是：S-Denoiser 不能太多，也不能没有。太多时，模型会向 prefix continuation 偏斜；完全没有时，few-shot 与生成相关能力会掉。经验上，S 维持约 20% 左右，而其余配给 R 和 X，更容易稳定。

可以把这个现象理解成能力预算分配问题：

| 配置倾向 | 好处 | 代价 |
|---|---|---|
| S 占比过高 | 续写更顺手 | 理解类、补全类任务容易波动 |
| 只有 R | 局部补空强 | 长程生成和 few-shot 风格不足 |
| 只有 X | 粗重建能力强 | 训练信号过粗，收敛更难 |
| R/S/X 混合 | 综合能力最好 | 数据管线更复杂 |

一个常见坑是把 UL2 简化成“多加几个 mask 比例”。这不够。UL2 真正重要的是“目标形态多样化”，包括 prefix 型任务和极端去噪任务。只改 `r=0.15/0.5`，但没有 S 模式，通常得不到论文里的行为。

第二个坑是推理时忽略模式 token。训练时有显式模式路由，推理时却只给普通文本，相当于把“你该用哪种行为分布”这个信号删掉。结果不是模型完全失效，而是控制力变弱，输出风格和任务适配性更不稳定。

第三个坑是把 UL2 的收益误读成“模型越大越该做复杂目标”。更准确的说法是：当你需要一个模型兼顾阅读理解、知识补全、问答生成、少样本提示时，目标混合更有价值；如果你只做单一长文本续写，prefix LM 可能已经够用，UL2 的额外复杂度未必值得。

第四个坑是评估方式不匹配。UL2 训练出的模型有模式切换能力，如果评估 prompt 不体现任务属性，比如本该走 `[NLG]` 却不给生成导向前缀，或者本该走 `[NLU]` 却当自由续写来测，结果会低估它的真实能力。

---

## 替代方案与适用边界

如果只看路线，UL2 的替代方案主要有三类。

| 方案 | 训练方式 | 适合什么 | 主要短板 |
|---|---|---|---|
| Prefix LM | 左到右生成 | 对话、续写、开放生成 | 双向理解信号弱 |
| Span Corruption | 挖空再恢复 | 表示学习、条件生成、补全 | few-shot 续写不一定最佳 |
| UL2 | 多 denoiser 混合 | 同时覆盖理解、生成、重建 | 数据设计与调参更复杂 |

Prefix LM 的优点是目标简单，训练和推理接口统一，特别适合大规模生成式产品。Span corruption 的优点是对 encoder-decoder 任务友好，做翻译、摘要、抽取式到生成式转换时很自然。UL2 的优势则在于“能力面更宽”，尤其适合你不知道线上任务会落在哪一端的时候。

但 UL2 也有明确边界。

第一，它不自动等于更强的对话模型。聊天能力更多来自指令微调、偏好优化、推理数据，而不是预训练目标本身。第二，它不是参数效率最优解。一个只做续写的系统，未必需要承担多目标训练的复杂度。第三，它要求更严格的数据构造与实验设计，否则很容易只学到“复杂”，没学到“统一”。

从工程角度，如果你的目标是做一个“多任务底座”，后面还要接 instruction tuning、few-shot、CoT prompting，那么 UL2 比单一目标更合适。如果目标是极致吞吐的续写服务，decoder-only 路线往往更直接。如果目标是传统 seq2seq 任务且任务边界稳定，T5 风格目标仍然足够有效。

Flan-UL2 的意义也在这里。它说明 UL2 不是停留在论文设想，而是可以作为高质量 checkpoint 被继续指令微调，并在公开基准上取得竞争力结果。对工程团队来说，这意味着 UL2 更像“强基础模型配方”，而不是一个只能在论文里成立的技巧。

---

## 参考资料

- Tay et al.，《UL2: Unifying Language Learning Paradigms》，ICLR 2023，OpenReview 论文页面与 PDF：https://openreview.net/forum?id=6ruVLB727MC 、https://openreview.net/pdf?id=6ruVLB727MC
- Hugging Face 模型卡，`google/flan-ul2`：https://huggingface.co/google/flan-ul2
- Yi Tay，`A New Open Source Flan 20B with UL2`：https://www.yitay.net/blog/flan-ul2-20b
