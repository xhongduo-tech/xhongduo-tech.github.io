## 核心结论

DeepSeek-V1 67B 的核心设计，不是“把 LLaMA 2 放大一点”，而是在相近参数预算下，主动选择“更深、更窄、更强调专业数据”的路线。这里的“更深”指 Transformer 层数更多，“更窄”指每层前馈网络宽度更小。具体做法是：95 层 Dense Transformer，采用 Pre-Norm、RMSNorm、SwiGLU，以及 GQA（Grouped Query Attention，指多个 query 头共享更少的 KV 头）结构，用较小的单层 FFN 换来更深的网络深度。

和 LLaMA 2 70B 相比，DeepSeek-V1 67B 的典型差异可以概括为“窄而深 vs 宽而浅”。两者都属于标准自回归 Transformer，但 DeepSeek-V1 把参数更多分配到层数，而不是分配到每层的前馈宽度。这一选择的目标很明确：在保持大模型容量的同时，强化逻辑链条更长的任务，尤其是数学推理和代码生成。

另一个关键点是训练数据策略。DeepSeek-V1 不是只做通用文本堆料，而是在 2T tokens 规模上，明确加入英文、中文、数学、代码等多域混合，并通过 remix 控制这些领域的占比。这里的“remix”可以理解为重新配料，即不是按原始互联网分布直接训练，而是人为提高数学和编程样本比例，避免专业能力被普通自然语言样本稀释。

下面这张表先给出最重要的结构对比：

| 模型 | 参数量 | 层数 | $d_{model}$ | $d_{ff}$ | 注意力 | 训练 token |
|---|---:|---:|---:|---:|---|---:|
| LLaMA 2 70B | 约 70B | 80 | 8192 | 28672 | GQA | 约 2T |
| DeepSeek-V1 67B | 约 67B | 95 | 8192 | $\frac{8}{3}\times8192\approx21845$ | GQA，64 query / 8 KV | 约 2T |

结论可以直接落到任务层面：如果目标是普通对话、参数复用和通用语言生成，宽而浅的结构足够有效；如果目标是 few-shot 数学、代码和复杂推理，DeepSeek-V1 这种更深网络加专业数据强化的路线更有针对性。

---

## 问题定义与边界

这篇文章讨论的是 DeepSeek-V1 67B 的基础架构和预训练策略，不讨论后续指令微调、RLHF，也不讨论 MoE。这里的“Dense 架构”指每一层参数都会被完整激活，不像 MoE 那样每次只路由到部分专家。因为用户常把 DeepSeek 后续模型和早期 V1 混在一起，所以边界要先切清楚。

它要解决的问题可以拆成两层：

| 任务类别 | 主要要求 | 仅靠通用语料是否足够 | DeepSeek-V1 的针对性设计 |
|---|---|---|---|
| 普通对话 | 流畅表达、常识覆盖 | 大体足够 | 不是主要优化点 |
| 数学推理 | 多步符号运算、链式推导 | 不足 | 提高数学数据占比、增加深度 |
| 代码生成 | 语法约束、长依赖、模式迁移 | 不足 | 强化代码语料、提升层数 |
| 多语言 | 中英混合、跨语种迁移 | 有风险 | 显式混合中英文数据 |

一个常见误解是：只要把注意力改成 GQA，推理和代码能力就会自然变强。这个结论不成立。GQA 的主要价值是减少 KV cache 压力和注意力计算/存储开销，它本身不是“推理增强器”。如果只是把标准多头注意力改成 GQA，但层数仍停留在 80 层，模型往往只是更省，不一定更强。DeepSeek-V1 的关键是在 GQA 省下的预算上，继续把网络加深到 95 层。

玩具例子可以这样理解。假设你有固定预算能建两栋楼：

| 方案 | 楼层数 | 每层面积 | 总建筑面积 |
|---|---:|---:|---:|
| 宽而浅 | 80 | 28672 | 2293760 |
| 窄而深 | 95 | 21845 | 2075275 |

这不是精确参数公式，只是帮助理解“单层变窄，换更多层”。如果任务需要更长的逐层变换链条，95 层可能比 80 层更有优势；如果任务更依赖单层大容量映射，较宽 FFN 也可能更合适。

真实工程例子是代码助手。对一个 IDE 内嵌模型来说，用户提问往往不是简单续写，而是“看懂已有函数 -> 理解调用链 -> 生成补丁 -> 保持语法与风格一致”。这种任务依赖多步中间表示变换。更深的网络不保证一定更好，但在相近参数预算下，它更像是在给模型增加“推导步数”。

---

## 核心机制与推导

DeepSeek-V1 的核心机制可以拆成三部分：更窄的 FFN、GQA、以及更稳定的归一化与激活组合。

先看 FFN。设隐藏维度为 $d_{model}=8192$，DeepSeek-V1 采用：

$$
d_{ff} = \frac{8}{3} \times d_{model}
$$

代入可得：

$$
d_{ff} = \frac{8}{3}\times8192 \approx 21845
$$

而 LLaMA 2 70B 常见配置是：

$$
d_{ff}=28672
$$

这意味着 DeepSeek-V1 每层前馈更窄。前馈网络可以粗略理解为“每层内部做非线性变换的主计算块”。它越宽，单层表达容量越大；它越窄，单层计算越省。DeepSeek-V1 把省下来的预算换成更多层数，这就是“参数密度向深度倾斜”。

再看 GQA。标准多头注意力里，每个 query 头通常对应一组自己的 key 和 value。GQA 改成“更多 query 头共享更少的 KV 头”。DeepSeek-V1 的典型配置是 64 个 query 头、8 个 KV 头，即每 8 个 query 头共享 1 组 KV 头。它的直观作用是减少缓存和访存压力，特别是在长上下文推理时更明显。

注意力基本形式仍然是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

区别只在于头的组织方式。若 $n_{head}=64,\ n_{kv}=8$，则 query 头数与 KV 头数的关系是：

$$
\text{group size}=\frac{n_{head}}{n_{kv}}=\frac{64}{8}=8
$$

也就是 8 个 query 头共享一组 KV。这样做不会改变注意力公式本身，但会改变参数量、缓存形状和运行效率。

最后看稳定性组合。Pre-Norm 指“先归一化，再进入子层”；RMSNorm 是只按均方根缩放的归一化方式；SwiGLU 是一种门控前馈激活，能在参数效率和训练稳定性之间取得较好平衡。三者组合的目标很现实：95 层已经很深，如果层间信号和梯度不稳定，模型很难顺利训练完。

一个简化的层结构可以写成：

$$
x_{l+1} = x_l + \text{Attention}(\text{RMSNorm}(x_l))
$$

$$
x_{l+2} = x_{l+1} + \text{SwiGLU-FFN}(\text{RMSNorm}(x_{l+1}))
$$

它表达的是 Pre-Norm 残差结构：先做 RMSNorm，再做注意力或前馈，再加回残差。

如果只看机制，可以把 DeepSeek-V1 记成一句话：用更窄 FFN 腾预算，用 GQA 省 KV 成本，用 Pre-Norm + RMSNorm + SwiGLU 把 95 层训练稳住。

---

## 代码实现

下面给一个可运行的玩具实现，重点不是复现 67B，而是把结构关系说明白：Pre-Norm、RMSNorm、GQA、SwiGLU 这四件事如何放进一个 Transformer block。

```python
import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.w1(x))
        value = self.w2(x)
        return self.out(gate * value)

class GQAAttention(nn.Module):
    def __init__(self, dim=128, n_heads=8, n_kv_heads=2):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.group_size = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 把较少的 KV 头复制到每个 query 组
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim=128, n_heads=8, n_kv_heads=2, ff_mult=8/3):
        super().__init__()
        hidden_dim = int(dim * ff_mult)
        self.norm1 = RMSNorm(dim)
        self.attn = GQAAttention(dim, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-Norm Attention
        x = x + self.ffn(self.norm2(x))    # Pre-Norm FFN
        return x

x = torch.randn(2, 4, 128)
block = TransformerBlock()
y = block(x)

assert y.shape == x.shape
assert block.attn.group_size == 4
assert int(8192 * 8 / 3) == 21845
print("ok")
```

这个实现里最容易漏掉的一步，是 `repeat_interleave` 对 KV 头的扩展。如果你直接按标准多头注意力写 `K` 和 `V`，那就不是 GQA，而是普通 MHA。GQA 的关键不是公式换了，而是“query 头更多，KV 头更少，再按组复用”。

真实工程里，一个训练脚本通常还要把学习率调度和数据 remix 放进去。伪代码如下：

```python
def sample_domain():
    # 例子：人为提高数学/代码占比
    domains = ["english", "chinese", "math", "code"]
    probs = [0.50, 0.20, 0.15, 0.15]
    return random_choice(domains, probs)

def lr_schedule(step, warmup, s1, s2, base_lr):
    if step < warmup:
        return base_lr * step / warmup
    if step < s1:
        return base_lr
    if step < s2:
        return base_lr * 0.3
    return base_lr * 0.1
```

参数关系也可以先用一张表记住：

| 参数 | 含义 | DeepSeek-V1 典型值 |
|---|---|---|
| `d_model` | 隐藏维度，指 token 表示长度 | 8192 |
| `n_heads` | query 头数 | 64 |
| `n_kv_heads` | key/value 头数 | 8 |
| `d_ff` | FFN 中间层维度 | 约 21845 |
| `n_layers` | Transformer 层数 | 95 |

---

## 工程权衡与常见坑

第一类权衡，是“省出来的预算怎么花”。GQA 会让推理和训练更省，但省下预算以后有多种用法：可以扩上下文，可以扩 batch，也可以加深网络。DeepSeek-V1 选择加深网络。这个选择和它的目标一致，因为数学与代码任务常常需要更长的逐层抽象链条。

第二类权衡，是数据分布控制。互联网原始语料里，自然语言远多于数学和代码。如果直接按原始比例训练，模型的通用表达可能很好，但在 GSM8K、HumanEval、MBPP 这类任务上不一定理想。这里的坑不是“数据总量不够”，而是“有效任务分布不对”。

下面用一个示意表说明 remix 的影响。数字是说明趋势，不代表官方逐项公布值：

| 数据设定 | 数学 token 占比 | 代码 token 占比 | GSM8K few-shot | HumanEval |
|---|---:|---:|---:|---:|
| Vanilla mix | 2% | 4% | 较低 | 较低 |
| 轻度 remix | 5% | 8% | 回升 | 回升 |
| 强化 remix + 稳定 LR | 8% | 10% | 更稳定 | 更稳定 |

这张表想说明一个工程事实：数学和代码能力对样本占比很敏感。假设数学 token 从 5% 掉到 2%，模型依然会“懂语言”，但在需要严格推导的题目上更容易失分。原因并不神秘，就是训练目标里这类模式出现得不够频繁。

第三类坑，是把“结构优化”和“能力提升”混为一谈。GQA、RMSNorm、SwiGLU 这些设计首先是工程手段，不是魔法开关。它们提供的是更好的参数利用率、稳定性和效率。真正把这些优势转成 benchmark 分数的，是结构、深度、数据和训练调度一起配合。

如果把这些经验压缩成工程步骤，可以记成：

1. 先确定参数预算，再决定“宽”还是“深”。
2. 若使用 GQA，要同步考虑层数与 FFN 宽度，不要只改注意力。
3. 数据混合不能只看总 token，要看任务域占比。
4. 深层模型训练要配稳定归一化和分阶段学习率。
5. 验证指标不能只看通用困惑度，要单独看数学和代码集。

---

## 替代方案与适用边界

DeepSeek-V1 的路线并不是唯一正确路线，而是在“67B 左右、关注数学和代码、允许更深网络”的条件下很合理。换个目标，最优解就可能变了。

先看两类路线的对照：

| 方案 | 优点 | 缺点 | 更适合的任务 |
|---|---|---|---|
| 窄而深 | 更利于多步推理、参数利用更偏向深度 | 训练更难、推理链更长 | 数学、代码、复杂 few-shot |
| 宽而浅 | 单层容量大、实现更稳妥 | 深层推理链较短 | 普通对话、通用生成、低风险部署 |

如果业务目标是“低延迟回答普通问答”，LLaMA 2 70B 这种更传统的宽而浅路线会更稳，工程上也更容易复用已有优化。如果业务目标是“在不做大量任务微调的前提下，提高数学与代码基准”，DeepSeek-V1 这种路线更有针对性。

还有一些现实边界不能忽略：

| 约束条件 | 更合适的选择 |
|---|---|
| 训练不稳定、基础设施一般 | 减少层数，适当增加 `d_ff` |
| 推理显存敏感 | 保留 GQA，压缩 KV cache |
| 延迟极度敏感 | 选较浅模型，配量化或蒸馏 |
| 专业任务少、通用对话多 | 不必过度提高数学/代码占比 |

如果无法承受 95 层训练成本，一个可行替代是回到更保守的 FFN 比例，例如接近 $4\times d_{model}$，同时减少层数，再辅以少量 depth drop 或训练期稳定技巧。这样通常拿不到 DeepSeek-V1 那种“深度优先”的收益，但能降低训练风险。

所以适用边界可以归纳为一句话：当你明确需要“多语言 + 数学/代码 + 大参数 Dense 模型”时，DeepSeek-V1 的设计有明显价值；当你主要关心部署成本、延迟和通用问答时，未必值得走这么深。

---

## 参考资料

| 来源 | 内容摘要 | 主要用途 |
|---|---|---|
| dsdanielpark.github.io 对 DeepSeek 67B 的整理 | 汇总 DeepSeek-V1 67B 的结构、层数、GQA、训练 token 与 benchmark | 用于把握整体架构与公开指标 |
| deep-seek.chat 相关文章与社区整理 | 训练数据 remix、数学和代码强化的经验性说明 | 用于理解数据配比思路 |
| LLaMA 2 官方论文与模型卡 | LLaMA 2 70B 的层数、隐藏维度、GQA、训练规模 | 用于对照“宽而浅”路线 |
| Hugging Face 上 DeepSeek 系列 README | 后续模型评测与家族演进 | 用于确认 DeepSeek 系列后续方向，不与 V1 混淆 |

1. DeepSeek 67B 结构整理页面：用于核对 95 层、Pre-Norm、RMSNorm、SwiGLU、GQA 等要点。
2. DeepSeek 相关社区资料：用于理解 2T tokens、多语言、数学和代码 remix 的训练策略。
3. LLaMA 2 官方资料：用于核对 70B 的 80 层、8192 隐藏维度与 28672 FFN 宽度。
4. DeepSeek 系列 README 与评测页：用于把 V1 放回整个产品线中理解，避免把后续版本特性误记到 V1。
