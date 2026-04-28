## 核心结论

参数高效微调，英文是 Parameter-Efficient Fine-Tuning，简称 PEFT，白话讲就是“尽量不动大模型主体，只训练一小块额外参数，让模型学会新任务”。全参数微调则相反，它会直接更新基座模型的全部参数。

先给总判断：PEFT 不是“缩水版微调”，而是一类明确追求参数、显存、存储效率的方法族。它的核心优势不是表达能力最强，而是“用极少的可训练参数完成任务适配”。全参数微调的表达空间更完整，因此在一些任务上仍可能拿到更高上限，但代价是训练和部署都更重。

一句话类比：全参数微调像“重做整台机器的校准”，PEFT 像“给机器加一个可替换的外挂模块”。

最关键的结论有两个。

第一，PEFT 在很多任务上能逼近全参数微调，尤其当基座模型足够大、任务不是极端分布偏移、训练设置合理时，性能差距常常会缩小到可接受范围。  
第二，PEFT 不保证在所有任务上都和全参一样好。数据规模、任务类型、目标层选择、训练步数、学习率、模型规模，都会改变结论。

下面先看一个最小数字例子。假设一个模型总参数量是 100M，也就是 1 亿参数。

| 方案 | 训练参数量 | 占总参数比例 | 显存开销 | 存储开销 | 部署方式 | 典型方法 | 适用场景 |
|---|---:|---:|---|---|---|---|---|
| 全参数微调 | 100M | 100% | 高 | 每个任务存整模型 | 单任务单模型部署 | Full FT | 极致效果、资源充足 |
| PEFT | 100k | 0.1% | 低到中 | 每个任务只存 adapter | 基座模型 + adapter 切换 | LoRA、Prefix Tuning、Adapter | 多任务、低成本迭代 |

如果只训练 100k 个参数，那么占比就是：
$$
\frac{100000}{100000000}=0.001=0.1\%
$$

这就是常见的“0.1% vs 100%”量级差异。它不是微小优化，而是训练对象规模直接缩小了三个数量级。

---

## 问题定义与边界

为了避免概念混乱，先统一记号。

设基座模型参数为 $\theta_0$。微调后模型参数记为 $\theta$。统一写法可以写成：

$$
\theta = \theta_0 + \Delta\theta
$$

这里的 $\Delta\theta$ 可以理解为“为了适配任务而引入的变化量”。

但全参数微调和 PEFT 在这条公式里的含义不同。

- 全参数微调：$\Delta\theta$ 作用在全部原参数维度上，训练时 $\theta_0$ 的全部参数都参与反向传播并被更新。
- PEFT：通常冻结 $\theta_0$，只训练少量新增参数，或者只训练某种低维增量表示。也就是说，主体参数不动，变化集中在小规模模块里。

比较对象也要说清楚。本文比较的是“同一个预训练模型，在同一个下游任务上，采用两种微调路线”的差异，不是在比较不同模型、不同数据集、不同训练预算。否则结论会被混淆。

给新手一个判定题：

- 一个 10 亿参数模型，只训练 LoRA 的几百万参数，这属于 PEFT。
- 一个 10 亿参数模型，让 10 亿参数都参与梯度更新，这属于全参数微调。

边界表如下。

| 判断项 | 全参数微调 | PEFT |
|---|---|---|
| 是否冻结基座参数 | 否 | 通常是 |
| 是否新增可训练模块 | 不一定需要 | 通常需要 |
| 是否改变原参数形状 | 一般不改形状，直接更新值 | 通常不改原权重形状，只加增量或前缀 |
| 是否需要单独存 adapter | 否 | 是，通常单独保存 |
| 是否支持多任务切换 | 较弱，每任务一整份模型 | 强，切换 adapter 即可 |

这里的 adapter，白话讲就是“附着在基座模型上的小型任务插件”。

所以本文的核心边界不是“哪个方法绝对更强”，而是“在相同基座模型上，少量参数适配能否用更低成本换到足够接近的效果”。

---

## 核心机制与推导

全参数微调的机制最直接。它对模型中的全部权重求梯度，然后做参数更新。好处是表达能力完整，因为任何层、任何参数都可以朝任务目标调整。代价也很直接：训练显存高、优化状态占用大、分布式通信重、每个任务都要保存一整份模型。

PEFT 的思路是反过来：既然大模型已经学到了通用能力，那么下游任务也许不需要重新改写全部权重，只需要增加一小块“任务相关增量”即可。

### 玩具例子：LoRA 的参数量为什么会小很多

LoRA，Low-Rank Adaptation，白话讲就是“把原本很大的权重改变量，限制成两个小矩阵相乘的低秩形式”。

对于一个线性层：
$$
W_0 \in \mathbb{R}^{d_{out}\times d_{in}}
$$

全参数微调直接学习整个 $W$。LoRA 则写成：

$$
W = W_0 + \Delta W
$$

并且约束：

$$
\Delta W = BA,\quad
B\in\mathbb{R}^{d_{out}\times r},\quad
A\in\mathbb{R}^{r\times d_{in}},\quad
r \ll \min(d_{out}, d_{in})
$$

于是可训练参数量从：
$$
d_{out}d_{in}
$$
变成：
$$
r(d_{out}+d_{in})
$$

设 $W_0$ 是一个 $1000\times1000$ 的矩阵。

- 全参数微调：训练参数量是 $1000\times1000=1{,}000{,}000$
- LoRA 若取 $r=4$：训练参数量是 $4(1000+1000)=8000$

对比表如下。

| 方法 | 参数形式 | 可训练量 | 是否改动原权重 | 推理时影响 |
|---|---|---:|---|---|
| 全参数微调 | 直接更新 $W$ | $d_{out}d_{in}$ | 是 | 通常无额外结构 |
| LoRA | 学习 $\Delta W=BA$ | $r(d_{out}+d_{in})$ | 否，原权重冻结 | 可合并或在线叠加 |
| Prefix Tuning | 学习前缀表示 $P$ | 与前缀长度成正比 | 否 | 会引入额外前缀计算或缓存 |

LoRA 为什么常常有效？一个常见解释是：下游任务真正需要的更新方向，往往落在较低维的子空间里，不必为每个权重都分配独立自由度。这个结论不是数学定理，但在很多任务上成立得足够好。

### Prefix Tuning 的思路

Prefix Tuning，白话讲就是“不给模型改主干权重，而是给注意力机制塞一段可学习的虚拟前缀”。

可把前缀写成：
$$
P\in\mathbb{R}^{m\times d}
$$

其中 $m$ 是前缀长度，$d$ 是隐藏维度。模型会把这段前缀看成若干“看不见的提示 token”，在注意力计算时参与 key/value 或等价表示的构造。这样模型虽然主干冻结，但会被这段前缀引导到更适合任务的响应方式。

它和 LoRA 的差别在于：

- LoRA 改的是层内权重增量。
- Prefix Tuning 改的是模型处理输入时的上下文条件。

### 真实工程例子：同一基座维护 20 个行业版本

假设企业要基于同一个 70B 基座模型做 20 个行业版本，分别服务法律、医疗、金融、客服等领域。

如果走全参数微调：

- 每个行业都要训练并保存一整份模型。
- 发布一个小改动，也可能要重新管理大模型版本。
- 存储、回滚、灰度、AB 测试都更重。

如果走 LoRA 或 Prefix Tuning：

- 基座模型只保留一份。
- 每个行业只维护一份小 adapter。
- 部署时按请求路由切换 adapter，版本管理明显简单。

这就是 PEFT 在工程上经常被采用的原因。它不是论文里的技巧，而是直接影响存储、训练队列、发布流程和多租户服务架构。

---

## 代码实现

下面用 Hugging Face 生态展示最小实现思路。`transformers` 负责加载基座模型，`peft` 负责注入 LoRA 或 Prefix Tuning。主体训练流程和常规微调基本一致，区别主要在“冻结哪些参数”和“保存哪些权重”。

先看一个可运行的玩具代码，用来验证 LoRA 参数量公式。

```python
def lora_param_count(d_in: int, d_out: int, rank: int) -> int:
    return rank * (d_in + d_out)

def full_param_count(d_in: int, d_out: int) -> int:
    return d_in * d_out

d_in, d_out, rank = 1000, 1000, 4

full = full_param_count(d_in, d_out)
lora = lora_param_count(d_in, d_out, rank)

assert full == 1_000_000
assert lora == 8_000
assert lora / full == 0.008

print({"full": full, "lora": lora, "ratio": lora / full})
```

第一步是加载模型并冻结参数。冻结，白话讲就是“告诉优化器，这些参数不要更新”。

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

for param in model.parameters():
    param.requires_grad = False

assert all(not p.requires_grad for p in model.parameters())
```

第二步是注入 LoRA 配置。这里的 `target_modules` 表示要把 LoRA 加到哪些线性层上。不同模型命名不同，工程里不能照抄。

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

第三步是统计可训练参数量。这个步骤很重要，因为很多新手以为自己在做 PEFT，结果目标层没配对，或者某些参数没冻结，最后实际上训练了远超预期的参数。

```python
trainable = 0
total = 0

for p in model.parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()

ratio = trainable / total
assert trainable > 0
assert ratio < 0.1  # 这里只是示意：PEFT 通常远小于全参

print({
    "trainable": trainable,
    "total": total,
    "ratio": ratio,
})
```

训练脚本主体通常不用大改，还是常规的 `Trainer` 或自定义训练循环。关键差异是保存时只保存 adapter，而不是整模型。

```python
# 训练结束后，仅保存 adapter 权重
model.save_pretrained("./lora_adapter")
```

推理时再把 adapter 挂回基座模型即可。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = PeftModel.from_pretrained(base_model, "./lora_adapter")
model.eval()
```

对于 Prefix Tuning，整体流程类似，只是注入的不是低秩矩阵，而是前缀配置。对新手来说，最重要的理解不是 API 名字，而是这条主线：加载模型 -> 冻结主体 -> 注入小模块 -> 只训练小模块 -> 只保存小模块。

---

## 工程权衡与常见坑

很多人一看到“只训练 0.1% 参数”就直接下结论，认为它一定更优。这是不严谨的。PEFT 的优势是成本结构，不是无条件效果优势。

常见坑与规避如下。

| 常见坑 | 现象 | 为什么会出问题 | 规避建议 |
|---|---|---|---|
| 只看参数占比 | 认为参数越少越先进 | 参数少不等于效果稳 | 同时看任务指标、延迟、训练稳定性 |
| rank 过小 | LoRA 学不动 | 表达容量不足，容易欠拟合 | 从中等 rank 起调参，再压缩 |
| 训练条件不一致 | 对比结论失真 | 学习率、步数、数据清洗差异会盖过方法差异 | 保持相同数据和预算做对照 |
| 忽略推理成本 | 线上延迟变高 | Prefix 会增加前缀长度和缓存负担 | 离线评估吞吐和 KV cache 占用 |
| 低资源任务误判 | 少样本结果波动大 | 任务本身噪声高，结论不稳 | 多随机种子、多数据切分评估 |

再看一个对照情境。两个团队都说自己“用了 LoRA”。

- 团队 A：只在 attention 的少数层加 LoRA。
- 团队 B：在 attention + MLP 的更多层加 LoRA。

即使两边可训练参数量接近，最终效果也可能差很多。原因不是“LoRA 不稳定”，而是目标层不同，模型能被调整的功能区域不同。对语言模型来说，attention 和 MLP 承担的表示角色并不完全一样。

还有几个容易被忽略的点。

第一，LoRA 的 rank 不是越小越好。rank 太小会欠拟合，模型没有足够自由度表达任务差异。  
第二，Prefix Tuning 会增加前缀长度，可能影响注意力开销和 KV cache。  
第三，“PEFT 一定接近全参”不是通用规律。模型越大时，这种说法往往更容易成立；模型较小、任务分布偏移很大、需要深度重写行为时，差距可能更明显。  
第四，训练步数也很关键。PEFT 参数少，不代表一定更快收敛到最好点。有些任务反而需要更仔细的学习率和步数设置。

---

## 替代方案与适用边界

PEFT 不是单一方法，而是一整个方法谱系。它们共同目标是“少量参数适配”，区别在于把可训练能力放在哪里。

| 方法 | 优点 | 缺点 | 存储方式 | 适用场景 |
|---|---|---|---|---|
| Prompt Tuning | 参数极少，接入简单 | 对模型规模更敏感，小模型效果常弱 | 软提示参数 | 输入条件控制明显的任务 |
| Prefix Tuning | 不改主干权重，生成任务常见 | 可能增加推理开销 | 前缀参数/缓存 | 条件生成、多任务切换 |
| Adapter | 模块清晰，任务隔离好 | 会引入额外层结构 | 小型插入模块 | 长期维护多个任务版本 |
| LoRA | 参数效率高，实现成熟 | 目标层和 rank 选择影响大 | 低秩增量权重 | 当前最常见的工程方案 |
| 全参数微调 | 表达能力完整，上限最高 | 成本最高，版本最重 | 整模型权重 | 高价值单任务、追求极限效果 |

新手版选择规则可以直接写成两句。

- 如果你要同时维护 20 个行业模型，优先考虑 LoRA、Prefix Tuning 或 Adapter。
- 如果你只有一个高价值任务，数据量足够、资源充足、而且追求极致指标，可以认真评估全参数微调。

还可以按三个维度做判断。

| 条件 | 更适合 PEFT | 更适合全参数微调 |
|---|---|---|
| 任务数量 | 多任务、多租户 | 单任务、长期固定 |
| 数据量 | 中小规模常优先试 PEFT | 大规模高质量数据更值得试全参 |
| 预算 | 显存、存储、训练窗口受限 | 资源充足 |
| 目标 | 快速迭代、低成本上线 | 冲极限效果 |
| 分布偏移 | 中等偏移 | 偏移极大，需要深度改写能力 |

从方法边界看，PEFT 的真正价值是“把任务适配从大模型重训练，变成小模块管理问题”。这对工业系统特别重要。但如果任务要求模型内部表示被大范围重构，例如领域差异极大、标签质量很高、训练预算也足够，那么全参数微调仍然可能更合适。

---

## 参考资料

原始论文建议先读 LoRA，再读 Prefix Tuning 和 Prompt Tuning。这样先理解“低秩增量”和“软提示/前缀”的两条主线。之后再看系统研究，理解“为什么有时接近全参，有时又有差距”。最后读工程文档，把论文结论落到真实训练脚本和部署流程上。

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685)
2. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://huggingface.co/papers/2101.00190)
3. [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
4. [When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method](https://deepmind.google/research/publications/49667/)
5. [Hugging Face Transformers PEFT 文档](https://huggingface.co/docs/transformers/main_classes/peft)
6. [Hugging Face PEFT GitHub 仓库](https://github.com/huggingface/peft)
7. [Microsoft LoRA GitHub 仓库](https://github.com/microsoft/LoRA)
