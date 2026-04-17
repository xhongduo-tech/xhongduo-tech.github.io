## 核心结论

Gemini 1.5 Pro 的效率设计，核心不是“参数更少”，而是“每个 token 真正参与计算的参数更少”。MoE，Mixture of Experts，中文常译为“专家混合”，白话讲就是把一个大前馈层拆成很多专门子网络，再让一个调度器只挑少数几个来算。

对零基础读者，最重要的结论有两条。

第一，Gemini 1.5 Pro 之所以能把多模态输入、超长上下文和较高质量放在一起，依赖的是稀疏激活。稀疏激活，意思是“大模型里大部分参数在某个 token 上并不工作”。如果一个层里有很多专家，但每次只调用 2 个，那么总参数池可以很大，而单次推理成本仍然受控。

第二，公开资料与分析普遍描述 Gemini 1.5 Pro 的 MoE 设计接近“共享专家 + 大量可选专家”的思路。一个常见表述是：1 个共享专家负责通用能力，128 个可选专家负责更细粒度的专长，每个 token 只激活“共享专家 + 1 个私有专家”，于是单 token 激活比例约为

$$
\frac{2}{129}\approx 1.55\%
$$

这比小规模 MoE 更激进。它的意义不是炫耀“只开了 1.6%”，而是在百万级甚至更长上下文里，把每个 token 的边际成本压到足够低，使长文档、长视频、长代码仓的推理不至于线性爆炸到不可用。

下面先看一个总览表。

| 方案 | 专家总数 | 每 token 激活数 | 活跃/总专家比例 | 上下文能力倾向 |
|---|---:|---:|---:|---|
| 稠密前馈层 | 1 | 1 | 100% | 短到中等上下文，结构简单 |
| 所有专家都激活的“伪 MoE” | 129 | 129 | 100% | 几乎失去 MoE 的成本优势 |
| 稀疏 top-k MoE | 129 | 2 | 1.55% | 更适合超长上下文 |
| Mixtral 8x7B 风格 top-2 | 8 | 2 | 25% | 长上下文可用，但单位 token 激活更高 |

玩具例子可以这样理解：你有 129 位老师，1 位负责所有学生都要学的基础课，128 位分别擅长数学、代码、图像、法律、医学等细分方向。一个问题来了，不是把 129 位老师全都叫来，而是先上基础课老师，再额外叫 1 位最相关的老师。这样总师资很大，但每次上课的成本很低。

---

## 问题定义与边界

问题定义很直接：如果模型既要处理文本，又要处理图像、代码、长 PDF、长视频字幕，还要支持 1M 以上 token 的上下文，那么传统稠密 Transformer 很容易在推理成本上失控。

这里的“稠密”，白话讲就是“每一层的主要子网络几乎都要完整执行一次”。如果上下文从 8K 提升到 1M，注意力本身已经很贵；如果前馈层还是每个 token 都走完整大网络，总成本会继续膨胀。Gemini 1.5 Pro 要解决的就是：在总参数量级很大的前提下，把单 token 的实际计算压低。

可以把单 token 的前馈激活成本写成一个近似式：

$$
C_{\text{token}} \propto \#\text{active experts} \times \text{expert size}
$$

如果总共有很多专家，但每次只激活很少几个，那么 $C_{\text{token}}$ 主要取决于“激活了几个”，而不是“总共养了多少个专家”。

这篇文章的边界也需要说清楚。

1. 讨论重点是推理效率，不展开训练系统、数据配比和多模态编码器细节。
2. 讨论重点是 MoE 前馈层的稀疏激活，不把注意力优化、KV cache、分布式并行混在一起。
3. 关于 Gemini 1.5 Pro 的具体层数、每层路由实现、共享专家是否在所有层都同样存在，公开资料并未完全披露；文中的“共享专家 + 128 可选专家”属于公开分析中常见的结构描述，应理解为高可信的工程画像，不应误读为逐层实现细节的完整公开文档。

对新手，一个边界感很重要：MoE 不是“白送参数”。它只是把“总参数量”和“每次实际计算量”拆开。你可以拥有很大的参数池，但如果路由失衡、通信开销过高，系统照样会慢。

再看一个更贴近直觉的例子。假设你在处理一本几百页的 PDF。错误做法是：每读一个 token，都让所有专家都过一遍。这样成本等价于把一本书送给全公司所有人逐页审稿。Gemini 1.5 Pro 的思路是：当前 token 先走共享专家，再让路由器只挑一个最相关的私有专家。这样不是“人人都看”，而是“找最适合的人看”。

真实工程例子则更明确。Gemini 1.5 技术报告里展示了长上下文“needle in a haystack”类测试，也包括长文档、长代码和长视频检索。其意义不是单项 benchmark 漂亮，而是说明：当输入规模到 1M token 甚至更高时，模型仍能维持可工作的推理路径。没有稀疏 MoE，这类输入规模会更难落到可用成本区间。

---

## 核心机制与推导

MoE 层一般由两部分组成：gating 网络和 experts。gating 网络，中文常叫“门控网络”或“路由网络”，白话讲就是“负责决定该问谁”；experts，专家网络，白话讲就是“各自擅长一类模式的前馈子网”。

设输入为 $x$，第 $i$ 个专家的输出为 $E_i(x)$，gating 网络给出的权重为 $G_i(x)$。标准稀疏 MoE 的输出可以写成：

$$
y=\sum_{i\in \text{TopK}(G(x))} G_i(x)\cdot E_i(x)
$$

这里的 TopK，意思是“只保留分数最高的 $k$ 个专家”。这就是稀疏性的来源。不是所有专家都算，而是只有得分最高的少数几个参与。

如果把它展开成 Gemini 1.5 Pro 常见的结构化理解，可以写成：

$$
y = \alpha_{\text{shared}} E_{\text{shared}}(x) + \alpha_{j} E_{j}(x)
$$

其中共享专家 $E_{\text{shared}}$ 始终参与，$E_j$ 是从 128 个私有专家中选出的一个，$\alpha_{\text{shared}}$ 与 $\alpha_j$ 是门控分数归一化后的权重。白话讲：每个 token 都先经过“通用专家”，然后再补一个“最懂当前内容的专家”。

这个设计有三个直接收益。

第一，保底能力更稳定。共享专家相当于一个“公共底座”，能减少因为私有专家选错而导致的信息断裂。对多模态场景尤其有用，因为文本、图像描述、代码片段之间总有一部分跨模态通用模式。

第二，激活比极低。若总共 129 个专家、每次激活 2 个，则激活比例是 $2/129$。如果每个专家大小近似相当，那么每个 token 真正执行的前馈参数占比也接近这个数量级。

第三，路由粒度更细。专家越多，理论上专长划分越细，前提是路由和负载均衡做得住。否则专家多只是“管理成本更高”。

下面用一个玩具例子说明门控过程。

假设输入 token 表示的是一段 Python 代码里的 `for i in range(n)`。gating 网络看到这个输入后，可能输出如下分数：

| 专家 | 含义 | 分数 |
|---|---|---:|
| shared | 通用语言与语法 | 0.52 |
| expert_17 | 代码模式 | 0.31 |
| expert_42 | 数学表达 | 0.08 |
| expert_88 | 图像区域描述 | 0.03 |
| 其他 | 其他领域 | 很低 |

如果规则是“共享专家固定激活，私有专家取 top-1”，那么最终只计算 `shared` 和 `expert_17`，输出近似为：

$$
y = 0.52 E_{\text{shared}}(x) + 0.31 E_{17}(x)
$$

其余专家不执行。这里建议配合图示理解：输入先经过一个小路由器，路由器指向两个专家分支，再把结果按权重合并。

但只会选专家还不够，MoE 的难点在于负载均衡。负载均衡，白话讲就是“不能把所有活都压给少数明星专家”。如果路由器总爱选某几个专家，会出现三个问题：

1. 某些专家过热，单机或跨卡延迟上升。
2. 其他专家学不到东西，形成容量浪费。
3. batch 内 token 分布一偏，吞吐就抖动。

因此工程上通常会给路由附加负载正则项，例如让专家选择分布更均匀，或者使用更细化的离散选择机制。公开分析常提到 DSelect-k 一类方法。DSelect-k 可以粗略理解为“一种更可控的可微离散路由方法”，目标是在保持 top-k 稀疏性的同时，让训练更稳定、负载更可调。

形式上，可以把总损失写成：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{load}}
$$

其中 $\mathcal{L}_{\text{task}}$ 是原始任务损失，$\mathcal{L}_{\text{load}}$ 是负载均衡损失，$\lambda$ 控制两者权重。这个负载项不直接提高语言理解能力，但它决定了 MoE 是否真的跑得起来。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现。它不是训练级框架代码，但足够展示 Gemini 这类“共享专家 + 私有专家 top-1”路由的核心逻辑。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def shared_expert(x):
    # 通用专家：做一个稳定的线性变换
    return 0.8 * x + 1.0

def private_expert_code(x):
    return 1.5 * x + 0.2

def private_expert_math(x):
    return 1.2 * x + 0.5

def private_expert_vision(x):
    return 0.6 * x + 2.0

PRIVATE_EXPERTS = [
    ("code", private_expert_code),
    ("math", private_expert_math),
    ("vision", private_expert_vision),
]

def gate_scores(x, hint):
    # 这里用 hint 模拟一个小 gating 网络的输出偏好
    # 实际系统里它来自可学习路由器
    base = {
        "code": [2.0, 0.4, -0.5],
        "math": [0.2, 1.8, -0.2],
        "vision": [-0.3, 0.1, 2.2],
    }
    logits = base[hint]
    return softmax(logits)

def moe_forward(x, hint):
    shared_weight = 0.5
    probs = gate_scores(x, hint)

    # 选 top-1 私有专家
    top_idx = max(range(len(probs)), key=lambda i: probs[i])
    private_name, private_fn = PRIVATE_EXPERTS[top_idx]
    private_weight = probs[top_idx]

    y = shared_weight * shared_expert(x) + private_weight * private_fn(x)
    return y, private_name, probs

# 玩具例子：代码 token 更可能命中 code 专家
y_code, expert_code, probs_code = moe_forward(2.0, "code")
assert expert_code == "code"
assert probs_code[0] > probs_code[1] and probs_code[0] > probs_code[2]

# 玩具例子：视觉 token 更可能命中 vision 专家
y_vision, expert_vision, probs_vision = moe_forward(2.0, "vision")
assert expert_vision == "vision"
assert y_code != y_vision

print(round(y_code, 4), expert_code)
print(round(y_vision, 4), expert_vision)
```

这个例子故意保留了三个关键点。

1. 共享专家始终执行。
2. 私有专家只选一个。
3. 最终输出是加权组合，不是简单替换。

如果把它写成更接近工程实现的结构化伪代码，大致如下：

```python
def moe_layer(x_batch):
    gate_logits = gate_network(x_batch)

    shared_out = shared_expert(x_batch)

    private_scores = masked_softmax(gate_logits["private"])
    top1_idx = argmax(private_scores, dim=-1)

    private_out = dispatch_to_selected_expert(
        x_batch,
        expert_index=top1_idx,
        expert_table=private_experts,
    )

    y = gate_logits["shared_weight"] * shared_out \
        + gather_top1_weight(private_scores) * private_out

    load_penalty = compute_load_balance_loss(top1_idx)
    return y, load_penalty
```

真实工程里还要补三块内容。

第一，dispatch。dispatch 就是“把 token 真正送到被选中的专家设备或张量分片上”。这部分经常比数学公式更难，因为它直接关系到跨卡通信和显存布局。

第二，capacity control。capacity，白话讲就是“每个专家每个 batch 最多能接多少 token”。如果某个专家被太多 token 同时选中，系统需要丢弃、回退或重路由，否则会形成热点。

第三，load regularization。可以是负载均衡损失、KL 正则、importance loss，或者更复杂的路由算法。目标都一样：别让路由塌缩到少数专家。

下面给一个极简的负载均衡度量示意：

$$
\mathcal{L}_{\text{load}}=\sum_{i=1}^{N}\left(p_i-\frac{1}{N}\right)^2
$$

这里 $p_i$ 表示专家 $i$ 在一个 batch 中被选中的比例，$N$ 是私有专家总数。它的意思很朴素：如果每个专家使用率都接近平均值，则这个损失更小。

真实工程例子可以用“百万 token 文档问答”来理解。假设你把一本大型技术手册、若干接口文档、日志片段和一段代码仓摘要放进同一上下文。系统并不是对每个 token 都开完整大前馈，而是在每个 MoE 层里动态选择少量专家。于是输入越长，虽然总 token 数依然线性增加，但每个 token 的前馈常数项更低，这就是它能支撑超长上下文的关键之一。

---

## 工程权衡与常见坑

MoE 最容易被初学者误解成“参数白嫖器”。实际上，它换来的不是免费能力，而是一组新的工程权衡。

最核心的权衡是：你用更复杂的路由和系统调度，换取更低的每 token 激活成本。如果路由器足够好，这笔交易很值；如果路由器不稳，收益会被通信、排队和失衡吃掉。

先看常见风险表。

| 风险 | 现象 | 后果 | 缓解方式 |
|---|---|---|---|
| 专家失衡 | 少数专家总被命中 | 吞吐抖动、训练退化 | 负载项、KL 正则、DSelect-k |
| 稀疏切换开销 | token 分发与回收复杂 | 通信时间上涨 | 预分桶、专家并行优化 |
| 容量溢出 | 某个专家接收过多 token | 丢 token 或回退路径 | capacity 限制、备用专家 |
| 共享专家过重 | 公共路径太大 | 稀疏收益被抵消 | 控制共享专家尺寸 |
| top-k 设太大 | 激活专家过多 | 成本接近稠密模型 | 保持极低激活比 |
| 短上下文收益不足 | 路由成本占比变高 | 延迟优势不明显 | 对短任务用更小模型或 dense 模型 |

新手最常见的误区有两个。

第一个误区：以为“专家越多越好”。不对。专家总数变多，只说明容量池更大；如果路由不能把 token 合理分流，专家多反而加剧系统复杂度。

第二个误区：把 top-k 设成总专家数，觉得这样“信息最全”。这实际上等于让所有专家都激活，MoE 就退化成比普通稠密层更复杂、更难部署的结构。尤其在超长上下文里，这样做会把成本直接抬回去。

可以用一个简单指标来判断是否还保有 MoE 的优势：

$$
r_{\text{active}}=\frac{\#\text{active experts}}{\#\text{total experts}}
$$

对于 Mixtral 8x7B 风格 top-2 路由，$r_{\text{active}}=2/8=25\%$。对于“共享专家 + 128 私有专家、每次激活 2 个”的 Gemini 式描述，$r_{\text{active}}=2/129\approx1.55\%$。这个差距很大，意味着在超长上下文里，前者每 token 激活的专家占比约是后者的 16 倍。

当然，这不代表 Gemini 一定在所有任务上都“16 倍更快”。真实速度还受注意力、缓存、硬件并行、路由实现和多模态模块影响。但它足以说明设计方向：Gemini 把激活比压得更低，更像是在为“长上下文可负担”服务。

真实工程里还有一个坑：稠密/稀疏模式切换。某些部署环境下，为了提高吞吐，系统可能对小 batch、短序列、冷启动阶段采用不同执行策略。如果没有预热和阈值设计，MoE 可能出现“理论省算力，实际不省时延”的情况。工程上需要直接测 three metrics：单请求时延、批处理吞吐、跨卡通信占比。只看 FLOPs 不够。

---

## 替代方案与适用边界

最直接的对比对象是 Mixtral 8x7B 一类的 top-2 MoE。它的思想也很清楚：总共 8 个专家，每个 token 选 2 个。优点是结构更简单、路由空间更小、工程可控性更高；缺点是激活比例更高，长上下文下的单位 token 前馈成本更难进一步压低。

对比表如下。

| 模型/方案 | 专家总数 | 每 token 激活 | 激活比例 | 最大 context 倾向 | 更适合的场景 |
|---|---:|---:|---:|---|---|
| 稠密 Transformer | 1 | 1 | 100% | 中短上下文 | 延迟敏感、部署简单 |
| Mixtral 8x7B 风格 | 8 | 2 | 25% | 中长上下文 | 通用文本、工程复杂度适中 |
| Gemini 1.5 Pro 式稀疏 MoE | 129 | 2 | 1.55% | 超长上下文、多模态 | 长文档、长视频、长代码检索 |

这里的边界要讲清楚。

如果任务是短文本问答、普通聊天、几十到几百 token 的代码补全，那么极低激活比的优势未必能充分体现。因为这时总序列不长，注意力和前馈都还在可控区间，反而路由和通信开销更容易显得“占比偏高”。在这种场景，Mixtral 这类较小 MoE 甚至稠密模型可能更稳。

但如果任务换成“1M token 视频检索”“一本书加多份附录的联合问答”“超长代码仓依赖分析”，情况就不同了。长度一旦上去，每个 token 的常数项就变得非常重要。Gemini 这种更激进的稀疏设计，会比 8 专家 top-2 方案更有机会把总成本压住。

可以做一个直观比较。若忽略专家内部尺寸差异，只看专家激活比例：

- Mixtral 风格：$2/8=25\%$
- Gemini 式描述：$2/129\approx1.55\%$

那么在“专家尺寸相近”的粗略假设下，后者的前馈激活比例约为前者的 $\frac{25}{1.55}\approx 16.1$ 分之一。这个比例不能直接等于整机速度比，但足以解释为什么更适合 1M 以上甚至 10M token 的场景。

替代方案主要有三类。

1. 回到稠密模型。适合部署环境简单、硬件有限、延迟极其敏感、上下文不长的场景。
2. 使用更小的 MoE。比如专家更少、top-k 更保守，牺牲一部分超长上下文效率，换更容易的系统实现。
3. 结合其他长上下文手段。比如注意力裁剪、检索增强、分块摘要。它们不能替代 MoE，但可以和 MoE 配合，进一步降低总成本。

结论是：Gemini 1.5 Pro 的 MoE 设计不是“普适最优”，而是明显偏向“超长上下文 + 多模态 + 总成本可控”这一目标函数。只要目标函数变了，最优设计也会变。

---

## 参考资料

1. Google DeepMind，《Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context》。本文用于支撑两部分内容：一是 Gemini 1.5 Pro 采用 MoE 并面向百万级以上上下文；二是长文档、长视频、长代码上的实际长上下文能力与检索表现。
2. AloneReaders 的 Gemini 1.5 Pro 结构分析文章。本文主要用来支撑“共享专家 + 128 个可选专家、每 token 约激活 2/129”这一公开分析画像，并用于与 Mixtral 8x7B 做效率维度对比。
3. Sergey Nikolenko，《A Mixed Blessing I: Mixtures of Experts from Committee Machines to LLMs》。本文主要用来支撑 MoE 路由、负载均衡、DSelect-k 与专家失衡问题的工程解释，对“为什么 MoE 不只是公式，还涉及系统调度”这一点尤其重要。

读者如果要继续核对本文内容，建议按下面顺序读：

1. 先读 DeepMind 技术报告，确认 Gemini 1.5 的长上下文能力、MoE 定位和实验边界。
2. 再读结构分析文章，理解“共享专家 + 大量可选专家”的效率画像，以及为何激活比会压到很低。
3. 最后读 Nikolenko 的长文，补齐 MoE 路由、top-k、负载均衡和 DSelect-k 的工程背景。
