## 核心结论

DeepSeek-V2 的 DeepSeekMoE，本质是“更细的专家切分 + 始终在线的共享专家”两件事同时成立的稀疏前馈网络。MoE 的中文常见翻译是“混合专家”，意思是模型里有很多个前馈子网络，但每个 token 不会把它们全跑一遍，而是只选其中一部分。

它的关键配置可以先记成一句话：160 个路由专家负责专业化分工，2 个共享专家负责通用知识保底，每个 token 在一层里只激活 6 个路由专家，再叠加 2 个共享专家，所以总参数虽然达到 236B，但单 token 的激活参数量只有约 21B。

对零基础读者，可以先用一个玩具例子理解。假设你要批改一句话“Python 里怎么做异步 IO”，路由器会把“Python”“异步”“系统接口”这些局部模式分给几个擅长不同方向的小专家，而共享专家始终参与，负责“语言常识、上下文连接、通用语义”这类不该被任何单个路由专家独占的知识。这样做的结果不是“少算一些层”，而是“每层只算少数更合适的 FFN”。

和早期 GShard 一类粗粒度 MoE 相比，DeepSeekMoE 的提升不只是“专家更多”。它把一个大专家继续拆成更小、更专精的子专家，再用共享专家去承接重复性高的公共知识，减少不同专家反复学同一件事的浪费。因此在相近计算预算下，它通常能带来更好的专家专精度，并在公开报告里表现为相对粗粒度 MoE 平均约 2% 到 6% 的基准提升。

| 架构 | 路由专家 | 共享专家 | 每 token 激活 | 激活参数特征 | 相对粗粒度 GShard |
|---|---:|---:|---:|---|---|
| GShard 式粗粒度 MoE | 少量大专家 | 通常无 | 常见 top-k=2 | 单个专家较大，知识混合更多 | 基线 |
| DeepSeekMoE | 160 个细粒度专家 | 2 个常开专家 | 6 路由 + 2 共享 | 总参大，但单 token 激活低 | 平均提升约 2% 到 6% |

---

## 问题定义与边界

DeepSeekMoE 要解决的问题不是“怎样把参数堆到最大”，而是“怎样在超大参数规模下，依然保持训练和推理可接受的成本，同时让不同专家真正形成分工”。

这里的“专家”可以先理解成“多个并列的前馈网络”。标准 Transformer 层里的 FFN 是所有 token 都会经过的一套计算；MoE 则把这套 FFN 换成很多份候选子网络，由路由器决定一个 token 应该走哪几份。问题在于，如果专家太粗，一个专家内部仍然要覆盖很多主题，结果就是知识混在一起；如果专家太细，又容易出现训练不稳定、负载不均、专家闲置。

DeepSeekMoE 的边界条件比较明确：

| 约束项 | 含义 | 工程影响 |
|---|---|---|
| 160 个路由专家 | 动态选择的细粒度专家池 | 专精度提高，但路由与调度更复杂 |
| 2 个共享专家 | 每个 token 都会经过 | 提供通用知识，降低知识重复学习 |
| 每 token 激活 6+2 个专家 | 6 个路由 + 2 个共享 | 控制单 token 计算量 |
| 128K 上下文 | 能处理很长输入序列 | 对 KV cache、吞吐和显存要求高 |
| 8×H800 部署目标 | 面向真实集群推理 | 必须兼顾负载均衡与通信开销 |

新手可以把它理解成一个排班问题。你有 160 个“专业小组”和 2 个“常识小组”。每来一个 token，只允许请 6 个专业小组参与，否则算力就爆掉；但常识小组必须始终在线，否则很多跨领域、跨上下文的连接信息会丢失。

真实工程例子是长上下文问答。比如 128K 上下文里同时放了 API 文档、日志片段、数据库表结构和用户历史对话。一个 token 可能既需要“代码接口”知识，也需要“对话上下文”知识。若全靠路由专家动态选择，模型很容易把通用语义也切碎到多个专家里重复学习；共享专家的作用，就是把这部分高复用知识稳定承接下来。

---

## 核心机制与推导

DeepSeekMoE 的核心计算发生在 Transformer 层中的 FFN 位置。设第 $\ell$ 层中，token $t$ 进入 MoE 前的表示是 $u_t^\ell$，那么输出可以写成：

$$
h^\ell_t = \sum_{i=1}^{mN} g_{i,t}\,\mathrm{FFN}_i(u^\ell_t) + u^\ell_t
$$

这条公式可以逐项解释：

- $u_t^\ell$：当前层里 token 的输入隐藏状态，也就是进入专家前的向量表示。
- $\mathrm{FFN}_i$：第 $i$ 个专家对应的前馈网络。
- $g_{i,t}$：路由权重，表示“token $t$ 分给专家 $i$ 的强度”。
- $mN$：细粒度专家总数。对 DeepSeekMoE，可理解为路由专家池规模。
- $mK$：每个 token 实际保留的非零专家数，也就是 top-k 选择后真正参与计算的专家个数。在这里是 6 个路由专家。

“路由器”就是给每个 token 打分的模块。它通常先输出一组 logits，再经过 softmax 变成概率，然后只保留 top-k。softmax 可以先理解成“把原始分数变成可比较的权重分布”。因此 $g_{i,t}$ 大多是稀疏的，只有很少几个专家非零。

但 DeepSeekMoE 比普通 top-k MoE 多了一步：共享专家不参与竞争，它们是持续激活的。这意味着最终输出不是只有 top-6 路由专家，而是“6 个动态路由专家 + 2 个固定共享专家”的合成结果。用更贴近实现的形式，可以写成：

$$
h^\ell_t
=
\sum_{i \in \mathcal{T}_t} g_{i,t}\,\mathrm{FFN}^{route}_i(u^\ell_t)
+
\sum_{j=1}^{2} \mathrm{FFN}^{shared}_j(u^\ell_t)
+
u^\ell_t
$$

其中 $\mathcal{T}_t$ 是 token $t$ 被选中的 top-6 路由专家集合。

为什么“细粒度 + 共享”比“少数大专家”更合理？因为一个大专家往往要同时吸收多类模式，结果是知识纠缠。把专家拆细后，每个专家只学局部模式，专精更容易形成；但过细会导致通用知识被多份复制，所以再加入共享专家，把“高复用的公共部分”集中处理。这相当于把模型内部的知识拆成两类：

| 知识类型 | 适合谁学习 | 原因 |
|---|---|---|
| 通用语义、上下文连接、常识模式 | 共享专家 | 几乎所有 token 都会复用 |
| 领域化、模式化、局部特征 | 路由专家 | 不同 token 需要的分工不同 |

玩具例子可以更直观。假设一句输入是“如何定位 Python 服务的高延迟 SQL 请求”。其中“Python 服务”可能路由到语言专家，“高延迟”路由到性能诊断专家，“SQL 请求”路由到数据库专家；与此同时，共享专家处理句法连接、常识性语义与上下文一致性。最后这些专家输出按权重加和，再与残差 $u_t^\ell$ 相加，得到当前层输出。

残差可以先理解成“保留原始输入的一条直通路径”。它的作用是避免专家输出把原本有用的信息完全覆盖掉，也是 Transformer 稳定训练的重要机制。

---

## 代码实现

下面用一个可运行的 Python 玩具实现展示核心逻辑。它不是 DeepSeek-V2 的完整训练代码，而是把“路由专家 top-k 选择 + 共享专家常开 + 加权合并 + residual”这个机制压缩成最小版本。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_indices(values, k):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]

def linear_ffn(x, scale, bias):
    # 玩具 FFN：真实模型里会是两层线性层加激活函数
    return [scale * v + bias for v in x]

def add_vec(a, b):
    return [x + y for x, y in zip(a, b)]

def mul_vec(a, w):
    return [w * x for x in a]

def moe_forward(
    token_vec,
    router_logits,
    route_experts,
    shared_experts,
    top_k=2,
):
    probs = softmax(router_logits)
    chosen = topk_indices(probs, top_k)

    out = [0.0 for _ in token_vec]

    # 路由专家：只对 top-k 做加权求和
    route_weight_sum = sum(probs[i] for i in chosen)
    for i in chosen:
        norm_w = probs[i] / route_weight_sum
        expert_out = route_experts[i](token_vec)
        out = add_vec(out, mul_vec(expert_out, norm_w))

    # 共享专家：始终激活
    for expert in shared_experts:
        shared_out = expert(token_vec)
        out = add_vec(out, shared_out)

    # residual
    out = add_vec(out, token_vec)
    return out, chosen, probs

# 4 个路由专家，2 个共享专家
route_experts = [
    lambda x: linear_ffn(x, 0.5, 0.0),
    lambda x: linear_ffn(x, 1.0, 0.1),
    lambda x: linear_ffn(x, -0.2, 0.3),
    lambda x: linear_ffn(x, 0.8, -0.1),
]

shared_experts = [
    lambda x: linear_ffn(x, 0.1, 0.0),
    lambda x: linear_ffn(x, 0.0, 0.2),
]

token = [1.0, -1.0, 0.5]
router_logits = [2.0, 0.2, -1.0, 1.2]

out, chosen, probs = moe_forward(
    token_vec=token,
    router_logits=router_logits,
    route_experts=route_experts,
    shared_experts=shared_experts,
    top_k=2,
)

assert len(chosen) == 2
assert chosen[0] in range(4) and chosen[1] in range(4)
assert len(out) == len(token)
assert all(isinstance(v, float) for v in out)
assert abs(sum(probs) - 1.0) < 1e-9

print("chosen experts:", chosen)
print("output:", out)
```

这段代码里有几个新手最容易忽略的点：

1. `router_logits` 不是最终权重，先经过 softmax 才变成概率。
2. `top_k` 之后通常要重新归一化，否则保留下来的专家权重之和不是 1。
3. 共享专家不参加 top-k 竞争，它们直接被执行。
4. `residual` 不是可有可无的装饰，而是稳定训练和信息保留的关键。

如果把它翻译成更接近真实框架的伪代码，大致是：

```python
def deepseek_moe_layer(u):
    scores = router(u)                     # [num_route_experts]
    probs = softmax(scores)
    idx = topk(probs, k=6)
    gates = normalize(probs[idx])

    route_out = 0
    for i, g in zip(idx, gates):
        route_out += g * route_experts[i](u)

    shared_out = 0
    for expert in shared_experts:          # always active
        shared_out += expert(u)

    h = route_out + shared_out + u         # residual
    return h
```

真实工程实现里，还会再加上 expert parallel、capacity 限制、auxiliary balance loss、通信调度和数值稳定处理。也就是说，代码难点不在“写出 top-k”，而在“大批量 token 如何稳定、高效地分发到专家，再把结果并行收回”。

---

## 工程权衡与常见坑

DeepSeekMoE 的难点，不是概念上理解“选几个专家”，而是如何让这种稀疏结构在大规模训练和推理里不崩。

第一个坑是 routing collapse，也就是“路由崩塌”。这句话的白话解释是：本来应该很多专家共同分工，结果训练到一半，大量 token 都挤到少数几个专家上，其他专家几乎闲置。这样不仅浪费参数，还会造成负载不均、通信热点和泛化下降。因此训练里通常要加 balance loss，也就是“鼓励 token 更均匀分布到专家”的辅助损失。研究摘要里提到的 expert-level balance loss 量级约为 0.001，核心不是记死这个数，而是知道它太大或太小都不行。

第二个坑是 capacity。capacity 可以先理解成“每个专家一次最多接收多少 token”。如果没有上限，热门专家可能被淹没；如果上限过紧，大量 token 会被丢弃或重路由，反而伤害质量。MoE 的工程实现通常需要在吞吐、丢弃率和显存之间找平衡。

第三个坑是共享专家不能被误当成 fallback。fallback 的意思是“其他都不行时兜底”。但在 DeepSeekMoE 里，共享专家不是兜底，而是结构性组成部分。禁用共享专家，往往不是小幅掉点，而是整个知识分配机制被打断，loss 会明显变差，因为原本应该集中存放的公共知识被迫重新散落到路由专家里。

第四个坑是数值稳定。细粒度专家更多、路由更稀疏，梯度流动会更脆弱，所以常常需要 RMSNorm、输出缩放或其他稳定技巧。RMSNorm 可以先理解成“只按均方根大小做归一化”的归一化方法，它比 LayerNorm 更轻，在大模型里常用于稳定数值范围。

| 常见问题 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 路由崩塌 | 少数专家负载极高 | balance loss 太弱或路由偏置过强 | 监控 expert load，调 balance loss |
| 专家过载 | 热门专家 token 爆满 | capacity 太小或流量分布太偏 | 调整 capacity factor，观察丢弃率 |
| 收敛不稳 | loss 抖动、梯度异常 | 稀疏路由导致数值波动 | 加 RMSNorm、缩放因子、稳态初始化 |
| 共享专家失效 | 质量明显下降 | 通用知识无人承接 | 共享专家必须持续激活 |
| 吞吐不达标 | GPU 利用率低、通信重 | 专家分发与回收成本高 | 做 expert parallel 和批量路由聚合 |

真实工程例子是 8×H800 集群上的长上下文推理。表面上看，21B 激活参数已经比全激活大模型便宜很多；但如果路由分布极不均匀，某些 GPU 上的专家会成为热点，最终吞吐瓶颈不一定在矩阵乘法，而在专家间 token 搬运与同步。也就是说，MoE 的成本不只来自“算多少”，还来自“怎么调度”。

---

## 替代方案与适用边界

DeepSeekMoE 不是所有场景都最优。它特别适合“知识范围很广、但单个 token 只需要其中一部分知识”的任务，比如长对话、多领域问答、检索增强生成、代码与自然语言混合场景。这些任务里，专家分工能带来明显收益，而共享专家又能提供稳定的通用语义支撑。

如果任务本身更统一，比如风格单一、领域固定、样式重复的垂类生成，Dense Transformer 也就是“每层都全量激活”的稠密模型，可能更简单、更稳定。Dense 的白话解释是：不做专家选择，每个 token 都跑完整个 FFN 和注意力路径。它的缺点是参数一大，计算就跟着线性变贵；但优点是实现简单、训练稳定、调度成本低。

GShard 式粗粒度 MoE 则处在中间。它已经利用了稀疏激活节省计算，但专家粒度较粗，容易让一个专家承载过多混合知识。DeepSeekMoE 的改进，在于把“稀疏”从单纯节省计算，推进到“更细的知识分工”。

| 方案 | 激活参数量 | 训练/推理复杂度 | 知识分工能力 | 适用场景 |
|---|---|---|---|---|
| Dense Transformer | 高 | 低到中 | 弱，所有参数共同承担 | 参数预算充足、追求稳定 |
| GShard 式粗粒度 MoE | 中 | 中到高 | 中，已有专家分工 | 想降成本，但不追求极细分工 |
| DeepSeekMoE | 更低 | 高 | 强，细粒度专精 + 共享保底 | 长上下文、多领域、多轮交互 |

新手可以这样记边界：如果把模型比作教师团队，Dense 是“全体老师都来上每一节课”，GShard 是“请少数几位大老师轮流讲”，DeepSeekMoE 是“请少量更细分的小专家，再固定带上常识老师”。当你的任务需要很多专业方向同时存在，但每次只会用到其中一小部分时，DeepSeekMoE 才最有价值。

---

## 参考资料

1. DeepSeek-V2 论文，`arXiv:2405.04434`。重点是完整模型设计、DeepSeek-V2 的总体架构与性能结果。
2. DeepSeekMoE 技术报告，`arXiv:2401.06066`。重点是细粒度专家切分、共享专家设计和相对粗粒度 MoE 的机制解释。
3. NVIDIA NeMo / Megatron Bridge 的 DeepSeek-V2 文档。重点是工程部署视角，包括推理配置、集群运行与长上下文支持。
4. Docsaid 对 DeepSeek-V2 的技术解读。重点是把论文里的 MoE 结构和训练细节翻成更易读的说明，适合快速建立整体图景。
