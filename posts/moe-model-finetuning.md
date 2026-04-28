## 核心结论

MoE，Mixture-of-Experts，中文通常叫“专家混合模型”，可以理解为“不是所有 token 都走同一套前馈网络，而是先分流，再由少数专家处理”。MoE 微调的重点不是把全部参数再训练一遍，而是在尽量少改参数的前提下，保持路由稳定、专家分工有效、负载不过度偏斜。

对初级工程师来说，最重要的判断标准只有三个：

| 目标 | 直白解释 | 常见做法 | 必监控指标 |
|---|---|---|---|
| 任务适配 | 让模型学会新领域表达或新任务格式 | 训练 `gate`、部分专家、或专家上的 LoRA | 验证集 loss、任务指标 |
| 路由稳定 | 不要把分流规则训坏 | 限制 `gate` 更新幅度，保留均衡损失 | 专家命中率、路由熵 |
| 负载均衡 | 不要让 token 全挤到少数专家 | top-k 路由 + `L_bal` + 分阶段训练 | 每专家 token 占比、空转专家数 |

如果把 MoE 想成“分诊系统”，`gate` 就是分诊台，负责决定 token 去哪个专家；专家是不同科室。如果微调时只盯总 loss，不看路由分布，就可能出现“少数专家过载，其他专家几乎不工作”的情况。此时模型表面还能下降 loss，但结构已经偏了，后续训练通常会更脆弱。

因此，MoE 微调的实际目标不是单一的“把 loss 压低”，而是同时优化两件事：一是模型在新任务上的能力，二是专家利用率是否仍然健康。

---

## 问题定义与边界

本文讨论的是“MoE 模型的参数高效微调”，也就是在已有 MoE 基座上，只更新少量参数，让模型适配新任务。这里不讨论从零训练 MoE，也不讨论推理引擎层面的并行优化。

先统一记号：

| 记号 | 含义 | 白话解释 |
|---|---|---|
| $x$ | 输入 token 表示 | 一条 token 进入当前层后的向量 |
| `gate` | 路由器 | 决定 token 去哪些专家 |
| $FFN_e$ | 第 $e$ 个专家 | 第 $e$ 套前馈网络 |
| $S_k(x)$ | top-k 选中的专家集合 | 对 token $x$ 真正参与计算的专家 |
| $\alpha_e(x)$ | 专家权重 | 当前 token 对专家 $e$ 的依赖程度 |
| $W_e$ | 专家原始参数 | 专家本来的权重 |
| $\Delta W_e$ | 微调增量 | 训练时额外学出来的那一小部分 |
| $L_{bal}$ | 负载均衡损失 | 用来约束专家不要严重失衡 |

MoE 和普通稠密模型的核心区别是：普通模型的每个 token 都过同一套 FFN；MoE 模型只让 token 经过少量专家，因此总参数可以很大，但单次计算量不一定同比增加。

微调时常见的参数范围有四类：

| 策略 | 更新范围 | 成本 | 风险 |
|---|---|---|---|
| 全量微调 | 所有参数 | 最高 | 最容易破坏原路由 |
| 只训 `gate` | 只更新路由器 | 最低 | 容易只改分流，不改能力 |
| 只训部分专家 | 高频专家或指定专家 | 中等 | 专家选择错了会欠拟合 |
| 专家上挂 LoRA | 给专家权重加低秩增量 | 低到中等 | 需要处理专家不均衡 |

LoRA，Low-Rank Adaptation，中文可理解为“低秩增量微调”，本质是不给大矩阵全量更新，而是只学两个小矩阵乘积，减少可训练参数。对 MoE 来说，它通常挂在每个专家的线性层上，而不是只挂在共享层上。

---

## 核心机制与推导

MoE 前向传播可以写成两步。

第一步，`gate` 根据输入 token 计算路由概率：

$$
p(x)=softmax(W_g x)
$$

这里 $W_g$ 是路由器参数，输出是一个长度为专家数 $E$ 的概率向量。若采用 top-k 路由，就只保留概率最高的 $k$ 个专家，形成集合 $S_k(x)$。

第二步，按路由权重聚合专家输出：

$$
y(x)=\sum_{e\in S_k(x)} \alpha_e(x)\cdot FFN_e(x)
$$

其中 $\alpha_e(x)$ 是归一化后的 gate 权重。top-1 路由时，公式会退化成“只用一个专家输出”。

微调时最容易出问题的地方就在这里。因为 loss 的梯度不仅会更新专家参数，还会反过来影响 `gate`。如果 `gate` 过早偏向少数专家，就会形成一个自增强过程：

1. 某几个专家被分到更多 token。
2. 它们更新更多，短期表现更好。
3. `gate` 更愿意把 token 继续送过去。
4. 其他专家更难获得训练信号，逐渐闲置。

这就是常说的“路由塌缩”，意思是分流逻辑收缩到少数专家，结构上失去原本的专家分工。

一个玩具例子最容易看清楚。

设有 2 个专家，4 个 token，top-1 路由。某个 batch 中，`gate` 的平均概率是：

$$
P=[0.70, 0.30]
$$

真实分配比例是：

$$
f=[0.75, 0.25]
$$

常见的负载均衡损失可以写成：

$$
L_{bal}=E\cdot\sum_{e=1}^{E} f_e P_e
$$

代入 $E=2$：

$$
L_{bal}=2\times(0.75\times0.70+0.25\times0.30)=1.20
$$

如果后续训练变成更均衡的：

$$
P=[0.50,0.50],\quad f=[0.50,0.50]
$$

则：

$$
L_{bal}=2\times(0.50\times0.50+0.50\times0.50)=1.00
$$

这个值更接近平衡状态。直观上看，1.20 说明“路由概率和真实分配都在偏向同一个专家”，1.00 则说明两个专家承担的工作更平均。

LoRA 接入专家时，通常把专家权重写成：

$$
W_e' = W_e + \Delta W_e = W_e + B_eA_e
$$

其中 $\Delta W_e$ 是低秩更新，秩为 $r$。这意味着你不直接改动全部 $W_e$，而是只学习一个小容量补丁。好处是显存和可训练参数都更低；风险是如果所有专家都挂同样 rank，但训练数据只命中少数专家，就会出现“预算分配不合理”：热门专家 rank 不够，冷门专家 rank 浪费。

真实工程里，这个问题比玩具例子更明显。比如做金融客服领域适配，很多 token 都带有“额度、清算、授信、交易日、风控规则”等术语。如果这些 token 长期被某几个专家接收，那么最有效的策略往往不是“所有专家平均微调”，而是冻结基座，只训练 `gate`、少量高频专家，以及这些专家上的 LoRA。这样能让有限预算集中到真正承载新知识的路径上。

---

## 代码实现

下面给一个可运行的最小 Python 示例，展示三件事：如何计算负载均衡项、如何冻结参数、如何统计专家命中率。它不是完整训练脚本，但逻辑和真实工程一致。

```python
from collections import Counter
import math

def load_balance_loss(assignments, gate_probs, num_experts):
    """
    assignments: 每个 token 实际命中的专家 id，例如 [0, 0, 1, 0]
    gate_probs: 每个 token 的 gate 概率，例如 [[0.7,0.3], [0.8,0.2], ...]
    """
    total = len(assignments)
    freq = [0.0] * num_experts
    prob = [0.0] * num_experts

    for a in assignments:
        freq[a] += 1.0 / total

    for row in gate_probs:
        for e, p in enumerate(row):
            prob[e] += p / total

    return num_experts * sum(f * p for f, p in zip(freq, prob))

def route_entropy(avg_probs):
    return -sum(p * math.log(p + 1e-12) for p in avg_probs)

# 玩具例子：2 个专家，4 个 token
assignments = [0, 0, 1, 0]  # f = [0.75, 0.25]
gate_probs = [
    [0.7, 0.3],
    [0.7, 0.3],
    [0.7, 0.3],
    [0.7, 0.3],
]

loss = load_balance_loss(assignments, gate_probs, num_experts=2)
assert round(loss, 2) == 1.20

balanced_assignments = [0, 0, 1, 1]  # f = [0.5, 0.5]
balanced_probs = [
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
]
balanced_loss = load_balance_loss(balanced_assignments, balanced_probs, num_experts=2)
assert round(balanced_loss, 2) == 1.00

avg_probs = [0.7, 0.3]
entropy = route_entropy(avg_probs)
assert entropy > 0

hit_count = Counter(assignments)
assert hit_count[0] == 3
assert hit_count[1] == 1

print("imbalanced L_bal =", loss)
print("balanced   L_bal =", balanced_loss)
print("route entropy =", round(entropy, 4))
print("expert hits =", dict(hit_count))
```

在真实模型里，冻结逻辑通常先做成白名单，而不是黑名单。因为 MoE 层名字复杂，白名单更不容易漏。

```python
# 伪代码：冻结除 gate 和指定专家外的参数
trainable_keywords = [
    "router",              # gate
    "experts.3",           # 第 3 个专家
    "experts.7",           # 第 7 个专家
    "lora_",               # LoRA 增量参数
]

for name, param in model.named_parameters():
    param.requires_grad = any(k in name for k in trainable_keywords)
```

如果使用 PEFT，需要特别注意：有些 MoE 实现中的专家权重不是 `nn.Linear`，而是直接暴露成 `nn.Parameter`。这时只配 `target_modules` 可能漏掉专家参数，要考虑 `target_parameters`。

```python
# 伪代码：PEFT / LoRA 配置思路
lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],   # 共享层常见写法
    "target_parameters": [
        "model.layers.10.moe.experts.3.w1",
        "model.layers.10.moe.experts.3.w2",
    ],  # 专家若不是 nn.Linear，可直接指向参数名
}
```

训练时建议同步记录路由统计，而不是训练完再看。最低成本的统计项有三个：每专家命中率、平均 gate 概率、路由熵。路由熵可以理解为“分流分散程度”，值越低，说明路由越集中。

| 更新对象 | 作用 | 适合场景 |
|---|---|---|
| `gate` | 改分流逻辑 | 任务变化不大，但路由需重排 |
| 高频专家 | 学新领域知识 | 某些术语或模式明显集中 |
| 专家上的 LoRA | 低成本增量适配 | 显存紧张、参数预算受限 |
| 共享注意力层 | 改全局表示 | 任务分布变化较大 |

---

## 工程权衡与常见坑

MoE 微调里最常见的误判是“总 loss 在降，所以训练没问题”。这不够。因为 MoE 是带路由的结构，loss 下降只说明整体目标在优化，不说明专家分工仍然健康。

下面是最常见的坑位对照表：

| 问题 | 现象 | 原因 | 规避措施 |
|---|---|---|---|
| `gate` 塌缩 | 80% 以上 token 命中少数专家 | 路由更新过快，自增强偏置 | 加 `L_bal`，降低 `gate` 学习率，必要时先冻结 `gate` 热身 |
| 专家空转 | 一部分专家长期几乎无命中 | 训练数据窄，或 gate 已固化 | 先看命中率，再决定是否保留这些专家参与训练 |
| 统一 LoRA rank | 热门专家学不动，冷门专家浪费参数 | 所有专家预算平均分配 | 按命中率或重要性分配不同 rank |
| 参数没覆盖 | 训练后几乎没效果 | 专家参数不是 `nn.Linear`，LoRA 没挂上 | 检查 `requires_grad` 和参数名，必要时用 `target_parameters` |
| 推理变慢 | 训练可行，部署成本上升 | LoRA 未合并，MoE 路径本就稀疏复杂 | 合并权重，确认推理框架支持 fused MoE |

一个真实工程例子是金融客服适配。假设基础 MoE 模型已经具备通用中文问答能力，现在要支持金融术语、风控规则、账务解释。此时如果直接全量微调，当然可能得到更高上限，但成本高，而且容易破坏原本的通用路由。更稳的做法通常是：

1. 冻结共享主干。
2. 只训练 `gate` 和少量高频专家。
3. 在这些专家的 FFN 上挂 LoRA。
4. 每隔若干 step 统计专家命中率与路由熵。

如果发现训练后“金融术语样本准确率提升，但 2 个专家吃掉了绝大多数 token”，这不一定是成功。它可能意味着模型只学会了“把相关输入全压给一个专家”，而不是形成可泛化的专家分工。上线后遇到稍有变化的问法，稳定性可能变差。

所以工程上至少要同时看四类指标：任务指标、专家命中率、路由熵、每专家有效梯度或更新量。只看总分数，容易把结构性问题藏起来。

---

## 替代方案与适用边界

MoE 微调没有单一标准答案，关键取决于任务变化幅度、预算、部署约束。

| 方案 | 训练成本 | 收敛稳定性 | 推理代价 | 适用场景 |
|---|---|---|---|---|
| 只训 `gate` | 很低 | 中等，易偏路由 | 基本不变 | 只想重排专家分工 |
| 只训少数高频专家 | 低到中等 | 较稳 | 基本不变 | 领域术语迁移 |
| 专家上挂 LoRA | 低 | 较稳，但要控不均衡 | 可能略增 | 显存紧张的主流方案 |
| 全量微调 | 高 | 取决于数据和训练技巧 | 可能需重新部署优化 | 强分布漂移、大任务切换 |
| 先均衡再专化 | 中等 | 通常更稳 | 取决于实现 | 担心早期塌缩的场景 |

可以把选择逻辑压成一句话：

- 任务只是局部领域适配，优先考虑“少量专家 + LoRA”。
- 任务主要是路由重排，而知识本身变化不大，可以先试“只训 `gate`”。
- 任务分布和原模型差异极大，比如从通用客服转到代码生成，局部适配往往不够，需要扩大可训练范围。
- 预算极低、只允许极少参数更新时，先从 `gate` 或热门专家开始，而不是平均动所有专家。

MoE 的优势在于“总参数大，但单 token 激活稀疏”；它的难点也在这里。你不是只在微调一个函数，而是在微调“分流规则 + 多个子函数”的联合系统。只要路由不稳，再省参数的方案也可能训歪；只要专家严重失衡，再低的 loss 也不代表结构健康。

---

## 参考资料

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
3. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
4. [Hugging Face PEFT LoRA Developer Guide](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md)
5. [MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models](https://arxiv.org/abs/2402.12851)
6. [MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts](https://arxiv.org/abs/2404.15159)
