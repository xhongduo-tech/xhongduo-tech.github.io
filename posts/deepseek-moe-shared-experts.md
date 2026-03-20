## 核心结论

DeepSeekMoE 的共享专家机制，可以概括成一句话：把所有 token 都会反复用到的“通用知识”，从路由专家里单独抽出来，放进始终激活的少量共享专家中。

这里先定义几个术语。MoE，中文常写“专家混合”，就是一层里放很多个 FFN 子网络，但每个 token 只激活其中一小部分。共享专家，白话就是“每个 token 都必须经过的固定老师”；路由专家，白话就是“只在需要时被选中的专项老师”。路由器，白话就是“给 token 分配老师的打分器”。

DeepSeekMoE 的关键不是“多放几个专家”，而是“把专家分工改对”。共享专家负责通用语义、基本语言规律、跨任务都要用到的表示；路由专家负责更细的专业化模式，比如代码补全、数学推理、领域术语、某些语言分布。这样做的直接结果是：路由专家不用反复学习“基础课”，参数冗余下降，单位计算量下的有效容量上升。

一个直观判断标准是：如果去掉共享专家，再多给一个路由专家，计算量不变，但效果明显变差，那么共享专家学到的不是“可替代的平均能力”，而是“全局都要用的底座能力”。DeepSeekMoE 论文中的 Pile 实验就显示，关闭共享专家后损失从 1.808 上升到 2.414，说明共享部分不是装饰，而是结构性收益。

下表能把“参数大，但每次只算一部分”这件事说清楚：

| 模型 | 总参数 | 每 token 激活参数 | 相对同级 dense 计算量 | 结论 |
|---|---:|---:|---:|---|
| DeepSeekMoE 16B | 16.4B | 约 2.8B | 约 40% | 接近 7B dense 表现，但计算更省 |
| DeepSeekMoE 145B | 145B | 约 12B | 约 18.2% 到 28.5% | 在更大规模下继续保持稀疏计算优势 |

---

## 问题定义与边界

标准 MoE 的常见问题有两个。知识混杂，白话就是“一个专家被迫学很多不相关内容”；知识冗余，白话就是“多个专家都在重复学同样的基础能力”。共享专家机制主要解决第二个问题，同时也间接帮助第一个问题。

如果没有共享专家，路由专家虽然看起来分工不同，但实际训练中会不断重复吸收通用模式。比如几乎所有文本 token 都需要基础语法、常见词法、句子连接关系、通用语义线索。只靠路由选择，这些基础模式会在多个专家里重复出现，导致专家容量被“共通知识”吞掉。

DeepSeekMoE 的边界条件很明确：它不是免费增加计算。为了维持每个 token 的计算预算不变，加入 $K_s$ 个共享专家后，路由部分只能少选 $K_s$ 个专家。形式化写法是：

$$
y=\sum_{i=1}^{K_s}E_i^{(s)}(x)+\sum_{j\in \text{Top-}K_r'} g_j(x)E_j^{(r)}(x), \quad K_r'=K_r-K_s
$$

这里 $E_i^{(s)}$ 是共享专家，$E_j^{(r)}$ 是路由专家，$g_j(x)$ 是路由权重。这个公式表达的边界很重要：共享专家不是“额外加餐”，而是“把原来总共要激活的专家名额，切出一部分做固定底座”。

玩具例子可以这么看。假设一层原本每个 token 激活 8 个专家。现在把其中 2 个名额固定成共享专家，那么路由器只需要再从剩余专家里选 6 个。总激活数仍然是 8，所以计算预算没变。但这 6 个路由专家已经不必再学“基础语法课”，可以把容量留给更细的模式。

这件事只在“数据分布里确实存在大量共通知识”时最有价值。多语言、代码、推理混合训练符合这个条件；单一任务、单一领域、模式很窄的场景，收益通常会缩小。

---

## 核心机制与推导

DeepSeekMoE 的机制可以拆成两步：先固定底座，再做稀疏分流。

第一步，共享专家始终激活。始终激活，白话就是“无论 token 是谁，都先走这几个公共 FFN”。这一步负责提取跨上下文都稳定存在的通用特征。

第二步，路由器只在“其余专家”里做 Top-K 选择。Top-K，白话就是“只保留打分最高的几个专家，其它全部不算”。所以共享专家和路由专家并不是并列竞争关系，而是“固定公共路径 + 稀疏专项路径”的组合。

标准 MoE 常写成：

$$
h_t=\sum_{i=1}^{N} g_{i,t} \, \text{FFN}_i(x_t)
$$

DeepSeekMoE 则改成：

$$
h_t=\sum_{i=1}^{K_s}\text{FFN}^{(s)}_i(x_t)+\sum_{j=K_s+1}^{mN} g_{j,t}\,\text{FFN}^{(r)}_j(x_t)
$$

并且

$$
g_{j,t}=
\begin{cases}
s_{j,t}, & s_{j,t}\in \text{TopK}(\{s_{k,t}\}, mK-K_s) \\
0, & \text{otherwise}
\end{cases}
$$

其中 $s_{j,t}$ 是路由器对第 $j$ 个路由专家的亲和度分数。亲和度，白话就是“这个 token 适不适合交给该专家处理”。

为什么这能减冗余？因为共享专家会持续接触所有 token，它天然更适合压缩“高频共性”；路由专家则只接触被选中的子分布，更适合学习长尾模式。训练一段时间后，参数分工会更稳定：共享专家吸收公共基线，路由专家向特化方向收缩。

再看一个更完整的玩具例子。假设共有 20 个专家，设置 2 个共享专家、18 个路由专家，原预算是每 token 激活 8 个专家。那么实际执行时是“2 个共享 + 6 个路由”。如果某个 token 是普通英文句子，共享专家大概率已经提供了大部分语义骨架，路由器只需要补上语法细节、文体特征、上下文相关模式；如果 token 来自代码片段，路由器则更可能选中代码相关专家。底座不变，专项可变，这就是它的结构收益来源。

真实工程例子是多语言预训练。比如一个批次里同时有英文新闻、中文问答、Python 代码和数学证明。没有共享专家时，多个路由专家都会被迫学到“分词边界、通用语义连接、常见标点模式”这类基础规律。加入共享专家后，这部分可以集中存储，路由专家则更专注于“编程语法”“中文表达习惯”“数学符号上下文”等专项分布。对大模型来说，这种分工会直接影响参数利用率，而不只是解释上的好看。

---

## 代码实现

下面给一个可运行的最小 Python 版本。它不是训练代码，只是把“共享专家始终执行，路由专家走 Top-K”这件事明确表达出来。

```python
from math import exp

def softmax(xs):
    m = max(xs)
    exps = [exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def topk_indices(scores, k):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

def make_expert(scale, bias):
    def expert(x):
        return [scale * v + bias for v in x]
    return expert

def add_vec(a, b):
    return [x + y for x, y in zip(a, b)]

def moe_forward(x, shared_experts, routed_experts, router_logits, total_budget):
    k_shared = len(shared_experts)
    k_routed = total_budget - k_shared
    assert k_routed >= 0
    assert len(router_logits) == len(routed_experts)

    out = [0.0 for _ in x]

    # 1) shared experts are always on
    for expert in shared_experts:
        out = add_vec(out, expert(x))

    # 2) routed experts are selected by top-k
    probs = softmax(router_logits)
    picked = topk_indices(probs, k_routed)

    for idx in picked:
        weighted = [probs[idx] * v for v in routed_experts[idx](x)]
        out = add_vec(out, weighted)

    return out, picked

if __name__ == "__main__":
    x = [1.0, 2.0]
    shared = [make_expert(1.0, 0.0), make_expert(0.5, 1.0)]
    routed = [
        make_expert(2.0, 0.0),
        make_expert(0.1, 0.0),
        make_expert(1.5, -1.0),
    ]
    logits = [3.0, 0.2, 2.0]

    out, picked = moe_forward(x, shared, routed, logits, total_budget=4)

    assert len(picked) == 2          # total_budget=4, shared=2, so routed=2
    assert picked[0] == 0            # largest logit first
    assert all(isinstance(v, float) for v in out)
```

这个实现里最重要的不是 `softmax`，而是两条结构约束：

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| 共享专家循环 | 无条件执行所有共享专家 | 把共通知识固定沉淀到公共路径 |
| Top-K 路由 | 只选少量路由专家 | 保持稀疏计算，不让成本失控 |
| 总预算约束 | `k_routed = total_budget - k_shared` | 保证加共享专家后计算量不膨胀 |

真实工程里还会多两件事。第一，批量化 dispatch，把同一专家要处理的 token 合并起来算，不然 kernel 启动开销会吃掉收益。第二，负载均衡，让热门专家不要长期过载，否则会出现“少数专家一直忙，其它专家学不到东西”的训练塌缩。

---

## 工程权衡与常见坑

第一个权衡是共享比例。比例过低，通用知识仍会泄漏进路由专家；比例过高，路由专家剩余空间不够，专业化会被压制。公开总结里，一个常被提到的经验是把“共享专家 : 激活路由专家”控制在约 1:3。它不是定律，但符合 DeepSeekMoE 扩到更大规模时的常用配置，例如 2 个共享配 6 个激活路由。

第二个坑是误以为“共享专家越多越稳”。这通常不对。共享专家天然获得全部 token 的训练信号，已经很容易学强；如果给它们过多名额，路由专家会因为样本覆盖减少而变弱，最后模型退化成“带一点路由的半稠密 FFN”。

第三个坑是把共享专家当成负载均衡的替代品。不是。共享专家只解决参数分工，不解决路由偏斜。路由偏斜，白话就是“有些专家总被选中，另一些专家几乎没人用”。这仍然需要额外的负载均衡策略，否则路由专家无法真正形成稳定分工。

第四个坑是忽视延迟形态。理论 FLOPs 降了，不等于线上时延一定线性下降。MoE 常见瓶颈包括 token 重排、跨设备通信、专家并行不均衡。共享专家虽然数量少，但它们是 always-on 路径，会成为每个 token 的固定成本，所以实现时必须让共享专家的执行足够规整。

一个很能说明问题的结果是：在 DeepSeekMoE 论文里，直接禁用共享专家并补一个路由专家，Pile loss 从 1.808 恶化到 2.414。工程上这意味着，共享专家不是“为了好解释而加的结构”，而是模型已经真实依赖的一部分。如果训练配置里共享比例改动过大，通常必须重新做消融实验，而不能靠直觉迁移。

---

## 替代方案与适用边界

如果任务分布很单一，dense 模型常常更直接。dense，白话就是“所有参数每次都参与计算”。比如单领域客服、固定模板生成、窄领域分类，这些任务的“共通知识”和“专业知识”分界没那么强，共享专家带来的结构收益可能不足以覆盖 MoE 的系统复杂度。

经典 MoE 也是替代方案。它没有共享专家，所有专家都走路由，结构更简单，部署链路也更成熟。像 Mixtral 一类 top-2 路由模型，在硬件栈已经适配好的情况下，落地成本会更低。

还可以把几种思路并排看：

| 架构 | 共享专家 | 每 token 激活方式 | 优点 | 适合场景 |
|---|---:|---|---|---|
| Dense Transformer | 0 | 全参数激活 | 简单稳定 | 小中型模型、单任务 |
| 经典 MoE | 0 | Top-K 路由 | 稀疏计算明显 | 已有成熟专家并行基础设施 |
| DeepSeekMoE | 1-2 个或少量 | 共享专家 + Top-K 路由 | 更强参数分工，减少冗余 | 多语言、代码、推理混合预训练 |

所以适用边界很清楚。数据越杂、任务越多、总参数越大，共享专家机制越容易体现价值；任务越窄、系统越强调简单部署，这个机制的必要性越弱。

---

## 参考资料

1. DeepSeekMoE 论文，ACL 2024：<https://aclanthology.org/2024.acl-long.70/>
2. DeepSeekMoE 论文 PDF：<https://aclanthology.org/2024.acl-long.70.pdf>
3. Emergent Mind 对 DeepSeekMoE 的架构与规模总结：<https://www.emergentmind.com/topics/deepseekmoe>
4. Emergent Mind 对 DeepSeekMoE 统计分析论文的索引：<https://www.emergentmind.com/papers/2505.10860>
5. 对共享专家公式与路由形式的技术整理：<https://hmellor.github.io/ml-notes/models/deepseek/DeepSeekMoE>
6. 关于 DeepSeekMoE 比例经验与消融的二手总结：<https://www.tamanna-hossain-kay.com/post/2025/02/08/deepseek/>
