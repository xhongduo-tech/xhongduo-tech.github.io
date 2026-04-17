## 核心结论

Mixtral 8x7B 和 DeepSeekMoE 解决的是同一个问题：在不让每个 token 都跑完整个大模型的前提下，扩大参数容量。MoE，中文常译为“专家混合”，意思是把前馈网络拆成多组子网络，再由路由器为每个 token 只挑一部分子网络执行。

两者的主要差别不在“是不是 MoE”，而在“专家切得多细、一次激活多少个、通信要不要跨很多设备”。

| 项目 | Mixtral 8x7B | DeepSeekMoE（以 DeepSeek-V2 为代表） |
| --- | --- | --- |
| 专家组织 | 8 个粗粒度专家 | 细粒度路由专家 + 共享专家 |
| 典型路由 | top-2 | 共享专家始终激活，外加多路 top-k 路由 |
| 激活思路 | 少量专家，路径短 | 更多专家组合，专门化更强 |
| 主要瓶颈 | 专家负载不均 | all-to-all 通信与设备调度 |
| 更适合 | 单机或较小集群推理 | 多机多卡训练与大规模服务 |

结论可以压缩成一句话：Mixtral 用更简单的 top-2 粗专家，把“能跑起来”放在第一位；DeepSeekMoE 用更细的专家切分和共享专家，把“专家利用率和组合空间”放在第一位，因此上限更高，但工程成本明显更大。

---

## 问题定义与边界

先把比较对象说清楚。

Mixtral 8x7B 是一个具体模型。它在每层放 8 个前馈专家，每个 token 选 2 个专家执行，因此总参数约 46.7B，但推理时活跃参数约 13B。

DeepSeekMoE 首先是一套架构，不是单一模型名。它的核心设计是两件事：

1. 把专家切得更细。细粒度，意思是把原本一个较大的专家再拆成多个更小专家，让路由更灵活。
2. 加共享专家。共享专家，意思是每个 token 都会经过的公共专家，用来承载通用知识，减少路由专家重复学习相同内容。

因此，标题里的对比，严格来说不是“一个模型对一个模型”，而是“Mixtral 8x7B 这一类粗粒度 top-2 MoE”对“DeepSeekMoE 这一类细粒度 + 共享专家的 MoE 设计”。如果拿具体数字对比，最常见参照物是 DeepSeek-V2，因为它公开给出了 236B 总参数、21B 激活参数、128K 上下文这些工程指标。

边界也要说明：

| 比较维度 | 本文讨论 | 不讨论 |
| --- | --- | --- |
| 架构 | 路由、专家粒度、共享专家、通信 | 数据配比、RL、对齐细节 |
| 成本 | 激活参数、通信复杂度、部署门槛 | 云厂商报价、具体吞吐账单 |
| 能力 | 专家利用率、扩展性、工程可落地性 | 单一榜单上的绝对分数 |

玩具例子可以这样看。假设有 8 个专家，每次选 2 个，那么可用组合数是 $\binom{8}{2}=28$。如果有 160 个细粒度专家，每次选 6 个，可用组合数是 $\binom{160}{6}$，它会大得多。这不等于性能必然更强，但表示“路由可以表达的处理路径”明显更多。

真实工程例子更直接。一个小团队要把模型放进单机推理服务里，优先关心的是显存、延迟、框架兼容性，这时 Mixtral 更容易落地。一个大团队做多节点训练和大规模在线服务，能接受复杂通信栈，就会更在意更细的专家专门化和更大的组合空间，这时 DeepSeekMoE 更有吸引力。

---

## 核心机制与推导

MoE 层可以写成“路由器打分，再把 token 送到少数专家，最后加权求和”。

Mixtral 的简化形式是：

$$
h' = x + \sum_{i \in \mathrm{Top2}(p(x))} \tilde{p}_i(x)\,\mathrm{FFN}_i(x)
$$

这里 $x$ 是输入表示，$\mathrm{FFN}_i$ 是第 $i$ 个专家前馈网络，$p(x)$ 是路由器给 8 个专家的分数，$\tilde{p}_i$ 是对 top-2 专家重新归一化后的权重。

它的特点是：

1. 专家数不多，路由决策简单。
2. 每个 token 只走 2 条专家路径，计算和通信都比较省。
3. 容易出现专家偏载，所以通常要加 auxiliary loss。辅助损失，意思是训练时额外加一个“别让流量全挤到少数专家”的约束项。

DeepSeekMoE 的核心不是简单把 top-2 改成 top-6，而是把专家结构改了。论文给出的形式可以写成：

$$
h_t^l=\sum_{i=1}^{K_s}\mathrm{FFN}_i(u_t^l)+\sum_{i=K_s+1}^{mN}g_{i,t}\mathrm{FFN}_i(u_t^l)+u_t^l
$$

这里：

- $u_t^l$ 是第 $l$ 层第 $t$ 个 token 的输入。
- $K_s$ 是共享专家数量，共享专家始终参与。
- $mN$ 表示细分后的总专家数。
- $g_{i,t}$ 是路由权重，只对被选中的路由专家非零。
- 最后的 $+u_t^l$ 是残差连接，意思是保留输入主干，避免层间信息完全依赖专家输出。

这套设计的推导逻辑很直接。

第一步，传统 MoE 的问题不是“专家不够多”，而是“专家虽然多，但每个专家仍然太大，路由空间不够细”。如果还是 8 个大专家，top-2 的组合虽然有选择，但粒度偏粗。

第二步，把一个大专家切成多个小专家后，在相近 FLOPs 下可以选择更多、更细的组合。FLOPs 是浮点运算量，白话说就是一次前向传播要做多少乘加计算。

第三步，专家切细以后会出现新的问题：很多 token 都需要通用语言知识。如果全靠路由专家自己学，会产生冗余。共享专家就是把“所有 token 都需要的公共能力”单独抽出来，避免每个路由专家都重复学习这些内容。

所以，DeepSeekMoE 的收益不只来自“激活更多专家”，而来自“把通用能力和专门能力拆开，再让专门能力用更细粒度组合”。

这也是为什么不能只看“激活参数差不多”。如果两套系统都激活 12B 到 21B 左右参数，粗粒度 top-2 和细粒度多路路由的表达空间仍然完全不同。简单说，Mixtral 更像少量大模块拼装，DeepSeekMoE 更像大量小模块组合。

---

## 代码实现

下面先用一个可运行的玩具实现展示路由过程。它不是完整 Transformer，只保留“打分、选 top-k、归一化、聚合专家输出”这条主线。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def topk_indices(scores, k):
    pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in pairs[:k]]

def renorm(weights):
    s = sum(weights)
    return [w / s for w in weights]

def route_and_combine(token_value, router_logits, expert_bias, k, shared_bias=None):
    probs = softmax(router_logits)
    chosen = topk_indices(probs, k)
    chosen_w = renorm([probs[i] for i in chosen])

    output = token_value  # residual
    for idx, w in zip(chosen, chosen_w):
        output += w * (token_value + expert_bias[idx])

    if shared_bias is not None:
        for b in shared_bias:
            output += token_value + b

    return chosen, chosen_w, output

# Mixtral 风格：8 选 2
mixtral_logits = [0.2, 1.5, -0.3, 0.8, 2.0, -1.0, 0.1, 1.2]
mixtral_bias = [0.1 * i for i in range(8)]
chosen, weights, out = route_and_combine(1.0, mixtral_logits, mixtral_bias, k=2)

assert len(chosen) == 2
assert abs(sum(weights) - 1.0) < 1e-9
assert out > 1.0

# DeepSeekMoE 风格：路由专家 + 共享专家
deepseek_logits = [0.1 * i for i in range(10)]  # 用 10 个路由专家代替 160 个做玩具例子
deepseek_bias = [0.05 * i for i in range(10)]
chosen2, weights2, out2 = route_and_combine(
    1.0, deepseek_logits, deepseek_bias, k=3, shared_bias=[0.2, 0.3]
)

assert len(chosen2) == 3
assert abs(sum(weights2) - 1.0) < 1e-9
assert out2 > out  # 共享专家始终叠加，输出更大只是这个玩具例子的结果
```

这个例子对应两种实现思路：

| 步骤 | Mixtral 风格 | DeepSeekMoE 风格 |
| --- | --- | --- |
| 路由打分 | 对少量粗专家打分 | 对更多细粒度专家打分 |
| top-k 选择 | 常见是 top-2 | 常见是更多路由专家，同时叠加共享专家 |
| 聚合方式 | 只聚合被选中的少数专家 | 聚合共享专家和被选中的路由专家 |
| 分布式影响 | 通信相对轻 | token 分发和回收更复杂 |

如果写成接近真实框架的伪代码，大致如下：

```python
# Mixtral
scores = softmax(x @ W_router)
ids = top_k(scores, k=2)
y = x
for i in ids:
    y += norm(scores[i]) * expert[i](x)

# DeepSeekMoE
scores = softmax(x @ W_router)
ids = top_k(scores, k=6)           # 这里的 k 取决于具体模型配置
y = x
for j in shared_experts:
    y += j(x)
for i in ids:
    y += norm(scores[i]) * expert[i](x)
```

真实工程例子是分布式推理。Mixtral 常见做法是把一层或几层专家放在同一张卡或同一节点上，路由后局部执行，再把结果聚合回来。DeepSeekMoE 在大规模部署里通常需要更强的 expert parallel 通信，也就是把 token 发到拥有对应专家的设备，再把结果合并回来。这个过程常见瓶颈不是矩阵乘法，而是 all-to-all 分发和回收。

---

## 工程权衡与常见坑

第一类坑是负载均衡。

Mixtral 的风险是少数专家长期过热，其他专家利用率很低。表现出来就是某些专家 token 特别多，延迟和训练稳定性都变差。常见做法是加辅助负载均衡损失，让路由分布别过度偏向少数专家。

DeepSeekMoE 也需要负载均衡，但它的问题更复杂。因为专家更多、设备更多，失衡不只发生在“某个专家太忙”，还会发生在“某台机器太忙”或者“跨机链路太忙”。所以它的调度目标往往包含专家级、设备级、通信级三个层次。

第二类坑是通信。

Mixtral 因为每个 token 只找 2 个专家，跨设备转发的压力相对小。DeepSeekMoE 因为一个 token 可能要去多个路由专家，再叠加共享专家路径，系统如果没有高效的 dispatch/combine 内核，通信就会吞掉 MoE 省下来的算力。DeepEP 这类库本质上就是在解决这个问题：给 MoE 的 token 分发和聚合做专门优化。

第三类坑是不要把总参数和激活参数混为一谈。

Mixtral 的“46.7B 总参数，约 13B 激活参数”很好理解，但对 DeepSeek-V2 这类模型，只说“236B 总参数，21B 激活参数”仍然不够。因为 DeepSeek-V2 还叠加了 MLA。MLA 是 Multi-head Latent Attention，白话说是把注意力缓存做得更省，从而让长上下文和吞吐更可承受。所以 DeepSeek-V2 的工程收益不全来自 DeepSeekMoE，不能把它全部记到“专家更多”头上。

第四类坑是把“专家”理解成固定学科老师。

MoE 里的专家并不一定对应清晰的人类语义类别。很多研究发现，专家常常按语法模式、位置模式、局部统计规律分工，而不是稳定地出现“这个专家专门做数学，那个专家专门做代码”。因此，解释 MoE 时要把“专家”理解为“被路由器选择的一组参数子网络”，而不是拟人化角色。

---

## 替代方案与适用边界

如果需求是“单机能跑、工具链成熟、维护简单”，Mixtral 这一类粗粒度 top-2 MoE 仍然是更稳的选择。它不是绝对最强，但部署门槛低，收益也容易解释。

如果需求是“要更大模型容量、更长上下文、更强多机扩展”，DeepSeekMoE 这一类细粒度架构更有吸引力，尤其是在已经具备多节点网络、分布式调度和专用通信优化的团队里。

还要看到替代路线。

| 路线 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- |
| 稠密模型 | 结构简单，训练和推理链路直观 | 每个 token 都跑全参，成本高 | 中小模型或不追求极致扩展 |
| Mixtral 式粗粒度 MoE | 激活成本低，工程复杂度适中 | 专家粒度有限，组合空间较小 | 单机到中等规模集群 |
| DeepSeekMoE 式细粒度 MoE | 专家专门化更强，组合空间更大 | 通信与调度明显更难 | 大规模训练与服务 |
| 量化后的稠密模型 | 部署简单，硬件兼容性好 | 参数容量不如同级 MoE | 小团队、边缘部署 |

最后给一个选择规则。

如果你是零基础到初级工程师，先记住这条判断线：

1. 先问自己有没有多机多卡和高速互联。
2. 没有，就优先看 Mixtral 这类简单 MoE 或量化稠密模型。
3. 有，而且团队能处理 all-to-all、负载均衡、通信内核，再考虑 DeepSeekMoE。
4. 如果看到某个模型同时宣传“超大总参数”和“低激活成本”，一定再追问一句：它省下来的到底是计算、显存、还是 KV cache，分别靠什么做到的。

---

## 参考资料

- Mixtral of Experts, Mistral AI, arXiv: https://arxiv.org/abs/2401.04088
- DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models, DeepSeek-AI, arXiv: https://arxiv.org/abs/2401.06066
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, DeepSeek-AI, arXiv: https://arxiv.org/abs/2405.04434
- DeepEP: an efficient expert-parallel communication library, GitHub: https://github.com/deepseek-ai/DeepEP
