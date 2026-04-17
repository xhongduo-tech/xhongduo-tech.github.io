## 核心结论

共享专家（Shared Expert，白话讲就是“每个 token 都一定会经过的公共专家”）的核心作用，不是增加激活参数数量，而是把原本分散在大量路由专家中的通用知识收拢到少数固定通道里。这样，路由专家就不必反复学习“语言共性”“基础语义”“常见模式”这些公共内容，而可以把容量集中到更细的专业能力上。

DeepSeekMoE 的设计可以概括成一句话：**通用知识常驻，专业知识稀疏**。在一个 MoE 层里，先让 $K_s$ 个共享专家始终工作，再让路由器从剩余专家里选出 $mK-K_s$ 个路由专家参与计算。最终输出不是“只看被选中的专家”，而是：

$$
\text{MoE输出}=\text{共享专家输出}+\text{路由专家输出}+\text{残差}
$$

这带来两个直接结果：

| 设计 | 通用知识放在哪里 | 专业知识放在哪里 | 参数冗余 | 路由压力 |
|---|---|---|---|---|
| 纯路由 MoE | 分散在多个专家中 | 同样分散在多个专家中 | 高 | 高 |
| 共享专家 + 路由专家 | 集中在少数共享专家中 | 集中在路由专家中 | 低 | 更可控 |

玩具例子可以这样理解：假设一层里有 2 个共享专家和 128 个路由专家，每个 token 总共激活 8 个专家。那么它的处理顺序不是“从 130 个里随机挑 8 个”，而是“先固定走 2 个共享专家，再从 128 个路由专家里选 6 个”。这相当于每个 token 先读完两本“通用百科”，再去找六位“领域老师”补充细节。

这类结构的关键价值在于参数效率。dots.llm1 报告中的 `2 个共享 + 128 个路由` 配置，在总激活数仍为 8 的前提下，能匹配甚至超过更早的 `160 个纯路由专家` 配置。原因不是“共享专家更神奇”，而是**通用知识与专业知识被解耦后，路由专家不再浪费容量去重复学习公共特征**。

---

## 问题定义与边界

要理解共享专家，先要明确传统 MoE 的问题到底出在哪里。

MoE（Mixture of Experts，白话讲就是“很多个前馈网络里只激活少数几个”）的初衷，是用更大的总参数换更低的单 token 计算量。但在传统设计里，所有专家都处于同一地位：每个 token 进来后，由路由器挑几个专家来处理。问题在于，真实语言任务里有大量“所有 token 都需要”的基础能力，例如语法组合、常见词义、句法边界、代码中的基础模式。这些能力不属于某个专业领域，却会被迫在多个路由专家中重复学习。

于是会出现两个后果：

1. 多个专家内部都含有相似的公共表示，造成参数重复。
2. 路由器为了覆盖通用场景，必须把许多本不该专业化的模式也纳入竞争，导致专家分工不够干净。

DeepSeekMoE 的边界定义非常明确：它要解决的不是“如何让所有专家更平均”，而是“如何让通用知识有固定承载者，让路由专家只处理真正需要稀疏化的专业能力”。

因此它把专家分成两类：

| 类型 | 是否始终激活 | 负责内容 | 是否参与 Top-k 竞争 |
|---|---|---|---|
| 共享专家 | 是 | 跨领域通用知识 | 否 |
| 路由专家 | 否 | 专业化、细粒度能力 | 是 |

这意味着一层的职责顺序变成了：

1. 输入先经过共享专家，提取所有 token 都需要的公共信息。
2. 路由器再根据当前 token 的特征，从路由专家中选出少数专业专家。
3. 两部分输出相加，再加上残差连接。

可以用一个简化流程图表示：

```text
token hidden state u_t
        |
        v
+-------------------+
|  K_s 个共享专家    |   <- 始终激活
+-------------------+
        |
        +------------------+
        |                  |
        v                  v
共享输出 sum_shared     Router 打分
                           |
                           v
                  Top-(mK-K_s) 路由专家
                           |
                           v
                      路由输出 sum_routed
        \                  /
         \                /
          +------相加-----+
                 |
                 v
             + residual
                 |
                 v
              h_t^l
```

对白话类比要保持克制，但一个初学者容易理解的版本是：一个班级只安排两位老师负责所有学生都要学的基础课，其余老师只负责选修课。这样，选修课老师就不需要浪费时间重复讲加减乘除。

这里的适用边界也要说清楚。共享专家适用于“通用能力和专业能力同时存在”的大模型场景，尤其是多领域语言建模、代码与自然语言混合训练、长尾任务明显的预训练。如果任务本身极窄，比如只有单一垂直语料、模式高度固定，那么共享专家带来的收益可能会缩小，因为“公共知识”和“专业知识”的差异本来就不大。

---

## 核心机制与推导

DeepSeekMoE 的核心公式可以写成：

$$
h_t^l = \sum_{i=1}^{K_s} \mathrm{FFN}_i(u_t^l)
+ \sum_{i=K_s+1}^{mN} g_{i,t}\cdot \mathrm{FFN}_i(u_t^l)
+ u_t^l
$$

其中：

- $u_t^l$：第 $l$ 层、第 $t$ 个 token 的输入隐藏状态。
- $h_t^l$：这一层 MoE 的输出。
- $\mathrm{FFN}_i$：第 $i$ 个专家对应的前馈网络。
- $K_s$：共享专家数量。
- $mN$：专家总数。
- $mK$：每个 token 激活的总专家数。
- $g_{i,t}$：第 $i$ 个路由专家对 token $t$ 的门控权重。

路由部分的权重通常来自亲和度分数 $s_{i,t}$。亲和度可以理解为“这个 token 与某个专家的匹配程度”，常见形式是对 token 表示与专家向量做相似度计算，再做 Softmax：

$$
s_{i,t}=\mathrm{Softmax}_i\left((u_t^l)^\top e_i^l\right)
$$

其中 $e_i^l$ 是第 $i$ 个专家在该层的路由向量。真正进入计算的门控值是：

$$
g_{i,t}=
\begin{cases}
s_{i,t}, & s_{i,t}\in \mathrm{Top}_{mK-K_s}\{s_{j,t}\} \\
0, & \text{otherwise}
\end{cases}
$$

这组公式表达了三个事实。

第一，**共享专家不参与稀疏筛选**。它们无条件对每个 token 生效，所以前半项始终存在。

第二，**路由专家只在剩余预算里竞争**。如果总共激活 $mK$ 个专家，里面已经固定有 $K_s$ 个共享专家，那么路由器只需要在专业专家里选出 $mK-K_s$ 个。

第三，**残差仍然保留原输入**。这保证即便专家路由不完美，层输出也不会完全偏离主干网络。

下面给一个符号速查表：

| 符号 | 含义 | 直观理解 |
|---|---|---|
| $K_s$ | 共享专家数量 | 每个 token 必走的公共通道数 |
| $mN$ | 专家总数 | 这一层一共有多少个专家 |
| $mK$ | 激活专家总数 | 每个 token 实际使用多少个专家 |
| $s_{i,t}$ | 亲和度分数 | token 与专家的匹配度 |
| $g_{i,t}$ | 门控权重 | 真正参与加权求和的专家权重 |

把机制拆成三步，会更容易看清。

**第一步：共享输出。**  
对任意 token，都计算所有共享专家的 FFN 输出并求和。这一步相当于固定的“基础语义提取”。

**第二步：路由选择。**  
路由器只看路由专家集合，为当前 token 算出匹配分数，然后取 Top-$(mK-K_s)$。这是“专业能力分配”。

**第三步：合并与残差。**  
共享输出与路由输出相加，再加输入残差，形成最终层输出。

玩具例子：假设一层共有 130 个专家，其中前 2 个是共享专家，后 128 个是路由专家；每个 token 总激活 8 个专家。那么对任意 token：

- 共享专家 1、2 总会执行；
- 路由器只在 128 个候选里选 6 个；
- 最终输出是 `2 个共享输出 + 6 个路由输出 + 原输入残差`。

这个设计为什么会减少冗余？因为如果没有共享专家，那些所有 token 都需要的公共模式只能被多个路由专家各自学一份。随着专家数增长，重复的“基础能力副本”会越来越多。引入共享专家后，这部分能力被显式收口到固定通道，路由专家就更容易出现清晰分工。

---

## 代码实现

下面用一个可运行的 Python 例子，演示“共享专家始终激活，路由专家做 Top-k”的最小实现。这里不依赖深度学习框架，只用 `numpy` 表达前向过程，重点是机制而不是训练性能。

```python
import numpy as np

np.random.seed(0)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

class LinearFFN:
    def __init__(self, d_model, d_hidden):
        self.w1 = np.random.randn(d_model, d_hidden) / np.sqrt(d_model)
        self.w2 = np.random.randn(d_hidden, d_model) / np.sqrt(d_hidden)

    def __call__(self, x):
        h = np.maximum(0, x @ self.w1)
        return h @ self.w2

class SharedMoE:
    def __init__(self, d_model, d_hidden, num_shared, num_routed, topk_total):
        assert topk_total >= num_shared
        self.d_model = d_model
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.topk_total = topk_total
        self.topk_routed = topk_total - num_shared

        self.shared_experts = [LinearFFN(d_model, d_hidden) for _ in range(num_shared)]
        self.routed_experts = [LinearFFN(d_model, d_hidden) for _ in range(num_routed)]

        # 路由器只为路由专家打分，共享专家不参与竞争
        self.router_w = np.random.randn(d_model, num_routed) / np.sqrt(d_model)

    def forward(self, x):
        # x: [batch, d_model]
        batch = x.shape[0]

        shared_out = np.zeros_like(x)
        for expert in self.shared_experts:
            shared_out += expert(x)

        scores = softmax(x @ self.router_w)  # [batch, num_routed]

        routed_out = np.zeros_like(x)
        topk_indices = np.argsort(scores, axis=-1)[:, -self.topk_routed:]

        for b in range(batch):
            for idx in topk_indices[b]:
                weight = scores[b, idx]
                routed_out[b] += weight * self.routed_experts[idx](x[b:b+1])[0]

        # 残差连接
        y = shared_out + routed_out + x
        return y, scores, topk_indices

def load_balance_loss(scores, topk_indices, num_routed):
    # 一个简化版负载均衡指标：统计每个路由专家被选中的频率
    counts = np.zeros(num_routed, dtype=np.float64)
    for row in topk_indices:
        for idx in row:
            counts[idx] += 1.0
    counts /= counts.sum() + 1e-12

    target = np.ones(num_routed) / num_routed
    loss = np.sum((counts - target) ** 2)
    return loss

moe = SharedMoE(d_model=8, d_hidden=16, num_shared=2, num_routed=8, topk_total=4)

x = np.random.randn(3, 8)
y, scores, topk_indices = moe.forward(x)

assert y.shape == x.shape
assert scores.shape == (3, 8)
assert topk_indices.shape == (3, 2)   # topk_total=4，其中2个是共享专家，所以只选2个路由专家

lb = load_balance_loss(scores, topk_indices, num_routed=8)
assert lb >= 0.0

print("output shape:", y.shape)
print("topk routed indices:", topk_indices)
print("balance loss:", lb)
```

这段代码对应的逻辑是：

1. `shared_out` 直接把所有共享专家的输出求和。
2. `scores = softmax(x @ router_w)` 只给路由专家打分。
3. `topk_indices` 选出 `topk_total - num_shared` 个路由专家。
4. 最后做 `shared_out + routed_out + x`。

这正是 DeepSeekMoE 的结构分解。

如果把它写成更贴近工程实现的伪代码，可以简化成：

```python
shared_out = sum(shared_ffn_i(u) for i in range(K_s))
routing_scores = router(u)                 # only routed experts
topk_idx, topk_weight = topk(routing_scores, k=mK-K_s)
routed_out = combine(routed_ffn[topk_idx], topk_weight, u)
output = shared_out + routed_out + u
aux_loss = balance_loss(topk_idx, topk_weight)   # only routed experts
```

这里有一个非常重要的工程原则：**负载均衡损失只约束路由专家，不约束共享专家**。因为共享专家的定义就是“所有 token 都应经过它们”。如果把共享专家也放进 load balancing，优化器会试图让它们“更均匀”“更稀疏”或“更像普通专家”，这会直接破坏共享专家存在的意义。

真实工程例子里，dots.llm1 采用 `2 个共享专家 + 128 个路由专家`，每个 token 激活 `8` 个专家，其中 `2` 个共享、`6` 个路由。这个结构在总激活预算不变时，把路由专家数从更早的 `160` 纯路由设计缩减到 `128`，但并没有牺牲效果，说明共享专家确实在替路由专家承担公共表示学习。

---

## 工程权衡与常见坑

共享专家不是“免费收益”。它解决了一类问题，也引入了新的工程约束。

第一类权衡是**职责分离是否足够干净**。如果共享专家输出很弱，它们就只是名义上的共享；如果共享专家输出过强，路由专家可能退化成小修小补。工程上通常需要通过归一化和尺度控制，让两部分既能叠加，又不互相淹没。

第二类权衡是**负载平衡只该发生在路由专家侧**。普通 MoE 常会加入 auxiliary loss，让专家使用频率更均匀。但共享专家本来就必须一直被用到，所以它们不应参与这类平衡目标。

第三类权衡是**路由容量与共享容量的比例**。共享专家太少，无法承接足够多的公共知识；共享专家太多，又会侵占本应用于专业化的激活预算。

下面把常见工程条件、效果和坑放在一张表里：

| 工程条件 | 共享专家设计变化 | 效果 | 常见坑 + 规避 |
|---|---|---|---|
| 总激活预算固定，例如每 token 激活 8 个专家 | 从纯路由 8 个，改成 2 共享 + 6 路由 | 通用知识集中，路由更专注 | 共享专家太强会压制路由输出；用 RMSNorm 或缩放因子对齐尺度 |
| 路由专家很多，例如 128 或 160+ | 引入少量共享专家替代部分路由名额 | 降低公共知识重复学习 | 仍把共享专家放进负载均衡，会破坏“常驻专家”角色 |
| 长期大规模预训练 | 共享专家吸收跨域稳定模式 | 收敛更稳，专家分工更清晰 | 若门控全低精度，路由波动更大；实际工程常保留更高精度 gate |
| 多领域混合语料 | 共享专家承接语言共性，路由专家承接领域差异 | 参数利用率更高 | 如果语料分布极不均衡，少数路由专家仍可能过载，需要单独调路由正则 |

尺度问题尤其容易被低估。因为最终输出是直接相加：

$$
h = y_{\text{shared}} + y_{\text{routed}} + x
$$

如果 $y_{\text{shared}}$ 和 $y_{\text{routed}}$ 的方差不在同一量级，训练就容易震荡。常见处理方式有：

1. 在专家输出前后加 RMSNorm。
2. 对共享或路由分支增加可学习缩放因子。
3. 保持门控计算精度更高，减少 Top-k 抖动。

dots.llm1 的工程报告给了一个更完整的真实例子：它在 `8×40GB GPU` 环境下，采用 `interleaved 1F1B pipeline`，预训练数据规模约 `11.2T tokens`，并使用 `3-stage` 数据清洗和 `100% FP32 gate`。这些细节说明，共享专家能否真正发挥价值，并不只取决于公式，还依赖路由稳定性、训练精度和大规模数据下的负载控制。

对于初学者，一个最容易踩的坑是：看到共享专家总是激活，就误以为“那把共享专家数量继续加大就更好”。这通常不成立。共享专家占用的是每个 token 的固定计算预算。共享部分变多，意味着可用于专业化选择的路由名额减少，模型可能变成“通用很强，细分能力不够”。

---

## 替代方案与适用边界

共享专家并不是所有 MoE 的默认最优解。是否需要它，取决于任务是否真的存在“稳定的公共知识层”和“明显的专业化长尾”。

可以把常见方案做一个决策表：

| 场景 | 推荐架构 | 理由 |
|---|---|---|
| 大规模通用预训练，多领域混合语料 | 共享专家 + 路由专家 | 通用模式很多，适合把公共知识收口 |
| 代码、数学、自然语言混合训练 | 共享专家 + 路由专家 | 基础语言结构共享，专业推理可路由分工 |
| 单一垂直领域，语料风格高度一致 | 纯路由 MoE 或小型密集 FFN | 通用/专业边界不明显，共享收益有限 |
| 参数预算非常紧，但想提高总容量 | 共享专家 + 路由专家 | 可减少重复参数，提高有效容量利用率 |
| 极短上下文、低多样性任务 | 纯路由或密集层 | 共享通道可能带来额外固定成本但收益不大 |

把 `160 纯路由` 和 `2 共享 + 128 路由` 放在一起比较，会更容易看出边界：

| 方案 | 总体思路 | 优势 | 劣势 |
|---|---|---|---|
| 160 纯路由 | 所有能力都通过竞争式路由获得 | 结构简单，统一处理 | 通用知识重复学习，专家分工容易混杂 |
| 2 共享 + 128 路由 | 通用能力固定承载，专业能力稀疏选择 | 参数更省，分工更清晰 | 实现更复杂，需要处理尺度与正则边界 |

所以，什么时候不该用共享专家？至少有三种情况：

1. 任务非常垂直，几乎没有可抽象的跨样本通用模式。
2. 模型规模较小，专家数本来就不多，重复学习问题尚不突出。
3. 工程团队还没有能力稳定处理 Top-k 路由、平衡损失、精度管理等问题，此时增加共享机制可能先带来复杂度成本。

反过来说，如果模型已经进入“大量专家、多领域数据、希望提升参数效率”的区间，那么继续单纯把路由专家数量从 128 堆到 160、200，往往不是最优路径。因为这只是把通用知识复制到更多位置，而不是改变知识分布方式。共享专家的价值，正体现在**改变知识组织结构**，而不只是增加专家总数。

---

## 参考资料

1. DeepSeekMoE, ACL 2024：给出共享专家公式、门控定义与相关消融，适合看机制原型。  
   https://aclanthology.org/2024.acl-long.70/

2. DeepSeek Architecture “Aha Moment”：解释共享专家降低通用知识冗余的设计动机，适合先建立直觉。  
   https://www.infosys.com/iki/techcompass/deepseek-architecture-aha-moment.html

3. dots.llm1 Technical Report：提供 `2 共享 + 128 路由`、激活数、训练配置与工程结果，适合看落地细节。  
   https://www.researchgate.net/publication/392515125_dotsllm1_Technical_Report
