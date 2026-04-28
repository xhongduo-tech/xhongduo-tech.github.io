## 核心结论

推理负载均衡的目标，不是把请求“数量”平均分给每台机器，而是把“真实计算成本”与“实例实时状态”一起纳入决策。对大模型在线推理来说，连接数只是一个很弱的近似指标，因为不同请求的上下文长度、生成长度、是否命中缓存，都会让同样“1 个请求”对应完全不同的 GPU 占用。

最容易犯的错，是把推理服务当成普通 Web 服务去分流。普通接口里，1 个请求和 1 个请求的成本常常差不多；但在大模型推理里，1 个连接可能只是 200 token 的短问答，也可能是 20k token 的长文生成。前者可能几十毫秒进入首 token，后者可能长时间占住显存、KV cache 和队列。这里的 KV cache，可以理解成“模型把前文算过的中间结果先存起来，后续继续生成时直接复用”，它直接影响延迟和吞吐。

工程上更有效的做法，是给每个候选实例算一个统一负载分数，再把新请求发给分数最低的那台。一个常见的简化形式是：

$$
c_i = \alpha p_i + \beta g_i
$$

$$
L_j = \sum_i c_i + \lambda Q_j + \mu U_j + \nu A_j
$$

$$
j^* = \arg\min_j L_j
$$

其中，$p_i$ 是 prompt tokens，意思是输入上下文长度；$g_i$ 是预期生成 tokens，意思是模型预计还要输出多长；$Q_j$ 是实例队列深度；$U_j$ 是 GPU 忙碌度；$A_j$ 是会话亲和或缓存不命中的惩罚项。核心思想很直接：不要问“这台机器现在有几个请求”，而要问“这台机器如果再接这个请求，会不会更慢、更堵、更容易失守”。

下面这张表先把“连接数”和“真实负载”的差异拉开：

| 视角 | 请求 A | 请求 B | 表面上是否一样 | 实际代价差异 |
|---|---:|---:|---|---|
| 连接数 | 1 | 1 | 一样 | 几乎没有信息量 |
| prompt tokens | 200 | 20000 | 不一样 | 长上下文预填充成本差很多 |
| 生成 tokens | 64 | 2048 | 不一样 | 解码阶段占用时长差很多 |
| KV cache 占用 | 低 | 高 | 不一样 | 长会话更吃显存 |
| 实际负载 | 轻 | 重 | 完全不同 | 不能按 1:1 处理 |

---

## 问题定义与边界

推理负载均衡要解决的问题，是把“新进入的推理请求”分配给“当前最合适的推理实例”，从而降低排队、控制尾延迟，并避免某台 GPU 因为长上下文、长生成或热点会话而过载。这里的“实例”，可以理解成一台承载模型服务的进程、Pod 或 GPU worker。

对白话一点的解释，不是看门口排了几个人，而是看每个人手里拿的是外卖还是整箱货。人数相同，工作量可能完全不同。对推理服务来说，请求数量不等于工作量，这就是问题的起点。

本文讨论的是在线推理服务的路由与调度，边界比较明确：

| 维度 | 含义 | 本文是否讨论 |
|---|---|---|
| 请求特征 | prompt 长度、生成长度、是否会话复用 | 是 |
| 实例状态 | 队列深度、GPU 利用率、显存水位、缓存命中 | 是 |
| 路由目标 | 降低延迟、提升吞吐、避免热点失衡 | 是 |
| 不解决的问题 | 模型训练、离线批处理、参数并行设计 | 否 |
| 不解决的问题 | 纯四层网络转发，不看推理语义 | 否 |

所以，本文不是在讨论“如何让 TCP 连接更均匀”，也不是在讨论“训练集群怎么调度”。重点只有一个：在线推理入口如何把请求送到更合适的模型实例。

一个常见误区是，把 Kubernetes Service、Nginx、L4 SLB 的平均分流结果，直接当成推理层面已经平衡。事实上，这些网络层组件通常不知道请求里有多少 token，也不知道哪台实例的 KV cache 已经很热。它们能把流量分开，但未必能把算力用平。

---

## 核心机制与推导

先从最简单的策略往上走。

最少连接，意思是“当前谁的活看起来最少，就先给谁”。它适合请求成本差异不大的场景，但对大模型推理不够，因为它默认每个活差不多重。加权轮询，意思是“性能更强的机器多接一点”，它解决的是机器能力不等，不解决请求成本不等。一致性哈希，意思是“同一个 key 尽量稳定落到同一台机器”，它对会话亲和很有用，但容易产生热点。

更符合推理场景的做法，是把策略从“按请求数”升级到“按请求代价 + 实例状态 + 亲和关系”打分。请求代价可以先用一个很朴素的线性模型：

$$
c_i = \alpha p_i + \beta g_i
$$

这里的 $\alpha$ 和 $\beta$ 是权重，表示“输入 token”和“输出 token”各自有多重。很多模型里，长 prompt 的预填充成本和长生成的解码成本并不完全相同，所以这两个权重通常不该写成一样。

实例总负载可以继续写成：

$$
L_j = \sum_i c_i + \lambda Q_j + \mu U_j + \nu A_j
$$

$\sum_i c_i$ 表示实例上所有在途请求的累计代价；$Q_j$ 是排队深度；$U_j$ 是 GPU 忙碌度；$A_j$ 是亲和惩罚。比如一个会话如果迁到另一台实例，原本热的 KV cache 就失效了，这种“缓存不命中”要加罚分。最终选：

$$
j^* = \arg\min_j L_j
$$

这相当于“先给每个候选机器打分，再选分数最低的那台”。

看一个玩具例子。现在有两个实例 A 和 B。

- A 上有 2 个长请求，每个约 `8000 + 512` tokens
- B 上有 5 个短请求，每个约 `256 + 128` tokens

如果只看连接数，A 只有 2 个请求，B 有 5 个请求，系统会误以为 A 更空，下一条请求可能继续发给 A。但按 token 工作量看：

- A 的总 token 近似是 $2 \times (8000 + 512) = 17024$
- B 的总 token 近似是 $5 \times (256 + 128) = 1920$

结论反过来了。A 明显更重，B 才是更合理的目标。

再把实例状态加进来，判断会更稳。下面这张表把几个常见指标并列：

| 指标 | 能否反映真实负载 | 优点 | 局限 |
|---|---|---|---|
| 连接数/请求数 | 弱 | 简单，易拿到 | 忽略长短请求差异 |
| token 代价 | 强 | 贴近真实计算量 | 需要估计生成长度 |
| 队列深度 | 中强 | 能反映拥塞 | 不能单独代表 GPU 压力 |
| GPU 利用率 | 中 | 能看忙闲 | 平均值可能掩盖排队与碎片 |
| 缓存命中/会话亲和 | 强 | 影响 TTFT 和显存复用 | 过强亲和会形成热点 |

真实工程例子是多轮对话服务。用户连续追问时，如果同一会话始终落在同一实例，前文的 KV cache 可以直接复用，TTFT，也就是“首个 token 返回时间”，通常会更低。这时一致性哈希或 session affinity 很有价值。但如果某个大客户的会话突然爆热，死守亲和就会把单机打满。所以亲和只能是偏好，不应该是不可打破的铁规则。

---

## 代码实现

实现时建议拆成两层。第一层估计请求代价，第二层根据实例状态打分并选择实例。最小版本不需要复杂模型，先把“只看连接数”升级成“看 token + 看队列 + 看 GPU + 看亲和”，收益通常就很明显。

```python
from dataclasses import dataclass

@dataclass
class Request:
    prompt_tokens: int
    expected_gen_tokens: int
    session_id: str | None = None

@dataclass
class Instance:
    name: str
    in_flight_cost: float
    queue_depth: int
    gpu_util: float           # 0.0 ~ 1.0
    sessions: set[str]

ALPHA = 1.0   # prompt 权重
BETA = 1.2    # 生成权重
LAMBDA = 200  # 队列惩罚
MU = 1000     # GPU 忙碌惩罚
NU = 500      # 会话不命中惩罚

def estimate_cost(request: Request) -> float:
    return ALPHA * request.prompt_tokens + BETA * request.expected_gen_tokens

def score_instance(instance: Instance, request_cost: float, session_id: str | None) -> float:
    affinity_penalty = 0.0
    if session_id is not None and session_id not in instance.sessions:
        affinity_penalty = NU
    return (
        instance.in_flight_cost
        + request_cost
        + LAMBDA * instance.queue_depth
        + MU * instance.gpu_util
        + affinity_penalty
    )

def pick_instance(candidates: list[Instance], request: Request) -> Instance:
    request_cost = estimate_cost(request)
    return min(
        candidates,
        key=lambda ins: score_instance(ins, request_cost, request.session_id)
    )

# 玩具例子：A 连接更少，但负载更重
req = Request(prompt_tokens=512, expected_gen_tokens=256, session_id="s-1")
a = Instance(name="A", in_flight_cost=17024, queue_depth=1, gpu_util=0.80, sessions={"x"})
b = Instance(name="B", in_flight_cost=1920, queue_depth=2, gpu_util=0.45, sessions={"s-1", "y"})

chosen = pick_instance([a, b], req)
assert estimate_cost(req) == 512 + 1.2 * 256
assert chosen.name == "B"
```

这段代码故意保持最小化，但已经体现了三个关键步骤：

1. `estimate_cost(request)`：根据请求估计成本。
2. `score_instance(instance, cost)`：把实例当前状态和新请求叠加成统一分数。
3. `pick_instance(candidates, request)`：选择分数最低的实例。

如果要继续工程化，可以把权重参数做成配置，而不是写死在代码里：

```json
{
  "load_balance": {
    "alpha_prompt": 1.0,
    "beta_generation": 1.2,
    "queue_penalty": 200,
    "gpu_penalty": 1000,
    "affinity_penalty": 500,
    "slow_start_seconds": 60
  }
}
```

如果入口层用的是 Envoy 或类似网关，常见落地路径不是“一次性重写调度系统”，而是分阶段演进：

1. 先从 `round robin` 升级到 `least request`。
2. 再为有状态会话加 `ring hash` 或 session affinity。
3. 然后在上游控制面或自定义路由层加入 token-aware 打分。
4. 最后补上 slow start、热点迁移和异常改道。

这样做的原因很现实：可观测性、指标采集和回滚机制，往往比公式本身更决定成败。

---

## 工程权衡与常见坑

推理负载均衡没有“永远最优”的单一策略，只有更适合当前业务的折中。吞吐、尾延迟、缓存命中、热点抑制，这几个目标经常互相拉扯。

最常见的坑可以直接列出来：

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 只看连接数 | 长请求被误判为轻请求，热点实例先堆满 | 把 prompt/生成 token 纳入权重 |
| 只做哈希 | 热点 key 集中到单机，单实例被打爆 | 加二级选择、热点迁移、重平衡 |
| 只看平均 GPU 利用率 | 忽略排队、TTFT 恶化、显存碎片 | 同时看队列、显存水位、TTFT |
| 新副本直接满流量 | 冷实例瞬时抖动，预热不足 | slow start，逐步加权放量 |

这里特别解释两个常被忽略的问题。

第一，只看平均 GPU 利用率会失真。GPU 平均 70% 利用率，并不代表用户体验稳定。可能某台机器上排了几个超长上下文请求，TTFT 已经很差，但平均利用率看起来并不夸张。因为“平均值”会把瞬时堵塞和平滑阶段揉在一起。对推理服务来说，队列长度、P95 TTFT、显存水位往往更敏感。

第二，亲和性不能无条件坚持。多轮对话场景里，同一会话固定到同一实例能复用 KV cache，降低 TTFT，这是收益；但如果这个会话突然变热，比如用户开启超长代码审查或连续追问，它就会成为局部热点。这时必须允许“带损迁出”，也就是短期放弃缓存命中，换整体系统稳定。否则你优化了一个会话，拖垮的是整台机器的尾延迟。

一个实用经验是，把亲和当成“软约束”，不是“硬绑定”。也就是说，优先同机，但当负载分数超过阈值时，允许改道到别的实例。

---

## 替代方案与适用边界

不同业务阶段，合适的策略不一样。不要一开始就追求最复杂的全局最优，先选能稳定上线、可观测、可回滚的方案。

| 业务特征 | 推荐策略 | 适用边界 |
|---|---|---|
| 无状态问答 | `least request` 或加权轮询 | 请求长短差异不大，先求简单上线 |
| 多轮对话 | `ring hash/session affinity` + 负载兜底 | 需要复用 KV cache，但要防热点 |
| 长上下文生成 | token-aware 调度 + 队列/GPU 水位 | 连接数指标失真最严重 |
| 热点明显的少数 key | 哈希亲和 + 动态重平衡 | 防止少数 key 长期压垮单机 |

如果业务几乎没有会话状态，目标是快速落地，一个保守而有效的起点就是 `least request`。它比轮询更接近真实负载，工程复杂度也不高。

如果业务强依赖上下文复用，比如多轮对话、代码助手、长会话 agent，那么亲和路由通常是第一优先级。因为一旦频繁迁移实例，KV cache 复用收益就会丢失，TTFT 和成本都会变差。

如果你的业务包含大量长文本生成、长文档问答、超长上下文检索增强，那么 token-aware 调度的重要性会明显高于普通问答。因为这类场景里，prompt 长度和输出长度的方差很大，连接数几乎没有解释力。

最终真正稳定的系统，通常不是“只用某一种策略”，而是三层组合：

1. 哈希保亲和。
2. 负载打分做兜底。
3. 动态重平衡处理热点与异常。

这也是为什么工程里常见的答案不是“最少连接、加权轮询、一致性哈希三选一”，而是按业务把它们拼起来。

---

## 参考资料

1. [Envoy Load Balancing](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers.html)
2. [Least Request Load Balancing Policy](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/load_balancing_policies/least_request/v3/least_request.proto)
3. [Kubernetes Virtual IPs and Service Proxies](https://kubernetes.io/docs/reference/networking/virtual-ips/)
4. [NVIDIA Triton Inference Server Batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
5. [Preble: Efficient Distributed Prompt Scheduling for LLM Serving](https://arxiv.org/pdf/2407.00023)
6. [DualMap: Balancing Load Balancing and Cache Affinity in LLM Serving](https://openreview.net/forum?id=zCadrJ32Xn)
