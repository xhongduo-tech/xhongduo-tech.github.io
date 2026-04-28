## 核心结论

多 GPU 显存均衡分配的目标，不是“所有 GPU 的总显存加起来够不够”，而是“每一张 GPU 在下一秒还能不能继续接住新请求”。这里的“接住”指的是：不触发 OOM，不把尾延迟拖到不可接受，也不因为一次错误路由把后续并发堵死。

如果只看总空闲显存，很容易得出错误结论。8 张卡总共还空 37 GB，看起来很安全；但只要其中 1 张卡因为长上下文请求积压，KV cache 先顶到边缘，这张卡上的新请求就会直接失败，或者排队等待。系统整体吞吐并不会因为“别的卡还有空”自动获救。

工程上更合理的抽象是：

$$
M_i = W_i + A_i + K_i + F_i
$$

其中，$M_i$ 是第 $i$ 张 GPU 的实时显存占用；$W_i$ 是模型权重，也就是常驻显存的参数；$A_i$ 是激活和临时张量，也就是推理过程中短时间存在的中间结果；$K_i$ 是 KV cache，也就是为后续解码复用的历史注意力状态；$F_i$ 是碎片和安全余量，也就是不能当作真正可用空间的保守预算。

| 变量 | 含义 | 是否常驻 | 对“还能否接请求”的影响 |
|---|---|---:|---|
| $W_i$ | 模型权重 | 高 | 决定基础占用，通常较稳定 |
| $A_i$ | 激活/临时张量 | 低到中 | 在 prefill 阶段波动明显 |
| $K_i$ | KV cache | 中到高 | 长会话越多，占用越容易持续增长 |
| $F_i$ | 碎片/安全余量 | 中 | 不直接做计算，但决定是否容易 OOM |

玩具例子很简单。两张 80 GB 的卡，总显存 160 GB。现在 GPU1 已用 77 GB，GPU2 已用 46 GB，新请求预计新增 6 GB KV cache。  
结论不是“总共还空 37 GB，所以肯定能接”，而是“GPU1 会到 83 GB 直接 OOM，GPU2 会到 52 GB 仍然安全”。多 GPU 问题本质上是单卡约束问题，不是总量约束问题。

真实工程例子更常见于在线对话服务。短问答和长文档总结混跑时，长请求会把某张卡的 KV cache 快速堆高。如果路由器只做轮询，或者只看请求数不看显存，这张卡会先满，随后该卡上的排队、拒绝和重试会把整体并发拉垮。显存平衡策略必须和请求路由、会话粘性一起设计。

---

## 问题定义与边界

本文讨论的是推理服务里的多 GPU 显存调度，不是训练。更准确地说，问题是：当模型权重已经加载到多张 GPU 上之后，新请求应该被路由到哪一张卡，老会话是否应该继续留在原卡，以及在什么条件下迁移会话才值得。

这里的“会话粘性”是指：同一个用户会话尽量回到上次所在的 GPU，因为那张卡上已经有可复用的 KV cache，不必从头再算。它的好处是省算力、省延迟；坏处是可能把热点长期钉死在某几张卡上。

问题边界可以先划清：

| 范围 | 包含 | 不包含 |
|---|---|---|
| 服务阶段 | 推理请求路由、KV cache 管理、尾延迟控制 | 训练反向传播、优化器状态分布 |
| 资源对象 | 单卡显存实时占用、碎片、活跃会话 | 集群级长期容量规划 |
| 目标 | 稳定高并发、减少 OOM、提升吞吐 | 只求“能跑起来” |

还需要区分几个量。很多新手把“显存用了多少”和“还能不能接请求”画等号，这是不够的。真正决定路由的不是当前占用本身，而是当前占用加上该请求的新增代价：

$$
score_i(r) = M_i + \Delta K_i(r) + \lambda \cdot C_{sticky}(r, i)
$$

这里，$\Delta K_i(r)$ 是请求 $r$ 如果落到 GPU $i$ 上预计新增的 KV cache；$C_{sticky}(r, i)$ 是粘性代价，意思是这个请求如果不回原卡，需要承担多少额外成本；$\lambda$ 是权重，控制“平衡”和“复用”之间更偏向哪一边。

白话说，这个式子表达的是：不要问“这张卡现在空不空”，要问“这个请求放上去以后，这张卡会不会变成最危险的那张”。

---

## 核心机制与推导

多 GPU 显存均衡分配的核心，是把路由问题写成“最小代价选择”，而不是“平均发请求”。

第一步是计算单卡实时占用：

$$
M_i = W_i + A_i + K_i + F_i
$$

第二步是估算新请求的新增成本。新增成本通常不只和请求数有关，更和输入长度、输出长度、阶段有关。prefill 是把整段输入一次性编码，decode 是逐 token 生成。前者更吃激活和带宽，后者更依赖 KV cache 持续增长。  
所以，同样是一个请求，在 prefill 和 decode 阶段对显存的压力并不一样。

第三步是把粘性代价放进路由分数：

$$
score_i(r) = M_i + \Delta K_i(r) + \lambda \cdot C_{sticky}(r, i)
$$

如果某个会话已经在 GPU1 上积累了 20 GB 的 KV cache，把它迁到 GPU2，不是一次“免费切换”。你至少要考虑三类成本：

1. 拷贝成本：把已有状态搬过去需要时间。
2. 重建成本：如果不拷贝而是重算，prefill 可能要从头跑。
3. 失配风险：迁过去后，GPU2 也可能很快变热点，收益并不稳定。

因此迁移判断可以写成一个工程规则：

$$
迁移收益 > 拷贝成本 + 重建成本 + 失配风险
$$

这不是某篇论文的原式，而是服务层常用的决策框架。它的意思很直接：只有在迁移能显著降低尾延迟，或者能避免即将发生的 OOM 时，迁移才值得做。

玩具例子可以具体算一下。两张 80 GB GPU：

| GPU | $W$ | $A$ | $K$ | $F$ | 当前 $M$ |
|---|---:|---:|---:|---:|---:|
| GPU1 | 26 | 0 | 48 | 3 | 77 |
| GPU2 | 26 | 0 | 18 | 2 | 46 |

新请求预计 $\Delta K = 6$ GB。  
如果落到 GPU1，分数大致是 $77 + 6 = 83$，已经越界。  
如果落到 GPU2，分数大致是 $46 + 6 = 52$，仍然安全。  
因此即使 GPU1 上已有该用户邻近流量，也不能盲目继续压过去。

真实工程例子是长会话对话系统。某些用户会持续追问，KV cache 越积越大；另一些用户只做一两轮短问答。如果你按最少连接数路由，长会话所在的 GPU 连接数可能不多，但显存已经很紧。结果是调度器误以为“这张卡还闲”，继续发请求，直到局部 OOM。  
这也是为什么“连接数均衡”不等于“显存均衡”。

---

## 代码实现

实现上建议把系统拆成 5 个模块：状态采集、请求特征提取、路由打分、会话绑定、迁移判断。不要把这些逻辑糊在一个 `select_gpu()` 里，否则后面很难调参数、定位问题。

| 模块 | 职责 | 关键输入 | 关键输出 |
|---|---|---|---|
| 状态采集 | 收集每张卡实时显存、碎片、活跃会话 | 驱动指标、运行时统计 | `gpu_state` |
| 请求特征提取 | 估算输入长度、输出长度、阶段 | 请求元数据 | `request_profile` |
| 路由打分 | 计算每张卡的代价 | `gpu_state + request_profile` | `score` |
| 会话绑定 | 判断是否复用原卡 | `session_id` | 绑定目标或候选集 |
| 迁移判断 | 决定是否值得搬迁 | 历史负载、当前风险 | migrate / keep |

下面给一个可运行的最小 Python 例子。它没有接入真实 GPU 指标，但把关键决策逻辑表达清楚了：

```python
from dataclasses import dataclass

@dataclass
class Request:
    session_id: str
    input_tokens: int
    output_tokens: int
    sticky_gpu: int | None = None

@dataclass
class GPU:
    gpu_id: int
    capacity_gb: float
    weight_gb: float
    activation_gb: float
    kv_gb: float
    frag_gb: float

    @property
    def used_gb(self) -> float:
        return self.weight_gb + self.activation_gb + self.kv_gb + self.frag_gb

    def expected_kv_gb(self, req: Request) -> float:
        # 玩具估算：每 1000 token 约消耗 1 GB KV cache
        return (req.input_tokens + req.output_tokens) / 1000.0

    def sticky_cost(self, req: Request, sticky_penalty: float) -> float:
        if req.sticky_gpu is None or req.sticky_gpu == self.gpu_id:
            return 0.0
        return sticky_penalty

def score_gpu(gpu: GPU, req: Request, sticky_penalty: float = 8.0) -> float:
    return gpu.used_gb + gpu.expected_kv_gb(req) + gpu.sticky_cost(req, sticky_penalty)

def can_accept(gpu: GPU, req: Request, reserve_gb: float = 2.0) -> bool:
    return gpu.used_gb + gpu.expected_kv_gb(req) + reserve_gb <= gpu.capacity_gb

def select_gpu(gpus: list[GPU], req: Request) -> GPU:
    candidates = [g for g in gpus if can_accept(g, req)]
    if not candidates:
        raise RuntimeError("no GPU can safely accept this request")
    return min(candidates, key=lambda g: score_gpu(g, req))

g1 = GPU(gpu_id=1, capacity_gb=80, weight_gb=26, activation_gb=0, kv_gb=48, frag_gb=3)  # 77 GB
g2 = GPU(gpu_id=2, capacity_gb=80, weight_gb=26, activation_gb=0, kv_gb=18, frag_gb=2)  # 46 GB

req = Request(session_id="s1", input_tokens=4000, output_tokens=2000, sticky_gpu=1)  # 约新增 6 GB
chosen = select_gpu([g1, g2], req)

assert g1.used_gb == 77
assert g2.used_gb == 46
assert can_accept(g1, req) is False
assert can_accept(g2, req) is True
assert chosen.gpu_id == 2
print(chosen)
```

这个例子刻意展示一个常见事实：即使请求原本“粘”在 GPU1 上，只要 GPU1 已经接近边界，系统也应该允许它打破粘性，转去 GPU2。粘性是优化项，不是绝对约束。

迁移决策可以再写成伪代码：

```text
if target_gpu will OOM soon:
    estimate migration_benefit
    estimate copy_cost
    estimate rebuild_cost
    estimate mismatch_risk

    if migration_benefit > copy_cost + rebuild_cost + mismatch_risk:
        migrate_session()
    else:
        keep_session_and_queue_or_shed_load()
else:
    keep_session_on_current_gpu()
```

真实工程里，`gpu_state[i]` 往往至少会维护这些字段：`used`, `kv`, `frag`, `active_sessions`, `prefill_load`, `decode_load`。如果系统已经拆分 prefill / decode 池，那么路由器还需要先判断请求当前处于哪个阶段，再进入对应 GPU 池选卡。

---

## 工程权衡与常见坑

显存均衡不是越平均越好。因为“完全平均”通常意味着更频繁的迁移、更弱的 KV cache 复用，以及更复杂的路由状态同步。真正目标是稳定吞吐和更低尾延迟，不是把每张卡的利用率刻意抹平到一模一样。

常见坑如下：

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 只看 `free memory` | 明明“还有空”却接不住请求 | 没把 KV 增长和碎片算进去 | 单独跟踪 `K_i` 和 `F_i` |
| 全局强粘性 | 少数热点卡长期爆满 | 所有会话都被绑定回原卡 | 粘性只在安全阈值内生效 |
| 频繁迁移长上下文 | 吞吐下降、尾延迟变差 | 拷贝或重建成本过高 | 先算迁移收益判断式 |
| 不区分 prefill / decode | 某阶段突发拖垮整池 | 两阶段资源形态不同 | 分阶段预算甚至拆池 |
| 轮询路由 | 单卡局部 OOM | 无法感知实时容量 | 使用容量感知打分 |

这里最容易被低估的是碎片。碎片的白话解释是：显存总量看着还有，但因为被切成很多不连续小块，真正的大块分配申请可能失败。  
所以有些系统会给每张卡留固定安全余量，把这部分直接并入 $F_i$，宁可保守一点，也不要把系统推到 OOM 边缘。

另一个高频坑是长短请求混跑。短请求看起来轻，但会被长请求堵在热点 GPU 前面。长请求的 KV cache 具有“越跑越重”的特性，这和普通 Web 服务里的请求数统计非常不同。  
因此，只根据“活跃会话数”做负载均衡，往往会误判。活跃 10 个短会话，可能比 2 个超长上下文会话更轻。

---

## 替代方案与适用边界

显存均衡分配不是唯一方案，而是一种在“显存压力不均”场景下更稳的方案。如果你的业务请求非常短、并发不高、KV cache 很小，那复杂的容量感知路由未必比简单轮询更划算。

几种常见方案可以放在一起看：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 轮询 | 实现最简单 | 完全不感知显存状态 | 低并发、短请求 |
| 纯最少连接数 | 比轮询更贴近负载 | 连接数不等于显存压力 | 请求长度差异不大 |
| 容量感知路由 | 能直接规避局部 OOM | 需要实时状态和估算模型 | 长短请求混跑 |
| 会话粘性 | 可复用 KV cache，降时延 | 容易形成热点 | 多轮对话、状态复用强 |
| prefill / decode 解耦 | 降低阶段互相干扰 | 系统更复杂 | 高并发、长上下文服务 |

再看适用边界：

| 条件 | 简单策略可接受 | 建议容量感知 |
|---|---|---|
| 请求长度 | 短且接近 | 长短差异大 |
| 并发水平 | 低到中 | 中到高 |
| 会话持续时间 | 单轮或短会话 | 多轮持续增长 |
| 尾延迟敏感度 | 不高 | 很高 |

玩具例子可以这样理解：小饭馆只有两张桌子、客流也均匀时，服务员按顺序带位就够了；但如果一边是两人桌，一边是大桌，还有预约和长时间占座，就不能只按先来后到排。  
真实工程里也是一样。短文本问答、低并发、模型较小的服务，轮询或最少连接数通常足够；长文档总结、RAG 长上下文、多轮对话系统，则更适合容量感知路由，必要时再结合 prefill / decode 解耦。

所以，多 GPU 显存均衡分配不是“默认最先进”的方案，而是“当单卡显存成为瓶颈时最有必要”的方案。判断是否上这套复杂度，要看你的业务是否真的会被局部热点卡拖垮。

---

## 参考资料

1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://huggingface.co/papers/2309.06180)
2. [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)
3. [TensorRT-LLM Scheduler](https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html)
4. [TensorRT-LLM Disaggregated Serving](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/disagg-serving.html)
5. [Kubernetes Virtual IPs and Session Affinity](https://kubernetes.io/docs/reference/networking/virtual-ips/)
6. [Kubernetes Topology Aware Routing](https://kubernetes.io/docs/concepts/services-networking/topology-aware-routing/)
