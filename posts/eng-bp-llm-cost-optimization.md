## 核心结论

LLM 应用的成本优化，本质上不是“把模型换便宜一点”这么简单，而是在满足服务质量的前提下，系统性地降低每个成功请求的总消耗。这里的“服务质量”通常指响应正确率、延迟、稳定性和峰值承载能力。

最重要的结论有三条。

第一，成本必须按“每个成功输出”来算，而不是只看模型单价。一次请求真正付费的不只是输入和输出 token，还包括重试、回退、验证失败后的补跑，以及峰值时低效调度带来的浪费。

第二，最有效的优化手段通常不是单点技巧，而是三层组合：批处理、缓存、模型路由。批处理就是把多个请求凑成一批一起跑，让同一块 GPU 同时服务更多请求；缓存就是把重复计算结果复用；模型路由就是先判断任务难度，再决定交给小模型还是大模型。三者一起作用，才能同时压低单位成本。

第三，每一层优化都在交换东西。批处理换来更高吞吐，但会增加排队等待；缓存能省钱，但只对重复问题有效；模型路由能降单价，但会带来误判风险。工程上不能凭感觉优化，必须用指标验证收益。

一个新手能立刻理解的玩具例子是在线客服。假设每个用户问题都直接送进最强模型，且每次请求都独占一段推理资源，那么流量一上来，成本会线性飙升。如果改成“先查缓存，再把短时间内到达的请求拼成小批次，再把简单问题分配给轻模型”，效果就像拼车：同样一张显卡，空闲时间被摊给更多请求，单位请求成本下降。

| 手段 | 主要降低什么 | 对延迟的影响 | 实施难度 |
| --- | --- | --- | --- |
| 批处理 | 降低单位请求的算力空转成本 | 可能上升，因为要等待凑批 | 中 |
| 缓存 | 降低重复请求的 token 与推理成本 | 通常降低 | 中 |
| 模型路由 | 降低平均模型单价 | 可能不变，也可能因误判重试上升 | 中到高 |

---

## 问题定义与边界

先把问题定义清楚。LLM 成本优化讨论的不是“这个模型贵不贵”，而是“一个成功返回给用户的结果，平均花了多少钱”。这一定义比“每百万 token 价格”更贴近真实系统，因为线上系统不是理想状态：会有失败、重试、超时、回退、缓存命中和未命中。

可以把成本拆成四个核心变量：

| 变量 | 含义 | 在公式中的位置 | 为什么必须监控 |
| --- | --- | --- | --- |
| 输入 token | 用户输入、系统提示、检索上下文等总长度 | 成本分子 | 提示膨胀会直接放大成本 |
| 输出 token | 模型生成的回答长度 | 成本分子 | 冗长回答会持续吞预算 |
| 重试 token | 结构化失败、超时、回退导致的额外 token | 成本分子 | 常被忽略，但线上很常见 |
| 成功数 | 最终成功交付给用户的请求数 | 成本分母 | 成功率下降会抬高单位成本 |

因此，边界要先明确：

1. 你优化的是在线服务还是离线批任务。在线服务更看重尾延迟，不能无限等批次凑齐；离线任务更适合 aggressive batching，也就是更激进地做大批量调度。
2. 你的流量是否稳定。流量稳定时，批处理更容易持续命中合适批大小；流量抖动大时，固定批大小往往失效。
3. 你的请求是否重复。若大量问题相似，缓存收益很高；若每次输入都高度定制，缓存作用就会有限。
4. 你的任务复杂度是否分层明显。如果简单问题和复杂问题混在一起，而你又能较准确地区分它们，模型路由的收益会很大。

一个最小数值例子可以说明为什么“单位成本”比“模型单价”更重要。假设有 10 个请求，每个请求平均 1000 个 token，单价是每千 token \$0.012，那么一轮成本是：

$$
10 \times \frac{1000}{1000} \times 0.012 = 0.12\ \text{美元}
$$

如果通过缓存让其中 5 个请求直接复用结果，再通过批处理降低推理空转，平均每请求成本可能从 \$0.012 降到 \$0.002 到 \$0.004 这个量级。这里省下来的不只是 token，还包括资源调度效率。

所以，问题边界不是“怎么省 token”这么窄，而是“怎样在给定延迟、正确率和复杂度约束下，降低每个成功输出的总成本”。

---

## 核心机制与推导

先看最核心的度量公式：

$$
Cost = \frac{T_{in}\times P_{in} + T_{out}\times P_{out} + T_{retry}\times P_{retry}}{N_{success}}
$$

这里：

- $T_{in}$ 是输入 token 总数，也就是所有送进模型的提示长度总和。
- $P_{in}$ 是输入 token 单价，也就是每个输入 token 的价格。
- $T_{out}$ 是输出 token 总数，也就是模型生成内容的长度总和。
- $P_{out}$ 是输出 token 单价。很多商业模型里，输出 token 比输入 token 更贵。
- $T_{retry}$ 是重试额外消耗的 token，包括超时重跑、格式校验失败重跑、切换模型回退等。
- $N_{success}$ 是最终成功完成的请求数。

这个公式的重要性在于，它告诉你优化有三条路：

1. 降低分子，也就是减少输入、输出和重试总消耗。
2. 提高分母，也就是减少失败和无效请求，让更多请求一次成功。
3. 用更低的单价承载同等质量，也就是用路由或量化把任务分流到更便宜的推理路径。

### 1. 批处理为什么能省钱

批处理的原理不是减少 token，而是提高单位时间内的有效吞吐。通俗说，同一块 GPU 在处理单个请求时，往往有空隙没有被充分利用；把多个请求一起送进去，可以让更多矩阵计算并行执行。这样每个请求分摊到的硬件空转成本更低。

但批处理不免费。为了凑够一个批次，系统需要等待更多请求到达，因此延迟会上升。工程上通常不是追求“最大批”，而是追求“在延迟预算内的最优批”。

### 2. 缓存为什么能直接降低分子

缓存分两类。

第一类是结果缓存，也就是同样的问题直接返回同样答案。比如“营业时间是什么”“退款规则是什么”这种重复问题。

第二类是前缀缓存或 KV 缓存。KV 缓存可以理解为“模型已经读过的上下文记忆片段”，当多个请求共享长系统提示、长知识库前缀或长对话历史时，前面的计算可以复用，不用每次从头再算。

缓存之所以强，是因为它不是“便宜一点算”，而是“尽量不算”。

### 3. 模型路由为什么通常比统一降级更稳

很多团队一开始的省钱思路是“全部换小模型”。这很危险，因为复杂任务一旦失败，后续重试、回退到大模型、人工兜底，反而让总成本上升。

更稳的方法是模型路由。模型路由就是先做一次任务难度判断，简单请求走便宜模型，复杂请求走贵模型。这样优化的是平均成本，而不是盲目压低单次价格。

### 4. 提示膨胀为什么是最容易被忽略的成本杀手

线上系统最常见的问题不是模型突然涨价，而是 prompt creep，也就是提示逐渐膨胀。每加一点历史、规则、Few-shot 样例、检索文本，看起来都合理，但累积起来会显著增加成本。

一个真实可算的例子：某系统月调用 750 万次，平均输入从 900 token 增长到 1450 token，单价按每千 token \$0.012 计算，那么仅输入增加部分的月额外成本约为：

$$
\frac{(1450 - 900) \times 7{,}500{,}000}{1000} \times 0.012
= 49{,}500\ \text{美元}
$$

这说明一件事：优化 prompt 长度，往往比讨论“换不换模型”更直接。

### 5. 真实工程例子

设想一个多 Agent 平台。Agent 就是“会自动拆任务和调用工具的智能流程单元”。用户一个复杂请求，可能触发主 Agent、检索 Agent、总结 Agent、代码 Agent，最终形成一条链路。

如果每个 Agent 都独立调用大模型，且没有共享缓存、没有批处理、没有路由，这条链路的成本会近似按调用次数线性累加，峰值时还会因为资源抢占造成重试。此时真正的优化重点不是单个 Agent 提示词，而是整个系统层面：

- 把共享前缀做缓存
- 把同类请求聚合批处理
- 把低风险子任务下放到轻模型
- 对结构化输出失败设置更便宜的回退路径

这就是“成本优化是系统问题，不只是模型问题”。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它模拟三个动作：先查缓存，再做简单批处理，最后按复杂度路由模型。这里不依赖真实模型 API，重点是把决策流程写清楚。

```python
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Request:
    signature: str
    prompt: str
    complexity: int  # 1-10，数值越大表示越复杂


class Cache:
    def __init__(self):
        self._store: Dict[str, str] = {}

    def has(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> str:
        return self._store[key]

    def put(self, key: str, value: str) -> None:
        self._store[key] = value


class Router:
    def select_model(self, complexity: int) -> str:
        # 简单请求走小模型，复杂请求走大模型
        return "small-model" if complexity <= 4 else "large-model"


class Batcher:
    def __init__(self, max_batch_size: int = 3):
        self.max_batch_size = max_batch_size
        self.queue: List[Request] = []

    def append(self, req: Request) -> None:
        self.queue.append(req)

    def ready(self) -> bool:
        return len(self.queue) >= self.max_batch_size

    def flush(self) -> List[Request]:
        batch = self.queue[: self.max_batch_size]
        self.queue = self.queue[self.max_batch_size :]
        return batch


def run_model(model_name: str, prompts: List[str]) -> List[str]:
    # 这里用字符串拼接代替真实推理
    return [f"{model_name}:{p[:20]}" for p in prompts]


def handle_request(req: Request, cache: Cache, batcher: Batcher, router: Router):
    if cache.has(req.signature):
        return {"status": "cache_hit", "output": cache.get(req.signature)}

    batcher.append(req)

    if not batcher.ready():
        return {"status": "queued", "output": None}

    batch = batcher.flush()
    avg_complexity = sum(r.complexity for r in batch) / len(batch)
    model = router.select_model(avg_complexity)
    outputs = run_model(model, [r.prompt for r in batch])

    result = {}
    for item, output in zip(batch, outputs):
        cache.put(item.signature, output)
        result[item.signature] = output

    return {"status": "batch_run", "model": model, "outputs": result}


cache = Cache()
router = Router()
batcher = Batcher(max_batch_size=2)

r1 = Request(signature="a", prompt="hello world", complexity=2)
r2 = Request(signature="b", prompt="explain transformer attention", complexity=7)

resp1 = handle_request(r1, cache, batcher, router)
assert resp1["status"] == "queued"

resp2 = handle_request(r2, cache, batcher, router)
assert resp2["status"] == "batch_run"
assert resp2["model"] == "large-model"
assert "a" in resp2["outputs"]
assert "b" in resp2["outputs"]

resp3 = handle_request(r1, cache, batcher, router)
assert resp3["status"] == "cache_hit"
assert resp3["output"].startswith("large-model:")
```

这段代码虽然是玩具版本，但已经体现了真实工程中的三个关键步骤：

1. 缓存命中直接返回，避免重复推理。
2. 请求先入队，达到批大小再统一执行。
3. 根据平均复杂度选择模型，而不是所有请求统一走同一条路径。

如果映射到真实系统，还会再加四类指标：

| 指标 | 作用 | 动态调节什么 |
| --- | --- | --- |
| 平均批大小 | 判断批处理是否真正发生 | 调整最大批大小或等待时间 |
| P95 延迟 | 观察高位延迟是否失控 | 缩短凑批等待窗口 |
| Cache Hit Rate | 观察缓存是否有效 | 调整缓存粒度和过期策略 |
| 模型路由比例 | 观察小模型与大模型分流是否合理 | 调整复杂度阈值 |

一个真实工程例子是客服问答系统。流程通常是：

- 用户输入先做规范化，得到缓存签名
- 若命中 FAQ 或结果缓存，直接返回
- 若未命中，则进入 20 到 100 毫秒的小窗口等待凑批
- 批次形成后，根据问题类型决定走轻模型还是重模型
- 若输出不符合结构化要求，再触发一次受控重试
- 记录 token、延迟、重试率和路由结果，供后续调参

这和“每次来一个请求就直接调用最强模型”相比，复杂一些，但成本表现通常更稳定。

---

## 工程权衡与常见坑

成本优化最难的地方，不是知道有哪些手段，而是理解它们会互相牵连。

第一组权衡是吞吐和延迟。批处理越强，GPU 利用率通常越高，但用户等待时间也越长。对于在线系统，你通常会设一个最大等待窗口，例如 30 毫秒或 50 毫秒，超过就发车，不继续等更大批。

第二组权衡是准确率和单价。让更多请求走小模型，看上去能立刻降本，但如果误判导致复杂请求频繁失败，后续回退到大模型的重试成本会迅速吃掉收益。

第三组权衡是系统复杂度。缓存、路由、回退、批调度都要加状态和监控。若团队还没有基本可观测性，盲目叠加优化层，最后往往是不知道哪一层真的在省钱。

常见坑可以总结为下面这张表：

| 问题 | 监控指标 | 应对策略 |
| --- | --- | --- |
| Prompt creep | 平均输入 token、P95 输入 token | 精简系统提示，移除无效样例，设置 token 报警 |
| 输出过长 | 平均输出 token、截断率 | 限制回答格式，控制最大输出长度 |
| 结构化失败导致重试 | 重试率、解析失败率 | 降低输出自由度，增加格式校验，分离思考与结构化输出 |
| 峰值时全部打到大模型 | 路由比例、负载水位 | 设置降级阈值，优先保障核心路径 |
| 缓存命中率低于预期 | Cache Hit Rate、签名分布 | 优化归一化规则，避免把等价请求打散 |
| 批处理效果差 | 平均批大小、GPU 利用率 | 调整窗口长度，区分高峰和低峰策略 |

一个新手最容易踩的坑，是只看“模型单价”而不看“重试后总价”。例如高峰期不做回退控制，所有请求都送进最强模型，GPU 过载后延迟上升、超时增多、重试变多，最终既慢又贵。正确做法通常是预先设负载阈值：当系统负载高于阈值时，简单问题强制降到轻模型，复杂问题才保留给重模型，同时清理提示，防止 token 膨胀。

---

## 替代方案与适用边界

不是所有场景都适合同一套成本优化策略。

如果你是高并发在线服务，批处理、路由和缓存通常是主线，因为请求多、重复度高、吞吐压力明显。这类场景中，调度器价值最大。

如果你是低频但高价值任务，例如合同审查、医学辅助总结、关键代码生成，那么单次请求更贵但容错更低。这里可以少做激进路由，优先保证质量，再去优化 prompt 长度、上下文裁剪和结构化输出重试策略。

如果你流量波动极大，还有几种替代部署方式：

| 方案 | 可用性特点 | 单位成本特点 | 延迟特点 |
| --- | --- | --- | --- |
| Serverless Inference | 弹性强，适合波峰波谷明显 | 峰值友好，但长期高负载未必最省 | 冷启动可能较慢 |
| 专用 GPU | 稳定，适合持续高负载 | 利用率高时最省 | 延迟通常更稳 |
| Hybrid 路由 | 结合固定资源与弹性资源 | 综合成本通常更平衡 | 设计得当时兼顾稳定与弹性 |

还有一个常见替代方案是边缘缓存。它适合重复查询非常多的场景，比如企业知识问答、统一政策问答、标准客服流程。边缘缓存的优点是能在离用户更近的位置直接返回结果，减少中心推理压力；缺点是对个性化请求帮助有限。

再举一个适用边界明确的例子。对话系统流量不稳时，优先级调度通常比统一批处理更有效。简单问题先命中缓存或小模型，复杂问题再升级到大模型。这种做法的本质不是“便宜请求先处理”，而是“把低风险请求放到更低成本路径”，从而保护昂贵模型的容量。

因此，成本优化没有单一答案。判断标准只有一个：你的业务是否能接受该优化带来的延迟、复杂度或质量波动。如果不能接受，这个优化即使理论上省钱，也不适合上线。

---

## 参考资料

- Yotta Labs, *How to Optimize LLM Inference for Throughput and Cost* (2026)
- Hakia, *LLM Inference Optimization Techniques: Speed & Cost Guide* (2026)
- Burnwise, *The Complete Guide to LLM Cost Optimization* (2026)
- LayerLens, *LLM Cost Optimization: What Actually Drives Production Spend* (2026)
