## 核心结论

多模型推理调度，白话说，就是一套决定“哪些模型现在放在 GPU 里、哪些请求先执行、哪些模型先被挤出去”的运行规则。它的目标不是简单提高 GPU 利用率，而是在固定显存预算下，同时控制尾延迟和吞吐。

先给新手版本结论：如果一台 GPU 要服务 10 个模型，但长期只有 2 个模型最热门，那就应该让这 2 个模型常驻，把冷门模型按需加载。这样做的关键收益，不是平均延迟下降多少，而是避免热门请求反复遭遇冷启动，拖垮 p95 和 p99。

可以先用一个很小的公式抓住本质：

$$
E[T] = Q + T_{exec} + (1-h)\cdot T_{load}
$$

其中：

- $Q$ 是排队时间，意思是请求在真正执行前等了多久。
- $T_{exec}$ 是执行时间，意思是模型已经准备好后，算一次推理要多久。
- $T_{load}$ 是加载时间，意思是模型不在显存里时，把它装进来要多久。
- $h$ 是驻留命中率，意思是请求到来时，目标模型已经在显存里的概率。

这个公式直接说明：多模型调度真正优化的是三件事的组合结果，不是单点最优。

| 项目 | 核心内容 |
| --- | --- |
| 目标 | 在显存预算内同时控制尾延迟、吞吐、稳定性 |
| 主要收益 | 提高热门模型命中率，减少冷启动，降低资源空转 |
| 适用前提 | 模型热度分层明显，冷启动代价高，请求不均匀 |
| 不适用场景 | 所有模型都持续高热，或显存足够让全部模型常驻 |
| 常见误区 | 只看平均延迟、只用 LRU、把所有请求混成一个批队列 |

---

## 问题定义与边界

多模型推理调度讨论的是“共享硬件上的在线推理服务”。在线推理，白话说，就是用户请求来了以后系统立刻给结果；它不是离线跑一整批数据，也不是训练任务排队。

这里至少有四个子问题：

1. 模型驻留 `residency`：哪些模型现在保留在 GPU 显存中。
2. 请求路由 `routing`：新请求应该发到哪张卡、哪个实例。
3. 批处理 `batching`：哪些请求可以凑成一个批次一起算。
4. 模型驱逐 `eviction`：显存不够时，先把谁卸载出去。

这四件事是绑在一起的。只优化其中一项，整体通常不会好。比如你只做动态批处理，但不感知模型是否驻留，那么请求可能先进入批队列，最后才发现模型根本没加载，整个批次一起等冷启动，尾延迟反而更差。

一个典型边界场景是多租户平台。多租户，白话说，就是同一套基础设施同时服务多个团队或多个业务。比如白天电商搜索模型最热，晚上客服总结模型最热；模型集合变了，但 GPU 总数没变。调度问题就在于：如何让有限显存随着热度变化平稳切换，而不是每次流量变化都触发大规模抖动。

下面这张表先把术语定住：

| 术语 | 定义 |
| --- | --- |
| 模型 | 一个可独立提供推理能力的权重与计算图 |
| 实例 | 模型在某张 GPU 上的一个可执行副本 |
| 驻留集合 | 当前已经加载在 GPU 显存中的模型集合 |
| 内存预算 | 单卡或单节点可分配给模型的显存上限 |
| 命中率 | 请求到来时目标模型已驻留的比例 |
| 冷加载时间 | 模型从未驻留到可执行的准备时间 |
| 排队时间 | 请求因资源忙碌而等待的时间 |

本题不讨论的内容也要明确：

- 不讨论训练调度。
- 不讨论离线大批量推理。
- 不讨论单一超大模型的张量并行细节。
- 不讨论精细到算子级别的内核调度。
- 不假设每个模型都能被任意切分到任意显存碎片中。

所以本文的核心，是“多模型共享 GPU 的在线服务调度”。

---

## 核心机制与推导

第一层约束是容量。假设 GPU 显存预算为 $B$，模型 $m_i$ 的驻留开销为 $w_i$，那么任意时刻都必须满足：

$$
\sum_{m_i \in R} w_i \le B
$$

其中 $R$ 是当前驻留集合。这个式子看起来简单，但已经说明一个关键事实：驻留是排他资源，模型 A 常驻，等价于挤压模型 B 的常驻机会。

第二层是延迟。对单个请求而言：

$$
T = Q + T_{exec} + I_{miss}\cdot T_{load}
$$

其中 $I_{miss}$ 是一个指示变量，表示是否未命中驻留。取期望后就得到前面的：

$$
E[T] = Q + T_{exec} + (1-h)\cdot T_{load}
$$

这一步的意义是：热度高不等于必须常驻。真正值得常驻的模型，应该看“常驻带来的延迟收益”是否大于它占用的显存成本。一个直观评分可以写成：

$$
score(m)=\frac{\lambda_m \cdot T_{load,m}}{w_m}
$$

其中：

- $\lambda_m$ 是请求到达率，白话说就是这个模型有多热。
- $T_{load,m}$ 是冷启动代价。
- $w_m$ 是显存占用。

这个评分不是唯一正确答案，但它比“最近访问过就保留”的 LRU 更接近工程目标，因为它同时考虑了热度、冷启动损失和体积。

看一个玩具例子。假设一张 16GB GPU，要服务三个模型：

| 模型 | 占用 | 冷加载时间 | 请求率 |
| --- | --- | --- | --- |
| A | 6GB | 2s | 12 req/s |
| B | 8GB | 5s | 4 req/s |
| C | 10GB | 8s | 1 req/s |

按热度排序，会选 A 和 B，一共 14GB。按 LRU，谁最近访问就留谁，可能会把 C 留住。按上面的收益密度评分：

- $A: 12 \times 2 / 6 = 4$
- $B: 4 \times 5 / 8 = 2.5$
- $C: 1 \times 8 / 10 = 0.8$

所以 A 最值得常驻，B 次之，C 最不值得。这里的结论不是“永远不要加载 C”，而是“不要为了偶发请求让 C 长期挤占显存”。

再看命中率变化对延迟的影响。假设：

- $Q=20ms$
- $T_{exec}=40ms$
- $T_{load}=1200ms$

当命中率 $h=0.95$ 时：

$$
E[T]=20+40+(1-0.95)\cdot 1200=120ms
$$

当命中率降到 $h=0.70$ 时：

$$
E[T]=20+40+0.3\cdot 1200=420ms
$$

命中率只掉了 25 个百分点，平均响应却从 120ms 涨到 420ms。这就是为什么多模型调度里“驻留命中率”是一级指标。

实际系统通常按四层顺序工作：

| 层次 | 作用 | 影响 |
| --- | --- | --- |
| 放置 `placement` | 决定模型放在哪些卡上 | 决定可用容量与跨卡流量 |
| 路由 `routing` | 把请求送到哪个实例 | 决定命中率与排队 |
| 批处理 `batching` | 合并相近请求 | 决定吞吐与额外等待 |
| 驱逐 `eviction` | 显存不够时卸载谁 | 决定冷启动频率 |

真实工程例子：一个平台同时服务摘要、重写、分类、检索重排四类模型。白天分类和重排高热，晚上摘要和重写高热。如果系统只按最近访问驱逐，夜间流量切换时会把白天的热点模型和夜间的预热模型来回替换，形成“抖动”。更合理的做法是按时间窗口统计热度，用滞后阈值控制迁移，只在热度变化足够大时重排驻留集合。

---

## 代码实现

实现层面最重要的不是“写一个调度器类”，而是把状态拆清楚。至少要显式区分四段时间：`miss`、`load`、`queue`、`exec`。否则你只会看到总延迟升高，却不知道是冷启动、排队还是执行本身导致的。

下面给一个可运行的简化版 Python 示例。它没有接真实 GPU，但把“命中、驱逐、加载、路由”的核心逻辑保留了。

```python
from dataclasses import dataclass
from typing import Dict, Set, List

@dataclass
class ModelMeta:
    name: str
    mem_gb: int
    load_ms: int
    qps_weight: float
    exec_ms: int

    @property
    def score(self) -> float:
        # 收益密度：热度 * 冷加载代价 / 显存占用
        return self.qps_weight * self.load_ms / self.mem_gb


class MultiModelScheduler:
    def __init__(self, capacity_gb: int, models: List[ModelMeta]):
        self.capacity_gb = capacity_gb
        self.models: Dict[str, ModelMeta] = {m.name: m for m in models}
        self.resident: Set[str] = set()
        self.used_gb = 0

    def _evict_until_fit(self, needed_gb: int) -> List[str]:
        evicted = []
        while self.used_gb + needed_gb > self.capacity_gb:
            victim = min(self.resident, key=lambda name: self.models[name].score)
            self.resident.remove(victim)
            self.used_gb -= self.models[victim].mem_gb
            evicted.append(victim)
        return evicted

    def ensure_loaded(self, name: str):
        model = self.models[name]
        if name in self.resident:
            return {"hit": True, "evicted": [], "load_ms": 0}
        evicted = self._evict_until_fit(model.mem_gb)
        self.resident.add(name)
        self.used_gb += model.mem_gb
        return {"hit": False, "evicted": evicted, "load_ms": model.load_ms}

    def handle_request(self, name: str, queue_ms: int = 0):
        model = self.models[name]
        load_info = self.ensure_loaded(name)
        total_ms = queue_ms + model.exec_ms + load_info["load_ms"]
        return {
            "model": name,
            "hit": load_info["hit"],
            "evicted": load_info["evicted"],
            "queue_ms": queue_ms,
            "exec_ms": model.exec_ms,
            "load_ms": load_info["load_ms"],
            "total_ms": total_ms,
            "resident": sorted(self.resident),
        }


models = [
    ModelMeta("A", mem_gb=6, load_ms=2000, qps_weight=12, exec_ms=40),
    ModelMeta("B", mem_gb=8, load_ms=5000, qps_weight=4, exec_ms=60),
    ModelMeta("C", mem_gb=10, load_ms=8000, qps_weight=1, exec_ms=70),
]

sched = MultiModelScheduler(capacity_gb=16, models=models)

r1 = sched.handle_request("A")
r2 = sched.handle_request("B")
r3 = sched.handle_request("C")  # 需要驱逐
r4 = sched.handle_request("A")  # A 可能被重新加载

assert r1["hit"] is False
assert "A" in r1["resident"]
assert r2["hit"] is False
assert sched.used_gb <= 16
assert r3["load_ms"] == 8000
assert r4["total_ms"] >= 40  # 至少包含执行时间
```

这段代码表达了三个关键动作：

- `residency-aware routing`：先判断模型是否已驻留，再决定是否直接执行。
- `load/unload`：不够放时先驱逐，再加载。
- `dynamic batching`：真实系统里应在 `ready` 状态后进入每模型或每类模型自己的批队列，而不是全局混排。

一个更接近线上系统的简化状态机可以写成：

`cold -> loading -> ready -> evicting -> cold`

典型流程是：

1. 请求到达，查模型状态。
2. 若 `ready`，直接进入该模型批队列。
3. 若 `cold`，进入加载协调器，避免重复并发加载同一模型。
4. 若显存不足，按驱逐评分选择牺牲者。
5. 模型进入 `ready` 后再放行队列。
6. 记录 `miss/load/queue/exec` 四类指标。

这里有一个非常容易忽略的点：动态批处理应该和模型驻留感知结合。因为批处理的目标是提高吞吐，而驻留感知的目标是避免冷启动。如果你为了攒更大的批，把未驻留模型的请求也长时间堆在队列里，系统会同时损失吞吐和尾延迟。

---

## 工程权衡与常见坑

多模型推理调度的工程本质是三组冲突：

- 命中率 vs 显存占用
- 吞吐 vs 尾延迟
- 常驻模型数 vs 冷启动频率

显存里常驻的模型越多，命中率越高，但单个模型可分配到的实例数可能减少；批次凑得越大，吞吐越高，但请求为等批次会多排队；驱逐越激进，显存利用率越高，但冷启动会更频繁。

一个真实常见故障是：冷门大模型偶尔被访问一次，系统按 LRU 认为“最近访问过，应保留”，结果把高热小模型挤出显存。接下来几秒内，热门请求连续 miss，p99 从几十毫秒直接跳到秒级。这类事故通常不是算力不够，而是驱逐策略错了。

下面这张“坑与规避”表比抽象原则更有用：

| 常见坑 | 直接后果 | 规避方式 |
| --- | --- | --- |
| 只看平均延迟 | p95/p99 爆炸但报表还好看 | 以尾延迟和 miss rate 为主指标 |
| 路由不看驻留状态 | 请求被发到未加载实例，隐性排队 | 路由阶段先查驻留，再选实例 |
| 单纯 LRU | 热点模型被偶发大模型挤出 | 用热度+体积+冷启动代价联合评分 |
| 共享批队列 | 长请求拖慢短请求，尾延迟恶化 | 按模型或服务等级拆分批队列 |
| 直接覆盖线上版本 | 新策略抖动时全量受影响 | 灰度发布，可回退，可双写指标 |

再补两个工程细节。

第一，冷启动不只是一段“加载权重”的时间。它常常包含权重映射、显存分配、CUDA 图初始化、KV cache 预留、编译或 warmup。也就是说，`T_load` 往往比很多人想象的大，而且波动不小。

第二，观测粒度一定要到模型级。只看整机 GPU 利用率很容易误判。GPU 90% 利用率不代表系统健康，可能只是某一个大模型把卡占满，别的模型一直 miss。

---

## 替代方案与适用边界

多模型推理调度不是默认正确答案。很多场景下，更简单的方案反而更稳。

先看几种替代方案：

| 方案 | 适用场景 | 优点 | 缺点 | 是否需要模型驻留感知 |
| --- | --- | --- | --- | --- |
| 单模型独占部署 | 核心模型极热、SLO 很严 | 稳定、简单、尾延迟好控 | 资源利用率可能偏低 | 否 |
| 静态分片 | 模型集合稳定、热度可预测 | 运维简单 | 负载变化时弹性差 | 弱需要 |
| 固定副本 + 负载均衡 | 少量高热模型 | 容易扩缩容 | 对长尾模型不经济 | 否 |
| 按租户隔离 | 强隔离、计费清晰 | 风险隔离好 | 容易造成碎片浪费 | 弱需要 |
| 多模型动态调度 | 热度变化快、长尾明显 | 利用率高、可承载更多模型 | 系统复杂，调参难 | 是 |

可以把决策思路压缩成三个问题：

1. 是否存在明显长尾模型？
2. 冷启动是否足够贵，贵到会伤害 SLO？
3. GPU 显存是否不足以让热点模型全集长期常驻？

如果三个答案大多是“是”，多模型调度值得做。反过来，如果所有模型都持续高频请求，或者显存本来就够，复杂调度带来的收益会明显下降，甚至不如固定副本稳定。

两个对比例子最容易看清边界。

玩具例子：一个小团队有 2 个高热模型和 20 个低频模型。每天 90% 请求打到前 2 个模型，剩下模型只是偶尔调用。这时“热点常驻 + 冷门按需加载 + 每模型单独批队列”通常是最合适的。

真实工程例子：一个平台给十几个业务线提供统一 LLM 服务，但每个业务线白名单固定、流量全天都高，而且每个模型都能稳定吃满一张卡。这时多模型动态调度就不是最佳方案，固定副本或租户隔离更合适，因为系统主要矛盾已经不是“长尾与冷启动”，而是“稳定吞吐与隔离性”。

所以，是否采用多模型推理调度，不是看它“高级不高级”，而是看你的流量结构是不是在为它买单。

---

## 参考资料

1. [MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving](https://proceedings.mlr.press/v235/duan24a.html) 说明多模型共置、热度感知放置与自适应批调度，本文第 3 节和第 6 节主要借鉴其问题建模。
2. [Serving DNNs like Clockwork: Performance Predictability from the Bottom Up](https://www.microsoft.com/en-us/research/publication/serving-dnns-like-clockwork-performance-predictability-from-the-bottom-up/) 说明为什么在线推理应围绕尾延迟和可预测性设计，本文第 1 节与第 5 节主要依赖这条思路。
3. [NVIDIA Triton Inference Server: Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) 说明动态批处理的配置与约束，本文第 4 节关于 per-model batching 的实现建议主要参考此文档。
4. [NVIDIA Triton Inference Server Architecture / Model Management](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2590/user-guide/docs/index.html) 说明模型仓库、显式加载/卸载和每模型调度器接口，适合把本文的简化逻辑映射到实际产品能力。
5. [vLLM Distributed Inference and Serving](https://docs.vllm.ai/en/v0.10.0/serving/distributed_serving.html) 说明当问题变成单模型过大或多卡并行时，应优先考虑并行部署而不是多模型驻留调度，本文第 6 节用于界定替代方案。
