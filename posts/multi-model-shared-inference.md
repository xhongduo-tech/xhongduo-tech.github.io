## 核心结论

多模型共享推理，指多个模型不再各自独占一套推理实例，而是共享同一组计算、内存和模型存储资源，由运行时按请求动态决定哪些模型驻留在内存里。这里的“驻留”可以先理解成：模型已经被加载到可直接执行推理的内存或显存中，不需要再从远端拉取和初始化。

它成立的前提，不是“模型越多越好”，而是请求分布存在明显长尾。所谓“长尾”，就是少数模型请求很多，多数模型请求很少。此时让所有模型都常驻，会浪费大量 GPU/CPU 和内存；让热点模型常驻、冷模型按需加载，整体资源利用率会显著提升。

可以把是否适合共享推理压缩成一个判断式：

$$
\text{共享划算} \approx \text{长尾程度高} + \text{冷启动可接受} + \text{工作集可装入内存}
$$

这里“工作集”可以先理解成：一段时间内真正活跃、需要经常被访问的那批模型集合。

玩具例子很简单。假设有 20 个小模型，每个模型单独部署都要占 1GB 内存，但任意时刻真正有流量的只有 3 到 4 个。如果每个模型都独占实例，要准备接近 20GB；如果做共享池，只保留最近活跃的 4 个模型常驻，内存占用可能降到 4GB 到 6GB，代价是少数冷模型第一次请求会慢一些。

真实工程例子通常发生在多租户 SaaS 平台。所谓“多租户”，就是很多客户共用同一套服务，但每个客户有自己的模型版本。比如广告排序、文档分类、风险打分这类场景，平台可能维护几百个租户模型。单个租户请求不高，但整体请求量很大。如果每个租户都独占一个推理端点，实例空转会非常严重；如果进入共享池，热租户模型常驻，冷租户模型按需加载，成本会更接近真实负载。

---

## 问题定义与边界

多模型共享推理不是“多个模型同时跑”这么宽泛的说法。更准确的定义是：多个模型共享同一套 serving 实例、模型仓库和缓存机制，请求先路由到实例，再由实例决定模型是否已驻留、是否需要加载、是否需要驱逐其他模型。

为了讨论清楚，先给出一组最小定义：

| 术语 | 含义 | 白话解释 |
|---|---|---|
| 模型 $m$ | 可被请求执行的推理单元 | 一份可加载、可预测的模型文件 |
| 实例 $i$ | 承载推理进程的运行节点 | 一台机器或一个容器里的服务进程 |
| 驻留集合 $R_i$ | 当前实例中已加载模型的集合 | 现在“放在手边”的模型 |
| 命中率 $h_{i,m}$ | 请求到来时模型已驻留的概率 | 这个模型有多大概率不用重新加载 |
| 冷启动 | 模型未驻留时的加载与初始化过程 | 请求来了才现搬模型进来 |

边界要先讲清，因为这个方案很容易被误用。它适合的是“模型数量多，但同时活跃模型不多”的场景；不适合“几乎所有模型都持续高频访问”的场景。后者看起来也能共享，但共享层只会退化成频繁装载和互相挤压。

一个常见边界是内存预算。假设一个端点托管 100 个模型，其中 10 个热点、90 个长尾。如果这 10 个热点模型已经把显存或内存基本吃满，再继续往池里塞模型，只会让驱逐更频繁，最终 p95、p99 尾延迟变差。这里的“尾延迟”可以先理解成：最慢那一小部分请求的耗时，通常比平均值更能反映系统稳定性。

下面这张表可以作为初筛标准：

| 维度 | 适合场景 | 不适合场景 | 风险信号 |
|---|---|---|---|
| 请求分布 | 明显长尾 | 热点高度集中且很多 | 命中率持续下降 |
| 模型体积 | 小到中等 | 单模型极大 | 一次加载就占掉大半内存 |
| SLA | 允许少量冷启动 | 严格低延迟硬 SLA | p99 对冷启动极敏感 |
| 框架兼容性 | 模型格式较统一 | 框架混杂严重 | 单实例运行时复杂度高 |
| 故障隔离 | 可接受共享故障域 | 强租户隔离要求 | 单实例出问题影响面过大 |

所以，问题的核心不是“能不能把多个模型放一起”，而是“共享之后，是否还能在内存预算、延迟目标和运维复杂度之间保持平衡”。

---

## 核心机制与推导

共享推理的请求路径，通常可以拆成五步：

1. 路由请求到某个实例。
2. 检查目标模型是否已驻留。
3. 命中则直接执行推理。
4. 未命中则下载并加载模型。
5. 如果内存不够，按策略驱逐旧模型，再把新模型放入驻留集合。

先看最基础的内存约束：

$$
\sum_{m \in R_i} S_m + B_i \le M_i
$$

其中，$S_m$ 是模型 $m$ 的内存占用，$B_i$ 是实例自身的基础开销，$M_i$ 是实例总内存预算。这个式子很朴素，但它决定了共享推理的第一性条件：驻留集合不能超过可用容量。否则系统不会“慢一点”，而是直接进入抖动状态。

再看期望延迟：

$$
E[T_{i,m}] = T_{exec}(m) + (1 - h_{i,m}) \cdot (T_{dl}(m) + T_{load}(m))
$$

其中，$T_{exec}(m)$ 是模型已驻留时的执行耗时，$T_{dl}(m)$ 是从模型仓库下载的耗时，$T_{load}(m)$ 是反序列化、初始化和放入运行时的耗时，$h_{i,m}$ 是命中率。

这个公式说明了一件很重要的事：共享推理的收益不是来自执行更快，而是来自高命中率时“避免重复加载”。模型一旦未命中，冷启动成本会直接加到请求上。

用一个玩具例子看得更直观。模型 A 命中率 95%，模型 B 命中率 85%，二者热执行时间都为 60ms，冷启动总成本都是 1.8s，那么：

$$
E[T_A] = 60 + 0.05 \times 1800 = 150\text{ms}
$$

$$
E[T_B] = 60 + 0.15 \times 1800 = 330\text{ms}
$$

只差 10 个百分点的命中率，期望延迟几乎翻倍。这就是为什么热点模型更适合常驻，而冷模型即使能共享，也不一定值得放在同一个池里长期占位。

还可以继续推一层。假设某模型占用空间较大，但命中率很低，它会对其他热点模型造成驱逐压力。那它的“驻留价值”就不高。一个常见近似做法是比较单位内存的收益，例如：

$$
V(m) \propto \frac{\lambda_m \cdot h_m \cdot C_m}{S_m}
$$

这里 $\lambda_m$ 是请求频率，$C_m$ 是一次冷启动的代价。这个式子不是严格最优解，但表达了工程直觉：请求越频繁、冷启动越贵、体积越小的模型，越值得常驻。

真实工程里，流程通常是“路由和驻留状态联动”。也就是说，路由器不能只看负载均衡，还要看哪个实例已经有该模型。如果同一个模型已在实例 3 驻留，就优先把请求打到实例 3，而不是随机打散到所有实例。否则命中率会被路由策略自己破坏。

这就是共享推理真正依赖的三层机制：

| 机制 | 作用 | 如果做错会怎样 |
|---|---|---|
| 路由感知驻留状态 | 提高命中率 | 请求被打散，重复冷启动 |
| 缓存策略控制驻留集合 | 维持工作集稳定 | 热点被频繁驱逐 |
| 模型仓库支持按需拉取 | 降低运维复杂度 | 发布与回滚困难 |

---

## 代码实现

实现上，至少要有四个模块：模型注册表、缓存管理器、加载器、请求路由器。

“模型注册表”可以先理解成一张映射表，记录模型 ID、版本、存储位置、大小和运行时类型；“缓存管理器”负责维护驻留集合；“加载器”负责下载和初始化模型；“请求路由器”决定请求先发到哪个实例。

下面是一个可运行的 Python 玩具实现。它没有真的执行深度学习推理，但完整表达了“命中、加载、驱逐、统计”的最小闭环。

```python
from collections import OrderedDict

class MultiModelCache:
    def __init__(self, capacity_mb, registry):
        self.capacity_mb = capacity_mb
        self.registry = registry
        self.resident = OrderedDict()
        self.used_mb = 0
        self.load_count = 0
        self.evict_count = 0
        self.hit_count = 0
        self.request_count = 0

    def has(self, model_id):
        return model_id in self.resident

    def touch(self, model_id):
        self.resident.move_to_end(model_id)

    def ensure_capacity(self, model_size_mb):
        while self.used_mb + model_size_mb > self.capacity_mb and self.resident:
            evicted_id, evicted_size = self.resident.popitem(last=False)
            self.used_mb -= evicted_size
            self.evict_count += 1

    def load(self, model_id):
        model_size = self.registry[model_id]["size_mb"]
        self.ensure_capacity(model_size)
        self.resident[model_id] = model_size
        self.used_mb += model_size
        self.load_count += 1

    def infer(self, model_id, x):
        self.request_count += 1

        if self.has(model_id):
            self.hit_count += 1
            self.touch(model_id)
        else:
            self.load(model_id)

        bias = self.registry[model_id]["bias"]
        return x + bias

    @property
    def hit_rate(self):
        if self.request_count == 0:
            return 0.0
        return self.hit_count / self.request_count


registry = {
    "A": {"size_mb": 200, "bias": 1},
    "B": {"size_mb": 300, "bias": 10},
    "C": {"size_mb": 400, "bias": 100},
}

cache = MultiModelCache(capacity_mb=700, registry=registry)

assert cache.infer("A", 1) == 2
assert cache.infer("A", 5) == 6
assert cache.infer("B", 2) == 12
assert "A" in cache.resident and "B" in cache.resident

# 加载 C 时容量不足，会驱逐最久未使用的模型
assert cache.infer("C", 0) == 100
assert cache.used_mb <= 700
assert cache.load_count == 3
assert cache.evict_count >= 1
assert round(cache.hit_rate, 2) == 0.25
```

这个例子里用了 LRU。所谓 LRU，就是“最近最少使用”策略，白话说就是优先驱逐最久没被访问的模型。它实现简单，适合热度相对稳定的场景；如果流量波动强、模型冷启动特别贵，单纯 LRU 往往不够，需要引入预热、分层缓存或带权重的驱逐策略。

真实工程里，请求处理逻辑通常类似下面这样：

```text
handle_request(model_id, input):
    instance = router.pick_instance(model_id)

    if instance.cache.has(model_id):
        metrics.hit(model_id, instance)
        return instance.infer(model_id, input)

    if not registry.exists(model_id):
        return error("model not found")

    artifact = repository.fetch(model_id)
    instance.cache.ensure_capacity(artifact.size)
    instance.loader.load(model_id, artifact)
    metrics.load(model_id, instance)

    return instance.infer(model_id, input)
```

真实工程例子可以看多租户文本分类平台。比如有 300 个租户，每个租户一个 LoRA 或轻量模型，日常同时活跃的只有 20 到 30 个。系统可以把“热租户”预热到共享池，把“冷租户”放在对象存储里，首次请求时再拉取。这样做的关键不是“模型能被加载”，而是要同时埋点这些指标：

| 指标 | 作用 |
|---|---|
| `hit_rate` | 观察驻留策略是否有效 |
| `load_latency` | 判断冷启动是否可接受 |
| `eviction_count` | 发现缓存抖动 |
| `resident_set_size` | 监控实际工作集大小 |
| `p95/p99` | 识别共享是否伤害尾延迟 |

---

## 工程权衡与常见坑

共享推理最大的收益是提高整体资源利用率，最大的风险则是尾延迟恶化。它不是为了让所有模型都更快，而是为了让系统整体更稳、更省。

最常见的坑，不在“推理代码写错”，而在“分组策略错了”。下面这张表比抽象讨论更有用：

| 问题 | 表现 | 根因 | 规避 |
|---|---|---|---|
| 工作集超过内存 | p99 急剧上升 | 热点模型互相驱逐 | 按热度拆池，限制共享范围 |
| 只看平均延迟 | 均值正常但用户投诉 | 冷启动集中打在尾部 | 强制监控 p95/p99 |
| 路由不看驻留状态 | 同模型多实例重复加载 | 负载均衡与缓存割裂 | 做驻留感知路由 |
| 模型尺寸差异过大 | 大模型挤掉多个小模型 | 单次驱逐成本不对称 | 按大小分组部署 |
| 框架差异严重 | 运行时复杂、故障多 | 同池混入多种执行栈 | 按框架或后端隔离 |
| 盲目关闭缓存 | 每次请求都冷加载 | 误以为这样更公平 | 保留缓存并做容量治理 |

一个高频失败模式是“冷模型突然变热”。例如某租户平时每天几十次请求，突然因为运营活动变成每分钟几百次。系统如果还把它当冷模型处理，就会不停加载它，同时驱逐原来的热点模型。结果不是这个租户慢，而是整个池子一起抖：A 模型来了把 B 挤走，B 请求来了又把 C 挤走，所有请求都在反复冷启动。

另一个坑是把共享范围拉得过大。表面看，池子越大，模型越多，资源越“统一”；实际上，统一过度会放大干扰。共享池本质上是一种“有边界的复用”，不是无限合并。尤其在强隔离场景，某个租户的流量异常不能轻易拖垮其他租户，此时就要接受更低资源利用率，换取更强故障隔离。

还有一个容易被忽略的问题是发布与回滚。共享推理通常依赖统一的模型仓库和加载协议。如果模型版本管理混乱，加载失败、权重不兼容、元数据错误都会在运行时暴露，而不是在部署时暴露。结果是系统看起来“部署成功”，实际第一批请求才报错。这类问题必须靠模型注册校验、版本冻结和灰度发布来挡住。

---

## 替代方案与适用边界

共享推理不是默认最优解，它只是多模型场景中的一种折中。真正的选型标准，不是“技术先进性”，而是访问分布、SLA、模型大小、运维复杂度和隔离要求。

下面给出一个直接的对比：

| 方案 | 核心思路 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| 专用端点 | 每个模型独占实例 | 延迟稳定，隔离最好 | 成本高，空转多 | 长期热点、严格 SLA |
| 多模型共享推理 | 多模型共享实例与缓存 | 利用率高，适合长尾 | 冷启动与缓存抖动 | 多租户、长尾明显 |
| ModelMesh 类统一管理 | 统一控制平面管理大量模型 | 模型数可扩展、调度更系统化 | 系统复杂度更高 | 大规模模型托管平台 |
| Triton 模型仓库管理 | 以模型仓库为中心管理加载 | 与推理后端结合紧 | 仍需自己处理共享策略 | 后端统一、工程能力较强 |

什么时候不该继续用共享池，而该拆出去做专用端点？判断标准通常很朴素：

1. 某几个模型已经长期高频，且彼此加起来就能吃满大部分缓存。
2. 这些模型的业务 SLA 很严，不能接受偶发冷启动。
3. 它们的收益已经不在“省机器”，而在“稳住尾延迟”。

这时继续共享，只会让热点模型互相争缓存，收益开始递减，风险开始增加。

相反，什么时候共享最划算？

1. 请求明显长尾。
2. 模型体积小到中等。
3. 可以接受少量首次请求变慢。
4. 模型格式和运行时相对统一。
5. 系统有能力做命中率、驱逐和加载耗时监控。

所以，多模型共享推理的适用边界可以浓缩成一句话：它适合“多而不同时热”的模型集合，不适合“少但一直热”的模型集合。前者主要矛盾是浪费，后者主要矛盾是稳定性；用同一种方案解决这两类问题，通常会失真。

---

## 参考资料

1. [Amazon SageMaker AI Multi-model endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)
2. [Set SageMaker AI multi-model endpoint model caching behavior](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-caching.html)
3. [KServe ModelMesh Overview](https://kserve.github.io/archive/0.14/modelserving/mms/modelmesh/overview/)
4. [Triton Model Repository](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1140/user-guide/docs/model_repository.html)
