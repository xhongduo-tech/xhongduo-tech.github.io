## 核心结论

推理服务故障恢复的本质，不是等待坏实例自己修好，而是把请求从故障实例上快速移走，再把流量切到健康实例。这里的“实例”可以理解为一份正在提供推理能力的服务副本，比如一个 Pod 或一个进程。

真正决定恢复能力的，不是单一重启命令，而是一整套闭环：健康检查发现故障、路由层摘流、调度系统补副本、模型 warmup 预热、灰度放量和失败回滚。只要其中一环做错，系统就可能表现为“进程活着，但服务不可用”。

一个最小判断标准是：当模型加载失败、GPU 不可用、依赖超时、5xx 激增时，系统应该先阻止故障实例继续接流量，再决定是否重启、替换或回滚版本。恢复关注的是“服务连续性”，不是“单机自愈”。

如果副本之间足够独立，冗余可以显著提高整体可用性。常见近似公式是：

$$
A = 1 - (1-a)^N
$$

其中 $a$ 是单副本可用性，$N$ 是副本数。它表达的是：只要 $N$ 个副本里至少一个能工作，服务就还能对外响应。

---

## 问题定义与边界

推理服务故障恢复，覆盖的是模型、实例、节点、路由和流量层面的恢复，不等于“进程崩了再拉起”。“故障”至少包括下面几类：

| 维度 | 典型故障 | 是否一定需要重启 | 是否必须摘流 |
| --- | --- | --- | --- |
| 进程层 | 进程崩溃、死锁 | 通常需要 | 是 |
| 模型层 | 权重未加载、模型初始化失败 | 不一定 | 是 |
| 资源层 | GPU 不可用、显存耗尽 | 可能需要迁移 | 是 |
| 依赖层 | 特征服务不可达、Redis 超时 | 不一定 | 视策略而定 |
| 路由层 | 5xx 激增、超时飙升 | 不一定 | 常常需要 |
| 版本层 | 新版本逻辑错误 | 回滚优先 | 是 |

这里要区分四个容易混淆的状态：

| 状态 | 含义 | 典型检查方式 |
| --- | --- | --- |
| 进程存活 | 服务进程还在运行 | liveness |
| 模型可用 | 权重加载完成，推理链路能跑通 | 模型自检 |
| 可接流量 | 当前实例适合继续处理请求 | readiness |
| 可恢复 | 系统能自动替换或回滚到健康态 | 编排与发布策略 |

一个常见误区是：进程活着，就算服务健康。这个判断在推理系统里经常是错的。比如某个 Pod 已经启动，但模型权重还在下载，或者 GPU Runtime 没初始化完成，这时 `liveness` 可以通过，但 `readiness` 必须失败。否则流量会被打到一个实际上不能服务的实例上。

所以，故障恢复讨论的是“服务是否还能稳定接流量”，而不是“容器是不是还没退出”。边界也要说清楚：如果整片可用区断电、上游鉴权服务整体宕机、或者错误版本已经全量扩散，那么单实例恢复机制本身并不能解决问题，它只能处理局部、独立、可隔离的故障。

---

## 核心机制与推导

一条完整的恢复链路，通常可以拆成五步：

`探测失败 -> 摘流 -> 重新调度或重启 -> warmup -> 恢复接流`

“摘流”指把实例从负载均衡或服务发现结果中移除；“warmup”指实例虽然起来了，但还需要先做模型加载、缓存填充、JIT 编译、连接池建立等预热动作，暂时不能接真实请求。

可以把它想成一个玩具例子。

假设你有 3 家外卖店都能做同一道菜，平台会把订单分给其中任意一家。现在其中 1 家突然停电。正确做法不是让平台继续给它派单，然后等它恢复；正确做法是先把这家店从派单列表里移除，再把订单交给另外 2 家。等它恢复供电、备菜完成后，再重新加入派单列表。推理服务的故障恢复，本质上就是这个机制，只是把“店”换成了“推理实例”。

从概率上看，如果第 $i$ 个副本的可用性是 $a_i$，并且服务只要有一个副本可用就算可用，那么整体可用性近似是：

$$
A_{service} = 1 - \prod_i (1-a_i)
$$

如果各副本同质且相互独立，公式可以简化为：

$$
A = 1 - (1-a)^N
$$

例如，3 个副本、每个副本独立可用性 $a=0.99$：

$$
A = 1 - (1-0.99)^3 = 1 - 0.01^3 = 0.999999
$$

也就是 99.9999%。这个数很好看，但它有前提：副本故障近似独立。如果 3 个副本都在同一台机器、同一个 AZ、同一个错误版本上，它们就可能一起坏，公式会严重高估可用性。

所以高可用的关键不是“副本数量多”，而是“副本分散到不同故障域”。“故障域”可以白话理解为会一起出问题的一组资源，比如同一节点、同一机架、同一可用区、同一镜像版本。

恢复链路里还有一个常被忽略的约束：滚动发布或节点排空时，系统必须保留足够多的健康副本。例如总副本数为 $N$，最大允许同时不可用的数量为 $u$，那么至少要保证：

$$
N-u
$$

个副本仍然可服务。否则发布动作本身就会把服务推入不可用状态。

真实工程里，一个更典型的例子是 Kubernetes + KServe/Knative 的在线推理服务。新模型版本发布后，先创建新 revision，只接很小比例的流量；如果新版本出现模型加载失败、5xx 飙升或延迟异常，`readiness` 会让它先退出流量池，灰度控制器再根据指标决定停止放量，甚至自动回滚到旧版。这里恢复依赖的是“可观察 + 可摘流 + 可回滚”，而不是“应用自己保证永不出错”。

---

## 代码实现

代码实现应围绕三个问题来设计：

1. 什么时候允许实例接流量  
2. 什么时候必须强制摘流  
3. 什么时候自动回滚或替换

先看一个最小 Kubernetes 配置片段：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
        - name: server
          image: my-registry/infer:v2
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz/ready
              port: 8080
            periodSeconds: 5
            failureThreshold: 2
          livenessProbe:
            httpGet:
              path: /healthz/live
              port: 8080
            periodSeconds: 10
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /healthz/startup
              port: 8080
            periodSeconds: 5
            failureThreshold: 24
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: inference-service
```

这段配置的职责边界很明确：

| 机制 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| `startupProbe` | 启动期给模型加载留时间 | 运行期摘流 |
| `readinessProbe` | 判断能否接流量 | 判断是否要重启 |
| `livenessProbe` | 判断是否进入不可恢复状态 | 判断模型是否预热完成 |
| `PodDisruptionBudget` | 防止维护操作同时下线太多副本 | 处理运行时业务故障 |

在应用层，`/healthz/ready` 不应该只返回“进程还在”，而应该真正覆盖推理路径上的关键条件，例如：

- 模型权重是否加载完成
- tokenizer 或特征处理器是否就绪
- GPU 是否可见且显存未异常
- 必要下游依赖是否可达
- 当前是否处于排空、回滚或 warmup 阶段

下面给一个可运行的 Python 玩具实现，模拟“健康检查失败后摘流，再等待替换副本接流”的控制逻辑：

```python
from dataclasses import dataclass

@dataclass
class Replica:
    name: str
    alive: bool
    model_loaded: bool
    in_pool: bool = True
    warming_up: bool = False

    def ready(self) -> bool:
        return self.alive and self.model_loaded and not self.warming_up

def reconcile(replicas):
    for r in replicas:
        if not r.ready():
            r.in_pool = False

    healthy = [r for r in replicas if r.ready()]
    if len(healthy) < 2:
        new_replica = Replica(
            name=f"replacement-{len(replicas)}",
            alive=True,
            model_loaded=False,
            in_pool=False,
            warming_up=True,
        )
        replicas.append(new_replica)

    return replicas

def finish_warmup(replica: Replica):
    replica.model_loaded = True
    replica.warming_up = False
    replica.in_pool = replica.ready()

replicas = [
    Replica("r1", alive=True, model_loaded=True),
    Replica("r2", alive=True, model_loaded=False),  # 模型加载失败
    Replica("r3", alive=True, model_loaded=True),
]

replicas = reconcile(replicas)

assert replicas[1].in_pool is False
assert len(replicas) == 4

replacement = replicas[-1]
finish_warmup(replacement)

assert replacement.ready() is True
assert replacement.in_pool is True
assert sum(1 for r in replicas if r.in_pool) >= 3
```

这段代码故意很简单，但表达了恢复控制器的核心原则：不健康实例先摘流，不等它原地恢复后再决定是否继续服务。

如果把它写成伪代码，流程通常是这样：

```text
if readiness_failed:
    remove_from_endpoints(instance)

if unavailable_replicas > threshold:
    create_replacement()

if replacement_started:
    wait_until_warmup_passed()

if error_rate_after_rollout > rollback_threshold:
    rollback_to_previous_revision()
else:
    gradually_increase_traffic()
```

真实工程例子里，KServe 或服务网格还会加上金丝雀发布、异常实例剔除、分区流量控制等机制。它们解决的是同一个问题：不要把恢复动作建立在“坏实例也许马上会好”这个假设上，而要建立在“流量必须优先避开故障态”这个原则上。

---

## 工程权衡与常见坑

故障恢复最常见的失败，不是因为没有探针，而是因为探针语义设计错了。

下面这个表比概念解释更有用：

| 常见坑 | 后果 | 规避方式 |
| --- | --- | --- |
| 把 `liveness` 当 `readiness` | 高负载下反复重启，放大故障 | `liveness` 只管死锁和不可恢复错误 |
| 只测进程，不测模型状态 | 权重未加载完成也接流量，持续报错 | `readiness` 覆盖模型、GPU、依赖状态 |
| 无预算重试 | 重试风暴压垮健康副本 | 指数退避、jitter、重试上限、过载时 fail fast |
| 副本在同一故障域 | 看似有冗余，实际一起挂 | 反亲和、拓扑分散、多 AZ |
| 灰度比例过大 | 坏版本快速扩散到全站 | 小流量 canary + 明确回滚阈值 |
| warmup 不摘流 | 刚启动实例被打满，启动更慢 | 预热完成前 readiness 必须失败 |
| 只监控平均延迟 | 长尾请求积压被忽略 | 监控 P95/P99、错误率、队列长度 |

这里有两个重点。

第一，`liveness` 和 `readiness` 必须分开。前者回答“它死没死”，后者回答“它现在能不能接业务流量”。如果混成一个探针，最典型的结果就是：实例只是暂时慢、依赖偶发抖动、GPU 正在恢复，但系统把它当成“必须重启”，结果造成更多抖动。

第二，恢复系统最怕重试风暴。新手常常觉得“请求失败就多试几次”很安全，实际上在推理场景里，失败往往伴随高算力占用和长尾延迟。盲目重试会让原本还能工作的健康副本也被挤爆。Google SRE 里反复强调的 overload，本质就是这种放大效应。

再举一个真实场景。某个 GPU 推理实例进程仍在运行，但 CUDA 上下文初始化卡住了。如果健康检查只访问 `/ping` 并返回 200，它会一直留在流量池里；结果入口层持续把请求发给它，请求不断超时，客户端再触发重试，最终整个服务的延迟和错误率一起被拉高。这个问题不是“没有重启”，而是“没有及时摘流”。

因此，工程上更稳的做法通常是：

- 先把 `readiness` 设计成严格门禁
- 再把 `liveness` 设计成保守重启
- 对客户端和网关加重试预算
- 对发布系统加自动回滚阈值
- 对部署拓扑加故障域分散

---

## 替代方案与适用边界

故障恢复不是唯一手段，也不是万能手段。它最适合处理独立、局部、可替换的故障，例如单实例异常、单节点损坏、某个新版本不稳定。

如果问题来自共享依赖或区域级故障，单纯重启推理实例通常意义不大。比如特征服务整体不可用，这时继续拉起新副本只会得到一批新的不可用副本，更合理的策略是快速失败、限流、降级或切到备用链路。

下面做一个对比：

| 方案 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| 故障恢复 | 单实例、单节点、单版本故障 | 自动化强，恢复快 | 对共享故障无能为力 |
| 限流降级 | 突发高峰、依赖抖动 | 保护系统不被打穿 | 功能或质量会下降 |
| 多活切流 | AZ/机房级故障 | 抗大故障能力强 | 成本高，数据一致性复杂 |
| 缓存兜底 | 热点请求、弱实时场景 | 成本低、响应快 | 结果可能过期 |
| 人工介入回滚 | 复杂事故、机制无法判定 | 决策灵活 | 恢复速度受人影响 |

适用边界可以总结成一句话：自动恢复适合“坏一部分”，不适合“大家一起坏”。

例如：

- 如果新模型版本有 bug，最有效的是灰度检测后自动回滚。
- 如果单个节点 GPU 损坏，最有效的是摘流后重调度到其他节点。
- 如果上游鉴权服务整体不可用，最有效的往往不是反复重启推理实例，而是直接 fail fast，必要时返回降级结果或静态错误页。
- 如果跨地域都有部署，多活切流可能比单地域内的自愈更关键。

所以，推理服务故障恢复应该被放在更大的高可用体系里看：它是局部修复工具，不是全局抗灾方案。

---

## 参考资料

1. [Kubernetes: Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
2. [Kubernetes: Disruptions / PodDisruptionBudget](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)
3. [Google SRE Book: Handling Overload](https://sre.google/sre-book/handling-overload/)
4. [KServe: Canary Rollout Strategy](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary)
5. [Knative: Gradual rollout of traffic to Revisions](https://knative.dev/docs/serving/rolling-out-latest-revision/)
6. [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)
7. [Envoy: Outlier Detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier)
