## 核心结论

多副本只是起点，不是答案。进程没死不代表模型已加载，跨可用区不代表状态能自动恢复。

推理服务高可用的目标，不是机械地“多起几个 Pod”，而是把不同类型的故障拆到不同故障域里处理，然后在业务允许的恢复时间内完成隔离和切流。这里的“故障域”可以理解为同一类故障会一起失效的边界，比如单个进程、单台机器、单个可用区、单个区域。实例故障、模型加载失败、可用区故障，属于三个不同层级的问题，保护手段也不同。

高可用最少要同时覆盖四件事：

1. 进程活着
2. 模型就绪
3. 流量可接
4. 状态可恢复

只做到第一件事，服务可能“看起来在线”但请求持续报错。只做到前两件事，跨可用区切过去后仍可能因为会话状态、限流计数、模型仓库不可用而无法恢复。真正有工程价值的设计，是把切流时间控制在 `RTO` 内，把数据损失控制在 `RPO` 内，并且明确这些目标值是否值得对应的成本。

可用性通常用近似公式表示：

$$
A \approx \frac{MTBF}{MTBF + MTTR}
$$

这里 `MTBF` 是平均无故障时间，白话讲是“平均多久坏一次”；`MTTR` 是平均修复时间，白话讲是“坏了以后平均多久恢复”。这个公式的含义很直接：系统想更可用，要么少出故障，要么更快恢复，后者通常更可控。

故障切换时间可以拆成：

$$
T_{failover} = T_{detect} + T_{route} + T_{warm} + T_{ready}
$$

也就是“发现故障 + 改路由 + 目标实例预热 + 真正就绪”的总时间。只有当 `T_failover ≤ RTO` 时，方案才算满足目标。

数据丢失窗口也可以近似为：

$$
L_{data} \approx replication\_lag
$$

即复制延迟大致就是最坏情况下的数据损失窗口。如果 `L_data > RPO`，说明你的“高可用”只保住了计算节点，没有保住状态一致性。

下面这张表可以先建立整体视角：

| 故障域 | 典型故障 | 保护目标 | 常见机制 |
|---|---|---|---|
| 进程/容器 | 死锁、崩溃、OOM | 单实例快速替换 | `liveness probe`、自动重启 |
| 实例 | 模型加载失败、本地盘损坏 | 异常实例不接流量 | `startup probe`、`readiness probe` |
| 节点 | 宿主机故障、GPU 故障 | 工作负载迁移 | 多节点部署、反亲和 |
| 可用区 | 网络隔离、机房故障 | 区域内继续服务 | 多 AZ、LB 摘除故障 AZ |
| 区域 | 大范围中断 | 跨区域兜底 | DNS failover、多区域部署 |
| 状态存储 | Redis 主从延迟、DB 故障 | 状态可恢复、数据少丢 | 复制、持久化、降级策略 |

---

## 问题定义与边界

先定义“高可用”到底保护什么，否则架构很容易为冗余而冗余。

对推理服务来说，至少要回答四个问题：

1. 保护的是请求成功率，还是只保护“端口能连通”？
2. 保护的是秒级切流，还是分钟级恢复也能接受？
3. 保护的是服务持续可访问，还是还要求会话状态不丢？
4. 保护的是“能返回结果”，还是还要求结果基于正确模型版本？

这就是 `RTO`、`RPO`、`MTBF`、`MTTR` 这些术语存在的原因。

| 术语 | 定义 | 白话解释 |
|---|---|---|
| `RTO` | Recovery Time Objective | 故障后最晚多久必须恢复 |
| `RPO` | Recovery Point Objective | 最多允许丢失多久的数据 |
| `MTBF` | Mean Time Between Failures | 平均多久出一次故障 |
| `MTTR` | Mean Time To Recovery | 出故障后平均多久修好 |

推理服务高可用不等于训练系统高可用，也不等于整个业务链路高可用。训练平台挂了，不一定影响线上推理；但模型仓库、对象存储、Redis、数据库、限流组件、服务发现、负载均衡，这些虽然不是“推理引擎本身”，却属于推理服务可用性的边界内系统。

一个新手最容易误判的点是：Pod 还在，不代表服务可用。

玩具例子：一个推理容器已经启动，HTTP 端口 `8000` 也监听成功，但模型文件挂载路径是空的。容器进程活着，所以普通 TCP 检查会成功；但真正请求一进来，服务要么返回 500，要么阻塞等待模型。正确做法不是让它继续接单，而是让 `readiness probe` 失败，把它从流量池摘掉。

这说明“存活”和“可用”不是一回事。为了避免混淆，需要把问题分层。

| 故障类型 | 影响层 | 需要的保护机制 | 对应探针或组件 |
|---|---|---|---|
| 进程崩溃 | 单容器 | 自动重启 | `liveness probe` |
| 模型加载慢 | 启动期 | 延迟接流，不误杀 | `startup probe` |
| 模型加载失败 | 单实例 | 拒绝流量、保留排查信息 | `readiness probe` |
| GPU 卡死 | 单节点 | 节点摘除、实例迁移 | K8s 调度、Node health |
| AZ 网络故障 | 可用区 | 区域级切流 | LB、mesh、跨 AZ |
| Redis 延迟过大 | 状态层 | 限制状态依赖、降级 | 复制监控、熔断 |
| 模型仓库不可访问 | 依赖层 | 缓存、预热、版本回滚 | 对象存储、镜像缓存 |

真实工程里，经常有人把“推理高可用”理解成“Deployment 副本数从 1 改成 3”。这只能降低单实例故障的影响，无法解决模型根本没准备好、AZ 同时失联、状态写在本地盘导致重建后全部丢失这类问题。高可用是目标约束，不是副本数量。

---

## 核心机制与推导

可用性不是抽象口号，可以从公式直接推导出工程动作。

先看：

$$
A \approx \frac{MTBF}{MTBF + MTTR}
$$

假设某服务 `MTBF=999h`，`MTTR=1h`，则：

$$
A \approx \frac{999}{999+1} = 0.999 = 99.9\%
$$

这意味着一千小时里平均有一小时不可用。对内部离线系统，也许够用；对实时推理接口，可能不够。这个公式的重点不是算一个漂亮百分比，而是提醒你：在大多数线上系统里，减少 `MTTR` 往往比盲目追求更大的 `MTBF` 更现实。因为硬件总会坏、网络总会抖，但切流链路、预热速度、探针配置、自动化恢复时间是可以工程优化的。

再看切换时间：

$$
T_{failover} = T_{detect} + T_{route} + T_{warm} + T_{ready}
$$

举一个数值例子：

- `T_detect = 10s`
- `T_route = 20s`
- `T_warm = 30s`
- `T_ready = 40s`

那么：

$$
T_{failover} = 10 + 20 + 30 + 40 = 100s
$$

如果业务要求 `RTO = 60s`，这套架构不合格。问题不在“有没有多副本”，而在切换链路太长。可能是健康检查周期太慢，可能是 LB 生效慢，可能是备用实例没有预热好，也可能是模型太大导致 ready 太慢。每一段都要单独测量。

数据层也一样。若复制延迟为 `8s`：

$$
L_{data} \approx 8s
$$

而业务 `RPO = 5s`，则状态方案不合格。你再多起多少推理副本，也保不住会话连续性或限流精度。

为了缩短 `T_failover`，Kubernetes 把健康检查拆成三类探针，这个拆分非常关键：

| 探针 | 保护问题 | 应该回答什么 | 常见误用 |
|---|---|---|---|
| `startup probe` | 启动慢 | “我是否已经完成初始化？” | 不配置，导致慢启动被误杀 |
| `readiness probe` | 是否能接流量 | “我现在是否可安全服务请求？” | 只查端口，不查模型就绪 |
| `liveness probe` | 是否卡死/失活 | “我是不是应该被重启？” | 拿它代替 readiness |

术语首次出现时可以这样理解：

- `startup probe`：启动探针，白话讲是“别在我还没准备好时就判我挂了”。
- `readiness probe`：就绪探针，白话讲是“我现在能不能接新请求”。
- `liveness probe`：存活探针，白话讲是“我是不是已经坏到该重启了”。

这三者不能混用。尤其是推理服务常见的“大模型冷启动”，如果模型加载需要 2 分钟，而你只配置了一个每 10 秒检查一次的 `liveness probe`，容器会在模型刚加载一半时被 Kubernetes 反复杀掉，永远启动不起来。这不是高可用，而是自我打断。

可以把一次故障恢复过程抽象成下面的时序草图：

```text
client
  |
  | request
  v
load balancer ----X----> az-a / instance-1
  |                  detect failure
  |------------------------------+
  |                              |
  |<----- route update ----------+
  |
  +------> az-b / instance-2
             warm model
             ready check pass
             serve traffic
```

也就是：

`detect -> route -> warm -> ready`

只要其中任何一步过长，故障恢复时间就会超标。工程上最常见的误区，是把注意力都放在“实例数量”上，却不测这个时序链路。

---

## 代码实现

最小可行实现的原则是：把“启动慢”“模型未就绪”“进程卡死”拆开处理；把模型文件、会话状态、限流计数外置；让实例尽量无状态。

先看一个可运行的玩具例子，用 Python 模拟切流是否满足 `RTO`，以及复制延迟是否满足 `RPO`：

```python
def availability(mtbf_hours: float, mttr_hours: float) -> float:
    assert mtbf_hours > 0
    assert mttr_hours >= 0
    return mtbf_hours / (mtbf_hours + mttr_hours)

def failover_time(t_detect: int, t_route: int, t_warm: int, t_ready: int) -> int:
    total = t_detect + t_route + t_warm + t_ready
    assert total >= 0
    return total

def meets_targets(rto: int, rpo: int, replication_lag: int, t_detect: int, t_route: int, t_warm: int, t_ready: int):
    tf = failover_time(t_detect, t_route, t_warm, t_ready)
    data_loss = replication_lag
    return {
        "failover_ok": tf <= rto,
        "data_ok": data_loss <= rpo,
        "t_failover": tf,
        "l_data": data_loss,
    }

a = availability(999, 1)
assert round(a, 3) == 0.999

result = meets_targets(
    rto=60,
    rpo=5,
    replication_lag=8,
    t_detect=10,
    t_route=20,
    t_warm=30,
    t_ready=40,
)

assert result["t_failover"] == 100
assert result["l_data"] == 8
assert result["failover_ok"] is False
assert result["data_ok"] is False

print("all assertions passed")
```

这个例子虽然简单，但它体现了一个重要工程原则：高可用目标必须可计算、可验证，而不是写在文档里的口号。

下面看 Kubernetes 的最小配置思路。假设使用 Triton Inference Server 承载模型：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: triton-inference
  template:
    metadata:
      labels:
        app: triton-inference
    spec:
      containers:
        - name: triton
          image: nvcr.io/nvidia/tritonserver:24.08-py3
          args:
            - "tritonserver"
            - "--model-repository=s3://my-model-repo"
            - "--strict-readiness=true"
            - "--exit-on-error=false"
          ports:
            - containerPort: 8000
            - containerPort: 8001
            - containerPort: 8002
          startupProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            periodSeconds: 10
            failureThreshold: 18
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            periodSeconds: 5
            failureThreshold: 2
            timeoutSeconds: 2
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8000
            periodSeconds: 15
            failureThreshold: 3
            timeoutSeconds: 2
          resources:
            limits:
              nvidia.com/gpu: 1
```

这个配置表达的含义是：

- `startupProbe` 等待模型真正加载完成，避免冷启动时被误杀。
- `readinessProbe` 检查 `/v2/health/ready`，只有模型已可服务才接流量。
- `livenessProbe` 检查 `/v2/health/live`，只处理进程失活或死锁问题。

Triton 的 `--strict-readiness=true` 很关键。它的作用可以白话理解为：“不是 HTTP 服务起来就算 ready，而是模型和服务状态都满足条件才算 ready。”这比只测端口连通更接近真实可用性。

真实工程例子：两个可用区分别部署一组 Triton 实例，前面挂区域级负载均衡。模型仓库放对象存储，实例启动时从对象存储拉取模型或挂载缓存卷；会话状态、限流计数、请求幂等键放 Redis 或数据库。某个 AZ 内实例连续 readiness 失败时，LB 将该 AZ 的后端摘除，流量切到另一侧。这样“模型加载失败”只影响单实例，“节点故障”只影响单节点，“AZ 故障”只影响单区，不会一层故障拖垮整组服务。

状态外置的伪代码可以写成这样：

```yaml
inference_service:
  stateless_compute: true
  model_repository: s3://my-model-repo
  session_store: redis://session-cluster:6379
  rate_limit_store: redis://rate-limit-cluster:6379
  request_log_store: postgres://app-db/prod
  local_disk_usage:
    cache_only: true
    no_source_of_truth: true
```

请求路径图可以概括为：

```text
Client
  -> Global DNS / Region Router
  -> Regional Load Balancer
  -> AZ-A or AZ-B Triton Pod
  -> Redis (session / counters)
  -> Object Storage (model repository)
  -> DB / Log system
```

这里最重要的设计点不是“用了多少组件”，而是“哪个组件保存真状态”。只要本地盘不是状态真源，实例就可以被随时重建；只要模型仓库不绑死在单机，AZ 级故障就有恢复空间。

---

## 工程权衡与常见坑

高可用不是免费午餐。跨 AZ、热备、双活、复制链路、健康检查、预热实例，都会增加成本、复杂度和新的故障面。很多系统不是“没有冗余”出问题，而是“冗余层级太多却没人验证切换路径”出问题。

先看常见坑：

| 坑位 | 后果 | 规避方式 |
|---|---|---|
| 把 `liveness` 当 `readiness` | 未就绪实例提前接流量或被反复重启 | 三类探针职责分离 |
| 只看端口，不看模型状态 | 模型没加载也继续接单 | readiness 直接检查模型就绪 |
| 把慢启动当故障 | 大模型永远起不来 | 配 `startup probe`，不要硬拉长 liveness |
| 状态写在本地盘 | 实例重建后状态丢失 | 会话、计数、元数据全部外置 |
| 忽略复制延迟 | 切流后状态回退 | 用监控衡量 `replication_lag` 并对齐 `RPO` |
| 误把 DNS 当毫秒级切流 | 故障摘除慢，客户端缓存不一致 | 细粒度切流交给 LB 或 service mesh |
| 没有先定 `RTO/RPO` | 冗余堆叠，成本失控 | 先定目标，再选架构 |

“模型启动慢但并未故障”是最常见误判之一。比如一个 30GB 模型在 GPU 节点上加载需要 90 秒，如果你没有 `startup probe`，而 `livenessProbe` 在 30 秒内连续失败 3 次就重启，那么系统会不断杀掉一个本来能成功启动的实例。这里的问题不是模型坏了，而是健康检查把正常启动过程误当作故障。

另一个典型坑是过度依赖 DNS。DNS 故障切换适合区域级兜底，但不适合秒级、实例级摘流。因为客户端、递归解析器、浏览器、SDK 都可能缓存 DNS 结果，TTL 也不是绝对生效时间。所以“某个 Pod ready 失败后立刻切掉它”的工作应该交给负载均衡器或 service mesh，而不是指望 DNS 立刻更新。

成本和可用性的权衡可以简单列成表：

| 方案级别 | 成本 | 运维复杂度 | 可恢复能力 | 适合目标 |
|---|---|---|---|---|
| 单实例 | 低 | 低 | 很弱 | 开发、测试 |
| 单 AZ 多副本 | 中 | 低 | 抗实例故障 | 内部低风险服务 |
| 多 AZ 主动-被动 | 中高 | 中 | 抗 AZ 故障 | 中等实时要求 |
| 多 AZ 主动-主动 | 高 | 高 | 抗实例与 AZ 故障，切流快 | 在线推理 API |
| 多区域兜底 | 很高 | 很高 | 抗区域故障 | 核心业务、高 SLA |

排查故障时，可以按这个清单看：

| 排查项 | 要问什么 |
|---|---|
| 探针 | readiness 是否真的代表模型可服务？ |
| 切流链路 | `detect/route/warm/ready` 各自耗时多少？ |
| 状态 | 会话、计数、缓存是否有真源和恢复路径？ |
| 模型仓库 | 新实例能否稳定拉到正确版本模型？ |
| 复制 | `replication_lag` 峰值是否超出 `RPO`？ |
| 回滚 | 错误模型发布后能否快速退回旧版本？ |

如果没有这些观测指标，再复杂的高可用架构也只是想象中的高可用。

---

## 替代方案与适用边界

不是所有推理服务都必须做双活，也不是所有业务都值得多区域。架构要跟 `RTO/RPO`、延迟要求、请求价值、状态强度匹配。

先看方案对比：

| 方案 | 延迟 | 成本 | 复杂度 | 适合场景 | 不适合场景 |
|---|---|---|---|---|---|
| 单 AZ 多副本 | 低 | 低 | 低 | 开发环境、内部工具、容忍短中断 | 核心公网推理 |
| 单 AZ 热备 | 中 | 中 | 中 | 流量不大、希望快速恢复 | 需要抗 AZ 故障 |
| 多 AZ 主动-被动 | 中 | 中高 | 中 | 业务需要分钟内恢复 | 极低 `RTO` 场景 |
| 多 AZ 主动-主动 | 低 | 高 | 高 | 实时聊天、在线搜索、低延迟 API | 成本敏感业务 |
| 多区域主动-被动 | 较高 | 很高 | 高 | 区域灾备、合规要求 | 普通中小流量服务 |
| 异步队列推理 | 高 | 中 | 中 | 批量处理、可重试任务 | 强实时交互 |
| 批处理推理 | 很高 | 低到中 | 低 | 离线评分、日报生成 | 在线秒级响应 |

新手可以先这样理解：

- “实时聊天回复”更适合低延迟多副本和快速切流，因为用户在等结果，`RTO` 常常要压到秒级或十秒级。
- “离线批量评分”更适合异步队列和重试，因为任务可以晚一点完成，不必为秒级故障切换支付双活成本。

也就是说，高可用不是只有一种形式。对于同步在线推理，请求超时本身就是用户可见故障；对于异步任务，短时间节点故障只要任务能重试，业务并不一定受损。

可以用一个简化决策树判断：

1. 先问 `RTO` 是否要求秒级。
2. 再问 `RPO` 是否要求几乎不丢状态。
3. 再问业务是否必须同步返回。
4. 最后才决定是单 AZ、多 AZ、主动-被动、主动-主动，还是异步方案。

更具体一点：

- 如果 `RTO` 是几分钟，`RPO` 宽松，且业务不强依赖会话状态，单 AZ 多副本或单 AZ 热备就可能够用。
- 如果 `RTO` 要求 30 秒内恢复，且不能因 AZ 故障全挂，至少应考虑多 AZ。
- 如果 `RTO` 很短、流量持续、用户强实时，主动-主动通常比主动-被动更稳，但代价更高。
- 如果业务天然可重试，异步队列比在线双活更划算。

适用边界的核心不是“哪种方案最先进”，而是哪种方案最匹配目标。一个没有明确 `RTO/RPO` 的双活系统，常常只是更贵、更难排查；一个目标清楚的单 AZ 热备系统，反而可能更可靠。

---

## 参考资料

本文的 `RTO/RPO` 定义参考 AWS，探针语义参考 Kubernetes，故障切换类型参考 Route 53，Triton readiness 建议参考 NVIDIA 官方部署文档。

1. [AWS Well-Architected: Recovery objectives (RTO and RPO)](https://docs.aws.amazon.com/wellarchitected/2023-04-10/framework/rel_planning_for_recovery_objective_defined_recovery.html)
2. [Kubernetes: Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
3. [Amazon Route 53: DNS failover types](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/dns-failover-types.html)
4. [NVIDIA Triton Inference Server Deployment Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2580/user-guide/docs/customization_guide/deploy.html)
5. [Google SRE Book: Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)
6. [Kubernetes: Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
