## 核心结论

推理服务灰度发布的本质，不是“把新模型部署上去”，而是“在稳定版继续兜底时，用一部分真实线上请求验证新模型能否持续满足生产要求”。这里的“兜底”可以理解为旧版本继续承担大多数流量，因此一旦新版本出现异常，系统还能快速退回安全状态。

灰度发布判断的重点也不是离线评测分数。离线分数是实验室指标，只能说明模型在静态数据集上的表现；真正决定能否上线的，是在线指标是否达标，比如延迟、超时率、5xx、点击率、转化率、资源消耗是否在可接受范围内。常见判定方式可以写成：

$$
n_c = pN,\quad n_s = (1-p)N
$$

其中，$N$ 是总请求量，$p$ 是灰度比例，$n_c$ 是灰度版请求数，$n_s$ 是稳定版请求数。只有在样本足够时，比较才有意义，因此通常先要求：

$$
n_c \ge n_{min}
$$

再判断灰度版指标 $m_c$ 是否满足：

$$
m_c \le T \quad \text{或} \quad m_c - m_s \le \epsilon
$$

其中，$T$ 是硬门限，$\epsilon$ 是相对稳定版允许的退化幅度，$m_s$ 是稳定版指标。

玩具例子很直接。假设每分钟有 10,000 个请求，先让 10% 流量走新模型，那么新模型每分钟只接 1,000 个请求，其余 9,000 个继续走旧模型。如果新模型的超时率、5xx 或业务指标明显变差，就应立刻停止放量并回滚，而不是继续赌后续会变好。

旧版、灰度版、镜像流量三者容易混淆，职责差异如下：

| 方案 | 是否真正返回给用户 | 主要职责 | 适合解决的问题 |
|---|---|---|---|
| 旧版 stable | 是 | 线上稳定兜底 | 保证当前服务可用 |
| 灰度版 canary | 是，但只服务部分流量 | 用真实流量验证新版 | 判断是否继续放量 |
| 镜像流量 shadow | 否 | 旁路计算与观测 | 先验证计算稳定性 |

---

## 问题定义与边界

灰度发布要解决的问题，是“新模型在真实生产环境中是否稳定、是否比旧版更好、是否值得继续放量”。这里的“真实生产环境”很关键，因为线上请求分布、缓存命中、用户行为、长尾输入、资源争用，往往都和离线环境不同。

它的适用前提可以压缩成四条检查清单：

| 条件 | 白话解释 | 为什么重要 |
|---|---|---|
| 可分流 | 请求能按比例、用户、Header 或规则切开 | 没法切流就谈不上灰度 |
| 可并行运行 | 新旧版本能同时存在 | 灰度要求双版本共存 |
| 可独立观测 | 能分别看到 stable 和 canary 的指标 | 否则不知道谁出了问题 |
| 可快速回滚 | 出问题能立刻切回旧版 | 降低事故持续时间 |

如果一个推荐模型只负责“给候选内容排序”，通常适合灰度，因为它本身不直接产生不可逆副作用。相反，如果模型的输出会直接触发支付、发券、扣费、风控封禁、不可撤销写库，这类强副作用场景就不能简单做流量替换，因为一次错误请求带来的损失不能靠回滚消除。

适合与不适合灰度的边界可以这样看：

| 适合灰度 | 不适合灰度 |
|---|---|
| 排序模型、召回模型、打分模型 | 支付决策、不可逆写入、强事务链路 |
| 读多写少、可重试、弱状态服务 | 强状态依赖、会话黏连强、单副本本地状态重 |
| 可以对 stable/canary 单独打指标 | 无法拆分监控口径 |
| 回滚只需切流量 | 回滚还要回补业务状态 |

指标类型至少要覆盖四类，否则判断会失真：

| 指标类型 | 典型指标 | 作用 |
|---|---|---|
| 延迟 | p95、p99、超时率 | 看用户是否明显变慢 |
| 错误率 | 5xx、模型推理失败率 | 看服务是否稳定 |
| 业务指标 | CTR、CVR、GMV、投诉率 | 看结果是否有业务价值 |
| 资源消耗 | CPU、GPU、内存、显存、缓存命中率 | 看是否能长期承载 |

真实工程里，很多上线失败不是“模型不准”，而是“模型算得出来，但代价过高”。这也是灰度发布和纯离线评测的根本区别。

---

## 核心机制与推导

灰度的基本机制就是流量切分。总流量 $N$ 中，比例 $p$ 给新版，比例 $1-p$ 给稳定版，因此：

$$
n_c = pN,\quad n_s=(1-p)N
$$

这两个量不是装饰公式，而是决定观测是否有统计意义。比如每分钟 10,000 请求，灰度 5%，那么新模型每分钟只处理 500 个请求。如果你只观察 1 分钟，且业务本身波动很大，这 500 个样本可能根本不够。于是实际系统会先看样本量是否满足 $n_c \ge n_{min}$，再进入指标判断。

一个标准判定流程通常是：

1. 放量到某个比例。
2. 暂停一段观察窗。
3. 检查样本量是否足够。
4. 检查新版是否超过硬门限 $T$。
5. 再检查相对 stable 的退化是否超过 $\epsilon$。
6. 决定继续放量、暂停排查或回滚。

可以用简化规则表示：

- 若 $n_c < n_{min}$，继续观察，不做结论。
- 若 $m_c > T$，立即回滚。
- 若 $m_c - m_s > \epsilon$，暂停或回滚。
- 否则继续放量。

玩具例子：每分钟 10,000 请求，灰度 5%，则 $n_c=500$。如果旧版 5xx 为 0.2%，新版升到 1.0%，而门限 $T=0.5\%$，则新版已超过硬门限，应立即回滚。这时候不能说“比例还低，再观察一下”，因为硬门限的意义就是发现不可接受风险时立刻终止。

镜像流量是另一个常见机制。它和灰度不同：用户仍收到 stable 的结果，新版只旁路执行计算，不参与响应返回。它适合先验证“新模型是否跑得通、资源是否爆、结果分布是否异常”，但不适合直接验证用户真实交互效果，因为用户并没有真正看到新版结果。

下表给出常见指标及其关注原因：

| 指标类型 | 为什么要看 | 典型门限示例 |
|---|---|---|
| p95/p99 延迟 | 平均值会掩盖长尾慢请求 | p99 不高于 stable 10% |
| 5xx 错误率 | 直接反映可用性 | 不超过 0.5% |
| 超时率 | 推理常见失败形式 | 不超过 stable 0.2 个百分点 |
| CTR/CVR | 模型上线最终目标常是业务改善 | 不低于 stable 或仅允许极小退化 |
| CPU/GPU 利用率 | 新版可能更耗资源 | 单实例不持续逼近上限 |

如果要画示意图，可以理解为两张图：一张是“入口流量按比例分到 stable 与 canary”，另一张是“放量 -> 观测 -> 达标继续 / 不达标暂停或回滚”。这两张图合起来，就是灰度发布的最小闭环。

---

## 代码实现

工程上常见组合是 Kubernetes + Argo Rollouts + Istio + Prometheus。Argo Rollouts 负责发布编排，Istio 负责流量控制，Prometheus 负责指标观测。

下面是一个最小的 Rollout 片段，表示先把 canary 权重设为 10%，暂停 10 分钟，再放到 50%，再暂停：

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: recommender-rollout
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: { duration: 10m }
        - setWeight: 50
        - pause: { duration: 10m }
```

如果只想让内部用户先命中新版，可以配 Header 路由。Header 可以理解为请求头中的标记字段，网关可据此决定走哪条流量路径。

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: recommender-vs
spec:
  hosts:
    - recommender.default.svc.cluster.local
  http:
    - match:
        - headers:
            x-canary-user:
              exact: "true"
      route:
        - destination:
            host: recommender-canary
          weight: 100
    - route:
        - destination:
            host: recommender-stable
          weight: 90
        - destination:
            host: recommender-canary
          weight: 10
```

判定逻辑不要写死在人的脑子里，而应尽量自动化。下面是一个可运行的玩具版 Python 代码，用于模拟“样本足够后，根据门限决定继续、暂停或回滚”：

```python
def decide_rollout(N, p, n_min, m_c, m_s, T, eps):
    n_c = p * N
    n_s = (1 - p) * N

    if n_c < n_min:
        return "observe"
    if m_c > T:
        return "rollback"
    if (m_c - m_s) > eps:
        return "pause_or_rollback"
    return "increase"

assert decide_rollout(10000, 0.05, 300, 0.010, 0.002, 0.005, 0.001) == "rollback"
assert decide_rollout(10000, 0.05, 600, 0.003, 0.002, 0.005, 0.002) == "observe"
assert decide_rollout(10000, 0.10, 300, 0.003, 0.002, 0.005, 0.002) == "increase"
```

真实工程例子：一个推荐模型在 Kubernetes 上升级，先 `setWeight: 10`，暂停 10 分钟，同时用 Prometheus 看 p95、p99、5xx、超时率、CTR。若 p99 翻倍，或 5xx 超门限，立即触发回滚。若只是内部验证，则先通过 `x-canary-user: true` 让员工流量命中新版，其余流量继续走 stable。

常见配置项和风险如下：

| 配置项 | 含义 | 常见风险 |
|---|---|---|
| `setWeight` | 设置 canary 流量比例 | 比例放大太快，来不及观测 |
| `pause` | 停止自动推进，留出观察窗 | 暂停太短，样本不足 |
| `managedRoutes` | 让 Rollouts 管理特定流量路由 | 回滚时路由被清理，人工规则丢失 |
| `setCanaryScale` | 单独控制 canary 副本数 | 与 `setWeight` 混用易失衡 |

---

## 工程权衡与常见坑

灰度发布最大的误区，是把它理解成“只要比例低，风险就低”。这不成立。风险不仅取决于流量占比，还取决于负载分布、缓存命中、单实例容量、GPU 显存、模型冷启动以及调用链依赖。

第一个常见坑是只看平均值。平均延迟接近 stable，不代表没有问题。比如新版平均延迟只高 3%，但 p99 翻倍，说明少数请求已经明显慢到影响用户体验，这种情况下不能继续放量。

第二个坑是样本量太小。若灰度比例很低，观测窗又短，结论会非常不稳定。实践里通常要设最小样本量门槛和最小观察窗，例如“每次放量后至少观测 10 分钟，且 canary 请求数至少达到 $n_{min}$”。

第三个坑是把流量占比误当资源占比。10% canary 不一定只带来 10% 资源压力。新模型可能更吃 GPU、缓存未预热、batch 策略不同，结果是 10% 流量压出了 30% 的资源消耗。

第四个坑是 `setCanaryScale` 与 `setWeight` 混用。前者控制副本数，后者控制流量比例。如果只给 canary 少量 pod，却导入较高流量，就会出现“90% 流量压在 10% pod 能力上”的失衡。

| 常见坑 | 后果 | 规避方法 |
|---|---|---|
| 只看均值 | 漏掉长尾退化 | 同时看 p95/p99 和分桶指标 |
| 样本量太小 | 误判继续放量 | 设 $n_{min}$ 和最短观察窗 |
| `setCanaryScale` 与 `setWeight` 混用 | 单副本过载 | 明确容量模型，先测单 pod 极限 |
| `managedRoutes` 回滚时被清理 | 手工流量规则丢失 | 将临时路由与托管路由分开 |
| 强状态依赖 | stable/canary 观测失真 | 避免对强状态链路直接灰度 |

监控规则也应提前准备，而不是发布时临时看图。比如可以约定：连续 5 分钟内 p99 超 stable 20%，或 5xx 超过 0.5%，则直接告警并阻断后续放量。

---

## 替代方案与适用边界

灰度发布不是唯一方案。选型标准通常是风险、成本、切换速度、是否需要真实写请求、是否允许双版本并存。

| 方案 | 是否影响用户 | 是否依赖真实请求 | 回滚速度 | 典型场景 |
|---|---|---|---|---|
| 镜像流量 | 否 | 是 | 快 | 先验证计算正确性与资源稳定性 |
| 灰度发布 | 是，影响部分用户 | 是 | 快 | 验证真实在线稳定性与业务效果 |
| 蓝绿发布 | 是，但切换通常一次完成 | 是 | 很快 | 追求整体验证后快速切换 |
| 分阶段灰度 | 是，逐步扩大 | 是 | 快 | 高风险模型升级 |
| 离线回放 | 否 | 否，使用历史请求 | 很快 | 先做预检与回归分析 |

可以用简化规则快速判断：

- 只验证结果正确性，先用镜像流量。
- 验证线上稳定性与真实业务效果，用灰度发布。
- 追求切换极快，且可提前准备完整新环境，用蓝绿发布。
- 风险高、需要逐步控制事故半径，用分阶段灰度。

灰度发布的适用边界也要说清楚。它适合“双版本能同时存在、请求可分流、指标可独立观测、回滚代价低”的系统；不适合“一次请求不可回头、强事务、强副作用、版本状态强耦合”的系统。超过这个边界，继续套灰度流程，通常只会制造虚假的安全感。

---

## 参考资料

下表按“平台实现”和“方法论”组织，用于支撑不同章节内容：

| 来源 | 覆盖内容 | 适合引用的章节 | 备注 |
|---|---|---|---|
| KServe Canary Rollout Strategy | KServe 的 canary 机制与限制 | 代码实现、适用边界 | 需注意 serverless 模式限制 |
| Argo Rollouts Canary | 分步放量、暂停、回滚 | 核心机制、代码实现 | 灰度发布主流程最直接 |
| Argo Rollouts Traffic Management | 与 Istio 等流量管理集成 | 代码实现、工程坑 | 适合解释流量切分 |
| Seldon Core 2 Experiment | 实验与模型比较机制 | 替代方案、真实工程例子 | 更偏实验平台设计 |
| Desiderata for next generation of ML model serving | 模型服务设计原则 | 问题边界、工程权衡 | 适合补方法论视角 |

1. [KServe Canary Rollout Strategy](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary)
2. [Argo Rollouts Canary](https://argo-rollouts.readthedocs.io/en/stable/features/canary/)
3. [Argo Rollouts Traffic Management](https://argo-rollouts.readthedocs.io/en/stable/features/traffic-management/)
4. [Seldon Core 2 Experiment](https://docs.seldon.io/projects/seldon-core/en/latest/contents/kubernetes/resources/experiment/index.html)
5. [Desiderata for next generation of ML model serving](https://mlatcl.github.io/publications/desiderata-for-next-generation-of-ml-model-serving.html)
