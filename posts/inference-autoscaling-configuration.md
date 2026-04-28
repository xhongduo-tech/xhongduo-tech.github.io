## 核心结论

推理自动扩缩容的核心不是“资源用满了再加机器”，而是“在服务真正变慢之前，把未来几分钟内会缺的容量先补上”。这里的“推理服务”指在线模型服务，比如 LLM 对话、RAG 检索后生成、图像模型在线推断。它和普通 Web API 的差别在于：新副本不是立刻可用，往往还要经历调度、拉镜像、加载权重、显存初始化、warmup 这些步骤。

因此，扩容主信号通常不应只看 CPU 或平均延迟，而应优先看以下四类量：

| 信号 | 白话解释 | 适合做主触发吗 | 原因 |
| --- | --- | --- | --- |
| 队列长度 $Q$ | 还有多少请求在等处理 | 是 | 直接反映排队压力 |
| 并发会话数 $C$ | 当前同时占用服务能力的活跃请求数 | 是 | 能较早反映容量紧张 |
| GPU 利用率 $U$ | 显卡忙不忙 | 适合做辅助校验 | 能反映算力饱和，但会受 batching 影响 |
| 平均延迟 | 平均请求花了多久 | 不适合单独主触发 | 平均值会掩盖尾延迟问题 |
| CPU 利用率 | 处理器忙不忙 | 通常不适合 | GPU 推理常常不是 CPU 先满 |

最关键的公式是把“扩容提前量”显式算出来：

$$
T_{lead} = T_{sched} + T_{pull} + T_{load} + T_{warm}
$$

其中：
- $T_{sched}$：Kubernetes 把 Pod 调度到节点的时间
- $T_{pull}$：拉取镜像的时间
- $T_{load}$：模型权重加载到内存或显存的时间
- $T_{warm}$：预热时间，比如第一次推理前的初始化

如果单副本单位时间能稳定处理 $\mu$ 个请求，那么队列触发阈值至少应满足：

$$
Q_{trigger} \approx \mu \cdot T_{lead} \cdot s
$$

其中 $s>1$ 是安全系数。白话说：如果你知道新副本两分钟后才能干活，那就必须允许系统在“两分钟的缺口”出现之前就开始扩容。

---

## 问题定义与边界

这篇文章讨论的不是通用的 Kubernetes HPA 入门，而是**在线推理服务的自动扩缩容配置**。目标不是把 GPU 利用率榨到最高，而是**在吞吐接近上限前，维持稳定延迟，特别是避免 P99 尾延迟失控**。P99 的意思是 99% 请求都比这个值更快，只有最慢的 1% 更慢。用户感知通常被这部分慢请求决定。

先看问题边界：

| 维度 | 本文是否重点覆盖 | 说明 |
| --- | --- | --- |
| 在线 LLM 推理 | 是 | 请求到达不均匀，排队明显 |
| GPU 服务 | 是 | 模型加载和显存预热是关键变量 |
| 有 batching 的系统 | 是 | 批处理会改变吞吐与延迟关系 |
| 普通无状态 REST API | 否 | 多数情况下 CPU/QPS HPA 即可 |
| 离线批处理推理 | 否 | 更关注作业完成时间，不是尾延迟 |
| 极低 QPS 服务 | 部分适用 | 复杂控制收益可能不大 |

为什么普通 Web 服务的经验不能直接搬过来？因为两者的“坏掉方式”不同。

玩具例子：一个普通图片上传 API，副本 5 秒就能起来，CPU 高了再扩也还来得及。一个 13B 模型推理服务，副本从创建到可接流量要 120 秒，这时“等 CPU 80% 再扩”通常已经晚了。因为流量峰值可能在 30 秒内打满队列，而新副本还没把权重放进显存。

因此，判断要不要采用这套方法，可以先问四个问题：

| 问题 | 如果回答是“是” | 含义 |
| --- | --- | --- |
| 模型启动是否慢 | 需要更早扩容 | 说明存在明显 $T_{lead}$ |
| 是否有 warmup | 不能把 Pod Ready 等同于可服务 | 说明“启动完成”不等于“容量生效” |
| 是否依赖 GPU | CPU 指标参考价值下降 | 真正瓶颈在显存和算力 |
| 是否存在 batching | GPU 利用率不能单独解释压力 | 批大小会改变利用率表现 |

只要这四项里占了两到三项，就已经不该把 CPU 当主指标。

---

## 核心机制与推导

这套方案本质上是一个多指标闭环控制系统。闭环的意思是：持续观测当前状态，估算需要多少副本，然后把结果反馈给调度系统。

可以按四步理解：

1. 定义观测量：队列长度 $Q$、并发会话数 $C$、GPU 利用率 $U$、预测到达率 $\hat{\lambda}$  
2. 定义单项目标副本数：每个指标单独算出“至少需要几个副本”  
3. 取最紧张约束：哪个指标要求的副本数最大，就按哪个走  
4. 对缩容加稳定窗口：避免刚扩完又缩，来回抖动

统一公式可以写成：

$$
r^* = clamp(r_{min}, r_{max},
\max(
\lceil Q / Q^* \rceil,
\lceil C / C^* \rceil,
\lceil U / U^* \rceil,
\lceil \hat{\lambda}(t + T_{lead}) / \mu \rceil
))
$$

其中：
- $r^*$：目标副本数
- $Q^*$：单副本可接受队列阈值
- $C^*$：单副本可接受并发阈值
- $U^*$：单副本目标 GPU 利用率阈值
- $\mu$：单副本稳定处理能力
- $\hat{\lambda}(t + T_{lead})$：考虑提前量后的短期负载预测

这几个量的直觉解释如下：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $Q$ | 当前队列长度 | 多少请求在排队 |
| $Q^*$ | 单副本队列预算 | 每个副本最多能安全背多少等待请求 |
| $C$ | 当前并发会话数 | 多少活跃请求正在占资源 |
| $C^*$ | 单副本并发预算 | 一个副本适合承载多少并发 |
| $U$ | GPU 利用率 | 显卡当前忙到什么程度 |
| $U^*$ | 目标 GPU 阈值 | 超过这个值就可能快到瓶颈 |
| $\mu$ | 单副本吞吐能力 | 一个副本平均每秒能稳态处理多少请求 |
| $\hat{\lambda}$ | 预测到达率 | 未来短时间内会来多少请求 |

数值玩具例子：

设当前系统配置允许单副本承受：
- $Q^* = 50$
- $C^* = 8$
- $U^* = 70\%$

监控观测到：
- $Q = 120$
- $C = 19$
- $U = 84\%$

那么：
- $\lceil 120 / 50 \rceil = 3$
- $\lceil 19 / 8 \rceil = 3$
- $\lceil 84 / 70 \rceil = 2$

因此目标副本数至少是 3。这里即使 GPU 角度只需要 2 个副本，也不能按 2 扩，因为队列和并发已经显示系统更紧张。

再看“为什么必须提前”的推导。假设当前单副本处理能力是每秒 4 个请求，扩容生效提前量是：

$$
T_{lead} = 20s + 15s + 55s + 30s = 120s
$$

如果突然多出一波持续 2 分钟的流量，那么这 120 秒内，系统只能靠旧副本扛住。若安全系数 $s=1.2$，则：

$$
Q_{trigger} \approx 4 \times 120 \times 1.2 = 576
$$

意思不是“队列到 576 才扩”，而是“如果你允许请求堆到这个量级，说明扩容信号已经晚了”。工程上往往会取更小、更保守的阈值，在趋势刚形成时就扩。

真实工程例子：KServe 部署 vLLM 服务，一个副本启动要 90 到 150 秒，且首次 batch 前有显存预热。业务高峰在整点和半点出现。若仅看平均延迟，监控常常在峰值发生后 20 到 40 秒才出现明显抬升；若看 `vllm:num_requests_waiting`，几秒内就能看到排队信号。此时队列是更早的因，延迟是更晚的果。

---

## 代码实现

落地时通常不自己手写 autoscaler 控制器，而是用 `Prometheus + KEDA + Kubernetes` 这条链路。

组件分工可以简化为：

| 组件 | 作用 | 关键配置 | 常见错误 |
| --- | --- | --- | --- |
| 推理服务 | 暴露队列、并发等指标 | 指标名称稳定、标签一致 | 指标语义不清，running 和 waiting 混淆 |
| Prometheus | 拉取并查询指标 | scrape interval、查询表达式 | 采样周期太长，信号变钝 |
| KEDA | 根据 PromQL 结果算副本数 | threshold、pollingInterval、cooldownPeriod | 阈值设成“漂亮数字”而非容量数字 |
| Kubernetes/HPA | 真正执行扩缩容 | min/max replicas | 和其他 HPA 规则冲突 |

下面是一个典型 KEDA 配置片段，主触发看队列：

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-autoscaler
spec:
  scaleTargetRef:
    name: vllm-service
  pollingInterval: 15
  cooldownPeriod: 360
  minReplicaCount: 1
  maxReplicaCount: 3
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: vllm_queue
        query: sum(vllm:num_requests_waiting)
        threshold: "5"
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: vllm_running
        query: sum(vllm:num_requests_running)
        threshold: "8"
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: gpu_util
        query: avg(DCGM_FI_DEV_GPU_UTIL)
        threshold: "70"
```

这段配置表达的是：每 15 秒看一次指标，至少保留 1 个副本，最多扩到 3 个副本，缩容冷却时间 360 秒。冷却时间就是缩容前要等多久，防止刚过一个小峰值就立刻缩回去。

如果想先在本地验证“多指标取最大值”的逻辑，可以用一个最小可运行的 Python 玩具实现：

```python
from math import ceil

def target_replicas(
    q, c, u, lambda_hat, mu,
    q_star=50, c_star=8, u_star=70,
    r_min=1, r_max=10
):
    candidates = [
        ceil(q / q_star),
        ceil(c / c_star),
        ceil(u / u_star),
        ceil(lambda_hat / mu),
    ]
    return max(r_min, min(r_max, max(candidates)))

# 玩具例子
r = target_replicas(
    q=120,
    c=19,
    u=84,
    lambda_hat=10,
    mu=4
)
assert r == 3

# 队列不高，但预测流量很高，也应该提前扩
r2 = target_replicas(
    q=10,
    c=4,
    u=50,
    lambda_hat=17,
    mu=4
)
assert r2 == 5

print("ok")
```

这个例子体现两个原则：
- 当前压力大时，队列和并发会把副本数推高
- 当前还没排队，但预测未来几分钟会冲高时，也应该提前扩

真实工程例子可以是这样的链路：vLLM 暴露 `num_requests_waiting` 和 `num_requests_running`，`dcgm-exporter` 暴露 `DCGM_FI_DEV_GPU_UTIL`，Prometheus 汇总后交给 KEDA，KEDA 再驱动 KServe 对底层 Deployment 扩缩容。这里最重要的不是 YAML 本身，而是阈值必须从压测反推出来，而不是拍脑袋写一个 `threshold: "5"`。

---

## 工程权衡与常见坑

这类系统最容易出问题的地方，不是“不会配置”，而是“指标和业务含义没对齐”。

常见坑可以直接列出来：

| 常见坑 | 为什么有问题 | 规避方式 |
| --- | --- | --- |
| 只看 CPU | GPU 推理时 CPU 往往不是先满的资源 | 用队列或并发做主触发 |
| 只看平均延迟 | 平均值会掩盖尾部慢请求 | 关注队列、P95、P99 |
| 只看 GPU 利用率 | batching、prefill/decode 阶段会让利用率解释失真 | GPU 只做辅助校验 |
| 不考虑 $T_{lead}$ | 扩容发生时容量缺口已经形成 | 用预热时间前移阈值 |
| 缩容窗口太短 | 流量抖动会造成频繁加载模型 | 缩容慢于扩容 |
| 同时混用 HPA 和 ScaledObject | 两套控制器会互相覆盖目标副本数 | 保持单一扩缩容真源 |

这里有一个很重要的工程原则：**扩容要快，缩容要慢**。因为扩容晚了会直接打到用户体验，缩容慢一点最多多花一些资源费。对于大模型服务，频繁缩容还有额外代价：模型被反复卸载和重新加载，显存与镜像缓存命中率下降，最终看似“节省副本”，实际上抬高了整体成本。

推荐把指标职责分开：

| 指标类型 | 推荐用途 | 不推荐用途 |
| --- | --- | --- |
| 队列长度 | 主扩容信号 | 单独决定缩容 |
| 并发会话数 | 主扩容信号 | 脱离模型类型直接照抄阈值 |
| GPU 利用率 | 校验算力是否接近饱和 | 单独主导扩容 |
| 平均延迟 | 观测服务质量趋势 | 直接触发扩容 |
| P99 延迟 | 告警和回溯分析 | 实时唯一控制信号 |
| 预测负载 | 提前扩容 | 在极低流量下过度复杂化 |

还有一个经常被忽略的点：不同模型的阈值不能共用。7B、13B、70B 模型在 token throughput、显存占用、batch 行为上完全不同。即使框架相同，`Q*=5` 对某个模型是安全值，对另一个模型可能已经太晚。因此阈值必须来自压测结果，至少要压出：
- 单副本稳态吞吐 $\mu$
- 不同 batch 大小时的延迟曲线
- 从创建 Pod 到第一次成功推理的 $T_{lead}$

没有这些基线，所谓“自动扩缩容配置”只是 YAML 形式的猜测。

---

## 替代方案与适用边界

不是所有推理服务都需要多指标闭环控制。系统越简单，方案也应该越简单。选择标准不是“方案是否高级”，而是“收益是否覆盖复杂度”。

常见方案对比如下：

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| CPU HPA | 普通无状态服务、启动很快 | 简单，平台原生支持 | 对 GPU 推理不敏感 |
| 基于 QPS 的 HPA | 请求成本接近恒定的服务 | 易理解，易沟通 | 无法反映排队和请求异质性 |
| 仅队列阈值扩缩容 | 队列语义清晰、流量波动大 | 直接反映压力 | 不校验算力可能误判 |
| 固定副本 + 预留冗余 | 峰值可预测、成本可接受 | 最稳定，运维简单 | 闲时浪费资源 |
| 基于预测的定时扩容 | 高峰有明显时间规律 | 能提前应对整点流量 | 对突发流量无能为力 |
| 多指标闭环控制 | 在线 LLM/GPU 推理 | 对尾延迟控制更稳 | 配置、监控、压测更复杂 |

什么时候没必要上这套复杂方法？

第一种，服务启动只要几秒，且没有显著 warmup。这种情况下 $T_{lead}$ 很短，CPU 或 QPS HPA 往往已经够用。

第二种，请求成本稳定且模型很小。比如一个轻量 embedding 服务，每个请求耗时接近、单副本吞吐可预测，队列和 GPU 双指标未必带来明显收益。

第三种，极低 QPS 场景。如果一天只有零星请求，复杂预测、KEDA 多触发器、定制告警链路的维护成本，可能高于它节省的机器成本。

第四种，离线批处理推理。离线任务看的是总完成时间和资源利用率，不是交互式延迟。此时队列增长未必是坏事，反而是正常调度行为。

所以适用边界可以压缩成一句话：**只有当请求会排队、启动有滞后、用户对尾延迟敏感时，多指标提前扩容才真正有价值。**

---

## 参考资料

1. [Kubernetes Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)
2. [KEDA Prometheus Scaler](https://keda.sh/docs/2.19/scalers/prometheus/)
3. [KEDA FAQ](https://keda.sh/docs/2.16/reference/faq/)
4. [vLLM Metrics](https://docs.vllm.ai/en/v0.8.1/design/v1/metrics.html)
5. [vLLM Production Stack: Autoscaling with KEDA](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html)
6. [KServe: Autoscaling with KEDA](https://kserve.github.io/website/docs/model-serving/predictive-inference/autoscaling/keda-autoscaler)
7. [NVIDIA DCGM Exporter](https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html)
8. [Knative Request Flow](https://knative.dev/v1.17-docs/serving/request-flow/)
9. [Triton Model Configuration / ModelWarmup](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_2610/user-guide/docs/user_guide/model_configuration.html)
