## 核心结论

大模型部署架构的本质不是“把模型跑起来”，而是用可接受的成本，稳定满足延迟、吞吐和可用性目标。对初级工程师来说，可以先记住三条：

1. 单机部署适合“模型能放进一台机器，且并发不高”的场景。
2. 分布式部署适合“模型放不下”或“单机吞吐不够”的场景，核心手段是模型并行、流水线并行和请求调度。
3. 混合部署适合“请求类型差异大”或“既要控制成本又要保留高峰能力”的场景，例如常规请求走小模型，复杂请求走大模型集群。

部署选择不是按“模型参数越大越高级”来做，而是按资源约束来做。最常见的约束有四个：模型权重占用、KV Cache 占用、并发量、故障恢复要求。KV Cache 可以理解为“模型在生成过程中为后续 token 保留的中间记忆”，上下文越长、并发越高，它越吃显存。

一个直接可用的判断表如下：

| 部署拓扑 | 适用模型规模 | 典型目标 | 优点 | 主要限制 |
| --- | --- | --- | --- | --- |
| 单机单卡 | 1B 到数十亿参数 | 快速上线、低运维复杂度 | 架构简单、排障容易 | 容量和吞吐上限低 |
| 单机多卡 | 6B 到几十B | 提升单机容量 | 不需要跨机网络 | 受限于单机 PCIe/NVLink 和机箱规模 |
| 多机分布式 | 20B 到 70B 及以上 | 放下大模型、支撑高并发 | 可横向扩展 | 网络、调度、故障处理复杂 |
| 混合部署 | 多模型、多服务等级 | 成本和性能平衡 | 能分层承接请求 | 路由与治理逻辑更复杂 |

如果只用一句话概括：部署架构是“模型切多细、请求怎么分、资源怎么扩、故障怎么切”的组合问题。

---

## 问题定义与边界

讨论“大模型部署架构”时，先把边界说清楚，否则容易把训练方案、评测方案和在线推理混在一起。

本文只讨论在线推理，也就是外部请求到来后，服务返回文本结果的过程，不讨论训练和微调。在线推理的核心指标通常是：

| 指标 | 白话解释 | 常见关注点 |
| --- | --- | --- |
| 首 token 延迟 | 用户发请求后，多久看到第一个字 | 聊天体验是否“卡住” |
| 吞吐 | 单位时间能处理多少请求或 token | 机器利用率和成本 |
| 并发 | 同时挂在系统里的请求数 | 队列是否堆积 |
| 可用性 | 服务是否持续可访问 | 故障时是否能自动恢复 |
| 单位成本 | 每次请求花多少钱 | 商业可持续性 |

零基础读者最容易忽略的是：模型大小不是唯一变量。即使是同一个 7B 模型，短上下文、低并发和长上下文、高并发，对显存要求完全不同。因为实际显存占用大致可写成：

$$
总显存 \approx 权重显存 + KV\ Cache显存 + 框架/运行时开销
$$

其中权重显存决定“模型能否装进去”，KV Cache 显存决定“并发和上下文能否扛住”。

一个玩具例子：

你有一个 6B 模型，量化后能放进单张 GPU。每天只有少量内部请求，请求上下文也不长。这时单机部署通常是最优解。原因不是它“性能最好”，而是它的系统总复杂度最低。

一个真实工程例子：

团队要上线一个 34B 或 70B 级别的问答服务，请求量有昼夜波动，还要求故障时自动切换。此时单机方案即使理论上能跑，也往往因为显存、吞吐或可用性要求失败，必须引入多卡甚至多机，并在前面加负载均衡、自动扩缩容和健康检查。

因此，部署架构的决策边界通常可以表述为：

| 触发条件 | 单机还能撑住吗 | 何时升级架构 |
| --- | --- | --- |
| 模型权重放不进单卡 | 不能 | 升级到单机多卡或多机分片 |
| 单机吞吐跟不上峰值流量 | 通常不能 | 增加副本或做分布式调度 |
| 需要跨区容灾 | 不能完整满足 | 增加多区域副本和故障转移 |
| 请求差异很大 | 勉强能做 | 用混合部署做分层路由 |

---

## 核心机制与推导

分布式部署最核心的两个词是模型并行和流水线并行。

模型并行可以理解为“同一层的参数拆到多张卡上一起算”；流水线并行可以理解为“不同层分给不同卡，像流水线一样一段一段传递”。前者主要解决“单层太大装不下”，后者主要解决“整网太深单卡放不下或利用率低”。

最常见的资源规划公式是：

$$
GPU总数 = tensor\ parallelism \times pipeline\ parallelism
$$

其中 tensor parallelism 表示张量并行度，也就是一层内部被拆成几份；pipeline parallelism 表示流水线并行度，也就是模型层级被切成几段。

例如：

- tensor parallelism = 4
- pipeline parallelism = 2

那么需要的 GPU 数量就是：

$$
4 \times 2 = 8
$$

这个公式看起来简单，但实际非常重要。很多部署失败不是因为不会配框架，而是前期资源预算少算了一维，只按“4 卡就够”申请机器，结果真正上线需要 8 卡。

可以把整个推理拓扑理解成下面的流程：

1. 请求先进入网关或负载均衡器。
2. 调度层根据模型、副本负载、队列长度决定把请求送到哪个推理服务。
3. 推理服务内部再决定是否做动态批处理。动态批处理可以理解为“把短时间内到达的多个请求合并起来一起算”，用更高吞吐换一点排队时间。
4. 如果模型是分布式部署，请求会在多张 GPU 之间按 tensor/pipeline 规则完成一次前向计算。
5. 结果返回后，监控系统记录延迟、错误率、GPU 利用率和队列长度，用于扩缩容和告警。

这里有一个常见误区：很多人以为“多卡一定更快”。这不准确。多卡分片解决的是“装不下”和“吞吐不够”，但也引入通信开销。跨卡、跨机通信会增加额外延迟。于是首 token 延迟可以粗略理解为：

$$
T_{total} = T_{compute} + T_{communication} + T_{queue}
$$

- $T_{compute}$ 是真正做矩阵计算的时间。
- $T_{communication}$ 是多卡同步和传输的时间。
- $T_{queue}$ 是请求在系统里等待批处理、等待空闲 worker 的时间。

单机方案通常 $T_{communication}$ 低，但容量有限；多机方案容量大，但 $T_{communication}$ 更高。真正的架构设计就是平衡这三项。

再看一个玩具例子。

假设一个小模型在单卡上每秒能处理 40 个请求，高峰只有 15 个请求每秒。这时你没有必要做复杂分布式，因为系统瓶颈不存在。相反，复杂架构会增加维护成本。

再看真实工程例子。

如果一个 70B 模型必须用 8 张 GPU 才能放下，而业务高峰到来时单个副本只能支撑 8 QPS，请求峰值却达到 40 QPS，那么你不仅要做 8 卡模型并行，还要做至少 5 组副本的横向扩展。也就是说，系统设计不只是“8 张卡能不能跑”，而是“8 张卡为一组时，一共要几组，调度怎么分，故障时能不能自动切到别组”。

---

## 代码实现

下面先给一个最小可运行的 Python 例子，用来模拟部署规划。它不依赖外部框架，但能把核心计算讲清楚。

```python
from math import ceil

def required_gpus(tensor_parallel: int, pipeline_parallel: int) -> int:
    assert tensor_parallel >= 1
    assert pipeline_parallel >= 1
    return tensor_parallel * pipeline_parallel

def required_replicas(peak_qps: int, qps_per_replica: int) -> int:
    assert peak_qps >= 0
    assert qps_per_replica > 0
    return max(1, ceil(peak_qps / qps_per_replica))

def cluster_gpu_budget(
    tensor_parallel: int,
    pipeline_parallel: int,
    peak_qps: int,
    qps_per_replica: int
) -> int:
    per_replica_gpus = required_gpus(tensor_parallel, pipeline_parallel)
    replicas = required_replicas(peak_qps, qps_per_replica)
    return per_replica_gpus * replicas

# 玩具例子：4 路 tensor 并行，2 路 pipeline 并行
assert required_gpus(4, 2) == 8

# 如果单个副本能扛 8 QPS，峰值是 40 QPS，则需要 5 个副本
assert required_replicas(40, 8) == 5

# 总 GPU 预算 = 8 * 5 = 40
assert cluster_gpu_budget(4, 2, 40, 8) == 40

print("deployment planning ok")
```

这个例子表达了一个工程事实：部署预算要同时看“每个副本需要几张卡”和“需要几个副本”，不能只看其中一个。

真实工程里，如果你在 SageMaker 上用 DJL Serving 和 DeepSpeed，可以通过环境变量控制张量并行度。示意配置如下：

```bash
docker pull deepjavalibrary/djl-serving:0.19.0-deepspeed

aws sagemaker create-model \
  --model-name gpt-j \
  --primary-container Image=123456789012.dkr.ecr.us-east-1.amazonaws.com/djl-deepspeed:latest,ModelDataUrl=s3://your-bucket/gpt-j.tar.gz,Environment="{TENSOR_PARALLEL_DEGREE=2,MAX_BATCH_DELAY=100,BATCH_SIZE=4}"
```

这里几个字段的作用要分清：

| 配置项 | 作用 | 影响 |
| --- | --- | --- |
| `TENSOR_PARALLEL_DEGREE` | 模型在多少张卡上分片 | 决定单副本资源需求 |
| `BATCH_SIZE` | 一次合并多少请求 | 提高吞吐，但可能增加等待 |
| `MAX_BATCH_DELAY` | 最长等待多少时间再凑批 | 影响延迟和利用率 |
| 副本数/HPA 配置 | 启动多少个服务副本 | 决定整体吞吐和容灾能力 |

如果用 Kubernetes + Triton/TensorRT-LLM，一般会再加一层服务治理：

1. Ingress 或网关负责入口流量。
2. Service 或负载均衡器把流量分发到多个推理 Pod。
3. Pod 内部绑定 GPU 资源，按 tensor/pipeline 配置启动。
4. HPA 或自定义扩缩容器根据队列长度、GPU 利用率等指标扩副本。
5. Prometheus、日志系统和告警系统负责观测与恢复。

对于初学者，重要的不是一开始把平台全搭出来，而是先知道代码和配置分别控制哪一层：模型分片控制“能不能跑”，批处理控制“快不快”，副本和调度控制“扛不扛峰值”。

---

## 工程权衡与常见坑

部署问题通常不是“模型起不来”，而是“上线后不稳定”。下面这些坑最常见。

第一类坑是扩缩容指标选错。只盯 CPU 或内存，往往会误判。推理服务最容易先满的是 GPU 显存、GPU 利用率或请求队列，而不是 CPU。一个典型现象是：CPU 只用了 30%，但 GPU 已经打满，用户仍然超时。

更合理的指标优先级如下：

| 指标 | 优先级 | 适用原因 | 限制 |
| --- | --- | --- | --- |
| 队列长度 | 高 | 最直接反映请求积压 | 需要服务暴露队列指标 |
| GPU 利用率 | 高 | 直接反映推理压力 | 不能单独反映排队情况 |
| GPU 显存占用 | 高 | 能发现 KV Cache 压力 | 对短时波动不敏感 |
| Batch 大小 | 中 | 反映调度器是否已饱和 | 与业务流量关系间接 |
| CPU/内存 | 低 | 适合普通 Web 服务 | 对 GPU 推理参考价值有限 |

第二类坑是把“模型能跑”误当成“服务可用”。单副本部署即使性能足够，也可能在进程崩溃、节点故障或区域异常时完全中断。因此至少要有：

| 检查项 | 目的 |
| --- | --- |
| 健康检查 | 判断进程是否还能接流量 |
| 就绪检查 | 防止未加载完成的副本接入 |
| 自动重启 | 节点或进程失败后恢复 |
| 多副本 | 避免单点故障 |
| 跨区容灾 | 防止单可用区失效 |
| 日志与指标 | 便于定位慢请求和错误峰值 |

第三类坑是忽略 KV Cache。很多团队上线前只测权重是否能装进显存，没有测长上下文和多并发。结果一到真实场景，显存暴涨，服务频繁 OOM。这个问题的规避方法不是只靠“买更大卡”，而是同时做上下文长度限制、批大小控制、缓存回收和容量规划。

第四类坑是动态批处理配得过大。批处理越大，吞吐通常越高，但首 token 延迟也可能上升。对聊天场景，如果用户在意交互感，就不能盲目把批次拉满。工程上常见做法是为不同接口设置不同策略：实时聊天走小批次，离线生成走大批次。

一个实用的故障转移 checklist 如下：

| 项目 | 最低要求 |
| --- | --- |
| 监控 | 采集延迟、错误率、队列、GPU 指标 |
| 健康探测 | 区分 liveness 和 readiness |
| 自动扩缩容 | 基于队列或 GPU 指标，而不是只看 CPU |
| 副本隔离 | 不同副本分散到不同节点 |
| 跨区容灾 | 至少有主备区域或可切换区域 |
| DNS/流量切换 | 健康异常时能自动摘除故障入口 |

---

## 替代方案与适用边界

单机、集群、混合三种模式，没有绝对优劣，只有适用边界。

| 模式 | 最适合的场景 | 不适合的场景 |
| --- | --- | --- |
| 单机部署 | 内部工具、低并发 API、验证阶段 | 超大模型、强容灾、高峰明显 |
| 分布式集群 | 大模型生产服务、高并发、多租户 | 团队缺少平台能力、预算很紧 |
| 混合部署 | 请求分层明显、冷热流量差异大 | 路由策略无法稳定定义 |

混合部署是近几年很常见的折中方案。它的思路不是“所有请求都打到最强模型”，而是把请求按复杂度、时延等级或成本等级分流。例如：

- 边缘或普通节点部署小模型，处理轻量问答、摘要、分类。
- 云端大集群部署大模型，处理复杂推理、长上下文、关键业务请求。
- 调度层根据请求特征把流量送到不同后端。

这样做的价值在于，昂贵的多机大模型只服务真正需要它的请求。否则，大量简单请求也占用高价 GPU，成本会迅速失控。

还要说明一个边界：不是所有团队都适合自己搭完整分布式平台。如果团队规模小、上线速度优先，使用云厂商托管推理或现成推理平台，往往比自建 Triton + K8s + 监控 + 调度更现实。自建的优势是灵活和可控，代价是平台复杂度、排障成本和长期维护压力。

最终可以用一个简化决策法：

1. 先判断模型能否放进单机。
2. 再判断单机吞吐能否覆盖峰值。
3. 再判断业务是否要求高可用和跨区容灾。
4. 如果前三步任一失败，再引入更复杂的分布式或混合架构。

这个顺序很重要。对初学者来说，最稳妥的路径不是一开始就追求“最先进架构”，而是按约束逐层升级。

---

## 参考资料

- WhaleFlux，多机部署与模型并行策略指南：<https://www.whaleflux.com/blog/how-to-deploy-llms-at-scale-multi-machine-inference-and-model-deployment/>
- AWS，使用 SageMaker、DJL Serving 与 DeepSpeed 部署 GPT-J 推理：<https://aws.amazon.com/blogs/machine-learning/deploy-large-models-on-amazon-sagemaker-using-djlserving-and-deepspeed-model-parallel-inference/>
- NVIDIA，TensorRT-LLM 与 Triton 的 Autoscaling、Load Balancing、多 GPU 部署教程：<https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2600/user-guide/docs/tutorials/Deployment/Kubernetes/TensorRT-LLM_Autoscaling_and_Load_Balancing/README.html>
- Google Cloud，GKE 上机器学习推理的 autoscaling 最佳实践：<https://cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling>
- UMA Technology，多区域 autoscaling 组与故障恢复建议：<https://umatechnology.org/disaster-recovery-plans-for-autoscaling-groups-monitored-through-cloud-native-logging/>
