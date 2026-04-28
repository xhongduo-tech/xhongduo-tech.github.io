## 核心结论

SageMaker 模型部署的核心不是“把模型传上去”，而是在给定实例规格下，先满足显存约束，再满足吞吐约束，最后再用端点模式和扩缩容把延迟与成本压到 SLA 以内。这里的 SLA 可以先理解成“服务承诺”，通常体现为可用性、平均延迟、p95/p99 延迟和单位成本是否达标。

先给结论：

1. 先判断“能不能部署”，本质是看单副本峰值显存占用 $L$ 能否放进单卡显存 $V$，即 $L \le V$。
2. 再判断“该怎么部署”，高频、延迟敏感模型优先单模型端点；低频、长尾、多租户模型优先多模型端点。
3. 最后判断“成本和 SLA 是否成立”，核心看吞吐 $Q \le I \cdot r$ 是否成立，以及扩缩容能否跟上流量变化。

一个新手最容易踩错的点是：显存不够不是“性价比差”，而是“根本不能部署”。例如模型峰值显存占用是 `24 GiB`，而某 GPU 只有 `22 GiB` 显存，那么这个实例直接出局，后面谈成本没有意义。

部署决策总览表如下：

| 端点模式 | 适用场景 | 是否共享实例 | 核心风险 | 主要优化手段 |
| --- | --- | --- | --- | --- |
| 单模型端点 | 高频、低延迟、严格 SLA | 否 | 利用率低，成本偏高 | 选更合适实例、压测定容量、目标跟踪扩缩容 |
| 多模型端点 | 多模型、长尾流量、成本敏感 | 是 | 冷加载、缓存抖动、尾延迟上升 | 按热冷拆分模型、提高缓存命中、调实例和 cooldown |
| 异步/批处理 | 非实时请求 | 可共享 | 响应不即时 | 队列化、批量化、离峰处理 |

---

## 问题定义与边界

本文讨论的是 **SageMaker 实时推理部署**。实时推理可以理解为“请求来了，要尽快返回结果”的在线服务。本文不讨论训练，不讨论离线批处理，也不把“模型文件上传到 S3”误认为部署完成。

更准确地说：

$$
\text{SageMaker 模型部署} = \text{在实例规格、吞吐、显存和冷启动之间做约束优化}
$$

这里有两个核心对象。

- 单模型端点：一个端点实例主要服务一个模型。白话讲，就是“一台机器只跑一个主要服务”。
- 多模型端点：多个模型共享同一组实例资源。白话讲，就是“一个服务池里挂很多模型，用请求里的模型名决定本次加载谁”。

玩具例子先建立直觉：

- 单模型端点像把一家店的厨房专门留给一个招牌菜，流程最稳。
- 多模型端点像共享厨房，平时更省，但某道冷门菜第一次有人点时，后厨要先把食材拿出来，延迟会更高。

边界说明表如下：

| 部署方式 | 是否实时推理 | 是否有冷启动 | 是否共享实例 | 是否适合长尾模型 |
| --- | --- | --- | --- | --- |
| 单模型端点 | 是 | 通常较少 | 否 | 一般 |
| 多模型端点 | 是 | 有，尤其首次请求或缓存失效时 | 是 | 是 |
| Async Inference | 否，偏异步 | 可接受 | 可共享 | 视业务而定 |
| Batch Transform | 否，离线 | 不重要 | 批任务式 | 不适合在线长尾 |

因此，本文的边界很明确：讨论的是 **SageMaker 实时端点中的单模型端点与多模型端点**，以及它们如何和实例规格、扩缩容策略一起工作。

---

## 核心机制与推导

先讲硬约束，再讲软优化。

### 1. 硬约束一：显存必须装得下

记：

| 符号 | 含义 |
| --- | --- |
| $L$ | 单副本峰值显存占用，单位 GiB |
| $V$ | 单 GPU 显存，单位 GiB |
| $I$ | 实例数 |
| $r$ | 单实例稳定吞吐，单位 token/s |
| $Q$ | 总到达速率，单位 token/s |
| $C_h$ | 单实例小时成本，单位美元/小时 |
| hit rate | 模型缓存命中率，表示请求到来时模型已在本地可直接服务的比例 |

第一条硬约束是：

$$
L \le V
$$

如果不成立，部署直接失败。这里的“峰值显存占用”不是模型文件大小，而是推理过程中权重、KV Cache、框架开销、并发批次共同作用后的峰值占用。

必须先强调一次你给的大例子。根据 AWS 官方实例规格文档，`g5` 的 A10G 单卡显存约 `22 GiB`，`p4d.24xlarge` 的 A100 单卡显存约 `40 GiB`。如果某模型峰值显存占用是 `24 GiB`，那么：

- 在 `g5` 单卡上：`24 > 22`，装不下。
- 在 `p4d` 单卡上：`24 <= 40`，可以装下。

这不是“贵一点会更快”的问题，而是“便宜实例根本没资格进入候选集”的问题。

### 2. 硬约束二：吞吐必须兜得住流量

第二条硬约束是：

$$
Q \le I \cdot r
$$

这里的 `吞吐` 可以先理解成“每秒稳定处理多少 token 或请求”。如果总流量超过总处理能力，请求就会排队，队列一长，延迟就会上去，SLA 就会失守。

玩具例子：

- 单实例稳定吞吐 $r = 120$ token/s
- 端点有 $I = 3$ 个实例

则总稳定吞吐约为：

$$
I \cdot r = 3 \cdot 120 = 360 \text{ token/s}
$$

如果业务高峰 $Q = 300$ token/s，通常可承受；如果高峰 $Q = 500$ token/s，就会积压，除非提升实例数、提升单实例性能，或者降低每次请求的生成长度。

### 3. 成本估算先看单位吞吐

如果单实例小时成本为 $C_h$，单实例稳定吞吐为 $r$，则单位百万 token 成本可近似写成：

$$
cost/1M \approx \frac{C_h}{3600 \cdot r} \cdot 10^6
$$

这不是账单精算公式，而是工程估算公式，适合快速排除明显不合理的方案。

例如假设某实例小时成本 $C_h = 3.06$ 美元，稳定吞吐 $r = 120$ token/s，则：

$$
cost/1M \approx \frac{3.06}{3600 \cdot 120} \cdot 10^6 \approx 7.08 \text{ 美元}
$$

这一步的价值在于：你可以把“实例贵不贵”转成“每百万 token 成本是否可接受”。因为更贵的实例如果吞吐提升更多，单位成本反而可能更低。

### 4. 多模型端点的真正变量是缓存命中率

多模型端点不是简单地“多个模型一起跑”。它依赖缓存。缓存可以理解成“模型已经下载并加载在本机可直接服务的状态”。

平均延迟可以粗略理解为：

$$
\mathbb{E}[latency] \approx hit\ rate \cdot T_{hot} + (1-hit\ rate) \cdot T_{cold}
$$

其中：

- $T_{hot}$：命中缓存时的热路径延迟
- $T_{cold}$：未命中缓存时的冷路径延迟，包含下载、解压、加载、可能的模型卸载

因为通常 $T_{cold} \gg T_{hot}$，所以命中率一旦下降，平均延迟和尾延迟都会明显变差。单位成本也会被拉高，因为实例时间被花在“搬模型”和“重新加载”上，而不是实际推理上。

真实工程例子：

一个多租户 SaaS 有 1 个高频主模型，日常流量占 80%，另外还有几十个低频专用模型，按租户或场景区分。正确做法通常是：

- 高频主模型走单模型端点，单独保 p95/p99。
- 长尾模型放入多模型端点，共享资源并接受偶发冷启动。
- 扩缩容基于端点整体流量设置，不以某个长尾模型的瞬时请求做阈值。

这样做的原因很直接：热门模型的目标是稳定，长尾模型的目标是摊薄成本，这两类目标不应混进同一套缓存与调度策略。

---

## 代码实现

先建立最小心智模型：部署链路通常是 `S3 -> Model -> EndpointConfig -> Endpoint`。上传模型文件只是第一步，不是部署完成。

### boto3 创建单模型端点

下面先给一个能本地运行的容量估算脚本，用来判断“显存能否放下”和“单位百万 token 成本大概是多少”。

```python
from dataclasses import dataclass

@dataclass
class DeployPlan:
    model_vram_gib: float
    gpu_vram_gib: float
    instances: int
    throughput_per_instance_tps: float
    hourly_cost_usd: float

    def fits(self) -> bool:
        return self.model_vram_gib <= self.gpu_vram_gib

    def total_throughput(self) -> float:
        return self.instances * self.throughput_per_instance_tps

    def cost_per_1m_tokens(self) -> float:
        return self.hourly_cost_usd / (3600 * self.throughput_per_instance_tps) * 1_000_000

g5_plan = DeployPlan(
    model_vram_gib=24,
    gpu_vram_gib=22,
    instances=2,
    throughput_per_instance_tps=120,
    hourly_cost_usd=3.06,
)

p4d_plan = DeployPlan(
    model_vram_gib=24,
    gpu_vram_gib=40,
    instances=2,
    throughput_per_instance_tps=120,
    hourly_cost_usd=3.06,
)

assert g5_plan.fits() is False
assert p4d_plan.fits() is True
assert p4d_plan.total_throughput() == 240
assert round(p4d_plan.cost_per_1m_tokens(), 2) == 7.08
```

真正创建单模型端点时，流程一般如下：

```python
import boto3

sm = boto3.client("sagemaker")

role_arn = "arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
model_name = "demo-single-model"
endpoint_config_name = "demo-single-config"
endpoint_name = "demo-single-endpoint"

container = {
    "Image": "<account>.dkr.ecr.<region>.amazonaws.com/my-inference-image:latest",
    "ModelDataUrl": "s3://my-bucket/models/model.tar.gz",
    "Environment": {
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20"
    },
}

sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role_arn,
    PrimaryContainer=container,
)

sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": "ml.g5.2xlarge",
            "InitialVariantWeight": 1.0,
        }
    ],
)

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name,
)
```

单模型端点的优点是简单。请求到了，不需要再判断“这次该加载哪个模型”。

### 多模型端点的模型目录组织

多模型端点要求多个模型共享同一套推理容器和实例池。常见做法是把模型按目录放进同一个 S3 前缀。

```text
s3://my-bucket/mme/
  tenant-a/model.tar.gz
  tenant-b/model.tar.gz
  tenant-c/model.tar.gz
```

创建模型时，容器要启用 `MultiModel` 模式，请求时再通过目标模型名决定本次加载哪一个模型。

```python
import boto3

sm = boto3.client("sagemaker")
runtime = boto3.client("sagemaker-runtime")

sm.create_model(
    ModelName="demo-mme-model",
    ExecutionRoleArn="arn:aws:iam::<account-id>:role/SageMakerExecutionRole",
    Containers=[
        {
            "Image": "<account>.dkr.ecr.<region>.amazonaws.com/my-mme-image:latest",
            "Mode": "MultiModel",
            "ModelDataUrl": "s3://my-bucket/mme/",
        }
    ],
)

response = runtime.invoke_endpoint(
    EndpointName="demo-mme-endpoint",
    ContentType="application/json",
    TargetModel="tenant-a/model.tar.gz",
    Body=b'{"inputs":"hello"}',
)
```

这个请求中的 `TargetModel` 就是“目标模型名”。如果该模型已在本机缓存中，走热路径；如果没有，就会触发下载和加载。也正因为这样，多模型端点不能只看平均延迟，必须重点看 p95/p99 和缓存命中率。

### 按 InvocationsPerInstance 设置扩缩容

自动扩缩容一般用 `Application Auto Scaling` 配合 `InvocationsPerInstance` 做目标跟踪。这个指标可以先理解成“每个实例每分钟平均被调用多少次”。

```python
import boto3

aas = boto3.client("application-autoscaling")

resource_id = "endpoint/demo-mme-endpoint/variant/AllTraffic"

aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=8,
)

aas.put_scaling_policy(
    PolicyName="mme-invocations-tracking",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleOutCooldown": 300,
        "ScaleInCooldown": 600,
    },
)
```

这里要注意一件非常重要的事：多模型端点扩缩容看的通常是 **端点聚合流量**，不是某一个模型单独的局部峰值。也就是说，即使某个冷门模型突然热起来，只要总指标没有及时反映出来，扩容也可能慢半拍。所以阈值必须靠压测来定，不能照抄示例值。

---

## 工程权衡与常见坑

SageMaker 里最容易被低估的，不是“怎么建端点”，而是“端点在真实流量下怎么退化”。

常见坑与规避如下：

| 问题 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 缓存命中率低导致抖动 | p95/p99 明显变差，偶发超时 | 工作集大于实例可缓存容量，反复卸载和重载模型 | 增大实例内存/显存，增加实例数，热冷模型拆分 |
| 扩缩容跟不上流量 | 突发高峰时排队严重 | 阈值设得太激进或 cooldown 不合理 | 先压测再设 TargetValue，缩短 scale-out cooldown |
| 把 NVMe 误解为显存 | 加载快了，但模型仍启动失败 | NVMe 只是本地存储，不是 GPU 显存 | 先检查 $L \le V$，再谈 NVMe 优化 |
| 热门模型与长尾模型混放 | 主模型 p95/p99 被拖慢 | 长尾模型冷加载干扰热点模型 | 热门模型独立单模型端点，长尾模型放 MME |
| 只看平均延迟 | 监控看着正常，用户仍抱怨卡顿 | 尾延迟被冷启动掩盖 | 重点监控 p95/p99、ModelCacheHit、加载/卸载指标 |

重点展开两个坑。

第一，多模型端点的缓存不是“越多模型越省钱”的魔法。只要工作集超过内存或显存容量，就会出现 thrashing，可以理解成“缓存抖动”：刚加载进去的模型很快又被挤出去，下次请求又要重新加载。这时成本未必更低，因为机器时间被浪费在重复加载上。

第二，`g5` 这类实例带 NVMe，只能改善模型从本地盘读取和加载的速度，不能把 `22 GiB` 显存变成 `40 GiB`。所以“有 NVMe”解决的是冷加载速度，不解决显存硬约束。

再看一个真实工程坑例。某 SaaS 团队把高频主模型和几十个低频专用模型都放到同一个 MME，表面上实例数下降了，但主模型的 p95/p99 明显恶化。原因不是主模型本身变慢，而是共享实例后，长尾模型的冷启动和缓存挤压影响了整体服务池。正确做法是：主模型单独部署，长尾模型单独一组 MME。

---

## 替代方案与适用边界

“单模型端点 vs 多模型端点”不是唯一选择。真正的工程判断是：业务到底要的是低延迟、低成本，还是高弹性。

| 方案 | 适用场景 | 优点 | 缺点 | 是否有冷启动 | 是否适合严格 SLA |
| --- | --- | --- | --- | --- | --- |
| 单模型端点 | 高频主模型、稳定流量、低延迟 | 稳定、简单、尾延迟更可控 | 利用率可能低，成本高 | 一般较少 | 是 |
| 多模型端点 | 多租户、模型多、长尾明显 | 资源共享，成本更友好 | 冷加载、抖动、调优复杂 | 有 | 一般不如单模型稳 |
| Serverless Inference | 流量稀疏、无需长期保活 | 按需触发，省闲置成本 | 冷启动更明显 | 有 | 通常不适合严格 SLA |
| Async Inference | 请求耗时长、可异步返回 | 能消化长任务 | 非实时 | 可接受 | 否 |
| Batch Transform | 离线批量任务 | 成本可控，吞吐高 | 不适合在线请求 | 不重要 | 否 |

给一个新手版选择题：

- 高频主模型，延迟敏感：单模型端点。
- 长尾专用模型很多，访问稀疏：多模型端点。
- 不是实时请求：优先考虑 Async Inference 或 Batch Transform。
- 流量极不稳定且能接受冷启动：再考虑 Serverless。

最后把适用边界说清楚：

- 单模型端点适合热门、稳定、低延迟、严格 SLA。
- 多模型端点适合长尾、共享资源、成本优先。
- 其他方案适合非实时任务，或流量极度波动的场景。

因此，部署决策的正确顺序始终是：

1. 先过显存门槛。
2. 再过吞吐门槛。
3. 然后按业务形态选择端点模式。
4. 最后用扩缩容和缓存策略验证成本与 SLA 是否同时成立。

---

## 参考资料

1. [Single-model endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-single-model.html)
2. [Multi-model endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)
3. [Set Auto Scaling Policies for Multi-Model Endpoint Deployments](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints-autoscaling.html)
4. [Instance recommendations for multi-model endpoint deployments](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoint-instance.html)
5. [Specifications for Amazon EC2 accelerated computing instances](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
6. [Auto scaling policy overview](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-policy.html)
