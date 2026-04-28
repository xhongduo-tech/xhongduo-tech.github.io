## 核心结论

Azure ML 在线端点配置的重点，不是“把模型传上去”这一步，而是把请求如何进入、如何分流、如何失败、如何被保护这四件事讲清楚。

先给结论：

1. `endpoint` 是统一 HTTP 入口，`deployment` 才是真正承载模型推理的后端单元。新手可以把 `endpoint` 理解为“门牌号”，把 `deployment` 理解为“门后的一组房间和服务员”。请求先到门牌号，再被分到具体房间。
2. 线上表现主要由四组配置共同决定：容量配置 `instance_count`，并发配置 `max_concurrent_requests_per_instance`，流量配置 `traffic` / `mirror_traffic`，安全配置 `auth_mode` 与 `public_network_access`。
3. 托管推理把主机、补丁、基础监控这些运维工作交给 Azure ML，但容器启动时间、镜像拉取时间、模型加载时间、探针阈值和请求超时，仍然直接决定可用性。
4. 一个 `endpoint` 下挂 `blue` 和 `green` 两个 `deployment`。`blue` 负责 90% 线上流量，`green` 先只接 10% 灰度流量；如果 `instance_count` 太小，先暴露出的是排队、`429`、`P99` 上升和超时，而不是模型逻辑错误。
5. 生产环境默认建议优先考虑 `aad_token` 认证，并尽量关闭公网入口，即 `public_network_access: disabled`，再通过私网访问。原因很直接：静态密钥更方便试验，身份令牌更适合长期生产治理。

| 概念 | 作用 | 直接影响指标 |
| --- | --- | --- |
| `endpoint` | 统一推理入口，承接域名、鉴权、路由 | 可访问性、路由正确率 |
| `deployment` | 实际跑模型容器的后端单元 | 吞吐、延迟、错误率 |
| `instance_count` | 副本数，也就是后端实例数量 | QPS 上限、`429`、P99 |
| `traffic` | 生产流量按比例分配到各 deployment | 灰度效果、版本暴露面 |
| `auth_mode` | 调用时的鉴权方式 | 401、越权风险、运维复杂度 |
| `public_network_access` | 是否允许公网直接访问 | 攻击面、网络隔离强度 |

---

## 问题定义与边界

本文只讨论 Azure ML 的 `managed online endpoint`，也就是托管在线推理端点。不展开训练作业、批处理推理、离线评估，也不展开模型本身的效果优化。

真正要回答的问题只有一个：

一个在线请求从客户端发出后，如何从入口路由到某个 deployment，并在什么条件下成功、超时、排队、被拒绝或被判为不可用？

新手版理解可以写成一句话：

`endpoint` 像门牌号，`deployment` 像门后不同房间。请求先到门牌号，再按流量规则进房间；如果房间人数太少，来的人太多，就会排队、超时或被拒绝。

这件事的边界要画清楚，否则很容易把“资源创建成功”和“线上可稳定服务”混为一谈。Azure 门户里看到资源创建完成，不等于线上推理链路已经健康；只有下面这条链路稳定，配置才算成立：

```text
Client
  -> auth
  -> Endpoint
  -> Traffic routing
  -> Deployment
  -> probe check
  -> queue
  -> Model container
  -> timeout check
  -> Response
```

四个关键检查点分别是：

1. `auth`：请求有没有合法身份，没过就直接 `401`。
2. `probe`：容器是否处于可接流量状态，没过就不会稳定接流量，甚至会反复重启。
3. `queue`：请求超过当前并发处理能力后，会先进入等待，再决定是否拒绝。
4. `timeout`：请求执行太久，即使模型最终能算出来，也可能在超时边界前被系统判失败。

玩具例子可以这样想：

你开了一家只有 2 个服务员的小店，门口有统一接待台。接待台就是 `endpoint`，服务员所在的具体窗口就是 `deployment`。如果一瞬间来了 30 个人，而每个窗口一次只能认真处理 4 个人，那么超出的请求一定会先排队，之后出现等待过长、拒绝进入或者超时离开。这不是“菜做错了”，而是“容量没配够”。

---

## 核心机制与推导

理解 Azure ML 在线端点，最重要的不是记字段名，而是把容量公式和生命周期顺序记住。

先看并发上限。官方排障文档给出的核心约束可以近似写为：

$$
Q_{\max} \approx 2 \times I \times C
$$

其中：

- $I = \text{instance\_count}$，副本数
- $C = \text{max\_concurrent\_requests\_per\_instance}$，单实例允许并发请求数

这里“并发”不是“每秒请求数”，而是“同一时刻系统能同时挂住并处理多少个请求”。白话解释：它描述的是柜台前同时能被接住的人数，不是一天总共来了多少人。

例如：

- 假设 `I=2`
- 假设 `C=4`

则并行处理上限约为：

$$
Q_{\max} \approx 2 \times 2 \times 4 = 16
$$

第 17 个请求同时到达时，系统可能先排队，然后触发拒绝或超时。官方排障页明确写到，超过 `2 * max_concurrent_requests_per_instance * instance_count` 的并行请求会被拒绝，并表现为 `429 Too many pending requests`。

如果此时再配置：

```json
{"blue": 90, "green": 10}
```

那约 10% 的请求会进入 `green` 做灰度，90% 仍进入 `blue`。注意，这里的“约 10%”是流量分配规则，不是性能保证。若 `green` 只有 1 个实例、单实例并发又很低，它即使只接 10% 流量，也可能先爆掉。

再看推理链路里的三类关键机制。

第一类是请求时间机制：

- `request_timeout_ms`：单次推理超时时间。白话解释：系统愿意等这个请求多久。
- `max_concurrent_requests_per_instance`：单实例同时接多少请求。
- `max_queue_wait_ms`：旧的排队等待上限，官方 schema 已标记为 deprecated，思路上应优先通过 `request_timeout_ms` 覆盖网络和排队等待。

第二类是探针机制：

- `startup_probe`：启动探针，判断容器是不是“终于启动起来了”。
- `readiness_probe`：就绪探针，判断容器“现在能不能接真实流量”。
- `liveness_probe`：存活探针，判断容器“是不是活着；不活就该重启”。

第三类是流量机制：

- `traffic`：真实生产流量按权重分配。
- `mirror_traffic`：镜像流量，也叫 shadow traffic，客户端只收到主版本响应，但一部分请求会被复制给影子版本做验证。

可以把整个时序写成下面这样：

```text
请求到达
  -> 鉴权通过?
     否 -> 401
     是 -> 进入 endpoint
  -> 是否指定 azureml-model-deployment?
     是 -> 直达目标 deployment
     否 -> 按 traffic / mirror_traffic 规则路由
  -> deployment 对应容器 readiness 是否通过?
     否 -> 不稳定接流量 / 返回失败
     是 -> 尝试占用实例并发槽位
  -> 并发槽位是否足够?
     否 -> 排队
     排队过多 -> 429 或超时
     是 -> 执行模型推理
  -> 是否超过 request_timeout_ms?
     是 -> 408/504 类超时表现
     否 -> 返回响应
```

下面这张表适合记配置和指标之间的关系。关于默认值，本文以当前官方 YAML schema 中明确列出的值为准；`startup_probe` 的字段形状与其他 probe 一致，工程上通常按同样的探针参数族来理解和调优。

| 字段 | 作用 | 默认值 |
| --- | --- | --- |
| `request_timeout_ms` | 单次请求最多执行多久 | `5000` |
| `max_concurrent_requests_per_instance` | 单实例最大并发请求数 | `1` |
| `max_queue_wait_ms` | 请求最多排队多久，已弃用 | `500` |
| `startup_probe` | 容器启动是否完成 | 通常使用 `ProbeSettings` 字段族 |
| `readiness_probe` | 容器是否可接流量 | `ProbeSettings` 默认：`initial_delay=10s`、`period=10s`、`timeout=2s`、`success_threshold=1`、`failure_threshold=30` |
| `liveness_probe` | 容器是否仍健康，失败可触发重启 | `ProbeSettings` 默认：`initial_delay=10s`、`period=10s`、`timeout=2s`、`success_threshold=1`、`failure_threshold=30` |

真实工程例子：

一个文本分类模型在 `blue` 上稳定运行，准备上线 `green` 新版本。团队先把 `green` 设成 `instance_count=1`，`traffic=10%`，同时要求只能内网服务调用，于是把 `auth_mode` 设为 `aad_token`，把 `public_network_access` 设为 `disabled`。上线后，第一批问题通常不是“预测错了”，而是：

- `green` 首次拉镜像和加载模型需要 35 到 45 秒
- readiness probe 过早探测，导致 deployment 一直不 Ready
- 少量真实流量一进来，`P99` 先升高
- 高峰一到，`429` 和 `5xx` 率先暴露容量不足

这就是为什么配置项必须放在同一张脑图里看，而不是孤立地逐个背。

---

## 代码实现

代码实现建议按三层来看：`endpoint` 配置、`deployment` 配置、调用方式。因为 Azure ML 的对象边界本来就是这么设计的。

先给一个最小 managed online endpoint 示例，重点只放两个字段：

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: my-azureml-endpoint
auth_mode: aad_token
public_network_access: disabled
```

这两个字段的意义分别是：

- `auth_mode: aad_token`：要求调用方使用 Microsoft Entra 身份令牌，而不是长期不失效的静态 key。
- `public_network_access: disabled`：关闭公网入口，请求必须通过工作区私有终结点路径进入。

再看 deployment。真正影响线上表现的大部分配置都在这里：

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: blue
endpoint_name: my-azureml-endpoint

model:
  path: ./model

code_configuration:
  code: ./src
  scoring_script: score.py

environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest
  conda_file: ./environment/conda.yaml

instance_type: Standard_DS3_v2
instance_count: 1

request_settings:
  request_timeout_ms: 5000
  max_concurrent_requests_per_instance: 1

readiness_probe:
  initial_delay: 20
  period: 10
  timeout: 2
  failure_threshold: 30

liveness_probe:
  initial_delay: 20
  period: 10
  timeout: 2
  failure_threshold: 30
```

这里每个关键字段都不是“装饰项”：

- `instance_count` 决定副本数。
- `request_timeout_ms` 决定系统最多等多久。
- `max_concurrent_requests_per_instance` 决定单实例允许吃下多少并发。
- `readiness_probe` 决定 deployment 什么时候才算可以接流量。
- `liveness_probe` 决定进程卡死时是否会被拉起。

然后再增加一个 `green` deployment，用于灰度：

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: green
endpoint_name: my-azureml-endpoint

model:
  path: ./model-v2

code_configuration:
  code: ./src
  scoring_script: score.py

environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest
  conda_file: ./environment/conda.yaml

instance_type: Standard_DS3_v2
instance_count: 1

request_settings:
  request_timeout_ms: 5000
  max_concurrent_requests_per_instance: 1
```

创建后，可以把线上真实流量分成 `blue=90, green=10`：

```bash
az ml online-endpoint update \
  --name my-azureml-endpoint \
  --traffic "blue=90 green=10"
```

如果你只是想验证 `green` 是否可用，不必马上改线上权重，直接指定 deployment 即可：

```bash
az ml online-endpoint invoke \
  --name my-azureml-endpoint \
  --deployment-name green \
  --request-file sample-request.json
```

如果走 HTTP 请求，也可以显式带认证头和部署头。`Authorization` 是身份认证头，白话解释就是“我是谁”；`azureml-model-deployment` 是定向路由头，白话解释就是“我要去哪个后端版本”。

```bash
curl -X POST "$SCORING_URI" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "azureml-model-deployment: green" \
  -d @sample-request.json
```

下面给一个可运行的 Python 玩具程序，用来验证并发上限和灰度分流的基本计算逻辑：

```python
def q_max(instance_count: int, max_concurrent_requests_per_instance: int) -> int:
    return 2 * instance_count * max_concurrent_requests_per_instance

def route_counts(total_requests: int, blue_weight: int, green_weight: int):
    assert blue_weight + green_weight == 100
    blue = round(total_requests * blue_weight / 100)
    green = total_requests - blue
    return blue, green

# 并发上限示例：I=2, C=4
assert q_max(2, 4) == 16

# 灰度分流示例：100 个请求，blue=90, green=10
blue, green = route_counts(100, 90, 10)
assert blue == 90
assert green == 10

# 一个简单容量判断
incoming_parallel_requests = 17
capacity = q_max(2, 4)
assert incoming_parallel_requests > capacity
print("capacity =", capacity, "incoming =", incoming_parallel_requests)
```

这段代码当然不等于 Azure ML 的内部实现，但它足够帮助初学者建立正确直觉：流量比例和容量上限是两个不同维度，必须同时满足。

---

## 工程权衡与常见坑

在线推理配置不是“值越大越稳”。很多字段是在不同风险之间做交换。

第一个权衡是 `instance_count` 和成本。

副本数太小，问题会优先表现为 `429`、排队和 `P99` 抬升；副本数太大，成本上升明显。Azure 官方 schema 还特别提醒：高可用场景通常建议至少 `3` 个实例。这不是绝对规则，但它提醒你，`1` 个实例更像最小可运行，而不是生产稳态。

第二个权衡是 `max_concurrent_requests_per_instance`。

这个值太小，节点利用率低，还可能更早触发 `429`；太大，则问题从“直接拒绝”转成“所有请求都变慢”。因为模型如果本身不支持真实并发，强行把更多请求塞给一个实例，只会让线程争用、显存争用或 GIL/进程池竞争更严重。

第三个权衡是探针和超时。

一个模型加载需要 40 秒，但 `startup_probe.initial_delay` 只给 10 秒，`failure_threshold` 又很小，就会被误判为启动失败。正确做法不是盲目加大实例数，而是先放宽探针和超时，再看真实瓶颈。否则你只是在扩大一个错误配置的成本。

第四个权衡是公网访问与可运维性。

公网入口更方便联调，但生产环境攻击面会显著扩大。对外开放意味着你要额外承担凭据泄露、误调用、来源控制和网络边界治理的问题。能走私网就优先走私网。

常见坑可以总结成下面这张表：

| 坑点 | 表现 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| `instance_count` 过小 | `429`、排队、P99 抬升 | 峰值并发超过容量上限 | 按峰值并发反推实例数，并留余量 |
| `max_concurrent_requests_per_instance` 过高 | 平均延迟和 P99 变差 | 单实例真实可并发能力不足 | 用压测找到模型真实并发上限 |
| `startup/readiness/liveness probe` 过激进 | 部署反复不 Ready、重启、误判失败 | 镜像拉取或模型加载比探针节奏慢 | 拉长 `initial_delay`，合理增大 `failure_threshold` |
| 仍按 `maxQueueWait` 旧思路配置 | 排队和超时判断混乱 | 该字段已 deprecated | 优先以 `request_timeout_ms` 为主调优 |
| 公网未关闭 | 暴露面过大，安全审计压力高 | `public_network_access` 仍为 `enabled` | 生产改为私网访问并配 `aad_token` |

---

## 替代方案与适用边界

如果你的目标是“尽快、稳定、少运维地把模型变成 HTTP 服务”，`managed online endpoint` 通常是第一选择。它适合大多数团队，因为基础设施管理、补丁、主机恢复、监控集成都由平台接手。

如果团队已经有成熟 Kubernetes 平台，并且需要更强的自定义控制，比如统一 sidecar、特定调度策略、复杂服务网格或自定义运维链路，那么 `Kubernetes online endpoint` 更合适。但代价也直接：你要自己承担更多平台复杂度。

| 方案 | 更适合什么场景 | 代价 |
| --- | --- | --- |
| `managed online endpoint` | 大多数托管推理、灰度发布、快速上线 | 自定义空间较少，但运维简单 |
| `Kubernetes online endpoint` | 已有 K8s 平台、需要统一平台治理 | 基础设施和运维复杂度更高 |

流量切换本身也有两条不同路径，很多新手会混淆：

| 方式 | 作用 | 适用边界 |
| --- | --- | --- |
| `traffic` 分流 | 真实生产流量按比例打到多个 deployment | 做灰度、A/B、逐步切流 |
| `azureml-model-deployment` 直达 | 请求直接命中指定 deployment | 验证新版本，不影响线上默认权重 |

所以：

- 如果你只是想验证 `green` 是否可用，直接用 `azureml-model-deployment: green` 发送请求，不必先改线上权重。
- 如果你要做真实灰度，就改 `traffic` 为 `blue=90, green=10`，然后观察 `P99`、`429`、`5xx` 和探针日志。

网络接入策略也有明确边界：

| 接入方式 | 优点 | 适用边界 |
| --- | --- | --- |
| `public endpoint` | 联调简单、接入快 | 开发测试或明确受控的公开调用 |
| `private endpoint + managed VNet` | 网络边界强，攻击面小 | 生产环境、内网系统调用 |

最终判断标准很简单：

- 你更看重少运维和快速上线，用 `managed online endpoint`。
- 你更看重平台统一控制且能自管集群，用 `Kubernetes online endpoint`。
- 你要验证新版本，用 deployment 直达。
- 你要让真实用户逐步接入新版本，用 `traffic` 分流。
- 你要收缩安全边界，用 `aad_token + private endpoint`。

---

## 参考资料

1. [Online endpoints for real-time inference](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2)
2. [CLI (v2) managed online deployment YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online?view=azureml-api-2)
3. [Online endpoints YAML reference](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-endpoint-online?view=azureml-api-2)
4. [Safe rollout for online endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-safely-rollout-online-endpoints?view=azureml-api-2)
5. [Troubleshoot online endpoint deployment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints?view=azureml-api-2)
6. [Authenticate Clients for Online Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-online-endpoint?view=azureml-api-2)
7. [Secure managed online endpoints by using network isolation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-online-endpoint?view=azureml-api-2)
