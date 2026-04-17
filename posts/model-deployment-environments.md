## 核心结论

模型部署环境不是“云一定先进”或“本地一定便宜”的二选一题，而是一个按工作负载特征分配资源的问题。工作负载，白话说，就是模型在真实业务里到底怎么被调用：是偶尔跑几次，还是 24 小时持续跑。

云服务更适合三类场景：第一，需求波动大，今天 100 次请求、明天 10 万次请求；第二，团队还在试错，模型和产品方向都没稳定；第三，业务需要快速进入多个地区，直接借助 AWS、Azure、GCP 的现成区域能力。它的核心价值是弹性，意思是资源可以按需扩缩，不必先把机器买回来。

本地部署更适合另外三类场景：第一，模型推理负载长期稳定且利用率高；第二，数据敏感，不能轻易离开企业私网；第三，延迟要求极低，比如工厂边缘设备、金融风控前置节点、医院内网影像分析。本地部署的本质优势是可控，意思是硬件、网络、数据路径都由自己掌握。

混合部署通常是更现实的答案。敏感数据和低延迟链路留在本地，一般性推理、离线训练、流量突发时的扩容放到云上。很多团队最后不是“选一种”，而是先上云，再把稳定核心流量逐步迁回本地，把波峰流量和实验流量继续留在云端。

| 维度 | 云服务 | 本地部署 |
|---|---|---|
| 成本结构 | 以 OpEx 为主，按量付费 | 以 CapEx 为主，前期投入高 |
| 扩容速度 | 分钟级甚至自动扩容 | 采购、上架、调试通常按周或月 |
| 数据控制 | 受限于云平台边界与策略 | 完全自主控制 |
| 合规处理 | 依赖云厂商认证与区域能力 | 更容易做强隔离与定制审计 |
| 延迟 | 受公网/专线和区域影响 | 可做到局域网或边缘近端低延迟 |
| 运维负担 | 平台代管较多 | 企业自担机房、监控、冷却、维修 |
| 适合负载 | 波动型、试验型、全球型 | 稳定型、敏感型、低延迟型 |

---

## 问题定义与边界

这里先把概念分清。

云服务，指计算资源由第三方提供商通过互联网交付，常见是公有云的多租户环境。多租户，白话说，就是很多客户共享同一套底层物理基础设施，但逻辑上彼此隔离。

本地部署，指模型运行在企业自有机房、私有数据中心，或者更靠近现场的边缘设备上。边缘设备，白话说，就是把计算放到离数据源更近的地方，比如工厂网关、门店服务器、医院影像工作站。

真正影响决策的边界主要有四个。

| 边界维度 | 应该问的问题 | 更偏向云 | 更偏向本地 |
|---|---|---|---|
| 数据敏感度 | 数据能否离开企业控制域 | 可以脱敏、可跨域 | 不能出域、强主权要求 |
| 延迟要求 | 请求必须多快返回 | 100ms 以上可接受 | 20ms 甚至更低 |
| 成本模型 | 预算更适合月付还是一次投入 | 现金流敏感、先试再说 | 规模稳定、可摊销硬件 |
| 运维能力 | 团队能否管硬件和机房 | 没有专门 infra 团队 | 有 SRE/运维/机房能力 |

玩具例子：一个 3 人团队做简历分类助手，每天只有工作时间有请求，晚上几乎没人用。这类负载不稳定，模型也可能两周一换，直接上云最合理。原因不是“云更高级”，而是买本地 GPU 大概率会长期闲置。

真实工程例子：银行做 KYC，也就是“了解你的客户”的身份核验流程。身份证、护照、住址证明这类材料通常涉及敏感个人信息，核心 OCR 与审核推理更适合放在私有网络；而营销推荐、非敏感知识问答、国际多语言客服可以放到公有云。这里的边界不是模型大小，而是数据边界和审计边界。

所以，这篇文章讨论的不是“训练一定在哪”，而是更常见的“模型推理和配套服务应该部署在哪”。因为大多数团队真正持续付费的部分，往往不是训练，而是上线后的长期推理。

---

## 核心机制与推导

先用一个最小可计算模型建立判断框架。

设：
- $U$ 是资源利用率，表示机器实际被使用的时间占总时间的比例。
- $R_{\text{cloud}}$ 是云端单位时间成本，比如每小时 GPU 实例费用。
- $CapEx$ 是本地一次性投入，比如服务器、GPU、交换机。
- $OpEx$ 是本地持续运营成本，比如电费、冷却、机房、运维。
- $T$ 是摊销周期，比如按 3 年或 5 年计算总小时数。

则云端总成本可近似写成：

$$
C_{\text{cloud}} = U \times T \times R_{\text{cloud}}
$$

本地总成本可近似写成：

$$
C_{\text{onprem}} = CapEx + OpEx
$$

若换成单位时间视角，本地的等效小时成本是：

$$
R_{\text{onprem}} = \frac{CapEx + OpEx}{T}
$$

当且仅当：

$$
U \times R_{\text{cloud}} > R_{\text{onprem}}
$$

也就是

$$
U > \frac{CapEx + OpEx}{R_{\text{cloud}} \times T}
$$

时，本地部署在纯成本上开始更有优势。

这个式子表达的结论很直接：是否值得本地化，关键不在“机器贵不贵”，而在“你是否真的把它持续用起来”。如果机器大部分时间空着，本地投入很难摊平；如果机器长期高占用，本地成本会越来越像固定成本，反而更可控。

玩具例子：假设云端某 GPU 成本是每小时 \$40，本地 3 年总成本折算后相当于每小时 \$24，那么阈值就是：

$$
U > \frac{24}{40} = 0.6
$$

也就是利用率高于 60% 时，本地开始有成本优势。若每天大约运行 18 小时，则利用率约为：

$$
U = \frac{18}{24} = 0.75
$$

这时就应认真评估本地部署。

真实工程例子：某 AI 团队起步时先用云上 GPU 实例部署实验模型，因为一开始请求量不稳定、版本频繁变化，云端的试错成本最低。半年后，他们发现核心 API 的白天流量稳定、夜间批处理也固定，平均利用率接近 70%。这时把核心推理迁到本地 8 卡机架，剩余突发流量继续留云，就形成了典型的混合部署。

但成本公式只解决“钱”的问题，不解决“能不能做”的问题。真实决策还要加上两个约束：

第一，合规约束。如果数据不能出域，哪怕云更便宜，也不能直接放公有云。  
第二，延迟约束。如果请求链路必须低于某阈值，例如 $20ms$，而公网和跨地域链路无法稳定满足，那么本地或边缘就是硬约束。

因此一个更工程化的决策顺序通常是：

1. 先看合规和数据主权，不能出域则优先本地或主权云。
2. 再看延迟，低延迟刚需则优先本地或边缘。
3. 再看利用率和成本阈值，高利用率再评估本地摊销。
4. 最后看团队运维能力，没有能力维护机房时，不要为了“理论便宜”强上本地。

---

## 代码实现

下面给一个可运行的简化决策器。它不替代正式容量规划，但足够做一轮初筛。

```python
from dataclasses import dataclass

@dataclass
class DeploymentInput:
    monthly_hours: float
    cloud_hourly_cost: float
    onprem_total_cost: float
    amortized_months: int
    sensitive_data: bool
    target_latency_ms: float
    team_can_run_datacenter: bool

def choose_deployment(x: DeploymentInput) -> str:
    if x.sensitive_data or x.target_latency_ms < 20:
        return "on-prem"

    total_hours = 24 * 30 * x.amortized_months
    utilization = x.monthly_hours / (24 * 30)
    onprem_hourly_equivalent = x.onprem_total_cost / total_hours

    if (
        utilization >= 0.65
        and onprem_hourly_equivalent < x.cloud_hourly_cost
        and x.team_can_run_datacenter
    ):
        return "on-prem"

    if 0.35 <= utilization < 0.65 and x.team_can_run_datacenter:
        return "hybrid"

    return "cloud"


toy = DeploymentInput(
    monthly_hours=180,          # 每月只跑 180 小时，明显不稳定
    cloud_hourly_cost=40,
    onprem_total_cost=600000,
    amortized_months=36,
    sensitive_data=False,
    target_latency_ms=120,
    team_can_run_datacenter=False,
)
assert choose_deployment(toy) == "cloud"

stable_core = DeploymentInput(
    monthly_hours=600,          # 约等于每天 20 小时
    cloud_hourly_cost=40,
    onprem_total_cost=600000,
    amortized_months=36,
    sensitive_data=False,
    target_latency_ms=50,
    team_can_run_datacenter=True,
)
assert choose_deployment(stable_core) in {"on-prem", "hybrid"}

sensitive_case = DeploymentInput(
    monthly_hours=100,
    cloud_hourly_cost=40,
    onprem_total_cost=800000,
    amortized_months=36,
    sensitive_data=True,
    target_latency_ms=80,
    team_can_run_datacenter=True,
)
assert choose_deployment(sensitive_case) == "on-prem"

print("all assertions passed")
```

这个实现故意保持简单，但已经体现了三个核心规则：

1. 合规和低延迟优先级高于成本。
2. 高利用率才值得考虑本地摊销。
3. 没有运维能力时，不要把本地当成默认答案。

在真实工程里，通常还会继续加入这些参数：
- 云端出口流量费，也就是数据从云里传出来的费用。
- 预留实例或 Spot 折扣，也就是云上不同购买方式的价格差。
- 本地冗余系数，比如必须多买 20% 容量应对故障。
- 峰谷流量比，用来判断混合部署是否优于纯本地。
- 数据驻留要求，比如某些数据只能在指定国家或私有区域处理。

---

## 工程权衡与常见坑

很多团队不是败在“选错方向”，而是败在“漏算约束”。

| 风险类型 | 常见表现 | 影响 | 缓解措施 |
|---|---|---|---|
| 云端费用失控 | 忽略出口流量费、闲置实例、跨区传输 | 账单持续超预算 | 做 FinOps 报表，定期清理闲置资源 |
| 厂商锁定 | 过度依赖单一云的专有服务 | 迁移成本极高 | 核心链路尽量接口标准化 |
| 本地利用率不足 | 机器买回后长期空转 | 折旧无法摊平 | 先用云验证，再按稳定流量回迁 |
| 本地运维能力不足 | 监控、冷却、备件、升级跟不上 | 故障恢复慢，服务不稳定 | 没有团队就不要上重资产 |
| 混合治理失控 | 云上和本地权限、日志、审计分裂 | 安全盲区，排障困难 | 统一 IAM、统一监控、统一日志 |

一个常见误区是只比较“算力单价”。这是不够的。云端要把实例费、存储费、网络费、日志费、跨区费一起算；本地要把采购、折旧、电力、冷却、机柜、备件、人力一起算。只拿“每小时 GPU 单价”对比，很容易得出错误结论。

第二个常见误区是把“数据隐私”理解成“全部本地”。其实很多系统可以分层。比如用户原始文档留本地，脱敏后的结构化字段上云跑推荐；或者本地做首轮推理，云端做补充分析与批处理。真正重要的是数据流设计，而不是口号式地说“我们不上云”。

第三个常见误区是忽略迁移成本。云转本地、本地转云都不是复制文件那么简单。你要迁的是镜像、模型权重、缓存策略、监控体系、容灾流程、身份认证和网络策略。技术上能迁，不代表组织上能快速迁。

---

## 替代方案与适用边界

除了“纯云”和“纯本地”，至少还有三种常见替代方案。

第一种是混合云。核心思路是把稳定且敏感的流量留在私有环境，把突发和通用能力放到公有云。它适合金融、医疗、政务这类既要合规又要弹性的场景，也是最常见的企业路径。

第二种是主权云或行业专属云。主权云，白话说，就是在特定国家、地区或监管框架下运行的云环境，重点解决数据驻留和监管审计问题。它比纯公有云更合规，比完全自建更省事，但通常价格更高、可用服务更少。

第三种是边缘部署加中心云。边缘负责近端实时推理，中心云负责模型更新、离线训练、日志归档和全局编排。它适合门店、工厂、车载、医院设备等场景。

可以把选择逻辑压缩成一棵决策树：

1. 数据是否允许离开本地控制域？
2. 如果不允许，优先本地或主权云。
3. 如果允许，再看延迟是否必须低于 20ms。
4. 若必须低延迟，优先边缘或本地。
5. 若延迟不是硬约束，再看利用率是否长期高于 60% 到 70%。
6. 若高利用率且团队有运维能力，评估本地或混合。
7. 若负载波动大、团队小、业务还在试错，优先云。

对零基础到初级工程师来说，最实用的落地建议只有一句：先按“约束优先、成本第二”的顺序决策。合规和延迟是硬边界，成本优化建立在边界允许的前提上。很多成熟团队的路径都类似：先用云把业务跑起来，再把稳定核心部分回迁到本地，最后形成混合部署。

---

## 参考资料

1. NanoGPT, *Cloud vs On-Prem: AI Deployment Cost Breakdown*, 2025: https://nano-gpt.com/blog/cloud-vs-on-prem-ai-deployment-cost-breakdown  
2. Microsoft Azure, *What are public, private, and hybrid clouds?*: https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-are-private-public-hybrid-clouds/  
3. DigitalOcean, *Comparing AWS, Azure, and GCP for Startups in 2026*: https://www.digitalocean.com/resources/articles/comparing-aws-azure-gcp  
4. Hypersense, *Cloud vs. On-Premise Infrastructure: Comparison Guide for IT Leaders*, 2025: https://hypersense-software.com/blog/2025/07/31/cloud-vs-on-premise-infrastructure-guide/
