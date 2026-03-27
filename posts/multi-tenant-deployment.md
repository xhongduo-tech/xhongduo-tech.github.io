## 核心结论

多租户部署（multi-tenant deployment，白话说：一套系统同时服务多个客户）不是“大家共用一台机器”这么简单，它的本质是：**共享基础设施，隔离数据、权限、网络与资源**。这样做的直接收益有三点：

1. 同一套代码、同一套运维流程可以服务多个租户，发布和回滚更集中。
2. 机器、存储、网络可以复用，平均成本更低。
3. 新增租户不需要再复制一整套环境，扩容速度更快。

但它成立的前提也很明确：**共享必须建立在可验证的隔离之上**。如果没有租户级鉴权、资源配额、限流、审计，多租户就会退化成“多人共用同一个故障域”。

可以先用一个表把几种常见部署模型对齐：

| 模型 | 共享程度 | 隔离强度 | 成本 | 个性化能力 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| 完全共享 | 高 | 低到中 | 最低 | 较弱 | 租户多、单租户体量小 |
| 完全独立 | 低 | 最高 | 最高 | 最强 | 强监管、大客户、定制化强 |
| 混合隔离 | 中 | 中到高 | 中 | 中到强 | 主流 SaaS、分层服务 |
| 分层租户 | 核心层共享，关键层独立 | 可控 | 中到高 | 强 | 普通客户共享，重点客户独享 |

一个面向初级工程师的判断标准是：**多租户不是为了“省机器”，而是为了在可控风险内提高复用率**。如果隔离做不好，后续付出的事故成本会高于节省的资源成本。

---

## 问题定义与边界

问题定义可以写成一句话：**在同一套计算、存储和网络底座上，为多个租户提供逻辑独立的运行环境。**

这里的“逻辑独立”不是抽象概念，而是可以拆成四个边界：

| 维度 | 共享内容 | 必须隔离的内容 | 常见实现 |
| --- | --- | --- | --- |
| 计算 | 节点、容器运行时 | CPU、内存、GPU、进程权限 | `requests/limits`、cgroup、容器隔离 |
| 网络 | 物理网络、CNI | 跨租户访问路径 | namespace、NetworkPolicy、VLAN |
| 数据 | 数据库实例、存储集群 | 表、行、对象、密钥 | schema、RLS、独立库、KMS |
| 权限 | 身份系统、网关 | token、角色、租户上下文 | tenant-scoped auth、RBAC、审计 |

边界之外的内容也要说明清楚，否则设计会失焦。

第一，**多租户不等于多环境**。开发、测试、生产是环境隔离；租户 A、租户 B 是业务隔离，这两者不是一回事。

第二，**多租户不等于所有层都共享**。数据库可以独立，计算层可以共享；网络可以共享底座，但策略必须隔离。真正的工程设计往往是“部分共享，部分独立”。

第三，**多租户的核心风险不是功能错误，而是边界穿透**。一个接口功能正常返回，并不代表它安全；如果没有带上租户条件，返回了别家数据，这就是典型的多租户故障。

玩具例子可以帮助建立直觉。假设有一个在线记账应用，服务两个小团队：`team-a` 和 `team-b`。它们都访问同一个 Web 服务和同一个数据库实例。此时最少要满足三件事：

1. 登录后每个请求都带租户标识。
2. 查询账单时必须附带 `tenant_id` 条件。
3. 一个团队的批量导出任务不能把另一个团队的接口拖慢到超时。

这就是问题边界的最小闭环：**身份边界、数据边界、资源边界**。

---

## 核心机制与推导

多租户系统能稳定运行，靠的不是某一个组件，而是一组联动机制。

先看资源管理。资源配额（resource quota，白话说：先把每户最多能占多少资源写死）常用来限制一个租户在 CPU、内存、对象数上的总占用。最基本的约束可以写成：

$$
\sum_{i=1}^{n} R_i \le C_{total}
$$

其中，$R_i$ 是第 $i$ 个租户的保底资源请求，$C_{total}$ 是集群可分配总容量。

如果进一步细分 CPU 和内存：

$$
\sum_{i=1}^{n} CPU^{request}_i \le CPU_{cluster}, \quad
\sum_{i=1}^{n} MEM^{request}_i \le MEM_{cluster}
$$

这个公式的意义很直接：**所有租户的保底承诺之和，不能超过系统真实能兑现的容量**。否则系统只是“看起来分配了资源”，一到高峰就会同时违约。

玩具例子：集群总共有 100 核 CPU，计划接入 10 个租户。每个租户的 `requests.cpu = 8`，那么总承诺是：

$$
10 \times 8 = 80 \le 100
$$

这意味着还留了 20 核缓冲，系统可以承受短时间波动。如果你把每个租户都设成 12 核保底，那么总承诺变成 120，公式已经不成立，后续再做限流、优先级都只是补救。

再看隔离机制，通常分四层：

1. 网络隔离  
网络隔离（白话说：让租户之间默认不能互相访问）控制“谁能连到谁”。Kubernetes 中常见做法是 `namespace + NetworkPolicy`。这样一个租户的 Pod 默认无法直接访问别的租户服务。

2. 容器隔离  
容器隔离（白话说：每个租户任务跑在彼此分开的运行空间里）依赖 namespace、cgroup、capability、seccomp 等机制，避免进程、文件系统、内核能力互相污染。

3. 资源隔离  
资源隔离（白话说：先给每户划预算，再限制超用）依赖 `requests/limits`、`ResourceQuota`、`LimitRange`、优先级和限流。它解决的是 noisy neighbor，即“嘈杂邻居”问题，某个租户突然高负载，不该让其他租户一起抖动。

4. 数据隔离  
数据隔离（白话说：即使共用数据库，也只能读写自己的那部分）可以做成独立库、独立 schema、共享表加 `tenant_id`，或者数据库行级安全策略（RLS）。这一层最容易因为代码疏漏而出事故。

真实工程例子：一个面向企业的 AI 工作流平台，多个公司共用同一个推理网关和任务调度器。某租户在晚高峰批量触发文档解析与 embedding 任务，瞬间打满 CPU 和对象存储带宽。如果没有租户级并发上限和队列优先级，另一个租户的在线问答请求就会超时。这个问题不是“模型慢”，而是**多租户资源治理失效**。

可以把推导关系总结成一句话：**隔离决定边界是否安全，配额决定边界是否稳定，限流与优先级决定边界在高峰时是否还能成立。**

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，用来演示“租户配额检查 + 限流判断”的基本逻辑。它不是生产代码，但足够把规则讲清楚。

```python
from dataclasses import dataclass

@dataclass
class TenantQuota:
    tenant: str
    cpu_request: int
    mem_request_gi: int
    qps_limit: int

def can_admit_cluster(tenants, total_cpu, total_mem_gi):
    used_cpu = sum(t.cpu_request for t in tenants)
    used_mem = sum(t.mem_request_gi for t in tenants)
    return used_cpu <= total_cpu and used_mem <= total_mem_gi

def allow_request(current_qps, quota: TenantQuota):
    return current_qps <= quota.qps_limit

tenants = [
    TenantQuota("acme", cpu_request=8, mem_request_gi=16, qps_limit=200),
    TenantQuota("globex", cpu_request=8, mem_request_gi=16, qps_limit=150),
    TenantQuota("initech", cpu_request=4, mem_request_gi=8, qps_limit=80),
]

assert can_admit_cluster(tenants, total_cpu=32, total_mem_gi=64) is True
assert can_admit_cluster(tenants, total_cpu=16, total_mem_gi=64) is False

assert allow_request(120, tenants[0]) is True
assert allow_request(220, tenants[0]) is False

print("quota checks passed")
```

这个例子表达了两个工程事实：

1. 集群接纳新租户前，先算总承诺是否超出底座容量。
2. 单个租户即使合法登录，也不代表它可以无限占用吞吐。

落到 Kubernetes，最常见的做法是按租户划分 `namespace`，再给 namespace 配置 `ResourceQuota` 和 `LimitRange`。

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-quota
  namespace: tenant-acme
spec:
  hard:
    requests.cpu: "8"
    limits.cpu: "12"
    requests.memory: "16Gi"
    limits.memory: "24Gi"
    pods: "40"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-default-limits
  namespace: tenant-acme
spec:
  limits:
    - type: Container
      defaultRequest:
        cpu: "250m"
        memory: "256Mi"
      default:
        cpu: "1"
        memory: "1Gi"
```

`ResourceQuota` 管的是租户总盘子，`LimitRange` 管的是单个容器默认上下限。两者配合，才能避免“有人忘了写 limits，结果一个 Pod 抢光全租户预算”。

网络层还需要补上最小访问策略：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-cross-tenant
  namespace: tenant-acme
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
```

这类策略的含义是先默认收紧，再按需放行。对初级工程师来说，一个重要认知是：**多租户的默认值应该是拒绝，不应该是放开后再补漏洞。**

真实工程例子可以这样落地。假设你在做一个“文档上传 + 向量检索 + 大模型问答”的 SaaS：

- `tenant-a` 是试用客户，只给 2 核 CPU、4Gi 内存、QPS 20。
- `tenant-b` 是付费企业客户，给 16 核 CPU、32Gi 内存、QPS 200。
- 离线建索引任务走低优先级队列，在线问答走高优先级队列。
- 上传文件存储在共享对象存储里，但路径前缀强制带 `tenant_id/`。
- 检索服务查询向量库时必须附带租户过滤条件。

这样同一套系统可以服务不同等级客户，但系统行为仍然可预测。

---

## 工程权衡与常见坑

多租户部署最常见的误区，不是“不会做”，而是“只做了一半”。

| 问题 | 常见原因 | 直接后果 | 缓解方式 |
| --- | --- | --- | --- |
| 越权读写 | 查询漏了 `tenant_id`，或鉴权不带租户上下文 | 数据串租户 | 强制 tenant-scoped auth、RLS、审计 |
| 性能干扰 | 没有 `requests/limits`、没有队列隔离 | noisy neighbor | 配额、限流、优先级队列 |
| 成本失控 | 只看总资源，不看租户账单和峰值 | 资源浪费、毛利下降 | 租户级观测和计费 |
| 排障困难 | 指标没有按租户打标签 | 发现问题慢 | metrics/logs/traces 加租户维度 |
| 规则漂移 | 手工配置过多，环境不一致 | 线上行为不可预测 | 模板化、策略即代码 |

先看最严重的一类：**数据隔离失效**。共享数据库时，很多团队会说“应用层已经判断过权限”，但这不够。因为一次漏条件的 SQL、一次缓存 key 设计错误、一次后台导出脚本失误，都可能越过应用层假设。对于高风险数据，数据库层最好仍有兜底策略，例如 RLS 或独立 schema。

第二类是**性能干扰**。比如金融租户在月末跑批，大量触发报表查询；同时医疗租户在线 API 正在处理实时请求。如果没有并发限制、读写隔离、队列优先级，最终用户只会看到“系统不稳定”，不会关心是哪个邻居把资源吃满了。

第三类是**观测缺失**。可观测性（observability，白话说：让系统内部状态能被持续看见）在多租户里不是锦上添花，而是计费、限流、故障定位的基础。如果 CPU、延迟、错误率、存储增长都没有租户维度，你根本不知道谁在制造峰值，也无法证明某个隔离策略是否有效。

还有几个常见坑值得单独记住：

1. 只配 `limits`，不配 `requests`。  
结果是调度器无法做稳定承诺，节点容易过度装箱。

2. 只做 namespace，不做 NetworkPolicy。  
结果是“看起来分开了”，但网络上仍然可能横向访问。

3. 缓存 key 不带租户前缀。  
结果是数据直接串租户，这类问题在 Redis 和 CDN 层都很常见。

4. 日志、对象存储路径、消息队列 topic 不带租户维度。  
结果是审计困难，回溯成本高。

工程上真正可行的原则是：**租户标识要从入口一路传到存储、缓存、消息、日志、监控和计费。** 只在入口层做一次判断，远远不够。

---

## 替代方案与适用边界

多租户不是唯一答案。更准确地说，多租户是一组架构谱系，不是单一模式。

| 方案 | 适配场景 | 主要优点 | 主要代价 |
| --- | --- | --- | --- |
| 独立数据库 | 租户少、监管强、数据敏感 | 隔离最直观 | 成本高、运维多 |
| 独立 Schema | 中等规模 B2B | 隔离与成本较平衡 | 迁移和治理复杂 |
| 共享表 + `tenant_id` | 大量小租户 | 成本最低、扩展快 | 越权风险最高 |
| 混合模式 | 客户层次差异大 | 可分层运营 | 架构复杂度更高 |
| 独立集群/VM | 超大客户、强 SLA | 故障域最小 | 成本和交付复杂度最高 |

如何选型，可以按三个问题判断：

1. 单租户价值是否足够高？  
如果单个客户合同金额很高，独立环境的额外成本可能是合理的。

2. 合规要求是否明确要求物理或实例级隔离？  
如果答案是是，那么共享数据库方案通常就不合适。

3. 租户数量和规模分布是否长尾？  
如果有大量小客户和少量大客户，混合模式通常更实际。

一个简单的经验法则是：

- 小租户多、标准化高：优先共享架构。
- 中型企业客户为主：优先 schema 或混合模式。
- 金融、医疗、政务等高监管：优先更强隔离，必要时独立集群。

所以“最佳实践”不是一开始就把所有租户做成完全独立，也不是所有租户都塞进同一个共享表。更合理的路径通常是：**先把租户边界模型设计对，再根据客户等级把隔离强度做成可升级的。**

---

## 参考资料

- Qrvey, *Multi-Tenant Deployment: 2026 Complete Guide & Examples*  
  https://qrvey.com/blog/multi-tenant-deployment/

- Kubernetes Documentation, *Resource Quotas*  
  https://kubernetes.io/docs/concepts/policy/resource-quotas/

- AddWeb Solution, *The Multi-Tenant Performance Crisis: Advanced Isolation Strategies for 2026*  
  https://www.addwebsolution.com/blog/multi-tenant-performance-crisis-advanced-isolation-2026

- Zylos, *Multi-Tenant Security Patterns for SaaS and AI Agent Platforms*  
  https://zylos.ai/research/2026-02-23-multi-tenant-security-patterns-saas-platforms
