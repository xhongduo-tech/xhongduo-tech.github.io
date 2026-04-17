## 核心结论

Harbor 是一个面向企业内网或多集群场景的私有容器镜像仓库。容器镜像仓库可以直白理解为“集中存放 Docker 镜像的服务器”。它不是单个进程，而是一组协同组件：`Portal` 负责网页界面，`Core` 负责权限、策略和 API，`Registry` 负责真正保存镜像内容，`Jobservice` 负责异步任务，`Trivy` 负责漏洞扫描。

如果只记三个结论，可以记这三个：

1. Harbor 的核心价值不是“能存镜像”，而是“能按策略管理镜像”。
2. 企业里真正有用的闭环是：`推送即扫 -> 不合格阻断 -> 合格复制到目标环境`。
3. HTTPS 不是可选优化，而是最低安全前提。没有 HTTPS，私有仓库在传输链路上就不可信。

一个新手能感知到的运行路径是这样的：浏览器打开 Harbor 页面后，`Portal` 把操作请求交给 `Core`；`Core` 调用 `Registry` 查询或写入镜像；如果需要扫描、复制、垃圾回收，`Core` 会把任务交给 `Jobservice` 排队执行；`Trivy` 扫描后把漏洞结果写回 Harbor，项目策略再决定是否允许该镜像继续被拉取或复制。

下表是 Harbor 常见组件与职责：

| 组件 | 直白解释 | 主要职责 | 是否直接存镜像 |
|---|---|---|---|
| Portal | 网页前端，给人看的控制台 | 展示项目、镜像、扫描结果、规则配置 | 否 |
| Core | 中控层，负责“做决定” | 用户认证、权限、API、策略、事件协调 | 否 |
| Registry | 镜像仓库本体 | 保存 OCI 镜像层、Manifest、标签 | 是 |
| Jobservice | 后台任务队列 | 复制、扫描、GC、Webhook 等异步任务 | 否 |
| Trivy | 漏洞扫描器 | 分析镜像中软件包和 CVE 漏洞 | 否 |
| ChartMuseum | Helm Chart 仓库 | 管理 Helm Chart 包 | 否 |

漏洞门控可以抽象成一个简单规则。设严重级别阈值为 $S$，则允许条件可以写成：

$$
severity < S
$$

如果项目设置 `S = High`，那么 `High` 和 `Critical` 级别漏洞的镜像都会被阻止拉取或部署。这就是“策略门控”的本质：不是只看报告，而是把报告变成行为约束。

---

## 问题定义与边界

本文讨论的问题，不是“如何搭一个能跑的 Docker Registry”，而是“如何搭一个符合企业镜像治理要求的 Harbor 私有仓库”。这里的“治理”可以白话理解为：谁能上传、哪些镜像合格、哪些镜像要同步到别的环境，都有明确规则。

目标边界如下：

| 范围 | 本文覆盖 | 说明 |
|---|---|---|
| Harbor 基础架构 | 是 | Core、Portal、Registry、Jobservice、Trivy |
| HTTPS 配置 | 是 | 自签名证书场景 |
| 镜像复制规则 | 是 | 事件驱动、过滤、路径改写 |
| 漏洞扫描与阻断 | 是 | Trivy、自动扫描、阈值门控 |
| 自定义插件开发 | 否 | 不讨论 Harbor 二次开发 |
| 第三方对象存储 | 否 | 不展开 S3、OSS、MinIO 后端细节 |
| 备份与灾备 | 否 | 不讨论数据库/镜像全量备份方案 |
| Kubernetes 编排部署 | 否 | 不展开 Helm 安装或高可用集群 |

文字版流程图可以写成：

`用户/CI 推送镜像 -> Portal/Core 接收请求 -> Registry 存镜像 -> Core 触发事件 -> Jobservice 排队扫描/复制 -> Trivy 返回漏洞结果 -> Harbor 根据策略允许或阻止后续拉取`

对新人来说，核心问题可以压缩成三件事：

1. `Registry` 怎样安全、持久地保存镜像。
2. `Jobservice` 怎样稳定地处理扫描和复制等后台任务。
3. `Trivy + 策略` 怎样把“发现漏洞”变成“阻止高危镜像进入生产”。

这意味着本文不关注容器编排、Service 暴露、Ingress、LB、跨机房网络等外部基础设施。那些内容会影响部署形态，但不会改变 Harbor 的机制本身。

---

## 核心机制与推导

Harbor 的镜像治理有两个关键机制：复制规则和漏洞门控。

先看复制规则。复制规则就是“什么镜像、在什么时机、复制到哪里、复制后名字怎么变”的配置。它常见的判断可以抽象为：

$$
match(pattern, artifact) \rightarrow \{true,false\}
$$

这里的 `artifact` 可以白话理解为“一个待处理对象”，比如某个镜像仓库中的 `project-a/backend:v1.2.0`。`pattern` 是匹配规则，比如 `project-a/**`。当 `match` 为真，Harbor 就会把这个对象纳入复制范围。

路径改写则常写成：

$$
flatten_n(image)
$$

它的意思是“丢掉镜像路径前面的 n 层目录”。例如镜像名为：

`project-a/dev/backend:v1`

如果使用 `flatten_1`，可理解为去掉第一层前缀，结果变成：

`dev/backend:v1`

如果使用 `flatten_2`，结果可能进一步变成：

`backend:v1`

这类改写在“中心仓库路径深、边缘仓库路径浅”的场景里很常见。

常用匹配符号如下：

| 通配符 | 含义 | 例子 | 效果 |
|---|---|---|---|
| `*` | 匹配单层任意字符 | `backend-*` | 匹配 `backend-api`，不跨 `/` |
| `**` | 匹配多层路径 | `project-a/**` | 匹配 `project-a/x/y:z` |
| `?` | 匹配单个字符 | `v1.?` | 匹配 `v1.2`，不匹配 `v1.23` |
| `{a,b}` | 多选一 | `{dev,prod}/**` | 匹配 `dev/...` 或 `prod/...` |

玩具例子可以这样看。假设你只有两个镜像：

- `project-a/nginx:v1`
- `project-b/redis:v1`

复制规则写成“匹配 `project-a/**`，事件驱动，目标为边缘仓库，`flatten_1`”。那结果只有第一个镜像会被复制，而且在目标侧可能变成 `nginx:v1`。对零基础读者，这可以简单理解为：“名字里带 `project-a/` 的镜像才搬走，搬过去时把公共前缀删掉。”

再看漏洞门控。扫描不是目的，决策才是目的。Harbor 用 Trivy 扫描镜像中的软件包和漏洞数据库，把结果映射为严重级别，例如 `Low`、`Medium`、`High`、`Critical`。项目级策略再定义阈值 $S$。允许条件仍然是：

$$
severity < S
$$

如果 $S = High$，那么任何 `High` 或 `Critical` 漏洞都会让镜像不满足策略。

这里可以用一个非常简化的逻辑表示：

```json
{
  "scanOnPush": true,
  "preventVulnerableImages": true,
  "severityThreshold": "High"
}
```

它的语义是：

- 推送后立即扫描。
- 如果扫描结果中存在 `High` 及以上漏洞，则阻止镜像继续被拉取。
- 没有达到阈值的镜像可以继续流转。

真实工程例子更接近下面这个场景：总部有一个中心 Harbor，所有业务镜像先推到这里。镜像上传后自动触发 Trivy 扫描。若结果包含高危漏洞，镜像只能留在“待处理”状态，不能进入生产复制链路。只有通过门控的镜像，才会被事件驱动地复制到华东、华北、海外等边缘仓库，供各地 Kubernetes 集群就近拉取。这样做有两个直接好处：

1. 各地不需要重复从公网拉基础镜像，速度更稳定。
2. 边缘环境默认只接触“已批准”的镜像版本，而不是任何上传成功的版本。

---

## 代码实现

Harbor 的大部分配置都在 UI 完成，但把它抽象成数据结构更容易理解。下面先给一个“复制规则 + 事件触发”的简化 JSON：

```json
{
  "name": "replicate-project-a",
  "trigger": "event",
  "enabled": true,
  "filters": [
    {
      "type": "name",
      "value": "project-a/**"
    }
  ],
  "destNamespace": "edge",
  "override": true,
  "speed": 10240,
  "chunkSize": 10485760,
  "flatten": 1
}
```

这个规则表达的是：

- 触发方式是事件驱动，不是手动点执行。
- 只复制名字匹配 `project-a/**` 的镜像。
- 带宽限制是 `10240 KB/s`。
- 分块大小是 `10 MB`。
- 路径拍平一层，减少目标侧层级。

把这个过程用伪代码写出来会更直观：

```python
def should_replicate(image_name: str, severity: str, threshold: str = "High") -> bool:
    order = ["None", "Low", "Medium", "High", "Critical"]
    severity_rank = order.index(severity)
    threshold_rank = order.index(threshold)

    name_ok = image_name.startswith("project-a/")
    vuln_ok = severity_rank < threshold_rank
    return name_ok and vuln_ok


assert should_replicate("project-a/backend:v1", "Medium") is True
assert should_replicate("project-a/backend:v1", "High") is False
assert should_replicate("project-b/redis:v1", "Low") is False
```

这段 Python 代码是可运行的，也准确表达了 Harbor 中最关键的决策链：先看名称是否匹配复制范围，再看漏洞是否低于门控阈值。真实 Harbor 的实现当然更复杂，但原理就是这两步组合。

如果按 UI 操作理解，一个最小可用配置可以分成这几步：

1. 创建项目，例如 `project-a`。
2. 在项目设置里打开自动扫描，即 `scan on push`。
3. 打开“Prevent vulnerable images from running”，并把阈值设为 `High`。
4. 配置目标注册表凭据。
5. 新建复制规则，触发模式选 `Event Based`。
6. 设置过滤条件为 `project-a/**`。
7. 设置带宽和 chunk 参数，避免大镜像复制对链路造成尖峰压力。

“推送即扫”和“手动扫描”的差别可以用表表示：

| 方式 | 触发时机 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| 推送即扫 | 镜像上传后立即执行 | 问题发现早，适合门控 | 上传延迟略增 | 生产仓库、准入严格环境 |
| 手动扫描 | 管理员或开发者主动触发 | 灵活，适合排查 | 容易漏扫 | 测试环境、历史镜像补扫 |

如果要给新手一句最实用的话，就是：生产项目默认应该启用“推送即扫 + 阻止高危镜像”，否则扫描结果只是一份报告，不会形成治理效果。

---

## 工程权衡与常见坑

第一类坑是 HTTPS。

Harbor 不是“能打开网页就算可用”。如果仓库以 HTTP 暴露，镜像上传、登录凭据、API 调用都存在被中间人篡改或窃听的风险。中间人可以白话理解为“你和服务器之间插了一个偷偷改包的人”。所以工程上更准确的说法不是“建议开 HTTPS”，而是“没有 HTTPS 就不应上线”。

自签名证书场景通常要处理两个位置：

| 位置 | 需要放什么 | 用途 |
|---|---|---|
| `/data/cert/` | `server.crt`、`server.key` | Harbor 服务端使用 |
| `/etc/docker/certs.d/<域名>/` | `ca.crt` 或相关证书文件 | Docker 客户端信任该 CA |

一个简化的 OpenSSL 生成命令如下：

```bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout harbor.key \
  -out harbor.crt \
  -subj "/CN=harbor.example.com"
```

放置完证书后，服务端要重载 Harbor，客户端要重启 Docker。很多“登录失败”或“x509 unknown authority”本质上都不是 Harbor 逻辑问题，而是证书没有被客户端信任。

第二类坑是“开了扫描，但没开阻断”。

这类配置最容易造成误判。运维同学看到界面里已经有 Trivy 结果，就以为高危镜像会自动被挡住。实际上如果没有打开 `Prevent vulnerable images from running`，Trivy 只是报告系统，不是门禁系统。也就是说，漏洞会被看见，但不会被阻止。

第三类坑是“事件驱动复制不处理删除”。

很多人以为源仓库删除标签后，目标仓库也会自动同步删除。通常不是这样。事件驱动复制更像“新增时推送出去”，而不是“双向状态镜像”。所以做版本治理时，要把删除、保留策略、GC 分开考虑。

第四类坑是复制参数配置过于理想化。

如果链路带宽有限，而你没有设置复制限速或 chunk 大小，大镜像复制可能冲击业务网络。`10240 KB/s` 和 `10 MB chunk` 不是固定标准，但它们是一个适合入门理解的最小控制点：带宽限制控制总吞吐，chunk 大小控制分片传输粒度。

第五类坑是把 Harbor 当作“只给开发用的镜像盘”。

真实工程里，Harbor 更像供应链入口。镜像如果要进入测试、预发、生产，最好都经过同一套扫描和复制策略。否则开发环境拉的是一个版本，生产环境手工传的是另一个版本，最后连“生产到底跑了什么镜像”都难以追溯。

---

## 替代方案与适用边界

不是所有场景都必须上 Harbor。判断标准很简单：你是否需要“策略化镜像治理”。

下表给一个实用对比：

| 方案 | 镜像存储 | UI | 漏洞扫描 | 复制规则 | 适合场景 |
|---|---|---|---|---|---|
| Harbor | 有 | 有 | 内置/集成 | 有 | 企业内网、多集群、多环境治理 |
| Docker Registry | 有 | 基本无 | 无 | 无 | 极简私有仓库、只求能推拉 |
| 云厂商镜像仓库 | 有 | 有 | 通常有 | 常常有 | 云上托管、少自运维 |
| 代理缓存方案 | 部分 | 视产品而定 | 通常弱 | 通常弱 | 加速公网拉取、非严格治理 |

如果你只是想在一台机器上存放几个内部镜像，纯 `Docker Registry` 更轻，维护成本也更低。它的问题不是不能用，而是缺少治理能力：没有统一 UI、没有扫描门控、没有灵活复制规则，很多工作只能靠人工补。

Harbor 适合的典型场景是：

- 公司有多个 Kubernetes 集群。
- 镜像必须先经过安全扫描。
- 不同机房或边缘节点需要本地拉取副本。
- 需要按项目、团队、角色做权限隔离。

可以把它的典型流转想成：

`中心 Harbor 接收镜像 -> 自动扫描 -> 通过策略校验 -> 复制到边缘 Harbor/Registry -> 边缘集群拉取本地副本`

如果资源很有限，或者团队目前还没有镜像准入要求，直接上 Harbor 可能会显得“过配”。因为 Harbor 带来的不只是功能，还有维护成本：证书、任务队列、扫描数据库更新、用户权限、存储清理，都需要持续管理。

所以适用边界可以概括为一句话：当“镜像安全、流转控制、跨环境分发”开始成为问题时，Harbor 才真正体现价值；如果只是为了存镜像，它并不是最轻的选择。

---

## 参考资料

| 资料 | 用途 | 适用章节 | 版本/来源 |
|---|---|---|---|
| Harbor Architecture Overview | 说明 Core、Portal、Registry、Jobservice 等架构关系 | 核心结论、问题定义、核心机制 | Harbor GitHub Wiki |
| Configure HTTPS Access to Harbor | 说明 HTTPS、自签证书、Docker 客户端证书配置 | 代码实现、工程权衡 | Harbor Docs 2.14 |
| Create Replication Rules | 说明复制规则、事件触发、过滤与路径处理 | 核心机制、代码实现 | Harbor Docs latest |
| Harbor/Trivy 漏洞扫描相关说明 | 说明扫描器、漏洞结果与项目策略 | 核心机制、工程权衡 | Trivy/相关平台文档 |

1. Harbor Architecture Overview: https://github.com/goharbor/harbor/wiki/Architecture-Overview-of-Harbor
2. Configure HTTPS Access to Harbor: https://goharbor.io/docs/2.14.0/install-config/configure-https/
3. Create Replication Rules: https://goharbor.io/docs/latest/administration/configuring-replication/create-replication-rules/
4. Harbor 漏洞扫描示例说明: https://cloud.google.com/distributed-cloud/hosted/docs/latest/appliance/platform-application/pa-ao-operations/scan-vulnerabilities
