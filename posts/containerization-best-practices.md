## 核心结论

容器化部署的本质，是把“大模型服务需要什么运行环境”写成可重复执行的定义，再把“这些实例应该在哪里、以多少副本、在什么状态下接流量”交给编排系统处理。对大模型推理服务来说，这一步不是锦上添花，而是把“能跑”升级成“稳定地反复跑”。

Docker 解决的是构建一致性。镜像是应用运行快照，也就是把基础镜像、依赖包、启动命令和必要文件打包成一个可分发产物。Kubernetes 解决的是调度一致性。它根据声明式配置，也就是“描述目标状态而不是手工执行步骤”的方式，持续把集群实际状态收敛到你想要的状态。

从工程链路看，最佳实践不是单个工具，而是一条闭环：

| 阶段 | 主要工具 | 目标 | 关键配置 |
|---|---|---|---|
| 镜像构建 | Docker、Registry | 生成可复用、可追踪的运行产物 | 固定基础镜像版本、多阶段构建、`.dockerignore`、SHA/语义化 tag |
| 调度部署 | Kubernetes | 让服务在集群中稳定运行 | `Deployment`/`StatefulSet`、`ConfigMap`、`Secret`、`resources`、`probes` |
| 持续交付 | CI/CD、GitOps | 自动化发布与回滚 | build/push、更新 manifest、Argo CD/Flux 同步 |
| 可观测性 | Prometheus、Alertmanager、Fluent Bit | 发现故障并追踪根因 | 指标采集、日志聚合、标签规范、告警限高 |

玩具例子可以这样理解：先写一个多阶段 `Dockerfile`，只把 `requirements.txt`、应用代码和模型启动脚本复制进最终镜像，再推送到镜像仓库；然后在 Kubernetes 中创建一个 `Deployment`，引用这个镜像，并通过 `ConfigMap` 注入推理参数，比如 `MODEL_NAME`、`MAX_BATCH_SIZE`。这已经覆盖了一个最小可上线链路。

真实工程例子通常更复杂。一个 AI 平台会把每次提交构建成带 Git SHA 的镜像，如 `inference-api:3f82c1a`，推送到私有 registry；随后 CI 只修改 Git 仓库中的 Helm 或 Kustomize 配置，CD 系统再把这些变更同步到生产集群。这样回滚不是“重新手工部署”，而是“把 Git 状态切回前一个版本”。

---

## 问题定义与边界

本文讨论的问题，是如何在多集群、异构硬件环境里稳定运行大模型推理服务。异构硬件，白话说，就是有些节点是 CPU，有些节点带 GPU，而且不同节点配置不完全一样。稳定运行意味着四件事：镜像在不同机器上行为一致、调度不会把节点压垮、服务没有准备好时不接流量、出现故障时能自动恢复和回滚。

边界也要讲清楚。本文只聚焦 Docker 和 Kubernetes 这一层，也就是“怎么打包”和“怎么部署”。不展开训练流程、模型量化、特征工程、数据集预处理、向量数据库选型等问题。因为这些问题重要，但不属于容器化部署的主线。

对新手，一个好理解的玩具例子是把容器文件系统理解成博客仓库里的 `posts/` 目录：容器启动时看到的是一个已经整理好的目录树。`Dockerfile` 就是在说明这个目录树如何一层层生成，比如先选 `python:3.11-slim` 作为基础，再安装依赖，再复制应用文件。你不需要在每台机器上手动执行这些步骤，镜像已经把步骤的结果固定下来。

大模型部署的基本稳定性可以先抽象成一个公式：

$$
服务稳定 = 可控资源 + 健康状态 + 自动扩缩
$$

这里“可控资源”指 CPU、内存、GPU 的申请与上限明确；“健康状态”指服务在冷启动、正常运行、卡死失效这几种状态能被系统识别；“自动扩缩”指流量变化时系统能增加或减少副本，而不是靠人盯着控制台临时操作。

如果这个边界不清楚，团队常见误区是：把所有稳定性问题都归结为“模型太大”或“机器不够”，结果忽略了镜像膨胀、探针缺失、配置散落、日志丢失这些更常见的工程问题。

---

## 核心机制与推导

先看资源模型。Kubernetes 中最重要的一组配置是 `requests` 和 `limits`。`requests` 是“至少给我预留多少资源”，调度器会依据它决定 Pod 放到哪个节点；`limits` 是“最多只能用到哪里”，kubelet 会据此进行 CPU 节流或在内存超限时触发 OOM 处理。

它满足一个近似关系：

$$
requests \le 实际使用 \le limits
$$

更准确地说，实际使用可能低于 `requests`，也可能在短时间内逼近 `limits`；但从调度视角，系统优先看的是 `requests`。因此一个经验结论是：

$$
调度压力 \approx \sum Pod.requests
$$

这不是精确数学公式，而是理解调度的最短路径。节点是否还能放新 Pod，首先不是看“你程序通常只用多少”，而是看“你声明要预留多少”。

玩具例子：一个推理服务设置

- `requests.cpu = 500m`
- `requests.memory = 1Gi`
- `limits.cpu = 1`
- `limits.memory = 2Gi`

`500m` 表示 0.5 个 CPU 核心。调度器会把它当成“至少要有半个核和 1Gi 内存的空余”，再决定节点归属。假设节点只剩 800m CPU 可分配，那么最多只能再放一个这样的 Pod，而不是两个。`limits` 则保证这个实例即使突然负载升高，也不会无限抢占整台机器。

再看探针链路。探针是“系统判断容器是否健康的自动检查”。首次出现时可以直接理解成“容器给平台的自检接口”。对大模型服务，推荐按 `startup -> readiness -> liveness` 三段设计：

| 探针 | 作用 | 典型问题 |
|---|---|---|
| `startupProbe` | 判断冷启动是否完成 | 模型加载慢，服务几分钟内都还没准备好 |
| `readinessProbe` | 判断是否可以接流量 | 服务进程活着，但模型还没加载完或下游没连通 |
| `livenessProbe` | 判断是否需要重启 | 线程死锁、事件循环卡死、进程假活 |

这三者顺序不能乱。很多大模型服务启动时要加载权重、初始化 tokenizer、建立 GPU 上下文，几十秒到几分钟都可能正常。如果你只配 `livenessProbe`，系统可能在模型还没热身完成时就不断重启它。`startupProbe` 的意义就是：在冷启动窗口内，先别用更激进的存活检查干扰服务。

例如：

- `periodSeconds: 10`
- `failureThreshold: 30`

表示每 10 秒检查一次，连续失败 30 次才算失败。总等待时间约为 $10 \times 30 = 300$ 秒，也就是 5 分钟。这个配置对大模型初始化是合理的起点。

真实工程里，调度与探针要一起看。一个推理 Pod 可能申请 1 张 GPU、4 核 CPU、16Gi 内存，但服务预热依赖下载额外模型分片，前 2 分钟无法接单。如果没有 `readinessProbe`，Service 会过早把请求发给它，结果不是 200，而是连续 503 或超时。业务方往往误以为“扩容无效”，实际问题是“未就绪实例被当成可用实例”。

Deployment 和 StatefulSet 的选择也要基于机制。Deployment 适合无状态服务，也就是副本之间没有稳定身份要求；StatefulSet 适合有稳定网络标识、持久化卷绑定需求的场景，比如推理缓存节点、带本地索引副本的服务。大多数在线推理 API 首选 Deployment，只有当你明确需要固定持久卷或有序发布时，才转向 StatefulSet。

---

## 代码实现

先给一个最小但工程上靠谱的镜像示例。多阶段构建，白话说，就是把“安装依赖”和“最终运行”拆成两个阶段，避免把编译缓存、临时工具一起打进产物镜像。

```Dockerfile
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /install /usr/local
COPY serve.py .
COPY app/ ./app/

EXPOSE 8080
CMD ["python", "serve.py"]
```

对应的 `.dockerignore` 应至少排除这些内容，避免把无关文件打进构建上下文：

```gitignore
.git
.gitignore
__pycache__
*.pyc
tests/
docs/
notebooks/
*.log
.env
```

固定基础镜像版本很关键。不要只写 `python:3.11-slim` 然后长期不管，应该至少固定到明确 tag，最好在团队流程里定期升级并验证。否则同一份 `Dockerfile` 在不同时间构建出的基础层可能已经变化。

下面是一个简化的 Kubernetes `Deployment`。它同时演示了 `ConfigMap`、`Secret`、资源限制和三类探针。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-infer-config
data:
  MODEL_NAME: "qwen2.5-7b-instruct"
  MAX_BATCH_SIZE: "8"
  LOG_LEVEL: "info"
---
apiVersion: v1
kind: Secret
metadata:
  name: llm-infer-secret
type: Opaque
stringData:
  API_TOKEN: "replace-me"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-infer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-infer
  template:
    metadata:
      labels:
        app: llm-infer
    spec:
      containers:
        - name: api
          image: registry.example.com/llm-infer:3f82c1a
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: llm-infer-config
            - secretRef:
                name: llm-infer-secret
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "1"
              memory: "2Gi"
          startupProbe:
            httpGet:
              path: /healthz/startup
              port: 8080
            periodSeconds: 10
            failureThreshold: 30
          readinessProbe:
            httpGet:
              path: /healthz/ready
              port: 8080
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /healthz/live
              port: 8080
            periodSeconds: 10
            failureThreshold: 3
```

如果服务依赖持久化模型缓存或本地索引，并且每个副本都需要稳定卷绑定，就把 `Deployment` 换成 `StatefulSet`。但没有明确状态需求时，不要为了“看起来高级”而默认使用 StatefulSet，因为它发布和缩容都更保守。

下面给一个可运行的 Python 小程序，用来演示“请求资源能否放进节点”的最小判断逻辑。它不是 Kubernetes 源码，只是帮助理解调度约束。

```python
from dataclasses import dataclass

@dataclass
class PodRequest:
    cpu_m: int
    memory_mib: int

@dataclass
class NodeCapacity:
    cpu_m: int
    memory_mib: int

def can_schedule(node: NodeCapacity, pods: list[PodRequest], incoming: PodRequest) -> bool:
    used_cpu = sum(p.cpu_m for p in pods)
    used_mem = sum(p.memory_mib for p in pods)
    return (
        used_cpu + incoming.cpu_m <= node.cpu_m
        and used_mem + incoming.memory_mib <= node.memory_mib
    )

node = NodeCapacity(cpu_m=2000, memory_mib=4096)
existing = [
    PodRequest(cpu_m=500, memory_mib=1024),
    PodRequest(cpu_m=500, memory_mib=1024),
]
incoming_ok = PodRequest(cpu_m=500, memory_mib=1024)
incoming_bad = PodRequest(cpu_m=700, memory_mib=2300)

assert can_schedule(node, existing, incoming_ok) is True
assert can_schedule(node, existing, incoming_bad) is False
```

CI/CD 方面，建议把 CI 和 CD 解耦。CI 只负责三件事：测试、构建镜像、推送镜像。CD 只负责更新部署声明并同步集群。典型流程如下：

1. 代码提交后运行单元测试与镜像安全扫描。
2. 构建镜像并打上 Git SHA tag。
3. 将镜像推送到 registry。
4. 自动修改 GitOps 仓库中的镜像 tag。
5. Argo CD 或 Flux 发现 Git 变更并同步到集群。
6. 观察发布指标，不健康则回滚到前一版 manifest。

真实工程例子中，这种方式的优势非常直接：你不再需要给 CI 系统发放过高的集群写权限，部署状态也能通过 Git 历史完整追溯。

---

## 工程权衡与常见坑

最佳实践不是“配置越多越好”，而是在复杂度和稳定性之间做有边界的取舍。大模型服务尤其容易踩下面几类坑：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 不设 `resources.requests` | Scheduler 误判可用容量，节点被堆满 | 从压测结果反推最小稳定请求值 |
| 只设 `limits` 不设 `requests` | 实例可被塞进过多节点空隙，运行时互相争抢 | `requests` 和 `limits` 成对设计 |
| 没有 `readinessProbe` | 未就绪 Pod 也进入流量池，出现 503 | 把模型加载完成、依赖连通作为 ready 条件 |
| `livenessProbe` 过严 | 冷启动阶段被误杀，循环重启 | 先配 `startupProbe`，再调 `liveness` |
| 日志写在容器本地 | 容器重建后证据丢失 | 用 Fluent Bit/Fluentd 聚合到外部系统 |
| 镜像 tag 只用 `latest` | 无法追踪与回滚 | 使用 Git SHA 或版本号 |
| Secret 写死在镜像里 | 凭证泄漏且难轮换 | 用 `Secret` 或外部密钥管理系统 |

一个常见新手事故是：服务已经监听 8080 端口，但模型还没完全加载，于是应用进程“活着”，业务接口却还不能用。如果没有 `readinessProbe`，Service 会持续把请求打进去，表面现象是“扩容后错误更多了”。解决方式不是盲目加副本，而是先让未准备好的 Pod 不进入后端列表，同时在网关层设置并发和超时控制。

监控和日志也不能留在“以后再补”。容器是短生命周期对象，Pod 被驱逐、重建、漂移都很常见。如果日志只存在容器文件系统里，排障时通常已经找不到关键上下文。指标系统推荐最小集为 Prometheus + Alertmanager，日志系统推荐最小集为 Fluent Bit + 集中式存储。对大模型服务，还应增加以下指标：

- 请求延迟分位数，如 P50、P95、P99
- 每个模型实例的并发数与队列长度
- GPU 显存使用率或 CPU/内存利用率
- Pod 重启次数
- 就绪副本数与期望副本数差值

资源设置上还有一个现实权衡：`limits` 设得太紧，会导致吞吐量上不去；设得太宽，又可能让单个 Pod 吃掉整机资源。最稳妥的方法不是凭经验拍脑袋，而是先做压测，找出“在 SLA 内能稳定运行”的请求资源，再留出有根据的安全余量。

---

## 替代方案与适用边界

不是所有项目都应该一开始上 Kubernetes。工具选型要服从规模与约束。

| 方案 | 成本 | 自动扩缩 | 资源隔离 | 适用场景 |
|---|---|---|---|---|
| Docker Compose | 低 | 弱 | 一般 | 本地开发、小团队单机部署 |
| Kubernetes | 中到高 | 强 | 强 | 多实例生产环境、GPU 调度、滚动发布 |
| Serverless/Knative | 中 | 强 | 中 | 突发流量、事件驱动接口 |
| 传统虚拟机部署 | 中 | 弱 | 中 | 历史系统迁移、平台约束较强 |

Docker Compose 适合玩具例子和本地联调。比如一个小团队先用 `docker compose up` 启动 API、Redis 和 Nginx，快速验证业务流程。这种方式部署成本低，也容易上手。但它缺乏成熟的跨节点调度、自动恢复、声明式发布、灰度流量和 GitOps 回滚能力，不适合作为大模型生产环境的长期解法。

Serverless 或 Knative 适合突发流量明显、空闲时间长的服务，因为它可以按需缩容甚至缩到零。但对大模型来说，冷启动成本通常较高，模型加载时间和 GPU 初始化开销不容忽视，所以它更适合轻量模型、异步任务或对冷启动容忍度较高的场景。需要稳定低延迟推理时，Kubernetes 仍然更可控。

真实工程中，常见路线是“两阶段演进”：

1. 研发初期用 `docker compose` 做单机调试，快速把依赖关系跑通。
2. 进入生产后迁移到 Kubernetes，用 Deployment、探针、HPA、GitOps 和监控体系接管上线流程。

因此，Kubernetes 不是“高级版 Docker”，而是当你开始需要多节点调度、故障恢复、滚动发布和统一观测时，最自然的下一层抽象。

---

## 参考资料

1. Docker 官方文档，`Docker overview`。焦点：容器、镜像、运行时的基本概念。  
2. Docker 官方文档，`Dockerfile best practices`。焦点：多阶段构建、固定基础镜像版本、`.dockerignore`、减少镜像体积。  
3. Kubernetes 官方文档，`Manage Resources for Containers`。焦点：`requests`、`limits`、调度与资源限制机制。  
4. Kubernetes 官方文档，`Deployment`。焦点：Deployment 的声明式发布与副本管理。  
5. Kubernetes 官方文档，`ConfigMap` 与 `Secret` 相关章节。焦点：配置注入与敏感信息管理。  
6. Kubernetes 官方文档，探针相关章节。焦点：`startupProbe`、`readinessProbe`、`livenessProbe` 的语义与适用场景。  
7. Sfeir Institute 关于 Kubernetes CI/CD 与 GitOps 的课程资料。焦点：CI 构建镜像、CD 同步 manifest、Git 回滚、自动化发布。  
8. Kubernetes 社区最佳实践与常见坑文章。焦点：资源配置缺失、探针错误、监控日志策略不足带来的故障模式。  
9. 新手阅读顺序建议：先读 Kubernetes 官方 `manage-resources-containers`，理解资源声明；再读 Deployment 和 Probe 文档，理解服务如何进入流量；最后看 Dockerfile 最佳实践，把构建与部署串成一条链路。
