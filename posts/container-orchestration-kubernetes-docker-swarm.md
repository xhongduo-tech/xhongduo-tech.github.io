## 核心结论

容器编排的本质，是把“我要几个实例、如何暴露、如何更新、失败后怎么办”写成声明，再交给平台持续执行。声明式配置，意思是你描述目标状态，而不是手工写一串部署命令。对于团队来说，这比在多台机器上反复执行 `docker run`、`docker-compose up` 更稳定，因为系统会不断把实际状态拉回目标状态。

Kubernetes 和 Docker Swarm 都在做这件事，但定位不同。Kubernetes 更像完整的集群操作系统，控制面，意思是负责全局调度、状态管理和恢复的管理层，能力全面，适合中大型系统、跨云环境和复杂发布流程。Docker Swarm 更像 Docker 生态里的轻量编排层，和 Docker Engine 集成紧，命令简单，适合小团队快速上手。

对零基础工程师，最重要的判断标准不是“哪个更高级”，而是“你的系统有没有进入多节点、多环境、持续更新的阶段”。如果只有一台机器、单个服务、更新频率低，编排收益有限；一旦涉及多个节点、滚动更新、服务发现、自动恢复，编排平台就从“可选项”变成“基础设施”。

先看一张最核心的对比表：

| 方式 | 你写的内容 | 平台负责的事 | 风险 | 运维负担 |
|---|---|---|---|---|
| 手动命令 | `docker run`、登录机器改配置 | 几乎没有 | 漏步骤、版本不一致、故障恢复慢 | 高 |
| 脚本化部署 | Shell、CI 脚本、定时任务 | 部分自动化 | 脚本漂移、难回滚、状态不可见 | 中高 |
| 声明式编排 | YAML/Compose 声明目标状态 | 调度、扩缩容、服务发现、自愈、滚动更新 | 学习成本更高 | 中低 |

玩具例子可以这样理解。假设你有一个 Web 服务，要保持 2 个实例在线。手动方式是你自己开两个容器，挂了再手工补。编排方式是写 `replicas: 2`，平台发现少了一个，就自动补回去。这就是“期望状态驱动”的核心收益。

真实工程例子更典型。一个电商团队早期在几台机器上跑 `docker-compose`，每次发布都要登录节点逐个更新。后面迁移到 Kubernetes 后，Deployment、Service、ConfigMap 都存进 Git，CI/CD 直接执行 `kubectl apply`，滚动更新自动完成，运维主要盯异常而不是盯命令执行过程。这类收益不在于“容器能不能跑起来”，而在于“系统能不能长期稳定地更新和恢复”。

---

## 问题定义与边界

容器编排要解决的不是“如何启动一个容器”，而是“如何在多节点环境中持续维护一组服务的生命周期”。生命周期，意思是从创建、调度、扩缩容、健康检查、故障恢复到下线的完整过程。

它至少包含四类问题：

| 问题 | 含义 | 典型后果 |
|---|---|---|
| 调度 | 容器放到哪台机器上 | 某些机器过载，某些机器空闲 |
| 服务发现 | 客户端如何找到服务实例 | IP 变化导致调用失败 |
| 扩缩容 | 流量变化时如何增减实例 | 高峰扛不住，低峰浪费资源 |
| 更新与回滚 | 新版本如何替换旧版本 | 发布中断、故障扩大 |

Kubernetes 与 Docker Swarm 都覆盖这些基础问题，但边界不一样。

| 维度 | Kubernetes | Docker Swarm |
|---|---|---|
| 控制面 | 分层明确，组件多，能力完整 | 更轻量，和 Docker Engine 紧耦合 |
| 声明模型 | 资源对象丰富，如 Pod、Deployment、Service | 以 service/task 为核心，模型更简单 |
| 扩展性 | 强，支持复杂调度、策略和生态扩展 | 中等，适合较简单场景 |
| 云集成 | 各大云厂商普遍支持 | 相对较弱 |
| 上手成本 | 高 | 低 |

这张表背后的边界很重要。Docker Swarm 适合“我已经熟悉 Docker，希望很快把单机容器扩展到多机”。Kubernetes 适合“我要长期维护一个会持续演进的集群系统”。

一个典型的小团队场景是：先用 `docker stack deploy` 在 Swarm 上部署几个服务，因为 Compose 迁移成本低；当需求变成多环境隔离、更细粒度滚动更新、统一配置管理、跨云部署时，再迁移到 Kubernetes。这里不是说 Swarm 不能用，而是它的控制面深度和生态宽度决定了适用上限。

因此本文讨论的边界是：多节点、长期运行、需要自动恢复和持续发布的 Web 服务或模型服务。单机实验、一次性批处理任务、极小规模内网工具，不一定值得直接引入 Kubernetes。

---

## 核心机制与推导

先把 Kubernetes 的几个基本对象讲清楚。

Pod 是最小调度单元。白话讲，Kubernetes 不直接调度单个容器，而是调度一个“容器小组”。同一个 Pod 里的容器共享网络和部分存储，所以它适合放紧密协作的进程，比如主服务容器加日志采集容器。

Service 是稳定入口。白话讲，Pod 会重建，IP 会变，但 Service 提供一个稳定名字和访问地址，把流量转发给一组符合标签条件的 Pod。

Deployment 是副本与更新控制器。控制器，意思是一个持续观察并纠正状态的后台程序。你写 `replicas: 3`，Deployment 会确保集群里始终有 3 个符合模板的 Pod。

ConfigMap 和 Secret 是配置容器外置化。外置化，意思是把配置从镜像里拿出来，避免每次改配置都重新打镜像。ConfigMap 用于普通配置，Secret 用于敏感信息。

Kubernetes 的核心思想可以写成一个非常简单的状态约束问题。假设目标副本数为 $r$，滚动更新参数为 `maxUnavailable=u`、`maxSurge=s`，那么系统在更新期间要尽量满足：

$$
availableReplicas \in [r-u,\; r+s]
$$

这条式子表达的是：更新时允许短暂少一些可用实例，也允许短暂多一些实例，但不能无限偏离目标。比如：

- `replicas: 2`
- `maxUnavailable: 1`
- `maxSurge: 1`

那么可用区间是：

$$
availableReplicas \in [2-1,\; 2+1] = [1,\; 3]
$$

这意味着滚动更新时，系统至少要保证 1 个实例可用，最多临时跑到 3 个实例。于是典型过程是：

1. 先新建 1 个新版本 Pod，总数临时增加。
2. 等新 Pod 就绪后，删掉 1 个旧 Pod。
3. 再建下一个新 Pod。
4. 再删一个旧 Pod，直到全部替换完成。

玩具例子如下。你有一个 Nginx 服务，平时跑 2 个副本。现在把镜像从 `1.21` 升级到 `1.22`。如果没有滚动更新，你可能会先停旧实例再起新实例，造成短暂中断。Deployment 则按照上述约束逐步替换，所以用户请求仍然能被至少一个健康副本处理。

真实工程例子是模型推理服务。某推荐系统线上跑 8 个推理实例，升级新模型时不能整体下线，因为推荐请求是持续到来的。此时 Deployment 负责控制替换节奏，Service 始终把流量送到就绪实例，readiness probe 决定新实例什么时候能接流量，ConfigMap 或模型版本参数决定服务启动时加载哪个配置。整个过程中，开发者只声明目标状态，不手工编排替换顺序。

从 Docker Swarm 的角度看，思路类似，只是对象更少。Swarm 的 service 对应“我要长期运行的一组任务”，task 对应具体实例，manager 节点负责调度，worker 节点负责执行。它也支持副本和负载均衡，但控制粒度和扩展方式比 Kubernetes 更有限。

---

## 代码实现

先给一个新手能直接看懂的 Kubernetes 示例。这个示例声明了三个对象：ConfigMap、Deployment、Service。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: demo-config
immutable: true
data:
  APP_NAME: "demo-web"
  APP_ENV: "prod"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo-web
spec:
  replicas: 2
  selector:
    matchLabels:
      app: demo-web
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: demo-web
      annotations:
        config-hash: "v1"
    spec:
      containers:
      - name: web
        image: nginx:1.21
        ports:
        - containerPort: 80
        envFrom:
        - configMapRef:
            name: demo-config
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 3
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 10
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "256Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: demo-web-svc
spec:
  selector:
    app: demo-web
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
```

这个清单表达了三个关键信息：

| 对象 | 作用 | 为什么重要 |
|---|---|---|
| ConfigMap | 注入普通配置 | 改配置不必重建镜像 |
| Deployment | 保持 2 个副本并滚动更新 | 实例挂了自动补，发版不中断 |
| Service | 提供稳定访问入口 | Pod 重建后地址仍然稳定 |

如果你从 Docker Swarm 迁移过来，可以把它和下面的思路做类比：`docker service create --replicas 2` 大致相当于“我声明一个长期运行的服务并保持 2 个副本”；而 Kubernetes 在此基础上再拆成 Deployment、Pod、Service、ConfigMap 等对象，让职责更清晰。

再给一个可运行的 Python 小程序，用来模拟滚动更新时的副本约束。它不依赖 Kubernetes，但能帮助理解公式是否成立。

```python
def rollout_steps(replicas: int, max_unavailable: int, max_surge: int):
    lower = replicas - max_unavailable
    upper = replicas + max_surge
    assert lower >= 0
    assert upper >= replicas

    states = []
    available = replicas

    # 启动一个新副本
    available += max_surge
    assert lower <= available <= upper
    states.append(("surge", available))

    # 删除一个旧副本
    available -= 1
    assert lower <= available <= upper
    states.append(("replace_old", available))

    # 再启动一个新副本并删除一个旧副本
    available += 1
    assert lower <= available <= upper
    states.append(("surge_again", available))

    available -= 1
    assert lower <= available <= upper
    states.append(("finish", available))

    return states


result = rollout_steps(replicas=2, max_unavailable=1, max_surge=1)
assert result[0] == ("surge", 3)
assert result[-1] == ("finish", 2)
print(result)
```

这个例子说明：即使在替换过程中，副本数会波动，但只要始终落在允许区间内，服务就能以可控方式完成升级。

真实工程里，还会把 ConfigMap 或 Secret 的变更和 Deployment 绑定起来。常见做法是在 Pod 模板里加一个配置哈希注解，比如 `config-hash: "v1"`。配置更新后同时更新哈希值，Deployment 才会识别模板变化并触发滚动更新。否则你改了配置对象，运行中的 Pod 不一定自动重建。

---

## 工程权衡与常见坑

第一个常见坑是直接创建裸 Pod。裸 Pod，意思是没有 Deployment、StatefulSet、Job 等上层控制器管理的 Pod。它能跑，但节点宕机、Pod 被删、版本要更新时，没有控制器帮你自动恢复或滚动替换。线上长期服务几乎不应使用裸 Pod。

第二个坑是把 `hostPort` 或 `hostNetwork` 当默认方案。它们的意思分别是“直接占用宿主机端口”和“直接共享宿主机网络”。这样做虽然简单，但会把 Pod 和具体节点绑定得很死。一个节点上 8080 端口已经占了，你的新 Pod 就不能调度到这台机器，集群弹性会明显下降。

第三个坑是把启动成功误当成“可以接流量”。Kubernetes 里至少要区分三类探针：

| 探针 | 作用 | 失败后果 |
|---|---|---|
| startupProbe | 判断应用是否完成冷启动 | 适合启动慢的服务 |
| livenessProbe | 判断应用是否还活着 | 失败会重启容器 |
| readinessProbe | 判断应用是否可接流量 | 失败会从 Service 后端移除 |

很多新手只配 liveness，不配 readiness，结果容器进程虽然启动了，但模型还没加载完、数据库连接池还没建好，流量已经打进来了，最终表现为“刚发布就 502”。

第四个坑是配置散落。所谓散落，就是一部分写在镜像里，一部分写在脚本里，一部分靠人工登录机器改。这样做的直接后果是环境不可复现。正确做法是：YAML 进 Git，ConfigMap/Secret 统一管理，镜像只放程序和稳定依赖。

下面这张表可以当检查单：

| 风险项 | 具体问题 | 对策 |
|---|---|---|
| 裸 Pod | 宕机后不会自动补，无法滚动更新 | 用 Deployment、Job、StatefulSet |
| `hostPort` / `hostNetwork` | 绑死节点资源，调度受限 | 优先使用 Service 和集群网络 |
| 配置散落 | 环境不一致，难以追溯 | 配置全部纳入 Git 和对象管理 |
| 探针混用 | 容器未准备好就接流量 | 区分 startup、liveness、readiness |

真实工程例子是合规环境中的 API 服务。服务启动时要先拉取证书、加载规则、建立数据库连接。此时应该把 `readinessProbe` 指向 `/healthz` 或更严格的就绪接口，只有当依赖全部准备完成时才返回成功，Service 才把该 Pod 纳入后端。否则发布窗口里，网关会把请求打到尚未就绪的实例上，造成间歇性失败。

另外，资源配置也不能忽略。`requests` 是调度时预留的最低资源，`limits` 是允许使用的上限。不给 `requests`，调度器无法更准确地放置 Pod；给得过小，则高峰期容易被限流或 OOM。对于模型服务尤其要注意内存，因为模型加载常常比 Web 服务更重。

---

## 替代方案与适用边界

Docker Swarm 的优势在于简单。对已经熟悉 Docker CLI 和 Compose 的团队，它几乎是最自然的第一步：初始化集群、创建 service、声明副本数、自动负载均衡，学习曲线比 Kubernetes 平缓很多。

一个典型场景是内部工具平台。团队有 3 到 5 台机器，要运行几个 Web 服务、一个数据库代理、一个监控组件。此时 Swarm 往往够用，因为需求重点是“快速把多机跑起来”，而不是“构建一个完整的平台能力层”。

但如果进入以下场景，Kubernetes 会更合适：

| 场景 | 更适合的方案 | 原因 |
|---|---|---|
| 小团队、少量服务、本地到多机过渡 | Docker Swarm | 上手快，和 Docker 习惯一致 |
| 多环境隔离、复杂滚动更新 | Kubernetes | 控制器和对象模型更细 |
| 多云或混合云部署 | Kubernetes | 云厂商支持和生态更成熟 |
| 需要丰富生态，如 Helm、Operator、服务网格 | Kubernetes | 扩展能力更强 |

玩具例子是一个新项目早期只有 3 个 API 副本。你可以直接用：

`docker service create --name api --replicas 3 my-api:1.0`

这已经能让你体验分布式部署、基本调度和负载均衡。对于团队学习曲线，这种方式成本最低。

真实工程例子则相反。一个跨区域模型推理平台需要按命名空间隔离团队、按节点标签放置 GPU 工作负载、按环境区分配置、用 Helm 管理发布版本，还要接入云上的负载均衡和存储。这时 Swarm 的简洁会变成能力不足，而 Kubernetes 的复杂度反而是必要成本。

所以选择标准不是“喜欢哪个命令”，而是“你的系统复杂度是否需要这套控制面”。如果只是想摆脱单机脚本，Swarm 很合适；如果目标是长期演进的生产平台，Kubernetes 更稳妥。

---

## 参考资料

- IBM, Container orchestration 概念与优势介绍: https://www.ibm.com/think/topics/container-orchestration
- Docker Docs, Swarm mode key concepts: https://docs.docker.com/engine/swarm/key-concepts/
- Kubernetes Docs, Pod overview: https://kubernetes.io/docs/concepts/workloads/pods/pod-overview/
- Kubernetes Docs, Deployment: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Kubernetes Docs, Configuration best practices: https://kubernetes.io/docs/concepts/configuration/overview/
