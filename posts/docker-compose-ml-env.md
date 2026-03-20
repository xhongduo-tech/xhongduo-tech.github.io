## 核心结论

Docker Compose 是一种声明式编排工具。声明式的意思是，你只描述“系统应该长成什么样”，而不是逐条写“先执行什么命令、再执行什么命令”。在 `docker-compose.yml` 或 `compose.yaml` 里，最核心的三层结构是 `services`、`networks`、`volumes`：

| 关键字 | 含义 | 解决的问题 |
| --- | --- | --- |
| `services` | 服务，也就是一组要运行的容器定义 | 每个进程如何启动、暴露什么端口、依赖谁 |
| `networks` | 网络，也就是容器之间如何互联 | 服务间通信、服务名解析 |
| `volumes` | 卷，也就是可持久化的数据目录 | 容器重建后数据仍然保留 |

对初学者，最重要的认识只有两个。

第一，Compose 管的是“一个系统”，不是单个容器。Web、数据库、缓存、训练任务、推理服务、监控组件，应该作为一个整体一起描述。

第二，`depends_on` 只解决“依赖关系”，不天然解决“服务真正可用”。如果数据库进程刚启动但还没准备好接受连接，业务容器仍然可能失败。因此，真实工程里几乎总要把 `depends_on` 和 `healthcheck` 一起使用。

如果把一个多容器系统抽象成有向图，服务是节点，依赖是边，那么 Compose 做的事情可以理解为：根据图的结构创建容器、网络和卷，并按依赖顺序启动。形式化地写，可以把启动顺序看成一个偏序关系：

$$
A \prec B \Rightarrow B \text{ 必须在 } A \text{ 满足条件后启动}
$$

这里“满足条件”不一定只是“进程存在”，也可以是“健康检查通过”或“一次性任务成功完成”。

---

## 问题定义与边界

Compose 解决的问题是：在单机或单个 Docker 引擎环境里，如何稳定地管理一组相互协作的容器。

“单机”不是说只能本机开发，也包括一台云服务器、一台实验机、一台带 GPU 的训练机。只要核心前提还是“一台主机上运行多个容器协同工作”，Compose 就是合适工具。

一个玩具例子最容易说明问题。假设你有两个服务：

1. `app`：一个 Python Web 服务
2. `db`：一个 PostgreSQL 数据库

如果只用 `docker run`，你要手动创建网络、指定容器名、映射端口、挂载数据目录、设置环境变量，还要控制启动顺序。服务一多，命令会迅速失控。Compose 的目标就是把这些零散命令收束为一个配置文件。

最小示例：

```yaml
services:
  app:
    image: python:3.12-slim
    command: ["python", "-m", "http.server", "8000"]
    ports:
      - "8000:8000"
    depends_on:
      - db

  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: example
```

这段配置能表达“有两个服务，`app` 依赖 `db`”。但它的边界也很明确：Compose 不是集群调度器。它不擅长以下问题：

| 场景 | Compose 是否擅长 | 原因 |
| --- | --- | --- |
| 本地开发环境 | 是 | 配置简单，启动快 |
| 单机测试环境 | 是 | 依赖关系清晰，便于复现 |
| 单机 GPU 训练机 | 是 | 便于统一管理训练、推理、监控 |
| 多主机自动扩缩容 | 否 | 缺少集群级调度能力 |
| 大规模高可用生产集群 | 否 | 缺少原生服务发现、滚动升级、弹性伸缩体系 |

所以，Compose 的边界不是“能不能跑生产”，而是“这个生产系统是不是本质上仍然是单机编排”。很多中小系统、内部工具、模型实验平台，完全可以长期使用 Compose。

---

## 核心机制与推导

先看 `services`。每个 service 本质上是“一个容器实例模板”。镜像、命令、端口、环境变量、挂载目录、重启策略，都是围绕这个模板展开。

再看 `networks`。网络可以理解为“容器加入的通信平面”。在同一个 Compose 网络内，服务名默认可作为 DNS 名称直接解析。也就是说，`app` 访问数据库时，主机名通常直接写 `db`，不需要查 IP。

再看 `volumes`。卷可以理解为“脱离容器生命周期的数据目录”。容器删了，卷不一定删；新容器重新挂载同一个卷，数据还在。

一个真实的依赖推导是这样的：

1. 如果 `infer` 服务需要读模型文件，而模型文件由 `trainer` 生成，那么两者必须共享卷。
2. 如果 `monitor` 需要抓取 `infer` 的指标接口，那么两者必须在同一网络内。
3. 如果 `infer` 启动前必须等 `trainer` 把模型准备好，那么仅有启动顺序还不够，必须定义“就绪条件”。

因此，完整关系不是“先后启动”这么简单，而是：

| 关系类型 | Compose 对应能力 | 典型字段 |
| --- | --- | --- |
| 通信关系 | 谁能互相访问 | `networks` |
| 数据关系 | 谁读写同一份数据 | `volumes` |
| 时序关系 | 谁必须等谁准备好 | `depends_on` + `healthcheck` |
| 资源关系 | 谁需要 GPU、内存、CPU | `deploy.resources` |

`depends_on` 的关键误区在于，很多人以为它表示“等上游服务可用”。实际上默认更接近“等上游容器启动过”。这两者不是一回事。

数据库举例：

- “容器启动”表示 PostgreSQL 进程被拉起来了。
- “服务健康”表示 PostgreSQL 已完成初始化，并能接受连接。

所以在工程上，应该把“依赖条件”写成显式约束：

```yaml
services:
  app:
    image: my-app:latest
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: example
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 10
```

这里的 `healthcheck` 是健康检查，意思是容器内部定期执行一个探测命令；只有探测成功，Compose 才把它视为 `healthy`。

数学上可以把可用性条件写成：

$$
Ready(service) = Started(service) \land HealthPassed(service)
$$

仅有 `Started(service)` 不足以推出 `Ready(service)`。这就是为什么“看起来都启动了，但业务还是连不上数据库”。

GPU 资源也是类似逻辑。容器不是天然就能拿到 GPU，必须显式声明设备请求。现代 Compose 配置通常写在 `deploy.resources.reservations.devices` 下，核心字段是：

- `driver`：设备驱动，这里通常是 `nvidia`
- `capabilities`：设备能力，表示要什么类型的 GPU 功能
- `count` 或 `device_ids`：请求多少张卡，或指定哪几张卡

一个典型配置：

```yaml
services:
  trainer:
    image: my-trainer:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              count: 1
```

含义很直接：`trainer` 这个服务启动时，需要 1 张 NVIDIA GPU。

---

## 代码实现

先给一个玩具例子。目标是启动一个 Web 服务和一个 Redis，验证容器名解析与依赖关系。

```yaml
services:
  web:
    image: nginx:1.27
    ports:
      - "8080:80"
    networks:
      - app-net
    depends_on:
      redis:
        condition: service_started

  redis:
    image: redis:7
    networks:
      - app-net

networks:
  app-net:
```

这个例子说明三件事：

1. 两个服务在同一个网络里，可以用服务名互相访问。
2. `web` 的启动顺序依赖 `redis`。
3. 网络是声明式创建的，不需要手动 `docker network create`。

下面给真实工程例子：用 Compose 管理训练、推理和监控三个容器。场景是单机 AI 开发环境。

```yaml
services:
  trainer:
    image: myorg/trainer:latest
    command: ["python", "train.py"]
    volumes:
      - models:/workspace/models
      - datasets:/workspace/datasets
    networks:
      - app-net
    healthcheck:
      test: ["CMD-SHELL", "test -f /workspace/models/latest/model.bin"]
      interval: 10s
      timeout: 3s
      retries: 12
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              count: 1

  infer:
    image: myorg/infer:latest
    command: ["python", "serve.py"]
    ports:
      - "8000:8000"
    volumes:
      - models:/workspace/models:ro
    networks:
      - app-net
    depends_on:
      trainer:
        condition: service_healthy

  monitor:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    networks:
      - app-net
    depends_on:
      infer:
        condition: service_started

volumes:
  models:
  datasets:

networks:
  app-net:
```

这份配置的设计重点如下：

| 服务 | 角色 | 关键配置 |
| --- | --- | --- |
| `trainer` | 训练模型 | GPU 资源、数据卷、模型产出健康检查 |
| `infer` | 提供推理接口 | 只读挂载模型卷，等待训练完成 |
| `monitor` | 拉取指标与监控 | 与推理服务同网，按顺序启动 |

注意这里的健康检查不是“HTTP 返回 200”，而是“模型文件已经生成”。这说明健康检查的本质不是固定语法，而是“用一个可验证条件表达服务是否真的可用”。

再给一个可运行的 Python 代码块，用来模拟 Compose 里的依赖调度逻辑。它不是 Compose 本身，而是帮助理解“为什么健康检查比单纯启动更重要”。

```python
from collections import defaultdict, deque

services = {
    "db": {"healthy": True, "depends_on": []},
    "trainer": {"healthy": True, "depends_on": ["db"]},
    "infer": {"healthy": False, "depends_on": ["trainer"]},
}

def can_start(service_name, services_state):
    deps = services_state[service_name]["depends_on"]
    return all(services_state[d]["healthy"] for d in deps)

assert can_start("trainer", services) is True
assert can_start("infer", services) is True

services["trainer"]["healthy"] = False
assert can_start("infer", services) is False

def topo_order(services_state):
    graph = defaultdict(list)
    indegree = {name: 0 for name in services_state}
    for name, cfg in services_state.items():
        for dep in cfg["depends_on"]:
            graph[dep].append(name)
            indegree[name] += 1

    q = deque([name for name, deg in indegree.items() if deg == 0])
    order = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return order

order = topo_order(services)
assert order[0] == "db"
assert "trainer" in order and "infer" in order
print("startup order:", order)
```

这段代码表达的结论是：依赖顺序可以拓扑排序，但真正能否启动，还要看依赖服务是否健康。

---

## 工程权衡与常见坑

第一个常见坑是把 `depends_on` 当成“就绪保证”。这会导致系统在 `docker compose up` 时偶发失败，尤其是数据库、消息队列、模型加载服务。解决方法不是盲目加 `sleep 10`，而是写可验证的 `healthcheck`。

第二个坑是卷和绑定挂载混用却不理解差异。

- 卷 `volumes` 适合持久化和跨容器共享，由 Docker 管理生命周期。
- 绑定挂载 `./data:/app/data` 适合本地开发，直接映射宿主机目录。

如果训练任务要长期保存模型，优先考虑命名卷；如果你要实时修改代码并让容器立即看到，优先考虑绑定挂载。

第三个坑是网络隔离没想清楚。Compose 默认会创建项目级网络，这通常够用。但如果把数据库端口暴露到宿主机，其实是扩大了攻击面。很多内部依赖根本不需要 `ports`，只需要放在同一网络里即可。

第四个坑是 GPU 配置沿用旧写法。很多旧文章还在写 `runtime: nvidia`。在新环境里，更稳妥的做法是声明设备请求，并确认宿主机已经正确安装 NVIDIA Container Toolkit。否则 Compose 文件写得再对，容器内也看不到显卡。

第五个坑是把 Compose 当作“部署脚本集合”。Compose 文件不应该堆满大量临时逻辑，例如启动前先下载十几个文件、执行一长串 shell 拼接命令。更好的做法是：

1. 镜像里固化稳定依赖
2. 用 `command` 或 `entrypoint` 只做最小启动逻辑
3. 用 `healthcheck` 表达可用性
4. 用卷表达数据流

这样系统结构才清晰。

---

## 替代方案与适用边界

如果需求只是“本地开发环境一键启动”，Compose 几乎总是第一选择。它比手写脚本稳定，因为资源关系写在结构化 YAML 里；它也比直接上 Kubernetes 成本低，因为认知负担小很多。

但 Compose 不是所有场景的终点。

| 方案 | 适用场景 | 不适用场景 |
| --- | --- | --- |
| 纯 `docker run` | 单个临时容器 | 多服务协同 |
| Docker Compose | 单机多容器系统 | 大规模多主机集群 |
| Kubernetes | 集群级调度、高可用、弹性扩缩容 | 小团队本地开发的轻量场景 |

对零基础读者，最实用的判断标准是：

- 如果你只有一台机器，要管理 Web、DB、Redis、训练、推理、监控这些协作组件，用 Compose。
- 如果你要在多台机器之间调度副本、自动恢复、灰度发布、弹性扩容，用 Kubernetes。
- 如果你只是想临时跑一个数据库或一个测试服务，单个 `docker run` 就够了。

还有一个现实边界：Compose 能描述资源，但不负责替你准备宿主机能力。比如 GPU 场景，Compose 只能声明“我要 GPU”，不能替你安装驱动、配置运行时、解决 CUDA 版本兼容。宿主机环境仍然是前提条件。

所以最稳妥的工程理解是：Compose 负责“容器系统的声明与编排”，不负责“宿主机平台能力的构建”。

---

## 参考资料

- Docker Docs, Compose file reference: https://docs.docker.com/compose/compose-file/
- Docker Docs, Networks reference: https://docs.docker.com/reference/compose-file/networks/
- Docker Docs, Volumes reference: https://docs.docker.com/reference/compose-file/volumes/
- Docker Docs, Deploy resources reference: https://docs.docker.com/reference/compose-file/deploy/
- Docker Docs, Startup order and dependencies: https://docs.docker.com/compose/startup-order/
- NVIDIA Container Toolkit 文档: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
