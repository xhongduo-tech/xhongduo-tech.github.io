## 核心结论

Pod 是 Kubernetes 的最小调度单元。调度单元可以理解为“调度器一次摆放的完整对象”：不是单个容器，而是一组必须一起运行的容器。一个 Pod 里的主容器负责业务，sidecar 容器负责补充能力，比如代理、日志收集、证书刷新或监控上报。

Pod 之所以存在，不是为了把多个无关进程硬塞进一个壳里，而是为了表达“这些进程必须共享同一份网络和存储环境”。同一 Pod 内的容器共享网络命名空间，也就是共享 IP 和端口空间；还可以共享卷，也就是共享文件目录。这种设计适合强耦合协作，不适合把一堆彼此独立的服务胡乱拼装。

资源层面要抓住两件事：

| 项目 | `requests` | `limits` |
|---|---|---|
| 含义 | 最低保障，调度时按它预留 | 运行上限，内核/`cgroup` 按它约束 |
| 主要作用阶段 | 调度阶段 | 运行阶段 |
| 对节点选择的影响 | 直接影响能否被放上节点 | 不直接决定放置 |
| 对 CPU 的效果 | 影响分配权重与可获得份额 | 超过后会被节流 |
| 对内存的效果 | 影响可被驱逐优先级 | 超过后可能 OOMKill |

对大多数业务 Pod，可以先记住两条公式：

$$
\text{Pod effective requests}=\max(\sum \text{app+sidecar requests},\ \text{init max request})
$$

$$
\text{Pod effective limits}=\max(\sum \text{app+sidecar limits},\ \text{init max limit})
$$

如果没有 init 容器，公式就退化成常见形式：

$$
\text{Pod requests}=\sum(\text{主容器}+\text{sidecar})
,\quad
\text{Pod limits}=\sum(\text{主容器}+\text{sidecar})
$$

玩具例子：把主容器看成处理请求的 Web 服务，把 sidecar 看成统一日志收集器。调度器看到整个 Pod 的 `requests` 后，才决定把它放到哪台节点；运行时 `limits` 再把它限制在“最多能吃多少 CPU 和内存”的边界里，避免某个 Pod 抢光整机资源。

---

## 问题定义与边界

Pod 解决的是“多个强耦合进程如何作为一个逻辑单元被部署、调度、启停和恢复”的问题，不解决“所有服务都应该塞进一个 Pod”这种误解。

这里的边界要分清：

| 问题 | Pod 负责的边界 | 不应混淆的边界 |
|---|---|---|
| 调度 | 以整个 Pod 为单位放到节点 | 不按单个容器分别调度 |
| 网络 | 同 Pod 内容器共享 IP | Pod 之间仍然通过 Service/网络策略通信 |
| 存储 | 可共享同一个卷 | 不等于自动共享所有宿主机文件 |
| 启动 | 可定义 init、startup、readiness | 不是应用内部依赖编排器 |
| 终止 | 可定义 `preStop` 和宽限期 | 不保证业务自动优雅退出，应用本身仍要配合 |
| 容器职责 | 主容器做业务，sidecar 做辅助 | sidecar 不应反客为主承担主业务 |

白话解释一下“共享命名空间”：命名空间就是进程看到的运行环境边界。共享网络命名空间，等价于这些容器像运行在同一台小机器里，能直接通过 `localhost` 通信。

初学者常见误区有两个。第一，把 Pod 当“虚拟机替身”，结果把数据库、应用、日志、定时任务全放一起，故障域被绑死。第二，只看到容器，不看 Pod，总以为给主容器 250m CPU 就够了，却忘了 sidecar 也在持续吃资源。

可以用一个餐厅类比理解调度边界，但只停留在边界层面：`requests` 像预订桌位，表示“至少要留出这些座位我才来”；优雅关机像餐厅打烊时先停止接新客，再等在场客人吃完，而不是立刻关灯赶人。

---

## 核心机制与推导

Kubernetes 在调度时看的是 Pod 的有效请求值，而不是瞬时实际使用值。原因很直接：调度是“预先摆放”，必须依赖声明式配额，而不能赌运行时刚好不冲突。

### 1. 资源如何推导

假设一个 Pod 有两个长期运行容器：

- `app`: `cpu=250m, mem=256Mi`
- `proxy`: `cpu=100m, mem=128Mi`

那么：

$$
\text{effective cpu request}=250m+100m=350m
$$

$$
\text{effective memory request}=256Mi+128Mi=384Mi
$$

如果再加一个只在启动时运行的 init 容器：

- `init-db`: `cpu=600m, mem=128Mi`

则 CPU 的有效请求会变成：

$$
\max(350m, 600m)=600m
$$

因为 init 容器虽然不长期运行，但启动阶段也要占资源，调度器必须预留得下。

### 2. `requests` 与 `limits` 分别控制什么

- `requests` 决定调度器是否认为这台节点“放得下”
- `limits` 决定 kubelet 和底层 `cgroup` 在运行时如何约束容器
- CPU 超过 `limits` 常见表现是被节流
- 内存超过 `limits` 常见表现是进程被杀，出现 OOMKill

一个常见配置是：

- `requests`: `cpu: 250m`, `memory: 256Mi`
- `limits`: `cpu: 500m`, `memory: 512Mi`

这表示“调度时至少给我 250m/256Mi，但运行中允许我短时冲到 500m/512Mi”。这比把请求和上限都设到峰值更容易调度，也比完全不设上限更安全。

### 3. QoS 如何判定

QoS 是服务质量等级，可以理解为“节点资源紧张时，系统优先保谁、先驱逐谁”的分层。

| QoS | 判定条件 | 工程含义 |
|---|---|---|
| Guaranteed | Pod 内每个容器都设置 CPU/内存 `requests=limits` | 最稳定，最不容易在压力下被驱逐 |
| Burstable | 至少一个容器设置了请求或上限，但不全相等 | 最常见，允许弹性 |
| BestEffort | 所有容器都不设请求和上限 | 最容易被驱逐，不建议生产使用 |

可写成简化规则：

$$
\text{Guaranteed} \iff \forall c,\ request_{cpu}(c)=limit_{cpu}(c)\land request_{mem}(c)=limit_{mem}(c)
$$

否则只要设置了部分资源，一般就是 `Burstable`；完全不设才是 `BestEffort`。

### 4. 探针与生命周期如何影响稳定性

探针是 kubelet 判断容器状态的周期性检查。

- `livenessProbe`：活着吗，不活就重启
- `readinessProbe`：能接流量吗，不能接就从 Service 端点摘除
- `startupProbe`：是否还在启动中，成功前先别按 liveness/readiness 的标准催促

一个简化流程可以写成：

```text
创建 Pod
-> 调度器按 effective requests 选节点
-> kubelet 拉起 init / app / sidecar
-> startupProbe 成功后
-> readinessProbe 成功 => 加入 Service 端点
-> livenessProbe 连续失败 => 重启容器
-> 删除 Pod 时执行 preStop
-> 发送 SIGTERM
-> 宽限期结束仍未退出 => SIGKILL
```

真实工程例子：服务网格场景里，主容器处理业务请求，Linkerd 或其他代理 sidecar 负责透明代理。如果主容器一收到删除信号就立即退出，而 sidecar 又没有正确等待、摘流量和清连接，正在处理的请求可能被截断，表现为滚动发布期间 5xx 突增。

---

## 代码实现

下面先给一个可直接运行的 Python 玩具程序，用来计算 Pod 的有效资源和 QoS，帮助把规则从“背概念”变成“能推出来”。

```python
from typing import List, Dict, Optional

def parse_cpu(v: Optional[str]) -> int:
    if v is None:
        return 0
    return int(v[:-1]) if v.endswith("m") else int(float(v) * 1000)

def parse_mem(v: Optional[str]) -> int:
    if v is None:
        return 0
    if v.endswith("Mi"):
        return int(v[:-2])
    if v.endswith("Gi"):
        return int(float(v[:-2]) * 1024)
    raise ValueError(v)

def effective_pod_resources(apps: List[Dict], inits: List[Dict]):
    app_cpu_req = sum(parse_cpu(c["requests"].get("cpu")) for c in apps)
    app_mem_req = sum(parse_mem(c["requests"].get("memory")) for c in apps)
    app_cpu_lim = sum(parse_cpu(c["limits"].get("cpu")) for c in apps)
    app_mem_lim = sum(parse_mem(c["limits"].get("memory")) for c in apps)

    init_cpu_req = max([parse_cpu(c["requests"].get("cpu")) for c in inits] or [0])
    init_mem_req = max([parse_mem(c["requests"].get("memory")) for c in inits] or [0])
    init_cpu_lim = max([parse_cpu(c["limits"].get("cpu")) for c in inits] or [0])
    init_mem_lim = max([parse_mem(c["limits"].get("memory")) for c in inits] or [0])

    return {
        "cpu_request_m": max(app_cpu_req, init_cpu_req),
        "mem_request_mi": max(app_mem_req, init_mem_req),
        "cpu_limit_m": max(app_cpu_lim, init_cpu_lim),
        "mem_limit_mi": max(app_mem_lim, init_mem_lim),
    }

def qos(containers: List[Dict]) -> str:
    any_set = False
    guaranteed = True
    for c in containers:
        rq, lm = c["requests"], c["limits"]
        if rq or lm:
            any_set = True
        if rq.get("cpu") != lm.get("cpu") or rq.get("memory") != lm.get("memory"):
            guaranteed = False
    if guaranteed and any_set:
        return "Guaranteed"
    if any_set:
        return "Burstable"
    return "BestEffort"

apps = [
    {"requests": {"cpu": "250m", "memory": "256Mi"}, "limits": {"cpu": "500m", "memory": "512Mi"}},
    {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"cpu": "200m", "memory": "256Mi"}},
]
inits = [
    {"requests": {"cpu": "600m", "memory": "128Mi"}, "limits": {"cpu": "600m", "memory": "128Mi"}}
]

r = effective_pod_resources(apps, inits)
assert r["cpu_request_m"] == 600
assert r["mem_request_mi"] == 384
assert qos(apps) == "Burstable"
```

下面是一个更接近生产的 Pod YAML。主容器负责业务，sidecar 负责代理；资源、探针和优雅退出一起配齐。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: svc-with-proxy
  labels:
    app: svc-with-proxy
spec:
  terminationGracePeriodSeconds: 40
  containers:
    - name: app
      image: app:v1
      ports:
        - containerPort: 8080
      resources:
        requests:
          cpu: "250m"
          memory: "256Mi"
        limits:
          cpu: "500m"
          memory: "512Mi"
      startupProbe:
        httpGet:
          path: /healthz
          port: 8080
        periodSeconds: 5
        failureThreshold: 12
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        periodSeconds: 5
        failureThreshold: 2
      livenessProbe:
        httpGet:
          path: /healthz
          port: 8080
        periodSeconds: 10
        failureThreshold: 3
      lifecycle:
        preStop:
          exec:
            command: ["sh", "-c", "sleep 5"]
      volumeMounts:
        - name: shared-logs
          mountPath: /var/log/app

    - name: proxy
      image: cr.l5d.io/linkerd/proxy:stable-2.18.0
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "200m"
          memory: "256Mi"
      readinessProbe:
        httpGet:
          path: /ready
          port: 4191
        periodSeconds: 5
        failureThreshold: 2
      lifecycle:
        preStop:
          exec:
            command: ["sh", "-c", "sleep 10"]

  volumes:
    - name: shared-logs
      emptyDir: {}
```

这个 YAML 的重点不是语法，而是组合关系：

- 主容器和 sidecar 一起构成同一个逻辑单元
- `requests` 让它更容易被合理摆放
- `limits` 给出上界，避免失控
- `startupProbe` 防止慢启动被误判
- `readinessProbe` 控制是否接流量
- `preStop` 和 `terminationGracePeriodSeconds` 给优雅退出留时间

---

## 工程权衡与常见坑

最常见的问题不是“不会写 YAML”，而是“资源语义和生命周期语义没想清楚”。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只设 `limits` 不设 `requests` | 常见场景下会把请求默认补成同值，调度变保守 | 显式分开写 `requests` 和 `limits` |
| 忘记计算 sidecar 资源 | 节点明明空闲却调度失败，或实际资源打满 | 按整个 Pod 预算，不只看主容器 |
| `livenessProbe` 过严 | 暂时抖动也被重启，形成雪崩 | 慢启动用 `startupProbe`，活性阈值放宽 |
| `readinessProbe` 过松 | 故障 Pod 长时间不摘流量 | 让 readiness 反映“能否安全接请求” |
| `preStop` 太长但宽限期太短 | hook 没跑完就被强杀 | 满足 `preStop + 应用退出时间 <= terminationGracePeriodSeconds` |
| sidecar 没有自己的 probe/hook | 主应用正常，辅助链路却已失效 | sidecar 也要视为生产容器对待 |

一个典型误区是“我给 Pod 只设了 `limits.cpu=1`，为什么更难调度了”。原因是调度器并不看你“平时只用 100m”，而是看声明值；如果请求被默认抬到 1 核，节点碎片会迅速变多，很多本来能落下的小 Pod 都放不进去了。

另一个误区是把探针当“重启按钮”。`livenessProbe` 适合检测死锁、线程卡死、事件循环不再推进这类“重启有意义”的故障；如果只是下游数据库暂时超时，更合理的动作通常是 readiness 失败、先摘流量，而不是把自己重启到更不稳定。

---

## 替代方案与适用边界

Sidecar 不是默认答案，只是在“强耦合、同生命周期、同网络/卷环境”时最合适。

| 方案 | 适合场景 | 不适合场景 |
|---|---|---|
| Sidecar | 代理、证书刷新、同 Pod 日志处理、本地共享缓存 | 辅助功能与业务生命周期无强绑定 |
| DaemonSet | 节点级日志采集、节点监控、CNI/CSI 组件 | 需要按业务 Pod 单独隔离配置 |
| 独立服务 | 共享网关、集中鉴权、统一转码 | 需要 `localhost` 紧耦合通信 |
| CSI/平台能力 | 存储挂载、密钥注入、节点能力复用 | 需要进程级伴生逻辑 |

初学版决策规则可以很简单：

- 需要和主应用共享 `localhost`、共享卷、共享退出节奏，优先考虑 sidecar
- 需求是“整台节点只跑一份”，优先考虑 DaemonSet
- 需求本质是“全局共享能力”，优先考虑独立服务或平台组件

例如，若只是把宿主机日志统一采集走，DaemonSet 往往比每个 Pod 塞一个日志 sidecar 更省资源、更易治理；但如果日志必须先和主应用共享本地卷，再做按业务实例的实时加工，sidecar 仍然更合理。

---

## 参考资料

下面这些资料足够形成一条完整学习链：先看官方定义，再看资源与探针，再看真实 sidecar 的优雅退出实践。

| 来源名称 | 内容焦点 | 用途 |
|---|---|---|
| [Kubernetes Pods](https://kubernetes.io/docs/concepts/workloads/pods/) | Pod 的基本模型与共享语义 | 建立总体概念 |
| [Kubernetes Sidecar Containers](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/) | sidecar 角色、资源统计、与 init 的关系 | 理解 Pod 内多容器协作 |
| [Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) | `requests`、`limits`、调度与运行约束 | 搞清资源语义 |
| [Pod Quality of Service Classes](https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/) | Guaranteed / Burstable / BestEffort | 理解驱逐与稳定性边界 |
| [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) | Pod 生命周期、终止过程、探针作用 | 连接调度与运行时行为 |
| [Container Lifecycle Hooks](https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/) | `PostStart`、`PreStop` 的执行时机 | 配置优雅退出 |
| [Configure Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/) | 三类探针的配置与区别 | 调参时查具体字段 |
| [Linkerd Graceful Pod Shutdown](https://linkerd.io/2-edge/tasks/graceful-shutdown/) | sidecar 代理在优雅退出中的处理方式 | 对照真实工程场景 |

快速利用这些资料的顺序建议是：先看 Pod 和 sidecar 定义，再看资源管理和 QoS，最后看 lifecycle、probes 和 Linkerd 的关闭顺序。这样不会只会背字段，而是能把“为什么这样配”也一起串起来。
