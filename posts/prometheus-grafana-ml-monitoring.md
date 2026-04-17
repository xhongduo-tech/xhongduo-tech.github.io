## 核心结论

Prometheus + Grafana 的核心价值，不是“把很多图画出来”，而是把监控链路拆成了四个职责清晰的部分：指标暴露、采集存储、查询展示、告警分发。

先给结论：

| 组件 | 角色 | 关键能力 |
| --- | --- | --- |
| Prometheus | 时序数据库，按周期抓取指标 | Pull `/metrics`、写入时序数据、生成 `up` |
| Grafana | 可视化层 | 把 PromQL 封装成图表、变量、仪表盘 |
| Alertmanager | 告警路由层 | 去重、分组、抑制、分发到邮件/IM/电话 |
| dcgm-exporter | GPU 指标暴露器 | 暴露显存、温度、功率、利用率等 NVIDIA GPU 指标 |

Prometheus 仍然以 pull 模型为核心。pull 模型可以白话理解为：不是业务主动“上报”给监控，而是 Prometheus 定时去目标机器上“拉”指标。目标只需要在 `/metrics` 提供一个 HTTP 文本接口，Prometheus 每隔 `scrape_interval` 访问一次。访问成功时，这个目标的 `up=1`；访问失败时，`up=0`。这件事非常重要，因为它把“监控系统能否连上目标”也变成了可查询的数据。

对 GPU 集群监控来说，这套模型尤其合适。训练节点长期运行，dcgm-exporter 可以稳定暴露 GPU 指标，Prometheus 周期采集后，Grafana 负责按 `instance`、`gpu`、`job` 维度做筛选和展示，Alertmanager 再把阈值告警发到合适的接收者。

如果只记三句话，可以记住：

1. Prometheus 负责“拉、存、查”，Grafana 负责“看”，Alertmanager 负责“发”。
2. `up` 不是业务指标，但它是排障时最先看的系统健康指标。
3. GPU 监控面板不是只看利用率，至少要同时看显存、温度、功耗和节点可达性。

---

## 问题定义与边界

本文讨论的对象，是分布式训练或多 GPU 节点环境中的基础监控系统。边界要先说清楚，否则很容易把“监控平台搭好了”和“真正监控到了关键问题”混为一谈。

这里的“监控”主要覆盖三类问题：

| 类别 | 典型指标 | 监控目的 |
| --- | --- | --- |
| 节点可达性 | `up`、采集时延 | 判断 exporter 和网络是否正常 |
| GPU 资源状态 | 显存、温度、功率、利用率 | 判断训练是否打满资源、是否过热 |
| 训练运行状态 | 吞吐、step time、自定义业务指标 | 判断模型训练是否退化或卡住 |

边界也很明确：

1. Prometheus 默认适合长期运行、可被 HTTP 访问的目标。
2. 如果目标在 NAT、防火墙后面，Prometheus 拉不到，就不是 PromQL 写得好不好的问题，而是网络可达性问题。
3. dcgm-exporter 只能提供 DCGM 暴露出来的 GPU 指标，不会自动知道你的训练 loss、tokens/s、queue depth，这些需要额外的业务 exporter 或应用内埋点。

“Pull 模型”首次出现时可以这样理解：监控服务器主动去找你，而不是你主动去找监控服务器。这个设计的优点是采集链路简单，缺点是必须保证 Prometheus 到目标网络可达。

一个玩具例子：

假设你只有两台训练节点：

- `node1:9400`
- `node2:9400`

Prometheus 每 15 秒抓一次：

- 如果 `node1` 正常返回 `/metrics`，则 `up{instance="node1:9400"} = 1`
- 如果 `node2` 的 exporter 进程挂了，或者端口被防火墙拦住，则 `up{instance="node2:9400"} = 0`

这时你还没看任何 GPU 图表，就已经知道“监控数据本身是否可信”。因为如果目标根本没被成功采集，那后面的温度和显存图都是空的，问题首先不在模型，而在采集链路。

一个真实工程例子：

在 8 台 A100 节点组成的分布式训练集群里，常见需求不是“看单卡使用率”，而是回答下面这些问题：

- 某个训练任务是不是只打满了 6 台机器，另外 2 台实际掉线？
- 某一台机器是不是温度过高后自动降频？
- 为什么 global batch size 没变，但吞吐下降了 18%？
- 某个版本上线后，显存占用为什么持续抬高？

这些问题都要求监控系统同时具备：

- 节点级视角：按 `instance` 看机器差异
- GPU 级视角：按 `gpu` 看卡间不均衡
- 时间窗口视角：看瞬时值，也看 5 分钟、30 分钟趋势
- 告警视角：在真正异常时自动通知

因此，本文的边界不是“可观测性全家桶”，而是以 Prometheus 为中心，解决 GPU 训练环境中的基础采集、查询、展示和告警问题。

---

## 核心机制与推导

Prometheus 存的是时序数据。时序数据可以白话理解为：同一个指标，在不同时间点的采样值序列。比如 GPU 0 的显存使用量，10:00 是 8 GiB，10:01 是 10 GiB，10:02 是 12 GiB，这就是一条时间序列。

PromQL 是查询这些时序数据的语言。它不是 SQL，而是更接近“对一组时间序列做过滤、聚合、函数运算”。

先看 GPU 显存利用率的基本推导。dcgm-exporter 通常会暴露已用显存和空闲显存。显存利用率可以写成：

$$
mem\_util = \frac{used}{used + free}
$$

对应到指标名称，就是：

$$
mem\_util = \frac{DCGM\_FI\_DEV\_FB\_USED}{DCGM\_FI\_DEV\_FB\_USED + DCGM\_FI\_DEV\_FB\_FREE}
$$

这个公式的意义很直接：显存总量 = 已用 + 空闲，因此占用率就是已用占总量的比例。

玩具例子：

如果某张卡上：

- `DCGM_FI_DEV_FB_USED = 8192`
- `DCGM_FI_DEV_FB_FREE = 8192`

那么：

$$
mem\_util = \frac{8192}{8192 + 8192} = 0.5
$$

也就是 50%。

在 PromQL 里，可以直接写：

```promql
DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)
```

但实际工程里，通常不会只看瞬时值，因为训练过程会抖动。更常见的是看区间聚合，比如 5 分钟平均值：

```promql
avg_over_time(
  (
    DCGM_FI_DEV_FB_USED{job="gpu-dcgm"}
    /
    (DCGM_FI_DEV_FB_USED{job="gpu-dcgm"} + DCGM_FI_DEV_FB_FREE{job="gpu-dcgm"})
  )[5m]
)
```

这里 `avg_over_time` 的意思可以白话解释为：对过去 5 分钟的采样点求平均。这样比单点值更适合面板展示和告警判断。

PromQL 常见查询可以分为三类：

| 查询类型 | 含义 | 例子 |
| --- | --- | --- |
| 过滤 | 取出满足标签条件的序列 | `up{job="gpu-dcgm"}` |
| 聚合 | 对多个序列做统计 | `avg by (instance) (...)` |
| 区间函数 | 对一段时间窗口计算 | `avg_over_time(...[5m])` |

再看 `up` 的机制。`up` 是 Prometheus 自动生成的指标，不需要 exporter 自己暴露。它表示“最近一次抓取是否成功”。这意味着：

- `up=1` 不代表业务正常，只代表采集成功
- `up=0` 往往先说明 exporter、网络、DNS、TLS 或认证有问题
- 任何业务指标分析之前，都应该先确认 `up`

这是 pull 模型的一个关键优势。因为采集失败本身就变成了可查询的数据，不需要再额外写“心跳上报”逻辑。

真实工程里，Grafana 的分布式训练面板通常至少拆成三层：

1. 集群总览
   - 在线 GPU 数
   - `up` 异常节点数
   - 平均显存利用率
   - 平均功耗
2. 节点明细
   - 按 `instance` 展示每台机器的 GPU 温度、显存、功耗
3. GPU 明细
   - 按 `instance` + `gpu` 展示单卡差异，定位某一张卡异常

例如，一个训练任务吞吐下降，但 GPU 利用率表面看都在 95% 左右。如果切到节点明细面板，可能会发现其中一台机器的功耗明显偏低、温度偏高，说明不是“GPU 没忙”，而是“GPU 在忙但性能状态异常”，这就是多指标联合分析的意义。

---

## 代码实现

下面给出一套最小可运行思路，包括 Prometheus 采集、Alertmanager 路由，以及一个用 Python 演示显存利用率计算的玩具脚本。

Prometheus 采集配置示例：

```yaml
global:
  scrape_interval: 15s
  scrape_timeout: 10s

scrape_configs:
  - job_name: gpu-dcgm
    static_configs:
      - targets:
          - node1:9400
          - node2:9400
          - node3:9400
        labels:
          role: training-gpu
```

这段配置的意思很直接：

- 每 15 秒抓一次
- 每次抓取最多等 10 秒
- 目标是三个 GPU 节点的 dcgm-exporter
- 给这些目标统一打上 `role="training-gpu"` 标签

如果要做 Grafana 面板，一个很实用的查询是按节点看过去 5 分钟的平均显存利用率：

```promql
avg by (instance) (
  avg_over_time(
    (
      DCGM_FI_DEV_FB_USED{job="gpu-dcgm"}
      /
      (DCGM_FI_DEV_FB_USED{job="gpu-dcgm"} + DCGM_FI_DEV_FB_FREE{job="gpu-dcgm"})
    )[5m]
  )
)
```

如果要看异常节点，可以直接查：

```promql
up{job="gpu-dcgm"} == 0
```

如果要看过去 1 分钟内采集成功率过低的节点，可以写：

```promql
avg_over_time(up{job="gpu-dcgm"}[1m]) < 0.5
```

这比“单次失败就告警”更稳，因为网络偶发抖动很常见。

Alertmanager 最小路由示例：

```yaml
route:
  receiver: team-email
  group_by: ['alertname', 'instance']
  routes:
    - matchers:
        - severity="warning"
      receiver: team-chat
      continue: true
    - matchers:
        - severity="critical"
      receiver: oncall-pager

receivers:
  - name: team-email
  - name: team-chat
  - name: oncall-pager

inhibit_rules:
  - source_matchers:
      - severity="critical"
    target_matchers:
      - severity="warning"
    equal: ['alertname', 'instance']
```

这里有三个容易忽略的点：

1. 根 `route` 不应该带匹配条件，它要接住所有告警。
2. `continue: true` 表示命中当前子路由后，继续向后匹配，而不是到此停止。
3. `inhibit_rules` 是抑制规则，意思是：如果同一个 `alertname`、同一个 `instance` 上已经有 `critical`，那么对应的 `warning` 就不要再发，避免重复轰炸。

下面是一个可运行的 Python 玩具脚本，用来演示显存利用率计算和简单告警判断：

```python
from dataclasses import dataclass

@dataclass
class GpuSample:
    used_mib: int
    free_mib: int
    temperature_c: int
    power_w: int

def mem_util(sample: GpuSample) -> float:
    total = sample.used_mib + sample.free_mib
    assert total > 0, "total memory must be positive"
    return sample.used_mib / total

def should_alert(sample: GpuSample) -> bool:
    util = mem_util(sample)
    return util > 0.95 and sample.temperature_c > 85

toy = GpuSample(used_mib=8192, free_mib=8192, temperature_c=70, power_w=210)
assert abs(mem_util(toy) - 0.5) < 1e-9
assert should_alert(toy) is False

realistic = GpuSample(used_mib=38912, free_mib=1024, temperature_c=88, power_w=315)
assert mem_util(realistic) > 0.97
assert should_alert(realistic) is True

print("ok")
```

这个脚本对应的工程含义是：

- 玩具例子里，显存占用 50%，温度 70°C，不需要告警。
- 真实工程例子里，一张 40GiB 级别的卡显存几乎打满，温度达到 88°C，就可以视为需要重点关注的状态。

当然，生产环境不会用 Python 直接从 Prometheus 拉值后做判断，而是把阈值写成 Prometheus 规则。但这个脚本可以帮助新手先把公式和判断逻辑理清楚。

一个更贴近生产的告警规则如下：

```yaml
groups:
  - name: gpu-alerts
    rules:
      - alert: GpuExporterDown
        expr: avg_over_time(up{job="gpu-dcgm"}[1m]) < 0.5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU exporter unavailable on {{ $labels.instance }}"

      - alert: GpuMemoryHigh
        expr: (
          DCGM_FI_DEV_FB_USED{job="gpu-dcgm"}
          /
          (DCGM_FI_DEV_FB_USED{job="gpu-dcgm"} + DCGM_FI_DEV_FB_FREE{job="gpu-dcgm"})
        ) > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high on {{ $labels.instance }}"
```

这类规则的重点不是“语法写出来”，而是时间窗口和阈值要和你的训练模式匹配。

---

## 工程权衡与常见坑

Prometheus + Grafana 很强，但不是“配置完就万事大吉”。真正的难点在工程权衡。

第一类坑是采集频率设置不合理。

如果 `scrape_interval=5s`，但你的 exporter 偶尔 8 秒才响应一次，那么 Prometheus 会频繁采集失败，`up` 反复变成 0。这时问题不一定是服务挂了，而是你把采样频率设得比系统实际响应能力还激进。

常见规避方式：

| 问题 | 根因 | 规避方式 |
| --- | --- | --- |
| `up` 频繁抖动 | `scrape_interval` 太短或网络抖动 | 拉长采集周期，检查 `/targets` 页面 |
| 告警风暴 | 单次失败即告警 | 使用 `avg_over_time` 和 `for` 窗口 |
| 图表“锯齿感”严重 | 只看瞬时值 | 改看 5 分钟或 15 分钟聚合 |

第二类坑是把 `up=1` 误解成“服务健康”。

`up=1` 只表示 Prometheus 成功拉到了 `/metrics`。它不代表训练一定正常，也不代表数据一定正确。一个 exporter 完全可能活着，但 GPU 采集线程卡住，导致指标不更新。工程里通常还会补充看：

- 指标时间戳是否推进
- 关键指标是否长时间恒定不变
- 业务指标是否同步变化

第三类坑是 Grafana 面板设计只看“均值”。

均值会掩盖偏斜。比如 8 张卡里 7 张利用率 98%，1 张只有 10%，平均值仍然很高，但训练实际上已经不均衡。对分布式训练，至少要有下面几类面板：

- 按 `instance` 的节点级面板
- 按 `instance,gpu` 的单卡级面板
- 带变量过滤器的任务级面板
- 同时展示显存、温度、功耗，而不是只放 utilization

第四类坑是 Alertmanager 根路由配置错误。

根 `route` 的职责是接收所有告警并作为路由树入口。如果你在根路由就加上 `matchers`，那么不匹配的告警可能直接走默认处理甚至丢失预期路径。另一个常见误区是忘了写 `continue: true`，导致告警只进入第一个匹配分支，后续接收者收不到。

第五类坑是抑制规则写得过粗。

比如只写：

```yaml
equal: ['alertname']
```

在某些环境下可能过度抑制。更稳妥的方式通常是把 `instance`、`job` 甚至 `gpu` 也纳入相等条件，确保“同一对象上的高级别告警抑制低级别告警”，而不是跨对象误伤。

真实工程例子：

某团队给 16 台训练节点设置了 `scrape_interval: 5s`，同时在 Alertmanager 中把“连续 1 次 `up=0`”定义为 critical。结果是白天网络轻微波动时，监控群几乎每 10 分钟就被刷屏一次。后来他们做了三件事后才稳定：

1. 把 `scrape_interval` 调整到 15 秒。
2. 把告警表达式改成 `avg_over_time(up[1m]) < 0.5`。
3. 增加 `for: 2m`，过滤短暂抖动。

这说明监控系统不是越敏感越好，而是要把采集节奏、系统噪声和告警成本一起考虑。

---

## 替代方案与适用边界

Prometheus 的 pull 模型不是唯一方案，只是对长期运行的服务最合适。

最常被提到的替代方案是 Pushgateway。它的白话解释是：短命任务没法一直暴露 HTTP，就先把结果推给一个中间站，Prometheus 再去抓这个中间站。

适用边界如下：

| 方案 | 适用场景 | 主要代价 |
| --- | --- | --- |
| Prometheus Pull | 长期运行服务、exporter、节点监控 | 需要网络可达 |
| Pushgateway | 短命批处理、定时任务 | 没有天然 `up`，需要清理陈旧数据 |
| PushProx | Prometheus 无法直连目标 | 架构更复杂 |
| 托管监控平台 | 不想自运维 Prometheus/Grafana/Alertmanager | 灵活性和控制力下降 |

Pushgateway 适合什么？适合“任务结束就没了”的场景，例如夜间模型清理、离线数据校验、一次性特征回填。因为这类任务结束后，进程退出，Prometheus 不一定能在任务存活期间及时抓到它。

但 Pushgateway 不适合替代节点级 GPU 监控。因为 GPU 节点是长期运行的，pull 模型天然更合适，而且 `up` 能直接暴露采集链路问题。你如果把长期服务也改成 push，反而失去了 Prometheus 原本最有价值的一部分。

另一个替代思路是让 Prometheus 部署得更靠近目标网络，而不是强行让目标“推出来”。如果训练节点在私有网络中，通常更合理的做法是：

- 把 Prometheus 也部署到私有网络内
- 或者使用 PushProx 这类穿透方案
- 而不是先把所有指标汇总推送到外部

真实工程里还有一种边界：当团队不想维护 Prometheus、Grafana、Alertmanager 三套组件时，可能会选择托管 Observability 平台。这类平台的优点是部署快、统一管理；缺点是查询模型、存储策略、保留周期、告警编排的控制力通常不如原生 Prometheus 体系。

因此，一个实用判断标准是：

- 如果你需要对 PromQL、抓取策略、告警路由有很强控制，优先原生栈。
- 如果你只是想“先把监控跑起来”，且团队运维资源有限，可以考虑托管方案。
- 如果你的任务是短命批处理，用 Pushgateway；如果是长期 GPU 节点，继续坚持 pull。

---

## 参考资料

- Prometheus 文档总览：https://prometheus.io/docs/
- Prometheus 配置文档（`scrape_interval`、`scrape_configs`）：https://prometheus.io/docs/prometheus/3.3/configuration/configuration/
- Prometheus 查询基础与 PromQL：https://prometheus.io/docs/prometheus/latest/querying/basics/
- Prometheus 函数文档（`avg_over_time` 等）：https://prometheus.io/docs/prometheus/3.1/querying/functions/
- Prometheus 关于 Pushgateway 的实践建议：https://prometheus.io/docs/practices/pushing/
- Prometheus Alertmanager 配置文档（`route`、`continue`、`inhibit_rules`）：https://prometheus.io/docs/alerting/latest/configuration/
- Prometheus Agent / Pull 模型相关说明：https://prometheus.io/blog/2021/11/16/agent/
- Grafana 官方文档，Prometheus 数据源与查询基础：https://grafana.com/docs/grafana/latest/fundamentals/intro-to-prometheus/
- NVIDIA DCGM Exporter 官方文档：https://docs.nvidia.com/datacenter/dcgm/3.1/gpu-telemetry/dcgm-exporter.html
- NVIDIA DCGM Exporter Grafana Dashboard：https://grafana.com/grafana/dashboards/24448-nvidia-dcgm-exporter-dashboard/
- PromLabs 关于 Prometheus 状态页与 target 排障：https://training.promlabs.com/training/monitoring-and-debugging-prometheus/web-status-pages/target-scrape-status/
- 关于 `up` 指标行为的讨论：https://stackoverflow.com/questions/55162188/prometheus-how-up-metrics-works
