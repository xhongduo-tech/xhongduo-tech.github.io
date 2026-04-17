## 核心结论

Prometheus 与 Grafana 的组合，本质上是在解决一件事：把“系统现在发生了什么”变成可以持续记录、可查询、可告警的时间序列，再把这些结果展示给人和通知给系统。

Prometheus 是**时序数据库**，白话说，就是专门存“某个数值随时间怎么变化”的数据库。它负责定期从目标服务拉取指标、按标签存储、再用 PromQL 查询。Grafana 是**可视化与告警平台**，白话说，就是把这些数字画成图、做成仪表盘，并在满足条件时发出通知。

对初级工程师最重要的结论有三条：

1. 监控不是先画图，而是先定义指标。
2. Prometheus 的核心价值不是“存数据”，而是“按标签聚合和查询数据”。
3. Grafana 的核心价值不是“漂亮面板”，而是把查询、展示、告警、通知串成闭环。

一个最小可工作的链路可以写成：

$$
\text{Exporter / App} \rightarrow \text{/metrics} \rightarrow \text{Prometheus Scrape} \rightarrow \text{TSDB} \rightarrow \text{PromQL} \rightarrow \text{Grafana Dashboard / Alert}
$$

玩具例子：Prometheus 每隔 15 秒抓取一台机器的 CPU 使用率，Grafana 把最近 1 小时的 CPU 曲线画出来；如果 CPU 连续 5 分钟超过 80%，就触发 `HighCPUUsage` 告警。

真实工程例子：一套在线模型服务同时暴露系统指标、接口延迟、推理吞吐、错误率、业务转化率。Prometheus 统一采集，Grafana 用一个仪表盘同时观察“机器是否顶满”“接口是否变慢”“业务是否掉单”，这样才能判断问题到底出在资源、程序还是业务链路。

---

## 问题定义与边界

这里讨论的“监控系统”，目标不是做全能运维平台，而是建立一条清晰的数据链路：**采集 -> 存储 -> 查询 -> 展示 -> 告警**。

“指标”这个词首次出现时要说明：指标就是一个可以量化的数字，例如 CPU 使用率、请求数、错误数。Prometheus 主要处理的是这类数值型时间序列，不直接解决日志检索、调用链追踪、自动修复等问题。

本文边界如下：

| 对象 | 监控内容 | 工具角色 | 本文是否覆盖 |
| --- | --- | --- | --- |
| 主机/容器 | CPU、内存、磁盘、网络 | Prometheus 抓取 exporter 指标 | 是 |
| 应用服务 | 延迟、吞吐、错误率、队列长度 | Prometheus 抓取应用 `/metrics` | 是 |
| 业务系统 | 下单数、支付成功率、转化率 | 应用埋点后暴露指标 | 是 |
| 日志系统 | 全文检索、错误上下文 | Loki/ELK 等 | 否 |
| 链路追踪 | 请求经过哪些服务 | Jaeger/Tempo 等 | 否 |
| 自动化运维 | 自动扩容、自动恢复 | K8s/HPA/运维平台 | 否 |

因此，问题定义可以收缩成一句话：我们要为一组服务建立可观测的指标闭环，重点关心性能、可用性和业务变化，并在异常时及时通知。

一个常见新手误区是把“监控”理解成“只看机器状态”。这不够。因为机器健康不代表服务健康，服务健康也不代表业务健康。一个模型 API 可能 CPU 很低，但因为外部依赖超时导致响应时间暴涨；也可能接口一切正常，但某次版本变更让转化率明显下降。所以实际监控至少分三层：

- 系统指标：机器和容器是否健康。
- 应用指标：服务是否能稳定处理请求。
- 业务指标：系统是否真的在创造业务结果。

---

## 核心机制与推导

Prometheus 默认采用 **pull 模型**，白话说，就是 Prometheus 主动去目标地址抓数据，而不是目标自己把数据推过来。典型过程是：

1. 应用或 exporter 在 `/metrics` 暴露文本格式指标。
2. Prometheus 按配置周期执行 scrape，例如每 15 秒一次。
3. 每次抓取结果形成带标签的时间序列。
4. 查询时用 PromQL 聚合、过滤、计算。
5. 告警规则周期性执行查询，满足条件后进入告警状态。
6. Grafana 读取 Prometheus 结果画图，或自己评估告警并发送通知。

这里的“标签”首次出现时要说明：标签就是给指标附加的维度，例如 `instance="10.0.0.1:9100"`、`job="node"`、`method="POST"`。它让同一个指标名可以区分不同机器、接口或环境。

例如 `node_cpu_seconds_total` 表示 CPU 各模式累计运行时间。它通常带有 `cpu` 和 `mode` 标签。若要估算 CPU 忙碌率，可以先算单位时间内 idle 增量，再用总量减掉 idle：

$$
\text{CPU Busy} = 100 \times \left(1 - \text{avg by(instance)}(\text{rate}(node\_cpu\_seconds\_total\{mode="idle"\}[5m]))\right)
$$

这里 `rate()` 的白话解释是：看一个累计计数器在某个时间窗口里的平均增长速度。因为 CPU 时间是不断累计的，所以不能直接看原值，要看增长速率。

再看一个按 CPU 核心统计忙碌率的 PromQL：

```promql
sum by(cpu) (rate(node_cpu_seconds_total{mode!="idle"}[1m]))
```

它的意思是：对每个 `cpu` 标签，把非 idle 模式的增长率加起来。这个查询适合观察单核是否热点过高。

告警机制可以理解成“定时判断条件是否持续成立”。例如：

- 查询表达式：CPU 使用率 > 80%
- 持续时间：`for: 5m`
- 评估间隔：每 1 分钟

只有连续 5 分钟每次都满足，告警才真正触发。`for` 的作用是抗抖动，白话说，就是避免瞬时毛刺把人吵醒。

玩具例子：一台测试机因为执行编译任务，CPU 在 20 秒内升到 95%，但很快回落。如果没有 `for`，会产生误报；加上 `for: 5m` 后，这个短时尖峰不会触发告警。

真实工程例子：在线推荐模型服务在晚高峰出现延迟升高。面板上同时看到 `cpu_usage` 稳定、`request_rate` 上升、`p99_latency` 激增、`error_rate` 略升。这个组合通常说明不是机器被打满，而是某个依赖或线程池进入排队。监控的价值就在于通过多维指标做因果缩小，而不是只盯一条曲线。

---

## 代码实现

下面给出一个最小但工程上可用的实现片段。

Prometheus 抓取配置：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 1m

rule_files:
  - /etc/prometheus/rules/node_rules.yml

scrape_configs:
  - job_name: node_exporter
    static_configs:
      - targets:
          - 10.0.0.11:9100
          - 10.0.0.12:9100
        labels:
          env: prod

  - job_name: model_service
    metrics_path: /metrics
    static_configs:
      - targets:
          - 10.0.1.21:8080
          - 10.0.1.22:8080
        labels:
          service: inference-api
          env: prod
```

Prometheus 告警规则：

```yaml
groups:
  - name: infra_rules
    rules:
      - alert: HighCPUUsage
        expr: 100 * (1 - avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))) > 80
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Instance CPU usage is high"
          description: "CPU usage on {{ $labels.instance }} is above 80% for 5 minutes"

      - alert: HighP99Latency
        expr: histogram_quantile(0.99, sum by(le, instance) (rate(http_request_duration_seconds_bucket[5m]))) > 0.8
        for: 10m
        labels:
          severity: critical
          team: ml-serving
        annotations:
          summary: "P99 latency is high"
          description: "P99 latency on {{ $labels.instance }} is above 800ms"
```

Grafana 面板查询通常直接写 PromQL，例如：

```promql
100 * (1 - avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])))
```

如果使用 Grafana Unified Alerting，一个简化后的规则结构可以写成：

```json
{
  "title": "HighCPUUsage",
  "condition": "A",
  "data": [
    {
      "refId": "A",
      "datasourceUid": "prometheus-main",
      "model": {
        "expr": "100 * (1 - avg by(instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))",
        "intervalMs": 60000,
        "maxDataPoints": 43200
      }
    }
  ],
  "noDataState": "NoData",
  "execErrState": "Error",
  "for": "5m",
  "annotations": {
    "summary": "CPU usage above 80%"
  },
  "labels": {
    "severity": "warning"
  }
}
```

下面用一个可运行的 Python 片段模拟“连续 5 分钟超过阈值才告警”的逻辑：

```python
def should_fire_alert(values, threshold=80, required_points=5):
    """
    values: 最近 N 分钟的 CPU 百分比，每分钟一个点
    """
    if len(values) < required_points:
        return False
    tail = values[-required_points:]
    return all(v > threshold for v in tail)

# 玩具例子：只有最后 3 个点超阈值，不告警
samples_a = [45, 62, 81, 85, 88]
assert should_fire_alert(samples_a, threshold=80, required_points=5) is False

# 连续 5 个点都超阈值，告警
samples_b = [82, 84, 86, 88, 91]
assert should_fire_alert(samples_b, threshold=80, required_points=5) is True

# 有一个点回落，也不告警
samples_c = [82, 84, 79, 88, 91]
assert should_fire_alert(samples_c, threshold=80, required_points=5) is False
```

这个例子虽然简化，但对应的就是 Prometheus/Grafana 中 `expr + for` 的基本判断模型。

---

## 工程权衡与常见坑

Prometheus 与 Grafana 上手很快，但真正上线后，问题通常不在“会不会配”，而在“指标设计是否收敛”。

最典型的问题是**高基数**。高基数的白话解释是：标签取值太多，导致时间序列数量爆炸。比如你给每个请求都加 `user_id` 标签，百万用户就可能制造百万条序列。Prometheus 的内存和查询性能会被直接拖垮。

| 坑 | 后果 | 规避措施 |
| --- | --- | --- |
| 给指标加 `user_id`、`order_id` 这类动态标签 | 时间序列爆炸，Prometheus OOM | 只保留有限枚举标签，把明细问题交给日志系统 |
| 抓取间隔太短 | 存储量和查询负担上升 | 大多数系统指标用 15s 到 60s 即可 |
| 告警没有 `for` | 毛刺引发误报 | 对资源类告警设置 3 到 10 分钟缓冲 |
| 只监控系统指标，不监控应用与业务 | 找不到真正故障点 | 至少补齐延迟、吞吐、错误率三件套 |
| 直方图桶设计不合理 | P95/P99 失真 | 按业务延迟量级设计 bucket |
| 面板过多但缺乏总览 | 故障时找不到重点 | 先做总览页，再做问题定位页 |
| `No Data` 语义不清 | 无法区分“服务挂了”和“暂时没流量” | 给空闲服务定义心跳指标，明确 no data 策略 |

这里有一个重要权衡：标签越多，查询维度越灵活；但标签越多，成本越高。工程上应该优先保留稳定且有限的标签，例如 `instance`、`job`、`service`、`env`、`status_code`。不要把日志系统该做的事塞进指标系统。

另一个常见坑是“直接抄官方面板”。官方面板适合演示，不一定适合你的业务。真正有用的面板通常按排障顺序设计：

1. 总览页：服务是否整体异常。
2. 资源页：CPU、内存、磁盘、网络是否顶满。
3. 应用页：延迟、吞吐、错误率是否异常。
4. 业务页：关键业务指标是否偏移。

真实工程里，告警也要分级。CPU 85% 持续 5 分钟，通常是 `warning`；接口错误率 20% 持续 2 分钟，可能就是 `critical`。如果所有告警都发到同一个群，最后结果往往是没人看。

---

## 替代方案与适用边界

Prometheus 不是唯一方案，但它在云原生和动态服务环境里很强，原因在于 pull 模型配合服务发现更自然。Kubernetes 中 Pod 会频繁变化，Prometheus 通过服务发现自动更新目标列表，比手动推送稳定得多。

不过 pull 模型也有边界。短生命周期任务，比如几秒钟就结束的批处理作业，Prometheus 可能来不及抓到它的指标。这时可以使用 Pushgateway 作为中转，但要注意：Pushgateway 适合短任务上报结果，不适合长期服务替代正常 scrape。

| 方案 | 适合场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| Prometheus pull | 长期运行服务、K8s、容器平台 | 自动发现、统一抓取、模型简单 | 短任务不易抓到 |
| Pushgateway | 短批任务、定时作业 | 任务结束前也能上报结果 | 容易被误用成长期指标入口 |
| Grafana Native Alerts | 直接在 Grafana 做告警 | 配置集中，和面板结合紧密 | 依赖 Grafana 自身可用性 |
| Prometheus + Alertmanager | 大规模告警分发 | 路由、抑制、聚合能力强 | 链路更长，组件更多 |

如果团队规模小，Grafana 自带告警已经够用；如果团队大、告警路由复杂、需要去重和抑制，就更适合 Prometheus 规则配合 Alertmanager。

一个简化判断标准是：

- 服务长期运行：优先 Prometheus scrape。
- 任务很短：考虑 Pushgateway。
- 告警逻辑简单：Grafana native alerts 足够。
- 通知链复杂、多团队值班：Alertmanager 更稳。

因此，Prometheus + Grafana 并不是“万能监控标准答案”，而是在大多数 Web 服务、微服务、模型服务场景下，复杂度与收益比较平衡的一组方案。

---

## 参考资料

- Prometheus 官方首页：https://prometheus.io/
  主题：Prometheus 的整体架构与定位。

- Prometheus Instrumentation Best Practices：https://prometheus.io/docs/practices/instrumentation/
  主题：指标命名、标签设计、埋点策略。

- Prometheus Querying Basics：https://prometheus.io/docs/prometheus/latest/querying/basics/
  主题：PromQL 基本语义与查询模型。

- Grafana Alerting Fundamentals：https://grafana.com/docs/grafana/latest/alerting/fundamentals/
  主题：告警规则、告警实例、通知链路。

- Grafana Alerting Get Started：https://grafana.com/tutorials/alerting-get-started-pt6/
  主题：Prometheus、Grafana、Alertmanager 的最小实践链路。

- Yandex Cloud Prometheus Alerting Rules Example：https://yandex.cloud/en/docs/monitoring/operations/prometheus/alerting-rules
  主题：CPU 告警规则示例与持续时间配置。
