---
title: 可观测性（Prometheus/OpenTelemetry）
date: 2026-08-11
---

# 可观测性（Prometheus/OpenTelemetry）

<div class="epigraph">
<p>监控回答「它坏了吗」，可观测性回答「它为什么坏了」——而后者才让人能修好它。</p>
<footer>—— 意译自 Charity Majors（Honeycomb 联合创始人）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么最后一课是「看」系统

前面十五课把系统「造」了出来：容器、编排、存储、网络、安全、发布。但一个跑在云上的系统，最大的不确定性来自「它实际怎么样了」——副本真的在吗？时延是多少？哪个服务在拖累谁？日志去哪了？**可观测性**是把系统「透明化」的能力：它不改变系统行为，却决定了你能否排障、扩容、决策。它是云原生世界里「系统知道自己活得好不好」的那面镜子。

## 1 三大支柱：指标、日志、追踪

可观测性的三个数据源各回答一个问题：

- **指标（Metrics）**：**系统现在的状态数值**——RPS、时延、错误率、CPU。可聚合、可告警，是「系统体温计」。
- **日志（Logs）**：**发生了什么事件的记录**——一条条带时间戳的事件。可检索、可排查，是「系统日记」。
- **追踪（Traces）**：**一次请求在多个服务间的路径与耗时**——跨服务串起来，是「系统心电图」。

三者互补：指标告诉你「哪里异常」，日志告诉你「发生了什么」，追踪告诉你「为什么是这条链路出了问题」。**核心概念：RED 方法论**——服务健康用三个指标衡量：**R**ate（请求速率）、**E**rrors（错误率）、**D**uration（耗时分布）；基础设施侧另有 **USE** 方法：**U**tilization（利用率）、**S**aturation（饱和度）、**E**rrors（错误）。<span class="marginnote">RED 与 USE 是「该观测哪些指标」的模板：对每个服务问「每秒多少请求、多少失败、多慢」，对每个资源问「用了多少、还多满、有没有错」——先回答这两组问题，再谈高级观测。</span>

## 2 Prometheus：指标的事实标准

**核心概念：Prometheus**：云原生监控的事实标准，核心设计是**拉取（pull）模型**：

- 应用暴露 `/metrics` 端点（自带 client library 或通过 exporter 代理），Prometheus 定期**主动拉取**并存储为时间序列。
- 查询语言 **PromQL** 对时序数据做过滤、聚合、运算。
- 告警：**Alertmanager** 按规则（如 `rate(http_requests_total[5m]) > 100`）发送通知。

```
rate(http_requests_total{job="web",status="5xx"}[5m]) / rate(http_requests_total{job="web"}[5m])
```

这条 PromQL 算「web 服务过去 5 分钟 5xx 占比」。**公式解析：率怎么算**：`rate(x[5m])` = 计数器在过去 5 分钟的增长量 ÷ 300 秒——把「累计值」变成「每秒速率」。

$$\text{rate}(x[t]) = \frac{x(t) - x(t - \Delta t)}{\Delta t}, \qquad \Delta t = 300\,\text{s}$$

- $x(t)$：计数器（如请求总数）在时刻 $t$ 的值。
- 分子：窗口起点与终点的差值（这段时间新增的数量）。
- 分母：窗口长度。

三步拆解：

- **第一步，计数器必须单调递增**：请求数、字节数这类指标用**计数器（counter）**表达，只增不减——才能算「增量」。
- **第二步，窗口平滑抖动**：`[5m]` 的窗口把秒级毛刺平均掉，得到**速率趋势**而非瞬时值——所以告警阈值要配「持续 N 分钟」，避免噪声误报。
- **第三步，RED 三件套都靠它**：Rate（`rate(req_total)`）、Errors（`rate(5xx)/rate(total)`）、Duration（`histogram_quantile(0.95, rate(latency_bucket[5m]))` 求 P95）。**看懂 PromQL 就是看懂「怎么把原始计数翻译成业务健康度」**。

## 3 OpenTelemetry：观测的「统一语言」

Prometheus 解决了指标，但日志与追踪各有一套 SDK、一套协议——生态碎片化。**OpenTelemetry（OTel）** 想做观测界的标准接口：

- **API/SDK**：语言无关的**统一接口**，应用只写一次埋点（生成 trace/metrics/logs），不绑定后端。
- **Collector**：一个独立组件，接收、处理、导出遥测数据——支持 OTLP（OpenTelemetry Protocol）到 Prometheus、Jaeger、Loki 等各种后端。
- **语义约定**：字段命名统一（如 `http.request.method`），让跨服务的查询一致。<span class="marginnote">OTel 的定位与 CSI/CNI/CRI 完全同构：<strong>接口标准化，实现百花齐放</strong>。应用埋点一次，后端随便换——这是「插件化哲学」在观测领域的又一次胜利。服务网格里的 Envoy 也在用 OTel 格式上报 trace，网格与观测在此合流。</span>

**辨析｜易错点：Prometheus 与 OTel 的关系**。它们不是竞争而是接力：**OTel 负责「采集与标准化」，Prometheus 负责「存储与查询」**。OTel Collector 可以把指标导出成 Prometheus 格式（甚至直接由 Prometheus 拉取）。选型建议：**新项目优先 OTel 埋点，指标落 Prometheus + Grafana，日志落 Loki/ELK，追踪落 Tempo/Jaeger**——一套打通。

## 4 可观测性在云原生的特别之处

云原生的动态性（Pod 随机调度、副本随时伸缩、金丝雀切换）让可观测性更难，也更必要：

- **对象即标签**：Prometheus 天然按 label 关联指标，Kubernetes 的标签（`namespace`、`pod`、`app`）直接变成查询维度——**这正是《对象模型》里 labels 设计被复用得最成功的地方**。
- **服务网格送数据**：所有流量经 Envoy，指标/trace 自动生成（见《服务网格》），微服务排障的「链路黑箱」被打开。
- **Kubernetes 层面的指标**：kubelet 暴露 `/metrics`，配合 kube-state-metrics（对象状态的指标化）与 node-exporter（节点资源）——覆盖节点、Pod、对象三层。

## 5 从监控到排障：把三根支柱连起来

一次典型排障流程，恰好用满三大支柱：Grafana 面板看到 `web` 服务 P95 时延飙高（指标）→ 打开追踪看该时段 `web → db` 调用的耗时分布（追踪）→ 发现数据库连接池等待，再看数据库节点日志（日志）→ 定位到慢查询。**指标给方向，追踪给路径，日志给细节**——三者配合才是可观测性，任何一根独木难支。

## 6 小结

- 可观测性三支柱：**指标**（状态数值）、**日志**（事件记录）、**追踪**（跨服务路径）；RED/USE 告诉你该测什么。
- **Prometheus** 用**拉取模型**采集指标，PromQL 把计数器翻译成速率/占比/分位值。
- `rate(x[t])` = 窗口内增量 ÷ 窗口长度；计数器单调递增、窗口平滑抖动。
- **OpenTelemetry** 是统一接口（API/SDK/Collector/OTLP），与 Prometheus 是「采集 vs 存储」的分工。
- 云原生的标签体系让指标天然可切片；服务网格自动生产可观测数据。
- 排障 = 指标定方向 → 追踪找路径 → 日志抠细节。

至此，我们从「容器为什么存在」一路走到「怎么知道系统活得好不好」，走完了容器与云原生的完整旅程。下一站，你可以带着这份地图回到第二级《分布式系统》与《操作系统》，把「机制」与「原理」重新对齐——或者继续向 AI 基础设施的部署层出发。
