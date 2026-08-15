---
title: 控制器与声明式 API
date: 2026-08-07
---

# 控制器与声明式 API

<div class="epigraph">
<p>不要把系统写成一张张操作指令，把它写成你想要的结果；让一个持续盯梢的循环去完成它。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 容器与云原生 ｜ Burns Ch.9 ｜ 2026-08-07</p>
</div>

## 为什么「控制器」是 K8s 的心脏

前面我们有了 Pod（一个实例）和 Service（发现一群实例），但「始终有 3 个副本」「升级不中断」「节点挂了自动补」这些**运维契约**还没有对象来兑现。兑现者就是**控制器（controller）**。它们是控制回环的具体实现，是声明式 API 的引擎。学完这一节，你就能看懂 Kubernetes 里一切「自动」背后那台不断转动的机器。

## 1 控制器是什么

**核心概念：控制器**：一个运行在控制平面里的**循环程序**，反复执行「观察（observe）→ 比较（compare）→ 修正（correct）」三个动作。它 watch API Server 上的对象变化，把实际状态拉向期望状态。

控制器不靠「事件驱动一次」，而是**无限循环**。以 ReplicaSet 控制器为例：

```
for {
  observed = list(Pods with selector)
  delta = spec.replicas - len(observed)
  if delta > 0: create(delta) pods
  if delta < 0: delete(-delta) pods
  sleep/await watch
}
```

即使集群什么都没发生，控制器也保持 watch 状态——任何一处偏离（有人误删了一个 Pod、节点宕机）都会立刻触发补差。这就是**自愈**的来源，也是声明式模型的执行机制。<span class="marginnote">控制器的哲学与「告警后人工介入」截然相反：它不等人来修，而是持续把现实拉回理想。用一句话概括：<strong>声明式 = 描述目标，控制器 = 永久负责</strong>。这套思路直接脱胎于控制论（cybernetics），与你在概率/随机过程课里见到的反馈回路同源。</span>

实现上的细节值得了解：控制器不直接反复查询 API Server，而是通过 **informer + 本地缓存（cache）** 订阅对象变化——`watch` 建立长连接收增量事件，`list` 全量同步一次后用增量事件维护本地缓存。这样高频对象（Pod）的增删改不会把 etcd/API Server 压垮，控制器读的是自己那份「本地投影」。**一切自动化都建立在「高效地知道发生了什么」之上**，informer 就是那个高效的耳目。

## 2 主力控制器家族

- **ReplicaSet**：保证「恰好 $n$ 个副本」，只懂数量，不懂更新。
- **Deployment**：包在 ReplicaSet 外面的「发布管理器」——管理滚动更新、回滚、暂停/恢复。Deployment 不直接管 Pod，而是**管 ReplicaSet**（新版本 = 新 ReplicaSet，渐进替换）。
- **StatefulSet**：给 Pod 稳定身份（`web-0`、`web-1`……）与稳定存储，用于数据库、消息队列等有状态应用（见《持久化存储》）。
- **DaemonSet**：保证「每个（符合条件的）节点上都跑一份」，如日志采集、网络插件、节点监控。
- **Job / CronJob**：一次性任务与定时任务，跑完 `Succeeded` 即终。

| 控制器 | 管什么 | 典型用途 |
| --- | --- | --- |
| ReplicaSet | 副本数量 | 数量保证 |
| Deployment | ReplicaSet + 发布 | 无状态应用上线 |
| StatefulSet | 稳定身份 + 稳定卷 | 数据库、消息队列 |
| DaemonSet | 每节点一份 | 日志/网络/监控代理 |
| Job / CronJob | 一次性 / 定时 | 迁移、备份、批处理 |

**辨析｜易错点：Deployment 与 ReplicaSet 的关系**。新手常以为 Deployment 直接管理 Pod。实际上链条是三层：**Deployment → ReplicaSet → Pod**。Deployment 每次发布新版本就创建一个新 ReplicaSet，并慢慢把流量/副本从旧 RS 挪到新 RS；回滚就是把副本数挪回去。Deployment 的 `status` 里会记录新旧 RS 各自的副本数——这是滚动更新的计账本。

命令式与声明式的差别，用一张表对照最清楚：

| 维度 | 命令式（imperative） | 声明式（declarative） |
| --- | --- | --- |
| 用户给什么 | 操作步骤（`kubectl run` 建/删） | 期望状态（`kubectl apply` 一份 YAML） |
| 失败怎么办 | 操作失败就停在半路 | 控制器持续补差直到收敛 |
| 谁负责「怎么做」 | 用户与脚本 | 控制器 |
| 可审计性 | 靠操作日志拼凑 | 对象本身就是变更历史 |

**`kubectl apply` 优于 `kubectl create`/`kubectl delete` 的原因就在这张表里**：`apply` 提交的是「目标状态」，后续任何 `apply` 都以当前文件为期望值做调和；而 `create`/`delete` 是「一次性命令」，无法重放、无法收敛。

## 3 滚动更新：把大爆炸拆成小步

升级 10 个副本的 Deployment，最怕「全部停掉再启动」（流量瞬间全断）。滚动更新用「**先起新，再停旧**」的节奏，让新旧副本数量满足：

$$
n_{\text{desired}} = n_{\text{new}} + n_{\text{old}}, \qquad \forall t:\ n_{\text{new}}(t) + n_{\text{old}}(t) \le n_{\text{desired}} + n_{\text{maxSurge}}
$$

（$n_{\text{maxSurge}}$ 与 `maxUnavailable` 由滚动策略控制，默认各允许 25% 的瞬时超卖/欠卖。）

滚动期间新旧版本并存，readiness 探针确保新 Pod 就绪后才继续推进——这就是为什么 Service 能全程无感：后端列表同时含新旧 Pod，新 Pod 就绪即加入，旧 Pod 退出即移除。

默认策略 `maxSurge=25%、maxUnavailable=25%` 对一个 10 副本的服务意味着：任意时刻最多 13 个 Pod 同时存在（多出 3 个新 Pod），同时至少 8 个旧 Pod 在服务（最多停 2 个）。**这套「新多旧不少」的记账，保证升级窗口里永远有人接得住流量**。若应用无法接受「新旧并存」（比如数据库 schema 不兼容），就要换成 `maxSurge=0, maxUnavailable=1` 的「先删一个再补一个」策略——代价是升级变慢。

## 4 公式解析：滚动更新的推进判据

滚动更新的每一步都做一个决策：**可以再停一个旧 Pod 吗？** 判据是 readiness：

$$
\text{advance}(t) \iff \frac{n_{\text{ready,new}}(t)}{n_{\text{new}}(t)} \ge 1
$$

- $n_{\text{ready,new}}(t)$：新 ReplicaSet 中已就绪（readinessProbe 通过）的副本数。
- $n_{\text{new}}(t)$：新 ReplicaSet 的目标副本数。
- 判据含义：**只有当前这一批新 Pod 全部就绪，才推进下一批**。

三步拆解：

- **第一步，就绪是硬门槛**：新 Pod 没通过 readiness 探针，就永远卡住，不继续停旧 Pod——避免了「新的是坏的，旧的又被停了」。
- **第二步，就绪定义由应用给**：`readinessProbe` 可以是 HTTP 健康检查、TCP 连通、命令退出码。应用要诚实回答「我现在能不能接流量」——回答错了，滚动更新就会踩坑。
- **第三步，这就是「逐步收敛」**：整个发布过程只是控制器反复执行「起一个→等就绪→停一个」，直到 $n_{\text{old}} = 0$。回滚则把方向反过来——**滚动更新的全部复杂性，都被控制回环吸收掉了**。

**补充｜延伸：ownerReferences 与垃圾回收**。Deployment 创建 ReplicaSet、ReplicaSet 创建 Pod，靠的不是「名字约定」，而是对象上的 **ownerReference** 字段。删掉 Deployment 时，垃圾回收器顺着 ownerReference 自动级联删除它名下的 RS 与 Pod；反过来，手动删掉一个 RS，Deployment 控制器会立刻再造一个——**对象之间的父子关系是被显式记录、并被回收器兑现的**，这是声明式模型「可审计、可组合」的另一面。

## 5 声明式 API 的工程含义

- **可复现**：同一份 `spec` 在任何集群产生相同结果——环境即代码。
- **可审计**：对象在 etcd 里的历史（配合版本控制）就是变更日志。
- **可组合**：控制器之上还能叠控制器（如 Argo Rollouts 定制发布策略），因为大家都在同一套 watch 机制上协作。<span class="marginnote">「控制器之上叠控制器」是 Kubernetes 生态无限扩展的秘诀：只要新控制器 watch 的对象变了它就动，系统的行为就等于这些控制器行为的组合——这让人想起函数式编程里「纯函数 + 组合」的优雅。</span>
- **可 GitOps**：把 `spec` 放进 Git，由持续交付工具（Argo CD、Flux）负责「把 Git 里的期望状态 apply 到集群」，控制器再负责「让实际状态收敛到它」。**人与机器的分工由此清晰分层：人管 Git（意图），控制器管集群（执行）**。

**补充｜延伸：控制器的「选主」**。`kube-controller-manager` 本身也跑多副本保证高可用，但同一时刻只有一个副本是「领导者」在真正执行调和，其余处于 standby——避免多个控制器同时 create/delete 造成竞态。这个「单领导者执行、多副本待命」的模式，与 etcd 的 Raft 选主、《分布式系统设计模式》里的 Leader election 是同一个思想，只是粒度不同：etcd 选主保护数据一致性，控制器选主保护「别让两个控制器打架」。

## 6 小结

- **控制器**是「观察 → 比较 → 修正」的无限循环，是自愈与自动化的引擎。
- 控制器家族：Deployment（发布）、ReplicaSet（数量）、StatefulSet（稳定身份）、DaemonSet（每节点一份）、Job/CronJob（任务）。
- 管理链是 **Deployment → ReplicaSet → Pod**，三层分明；父子关系由 ownerReference 显式记录并级联回收。
- **滚动更新**按「新就绪才停旧」推进，$\text{advance} \iff n_{\text{ready,new}} = n_{\text{new}}$；maxSurge/maxUnavailable 决定「新多旧不少」的节奏。
- 声明式 API 带来可复现、可审计、可组合。

在下一节，我们让系统「按需自动增减副本」——进入 **弹性伸缩 HPA/VPA**。
