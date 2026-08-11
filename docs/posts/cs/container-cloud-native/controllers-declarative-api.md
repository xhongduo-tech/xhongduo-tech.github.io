---
title: 控制器与声明式 API
date: 2026-08-11
---

# 控制器与声明式 API

<div class="epigraph">
<p>不要把系统写成一张张操作指令，把它写成你想要的结果；让一个持续盯梢的循环去完成它。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
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

## 2 主力控制器家族

- **ReplicaSet**：保证「恰好 $n$ 个副本」，只懂数量，不懂更新。
- **Deployment**：包在 ReplicaSet 外面的「发布管理器」——管理滚动更新、回滚、暂停/恢复。Deployment 不直接管 Pod，而是**管 ReplicaSet**（新版本 = 新 ReplicaSet，渐进替换）。
- **StatefulSet**：给 Pod 稳定身份（`web-0`、`web-1`……）与稳定存储，用于数据库、消息队列等有状态应用（见《持久化存储》）。
- **DaemonSet**：保证「每个（符合条件的）节点上都跑一份」，如日志采集、网络插件、节点监控。
- **Job / CronJob**：一次性任务与定时任务，跑完 `Succeeded` 即终。

**辨析｜易错点：Deployment 与 ReplicaSet 的关系**。新手常以为 Deployment 直接管理 Pod。实际上链条是三层：**Deployment → ReplicaSet → Pod**。Deployment 每次发布新版本就创建一个新 ReplicaSet，并慢慢把流量/副本从旧 RS 挪到新 RS；回滚就是把副本数挪回去。Deployment 的 `status` 里会记录新旧 RS 各自的副本数——这是滚动更新的计账本。

## 3 滚动更新：把大爆炸拆成小步

升级 10 个副本的 Deployment，最怕「全部停掉再启动」（流量瞬间全断）。滚动更新用「**先起新，再停旧**」的节奏，让新旧副本数量满足：

$$
n_{\text{desired}} = n_{\text{new}} + n_{\text{old}}, \qquad \forall t:\ n_{\text{new}}(t) + n_{\text{old}}(t) \le n_{\text{desired}} + n_{\text{maxSurge}}
$$

（$n_{\text{maxSurge}}$ 与 `maxUnavailable` 由滚动策略控制，默认各允许 25% 的瞬时超卖/欠卖。）

滚动期间新旧版本并存，readiness 探针确保新 Pod 就绪后才继续推进——这就是为什么 Service 能全程无感：后端列表同时含新旧 Pod，新 Pod 就绪即加入，旧 Pod 退出即移除。

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

## 5 声明式 API 的工程含义

- **可复现**：同一份 `spec` 在任何集群产生相同结果——环境即代码。
- **可审计**：对象在 etcd 里的历史（配合版本控制）就是变更日志。
- **可组合**：控制器之上还能叠控制器（如 Argo Rollouts 定制发布策略），因为大家都在同一套 watch 机制上协作。<span class="marginnote">「控制器之上叠控制器」是 Kubernetes 生态无限扩展的秘诀：只要新控制器 watch 的对象变了它就动，系统的行为就等于这些控制器行为的组合——这让人想起函数式编程里「纯函数 + 组合」的优雅。</span>

## 6 小结

- **控制器**是「观察 → 比较 → 修正」的无限循环，是自愈与自动化的引擎。
- 控制器家族：Deployment（发布）、ReplicaSet（数量）、StatefulSet（稳定身份）、DaemonSet（每节点一份）、Job/CronJob（任务）。
- 管理链是 **Deployment → ReplicaSet → Pod**，三层分明。
- **滚动更新**按「新就绪才停旧」推进，$\text{advance} \iff n_{\text{ready,new}} = n_{\text{new}}$。
- 声明式 API 带来可复现、可审计、可组合。

在下一节，我们让系统「按需自动增减副本」——进入 **弹性伸缩 HPA/VPA**。
