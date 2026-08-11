---
title: Kubernetes 对象模型
date: 2026-08-11
---

# Kubernetes 对象模型

<div class="epigraph">
<p>Kubernetes 里没有「命令」，只有「对象」；你描述它，系统让它发生。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要对象模型

上一节我们把集群拆成了大脑与肌肉，但「用户到底对集群说些什么」还没定型。Kubernetes 的回答是：**一切皆对象**。Pod 是对象，Service 是对象，Deployment、ConfigMap、Namespace 都是对象。对象模型就是 Kubernetes 的「语法」，搞懂它，你就掌握了与集群对话的正确姿势；不懂它，你会写出「看似能跑、实则充满隐患」的清单文件。<span class="marginnote">「一切皆对象」并非新发明——它和 Unix 的「一切皆文件」、面向对象编程的「对象」是同一种思维：把世界抽象成少量、统一、可组合的结构。</span>

## 1 对象的骨架：type 与 metadata 与 spec/status

每个 Kubernetes 对象都长着同一副骨架，写在 YAML 里：

```yaml
apiVersion: apps/v1        # ① 该对象所属的 API 组与版本
kind: Deployment            # ② 对象的类型
metadata:                   # ③ 身份与元数据
  name: web
  namespace: default
  labels:
    app: web
spec:                       # ④ 期望状态（用户填写）
  replicas: 3
status:                     # ⑤ 实际状态（系统填写）
  availableReplicas: 3
```

- **`apiVersion` + `kind`**：确定「这份文档是什么」。同一种 `kind` 可能存在于多个版本（`v1`、`v1beta1`……），`apiVersion` 保证解析器不会认错。这是对象模型的「类型系统」。
- **`metadata`**：对象的**身份**。`name` 在同一命名空间内唯一；`namespace` 决定归属；`labels` 与 `annotations` 提供「附加信息」。
- **`spec`**：用户**声明**的期望状态——上节说的 `Desired`。
- **`status`**：系统**观测**到的实际状态——上节说的 `Observed`，**由控制器写入，用户不应手写**。

**核心概念：`spec` 与 `status` 的分工**：`spec` 是「我要的」，`status` 是「它实际是」，二者之差就是控制回环的 $\Delta$。用户只写 `spec`，`status` 永远交给系统。

## 2 Labels 与 Selectors：对象的检索语言

`metadata.labels` 是**键值对**（如 `app: web`、`tier: frontend`），它的用处不在「存数据」，而在**被选择**。控制器和 Service 通过 **selector** 按标签挑选对象：

$$\text{selected} = \{ o \in \text{all} \mid \text{match}(o.\text{labels}, \text{selector}) \}$$

一个 Deployment 靠 `selector.matchLabels` 找到「自己管辖的 Pod」，而不是靠名字或顺序。标签选择器支持两种写法：`app=web`（等值）、`app in (web, api)`（集合）。<span class="marginnote">把「对象之间的关系」从「硬编码名字」改为「按标签检索」，是 Kubernetes 可扩展性的关键——新对象只要贴上匹配的标签，就自动被选中，无需改任何代码。这与数据库里用索引查询、而不是遍历整表，是同一种思维。</span>

**辨析｜易错点：Labels 与 Annotations**：两者都在 `metadata` 里，但用途完全不同。

**Labels**：用于**选择与分组**——必须有严格的键值格式，参与 selector 匹配，因此是「结构性」的。
- **Annotations**：用于**存信息**——不参与选择，可以容纳任意非结构化内容（版本说明、负责人、工具参数）。

最容易犯的错：把「标识身份」的事交给 annotations（结果无法被 selector 选中），或把「临时备注」写进 labels（结果污染了选择维度）。规则一句话：**能被选中的进 labels，只供人/工具读的进 annotations。**

## 3 Namespace：逻辑上的租户隔离

`metadata.namespace` 把集群切成多个**逻辑分区**。同一命名空间内的对象可以互相裸名引用（`Service: web`），跨命名空间必须带全名（`web.default.svc`）。命名空间是**资源配额（ResourceQuota）与 RBAC 授权**的边界，但**不是**网络与安全的强制边界（网络隔离靠 NetworkPolicy，见《安全》一课）。<span class="marginnote">Namespace 是「软分区」：它管的是名字与权限的边界，不管流量。把「命名空间隔离」当成「安全隔离」是常见误用——真正要隔离生产与开发流量，必须配 NetworkPolicy 或物理拆集群。</span>

## 4 公式解析：标签选择器的命中

标签选择器是对象模型里少有的「公式」：

$$
S(L) = \{ o \mid \forall (k, v) \in L: o.\text{labels}[k] = v \}
$$

- $L$：selector 里的键值对集合，如 $\{(\text{app}, \text{web})\}$。
- $o.\text{labels}[k] = v$：对象 $o$ 的标签中键 $k$ 的取值等于 $v$。
- $S(L)$：命中集合。

三步拆解：

- **第一步，约束是「全部满足」**：selector 里所有键值对都要匹配，缺一个就落选——这是「AND 语义」。
- **第二步，选择不看名字**：对象叫什么、创建顺序如何，一律不参与；只有标签参与匹配。所以「Pod 是不是我的」在任何一个时刻都可由标签唯一判定。
- **第三步，这是声明式组合的接口**：Deployment、Service、NetworkPolicy 都靠同一套选择语言互相指认。**标签把「对象之间的边」显式化了**——这在系统变大后，是唯一能靠工具审计的关联方式。

## 5 用对象模型重读集群架构

把前面的知识串起来：kubectl 提交一份带 `spec` 的对象 → API Server 校验后写入 etcd → 对应控制器 watch 到变化，算出 $\Delta$ → 创建/修改下层对象 → 循环往复，直到 `status` 追上 `spec`。**对象之间靠 labels 结成网，靠 spec/status 形成回环**——架构课的骨架，现在有了血肉。

## 6 小结

- 每个对象由 **type（apiVersion+kind）**、**metadata**、**spec**、**status** 四部分构成。
- **spec 是用户声明的期望**，**status 是系统观测的实际**，差即 $\Delta$。
- **Labels 用于选择**（参与 selector），**Annotations 用于存信息**（不参与选择）。
- Namespace 是逻辑租户边界，管名字与配额，**不管网络流量**。
- 标签选择器是「AND 语义」的集合过滤，是对象之间关系的接口。

在下一节，我们看 Kubernetes 的最小调度单元如何在集群里落位并相互发现——进入 **Pod 与服务发现**。
