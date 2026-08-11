---
title: 持久化存储与 CSI
date: 2026-08-11
---

# 持久化存储与 CSI

<div class="epigraph">
<p>Pod 会消失，节点会宕机，但数据必须活下来——持久化是把「瞬态的计算」和「长命的数」分开的那条线。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么数据要「另立门户」

前几节课反复强调：**Pod 是可抛弃的**，容器里的可写层随时会丢。但现实世界有大量「丢了就完了」的数据——数据库的记录、消息队列的积压、模型的权重。持久化存储解决的就是：**让数据的生命周期独立于 Pod 的生命周期**。它是「无状态应用可抛弃」这一命题的另一半——正因为数据被搬到了外面，应用进程才能放心地被反复销毁重建。

## 1 卷：Pod 与数据的桥梁

**核心概念：卷（Volume）**：挂载进 Pod 的存储单元，生命周期与 Pod 相同（Pod 没了卷可能还在）。卷把「进程看到的文件路径」与「真实存储的位置」解耦：

- **emptyDir**：Pod 内共享的临时目录，Pod 删除即清空——给同 Pod 多容器交换数据用。
- **hostPath**：挂宿主机目录——仅测试用，生产有节点绑定问题。
- **网络存储**：真正重要的卷，来自存储系统（云盘、NFS、Ceph）——数据存在**集群之外**，Pod 没了、节点没了，数据还在。

判断卷用哪种，问一个问题：**数据跟谁活？** 跟 Pod 活 → emptyDir；跟宿主机活 → hostPath；跟集群活 → 网络存储。<span class="marginnote">容器 → 卷 → 存储系统的三层解耦，是「位置无关」哲学的具体化：应用不再关心数据在本地还是远端，只认一个挂载点。这与前面「后端服务可替换」「镜像内容寻址」是同一条主线。</span>

## 2 PV、PVC 与 StorageClass：把「存储」变成一门语言

Pod 不该直接指定「我要 AWS 的 gp3 盘、50GB」——那样 Pod 就和具体存储绑定死了。Kubernetes 用三个对象把「要什么」与「怎么提供」分开：

**PersistentVolume（PV）**：管理员提供的**一块具体存储**（云盘、NFS 卷），集群里的资源。
- **PersistentVolumeClaim（PVC）**：应用**声明**的存储需求——「我要 50GB，可读写多节点」。声明式思维第 N 次出现：不指定哪块盘，只描述需求。
- **StorageClass**：**动态供应**的配方——「要哪种盘（`gp3`/`ssd`/`nfs`）、多少 IOPS」。PVC 引用 StorageClass，由 provisioner 自动创建 PV 并绑定。

声明式 + 动态供应的效果：开发者写一个 PVC，集群自动准备好一块盘，Pod 挂载即用。**存储从「运维手工准备」变成了「按需自助申请」**——这与其他对象的声明式模型完全同构。

## 3 CSI：存储插件的通用接口

Kubernetes 不能认识全世界每一种存储（云盘、SAN、NFS、对象存储……）。答案是接口：**CSI（Container Storage Interface）**——存储厂商只要实现一组 gRPC 接口（`CreateVolume`、`DeleteVolume`、`Mount`、`Unmount`），就能无缝接入 Kubernetes。CSI 驱动部署为 DaemonSet/StatefulSet，对外是标准接口，对内是任意实现。<span class="marginnote">CSI 与 CRI（容器运行时接口）、CNI（网络接口）并称 Kubernetes 的「三驾马车」：<strong>插件化把生态的多样性变成了接口下的可选实现</strong>。这是分布式系统设计里的「适配器模式」在存储领域的落地。</span>

CSI 接口要保证的三件事：**幂等**（重复调用不产生副作用）、**可恢复**（失败后状态可查询）、**协调**（与快照、扩容、加密等操作协同）。存储驱动是生产事故的重灾区，所以接口规范被设计得极其保守。

## 4 公式解析：持久化的绑定关系

存储的绑定可以用一个简洁的关系式描述：

$$
\text{Bound}(PV, PVC) \iff \text{capacity}(PV) \ge \text{request}(PVC) \land \text{mode}(PV) \supseteq \text{mode}(PVC)
$$

- $\text{capacity}(PV) \ge \text{request}(PVC)$：PV 容量要覆盖 PVC 的请求。
- $\text{mode}(PV) \supseteq \text{mode}(PVC)$：PV 的访问模式要兼容 PVC 要求的模式。
- 绑定后 PVC 与 PV 一对一，Pod 通过 PVC 引用。

访问模式（access modes）是一张必须记牢的对照表：

| 模式 | 读写方式 | 典型场景 |
| --- | --- | --- |
| `ReadWriteOnce` (RWO) | 单节点读写 | 单副本数据库 |
| `ReadOnlyMany` (ROX) | 多节点只读 | 共享配置、模型权重 |
| `ReadWriteMany` (RWX) | 多节点读写 | 共享文件系统、日志聚合 |

三步拆解：

- **第一步，容量约束**：PVC 是「需求声明」，PV 是「供给」，绑定是**供需匹配**——这与其他任何资源的调度（CPU、内存）是同一套语言。
- **第二步，模式约束是硬性的**：访问模式由底层存储能力决定（块存储天然只能 RWO，文件系统才能 RWX）。**在创建 PVC 前想清楚访问模式，是数据库上云最常见的架构决策**。
- **第三步，动态 vs 静态**：静态供应 = 管理员先建好 PV 等 PVC 来绑；动态供应 = StorageClass 按 PVC 现场造。生产中几乎总是用动态供应。

## 5 有状态应用的配套：StatefulSet

「持久化 + 多副本」需要稳定身份：Pod 被重建后还要认领**同一块**数据。这就是 **StatefulSet** 的职责——它给 Pod 稳定名字（`db-0`、`db-1`）与**稳定的 PVC 绑定**（每个副本有自己的卷，重建后重新挂回）。配合分布式系统课里的分片模式，StatefulSet 是「数据库上 K8s」的基本骨架。

**辨析｜易错点**：StatefulSet 的卷**不会**随 Pod 删除而自动删除——删除 StatefulSet 默认保留 PVC，这是有意设计（怕误删数据）。**清理数据要手动删 PVC**——新手删了 StatefulSet 以为数据清了，结果账单还在跑。

## 6 小结

- **卷**把「进程看到的路径」与「真实存储」解耦；数据跟谁活就选哪种卷。
- **PV/PVC/StorageClass** 把「要什么」「怎么提供」分离，动态供应让存储自助化。
- **CSI** 是存储插件的统一接口，幂等、可恢复、可协调是其底线。
- 绑定关系：容量与访问模式都要匹配；访问模式由底层存储决定。
- **StatefulSet** 提供稳定身份与稳定卷绑定；删它**不删卷**，清理要手动删 PVC。

在下一节，我们从「数据怎么存」转向「数据怎么走」——进入 **网络模型（CNI/Service Mesh）**。
