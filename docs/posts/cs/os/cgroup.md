---
title: Cgroup：CPU、内存与 I/O 资源控制（v1 与 v2）
date: 2026-08-07
---

# Cgroup：CPU、内存与 I/O 资源控制（v1 与 v2）

<div class="epigraph">
<p>Namespace 让容器「看到」独立的世界，cgroup 让容器「只能用」分到的资源——一个管视野，一个管配额。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》与 cgroup 文档 ｜ 2026-08-07</p>
</div>

## 为什么从 cgroup 开始

Namespace 隔离「视图」，但**资源限制**要靠 **cgroup**——否则一个容器能吃掉全部 CPU/内存。**cgroup（control group）**是 Linux 的资源控制机制：把进程分组，对每组**限制、记账、隔离** CPU、内存、I/O 等资源。它是容器「资源配额」的实现基础，也是 systemd、Kubernetes 资源管理的底层。<span class="marginnote">回顾《Namespace》：容器 = 隔离的视图。补上 cgroup——<strong>容器 = Namespace（视图隔离）+ cgroup（资源限制）+ rootfs（根文件系统）</strong>。Namespace 是「墙」（隔离视野），cgroup 是「闸」（限制用量）。</span>

## 1 cgroup 是什么

**cgroup（control group，控制组）**：Linux 内核把进程**分组**，对每组进行**资源控制**的机制。

**cgroup 能做三件事**：

- **限制（limit）**：最多能用多少 CPU/内存。
- **记账（accounting）**：统计每个组用了多少资源。
- **隔离（isolation）**：组的资源互不干扰。

**cgroup 的层次结构**：cgroup 组织成**树**——父组限制总量，子组在父组配额内再分配。

```
cgroup 树：
  /（根）
  ├── 容器1（cpu 限额 2 核、内存 1GB）
  │   ├── 进程 A
  │   └── 进程 B
  └── 容器2（cpu 限额 1 核、内存 512MB）
      └── 进程 C
```

## 2 cgroup 的资源控制维度

**CPU 控制**：

- **cpu.shares**：CPU 份额权重（相对比例，非绝对限制）。
- **cpu.cfs_quota_us / cfs_period_us**：**绝对限制**——每 period 内最多用 quota 微秒（如每 100ms 最多用 50ms = 半核）。
- **cpu.max（v2）**：`quota period` 格式。

**内存控制**：

- **memory.limit_in_bytes**：内存上限（超出触发 OOM 或回收）。
- **memory.max（v2）**：同上。
- **memory.usage_in_bytes**：当前用量（记账）。

**I/O 控制**：

- **blkio（v1）**：块设备 I/O 权重与带宽限制。
- **io.max（v2）**：`read/write` 带宽与 IOPS 限制。

**PID 控制**：**pids.max**——限制组内进程/线程数（防 fork 炸弹）。

**公式解析：CFS 配额 = CPU 核数限制**

cgroup 的 CPU 绝对限制：

$$\text{CPU 核数上限} = \frac{\text{cpu.max.quota}}{\text{cpu.max.period}}$$

- **quota**：每个周期最多使用的 CPU 时间（微秒）。
- **period**：周期长度（默认 100ms = 100000µs）。
- 设 quota = 50000µs、period = 100000µs：上限 $= 0.5$ 核。
- 设 quota = 200000µs：上限 $= 2$ 核。

**直觉**：**「每周期多少微秒」=「几个核」**——这是 Docker `--cpus=0.5` 的底层实现。cgroup 把「核数」翻译成「CPU 时间配额」。

## 3 cgroup v1 vs v2

**cgroup v1（传统）**：

- **每个资源一个控制器树**：cpu 一棵树、memory 一棵树、blkio 一棵树——**多棵树并存**。
- 一个进程可以属于**多棵树**（每棵一个组）。
- 管理复杂：不同资源的组不同步，层级难一致。

**cgroup v2（现代，Linux 4.5+）**：

- **统一层级（unified hierarchy）**：**一棵树管所有控制器**——一个进程属于一个组，组内所有控制器统一配置。
- 更清晰的层次语义、更好的原子性。
- **systemd、Docker、Kubernetes 已迁移到 v2**。

| 维度 | cgroup v1 | cgroup v2 |
| --- | --- | --- |
| 层级 | 每控制器一棵树 | **统一一棵树** |
| 进程归属 | 每控制器各自分组 | **一个组全资源** |
| 控制器 | 独立挂载 | 统一管理 |
| 现代地位 | 遗留 | **标准** |

**辨析｜易错点：** 「cgroup 能限制一切」是过度乐观。**cgroup 限制的是「可控的配额资源」（CPU 时间、内存、I/O），不能限制「不可控的副作用」**——比如进程触发的中断风暴、内核资源。且 cgroup 需要内核特性与正确配置。**「限制资源 ≠ 完全隔离」**——它管「用多少」，不管「干什么」。

## 4 核心对比表：Namespace vs cgroup

| 维度 | Namespace | cgroup |
| --- | --- | --- |
| 管什么 | 视图（看到什么） | 资源（能用多少） |
| 机制 | 隔离视图 | 限制/记账 |
| 类比 | 墙 | 闸 |
| 容器角色 | 隔离 | 配额 |
| 不防什么 | 不防资源滥用 | 不防视图入侵 |

**设计启示**：Namespace + cgroup 是「**隔离 + 配额**」的完整组合——**隔离让容器「看不全」，配额让容器「用不多」**。两者合起来才构成容器的资源语义。这也呼应了操作系统的两大主题：**资源管理与保护**——cgroup 是资源管理，Namespace 是保护（视图级）。

## 5 小结

- **cgroup**：进程分组 + 资源控制（限制、记账、隔离）。
- 三大控制维度：**CPU**（份额/配额）、**内存**（上限）、**I/O**（带宽/IOPS）。
- CPU 配额：`quota/period` = 核数上限（Docker `--cpus` 的实现）。
- **cgroup v2** 统一层级取代 v1 的多树——现代标准。
- 容器 = **Namespace（视图隔离）+ cgroup（资源配额）+ rootfs**。

在下一节，我们把 Namespace + cgroup + 镜像组装起来——**Docker 原理：镜像、联合文件系统与容器运行时**。
