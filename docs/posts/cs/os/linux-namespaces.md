---
title: Linux Namespace：进程隔离的六大命名空间
date: 2026-08-07
---

# Linux Namespace：进程隔离的六大命名空间

<div class="epigraph">
<p>容器不是「跑另一个系统」，而是「让进程以为自己独占一台机器」——Namespace 是这台「假机器」的六块墙。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》与容器原理 ｜ 2026-08-07</p>
</div>

## 为什么从 Namespace 开始

容器与虚拟机的根本区别：**虚拟机虚拟化「硬件」，容器虚拟化「操作系统视图」**。**Linux Namespace** 是容器隔离的基石——它让一组进程看到**独立的系统视图**（独立的 PID、网络、挂载点…），仿佛独占一台机器。这一节看 Namespace 的机制与六大类型。<span class="marginnote">回顾 KVM：虚拟机 = 虚拟硬件 + 完整客户 OS。容器更轻——<strong>没有虚拟硬件，容器里的进程直接跑在宿主内核上，只是通过 Namespace 看到「隔离的视图」</strong>。<strong>「虚拟化硬件 vs 虚拟化视图」是容器与虚拟机的分水岭。</strong></span>

## 1 Namespace：进程的「独立视角」

**Namespace（命名空间）**：Linux 内核的一种**资源隔离机制**——把「进程看到的系统资源」包装成**独立视图**。同一 Namespace 内的进程共享视图，不同 Namespace 的进程互不可见。

**关键认识**：

- **Namespace 是「视图隔离」，不是「资源独占」**——PID namespace 隔离的是「看到哪些 PID」，不是「PID 真只有一个」。
- **每个进程属于一组 Namespace**（每种资源一个）——`clone`/`unshare` 时指定。
- **容器 = 一组 Namespace + 资源限制（cgroup）**——两个机制配合（见《cgroup》）。

## 2 六大 Namespace

Linux 提供多种 Namespace，容器主要用六个：

| Namespace | 隔离什么 | 容器里的效果 |
| --- | --- | --- |
| **PID** | 进程号视图 | 容器内进程 PID 从 1 开始，看不到宿主进程 |
| **Mount** | 挂载点视图 | 容器有自己的文件系统树（rootfs） |
| **Network** | 网络栈 | 容器有自己的网卡、IP、路由 |
| **UTS** | 主机名 | 容器有独立 hostname |
| **IPC** | 进程间通信 | 容器有独立的 System V IPC/消息队列 |
| **User** | 用户与权限 | 容器内有独立的 UID 映射 |

**PID Namespace**（最常用）：容器内第一个进程 PID = 1（像 init）——容器内 `ps` 只看得到容器内进程；**容器内 PID 与宿主 PID 是两套编号**（映射关系由内核管理）。

**Mount Namespace**：容器有自己独立的挂载点树——**容器的 `/` 是它的 rootfs**（镜像的根文件系统），宿主机的目录树被「隐藏」；容器可自由挂载而不影响宿主。

**Network Namespace**：每个容器有自己的**网络栈**（网卡、IP、路由表）——容器间的网络像「独立的机器」互连（通过 veth/网桥）。

**User Namespace**：容器内 root（UID 0）映射到宿主机的**非特权用户**——**容器内「我是 root」不代表宿主 root**（安全关键，见《容器安全》）。

**公式解析：PID Namespace 的编号映射**

设宿主某进程 PID = 1000，它在嵌套 PID Namespace 内的编号：

$$\text{ns 内 PID} = \text{该 Namespace 内看到的编号} \neq \text{宿主 PID}$$

- 宿主 PID 1000 的进程，若它所在的 namespace 里是第 3 个进程，则 ns 内 PID = 3。
- **同一进程，宿主看是 1000，容器看是 3**——两套编号并存，内核维护映射。
- **容器内第一个进程（PID 1）在该 namespace 内编号为 1**——它承担「回收孤儿进程」的 init 职责。

**直觉**：Namespace 是「**给每个容器发一副滤镜**」——同一条进程/资源，不同容器看到不同的「编号与视图」。**「看到什么」是隔离，「是否存在」不是。**<span class="marginnote">PID Namespace 嵌套还带来「<strong>init 进程职责</strong>」：容器 PID 1 要像 init 一样<strong>回收孤儿进程、处理信号</strong>——否则容器里会出现僵尸堆积（回顾《孤儿进程》）。这也是容器 PID 1 进程（如 `tini`）被反复强调的原因。</span>

## 3 Namespace 的创建与管理

**创建 Namespace**：

- **`clone(flags)`**：创建子进程时指定 `CLONE_NEWPID`、`CLONE_NEWNET` 等——子进程进入新 Namespace。
- **`unshare(flags)`**：当前进程进入新的 Namespace（不创建子进程）。
- **`setns(fd, flags)`**：加入已有的 Namespace。

```c
// 创建容器式的隔离进程（概念）
clone(child_fn, stack,
      CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWUTS |
      CLONE_NEWIPC | CLONE_NEWNS | SIGCHLD, NULL);
```

**Docker 的实践**：Docker 用 `clone` 创建容器进程，指定六个 Namespace 标志——容器就「看到」独立的系统视图。**这六个标志是容器的「六块墙」。**

**管理命令**：`lsns`（列出 Namespace）、`nsenter`（进入 Namespace）、`unshare`（创建并进入）。

**辨析｜易错点：** 「Namespace = 沙箱安全」是过度简化。**Namespace 提供「视图隔离」，不是「安全边界」**——它让进程「看不见」其他视图，但**攻击者若突破（如利用内核漏洞、`CAP_SYS_ADMIN`），视图隔离挡不住**。**安全隔离需要 Namespace + cgroup + Capability + Seccomp 的组合**（见《容器安全》）。**「看不见 ≠ 碰不到」。**

## 4 核心对比表：虚拟机 vs 容器

| 维度 | 虚拟机（KVM） | 容器（Namespace） |
| --- | --- | --- |
| 隔离对象 | 硬件 | 操作系统视图 |
| 内核 | 独立客户 OS | **共享宿主内核** |
| 隔离强度 | 强（硬件级） | 弱（视图级） |
| 启动速度 | 秒级 | **毫秒级** |
| 资源开销 | 大（完整 OS） | **小（共享内核）** |
| 安全边界 | 硬件 | 需组合（ns+cgroup+seccomp） |

## 5 小结

- **Namespace**：资源视图隔离——让进程「看到」独立的系统。
- 六大容器 Namespace：**PID、Mount、Network、UTS、IPC、User**。
- PID Namespace：容器内 PID 从 1 开始，两套编号由内核映射。
- 创建：`clone`/`unshare`/`setns`；Docker 用 clone 的六个标志。
- **Namespace 是视图隔离，不是安全边界**——安全需组合 cgroup、Capability、Seccomp。

在下一节，我们深入两个关键 Namespace——**Mount 与 Network Namespace 详解**。
