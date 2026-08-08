---
title: Mount 与 Network Namespace 详解
date: 2026-08-07
---

# Mount 与 Network Namespace 详解

<div class="epigraph">
<p>容器的根文件系统与网络，是两个最「活」的隔离维度——一个决定「我看到什么文件」，一个决定「我和谁相连」。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》与 Docker 网络 ｜ 2026-08-07</p>
</div>

## 为什么从 Mount 与 Network Namespace 开始

六大 Namespace 里，**Mount** 与 **Network** 是容器「看起来像一台独立机器」的两个关键——**Mount 隔离文件系统视图**（容器的根目录是自己的 rootfs），**Network 隔离网络栈**（容器有自己的 IP 与网卡）。这一节把这两个最复杂的 Namespace 讲透，它们是理解 Docker 镜像与网络的基础。<span class="marginnote">回顾《Namespace》：Mount 让容器有自己的文件系统树，Network 让容器有自己的网络栈。这一节深入——<strong>Mount 是「视图隔离 + 共享子树的精细控制」，Network 是「虚拟网卡 + 网桥 + NAT」的完整虚拟网络</strong>。</span>

## 1 Mount Namespace：独立的文件系统视图

**Mount Namespace**：隔离**挂载点视图**——Namespace 内的进程看到的挂载点集合是独立的，与宿主和其他 Namespace 不同。

**关键语义**：

- 容器创建时，pivot_root 切换到**容器的 rootfs**（镜像的根目录）——容器的根目录不再是宿主的根目录。
- **容器的挂载操作**（mount/umount）**只影响自己的 Mount Namespace**——不影响宿主。
- 宿主与容器可以**共享部分挂载**（如只读挂载镜像层、bind mount 卷）。

**Mount Namespace 的实现机制**：

- 内核维护**挂载点树**（mount tree），每个挂载点有**传播属性**（共享/从属/私有）。
- **共享挂载（shared）**：一个 Namespace 挂载，共享的 Namespace 都能看到（传播）。
- **私有挂载（private）**：挂载变化不传播。

**容器镜像的挂载**（Docker 的分层）：

- 容器根 = **只读镜像层叠加 + 可写层**（联合文件系统，见《Docker 原理》）。
- 数据卷 = **bind mount 宿主目录**进容器。
- 这些都是「在容器的 Mount Namespace 里做挂载」。

**公式解析：pivot_root 切换根**

```c
int pivot_root(const char *new_root, const char *put_old);
```

- **new_root**：新根（容器的 rootfs）。
- **put_old**：旧根被移到的位置（之后 umount）。
- 执行后：容器的根目录变成 new_root——**宿主根目录在容器视图里被「换掉」**。

**直觉**：Mount Namespace + pivot_root = 「**容器的世界从根开始就是自己的**」——文件系统视图彻底隔离（回顾《文件系统挂载》的挂载点概念在 Namespace 层的应用）。

## 2 Network Namespace：独立的网络栈

**Network Namespace**：隔离**完整的网络栈**——每个 Network Namespace 有自己独立的：

- 网卡（虚拟网卡 veth、loopback）。
- IP 地址、路由表。
- 防火墙规则（iptables/netfilter）、socket 表。

**容器网络的关键组件**：

- **veth pair（虚拟以太网对）**：一根「虚拟网线」，两端分别在容器与宿主网桥——数据从一端进、另一端出。
- **Linux 网桥（bridge）**：虚拟交换机，连接多个容器的 veth——容器间互通。
- **NAT**：容器访问外网时，宿主做**地址转换**（把容器私有 IP 映射成宿主 IP）。

```
容器A(veth) ─┐
            ├─ docker0 网桥 ── NAT ── 宿主网卡 ── 外网
容器B(veth) ─┘
```

**容器的网络模式**（Docker）：

- **bridge（默认）**：容器接在 docker0 网桥，通过 NAT 访问外网。
- **host**：容器共享宿主网络栈（无隔离，性能好）。
- **none**：无网络。
- **container**：共享另一个容器的网络栈。

**公式解析：容器网络的外网访问（NAT）**

容器 IP（私有）→ 访问外网：

$$\text{源地址} = \text{容器私有 IP} \xrightarrow{\text{NAT}} \text{宿主 IP}$$

- 容器发往外网的包，源地址被 NAT 成宿主 IP。
- 回包经 NAT 反向映射回容器——**容器「借」宿主的公网身份上网**。
- 隔离：外部世界只看到宿主，看不到容器（**这是隔离也是限制**）。

**辨析｜易错点：** 「容器有独立 IP 就是独立机器」是简化理解。**容器的 IP 是虚拟的（veth + 网桥分配），它的「独立网络」建立在宿主的虚拟网络之上**——容器网络性能不如宿主直连（要过 veth + 网桥 + NAT）。**高性能场景用 host 模式或 Macvlan**（让容器直接有宿主网络接口）。**「看起来独立」与「物理独立」不同。**

## 3 核心对比表：两个 Namespace 的隔离对象

| 维度 | Mount Namespace | Network Namespace |
| --- | --- | --- |
| 隔离什么 | 挂载点/文件系统视图 | 网络栈（网卡/IP/路由） |
| 容器里效果 | 自己的 rootfs | 自己的 IP/网卡 |
| 关键机制 | pivot_root、挂载传播 | veth、网桥、NAT |
| 跨容器共享 | 镜像层（只读共享） | 网桥互连 |
| 与外界的边界 | 挂载点树 | NAT + 路由 |

**设计启示**：Mount 与 Network Namespace 展示了「**虚拟化视图**」的两种范式——**Mount 用「挂载点树的分叉」隔离文件，Network 用「虚拟网卡 + 网桥」隔离网络**。它们都建立在「**内核把真实资源抽象成可复制的视图**」之上——这是容器「轻量虚拟化」的本质。

## 4 小结

- **Mount Namespace**：隔离挂载点视图——pivot_root 切换根到容器 rootfs，容器挂载不影响宿主。
- 挂载传播（共享/私有）控制跨 Namespace 的挂载可见性。
- **Network Namespace**：隔离完整网络栈——独立网卡、IP、路由。
- 容器网络 = **veth pair + 网桥 + NAT**；Docker 有 bridge/host/none/container 四种模式。
- 容器的「独立」是虚拟视图——性能与安全需额外考虑（host 模式、NAT 边界）。

在下一节，我们看容器的资源限制——**Cgroup：CPU、内存与 I/O 资源控制（v1 与 v2）**。
