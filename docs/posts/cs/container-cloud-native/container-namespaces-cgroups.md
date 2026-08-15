---
title: 容器原理：namespaces 与 cgroups
date: 2026-08-07
---

# 容器原理：namespaces 与 cgroups

<div class="epigraph">
<p>容器不是轻量级的虚拟机——它们是操作系统的进程，只是被看得见的世界被裁剪过。</p>
<footer>—— Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 容器与云原生 ｜ Burns Ch.2 ｜ 2026-08-07</p>
</div>

## 为什么从容器原理开始

「从极限到大模型」的第三级里，你已经学过操作系统与进程、网络与文件系统。容器正是这几门课在**进程隔离**这一点上的交汇：一个 Docker 容器、一个 Kubernetes Pod 里的容器，本质上都是一个 Linux 进程——只不过它的视角被裁剪了，它的资源被限定了。把这条原理讲清楚，后面的一切（镜像、Pod、调度、网络）都有了地基。<span class="marginnote">容器技术的关键洞察：它没有发明新的执行单元，复用内核已有的进程模型，只是给进程装上了「特制的眼镜」（namespaces）和「限量的饭票」（cgroups）。</span>

## 1 从虚拟机到容器

虚拟机的思路是**硬件级虚拟化**：宿主机上跑一个 hypervisor（如 KVM、VMware），每个虚拟机里装一个完整的客户操作系统（guest OS），各自拥有一套完整的内核。隔离彻底，但开销巨大——每个 VM 都要吃一份内核、一份进程表、一份文件系统缓存。

容器的思路是**操作系统级虚拟化**：所有容器共享宿主机的同一个内核，容器之间只是进程级隔离。它借用两个内核机制：

**namespaces**：决定「这个进程能看到什么」——进程表、网络栈、挂载点、主机名。
- **cgroups**：决定「这个进程能用多少」——CPU、内存、I/O、进程数。

| 维度 | 虚拟机 | 容器 |
| --- | --- | --- |
| 隔离级别 | 硬件 / 内核级 | 操作系统级（进程） |
| 内核 | 每个 VM 一个 | 共享宿主机内核 |
| 启动开销 | 秒级，GB 级内存 | 毫秒级，MB 级内存 |
| 密度 | 低 | 高 |
| 隔离强度 | 强（内核天然隔开） | 较弱（依赖内核安全机制） |

这组「毫秒级 / MB 级 vs 秒级 / GB 级」的差距，正是容器能在同一台机器上把密度做到上百个的原因。但代价也在表格最后一行：**共享内核的隔离强度取决于内核自身的安全机制**，这会在第五节专门辨析。

**核心概念：容器（container）**：一个运行在宿主内核之上、通过 namespaces 获得独立视图、通过 cgroups 获得资源上限的进程（或进程组）。

## 2 隔离视图：namespaces

namespaces 是 Linux 内核提供的一族机制，每个 namespace 是**一种系统资源的独立视图**。进程通过 `clone(2)`（带 `CLONE_NEW*` 标志）或 `unshare(2)` 创建新 namespace，用 `setns(2)` 加入已有 namespace。

| Namespace | 隔离的资源 | 内核引入版本 |
| --- | --- | --- |
| `mnt` | 挂载点 / 文件系统树 | 2.4.19 |
| `pid` | 进程号 | 2.6.24 |
| `net` | 网络栈（接口、路由、端口） | 2.6.24 |
| `uts` | 主机名与域名 | 2.6.19 |
| `ipc` | System V IPC、消息队列 | 2.6.19 |
| `user` | 用户与组 ID 映射 | 3.8 |
| `cgroup` | cgroup 根目录 | 4.6 |
| `time` | 启动时间与单调时钟 | 5.6 |

最直观的是 **PID namespace**：容器里的进程看到自己的 PID 编号从 1 开始，看不到宿主机上其他进程；反过来宿主机能看到容器里的进程（此时它的 PID 是宿主机视角的编号）。「容器里的进程 1」对应宿主机里的一个普通进程——**PID namespace 并没有创造新的进程，只是换了一张进程表的投影**。<span class="marginnote">这个「投影」视角是理解容器的关键隐喻：namespace 不复制任何内核对象，只改变「看」的方式。你在容器里执行 <code>ps</code> 看到的列表，与宿主机看到的列表来自同一个进程，只是过滤与重编号的结果。</span>

`net` namespace 同样重要：每个容器有自己的回环接口、iptables 规则与端口空间，所以两个容器可以各自监听 80 端口而不冲突——这是后面 Pod 网络（每 Pod 一个 IP）的基石。

**辨析｜易错点：容器镜像「看着像新系统」，其实是同一批内核对象**。初学者常被 `docker run` 的 `-h`（hostname）、`--network`、`--pid` 这些参数迷惑。它们不过是创建/加入对应 namespace 的快捷开关：`-h` 决定 `uts` namespace 里的主机名，`--network none` 表示新容器只挂自己的 `net` namespace 而不接入宿主机网络。**没有新内核、没有新设备，只有视角的裁剪。**

把参数与机制对应起来，就不再有「黑魔法」的感觉：

| `docker run` 参数 | 对应机制 | 效果 |
| --- | --- | --- |
| `-h <name>` | `uts` namespace | 设置容器内主机名 |
| `--network none` | `net` namespace | 独立网络栈，不接宿主机网络 |
| `--pid host` | `pid` namespace | 与宿主机共享进程表 |
| `--userns-remap` | `user` namespace | 容器内 root 映射为非特权用户 |

怎么看一个进程挂了哪些 namespace？Linux 把每个进程的 namespace 符号链接暴露在 `/proc/<pid>/ns/` 下——`ls -l /proc/1/ns/` 会列出 `mnt`、`pid`、`net`、`uts` 等条目，每条都指向一个 namespace 编号。两个进程若指向同一个编号，就说明它们共享该 namespace。这套 `/proc` 视图是排障时的「透视镜」：容器里的进程到底隔离了哪几层、和宿主机共享了哪几层，一眼便知。

## 3 资源约束：cgroups

隔离解决「看得到」，还缺「用得起」。如果没有 cgroups，一个容器可以写满宿主内存、打满全部 CPU 核。**cgroups 把进程分进层级化的组，对每组施加资源上限**。

cgroup v1 里每种资源是一棵独立的树（cpu、memory 各管各的）；**cgroup v2（Kubernetes 与主流运行时默认）采用统一层级（unified hierarchy）**，所有控制器挂在同一棵树上，子组继承父组的上限。<span class="marginnote">cgroup v2 的「统一层级」哲学与容器世界里「一切皆树」的直觉一致：组的嵌套 = 配额嵌套，叶组是实际运行的进程。核心控制器包括 <code>cpu</code>（权重与带宽）、<code>memory</code>（上限与回收）、<code>io</code>（磁盘带宽）、<code>pids</code>（进程数上限）。</span>

几个最常用的控制参数：

- `cpu.weight`：CPU 份额权重，默认 1024，相对值而非绝对值。
- `cpu.max`：`$quota period$` 形式，限制带宽。
- `memory.max`：内存硬上限，超限触发回收或 OOM。
- `pids.max`：组内最多可运行的进程数（防止 fork 炸弹）。

## 4 公式解析：CPU 份额如何分配

容器编排器给容器设的 CPU 需求（Kubernetes 里叫 `requests`）最终落到 cgroup 的权重上。当 CPU 争抢时，分配遵循**加权比例**：

$$
\text{share}_p = \frac{w_p}{\sum_{q \in \text{group}} w_q} \times W_{\text{group}}
$$

- $w_p$：容器 $p$ 的 `cpu.weight`（默认 1024）。
- $\sum_{q \in \text{group}} w_q$：同一父组下所有活跃容器的权重之和。
- $W_{\text{group}}$：父组可用的 CPU 总量（通常就是核数）。

三步拆解：

- **第一步，理解权重是相对值**：$w_p$ 本身没有绝对含义，只有「占总量多少」才被消费。设两个容器，权重 2048 与 1024，那么前者分到 $\frac{2048}{3072} \approx \frac{2}{3}$ 的 CPU，后者 $\frac{1}{3}$。
- **第二步，总量由父组决定**：分母是同一层级所有容器的权重和，所以「你分到多少」由邻居决定——新增一个容器会按比例摊薄所有人，这正是「超额分配（overcommit）」允许的原因。
- **第三步，超限时也守恒**：CPU 空闲时任何容器都可以冲到上限（权重只决定**争抢时**的比例）；只有争抢时 $\sum \text{share}_p = W$ 才严格成立。这也是为什么 Kubernetes 中 CPU `requests` 被当作「保证值」，`limits` 才是硬顶。

再算一个具体的数字。4 核宿主机上跑三个容器，权重分别是 1024、1024、2048：权重和 $= 4096$，三者的份额依次是 $\frac{1024}{4096} \times 4 = 1$、$1$、$2$ 个核。若第四个容器（权重 1024）加入，权重和变 $5120$，前三个各降到约 $0.8$ 核——**新邻居一进来，人人让出一份**。若容器 A 只声明了 `limits` 而没有 `requests`，它的实际行为是「有饭就吃」——这正是超额分配产生的原因。<span class="marginnote">Kubernetes 里 CPU 是「可分时」的资源：<code>requests</code> 决定保留份额，<code>limits</code> 决定硬顶；而内存是「不可让渡」的——<code>requests</code> 就是调度依据，超 <code>limits</code> 直接 OOM。CPU 与内存的不对称，是资源管理里最常被忽略的一条。</span>

## 5 辨析｜易错点：容器安全 ≠ 虚拟机安全

**最容易犯的错误，是把容器当成「小而快的虚拟机」**。二者在安全模型上完全不同：

- 容器共享宿主机内核，一旦应用通过漏洞拿到内核权限（或错误挂载了宿主机目录、`/proc` 等敏感路径），就等于拿到了宿主机权限，这不是「逃出容器」，而是「从容器内打穿了共享的内核」。
- 因此生产环境的加固依赖**纵深防御**：`user` namespace（把容器内 root 映射成宿主机非特权用户）、`seccomp`（限制系统调用集合）、`capabilities`（默认丢弃容器用不到的 capabilities）、只读根文件系统。这些策略在《安全（RBAC/NetworkPolicy/Secrets）》一课里会系统展开。

**另一个易错点**：容器重启 ≠ 机器重启。重启只是重新执行一次进程（并清空可写层以外的状态），内核不重启，宿主机其他进程不受影响——这也是「容器崩溃了，主机还在跑」这句运维谚语的由来。

**还要记住**：namespaces 是「隔离」而不是「安全」——`user` namespace 出现之前，容器内的 root 在宿主机眼里仍是 root，内核不区分「容器内 root」与「宿主机 root」。**隔离解决的是可用性（互不干扰），安全还需要额外的隔离机制**，这两件事别混为一谈。

## 6 小结

- 容器 = **namespaces（隔离视图）** + **cgroups（资源上限）** + 共享宿主机内核的进程。
- namespaces 有 8 种，覆盖挂载、进程表、网络、主机名、IPC、用户、cgroup 根、时钟。
- cgroups v2 用**统一层级**管理 CPU、内存、I/O、进程数，配额沿树继承。
- CPU 分配是**加权比例**：`share = 权重 / 同层权重和 × 总带宽`；CPU 的 requests/limits 与内存的行为并不对称。
- 容器的安全边界**弱于虚拟机**，必须靠 user namespace、seccomp、capabilities 纵深加固。
- 隔离 ≠ 安全：namespaces 解决互不干扰，安全还需要额外的内核机制。

在下一节，我们回答：这些隔离的进程，是怎么被打包、分发、还原成「镜像」的——这就是 **Docker 镜像与运行时**。
