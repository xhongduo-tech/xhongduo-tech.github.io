## 核心结论

Linux 容器的“隔离”不是一个单一功能，而是两类内核机制的组合。

Namespace 是命名空间，白话解释是“给进程一套独立看到的系统视图”。它把原本全局共享的 PID、网络设备、挂载点、主机名、进程间通信资源切成多个彼此隔离的视图。容器里的 `ps` 只能看到同一 PID namespace 的进程，`ip addr` 看到的是自己的网卡和路由，`hostname` 看到的是自己的 UTS 名称。

Cgroup 是控制组，白话解释是“给一组进程写资源使用规则”。它不负责“看见什么”，而负责“最多能用多少、竞争时按什么比例分”。在 cgroup v2 下，`cpu.max` 给 CPU 硬配额，`cpu.weight` 给 CPU 相对权重，`memory.high` 提前触发内存回收，`memory.max` 给出内存硬上限，`io.weight` 决定 I/O 竞争时的相对优先级。

两者的分工可以直接对照：

| 机制 | 负责内容 | 典型效果 | 关键接口 |
|---|---|---|---|
| Namespace | 资源视图隔离 | 容器只看到自己的进程、网卡、挂载点、主机名、IPC 对象 | `clone()`、`unshare()`、`setns()` |
| Cgroup v2 | 资源使用控制 | 限 CPU、限内存、调 I/O 权重 | `/sys/fs/cgroup/*` 控制文件 |
| PID namespace | 进程编号隔离 | 容器内 PID 1 是自己的 init | `/proc` 视图变化 |
| Network namespace | 网络栈隔离 | 独立网卡、IP、路由表、iptables | `ip addr`、`ip route` |
| Mount namespace | 挂载点隔离 | 不同容器看到不同文件系统树 | `mount` |
| UTS namespace | 主机名隔离 | 容器内 `hostname` 独立 | `uname`、`hostname` |
| IPC namespace | IPC 资源隔离 | 隔离共享内存、消息队列、信号量 | SysV IPC / POSIX IPC |
| CPU 控制 | 算力限制 | `cpu.max` 限最大 CPU，用 `cpu.weight` 控竞争比例 | `cpu.max`、`cpu.weight` |
| Memory 控制 | 内存限制 | `memory.high` 回收，`memory.max` 硬限 | `memory.high`、`memory.max` |
| I/O 控制 | 块设备竞争控制 | 降低某容器磁盘竞争优先级 | `io.weight` |

最短结论是：Namespace 建“边界”，Cgroup 管“额度”。容器运行时例如 `runc`、`containerd` 把这两者组装起来，才形成今天常见的容器隔离环境。

---

## 问题定义与边界

问题可以表述为：在同一台 Linux 机器上，多个应用如何共用一个内核，同时又做到“互相看不见太多、互相抢不到太多”。

对零基础读者，先把问题压缩成一句话：

| 机制 | 新手版理解 |
|---|---|
| Namespace | 给每个容器一套自己看到的系统图 |
| Cgroup | 给每个容器写资源用量的规则 |

这两个机制解决的是不同维度的问题。

第一类问题是“视图冲突”。如果没有 PID namespace，容器 A 里的进程会直接看到容器 B 的 PID；如果没有 network namespace，多个容器会直接共用网络栈；如果没有 mount namespace，一个容器对挂载点的修改可能影响其他容器。

第二类问题是“资源争抢”。即使看不见彼此，多个容器仍共享同一个内核调度器、同一块物理内存、同一块磁盘。没有 cgroup，某个容器照样可以把 CPU 吃满、把内存打爆、把磁盘写满，导致其他容器性能下降甚至被连带拖死。

本文边界只讨论内核层机制：

- 讨论 Namespace 与 cgroup v2。
- 讨论容器运行时如何把它们拼起来。
- 不展开 Docker CLI、Kubernetes 调度器、CNI 网络插件、镜像分发系统。
- 不展开 seccomp、capabilities、LSM 这类安全机制，它们重要，但不是本文主轴。

可以把架构抽象成下面这张伪图：

```text
应用进程
   |
   v
容器运行时（runc/containerd）
   |-------------------- 创建 namespace --------------------|
   |-------------------- 配置 cgroup v2 --------------------|
   v
Linux 内核
   |- PID / NET / MNT / UTS / IPC namespace：隔离“看到什么”
   |- /sys/fs/cgroup/<group>/...         ：限制“能用多少”
   v
CPU / Memory / Disk / Network Stack / Filesystem
```

“单核 Linux”这个设定也很有帮助，因为它把 CPU 限制讲清楚了。在单核上，`cpu.max = 50000 100000` 的意思很直观：在每 100000 微秒的周期里，最多给这组进程 50000 微秒 CPU 时间，也就是最多 50% 的一颗核。

---

## 核心机制与推导

先看 Namespace。

Namespace 的本质是内核给进程附加一个“命名空间上下文”。当进程通过 `clone()` 或 `unshare()` 创建新 namespace 后，它之后访问某些全局资源时，内核不再返回全局默认视图，而是返回“该 namespace 对应的那一份”。

以 PID namespace 为例：

1. 宿主机上有很多进程。
2. 运行时创建新的 PID namespace。
3. 容器主进程进入该 namespace。
4. 该 namespace 内部重新编号，容器主进程通常看到自己是 PID 1。
5. 容器里执行 `ps`，只能枚举这个 namespace 内的进程。

玩具例子可以这样理解：

- 宿主机有进程 `100, 101, 102, 103`
- 新容器只启动了两个进程
- 容器内部看到的是 `1, 7`
- 宿主机仍看到它们真实宿主 PID，例如 `24501, 24517`

也就是说，“同一个进程有宿主视角的 PID，也有容器视角的 PID”。这是 namespace 隔离最容易让新手混淆的点。

再看 Network namespace。网络命名空间隔离的是完整网络栈，白话解释是“不是只隔离一个 IP，而是隔离整套网卡、地址、路由、端口空间”。因此容器内执行：

```sh
ps
ip addr
hostname
cat /sys/fs/cgroup/cpu.max
```

会分别呈现不同层面的隔离结果：

- `ps`：只能看到容器自己的 PID 视图。
- `ip addr`：看到独立的网卡和地址。
- `hostname`：看到独立主机名。
- `cat /sys/fs/cgroup/cpu.max`：看到当前 cgroup 的 CPU 配额配置。

Namespace 只负责“逻辑空间分隔”，不负责“资源多少”。资源控制要看 cgroup v2。

cgroup v2 的核心思想是把一组进程挂到一棵统一层级树上。每个目录就是一个控制组，每个控制文件就是一条规则。运行时在 `/sys/fs/cgroup/<容器名或容器ID>/` 下创建目录，然后写入控制文件。

最常见的三个资源控制如下。

第一，CPU 硬配额。

`cpu.max` 形式是：

$$
cpu.max = quota\ period
$$

其中：

- `quota` 是一个周期内最多可用的 CPU 时间，单位通常是微秒。
- `period` 是统计周期长度。

因此单核下：

$$
CPU\ 使用上限 = \frac{quota}{period}
$$

如果：

$$
quota = 50000,\ period = 100000
$$

那么：

$$
\frac{50000}{100000} = 0.5
$$

表示这组进程最多使用 50% 的一颗核。

第二，CPU 相对权重。

`cpu.weight` 不是硬限制，而是竞争比例。若两个 cgroup 同时抢 CPU，分配倾向近似按：

$$
share_i = \frac{weight_i}{\sum weight}
$$

例如两个容器权重分别为 100 和 200，则竞争时它们大致按 1:2 分 CPU 时间。注意这里的关键词是“竞争时”。如果系统空闲，低权重组仍可能跑满可用 CPU，这就是为什么 `cpu.weight` 不能替代 `cpu.max`。

第三，内存双层限制。

- `memory.high`：高水位线，白话解释是“超过这里先开始压制和回收”，通常触发 reclaim，不直接 OOM。
- `memory.max`：硬上限，白话解释是“再往上就不允许了”，超出后可能触发 cgroup 内 OOM。

这两个阈值配合使用更合理。比如：

- `memory.high = 1G`
- `memory.max = 2G`

表示容器到 1G 后就会遇到回收压力，超过 2G 则直接触发更强硬的限制。

I/O 也是类似思路。`io.weight` 是相对权重，不是固定带宽值。若某容器设为 `default 50`，而另一容器保持默认 100，那么在同一块块设备上竞争时，它的相对优先级接近默认的一半。

把运行顺序按机制串起来，大致是：

| 步骤 | 动作 | 目的 |
|---|---|---|
| 1 | 解析 OCI 配置 | 读取 namespace、挂载、资源限制 |
| 2 | `clone()` / `unshare()` | 创建 PID/NET/MNT/UTS/IPC namespace |
| 3 | 准备 rootfs 与挂载 | 让容器看到自己的文件系统 |
| 4 | 在 `/sys/fs/cgroup` 创建目录 | 建立对应 cgroup |
| 5 | 写入 `cpu.max`、`memory.max` 等 | 设置资源规则 |
| 6 | 把进程 PID 写入 `cgroup.procs` | 将容器进程纳入控制组 |
| 7 | `execve()` 启动业务进程 | 开始在新视图和新限额下运行 |

这里最重要的推导关系是：Namespace 改的是“访问资源时返回哪份视图”，cgroup 改的是“调度和记账时允许用多少资源”。

---

## 代码实现

先看一个最小化 shell 版本。它不是完整容器运行时，但足以说明 cgroup v2 的控制方式。

```sh
# 假设当前系统使用 unified cgroup v2
# 创建一个控制组目录
mkdir -p /sys/fs/cgroup/demo-container

# 把 CPU 限制为单核 50%
# 100000us 周期内，只允许跑 50000us
echo "50000 100000" > /sys/fs/cgroup/demo-container/cpu.max

# 内存在 1G 开始触发回收压力
echo "1G" > /sys/fs/cgroup/demo-container/memory.high

# 内存硬上限 2G
echo "2G" > /sys/fs/cgroup/demo-container/memory.max

# I/O 权重设为默认的一半
echo "default 50" > /sys/fs/cgroup/demo-container/io.weight

# 把目标进程加入该 cgroup
# 假设 12345 是容器主进程在宿主机视角下的 PID
echo 12345 > /sys/fs/cgroup/demo-container/cgroup.procs

# 读取配置验证
cat /sys/fs/cgroup/demo-container/cpu.max
cat /sys/fs/cgroup/demo-container/memory.high
cat /sys/fs/cgroup/demo-container/memory.max
cat /sys/fs/cgroup/demo-container/io.weight
```

接着看一个简化版伪代码，描述 `runc` 一类运行时的关键顺序：

```text
load_oci_spec()

child_pid = clone(
  flags = CLONE_NEWPID |
          CLONE_NEWNET |
          CLONE_NEWNS  |
          CLONE_NEWUTS |
          CLONE_NEWIPC
)

if child_pid == 0:
    setup_mounts()
    setup_hostname()
    exec_container_init()
else:
    mkdir("/sys/fs/cgroup/<id>")
    write("cpu.max", "50000 100000")
    write("memory.high", "1G")
    write("memory.max", "2G")
    write("io.weight", "default 50")
    write("cgroup.procs", child_pid)
    wait(child_pid)
```

再给一个可运行的 Python 玩具例子。它不直接操作内核，而是把 cgroup 的 CPU 配额和权重分配逻辑算出来，帮助理解公式。

```python
def cpu_quota_ratio(quota_us: int, period_us: int) -> float:
    assert quota_us > 0
    assert period_us > 0
    return quota_us / period_us


def cpu_share(weights, target):
    assert target in weights
    total = sum(weights.values())
    assert total > 0
    return weights[target] / total


# 玩具例子 1：cpu.max = 50000 100000 => 50% 单核
ratio = cpu_quota_ratio(50000, 100000)
assert ratio == 0.5

# 玩具例子 2：两个容器按权重 100:200 分 CPU
weights = {"A": 100, "B": 200}
assert abs(cpu_share(weights, "A") - (1 / 3)) < 1e-9
assert abs(cpu_share(weights, "B") - (2 / 3)) < 1e-9

print("quota ratio:", ratio)
print("A share:", cpu_share(weights, "A"))
print("B share:", cpu_share(weights, "B"))
```

真实工程例子可以放到 `containerd + runc` 链路里理解。

1. 用户通过 Docker 或 Kubernetes 发起启动请求。
2. `containerd` 接收请求，生成或读取 OCI 规范。
3. `containerd` 调用 `runc`。
4. `runc` 创建 namespace，准备 rootfs，配置挂载。
5. `runc` 在 `/sys/fs/cgroup` 下创建容器控制组并写入 `cpu.max`、`memory.max` 等。
6. 业务进程作为容器 init 启动。
7. 从此它看到的是隔离视图，消耗的是受控额度。

因此容器技术在实现上并不神秘。核心不是“模拟一台机器”，而是“让一个普通 Linux 进程带着隔离视图和受控额度运行”。

---

## 工程权衡与常见坑

工程上最常见的误解，是把“配置写进去了”误认为“行为已经完全符合直觉”。实际上每个控制项都有边界。

先看常见坑表：

| 坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 把 `memory.high` 当成 OOM 线 | 容器变慢但没被杀 | 它主要触发 reclaim 和 throttle，不是硬杀 | 结合 `memory.max`、监控 `memory.events` |
| 只配 `cpu.weight` 不配 `cpu.max` | 空闲时容器照样能吃满 CPU | `cpu.weight` 只在竞争时生效 | 需要硬上限时配合 `cpu.max` |
| cgroup 驱动不一致 | 资源规则混乱、层级不一致 | kubelet 与 runtime 分别管理不同路径 | 统一用 `systemd` 或统一用 `cgroupfs` |
| 忽略宿主机视角 PID | 排障时找错进程 | 容器内 PID 与宿主机 PID 不同 | 用宿主机 `ps`、`crictl`、`ctr` 对照 |
| 误以为 namespace 等于强安全隔离 | 容器共享内核，内核漏洞仍可打穿 | Namespace 是隔离视图，不是独立内核 | 高安全场景考虑微虚拟机或虚拟机 |
| 只限制内存不看回收行为 | 延迟抖动明显 | reclaim 会增加 CPU 和 I/O 压力 | 监控 PSI、page fault、major fault |

`memory.high` 是最值得单独说明的一项。它的设计目标不是直接杀死进程，而是提前给系统一个“开始收缩”的机会。所以真实现象常常不是 OOM，而是延迟变差、吞吐下降、fault 增多。对在线业务来说，这比直接 OOM 更隐蔽，因为服务可能没有挂，但 SLA 已经恶化。

建议至少监控这几类指标：

| 指标 | 用途 |
|---|---|
| `memory.current` | 看当前内存占用 |
| `memory.events` | 看 high/max/oom 是否发生 |
| PSI 内存压力 | 看回收是否持续影响调度 |
| CPU 使用率与 throttled 时间 | 看 `cpu.max` 是否经常打满 |
| I/O 等待时间 | 看 `io.weight` 调整后是否产生饥饿 |

Kubernetes 场景下还有一个经典问题：`kubelet` 使用 `systemd` driver，而容器运行时仍使用 `cgroupfs`。结果是同一组进程可能被两个不同控制体系同时组织，路径和层级不统一，排障时会出现“你改了一个目录，实际生效的是另一个目录”的问题。实践上通常建议统一驱动，现代发行版中常见选择是统一到 `systemd`。

还有一个容易忽略的点：Namespace 隔离的是“默认视图”，不是“绝对不存在”。例如宿主机管理员仍然可以从宿主机视角看到容器进程、网络接口和 cgroup 层级。容器内部“看不见”，不等于宿主机内核“不知道”。

---

## 替代方案与适用边界

先比较 cgroup v1 与 v2。

| 方案 | 特征 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| cgroup v1 | 多子系统分散挂载 | 兼容老系统和老工具 | 层级分裂，配置心智负担大 | 历史系统兼容 |
| cgroup v2 | 统一层级 | 管理一致，容器平台更容易统一治理 | 老旧工具链迁移有成本 | 新部署容器平台，推荐 |
| 虚拟机 | 独立客户机内核 | 隔离更强，边界更硬 | 启动慢、开销更高 | 高安全、多租户强隔离 |
| Namespace + Cgroup 容器 | 共享宿主机内核 | 启动快、开销低、部署密度高 | 隔离强度弱于虚拟机 | 微服务、CI/CD、弹性平台 |

为什么 cgroup v2 更适合现代容器平台？因为它把控制接口统一到了同一棵树上，资源控制模型更一致，运行时、systemd、监控系统之间更容易形成统一约定。对于 Docker、containerd 这类现代运行时，v2 通常是更自然的选择。

但这不代表容器永远优于虚拟机。如果场景要求：

- 多租户互不信任
- 内核攻击面必须尽量缩小
- 合规要求接近“硬边界”

那么虚拟机或微虚拟机通常更合理。代价是多一层客户机内核，启动成本、内存占用、镜像体积、运维复杂度都会上升。

可以用一句话概括取舍：

- 要更高密度和更快启动，用 Namespace + Cgroup。
- 要更强隔离和更硬边界，用虚拟机栈。

因此 Namespace 与 Cgroup 的适用边界很清楚：它们非常适合“同一组织内的应用隔离”和“资源治理”，但不是一切安全问题的终点。

---

## 参考资料

1. Linux Kernel 官方文档 `Control Group v2`
   作用：支撑 `cpu.max`、`cpu.weight`、`memory.high`、`memory.max`、`io.weight` 的语义与行为边界，尤其是 `memory.high` 触发回收而非直接 OOM 的理解。

2. DIY 容器教程中的 Linux Namespace 分类说明
   作用：帮助建立 PID、Network、Mount、UTS、IPC namespace 的基础分类认知，适合理解“隔离的是哪类系统视图”。

3. Atlantbh 关于 Docker 底层机制的文章 `How Docker Containers Work Under the Hood`
   作用：帮助串起 `containerd`、`runc`、OCI 配置、namespace 创建与 cgroup 写入之间的运行时链路。
