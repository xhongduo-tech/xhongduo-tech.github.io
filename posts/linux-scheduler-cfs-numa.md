## 核心结论

Linux 调度器解决的是“谁先用 CPU”这个问题。CPU 可以理解为处理器真正执行指令的时间片资源。对普通任务，Linux 主要依赖 CFS（Completely Fair Scheduler，完全公平调度器）分配 CPU；对必须优先响应的任务，Linux 提供实时调度类 `SCHED_FIFO` 和 `SCHED_RR`。

最关键的结论有三条：

1. CFS 不直接按“谁先来”调度，而是按每个任务的 `vruntime` 决定下一个运行者。`vruntime` 可以白话理解为“按权重折算后的已使用 CPU 时间”，谁“欠 CPU”更多，谁先运行。
2. 实时任务永远先于普通任务。只要运行队列里存在 `SCHED_FIFO` 或 `SCHED_RR` 任务，普通的 `SCHED_OTHER`/CFS 任务就要让路。
3. NUMA（Non-Uniform Memory Access，非一致内存访问）会影响调度效果。白话说，多路服务器里的内存不是离所有 CPU 都一样近；线程跑在一个节点、内存却分配在另一个节点时，访问会更慢。

一个玩具例子：有三辆车比赛，表里记录它们已经跑了多少公里。CFS 每次让“跑得最少”的车继续跑；如果来了一辆救护车，它不排队，直接插到最前，这就是实时调度的优先级语义。

| 策略 | 调度对象 | 优先级位置 | 是否时间片轮转 | 典型行为 |
|---|---|---:|---|---|
| CFS / `SCHED_OTHER` | 普通进程 | 低于实时类 | 是，但由公平模型控制 | 谁 `vruntime` 小谁先运行 |
| `SCHED_FIFO` | 实时进程 | 高于 CFS | 否 | 一直运行到阻塞、退出或被更高优先级任务抢占 |
| `SCHED_RR` | 实时进程 | 高于 CFS | 是 | 同优先级实时任务按固定时间片轮转 |

---

## 问题定义与边界

调度器的目标不是“绝对平均”，而是在多核机器上尽量逼近“理想公平”。理想公平的意思是：如果两个可运行任务权重一样，那么长期看它们应大致各拿到一半 CPU；如果权重不同，高权重任务应拿到更多份额。

这里有三个边界必须先说清：

| 维度 | 作用 | 边界 |
|---|---|---|
| `nice` 值 | 调整普通任务权重 | 只影响 CFS 任务，不影响实时类优先级 |
| 实时优先级 | 决定 `SCHED_FIFO/RR` 的先后 | 一旦配置不当，普通任务可能长期得不到 CPU |
| NUMA 拓扑 | 决定本地/远程内存访问成本 | 只在多 NUMA 节点机器上显著，单路小机器影响有限 |

新手版可以这样理解：三个人分蛋糕，CFS 会优先照顾“吃得少的人”；实时任务像 VIP，先吃；NUMA 像厨房分布在不同楼层，离你远的厨房拿食材更慢。

调度器还只处理 runnable task，也就是“已经准备好、只差 CPU”的任务。一个正在等磁盘、等网络、等锁的任务，不会参与当前 CPU 竞争。因此，很多“程序变慢”并不一定是调度器问题，也可能是 I/O、锁竞争或远程内存访问问题。

---

## 核心机制与推导

CFS 的核心变量是 `vruntime`。它不是实际运行时间，而是按权重归一化后的运行时间：

$$
vruntime += delta\_exec \times \frac{NICE\_0\_LOAD}{load.weight}
$$

其中：

- `delta_exec`：本次真实运行了多久
- `load.weight`：任务权重，由 `nice` 决定
- `NICE_0_LOAD`：`nice=0` 的基准权重

白话解释：高优先级任务权重大，所以同样跑 1ms，它的 `vruntime` 增长得更慢；低优先级任务权重小，所以同样跑 1ms，它的“已占用份额”会被记得更重。

| `nice` | 常见权重示意 | 相同 `delta_exec` 下 `vruntime` 增长 |
|---:|---:|---|
| -5 | 较大 | 较慢 |
| 0 | 1024 | 基准 |
| +5 | 较小 | 较快 |

这就是为什么 CFS 能同时做到“高优先级多拿 CPU”和“长期不失衡”。任务被放在红黑树里。红黑树可以白话理解为“一种能快速找到最小值的平衡有序树”。树节点按 `vruntime` 排序，最左边的节点就是当前最该运行的任务。

玩具例子：

- 任务 A：`nice=-5`
- 任务 B：`nice=0`
- 任务 C：`nice=+5`

初始时三者 `vruntime=0`。若都运行一小段真实时间：

- A 因权重大，`vruntime` 只加少量
- B 加中等
- C 因权重小，`vruntime` 加得更多

结果就是 A 更容易继续留在红黑树左侧，从而获得更多 CPU。这里不是“插队”，而是“同一套公平公式下折算后的结果”。

CFS 还是 per-CPU runqueue，也就是“每个 CPU 核心有自己的运行队列”。这样做的原因是跨核共享一个全局队列锁竞争太重。内核会在局部公平和全局负载均衡之间折中：先在本核选最小 `vruntime`，再通过周期性 balance 和唤醒迁移，避免某个核太忙、另一个核太闲。

实时类的规则更直接：

- `SCHED_FIFO`：同优先级下，谁先运行谁一直跑，直到主动阻塞、退出，或被更高优先级实时任务抢占。
- `SCHED_RR`：同优先级下按时间片轮转，时间片到就换同优先级下一个任务。

因此调度类本身就形成了优先级链：实时类在前，CFS 在后。这是 Linux 调度“先分类，再类内选择”的基本结构。

---

## 代码实现

内核里，CFS 主要在 `sched_fair.c`，实时类主要在 `sched_rt.c`。即使不读完整源码，也要抓住两个动作：

1. 更新当前任务的 `vruntime`
2. 从可运行任务集合里挑出最合适的下一个任务

简化后的伪代码如下：

```c
task *pick_next_task() {
    if (has_runnable_rt_task()) {
        return pick_next_task_rt();
    }
    return pick_next_task_fair();
}
```

CFS 类内的核心流程可以写成：

```c
void update_curr(task *curr, u64 delta_exec) {
    curr->vruntime += delta_exec * NICE_0_LOAD / curr->weight;
}

task *pick_next_task_fair(rb_tree *tree) {
    return leftmost_node(tree);  // vruntime 最小的任务
}
```

再看一个可运行的 Python 玩具实现。它不等于内核源码，但能把公式和选择逻辑跑通。

```python
from dataclasses import dataclass

NICE_0_LOAD = 1024
WEIGHTS = {-5: 3121, 0: 1024, 5: 335}

@dataclass
class Task:
    name: str
    nice: int
    vruntime: float = 0.0

    @property
    def weight(self):
        return WEIGHTS[self.nice]

def update_vruntime(task: Task, delta_exec: float) -> None:
    task.vruntime += delta_exec * (NICE_0_LOAD / task.weight)

def pick_next(tasks):
    return min(tasks, key=lambda t: t.vruntime)

tasks = [Task("A", -5), Task("B", 0), Task("C", 5)]

# 初始都相等，任选最小，Python 会取第一个
assert pick_next(tasks).name == "A"

# 每个任务都运行 3ms
for t in tasks:
    update_vruntime(t, 3.0)

# 高权重任务 vruntime 增长更慢
assert tasks[0].vruntime < tasks[1].vruntime < tasks[2].vruntime

# 下一次应优先选 vruntime 最小的 A
assert pick_next(tasks).name == "A"
```

真实工程例子：在双路服务器上做 CPU 密集型预处理，例如批量特征提取、日志压缩、向量化转换。任务本身几乎不等 I/O，看起来“CPU 打满”。这时如果线程在 NUMA 节点 0 的核上跑，但内存页大量分配在节点 1，CFS 仍然会公平地给它 CPU，但这个“公平运行”的任务会不断访问远程内存，导致吞吐下降。调度公平不等于访问高效，这正是 NUMA 感知必要的地方。

常用控制方式：

```bash
numactl --cpunodebind=0 --membind=0 python preprocess.py
```

这条命令的含义是：线程尽量只在 0 号 NUMA 节点的 CPU 上跑，同时内存也尽量从 0 号节点分配。

---

## 工程权衡与常见坑

第一个坑是把“CPU 占用高”误判为“调度器不公平”。如果 `top` 里一个任务占满核心，不代表 CFS 出错；它可能只是高权重任务，也可能其他任务都在等待 I/O。

第二个坑是滥用实时调度。实时任务的白话含义是“必须比普通任务更早拿到 CPU 的任务”。这不是“让程序更快”的通用开关。`SCHED_FIFO` 配太多，或者优先级给太高，会让内存回收、日志刷盘、文件系统工作线程都拿不到 CPU，机器表面上没死机，实际已经处于半失活状态。

第三个坑是忽略 NUMA。对单机小服务，默认策略通常够用；但在双路或多路机器上，如果是 CPU 密集且内存带宽敏感的程序，不绑核、不绑内存，跨节点访问可能带来显著额外延迟。

| 场景 | 未绑定 NUMA | 绑定 `cpunodebind + membind` |
|---|---|---|
| CPU 密集型批处理 | 线程可能频繁访问远程内存 | 本地访问更多，吞吐更稳定 |
| 延迟敏感服务 | 尾延迟波动更大 | 尾延迟通常更可控 |
| 线程迁移频繁 | 缓存局部性差 | 更容易保持缓存和内存局部性 |

排查时可以按这个顺序看：

| 检查项 | 命令示例 | 看什么 |
|---|---|---|
| 调度策略 | `ps -eo pid,cls,rtprio,ni,cmd` | 是否存在 RT 任务、优先级是否过高 |
| NUMA 命中 | `numastat -p <pid>` | 远程内存访问是否明显 |
| CPU 分布 | `taskset -pc <pid>` | 线程是否被限制或频繁迁移 |
| 拓扑信息 | `numactl --hardware` | 节点数量、CPU 分布、节点距离 |

工程上常见的稳妥做法是：只给极少数、确有实时需求的线程配置 `SCHED_FIFO` 或 `SCHED_RR`，并且配合 CPU 亲和性与 NUMA 绑定；其余任务继续使用 CFS。这样既保留系统响应能力，又避免普通业务被长期饿死。

---

## 替代方案与适用边界

如果业务是 Web 服务、脚本任务、CI 构建、通用后台进程，默认 CFS 往往已经足够。它的优势是简单、鲁棒、整体公平，不需要人工维护复杂优先级。

如果业务是音频处理、工业控制、交易撮合中的关键路径、低延迟采集线程，可以考虑少量实时线程。但前提是你清楚它们的最坏执行时间，并且知道哪些线程必须抢占普通任务。

NUMA 绑定也不是越强越好。过度绑定会降低系统弹性：某个节点忙、另一个节点闲时，任务无法充分利用空闲资源。因此它更适合以下情况：

| 方案 | 适用场景 | 限制 |
|---|---|---|
| 仅用 CFS | 通用服务、轻负载、交互型程序 | 对极低延迟保障有限 |
| `SCHED_FIFO` | 极少量必须立即响应的线程 | 容易饿死普通任务 |
| `SCHED_RR` | 多个同优先级实时线程需轮转 | 仍然高于普通任务，配置不当风险仍大 |
| NUMA 绑定 | 双路/多路服务器上的 CPU 密集任务 | 可能降低跨节点负载均衡弹性 |
| `cgroup cpuset`/`numad` | 容器化或多业务共享机器 | 管理复杂度更高 |

新手版理解可以很直接：实时任务像紧急电话专线，只开几条；普通任务走共享线路；NUMA 绑定像把厨房和餐厅安排在同一层，减少来回搬运。对没有 NUMA 拓扑的小机器，没必要为了“看起来更专业”而强行绑核。

---

## 参考资料

| 资料 | 主题 | 适用对象 | 关键收获 |
|---|---|---|---|
| Linux Kernel Documentation: `sched-design-CFS` | CFS 设计、`vruntime`、红黑树 | 想理解调度核心机制的读者 | 明确 CFS 如何逼近理想公平 |
| Linux Kernel Documentation: scheduler policy 文档 | `SCHED_FIFO`、`SCHED_RR`、优先级语义 | 需要配置实时线程的工程师 | 搞清实时类为何总在 CFS 前面 |
| `numactl` 与 `numastat` 手册 | NUMA 绑定与观测 | 做性能调优的工程师 | 知道如何定位远程内存访问 |
| Intel/ARM 性能优化指南 | NUMA、缓存、本地性优化 | 做服务器调优的读者 | 理解为何“调度公平”不等于“访问高效” |

建议阅读顺序也很固定：

1. 先读 CFS 设计文档，建立 `vruntime` 与红黑树的概念。
2. 再看实时调度策略，理解为什么实时类会抢占普通类。
3. 最后结合 `numactl`、`numastat` 和厂商性能指南，处理多路服务器上的实际性能问题。
