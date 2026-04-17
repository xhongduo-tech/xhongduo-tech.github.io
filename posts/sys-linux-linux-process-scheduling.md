## 核心结论

Linux 默认的普通进程调度器是 CFS，英文是 Completely Fair Scheduler，白话说就是“尽量把 CPU 时间公平分给所有正在等着运行的任务”。

它不先给每个任务写死一个固定时间片，而是为每个可运行任务维护一个 `vruntime`。`vruntime` 是“虚拟运行时间”，白话说就是“按权重折算后的已运行时长”。谁的 `vruntime` 最小，说明谁“欠的 CPU 时间最多”，调度器就优先让谁运行。

CFS 的核心数据结构是红黑树。红黑树是一种平衡二叉搜索树，白话说就是“能在插入、删除、找最小值时保持较稳定速度的有序集合”。CFS 用 `vruntime` 作为键，把所有可运行任务放进这棵树里，每次只取最左节点，也就是 `vruntime` 最小的任务。

普通任务的优先级主要通过 `nice` 调整。`nice` 不是“绝对优先级”，而是“影响 CPU 份额比例的权重开关”。`nice` 越低，权重越高，`vruntime` 增长越慢，因此这个任务在长期内能拿到更多 CPU 时间；`nice` 越高，权重越低，`vruntime` 增长越快，拿到的 CPU 时间更少。

简化地看，CFS 依赖下面这个关系：

$$
vruntime \;+=\; \delta_{exec}\times \frac{NICE\_0\_LOAD}{weight}
$$

其中 $\delta_{exec}$ 是这次实际运行的时间，`weight` 是由 `nice` 映射得到的权重，`NICE_0_LOAD` 可理解为默认 nice=0 的基准权重。

新手可以把它想成一本“谁跑得少谁先来”的账本。所有任务都按“欠账多少”排队，调度器每次只看账本最前面那一个；`nice` 值决定这个任务记账增长得快还是慢。

---

## 问题定义与边界

进程调度要解决的问题不是“谁先执行一次”，而是“在很多任务同时可运行时，CPU 时间如何持续分配”。现代系统里，浏览器、数据库、日志线程、网络线程、后台服务都可能同时竞争 CPU。调度器必须在三个目标之间找平衡：

| 目标 | 含义 | 为什么难 |
|---|---|---|
| 公平 | 多个普通任务长期拿到与权重匹配的 CPU 份额 | 任务数会动态变化 |
| 低开销 | 插入、删除、选下一个任务不能太慢 | 调度本身也消耗 CPU |
| 响应性 | 交互任务不能一直等 | 公平和低延迟常冲突 |

Linux 中至少要区分三类常见调度策略：

| 调度类 | 面向对象 | 是否进入 CFS 红黑树 | 主要特点 |
|---|---|---|---|
| `SCHED_OTHER` / `SCHED_NORMAL` | 普通进程 | 是 | 默认策略，讲公平 |
| `SCHED_FIFO` | 实时任务 | 否 | 先到先跑，不主动时间片轮转 |
| `SCHED_RR` | 实时任务 | 否 | 实时任务之间按时间片轮转 |
| `SCHED_DEADLINE` | 强时限任务 | 否 | 按运行时限和周期做保证 |

所以本文的边界很明确：重点解释普通任务的 CFS；实时类只说明它与 CFS 的边界，不展开成完整的实时调度专题。

还有一个容易混淆的边界是“任务”和“调度实体”。调度实体 `sched_entity` 可以理解为“被调度器计费和排队的对象”。对白话理解来说，可以先把它当作一个线程；在开启 cgroup 调度时，它也可能代表一组任务。也就是说，CFS 调度的不是抽象的“程序名”，而是具体的“可运行实体”。

玩具例子：三辆玩具火车都想上同一条铁轨。CFS 不会让一辆车一直跑到底，也不会简单按固定顺序死轮转，而是记录“谁最近跑得少”，每次把最欠跑的那辆先放上轨道。

真实工程例子：一台 8 核应用服务器同时运行 Nginx、Java 服务、日志采集、监控 agent。它们大多属于普通任务，默认走 CFS；如果某个核上再塞入音视频低延迟线程并要求严格抖动控制，就要考虑把这类线程切到实时策略，而不是继续只靠 CFS 的 `nice` 微调。

---

## 核心机制与推导

先看 CFS 想逼近的理想模型。假设有 $n$ 个任务同时可运行，理想 CPU 应该在总时长 $T$ 内按权重比例给出份额：

$$
t_i = \frac{w_i}{\sum w} \times T
$$

其中 $w_i$ 是任务 $i$ 的权重。白话说，权重越大，长期分到的 CPU 时间越多。

### 1. 为什么要引入 `vruntime`

如果只记录“实际运行了多久”，无法体现权重差异。一个高权重任务和一个低权重任务都跑了 10 ms，看似一样，但公平目标并不一样。于是 CFS 不直接比较真实运行时间，而比较折算后的虚拟运行时间。

公式可写成：

$$
vruntime_{new}=vruntime_{old}+\delta_{exec}\times \frac{NICE\_0\_LOAD}{weight}
$$

这意味着：

| 情况 | 权重 `weight` | `vruntime` 增长速度 | 长期 CPU 份额 |
|---|---|---|---|
| `nice` 更低 | 更大 | 更慢 | 更多 |
| `nice` 默认 | 基准 | 基准 | 基准 |
| `nice` 更高 | 更小 | 更快 | 更少 |

关键点不是“高优先级立刻抢占所有人”，而是“高权重任务在相同真实运行时间下，账本增长更慢，因此更常处于树的左边”。

### 2. 红黑树为什么合适

调度器需要反复做三件事：

1. 新任务变成可运行，要插入有序集合。
2. 当前任务阻塞或睡眠，要从集合中移除。
3. 每次选出 `vruntime` 最小的任务运行。

数组适合顺序遍历，不适合频繁有序插入；链表适合插删，不适合高效找最小值。红黑树能把插入、删除、查找最小值控制在通常 $O(\log N)$ 的量级，因此适合“任务很多而且状态频繁变化”的场景。

### 3. `min_vruntime` 的作用

`min_vruntime` 可以理解为“当前运行队列的参考起点”，白话说就是“这棵树里最靠前位置的大致基线”。它有两个作用：

1. 避免新加入任务因为历史值太小而获得不合理优势。
2. 让不同任务的 `vruntime` 比较保持稳定，不至于无限膨胀。

### 4. 玩具例子：两任务分 CPU

假设两个任务同时可运行：

- `p1`：`weight = 1024`，相当于默认 `nice=0`
- `p2`：`weight = 820`，相当于较低一级的权重

总权重是：

$$
1024 + 820 = 1844
$$

5 秒内理论 CPU 份额约为：

$$
t_1 = \frac{1024}{1844}\times 5 \approx 2.78s
$$

$$
t_2 = \frac{820}{1844}\times 5 \approx 2.22s
$$

也就是 `p1` 约占 55%，`p2` 约占 45%。注意，这不是说调度器会先让 `p1` 连跑 2.78 秒，再让 `p2` 连跑 2.22 秒；真实情况是它们会频繁切换，但长期累计结果接近这个比例。

如果 `p1` 跑了 10 ms，那么它的 `vruntime` 增量约为：

$$
10\times \frac{1024}{1024}=10
$$

而 `p2` 跑了同样的 10 ms，增量约为：

$$
10\times \frac{1024}{820}\approx 12.49
$$

因此 `p2` 的账本增长更快，更容易被放到树的右边；`p1` 账本增长慢，更容易再次获得调度。

### 5. 从机制到代码路径

CFS 运行时主要围绕几个动作：

| 动作 | 典型函数 | 作用 |
|---|---|---|
| 入队 | `enqueue_task_fair()` | 把实体放进 CFS 运行队列 |
| 更新当前任务 | `update_curr()` | 统计本次运行时间并增加 `vruntime` |
| 选下一个 | `pick_next_task_fair()` | 找红黑树最左节点 |
| 出队 | `dequeue_task_fair()` | 任务阻塞或离开运行队列时移除 |

所以从抽象上看，CFS 并不神秘：更新当前任务账本，把它放回有序结构，再拿出最欠跑的那个。

---

## 代码实现

Linux 内核中 CFS 的主体代码在 `kernel/sched/fair.c`。理解源码时，先盯住两个结构：

| 结构/字段 | 作用 | 白话解释 |
|---|---|---|
| `struct sched_entity` | 调度实体 | 被排队和计费的对象 |
| `se.vruntime` | 虚拟运行时间 | 折算后的“已跑账本” |
| `se.exec_start` | 本次开始执行时间 | 这轮从什么时候开始跑 |
| `se.sum_exec_runtime` | 累计真实运行时间 | 一共实际跑了多久 |
| `se.load.weight` | 权重 | 决定 CPU 份额比例 |
| `struct cfs_rq` | CFS 运行队列 | 一棵树和一些统计数据 |
| `cfs_rq.min_vruntime` | 当前最小基线 | 队列参考起点 |

可以把它简化成下面这段伪代码：

```c
struct sched_entity {
    u64 vruntime;
    u64 exec_start;
    u64 sum_exec_runtime;
    struct load_weight load;
};

struct cfs_rq {
    struct rb_root_cached tasks_timeline;
    u64 min_vruntime;
};

void update_curr(struct cfs_rq *cfs_rq, struct sched_entity *curr, u64 delta_exec) {
    curr->sum_exec_runtime += delta_exec;
    curr->vruntime += delta_exec * NICE_0_LOAD / curr->load.weight;
}

struct sched_entity *pick_next_task_fair(struct cfs_rq *cfs_rq) {
    if (rb_empty(&cfs_rq->tasks_timeline))
        return NULL;

    // 最左节点就是 vruntime 最小，也就是“最欠跑”的任务
    return rb_leftmost(&cfs_rq->tasks_timeline);
}
```

这里的 `rb_leftmost` 很关键。它不是“随便找一个节点”，而是“直接拿到当前树中最小键值对应的节点”。由于键是 `vruntime`，因此它就对应“最应该被补偿运行的任务”。

下面用一个可运行的 Python 玩具模型模拟 CFS 的核心思想。它不是内核实现，但能验证“权重越大，长期 CPU 份额越多”：

```python
from dataclasses import dataclass

NICE_0_LOAD = 1024

@dataclass
class Task:
    name: str
    weight: int
    vruntime: float = 0.0
    runtime: float = 0.0

def run_cfs(tasks, ticks=10000, delta_exec=1.0):
    for _ in range(ticks):
        current = min(tasks, key=lambda t: t.vruntime)
        current.runtime += delta_exec
        current.vruntime += delta_exec * (NICE_0_LOAD / current.weight)
    return tasks

tasks = [
    Task("p1", 1024),
    Task("p2", 820),
]

run_cfs(tasks, ticks=18440, delta_exec=1.0)

total = sum(t.runtime for t in tasks)
share_p1 = tasks[0].runtime / total
share_p2 = tasks[1].runtime / total

expected_p1 = 1024 / (1024 + 820)
expected_p2 = 820 / (1024 + 820)

assert abs(share_p1 - expected_p1) < 0.01
assert abs(share_p2 - expected_p2) < 0.01
assert tasks[0].runtime > tasks[1].runtime

print(round(share_p1, 4), round(share_p2, 4))
```

这段代码的逻辑和 CFS 抽象一致：

1. 每轮都选 `vruntime` 最小的任务。
2. 运行一个固定的 `delta_exec`。
3. 用 `1024 / weight` 修正 `vruntime` 增量。
4. 长期结果逼近权重比例。

真实工程例子：在 Web 服务机器上，如果一个 API 进程、一个日志压缩进程、一个离线统计线程都跑在普通调度类下，那么即使日志压缩线程很吃 CPU，只要它的权重没特别高，CFS 也会通过 `vruntime` 控制它不要长期独占整核。你会看到它可能短时间内很活跃，但累计份额会被拉回到公平区间。

---

## 工程权衡与常见坑

CFS 的优点是通用、公平、复杂度稳定，但它不是“低延迟万能方案”。工程上最常见的误判，是拿 `nice` 去解决本质上属于实时调度的问题。

### 1. `nice` 只能调比例，不能给硬保证

`nice` 调整的是权重，不是“下一毫秒一定轮到你”。如果机器上有很多 CPU 密集型任务，某个线程即使设成更低的 `nice`，也只是长期份额更高，不代表抖动一定足够小。

### 2. 实时任务会压制普通任务

`SCHED_FIFO` 和 `SCHED_RR` 的优先级高于普通 CFS 任务。如果实时线程配置不当，普通任务可能长期拿不到 CPU。Linux 提供了 `sched_rt_runtime_us` 和 `sched_rt_period_us` 控制 RT 带宽，工程上常用经验是保持：

$$
\frac{runtime}{period} \le 0.95
$$

含义是：给实时任务预留大部分 CPU，但不要把普通任务完全饿死。

### 3. CPU 亲和性和缓存干扰常被忽略

即使调度策略正确，如果高频中断、缓存抖动、跨核迁移很多，延迟仍会恶化。CPU 亲和性就是“把任务绑定到特定核上”，白话说就是“少换座位，缓存更热”。对低延迟任务，经常还会配合隔离核与减少 tick 干扰。

### 4. 常见坑总表

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 只调 `nice` 期待实时性 | 延迟仍抖动 | `nice` 只影响份额比例 | 改用 RT 或隔离核 |
| RT runtime 设过大 | 普通任务卡死 | RT 几乎占满 CPU | 控制 `runtime/period` |
| 任务频繁跨核迁移 | 延迟不稳定 | cache 失效、迁移开销 | 设 CPU 亲和性 |
| 中断和业务线程同核 | 峰值延迟高 | 中断抢占业务执行 | 用 `isolcpus`、IRQ 绑定 |
| 把吞吐优化当低延迟优化 | 平均值好看，尾延迟差 | 指标目标错位 | 分开看 p99 和平均吞吐 |

### 5. 真实工程例子：低延迟音视频或交易线程

这类场景往往会这样做：

1. 用 `isolcpus=` 把一个或多个 CPU 核从通用调度中隔离出来。
2. 用 `taskset` 把关键线程绑到隔离核。
3. 用 `chrt` 把关键线程设成 `SCHED_FIFO` 或 `SCHED_RR`。
4. 其他普通业务仍留在 CFS 管辖的核上。

例如：

```bash
taskset -c 3 chrt -f 80 ./latency_worker
```

这条命令的含义是：把 `latency_worker` 绑到 CPU 3，并设成 FIFO 实时策略、优先级 80。此时它已经不再依赖 CFS 的“公平”，而是转向“确定性优先”。

---

## 替代方案与适用边界

CFS 适合绝大多数通用服务，但如果目标从“公平共享”变成“确定性延迟”或“时限保证”，就要换策略。

| 方案 | 适用场景 | 优点 | 代价 |
|---|---|---|---|
| CFS | 通用服务器、桌面、多任务混跑 | 公平、稳定、默认可用 | 不给硬实时保证 |
| `SCHED_FIFO` | 极低延迟、顺序确定性强 | 抢占强、行为直观 | 配置不当会饿死别人 |
| `SCHED_RR` | 多个实时任务轮转 | 比 FIFO 更均衡 | 仍需谨慎控制带宽 |
| `SCHED_DEADLINE` | 明确周期/截止时间的任务 | 可表达运行预算和时限 | 配置复杂，调试成本高 |
| CFS + cgroup | 多租户、容器限额 | 易做资源隔离 | 仍是公平模型，不是硬实时 |

### 1. 什么时候继续用 CFS

如果你的目标是这些，优先保留 CFS：

- Web 服务、批处理、数据库后台线程
- 容器混部，需要按组做 CPU 配额
- 更关心总体吞吐和长期公平，而不是单次抖动

### 2. 什么时候考虑 RT

如果你的目标是这些，单靠 CFS 往往不够：

- 音视频采集与播放链路
- 工业控制线程
- 高频交易或低延迟数据通道
- 明确要求某线程必须在极短时间内响应

这时常见组合是“隔离核 + `taskset` + `chrt` + RT 带宽限制”。白话说就是：先把跑道单独留出来，再让关键任务以实时规则使用这条跑道。

### 3. 什么时候考虑 `SCHED_DEADLINE`

当任务天然具有“每周期需要多少运行时间、最晚何时完成”的约束时，`SCHED_DEADLINE` 更接近问题本身。它适合“预算 + 周期 + 截止时间”明确的场景，但配置和验证都比 CFS、FIFO、RR 更难。

### 4. 不要把所有问题都推给调度器

很多性能问题并不是调度器本身引起的，而是：

- 锁竞争
- 用户态忙等
- GC 暂停
- I/O 抖动
- NUMA 远程内存访问

如果应用层线程模型本身不稳定，再好的调度策略也只是止损，不会把坏设计变成低延迟系统。

---

## 参考资料

- Linux Kernel Documentation, CFS Scheduler Design  
  说明 CFS 的设计目标、`vruntime`、红黑树与 `min_vruntime`，是理解原理的主文档。
- Linux Kernel Documentation, Scheduler documentation index  
  适合建立整体地图，先看 CFS，再看调度类与策略差异。
- Linux Kernel Documentation, Scheduler Nice Design  
  重点解释 `nice` 到权重的设计思想，适合理解“为什么 nice 不是线性优先级”。
- Linux Kernel source, `kernel/sched/fair.c`  
  对应 `enqueue_task_fair()`、`update_curr()`、`pick_next_task_fair()` 等实现细节。
- Ubuntu Real-time documentation, kernel boot parameters  
  适合查 `isolcpus`、实时内核参数、隔离核相关配置。
- Linux `chrt(1)` 与 `taskset(1)` 手册  
  工程上最直接的命令入口，用于设置调度策略和 CPU 亲和性。
- 补充读物：高级 CFS/调度分析文章  
  可用于辅助理解权重份额、cgroup 调度和实际例子，但应以内核官方文档为准。
