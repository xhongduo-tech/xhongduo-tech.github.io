## 核心结论

Linux 性能分析不要从单一工具开始，而要按层级推进。对初学者最稳妥的路径是：先用 `top`、`htop`、`vmstat 1`、`iotop` 看系统全貌，再决定是 CPU、系统调用还是块设备 I/O 出问题；确认方向后，再用 `perf stat` 做计数，用 `perf record -g` 做采样和调用栈分析，最后在 I/O 场景下用 `bpftrace` 看延迟分布。

这个工具箱的核心价值，是把“资源是否紧张”和“代码到底卡在哪一层”分开看。`perf` 适合回答“CPU 周期花在哪”；`strace` 适合回答“是不是 syscall 太多或太慢”；`bpftrace` 适合回答“块设备 I/O 延迟到底有多高”；`top/vmstat/iotop` 负责给出系统级信号，避免一上来就盲目抓火焰图。

下面这张表先建立整体地图：

| 工具 | 观察层级 | 最适合回答的问题 | 常见命令 |
|---|---|---|---|
| `top` / `htop` | 系统与进程 | 哪个进程吃 CPU、内存是否紧张、load 是否异常 | `top`、`htop` |
| `vmstat` | 系统整体 | CPU 在忙什么、是否在等 I/O、块读写是否异常 | `vmstat 1` |
| `iotop` | 进程级 I/O | 哪个进程在持续读写磁盘、谁在等 I/O | `iotop -oPa` |
| `perf stat` | 进程 / 全机计数 | 周期、指令、缓存缺失是否异常，IPC 高不高 | `perf stat -p <pid>` |
| `perf record/report` | 函数 / 调用链 | 热点函数在哪，调用链如何进入热点 | `perf record -g ...` |
| `strace` | 系统调用 | `read`、`write`、`futex`、`epoll_wait` 谁耗时多 | `strace -c -T -p <pid>` |
| `bpftrace` | 内核事件 / 块层 | I/O 延迟分布怎样，慢请求来自哪类设备操作 | `bpftrace biolatency.bt` |

玩具例子可以非常简单：先用 `top` 看到某进程 CPU 约 90%，再用 `perf stat -p <pid>` 看 `cycles` 和 `instructions`，如果 IPC 很低，通常说明流水线利用不理想或大量停顿；接着 `perf record -p <pid> -g -- sleep 10`，最后 `perf report` 或火焰图定位热函数。

---

## 问题定义与边界

这里讨论的“性能分析”，目标不是写出完整监控系统，而是在 Linux 主机上快速判断瓶颈属于哪一类，并把问题缩小到可以行动的范围。常见类别有四种：

| 类别 | 直观表现 | 第一批工具 |
|---|---|---|
| CPU 饱和 | 进程 CPU 高，响应慢，负载上升 | `top`、`vmstat`、`perf` |
| 系统调用开销高 | 用户态代码不忙，但内核态时间多、频繁切换 | `strace`、`perf` |
| 磁盘 I/O 慢 | `wa` 高、吞吐不高但延迟大 | `vmstat`、`iotop`、`bpftrace` |
| 内存 / 交换问题 | `si/so` 出现、缓存抖动、I/O 被拖慢 | `vmstat`、`top` |

边界也要说清楚。

第一，`top`、`vmstat`、`iotop` 是“观察信号”，不是根因分析器。它们能告诉你系统现在像不像 CPU 压力、I/O 压力或内存压力，但不能直接指出哪一行代码热。

第二，`perf stat` 和 `perf record` 主要处理 CPU 相关问题。前者是“计数”，后者是“采样”。计数像体检报告，采样像热点地图。

第三，`strace` 只看系统调用。它能说明进程频繁进内核，或者某类 syscall 明显慢，但如果真正热点在纯用户态计算，`strace` 不会告诉你。

第四，`bpftrace` 适合在内核事件上做无侵入观测。无侵入的意思是不用改业务代码就能追踪内核事件。它非常适合块 I/O 延迟、文件打开、网络内核路径等问题，但不等于完整应用级 tracing。

对新手，一个非常实用的定界流程是：

1. 服务响应慢，先问“是不是 CPU 资源耗尽”。
2. 用 `top`/`htop`/`vmstat 1` 看 `us`、`sy`、`wa`。
3. 如果 `us` 很高，优先走 `perf`。
4. 如果 `sy` 很高，优先补 `strace` 或 `perf` 的内核栈。
5. 如果 CPU 不高但 `wa`、`bi`、`bo` 上升，转向 `iotop` 和 `bpftrace`。

把计数写成公式，会更容易理解：

$$
event\_freq = \frac{event\_count}{elapsed\_time}
$$

它表示单位时间内发生了多少次事件。比如 `cycles` 每秒多少，`cache-misses` 每秒多少。这个值本身不是结论，但它能帮助你把“看起来很多”变成“每秒到底多少”。

---

## 核心机制与推导

先看 `perf stat`。它依赖硬件性能计数器，硬件性能计数器可以理解为 CPU 内部的计数器，用来记录周期、指令、缓存缺失、分支失误等事件总数。`perf stat` 结束时给你的是总量，因此特别适合做“这段程序总体运行特征是什么”的判断。

最常见的指标是：

$$
IPC = \frac{instructions}{cycles}
$$

IPC 是每个周期退休多少条指令。白话说，CPU 每跳一次时钟，平均真正干了多少活。IPC 高，通常说明算得顺；IPC 很低，常见原因是缓存未命中、分支预测失败、锁竞争、流水线停顿。

再看 `perf record -g`。它不是数总量，而是按频率采样调用栈。采样可以理解为“每隔一小段时间拍一张栈快照”。如果某个函数在很多快照里都出现，它就会在 `perf report` 或火焰图中占更宽的宽度。火焰图的横向宽度表示累计样本占比，不表示时间线先后；纵向表示调用关系，不表示谁更慢。新手最容易误读的是把火焰图当作时间轴，这是错误的。

`strace` 的机制完全不同。它拦截系统调用入口和出口，统计每个 syscall 的次数和耗时。系统调用是用户态进程向内核请求服务的边界，比如 `read`、`write`、`openat`、`futex`。`strace -T` 给每次 syscall 的耗时，`strace -c` 给聚合统计。`-O` 用来扣除追踪本身的测量开销，否则你看到的 syscall 总时间会混入 `strace` 自己的成本。

`bpftrace` 则基于 eBPF。eBPF 可以理解为“在内核里安全运行的小程序”。`bpftrace` 允许你把脚本挂到 tracepoint 或 kprobe 上。tracepoint 是内核预留的稳定观测点，kprobe 是对内核函数做动态探针。做 I/O 延迟时，常见思路是在请求开始时记录时间戳，在结束时减掉开始时间，再用直方图展示延迟分布。

系统工具的字段也必须能读懂。下面这张表是入门必须记住的：

| 字段 | 来自 | 含义 | 常见解读 |
|---|---|---|---|
| `us` | `top`/`vmstat` | 用户态 CPU 百分比 | 业务代码在忙 |
| `sy` | `top`/`vmstat` | 内核态 CPU 百分比 | syscall、内核路径、驱动在忙 |
| `id` | `top`/`vmstat` | 空闲 CPU 百分比 | 机器还有余量 |
| `wa` | `top`/`vmstat` | 等待 I/O 的 CPU 百分比 | 磁盘或块层可能卡住 |
| `bi` | `vmstat` | 从块设备读入速率 | 读压力信号 |
| `bo` | `vmstat` | 发往块设备写出速率 | 写压力信号 |
| `r` | `vmstat` | 就绪队列中的可运行任务 | 长期偏高说明 CPU 竞争 |
| `b` | `vmstat` | 不可中断睡眠任务数 | 常与 I/O 等待相关 |

玩具例子：一个 Python 脚本不断做字符串拼接，`top` 看到 CPU 很高，`perf stat` 可能显示 `instructions` 很多、`cache-misses` 不明显，这更像纯计算热点。相反，数据库写入变慢时，`top` 里的 CPU 可能并不高，但 `vmstat 1` 里的 `wa` 上升，`iotop` 能看到写线程活跃，`bpftrace` 的 `biolatency` 直方图会出现高延迟尾部，这就是典型 I/O 路径问题。

---

## 代码实现

先给一个最小可运行的数值玩具例子，用来理解 `IPC` 和事件频率，而不是替代真实测量：

```python
def calc_ipc(instructions: int, cycles: int) -> float:
    assert cycles > 0
    return instructions / cycles

def event_freq(event_count: int, elapsed_time: float) -> float:
    assert elapsed_time > 0
    return event_count / elapsed_time

ipc = calc_ipc(1_403_561_257, 2_066_201_729)
freq = event_freq(2_066_201_729, 0.956217)

assert round(ipc, 3) == 0.679
assert int(freq) == 2160755946

print(f"IPC={ipc:.3f}")
print(f"cycles/s={freq:.0f}")
```

真实排查时，命令序列通常是下面这样。

先看系统整体状态：

```bash
top
vmstat 1
iotop -oPa
```

如果怀疑是某个进程吃满 CPU：

```bash
top -Hp <pid>
perf stat -p <pid> sleep 10
perf stat -d -p <pid> sleep 10
```

`-d` 会补充缓存相关指标。重点先看 `instructions`、`cycles`、`cache-misses`，再结合 `IPC = instructions / cycles` 判断是否存在明显停顿。

如果要看热点函数和调用链：

```bash
perf record -p <pid> -g -- sleep 10
perf report
```

如果要在全机范围采样：

```bash
perf record -a -F 1000 -g -- sleep 5
perf report
perf script report flamegraph
```

这里 `-a` 是全 CPU 采样，`-F` 是目标采样频率，`-g` 是采集调用栈。`perf record` 的关键选项可以这样记：

| 参数 | 作用 | 常见场景 | 结果 |
|---|---|---|---|
| `-a` | 全机采样 | 不确定哪个进程有问题 | 覆盖所有 CPU |
| `-p <pid>` | 指定进程 | 已锁定目标服务 | 聚焦单进程 |
| `-F 1000` | 目标采样频率 | 短时抓热点 | 平衡粒度与开销 |
| `-g` | 记录调用栈 | 需要火焰图或调用链 | 生成可展开栈信息 |
| `-o file` | 输出文件 | 多次实验对比 | 自定义 `perf.data` |

如果怀疑 syscall 开销高：

```bash
strace -T -p <pid>
strace -c -p <pid>
strace -c -O 5us -p <pid>
```

`-T` 适合看单次调用耗时，`-c` 适合看汇总，`-O` 用来补偿追踪开销。实际值通常要结合你的机器和负载试。

如果怀疑块 I/O 延迟：

```bash
sudo bpftrace biolatency.bt
sudo bpftrace biosnoop.bt
```

如果系统里没有现成脚本，也可以直接写一个最小脚本，按块 I/O 请求开始和结束计算延迟直方图：

```bash
sudo bpftrace -e '
tracepoint:block:block_rq_issue { @start[args->sector] = nsecs; }
tracepoint:block:block_rq_complete /@start[args->sector]/ {
  @lat = hist((nsecs - @start[args->sector]) / 1000);
  delete(@start[args->sector]);
}'
```

这个直方图单位是微秒。白话说，它告诉你“慢 I/O 是偶发几个，还是整个分布都偏慢”。

真实工程例子：假设一个 `redict-server` 或数据库实例响应抖动。先用 `vmstat 1` 发现 `wa` 持续升高，再用 `iotop -oPa` 看到写线程占据主要 I/O；这时不是直接去改业务代码，而是先运行：

```bash
perf record -g --pid "$(pgrep redict-server)" -F 999 -- sleep 60
perf report -g graph,0.5,caller
sudo bpftrace biolatency.bt
```

如果 `perf report` 里 CPU 热点并不高，但 `biolatency` 出现明显长尾，就说明瓶颈更像块设备或文件系统路径，而不是纯计算热点。

补充一个经常被忽略的命令，用于观察空闲周期：

```bash
perf stat -e idle-cycles -a sleep 1
```

火焰图主要展示 active 栈，也就是“正在执行时采到的栈”；如果系统大量时间根本没在执行用户代码，火焰图不会替你解释空闲去哪了，这时需要 `perf stat` 这类计数补上视角。

---

## 工程权衡与常见坑

性能工具不是“开了就一定对”。每个工具都有误差来源和适用边界。

第一类坑是把观测工具当成零开销。`strace` 尤其明显。它通过拦截 syscall 工作，天然会改变被测程序的执行节奏，所以 `strace -c` 的总耗时不等于“完全原始”的 syscall 成本。`-O` 的作用就是尽量扣掉这部分测量开销，但它依然不是完全零误差。

第二类坑是把火焰图当成完整真相。火焰图非常适合看“忙的时候谁最热”，但对“为什么没在忙”并不完整。比如线程都在 I/O 等待、被锁阻塞、或 CPU 空闲很多时，单看火焰图会遗漏整体资源状态。

第三类坑是采样频率设得过高。`perf record -F 9999` 在生产上很可能增加额外负担，尤其在全机 `-a` 模式下。频率太低又会丢细粒度热点。经验上先从 `99`、`199`、`999` 这种量级开始，再视情况调整。

第四类坑是 `iotop` 输出不完整或不准。它依赖内核配置，例如 `CONFIG_TASK_IO_ACCOUNTING`、`CONFIG_TASK_DELAY_ACCT`、`CONFIG_TASKSTATS` 和相关计数能力；较新内核还常涉及 `kernel.task_delayacct` 是否开启。没有这些条件时，`IO%`、`SWAPIN` 或部分统计会缺失。

下面用表格总结：

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| `strace -c` 时间偏大 | syscall 总时间看起来异常高 | 追踪本身有开销 | 用 `-O` 做补偿，并与不追踪时的总耗时对比 |
| 火焰图看不到“等待” | 图上热点不明显，但服务仍然慢 | 火焰图只展示采样到的 active 栈 | 配合 `vmstat`、`iotop`、`perf stat -e idle-cycles` |
| `perf` 影响业务 | 采样时延抖动增大 | 采样频率太高或全机采样太广 | 降低 `-F`，缩短采样窗口，优先针对单进程 |
| `iotop` 字段缺失 | `IO%` 或 `SWAPIN` 不可用 | 内核 accounting 未启用 | 检查内核配置和 `kernel.task_delayacct` |
| `strace` 输出太多 | 很难读出重点 | 未做聚合或过滤 | 先 `-c`，再按 syscall 类别或 PID 过滤 |
| 只看单次快照 | 问题复现不了 | 瞬时数据无法反映趋势 | `vmstat 1` 连续看趋势，必要时多轮采样 |

---

## 替代方案与适用边界

不是每次都要上 `perf + strace + eBPF` 全套。工具应该和问题规模匹配。

如果你只是在值班排障，先确认机器是不是整体资源紧张，`top` 和 `vmstat 1` 往往就够了。它们轻量、几乎所有 Linux 都自带，而且能快速回答“CPU 忙不忙”“是不是在等 I/O”“有没有交换压力”。

如果已经确认某进程 CPU 异常，最优先的是 `perf`，因为它能直接把热点映射到函数与调用链。相比之下，`strace` 更适合“内核边界很多”的问题，例如 `futex`、`epoll_wait`、`read/write` 频繁或变慢。

如果问题明显落在块设备延迟，不要只盯着吞吐。吞吐正常不代表延迟没问题。`bpftrace` 的价值在于直接看延迟分布和长尾，这一点比单纯 `iotop` 更强。

也有替代方案：

| 工具 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| `perf top` | 实时看热点 | 信息不如离线报告完整 | 临时在线观察 CPU 热点 |
| `ftrace` | 内核路径能力强 | 使用门槛较高 | 深入内核函数与调度路径 |
| `systemtap` | 功能强大 | 部署和脚本复杂 | 老环境或特定诊断场景 |
| `bcc` 工具集 | 封装了很多 eBPF 脚本 | 依赖更多 | 要快速追 opensnoop、execsnoop、tcplife 等 |
| `top/vmstat` | 轻量、总览快 | 没法定位到函数 | 初筛、值班、容量判断 |

新手实践时可以记住一句话：`top/vmstat` 决定“方向”，`perf/strace/bpftrace` 决定“根因”。线上只需看 `top` 时，不要硬上复杂 tracing；但当你已经看到 `wa` 或 `bi/bo` 异常，再进入 `bpftrace biolatency.bt` 才是合理升级。

---

## 参考资料

| 类别 | 资料 | 说明 |
|---|---|---|
| perf | [perf wiki tutorial](https://perfwiki.github.io/main/tutorial/) | `perf stat`、`perf record`、`perf report` 与火焰图入门 |
| strace | [strace man page on man7](https://man7.org/linux/man-pages/man1/strace.1.html) | `-T`、`-c`、`-O` 等选项说明 |
| strace | [Strace little book: Set the overhead](https://nanxiao.gitbooks.io/strace-little-book/content/posts/set-the-overhead-for-tracing-system-calls.html) | 解释 `-O` 为什么能修正统计偏差 |
| bpftrace | [bpftrace docs](https://bpftrace.org/docs/0.21) | 语言、探针模型、命令行参数 |
| bpftrace | [bpftrace examples](https://bpftrace.github.io/) | 直方图脚本与 tracepoint/kprobe 示例 |
| vmstat | [procps-ng vmstat man page](https://www.mankier.com/8/vmstat) | `us/sy/id/wa`、`bi/bo` 等字段定义 |
| iotop | [iotop man page](https://www.man.he.net/man8/iotop) | 内核配置依赖与输出字段说明 |
| 工程案例 | [redict CPU profiling guide](https://redict.io/docs/usage/optimization/cpu-profiling/) | 真实服务上用 `perf record -g` 的示例 |
