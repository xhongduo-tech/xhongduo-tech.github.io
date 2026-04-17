## 核心结论

Linux 性能分析工具不是“哪个更强”的问题，而是“当前瓶颈在哪一层”的问题。对零基础读者，最稳妥的路径是按 USE 方法排查：Utilization 是资源用了多少，Saturation 是资源是否排队拥塞，Errors 是是否出现错误或丢包。先看 CPU、内存、磁盘、网络、锁这几类资源，再决定用哪种工具。

一个实用判断链条如下：

| 观察层级 | 先看什么 | 典型工具 | USE 维度 | 适合回答的问题 |
| --- | --- | --- | --- | --- |
| 资源层 | CPU、内存、IO、网络是否异常 | `top` `vmstat` `pidstat` `iostat` `ss` | U/S/E | 哪类资源先出问题 |
| 进程层 | 哪个进程最重 | `pidstat` `ps` | U/S | 是谁在吃资源 |
| 系统调用层 | 进程在向内核请求什么 | `strace` | S/E | 卡在读写、网络、锁还是文件 |
| 函数热点层 | 哪些函数最耗 CPU | `perf` | U | 热点函数在哪里 |
| 内核/用户联动层 | 高频事件具体经过哪条路径 | eBPF / `bpftrace` | U/S/E | 低开销跟踪复杂路径 |

玩具例子：网页突然变慢，先开 `top`，看到某个核心上 CPU 用户态占用 `us` 很高；再用 `pidstat -u 1` 发现是 `python` 进程；接着用 `perf record -F 99 -p PID -g -- sleep 15` 采样；最后在 `perf report` 或火焰图里看到某个 JSON 解析函数占比异常高。这个流程比一上来就抓全量日志更稳，因为它先定位资源，再深入机制。

真实工程里，关键不是工具多，而是避免过度观测。`strace` 全量拦截、`perf` 过高频采样、eBPF map 设计不当，都会把“分析工具”变成新的性能问题。

---

## 问题定义与边界

本文讨论的是 Linux 主机或容器上的性能瓶颈定位，不讨论大规模 APM 平台建设，也不讨论分布式链路追踪系统。目标很明确：在可接受开销内，判断慢在哪里，并把范围收敛到资源、进程、系统调用或函数。

这里先定义几个边界：

| 维度 | 本文关注 | 不重点展开 |
| --- | --- | --- |
| 对象 | 单机 Linux、单进程、单服务、容器内进程 | 跨集群容量规划 |
| 目标 | 找瓶颈、给出下一步动作 | 建长期监控平台 |
| 方法 | 采样、追踪、事件统计 | 全量日志回放 |
| 开销要求 | 生产环境尽量低开销 | 离线实验室可重型观测 |

USE 方法为什么适合新手，是因为它先回答“有没有异常”，再回答“异常发生在哪里”。例如 CPU 利用率可以粗略写成：

$$
U_{\text{cpu}} = 1 - \frac{\text{idle}}{\text{total}}
$$

其中 `idle` 是 CPU 空闲时间，`total` 是所有 CPU 时间字段之和，常来自 `/proc/stat`。白话解释：CPU 总时间像一个总账本，空闲占比越低，说明忙碌占比越高。

如果 `vmstat 2 6` 连续输出都接近 `id=100`、`wa=0`，说明 CPU 和 IO 基本没压力。这时继续上 `perf`、`strace` 往往是浪费，因为系统还没有表现出资源异常。

下面这个最小例子可以帮助理解：

| `vmstat` 字段 | 值 | 解释 | 是否值得继续深挖 |
| --- | --- | --- | --- |
| `us` | 0 | 用户态 CPU 基本没跑业务代码 | 否 |
| `sy` | 0 | 内核态 CPU 基本没忙 | 否 |
| `id` | 100 | CPU 全空闲 | 否 |
| `wa` | 0 | 没有明显 IO 等待 | 否 |

所以，性能分析的第一原则不是“尽量多抓”，而是“先证明值得抓”。

---

## 核心机制与推导

`top`、`pidstat`、`vmstat` 这类工具本质上是在读内核已经维护好的统计量。它们适合回答宏观问题：哪个资源忙、哪个进程重、是否有排队。

`strace` 的机制不同。它通过系统调用跟踪，把进程进入内核的调用截获出来。系统调用可以理解成“用户态程序向内核申请服务的入口”，例如 `read`、`write`、`open`、`connect`。如果一个进程大量卡在 `futex`，通常意味着锁竞争；如果大量卡在 `read`，可能是磁盘或网络等待；如果 `connect` 很慢，可能是下游服务或网络问题。

`perf` 是采样型工具。采样的意思不是记录每一次事件，而是按周期抽样。设采样周期为 $P$，采样频率为 $F$，则：

$$
F = \frac{1}{P}
$$

如果 `perf record -F 100`，表示目标频率约为每秒 100 次，也就是每 10 ms 采一次。白话解释：不是把整部电影逐帧录下来，而是每隔固定时间拍一张照片，再通过足够多的照片估计最常出现的函数热点。

这也是火焰图成立的原因。火焰图不是“时间线”，而是“样本堆叠图”：某个函数在样本里出现得越多，图上就越宽。宽，不代表单次调用慢；宽代表“累计占比高”。

`perf` 支持多种事件来源：

| 事件类型 | 含义 | 适用场景 | 例子 |
| --- | --- | --- | --- |
| 硬件事件 | CPU 硬件计数器 | 分析 CPU 周期、缓存缺失 | `cycles` `cache-misses` |
| 软件事件 | 内核提供的软件统计 | 通用 CPU 分析 | `cpu-clock` `task-clock` |
| tracepoint | 内核预埋稳定事件点 | 系统行为分析 | 调度、syscall、页错误 |
| kprobe | 动态挂到内核函数 | 深入特定内核路径 | 文件系统、网络栈 |
| uprobe | 动态挂到用户态函数 | 分析应用内部函数 | 业务函数、库函数 |

eBPF 可以看作更灵活的内核内观测框架。它允许在 tracepoint、kprobe、uprobe 等位置挂程序，把统计逻辑放在内核里先聚合，再把结果批量送回用户态。白话解释：不是每发生一次事件就立即把消息发给分析程序，而是先在内核里记账，最后再取汇总结果，因此上下文切换更少，开销通常更低。

玩具例子：假设你只想知道系统里谁最常调用 `open`，用 `strace` 跟踪每一条 `open` 当然能做到，但会输出大量明细。eBPF 的做法是直接在内核里按进程名计数，只返回“谁调用了多少次”的聚合结果。

真实工程例子：线上一个日志服务 CPU 打满。先用 `pidstat -u -p PID 1` 确认单进程持续高 CPU，再用 `perf record -F 99 -p PID -g -- sleep 30` 生成热点，发现大量样本堆在字符串格式化函数上。随后再用 eBPF 挂到 `sys_enter_write` 和 `sys_exit_write`，统计每次写日志的字节数和耗时，最终确认根因不是磁盘慢，而是应用在用户态做了过多字符串拼接。

---

## 代码实现

下面先给一个可以运行的 Python 小脚本，用来理解 CPU 利用率公式。它不依赖 Linux 环境，可以直接运行。

```python
def cpu_utilization(prev_idle, prev_total, curr_idle, curr_total):
    idle_delta = curr_idle - prev_idle
    total_delta = curr_total - prev_total
    assert total_delta > 0
    return 1.0 - idle_delta / total_delta

# 玩具例子：两次采样之间，总时间增加 100，空闲时间增加 20
u = cpu_utilization(prev_idle=1000, prev_total=5000, curr_idle=1020, curr_total=5100)

assert abs(u - 0.8) < 1e-9
print(f"CPU utilization: {u:.2%}")
```

上面结果是 `80%`，意思是这段时间内 CPU 有 80% 在忙。

再看命令链条。初级工程师最常用的不是“单个神奇命令”，而是一组逐步收敛的命令。

```bash
# 1) 先看系统整体资源
top
vmstat 2 6
pidstat -u -r -d 1

# 2) 如果怀疑某个进程，盯住它
pidstat -p 12345 -u -r -d 1

# 3) 如果怀疑系统调用层面卡住，限定 syscall 类型跟踪
strace -p 12345 -T -tt -e trace=read,write,futex

# 4) 如果怀疑 CPU 热点，做调用栈采样
sudo perf record -F 99 -p 12345 -g -- sleep 15
sudo perf report

# 5) 如果要做系统级采样
sudo perf record -F 99 -a -g -- sleep 15

# 6) 如果希望用 eBPF 低开销统计 open 次数
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }'
```

其中几个参数要看懂：

| 命令 | 关键参数 | 含义 | 输出价值 |
| --- | --- | --- | --- |
| `pidstat -u -r -d 1` | `-u -r -d` | 看 CPU、内存、磁盘 | 快速定位哪个进程异常 |
| `strace -p PID -T -tt` | `-p` `-T` `-tt` | 附着进程，显示耗时和时间戳 | 看 syscall 是否慢 |
| `strace -e trace=...` | `-e trace=` | 只跟踪指定 syscall | 降低噪声和开销 |
| `perf record -F 99 -g` | `-F` `-g` | 设采样频率并记录调用栈 | 生成热点画像 |
| `perf record -a` | `-a` | 全机采样 | 适合系统级热点 |
| `bpftrace -e '...'` | 内联脚本 | 在内核挂点并聚合 | 高频、低开销统计 |

如果要生成火焰图，典型流程是先有 `perf.data`，再导出折叠栈，最后渲染 SVG。不同发行版命令略有差异，但核心步骤不变：采样、折叠、渲染。

真实工程例子可以按下面的诊断顺序执行：

1. 页面超时，先看 `top`，发现 CPU 高。
2. 用 `pidstat -p PID 1` 确认是业务进程，而不是系统进程。
3. 用 `strace -p PID -e trace=futex -T` 判断是否锁竞争。
4. 如果不是锁而是纯 CPU，改用 `perf record -p PID -g -- sleep 20`。
5. 若要确认某个内核路径的高频行为，再补一段 eBPF 统计。

这个顺序的价值在于，每一步都比上一步更细，但也更贵。

---

## 工程权衡与常见坑

性能分析工具本身也会消耗资源。工程上的核心权衡是：信息密度越高，通常开销越高；覆盖面越全，通常噪声越多。

`strace` 最典型的问题是“看得太细”。它能告诉你每个 syscall 的参数、返回值、耗时，但代价是每次 syscall 都可能触发额外处理。对高 QPS 服务，如果直接全量附着，可能明显拖慢业务。新手常犯的错误是把 `strace` 当成默认工具，其实它更适合短时、定向、限定 syscall 的检查。

`perf` 的常见坑是采样频率过高。频率不是越高越好，因为过高会引入抖动，尤其在小延迟服务中。99Hz、199Hz 这类频率常被优先采用，是因为它们通常已经足够看出热点趋势。

eBPF 的常见坑则是“脚本写得像实时日志系统”。如果不停 `print`，或者 map 键过多、生命周期不受控，内核态聚合带来的好处会被抵消。

下面是常见坑表：

| 常见坑 | 表现 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| `strace` 全量跟踪生产进程 | 业务明显变慢 | syscall 级别拦截过重 | 只跟踪目标 PID、目标 syscall、短时间执行 |
| `perf -F` 设太高 | 服务出现 jitter | 采样中断太频繁 | 从 `99` 或 `199` 起步，必要时再升 |
| `perf` 不带 `-g` | 只看到叶子函数 | 没有调用栈 | 分析 CPU 热点时优先保留栈 |
| eBPF map 无限增长 | 内存增加或结果失真 | key 设计过细 | 只按需要聚合，控制 key 基数 |
| eBPF 大量打印 | CPU 飙高 | 用户态读取日志过多 | 少打印，多聚合 |
| 先抓明细后看资源 | 数据太多但没有结论 | 跳过 USE 步骤 | 先看资源异常，再定向深入 |

一个简单经验是：如果你还不能回答“我为什么要抓这类数据”，就先不要抓。

---

## 替代方案与适用边界

并不是所有问题都要上 `perf` 或 eBPF。工具应该和问题规模匹配。

| 方法 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| `top` / `vmstat` / `pidstat` | 简单、几乎随处可用 | 只能看到宏观现象 | 初筛、值班排障 |
| `strace` | syscall 细节清楚 | 开销较高、明细过多 | 单进程、短时、怀疑 IO/锁 |
| `perf stat` | 宏观硬件/软件计数好用 | 不给详细调用栈 | 先看全局趋势 |
| `perf record` | CPU 热点定位强 | 需要一定阅读门槛 | CPU 打满、找热点函数 |
| eBPF / `bpftrace` | 灵活、低开销、可聚合 | 学习门槛高、内核版本相关 | 生产环境、高频事件、复杂路径 |

如果你只是调一个单进程脚本，`strace` 和 `perf` 往往已经够用。如果你面对的是高并发线上服务，优先级通常会变成：

1. `pidstat` / `vmstat` 定位资源。
2. `perf stat` 看整体计数。
3. `perf record` 做定向采样。
4. 只有在需要内核与用户态联动细节时，再上 eBPF。

还有一个边界要明确：并非所有“慢”都是 CPU 问题。比如请求响应时间高，但 CPU 空闲、磁盘不忙、网络无重传，这时可能是锁、下游依赖或应用层队列问题。工具给的是证据，不是自动答案。

---

## 参考资料

| 来源 | 摘要 | 用途 |
| --- | --- | --- |
| Brendan Gregg, USE Method for Linux Performance Analysis | 解释 USE 方法如何从资源利用、饱和、错误三维排查 | 建立整体分析框架 |
| Brendan Gregg, Linux Performance Tools | 总览常见 Linux 性能工具及定位顺序 | 工具地图 |
| Linux `perf` 官方文档 | 说明 `perf stat`、`perf record`、事件类型和调用栈采样 | 理解 `perf` 工作机制 |
| `strace` 官方文档 | 说明 syscall 跟踪语法、过滤方式和限制 | 精准使用 `strace` |
| bpftrace 文档 | 说明 tracepoint、kprobe、uprobe 及聚合语法 | 入门 eBPF 观测 |

- USE Method for Linux Performance Analysis: https://www.brendangregg.com/USEmethod/use-linux.html
- Linux perf examples and basics: https://perf.wiki.kernel.org/
- strace official site and docs: https://strace.io/
- bpftrace documentation: https://bpftrace.org/docs/
- Linux `vmstat` and `/proc/stat` 相关说明可参考手册页与系统文档
