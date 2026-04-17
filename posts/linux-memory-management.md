## 核心结论

Linux 内存管理的核心不是“程序直接拿到物理内存”，而是“程序先拿到一段虚拟地址空间”。虚拟地址空间可以理解为进程看到的“私有内存地图”：代码段放指令，堆用于动态分配，栈保存函数调用现场，`mmap` 区承载共享库、文件映射和大块匿名映射。CPU 真正访问某个地址时，如果对应页还没有准备好，就会触发 **Page Fault**，也就是“访问到了一个当前还不能直接使用的页”，然后由内核决定是补页、换入、扩展栈，还是直接报错。

当物理内存和可回收页都不够时，Linux 不会无限等待，而会进入 **OOM Killer** 逻辑。OOM 是 Out Of Memory，意思是“系统已经没法满足新的内存需求”。OOM Killer 会根据进程占用和优先级计算分数，优先杀掉“代价最小、释放最多”的进程。对工程系统来说，真正的第一道防线不是等 OOM 来救场，而是提前把 `vm.swappiness`、`vm.overcommit_memory`、`oom_score_adj`、swap 容量和 cgroup 限额配好。

一个最小直观例子：机器有 10 GB RAM，一个新程序突然还要再申请 3 GB。内核会先尝试回收页缓存、写回脏页、换出冷页；如果这些动作之后仍然无法满足分配，才会进入 OOM 选择流程，杀掉分数最高的进程，而不是“谁最后申请内存就杀谁”。

| 参数 | 它控制什么 | 典型影响 | 常见建议 |
|---|---|---|---|
| `vm.swappiness` | 匿名页和页缓存谁更早被换出 | 值高时更早用 swap，值低时更偏向保留匿名页 | 通用服务 `10~20` |
| `vm.overcommit_memory` | 内核是否允许“先答应、以后再说”的内存承诺 | `0` 启发式，`1` 更激进，`2` 严格限制 | 关键任务机器常用 `2` |
| `oom_score_adj` | 对单进程 OOM 分数做人工偏移 | 值越大越容易被杀，越小越难被杀 | 核心服务调低，批任务调高 |

---

## 问题定义与边界

这篇文章讨论的是 **Linux 内核级内存管理**，也就是页表、缺页异常、swap、OOM Killer 和 `/proc/sys/vm/*` 这类机制。它不讨论 Java/Go 的垃圾回收，也不展开 NUMA、HugeTLB、GPU 显存分页、用户态内存池等更细分主题。

先把地址空间布局说清楚。进程的虚拟地址空间通常包含以下区域：

| 区域 | 作用 | 增长方向 | 常见来源 |
|---|---|---|---|
| 代码段 | 存放可执行指令 | 基本固定 | ELF 可执行文件 |
| 数据段/静态段 | 全局变量、静态变量 | 基本固定 | ELF 可执行文件 |
| 堆 | 动态分配内存 | 向高地址增长 | `malloc/new` |
| `mmap` 区 | 文件映射、共享库、匿名映射 | 常由内核管理 | `mmap`、共享库加载 |
| 栈 | 函数调用帧、局部变量 | 向低地址增长 | 线程运行时 |

这里的 **VMA**，全称 Virtual Memory Area，可以理解为“同一段内存属性一致的区间描述”，例如一段可读可写匿名映射，或一段只读文件映射。Page Fault 发生后，内核首先不是找物理页，而是先看这个地址落在哪个 VMA 上，因为只有先确认“这段地址本来该不该访问”，才知道后续是补页还是报错。

新手最容易混淆的一点是：`malloc(1<<30)` 成功，不代表 1 GB 物理内存已经立刻到手。很多时候这只是拿到了一段虚拟地址范围。第一次真正写入某个页时，CPU 发现页表中没有有效映射，才会触发缺页，进入内核补页流程。这就是“申请成功但一写就变慢甚至 OOM”的根源。

简化伪代码可以写成这样：

```text
ptr = malloc(1GB)
for each page in ptr:
    write page
    if page not present in page table:
        raise page fault
        kernel:
            find VMA
            check permission
            alloc physical page or swap-in
            update page table
            resume instruction
```

这个边界很重要：Page Fault 不一定是错误。它常常只是“第一次访问某页”的正常路径。只有地址不合法、权限不对，或系统真的没法补页时，它才会变成 `SIGSEGV` 或 OOM 事件。

---

## 核心机制与推导

Page Fault 的处理路径可以压缩成一条主线：

`CPU 访问虚拟地址 -> 页表缺项/权限不符 -> 硬件记录故障地址到 CR2 -> 内核查 VMA -> 分配或换入页 -> 更新页表 -> 重启指令或发信号`

这里的 **CR2** 可以理解为“x86 CPU 记住本次故障地址的寄存器”。内核进入页错误处理函数后，会先读出这个地址，再结合错误码判断是“不存在页”还是“权限错误”。

一个玩具例子：

1. 进程调用 `malloc(8192)`，拿到两页虚拟地址。
2. 这时页表里可能还没有实际物理页。
3. 第一次写 `buf[0]`，CPU 发现该虚拟页不存在，触发 Page Fault。
4. 内核在 VMA 中确认这段堆内存合法且可写。
5. 内核分配一个物理页，清零，更新页表。
6. CPU 回到用户态，重新执行刚才那条写指令。
7. 程序继续运行，像什么都没发生过一样。

如果访问的是映射文件页，补页动作可能变成“从磁盘读入页面缓存”；如果访问的是被换出的匿名页，动作可能变成“从 swap 读回”；如果访问越过了可增长栈的边界，内核还可能扩展栈；如果访问地址根本不在任何合法 VMA 中，就会触发非法访问。

这个过程背后的核心判断是：页错误是“可修复”还是“不可修复”。可修复就补页，不可修复就报错。

OOM Killer 的触发，则发生在“页错误需要新页，但系统无法提供”的更后阶段。常见路径是：

1. 进程触发缺页，需要分配新物理页。
2. 内核尝试直接回收、后台回收、写回、swap-out。
3. 如果仍然无法拿到足够页框，就进入 OOM 路径。
4. 内核给候选进程打分，选择牺牲对象。
5. 杀掉目标进程，回收其内存，再让系统继续推进。

常见的近似理解公式是：

$$
\text{badness score} \approx \frac{\text{current\_rss}}{\text{total\_mem}} \times 1000 + \text{oom\_score\_adj}
$$

其中 **RSS** 是 Resident Set Size，可以理解为“当前真正驻留在物理内存中的页”。不少文章会把它口语化成：

$$
\text{badness score} \approx \text{usage\_pct} \times 10 + \text{oom\_score\_adj}
$$

这不是逐行等价于内核源码的唯一表达式，但足够帮助初学者建立判断框架：占得越多、人工加分越高，越容易被杀。

继续用 10 GB RAM 的例子。某进程当前 RSS 为 3 GB，那么它大致占了 30%，基础分接近 300；如果它的 `oom_score_adj=200`，那最终危险度就明显上升。相反，数据库主进程如果设置了负值，例如 `-500`，就相当于人为告诉内核“别先动它”。

`vm.swappiness` 会影响 OOM 之前系统有多愿意使用 swap：

| `vm.swappiness` | 倾向 | 结果 |
|---|---|---|
| `0` | 极度克制使用 swap | 匿名页尽量留在 RAM，容易更早逼近 OOM |
| `60` | 默认均衡思路 | 通用发行版常见默认值 |
| `100` | 更积极把冷页换出 | RAM 压力缓解更早，但磁盘 IO 和延迟风险更高 |

`vm.overcommit_memory` 则决定“内核对未来内存需求是否乐观”：

| 值 | 含义 | 风险 |
|---|---|---|
| `0` | 启发式判断 | 通用，但不可完全预测 |
| `1` | 总是允许 overcommit | 申请更容易成功，但 OOM 可能来得更突然 |
| `2` | 严格限制承诺量 | 更保守，更适合关键服务预留 |

如果把整条机制写成文本流程图，就是：

```text
访问虚拟地址
-> CPU 检查页表失败
-> CR2 记录故障地址
-> 内核查找 VMA
-> 合法且可修复？
   -> 否：SIGSEGV / BUS ERROR
   -> 是：分配物理页或 swap in / 文件回填
-> 更新页表
-> 重启原指令
-> 若分配阶段无可用内存：进入 OOM Killer
```

---

## 代码实现

用户态程序改不了内核的缺页路径，但可以观察和调优它的结果。最常见的抓手有三个：

1. 读 `/proc/meminfo` 看物理内存和 swap 状态。
2. 读 `/proc/<pid>/oom_score` 与 `/proc/<pid>/oom_score_adj` 看某进程在 OOM 里的危险程度。
3. 改 `sysctl` 或 `/proc/<pid>/oom_score_adj`，影响系统整体和单进程行为。

先给一个可运行的 Python 玩具脚本。它不依赖特权，只是根据一个简化公式估算 OOM 风险，用来帮助理解 badness 分数，不是复刻内核源码：

```python
def estimate_badness(current_rss_gb: float, total_mem_gb: float, oom_score_adj: int = 0) -> int:
    assert total_mem_gb > 0
    assert current_rss_gb >= 0
    assert -1000 <= oom_score_adj <= 1000
    usage_pct = current_rss_gb / total_mem_gb
    score = int(usage_pct * 1000) + oom_score_adj
    if score < 0:
        return 0
    if score > 1000:
        return 1000
    return score

toy = estimate_badness(3, 10, 0)
assert toy == 300

batch_job = estimate_badness(6, 10, 200)
assert batch_job == 800

critical_service = estimate_badness(3, 10, -200)
assert critical_service == 100

print("toy:", toy)
print("batch_job:", batch_job)
print("critical_service:", critical_service)
```

真实工程里更有用的是一个观测脚本。下面这个 shell 片段读取总内存、可用内存、swap 使用、指定进程的 OOM 分数，并输出一个最小建议：

```sh
#!/bin/sh
PID="${1:-$$}"

mem_total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
mem_avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
swap_total_kb=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
swap_free_kb=$(awk '/SwapFree/ {print $2}' /proc/meminfo)

oom_score=$(cat "/proc/$PID/oom_score")
oom_score_adj=$(cat "/proc/$PID/oom_score_adj")

printf "PID=%s\n" "$PID"
printf "MemTotal=%s kB\n" "$mem_total_kb"
printf "MemAvailable=%s kB\n" "$mem_avail_kb"
printf "SwapUsed=%s kB\n" "$((swap_total_kb - swap_free_kb))"
printf "oom_score=%s\n" "$oom_score"
printf "oom_score_adj=%s\n" "$oom_score_adj"

if [ "$oom_score" -gt 500 ]; then
  printf "advice: this process is relatively easy to kill under OOM\n"
else
  printf "advice: current OOM risk is not high\n"
fi
```

如果要主动调整行为，常见命令如下：

```sh
# 查看某进程当前的 OOM 偏置
cat /proc/12345/oom_score_adj

# 把一个批处理任务调得更容易被杀
printf "300\n" | sudo tee /proc/12345/oom_score_adj

# 把关键服务调得更难被杀
printf -- "-500\n" | sudo tee /proc/23456/oom_score_adj

# 降低 swappiness，减少匿名页被换出的积极性
sudo sysctl vm.swappiness=10

# 严格 overcommit，避免“先承诺过多、之后再崩”
sudo sysctl vm.overcommit_memory=2
sudo sysctl vm.overcommit_ratio=50
```

这里的 **真实工程例子** 是深度学习训练。训练进程不只吃显存，也会吃 CPU 内存和 swap。数据预处理、pinned memory、checkpoint、GPU-CPU 迁移都可能推高主机侧内存压力。此时如果 swap 很小，训练任务会更早触发 OOM；如果 swap 很大但 `swappiness` 又过高，系统可能陷入频繁换页，训练吞吐明显下降。也就是说，swap 不是“越大越好”，而是“要有，但不能指望它替代 RAM”。

---

## 工程权衡与常见坑

第一类常见坑是把 Page Fault 全当异常。实际上，按需分配下的首次缺页是正常成本；真正危险的是 **major fault** 变多，也就是需要从磁盘或 swap 读入，这会直接引入毫秒级甚至更高延迟。对在线服务，这种延迟尖峰会转化为长尾请求；对训练任务，这会转化为 step time 抖动。

第二类坑是对 swap 的极端态度。完全关闭 swap，看起来“避免了磁盘拖慢”，但代价是匿名内存几乎没有缓冲区，一旦瞬时峰值出现，就可能直接 OOM。反过来，把 swap 开很大且 `swappiness` 过高，系统会愿意更早把匿名页换出去，结果是机器没死，但业务慢得像死了一样。

Run:ai 和 NVIDIA 在训练场景里反复强调的一点是：一些 GPU 工作负载会消耗大量主机侧 swap 作为迁移和缓冲空间，单任务甚至可能接近对应 GPU 显存量级。比如 80 GB GPU 的训练任务，主机侧若没有足够 swap 文件，任务不一定马上报错，但会在压力上升时出现明显退化。这就是“预留 swap 的工程价值”：它不是为了常态依赖，而是为了吸收高峰和避免直接崩溃。

第三类坑是只盯 OOM Killer，不做预防。OOM Killer 本质上是最后的止损器，不是资源治理方案。尤其在多租户主机上，如果没有 overcommit、cgroup、优先级隔离，最终被杀的进程可能恰好是最关键的那个。

下面是常见调参建议：

| 参数/设置 | 建议值 | 预期效果 | 风险 |
|---|---|---|---|
| `vm.swappiness` | `10~20` | 降低匿名页过早换出 | 峰值时更可能早触发 OOM |
| `vm.overcommit_memory` | `2` | 严格控制承诺内存 | 某些大申请会更早失败 |
| `vm.overcommit_ratio` | `50` | 给系统保留更明确边界 | 需结合业务峰值评估 |
| `oom_score_adj` 批任务 | `100~300` | 出事先杀批任务 | 批任务稳定性下降 |
| `oom_score_adj` 核心服务 | `-200~-800` | 降低核心服务被杀概率 | 可能把风险转嫁给其他进程 |
| `panic_on_oom` | `0` | OOM 时继续用 Killer 而非 panic | 某些极端场景下恢复不彻底 |

一个常见的 `sysctl` 片段如下：

```conf
vm.swappiness = 10
vm.overcommit_memory = 2
vm.overcommit_ratio = 50
vm.panic_on_oom = 0
```

要注意两个误区：

1. `oom_score_adj=-1000` 不是“永远安全”，而是“尽量不杀”。如果系统约束配置失衡，内核仍可能进入非常糟糕的状态。
2. `malloc` 成功不等于业务一定安全。严格来说，真正的风险常常在“写入并触发实际分配”的时刻才暴露。

---

## 替代方案与适用边界

如果目标是“不要让某个进程把整机拖死”，仅靠 OOM Killer 太晚了，更稳妥的手段是 **cgroup**。cgroup 可以理解为“把一组进程放进同一个资源笼子”。在 cgroup v2 下，`memory.max` 设上限，`memory.high` 设软阈值，效果比事后等 OOM 更可控。

最小例子：

```sh
echo 4G | sudo tee /sys/fs/cgroup/myjob/memory.max
echo 3G | sudo tee /sys/fs/cgroup/myjob/memory.high
echo $$ | sudo tee /sys/fs/cgroup/myjob/cgroup.procs
```

这表示把当前 shell 加入 `myjob` 组，并把它所在组的内存限制在 4 GB，3 GB 开始就进入更强的回收压力。这样即使该组内发生 OOM，影响范围也更容易被限制在组内，而不是整机扩散。

另一个可选手段是 `mlock`。它可以理解为“把关键页锁在内存里，不允许被换出”。这适合少量、关键、对延迟敏感的内存，例如实时控制、极低延迟服务中的关键索引页。但它不适合大规模滥用，因为锁页本身就是在减少系统可回收空间。

三种思路的比较如下：

| 方案 | 解决的问题 | 适用场景 | 不适用场景 |
|---|---|---|---|
| OOM Killer | 最后阶段止损 | 通用系统兜底 | 需要强隔离和可预测资源治理 |
| cgroup 限制 | 提前约束单组资源 | 容器、多租户、批任务平台 | 单机单进程简单场景中配置复杂度偏高 |
| `mlock` | 防止关键页被换出 | 实时系统、关键缓存 | 大内存训练、普通批处理 |

所以适用边界可以简单记成：

1. 单机普通服务：先调 `swappiness`、`overcommit`、`oom_score_adj`。
2. 多租户或容器环境：优先上 cgroup，OOM 只做兜底。
3. 深度学习和大型数据处理：准备合理 swap，但不要让系统长期依赖 swap 维持常态吞吐。
4. 极低延迟场景：只锁极少量关键页，不要把 `mlock` 当通用内存管理工具。

---

## 参考资料

| 类别 | 主旨 | URL |
|---|---|---|
| 官方文档 | `/proc/sys/vm`、`swappiness`、`overcommit`、`panic_on_oom` 说明 | https://docs.kernel.org/6.15/admin-guide/sysctl/vm.html |
| 专题文章 | 虚拟地址空间、VMA、Page Fault 处理路径 | https://kahibaro.com/course/27-linux/1866-memory-management |
| 专题文章 | `oom_score`、`oom_score_adj` 与 OOM 生存策略 | https://rrampage.github.io/2018/10/04/surviving-the-linux-oom-killer/ |
| 厂商建议 | 训练任务中的 swap 规划与资源优化 | https://run-ai-docs.nvidia.com/saas/platform-management/runai-scheduler/resource-optimization/memory-swap |
| 系统调优文档 | 生产环境内存调优建议 | https://documentation.suse.com/sles/15-SP4/html/SLES-all/cha-tuning-memory.html |

- Kernel 文档：适合查参数定义，尤其是 `vm.swappiness`、`vm.overcommit_memory`、`vm.panic_on_oom` 的正式语义。
- Kahibaro：适合理解虚拟地址空间布局和页错误处理流程，帮助把“地址不在页表里”这件事和内核补页关联起来。
- rrampage：适合理解 `oom_score`、`oom_score_adj` 的直观含义，建立“谁更容易被杀”的工程判断。
- Run:ai / NVIDIA：适合理解深度学习训练为什么会受 swap 影响，以及为什么“有 swap”和“别过度依赖 swap”必须同时成立。
