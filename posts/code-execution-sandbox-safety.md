## 核心结论

代码执行沙箱，是指在运行不可信代码时，先给它套上一层受控执行环境，再明确规定它“能做什么、最多做多少、能看到什么”。白话说，就是把陌生程序关进一间权限很少、资源很小、出口很少的房间里运行。

它的目标不是“绝对安全”。绝对安全在通用操作系统上通常不存在，因为内核、运行时、依赖库和配置都可能有漏洞。沙箱真正有效的目标，是把破坏面压缩到可控范围内，让失败尽量局部化、低成本、可观测。

真正有工程价值的方案，几乎都不是单点防护，而是四层一起做：

| 层次 | 解决的问题 | 典型手段 |
|---|---|---|
| 系统调用控制 | 不让代码直接碰高风险内核能力 | `seccomp-BPF` 白名单 |
| 资源限制 | 不让代码吃满 CPU、内存、进程表 | `cgroup v2`、`RLIMIT` |
| 可见面隔离 | 不让代码看到不该看到的文件、网络、设备 | namespace、`chroot`、只挂载必要目录 |
| 最小权限 | 即使进到环境里，也不给多余能力 | 非 root、去 capability、最少 FD 传递 |

因此，判断一个沙箱是否靠谱，不能只看“有没有超时”，也不能只看“跑在容器里”。要看它是否同时回答了三个问题：它能调用哪些内核能力，最多消耗多少资源，能接触到哪些宿主对象。

---

## 问题定义与边界

沙箱要解决的问题，是执行来自用户、模型或外部系统的不可信代码时，如何避免三类后果：

1. 读到不该读的数据，比如宿主机文件、环境变量、凭证、其他用户结果。
2. 做出不该做的动作，比如发网络请求、挂载文件系统、调试其他进程。
3. 吃掉不该吃的资源，比如死循环占满 CPU，申请大内存拖垮机器，或创建过多进程把系统进程表打满。

这里的“不可信代码”，意思不是“代码一定恶意”，而是“系统不能假定它守规矩”。在线判题、LLM 代码解释器、插件市场、自定义脚本平台，本质上都属于这个问题。

但沙箱也有明确边界。它不是完整防火墙，不是漏洞修复器，也不是供应链审计工具。下面这个表最重要：

| 项目 | 能防什么 | 不能防什么 | 常见误解 |
|---|---|---|---|
| 系统调用过滤 | 限制危险内核入口 | 已允许调用里的内核漏洞 | “拦了几个 syscall 就安全了” |
| 资源限制 | 防止 CPU/内存/进程数失控 | 算法层逻辑错误 | “设了超时就够了” |
| 文件/网络隔离 | 限制可见资源范围 | 业务层鉴权错误 | “容器默认就完全隔离” |
| 最小权限 | 降低被利用后的破坏能力 | 已授权资源被合法滥用 | “非 root 就没风险” |

玩具例子最容易说明边界。在线判题里，用户提交三段代码：

- `while True: pass`：这是 CPU 边界问题。
- `a = "x" * (1024**3)`：这是内存边界问题。
- 不断 `fork()` 子进程：这是进程数边界问题。

如果系统只设置“2 秒超时”，第一种通常能挡住，第二种和第三种却可能在 2 秒内先把宿主机拖垮。也就是说，超时只覆盖“活太久”，不覆盖“吃太多”。

---

## 核心机制与推导

沙箱的核心机制可以写成三条约束：

$$
allow(s)=1[s \in S_{allow}]
$$

$$
T_{cpu} \le Q_{cpu}, \quad M_{rss} \le M_{max}, \quad P \le P_{max}
$$

这里：

- 系统调用，是程序向内核请求服务的入口，白话说就是程序真正碰操作系统能力的门。
- 白名单，是只允许明确列出的能力，其他一律拒绝的策略。
- `RSS` 是常驻内存大小，白话说就是进程当前真正占着不放的物理内存近似量。

第一条约束控制“能做什么”。若 `s` 不在 `S_allow` 里，就返回错误、触发陷阱或直接杀进程。第二条约束控制“最多做多少”。即使允许执行，也不能无限制消耗资源。

用一个最小数值例子说明：

- `Q_cpu = 100ms`
- `M_max = 64MiB`
- `P_max = 32`

再假设系统调用白名单只允许 `read/write/exit/fstat/mmap` 这类最低限度调用。

那么三种故障的结果分别是：

| 触发行为 | 约束撞线 | 处理动作 | 结果 |
|---|---|---|---|
| 死循环 | `T_cpu > 100ms` | 节流或终止 | 任务超时退出 |
| 申请 128MiB 内存 | `M_rss > 64MiB` | OOM kill 或分配失败 | 进程被杀或报内存错误 |
| `fork` 风暴 | `P > 32` | 拒绝创建新进程 | 无法继续扩散 |

为什么“只做超时”不够，现在可以精确说明。因为失控不只表现为时间维度，还表现为空间和并发维度。系统风险可近似看成：

$$
Risk \approx Capability \times Visibility \times Resource\ Budget
$$

- `Capability` 是能调用多少危险能力。
- `Visibility` 是能看到多少宿主对象。
- `Resource Budget` 是能消耗多少资源。

只限制时间，相当于只压了一个维度。若另外两个维度保持开放，整体风险仍然很高。

真实工程例子是 LLM 代码执行平台。用户让模型生成一段 Python，再交给后端实际执行。这里最危险的不是“代码跑慢”，而是代码可能去扫本地目录、打外网、读环境变量、创建子进程、压垮共享节点。所以工程上必须把授权模型做成“先拒绝，再最小开放”，而不是“默认全开，出事再补规则”。

---

## 代码实现

落到 Linux 实现，最常见的组合是 `seccomp-BPF + cgroup v2 + namespace + 最小权限`。

`seccomp-BPF` 可以理解成“系统调用过滤器”。它不是拦截源代码，而是在进程触发 syscall 时做判定。规则常见写法是白名单：只放行最小必需调用，其他直接 `KILL` 或返回 `EPERM`。

伪代码如下：

```c
// 伪代码：系统调用白名单
if (syscall in {read, write, exit, fstat, mmap, munmap, brk, rt_sigreturn}) {
    allow();
} else {
    kill_or_errno();
}
```

资源限制通常由 `cgroup v2` 完成。可以把它理解成“给一组进程发预算单”，预算超了就限流、拒绝或杀掉。配置示意：

```text
cpu.max = 100000 100000
memory.max = 67108864
pids.max = 32
```

含义分别是：

- `cpu.max`：一个周期内最多吃多少 CPU 时间。
- `memory.max`：内存硬上限 64MiB。
- `pids.max`：最多 32 个进程或线程。

namespace 是“命名空间隔离”，白话说是让进程看到一套被裁剪过的世界。常见包括：

- mount namespace：只给它看到指定文件系统。
- pid namespace：让它只能看到自己的进程树。
- network namespace：默认不给外网，或只给受控网络。
- user namespace：把容器内的“root”映射成宿主上的非特权身份。

`chroot` 或只读根文件系统进一步缩小文件可见面。FD 传递最小化也很关键。FD 就是文件描述符，白话说是进程手里已经打开的资源把手。哪怕文件系统隔离做得不错，如果父进程把宿主上的敏感 FD 直接传进去，沙箱照样会漏。

下面给一个可运行的 Python 玩具实现。它不是真正的内核级沙箱，只是把“白名单 + 资源预算”的判定逻辑抽象出来，帮助理解机制。

```python
from dataclasses import dataclass

@dataclass
class SandboxPolicy:
    allowed_syscalls: set
    cpu_quota_ms: int
    memory_max_mib: int
    pids_max: int

def syscall_allowed(policy: SandboxPolicy, syscall: str) -> bool:
    return syscall in policy.allowed_syscalls

def resource_allowed(policy: SandboxPolicy, cpu_ms: int, memory_mib: int, pids: int) -> bool:
    return (
        cpu_ms <= policy.cpu_quota_ms
        and memory_mib <= policy.memory_max_mib
        and pids <= policy.pids_max
    )

policy = SandboxPolicy(
    allowed_syscalls={"read", "write", "exit", "fstat", "mmap"},
    cpu_quota_ms=100,
    memory_max_mib=64,
    pids_max=32,
)

# 允许的正常请求
assert syscall_allowed(policy, "read") is True
assert resource_allowed(policy, cpu_ms=20, memory_mib=8, pids=1) is True

# 禁止的高风险行为
assert syscall_allowed(policy, "fork") is False
assert syscall_allowed(policy, "ptrace") is False

# 三类典型越界
assert resource_allowed(policy, cpu_ms=101, memory_mib=8, pids=1) is False
assert resource_allowed(policy, cpu_ms=20, memory_mib=128, pids=1) is False
assert resource_allowed(policy, cpu_ms=20, memory_mib=8, pids=33) is False
```

这个玩具例子对应真实系统里的三层实现关系：

| 抽象逻辑 | 真实工程里的常见做法 |
|---|---|
| `syscall_allowed` | `seccomp-BPF` 过滤 syscall |
| `resource_allowed` | `cgroup v2` / `RLIMIT` 限制资源 |
| 独立 policy | 每次任务启动时注入独立执行配置 |

---

## 工程权衡与常见坑

第一类坑，是把 `seccomp` 当成完整沙箱。它只解决“哪些 syscall 能进入内核”，不解决“内核本身是否有漏洞”。如果允许的调用路径里存在漏洞，攻击仍可能发生。所以它必须和隔离边界、最小权限、内核更新一起使用。

第二类坑，是只限 CPU，不限内存和进程数。这样会出现两个高频失败模式：

- `fork bomb` 还没等超时，就先把系统进程表占满。
- 内存膨胀还没等超时，就先触发宿主 OOM，误伤别的租户。

第三类坑，是做 syscall 黑名单，而不是白名单。黑名单的问题在于默认开放面太大，只要漏掉一个危险入口，攻击面就还在。白名单的默认状态是“不允许”，更适合不可信执行。

第四类坑，是规则没有检查架构。不同架构上 syscall 编号可能不同，如果过滤规则没校验 `arch`，可能出现错判甚至绕过。

第五类坑，是忘了“可见面”本身就是能力。很多系统明明限制了 syscall，却把 `/proc`、宿主目录、云凭证、Docker socket 或环境变量直接暴露给了沙箱。这样即使进程老老实实只调用 `read`，也能把敏感信息全读走。

下面这个对照表最实用：

| 错误做法 | 风险 | 正确做法 |
|---|---|---|
| 只设置超时 | 挡不住内存膨胀和进程爆炸 | 同时限制 CPU、内存、`pids` |
| 只做黑名单 | 漏掉一项就可能失守 | 优先做 syscall 白名单 |
| 容器内直接跑 root | 被利用后破坏面更大 | 非 root 运行，去 capability |
| 给完整网络 | 可外连、扫描、数据外传 | 默认断网，只按需开放 |
| 挂完整文件系统 | 可读宿主敏感文件 | 最小挂载，只读根，临时工作目录 |
| 不做观测 | 失败原因不清楚，难以调参 | 记录 syscall 拒绝、OOM、超时、退出码 |

真实工程里还要考虑两个权衡。第一是性能。限制越细，启动和维护成本通常越高。第二是兼容性。白名单太窄，很多语言运行时自己就起不来，比如 Python、Node.js、JVM 各自依赖的 syscall 集合不同。因此工程实践不是“越少越好”，而是“以可运行的最小集合为准”。

---

## 替代方案与适用边界

沙箱不是只有一种做法。常见方案可以按“隔离强度、性能、复杂度”来比较。

| 方案 | 隔离强度 | 性能开销 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|
| 进程级沙箱 + `seccomp/cgroup` | 中 | 低 | 中 | 单机任务执行、在线判题 |
| 容器 | 中 | 低到中 | 中 | 快速落地的多租户执行 |
| gVisor | 中到高 | 中 | 中到高 | 希望增强 syscall 隔离但保留容器体验 |
| microVM（如 Firecracker） | 高 | 中 | 高 | 高风险第三方代码、强租户隔离 |
| Wasm runtime | 取决于 runtime 和宿主设计 | 低到中 | 中 | 插件执行、规则脚本、跨语言轻量任务 |

容器的优点是成熟、快、生态完整，适合先落地。缺点是边界仍然共享宿主内核，所以不能把它当成虚拟机等价物。gVisor 的思路是用用户态内核代理大量 syscall，减少直接进入宿主内核的表面积。microVM 则把边界抬到虚拟化层，安全感更强，但冷启动、镜像管理、网络与存储编排都会更复杂。

Wasm runtime 适合另外一类场景。WebAssembly 可以理解成一种受控字节码格式，很多 runtime 天生就把系统能力做成显式注入，而不是默认开放。这很适合插件、模板执行、用户自定义规则等场景。但如果你的用户提交的是任意 Python/C/C++，Wasm 通常不能直接替代通用 Linux 沙箱。

一个真实工程选型例子：

- 如果你在做 LLM 代码执行平台，第一阶段通常用“容器 + `seccomp` 白名单 + `cgroup v2` + 默认断网”最快。
- 如果后续开始执行来源复杂的第三方代码，且租户隔离要求明显提高，就应该考虑升级到 Firecracker 这类 microVM。
- 如果业务是执行用户规则、表达式、小插件，而不是通用系统程序，那么 Wasm runtime 往往更干净，因为它默认就不暴露宿主能力。

因此，选型边界可以概括成一句话：轻量、高频、成本敏感的执行，优先容器化沙箱；高风险、多租户、强隔离需求，优先 microVM；能力模型可重定义的插件场景，优先 Wasm。

---

## 参考资料

1. [Linux seccomp BPF 官方文档](https://www.kernel.org/doc/html/v6.6/userspace-api/seccomp_filter.html)
2. [Linux cgroup v2 官方文档](https://kernel.org/doc/html/next/admin-guide/cgroup-v2.html)
3. [Linux namespaces 官方文档](https://docs.kernel.org/admin-guide/namespaces/index.html)
4. [gVisor 官方文档](https://gvisor.dev/docs/)
5. [Firecracker 设计文档](https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md)
6. [Wasmtime ResourceLimiter 文档](https://docs.wasmtime.dev/api/wasmtime/trait.ResourceLimiter.html)
