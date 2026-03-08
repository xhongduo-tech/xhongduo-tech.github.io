## 核心结论

Agent 代码沙箱解决的不是“模型会不会写代码”，而是“模型写出来、但还没被人工审核的代码，能不能被限制在一组可验证的边界里执行”。

这里的“沙箱”，可以先按新手能操作的方式理解成一句话：

> 给不受信任的代码一套默认拒绝、按需放行的运行环境。

从公开资料看，常见隔离层大致分三层：

| 层级 | 白话解释 | 主要能力 | 主要短板 |
| --- | --- | --- | --- |
| 容器 | 把进程关进隔离出来的进程/文件系统/网络视图里 | namespace、cgroup、挂载隔离、资源配额 | 仍共享宿主内核 |
| gVisor 一类用户态内核层 | 在应用和宿主内核之间加一层“用户态内核” | 拦截并重实现大量 syscall，缩小直接攻击宿主内核的机会 | 兼容性和性能都比纯容器更复杂 |
| microVM | 给任务单独分配一个轻量虚拟机和独立内核 | 硬件虚拟化边界、更强多租户隔离 | 启动、镜像、调度和成本更重 |

因此可以用一个近似公式概括：

$$
安全等级 \approx \text{microVM} > \text{gVisor类隔离} > \text{纯容器}
$$

但这个公式只表示“隔离层强度的相对关系”，不表示“用了更强隔离层就自动安全”。真正有效的是多层叠加：

$$
\text{实际安全性} = \text{隔离层强度} \times \text{最小权限执行纪律}
$$

如果文件系统放开、网络全放、超时不设、系统调用不过滤，那么再强的底层隔离也会被上层配置抵消。

截至 2026 年 3 月 8 日，三个常被拿来讨论的产品，公开信息可以这样理解：

| 服务 | 公开可确认的信息 | 关键边界 |
| --- | --- | --- |
| OpenAI Code Interpreter | 文档把 `container` 描述为可运行 Python 的全沙箱虚拟机；默认内存档位是 `1g`；空闲 20 分钟过期 | 以短生命周期、受控文件环境、内存档位和自动过期作为边界 |
| Claude 沙箱能力 | Claude Code 文档公开说明：默认只写当前工作目录；网络访问通过沙箱外代理做域名限制；macOS 用 Seatbelt，Linux/WSL2 用 bubblewrap | 重点在 OS 原语和策略控制，公开文档没有详细披露统一底层虚拟化实现 |
| E2B | 官方公开说明每个 sandbox 由 Firecracker microVM 驱动；最长会话可到 24 小时（套餐相关） | 以内核级隔离和短命/限时环境作为主要边界 |

一个最小化的直观例子是：

- 只能写 `/workspace`
- 只能访问 `pypi.org` 和 `files.pythonhosted.org`
- 单条命令 60 秒超时
- 内存上限 1 GiB
- 会话空闲自动销毁

这时即使 Agent 生成了恶意脚本，它能做的事也会被压缩到这个小盒子里。它不能直接改宿主的 `~/.ssh`，也不能把数据发往任意域名。

---

## 问题定义与边界

问题定义很具体：要安全执行“模型刚刚生成、还没有经过人工审核”的代码。

这里先把几个容易混淆的词说清楚：

| 术语 | 含义 | 为什么重要 |
| --- | --- | --- |
| 不受信任代码 | 不是说代码一定恶意，而是你还不能默认它安全 | 安全设计必须按最坏情况建模 |
| 宿主机 | 真正运行沙箱的机器 | 一旦被越权访问，风险会从任务级升级到系统级 |
| 边界 | 沙箱允许代码接触到的资源集合 | 边界定义不清，隔离就无法验证 |
| 最小权限 | 只给任务完成目标所必需的权限 | 权限越少，出事后的破坏面越小 |

这类代码的风险面通常有四类：

| 攻击面 | 白话解释 | 典型后果 |
| --- | --- | --- |
| 文件系统 | 乱读乱写磁盘 | 读取密钥、篡改配置、植入持久化后门 |
| 网络 | 任意连接外部服务 | 数据外传、下载二阶段恶意载荷 |
| 系统调用 | 直接向内核申请高危能力 | 提权、逃逸、调试别的进程、挂载新文件系统 |
| 资源消耗 | 死循环、 fork 炸弹、超大内存分配 | 节点被拖垮，形成拒绝服务 |

所以沙箱不是“把目录切一下”就结束。文件隔离只能限制“读写哪里”，不能限制“往哪发数据”；网络隔离只能限制“连到哪”，不能限制“本地改了什么”。两者必须一起出现。

可以把它写成一个更工程化的边界模型：

$$
\text{Sandbox Boundary} =
\{\text{filesystem}, \text{network}, \text{syscalls}, \text{cpu}, \text{memory}, \text{time}, \text{lifetime}\}
$$

任何一个集合元素缺位，攻击面就会重新打开。

另一个经常被低估的边界是“时间”。短命环境本身就是安全机制，因为它减少了长期潜伏和状态残留：

| 约束 | 常见区间 | 作用 |
| --- | --- | --- |
| 单命令超时 | `30s` 到 `120s` | 防止卡死、阻塞和无限执行 |
| 内存限制 | `512 MiB` 到 `2 GiB` | 防止单任务拖垮节点 |
| CPU/进程数限制 | 按节点配额设定 | 防止 fork 炸弹和资源争抢 |
| 空闲过期 | `5min` 到 `20min` 甚至更长 | 降低状态残留和横向滥用机会 |
| 会话总寿命 | 单请求级到数小时 | 在可用性和风险之间折中 |

这些数字不是安全定律，只是可用性、成本和风险的折中。OpenAI 公开文档里的 `1g` 默认档位和 20 分钟空闲过期，就是典型例子：够多数分析任务使用，但不鼓励把环境长期当成持久服务器。

---

## 核心机制与推导

判断一次代码执行是否安全，本质上是在问：攻击者需要连续穿透多少层边界，才能从“任务内破坏”升级成“宿主级破坏”。

可以粗略写成：

$$
P(\text{逃逸成功}) \approx
P(\text{文件策略失效})
\times
P(\text{网络策略失效})
\times
P(\text{syscall/隔离层失效})
$$

这不是形式化证明，但它能说明一个关键工程事实：

> 多层隔离的价值，不是保证某一层永远不出错，而是把“单点失败”变成“多点连续失败”。

三层机制分别负责不同事情。

| 层 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| 容器 | 隔离进程视图、挂载点、用户空间、网络命名空间和资源配额 | 不能消除共享宿主内核的事实 |
| gVisor 类层 | 用用户态内核拦截和重实现应用侧 syscall 接口 | 不能保证所有 Linux 程序都 100% 无差别兼容 |
| microVM | 用独立 guest kernel 和虚拟硬件提供更强边界 | 不能替代上层最小权限策略 |

### 1. 文件系统隔离

文件系统隔离通常至少做三件事：

1. 只给工作目录写权限
2. 把系统目录设为只读、不可见或根本不挂载
3. 对确实需要写的额外目录做显式 allowlist

Claude Code 的公开文档就是这个思路：默认只能写当前工作目录；如果 `kubectl` 或 `terraform` 必须写额外路径，再用 `sandbox.filesystem.allowWrite` 精确放开。

这背后的原则很简单：

$$
\text{可写路径集合} \subseteq \text{任务必需路径集合}
$$

如果你把整个家目录、Docker socket 或宿主源码树都挂进去，可写范围就已经大于任务真实需求，隔离价值会立刻下降。

### 2. 网络隔离

网络隔离常见做法也很直接：

1. 默认拒绝所有出站
2. 只放行白名单域名、固定代理出口或特定 API
3. 所有子进程继承同样限制

Claude Code 公开文档说明，网络访问由沙箱外代理控制，并且限制对所有脚本、程序和子进程生效。这个设计很关键，因为很多恶意行为不需要高深技巧，只需要一条 `curl` 或一个 `requests.post()`。

对新手来说，可以把它理解成：

- 文件系统隔离回答“你能拿到什么”
- 网络隔离回答“你能把它发到哪里”

如果只有前者没有后者，被 prompt injection 或恶意依赖诱导后的代码，仍可能把工作区里的敏感文件传出去。

### 3. 系统调用限制

系统调用是程序向操作系统申请能力的入口，比如：

- 创建进程
- 映射内存
- 打开文件
- 挂载文件系统
- 调试其他进程
- 配置网络接口

容器环境里常见的是 `seccomp` 过滤；Docker 官方文档说明，默认 profile 采用 allowlist 思路，只允许一组常见 syscall。gVisor 更进一步，把大量 syscall 留在用户态内核里处理，减少应用直接接触宿主内核的机会。

这层的设计原则可以写成：

$$
\text{Allowed Syscalls} = \text{任务最低必需集合}
$$

像 `mount`、`ptrace`、高危 `bpf` 相关能力，以及不必要的 namespace 操作，通常都不应该默认开放。

### 4. 资源与生命周期限制

资源限制不是“附加项”，而是第四道安全边界。因为很多攻击并不追求逃逸，只追求拖垮机器。

| 限制项 | 目标 | 不设会怎样 |
| --- | --- | --- |
| 内存上限 | 防止单任务 OOM 整个节点 | pandas、大模型推理、恶意分配都能拖垮宿主 |
| CPU 时间 | 防止死循环长期占满核 | 任务饥饿，其他租户受影响 |
| 进程数上限 | 防止 fork 炸弹 | 很快耗尽 PID 和调度资源 |
| 文件大小/磁盘配额 | 防止写爆磁盘 | 节点不可用，日志和系统服务异常 |
| 空闲 TTL | 防止环境长期残留 | 恶意状态更容易潜伏 |

### 5. 公开产品的工程形态

公开信息里，OpenAI 和 E2B 展示了两种典型产品形态：

| 形态 | 代表 | 公开特征 | 适合理解成什么 |
| --- | --- | --- | --- |
| 平台托管型分析环境 | OpenAI Code Interpreter | 把环境抽象成 `container`，支持文件和 Python 执行，默认内存档位可选，空闲自动过期 | 平台托管的短命执行环境 |
| microVM 执行单元 | E2B | 明确说明由 Firecracker microVM 驱动，可把会话保持到更长时间 | 把轻量 VM 当作 Agent 的执行单元 |

E2B 的公开材料适合说明一个常见工程事实：长任务并不必然要求牺牲隔离，只是你需要愿意支付更高的调度和运维成本。

---

## 代码实现

下面给两个示例。

第一个是“策略判定器”，作用是把边界规则写清楚。它不是真正的 OS 沙箱，但适合先把权限模型讲明白。

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class SandboxPolicy:
    writable_roots: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    memory_mb: int
    timeout_s: int


def _normalize(path: str) -> PurePosixPath:
    return PurePosixPath(path).as_posix() and PurePosixPath(path)


def can_write(path: str, policy: SandboxPolicy) -> bool:
    target = PurePosixPath(path)
    for root in policy.writable_roots:
        root_path = PurePosixPath(root)
        if target == root_path or root_path in target.parents:
            return True
    return False


def can_connect(domain: str, policy: SandboxPolicy) -> bool:
    return domain in policy.allowed_domains


def can_run(requested_memory_mb: int, requested_time_s: int, policy: SandboxPolicy) -> bool:
    return requested_memory_mb <= policy.memory_mb and requested_time_s <= policy.timeout_s


if __name__ == "__main__":
    policy = SandboxPolicy(
        writable_roots=("/workspace",),
        allowed_domains=("pypi.org", "files.pythonhosted.org"),
        memory_mb=1024,
        timeout_s=60,
    )

    assert can_write("/workspace/result/output.csv", policy) is True
    assert can_write("/workspace", policy) is True
    assert can_write("/etc/passwd", policy) is False

    assert can_connect("pypi.org", policy) is True
    assert can_connect("evil.example.com", policy) is False

    assert can_run(512, 30, policy) is True
    assert can_run(2048, 30, policy) is False
    assert can_run(512, 120, policy) is False

    print("policy checks passed")
```

第二个示例是“最小可运行的本地执行器”。它仍然不是完整沙箱，因为真正的网络拦截、syscall 过滤和挂载隔离应由 OS、容器运行时、gVisor 或 microVM 实现；但它至少把工作目录、超时和内存限制落到了实际执行流程里。

下面代码可在 `macOS/Linux + Python 3.11+` 运行：

```python
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalSandboxConfig:
    workspace: Path
    timeout_s: int = 5
    memory_mb: int = 256


def limit_resources(memory_mb: int) -> None:
    bytes_limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def run_python_snippet(code: str, config: LocalSandboxConfig) -> subprocess.CompletedProcess[str]:
    config.workspace.mkdir(parents=True, exist_ok=True)

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(config.workspace),
    }

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=config.workspace,
        env=env,
        text=True,
        capture_output=True,
        timeout=config.timeout_s,
        preexec_fn=lambda: limit_resources(config.memory_mb),
        check=False,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="agent-sandbox-") as tmp:
        config = LocalSandboxConfig(workspace=Path(tmp), timeout_s=3, memory_mb=128)

        safe_code = """
from pathlib import Path
Path("result.txt").write_text("hello sandbox\\n", encoding="utf-8")
print(Path("result.txt").read_text(encoding="utf-8").strip())
"""

        result = run_python_snippet(safe_code, config)
        print("return_code:", result.returncode)
        print("stdout:", result.stdout.strip())
        print("stderr:", result.stderr.strip())
        print("files:", sorted(p.name for p in Path(tmp).iterdir()))
```

运行结果应类似：

```text
return_code: 0
stdout: hello sandbox
stderr:
files: ['result.txt']
```

这个示例有三个值得新手注意的点：

| 机制 | 代码里怎么体现 | 它解决什么问题 |
| --- | --- | --- |
| 工作目录收缩 | `cwd=config.workspace` | 把默认文件读写面收缩到临时目录 |
| 超时 | `timeout=config.timeout_s` | 避免任务无限挂起 |
| 资源上限 | `RLIMIT_AS`、`RLIMIT_CPU`、`RLIMIT_NPROC` | 防止单任务滥用内存、CPU 和进程数 |

但也必须明确，它**没有**解决下面这些事情：

| 未解决项 | 为什么没解决 |
| --- | --- |
| 网络出站限制 | 需要代理、防火墙、eBPF 或容器/VM 网络策略 |
| syscall 白名单 | 需要 `seccomp`、gVisor、OS sandbox 或 hypervisor 支持 |
| 宿主内核共享风险 | 这个示例本质仍是宿主机上的普通子进程 |
| 挂载隔离 | 没有单独 mount namespace 或独立根文件系统 |

所以一个更准确的说法是：

> 这段代码是“沙箱控制器的雏形”，不是“生产级沙箱”。

如果把它映射回真实服务设计，最小初始化模型通常是下面这组参数：

| 参数 | 示例值 | 作用 |
| --- | --- | --- |
| `memory_limit` | `1g` 到 `2g` | 防止 OOM 拖垮节点 |
| `timeout` | `30s` 到 `120s` | 防止单命令挂死 |
| `idle_ttl` | `5min` 到 `20min` | 自动回收短命环境 |
| `allow_write` | 仅工作目录 | 限制持久化破坏面 |
| `allowed_domains` | 必要域名白名单 | 防止任意外联 |
| `syscall_policy` | 默认拒绝，仅放行必要项 | 压缩内核攻击面 |
| `process_limit` | 例如 `32` 到 `256` | 防止 fork 炸弹 |
| `disk_quota` | 按任务定额 | 防止写爆磁盘 |

---

## 工程权衡与常见坑

最常见的误区，是把“隔离强度”只看成一个维度。真实工程里你同时在调四个旋钮：

$$
\text{方案选择} = f(\text{安全}, \text{性能}, \text{兼容性}, \text{成本})
$$

这四个量经常互相拉扯。

| 常见问题 | 现象 | 根因 | 规避策略 |
| --- | --- | --- | --- |
| 超时过短 | 合法任务被误判失败 | 把交互式工具超时和批处理超时混成一套 | 按任务类型分级，长任务改异步 |
| 内存过小 | `pandas`、压缩解压、图像处理频繁 OOM | 资源预算和任务类型不匹配 | 预估峰值，按任务分档 |
| 网络放太宽 | prompt injection 后数据外传 | 只隔离文件，不隔离出站 | 默认拒绝，只开白名单域名 |
| 写权限放太宽 | 环境被植入持久化后门 | 图省事开放整个家目录 | 只开放精确路径 |
| 兼容性差 | 某些 CLI 在沙箱里不可用 | 依赖额外 socket、watcher、系统目录 | 单独豁免，并保留权限审计 |
| 只信单层容器 | “已经 Docker 了”仍担心逃逸 | 共享内核风险没有消失 | 高风险任务上升到 gVisor 或 microVM |

### 1. 超时不是小参数，而是系统行为

公开 issue 能看到一个很现实的问题：工具层超时和实际作业时长，经常不是一回事。

Anthropic 公开 issue 中有两类现象值得拿来说明工程坑点：

- TypeScript Agent SDK 曾有 30 秒硬编码工具超时的公开报告
- Claude Code 社区 issue 中又能看到 Bash 工具存在约 2 分钟级超时、默认值和最大值配置讨论

这说明一个事实：

> 不要把“单次工具调用”当成“通用作业调度系统”。

一个 shell 调用如果天然要跑 8 分钟，最稳的办法通常不是把同步超时一路调大，而是把任务拆成可观察、可恢复、可重试的阶段。

### 2. 长任务要拆阶段，不要赌一次跑完

更稳的设计通常是：

1. 准备阶段
2. 执行阶段
3. 产物落盘阶段
4. 状态回传阶段

每一步都应该满足两个条件：

- 失败后能知道停在什么位置
- 重新执行不会把系统搞乱

例如“下载数据 + 处理 + 生成报告”这个任务，不要设计成一条 600 秒 bash 命令；更好的做法是：

| 阶段 | 输入 | 输出 | 失败后怎么恢复 |
| --- | --- | --- | --- |
| 下载 | 数据源 URL | 本地源文件 | 检查文件是否完整，必要时断点续传 |
| 处理 | 源文件 | 中间结果 | 读取中间状态继续算 |
| 报告生成 | 中间结果 | HTML/PDF/CSV | 重新生成最终产物 |
| 回传 | 产物路径 | 上传确认状态 | 重试上传，不重跑计算 |

### 3. “需要更多权限”不等于“全开权限”

很多新手第一次做沙箱，会因为某个工具跑不通，直接把权限从“精确放行”改成“全部放行”。这通常是把兼容性问题变成安全问题。

更稳的处理顺序应该是：

1. 先确认工具到底缺什么权限
2. 再只补那一项权限
3. 记录这次豁免是否长期必要
4. 给豁免留审计线索

比如 `terraform` 只需要写 `~/.terraform.d` 或某个缓存目录，那就开放这个路径，不要顺手把整个 `~` 放开。

---

## 替代方案与适用边界

方案选择不要只看“谁更安全”，而要看“你的威胁模型值不值得为更强隔离付出更高成本”。

| 方案 | 启动速度 | 资源开销 | 安全等级 | 适用场景 |
| --- | --- | --- | --- | --- |
| 纯容器 | 快 | 低 | 中 | 低风险、短任务、大量并发 |
| 容器 + gVisor 类层 | 中 | 中 | 中高 | 中风险、多租户、需要更强 syscall 边界 |
| microVM | 较慢 | 较高 | 高 | 高敏感、多租户强隔离、长任务 |

可以把选择逻辑压缩成一个决策表：

| 业务特征 | 更合适的方案 | 原因 |
| --- | --- | --- |
| 教学、图表生成、轻量数据清洗 | 纯容器或平台托管沙箱 | 启动快、成本低、风险可控 |
| 第三方代码执行、多租户中风险任务 | 容器 + gVisor 类层 | 比纯容器更能压缩内核攻击面 |
| 金融、医疗、企业内部高敏自动化 | microVM | 不共享宿主内核，边界更强 |
| 长时间运行且带较高敏感数据 | microVM + 严格网络/文件策略 | 长寿命会话本身会扩大风险面，需要更强底层隔离 |

新手版判断法可以进一步简化成三句：

- 任务短、价值低、量大：先用容器
- 任务中风险、担心 syscall 和宿主暴露面：上 gVisor 类方案
- 任务高敏感、不能接受共享宿主内核：直接 microVM

但要记住一个更重要的边界：

> 如果你的业务必须长期挂载宿主目录、开放宽泛外网、允许 Docker socket、允许任意 Unix socket，那么再强的沙箱也会被你自己削弱。

安全从来不是产品名，而是“默认拒绝 + 精确放行 + 生命周期短 + 审计可追踪”的执行纪律。

---

## 参考资料

下表资料已按 2026 年 3 月 8 日可公开访问内容交叉核对。

| 资料 | 内容摘要 | 用途 |
| --- | --- | --- |
| [OpenAI Code Interpreter 文档](https://developers.openai.com/api/docs/guides/tools-code-interpreter) | 公开说明 `container`、默认 `1g` 内存档位、空闲 20 分钟过期，并将容器描述为全沙箱虚拟机 | 作为 OpenAI 托管代码执行环境的公开依据 |
| [Claude Code Sandboxing 文档](https://code.claude.com/docs/en/sandboxing) | 说明默认工作目录写权限、`sandbox.filesystem.allowWrite`、域名级网络控制、macOS Seatbelt、Linux bubblewrap | 作为 Claude Code 本地/CLI 沙箱边界的主要依据 |
| [Claude Code Execution Tool 文档](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool) | 说明代码执行工具可运行 Bash 与文件操作，默认无外网，文件访问限制在工作区 | 作为 Claude 托管代码执行工具边界的依据 |
| [gVisor 官方文档](https://gvisor.dev/docs/) | 说明 gVisor 在用户态充当应用内核，拦截并实现系统调用接口，同时存在兼容性和性能权衡 | 用于解释 gVisor 类隔离层的机制 |
| [Docker seccomp 官方文档](https://docs.docker.com/engine/security/seccomp/) | 说明容器可通过 `seccomp` 限制 syscall，默认 profile 采用 allowlist 思路 | 用于说明容器侧 syscall 过滤的常见做法 |
| [Firecracker 官方仓库](https://github.com/firecracker-microvm/firecracker) | 说明 microVM 面向安全多租户、低开销工作负载，强调最小化设备模型和攻击面 | 用于说明 microVM 的基本定位 |
| [E2B Sandbox 文档](https://e2b.dev/docs/sandbox) | 说明 sandbox 生命周期、可配置 timeout，以及长会话上限 | 作为 E2B 会话时长边界的公开依据 |
| [E2B AI Agents 页面](https://e2b.dev/ai-agents/rebyte) | 明确说明沙箱由 Firecracker microVM 驱动 | 作为 E2B 使用 microVM 的公开依据 |
| [Anthropic SDK 30 秒超时 issue](https://github.com/anthropics/claude-agent-sdk-typescript/issues/42) | 展示工具层 30 秒硬编码超时的公开报告 | 用于说明“工具超时 != 作业系统”的工程坑 |
| [Claude Code timeout 相关 issue](https://github.com/anthropics/claude-code/issues/2492) | 讨论 Bash 默认/最大超时环境变量与不同命令需要不同超时的现实问题 | 用于说明超时配置的复杂性 |
