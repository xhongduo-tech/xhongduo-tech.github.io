## 核心结论

容器运行时安全不是“开一个开关”就够，而是把不同攻击面拆开限制。**Seccomp** 是“系统调用过滤器”，白话说就是“进程想向内核申请什么能力，先过白名单”；**AppArmor** 是“路径与能力访问控制”，白话说就是“即使程序已经在跑，也只能碰允许的文件和内核接口”；**非 root 容器** 是“不要让进程默认拿管理员身份”；`--read-only` 与 `--tmpfs` 则进一步限制“写哪里、能不能持久化”。

这几层配合的价值在于：攻击者通常不会只走一条路径。容器逃逸、横向移动、持久化植入，往往同时依赖“危险 syscall + 敏感路径写入 + 高权限身份”。如果只做其中一层，另一层仍可能被利用。更稳妥的做法是：

| 机制 | 控制对象 | 默认思路 | 主要拦截什么 |
|---|---|---|---|
| Seccomp | 系统调用 | 默认拒绝，按需放行 | 危险内核接口，如 `add_key`、部分 `clone` 场景 |
| AppArmor | 文件路径与能力 | 默认模板约束 | 写敏感路径、访问危险 `/proc` 节点 |
| 非 root | 进程身份 | 不给 UID 0 | 降低写系统目录、改权限、绑定特权设备的能力 |
| 只读根文件系统 | 镜像层写入 | 默认不可写 | 篡改 `/etc`、落地后门 |
| tmpfs | 临时可写目录 | 仅内存存在 | 让程序能写缓存，但不能持久化 |

一个新手最该记住的结论是：`docker-default` 之类的默认安全配置并不等于“绝对安全”，但它们已经能显著缩小攻击面。对大多数业务容器，`默认 Seccomp + 默认 AppArmor + 非 root + --read-only + --tmpfs /tmp` 是一个合理起点。

---

## 问题定义与边界

本文讨论的是**运行时安全**，不是镜像供应链安全，也不是集群网络隔离。问题可以写成一句话：

> 当容器内进程被恶意输入触发，怎样限制它对主机内核、文件系统和设备的可操作面，同时尽量不破坏业务可用性？

攻击链通常长这样：

```text
恶意输入
  -> 进程执行异常代码
  -> 调用高风险 syscall 申请更多内核能力
  -> 写入 /etc、/root、应用数据目录或设备节点
  -> 留下持久化 payload
  -> 等待重启后继续生效
```

对应的防御层也正好分层：

```text
Seccomp: 限制“能向内核发什么请求”
AppArmor: 限制“能访问什么路径和内核暴露面”
non-root: 限制“以谁的身份访问”
read-only + tmpfs: 限制“能写哪里、能否持久化”
```

这里有几个边界必须先说清：

| 边界 | 说明 | 为什么重要 |
|---|---|---|
| syscall 白名单不是万能的 | 某些正常程序依赖大量基础 syscall，如 `openat`、`mmap`、`futex` | 误删会直接把应用打挂 |
| AppArmor 主要看路径与能力 | 它擅长限制 `/proc`、`/sys`、文件读写，不直接替代 syscall 过滤 | 两者职责不同，不能互相替代 |
| 非 root 不等于没风险 | 进程仍可读业务密钥、打本容器内服务、发网络请求 | 只能降权，不能清空风险 |
| GPU/设备访问是特殊场景 | `/dev/nvidia*`、驱动库、组权限都可能影响可用性 | 安全收紧后最容易先在这里出问题 |

“玩具例子”可以这样理解。假设你有两个容器：

1. 容器 A：默认 root，无 Seccomp，自由写根文件系统。
2. 容器 B：默认 profile、非 root、`--read-only --tmpfs /tmp`。

如果恶意程序尝试调用 `add_key`，容器 A 可能继续往下走；容器 B 很可能直接被 Seccomp 返回 `EPERM`。如果恶意程序再尝试改 `/etc/passwd`，容器 B 还会遇到只读文件系统或权限不足。也就是说，第二种不是“不会被打”，而是“被打后更难扩大影响”。

---

## 核心机制与推导

先看 Seccomp。**系统调用**就是“用户态程序向内核申请服务的入口”。容器进程最终能做什么，很多时候取决于它还能调用哪些 syscall。

Docker 默认 Seccomp profile 的核心逻辑可以抽象成：

$$
\text{result}(s)=
\begin{cases}
\text{ALLOW}, & s \in W \\
\text{ERRNO(EPERM)}, & s \notin W
\end{cases}
$$

其中 $W$ 是允许的 syscall 集合。也就是 `defaultAction = SCMP_ACT_ERRNO`，白名单里的规则再单独设成 `SCMP_ACT_ALLOW`。

一个简化后的 profile 结构长这样：

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "exit", "futex", "openat", "mmap"],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["add_key", "keyctl", "acct"],
      "action": "SCMP_ACT_ERRNO"
    }
  ]
}
```

这里 `openat` 必须特别理解。它是“按目录上下文打开文件”的系统调用，白话说就是现代 Linux 程序几乎天天都会用它读库文件、配置文件、设备节点。很多人第一次精简 profile 时喜欢把它删掉，结果不是“更安全”，而是“容器根本起不来”，GPU 驱动加载、动态链接库读取、普通日志文件打开都会受影响。

再看 AppArmor。它是 Linux 的一个 **LSM（Linux Security Module，Linux 安全模块）**，白话说就是“内核里负责额外做访问判定的一层”。和 Seccomp 不同，它更关心“这个进程能不能读写某条路径、访问某类资源”。

典型规则片段如下：

```text
#include <abstractions/base>

file,
deny @{PROC}/sysrq-trigger rwklx,
deny @{PROC}/mem rwklx,
deny /sys/firmware/** rwklx,
```

这里 `deny @{PROC}/sysrq-trigger rwklx` 的意思是：不允许容器进程读、写、锁、链接或执行这个 `/proc` 下的敏感接口。`/proc/sysrq-trigger` 本质上是内核控制入口之一，不该让普通容器碰。Seccomp 挡的是“你能不能发起某类内核请求”，AppArmor 挡的是“你能不能摸到某个危险入口”。两层不是重复，而是互补。

只读根文件系统与 tmpfs 的推导也很直接。假设攻击者想落地一个持久化文件：

$$
\text{持久化成功} = \text{有写权限} \land \text{目标路径可持久保存}
$$

加上 `--read-only` 后，根文件系统不满足“可写”；把 `/tmp` 这类必须可写目录改成 `tmpfs` 后，即使满足“可写”，也不满足“可持久保存”，因为内存文件系统在容器停止后就消失。

三种默认行为可以并排看：

| 机制 | 默认拒绝对象 | 允许条件 | 拒绝后现象 |
|---|---|---|---|
| Seccomp | 未列入白名单的 syscall | profile 明确允许 | 常见为 `EPERM` 或功能异常 |
| AppArmor | 未被规则允许的路径/能力 | profile 允许 | 文件访问失败、权限错误 |
| Read-only rootfs | 根文件系统写入 | 明确挂载可写卷或 tmpfs | 程序报只读文件系统 |

---

## 代码实现

先给一个“玩具例子”的运行方式。目标不是做最严 profile，而是让初学者看到“同一容器，被加固前后行为差异”。

```bash
docker run --rm -it \
  --security-opt seccomp=/path/to/seccomp.json \
  --security-opt apparmor=docker-default \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp:rw,exec,nosuid,size=64m \
  --tmpfs /run:rw,nosuid,size=16m \
  python:3.12-slim bash
```

每一项的作用如下：

| 参数 | 作用 | 典型收益 |
|---|---|---|
| `--security-opt seccomp=...` | 挂载自定义 syscall 白名单 | 阻断危险 syscall |
| `--security-opt apparmor=docker-default` | 启用默认 AppArmor 模板 | 限制敏感路径访问 |
| `--user 1000:1000` | 非 root 运行 | 减少系统级写权限 |
| `--read-only` | 根文件系统只读 | 阻断持久化改写 |
| `--tmpfs /tmp` | 给应用一个内存型临时目录 | 兼顾可用性与不落盘 |
| `--tmpfs /run` | 允许部分运行时状态写入 | 避免某些守护进程启动失败 |

如果是 Kubernetes，思路相同，只是入口从 CLI 变成 Pod 字段：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hardened-app
  annotations:
    container.apparmor.security.beta.kubernetes.io/app: runtime/default
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: python:3.12-slim
      command: ["python", "-c", "print('ok')"]
      securityContext:
        readOnlyRootFilesystem: true
        allowPrivilegeEscalation: false
      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: run
          mountPath: /run
  volumes:
    - name: tmp
      emptyDir:
        medium: Memory
    - name: run
      emptyDir:
        medium: Memory
```

下面这个 Python “可运行玩具代码”不是直接操作内核，而是把上述判定逻辑抽象成一个最小模型，帮助理解“多层约束是如何叠加的”。

```python
from dataclasses import dataclass

@dataclass
class Policy:
    allowed_syscalls: set
    denied_paths: set
    uid: int
    read_only_rootfs: bool
    writable_tmpfs: set

def can_do(policy: Policy, syscall: str, path: str | None = None) -> bool:
    if syscall not in policy.allowed_syscalls:
        return False

    if path is None:
        return True

    for denied in policy.denied_paths:
        if path.startswith(denied):
            return False

    if policy.read_only_rootfs and path.startswith("/") and not any(
        path.startswith(prefix) for prefix in policy.writable_tmpfs
    ):
        return False

    if policy.uid != 0 and path in {"/etc/passwd", "/etc/shadow"}:
        return False

    return True

policy = Policy(
    allowed_syscalls={"read", "write", "openat", "futex"},
    denied_paths={"/proc/sysrq-trigger", "/sys/firmware"},
    uid=1000,
    read_only_rootfs=True,
    writable_tmpfs={"/tmp", "/run"},
)

assert can_do(policy, "openat", "/tmp/app.log") is True
assert can_do(policy, "add_key") is False
assert can_do(policy, "openat", "/etc/passwd") is False
assert can_do(policy, "openat", "/proc/sysrq-trigger") is False
print("policy model ok")
```

“真实工程例子”则常见于 GPU 推理或训练容器。你可能需要：

```bash
docker run --rm -it \
  --gpus all \
  --user 1000:1000 \
  --group-add video \
  --security-opt apparmor=docker-default \
  --security-opt seccomp=/path/to/seccomp.json \
  --read-only \
  --tmpfs /tmp:rw,exec,size=128m \
  nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

这里的难点不是“GPU 能不能看见”，而是“非 root + 安全限制后，还能不能合法访问 `/dev/nvidia*` 和驱动库”。这正是容器安全在工程上最容易出摩擦的地方。

---

## 工程权衡与常见坑

最常见的误区是把“最小权限”理解成“删得越多越安全”。实际工程里，安全配置必须和程序调用面一起看。

| 坑 | 症状 | 根因 | 解决 |
|---|---|---|---|
| Seccomp 过度精简 | 程序启动即报权限错误 | 把 `openat`、`mmap`、`futex` 等基础 syscall 删掉了 | 从官方默认 profile 起步，逐项收缩 |
| 只开 `--read-only` | 程序写日志、建 socket、缓存时报错 | `/tmp`、`/run` 不可写 | 补 `--tmpfs /tmp`、`--tmpfs /run` |
| 非 root 跑 GPU 失败 | `Failed to initialize NVML: Insufficient Permissions` | `/dev/nvidia*` 权限或组不匹配，或安全标签拦截 | 增加 `video` 组、校准设备权限、检查平台安全策略 |
| AppArmor 自定义过严 | 容器能启动但访问 `/proc`、设备或库异常 | 路径规则覆盖了必需访问 | 先用 `docker-default`，再增量修改 |
| 以为 non-root 足够 | 容器仍可读业务配置、访问下游服务 | 身份降权不等于数据隔离 | 联合网络、密钥、卷权限策略一起做 |

一个典型“真实工程坑”是 GPU 场景。很多团队会先把容器改成非 root，再打开 `RuntimeDefault` Seccomp，结果 `nvidia-smi` 直接失败。表面现象像是 GPU 驱动坏了，实际上可能是三类问题之一：

1. `openat` 或相关 syscall 被你自定义 profile 误伤，导致驱动库打不开。
2. `/dev/nvidia*` 属于 `video` 组，而容器用户没被加入对应组。
3. 某些平台额外的安全标签机制阻断了设备访问。

另一个高频坑是只读根文件系统。很多应用会把 PID 文件、Unix socket、临时缓存、编译中间文件写到 `/tmp` 或 `/run`。你一旦只开 `--read-only` 却不补 tmpfs，应用就会在完全“正常”的业务路径里崩掉。这个问题看起来像程序 bug，本质上是运行时目录假设没被满足。

一个实用调试顺序是：

```text
先用默认 profile 跑通
-> 开 non-root
-> 开 read-only
-> 给 /tmp、/run 补 tmpfs
-> 再逐步精简 seccomp / apparmor
-> 每次只改一层，并记录失败 syscall 或路径
```

---

## 替代方案与适用边界

如果业务要求更高，或者运行平台不是 Docker 默认栈，还可以考虑其他方案。

| 方案 | 适用场景 | 优点 | 代价/边界 |
|---|---|---|---|
| 只用 Seccomp | 先快速缩 syscall 面 | 配置直观，收益明显 | 不能控制文件路径 |
| Seccomp + AppArmor | 通用 Linux 容器 | syscall 与路径双重限制 | 需要理解两套规则 |
| Seccomp + SELinux + tmpfs | 高合规、多租户环境 | 标签隔离更强 | 规则复杂，平台相关性更强 |
| user namespaces | 强化 UID 映射隔离 | 容器内 root 映射到宿主非特权 UID | 与卷、设备、老应用兼容性需验证 |
| eBPF 观测/策略 | 要先观测真实调用面 | 可做更细粒度分析 | 学习与运维成本高 |

可以按场景粗分：

| 场景 | 推荐组合 |
|---|---|
| 纯 CPU Web 服务 | `RuntimeDefault seccomp + 默认 AppArmor + non-root + read-only + tmpfs` |
| GPU 推理/训练 | 在上面基础上增加设备权限与组映射，谨慎调整 syscall |
| 高合规多租户 | 再叠加 SELinux、user namespaces、设备最小暴露 |
| 老旧单体应用 | 先做 non-root 和只读根文件系统，再逐步引入 seccomp/app profile |

对初级工程师，一个务实判断标准是：

1. 如果你还不知道程序到底依赖哪些 syscall，不要从“全自定义 Seccomp”开始。
2. 如果你的容器要碰设备、驱动、FUSE、GPU，不要假设默认最小权限一定能直接跑通。
3. 如果你只能做一件事，优先做“非 root + 只读根文件系统 + 必需 tmpfs”；这是收益高、理解成本相对低的一步。
4. 如果你能做两件事，再把 `RuntimeDefault` Seccomp 和默认 AppArmor 加上，形成完整基线。

---

## 参考资料

| 资源 | 覆盖面 | 适用问题 |
|---|---|---|
| Docker Seccomp 文档: https://docs.docker.com/engine/security/seccomp/ | 默认 profile、`SCMP_ACT_ERRNO`、允许/拒绝模型 | 想理解 Docker 默认 syscall 白名单时看 |
| Docker AppArmor 文档: https://docs.docker.com/engine/security/apparmor/ | `docker-default`、规则语法、敏感路径限制 | 想理解文件路径与能力控制时看 |
| NVIDIA Container Toolkit Troubleshooting: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html | GPU 容器、非 root、设备权限、SELinux 相关问题 | `nvidia-smi` 或 NVML 权限异常时看 |
| Docker 运行时加固实践说明: https://williamzujkowski.github.io/posts/docker-runtime-security-hardening-with-linux-security-modules/ | `--read-only`、`--tmpfs`、LSM 组合思路 | 想补齐只读根文件系统与 tmpfs 边界时看 |
