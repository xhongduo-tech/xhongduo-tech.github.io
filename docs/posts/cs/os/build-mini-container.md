---
title: 手写一个简易容器：Namespace + Cgroup + rootfs
date: 2026-08-07
---

# 手写一个简易容器：Namespace + Cgroup + rootfs

<div class="epigraph">
<p>「容器不过是一个被隔离的进程」——这一节，我们用几段 C 代码证明这句话。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》与容器实战 ｜ 2026-08-07</p>
</div>

## 为什么从手写容器开始

理论讲完了 Namespace、cgroup、rootfs——现在把它们**拼成一个真的容器**。这一节用 C 代码实现一个**最小容器**：用 `clone` 隔离视图、用 `pivot_root` 切换根、用 cgroup 限制资源。做完这一步，你会彻底理解「容器 = 带隔离和限制的进程」。<span class="marginnote">回顾容器三件套：Namespace（视图）、cgroup（配额）、rootfs（根）。手写容器的过程就是<strong>逐个调用这些系统调用</strong>——`clone`（Namespace）、`pivot_root`（rootfs）、写 cgroup 文件（配额）。<strong>没有魔法，都是系统调用。</strong></span>

## 1 第一步：clone 创建隔离进程

用 `clone` 带 Namespace 标志创建子进程——子进程进入独立视图：

```c
#define STACK_SIZE (1024 * 1024)
static char child_stack[STACK_SIZE];

static int child_fn(void *arg) {
    printf("container: PID=%d\n", getpid());  // 容器内 PID 从 1 开始
    execvp((char *)arg, (char *[]){arg, NULL}); // 执行 /bin/sh
    return 0;
}

int main(int argc, char **argv) {
    pid_t pid = clone(child_fn, child_stack + STACK_SIZE,
        SIGCHLD | CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC,
        argv[1]);   // 例如 /bin/sh
    waitpid(pid, NULL, 0);
    return 0;
}
```

**发生了什么**：

- `CLONE_NEWPID`：子进程进入新的 **PID Namespace**——`getpid()` 返回 1（回顾《Namespace》）。
- `CLONE_NEWNS`：新的 **Mount Namespace**——之后可以自由挂载。
- `CLONE_NEWUTS` / `CLONE_NEWIPC`：隔离主机名与 IPC。
- `execvp`：执行 `/bin/sh`——**容器里的 shell**。

## 2 第二步：pivot_root 切换根（rootfs）

没有 rootfs 的「容器」还看到宿主的 `/`。用 `pivot_root` 把根切到自己的目录：

```c
// 在 child_fn 里：
static void setup_rootfs(const char *rootfs) {
    chdir(rootfs);                                  // 进入新根目录
    mkdir(".oldroot", 0700);
    syscall(SYS_pivot_root, ".", ".oldroot");       // 切换根：. 是新根
    chdir("/");
    rmdir("/.oldroot");                             // 移除旧根挂载
}
```

**发生了什么**：

- `rootfs` 指向一个**最小根文件系统目录**（如 busybox 提供的 `/bin/sh` 环境）。
- `pivot_root` 把**当前进程的根**换成 rootfs——容器的 `/` 变成自己的世界。
- 之后 `/bin/sh` 的路径、库、命令都从 rootfs 里解析——**与宿主文件系统隔离**。

**准备 rootfs**：用 **busybox** 做一个极简根：

```bash
mkdir /tmp/myroot
busybox --install /tmp/myroot/bin
mkdir -p /tmp/myroot/proc /tmp/myroot/dev
```

## 3 第三步：挂载 proc 与 cgroup 限制

**挂载 proc**：容器内 `ps` 需要 `/proc`（看自己 Namespace 的进程）：

```c
mount("proc", "/proc", "proc", 0, NULL);
mount("tmpfs", "/dev", "tmpfs", 0, NULL);
```

**配置 cgroup（限制 CPU/内存）**——写 cgroup 文件：

```c
static void setup_cgroup(void) {
    // v2 统一层级：把容器进程加入组并设限额
    FILE *f = fopen("/sys/fs/cgroup/mycontainer/cgroup.procs", "w");
    fprintf(f, "%d\n", getpid());          // 加入组
    fclose(f);

    f = fopen("/sys/fs/cgroup/mycontainer/cpu.max", "w");
    fprintf(f, "50000 100000\n");          // 半核限额（回顾 cgroup）
    fclose(f);

    f = fopen("/sys/fs/cgroup/mycontainer/memory.max", "w");
    fprintf(f, "1073741824\n");            // 1GB 内存上限
    fclose(f);
}
```

**公式解析：cgroup 限制的写入**

$$\text{容器 CPU} = \frac{50000}{100000} = 0.5 \text{ 核}, \qquad \text{内存} = 1\text{GB}$$

- `cpu.max` 写 `"quota period"`——每 100ms 最多 50ms = 半核（回顾《cgroup》）。
- `memory.max` 写字节数——超出触发回收/OOM。
- **写文件 = 配置内核**——cgroup 的控制面就是 `/sys/fs/cgroup` 下的文件。

**直觉**：**cgroup 配置 = 写文件**——内核把「控制接口」暴露成文件系统，用户通过写文件设置限额。这是「**一切皆文件**」哲学的又一次体现（回顾《文件概念》）。

**辨析｜易错点：** 「手写容器 = Docker」是过度类比。**这个迷你容器缺少 Docker 的工程化部分**——镜像分层（OverlayFS）、网络（veth/网桥/NAT）、OCI 规范、进程守护、安全加固（Capability/Seccomp）。**它证明了「容器的核心机制很简单」，但 Docker 的价值在「把简单机制工程化成可用产品」。**

## 4 完整流程与验证

```
编译运行：
  gcc mini_container.c -o mini
  ./mini /bin/sh

验证：
  # 容器内
  ps -ef        → 只看到容器内进程（PID 1 是 sh）
  hostname      → 隔离的主机名
  cat /proc/meminfo → 被 cgroup 限制的内存视图
```

**每步对应一个机制**：

| 步骤 | 系统调用/机制 | 隔离/限制 |
| --- | --- | --- |
| clone 带标志 | `CLONE_NEWPID/NEWNS/...` | 视图隔离 |
| pivot_root | `pivot_root` | 根文件系统 |
| mount proc | `mount` | /proc 视图 |
| 写 cgroup | `/sys/fs/cgroup/*` | CPU/内存限额 |

## 5 核心对比表：迷你容器 vs Docker

| 维度 | 迷你容器 | Docker |
| --- | --- | --- |
| 隔离 | clone 标志 | 同 + 网络（veth/网桥） |
| rootfs | 手动 busybox | OverlayFS 镜像分层 |
| 资源限制 | 手写 cgroup | 自动 cgroup 配置 |
| 镜像分发 | 无 | Docker Hub/OCI 仓库 |
| 安全加固 | 无 | Capability + Seccomp |

## 6 小结

- **手写容器三步**：`clone`（Namespace 隔离）→ `pivot_root`（rootfs）→ 写 cgroup（资源限制）。
- `CLONE_NEWPID` 让容器内 PID 从 1 开始；`pivot_root` 把根切到自己的目录。
- **cgroup 配置 = 写 `/sys/fs/cgroup` 下的文件**——一切皆文件。
- 迷你容器证明了「**容器 = 隔离 + 限制的进程**」——没有魔法。
- Docker 的价值在工程化：镜像分层、网络、安全加固、OCI 标准。

在下一节，我们补上容器的安全短板——**容器安全：Capability、Seccomp 与 AppArmor**。
