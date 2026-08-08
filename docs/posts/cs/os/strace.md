---
title: strace：跟踪进程的系统调用
date: 2026-08-07
---

# strace：跟踪进程的系统调用

<div class="epigraph">
<p>想知道一个程序「到底在做什么」，最好的办法是看它给内核打了哪些电话——strace 就是那部窃听电话。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 恐龙书 §2.3 与 Linux 工具链 ｜ 2026-08-07</p>
</div>

## 为什么从 strace 开始

理论讲完了系统调用，这一节拿起**真实工具**——**strace**：它拦截并打印进程的所有系统调用，是 Linux 排障的「第一工具」。调试「程序为什么打不开文件」「为什么慢」「为什么卡住」，strace 一查便知。这一节看 strace 的原理（ptrace）与实战用法。<span class="marginnote">回顾《系统调用原理》：程序的一切系统级操作都走系统调用。strace 的原理就是「观察这些系统调用」——它靠 <strong>ptrace</strong> 系统调用接管目标进程，在每次系统调用进出时记录。<strong>strace 本身也是一个「系统调用使用者」，只是它的用途是观察别人。</strong></span>

## 1 strace 是什么

**strace**：Linux 的**系统调用跟踪器**——运行并跟踪一个进程，打印它发出的每一个系统调用及其参数、返回值。

```
$ strace ./hello
```

**输出示例**：

```
execve("./hello", ["./hello"], 0x7ffd...) = 0
openat(AT_FDCWD, "/etc/passwd", O_RDONLY) = 3
read(3, "root:x:0:0:root:/root:/bin/bash\n", 4096) = 33
...
write(1, "hello\n", 6) = 6
close(1)                              = 0
exit_group(0)                         = 0
```

- openat：打开文件返回 fd 3。
- write：往标准输出写 6 字节。

**strace 的价值**：**看清程序「真正做了什么」**——不靠猜，靠记录。

## 2 strace 的原理：ptrace

**strace 的实现基于 ptrace 系统调用**：

**ptrace**：父进程可**观察和控制**子进程的执行——包括在每次系统调用进入/退出时暂停子进程、读取其寄存器。

**strace 的工作流程**：

1. fork 一个子进程（或附加到目标进程）。
2. 用 ptrace(PTRACE_TRACEME) 让子进程被跟踪。
3. 子进程每次**进入系统调用**时被暂停，strace 读取其寄存器（调用号、参数）。
4. 子进程**从系统调用返回**时再次被暂停，strace 读取返回值。
5. 打印并继续。

**ptrace 的其他用途**：

- **调试器**（gdb）——断点、单步、读写内存。
- **代码注入**——ptrace(PTRACE_POKEDATA) 写目标进程内存。
- **沙箱/安全工具**——跟踪进程行为（回顾容器安全的 Seccomp 也借 ptrace 家族）。

**辨析｜易错点：** 「strace 无开销」是误解。**ptrace 会让进程在每次系统调用进出时停顿两次**——性能显著下降（可达 10~100 倍减速）。**strace 是诊断工具，不是日常运行工具**；生产环境排查完要立刻卸下（strace -p 附加后 Ctrl-C）。

## 3 strace 的实战场景

**场景一：程序打不开文件，为什么？**

```
$ strace -e trace=open,openat ./app
openat(AT_FDCWD, "/etc/missing.conf", O_RDONLY) = -1 ENOENT (No such file or directory)
```

**场景二：程序启动慢，卡在哪？**

```
$ strace -c ./app      # 汇总每个系统调用的耗时
% time  seconds  usecs/call  calls  errors  syscall
------ ------- ----------- ------ ------- ----------------
92.31  1.23456     12345    100        0  read
 4.10  0.05432       543    100        0  write
...
```

**场景三：程序访问了什么文件/网络？**

```
$ strace -e trace=open,openat,connect ./app
openat(AT_FDCWD, "/etc/resolv.conf", O_RDONLY) = 3
socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) = 4
connect(4, {sa_family=AF_INET, sin_port=htons(53), ...}, 16) = 0
```

**场景四：权限问题**

```
$ strace ./app 2>&1 | grep -i denied
openat(AT_FDCWD, "/var/log/app.log", O_WRONLY) = -1 EACCES (Permission denied)
```

**核心思路**：**strace 把「程序做了什么」变成「可见的日志」**——所有系统级行为（文件、网络、进程、内存）都无处遁形。

## 4 公式解析：strace 输出怎么读

strace 的一行输出遵循固定格式：

$$\underbrace{\text{系统调用名}}_{\text{openat}}(\underbrace{\text{参数}}_{\text{AT_FDCWD, "/etc/x", O_RDONLY}}) = \underbrace{\text{返回值}}_{\text{3}}$$

- **系统调用名**：内核函数对应的用户接口（openat、read、write…）。
- **参数**：实参的格式化展示（路径、标志、长度）。
- **返回值**：成功时非负（fd 号、字节数）；失败时 -1（如 ENOENT = 文件不存在）。

**失败读法**：-1 → 错误码 EACCES（权限被拒）。**errno 名字（回顾《系统调用表》的负值约定）是判断失败原因的关键**——ENOENT 不存在、EACCES 无权限、EAGAIN 稍后再试。

**直觉**：strace 输出 = 「系统调用的流水账」——**每行一个调用、参数、结果**。会读 errno，就会读 strace。

## 5 核心对比表：strace 常见用例

| 问题 | strace 命令 | 看什么 |
| --- | --- | --- |
| 文件打不开 | strace -e trace=open,openat ./app | 返回 ENOENT/EACCES |
| 启动慢 | strace -c ./app | 耗时最多的调用 |
| 访问了什么 | strace -e trace=open,connect ./app | open/connect 的目标 |
| 卡住不动 | strace -p `<PID>` | 阻塞在哪个调用 |

## 6 小结

- **strace** 跟踪并打印进程的全部系统调用（调用名、参数、返回值）。
- 原理：**ptrace** 在每次系统调用进出时暂停进程、读取寄存器。
- ptrace 也是 gdb、沙箱、安全工具的基础。
- 实战四问：**文件为什么打不开、启动为什么慢、访问了什么、卡在哪**。
- 读输出 = 读系统调用名 + 参数 + 返回值（errno）；strace 有显著开销，用完即卸。

在下一节，我们速查最常用的系统调用——**常见系统调用速查：open/read/write/fork/exec/wait**。
