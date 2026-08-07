---
title: fork、vfork 与 clone 的区别
date: 2026-08-07
---

# fork、vfork 与 clone 的区别

<div class="epigraph">
<p>创建进程有四种姿势：全复制、半借用、按需共享、干脆自己造——fork、vfork、clone 是它们的名字。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》§3 与 Linux 手册 ｜ 2026-08-07</p>
</div>

## 为什么从 fork/vfork/clone 开始

Linux 创建新执行流的系统调用有三个：**fork**（复制进程）、**vfork**（半借用）、**clone**（按需共享）。它们的区别本质是「**父子之间共享什么**」——这正是「进程 vs 线程」的内核分界线。这一节把三者讲透，并揭示 Linux 统一创建机制的优雅设计。<span class="marginnote">回顾《进程创建》：fork 创建进程、exec 换程序。Linux 把「创建」拆得更细——<strong>fork/clone 决定「父子共享什么」，exec 决定「运行什么」</strong>。线程（pthread_create）底层也是 clone，只是共享了内存与文件。</span>

## 1 fork：复制一切（写时复制）

**fork**：创建子进程，**复制父进程的几乎一切**——内存、文件表、信号、环境。

```c
pid_t pid = fork();
```

**fork 的语义**：

- **内存**：写时复制（COW，回顾《写时复制》）——初始共享，写时才复制。
- **文件表**：复制（fd 表复制，但打开文件表项共享——`dup` 语义）。
- **父子关系**：完全独立，各自执行。

**fork 的特点**：**最大程度复制 + COW 优化**——子进程是父进程的「近似克隆」，但共享底层物理页直到写。

## 2 vfork：借父进程的内存

**vfork**：创建子进程，但**不复制内存——子进程直接使用父进程的地址空间**，且**父进程阻塞**直到子进程 exec/exit。

```c
pid_t pid = vfork();
```

**vfork 的语义**：

- **共享地址空间**：子进程用父进程的内存（无 COW，真共享）。
- **父进程阻塞**：vfork 后父进程挂起，等子进程 exec 或 exit 才恢复。
- **子进程必须立即 exec/exit**：不能修改父进程的变量再返回——否则破坏父进程状态。

**为什么存在**：**历史遗留优化**——早期 fork 没有 COW，复制整个地址空间很贵。vfork 假设「子进程马上 exec」——既然 exec 会替换地址空间，**复制纯属浪费**，干脆借用父进程的。

**现代地位**：**有了 COW，fork 已经足够便宜**——vfork 的「省复制」意义消失，且它的「共享地址空间 + 父阻塞」语义非常危险（子进程改父进程内存）。**现代代码几乎不用 vfork**（除非极特殊情况），教科书仍保留它以理解历史。

## 3 clone：按需共享（线程的基础）

**clone**：创建新执行流，**精确指定父子共享什么**——通过 flags 参数控制。

```c
int clone(int (*fn)(void *), void *stack, int flags, void *arg);
```

**clone 的 flags**（共享什么由位标志决定）：

| flag | 含义 |
| --- | --- |
| `CLONE_VM` | 共享地址空间（线程的关键） |
| `CLONE_FS` | 共享文件系统信息（cwd、umask） |
| `CLONE_FILES` | 共享打开文件表 |
| `CLONE_SIGHAND` | 共享信号处理表 |
| `CLONE_THREAD` | 成为同一线程组的线程 |

**clone 是万能创建器**：

- `fork()` = `clone(..., SIGCHLD, ...)`（不共享内存）——**进程**。
- `pthread_create()` = `clone(..., CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD, ...)`——**线程**（共享一切）。

**核心洞察**：**Linux 没有独立的「线程创建系统调用」——pthread_create 底层就是 clone**。**「进程 vs 线程」不是内核的两种东西，而是 clone 的 flags 不同**——共享内存 = 线程，不共享 = 进程。

**辨析｜易错点：** 「fork 创建进程、clone 创建线程」是简化但不准确。**clone 既能创建进程也能创建线程——取决于 flags**。fork 只是 clone 的一个「默认参数」版本。**「进程/线程」是用户视角的分类，内核视角只有「带不同共享选项的 task_struct」**（回顾《task_struct》的 mm/files 共享）。

## 4 核心对比表：三种创建方式

| 维度 | fork | vfork | clone |
| --- | --- | --- | --- |
| 内存 | COW 复制 | 共享（父阻塞） | 由 flags 决定 |
| 父进程 | 不阻塞 | **阻塞** | 不阻塞 |
| 是否 exec | 可自由选择 | **必须立即 exec/exit** | 可自由选择 |
| 子进程身份 | 独立进程 | 独立进程 | 进程或线程 |
| 现代使用 | **标准** | 几乎不用 | **线程/进程通用** |

**设计启示**：fork/vfork/clone 的演化是「**从固定语义到可配置语义**」——fork 固定「复制」，clone 让「共享什么」可配置。**Linux 用一个 clone 统一了进程与线程**，这是「少而通用的原语」胜过「多而专用的原语」的典范。

## 5 小结

- **fork**：COW 复制几乎一切，子进程独立，标准进程创建。
- **vfork**：共享父内存 + 父阻塞，子进程必须立即 exec——历史优化，COW 后几乎淘汰。
- **clone**：用 flags 精确控制共享什么——`CLONE_VM` 共享内存 = 线程。
- **pthread_create 底层是 clone**——「进程 vs 线程」只是 clone flags 的差异。
- Linux 没有独立的线程创建调用，一个 clone 统一两者。

在下一节，我们看进程「换程序」的机制——**exec 家族与程序加载：ELF 格式解析**。
