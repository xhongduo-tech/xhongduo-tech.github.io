---
title: Linux 进程描述符 task_struct 详解
date: 2026-08-07
---

# Linux 进程描述符 task_struct 详解

<div class="epigraph">
<p>一个进程在 Linux 内核里是什么？——是一棵巨大结构体的树根，所有关于它的信息都挂在上面。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》§3 与 OSTEP ｜ 2026-08-07</p>
</div>

## 为什么从 task_struct 开始

理论上的 **PCB（进程控制块）** 在 Linux 里就是 **task_struct**——内核为每个进程维护的巨大数据结构。它是进程的「全部家当」：状态、PID、内存、文件、信号、调度信息、父子关系。理解 task_struct，等于把前面「PCB、状态、调度、内存、文件」所有概念在 Linux 上「对号入座」。<span class="marginnote">回顾《进程控制块》：PCB 分「标识、处理器状态、调度信息、内存与资源」四组。task_struct 就是 Linux 版 PCB 的完整实现——它比教材里的 PCB 大得多（几百个字段），但骨架完全对应。<strong>理论 PCB 是「清单」，task_struct 是「实物」。</strong></span>

## 1 task_struct 的位置与组成

**task_struct**：`include/linux/sched.h` 中定义，Linux 每个进程（线程也是）一个。它包含进程的全部信息，主要分组：

| 分组 | 字段示例 |
| --- | --- |
| 标识 | `pid`（进程号）、`tgid`（线程组号） |
| 状态 | `state`（TASK_RUNNING/INTERRUPTIBLE...） |
| 调度 | `prio`（优先级）、`vruntime`、`sched_class` |
| 内存 | `mm`（内存描述符指针）、`active_mm` |
| 文件 | `files`（打开文件表指针）、`fs`（fs_struct） |
| 信号 | `signal`、`sighand`（信号处理表） |
| 父子 | `parent`、`children`、`real_parent` |
| 线程 | `thread`（线程专有状态）、`thread_info` |

**task_struct 的组织**：内核用**双向链表**把 task_struct 串起来——`for_each_process` 遍历所有进程；用**红黑树**按 PID 快速查找（`find_task_by_pid`）。

## 2 关键字段解读

**① 状态 state**：进程当前状态（回顾五状态模型）：

- `TASK_RUNNING`：就绪或运行。
- `TASK_INTERRUPTIBLE`：可中断睡眠（等事件，信号能唤醒）。
- `TASK_UNINTERRUPTIBLE`：不可中断睡眠（等 I/O，信号不唤醒）。
- `TASK_STOPPED`、`TASK_TRACED`：停止/被跟踪。
- `EXIT_ZOMBIE`、`EXIT_DEAD`：僵尸/已死。

**② 标识 pid/tgid**：

- `pid`：每个线程独立的线程号。
- `tgid`：**线程组 ID**——组内所有线程共享 tgid（进程号）。`getpid()` 返回 tgid，`gettid()` 返回 pid。

**③ 内存 mm**：

- `mm` 指向 **mm_struct**（内存描述符）——包含页表、代码/数据/栈段的地址范围、mmap 链表。
- **线程共享 mm**（`mm` 相同），进程独立 mm——这就是「线程共享地址空间、进程不共享」的内核实现。

**④ 文件 files**：

- `files` 指向 **files_struct**——打开文件表（fd 表）。线程共享 files（fd 共享），fork 时复制（COW）。

**辨析｜易错点：** 「pid = 进程号」在**线程语境**下不准确。**内核里 `pid` 是线程号，`tgid` 才是用户看到的进程号（PID）**——一个多线程进程，用户看到 1 个 PID，内核看到 N 个 task_struct（N 个 pid、共享 tgid）。**`ps` 显示的是 tgid；`top` 里每个线程是独立的 pid。**

## 3 task_struct 的管理：链表与红黑树

内核高效管理大量 task_struct：

- **双向链表**：`tasks` 字段把所有进程串成环——遍历（`ps`）用。
- **红黑树（PID 哈希）**：按 pid 快速查找——`kill(pid)`、`getpid` 相关操作 O(log n)。
- **就绪队列（runqueue）**：每个 CPU 一个，用红黑树按 vruntime 组织（回顾 CFS）——调度器 O(log n) 选进程。

**内核如何找「当前进程」**：通过 `current` 宏——从 `thread_info` 或栈指针计算当前 task_struct 地址。`current->pid` 就是「我是谁」。

**公式解析：current 的定位**

```c
#define current get_current()
```

x86-64 上 `current` 通过 **thread_info 的栈指针**计算：

$$\text{current} = \text{内核栈底部} - \text{THREAD_INFO_OFFSET}$$

- 每个进程有独立**内核栈**，栈底附近存 thread_info 指针。
- 从栈指针反推 task_struct 地址——**O(1) 找到「当前进程」**。
- 不用搜索、不用全局变量——**从「我正在用哪个栈」就知道「我是谁」**。

**直觉**：`current` 是内核代码最常用的宏——任何「当前进程」的访问（`current->pid`、`current->mm`）都靠它。**「从栈定身份」是内核的经典技巧**：每个进程的内核栈唯一，栈即身份。

## 4 核心对比表：PCB（理论） vs task_struct（Linux）

| 维度 | PCB（教材） | task_struct（Linux） |
| --- | --- | --- |
| 进程标识 | PID | pid + tgid |
| 状态 | 五状态 | state 位掩码 |
| 调度 | 优先级等 | prio + vruntime + sched_class |
| 内存 | 地址空间 | mm（mm_struct 指针） |
| 文件 | 文件表 | files（files_struct 指针） |
| 组织 | 队列 | 链表 + 红黑树 + runqueue |

## 5 小结

- **task_struct** 是 Linux 的 PCB——进程的全部信息集中于此。
- 关键字段：**pid/tgid、state、mm（内存）、files（文件）、prio/vruntime（调度）**。
- **线程共享 mm 与 files，进程独立**——共享地址空间与 fd 的内核实现。
- 组织：**双向链表**（遍历）+ **红黑树**（按 PID 查）+ **runqueue**（调度）。
- **`current`** 从内核栈定位当前进程——「栈即身份」。

在下一节，我们看创建进程的三种方式——**fork、vfork 与 clone 的区别**。
