---
title: MPI 基础（进程/通信子）
date: 2026-08-07
---

# MPI 基础（进程/通信子）

<div class="epigraph">
<p>MPI 不是一种语言，而是一个协议——它把「并行」讲成每个人都能写的话。</p>
<footer>—— 杰克 · 东加拉（Jack Dongarra），MPI 创始人之一</footer>
</div>

<div class="article-byline">
<p>第三级 · 高性能计算 ｜ 陈国良《并行计算》 第四章 §4.1 ｜ 2026-08-07</p>
</div>

## 为什么从 MPI 开始

前面七篇都在讲「原理」，从这一篇起开始写「代码」。

分布式内存机器上，进程之间靠消息传递协作——这件事需要一套统一、可移植的标准。

它就是 **MPI（Message Passing Interface，消息传递接口）**。

自 1994 年 MPI-1 发布以来，MPI 一直是分布式并行程序的工业标准，几乎每个超算上的天气、材料、流体代码都跑在它上面。

<span class="marginnote">大模型训练的分布式通信库（NCCL、Gloo）在思想上与 MPI 一脉相承：all-reduce、点对点收发这些原语，MPI 三十年前就定义好了。</span>

本节的纲：

先弄懂 MPI 世界里的三个核心概念——**进程、排名（rank）、通信子（communicator）**，再读通第一个 MPI 程序。

## 1 MPI 是什么

**MPI（Message Passing Interface）**是消息传递编程的**标准接口规范**，不是某种具体实现。

- 它规定了函数名、参数、语义，但没有规定实现细节；
- 具体实现有 OpenMPI、MPICH、Intel MPI 等，都是同一套 API。

MPI 解决的关键问题是**可移植性**：

今天在笔记本上写好的并行程序，明天可以拿到百亿亿次超算上编译运行。

MPI 程序的运行形态是一个**程序多份执行**：

用 `mpirun -np 4 ./hello` 启动，同一份可执行文件被复制成 4 个**进程（process）**。

## 2 SPMD：单程序多数据

MPI 程序的基本执行模型叫 **SPMD（Single Program, Multiple Data，单程序多数据）**：

**每个进程运行同一份代码，但处理的数据与执行的路径各不相同。**

进程用 `rank` 区分彼此：

```c
MPI_Init(&argc, &argv);            // 启动 MPI 环境
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 我是谁？
MPI_Comm_size(MPI_COMM_WORLD, &size); // 一共有几个？
if (rank == 0) {
    /* 只有根进程才做的事：读输入、汇总结果 */
} else {
    /* 其他进程做的事：算各自的分片 */
}
MPI_Finalize();                    // 清理 MPI 环境
```

<span class="marginnote">SPMD 与 SIMD 容易混淆：SIMD 是「一条指令同时喂多份数据」，SPMD 是「一份程序复制成多份、各自独立执行」。前者是硬件级，后者是软件级。</span>

**辨析｜易错点：** MPI 程序**没有**「main 只跑一次」的直觉。

`main` 被每个进程各执行一遍，代码里必须靠 `rank` 判断「这段该谁做」。

新手最常见的错误，就是忘了给不同进程分配不同工作，让所有进程重复做同一件事。

## 3 通信子：进程的社交圈

**通信子（communicator）**是 MPI 中「一组可以互相通信的进程」的容器。

它有两个核心作用：

- 给进程一个**命名空间**：进程在这个通信子里的唯一编号叫 **rank**；
- 给通信一个**安全边界**：不同通信子里的消息互相隔离，不会串扰。

MPI 启动时自动建好一个默认通信子 `MPI_COMM_WORLD`，包含全部进程。

你还可以用 `MPI_Comm_split` 把大通信子**切出**小组，比如按颜色分成两队。

**rank 的编号习惯：**

- rank 从 0 开始，最大是 `size - 1`；
- rank 0 常被约定为「根进程」（root），负责读输入、做汇总。

## 4 代码解析：第一个 MPI 程序

把上面几块拼起来，就是一个完整的「Hello, World」：

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello from rank %d of %d\n", rank, size);
    MPI_Finalize();
    return 0;
}
```

用 `mpirun -np 4 ./hello` 运行，输出可能是：

```
Hello from rank 2 of 4
Hello from rank 0 of 4
Hello from rank 3 of 4
Hello from rank 1 of 4
```

三步读懂这个程序：

- **第一步，`MPI_Init`**：每个进程进入 MPI 环境，准备通信基础设施；
- **第二步，`MPI_Comm_rank/size`**：查询「我是谁、我们几个」，这是后续一切分支与通信的依据；
- **第三步，`MPI_Finalize`**：退出 MPI 环境，回收资源。

注意输出的顺序**不保证**：

进程各自独立运行，谁先打印由调度器说了算——这正是并行程序与串行程序的第一个直观差别。

## 5 核心对比：MPI 与多线程

| 维度 | MPI | 多线程（OpenMP/Pthreads） |
| --- | --- | --- |
| 执行单元 | 进程（独立地址空间） | 线程（共享地址空间） |
| 通信方式 | 显式消息传递 | 读写共享变量 |
| 内存隔离 | 强，天然不串扰 | 弱，需要锁保护 |
| 可运行范围 | 单机到超级计算机 | 通常单机 |
| 调试难度 | 消息时序难查 | 竞态难查 |

**核心结论：** MPI 把「并行」显式化了——数据流动肉眼可见，代价是编程繁琐。

但正因如此，MPI 程序可以在**任何规模的机器**上运行，这是它三十年来不可替代的根本原因。

## 6 小结

- **MPI** 是消息传递编程的标准接口，可移植到任意规模机器。
- 执行模型是 **SPMD**：一份程序、多进程执行，靠 `rank` 分流。
- 三个核心概念：**进程、rank（编号）、通信子（进程的容器）**。
- 第一个程序四步走：`MPI_Init` → 查 `rank/size` → 干活 → `MPI_Finalize`。
- 铁律：**MPI 程序没有「main 只跑一次」，必须用 rank 分配工作。**

在下一节，我们让进程真正「说上话」：学习 **MPI 集体通信**，让所有进程一次性完成广播、归约与分发。
