---
title: MPI 集体通信
date: 2026-08-07
---

# MPI 集体通信

<div class="epigraph">
<p>当你向所有人喊话时，效率取决于你嗓门之外的安排。</p>
<footer>—— 谚语</footer>
</div>

<div class="article-byline">
<p>第三级 · 高性能计算 ｜ 陈国良《并行计算》 第四章 §4.2 ｜ 2026-08-07</p>
</div>

## 为什么从集体通信开始

上一篇的 Hello World 里，进程之间还没真正说过话。

「让所有进程一起干一件事」是并行程序最频繁的需求：

把一份参数告诉所有人、把结果收集到根进程、把局部和归约成全局和。

自己用点对点写这些，费时又容易错。

MPI 提供了一组**集体通信（collective communication）**原语，一句调用完成全队协作。

<span class="marginnote">集体通信的正确用法是「所有进程都必须调用同一个集体操作，按相同顺序」——谁少调一次，谁就先死锁。</span>

本节把七个最常用的集体操作讲透：语义、签名、算法与成本。

## 1 从点对点到集体

点对点通信是「你发给我」。

集体通信是「全体同时参与一个动作」。

MPI 里每个集体操作都要求：**通信子内所有进程都调用该函数**。

一个进程的 `MPI_Bcast` 需要其他进程也调用 `MPI_Bcast` 才能配对完成。

集体操作都带一个**根进程（root）**参数：

- root 是「主角」，持有被广播/被收集的数据；
- 其他进程是「群演」，各自提供/接收自己的分片。

集体操作还要求指定 **MPI 数据类型（MPI_Datatype）**：

`MPI_INT`、`MPI_DOUBLE`、`MPI_CHAR`……MPI 需要知道每条消息的「元素大小」，才能正确编组传输。

## 2 广播与分发：Bcast 与 Scatter

**`MPI_Bcast`：一传众**。

根进程把一条数据复制给所有人。

```c
MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// 根进程持有 x，调用后所有进程的 x 都一样
```

语义要点：**所有人收到的内容相同**。

**`MPI_Scatter`：一拆众**。

根进程把一个大数组按块**分片**，每人拿自己那块。

```c
MPI_Scatter(sendbuf, n, MPI_DOUBLE,
            recvbuf, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// 根进程的 sendbuf 被切成 size 块，第 i 块给 rank i
```

<span class="marginnote">把 1 万个数分给 10 个进程，每个进程拿到 1000 个数，正好构成「数据并行」的起点——每人算自己的切片。</span>

**辨析｜易错点：** `MPI_Bcast` 与 `MPI_Scatter` 的关键差别：

**Bcast 复制同一份，Scatter 分发不同片。**

## 3 收集与归约：Gather 与 Reduce

**`MPI_Gather`：众合一**。

每个进程给根进程一块数据，根进程按 rank 顺序拼接成大数组。

```c
MPI_Gather(sendbuf, n, MPI_DOUBLE,
           recvbuf, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// 根进程把各块按 rank 0,1,2,... 拼起来
```

**`MPI_Allgather`：众合众**。

Gather 之后再把完整结果广播给所有人——**人人拿到完整拼图**。

**`MPI_Reduce`：众算一**。

每个进程给一个局部结果，在根进程上做归约运算：

```c
MPI_Reduce(&local, &global, 1, MPI_DOUBLE,
           MPI_SUM, 0, MPI_COMM_WORLD);
// global = 所有进程 local 的和，只有根进程拿到
```

**`MPI_Allreduce`：众算众**。

Reduce 之后把结果再广播给所有人——**人人拿到全局结果**。

归约操作由 `MPI_Op` 指定：`MPI_SUM`（求和）、`MPI_MAX`（取最大）、`MPI_MIN`、`MPI_PROD` 等。

<span class="marginnote">`MPI_Allreduce` 就是大模型梯度同步的鼻祖：每个 GPU 算完局部梯度，all-reduce 一下，人人拿到平均梯度再更新参数。</span>

## 4 代码解析：一个 all-reduce 求和

把「每个进程算自己那段的和，再求全局和」写成完整程序：

```c
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local = rank + 1.0;      // 每个进程的局部值
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    // global = 1+2+...+size = size*(size+1)/2，人人一致
    MPI_Finalize();
    return 0;
}
```

三步读懂：

- **第一步，构造局部值**：每个进程各自准备 `local`，内容可以各不相同；
- **第二步，`MPI_Allreduce`**：所有进程的 `local` 求和，结果写入每个人的 `global`；
- **第三步，验证结果**：若 4 个进程，`global` 应为 $4 \times 5 / 2 = 10$。

**辨析｜易错点：** `MPI_Reduce` 的结果**只有根进程**有，`MPI_Allreduce` **人人**有。

若在 `MPI_Reduce` 后让非根进程读 `global`，读到的是一块未定义内存。

## 5 公式解析：树状广播的成本

`MPI_Bcast` 的朴素实现是根进程逐一发给每个人，成本随进程数线性增长。

标准实现用**树状广播**：根进程发给几个「代表」，代表再向下转发。

设延迟为 $\alpha$、带宽为 $\beta$、消息 $n$ 字节、进程 $p$ 个。

树状广播的时间近似：

$$T_{\text{bcast}} \approx \log_2 p \cdot \left(\alpha + \frac{n}{\beta}\right)$$

拆三步理解：

- **第一步，看 $\alpha$**：每一层转发都是一次「发消息」的固定延迟，树高 $\log_2 p$，所以延迟部分乘 $\log_2 p$。
- **第二步，看 $n/\beta$**：每一层转发完整数据，传输部分也乘 $\log_2 p$。
- **第三步，对比朴素法**：朴素法 $T = (p-1)(\alpha + n/\beta)$，树状法从 $O(p)$ 降到 $O(\log p)$——进程越多，节省越惊人。

同一套思想（树状/环状/桶状）是所有集体通信优化的底层逻辑：

**用并行转发替代串行广播，让「发消息」这件事本身也并行起来。**

## 6 核心对比：七个集体操作

| 操作 | 方向 | 根进程结果 | 人人结果 | 数据是否一致 |
| --- | --- | --- | --- | --- |
| `MPI_Bcast` | 一→多 | 有 | 有 | 同一份 |
| `MPI_Scatter` | 一→多 | 有 | 有 | 各拿各片 |
| `MPI_Gather` | 多→一 | 有 | 无 | 拼成整块 |
| `MPI_Allgather` | 多→多 | 有 | 有 | 拼成整块 |
| `MPI_Reduce` | 多→一 | 有 | 无 | 归约结果 |
| `MPI_Allreduce` | 多→多 | 有 | 有 | 归约结果 |
| `MPI_Barrier` | 同步 | — | — | — |

## 7 小结

- 集体通信要求**全员调用、顺序一致**，否则死锁。
- **Bcast 复制、Scatter 分发、Gather 拼合、Reduce 归约**，四个动词记牢。
- `MPI_Reduce` 只有根进程拿结果，`MPI_Allreduce` 人人有。
- 集体操作底层用**树状/环状**算法，成本从 $O(p)$ 降到 $O(\log p)$。
- 铁律：**先分清「要不要 root」，再决定用 Reduce 还是 Allreduce。**

在下一节，我们回到更底层的沟通：学习 **MPI 点对点与非阻塞通信**，把通信与计算重叠起来。
