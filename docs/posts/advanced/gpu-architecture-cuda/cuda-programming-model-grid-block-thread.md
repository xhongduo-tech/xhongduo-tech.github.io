---
title: CUDA 编程模型（grid/block/thread、kernel 启动）
date: 2026-08-07
---

# CUDA 编程模型（grid/block/thread、kernel 启动）

<div class="epigraph">
<p>抽象的目的不是含糊，而是创造一个能在其中获得绝对精确的新的语义层次。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra），《谦卑的程序员》（The Humble Programmer），1972</footer>
</div>

<div class="article-byline">
<p>第四级 · GPU 架构与 CUDA 并行编程 ｜ Kirk & Hwu, Programming Massively Parallel Processors, 4e, Ch2–Ch3 ｜ 2026-08-07</p>
</div>

## 为什么从 CUDA 编程模型开始

上一篇讲透了 SIMT 硬件：warp 是硬件调度的单位，线程是硬件执行的影子。但程序员不能直接面向 warp 编程——那太低级、太容易出错。**CUDA 编程模型就是 SIMT 硬件的「高层接口」**：它给你三个概念（grid、block、thread）和一句启动语法，剩下的切分、调度、执行全交给运行时。

理解 CUDA 编程模型，是「会用 GPU」的入场券。后面讲内存、讲占用率、讲流，全部建立在这套概念上。<span class="marginnote">PMPP 第 2 章 "Heterogeneous Data Parallel Computing" 讲 kernel 与 host/device 分工，第 3 章 "Multidimensional Grids and Data" 专门讲多维 grid 与数据映射——本篇合并这两章的核心。官方定义见《CUDA C++ Programming Guide》§2。</span>

## 1 异构模型：host 与 device

CUDA 程序默认跑在**异构（heterogeneous）**系统上：**CPU 是 host（主机），GPU 是 device（设备）**，二者通过 PCIe 总线或 NVLink 相连，各有各的内存空间。

- **host 代码**：普通的 C/C++ 代码，由 CPU 执行，负责数据准备、内存管理、kernel 启动与结果回收。
- **device 代码**：用 `__global__` 修饰的函数，由 GPU 执行，称为 **kernel（内核）**。

一个典型 CUDA 程序的生命周期是：CPU 分配显存 → CPU 把输入从主存拷到显存 → CPU 启动 kernel → GPU 执行 → CPU 把结果拷回主存 → CPU 释放内存。<span class="marginnote">「异构」是理解 CUDA 一切开销的起点：每一次主存↔显存的拷贝都要经过 PCIe（单次传输带宽几十 GB/s，远低于 GPU 内部带宽），所以「减少主机-设备拷贝」是优化的第一直觉——这一主题在《内存层次》与《流、事件与并发执行》两篇会反复出现。</span>

**核心约束：kernel 不能直接访问 host 内存，反之亦然。** 两边各有各的地址空间，必须显式拷贝。这条约束看似麻烦，却正是正确性保证——数据不会因为 GPU 乱写而悄悄破坏 CPU 的状态。

## 2 三个概念：grid、block、thread

CUDA 把 GPU 上要跑的所有线程组织成三层层级，这就是编程模型的心脏：

- **thread（线程）**：最小执行单位，每个线程执行 kernel 函数的一份拷贝，拥有独立的寄存器与局部变量。
- **block（线程块）**：一组线程的集合，同一 block 内的线程**共享一块共享内存**，并可通过同步原语协作（见《共享内存与 bank conflict、同步原语》一篇）。一个 block 只在一个 SM 上执行。
- **grid（网格）**：一次 kernel 启动的全部线程，由若干 block 组成。

启动 kernel 的语法是：

```c
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args);
```

其中 `gridDim` 是网格里 block 的排布，`blockDim` 是每个 block 里线程的排布。比如 `kernel<<<10, 128>>>` 表示「10 个 block，每个 128 个线程」，总计 1280 个线程。<span class="marginnote">这里藏着 SIMT 的翻译：运行时把每个 block 按 32 线程切分成 warp。`blockDim = 128` 的 block 正好被切成 4 个完整的 warp，没有残缺线程束——这也是上一篇「block 取 32 的倍数」建议的来历。</span>

## 3 公式解析：从线程坐标到数据下标

线程是「匿名」的——它不知道自己是第几个。程序员必须用内建变量推导出「我是谁」，再用它计算「我该处理哪份数据」。这就是**索引映射**，CUDA 编程最基本的动作。

对一维 grid，第 `blockIdx.x` 个 block 内第 `threadIdx.x` 个线程，处理的全局数据下标为：

$$
i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

拆解这条公式：

- **$\text{threadIdx.x}$**：线程在 block 内的局部编号，范围 $0 \sim \text{blockDim.x} - 1$。
- **$\text{blockIdx.x}$**：block 在整个 grid 中的编号，范围 $0 \sim \text{gridDim.x} - 1$。
- **$\text{blockDim.x}$**：每个 block 的线程数。前面的 block 每个都贡献 `blockDim.x` 个线程，所以当前 block 要加上「排在它前面的所有 block 的线程总数」`blockIdx.x * blockDim.x`。
- **$i$**：全局线程编号，范围 $0 \sim \text{gridDim.x} \times \text{blockDim.x} - 1$。

**直觉：先算出「我前面有几个线程」，再加上「我在 block 里的位置」，就是我的全局编号。** 数据总量通常不是线程数的整数倍，所以后面还要加一个越界判断 `if (i < N)`。<span class="marginnote">这个 `if (i < N)` 判断正是上一篇讲过的「分歧」的温和案例：只有末尾一个 warp 部分线程被掩码，其余 warp 全满——把尾部判断写成 `i < N` 而非 `i % something`，就是「让分歧发生在 warp 边界」的实践。</span>

## 4 一个完整的例子：向量加法

把上面的概念串成一个最小 kernel——向量加 $C = A + B$：

```c
__global__ void vecAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局索引
    if (i < N) {                                    // 越界保护
        C[i] = A[i] + B[i];
    }
}

int main() {
    // ... 分配并拷贝 A, B 到显存（略） ...
    int threads = 128;
    int blocks = (N + threads - 1) / threads;       // 向上取整
    vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);  // 启动 kernel
    // ... 拷回结果（略） ...
}
```

读这段代码，注意三件事：

- **每个线程处理一个元素**：`i` 就是数据下标，线程数 $\ge$ 数据数即可覆盖全数组。
- **`blocks` 向上取整**：`(N + threads - 1) / threads` 保证覆盖不足一个 block 的尾巴。
- **`if (i < N)` 兜底**：当 N 不是线程总数的整数倍时，多出来的线程什么都不做。

这就是 CUDA 的「写一个、跑一万个」：你只写了一个线程对 `C[i]` 的赋值，硬件把这一份逻辑复制到 `blocks * threads` 个线程上并行执行。

## 5 多维 grid：二维、三维数据怎么排

一维 grid 够处理一维数组，但图像、矩阵、体素数据天然是二维/三维的。CUDA 允许 grid 与 block 都是最多三维的：`gridDim` 可以是 `dim3(2,3)`，`blockDim` 可以是 `dim3(8,8)`，访问时用 `threadIdx.x`、`threadIdx.y`、`threadIdx.z`。

对二维 block，二维数据的下标映射是：

$$
\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}, \qquad
\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

二维组织最直观的好处是**数据局部性**：处理图像时，相邻线程访问相邻像素，天然满足「合并内存访问」的要求（下一篇会讲）——这也是「让线程布局匹配数据布局」的又一例。<span class="marginnote">PMPP 第 3 章把多维 grid 讲得很细：它不只是「把坐标变多」，而是「让 thread 的空间排布与数据/图像的空间排布一致」，从而最大化缓存与带宽利用率。</span>当数据是矩阵时，行优先还是列优先、block 是宽矩形还是高矩形，都会影响性能——这就是 CUDA 编程中「布局」学问的起点。

## 6 边界与约束：kernel 不是什么都能做

kernel 虽然是 C 函数的亲戚，但有一组硬约束，是新手最容易踩坑的地方：

- **默认不能递归、不能动态分配大量栈内存**（有动态并行但限制很多，见《统一内存与新特性》一篇）。
- **kernel 与 host 之间没有共享的普通指针**：需要指针必须是设备内存，传参可以传普通标量。
- **`printf` 在 kernel 内可用但有性能代价**（计算能力 2.x 起支持），一般只用于调试。
- **线程间无法直接「看见」对方的寄存器**：跨线程通信必须通过显存、共享内存或全局内存，再配同步。

这些约束不是缺陷，而是「为吞吐让路」的代价：简单的执行单元、无虚拟栈的硬件，才能把晶体管全部用在并行计算上。<span class="marginnote">记住一个思维转换：CPU 上「线程是抽象」，GPU 上「线程是实体」。GPU 线程极轻（一个线程只占一组寄存器），所以能开几十万个；但也正因如此，跨线程通信必须走显式的内存通道。</span>

## 7 小结

- CUDA 程序是**异构**的：host（CPU）负责控制与拷贝，device（GPU）负责并行计算，两侧内存分离，必须显式拷贝。
- 线程层级：**grid → block → thread**；启动语法 `kernel<<<gridDim, blockDim>>>(args)`。
- 全局索引公式 $i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$，配 `if (i < N)` 防越界。
- 一维公式可推广到二维/三维：`threadIdx.y` 映射到行，`threadIdx.x` 映射到列。
- kernel 有硬约束（不可随意递归、跨线程通信走显式内存），这是吞吐架构的代价。

在下一节，我们将进入性能的第一战场：**内存层次（全局/共享/常量/纹理内存、合并访问）**。
