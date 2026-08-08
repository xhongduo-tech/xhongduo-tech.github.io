---
title: 线程层次：Grid、Block、Thread 的组织与索引
date: 2026-08-07
---

# 线程层次：Grid、Block、Thread 的组织与索引

<div class="epigraph">
<p>编号应当从零开始。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra，《Why numbering should start at zero》，1982）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第一篇 ｜ 2026-08-07</p>
</div>

## 为什么从线程层次开始

前两课我们从硬件出发：SM 是执行单元，warp 是调度单位，硬件把线程切分成 warp 来批量执行。但程序员写 CUDA 时，面前并不是 warp 这个硬件概念，而是一套软件概念——**线程（Thread）、线程块（Block）、网格（Grid）** 的三级层次。这是程序员与硬件之间的契约：你以什么粒度组织线程，硬件就按什么粒度把计算铺到 SM 上。

这一课要回答三件事：三级层次各自是什么、之间有哪些限制；如何用 threadIdx / blockIdx / blockDim / gridDim 把线程「身份」翻译成内存下标；以及 block 如何被调度到 SM、为何 block 大小是 32 的倍数。这一课是写任何 kernel 的第一行代码，也是后续 Occupancy、共享内存、合并访存的地基。<span class="marginnote">上一课《SIMT 与 warp》里「block 的线程按 threadIdx.x 切成 warp」是硬件的视角；这一课从软件视角补全：block 为什么存在、block 之间的边界在哪里、为什么 __syncthreads 只能在 block 内用。两者拼起来才是完整的映射。</span>

## 1 三级层次：Thread、Block、Grid

**线程（thread）**：CUDA 程序执行的最小单位，一段「每个线程做一件事」的代码副本。线程有自己的寄存器、自己的程序计数器（Volta 之后），通过内置变量知道自己的编号。

**线程块（block）**：一组线程的集合，最大 1024 个线程。block 有三个关键性质：

**块内同步**：同一 block 的线程可以用 __syncthreads() 同步，并共享一块**共享内存（Shared Memory）**。
**块间隔离**：不同 block 的线程之间**不能**同步（除非用合作式网格同步 Cooperative Groups），也不能访问彼此的共享内存。
**单 SM 驻留**：一个 block 完整地运行在同一个 SM 上，从开始到结束不迁移。

**网格（grid）**：一次 kernel 启动创建的全部 block 的集合。网格就是「这一次计算的全部范围」，它由硬件/驱动分发到 GPU 上所有可用的 SM。<span class="marginnote">把「网格—块—线程」类比一个学校的运动会：网格是整场运动会，block 是各班级方阵，thread 是方阵里的学生。方阵内部可以整齐划一（__syncthreads）、可以共享物资（共享内存）；不同方阵之间互不干涉、各自按自己的节奏走完。</span>

kernel 启动语法 <<<grid, block>>>` 就是一次指定两个维度：网格有多少 block、每个 block 有多少线程。

CUDA 提供 4 个内置变量来让线程认清自己的位置：

| 内置变量 | 类型 | 含义 | 例子 |
| --- | --- | --- | --- |
| threadIdx | uint3 | 本线程在 block 内的坐标 | threadIdx.x = 42 |
| blockIdx | uint3 | 本 block 在 grid 内的坐标 | blockIdx.x = 3 |
| blockDim | dim3 | block 每维的线程数（对全体线程一致） | blockDim.x = 256 |
| gridDim | dim3 | grid 每维的 block 数（对全体线程一致） | gridDim.x = 10 |

![Grid、Block、Thread 三级线程层次与索引示意](/images/ai-infra/thread-hierarchy-1.svg)

**重点：blockDim/gridDim 是「全局常量」，threadIdx/blockIdx 是「每线程变量」。** 前者描述结构（长什么样），后者描述身份（我在哪）——混淆这两对，是所有索引 bug 的源头。

硬件映射关系一句话：**网格里的 block 被分发到不同的 SM；block 里的线程被切成 warp；warp 被调度器发射执行。**

## 2 一维索引与二维索引：把「身份」翻译成内存地址

一个 kernel 拿到的是扁平的一维数组，而线程有三维坐标，于是最常做的事就是「**把 (blockIdx, threadIdx) 换算成全局下标**」。一维情况是最基本公式：

$$
i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

**直觉**：先数「我前面的 block 一共贡献了多少线程」（每个 block 有 blockDim.x 个），再加上「我在自己 block 内的位置」。例如 blockIdx.x = 3、blockDim.x = 256、threadIdx.x = 42，则 $i = 3 \times 256 + 42 = 810$。

二维情况稍复杂。设处理一张 $W \times H$ 的图，用二维 grid 与二维 block：

```cuda
__global__ void kernel2d(const float* in, float* out, int W, int H) {
    int tx = threadIdx.x, ty = threadIdx.y;   // block 内坐标
    int bx = blockIdx.x,  by = blockIdx.y;    // block 在 grid 内坐标
    int row = by * blockDim.y + ty;           // 全局行
    int col = bx * blockDim.x + tx;           // 全局列
    if (row < H && col < W) {                 // 边界 guard：防止尾部线程越界
        out[row * W + col] = in[row * W + col] * 2.0f;
    }
}
```

这里的 `row * W + col` 就是「先按行数偏移，再加列」——与一维公式同构，只是把「行」当成外层循环。**注意代码里始终带着 `if (row < H && col < W)` 边界检查**：当数据维度不是 block 维度的整数倍时，尾部线程会越界，必须用 guard 拦下。这是所有 CUDA 初学者踩过的最多的坑。<span class="marginnote">边界检查的代价极小（一条比较+分支），却能让 kernel 在任意尺寸的输入上正确运行。GPU 上的「求整」习惯是「先算一个覆盖全部数据的网格，再在 kernel 里用 guard 过滤」——这与 CPU 上精确计算循环边界完全不同。</span>

## 3 block 与 SM 的对应：一次启动，如何铺到硬件

一次 kernel 启动后，GPU 端的工作分配是这样的：

1. 驱动把 grid 里的 block **逐个分配给空闲的 SM**；每个 SM 能同时驻留多个 block（受线程数、寄存器数、共享内存三重上限约束，详见《Occupancy》）。
2. block 一旦上 SM 就「钉死」在该 SM 上，直到执行完。
3. block 内的线程被切成 warp，交到该 SM 的 4 个 warp 调度器手里。

由此得出几个直接影响性能的推论：

**推论一：block 是负载均衡的最小单位。** 假设网格有 100 个 block、GPU 上 100 个 SM 每个能驻留 1 个 block，则各 SM 各拿一个，均衡；若网格只有 3 个 block 而 GPU 有 100 个 SM，则 3 个 SM 干活、97 个空转——**网格的 block 数太少会「喂不饱」GPU**。一般建议网格 block 数至少是「SM 数 × 每 SM 可驻留 block 数」的若干倍。

**推论二：block 越大，同步与共享越方便，但粒度越粗。** block 大意味着更少的 block、更粗的负载均衡粒度；block 小意味着更多 block、调度开销略增但均衡更细。实践中 128–512 线程是常见区间。

**推论三：block 大小取 32 的倍数。** 因为 block 的线程是按 32 个一组的 warp 执行的，末尾不满 32 的 warp 会浪费槽位与寄存器。取 128/256/512 让每个 warp 都「满编」。

当数据量远大于一次网格能覆盖的范围时，用**网格跨越循环（grid-stride loop）**：让每个线程处理多个元素，步长为 gridDim.x * blockDim.x：

```cuda
__global__ void saxpy_grid_stride(float* out, const float* x, const float* y,
                                  float a, int n) {
    int stride = gridDim.x * blockDim.x;      // 整个网格覆盖的线程总数
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += stride) {                       // 每次跨过整个网格
        out[i] = a * x[i] + y[i];
    }
}
```

这样网格尺寸不再受数据量限制，可以固定为一个「恰好能装满 GPU」的规模。<span class="marginnote">grid-stride loop 还能顺带提升访存局部性、减少启动开销，是「数据规模变化」场景下的标准写法。它把「一次启动铺满数据」变成「一次启动铺满 GPU，再循环取数据」——两种哲学，适合不同场景。</span>

## 4 公式解析：二维 grid/block 的线性化

把二维索引完整地摊成一个全局线性下标，是理解「线程如何一一对应到内存地址」的试金石。设 grid 每维有 $G_x, G_y$ 个 block，block 每维有 $B_x, B_y$ 个线程，当前线程的 block 坐标为 $(b_x, b_y)$，块内坐标为 $(t_x, t_y)$，全局线性下标为 $i$：

$$
i = (b_y \cdot G_x + b_x) \cdot (B_x \cdot B_y) + t_y \cdot B_x + t_x
$$

对这条式子做三步拆解：

- **第一步，先数 block 的前后**：$b_y \cdot G_x + b_x$ 是把二维 block 坐标按「行主序」变成一维 block 序号——先数上方有多少整行 block（$b_y \cdot G_x$），再加本行内的列偏移 $b_x$。
- **第二步，乘以每 block 的线程数**：每个 block 有 $B_x \cdot B_y$ 个线程，所以前面的 block 一共贡献了 $(b_y G_x + b_x) \cdot B_x B_y$ 个线程。
- **第三步，加本 block 内的线程偏移**：块内同样是行主序，先数上方的整行线程 $t_y \cdot B_x$，再加本行的列偏移 $t_x$。

于是整个下标 = 前面的 block 贡献的线程数 + 块内前面的线程数。这与一维公式 $i = b \cdot B + t$ 完全同构，只是每一层都多了一个维度。**读懂这条公式，等于同时读懂了 blockIdx、blockDim、threadIdx 三个变量的语义**——它们是这条公式的三个因数。

顺带一提，CUDA 的坐标从 0 开始编号。迪杰斯特拉在 1982 年那篇著名短文里论证了「半开区间 + 从 0 开始」的编号方式最优雅：它让下标既是偏移量又是数量，让 `i < n` 这样的边界判断永远干净。你在上面所有公式里看到的 `blockIdx.x * blockDim.x + threadIdx.x` 这类下标，本质上就是「从 0 开始」才写得出的偏移计算。

## 5 辨析与易错点

**辨析｜易错点一：四个内置变量分不清。** threadIdx 是「我在 block 里排第几」（每线程不同），blockIdx 是「我的 block 在 grid 里排第几」（同 block 内所有线程相同），blockDim 是「block 有多大」（全局一致），gridDim 是「grid 有多大」（全局一致）。一个自测：`i = blockIdx.x * blockDim.x + threadIdx.x` 这个式子里的 blockIdx 与 threadIdx 是变量，blockDim 是常量——写错任何一个，下标就错位。

**辨析｜易错点二：block 是软件概念，warp 是硬件概念。** 你在代码里写的 <<<grid, 100>>>` 是「100 个线程组成的 block」，硬件执行时把它切成 $\lceil 100/32 \rceil = 4$ 个 warp，最后一个 warp 只有 4 个活跃线程。你不能在代码里直接控制 warp 的划分，只能通过「block 大小取 32 的倍数」来保证不浪费。

**辨析｜易错点三：__syncthreads() 只在 block 内有效。** 它保证「block 内所有线程到达该点之后，才继续往下走」。不同 block 之间没有这种保证，所以**不同 block 之间不能通过共享内存交换数据**（那属于全局内存 + 全局同步/多次 kernel 的范畴）。试图跨 block 同步，是经典的死锁/未定义行为来源。

## 6 小结

- **三级层次**：Thread（执行单位）→ Block（同步域 + 共享内存域，≤1024 线程）→ Grid（一次启动的全部 block）。
- **四个内置变量**：threadIdx/blockIdx 是身份，blockDim/gridDim 是结构；一维全局下标 $i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$。
- **硬件映射**：block 整体驻留一个 SM 且不迁移；block 内线程按 32 切 warp；网格太大时用 grid-stride loop。
- **实践规则**：block 大小取 32 的倍数、网格 block 数要能喂饱所有 SM、始终带边界 guard。
- **二维线性化**：$i = (b_y G_x + b_x) B_x B_y + t_y B_x + t_x$，与一维同构，从 0 开始编号。

在下一节，我们将从「线程怎么组织」进入「线程怎么被调度、以及不听话的代价」——**warp 调度与分支分化（divergence）**，那里有本专题第一个真正的性能陷阱。
