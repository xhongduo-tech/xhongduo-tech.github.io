## 核心结论

CUDA 的执行层次是 `Grid -> Block -> Thread`。`Grid` 是一次 kernel 启动生成的全部工作集合，白话说就是“这一轮 GPU 要做完的所有小任务”；`Block` 是可以在同一个 SM 上协作执行的一组线程，白话说就是“能共享片上资源的一队人”；`Thread` 是最小编程单位，白话说就是“真正负责一个数据元素或一个局部计算的执行者”。

这三层里，真正接近硬件调度粒度的是 `Warp`。`Warp` 是 32 个线程组成的执行组，白话说就是“SM 一次按同一条指令推进的一小队线程”。Block 内线程会被切成若干个 warp，由 SM 以 `SIMT` 模式执行。SIMT 是 single-instruction multiple-threads，意思是“很多线程看起来各自写代码，但硬件按一组线程共同推进指令”。

最重要的设计约束有三条：

| 层次 | 你控制什么 | 硬件保证什么 | 不能假设什么 |
| --- | --- | --- | --- |
| Grid | 总工作量、Block 总数 | 全部 Block 最终会执行 | Block 执行先后顺序 |
| Block | 线程数、局部协作方式 | 一个 Block 在单个 SM 上执行 | Block 能和别的 Block 直接同步 |
| Warp | 通常不直接创建，只通过线程布局影响 | 32 线程为一组执行 | 分支后仍保持满效率 |

并行规模的第一步估算公式是：

$$
总线程数 = gridDim.x \times gridDim.y \times blockDim.x \times blockDim.y
$$

如果 `Grid=(4,4)`，`Block=(16,16)`，那么总线程数是：

$$
4 \times 4 \times 16 \times 16 = 4096
$$

每个 Block 有 $16 \times 16 = 256$ 个线程，因此每个 Block 会被拆成：

$$
warpCountPerBlock = \lceil 256 / 32 \rceil = 8
$$

这说明：你写的是 256 个线程协作的 Block，硬件看到的是 8 个 warp 在若干个 SM 上被调度。

---

## 问题定义与边界

本文要解决的问题不是“怎么把 CUDA 代码写对”，而是“为什么 Grid、Block、Thread 要这样分层，以及这套分层和 SM、CUDA Core、Warp 到底怎么对应”。这决定了你后面写矩阵乘、卷积、归约、扫描时，性能会不会从一开始就跑偏。

先划清边界。

第一，`SM` 是 Streaming Multiprocessor，中文常叫流多处理器，白话说就是“GPU 内部一个能独立调度多个 warp 的计算工作站”。`CUDA Core` 是 SM 内部执行标量算术的一类功能单元，白话说就是“真正做加减乘除的计算器”。二者不是一一对应关系，不是“一个 SM 等于一个 CUDA Core”，而是“一个 SM 里面有很多 CUDA Core、寄存器、共享内存、warp 调度器等资源”。

第二，`Block` 和 `SM` 的关系是“执行归属”而不是“永久绑定”。一个 Block 在生命周期内只会落到一个 SM 上执行，但 Grid 里的很多 Block 会被硬件按资源情况动态分配到空闲 SM。于是 `Grid size` 可以远大于 `SM` 数量，GPU 会分批把 Block 跑完。

第三，跨 Block 不能假设同步。原因不是“有时候不能”，而是 CUDA 编程模型从设计上就要求 Block 能按任意顺序运行，才能让同一份程序在几十个 SM 或上百个 SM 的 GPU 上都成立。

下面这个思路就是危险的：

```cpp
__device__ int flag = 0;

__global__ void bad_kernel(float* data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        flag = 1;
    }

    if (blockIdx.x == 1 && flag == 1) {
        data[0] = 42.0f;  // 错误：假设 block 0 一定先于 block 1 完成
    }
}
```

这里的问题不是语法，而是执行模型。`blockIdx.x == 1` 的 Block 可能先被调度，也可能和 `blockIdx.x == 0` 并行存在，甚至后者因为资源不足还没上 SM。正确做法是把跨 Block 依赖拆成两个 kernel，或通过 host 端同步后再启动下一轮。

这个边界可以压缩成三点：

| 边界 | 含义 | 直接后果 |
| --- | --- | --- |
| Block 在单个 SM 上执行 | Block 内可以高效同步与共享数据 | `__syncthreads()` 只对本 Block 有效 |
| Grid 可以远大于 SM 数 | Block 会被分批调度 | 不能把 Block 数量等同于并发数 |
| Block 无顺序保证 | Block 之间默认独立 | 跨 Block 依赖必须拆阶段 |

玩具例子可以这样看：你有 10000 个数组元素要加一，开 `Grid=(40,1,1)`、`Block=(256,1,1)`。你并不是“同时启动了 10240 个真实硬件线程”，而是向 GPU 说明“这里有 40 个 Block，每个 Block 256 个线程，请你按资源自行调度”。真正并发上去的是当前若干个活跃 Block 内的若干个 warp。

---

## 核心机制与推导

先看映射链路：

`Grid` 把工作拆成很多 `Block`，`Block` 再拆成很多 `Thread`，而硬件实际以 `Warp` 为单位在 `SM` 上调度。

因此，理解性能时最有用的不是“一个线程做什么”，而是“一个 warp 里的 32 个线程是不是在做相近的事”。

### 1. Block 到 Warp 的推导

如果一个 Block 有 `threadsPerBlock` 个线程，那么：

$$
warpCountPerBlock = \left\lceil \frac{threadsPerBlock}{32} \right\rceil
$$

例如 `threadsPerBlock = 256`，则 `warpCountPerBlock = 8`。  
如果是 `threadsPerBlock = 300`，则 `warpCountPerBlock = 10`，但最后一个 warp 只有 12 个线程有效，剩余 20 个 lane 空置。`lane` 是 warp 内线程位置编号，白话说就是“32 人小队里的座位号”。

这就是为什么 Block 线程数通常选 128、256、512 这类 32 的倍数。不是语法要求，而是为了不浪费 warp 槽位。

### 2. SIMT 与分支分歧

SIMT 下，同一个 warp 中的线程通常一起推进同一条指令。如果出现条件分支，比如一半线程进 `if`，另一半线程走 `else`，硬件会把一条路径先执行，再把另一条路径执行，没有走该路径的线程会被屏蔽。这个现象叫 `branch divergence`，中文常说分支分歧，白话说就是“同一队线程意见不一致，结果只能排队分批走”。

典型公式不是算术耗时公式，而是有效利用率概念：

$$
occupancy = \frac{activeWarpCount}{maxWarpPerSM}
$$

它表示一个 SM 上当前活跃 warp 数占理论上限的比例。占用率高不等于一定最快，但占用率太低时，访存等待和流水线空泡更难被别的 warp 隐藏。

下面这个数值例子最直观：

- 一个 Block 有 256 线程，对应 8 个 warp。
- 假设某个 SM 同时容纳 4 个这样的 Block，则该 SM 上最多有 32 个 warp 来回切换。
- 如果其中一个 warp 执行 `if (threadIdx.x % 2 == 0)`，那么 32 个线程会裂成 16/16 两组。
- 结果不是“并行做两件事”，而是“先执行偶数线程路径，再执行奇数线程路径”。

可以用表格看清这个损失：

| 情况 | Warp 内活跃线程 | 指令发射次数 | 结果 |
| --- | --- | --- | --- |
| 无分歧，32 线程同一路径 | 32/32 | 1 次 | 满效率 |
| 16 线程进 `if`，16 线程不进 | 16/32 然后 16/32 | 约 2 次 | 同一 warp 被串行化 |
| 8/8/8/8 四种路径 | 最多 8/32 每次 | 最多 4 次 | 利用率更低 |

### 3. SM、Warp 调度器、CUDA Core 的关系

初学者最容易混淆的是“warp 有 32 线程，所以 SM 一定有 32 个 CUDA Core 吗”。不能这样推。

更准确的说法是：

- `SM` 是调度与资源容器。
- `Warp` 是执行调度粒度。
- `CUDA Core` 是 SM 内的一类算术执行单元。
- 一个 warp 的一条指令如何映射到多少个执行单元、分几拍发出，取决于架构实现。

所以写 CUDA 代码时，应该依赖编程模型保证：Block 在单 SM 上，warp 是 32 线程，SIMT 可能因分支而屏蔽线程。不要把某代 GPU 的具体 CUDA Core 数量直接写成算法前提。

---

## 代码实现

先给一个最小可运行的 Python 玩具例子，用来验证索引映射和 warp 计数。它不是 CUDA 运行环境，但能把公式和层次关系算清楚。

```python
import math

def global_xy(block_idx, thread_idx, block_dim):
    bx, by = block_idx
    tx, ty = thread_idx
    bdx, bdy = block_dim
    x = bx * bdx + tx
    y = by * bdy + ty
    return x, y

def total_threads(grid_dim, block_dim):
    gx, gy = grid_dim
    bx, by = block_dim
    return gx * gy * bx * by

def warp_count_per_block(block_dim):
    bx, by = block_dim
    threads = bx * by
    return math.ceil(threads / 32)

grid = (4, 4)
block = (16, 16)

assert total_threads(grid, block) == 4096
assert warp_count_per_block(block) == 8
assert global_xy((0, 0), (0, 0), block) == (0, 0)
assert global_xy((1, 0), (3, 2), block) == (19, 2)
assert global_xy((2, 3), (15, 15), block) == (47, 63)

print("all checks passed")
```

这个玩具例子对应的含义是：

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| `grid=(4,4)` | 16 个 Block | 工作被拆成 16 片 |
| `block=(16,16)` | 每 Block 256 线程 | 每片内部 16x16 排布 |
| `warp_count_per_block=8` | 8 个 warp | SM 调度看到的是 8 个执行组 |

接着看真实 CUDA 内核。下面是一个二维矩阵乘的基础版本，重点不是最优性能，而是说明 Grid/Block/Thread 如何映射到数据坐标：

```cpp
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            acc += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
```

启动配置通常写成：

```cpp
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
matMulNaive<<<grid, block>>>(A, B, C, N);
```

这里的映射关系是：

- 一个线程负责输出矩阵 `C[row][col]` 的一个元素。
- 一个 `16x16` 的 Block 负责一个 `16x16` 输出子块。
- 整个 Grid 覆盖完整矩阵。

真实工程例子里，矩阵乘通常不会停在这个 naive 版本，而是继续做 `tiling`。`tiling` 是分块复用数据，白话说就是“先把一小块 A 和一小块 B 放进共享内存，再让 Block 内线程反复用，减少全局内存读取”。这时 Block 内线程需要 `__syncthreads()` 协调共享内存装载，但这个同步仍然只在单个 Block 内有效。

例如 `N=1024`、`block=(16,16)` 时：

| 项 | 数值 |
| --- | --- |
| `grid.x = ceil(1024/16)` | 64 |
| `grid.y = ceil(1024/16)` | 64 |
| Block 总数 | 4096 |
| 每 Block 线程数 | 256 |
| 每 Block warp 数 | 8 |

这 4096 个 Block 不会一起塞进 GPU，而是由多个 SM 分批执行。你只需要保证每个 Block 逻辑独立，硬件就能自动扩展。

---

## 工程权衡与常见坑

第一类坑是分支分歧。新手常写：

```cpp
if (threadIdx.x % 2 == 0) {
    do_a();
} else {
    do_b();
}
```

这几乎保证同一个 warp 内线程交错分布到两条路径。更好的写法通常是让条件按连续区间切分，例如按 `threadIdx.x < 16` 或按数据块边界划分，这样更容易让同一 warp 中的大多数线程走一致路径。

第二类坑是 Block 大小不是 32 的倍数。比如 48 线程一个 Block 在语法上合法，但它会形成 2 个 warp，其中第二个 warp 只有 16 个线程有效。你并没有得到“精确 48 线程的优雅配置”，而是得到了“一个满 warp 加一个半空 warp”。

第三类坑是误把 occupancy 当唯一目标。占用率高能帮助隐藏延迟，但不是越高越好。更大的 Block 会带来更多线程，但也会占用更多寄存器和共享内存，反而可能减少一个 SM 能同时挂住的 Block 数。工程上要在这些资源之间平衡。

| 常见坑 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| Warp divergence | 同一 warp 执行变慢 | 分支路径被串行化 | 让 warp 内条件尽量一致 |
| Block 非 32 倍数 | 最后一个 warp 低利用率 | 部分 lane 永久空置 | 优先选 128/256/512 线程 |
| 跨 Block 依赖 | 结果不稳定或错误 | Block 无顺序保证 | 拆成多个 kernel 阶段 |
| 过大 Block | 活跃 Block 数下降 | 寄存器或共享内存吃满 | 用 profiler 结合资源使用调参 |
| 只看线程数不看访存 | 算力不满 | 内存访问未合并 | 让相邻线程访问相邻地址 |

一个常见真实工程场景是图像卷积。假设每个线程负责一个像素，如果线程按二维连续布局访问输入图像，warp 内地址通常也是连续的，更容易形成合并访存；如果线程索引映射混乱，即使算术量一样，也可能因为内存访问模式差而明显变慢。也就是说，Grid/Block/Thread 的层次不只是“怎么编号”，而是直接影响 SM 能不能把 warp 跑顺。

---

## 替代方案与适用边界

标准 CUDA 模型已经覆盖了绝大多数 kernel：Grid 负责全局并行，Block 负责局部协作，warp 是硬件执行粒度。如果你的需求只是“每个输出元素独立计算”或“Block 内做共享内存归约”，标准 Grid/Block 就够了。

更进一步的方案是 `Thread Block Cluster`。`Cluster` 是 compute capability 9.0 及以上可选的一层，位于 Block 之上，白话说就是“把多个相邻 Block 作为一组，要求它们共同调度到同一个 GPC”。这样可以用 Cooperative Groups 做更强的跨 Block 协作，并访问 distributed shared memory。

它的适用边界如下：

| 方案 | 适合什么问题 | 优点 | 限制 |
| --- | --- | --- | --- |
| 标准 Grid/Block | 大多数数据并行任务 | 模型简单，可移植性强 | Block 间不能直接同步 |
| Cooperative Groups（Block 内或部分扩展） | 更清晰的组内协作 | 同步语义更明确 | 仍受硬件层级限制 |
| Cluster + Cooperative Groups | 需要一小组 Block 强协作 | 可在 cluster 内通信同步 | 仅部分硬件支持，配置更复杂 |

关于 cluster，大多数设备可移植的经验值是：

$$
clusterSize \le 8
$$

但这不是“所有 GPU 永远固定等于 8”。更准确地说，8 是可移植的 cluster 大小上限之一，实际最大值与架构相关，工程上应通过 `cudaOccupancyMaxPotentialClusterSize` 查询，再决定 launch 配置。

因此，替代方案不是“新模型一定更强”，而是“当标准 Block 边界妨碍合作算法时，再引入更高层级约束”。如果只是普通矩阵乘、逐元素变换、卷积前向，大多数情况下仍先把标准 Grid/Block/warp 模型吃透。

---

## 参考资料

- NVIDIA CUDA Programming Guide, 1.2 Programming Model: https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/01-introduction/programming-model.html
- NVIDIA CUDA Programming Guide, 3.2 Advanced Kernel Programming: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
- NVIDIA Technical Blog, CUDA Refresher: The CUDA Programming Model: https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
- NVIDIA CUDA C Programming Guide 11.8, Thread Block Clusters and occupancy APIs: https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html
