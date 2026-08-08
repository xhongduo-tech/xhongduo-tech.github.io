---
title: CUDA 与 GPU 并行编程
date: 2026-08-07
---

# CUDA 与 GPU 并行编程

<div class="epigraph">
<p>GPU 不是更多核的 CPU，而是一座为吞吐而生的数据工厂。</p>
<footer>—— NVIDIA CUDA 编程指南</footer>
</div>

<div class="article-byline">
<p>第三级 · 高性能计算 ｜ 陈国良《并行计算》 第六章 §6.1 ｜ 2026-08-07</p>
</div>

## 为什么从 CUDA 开始

今天的 Top500 榜首，没有一台不靠 GPU 加速。

GPU 用几十倍的处理器数、以**吞吐优先**的设计，成为大模型训练与科学计算的主力引擎。

而 **CUDA（Compute Unified Device Architecture）**是 NVIDIA 提供的并行编程平台：

写一个 kernel，就能让上万线程在同一块芯片上同时干活。

<span class="marginnote">大模型的训练与推理、数值模拟、图像处理，底层几乎都是 CUDA kernel。理解 CUDA 的线程与内存模型，就理解了当代 AI 算力的物理基础。</span>

本节的纲：

GPU 为什么快、线程怎么组织、数据怎么流动、以及一个完整的 kernel 长什么样。

## 1 GPU 凭什么快：SIMT 与海量线程

CPU 的核心设计目标是**低延迟**：单线程要快，所以堆缓存、堆乱序执行。

GPU 的核心设计目标是**高吞吐**：不在乎单线程多快，而在乎**同时干活的线程够不够多**。

GPU 的执行模型叫 **SIMT（Single Instruction, Multiple Thread，单指令多线程）**：

一批线程（**warp**，通常 32 个）**共享一条指令流**；
32 个线程对各自的**数据**同时执行同一条指令；
如果线程分支分叉，warp 会**串行执行**每个分支再合并——这就是「发散（divergence）」性能杀手。

关键数字感受一下：

一块旗舰 GPU 有上万流处理器核，可以同时驻留数十万个线程。

它们靠**切换线程来掩盖延迟**：

这个 warp 等内存时，调度器立刻切到另一个 warp 算。

<span class="marginnote">SIMT 与 SIMD 不同：SIMD 是单线程内的向量数据并行（如 AVX），SIMT 是整批线程的并行。GPU 的 warp 更像「一群按 SIMD 步调走路的线程」。</span>

## 2 CUDA 编程模型：host/device 与 kernel

CUDA 把计算分成两侧：

- **host（主机）**：CPU 及系统内存，负责控制、数据搬运；
- **device（设备）**：GPU 及其显存，负责大规模并行计算。

程序员写一个**内核函数（kernel）**，用 `__global__` 声明，在 GPU 上执行：

```c
__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)                    // 越界检查：保证最后一块不越界
        c[i] = a[i] + b[i];
}
```

调用时指定网格与块的大小：

```c
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```

**辨析｜易错点：** kernel 里的 `printf` 不能替代调试器，`越界检查`（if (i < n)）必须保留——当 `n` 不是 `blockDim.x` 的整数倍时，最后一块的线程会越界。

## 3 线程组织：grid / block / thread

CUDA 的线程是三层嵌套结构：

- **线程（thread）**：最小编程单元；
- **线程块（block）**：一组线程，块内可共享内存、可同步；
- **网格（grid）**：一组块，对应一次 kernel 启动。

每个线程的身份由三个内建变量定位：

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

- `threadIdx.x`：线程在块内的编号；
- `blockDim.x`：一个块有几个线程；
- `blockIdx.x`：本块在网格里的编号。

这个公式是 CUDA 的「身份证」：

**每个线程用它算出自己该处理哪个数据元素。**

<span class="marginnote">块的大小（如 256）是调优参数：太小则调度开销大，太大则超过硬件上限。块内线程数通常取 32 的倍数（warp 对齐）。</span>

## 4 内存层次：从寄存器到全局内存

CUDA 的内存按「速度 × 容量 × 作用范围」分五层：

| 内存 | 作用范围 | 速度 | 说明 |
| --- | --- | --- | --- |
| 寄存器 | 单线程 | 最快 | 线程私有，容量极小 |
| 共享内存 | 块内线程 | 快 | 块内协作的「手写缓存」 |
| 全局内存 | 全部线程 | 慢 | 显存主体，kernel 的主战场 |
| 常量/纹理 | 只读 | 快 | 有专用缓存与硬件优化 |

**共享内存**是 CUDA 性能的灵魂：

块内线程用 `__shared__` 声明的变量互相对话，不用反复读写慢速全局内存。

**全局内存**的访问要讲「**合并访问（memory coalescing）**」：

同一个 warp 里 32 个线程的地址**连续**，硬件就能合成少数几次宽传输；地址七零八落，就退化成几十次小传输，带宽腰斩。

**公式解析：** 一次合并访问的传输次数：

$$T_{\text{trans}} = \left\lceil \frac{\text{字节数}}{128} \right\rceil$$

- **第一步，看分子**：warp 访问的字节跨度；
- **第二步，看分母**：现代 GPU 一次内存事务约 128 字节；
- **第三步，做除法**：跨度超过 128 字节，传输次数成倍增加。

**优化铁律：让相邻线程访问相邻地址。**

这正是「线程 i 处理元素 i」比「线程 i 处理元素 i 的某种打乱」快得多的原因。

## 5 代码解析：完整的向量加法

host 侧负责数据搬运与 kernel 启动：

```c
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, n * sizeof(float));           // 在显存里分配
cudaMalloc(&d_b, n * sizeof(float));
cudaMalloc(&d_c, n * sizeof(float));

cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);  // 拷入
cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);   // 启动 kernel
cudaDeviceSynchronize();

cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);  // 拷回
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
```

三步读懂：

- **第一步，`cudaMalloc` + `cudaMemcpy`**：数据从 host 搬进显存，算完再搬回——搬运本身很贵，好程序会尽量减少往返；
- **第二步，`vecAdd<<<gridSize, blockSize>>>`**：启动 kernel，网格大小由数据量除以块大小向上取整；
- **第三步，校验**：把结果拷回 CPU 与串行结果对比——**并行程序的第一条守则：写对校验再谈性能**。

<span class="marginnote">`cudaDeviceSynchronize()` 是显式同步点：它保证之前的所有 kernel 都已完成。异步流（stream）技术可以重叠拷贝与计算，把搬运时间藏进计算时间里。</span>

## 6 核心对比：CPU 与 GPU

| 维度 | CPU | GPU |
| --- | --- | --- |
| 设计目标 | 低延迟（单线程快） | 高吞吐（线程海量） |
| 核心数 | 几个到几十个 | 几千到上万 |
| 线程切换 | 重（上下文切换） | 轻（warp 轮换） |
| 适用 | 串行控制流、复杂逻辑 | 大规模数据并行 |
| 内存 | 大缓存、低延迟 | 高带宽、高延迟隐藏 |

**核心结论：** CPU 是「少数精兵」，GPU 是「人海战术」。

两者不是替代关系——现代超算用 CPU 管控制流与稀疏逻辑，用 GPU 扛稠密数值计算，各司其职。

## 7 小结

- GPU 靠 **SIMT** 与海量线程换吞吐，**warp（32 线程）**共享指令流。
- CUDA 分 **host/device**，kernel 用 <<<grid, block>>>` 启动。
- 线程身份公式：$i = \text{blockIdx} \times \text{blockDim} + \text{threadIdx}$。
- 内存五层，**共享内存**管块内协作，**全局内存**要合并访问。
- 铁律：**先写对校验，再谈性能；相邻线程访问相邻地址。**

在下一节，我们把 MPI 与 OpenMP/CUDA 拼起来：学习**混合并行**，让超算的每一层硬件都被用满。
