---
title: Shared Memory 编程与 __syncthreads() 同步语义
date: 2026-08-07
---

# Shared Memory 编程与 `__syncthreads()` 同步语义

<div class="epigraph">
<p>分布式系统是这样一种系统：一台你根本不知道它存在的计算机出错，能让你的计算机没法用。</p>
<footer>—— 莱斯利 · 兰波特（Leslie Lamport）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第一篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Shared Memory 编程开始

上一课我们认识了 Shared Memory 的 bank 结构：它快，但也「娇气」——访问模式不对就会 bank conflict。这一课把它真正用起来：**如何声明、如何读写、如何同步**。Shared Memory 是一切「先搬块、再算块」算法（tiling 矩阵乘、归约、scan、图像卷积）的地基，也是第五篇《显存优化》里「激活重计算」等技巧在数据侧的先修课。<span class="marginnote">兰波特这句话本来说的是分布式系统，但对 Shared Memory 编程同样成立：<strong>一个你不知道何时写完的线程，能让你的结果出错</strong>。共享内存的本质是把「跨线程可见」变成免费的东西——代价是同步的责任从硬件移交给了你。理解 `__syncthreads()` 的语义，就是在理解「并发正确性」这件事本身。</span>

## 1 一块能精确控制的快速内存

回顾内存地图：Shared Memory 是每个 block **独占**的片上 SRAM——A100 每个 SM 最多 164KB，H100 到 228KB。相比 L1 缓存靠硬件自动命中，Shared Memory 的**内容、布局、生命周期全部由你决定**，这是它的确定性优势，也是它的使用门槛。

两个数字记一辈子：

- **快**：聚合带宽几十 TB/s，访问延迟约 20–30 周期，比全局内存（400–800 周期）快一个数量级。
- **小**：一个 block 能用的只有几十到两百 KB，必须**像管理小旅馆一样精打细算**。

典型用法是**分块（tiling）**：把大矩阵切成 tile，先把一个 tile 从全局内存搬进 Shared Memory，block 内所有线程反复复用，全局访问次数从「每个输出读两次矩阵」降到「每块只读一次」——这正是减少上一课那个 $B$ 值、把内存瓶颈打成计算瓶颈的标准动作。<span class="marginnote">大模型推理里的 KV Cache 分块管理、attention 里的 tile 化处理（FlashAttention 的 core 思想之一），本质上都是同一招：<strong>把「全局 → 片上」的搬运次数降到最低，再在片上做高复用计算</strong>。这条思路会一直延伸到第九篇《推理基础设施》。</span>

## 2 `__shared__` 声明：静态与动态

Shared Memory 有两种声明方式。

**静态分配**——编译期大小固定，写在 kernel 内或文件作用域：

```cpp
__global__ void tile_matmul(...) {
    // 每个 block 私有，256 个 float = 1KB
    __shared__ float tile[16][16];
    // 直接读写，不需要 malloc / free
    tile[threadIdx.y][threadIdx.x] = value;
}
```

**动态分配**——大小在启动 kernel 时通过**第三个配置参数**给出，配合 `extern __shared__`：

```cpp
// 启动时指定动态 shared 大小（字节）
int smem_bytes = tile_dim * tile_dim * sizeof(float);
tile_matmul<<<grid, block, smem_bytes>>>(...);

// kernel 内声明为无大小的外部数组
__global__ void tile_matmul(...) {
    extern __shared__ float tile[];
    // 动态块内再自行切分布局
    float* a_part = tile;
    float* b_part = tile + tile_dim * tile_dim;
}
```

两种方式都受同一个上限约束：**每 block 可用的 Shared Memory 有最大值**（A100 上可通过编译选项放宽到 164KB，默认更小）。用超了，kernel 直接启动失败。**Shared Memory 不会自动清零**——里面是上一位使用者留下的数据，用前必须显式初始化，这是新手第一处翻车点。

## 3 `__syncthreads()`：给共享内存「立字据」

Shared Memory 是块内所有线程共享的，但**硬件不保证读到你刚写入的值**——除非你立字据。**`__syncthreads()`** 是一个 **block 级屏障（barrier）**：调用它的所有线程必须**全部到达**，才能继续前进。

它承诺两件事：

1. **同步**：block 内所有线程都执行到了这一行之后，程序才继续——先到的人等后到的人。
2. **可见性（内存栅栏）**：屏障之前每个线程对 Shared Memory（以及同一 block 的全局内存）的写入，在屏障之后对**所有**线程可见。

没有它，两个线程同时写同一块 Shared Memory、再互相读，结果**未定义**——可能读到旧值、垃圾值，甚至因为编译器重排读到你意想不到的东西。<span class="marginnote">这就像两个人共用一个白板：你写下结果、回身去喝水，对方看到的是你写了一半的字。<strong>`__syncthreads()` 就是那个「写完整、再转身」的约定</strong>。编译器不会替你猜——CUDA 的内存模型要求你显式同步，猜错了就是你背锅。</span>

注意它的边界：`__syncthreads()` 是 **block 级**的，只同步这一个 block 里的线程；**它绝不同步整个 grid**。跨 block 的全局同步需要 cooperative launch 或原子操作，代价昂贵，能避则避。

## 4 公式解析：树形归约的 $\log_2 n$ 步

Shared Memory 最经典的练兵场是**归约（reduction）**——把 $n$ 个数求和。串行是 $n - 1$ 次加法；在 GPU 上，可以让「每次相加的两方」并行，于是每一层把活跃线程减半：

$$
\text{层数} = \log_2 n
$$

对这条式子做三步拆解：

- **第一步，理解并行加法树**：第 1 层，$\frac{n}{2}$ 个线程两两相加，产生 $\frac{n}{2}$ 个部分和；第 2 层，$\frac{n}{4}$ 个线程再两两相加……每一步活跃线程减半。
- **第二步，数层数**：从 $n$ 个元素到 1 个结果，活跃线程每层减半，正好需要 $\log_2 n$ 层。对 $n = 1024$，$\log_2 1024 = 10$ 层；对 $n = 2^{20} = 1048576$，20 层。
- **第三步，看同步成本**：**每一层结束都必须有一次 `__syncthreads()`**，保证上一层写进 Shared Memory 的部分和，在下一层读之前全部就绪。所以总同步次数也是 $\log_2 n$——这正是「Shared Memory 归约 = 分治 + 每层立字据」的定量表达。

一个典型实现片段：

```cpp
__global__ void block_reduce(const float* in, float* out, int n) {
    __shared__ float s[1024];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (t < n) ? in[t] : 0.f;
    __syncthreads();                        // 全员写完后，才允许开始归约

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();                    // 每一层必须同步
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
```

注意每层 `stride` 减半时的两个细节：**只让前 `stride` 个线程干活**（其余线程空转但必须到达 `__syncthreads()`）；`threadIdx.x + stride` 这个访问模式在 stride 是 2 的幂时恰好无 bank conflict（上一课的结论在这里兑现）。<span class="marginnote">若用「线程 $t$ 与 $t + \text{blockDim}/2$ 配对」的另一种写法，归约也能跑，但访问模式会产生 2 路 bank conflict。哪种写法快，正是<strong>合并访存与 bank conflict 两条主线在真实 kernel 里的交汇</strong>。</span>

## 5 辨析｜易错点

- **「`__syncthreads()` 是全 grid 同步」**——错，它只同步**当前 block**。跨 block 同步要用 cooperative groups（`grid.sync()`），且要求 kernel 以 cooperative launch 方式启动，限制很多。
- **「在条件分支里调用 `__syncthreads()` 没问题」**——**死锁高危**。若 block 里只有部分线程走到这个分支，另一部分永远到不了屏障，整个 block 卡死。CUDA 规范称这是非法用法，后果未定义。上一课结尾已经预告过这个坑，这里是它的全貌。
- **「Shared Memory 用完自动清零 / 自动释放」**——不会清零（初值未定义）；生命周期随 block 结束自动结束，无需手动释放，但也意味着**数据不会跨 block 保留**。
- **「`__syncthreads()` 也同步全局内存、跨 block 可见」**——它含内存栅栏，但只对同一 block 有效；跨 block 的全局可见性需要原子操作或 cooperative sync。
- **「Shared Memory 越大越好」**——Shared 是**有限资源**，每 block 用得多，SM 上能驻留的 block 就少，**occupancy（占用率）随之下降**（下一课的主题）。用 32KB 省下来，往往比多缓存一点数据更值钱。

## 6 小结

- Shared Memory 是 **block 内共享、片上、程序员管理**的快速暂存，A100 / H100 每 SM 164 / 228KB，延迟 20–30 周期，比全局内存快一个数量级。
- 静态分配 `__shared__ float t[16][16];`；动态分配 `extern __shared__ float t[];` + 启动第三参数指定字节数；**初值未定义，用前必初始化**。
- `__syncthreads()` 是 **block 级屏障**：所有线程到达才放行，并保证屏障前写、屏障后读可见——共享内存的正确性全靠它。
- 树形归约把 $n-1$ 次串行加法压成 $\log_2 n$ 层，**每层一次同步**，是分治 + 显式同步的经典模板。
- Shared 是有限资源，用得多则 occupancy 低——性能是「少搬数据」与「多留并发」的权衡。

在下一节，我们把目光从「用什么存」转向「能并发多少」——**Occupancy（占用率）的计算与调优**：四个硬资源上限如何决定一个 SM 上能驻留多少个 block，以及为什么 100% 占用率不一定最快。
