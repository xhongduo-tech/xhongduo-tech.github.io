---
title: 矩阵乘法 kernel 优化实战：从 naive 到 tiling
date: 2026-08-07
---

# 矩阵乘法 kernel 优化实战：从 naive 到 tiling

<div class="epigraph">
<p>高性能计算的第一法则：把一切变成矩阵乘法。</p>
<footer>—— HPC 谚语（常被归于杰克 · 唐加拉 Jack Dongarra）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第一篇 ｜ 2026-08-07</p>
</div>

## 为什么从矩阵乘法开始

深度学习的主计算几乎就是矩阵乘法：全连接层是 $Y = XW$，Transformer 的 attention 是 $QK^\top$ 与 $PV$，卷积也常被重排成隐式 GEMM。可以说，**HPC 谚语「把一切变成矩阵乘法」在神经网络里字面为真**。这一课我们亲手写一个矩阵乘法 kernel，从最朴素、最「慢」的版本出发，用上一课学会的算术强度尺子诊断问题，再用 tiling 分块把它从访存瓶颈拉回计算瓶颈。<span class="marginnote">这一课是把前几课全部武器——合并访存、Shared Memory、$PV$、occupancy、启动开销——一次性集成到同一个 kernel 里。学完它，你就具备了读任何 CUDA 高性能代码的「心法」：<strong>先问瓶颈是计算还是访存，再决定往哪使劲</strong>。</span>

## 1 规模与理论：一次乘加要读两个数

设 $C = A \times B$，其中 $A$ 是 $M \times K$，$B$ 是 $K \times N$，$C$ 是 $M \times N$。每个输出元素 $C_{ij}$ 要累加 $K$ 次乘加：

$$
C_{ij} = \sum_{k=1}^{K} A_{ik} \cdot B_{kj}
$$

总计算量是

$$
\text{FLOPs} = 2MKN
$$

（$MKN$ 次乘 + $MKN$ 次加）。一个关键观察：**每个输出元素依赖 $A$ 的一整行与 $B$ 的一整列，共 $2K$ 个输入数**。如果每次都从全局内存取，那么「读两个数 → 做一次乘加」就成了内循环——访存量与计算量一比一，瓶颈在内存。

## 2 Naive kernel：为什么它跑在访存速度上

最朴素的想法：一个线程算一个输出元素，外层遍历 $K$。

```cuda
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];   // 每步从全局内存读两个数
        }
        C[row * N + col] = sum;
    }
}
```

这个版本**能跑对，但跑不快**。原因是内循环里每个线程都要从全局内存读 $A$ 与 $B$ 各一个数：每算一个输出元素，全局访问 $2K$ 个数。同一行的元素被不同线程反复读、同一列的元素也反复读，**没有任何复用**——数据在全局内存和计算单元之间来回搬运，搬运成了主旋律。<span class="marginnote">注意 naive 的访问模式本身并非「不合并」：naive kernel 里 col 是连续线程，是合并的；它的问题是<strong>总访问量太大</strong>，不是不合并。合并访存解决「一次取数传多少人」，tiling 解决「取一次数被算多少次」——两件事要分清。</span>

## 3 Tiling：先搬块、再算块

解决复用问题的标准答案是 **tiling（分块）**：把输出矩阵切成 $TILE \times TILE$ 的小块，**一个 block 负责算一块**。算一块之前，先把需要的 $A$ 的 $TILE \times K$ 子块和 $B$ 的 $K \times TILE$ 子块从全局内存搬进 Shared Memory，之后 block 内所有线程在片上反复复用这两块数据，而不是每次都回全局内存取。<span class="marginnote">这正是《Shared Memory 编程》里立下的规矩——<strong>「先搬块、再算块」，让数据在片上多待一会儿</strong>。复用因子是 $TILE$：一个从全局读出的 $A$ 元素，会被同一行的 $TILE$ 个输出共用。</span>

```cuda
#define TILE 32

__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 先搬块：A 的 TILE×TILE 子块与 B 的 TILE×TILE 子块进 Shared Memory
        int a_col = k0 + threadIdx.x;
        int b_row = k0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;   // 越界补零
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();                 // 同步 1：写 tile 之后必须同步

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();                 // 同步 2：内循环算完必须同步
    }

    __syncthreads();                     // 同步 3：写回 C 前，确保所有线程读完共享块
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

三处 `__syncthreads()` 的位置是正确性的命门：**写 tile 之后必须同步**（否则有人读到没写全的数据），**内循环算完必须同步**（否则下一轮 tile 覆盖了还没读完的旧 tile）。这正好兑现了《Shared Memory 编程》里「每层立字据」的纪律。

## 4 公式解析：算术强度从 0.25 到 8

**算术强度（arithmetic intensity）**：每从内存搬 1 字节，平均做多少次浮点运算，记作 $\text{AI} = \text{FLOPs} / \text{Bytes}$。它直接决定瓶颈类型：AI 低于机器的 **ridge point**（算力与带宽之比）则访存瓶颈，高于则计算瓶颈。

先算 **naive 版本**。每个输出元素读 $2K$ 个 float（$8K$ 字节），算 $2K$ 次浮点：

$$
\text{AI}_{\text{naive}} = \frac{2K}{8K} = 0.25 \ \text{flops/byte}
$$

再算 **tiled 版本**。一个 block 负责 $TILE \times TILE$ 个输出：搬入 $A$ 的 $TILE \times K$ 与 $B$ 的 $K \times TILE$，共 $2 \cdot TILE \cdot K$ 个 float（$8 \cdot TILE \cdot K$ 字节），计算 $2 \cdot TILE^2 \cdot K$ 次浮点：

$$
\text{AI}_{\text{tiled}} = \frac{2 \cdot TILE^2 \cdot K}{8 \cdot TILE \cdot K} = \frac{TILE}{4}
$$

做三步拆解：

- **第一步，看懂 naive 的 0.25**：分子分母的 $K$ 约掉了，只剩常数——无论矩阵多大，naive 的算术强度恒为 0.25，低得可怜。
- **第二步，看懂 tiled 的 $TILE/4$**：分子里 $TILE^2$（一个 block 算的输出数）被分母里的 $TILE$（一次搬入的元素数）抵消一次，留下 $TILE/4$。**分块越大，算术强度越高**。
- **第三步，对齐 ridge point**：以 A100 为例，FP32 峰值约 19.5 TFLOPS、HBM 带宽约 2 TB/s，ridge point $\approx \frac{19.5}{2} \approx 10$ flops/byte。naive 的 0.25 比它低两个数量级——**GPU 远远没吃饱，瓶颈在内存**；tiled 用 $TILE = 32$ 得 $\text{AI} = 8$，已逼近 ridge，$TILE = 64$ 得 16，正式跨入计算瓶颈。<span class="marginnote">把「AI 与 ridge 点对照」画成一张图，就是《Roofline 模型》——本篇第一篇的收尾篇会专门做定量分析。今天先记住结论：<strong>tiling 的实质是把 AI 从 0.25 抬到 ridge 之上</strong>。</span>

一个数字对照让复用更直观：naive 每取一个数只用一次；$TILE = 32$ 时每取一个数被复用 32 次。**全局访存总量因此下降了 $TILE$ 倍**——这比任何「更快的乘法」都值钱。

## 5 从 tiling 到实战：寄存器、向量化与更多

Shared Memory tiling 只是第一级台阶。真实的高性能 GEMM（如 cuBLAS、CUTLASS）在其上还要叠加几层：

**寄存器分块（register tiling）**：让每个线程算 $4\times4$ 或更多个输出元素，累加器留在寄存器里，进一步放大复用、减少共享内存访问。
**向量化访存**：用 `float4`（一次取 16 字节）搬运 tile，减少访存指令数，天然对齐。
**Bank conflict 规避**：shared 上做 padding 或转置，避免内循环访问模式踩中 bank。
**Double buffering**：用两条共享内存缓冲区，让「搬下一块」与「算当前块」重叠（上一课 Stream 的思想在块内复用）。

这些手段的收益评估，正是本专题后续《性能剖析》的标准演练；而 cuBLAS 之所以能把矩阵乘打成接近硬件峰值，靠的就是把上述每一层都压榨到位。<span class="marginnote">还有一个层次值得期待：<strong>Tensor Core</strong>——GPU 里专门算矩阵乘的硬件，FP16 下算力比 FP32 高一个数量级，下一课就讲它。</span>

## 6 辨析｜易错点

- **「naive 慢是因为不合并」**——不准确。naive 的问题首先是**访存总量太大**（AI = 0.25），合并访存只解决「一次取数传多少人」，解决不了「取了多少次」。
- **「`__syncthreads()` 放哪都行」**——错。它必须成对出现：写 tile 后、内循环算完后各一次；漏掉或放错位置，读到的就是旧值或未定义值，甚至死锁。
- **「TILE 越大越好」**——不是。TILE 增加占用的 Shared Memory 与寄存器也增加，超出后 occupancy 下降，SM 上驻留的 block 变少，并发度受损。TILE = 32 是 fp32 下常用的平衡点。
- **「矩阵边长不能被 TILE 整除就没法做」**——可以，用边界判断 + 补零（上面的 kernel 已示范 `越界` 分支补零）。代价是边界分支，主循环仍全速。
- **「tiling 之后一定算力打满」**——不一定。还要看内循环是否踩 bank conflict、occupancy 是否足够、以及有没有向量化；tiling 只是把瓶颈从「访存」挪到「计算」，剩下的效率要靠下一层优化。

## 7 小结

- 矩阵乘法总计算量 $2MKN$；每个输出元素依赖 $2K$ 个输入，naive 写法让「读两数算一次」成为内循环。
- naive 版本的**算术强度恒为 0.25 flops/byte**，远低于 A100 约 10 的 ridge point——内存瓶颈。
- **tiling** 用 block 算一块、先搬块再算块，算术强度提升到 $TILE/4$：$TILE=32$ 得 8、$TILE=64$ 得 16，跨越 ridge 进入计算瓶颈。
- 分块的本质是**复用**：每个从全局读出的数被 $TILE$ 个输出共用，全局访存总量下降 $TILE$ 倍。
- 实战 GEMM 还要叠加寄存器分块、向量化、bank conflict 规避与 double buffering。

在下一节，我们正式认识那个让矩阵乘法「快得不像话」的硬件单元——**Tensor Core**：它一次指令就能算完 16×8×8 的乘加，把 FP16 算力推高一个数量级，也是大模型训练的算力支柱。
