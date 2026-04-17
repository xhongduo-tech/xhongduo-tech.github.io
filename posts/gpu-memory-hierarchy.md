## 核心结论

GPU 内存优化的核心不是“让计算更快”，而是“让数据尽量停留在更近的层次里”。这里的“层次”可以理解为数据离计算单元有多近：越近，延迟越低，带宽通常越高，但容量越小。

对大多数 CUDA 初学者，最重要的结论有三条：

1. `Register` 是每个线程私有的最快存储，适合放当前线程反复使用的标量和小片数据。
2. `Shared Memory` 是同一个线程块共享的片上内存，可以把它理解为“程序员手动管理的 L1 缓存”，适合做 tile 分块、数据复用、线程协作。
3. `Global Memory` 通常落在 HBM 上，容量最大但延迟最高，必须依赖访问合并（memory coalescing）和缓存命中来摊薄代价。

下面这张表先建立整体直觉。数值是常见数量级，不同架构会有差异，但优化方向不变。

| 层次 | 白话解释 | 典型可见性 | 典型容量级别 | 典型延迟级别 | 优化重点 |
|---|---|---|---|---|---|
| Register File | 线程手边的小本子 | 单线程私有 | 最小 | 约 1 cycle | 减少溢出、提高复用 |
| Shared Memory / L1 | 线程块内部共用工作台 | 单个 block / 单个 SM | 小 | 约 20~30 cycles | tile、复用、避开 bank conflict |
| L2 Cache | 全芯片共享缓存 | 所有 SM 可见 | 中 | 约 200 cycles | 提高局部性、减少 HBM 往返 |
| Global Memory / HBM | 显存主仓库 | 全局可见 | 最大 | 约 400~1000 cycles | 合并访问、减少事务数 |

一个玩具例子足够说明差异：一个 warp 有 32 个线程，如果每个线程各读一个连续的 `float`，总共正好是 $32 \times 4 = 128$ 字节，通常可以压成一次 128B 事务；如果每个线程跨很大步长去读，硬件可能需要拆成很多次事务，延迟和带宽都明显恶化。

真实工程里最经典的例子是矩阵转置或矩阵乘法。直接按列访问全局内存，通常会打破合并访问；把数据先按 tile 读入 shared memory，再在 shared 中变换访问方向，就能把“高延迟、低效率”的 HBM 访问变成“少次数、可复用”的访问模式。

---

## 问题定义与边界

问题可以定义为：在 GPU 上，同样的算术计算，为什么有的 kernel 很快，有的却慢几个数量级？

答案通常不在 ALU，而在访存路径。GPU 的并行度很高，但每一层内存的速度、容量和共享范围都不同。你面对的是一个三角约束：

- 速度：越靠近计算单元越快。
- 容量：越快的层次通常越小。
- 可见性：越快的层次通常越局部，只对线程自己或线程块可见。

因此，优化目标不是“所有数据都放到最快的地方”，因为放不下；真正目标是：

- 让热点数据尽量留在 register 或 shared memory；
- 让访问 global memory 时尽量合并；
- 让 shared memory 的访问不发生或少发生 bank conflict；
- 在寄存器占用、shared memory 占用、occupancy 之间做平衡。

这里先给一个边界清晰的对比例子。

假设一个 warp 读取 `float a[ ]`：

- 模式 A：线程 `t` 读取 `a[base + t]`
- 模式 B：线程 `t` 读取 `a[base + 32*t]`

模式 A 是连续访问，通常可以合并；模式 B 是大步长访问，每个线程都跳到不同 cache line，事务数会显著增加。虽然两者“每个线程都只读一次”，但硬件看到的是完全不同的总线行为。

可以把 GPU 内存可见性理解成下面这个结构：

| 层次 | 谁能看到 | 典型用途 |
|---|---|---|
| Register | 只有当前线程 | 局部变量、累加器 |
| Shared Memory | 当前 block 的所有线程 | tile、协作加载、中间缓存 |
| L2 Cache | 所有 SM 间共享命中 | 缓存全局数据、降低 HBM 访问 |
| Global Memory | 所有线程和主机都可经接口访问 | 输入、输出、大规模参数 |

边界也必须说清楚：

- 不是所有算法都适合 shared memory 优化。若数据复用很弱，手动搬运的收益可能不够覆盖同步与占用成本。
- 不是 shared memory 越大越好。它会挤占 SM 资源，降低并发 block 数。
- 不是寄存器越多越好。寄存器过多会导致 register spill，也就是编译器把本应在寄存器里的变量溢出到更慢的 local/global 路径。

---

## 核心机制与推导

先看全局内存的合并访问。合并访问的白话解释是：一个 warp 的线程如果访问足够连续，硬件可以把它们打包成更少的大事务，而不是 32 个分散小事务。

对 `float` 数组，若线程索引为 `threadIdx.x`，最理想的地址模式是：

$$
addr(t) = base + t \cdot sizeof(float)
$$

对一个 warp 的 32 个线程，总访问范围是：

$$
32 \cdot 4 = 128\text{B}
$$

如果 `base` 对齐合理，这 32 次逻辑加载通常可以落入一个 128B cache line 或少量相邻事务中。反过来，如果地址模式是：

$$
addr(t) = base + t \cdot stride \cdot sizeof(float)
$$

当 `stride` 很大时，不同线程会分散到多个 cache line，事务数上升，带宽利用率下降。

下面用一个玩具例子说明：

- 连续访问：线程 0~31 读取 `a[0]` 到 `a[31]`
- 跨步访问：线程 0~31 读取 `a[0], a[32], a[64], ...`

前者像 32 个人排队一次性领货；后者像 32 个人跑去 32 个不同仓位，各自取一次。数据量相同，但总调度成本完全不同。

再看 shared memory 的银行冲突。bank 可以理解为 shared memory 内部并行服务的“通道组”。现代 GPU 常见是 32 个 bank。一个常见近似公式是：

$$
bankIdx = \left(\frac{addr}{4}\right) \bmod 32
$$

这里假设按 4 字节字宽映射。若同一时刻多个线程访问落在同一个 bank 上，而且这些访问不能广播，就会出现 bank conflict。冲突越高，访问越接近串行化。

最经典的冲突场景是二维 tile 的列访问。假设声明：

```cpp
__shared__ float tile[32][32];
```

按行访问时，连续线程通常落在不同 bank，问题不大；但按列访问时，每个线程访问 `tile[threadIdx.x][fixed_col]`，相邻线程地址相差一个整行跨度：

$$
stride = 32 \times 4 = 128\text{B}
$$

代入 bank 编号后，低位模式重复，容易形成 32-way conflict。

解决办法是 padding，也就是额外补一列：

```cpp
__shared__ float tile[32][33];
```

这时列访问的行跨度变成：

$$
stride = 33 \times 4 = 132\text{B}
$$

对应 bank 步进从 32 变为 33，而

$$
33 \bmod 32 = 1
$$

这意味着相邻线程在列访问时会轮流落到不同 bank，从高冲突退化为接近无冲突的模式。

这就是为什么 `tile[32][32+1]` 是 CUDA 教材和最佳实践里反复出现的写法。多出来的一列不参与有效数据计算，只负责打乱 bank 映射。

把上面两件事合起来，shared memory 的真正价值就清楚了：

1. 从 global memory 连续读入，保证 coalescing。
2. 在 shared memory 中重排访问方向，避免重复全局访问。
3. 用 padding 避免 shared 内部的 bank conflict。
4. 最后再以合并友好的方式写回 global memory。

这本质上是在做“少量昂贵搬运，换大量便宜复用”。

---

## 代码实现

先给一个可运行的 Python 小程序，用来模拟 bank 映射。它不是 CUDA 代码，但能帮助初学者验证为什么 `32` 会冲突、`33` 会分散。

```python
def bank_indices(stride_words: int, num_threads: int = 32, num_banks: int = 32):
    # 假设每个线程访问同一列，不同行，地址间隔为 stride_words 个 4B word
    return [(t * stride_words) % num_banks for t in range(num_threads)]

banks_32 = bank_indices(32)
banks_33 = bank_indices(33)

# stride=32 时，所有线程都落到同一个 bank
assert len(set(banks_32)) == 1
assert banks_32[:4] == [0, 0, 0, 0]

# stride=33 时，32 个线程正好打散到 32 个 bank
assert len(set(banks_33)) == 32
assert banks_33[:4] == [0, 1, 2, 3]

print("stride=32:", banks_32[:8])
print("stride=33:", banks_33[:8])
```

再看一个典型的 CUDA tile 结构。这里用矩阵转置作为例子，因为它同时展示了 global coalescing 和 shared padding。

```cpp
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeCoalesced(float* out, const float* in, int width, int height) {
    // +1 的目的不是存更多数据，而是打破 shared memory 的 bank 冲突
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 从 global memory 连续加载到 shared memory
    // 同一 warp 中 threadIdx.x 连续，通常可形成 coalesced load
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && y + j < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }

    __syncthreads();

    // 2. 交换 block 坐标，准备写出转置后的结果
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. 从 shared memory 读出并写回 global memory
    // 因为做了 padding，列方向读取 tile 时不会形成严重 bank conflict
    // 写回时 threadIdx.x 仍然连续，通常可形成 coalesced store
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && y + j < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

这个 kernel 的流程可以概括为：

```text
global 连续读
-> shared 暂存
-> block 内同步
-> shared 改变访问方向
-> global 连续写
```

玩具例子可以设成 `32x32` 的小矩阵：

- 若直接从输入矩阵按列读，再按行写，global 读会变成大步长访问；
- 若先按行读入 `tile`，再在 shared 里转向，global 侧依然保持行方向连续；
- 若 `tile[32][32]` 不加 padding，shared 的列访问容易冲突；
- 若改成 `tile[32][33]`，冲突显著减轻。

真实工程例子是矩阵乘法。常见写法是把 A、B 的一个 tile 拉入 shared memory，再让每个线程在寄存器里累加多个乘加结果。这样做有三个收益：

- A、B 的每个元素被 block 内多个线程复用；
- HBM 访问次数显著减少；
- 中间累加器放在寄存器，不必每一步都写回 shared 或 global。

如果你把线程索引展平成线性 id，也要关注 warp 组织方式。常见写法：

```cpp
int linear_tid = threadIdx.x + threadIdx.y * blockDim.x;
```

这样更容易看清某一组 32 个线程是否在访问连续地址，从而判断 coalescing 是否成立。

---

## 工程权衡与常见坑

shared memory 优化不是“必赚不赔”，而是典型的工程权衡。你多拿一点 shared memory，SM 上可同时驻留的 block 可能就少一点；你多用一点寄存器，occupancy 可能下降，甚至发生 spill。

下面用表格列出最常见的问题。

| 问题 | 影响 | 常见现象 | 规避手段 |
|---|---|---|---|
| Global 访问不连续 | 事务数增加，带宽利用率低 | kernel 吞吐远低于理论值 | 按 warp 连续布局数据，优先让 `threadIdx.x` 映射连续地址 |
| Shared 列访问无 padding | bank conflict，shared 访问近似串行化 | 明明搬到 shared 了还是不快 | 用 `TILE_DIM + 1` 等 padding 打乱 stride |
| Shared 用量过大 | occupancy 降低 | 每个 SM 同时活跃 block 变少 | 减小 tile，评估实际复用收益 |
| 寄存器过多 | spill 到 local/global，性能恶化 | 编译输出寄存器占用过高 | 控制展开规模，检查编译报告 |
| 频繁 `__syncthreads()` | 同步开销上升 | 算法结构被 barrier 卡住 | 只在真正需要 block 协作时同步 |
| 误以为 L1/L2 会自动救场 | 实际局部性不足时收益有限 | 数据模式随机时缓存命中差 | 主动做 tile，减少不规则访问 |

这里给一个常见坑的具体说明。

在矩阵乘法里，如果你写：

```cpp
__shared__ float Bs[32][32];
```

并让线程在乘法内环中频繁按列访问 `Bs[k][tx]` 或相关变体，可能会触发明显冲突。很多初学者看到“shared 比 global 快”就以为一定更快，但 shared 不是零成本。它快，前提是访问模式也要适合它的物理组织。

另一个坑是过度追求 occupancy。occupancy 的白话解释是“SM 上同时挂了多少活跃线程/warp”。它重要，但不是唯一目标。若你为了提高 occupancy 把 tile 缩得太小，导致数据复用下降，总性能可能反而更差。实际工程里更常见的做法是：

- 先保证 global 访问合并；
- 再消除明显 bank conflict；
- 再看 shared / register 占用是否让并发度掉得过多；
- 最后根据 profiling 决定是否调 `cudaFuncSetCacheConfig` 或 kernel 配置。

真实工程里，NVIDIA 在矩阵转置案例中展示过：仅通过合并访问和 shared padding，带宽可以从很低的水平跃升到接近设备可用带宽上限的区间。这个例子说明，访存模式往往比单纯“多写几行算术优化”更关键。

---

## 替代方案与适用边界

不是所有场景都应该手写 shared memory tile。下面给出三个常见方案的边界。

| 方案 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| Shared tile | 有明显数据复用，线程块内需要协作 | 控制力强，收益高 | 代码复杂，吃 shared 资源 |
| Register tile | 每线程可持有少量热点数据 | 延迟最低，适合累加器 | 容易增加寄存器压力 |
| L2-only / 直接访问 | 复用弱、访问已较连续 | 实现简单 | 对缓存命中和硬件行为依赖更强 |

先说 shared tile。它适合：

- 矩阵乘法、卷积、转置、stencil 等有块状局部性的算法；
- 一个 block 内多个线程会反复读同一批数据；
- 你能清楚设计加载、同步、复用、写回四个阶段。

再说 register tile。register tiling 的白话解释是：让每个线程一次处理不止一个输出元素，把多个中间结果都保存在寄存器里。例如每个线程处理 `2x2` 输出块，而不是 `1x1`。这样做可以减少 shared 或 global 的往返，但代价是寄存器占用升高。

一个简化思路是：

- shared memory 只放当前阶段全线程共享的数据；
- 每线程自己的多个累加器全部放寄存器；
- 线程间交换少量数据时，优先考虑 warp-level primitive。

如果 shared memory 预算过紧，这往往比盲目扩大 tile 更稳妥。

最后是 L2-only。若你的访问已经天然连续，而且几乎没有 block 内复用，那么手动搬到 shared memory 可能只是增加额外 load/store 和同步。比如一些纯流式处理场景：

- 每个元素只读一次、算一次、写一次；
- 相邻线程本来就访问连续地址；
- 数据复用极弱。

这时保持全局访问合并，尽量让 L2/L1 自动缓存发挥作用，可能就是更好的工程选择。

因此，适用边界可以概括成一句话：

- 有明显复用，用 shared tile；
- 有线程私有的小规模复用，用 register tile；
- 几乎无复用，但访问已连续，就让 L2 和合并访问解决问题。

---

## 参考资料

1. NASA HECC, *Basics on NVIDIA GPU Hardware Architecture*  
   结论摘要：给出 GPU 内存层次、容量与典型延迟数量级，适合作为 register / shared / L2 / HBM 的整体参照。

2. NVIDIA, *CUDA C Best Practices Guide 12.2*  
   结论摘要：系统说明 coalescing、shared memory、padding 与矩阵转置等案例，展示 `TILE_DIM+1` 的工程价值。

3. AMD ROCm HIP Docs, *Performance Optimization*  
   结论摘要：解释 memory coalescing 与 bank conflict 的通用原理，虽然面向 HIP，但访存模式分析对 CUDA 同样适用。

4. NVIDIA, *CUDA Programming Guide*  
   结论摘要：补充 warp、memory transaction、shared memory 组织和同步语义，是理解底层行为的官方定义来源。

5. 相关矩阵转置与矩阵乘法优化案例文档  
   结论摘要：展示“先合并 global，再在 shared 中重排”的通用套路，以及 padding 对带宽提升的实际影响。
