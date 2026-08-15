---
title: 统一内存与新特性（页迁移、动态并行、协作组）
date: 2026-08-07
---

# 统一内存与新特性（页迁移、动态并行、协作组）

<div class="epigraph">
<p>计算机科学的任何问题，都可以用增加一层间接层来解决。</p>
<footer>—— 大卫 · 惠勒（David Wheeler）</footer>
</div>

<div class="article-byline">
<p>第四级 · GPU 架构与 CUDA 并行编程 ｜ Kirk & Hwu, Programming Massively Parallel Processors, 4e, Ch21；CUDA C++ Programming Guide（统一内存/动态并行/协作组） ｜ 2026-08-07</p>
</div>

## 为什么从统一内存开始

前几篇的模型里，host 与 device 各有一块内存，程序员要手动 `cudaMemcpy`——正确但繁琐，而且拷贝是性能大坑。**统一内存（unified memory）** 想做的，就是惠勒这句话的又一次实践：在 host 与 device 之上再垫一层抽象，让「一个指针」在两套内存间自动流转。

这一篇讲的是 CUDA 的「新特性层」：统一内存、动态并行、协作组。它们不是每个程序都需要，但理解它们，等于理解了「现代 CUDA 如何在易用性与性能之间做取舍」，也为阅读《性能分析与调优》《多 GPU 编程》提供现代语境。<span class="marginnote">PMPP 第 21 章 "CUDA Dynamic Parallelism" 专讲「kernel 里启动 kernel」；统一内存与协作组的官方定义见《CUDA C++ Programming Guide》。第 22 章则把「易用性 vs 性能」的权衡总结为未来趋势——正是本篇的三观。</span>

## 1 统一内存：一个指针，两套内存

传统 CUDA 里，host 指针只能被 CPU 用，device 指针只能被 GPU 用，数据要跨边界必须显式拷贝。**统一内存**用 `cudaMallocManaged` 分配一块「托管内存」，这块内存在 host 与 device 之间共享一个虚拟地址：

```c
float *x;
cudaMallocManaged(&x, N * sizeof(float));   // 一块两边都能用的内存
init(x);                                    // CPU 侧读写
kernel<<<grid, block>>>(x, N);              // GPU 侧直接用同一指针
cudaDeviceSynchronize();                    // 等 GPU 用完
```

对 GPU 而言，托管内存就是普通的全局内存——**同一份代码、同一个指针，CPU 和 GPU 都能用，不再需要手动拷贝。** 它特别适合：结构体含指针的数据、代码要反复在 host/device 之间横跳、以及「想先跑对再说性能」的原型阶段。<span class="marginnote">重要背景：CUDA 11 起，Linux 上支持 <strong>HMM（Heterogeneous Memory Management，异构内存管理）</strong>，统一内存可以真正借用系统内存并双向迁移，能处理「显存放不下」的大数据集——这是统一内存在大模型推理、图分析里越来越常见的原因。</span>

## 2 页迁移：数据在需要时才搬

统一内存不是「魔法复制」，它背后是**按需页迁移（on-demand page migration）**：

- GPU 首次访问某段托管内存时，触发**缺页（page fault）**，运行时把该页从主存搬进显存。
- GPU 用完、CPU 又要读时，再搬回——搬移以页（典型 4 KB 或 2 MB 大页）为单位。

**代价是隐形的性能税**：缺页要等搬运完成（几十到几百微秒），频繁跨端访问会带来「页抖动」。所以统一内存的工程铁律是：**数据在哪端就用哪端，别让同一份数据在两端之间反复横跳。** 现代运行时会尽量把「谁在用」记下来做迁移决策，但程序员仍然应该尽量让数据「待」在主要消费它的那一端。

## 3 核心对比表：显式拷贝 vs 统一内存

这是「易用性 vs 性能」权衡的典型战场：

| 维度 | 显式拷贝（cudaMemcpy） | 统一内存（cudaMallocManaged） |
| --- | --- | --- |
| 代码复杂度 | 高：要管理多套指针与拷贝时机 | 低：一个指针到处用 |
| 性能可预测性 | 高：拷贝开销在明处 | 低：缺页迁移开销在暗处 |
| 大数据集 | 受显存容量限制 | 可超售（借用主存） |
| 跨端频繁访问 | 每端数据都放在对的地方 | 易页抖动 |
| 推荐场景 | 性能关键、数据流动清晰 | 原型、结构复杂、数据集超大 |

**关键结论：没有「统一内存更好」这回事，只有「更省心」与「更可控」的分工。** 性能关键路径用显式拷贝，复杂/超大数据用统一内存——两者都是正经的工程选择，不是新旧替代。<span class="marginnote">实战里最常见的最优解是「混合」：控制路径用统一内存简化代码，计算热路径用显式拷贝把数据钉在正确的一端。Nsight 会直接报告页迁移次数与缺页开销，帮你看清税交在哪。</span>

## 4 动态并行：GPU 自己决定并行度

有些问题「递归」到让人头皮发麻：网格加密、自适应细分、多体问题里的层级——**工作负载的形状在运行前根本不知道**。动态并行（dynamic parallelism）就是答案：**GPU kernel 里可以再启动 kernel**，让「并行度」由 GPU 在运行时自己决定。

```c
__global__ void recurse(int depth, float *data) {
    if (depth == 0) { process(data); return; }
    recurse<<<2, 128>>>(depth - 1, data);   // kernel 里启动 kernel
}
```

动态并行解决了一类「CPU 必须频繁回流、再启动」的问题，但也带来新约束：**递归深度与内核数量受限、嵌套启动开销大、调试困难。** PMPP 第 21 章给出它的完整规则（grid 同步、嵌套深度限制、内存约束）。<span class="marginnote">辨析｜易错点：动态并行不是「随便递归」。每次嵌套启动都有固定开销，且嵌套 kernel 的资源要提前预留——把「递归」当默认工具用，往往会踩进「慢 + 挂死」的坑。多数自适应算法用「CPU 分批调度」也能实现，动态并行只在「运行期才知道规模」时才真正划算。</span>

## 5 协作组：把同步玩出细粒度

传统 CUDA 只有一种同步：`__syncthreads()`（整个 block）。但很多算法需要**更小或更大的同步粒度**。**协作组（cooperative groups）** 提供了可编程的同步与协作单元：

- `thread_block`：整块同步，等价于 `__syncthreads()`，但更清晰。
- `thread_block_tile<32>`：把 block 切成 32 线程的 tile，做 **tile 内同步**（比全块同步更轻、更局部）。
- `grid_group` / 协作 launch：让**整个 grid** 可以同步（需 `cudaLaunchCooperativeKernel` 与特殊资源保证）。
- `this_thread_block().sync()`：显式调用，语义更明确。

协作组的价值在于：**让「同步」的粒度匹配「算法的依赖结构」**——只需要邻居同步时不必整个 block 停下来。它是现代 CUDA 里「显式管理协作」的正式化版本，也是《共享内存与 bank conflict、同步原语》篇那种「手动协作」思想的 API 化。<span class="marginnote">协作组还支持跨 warp 的洗牌（shuffle）、归约等原语，很多现代库（如 CUB、Thrust）内部就用它做高性能原语——读这类源码时认识协作组能少绕很多弯。</span>

## 6 什么时候用、什么时候别用

把本专题的「现代特性」收成一份决策清单：

- **统一内存**：数据流复杂、原型验证、显存放不下大集 → 用；性能关键路径且数据流动清晰 → 用显式拷贝。
- **动态并行**：运行期才知道工作形状、层级递归 → 考虑；能被 CPU 分批调度替代 → 别用它。
- **协作组**：需要 tile 级/网格级同步、写通用并行原语 → 用；只需全块同步 → 传统 `__syncthreads` 足够。

**一条总原则：先让程序正确，再用分析器看数据说话。** 这些「新特性」的本质都是「易用性换更多调度自由度」，而调度自由度是否值得，最终要用 Nsight 的实测回答——这正是下一篇文章的主题。

## 7 主动管理：prefetch 与 advice

统一内存的「按需迁移」是省心，但省心也有代价——缺页是「发现了才搬」，首次访问总要等。要把它调优到接近显式拷贝的性能，CUDA 给了两个主动控制旋钮。

**第一个旋钮：预取（prefetch）**。`cudaMemPrefetchAsync(ptr, size, deviceId)` 在 GPU 计算开始前，主动把数据搬到目标设备，把「缺页等待」从计算路径里挪到准备阶段：

```c
cudaMemPrefetchAsync(x, N * sizeof(float), cudaCpuDeviceId); // 预取到 CPU
// ... CPU 填充数据 ...
cudaMemPrefetchAsync(x, N * sizeof(float), deviceId);        // 预取到 GPU
kernel<<<grid, block>>>(x, N);                               // 不再缺页等待
```

**第二个旋钮：访问建议（advice）**。`cudaMemAdvise(ptr, size, advice, deviceId)` 告诉运行时数据的访问模式：

- `cudaMemAdviseSetReadMostly`：这块数据基本只读（如模型权重）——运行时把它复制到每张卡，避免反复迁移。
- `cudaMemAdviseSetAccessedBy`：某设备将高频访问——提前把页放在那端。

**辨析｜易错点：** 预取与建议是「优化」，不是「正确性」——加了它们，结果必须不变；若数据在两端都频繁写，prefetch 反而引发更多迁移抖动。**它们真正的价值在「数据生命周期明确」的场景**（一次搬入、长期只读）：训练时的权重、推理时的 KV 缓存，都是典型目标。<span class="marginnote">一个直觉类比：按需迁移是「按需调货」，prefetch 是「提前备货」，advice 是「告诉仓库这货只在哪个门店用」。对生命周期清晰的大块数据，主动备货能把这套机制从「省心但慢」变成「又快又省心」。</span>

**工程结论：统一内存的默认行为是「正确优先」，prefetch/advice 是「性能可达」的钥匙。** 用上它们，托管内存在大数据集场景的差距会大幅缩小——甚至在只读大权重场景超过显式拷贝（省掉了手动拷贝的往返），这也是它在大模型推理、图分析里越来越常见的原因。

## 8 小结

- **统一内存**用 `cudaMallocManaged` 让一个指针两端可用，靠**按需页迁移**隐式搬数据；省心但藏了缺页开销。
- 显式拷贝与统一内存是**分工而非替代**：性能关键路径用前者，复杂/超大场景用后者。
- **动态并行**让 kernel 内再启动 kernel，解决运行期才知形状的递归问题，但有深度与开销限制。
- **协作组**提供 tile 级、网格级同步，让同步粒度匹配算法依赖。
- 决策总原则：先正确、再实测；新特性的取舍靠 Nsight 数据说话。

在下一节，我们把前面的所有量化工具收拢成一套方法：**性能分析与调优（Nsight、Roofline、内存/计算受限判定）**。
