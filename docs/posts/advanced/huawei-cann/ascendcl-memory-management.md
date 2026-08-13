---
title: 内存管理：申请、释放与缓存策略
date: 2026-08-07
---

# 内存管理：申请、释放与缓存策略

<div class="epigraph">
<p>内存是唯一的资源，其余都是它的错觉。</p>
<footer>—— 安德鲁 · 塔能鲍姆（Andrew S. Tanenbaum，自操作系统名言）</footer>
</div>

<div class="article-byline">
<p>第四级 · 华为 CANN 计算架构 ｜ 华为昇腾 CANN 开发指南 ｜ 2026-08-07</p>
</div>

## 为什么从内存管理开始

资源就绪之后，第二道门是**数据上卡**——数据要住进设备内存才能被算子访问。昇腾的内存管理与普通 C/C++ 的 `malloc/free` 有一个关键差别：**它暴露了存储层级与缓存语义**。申请同一块内存，可以指定大页、指定是否走 L2 缓存、指定从哪个池分配。这些选择直接决定数据被哪个层级的存储服务、访问速度快不快。理解 AscendCL 的内存 API，其实是在理解「如何为你的数据选择正确的家」。<span class="marginnote">对照第三级《计算机组成原理》的存储器层次与 CUDA 的 `cudaMalloc/cudaMemcpy/cudaHostAlloc`：AscendCL 的内存模型在 API 层面高度相似，但多出了「缓存策略」「大页」这类更细的硬件感知选项——这正是专用 NPU 软件栈比通用 GPU 栈更贴近硬件的一个体现。</span>

## 1 主机内存与设备内存：两个世界

AscendCL 明确区分两类内存：

**主机内存（Host Memory）**：CPU 侧普通内存，用 `aclrtMallocHost` 申请、`aclrtFreeHost` 释放。它常被用作「数据暂存区」——模型输入先放这里，再搬到设备。
**设备内存（Device Memory）**：昇腾设备侧内存（HBM/DDR），用 `aclrtMalloc` 申请、`aclrtFree` 释放。算子只能直接访问设备内存。

两者之间靠 `aclrtMemcpy` 搬运。该函数通过**拷贝类型**参数声明方向：

| 拷贝类型 | 方向 |
| --- | --- |
| `ACL_MEMCPY_HOST_TO_DEVICE` | 主机 → 设备 |
| `ACL_MEMCPY_DEVICE_TO_HOST` | 设备 → 主机 |
| `ACL_MEMCPY_DEVICE_TO_DEVICE` | 设备 → 设备 |
| `ACL_MEMCPY_HOST_TO_HOST` | 主机 → 主机 |

**易错点**：设备内存只能在设备上被算子访问，不能直接在主机侧解引用；反之亦然。试图把设备指针当主机指针用，会得到非法访问。<span class="marginnote">「两个世界」的模型与 CUDA 的 host/device 内存完全一致：`cudaMalloc` 对应 `aclrtMalloc`，`cudaMemcpy` 对应 `aclrtMemcpy`。熟悉 CUDA 内存模型的读者，把名字换掉即可迁移大部分认知。</span>

## 2 设备内存的申请策略

`aclrtMalloc` 的签名比普通 `malloc` 多出两个参数——**分配策略**与**标志位**：

```c
aclError aclrtMalloc(void **devPtr, size_t size,
                     aclrtMemMallocPolicy policy,
                     uint64_t flags);
```

**policy**：`ACL_MEM_MALLOC_HUGE_FIRST`（优先大页）、`ACL_MEM_MALLOC_HUGE_ONLY`（仅大页）、`ACL_MEM_MALLOC_NORMAL_ONLY`（仅普通页）。大页（huge page）减少页表开销、提升大块数据的 DMA 效率，训练场景常优先使用。
**flags**：控制缓存行为，如 `ACL_MEM_MALLOC_HOOK_ENABLE` 等，用于内存挂钩与统计。

另一个重要 API 是 **`aclrtMallocCached`**：申请**带 L2 缓存**的内存。默认 `aclrtMalloc` 得到的内存不被 L2 缓存服务（数据直接走 HBM），而 `aclrtMallocCached` 申请的内存会驻留/经 L2 缓存，适合**被反复读取的数据**（如模型权重、频繁复用的中间张量）。<span class="marginnote">「缓存与否」的选择要付出代价：`aclrtMallocCached` 提升重复访问的命中率，但写回与一致性处理有开销；`aclrtMalloc` 适合流式数据（只读一次就丢）。选错缓存策略，是昇腾性能调优里最容易被忽视的「隐形损耗」之一。</span>

## 3 内存池与内存复用

频繁调用 `aclrtMalloc/free` 有两个代价：**系统调用开销**与**碎片化**。昇腾在运行时层面提供了**内存池**机制来缓解：

运行时按需向设备申请大块内存，切成小块放进池中；
程序释放的块回到池里，后续申请直接复用，避免反复向驱动要内存；
- 推理引擎（如模型执行框架）内部也会做**内存复用**：不同算子的中间张量若生命周期不重叠，可共享同一块物理内存。

这套机制的意义在于：**内存复用率直接决定单卡能跑多大的模型**。大模型训练时「把峰值内存压下去」的核心手段之一，就是让图引擎在编译期做内存规划，让不冲突的算子共享缓冲区。这一思想在《内存复用与算子执行性能优化》一节还会展开。<span class="marginnote">内存池与复用并非昇腾独有：`cudaMallocAsync` 就是 CUDA 对内存池的官方支持，PyTorch 的 `torch.cuda.memory` 缓存分配器也是同一思路。理解「池化」这个通用抽象，比死记某个 API 更值钱。</span>

## 4 公式解析：一块张量要占多少设备内存

计算张量内存占用是昇腾开发的基本功，尤其在做内存预算与对齐检查时。

$$
\text{字节数} = \prod_{i} \text{shape}[i] \times \text{dtype 字节数}
$$

以 FP16、形状为 $[1, 3, 224, 224]$ 的特征图为例拆四步：

- **第一步，算元素总数**：$1 \times 3 \times 224 \times 224 = 150528$ 个元素。
- **第二步，确定类型字节数**：FP16 占 2 字节，故 $\text{字节数} = 150528 \times 2 = 301056$ B。
- **第三步，换算单位**：$301056 / 1024 = 294$ KiB（昇腾文档常用 KB/KiB 混用，注意换算）。
- **第四步，检查对齐**：昇腾设备内存要求按 32 字节对齐，$301056$