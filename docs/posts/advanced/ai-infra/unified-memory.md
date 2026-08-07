---
title: 统一内存（Unified Memory）与显存超订（oversubscription）
date: 2026-08-07
---

# 统一内存（Unified Memory）与显存超订（oversubscription）

<div class="epigraph">
<p>让 CPU 与 GPU 共享一个记忆，而不是各抱着一块。</p>
<footer>—— 理查德 · 胡迪（Richard Huddy，NVIDIA 开发者关系）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ CUDA 编程指南（Unified Memory 章节）· 显存优化篇 ｜ 2026-08-07</p>
</div>

## 为什么从统一内存开始

上一节的 ZeRO-Offload 告诉我们「显存不够可以搬去 CPU」。但「搬」是谁在搬？是框架显式调 `cudaMemcpy`，还是硬件自动搞定？——**统一内存（Unified Memory, UM）** 就是后者的答案：CUDA 让 CPU 与 GPU 共享同一个虚拟地址空间，数据按需在两者间迁移，程序员不用手写拷贝。

更进一步，统一内存支持**显存超订（oversubscription）**：让一个「超过显存容量」的程序也能跑起来——不活跃的页被自动换出到 CPU 内存。理解 UM 与超订，是理解「为什么有些训练显存爆了还能跑、有些却 OOM」的关键。

## 1 统一内存的思想：一个地址空间

传统 CUDA 程序里，`malloc` 的是 CPU 内存，`cudaMalloc` 的是 GPU 显存，两者分离，拷贝靠 `cudaMemcpy` 显式搬运。**统一内存打破这堵墙**：

- `cudaMallocManaged` 分配一块「托管内存」，CPU 和 GPU 都能访问。
- 访问发生在哪一侧，**CUDA 驱动按页把数据迁移到哪一侧**。
- 程序员不关心物理位置，只看到一个统一的虚拟地址。<span class="marginnote">关键机制是「按页迁移 + 缺页处理」：GPU kernel 访问一个尚未在显存的页时，触发 page fault，驱动从 CPU 内存把该页拷进显存，再继续执行。对程序员透明，但页迁移本身有成本——后面会算这笔账。</span>

## 2 页迁移与缺页的性能账

UM 不是免费的。每次页迁移要付出「页大小 ÷ 带宽」的时间：

$$T_{\text{migrate}} = \frac{P_{\text{size}}}{B_{\text{link}}}$$

其中 $P_{\text{size}}$ 是页大小（默认 64KB–2MB），$B_{\text{link}}$ 是 CPU–GPU 链路带宽（PCIe 约 32GB/s）。更糟的是**缺页是串行的**：GPU 等页到位才能继续执行，页面一多、访问一随机，性能直接崩。

训练中的教训很直接：

- **顺序访问**（如逐层跑模型、顺序读 batch）：页迁移开销可被重叠掩盖，还能接受。
- **随机访问**（如散列的 embedding 表、随机 sample）：频繁缺页，GPU 大部分时间在等页——**性能灾难**。<span class="marginnote">这就是为什么「统一内存 + 大数据集」在训练里是禁忌：数据加载若走 UM 且随机访问，缺页风暴会让 GPU 利用率跌到个位数。业界做法是：数据集用普通显式拷贝预取，绝不让它走 UM 的随机缺页路径。</span>

## 3 显存超订：让程序超过显存跑起来

**超订（oversubscription）** 指让「显式分配总量 > 物理显存」的程序仍能运行——前提是硬件/驱动支持「换出」：

1. 分配超过显存时，UM 把「最久未用」的页**换出到 CPU 内存**。
2. 需要时再换回。
3. 程序视角：显存「看起来」比物理大得多。

这依赖 **concurrent managed access** 与驱动的 LRU 换页。效果：**一个 120GB 需求、80GB 显存的程序可以跑**，代价是慢（频繁换页）。<span class="marginnote">超订的本质是「让操作系统式的虚拟内存思想走进显存」：物理显存当「L1 cache」，CPU 内存当「内存」，页在两级间流动。训练里唯一适合超订的是「不常访问、又必须存在」的东西——比如超大但不常更新的 embedding 表、或推理时的长 KV cache。</span>

## 4 在训练中的正确用法：能超订什么

超订不是银弹。训练里什么能超订、什么绝对不能：

**能超订（访问稀疏、顺序）：**

- 大型稀疏 embedding 表（大模型 embedding 可达数十 GB，但单 batch 只访问其中一小片）。
- 推理时的 KV cache（随请求增长，超订让长上下文「勉强」装下）。
- 检查点暂存、不常访问的优化器状态。

**绝对不能超订（每步都全量访问）：**

- 参数、梯度、激活——它们每步都被完整读写，超订等于每步全量换页，慢到不可用。

**正确的姿势**：默认关闭 UM 超订，只对「确定稀疏访问」的张量显式启用，并配以**预取（prefetch）**——在 kernel 启动前 `cudaMemPrefetchAsync` 把要用的页提前搬到显存。<span class="marginnote">预取是把 UM 从「被动缺页」变成「主动搬运」的关键：你提前告诉驱动「接下来要碰这些页」，它趁 kernel 还没跑先把页搬好，缺页就消失了。训练代码里给每个 stage 的数据预取，是 UM 场景下唯一的性能保障。</span>

## 5 公式解析：超订下的有效带宽

设程序总工作集 $W$，物理显存 $M$，超订部分 $(W - M)$ 常驻 CPU。若每步访问全部 $W$，其中 $(W-M)$ 需要换页，则每步的有效带宽需求：

$$B_{\text{effective}} = \frac{W}{T_{\text{compute}}} \quad \text{vs} \quad B_{\text{link}} \text{（PCIe 上限）}$$

- **$W$（工作集）**：模型状态 + 激活 + 数据集触达量。
- **$T_{\text{compute}}$（计算时间）**：GPU 算完一步的时间。
- **瓶颈判据**：当 $\frac{W}{T_{\text{compute}}} > B_{\text{link}}$ 时，GPU 在「等数据」——算力利用率被 PCIe 卡死。

**判据**：若 $\frac{W-M}{T_{\text{compute}}} \ll B_{\text{link}}$，超订可接受（换页被计算盖住）；否则超订就是慢性自杀。<span class="marginnote">用这个判据回看训练：参数+梯度+激活每步全量访问，$W \approx 16\Psi + \text{Act}$，几乎必然超过 PCIe 带宽——所以「把模型状态交给 UM 超订」必然慢。而 embedding 表每步只访问很小一片，$W_{\text{实际}} \ll$ 全表，超订就划算。这就是「能超订 vs 不能超订」的定量分界。</span>

## 6 辨析｜易错点：统一内存的常见误区

**辨析｜易错点：**
- **UM ≠ 自动加速**：它消除的是「手动拷贝」的编码负担，不是物理传输成本；该慢的还是慢。
- **UM ≠ offload**：offload 是框架主动选择「哪些放 CPU」，UM 是驱动按访问自动搬——前者可预测，后者靠缺页，性能更难控制。
- **超订不保证安全**：非托管内存的越界、以及某些 GPU（无 concurrentManagedAccess）会直接报错或超时。
- **别让随机访问走 UM**：embedding 散列、采样器随机读取是缺页风暴高发区，要显式预取或普通拷贝。
- **PyTorch 的 `cudaMallocAsync` 与 UM 不同**：前者管「显存分配器」，后者管「CPU-GPU 统一地址」，别混。

## 7 小结

- **统一内存**：CPU/GPU 共享虚拟地址空间，按页自动迁移，免手写拷贝。
- **页迁移成本**：每页 $P_{\text{size}}/B_{\text{link}}$，随机访问触发缺页风暴。
- **超订**：让分配总量超过显存，靠 LRU 换页维持运行，代价是慢。
- **能超订的**：稀疏访问的东西（embedding、KV cache）；**不能的**：每步全量访问的参数/梯度/激活。
- **正确姿势**：默认关闭，稀疏张量显式启用，配 `cudaMemPrefetchAsync` 预取。

## 8 进阶与延伸

**动手验证缺页的开销**：用 `cudaMallocManaged` 分配一个大数组，先「顺序访问」测一遍耗时，再「随机访问」测一遍——你会看到随机访问慢几个数量级。这就是「UM 别碰随机访问」这条纪律的实验证据。

**几个值得进一步挖的方向**：

- **`cudaMemPrefetchAsync` 的正确用法**：prefetch 在 kernel 启动前把页搬到显存——配合「每阶段数据预取」，可以把 UM 的缺页开销几乎清零。怎么写一个「预取先行」的数据管线？
- **UM 与 offload 的异同**：UM 是「驱动按访问自动搬」，offload 是「框架主动选位置」——一个靠缺页、一个靠显式，性能可预测性差多少？
- **超订的风险**：`concurrentManagedAccess` 不支持的 GPU 上，超订可能直接报错——怎么检测你的硬件支不支持？这是 UM 方案可行性的前提。

**自测题**：为什么「把模型状态交给 UM 超订」必然慢？用本篇的判据 $\frac{W-M}{T_{\text{compute}}} \ll B_{\text{link}}$ 算一次，你就得到定量答案了。

## 9 动手实践清单

- 用 `cudaMallocManaged` 分配大数组，对比「顺序访问 vs 随机访问」的耗时。
- 加 `cudaMemPrefetchAsync` 预取，观察缺页是否消失。
- 检查你的 GPU 是否支持 `concurrentManagedAccess`。
- 把「稀疏访问的 embedding」走 UM、把「每步全量访问的参数」走普通分配，对比性能。
- 用 $\frac{W-M}{T_{\text{compute}}}$ 判据算你的「超订可行性」。
- 观察超订下的换页开销在 profiler 里的表现。
- 验证「UM ≠ offload」——两者对「谁决定数据位置」的差异。

在下一节，我们继续追击「显存明明够却 OOM」的另一元凶——**显存碎片与 PyTorch 的 caching allocator**。
