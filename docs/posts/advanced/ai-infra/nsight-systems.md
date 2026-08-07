---
title: Nsight Systems（nsys）：时间线分析与 CPU-GPU 协同诊断
date: 2026-08-07
---

# Nsight Systems（nsys）：时间线分析与 CPU-GPU 协同诊断

<div class="epigraph">
<p>系统级剖析，是看清整条流水线的唯一方式。</p>
<footer>—— 马克 · 哈里斯（Mark Harris，NVIDIA 开发者技术）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ NVIDIA Nsight Systems 官方文档 · 监控与剖析篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Nsight Systems 开始

上一节的 PyTorch Profiler 是「算子级」剖析——它告诉你「哪个 PyTorch 算子慢」。但训练慢的原因常常在算子之外：kernel 启动排队、CPU 与 GPU 不同步、内存拷贝、NCCL 通信等待。要看清这些**跨 CPU、GPU、网络的时间线**，需要系统级剖析器——**Nsight Systems（nsys）**。

nsys 与 PyTorch Profiler 的关系：后者是「应用内视角」（看算子），前者是「系统视角」（看 CPU 线程、GPU 流、CUDA API、通信事件如何交织）。两者的关系像「医院科室 vs 全身检查」——算子慢要看 PyTorch Profiler，整机协同问题要看 nsys。本篇讲透 nsys 的时间线怎么看、CPU-GPU 协同问题怎么诊断。

## 1 nsys 是什么：采样 + 追踪

nsys 的核心是「时间线（timeline）」：记录 CPU 上每个线程在干嘛、GPU 上每个 kernel 何时执行、CUDA API 何时被调用、NCCL 通信何时发生。

两种工作模式：

- **`nsys profile`**：追踪运行命令，记录完整调用轨迹——精确但开销大。
- **`nsys stats`**：对生成的 report 做统计分析，输出汇总表。

典型用法：

```bash
nsys profile -o my_report --trace=cuda,nvtx,osrt python train.py
```

`--trace` 控制追踪哪些事件：CUDA 调用、NVTX 标记（PyTorch 自动埋的点）、操作系统运行时（线程调度）。<span class="marginnote">nsys 与 PyTorch Profiler 的 trace 格式是兼容的——PyTorch 的 NVTX 标记会直接出现在 nsys 时间线里。实际工作流常是「nsys 抓系统级 trace → 看到可疑区域 → 用 PyTorch Profiler 或 Nsight Compute 钻进去」——从系统到算子的层层深入。</span>

## 2 时间线的阅读：CPU、GPU、通信三轨

nsys 的时间线视图（Nsight Systems GUI）有几条关键轨道：

- **CPU 线程行**：每个线程的时间线——CPU 在提交算子、还是在等待。
- **CUDA API 行**：`cudaMemcpy`、`cudaLaunchKernel` 等调用的时间点。
- **GPU 行（kernel）**：GPU 上各 stream 的 kernel 执行段。
- **NCCL 行**：集合通信事件（AllReduce 等）。

**阅读的核心问题**：CPU 是否一直「领先于」GPU？——理想流水线里 CPU 提前提交好一串 kernel，GPU 无间隙执行；如果 GPU 每个 kernel 之间有空隙、且空隙前后跟着 CPU 活动，就是「CPU 喂不饱 GPU」或「同步等待」。<span class="marginnote">「CPU 领先 GPU」是健康流水线的标志：看时间线上 CPU 的 CUDA API 提交点是否早于对应 GPU kernel 的执行段。如果 API 提交与 kernel 执行「贴得太紧」（CPU 提交完一个就等），说明 CPU 没提前量——典型的 kernel 启动开销或同步调用问题。</span>

## 3 常见模式一：GPU 空隙（idle gaps）

时间线上 GPU 的空隙是最值得盯的。按空隙出现的位置分类：

- **均匀分布的小空隙**：CPU 每个算子之间都要「准备」——kernel 启动开销主导。对策：kernel 融合、减少算子数。
- **集中在特定位置的空隙**：某个同步点（`.item()`、`.cpu()`、`torch.cuda.synchronize`）——GPU 等 CPU 结束。对策：去掉同步调用、异步化。
- **通信后的空隙**：AllReduce 之后的空隙——通信没与计算重叠。对策：通信计算重叠（第二篇）。

**每个空隙背后都有一个「在等谁」的答案**——找出「等谁」，就找到了优化方向。<span class="marginnote">诊断空隙的实用技巧：把空隙前后的事件放大看——空隙前最后一个 kernel 是哪个、空隙后第一个事件是什么。如果空隙前是 `memcpy D2H`（GPU→CPU），多半是「把数据拷回 CPU 等结果」的同步；如果空隙后紧跟 NCCL，多半是通信等待。<strong>空隙的「边界事件」是诊断的关键线索</strong>。</span>

## 4 常见模式二：CPU-GPU 同步问题

CPU 与 GPU 是「异步协作」的：CPU 提交任务、GPU 排队执行。一旦出现「强制同步」，两者就从异步退化为同步——性能瞬间崩塌。典型强制同步：

- **`.item()` / `.tolist()`**：把 GPU 张量取回 CPU，必须等 GPU 算完。
- **`.cpu()` / `.numpy()` 转换**：D2H 拷贝 + 隐式同步。
- **`torch.cuda.synchronize()`**：显式同步。
- **`print(tensor)`**：也会触发同步。
- **Python 循环里逐元素操作**：每个操作都可能触发同步。

**在 nsys 时间线上的表现**：某个 CPU 调用后，CPU 线程「卡住」很长一段（等 GPU），GPU 的后续 kernel 推迟。**这些同步点如果出现在训练循环的热路径上，就是性能杀手**。<span class="marginnote">「训练里为什么慢」的经典答案之一：代码里有 `.item()` 用于打印 loss——每次 `.item()` 都让 GPU 停一下，几百次迭代就是几百次停顿。nsys 时间线上会看到 CPU 在 `.item()` 处等 GPU 的「长停顿」。对策：loss 用异步方式汇总（如每 N 步同步一次）。</span>

## 5 公式解析：从时间线估算重叠效率

设一步的时间线上，GPU 计算总时长 $T_{\text{compute}}$、通信总时长 $T_{\text{comm}}$、CPU 提交总时长 $T_{\text{cpu}}$、步总时长 $T_{\text{step}}$。

**通信计算重叠率**（comm 被计算盖住的比例）：

$$\text{Overlap} = 1 - \frac{T_{\text{comm, exposed}}}{T_{\text{comm, total}}}$$

其中 $T_{\text{comm, exposed}}$ 是「GPU 空隙里暴露出来的通信时间」（未被计算覆盖的部分）。

- **$T_{\text{comm, total}}$（总通信量）**：所有 NCCL 事件的总时长。
- **$T_{\text{comm, exposed}}$（暴露通信）**：从时间线看，通信期间「GPU 没在算」的时间。
- **理想目标**：$\text{Overlap} \to 1$——所有通信都被计算盖住，GPU 从不为通信停。<span class="marginnote">这个「重叠率」是从 nsys 时间线量化的核心指标：通信总量没法减（模型决定的），但「暴露的通信」可以优化（重叠、调度）。若重叠率只有 50%，说明一半的通信时间在干等——优化通信计算重叠的空间很大；这是分布式训练性能剖析的黄金指标。</span>

## 6 辨析｜易错点：nsys 的常见误区

**辨析｜易错点：**
- **「nsys 就是另一个 PyTorch Profiler」**：定位不同——nsys 看系统/协同，PyTorch Profiler 看算子，配合使用。
- **「时间线越长越好」**：trace 越长越难读、文件越大；抓「代表步」即可，别整训练跑 trace。
- **「GPU 忙就没问题」漏了同步**：GPU 忙但 CPU 在干等也算浪费——要看 CPU 是否在「等 GPU 结束后才能继续」。
- **「通信时间长就是网络慢」**：通信时间可能来自「没重叠」（通信在干等）而非「带宽不够」——先看重叠率。
- **别忽略「启动开销」**：几千个小 kernel 的启动开销，在时间线上是「锯齿状」的 GPU 活动——融合是解药。

## 7 小结

- **nsys 的定位**：系统级时间线剖析，看 CPU/GPU/通信如何协同——与算子级的 PyTorch Profiler 互补。
- **时间线三轨**：CPU 线程、CUDA API/GPU kernel、NCCL 通信。
- **核心问题**：CPU 是否领先 GPU？GPU 空隙在哪、在等谁？
- **两类常见病**：GPU 空隙（启动开销/同步/通信未重叠）与 CPU-GPU 强制同步（`.item()`/`.cpu()`）。
- **重叠率**：$1 - T_{\text{exposed}}/T_{\text{total}}$，分布式训练的黄金指标。

## 8 进阶与延伸

**动手抓一份 nsys trace**：用 `nsys profile --trace=cuda,nvtx python train.py` 抓 50 步训练，在 Nsight Systems GUI 里打开——数一数「GPU 空隙」有几处、每处空隙前后是什么事件。这就是「时间都去哪了」的第一手答案。

**几个值得进一步挖的方向**：

- **CPU 领先 GPU 的量化**：时间线上怎么测量「CPU 提交提前量」？看「CUDA API 提交点」与「对应 kernel 开始」之间的间隔——提前量足够，GPU 才无间隙。
- **同步点的定位**：`.item()`、`.cpu()`、`torch.cuda.synchronize` 在时间线上长什么样？用 nsys 的「sync 标记」过滤，找出训练热路径上的隐式同步。
- **多 rank 的 nsys**：分布式训练怎么抓多 rank 的 trace？`nsys` 对每个 rank 单独抓，再合并看——跨 rank 的通信等待在合并时间线上一目了然。

**自测题**：为什么「GPU 空隙 = 瓶颈位置」？如果你能说清「每个空隙背后都有一个在等谁」，就掌握了 nsys 时间线的核心读法。

## 9 动手实践清单

- 用 `nsys profile --trace=cuda,nvtx` 抓 50 步训练，在 GUI 里打开。
- 数「GPU 空隙」的数量，观察每个空隙前后的边界事件。
- 用「CPU 是否领先 GPU」判断流水线健康度。
- 定位训练热路径上的 `.item()`/`.cpu()` 强制同步。
- 用「重叠率 = 1 − 暴露/总」量化通信重叠。
- 检查「kernel 启动开销」在时间线上是否锯齿状。
- 画「空隙位置 → 瓶颈类型」的诊断对照表。

在下一节，我们钻到 kernel 内部——**Nsight Compute（ncu）**：kernel 级瓶颈定位。
