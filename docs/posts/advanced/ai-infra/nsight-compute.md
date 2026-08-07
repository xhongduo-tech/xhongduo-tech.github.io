---
title: Nsight Compute（ncu）：kernel 级瓶颈定位
date: 2026-08-07
---

# Nsight Compute（ncu）：kernel 级瓶颈定位

<div class="epigraph">
<p>算子慢不可怕，可怕的是不知道为什么慢。</p>
<footer>—— 加里 · 安塔尼斯（Gary Antounian，GPU 性能工程师）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ NVIDIA Nsight Compute 官方文档 · 监控与剖析篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Nsight Compute 开始

nsys 告诉你「哪个 kernel 占了时间线」，但没告诉你「这个 kernel 内部为什么慢」。**Nsight Compute（ncu）** 是 NVIDIA 的 **kernel 级剖析器**——它钻进单个 CUDA kernel 内部，度量它的 SM 占用、内存吞吐、指令级瓶颈，最后给出一个「瓶颈是计算还是访存」的结论。它是「Roofline 模型」的实测工具。

本篇讲透 ncu 的用法与关键指标：怎么跑、看哪几个数、怎么从「吞吐率」读出瓶颈类型。读完你就能给任意一个 kernel 做「体检」，定位它是「算不动」还是「搬不快」。

## 1 ncu 是什么：kernel 内部的显微镜

对比工具链的分层：

- **监控**（DCGM）：看「资源忙不忙」。
- **nsys**：看「系统时间线，哪个 kernel 占时间」。
- **ncu**：看「单个 kernel 内部，为什么慢」——SM 利用率、访存吞吐、指令吞吐、占用率。

ncu 的用法：**先 nsys 找到最贵的 kernel，再 ncu 钻进去**。它一次剖析一个（或几个）kernel，输出极其详细的硬件计数器报告。

```bash
nsys profile -o rep python train.py   # 先找贵 kernel
ncu --kernel-name regex --set full python train.py  # 再钻进去
```

**关键原则**：ncu 开销大，只对「值得优化的 kernel」做——通常是 nsys 里耗时前几的。<span class="marginnote">「先 nsys 定位、再 ncu 深挖」是 GPU 性能剖析的标准流程：nsys 是「全局找嫌疑人」，ncu 是「对嫌疑人做审讯」。直接拿 ncu 剖析整个程序不现实（太慢），必须先用 nsys 缩小范围。这套「先粗后细」的两段式是 GPU 调优的方法论。</span>

## 2 核心指标：Roofline 的两个轴

ncu 的核心输出是「吞吐率」（achieved throughput）——对照 Roofline 模型的两个轴：

- **计算吞吐（Compute Throughput）**：SM 上的 FMA/ALU/Tensor Core 利用率（%）。
- **访存吞吐（Memory Throughput）**：显存带宽利用率（%）。

**瓶颈判定**：哪个轴「顶到 100%」，kernel 就受哪个约束：

- 计算吞吐 ~90% + 访存 ~30% → **计算瓶颈**（优化算法/减少 FLOPs）。
- 访存吞吐 ~90% + 计算 ~30% → **访存瓶颈**（优化数据布局/减少搬运）。
- 两者都低（<50%）→ **其他瓶颈**（占用率不足、延迟受限、指令开销）。<span class="marginnote">「两个吞吐都低」是最反直觉也最常见的情况：kernel 既不计算密集也不访存密集，但就是慢。这通常是「延迟受限」——SM 里活跃线程太少（占用率低），数据来了没线程去算。对策是提高并行度（加大 block、增加活跃 warp），而不是优化计算或访存。</span>

## 3 其他关键指标

除了两大吞吐，ncu 还输出一组「诊断性指标」：

- **Achieved Occupancy（实际占用率）**：活跃 warp / 最大 warp。太低 → 并行度不足；太高但慢 → 别的问题。
- **L2/L1 命中率**：缓存命中差 → 访存模式差。
- **Warp Stall 原因**：指令等待什么（内存？依赖？同步？）——「stall」是 kernel 慢的微观解释。
- **指令吞吐**：执行了多少条指令、每周期几条——看「指令开销」是否过大。

**「占用率」是 kernel 诊断的第一开关**：占用率低，几乎一切后续优化（计算/访存）都打折扣——因为没足够的线程在跑。<span class="marginnote">占用率与性能不是线性关系：占用率太低（<25%）肯定慢，但占用率 100% 也可能慢（如果瓶颈在访存延迟）。正确用法是把占用率当「必要条件」——先保证它够（比如 ≥50%），再谈计算/访存优化。很多 kernel 慢的第一步修复就是「提高占用率」。</span>

## 4 从 ncu 到优化：三类对策

根据 ncu 诊断，优化方向分三类：

| 诊断 | 病因 | 对策 |
| --- | --- | --- |
| 计算吞吐高 | 计算瓶颈 | 算法简化、用更优 kernel（Tensor Core） |
| 访存吞吐高 | 访存瓶颈 | 数据布局（AoS→SoA）、向量化、减少搬运 |
| 占用率低 | 并行度不足 | 减小寄存器/共享内存用量、加大 grid |
| Warp Stall（内存等待） | 延迟受限 | 提高 ILP（指令级并行）、预取 |
| 指令吞吐高 | 指令开销 | 减少逐元素指令、融合 |

**关键原则**：ncu 的诊断要「对症下药」——它是计算瓶颈就别去优化访存，那是缘木求鱼。<span class="marginnote">「诊断对了再动手」是 kernel 调优的铁律：改数据布局对「计算瓶颈」无效，加并行对「访存瓶颈」也可能无效。ncu 的价值就是「先给准确的病名」，避免瞎试。一个常见反例：访存瓶颈的 kernel，有人去换更快的算法（没用），实际是「访问模式差、缓存命中低」——该改布局。</span>

## 5 公式解析：瓶颈判定的定量依据

设 kernel 的实测计算吞吐 $U_c$、访存吞吐 $U_m$（都是 0–100%），Roofline 模型的瓶颈判定：

$$\text{Bottleneck} = \begin{cases} \text{Compute}, & U_c \gg U_m \text{ 且 } U_c \text{ 接近上限} \\ \text{Memory}, & U_m \gg U_c \text{ 且 } U_m \text{ 接近上限} \\ \text{Latency/Other}, & \text{两者都低} \end{cases}$$

ncu 还会给出 **Speed-of-Light（SOL）** 分数：kernel 离「理论最优」还差多远。它来自「实测时间 / 理论下界时间」：

$$\text{SOL} = \frac{\min(\text{compute-bound time}, \text{memory-bound time})}{T_{\text{measured}}}$$

- **计算下界**：$\frac{\text{FLOPs}}{C_{\text{peak}}}$（总 FLOPs ÷ 峰值算力）。
- **访存下界**：$\frac{\text{Bytes}}{B_{\text{peak}}}$（总字节 ÷ 峰值带宽）。
- **下界取两者较小**：理论上限由「更强的瓶颈」决定；SOL 越高越好。<span class="marginnote">SOL 是 ncu 最有价值的「一句话结论」：SOL 90% 说明 kernel 已接近理论极限，再优化空间很小（该换思路了）；SOL 30% 说明有大把空间，值得深挖。它是「值不值得优化」的量化答案——比看一堆吞吐数字更直接。</span>

## 6 辨析｜易错点：ncu 的常见误区

**辨析｜易错点：**
- **「ncu 对整个程序跑」**：ncu 按 kernel 剖析，开销极大，必须先用 nsys 定位再单个 kernel 钻。
- **「吞吐高 = 一定好」**：两个吞吐都 90% 罕见且矛盾（同时计算+访存瓶颈）；通常一个高另一个低才正常。
- **「占用率 100% 就是最优」**：占用率高但 SOL 低，说明「线程多但都在等」——可能是延迟受限。
- **「ncu 数字不用对标硬件」**：40% 吞吐在 A100 与 H100 上含义不同——要结合硬件规格解读。
- **别忽略「L2 命中率」**：低命中率会让访存吞吐虚高（读了很多没用数据）——看命中率才能判断「访存模式」对不对。

## 7 小结

- **ncu 的定位**：kernel 级剖析，钻进单个 CUDA kernel 看内部瓶颈。
- **两大吞吐**：计算吞吐与访存吞吐——哪个顶 100% 就是哪个瓶颈。
- **占用率与 stall**：低占用率 = 并行度不足；warp stall 原因 = 微观瓶颈。
- **SOL 分数**：实测时间 / 理论下界，判断「值不值得优化」。
- **工作流**：nsys 找贵 kernel → ncu 钻进去 → 按诊断对症下药。

## 8 进阶与延伸

**动手剖一个 kernel**：用 `ncu --set full` 剖析一个你模型的贵 kernel（先用 nsys 找到它）——看「计算吞吐 vs 访存吞吐」两个数，判断它是计算瓶颈还是访存瓶颈，再对比你的直觉。多数人的第一反应是错的，这正是为什么要实测。

**几个值得进一步挖的方向**：

- **SOL 分数的解读**：ncu 的「Speed of Light」把 kernel 离理论极限的距离量化成 0–100%——SOL 90% 的 kernel 还要不要优化？「边际收益递减」的判断依据是什么。
- **Warp Stall 的分布**：ncu 的「Warp State」显示每个 warp 在等什么（内存、依赖、同步）——「内存等待占比高」怎么对症？提高占用率还是改进访存模式？
- **occupancy 与瓶颈的联动**：占用率 50% 且访存瓶颈——加并行（提高占用率）能不能缓解访存瓶颈？「占用率 × 访存」的二维诊断怎么组合？

**自测题**：为什么「两个吞吐都低」说明是延迟受限？如果你能说清「线程不够 → 数据到了没人算」，就理解了占用率在 kernel 性能里的角色。

在下一节，我们把剖析从「算子」扩展到「通信」——**NCCL 通信的观测与重叠效果验证**。
