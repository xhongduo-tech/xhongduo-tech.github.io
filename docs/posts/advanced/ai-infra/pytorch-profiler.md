---
title: PyTorch Profiler：算子耗时分析与 trace 解读
date: 2026-08-07
---

# PyTorch Profiler：算子耗时分析与 trace 解读

<div class="epigraph">
<p>性能优化从「时间都去哪了」开始。</p>
<footer>—— 乔治 · 斯沃洛（George Swallow，软件性能研究者）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ PyTorch Profiler 官方文档 · 监控与剖析篇 ｜ 2026-08-07</p>
</div>

## 为什么从 PyTorch Profiler 开始

监控告诉你「GPU 利用率低」，但没告诉你「哪个算子慢、为什么慢」。**性能剖析（profiling）** 是下一步：把一次训练迭代的时间逐算子拆开，看时间到底花在哪。而 PyTorch Profiler（`torch.profiler`）是 PyTorch 生态的剖析入口——它记录每个算子的耗时、显存、调用栈，产出一份可以解读的 **trace（时间线）**。

本篇讲透 PyTorch Profiler 的使用与解读：怎么跑一次剖析、trace 怎么看、常见瓶颈长什么样。读完你就能「抓一个训练 step 的时间分解」，这是所有 PyTorch 性能优化的起点。

## 1 跑一次剖析：最小用法

PyTorch Profiler 的最小用法：

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    train_step()  # 要剖析的那一步

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

- **CPU + CUDA 双活动**：同时记录 CPU 端与 GPU 端的算子。
- **`record_shapes=True`**：记录每个算子的输入形状——判断「是不是某个张量形状导致慢」。
- **`sort_by="cuda_time_total"`**：按 GPU 耗时排序，直接看「最贵的算子」。
- **导出 trace**：`prof.export_chrome_trace("trace.json")`，在 `chrome://tracing` 或 Perfetto 里可视化时间线。

**剖析是「抽样调查」而非「全程监控」**：抓几个代表性的 step 剖析，而不是一直开着（开销大）。<span class="marginnote">剖析的正确姿势是「选代表步」：训练前几百步（warmup 后）和稳定期各抓一次——早晚期性能特征不同。别从头到尾开着 profiler，那会把训练拖慢 10 倍。用 `schedule` 参数可以「预热 n 步 → 剖析 m 步 → 跳过」的采样模式。</span>

## 2 读懂算子表：三个关键数字

`key_averages().table()` 输出的每行有几个关键列：

- **`Self CUDA Time`**：该算子**自身**（不含子算子）的 GPU 耗时——真正花在它身上的时间。
- **`CUDA Time Total`**：含子算子的总耗时（对封装算子有意义）。
- **`CPU Time`**：CPU 端耗时。**CPU 耗时 > GPU 耗时** 说明 CPU 喂不饱 GPU（kernel 启动/数据准备瓶颈）。
- **`Calls`**：调用次数。

**第一眼看的三个判据**：

1. 排第一的算子是谁？占总时间多少？——前 3 个算子通常占 60%+ 时间。
2. 有没有「CPU 时间 > GPU 时间」的算子？——CPU 瓶颈信号。
3. 有没有「叫了很多次的小算子」？——kernel 启动开销信号。<span class="marginnote">「Self CUDA Time」是区分「慢的算子」与「包了很多慢算子的封装」的关键：一个自定义模块可能显示很长，但那是子算子堆出来的；真正要优化的是 Self 时间大的原子算子。按 Self 时间排序，是剖析的第一步习惯。</span>

## 3 读 trace：时间线里看调度

`trace.json` 的时间线视图揭示「算子怎么排队的」：

- **CPU 行（绿色）**：CPU 提交算子的时间点。
- **GPU 行（CUDA kernels）**：GPU 实际执行的时间段。
- **空隙（gap）**：GPU 没活干的时间——**这是性能问题的富矿**：GPU 空闲 = CPU 没喂够 / 在等通信 / 在等同步。
- **overlap**：CPU 提交下一个算子的时间，与 GPU 执行当前算子重叠——好的流水线应该「CPU 一直领先于 GPU」。

**trace 的黄金问句**：时间线上「GPU 有没有空隙、空隙出现在哪」——空隙的位置就是瓶颈的位置。<span class="marginnote">在 Chrome trace 里按「GPU 上的空隙」看：如果空隙分布均匀，可能是 CPU 提交太慢（kernel 启动开销）；如果空隙集中在特定位置，可能是「同步点」（如 `.item()`、`all_reduce` 等待）。<strong>空隙不是「空闲」，是「瓶颈」</strong>——每个空隙背后都有一个「在等什么」的答案。</span>

## 4 常见瓶颈模式

从算子表与 trace 里，几类常见瓶颈一眼可辨：

| 症状 | 模式 | 对策 |
| --- | --- | --- |
| CPU 时间 ≫ GPU 时间 | 每个算子都短、CPU 忙 GPU 闲 | kernel 融合、减小启动次数 |
| 某算子独占 30%+ 时间 | 单个大算子 | 优化该算子（切分/更优 kernel） |
| 空隙集中在同步点 | `.item()`、`.cpu()`、AllReduce 等待 | 异步化、通信计算重叠 |
| 小算子调用上千次 | 逐元素操作 | 融合、向量化 |
| 显存峰值高 | 大临时张量 | 重计算、减少拷贝 |

**「先看最贵的算子」是剖析的元规则**：优化前 3 个算子通常能解决 80% 的时间问题；追着第 10 名优化是浪费。<span class="marginnote">「二八法则」在剖析里很灵：排前 3 的算子占总时间 60%+，排前 10 的占 90%+。先优化最贵的，收益最大；这也符合「优化瓶颈层」的木桶原理——别在小算子身上精打细算。</span>

## 5 公式解析：从剖析表计算优化收益

设剖析给出算子 $i$ 的耗时 $t_i$，迭代总耗时 $T = \sum t_i$（加空隙）。算子 $i$ 的**耗时占比**：

$$p_i = \frac{t_i}{T}$$

**优化算子 $i$ 后（耗时降为 $t_i'$）的整体提速**（Amdahl 形式）：

$$\text{Speedup} = \frac{T}{T - (t_i - t_i')} = \frac{1}{1 - p_i (1 - \frac{t_i'}{t_i})}$$

- **$p_i$（占比）**：算子越贵，优化它收益越大。
- **$t_i'/t_i$（加速比）**：把该算子加速到几倍。
- **上限**：即使把 $p_i$ 的算子优化到 0，整体也只能提速 $\frac{1}{1-p_i}$——**占比决定了优化的天花板**。

代入数字：若注意力算子占 40%，把它优化 2 倍 → 整体提速 $\frac{1}{1-0.4\times0.5} = 1.25$，即 25%。**这个公式告诉你「值不值得优化」**：占比小于 10% 的算子，优化到 0 也只提速 11%，不值得费劲。<span class="marginnote">Amdahl 定律在剖析里的应用：优化的天花板由「优化对象的占比」决定。剖析完先算各算子的 $p_i$，选占比最高的 2–3 个下手——这是「剖析驱动优化」的决策依据，避免「优化了但整体没快多少」的无效劳动。</span>

## 6 辨析｜易错点：剖析的常见误区

**辨析｜易错点：**
- **「剖析一次就够」是错觉**：不同 batch、不同阶段（warmup vs 稳定）性能特征不同，要多抓几个代表步。
- **「只看算子表不看 trace」漏掉调度问题**：算子表看「谁慢」，trace 看「谁在等」——空隙问题只有 trace 看得到。
- **「CPU 时间不重要」是错觉**：CPU 喂不饱 GPU 是常见瓶颈，CPU 时间 > GPU 时间就是信号。
- **「profiler 开着跑完整训练」**：开销大，要抽样；长跑还会积累巨量 trace。
- **别忽略「显存剖析」**：`profile_memory=True` 看每个算子的显存分配——OOM 问题要在剖析表里看显存峰值算子。

## 7 小结

- **最小用法**：`torch.profiler.profile` 抓 CPU+CUDA，`key_averages().table()` 看算子耗时排序。
- **三个关键列**：Self CUDA Time（自身耗时）、CPU vs GPU Time（喂饱判断）、Calls（启动开销）。
- **trace 解读**：看 GPU 空隙——空隙位置 = 瓶颈位置；CPU 领先 GPU 才是好流水线。
- **常见瓶颈**：CPU 瓶颈、单算子独占、同步点空隙、千次小算子。
- **优化决策**：按占比 $p_i$ 选最贵算子下手，Amdahl 定律决定天花板。

## 8 进阶与延伸

**动手剖析一个训练 step**：用 `torch.profiler` 抓你模型的一个 step，按 `cuda_time_total` 排序——列出前 5 个算子，算它们的总占比。你会发现「2–8 法则」在训练里一样成立：前几个算子吃了大半时间。

**几个值得进一步挖的方向**：

- **`record_shapes` 的价值**：记录输入形状后，你能看到「同一个算子因形状不同而耗时不同」——哪个 shape 最贵？这帮助定位「是不是某个张量形状导致的慢」。
- **`with_stack` 与调用栈**：profiler 可以记录每个算子的 Python 调用栈——怎么用它从「慢算子」反查「是模型代码里的哪一行」触发的？
- **`schedule` 的采样**：profiler 的 `schedule` 参数可以「预热 N 步 → 剖析 M 步 → 跳过」——怎么配才能「抓到代表性 step」又不拖慢训练？

**自测题**：为什么「按 Self CUDA Time 排序」比「按 Total」更能定位瓶颈？如果你能说清「Self 是原子算子的真实耗时」，就掌握了算子剖析的读表方法。

在下一节，我们把剖析从 PyTorch 扩展到系统级——**Nsight Systems（nsys）**：时间线分析与 CPU-GPU 协同诊断。
