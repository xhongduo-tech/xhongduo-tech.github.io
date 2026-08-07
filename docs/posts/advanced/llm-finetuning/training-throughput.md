---
title: 训练吞吐调优：从单卡到多卡的 MFU 分析
date: 2026-08-07
---

# 训练吞吐调优：从单卡到多卡的 MFU 分析

<div class="epigraph">
<p>你的显卡每秒钟做的浮点运算，有多少是白干的？</p>
<footer>—— 引意自 AI 基础设施实践常谚</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第三章 ｜ 2026-08-07</p>
</div>

## 为什么从吞吐调优开始

显存解决「放不放得下」，吞吐解决「跑得快不快」。同样微调一条 7B 模型，有人 3 小时训完，有人要 20 小时——差距往往不在硬件，而在**利用率**。GPU 是昂贵的资源，一台 A100 一小时的成本以十美元计，利用率差 5 倍就是差一个数量级的钱。

本节的核心工具是 **MFU（Model FLOPs Utilization，模型算力利用率）**——一个回答「GPU 的理论算力被用到了几成」的数字。从单卡到多卡，MFU 是所有吞吐调优的度量衡：调优前先测它，调优后再测它，一切改进都要落在「MFU 变高了」上。<span class="marginnote">MFU 的发明者是 Megatron-LM 团队的 Narayanan 等人：他们在《Efficient Large-Scale Language Model Training on GPU Clusters》里用 MFU 对比了不同并行策略的吞吐，从此「MFU 上了 50% 没有」成了大模型训练圈的口头禅。</span>

## 1 训练到底花了多少计算：6N 法则

要算利用率，先算「该用多少算力」。训练一个 Transformer，处理每个 token 所需的浮点运算，有一个著名的经验法则：

$$
\text{FLOPs}_{\text{per token}} \approx 6N
$$

其中 $N$ 是参数量。逐项拆解这 6 份：

- **前向 2N**：每个参数在一次前向里大约参与 1 次乘加（MAC），1 次 MAC = 2 次 FLOP（一次乘法 + 一次加法），故 2N；
- **反向 4N**：反向要算两层梯度——激活的梯度（2N）与权重的梯度（2N），合计 4N；
- **合计 6N**：这是「全参微调 / 预训练」的每 token 算力。注意**推理只有前向**，约 2N；LoRA 只更新低秩参数，但前向反向仍要跑完整模型——算力没省，省的是显存与优化器开销。

于是训练 $D$ 个 token 的总算力是 $6ND$。这个数字与实现无关，**是物理下限**——任何训练框架、任何并行策略，处理这些 token 都至少要花这么多 FLOPs。它成了 MFU 的分子。

## 2 公式解析：MFU 怎么算

**MFU = 实际算力 ÷ 理论峰值算力**。把「每步实际算力」算出来，除以 GPU 的峰值 FLOPs，就得：

$$
\mathrm{MFU} = \frac{6 \cdot N \cdot T_{\text{step}}}{P_{\text{peak}} \cdot \Delta t}
$$

逐项拆解：

- $N$：参数量；
- $T_{\text{step}}$：**每步处理的 token 数** = 全局 batch size × 序列长度（多卡时乘上卡数对应的全局 batch）；
- $6 \cdot N \cdot T_{\text{step}}$：这一步必须完成的有效 FLOPs（物理下限）；
- $\Delta t$：这一步的实际耗时（秒），由测时得到；
- $P_{\text{peak}}$：GPU 峰值 FLOPs——A100 约 **312 TFLOPS**（BF16），H100 约 **989 TFLOPS**。多卡时取总和。

代入一个例子：用 8 卡 A100 训 7B 模型，全局 batch 1024、序列 2048（即每步 2.1M token），实测每步 84 秒：

$$
\mathrm{MFU} = \frac{6 \cdot 7\times10^9 \cdot 2.1\times10^6}{8 \cdot 312\times10^{12} \cdot 84} \approx \frac{8.8\times10^{16}}{2.1\times10^{17}} \approx 0.42 \ (42\%)
$$

42% 是个**不错的成绩**。经验刻度：**>50% 优秀，30%–50% 正常，<20% 说明有严重的吞吐问题**（多半是通信没重叠、kernel 太碎、或数据加载饥饿）。用同一组数反推吞吐：42% × 8 卡峰值 2.5×10¹⁵ FLOPs/s，除以每 token 的 4.2×10¹⁰ FLOPs，约合每秒 2.5 万 token——这就是「MFU 与吞吐」两本账的换算关系。<span class="marginnote">MFU 与显存无关，是纯「算得快不快」的指标；另一个常见指标 <strong>吞吐（tokens/sec）</strong>是绝对速度，MFU 是相对效率。同样 10k token/s，在 A100 上 MFU 只有 30%，在 H100 上可能只有 20%——要跨硬件比效率，看 MFU。</span>

## 3 单卡吞吐：先把地基打牢

多卡吞吐是单卡吞吐乘上扩展率，所以先优化单卡。

单卡 MFU 的头号敌人是**内存墙**：许多算子不是算力瓶颈，而是**带宽瓶颈**——算得再快，也在等数据从显存搬进寄存器。对这类算子（如 LayerNorm、激活函数、逐元素操作），GPU 的计算单元大量闲置。对策：

- **融合 kernel（kernel fusion）**：把多个算子合成一个（如 FlashAttention 把 attention 的读、算、写融成一个 kernel），减少显存往返；
- **大 batch / 长序列**：把「每 token 的固定开销」摊薄，让 GPU 更多时间在「密集矩阵乘」这种高 MFU 算子上；
- **避免 CPU-GPU 同步**：Python 里一句 `tensor.item()` 就会阻塞流水线，训练循环里要杜绝；
- **避免小张量、循环式操作**：把多次小前向合成一次大前向（batch 上做文章），kernel 启动开销才摊得开。

一句话：**单卡 MFU 高不高，看「密集计算占比」**——让 GPU 的大多数时间在跑 GEMM（通用矩阵乘），而不是在等数据、切 kernel。

## 4 多卡吞吐：通信与计算的重叠

多卡之后，MFU 的新敌人是**通信**。每次 all-reduce / all-gather 都在让 GPU 停下等网络。多卡 MFU 优化有三大招：

**第一招：通信-计算重叠**。ZeRO/FSDP 的通信应该与「相邻层的计算」同时进行——计算当前层时，后台通信下一层。框架默认开重叠，但要检查是否真的生效（比如 `overlap_comm`、`reduce_scatter` 选项）。**这是多卡 MFU 的头号杠杆**：不开重叠，8 卡 MFU 可能从 42% 掉到 25%。

**第二招：梯度累积要适度**。梯度累积把通信频率降为原来的 $1/G$（G 为累积步数），但每步的 batch 变小、单步 MFU 下降。平衡点通常在「累积到让通信占比 <20%」附近——累积太多，单步算力利用率反而下降。

**第三招：先小规模测扩展性**。用「1 卡 → 2 卡 → 4 卡」逐级对比 MFU：若从 1 卡到 2 卡 MFU 掉了一半，说明通信没重叠或带宽不够；若接近线性，再往大了扩。**扩展性（scaling efficiency）= 多卡 MFU ÷ 单卡 MFU**，是衡量并行方案优劣的金标准。<span class="marginnote">排查多卡吞吐问题时，先做一个「数据加载测试」：让模型跑空循环（不更新），看 GPU 是否吃饱。如果 GPU 利用率本就不高，问题在数据管线而非并行；如果 GPU 满载但 MFU 低，问题在 kernel 与通信。分而治之，别眉毛胡子一把抓。</span>

## 5 常见瓶颈速查

把吞吐问题按「GPU 利用率」与「MFU」两个信号分类排查：

| 症状 | 可能的根因 | 排查方向 |
| --- | --- | --- |
| GPU 利用率 <80% | 数据加载饥饿、CPU-GPU 同步 | 检查 dataloader worker、`item()` 调用 |
| GPU 满载但 MFU <20% | kernel 碎片化、带宽瓶颈 | 开 FlashAttention、融合算子 |
| 多卡 MFU 骤降 | 通信未重叠、带宽不足 | 查 `overlap_comm`、节点拓扑 |
| MFU 正常但吞吐低 | batch 太小、序列太短 | 放大 batch / 序列，摊薄固定开销 |
| 显存够但频繁换页 | 分页优化器抖动 | 减小 batch，或主动 offload |

记住一句总纲：**MFU 是果，不是因**——它告诉你「病在哪」，但病因要靠上面的分类去定位。测 MFU 用「训练器内置的 `torch.profiler` + 手动计时」组合，每调整一个旋钮就重测一次，用数字说话。

一段最朴素的 MFU 测量代码（预热后计时若干步取均值）：

```python
def measure_mfu(model, tokens_per_step, peak_flops, warmup=3, steps=10):
    for _ in range(warmup): model.train_step()          # 预热，填满缓存
    t0 = time.perf_counter()
    for _ in range(steps): model.train_step()
    dt = (time.perf_counter() - t0) / steps
    n_params = sum(p.numel() for p in model.parameters())
    return 6 * n_params * tokens_per_step / (peak_flops * dt)
```

要诀是**预热后再计时**：头几步数据管线、CUDA 上下文还没热起来，测出来会虚低。测完按第 5 节的症状分类去定位瓶颈，改一个旋钮重测一次——调优是「测 → 改 → 再测」的循环，MFU 只是这个循环的仪表盘。

## 6 小结

- **6N 法则**：训练每个 token 至少花 $6N$ FLOPs（前向 2N + 反向 4N），推理约 $2N$——这是 MFU 的分子，物理下限。
- **MFU = $6N T_{\text{step}} / (P_{\text{peak}} \Delta t)$**：实际算力 ÷ 理论峰值；>50% 优秀、30%–50% 正常、<20% 有问题。
- **单卡优化**：融合 kernel、大 batch、杜绝 CPU 同步——让 GPU 多在跑 GEMM。
- **多卡优化**：通信-计算重叠是第一杠杆，梯度累积适度，用「1→2→4 卡」测扩展性。
- **排查二分法**：GPU 不忙查数据管线，GPU 忙而 MFU 低查 kernel 与通信。

在下一节，我们从「跑得快」回到「训得稳」：**训练稳定性——loss 尖峰、梯度裁剪与学习率调度**，把训练中那些「loss 突然起飞」的惊魂时刻一次讲清。
