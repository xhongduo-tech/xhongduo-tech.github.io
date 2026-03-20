## 核心结论

流水线并行的微批次调度，目标是让多块 GPU 像装配线一样持续工作，而不是一部分在算、另一部分在等。做法是把一个 `mini-batch` 拆成 $m$ 个 `micro-batch`。`micro-batch` 可以理解为“更小的一份训练样本”，它会沿着 $p$ 个流水线阶段依次前向，再依次反向。

最核心的量化结论是：

$$
\text{Bubble} \approx \frac{p-1}{m}
$$

这里的 `Bubble` 是空泡率，也就是流水线里 GPU 空转的比例；$p$ 是流水线级数；$m$ 是微批次数。这个式子的直觉很简单：流水线启动和收尾各会产生固定长度的空闲，而把一个 batch 切得越细，这个固定损耗就越容易被摊薄。

因此，朴素 GPipe 调度下，如果希望空泡率低于 25%，常见经验是让：

$$
m \ge 4p
$$

因为这时：

$$
\frac{p-1}{m} < \frac{p}{4p}=0.25
$$

但只增加 $m$ 并不总是免费。GPipe 采用“先全前向，再全后向”的同步调度，需要缓存大量激活值。`激活值` 可以理解为“前向计算中间结果，反向传播时还要再用一次”。如果每个 micro-batch 都要保留一份，那么峰值显存会接近 $O(m \cdot A)$，其中 $A$ 表示单个 micro-batch 的激活占用。

交错调度如 1F1B 的价值就在这里。`1F1B` 是 “one-forward-one-backward” 的缩写，意思是稳定阶段尽量做到“做完一个 micro-batch 的前向，就尽快做某个已就绪 micro-batch 的反向”。这样激活不会随 $m$ 线性堆积，峰值显存通常可从：

$$
O(m \cdot A)
$$

下降到与流水线深度相关的级别，工程上常近似理解为每张卡只保留 $O(1/p)$ 比例的总激活，或者说只需要保留约 $O(p)$ 份在途激活，而不再随 $m$ 继续增长。

一个最小数值例子：8 块 GPU 构成 8 个 stage，即 $p=8$，如果一个 mini-batch 被切成 $m=32$ 个 micro-batch，那么 GPipe 的空泡率约为：

$$
\frac{7}{32}\approx 21.9\%
$$

这说明即使切到 32 份，仍然会有约 22% 的时间被暖启动和冷收尾吃掉。若换成 1F1B，吞吐上的空泡并不会神奇消失，但激活保留量显著下降，显存压力会比 GPipe 小得多。

---

## 问题定义与边界

问题可以定义为：在固定 GPU 数量下，把一个大模型切成 $p$ 个流水线阶段后，如何安排 $m$ 个 micro-batch 的前向与反向顺序，使 GPU 利用率高、显存不过载、通信不会压垮系统。

这个问题有三个边界。

第一，$m$ 不能太小。因为流水线像工厂产线，刚开工时后面的工位还没有活干，快结束时前面的工位已经没活干，这两段空闲就是 bubble。如果 $m$ 很小，固定的暖冲和冷却成本几乎无法摊薄。

第二，$m$ 也不能无限大。GPipe 的同步方式要求前向阶段先把所有 micro-batch 跑完，再统一做反向，这会导致早期 micro-batch 的激活长期驻留显存。理论上吞吐更好，但显存可能先爆掉。

第三，阶段间通信必须可控。流水线并行不是单卡自娱自乐，每个 stage 都要把激活发给下游、把梯度发回上游。`通信` 可以理解为“卡与卡之间搬运中间结果”。如果链路慢、带宽低，调度再漂亮也会被通信阻塞。

下面用一个玩具例子说明边界。假设有 8 个 stage，对应 8 块 GPU。如果只切成 8 个 micro-batch，即 $p=8,m=8$，那么：

$$
\text{Bubble} \approx \frac{7}{8}=87.5\%
$$

这说明流水线绝大部分时间都在暖机或收尾，几乎没有进入稳定高利用率区间。反过来，如果盲目切成 128 个 micro-batch，空泡率很低，但每个 micro-batch 都要保留一份激活，GPipe 很可能直接 OOM。

| 风险边界 | 现象 | 直接后果 | 常见对策 |
|---|---|---|---|
| $m$ 太小 | 暖冲/冷却占比高 | GPU 利用率低，吞吐差 | 增加 micro-batch 数，目标常取 $m \ge 4p$ |
| $m$ 太大 | 激活缓存堆积 | GPipe 容易 OOM | 改用 1F1B、重计算、减小单个 micro-batch |
| stage 切分不均 | 有的阶段算得慢 | 整条流水线被最慢 stage 限速 | 重新分层，做 load balance |
| 通信过慢 | 激活/梯度传输等待 | 计算被通信空转吞掉 | 提升带宽、减少同步点、做算通重叠 |
| 权重版本不一致 | 前向与反向使用不同参数 | 梯度不正确或收敛异常 | 使用同步更新或严格的 weight stashing 策略 |

真实工程里，问题不只是“有没有 bubble”，而是要同时满足吞吐、显存、收敛稳定性。很多训练系统最后选的不是理论最优调度，而是“在这套集群上能稳定跑满一周”的调度。

---

## 核心机制与推导

先看 GPipe。它的基本顺序是：一个 mini-batch 拆成 $m$ 个 micro-batch 后，所有 stage 先做完这 $m$ 个 micro-batch 的前向，再做完这 $m$ 个 micro-batch 的反向。它的优点是实现直观、语义清晰，而且所有 micro-batch 使用同一版本的权重，调试和验证比较容易。

为什么它有空泡？因为流水线深度是 $p$。第一个 micro-batch 想走到最后一个 stage，需要经过 $p-1$ 次传递；最后一个 micro-batch 的反向梯度想传回第一个 stage，也要经过 $p-1$ 次传递。稳定区间之外的这些空档，就是 bubble 的来源。

如果把稳定工作量近似看成 $m$，把额外空闲看成 $p-1$，就得到常见估计：

$$
\text{Bubble} \approx \frac{p-1}{m}
$$

这个式子不是在所有硬件和实现上都精确相等，但足够作为调参的一阶近似。

继续看显存。设单个 micro-batch 在某 stage 上需要保留的激活大小为 $A$。GPipe 先全前向再全后向，因此前向早期产生的激活必须一直等到反向开始时才能释放。于是峰值激活规模大致随 $m$ 增长：

$$
\text{Activation Memory}_{\text{GPipe}} = O(m \cdot A)
$$

这也是为什么单纯增大 $m$ 会很快触碰显存上限。

1F1B 的机制不同。它在流水线进入稳定态后，尽量让前向和反向交替进行。直觉上，它不是“把所有杯子都倒满再一起喝”，而是“前面刚倒出一杯，后面有条件就立刻喝掉一杯”。这样某个 micro-batch 的激活不需要等很久，反向一完成就可以释放。

因此，1F1B 的关键收益不是把 $\frac{p-1}{m}$ 这个启动损耗彻底消灭，而是把激活保留窗口缩短。工程上常将其理解为：每个 stage 只需要保留与在途 pipeline depth 相当的激活数量，而不是保存全部 $m$ 个 micro-batch 的激活。于是峰值从：

$$
O(m \cdot A)
$$

下降到与流水线深度相关的规模。若进一步做 interleaved 设计或激活回收，常见表述是每卡只承担约 $O(1/p)$ 比例的总激活压力。

还是看 $p=8,m=32$ 这个例子：

- GPipe：bubble 约为 $7/32\approx 22\%$，而且要保留多份前向激活。
- 1F1B：吞吐上仍受暖冲/冷却影响，但稳定阶段“前向一份、反向一份”交错推进，激活不会堆到 32 份。

这里要强调一个概念：`pipeline depth = p`。它的意思是，一次最多会有约 $p$ 个不同 micro-batch 同时处在流水线不同位置上。这个数决定了交错调度下“在途状态”的上限，也解释了为什么 1F1B 的显存增长不再跟 $m$ 强绑定。

---

## 代码实现

下面先给一个可运行的 Python 玩具程序，用来估算 bubble，并对比 GPipe 与 1F1B 的激活保留上界。这里的模型不是完整训练器，而是帮助理解调度变量之间的关系。

```python
from math import ceil

def bubble_rate(p: int, m: int) -> float:
    assert p >= 1 and m >= 1
    if p == 1:
        return 0.0
    return (p - 1) / m

def gpipe_activation_slots(m: int, activation_per_micro: int) -> int:
    assert m >= 1 and activation_per_micro >= 1
    return m * activation_per_micro

def onef1b_activation_slots(p: int, activation_per_micro: int) -> int:
    assert p >= 1 and activation_per_micro >= 1
    # 稳定阶段近似只需保留与流水线深度相关的在途激活
    return p * activation_per_micro

# 玩具例子：8-stage, 32 micro-batches
p, m, A = 8, 32, 10
bubble = bubble_rate(p, m)

assert round(bubble, 4) == round(7 / 32, 4)
assert bubble < 0.25
assert gpipe_activation_slots(m, A) == 320
assert onef1b_activation_slots(p, A) == 80
assert gpipe_activation_slots(m, A) > onef1b_activation_slots(p, A)

print("bubble=", bubble)
print("gpipe_slots=", gpipe_activation_slots(m, A))
print("1f1b_slots=", onef1b_activation_slots(p, A))
```

上面这段代码表达了两个事实。

第一，空泡率与 $m$ 成反比。把 $m$ 从 16 提到 32，bubble 近似减半。

第二，GPipe 的激活槽位跟 $m$ 线性相关，而 1F1B 更接近跟 $p$ 相关。这里的 `activation_slots` 可以理解为“要同时留在显存里的激活份数”。

实际调度器需要维护的不是一个公式，而是每个 micro-batch 在每个 stage 的状态。最小化伪代码通常长这样：

```python
# 每个 stage 的局部循环
for step in range(total_steps):
    if can_run_forward():
        micro = pick_next_forward_micro()
        forward(micro)
        save_activation(micro)
        send_activation_to_next_stage(micro)
        enqueue_for_backward(micro)

    if can_run_backward():
        micro = pick_next_backward_micro()
        recv_grad_from_next_stage(micro)
        backward(micro)
        release_activation(micro)
        send_grad_to_prev_stage(micro)
```

这段伪代码里有三个关键点。

第一，`enqueue_for_backward(micro)`。它表示前向做完的 micro-batch 不能直接丢掉，因为稍后反向还要用。

第二，`release_activation(micro)`。这是 1F1B 节省显存的关键动作。只要某个 micro-batch 的反向在当前 stage 完成，对应激活就应该尽快释放，而不是等整轮 mini-batch 结束。

第三，`pick_next_forward_micro()` 和 `pick_next_backward_micro()` 不能随便选。真实框架里常要处理权重版本一致性、梯度累计边界、跨 stage 通信顺序等约束。如果前向和反向使用的权重版本错位，训练可能不稳定。

一个真实工程例子是大模型训练框架中的 pipeline engine。假设你在 8 张 GPU 上训练一个数十亿到百亿参数模型，做法通常是：

1. 按层把模型切成 8 个 stage。
2. 把 global batch 拆成很多 micro-batch。
3. 用 1F1B 或 interleaved 1F1B 执行稳定调度。
4. 在显存吃紧时，再叠加 activation recomputation。`重计算` 的意思是“前向中间结果不全存，反向时需要时再重算一次”。

这样做的直接收益是：即使 $m$ 很大以降低 bubble，显存也不至于按 $m$ 一路线性爆炸。

---

## 工程权衡与常见坑

流水线并行的工程选择，核心不是“选哪个名字更高级”，而是要在吞吐、显存、复杂度之间找平衡。

GPipe 的优点是简单。调度清晰、行为容易验证、权重版本管理也相对直接。缺点同样明确：$m$ 稍大，激活就会堆得很快。1F1B 的优点是显存友好，更适合在有限显存下扩大 $m$；但缺点是调度更复杂，通信、依赖关系、边界步骤更容易出错。

常见坑如下表。

| 常见坑 | 典型表现 | 原因 | 对策 |
|---|---|---|---|
| $m$ 太小 | GPU 经常空转 | $(p-1)/m$ 太大 | 先把 $m$ 提到 $4p$ 附近再测 |
| $m$ 太大 | GPipe OOM | 激活按 $O(m \cdot A)$ 堆积 | 换 1F1B，或启用重计算 |
| stage 不均衡 | 某一张卡长期 100%，其他卡等待 | 层切分不均 | 重新划分 stage，按耗时而非层数平均 |
| 通信拖慢训练 | NCCL 时间占比高 | 激活/梯度包过多或链路慢 | 增大消息粒度，做通信与计算重叠 |
| 梯度累计边界出错 | loss 波动异常 | micro-batch 与 optimizer step 对齐错误 | 明确 global batch = micro-batch × data-parallel × accumulation |
| 权重版本错配 | 收敛不稳甚至发散 | 前向和反向使用了不同参数快照 | 使用同步调度或 weight stashing |

再看一个具体场景。设 $p=8,m=32$。从 bubble 看，这已经是可接受配置，因为：

$$
\frac{7}{32}\approx 0.219
$$

但如果模型很大、序列很长，GPipe 仍可能因为要缓存 32 份激活而爆显存。这时常见做法不是把 $m$ 降回去，因为那会直接提高 bubble，而是优先：

1. 改成 1F1B。
2. 启用 activation recomputation。
3. 重新切 stage，避免某一段过宽或过深。

还有一个经常被低估的问题是通信不平衡。1F1B 理论上显存更优，但如果网络很慢，某些 stage 可能在等梯度回来，结果看起来像“显存省了，吞吐却没上去”。所以流水线并行不是只盯公式，还要看 profiling。若算子耗时只占 50%，剩下都在等通信，那么下一步应该优化网络路径，而不是继续调 $m$。

---

## 替代方案与适用边界

最常见的对比是 GPipe vs 1F1B。

| 方案 | 核心思路 | 空泡表现 | 显存表现 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|---|
| GPipe | 先全前向，再全后向 | 约为 $(p-1)/m$，靠增大 $m$ 摊薄 | 激活接近 $O(m \cdot A)$ | 低 | 原型验证、实现优先 |
| 1F1B | 稳定阶段交错前向和反向 | 吞吐仍受暖冲/冷却影响，但更易稳定高利用率 | 激活显著低于 GPipe，近似与 $p$ 相关 | 中 | 大模型训练主流方案 |
| Interleaved / SlimPipe 类 | 把 stage 再细分并交错执行 | 可进一步压缩气泡 | 显存继续优化 | 高 | 超大模型、追求极致吞吐 |
| FSDP + Pipeline 混合 | 参数分片 + 流水线 | 取决于具体组合 | 参数与激活都可压缩 | 很高 | 显存极限场景 |

如果用一个直观类比：

- GPipe 像先把所有咖啡都倒好，再统一端出去。
- 1F1B 像倒完一杯就尽快送走一杯。
- SlimPipe 或 interleaved 变体，则像把每次操作拆得更细，让桌面上同时堆着的杯子更少。

这个类比只用于帮助理解节奏，不替代正式定义。正式结论仍然是：GPipe 用更简单的同步顺序换实现成本，1F1B 用更复杂的调度换显存效率，interleaved 方案则进一步在 bubble 与内存之间做精细化折中。

适用边界可以概括为：

- 如果你刚搭建训练系统，优先验证正确性，GPipe 更合适。
- 如果你已经被显存卡住，且需要把 $m$ 拉高以压 bubble，优先考虑 1F1B。
- 如果单纯 pipeline 还不够，需要继续突破参数规模，就考虑和张量并行、数据并行、FSDP 混合使用。

真实工程里常见方案不是“纯 pipeline”，而是多种并行策略叠加。原因很直接：流水线并行主要解决“模型太深，单卡放不下”的问题；FSDP 主要解决“参数太大”的问题；张量并行主要解决“单层太宽”的问题。不同瓶颈，要用不同工具。

---

## 参考资料

1. Cloud Tencent，《Megatron-LM 分布式执行调研》：给出流水线 bubble 的近似公式 $(p-1)/m$，并解释为什么常取 $m \ge 4p$ 作为经验阈值。  
2. Atlantis Press / torchgpipe 相关材料：说明 GPipe 在微批数足够大时 bubble 会被摊薄，同时展示交错调度对激活占用的改善。  
3. Emergent Mind 关于 pipeline parallelism schemes 的综述：总结 1F1B、interleaved 等调度策略，并强调交错调度对显存与吞吐平衡的意义。  
4. DeepSpeed / Megatron-LM 相关工程文档与实践文章：展示真实训练系统中如何将 pipeline、重计算、梯度累计组合使用。  
5. 大模型训练实践文章中关于 $p=8,m=176$ 等案例的记录：说明真实工程往往采用 $m \gg p$ 来压低 bubble，再用重计算和交错调度控制显存。
