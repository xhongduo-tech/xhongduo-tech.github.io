## 核心结论

GPipe 的核心目标，是在**不改变同步 SGD 语义**的前提下，把一个大模型拆到多个加速器上训练。这里的同步 SGD，可以先理解为“一个大 batch 的所有样本都算完梯度后，再统一更新一次参数”。

它做了两件事：

1. 把模型按层切成 $K$ 段，每段放到一个 stage，也就是一个只负责部分层的执行单元。
2. 把一个 mini-batch 再切成 $M$ 个 micro-batch，也就是“大 batch 里的小批次”。

GPipe 的关键不是“切分”本身，而是**更新时机**：参数不会在每个 micro-batch 后立刻更新，而是等这一个 mini-batch 的全部 $M$ 个 micro-batch 都完成前向和反向后，把梯度累积起来：

$$
G=\sum_{i=1}^{M} g_i
$$

然后再做一次统一更新。只要 loss 的缩放处理正确，这和直接用原始 mini-batch 做一次同步训练是等价的。

它的第二个关键是**流水线调度**。流水线，可以先理解为“不同设备在同一时刻处理不同 micro-batch 的不同模型段”。这样多个 stage 可以并行工作，不再是“一张卡把整网算完，下一张卡再开始”。

但流水线一定有空转区，也叫 bubble，常用近似公式是：

$$
\text{bubble}=\frac{K-1}{M+K-1}
$$

所以 $M$ 太小，空洞比例就高；$M$ 足够大，设备空闲时间就少。

第三个关键是 **re-materialization**，也常叫重材料化或激活重算。白话说，就是“前向时不把所有中间激活都存下来，反向需要时再重新算一遍”。这样峰值激活内存可以从朴素做法的 $O(N\times L)$ 降到近似：

$$
O\left(N+\frac{L}{K}\times\frac{N}{M}\right)
$$

其中 $L$ 是总层数，$N$ 是 mini-batch 大小。直观上，切分层数减少了单卡负责的层，切分 micro-batch 减少了单次需要保留的激活量，两者一起降低内存压力。

一个最容易理解的玩具例子是：8 层 Transformer，切成 4 个 stage；一个大小为 1024 的 mini-batch，拆成 16 个 micro-batch，每个 64。第 1 个 micro-batch 先后经过 stage1、2、3、4；当它到达最后一个 stage 并开始反向后，前面的 stage 不一定空着，它们可能正在处理后续 micro-batch。等第 16 个 micro-batch 也反向结束后，才把 16 份梯度加总，做一次参数更新。

---

## 问题定义与边界

GPipe 要解决的问题，不是“怎么把 batch 切小”这么简单，而是：

- 单卡放不下完整模型
- 单卡也放不下想要的 batch size
- 还希望训练语义尽量保持和同步大 batch 一致

这时需要同时处理三个维度：

| 名称 | 白话解释 | 主要作用 | 会影响什么 | 同步发生时机 |
| --- | --- | --- | --- | --- |
| mini-batch | 一次参数更新对应的完整样本集合 | 决定一次更新看多少数据 | 收敛稳定性、统计量 | 一次 mini-batch 结束后 |
| micro-batch | mini-batch 切出来的小块 | 让流水线能流动，减小单次显存压力 | 空洞比例、吞吐、激活缓存 | 通常不单独触发参数同步 |
| partition / stage | 模型按层切出来的一段 | 让不同设备分别负责不同层 | 跨设备通信、负载均衡 | 同步的是边界激活和梯度 |

设：

- mini-batch 大小为 $N$
- micro-batch 个数为 $M$
- 模型层数为 $L$
- partition 数为 $K$

那么每个 micro-batch 的大小大约是 $\frac{N}{M}$，每个 stage 负责大约 $\frac{L}{K}$ 层。GPipe 的设计边界，本质上是这三类约束之间的平衡：

1. **内存边界**：单个 stage 只能保存有限激活和参数。
2. **计算边界**：各 stage 计算量不能差太多，否则慢的 stage 会拖住整条流水线。
3. **收敛边界**：参数更新频率被降到“每个 mini-batch 一次”，所以学习率、warmup、动量统计都要按真实 batch 重新理解。
4. **通信边界**：stage 之间只传边界激活和边界梯度，但跨设备频繁传输仍然有代价。

新手最容易混淆的一点是：**micro-batch 不是新的优化步长单位**。它只是为了调度和显存管理而引入的执行单位。真正的优化步，仍然对应整个 mini-batch。

举个简单场景：你希望训练 batch size 为 1024，但单卡一次最多只能跑 64。如果模型还大到放不下一整网，那就不能只做普通梯度累积，还要做层切分。于是可以把 1024 拆成 16 个 micro-batch，每个 64，再把模型切到 4 张卡上。每张卡只负责部分层，每次只处理一个 micro-batch 的局部计算，最终把 16 份梯度合并成一次同步更新。

这里的边界也很明确：GPipe 适合**层状结构明显、按层切分自然的大模型**，例如 Transformer 编码器、解码器堆叠网络；如果模型结构高度分叉、控制流复杂，stage 切分和重算实现都会更麻烦。

---

## 核心机制与推导

GPipe 的执行机制可以拆成三个部分：前向调度、反向调度、梯度累积。

### 1. 前向调度

前向时，$M$ 个 micro-batch 按顺序进入 stage1，再依次传给 stage2、stage3，直到 stageK。每个 stage 都像装配线上的一个工位，只处理自己负责的层。

如果记 $f_k$ 表示第 $k$ 个 stage 的前向函数，那么第 $i$ 个 micro-batch 的前向就是：

$$
h_i^{(k)} = f_k\left(h_i^{(k-1)}\right)
$$

其中 $h_i^{(0)}$ 是输入 micro-batch 本身。

### 2. 反向调度

GPipe 论文中的经典调度通常被概括为 F-then-B。白话说，就是先把一批 micro-batch 的前向推满流水线，再按逆序做反向。每个 micro-batch 的梯度会从最后一个 stage 往前传。

若第 $i$ 个 micro-batch 的 loss 为 $\ell_i$，则对应参数梯度是：

$$
g_i=\nabla_\theta \ell_i
$$

stage 内部反向时，还需要用到前向激活。为了省显存，GPipe 不会完整保留所有中间激活，而是只缓存必要边界，其他激活在反向前临时重算，这就是 re-materialization。

### 3. 梯度累积与同步更新

最核心的点在这里。假设整个 mini-batch 的 loss 定义为各 micro-batch loss 的平均：

$$
\mathcal{L} = \frac{1}{M}\sum_{i=1}^{M}\ell_i
$$

那么总梯度就是：

$$
\nabla_\theta \mathcal{L} = \frac{1}{M}\sum_{i=1}^{M}\nabla_\theta \ell_i
= \frac{1}{M}\sum_{i=1}^{M}g_i
$$

这说明，只要你在每个 micro-batch 上得到 $g_i$ 后做累积，并在更新前除以 $M$，结果就等价于直接对完整 mini-batch 求梯度。也正因为如此，GPipe 虽然执行上拆成了很多 micro-batch，但优化语义仍然是同步大 batch。

### 4. Bubble 公式为什么成立

流水线起步时，后面的 stage 还没活干；流水线收尾时，前面的 stage 也会提前空闲。这两段空闲时间构成 bubble。对均匀 stage、均匀 micro-batch 的近似模型，bubble 比例可写成：

$$
\text{bubble}=\frac{K-1}{M+K-1}
$$

这个公式反映两个直觉：

- $K$ 越大，流水线越深，起步和收尾损失越明显。
- $M$ 越大，可被重叠的工作越多，空洞被摊薄。

玩具例子：$K=4, M=16$ 时，

$$
\text{bubble}=\frac{3}{19}\approx 15.8\%
$$

说明大部分时间都在做有效计算。若 $K=4, M=4$，则：

$$
\text{bubble}=\frac{3}{7}\approx 42.9\%
$$

这时近一半时间都有设备在等。

### 5. 内存公式为什么下降

朴素训练里，很多层的激活都要一直保存到反向完成，激活内存大致随 batch 和层数同时增长，可近似看作 $O(N\times L)$。

GPipe 通过两件事降低它：

- 每张卡只负责 $\frac{L}{K}$ 层
- 每次只处理大小为 $\frac{N}{M}$ 的 micro-batch
- 非边界激活不长时间保存，反向时再重算

于是峰值激活内存可以近似写成：

$$
O\left(N+\frac{L}{K}\times\frac{N}{M}\right)
$$

这里的 $O(N)$ 项可以理解为边界缓存或必要保留项，第二项才是局部层激活的主项。它说明：增大 $K$ 或增大 $M$，都可能降低单设备激活压力，但代价分别是更多通信和更多调度开销。

真实工程例子可以看 GPipe 论文中的大规模 Transformer 训练：模型深度达到 128 层，参数规模约 6B，运行在 128 个 TPU 分区上。它之所以能训练，不只是“机器多”，而是流水线切分、micro-batch 调度、同步梯度累积和重材料化同时成立。否则单个设备既放不下模型，也保不住激活。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，重点演示“micro-batch 梯度累积后再统一更新”这件事与整 batch 直接更新是等价的。这里不用深度学习框架，直接用线性回归的平方误差来验证。

```python
import numpy as np

def grad_for_batch(x, y, w):
    # loss = mean((x*w - y)^2)
    pred = x * w
    grad = np.mean(2 * x * (pred - y))
    return grad

def grad_with_micro_batches(x, y, w, m):
    xs = np.array_split(x, m)
    ys = np.array_split(y, m)
    total = 0.0
    total_n = 0
    for xb, yb in zip(xs, ys):
        pred = xb * w
        # 对每个 micro-batch 求“sum loss”对应的梯度，再最后除总样本数
        grad_sum = np.sum(2 * xb * (pred - yb))
        total += grad_sum
        total_n += len(xb)
    return total / total_n

x = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
y = np.array([2., 4., 6., 8., 10., 12., 14., 16.])
w = 0.5

g_full = grad_for_batch(x, y, w)
g_micro = grad_with_micro_batches(x, y, w, m=4)

assert np.allclose(g_full, g_micro)

lr = 0.01
w1 = w - lr * g_full
w2 = w - lr * g_micro

assert np.allclose(w1, w2)
print("梯度累积后的统一更新，与整 batch 直接更新等价")
```

这段代码验证的是 GPipe 的数学基础，不是完整流水线。完整实现还需要 stage 切分、激活传递、边界缓存和反向调度。

下面给一个更贴近 GPipe 的伪代码。它强调三个角色：`partition_id`、`micro_batch_id`、`last_micro_batch`。

```python
def train_one_minibatch(minibatch, partitions, optimizer, M):
    micro_batches = split_into_micro_batches(minibatch, M)
    grad_accum = [zero_like(p.params) for p in partitions]

    forward_cache = {}

    # 1. 前向：把所有 micro-batch 推入 pipeline
    for micro_batch_id, micro in enumerate(micro_batches):
        x = micro
        for partition_id, part in enumerate(partitions):
            x, boundary_state = part.forward(x)
            # 只缓存边界，内部激活在反向时重算
            forward_cache[(micro_batch_id, partition_id)] = boundary_state

    # 2. 反向：按 micro-batch 逆序回传
    for micro_batch_id in reversed(range(M)):
        grad = None
        for partition_id in reversed(range(len(partitions))):
            part = partitions[partition_id]
            boundary_state = forward_cache[(micro_batch_id, partition_id)]

            # 需要时重材料化：根据边界状态重算本 partition 内部前向
            part.rematerialize(boundary_state)

            grad, param_grad = part.backward(grad)
            grad_accum[partition_id] += param_grad

    # 3. 所有 micro-batch 都完成后，再统一更新
    last_micro_batch = True
    if last_micro_batch:
        for part, g in zip(partitions, grad_accum):
            part.set_grad(g / M)
        optimizer.step()
        optimizer.zero_grad()
```

这段逻辑里最容易被忽略的是两点：

1. `grad_accum += param_grad` 不是可选优化，而是语义核心。
2. `optimizer.step()` 必须在整个 mini-batch 完成后触发，而不是每个 micro-batch 后触发。

下面用表格把几个关键状态对应起来：

| 字段 | 含义 | 什么时候变化 | 是否影响参数更新 |
| --- | --- | --- | --- |
| `partition_id` | 当前是第几个 stage | 激活或梯度跨 stage 流动时变化 | 间接影响 |
| `micro_batch_id` | 当前是 mini-batch 内第几个小批次 | 每推进一个 micro-batch 就变化 | 不直接触发更新 |
| `forward_cache` | 保存边界激活或最小状态 | 前向结束时写入，反向时读取 | 不直接影响 |
| `grad_accum` | 累积后的参数梯度 | 每个 micro-batch 反向后增加 | 直接决定更新值 |
| `last_micro_batch` | 当前 mini-batch 是否已全部完成 | 只在 mini-batch 尾部为真 | 直接触发统一更新 |

一个新手友好的玩具例子是：

- 4 个 stage
- 16 个 micro-batch
- 每个 stage 计算时间大致相同

那么时间推进可以理解成：

- 早期：stage1 先忙，其他 stage 逐步被填满
- 中期：stage1/2/3/4 都在处理不同 micro-batch，吞吐最高
- 后期：最后几个 micro-batch 逐步退出，前面的 stage 先闲下来

一个真实工程例子则是大模型预训练中的 Transformer 堆叠。通常会把若干连续层分给一个 stage，例如：

- stage1: embedding + block 1-12
- stage2: block 13-24
- stage3: block 25-36
- stage4: block 37-48 + lm head

每个 micro-batch 在 stage 间传的是张量边界，不是完整模型。真正需要谨慎的是负载均衡，因为 embedding、attention、MLP、输出头的代价不完全一样，按“层数平均切”不一定等于“FLOPs 平均切”。

---

## 工程权衡与常见坑

GPipe 理论上清晰，但工程上常见问题很多，尤其是吞吐、收敛和内存三者会互相拉扯。

| 问题 | 典型症状 | 原因 | 常见缓解方法 |
| --- | --- | --- | --- |
| micro-batch 太少 | GPU/TPU 经常空闲，吞吐低 | bubble 比例高 | 增大 $M$，经验上常取 $M \ge 4K$ |
| stage 负载不均 | 某一张卡长期最忙，其他卡等待 | 按层均分不等于按算力均分 | 按 FLOPs、激活大小、通信量重新切分 |
| 重材料化过多 | 反向明显变慢 | 重算增加额外前向开销 | 只重算高收益部分，保留关键激活 |
| 梯度累积周期变长 | 更新变稀疏，收敛变慢或不稳 | 等效 batch 变大 | 调整学习率、warmup、动量超参 |
| 边界通信过重 | stage 间传输成为瓶颈 | 激活太大或设备互联慢 | 减少边界宽度，优化切分位置 |
| BatchNorm 类模块异常 | 统计量波动，训练不稳定 | micro-batch 太小导致统计失真 | 改用 LayerNorm、SyncBN 或重新设计统计方式 |

先看 bubble 的问题。设 $K=4, M=4$，空洞约为 $3/7 \approx 43\%$。这意味着不少时间设备都在等。很多人会误以为“已经用了 4 张卡，吞吐应该接近 4 倍”，但流水线不是天然满载，只有当 $M$ 足够大时，设备利用率才会上来。

再看梯度累积的坑。因为参数只在整个 mini-batch 末尾更新一次，所以如果你原本单卡训练时每 64 个样本就更新一次，现在改成 16 个 micro-batch 累积后才更新，那么优化器看到的是 batch size 1024 的梯度统计。学习率计划如果不跟着改，常见后果有两个：

- 学习率太小，训练变慢
- 学习率太大，前期震荡明显

重材料化也不是“越多越好”。它的收益是省激活内存，代价是增加重算时间。对于已经被通信拖慢的 stage，再叠加大量重算，可能反而把整条流水线的最慢工位进一步拖慢。工程上常见做法不是“全量重算”，而是只重算收益高的内部块，边界或代价高的节点保留缓存。

还有一个典型误区是：**梯度累积不等于完全没有同步成本**。GPipe 确实减少了“每个 micro-batch 立刻做全局参数同步”的频率，但 stage 之间仍然要交换边界激活和边界梯度；如果再叠加数据并行或参数分片，还会有更复杂的同步路径。

---

## 替代方案与适用边界

GPipe 不是唯一的流水线并行方案，也不是所有显存问题的默认答案。工程上最常被拿来对比的是 1F1B 和 ZeRO/FSDP。

| 方案 | 适合场景 | 同步频率 | 实现复杂度 | 主要优势 | 主要边界 |
| --- | --- | --- | --- | --- | --- |
| GPipe | 深层、顺序结构明显的大模型 | 通常按 mini-batch 统一更新 | 中等 | 语义清晰，易于保持同步 SGD 等价 | bubble 依赖 $M$，重算开销明显 |
| 1F1B | stage 多、希望降低激活驻留时间 | 仍可保持同步更新，但调度更细 | 较高 | 更早交错前向与反向，常能降低内存和空洞 | 调度复杂，状态管理更难 |
| ZeRO / FSDP | 单卡放不下参数、优化器状态或梯度 | 与数据并行步一致 | 中到高 | 重点解决参数、梯度、优化器状态分片 | 不直接解决层级流水线问题 |

### GPipe 与 1F1B

1F1B 可以先理解为“一个前向、一个反向交替推进”的流水线调度。和 GPipe 相比，它往往能让激活更早释放，也更容易把前向和反向交织得更紧，从而降低部分空洞和显存峰值。

但代价是调度复杂。你要更精细地管理：

- 哪个 stage 当前在做前向还是反向
- 哪个 micro-batch 的激活仍需保留
- 参数版本是否一致

对于零基础到初级工程师，理解 GPipe 更合适，因为它先把“同步梯度累积”这个核心逻辑讲清楚，再去理解更复杂的 1F1B 会更顺。

### GPipe 与 ZeRO/FSDP

ZeRO 或 FSDP 的重点不是把模型按层切成流水线，而是把**参数、梯度、优化器状态**分散到多设备上。白话说，它解决的是“模型状态太大”问题，不是“按层串行执行整网”问题。

所以两者边界不同：

- 如果模型结构天然适合按层切，且单设备激活压力大，GPipe 合适。
- 如果模型不易切 stage，但参数和优化器状态放不下，ZeRO/FSDP 更直接。
- 大型训练里它们还常被组合使用，例如 pipeline parallel + data parallel + optimizer sharding 同时存在。

一个简单对比场景：

- 只有 2 张卡、模型层数不多，但单层很重，这时 1F1B 往往更有效。
- 模型层数很多、天然可顺序切分，且你希望先保证同步训练语义清晰，GPipe 更合适。
- 模型主要是参数量太大，而不是激活链条太长，ZeRO/FSDP 更合适。

所以不要把 GPipe 理解成“多卡训练通用解”。它更准确的定位是：**面向深层顺序网络的同步流水线并行方案**。

---

## 参考资料

1. Huang, Y., Cheng, Y., Bapna, A., et al. *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*. NeurIPS 2019.
2. Emergent Mind. *Pipeline Parallelism Schemes*.
3. Sighing Now. *Pipeline Model Parallelism in LLM Pretraining*.
4. Minjia Zhang. *GPipe 课程复盘材料*.
5. Shagun Sodhani. *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism*.
6. GPipe 原论文 PDF: https://papers.neurips.cc/paper/8305-gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism.pdf
7. EmergentMind 文章: https://www.emergentmind.com/topics/pipeline-parallelism-schemes
8. Pipeline 并行博客: https://sighingnow.github.io/machine%20learning/pipeline-model-parallelism.html
9. 课程资料整理: https://minjiazhang.github.io/courses/fall24-resource/GPipe.pdf
10. 读书笔记: https://shagunsodhani.com/papers-I-read/GPipe-Easy-Scaling-with-Micro-Batch-Pipeline-Parallelism
