## 核心结论

异步流水线并行的核心问题，不是“算得快不快”，而是“同一个 micro-batch 的前向和反向是不是在同一份权重上完成”。如果前向用的是 $W^{(t)}$，反向却在参数已经更新后读到了 $W^{(t+s)}$，那么这个梯度就不再严格对应原来的前向计算图，这个现象叫 **weight staleness**，也就是“权重版本变旧”，白话说就是前后两次计算对不上同一套参数。

PipeDream 的做法是 **weight stashing**，即“把历史权重暂存起来”，让某个 micro-batch 的反向仍然使用它前向当时那一版参数。这样能保持单个 micro-batch 内部的一致性，但代价是要保存多份权重副本，显存压力线性上升。

PipeDream-Flush 的做法是 **周期性清空流水线**，也就是每处理完一批设定好的 micro-batch，就停止继续灌入新任务，等流水线里的前向和反向全部完成，再统一更新并开始下一轮。它不是靠保存很多旧权重解决问题，而是靠“让不同 stage 重新对齐时间”把陈旧版本消掉，代价是会产生 bubble，也就是“流水线空转时隙”。

下面这张表先给出三种策略的结论：

| 方案 | 前向/反向权重版本 | 显存占用 | Bubble | 收敛稳定性 |
|---|---|---:|---:|---|
| 无一致性控制 | 前向 $W^{(t)}$，反向可能是 $W^{(t+s)}$ | 低 | 低 | 风险最高 |
| Weight Stashing | 同一 micro-batch 都用 $W^{(t)}$ | 高 | 低 | 较稳定 |
| PipeDream-Flush | flush 边界内统一到同一全局版本 | 中 | 高 | 最接近同步训练 |

一个必须记住的玩具例子是：三阶段流水线、1F1B 调度下，micro-batch 3 的前向在 $t=3$ 用到 $W^{(3)}$，等它的反向回来时，系统可能已经更新到了 $W^{(5)}$。如果没有 stashing，这个梯度就建立在“前向和反向不是同一份模型”的事实之上；如果用了 stashing，反向还能拿回 $W^{(3)}$；如果用了 Flush，并且每 $P=3$ 个 micro-batch flush 一次，那么在下一个周期开始前，各 stage 会重新对齐到同一个最新版本。

---

## 问题定义与边界

**流水线并行**是“把模型按层切成多个阶段，让不同 GPU 像装配线一样同时处理不同 micro-batch”。它的目的很直接：提高设备利用率，让前面的 stage 不必等后面的 stage 完成才能接新数据。

问题出在异步。设某个 micro-batch $m$ 的前向使用参数 $W^{(t)}$，但由于流水线里还有别的 micro-batch 在更后面的位置执行，系统可能已经在若干 stage 上进行了更新。当 $m$ 的反向到来时，它读到的可能已经是：

$$
W_{\text{bwd}} = W^{(t+s)}, \quad s > 0
$$

其中 $s$ 是**版本延迟**，白话说就是“反向比前向晚了多少次更新”。

如果训练是严格同步的，那么理想状态应该是：

$$
W_{\text{fwd}} = W_{\text{bwd}} = W^{(t)}
$$

也就是同一个样本的损失、激活值、梯度都建立在同一组权重上。异步流水线破坏的，正是这个条件。

这个问题的边界由四个因素决定：

| 因素 | 含义 | 对 staleness 的影响 |
|---|---|---|
| stage 数 | 模型被切成几段 | 越多，前后距离越长 |
| micro-batch 数 | 一次灌入流水线的小批次数 | 越多，版本交错越明显 |
| 调度方式 | 如 1F1B | 决定前反向是否交叠 |
| 更新策略 | 每步更新还是 flush 后统一更新 | 决定 $s$ 是否扩大 |

对新手，一个直白类比是：四人接力赛里，前两个人已经按旧鞋跑完了，第三个人准备接棒时，教练突然把全队鞋都换成了新款。最后复盘整场比赛时，你已经无法说“这一棒到底是靠哪双鞋跑出来的”。这里的“鞋”就是权重版本。

再看一个时间线示意。假设 3 个 stage，采用 1F1B：

```text
t1:  MB1-F stage1
t2:  MB1-F stage2 | MB2-F stage1
t3:  MB1-F stage3 | MB2-F stage2 | MB3-F stage1
t4:  MB1-B stage3 | MB2-F stage3 | MB3-F stage2
t5:  MB1-B stage2 | MB2-B stage3 | MB3-F stage3
t6:  MB1-B stage1 | MB2-B stage2 | MB3-B stage3
```

如果某些 stage 在局部反向完成后立刻更新，那么 MB3 前向看到的是较早版本，MB3 反向看到的却可能是更新后的版本。只要“更新发生在它的前向和反向之间”，staleness 就出现了。

---

## 核心机制与推导

先写清楚不一致是怎么来的。设第 $k$ 个 micro-batch 的前向在逻辑时间 $t_k^{f}$ 执行，反向在 $t_k^{b}$ 执行。如果系统在这两者之间发生了 $\Delta_k$ 次参数更新，则：

$$
W_k^{f} = W^{(t_k^{f})}, \qquad W_k^{b} = W^{(t_k^{f} + \Delta_k)}
$$

当 $\Delta_k > 0$ 时，前后不一致。把 $s_k = \Delta_k$，就得到更常见的写法：

$$
W_k^{b} = W_k^{f+s_k}
$$

### 1. PipeDream 的 weight stashing

它的思想很朴素：前向时不仅算激活，还把当前权重版本号甚至完整副本与该 micro-batch 绑定。等反向回来，不去读“现在最新”的参数，而是回到当时那份。

于是对每个 micro-batch，都强制满足：

$$
W_k^{f} = W_k^{b} = W^{(v_k)}
$$

其中 $v_k$ 是这个 micro-batch 在前向时绑定的版本号。

这不是让全局所有 micro-batch 同步，而是让“单个 micro-batch 内部一致”。所以它解决的是 **局部一致性**。

代价也直接：如果一个 stage 最多同时悬挂 $M$ 个尚未完成反向的 micro-batch，就可能需要近似保存 $M$ 份参数快照。若该 stage 参数量为 $|\theta|$，则 stash 的额外内存近似为：

$$
\text{Memory}_{stash} \approx M \cdot |\theta|
$$

实际工程里通常会做共享存储、版本索引、增量保存等优化，但量级关系不会变。

### 2. PipeDream-Flush 的周期同步

Flush 的思想不是“保存旧版本”，而是“不要让新旧版本一直混在同一条流水线上”。做法是按周期灌入 $P$ 个 micro-batch，然后停止注入新任务，等流水线内部全部完成，再统一进入下一轮。

于是每个 flush 周期都形成一个边界：

```text
[灌入 P 个 micro-batch] -> [排空流水线] -> [统一更新/进入下一周期]
```

如果这个边界执行严格，那么周期内部不会再不断漂移到新的全局权重版本。可以把它理解为把长期存在的 $s>0$ 压回到周期边界上，使跨周期的版本不再交叉污染。工程上常把它近似理解为“在 flush 边界把 $s \to 0$”。

### 3. Bubble 与吞吐的关系

**Bubble** 是“某些 stage 无活可干的空档”。白话说就是 GPU 亮着，但这一拍没有有效计算。

假设流水线有 $n$ 个 stage，一个 flush 周期处理 $P$ 个 micro-batch，则理想化吞吐利用率常写成：

$$
\text{Utilization} \approx \frac{P}{P + n - 1}
$$

这里的 $n-1$ 可以理解为填充和排空阶段带来的额外空档。于是有两个直接结论：

| $P$ 相对 $n$ 的大小 | Bubble 比例 | 吞吐表现 |
|---|---|---|
| $P \ll n$ | 高 | 很差 |
| $P \approx n$ | 中 | 可接受 |
| $P \gg n$ | 低 | 更高 |

所以 Flush 不是“免费同步”。你用更短的周期换来了更好的版本一致性，同时也引入更频繁的流水线排空。

### 4. 玩具例子

设 3 个 stage，每个 stage 一个时间单位，采用 1F1B，micro-batch 3 前向在 $t=3$ 使用 $W^{(3)}$。当它的反向在 $t=6$ 到达时，若系统已经因其他 micro-batch 更新两次，则它读到的是 $W^{(5)}$，即：

$$
s = 5 - 3 = 2
$$

这时梯度并不对应原本那次前向。  
如果启用 stashing，MB3 的反向仍读取缓存的 $W^{(3)}$。  
如果采用 Flush 且周期 $P=3$，那么在这个周期排空前不会继续灌入新的版本漂移，周期边界后所有 stage 再统一到下一版全局参数。

---

## 代码实现

下面用一个简化的可运行 Python 例子演示两个关键点：

1. 前向时把权重版本和 micro-batch 绑定。
2. 反向时按 micro-batch id 取回对应版本，而不是直接读当前最新权重。

```python
from dataclasses import dataclass
from collections import deque

@dataclass
class WeightVersion:
    version: int
    value: float

class PipelineStage:
    def __init__(self, init_weight: float):
        self.current = WeightVersion(version=0, value=init_weight)
        self.stash = {}  # micro_batch_id -> WeightVersion
        self.inflight = deque()
        self.flush_boundary = []

    def forward(self, micro_batch_id: int, x: float) -> float:
        # 记录该 micro-batch 前向时所见的权重版本
        self.stash[micro_batch_id] = WeightVersion(
            version=self.current.version,
            value=self.current.value,
        )
        self.inflight.append(micro_batch_id)
        return self.current.value * x

    def local_update(self, grad: float, lr: float = 0.1):
        # 模拟异步局部更新：当前最新权重会继续前进
        new_value = self.current.value - lr * grad
        self.current = WeightVersion(
            version=self.current.version + 1,
            value=new_value,
        )

    def backward_without_stash(self, micro_batch_id: int, x: float) -> tuple[int, float]:
        # 错误做法：直接使用“当前最新”权重
        grad = x
        return self.current.version, grad

    def backward_with_stash(self, micro_batch_id: int, x: float) -> tuple[int, float]:
        # 正确做法：回到该 micro-batch 前向时绑定的版本
        w = self.stash[micro_batch_id]
        grad = x
        return w.version, grad

    def flush(self):
        # 模拟 flush：等待 inflight 任务完成，再清理版本引用
        while self.inflight:
            micro_batch_id = self.inflight.popleft()
            self.stash.pop(micro_batch_id, None)

# 玩具例子：MB3 前向看见 version=3，反向时当前已推进到 version=5
stage = PipelineStage(init_weight=2.0)

# 先推进到 version 3
for _ in range(3):
    stage.local_update(grad=1.0)

y = stage.forward(micro_batch_id=3, x=10.0)
assert y == stage.current.value * 10.0
assert stage.stash[3].version == 3

# 再推进到 version 5，制造 staleness
for _ in range(2):
    stage.local_update(grad=1.0)

version_no_stash, grad1 = stage.backward_without_stash(micro_batch_id=3, x=10.0)
version_stash, grad2 = stage.backward_with_stash(micro_batch_id=3, x=10.0)

assert version_no_stash == 5
assert version_stash == 3
assert grad1 == grad2 == 10.0

stage.flush()
assert 3 not in stage.stash
```

这个例子虽然简化掉了激活缓存、链式求导、多 stage 通信，但核心机制已经完整：

| 操作 | 无 stash | 有 stash |
|---|---|---|
| 前向 | 读取当前版本 | 读取当前版本并登记 |
| 反向 | 读取“现在最新”版本 | 读取登记时的历史版本 |
| 结果 | 版本可能漂移 | 同一 micro-batch 内一致 |

真实系统里的实现通常会再加三层结构。

第一层是 **版本索引表**，把 `micro_batch_id -> version_id` 映射下来，避免每次都保存完整模型副本。  
第二层是 **环形缓冲区**，也就是固定大小的循环队列，用来复用 stash 槽位，避免动态分配造成碎片。  
第三层是 **flush 控制器**，在达到周期阈值时暂停新 micro-batch 注入，等待所有 stage 排空。

一个接近工程伪代码的流程如下：

```python
for micro_id in range(num_micro_batches):
    if not flushing:
        version_id = current_version
        version_map[micro_id] = version_id
        stash_buffer[version_id] = snapshot_if_needed(weights)
        forward(micro_id, weights)

    if ready_for_backward(micro_id):
        version_id = version_map[micro_id]
        backward_weights = stash_buffer[version_id]
        backward(micro_id, backward_weights)

    if should_flush(micro_id, period=P):
        flushing = True
        drain_pipeline()
        sync_all_stages()
        reclaim_old_versions()
        flushing = False
```

“前向写入，后向读取”这条约束，是保证一致性的核心。只要反向绕过这条约束，直接去读当前全局参数，staleness 就会重新出现。

真实工程例子是训练 70 亿参数语言模型时的流水线并行。此时单个 stage 的参数量已经很大，如果每个悬挂 micro-batch 都保留完整 stash，显存会非常紧。于是系统往往会在 PipeDream-2BW 这类设计里，把流水线并行、数据并行、激活重计算、优化器分片一起使用，目的不是某一种技术单独解决问题，而是在“吞吐、显存、收敛”三者之间找到可运行区间。

---

## 工程权衡与常见坑

真正落地时，问题通常不是“不知道原理”，而是“参数一大，原理不再便宜”。

最常见的权衡如下：

| 问题 | 现象 | 根因 | 常见处理 |
|---|---|---|---|
| 显存爆满 | GPU OOM | stash 副本过多 | 缩短 inflight、减少 micro-batch、配合分片 |
| 吞吐下降 | GPU 空转变多 | flush 太频繁 | 增大 $P$，降低 bubble 比例 |
| 收敛不稳 | loss 抖动或退化 | 不同 stage 的 $s$ 分布不同 | 强同步边界、延迟补偿、统一更新点 |
| 调试困难 | 结果偶发不一致 | 版本映射与回收出错 | 显式记录 version id，严格生命周期管理 |

### 1. Stashing 不只占权重内存

很多初学者以为 stash 只是“多存几份参数”，但工程上还要配合保存激活、优化器状态引用、通信缓冲区。70 亿参数模型即使使用半精度，单份权重也已经非常大，再乘以多个 inflight micro-batch，很快就会逼近显存上限。

### 2. Flush 周期太短会把流水线打成“半同步串行”

Flush 的确能改善一致性，但如果 $P$ 太小，流水线还没进入高利用率就开始排空。比如 stage 数为 8，而每次只灌入 4 个 micro-batch，很多时间都花在填充和排空上，吞吐会明显下降。

### 3. 不同 stage 的陈旧度并不相同

这点很重要。实际系统中，靠前的 stage 和靠后的 stage 看到的版本延迟分布可能不一样。也就是说，某些层总是在更“旧”的上下文中计算梯度，而另一些层相对更新。这会带来层间优化行为不一致，表现为训练不稳定甚至最终精度下降。

### 4. 版本回收比版本保存更容易出 bug

保存一份版本不难，难的是知道什么时候可以安全释放。  
如果某个 micro-batch 的反向尚未完成，你提前回收了它对应的 stash，轻则结果错误，重则直接读到被覆盖的数据。工程上通常会给每个版本增加引用计数或完成标记，而不是凭经验“估计应该用完了”。

---

## 替代方案与适用边界

不是所有场景都值得为异步流水线处理 staleness。很多时候，换一种并行策略更便宜。

| 方案 | 主要机制 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| 异步 pipeline + stashing | 绑定历史版本 | 吞吐高 | 显存高 | stage 少、micro-batch 中等、追求吞吐 |
| PipeDream-Flush | 周期排空同步 | 收敛更稳 | bubble 多 | 能接受周期停顿、希望更接近同步 |
| 同步 pipeline | 全局更新点一致 | 语义简单 | 并行效率偏低 | 新手实现、稳定性优先 |
| 仅数据并行/ZeRO | 模型不切流水线 | 训练逻辑简单 | 单卡放不下大模型时受限 | 模型还没大到必须 pipeline |
| 更细粒度模型并行 | 张量/算子级切分 | 降低单卡内存压力 | 通信复杂 | 高速互联、系统栈较成熟 |

对初学者，一个非常实用的判断标准是：

如果你现在连“一个 micro-batch 的前向和反向到底用的是哪一版权重”都很难追踪清楚，那么不要优先上异步流水线。先用同步 pipeline，或者干脆先用更小 batch、只做数据并行，把训练语义跑通。

如果硬件放不下多份权重副本，但你又必须做 pipeline，那么 Flush 往往比盲目 stashing 更现实，因为它把问题从“多存很多旧版本”转成“接受一些空转时间”。

如果你的目标是最高吞吐，而且 stage 数不多、显存预算足够，那么 stashing 的价值会更高，因为它能减少 flush 带来的 bubble。

本质上，选择标准只有一句话：  
你是在用显存买一致性，还是用空转时间买一致性。

---

## 参考资料

1. Narayanan, Deepak, et al. *PipeDream: Fast and Efficient Pipeline Parallel DNN Training*. SOSP 2019. 该论文系统定义了异步流水线中的 weight staleness，并提出 weight stashing 作为核心解决方案。链接：https://aaronharlap.github.io/papers/pipedream-full.pdf

2. Narayanan, Deepak, et al. *Memory-Efficient Pipeline-Parallel DNN Training*. ICML 2021. 常被称为 PipeDream-2BW，重点讨论如何在更低内存开销下做稳定的流水线训练，并扩展到更现实的大模型训练场景。链接：https://www.microsoft.com/en-us/research/publication/memory-efficient-pipeline-parallel-dnn-training/

3. Device Report 对 PipeDream-Flush 与相关流水线同步策略的整理。适合从工程角度理解 flush 如何通过周期性排空来降低陈旧权重问题。链接：https://device.report/m/ea07a1ee072f1fd169aa7fe48329806e0e8dd55fa300ecdeccad3818a4fd9755_doc

4. 关于大模型并行与内存优化的技术博客综述。适合初学者建立 pipeline、张量并行、内存优化之间的整体关系，再回头理解 stashing 与 flush 的取舍。链接：https://syhya.github.io/posts/2025-03-01-train-llm/
