## 核心结论

1F1B 是流水线并行中的一种训练调度。这里的“调度”不是新算法，而是一个很具体的问题：**在每个时刻，每个流水线阶段到底做前向、做反向，还是等待通信**。

它的核心价值不是提升单步理论 FLOPs 上限，而是降低训练时的**激活显存峰值**。更准确地说，1F1B 主要把激活占用从 GPipe 常见的

$$
O(M)
$$

压到与流水线深度同阶的

$$
O(P)
$$

附近。

符号先统一：

| 符号 | 含义 | 直白解释 |
|---|---|---|
| $M$ | microbatch 数 | 一个 mini-batch 被切成多少小份 |
| $P$ | pipeline stages 数 | 模型被切成多少段，通常也对应多少个流水线设备 |
| activation | 激活 | 前向产生、反向还要再次使用的中间结果 |
| in-flight microbatch | 在飞 microbatch | 已进入流水线、但尚未完成反向的 microbatch |

GPipe 的典型执行方式是：

1. 先把所有 microbatch 的前向都做完。
2. 再统一开始所有 microbatch 的反向。

这样做的问题很直接：**前向期间产生的大量激活必须一直保留到反向开始**。如果 $M$ 很大，激活显存就会随 $M$ 近似线性增长。

1F1B 则在稳态阶段改成“一次前向、一次反向”交替推进。结果不是“没有激活了”，而是**每个阶段不再无限堆积只做完前向、还没来得及反向的 microbatch**。因此常见近似写法是：

$$
\text{peak activation} \approx O(\text{in-flight microbatches}) \le O(P)
$$

这个结论的重点是：

- 显存并没有变成常数。
- 下降的是**激活这部分**，不是参数、梯度、优化器状态全部消失。
- 激活峰值的主导因素，从 “你切了多少个 microbatch” 变成了 “流水线有多深”。

但 1F1B 不是免费午餐，它成立有两个前提：

1. 训练过程必须经历 `Warmup -> Steady -> Drain` 三个阶段，流水线泡沫仍然存在。
2. 需要 **weight stashing**。也就是：**某个 microbatch 前向看到哪一版参数，后续对应反向就必须回到同一版参数语义上**。否则前向和反向对应的不是同一条计算图，训练含义会变。

---

## 问题定义与边界

流水线并行要解决的问题是：**模型太大，一张卡放不下；或者模型虽放得下，但单卡算不满**。

做法是把模型按层切成多个阶段。例如一个 48 层 Transformer，可以按层切成 4 段，每段放到一张 GPU 上。前一段算完激活后，不等整个模型跑完，就立刻把结果发给下一段。这样不同设备可以像工厂流水线一样并行工作。

但训练不仅有前向，还有反向。反向计算不能凭空进行，它需要前向时留下来的中间值。这里有两个初学者最容易混淆的点：

| 概念 | 是什么 | 为什么占显存 |
|---|---|---|
| 参数 | 模型权重 | 模型本体必须常驻 |
| 梯度 | 参数的导数 | 反向和优化器更新需要 |
| 优化器状态 | 如 Adam 的一阶、二阶矩 | 往往比参数更大 |
| 激活 | 前向中间结果 | 反向求导时要回看 |
| 通信缓冲 | 阶段间传输缓存 | 流水线/并行通信需要 |

1F1B 优化的是第四项，也就是**激活保留方式**。它不直接减少参数、优化器状态，也不替代 ZeRO、张量并行或激活重计算。

下面先看 GPipe 和 1F1B 的对比：

| 策略 | 前向/反向组织方式 | 激活峰值趋势 | 吞吐特征 | 主要问题 |
|---|---|---:|---|---|
| GPipe | 先全部前向，再全部反向 | $O(M)$ | 逻辑直观 | microbatch 多时显存迅速上涨 |
| 1F1B | 稳态时交替执行一次 F 和一次 B | $O(P)$ | 稳态更平衡 | 调度更复杂，需要权重版本管理 |
| 无流水线 | 单卡或纯数据并行 | 由单卡 batch 决定 | 最简单 | 模型规模受单卡约束 |

看一个最小例子。假设：

- 有 4 张 GPU
- 模型切成 4 个阶段，即 $P=4$
- 一个 mini-batch 被切成 16 个 microbatch，即 $M=16$

如果用 GPipe：

- microbatch 1 到 16 会依次流过所有阶段做前向
- 在反向真正开始前，很多中间激活都不能释放
- 越早进入流水线的阶段，越容易积压大量“已前向、未反向”的激活

如果用 1F1B：

- 先经过 warmup 把流水线填起来
- 一旦某个阶段拿到来自后面的梯度，就可以开始反向
- 后续进入稳态，阶段的工作节奏接近 “前向一个，反向一个”
- 于是激活会被更早消费，不再一直堆到 mini-batch 末尾

边界也要说清楚。1F1B 不解决以下问题：

- 模型参数本身已经大到放不下
- Adam 状态占用远超激活
- 通信链路太慢导致流水线等待严重
- microbatch 太小导致单卡 kernel 利用率恶化

所以在真实大模型训练里，1F1B 通常不是单独使用，而是和下面这些手段叠加：

| 手段 | 主要解决什么 |
|---|---|
| 张量并行 | 单层算子过大，单卡算不下 |
| ZeRO / FSDP | 参数、梯度、优化器状态太大 |
| 激活重计算 | 激活显存仍然偏高 |
| 混合精度 | 减少参数、激活、通信成本 |
| 流水线并行 + 1F1B | 模型切分后降低激活驻留峰值 |

真实工程里，几十亿到上百亿参数的 Transformer 往往同时存在这几个问题。1F1B 之所以长期被采用，不是因为它“最先进”，而是因为**它抓住了流水线训练里一个非常硬的瓶颈：激活留存时间过长**。

---

## 核心机制与推导

1F1B 可以分成三个阶段理解：

1. Warmup：先把流水线填满。
2. Steady：进入稳定循环，前向和反向交替推进。
3. Drain：停止注入新前向，把剩余反向全部做完。

### 1. Warmup 为什么必要

流水线初始是空的。以 4 个阶段为例：

- 第 0 阶段先拿到 `mb0` 做前向
- 第 1、2、3 阶段一开始都还没有输入
- 反向更不可能立刻开始，因为最后一段连前向结果都还没拿到

所以必须先注入若干个 microbatch，把流水线“灌满”。这就是 warmup。

如果阶段编号从 0 开始，到 $P-1$ 结束，那么第 $i$ 个阶段在 1F1B 中通常需要做的 warmup 前向次数近似为：

$$
P - i - 1
$$

含义很直白：**越靠前的阶段，离反向信号回来越远，因此必须先做更多次前向，才能等到第一笔可执行的反向任务。**

举例，$P=4$ 时：

| 阶段 | 编号 $i$ | 典型 warmup 前向次数 |
|---|---:|---:|
| 第 0 阶段 | 0 | 3 |
| 第 1 阶段 | 1 | 2 |
| 第 2 阶段 | 2 | 1 |
| 第 3 阶段 | 3 | 0 |

这张表的意义是：1F1B 不是从第一个时刻就天然“前反交替”的。它必须先经历一个不平衡的填充期。

### 2. Steady 为什么能降内存

稳态阶段的关键不是机械地写成 `F, B, F, B, ...`，而是下面这个更本质的条件：

> 一个阶段不会长期只做前向、而不消费已经返回的反向。

GPipe 的问题是先“只进不出”。1F1B 的改进是让“新激活进入”和“旧激活被反向消费”这两件事在稳态里接近平衡。

从库存的角度看：

- GPipe：仓库不断进货，直到所有货都堆满，最后才统一出货
- 1F1B：仓库一边进货，一边发货，库存上限被明显压住

因此常见近似可写成：

$$
\text{activation memory}_{\text{GPipe}} \approx O(M)
$$

$$
\text{activation memory}_{\text{1F1B}} \approx O(P)
$$

为什么是 $O(P)$ 而不是更小？因为流水线再怎么优化，也会有一批 microbatch 正在不同阶段上“在飞”。这些尚未完成反向的 microbatch 仍然需要保留各自的激活，只是其数量不再随着 $M$ 一直线性增长。

再看一个更具体的直观表。假设 $P=4$，稳态时某个中间阶段可能同时面对的对象大致如下：

| 类型 | 数量级 | 说明 |
|---|---:|---|
| 正在做前向的 microbatch | 1 | 当前拍次处理 |
| 正在做反向的 microbatch | 1 | 当前拍次处理 |
| 等待后续梯度返回、因此要保留激活的 microbatch | 与流水线深度同阶 | 不再随 $M$ 无上限累积 |

所以 1F1B 的本质，不是“激活不需要了”，而是**把激活存活时间从“等整个 mini-batch 前向完成”缩短到“等相关反向尽快回来”**。

### 3. 泡沫从哪里来

流水线泡沫（bubble）指的是某些设备在某些时刻没有有效工作。原因不是实现差，而是流水线天然有头尾：

- 刚开始时，后面的阶段还没拿到输入
- 快结束时，前面的阶段已经没有新的前向可做，只能等剩余反向清空

一个常见近似公式是：

$$
\text{bubble ratio} \approx \frac{P - 1}{M}
$$

这个公式不是逐拍精确公式，但非常适合做一阶判断。它说明了两个工程事实：

1. 流水线越深，固定头尾损失越大。
2. microbatch 越多，这部分固定损失越容易被摊薄。

下面代入几个例子：

| $P$ | $M$ | 近似 bubble ratio | 解释 |
|---:|---:|---:|---|
| 4 | 16 | $\frac{3}{16}=18.75\%$ | 可以接受，流水线基本能发挥作用 |
| 8 | 64 | $\frac{7}{64}\approx 10.9\%$ | 较理想，头尾损失被摊薄 |
| 8 | 8 | $\frac{7}{8}=87.5\%$ | 已经偏差，流水线利用率很差 |
| 8 | 2 | $\frac{7}{2}=350\%$ | 说明稳态几乎不存在，调度收益非常有限 |

这里最后一行看上去会超过 100%，因为它本来就是近似“开销与有效工作量的相对规模”，不是严格概率意义上的百分比。它表达的意思是：**当 $M$ 太小时，warmup 和 drain 几乎支配整个过程，1F1B 的稳态优势根本发挥不出来。**

### 4. 为什么必须 weight stashing

这是 1F1B 最容易被一句话带过、但工程上最不能省的部分。

weight stashing 指的是：**为每个在飞 microbatch 保存它前向时看到的参数版本信息，保证后续反向在同一条语义路径上完成。**

前向可以写成：

$$
y = f(x; w^{(t)})
$$

那么对应反向必须计算的是同一个函数图上的梯度：

$$
\frac{\partial \mathcal{L}}{\partial w^{(t)}}
\quad \text{and} \quad
\frac{\partial \mathcal{L}}{\partial x}
\text{ under } w^{(t)}
$$

如果某个 microbatch 的前向用的是 $w^{(t)}$，但反向时系统已经把参数推进到了 $w^{(t+1)}$，那你反向对应的就不再是原来的前向图。直观说，就是：

- 激活来自旧参数路径
- 反向却按新参数路径解释
- 这两个东西不匹配

看一个简化时间线：

| 时刻 | 事件 | 参数版本 |
|---|---|---|
| $t_1$ | `mb0` 做前向 | $w^{(0)}$ |
| $t_2$ | `mb1` 做前向 | $w^{(0)}$ |
| $t_3$ | 某些更新发生 | $w^{(1)}$ |
| $t_4$ | `mb0` 开始反向 | ? |

如果在 $t_4$ 直接用当前系统参数 $w^{(1)}$ 去解释 `mb0` 的反向，那么 `mb0` 的梯度就不再对应它在 $t_1$ 时真正走过的那条计算图。

所以正确做法是：

1. 前向时记录该 microbatch 对应的权重版本。
2. 反向时取回同一版本语义。
3. 再按框架约定的 flush/update 规则推进参数。

这也是为什么很多资料会强调：**1F1B 不只是“前反交替”，而是“前反交替 + 参数版本管理”**。少了后者，训练语义往往已经变了。

---

## 代码实现

下面给一个可以直接运行的 Python 玩具实现。它不依赖 GPU，也不模拟 NCCL 通信，而是专门演示两件最核心的事情：

1. 1F1B 的 `Warmup -> Steady -> Drain` 调度节奏
2. 每个 microbatch 的前向与反向必须绑定到同一权重版本

为了便于检查，我们用最简单的单阶段线性模型：

$$
y = w x
$$

损失设成：

$$
\mathcal{L} = \frac{1}{2}(y - \text{target})^2
$$

于是梯度为：

$$
\frac{\partial \mathcal{L}}{\partial y} = y - \text{target}
$$

$$
\frac{\partial \mathcal{L}}{\partial w} = (y - \text{target})x
$$

完整代码如下：

```python
from collections import deque
from dataclasses import dataclass


@dataclass
class ForwardRecord:
    microbatch_id: int
    version: int
    weight_snapshot: float
    x: float
    y: float
    target: float


class Stage:
    def __init__(self, stage_id: int, init_weight: float = 2.0, lr: float = 0.1):
        self.stage_id = stage_id
        self.weight = init_weight
        self.lr = lr
        self.version = 0
        self.stash = deque()
        self.grad_log = []

    def forward(self, microbatch_id: int, x: float, target: float) -> float:
        y = self.weight * x
        record = ForwardRecord(
            microbatch_id=microbatch_id,
            version=self.version,
            weight_snapshot=self.weight,
            x=x,
            y=y,
            target=target,
        )
        self.stash.append(record)
        return y

    def backward(self, microbatch_id: int):
        if not self.stash:
            raise RuntimeError("stash is empty, backward has no matching forward")

        rec = self.stash.popleft()
        assert rec.microbatch_id == microbatch_id, (
            f"microbatch mismatch: expect {rec.microbatch_id}, got {microbatch_id}"
        )

        # dL/dy for L = 0.5 * (y - target)^2
        grad_y = rec.y - rec.target
        grad_w = grad_y * rec.x
        grad_x = grad_y * rec.weight_snapshot

        self.grad_log.append(
            {
                "microbatch_id": microbatch_id,
                "version_used_in_backward": rec.version,
                "weight_snapshot": rec.weight_snapshot,
                "grad_w": grad_w,
                "grad_x": grad_x,
            }
        )
        return rec.version, rec.weight_snapshot, grad_w, grad_x

    def commit(self, grad_w: float):
        self.weight -= self.lr * grad_w
        self.version += 1


def simulate_1f1b(num_microbatches=6, warmup_steps=2):
    stage = Stage(stage_id=0, init_weight=2.0, lr=0.1)
    xs = [1.0, 2.0, 1.5, 0.5, 3.0, 2.5]
    targets = [1.0, 0.0, 2.0, 1.0, -1.0, 0.5]

    events = []

    # Warmup
    for mb in range(min(warmup_steps, num_microbatches)):
        y = stage.forward(mb, xs[mb], targets[mb])
        events.append(
            {
                "phase": "warmup",
                "op": "F",
                "microbatch": mb,
                "weight_version": stage.version,
                "weight_value": stage.weight,
                "y": y,
                "stash_size": len(stage.stash),
            }
        )

    # Steady: each new forward is followed by one backward
    for mb in range(warmup_steps, num_microbatches):
        y = stage.forward(mb, xs[mb], targets[mb])
        events.append(
            {
                "phase": "steady",
                "op": "F",
                "microbatch": mb,
                "weight_version": stage.version,
                "weight_value": stage.weight,
                "y": y,
                "stash_size": len(stage.stash),
            }
        )

        bwd_mb = mb - warmup_steps
        version, weight_snapshot, grad_w, _ = stage.backward(bwd_mb)
        events.append(
            {
                "phase": "steady",
                "op": "B",
                "microbatch": bwd_mb,
                "weight_version_used": version,
                "weight_snapshot": weight_snapshot,
                "grad_w": grad_w,
                "stash_size": len(stage.stash),
            }
        )
        stage.commit(grad_w)
        events.append(
            {
                "phase": "steady",
                "op": "commit",
                "new_version": stage.version,
                "new_weight": stage.weight,
            }
        )

    # Drain: finish remaining backward passes
    next_bwd_mb = num_microbatches - warmup_steps
    while stage.stash:
        version, weight_snapshot, grad_w, _ = stage.backward(next_bwd_mb)
        events.append(
            {
                "phase": "drain",
                "op": "B",
                "microbatch": next_bwd_mb,
                "weight_version_used": version,
                "weight_snapshot": weight_snapshot,
                "grad_w": grad_w,
                "stash_size": len(stage.stash),
            }
        )
        stage.commit(grad_w)
        events.append(
            {
                "phase": "drain",
                "op": "commit",
                "new_version": stage.version,
                "new_weight": stage.weight,
            }
        )
        next_bwd_mb += 1

    assert stage.version == num_microbatches
    assert len(stage.stash) == 0

    return events, stage.grad_log, stage.weight, stage.version


if __name__ == "__main__":
    events, grad_log, final_weight, final_version = simulate_1f1b()

    print("=== schedule events ===")
    for e in events:
        print(e)

    print("\n=== backward log ===")
    for g in grad_log:
        print(g)

    print("\nfinal_weight =", final_weight)
    print("final_version =", final_version)

    # Check that earlier microbatches really use earlier versions.
    assert grad_log[0]["version_used_in_backward"] == 0
    assert grad_log[1]["version_used_in_backward"] == 0
    assert grad_log[2]["version_used_in_backward"] == 1
```

这段代码可以直接运行，输出会包含三类信息：

| 输出块 | 作用 |
|---|---|
| `schedule events` | 展示 warmup / steady / drain 的调度过程 |
| `backward log` | 展示每个 microbatch 的反向到底用了哪一版权重 |
| `final_weight/final_version` | 验证整个训练过程确实推进了参数版本 |

这个玩具实现虽然只有单阶段，但已经把 1F1B 最本质的约束保留下来了：

1. `stash` 必须是按 microbatch 顺序入队、出队的。
2. `backward()` 不能直接读取“当前权重”，而是要回到该 microbatch 前向时对应的那份语义记录。

如果你把 `stash` 去掉，改成在 `backward()` 里直接访问当前 `self.weight`，那么下面这件事就会发生：

- 早期 microbatch 的前向发生在旧参数版本
- 反向却拿到新参数版本
- 梯度不再对应同一条前向路径

这正是 weight stashing 不能省的原因。

下面再给一个更贴近真实系统的伪代码。它展示的是**多阶段流水线**里的控制主线，而不是数值细节：

```python
from collections import deque


def run_1f1b(stage_id, num_stages, num_microbatches):
    warmup_steps = min(num_stages - stage_id - 1, num_microbatches)
    stash_queue = deque()

    # Warmup
    for _ in range(warmup_steps):
        mb = recv_forward_id()
        x = recv_forward_tensor(mb)

        version = current_weight_version()
        stash_queue.append((mb, version))

        y = forward_compute(x, weight_of(version))
        send_forward_tensor(mb, y)

    # Steady
    for _ in range(num_microbatches - warmup_steps):
        # one forward
        fwd_mb = recv_forward_id()
        x = recv_forward_tensor(fwd_mb)

        version = current_weight_version()
        stash_queue.append((fwd_mb, version))

        y = forward_compute(x, weight_of(version))
        send_forward_tensor(fwd_mb, y)

        # one backward
        bwd_mb, grad_out = recv_backward_tensor()
        saved_mb, saved_version = stash_queue.popleft()
        assert bwd_mb == saved_mb

        grad_in, grad_w = backward_compute(grad_out, weight_of(saved_version))
        send_backward_tensor(bwd_mb, grad_in)
        accumulate_grad(saved_version, grad_w)

    # Drain
    while stash_queue:
        bwd_mb, grad_out = recv_backward_tensor()
        saved_mb, saved_version = stash_queue.popleft()
        assert bwd_mb == saved_mb

        grad_in, grad_w = backward_compute(grad_out, weight_of(saved_version))
        send_backward_tensor(bwd_mb, grad_in)
        accumulate_grad(saved_version, grad_w)

    optimizer_step()
    advance_weight_version()
```

这个伪代码里有三个容易被忽略但非常关键的点：

| 位置 | 容易忽略什么 | 后果 |
|---|---|---|
| `stash_queue.append((fwd_mb, version))` | 没保存版本 | 反向会错用当前权重 |
| `assert bwd_mb == saved_mb` | 没做 microbatch 对齐检查 | 前后向配对可能错乱 |
| `optimizer_step()` 放在 flush 逻辑之后 | 过早更新参数 | 训练语义变成异步漂移 |

再看一个 queue 状态表，帮助把“当前版本”和“历史版本”分开理解：

| 时刻 | 操作 | 当前系统参数版本 | stash queue |
|---|---|---:|---|
| t1 | F(mb0) | v0 | [mb0:v0] |
| t2 | F(mb1) | v0 | [mb0:v0, mb1:v0] |
| t3 | B(mb0) | v0 | [mb1:v0] |
| t4 | commit | v1 | [mb1:v0] |
| t5 | F(mb2) | v1 | [mb1:v0, mb2:v1] |
| t6 | B(mb1) | v1 | [mb2:v1] |

这张表最重要的结论是：

> “系统当前在用什么版本的参数” 和 “某个历史 microbatch 的反向应该对应什么版本” 是两条不同的时间线。

真实工程实现里，Megatron 一类框架会把这些动作统一封装进调度器中，例如把前向、反向、激活发送、梯度接收、weight version 管理以及通信重叠统一调度。`combined_1f1b` 这类命名，通常表示“在一个总调度框架里组织完整的 1F1B 执行流”，而不是只写一个简单的 for 循环。

---

## 工程权衡与常见坑

1F1B 的收益很明确，但它不是简单把 GPipe 替换一下就结束。它的代价同样真实。

先看一个总对比：

| 问题 | 有 weight stashing | 无 weight stashing |
|---|---|---|
| 前向/反向版本一致性 | 可以保证 | 通常会被破坏 |
| 梯度语义 | 更接近同步 flush 语义 | 往往变成异步漂移 |
| 实现复杂度 | 更高 | 低，但不可靠 |
| 长时间稳定训练 | 可以 | 风险很大 |

第一个常见坑是：**把“显存下降”误解成“训练语义自动不变”**。

这是错误的。显存优化和训练语义是两回事。1F1B 确实能减少激活驻留量，但如果你没有正确管理参数版本，那么你得到的是“另一种训练过程”，而不是原本 mini-batch 语义下的等价实现。

第二个坑是：**只看稳态，不看 warmup 和 drain**。

很多入门图只画中间那段 `F/B/F/B`，看起来每张卡都满负荷工作。但真实运行必然有开头和结尾。只要有头尾，泡沫就不可能完全消失，只能靠更大的 $M$ 去摊薄：

$$
M \gg P \Rightarrow \frac{P-1}{M} \to 0
$$

如果做不到，比如：

- $P=8$
- $M=2$

那就算你实现了 1F1B，也几乎吃不到稳态收益。

第三个坑是：**为了减泡沫，把 microbatch 切得过小**。

这也是新手经常反直觉的地方。你可能觉得：

- $M$ 大一点，泡沫更小
- 所以越大越好

但真实情况不是单调的。因为 $M$ 变大时，单个 microbatch 会变小，而小到一定程度后：

- GEMM 尺寸变差
- kernel launch 开销占比上升
- Tensor Core 利用率下降
- 通信频率变高

于是性能经常出现下面这种形状：

| $M$ 区间 | 现象 |
|---|---|
| 太小 | 泡沫大，设备空转明显 |
| 适中 | 泡沫可接受，kernel 效率也还好 |
| 太大 | microbatch 太碎，单卡效率和通信效率都下降 |

第四个坑是：**只盯激活显存，不看通信成本**。

流水线并行天然需要阶段间传输激活和梯度。1F1B 稳态下前向、反向都在持续推进，这意味着链路被频繁使用。如果互联较弱，比如：

- PCIe 带宽不足
- 跨节点网络延迟偏高
- 计算与通信缺少有效重叠

那么理论上的调度优势会被通信等待吃掉。

第五个坑是：**误以为 1F1B 能单独解决所有显存问题**。

它只控制“激活如何留、留多久”。如果主要瓶颈其实是：

- 参数本身太大
- Adam 状态过大
- 梯度缓存过大

那仅靠 1F1B 不够，必须叠加其他手段。

下面用三阶段视角再看一次 bubble footprint：

| 阶段 | 新前向进入 | 反向回收 | 哪些设备更容易空转 |
|---|---|---|---|
| Warmup | 多 | 无或少 | 靠近尾部的阶段 |
| Steady | 持续 | 持续 | 最少 |
| Drain | 无 | 多 | 靠近头部的阶段 |

可以把它记成一句简单的话：

> warmup 时“后面等前面”，drain 时“前面等后面”，只有 steady 才是流水线真正最值钱的区间。

---

## 替代方案与适用边界

1F1B 不是唯一选项。它只是在“已经决定做流水线并行”的前提下，一个非常重要的调度方案。

先看横向比较：

| 策略 | 适用场景 | 激活峰值趋势 | 优点 | 局限 |
|---|---|---:|---|---|
| GPipe | 显存尚可、优先实现简单 | $O(M)$ | 逻辑最直观 | microbatch 多时激活膨胀明显 |
| 1F1B | 长流水线、大模型训练 | $O(P)$ | 激活更稳，稳态吞吐更合理 | 需要版本管理与更复杂调度 |
| 激活重计算 | 显存紧、可接受额外算力 | 可进一步降低 | 不必改变并行拓扑 | 增加重复计算时间 |
| ZeRO / FSDP | 参数、梯度、优化器状态是主瓶颈 | 不直接针对激活 | 状态显存下降显著 | 不能替代流水线调度 |
| 混合并行 | 超大模型 | 取决于组合方式 | 扩展性最好 | 系统复杂度最高 |

什么时候 1F1B 值得上，通常满足以下条件：

| 条件 | 为什么重要 |
|---|---|
| 模型必须切到多卡才能放下 | 否则流水线本身就未必必要 |
| $M$ 明显大于 $P$ | 否则泡沫无法被摊薄 |
| 激活是主要显存瓶颈 | 这样 1F1B 的收益最直接 |
| 互联和通信重叠还可以 | 否则调度收益会被通信吞掉 |
| 团队能维护调度与版本逻辑 | 否则工程风险高 |

什么时候它不一定值得上：

- 单卡已经能放下模型和目标 batch
- 只有很少的 microbatch，却切出很深的流水线
- 机器互联较弱，通信等待很重
- 团队当前更需要简单、可定位问题的训练系统

一个新手常见误区是把 1F1B 和 ZeRO 看成替代关系。其实它们解决的问题不同：

| 方案 | 主目标 |
|---|---|
| 1F1B | 降低激活驻留峰值，优化流水线调度 |
| ZeRO / FSDP | 分片参数、梯度、优化器状态 |
| 激活重计算 | 用额外计算换更低激活显存 |

所以在真实系统里，它们往往是叠加关系，而不是二选一。

再给两个判断例子：

| 场景 | 判断 |
|---|---|
| $M=64,\;P=8$ | 泡沫约 $7/64$，1F1B 往往是合理选择 |
| $M=2,\;P=8$ | 泡沫极重，稳态几乎不存在，1F1B 很难发挥优势 |

至于 `combined_1f1b` 这类工程实现，通常适合在下面几种情况下引入：

- 你已经确定要做流水线并行
- 前向、反向、通信重叠都对最终吞吐很关键
- 团队可以接受更复杂的运行时控制流

如果只是做教学实验、小模型验证，或者你当前的主要目标是快速定位 correctness 问题，那么更朴素的调度方式反而更容易调试。

最后可以把选型逻辑简化成一句判断：

> 如果主瓶颈是激活驻留，并且流水线足够长、microbatch 足够多、通信条件也尚可，那么 1F1B 往往是值得上的；否则它可能只是把系统复杂度提前引入。

---

## 参考资料

下面按“先读什么、解决什么问题”来整理。

| 来源 | 侧重点 | 适合解决什么问题 |
|---|---|---|
| PipeDream: Generalized Pipeline Parallelism for DNN Training | 1F1B、pipeline flush、weight stashing 的正式来源 | 想建立严格概念时先读论文 |
| GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | GPipe 的批处理式流水线 | 想对比为什么 GPipe 会产生 $O(M)$ 激活占用 |
| NVIDIA Megatron Core 文档中的 pipeline parallel 调度接口 | 工程实现、`combined_1f1b`、实际 API 组织方式 | 想理解真实系统如何把调度落地 |
| 各类 warmup / steady / drain 分析文章 | 三阶段时间线、泡沫直觉、拍点图 | 刚入门时建立执行过程感知 |
| weight stashing 相关训练笔记 | 为什么版本管理不能省 | 容易卡在“为什么一定要存版本”时阅读 |

建议阅读顺序：

1. 先读 GPipe 和 PipeDream，建立“为什么需要流水线调度差异”的框架。
2. 再读 warmup / steady / drain 的解释型文章，把时间线看明白。
3. 最后回到 Megatron Core 一类工程文档，对照真实实现理解 `combined_1f1b`。

推荐检索标题：

- `GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism`
- `PipeDream: Generalized Pipeline Parallelism for DNN Training`
- `Megatron Core pipeline parallel combined_1f1b`
- `pipeline parallel warmup steady drain`
- `weight stashing pipeline training`

如果你是第一次接触这个主题，阅读时可以重点盯住三件事：

| 阅读目标 | 具体看什么 |
|---|---|
| 看懂为什么 1F1B 能降内存 | 对比 GPipe 与 1F1B 的激活存活时间 |
| 看懂为什么不能省 weight stashing | 前向参数版本与反向版本必须一致 |
| 看懂什么时候 1F1B 不划算 | 泡沫公式、microbatch 过小、通信瓶颈 |
