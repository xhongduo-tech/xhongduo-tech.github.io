## 核心结论

DP 指数据并行，意思是多张卡各自算一份前向和反向，再把梯度同步后统一更新参数。DDP 的核心优化不是“把所有梯度算完再一次性同步”，而是把梯度按固定大小装进 bucket，中文可理解为“通信桶”。当一个桶里的梯度已经凑够并且前一个桶已经启动通信时，DDP 就立刻发起这个桶的 AllReduce，也就是“多卡把同名梯度求和再取平均”的集体通信。

这件事的价值在于把通信塞进反向传播的空隙里。后面的层先产生梯度，前面的层还在继续反向计算，因此通信和计算可以并行。简化写法是：

$$
\sum_{i \in \mathcal{P}_k} g_i \ge B
$$

且

$$
t_k = \max(t_{k-1}, \operatorname{grad\_ready}(\mathcal{P}_k))
$$

其中 $g_i$ 是参数梯度大小，$B$ 是桶阈值，$\mathcal{P}_k$ 是第 $k$ 个桶的参数集合，$t_k$ 是第 $k$ 个桶最早能启动通信的时刻。直观条件可以记成：

$$
\text{communication\_start} \ge \text{bucket\_full} \land \text{previous\_bucket\_started}
$$

新手可以先用一个不严格但够用的画面理解：梯度像水流，bucket 像水桶，桶一满就先送去远端泵站同步，后面的水还在继续流，所以“送桶”和“继续放水”同时发生。正式表述就是“梯度分桶后异步 AllReduce 与后续反向计算重叠”。

`bucket_cap_mb` 决定桶大小，默认是 25 MiB。`gradient_as_bucket_view=True` 会让梯度直接指向桶缓冲区，也就是“梯度张量不再单独拷贝一份，而是直接复用通信缓冲区”，这样可以减少峰值显存和一次拷贝开销。

| `bucket_cap_mb` | 通信次数 | 首次通信启动时机 | 重叠潜力 | 典型风险 |
|---|---:|---|---|---|
| 过小，如 8 | 多 | 很早 | 高，但碎片化 | 启动开销过多 |
| 中等，如 25 | 中 | 较早 | 通常均衡 | 不一定适合大模型 |
| 较大，如 64 | 少 | 较晚 | 取决于层分布 | 第一桶启动太晚 |
| 很大，如 128+ | 很少 | 很晚 | 可能下降 | 退化成“快算完才通信” |

---

## 问题定义与边界

问题本质是：单卡反向传播只关心算子执行顺序，多卡训练还要支付通信延迟。延迟指“消息发出去前后的固定成本”，不是带宽本身。若每一层梯度都单独 AllReduce，小张量通信会很多，延迟成本会被反复支付；若等全部梯度都算完再同步，又完全失去计算和通信重叠。

DDP 的边界很明确：

1. 它解决的是数据并行下的梯度同步调度，不解决参数分片。
2. 它依赖反向传播“后层先出梯度、前层后出梯度”的顺序。
3. 它是否有效，取决于模型层级分布、网络带宽、网络延迟、bucket 大小三者是否匹配。
4. 默认 25 MiB 是通用经验值，不是最优值。

先看一个玩具例子。假设模型有 6 层，每层梯度 5 MiB，按反向顺序依次 ready。若 `bucket_cap_mb=25`，则前 5 层梯度正好填满第一个桶。第五个梯度 ready 时，第一个桶立刻发起 AllReduce；第 6 层对应的是更前面的层，它的反向计算如果还没结束，就能和这次 AllReduce 重叠。

如果每层不是 5 MiB，而是 50 MiB，那么单层本身就超过 25 MiB。此时每个大层几乎都会单独占一个桶，通信会更早启动，但桶数暴增，启动开销和调度成本也会上升。也就是说，“更早启动”不等于“更高吞吐”。

可以把触发逻辑拆成三个要素：

| 要素 | 条件 | 影响 |
|---|---|---|
| 桶满条件 | 桶内梯度累计达到阈值 | 决定何时具备启动资格 |
| 通信等待 | 前一个桶已经按固定顺序启动 | 保证各 rank 通信顺序一致 |
| 计算重叠 | 后续还有未完成的反向计算 | 决定能否真正隐藏通信 |

因此，DDP 不是简单地“桶满就发”，而是“桶满且顺序允许时发”。顺序约束很重要，因为 AllReduce 是集体通信，各个 rank 必须按相同次序进入，否则可能挂起。

---

## 核心机制与推导

DDP 会按参数顺序把梯度映射进多个 bucket。bucket 可以理解为“一块连续通信缓冲区”。反向传播过程中，每个参数的 `AccumulateGrad` 钩子会在该参数梯度 ready 时通知 DDP。某个桶里所有参数的梯度都 ready 后，这个桶被标记为可通信。

设第 $k$ 个桶包含参数集合 $\mathcal{P}_k$，总梯度大小满足：

$$
\sum_{i \in \mathcal{P}_k} g_i \ge B
$$

那么它最早启动通信的时间不是单独看“桶满时刻”，而是：

$$
t_k = \max \left(t_{k-1}, \operatorname{grad\_ready}(\mathcal{P}_k)\right)
$$

这里 $\operatorname{grad\_ready}(\mathcal{P}_k)$ 表示该桶最后一个梯度 ready 的时刻。`max` 的含义很直接：前一个桶没发，当前桶不能抢跑；当前桶自己还没满，也不能发。

再看一个数值推导。假设每层梯度 $g_i = 6$ MiB，桶大小 $B=25$ MiB，每层反向耗时 2 ms，单个 30 MiB 左右桶的 AllReduce 需要 5 ms。因为 $4 \times 6 = 24$ MiB 还没满，必须等到第 5 个梯度 ready，桶总量变成 30 MiB 才能触发。所以第一个桶在第 5 层梯度 ready 时启动，大约是反向开始后 10 ms；如果后面还有第 6、7 层的反向计算各 2 ms，那么前 4 ms 的通信被计算覆盖，只剩 1 ms 暴露给总步长。

可以把时序写成文本图：

| 时间段 | 发生的事 |
|---|---|
| 0-10 ms | 第 1 到第 5 个梯度依次 ready，桶逐渐填满 |
| 10 ms | 第一个桶满，启动异步 AllReduce |
| 10-14 ms | 第 6、7 层继续反向，通信并行进行 |
| 14-15 ms | 通信尾巴暴露，等待结束 |

如果梯度大小不均衡，触发顺序会更敏感。比如某一层梯度特别大，它可能单独形成一个桶，导致通信提前启动；但如果很多小层堆在一起，第一桶会被拖到很晚才满。大模型调参时，真正决定表现的往往不是“平均层大小”，而是“尾部几层梯度分布”。

`gradient_as_bucket_view` 的机制也要说清。view 指“共享底层存储的视图，不复制数据”。默认情况下，梯度可能先写到参数的 `.grad`，再拷贝进通信桶；开启这个选项后，`.grad` 直接指向桶中的对应切片。好处是少一次梯度拷贝，并减少一份梯度级别的峰值显存。代价是 `.grad.detach_()` 这类会修改张量元信息的操作不再允许。

---

## 代码实现

最小可用写法如下，参数必须在 DDP 构造阶段传入，因为 bucket 的建立和梯度到 bucket 的映射是在构造期完成的：

```python
import torch

model = torch.nn.parallel.DistributedDataParallel(
    model,
    bucket_cap_mb=64,
    gradient_as_bucket_view=True,
)
```

`64 MiB` 常见于高带宽机器，因为它能减少过多小桶；`gradient_as_bucket_view=True` 常用于大模型以降低峰值显存，但要确保训练代码没有对 `.grad` 做 `detach_()`。

下面这个 Python 例子不依赖 GPU，可以直接运行，用来模拟“桶满即启动”的调度逻辑：

```python
def schedule_buckets(grad_sizes_mb, bucket_cap_mb):
    buckets = []
    current = []
    current_sum = 0

    for idx, g in enumerate(grad_sizes_mb):
        current.append((idx, g))
        current_sum += g
        if current_sum >= bucket_cap_mb:
            buckets.append(current)
            current = []
            current_sum = 0

    if current:
        buckets.append(current)
    return buckets

# 玩具例子：5 个 5 MiB 梯度，25 MiB 恰好装满一个桶
b1 = schedule_buckets([5, 5, 5, 5, 5], 25)
assert len(b1) == 1
assert sum(g for _, g in b1[0]) == 25

# 真实些的例子：6 MiB 梯度，25 MiB 阈值，需要第 5 个梯度才触发
b2 = schedule_buckets([6, 6, 6, 6, 6, 6, 6], 25)
assert len(b2) == 2
assert sum(g for _, g in b2[0]) == 30
assert sum(g for _, g in b2[1]) == 12
```

参数选择可以先按下面的表格起步：

| 字段 | 含义 | 建议值 |
|---|---|---|
| `bucket_cap_mb` | 单个通信桶上限 | 25 起步，大模型常试 32/64 |
| `gradient_as_bucket_view` | 梯度是否直接复用桶缓冲区 | 显存紧张时优先开 |
| `find_unused_parameters` | 是否检测未参与反向的参数 | 动态图或条件分支时再开 |

真实工程例子：8 卡 DGX A100 训练 4B Transformer 时，如果每层梯度切得很碎，默认 25 MiB 可能导致 bucket 数量偏多，NCCL 启动次数上升，step time 被通信延迟拉长。实践中常见做法是把 `bucket_cap_mb` 提到 64 MiB，同时开启 `gradient_as_bucket_view=True`，再用 profiler 检查是否“通信条带变少但仍能覆盖在 backward 下面”。

---

## 工程权衡与常见坑

第一类权衡是“延迟”和“重叠”的冲突。桶太小，AllReduce 启动很早，但次数太多；桶太大，次数少，但第一桶迟迟不发，最后容易堆到 backward 尾部集中等待。

第二类权衡是“显存”和“代码兼容性”的冲突。`gradient_as_bucket_view=True` 往往能省下一份梯度量级的峰值显存，但要求优化器和训练框架不要对 `.grad` 做原地 `detach_()`。

第三类问题是“第一步测不准”。官方文档明确指出，梯度在第一次迭代之后才会变成 bucket view，所以显存收益要在至少一次完整 iteration 后再测。

调优时可以按下面的表执行：

| 调优动作 | 预期影响 | 检验指标 |
|---|---|---|
| `bucket_cap_mb: 25 -> 64` | 通信次数减少 | NCCL kernel 数量、step time |
| 开启 `gradient_as_bucket_view` | 峰值显存下降 | `torch.cuda.max_memory_allocated()` |
| 关闭不必要的 `find_unused_parameters` | 减少图遍历和额外同步 | backward 时间、日志告警 |
| 观察层级梯度分布 | 判断第一桶是否启动过晚 | profiler 时序图 |

显存检查通常这样做：

```python
peak = torch.cuda.max_memory_allocated()
print(f"peak memory: {peak / 1024**2:.1f} MiB")
```

常见坑有四个：

1. 只看平均通信时间，不看是否与 backward 重叠。平均值可能没变，但尾部暴露减少后总 step time 仍会下降。
2. 训练脚本里对 `.grad` 调 `detach_()`，开启 view 后直接报错。
3. 模型存在未用参数，却没有设置 `find_unused_parameters=True`，结果某些 bucket 永远等不到 ready。
4. 首轮 profile 就下结论，忽略了 bucket 重建和 view 生效时机。

---

## 替代方案与适用边界

默认 DDP bucket 适合大多数规则网络，但它不是所有场景的最优解。若模型层大小极度不均衡、存在大量条件分支、或者某些模块梯度特别大，默认分桶可能让重叠效果波动很大。

可选方案如下：

| 方案 | 适用场景 | 优点 | 复杂度 |
|---|---|---|---|
| 默认 DDP bucket | 常规 CNN、Transformer | 简单、稳定 | 低 |
| 调整 `bucket_cap_mb` | 大模型、网络较快 | 最低成本调优 | 低 |
| 自定义通信 hook | 稀疏模型、层级不均衡明显 | 可按模块稳定调度 | 中高 |
| 更粗粒度并行策略，如 ZeRO/FSDP | 参数太大、显存成为主瓶颈 | 同时降显存和通信量 | 高 |

真实边界也要讲清楚：

1. 如果网络很慢，通信本身远大于计算，再好的重叠也只能“遮住一部分”，不能把慢网卡变快。
2. 如果某些层单层梯度就巨大，bucket 设计空间会变小，因为单层本身就可能占满或超过一个桶。
3. 如果模型动态图很强，`find_unused_parameters` 和更复杂的 hook 逻辑会增加维护成本。

自定义 hook 的伪代码大致如下，意思是“当某个 bucket ready 时，用你自己的策略处理通信”：

```python
def my_comm_hook(state, bucket):
    tensor = bucket.buffer()
    fut = torch.distributed.all_reduce(tensor, async_op=True).get_future()
    return fut.then(lambda fut: fut.value()[0] / state["world_size"])

ddp_model.register_comm_hook({"world_size": 8}, my_comm_hook)
```

这类方案更适合已经能稳定读 profiler、能解释 bucket 时序、并且愿意维护通信逻辑的团队。对零基础到初级工程师，优先级仍然是：先理解默认 bucket，再调 `bucket_cap_mb`，最后才考虑 hook。

---

## 参考资料

- [PyTorch DistributedDataParallel 文档（PyTorch 2.10，2026）](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)：给出 `bucket_cap_mb` 默认 25 MiB、`gradient_as_bucket_view` 的显存收益与 `detach_()` 限制。
- [Notes on PyTorch's Distributed Data Parallel (DDP)（2025）](https://skrohit.github.io/posts/Notes_on_DDP/)：解释梯度 bucket、反向钩子、按桶触发 AllReduce 以及为何要保持跨 rank 的一致顺序。
- [PyTorch Issue #118421: Increase default bucket_cap_mb value（2024）](https://github.com/pytorch/pytorch/issues/118421)：展示社区对默认 25 MiB 是否过小的讨论，适合理解“默认值不等于最优值”。

| 资料 | 主要信息 | 适合用途 |
|---|---|---|
| PyTorch 官方文档 | 参数定义、行为边界、已知限制 | 写训练代码前查准语义 |
| 社区技术笔记 | bucket 触发机制、重叠直觉 | 建立机制理解 |
| GitHub Issue | 默认值争议、经验背景 | 调参时判断是否该改默认值 |
