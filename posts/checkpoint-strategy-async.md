## 核心结论

检查点策略的本质，不是“多久存一次”这么简单，而是在**保存频率、存储层级、阻塞时间、恢复时间**之间做总成本最小化。检查点就是训练过程中某一时刻的完整状态快照，通常包含模型参数、优化器状态、随机数状态、数据迭代位置等；它的作用是机器或进程故障后，能从最近状态继续，而不是从头重算。

对大模型训练，最优实践通常不是直接把完整状态同步写到远端存储，而是走三段流水线：

| 步骤 | 作用 | 说明 |
| --- | --- | --- |
| CPU 缓冲 | 解决短阻塞 | 先把 GPU 上的训练状态拷到 CPU staging buffer，尽快释放训练线程 |
| 后台刷盘 | 保障持久性 | 后台线程或独立进程把 CPU 缓冲异步写入 NVMe、并行文件系统或对象存储 |
| 远端副本 | 保障恢复 | 形成节点损坏后仍可读取的持久副本，避免“机器没了，checkpoint 也没了” |

这套设计的核心价值是：**训练线程只承担短暂的 GPU→CPU 复制开销，耗时更长的 CPU→远端写入被放到后台**。这样即使 checkpoint 很大，GPU 也不会长时间空转。

再往上一层看，ZeRO 分片 checkpoint 与自动故障恢复是必须配套的。ZeRO 分片是把模型参数、梯度、优化器状态拆散存到不同 rank 上，目的是降低单卡内存占用；代价是恢复时不能把一个文件直接塞回去，而要按分片规则重建。健康检查与自动重启脚本则负责在实例故障后自动找到最近一个“完整可用”的 checkpoint，并拉起任务。

结论可以压缩成一句话：**高频 checkpoint 只有在异步化、分层存储和自动恢复同时成立时，才真正有工程价值。**

---

## 问题定义与边界

训练里真正关心的指标，不是“checkpoint 写了多少秒”，而是一次故障平均浪费多少训练时间。一个常见近似公式是：

$$
T_{\text{wasted}} = t_{\text{ckpt}} + \frac{1}{2f} + t_{\text{rtvl}}
$$

其中：

- $t_{\text{ckpt}}$：保存 checkpoint 带来的有效阻塞时间
- $f$：checkpoint 频率，单位是每秒保存多少次
- $t_{\text{rtvl}}$：恢复并重新加载最近可用 checkpoint 的时间

这里的 $\frac{1}{2f}$ 表示“故障平均发生在两次 checkpoint 的中点”。白话说，如果你每 10 分钟保存一次，机器坏掉时平均要重算 5 分钟。

这个公式说明了两个事实。

第一，**保存更频繁，平均重算时间会下降**。第二，**如果保存本身太慢，频率再高也没意义**。因为 $1/f$ 不能无限缩小，至少不能小到比一次 checkpoint 的实际完成时间还短，否则新的保存任务会和上一次互相堆积，最终把训练拖死。

看一个玩具例子。设：

- $t_{\text{ckpt}} = 90$ 秒
- $t_{\text{rtvl}} = 120$ 秒
- 每 10 分钟保存一次，$f = 1/600$

则：

$$
T_{\text{wasted}} = 90 + 300 + 120 = 510 \text{ 秒}
$$

如果改成每 1 分钟保存一次，$f = 1/60$，则：

$$
T_{\text{wasted}} = 90 + 30 + 120 = 240 \text{ 秒}
$$

平均浪费时间减半以上。但这个结论有边界：如果你的 checkpoint 写入是同步直写远端，且每次就要 80 到 100 秒，那“每分钟一次”根本不可行。也就是说，**高频 checkpoint 的前提，不是意愿，而是异步能力和足够的本地缓冲层**。

这篇文章讨论的边界主要是分布式训练，尤其是多卡、多机场景。单机小模型也有 checkpoint，但问题简单得多：文件小、恢复快、节点数量少，故障率不高，不一定需要 ZeRO 分片和异步多层存储。

---

## 核心机制与推导

异步 checkpoint 可以拆成一条逻辑流水线：

$$
\text{GPU state} \rightarrow \text{CPU staging} \rightarrow \text{local / remote storage} \rightarrow \text{recovery}
$$

每一段解决的是不同问题。

第一段是 GPU 到 CPU staging。staging 可以理解为“临时周转区”，也就是先把数据放到 CPU 内存中的中间缓冲。这个阶段通常仍然会短暂阻塞训练，因为你得把当前状态从显存取出来，但比“直接写对象存储直到写完”为止要短得多。

第二段是 CPU 到存储。这里才是耗时大户，因为可能涉及 NVMe、网络、分布式文件系统、对象存储。异步 checkpoint 的关键，就是让这一步在后台线程、后台进程或独立通信组里完成，不占住训练主路径。

第三段是恢复。恢复时系统要回答三个问题：

| 问题 | 作用 | 说明 |
| --- | --- | --- |
| 哪个 checkpoint 可用 | 避免读到半成品 | 只能加载已经完整落盘、元数据一致的版本 |
| 数据在哪里 | 选择最快恢复路径 | 优先本地内存或本地盘，其次远端 |
| 如何重建训练状态 | 保证继续训练正确 | 不只恢复参数，还要恢复优化器、随机状态和进度 |

ZeRO-3 让这件事再复杂一层。ZeRO 是一种“把训练状态拆开分给不同设备保存”的方法，白话说，不再要求每张卡都存一份完整模型，而是大家各存自己负责的片段。好处是能训练更大的模型，坏处是 checkpoint 不再是一个独立完整文件，而是一组分片文件。保存时每个 rank 写自己的 shard，恢复时要按原规则组装，或者合并成 fp32 全量权重后再加载到普通 PyTorch 模型。

真实工程例子是 Gemini。它在大规模分布式训练里把 checkpoint 放在 CPU 内存层，并结合故障检测与节点替换机制，在出现大量模拟故障和实例失效时，仍能在秒级定位并恢复最近状态。这个实践说明，**当故障足够频繁时，“恢复路径是否自动化”与“checkpoint 是否高频”同样重要**。你可以把 checkpoint 保存得很勤，但如果故障后要人工查版本、手工拼 shard、再人工改启动参数，那实际恢复时间仍然很长。

所以从机制上看，完整方案不是一个“保存函数”，而是四件事的组合：

- 低阻塞状态导出
- 后台持久化写入
- 分片状态的可重建性
- 故障后的自动发现与自动重启

少任何一项，都只能算“能存”，不算“能稳”。

---

## 代码实现

先用一个可运行的 Python 玩具程序，把“保存频率 vs 平均浪费时间”算清楚：

```python
def wasted_time(t_ckpt: float, interval_sec: float, t_retrieve: float) -> float:
    assert t_ckpt >= 0
    assert interval_sec > 0
    assert t_retrieve >= 0
    return t_ckpt + interval_sec / 2 + t_retrieve

ten_min = wasted_time(t_ckpt=90, interval_sec=600, t_retrieve=120)
one_min = wasted_time(t_ckpt=90, interval_sec=60, t_retrieve=120)

assert ten_min == 510
assert one_min == 240
assert one_min < ten_min

def storage_growth(checkpoint_tb: float, per_day: int, days: int) -> float:
    assert checkpoint_tb > 0
    assert per_day > 0
    assert days > 0
    return checkpoint_tb * per_day * days

weekly_tb = storage_growth(checkpoint_tb=18, per_day=48, days=7)
assert weekly_tb == 6048  # 约 6 PB
print("ok")
```

这个例子只算两件事：平均浪费时间和存储膨胀。它足够说明为什么工程上一定要把“频率”和“容量”一起设计。

如果落到 PyTorch Distributed Checkpoint，思路通常是：

```python
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

# 独立 process group 用来隔离 checkpoint 通信
pg = dist.new_group(backend="gloo")

state = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}

future = dcp.async_save(
    state_dict=state,
    checkpoint_id=save_dir,
    process_group=pg,
)

train_step()

# 只在你需要确认持久化成功时才等待
future.result()
```

这里有两个关键点。

第一，`async_save` 不是“完全零成本”。它通常仍然要先完成一段同步导出，例如把 GPU 上的状态复制到 CPU 可管理的缓冲区。真正被异步化的，是后续写盘和远端持久化。

第二，checkpoint 不能只存 `model.state_dict()`。对训练恢复来说，至少还要存：

- `optimizer.state_dict()`
- 当前全局步数 `global_step`
- 学习率调度器状态
- 数据加载进度或样本偏移
- 随机数种子状态

否则你恢复出来的，只是“长得像原模型”的参数，不是严格继续同一条训练轨迹。

如果用 DeepSpeed ZeRO-3，保存和加载还要注意“引擎状态”和“原始模型状态”不是一回事。常见做法是让 DeepSpeed 保存分片 checkpoint，恢复时重新初始化 engine，再调用它的 `load_checkpoint`；若需要导出普通 PyTorch 可读的完整权重，则用 `zero_to_fp32` 或 `load_state_dict_from_zero_checkpoint` 做合并。

一个真实工程里的恢复流程通常是：

1. 健康检查 agent 发现节点故障。
2. 调度系统替换坏节点或重启作业。
3. 恢复脚本扫描最新 manifest，manifest 就是 checkpoint 元数据清单，用来记录哪些 shard 已经写完整。
4. 找到最近一个完整 checkpoint。
5. 重建 ZeRO engine，加载模型、优化器、步数和数据进度。
6. 从对应 step 继续训练。

这套流程要自动化，不能指望人工介入。

---

## 工程权衡与常见坑

检查点系统最常见的误区，是只盯“恢复快不快”，不盯“平时贵不贵”。

第一个坑是 checkpoint 爆炸。假设一次 checkpoint 18 TB，每半小时一次，一周就是 $18 \times 48 \times 7 = 6048$ TB，也就是约 6 PB。这个量级下，存储费用、对象列表操作、元数据管理都会出问题。常见对策是“最近 K 个 + 最佳一个 + 归档老版本”，而不是无限保留。

第二个坑是异步写入反过来抢训练资源。异步不等于免费。后台 flush 会消耗 CPU 内存、PCIe 带宽、网卡带宽，严重时会干扰梯度同步。规避方法通常有三类：

| 坑 | 影响 | 规避 |
| --- | --- | --- |
| 检查点爆炸 | 存储和账单失控 | 只保留最近 K 个，旧版本转 Nearline/Archive |
| 异步竞争 | 吞吐下降，step time 抖动 | 限制 staging buffer 大小，隔离 process group |
| 半成品 checkpoint | 恢复失败或读脏数据 | 用 manifest/commit 标记，仅发布完整版本 |
| 只存模型不存优化器 | 恢复后曲线漂移 | 同时保存优化器、调度器、RNG 和数据进度 |
| 健康检查不足 | 故障后长期停机 | 加 agent、自动重启脚本和自检逻辑 |

第三个坑是“最新文件不一定是最新可恢复版本”。异步系统里，某个目录时间戳最新，不代表它已经完整写完。正确做法是写完所有 shard 后，再写一个完成标记或 manifest。恢复程序只认 manifest，不认目录名字。

第四个坑是 ZeRO world size 变化。world size 就是并行进程总数。如果训练时用 64 卡，恢复时只剩 32 卡，很多分片 checkpoint 不能直接原样读回。这时要么依赖框架的弹性恢复能力，要么先把 shard 合并成通用格式，再在新并行度下重新切分。

最后一个坑是只考虑训练进程，不考虑编排系统。真实故障往往不是 Python 进程抛异常，而是 GPU 掉卡、节点失联、文件系统抖动、作业被调度器驱逐。没有健康检查、自动重试、失败原因分类和冷启动脚本，checkpoint 文件再完整，恢复链路也会断。

---

## 替代方案与适用边界

不是所有训练都需要“异步 + ZeRO + 自动恢复”的完整方案。选择取决于模型大小、故障率、预算和恢复目标。

| 方案 | 适用边界 | 恢复代价 |
| --- | --- | --- |
| 远端同步写 | 小模型、低频保存、可接受分钟级暂停 | 读取全量，阻塞明显 |
| 本地磁盘同步写 | 单机或少量节点，节点稳定 | 节点损坏时可能丢失 checkpoint |
| CPU staging + 异步远端写 | 中大模型、高频 checkpoint | 恢复更快，训练阻塞更小 |
| ZeRO 分片 checkpoint | 超大模型、显存紧张 | 恢复需要分片重建或合并 |
| 通用格式导出 fp32 | 需要跨框架或跨并行度迁移 | 导出慢，但兼容性更好 |

对初级工程师，一个简单判断标准是：

- 单机实验、模型不大：先做同步 checkpoint，保证能恢复即可。
- 多机多卡、训练几天以上：至少做“最近 K 个 + 完整状态保存 + 自动加载最新成功版本”。
- 大模型、故障频率高、checkpoint 以 TB 计：必须考虑异步化、分层存储、ZeRO 分片和自动故障恢复。

也就是说，**复杂方案不是为了“高级”，而是因为规模上来后，简单方案已经不够用**。当 checkpoint 从 GB 变成 TB，当训练从 1 台机变成几十台机，问题性质就变了。

---

## 参考资料

- AWS Storage Blog, Architecting scalable checkpoint storage for large-scale ML training on AWS: https://aws.amazon.com/blogs/storage/architecting-scalable-checkpoint-storage-for-large-scale-ml-training-on-aws/
- PyTorch Blog, Reducing Model Checkpointing Times by Over 10x with PyTorch Distributed Asynchronous Checkpointing: https://pytorch.org/blog/reducing-checkpointing-times/
- PyTorch Tutorial, Distributed Asynchronous Checkpointing Recipe: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
- DeepSpeed Documentation, Model Checkpointing: https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
- Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints, SOSP 2023: https://zhuangwang93.github.io/docs/Gemini_SOSP23.pdf
- SageMaker HyperPod Auto Resume Documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-auto-resume.html
- Patent discussion of wasted-time style checkpoint formula: https://patents.justia.com/patent/20240428082
