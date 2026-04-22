## 核心结论

分布式训练的容错目标不是“避免故障”，而是“故障发生后尽量少丢训练进度”。大规模训练运行时间长、机器数量多，GPU、节点、网络、共享存储都可能出问题。checkpoint 是最核心的容错手段：定期把训练状态保存下来，故障后从最近一次保存点继续。

真正可用的 checkpoint 不是只保存模型参数。完整训练状态可以写成：

$$
S=\{W,O,R,M\}
$$

其中，$W$ 是模型参数，$O$ 是优化器状态，$R$ 是随机数状态，$M$ 是训练元信息。随机数状态指 dropout、数据打乱、采样等依赖的随机生成器位置；元信息指当前 step、epoch、并行配置、版本号等恢复训练所需的上下文。

| 状态组成 | 含义 | 是否必须保存 | 是否影响恢复一致性 |
|---|---|---:|---:|
| $W$ | 模型参数 | 是 | 是 |
| $O$ | 优化器状态，例如 Adam 的动量和方差 | 是 | 是 |
| $R$ | 随机数状态 | 是 | 是 |
| $M$ | step、并行度、数据位置、格式版本 | 是 | 是 |

玩具例子：4 个 rank 训练时，如果每 1000 step 保存一次 checkpoint，节点在第 2578 step 故障，最近可恢复点是第 2000 step，最多丢失最近 999 step 的训练进度。若改成每 200 step 保存一次，最近可恢复点是第 2400 step，最坏损失从 999 step 降到 199 step，但写盘次数会增加 5 倍。

真实工程例子：训练 100B 参数以上模型时，一次任务可能跑数天到数周。集群中出现 GPU ECC 错误、节点重启、交换机抖动并不罕见。工程上通常把 checkpoint 分片保存到共享文件系统，保存尽量异步执行，恢复时允许用新的 world size 继续训练。

---

## 问题定义与边界

分布式训练，是把一次模型训练拆到多个进程、多个 GPU、多个节点上协同执行。rank 是分布式训练里的进程编号；world size 是参与训练的总 rank 数。4 张 GPU 各跑一个进程时，通常就是 world size = 4，rank 从 0 到 3。

分布式训练中的故障来源包括 GPU ECC 错误、节点重启、进程退出、网络抖动、共享存储异常。任何一种都可能打断整次训练。容错不是保证训练永远不中断，而是保证中断后可以恢复，并从最近一次有效 checkpoint 继续训练。

“恢复”和“继续训练”不是一回事。恢复是把历史保存的训练状态重新加载到当前进程组；继续训练是在恢复后的状态上再次执行 forward、backward、optimizer step。恢复成功只说明状态加载正确，继续训练还要求数据顺序、随机数、优化器状态和并行配置能够协同工作。

| 故障类型 | 影响范围 | 是否可恢复 | 典型处理方式 |
|---|---|---:|---|
| GPU ECC 错误 | 单卡或单节点进程退出 | 通常可恢复 | 重新调度作业，从最近 checkpoint 加载 |
| 节点重启 | 节点上所有 rank 中断 | 可恢复 | 换节点重启，读取共享存储 checkpoint |
| 进程退出 | 某个 rank 异常结束 | 可恢复 | 整组训练停止后重新拉起 |
| 网络抖动 | collective 通信卡死或超时 | 视情况可恢复 | 终止当前作业，重新初始化进程组 |
| 共享存储异常 | 保存或加载失败 | 不一定 | 重试、保留上一个完整 checkpoint |

新手版本可以这样理解：单机训练像“写作时自动保存文档”，分布式训练像“多人同时协作编辑同一份大文档”。某个协作者掉线后，系统需要能从最近一次自动保存继续，而不是从头再写。但这个比喻只用于直观理解，技术上 checkpoint 保存的是参数张量、优化器张量、随机数状态和元信息，不是简单的文档文本。

容错边界还包括性能边界。异步 checkpoint 能减少训练主线程等待写盘的时间，但不能让保存完全没有成本。staging 会占用 CPU 内存，后台写盘会占用 I/O 带宽，共享存储也可能成为瓶颈。

---

## 核心机制与推导

分布式 checkpoint 的基本机制是“按 rank 分片保存，恢复时按目标并行度重分片”。分片，是把一个完整训练状态拆成多个部分，每个 rank 只负责自己那一份。重分片，是恢复时根据新的 world size 重新切分这些状态。

设训练状态为：

$$
S=\{W,O,R,M\}
$$

保存时按当前 world size $R$ 切成：

$$
\{S_i\}_{i=0}^{R-1}
$$

恢复时如果目标 world size 变成 $R'$，需要做：

$$
\{S_i\}_{i=0}^{R-1}\rightarrow\{S'_j\}_{j=0}^{R'-1}
$$

这就是 load-time resharding：加载时重分片。它的价值是恢复不强依赖原来的 rank 数和拓扑。更准确地说，“支持从任意 rank 恢复”不是任意单个 rank 独自恢复全部训练，而是新的 rank 组可以一起参与加载，并按新的并行配置恢复状态。

玩具例子：4 个 rank 训练，参数总量 8 GB，优化器状态 16 GB，总状态 24 GB。平均分片后，每个 rank 保存约 6 GB。若恢复时改成 8 个 rank，就不能让每个 rank 继续读取原来的 6 GB 分片直接训练，而要把原来的 4 份分片重新切成 8 份，每个 rank 约持有 3 GB 状态。

流程可以写成：

```text
训练主循环
  |
  |-- 到达保存 step
  v
收集训练状态 S={W,O,R,M}
  |
  v
按 rank 打包分片 S_i
  |
  v
staging 到 CPU / 后台缓冲区
  |
  v
后台写入共享存储
  |
  v
故障后重启作业
  |
  v
新 rank 组加载 checkpoint
  |
  v
按目标 world size 重分片
  |
  v
继续训练
```

异步 checkpoint 的本质是把 `staging -> write` 挪到后台执行。训练线程在提交保存请求后继续跑下一步，后台线程或进程负责把数据写入共享存储。这里的关键约束是并发控制：前一次保存未完成时，不能覆盖同一份状态，也不能无限堆积保存请求。

| 方式 | 主训练是否等待写盘 | 实现复杂度 | 资源开销 | 适用场景 |
|---|---:|---:|---:|---|
| 同步保存 | 是 | 低 | I/O 峰值明显 | 小模型、短任务 |
| 异步保存 | 尽量否 | 中到高 | CPU 内存和后台 I/O | 大模型、长任务 |
| 分布式异步保存 | 尽量否 | 高 | 需要队列、分片、共享存储 | 多节点大规模训练 |

真实工程例子：Megatron-LM、Megatron Core 和 DeepSpeed 都会围绕分布式状态保存做工程封装。它们关注的问题不是“怎么调用一次 `torch.save`”，而是多 rank 如何同时写、ZeRO 或张量并行下状态如何组织、恢复时如何适配新的并行配置，以及如何避免 checkpoint 文件膨胀。

---

## 代码实现

代码层面要把四件事分开：训练状态收集、分片打包、后台写盘、恢复加载。混在一个函数里会让错误处理、并发控制和跨 world size 恢复都变得很难维护。

下面是一个可运行的简化实现。它不依赖真实 GPU，只演示 checkpoint 状态组成、分片保存、异步队列控制和恢复时重分片。

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import pickle
import random
import tempfile


def collect_train_state(weights, optimizer_state, step, seed):
    rng = random.Random(seed)
    return {
        "W": list(weights),
        "O": list(optimizer_state),
        "R": {"seed": seed, "sample": rng.random()},
        "M": {"step": step, "format_version": 1},
    }


def shard_list(values, world_size):
    return [values[i::world_size] for i in range(world_size)]


def pack_shards(state, world_size):
    weight_shards = shard_list(state["W"], world_size)
    optim_shards = shard_list(state["O"], world_size)
    return [
        {
            "rank": rank,
            "world_size": world_size,
            "W": weight_shards[rank],
            "O": optim_shards[rank],
            "R": state["R"],
            "M": state["M"],
        }
        for rank in range(world_size)
    ]


def write_shards(shards, storage_path):
    path = Path(storage_path)
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "world_size": len(shards),
        "step": shards[0]["M"]["step"],
        "format_version": shards[0]["M"]["format_version"],
    }
    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    for shard in shards:
        with (path / f"rank_{shard['rank']}.pkl").open("wb") as f:
            pickle.dump(shard, f)

    return str(path)


class AsyncCheckpointWriter:
    def __init__(self, max_pending=1):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.max_pending = max_pending

    def async_save(self, shards, storage_path):
        if self.pending is not None and not self.pending.done():
            raise RuntimeError("previous checkpoint is still writing")
        self.pending = self.executor.submit(write_shards, shards, storage_path)
        return self.pending

    def close(self):
        if self.pending is not None:
            self.pending.result()
        self.executor.shutdown(wait=True)


def load_and_reshard(storage_path, target_world_size):
    path = Path(storage_path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))

    old_shards = []
    for rank in range(meta["world_size"]):
        with (path / f"rank_{rank}.pkl").open("rb") as f:
            old_shards.append(pickle.load(f))

    full_w = []
    full_o = []
    for shard in old_shards:
        full_w.extend(shard["W"])
        full_o.extend(shard["O"])

    restored = {
        "W": shard_list(full_w, target_world_size),
        "O": shard_list(full_o, target_world_size),
        "R": old_shards[0]["R"],
        "M": old_shards[0]["M"] | {"restored_world_size": target_world_size},
    }
    return restored


with tempfile.TemporaryDirectory() as d:
    state = collect_train_state(
        weights=list(range(8)),
        optimizer_state=list(range(100, 108)),
        step=2578,
        seed=1234,
    )

    writer = AsyncCheckpointWriter()
    shards = pack_shards(state, world_size=4)
    handle = writer.async_save(shards, Path(d) / "ckpt_2578")

    # 训练主线程此时可以继续执行下一步；这里只等待后台写盘完成，便于测试。
    assert handle.result().endswith("ckpt_2578")

    restored = load_and_reshard(Path(d) / "ckpt_2578", target_world_size=8)

    assert len(restored["W"]) == 8
    assert all(len(s) == 1 for s in restored["W"])
    assert restored["M"]["step"] == 2578
    assert restored["M"]["restored_world_size"] == 8
    assert restored["R"]["seed"] == 1234

    writer.close()
```

新手版本伪代码可以更短：

```python
state = collect_train_state(model, optimizer, rng, meta)
handle = async_save(state, storage_path)
# 训练继续跑
restored = load_and_reshard(storage_path, target_world_size=8)
```

这段伪代码表达两个关键点：保存不阻塞训练主线程；恢复时显式接收目标 world size，并在加载阶段完成 reshaping 或 resharding。

| 模块 | 输入 | 输出 | 失败点 |
|---|---|---|---|
| 状态收集 | model、optimizer、rng、meta | 完整训练状态 | 漏存随机数或 step |
| 分片打包 | 训练状态、world size | 每个 rank 的分片 | 分片顺序不稳定 |
| 后台写盘 | 分片、存储路径 | checkpoint 文件 | 队列堆积、写盘失败 |
| 恢复加载 | checkpoint、target world size | 新 rank 分片 | 不支持重分片、元信息不兼容 |

真实框架里，PyTorch Distributed Checkpoint 提供分布式保存和加载入口，Megatron Core 提供面向并行训练的分布式 checkpoint 包，DeepSpeed 在 ZeRO 场景下要求所有进程参与保存。工程实现应优先使用这些框架能力，而不是自己手写完整格式。

---

## 工程权衡与常见坑

checkpoint 间隔是最直接的权衡。间隔太长，故障后回滚很远；间隔太短，I/O、CPU staging 内存、对象存储请求数和训练抖动都会增加。

若 checkpoint 间隔为 $K$ step，单次故障的最坏进度损失是：

$$
L_{max}=K-1
$$

所以从 1000 step 改到 200 step：

$$
1000-1=999,\quad 200-1=199
$$

最坏损失从 999 step 降到 199 step，但保存频率提高为原来的 5 倍。大模型训练里，这不是免费优化。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只保存模型权重 | 恢复后训练曲线漂移 | 同时保存参数、优化器、随机数、元信息 |
| 只让 rank 0 保存 | 状态不完整或 rank 0 成为瓶颈 | 所有 rank 参与分布式保存 |
| checkpoint 间隔过长 | 故障后丢失大量 step | 按故障率、I/O 能力和训练成本设定间隔 |
| 异步队列无限堆积 | CPU 内存被 staging buffer 吃满 | 限制 pending 请求数，通常只允许 1 个未完成保存 |
| 存到本地盘 | 换节点后读不到 | 写入共享存储或对象存储 |
| 共享底层 storage 被原样序列化 | checkpoint bloat，文件膨胀 | clone 或 deduplication，避免保存无关底层块 |
| 随机数未对齐 | dropout、采样、数据顺序不一致 | 保存并恢复 RNG 状态 |
| 格式不含并行元信息 | 无法跨配置恢复 | 保存 world size、并行策略、版本号 |

直观版本：“只保存模型权重”就像只保存草稿正文，不保存光标位置、撤销栈和编辑器状态。恢复后内容还在，但后续编辑结果可能不一样。对应到训练里，权重相同不代表训练状态相同；Adam 的动量、学习率调度器位置、数据加载位置和随机数状态都会影响后续更新。

还要注意共享存储。本地 NVMe 写入快，但节点失效后可能无法读取；共享文件系统方便恢复，但容易被大量 rank 同时写爆；对象存储扩展性强，但延迟和一致性语义需要额外处理。工程上通常要做限流、保留最近 N 个 checkpoint、写入临时目录后原子切换完成标记，避免恢复读到半成品。

---

## 替代方案与适用边界

不是所有训练都需要复杂的分布式异步 checkpoint。方案选择取决于模型规模、训练时长、并行方式、故障概率和恢复要求。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 单文件 checkpoint | 简单，易调试 | 大模型下写入慢，rank 0 瓶颈明显 | 单机小模型 |
| 分片 checkpoint | 并行写入，适合多 rank | 格式和恢复逻辑更复杂 | 多 GPU、多节点训练 |
| 异步 checkpoint | 减少训练主路径等待 | 占用 CPU 内存和后台 I/O | 长时间大模型训练 |
| 仅保存权重 | 文件小，推理加载方便 | 不适合严格续训 | 发布模型、评估、推理 |

适用边界可以再细分：

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| 小模型、单机训练 | 单文件 checkpoint | 简单可靠，工程成本低 |
| 固定并行度训练 | 分片 checkpoint | 不一定需要复杂重分片 |
| 强一致性要求 | 完整训练状态 checkpoint | 必须保存优化器和随机数 |
| 频繁抢占场景 | 高频异步 checkpoint | 降低抢占造成的 step 损失 |
| 频繁改变 world size | 支持 load-time resharding 的格式 | 恢复时需要跨并行度重分片 |
| 只做推理发布 | 仅保存权重 | 不需要优化器状态 |

新手版本：如果你只是在单机上训练一个小模型，直接存一个完整文件就够了；但如果是 100B 参数以上的大模型训练，必须考虑分片、共享存储、异步写入和跨配置恢复。

某些简化格式恢复更快，但不一定支持跨并行度重分片。例如固定数据并行度下的快照格式可以针对当前拓扑优化读取速度，但当恢复时 world size 改变，就可能无法直接加载。反过来，通用分布式 checkpoint 格式元信息更多、实现更复杂，但更适合长期训练和弹性调度。

工程上不要把 checkpoint 当成训练脚本最后的附属功能。它决定故障后能不能恢复，恢复后训练是否一致，以及集群资源是否会被写盘压力拖垮。对大规模训练来说，checkpoint 是训练系统的一部分，不是简单的文件保存。

---

## 参考资料

| 来源 | 能支持的结论 | 适合放在正文哪一段 |
|---|---|---|
| PyTorch DCP | 多 rank 并行保存、加载和 load-time resharding | 核心机制与代码实现 |
| PyTorch Async DCP | `async_save`、后台 staging、并发与内存约束 | 核心机制与工程权衡 |
| Megatron Core | 分布式 checkpoint 与跨并行配置恢复 | 真实工程例子 |
| DeepSpeed 文档 | 所有进程参与保存、ZeRO checkpoint | 代码实现与常见坑 |
| DeepSpeed 源码说明 | clone tensor 避免 checkpoint bloat | 工程权衡 |

1. [Getting Started with Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
2. [Asynchronous Saving with Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
3. [Megatron Core dist_checkpointing package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html)
4. [DeepSpeed Model Checkpointing](https://deepspeed.readthedocs.io/en/stable/model-checkpointing.html)
5. [DeepSpeed checkpoint utils source](https://deepspeed.readthedocs.io/en/stable/_modules/deepspeed/checkpoint/utils.html)
