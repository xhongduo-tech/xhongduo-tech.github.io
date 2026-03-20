## 核心结论

分布式训练里的 Checkpoint，不是“把一个大模型存成一个大文件”，而是“把不同并行维度上的局部状态，按 rank 分片保存，并额外写一份能描述这些分片如何拼回来的 metadata”。

最关键的结论有两条：

1. `torch_dist` 风格的分布式 Checkpoint，会让每个 TP/PP rank 各自写自己的 `ShardedStateDict`、优化器状态、RNG 状态，通常形成 `__<tp>_<pp>.distcp` 这类文件；这样保存时是并行写盘，单文件不会大到难以管理。
2. 真正让 Checkpoint 可恢复、可迁移、可变更并行度的，不只是参数文件本身，而是 `.metadata`、`metadata.json`、`latest_train_state.pt` 这类元数据文件。它们记录了并行拓扑、层到 stage 的映射、参数切分方式、优化器分片布局。加载时先读 metadata，才能知道每个 rank 应该拿哪一段、拼哪一段。

直观版可以这样理解：训练时，每个 TP/PP rank 只保存自己负责的参数块、优化器块和随机数状态，目录里会出现 `__1_0.distcp`、`__1_1.distcp` 之类的文件；恢复时，程序先读 metadata，再告诉各个 rank：“你该从哪些分片里取数据，取完以后沿哪个维度拼接或重新切分”。

| 并行维度 | 保存内容 | 常见文件/信息 |
| --- | --- | --- |
| TP，张量并行，意思是把一个张量切成多片分别算 | 参数分片、切分维度、offset | `__<tp>_<pp>.distcp` 内的参数 shard |
| PP，流水并行，意思是把模型层分给不同 stage | 本 stage 层参数、层名到 stage 的映射 | metadata 中的 layer mapping |
| DP，数据并行，意思是每份模型看不同数据 | RNG、训练步数、部分或全部优化器状态 | optimizer shard、train state |
| ZeRO，优化器分片，意思是优化器状态不再全量复制 | 动量、方差、master weight 的分片布局 | optimizer metadata |
| 全局元数据 | 并行拓扑、checkpoint 格式、恢复入口 | `.metadata`、`metadata.json`、`latest_train_state.pt` |

---

## 问题定义与边界

问题的本质是：当模型同时使用 TP、PP、DP，甚至叠加 ZeRO 时，如何把“每个 rank 手里只有一部分”的训练状态安全落盘，并在之后恢复成同一个训练状态，或者迁移到新的并行度继续训练。

这里的“训练状态”不只包含模型参数，还至少包括：

| 状态类型 | 是否必须保存 | 原因 |
| --- | --- | --- |
| 模型参数 | 是 | 不保存就无法恢复模型 |
| 优化器状态 | 通常是 | 不保存会丢失训练动量，继续训练会漂移 |
| RNG 状态，随机数生成器状态，意思是控制 dropout 和采样随机性的内部状态 | 通常是 | 影响可复现性 |
| 训练进度 | 是 | 需要知道 global step、epoch、lr scheduler 位置 |
| 并行拓扑 metadata | 是 | 不知道 shard 关系就无法拼装 |

对新手来说，可以先把三层并行拆开看：

1. TP：把一个大权重矩阵沿列或行切开，每个 rank 存一块。
2. PP：把模型按层切成多个 stage，每个 stage 只存自己负责的层。
3. DP：每组模型副本处理不同 batch。若启用 ZeRO，优化器状态也会被进一步切开，所以并不是每个 DP rank 都存完整 optimizer。

如果把一个参数张量记作 $W$，那么不同维度负责的不是同一件事：

- TP 负责“同一层里的张量切片”
- PP 负责“不同层归哪个 stage”
- DP/ZeRO 负责“优化器和训练状态是否分片”

可简写为：

$$
\text{Checkpoint} = \{\text{Param Shards by TP/PP}\} + \{\text{Optimizer Shards by DP/ZeRO}\} + \{\text{Metadata}\}
$$

边界也必须说清楚，否则很容易误解：

| 边界项 | 需要关注什么 | 常见误区 |
| --- | --- | --- |
| TP | 切分维度与 offset | 只记 rank id，不记切分规则 |
| PP | 每个 stage 包含哪些层 | 只存参数，不存层映射 |
| DP | 是否每份都有完整 optimizer | 把 DP 理解成“完全复制一切” |
| ZeRO stage 1/2/3 | 分片粒度不同，stage 3 最复杂 | 以为 load 单 rank 就能得到完整参数 |
| `dp_reshardable` | 保存快，但不一定支持并行度变化 | 误以为任意并行度都能恢复 |
| `fully_reshardable` | 更灵活，适合迁移 | 忽略额外保存与恢复成本 |
| metadata 可见性 | 所有加载节点都要能看到 | 只在 rank0 本地盘存在 |

---

## 核心机制与推导

先看最核心的 TP 切分。设一个全量线性层权重为：

$$
W \in \mathbb{R}^{d_{out} \times d_{in}}
$$

若采用列切分的 Tensor Parallel，TP 大小为 $N$，则第 $i$ 个 rank 保存：

$$
W^{(i)} \in \mathbb{R}^{d_{out} \times (d_{in}/N)}
$$

对应偏移量为：

$$
offset_i = i \cdot \frac{d_{in}}{N}
$$

这意味着 checkpoint 里不能只写“这是 rank 2 的权重”，还必须能表达“这是沿输入维切出来、起始偏移为 2048 的那一段”。

玩具例子如下。

假设某个全连接层满足：

- $h = 4096$
- 使用 ColumnParallel
- 全量权重形状是 $4096 \times 4096$
- TP = 4

则每个 rank 只存：

$$
4096 \times 1024
$$

四个 rank 的 offset 分别是：

| TP rank | shard 形状 | offset |
| --- | --- | --- |
| 0 | $4096 \times 1024$ | 0 |
| 1 | $4096 \times 1024$ | 1024 |
| 2 | $4096 \times 1024$ | 2048 |
| 3 | $4096 \times 1024$ | 3072 |

恢复时只需要按 offset 排序，然后沿输入维拼接：

$$
W = \text{concat}(W^{(0)}, W^{(1)}, W^{(2)}, W^{(3)}, \text{dim}=1)
$$

如果是 RowParallel，切分逻辑就换成另一维，核心思想不变：checkpoint 里必须保存“形状 + 切分维 + offset”，否则无法重建全局张量。

PP 的机制不同。PP 不是切同一个矩阵，而是切模型层。例如 48 层 Transformer，PP=4 时，每个 stage 可能负责 12 层。此时 checkpoint 必须知道“第 13 到 24 层在 stage 1”，通常依赖类似 `build_pipeline_layer_name_mapping` 的逻辑构建层名到 stage 的映射。没有这份映射，加载时即使张量文件存在，也不知道该把哪一层交给哪个 stage。

ZeRO 的重点在优化器。以 Adam 为例，一个参数除了参数值本身，还对应一阶矩 `m` 和二阶矩 `v`。ZeRO 会把这些状态再按 DP rank 分片，所以 optimizer checkpoint 里不仅要有数据，还要有“哪段状态属于哪个参数、哪个 rank、总长度是多少”的元信息。恢复时的核心动作通常是 concat 或 reshard。

可以把三类 metadata 概括为：

| 机制 | 关键 metadata 字段 | 用途 |
| --- | --- | --- |
| TP | `axis`、`offset`、`global_shape` | 告诉系统如何合并参数分片 |
| PP | `layer_name -> stage_id` | 告诉系统每层属于哪个 pipeline stage |
| ZeRO | `param_id`、`dp_rank`、`shard_range` | 告诉系统如何恢复优化器分片 |

真实工程例子是训练 70B 级模型时的 checkpoint 目录。目录中可能同时存在多个 `__0_0.distcp`、`__1_0.distcp`、`__0_1.distcp` 文件，对应不同 TP/PP rank；再配合 `.metadata`、`metadata.json`、`latest_train_state.pt`、`run_config.yaml`，系统才能从“很多局部文件”恢复成“一个全局训练状态”。

---

## 代码实现

下面用一个最小 Python 例子模拟“切分保存”和“按 metadata 恢复”。代码可直接运行，重点是理解 `axis` 和 `offset` 的作用。

```python
import numpy as np

def shard_column_parallel(weight: np.ndarray, tp_size: int):
    assert weight.ndim == 2
    d_out, d_in = weight.shape
    assert d_in % tp_size == 0
    shard_width = d_in // tp_size
    shards = []

    for tp_rank in range(tp_size):
        start = tp_rank * shard_width
        end = start + shard_width
        shard = {
            "tp_rank": tp_rank,
            "axis": 1,
            "offset": start,
            "global_shape": weight.shape,
            "data": weight[:, start:end].copy(),
        }
        shards.append(shard)
    return shards

def restore_from_shards(shards):
    assert len(shards) > 0
    axis = shards[0]["axis"]
    global_shape = tuple(shards[0]["global_shape"])
    ordered = sorted(shards, key=lambda x: x["offset"])
    restored = np.concatenate([s["data"] for s in ordered], axis=axis)
    assert restored.shape == global_shape
    return restored

# 玩具例子：4096x4096 太大，不利于展示，先用 4x8 做同构演示
W = np.arange(32).reshape(4, 8)
shards = shard_column_parallel(W, tp_size=4)

assert shards[0]["data"].shape == (4, 2)
assert shards[1]["offset"] == 2
assert shards[2]["offset"] == 4
assert shards[3]["offset"] == 6

W_restored = restore_from_shards(shards)
assert np.array_equal(W, W_restored)

print("ok")
```

上面代码对应的工程含义是：

1. 保存时，不存完整 `W`，而是每个 rank 存自己那段 `data`。
2. 同时必须保存 `axis`、`offset`、`global_shape`。
3. 恢复时先按 `offset` 排序，再沿 `axis` 拼接。

把它映射到实际工程，伪代码通常是这样：

```python
def save_sharded_checkpoint(model, optimizer, parallel_state, ckpt_dir):
    layer_mapping = build_pipeline_layer_name_mapping(parallel_state.pp_stage_id)
    sharded_state = build_sharded_state_dict(
        model=model,
        optimizer=optimizer,
        tp_rank=parallel_state.tp_rank,
        pp_rank=parallel_state.pp_rank,
        dp_rank=parallel_state.dp_rank,
        layer_mapping=layer_mapping,
    )
    save_to_distcp(
        path=f"{ckpt_dir}/__{parallel_state.tp_rank}_{parallel_state.pp_rank}.distcp",
        state_dict=sharded_state,
    )
    if parallel_state.global_rank == 0:
        write_metadata_json(ckpt_dir, parallel_state, layer_mapping)
        write_latest_train_state(ckpt_dir)
```

如果要把旧 checkpoint 转成 `torch_dist`，逻辑通常是：

1. 先读单文件或旧格式 checkpoint。
2. 根据目标 TP/PP/DP 配置，重建全局参数视图。
3. 再按新配置重新切 shard。
4. 写出新的 `__<tp>_<pp>.distcp` 和 metadata。

这就是 `tools/checkpoint/convert.py` 这类工具的核心价值。它做的不是简单“改文件名”，而是“读旧布局，生成新布局”。

旧格式和 `torch_dist` 的差异可以总结为：

| 方案 | 文件布局 | 优点 | 代价 |
| --- | --- | --- | --- |
| 传统单文件 checkpoint | 一个或少量大文件 | 结构简单 | 大模型下 IO 压力大，跨并行度迁移弱 |
| 旧式按 rank 存 torch checkpoint | 每 rank 一个文件，但 metadata 弱 | 好于单文件 | 不一定能安全变更并行度 |
| `torch_dist` | 多个 `.distcp` + metadata | 适合大模型和并行度迁移 | 实现更复杂 |

---

## 工程权衡与常见坑

第一个常见坑是把 `dp_reshardable` 当成“万能可迁移格式”。实际并不是。很多实现里，它更偏向当前 DP 布局可高效恢复，但不保证你之后把 TP、PP、DP 改掉还能直接加载。

典型错误场景是：你在 `TP=4, PP=2` 下训练并保存了 checkpoint，之后想改成 `TP=8, PP=1` 继续训练。如果保存时只用了偏保守的 `dp_reshardable` 布局，加载阶段可能直接失败，因为优化器或参数 shard 的 metadata 不足以支撑重新切分。

第二个坑是 metadata 只在 rank0 本地可见。分布式环境里，rank0 写出的 `.metadata` 如果只落在本机盘，别的节点加载时即使拿到了 `.distcp` 文件，也会因为缺少入口元数据而失败。这个问题本质不是“参数丢了”，而是“拼图说明书丢了”。

第三个坑是误解 ZeRO stage 3。stage 3 下参数、梯度、优化器状态都可能分片，所以单个 rank 加载后看到的只是局部状态，不是完整参数。若你的后处理逻辑默认“load 完就能导出完整模型”，通常会踩坑，需要显式 gather 或 concat。

常见问题与规避策略如下：

| 常见坑 | 现象 | 规避策略 |
| --- | --- | --- |
| metadata 只在 rank0 可见 | 跨节点恢复失败 | 确保共享存储或显式同步 metadata |
| `dp_reshardable` 后改并行度 | load 报错或 optimizer 不匹配 | 迁移前先保存一次 `fully_reshardable` |
| ZeRO stage 3 直接导出 | 得到不完整权重 | 先 gather/concat 完整参数 |
| PP 层映射变化 | stage 找不到对应层 | 保存并校验 layer mapping |
| TP offset 错误 | 拼接后 shape 对但数值错 | 同时校验 `axis`、`offset`、`global_shape` |

工程上常见做法是：平时训练为了性能，用当前更高效的分片格式；需要迁移并行度、迁移集群、导出全量模型时，额外保存一次 `fully_reshardable` checkpoint，作为“中转格式”。

例如配置上通常会显式打开类似能力：

```bash
--dist-ckpt-format=torch_dist
--dist-ckpt-optim-fully-reshardable
```

这类参数的含义可以理解为：优化器状态要按“未来可重新切分”的方式保存，而不是只满足“当前布局最快恢复”。

---

## 替代方案与适用边界

不是所有项目都需要 `torch_dist`。如果你的模型规模小、并行度固定、未来也不打算迁移训练拓扑，那么单文件 checkpoint 或简单的 HF 风格 checkpoint 更直接。

真正适合 `torch_dist` 的场景，是“大模型 + 多维并行 + 未来可能变更并行度或部署形态”。此时 checkpoint 的核心诉求不再是“能存下来”，而是“能在不同机器、不同 rank 拓扑下重新装回去”。

三种常见方案可以这样比较：

| 方案 | 适合场景 | 并行度扩展能力 | IO 特性 | 迁移能力 |
| --- | --- | --- | --- | --- |
| 单文件 HF checkpoint | 单机、小模型、TP=1/PP=1 | 弱 | 单文件大 | 主要面向推理或简单微调 |
| DeepSpeed 常规 checkpoint | 固定 ZeRO 训练场景 | 中 | 分片较多 | 对原训练拓扑友好 |
| Megatron `torch_dist` | 大模型、多维并行、需迁移 | 强 | 并行写盘更稳 | 最适合跨并行度恢复 |

可以用一个简单决策树判断：

1. 如果 `TP=1`、`PP=1`、模型不大，优先单文件或 HF checkpoint。
2. 如果主要是固定集群上的 ZeRO 训练，且不常改并行度，可用更简单的分片方案。
3. 如果模型已经大到必须依赖 TP/PP，或者未来可能从 `TP=4` 切到 `TP=8`，优先 `torch_dist`，并保留完整 metadata。
4. 如果要做训练恢复、格式转换、导出到 HF 三者兼顾，应把“全局 metadata 完整性”当作一等公民，而不是附属文件。

一句话总结适用边界：`torch_dist` 不是为了让 checkpoint“更复杂”，而是为了让大模型训练在复杂并行下仍然“可恢复、可迁移、可验证”。

---

## 参考资料

1. NVIDIA Megatron Core `dist_checkpointing` API 文档  
   https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html  
   说明分布式 checkpoint 的核心接口、metadata、reshardable 保存策略。

2. NVIDIA NeMo Megatron Bridge Checkpointing 文档  
   https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html  
   说明 `.distcp` 文件结构、`.metadata`、`latest_train_state.pt` 等目录布局。

3. RLinf 并行训练教程（5D Parallelism）  
   https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/5D.html  
   用较直观方式解释 TP、PP、DP、ZeRO 的切分关系，适合理解 shard 机制。

4. DeepSpeed ZeRO 论文  
   https://arxiv.org/abs/1910.02054  
   说明优化器状态、梯度、参数分片的理论基础，是理解 ZeRO checkpoint 的底层来源。

5. Megatron-LM `tools/checkpoint/convert.py` 相关实现与说明  
   可在 Megatron-LM 仓库中查看对应工具脚本与 README  
   说明旧 checkpoint 到新分布式 checkpoint、以及与其他格式互转的工程流程。
