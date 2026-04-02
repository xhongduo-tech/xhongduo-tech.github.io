## 核心结论

FSDP，Fully Sharded Data Parallel，中文常译为“完全分片数据并行”，本质上仍是数据并行。白话说，它没有把一个 batch 的计算流程拆成流水线，也没有要求你手工把矩阵切到不同卡上；它做的事是把**模型状态**拆开保存，让每张 GPU 平时只保留自己那一片参数、梯度和优化器状态。

它成立的关键原因是：训练某一层时，真正必须“看见整层参数”的时间很短。于是 FSDP 采用“先临时拼齐，算完立刻拆回去”的策略，把全量参数只保留在局部时间窗口里，而不是整轮训练都常驻显存。对 `FULL_SHARD` 策略，可以写成：

$$
\Theta = \bigsqcup_{r=0}^{N-1}\Theta_r
$$

其中 $\Theta$ 是模型总参数集合，$\Theta_r$ 是第 $r$ 张卡持有的那一片，符号 $\bigsqcup$ 表示“不重叠并集”。前向和反向的核心流程是：

$$
\text{all-gather}(\Theta_r)\rightarrow \Theta_L \rightarrow \text{compute} \rightarrow \text{reduce-scatter}\left(\frac{\partial \mathcal{L}}{\partial \Theta_L}\right)\rightarrow \frac{\partial \mathcal{L}}{\partial \Theta_r}
$$

直观结论有三条：

| 方案 | 每卡参数副本 | 主要同步方式 | 显存峰值特征 |
|---|---:|---|---|
| DDP | 1 份完整模型 | all-reduce 梯度 | 接近整模型常驻 |
| FSDP `FULL_SHARD` | 约 $1/N$ 模型 | all-gather + reduce-scatter | 更接近“单层峰值”而不是“整模型峰值” |
| FSDP `SHARD_GRAD_OP` | 参数全量，梯度/优化器分片 | reduce-scatter 等 | 介于 DDP 与 `FULL_SHARD` 之间 |

玩具例子：3 张卡训练一个只有 3 层的小模型，卡 0、1、2 各存 1/3 参数。进入第 2 层时，三张卡先把第 2 层参数临时拼齐，算完输出后立刻把梯度拆回三张卡，再把全量参数释放。于是显存里不会一直躺着整套模型。

真实工程例子：4 卡训练 2.4 亿参数模型，如果使用 `FULL_SHARD`，每卡常驻大约 6000 万参数对应的 shard；某层计算时临时 all-gather 到全量，反向后 reduce-scatter 并 reshard，释放上下层不再需要的全量表示。它不能让通信消失，但能显著降低冗余显存。

---

## 问题定义与边界

FSDP 解决的问题不是“分布式训练更快”这个泛化命题，而是更具体的：**单卡装不下完整训练状态时，怎样尽量保留数据并行的编程模型，同时把显存压力压下去。**

训练时真正占显存的不是只有参数，还包括三类状态：

| 项目 | 白话解释 | 是否常见为瓶颈 |
|---|---|---|
| 参数 | 模型权重本身 | 是 |
| 梯度 | 反向传播得到的更新量 | 是 |
| 优化器状态 | 例如 Adam 的动量和二阶矩 | 很常见 |
| 激活 | 前向中间结果，反向要回看 | 大 batch 或长序列时很常见 |

如果你用传统 DDP，每张卡都保存完整参数，梯度和优化器状态通常也跟着全量存在。模型一旦上到数亿参数，24GB 或 48GB 显存很容易先被状态占满，而不是先被算力打满。

新手常见场景是：24GB GPU 训练一个 100M 以上参数模型，前向刚开始还能跑，反向或者 Adam 初始化时 OOM。原因通常不是“代码写错”，而是**整模型副本 + 梯度 + 优化器状态**的总和已经超过显存。FSDP 的思路是只让每张卡长期保留自己的 shard，前向需要某层时再临时聚齐。

它的适用边界可以粗略看成下面这张表：

| 模型参数量 | 单卡显存 | 是否优先考虑 shard |
|---|---|---|
| 明显小于单卡可容纳上限 | 充足 | 通常不必，DDP 更简单 |
| 接近单卡上限 | 紧张 | 应该考虑 FSDP |
| 超过单卡上限很多 | 非常紧张 | FSDP 很可能需要，再叠加激活检查点或混合并行 |
| 单层本身都过大 | 再多卡也紧张 | 仅靠 FSDP 可能不够，需模型并行 |

这里有一个边界必须说清：FSDP 不是“任何超大模型都能单靠它训练”。如果某一层在 all-gather 后的临时全量参数，再加上该层激活，依然放不进单卡显存，那么 FSDP 也会短暂 OOM。它优化的是**冗余常驻内存**，不是消灭单层峰值。

---

## 核心机制与推导

FSDP 的核心不是“把模型切开”，而是“把模型状态在时间上做精细管理”。

设总参数为 $\Theta$，有 $N$ 个 rank。FSDP 做的第一步是按 rank 分片：

$$
\Theta = \bigsqcup_{r=0}^{N-1} \Theta_r,\qquad |\Theta_r| \approx \frac{|\Theta|}{N}
$$

白话说，4 张卡时，每张卡只长期保留 1/4 左右的参数、对应梯度和优化器状态。

然后进入某个被 FSDP 包裹的模块 $L$ 时，训练流程近似如下：

1. all-gather 当前模块所需 shard，临时恢复该模块全量参数 $\Theta_L$
2. 执行前向计算
3. 如果策略是 `FULL_SHARD`，前向后可立即 reshard，释放全量参数
4. 反向到该模块时，再次 unshard 或利用已调度好的全量表示
5. 得到局部梯度后做 reduce-scatter，把全量梯度拆回各 rank
6. 优化器只更新本地 shard，对应优化器状态也只保留本地 shard

为什么这比 DDP 更省显存？因为 DDP 等价于“每张卡都常驻整模型，然后把梯度 all-reduce 同步”；而 FSDP 可以看成把一部分 all-reduce 语义分解成了 all-gather 和 reduce-scatter。PyTorch 教程也明确把 FSDP 描述为 DDP 的 all-reduce 分解。

初学者版理解方式可以是“拼图”：

- DDP：每个人桌上都放一整幅拼图，大家分别计算，再把修改意见同步
- FSDP：每个人桌上只放自己那块拼图，轮到某一块要算时，先把这一块临时拼齐，算完马上拆回去

玩具例子：3 张卡、一个线性层权重共有 12 个数。

- 卡 0 保存 `[w0,w1,w2,w3]`
- 卡 1 保存 `[w4,w5,w6,w7]`
- 卡 2 保存 `[w8,w9,w10,w11]`

做这一层前向时，三张卡各自参与 all-gather，临时都拿到 12 个数，完成矩阵乘法。反向时每张卡先得到本地全量梯度视图，再通过 reduce-scatter 各自只保留 4 个梯度。优化器更新后，每张卡继续只持有自己的 4 个参数和状态。

真实工程里，这个流程还能和预取重叠。比如当前层计算时，下一层的 all-gather 可以提早发起，用通信和计算重叠换吞吐。但代价是瞬时内存会上升，所以 PyTorch 又提供 `limit_all_gathers` 之类的速率限制，防止你同时预取太多层，把显存重新顶爆。

---

## 代码实现

PyTorch 中最重要的约束只有一句：**先 wrap，再创建 optimizer。**因为 FSDP 会改写参数变量的组织方式，优化器如果提前拿到旧参数引用，后面状态会错位。

下面先给一个可运行的玩具代码，演示“整模型副本”和“完全分片”在内存量级上的差异。它不是在跑真实通信，只是把数学账算清楚。

```python
def estimate_memory_units(total_params, world_size, bytes_per_param=2, adam_multiplier=3):
    """
    简化估算：
    - 参数本体: 1 份
    - 梯度: 1 份
    - Adam 状态: 2 份
    因此训练态总计约 4 份参数量，这里用 adam_multiplier=3 表示 参数+梯度+2个状态 = 4
    """
    ddp_per_rank = total_params * (1 + adam_multiplier) * bytes_per_param
    fsdp_per_rank_steady = (total_params / world_size) * (1 + adam_multiplier) * bytes_per_param
    return ddp_per_rank, fsdp_per_rank_steady

ddp, fsdp = estimate_memory_units(total_params=240_000_000, world_size=4)
assert ddp == 240_000_000 * 4 * 2
assert fsdp == (240_000_000 / 4) * 4 * 2
assert fsdp < ddp
print({"ddp_bytes_per_rank": ddp, "fsdp_bytes_per_rank_steady": fsdp})
```

真实训练中，最小使用流程通常是：

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def build_fsdp_model(model, device):
    auto_wrap = lambda module, recurse, nonwrapped_numel: size_based_auto_wrap_policy(
        module, recurse, nonwrapped_numel, min_num_params=1_000_000
    )
    fsdp_model = FSDP(
        model.to(device),
        auto_wrap_policy=auto_wrap,
        device_id=device,
        limit_all_gathers=True,
    )
    return fsdp_model

def train_one_step(fsdp_model, optimizer, batch, target, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    pred = fsdp_model(batch)
    loss = loss_fn(pred, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def save_sharded_checkpoint(fsdp_model, optimizer, path):
    FSDP.set_state_dict_type(
        fsdp_model,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )
    state = {
        "model": fsdp_model.state_dict(),
        "optim": FSDP.optim_state_dict(fsdp_model, optimizer),
    }
    torch.save(state, path)

# 关键顺序：
# 1. model = MyModel()
# 2. fsdp_model = build_fsdp_model(model, device)
# 3. optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
```

这里有三个工程点比 API 名字更重要。

第一，`auto_wrap_policy` 或手工 wrap 决定 shard 粒度。粒度太粗，一个模块太大，all-gather 峰值仍可能爆；粒度太细，通信次数和调度开销会上去。

第二，checkpoint 默认不要先想 `FULL_STATE_DICT`。对大模型，更稳的路径是 `SHARDED_STATE_DICT`，每个 rank 保存自己的 shard，再用分布式 checkpoint 工具做汇总或恢复。

第三，FSDP 经常和 mixed precision、activation checkpointing、meta device 初始化一起用。前者减少参数和激活字节数，后两者分别压运行峰值和初始化峰值。

真实工程例子：训练一个多层 Transformer 时，通常会把每个 block 作为 wrap 单元，而不是整个模型只包一层 FSDP。这样每次 all-gather 的对象接近“单个 block”，更容易控制峰值，也更适合通信和计算重叠。

---

## 工程权衡与常见坑

FSDP 的好处非常明确，但坑也非常集中，基本都和“临时全量”和“状态管理”有关。

| 常见坑 | 触发原因 | 规避方式 |
|---|---|---|
| 前向/反向仍然 OOM | 某层 all-gather 后临时全量太大，或激活太大 | 减小 wrap 粒度、开混合精度、加 activation checkpointing |
| 初始化就 OOM | 先在 GPU 上构完整模型再 wrap | 用 meta device 或 CPU 初始化，再交给 FSDP materialize |
| checkpoint 保存炸内存 | 用 `FULL_STATE_DICT` 让 rank0 聚齐整模型 | 大模型优先 `SHARDED_STATE_DICT` |
| optimizer 状态错乱 | wrap 前创建 optimizer | 必须先 FSDP wrap，再建 optimizer |
| 训练挂起 | 各 rank 没执行相同的 FSDP 模块顺序 | 保证控制流一致，避免按 rank 条件分支跑不同层 |
| 某子模块 forward 报错 | 直接调用未单独 FSDP 化的内部子模块 | 调根模块，或把目标子模块本身也纳入 FSDP 包裹 |

新手最容易踩的是 checkpoint。比如训练到一半要续训，图省事直接：

```python
torch.save(model.state_dict(), "ckpt.pt")
```

如果此时模型很大，这么做要么只存下当前 rank 的碎片，要么在取 full state dict 时把 rank0 内存打爆。正确思路是：训练态保存 shard，恢复时也按 shard 加载。只有在模型较小、明确需要单文件导出时，才考虑 full state dict，并常配合 `offload_to_cpu=True`。

另一个常见误解是“用了 FSDP 就一定比 DDP 快”。这不对。FSDP 优先解决的是**内存可训练性**，不是无条件吞吐最优。对于本来就能轻松放进单卡显存的中小模型，DDP 往往更简单、同步路径也更直接，整体性能可能更好。

还有一个真实工程约束：共享参数、模块外部持有参数引用、复杂自定义 forward 路径，都会让 FSDP 更难配置。因为它会改写参数生命周期，如果代码里到处缓存 `module.weight` 的外部引用，或者只在某些 rank 上执行某些层，分片状态就可能失配。

---

## 替代方案与适用边界

FSDP 不是分布式训练的唯一解，它在“显存优先”维度很强，但并不覆盖所有问题。

| 方案 | 显存需求 | 通信复杂度 | 实现难度 | 适用场景 |
|---|---|---|---|---|
| DDP | 高，每卡整模型 | 中 | 低 | 模型能放进单卡，优先稳定与简单 |
| FSDP | 低到中，取决于 shard 粒度 | 中到高 | 中 | 模型接近或超过单卡上限，想保留数据并行心智模型 |
| Tensor Parallel / 模型并行 | 可处理更大单层 | 高，且依赖拓扑 | 高 | 单层都放不下，需把算子本身拆开 |
| Pipeline Parallel | 降低单卡层数负担 | 中到高 | 高 | 深模型、分阶段流水训练 |

如果模型参数本来就在单卡显存内，且团队优先考虑可维护性，通常先上 DDP。它更容易 debug，也更少遇到 state dict 和 wrap 粒度问题。

如果模型已经开始被参数、梯度、优化器状态压到显存边缘，FSDP 往往是第一选择。它不要求你理解每个矩阵怎么切，只要求你把模块边界和 checkpoint 流程理顺。

如果单层矩阵本身已经大到 all-gather 后也放不下，那就超出 FSDP 的舒适区了。这时更适合 Tensor Parallel，也就是把一个大矩阵乘法本身拆到多卡上算。很多真实系统会混合使用 FSDP + Tensor Parallel，再叠加 activation checkpointing。

题目里给的 8×A100、30B 到 100B 例子，可以这样理解：如果 30B 在目标精度、batch 和序列长度下已经能稳定放进方案预算，DDP 或较轻的分片就够；如果要继续推到 100B，单靠整模型副本明显不现实，这时就要引入 FSDP，甚至和 Tensor Parallel 混用。

---

## 参考资料

1. PyTorch 官方 FSDP 文档：<https://docs.pytorch.org/docs/stable/fsdp.html>
2. PyTorch 官方教程，解释 all-gather / reduce-scatter 工作流：<https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html>
3. PyTorch 官方博客，包含 175B 与 1T 模型的集群实验指标：<https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>
4. Runebook 对 FSDP OOM、meta device、checkpoint 的排错说明：<https://runebook.dev/en/docs/pytorch/fsdp/torch.distributed.fsdp.FullyShardedDataParallel>
5. Runebook 对 `StateDictSettings` 与分片保存/加载的补充说明：<https://runebook.dev/en/docs/pytorch/fsdp/torch.distributed.fsdp.StateDictSettings>
