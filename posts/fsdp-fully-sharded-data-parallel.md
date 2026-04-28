## 核心结论

FSDP（Fully Sharded Data Parallel，完全分片数据并行）不是“更快的 DDP”，而是“把完整模型拆成分片常驻显存，按需聚齐计算，算完立即再分片”。术语里的“分片”可以先理解成“把一整份大对象切成多段，分别放到不同 GPU 上”。

设全量参数为 $\theta$，共有 $N$ 张卡，第 $r$ 张卡平时只持有自己的参数分片 $\theta_r$。稳态下，每卡常驻的参数、梯度、优化器状态都近似缩小到原来的 $1/N$。这就是它能训练大模型的根本原因。

新手版可以这样理解：

- DDP 像每个人都拿一份完整菜单，谁都常驻持有全菜单。
- FSDP 像每个人只拿菜单的一页，真正点菜前再把整本菜单临时拼起来，点完立刻拆回去。

区别不在“有没有完整菜单”，而在“谁平时常驻持有完整菜单”。

| 方案 | 每卡常驻模型副本 | 主要通信模式 | 主要目标 | 适用场景 |
|---|---:|---|---|---|
| DDP | 完整一份 | `all-reduce` | 简单、稳定、吞吐高 | 模型单卡放得下 |
| FSDP | 约 `1/N` 分片 | `all-gather + reduce-scatter` | 降低显存占用 | 大模型全参训练/微调 |

最小初始化示意先看整体，不必一开始理解全部细节：

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def build_fsdp_model(model: torch.nn.Module):
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    model = model.cuda()
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, optimizer
```

---

## 问题定义与边界

FSDP 解决的问题不是“训练太慢”，而是“参数、梯度、优化器状态太大，单卡显存放不下”。这里的“优化器状态”可以先理解成“优化器为了更新参数而额外保存的历史信息”，例如 Adam 会额外维护一阶矩和二阶矩。

训练时显存大致可以拆成：

$$
\text{Memory} \approx \text{Parameters} + \text{Gradients} + \text{Optimizer States} + \text{Activations}
$$

其中 FSDP 主要优化前三项，对激活值的帮助有限。激活值是前向传播中间结果，用于反向求梯度；它通常要靠 activation checkpointing 之类的方法继续压缩。

玩具例子：

- 你有一个 7B 模型，单卡只有 24GB。
- 如果用 DDP，每张卡都要放完整参数、完整梯度、完整优化器状态。
- 如果用 Adam，全参训练时显存压力通常很快超过单卡上限。
- FSDP 把这三类张量按 rank 切开后，训练才有机会跑起来。

真实工程例子：

- 8 张 GPU 做 7B 或 13B 模型全参数微调。
- 如果坚持 DDP，往往只能极端缩小 batch size，甚至直接 OOM。
- 换成 FSDP 后，常见做法是按 Transformer block 自动 wrap，再叠加 mixed precision 和 activation checkpointing，把训练拉回可运行区间。

| 维度 | FSDP 能解决什么 | FSDP 不能直接解决什么 | 何时不该用 |
|---|---|---|---|
| 显存 | 降低参数/梯度/优化器状态常驻显存 | 激活值爆炸本身 | 小模型本来就能用 DDP 跑稳 |
| 并行语义 | 保持数据并行训练方式 | 不能替代张量并行的算子切分 | 你需要极致吞吐而不是省显存 |
| 工程改造 | 相对少改模型结构 | 不能免除分布式通信成本 | 纯推理场景通常不是首选 |

下面这段对照代码能说明为什么 DDP 会直接撞到边界：

```python
# 只表示思路，不是完整训练脚本
model = HugeModel().cuda()
ddp_model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

# 问题在于：每张卡都持有完整参数、完整梯度、完整优化器状态
# 如果模型本体已经接近单卡显存上限，训练基本必然 OOM
```

---

## 核心机制与推导

FSDP 的核心流程是：

1. 模型参数按 rank 分片常驻。
2. 某个 FSDP 单元准备前向时，先对该单元参数做 `all-gather`，临时拼成完整参数。
3. 前向和反向用完整参数完成计算。
4. 该单元算完后，参数重新 `reshard`，也就是再次拆回分片。
5. 梯度同步不再走 DDP 的 `all-reduce`，而是走 `reduce-scatter`，每张卡只留下自己的梯度分片。
6. 优化器只更新本 rank 持有的参数分片。

白话解释：

- `all-gather` 就是“把大家手里的碎片收齐，临时拼成完整副本”。
- `reduce-scatter` 就是“先把梯度求和，再把结果切片分回各卡”。

这也是为什么常说 FSDP 在通信上把 DDP 的 `all-reduce` 拆成了 `all-gather + reduce-scatter`。

玩具数值例子：

- 总参数数目 $P = 12$
- GPU 数目 $N = 3$
- 每张卡平时持有 $12/3 = 4$ 个参数
- 使用 Adam 时，每个参数通常对应 2 份优化器状态

于是：

- DDP 每卡约保存：参数 12 + 梯度 12 + Adam 状态 24 = 48
- FSDP 稳态每卡约保存：参数 4 + 梯度 4 + Adam 状态 8 = 16

一般化表达是：

$$
|\theta_r| \approx \frac{|\theta|}{N}, \quad |g_r| \approx \frac{|g|}{N}, \quad |m_r|, |v_r| \approx \frac{1}{N}
$$

所以稳态下每卡与参数相关的常驻显存，近似变成原来的 $1/N$。

| 阶段 | DDP 张量状态 | `FULL_SHARD` 张量状态 |
|---|---|---|
| 前向前 | 完整参数常驻 | 仅参数分片常驻 |
| 前向计算 | 直接用完整参数 | 先 `all-gather` 再算 |
| 反向计算 | 本地算梯度后 `all-reduce` | 用完整参数反向后 `reduce-scatter` 梯度 |
| `optimizer.step()` | 每卡更新完整参数 | 每卡只更新自己的参数分片 |

简化训练伪代码如下：

```python
# 逻辑伪代码
for fsdp_unit in model:
    full_param = all_gather(shard_param)   # 临时聚齐
    out = forward(full_param, x)
    free_or_reshard(full_param)            # 算完再拆回去

loss.backward()

for fsdp_unit in model:
    shard_grad = reduce_scatter(full_grad) # 只保留本 rank 需要的梯度分片

optimizer.step()  # 只更新本 rank 持有的参数分片
```

下面这段 Python 代码可以直接运行，帮助你先把“为什么省显存”算明白：

```python
def ddp_elements(param_count: int, adam: bool = True) -> int:
    optim = 2 * param_count if adam else param_count
    return param_count + param_count + optim  # params + grads + optim states

def fsdp_elements(param_count: int, world_size: int, adam: bool = True) -> int:
    shard = param_count // world_size
    optim = 2 * shard if adam else shard
    return shard + shard + optim

P = 12
N = 3

ddp = ddp_elements(P)
fsdp = fsdp_elements(P, N)

assert ddp == 48
assert fsdp == 16
assert fsdp * N == ddp  # 这个玩具例子里恰好线性对齐

print(ddp, fsdp)
```

---

## 代码实现

工程上最常见的一组组合是：

- `FULL_SHARD`
- `auto_wrap_policy`
- mixed precision
- activation checkpointing
- `use_orig_params=True`
- `SHARDED_STATE_DICT` 保存 checkpoint

这里最容易错的一点不是 API 名字，而是顺序：先 wrap，再创建 optimizer。原因很直接，optimizer 必须绑定 FSDP 管理后的参数对象，而不是 wrap 之前的原始参数引用。

下面给一段完整但尽量短的 PyTorch 示例：

```python
import os
import torch
import torch.distributed as dist
from torch import nn
from functools import partial
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

class Block(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.ff(x)

class TinyModel(nn.Module):
    def __init__(self, dim=128, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x.mean(dim=1))

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup()

    model = TinyModel().cuda()
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(8, 16, 128, device="cuda")
    y = torch.randint(0, 10, (8,), device="cuda")

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    )
    model_state = model.state_dict()
    optim_state = FSDP.optim_state_dict(model, optimizer)

    assert isinstance(model_state, dict)
    assert isinstance(optim_state, dict)

if __name__ == "__main__":
    main()
```

| 配置项 | 推荐值 | 作用 |
|---|---|---|
| `sharding_strategy` | `FULL_SHARD` | 参数、梯度、优化器状态全分片 |
| `auto_wrap_policy` | 按 Transformer block 包裹 | 控制聚齐/分片粒度 |
| `mixed_precision` | `bf16` 优先 | 降低显存和通信压力 |
| `activation_checkpointing` | 大模型常开 | 继续压低激活值显存 |
| `use_orig_params` | `True` | 更好兼容原始参数访问与 `torch.compile()` |
| `SHARDED_STATE_DICT` | 保存训练 checkpoint | 避免保存完整副本时的额外内存压力 |

真实工程里，7B/13B 级别模型常见配置就是“按 block 自动 wrap + `FULL_SHARD` + `bf16` + activation checkpointing + shard checkpoint”。这套组合的目标不是把单卡吞吐做到最高，而是先让全参训练进入“能跑且可恢复”的区间。

---

## 工程权衡与常见坑

FSDP 的优势是省显存，代价是通信复杂度和工程复杂度都上升。你需要接受一个事实：很多错误不是数学错误，而是生命周期管理错误。

最常见的坑如下：

| 常见坑 | 错误现象 | 正确做法 |
|---|---|---|
| 先建 optimizer 再 wrap | 参数不更新或状态异常 | 先 `FSDP(model)`，再 `optimizer(model.parameters())` |
| 训练中直接改参数 | 修改看起来生效，下一轮又丢失 | 用 `summon_full_params` 包裹修改 |
| 错把 shard checkpoint 当普通 state dict 载入 | `load_state_dict` 报错或键不匹配 | 用 FSDP 的 state-dict 流程或 distributed checkpoint |
| `sync_module_states=True` 时模型还在 CPU | 初始化同步阶段失败 | 先把模块放到 GPU，或指定 `device_id` |
| wrap 粒度太细 | 通信频繁，吞吐下降 | 一般按 Transformer block 级别 wrap |
| 保存 full state dict | rank0 内存压力过大 | 训练保存优先用 shard，导出再聚齐 |

先看一个“错误写法 vs 正确写法”的对比：

```python
# 错误写法
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model = FSDP(model)  # optimizer 绑定的是 wrap 之前的参数

# 正确写法
model = MyModel().cuda()
model = FSDP(model, use_orig_params=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

另一个高频坑是“直接改参数”。FSDP 在前向/反向过程中会把参数替换成不同视图。视图可以先理解成“共享同一底层存储、但表现形式可能不同的张量对象”。因此你以为改的是模型参数，实际上可能改的是临时聚齐出来的副本。

```python
# 错误：修改可能不持久
with torch.no_grad():
    for p in model.parameters():
        p.mul_(0.99)

# 正确：需要在完整参数上下文里改
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

with FSDP.summon_full_params(model, writeback=True):
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.99)
```

关于 `sync_module_states=True`，可以把它理解成“构造 FSDP 时，把 rank 0 的初始参数和 buffer 广播给其他 rank”。这一步本质依赖 GPU 通信，所以模型初始化要满足：

$$
\text{module device} = \text{GPU} \quad \text{or} \quad \text{device\_id is set}
$$

否则同步阶段就可能失败。这个坑经常出现在“rank0 从 checkpoint 加载完参数，但其他 rank 还在 CPU 上等同步”的流程里。

---

## 替代方案与适用边界

FSDP 不是唯一方案。选型时真正要看的是三件事：

1. 模型大小是否已经超过单卡训练容量。
2. 通信成本是否还能接受。
3. 你愿意承担多少工程复杂度。

一个简洁判断式可以写成：

$$
\text{If } \text{Params} + \text{Grads} + \text{Optim States} + \text{Activations} > \text{Single-GPU Memory},
$$

且你仍希望保持数据并行语义而不大改模型结构，那么优先考虑 FSDP。

| 方案 | 解决重点 | 代码改造成本 | 通信特征 | 适用边界 |
|---|---|---:|---|---|
| DDP | 简单复制训练 | 低 | `all-reduce` | 模型放得下单卡 |
| FSDP | 参数相关显存压缩 | 中 | `all-gather + reduce-scatter` | 大模型全参训练 |
| ZeRO | 与 FSDP 类似的状态分片 | 中到高 | 依实现而定 | DeepSpeed 生态常见 |
| Tensor Parallel | 单层算子切分 | 高 | 高频跨卡算子通信 | 单层都放不下时 |
| Pipeline Parallel | 层级切分 | 高 | stage 间传递激活 | 超大模型、多机长链路 |

新手版判断可以很直接：

- 模型不大，优先 DDP。
- 模型单卡放不下，但你又不想重写模型切分逻辑，优先 FSDP。
- 模型再大到单层都装不下，通常要叠加张量并行。
- 如果你已经深度依赖 DeepSpeed 生态，ZeRO 也可能是更自然的选择。

下面是一段配置判断示意：

```python
def choose_strategy(model_fits_single_gpu: bool,
                    need_full_param_training: bool,
                    want_min_model_refactor: bool,
                    layer_too_large_for_single_gpu: bool) -> str:
    if model_fits_single_gpu:
        return "DDP"
    if layer_too_large_for_single_gpu:
        return "Tensor Parallel or Hybrid Parallel"
    if need_full_param_training and want_min_model_refactor:
        return "FSDP"
    return "ZeRO or hybrid solution"

assert choose_strategy(True, True, True, False) == "DDP"
assert choose_strategy(False, True, True, False) == "FSDP"
```

FSDP 的适用边界可以概括成一句话：它适合“单机或多机的大模型全参训练，希望尽量保留普通 PyTorch 模型写法”的场景，但它不是零代价魔法，也不是所有分布式训练问题的统一答案。

---

## 参考资料

| 来源 | 用途 | 推荐阅读顺序 |
|---|---|---:|
| PyTorch FSDP 官方 API | 查参数含义、状态字典、`use_orig_params`、`summon_full_params` | 1 |
| PyTorch FSDP2 教程 | 建立整体流程认知，理解前向/反向中的分片与聚齐 | 2 |
| PyTorch 源码 `fully_sharded_data_parallel.py` | 核对行为细节与限制条件 | 3 |
| FSDP 经验论文 | 看大规模训练中的设计取舍与经验总结 | 4 |

1. [PyTorch FSDP 官方 API](https://docs.pytorch.org/docs/stable/fsdp.html)：对应文中 API 用法、`use_orig_params=True`、`sync_module_states=True`、`summon_full_params`、`StateDictType.SHARDED_STATE_DICT`。
2. [PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)：对应文中“按需聚齐、算完再分片”的整体机制理解。
3. [PyTorch 源码 fully_sharded_data_parallel.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py)：对应文中参数视图替换、修改参数需谨慎、构造与运行时行为细节。
4. [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)：对应文中的工程权衡、扩展经验与大规模训练实践。
