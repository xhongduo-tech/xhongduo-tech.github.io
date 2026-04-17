## 核心结论

FSDP，Fully Sharded Data Parallel，直白说就是“把一个大模型拆成很多小份，分散放到多张卡上，只有轮到某一层计算时才临时拼回完整参数”。它是 PyTorch 原生提供的全分片并行方案，设计目标与 DeepSpeed 的 ZeRO Stage 3 一致：把参数、梯度、优化器状态都做分片，而不是像 DDP 那样每张卡都保留完整副本。

这件事的工程意义很直接。传统 DDP，Distributed Data Parallel，也就是“每张卡都放一整份模型副本”，模型一旦大到单卡放不下，训练就卡死。FSDP 的思路是让每个 rank，也就是“分布式训练中的一个进程/一张卡对应的执行单元”，只常驻自己那一份状态，训练时通过 all-gather，也就是“把多张卡上的分片临时拼成完整张量”的通信操作，给当前活跃层提供完整参数。前向和反向结束后，再把完整参数释放掉，只保留分片。

如果把总参数量记为 $P$，总梯度量记为 $G$，总优化器状态量记为 $O$，GPU 数记为 $N$，那么理想化的每卡常驻状态近似是：

$$
\text{per-rank storage} \approx \frac{P}{N} + \frac{G}{N} + \frac{O}{N}
$$

必要时，当前 active FSDP 实例，也就是“此刻正在执行前向或反向的那个 FSDP 包裹模块”，会临时 all-gather 出完整参数来做计算。因此 FSDP 不是“完全不存全量参数”，而是“绝大多数时间不常驻全量参数”。

下面这张表可以先抓住最核心的记忆点。设共有 $N$ 张卡：

| 状态 | DDP 每卡常驻 | FSDP 每卡常驻 | 该层计算结束后是否释放全量副本 |
| --- | --- | --- | --- |
| 参数 Parameters | $P$ | $\frac{P}{N}$ | 是 |
| 梯度 Gradients | $G$ | $\frac{G}{N}$ | 是，保留分片 |
| 优化器状态 Optimizer States | $O$ | $\frac{O}{N}$ | 常驻分片，不保留全量 |

玩具例子可以这样理解。假设模型只有 4 层，2 张卡训练。DDP 的做法是两张卡都放 4 层完整参数；FSDP 的做法是每张卡只放一半分片。计算第 1 层时，先把第 1 层的参数临时拼完整，算完立刻拆回；接着再轮到第 2 层。这样显存峰值接近“当前层大小 + 通信缓冲”，而不是“整个模型大小”。

真实工程例子是 GPT 级别模型训练。PyTorch 团队公开过用 FSDP 在 AWS 上训练 GPT-175B 和 1T 模型的结果，175B 在 128 张 A100 上达到每卡 159 TFLOP/s；这类规模如果按 DDP 保留全量副本，单卡内存根本不可能承载。FSDP 配合 activation checkpointing，也就是“前向时不把中间激活全存下来，反向时需要时再重算”，以及 CPU offload，也就是“把不用的分片搬到 CPU 内存”，已经是大模型训练的标准组合。

---

## 问题定义与边界

FSDP 要解决的问题，不是“训练能不能更快”，而是更底层的“训练能不能装得下”。当模型进入 10B、70B、175B 这类量级时，参数本身、梯度、优化器状态三部分叠加，单卡显存压力会迅速失控。以 Adam 为例，优化器状态通常远大于单纯参数副本，因此很多时候真正先爆的不是参数，而是优化器状态。

边界先写清楚：

| 约束项 | 影响 |
| --- | --- |
| GPU 数量 $N$ | 决定每卡能摊掉多少分片 |
| 单卡显存 | 决定能否承受“当前层完整参数 + 激活 + 通信缓冲”的峰值 |
| CPU 内存是否可用 | 决定能否使用 CPU offload 进一步降显存 |
| 网络带宽 | 决定频繁 all-gather / reduce-scatter 的代价是否可接受 |
| 混合精度是否可用 | 决定参数、激活、通信是否能进一步缩小 |
| activation checkpointing 是否开启 | 决定激活常驻量是否可控 |

严格地说，FSDP 只能保证“分片常驻”，不能保证“峰值永远等于 $\frac{P}{N}$”。因为当前层执行时还是要临时拿到完整参数。更贴近工程实际的表达是：

$$
\text{peak memory} \approx \text{active layer full params} + \text{local shard states} + \text{activations} + \text{comm buffers}
$$

新手常见误解是：两卡训练 20B 模型，是不是每张卡只需要放 10B 参数就够了。答案是不够。因为除了参数 $P$，还有梯度 $G$ 和优化器状态 $O$。FSDP 的理想常驻近似是：

$$
\frac{P}{N} + \frac{G}{N} + \frac{O}{N}
$$

如果是 2 卡训练 20B 参数模型，那么每张卡常驻的参数分片大约是 10B，对应的梯度分片也是 10B 量级，优化器状态还要再占一块。如果再开 Adam，状态量通常接近甚至超过参数量。真正的边界不是“参数除以卡数”，而是“参数、梯度、优化器状态三者都要除以卡数后还能装下，并且还能承受活跃层临时聚合的峰值”。

所以 FSDP 的问题边界可以总结成一句话：它解决的是“大模型训练的状态常驻问题”，但它仍然受制于“活跃层瞬时峰值”和“通信成本”。这也是为什么 FSDP 常常要和 mixed precision、CPU offload、activation checkpointing 一起出现，而不是单独使用。

---

## 核心机制与推导

FSDP 的核心机制可以压缩成一个循环：

$$
\text{gather} \rightarrow \text{forward/backward compute} \rightarrow \text{scatter/reshard}
$$

这里的 scatter 或 reshard，直白说就是“把刚才临时拼完整的参数再次拆回分片状态”。因此 FSDP 的关键不是单次节省了多少，而是它把“全量参数常驻”改成了“全量参数只在当前层短暂停留”。

如果模型按层做嵌套包装，训练过程可以理解成：

1. 进入某一层前，把这层参数 all-gather 成完整权重。
2. 这一层执行前向。
3. 如果开启 activation checkpointing，不保留这层大部分激活，只保留必要边界信息。
4. 这一层前向结束后，参数重新分片。
5. 反向到这层时，再次为这层准备所需完整参数，完成梯度计算。
6. 反向结束后，梯度做 reduce-scatter，也就是“先聚合再切分回各 rank”，最终只保留本地梯度分片。

玩具例子最适合说明这一点。假设有一个 4 层 MLP，每层参数 1GB，共 4GB，2 张卡训练。

- DDP：每张卡常驻 4GB 参数，还要加梯度和优化器状态。
- FSDP 分层包裹：每张卡平时只常驻每层的一半，总共 2GB 参数分片。
- 计算某一层时，该层会临时 all-gather 成 1GB 完整参数。
- 这层算完后，立刻回到 0.5GB 本地分片状态。

于是这张卡上“和参数相关的峰值”更接近：

$$
0.5 \text{GB 本地分片} + 1.0 \text{GB 当前层完整参数} + \text{buffer}
$$

而不是直接背着整个 4GB 模型走完整个前后向。

真实工程里，模型不会是平均分层，通信也不会没有开销，所以精确峰值不会这么整齐。但趋势成立：如果 auto wrap，也就是“按规则自动给子模块套 FSDP 包装”，能把大 Transformer block 作为基本单位拆开，那么每次只需为当前 block 聚合参数。对 GPT-2、LLaMA 这类堆叠式结构，通常会把每个 Transformer block 作为 FSDP 单元，这样峰值显存常常接近“单层参数 + 本地分片 + 通信缓冲”。

mixed precision，混合精度，白话说就是“前向、反向、通信尽量用更小的数据类型，比如 bf16/fp16，减少内存和带宽”。activation checkpointing 的作用则是把激活显存换成额外算力。两者叠加时，经常能把峰值进一步压到：

$$
\text{peak} \approx \text{single-layer full params in bf16} + \text{comm buffer} + \text{minimal activations}
$$

这也是 FSDP 适合 Transformer 的根本原因：模型有明显层结构，可以自然做层级包装；每层前后向相对独立，方便“用完即拆”。

---

## 代码实现

FSDP 最容易犯错的地方不是 API 名字，而是对象构造顺序。原则只有一句：先 wrap 模型，再构造优化器。因为优化器拿到的是参数对象引用；如果你先对原始模型建优化器，再把模型包装成 FSDP，参数展平、分片后的组织方式已经变了，状态保存和恢复很容易错位。

下面先给一个可运行的 Python 玩具代码。它不依赖 GPU，也不真的调用 FSDP，而是模拟“全量状态”和“按卡分片后的常驻状态”差异，帮助理解公式：

```python
def per_rank_storage(params_gb, grads_gb, optim_gb, world_size):
    assert world_size > 0
    return (params_gb + grads_gb + optim_gb) / world_size

def peak_with_active_layer(shard_resident_gb, active_layer_full_gb, comm_buffer_gb, act_gb):
    return shard_resident_gb + active_layer_full_gb + comm_buffer_gb + act_gb

# 玩具例子：20B 模型抽象成 40GB 参数、40GB 梯度、80GB Adam 状态
resident = per_rank_storage(params_gb=40, grads_gb=40, optim_gb=80, world_size=2)
assert resident == 80

# 假设当前活跃层完整参数 6GB，通信缓冲 2GB，激活 4GB
peak = peak_with_active_layer(shard_resident_gb=resident, active_layer_full_gb=6, comm_buffer_gb=2, act_gb=4)
assert peak == 92

# 对比 DDP：每卡常驻完整状态
ddp_resident = 40 + 40 + 80
assert ddp_resident == 160
assert resident < ddp_resident
```

再看接近真实训练代码的写法。这里的 `auto_wrap_policy` 用来指定“哪些子模块要单独包成 FSDP 单元”。

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

class TransformerBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.ffn(x)

class TinyGPT(nn.Module):
    def __init__(self, layers=4, dim=768):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(layers)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)

model = TinyGPT().cuda()

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

auto_wrap = transformer_auto_wrap_policy({TransformerBlock})

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=mp_policy,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

这个例子里，关键顺序是：

| 步骤 | 是否必须 |
| --- | --- |
| 先构造原始 `nn.Module` | 是 |
| 再 `FSDP(model, ...)` 包装 | 是 |
| 最后 `optimizer = AdamW(model.parameters())` | 是 |

如果要保存 checkpoint，FSDP 还涉及 `state_dict` 类型选择。粗略理解：

- `FULL_STATE_DICT`：导出完整模型权重，适合做通用保存，但会更占内存。
- `SHARDED_STATE_DICT`：按分片保存，更适合大模型训练过程中的断点续训。
- `LOCAL_STATE_DICT`：每个 rank 保存自己的本地状态，适合特定恢复流程。

工程上常见做法是训练中保存分片状态，导出推理权重时再转换成 full state dict。

---

## 工程权衡与常见坑

FSDP 不是“开关一打就结束”的功能，它把内存问题转成了通信、初始化、checkpoint、框架兼容性问题。下面这些坑比 API 本身更重要。

| 常见坑 | 现象 | 规避策略 |
| --- | --- | --- |
| 优化器构造顺序错误 | 保存或恢复后参数和优化器状态对不上 | 先 `FSDP(model)`，再 `optimizer(model.parameters())` |
| Accelerate 多模型顺序不一致 | `save_state/load_state` 行为异常 | `prepare(model1, model2, optim1, optim2)` 的顺序严格对应 |
| `predict_with_generate` 与 FSDP 插件不兼容 | 生成任务报错或行为异常 | 生成逻辑单独跑，或切回非 FSDP 推理路径 |
| auto wrap 粒度过粗 | 峰值显存仍然很高 | 按 Transformer block 粒度拆分，不要只包最外层 |
| CPU offload 开太多 | 显存降了但吞吐显著下降 | 先确认是否真被显存卡住，再权衡 PCIe/CPU 带宽 |
| 混合精度配置不一致 | dtype 混乱，数值或性能异常 | 嵌套 FSDP 时明确 `MixedPrecision` 策略 |
| checkpoint 类型选错 | 保存成功但恢复复杂或内存暴涨 | 训练续训优先 sharded，导出模型再转 full |

先说最重要的一条：在 Hugging Face Accelerate 里，如果多个模型和优化器一起 `prepare`，优化器顺序必须和对应模型顺序一致，否则 `accelerator.save_state()` 和 `accelerator.load_state()` 可能出现错误行为。这类问题最麻烦，因为训练过程未必立刻报错，往往是恢复训练后 loss 异常，或者参数更新不符合预期。

一个简化后的工程例子如下：

```python
model = FSDP(model, auto_wrap_policy=auto_wrap)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

错误写法通常是先对未包装模型建优化器，或者在多模型场景里把 `model_a, model_b, optim_b, optim_a` 这样的顺序混过去。FSDP 会重新组织参数，优化器状态必须跟着最终参数结构走。

再说 `predict_with_generate`。这是 Hugging Face 一些 seq2seq 脚本里的生成路径。Accelerate 文档明确提到它和 FSDP 插件不兼容。原因并不神秘：训练阶段的分片和生成阶段对完整权重、缓存、同步时机的要求不同。工程上通常会把“训练”和“批量生成评估”拆成两条路径，而不是硬塞进同一个 FSDP 训练进程里。

最后一个常见误区是“FSDP 一定比 DDP 更优”。不对。FSDP 的收益是显存可控，但代价是每层前后向都要更频繁通信。如果模型不够大，或者机器间带宽一般，DDP 加混合精度可能更简单、更稳定，整体吞吐也可能更高。

---

## 替代方案与适用边界

从方案族谱看，FSDP 不是孤立存在的。它和 FairScale ZeRO-3、DeepSpeed ZeRO-3 解决的是同一类问题：把参数、梯度、优化器状态都做分片。差异主要在生态、封装方式和工程集成。

| 方案 | 额外依赖 | 与 PyTorch 原生集成 | 初始化/使用体验 | 适合场景 |
| --- | --- | --- | --- | --- |
| FSDP | 无额外核心依赖 | 最强 | 原生 API，`auto_wrap` 直接用 | 希望走 PyTorch 官方路径 |
| DeepSpeed ZeRO-3 | 需要 DeepSpeed 引擎 | 中等 | 功能全，但引擎配置更多 | 超大模型、已有 DeepSpeed 栈 |
| FairScale ZeRO-3 / FSDP | 需要 FairScale | 较弱 | 更偏研究与历史兼容 | 老项目迁移、研究验证 |

为什么很多团队优先选 FSDP？因为它与 PyTorch 原生 API 紧耦合，调试链路、checkpoint、混合精度、分布式基础设施更统一，不必再引入一层额外训练引擎。对纯 PyTorch 项目，这通常意味着更低的系统复杂度。

但 FSDP 也有明确适用边界。它非常依赖高带宽通信，因为每个活跃模块都要 all-gather 参数，反向还会有 reduce-scatter 或类似同步。单机 8 卡 NVLink 环境通常比较友好；跨节点低带宽环境就可能被通信拖垮。如果模型本来就能放进单卡，或者只比单卡稍大一点，那么半精度 DDP、ZeRO-2、梯度累积、activation checkpointing 这些组合往往更划算。

可以用一句判断标准来选：

- “模型装不下”是第一问题，优先看 FSDP / ZeRO-3。
- “模型装得下但吞吐不够”是第一问题，优先先看 DDP、混合精度、编译和数据流水。
- “网络差”是硬约束，FSDP 要谨慎。

新手可以记住这个简化版对比：

| 问题场景 | 更合适的方案 |
| --- | --- |
| 单卡放不下，且多卡带宽高 | FSDP |
| 单卡勉强能放下，想保持实现简单 | DDP + bf16/fp16 |
| 已有 DeepSpeed 训练栈 | DeepSpeed ZeRO-3 |
| 需要极致显存节省且接受更复杂通信 | FSDP + checkpointing + CPU offload |

---

## 参考资料

1. [PyTorch FSDP 官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)  
   目标读者：文档/工程  
   说明：API 定义最完整，明确指出 `FullyShardedDataParallel` 是对参数做跨 worker 分片的包装器，并提供 `CPUOffload`、`MixedPrecision`、`StateDictType` 等核心接口。

2. [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://docs.pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)  
   目标读者：工程  
   说明：解释 FSDP 与 ZeRO-3 的关系、嵌套包裹的工作方式，以及 GPT-175B、1T 模型的公开基准结果。

3. [Advanced Model Training with Fully Sharded Data Parallel (FSDP)](https://docs.pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)  
   目标读者：进阶  
   说明：更关注混合精度、device 设置、真实训练样例，是从 API 走向工程实践的过渡材料。

4. [Hugging Face Accelerate FSDP 指南](https://huggingface.co/docs/accelerate/usage_guides/fsdp)  
   目标读者：工程  
   说明：给出 `FULL_SHARD` 与 ZeRO Stage-3 的映射，并明确记录多模型优化器顺序、`predict_with_generate` 兼容性等常见坑。

建议新手阅读顺序：先看官方文档把概念和 API 对齐，再看 PyTorch 博客理解整体设计，再看高级教程和 Accelerate 指南补工程细节。
