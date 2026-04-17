## 核心结论

分布式训练先要回答一个基本问题：显存到底被谁占满了。对大语言模型训练，最核心的四项是参数、梯度、优化器状态、激活值。

若参数量记为 $\Phi$，采用混合精度训练、优化器为 Adam，则常见近似是：

$$
M_{\text{param}} = 2\Phi,\quad
M_{\text{grad}} = 2\Phi,\quad
M_{\text{opt}} = 12\Phi
$$

因此模型状态总内存为：

$$
M_{\text{state}} = M_{\text{param}} + M_{\text{grad}} + M_{\text{opt}} \approx 16\Phi
$$

这里的“模型状态”可以理解为“训练时必须长期保留的账本”，不是只看权重文件大小。175B 模型代入后，单是参数、梯度、优化器状态就约为：

$$
16 \times 175\text{B} \approx 2.8\text{TB}
$$

这还没有算激活值。激活值是前向传播中间结果，白话说就是“为了反向传播临时记住的每层输出”，它通常与 `batch × seq × hidden × layers` 成正比，很多时候比模型状态更早触发 OOM。

因此，大模型训练的内存问题不是“多加几张卡”就结束，而是要系统拆成三类分摊：

| 机制 | 解决什么问题 | 核心效果 |
|---|---|---|
| ZeRO-3 | 数据并行下模型状态重复存放 | 参数、梯度、优化器状态按 DP 分片 |
| TP（Tensor Parallel） | 单层矩阵太大 | 单层权重与部分激活按张量维度切分 |
| PP（Pipeline Parallel） | 全部层放不下 | 按层切段，每张卡只放一部分网络 |

最小玩具例子是：175B 模型若直接在每卡完整保留状态，训练根本无法开始；若采用 ZeRO-3 且 `DP=8`，则每卡模型状态近似降为 $2.8\text{TB}/8 \approx 350\text{GB}$，再叠加 `TP=8`、`PP=4` 对权重与激活进一步切分，单卡需求才可能压到几十 GB 量级。

---

## 问题定义与边界

这篇文章只讨论训练阶段的显存分析，不讨论推理阶段 KV Cache，也不讨论 CPU offload、NVMe offload 这类“把压力转移到别处”的方案。

训练显存可以粗分为四类：

| 内存项 | 是什么 | 典型量级 | 主要依赖 |
|---|---|---|---|
| 参数 | 模型权重，白话说就是“要学习的数值” | $2\Phi$ | 参数量、精度 |
| 梯度 | 每个参数的更新方向 | $2\Phi$ | 参数量、精度 |
| 优化器状态 | Adam 的一阶矩、二阶矩、主权重副本 | $12\Phi$ | 参数量、优化器类型 |
| 激活值 | 反向传播要回看的中间结果 | 常常极大 | batch、seq、hidden、layers、checkpoint 策略 |

激活值常见近似可以写成：

$$
M_{\text{act}} \approx B \times S \times H \times L \times \text{coeff}
$$

其中：

- $B$ 是 micro-batch size，也就是单次进卡的小批量
- $S$ 是序列长度
- $H$ 是 hidden size，即每个 token 的特征维度
- $L$ 是层数
- `coeff` 是经验系数，取决于 attention、MLP、并行方式和是否做 activation checkpoint

一个容易误判的地方是：很多人先看模型文件大小，比如“7B 模型 fp16 只有十几 GB”，于是误以为训练只需要二十几 GB。这个判断漏掉了梯度、优化器状态和激活值。

玩具例子如下。假设只有 1B 参数，Adam 混合精度训练：

- 参数：约 `2GB`
- 梯度：约 `2GB`
- 优化器状态：约 `12GB`
- 模型状态合计：约 `16GB`

如果此时 `batch=4, seq=2048, hidden=4096, layers=32`，激活值可能远高于 16GB。也就是说，“模型不大”不等于“训练不爆显存”。

真实工程例子是 175B 模型。此时模型状态已经是 2.8TB 量级，任何单卡 H100、A100 都不可能完整容纳；如果序列长度再上升到 4k、8k，激活压力会继续抬升。于是问题边界非常明确：你要同时处理“长期状态太大”和“瞬时激活峰值太高”两个问题。

---

## 核心机制与推导

先统一一个最常用的训练近似模型。

### 1. 模型状态为什么是 $16\Phi$

若参数本体用 fp16/bf16 存储，每参数 2 字节，则参数内存约为：

$$
M_{\text{param}} = 2\Phi
$$

梯度通常也按 2 字节近似：

$$
M_{\text{grad}} = 2\Phi
$$

Adam 优化器常见要保留三类额外状态：

- fp32 master weights，白话说就是“更高精度的主副本”：$4\Phi$
- 一阶矩 $m$：$4\Phi$
- 二阶矩 $v$：$4\Phi$

因此：

$$
M_{\text{opt}} = 12\Phi
$$

总和即：

$$
M_{\text{state}} = 2\Phi + 2\Phi + 12\Phi = 16\Phi
$$

这就是很多资料里反复出现的 `16Φ` 公式。

### 2. ZeRO 为什么能显著降内存

ZeRO 的本质是“去冗余”。传统数据并行里，每张卡都存一整份参数、梯度、优化器状态，计算简单，但非常浪费。ZeRO 将这些状态按数据并行组切片。

- ZeRO-1：只切优化器状态
- ZeRO-2：切优化器状态和梯度
- ZeRO-3：连参数也切

若使用 ZeRO-3，且数据并行规模为 $D$，则单卡模型状态近似降为：

$$
M_{\text{state, per-gpu}} \approx \frac{16\Phi}{D}
$$

白话说，原来每个人都背整本账本，现在每个人只背其中一页。

但 ZeRO-3 不是免费午餐。某层要计算时，本卡没有完整参数，就要先 all-gather 把该层参数临时拼起来；梯度回传后还要 reduce-scatter。于是显存省下来了，通信压力上来了。

### 3. TP 和 PP 为什么还需要叠加

只靠 ZeRO-3，解决的是“数据并行副本重复”问题，但单层矩阵可能依然过大。

TP（张量并行）把单层内部的矩阵乘法拆开。白话说，不是一张卡算完整线性层，而是多张卡一起算同一层的不同切片。这样参数和对应激活会沿隐藏维或输出维分摊。

PP（流水并行）把模型层切成多个 stage。白话说，一张卡只负责若干层，而不是整个网络。这样可以进一步降低每卡持有的层数。

因此，大模型常见组合是：

$$
\text{单卡状态} \approx \frac{16\Phi}{DP \times TP \times PP_{\text{layer-share}}}
$$

严格说，TP 与 PP 对不同内存项的影响不完全一样，不能简单统一除法，但“它们共同分摊模型与激活”这个方向是正确的。

### 4. 激活为什么常常比状态更难处理

激活不是长期持久状态，而是训练过程中的“峰值负载”。很多层的输出为了反向传播必须保留到很后面，因此峰值显存通常出现在前向末尾或反向早期。

activation checkpoint 的作用是“少存，多算”。白话说，不把所有中间结果都留下来，只保留关键节点，反向时再重算一遍前向。这样显存下降，计算时间上升。

gradient accumulation 的作用是“拆大 batch”。白话说，不一次把大批样本全塞进卡里，而是分多次前后向累计梯度。这样吞吐可能下降，但显存峰值更可控。

一个简化对比如下：

| 策略 | 模型状态 | 激活峰值 | 通信 | 计算量 |
|---|---|---|---|---|
| 纯 DP | 高 | 高 | 中 | 低 |
| ZeRO-3 | 低 | 高 | 高 | 中 |
| ZeRO-3 + TP | 更低 | 更低 | 更高 | 中 |
| ZeRO-3 + TP + PP | 更低 | 更低 | 高且复杂 | 中到高 |
| 再加 checkpoint | 不变 | 明显下降 | 不变 | 上升 |

以 175B 为例，若 `DP=8, TP=8, PP=4`，可以这样理解：

- ZeRO-3 把模型状态按 8 份分给数据并行组
- TP 把单层内部大矩阵再切 8 份
- PP 把网络层数切 4 段
- checkpoint 继续压缩激活峰值

这不是某一个技巧单独解决问题，而是多个机制叠加才把系统拉回可训练区间。

---

## 代码实现

先给一个可运行的 Python 玩具脚本，用来估算模型状态和激活量级。

```python
from math import ceil

GB = 1024 ** 3
TB = 1024 ** 4

def model_state_bytes(params_billion: float) -> float:
    phi = params_billion * 1e9
    return 16 * phi  # 16 bytes per parameter in mixed precision Adam training

def activation_bytes(batch: int, seq: int, hidden: int, layers: int, coeff: float) -> float:
    return batch * seq * hidden * layers * coeff

def per_gpu_state_bytes(params_billion: float, dp: int) -> float:
    return model_state_bytes(params_billion) / dp

state_1b = model_state_bytes(1)
assert round(state_1b / GB) == 15  # about 16 GB in binary units, rounded down to 15.x

state_175b = model_state_bytes(175)
assert round(state_175b / TB, 1) == 2.5  # about 2.55 TiB, commonly written as ~2.8 TB decimal

per_gpu_175b_dp8 = per_gpu_state_bytes(175, 8)
assert per_gpu_175b_dp8 / GB > 300

# toy activation estimate
act = activation_bytes(batch=4, seq=2048, hidden=4096, layers=32, coeff=13)
assert act / GB > 12

print(f"1B model state: {state_1b / GB:.2f} GiB")
print(f"175B model state: {state_175b / TB:.2f} TiB")
print(f"175B with ZeRO-3 DP=8 per GPU state: {per_gpu_175b_dp8 / GB:.2f} GiB")
print(f"Toy activation estimate: {act / GB:.2f} GiB")
```

上面故意把“十进制 TB”和“二进制 TiB”区分开。工程里文档常写 `2.8TB`，但脚本按二进制单位算会更接近 `2.55TiB`。两者不是矛盾，而是单位口径不同。

下面是一个 DeepSpeed 配置片段，展示 ZeRO-3、梯度累积和混合精度的关键字段：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "allgather_partitions": true
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  },
  "pipeline": {
    "pipeline_parallel_size": 4
  },
  "tensor_parallel": {
    "tp_size": 8
  }
}
```

关键字段可以直接理解为：

| 字段 | 作用 | 为什么重要 |
|---|---|---|
| `zero_optimization.stage=3` | 参数/梯度/优化器全分片 | 解决模型状态冗余 |
| `gradient_accumulation_steps` | 多步累计梯度 | 用时间换显存 |
| `activation_checkpointing` | 重算中间层 | 压低激活峰值 |
| `pipeline_parallel_size` | 按层切网络 | 单卡不再存全模型 |
| `tp_size` | 按张量维度切层 | 单层矩阵和激活进一步缩小 |

训练循环里的思路通常是这样：

```python
# pseudo code
for global_step in range(max_steps):
    optimizer.zero_grad()

    for micro_step in range(grad_accum_steps):
        with activation_checkpointing():
            loss = model(input_batch[micro_step])
        engine.backward(loss)

    engine.step()
```

真实工程例子是：在 8 卡节点上训练 10B 级模型，如果关闭 activation checkpoint，前向后半段可能直接因激活峰值 OOM；打开后虽然反向更慢，但训练能稳定跑起来。这类改动不是“优化一点点”，而是“从不能跑到能跑”。

---

## 工程权衡与常见坑

分布式训练的难点不在公式，而在“公式之外还有很多临时内存和通信副作用”。

常见坑如下：

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 只算 `16Φ` 不算激活 | 纸面能跑，实际 OOM | 激活峰值更早爆 | 先估激活，再定 batch 与 seq |
| 忽略临时缓冲区 | 显存使用高于理论值 | kernel workspace、通信 buffer 额外占用 | 预留 10% 到 30% 安全边界 |
| ZeRO-3 速度变慢 | GPU 利用率低 | all-gather / reduce-scatter 太频繁 | 提升带宽、调 bucket、减少碎片化 |
| TP 太大 | 通信压过计算 | 层内每步都要同步 | 优先利用节点内高速互联 |
| PP 太深但 micro-batch 太少 | 流水线空泡大 | stage 无法被填满 | 保证 micro-batch 数不小于 stage 数 |
| checkpoint 开太多 | 训练极慢 | 重算比例过高 | 只 checkpoint 大块层，不要全图无差别重算 |

通信带宽可以用一个很粗的判断式理解：

$$
T_{\text{comm}} \approx \frac{\text{通信数据量}}{\text{有效带宽}}
$$

如果 `T_comm` 接近甚至超过 `T_compute`，那你虽然省了显存，但总吞吐会明显下滑。ZeRO-3 很依赖通信链路，NVLink、NVSwitch、IB 之类高速互联不是“锦上添花”，而是很多配置能不能成立的前提。

一个典型工程坑是：8 卡训练时，模型状态已经通过 ZeRO-3 压下来了，于是团队继续增大 sequence length；结果激活值迅速膨胀，显存再次爆掉。最后发现不是 ZeRO 失效，而是激活建模没更新。解决方式通常是同时调整：

- 降低 micro-batch size
- 提高 gradient accumulation
- 打开 activation checkpoint
- 重新评估 TP/PP 划分
- 监控每步峰值显存，而不是只看平均值

---

## 替代方案与适用边界

不是所有模型都需要 ZeRO-3 + TP + PP。并行方案应该按规模选，不应一开始就把系统复杂度拉满。

| 方案 | 显存收益 | 通信成本 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|
| 纯 DP | 低 | 中 | 低 | 小模型，单卡可放完整状态 |
| ZeRO-1 | 中 | 中 | 低到中 | 优化器状态开始成为瓶颈 |
| ZeRO-2 | 更高 | 中到高 | 中 | 梯度也明显占用显存 |
| ZeRO-3 | 很高 | 高 | 中到高 | 参数本体都放不下 |
| TP | 针对单层有效 | 高 | 高 | 单层矩阵太大，需节点内高速互联 |
| PP | 针对层数有效 | 中 | 高 | 模型很深，单卡放不下全部层 |
| Checkpoint | 压激活明显 | 无额外通信 | 低到中 | 激活峰值主导显存 |
| Grad Accum | 压 micro-batch 峰值 | 无额外通信 | 低 | 想保留全局 batch 但单步放不下 |

可以用一个简单判断法：

- 10B 以内，很多场景纯 DP 或 ZeRO-1/2 就够
- 10B 到数十 B，通常开始认真考虑 ZeRO-2、ZeRO-3 和 checkpoint
- 100B 以上，往往必须把 ZeRO-3、TP、PP 组合起来看
- 若互联带宽差，宁可降低并行切分复杂度，也不要盲目扩大 TP 或 ZeRO-3 范围

新手常问：为什么 10B 模型还能考虑 `DP + ZeRO-1`，175B 却基本必须 `ZeRO-3 + TP + PP`？原因不神秘，只有量级差异。10B 的模型状态约 `160GB`，在多卡下仍有一定腾挪空间；175B 的模型状态是 `2.8TB` 量级，已经远超“简单复制几份”的容忍区间。

所以，适用边界可以总结成一句话：小模型优先简单，大模型优先可行，超大模型再在可行解上做性能优化。

---

## 参考资料

- DeepSpeed ZeRO 论文解读  
  链接：https://deep-paper.org/en/paper/1910.02054/?utm_source=openai  
  用途：理解 `16Φ` 公式、ZeRO 分阶段设计、显存与通信的基本 trade-off。

- Harold Benoit 关于 ZeRO 的工程笔记  
  链接：https://haroldbenoit.com/notes/ml/engineering/training/parallelism/zero-redundancy-optimizer?utm_source=openai  
  用途：适合理清参数、梯度、优化器状态分别如何分布，适合作为训练内存的入门账本。

- Quentin Anthony 的训练内存估算笔记  
  链接：https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5?utm_source=openai  
  用途：适合理解激活估算、经验系数和工程上为什么“激活可能比模型状态更大”。

- Frontier/OLCF 相关训练报告  
  链接：https://www.osti.gov/servlets/purl/2438819?utm_source=openai  
  用途：查看大模型在真实超算环境中的内存、并行组合与系统约束，尤其适合理解 100B+ 规模下为什么必须联合使用 ZeRO、TP、PP。
