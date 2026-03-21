## 核心结论

张量并行推理的本质，是把同一层里的大矩阵乘法横向拆给多张 GPU 同时做，而不是把不同层分给不同卡。这里的“张量并行”可以先理解成“把一层内部的参数矩阵切片后并行计算”。

在 Transformer 里，最适合这样拆的是 Attention 和 FFN。Attention 中，$Q/K/V$ 投影通常按列分片，也就是每张卡负责一部分输出维度；如果模型是多头注意力，这基本等价于“每张卡负责若干个 attention head”。FFN 中，第一层线性通常也是列分片，第二层线性通常是行分片。列分片后的局部计算可以独立完成，行分片后的局部结果必须做一次 all-reduce 才能恢复成完整隐状态。

因此，推理阶段 TP 的收益和代价都很直接：

| 维度 | 结果 |
|---|---|
| 计算量 | 每层 FLOPs 近似按卡数线性分摊 |
| 显存 | 每卡只保存部分权重，显存压力下降 |
| 延迟 | 会增加跨卡同步，尤其是 all-reduce |
| 吞吐 | 单机高速互联下通常有收益，跨节点容易被通信吃掉 |

Megatron-LM 风格 TP 的关键规律可以写成：
$$
O=\sum_{i=1}^{P} O_i
$$
其中 $P$ 是张量并行大小，$O_i$ 是第 $i$ 张卡算出的部分输出。这个求和通常由 all-reduce 完成。结论不是“多卡一定更快”，而是“多卡把算力问题变成了通信问题”。

一个真实工程量级的例子是 Qwen-14B。按公开资料，它有 40 层、40 个 attention heads。若采用 Megatron 风格 TP，那么每层 attention 后 1 次同步、FFN 后 1 次同步，总共约 $40\times 2+1=81$ 次通信。最后那个 `+1` 一般对应词表输出或末端聚合。结论很明确：TP 能让模型装得下、算得动，但你必须为节点内总带宽留预算。

---

## 问题定义与边界

这篇文章讨论的是推理阶段的张量并行，不讨论训练阶段的梯度同步，也不讨论 MoE 的专家并行。目标问题是：当单卡显存放不下模型，或者单卡内存带宽不够时，如何把同一层拆到多卡上执行，并保证结果与单卡数学上等价。

这里有两个硬约束。

第一，attention head 数必须能被 TP 大小整除。因为最常见做法是按 head 平均分卡。设 head 数为 $H$，TP 大小为 $P$，则必须满足：
$$
H \bmod P = 0
$$

第二，FFN 的中间隐藏维也必须能被 TP 大小整除。FFN 可以先理解成“两层大矩阵加一个激活函数”的模块，第一层把维度升高，第二层再降回去。若升高后的维度不能均分，每张卡拿到的切片大小就不一致，框架通常直接拒绝初始化。

可行性判断可以写成：

| 检查项 | 约束 |
|---|---|
| Attention 头数 | $H \bmod P = 0$ |
| FFN 中间维 | $d_{ff} \bmod P = 0$ |
| 单卡部署拓扑 | 优先单节点 NVLink，次选单节点 PCIe |
| 多节点部署 | 通常让 TP 限制在节点内，跨节点用 PP |

一个常见报错能直接说明这个边界。Qwen-14B 的 head 数是 40。如果你设置 `tp_size=3`，就会触发类似：
`Total number of attention heads (40) must be divisible by tensor parallel size (3)`

这不是实现细节，而是数学分片方式本身不成立。

所以，TP 不是“想开几卡就开几卡”。它首先受模型结构约束，其次受硬件互联约束。理论可行，不代表工程划算；工程能跑，也不代表延迟更低。

---

## 核心机制与推导

先看注意力的基础公式：
$$
Attention(Q,K,V)=\text{softmax}(QK^\top)V
$$

这里的 $Q/K/V$ 分别是 query、key、value，可以先理解成“输入经过三组不同线性变换后得到的三个表示”。在多头注意力里，每个 head 都独立执行这套计算，所以 head 天然适合并行。

### 玩具例子：两卡、四个 heads

假设某层有 4 个 heads，TP 大小 $P=2$。那每张卡负责 2 个 heads。

1. 输入隐状态 $X\in \mathbb{R}^{b\times d}$ 广播到两张卡。
2. 每张卡持有自己那部分列分片的 $W_Q,W_K,W_V$。
3. 卡 0 算 head 0,1 的 $Q_0,K_0,V_0$；卡 1 算 head 2,3 的 $Q_1,K_1,V_1$。
4. 两张卡各自独立完成 softmax 注意力，不需要通信。
5. 得到局部输出后，进入 output projection，也就是把多头输出投回原始隐空间。
6. 这一步通常按行分片，于是每张卡只能得到“对完整输出的一部分贡献”，记为 $O_0,O_1$。
7. 最后执行一次 all-reduce，得到：
$$
O=O_0+O_1
$$

关键点在第 4 步和第 6 步。head 内计算是独立的，所以不通信；投回完整隐空间时，各卡结果是“部分和”，所以必须同步。

同样的逻辑也适用于 FFN。设第一层是 $Y=\phi(XW_1)$，第二层是 $Z=YW_2$，其中 $\phi$ 是激活函数，可以先理解成“逐元素非线性变换”。Megatron 风格通常这样拆：

| 模块 | 分片方式 | 本地可独立完成 | 是否需要 all-reduce |
|---|---|---|---|
| Q/K/V 投影 | 列分片 | 是 | 否 |
| Attention 计算 | 按 head 本地计算 | 是 | 否 |
| Output Projection | 行分片 | 部分可做 | 是 |
| FFN 第一层 | 列分片 | 是 | 否 |
| FFN 第二层 | 行分片 | 部分可做 | 是 |

所以一层 Transformer 里，Megatron 风格 TP 的通信点通常就是两个：attention 输出投影后一次，FFN 第二层后一次。

从代数上看，这样拆之所以成立，是因为矩阵乘法对列切分和行切分都保持线性。列分片对应“把输出维拆开独立算”，行分片对应“各卡各算一部分，再把部分和加起来”。这就是为什么最终通信模式常落到 all-reduce 上。

---

## 代码实现

先给一个最小可运行玩具代码，用 `numpy` 验证“行分片后 all-reduce 求和”和“完整矩阵一次性计算”数学等价。

```python
import numpy as np

def row_parallel_linear(x, w, tp_size):
    # x: [batch, in_dim]
    # w: [in_dim, out_dim]
    in_dim = w.shape[0]
    assert in_dim % tp_size == 0

    shard = in_dim // tp_size
    partials = []
    for i in range(tp_size):
        x_i = x[:, i * shard:(i + 1) * shard]
        w_i = w[i * shard:(i + 1) * shard, :]
        partials.append(x_i @ w_i)

    # all-reduce(sum)
    return sum(partials)

def column_parallel_linear(x, w, tp_size):
    # 列分片：每张卡得到部分输出，拼接后等于完整结果
    out_dim = w.shape[1]
    assert out_dim % tp_size == 0

    shard = out_dim // tp_size
    outputs = []
    for i in range(tp_size):
        w_i = w[:, i * shard:(i + 1) * shard]
        outputs.append(x @ w_i)
    return np.concatenate(outputs, axis=1)

np.random.seed(0)
x = np.random.randn(2, 4)
w_row = np.random.randn(4, 6)
w_col = np.random.randn(4, 6)

full_row = x @ w_row
tp_row = row_parallel_linear(x, w_row, tp_size=2)
assert np.allclose(full_row, tp_row, atol=1e-6)

full_col = x @ w_col
tp_col = column_parallel_linear(x, w_col, tp_size=2)
assert np.allclose(full_col, tp_col, atol=1e-6)

print("tensor parallel toy example passed")
```

这段代码对应两个事实：

1. 列分片时，各卡算出的输出直接拼接就是完整结果。
2. 行分片时，各卡只得到部分贡献，必须求和才能恢复完整结果。

真实框架里，不会手写这些切片，而是让库帮你安排进程组和通信。Hugging Face `transformers` 已经暴露了一个对新手比较友好的入口：

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    tp_plan="auto",
    torch_dtype=torch.bfloat16,
)
inputs = torch.tensor([[1, 2, 3, 4]], device="cuda")
outputs = model(inputs)
print(outputs.logits.shape)
```

对应启动方式是：

```bash
torchrun --nproc_per_node 4 demo_tp.py
```

这里有三个工程含义。

第一，`torchrun` 会起 4 个进程，通常一进程对应一张 GPU。  
第二，`tp_plan="auto"` 会让框架基于模型结构自动选择哪些层做列分、哪些层做行分。  
第三，真正的同步仍然依赖底层通信库，比如 NCCL。`tp_plan` 只是定义“该怎么切”，不负责消除通信成本。

### 真实工程例子：vLLM 部署 Qwen-14B

在 vLLM 中常见参数是 `--tensor-parallel-size` 或配置项 `tensor_parallel_size`。如果部署 Qwen-14B 且设置 `tp_size=4`，那么 40 个 heads 会被分成每卡 10 个，结构上可行。此时每层仍然存在 attention 和 FFN 的同步点，整次推理要经历大量小而频繁的 all-reduce。短序列、低 batch、低带宽互联时，通信开销可能比你预想得更早成为瓶颈。

---

## 工程权衡与常见坑

TP 最容易被误解的一点，是“显存省了，所以一定更快”。实际情况是：显存压力下降，计算并行度上升，但每层都插入了同步屏障。推理尤其怕这种同步，因为单次请求更看重首 token 延迟而不是总吞吐。

常见问题可以直接整理成表：

| 问题 | 现象 | 原因 | 缓解方式 |
|---|---|---|---|
| head 数不可整除 | 初始化直接报错 | 无法均分 attention heads | 只选择 $H$ 的因子作为 TP 大小 |
| FFN 维不可整除 | 模型加载失败或 shape 错误 | 中间维不能均分 | 检查配置里的 `intermediate_size` |
| 单机无高速互联 | 延迟收益很差 | all-reduce 被 PCIe 拖慢 | 优先 NVLink 机器 |
| 跨节点直接开 TP | 首 token 延迟陡增 | 节点间带宽远低于节点内 | TP 限制在节点内，跨节点用 PP |
| NCCL 版本过旧 | 启动失败、hang、非法访存 | 通信库兼容性问题 | 升级到已修复版本 |
| 小 batch 场景收益差 | GPU 利用率不高 | 通信占比高于计算 | 结合 batching、KV cache 优化 |

有一个最近几年非常典型的坑：RTX 5090 / 部分 60xx 系列在旧版 NCCL 上跑 TP，会出现 `illegal memory access` 或 `Custom allreduce is disabled` 一类报错。公开社区讨论指向的结论是，旧版本 NCCL 在这些卡上的 P2P 路径有已知问题，升级到 2.26.5+，更稳妥是 2.27.3 或 2.27.7，才更有机会稳定跑起 vLLM 的 TP。这里要注意，问题表面像“vLLM 不支持”，实质常常是“底层通信栈不稳定”。

Qwen-14B 的 81 次 all-reduce 也说明了另一个事实：TP 并不是只在“大矩阵”阶段通信一次，而是每层都重复发生。层数一高，通信次数会非常可观。你能否从 TP 中获益，往往不由参数量决定，而由“每次同步的代价”决定。

---

## 替代方案与适用边界

如果通信已经成为主瓶颈，替代思路不是简单“再加卡”，而是换并行方式。

最常见替代是 Pipeline Parallelism，简称 PP，可以先理解成“按层切模型，每张卡或每组卡负责连续的一段层”。它与 TP 的核心差别在于：TP 在一层内部频繁同步，PP 在层与层之间传递激活。对跨节点场景，PP 通常更友好，因为同步次数更少。

两者在推理场景的区别可以概括为：

| 方案 | 怎么切 | 延迟特点 | 吞吐特点 | 适合场景 |
|---|---|---|---|---|
| Tensor Parallelism | 切层内矩阵 | 单层更快，但同步多 | 单机高速互联下较好 | 单节点、多 GPU、高带宽 |
| Pipeline Parallelism | 切连续层 | 通信次数少，但流水有气泡 | 大模型跨节点更实用 | 多节点、超大模型 |
| TP + PP | 节点内 TP，节点间 PP | 折中 | 实战最常见 | 集群部署 |

Berkeley 的 PartitionSearch 报告还给出了 Megatron 之外的两类替代策略：Projection-Replicated 和 Weight-Gathered。可以把它们理解成“用更多 FLOPs 或更多局部内存，换更少通信”。

| 策略 | 主要思路 | 优点 | 代价 | 适用边界 |
|---|---|---|---|---|
| Megatron | 标准列分/行分 + all-reduce | 计算最规整，实现成熟 | 通信随输入长度增长 | 单机高速互联、常规场景 |
| Projection-Replicated | 复制部分投影权重，减少同步 | 通信更低 | 额外 FLOPs 和内存 | 通信先成为瓶颈时 |
| Weight-Gathered | 需要时聚合权重，避免某些激活同步 | 长序列下通信更稳 | 权重 gather 很重 | 超长上下文、低带宽场景 |

特别是 Weight-Gathered MLP，它的通信量更多与模型权重大小相关，而不随输入长度线性增长。这意味着当序列很长时，它可能比标准 Megatron 更划算。代价是实现复杂、权重调度重，而且对显存和带宽的要求模式完全不同。

所以，选型不要只问“TP 还是 PP”，而要问两个更精确的问题：

1. 你的瓶颈是显存、算力还是通信？
2. 你的通信发生在节点内还是节点间？

如果答案是“单节点、NVLink、模型太大但还放得下节点内”，TP 往往是第一选择。  
如果答案是“多节点、长序列、首 token 延迟敏感”，那就应该优先考虑 PP，或者 TP 只放在节点内。

---

## 参考资料

- Hugging Face, “Tensor Parallelism (TP) in Transformers: 5 Minutes to Understand”, 2025-12-04: https://huggingface.co/blog/qgallouedec/tp
- Isaac Ong, “Efficient Distributed LLM Inference with Dynamic Partitioning”, UC Berkeley Technical Report No. UCB/EECS-2024-108, 2024-05-16: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-108.pdf
- Alibaba Cloud Community, “Analyzing the Distributed Inference Process Using vLLM and Ray from the Perspective of Source Code”: https://www.alibabacloud.com/blog/601427
- Red Hat Developer, “Distributed inference with vLLM”, 2025-02-06，最后更新 2025-03-26: https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm
- vLLM Forums, “vLLM does not work with 2x 5090 in tp 2”, 2025-09-18: https://discuss.vllm.ai/t/vllm-does-not-work-with-2x-5090-in-tp-2/1630
- vLLM Forums, “Added second 5090 and turne on tensor parallel 2”, 2025-09-18: https://discuss.vllm.ai/t/added-second-5090-and-turne-on-tensor-parallel-2/1629
