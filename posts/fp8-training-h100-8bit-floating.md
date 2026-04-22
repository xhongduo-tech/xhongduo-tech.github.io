## 核心结论

FP8 训练不是把模型里的所有数直接改成 8 位，而是在可控模块中执行：

$$
\text{FP8} = \text{scaling} + \text{quantization} + \text{Tensor Core execution}
$$

其中 `scaling` 是缩放：先把张量按比例压到 FP8 可表示范围内；`quantization` 是量化：把高精度数映射到低精度格式；`Tensor Core execution` 是在 H100 的 Hopper Tensor Cores 上执行矩阵乘等核心计算。

H100 支持两种常见 FP8 格式：`E4M3` 和 `E5M2`。`E4M3` 指 1 位符号位、4 位指数、3 位尾数，精度相对更好，动态范围较小；`E5M2` 指 1 位符号位、5 位指数、2 位尾数，动态范围更大，精度更粗。训练中常见策略是：前向的激活和权重使用 `E4M3`，反向的梯度使用 `E5M2`。

| 对象 | 常用格式 | 原因 |
|---|---:|---|
| 权重 | E4M3 | 数值分布通常较稳定，需要更高精度 |
| 激活 | E4M3 | 前向误差会直接影响 loss，需要更细粒度表示 |
| 梯度 | E5M2 | 反向传播中数值范围更宽，更需要防溢出 |
| 累积结果 | FP16/BF16/FP32 | 矩阵乘累积仍需要更高精度保证稳定性 |

和 BF16 相比，FP8 的价值来自三个方面：显存占用更低、带宽压力更小、H100 Tensor Core 的吞吐更高。新手版理解是：`1000` 这种数如果直接塞进 `E4M3`，可能超过范围；先乘缩放因子把最大值压到可表示范围，再转 FP8，训练才可能稳定。

| 方案 | 显存 | 带宽压力 | Tensor Core 吞吐 | 稳定性 |
|---|---:|---:|---:|---|
| BF16 | 较高 | 较高 | 较高 | 高 |
| FP8 | 更低 | 更低 | 更高 | 依赖缩放和实现 |
| INT8 | 更低 | 更低 | 高 | 多用于推理，不等同于 FP8 训练 |

真实工程例子是 NVIDIA Transformer Engine 教程中对 Hugging Face Llama 2/3 的改造：把 decoder layer 替换成 TE 的 `TransformerLayer`，在 H100 上比较 BF16 和 FP8 微调。Llama 2 7B 的 step time 从 BF16 的 248 ms 降到 FP8 的 160 ms，Llama 3 8B 从 270 ms 降到 185 ms。这个收益不是格式单独带来的，而是低带宽、更高吞吐和 TE fused 路径共同作用的结果。

---

## 问题定义与边界

这篇文章讨论的问题不是“FP8 是什么”，而是“为什么 H100 上的 FP8 能用于训练，以及它的边界在哪里”。

FP8 训练是指：在深度学习训练中，把部分张量转换成 8 位浮点格式参与计算，同时用张量级缩放因子控制数值范围。张量是多维数组，模型里的权重、激活、梯度都可以看作张量。

| 概念 | 白话定义 |
|---|---|
| FP8 训练 | 部分训练计算使用 8 位浮点格式，并配合缩放控制误差和溢出 |
| 不是 FP8 训练 | 把整个模型、所有算子、所有状态无脑强制改成 8 位 |
| `fp8_autocast` | TE 提供的上下文管理器，只在上下文内对支持的路径启用 FP8 |
| `amax` | 张量中绝对值最大的元素，即 `max(abs(x))` |
| `scale` | 缩放因子，用来把张量压到 FP8 可表示范围 |

新手版反例：如果把全模型所有算子都切到 FP8，LayerNorm、softmax、自定义 kernel 或数值敏感路径可能出现明显误差，训练 loss 会抖动甚至发散。正确边界是：只把 Transformer Engine 标记为 FP8-safe 的模块放进 `fp8_autocast`，其他路径继续使用 BF16、FP16 或 FP32。

适合 FP8 的路径包括 Transformer 的线性层、attention 相关矩阵乘、部分 MLP 路径。不适合的路径包括未验证的自定义算子、需要高精度累积的敏感路径、数值范围变化剧烈且没有可靠缩放策略的模块。

---

## 核心机制与推导

FP8 的关键机制是“按张量缩放 + 动态选择 scale + 延迟更新历史 amax”，不是单纯量化。

核心公式可以写成：

```text
a = max(|x|)
s = 2^{-m} * F / a
x_fp8 = clip(round(x * s))
x_hat = x_fp8 / s
```

其中 `x` 是原始张量，`a` 是当前张量的最大绝对值，`m` 是 margin，用来给最大值留安全余量，`F` 是 FP8 格式的最大可表示绝对值。常见情况下，`E4M3` 的 `F` 可取 448，`E5M2` 的 `F` 可取 57344。`clip` 是截断，超过范围的值会被限制在最大或最小可表示值；`round` 是舍入，把连续实数映射到离散 FP8 值。

玩具例子：设 `x = [1000, 250]`，使用 `E4M3`，`m = 0`。则：

$$
a = 1000,\quad s = 448 / 1000 = 0.448
$$

缩放后：

```text
x * s = [448, 112]
```

最大值刚好落在 `E4M3` 范围内。反量化时：

```text
x_hat = x_fp8 / 0.448
```

如果不缩放，`1000` 可能直接溢出；如果缩放太激进，小数值又会损失太多精度。因此 scale 的选择决定了 FP8 训练的稳定性。

| 格式 | 指数位 | 尾数位 | 特点 | 更适合 |
|---|---:|---:|---|---|
| E4M3 | 4 | 3 | 精度相对更好，范围较小 | 前向激活、权重 |
| E5M2 | 5 | 2 | 范围更大，精度更粗 | 反向梯度 |

Transformer Engine 的 `DelayedScaling` 不是每一步只看当前 `amax`。它会保存历史 `amax`，用历史统计估计下一步的 scale。这样做的原因是当前 batch 的数值可能偶然变大或变小，如果 scale 每一步剧烈变化，量化误差也会抖动。延迟缩放用历史窗口平滑 scale，让训练更稳定。

---

## 代码实现

代码层面的重点不是“把全局精度改成 FP8”，而是“只把可控模块包进 TE 的 FP8 上下文”。典型训练路径是：替换模块、打开 `fp8_autocast`、让 TE 在训练过程中维护 `amax` 和 scale 状态。

模块替换示意：

| 原始模块 | TE 替换方向 | 目的 |
|---|---|---|
| `torch.nn.Linear` | `transformer_engine.pytorch.Linear` | 让线性层进入 FP8 matmul 路径 |
| Hugging Face decoder layer | TE `TransformerLayer` | 复用 TE 的 fused Transformer 实现 |
| 普通 attention/MLP block | TE 支持的对应模块 | 降低手写 FP8 逻辑风险 |

新手版伪代码如下：

```python
# 伪代码：需要安装 NVIDIA Transformer Engine，并在 H100 等支持硬件上运行
from transformer_engine.pytorch import fp8_autocast

for batch in dataloader:
    optimizer.zero_grad()

    with fp8_autocast(enabled=True):
        loss = model(batch["input_ids"], labels=batch["labels"]).loss

    loss.backward()
    optimizer.step()
```

训练流程可以概括为：

```text
forward
  -> amax 收集
  -> scale 更新
  -> FP8 cast
  -> FP8 Tensor Core matmul
  -> 高精度累积
  -> backward
  -> 梯度路径 amax reduction
```

下面是一个可运行的 Python 玩具实现，用普通数值模拟“缩放、量化、反缩放”的过程。它不实现真实 `E4M3` 编码，只演示 FP8 训练中最重要的 scale 思想。

```python
def fake_fp8_quantize(x, fp8_max=448.0, margin=0):
    amax = max(abs(v) for v in x)
    assert amax > 0

    scale = (2 ** (-margin)) * fp8_max / amax

    q = []
    for v in x:
        scaled = round(v * scale)
        clipped = max(-fp8_max, min(fp8_max, scaled))
        q.append(clipped)

    restored = [v / scale for v in q]
    return scale, q, restored


x = [1000.0, 250.0]
scale, q, restored = fake_fp8_quantize(x)

assert abs(scale - 0.448) < 1e-9
assert q == [448.0, 112.0]
assert abs(restored[0] - 1000.0) < 1e-9
assert abs(restored[1] - 250.0) < 1e-9
```

真实工程实现中，不应该自己手写这种量化逻辑去替换训练框架。更常见的做法是：在 Hugging Face Llama 的 decoder layer 上替换 TE 实现，用 BF16 和 FP8 两组配置分别跑相同 batch、相同序列长度、相同优化器，再比较 step time、显存峰值和 loss 曲线。

---

## 工程权衡与常见坑

FP8 的收益通常不是“数值更小”本身，而是系统级收益：权重和激活占用更少显存，GPU 读写带宽压力下降，H100 Tensor Cores 对 FP8 matmul 有更高吞吐，TE 又把 cast、scale、matmul 等路径做了融合。

主要风险来自范围管理和分布式一致性。新手常见错误是照搬 FP16 的单一 loss scaling。loss scaling 是对 loss 或梯度整体缩放，主要解决 FP16 下梯度下溢；FP8 训练需要按张量维护独立 scale，因为不同层、不同张量的数值范围完全不同。

| 坑点 | 后果 | 修正思路 |
|---|---|---|
| 所有算子强制 FP8 | loss 发散或精度明显下降 | 只使用 TE-safe 模块 |
| 用单一 loss scaling 代替张量级 scale | 层间数值范围无法被正确处理 | 每个 FP8 张量维护独立 `amax/scale` |
| 误以为纯 E5M2 可训练 | 精度太粗，TE 常见训练不支持纯 E5M2 | 使用 `HYBRID` 配置 |
| 分布式不做 `amax reduction` | 不同 GPU scale 不一致，结果漂移 | 在并行组内同步 `amax` |
| Linear 形状不满足要求 | 不能进入高效 FP8 kernel | 对维度做 padding 或回退高精度路径 |

错误示例：

```python
# 错误思路：把整个模型所有参数直接转成某种 8 位格式
# model = force_everything_to_fp8(model)
```

正确思路是：保持优化器状态、归一化、敏感算子等必要路径为高精度，只把 TE 支持并验证过的 Transformer 线性层、attention、MLP 路径放进 `fp8_autocast`。FP8 是受控混合精度训练，不是全模型低精度替换。

---

## 替代方案与适用边界

FP8 不是唯一低精度训练方案。它的优势集中在 Hopper 架构，尤其是 H100 上的大模型 Transformer 训练和微调。如果硬件、模型结构或数值稳定性不满足条件，BF16 仍然是更稳妥的选择。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| BF16 | 动态范围大，稳定，通用 | 显存和带宽开销高于 FP8 | 大多数训练默认选择 |
| FP16 | 吞吐较高，生态成熟 | 动态范围较小，常需 loss scaling | 较稳定的传统混合精度训练 |
| FP8 | 显存低，带宽低，H100 吞吐高 | 依赖 scale、TE 和算子支持 | Hopper + Transformer |
| INT8 | 推理效率高 | 通常不用于标准训练反向传播 | 推理量化、部署优化 |

新手版边界：如果 GPU 不支持 Hopper Tensor Core，FP8 的收益和可用性都会明显下降。即使软件层面能模拟 FP8，也不等于训练会更快。

工程版边界：如果模型大量依赖不规则自定义算子、数值敏感计算、非 Transformer 结构，或者你的训练目标对微小数值误差非常敏感，BF16 可能比 FP8 更合适。FP8 适合 H100、Transformer、大模型训练和微调；不适合非 Hopper 硬件、强数值敏感任务、未验证自定义算子密集模型。

---

## 参考资料

| 主题 | 资料 | 用途 |
|---|---|---|
| 格式与机制 | FP8 Formats for Deep Learning | 解释 `E4M3/E5M2` |
| 实现细节 | Transformer Engine `recipe.h` | 解释 scaling 和 `DelayedScaling` |
| 工程教程 | TE Llama 2/3 fine-tuning | 解释实际训练路径和 step time |
| 硬件背景 | NVIDIA Hopper Architecture | 解释 H100 Tensor Core 能力 |

1. [NVIDIA Transformer Engine: Using FP8 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.5/user-guide/examples/fp8_primer.html)
2. [NVIDIA Transformer Engine C API recipe.h](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.9/user-guide/api/c/recipe.html)
3. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
4. [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
5. [Accelerating Hugging Face Llama with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html)
