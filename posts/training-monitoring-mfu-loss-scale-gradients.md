## 核心结论

训练监控里，`MFU`、`Loss Scale`、`梯度统计`分别回答三个问题：算力有没有被有效使用、混合精度训练是否稳定、反向传播是否出现异常。三者不能混为一谈。

`MFU` 是 Model FLOPs Utilization，白话说就是“模型实际吃到的算力占硬件峰值算力的比例”。公式是：

$$
MFU = \frac{F_{achieved}}{F_{peak}}
$$

其中 `F_achieved` 是训练过程中实际达到的有效 FLOPs，`F_peak` 是硬件理论峰值 FLOPs。大模型训练中，40% 到 50% 常被视为不错的工程水平，但这个范围依赖模型结构、GPU、并行策略、通信拓扑和 kernel 优化，不是统一标准。

`Loss Scale` 是混合精度训练中的损失缩放系数，白话说就是“先把 loss 放大，避免 fp16 梯度太小丢失，再在更新前缩回来”。它不是 loss 本身。`Loss Scale` 过低或连续下降，通常说明训练中频繁检测到 `Inf/NaN`，也就是数值溢出或非法数。

梯度统计包括 `grad_norm`、`max_abs_grad`、`Inf/NaN` 计数、每层梯度范数等。它们看的是反向传播是否健康。`grad_norm` 是梯度整体大小，白话说就是“这一步参数更新的冲击有多大”。

| 指标 | 回答的问题 | 主要风险 | 一句话判断 |
|---|---|---|---|
| `MFU` | 算力是否吃满 | 数据等待、通信开销、kernel 低效、padding 浪费 | 看效率 |
| `Loss Scale` | 混合精度是否稳定 | fp16 overflow、NaN、频繁跳步 | 看稳定 |
| `grad_norm` / `max_abs_grad` / `Inf/NaN` | 反向传播是否异常 | 梯度爆炸、异常样本、loss 实现错误 | 看健康状态 |

如果 `tokens/s` 看起来还不错，但 `MFU` 从 45% 掉到 30%，同时 `Loss Scale` 连续下降、`grad_norm` 突然飙升，那么问题更可能不是“模型变慢了”，而是“算力利用率下降 + 数值不稳定 + 反向传播异常”叠加。

结论可以压缩成一句话：`MFU` 看效率，`Loss Scale` 看稳定，`梯度统计` 看健康状态。

---

## 问题定义与边界

训练监控最常见的误判，是把吞吐、数值精度、梯度健康混在一起解释。`tokens/s` 高，不代表训练健康。比如 batch 变大时 `tokens/s` 可能上升，但如果 `MFU` 没变好、`Loss Scale` 在掉、`grad_norm` 在尖峰，说明只是表面吞吐更高，训练稳定性未必更好。

| 指标名 | 关注对象 | 异常信号 | 典型误判 |
|---|---|---|---|
| `tokens/s` | 每秒处理多少 token | 下降、波动大 | 误以为高吞吐等于高效率 |
| `MFU` | 有效计算占峰值算力比例 | 明显低于历史基线 | 误以为它能说明模型收敛好坏 |
| `Loss Scale` | AMP 的动态缩放状态 | 连续下降、长期很低 | 误以为它是训练 loss |
| `grad_norm` | 全局梯度大小 | 突然变大、长期漂移 | 误以为它直接代表模型质量 |
| `max_abs_grad` | 单个梯度最大绝对值 | 个别层尖峰 | 只看全局均值而漏掉局部异常 |
| `Inf/NaN` | 数值是否合法 | 出现非有限值 | 只看 loss 曲线而忽略跳步 |

这里有几个边界必须明确。

第一，`MFU` 不是收敛指标。它只能说明硬件是否被模型计算有效利用，不能说明模型最终效果。一个训练任务可以有很高 `MFU`，但因为数据质量差或学习率错误而完全不收敛。

第二，`Loss Scale` 不是 loss 本身。`Loss Scale` 下降通常说明 AMP 检测到了数值风险，但它不等价于训练目标变差。一次下降可能只是短暂 overflow，连续下降才更值得排查。

第三，`grad_norm` 不是模型质量本身。它可以提示梯度爆炸、异常样本、学习率过大或 loss 实现问题，但不能单独说明模型“好”或“坏”。

术语约定如下：

| 术语 | 含义 |
|---|---|
| `F_achieved` | 单位时间内实际完成的模型有效 FLOPs |
| `F_peak` | 硬件理论峰值 FLOPs |
| `Inf` | infinity，无穷大，通常来自数值溢出 |
| `NaN` | not a number，非法数，通常来自无效运算 |
| `unscale_()` 之前的梯度 | 被 `Loss Scale` 放大后的梯度 |
| `unscale_()` 之后的梯度 | 恢复到真实尺度、可用于统计和裁剪的梯度 |

---

## 核心机制与推导

`MFU` 比单看 `tokens/s` 更接近真实算力利用率，因为它把“处理了多少 token”转换成“实际完成了多少有效模型计算”。`tokens/s` 受 batch size、序列长度、padding、数据加载、通信和显存策略影响很大。两个任务的 `tokens/s` 不同，不一定说明效率不同；同一个任务的 `MFU` 明显下降，通常更能说明算力利用出了问题。

一个玩具例子：某 GPU 峰值是 `100 TFLOPs`，训练实测有效吞吐是 `45 TFLOPs`，则：

$$
MFU = \frac{45}{100} = 45\%
$$

如果下一次实验 `tokens/s` 更高，但有效计算只有 `35 TFLOPs`，`MFU` 反而变成 35%，这说明硬件被有效利用的比例下降了。可能原因包括更长 padding、通信等待、数据加载阻塞、kernel 选择变差。

混合精度训练的核心链路是：前向计算得到 loss，然后缩放 loss，再反向传播，接着反缩放梯度，检测 `Inf/NaN`，最后决定更新或跳过。

公式如下：

$$
\tilde{L} = S \cdot L
$$

$$
\tilde{g} = S \cdot g
$$

$$
g = \frac{\tilde{g}}{S}
$$

其中 `S` 是 `Loss Scale`，`\tilde{L}` 是缩放后的 loss，`\tilde{g}` 是缩放后的梯度。这样做的目的不是改变梯度方向，也不是让模型更新更大，而是避免 `fp16` 下很小的梯度发生下溢。下溢是指数值太小，小到当前精度无法表示，最后被变成 0 或损失有效位。

例如 `S = 1024` 时，原始梯度 `g = 0.003`，缩放后是：

$$
\tilde{g} = 1024 \times 0.003 = 3.072
$$

更新前再反缩放：

$$
g = \frac{3.072}{1024} = 0.003
$$

如果检测到 `Inf/NaN`，优化器通常会跳过这一步，并调小 `S`。所以 `Loss Scale` 连续下降，是一个重要事件。

全局梯度范数常用 L2 范数。L2 范数是把所有梯度元素平方、求和、再开方，用来衡量整体梯度大小：

$$
||g||_2 = \sqrt{\sum_i ||g_i||_2^2}
$$

如果要做梯度裁剪，通常计算裁剪系数：

$$
c = \min(1, \frac{M}{||g||_2 + \epsilon})
$$

其中 `M` 是最大允许范数，`\epsilon` 是防止除零的小常数。如果 `||g||_2` 小于 `M`，`c = 1`，不改变梯度；如果超过 `M`，所有梯度乘以 `c`。

玩具例子：两组梯度 `g1 = (3, 4)`、`g2 = (12, 5)`，总范数是：

$$
\sqrt{3^2 + 4^2 + 12^2 + 5^2} = \sqrt{194} \approx 13.93
$$

若阈值 `M = 10`，裁剪系数约为：

$$
c = \frac{10}{13.93} \approx 0.718
$$

| 指标 | 采集时机 | 数值升高/降低的含义 | 常见误读 |
|---|---|---|---|
| `MFU` | step 完成后，结合耗时和 FLOPs 估算 | 升高通常表示算力利用更好，降低可能表示等待或低效开销增加 | 把它当成模型效果指标 |
| `Loss Scale` | AMP scaler 更新前后 | 连续下降说明频繁 overflow 风险 | 把一次下降当成严重故障 |
| `grad_norm` | `unscale_()` 之后、裁剪之前 | 尖峰可能是异常样本、学习率过大、loss 错误 | 在缩放梯度上统计 |
| `Inf/NaN` | optimizer step 前 | 出现说明本步数值非法 | 只看 loss 而忽略跳步 |

---

## 代码实现

下面的 Python 代码是一个可运行的最小例子，用纯 Python 展示 `MFU`、`Loss Scale`、梯度范数和裁剪系数的计算。它不是完整训练框架，但能验证核心数学关系。

```python
import math

def compute_mfu(f_achieved, f_peak):
    return f_achieved / f_peak

def scaled_gradient(g, loss_scale):
    scaled = g * loss_scale
    unscaled = scaled / loss_scale
    return scaled, unscaled

def global_grad_norm(grads):
    return math.sqrt(sum(x * x for group in grads for x in group))

def clip_coef(norm, max_norm, eps=1e-6):
    return min(1.0, max_norm / (norm + eps))

mfu = compute_mfu(45.0, 100.0)
assert abs(mfu - 0.45) < 1e-12

scaled, unscaled = scaled_gradient(0.003, 1024)
assert abs(scaled - 3.072) < 1e-12
assert abs(unscaled - 0.003) < 1e-12

norm = global_grad_norm([(3, 4), (12, 5)])
assert abs(norm - math.sqrt(194)) < 1e-12

coef = clip_coef(norm, 10)
assert 0.717 < coef < 0.719
```

在 PyTorch 里，关键顺序是：先 `scaler.scale(loss).backward()`，再 `scaler.unscale_(optimizer)`，然后读取 `grad_norm`，最后决定是否 `clip_grad_norm_` 和 `scaler.step(optimizer)`。如果在 `unscale_()` 之前就统计范数，看到的是被放大的梯度，结果会假高。

```python
import torch

scaler = torch.cuda.amp.GradScaler()
max_norm = 1.0

optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(batch)

scaler.scale(loss).backward()

# 必须先反缩放，否则统计到的是被 Loss Scale 放大的梯度。
scaler.unscale_(optimizer)

# 这里得到的是裁剪前的真实尺度 grad_norm。
grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
    model.parameters(),
    max_norm=max_norm,
)

old_scale = scaler.get_scale()

# 如果 AMP 检测到 Inf/NaN，这一步可能被跳过。
scaler.step(optimizer)
scaler.update()

new_scale = scaler.get_scale()
loss_scale_dropped = new_scale < old_scale

optimizer.zero_grad(set_to_none=True)
```

| 采集点 | 指标 | 正确时机 | 错误后果 |
|---|---|---|---|
| step 耗时之后 | `tokens/s`、`MFU` | 每个 global step 完成后 | 把数据等待和计算效率混在一起 |
| `backward()` 之后 | 原始梯度存在性 | 可检查参数是否有 grad | 不能直接用于真实范数判断 |
| `unscale_()` 之后 | `grad_norm`、`max_abs_grad` | 裁剪前 | 统计到被放大的梯度 |
| `clip_grad_norm_()` 之后 | 裁剪后梯度 | 裁剪后 | 无法知道原始尖峰有多大 |
| `scaler.update()` 前后 | `Loss Scale` 变化 | 对比 old/new scale | 漏掉 overflow 和 step skip 事件 |

更完整的训练监控会记录这些字段：`step_time`、`tokens_per_second`、`mfu`、`loss`、`loss_scale`、`loss_scale_drop_count`、`grad_norm_before_clip`、`grad_norm_after_clip`、`max_abs_grad`、`nan_inf_count`、`step_skipped`。真实工程中还应按层记录梯度统计，因为全局均值会掩盖某一层的异常尖峰。

---

## 工程权衡与常见坑

这些指标都需要基线。单点值异常不一定真的有问题，趋势和组合信号更重要。

`Loss Scale` 偶尔从 `8192` 降到 `4096` 不一定是事故；如果随后又稳定恢复，通常是一次短暂波动。真正需要处理的是连续多次下降，同时伴随 `loss` 抖动和 `grad_norm` 尖峰。

真实工程例子：训练大模型时，线上仪表盘显示 `MFU` 从 `46%` 掉到 `31%`，`Loss Scale` 在 20 分钟内从 `8192` 连续降到 `256`，`grad_norm` 原本稳定在 `2-5`，随后突然冲到 `60+`。最后定位到两个问题：长样本比例暴涨导致 padding 和激活开销增加，另一个是某批数据中出现异常长序列触发梯度溢出。处理方式是修数据分桶和截断策略，加梯度裁剪，检查学习率 warmup，并把 `Loss Scale` 连续下降与 `grad_norm` 尖峰接入告警。

| 做法 | 优点 | 问题 |
|---|---|---|
| 只看 `tokens/s` | 简单直观 | 看不出硬件峰值利用率 |
| 看 `MFU` | 更接近算力利用率 | 需要明确 FLOPs 估算口径 |
| 只看全局均值 | 实现简单 | 容易掩盖单层异常 |
| 看分层统计 | 能定位异常层 | 日志量更大 |
| 单点阈值 | 容易配置 | 容易误报 |
| 移动中位数基线 | 更稳健 | 需要历史窗口 |

常见坑包括：

| 常见坑 | 后果 | 建议 |
|---|---|---|
| 在 `scaler.unscale_()` 前统计梯度 | `grad_norm` 假高 | 反缩放后再统计 |
| 把 `Loss Scale` 波动误判为 bug | 误报 | 看连续下降和是否伴随跳步 |
| 只监控全局均值 | 漏掉局部层异常 | 增加每层 `grad_norm` 和 `max_abs_grad` |
| 忽略长序列 | padding 和显存开销异常 | 监控序列长度分布 |
| 忽略学习率 warmup | 把正常初期波动当故障 | 告警规则区分训练阶段 |
| 不记录 step skip | 无法解释 loss 停顿 | 记录 AMP skip 事件 |

告警阈值应来自历史基线，而不是拍一个固定数。常见工程经验是：`MFU` 低于过去 10 到 30 分钟移动中位数的 10% 到 20%，`grad_norm` 超过基线 3 倍，或 `Loss Scale` 连续多次下降，就触发告警。这些阈值是实践经验，不是统一标准，必须按模型、硬件、batch、序列长度和优化器重新校准。

---

## 替代方案与适用边界

如果你的问题是“训练是否收敛”，那 `MFU` 不能回答；更应该看 `loss` 曲线、验证集指标、学习率曲线和梯度噪声。反过来，如果你关心的是硬件是否被充分使用，`loss` 再漂亮也不能替代 `MFU`。

| 目标 | 首选指标 | 可替代或补充指标 | 不适用场景 |
|---|---|---|---|
| 判断算力利用率 | `MFU` | `tokens/s`、GPU utilization、kernel trace | 判断模型质量 |
| 判断混合精度稳定性 | `Loss Scale`、step skip | `Inf/NaN` 计数、overflow 事件 | 判断最终精度 |
| 判断反向传播健康 | `grad_norm`、每层梯度 | `max_abs_grad`、梯度直方图 | 判断硬件利用率 |
| 判断训练是否收敛 | train loss、validation metric | 学习率曲线、困惑度、准确率 | 判断 GPU 是否吃满 |
| 判断数据管道瓶颈 | data loader 等待时间 | CPU 利用率、IO 吞吐、prefetch 队列 | 判断梯度是否爆炸 |
| 判断分布式通信瓶颈 | 通信耗时 | all-reduce 时间、pipeline bubble | 判断 loss scale 是否合理 |

可补充监控项包括：`tokens/s`、data loader 等待时间、通信耗时、GPU 显存、activation recomputation 时间、`loss`、验证集指标、每层 `grad_norm`、`step skip rate`、学习率、序列长度分布、padding 比例。

这三类指标的边界要清楚：`MFU` 适合看效率，`Loss Scale` 适合看混合精度稳定性，梯度统计适合看训练数值健康。三者都不能单独代表“训练成功”。训练成功至少还需要看收敛曲线、验证集表现、数据质量和目标任务指标。

---

## 参考资料

1. [NVIDIA NeMo-AutoModel Performance Summary](https://docs.nvidia.com/nemo/automodel/latest/performance-summary.html)
2. [NVIDIA Megatron-LM README / Performance Benchmarking](https://github.com/NVIDIA/Megatron-LM)
3. [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
4. [PyTorch Automatic Mixed Precision package](https://docs.pytorch.org/docs/stable/amp.html?highlight=autocast)
5. [PyTorch clip_grad_norm_ documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html)
6. [Reducing Activation Recomputation in Large Transformer Models](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html)
