## 核心结论

混合精度训练不是“把整个 Transformer 改成 16 位”，而是按张量职责分配精度。对 Transformer，真正重要的不是某个格式本身，而是哪些路径负责“短期计算”，哪些路径负责“长期累计”。

先给出结论：

1. 对 Transformer，通常保留参数主副本、优化器状态、梯度累加器为 FP32；前向激活、部分中间结果、反向梯度可放到 FP16 或 BF16。
2. FP16 与 BF16 的关键差异不在“都占 16 位”，而在指数位。指数位可以理解为“能覆盖多大数量级范围”的能力。BF16 的动态范围接近 FP32，因此在注意力、残差连接、LayerNorm 这类数值跨度大的路径上通常更稳。
3. FP16 训练往往依赖 loss scaling。做法是先把 loss 或梯度整体放大，再走低精度通道，更新前再缩回去，避免很小的梯度直接被舍入成 0。
4. QAT 是量化感知训练，目标主要是让模型适应部署时的低比特推理；混合精度训练的目标主要是降低训练显存、提升吞吐。两者都涉及“低精度”，但解决的是两类不同问题。

下面这张表先把差异固定下来：

| 格式 | 总位宽 | 指数位 | 尾数位 | 动态范围特点 | 在 Transformer 中的典型用途 |
|---|---:|---:|---:|---|---|
| FP32 | 32 | 8 | 23 | 范围大、精度高 | 参数主副本、优化器状态、归一化统计、累加 |
| FP16 | 16 | 5 | 10 | 范围较小，易下溢/溢出 | 激活、梯度、GEMM 输入输出，常配合 loss scaling |
| BF16 | 16 | 8 | 7 | 范围接近 FP32，尾数更粗 | 激活、梯度、注意力/残差路径，通常比 FP16 更稳 |

如果用一句面向新手的话概括：混合精度不是把“账本”压缩，而是把“流水”压缩。中间计算可以低精度，长期累计的状态不能随便降精度，否则误差会在多轮训练里持续积累，最后直接改变优化方向。

再把 Transformer 里最常见的张量分工列清楚：

| 张量类型 | 是否常驻 | 典型精度 | 原因 |
|---|---|---|---|
| 模型参数主副本 | 是 | FP32 | 参数需要跨很多 step 持续累计微小更新 |
| 优化器状态（如 Adam 一阶、二阶动量） | 是 | FP32 | 这是优化器的长期记忆，低精度噪声会影响收敛 |
| 梯度累加器 | 视训练策略而定 | FP32 | 多步累加时容易吞掉小量 |
| 前向激活 | 否 | FP16/BF16 | 占显存大，且生命周期短 |
| 大部分 matmul 输入输出 | 否 | FP16/BF16 | 吞吐提升主要来自这里 |
| 归约统计量（均值、方差、某些 logits 归一化） | 否或短期 | 常保留/回升到 FP32 | 数值稳定性要求高 |

---

## 问题定义与边界

问题不是“低精度能不能用于 Transformer”，而是“Transformer 里哪些张量可以安全降精度，哪些张量必须保住高精度”。

设训练过程中所有张量构成集合 $\mathcal{T}$，将其分为两部分：

$$
\mathcal{T} = \mathcal{T}_{\text{FP32}} \cup \mathcal{T}_{\text{cast}}, \quad
\mathcal{T}_{\text{FP32}} \cap \mathcal{T}_{\text{cast}} = \varnothing
$$

其中：

- $\mathcal{T}_{\text{FP32}}$：必须保留高精度的张量集合，通常包括参数主副本、优化器状态、梯度累加量、部分归约统计量。
- $\mathcal{T}_{\text{cast}}$：允许在计算过程中临时降为 FP16/BF16 的张量集合，通常包括激活、部分中间结果、反向传播中的局部梯度。

边界可以进一步写成：

$$
\text{activation},\ \text{local gradient} \in \mathcal{T}_{\text{cast}}
$$

$$
\text{weights}_{\text{master}},\ \text{optimizer state},\ \text{accumulator} \in \mathcal{T}_{\text{FP32}}
$$

这个边界不是经验主义拍脑袋，而是由 Transformer 的结构决定的。Transformer 中有三类路径天然更容易放大数值问题：

| 组件 | 为什么敏感 | 低精度风险 |
|---|---|---|
| Attention 打分与 softmax | 点积后再指数归一化，数值跨度大 | 溢出、softmax 饱和、NaN |
| Residual 残差连接 | 多层反复相加，误差会逐层累积 | 小量被吞掉，长期漂移 |
| LayerNorm | 依赖均值和方差，方差很小时敏感 | 归一化不稳定、梯度异常 |

把这个问题讲得更直观一点。以注意力打分为例，设 query 和 key 的单头维度为 $d_k$，则打分为：

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

随后进入 softmax：

$$
A_{ij} = \frac{e^{S_{ij}}}{\sum_j e^{S_{ij}}}
$$

如果某一行里的最大值远大于其他值，那么指数运算会迅速拉开差距。此时如果低精度表示已经把较小项全部舍入到同一档，softmax 输出就会非常接近 one-hot。训练不一定立刻报错，但梯度会变得很尖锐，优化更难稳定。

再看一个更简单的梯度例子。假设某层真实梯度是：

$$
g = 3 \times 10^{-8}
$$

对 FP32，这个数可以正常表示；但对 FP16，这个量级可能直接下溢成 0。结果是：

$$
g_{\text{stored}}^{(\text{FP16})} \approx 0
$$

一旦发生这种情况，这部分梯度信息等价于“被删除”。训练表面上还在继续，但参数实际上没有接收到这部分更新信号。

对新手来说，可以把这里的边界理解成一条简单规则：

- 能接受一点舍入误差、生命周期短、主要负责吞吐的路径，可以降精度。
- 要跨 step 保存、负责累计和统计、决定长期优化方向的路径，尽量保留 FP32。

这也是为什么混合精度训练里常见的说法不是“全模型 FP16”，而是“FP32 master weights + low-precision compute”。

---

## 核心机制与推导

混合精度训练经常和另一类概念混在一起：量化感知训练。两者都涉及低精度，但机制不同，目标也不同。

先看混合精度训练中的 loss scaling。设损失函数为 $\mathcal{L}$，参数为 $\theta$，loss scale 为 $L$。其训练步骤可写为：

$$
\begin{aligned}
\mathcal{L}' &= L \cdot \mathcal{L} \\
g' &= \nabla_\theta \mathcal{L}' \\
g &= \frac{g'}{L} \\
\theta &\leftarrow \theta - \eta g
\end{aligned}
$$

这四步的含义分别是：

1. 先把 loss 乘上一个较大的缩放因子 $L$。
2. 反向传播得到被放大的梯度 $g'$。
3. 参数更新前再把梯度除以 $L$，恢复到真实尺度。
4. 用恢复后的梯度更新 FP32 主副本参数。

核心逻辑只有一句：不是为了“改变训练目标”，而是为了“让小梯度在低精度通道里先活下来”。

举一个更具体的数值例子。若真实梯度为：

$$
g = 1 \times 10^{-8}
$$

如果直接存到 FP16 路径中，可能得到：

$$
g_{\text{fp16}} = 0
$$

若取 $L = 2^{16} = 65536$，则有：

$$
g' = g \cdot L = 6.5536 \times 10^{-4}
$$

这个值已经进入低精度更容易表示的范围。更新前再除回去：

$$
\frac{g'}{L} = g
$$

于是我们保住了梯度的“存在性”，又不改变它的数学含义。

### 为什么 BF16 常常不那么依赖 loss scaling

BF16 和 FP16 都是 16 位，但位分配不同：

- FP16：1 位符号位，5 位指数位，10 位尾数位
- BF16：1 位符号位，8 位指数位，7 位尾数位

区别在于指数位。BF16 的指数位与 FP32 一样都是 8 位，所以能覆盖的数量级范围更大。也就是说：

- FP16 的问题更多出在“范围不够”
- BF16 的问题更多出在“精度较粗”

对 Transformer 来说，动态范围通常比局部尾数精度更关键，因为注意力分数、归一化统计、残差叠加都会产生大跨度数值。因此很多实际训练会优先选择 BF16。

可以用一个简化判断来记：

| 格式 | 更常见的主要问题 |
|---|---|
| FP16 | 下溢、溢出、需要 loss scaling |
| BF16 | 舍入更粗，但不容易因为范围不够而崩掉 |

### 混合精度与 QAT 的区别

QAT 不是训练提速机制，而是部署适配机制。它的目标是在训练时就显式模拟量化误差，让模型在最终转成 INT8 或更低比特时性能损失更小。

一个常见的对称量化形式是：

$$
x_q = \text{clip}\left(\left\lfloor \frac{x}{s} \rceil\right., -q_{\max}, q_{\max}\right)
$$

反量化后得到近似值：

$$
\hat{x} = s \cdot x_q
$$

其中：

- $s$ 是 scale，用来把实数映射到整数网格
- $q_{\max}$ 由位宽决定，例如 INT8 常见范围与 8 位整数表示相关
- $\lfloor \cdot \rceil$ 表示四舍五入

QAT 的困难在于舍入操作不可导，所以常用 STE（Straight-Through Estimator，直通估计）近似反向传播。简化写法是：

$$
\frac{\partial \hat{x}}{\partial x} \approx 1
$$

它的含义不是“量化真的可导”，而是“前向按量化误差走，反向给一个可训练的近似梯度”。

因此两者的边界很清楚：

| 机制 | 训练时做什么 | 主要目标 |
|---|---|---|
| 混合精度训练 | 仍做浮点计算，只是不同路径用不同位宽 | 降显存、提吞吐 |
| QAT | 训练时显式插入量化/反量化模拟 | 适配低比特部署 |

真实工程里，两者甚至可以叠加。例如：

1. 预训练阶段用 BF16 提升吞吐。
2. 微调或部署适配阶段再做 QAT，准备 INT8 推理。

### 为什么优化器状态通常保留 FP32

以 AdamW 为例，它要维护一阶动量 $m_t$ 和二阶动量 $v_t$：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

它们本质上是长期统计量。若这些状态长期保存在低精度里，就会出现两个问题：

1. 很小的增量会被直接舍掉，统计量更新失真。
2. 多轮累计后的误差不是随机噪声，而是会反馈到学习率校正和更新方向。

所以大模型训练中，常见策略是：

- 计算路径尽量低精度
- 关键累计状态尽量高精度

这也是“算得快”和“学得稳”之间最常见的折中。

---

## 代码实现

下面先给一个可运行的 Python 玩具示例。它不依赖 GPU，也不依赖 PyTorch，只用“离散量化步长”模拟低精度吞掉小梯度的现象，并演示 loss scaling 如何让梯度保留下来。

```python
from decimal import Decimal, ROUND_HALF_EVEN

def fake_low_precision_quantize(x, step):
    # 用固定量化步长模拟低精度存储
    x = Decimal(str(x))
    step = Decimal(str(step))
    q = (x / step).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
    return float(q * step)

def loss_scale_demo(grad, scale, step=1e-4):
    scaled_grad = grad * scale
    stored = fake_low_precision_quantize(scaled_grad, step)
    recovered = stored / scale
    return {
        "grad": grad,
        "scale": scale,
        "scaled_grad": scaled_grad,
        "stored_in_low_precision": stored,
        "recovered_grad": recovered,
    }

def main():
    grad = 1e-8

    no_scale = fake_low_precision_quantize(grad, step=1e-4)
    assert no_scale == 0.0

    result = loss_scale_demo(grad=grad, scale=65536, step=1e-4)

    assert result["stored_in_low_precision"] > 0.0
    assert result["recovered_grad"] > 0.0
    assert abs(result["recovered_grad"] - grad) < 5e-9

    print("without loss scaling:", no_scale)
    print("with loss scaling:")
    for k, v in result.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
```

这个示例对应的不是 IEEE 严格 FP16 仿真，而是一个更适合教学的最小模型。它想说明的只有一件事：

- 如果低精度通道分辨率很粗，小梯度会直接掉成 0。
- 先整体放大，再写入低精度通道，最后再缩回去，可以避免这类信息丢失。

再看一个真实训练中的 PyTorch 例子。下面的代码可以在 GPU 上启用自动混合精度；如果没有 CUDA，它会退回普通 FP32 路径，不会报错。

```python
import torch
from torch import nn

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.GELU(),
        nn.Linear(64, 4),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(8, 16, device=device)
    y = torch.randint(0, 4, (8,), device=device)

    # BF16 通常更稳；若硬件不支持则退到 FP16
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16

    # 只有 FP16 路径通常强依赖 GradScaler；BF16 可不启用
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and amp_dtype == torch.float16)

    optimizer.zero_grad(set_to_none=True)

    if use_cuda:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
            loss = loss_fn(logits, y)
    else:
        logits = model(x)
        loss = loss_fn(logits, y)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    assert torch.isfinite(loss).item()
    print("loss:", float(loss))

if __name__ == "__main__":
    main()
```

这段代码里的关键点不是 API 名字，而是对应的训练语义：

| 步骤 | 作用 | 为什么重要 |
|---|---|---|
| `autocast` | 为算子自动选择低精度路径 | 提升吞吐，减少激活显存 |
| `GradScaler` | 对 FP16 路径做 loss scaling | 避免小梯度下溢 |
| `unscale_(optimizer)` | 裁剪前恢复真实梯度尺度 | 否则裁剪阈值含义失真 |
| `clip_grad_norm_` | 控制梯度爆炸 | 注意力和长序列训练中常见 |
| `optimizer.step()` | 用主副本参数完成更新 | 保住长期累计精度 |

如果进一步把它映射回 Transformer，可以得到一张更接近实战的“精度放置图”：

| 模块 | 常见计算精度 | 常见累计/统计精度 |
|---|---|---|
| Q/K/V 投影、FFN matmul | FP16/BF16 | 参数主副本 FP32 |
| Attention logits 生成 | FP16/BF16 起算，关键稳定化常回到 FP32 | softmax 相关归约常高精度 |
| LayerNorm | 输入可低精度 | 均值、方差、某些实现中的归约保留 FP32 |
| 残差加法 | 常为低精度输入 | 部分框架在关键路径做高精度累加 |
| AdamW 状态 | 不建议低精度长期保存 | FP32 |

对新手来说，真正应当记住的是：混合精度不是“某个全局开关打开后所有问题自动解决”，而是框架替你做了大量“哪些操作降精度、哪些操作保精度”的选择。只要你写了自定义算子、自定义 LayerNorm 或特殊注意力模块，就需要重新检查这些边界是否还成立。

---

## 工程权衡与常见坑

混合精度的收益主要来自两点：

1. 激活和中间结果占用更少显存。
2. GPU 在低精度矩阵乘上吞吐更高。

但收益并不是无条件的。它把一部分“硬件效率问题”换成了“数值稳定性管理问题”。

先看常见失效模式：

| 格式 | 典型失效模式 | 常见位置 | 应对措施 | 硬件要求 |
|---|---|---|---|---|
| FP16 | 梯度下溢为 0 | 深层网络反向传播 | loss scaling、梯度累加 | 多数现代 GPU 支持 |
| FP16 | softmax/归一化溢出，出现 NaN | attention logits、LayerNorm 前后 | 减去最大值、关键统计保留 FP32 | 多数现代 GPU 支持 |
| BF16 | 舍入更粗，局部差值不明显 | 极小差值比较、细粒度统计 | 关键归约与状态保留 FP32 | 原生 BF16 支持更理想 |
| 二者共性 | 优化器状态精度不足导致震荡 | AdamW 动量、方差估计 | 状态保持 FP32 | 无特殊额外要求 |

下面把工程里最常见的坑补全说清楚。

### 1. 把“计算低精度”和“累计低精度”混为一谈

这是最常见误解。低精度计算不等于低精度存档。

例如某参数每一步只更新一个很小的增量 $\Delta \theta$：

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

若 $\eta g_t$ 很小，而参数本身数值较大，那么低精度主副本可能直接看不到这次变化。长期下来，模型会表现成“训练在跑，但学得很慢甚至不学”。

### 2. 在 `unscale_` 之前做梯度裁剪

若梯度已被 scale 放大 $L$ 倍，而你先裁剪再反缩放，那么实际裁剪阈值也被错误放大了。正确顺序是：

1. `scale(loss).backward()`
2. `unscale_(optimizer)`
3. `clip_grad_norm_`
4. `step()`

顺序错了，训练未必立刻崩，但梯度控制已经失真。

### 3. 忽略 softmax 前的稳定化处理

attention 中常见做法是先减去行最大值：

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

这一步不是数学多余项，而是数值稳定化的基础操作。低精度训练里它更重要，因为指数运算对范围极其敏感。

### 4. 误以为 BF16 一定“更精确”

不是。BF16 的优势是动态范围，不是尾数精度。它更稳，不代表它在每个局部数值上都更细。

如果任务特别依赖很小差值的比较，BF16 也可能引入更粗的舍入误差。因此工程里常见策略不是“全 BF16 无脑跑”，而是“BF16 计算 + FP32 关键归约”。

### 5. 没有监控 `inf`、`nan` 和 loss 异常跳变

训练不会总是立即报错。很多时候问题表现为：

- loss 偶发尖峰
- 梯度范数突然异常
- 验证集指标持续漂移
- 若干 step 后出现 `nan`

因此实际训练里通常会监控：

| 监控项 | 作用 |
|---|---|
| loss 是否有限 | 第一层告警 |
| 梯度范数 | 检查爆炸/反常裁剪 |
| 参数更新范数 | 看是否出现“几乎不更新” |
| 激活/梯度中的 `nan` 比例 | 排查具体模块 |
| scaler 动态变化 | 观察 FP16 是否经常溢出 |

### 6. 在不值得的场景里强开混合精度

以下场景收益可能不明显：

- 模型很小，矩阵乘不是瓶颈
- batch 很小，GPU 没吃满
- 任务瓶颈在数据读取、CPU 预处理或通信
- 训练主要用于调试、对照实验和严格复现

这时完全 FP32 可能更省心。

### 7. 自定义层默认相信框架会自动兜底

框架对标准 `Linear`、`LayerNorm`、`Attention` 往往有成熟处理；但一旦你写了自定义 kernel、自定义归一化、自定义 fused op，就要重新确认：

- 哪些输入被 cast 了
- 哪些归约仍在 FP32
- softmax 或方差计算是否稳定
- backward 中是否也保持了合理精度路径

从工程视角看，混合精度最合理的理解是：

- 压缩的是带宽和显存压力最大的计算路径
- 不压缩的是承担统计、累计、更新职责的状态路径

前者决定效率，后者决定训练是否还能收敛。

---

## 替代方案与适用边界

如果目标是训练提速或节省显存，常见方案可以分为三类：

| 方案 | 吞吐 | 稳定性 | 部署价值 | 适用场景 |
|---|---|---|---|---|
| 完全 FP32 | 最低 | 最高 | 一般 | 调试、基线、对稳定性极端敏感 |
| 混合精度训练 | 高 | 中到高，取决于实现 | 中 | 大多数现代 Transformer 训练 |
| QAT | 训练期较慢 | 中 | 高 | 目标是 INT8/更低比特部署推理 |

### 什么时候优先 BF16

如果硬件原生支持 BF16，Transformer 训练通常优先尝试 BF16。原因不是它“更高级”，而是它的动态范围更接近 FP32，在以下路径上更稳：

- attention logits
- residual 累加
- LayerNorm 附近的数值波动
- 大模型训练后期的小步更新过程

可以把经验规则写得直接一些：

1. 有原生 BF16 支持时，优先 BF16 混合精度。
2. 没有 BF16，但有成熟 FP16 Tensor Core 支持时，使用 FP16 + loss scaling。
3. 若训练目标是极致稳定、调试和对照复现，FP32 仍然合理。

### 什么时候 FP16 仍然是成熟路线

FP16 的问题更集中在数值范围，但生态非常成熟。只要配好下面几项，它仍然是主流可用方案：

- `GradScaler`
- 梯度裁剪
- `nan/inf` 监控
- softmax 稳定化
- 关键统计量 FP32 保留

很多项目里，FP16 不是“不能用”，而是“要更认真地管稳定性”。

### 什么时候要看 QAT，而不是只看混合精度

如果最终目标是部署到：

- 移动端
- 边缘设备
- 高吞吐推理服务
- 显存或带宽极其受限的环境

那么混合精度通常不够。因为混合精度解决的是训练期效率，不是 INT8/INT4 推理适配。此时更需要考虑：

- PTQ（后训练量化）
- QAT（量化感知训练）
- 激活离群值处理
- 每层 scale 分配
- 权重与激活不同 bit-width 策略

两者不是互斥关系，而是不同阶段的工具链：

| 阶段 | 关注点 | 常见技术 |
|---|---|---|
| 训练阶段 | 显存、吞吐、稳定性 | BF16 / FP16 混合精度 |
| 部署阶段 | 延迟、带宽、成本 | PTQ / QAT / 低比特推理 |

### 对新手更实用的选择建议

如果你第一次训练 Transformer，可以直接按下面决策：

1. 先看硬件是否支持 BF16。支持就先试 BF16。
2. 不支持 BF16，就用 FP16 + `GradScaler`。
3. 若出现 `nan`、loss 抖动、训练后期不稳定，检查 softmax、LayerNorm、梯度裁剪和优化器状态精度。
4. 若目的是部署到 INT8 甚至更低比特，再单独考虑 QAT 或 PTQ。
5. 自定义层较多时，不要默认相信自动混合精度会替你处理所有边界。

一句话总结适用边界：

- 混合精度解决“训练时怎么更快、更省显存”。
- QAT 解决“部署时怎么更低比特、损失更小”。
- FP32 解决“先把基线跑稳”。

---

## 参考资料

| 来源 | 内容摘要 | 覆盖章节 |
|---|---|---|
| NVIDIA: Mixed Precision Training | 经典工程实践资料，解释 FP16 训练、master weights、loss scaling 和数值稳定性 | 核心结论、核心机制、工程权衡 |
| Google Cloud TPU / bfloat16 资料 | 说明 BF16 的位分配、动态范围与训练稳定性的关系 | 核心结论、问题定义、替代方案 |
| Micikevicius et al., *Mixed Precision Training* | 系统阐述混合精度训练的基本方法与稳定性条件 | 核心机制、工程权衡 |
| Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* | 量化与 QAT 的代表性论文，解释 scale、round、STE 等机制 | 核心机制、替代方案 |
| PyTorch AMP 文档 | 自动混合精度与 `GradScaler` 的实际 API 和使用边界 | 代码实现、工程权衡 |
| Transformer 原始论文与后续训练实践资料 | 帮助理解 attention、residual、LayerNorm 的数值敏感来源 | 问题定义、工程权衡 |

建议阅读顺序：

1. 先看混合精度训练的工程材料，建立“哪些张量保留 FP32”的基本边界。
2. 再看 BF16 与 FP16 的格式差异，理解为什么 Transformer 常偏向 BF16。
3. 接着看 AMP 文档，把理论映射到真实训练代码。
4. 最后看 QAT 论文或量化资料，明确它与混合精度是两个问题域。

如果只保留一条阅读主线，可以按这个问题顺序读：

1. 为什么小梯度会在 FP16 中消失？
2. 为什么 BF16 在 Transformer 上通常更稳？
3. 为什么优化器状态不能轻易低精度长期保存？
4. 为什么 QAT 不是混合精度的替代词？
