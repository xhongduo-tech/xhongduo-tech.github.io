## 核心结论

FP8 推理的目标，不是把整个模型“统一降成 8 位”，而是把推理中最耗时、最占带宽的大矩阵乘法路径压到 8 位浮点，再用缩放因子把数值映射回可表示范围内。这里的“缩放因子”就是一个比例尺，作用是让原本太大或太小的数值先被压到 FP8 能装下的区间里，再参与计算。

常见做法是用 `E4M3` 或 `E5M2` 两种 FP8 格式配合 `amax + scale` 定标。`amax` 是张量绝对值最大值，也就是“这一批数据里最大的幅度”；`scale` 是按这个最大值算出来的比例。一个典型公式是：

$$
amax = \max_i |x_i|,\quad s = \frac{amax}{448}
$$

$$
x_q = castToFp8(clip(x / s, -448, 448)),\quad x_{\hat{}} = x_q \cdot s
$$

这里 `clip` 表示截断到 FP8 可表示范围，`castToFp8` 表示按 FP8 格式存储。直白地说，就是“先按比例缩小，再塞进更小的盒子里，取出来时再按同样比例放大”。

FP8 的主要收益来自三件事：显存更省、数据搬运更少、原生 FP8 Tensor Core 吞吐更高。它不是单靠“位宽更小”自动变快。如果硬件没有原生 FP8 路径，或者量化后图被切碎、频繁做量化与反量化，收益会迅速缩水。

| 格式 | 位宽 | 动态范围 | 单位数据占用 | 常见推理吞吐潜力 | 精度风险 |
|---|---:|---|---|---|---|
| FP16 | 16 bit | 中等 | 基线 | 稳定 | 低 |
| BF16 | 16 bit | 较大 | 基线 | 稳定 | 低 |
| FP8 E4M3 | 8 bit | 较小 | 约减半 | 高 | 中 |
| FP8 E5M2 | 8 bit | 较大 | 约减半 | 高 | 较高 |

一句话定义：FP8 推理是在可接受精度损失内，把 LLM 或卷积网络中的主计算路径改为 FP8，并通过缩放因子控制量化误差与溢出风险。

---

## 问题定义与边界

FP8 推理主要解决的是推理阶段的带宽瓶颈和吞吐瓶颈。更具体地说，当模型里存在大量 `X @ W` 这样的矩阵乘法时，权重读取、激活搬运、Tensor Core 计算都会消耗大量时间。FP8 的价值，集中体现在这些“算得多、搬得多”的位置。

适合 FP8 的通常是高算力密度算子，也就是“每次搬运一块数据，都能做很多乘加运算”的算子，例如线性层、QKV 投影、MLP、卷积。相反，像 LayerNorm、Residual Add、Softmax 这种对数值敏感、而且常常不是吞吐主瓶颈的算子，通常继续保留 BF16 或 FP16。

| 算子类型 | 是否适合 FP8 | 原因 |
|---|---|---|
| Linear / GEMM | 适合 | 计算密集，Tensor Core 收益明显 |
| QKV Projection | 适合 | 本质是大线性层 |
| MLP / FFN | 适合 | 参数量大、带宽占比高 |
| Conv | 适合 | 在支持的实现上收益明显 |
| LayerNorm | 通常不建议 | 对缩放和舍入误差敏感 |
| Softmax | 通常不建议 | 指数与归一化对数值稳定性要求高 |
| Residual Add | 通常保留高精度 | 累加误差容易放大 |
| 小矩阵算子 | 收益不稳定 | 量化开销可能盖过计算收益 |

真实工程里，一个典型场景是在 H100 上部署大语言模型。模型的线性层很多，`QKV` 和 `MLP` 占了主要时间，因此它们适合 FP8；但注意力里的 `softmax`、残差连接、归一化层通常还是保留更高精度。新手可以把它理解成：不是整条计算图都压成 8 位，而是只压缩“最贵的那一段”。

推理图可以抽象成下面的混合精度流程：

```python
def block(x, w_qkv, w_o, w_ffn1, w_ffn2):
    # 高精度保留段
    x_norm = layer_norm_bf16(x)

    # FP8 主路径
    qkv = fp8_linear(x_norm, w_qkv)
    attn = attention_core_bf16(qkv)   # softmax 等敏感部分保留高精度
    x = residual_add_bf16(x, fp8_linear(attn, w_o))

    # 再次进入 FP8 主路径
    y = layer_norm_bf16(x)
    y = fp8_linear(y, w_ffn1)
    y = gelu_bf16(y)
    y = fp8_linear(y, w_ffn2)

    return residual_add_bf16(x, y)
```

---

## 核心机制与推导

FP8 量化的核心不是“砍位宽”，而是“先定标，再量化”。定标的含义，是先找到一组数据里最大的绝对值，然后据此计算缩放因子，让主要数值尽量落在 FP8 的有效范围内。

最简单的玩具例子如下。设：

$$
x = [-3.5, 1.0, 7.0]
$$

那么：

$$
amax = \max_i |x_i| = 7,\quad s = \frac{7}{448} = 0.015625
$$

于是：

$$
x / s = [-224, 64, 448]
$$

这三个值都还在可表示范围内，所以量化再反量化后，可以恢复成原值。如果再来一个 `9.0`，那么：

$$
9.0 / 0.015625 = 576
$$

它超过了 `448`，会被截断成 `448`，反量化后只剩 `7.0`。这部分损失就叫饱和误差，也就是“值太大，盒子装不下，被压平了”。

`E4M3` 和 `E5M2` 的差别，本质是指数位和尾数位的分配不同。指数位越多，动态范围越大；尾数位越多，数值精细度越高。

| 格式 | 指数/尾数 | 动态范围特点 | 精度特点 | 常见用途 |
|---|---|---|---|---|
| E4M3 | 4 指数 / 3 尾数 | 较小 | 相对更细 | 推理主格式，常用于前向 |
| E5M2 | 5 指数 / 2 尾数 | 更大 | 相对更粗 | 更关注范围时使用 |

为什么会有“每层缩放”和“每块缩放”？因为整个张量共用一个 `s` 时，如果少数极大值把 `amax` 拉得很高，绝大多数普通值就会被压得过小，导致量化后分辨率不够。把一个大张量拆成多个层、多个通道、多个 block 分别定标，本质是在缩小“共享比例尺”的作用范围。

下面是一个可运行的最小 Python 实现，演示 `amax`、缩放、量化、反量化，以及饱和误差：

```python
import numpy as np

FP8_MAX = 448.0

def quantize_fp8_like(x: np.ndarray):
    amax = np.max(np.abs(x))
    scale = amax / FP8_MAX if amax > 0 else 1.0
    q = np.clip(x / scale, -FP8_MAX, FP8_MAX)
    # 这里用四舍五入模拟离散表示，真实 FP8 还包含指数/尾数编码
    q = np.round(q)
    x_hat = q * scale
    return q, x_hat, scale

x = np.array([-3.5, 1.0, 7.0], dtype=np.float32)
q, x_hat, scale = quantize_fp8_like(x)

assert np.isclose(scale, 7.0 / 448.0)
assert np.allclose(x_hat, x, atol=scale)

x2 = np.array([9.0], dtype=np.float32)
q2, x_hat2, scale2 = quantize_fp8_like(x2)
assert q2[0] == 448.0
assert np.isclose(x_hat2[0], 9.0)  # 单元素场景不会饱和，因为 scale 跟着变
```

这个例子里最后一个 `assert` 成立，是因为单元素张量会重新计算自己的 `scale`。真实工程中的饱和，通常出现在“共享同一个 `scale` 的多元素张量”里，而不是单个元素单独定标。

---

## 代码实现

实现 FP8 推理，要把“离线处理”和“在线处理”分开看。离线处理通常发生在模型加载或校准阶段，用来准备权重、收集统计信息、生成适配 kernel 的元数据。在线处理发生在真正推理时，主要负责激活的 `amax` 统计、`scale` 更新和 FP8 kernel 调用。

从计算图角度看，矩阵乘法不是简单的 `A_q @ B_q`，而是：

$$
y = (A_q \cdot s_a) @ (B_q \cdot s_b)
$$

如果底层 kernel 支持融合，反缩放通常不会显式展开成两个大张量乘法，而是以内核参数形式进入计算。等价理解可以写成：

$$
y = (A_q @ B_q) \cdot (s_a \cdot s_b)
$$

下面是一段 PyTorch 风格伪代码：

```python
class FP8Linear:
    def __init__(self, weight_fp16):
        self.weight_master = weight_fp16
        self.w_amax = self.weight_master.abs().max()
        self.w_scale = self.w_amax / 448.0
        self.weight_fp8 = cast_to_fp8(self.weight_master / self.w_scale)

    def forward(self, x_bf16):
        x_amax = x_bf16.abs().max()
        x_scale = max(x_amax / 448.0, 1e-12)

        x_fp8 = cast_to_fp8(x_bf16 / x_scale)

        # 实际收益依赖底层 FP8 GEMM kernel
        y = fp8_gemm(
            a=x_fp8,
            b=self.weight_fp8,
            a_scale=x_scale,
            b_scale=self.w_scale,
            out_dtype="bf16",
        )
        return y
```

真实工程例子可以看成三步：

1. 权重离线校准，转换为 FP8 或保留高精度主副本并记录 `scale`。
2. 激活在运行时按层或按块更新 `amax`。
3. 通过 TensorRT 或 Transformer Engine 调用原生 FP8 kernel。

配置示意如下：

```python
# Transformer Engine 风格示意
fp8_recipe = {
    "format": "E4M3",
    "scaling": "per-tensor-or-per-block",
    "amax_history_len": 16,
}

with fp8_autocast(enabled=True, recipe=fp8_recipe):
    y = transformer_block(x)
```

| 实现方式 | 权重处理 | 激活处理 | 性能潜力 | 实现复杂度 |
|---|---|---|---|---|
| 离线量化 | 预先量化并存储 scale | 运行时少量处理 | 高 | 中 |
| 运行时动态量化 | 保留高精度副本 | 每次推理统计 `amax` | 中到高 | 高 |
| 混合精度 | 只量化主路径 | 敏感层保留 BF16/FP16 | 最稳妥 | 中 |

---

## 工程权衡与常见坑

FP8 是否真的变快，取决于总成本，而不是位宽本身。一个常见误解是“8 位一定比 16 位快”。实际要看 `Q/DQ` 插入位置、shape 是否对齐、kernel 是否融合、batch 和序列长度是否足够大。

误差也不是单一来源，而是几部分叠加：

$$
total\_error = quantization\_error + saturation\_error + accumulation\_error
$$

其中量化误差来自舍入，饱和误差来自截断，累加误差来自长链路上的乘加累计。

| 常见坑 | 现象 | 规避手段 |
|---|---|---|
| 全图无脑 FP8 化 | 精度骤降 | 只量化大 GEMM，保留敏感算子高精度 |
| scale 太粗 | 溢出、截断多 | 改为分层或分块定标 |
| scale 太碎 | 元数据和开销变大 | 只在热点层细化 |
| 小 batch / 小矩阵 | 速度不升反降 | 优先在大模型、大矩阵场景启用 |
| 频繁 Q/DQ | 带宽收益被抵消 | 尽量让量化区间连续、kernel 融合 |
| 校准数据不代表真实流量 | 线上精度波动 | 用代表性 prompt 或样本做校准 |

错误做法与推荐做法的差别很典型：

```python
# 错误做法：每个小算子都单独量化/反量化
x = dq(fp8(mm(q(x), q(w1))))
x = dq(fp8(add(q(x), q(residual))))
x = dq(fp8(mm(q(x), q(w2))))

# 推荐做法：把连续的大计算段尽量合并到同一条低精度路径
x_fp8 = q(x)
y_fp8 = fused_fp8_block(x_fp8, w1_fp8, w2_fp8)
x = residual_add_bf16(dq(y_fp8), residual)
```

对新手来说，判断标准可以简化为一句话：只有当“量化、搬运、反量化、计算”加起来仍然比原先便宜时，FP8 才有意义。

---

## 替代方案与适用边界

FP8 不是唯一低精度方案。它更像是“在支持它的硬件上，偏激进但高收益的一档”。如果硬件不支持原生 FP8，或者模型对精度极其敏感，BF16、FP16、INT8、INT4 往往更合适。

| 格式 | 适用条件 | 优点 | 风险 |
|---|---|---|---|
| BF16 | 通用 GPU，重视稳定性 | 动态范围大，部署稳 | 压缩收益有限 |
| FP16 | 常规半精度部署 | 生态成熟 | 范围不如 BF16 |
| FP8 | Hopper 等原生支持平台 | 吞吐和带宽收益高 | 对实现和缩放策略敏感 |
| INT8 | 成熟量化工具链 | 压缩效果好，兼容广 | 校准要求高 |
| INT4 | 极致压缩场景 | 显存收益最大 | 精度风险更高 |

一个实用的定性判据是：如果性能收益 $\Delta P$ 明显大于精度损失成本 $\Delta E$，并且工程复杂度可控，那么该格式值得采用。也就是：

$$
\text{choose format if } \Delta P \gg \Delta E \text{ and deployment cost is acceptable}
$$

可以用一个简单决策树理解：

```python
def choose_dtype(has_native_fp8, model_sensitive, graph_fragmented):
    if not has_native_fp8:
        return "BF16 or INT8"
    if model_sensitive:
        return "BF16"
    if graph_fragmented:
        return "FP16/BF16 or selective INT8"
    return "FP8"
```

真实工程里，如果部署平台不是 Hopper/H100 这类支持原生 FP8 的设备，那么 FP8 常常只有“格式转换”的成本，没有足够的 kernel 红利。反过来，如果是大模型、长序列、吞吐优先、平台支持原生 FP8，那么 FP8 往往是值得优先验证的方案。

---

## 参考资料

| 资料 | 类型 | 覆盖内容 | 建议阅读顺序 |
|---|---|---|---|
| FP8 Formats for Deep Learning | 论文 | FP8 格式定义与理论动机 | 1 |
| TensorRT Quantized Types | 官方文档 | 推理图中的量化实现 | 2 |
| Transformer Engine FP8 Primer | 官方文档 | FP8 训练与推理使用方式 | 3 |
| CUDA Math API FP8 Intrinsics | 官方文档 | 底层 FP8 接口与数据类型 | 4 |
| Transformer Engine GitHub | 官方仓库 | 代码与示例 | 5 |

1. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
2. [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
3. [NVIDIA Transformer Engine: Using FP8 and FP4 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
4. [CUDA Math API: FP8 Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP8.html)
5. [NVIDIA Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
