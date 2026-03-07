## 核心结论

多模态模型量化，指把模型中的浮点权重和激活压缩成更低位宽的整数表示，以降低显存占用、减少访存带宽、提升推理吞吐。先给结论：**视觉编码器和语言模型不能按同一套激进策略量化**。视觉编码器处理二维像素结构，局部纹理、边缘、颜色变化会直接影响后续特征；语言模型主要处理高维 token 表示和大矩阵乘法，冗余更高，量化误差更容易在后续层被吸收。因此在部署上，通常应先量化语言侧，再谨慎评估视觉侧。

对初学者，可以把“降低位宽”理解成“把连续实数压进更少的刻度里”。刻度越少，表示越粗。对语言模型来说，很多层的输出带有统计冗余，粗一点通常还能工作；对视觉模型来说，一些看似很小的数值可能恰好对应边缘、细字、图表线条，这类信息被压坏后，很难在后续层恢复。

公开 benchmark 也支持这一判断。以 Qwen2-VL-72B 为例，在 MMMU 上，BF16 为 65.44，GPTQ-INT8 为 64.56，GPTQ-INT4 为 64.00，AWQ-INT4 为 64.22。也就是说，语言解码器降到 INT4 后，准确率下降仍控制在约 1.5 个点内，但显存和吞吐收益已经足够明显。这说明**部署优先级通常应是“先量语言侧，再谨慎碰视觉侧”**。

| 方案 | 位宽 | MMMU 准确率 | 相对 BF16 变化 | 显存占用趋势 | 适合场景 |
|---|---:|---:|---:|---|---|
| BF16 | 16-bit | 65.44 | 基线 | 最高 | 离线评测、精度优先 |
| GPTQ-INT8 | 8-bit | 64.56 | -0.88 | 明显下降 | 在线推理、先保精度 |
| GPTQ-INT4 | 4-bit | 64.00 | -1.44 | 大幅下降 | 显存紧张、吞吐优先 |
| AWQ-INT4 | 4-bit | 64.22 | -1.22 | 大幅下降 | 语言层低位部署 |

这里不要把结论误解成“视觉侧不能量化”。更准确的说法是：**视觉侧可以量化，但通常要更保守，常见做法是保留 BF16/FP16，或只把部分层降到 INT8。**

再补一个工程判断：

| 模块 | 常见量化起点 | 常见下探终点 | 风险等级 | 常见部署顺序 |
|---|---|---|---|---|
| 视觉编码器 ViT | BF16 / FP16 | INT8 | 高 | 最后动 |
| 视觉到语言投影层 | FP16 | INT8 / 局部 INT4 | 中高 | 第二步 |
| 语言模型 LLM | INT8 | INT4 | 中低 | 第一步 |

---

## 问题定义与边界

问题定义很具体：在尽量不破坏图文对齐能力的前提下，把多模态模型压缩到更低位宽。所谓图文对齐，可以先理解成“图片提取出来的语义，进入语言模型后仍能被正确解释”。如果量化以后模型还能看图说话、识别图表、理解文档、完成视觉问答，就说明压缩没有破坏关键能力。

多模态模型通常至少包含三部分：

| 组件 | 输入类型 | 主要计算形态 | 对量化敏感度 | 原因 |
|---|---|---|---|---|
| 视觉编码器 ViT | 图像 patch | 二维空间特征提取 | 高 | 依赖局部细节，激活分布不均匀 |
| 融合层 / 投影层 | 视觉 token 到语言空间 | 特征映射与对齐 | 中高 | 少量误差就可能破坏图文映射 |
| 语言模型 LLM | token 序列 | 大矩阵乘法 + 自注意力 | 中低 | 参数冗余高，误差可被部分吸收 |

“激活分布不均匀”是理解视觉量化敏感性的关键。初学者可以这样看：不是所有数值都差不多大，而是少数值特别大，大多数值比较小。统一量化时，少数大值会决定缩放比例，导致大量小值只能挤在很少的整数刻度里。视觉里的小值未必不重要，它们可能对应边缘、笔画、阴影或细小目标。语言层也存在异常值，但整体上线性层更深、残差更多，误差有更大概率被“摊薄”。

一个常用的误差表达式是：

$$
\varepsilon = \|Wx - s_w s_x W_q x_q\|
$$

其中：

| 符号 | 含义 |
|---|---|
| $W$ | 原始浮点权重 |
| $x$ | 原始浮点激活 |
| $W_q$ | 量化后的整数权重 |
| $x_q$ | 量化后的整数激活 |
| $s_w, s_x$ | 权重和激活的缩放因子 |
| $\varepsilon$ | 量化前后输出的近似误差 |

这里的目标不是让每一层完全无误差，而是让最终任务误差可接受。很多初学者会误以为“某层误差大一点，最后一定坏”。实际不是这样。量化看的是**误差是否落在模型还能容忍的范围内**，而不是某一层必须近似到机器精度。

玩具例子可以这样理解。假设一张图只包含“黑底上的一条细白线”。对人来说，这张图很简单；但对模型来说，那条线可能对应少量但关键的高响应通道。如果视觉层量化过粗，这些响应会被截断或合并，后续就可能把“白线”看成噪声。相比之下，一段语言序列中如果某一层几个通道发生小误差，后面几十层还有机会通过残差连接和上下文信息把语义拉回来。

因此，边界可以先定成三条：

1. 视觉编码器优先保精度，通常从 BF16/FP16 或 INT8 起步。
2. 语言模型优先做低位压缩，可先试 INT8，再试 INT4。
3. 全模型同时激进量化风险最高，因为视觉误差和语言误差会在对齐层叠加，并以非线性方式放大。

再补一条常被忽略的边界：**“文本任务精度稳定”不等于“多模态任务也稳定”**。如果只看语言侧指标，例如 perplexity，很可能低估图文对齐已经被破坏的程度。因此评估必须覆盖视觉问答、文档理解、图表理解和 OCR。

---

## 核心机制与推导

量化的基础流程，是把浮点数映射到有限的整数区间。静态量化，指先用一批样本统计范围，再固定缩放因子；动态量化，指在推理时根据当前输入重新估计激活范围。前者实现简单、部署稳定，后者对输入变化更灵活，但实现复杂度更高。

以线性层

$$
y = Wx + b
$$

为例，常见的对称量化写法是：

$$
W_q = \text{round}(W / s_w), \quad x_q = \text{round}(x / s_x)
$$

其中缩放因子常由最大绝对值确定：

$$
s_w = \frac{\max(|W|)}{Q_{\max}}, \quad
s_x = \frac{\max(|x|)}{Q_{\max}}
$$

如果是 INT8，则常取 $Q_{\max}=127$；如果是对称 INT4，则常取 $Q_{\max}=7$。

量化后，整数域里的近似计算可以写成：

$$
y_q = W_q x_q
$$

再通过缩放恢复到浮点近似值：

$$
\hat y = s_w \cdot s_x \cdot y_q + b
$$

因此单层前向的完整近似关系是：

$$
Wx + b \approx s_w s_x (W_q x_q) + b
$$

这一步的直觉是：先把实数缩小到整数刻度里，利用低位算子完成乘加，再乘回比例尺。好处是内存占用和访存量下降，坏处是引入舍入误差与截断误差。

误差主要来自三个位置：

| 步骤 | 误差来源 | 对 VLM 的影响 |
|---|---|---|
| 权重量化 | 权重被离散化 | 会跨层累积，影响整体稳定性 |
| 激活量化 | 动态范围被截断或压缩 | 对视觉层尤其敏感 |
| 反量化恢复 | 缩放因子估计不准 | 会放大前两类误差 |

更细一点，可以把量化误差拆成：

$$
W = s_w W_q + \delta_W,\quad x = s_x x_q + \delta_x
$$

代回原式：

$$
Wx = (s_w W_q + \delta_W)(s_x x_q + \delta_x)
$$

展开后得到：

$$
Wx = s_w s_x W_q x_q + s_w W_q \delta_x + s_x \delta_W x_q + \delta_W \delta_x
$$

其中：

- $s_w s_x W_q x_q$ 是量化后的主项
- $s_w W_q \delta_x$ 是激活误差传播项
- $s_x \delta_W x_q$ 是权重误差传播项
- $\delta_W \delta_x$ 是高阶交叉项

这组展开式解释了为什么“视觉和语言一起激进量化”风险更高：如果权重和激活都压得很低，交叉误差项会变大，而多模态模型恰好又在对齐层上很怕这种误差耦合。

AWQ，Activation-aware Weight Quantization，白话讲是“看激活的重要性来决定权重怎么量”。它优化的不是单独的权重误差，而是量化后对真实输入输出的影响。抽象目标可写成：

$$
\min_{\hat W} \|(W-\hat W)X\|
$$

这里 $X$ 是校准集上的激活样本。意思是：即使某些权重本身误差稍大，只要经过真实激活后输出误差仍小，也可以接受。AWQ 的关键不在“每个权重都尽量准”，而在“重要通道优先保住”。

GPTQ，Generative Pre-trained Transformer Quantization，则更强调二阶信息近似。它通常利用 Hessian 近似来估计不同方向上的量化误差代价，目标常写成局部二次近似：

$$
\Delta \mathcal{L} \approx \frac{1}{2}(w-\hat w)^T H (w-\hat w)
$$

其中 $H$ 是 Hessian 近似矩阵，表示损失对参数扰动的敏感程度。若某个方向曲率更大，说明这个方向上哪怕很小的量化误差，也可能显著影响输出，因此需要优先补偿。对初学者来说，可以把它理解成：**不是所有参数都同样重要，GPTQ 会尽量把“危险的误差”留小。**

为什么 AWQ 和 GPTQ 常先在语言层见效？原因有三点：

| 原因 | 语言层 | 视觉层 |
|---|---|---|
| 主体算子是否是大线性层 | 是 | 不完全是 |
| 冗余是否更高 | 更高 | 更低 |
| 对局部细节是否极敏感 | 相对不敏感 | 非常敏感 |

再给一个玩具数值例子。若激活向量为

$$
x=[0.1, 0.2, 2.8]
$$

并使用对称 INT4，取 $Q_{\max}=7$，则

$$
s_x = \frac{2.8}{7}=0.4
$$

量化后：

$$
x_q = \text{round}(x / 0.4) = [0,1,7]
$$

反量化得到：

$$
\hat x = [0,0.4,2.8]
$$

可以看到，最大的值 2.8 基本保住了，但 0.1 直接变成 0，0.2 被粗糙地变成 0.4。若这些小值刚好对应视觉边缘、细字笔画或图表刻度线，那么信息损失就会很明显。

再看一个对比：

| 原始值 | INT4 反量化值 | 绝对误差 | 可能语义影响 |
|---:|---:|---:|---|
| 0.1 | 0.0 | 0.1 | 小响应直接丢失 |
| 0.2 | 0.4 | 0.2 | 幅度被放大 |
| 2.8 | 2.8 | 0.0 | 大值保留较好 |

这就是视觉侧量化更难的原因之一：**对视觉来说，小值不一定不重要；它们可能是结构信息本身。**

---

## 代码实现

下面给出一个可以直接运行的最小 Python 例子，演示对线性层做对称量化。代码重点覆盖四件事：缩放、量化、反量化、误差检查。为了避免“代码能看但跑不通”，这里把必要的辅助函数都补齐，并且给出一组可复现的输出逻辑。

```python
import numpy as np


def symmetric_quantize(x, bits=8):
    """
    对称量化:
    把浮点数组 x 映射到 [-qmax, qmax] 的整数区间。
    返回:
        q: 量化后的整数数组
        scale: 缩放因子
    """
    if bits < 2:
        raise ValueError("bits must be >= 2")

    qmax = 2 ** (bits - 1) - 1
    max_abs = float(np.max(np.abs(x)))

    # 全零输入时给 1.0，避免除零
    scale = max_abs / qmax if max_abs > 0 else 1.0
    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax).astype(np.int32)
    return q, scale


def dequantize(q, scale):
    """把整数张量恢复成近似浮点张量。"""
    return q.astype(np.float32) * np.float32(scale)


def linear_forward(W, x, b):
    """原始浮点线性层前向: y = Wx + b"""
    return W @ x + b


def quantized_linear_forward(W, x, b, w_bits=4, x_bits=4):
    """
    量化版线性层前向:
    1. 分别量化权重和激活
    2. 在整数域执行矩阵乘法
    3. 通过 sw * sx 反量化
    """
    Wq, sw = symmetric_quantize(W, bits=w_bits)
    xq, sx = symmetric_quantize(x, bits=x_bits)

    # int32 足够容纳这个玩具例子的整数乘加
    y_int = Wq @ xq
    y = dequantize(y_int, sw * sx) + b.astype(np.float32)
    return y, {
        "Wq": Wq,
        "xq": xq,
        "sw": sw,
        "sx": sx,
        "y_int": y_int,
    }


def max_abs_error(a, b):
    return float(np.max(np.abs(a - b)))


def relative_l2_error(a, b, eps=1e-12):
    denom = float(np.linalg.norm(a) + eps)
    return float(np.linalg.norm(a - b) / denom)


def main():
    # 一个 2x3 线性层
    W = np.array(
        [
            [0.2, -0.5, 1.3],
            [1.0, -0.7, 0.1],
        ],
        dtype=np.float32,
    )
    x = np.array([0.1, 0.2, 2.8], dtype=np.float32)
    b = np.array([0.05, -0.02], dtype=np.float32)

    y_fp = linear_forward(W, x, b)
    y_q, info = quantized_linear_forward(W, x, b, w_bits=4, x_bits=4)

    abs_err = max_abs_error(y_fp, y_q)
    rel_err = relative_l2_error(y_fp, y_q)

    print("=== Floating-point forward ===")
    print(y_fp)

    print("\n=== Quantization metadata ===")
    print("Wq =\n", info["Wq"])
    print("xq =", info["xq"])
    print("sw =", info["sw"])
    print("sx =", info["sx"])
    print("integer matmul output =", info["y_int"])

    print("\n=== Quantized forward ===")
    print(y_q)

    print("\n=== Error metrics ===")
    print("max abs error =", abs_err)
    print("relative l2 error =", rel_err)

    # 基本可运行性和数值 sanity check
    assert y_fp.shape == y_q.shape
    assert np.isfinite(y_q).all()
    assert abs_err < 1.0, f"unexpectedly large error: {abs_err}"
    assert rel_err < 0.5, f"unexpectedly large relative error: {rel_err}"


if __name__ == "__main__":
    main()
```

如果手算一遍，你会发现这个例子里第三个输入分量 2.8 决定了缩放因子，而较小的分量被映射得很粗。这正是视觉侧量化的核心难点。对于语言模型，大量线性层和残差结构通常会让这种粗糙映射的影响没那么直接；但对视觉细节，它可能立刻变成可见的信息损失。

再给一个初学者更容易对应工程现实的解释：

| 步骤 | 代码里的实现 | 工程里的对应动作 |
|---|---|---|
| 统计范围 | `np.max(np.abs(x))` | 用校准集收集每层激活范围 |
| 生成整数表示 | `np.round(x / scale)` | 权重/激活离散化 |
| 整数乘加 | `Wq @ xq` | 调用低位 GEMM / kernel |
| 恢复数值 | `scale * q` | 反量化或融合缩放 |
| 验证误差 | `max_abs_error` | 对比任务指标和层输出 |

真实工程里，流程不会只量一个层，而是遍历模型中的所有线性层，并使用一小批校准样本收集激活统计。校准集不是训练集，它的作用不是更新参数，而是回答一个更实际的问题：**“线上真实输入大概长什么样？”**

下面给出更接近工程的伪代码，补全为可读的流程版本：

```python
def collect_activation_stats(model, calib_loader):
    """
    遍历校准集，记录每层激活的统计信息。
    常见统计包括:
    - amax: 最大绝对值
    - mean_abs: 平均绝对值
    - p99: 99分位数，用于减少极端值影响
    """
    stats = {}

    for layer in model.linear_layers():
        stats[layer.name] = {
            "amax": 0.0,
            "mean_abs": 0.0,
            "num_batches": 0,
        }

    for batch in calib_loader:
        activations = model.forward_collect(batch)
        for name, act in activations.items():
            amax = float(abs(act).max())
            mean_abs = float(abs(act).mean())

            stats[name]["amax"] = max(stats[name]["amax"], amax)
            stats[name]["mean_abs"] += mean_abs
            stats[name]["num_batches"] += 1

    for name, item in stats.items():
        if item["num_batches"] > 0:
            item["mean_abs"] /= item["num_batches"]

    return stats


def quantize_weight(weight, bits, method="gptq"):
    if method == "gptq":
        return gptq_quantize(weight, bits)
    elif method == "awq":
        return awq_quantize(weight, bits)
    else:
        return naive_quantize(weight, bits)


def apply_compensation(layer, act_stats, method="awq"):
    """
    用校准统计对量化层做补偿。
    实际实现会更复杂，这里只表达职责边界。
    """
    if method == "awq":
        layer.weight = awq_rescale(layer.weight, act_stats)
    elif method == "gptq":
        layer.weight = gptq_error_correction(layer.weight)


def replace_forward_with_quantized(layer):
    layer.forward = layer.quantized_forward


def quantize_model(model, calib_loader, method="gptq", bits=4):
    stats = collect_activation_stats(model, calib_loader)

    for layer in model.linear_layers():
        layer.qweight, layer.scale = quantize_weight(
            layer.weight, bits=bits, method=method
        )
        apply_compensation(layer, stats[layer.name], method=method)
        replace_forward_with_quantized(layer)

    return model
```

各函数职责可以压缩成下面这张表：

| 函数 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `collect_activation_stats` | 模型、校准集 | 每层激活统计 | 为缩放和异常值处理提供依据 |
| `quantize_weight` | 浮点权重、位宽、方法 | 整数权重与 scale | 真正执行权重量化 |
| `apply_compensation` | 层、激活统计 | 修正后的量化层 | 用 AWQ/GPTQ 降低输出误差 |
| `replace_forward_with_quantized` | 原始层 | 量化前向实现 | 让推理路径真正切换到低位 |
| `quantize_model` | 模型、校准集 | 量化后的模型 | 串起完整流程 |

如果把这套流程放到真实 VLM 上，常见做法不是一次性把所有模块都变成 INT4，而是分阶段推进：

1. 先把语言解码器做 W8A8、W4A16 或 W4A8。
2. 测多模态基准，看是否出现 OCR、图表、细节理解退化。
3. 若结果稳定，再考虑投影层或部分融合层。
4. 最后才评估视觉主干是否值得进一步下压。

这套顺序不是保守主义，而是误差传播路径决定的。ViT 处在信息链路前端，出错会一路带到对齐层和语言侧；LLM 在链路后端，且参数量更大、收益更高，所以更适合作为第一量化目标。

---

## 工程权衡与常见坑

工程上最重要的问题不是“能不能量化”，而是“应该先量哪里、量到什么程度、用什么验证”。经验顺序通常是：**LLM 先行，Fusion 次之，ViT 最后**。原因很直接，语言层通常占据主要参数量和大部分算力开销，压缩收益最大；同时它对低位更耐受，失败成本更低。

公开实践中，使用 `LLM Compressor + vLLM` 对 Pixtral、Qwen2.5-VL 这类模型进行 8-bit 或 4-bit 压缩时，通常可以在较高恢复率下换取明显吞吐提升。需要注意的是，这类结果不能直接理解成“任何 VLM 都能无脑压到 INT4”。它真正说明的是：**在合适的模块选择、校准数据和推理框架支持下，量化已经是可上线的工程手段，而不只是论文技巧。**

常见坑可以先用一张表说明：

| 常见坑 | 现象 | 根因 | 规避策略 |
|---|---|---|---|
| ViT 和 LLM 同时激进量化 | 精度突然塌陷 | 视觉误差与语言误差在对齐层叠加 | 先只量 LLM，再逐步扩展 |
| 校准集太小或偏分布 | INT4 结果不稳定 | 缩放因子不代表真实线上输入 | 校准集覆盖 OCR、图表、自然图像 |
| 直接照搬 AWQ 到视觉层 | 局部视觉细节明显退化 | 视觉层统计特性不同于 LLM | 视觉侧优先 INT8 或保持 FP16 |
| 只看 perplexity 不看 VLM 任务 | 文本正常，视觉问答变差 | 多模态对齐没有被评估 | 必测 MMMU、DocVQA、ChartQA 等 |
| 忽略异常值 | 个别样本退化严重 | 少数极端激活撑大缩放范围 | 做分组量化、通道级缩放、异常值保护 |
| 框架不支持高效 INT4 内核 | 理论省显存，实测吞吐一般 | 算子没有真正优化 | 先确认推理框架和硬件内核支持 |
| 只量权重不看 KV cache | 长上下文仍然很吃显存 | 显存瓶颈不只在参数 | 同时评估权重、激活、KV cache 策略 |

这里有一个新手很容易忽略的点：**INT4 的问题通常不体现在平均样本上，而体现在长尾样本上。** 平时看十张图都正常，并不代表方案可以上线。第十一张带细小表格、复杂背景、低对比度文字的图片突然失败，往往不是“模型随机出错”，而是量化缩放没有覆盖这类输入分布。

再看一个更贴近线上场景的判断表：

| 业务类型 | 量化容忍度 | 推荐策略 |
|---|---|---|
| 图片描述、一般视觉问答 | 中高 | 先量 LLM 到 INT4，再评估 |
| 文档 OCR | 低 | ViT 保 FP16/BF16，LLM 可 INT8/INT4 |
| 图表理解 | 低到中 | 融合层保高精度，视觉侧慎动 |
| 多图长上下文对话 | 中 | 关注 LLM 与 KV cache 压缩 |
| 医学影像或工业质检 | 很低 | 优先混合精度或 QAT |

另一个实际权衡是吞吐和实现复杂度。INT4 理论压缩率更高，但如果推理框架对该格式支持不好，实际收益可能不如成熟的 INT8。原因不在算法，而在工程栈：低位算子是否有高效 kernel、是否支持 fused dequant、是否能和现有 batch 策略配合，都会直接决定线上表现。

因此，量化方案不能只看论文表格，还要看下面三件事：

1. 你的硬件是否对 INT4 / FP8 / INT8 有成熟支持。
2. 你的推理框架是否真的提供高效实现，而不是只是“能跑”。
3. 你的评测是否覆盖了业务里最脆弱的输入类型。

如果只满足第一条，不满足后两条，理论收益很容易变成部署复杂度。

---

## 替代方案与适用边界

如果任务对视觉细节极敏感，比如文档 OCR、医学图像、图表理解，那么最稳妥的方案通常不是“全 INT4”，而是混合精度。所谓混合精度，就是不同模块采用不同位宽，把低位优先用在最耐压、收益最大的部分。

常见组合如下：

| 方案 | 组合 | 适用场景 | 风险 |
|---|---|---|---|
| LLM-only INT4 | ViT FP16, Fusion FP16, LLM INT4 | 显存紧张，先追吞吐 | 视觉侧显存收益有限 |
| LLM INT8 + ViT FP16 | ViT 保高精度，LLM 温和压缩 | 精度优先的在线服务 | 压缩率不如 INT4 |
| LLM INT4 + Fusion FP16 | 语言激进压缩，对齐层保守 | 图文对齐较敏感模型 | 融合层仍占部分显存 |
| 全 INT8 | ViT/Fusion/LLM 都做 INT8 | 需要统一格式、控制复杂度 | 收益中等但更稳 |
| 全 INT4 但保关键层高精度 | 首层、末层、Fusion FP16 | 已有充分校准资源 | 调参与回归成本高 |
| QAT | 量化感知训练 | 允许再训练、任务定制 | 成本最高 |

“语言先行量化”是一条很实用的保守路线：

1. 先对 LLM 层应用 AWQ 或 GPTQ。
2. 保留 ViT 和 Fusion 为 FP16/BF16。
3. 在多模态任务上验证恢复率，而不是只看文本指标。
4. 若仍需压缩，再尝试把投影层或部分视觉层降到 INT8。
5. 只有在校准覆盖充分、框架支持成熟时，才考虑更深的视觉侧低位化。

这条路线的价值在于，它先压缩最耐压、收益最大的模块，同时保住最脆弱的视觉链路。初学者可以把它理解成“先动风险最低的地方，再逐步扩展影响范围”。

如果没有可用校准集，或者业务不允许离线跑激活统计，那么更适合退回到 INT8，或者只做权重量化、不做激进激活量化。因为激活量化比权重量化更依赖真实输入分布，缺少校准样本时更容易在长尾输入上失败。

若任务高度依赖小目标、细文字、颜色差异，量化感知训练（QAT）通常会比纯后训练量化（PTQ）更可靠。两者的差别可以用一张表概括：

| 方案 | 是否需要再训练 | 工程成本 | 精度上限 | 适合场景 |
|---|---|---:|---:|---|
| PTQ | 否 | 低 | 中高 | 快速部署、通用模型 |
| QAT | 是 | 高 | 更高 | 专用任务、严格精度要求 |

还可以补充两个常见替代方向：

| 替代方向 | 核心思路 | 适用边界 |
|---|---|---|
| KV cache 量化 | 压缩长上下文推理中的缓存 | 长对话、多图场景收益明显 |
| 结构化裁剪 / 蒸馏 | 减少参数规模而不是只降位宽 | 允许重训练或模型重构时更合适 |

因此，量化不是唯一的压缩手段。若瓶颈主要来自长上下文缓存，而不是模型权重，那么只做权重量化未必能解决问题；若业务允许离线训练，蒸馏和 QAT 可能比激进 PTQ 更稳。

总结成一句工程决策规则就是：**先找真正的瓶颈，再选最匹配的压缩手段，不要默认“全模型 INT4”就是最优解。**

---

## 参考资料

- [Red Hat Developer] `Enable 3.5× Faster Vision-Language Models with Quantization`：用于理解 `vLLM + LLM Compressor` 在 Pixtral、Qwen2.5-VL 等模型上的恢复率与吞吐表现，重点看模块化量化流程与部署收益。
- [The Moonlight] `Towards Understanding Best Practices for Quantization of Vision-Language Models`：适合建立整体判断，尤其是“视觉编码器更敏感、语言模型更适合优先低位化”的经验总结。
- [OpenLM] `Qwen2-VL` 量化 benchmark：可直接对比 Qwen2-VL-72B 在 BF16、GPTQ-INT8、GPTQ-INT4、AWQ-INT4 等设置下的指标变化，适合作为部署前的精度参照。
- [AWQ 论文与 Mit Han Lab 相关实现/说明]：重点看 AWQ 如何利用激活分布保护重要通道，以及它为什么在以大线性层为主体的模块上更有效。
- [GPTQ 论文与开源实现]：重点理解 Hessian 近似、逐块量化与误差补偿，而不是把 GPTQ 只理解成“简单四舍五入”。
- [bitsandbytes / AutoGPTQ / llm-compressor / vLLM 文档]：适合落地实现时查看格式支持、kernel 路径和实际部署约束。
- [DocVQA / ChartQA / MMMU 等公开基准]：用于补足评测维度，避免只看语言侧指标而误判多模态量化效果。

参考资料的使用建议也可以先按下面的顺序：

| 目的 | 先看什么 |
|---|---|
| 建立总体判断 | VLM 量化 best practice 总结 |
| 理解算法差异 | AWQ 与 GPTQ 原理材料 |
| 看公开结果 | Qwen2-VL 等模型 benchmark |
| 做工程落地 | vLLM、llm-compressor、相关框架文档 |
| 做业务回归 | MMMU、DocVQA、ChartQA 等任务基准 |
