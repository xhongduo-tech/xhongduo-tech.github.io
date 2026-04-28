## 核心结论

边缘模型量化加速的目标，不是“把模型文件变小”本身，而是让推理真正走到低位宽执行路径。这里的“低位宽”可以先理解成“用更少比特表示数值”，例如 `INT8` 用 8 位整数替代 `FP32` 的 32 位浮点。只有当两层条件同时满足时，量化才通常会带来真实加速：

1. 数值表示已经变成低位宽，例如权重和激活都变成 `int8`。
2. 算子执行也确实命中目标硬件的原生 kernel，例如手机 SoC 的 NPU、DSP，或者 CPU 的 SIMD 整数指令。

新手最容易混淆的一点是：`FP32 -> INT8` 不等于一定更快。如果模型虽然存成 `int8`，但推理时先 `dequantize` 回浮点，再按浮点卷积去算，那么这主要是压缩，不一定是加速。

量化的核心公式只有两条：

$$
q = clamp(round(x / s) + z)
$$

$$
x \approx s \cdot (q - z)
$$

其中 `scale` 是缩放因子，可以理解成“一个整数单位对应多少真实数值”；`zero_point` 是零点，可以理解成“浮点 0 在整数空间落在哪个位置”。

INT8 往往是收益、精度、工程复杂度三者之间最平衡的选择。INT4 更激进，常用于进一步压缩参数或降低带宽压力，但更依赖校准质量、算子支持和硬件后端实现。很多真实项目里，如果瓶颈已经是内存带宽而不是纯算力，量化往往比单纯换一个更小模型更有效，因为它直接减少了“搬数据”的成本。

| 数值格式 | 每个参数占用 | 相对 FP32 大小 | 典型收益 | 常见适用场景 |
| --- | --- | --- | --- | --- |
| FP32 | 4 字节 | 1x | 精度基线，兼容性最好 | 训练、未优化推理 |
| FP16 | 2 字节 | 0.5x | 带宽下降，部分硬件加速明显 | GPU、支持半精度的移动端 |
| INT8 | 1 字节 | 0.25x | 压缩和加速较平衡 | 边缘推理默认起点 |
| INT4 | 0.5 字节 | 0.125x | 更激进压缩，带宽收益大 | weight-only、特定 MatMul 场景 |

---

## 问题定义与边界

讨论“边缘量化加速”时，必须区分三个层次：模型压缩、算子加速、端到端加速。

| 概念 | 关注点 | 典型现象 | 是否等于整体更快 |
| --- | --- | --- | --- |
| 模型压缩 | 文件更小、内存更省 | 权重从 100MB 变成 25MB | 不一定 |
| 算子加速 | 单个卷积或矩阵乘更快 | `Conv`/`MatMul` 的 kernel 延迟下降 | 不一定 |
| 端到端加速 | 整体推理链路更快 | 从输入到输出总时延下降 | 才是最终目标 |

这三个概念不是一回事。一个模型可以压缩成功，但没有算子加速；也可以局部算子加速成功，但因为前后处理、内存拷贝、回退算子太多，端到端收益仍然有限。

本文边界只讨论部署相关量化，不讨论训练阶段的大规模分布式压缩优化，重点包括：

- `post-training quantization`：训练后量化，指模型训练完再做量化转换。
- `static quantization`：静态量化，指先用代表性数据校准激活范围，再固定量化参数。
- `dynamic quantization`：动态量化，指推理时再按输入动态估计部分激活范围。
- `weight-only quantization`：仅权重量化，激活仍保持更高精度，常用于大模型推理和带宽优化。

一个新手版反例很典型：模型文件已经是 `int8`，但执行器不支持 `int8 Conv`，于是每一层前后都做 `dequantize -> float op -> quantize`。这种情况属于“压缩成功”，不属于“量化加速成立”。

---

## 核心机制与推导

量化的本质，是用离散整数近似连续浮点数。整数空间不连续，所以必须借助 `scale` 和 `zero_point` 做映射。

量化公式：

$$
q = clamp(round(x / s) + z)
$$

反量化公式：

$$
x \approx s \cdot (q - z)
$$

这里 `clamp` 表示截断到合法整数范围，例如 `INT8` 常见范围是 $[-128,127]$。一个玩具例子：

- 设激活 $x=0.86$
- 设步长 $s=0.02$
- 设零点 $z=3$

则：

$$
q = round(0.86 / 0.02) + 3 = 46
$$

反量化后：

$$
x' = 0.02 \cdot (46 - 3) = 0.86
$$

这说明整数路径可以逼近原浮点值。硬件真正受益的关键，在于卷积和矩阵乘这类核心算子可以转成整数乘加。点积公式通常写成：

$$
y \approx s_a \cdot s_w \cdot \sum (q_a - z_a)(q_w - z_w)
$$

其中：

- `a` 表示激活。
- `w` 表示权重。
- 累加通常会放在更高位宽寄存器里，例如 `int32`，避免溢出。

研究摘要里的最小例子可以直接看出机制：

- 激活：`x=[0.86, -0.14]`，`s_a=0.02`，`z_a=3`
- 权重：`w=[0.50, -0.10]`，`s_w=0.01`，`z_w=0`

量化后：

- `q_a=[46, -4]`
- `q_w=[50, -10]`

整数点积：

$$
(46-3)\times 50 + (-4-3)\times(-10) = 2220
$$

反量化：

$$
2220 \times 0.02 \times 0.01 = 0.444
$$

原始浮点点积：

$$
0.86 \times 0.50 + (-0.14)\times(-0.10) = 0.444
$$

两者一致。这个例子足够小，能直接说明“整数算子为什么可行”。

工程上还要区分几组常见设计：

| 设计项 | 方案 A | 方案 B | 工程影响 |
| --- | --- | --- | --- |
| 量化方式 | 对称量化 | 非对称量化 | 对称量化常更利于权重整数 kernel，非对称更适合偏移明显的数据 |
| scale 粒度 | per-tensor | per-channel | 按通道通常精度更好，特别是卷积权重 |
| 激活处理 | static calibration | dynamic quantization | 静态量化更适合边缘部署，动态更灵活但不一定最快 |

真实工程里，常见组合是：权重对称量化、按通道 `scale`、激活静态校准。这是因为卷积权重在不同输出通道上分布差异明显，`per-channel` 能更稳定地保精度，同时更容易让后端调用原生 `int8 kernel`。

如果把执行链路画成两张图，第一张应是“浮点数 -> 整数编码 -> 反量化输出”；第二张应是“浮点算子链路 vs 整数量化链路”。后者最重要的信息不是公式，而是看中间有没有插入大量 `Quantize/Dequantize` 节点。节点越多，越说明执行路径不干净。

---

## 代码实现

先给一个最小可运行的玩具实现，演示量化、整数点积、反量化的全过程。

```python
import math

def quantize(values, scale, zero_point, qmin=-128, qmax=127):
    qs = []
    for x in values:
        q = round(x / scale) + zero_point
        q = max(qmin, min(qmax, q))
        qs.append(int(q))
    return qs

def dequantize(qvalues, scale, zero_point):
    return [scale * (q - zero_point) for q in qvalues]

def int_dot(qx, qw, za, zw):
    acc = 0
    for a, w in zip(qx, qw):
        acc += (a - za) * (w - zw)
    return acc

# 玩具例子
x = [0.86, -0.14]
w = [0.50, -0.10]

sa, za = 0.02, 3
sw, zw = 0.01, 0

qx = quantize(x, sa, za)
qw = quantize(w, sw, zw)

acc = int_dot(qx, qw, za, zw)
y_hat = sa * sw * acc
y_fp = sum(a * b for a, b in zip(x, w))

assert qx == [46, -4]
assert qw == [50, -10]
assert abs(y_hat - y_fp) < 1e-6

print("quantized x:", qx)
print("quantized w:", qw)
print("int accumulator:", acc)
print("dequantized dot:", y_hat)
print("float dot:", y_fp)
```

这个例子只展示数值机制，不代表完整部署。真实部署还涉及图优化、算子融合、校准集和后端映射。

真实工程例子可以看 TFLite 的 full integer quantization。思路是：先准备代表性样本统计激活分布，再把权重和激活都量化成 `int8`，最后强制导出支持整数推理的模型。

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")

def representative_dataset():
    for sample in calibration_samples:  # 代表性输入样本
        yield [sample]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)
```

不同框架的量化入口不同，但核心问题相同：有没有校准、是不是全整数、后端能不能真正执行低位宽算子。

| 框架 | 常见入口 | 典型部署目标 | 关注点 |
| --- | --- | --- | --- |
| TensorFlow Lite | `TFLiteConverter` | 手机、嵌入式 | delegate 是否支持 int8 |
| ONNX Runtime | `quantize_static` / `quantize_dynamic` | CPU、边缘设备 | EP 是否支持量化算子 |
| TensorRT | 显式量化或校准构建 engine | NVIDIA GPU/Jetson | 是否命中量化层与 Tensor Core |

例如 ONNX Runtime 静态量化通常类似：

```python
from onnxruntime.quantization import quantize_static, QuantType

quantize_static(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    calibration_data_reader=my_reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
```

代码写完只是第一步。真正重要的是验证执行日志，确认哪些层走了 `int8`，哪些层回退到了浮点。

---

## 工程权衡与常见坑

最大的工程风险通常不是“量化误差本身”，而是“算子没命中原生 kernel”。如果主干卷积是 `int8`，但 `Resize`、`Concat`、`CustomOp`、后处理 NMS 仍在浮点路径，整体收益会被明显吃掉。

| 坑点 | 现象 | 规避方法 |
| --- | --- | --- |
| 只量化权重，不量化激活 | 模型变小，但时延下降有限 | 先确认目标后端是否支持全量 `int8` |
| 校准集不代表真实输入 | 离线指标正常，线上精度掉 | 校准样本覆盖真实分布、亮度、尺寸、场景 |
| 部分算子回退浮点 | 图里插入大量 `Dequantize` | 检查算子覆盖率，必要时改模型结构 |
| 盲目上 INT4 | 精度掉幅大，适配成本高 | 先以 INT8 为基线，再评估 INT4 |
| 忽略硬件代际差异 | 新设备快，老设备几乎无收益 | 分设备 profiling，不要只测单机 |

经验规则很直接：校准集越覆盖真实分布，`scale` 越稳定，激活截断越少，精度越不容易掉。换成数学语言，就是激活真实分布与校准分布的偏差越小，量化映射误差越可控。

一个真实工程例子是手机端目标检测。你把 backbone 和 head 都量化到 `int8`，但后处理里还有浮点解码和 NMS。结果可能出现两种情况：

1. 主干延迟明显下降，但端到端只快了 15%。
2. 模型体积降很多，但功耗收益一般，因为 CPU 还在跑一段浮点后处理。

因此检查日志很关键。不同后端日志格式不同，但你至少应确认：

```text
Conv2D -> int8 kernel
DepthwiseConv2D -> int8 kernel
Add -> int8 kernel
ResizeBilinear -> float fallback
CustomNMS -> float fallback
```

看到这种输出时，结论不应是“量化没用”，而应是“量化命中不完整”。解决方向包括替换不支持的算子、做图融合、调整模型结构，或者接受局部收益。

INT8 是默认起点，因为它通常在精度、兼容性、工具链成熟度上最稳。INT4 更适合以下场景：模型特别大、带宽瓶颈明确、后端明确支持低位宽权重加载，且团队能承担更严格的回归测试。

---

## 替代方案与适用边界

量化不是唯一手段。边缘部署常见优化方案还有剪枝、蒸馏、低秩分解、权重共享和纯 FP16 推理。选择标准不是“哪个更先进”，而是“哪个最符合设备能力和业务指标”。

| 方案 | 优点 | 主要代价 | 更适合什么情况 |
| --- | --- | --- | --- |
| INT8 全量量化 | 速度和体积较平衡 | 需要算子覆盖和校准 | 大多数边缘推理 |
| INT4 weight-only | 压缩率高，省带宽 | 精度与适配风险更高 | 大模型、内存受限场景 |
| FP16 | 精度更稳，很多 GPU 友好 | 压缩不如整数激进 | 设备 FP16 支持强但 INT8 一般 |
| 剪枝 | 可减少计算量 | 稀疏收益依赖后端 | 结构可重训、后端支持稀疏时 |
| 蒸馏 | 保精度能力强 | 需要重新训练 | 追求稳健精度时 |

可以用一个简化选择表来判断：

| 条件 | 更优先方案 |
| --- | --- |
| 设备有成熟 `int8` NPU/DSP | INT8 全量量化 |
| 设备 FP16 支持强，INT8 支持弱 | FP16 |
| 模型超大，瓶颈是权重搬运 | INT4 weight-only |
| 业务对精度非常敏感 | FP16 或蒸馏 |
| 模型含大量不支持量化的自定义算子 | 先改结构，或考虑非量化方案 |

新手版判断可以记成一句话：先看设备支持什么，再看你能容忍多少精度损失，最后看模型图里有多少算子真的能走低位宽路径。没有硬件支持的量化，常常只是格式转换；没有校准质量保证的量化，常常只是风险放大。

---

## 参考资料

下表按“规范、工具链、论文”分组，便于从定义一路查到实现。

| 资料名称 | 类型 | 适合阅读目的 |
| --- | --- | --- |
| TensorFlow Lite 8-bit quantization specification | 规范 | 确认 TFLite `int8` 数值约定与算子要求 |
| TensorFlow Lite Post-training Quantization | 工具链 | 查训练后量化的实际转换方法 |
| ONNX Runtime Quantize ONNX Models | 工具链 | 查 ONNX 静态/动态量化入口与限制 |
| TensorRT Working with Quantized Types | 工具链 | 查 NVIDIA 平台量化部署细节 |
| A White Paper on Neural Network Quantization | 论文综述 | 系统理解误差来源、方案分类和设计权衡 |

1. [TensorFlow Lite 8-bit quantization specification](https://www.tensorflow.org/lite/performance/quantization_spec)
2. [TensorFlow Lite Post-training Quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)
3. [ONNX Runtime: Quantize ONNX Models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
4. [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.15.1/inference-library/work-quantized-types.html)
5. [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)
