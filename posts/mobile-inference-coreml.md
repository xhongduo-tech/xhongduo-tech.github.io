## 核心结论

Core ML 在移动端推理里的核心价值，不是把模型文件变成 `.mlmodel`，而是把模型图里的算子尽可能编译到更快、更省电的硬件后端。这里的“算子”可以理解成模型里的最小计算单元，比如矩阵乘、卷积、归一化、激活函数。

对新手最重要的一句话是：**同样是一个模型，能被 Core ML 识别并拆成标准算子的部分，会跑得很快；不能识别的注意力变体、动态 shape、自定义层，往往会掉到通用路径，速度、功耗和温度都会变差。**

这也是为什么“转换成功”不等于“性能达标”。一个 100 层模型成功转成 Core ML，只能说明格式上能加载，不代表 100 层都跑在最强硬件上。真正决定体验的是三件事：

| 因素 | 它实际决定什么 | 结果 |
|---|---|---|
| 算子是否可编译 | 能不能被 Core ML 编译器识别成标准执行图 | 决定是否有机会用高性能后端 |
| shape 是否稳定 | 输入尺寸是否可预测、可约束 | 决定编译器能否做静态优化 |
| 是否发生回退 | 某些层是否退回通用执行路径 | 决定尾延迟、功耗、温度 |

下面这张后端映射示意表可以先建立直觉：

| 后端 | 白话解释 | 适合的算子类型 | 常见限制 | 性能特征 |
|---|---|---|---|---|
| `ANE` | Apple Neural Engine，苹果的神经网络专用芯片 | 标准神经网络算子、规则张量流 | 对动态性、自定义行为更敏感 | 延迟低、功耗低，通常最优 |
| `GPU` | 图形处理器，也可做并行数值计算 | 并行度高、规则性较强的张量计算 | 某些控制流或特殊算子不理想 | 吞吐较好，功耗通常高于 ANE |
| `CPU` | 通用处理器，兼容性最强 | 不规则逻辑、回退算子、自定义层 | 并行能力和专用性较弱 | 最稳但通常最慢，功耗和发热可能更差 |

---

## 问题定义与边界

本文讨论的“移动端推理 Core ML”，指的是 **iPhone / iPad 等 Apple 设备上，模型在端侧本地执行的推理链路**。这里的“推理”就是模型已经训练好，只做前向计算，把输入变成输出，不涉及训练、反向传播，也不讨论服务端部署。

本文关注的问题边界是：

1. 模型如何从原始框架进入 Core ML。
2. 模型图中的算子如何被编译。
3. 编译后的算子如何映射到 `ANE / GPU / CPU`。
4. 为什么有些节点会发生回退，以及回退后性能为什么明显下降。

不展开的内容包括：iOS UI、App 架构、训练流程、分布式推理、服务端调度。

为了避免概念混淆，先给出术语表：

| 术语 | 简明定义 |
|---|---|
| 模型转换 | 把 PyTorch、TensorFlow 等模型表示转成 Core ML 可接受的模型格式 |
| 算子编译 | 把模型里的计算节点分析、重写、融合成可执行图 |
| 后端映射 | 给每个可执行节点选择实际运行硬件，如 `ANE / GPU / CPU` |
| 回退路径 | 某节点无法走理想后端时，退回更通用执行方式 |
| 动态 shape | 输入尺寸不固定，比如序列长度会变 |
| custom layer | 自定义层，指 Core ML 原生不能直接表达、需要开发者补实现的层 |

一个边界示例很重要：**一个 100 层模型成功转换成 Core ML，不代表这 100 层都跑在 ANE 上。** 有可能只有大部分卷积或线性层被编译到高性能后端，而注意力中的某些变体、动态索引、特殊 reshape 或自定义层退回了 CPU。这种情况下，文件转换已经“成功”，但实际端侧体验可能依旧很差。

---

## 核心机制与推导

Core ML 推理性能可以用一个统一记号描述。设第 $i$ 个算子的运行后端为：

$$
b_i \in \{ANE, GPU, CPU\}
$$

第 $i$ 个算子的耗时不是固定常数，而是由后端、输入 shape、内存布局共同决定：

$$
t_i = t_i(b_i, shape_i, layout_i)
$$

整个模型总耗时可以写成：

$$
T_{total} = \sum_{i=1}^{N} t_i + T_{bridge}
$$

这里的 $T_{bridge}$ 是跨后端切换、数据搬运、图边界处理等额外成本。白话说，哪怕单个回退层本身不算特别慢，只要它让图被切碎，来回切换硬件，也会把总延迟拉高。

### 为什么 shape 稳定更重要

“shape”就是张量维度，比如批大小、通道数、序列长度。编译器最喜欢的是静态 shape，因为这样可以提前决定内存布局、算子融合方式和执行计划。`RangeDim` 的意思是“维度允许变化，但变化范围有边界”。对于编译器来说，有界范围通常比无界范围更容易优化。

可以把常见情况总结成下面这张推导表：

| 条件 | 编译器行为 | 后端结果 | 性能影响 |
|---|---|---|---|
| 静态 shape | 最容易做融合与静态规划 | 更容易上 `ANE` | 延迟和功耗通常最好 |
| 有界 `RangeDim` | 可做部分静态优化 | 可能仍能上 `ANE/GPU` | 比无界动态 shape 更稳 |
| 无界动态 shape | 优化空间变小 | 更容易走保守路径 | 延迟抖动、热量上升 |
| 能表达成 composite operator | 可组合成标准图 | 更容易映射专用后端 | 通常优于自定义实现 |
| 只能写 custom layer | 编译器难以深度理解 | 容易走通用执行路径 | 兼容性有了，性能常变差 |

这里的“composite operator”可以理解成“组合算子”，即把复杂操作拆成 Core ML 能理解的一组标准算子。如果编译器能看懂这组结构，就还有机会继续优化；如果只能塞一个黑盒自定义层，后端优化空间就会大幅下降。

### 玩具例子：10 个回退节点如何拖慢整体

假设一个模型一共有 100 个算子：

- 90 个算子能跑在 `ANE`，每个耗时 `0.02 ms`
- 10 个算子因为动态 shape 回退到 `CPU`，每个耗时 `0.30 ms`
- 额外桥接开销 $T_{bridge} = 0.50 ms$

则总耗时为：

$$
T_{total} = 90 \times 0.02 + 10 \times 0.30 + 0.50 = 5.30 \text{ ms}
$$

如果那 10 个算子也能在 `ANE` 上执行：

$$
T_{total} = 100 \times 0.02 + 0.50 = 2.50 \text{ ms}
$$

差异不是一点点，而是接近翻倍。新手常犯的错误是只看“大多数层已经很快”，忽略少数回退层会制造明显尾延迟。

### 真实工程例子：手机端 LLM 解码

以手机端大语言模型解码为例。解码时输入长度会不断增长，注意力依赖 `KV cache`，也就是保存历史键值向量的缓存区。这个过程天然带来 shape 变化、内存增长和频繁访问。

如果：

1. 序列长度设置成无界动态范围；
2. 某个 attention 变体不能被 Core ML 组合优化；
3. KV cache 布局不利于专用后端；

那么常见后果是：

- 首 token 延迟看起来还能接受；
- 连续生成几十个 token 后温度明显上升；
- 设备触发降频，后续 token 更慢；
- 批处理能力被内存峰值压住。

内存约束可以粗略写成：

$$
M_{peak}(B) \approx M_{model} + B \cdot (M_{act} + M_{kv}) + M_{ws}
$$

其中 $B$ 是并发或批大小，$M_{act}$ 是激活内存，$M_{kv}$ 是 KV cache 内存，$M_{ws}$ 是工作区内存。移动端设备的可用内存远小于服务器，所以即使算子后端选择正确，过大的批量也可能根本跑不稳。

---

## 代码实现

工程上不能只做“导出模型”这一步，还要同时处理输入约束、执行后端选择和性能验证。下面先用一个可运行的 Python 玩具脚本，把前面的公式落成代码。

```python
def total_latency(num_ane_ops, num_cpu_ops, t_ane_ms, t_cpu_ms, t_bridge_ms):
    total = num_ane_ops * t_ane_ms + num_cpu_ops * t_cpu_ms + t_bridge_ms
    return round(total, 4)

slow = total_latency(90, 10, 0.02, 0.30, 0.50)
fast = total_latency(100, 0, 0.02, 0.30, 0.50)

assert slow == 5.3
assert fast == 2.5
assert slow > fast

def peak_memory_mb(model_mb, act_mb, kv_mb, workspace_mb, batch_size):
    return model_mb + batch_size * (act_mb + kv_mb) + workspace_mb

m1 = peak_memory_mb(model_mb=900, act_mb=40, kv_mb=60, workspace_mb=120, batch_size=1)
m4 = peak_memory_mb(model_mb=900, act_mb=40, kv_mb=60, workspace_mb=120, batch_size=4)

assert m1 == 1020
assert m4 == 1320
assert m4 > m1

print("slow(ms)=", slow)
print("fast(ms)=", fast)
print("mem_batch1(MB)=", m1)
print("mem_batch4(MB)=", m4)
```

上面代码虽然不是直接调用 Core ML，但它准确表达了两个工程事实：

1. 少量 CPU 回退会显著拉高总延迟。
2. 批量增大时，移动端峰值内存会很快吃满。

下面给出一个最小转换流程示例，展示“导出模型 -> 指定有界 shape -> 转成 Core ML -> 设置执行单元 -> 做一次推理验证”的链路。代码以 `coremltools` 风格为主，便于理解流程，实际细节会随模型类型变化。

```python
import numpy as np
import coremltools as ct

# 假设 traced_model 是已经导出的 TorchScript / MIL 可转换模型
traced_model = ...

seq_len = ct.RangeDim(lower_bound=1, upper_bound=128, default=32)

mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=(1, seq_len),
            dtype=np.int32,
        )
    ],
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.save("ToyModel.mlpackage")

x = np.array([[1, 2, 3, 4]], dtype=np.int32)
out = mlmodel.predict({"input_ids": x})

print("output keys:", out.keys())
```

这段代码里有四个要点：

1. `RangeDim(lower_bound=1, upper_bound=128)` 给的是**有界动态 shape**，不是无界输入。
2. `convert_to="mlprogram"` 表示使用较新的程序表示，适合更复杂模型图。
3. `compute_units=ct.ComputeUnit.ALL` 的意思是允许系统使用所有可用计算单元。
4. `predict()` 只是功能验证，不等于性能验证。

`computeUnits=all` 首次出现时一定要理解准确：它的白话意思是“系统可以自己挑 `ANE / GPU / CPU`”，不是“强制全部算子跑 ANE”。最终是否能上 ANE，还是看模型图能不能被编译器接受。

如果你需要在 iOS 侧显式设置，也会看到类似接口：

```swift
import CoreML

let config = MLModelConfiguration()
config.computeUnits = .all

let model = try MyModel(configuration: config)
```

验证“有没有真的跑上目标后端”不能只看配置代码，还要看实际 profiling、延迟变化、温度变化，以及模型图是否存在回退节点。

如果模型里确实有 Core ML 无法表达的算子，才考虑 custom layer。下面只是一个极简占位示意：

```python
# 仅示意：当算子无法写成标准或 composite operator 时，
# 才考虑 custom layer / custom operator 路径。
class MyCustomOp:
    def __init__(self, alpha: float):
        self.alpha = alpha
```

这一步的顺序不能反过来。正确思路是：**先尝试标准算子表达，再尝试 composite operator，最后才用 custom layer。**

---

## 工程权衡与常见坑

Core ML 部署最常见的问题，不是“转不过”，而是“转得过但跑不快”。这类问题本质上都落在三个约束上：编译器可优化性、后端覆盖率、内存峰值。

下面按“现象 -> 原因 -> 规避方式”整理成坑位清单：

| 现象 | 原因 | 规避方式 |
|---|---|---|
| `computeUnits=all` 但速度仍慢 | `all` 只是允许选择，不保证上 `ANE` | 检查算子可编译性、shape 约束和回退节点 |
| 模型能转但发热严重 | 部分节点落到 `CPU/GPU`，桥接频繁 | 减少回退点，优先稳定 shape 和标准算子 |
| 输入一变长，延迟突然抖动 | 无界动态 shape 导致优化退化 | 优先使用有界 `RangeDim` |
| 自定义层很多，性能没有改善 | custom layer 可运行但难深度优化 | 只在最后手段使用自定义层 |
| batch 稍大就崩或明显降速 | 峰值内存过高，移动端约束更紧 | 降 batch、压 KV cache、减少工作区 |
| reshape / transpose 很多 | 张量布局频繁变化，图被切碎 | 减少无意义重排，尽量统一布局 |
| 把 Core ML 当格式转换器 | 只关注导出成功，不关注执行图质量 | 先分析模型结构是否适合 Core ML |

其中有两个误区最值得单独强调。

### 误区一：`computeUnits=all` 不等于一定上 `ANE`

这是最常见的新手误解。`all` 的含义只是把选择权交给系统。系统是否能选到 `ANE`，前提是：

1. 这个算子能被 Core ML 表达；
2. 这个算子在当前 shape 下可被目标后端支持；
3. 图的前后文不会让切换成本高到不划算。

因此，配置写对了只是第一步，真正的关键仍然是模型图本身。

### 误区二：无界动态 shape 看起来更灵活，实际上更容易失速

很多人为了“兼容所有输入长度”，直接给非常宽甚至无界的 shape 范围。问题在于，移动端推理追求的不只是可运行，还要低延迟、低功耗、低热量。无界动态 shape 会让编译器更保守，很多原本能静态规划的优化做不了，最后导致性能显著下降。

如果业务输入确实变化大，工程上也应优先问一句：**这个变化范围能否被约束？** 只要能给出合理上界，编译器通常就更有机会优化。

---

## 替代方案与适用边界

Core ML 不是所有移动端模型的唯一答案。它更适合以下场景：

- 标准算子占比高；
- shape 可控，最好静态或有界动态；
- 目标平台明确是 Apple 生态；
- 追求系统级集成、端侧能耗和本地体验。

如果模型大量依赖不规则 attention、频繁变长输入、或者自定义算子无法被 composite 表达，那么即使能转成 Core ML，也未必是最优方案。

下面给出一个对比表：

| 方案 | 性能 | 兼容性 | 开发复杂度 | 端侧集成 |
|---|---|---|---|---|
| Core ML | Apple 设备上通常最好，前提是可编译性高 | 主要面向 Apple 生态 | 中等，需理解转换和后端限制 | 很强，和 iOS/macOS 体系结合紧密 |
| 纯 CPU 推理 | 通常最慢 | 最稳，限制最少 | 较低，调试直观 | 一般，容易上线但难做高性能 |
| 其他移动推理框架 | 取决于框架和硬件支持 | 跨平台通常更好 | 中到高，需额外适配 | 视框架而定，系统级整合未必如 Core ML |

实际选择可以这样判断：

1. 如果你的模型结构标准、Apple 平台优先，先尝试 Core ML。
2. 如果模型动态性很强、算子非常非标准，先评估是否需要保留其他移动推理框架。
3. 如果目标只是“先跑通验证功能”，CPU 路径往往最省时间，但不要把它误判为最终性能方案。

所以，Core ML 的适用边界不是“能不能转”，而是“转进去以后，算子能不能稳定落到合适硬件，并且在内存约束下持续跑稳”。

---

## 参考资料

1. [MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits)
2. [Creating and Integrating a Model with Custom Layers](https://developer.apple.com/documentation/coreml/creating-and-integrating-a-model-with-custom-layers)
3. [Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
4. [Composite Operators](https://apple.github.io/coremltools/docs-guides/source/composite-operators.html)
5. [Custom Operators](https://apple.github.io/coremltools/docs-guides/source/custom-operators.html)
6. [What Is Core ML Tools?](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html)
