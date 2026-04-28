## 核心结论

ONNX 转换工具解决的问题，不是“把模型存成一个 `.onnx` 文件”，而是把训练框架里的计算图、算子语义和形状约束，改写成后端还能继续理解、检查和优化的中间表示。这里的“中间表示”，可以理解成一种跨框架、跨运行时、跨硬件都尽量能读懂的统一描述。

一句话给新手解释整条链路：`PyTorch 模型 -> ONNX 中间图 -> ORT / TensorRT 再优化`。ONNX 只是交换格式，不是最终执行器。真正执行推理的，通常是 ONNX Runtime（简称 ORT，意思是 ONNX 的通用运行时）或者 TensorRT（NVIDIA 的高性能推理后端）。

可以把转换过程写成一个总公式：

$$
\text{训练框架图} + \text{形状约束} \rightarrow \text{ONNX 图} \rightarrow \text{后端可执行图}
$$

导出成功只说明“这个模型可以被表达成 ONNX 语法”，不说明“这个模型已经可以部署”。真正可部署，至少还要过四关：shape inference（形状推导，意思是自动推断张量维度）、后端算子兼容、动态 shape 配置、性能 profile 配置。

| 对比项 | 导出成功 | 可部署成功 |
| --- | --- | --- |
| 含义 | 成功生成 `.onnx` 文件 | 目标后端能稳定推理并满足性能要求 |
| 检查重点 | 图能否写出来 | 图能否被解析、推形状、选内核、跑通输入范围 |
| 常见误判 | 以为文件存在就算完成 | 实际上 ORT 能跑不代表 TensorRT 能跑 |
| 风险来源 | 算子可表达 | 算子不兼容、动态维不完整、profile 不匹配 |
| 结果 | “语法过关” | “工程落地过关” |

---

## 问题定义与边界

“模型格式转换”主要解决部署问题，不解决训练问题。更准确地说，它解决的是三类跨越：

1. 跨框架：比如模型原来在 PyTorch 里定义，但部署端不想依赖 PyTorch。
2. 跨运行时：比如同一个 ONNX 图，希望既能给 ORT 跑，也能给 TensorRT 跑。
3. 跨硬件：比如先在 CPU 上验证，再在 NVIDIA GPU 上追求更高吞吐。

边界也必须先说清。训练框架、ONNX、ORT、TensorRT 不是同一个层面的东西，它们各自负责的事情不同。

| 组件 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| 训练框架 | 定义模型、训练参数、保留动态图语义 | 不保证导出后所有后端都兼容 |
| ONNX | 作为交换格式描述图、算子、张量和部分形状 | 不直接负责执行性能 |
| ORT | 通用执行与图优化，兼容性通常较高 | 不一定给出极致 GPU 性能 |
| TensorRT | 面向 NVIDIA GPU 做高性能推理优化 | 不负责接纳所有 ONNX 算子与动态行为 |

一个常见误解是：“模型能导出 ONNX，就一定能在 TensorRT 跑。”正确说法是：ONNX 只是输入材料，TensorRT 还要能解析算子、推导形状、识别动态维，并且为输入范围建立 optimization profile（优化配置，意思是为不同输入形状准备可执行计划）才能跑起来。

真正的断层主要出现在这几类场景：

| 场景 | 为什么会出问题 |
| --- | --- |
| 动态图 | 运行时路径可能依赖真实输入，导出后未必能完整静态化 |
| 符号维 | 维度不是固定数字，而是 `B`、`T` 这种符号；后端不一定能一路推下去 |
| 控制流 | `if/loop` 之类结构在不同后端支持程度不同 |
| 自定义算子 | 训练框架里能算，不代表 ONNX 或后端有等价实现 |
| opset 版本差异 | opset 是 ONNX 算子版本集合，后端支持范围有限 |

所以，“转换”这件事的边界不是文件生成，而是后端是否还能继续理解这张图。

---

## 核心机制与推导

导出过程本质上是三步：

1. 把训练框架内部图规范化。
2. 把规范化后的功能算子映射成 ONNX op。
3. 把静态信息和符号形状写进图里，交给后端继续做 shape inference 和优化。

可以用一条机制链表示：

`PyTorch/torch.export -> 功能化算子 -> ONNX op -> 后端图`

这里的“功能化算子”，意思是把框架内部一些复杂或带副作用的操作，改写成更基础、更可分析的算子组合。因为后端不理解训练框架里的所有内部细节，它更擅长处理规范化后的标准图。

attention 是最能说明问题的例子。自注意力公式常写成：

$$
A = softmax\left(\frac{QK^T}{\sqrt{d}} + M\right)
$$

其中：

- `Q/K/V` 是查询、键、值张量。
- `d` 是每个 attention head 的隐藏维。
- `M` 是 mask，白话说就是“哪些位置允许看、哪些位置不允许看”的约束矩阵。

如果设：

- `B` = batch size
- `T` = 序列长度
- `H` = 头数
- `d` = 每头维度
- 则 `Q,K,V ∈ R^{B×H×T×d}`

那么 shape 推导链大致如下：

| 阶段 | 输入/操作 | 输出 shape | 说明 |
| --- | --- | --- | --- |
| 1 | `Q, K, V` | `B×H×T×d` | 三个输入张量形状一致 |
| 2 | `QK^T` | `B×H×T×T` | 最后两维做矩阵乘，注意力分数变成两两位置关系 |
| 3 | `+ M` | `B×H×T×T` | `M` 常从 `B×1×1×T` 广播到完整形状 |
| 4 | `softmax` | `B×H×T×T` | 概率化，不改变 shape |
| 5 | `A @ V` | `B×H×T×d` | 再和 `V` 相乘，回到每个位置的表示 |

玩具例子先看小尺寸。设 `B=1, H=2, T=4, d=4`，则：

- `Q/K/V` 的 shape 都是 `[1, 2, 4, 4]`
- `QK^T` 的 shape 是 `[1, 2, 4, 4]`
- 元素总数是 $1 \times 2 \times 4 \times 4 = 32$

如果只把 `T` 改成 `1024`，则：

- `QK^T` 的 shape 变成 `[1, 2, 1024, 1024]`
- 元素总数变成 $1 \times 2 \times 1024 \times 1024 = 2,097,152$

这说明一个关键事实：attention 中最重的一部分常常随着 $T^2$ 增长。动态长度不是简单写个“可变”就结束，必须给出上限和优化范围，否则后端即使接受了动态维，也可能在长序列下出现显存爆炸或性能退化。

真实工程例子更典型。以中文 BERT 或轻量 LLM 服务为例，训练在 PyTorch，部署时常导出 ONNX，再交给 TensorRT。这里如果只给 `input_ids` 标动态，而忘了 `attention_mask`、`past_key_values` 或输出维度，往往会出现一种表面正常、实际上不可交付的状态：ORT 可以跑通样例输入，但 TensorRT 在解析时因为广播关系或 reshape 链上的符号维断掉而失败。

本质原因不是 ONNX “不行”，而是图中关于 shape 的信息没有被完整保留下来，导致后端无法继续做正确的推导。

---

## 代码实现

工程上不要只写一个 `torch.onnx.export()` 就结束。更稳的流程应拆成四段：导出、验证、后处理、运行时配置。

先看一个可运行的 Python 玩具例子，它不依赖深度学习框架，目的是把 attention 的 shape 传播逻辑说清楚。

```python
from math import prod

def attention_shapes(B: int, H: int, T: int, d: int):
    qkv = (B, H, T, d)
    qk_t = (B, H, T, T)
    mask = (B, 1, 1, T)
    broadcast_mask = (B, H, T, T)
    output = (B, H, T, d)
    return {
        "qkv": qkv,
        "qk_t": qk_t,
        "mask": mask,
        "broadcast_mask": broadcast_mask,
        "output": output,
        "score_elements": prod(qk_t),
    }

small = attention_shapes(B=1, H=2, T=4, d=4)
large = attention_shapes(B=1, H=2, T=1024, d=4)

assert small["qkv"] == (1, 2, 4, 4)
assert small["qk_t"] == (1, 2, 4, 4)
assert small["score_elements"] == 32
assert large["qk_t"] == (1, 2, 1024, 1024)
assert large["score_elements"] == 1 * 2 * 1024 * 1024
assert large["score_elements"] > small["score_elements"] * 1000

print("small:", small)
print("large:", large)
```

上面这个例子说明：动态 shape 的核心不是“能不能变”，而是“变了以后后端还能不能推得清、跑得动”。

下面是工程代码骨架，重点不是 API 拼写细节，而是每一段负责什么。

```python
import torch

# 1. 导出：把可变输入写进 ONNX 图
torch.onnx.export(
    model,
    args=(input_ids, attention_mask),
    f="model.onnx",
    opset_version=17,
    dynamic_shapes={
        "input_ids": {0: "B", 1: "T"},
        "attention_mask": {0: "B", 1: "T"},
    },
    verify=True,  # 用 ONNX Runtime 做基本一致性校验
    custom_translation_table={
        # 某些框架算子无法直接映射时，在这里提供替代表达
    },
)

# 2. 后处理：shape inference
import onnx
from onnx import shape_inference

model_onnx = onnx.load("model.onnx")
inferred = shape_inference.infer_shapes(model_onnx)
onnx.save(inferred, "model.inferred.onnx")

# 3. 运行时验证：先用 ORT 跑通
import onnxruntime as ort
sess = ort.InferenceSession("model.inferred.onnx", providers=["CPUExecutionProvider"])

# 4. 交给 TensorRT 时，再配置动态 shape profile
# min / opt / max 不是装饰参数，而是决定引擎能否覆盖输入范围
```

几个关键参数必须明确：

| 参数/步骤 | 作用 | 不配会怎样 |
| --- | --- | --- |
| `dynamic_shapes` | 把输入哪些维度可变写进图里 | 后端把本应可变的维度当固定值 |
| `custom_translation_table` | 自定义算子映射规则 | 导出阶段直接失败，或语义不等价 |
| `verify=True` | 用 ORT 做导出结果的基础校验 | 容易出现导出成功但数值早已偏掉 |
| `opset_version` | 指定 ONNX 算子版本集合 | 版本过新后端不认，过旧表达力不够 |
| `shape inference` | 补全更多可推导的中间 shape | 后端优化链断掉 |
| `constant folding` | 常量折叠，提前算掉静态子图 | 图更复杂、运行时负担更重 |
| `optimization profiles` | 给 TensorRT 指定 `min/opt/max` 输入范围 | 动态 shape 无法落地为可执行引擎 |

步骤视角再看一遍：

| 输入 | 导出参数 | 产物 | 校验点 | 下一步 |
| --- | --- | --- | --- | --- |
| PyTorch 模型 + 样例输入 | `opset_version`, `dynamic_shapes`, `verify=True` | `model.onnx` | 导出是否成功、数值是否近似一致 | 跑 shape inference |
| ONNX 图 | shape inference | `model.inferred.onnx` | 中间节点 shape 是否补全 | 跑 ORT |
| 推理样例 | ORT session | 基础可运行图 | 输出 shape、数值、异常算子 | 做 constant folding/性能验证 |
| ONNX 图 + 目标输入范围 | TensorRT profile | 引擎 | 是否解析成功、是否覆盖真实流量 | 上线压测 |

真实工程里，建议先在 ORT 跑通并逐层比对，再交给 TensorRT。因为 ORT 通常更像“通用检查器”，而 TensorRT 更像“性能优化器”。前者适合查正确性，后者适合追性能。

---

## 工程权衡与常见坑

最大的工程问题通常不是“导不出来”，而是“导出来以后图不稳”。图不稳，意思是同一份 ONNX 图在不同输入、不同后端、不同 profile 下行为不一致，或者性能极不稳定。

先看最常见的坑位表。

| 坑位 | 现象 | 根因 |
| --- | --- | --- |
| `dynamic_axes` / `dynamic_shapes` 不完整 | ORT 能跑，TensorRT 失败 | 相关输入输出没有一起标动态 |
| `opset` 过高 | 后端直接报不支持 | 导出器能写，后端 parser 不认 |
| shape inference 中断 | 某些中间节点 shape 变成未知 | `Reshape/Concat/Gather` 依赖符号维但信息不够 |
| 自定义算子缺失 | 导出或部署失败 | 没有 ONNX 等价表达，或后端无实现 |
| `T×T` 广播爆显存 | 长序列性能雪崩 | attention score/mask 扩成大矩阵 |

新手最容易踩的坑是：只给 `input_ids` 标动态，忘了 `attention_mask` 和相关输出。结果是 ORT 因为容忍度高还能跑通示例，但 TensorRT 需要更严格的 shape 关系，一解析到广播链就失败。

进阶一点的坑是：`Reshape`、`Concat`、`Gather` 这类算子如果依赖符号维，前面少一条 shape 信息，后面整条推导链都会断。后端一旦拿不到明确维度，就很难继续做 kernel 选择和内存规划。

正确性、可维护性、性能三者之间也要做平衡：

| 目标 | 更保守的做法 | 代价 |
| --- | --- | --- |
| 正确性 | 动态维少一些，先固定关键输入范围 | 灵活性下降 |
| 可维护性 | 尽量使用标准 ONNX 算子，不引入 custom op | 可能牺牲部分模型表达 |
| 性能 | 为特定硬件收紧 shape 范围、做 profile 调优 | 通用性下降，迁移成本上升 |

实际排查顺序建议固定，不要一上来就盲调 TensorRT：

1. 先查目标后端的 operator matrix，也就是算子支持列表。
2. 补全动态轴，不只看主输入，还要看 mask、cache、输出相关维度。
3. 跑 ONNX shape inference，确认中间张量 shape 没有大面积未知。
4. 做 constant folding，尽量减掉静态子图噪声。
5. 用目标 runtime 逐层验证，确认是导出问题、shape 问题还是后端兼容问题。

这套顺序的价值在于把问题分层。否则你看到“TensorRT build failed”，并不知道是算子不支持、profile 不合法，还是前面的 shape 已经错了。

---

## 替代方案与适用边界

如果目标是高兼容、少折腾，`ONNX + ORT` 往往更稳。ORT 的优势是通用性强、调试相对简单，尤其适合 CPU 服务、离线批处理、或者先做正确性验证的阶段。

如果目标是在 NVIDIA GPU 上追求极致吞吐，`ONNX + TensorRT` 更合适。但前提非常明确：输入 shape 范围清晰，关键算子都能被解析，必要时还能接受算子替换或 plugin 扩展。

如果模型里有大量自定义算子、复杂控制流、或者动态行为极强，继续强行走 ONNX 可能得不偿失。此时直接使用原生框架 runtime，或者使用后端专用格式，反而更稳。

| 方案 | 适用场景 | 优势 | 限制 |
| --- | --- | --- | --- |
| ORT | CPU 服务、兼容性优先、验证优先 | 通用、稳定、易调试 | 极致性能通常不如 TensorRT |
| TensorRT | NVIDIA GPU 在线推理、吞吐优先 | 高性能、图融合和 kernel 优化强 | 对动态 shape、算子兼容要求高 |
| 原生框架 | 自定义算子多、控制流复杂、快速迭代 | 表达力强、与训练端一致 | 部署包更重，跨环境迁移弱 |

新手版建议很简单：CPU 服务优先选 ORT。理由不是“它最先进”，而是它兼容性高、反馈直接，适合先把正确性做稳。

工程版建议更具体：NVIDIA GPU 在线推理优先考虑 TensorRT，但要先回答三个问题：

1. 是否强依赖自定义算子？
2. 是否需要极致吞吐？
3. 是否能固定或收紧输入范围？

可以把决策逻辑写成一棵简化决策树：

| 判断问题 | 是 | 否 |
| --- | --- | --- |
| 是否强依赖自定义算子 | 优先原生框架或专用后端 | 继续看下一项 |
| 是否需要极致吞吐 | 优先 TensorRT | 优先 ORT |
| 是否能固定输入范围 | TensorRT 更容易发挥 | ORT 更稳，TensorRT 成本变高 |

结论不是“所有模型都应该转 ONNX”，而是“当你需要跨框架部署，并且后端能承接这张图时，ONNX 才是高性价比路径”。

---

## 参考资料

1. [PyTorch ONNX Exporter](https://docs.pytorch.org/docs/stable/onnx_export.html)：用于核对 `dynamic_shapes`、`custom_translation_table`、`verify=True` 等导出参数。
2. [ONNX Shape Inference](https://onnx.ai/onnx/repo-docs/ShapeInference.html)：用于理解 shape inference 的覆盖范围、推导机制和局限。
3. [TensorRT Working with Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html)：用于核对动态维、`-1` 占位和 `min/opt/max` profile 约束。
4. [TensorRT Architecture Overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)：用于理解 ONNX parser、图优化和 TensorRT 的执行架构。
5. [ONNX Runtime Custom Operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)：用于处理自定义算子在运行时的注册与兼容问题。
