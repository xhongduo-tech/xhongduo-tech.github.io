## 核心结论

模型格式兼容性处理的目标，不是把模型“导出成某个文件”，而是让训练框架里的计算语义，在部署框架里仍然表示同一件事。

直白一点说，`model.onnx` 能写出来，只说明“文件生成成功”；目标 runtime 能加载，只说明“它大致看懂了这个图”；固定输入下输出还能对齐，才说明“语义基本一致”。这三件事不是一回事。

兼容性问题通常集中在四类约束：

| 训练侧语义 | 部署侧要求 | 常见失败表现 |
| --- | --- | --- |
| `opset`，即算子版本集合 | 目标 runtime 支持对应版本 | 加载时报不支持的 op 或属性 |
| `shape`，即张量形状规则 | 动态维、范围、profile 可被接受 | 构建失败、推理时报维度不匹配 |
| `dtype`，即数据类型 | `fp32/fp16/int8` 等必须可执行 | 自动回退、精度异常、直接报错 |
| 广播与自定义算子语义 | 规则要与训练侧一致 | 能跑但结果漂移，或找不到算子 |

因此，模型格式兼容性的本质不是“转格式”，而是“跨框架保持等价计算”。

---

## 问题定义与边界

本文讨论的“兼容性”，至少分三层：

| 层级 | 判定标准 | 说明 |
| --- | --- | --- |
| 可导出 | 文件能生成 | 只证明导出器完成了图序列化 |
| 可加载 | 目标 runtime 能解析并建图 | 说明目标引擎认识这些节点 |
| 可对齐 | 同一输入下输出误差在可接受范围内 | 说明语义基本一致 |

很多新手把第一层误当成最终成功标准，这是最常见的误区。

一个最小场景是：你在 PyTorch 里训练了一个文本分类模型，导出为 ONNX，在 ONNX Runtime 里能跑；但切到 TensorRT 时，构建阶段报错。原因可能不是模型坏了，而是 TensorRT 对动态 `shape` 的接受方式更严格，或者某个算子的 `opset` 太新。也就是说，同一个文件，对不同推理引擎并不保证同样可用。

本文边界聚焦在“模型格式转换与运行时兼容”，不展开以下内容：

| 不展开的话题 | 原因 |
| --- | --- |
| 训练策略本身 | 它影响精度，但不直接决定格式兼容 |
| 量化算法细节 | 本文只讨论量化后 dtype 带来的兼容性影响 |
| 模型压缩与蒸馏 | 这是模型优化问题，不是格式语义问题 |
| 集群调度与服务治理 | 属于上线系统层，不是图语义层 |

可以把问题理解成一条链路：

1. 训练框架定义计算图。
2. 导出器把图翻译为中间格式。
3. 部署 runtime 再把中间格式翻译为自己的执行计划。
4. 只要任意一次翻译丢失信息，最终结果就可能变掉。

---

## 核心机制与推导

先给抽象表达。设训练侧输出为

$$
y_S = f_S(x; v, s, \tau, r)
$$

部署侧输出为

$$
y_D = f_D(x; v, s, \tau, r)
$$

这里：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $x$ | 输入张量 | 喂给模型的数据 |
| $v$ | 算子版本约束 | 这一层“该怎么计算”的版本定义 |
| $s$ | 形状约束 | 每个张量尺寸如何变化 |
| $\tau$ | 数据类型约束 | 用什么精度存和算 |
| $r$ | 广播、属性、自定义规则 | 细节语义，比如维度补齐怎么做 |

要让模型兼容，目标不是文件相同，而是希望

$$
f_S \approx f_D
$$

更具体地说，至少要满足：

$$
v \in V_D,\quad s \in S_D,\quad \tau \in T_D,\quad r \in R_D
$$

其中 $V_D, S_D, T_D, R_D$ 分别表示部署 runtime 能接受的算子版本集合、形状集合、dtype 集合和规则集合。

为什么任意一项不满足都会出问题：

1. 若 $v \notin V_D$，runtime 根本不认识这个算子版本，通常直接加载失败。
2. 若 $s \notin S_D$，它可能认识算子，但无法为该 shape 建立执行计划，常见于动态维范围不合法。
3. 若 $\tau \notin T_D$，可能报错，也可能静默回退到别的精度，结果是性能和数值都变。
4. 若 $r \notin R_D$，最危险，因为它常常“能跑”，但结果已经偏了。

### 玩具例子：广播规则导致的隐蔽偏差

广播，白话说，就是“尺寸不完全一样的张量，按某种补齐规则自动参与计算”。

设有矩阵：

$$
A=\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

和向量：

$$
b=[10,20]
$$

如果 runtime 把 `b` 解释为按列广播，那么

$$
A+b=
\begin{bmatrix}
11 & 22 \\
13 & 24
\end{bmatrix}
$$

如果某个错误转换把它当成按行广播，对应的结果就会不同。新手常见误解是“反正维度能对上就行”，但广播的方向、补维规则、是否允许隐式扩展，都属于语义的一部分。

### 真实工程例子：同一 ONNX 文件在 ORT 能跑，在 TensorRT 失败

设一个检测模型输入为 `[N, 3, H, W]`，训练时 `H,W` 可以变化。你导出 ONNX 时把动态轴都放开了，ONNX Runtime 因为解释器更宽容，可以在运行时逐步处理；但 TensorRT 需要提前构建 engine，它要求给出明确的最小、最优、最大 profile，例如：

- `min=(1,3,320,320)`
- `opt=(1,3,640,640)`
- `max=(4,3,1280,1280)`

如果你没有提供，或者范围过大、跨度不合理，TensorRT 可能构建失败。即便构建成功，某些算子也可能因新 `opset` 属性不支持而退化或报错。这里的本质不是“TensorRT 更差”，而是它选择了更强约束来换性能。

---

## 代码实现

工程上最小可执行链路应当是：导出、校验、加载、对齐，而不是只做第一步。

下面先用一个纯 Python 玩具例子模拟“广播语义不一致会导致数值偏差”。代码可直接运行。

```python
from typing import List

def add_row_broadcast(matrix: List[List[float]], vec: List[float]]) -> List[List[float]]:
    # 把 vec 当作按列展开到每一行
    return [[x + v for x, v in zip(row, vec)] for row in matrix]

def add_col_broadcast(matrix: List[List[float]], vec: List[float]]) -> List[List[float]]:
    # 把 vec 当作每个标量作用到整行
    assert len(matrix) == len(vec)
    return [[x + vec[i] for x in row] for i, row in enumerate(matrix)]

A = [
    [1.0, 2.0],
    [3.0, 4.0],
]
b = [10.0, 20.0]

row_result = add_row_broadcast(A, b)
col_result = add_col_broadcast(A, b)

assert row_result == [[11.0, 22.0], [13.0, 24.0]]
assert col_result == [[11.0, 12.0], [23.0, 24.0]]
assert row_result != col_result

def max_abs_diff(x, y):
    diff = 0.0
    for rx, ry in zip(x, y):
        for a, b in zip(rx, ry):
            diff = max(diff, abs(a - b))
    return diff

assert max_abs_diff(row_result, col_result) == 10.0
print("broadcast semantic mismatch detected")
```

这段代码的作用不是复现某个具体框架，而是强调一件事：只要部署侧对“该怎么补维”的理解变了，即便张量都能算，输出也可能不再等价。

下面给出更接近真实部署的最小流程。第一段是导出思路：

```python
# pseudo code
import torch

model = MyModel().eval()
x = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    x,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

这里的检查点不是“导出了”，而是：

| 检查项 | 为什么要看 |
| --- | --- |
| `opset_version` | 太新目标引擎可能不支持，太旧可能表达不了语义 |
| `dynamic_axes` | 放开哪些维度，决定后续 runtime 如何建图 |
| 输入样例 | 导出时会固化部分路径，样例错误会污染图 |

第二段是图校验和 runtime 加载思路：

```python
# pseudo code
import onnx
import onnxruntime as ort
import numpy as np

model = onnx.load("model.onnx")
onnx.checker.check_model(model)

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
x = np.random.randn(1, 3, 224, 224).astype("float32")
y = sess.run(["output"], {"input": x})[0]

assert y is not None
```

第三段是做输出对齐，不要跳过：

```python
# pseudo code
import numpy as np

torch_out = model(torch.tensor(x)).detach().cpu().numpy()
ort_out = sess.run(["output"], {"input": x})[0]

max_abs_diff = np.max(np.abs(torch_out - ort_out))
rel_error = max_abs_diff / (np.max(np.abs(torch_out)) + 1e-8)

assert max_abs_diff < 1e-4
assert rel_error < 1e-4
```

如果你后面还要接 TensorRT，那么动态 shape profile 也必须显式设计。概念上像这样：

```python
# pseudo code
profile.set_shape(
    "input",
    min=(1, 3, 320, 320),
    opt=(1, 3, 640, 640),
    max=(4, 3, 1280, 1280),
)
```

这里 `min/opt/max` 的意思分别是最小可接受形状、最常见工作形状、最大可接受形状。很多人把范围一口气放得很大，结果不是更通用，而是更难构建、更难优化。

如果模型里有自定义算子，自定义算子就是“标准图里没有、需要额外实现的节点”，那么还必须在 runtime 注册；否则不是数值偏差，而是直接找不到实现。

---

## 工程权衡与常见坑

真实工程里，最常见的错误不是不会导出，而是验证链路太短：只看到 `.onnx` 文件生成，就认为部署完成。

下面是高频问题汇总：

| 常见坑 | 根因 | 典型表现 | 规避方式 |
| --- | --- | --- | --- |
| `opset` 随手升到最新 | 以为越新越好 | 目标引擎解析失败 | 先查目标 runtime 支持矩阵，再定版本 |
| 动态 `shape` 写太松 | 想一次兼容所有输入 | profile 难构建，性能差 | 按真实业务流量收敛输入范围 |
| `dtype` 一步降到 `fp16/int8` | 只盯吞吐 | 精度漂移、回退执行 | 先做 `fp32` 对齐，再逐级降精度 |
| 自定义 op 没注册 | 忘记部署侧实现 | 加载失败 | 导出前确认替换方案，部署前确认注册链路 |
| 忽略广播差异 | 以为小维度会自动处理 | 输出悄悄偏离 | 固定输入做逐层对齐 |
| 只测能加载，不测数值 | 把“可运行”当“正确” | 线上结果污染 | 把 `max_abs_diff`、任务指标一起纳入验收 |

一个真实工程例子是目标检测部署。训练团队在 PyTorch 中用了较新的插值或 reshape 路径，导出 ONNX 后，ONNX Runtime 可以跑通；但切到 TensorRT 时出现两类问题：

1. 某个算子属性只在较新 `opset` 才有，TensorRT 当前版本不支持。
2. 输入图片尺寸是动态的，但业务侧实际只会出现少数几个分辨率，导出时却把范围开得很大，导致 engine 构建时间长、显存高、性能不稳定。

最后的解决方式通常不是“继续硬调 TensorRT”，而是三步一起做：

1. 回退到目标引擎稳定支持的 `opset`。
2. 把动态 shape 收敛到真实业务范围。
3. 对无法兼容的特殊节点，改写成更通用的算子组合。

一个实用排查顺序是：

1. 先固定输入和 `fp32`，排除动态 shape 与低精度干扰。
2. 再验证目标 runtime 是否能加载。
3. 再比较输出误差，而不是只看有没有报错。
4. 最后才开启动态 shape、`fp16`、插件算子和进一步优化。

这个顺序的意义是把问题分层。否则多个变量一起变化时，你很难知道失败来自格式、shape、dtype 还是插件实现。

---

## 替代方案与适用边界

兼容性问题不一定都该靠“继续修导出图”解决。有时换运行时、换实现方式，成本更低。

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 重新导出并降级算子 | 路径最标准，维护简单 | 可能牺牲部分新特性 | 优先兼容性，目标环境复杂 |
| 注册 custom op | 保留原始语义最完整 | 部署和维护成本高 | 必须保留特殊算子 |
| 使用更宽容的 runtime | 更容易先跑通 | 性能可能一般 | 先求稳定上线，再做优化 |
| 保留训练图部分执行 | 改动小，语义最稳 | 系统复杂，跨框架调度麻烦 | 少量节点难以转换 |
| 改写为更通用算子组合 | 兼容性通常更好 | 需要额外开发与验证 | 常见算子能等价替换时 |

可以按场景理解：

- 如果你的首要目标是“多平台都能跑”，就应使用保守 `opset`、减少自定义算子、缩小动态 shape 变化面。
- 如果你的首要目标是“TensorRT 上的极致性能”，那就不该等导出后再修，而应在建模和导出阶段就围绕它的支持边界设计。
- 如果模型业务价值高度依赖某个特殊算子，那么接受 custom op 或子图拆分带来的部署复杂度，往往比强行替换更现实。

统一格式并不能解决全部问题。ONNX 很重要，但它更像“公共中间语言”，不是“自动保证等价执行的魔法文件”。一旦底层 runtime 对算子、shape、dtype 或规则的解释不同，仍然需要工程化验证。

---

## 参考资料

1. [ONNX Versioning](https://onnx.ai/onnx/repo-docs/Versioning.html)  
用于理解 `opset` 与算子版本如何影响第 3 章的兼容性约束。

2. [ONNX Broadcasting in ONNX](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md)  
用于理解第 3 章和第 5 章中广播语义差异为什么会导致隐蔽数值偏差。

3. [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)  
用于对应第 2 章和第 4 章，说明训练图如何被导出为中间格式。

4. [NVIDIA TensorRT Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)  
用于对应第 4 章和第 5 章，解释动态 shape profile 为什么是 TensorRT 兼容性的核心约束。

5. [ONNX Runtime Custom Operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)  
用于对应第 4 章和第 6 章，说明自定义算子为什么必须在部署侧注册。
