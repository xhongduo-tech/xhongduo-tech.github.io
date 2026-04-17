## 核心结论

模型导出，本质上是把“训练时可用的模型表示”变成“部署时更容易运行的模型表示”。训练框架关心梯度、优化器、动态图；部署框架关心推理图、算子融合、显存占用和目标硬件。对初学者来说，最重要的不是背格式名称，而是理解一条统一链路：

$$
\text{训练模型} \rightarrow \text{中间格式} \rightarrow \text{优化/编译} \rightarrow \text{部署运行时}
$$

其中：

| 格式 | 一句话定义 | 主要优势 | 主要短板 | 典型目标 |
|---|---|---|---|---|
| ONNX | 中立交换格式，也就是不同框架之间通用的“模型文件语言” | 兼容性强，便于跨框架、跨平台 | 算子兼容受版本影响 | 跨平台部署、后续转 TensorRT |
| TorchScript | PyTorch 原生可执行格式，也就是把 PyTorch 模型变成可保存、可加载的计算图 | 保留较多 PyTorch 语义，控制流支持相对自然 | 通用性不如 ONNX | 仍以 PyTorch 生态为主的部署 |
| TensorRT | NVIDIA 的推理编译引擎，也就是针对 GPU 深度优化后的执行计划 | 性能高，支持 FP16/INT8、算子融合、Tensor Core | 强依赖 NVIDIA GPU，解析约束多 | 服务器 GPU 推理 |
| TFLite | TensorFlow Lite 的轻量部署格式，也就是移动端和嵌入式的简化模型格式 | 运行时小，适合移动端 | 表达能力和算子覆盖较窄 | Android、Edge、嵌入式 |

实践上可以把它理解成两层选择：

1. 先选“中间格式”：ONNX 或 TorchScript。
2. 再选“目标运行时”：TensorRT、TFLite、ONNX Runtime 等。

新手最稳妥的路线通常是：PyTorch 权重先导出 ONNX，再用 `trtexec --onnx=model.onnx --batch=32 --fp16` 编译出 TensorRT engine，在 GPU 上部署。原因很直接：ONNX 负责通用性，TensorRT 负责性能，两者分工清晰。

---

## 问题定义与边界

“模型导出”不是简单地把 `.pt` 改成 `.onnx`。它解决的是三个部署问题：

1. 跨平台运行。训练在 PyTorch，部署可能在 C++ 服务、移动端、边缘设备。
2. 性能优化。训练图里很多信息对推理无用，部署时可以折叠、融合、量化。
3. 减少依赖。部署端不想带完整训练框架，只想带最小 runtime。

边界也必须说清。不是所有模型都能顺利导出，更不是所有导出文件都能顺利加速。常见限制有：

| 边界项 | 具体含义 | 典型后果 |
|---|---|---|
| 输入输出类型 | 很多部署格式只接受张量，不接受 `dict`、`str`、复杂对象 | 导出失败或接口要重写 |
| 算子支持 | 目标 runtime 只支持一部分算子 | 转换时报 “unsupported op” |
| 动态形状 | 输入长度、图片尺寸变化太大 | 需要显式配置 profile |
| 量化误差 | FP32 改成 FP16/INT8 会有误差 | 精度下降 |
| 显存与 Batch | 批量越大，占用越大 | 吞吐上升但延迟、显存变差 |

玩具例子可以先看一个最简单场景。你训练了一个两层卷积网络，在 notebook 里直接 `model(x)` 没问题；但上线时服务器只有推理环境，没有训练代码，这时就要把模型变成 ONNX 或 TensorRT engine。这个例子里，导出是为了“脱离训练环境”。

真实工程例子更能说明边界。假设一个 Transformer 模型里包含 PyTorch 自定义模块，导出 ONNX 后再交给 TensorRT 解析，结果报错：某个子图无法 constant fold，或者某个自定义算子没有实现。这里的意思是：TensorRT 希望拿到尽量静态、尽量标准的计算图，但你的图里还保留了它无法理解的部分。工程上常见做法是先用 Polygraphy 做常数折叠、shape 简化，必要时改写模型结构，再继续编译。

所以，模型导出不是“保存文件”，而是“把模型表达形式压缩到目标 runtime 能接受且能高效执行的边界内”。

---

## 核心机制与推导

ONNX、TorchScript、TensorRT 的差别，不在文件后缀，而在它们保留信息的层次不同。

ONNX 是交换格式，重点是“让别的系统看懂”。TorchScript 是 PyTorch 原生静态化表示，重点是“让 PyTorch 生态外也能执行”。TensorRT 不是单纯格式，而是编译器加执行引擎，重点是“针对 NVIDIA GPU 重新安排计算”。

从推理优化角度，TensorRT 常做四类事：

| 优化动作 | 白话解释 | 结果 |
|---|---|---|
| 常数折叠 | 提前把固定不变的计算算完 | 减少运行时计算 |
| 算子融合 | 把多个小操作合成一个大操作 | 减少访存和 kernel launch |
| 精度降阶 | 从 FP32 改成 FP16 或 INT8 | 更快、更省显存 |
| Kernel 选择 | 按硬件和 shape 选最快实现 | 吞吐和延迟更优 |

为什么 `--batch=32 --fp16` 常常有效？因为 GPU 吞吐通常受并行度和数据类型共同影响。可以用一个近似理解：

$$
\text{吞吐量} \approx \text{batch size} \times \text{单卡并行利用率} \times \text{低精度加速收益}
$$

这不是严格公式，但足够说明方向。`batch=1` 时，GPU 可能没吃满；`batch=32` 后，并行度更高。`fp16` 时，数据更小，很多 GPU 还能走 Tensor Core。于是吞吐一般提升明显，但代价是单请求延迟可能增加，数值精度也可能轻微下降。

玩具例子：一个普通 CNN 做图片分类。先导出 ONNX，再用 TensorRT 编译：

1. `torch.onnx.export(...)` 生成 `model.onnx`
2. `trtexec --onnx=model.onnx --batch=32 --fp16`
3. 得到 `model.engine`

这三步里，第一步解决“可移植”，第二步解决“可执行且更快”，第三步的产物才是部署端真正加载的二进制。

真实工程例子：Vision Transformer 或 LLM 子模块的导出。训练图里可能有 LayerNorm、MatMul、Residual Add、GELU 等一串操作。TensorRT 会尝试把一部分模式融合成更高效的执行单元，再根据 profile 选择最适合某组输入 shape 的 kernel。如果 profile 只覆盖短序列，长序列上线时就可能退化甚至报错。这说明“导出成功”不等于“部署成功”，profile 也是模型定义的一部分。

再看格式职责的分工：

| 阶段 | 负责什么 | 不负责什么 |
|---|---|---|
| ONNX/TorchScript | 把模型图表达出来 | 不保证一定最快 |
| TensorRT/TFLite | 结合硬件做编译优化 | 不替你修复所有不兼容结构 |
| Runtime | 真正执行推理 | 不负责训练语义 |

理解这点后，很多问题都能定位：导出失败，多半是图表达问题；编译失败，多半是目标 backend 不支持；速度不理想，多半是 batch、精度、profile 或访存模式没有调好。

---

## 代码实现

下面给一个“PyTorch → ONNX → TensorRT”的最小工程示例。它分三步：

1. 定义并导出 PyTorch 模型。
2. 用 ONNX checker 验证导出结果。
3. 调用 `trtexec` 编译 engine。

```python
import os
import subprocess
import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)

def export_to_onnx(path="model.onnx"):
    model = TinyCNN().eval()
    dummy = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 2)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )

    assert os.path.exists(path)
    return path

def build_tensorrt_engine(onnx_path, engine_path="model.engine"):
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--minShapes=images:1x3x32x32",
        "--optShapes=images:32x3x32x32",
        "--maxShapes=images:64x3x32x32",
    ]
    # 实际运行前请确保系统已安装 TensorRT 的 trtexec
    return cmd

if __name__ == "__main__":
    onnx_path = export_to_onnx()
    cmd = build_tensorrt_engine(onnx_path)
    assert cmd[0] == "trtexec"
    print("Exported:", onnx_path)
    print("Compile command:", " ".join(cmd))
```

这段代码里，`assert` 做了两件事：先验证 PyTorch 前向输出形状，再验证 ONNX 文件确实生成。对初学者来说，这很重要，因为很多“转换失败”其实发生在更前面，例如模型本身没切到 `eval()`、输入 shape 不符合预期、动态轴写错。

如果你还需要 TorchScript，对应代码通常是：

```python
scripted = torch.jit.trace(model.eval(), dummy)
scripted.save("model.ts")
```

这里的 `trace` 意思是“按一次示例输入记录计算图”。如果模型里有明显控制流，比如 `if`、`for` 依赖输入数据，往往要考虑 `torch.jit.script`，因为它更接近“按代码语义编译”。

真实工程里，还常在 ONNX 之后增加一步图清理。比如用 Polygraphy 先做 constant-fold，把图里的常量计算提前展开，再交给 TensorRT。这样做的目的不是“更优雅”，而是减少 TensorRT 解析失败的概率。

新手可以先记住最实用的版本：训练结束后执行两件事，先 `torch.onnx.export(...)` 生成 `model.onnx`，再运行 `trtexec --onnx=model.onnx --batch=32 --fp16` 或等价 profile 参数编译 engine。前者解决格式统一，后者解决部署性能。

---

## 工程权衡与常见坑

部署不是“谁最快就选谁”，而是“在兼容、性能、维护成本之间取平衡”。

第一组权衡是兼容性和优化深度。ONNX 通用，但离硬件较远；TensorRT 很快，但限制多。TorchScript 保留了 PyTorch 习惯，但不适合所有跨平台场景。TFLite 适合移动端，但量化后要重新验精度。

第二组权衡是吞吐、延迟和显存。大 batch 常提高吞吐，但单请求延迟上升；FP16/INT8 更快，但需要验证精度；动态 shape 更灵活，但 profile 复杂，engine 也可能更大。

常见坑可以整理成表：

| 问题 | 现象 | 原因 | 应对策略 |
|---|---|---|---|
| ONNX 不支持某算子 | 导出或解析时报 unsupported op | 模型结构超出标准算子集 | 改写模块、升级 opset、拆分子图 |
| TensorRT 报 `Non-constant input to constant fold` | 编译中断 | 图里还有未折叠常量或 shape 推导不完整 | 先跑 Polygraphy constant-fold |
| 输入输出是 `dict` 或字符串 | 导出失败 | 部署图偏好纯张量接口 | 改成 tuple/list/tensor |
| 动态 shape 配错 | 某些 batch 或分辨率下失败 | profile 没覆盖线上范围 | 明确配置 `min/opt/max` |
| TFLite 量化后精度下降 | 线上预测偏差 | 低精度表示带来误差 | 做校准和 A/B 验证 |
| TorchScript 与 eager 行为不一致 | 推理结果异常 | trace 没覆盖控制流 | 改用 script 或重构 forward |

玩具例子：一个分类模型输入就是固定大小图片，最容易做。因为输入 shape 稳定、算子简单、没有复杂控制流，通常 ONNX 和 TensorRT 都很顺。

真实工程例子：一个包含自定义注意力模块的 Transformer。你可能先能导出 ONNX，但 TensorRT 解析时报错。这时不要只盯着报错字符串，而要检查三件事：

1. 是否存在非标准算子。
2. 是否有动态 shape 没设 profile。
3. 是否需要先用 Polygraphy 做常数折叠和图清理。

很多人第一次踩坑，是因为把“导出成功”当成“部署完成”。实际上至少还要做两轮验证：

1. 数值验证：PyTorch、ONNX Runtime、TensorRT 输出误差是否在容忍范围内。
2. 性能验证：延迟、吞吐、显存是否满足真实业务条件。

特别是 TFLite 量化模型，哪怕最大误差只有 `5.8e-5` 这种级别，也要看任务类型。分类模型可能无感，排序或回归模型就未必。

---

## 替代方案与适用边界

TensorRT 不是唯一答案。部署方案应该由设备、团队熟悉度、维护成本共同决定。

| 方案 | 适用设备 | 优点 | 不适合的场景 |
|---|---|---|---|
| TensorRT | NVIDIA GPU 服务器 | 性能强，优化深 | 非 NVIDIA 环境 |
| ONNX Runtime | 服务器、桌面、跨平台 | 通用、接入简单 | 极致 GPU 性能要求 |
| TFLite | Android、嵌入式 | 运行时轻，移动端友好 | 大型复杂动态图 |
| Core ML | Apple 设备 | 与苹果生态集成好 | 非苹果平台 |
| TVM | 定制硬件、异构设备 | 编译灵活，可深度定制 | 学习和调优成本高 |
| TorchScript | PyTorch 主导环境 | 原生、迁移平滑 | 跨框架通用性要求高 |

如果你的目标是云端 GPU 服务，并且机器明确是 NVIDIA，TensorRT 往往是优先选项。如果目标是 Android 手机，TFLite 通常更现实，因为它的 runtime 更小、集成更直接。对于“先跑起来再优化”的团队，ONNX Runtime 常常是很好的中间答案。

一个移动端真实工程例子：要把一个轻量 Transformer 部署到 Android。此时选择 TFLite 通常比 TensorRT 更合理，因为手机没有 NVIDIA 数据中心 GPU。流程会变成：训练权重 → 导出 TFLite → 在 App 里用 `Interpreter` 加载 `.tflite` 执行。这里你放弃了一部分极致优化能力，换来了部署复杂度的显著下降。

再强调一次适用边界：

1. 强控制流、动态图明显的模型，TorchScript 往往比 ONNX 更自然。
2. 强跨平台诉求时，ONNX Runtime 比 TensorRT 更省心。
3. 极致 GPU 推理性能时，TensorRT 优先。
4. 移动端和嵌入式时，TFLite、Core ML、TVM 更常见。

所以，“最佳格式”并不存在，只有“最适合当前目标平台和维护成本的格式”。

---

## 参考资料

- NVIDIA TensorRT Developer Guide
- NVIDIA TensorRT Best Practices
- PyTorch ONNX Export 文档
- PyTorch TorchScript 文档
- NVIDIA Transformer Engine ONNX Export 示例
- Polygraphy 使用文档
- Hugging Face Transformers TFLite 导出指南
- ONNX 官方文档
- ONNX Runtime 官方文档
