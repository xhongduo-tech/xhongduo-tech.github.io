## 核心结论

模型蒸馏到边缘设备部署，核心不是“把模型做小”这一个动作，而是把教师模型的判别能力尽量迁移到学生模型，再与量化、算子融合、硬件加速协同，最终让模型满足设备预算。这里的设备预算，指目标硬件可接受的时延、功耗、内存和包体上限。

对初级工程师最重要的判断标准是：端侧模型是否可落地，不能只看参数量，也不能只看离线精度。真正需要同时检查的指标通常有 `accuracy`、`P50 latency`、`P95 latency`、峰值 `RAM`、功耗、模型文件大小，以及算子覆盖率。算子覆盖率的意思是，模型里有多少算子真正被目标硬件加速器接管执行。

一个直观例子是：服务器上的大模型分类效果很好，但手机 CPU 跑一次要 300ms，内存占用也超预算。此时常见路径不是直接硬塞上去，而是先用蒸馏把大模型的输出分布教给小模型，再做 `int8` 量化和算子优化，最后在手机上把时延压到例如 30ms 到 60ms 的量级，同时控制功耗和发热。

| 方法 | 主要目标 | 直接收益 | 不能单独保证什么 |
|---|---|---|---|
| 蒸馏 | 把大模型知识迁移给小模型 | 小模型精度更稳 | 不直接保证端侧更快 |
| 量化 | 把浮点表示压成低比特表示 | 模型更小、计算更省 | 不自动解决算子兼容 |
| 加速/融合 | 减少真实执行开销 | 时延下降、吞吐上升 | 不直接提高精度 |

---

## 问题定义与边界

本文讨论的是“训练好的大模型，怎样压缩并部署到边缘设备”。边缘设备，指手机、IoT 模块、摄像头、车载终端、工业盒子这类本地算力有限、功耗敏感、网络不稳定或要求离线工作的设备。讨论重点是 teacher 到 student 的压缩链路，以及部署时的量化和执行优化，不讨论大模型预训练、分布式训练、云端推理服务优化。

蒸馏、量化、delegate/fusion 是三类不同职责的技术。delegate 可以理解为“把一部分算子下发给更适合的硬件后端执行”，例如 NNAPI、GPU、DSP、NPU。fusion 可以理解为“把多个相邻算子合成更高效的执行单元”，减少中间访存和调度开销。它们不是同一件事，但在端侧部署里通常同时出现。

如果只做蒸馏，常见结果是学生模型精度还可以，但文件仍偏大，推理依旧慢。如果只做量化，模型虽然更小，但因为没有蒸馏补偿，小模型本身表达能力不足，精度可能明显下滑。真正能上线的端侧方案，通常是三者配合，而不是押注单点技术。

| 项目 | 作用 | 不解决什么 |
|---|---|---|
| 蒸馏 | 保持小模型任务精度 | 不直接保证端侧速度 |
| 量化 | 减少模型存储和推理开销 | 不自动解决算子不兼容 |
| 算子融合 / delegate | 提升真实运行速度 | 不直接提升精度 |

从边界上说，本文更适合分类、检测、关键词识别、轻量 NLP 编码器这类已有成熟端侧工具链的任务。对于超大生成模型、超长上下文推理、多机协同推理，这套方法仍有价值，但问题结构会复杂得多，不属于这里的核心范围。

---

## 核心机制与推导

蒸馏的核心，是让学生模型学习教师模型的输出分布，而不只学习最终标签。输出分布可以理解为“模型对各个候选类别分别给了多少概率质量”。硬标签只告诉你“正确答案是猫”，软输出还会告诉你“这张图 80% 像猫，15% 像狐狸，5% 像狗”。这部分“相近类别关系”正是蒸馏能帮助小模型保精度的原因。

常见蒸馏形式写成：

$$
p_t = softmax(z_t / T), \quad p_s = softmax(z_s / T)
$$

$$
L = \alpha \cdot L_{sup}(y, z_s) + (1 - \alpha)\cdot T^2 \cdot KL(p_t \parallel p_s)
$$

这里：
- `logits` 指 softmax 之前的原始输出分数。
- `T` 是温度，作用是把分布“拉平”一些，让非最大类别的信息更明显。
- `α` 是监督损失和蒸馏损失之间的权重。
- `KL` 散度可以理解为“两个分布差多远”的度量。

为什么要除以 `T`？因为如果教师输出太尖锐，除了最大类，其他类别概率都接近 0，学生学不到类别之间的细微关系。较高的 `T` 会让分布更平滑，软信息更多。为什么损失里要乘 $T^2$？这是为了在反向传播时保持梯度尺度更稳定，属于蒸馏里的标准处理。

最小数值例子如下：

$$
L_{sup} = 0.30,\quad KL = 0.12,\quad T = 2,\quad \alpha = 0.7
$$

$$
L = 0.7 \times 0.30 + 0.3 \times 2^2 \times 0.12
= 0.21 + 0.144
= 0.354
$$

这个例子说明，训练学生模型时并不是“只学老师”或“只学标签”，而是两部分共同作用。

玩具例子可以用二分类解释。假设任务是区分“猫”和“狗”。硬标签只有 `cat=1, dog=0`。但教师模型对一张模糊图片给出的概率可能是 `cat=0.55, dog=0.45`。这说明图片里狗的特征也很多。学生如果只看硬标签，会把它当成绝对的猫；如果看教师分布，会学到“这是个边界样本”，从而在小模型容量有限时做出更稳的决策。

量化的核心机制不同。量化是把连续浮点数映射到低比特整数表示，例如 `float32 -> int8`。常见仿射量化公式是：

$$
q = round(x / s) + z
$$

$$
\hat{x} = s \cdot (q - z)
$$

其中 `s` 是 `scale`，表示一个整数步长对应多少真实值；`z` 是 `zero-point`，表示整数空间中哪个值对应真实数值 0。白话说，量化是在用更便宜的整数网格近似原来的浮点空间。近似一定会带来误差，所以需要校准，也就是根据代表性数据估计合理的量化范围。

| 机制 | 负责什么 | 直观目标 |
|---|---|---|
| 蒸馏 | 学教师分布 | 保精度 |
| 量化 | 浮点转整数 | 减字节、降算力开销 |
| fusion / delegate | 合并或下沉算子 | 减真实执行开销 |

真实工程例子是手机端离线图像分类。教师模型可能是服务器训练的 `EfficientNet` 或更大的 ViT 变体，学生模型换成 `MobileNetV3`。训练时加入蒸馏，部署时导出 `TFLite int8`，运行时尽量让卷积、激活、池化等算子由 NNAPI、GPU 或 DSP 接管。最终看的不是论文精度，而是目标手机上的 `P50/P95 latency`、温升和电量消耗。

---

## 代码实现

工程实现通常分三层：蒸馏训练、量化导出、端侧验证。顺序通常也是这三步。对初学者来说，最容易忽略的是第三步：在 PC 上导出成功，不代表在真机上可用。

下面是一个可运行的最小 Python 示例，只演示蒸馏损失与简单量化逻辑，不依赖深度学习框架：

```python
import math

def softmax(logits, T=1.0):
    scaled = [x / T for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def cross_entropy_one_hot(target_index, pred_probs):
    eps = 1e-12
    return -math.log(max(pred_probs[target_index], eps))

def kl_divergence(p, q):
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q):
        total += pi * math.log(max(pi, eps) / max(qi, eps))
    return total

def distillation_loss(student_logits, teacher_logits, label_index, T=2.0, alpha=0.7):
    student_probs = softmax(student_logits, T=1.0)
    p_t = softmax(teacher_logits, T=T)
    p_s = softmax(student_logits, T=T)

    l_sup = cross_entropy_one_hot(label_index, student_probs)
    l_kd = kl_divergence(p_t, p_s)
    loss = alpha * l_sup + (1 - alpha) * (T ** 2) * l_kd
    return loss, l_sup, l_kd

def affine_quantize(xs, scale, zero_point):
    q = [round(x / scale) + zero_point for x in xs]
    q = [max(-128, min(127, v)) for v in q]
    x_hat = [scale * (v - zero_point) for v in q]
    return q, x_hat

teacher_logits = [4.0, 1.0, -0.5]
student_logits = [2.5, 1.2, 0.1]
loss, l_sup, l_kd = distillation_loss(student_logits, teacher_logits, label_index=0, T=2.0, alpha=0.7)

assert loss > 0
assert l_sup > 0
assert l_kd >= 0

xs = [0.12, -0.36, 0.88]
q, x_hat = affine_quantize(xs, scale=0.01, zero_point=0)
assert len(q) == len(xs)
assert max(abs(a - b) for a, b in zip(xs, x_hat)) < 0.01

print("loss=", round(loss, 6))
print("quantized=", q)
print("dequantized=", x_hat)
```

如果换成真实训练脚本，逻辑通常是：

1. 冻结 `teacher` 参数，只做前向推理。
2. `student` 前向推理，得到学生 logits。
3. 计算监督损失和蒸馏损失。
4. 按 `α` 融合后反向传播，只更新学生模型。
5. 导出学生模型到 `ONNX`、`TFLite` 或 TensorRT 可接受的中间格式。
6. 用代表性校准集做 `int8` 校准。
7. 在目标设备上测时延、内存、功耗和算子覆盖率。

量化导出可以按这个步骤理解：

| 步骤 | 目的 | 产出 |
|---|---|---|
| 导出浮点 student | 固定推理图 | `fp32` 模型文件 |
| 准备校准集 | 估计激活范围 | 代表性样本 |
| 执行 PTQ/QAT | 生成低精度版本 | `int8` 模型 |
| 检查算子兼容 | 确认后端可执行 | 支持/回退列表 |
| 真机验证 | 评估实际收益 | 时延、内存、功耗结果 |

端侧验证清单建议固定下来：
- 输入预处理是否与训练一致，例如归一化、颜色通道顺序、resize 策略。
- 输出后处理是否一致，例如 softmax、阈值、NMS。
- `P50 latency` 和 `P95 latency` 是否都达标。
- 峰值内存是否低于设备预算。
- 模型包体是否满足下载和存储要求。
- 关键算子是否被 delegate 接管，而不是大面积回退 CPU。

`TFLite / ONNX / TensorRT` 的角色也要分清。它们不是“同一个部署框架的不同名字”，而是不同生态下的中间格式或推理后端。选型通常取决于目标设备和现有训练框架。

---

## 工程权衡与常见坑

端侧部署里最常见的误区，是把“模型变小”误当成“部署成功”。参数量下降，只能说明理论存储压力可能减轻，不能直接推出时延一定下降。真实设备上的瓶颈经常来自访存、算子调度、后处理、线程切换，以及不被加速器支持的算子回退。

第二个高频问题是校准集不代表线上分布。校准集可以理解为“拿来估计激活范围的一小批真实样本”。如果这批数据太干净、太简单、和线上输入差异很大，量化后的范围估计就会失真，导致激活饱和，最终精度明显掉点。

一个真实坑是：模型论文里标注 `int8` 推理很快，但上手机后速度没明显改善。排查后发现，卷积部分被 delegate 接管了，但某些 reshape、custom op、后处理算子不支持，整条链路频繁在 CPU 与加速器之间切换，额外的数据搬运把理论收益吃掉了。

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 只看参数量 | 误判模型是否能部署 | 以目标设备实测为准 |
| 只做蒸馏不做量化 | 包体仍然大 | 蒸馏后继续做 `int8` |
| 校准集不代表线上数据 | 精度掉点明显 | 用代表性样本做校准 |
| 算子不被 delegate 接管 | 加速落空 | 检查算子覆盖率 |
| 只测平均延迟 | 忽略长尾卡顿 | 同时看 `P50/P95` |

实际项目里建议先定预算，再回推模型设计。例如：
- 时延预算：`P95 < 80ms`
- 峰值内存：`< 200MB`
- 功耗：连续运行 10 分钟不过热
- 包体大小：模型文件 `<= 20MB`
- 算子覆盖率：核心计算算子大部分可下沉

如果预算先不清楚，优化方向就会漂移。你可能花很多时间做复杂蒸馏，却发现真正卡住上线的是包体超标，或者后处理太慢。

---

## 替代方案与适用边界

蒸馏不是唯一方案，它适合“教师模型明显更强，而学生模型仍有学习空间”的场景。也就是说，teacher 和 student 的能力差距越清晰，蒸馏通常越有价值。如果学生本来就很小、任务又很难，蒸馏能带来的收益可能有限。

剪枝是另一类方法。剪枝可以理解为“删掉不重要的参数或通道”，目标是减少计算量和存储量。它的难点是结构性约束较多，很多剪枝结果理论上更稀疏，但在实际硬件上不一定更快。QAT，也就是量化感知训练，可以理解为“训练时就模拟量化误差”，通常比纯后训练量化更稳，但训练成本更高。直接换轻量模型则最简单，例如从 ResNet 改成 MobileNet、ShuffleNet、EfficientNet-Lite。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 蒸馏 | 保精度能力强 | 还需配合部署优化 | teacher 明显更强时 |
| 剪枝 | 可减少计算量 | 结构复杂，训练不稳定 | 网络冗余大时 |
| QAT | 量化后精度更稳 | 训练成本更高 | 对精度敏感时 |
| 直接换轻量模型 | 简单直接 | 上限受限 | 资源极紧张时 |

适用边界可以简单记成三条：
- teacher 与 student 差距明显时，蒸馏通常值得做。
- 目标硬件有明确 `int8` 支持时，全整数量化收益更大。
- 算子能被硬件 delegate、TensorRT 或类似后端大面积覆盖时，真实部署收益才会充分释放。

如果目标设备极弱，而任务又要求高精度，常见结论不是“蒸馏一定能解决”，而是需要组合方案：轻量架构 + 蒸馏 + QAT + 算子重写，必要时还要修改任务形式，比如降低输入分辨率、减少类别数、把一阶段任务拆成级联任务。

---

## 参考资料

1. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
2. [Keras Example: Knowledge Distillation](https://keras.io/examples/vision/knowledge_distillation/)
3. [TensorFlow Lite Post-training Quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)
4. [TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates)
5. [NVIDIA TensorRT Performance Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
6. [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
