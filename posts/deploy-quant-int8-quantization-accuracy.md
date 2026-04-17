## 核心结论

INT8 量化的精度分析，本质是在回答一个工程问题：把模型里的浮点数压到 `[-128, 127]` 这 256 个整数后，任务指标会掉多少，换来的吞吐、显存、带宽收益值不值。这里的“量化”就是把连续实数映射到有限整数格点，目的是让硬件用更便宜的整数运算替代浮点运算。

精度损耗通常不是随机发生的，而是集中来自两类因素：一类是 `scale` 估计不准，导致大量常规值被粗糙取整；另一类是 `outlier`，也就是少量特别大的离群值，它会把整层量化刻度拉粗，让绝大多数正常值一起受损。QAT（Quantization Aware Training，量化感知训练，白话说就是“训练时提前让模型适应量化误差”）通常能把精度保持在接近 FP32 的水平；PTQ（Post-Training Quantization，训练后量化，白话说就是“模型训完后再直接压缩”）则强依赖校准样本是否覆盖真实输入分布。

一个常被引用的真实工程结论来自视觉检测场景：PeopleNet 系列模型在 QAT 下，INT8 的 mAP 可以做到约等于 FP32 基线的 `99.7%`；而在校准不足或分布不匹配的 PTQ 场景下，mAP 可能直接掉到不到 `60%`。这说明“INT8 是否可用”不是只看位宽，而是看误差控制机制是否成立。

| 方案 | 数值表示 | 典型精度表现 | 典型加速表现 | 结论 |
| --- | --- | --- | --- | --- |
| 无量化 | FP32 | 基线 mAP = 100% | 基线 | 精度最好，成本最高 |
| PTQ | INT8 | 好的校准下可接近基线；失败案例可掉到 `<60%` | 若算子覆盖完整，通常可获得显著吞吐提升 | 成本低，但稳定性最依赖样本分布 |
| QAT | INT8 | 常见可达 FP32 基线的 `99%+`，案例中约 `99.7%` | 与 PTQ 同级，前提是最终都落到 INT8 内核 | 精度最稳，但训练成本更高 |

---

## 问题定义与边界

问题可以定义为：给定权重和激活，把它们从高精度浮点数压缩到 256 个离散等级后，如何在代表输入上尽量保持任务准确率。这里的“激活”就是网络中间层的输出，白话说就是“数据在每一层里流动时产生的临时数值”。

最基础的量化映射可以写成：

$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{scale}\right), q_{\min}, q_{\max}\right)
$$

其中 `round` 表示四舍五入，`clip` 表示截断到整数范围内。对 INT8 对称量化来说，通常是 $q_{\min}=-127,\ q_{\max}=127$。这一步做的事情很直接：先除以刻度，把浮点数缩放到整数格点附近；再取整；最后防止越界。

新手可以先用一个玩具例子理解。假设你有一张图的亮度值，本来可以在 `0-255` 之间自由变化，现在你硬要只保留 `0-15` 这 16 档。问题不只是“档位变少了”，更关键的是“每一档对应多大范围”由谁决定。如果刻度定得太粗，细节全糊；如果刻度定得太细，亮处又会溢出。INT8 量化里每层的 `scale`，干的就是这个“定档位间距”的工作。

边界也要说清。INT8 不是自动等于低损失。它更适合卷积、全连接这类数值分布相对稳定、硬件支持成熟的部分；对首层、尾层、残差汇合点、注意力层这类敏感位置，误差容易被后续非线性或归一化放大。也就是说，量化分析关注的不是单点误差，而是误差在整张计算图中如何传播。

---

## 核心机制与推导

最常见的是对称量化。它假设一层张量的实数范围大致落在 $[-amax, amax]$ 内，于是定义：

$$
scale = \frac{amax}{127}
$$

$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{scale}\right), -127, 127\right)
$$

$$
x' = q \cdot scale
$$

其中 $x$ 是原始浮点值，$q$ 是量化后的整数，$x'$ 是反量化后的近似值。`amax` 可以理解成“这层允许覆盖到的最大幅值”。如果 `amax` 取太小，会发生截断；如果 `amax` 取太大，整数格点就会变稀，导致常见值的分辨率下降。

手工算一个最小例子。若某层权重范围在 `[-1, 1]`，那么：

$$
scale = \frac{1}{127} \approx 0.007874
$$

对一个权重 $w=0.35$：

$$
q = \operatorname{round}(0.35 / 0.007874) = \operatorname{round}(44.45) = 44
$$

反量化得到：

$$
w' = 44 \times 0.007874 \approx 0.346
$$

误差约为：

$$
|w - w'| \approx 0.004
$$

这个误差看起来不大，但如果某层特别敏感，或者很多层连续叠加，就可能转成可见的指标下降。

为什么 `outlier` 会伤精度？看一个直方图式的思路：假设一层里 99.9% 的激活都落在 `[-0.5, 0.5]`，只有极少数值突然冲到 `4.0`。如果你为了覆盖这个峰值，把 `amax` 设成 `4.0`，那么：

$$
scale = \frac{4.0}{127} \approx 0.0315
$$

此时原本集中在 `[-0.5,0.5]` 的大多数值，只能用很粗的间隔表示，误差明显上升。可以把它想成下面这种分布：

```text
常规激活:        ████████████████████████
离群激活:                               █
数值区间:   -0.5 ............ 0.5 ................. 4.0
amax 取 4.0 后，整层都要按 4.0 的尺度切分
```

这就是为什么精度分析常常最后落到两个核心控制点：

| 控制点 | 它解决什么 | 失效后会怎样 |
| --- | --- | --- |
| `scale` 估计 | 让整数格点覆盖主要分布 | 大量正常值取整误差变大 |
| `outlier` 处理 | 防止少数峰值主导全层刻度 | 大部分值被迫用过粗的量化步长 |

QAT 和 PTQ 的差别，也正体现在这里。QAT 在前向图里插入 fake-quant，也就是“假量化层”，白话说是“训练时先模拟一次取整和截断，但梯度还照常近似传回去”。模型会逐步把权重和激活分布调整到更适合整数表示。PTQ 则不改训练，只在部署前拿一批代表样本统计 `amax`。所以 QAT 是“让模型适应噪声”，PTQ 是“让刻度适应数据”。

真实工程例子可以直接看视觉检测模型。以 NVIDIA TAO 中的 PeopleNet/DetectNet_v2 路线为代表，QAT 版本的 INT8 推理 mAP 通常只比 FP32 略低，接近 `99.7%` 基线；而 PTQ 若校准覆盖不足，mAP 可掉到 `<60%`。同样是 INT8，差距不是来自位宽本身，而是来自“量化参数是否与模型和数据分布匹配”。

---

## 代码实现

落地实现通常分三步：统计 `amax`，计算 `scale`，执行量化与反量化。下面这个玩具代码可以直接运行，展示单层对称 INT8 量化的核心逻辑。

```python
import numpy as np

def quantize_symmetric_int8(x: np.ndarray):
    max_val = np.max(np.abs(x))
    scale = max_val / 127 if max_val > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    x_dequant = q.astype(np.float32) * scale
    return q, x_dequant, scale

# 玩具例子：手工验证 0.35 的量化误差
x = np.array([0.35, -0.82, 0.0, 1.0], dtype=np.float32)
q, x_dequant, scale = quantize_symmetric_int8(x)

assert q.dtype == np.int8
assert np.isclose(scale, 1.0 / 127.0, atol=1e-6)
assert q[0] == 44
assert np.isclose(x_dequant[0], 44 * scale, atol=1e-6)
assert abs(float(x[0] - x_dequant[0])) < 0.01

# outlier 对 scale 的影响
x_small = np.array([0.10, 0.12, 0.08, -0.09], dtype=np.float32)
_, _, s_small = quantize_symmetric_int8(x_small)

x_with_outlier = np.array([0.10, 0.12, 0.08, -0.09, 4.0], dtype=np.float32)
_, _, s_outlier = quantize_symmetric_int8(x_with_outlier)

assert s_outlier > s_small
```

这段代码对应的逻辑就是：

```python
scale = max_val / 127
q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
x_dequant = q * scale
```

PTQ 的关键不在这三行，而在 `max_val` 从哪来。常见做法是部署前拿一批代表样本跑一轮前向，把每层激活的最大绝对值、分位数或直方图统计出来，再选合适的 `amax`。如果代表样本不足，统计出来的范围就会偏掉。

QAT 则是在训练阶段给每个待量化节点挂 fake-quant hook。伪代码可以写成：

```python
def fake_quant(x, amax):
    scale = amax / 127
    q = clip(round(x / scale), -127, 127)
    return q * scale  # 前向看起来像 INT8，反向通常用直通估计
```

训练时前向使用 `fake_quant(x)`，这样模型在更新权重时，就会主动避开那些一量化就失真的分布区域。工程上常见做法是只对卷积、线性层以及它们的激活插 fake-quant，对特别敏感的首尾层和注意力投影层保留更高精度。

---

## 工程权衡与常见坑

真正上线时，INT8 的核心不是“能不能量化”，而是“哪些层该量化到什么程度”。一个常见坑是首层、尾层、注意力层盲目全量化。因为这些层往往直接接触原始输入、分类边界或长程依赖，对微小数值变化更敏感。

另一个常见坑是统一使用 per-tensor scale，也就是整层共享一个刻度。白话说，就是“一整层所有通道都用同一把尺子”。如果某几个通道出现峰值，整层都会被它拖粗。真实工程里，某注意力层统一 scale 后精度掉 `5%`，后来改成 per-channel 量化并对激活做 outlier clipping，才把失真追回来。per-channel 的意思是“每个通道单独定刻度”，这样更细，但实现和硬件支持也更复杂。

| 常见坑 | 典型现象 | 常见对策 |
| --- | --- | --- |
| 首层/尾层敏感 | 指标突然明显下滑 | 保留 FP16/FP32，做混合精度 |
| 校准样本与部署分布不一致 | PTQ 离线看着正常，上线后恶化 | 用 `>1000` 条代表样本覆盖真实场景 |
| outlier 拉大 scale | 大多数正常值分辨率下降 | percentile/clipping、per-channel |
| 算子无 INT8 内核 | 实际吞吐不升反降 | 先检查部署框架的 operator coverage |
| 残差分支量纲不一致 | 融合点误差累积 | 单独校准分支或保留高精度 |
| 注意力层统一量化 | 长上下文任务掉点明显 | attention projection 保留 FP16 |

需要强调一个工程事实：QAT 和 PTQ 最终推理速度不一定因为“方法不同”而不同。它们只要最后都落成同样的 INT8 kernel，理论吞吐相近。真正拉开差距的，往往是有没有因为精度问题被迫回退部分层到 FP16/FP32，以及部署框架是否真的支持全图 INT8。

---

## 替代方案与适用边界

如果模型对量化极其敏感，不必执着于“全模型 INT8”。BF16（Brain Floating Point 16，白话说就是“指数范围接近 FP32、但尾数更短的 16 位浮点”）和混合精度，往往是更稳的折中方案。尤其在 Transformer、检测头、排序模型这类边界敏感任务上，保留关键路径的高精度，比追求全量 INT8 更实际。

一个典型真实工程例子是 Transformer 推理：如果 self-attention 的投影层量化后精度掉得太多，可以保留 attention projection 为 FP16，把其余卷积、MLP、线性层量化为 INT8。这样吞吐未必达到“全量 INT8 理论上限”，但总体收益通常比“全量压下去后再因错误回退”更稳定。

| 方案 | 适用场景 | 精度风险 | 吞吐收益 | 说明 |
| --- | --- | --- | --- | --- |
| INT8 全量化 | 卷积/FC 主导、硬件支持完整 | 中到高 | 高 | 最看重校准与敏感层处理 |
| INT8 混合精度 | 有少数敏感层 | 低到中 | 中到高 | 工程上最常见 |
| BF16 | 大模型、注意力敏感、想少改训练 | 低 | 中 | 容错高，但压缩率不如 INT8 |

适用边界可以直接概括为三句话。第一，如果代表样本足够、分布稳定、硬件支持完善，PTQ 是最低成本方案。第二，如果任务指标很硬，且模型允许再训练，QAT + per-channel + activation clipping 往往是 INT8 边界内最稳的路线。第三，如果 attention、首尾层、长尾样本对业务影响极大，就应该接受“不是所有层都适合 INT8”这个现实。

---

## 参考资料

1. NVIDIA TensorRT 文档《Working With Quantized Types》：用于确认对称量化公式、量化类型和部署流程。  
2. NVIDIA Developer Blog《Achieving FP32 Accuracy for INT8 Inference Using Quantization-Aware Training with TensorRT》：用于 QAT/PTQ 的机制说明和精度对比。  
3. NVIDIA Developer Blog《Achieving FP32 Accuracy for INT8 Inference with Quantization Aware Training in the NVIDIA TAO Toolkit》：用于 PeopleNet 等视觉模型的真实工程案例。  
4. SystemOverflow《Quantization Failure Modes and Mitigation Strategies》：用于整理校准偏差、outlier、敏感层、算子覆盖等失败模式。  
5. Emergent Mind《GPU-Accelerated INT8 Quantization》：用于补充量化误差、离群值与硬件推理路径的背景。
