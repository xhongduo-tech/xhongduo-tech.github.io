## 核心结论

混合精度量化策略的本质，是在同一个模型里不给所有层一刀切地使用同一种位宽，而是把有限的“精度预算”优先给更敏感的层，把更低的位宽留给更耐噪声的层。位宽就是每个数用多少个二进制位表示，位宽越低，内存和算力开销越小，但数值误差通常越大。

它之所以成立，不是因为“低位宽总是足够”，而是因为不同层对量化误差的容忍度差异很大。大矩阵乘法层，尤其是很多 FFN 层，常常参数多、计算重，但对小幅数值扰动更鲁棒；LayerNorm、输出头、残差分支、部分注意力路径则更容易把微小误差放大。

对真实部署来说，决策驱动力通常不是“理论上最省位数”，而是“在目标硬件上最便宜”。因此混合精度不是单纯追求精度，也不是单纯追求压缩，而是在失准门槛、BitOps、显存、带宽、功耗和硬件支持之间找最优点。BitOps 可以直观理解为“按位宽计算后的乘加成本”，它比只看 FLOPs 更能反映量化后的真实计算代价。

一个新手容易理解的结论是：`全部 INT8` 和 `关键层 FP16 + 其余 INT4` 不是“谁更先进”，而是两个预算分配方案。前者简单稳妥，后者更激进。若模型敏感层被正确识别，后者经常能实现“延迟下降约 30%，精度只降约 0.1%”这种更优折中。可以把它理解成两个开关分别控制“速度”和“清晰度”，关键是别把清晰度开关关在最敏感的位置上。

| 位宽类别 | 典型层 | 预期收益 | 常见风险 |
|---|---|---|---|
| INT4 | FFN 大矩阵、部分线性层、部分 KV cache | 显著降内存与 BitOps，功耗更低 | 离群值多时误差放大，可能损伤困惑度或分类边界 |
| INT8 | 大多数线性层、卷积层、较稳定的 attention 投影 | 性能和精度折中较稳 | 对极端预算场景压缩还不够 |
| FP16 / BF16 | LayerNorm、输出头、残差关键路径、敏感 attention 模块 | 稳定数值范围，降低崩溃风险 | 节省资源有限 |
| FP32 | 少数校验节点、参考输出、某些不支持量化的算子 | 最稳定 | 成本最高，通常不适合主路径部署 |

---

## 问题定义与边界

混合精度量化要解决的问题，可以写成一句话：在已知总预算的前提下，选择每一层的位宽，使量化后的性能损失最小。这里的预算可以是模型大小、BitOps、延迟、能耗，或者它们的组合约束。

更形式化地看，设第 $i$ 层量化带来的权重扰动为 $\Delta w_i$，该层对损失的敏感度由 Hessian 近似描述。Hessian 可以白话理解为“损失函数曲面的弯曲程度”，弯得越厉害，说明这层对扰动越敏感。常用近似写法是：

$$
\Delta L \approx \frac{1}{2}(\Delta w)^T H (\Delta w)
$$

如果再加上预算约束：

$$
\min_{\{b_i\}} \sum_i \Delta L_i
\quad
\text{s.t.}
\quad
\sum_i \text{cost}(b_i, i) \le B
$$

其中 $b_i$ 是第 $i$ 层位宽，$\text{cost}$ 可以是参数存储、BitOps 或延迟代理值，$B$ 是总预算。这个式子说明：高敏感层应拿更高位宽，因为相同扰动会造成更大损失；低敏感层可以被更激进地压缩。

边界条件必须先讲清楚，否则策略会失效：

| 边界项 | 含义 | 对策略的影响 |
|---|---|---|
| 硬件可用位宽 | 芯片是否原生支持 INT4/INT8/FP16 混合执行 | 不支持时理论最优方案会退化成慢方案 |
| 校准集代表性 | 用于估计量化范围和误差的样本是否接近线上分布 | 分布偏移会让敏感度评估失真 |
| 部署场景 | 云端 GPU、移动端 NPU、CPU、边缘设备 | 决定预算重点是延迟、功耗还是内存 |
| 流程类型 | PTQ 或 QAT | PTQ 适合已有模型快速部署，QAT 适合可重训场景 |

一个玩具例子最容易说明预算分配。假设有一个 3 层 MLP，总预算是 20 位，不是每个权重 20 位，而是“给 3 层分配的总权重等级预算”这个简化抽象。若第 1 层和第 2 层更敏感，第 3 层较鲁棒，就分配为 `8/8/4`。这和“预算 20 块钱，重要零件花多一点”是同一个逻辑。关键不在于 `8/8/4` 这个数字本身，而在于预算不是平均分，而是按敏感度分。

---

## 核心机制与推导

混合精度策略的核心机制有两部分：一部分来自数学上的“误差最小化”，另一部分来自工程上的“硬件可执行”。

先看误差。量化本质上是在连续值上引入离散化误差。若把某层量化误差视为噪声 $\epsilon_i$，那么敏感层上同样大小的噪声会造成更大的损失变化。二阶近似给出了一个可操作的判断指标：当层的 Hessian trace、最大特征值或其他代理指标更大时，说明这层更脆弱，应保留更高位宽。

推导可以简化成三步：

1. 对未量化模型做校准前向，采样每层激活或权重统计。
2. 对每层试探不同位宽，估计误差代理，如 MSE、KL 散度或 Hessian 对角近似。
3. 在预算约束下，优先把高位宽给“单位成本带来最大损失下降”的层。

因此常见求解不是暴力枚举，而是贪心、动态规划、整数规划或可微搜索。直觉上，这像有限预算下的资源调度问题。

下面这个表格可以帮助新手把“敏感”和“鲁棒”理解为“哪些零件容错高，哪些必须稳”：

| 模块类型 | 典型特征 | 对量化误差的容忍度 | 常见推荐位宽 |
|---|---|---|---|
| 大矩阵乘法（FFN） | 计算量大、参数占比高 | 中到高 | INT4 / INT8 |
| Attention 投影层 | 计算重，但受上下文分布影响 | 中等 | INT8，部分可 INT4 |
| LayerNorm | 直接影响归一化尺度 | 低 | FP16 / BF16 |
| 残差加法路径 | 误差会累积到后续层 | 低到中 | FP16 / BF16 |
| 输出头 / 首尾层 | 直接影响输入编码或最终分布 | 低 | FP16 / INT8 |

再看硬件耦合。很多初学者误以为“位宽越低就一定越快”，这是错误的。只有当目标后端真正支持相应算子，且图编译器能把量化节点融合成高效 kernel 时，低位宽才会兑现成延迟收益。否则会出现三种反直觉情况：

1. 某些 INT4 算子回退到 FP16 或 FP32。
2. 混合位宽导致频繁重排和类型转换。
3. 理论 BitOps 下降了，但实际内存访问更碎，延迟反而上升。

因此完整目标更接近：

$$
\min_{\{b_i\}} \sum_i \Delta L_i + \lambda \cdot \text{LatencyPenalty}(\{b_i\})
$$

这里 $\lambda$ 是把硬件代价并入目标函数的权重。也就是说，混合精度不是只在数学空间里分配位宽，还要接受编译器和硬件的二次筛选。

一个真实工程例子是 LLM 推理。常见做法不是把整个模型都压成一种精度，而是把 FFN、部分 attention 线性层压到 INT4/INT8，把 LayerNorm、残差关键路径、输出头保留在 FP16 或 BF16。这样做的原因很直接：前者吃掉大部分计算和显存，后者决定数值稳定性。很多部署流程还会在导出 ONNX 时把量化意图写进量化节点元数据，再让后端按硬件友好规则做融合和重写。

一个简化的搜索框架可以写成伪代码：

```text
for layer in model:
    for bit in [4, 8, 16]:
        loss_proxy[layer, bit] = estimate_quant_error(layer, bit)
        hw_cost[layer, bit] = estimate_bitops_or_latency(layer, bit)

while budget_not_satisfied:
    choose layer, bit that gives best
    "loss reduction / extra cost" ratio
    update allocation
validate on target hardware
```

这类方法的关键不是算法名字，而是两个判断标准始终同时存在：误差代理和硬件成本代理。

---

## 代码实现

Post-training Quantization，简称 PTQ，就是不重新训练大模型，只用少量校准数据完成量化。混合精度的 PTQ 流程通常分 4 步：

1. 采集 calibration 数据。
2. 逐层评估敏感度。
3. 在预算下搜索位宽分配。
4. 在目标硬件上验证精度和延迟。

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示“按敏感度和成本分配位宽”的核心逻辑。这里把每层误差近似成“敏感度乘以量化噪声”，位宽越低，噪声越大；把成本近似成“参数量乘位宽”。

```python
from dataclasses import dataclass
from itertools import product

@dataclass
class Layer:
    name: str
    sensitivity: float   # Hessian/MSE/KL 等代理，越大越敏感
    params_m: float      # 参数量，单位：million

BITS = [4, 8, 16]

def quant_noise(bit: int) -> float:
    # 简化代理：位宽越高，噪声越小
    return {4: 1.0, 8: 0.35, 16: 0.08}[bit]

def layer_loss(layer: Layer, bit: int) -> float:
    return layer.sensitivity * quant_noise(bit)

def layer_cost(layer: Layer, bit: int) -> float:
    return layer.params_m * bit

def search_best_allocation(layers, budget_bits_m):
    best = None
    best_score = float("inf")
    for alloc in product(BITS, repeat=len(layers)):
        cost = sum(layer_cost(layer, bit) for layer, bit in zip(layers, alloc))
        if cost > budget_bits_m:
            continue
        loss = sum(layer_loss(layer, bit) for layer, bit in zip(layers, alloc))
        if loss < best_score:
            best_score = loss
            best = alloc
    return best, best_score

layers = [
    Layer("input_proj", sensitivity=4.5, params_m=1.0),
    Layer("ffn", sensitivity=1.4, params_m=6.0),
    Layer("output_head", sensitivity=3.8, params_m=1.2),
]

alloc, score = search_best_allocation(layers, budget_bits_m=40.0)
result = dict(zip([l.name for l in layers], alloc))

# 在这个预算下，更敏感的输入层/输出层应优先得到更高位宽
assert result["input_proj"] >= result["ffn"]
assert result["output_head"] >= result["ffn"]

print("best allocation:", result)
print("proxy loss:", round(score, 4))
```

这个代码块对应的直觉是：像尝试不同厚度的笔画来写字，主干笔画不能糊，边缘能省一点。真实系统当然更复杂，但搜索流程就是这个骨架。

如果放到工程环境里，伪代码通常会多三类日志：

```python
def log_trial(layer_name, bit, err_proxy, bitops, latency_ms, acc_drop):
    print(
        f"layer={layer_name}, bit={bit}, "
        f"err={err_proxy:.4f}, bitops={bitops:.2f}, "
        f"latency_ms={latency_ms:.2f}, acc_drop={acc_drop:.3f}"
    )
```

你需要记录的不只是“误差变大了没有”，还包括：

| 日志项 | 作用 |
|---|---|
| `err_proxy` | 观察层级敏感度估计是否稳定 |
| `bitops` | 看理论计算成本是否下降 |
| `latency_ms` | 看目标设备上是否真的更快 |
| `acc_drop` | 看端到端任务指标是否在门槛内 |

真实工程例子可以以一个导出部署流程来理解：先在 PyTorch 或其他框架中得到每层位宽决策，再导出 ONNX，把量化节点写成图结构中的 `QuantizeLinear/DequantizeLinear` 或等价表示，最后交给 TensorRT、ONNX Runtime、NPU 编译器之类的后端决定哪些层可融合、哪些层需要保留高精度。此时混合精度策略不是停留在“论文里的位宽向量”，而是进入“编译器能否兑现”的阶段。

---

## 工程权衡与常见坑

工程里最常见的错误，是把混合精度当成纯数学优化问题。实际部署中，你需要同时权衡四件事：精度、资源、构建复杂度、可维护性。位宽分得越细，理论上越接近最优，但图编译、测试矩阵、回归验证的复杂度也越高。

典型坑和规避方式如下：

| 坑 | 现象 | 根因 | 规避策略 |
|---|---|---|---|
| 首层/尾层量化过猛 | 整体精度突然崩 | 输入编码或输出分布被直接破坏 | 首尾层优先保留 INT8 以上，必要时 FP16 |
| LayerNorm 出问题 | 输出抖动、NaN、任务指标不稳定 | 归一化对范围误差敏感 | LayerNorm 保留 FP16/BF16 |
| 校准集不代表线上分布 | 离线验证好，线上变差 | 范围统计和离群值估计失真 | 使用至少 1000 条代表样本，覆盖主要场景 |
| 硬件不支持特定位宽 | 理论 BitOps 降了，实际更慢 | 算子回退或频繁重排 | 必须在目标设备测 latency，不只看模拟器 |
| 只看层误差不看端到端 | 单层看起来正常，整网失真 | 误差在残差和深层累积 | 做整网验证和关键任务集回归 |
| 混合过细 | 部署图复杂、维护困难 | 位宽组合过多 | 从层级混合开始，不要一上来做通道级混合 |

必须特别强调一个失败场景：只用 100 条非代表性样本做校准。例如这些样本都来自短文本、固定模板或单一类别，导致 LayerNorm 和激活范围估计偏大或偏小。结果是上线后遇到真实分布，输出层的 logits 爆炸，轻则精度掉点，重则出现 NaN。更稳妥的做法是使用不少于 1000 条代表数据，覆盖主要输入长度、主题、类别和极值情况，然后在目标设备完整跑一遍 latency 与任务指标。

还有一个常被忽略的点：理论上 `INT4 < INT8 < FP16` 的成本顺序没错，但在某些 GPU 或加速器上，INT4 kernel 可能没有 INT8 成熟，甚至会触发格式转换和访存瓶颈。这时“关键层 FP16 + 其余 INT4”不一定比“全部 INT8”更快。混合精度策略必须以目标设备实测为准，而不是以论文表格为准。

---

## 替代方案与适用边界

混合精度不是唯一选择，它只是“已有模型、预算明确、希望尽量不重训”时很有吸引力的一类方案。判断是否采用它，可以沿着下面的决策路径思考：

1. 你能不能重新训练或微调模型？
2. 目标硬件是否支持混合位宽高效执行？
3. 延迟、显存、功耗里，哪个是主约束？
4. 允许的精度下降上限是多少？

几种常见方案对比如下：

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 统一 INT8 | 要求稳、工程复杂度低 | 部署成熟，风险可控 | 压缩和加速不够激进 |
| 混合精度 PTQ | 已有模型、无法重训、预算明确 | 能在较小改动下逼近更优折中 | 强依赖校准集和硬件支持 |
| QAT | 可训练、对精度要求高 | 通常比 PTQ 更稳 | 成本高，需要训练流程 |
| 统一 FP16/BF16 | GPU 友好、延迟不是核心瓶颈 | 数值稳定，工程简单 | 压缩收益有限 |
| 蒸馏 + 低秩 + 量化 | 极限压缩场景 | 可能取得更强压缩率 | 流程复杂，调参成本高 |

两个具体 case 更容易判断边界。

Case A：边缘设备部署，内存和功耗都紧，芯片支持 INT4/INT8。此时混合精度通常值得优先考虑。原因是边缘设备上显存和带宽是硬约束，把 FFN、部分投影层降到 INT4，保留 LayerNorm 和输出头为 FP16/INT8，往往能明显降低资源占用，同时把精度损失压在可接受范围内。

Case B：云端 GPU 部署，支持 BF16，吞吐优先，但单次 latency 不是瓶颈。此时未必需要混合精度。若模型本来就能在 BF16 下稳定高效运行，而混合精度带来的工程复杂度、算子兼容风险和回归测试成本更高，那么统一 BF16 反而是更合理的选择。

所以适用边界可以压缩成一句话：当你有明确预算压力、硬件支持混合位宽、又不希望大规模重训时，混合精度是高性价比方案；当模型极度敏感、硬件支持差、或延迟不是主矛盾时，优先考虑统一高精度或更成熟的统一量化方案。

---

## 参考资料

1. Emergent Mind. “Mixed-Precision Quantization.” *Emergent Mind*, updated 8 Oct. 2025, https://www.emergentmind.com/topics/mixed-precision-quantization  
   注：总结了混合精度量化的总体定义、敏感度分配思想和部署导向视角。

2. Emergent Mind. “Adaptive Mixed-Precision Quantization.” *Emergent Mind*, updated 3 Sept. 2025, https://www.emergentmind.com/topics/adaptive-mixed-precision-quantization  
   注：重点在 Hessian 驱动、可微搜索和预算约束下的位宽分配方法。

3. Pandey, Nilesh Prasad, et al. “A Practical Mixed Precision Algorithm for Post-Training Quantization.” *DeepAI*, 10 Feb. 2023, https://deepai.org/publication/a-practical-mixed-precision-algorithm-for-post-training-quantization  
   注：强调只用小规模无标签校准集即可做实用型 PTQ 混合精度搜索，适合工程落地理解。

4. Spuler, David. “Mixed Precision Quantization.” *AussieAI*, 10 Mar. 2026, https://www.aussieai.com/research/mixed-precision-quantization  
   注：从 LLM 视角整理了层级、块级和更细粒度混合位宽研究。

5. SystemOverflow. “Quantization Failure Modes and Mitigation Strategies.” *SystemOverflow*, accessed 4 Apr. 2026, https://www.systemoverflow.com/learn/ml-model-optimization/model-quantization/quantization-failure-modes-and-mitigation-strategies  
   注：适合补工程实践中的失败模式，如敏感层崩溃、校准集偏移、硬件回退与 NaN/Inf 风险。

6. ApX Machine Learning. “Mixed-Precision Quantization.” *ApX LLM Compression & Acceleration Course*, https://apxml.com/courses/llm-compression-acceleration/chapter-2-advanced-quantization-techniques/mixed-precision-quantization  
   注：从 LLM 压缩课程视角解释为何在 FFN、Attention、KV cache 和归一化层上采用不同精度。
