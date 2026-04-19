## 核心结论

推荐模型的量化与剪枝，是在尽量不损伤排序效果的前提下，用更低比特表示和更少有效参数降低模型的存储、带宽和推理成本。量化是降比特，把 `FP32`、`FP16` 等浮点参数转成 `INT8` 或更低精度；剪枝是去掉不重要参数，让部分权重不再参与计算。

推荐模型压缩的第一优先级通常不是“把 MLP 变小”，而是先看 `Embedding` 表。`Embedding` 表是一张把离散 ID 映射成向量的大表，例如用户 ID、商品 ID、类目 ID 都会查表得到向量；在大规模推荐系统里，它经常占据绝大多数参数、显存和内存带宽。可以把推荐模型理解成“字典 + 打分器”：`Embedding` 表是大字典，`Dense Tower` 和 `Scoring Head` 是打分器。字典太大时，先压缩字典通常比只压缩打分器更有效。

$$
\text{模型压缩} = \text{低比特表示} + \text{参数稀疏化} + \text{工程落地}
$$

| 维度 | 压缩目标 | 典型收益 | 主要风险 |
|---|---:|---:|---|
| 内存 | 降低参数存储 | 更低显存和内存占用 | 热点表量化后掉点 |
| 推理延迟 | 减少访存和计算 | 更低 `p95 latency` | 稀疏化未必被硬件加速 |
| 吞吐 | 单机处理更多请求 | 更高 QPS | 反量化和索引开销抵消收益 |
| 精度 | 保持排序指标 | 尽量维持 `AUC`、`NDCG` | 校准集不代表线上分布 |

核心判断标准很简单：压缩不是为了让模型文件好看，而是为了在 `AUC`、`NDCG`、延迟、显存和稳定性之间取得可部署的平衡。

---

## 问题定义与边界

本文讨论的是典型大规模推荐模型，不是通用 CNN 或 Transformer。推荐模型通常包含大量稀疏特征，也就是取值空间很大但每次请求只命中少量 ID 的特征，例如用户、商品、城市、设备、类目。模型结构可以简化为：

```text
稀疏特征 ID
   ↓
Embedding 层：把 ID 查成向量
   ↓
Dense Tower：多层感知机，融合连续特征和 Embedding 向量
   ↓
Scoring Head：输出点击率、转化率或排序分数
```

`PTQ` 是训练后量化，即先训练浮点模型，再用校准数据把权重量化；`QAT` 是量化感知训练，即训练或微调时模拟量化误差，让模型提前适应低精度；结构化剪枝是按通道、神经元、行、列或整块删除参数；按表/按行量化是对不同 `Embedding` 表或不同向量行使用不同量化参数。

如果一个模型参数很多，但 90% 都在 `Embedding` 表里，那么剪掉几层 MLP 往往收益有限。真正该先处理的是大表、热点表和高带宽访问路径。

| 对象 | 是否适合直接量化/剪枝 | 说明 |
|---|---:|---|
| 超大 `Embedding` 表 | 适合，但要分表处理 | 常是存储和带宽瓶颈 |
| 热点 `Embedding` 表 | 谨慎 | 高频访问，精度损失会被放大 |
| 长尾 `Embedding` 表 | 通常适合低比特 | 访问少，统计不稳定，需校准 |
| Dense Tower | 适合 `PTQ` 或 `QAT` | 计算密集，工具链成熟 |
| Scoring Head | 谨慎 | 直接影响最终排序分数 |
| 全模型一刀切 INT8 | 不建议 | 推荐模型各模块敏感度不同 |
| 非结构化剪枝 | 只适合特定部署条件 | 没有稀疏 kernel 时不一定加速 |

本文边界是：重点讨论 `PTQ`、`QAT`、结构化剪枝、按表/按行量化，以及它们在推荐模型中的工程适用性；不展开哈希特征、召回索引压缩、蒸馏训练和多目标排序损失设计。

---

## 核心机制与推导

量化的核心是把连续浮点值映射到有限整数空间。`scale` 是缩放比例，用来决定一个整数单位代表多大的浮点间隔；`zero point` 是零点偏移，用来让浮点零尽量映射到整数零附近。

$$
x_q = \text{clip}(\text{round}(x / s) + z, q_{\min}, q_{\max})
$$

$$
\hat{x} = s(x_q - z)
$$

其中 $x$ 是原始浮点值，$x_q$ 是量化后的整数，$\hat{x}$ 是反量化后的近似值，$s$ 是 `scale`，$z$ 是 `zero point`。量化像把精细尺子换成粗尺子，仍能测量，但分辨率变低。

玩具例子：设 `x = [0.2, -0.7, 1.1]`，取 `s = 0.1`，`z = 0`，整数范围足够大，则：

| 原始值 `x` | `round(x / s)` | 量化值 `x_q` | 反量化 `x_hat` |
|---:|---:|---:|---:|
| 0.2 | 2 | 2 | 0.2 |
| -0.7 | -7 | -7 | -0.7 |
| 1.1 | 11 | 11 | 1.1 |

这个例子没有误差，是因为数值刚好落在 `0.1` 的网格上。真实模型权重通常不会这么整齐，所以会产生舍入误差。

剪枝的核心是用掩码屏蔽不重要参数。掩码是一个只包含 0 和 1 的开关，1 表示保留，0 表示剪掉。

$$
w'_i = m_i w_i,\quad m_i \in \{0,1\}
$$

稀疏率表示被剪掉的比例：

$$
\rho = 1 - \frac{||m||_0}{N}
$$

其中 $||m||_0$ 表示掩码中非零元素数量，$N$ 是参数总数。设 `w = [0.4, 0.1, -0.05]`，`m = [1, 0, 0]`，剪枝后 `w' = [0.4, 0, 0]`，稀疏率是 $1 - 1/3 = 2/3$。

| 方案 | 训练时机 | 精度稳定性 | 实现复杂度 | 适用场景 |
|---|---|---:|---:|---|
| `PTQ` | 训练完成后 | 中等 | 低 | 快速试点、模型敏感度低 |
| `QAT` | 训练或微调中 | 高 | 中到高 | 精度敏感、`PTQ` 掉点明显 |
| 结构化剪枝 | 训练后或微调中 | 中等 | 中 | 追求真实加速 |
| 非结构化剪枝 | 训练后或微调中 | 不稳定 | 中 | 有稀疏 kernel 和部署支持 |

真实工程例子：一个广告排序模型包含用户表、商品表、上下文表和一个三层 MLP。用户表和商品表合计数百 GB，MLP 只有几百 MB。此时把 MLP 从 `FP16` 剪到一半，并不能解决单机放不下的问题；更有效的路径是对 `Embedding` 表做 `INT8` 或混合精度压缩，同时保留关键热点表的高精度。

---

## 代码实现

实现时应把三条线分开：`PTQ` 流程、`QAT` 流程、剪枝流程。三者入口不同，验证方式也不同。

| 文件 | 职责 |
|---|---|
| `train_fp32.py` | 训练浮点基线模型 |
| `ptq_calibrate.py` | 收集校准数据并计算量化参数 |
| `qat_finetune.py` | 插入 fake quant 并做短轮数微调 |
| `prune_model.py` | 生成掩码并导出稀疏或结构化模型 |

推荐流程如下：

```text
浮点训练 -> 评估基线 -> 选择 PTQ/QAT/剪枝
        -> 校准或微调 -> 导出模型 -> 压测验证
```

下面是一个可运行的最小 Python 例子，包含量化参数计算、fake quant 前向和掩码剪枝：

```python
import numpy as np

def calc_quant_params(x, qmin=-128, qmax=127):
    x_min, x_max = float(np.min(x)), float(np.max(x))
    scale = (x_max - x_min) / (qmax - qmin)
    if scale == 0:
        scale = 1.0
    zero_point = int(round(qmin - x_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    return scale, zero_point

def quantize(x, scale, zero_point, qmin=-128, qmax=127):
    q = np.round(x / scale + zero_point)
    return np.clip(q, qmin, qmax).astype(np.int8)

def dequantize(q, scale, zero_point):
    return scale * (q.astype(np.float32) - zero_point)

def fake_quant(x, qmin=-128, qmax=127):
    scale, zero_point = calc_quant_params(x, qmin, qmax)
    q = quantize(x, scale, zero_point, qmin, qmax)
    return dequantize(q, scale, zero_point)

def prune_by_mask(w, mask):
    return w * mask

x = np.array([0.2, -0.7, 1.1], dtype=np.float32)
x_hat = fake_quant(x)

w = np.array([0.4, 0.1, -0.05], dtype=np.float32)
m = np.array([1.0, 0.0, 0.0], dtype=np.float32)
w_pruned = prune_by_mask(w, m)

assert x_hat.shape == x.shape
assert np.allclose(w_pruned, np.array([0.4, 0.0, 0.0], dtype=np.float32))
assert np.count_nonzero(m) == 1
```

在推荐系统里，难点不是写出 `quantize()`，而是保持训练、导出和线上服务一致。比如按行量化 `Embedding` 时，每一行可能有自己的 `scale`，导出时必须把权重和量化参数一起保存；线上查表后如果需要反量化，也要保证和离线评估完全一致。

混合精度配置可以直接表达工程策略：

```python
quant_policy = {
    "user_embedding_hot": "fp16",
    "item_embedding_hot": "fp16",
    "item_embedding_tail": "int8_rowwise",
    "context_embedding": "int8_tablewise",
    "dense_tower": "int8_qat",
    "scoring_head": "fp16",
}

assert quant_policy["user_embedding_hot"] == "fp16"
assert quant_policy["item_embedding_tail"].startswith("int8")
```

这段配置表达的原则是：热点表和最终打分头保留较高精度，长尾表和中间层优先压缩。

---

## 工程权衡与常见坑

推荐系统里，压缩成功不等于线上可用。离线校准集、热点 ID、长尾 ID、流量峰值、分片方式和服务框架都会影响结果。你把模型压小了，但线上请求仍然要访问同样多的 `Embedding` 行，甚至还要处理额外反量化逻辑，延迟未必下降。看起来参数少了，不代表真正跑得快。

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 校准集偏差 | 离线正常，线上掉点 | 覆盖热门 ID、长尾 ID、峰值流量 |
| 一刀切量化全部 `Embedding` | 热点表精度损失明显 | 按表、按行做混合精度 |
| 非结构化剪枝无实际加速 | 参数少但延迟不降 | 使用结构化剪枝或稀疏 kernel |
| 从零开始做 `QAT` | 训练成本高且不稳定 | 先训练浮点基线，再短轮数微调 |
| 只看 `AUC` 不看延迟和显存 | 离线指标好但部署失败 | 同时报系统指标和线上指标 |

| 指标类型 | 指标 | 说明 |
|---|---|---|
| 离线排序 | `AUC` | 二分类排序区分能力 |
| 离线排序 | `NDCG` | 排序位置质量 |
| 在线服务 | `p95 latency` | 95% 请求延迟，反映尾部体验 |
| 系统资源 | `memory footprint` | 模型权重、缓存和运行时内存 |
| 稳定性 | 错误率、超时率 | 压缩后服务是否稳定 |

一个常见工程做法是分层验证：先在离线验证集比较 `FP32`、`PTQ INT8`、`QAT INT8`；再在影子流量中看 `p95 latency`、内存、错误率；最后小流量灰度。只要其中一项明显异常，就不要直接全量上线。

---

## 替代方案与适用边界

量化和剪枝不是唯一方案。很多推荐系统应该先做混合精度、分表处理、分片存储或专门的 `Embedding` 压缩，再考虑更激进的稀疏化。目标如果只是降低显存，混合精度量化可能比剪枝更直接；目标如果是真实降低延迟，结构化剪枝、分片部署和高效 kernel 往往比非结构化稀疏更可靠。

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| `PTQ` | 快速、改动小 | 精度不一定稳 | 快速试点、低风险模块 |
| `QAT` | 精度更稳 | 需要训练链路支持 | 精度敏感、`PTQ` 掉点 |
| 结构化剪枝 | 更可能真实加速 | 可能需要微调架构 | 延迟瓶颈在计算层 |
| 非结构化剪枝 | 参数压缩率高 | 没有硬件支持时不快 | 有稀疏算子和部署框架 |
| `Embedding` 低比特压缩 | 直接降低大表成本 | 对热点表敏感 | `Embedding` 远大于 MLP |
| 分片存储 | 解决单机容量 | 系统复杂度更高 | 表规模超过单机内存 |

| 场景 | 推荐优先方案 |
|---|---|
| 只想快速试点 | 先做 `PTQ`，观察掉点和延迟 |
| 对精度极敏感 | 浮点基线后做 `QAT` 微调 |
| 要求真实加速 | 优先结构化剪枝和部署端算子支持 |
| `Embedding` 远大于 MLP | 优先按表/按行量化和分片 |
| 热点 ID 影响大 | 热点表保留 `FP16` 或更高精度 |
| 线上内存是瓶颈 | `Embedding` 压缩优先于剪 MLP |

可以按下面的决策流程选择方案：

```text
先定位瓶颈
├─ 显存/内存瓶颈：优先 Embedding 低比特、混合精度、分片
├─ 延迟瓶颈：优先结构化剪枝、算子融合、推理框架优化
├─ 精度瓶颈：优先 QAT，热点表保留高精度
└─ 快速验证：先 PTQ，再决定是否升级到 QAT
```

最终原则是：先压缩真正占成本的模块，再用线上指标验证收益；不要只因为某个方法在论文里压缩率高，就直接套到推荐模型上。

---

## 参考资料

1. [TensorFlow: Post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)：适合了解 `PTQ` 的基本流程、校准和训练后导出思路。
2. [TensorFlow: Quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training.md)：适合了解 `QAT` 如何在训练中模拟量化误差。
3. [PyTorch: Quantization-aware training for LLMs](https://docs.pytorch.org/blog/quantization-aware-training/)：适合了解 PyTorch 生态中 `QAT` 的实践方式和精度权衡。
4. [TorchRec: Introduction tutorial](https://docs.pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html)：适合了解推荐模型中的稀疏特征、`Embedding` 和训练结构。
5. [TorchRec: Inference API reference](https://docs.pytorch.org/torchrec/inference-api-reference.html)：适合查看推荐模型推理侧 API 和部署相关接口。
6. [JMLR: Sparsity in Deep Learning](https://jmlr.org/beta/papers/v22/21-0366.html)：适合系统理解深度学习稀疏化和剪枝研究脉络。
7. [Google Research: To prune, or not to prune](https://research.google/pubs/to-prune-or-not-to-prune-exploring-the-efficacy-of-pruning-for-model-compression/)：适合理解剪枝是否有效、何时有效，以及和模型压缩的关系。
8. [Google Research: Differentiable Product Quantization for End-to-End Embedding Compression](https://research.google/pubs/differentiable-product-quantization-for-end-to-end-embedding-compression/)：适合了解面向 `Embedding` 压缩的产品量化方法。
