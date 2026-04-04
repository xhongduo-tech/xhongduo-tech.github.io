## 核心结论

INT4 量化是把权重或激活压缩到 4 bit，也就是每个数只能落在很少的离散取值上。白话说，原来可以细调的旋钮，现在只剩 16 个档位。它的直接收益是显著降低模型参数、KV cache 和访存带宽开销；如果硬件和内核真正支持低比特计算，还可能带来更高吞吐。

但 INT4 难点不在“能不能存成 4 bit”，而在“存完以后还能不能算对”。核心原因有两个：

1. 表示格太少。INT8 有 $2^8=256$ 个格，INT4 只有 $2^4=16$ 个格，量化步长会明显变粗，误差更容易扩散。
2. 缩放尺度极度敏感。只要某一层里有少数离群点，也就是数值特别大的权重或激活，整层的量程就会被它们拉大，大量小值被压成同一个整数，甚至直接变成 0。

所以，真实系统里几乎不会用“整层统一 INT4”这种朴素做法，而是配合 group-wise scaling、混合精度、KV 单独处理、双重量化、NF4 这类机制。新手可以把它理解成：不是把所有积木都切成同一个高度，而是分成若干小堆，每堆单独选刻度，尽量保留细节。

INT4 不是“比 INT8 再省一点”的线性升级，而是从“轻微失真”进入“非常容易失真”的区间。它能成立，靠的是更精细的分组、格式设计和系统级融合，而不是单纯把 bit 数减半。

---

## 问题定义与边界

先看最常见的对称量化。对称量化就是让正负两侧共用同一个尺度，不单独偏移零点。设浮点权重为 $w$，量化整数为 $q$，则常见写法是：

$$
s = \frac{\max(|w|)}{q_{\max}}, \qquad q = \text{clip}\left(\text{round}\left(\frac{w}{s}\right), -q_{\max}, q_{\max}\right)
$$

对于对称 INT4，通常取 $q_{\max}=7$，所以整数范围近似是 $[-7, 7]$。解量化时再做：

$$
\hat{w} = q \cdot s
$$

关键问题是：$\max(|w|)$ 一旦被少数大值主导，整个层的步长 $s$ 就会变大，小权重会被“挤扁”。

玩具例子如下。设一组权重为 `[0.02, 0.1, 0.5, 0.6]`，则：

$$
s = \frac{0.6}{7} \approx 0.0857
$$

于是映射结果大致如下：

| 原始权重 | $w/s$ | 四舍五入后整数 | 解量化值 |
|---|---:|---:|---:|
| 0.02 | 0.233 | 0 | 0.0000 |
| 0.10 | 1.167 | 1 | 0.0857 |
| 0.50 | 5.833 | 6 | 0.5142 |
| 0.60 | 7.000 | 7 | 0.6000 |

这里最小的 `0.02` 直接变成了 `0`。对白话理解来说，这相当于模型原本还能分辨“很弱但有意义”的信号，现在被当成“完全没有”。如果这种情况大面积发生，注意力分数、门控输出、残差细节都会被抹掉。

INT4 的边界也必须说清楚：

| 场景 | INT4 适用性 | 原因 |
|---|---|---|
| 权重量化 | 较常见 | 权重分布相对稳定，可离线校准 |
| 激活量化 | 更难 | 输入相关，动态范围随样本变化 |
| KV cache 量化 | 可行但敏感 | 会长期累积影响生成质量 |
| decoder-only 全面 W4A4 | 风险高 | 自回归误差累积，输出层和注意力更敏感 |
| encoder-only / encoder-decoder 部分层 | 更稳 | 非自回归或误差不易层层放大 |

这也是为什么“INT4 能不能用”不能脱离模型结构讨论。对 decoder-only 模型，尤其长上下文生成，KV cache 和输出投影层的动态范围往往更大，直接全链路 W4A4 很容易精度骤降。

---

## 核心机制与推导

### 1. 为什么 group-wise scaling 有效

group-wise scaling 就是把一大层权重切成很多小组，每组单独算自己的 scale。白话说，不再让一个特别高的积木决定整栋楼的楼层高度，而是每一小段自己定刻度。

假设把 `[0.02, 0.1, 0.5, 0.6]` 分成两组：

- 第 1 组：`[0.02, 0.1]`
- 第 2 组：`[0.5, 0.6]`

则第一组的尺度变成：

$$
s_1 = \frac{0.1}{7} \approx 0.0143
$$

第二组的尺度变成：

$$
s_2 = \frac{0.6}{7} \approx 0.0857
$$

此时第一组里的 `0.02` 会量化到：

$$
q = \text{round}(0.02 / 0.0143) = 1
$$

解量化后约为 `0.0143`，虽然仍有误差，但至少不再被压成 0。核心逻辑不是“误差消失了”，而是“误差被局部化了”。离群点只污染本组，不再污染整层。

### 2. NF4 为什么常被拿来替代均匀 INT4

NF4 是 QLoRA 引入的一种 4 bit 非均匀量化格式。白话说，它不是把 16 个格子平均铺开，而是把更多格子放在 0 附近，因为神经网络权重通常集中在 0 附近、近似正态分布。

这点很重要：NF4 不是简单的“4 bit 浮点，拆成符号位、指数位、尾数位”那种 IEEE 风格小浮点；它更接近“固定码本”，也就是预先定义 16 个更适合权重分布的代表值，再把每个权重映射到最接近的码本值。

机制可以写成两步：

1. 先做块级归一化，例如按组除以本组 absmax：
   $$
   x = \frac{w}{\max(|w|)}
   $$
2. 再把 $x \in [-1,1]$ 映射到 NF4 码本中最近的一个值。

因此，NF4 的优势主要来自“码本分布更贴近真实权重分布”，不是来自传统浮点的指数表达能力。

### 3. 双重量化为什么能进一步省内存

group-wise scaling 会带来额外开销，因为每组都要存一个 scale。双重量化就是“把 scale 再量化一次”。白话说，先压货物，再压标签。

如果每 128 个权重存一个 FP16 scale，那么 scale 开销约为：

$$
\frac{16}{128} = 0.125 \text{ bit/weight}
$$

这看起来不大，但在数十亿参数模型里仍然可观。把 scale 再压成 8 bit 或更低，可以继续减少总内存占用，同时通常不会显著增加误差。

---

## 代码实现

下面用一个可运行的玩具实现展示“按组对称 INT4 量化”。它不是生产级代码，但能准确说明流程：输入浮点权重，按组求 scale，量化到 INT4，再解量化验证误差。

```python
import math

def quantize_groupwise_int4(weights, group_size=2):
    qmax = 7
    qweights = []
    scales = []

    for i in range(0, len(weights), group_size):
        group = weights[i:i + group_size]
        absmax = max(abs(x) for x in group) if group else 1.0
        scale = absmax / qmax if absmax != 0 else 1.0
        scales.append(scale)

        for w in group:
            q = round(w / scale)
            q = max(-qmax, min(qmax, q))
            qweights.append(int(q))

    return qweights, scales

def dequantize_groupwise_int4(qweights, scales, group_size=2):
    out = []
    idx = 0
    for scale in scales:
        for _ in range(group_size):
            if idx >= len(qweights):
                break
            out.append(qweights[idx] * scale)
            idx += 1
    return out

# 玩具例子：有小值，也有离群点
weights = [0.02, 0.10, 0.50, 0.60]

# 整体一组：小值容易被压扁
q_all, s_all = quantize_groupwise_int4(weights, group_size=4)
dq_all = dequantize_groupwise_int4(q_all, s_all, group_size=4)

# 分两组：小值保留更好
q_g, s_g = quantize_groupwise_int4(weights, group_size=2)
dq_g = dequantize_groupwise_int4(q_g, s_g, group_size=2)

# 断言：第一种做法会把 0.02 压成 0，对应解量化值也是 0
assert dq_all[0] == 0.0

# 断言：分组后 0.02 不再被压成 0
assert dq_g[0] != 0.0

# 断言：分组后的误差更小
mse_all = sum((a - b) ** 2 for a, b in zip(weights, dq_all)) / len(weights)
mse_g = sum((a - b) ** 2 for a, b in zip(weights, dq_g)) / len(weights)
assert mse_g < mse_all

print("global:", q_all, s_all, dq_all, mse_all)
print("group-wise:", q_g, s_g, dq_g, mse_g)
```

如果继续向工程实现推进，流程通常会变成：

| 步骤 | 作用 | 典型做法 |
|---|---|---|
| 分组 | 限制离群点影响范围 | group size 32/64/128 |
| 统计尺度 | 为每组建立量程 | absmax、percentile、校准集统计 |
| 编码 | 存 4 bit 权重 | INT4 或 NF4 码本索引 |
| 压缩 scale | 降低额外存储 | double quantization |
| 融合 kernel | 避免频繁反量化 | GEMM / attention 内核融合 |

真实工程例子是大模型推理服务。以 QServe 为代表的系统并不是单纯把权重存成 W4，而是采用 W4A8KV4：权重 4 bit、激活 8 bit、KV cache 4 bit。原因很直接，激活和部分算子对误差更敏感，保留到 8 bit 更稳，而 KV cache 单独设计内核后又能明显省带宽。

---

## 工程权衡与常见坑

INT4 的第一类坑是“理论压缩了，实际没加速”。如果运行时不断把 INT4 解包回 FP16 再算，低比特省下的带宽会被反量化开销吃掉。可以把它想成：文件明明压缩了，但程序每读一段就重新完整解压一次，吞吐自然上不去。

QServe 指出的核心问题就是这个。已有 INT4 方案在 GPU 上，权重或部分和的反量化可能吃掉 20% 到 90% 的运行时间。因此真正有效的系统设计不是“只做量化”，而是“量化和 kernel 一起改”。

一个简化对比如下：

| 方案 | 访存开销 | 反量化开销 | 吞吐表现 |
|---|---|---|---|
| 直接存 INT4，算前解量化 | 低 | 高 | 常被拖慢 |
| W4A8 + 融合 GEMM | 较低 | 中低 | 更容易稳定提速 |
| W4A8KV4 + 融合 attention | 更低 | 可控 | 大批量服务更有优势 |

第二类坑是“所有层一刀切”。实际最敏感的通常包括：

- 输出层 `lm_head`
- attention 中的 key/value 路径
- 少数含离群点明显的线性层
- 长上下文下反复复用的 KV cache

这些部分一旦失真，不是只错一次，而是会在自回归生成中逐 token 传播。Wu 等人在 ICML 2023 的实验就显示，W4A4 对 encoder-only 和 encoder-decoder 相对稳定，但对 decoder-only 容易出现显著精度下降。

第三类坑是“只看平均误差，不看任务误差”。量化后的均方误差 MSE 小，不代表生成质量一定稳。分类模型更能容忍局部误差，而语言生成对概率排序、长程依赖、少数关键通道更敏感。工程里必须看真实任务指标，例如 perplexity、长文本问答、代码生成和长上下文检索，而不是只看离线重建误差。

---

## 替代方案与适用边界

如果某些层离群点太多，或者模型本身对量化特别敏感，直接全模型 INT4 往往不是最优解。更现实的选择通常是混合精度。

| 模块 | 推荐精度 | 适用性说明 |
|---|---|---|
| 大部分线性层权重 | W4 | 节省显存和带宽，常见主战场 |
| 激活 | A8 或更高 | 动态范围波动大，INT8 更稳 |
| KV cache | KV4 或 KV8 | 取决于内核支持与长上下文要求 |
| `lm_head` / 敏感层 | INT8 / FP16 | 避免输出分布明显失真 |
| 极端离群层 | FP16 保留 | 少量保留高精度比整体崩坏更划算 |

另外两类替代路径也很常见。

第一类是“先适配，再量化”。例如先做 LoRA 微调或使用 GPTQ、AWQ 这类后训练量化方法，让权重分布变得更适合低比特，再落到 INT4。白话说，不是先把灯全调暗，而是先把最刺眼的灯泡处理掉。

第二类是“退一步用 W4A8 或 INT8”。如果部署目标是稳定服务而不是极限压缩，那么 W4A8 往往是性价比最高的点：大头的权重已经省下来了，激活仍保留足够精度，系统实现难度也比纯 W4A4 更可控。

所以适用边界可以总结为：

- 追求极致省显存，且有专用 kernel 支持时，INT4 值得做。
- 追求稳定上线、低风险回归时，优先考虑 W4A8、部分层保留 INT8/FP16。
- 对 decoder-only 长上下文生成，不要默认“全链路 4 bit 一定可用”，应先做分层评估。

---

## 参考资料

1. Wu, Xiaoxia et al. *Understanding Int4 Quantization for Language Models: Latency Speedup, Composability, and Failure Cases*. ICML 2023. 说明了 W4A4 在 encoder-only、encoder-decoder 与 decoder-only 上的差异，以及 decoder-only 的失败案例。  
   URL: https://proceedings.mlr.press/v202/wu23k.html

2. Lin, Yujun et al. *QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving*. MLSys 2025. 重点讲解 W4A8KV4、QoQ、SmoothAttention、权重重排与融合内核如何减少 GPU 反量化瓶颈。  
   URL: https://proceedings.mlsys.org/paper_files/paper/2025/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html

3. Dettmers, Tim et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. NF4 与 double quantization 的主要来源，说明了为什么非均匀 4 bit 格式更适合近似正态分布的权重。  
   URL: https://papers.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html

4. Brenndoerfer, Michael. *INT4 Quantization: Group-wise Methods & NF4 Format for LLMs*. 2026. 对 group-wise、NF4、双重量化给出了适合工程实践的数值化示例。适合做入门后的机制复习。  
   URL: https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms
