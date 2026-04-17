## 核心结论

MoE，Mixture of Experts，中文常译为“混合专家模型”，可以理解为“很多个子网络里，每次只挑少数几个来算”的结构。它的量化难点，不是“参数更多”这么简单，而是“稀疏激活 + 动态路由”同时存在。

结论先写清楚：

1. 稠密模型的 PTQ，Post-Training Quantization，中文是“训练后量化”，直接搬到 MoE 上通常会掉分更明显。原因不是单一误差，而是三类误差叠加：专家校准样本不够、专家间分布不一致、路由器对量化误差极敏感。
2. 最稳妥的工程做法通常不是“全模型统一 4bit”，而是 `FFN 专家权重 W4A16`，也就是前馈网络权重用 4bit、激活保留 FP16；路由器和 LayerNorm 保持 FP16 或 BF16。
3. MoE 的校准集必须做“专家均衡”。白话说，就是不能只拿一批文本直接跑完就量化，因为热门专家会被反复看到，冷门专家几乎没见过，最后量化器只学会照顾热门专家。
4. 路由器最好单独处理。因为路由 logits 只要有轻微偏移，就可能让 top-k 专家选择翻转。专家一旦选错，后面整条计算路径都变了，误差会连锁放大。
5. 如果预算允许，按专家频率做混合位宽通常比统一位宽更合理：高频专家保 INT8，低频专家降 INT4；按组量化，如 `group_size=128`，也比整层一个统一 scale 更稳。

先看一个最小估算。假设某层有 128 个专家，top-2 路由，300 条校准样本按每条只贡献一次路由决策粗略估计，则每个专家平均只被看到：

$$
300 \times 2 / 128 \approx 4.7
$$

这不是“4.7 个 token 很少”这么简单，而是“很多专家几乎没有有效校准统计量”。对稠密层来说，所有权重每次前向都参与；对 MoE 专家来说，大量权重长期处于“没被看见”的状态。

| 目标模块 | 常见量化选择 | 主要风险 | 更稳妥的实践 |
| --- | --- | --- | --- |
| 专家 FFN 权重 | INT4 / INT8 | 冷门专家校准不足 | W4A16，必要时高频专家 INT8 |
| 激活 | INT8 / FP16 | 激活离群值放大量化误差 | 优先保 FP16 |
| 路由器权重 | INT8 | logits 偏移导致 top-k 翻转 | 保 FP16/BF16，或单独做对齐优化 |
| LayerNorm | INT8 | 数值稳定性下降 | 保 FP16/BF16 |

---

## 问题定义与边界

本文讨论的是 MoE 大模型的 PTQ，不是 QAT。QAT，Quantization-Aware Training，中文是“量化感知训练”，会在训练时把量化误差纳入优化；PTQ 则是在模型训练完成后再压缩。两者都能做 MoE 量化，但工程里更常见的是 PTQ，因为成本低、落地快。

边界也要说清楚。这里的对象主要是带路由器的 Transformer-MoE 层，例如一个 token 进入某层后，先通过 gate 产生路由分数，再选 top-1 或 top-2 专家执行 FFN。我们关注的不是 KV Cache 量化，也不是全链路推理调度，而是“专家权重 + 路由器”这两个核心部件在低比特下为什么容易失稳。

MoE 与稠密模型在量化上的本质差异，可以压缩成一句话：

- 稠密模型的问题更像“每层都有误差，但误差路径固定”。
- MoE 的问题则是“误差不仅改变数值，还可能改变路径”。

这里的“路径”就是路由结果。只要 top-k 选择变了，后面的专家就变了，误差不再是局部近似，而是结构级偏移。

一个玩具例子就够说明问题。设某 token 的路由 logits 原本是：

| 专家 | 原始 logit |
| --- | --- |
| E1 | 1.02 |
| E2 | 1.00 |
| E3 | 0.20 |
| E4 | -0.10 |

如果 top-2，原本会选 `E1, E2`。现在把 gate 权重量化后，logits 变成：

| 专家 | 量化后 logit |
| --- | --- |
| E1 | 0.98 |
| E2 | 1.01 |
| E3 | 0.21 |
| E4 | -0.09 |

数值看起来只改了百分之几，但排序已经翻了。若后续实现里还依赖 softmax 权重分配，则不只是“E1 和 E2 交换顺序”，而是整个专家输出加权都变了。

再看覆盖问题。若 128 专家、top-2、300 条校准样本，平均每专家约 4.7 次激活，这只是均值。真实分布通常长尾，热门专家远高于均值，冷门专家可能接近 0 次。于是标准 AWQ、GPTQ 这类依赖少量校准样本估计误差的办法，在 MoE 上首先撞到的不是算法本身，而是样本覆盖失真。

---

## 核心机制与推导

理解 MoE 量化，最关键的是把“数值误差”和“路由误差”分开看。

第一类是专家权重量化误差。它与稠密模型相似，本质是用低比特表示矩阵后，线性层输出发生近似偏差。第二类是路由器误差。它更危险，因为 gate 的输出不是普通隐藏状态，而是决定“哪个专家会被调用”的控制信号。

因此，很多 MoE 专用 PTQ 方法都会对 router consistency，也就是“路由一致性”单独建模。白话说，就是不仅要求量化后 logits 接近原值，还要求它们诱导出的路由分布尽量不变。

一个常见目标可以写成：

$$
\min_\theta \mathbb{E}_{\tilde{x}}\left[
\|\tilde{x}W^{\text{gate}}-\tilde{x}Q_\theta(W^{\text{gate}})\|_2^2
+
\lambda \, \mathrm{KL}\!\left(
\mathrm{softmax}(\tilde{x}W^{\text{gate}})
\;\|\;
\mathrm{softmax}(\tilde{x}Q_\theta(W^{\text{gate}}))
\right)
\right]
$$

这里：

- $\tilde{x}$ 是校准输入，也就是采样得到的隐藏状态。
- $W^{\text{gate}}$ 是原始路由器权重。
- $Q_\theta(\cdot)$ 是量化算子，$\theta$ 表示 scale、zero-point、group 配置等量化参数。
- 第一项 MSE 约束 logits 的绝对误差。
- 第二项 KL 约束 softmax 后的分布差异，也就是尽量保持路由排序和权重分配不变。
- $\lambda$ 是平衡系数，用来控制“数值逼近”和“分布一致”谁更重要。

为什么需要两项同时存在？因为只做 MSE 不够。MSE 更关注整体数值接近，但它不保证排序稳定。对于 top-k 路由，排序往往比绝对值更重要。KL 项则能更直接地惩罚“分布形状变了”的情况，尤其是边界专家之间的相对关系。

| 约束项 | 它控制什么 | 不加时的风险 | 修复效果 |
| --- | --- | --- | --- |
| MSE | 原始 logits 的数值偏差 | logit 整体漂移，softmax 温度变化 | 降低绝对误差 |
| KL | softmax 后的分布偏差 | top-k 翻转、路由权重改写 | 保持专家选择稳定 |

除了路由，一层里不同专家的分布差异也会破坏统一量化。这里的“分布差异”，白话说就是“不同专家见到的数据类型不同，所以权重和激活的尺度也不同”。有的专家常处理代码 token，有的常处理数学推导，有的偏自然语言；它们的激活范围和权重统计量并不一致。

如果整层只用一个统一 scale，结果通常是两头吃亏：

- 为了覆盖大幅值专家，scale 被拉大，小幅值专家的有效分辨率变差。
- 为了照顾小幅值专家，scale 被压小，大幅值专家更容易截断。

所以更合理的办法是 `per-group` 或 `per-expert` 量化。`group_size=128` 的意思不是“每 128 个专家一组”，而通常是“沿权重张量某一维，每 128 个连续元素共享一个 scale”。它的作用是缩小同一组里的分布跨度，让量化尺度更贴近局部统计，而不是用一个整层参数硬套全部权重。

---

## 代码实现

工程上可以把 MoE PTQ pipeline 拆成四步：

1. 用高精度模型跑一遍校准集，记录每层的 gate logits、路由频率、专家输入激活。
2. 按专家频率做再采样，构造更均衡的校准子集。
3. 先量化专家 FFN，再单独优化 router；专家用 group-wise scale，router 用 MSE + KL 对齐。
4. 设定 mixed precision 规则：FFN 权重量化，激活保 FP16，router 和 LayerNorm 保高精度。

下面给一个可以运行的玩具 Python 例子。它不依赖深度学习框架，只演示两个事实：

- 校准覆盖在 MoE 里会非常稀疏。
- 轻微量化误差可能改写 top-k 路由。

```python
from collections import Counter
import math

def avg_expert_hits(num_samples: int, top_k: int, num_experts: int) -> float:
    return num_samples * top_k / num_experts

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_indices(xs, k):
    return sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)[:k]

# 玩具例子 1：覆盖估算
hits = avg_expert_hits(num_samples=300, top_k=2, num_experts=128)
assert abs(hits - 4.6875) < 1e-9

# 玩具例子 2：量化误差导致路由翻转
orig_logits = [1.02, 1.00, 0.20, -0.10]
quant_logits = [0.98, 1.01, 0.21, -0.09]

orig_top2 = topk_indices(orig_logits, 2)
quant_top2 = topk_indices(quant_logits, 2)

assert orig_top2 != quant_top2
assert orig_top2 == [0, 1]
assert quant_top2 == [1, 0]

# 玩具例子 3：KL 可感知分布变化
p = softmax(orig_logits)
q = softmax(quant_logits)
kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
assert kl > 0.0

print("average_hits =", hits)
print("orig_top2 =", orig_top2)
print("quant_top2 =", quant_top2)
print("kl =", round(kl, 8))
```

如果换成真实工程，伪代码大致如下：

```python
def calibrate_moe_layer(layer, calib_tokens, lambda_kl=0.1):
    ref_gate_logits = []
    expert_inputs = {eid: [] for eid in range(layer.num_experts)}
    expert_freq = [0 for _ in range(layer.num_experts)]

    for hidden in calib_tokens:
        logits = layer.gate_fp16(hidden)
        topk = select_topk(logits, k=layer.top_k)
        ref_gate_logits.append(logits)

        for eid in topk:
            expert_freq[eid] += 1
            expert_inputs[eid].append(hidden)

    balanced_inputs = rebalance_by_expert_freq(expert_inputs, expert_freq)

    # 1) 量化专家 FFN，按专家或按组求 scale
    for eid in range(layer.num_experts):
        layer.experts[eid].weight_q = quantize_groupwise(
            layer.experts[eid].weight_fp16,
            bits=pick_bits_by_freq(expert_freq[eid])  # 高频可 INT8，低频可 INT4
        )

    # 2) 单独优化 router，目标是 MSE + KL
    q_gate_weight = init_quant(layer.gate_weight_fp16)
    for hidden, ref_logits in zip(calib_tokens, ref_gate_logits):
        pred_logits = hidden @ dequant(q_gate_weight)
        loss = mse(pred_logits, ref_logits) + lambda_kl * kl_div(
            softmax(ref_logits), softmax(pred_logits)
        )
        update_quant_params(q_gate_weight, loss)

    return layer
```

这个流程里有两个新手最容易忽略的点。

第一，路由器损失不是拿量化权重直接和原始权重做差，而是要比较“输入经过线性层后的 logits”。因为真正决定 top-k 的是输出分布，不是参数张量本身。

第二，“只调 gate weights，不动 expert weights”不是没用，而是先把路径稳定住。实践里往往先保证路由不乱，再去压专家 FFN；如果一上来全部统一量化，很难判断掉分到底来自专家误差还是路由翻转。

下面这张表可以作为默认配置起点：

| Tensor | 推荐 dtype | 原因 |
| --- | --- | --- |
| Expert FFN weight | INT4 或 INT8 | 参数占比最大，压缩收益最高 |
| Expert activation | FP16 | 动态范围大，保精度更稳 |
| Gate weight | FP16/BF16 或单独 PTQ | 路由对误差极敏感 |
| Gate logits | FP16 参考保存 | 用于 MSE + KL 对齐 |
| LayerNorm weight | FP16/BF16 | 稳定性优先 |

真实工程例子可以参考 LMDeploy 的 W4A16 思路：专家 FFN 权重降到 4bit，激活保 FP16，重点敏感模块保更高精度。它不是说“4bit 一定最准”，而是说明在部署场景里，`权重压缩收益` 与 `路由稳定性` 之间，通常要优先守住后者。

---

## 工程权衡与常见坑

MoE 量化最常见的失败，不是算法没写对，而是工程假设错了。

第一个坑是校准样本不均衡。直接从验证集截 128 条或 256 条文本去跑，看起来和稠密模型一样省事，但在 MoE 上常常等于只校准了热门专家。解决办法是专家均衡采样，至少统计每层每个专家的命中次数，再补齐冷门专家样本。

第二个坑是统一 scale。统一 scale 的优点是实现最简单，缺点是最不适合 MoE。专家分布差异大时，统一 scale 会让一部分专家严重欠拟合量化区间。group-wise 量化或 expert-aware 量化通常更合理。

第三个坑是把路由器也一刀切压成 INT8。对普通线性层，INT8 常常足够；对 gate，不一定。因为 gate 的职责不是生成“近似可接受”的中间表示，而是做离散选择。选择一旦错，后面全错。

| 常见坑 | 现象 | 原因 | 规避措施 |
| --- | --- | --- | --- |
| 校准样本不足 | 冷门专家误差特别大 | 长尾专家几乎没见过样本 | 做专家均衡采样 |
| 路由翻转 | top-k 与 FP16 不一致 | logits 微小偏移改变排序 | router 保 FP16，或加 MSE+KL 对齐 |
| 统一 scale | 某些专家掉分异常 | 专家间尺度差异大 | group-wise / per-expert scale |
| 全部统一 4bit | 总体吞吐好但质量掉很多 | 敏感模块被过度压缩 | mixed precision |
| 校准集分布失真 | 线上效果比离线差 | 校准 token 不代表真实请求 | 用真实业务 token 分布做校准 |

校准样本来源也有规则。最好来自真实推理流量，至少要覆盖你关心的任务分布，比如代码、数学、对话、检索问答。如果模型线上大多数请求是代码补全，却拿百科文本去校准，热门专家集合本身都可能变掉，后续所有量化统计都会偏。

混合精度可以用一条简单规则起步：

- 专家 FFN：优先 INT4。
- 高频专家：必要时升到 INT8。
- Gate、LayerNorm、嵌入等敏感模块：保 FP16/BF16。
- 激活：优先保 FP16，除非你已经验证过激活量化不会触发明显退化。

---

## 替代方案与适用边界

如果你追求的是“先稳定上线”，最保守的替代方案其实不是复杂的 MoE 专用 PTQ，而是只量化专家 FFN，完全不碰 router。这样做损失一部分压缩率，但能显著降低路径级错误。

如果你追求的是“同样显存下尽量高精度”，那频率感知位宽会更合适。思路很简单：被频繁选中的专家，对整体质量贡献更大，也更值得分配高位宽。可以写成一个朴素规则：

$$
\text{bits}(e)=
\begin{cases}
8, & \text{freq}(e) > \tau \\
4, & \text{freq}(e) \le \tau
\end{cases}
$$

这里 $\tau$ 是频率阈值。白话说，就是“热专家更贵，冷专家更便宜”。这类思路适合专家热度明显分层、请求分布长期稳定的场景。

还有一种常见方案是按组量化，而不是按专家频率显式分层。它不需要维护在线热度统计，实现更简单，适合离线导出模型格式时直接落地。比如 GGUF 一类格式里常见的 group-wise 量化，本质就是缩小同组元素的统计差异，让局部 scale 更精细。

| 方案 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- |
| 全部统一 4bit | 实现最简单，压缩率高 | 路由和冷门专家容易掉分 | 只适合对精度不敏感的场景 |
| 专家 FFN 4bit，router 保 FP16 | 稳定，落地快 | 压缩率不是最大 | 大多数生产部署的默认起点 |
| 高频专家 INT8，低频专家 INT4 | 质量与显存折中更好 | 需要频率统计与更复杂导出 | 请求分布稳定、专家热度长尾明显 |
| group-wise 量化 | 不依赖在线热度，通用性强 | 不能直接解决路由翻转 | 适合离线模型发布 |
| QAT 或再训练微调 | 精度最好潜力更大 | 成本高、周期长 | 高价值模型、允许训练资源 |

所以，适用边界可以归纳为：

- 如果你是第一次做 MoE 部署，不要从“全模型全张量统一低比特”开始。
- 如果模型路由本身边界很近，很多 token 的 top-1 和 top-2 分差都很小，那 router 更不该量化。
- 如果业务流量稳定，频率感知混合位宽值得做。
- 如果只是离线分发模型文件，希望兼顾兼容性与精度，group-wise 是现实选择。

---

## 参考资料

1. MoEQuant: *Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance*  
   重点：说明 MoE PTQ 的两类失衡问题，即专家间样本不均衡与专家内样本关联差异。  
   访问：https://proceedings.mlr.press/v267/chen25aa.html

2. EAQuant: *Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization*  
   重点：把路由一致性单独建模，用 logits 对齐与分布约束缓解 top-k 翻转。  
   访问：https://arxiv.org/abs/2506.13329

3. LMDeploy W4A16 Quantization 文档  
   重点：工程侧的 W4A16 流程，说明 4bit weight-only 在部署中的实际做法与限制。  
   访问：https://lmdeploy.readthedocs.io/en/v0.5.2/quantization/w4a16.html

4. GGUF 量化方案与 llama.cpp 系列文档  
   重点：按组量化的工程实现思路，适合理解 `group_size` 对精度的意义。  
   访问：https://github.com/ggml-org/llama.cpp

5. DynaExq 相关资料  
   重点：按专家热度动态分配位宽，高频专家保更高精度，低频专家使用更低比特。  
   访问：https://www.emergentmind.com/topics/dynaexq
