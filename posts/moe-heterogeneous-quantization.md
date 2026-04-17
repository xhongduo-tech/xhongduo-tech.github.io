## 核心结论

MoE（Mixture of Experts，专家混合模型，指“模型里有很多专家子网络，但每个 token 只会调用其中少数几个”）量化，最有价值的做法通常不是“全模型统一 4-bit”，而是“按专家重要性分配不同位宽”。原因在于：MoE 的大部分参数集中在专家层，但推理时只有少量专家参与计算，所以不同专家被压低精度后，对最终输出的伤害并不相同。

对部署最关键的结论有三条。

第一，只量化专家，往往就已经覆盖了主要收益。QMoE、MoQE 一类方法的共同点是把压缩重点放在专家权重，而不是先改 router 或共享层。这样做的直接收益是模型体积和权重带宽先明显下降，而行为变化通常比“全模型一起激进量化”更可控。MoQE 的公开结果表明，只量化专家权重、并让低重要专家承担更激进的低 bit，就可以把模型体积压到原 FP16 的约 20%，并带来约 1.24 倍推理加速。

第二，专家级混合精度比统一位宽更符合 MoE 的结构。统一 4-bit 的假设是“所有专家同样重要”，但这在 MoE 中通常不成立。高频专家，也就是 router 经常选中的专家，对困惑度和下游任务更敏感；低频专家即使压到 2-bit，整体退化也未必明显。因此，“高频专家保留 8-bit 或 4-bit，低频专家下探到 3-bit 或 2-bit”通常能得到更好的精度-显存折中。

第三，极低 bit 场景下，不能只看激活频率，还要看全局敏感度。GEMQ 这类方法强调的不是“哪一层谁最常被调用”，而是“整个模型里，哪些专家一旦量化就最容易放大损失”。这类全局分配能避免一个常见错误：某些专家调用频率不算高，但它们一旦被压得过低，会破坏后续路由或把误差传播到更多层。

一个直观例子是 Mixtral-8x7B。它总参数约 47B，但每个 token 常见是 top-2 路由，所以单 token 的活跃参数规模接近 13B。这里要区分两个概念：模型总权重显存，和单步实际参与计算的活跃路径。部署时如果共享部分保持较稳的位宽，再对专家做 2-bit 到 8-bit 的异构量化，单卡部署门槛会显著下降；公开工作也反复说明，只要高敏感专家不过度压缩，困惑度上升可以控制在工程可接受范围内。结论不是“所有 MoE 都能无损压到极低 bit”，而是“异构量化已经是 MoE 部署里最值得优先验证的路线”。

---

## 问题定义与边界

MoE 量化讨论的不是“能不能量化”，而是“有限显存预算下，哪些专家应该保留更高精度”。边界先说清楚：

1. 这里讨论的是推理阶段的权重量化，重点是专家权重。
2. 默认对象是稀疏激活的 Transformer MoE，而不是传统 dense 模型。
3. 重点优化目标通常是权重显存、权重带宽和可接受精度，而不是训练成本。

MoE 的基本结构可以简化成三部分：

| 组件 | 作用 | 量化时的典型处理 |
| --- | --- | --- |
| 共享层 | 所有 token 都会经过的公共变换 | 通常保守量化，常见是 8-bit 或 4-bit |
| Router | 决定 token 去哪些专家 | 一般更谨慎，极低 bit 风险较大 |
| Experts | 真正承载大部分参数容量的子网络 | 是量化主战场 |

对很多开源 MoE 模型，90% 以上参数都集中在专家矩阵里。这意味着一个直接事实：如果专家不动，只调共享层，整体收益通常有限；如果专家位宽能降下来，体积和带宽才会真正下降。

设第 $i$ 个专家有：

- 参数量 $n_i$
- 激活频率 $f_i$
- 位宽 $b_i$
- 在位宽 $b_i$ 下的误差代价 $c_i(b_i)$

那么混合精度分配可以写成：

$$
\min \sum_i c_i(b_i)
\quad
\text{s.t.}\quad
\sum_i n_i b_i \le B_{\text{total}},\;
b_i \in \{2,3,4,8,16\}
$$

这个式子读成白话就是：总 bit 预算固定时，把高 bit 留给“最怕被量化坏”的专家。

但 $c_i(b_i)$ 不是一个固定数字，它至少受三类因素影响：

| 因素 | 白话解释 | 影响 |
| --- | --- | --- |
| 激活频率 $f_i$ | 这个专家被 router 选中的次数多不多 | 选中越多，误差越容易累计 |
| 参数规模 $n_i$ | 这个专家有多大 | 体积越大，降 bit 的节省越明显 |
| 敏感度 $s_i$ | 这个专家一旦量化会不会明显拉高损失 | 决定“能不能压狠” |

所以“高频专家高位宽”只是一个经验起点，不是完整答案。更准确的说法是：bit 应该分给“单位 bit 带来最大精度收益”的专家。

### 玩具例子

假设 4 个专家参数量分别为 4GB、3GB、2GB、1GB，FP16 下总共 10GB；频率分别为 60%、25%、10%、5%。如果统一 4-bit，那么静态权重显存是：

$$
10 \times \frac{4}{16} = 2.5\text{GB}
$$

如果做异构量化，分配成：

- 专家 1：8-bit
- 专家 2：4-bit
- 专家 3：4-bit
- 专家 4：2-bit

那么静态权重显存是：

$$
4\times\frac{8}{16} + 3\times\frac{4}{16} + 2\times\frac{4}{16} + 1\times\frac{2}{16}
= 3.625\text{GB}
$$

它比统一 4-bit 更大，但精度往往更稳。这里恰好体现出 MoE 异构量化的本质：它不是无条件追求最低显存，而是在相近预算下，把显存花在更值钱的专家上。

---

## 核心机制与推导

专家级异构量化可以拆成四步：统计、建模、分配、补偿。

### 1. 统计专家重要度

先用 FP16 或 BF16 跑一遍校准集。校准集就是一小批代表真实业务分布的数据，用来估计量化后哪部分最容易出问题。常见统计包括：

- 每个专家的路由频率
- 每个专家输出的激活分布
- 量化前后损失变化
- 梯度、Hessian 近似或其他敏感度指标

一个常见的简化代价函数是：

$$
c_i(b) \approx (\alpha f_i + \beta s_i)\, e_i(b)
$$

其中：

- $e_i(b)$ 是专家 $i$ 在位宽 $b$ 下的量化误差估计
- $f_i$ 是激活频率
- $s_i$ 是敏感度
- $\alpha,\beta$ 是人为设定或搜索得到的权重

它表达的就是：同样的量化误差，落在高频或高敏感专家上会更危险。

### 2. 在全局预算下分配位宽

如果位宽只能从 $\{2,3,4,8,16\}$ 里选，本质上就是一个离散优化问题。GEMQ 强调“全局分配”，意思不是每层自己给专家定 bit，而是整个模型共用同一个预算池。

这一步为什么重要？因为 MoE 的误差不是局部消失的。一个上游关键专家被压坏，后面多层都可能接收更差的隐藏状态，甚至连 router 的偏好都会跟着变。

下面这个表可以快速建立直觉：

| 位宽 | 相对 FP16 的权重显存 | 精度风险 | 常见适用对象 |
| --- | --- | --- | --- |
| 16-bit | 1.00x | 最低 | 极高敏感模块、router 邻近关键块 |
| 8-bit | 0.50x | 很低 | 高频专家、共享层 |
| 4-bit | 0.25x | 可控 | 大多数中频专家 |
| 3-bit | 0.1875x | 明显上升 | 中低频专家 |
| 2-bit | 0.125x | 最高 | 低频且低敏感专家 |

### 3. 按专家独立量化

拿到 $b_i$ 后，再对每个专家独立使用 GPTQ、AWQ 或其他 PTQ（Post-Training Quantization，后训练量化）方法。这里的关键变化不是“换了某种神奇量化器”，而是量化单位从“整层”变成“单个专家”。

这一步对 MoE 特别有效，因为专家天然是相对独立的子模块。把一个冷门专家从 4-bit 继续压到 2-bit，影响的是它自己的路径，而不是像 dense FFN 那样把所有 token 都一起拖下水。

### 4. 修复路由漂移

路由漂移是 MoE 量化里最容易被忽略的问题。量化后专家输出分布改变，router 看到的表征也会改变，最终导致“该选的专家没被选，不该选的专家反而进入 top-k”。

因此极低 bit 方案往往还要做补偿，常见做法有：

- 冻结专家，只微调 router
- 对路由相关参数做小规模校正
- 渐进式量化，每量化一部分就重新统计一次

很多工作都指向同一结论：2-bit 或更低 bit 时，如果完全不处理路由漂移，模型很容易不是“略差一点”，而是“行为模式变了”。

### 真实工程例子：Mixtral-8x7B

Mixtral-8x7B 是理解这件事最直观的样本。它的总参数约 47B，但每个 token 只激活 8 个专家中的 2 个，所以活跃参数约 13B。这里有三个工程含义：

1. 总参数规模很大，不等于单 token 计算量也同样大。
2. 显存瓶颈经常来自“存下所有专家权重”，而不是单次 matmul。
3. 低 bit 的价值不仅是省显存，还包括降低专家切换时的带宽压力。

公开工作如 QMoE、MoQE、MxMoE 的结果共同说明：对 MoE，专家级混合精度通常比统一位宽更容易拿到更好的精度-速度-显存平衡；但是否能真正提速，还取决于后端 kernel 是否支持混合 bit 的 GroupGEMM 或 fused execution。也就是说，算法分配出好 bit 只是第一步，系统执行层能不能跟上，决定了它是不是“工程可用”。

---

## 代码实现

下面用一个最小可运行的 Python 例子演示“在固定预算下，为不同专家搜索不同 bit”。它不依赖深度学习框架，只演示分配逻辑，便于先把思路看清楚。

```python
from itertools import product

experts = [
    {"name": "e0", "params_mb": 4000, "freq": 0.60, "sens": 1.00},
    {"name": "e1", "params_mb": 3000, "freq": 0.25, "sens": 0.80},
    {"name": "e2", "params_mb": 2000, "freq": 0.10, "sens": 0.45},
    {"name": "e3", "params_mb": 1000, "freq": 0.05, "sens": 0.20},
]

BITS = (2, 4, 8, 16)


def memory_cost(expert, bit):
    # 以 FP16 为基准按比例缩放
    return expert["params_mb"] * bit / 16


def error_cost(expert, bit):
    # 一个可解释的近似代价：
    # 位宽越低，误差越大；频率和敏感度越高，代价越大
    quant_error = (16 / bit) - 1
    return expert["freq"] * expert["sens"] * quant_error


def total_memory(assign):
    return sum(memory_cost(e, assign[e["name"]]) for e in experts)


def total_error(assign):
    return sum(error_cost(e, assign[e["name"]]) for e in experts)


def brute_force_assign(budget_mb):
    best_assign = None
    best_error = float("inf")
    best_memory = None

    for choice in product(BITS, repeat=len(experts)):
        assign = {experts[i]["name"]: choice[i] for i in range(len(experts))}
        used = total_memory(assign)
        if used > budget_mb:
            continue

        err = total_error(assign)
        if err < best_error:
            best_assign = assign
            best_error = err
            best_memory = used

    if best_assign is None:
        raise ValueError("budget too small")

    return best_assign, best_memory, best_error


assign, used, err = brute_force_assign(budget_mb=3000)

print("assignment:", assign)
print("memory_mb:", used)
print("approx_error:", round(err, 4))

assert used <= 3000
assert set(assign.keys()) == {"e0", "e1", "e2", "e3"}
assert set(assign.values()).issubset(set(BITS))
assert assign["e0"] >= assign["e3"]  # 高频高敏专家通常不会比冷门专家更低
```

这段代码和真实系统的差别在于：

| 这里做的事 | 真实工程里会怎么替换 |
| --- | --- |
| `freq` 手工给出 | 来自校准集的真实路由统计 |
| `sens` 手工给出 | 来自损失变化、梯度或 Hessian 近似 |
| `error_cost` 是简化公式 | 会用更贴近量化误差的代理指标 |
| 暴力搜索全部组合 | 真实模型会用贪心、动态规划、LP 或启发式搜索 |

如果继续往工程实现推进，流程通常是：

1. 用 FP16 模型跑校准集，记录 `expert -> {freq, sens}`。
2. 在总预算下计算 `expert -> bit` 映射。
3. 对每个专家分别执行 GPTQ/AWQ 类量化。
4. 检查量化前后的路由分布差异。
5. 如有明显漂移，只微调 router 或做小规模全局校正。
6. 部署时把高频高 bit 专家常驻 GPU，把低频低 bit 专家放到次级存储或异步加载路径。

新手最容易忽略的一点是：代码里“选 bit”只是上半场；真正难的是让推理后端高效执行这些不同 bit 的专家。

---

## 工程权衡与常见坑

先看一个趋势表：

| 策略 | 显存压力 | 精度风险 | 实现复杂度 | 适合场景 |
| --- | --- | --- | --- | --- |
| 全 FP16 | 最高 | 最低 | 低 | 机器资源充足 |
| 统一 4-bit | 中等 | 中等 | 低 | 先求能跑起来 |
| 专家级混合精度 | 更低 | 可控但依赖分配质量 | 高 | 显存紧、愿意做离线分析 |

### 常见坑 1：把所有专家统一压成 2-bit

这是最常见的误判。问题不只是单个矩阵误差变大，而是整条专家路径都可能变。MoE 的脆弱点不在“某个权重值不准”，而在“量化误差改变了专家输出分布，进一步改写 router 决策”。

### 常见坑 2：只看频率，不看敏感度

低频不等于不重要。有些专家像“边缘修正器”，平时不常被叫到，但负责特定语种、代码模式或长尾知识。一旦这些专家被压坏，退化不是均匀变差，而是某类输入突然掉得很厉害。

### 常见坑 3：只算权重，不算搬运

MoE 推理里，I/O 经常比算子本身更早成为瓶颈。尤其是 CPU/GPU 协同部署时，专家越碎、位宽越混，越容易出现频繁搬运和解码开销。QMoE 的价值就不只在压缩率，还在于它把压缩格式和 GPU 解码 kernel 一起设计。

### 常见坑 4：后端不支持异构 bit

理论上 bit 分得很漂亮，没有对应 kernel 也没用。如果推理框架不支持 expert-level mixed precision，最终可能退化成一堆小 kernel、频繁类型转换和差吞吐。部署前至少要确认三件事：

- 是否支持不同专家使用不同 bit
- 是否支持按 bit 分组执行 GroupGEMM
- 是否有 fused dequantize + matmul 路径

### 常见坑 5：校准集过窄

如果统计只来自很短、很单一的样本，那么“高频专家”可能只是这批样本的高频，而不是线上真实高频。代码、长上下文、多语种、工具调用，这些分布切换都可能让原来的 bit 分配失效。

---

## 替代方案与适用边界

专家级异构量化不是唯一答案，它适合“专家权重占主导，且允许做离线分析”的场景。如果边界条件变了，方案也要变。

| 方法 | 核心思路 | 是否强调全局分配 | 更适合什么场景 |
| --- | --- | --- | --- |
| QMoE | 极低 bit 压缩 + 自定义解码执行 | 否 | 超大 MoE、先解决“放不下” |
| MoQE | 只量化专家，利用专家的量化鲁棒性 | 弱 | 快速落地、先拿体积收益 |
| GEMQ | 用全局重要度和 LP 分配 bit，再做路由校正 | 是 | 追求更优精度-预算折中 |
| MxMoE | 同时考虑精度分配和 kernel 执行效率 | 是 | 非常关心实际吞吐 |
| MoPEQ | 用专家敏感度聚类，再给不同组分配 bit | 部分 | 拿不到细粒度统计时 |
| MoQAE | 把不同 bit 配置视为“专家”处理 KV cache | 不同问题 | 长上下文、KV cache 才是瓶颈 |

可以用一个简单判断逻辑：

1. 如果主要问题是“MoE 权重太大，根本放不进机器”，优先看 QMoE、MoQE 这类先解决存储问题的方法。
2. 如果可以做校准、允许少量微调，并且非常在意精度，优先看 GEMQ 一类全局分配方法。
3. 如果你的主要瓶颈已经不是专家权重，而是长上下文 KV cache，那么 MoQAE 更有针对性。
4. 如果推理后端根本不支持异构 kernel，那么统一 4-bit 往往比“理论更优、实际跑不快”的异构方案更现实。

所以它的适用边界可以概括成一句话：当专家权重是主要显存来源，系统允许离线统计并能执行 expert-level mixed precision 时，专家级异构量化最值得做；否则，统一位宽或更简单的 PTQ 往往更稳。

---

## 参考资料

1. QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models. arXiv 2023. https://arxiv.org/abs/2310.16795
2. QMoE 官方代码仓库. IST-DASLab. https://github.com/IST-DASLab/qmoe
3. Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness. Microsoft Research, 2023. https://www.microsoft.com/en-us/research/publication/mixture-of-quantized-experts-moqe-complementary-effect-of-low-bit-quantization-and-robustness/
4. Towards Global Expert-Level Mixed-Precision Quantization for Mixture-of-Experts LLMs (GEMQ). OpenReview ICLR 2026 投稿稿. https://openreview.net/forum?id=3f62aaf73f6601b2d45d141fb0664d64b01dd006
5. MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design. ICML 2025. https://proceedings.mlr.press/v267/duanmu25a.html
6. MoPEQ: Mixture of Mixed Precision Quantized Experts. ICCV Workshops 2025. https://openaccess.thecvf.com/content/ICCV2025W/BiVision/html/Chitty-Venkata_MoPEQ_Mixture_of_Mixed_Precision_Quantized_Experts_ICCVW_2025_paper.html
7. MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts. ACL 2025. https://aclanthology.org/2025.acl-long.531/
8. Mixtral-8x7B 模型卡. Mistral AI / Hugging Face. https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
9. Welcome Mixtral: a SOTA Mixture of Experts on Hugging Face. Hugging Face Blog. https://huggingface.co/blog/mixtral
