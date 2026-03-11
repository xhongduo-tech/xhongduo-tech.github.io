## 核心结论

FFN 第二层的压缩投影，通常记作 `W₂`，是把高维中间激活重新压回模型主维度 `d_model` 的线性层。线性层就是“做一次矩阵乘法的变换器”。在 SwiGLU/GeGLU 结构里，前面先把特征扩到约 `4×d_model`，再由 `W₂` 压回去，因此 `W₂` 是整个 FFN 残差分支的唯一出口，也是信息瓶颈。

FFN 的常用写法是：
$$
\mathrm{FFN}(x)=W_2\big(\mathrm{SiLU}(xW_{\text{gate}})\odot xW_{\text{value}}\big)
$$
其中 `SiLU` 是一种平滑门控激活，可以理解成“连续版开关”；`\odot` 是逐元素乘法，可以理解成“每个通道自己和自己配对相乘”。

对初学者，一个玩具例子足够说明问题：假设 `d_model=2`，先扩到 `d_ff=8`。前两层把输入拆成 8 条中间通道，但门控后真正有信号的可能只剩 4 条左右，最后只能靠 `W₂` 把这 4 条有效通道重新组合成 2 维输出。也就是说，真正决定“哪些有效信息被送回残差流”的，是 `W₂`。

| 投影位置 | 作用 | 调优重要性 |
|---|---|---|
| Query/Key | 计算注意力相似度 | 低 |
| Value | 搬运注意力内容 | 中 |
| Up/Gate | 扩维并做门控 | 中 |
| Output/Down | 写回主干表示 | 高 |

S2FT 的实验结论可概括为：`Query/Key << Value/Up/Gate < Output/Down`。对很多 LLaMA 类模型，在参数预算固定时，只调 Down projection 往往已经非常接近“多处一起调”。

---

## 问题定义与边界

这里先统一记号。若把输入写成行向量右乘矩阵，则 `W₂ ∈ ℝ^{d_ff×d_model}`；若按 PyTorch `nn.Linear(d_ff, d_model)` 的权重存储，则张量形状是 `[d_model, d_ff]`。两种写法只是约定不同，表示的是同一个映射：把 `d_ff` 维向量映射回 `d_model` 维。

边界也要说清楚。本文讨论的是密集 Transformer 中 FFN 的下投影，不讨论 MoE 中专家路由，也不讨论把 FFN 整体替换成稀疏专家后的额外门控问题。

一个常见误解是：“既然 Up 层先把维度拉大，为什么不优先调 Up？”原因是 Up 负责制造候选特征，Down 负责把候选特征写回主干。写回主干就是“把中间结果真正送回模型状态”。如果写回这一步不对，前面扩得再好也回不到残差流。

以 `d_model=4096, d_ff=16384` 为例，若门控后约一半通道有效，则 `W₂` 实际每个 token 看到的有效输入大约是 8192 个通道，而不是完整的 16384 个。

| `d_model` | `d_ff` | 门控后有效比例 | `W₂` 接收的有效通道数 |
|---|---:|---:|---:|
| 4096 | 16384 | 100% | 16384 |
| 4096 | 16384 | 50% | 8192 |
| 4096 | 16384 | 25% | 4096 |

---

## 核心机制与推导

SwiGLU 的门控可以拆成四步看：

| 步骤 | 公式 | 维度变化 | 直观含义 |
|---|---|---|---|
| 1 | `g=xW_gate` | `d_model→d_ff` | 生成门信号 |
| 2 | `v=xW_value` | `d_model→d_ff` | 生成值信号 |
| 3 | `h=SiLU(g)⊙v` | `d_ff→d_ff` | 只有被门打开的值能通过 |
| 4 | `y=hW₂` | `d_ff→d_model` | 压回主干维度 |

这里的“门控”就是用一个通道去控制另一个通道是否被放大、缩小或接近抑制。虽然 SiLU 不像 ReLU 那样严格把负半轴全部截断成 0，但它会显著压低负区输入，因此工程上常把它理解为“只有一部分通道真正导通”。这就是为什么说 `W₂` 的梯度往往更集中在少数有效通道上。

从方差看，若每层残差都往主干里加一个 FFN 输出，而 `W₂` 初始化过大，层数一深，残差方差就会累积。一个常见做法是对残差分支输出投影做深度缩放：
$$
W_2 \sim \mathcal N\left(0,\ \frac{1}{\sqrt{2L}}\cdot \sigma_0\right)
$$
若把基础标准差写成与输入宽度相关的形式，也常见成：
$$
\mathrm{Var}(W_2)\approx \frac{2}{d_{ff}\cdot L}
$$
其中 `L` 是层数。直观上，这表示模型越深，残差出口的初始扰动越要小，否则主干表示会被每层不断放大。

---

## 代码实现

下面给一个可运行的最小 Python 例子，展示 `W₂` 如何接收门控后的稀疏激活，以及如何做深度缩放初始化。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_layers: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

        # 常见工程写法：残差出口按层数缩放
        std = math.sqrt(2.0 / (d_ff * n_layers))
        nn.init.normal_(self.w_down.weight, mean=0.0, std=std)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        value = self.w_value(x)
        hidden = gate * value
        out = self.w_down(hidden)
        return out, hidden

torch.manual_seed(0)

d_model, d_ff, L = 4, 16, 8
x = torch.randn(2, d_model)
ffn = SwiGLUFFN(d_model, d_ff, L)

y, h = ffn(x)

assert y.shape == (2, d_model)
assert h.shape == (2, d_ff)
assert ffn.w_down.weight.shape == (d_model, d_ff)

# 统计“接近零”的门控通道比例，只是玩具观测，不是理论常数
near_zero_ratio = (h.abs() < 1e-3).float().mean().item()
assert 0.0 <= near_zero_ratio <= 1.0
print("hidden near-zero ratio:", round(near_zero_ratio, 4))
```

真实工程例子是 LLaMA 类模型上的 LoRA 微调。若你只给 `W₁` 或 `gate/value` 挂 LoRA，模型能学到“生成哪些候选特征”；若给 `W₂` 挂 LoRA，模型学到的是“哪些候选特征真正写回主干”。对分类、推理、指令跟随这类任务，后者通常更直接影响最终行为。

---

## 工程权衡与常见坑

`W₂` 重要，不等于“只调它永远最好”。它的优势来自信息瓶颈，但风险也来自信息瓶颈。

| 常见坑 | 问题本质 | 规避策略 |
|---|---|---|
| 忽略门控稀疏性 | 更新落在低效通道 | 用激活统计选通道 |
| `W₂` 初始化过大 | 残差方差层层累积 | 做 `√L` 级别缩放 |
| LoRA 随机压缩 | 低秩子空间没对齐有效特征 | 用数据感知投影，如 IPA |
| rank 太低且位置错 | 参数省了，但更新打不到关键出口 | 优先给 `W₂`，再考虑 Output |

要特别注意一个事实：LoRA 里下投影矩阵 `A` 是“先压缩输入”的低秩入口。若这个入口是随机的，而且训练后变化又很小，那么它就像一个固定的随机漏斗，很多有效信息在进入可训练分支前已经丢掉了。IPA 这类方法的价值就在于把这个随机漏斗替换成“尽量保信息”的漏斗。

---

## 替代方案与适用边界

如果参数预算非常紧，优先级通常是：先调 `Down`，再考虑 `Output`，最后才是 `Up/Gate/Value`。这不是因为前者“更高级”，而是因为它们更接近主干表示的写回点。

| 方案 | 适用场景 | 局限 |
|---|---|---|
| Down-only | 极低参数预算、先求有效 | 表达能力不如多点联合 |
| Down + Output | 兼顾 FFN 与注意力写回 | 参数更多 |
| 全投影 LoRA | 追求上限 | 成本更高，未必更稳 |
| IPA + Down LoRA | 低秩入口容易丢信息时 | 需要额外预处理或校准数据 |

对新手可以这样理解：只更新 Down，像是在“改总闸后的最后一级配电”；更新所有投影，则像从发电、分路、总闸到回写全部都改。预算有限时，先改最后一级通常最划算。

但它也有边界。如果任务主要依赖新特征生成，而不是旧特征重组，例如需要显著改变中间表征分布，那么只调 `W₂` 可能不够，此时 `Up/Gate` 的价值会上升。

---

## 参考资料

1. NeurIPS 2024《S2FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity》：给出 FFN 各投影的调优效果排序，结论是 `Query/Key << Value/Up/Gate < Output/Down`，并在部分模型上只调 Down。
2. Radford et al.《Language Models are Unsupervised Multitask Learners》：给出残差分支按深度缩放初始化的经典做法，即残差层权重按 `1/√N` 缩放。
3. IPA《An Information-Preserving Input Projection Framework for Efficient Foundation Model Adaptation》：指出 LoRA 的下投影入口常接近随机压缩，训练后变化有限，因此建议用保信息投影替代。
4. NVIDIA Megatron Core 的 SwiGLU 文档：明确 SwiGLU 计算形式是 `SiLU(y1) * y2`，对应门控乘法结构。
5. TinyTorch / MLSys 相关激活函数材料：说明门控激活会让有效通道分布变得不均匀，从而使后续 `W₂` 的梯度更集中在少数通道。
