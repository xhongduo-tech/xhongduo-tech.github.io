## 核心结论

注意力头专化，指的是同一层里并行的多个 attention head 在训练后会出现稳定分工：有些头更偏向局部位置关系，有些头更容易对齐句法依赖，有些头更像在聚合语义线索或句子级信息。更准确地说，这不是“每个头只做一件事”，而是“某些头对某类模式表现出显著偏好”。

这件事重要，主要有两层原因。

第一，它给 Transformer 的可解释性提供了一个可操作的观察单位。整层 attention 很难直接解释，但“第几层第几个头长期在关注什么”是可以统计、可视化、可干预的。如果某个头在大量样本中都稳定连接“动词 -> 直接宾语”“代词 -> 先行词”“当前词 -> 前一个词”，那么我们至少可以说，这个头对某类依赖关系存在功能倾向。

第二，它给模型压缩和部署提供了结构化入口。大量研究表明，不同头的重要性差异很大，少数关键头承担了主要有效计算，而很多头被剪掉后，模型指标只会小幅下降。这意味着 head pruning 不是拍脑袋删结构，而是有经验规律和实验支持的。

先看一个最小例子。句子“猫追老鼠”里，如果某个头在“追”这个位置上总是高权重指向“老鼠”，它就可能在编码“动词与宾语”的关系；如果另一个头让每个词都主要关注前一个词，它更接近一个位置偏移头。前者更可能影响语义理解，后者更多提供局部顺序信息。两者都可能有用，但重要性通常不同。

真实研究里的结论更明确。Voita 等人在机器翻译 Transformer 编码器上发现，原始 48 个 encoder heads 中，剪掉 38 个后 BLEU 只下降约 0.15，说明大量头存在冗余。Clark 等在分析 BERT 时发现，某些特定头与句法关系高度对齐，例如 BERT 的第 8 层第 10 个头对直接宾语依赖 `dobj` 表现出很高的匹配率。这些结果共同说明：头不是均匀工作的，专化现象既能被观察到，也能被利用起来。

| 场景 | 结论 | 代表数值 |
| --- | --- | --- |
| BERT 句法头分析 | 少数头对特定依赖高度专化 | layer 8 head 10 对 `dobj` 约 86.8% |
| WMT 翻译头剪枝 | 大量头可删而性能基本保持 | 剪 38/48 个头，BLEU 约降 0.15 |
| 边缘部署/参数高效微调 | 先保留关键头可明显降算力 | 在保留主干能力前提下显著减少注意力计算 |

对新手来说，可以先把 multi-head attention 理解成“同一层里并行跑多个检索器”。每个头都从输入里挑选自己最有用的信息，但因为参数不同、训练信号不同，最后它们不会完全学成一样，于是出现分工。这种分工并不总是绝对清晰，但足够稳定，足以支持解释和剪枝。

---

## 问题定义与边界

本文讨论的是 Transformer 中 multi-head attention 的“头级别”现象，而不是整层替换、模型蒸馏、低秩分解或全结构重写。边界可以压缩成两个问题：

第一，头是否真的学到了不同模式，而不只是随机波动。  
第二，怎样识别冗余头，并在不明显伤害任务性能的前提下把它们剪掉。

这里的“模式”主要包括四类：

| 模式类型 | 典型含义 | 常见例子 |
| --- | --- | --- |
| 句法模式 | 词与词之间的结构关系 | 动词关注宾语，代词关注先行词 |
| 语义模式 | 内容相关、主题相关、实体相关 | “医生”关注“医院”，“苹果”关注“水果” |
| 位置模式 | 固定偏移或边界标记 | 总看前一个词、下一个词、句首、分隔符 |
| 全局聚合模式 | 收集句子级摘要信息 | `[CLS]` 聚合整句信息，标点或句界汇聚 |

“头是否重要”也不能只看注意力热图。热图只是“看向哪里”，不是“这个头对最终损失有多大贡献”。工程上更可靠的指标是：如果把这个头关掉，损失会增加多少。为了可微地做这件事，常见做法是在每个头前面放一个门控变量 $\xi_h$，再看损失对它的敏感度。简化写法是：

$$
I_h=\mathbb{E}_{(x,y)\sim D}\left|\frac{\partial \mathcal{L}(x,y)}{\partial \xi_h}\right|
$$

其中：

- $\mathcal{L}(x,y)$ 是样本 $(x,y)$ 上的任务损失
- $\xi_h$ 是第 $h$ 个头的门控变量
- $I_h$ 越大，表示损失对这个头越敏感
- 从一阶近似看，$I_h$ 大的头更不该被删

这个公式的直观含义是：如果轻微调小一个头的输出，模型损失立刻明显变差，那么这个头大概率承担了关键计算；反过来，如果损失几乎不变，这个头更可能冗余。

这里还有一个经常被误解的点。一个头“看起来像句法头”，不代表它“只做句法”。注意力输出最终进入残差连接、LayerNorm 和后续 MLP，它参与的是整体表示更新，而不是单独输出一个“句法标签”。因此，可解释性在这里应理解为“功能倾向可归纳”，而不是“这个头的唯一因果作用已被完全证明”。

换句话说，解释性分析的目标不是把头神化成一个个独立模块，而是回答一个更实际的问题：哪些头长期表现出稳定模式，这些模式对任务是否重要，以及这种稳定性是否足以支持压缩或诊断。

---

## 核心机制与推导

单个 attention head 的标准形式是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

其中：

- $Q$ 是 query，表示“当前位置想找什么”
- $K$ 是 key，表示“每个位置能被怎样匹配”
- $V$ 是 value，表示“匹配成功后实际取回什么内容”
- $d_h$ 是单头维度，用来缩放点积，避免数值过大

如果写成第 $i$ 个 token 对第 $j$ 个 token 的注意力权重，可以记为：

$$
\alpha_{ij}^{(h)}=
\frac{\exp\left(q_i^{(h)}\cdot k_j^{(h)}/\sqrt{d_h}\right)}
{\sum_{t}\exp\left(q_i^{(h)}\cdot k_t^{(h)}/\sqrt{d_h}\right)}
$$

于是第 $h$ 个头在位置 $i$ 的输出是：

$$
z_i^{(h)}=\sum_j \alpha_{ij}^{(h)}v_j^{(h)}
$$

整层多头拼接后再线性映射：

$$
\mathrm{MHA}(X)=\mathrm{Concat}(Z^{(1)},Z^{(2)},\dots,Z^{(H)})W^O
$$

为什么会出现专化？核心原因不是“模型被硬编码要求分工”，而是下面三个条件共同作用：

1. 多个头并行存在，每个头有独立的 $W_Q,W_K,W_V$ 参数。
2. 所有头共享同一个训练目标，都会收到反向传播信号。
3. 当某个模式已经被若干头学会后，继续复制同样模式的边际收益会下降。

这会带来一种近似的“功能竞争”。如果一个头已经很好地处理了固定位置偏移，那么另一个头更容易通过学习实体对齐、句法边或全局汇聚来继续降低损失。最终结果不是严格分工，而是统计上的偏好分化。

解释性分析通常有两条主线。

第一条线是可视化和对齐分析。  
做法是把注意力图画出来，再与句法依存树、共指标注、分隔符位置、特殊 token 等结构进行对齐统计。例如：

- 如果一个头经常从动词位置高权重指向其宾语，它可能偏句法
- 如果一个头总是从代词指向先行词，它可能偏共指
- 如果一个头总盯着前一个词，它可能是相对位置头
- 如果一个头频繁把信息汇聚到 `[CLS]`，它可能参与句级摘要

第二条线是可干预分析。  
相比“看图”，更重要的是“动手关掉看看会怎样”。常见做法是给每个头加门控 $g_h$，再联合优化任务损失与稀疏正则：

$$
o=\sum_{h=1}^{H} g_h z_h
$$

$$
\mathcal{L}_{total}=\mathcal{L}_{task}+\lambda\sum_h g_h
$$

上式是最简写法。实际论文里常见的是 Hard Concrete 或其他近似 $L_0$ 正则化门控，而不是直接对 $g_h$ 线性惩罚。原因很简单：目标是把某些头真正推向“开/关”两种状态，而不是只得到一堆模糊的连续权重。

从一阶泰勒展开可以得到一个重要近似。设原始输出为 $o=\sum_h g_h z_h$，如果把某个门控从当前值微调到 0，损失变化可写成：

$$
\Delta \mathcal{L}
\approx
\frac{\partial \mathcal{L}}{\partial g_h}\Delta g_h
$$

若直接考虑“关掉该头”，则 $\Delta g_h\approx -g_h$，因此：

$$
|\Delta \mathcal{L}|
\approx
\left|\frac{\partial \mathcal{L}}{\partial g_h}g_h\right|
$$

这就是很多“梯度重要性”指标的来源。它不是精确真值，但在工程上足够有用：可以快速筛出一批风险高的头和一批可疑冗余头，再用消融实验复核。

不同类型头的经验性行为通常如下：

| 头类型 | 典型行为 | 常见位置 | 对新手的直观理解 |
| --- | --- | --- | --- |
| 句法头 | 动词关注宾语、修饰词关注中心词、代词关注先行词 | 中层较常见 | 像在追踪“谁和谁有结构关系” |
| 语义头 | 聚合同主题词、同实体词、上下文相关词 | 中高层较常见 | 像在收集“哪些内容在讲同一件事” |
| 位置头 | 固定看前一个词、后一个词、分隔符 | 浅层较常见 | 像在补充局部顺序和边界信息 |
| 全局/汇聚头 | 高频关注 `[CLS]`、句界、标点 | 各层都可能出现 | 像在做局部到全局的信息汇总 |

需要强调的一点是：这些类别不是互斥标签。同一个头可能同时带有位置偏好和语义偏好，也可能在不同数据分布上表现不同。所以“专化”最好理解成概率性的功能偏向，而不是逻辑上的唯一归属。

---

## 代码实现

下面给一个可直接运行的 Python 玩具实现。它不依赖深度学习框架，但完整演示四件事：

1. 用 `sigmoid` 把 gate logit 映射成连续门控  
2. 计算头输出的加权组合  
3. 按阈值做近似剪枝  
4. 在剪枝后做重标定，并用一阶梯度近似统计重要性

```python
import math
from dataclasses import dataclass
from typing import List


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Head:
    name: str
    output: float
    gate_logit: float
    grad_wrt_gate: float


def compute_gates(heads: List[Head]) -> List[float]:
    return [sigmoid(h.gate_logit) for h in heads]


def combine_outputs(heads: List[Head], gates: List[float]) -> float:
    return sum(g * h.output for g, h in zip(gates, heads))


def prune_by_threshold(gates: List[float], threshold: float = 0.5) -> List[int]:
    return [1 if g > threshold else 0 for g in gates]


def masked_sum(heads: List[Head], mask: List[int]) -> float:
    return sum(m * h.output for m, h in zip(mask, heads))


def rescale_after_pruning(value: float, total_heads: int, kept_heads: int) -> float:
    if kept_heads == 0:
        raise ValueError("All heads were pruned; cannot rescale.")
    return value * (total_heads / kept_heads)


def importance_scores(heads: List[Head]) -> List[float]:
    return [abs(h.grad_wrt_gate) for h in heads]


def main() -> None:
    # 三个玩具头：
    # 1) 位置头：贡献较小
    # 2) 句法头：贡献最大
    # 3) 冗余头：几乎无贡献
    heads = [
        Head(name="positional_head", output=0.20, gate_logit=-1.5, grad_wrt_gate=0.08),
        Head(name="syntactic_head", output=1.00, gate_logit=2.0, grad_wrt_gate=1.20),
        Head(name="redundant_head", output=0.05, gate_logit=-3.0, grad_wrt_gate=0.01),
    ]

    gates = compute_gates(heads)
    combined = combine_outputs(heads, gates)

    mask = prune_by_threshold(gates, threshold=0.5)
    kept = sum(mask)

    pruned = masked_sum(heads, mask)
    rescaled = rescale_after_pruning(pruned, total_heads=len(heads), kept_heads=kept)

    importance = importance_scores(heads)
    most_important_idx = max(range(len(importance)), key=lambda i: importance[i])

    # 可运行断言
    assert mask == [0, 1, 0]
    assert heads[most_important_idx].name == "syntactic_head"
    assert abs(pruned - 1.0) < 1e-9
    assert abs(rescaled - 3.0) < 1e-9

    print("Per-head gates:")
    for head, gate in zip(heads, gates):
        print(f"  {head.name:16s} gate={gate:.6f}")

    print("\nThreshold mask:", mask)
    print("Combined output before hard pruning:", round(combined, 6))
    print("Output after hard pruning:", round(pruned, 6))
    print("Output after rescaling:", round(rescaled, 6))

    print("\nImportance scores (|dL/dg_h|):")
    for head, score in zip(heads, importance):
        print(f"  {head.name:16s} importance={score:.4f}")

    print(f"\nMost important head: {heads[most_important_idx].name}")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出会类似：

```text
Per-head gates:
  positional_head  gate=0.182426
  syntactic_head   gate=0.880797
  redundant_head   gate=0.047426

Threshold mask: [0, 1, 0]
Combined output before hard pruning: 0.919463
Output after hard pruning: 1.0
Output after rescaling: 3.0

Importance scores (|dL/dg_h|):
  positional_head  importance=0.0800
  syntactic_head   importance=1.2000
  redundant_head   importance=0.0100

Most important head: syntactic_head
```

这个玩具例子表达的是一个非常具体的工程事实：

- 第二个头的重要性最高，因此保留
- 第一个和第三个头被阈值裁掉
- 裁掉后如果完全不校准，输出幅度会变小
- 用 $\frac{H}{H'}$ 做简单 rescale，可以粗略补偿“并联分支数量变少”的影响

这里的重标定公式是：

$$
o_{\text{rescaled}} = o_{\text{pruned}} \cdot \frac{H}{H'}
$$

其中：

- $H$ 是原始头数
- $H'$ 是保留头数
- $o_{\text{pruned}}$ 是剪枝后的头输出和

这当然不是唯一做法。真实模型里更常见的策略有：

| 做法 | 核心思路 | 适用情况 |
| --- | --- | --- |
| 简单 rescale | 按头数比值补偿幅度 | 快速实验、静态剪枝 |
| 层级校准 | 针对每层输出重新标定 | 剪枝比例较高时更稳 |
| 短暂微调 | 用少量数据恢复分布 | 工程上最常见 |
| 蒸馏恢复 | 让剪枝模型对齐原模型输出 | 对精度敏感的场景 |

如果把这个思路放到真实 PyTorch 训练循环里，流程通常是：

1. 在每个 head 输出后乘门控 $g_h$
2. 前向计算任务损失
3. 反向传播并累积 $\left|\partial \mathcal{L}/\partial g_h\right|$
4. 加入稀疏正则，让低价值头逐渐接近关闭
5. 到达阈值后固化 head mask
6. 剪枝后做 layer-wise 校准或短暂微调
7. 导出新的静态结构用于部署

对新手来说，最重要的是理解“为什么不能只看热图”。热图回答的是“头看向哪里”，梯度或消融回答的是“这个头删掉会不会出事”。前者适合解释，后者决定能不能安全剪枝。两者必须配合。

再给一个直观例子。假设两个头都喜欢关注逗号：

- 头 A 只是机械地把注意力打到逗号上，删掉几乎没影响
- 头 B 借助逗号定位从句边界，再把句法信息传给后续层，删掉会掉点

从热图上看，它们都“像分隔符头”；从重要性上看，它们可能完全不同。这正是解释性分析和压缩决策不能混为一谈的原因。

---

## 工程权衡与常见坑

最常见的坑不是“删多了”，而是“删得太机械”。头剪枝删掉的是残差支路中的并联信息源。如果直接把若干 head 置零，却不做重标定、校准或轻微微调，后续层看到的表示分布就会偏离训练时状态，表现为分类分数波动、生成稳定性下降、长文本行为变差。

这个问题可以用一个更工程化的方式理解。Transformer 一层的输出通常是：

$$
x_{l+1} = \mathrm{LayerNorm}\left(x_l + \mathrm{MHA}(x_l)\right)
$$

如果 `MHA(x_l)` 的幅度因为剪枝突然显著减小，那么残差和 attention 分支的相对比例就变了。LayerNorm 虽然会做归一化，但并不能完全抵消“有用方向被删掉”的问题，也不能保证各层统计特性自动恢复。

| 剪枝阶段 | 需注意事项 | 典型风险 | 更稳妥的做法 |
| --- | --- | --- | --- |
| 剪枝前 | 分层统计重要性，不要只看总平均 | 浅层关键位置头被误删 | 先做分层消融和验证集评估 |
| 剪枝中 | 逐步稀疏，不要一步清空 | 训练震荡、损失突然升高 | 使用门控正则逐渐逼近稀疏 |
| 剪枝后 | 做 rescale、校准或短暂微调 | 残差尺度漂移 | 用少量校准数据恢复表示分布 |
| 部署前 | 固化 head mask 并测真实延迟 | 训练有效、上线无收益 | 验证硬件端是否真减少计算 |

第二个常见坑是“只看注意力热图”。热图容易看、也容易讲故事，但它不是充分证据。一个头的图案很稳定，不代表它对任务有决定性贡献；一个头的图案不直观，也不代表它没用。更稳的流程应该至少包含三类证据：

| 证据类型 | 回答的问题 | 局限 |
| --- | --- | --- |
| 热图/对齐统计 | 这个头常看哪里 | 不能直接说明重要性 |
| 梯度重要性 | 损失对这个头是否敏感 | 只是一阶近似 |
| 消融实验 | 真删掉后指标会怎样 | 成本较高，但最可信 |

第三个坑是“按统一比例全层剪”。这在工程上很诱人，因为配置简单，但通常不够稳。经验上：

- 浅层更偏词法和位置，往往更脆弱
- 中层常出现较清晰的句法或语义专化
- 高层是否冗余更依赖任务类型

对分类、检索这类任务，高层一部分头可能确实可删；对开放式生成，某些看似平时不活跃的头在少数上下文里可能承担兜底作用。统一按“每层都删 50%”很容易把真正关键的稀有头删掉。

第四个坑是“只看离线指标，不看线上收益”。Head pruning 的目标往往是推理加速，但能不能真正变快，取决于具体实现。若只是逻辑上屏蔽了 head，而底层 kernel 仍按原始张量形状计算，延迟未必明显下降。也就是说：

- “模型参数看起来变少了”不等于“GPU 上一定更快”
- “FLOPs 理论下降”不等于“线上 QPS 一定提高”
- 真正部署时，需要把剪枝后的结构静态化，避免空算

第五个坑是“把专化当成硬规则”。例如，看到一篇论文说“某层某头常常是句法头”，就照搬到别的模型、别的语言、别的任务上，这是不稳的。专化现象是统计规律，不是固定模板。不同模型规模、分词方式、预训练语料和任务目标都会改变头的功能分布。

对新手最实用的经验是：先把 head pruning 当成一个“验证驱动”的工程优化问题，而不是一个“看到热图就能下结论”的解释性故事。先测重要性，再做小比例剪枝，再校准，再看真实部署收益，这个顺序通常比直接大刀阔斧删头更靠谱。

---

## 替代方案与适用边界

头剪枝不是唯一选择。它的优势是结构明确、部署路径清晰，因为改动直接发生在 attention head 维度，容易转换成静态结构；它的缺点是表达能力会被硬性删掉，一旦判断错了，恢复成本比纯参数高效微调更高。

把它放到压缩方法家族里，可以更清楚地看优缺点：

| 方案 | 核心思路 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- | --- |
| Head Pruning | 删掉冗余头 | 直接降注意力计算，结构清晰 | 误删关键头会伤能力 | 边缘部署、推理加速 |
| LoRA/PEFT | 增加小规模可训练增量参数 | 微调成本低，主干少改动 | 推理主干通常不变 | 多任务适配、低成本微调 |
| Structured Sparsity | 对层、通道、块做结构化稀疏 | 有机会得到更大硬件收益 | 搜索与实现复杂 | 大规模压缩 |
| Quantization | 降低权重和激活位宽 | 通常收益直接、成熟 | 可能引入数值误差 | 推理加速、内存压缩 |
| Head Pruning + PEFT | 先删冗余头，再轻量适配 | 压缩与任务适配兼得 | 调参复杂，验证成本高 | 资源受限场景 |

什么时候 head pruning 更合适？通常有三个条件：

1. 你已经知道模型存在明显冗余  
2. 目标是减少推理成本，而不是单纯减少可训练参数  
3. 部署链路允许把剪枝后的结构真正固化下来  

什么时候不该激进使用？通常也是三个条件：

1. 任务依赖长距离生成和细粒度风格控制  
2. 模型要处理很长尾、很少见的上下文  
3. 你没有足够校准数据验证剪枝副作用  

适用边界也要讲清楚。对分类、检索、抽取、句法分析这类任务，头专化现象通常更容易被观测和复现，因为有效依赖关系相对集中、评价指标也更稳定。对开放式长文本生成，情况更复杂。某些头在多数样本上看起来贡献很低，但在少数关键上下文里可能负责消歧、维持叙事一致性、追踪远距离约束。如果一次性激进剪枝，离线平均指标可能还行，线上长尾质量却会明显受损。

因此，更稳妥的策略往往不是“尽可能多删”，而是：

| 场景 | 更稳妥的策略 |
| --- | --- |
| 分类/检索/抽取 | 可较积极做分层剪枝，并配合短暂微调 |
| 机器翻译/摘要 | 中等强度剪枝，重点关注中层和高层稳定性 |
| 开放式生成 | 温和剪枝，加校准微调，重点盯长尾样本 |
| 端侧部署 | 先验证结构固化后是否真减少延迟，再决定剪枝比例 |

如果目标是“少花显存、少花算力，但又尽量不碰主干能力”，实践上常见的路线不是单独使用某一种方法，而是组合：

1. 先做量化，拿一轮最稳的收益
2. 再做温和 head pruning，去掉明显冗余
3. 最后用 PEFT 做轻量恢复或任务适配

这样通常比单独在 head pruning 上追求极限压缩更稳，也更符合真实部署的风险控制方式。

---

## 参考资料

下面列出的资料分别对应“现象观测”“可微剪枝方法”“跨模型功能专化趋势”三条主线。

| 来源 | 贡献 | 关联结论 |
| --- | --- | --- |
| Clark et al., *What Does BERT Look At?* | 系统分析 BERT 不同头与句法、共指、分隔符等模式的对应关系 | 说明头级别现象具有可观察的功能倾向 |
| Voita et al., *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned* | 用门控与稀疏化方法分析哪些头关键、哪些头可删 | 说明少数关键头承担主要贡献，冗余头可结构化剪枝 |
| Caucheteux, King, *Shared functional specialization in transformer-based language models and the human brain* | 从更宏观角度研究 Transformer 表征专化与脑语言网络的对应趋势 | 支撑“功能分化是稳定现象，不只是偶然噪声” |

为了方便初学者建立阅读顺序，可以按下面方式看：

| 阅读顺序 | 建议先看什么 | 目的 |
| --- | --- | --- |
| 第 1 篇 | Clark et al. | 先建立“头会出现可解释模式”的直觉 |
| 第 2 篇 | Voita et al. | 再理解“专化不仅可看，还可用于剪枝” |
| 第 3 篇 | Caucheteux & King | 最后再看更大尺度的功能专化讨论 |

参考文献：

- Clark, K., Khandelwal, U., Levy, O., Manning, C. D. *What Does BERT Look At?* Proceedings of the 2019 ACL Workshop BlackboxNLP. https://aclanthology.org/W19-4828/
- Voita, E., Talbot, D., Moiseev, F., Sennrich, R., Titov, I. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL 2019. https://aclanthology.org/P19-1580/
- Caucheteux, C., King, J.-R. *Shared functional specialization in transformer-based language models and the human brain.* Nature Communications, 2024. https://www.nature.com/articles/s41467-024-49173-5

如果只保留一句总结这些参考资料的共同结论，那就是：注意力头既不是完全同质的，也不是每个都同样关键。它们会表现出可归纳的功能偏好，而这种偏好既能帮助解释模型行为，也能为结构化压缩提供依据。
