## 核心结论

Claude 的“特征引导”本质上是 activation steering，中文可理解为“在推理时直接推一把中间激活”，不改模型权重，只改模型当前这一步的内部状态。Anthropic 用 SAE 提取出的特征方向证明了一件事：如果你已经找到一个可解释的语义方向，比如 “Golden Gate Bridge（金门大桥）”，那么可以在残差流上加一个偏移量，让模型更稳定地朝这个概念说话。

核心公式只有一行：

$$
x' = x + \alpha f
$$

其中 $x$ 是原始残差流向量，残差流可以理解成“每一层不断往后传的工作记忆”；$f$ 是某个特征方向；$\alpha$ 是强度系数。$\alpha$ 越大，这个方向对后续生成的影响越强。

这比 prompt 更细。prompt 是在输入侧“劝”模型，steering 是在中间表示里“推”模型。前者靠语言指令，后者靠向量干预。Golden Gate Claude 之所以会在任何话题都扯到金门大桥，不是因为提示词写得奇怪，而是因为对应特征被强行放大。

但边界也很明确：它适合连续的语义偏向控制，不适合要求严格语法状态机的任务。让模型“更像在谈桥”，steering 很有效；让模型“必须输出合法 JSON”，steering 反而可能破坏结构。已有实验里，steering-only 的有效 JSON 率从 86.8% 降到 24.4%，而常规 fine-tune 可到 96.6%，约束解码可到 100%。

---

## 问题定义与边界

这项技术要解决的问题，不是“让模型学会新任务”，而是“在已有能力上改变偏向方向”。这两者差别很大。

如果一个模型已经知道什么是桥、什么是礼貌、什么是危险内容，那么你可以试着沿对应特征方向去放大或抑制它的倾向。这里的“特征”不是输入词，而是 SAE 从中间激活里拆出来的可解释因子。SAE，全称 Sparse Autoencoder，中文可理解为“稀疏自编码器”，作用是把一大团难解释的激活拆成很多更稀疏、较容易命名的方向。

边界可以直接看成下面这张表：

| 任务类型 | 目标 | 是否适合 steering | 原因 |
|---|---|---:|---|
| 话题偏向 | 多谈某个概念 | 适合 | 属于连续语义方向 |
| 风格调节 | 更正式、更简洁 | 较适合 | 也是语义或行为分布 |
| 安全干预 | 降低有害倾向 | 适合 | 可对应稳定特征 |
| JSON/XML 结构 | 必须语法合法 | 不适合单独使用 | 需要离散状态跟踪 |
| SQL/代码语法 | 必须可解析 | 不适合单独使用 | 需要括号、引号、缩进等约束 |

玩具例子很直观。假设你找到了“金门大桥”特征，把 $\alpha=5$ 加到该方向上，模型就会在自我介绍、旅游建议、哲学问题里都更容易提到金门大桥。这说明它抓住了“概念偏向”。

但如果你把同样的方法用于“合法 JSON”，问题就变了。JSON 合法性不是一个连续语义，而是一组离散规则：什么时候开引号、什么时候关括号、当前嵌套深度是多少。这更像状态机，不像一个单一语义方向。因此 steering 可能只会把输出推向“更像 JSON 的口气”，却不保证“真正可解析”。

---

## 核心机制与推导

先看 SAE 怎么把特征找出来。

给定某层残差流向量 $x \in \mathbb{R}^D$，SAE 先做编码，再做解码：

$$
z = \mathrm{ReLU}(W_e x + b_e)
$$

$$
\hat{x} = W_d z + b_d
$$

其中 $z$ 是稀疏特征激活，意思是“多数特征为 0，少数特征被点亮”。训练目标通常写成：

$$
\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1
$$

第一项要求“重建得像原激活”，第二项要求“点亮的特征尽量少”。这样得到的 decoder 列向量可以看成特征方向，某个方向如果总在“Golden Gate Bridge”相关文本上高激活，就能被解释为一个“金门大桥特征”。

推导到 steering 时就简单了。设某一层当前残差流是 $x$，对应特征方向为 $f$。如果不干预，后续层处理的是 $x$；如果要放大这个语义，就改成：

$$
x' = x + \alpha f
$$

从几何上看，就是在原向量上叠加一根箭头。若 $f$ 是单位向量，而原来在这个方向上的投影是 0.8，取 $\alpha=5$ 后，新投影变成 5.8。模型后续层会把这解释成“这个概念现在非常重要”。

可以用一个极小的二维玩具例子理解。设：

$$
x = (1, 0.2), \quad f = (0, 1), \quad \alpha = 5
$$

那么：

$$
x' = (1, 5.2)
$$

原来第二维只是弱信号，干预后第二维成为主导信号。如果第二维恰好对应“桥梁/金门大桥”语义，模型就会显著改变回答内容。

真实工程例子就是 Golden Gate Claude。Anthropic 展示过，把 Golden Gate Bridge 对应特征 clamp 到高于平常最大激活很多倍后，模型会持续把话题拉回那座桥，甚至在某些上下文里把自己说成那座桥。这说明特征不只是“相关”，而是对行为有因果影响。

---

## 代码实现

工程上，最常见做法不是改参数，而是在推理时给目标层加 hook。hook 可以理解成“在张量流过某一层时插一段自定义逻辑”。

下面这段 `python` 代码是一个可运行的最小示意，展示 `residual + α·f` 的位置：

```python
import numpy as np

def steer_residual(residual, feature, alpha):
    residual = np.asarray(residual, dtype=float)
    feature = np.asarray(feature, dtype=float)
    assert residual.shape == feature.shape
    return residual + alpha * feature

# 玩具例子：第二维表示“Golden Gate Bridge”方向
x = np.array([1.0, 0.2])
f = np.array([0.0, 1.0])
alpha = 5.0

x_prime = steer_residual(x, f, alpha)

assert np.allclose(x_prime, np.array([1.0, 5.2]))
assert x_prime[1] > x[1]
print(x_prime)
```

放到真实 Transformer 中，逻辑通常是这样：

```python
def residual_hook(residual, layer_id, target_layer, feature_vec, alpha):
    if layer_id != target_layer:
        return residual
    return residual + alpha * feature_vec

# 推理流程示意
for layer_id, layer in enumerate(model.layers):
    residual = layer(residual)
    residual = residual_hook(
        residual=residual,
        layer_id=layer_id,
        target_layer=12,
        feature_vec=golden_gate_feature,
        alpha=5.0,
    )
```

如果系统里已经有 SAE，还会多一步：先把当前激活编码成特征空间，用于定位和分析当前哪些特征在工作；但真正干预时，常见实现仍然是直接把 feature decoder 对应的方向向量加回残差流。

真实工程里要注意三件事：

| 实现点 | 正确做法 | 常见错误 |
|---|---|---|
| 干预位置 | 在特定层 residual stream 上加向量 | 加在 logits 上，效果往往粗暴且不稳定 |
| 强度控制 | 从小到大扫 $\alpha$ | 一上来用很大值，导致输出失真 |
| 观测指标 | 同时看语义效果和格式正确率 | 只看“像不像目标风格”，忽略结构损坏 |

如果要做线上实验，建议记录每层干预后的输出差异、目标词概率变化、格式错误率和拒答率。因为 steering 的问题通常不是“完全没效果”，而是“有了效果，但副作用同时被放大”。

---

## 工程权衡与常见坑

第一类坑是把 steering 当成“轻量 fine-tune”。这是误判。fine-tune 通过梯度更新权重，是真正在学任务；steering 只是把当前激活推向某个方向。它能放大已有表征，不能凭空创造稳定的语法能力。

第二类坑是强度过大。$\alpha$ 不是越大越好。当特征被过度放大时，模型会进入异常分布区，出现话题污染、人格漂移、幻觉增加，甚至整个回答都被单一概念劫持。Golden Gate Claude 的演示有价值，恰恰因为它把这个现象暴露得非常极端。

第三类坑是把“语义像”误当成“结构对”。下面这张表能说明问题：

| 方法 | 有效 JSON 率 | Micro F1 | 结论 |
|---|---:|---:|---|
| Base model | 86.8% | 7.7% | 有基础格式先验，但任务能力弱 |
| Steering only | 24.4% | 1.5% | 结构明显崩坏 |
| Fine-tuning | 96.6% | 91.2% | 任务学习有效 |
| Constrained decoding | 100% | 取决于底模内容能力 | 结构最稳 |

这说明结构化任务里最危险的误区是：模型“看起来更懂 JSON”，但其实括号、引号、逗号更容易错。原因是语义方向和语法状态不是同一类对象。前者是向量空间里的连续偏移，后者更像离散自动机。

真实工程例子可以想成一个信息抽取服务：输入病历文本，输出 JSON 给下游系统。如果你用 steering 去“加强 JSON 感”，模型也许会输出更像结构化文本的内容，但 24.4% 的合法率意味着大量请求会在解析器处直接失败。对线上系统来说，这不是“精度下降”，而是接口不可用。

---

## 替代方案与适用边界

如果目标是语义偏向控制，steering 仍然是很有价值的工具。它适合：

| 需求 | 推荐方法 |
|---|---|
| 调整人格、礼貌度、话题偏向 | Prompt + steering |
| 抑制或放大某类语义特征 | SAE steering |
| 学会新任务或固定输出格式 | Fine-tune |
| 必须保证 JSON/XML/SQL 合法 | Constrained decoding |
| 既要任务能力又要格式合法 | Fine-tune + constrained decoding |

这里的分工很重要。fine-tune 负责“学规则和任务映射”，constrained decoding 负责“保证输出不越界”，steering 负责“微调语义方向”。三者不是互斥关系，但也不能混着乱用。

对初学者，一个简单判断法是：

1. 你想改的是“模型更倾向说什么”，还是“模型必须按什么格式说”？
2. 如果是前者，优先考虑 prompt 或 steering。
3. 如果是后者，优先考虑 fine-tune 或约束解码。
4. 如果两者都要，先保证结构，再讨论语义偏向。

所以，Claude 的特征引导最准确的定位不是“万能控制器”，而是“语义层面的精细旋钮”。它比 prompt 更底层、更稳定，但也更危险，因为它直接作用于模型内部表示。一旦目标是连续语义，它非常强；一旦目标是离散语法，它往往不是正确工具。

---

## 参考资料

- Anthropic / Transformer Circuits, “Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet”  
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

- Hugging Face, Maziyar Panahi, “From Golden Gate Bridge to Broken JSON: Why Anthropic's SAE Steering Fails for Structured Output”  
  https://huggingface.co/blog/maziyarpanahi/sae-steering-json

- Jonny Davis, “Understanding Anthropic’s Golden Gate Claude”  
  https://medium.com/%40jonnyndavis/understanding-anthropics-golden-gate-claude-150f9653bf75
