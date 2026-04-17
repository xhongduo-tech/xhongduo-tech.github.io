## 核心结论

机械可解释性研究的不是“模型最后答对了什么”，而是“模型内部到底做了哪些可追踪计算”。这里的“可追踪”可以先理解成：你能指出某一段内部表示代表什么信息，以及它通过哪些模块影响了最终输出。

对 transformer 而言，最常见的拆解对象是两类东西：

1. `feature`：特征。白话说，就是残差流里某个“有语义的方向”，例如“当前 token 像是在复制前文”“这里出现了左括号未闭合”。
2. `circuit`：电路。白话说，就是一组 attention head、MLP 和残差流路径拼起来，完成某个具体功能的计算链。

核心判断标准不是“这个位置看起来相关”，而是“把这条路径改掉，输出是否按预期变化”。因此，机械可解释性的标准流程通常是：

输入 → 找到 feature 方向 → 追踪 circuit 路径 → 做 patching 验证因果 → 判断这段机制是否真的实现了该功能。

可以把它看成逆向工程。不是只观察灯泡亮不亮，而是把电路板拆开，找出哪条线路在通电、哪几个元件协同工作、断掉其中一段后会发生什么。

一个新手版高层示意如下：

```text
输入 token
   ↓
残差流状态 r
   ↓  投影到 feature 方向 d_f
feature 激活强度
   ↓
经过 attention head / MLP 组成的 circuit
   ↓
logits 变化
   ↓
输出 token
```

这套方法的价值在于，它试图回答“模型如何推理”，而不只是“哪些输入和输出相关”。

---

## 问题定义与边界

机械可解释性的正式目标，是把神经网络中原本难以理解的内部计算，还原成接近“算法步骤”的结构化解释。这里的重点不是给整个模型贴一个自然语言标签，而是识别出具体功能由哪些内部部件实现。

边界必须说清楚。

首先，它解释的通常是“局部行为”，不是“整个模型的全部智能”。例如，研究者发现一个 `induction head` 电路。`induction head` 可以先理解成：模型在上下文中发现重复模式后，学会沿着前文位置去复制后续 token 的注意力头。这能解释一部分 in-context learning，但不能说明模型全部写作、推理、规划能力都来自这条电路。

其次，解释对象往往不是单个 neuron。原因是 `polysemanticity` 很常见。这个词可以先理解成：一个神经元可能混合编码多个互不相关的含义。所以机械可解释性更常研究“方向”或“子空间”，而不是执着于“第 1234 个 neuron 就代表逗号”。

下表说明了“解释到了哪里”和“还没解释什么”：

| 解释范围 | 未覆盖行为 | 典型验证手段 |
| --- | --- | --- |
| 某个 attention head 是否实现复制前文模式 | 整篇长文生成的整体规划 | activation patching |
| 某个 feature 是否表示括号配对状态 | 所有语法现象的统一机制 | direction probing + ablation |
| 一组 head+MLP 是否构成局部 circuit | 模型在开放任务中的全部策略 | path patching |
| 某层残差流中的特定语义方向 | 该语义在全模型中的唯一来源 | cross-layer tracing |

所以，机械可解释性不是“把黑箱彻底打开”，而是“在明确问题边界的前提下，对一段可复现的内部计算给出因果解释”。

玩具例子可以这样理解：如果你解释了“模型看到 `A B A` 后倾向预测 `B`”这一现象的内部路径，你解释的是一个局部模式补全机制，不是“语言理解本身”。

---

## 核心机制与推导

transformer 的核心中间状态之一是残差流。残差流可以先理解成：每一层处理后不断累积、传递的主表示向量。设当前某个位置的残差流为 $\mathbf{r}$，某个 feature 对应的方向为 $\mathbf{d}_f$，那么这个 feature 的激活强度通常写成：

$$
\text{activation}_f = \mathbf{r} \cdot \mathbf{d}_f
$$

这就是向量点积。白话说，它衡量当前状态在这个“语义方向”上有多强。

如果 $\mathbf{d}_f$ 已归一化，那么点积越大，说明这个 feature 越强。若为负，则说明该状态朝相反语义方向偏移。

最小数值例子：

$$
\mathbf{r} = [2,1], \quad \mathbf{d}_f = [1,0]
$$

则有：

$$
\mathbf{r} \cdot \mathbf{d}_f = 2 \times 1 + 1 \times 0 = 2
$$

这表示“由 $\mathbf{d}_f$ 所代表的 feature”在当前上下文中以强度 2 被激活。新手可以把它理解成：残差流里有很多可能的信息方向，现在我们拿一个探针沿某个方向去量，量出来是 2，说明这条语义线索很明显。

但 feature 还不是 circuit。feature 解决的是“这里有什么信息”，circuit 解决的是“这些信息如何被后续模块利用”。

一个简化的 induction 电路可画成：

```text
Head A: 记录“当前位置前一个 token 是 X”
   ↓
残差流写入“X 的身份”
   ↓
Head B: 去前文找到另一个 X 出现的位置
   ↓
读取“X 后面跟着什么 token”
   ↓
把那个 token 的偏好写入当前 logits
```

这里的关键推导不是形式化证明，而是因果链拆分：

1. 某个 feature 在残差流中出现。
2. 一个 head 读取这个 feature，并把“指向前文某位置”的信息写出去。
3. 另一个 head 使用这个位置信息，读取前文后续 token。
4. 最终 logits 朝复制模式的方向偏移。

真实工程例子是大模型的 in-context pattern completion。给定上下文：

```text
A → 1
B → 2
A →
```

模型预测 `1`，可能并不是“理解了字母和数字的数学关系”，而是激活了一条“看见重复键，去复制前文对应值”的电路。机械可解释性的任务，就是证明这条说法对应到内部计算，而不是只停留在输出现象。

---

## 代码实现

下面给出一个可运行的玩具实现，分成三层：

1. `feature lookup`：定义 feature 方向并计算投影。
2. `activation logging`：记录不同输入下的激活。
3. `circuit validation`：通过“打补丁”或“置零”模拟验证某条路径是否因果有效。

```python
import math

def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))

def norm(v):
    return math.sqrt(dot(v, v))

def normalize(v):
    n = norm(v)
    assert n > 0
    return [x / n for x in v]

def feature_activation(residual, direction):
    d = normalize(direction)
    return dot(residual, d)

def patch_head_output(original_heads, head_idx, patched_value):
    patched = [list(h) for h in original_heads]
    patched[head_idx] = list(patched_value)
    return patched

def combine_heads(head_outputs):
    # 玩具模型：把多个 head 输出直接相加，得到对 logits 的总影响
    dim = len(head_outputs[0])
    total = [0.0] * dim
    for h in head_outputs:
        for i in range(dim):
            total[i] += h[i]
    return total

# feature lookup: residual 在“复制模式”方向上的激活
residual = [2.0, 1.0]
direction = [1.0, 0.0]
act = feature_activation(residual, direction)
assert abs(act - 2.0) < 1e-9

# circuit validation: 两个 head 共同推动 token_0 的 logits
heads = [
    [0.8, 0.1],   # Head A
    [0.7, -0.2],  # Head B
]
combined = combine_heads(heads)
assert combined[0] == 1.5

# 把 Head B 置零，相当于做一次简化 ablation / patching
patched_heads = patch_head_output(heads, 1, [0.0, 0.0])
patched_combined = combine_heads(patched_heads)

# token_0 的偏好下降，说明 Head B 对输出有因果贡献
assert patched_combined[0] == 0.8
assert patched_combined[0] < combined[0]
```

这段代码不是完整 transformer，而是把关键思想抽出来了。它表达的是：先测 feature 强度，再测试“拿掉某个部件后输出是否变化”。

下表把代码和机制对应起来：

| 代码部分 | 对应机制 | 作用 |
| --- | --- | --- |
| `feature_activation` | feature 投影 | 量化语义方向强度 |
| `normalize` | 方向归一化 | 避免向量长度干扰解释 |
| `combine_heads` | 多 head 汇总 | 模拟 circuit 对输出的联合贡献 |
| `patch_head_output` | patching / ablation | 验证某个 head 是否因果必要 |

如果落到真实工程中，通常会做这三件事：

| 工程步骤 | 真实做法 | 目的 |
| --- | --- | --- |
| 读取中间表示 | hook 某层残差流、attention 输出、MLP 输出 | 获取可分析对象 |
| 定义方向 | 用人工发现方向、稀疏自编码器特征或线性 probe | 找到可能有语义的 feature |
| 因果验证 | activation patching、path patching、ablation | 区分“相关”与“机制” |

真实工程例子：分析客服模型是否在“检测到退款意图”后触发了某条安全电路。可以记录若干层残差流，寻找与“退款请求”强相关的 feature 方向，再在推理时 patch 掉可疑 head，观察模型是否失去对退款流程的正确调用。如果 patch 后行为显著退化，才说明这条路径可能真在执行该功能。

---

## 工程权衡与常见坑

机械可解释性最常见的误区，是把“看见局部规律”误当成“理解了整个模型”。

第一类坑是覆盖范围过小。你可能成功复现了“复制上一个 token”行为，但这不代表你解释了摘要、写诗、工具调用。复杂任务往往由大量分布式计算共同完成，已知 circuit 只是其中一小块。

第二类坑是把 attention weight 当成解释。attention weight 可以先理解成：某个位置对另一个位置分配了多少读取权重。但“看哪里”不等于“写了什么”，高注意力也不等于高因果贡献。真正影响输出的，是读到的信息、写入残差流的内容，以及后续路径如何使用这些内容。

第三类坑是忽略 superposition。`superposition` 可以先理解成：多个特征被压缩叠加在同一组参数里。这样一来，单 neuron 观察会非常混乱，因为一个 neuron 可能同时参与多个语义。

第四类坑是只做相关分析，不做干预验证。没有 patching、ablation 或路径替换，就很难说“这条路径实现了该功能”，最多只能说“它和这个行为一起出现”。

| 坑项 | 典型错误 | 避坑策略 | 验证方法 |
| --- | --- | --- | --- |
| 以偏概全 | 解释一个局部模式后声称理解了模型 | 明确写出行为边界 | task-specific evaluation |
| 误把高 attention 当解释 | 只看注意力热图下结论 | 同时检查写入内容和下游影响 | path patching |
| 迷信单 neuron | 把 neuron 直接等同于概念 | 转向方向或子空间分析 | feature projection |
| 只看相关不做因果 | 看到激活同步变化就下结论 | 做 ablation / patching | output delta 对比 |

一个新手版例子：你发现某个 circuit 能稳定复现“把前文同样的键对应值复制出来”。这说明它可能参与模式补全，但如果让模型写一首诗，这条 circuit 很可能几乎不发挥作用。已知部分和未知部分必须分开写。

---

## 替代方案与适用边界

机械可解释性不是唯一的可解释方法。另一大类方法是统计归因，例如 saliency、LIME、SHAP。它们更擅长回答“哪些输入重要”，但不擅长回答“内部是怎么算的”。

LIME 可以先理解成：在输入附近做很多扰动，再拟合一个简单代理模型，看哪些输入维度最影响预测。它适合快速给出局部重要性提示，但它通常不告诉你 transformer 内部哪几个 head、哪层残差流、哪条路径在执行计算。

两类方法的差异如下：

| 维度 | statistical attribution（LIME/SHAP 等） | mechanistic circuit analysis |
| --- | --- | --- |
| 解释粒度 | 输入特征重要性 | 内部 feature 与 circuit |
| 核心问题 | 哪些输入影响结果 | 模型内部如何计算 |
| 验证方式 | 扰动输入后看输出变化 | patching / ablation / path tracing |
| 成本 | 通常较低 | 通常较高 |
| 适用场景 | 快速定位相关输入 | 研究内部机制、安全分析、失效诊断 |

所以，如果你的目标是“找出哪几个 token 比较重要”，统计归因就够了；如果你的目标是“确认模型是不是通过某条复制电路完成这个预测”，机械可解释性更合适。

它的适用边界也很明确：

1. 对结构清晰、可插桩的 transformer 最有用。
2. 对需要追踪内部算法的安全与对齐问题特别有价值。
3. 对超大规模、开放式复杂行为，成本会迅速上升，且很难做到完整覆盖。
4. 对某些重参数化严重、机制高度分布式的网络，局部电路解释可能不稳定，此时代理模型或局部拟合更现实。

新手版判断标准可以记成一句话：想知道“哪个输入重要”，先用归因；想知道“模型内部怎么算”，才进入机械可解释性。

---

## 参考资料

- Wikipedia: *Mechanistic interpretability*。适合先建立定义和问题范围。
- ProbablyAligned: *Mechanistic Interpretability: Circuits, Superposition, and Sparse Autoencoders*。适合看 circuits、induction head 与实践直觉。
- LearnMechInterp: *What is Interpretability?*。适合理解 feature、方向投影与基础公式。
- Springer 综述：*Explaining AI through mechanistic interpretability*。适合补足哲学视角与研究议题边界。
- Acalytica 相关文章。适合快速浏览工程风险、superposition、验证方法等常见问题。
