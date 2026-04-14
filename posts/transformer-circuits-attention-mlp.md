## 核心结论

Transformer Circuits 关心的不是“这一层整体在干什么”，而是“哪些注意力头和哪些 MLP 神经元拼起来，形成了一个可描述的算法”。这里的核心结论是：很多看起来像“上下文学习”的行为，能被拆成一个相对清晰的子电路，即“模式匹配→复制→放大输出”。

其中，**注意力头**可以理解成“在上下文里找相关位置的检索器”，负责定位“当前模式上一次出现在哪里”；**MLP** 可以理解成“把检索结果变成稳定输出的非线性变换器”，负责把复制来的向量推到更容易成为下一个 token 的方向上。

最典型的例子是 **induction head**。它的作用不是直接理解语义，而是学会一种序列算法：如果我在当前位置看到一个曾经出现过的前缀，就去找那次前缀后面跟着什么，然后把那个“后继 token”复制出来。配合 copying head 和后续 MLP，这就形成了一个可工作的 in-context learning 子电路。

玩具例子是序列 `[A, B, A]`。第三个 token 是 `A`，模型会把它的 query 与上文 key 比较，发现第一个 `A` 最相似，于是把注意力集中到第一个 `A` 附近。接着，OV 路径把“那个 A 后面跟着的 B”对应的信息写回当前 residual stream。MLP 再把这个和 `B` 相关的表示做非线性放大，最终使 `B` 的 logit 更高，于是下一个 token 更可能输出 `B`。

真实工程里，这种机制常出现在 few-shot prompt 中。比如提示词里多次出现：

`Question: ... Answer: ...`

当模型看到一个新 `Question:` 时，某些 induction/copying 头会尝试定位之前相似的问答模板，再把对应的 `Answer:` 之后那段模式拉过来。MLP 则帮助模型把这个“复制过来的候选答案方向”变成更稳定的预测，而不是一闪而过的弱信号。

| 模块 | 白话解释 | 主要作用 | 对输出的贡献 |
|---|---|---|---|
| Attention | 在上下文里找相关位置 | 找到“上一次出现的相同前缀” | 决定复制来源 |
| OV 路径 | 把找到的位置内容搬回来 | 复制后继 token 的表示 | 把候选答案写入 residual |
| MLP | 做非线性特征放大 | 强化与目标 token 相关的方向 | 提高目标 logit |
| Residual Stream | 各模块共享的主通道 | 累加 attention 与 MLP 写入 | 保留并传递复制信号 |

可以把数据流画成一个简图：

`query -> key 匹配 -> softmax 聚焦 -> OV 复制后继表示 -> residual stream -> MLP 放大 -> logits`

---

## 问题定义与边界

这里要回答的问题不是“Transformer 为什么强”，而是一个更窄的问题：**当 prompt 中已经出现过某种模式时，模型怎样把这段模式在新位置复用出来**。

更具体地说，in-context learning 至少要做两件事：

1. 识别当前输入和上文哪一段模式相似。
2. 把那段模式后面跟着的内容复制到当前预测里。

对于 induction circuit，这两步分别主要由 attention 和 MLP 协同完成。attention 先做“找”，MLP 再做“稳”。

新手版本可以直接记成一句话：**模型先找到“上一次类似提问在哪里”，再复制“那次提问后面的回答是什么”。**

但这个结论有明确边界。它更适用于以下情况：

| 输入条件 | 模块是否容易工作 | 常见输出效果 |
|---|---|---|
| prompt 中有明显重复前缀 | attention 容易对齐，MLP 容易放大 | 复制成功 |
| 问答模板多次出现 | induction/copying 头容易形成 | few-shot 效果明显 |
| 几乎没有重复模式 | attention 找不到稳定匹配 | 退回到一般语言建模 |
| 重复极少且训练中未学到 prefix matching | 电路难形成 | 更像 n-gram 猜测 |

这里的 **prefix matching**，白话说就是“按前缀去找曾经出现过的相同开头模式”。

还要强调一个容易误解的点：`[A, B, A]` 这个最小例子只能说明“第三个 A 能把注意力对准第一个 A”，但不能说明所有复制都会成功。因为 induction head 真正擅长的是“根据重复前缀预测后继 token”，而不是凭空创造新模式。如果 prompt 里根本没有可靠重复，模型就没有足够证据知道应该复制什么。

例如：

- 在 `[A, B, A]` 中，第三个 `A` 可以对齐第一个 `A`。
- 但如果训练阶段从未学会“相同 token 后继往往可复制”的规则，或者样本里这种规则极少，它也可能只输出一个高频 token，而不是 `B`。
- 所以 induction head 的前提不是“有相同 token 就够了”，而是“有可学习的重复结构”。

从边界上看，本文聚焦的是 **induction head + position-wise MLP** 如何协同完成复制，不展开讨论更复杂的多头组合、跨层 superposition，也不把所有 in-context learning 都归因于这一路径。很多任务还会调用更高层语义头、位置头、名称移动头等机制。

---

## 核心机制与推导

先看标准 attention 公式。设输入残差表示为 $X \in \mathbb{R}^{n \times d}$，其中 $n$ 是序列长度，$d$ 是隐藏维度。对某个头 $h$：

$$
Q^h = XW_q^h,\quad K^h = XW_k^h,\quad V^h = XW_v^h
$$

这里的 **query** 可以白话理解成“我现在要找什么”，**key** 是“我这里能被什么匹配”，**value** 是“如果你选中我，就把什么内容拿走”。

注意力权重为：

$$
P^h = \mathrm{softmax}\left(\frac{Q^h(K^h)^\top}{\sqrt{d_h}} + M\right)
$$

其中 $M$ 是 mask，白话说就是“禁止偷看未来 token 的约束”。

头的输出为：

$$
H^h = P^h V^h W_o^h
$$

其中 $W_o^h$ 与前面的 $W_v^h$ 一起常被合称为 **OV 路径**。它的白话解释是：**先从被选中的位置读出 value，再投影回模型主空间**。

对 induction head 来说，关键不在公式本身，而在 softmax 峰值的含义。若当前位置 $t$ 的 query 与某个早期位置 $i$ 的 key 高度相关，那么：

$$
P^h_{t,i} \approx 1
$$

这表示模型认为“第 $i$ 个位置是当前最值得参考的过去位置”。如果第 $i$ 个位置对应的是“此前出现过的相同前缀”，那么通过 OV 路径，模型就可能把“那个前缀之后的后继信息”写回当前位置。

玩具例子 `[A,B,A]` 可以拆成以下顺序：

| 步骤 | 输入 | 发生什么 | 输出 |
|---|---|---|---|
| 1 | 第三个 `A` 的表示 | 计算 query | 当前要找“和我相似的过去位置” |
| 2 | 所有历史 token 的表示 | 计算 key 并做 $Q\cdot K$ | 第一个 `A` 得分最高 |
| 3 | softmax | 注意力集中到第一个 `A` | 得到尖峰权重 |
| 4 | OV 投影 | 从那个位置读取后继相关表示 | residual 中出现 `B` 方向信号 |
| 5 | MLP | 非线性放大 `B` 方向 | `B` 的 logit 升高 |

如果把残差写出来，更容易看清 attention 和 MLP 的配合。设第 $l$ 层输入残差为 $r^l$，attention 后为：

$$
r^{l+} = r^l + \sum_h H^h
$$

然后送入 MLP：

$$
\mathrm{MLP}^l(r^{l+}) = W_2 \sigma(W_1 r^{l+})
$$

最终得到：

$$
r^{l+1} = r^{l+} + \mathrm{MLP}^l(r^{l+})
$$

这里的 **激活函数 $\sigma$** 可以白话理解成“让网络只在某些特征足够强时明显响应的开关”。

为什么 MLP 很重要？因为 attention 只是把某个方向写进 residual stream，这个方向未必足够强，也未必正好对应目标 token 的输出方向。MLP 会把 residual 中若干组合特征做非线性重编码，把“像是应该输出 B”的弱证据，转成“更明确支持 B logit”的强证据。

这也是为什么只看 attention heatmap 往往不够。heatmap 告诉你模型“看向了哪里”，但不保证“看到了以后真的把什么写进了输出”。真正决定行为的是：

1. query-key 是否找对位置；
2. OV 是否把有用的后继表示搬回来；
3. MLP 是否把这个表示变成稳定的输出偏置。

真实工程例子可以看 few-shot QA。假设 prompt 中有多组：

- `Question: 2+2=? Answer: 4`
- `Question: 3+3=? Answer: 6`

当模型看到新的一行 `Question: 5+5=? Answer:` 时，某些头会对“Question ... Answer”这个局部模板做匹配。它并不一定真正“理解数学”后再输出，而可能先借 induction 电路找到“问答模板后面的内容通常是答案”。如果这个问题过于简单，模型可能还会叠加参数记忆；但从电路角度看，模板复用本身就是一种可解释的中间算法。

---

## 代码实现

下面用一个最小 Python 例子模拟 `[A, B, A]` 的前向过程。它不是完整 Transformer，只保留“匹配第一个 A，把 B 的方向写回，再用一个简单 MLP 放大”的主干。

```python
import math

def softmax(xs):
    exps = [math.exp(x - max(xs)) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

# 玩具 embedding
A = [1.0, 0.0]
B = [0.0, 1.0]

# 序列 [A, B, A]
X = [A, B, A]

# 第 3 个 token 的 query，假设更偏向匹配 A
q = [1.0, 0.0]

# keys 直接取 token 自身，便于演示
K = X

# values 不是“当前位置 token”，而是“若匹配到该位置，希望带回的后继信息”
# 对第 1 个 A，我们希望复制它后继的 B；其他位置给很弱信号
V = [
    [0.0, 1.0],   # 第1个A -> 后继是B
    [0.1, 0.1],   # 第2个B -> 无明显复制目标
    [0.1, 0.1],   # 第3个A -> 当前位自身
]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

scores = [dot(q, k) / math.sqrt(2) for k in K]
weights = softmax(scores)

# attention 输出
attn_out = [
    sum(weights[i] * V[i][j] for i in range(3))
    for j in range(2)
]

# residual: 当前 token 表示 + attention 写入
residual = [X[2][j] + attn_out[j] for j in range(2)]

# 一个极简 MLP：先线性投影，再做 ReLU，再映回 logits 空间
def relu(x):
    return max(0.0, x)

# W1: 把第二维(B方向)提取出来
hidden = [
    relu(1.5 * residual[1] - 0.2),  # 专门响应 B 方向
    relu(0.2 * residual[0])         # 弱响应 A 方向
]

# W2: hidden -> logits[A], logits[B]
logits = [
    0.3 * hidden[1],
    1.2 * hidden[0]
]

pred = "B" if logits[1] > logits[0] else "A"

assert weights[0] > weights[1] and weights[0] > weights[2]
assert attn_out[1] > attn_out[0]
assert pred == "B"

print("scores =", scores)
print("weights =", weights)
print("attn_out =", attn_out)
print("residual =", residual)
print("logits =", logits)
print("prediction =", pred)
```

这个例子的含义是：

| 变量名 | shape | 含义 |
|---|---|---|
| `X` | `(3, 2)` | 三个 token 的 toy embedding |
| `q` | `(2,)` | 第三个 token 发出的查询 |
| `K` | `(3, 2)` | 所有历史位置的 key |
| `V` | `(3, 2)` | 若选中某位置，带回的后继信息 |
| `weights` | `(3,)` | 对三个位置的注意力分布 |
| `attn_out` | `(2,)` | attention 写回 residual 的向量 |
| `residual` | `(2,)` | 当前表示与复制信号叠加后的结果 |
| `hidden` | `(2,)` | MLP 中间激活 |
| `logits` | `(2,)` | 对 A/B 的输出分数 |

如果写成更贴近实际模型的伪代码，可以概括为：

```python
# x: [seq_len, d_model]
Q = x @ W_q          # [seq_len, d_head]
K = x @ W_k          # [seq_len, d_head]
V = x @ W_v          # [seq_len, d_head]

scores = (Q @ K.T) / sqrt(d_head) + causal_mask
weights = softmax(scores, dim=-1)

head_out = weights @ V @ W_o
residual = x + head_out

mlp_hidden = activation(residual @ W_1 + b_1)
mlp_out = mlp_hidden @ W_2 + b_2

residual_next = residual + mlp_out
logits = residual_next @ W_U
```

在几何可解释性视角里，**几何** 指“不同 token、特征、方向在向量空间中的相对位置关系”。之所以要区分 attention 和 MLP，是因为前者更像“搬运已有方向”，后者更像“把若干方向组合成更可判别的方向”。

---

## 工程权衡与常见坑

真正把 induction circuit 训练出来，并不只是把模型堆大就行。它依赖训练语料里存在足够多、足够稳定的重复结构。否则模型更容易学到一种更懒的策略：只根据最近几个 token 做局部统计，也就是接近 n-gram 的行为。

这里的 **n-gram**，白话说就是“只看前面几个词的局部共现频率来猜下一个词”。

常见问题可以归纳成下表：

| 问题 | 症状 | 解决方案 |
|---|---|---|
| 重复 bigram 太少 | attention 不形成稳定尖峰 | 增加重复模板数据 |
| prompt 结构不一致 | query-key 匹配漂移 | 统一 few-shot 格式 |
| 复制信号被其他残差路径淹没 | 看到了但输出不稳定 | 加强正则或分层分析 OV->MLP 路径 |
| 训练过早偏向短程统计 | 模型像局部补全器 | curriculum 先教重复模式再加噪声 |
| 只看 heatmap 不看 OV/MLP | 解释错因果链路 | 联合分析 logits lens、激活与写入方向 |

**curriculum**，白话说就是“按难度分阶段喂训练样本”。

新手最容易遇到的工程误判是：看到模型 few-shot 效果差，就以为“attention 不行”或“上下文窗口不够”。实际上常见原因是训练 prompt 缺少重复问答模式，导致模型根本没学会“看到重复前缀就复制后继”的算法，只剩下最近邻式的局部续写。

这时通常有两个修复方向：

1. **数据增强**：在训练或微调语料里显式加入重复模板，如多组 `Question -> Answer`、多组 `Input -> Output`。
2. **curriculum 学习**：先用重复结构清晰的数据让模型形成 induction 行为，再逐步加入更自然、更嘈杂的样本。

下面是一个极简数据生成伪代码，展示如何往训练样本里插入重复模式：

```python
def build_sample(question, answer, distractor):
    demos = [
        f"Question: {question}\nAnswer: {answer}",
        f"Question: {distractor}\nAnswer: dummy",
        f"Question: {question}\nAnswer:"
    ]
    return "\n\n".join(demos)

sample = build_sample("CPU 是什么", "中央处理器", "GPU 是什么")
assert "Question: CPU 是什么" in sample
assert sample.count("Question:") == 3
```

另一个常见坑是低估 MLP。很多分析只盯着“这个头把注意力打到了哪里”，却忽略“这个头写回 residual 的内容是否真的支持目标输出”。如果 MLP 的通道没有把复制信号进一步稳定化，模型可能表现为：

- attention 图很漂亮；
- 但 logits 变化很小；
- 最终输出仍被其他高频路径覆盖。

所以工程上更稳妥的做法是把分析链路拉通：`attention pattern -> OV 写入 -> residual 变化 -> MLP 激活 -> logits 变化`。只看其中一段，容易把相关性误判成因果。

---

## 替代方案与适用边界

induction + MLP 电路很适合解释一类任务：**prompt 里有重复模式，且目标是把这个模式复用出来**。例如 few-shot 分类、问答模板、抽取式摘要、结构化转换。

但如果 prompt 里每个 question 都完全不同，没有可对齐的重复前缀，只靠 induction head 就很难复制。因为它的工作前提就是“找到以前出现过的相似模式”。没有相似模式，它就没有可靠的检索锚点。

新手可以这样理解：**如果历史里没有“上一次差不多的问题”，模型就没法靠 induction 电路去抄答案。**

这时常见替代方案有两类：

- **prefix tuning**：给模型前面塞一小段可训练前缀，相当于准备一组固定提示向量。
- **memory-augmented layer**：给模型接一个额外记忆模块，相当于多了一块可检索存储区。

| 方案 | 适用场景 | 数据需求 | 实现难度 | 局限 |
|---|---|---|---|---|
| Induction circuit | few-shot、重复模板明显 | 需要重复结构 | 低到中 | 无重复时效果有限 |
| Prefix tuning | 任务模板固定、想低成本适配 | 少量任务数据即可 | 中 | 泛化依赖前缀设计 |
| Memory layer | 长时依赖、需要外部检索 | 需要构建记忆读写机制 | 高 | 系统复杂、训练更难 |

一个真实工程判断标准是：

- 如果你的任务是“输入格式稳定、上下文里有示例、答案风格应跟示例一致”，优先考虑 induction circuit 的解释框架。
- 如果你的任务是“输入变化大，但希望长期保留某种固定行为”，prefix tuning 往往更直接。
- 如果你的任务是“上下文太长，重复模式不在当前窗口，或者需要跨会话记忆”，memory layer 更合适。

对于创作性文本、开放式推理、低重复度任务，induction 电路通常只是局部参与者，不会是全部解释。此时更高层语义电路、世界知识参数记忆和多步推理路径会变得更重要。

---

## 参考资料

1. Emergent Mind, *Induction Heads in Transformers*。
用途：入门 attention 如何做重复匹配与复制，适合理解“模式匹配→复制”的主线。对新手来说，可以把它理解成“attention 像找重复模式的雷达”。

2. Emergent Mind, *Induction Head in Transformers*。
用途：补充 induction head 作为子电路时的结构视角，帮助把单个头放到多层协同里看，而不是孤立地看 heatmap。

3. OpenReview, *How Transformers Implement Induction Heads*（2024）。
用途：关注训练动态与算法分解，说明 induction 行为不是凭空出现，而是在训练过程中逐渐形成的可学习机制。

4. Learn Mechanistic Interpretability, *MLPs in Transformers*。
用途：帮助理解 MLP 不只是“附属模块”，而是会把 residual 中的复制信号转换为更稳定的特征与 logit 偏置。

建议读原始论文和长文版本获取完整实验图表、训练设置和消融结果。博客式总结适合先建立因果链路，但数值结论、层级定位和头的具体分工，仍应回到原文核对。
