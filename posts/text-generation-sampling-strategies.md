## 核心结论

文本生成采样策略控制的是“模型每一步如何从候选 token 中选下一个 token”，不是让模型学到新知识，也不是改变模型能力。

token 是模型处理文本的最小片段，可以是一个字、一个词，也可以是词的一部分。模型生成文本时，并不是一次性写完整句话，而是重复执行同一个过程：根据已有上下文预测下一个 token 的分数，再把分数变成概率，最后选出一个 token 接到后面。

采样策略主要调节四件事：

| 参数 | 作用 | 直觉效果 | 适用场景 |
|---|---|---|---|
| `temperature` | 缩放概率分布 | 低温更稳定，高温更多样 | 控制整体随机性 |
| `top_k` | 只保留概率最高的 k 个 token | 去掉长尾低质量候选 | 降低发散风险 |
| `top_p` | 保留累计概率达到 p 的最小 token 集合 | 根据分布自适应裁剪候选 | 开放式生成、聊天、续写 |
| `repetition_penalty` | 降低已出现 token 的分数 | 减少复读和模板化 | 长文本、故事、闲聊 |

核心公式是：

$$
p_i = softmax(z_i / T)
$$

其中 $z_i$ 是第 $i$ 个 token 的原始分数，$T$ 是温度。`temperature=0.2` 时，概率分布更尖锐，模型更倾向选择最高概率 token，输出通常更保守。`temperature=1.0, top_p=0.9` 时，候选范围更宽，输出更自然，但也更容易发散。

同一个问题“介绍一下 Transformer”，低温参数可能输出稳定定义：“Transformer 是一种基于注意力机制的神经网络架构”。较高温度加 `top_p` 可能输出更丰富的表述：“Transformer 用自注意力直接建模序列中不同位置的关系，因此能并行处理文本”。这不是模型能力变强，而是选词策略变了。

---

## 问题定义与边界

采样发生在解码阶段。解码阶段是模型已经训练好之后，根据输入上下文一步步生成输出的过程。

本文只讨论解码阶段的采样策略，不讨论模型训练方法、提示词工程、模型架构设计，也不讨论检索增强生成。换句话说，本文关注的是：模型已经给出了下一步候选 token 的分数，我们如何处理这些分数并选出 token。

一个玩具例子：

输入是“今天”，模型下一步可能给出候选：

| 候选 token | 原始分数 |
|---|---:|
| 天气 | 5.1 |
| 我 | 3.2 |
| 股票 | 1.4 |
| 香蕉 | -0.8 |

如果选中“天气”，上下文变成“今天天气”。下一步模型再预测“很好”“不错”“很差”等候选。最终一句话是逐 token 生成出来的，不是一次性决定的。

典型流程如下：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 1 | 上下文 | logits | 模型给每个 token 打分 |
| 2 | logits | 缩放后 logits | 应用 `temperature` |
| 3 | 概率或 logits | 候选集合 | 应用 `top_k` / `top_p` |
| 4 | 已生成 token | 调整后 logits | 应用 `repetition_penalty` |
| 5 | 候选概率 | 下一个 token | 抽样或直接选择 |
| 6 | token | 新上下文 | 拼接后进入下一轮 |

logits 是模型输出的未归一化分数。它不是概率，可以为负，也不要求总和为 1。softmax 是把 logits 转成概率分布的函数，转完之后所有 token 的概率加起来等于 1。

---

## 核心机制与推导

基础 softmax 定义为：

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

加入温度后：

$$
p_i = softmax(z_i / T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

当 $T < 1$，大的 logit 会被进一步放大，分布更尖锐，模型更稳定。当 $T > 1$，logits 差距被压缩，分布更平坦，低概率 token 更容易被采到。当 $T$ 接近 0 时，行为接近总是选最高概率 token。

设某一步 logits 为：

| token | A | B | C | D |
|---|---:|---:|---:|---:|
| logits `z` | 4.0 | 3.0 | 2.0 | 1.0 |
| softmax 概率 | 0.64 | 0.24 | 0.09 | 0.03 |
| `top_k=2` | 保留 | 保留 | 删除 | 删除 |
| `top_p=0.90` | 保留 | 保留 | 保留 | 删除 |
| B 已出现且 `r=1.2` | 4.0 | 2.5 | 2.0 | 1.0 |

`top_k` 是固定数量截断：只保留概率最高的 $k$ 个 token，其他 token 的 logit 置为 $-\infty$，再重新归一化。`top_k=2` 时，只保留 A 和 B。

`top_p` 又叫核采样。它不是固定保留几个 token，而是从高到低排序后，取累计概率达到阈值 $p$ 的最小集合：

$$
S_p = \min S,\quad \sum_{i \in S} p_i \ge p
$$

在上面的例子中，A+B 的概率是 $0.64+0.24=0.88$，还没有达到 0.90，所以要加入 C。A+B+C 的概率是 0.97，因此 `top_p=0.90` 保留 A、B、C。

重复惩罚用于降低已经出现过的 token 再次出现的概率。常见实现中，若 token $t$ 已出现，惩罚系数为 $r>1$：

$$
z'_t =
\begin{cases}
z_t / r, & z_t > 0 \\
z_t \cdot r, & z_t < 0
\end{cases}
$$

正 logit 除以 $r$ 会变小，负 logit 乘以 $r$ 会更负，二者都会降低该 token 的相对概率。若 B 已经出现过，且 $z_B=3.0, r=1.2$，则 $z'_B=2.5$。

真实工程例子是客服回复。用户问“订单什么时候发货”，系统需要回答稳定、礼貌、少发散。参数通常会偏保守，例如低温度、中等 `top_p`，并使用轻微重复惩罚避免“请您耐心等待，请您耐心等待”这类复读。

---

## 代码实现

实际工程里通常不手写完整采样算法，而是在生成接口中配置参数。以 Hugging Face Transformers 为例，`do_sample=True` 表示启用采样。若 `do_sample=False`，生成通常走贪心解码或 beam search 相关逻辑，`temperature`、`top_k`、`top_p` 一般不会按采样方式生效。

```python
# 需要安装 transformers，并加载真实模型后运行。
# 这里只展示 generate() 的关键配置位置。

outputs = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    max_new_tokens=128,
)
```

采样流程可以写成伪代码：

```text
logits = model(context)
logits = apply_temperature(logits, temperature)
logits = apply_repetition_penalty(logits, generated_tokens, penalty)
logits = apply_top_k_filter(logits, k)
logits = apply_top_p_filter(logits, p)
probs = softmax(logits)
next_token = sample(probs)
context = context + next_token
```

下面是一个可运行的 Python 玩具实现，演示 softmax、温度、`top_k`、`top_p` 和重复惩罚的效果：

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]

def apply_temperature(logits, temperature):
    assert temperature > 0
    return [x / temperature for x in logits]

def apply_repetition_penalty(logits, seen_indexes, penalty):
    assert penalty >= 1.0
    adjusted = logits[:]
    for i in seen_indexes:
        if adjusted[i] > 0:
            adjusted[i] = adjusted[i] / penalty
        else:
            adjusted[i] = adjusted[i] * penalty
    return adjusted

def top_k_indexes(probs, k):
    return sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]

def top_p_indexes(probs, p):
    assert 0 < p <= 1
    ordered = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept = []
    acc = 0.0
    for i in ordered:
        kept.append(i)
        acc += probs[i]
        if acc >= p:
            break
    return kept

tokens = ["A", "B", "C", "D"]
logits = [4.0, 3.0, 2.0, 1.0]

base_probs = softmax(logits)
assert [round(x, 2) for x in base_probs] == [0.64, 0.24, 0.09, 0.03]

assert [tokens[i] for i in top_k_indexes(base_probs, 2)] == ["A", "B"]
assert [tokens[i] for i in top_p_indexes(base_probs, 0.90)] == ["A", "B", "C"]

hot_probs = softmax(apply_temperature(logits, 2.0))
assert hot_probs[0] < base_probs[0]
assert hot_probs[-1] > base_probs[-1]

penalized = apply_repetition_penalty(logits, seen_indexes=[1], penalty=1.2)
assert penalized[1] == 2.5
```

这段代码不是生产级实现，但它能说明核心关系：采样策略不是生成文本之后再修饰，而是在每一步选下一个 token 之前改变候选概率。

---

## 工程权衡与常见坑

采样参数之间不是独立的。`temperature` 提高后，低概率 token 更容易进入结果；如果同时 `top_p` 很高、`top_k` 很大，生成会明显更发散。`top_p` 太低会让输出过度保守，`top_k` 太小会让表达模板化。

| 坑位 | 现象 | 原因 | 规避 |
|---|---|---|---|
| `temperature` 太高 | 内容跳跃、事实不稳 | 概率分布过平 | 从 0.7 到 1.0 小步调整 |
| `top_p` 太低 | 表达单一、保守 | 候选集合太小 | 常从 0.8 到 0.95 起步 |
| `top_k` 太小 | 句子僵硬、重复套路 | 固定截断过强 | 常用 20 到 50 |
| `repetition_penalty` 太强 | 专有名词、代码、格式被破坏 | 重复 token 被误伤 | 用 1.05 到 1.2，必要时加白名单 |
| `do_sample=False` | 参数看似配置但效果不明显 | 没有进入采样模式 | 明确设置 `do_sample=True` |

实用起点如下：

| 任务 | `temperature` | `top_p` | `top_k` | `repetition_penalty` |
|---|---:|---:|---:|---:|
| 事实型问答 | 0.2-0.6 | 0.8-0.9 | 20-50 | 1.0-1.1 |
| 客服回复 | 0.4-0.8 | 0.85-0.95 | 20-50 | 1.05-1.15 |
| 开放式聊天 | 0.7-1.0 | 0.85-0.95 | 40-100 | 1.05-1.2 |
| 故事续写 | 0.8-1.2 | 0.9-0.98 | 50-100 | 1.05-1.2 |
| 强格式输出 | 0.0-0.3 | 可关闭或偏低 | 可关闭或偏低 | 谨慎使用 |

客服回复和故事续写的目标不同。客服回复要求准确、稳定、可控，低温度和较强约束更合适。故事续写需要新鲜表达和情节变化，适合更高温度和更宽的 `top_p`。

重复惩罚尤其要谨慎。代码生成中，括号、缩进、变量名、关键字本来就需要重复；法律、医学、金融文本中，专有名词也可能必须重复。惩罚过强会让模型为了避免重复而改写术语，反而降低正确性。

---

## 替代方案与适用边界

采样不是唯一解码方式。解码方式是“从模型给出的下一 token 分布中决定输出”的总体方法，采样只是其中一类。

| 方法 | 特点 | 优点 | 缺点 | 典型场景 |
|---|---|---|---|---|
| 贪心解码 | 每步选概率最高 token | 稳定、便宜、可复现 | 容易死板、局部最优 | 强格式、简单分类式回答 |
| beam search | 保留多条高分路径 | 适合寻找高概率序列 | 开放生成容易模板化 | 翻译、短摘要 |
| `top_k` | 固定保留 k 个候选 | 简单、可控 | 不适应分布形状 | 通用文本生成 |
| `top_p` | 按累计概率保留候选 | 自适应、更自然 | 参数过高会发散 | 聊天、续写、开放问答 |
| `temperature` | 缩放整体分布 | 控制随机性直接 | 不能单独过滤低质候选 | 与其他策略组合使用 |
| 典型采样 | 保留信息量接近典型值的 token | 减少异常候选 | 理解和调参成本更高 | 研究或高级生成系统 |
| Mirostat | 动态控制困惑度 | 维持稳定多样性 | 实现复杂、框架支持不一 | 长文本开放生成 |

摘要生成和开放式聊天可以说明边界。摘要生成通常要求忠于原文，随机性要低，beam search 或低温度采样更常见。开放式聊天不只追求最高概率句子，还要避免机械重复，因此常用 `temperature + top_p + repetition_penalty` 的组合。

选择建议：

| 目标 | 建议 |
|---|---|
| 事实型问答 | 低温度，减少随机性 |
| 客服回复 | 中低温度 + `top_p`，轻微重复惩罚 |
| 创作类任务 | 较高温度 + 较高 `top_p` |
| 强格式输出 | 尽量低随机性，必要时用约束解码 |
| 代码生成 | 谨慎使用重复惩罚，避免破坏语法结构 |

没有一种参数组合适合所有任务。参数选择应该从任务目标出发：是稳定性优先，还是多样性优先；是要严格格式，还是要自然表达；是短答案，还是长文本。采样策略的价值在于把这些目标映射到可调参数上。

---

## 参考资料

1. [The Curious Case of Neural Text Degeneration](https://pubs.cs.uct.ac.za/id/eprint/1407/)
2. [Hugging Face Transformers Text Generation](https://huggingface.co/docs/transformers/v4.42.4/en/main_classes/text_generation)
3. [Hugging Face Transformers Internal Generation Utilities](https://huggingface.co/docs/transformers/v4.21.1/internal/generation_utils)
4. [Hugging Face Transformers Generation Logits Process Source](https://huggingface.co/transformers/v3.5.1/_modules/transformers/generation_logits_process.html)
