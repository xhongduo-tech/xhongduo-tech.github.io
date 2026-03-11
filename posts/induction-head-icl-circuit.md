## 核心结论

归纳头（Induction Head，白话说就是“在上下文里找上次见过的模式，再把后续 token 抄过来”的注意力头）不是单个孤立头，而通常是一个两层电路。

第一层是 previous-token head（前一位头，白话说就是“把前一个 token 的信息写到当前位置”）。它让位置 $t$ 的残差流（residual stream，白话说就是“每层都在上面读写的公共向量通道”）带上位置 $t-1$ 的特征。

第二层才是 induction head。它用当前位置的内容去匹配“哪些历史位置前一位和我现在像”，一旦找到，就把那个历史位置的后继 token 复制到当前预测里。因此它实现的是一个非常具体的上下文学习规则：

$$
\text{如果上下文里出现过 } A \to B,\ \text{那么下次再看到 } A,\ \text{就提高 } B \text{ 的概率。}
$$

这解释了很多 ICL（in-context learning，上下文学习，白话说就是“不更新参数，只靠提示词当场学规则”）的最小机制。Olsson 等人在 2022 年进一步指出，这类电路会在训练中阶段性出现，并与 loss 曲线的突降同步。

---

## 问题定义与边界

这篇文章讨论的不是“模型为什么能做所有 few-shot 任务”，而是一个更窄的问题：

在什么条件下，Transformer 能只靠注意力电路，完成“见过一次 $A\to B$，下次再见 $A$ 就续写 $B$”？

它的边界很明确：

| 输入模式 | 可能触发的机制 | 最低条件 |
| --- | --- | --- |
| `A B ... A` | 标准归纳头 | 需要能匹配重复的 `A`，并复制其后继 `B` |
| `A B A B` | 标准归纳头更容易稳定出现 | 需要两层协同，且上下文里已有完整 bigram |
| `A X B ... A X` | 标准归纳头往往不够 | 需要更长程特征，可能要 averaging / generalized induction |
| `A ... A` 但后继不稳定 | 只能给出混合概率 | 上下文中同一前缀对应多个后继 |

所以，归纳头解决的是“上下文 bigram 回忆”问题，不是通用推理器。它最擅长的，是重复模式、别名映射、模板补全、规则延续这类任务。

一个玩具例子是 `A B A -> ?`。如果模型在前面见过 `A` 后面跟着 `B`，那么第二个 `A` 之后预测 `B` 的概率会升高。

一个真实工程例子是 few-shot 标签映射：

`positive -> 1`, `negative -> 0`, 然后给出 `positive -> ?`。  
如果模型内部存在稳定的归纳电路，它不需要“理解 1 是好情绪”，也能先做最小版本的模式复制：见过 `positive` 后面跟 `1`，那再次见到 `positive` 时把 `1` 推上去。

---

## 核心机制与推导

注意力基本公式不变：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

关键不在公式变了，而在两个头分工不同。

第一层 previous-token head 的作用是“位移一格地写信息”。如果位置 $i$ 原来是 token `A`，位置 $i-1$ 是 `B`，那么经过这一头后，位置 $i$ 的残差流里会混入“前一个 token 是 `B`”这一特征。于是当前 token 不再只表示“我是 A”，还表示“我是一个前面带 B 的 A”。

第二层 induction head 的 QK 电路（query-key 电路，白话说就是“决定看谁”）会做前缀匹配：当前位置如果是 `A`，它会在历史里找“后面跟着和我当前状态相似 token 的位置”。因为第一层已经把“前一位信息”写进来了，所以这个相似性不只是 token 本身，而是“带前缀的 token 状态”。

它的 OV 电路（output-value 电路，白话说就是“看到了之后写回什么”）再把匹配位置的后继信息送回当前位，从而提高下一个 token 的 logits（logits，白话说就是“softmax 前的原始打分”）。

用最小例子 `[A, B, A]` 看：

1. 第一个 `A` 后面出现了 `B`。
2. 第一层 previous-token head 把“前一位是 B”的信息写进第二个 `A` 附近的表示。
3. 第二层 induction head 发现“当前位置这个 A 的上下文形态，和前面那个 A 所在模式有关联”。
4. 它把前面那个 `A` 后面的 `B` 对应的值向量拷回当前位。
5. 输出层看到这个方向上的激活，就提高 `B` 的概率。

可以把它画成简化电路图：

```text
位置:      ...   A   B   ...   A   -> ?
层1头:           └──copy prev──►
残差流:      ... [A] [B] ... [A + prev=B]
层2头:                        └──match earlier A, attend to token after it──► B
结果:                                                        logit(B) 上升
```

论文和复现实验里常用两个分数找这类头：

$$
s_{\text{prev}}^{(\ell,h)}=\frac{1}{T-1}\sum_{i=2}^{T}P_{i,i-1}^{(\ell,h)}
$$

它衡量一个头是不是“总盯着前一位”。

$$
s_{\text{ind}}^{(\ell,h)}=\frac{1}{32}\sum_{i=33}^{64}P_{i,i-31}^{(\ell,h)}
$$

它衡量在重复序列实验里，一个头是不是沿着“上一段相同前缀之后一位”的对角线取注意力。两者都高，才更像完整归纳电路；只高一个，可能只是普通 copy 头或位置头。

---

## 代码实现

下面给一个可运行的玩具实现。它不是真正的 Transformer，而是把“找上次出现的前缀并复制后继”的核心逻辑写成程序，帮助把机制看清楚。

```python
from collections import defaultdict

def induction_predict(tokens):
    """
    给定 token 序列，预测最后一个 token 之后最可能出现什么。
    规则：
    1. 统计上下文里每个 token 的后继
    2. 如果最后一个 token 之前出现过，就返回最常见后继
    """
    assert len(tokens) >= 2

    successor_map = defaultdict(list)
    for i in range(len(tokens) - 1):
        successor_map[tokens[i]].append(tokens[i + 1])

    current = tokens[-1]
    candidates = successor_map.get(current, [])
    if not candidates:
        return None

    counts = defaultdict(int)
    for tok in candidates:
        counts[tok] += 1

    best_tok = max(counts.items(), key=lambda x: x[1])[0]
    return best_tok

# 玩具例子：A 后面见过 B
seq = ["A", "B", "C", "A"]
pred = induction_predict(seq)
assert pred == "B"

# 如果同一前缀对应多个后继，程序返回频率最高者
seq2 = ["A", "B", "A", "D", "A"]
pred2 = induction_predict(seq2)
assert pred2 in {"B", "D"}

print(pred, pred2)
```

如果把它映射回真实模型，可以把关键变量理解成这样：

| 变量 | 含义 | 对应论文视角 |
| --- | --- | --- |
| `successor_map` | 每个前缀 token 见过哪些后继 | 上下文 bigram 记忆 |
| `current` | 当前待续写的 token | 当前 query 所代表的位置 |
| `candidates` | 历史上这个 token 后面跟过什么 | induction head 关注到的位置后继 |
| `counts` | 多个后继时的竞争强度 | logits 上的相对抬升 |

更接近机制解释的伪代码是：

```python
for layer, head in all_heads:
    attn = get_attention_pattern(layer, head)

    prev_score = diagonal_average(attn, offset=-1)
    match_score = induction_diagonal_average(attn, repeat_gap=32)

    if prev_score > thresh_prev:
        mark_as_previous_token_head(layer, head)

    if match_score > thresh_ind:
        mark_as_induction_head(layer, head)
```

真实工程里不能只看 attention pattern，还要看该头输出后，目标 token 的 logits 是否真的上升。因为“看对了地方”不等于“写回了正确方向”。

---

## 工程权衡与常见坑

第一个坑是把“归纳头”误认为“任何复制头”。严格说，只有“匹配历史前缀 + 复制其后继”同时成立，才是归纳机制。只会盯前一位的头，不足以完成 `A B ... A -> B`；只会复制某类值而不做前缀匹配的头，也不行。

第二个坑是只做单头 ablation（消融，白话说就是“把某个头输出强行置零看性能掉多少”）就下结论。因为 previous-token head 和 induction head 是串联关系，拆掉任一侧都可能让整个电路失效。ICLR Blogposts 2026 在图追踪任务上的复现实验显示，消融排名靠前的 induction heads 会让准确率从约 0.90 下降到约 0.60，消融更多会到约 0.40；而只消融少量 top previous-token heads 也会迅速把准确率压到 0.60 以下。这说明两类头都关键，但作用点不同。

第三个坑是忽略冗余。大模型里经常不是“只有一个头会做这件事”，而是多个头部分重叠。你剪掉一个头，短期看性能没掉，不代表归纳机制不存在；可能只是别的头补位了。

第四个坑是把 loss 的突降当成玄学现象。Olsson 等人的核心观察之一是：归纳头不是平滑地一点点长出来，而常常和训练中的相变一起出现。工程上这意味着，如果你在微调时强改注意力结构、位置编码或头剪枝策略，可能正好破坏这个脆弱的形成阶段。

---

## 替代方案与适用边界

标准归纳头适合 bigram，也就是“看见 `A`，抄 `A` 后面的一个 token”。但现实任务常常更长，比如：

`A X B ... A X -> ?`

这时只看 `A` 的后继不够，因为中间还隔着 `X`。于是会出现两类扩展思路：

| 机制 | 解决什么问题 | 适用边界 |
| --- | --- | --- |
| 标准 induction head | `A B ... A -> B` | 短模式、bigram 复制 |
| pointer-arithmetic | 利用位置信息向后偏移一格或多格 | 依赖位置编码进入可组合子空间 |
| averaging + generalized induction | 先把一段历史压成摘要，再匹配更长模式 | 适合 `A X B ... A X` 这类多 token 前缀 |

pointer-arithmetic（指针算术，白话说就是“先找到旧位置，再用位置向量算出后继位置”）是 Anthropic 在 GPT-2 类模型中讨论过的一种替代实现。它不是靠 previous-token head 把“前一位内容”直接写进残差流，而是利用位置编码做偏移。

更一般的 induction 还会配合 averaging head（平均头，白话说就是“把一小段上下文混成一个摘要向量”）。例如在 `A X B ... C X D ... A X` 中，如果只看前一 token `X`，你无法区分应该续 `B` 还是 `D`；这时需要更长的上下文摘要，再由后续头完成匹配与复制。

所以，归纳头不是“ICL 的全部”，而是“ICL 最小可解释子电路之一”。当任务需要长程组合、层级规则、语义抽象时，单个标准归纳头通常不够。

---

## 参考资料

1. Catherine Olsson et al., *In-context Learning and Induction Heads*, 2022.  
   https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

2. Eshaan Nichani, Alex Damian, Jason D. Lee, *How Transformers Learn Causal Structure with Gradient Descent*, ICML 2024.  
   https://proceedings.mlr.press/v235/nichani24a.html

3. *In-context learning of representations can be explained by induction circuits*, ICLR Blogposts 2026.  
   https://iclr-blogposts.github.io/2026/blog/2026/iclr-induction/

4. *Transformer Circuit Exercises*, Anthropic Transformer Circuits Thread，含 induction head 的 K-composition 与 pointer-arithmetic 练习。  
   https://transformer-circuits.pub/2021/exercises/index.html

5. E. Farrell, *Induction head circuits for longer sequences*, 2023，讨论 generalized induction 与 averaging 头。  
   https://efarrell1.github.io/posts/generalised_induction/
