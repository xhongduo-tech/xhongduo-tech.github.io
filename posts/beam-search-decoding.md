## 核心结论

Beam Search 是一种解码策略：在生成序列时，每一步只保留得分最高的 `k` 个部分序列继续展开，用有限搜索近似寻找最终最优输出。

这里的“解码”指模型已经训练好之后，根据输入生成输出的过程。例如机器翻译模型看到“我喜欢机器学习”，需要一步步生成 “I like machine learning”。

| 策略 | 一句话定义 | 优点 | 缺点 |
|---|---|---|---|
| 贪心解码 | 每一步只选当前概率最高的词 | 快，简单 | 容易被早期局部最优带偏 |
| Beam Search | 每一步保留得分最高的 `k` 条候选路径 | 质量通常高于贪心，成本可控 | 比贪心慢，仍不保证全局最优 |
| 穷举搜索 | 枚举所有可能序列后选最优 | 理论上最充分 | 序列空间指数爆炸，工程上不可用 |

贪心解码像“每一步只看当前红绿灯”，当前路口哪个方向看起来最好就走哪边。Beam Search 像“同时保留几个备选路线图”，每一步把这些路线都继续往前推一段，再丢掉明显差的路线。若 `beam_width = 3`，模型每一步就保留 3 条当前得分最高的候选序列。

核心取舍很明确：`k` 越大，搜索越充分，越接近全局最优；但计算量、显存占用和延迟也会增加。Beam Search 的目标不是穷尽所有可能，而是在可接受成本下找到足够好的输出。

---

## 问题定义与边界

序列生成问题的目标是：给定输入 $X$，寻找一个输出序列 $Y=(y_1,\dots,y_T)$，使条件概率 $P(Y|X)$ 尽可能大。

通常不会直接乘概率，因为很多小概率相乘会造成数值下溢，也不方便累加比较。因此工程实现里常用对数概率：

$$
\log P(Y|X)=\sum_{t=1}^{T}\log p(y_t|y_{<t},X)
$$

其中 $y_{<t}$ 表示第 `t` 步之前已经生成的所有 token。token 是模型处理文本的基本单位，可以是一个词、一个子词，也可以是一个符号。

| 项目 | 说明 |
|---|---|
| 输入 | 源文本、语音特征、上文提示词等，统一记为 $X$ |
| 输出 | 模型逐步生成的序列 $Y=(y_1,\dots,y_T)$ |
| 目标 | 找到让 $\log P(Y|X)$ 最大的输出 |
| 适用场景 | 机器翻译、摘要生成、语音识别、结构化文本生成 |
| 非适用场景 | 需要强多样性、强随机性、开放式创作的生成任务 |

玩具例子：输入是“我喜欢机器学习”。第 1 步模型可能给出：

| 候选 token | 条件概率 |
|---|---:|
| I | 0.55 |
| We | 0.20 |
| You | 0.10 |
| It | 0.05 |

贪心解码会直接选 `I`，然后只沿着 `I` 后面的路径继续。Beam Search 如果 `k=3`，会保留 `I`、`We`、`You` 三条部分序列，再分别扩展它们的下一个词。这样做可以避免某个单步高概率选择导致整句质量下降。

真实工程例子：机器翻译系统中，输入 “我喜欢机器学习”，模型可能生成：

| 路径 | 前两步候选 | 说明 |
|---|---|---|
| 路径 A | I like | 最常见直译 |
| 路径 B | I love | 语气更强 |
| 路径 C | I enjoy | 更自然但依赖上下文 |

Beam Search 会同时保留多条路径，让后续 token 再参与整体比较。它不是训练方法，不会改变模型参数；它只发生在推理阶段，用来从模型概率分布中选择输出。

---

## 核心机制与推导

Beam Search 的核心循环是“扩展 + 剪枝”。

扩展是把当前保留的每条部分序列都接上词表中的候选 token。剪枝是计算所有新候选的累计分数，只保留 top-`k` 条继续。

| 步骤 | 当前 beam | 扩展后候选 | 剪枝结果 |
|---|---|---|---|
| 0 | `[""]` | `["I", "We", "You", ...]` | 保留 top-`k` |
| 1 | `["I", "We"]` | `["I like", "I love", "We like", "We are", ...]` | 保留 top-`k` |
| 2 | `["I like", "We are"]` | 继续扩展每条路径 | 保留 top-`k` |
| 结束 | 遇到 `<eos>` 或达到最大长度 | 汇总完成序列 | 选最终得分最高者 |

`<eos>` 是 end of sequence 的缩写，表示序列结束符。模型生成 `<eos>` 后，这条路径通常不再继续扩展。

基础累计分数为：

$$
\log P(Y|X)=\sum_{t=1}^{T}\log p(y_t|y_{<t},X)
$$

两步玩具例子如下。设 `beam_width = 2`。

第 1 步：

| 候选 | 概率 | 对数分数 |
|---|---:|---:|
| I | 0.6 | -0.511 |
| We | 0.3 | -1.204 |
| You | 0.1 | -2.303 |

保留 `I` 和 `We`。

第 2 步扩展：

| 新候选 | 条件概率 | 累计对数分数 |
|---|---:|---:|
| I like | 0.5 | -0.511 + log(0.5) = -1.204 |
| I love | 0.3 | -1.715 |
| We like | 0.4 | -2.120 |
| We are | 0.5 | -1.897 |

保留 `I like` 和 `I love`。这就是每步展开、每步剪枝。

但原始累计对数概率有一个重要问题：它天然偏向短序列。因为概率都小于等于 1，$\log p$ 通常小于等于 0，序列越长，累加项越多，分数越容易变小。

假设两条完整候选每步概率都是 `0.7`：

| 序列 | 长度 | 原始分数 |
|---|---:|---:|
| A | 2 | $\log(0.7 \times 0.7) \approx -0.713$ |
| B | 3 | $\log(0.7 \times 0.7 \times 0.7) \approx -1.071$ |

按原始分数，A 更高，因为 `-0.713 > -1.071`。这说明即使每一步质量一样，短序列也更容易胜出。

一个简单修正是看平均每步对数概率：

| 序列 | 原始分数 | 平均分数 |
|---|---:|---:|
| A | -0.713 | -0.357 |
| B | -1.071 | -0.357 |

平均后，长度偏置被削弱。

GNMT 系统中常用更细的修正形式：

$$
S(Y,X)=\frac{\log P(Y|X)}{lp(Y)}+cp(X,Y)
$$

长度惩罚为：

$$
lp(Y)=\left(\frac{5+|Y|}{6}\right)^\alpha
$$

其中 $|Y|$ 是输出长度，$\alpha$ 控制长度惩罚强度。$\alpha=0$ 时没有长度惩罚，$\alpha$ 越大，对长度的修正越强。

覆盖度惩罚为：

$$
cp(X,Y)=\beta\sum_{i=1}^{|X|}\log\left(\min\left(\sum_{j=1}^{|Y|}a_{i,j},1.0\right)\right)
$$

其中 $a_{i,j}$ 是第 `j` 个目标步对源输入第 `i` 个位置的注意力权重。注意力权重可以理解为：生成当前 token 时，模型主要在看输入里的哪些位置。覆盖度惩罚用于减少漏译或漏摘要，鼓励输出覆盖源输入的重要部分。

---

## 代码实现

Beam Search 可以拆成四部分：状态维护、候选扩展、分数计算、终止判断。

| 组成 | 作用 |
|---|---|
| 当前序列 | 已经生成的 token 列表 |
| 累计分数 | 当前路径的对数概率和或修正分数 |
| 是否结束 | 是否已经生成 `<eos>` |
| 注意力覆盖 | 记录源输入位置被关注的程度，用于覆盖惩罚 |

伪代码如下：

```text
beams = [空序列，分数 0]

for step in range(max_decode_len):
    candidates = []

    for beam in beams:
        if beam 已结束:
            candidates.append(beam)
            continue

        next_token_probs = model(beam.sequence)

        for token in vocab:
            new_sequence = beam.sequence + [token]
            new_score = beam.score + log(next_token_probs[token])
            candidates.append((new_sequence, new_score))

    beams = candidates 中分数最高的 top-k 条

    if 所有 beams 都已结束:
        break

return beams 中最终分数最高的序列
```

下面是一个可运行的 Python 玩具实现。它不依赖深度学习框架，只模拟模型在不同前缀下返回的下一个 token 概率。

```python
from math import log

EOS = "<eos>"

def toy_next_probs(prefix):
    table = {
        (): {"I": 0.6, "We": 0.3, "You": 0.1},
        ("I",): {"like": 0.5, "love": 0.3, EOS: 0.2},
        ("We",): {"are": 0.5, "like": 0.4, EOS: 0.1},
        ("You",): {"like": 0.7, EOS: 0.3},
        ("I", "like"): {"ML": 0.7, EOS: 0.3},
        ("I", "love"): {"ML": 0.4, EOS: 0.6},
        ("We", "are"): {"engineers": 0.6, EOS: 0.4},
        ("We", "like"): {"ML": 0.6, EOS: 0.4},
    }
    return table.get(tuple(prefix), {EOS: 1.0})

def length_penalty(length, alpha=0.0):
    return ((5 + length) / 6) ** alpha

def beam_search(beam_width=2, max_decode_len=4, alpha=0.0):
    beams = [([], 0.0, False)]

    for _ in range(max_decode_len):
        candidates = []

        for seq, score, ended in beams:
            if ended:
                candidates.append((seq, score, ended))
                continue

            for token, prob in toy_next_probs(seq).items():
                new_seq = seq + [token]
                new_score = score + log(prob)
                candidates.append((new_seq, new_score, token == EOS))

        def normalized_score(item):
            seq, score, _ = item
            effective_len = max(1, len([t for t in seq if t != EOS]))
            return score / length_penalty(effective_len, alpha)

        beams = sorted(candidates, key=normalized_score, reverse=True)[:beam_width]

        if all(ended for _, _, ended in beams):
            break

    best = max(
        beams,
        key=lambda item: item[1] / length_penalty(
            max(1, len([t for t in item[0] if t != EOS])),
            alpha,
        ),
    )
    return best

best_seq, best_score, ended = beam_search(beam_width=2, max_decode_len=4)
assert best_seq[:2] == ["I", "like"]
assert isinstance(best_score, float)
assert ended is True
print(best_seq, round(best_score, 3))
```

关键参数如下：

| 参数 | 含义 | 常见影响 |
|---|---|---|
| `beam_width` | 每步保留多少条候选 | 越大越慢，通常质量先升后趋稳 |
| `length_penalty_weight` | 长度惩罚强度 | 过小容易短，过大可能啰嗦 |
| `coverage_penalty_weight` | 覆盖惩罚强度 | 可减少漏译，但过大可能生成不自然 |
| `max_decode_len` | 最大生成长度 | 防止无限生成 |
| `eos` | 结束符 | 控制序列何时停止 |

真实工程中，Beam Search 往往还要处理 batch 维度。batch 是一次并行处理的样本集合。实现时不能只维护一组 beams，而是要为 batch 中每个输入样本分别维护 `k` 条候选，并在张量形状上处理 `[batch_size, beam_width, sequence_length]` 这类结构。

---

## 工程权衡与常见坑

Beam size 不是越大越好。更大的 `k` 会让每一步扩展更多候选，计算复杂度近似随 `k` 线性上升。如果词表大小为 $V$，每步需要从约 $k \times V$ 个候选里选 top-`k`。在 Transformer 模型中，解码还会受 KV cache、batch size、最大长度等因素影响。

| beam size | 常见用途 | 建议 |
|---:|---|---|
| 1 | 等价于贪心解码 | 用作速度基线 |
| 4 | 翻译、摘要常见起点 | 优先尝试 |
| 5 | 很多 NMT 系统的常用值 | 适合验证集扫描 |
| 8 | 搜索更充分 | 观察延迟是否可接受 |
| 10+ | 更高成本 | 只在质量确有收益时使用 |

常见问题通常不是“算法没跑通”，而是“打分目标和任务目标不一致”。

| 问题现象 | 可能原因 | 规避办法 |
|---|---|---|
| 译文过短 | 原始 `log P` 偏向短序列，`<eos>` 太容易胜出 | 加长度归一化、最小长度约束、EOS 惩罚 |
| 摘要提前结束 | 结束符概率过高，beam 中短路径占优 | 设置 `min_decode_len`，调低 EOS 优先级 |
| 重复词堆叠 | 模型本身分布有重复倾向，beam 放大高概率重复 | 加重复惩罚、no-repeat ngram 约束 |
| 漏译源句 | 注意力未覆盖源输入关键位置 | 检查注意力，适度加覆盖度惩罚 |
| beam 越大结果越差 | 搜索目标与评测目标不一致，可能出现 beam search curse | 在验证集上扫参，不默认追求大 beam |
| 输出过长 | 长度惩罚过强或 EOS 被压制 | 降低长度惩罚，检查结束条件 |

“beam search curse” 指在某些神经机器翻译场景中，beam size 增大后搜索更充分，但最终质量反而下降。它说明模型概率最高的序列不一定是人类评价最好的序列。Beam Search 只能优化你给它的打分函数，不能自动修复模型分布和任务指标之间的偏差。

EOS 处理尤其重要。若不允许过早输出 `<eos>`，可以加最小长度约束；若模型迟迟不结束，需要限制最大长度；若 `<eos>` 总是过早胜出，可以对结束符加惩罚或调长度归一化。工程上通常先看真实错例，再改规则。

真实工程例子：一个摘要系统在验证集上经常输出一句很短的摘要，例如“公司发布公告。” 原因可能不是模型不会摘要，而是原始对数概率让短句得分更高。此时应先比较不同长度惩罚下的输出，再看 ROUGE 或人工评估，而不是直接把 `beam_width` 从 4 加到 20。

---

## 替代方案与适用边界

Beam Search 适合“正确性和稳定性优先”的任务，不适合所有文本生成任务。

| 解码策略 | 核心逻辑 | 适合任务 | 主要问题 |
|---|---|---|---|
| Greedy | 每步选最高概率 token | 低延迟、结果要求不高 | 容易局部最优 |
| Beam Search | 每步保留 top-`k` 条路径 | 翻译、语音识别、结构化摘要 | 多样性不足，成本更高 |
| Random Sampling | 按概率随机采样 | 开放式创作 | 不稳定，可能跑偏 |
| Top-k Sampling | 只在概率最高的 k 个 token 中采样 | 聊天、故事生成 | k 难调，仍可能不稳定 |
| Nucleus Sampling | 在累计概率达到 p 的候选集合中采样 | 开放式生成 | 输出更随机，复现性较弱 |

Nucleus Sampling 又叫 top-p sampling。它不是固定保留 `k` 个 token，而是保留累计概率达到阈值 `p` 的最小候选集合。例如 `p=0.9` 时，只在覆盖 90% 概率质量的 token 集合中采样。

| 任务目标 | 推荐策略 | 原因 |
|---|---|---|
| 机器翻译 | Beam Search | 要尽量正确、完整、稳定 |
| 语音识别 | Beam Search 或带语言模型的 Beam Search | 需要在多个候选转写中找高分路径 |
| 结构化摘要 | Beam Search | 输出格式和事实稳定性更重要 |
| 故事续写 | Top-k / Nucleus Sampling | 需要多样性，不能太单一 |
| 闲聊机器人 | Sampling 类方法 | 多个合理回答都可接受 |
| 代码补全 | Greedy、Beam Search、采样混合 | 取决于是否要多个候选 |

同样是文本生成，翻译任务适合 Beam Search，因为目标通常是“尽量正确”。故事续写更适合采样，因为目标往往是“别太单一”。如果所有用户都输入同一个开头，Beam Search 倾向于给出非常相似的高概率续写；采样则能产生更多变化。

不该用 Beam Search 的典型场景有三类。第一，任务明确要求多样性，例如创意写作、角色对话、多候选头脑风暴。第二，延迟预算极低，例如实时交互里每毫秒都重要。第三，模型打分函数和业务目标差距很大，例如最高概率输出经常安全但空泛，此时扩大 beam 只会更稳定地找到这种空泛答案。

---

## 参考资料

1. [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://research.google/pubs/pub45610)  
   用途：对应长度惩罚、覆盖度惩罚和 GNMT 风格 Beam Search 修正。

2. [TensorFlow Text Decoding API: Beam search](https://www.tensorflow.org/text/guide/decoding_api)  
   用途：对应工程实现中的解码 API、beam width 和推理阶段使用方式。

3. [TensorFlow Addons BeamSearchDecoder](https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/BeamSearchDecoder)  
   用途：对应真实框架里的 Beam Search 参数、状态维护和结束符处理。

4. [Breaking the Beam Search Curse](https://aclanthology.org/D18-1342/)  
   用途：对应 beam size 变大不一定更好的问题，以及搜索目标和任务目标不一致的讨论。

5. [OpenNMT Beam Search Documentation](https://opennmt.net/OpenNMT/translation/beam_search/)  
   用途：对应长度归一化、覆盖度惩罚、n-gram 重复约束等工程细节。

| 标题 | 作者/机构 | 年份 | 对应章节 |
|---|---|---:|---|
| Google’s Neural Machine Translation System | Wu et al. | 2016 | 核心机制与推导 |
| TensorFlow Text Decoding API | TensorFlow | - | 代码实现 |
| BeamSearchDecoder | TensorFlow Addons | - | 代码实现 |
| Breaking the Beam Search Curse | Yang, Huang, Ma | 2018 | 工程权衡与常见坑 |
| OpenNMT Beam Search Documentation | OpenNMT | - | 工程权衡与常见坑 |
