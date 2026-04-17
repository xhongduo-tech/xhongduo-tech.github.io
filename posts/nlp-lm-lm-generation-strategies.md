## 核心结论

语言模型的生成策略，本质上是在“每一步选哪个下一个 token”这个问题上定义规则。token 可以理解为模型处理文字时使用的最小片段，可能是一个字、一个词，或者词的一部分。不同策略不会改变模型本身学到的知识，但会显著改变输出的稳定性、多样性、延迟和成本。

先给结论：

| 策略 | 决策方式 | 适合任务 | 典型参数 | 主要代价 |
| --- | --- | --- | --- | --- |
| 贪婪解码 Greedy | 每步直接取最大概率 token | 短答案、分类式生成、强约束补全 | 无 | 容易局部最优 |
| 束搜索 Beam Search | 同时保留多个高分前缀再比较 | 翻译、摘要、确定性改写 | `beam=4~8` | 计算量近似随 beam 线性增加 |
| 温度采样 Temperature | 用温度调节概率分布再采样 | 对话、续写、开放问答 | `T=0.7~1.0` | 温度过高会失控 |
| Top-k | 只在最高 `k` 个词中采样 | 需要限制尾部噪声时 | `k=20~100` | `k` 固定，适应性差 |
| Top-p / Nucleus | 在累计概率达到 `p` 的最小集合中采样 | 通用对话、创意生成 | `p=0.9~0.95` | 输出会随分布形状波动 |
| 推测解码 Speculative Decoding | 小模型起草，大模型验证 | 高吞吐推理、在线服务 | `draft K=3~8` | 需要双模型协同 |

如果任务目标是“尽量稳定、尽量可复现”，优先考虑贪婪或束搜索。如果任务目标是“自然、不死板、有一定表达变化”，优先考虑温度采样结合 top-p。对于服务端推理，如果瓶颈在大模型逐 token 生成，推测解码通常是最有工程价值的优化之一，因为它不主要改变文本风格，而是改变“每次前向传播能确认多少个 token”。

---

## 问题定义与边界

生成策略只解决“如何从模型给出的概率分布里选 token”，不解决模型知识是否正确，也不解决提示词是否合理。换句话说，它是推理阶段的控制器，不是训练阶段的能力来源。

这件事的边界要先说清楚：

1. 它不提升模型上限。
同一个基础模型，如果事实知识不足，换再复杂的解码策略，也不会凭空变得更懂。

2. 它会改变输出分布。
所谓输出分布，就是“哪些答案更常出现、哪些更少出现”的整体概率形状。温度、top-k、top-p 都是在直接干预这个分布。

3. 它影响任务风格。
翻译和摘要追求稳定、可比较、低随机；聊天、故事续写、文案生成需要一定随机性，否则内容会僵硬。

4. 它影响系统成本。
束搜索增加候选分支，推测解码增加额外草稿模型，二者都会改动算力分配，但方向不同。前者多半增加延迟，后者目标是减少平均延迟或提升吞吐。

一个简单的任务边界可以这样理解：

| 任务 | 目标 | 推荐策略 | 原因 |
| --- | --- | --- | --- |
| 机器翻译 | 准确、稳定、可复现 | Beam Search | 多路径比较更适合确定性任务 |
| 搜索问答 | 简洁、可靠 | Greedy 或低温采样 | 需要抑制发散 |
| 聊天助手 | 自然、不过度重复 | `temperature + top-p` | 兼顾流畅与多样性 |
| 创意写作 | 新颖、有变化 | 较高温度 + top-p | 允许探索长尾表达 |
| 高并发推理服务 | 更高吞吐 | Speculative Decoding | 目标是减少大模型逐步确认成本 |

“零基础”最容易混淆的一点是：随机采样不等于胡乱输出。它依然在高概率词附近采样，只是允许不是第一名的词被选中。相反，束搜索也不等于一定最好，它只是在“模型自己认为高分”的候选里找更优路径，如果模型评分机制本身偏短、偏保守，beam 也会把这种偏差放大。

玩具例子：假设下一步只有三个候选词：

| 候选词 | 概率 |
| --- | --- |
| “今天” | 0.62 |
| “此时” | 0.25 |
| “目前” | 0.13 |

贪婪解码永远选“今天”。采样则可能选“此时”或“目前”。如果你做的是法律条文翻译，通常不希望这种波动；如果你做的是对话生成，适度波动反而更自然。

---

## 核心机制与推导

先看最基础的生成过程。给定上下文 $x$，模型在第 $t$ 步输出一个条件概率分布：

$$
p(y_t \mid x, y_{<t})
$$

其中 $y_{<t}$ 表示此前已经生成的 token 序列。生成策略做的事，就是把这个分布变成一个具体选择。

### 1. 贪婪解码

贪婪解码最直接：

$$
y_t = \arg\max_i p(i \mid x, y_{<t})
$$

也就是每一步都选当前概率最大的 token。它快、稳定、实现简单，但有明显问题：每一步的局部最优，不一定组成全局最优。

玩具例子：模型要生成一句短语。

- 路径 A：第一步概率最高，但第二步之后越来越差
- 路径 B：第一步不是最高，但后续整体更通顺

贪婪解码一旦第一步选了 A，就不会再回头。

### 2. 束搜索

束搜索的思路是“不要只押一个答案，同时保留多个候选前缀”。前缀可以理解为“生成到当前步的半成品句子”。

假设 beam size 为 $k$，每一步保留评分最高的 $k$ 个前缀。序列评分常用累计对数概率：

$$
s(y)=\sum_{t=1}^{|y|}\log p(y_t \mid x, y_{<t})
$$

因为长句子会累积更多负对数值，所以工程里经常做长度归一化。常见形式是：

$$
s_{\text{norm}}(y)=\frac{\sum_{t=1}^{|y|}\log p(y_t \mid x, y_{<t})}{|y|^\alpha}
$$

其中 $\alpha$ 是长度惩罚参数。白话解释：它用来缓和模型“偏爱短句”的倾向。

为什么翻译常用 `beam=4~8`？因为翻译是高约束任务，正确表达通常集中在少数高概率路径里。beam 太小会漏掉合理候选，beam 太大又会带来额外计算，还可能出现所谓 beam search curse，也就是 beam 变大后质量反而下降。

### 3. 温度采样

模型真正输出给解码器的，通常不是概率，而是 logits。logits 可以理解为“还没经过归一化的原始打分”。温度 $T$ 的作用，是先把 logits 除以 $T$，再做 softmax：

$$
\tilde p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

这里有三个关键结论：

- $T<1$：分布更尖锐，高分词更占优
- $T=1$：保持原始分布
- $T>1$：分布更平坦，低分词更容易被采到

还是用一个数值例子。假设 logits 为 `[2.0, 1.0, 0.1]`。当 $T=0.5$ 时，最大值和其他值的差距被放大，第一名更像“几乎必选”；当 $T=1.5$ 时，后两项被抬高，更容易出现意外词。

这也是为什么“温度越高越有创造力”这句话只能算半对。更准确地说，温度越高，模型越愿意探索低概率词；低概率词里有新意，也有噪声。

### 4. Top-k 与 Top-p

只调温度还不够，因为尾部会保留大量极低概率 token。于是有了候选截断。

Top-k：只保留概率最高的 $k$ 个 token，再在里面采样。
白话解释：先把明显不靠谱的词全删掉，再在前几名里摇号。

Top-p：不固定词数，而是取累计概率达到阈值 $p$ 的最小集合。定义可以写成：

$$
V^{(p)}=\min \left\{V: \sum_{x\in V} P(x\mid x_{<t}) \ge p \right\}
$$

白话解释：不是固定取前 50 个词，而是“把最有希望的一批词拿出来，直到它们总概率够 0.9 为止”。

为什么 top-p 在对话里通常比 top-k 更常用？因为语言分布有时很尖，有时很平。固定 `k=50` 在不同上下文里代表的“概率质量”差别很大，而 top-p 会自适应当前分布形状。

一个直观例子：

| 排名 | 概率 |
| --- | --- |
| 1 | 0.70 |
| 2 | 0.18 |
| 3 | 0.06 |
| 4 | 0.03 |
| 5 | 0.02 |
| 其余 | 0.01 |

如果 `top-p=0.9`，那么只需要前 3 个词，因为 $0.70+0.18+0.06=0.94$。这时第 4 个词虽然排得不低，也不会进入采样集合。

### 5. 推测解码

推测解码不是改变“怎么选词”，而是改变“怎么更快确认词”。

普通自回归生成有一个根本瓶颈：第 $t+1$ 个 token 必须等第 $t$ 个 token 确认之后才能继续。所以大模型通常是一 token 一 token 地走，吞吐上限很容易被卡住。

推测解码的做法是：

1. 用一个更小、更快的草稿模型先连续起草 $K$ 个 token
2. 用目标大模型一次性验证这段草稿
3. 接受其中最长一致前缀
4. 从首次不一致的位置继续生成

如果草稿模型和目标模型分布足够接近，就会有较高接受率。常见的吞吐收益来自“大模型一次前向确认多个 token”，而不是每个 token 都单独跑一次。

一个常见的期望公式是：

$$
E=\frac{1-\alpha^{K+1}}{1-\alpha}
$$

其中 $\alpha$ 可以粗略理解为单步接受率，$K$ 是一次起草的 token 数。这个公式表达的是：若接受率稳定，单次迭代平均能确认的 token 数会大于 1。工程上，这正是推测解码能把吞吐拉高到 2 到 3 倍附近的原因之一。

真实工程例子：一个在线问答服务使用 70B 级别模型，原始路径每次只能确认 1 个 token，GPU 大量时间花在重复的逐步同步上。引入一个同系列小模型做 draft 后，大模型开始经常一次接受 3 到 5 个 token。用户看到的是首 token 后的续写更快，服务端看到的是整体 QPS 提升，而文本质量变化很小。

---

## 代码实现

下面先用一个不依赖第三方库的可运行 Python 例子演示温度、top-k、top-p 的核心逻辑。代码故意写得直接，因为目标是把机制讲清楚，而不是做工业级优化。

```python
import math
import random

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]

def apply_temperature(logits, temperature=1.0):
    assert temperature > 0
    return [x / temperature for x in logits]

def top_k_filter(probs, k):
    assert 1 <= k <= len(probs)
    idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    keep = set(idx)
    filtered = [p if i in keep else 0.0 for i, p in enumerate(probs)]
    s = sum(filtered)
    return [p / s for p in filtered]

def top_p_filter(probs, p):
    assert 0 < p <= 1
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    total = 0.0
    keep = []
    for i in order:
        keep.append(i)
        total += probs[i]
        if total >= p:
            break
    keep = set(keep)
    filtered = [v if i in keep else 0.0 for i, v in enumerate(probs)]
    s = sum(filtered)
    return [v / s for v in filtered]

def greedy_decode_step(logits, vocab):
    probs = softmax(logits)
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return vocab[idx], probs[idx]

def sample_decode_step(logits, vocab, temperature=1.0, top_p=None, top_k=None, seed=0):
    random.seed(seed)
    probs = softmax(apply_temperature(logits, temperature))
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)

    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return vocab[i], probs
    return vocab[-1], probs

vocab = ["今天", "此时", "目前", "随后"]
logits = [2.0, 1.0, 0.1, -1.2]

word, prob = greedy_decode_step(logits, vocab)
assert word == "今天"
assert 0 < prob < 1

sampled_word, probs = sample_decode_step(
    logits, vocab, temperature=0.8, top_p=0.9, seed=42
)
assert abs(sum(probs) - 1.0) < 1e-9
assert sampled_word in vocab

# top-p=0.9 后，尾部低概率词通常会被清掉
assert probs[-1] == 0.0

print("greedy:", word, prob)
print("sample:", sampled_word, probs)
```

这个例子体现了三个事实：

1. 贪婪解码只看最大概率项。
2. 温度先改分布，再谈采样。
3. top-p 和 top-k 都是“先截断候选，再重新归一化”。

如果要写一个最小束搜索，可以用“维护若干前缀及其累计 log 概率”的方式实现：

```python
import math

def beam_search_step(beams, next_token_probs, beam_size):
    new_beams = []
    for prefix, score in beams:
        for token, prob in next_token_probs.items():
            new_prefix = prefix + [token]
            new_score = score + math.log(prob)
            new_beams.append((new_prefix, new_score))
    new_beams.sort(key=lambda x: x[1], reverse=True)
    return new_beams[:beam_size]

beams = [([], 0.0)]
step1 = {"我": 0.6, "你": 0.3, "他": 0.1}
beams = beam_search_step(beams, step1, beam_size=2)

assert len(beams) == 2
assert beams[0][0] == ["我"]
```

真实工程里很少有人自己手写这些逻辑，通常会直接使用推理框架的参数接口。例如：

- Hugging Face Transformers：通过 `temperature`、`top_k`、`top_p`、`num_beams` 控制生成
- vLLM：同样支持采样参数，并支持 speculative decoding
- TensorRT-LLM：更偏部署与高性能推理，支持多种推测解码变体

一个真实工程例子是机器翻译服务：

- 线上接口收到一句中文
- 模型以 `beam=4` 生成 4 条高分候选
- 服务端对候选做长度约束、禁词过滤、术语表替换
- 最终返回分数最高且满足规则的结果

这类场景里，束搜索的价值不只是“更准一点”，还因为它天然给你 n-best 候选，方便后处理和 rerank。rerank 就是“二次排序”，用额外规则或模型重新选最合适答案。

---

## 工程权衡与常见坑

第一类坑是把“稳定”和“高质量”混为一谈。贪婪和低温采样确实更稳定，但稳定不代表更好。对于开放式问答，过于保守的参数会导致答案模板化、重复严重、信息密度下降。

第二类坑是 beam 开太大。很多人直觉上认为 beam 越大越好，但工程里经常不是这样。beam 增大后，模型会更执着于自己高分的保守路径，结果可能更短、更重复，甚至更差。因此翻译里常见的经验值是 `4~8`，而不是盲目拉到 `16` 或 `32`。

第三类坑是只调温度，不调候选截断。比如把 `T` 提高到 `1.3`，但仍允许长尾词大量参与采样，就容易出现局部语义漂移。更稳妥的做法通常是配合 `top-p=0.9~0.95`，让探索发生在“还算合理”的候选集合内。

第四类坑是把 top-k 当成通用默认值。top-k 的问题在于上下文变化时适应性不足。某些步的分布非常尖锐，前 5 个词已经覆盖 98% 概率；另一些步的分布很平，前 50 个词也可能只覆盖 60%。固定 `k` 无法反映这种变化，所以很多对话系统更偏好 top-p。

第五类坑是推测解码选错草稿模型。草稿模型越小越快，但如果它和目标模型分布差太远，接受率会低，结果就是大模型总在拒绝草稿，额外验证成本反而抵消收益。经验上，草稿模型最好和目标模型同系列、同 tokenizer、分布尽量接近。

第六类坑是把“平均吞吐提升”误解成“单请求尾延迟一定下降”。推测解码常常能显著提高整体吞吐，但在复杂调度、KV cache 压力、批处理波动存在时，P99 延迟不一定同步按比例改善。吞吐和尾延迟是两个指标，必须分开看。

可以把常见坑和对策压缩成一张表：

| 问题 | 现象 | 原因 | 常见对策 |
| --- | --- | --- | --- |
| Beam 太大 | 质量不升反降 | 模型偏差被放大 | `beam=4~8`，加长度惩罚 |
| 温度太高 | 语义漂移、胡言乱语 | 长尾 token 被放大 | 降低 `T`，增加 `top-p` 约束 |
| 只用 greedy | 重复、模板化 | 无探索能力 | 改为低温 `top-p` |
| top-k 固定过小 | 表达僵硬 | 候选集合过窄 | 改用 top-p 或增大 `k` |
| 草稿模型不匹配 | 推测解码收益差 | 接受率低 | 选同系列小模型 |
| 只看平均速度 | 线上体验不稳定 | 忽略尾延迟和批处理 | 同时监控 QPS、P95、P99 |

---

## 替代方案与适用边界

如果只从“下一词选择”看，主流方案可以分成确定性和随机性两大类，但真实工程里往往是组合使用，而不是二选一。

第一类替代方案是受约束解码。受约束解码就是“强制输出满足某些规则的文本”，比如必须包含某个术语、必须输出 JSON、不能出现禁词。这类方法在工具调用、结构化输出、代码生成里很常见。它和 beam 或 sampling 并不冲突，常常是叠加关系。

第二类替代方案是 rerank 流程。不是一步直接生成最终答案，而是先生成多个候选，再用规则、打分模型或业务特征做二次排序。对于翻译、检索增强生成、营销文案评选，这种流程比单纯调解码参数更稳定。

第三类替代方案是推测解码的单模型变体，例如 Medusa、EAGLE 一类思路。它们不一定需要一个完全独立的小模型，而是在目标模型上扩展多 token 预测能力。适用边界是：你愿意为更复杂的推理图和部署方案换取更高吞吐。

第四类替代方案是直接接受低随机度输出。如果任务本身是信息抽取、标签生成、格式修复，那么最优解往往不是复杂采样，而是简单的 greedy 或低温采样。解码策略必须服务任务，不是越高级越好。

可以用下面这张表收束：

| 方案 | 更适合什么问题 | 优点 | 边界 |
| --- | --- | --- | --- |
| Greedy | 格式化输出、短答案 | 简单、快、稳定 | 容易死板 |
| Beam Search | 翻译、摘要、改写 | 多候选、可重排 | 计算更贵 |
| Temperature + Top-p | 聊天、续写、开放生成 | 自然、多样、可调 | 参数敏感 |
| 受约束解码 | JSON、工具调用、术语约束 | 可控性强 | 实现复杂 |
| Rerank | 多候选评选 | 质量更稳 | 系统链路更长 |
| Speculative Decoding | 高并发推理 | 吞吐提升明显 | 依赖模型匹配度 |

最后给一个判断准则：

- 如果你更关心“别乱说”，优先降低随机性。
- 如果你更关心“别太像模板”，增加受控随机性。
- 如果你更关心“服务跑得更快”，优先看推测解码和系统级优化。
- 如果你更关心“输出必须满足格式”，优先看受约束解码，而不是单纯调温度。

生成策略不是玄学参数，而是把任务目标翻译成概率控制规则。把任务类型、质量指标、延迟预算先定义清楚，参数选择通常就不会跑偏。

---

## 参考资料

- Decoding Strategies: Temperature, Top-k, Top-p Explained: https://explainllm.ru/en/fundamentals/decoding
- OpenNMT Beam Search 文档: https://opennmt.net/OpenNMT/translation/beam_search/
- Beam Search Strategy 概述: https://www.emergentmind.com/topics/beam-search-strategy
- Top-p Sampling 词条: https://en.wikipedia.org/wiki/Top-p_sampling
- 温度缩放与采样讲解: https://machinelearningmastery.com/how-llms-choose-their-words-a-practical-walk-through-of-logits-softmax-and-sampling
- vLLM Speculative Decoding 文档: https://docs.vllm.ai/en/v0.11.2/features/spec_decode/
- NVIDIA 关于 Speculative Decoding 的介绍: https://developer.nvidia.com/blog/?p=92847
- TensorRT-LLM Speculative Decoding 文档: https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html
