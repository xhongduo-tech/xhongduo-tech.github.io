## 核心结论

Natural Questions，简称 NQ，可以理解为“用真实搜索问题做出来的开放域问答基准”。它不是让模型只在一小段已知上下文里找答案，而是要求系统面对真实用户查询、真实网页结构和“可能根本没有答案”的情况，因此更接近生产环境里的问答系统。

它的核心价值有三点。

第一，查询来自真实 Google 搜索。白话说，它不是研究者手工编出来的标准题，而是用户真实会搜的问题，所以分布更杂、更口语化，也更接近线上流量。

第二，标注分成 long answer 和 short answer 两层。白话说，系统先要找到“答案大概在哪一段”，再从这段里摘出“真正要输出的短答案”。这很像人类读网页时先看目录和段落，再抄出具体日期、名字或数字。

第三，评估也是分层进行。long answer 和 short answer 各自计算精确率、召回率和 F1，不是只看最终抽到的短文本是否对。公式是：

$$
\text{precision}=\frac{TP}{TP+FP},\quad
\text{recall}=\frac{TP}{TP+FN}
$$

$$
F1=\frac{2\cdot \text{precision}\cdot \text{recall}}{\text{precision}+\text{recall}}
$$

在论文报告中，人类标注一致性上界大约是 long answer 87%、short answer 76%。这说明 NQ 不是“做个 span extraction 就能轻松刷高分”的数据集，它本身就有歧义、缺答案、页面结构复杂等现实难点。

一个玩具例子最容易理解。查询是“Barack Obama born”。系统先在 Obama 的 Wikipedia 页面里找到包含出生信息的段落，这一步是 long answer；再从该段里抽出 “August 4, 1961”，这一步是 short answer。如果段落找对了但没抽出日期，那么 long F1 可以是 1，short F1 仍然可能是 0。

---

## 问题定义与边界

NQ 的输入不是“问题 + 已截好的标准上下文”，而是“Google 搜索查询 + 对应的 Wikipedia 页面”。输出则有三种可能：

| 字段 | 含义 | 工程作用 |
| --- | --- | --- |
| `query` | 用户真实搜索词 | 作为检索和阅读的起点 |
| `long answer coords` | 长答案在页面中的起止坐标 | 定位可回答问题的段落或结构块 |
| `short answer span` | 短答案在 long answer 内的文本片段 | 输出实体、短语、数字等 |
| `yes/no` | 是非型短答案 | 处理无法用文本片段表示的问题 |
| `null flag` | 页面中没有可接受答案 | 防止系统逢页必答 |

这里要先把边界讲清楚。

long answer 不是“整篇文章”，也不是“任意一句话”，而是页面结构里的一个较大区域，通常对应段落、列表、表格或其他 HTML 边界明确的块。short answer 则更细，通常是一个或多个 span。span 可以理解为“文本中的连续片段”。

为什么要区分这两层？因为开放域问答通常分成两步：先定位，再抽取。如果训练数据只告诉你最终短答案，不告诉你它来自哪个结构块，系统很容易学成“全页盲抽”，工程上非常不稳定。

再看一个新手更容易代入的例子。用户搜索 `blood pressure normal range`。系统面对的是整篇 Wikipedia 页面，而不是一段提前裁好的医学定义。它应该先找到讲血压范围的段落，再摘出里面的具体数值。如果页面没有明确给出标准范围，正确行为是返回 null，而不是硬猜一个数字。

这就是 NQ 和很多课堂练习型数据集的差异。它要求系统承认“不知道”。这点在生产里非常重要，因为错误回答往往比空回答更糟。

从规模看，NQ 训练集大约 30.7 万条，开发集和测试集各约 7.8k。这个规模足以训练较完整的检索加阅读流水线，也足以暴露很多实际问题，比如查询噪声、页面冗长、答案缺失和标注边界不唯一。

---

## 核心机制与推导

NQ 的核心机制可以概括成一句话：同一页面上同时做“粗定位”和“细抽取”，并分别计分。

先看标注形式。long answer 用起止坐标标出页面中的一个结构块。short answer 则可能是：

| 类型 | 说明 | 例子 |
| --- | --- | --- |
| 单个 span | 一段连续文本 | `August 4, 1961` |
| 多个 span | 多段文本共同组成答案 | 人名由多段节点拼接 |
| yes/no | 是非判断 | `yes` / `no` |
| null | 无可接受答案 | 页面不包含答案 |

这意味着模型不能只做单头输出。至少在概念上，要分别处理：

1. long answer 边界预测  
2. short answer 起止位置预测  
3. yes/no/null 判别  

推导评估时，long 和 short 是两套独立任务。假设有 100 个样本：

- 其中 80 个 long answer 预测正确
- 其中 60 个 short answer 预测正确
- 模型额外在 20 个本不该答的样本上瞎答了 short answer

那么 short precision、recall 就会下降，因为 short 任务不仅要“抽对”，还要“知道什么时候不该抽”。

公式仍然是标准信息检索里的定义：

$$
\text{precision}=\frac{TP}{TP+FP}
$$

精确率可以白话理解成“你报出来的答案里，有多少是真的”。

$$
\text{recall}=\frac{TP}{TP+FN}
$$

召回率可以白话理解成“本来该找出来的答案，你找出来了多少”。

$$
F1=\frac{2PR}{P+R}
$$

F1 是精确率和召回率的调和平均，白话说，它要求两者都不能太差。

继续用 “when was Barack Obama born?” 这个玩具例子：

- 标注 long answer：包含出生日期的段落
- 标注 short answer：`August 4, 1961`

如果模型预测到正确段落，但 short answer 留空，那么：

- long precision = 1，long recall = 1，long F1 = 1
- short precision = 0，short recall = 0，short F1 = 0

这说明“知道答案在哪”和“真正抽到答案”是两种不同能力。工程上不能因为检索好就误以为问答系统已经可用。

真实工程例子更能说明问题。假设你在做一个企业知识库 QA 系统，流程是“向量检索器召回 50 个段落，交给阅读器抽答案”。如果你只训练 short answer 抽取头，模型可能在错误段落里也强行输出一个看起来像答案的实体。线上结果会变成“每个问题都有答案，但很多答案其实来自错误上下文”。NQ 通过 long/short 分层标注，把这个问题直接暴露出来。

---

## 代码实现

落到实现上，NQ 最自然的架构是“检索器 + 阅读器”两阶段流水线。

检索器负责在大量 passages 里找到候选段落。passage 可以理解为“可独立排序的一小段文本”。阅读器则在候选段落上同时预测：

- 哪一段是 long answer
- short answer 的 start/end
- 是否为 yes/no/null

如果你刚入门，最简单的理解方式是：先用一个 ranker 选段落，再用一个 span classifier 在段落内部找答案。只是 NQ 比普通抽取任务多了一层 long answer 监督。

下面给一个可运行的玩具实现。它不依赖深度学习框架，只演示评分和“先 long 后 short”的判定逻辑。

```python
from dataclasses import dataclass

@dataclass
class Example:
    query: str
    gold_long: tuple | None
    gold_short: str | None
    pred_long: tuple | None
    pred_short: str | None

def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

def exact_match(a, b):
    return a == b

def score_examples(examples):
    long_tp = long_fp = long_fn = 0
    short_tp = short_fp = short_fn = 0

    for ex in examples:
        long_ok = exact_match(ex.pred_long, ex.gold_long)
        short_ok = exact_match(ex.pred_short, ex.gold_short)

        if ex.pred_long is not None and ex.gold_long is not None and long_ok:
            long_tp += 1
        elif ex.pred_long is not None and ex.gold_long is None:
            long_fp += 1
        elif ex.pred_long is None and ex.gold_long is not None:
            long_fn += 1
        elif ex.pred_long is not None and ex.gold_long is not None and not long_ok:
            long_fp += 1
            long_fn += 1

        if ex.pred_short is not None and ex.gold_short is not None and short_ok:
            short_tp += 1
        elif ex.pred_short is not None and ex.gold_short is None:
            short_fp += 1
        elif ex.pred_short is None and ex.gold_short is not None:
            short_fn += 1
        elif ex.pred_short is not None and ex.gold_short is not None and not short_ok:
            short_fp += 1
            short_fn += 1

    return {
        "long_f1": f1_from_counts(long_tp, long_fp, long_fn),
        "short_f1": f1_from_counts(short_tp, short_fp, short_fn),
    }

examples = [
    Example(
        query="when was Barack Obama born?",
        gold_long=(120, 180),
        gold_short="August 4, 1961",
        pred_long=(120, 180),
        pred_short="August 4, 1961",
    ),
    Example(
        query="blood pressure normal range",
        gold_long=None,
        gold_short=None,
        pred_long=None,
        pred_short=None,
    ),
    Example(
        query="who invented something",
        gold_long=(300, 360),
        gold_short="John Doe",
        pred_long=(300, 360),
        pred_short=None,
    ),
]

scores = score_examples(examples)
assert abs(scores["long_f1"] - 1.0) < 1e-9
assert 0.0 < scores["short_f1"] < 1.0
print(scores)
```

如果进一步接近真实训练流程，损失函数一般至少包含两部分。下面是更接近工程直觉的伪代码：

```python
# shared encoder
hidden_states = encoder(passage_tokens)

# long answer head
long_logits = long_head(hidden_states)
long_loss = cross_entropy(long_logits, long_targets)

# short answer head
short_start_logits, short_end_logits = short_head(hidden_states)
short_span_loss = (
    cross_entropy(short_start_logits, short_start_positions) +
    cross_entropy(short_end_logits, short_end_positions)
)

# yes/no/null classification
answer_type_logits = type_head(hidden_states[:, 0])
type_loss = cross_entropy(answer_type_logits, answer_type_targets)

total_loss = long_loss + short_span_loss + type_loss
```

真实工程里，检索部分可以先用 BEIR 中的 NQ slice 做 passage ranking 验证。原因很直接：如果 gold paragraph 根本召回不出来，阅读器再强也无从发挥。很多团队一开始把所有问题都归咎于 reader，其实瓶颈常在 retriever。

---

## 工程权衡与常见坑

NQ 最容易踩的坑，不在模型结构多复杂，而在任务边界理解错误。

| 坑 | 后果 | 规避 |
| --- | --- | --- |
| 忽略 null 样本 | 模型学会逢页必答，误报大量增加 | 保留 null 样本，并为 null 单独建类别 |
| 只训练 short span | long answer 无法稳定定位 | 共享编码器，额外设计 long head |
| 把整页直接送入阅读器 | 上下文过长，噪声极高 | 先检索或切分 passage，再做阅读 |
| 用单一 EM 指标看全部表现 | 看不出是检索坏还是抽取坏 | long/short 分别统计 F1 |
| 训练时不做 yes/no 建模 | 布尔问题被硬转成 span | 增加 answer type 分类头 |

先说 null。null 可以白话理解为“这页里没有可接受答案”。如果训练集中把 null 样本删掉，模型会形成错误先验：只要给它一页文本，它就应该吐一个实体出来。生产环境中，这种模型看起来“很积极”，实际上很危险。

再说 long/short 分离。很多初学者会想：“既然 short answer 最终才是用户看到的，为什么不只训练 short？”问题是 short 的正确性依赖上下文是否正确。只训练 short，模型可能在错误段落中也抽出一个格式上很像答案的片段，比如日期、数字或人名，结果文本看着合理，语义却完全错位。

真实工程例子是企业搜索问答。用户问“离职补偿金上限是多少”。检索器召回了五个 HR 文档，其中有一篇谈“绩效奖金上限”。如果 reader 没有 long answer 层面的边界意识，就可能直接从错误文档里抽出某个百分比，输出得像模像样。NQ 的 long answer 监督，本质上是在教系统先证明“你找对了地方”，再给最终答案。

另一个权衡是流水线与端到端。流水线更可控，容易定位问题：检索没召回，还是阅读没抽对，一眼能拆开看。端到端模型可能在论文上更优雅，但在真实迭代里，调试成本往往更高。对初级工程师来说，先把检索召回率、long answer 定位准确率和 short answer F1 分开做监控，通常比追求一个统一大模型更务实。

---

## 替代方案与适用边界

NQ 不是唯一的问答基准，它解决的是“真实查询 + 页面级上下文 + 可能无答案”的问题。如果你的任务边界不同，未必要直接上 NQ。

先看一个简单对比：

| 数据集 | 查询来源 | 是否有 null | 是否有 long answer | 适合什么任务 |
| --- | --- | --- | --- | --- |
| NQ | 真实搜索查询 | 有 | 有 | 检索 + 阅读的开放域 QA |
| SQuAD | 人工构造问题 | 部分版本有 | 无 | 单段抽取、快速验证 reader |
| BEIR NQ slice | 检索评测构造 | 以检索为主 | 通常不强调 | passage ranking、召回评估 |

如果你是新手，目标只是验证一个 span extraction 模型能不能在给定段落上抽出答案，那么 SQuAD 往往更简单。因为它通常已经给定上下文，任务更像“在这段里找答案”，而不是“先找段落，再找答案”。

如果你要搭一条真正的 end-to-end QA pipeline，尤其是“检索器 + 阅读器”的两阶段系统，NQ 更合适。因为它把真实查询噪声、长短答案分层和 null 决策都带进来了。

如果你的重点不在抽答案，而在验证 passage ranking，例如比较 BM25、双塔检索器和 reranker 谁更容易把 gold paragraph 召回到 top-k，那可以先使用 BEIR 中的 NQ slice。这样你能把问题隔离在检索阶段，不必一开始就把 reader 也卷进来。

因此，选择标准不是“哪个数据集更有名”，而是“你的系统现在缺哪一环”。

- 想快速验证 reader：优先 SQuAD
- 想验证检索质量：优先 BEIR NQ slice
- 想做接近真实搜索问答的全链路：优先 NQ

---

## 参考资料

1. Natural Questions 论文与 Google Research 页面：定义、数据规模、long/short answer、评估指标与 human upper bound  
   https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/

2. 官方 GitHub 仓库：数据结构、标注格式、baseline 工具、null 处理说明  
   https://github.com/google-research-datasets/natural-questions

3. BEIR 中的 NQ slice：用于 passage ranking 的检索评测切片  
   https://huggingface.co/datasets/orgrctera/beir_nq
