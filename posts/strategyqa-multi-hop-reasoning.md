## 核心结论

StrategyQA 不是“看一句问题，猜 yes/no”的数据集，而是“先补全隐含推理策略，再验证每一步证据”的数据集。更准确地说，官方论文给出的总规模是 **2,780** 题，其中 **2,290** 题训练、**490** 题测试；问题平均只有 **9.6** 个词，但训练集平均分解长度约 **2.93** 步，平均约 **2.33** 个证据段落。这说明它的难点不在长文本阅读，而在于把短问题还原成完整推理链。

“策略”可以先用一句白话理解：**策略就是把一个看起来很短的问题，拆成几步能查、能判、能组合的小问题**。如果模型不显式学这个东西，就很容易出现“看见关键词就抢答”的错误。

最经典的玩具例子是：

- 问题：`Did Aristotle use a laptop?`
- 正确策略：
  1. 确认 Aristotle 生活年代：公元前 384–322 年。
  2. 确认 laptop 出现年代：现代便携式笔记本电脑要到 20 世纪后期才出现。
  3. 比较时间线：人物早于技术出现很多年。
  4. 结论：`no`。

这个例子有价值，不是因为它难，而是因为它把 StrategyQA 的本质暴露得很清楚：**先定位时间线，再做事件比对**。模型如果跳过这两个动作，只靠 “Aristotle + laptop” 的表面共现，答案没有任何可靠性。

下面这个表格可以先把样本结构看清楚：

| 字段 | 含义 | 新手理解 |
| --- | --- | --- |
| `q` | 原始 yes/no 问题 | 用户真正问的一句话 |
| `D` | decomposition，分解步骤 | 把问题拆成若干可执行子问题 |
| `E` | evidence，每步证据 | 支撑每个子问题的维基段落 |
| `a` | 最终答案 | `yes` 或 `no` |

显式建模“策略”的原因很简单：**最终答案只有 1 bit 信息，而分解步骤暴露了模型是否真的会推理**。只看 yes/no，很多跳步错误会被掩盖；把 `D` 和 `E` 拉出来，错误就能被定位到具体一步。

---

## 问题定义与边界

一个 StrategyQA 样本可以写成四元组：

$$
(q, D, E, a)
$$

其中：

- $q$ 是问题；
- $D = \{d_1, d_2, \dots, d_k\}$ 是推理步骤；
- $E = \{e_1, e_2, \dots, e_k\}$ 是与步骤对应的证据；
- $a \in \{\text{yes}, \text{no}\}$ 是最终标签。

这里的“多跳”可以先用白话理解：**答案不是从一个事实直接读出来，而是要跨多个事实再合并**。StrategyQA 的特殊点在于，这些跳数通常**没有写在问题里**。问题很短，但推理链是隐含的。

可以把它画成一个最小流程：

`问题 q`  
→ `生成步骤 d1, d2, ..., dk`  
→ `为每一步找到证据 e1, e2, ..., ek`  
→ `执行比较/归纳/逻辑组合`  
→ `输出 yes/no`

继续看 Aristotle 的例子，这个流程就更具体：

`Did Aristotle use a laptop?`  
→ `人物生活年代？`、`笔记本何时出现？`  
→ `Aristotle 相关段落`、`laptop 历史段落`  
→ `时间是否重叠？`  
→ `no`

它的边界也要讲清楚。

| 维度 | StrategyQA 擅长评测什么 | 不擅长评测什么 |
| --- | --- | --- |
| 输出形式 | yes/no 决策 | 长答案生成 |
| 监督信号 | 分解步骤 + 证据 | 纯自由文本解释质量 |
| 推理类型 | 隐式多跳、常识与事实组合 | 超长链规划、复杂程序执行 |
| 工程用途 | 检查是否跳步、检索是否支持推理 | 评估文风、摘要质量 |

所以它更像一个**策略推理基准**，而不是百科问答全集。你用它最合适的场景，是检验系统是否真的会“拆问题并验证”，而不是检验系统能不能写一段自然语言解释。

---

## 核心机制与推导

StrategyQA 的核心机制可以概括成一句话：

$$
q \Rightarrow D \Rightarrow E \Rightarrow a
$$

意思是：先从问题推出策略，再用证据验证策略，最后再得出答案。这里的关键不是最后的 `a`，而是中间的 `D` 与 `E`。

“证据”也要先白话解释一下：**证据不是模型脑中的常识，而是外部文本里可以被检查的支撑材料**。有了证据，推理才可复核；没有证据，系统只是在猜。

还是看玩具例子，把步骤、证据、局部结论对齐：

| 步骤 | 对应 evidence | 局部结论 |
| --- | --- | --- |
| `d1`: Aristotle 生活在什么年代？ | Aristotle 的维基百科人物简介 | 公元前 384–322 年 |
| `d2`: Laptop 在什么年代出现？ | Laptop 历史或便携计算机相关段落 | 现代技术，远晚于古希腊 |
| `d3`: 两个时间段是否重叠？ | 对 `d1`、`d2` 的结果做比较 | 不重叠 |
| 最终答案 | 汇总 `d3` | `no` |

这个结构直接逼着模型做两件事：

1. **推导策略**：把短问题还原成隐藏子问题。
2. **验证策略**：为每个子问题找可检查证据。

这也是它比很多“显式多跳问答”更刁钻的地方。很多别的数据集把跳数写在问题里，模型顺着问句结构走就行；StrategyQA 常常要先猜出“该怎么问自己”。

官方论文的数据采集流程也在降低捷径。所谓“术语激发”，可以先白话理解为：**标注者不是照着长段落改写问题，而是被给到一些词语提示，自行构造问题**。这样做的好处是减少问题和证据之间的词面重叠，避免模型靠表面词匹配取巧。再加上对抗式过滤，能把太容易、能被浅层模式秒杀的问题筛掉一部分。

从工程角度，可以把总体推理流程写成伪代码：

```text
input: question q
D_hat = generate_decomposition(q)
for each step d_i in D_hat:
    E_hat_i = retrieve_evidence(d_i)
    s_i = verify(d_i, E_hat_i)
a_hat = aggregate(s_1, s_2, ..., s_k)
return a_hat
```

这里 `aggregate` 往往不是复杂神经网络，很多时候就是比较、包含、计数、先后顺序判断等离散操作。也正因为这样，StrategyQA 很适合暴露“模型语言看起来通顺，但逻辑上漏一步”的问题。

---

## 代码实现

工程里最常见的做法不是端到端直接答 yes/no，而是两阶段：

1. 先生成分解步骤；
2. 再按步骤检索和验证证据。

“few-shot”可以先白话理解：**先给模型看少量示例，再让它模仿这个格式工作**。在 StrategyQA 上，这通常比一句“请逐步思考”更稳定，因为它明确规定了输出应包含哪些步骤。

下面给一个可运行的 Python 玩具实现。它不追求覆盖真实数据集，只演示“时间线比较”这类最小策略如何写成程序，并用 `assert` 固定行为。

```python
import re

def normalize_year(text: str) -> int:
    text = text.strip().upper()
    m = re.search(r'(\d+)\s*(BCE|BC|CE|AD)?', text)
    assert m, f"cannot parse year from: {text}"
    year = int(m.group(1))
    era = m.group(2)
    if era in {"BCE", "BC"}:
        return -year
    return year

def answer_aristotle_laptop(person_start: str, person_end: str, laptop_start: str) -> str:
    start = normalize_year(person_start)
    end = normalize_year(person_end)
    laptop = normalize_year(laptop_start)

    assert start <= end, "person timeline is invalid"
    # 如果技术出现时间晚于人物去世时间，则人物不可能使用该技术
    if laptop > end:
        return "no"
    return "yes"

# 玩具例子
assert answer_aristotle_laptop("384 BCE", "322 BCE", "1970 CE") == "no"
assert answer_aristotle_laptop("1980 CE", "2020 CE", "1970 CE") == "yes"
```

真实工程例子会更像下面这样：

| Prompt 组成 | 作用 |
| --- | --- |
| 2-5 个 few-shot 示例 | 教模型输出分解步骤，而不是直接抢答 |
| 当前问题 `q` | 待推理对象 |
| 期望格式 | 约束模型输出 `steps -> evidence needs -> final answer` |
| 可选 verifier 指令 | 要求逐步检查每一步是否被证据支持 |

一个常见 pipeline 可以写成：

```text
few-shot examples
+ current question q
-> model generates D_hat

for d_i in D_hat:
    retrieve top-k paragraphs for d_i
    ask model or verifier:
        "Does this paragraph support step d_i?"
collect supported steps
-> final yes/no
```

如果你要做训练，可以把标签拆成两层：

- `step loss`：分解步骤是否接近人工标注；
- `answer loss`：最终 yes/no 是否正确。

如果你只做评测，也建议至少加一个一致性检查。例如步骤里前面说“Aristotle lived in BCE”，后面又把比较逻辑写成“technology existed before Aristotle”，这种前后不一致，即使最终凑巧答对，也应记为推理质量差。

---

## 工程权衡与常见坑

最常见的坑，是把 StrategyQA 当成普通 BoolQ 一样做“直接分类”。这会得到一个表面准确率，但你并不知道系统是不是靠策略推理得到的。

下面是最典型的错误对比：

| 方案 | 推理过程 | 优点 | 问题 |
| --- | --- | --- | --- |
| 无 evidence 直答 | 直接 `q -> yes/no` | 快，便宜 | 极易词汇匹配、不可审计 |
| 有 evidence 串联 | `q -> D -> E -> a` | 可检查、可定位错误 | 更慢，更耗上下文 |

拿 Aristotle 例子说，如果模型只看到 `Aristotle` 和 `laptop` 就输出答案，它可能碰巧答对，但这不代表系统具备泛化能力。换成别的时序问题，比如冷门发明、较少见人物，错误会马上暴露。

第二个坑，是**检索与推理耦合不清**。很多失败不是推理器差，而是第一步就没拿到对的段落。StrategyQA 的问题和证据往往词面重叠不高，所以检索器如果只依赖稀疏关键词，召回会很难看。结果是你误以为模型不会推理，实际上是它没看见关键证据。

第三个坑，是**把步骤生成得过细或过粗**。

- 过细：检索次数暴涨，成本高，上下文占满。
- 过粗：每步包含多个逻辑动作，验证不了是否跳步。

经验上，StrategyQA 很多题落在“约 3 步、约 2 个外部证据”的量级，这个粒度通常比较合适。

第四个坑，是**只看最终准确率，不看步骤一致性**。真实工程里，尤其是做 prompt 对比、RAG 组件对比时，建议额外记录：

- 步骤数是否异常膨胀；
- 每一步是否有证据支撑；
- 最终答案是否与中间结论一致；
- 同一道题多次采样是否稳定。

真实工程例子：如果你在做一个“检索增强问答”系统，想比较 `direct answer`、`CoT prompt`、`decompose+verify` 三种提示策略，StrategyQA 是很好的回归集。因为它既短小，跑得快；又能用 `D/E/a` 三层信号告诉你，问题到底出在检索、分解还是最终归纳。

---

## 替代方案与适用边界

如果你的任务是 yes/no 决策，而且你关心系统是否会“先拆再证”，StrategyQA 很合适。但它不是所有多跳任务的默认首选。

| 数据集 | 输入输出形式 | 核心监督 | 典型用例 | 边界 |
| --- | --- | --- | --- | --- |
| StrategyQA | 短问题 -> yes/no | 分解步骤 + 证据段落 | 评估隐式策略推理 | 不适合长答案生成 |
| HotpotQA | 问题 -> 文本答案 | supporting facts | 多文档事实问答 | 更强调答案抽取 |
| MuSiQue | 问题 -> 文本答案 | 组合式多跳构造 | 检验真实多跳依赖 | 成本更高，任务更重 |

和 HotpotQA 对比最容易理解。HotpotQA 更像“跨两篇或多篇文档找到事实答案”；StrategyQA 更像“先想清楚该问哪些子问题，再决定 yes/no”。前者更偏**事实抽取与支持句定位**，后者更偏**策略显式化与逻辑汇总**。

和 MuSiQue 对比，差别在于 MuSiQue 更强调“每一跳都真的依赖前一跳”，因此更适合测严格的连接式多跳；StrategyQA 则更适合作为低成本、可解释的 yes/no 推理评测集。

如果你的目标不是严格 yes/no，而是让系统生成完整解释，也可以把 StrategyQA 当作底座，扩成 `generate + verify` 流程：

1. 先生成分解步骤；
2. 再为每步找证据；
3. 最后把步骤和证据拼成可读解释。

但要注意，这时你测的已经不只是 StrategyQA 原始任务，而是“解释生成质量”。这和原基准的边界不同。

---

## 参考资料

| 资料 | 主要内容 | 适用场景 |
| --- | --- | --- |
| [Geva et al., 2021, TACL](https://aclanthology.org/2021.tacl-1.21/) | StrategyQA 的官方定义、采集流程、2,780 总样本、2,290/490 划分、问题长度与分解长度统计 | 需要核对基准设计、引用正式论文时 |
| [StrategyQA 官方论文 PDF](https://aclanthology.org/2021.tacl-1.21.pdf) | 表格和图更完整，能直接查到平均 9.6 词、2.93 步、2.33 段证据等细节 | 写技术文章、复核具体数字时 |
| [Hugging Face: voidful/StrategyQA](https://huggingface.co/datasets/voidful/StrategyQA) | 数据集卡片，说明每个训练样本包含 question、answer、decomposition、evidence | 想快速理解样本结构、接入代码时 |
| [alphaXiv StrategyQA 页面](https://www.alphaxiv.org/benchmarks/tel-aviv-university/strategyqa) | 用更直观方式总结任务特征和经典示例，如 Aristotle/laptop | 面向新手做任务介绍时 |
| [MDPI: Chain-of-Thought Prompt Optimization via Adversarial Learning](https://www.mdpi.com/2078-2489/16/12/1092) | 把 StrategyQA 当作 yes/no 推理基准比较不同提示与验证策略 | 做 prompt engineering 或评测框架设计时 |
| [HotpotQA 官方主页](https://hotpotqa.github.io/) | 多文档、多跳、支持句监督的代表性基准 | 需要和 StrategyQA 做任务选型比较时 |
| [MuSiQue, TACL 2022](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996/MuSiQue-Multihop-Questions-via-Single-hop-Question) | 通过组合单跳问题构造更严格的多跳依赖 | 需要更强连接式多跳测试时 |
