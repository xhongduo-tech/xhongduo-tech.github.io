## 核心结论

Agent 评测里的数据污染，本质是“模型在训练阶段已经见过题目、答案、解题路径，评测时像在做开卷回忆”。对零基础读者来说，可以把它理解成：考试前题库和参考答案已经被放进练习册，分数就不再代表真实能力。

这件事已经不是抽象风险。OpenAI 在 2026 年 2 月 23 日公开表示，不再把 SWE-bench Verified 当作前沿编码能力的主要指标，原因有两个：一是测试本身有缺陷，二是基准和相关代码仓库公开传播太广，前沿模型已经表现出对题目细节和 gold patch 的训练暴露迹象。GAIA 也有类似问题。H2O.ai 在 2025 年的公开技术博客里直接承认：GAIA 验证集问题和答案广泛在线可得，他们在运行时主动屏蔽部分泄漏网站后，模型仍会出现“可疑地准确”的猜测，这说明污染不仅可能是背答案，也可能是背搜索路径和中间步骤。

工程上不要把“排行榜分高”直接等价成“Agent 更强”。更可靠的做法是把防泄漏设计前置到评测体系里：用 n-gram 重叠做静态筛查，用 membership inference 做统计检测，用 canary token 做主动探针，再配合动态题库、旧题退役、私有测试集，才可能把“记忆”与“推理”尽量分开。

| 基准/方案 | 主要风险 | 更可靠的处理方向 |
| --- | --- | --- |
| SWE-bench Verified | 公开仓库、issue、PR、release notes 易进入训练语料 | 转向污染更低的 split 或私有题库 |
| GAIA 验证集 | 问题与答案在线可搜，工具链也可能记住搜索路径 | 区分验证/测试集，运行时屏蔽已知泄漏站点 |
| 公开静态 benchmark | 容易被爬虫抓取，长期必然老化 | 动态改写、持续更新、旧题退役 |
| 私有评测 | 保密性更强，但实现复杂 | 可信执行环境、最小暴露接口、审计流程 |

---

## 问题定义与边界

先给定义。数据污染，指评测样本本身，或与它高度等价的题面、答案、补丁、讨论串，已经出现在模型训练或微调语料里。这里的“高度等价”很重要，因为污染不一定是逐字重复，也可能只是换了变量名、换了说法，但核心求解路径没变。

边界要先画清楚，否则团队会把很多不同问题混在一起：

| 类型 | 白话解释 | 是否属于污染 |
| --- | --- | --- |
| 题面原文进入训练集 | 考题原封不动出现在练习册 | 是 |
| 参考答案进入训练集 | 没见过题，但见过标准解法 | 是 |
| issue/PR/论坛讨论进入训练集 | 没见最终答案，但见过关键提示 | 是 |
| 同逻辑不同表述 | 换皮题，但解题结构相同 | 通常也算 |
| 模型现场联网搜到答案 | 考场上查资料，不是训练污染 | 不属于训练污染，但属于评测隔离问题 |

这里要区分两类风险。

第一类是训练污染。模型参数里已经存了信息，哪怕评测时断网，它也可能答对。SWE-bench Verified 的危险就在这里。OpenAI 的公开说明不是“可能有点污染”，而是明确说他们测试的所有前沿模型都表现出至少部分题目和解答曾被见过的迹象。

第二类是运行时泄漏。Agent 在评测时通过搜索、网页、缓存、工具日志拿到了答案。GAIA 的问题更像是两类风险叠加：验证集在线广泛可得，导致既可能被训练记住，也可能被运行时搜到。

玩具例子最容易理解。假设有一道编程题：

“给定一个整数数组，返回出现次数超过一半的元素。”

如果这道题原文和标准解已经在公开博客里传播了很多次，那么模型即使不真正理解 Boyer-Moore 投票算法，也可能靠记忆直接给出正确实现。评测结果会看起来很好，但它没有证明模型能独立发现这个算法。

所以，问题边界不该只定义成“答案字符串是否出现在训练语料”。更合理的工程边界是：只要训练数据中出现了足以显著降低求解难度的题面片段、关键补丁、唯一术语、隐藏测试所依赖的上下文，就应该被视为污染风险。

---

## 核心机制与推导

防泄漏通常不是靠一个指标，而是三层机制叠加。

第一层是 n-gram 重叠检测。n-gram 可以理解成“连续 n 个 token 的切片”。把训练语料和评测题面都切成长度为 $n$ 的片段，如果一段长片段反复重合，说明这题很可能被见过。

一个最简单的判断逻辑是：

1. 把题面切成 10-gram 或 13-gram。
2. 在候选训练语料里查这些片段的命中次数。
3. 如果最长连续命中超过阈值，或者命中频次超过阈值，就标成可疑样本。

玩具例子：某测试题切出 120 个 10-gram，其中有 8 个 10-gram 在训练语料里各出现了 5 次，而团队的阈值设为“同一题任意关键 10-gram 命中次数 $\ge 3$ 就报警”，那这题就该先下线，而不是继续留在榜单里。

第二层是 membership inference，中文常译“成员推断攻击”。白话说，它是在猜“这条文本是不是训练集成员”。常见思路不是直接问模型“你见过吗”，而是看它对样本的拟合程度。若一条样本的平均负对数似然更低，说明模型对它更熟，因而更像训练数据。

常用形式是：

$$
\mathcal{L}(x) = -\frac{1}{L}\sum_{i=1}^{L}\log p_\theta(t_i \mid t_{<i})
$$

这里 $L$ 是 token 数，$t_i$ 是第 $i$ 个 token。$\mathcal{L}$ 越低，样本越可能是模型训练时见过或高度近似见过的内容。它不是法律意义上的“证据”，但在统计上很有用，适合批量打分和排序。

第三层是 canary token 注入。canary 可以理解成“故意放进去的唯一暗号”。做法是向测试样本或训练流程里注入一个极少自然出现的字符串，例如：

`ALPHA-9Q7X-CANARY-2026`

如果模型在后续生成时能高概率复现它，就说明模型可能记住了这个唯一序列。Carlini 等人的工作用 exposure 衡量这种风险，直观理解是：秘密越容易被模型从海量候选中排到前面，暴露程度越高。

这三类方法的关系可以这样理解：

| 方法 | 测什么 | 优点 | 局限 |
| --- | --- | --- | --- |
| n-gram 重叠 | 文本表面重复 | 快、便宜、适合预筛 | 抓不住同义改写 |
| membership inference | 模型对样本的“熟悉度” | 能发现非逐字污染 | 依赖概率接口或代理方法 |
| canary token | 模型是否记住唯一秘密 | 证据强、可主动设计 | 更适合闭环评测或训练审计 |

真实工程例子是 SWE-bench Verified。它的问题不只是“公开”，而是评测样本来自公开代码仓库、issue、PR、release notes 这些强相关上下文。只要训练语料收集覆盖到这些位置，模型就可能不靠真实推理，而是靠记忆恢复补丁细节。OpenAI 在 2026 年 2 月 23 日的公告里给出的具体例子，是模型能说出题面未明确要求、但 release notes 里才有的信息。这类行为就是典型的训练暴露信号。

另一个真实工程例子是 GAIA。H2O.ai 的公开描述表明，他们即使在运行时屏蔽部分已知泄漏网站，底层 LLM 仍会给出可疑地准确的中间步骤猜测。这意味着污染不只体现在最终答案上，也会体现在“如何搜、搜什么、下一步去哪找”这些策略层面。对 Agent 评测来说，这很关键，因为 Agent 不只是输出答案，它还输出行动路径。

---

## 代码实现

下面给一个最小可运行的污染检测脚本。它不是生产级方案，但足够说明管线核心：先做 token 级切片，再算重叠率，并结合命中次数做报警。

```python
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable


def normalize(text: str) -> list[str]:
    # 统一小写并去掉大部分标点，减少表面格式差异
    return re.findall(r"[a-z0-9_]+", text.lower())


def generate_shingles(text: str, k: int = 5) -> list[str]:
    tokens = normalize(text)
    if len(tokens) < k:
        return []
    return [
        hashlib.md5(" ".join(tokens[i:i + k]).encode("utf-8")).hexdigest()
        for i in range(len(tokens) - k + 1)
    ]


def jaccard_score(sample: str, corpus: str, k: int = 5) -> float:
    a = set(generate_shingles(sample, k))
    b = set(generate_shingles(corpus, k))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def repeated_hits(sample: str, corpus: str, k: int = 5) -> int:
    sample_shingles = generate_shingles(sample, k)
    corpus_counts = Counter(generate_shingles(corpus, k))
    return sum(1 for s in sample_shingles if corpus_counts[s] >= 2)


def avg_nll(probs: Iterable[float]) -> float:
    probs = list(probs)
    assert probs and all(0.0 < p <= 1.0 for p in probs)
    return -sum(math.log(p) for p in probs) / len(probs)


toy_question = "Write a function majority_element that returns the element appearing more than half the time."
leaked_corpus = """
In this tutorial, write a function majority_element that returns the element appearing
more than half the time. The standard solution uses Boyer Moore voting.
"""
clean_corpus = "This page explains binary search trees and hash tables."

score_leaked = jaccard_score(toy_question, leaked_corpus, k=4)
score_clean = jaccard_score(toy_question, clean_corpus, k=4)
hits_leaked = repeated_hits(toy_question, leaked_corpus + leaked_corpus, k=4)

assert score_leaked > score_clean
assert hits_leaked >= 1

member_like_nll = avg_nll([0.9, 0.8, 0.95, 0.85])
non_member_nll = avg_nll([0.2, 0.4, 0.3, 0.25])

assert member_like_nll < non_member_nll

print("leaked score:", round(score_leaked, 4))
print("clean score:", round(score_clean, 4))
print("repeated hits:", hits_leaked)
print("member-like nll:", round(member_like_nll, 4))
print("non-member nll:", round(non_member_nll, 4))
```

这段代码体现了三件事。

第一，`jaccard_score` 适合做批量预筛。评测集每新增一题，先和候选训练快照、已知公开网页、论坛归档做重叠比对。分数过高就直接进入人工复核。

第二，`repeated_hits` 补足了 Jaccard 的一个缺点。某些题目虽然整体改写很多，但关键短语被多次重复引用，这种情况单看集合相似度未必高，频次统计反而更敏感。

第三，`avg_nll` 模拟了 membership inference 的核心思想。生产环境通常不会这么简化，而是会做长度归一化、参考模型校准、采样伪似然等处理，但原则没变：模型越“轻松”地预测一段文本，这段文本越可疑。

如果把它放进真实评测流水线，通常会变成下面的顺序：

| 阶段 | 动作 | 输出 |
| --- | --- | --- |
| 入库前 | 题面去重、n-gram 扫描、网页回溯 | 可疑题单 |
| 评测前 | 对可疑题跑 membership inference 或代理检测 | 风险分层 |
| 评测时 | 屏蔽已知泄漏站点、记录工具轨迹 | 运行时审计日志 |
| 评测后 | 对高分样本做 canary/人工复核 | 退役或保留决定 |

一个更贴近工程的例子是代码类 benchmark。你准备评估一个修 bug Agent，题目来自开源仓库 issue。那就不能只扫描 issue 文本本身，还要把对应 PR、commit message、release notes、Stack Overflow 引用页一并纳入候选污染源。因为对代码 Agent 来说，真正泄漏价值最高的往往不是题面，而是补丁周边的讨论语境。

---

## 工程权衡与常见坑

第一类常见坑，是把“公开但没人刻意训练”误判成“安全”。这不成立。只要 benchmark 长期公开，爬虫、镜像站、论坛转载、教程二创就会不断扩散。今天干净，不代表三个月后还干净。

第二类坑，是只查答案，不查中间材料。真实泄漏往往出现在 issue 评论、代码 review、FAQ、博客教程、release notes。模型不一定记住最终答案，但记住一个唯一 API 名、一个隐藏参数、一个关键报错，就足以把题目从“推理”变成“检索式回忆”。

第三类坑，是只防训练污染，不防运行时泄漏。Agent 评测比纯文本问答更脆弱，因为 Agent 会用搜索、浏览器、文件系统、缓存、工具调用。GAIA 的经验说明，哪怕屏蔽了已知答案站点，模型仍可能借助训练中形成的搜索先验做出异常准确的步骤猜测。

第四类坑，是动态题库做得不够彻底。动态评测不是简单改改变量名。若核心实体名、唯一数字、函数签名、图片布局、网页标题都没变，模型仍可能识别“这是那道题”。DyCodeEval 的价值就在于它强调“保持核心逻辑不变，但系统化改变上下文表达”，也就是让题的语义目标保留，而表面可记忆特征尽量被打散。

第五类坑，是把私有评测神化。私有评测并不自动等于高质量。TRUCE 这类 private benchmarking 方案解决的是“不给模型直接看到测试集”，但它还需要回答另外两个问题：题目本身够不够好，执行环境是否可信。若私有题库设计差、评分器有漏洞，最终仍然会误判模型能力。

工程上常见的权衡是：

| 方案 | 好处 | 成本 |
| --- | --- | --- |
| 公开静态集 | 社区容易复现，榜单传播快 | 污染速度也最快 |
| 动态改写集 | 能削弱文本记忆 | 需要生成、验证、去重体系 |
| 连续抽样+退役 | 长期更稳 | 需要大题库和题目生命周期管理 |
| 私有评测 | 泄漏面最小 | 需要权限、基础设施、审计 |

一个实用原则是：把 benchmark 当成“会腐烂的资产”。它不是一次发布后永久有效，而是需要版本化、退役、替换、再校准。很多团队的问题不是不会做评测，而是不承认评测也有寿命。

---

## 替代方案与适用边界

如果你的目标是长期监控 Agent 能力，而不是只做一次论文实验，替代方案通常比传统公开静态榜单更重要。

第一类是动态改写。DyCodeEval 属于这一路线。它用多 agent 从种子题出发，提取核心逻辑，再改写上下文和表述，生成语义等价的新题。它适合代码类或规则清晰的问题，因为“逻辑不变、表述可变”比较容易验证。

第二类是连续抽样和旧题退役。白话说，就是每次评测都从更大的题库里抽新题，跑过的题要降权甚至退役。这样模型即使针对旧榜单过拟合，也难以长期刷分。它适合企业内部持续监控，因为组织通常有权限维护私有题库。

第三类是私有评测。TRUCE 提出的思路是：测试集不直接暴露给模型拥有方，在不同信任假设下，通过可信执行环境或密码学方案完成评测。它适合高价值、高对抗环境，例如模型供应商提交闭源模型、评测方又不愿暴露题库。

第四类是自动更新的防泄漏基准。AntiLeakBench 的核心想法不是“只收集新数据”，而是构造明确晚于模型知识边界、并带有新知识约束的样本，从而减少“虽然是新题，但答案早已是常识”的假干净情况。

这些方案的适用边界可以概括成下面这张表：

| 方案 | 最适用场景 | 不适用场景 |
| --- | --- | --- |
| DyCodeEval 式动态改写 | 代码题、逻辑结构清晰的任务 | 开放式创作、主观题 |
| 连续抽样+退役 | 长周期能力跟踪 | 一次性公开竞赛 |
| 私有评测/TRUCE | 闭源模型、高保密需求 | 社区完全公开复现 |
| AntiLeakBench 式自动更新 | 知识类基准持续维护 | 需要强人工主观评分的任务 |

最后给一个决策建议。若你是小团队，先做“三件套”就够了：公开题库前跑 n-gram 扫描，评测时封堵已知泄漏源，季度性退役旧题。若你是要对外发布 Agent 排行榜的平台，最低要求应该升级到：动态题库或私有测试集，加上污染审计报告。没有这些，排行榜更多是在测“训练暴露程度”和“刷榜技巧”，而不是测 Agent 的真实问题求解能力。

---

## 参考资料

- OpenAI. “Why SWE-bench Verified no longer measures frontier coding capabilities.” https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- H2O.ai. “H2O.ai Tops the General AI Assistant (GAIA) Test.” https://h2o.ai/blog/2025/h2o-ai-tops-the-general-ai-assistant-test
- Chen, Pusarla, Ray. “DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination.” ICML 2025 / PMLR. https://proceedings.mlr.press/v267/chen25ba.html
- Wu et al. “AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge.” ACL 2025. https://aclanthology.org/2025.acl-long.901/
- Chandran et al. “TRUCE: Private Benchmarking to Prevent Contamination and Improve Comparative Evaluation of LLMs.” Microsoft Research / arXiv 2024. https://www.microsoft.com/en-us/research/publication/truce-private-benchmarking-to-prevent-contamination-and-improve-comparative-evaluation-of-llms/
- Carlini et al. “The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks.” https://research.google/pubs/the-secret-sharer-evaluating-and-testing-unintended-memorization-in-neural-networks/
- Kaneko et al. “Sampling-based Pseudo-Likelihood for Membership Inference Attacks.” ACL Findings 2025. https://aclanthology.org/2025.findings-acl.465/
