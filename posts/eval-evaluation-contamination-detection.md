## 核心结论

评测污染检测的目标，不是证明模型“作弊”，而是判断某道评测题的高分到底来自泛化能力，还是来自训练时见过原题或近似题的记忆。这里的“污染”指评测样本进入了训练语料，或者被改写、翻译、模板化后以近重复形式进入训练语料。

设评测集为 $D_{\text{eval}}$，训练集为 $D_{\text{train}}$，污染集为 $D_{\text{contaminated}}$，则最常见的量化方式是：

$$
D_{\text{contaminated}} = D_{\text{eval}} \cap D_{\text{train}}, \quad
r=\frac{|D_{\text{contaminated}}|}{|D_{\text{total}}|}
$$

其中 $r$ 是污染率，表示评测集中有多少样本可能被记忆影响。玩具例子：一个 benchmark 有 1000 道题，如果检测出 25 道题在训练里出现过或近似出现，那么污染率就是 $25/1000=2.5\%$。这不意味着只有 2.5% 的结论失真，但至少说明有 2.5% 的分数不能直接解释成泛化。

单一方法通常不够。`n-gram` 重叠适合抓逐字复制，`嵌入` 相似度适合抓语义改写，`生成式复述匹配` 适合抓模板变化，`loss` 与 `成员推断` 只能提供间接证据。工程上更可靠的做法是多阶段交叉验证，而不是迷信某一个指标。

| 方法 | 抓到的对象 | 输出类型 | 主要优点 | 主要风险 |
| --- | --- | --- | --- | --- |
| n-gram 重叠 | 逐字或近逐字重复 | 直接匹配证据 | 精度高，解释强 | 抓不到复述 |
| 近重复搜索 / MinHash | 局部改写、顺序小变动 | 近重复候选 | 可扩展到大语料 | 阈值敏感 |
| 嵌入相似度 | 语义近似题 | 语义匹配分数 | 能抓复述 | 容易把同主题新题误报 |
| 生成式复述匹配 | 模板改写、翻译改写 | 复述一致性证据 | 对软污染更敏感 | 成本高、可重复性一般 |
| loss / 成员推断 | 训练见过后的统计异常 | 间接异常信号 | 无需完整训练集时仍可用 | 分布偏移下易失效 |

---

## 问题定义与边界

评测污染检测先要把对象说清楚。评测集 $D_{\text{eval}}$ 是拿来打分的题，训练集 $D_{\text{train}}$ 是模型训练时读过的文本。我们真正关心的不是“有没有相似知识”，而是“这道评测题是否以足够接近的形式提前暴露给模型”。

这里通常分两类：

| 类型 | 定义 | 常见判定条件 | 风险 |
| --- | --- | --- | --- |
| 硬污染 | 原题或大段文本直接出现 | 长 n-gram 命中、近重复文档命中 | 漏掉改写题 |
| 软污染 | 语义相同但表面形式不同 | 高语义相似、复述后仍可对齐 | 容易和领域共通知识混淆 |

“领域共通知识”是最大边界问题。它的白话解释是：某些表达本来就在很多地方都会出现，不代表抄到了测试题。比如 “Translate into English” 这种提示模板，或者“二分查找时间复杂度是多少”这类高度常见问题，它们可以出现在很多独立来源里。若只因为局部短语重叠就判定污染，误报会非常高。

因此污染判定不能等同于“有相似词”。更合理的边界是：

1. 是否存在长片段共享，且共享部分足以唯一定位到具体题目。
2. 是否存在题干、选项、答案结构同时相近。
3. 是否存在跨语言复述、同模板变量替换、代码变量改名等软变化。
4. 是否能排除公开常识、教材定义、通用指令模板。

新手可用一个更直观的比喻理解：训练集像旧作业，评测集像新考试。你不是在查“知识点有没有学过”，而是在查“考试题是不是从旧作业原样抄来，或者只换了几个字”。

---

## 核心机制与推导

第一层通常是 n-gram 检查。`n-gram` 的白话解释是“把文本切成连续的 n 个 token 或字符片段”。若评测题的多个长 n-gram 在训练集中都能找到，说明它很可能不是偶然相似，而是实际暴露过。

设评测样本 $x$ 的 n-gram 集合为 $G_n(x)$，训练集的 n-gram 索引为 $G_n(D_{\text{train}})$，则可定义硬污染重叠率：

$$
s_{\text{ngram}}(x)=\frac{|G_n(x)\cap G_n(D_{\text{train}})|}{|G_n(x)|}
$$

当 $n$ 足够大且 $s_{\text{ngram}}$ 很高时，硬污染证据就很强。工程里常见经验是对长片段更信任，对短片段更保守。

第二层通常是嵌入相似度。`嵌入` 的白话解释是“把一句话压成一个向量，再用向量距离衡量语义是否接近”。若两段文本表面不同，但余弦相似度很高，就可能是复述关系：

$$
\mathrm{sim}(u,v)=\frac{u\cdot v}{\|u\|\|v\|}
$$

但高相似不等于污染。因为“如何反转链表”和“单链表反转怎么写”在语义上本来就接近，它们可能是独立写出的同类题，而不是训练泄露。所以嵌入检索更适合当候选召回器，不适合作为唯一判决器。

第三层是生成式复述匹配。它的思路是：让模型或规则系统把题目改写成标准表达，再去检索训练语料，看是否能找到结构和答案都一致的样本。它擅长抓翻译污染、模板变量替换、措辞重写等软污染。

第四层是统计异常。`loss` 的白话解释是“模型在这道题上犯错有多严重”；`成员推断` 的白话解释是“根据模型输出特征，猜这条数据是不是训练里见过”。如果某些题在 loss 上异常低，或成员推断置信度异常高，它们可能是见过的样本。但这类证据是间接的。已有综述指出，在真实预训练语料和分布偏移场景里，这类方法可能接近随机猜测，因此不能脱离直接匹配证据单独使用。

可以把整个流程理解成一条漏斗：

`n-gram 精确命中` → `嵌入召回近邻` → `复述/翻译匹配` → `loss 或 MIA 作为旁证` → `人工复核`

玩具例子：评测题是“给定整数数组，返回和为 target 的两个下标”。训练集中没有完全相同文本，但出现了“在列表中找出两个元素，使其和等于指定值，并输出索引”。这时 n-gram 可能只命中少量短片段，嵌入相似度会很高，生成式复述后两者结构完全对齐，于是它会被归入软污染候选，而不是干净样本。

真实工程例子：在 MMLU、GSM8K、HumanEval 这类广泛传播的 benchmark 上，研究者发现只做字符串去重是不够的。题目被改写、翻译、变量替换后，模型仍可能从训练中获益，导致小模型在公开分数上接近更强模型。这说明“没查到原文”不等于“没有污染”。

---

## 代码实现

下面给一个最小可运行实现。它不依赖外部库，功能是：

1. 规范化文本。
2. 计算字符级 n-gram 重叠。
3. 用简单词袋向量近似嵌入相似度。
4. 只有当 n-gram 命中和语义相似同时超过阈值时，才标记为高风险候选。

```python
import math
import re
from collections import Counter

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def char_ngrams(text: str, n: int = 13):
    text = normalize(text)
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def ngram_overlap(eval_text: str, train_docs, n: int = 13) -> float:
    eval_grams = char_ngrams(eval_text, n)
    if not eval_grams:
        return 0.0
    train_grams = set()
    for doc in train_docs:
        train_grams |= char_ngrams(doc, n)
    return len(eval_grams & train_grams) / len(eval_grams)

def bow(text: str):
    tokens = re.findall(r"[a-z0-9_]+", normalize(text))
    return Counter(tokens)

def cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def max_semantic_sim(eval_text: str, train_docs) -> float:
    e = bow(eval_text)
    return max(cosine(e, bow(doc)) for doc in train_docs)

def detect(eval_text: str, train_docs, n=13, overlap_t=0.20, sim_t=0.55):
    overlap = ngram_overlap(eval_text, train_docs, n=n)
    sim = max_semantic_sim(eval_text, train_docs)
    flagged = overlap >= overlap_t or (overlap >= 0.05 and sim >= sim_t)
    return {"overlap": round(overlap, 4), "sim": round(sim, 4), "flagged": flagged}

train_docs = [
    "Given an integer array, return indices of the two numbers such that they add up to target.",
    "Translate into English: 你好，世界。",
    "Implement factorial using recursion."
]

eval_exact = "Given an integer array, return indices of the two numbers such that they add up to target."
eval_paraphrase = "Find two positions in an integer list whose values sum to the target."
eval_clean = "Explain why binary search needs sorted input."

r1 = detect(eval_exact, train_docs, n=13)
r2 = detect(eval_paraphrase, train_docs, n=13)
r3 = detect(eval_clean, train_docs, n=13)

assert r1["flagged"] is True
assert r2["sim"] > r3["sim"]
assert r3["flagged"] is False
print(r1, r2, r3)
```

这个实现故意保守。因为真实工程里最怕的是把“同领域正常相似”误报成污染。更常见的生产化流水线如下：

| 输入 | 处理 | 输出 |
| --- | --- | --- |
| 评测文本 | 规范化、切分、去模板噪声 | 待检索样本 |
| 训练语料 | 建 n-gram 哈希索引、MinHash/向量索引 | 可查询索引 |
| 候选样本对 | n-gram 匹配 + 相似度召回 | 污染候选集 |
| 高风险候选 | 复述比对、答案结构对齐 | 已确认/待人工复核 |
| 统计日志 | loss、置信区间、成员推断结果 | 间接旁证 |

如果数据规模很大，通常做法不是对每个评测样本扫描全量训练集，而是先用哈希或倒排索引做粗召回，再用向量检索和人工规则做精排。

---

## 工程权衡与常见坑

核心权衡是精度、召回和误报成本。`精度` 的白话解释是“标出来的里面到底有多少是真的”，`召回` 的白话解释是“真正污染的题有多少被抓到”。n-gram 方法精度高但召回有限，嵌入方法召回高但误报更多，统计异常成本低但解释性差。

| 误判来源 | 为什么会出错 | 对策 |
| --- | --- | --- |
| 通用提示模板 | 很多任务共享固定短语 | 去模板、只看长片段 |
| 同主题独立出题 | 语义相似但并非泄露 | 结合答案结构与上下文 |
| 翻译污染 | 英文原题被翻成别的语言 | 做跨语言检索 |
| 变量替换 | 代码或数学题只换变量名 | 结构标准化、AST 或模板归一 |
| 分布偏移 | MIA/loss 在新分布上失真 | 只把统计法当旁证 |
| 阈值写死 | 不同 benchmark 分布不同 | 分任务校准阈值 |

一个典型坑是只用 loss 或成员推断。表面上看，模型在某些题上特别自信、loss 特别低，似乎说明“见过”。但如果这些题本来就是格式规整、训练分布里大量存在的常见模式，模型也会显得很熟。此时如果没有直接匹配证据，结论就很弱。

另一个坑是把“知识重复”当成“题目污染”。例如问“快排平均时间复杂度是多少”，很多教材、博客、题库都会写到同一个答案。模型答对这题，不足以证明它见过某个 benchmark 的原题。真正值得警惕的是：题干组织、选项排列、错误选项、解释顺序都非常接近。

阈值策略通常比单指标硬判定更稳。一个简单规则可以写成：

```python
def final_decision(ngram_hits, overlap_ratio, semantic_sim, answer_match):
    return (
        ngram_hits >= 3 and overlap_ratio >= 0.15
    ) or (
        overlap_ratio >= 0.05 and semantic_sim >= 0.85 and answer_match
    )

assert final_decision(4, 0.18, 0.40, False) is True
assert final_decision(1, 0.03, 0.90, True) is False
assert final_decision(2, 0.06, 0.91, True) is True
```

这里的意思是：硬证据足够强时直接判；硬证据一般时，必须叠加高语义相似和答案结构一致，才进入污染集。

---

## 替代方案与适用边界

如果你拿不到完整训练语料，或者项目还处在训练前的数据治理阶段，可以采用更轻量的替代方案。

| 方案 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| 公共 benchmark 名单拦截 | 训练前数据清洗 | 实现简单 | 只能防已知 benchmark |
| 去重后的问答对筛查 | 中小规模语料 | 成本低 | 抓不到深度复述 |
| MinHash/近重复搜索 | 大规模语料 | 扩展性好 | 仍偏表层 |
| 动态 benchmark | 高价值长期评测 | 根源上减轻静态污染 | 设计与维护成本高 |
| 私有测试集轮换 | 企业内部评测 | 保密性好 | 难以外部复现 |

可把替代流程写成：

`公开 benchmark 黑名单过滤` → `训练语料去重` → `评测前输入检查` → `高风险样本人工复核`

它们各有边界。硬污染检测更适合后期评测清洗，因为此时目标明确，就是找“这道题有没有直接进训练”。软污染检测更适合预训练数据准备阶段，因为那时还有机会把复述、翻译、模板变体一并清掉。若团队只有 API 访问权，没有训练日志和训练语料，最好把结论写成“污染风险升高”而不是“已确认污染”。

真实工程里还有一个更激进的方向：动态 benchmark。它的核心思想不是继续和旧题做猫鼠游戏，而是每次评测动态生成新变量、新数值、新组合，让静态泄露失去价值。这类方法不能替代污染检测，但能减少污染对结论的破坏。

---

## 参考资料

| 来源 | 作用 |
| --- | --- |
| [Fu et al., 2025, Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions](https://aclanthology.org/2025.findings-naacl.291/) | 定义评测污染检测问题，综述检测假设，并指出部分成员推断方法在分布偏移下可能接近随机猜测 |
| [Emergent Mind: AI-Generated Data Contamination](https://www.emergentmind.com/topics/ai-generated-data-contamination) | 提供污染率、检测方法分类与工程管线整理 |
| [Emergent Mind: Soft Contamination in Language Model Benchmarks](https://www.emergentmind.com/topics/soft-contamination-in-language-model-benchmarks) | 说明软污染、翻译污染、代码污染对 benchmark 分数的影响 |
| [Yang et al., 2023, Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://huggingface.co/papers/2311.04850) | 说明仅靠字符串去重不足，复述样本会绕过传统去污染方法 |
| [Qian et al., 2024, VarBench: Robust Language Model Benchmarking Through Dynamic Variable Perturbation](https://aclanthology.org/2024.findings-emnlp.946/) | 作为动态 benchmark 替代思路，说明如何降低静态评测受污染的风险 |
| [Michael Brenndoerfer, Benchmark Contamination in LLMs: Detection & Mitigation Strategies](https://mbrenndoerfer.com/writing/benchmark-contamination-llm-detection-mitigation) | 用工程视角解释 n-gram、MinHash、嵌入相似度等方法的实际取舍 |
