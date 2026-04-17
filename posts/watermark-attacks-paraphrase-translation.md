## 核心结论

水印攻击的核心不是“删掉某几个特殊符号”，而是在**尽量保留原语义**的前提下，主动破坏检测器依赖的统计结构。这里的“统计结构”可以理解为：检测器希望在一段文本里看到某些更容易由模型生成、而不容易自然出现的分布偏差，例如 green/red token 比例、特定 n-gram 组合、采样路径偏向等。

对很多 token 级水印来说，最便宜、最有效的攻击就是 **paraphrase** 和 **translation**。前者是“同义改写”，意思不变但换词、换句式、换顺序；后者是“跨语言改写”，先翻译到另一种语言，再翻回原语言。它们共同点是：语义保留得较多，但 token 序列几乎被重写。于是，检测器原本依赖的“表层信号”会迅速变弱。

一个常被引用的结果是：DIPPER 这类改写攻击可让 DetectGPT 在 1% FPR 下的识别率从 70.3% 降到 4.6%。这里的 FPR 是“假阳性率”，意思是把人类文本错判为机器文本的比例。这个结果说明，**只要水印主要存在于 token 分布，攻击者就能用语义保持改写把它打散**。

这也解释了为什么后续工作开始把水印锚定到语义层。SemStamp 用句向量加 LSH 分区，SemaMark 用语义替代 hash，目标都不是保住原始 token，而是让“语义相近的改写文本”仍然落在同一类可检测区域内。换句话说，攻防焦点已经从“词长什么样”转向“句子表达的意思落在哪个语义区域”。

| 方案 | 主要信号层级 | 对 paraphrase 抗性 | 实现成本 | 延迟 |
|---|---|---:|---:|---:|
| token-level green/red | token 统计 | 低 | 低 | 低 |
| DetectGPT 类检测 | 文本局部概率曲率 | 低到中 | 中 | 中 |
| 检索式防御 | 语义相似度 | 中到高 | 高 | 中到高 |
| SemStamp | 句子级语义分区 | 高 | 高 | 中到高 |
| SemaMark | 语义增强的 token 打点 | 中到高 | 中 | 低到中 |

---

## 问题定义与边界

所谓“水印攻击”，这里特指一种**保语义、改表层**的攻击：攻击者不追求把文本变成乱码，也不追求删掉所有模型痕迹，而是通过同义改写、翻译、摘要、重采样、后处理等方法，让带水印文本的分布尽量靠近普通自由文本分布。

更形式化地说，设 $P_{X,S}$ 表示“带水印文本”的分布，$P_X$ 表示“普通文本”的分布。攻击者的目标不是一定生成完全不同的话，而是让检测器在这两个分布之间越来越难分。只要分不开，水印就等于失效。

这里有两个边界需要区分：

| 边界 | 关注对象 | 典型变化 | 检测器是否容易失效 |
|---|---|---|---|
| token-level gap | 词、子词、顺序、局部搭配 | 换词、倒装、翻译、压缩 | 容易 |
| semantic-level gap | 句子或段落的核心语义 | 事实关系、逻辑结构、意图 | 较难 |

“gap” 可以理解为“可分辨差距”。如果一个检测器只看 token-level gap，那么攻击者只要把词换掉，哪怕语义不变，也可能让检测结果崩掉。相反，如果检测器能利用 semantic-level gap，那么单纯改写就不一定够。

一个玩具例子最容易说明问题。

原句：
“缓存命中率下降会直接拉高数据库压力，因此需要先定位热点失效原因。”

paraphrase 后：
“当缓存命中变差时，数据库负载通常会上升，所以应优先排查热点数据为何没有被有效缓存。”

translation 链后：
中文 → 英文：
“A drop in cache hit rate directly increases database load, so the cause of hot-key invalidation should be identified first.”
英文 → 中文：
“缓存命中率下降会增加数据库负载，因此应先确认热点键失效的根因。”

这三句话语义几乎一致，但 token、短语边界、句法结构都明显不同。对于 green/red token 水印，检测对象几乎已经不是同一段文本。

真实工程里更常见的是“用户二次编辑”。例如，某内容平台接收一段模型生成的产品说明。发布前，运营人员会先让模型“改成更口语化”，再把中文版翻成英文给海外站点，最后由本地编辑重新润色。最终上线文本保留了原意，但水印若只依赖最初的 token 采样路径，基本已不可恢复。

---

## 核心机制与推导

检测问题可以抽象成一个分布判别问题。一个常见表述是：

$$
\Delta_{\text{gap}} = \max_f \left( \mathbb{E}_{P_{X,S}}[f(X,S)] - \mathbb{E}_{P_X}[f(X)] \right)
$$

这里的 $f$ 可以理解为“判别函数”，也就是检测器真正使用的统计量、得分函数或分类器。$\Delta_{\text{gap}}$ 越大，说明带水印文本与普通文本越容易区分；越小，则说明检测越难。

对 token 级水印来说，$f$ 往往依赖如下信号：

1. 当前上下文下 green token 的出现频率是否异常高。
2. 某些 hash 决定的 token 集合是否被偏置采样。
3. 文本局部概率分布是否有稳定偏移。

paraphrase/translation 的破坏逻辑并不复杂：

1. 先保留语义主干。
2. 再改写词形、语序、搭配和句法。
3. 让原始 token 统计特征不再稳定出现。
4. 结果是 $P_{X,S}$ 在表层分布上向 $P_X$ 靠近，$\Delta_{\text{gap}}$ 变小。

可以把它画成一个简化示意：

$$
P_{X,S}^{\text{token}} \xrightarrow{\text{paraphrase / translation}} P_X^{\text{token}}
$$

而防御者想做的是：

$$
P_{X,S}^{\text{semantic}} \not\approx P_X^{\text{semantic}}
$$

也就是说，表层可以变，但在语义空间中仍保留“这是模型生成且带有可识别约束”的差异。

SemStamp 的思路正是这样。它把句子映射到语义嵌入空间，再用 LSH 做分桶。LSH 是“局部敏感哈希”，白话说法是：语义相近的句子更可能被分到同一个桶里。生成时，系统通过 rejection sampling，也就是“拒绝采样”，不断重采样直到句子落入目标水印桶。这样一来，即便攻击者 later paraphrase，只要新句子的语义仍接近原句，它大概率还在相近语义区域内，检测信号就不会像 token 级水印那样瞬间归零。

SemaMark 走的是另一路：它不完全放弃 token 层，而是把打点规则从“纯 token hash”改成“语义替代 hash”。直观上看，它试图让“可互换的表达”共享部分水印逻辑，因此对翻译和改写更稳。

可以把逻辑链压缩成一句话：

**token 水印依赖表层稳定性，攻击者改写表层即可；语义水印依赖意义空间的一致性，攻击者若继续保语义，就不容易完全甩掉检测。**

---

## 代码实现

工程上，最容易让初学者理解的不是直接实现 SemStamp，而是先实现一个“检索式防御”原型。它不试图从文本本身恢复某个水印 bit，而是问一个更现实的问题：**这段话是否高度像某条历史生成文本的改写版？**

流程通常是：

1. 用句向量模型把候选文本编码成 embedding。
2. 到向量库里检索相似历史生成文本。
3. 如果命中足够高，相当于判定它与已知机生成内容语义接近。
4. 将其标记为“可能是 paraphrased generation”，再触发人工复验或更强检测。

下面是一个可运行的最小 Python 示例。它不是生产级语义编码器，而是用词集合做一个玩具版相似度，目的是把检索防御的逻辑讲清楚。

```python
import math
import re

def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower()))

def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)

def retrieve(candidate: str, corpus: list[str], threshold: float = 0.3):
    hits = []
    for doc in corpus:
        score = jaccard(candidate, doc)
        if score >= threshold:
            hits.append((doc, round(score, 4)))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits

generated_db = [
    "缓存命中率下降会直接拉高数据库压力因此需要先定位热点失效原因",
    "向量检索可以用于发现改写后的历史生成文本",
    "语义水印试图把信号锚定在句子含义而不是单个token"
]

candidate = "当缓存命中变差时数据库负载会上升所以应优先排查热点失效的原因"
hits = retrieve(candidate, generated_db, threshold=0.2)

assert len(hits) >= 1
assert hits[0][1] > 0.2

label = "likely paraphrased generation" if hits else "clean"
print(label)
print(hits[:2])
```

这个例子故意很简单，但结构和真实系统一致：

```python
candidate = llm.generate(prompt)
emb = sentence_encoder(candidate)
hits = db.retrieve(emb, top_k=5, threshold=0.8)
if hits:
    label = "likely paraphrased generation"
else:
    label = "clean"
```

玩具例子里，`sentence_encoder` 被一个粗糙的词集合相似度代替；真实系统里通常会换成专门的句向量模型、ANN 检索库和更严格阈值。

真实工程例子可以这样理解：一个 LLM API 服务商持续保存历史生成文本的 embedding，规模可到千万级。用户提交一段文本后，系统先做向量检索。如果它与某批历史生成结果高度接近，即便 token 完全不同，也可能被判为“对旧生成内容的 paraphrase 或 translation”。这类方案的优点是对改写更稳，缺点是需要维护大规模向量库，并承担额外检索延迟。

---

## 工程权衡与常见坑

最常见的误判是把“水印攻击”理解成“删水印字符串”。文本水印通常不是显式标签，而是采样偏置留下的隐式统计信号。所以真正的问题从来不是“有没有某个标记被删掉”，而是“原本可分辨的分布差是否还存在”。

第二个常见坑是过度相信 token-level green/red 列表。它实现确实简单，但工程环境里用户会复制、翻译、改写、摘要、拼接，甚至用另一个模型做后编辑。只要文本流经这些步骤，检测阈值就会漂。

下面这张表更适合做选型：

| 方案 | detectability | paraphrase 抗性 | 实现复杂度 | 线上延迟 | 典型坑 |
|---|---|---:|---:|---:|---|
| token-level hash | 高，原始文本下好用 | 低 | 低 | 低 | 翻译后几乎失效 |
| 语义检索防御 | 中到高 | 高 | 高 | 中到高 | 库维护成本高 |
| SemStamp | 高 | 高 | 高 | 中到高 | 生成链路改造大 |
| SemaMark | 中到高 | 中到高 | 中 | 低到中 | 依赖语义替换质量 |

还有三个容易踩的工程问题：

1. 阈值迁移失败  
同一个检测阈值，在原始文本、轻度改写文本、跨语言回译文本上通常不能直接复用。训练集和线上分布一变，FPR 与召回率会一起漂移。

2. 只检测，不做复验  
如果系统把“疑似 paraphrase 命中”直接当作最终判定，容易误伤高相似度的人类改写文本。更稳的做法是：检索命中后触发二次模型判别或人工审核。

3. 忽视多轮后处理  
攻击者不一定只做一次 paraphrase。生产环境中，文本可能经历“摘要 + 翻译 + 本地润色 + SEO 改写”多轮处理。单一 token 统计在这种链路下通常扛不住。

---

## 替代方案与适用边界

如果目标是“先部署、先上线、成本可控”，语义增强但仍保留 token 级效率的方案更现实，例如 SemaMark。它不需要像 SemStamp 那样深度改造整个生成过程，因此适合资源受限的 API 场景。

如果目标是“高价值内容、强鲁棒性、可接受更高成本”，SemStamp 更合理。它从句子语义空间直接建水印约束，对 paraphrase 的抵抗能力更强，但实现和运维负担也更高。

还要看到适用边界。没有哪种文本水印能在所有攻击下都稳定生存。若攻击者愿意接受一定语义漂移，或者用人工重写破坏嵌入邻域，那么语义水印也会被侵蚀。区别只在于：它让攻击成本从“简单换词”提升到“更大幅度地改变表达甚至语义”。

| 方案 | 适合场景 | 不适合场景 |
|---|---|---|
| token-level hash | 内部链路、文本不经过改写 | 用户可自由复制、翻译、二次编辑 |
| SemaMark | 低延迟 API、防改写需求中等 | 强对抗环境、需高鲁棒性证明 |
| SemStamp | 质量敏感、对 paraphrase 鲁棒性要求高 | 无法改造生成链路、预算紧张 |
| 检索防御 | 平台侧复验、历史生成可沉淀 | 无法保存大规模历史 embedding |

对初级工程师最重要的结论是：**如果你的威胁模型里包含 paraphrase 和 translation，就不要把 token 级水印当成最终方案。** 它可以是低成本基线，但不能承担“对抗性场景下的唯一证据”。

---

## 参考资料

- SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation (NAACL 2024). https://aclanthology.org/2024.naacl-long.226/
- A Robust Semantics-based Watermark for Large Language Model against Paraphrasing (Findings NAACL 2024). https://aclanthology.org/2024.findings-naacl.40/
- Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense (NeurIPS 2023). https://proceedings.nips.cc/paper_files/paper/2023/hash/575c450013d0e99e4b0ecf82bd1afaa4-Abstract-Conference.html
- Text Watermarks: Methods & Challenges (EmergentMind). https://www.emergentmind.com/topics/text-watermarks
