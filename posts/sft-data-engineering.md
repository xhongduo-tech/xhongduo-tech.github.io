## 核心结论

SFT 数据工程的目标，不是“把数据尽量收集多”，而是把会影响模型行为的噪声系统性去掉，再把真正有训练价值的样本尽量保留下来。对监督微调来说，最常见也最有效的流水线是三步：先去重，再做质量过滤，最后做高价值样本打分或排序。

第一步去重解决“重复学同一句话”的问题。第二步质量过滤解决“模型学错、学脏、学空”的问题。第三步价值打分解决“有限训练预算该先学什么”的问题。三步一起做，训练集才会同时具备覆盖度、可执行性和性价比。

| 阶段 | 主要目标 | 常用技术 | 典型风险 |
| --- | --- | --- | --- |
| 去重 | 删除完全重复和高度相似样本 | `hash`、n-gram Jaccard、MinHash、LSH | 误删长尾差异样本 |
| 质量过滤 | 删除无效、低质、异常格式数据 | 困惑度过滤、长度过滤、规则过滤 | 阈值过严导致误杀 |
| 价值打分 | 在剩余样本里优先保留更有训练收益的数据 | IFD、LLM 打分、人工规则融合 | 成本高、评分不稳定 |

玩具例子可以这样理解：抓取到 100 万条问答后，先用 `hash(text)` 删除字面完全一样的记录，再用 MinHash 找出“文字不完全一样但本质重复”的问答，最后再用困惑度和规则删掉乱码、广告和不通顺回答。剩下的数据量可能更小，但训练价值更高。

真实工程里，这个顺序非常重要。因为如果先做 LLM 打分，再去重，你会把 API 成本浪费在本该删除的重复样本上；如果先做困惑度过滤但不做近似去重，模型仍可能被同一类模板答案反复强化，导致风格单一、泛化下降。

---

## 问题定义与边界

SFT 数据工程处理的是“指令-回答对”或“多轮对话样本”的清洗，不是预训练语料的全量治理。预训练数据更关注覆盖面，SFT 更关注行为对齐，也就是让模型学会“按要求回答”。因此，同样一段文本，在预训练里可能能用，在 SFT 里却可能必须删除。

这里要先把三个概念说清楚。

“重复”分两类。精确重复指字节级或标准化后完全一样，比如空格、大小写、标点统一后文本相同。近似重复指表面表达不同，但信息内容几乎相同，比如“Python 如何排序列表”和“怎么给 Python list 排序”配上几乎一样的回答。前者适合哈希，后者适合集合相似度或语义相似度。

“高质量”不是一个抽象好坏感，而是一组可操作条件。常见条件包括：语言是否通顺、长度是否处于合理区间、格式是否完整、是否包含脏词或模板污染、回答是否真的响应了指令。这里“困惑度”可以理解为“语言模型读这段文字时有多吃力”，值太高常意味着文本异常、不自然或领域外噪声。

“高价值样本”也不是“看起来高级”的样本，而是对当前模型更有增益的样本。IFD，Instruction Following Difficulty，可以理解为“这条样本对模型有多难”。如果一个回答在没有指令时很容易生成，但在给定指令后仍不容易对齐，说明这条指令约束本身有训练价值。

一个常见边界问题是：标点不同、措辞略改，算不算重复？答案是不能只用一个规则。比如：

- `Q1`: “HTTP 404 是什么意思？”
- `Q2`: “HTTP 404是什么意思”

如果只差空格和标点，应先做标准化后精确去重。
如果两个问题一个问定义，一个问排查步骤，就算词面相似也不能直接删除。

近似重复通常先把文本切成若干 n-gram，也就是连续的 n 个词或字符片段，再计算 Jaccard 相似度：

$$
J(A, B)=\frac{|A \cap B|}{|A \cup B|}
$$

其中 $A$、$B$ 是两个文本的 shingle 集合。shingle 可以理解为“局部文本片段集合”，它把文本变成了可比较的离散单元。

若直接两两计算 Jaccard，复杂度会很高。于是常用 MinHash 做签名压缩，再用 LSH 做候选召回。LSH 的核心是：相似样本更容易落到同一个桶里。若每个 band 有 $R$ 行，总共有 $B$ 个 band，碰撞概率近似为：

$$
P_{\text{collision}} = 1 - (1 - s^R)^B
$$

其中 $s$ 是两条样本的相似度。这个公式告诉我们，band 和 row 的设置本质上是在调“召回”和“误报”的平衡。

IFD 的常见定义可以写成：

$$
IFD_\Theta(Q, A)=\frac{L_\Theta(A|Q)}{L_\Theta(A)}
$$

这里 $Q$ 是指令，$A$ 是回答，$L_\Theta(A|Q)$ 是模型在给定指令时生成回答的损失，$L_\Theta(A)$ 是不看指令时生成回答的损失。直白地说，就是“这段回答有多少难度是由指令带来的”。值越大，通常说明这条指令对“服从要求”的训练价值越高。

---

## 核心机制与推导

一个稳妥的 SFT 数据工程流程通常按下面顺序执行：

1. 标准化
2. 精确去重
3. 近似去重
4. 质量过滤
5. 价值打分
6. 分桶抽样与落盘

先做标准化，是因为很多“重复”只是格式差异。常见操作包括统一大小写、合并多余空格、统一全半角、去掉无意义标点抖动。标准化后的文本再做哈希，能把大量伪差异样本直接删掉。

精确去重很简单：对标准化文本计算哈希值，相同哈希的样本只保留一份。它快、稳定、便宜，但只能抓住“完全一样”的重复。

近似去重的核心是“集合相似度近似”。玩具例子如下。假设两个回答切成 4-gram 后：

- 文档 A 的片段集合：5 个
- 文档 B 的片段集合：5 个
- 交集：3 个
- 并集：5 个

则

$$
J(A,B)=\frac{3}{5}=0.6
$$

如果 LSH 参数设为 `band=4, row=2`，碰撞概率为：

$$
P=1-(1-0.6^2)^4 \approx 0.67
$$

这表示它们有约 67% 的概率被召回为候选近似重复。这里不是“100% 判定重复”，而是“高概率进入下一步精查”。这也是 LSH 的工程价值：先快速缩小候选，再做更精细判断。

可以把这条链路理解成：

| 环节 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| Shingling | 原始文本 | n-gram 集合 | 把文本转成可比较对象 |
| Jaccard/MinHash | 集合 | 紧凑签名 | 用较低代价近似相似度 |
| LSH | 签名 | 候选重复对 | 避免全量两两比对 |
| 精查规则 | 候选对 | 删除/保留决策 | 控制误删率 |

质量过滤一般由三类规则组成。

第一类是困惑度过滤。困惑度是语言模型对文本序列的平均不确定性，通常由 token 级负对数似然得到：

$$
\mathrm{PPL}(x)=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i|x_{<i})\right)
$$

值太高，往往意味着乱码、拼接错误、机器翻译残缺、语法极不自然，或文本根本不在模型可理解分布里。

第二类是长度过滤。过短样本常见问题是信息不足，例如回答只有“可以”“不行”“看情况”；过长样本则可能是网页整段复制、包含无关声明、重复啰嗦或被 prompt 污染。但长度不能用一个硬阈值解决，因为“返回状态码 404 表示资源不存在”本身就是高质量短回答。

第三类是规则过滤。规则过滤就是显式写出的质量约束，例如：

- 指令为空
- 回答为空
- 代码块未闭合
- 出现广告导流
- 出现“作为 AI，我不能”
- 多轮对话角色错位

这类规则很粗，但对明显坏样本很有效。

IFD 是排序层，不一定是强过滤层。它适合在质量过滤之后使用。因为如果原始数据里噪声很多，IFD 可能把“模型很困惑的垃圾文本”也排高。正确做法是：先确保样本基本可用，再用 IFD 估计训练收益。

真实工程例子可以参考常见的 Web 指令数据清洗流程：先统一标点和大小写，再做精确去重和 MinHash 近似去重，再用小语言模型计算困惑度过滤掉异常文本，再做长度和格式规则过滤，最后用 IFD 或外部大模型评分，从剩余数据中选出比例更高的高价值样本进入 SFT。这个顺序的本质是“便宜规则先做，昂贵判断后做”。

---

## 代码实现

下面给出一个可运行的简化版 Python 示例。它没有接入真实大模型 API，但完整展示了“标准化 → 精确去重 → 近似去重 → 长度/规则过滤 → 质量打分”的结构。真实工程中，你只需要把占位函数替换成实际模型或服务。

```python
import re
import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，。！？；：]", "", text)
    return text


def exact_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 4) -> set:
    text = normalize(text)
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rule_filter(sample: Dict) -> bool:
    q = sample["instruction"].strip()
    a = sample["answer"].strip()
    if not q or not a:
        return False
    if "http://广告" in a or "加微信" in a:
        return False
    if len(a) < 4:
        return False
    return True


def mock_perplexity(text: str) -> float:
    # 玩具版本：重复字符和异常符号越多，分数越差
    bad = len(re.findall(r"[\?\?@#￥%]+", text))
    repeat = max((text.count(ch) for ch in set(text)), default=1)
    return 10 + bad * 20 + max(0, repeat - 6) * 2


def mock_ifd(instruction: str, answer: str) -> float:
    # 玩具版本：指令约束越明确、回答越结构化，值越高
    cond = 1 + instruction.count("请") + instruction.count("步骤") + instruction.count("解释")
    uncond = max(1, len(answer) / 40)
    return cond / uncond


def pipeline(samples: List[Dict], near_dup_threshold: float = 0.8) -> List[Dict]:
    dedup_by_hash = {}
    for s in samples:
        merged = normalize(s["instruction"] + "\n" + s["answer"])
        h = exact_hash(merged)
        if h not in dedup_by_hash:
            dedup_by_hash[h] = s

    exact_deduped = list(dedup_by_hash.values())

    kept = []
    seen_shingles = []
    for s in exact_deduped:
        current = shingles(s["instruction"] + " " + s["answer"])
        is_dup = any(jaccard(current, prev) >= near_dup_threshold for prev in seen_shingles)
        if not is_dup:
            seen_shingles.append(current)
            kept.append(s)

    filtered = []
    for s in kept:
        if not rule_filter(s):
            continue
        ppl = mock_perplexity(s["answer"])
        if ppl > 35:
            continue
        s = dict(s)
        s["ifd_score"] = mock_ifd(s["instruction"], s["answer"])
        filtered.append(s)

    filtered.sort(key=lambda x: x["ifd_score"], reverse=True)
    return filtered


samples = [
    {"instruction": "解释 HTTP 404 的含义", "answer": "404 表示服务器找不到请求的资源。"},
    {"instruction": "解释HTTP 404的含义", "answer": "404表示服务器找不到请求的资源。"},
    {"instruction": "请分步骤说明如何排查 Python 导入错误", "answer": "1. 检查模块名。2. 检查虚拟环境。3. 检查 PYTHONPATH。"},
    {"instruction": "你好", "answer": "加微信领取资料"},
]

result = pipeline(samples, near_dup_threshold=0.75)

assert len(result) == 2
assert result[0]["ifd_score"] >= result[1]["ifd_score"]
assert all("加微信" not in x["answer"] for x in result)
```

这段代码体现了几个工程上必须明确的输入输出边界：

| 模块 | 输入 | 输出 | 默认关注点 |
| --- | --- | --- | --- |
| `normalize` | 原始文本 | 标准化文本 | 消除格式噪声 |
| `exact_hash` | 标准化文本 | 哈希值 | 精确去重 |
| `shingles` + `jaccard` | 文本 | 相似度 | 近似去重 |
| `rule_filter` | 样本字典 | 布尔值 | 快速删坏样本 |
| `mock_perplexity` | 回答文本 | 分数 | 语言自然度 |
| `mock_ifd` | 指令、回答 | 分数 | 训练价值排序 |

如果扩展到真实工程，参数建议显式配置，而不是写死在代码里：

| 参数 | 作用 | 常见默认值 | 调整方向 |
| --- | --- | --- | --- |
| `shingle_size` | n-gram 长度 | 3 或 4 | 越大越保守 |
| `num_perm` | MinHash 哈希数 | 128 或 256 | 越大越准但越慢 |
| `bands` / `rows` | LSH 切分参数 | 32/4 或 16/8 | 决定召回和误报 |
| `ppl_threshold` | 困惑度阈值 | 按分位数设定 | 领域差异很大 |
| `min_answer_len` | 最短回答长度 | 8 到 32 字符 | 需按任务分层 |
| `llm_score_threshold` | LLM 评分门槛 | 3/5 或 4/5 | 成本与质量平衡 |

如果要接入类似 Alpagasus 的大模型打分策略，通常不是“直接相信一次评分”，而是设计评分模板、重复采样和聚合策略，例如同时让模型打“有用性、事实性、遵循指令程度”，再做加权总分。这样单次波动更小。

---

## 工程权衡与常见坑

最大的误区是把数据工程理解成“阈值越严格越好”。事实正好相反。SFT 的目标不是做一个最干净的语料库，而是做一个最有训练收益的数据集。过激清洗会直接损失覆盖度。

第一类常见坑是去重过头。比如两个问答只在操作系统环境上不同，一个是 Ubuntu，一个是 Windows，如果仅因为主体词相似就删除其中一个，模型就会失去环境差异知识。对 FAQ 类数据尤其要小心，模板相似不等于训练价值相同。

第二类坑是困惑度误用。困惑度高不一定是坏数据，也可能是专业术语密集、代码片段多、数学符号多。比如芯片架构、定理证明、LaTeX 推导常常天然更难。正确做法通常不是单阈值硬砍，而是按领域、样本类型分别设阈值，或看分位数而不是绝对值。

第三类坑是长度规则误杀。新手很容易把“短回答”直接当低质数据删掉，但很多高质量答案本来就很短。例如：

- 问：“为什么会报 404？”
- 高质量短答：“因为请求的资源不存在或路径错误。”

如果简单规定“回答少于 20 字全部删除”，就会把这类真实有效样本一起删掉。更合理的方式是分层：对事实问答允许短答，对复杂操作题要求更长的解释或步骤。

第四类坑是 LLM 打分过于昂贵且不稳定。像 Alpagasus 这类策略的核心价值在于利用强模型给弱模型筛样本，但它的代价是 API 成本、速率限制和评分漂移。不同时间、不同提示词、不同温度设置都可能让评分波动。工程上常见做法是只对通过基础过滤的数据做评分，并采用小批量抽检验证阈值是否合理。

| 坑 | 后果 | 规避策略 |
| --- | --- | --- |
| 去重阈值过高 | 漏掉大量近似重复 | 提高召回，增加二次精查 |
| 去重阈值过低 | 误删长尾样本 | 分领域设阈值，保留来源多样性 |
| 困惑度单阈值 | 专业文本被误杀 | 按类型分桶设阈值 |
| 最短长度过严 | 短而准的答案被删除 | 问题类型分层规则 |
| LLM 打分直接裁决 | 成本高且不稳定 | 只用于重排或灰区样本 |
| 只看总量不看分布 | 某类任务被清空 | 统计每类保留率 |

一个实用配置思路是：先抽样 1000 条人工看，观察“误删”和“漏删”分别来自哪里，再调参数，而不是先相信某个论文里的默认阈值。因为不同数据源的脏数据形态差异很大，网页问答、客服日志、代码解释、数学题解的分布完全不同。

---

## 替代方案与适用边界

MinHash + 规则过滤不是唯一方案，它的优势是便宜、稳定、可解释，但并不总是最优。

第一种替代方案是嵌入去重。嵌入就是把文本编码成向量，向量间距离近，通常意味着语义相近。最常见的是计算余弦相似度：

$$
\cos(\mathbf{x}, \mathbf{y})=\frac{\mathbf{x}\cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}
$$

相比 Jaccard，嵌入方法能抓住“语义近但字面差异大”的重复，例如“如何压缩图片体积”和“怎样减小图片文件大小”。缺点是需要向量模型、索引系统，成本更高，也更难解释为什么两条样本被判重。

第二种替代方案是多模型质量集成。比如用小语言模型算困惑度，用规则模型检查格式，再用分类器判断是否回答了指令，最后把多个信号做加权。它比单一阈值更稳，但系统复杂度更高，需要标注数据做校准。

第三种替代方案是全量 LLM 裁判。也就是直接让大模型判断样本是否高质量、是否忠实、是否安全。这个方案在小规模高价值数据集上很有效，但在大规模流水线上成本通常不可接受，而且评分一致性是难点。

| 方案 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- |
| `hash` + MinHash + 规则 | 快、便宜、可解释 | 语义重复识别有限 | 中小规模、成本敏感场景 |
| 嵌入 + ANN 检索 | 语义去重更强 | 需要向量模型和索引 | 大规模、多表达变体场景 |
| 困惑度 + 规则 + 分类器集成 | 质量判断更稳 | 需要更多工程组件 | 中高质量生产数据流 |
| 全量 LLM 打分 | 质量上限高 | 成本高、漂移大 | 小而精数据集或最后重排 |

对零基础到初级工程师，一个可执行判断标准是：

- 数据量不大、预算有限、希望快速落地：先用 `hash + MinHash + 规则 + 简单困惑度`
- 数据量很大、表达方式很多、近义改写严重：考虑嵌入去重
- 已有高质量标注集、要做稳定生产：考虑多信号融合
- 数据量小但每条样本都贵：可以引入 LLM 打分做精修

因此，原方案并不是“老旧方案”，而是一个很强的基线。只有当你遇到明显的语义重复漏检、规则体系维护困难、或质量判断需要更细粒度标准时，才值得升级到更重的方案。

---

## 参考资料

| 资料 | 主要内容 | 使用章节 |
| --- | --- | --- |
| Data Filtering Stage for LLMs | 系统梳理精确去重、近似去重、质量过滤的常见流程 | 去重、质量过滤、工程权衡 |
| Instruction-Following Difficulty | 给出 IFD 的定义与用它筛选高价值指令样本的方法 | 问题定义、核心机制 |
| Alpagasus 相关工作 | 使用 ChatGPT 对指令数据评分并过滤低质量样本 | 代码实现、工程权衡 |
| EPFL 数据混合课程材料 | 介绍 Jaccard、MinHash、LSH 及数据混合中的过滤思想 | 问题定义、核心机制 |
| Falcon / Web 数据清洗经验文章 | 展示真实工程里标准化、近似去重、困惑度过滤的组合用法 | 核心机制、真实工程例子 |

可参考链接：
- https://rahatibnrafiq.github.io/llm_data_filtering/
- https://www.epfl.ch/labs/lions/wp-content/uploads/2025/04/ee-628-Lecture_03_Data-Mixtures.pdf
- https://cloud.baidu.com/article/3361164
- https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/132913337
