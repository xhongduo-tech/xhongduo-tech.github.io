## 核心结论

SFT，Supervised Fine-Tuning，指监督微调。它做的事情很具体：拿一批“指令或问题 -> 理想回答”的样本，继续训练基础模型，让模型学会按目标风格、目标格式、目标任务边界来回答。

SFT 数据质量，不是只看“标注贵不贵”或“条数多不多”，而是同时看三个维度：

| 维度 | 白话解释 | 直接影响 | 典型优化手段 |
| --- | --- | --- | --- |
| 指令多样性 | 问法、任务类型、场景覆盖够不够广 | 泛化能力、长尾任务表现、最差任务桶稳定性 | 扩展任务类型、改写同义问法、混合多领域数据、补足低频任务 |
| 响应质量 | 答案是否正确、完整、具体、可执行 | 学习信号强度、训练稳定性、错误风格是否被写入参数 | 人工抽检、规则过滤、长度阈值、代码可运行校验、拒答清洗 |
| 格式一致性 | 输入输出模板是否统一，字段是否完整 | 模型是否清楚“哪里是指令、哪里是回答” | 统一模板、补齐字段、清洗错误分隔符、统一消息结构 |

一个足够实用的工程近似是：

$$
\text{Performance} \propto \text{Quality}^{\alpha} \cdot \text{Quantity}^{\beta}, \quad \alpha > \beta
$$

它不是严格物理定律，而是经验判断：在 SFT 阶段，数据质量的边际收益通常高于单纯扩量。原因很直接。低质量数据不是“少学一点”，而是可能把错误答案、错误格式、错误风格稳定地教给模型。

先看一个最小例子。假设你的目标是教模型“写 Python 函数求列表平均值”。

如果你放入 10 条几乎同义的样本，例如：

- “写个函数求平均值”
- “写个函数求平均值”
- “写个函数求平均值”

模型主要学到的是一种固定表述。

但如果仍然只给 10 条样本，改成下面这种分布：

- “写一个 Python 函数计算列表平均值”
- “给出带类型注解的平均值函数”
- “空列表时应该返回什么”
- “把结果保留两位小数”
- “顺便写测试用例”
- “解释时间复杂度”
- “改成支持 `Iterable[float]`”
- “不要抛异常，返回 `None`”
- “给一个更健壮的实现”
- “把答案写成 `def` 代码块”

即使总条数完全不变，模型学到的任务边界也更广。它不只知道“平均值”这个词，还知道代码格式、异常处理、边界条件、解释方式、测试习惯。这就是“多样性比同类堆量更值钱”的最小示例。

真实工程里，这个现象会更明显。比如一个企业客服模型，如果 SFT 数据里 80% 都是 FAQ 和解释型问答，只有极少数代码、数学、表格分析、流程决策类样本，那么模型上线后往往会表现成：

- 很会解释概念
- 很会写礼貌回复
- 不太会做精确推理
- 不太会给可运行代码
- 不太会按结构化格式输出

问题不一定出在模型参数量，也不一定出在训练轮数，而是 SFT 数据分布已经把输出风格和能力偏好“压”向了解释型任务。

所以，SFT 数据建设最重要的判断不是“还能不能再加 10 万条”，而是先问三件事：

1. 任务覆盖是否足够广。
2. 答案是否真的值得学。
3. 模板是否统一到不会制造边界噪声。

---

## 问题定义与边界

这里讨论的“数据质量”，不是泛泛地说“数据越贵越好”，而是限定在 SFT 这一步可以直接治理的部分：

1. 指令覆盖度：任务分布是否覆盖目标场景。
2. 响应质量：答案是否正确、具体、完整、少空话、少伪拒答。
3. 模板一致性：样本结构是否稳定，字段是否完整，分隔符是否统一。

这个边界要说清楚，因为很多新手会把 SFT、预训练、RAG、偏好对齐混在一起。它们解决的问题不一样。

| 训练或系统环节 | 主要作用 | 解决什么问题 | 不能替代什么 |
| --- | --- | --- | --- |
| 预训练 | 学习广泛世界知识与语言模式 | “知道什么” | 不能保证按你想要的方式回答 |
| SFT | 学习任务格式、回答风格、输出边界 | “如何回答” | 不能凭空补足大量缺失知识 |
| RAG | 在推理时检索外部信息 | “实时查什么” | 不能代替基础表达能力 |
| 偏好对齐 / RLHF / DPO | 调整回答倾向与偏好排序 | “更像哪个答案” | 不能修复底层知识缺口 |

LIMA 给出的核心启发正是：对强基础模型来说，预训练已经承载了大部分通用知识，SFT 更像是在教模型“如何把已有知识组织成目标回答”。因此，少量高质量样本有时能逼近大量低质量样本的效果。

但这个结论有边界，不是“1000 条永远够用”。更准确的说法是：

- 当基础模型已经较强
- 当目标任务偏向通用指令跟随
- 当知识主要已存在于预训练中

这时，少量精筛 SFT 数据可能很有效。

反过来，如果你做的是下面这些任务：

- 金融合规条文问答
- 医疗机构内部流程问答
- 私有 API 编码助手
- 企业内部知识库问答
- 专有设备运维手册解析

那么知识本身在预训练里往往并不充分。此时 SFT 不能只追求“少而精”，还要覆盖目标知识边界，否则模型只会“答得像”，但内容不一定“答得对”。

数据混合可以写成：

$$
D_{\text{mixed}} = \sum_i w_i D_i,\quad \sum_i w_i = 1
$$

其中：

- $D_i$ 表示一个来源数据集
- $w_i$ 表示该数据集的采样权重

白话解释就是：训练时不是把所有数据一锅倒进去平均训练，而是按比例混合。例如 60% 通用问答、20% 代码、10% 数学、10% 领域问答。这个比例本身就是模型能力分布的一部分。

一个新手很容易误判的点是：看到论文里“1000 条精筛样本效果很好”，就直接得到“以后我只需要 1000 条数据”的结论。这是过度外推。正确理解应该是：

- 1000 条高质量数据常常适合做基线验证
- 但是否足够，要看基础模型强度和目标知识密度
- 对强模型做通用对齐，1000 条可能已经很有价值
- 对重领域任务，1000 条往往只是起点

可以把这个判断压成一张对照表：

| 方案 | 数据量 | 特征 | 更适合 |
| --- | --- | --- | --- |
| A | 1,000 条 | 人工筛选、格式统一、回答完整、重复低 | 快速验证基础指令跟随、建立高质量基线 |
| B | 50,000 条 | 来源混杂、模板不一、重复较多、质量波动大 | 只有在完成清洗和重采样后才适合扩覆盖 |
| C | 10,000 条 | 人工主干 + 合成补尾 + 明确权重 | 大多数实际工程场景的稳妥起点 |

如果任务是“企业知识库问答 + 少量代码解释”，A 往往更快跑出可用基线；如果任务是“开放域助理 + 多任务覆盖”，A 更像起步数据，后续仍然需要更多样化的数据补覆盖。

---

## 核心机制与推导

为什么“指令多样性、响应质量、格式一致性”这三个维度会同时起作用？关键在于搞清楚：模型在 SFT 中到底学什么。

### 1. 模型在学任务映射

所谓任务映射，就是模型看到某类输入，知道应该输出哪类答案。

比如这几种指令虽然都与“排序”有关，但任务映射不同：

| 指令 | 模型真正要学的任务 |
| --- | --- |
| “解释冒泡排序是什么” | 概念解释 |
| “写一个 Python 冒泡排序函数” | 代码生成 |
| “比较冒泡排序和快速排序” | 对比分析 |
| “把这段排序代码改成降序” | 代码编辑 |
| “用表格总结常见排序算法复杂度” | 结构化输出 |

如果训练数据几乎都是“请解释某个概念”，模型就会更擅长解释型回答，而不容易泛化到比较、步骤、改写、代码生成、结构化表格等其他任务形式。这就是为什么指令多样性会直接影响泛化能力。

EMNLP 2024 关于数据多样性的结果，核心支持的也是这个方向：更高的数据多样性，尤其能提升最差场景表现和整体鲁棒性。工程上可以把它理解成：不是只提高平均分，而是减少“某些题型明显不会”的情况。

### 2. 模型在学输出风格

SFT 不只是教“答什么”，也在教“怎么答”。

下面这些内容都会被模型学进去：

- 要不要分点
- 要不要先给结论
- 要不要展示推导步骤
- 代码是否要放在代码块里
- 碰到不确定情况时是说明假设还是直接拒答
- 表格应该如何对齐
- 答案长度应该多长

所以响应质量差，不是“影响一点效果”这么简单，而是可能把坏习惯直接写进参数：

| 低质量响应类型 | 模型可能学到的坏习惯 |
| --- | --- |
| 空洞礼貌回复 | 话很多但信息少 |
| 模板化拒答 | 在可回答问题上过度拒答 |
| 只给结论不解释 | 缺步骤、不可检查 |
| 代码不闭合 | 输出格式损坏 |
| 事实错误 | 稳定地产生错误内容 |

新手容易忽略的一点是：模型不会自动知道哪些样本“只是坏数据”。从训练角度看，只要样本被放进去，梯度就会把它当成监督信号。因此错误答案不是“浪费一次训练”，而是“明确告诉模型这个错误答案更接近目标”。

### 3. 模型在学边界标记

格式一致性的重要性，通常被低估。

如果你的训练集里混有三种模板：

```text
### Instruction
...
### Response
...
```

```text
User:
...
Assistant:
...
```

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

那模型学到的就不是一个稳定模板，而是三个互相竞争的边界系统。结果可能包括：

- 回答时莫名输出 `### Response`
- 模仿 `Assistant:` 前缀
- 把用户输入也继续补全
- 一部分样本学成“单轮指令”，另一部分学成“多轮对话”

对新手来说，可以把模板理解成“标签格式”。模板混乱，本质上就是标签噪声。

### 4. 一个简单的价值模型

可以把单个样本的有效训练价值写成示意式：

$$
v(x) = q(x)\cdot c(x)\cdot d(x)
$$

其中：

- $q(x)$ 表示响应质量得分
- $c(x)$ 表示格式一致性得分
- $d(x)$ 表示该样本对多样性的边际贡献

整个训练集的有效价值更接近：

$$
V(D) = \sum_{x \in D} v(x)
$$

这不是严格统计模型，但它足够解释工程现象：样本价值不是简单按条数相加，因为重复样本的 $d(x)$ 会很低，格式混乱样本的 $c(x)$ 会很低，错误响应的 $q(x)$ 会很低。

也可以进一步把“边际收益递减”写出来：

$$
\Delta V \approx q \cdot c \cdot \Delta d
$$

意思是：当新增样本与已有样本高度重复时，$\Delta d$ 很小，即使条数继续增加，总收益也不大。

### 5. 一个数字例子

假设两个数据方案，数量归一化分别是 $\text{Quantity}=0.7$ 和 $0.9$，质量分别是 $\text{Quality}=0.9$ 和 $0.6$。取 $\alpha=2,\beta=1$：

- 方案 A：$0.9^2 \times 0.7 = 0.567$
- 方案 B：$0.6^2 \times 0.9 = 0.324$

虽然 B 的数据更多，但 A 的综合收益更高。这个计算只是示意，不是通用打分器，但它准确表达了 SFT 的常见工程现象：低质量大集合，经常不如高质量中等规模集合。

### 6. 数据混合不是平均拼接

工程上做数据混合时，常见做法不是把所有源平均拼起来，而是先按质量排序，再按任务缺口调权重。例如：

| $w_i$ | 源数据类型 | 质量特征 | 目标任务 |
| --- | --- | --- | --- |
| 0.45 | 通用对话 | 人工标注、回答完整、风格统一 | 基础指令跟随 |
| 0.30 | 代码问答 | 代码可执行、含测试、低重复 | 代码生成与解释 |
| 0.15 | 数学推导 | 步骤清晰、符号规范、结论可核验 | 推理与计算 |
| 0.10 | 合成改写数据 | 覆盖长尾问法，但必须严格过滤 | 扩展表达多样性 |

这里真正重要的不是数字本身，而是顺序：

1. 先保证高质量主干。
2. 再用补充数据扩覆盖。
3. 低质量集合不能反客为主。

公开资料里，Llama 2 也明确强调过 SFT 标注来源和标注质量会显著影响下游效果。这说明同样是“几万条 SFT 数据”，不同来源之间的差异可能非常大。数量单位相同，不等于学习信号强度相同。

---

## 代码实现

下面给一个最小但可运行的 Python pipeline。目标不是替代完整数据工程，而是把新手最常见的三步先跑通：

1. 统一模板
2. 过滤明显低质响应
3. 计算简化质量分数并导出干净样本

代码只依赖 Python 标准库，可直接运行。

```python
import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

RAW_SAMPLES: List[Dict[str, str]] = [
    {
        "instruction": "解释什么是哈希表",
        "response": "哈希表是一种通过哈希函数把键映射到存储位置的数据结构，常用于高效查找、插入和删除。",
        "source": "human",
    },
    {
        "instruction": "写一个 Python 函数计算列表平均值",
        "response": "Sorry, I can't help with that.",
        "source": "synthetic",
    },
    {
        "instruction": "把下面句子改写得更正式",
        "response": "",
        "source": "human",
    },
    {
        "instruction": "写一个 Python 函数计算列表平均值",
        "response": (
            "```python\n"
            "from typing import Iterable, Optional\n\n"
            "def avg(xs: Iterable[float]) -> Optional[float]:\n"
            "    values = list(xs)\n"
            "    if not values:\n"
            "        return None\n"
            "    return sum(values) / len(values)\n"
            "```"
        ),
        "source": "human",
    },
]

REFUSAL_PATTERNS = [
    r"\bsorry\b",
    r"cannot help",
    r"can't help",
    r"无法帮助",
    r"不能提供",
]

MIN_RESPONSE_LENGTH = 12
QUALITY_THRESHOLD = 0.6


def normalize_text(text: str) -> str:
    """Normalize whitespace for duplicate detection and template stability."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in REFUSAL_PATTERNS)


def format_score(instruction: str, response: str) -> float:
    """A tiny heuristic for format consistency."""
    if not instruction or not response:
        return 0.0
    if "```" in response and response.count("```") % 2 != 0:
        return 0.4
    return 1.0


def quality_score(sample: Dict[str, str], duplicate_ratio: float) -> float:
    """
    Demo-only score:
    - empty response hurts
    - refusal hurts
    - poor format hurts
    - duplication hurts
    """
    instruction = normalize_text(sample.get("instruction", ""))
    response = sample.get("response", "").strip()

    empty_penalty = 1.0 if not response else 0.0
    refusal_penalty = 0.7 if is_refusal(response) else 0.0
    format_penalty = 1.0 - format_score(instruction, response)
    duplicate_penalty = min(1.0, duplicate_ratio)

    score = 1.0 - (
        0.35 * empty_penalty
        + 0.35 * refusal_penalty
        + 0.15 * format_penalty
        + 0.15 * duplicate_penalty
    )
    return max(0.0, round(score, 4))


def canonical_record(sample: Dict[str, str], duplicate_ratio: float) -> Dict[str, object]:
    instruction = normalize_text(sample.get("instruction", ""))
    response = sample.get("response", "").strip()

    prompt = f"### Instruction\n{instruction}\n\n### Response\n{response}"
    return {
        "instruction": instruction,
        "response": response,
        "source": sample.get("source", "unknown"),
        "quality_score": quality_score(sample, duplicate_ratio),
        "prompt": prompt,
    }


def clean_dataset(samples: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    samples = list(samples)
    normalized_pairs: List[Tuple[str, str]] = [
        (
            normalize_text(sample.get("instruction", "")),
            normalize_text(sample.get("response", "")),
        )
        for sample in samples
    ]
    pair_counter = Counter(normalized_pairs)

    cleaned: List[Dict[str, object]] = []
    for sample in samples:
        key = (
            normalize_text(sample.get("instruction", "")),
            normalize_text(sample.get("response", "")),
        )
        duplicate_ratio = (pair_counter[key] - 1) / max(1, len(samples))
        record = canonical_record(sample, duplicate_ratio)

        instruction = record["instruction"]
        response = record["response"]
        score = record["quality_score"]

        if not instruction:
            continue
        if not response:
            continue
        if is_refusal(str(response)):
            continue
        if len(str(response)) < MIN_RESPONSE_LENGTH:
            continue
        if float(score) < QUALITY_THRESHOLD:
            continue

        cleaned.append(record)

    return cleaned


def main() -> None:
    cleaned = clean_dataset(RAW_SAMPLES)

    assert len(cleaned) == 2
    assert all("### Instruction" in item["prompt"] for item in cleaned)
    assert all("### Response" in item["prompt"] for item in cleaned)
    assert all(float(item["quality_score"]) >= QUALITY_THRESHOLD for item in cleaned)

    print(json.dumps(cleaned, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

这段代码可以直接保存为 `clean_sft_data.py` 后运行：

```bash
python clean_sft_data.py
```

如果运行正常，你会看到清洗后的 JSON 输出。原始 4 条样本里：

- 1 条是空响应
- 1 条是模板式拒答
- 2 条是有效样本

最终只保留真正值得学的 2 条。这正是 SFT 清洗的基本逻辑：条数可以减少，但监督信号必须更干净。

为了让新手更容易理解，下面逐字段解释这份输出结构：

| 字段 | 作用 | 为什么重要 |
| --- | --- | --- |
| `instruction` | 用户指令本体 | 决定模型学什么任务映射 |
| `response` | 监督答案 | 决定模型学到什么风格和内容 |
| `source` | 数据来源 | 方便按来源审计质量 |
| `quality_score` | 简化质量分数 | 方便过滤和后续加权采样 |
| `prompt` | 统一后的训练模板 | 保证训练边界一致 |

### 如果你的原始数据是 `jsonl`

很多公开数据集并不是一个 JSON 数组，而是每行一个 JSON 对象，也就是 `jsonl`。处理思路不变，只是读写方式不同。最小可运行示例如下：

```python
import json
from pathlib import Path

from clean_sft_data import clean_dataset

def load_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def dump_jsonl(path: str, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    raw_records = list(load_jsonl("raw_data.jsonl"))
    cleaned = clean_dataset(raw_records)
    dump_jsonl("cleaned_data.jsonl", cleaned)

if __name__ == "__main__":
    main()
```

### 一个更实用的质量检查清单

上面的代码是入门版。真实工程里，通常还会继续加下面这些检查：

| 检查项 | 规则示例 | 目的 |
| --- | --- | --- |
| 代码块闭合 | ````` 出现次数必须成对 | 防止模型学坏格式 |
| 长度阈值 | `response` 至少 20 或 50 字 | 过滤空洞回答 |
| 语言一致性 | 中文指令优先配中文回答 | 减少风格漂移 |
| 近重复检测 | embedding 或 n-gram 相似度去重 | 减少换皮复制 |
| 来源配额 | 限制单一数据源占比 | 防止某来源支配风格 |
| 标签补齐 | 缺失 `instruction` / `response` 直接剔除 | 避免训练格式错乱 |

### 一个很容易被忽视的点

“代码可运行”本身就是质量的一部分。对于代码类 SFT 数据，只看“有没有代码块”远远不够，更好的标准是：

- 代码语法正确
- 边界条件被处理
- 示例输入输出自洽
- 最好带最小测试

比如上面平均值函数的实现，专门处理了空列表：

```python
def avg(xs):
    values = list(xs)
    if not values:
        return None
    return sum(values) / len(values)
```

如果你直接写 `sum(xs) / len(xs)`，空列表会报错。这种代码即使“看起来像答案”，也不一定是高质量监督信号。

---

## 工程权衡与常见坑

最常见的误区不是“数据太少”，而是“数据太乱”。SFT 的很多问题，不是训练器坏了，而是训练数据已经把偏差写死了。

### 1. 统一模板 vs 多模板混用

假设一部分数据长这样：

```text
### Instruction
...
### Response
...
```

另一部分长这样：

```text
User:
...
Assistant:
...
```

还有一部分直接保存成聊天消息数组。模型会把这几种格式同时当成“正确答案的一部分”来学。

结果通常是：

- 同一个问题有时输出 `### Response`
- 有时仿写 `Assistant:`
- 有时连用户前缀一起生成
- 多轮和单轮边界互相污染

对 SFT 来说，模板不是装饰，而是任务定义的一部分。模板混乱，本质上就是标签噪声。

### 2. 拒答、空答、短答污染

很多公开集或合成集里会混入这种回答：

- “Sorry, I can't help with that.”
- “这是一个很好的问题。”
- “具体情况具体分析。”
- “需要结合上下文决定。”

这些回答的问题不只是“信息少”，而是会把错误行为模式教给模型。模型可能因此学会：

- 在能答的问题上也拒答
- 用礼貌废话填充长度
- 回答看起来完整，实际没有结论

一个常见的示意指标是：

$$
\text{quality\_score} = 1 - (\text{empty} + \text{refusal} + \text{duplicate\_ratio})
$$

它不是严格评测指标，但足够做第一轮粗过滤。实际落地时，可以把不同项改成不同权重，而不是简单相加。

### 3. 任务分布偏斜

如果训练集 90% 都是 FAQ 和说明文，模型就会偏向输出解释型回答；如果代码类样本太少，哪怕基础模型预训练里见过代码，SFT 后也可能被通用对话风格覆盖，最终表现成：

- 会讲思路
- 不会给完整代码
- 代码没有测试
- 边界条件经常漏掉

这个现象可以理解为：SFT 在重新分配模型的“输出优先级”。

### 4. 合成数据回音室

“回音室”指模型拿自己或同类模型生成的数据继续训练自己。它便宜、扩量快，但风险也很明确：

- 同一套表达习惯被不断放大
- 某些口癖变成默认风格
- 表面上很多样，底层句式其实很像
- 错误推理模板被规模化复制

所以合成数据真正适合做的是“补覆盖”，不是“替代主干”。

### 5. 质量过滤顺序错了

很多人会先大量扩数据，再回头想办法清洗。这通常效率不高。更稳妥的顺序是：

1. 先统一模板
2. 再剔除空答、拒答、错误字段
3. 再去重
4. 最后才扩量

因为如果模板还没统一、字段还缺失、明显垃圾样本还没删，扩量只是在放大噪声。

下面把常见坑和规避方式压成一张表：

| 常见坑 | 现象 | 直接后果 | 规避策略 |
| --- | --- | --- | --- |
| 格式不一致 | 多套分隔符混用、字段缺失 | 模型混淆输入输出边界 | 训练前统一成单模板 |
| 空响应或过短响应 | 只有一句空话或无内容 | 学不到有效映射 | 设最小长度阈值 |
| 拒答样本过多 | 大量 “Sorry” 或模板化安全回复 | 过度拒答 | 规则过滤 + 人工抽检 |
| 重复数据 | 同义改写不足、换皮复制 | 边际收益很低 | 近重复检测、分桶采样 |
| 任务偏向 | QA 远多于代码/数学 | 某类能力被压制 | 按任务重采样或上采样 |
| 合成数据失真 | 看似通顺但内容空泛 | 学到错误风格 | 让高质量人工数据做主干 |

新手可以直接执行的最小策略其实很简单：

1. 把所有来源统一成一种模板。
2. 检查每条 `response` 是否非空、非拒答、非明显错误。
3. 去掉高重复样本。
4. 再根据任务缺口决定要不要扩量。

顺序不能反。因为“先扩再清”通常比“先清再扩”更贵，也更难定位问题。

---

## 替代方案与适用边界

SFT 数据建设常见有两条路线，不存在绝对优劣，只有阶段差异。

| 方案 | 优势 | 风险 | 适用边界 |
| --- | --- | --- | --- |
| 小规模人工精筛 | 可控、噪声低、模板稳定、风格统一 | 覆盖面有限、成本高、扩量慢 | 冷启动、内部模型、建立高质量基线 |
| 大规模合成扩展 | 覆盖广、便宜、扩表述快、长尾补齐容易 | 容易重复、失真、风格僵化 | 已有主干数据后用于补多样性 |
| 人工主干 + 合成补尾 | 兼顾质量与覆盖，是最常见工程解 | 流程复杂，需要过滤和配重 | 大多数中大型 SFT 项目 |

Self-Instruct 的价值主要在“扩覆盖”。它通过模型自生成 instruction-input-output，再过滤无效和相似样本，用较低人工成本补足人写数据不够广的问题。

Evol-Instruct 的价值主要在“提复杂度”。它不是只做同义改写，而是逐步把任务变难，例如：

- 从简单定义到多约束解释
- 从单步代码生成到带测试和边界条件
- 从直接问答到多步推理
- 从开放回答到格式化输出

对代码和数学类任务，这种“复杂度递进”通常比单纯增加同义问法更有效。

一个稳妥的流程通常是：

1. 先用约 1k 到数万条高质量人工或强审核数据建立主干。
2. 再用 Self-Instruct 或 Evol-Instruct 扩充长尾任务和表达方式。
3. 统一模板后，按质量分数和任务比例混合训练。
4. 始终保留人工高质量数据的主导地位。

举个简单配比示例。假设你已经有：

- 1,000 条人工精筛对话
- 50,000 条合成扩展数据

更稳妥的做法不是直接把 50,000 条全部压上去，而是先过滤和去重，再按类似 `0.6 : 0.4` 或 `0.7 : 0.3` 的比例混合人工与合成数据。这里关键不在“50,000”这个数字，而在“经过过滤且不喧宾夺主”。

边界也要明确：

| 维度 | 下限判断 | 上限判断 |
| --- | --- | --- |
| 数据量 | 几百到 1k 条高质量样本，通常已足够做基线实验 | 条数继续增大，如果质量不升反降，收益会快速变差 |
| 合成数据 | 可以用来补长尾问法和任务形式 | 如果不过滤，规模越大，错误风格越稳定 |
| 人工数据 | 成本高但最适合作为主干 | 如果只追求人工精筛而完全不扩覆盖，长尾泛化会不足 |

所以，替代方案不是“人工 vs 合成二选一”，而是：

- 先用人工数据建立正确分布
- 再用合成数据扩大覆盖边界
- 最后通过过滤、去重、配重来控制数据主导权

这比单纯追求“最大数据量”更符合 SFT 的实际规律。

---

## 参考资料

下面的资料优先级按“论文 > 官方/项目页 > 其他技术解读”排序。它们共同支持一个主线判断：SFT 数据建设的关键，不是机械扩量，而是围绕质量、覆盖、多样性和模板稳定性做工程治理。

| 来源 | 核心结论 | 可借鉴的工程行动 |
| --- | --- | --- |
| LIMA: Less Is More for Alignment, arXiv 2023 | 对强基础模型，少量高质量样本也能显著提升指令跟随能力，说明预训练已承载大量通用知识，SFT 更像学习回答方式 | 先做小规模高质量基线，再决定是否扩量 |
| Self-Instruct, arXiv 2022 | 通过模型自生成 instruction-input-output 并配合过滤，可以低成本扩展指令覆盖 | 用合成数据补长尾表达，但必须加过滤和去重 |
| Self-Instruct GitHub | 提供数据生成、过滤和样本构造的实际脚本与流程 | 不要只看论文结论，最好复用其过滤思路 |
| Data Diversity Matters for Robust Instruction Tuning, Findings of EMNLP 2024 | 多样性和质量存在权衡，但更高多样性对最差场景表现和鲁棒性有显著价值 | 不只看平均分，还要看最差任务桶和长尾任务表现 |
| Llama 2: Open Foundation and Fine-Tuned Chat Models, Meta 2023 | SFT 数据质量与标注来源差异会明显影响下游结果 | 建立来源审计、供应商抽检和数据配额机制 |
| WizardCoder | Evol-Instruct 风格的数据复杂化，对代码类指令提升明显 | 代码任务优先增加复杂度，而不是只堆同义问法 |
| WizardMath | 复杂度递进式构造对数学类推理任务有效 | 数学数据应强调步骤、约束和可核验性 |

可直接引用的原始链接：

- LIMA: https://arxiv.org/abs/2305.11206
- Self-Instruct: https://arxiv.org/abs/2212.10560
- Self-Instruct GitHub: https://github.com/yizhongw/self-instruct
- Data Diversity Matters for Robust Instruction Tuning: https://aclanthology.org/2024.findings-emnlp.195/
- Llama 2: https://arxiv.org/abs/2307.09288
- WizardCoder: https://arxiv.org/abs/2306.08568
- WizardMath: https://arxiv.org/abs/2308.09583

如果把这些结论落到自己的 SFT 实验里，一个更稳妥的做法不是一开始就追求“大而全”，而是先建立两组基线：

1. 一组是小规模高质量人工集。
2. 一组是大规模但未严格过滤的混合集。

然后在同一基础模型、同一训练配置下，至少比较三类指标：

| 指标 | 看什么 | 为什么重要 |
| --- | --- | --- |
| 平均表现 | 常规评测集平均分 | 反映整体水平 |
| 最差任务桶表现 | 长尾任务、少数类任务、失败率最高的子集 | 反映鲁棒性与覆盖问题 |
| 输出模板稳定性 | 是否稳定遵循目标格式、是否混入奇怪前缀 | 直接反映格式一致性是否学稳 |

这样你通常会很快看到，SFT 数据建设的第一优先级往往不是“再找更多条”，而是：

- 把已有样本变干净
- 把模板变统一
- 把任务分布变互补
- 再决定下一步要不要扩量
