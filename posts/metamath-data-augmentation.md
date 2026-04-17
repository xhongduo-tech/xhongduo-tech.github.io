## 核心结论

MetaMath 的核心不是“多造一些数学题”，而是系统性地把同一道题改写成多条不同的推理路径。它使用四种数据增强策略: 答案增强、问题重述、 FOBAR 反向推理、Self-Verification 自我验证，把 GSM8K 与 MATH 的约 1.5 万条基础样本扩展到 39.5 万条 MetaMathQA 样本，再用这些样本微调 7B 级模型。

这里的“数据增强”可以先理解成: 不改知识范围，但改变提问方式、解题顺序和验证方式，让模型见到更多“同题不同解法”的训练视角。对模型来说，新增样本的价值不在于又看见了数字 `5`、`4`、`5.5`，而在于它被迫学习了顺推、逆推、重述、验算这几种不同任务。

公开结果表明，这套方法把 LLaMA-2-7B 在 GSM8K 上提升到 66.5%，在 MATH 上提升到 19.8%。更重要的结论不是某个绝对分数，而是: 只做表面扩写很快会遇到上限，必须引入反向求解和自检，模型才会继续涨。换句话说，训练收益的关键不只是“样本更多”，而是“任务分布更宽”。

| 数据阶段 | 样本来源 | 规模 | 训练含义 |
| --- | --- | ---: | --- |
| 基础数据 | GSM8K 7,473 条 + MATH 约 7,500 条精选集 | 约 15K | 原始题目与标准解法 |
| 答案增强 | 扩写已有解题链 | 约 155K | 补足中间步骤，降低链条噪声 |
| 问题重述 | 改写题面表达 | 约 130K | 增加语义变体，减少模板依赖 |
| 全部策略合并 | 加入 FOBAR 与 Self-Verification | 395K | 扩展推理方向与验证能力 |

玩具例子可以直接看“牛肉题”。原题是: 买 5 袋牛肉，每袋 4 磅，每磅 5.5 美元，总价是多少。答案增强会把 $5 \times 4 \times 5.5 = 110$ 的中间步骤写得更细；问题重述会把题面改成“如果每袋 4 磅、单价 5.5 美元，5 袋共需多少钱”；FOBAR 会反过来问“总价 110 美元时一共买了几袋”；Self-Verification 会先陈述“5 袋牛肉共 110 美元”，再要求模型验证这个结论是否成立。四个版本围绕的是同一事实，但训练的思维方向已经不一样了。

---

## 问题定义与边界

MetaMath 解决的问题是: 当基础数学数据量有限时，怎样让模型学到更多推理结构，而不是只背更多题面。

“推理结构”可以先理解成: 模型从条件走到答案的路径形状，比如顺推、逆推、代入、验证、拆步。新手读者可以把它理解成“做题动作模板”。如果模型只学会一种动作模板，它在熟悉题面上能答对，但一旦换问法、倒着问、要求检查别人答案，性能就会明显掉。

它的边界也很明确。

第一，它主要针对数学推理数据的“视角稀缺”，不是通用知识补全。也就是说，MetaMath 并不负责教模型新的数学定理体系，而是让模型在已有题型上学会更多解题路线。

第二，它依赖强教师模型生成增强数据。论文实现里主要使用 GPT-3.5 生成扩写、重述与反推样本，所以增强质量上限很大程度受教师模型约束。教师模型如果经常算错、漏条件或乱改题意，增强出来的数据不但没有帮助，反而会污染训练集。

第三，并不是样本数越大越好。作者强调的是 question diversity，也就是“问题多样性”。白话讲就是: 新样本必须真的像新题，而不是把旧题换几个词。这个思想可以用一个多样性增益公式表示:

$$
dgain=\frac{1}{M}\sum_{x\in D_{new}}\min_{y\in D_{base}}\|f(x)-f(y)\|_2^2
$$

这里:
- $D_{new}$ 是新增样本集合
- $D_{base}$ 是基础样本集合
- $f(\cdot)$ 是文本嵌入，可以理解成“把一句题目压成一个向量”
- $\|f(x)-f(y)\|_2^2$ 是平方欧氏距离，可以理解成两道题在语义空间里隔多远
- $\min$ 表示新增样本只看自己离基础集中最近的那道题
- $M$ 是新增样本数

这个公式的意思很直接: 如果新增样本和基础集里最近的一道题都很远，说明它大概率提供了新的推理视角；如果距离很小，说明它可能只是近义改写，训练收益有限。

还是看牛肉题。原题求总价，FOBAR 题求袋数。这两题共享同一组数值，但目标变量变了，解题方向从顺推变成逆推，所以嵌入距离通常会比简单改写更大。这就是 MetaMath 为什么强调“反向推理”而不是只做同义句扩写。

再把边界说得更具体一点:

| 问题 | MetaMath 是否直接解决 | 原因 |
| --- | --- | --- |
| 模型不知道某个数学定理 | 否 | 它主要重排已有题型，不负责补新知识 |
| 模型会算但不稳 | 部分解决 | 答案增强和自检能提升稳定性 |
| 模型只会正向求值 | 是，重点解决 | FOBAR 专门扩展逆向求解 |
| 模型容易被问法扰动 | 是 | 问题重述增加语义覆盖 |
| 模型需要严格形式化证明 | 否 | 自然语言增强不足以替代定理证明系统 |

---

## 核心机制与推导

四类增强策略可以理解为四种向模型施加约束的方法。它们不是简单叠加，而是分别覆盖“讲清步骤”“换一种说法”“倒着求未知数”“做完后验算”这四类不同能力。

| 策略 | 输入变化 | 输出形式 | 主要训练作用 |
| --- | --- | --- | --- |
| 答案增强 | 题目不变 | 更细的解题链 | 让中间步骤更稳定 |
| 问题重述 | 题意不变、表达改变 | 新题面 + 原答案 | 增强语义鲁棒性 |
| FOBAR | 把答案或中间量改成条件 | 新问题 + 新解题链 | 训练逆向求解 |
| Self-Verification | 把题干改成陈述并要求检验 | 判断/验证链 | 训练结果校验 |

先看答案增强。它最像“把草稿写完整”。比如牛肉题不再直接写 $5 \times 4 \times 5.5$，而是先算总磅数 $5 \times 4 = 20$，再算总价 $20 \times 5.5 = 110$。这种增强提升的是链条清晰度，优点是稳定，缺点是新信息少，所以单独使用很快饱和。

再看问题重述。它处理的是表述扰动。比如“每袋 4 磅”换成“每个袋子装 4 磅”，“总共要付多少钱”换成“最终花费是多少”。这能减少模型对固定模板的依赖，但因为目标变量没变，推理骨架往往还是同一条。对新手来说，这一类增强最容易理解，因为它像“老师换一种问法”。

FOBAR 是关键。FOBAR 可以理解成“把原题的答案或中间变量翻回来当条件，再要求求另一个未知数”。这一步真正改变了信息流方向。原来是条件 $\rightarrow$ 答案，现在变成答案 $\rightarrow$ 某个条件。对模型来说，这不是语言改写，而是求解拓扑被改了。

如果把牛肉题写成代数式，原题是

$$
p = b \cdot w \cdot c
$$

其中:
- $b$ 表示袋数
- $w$ 表示每袋磅数
- $c$ 表示每磅价格
- $p$ 表示总价

原题求的是 $p$，FOBAR 可以改成求 $b$:

$$
b = \frac{p}{w\cdot c}
$$

两道题使用同一组变量，但未知数换了。训练信号也随之变化。模型不再只是学习“把条件代进去”，而是学习“看到结果后如何倒推原因”。

Self-Verification 训练的是“做完以后再检查”。比如给出“5 袋牛肉共 110 美元”，模型需要重新计算并判断结论是否一致。这里模型不只是生成答案，而是比较“声明值”和“自己推导值”是否相等。若记模型算出的结果为 $\hat{a}$，题面给出的声明值为 $a$，那么它本质上学习的是一个判别过程:

$$
verify(a,\hat{a})=
\begin{cases}
\text{True}, & a=\hat{a}\\
\text{False}, & a\neq \hat{a}
\end{cases}
$$

如果考虑数值题里常见的小数误差，工程上通常会把它写成近似判断:

$$
verify_\epsilon(a,\hat{a})=
\mathbf{1}\left(|a-\hat{a}|<\epsilon\right)
$$

其中 $\epsilon$ 是容忍误差阈值。这样做的原因很实际: 真实数据生成链路里经常出现 `110`、`110.0`、`109.999999` 这种格式差异，硬做字符串相等会误杀本来正确的样本。

为什么四者能互补，可以用推导关系理解:
- 答案增强主要降低单条链条的噪声
- 问题重述主要提升同义表达覆盖
- FOBAR 主要扩大目标变量空间
- Self-Verification 主要强化结果一致性检查

如果只做前两者，模型看到的仍然大多是“正向求答案”的任务族，所以提升会很快见顶。加入后两者后，训练目标从单一生成扩展为“生成 + 逆推 + 校验”，任务分布才真正变宽。这也是论文消融实验里后两者贡献更显著的原因。

真实工程例子可以看教育答疑系统。一个在线题库如果只喂“标准题 + 标准解”，模型经常能答出熟悉题面，但学生稍微换个问法、倒着问条件、或者要求检查某个同学的解答时，模型表现会掉得很明显。MetaMath 的四种策略对应的正是这三类真实变化: 改写题面、反向求未知数、验证已有解答。

---

## 代码实现

工程上可以把 MetaMath 看成“两阶段流水线”: 先生成增强样本，再做监督微调。真正难的不是写出几个 prompt，而是保证生成样本可追踪、可验证、可过滤。

第一阶段是数据生成。输入是一条基础样本，输出是若干策略标记后的新样本。最低要求是保留来源题、增强策略、生成内容、目标答案、以及是否通过校验这几个字段。否则后面很难做去重、抽检和错误回溯。

下面给一个可运行的 `python` 玩具版示例，用规则模拟四种增强。它不是论文原始实现，但数据结构、校验逻辑和任务区分方式与真实系统是一致的。代码可以直接运行。

```python
from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import Any, Dict, List
import json


@dataclass
class MathSample:
    question: str
    bags: int
    pounds_per_bag: int
    price_per_pound: Decimal

    @property
    def total_pounds(self) -> int:
        return self.bags * self.pounds_per_bag

    @property
    def total_price(self) -> Decimal:
        return Decimal(self.total_pounds) * self.price_per_pound


def answer_augmentation(sample: MathSample) -> Dict[str, Any]:
    steps = [
        f"先算总重量: {sample.bags} x {sample.pounds_per_bag} = {sample.total_pounds} 磅",
        f"再算总价: {sample.total_pounds} x {sample.price_per_pound} = {sample.total_price} 美元",
    ]
    return {
        "strategy": "answer_aug",
        "original_question": sample.question,
        "query": sample.question,
        "response": "\n".join(steps) + f"\n答案: {sample.total_price}",
        "target": str(sample.total_price),
    }


def paraphrase(sample: MathSample) -> Dict[str, Any]:
    query = (
        f"每袋牛肉重 {sample.pounds_per_bag} 磅，"
        f"每磅售价 {sample.price_per_pound} 美元。"
        f"如果一共买了 {sample.bags} 袋，总共要支付多少美元？"
    )
    return {
        "strategy": "paraphrase",
        "original_question": sample.question,
        "query": query,
        "response": f"总价 = {sample.bags} x {sample.pounds_per_bag} x {sample.price_per_pound} = {sample.total_price}",
        "target": str(sample.total_price),
    }


def fobar(sample: MathSample) -> Dict[str, Any]:
    query = (
        f"已知总价是 {sample.total_price} 美元，"
        f"每袋 {sample.pounds_per_bag} 磅，"
        f"每磅 {sample.price_per_pound} 美元。"
        f"一共买了多少袋牛肉？"
    )
    inferred_bags = sample.total_price / (Decimal(sample.pounds_per_bag) * sample.price_per_pound)
    return {
        "strategy": "fobar",
        "original_question": sample.question,
        "query": query,
        "response": (
            f"每袋价格 = {sample.pounds_per_bag} x {sample.price_per_pound} = "
            f"{Decimal(sample.pounds_per_bag) * sample.price_per_pound} 美元\n"
            f"袋数 = {sample.total_price} / "
            f"{Decimal(sample.pounds_per_bag) * sample.price_per_pound} = {inferred_bags}"
        ),
        "target": str(inferred_bags),
    }


def self_verify(sample: MathSample, claimed_total: Decimal) -> Dict[str, Any]:
    computed_total = sample.total_price
    verdict = computed_total == claimed_total
    query = (
        f"有人说: 买 {sample.bags} 袋牛肉，每袋 {sample.pounds_per_bag} 磅，"
        f"每磅 {sample.price_per_pound} 美元，总价是 {claimed_total} 美元。"
        f"这个说法正确吗？请先计算，再给出 True 或 False。"
    )
    return {
        "strategy": "self_verify",
        "original_question": sample.question,
        "query": query,
        "response": (
            f"重新计算总价 = {sample.bags} x {sample.pounds_per_bag} x {sample.price_per_pound} = {computed_total}\n"
            f"声明值 = {claimed_total}\n"
            f"判断结果 = {verdict}"
        ),
        "target": verdict,
    }


def validate_record(record: Dict[str, Any]) -> None:
    if record["strategy"] in {"answer_aug", "paraphrase"}:
        assert Decimal(record["target"]) == Decimal("110.0")
    elif record["strategy"] == "fobar":
        assert Decimal(record["target"]) == Decimal("5")
    elif record["strategy"] == "self_verify":
        assert isinstance(record["target"], bool)
    else:
        raise ValueError(f"Unknown strategy: {record['strategy']}")


def main() -> None:
    sample = MathSample(
        question="买 5 袋牛肉，每袋 4 磅，每磅 5.5 美元，总价是多少？",
        bags=5,
        pounds_per_bag=4,
        price_per_pound=Decimal("5.5"),
    )

    records: List[Dict[str, Any]] = [
        answer_augmentation(sample),
        paraphrase(sample),
        fobar(sample),
        self_verify(sample, Decimal("110.0")),
        self_verify(sample, Decimal("108.0")),
    ]

    for record in records:
        validate_record(record)

    print(json.dumps([asdict(sample), *records], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
```

这段代码体现了三个工程上容易忽略、但实际很关键的点。

第一，`Decimal` 比 `float` 更稳。因为很多价格类样本涉及小数，直接用浮点数会出现比较误差，导致本来正确的验证样本被误判为错误。

第二，不同策略必须保留显式标签。`answer_aug` 输出的是求值链，`self_verify` 输出的是真假判断，它们的目标格式根本不同。如果不保留 `strategy` 字段，后续训练时就很难区分任务类型。

第三，增强样本必须带上 `original_question`。这是后面去重、统计覆盖率、回溯教师模型错误的基础键。很多玩具实现只保留新题面，结果训练集出了问题也查不回原始来源。

真实系统会复杂得多，通常包含以下步骤:

| 步骤 | 输入 | 输出 | 关键检查 |
| --- | --- | --- | --- |
| 读取基础样本 | 数据集 JSON/JSONL | 标准化题目对象 | 字段齐全、答案可解析 |
| 调用教师模型 | 原题 + prompt 模板 | 增强题与推理链 | 是否偏题、是否改坏条件 |
| 自动过滤 | 新样本 | 合格样本 | 答案一致性、可解性、去重 |
| 写入训练集 | 合格样本列表 | SFT 数据 | 保留策略标签与来源字段 |
| 微调训练 | MetaMathQA | 新模型参数 | 监控各策略收益与格式稳定性 |

一个简化的配置可以长这样:

```json
{
  "strategies": {
    "answer_aug": 0.4,
    "paraphrase": 0.3,
    "fobar": 0.2,
    "self_verify": 0.1
  },
  "max_new_samples_per_item": 8,
  "dedup_by_embedding": true,
  "verification": {
    "check_numeric_equivalence": true,
    "drop_incorrect_generation": true,
    "solveability_check_for_fobar": true
  }
}
```

训练阶段最重要的一点，是把策略标签显式保留下来。因为 `self_verify` 样本和普通求值样本的目标格式不同，一个输出数值，一个输出验证结论。如果流水线不区分任务类型，模型会学到混乱的输出分布。简单说就是: 它会不知道什么时候该输出 `110`，什么时候该输出 `True`。

如果把这条流水线压缩成一句工程判断，那就是: MetaMath 不是“生成更多文本”，而是“生成更多不同类型、且能自动校验的监督信号”。

---

## 工程权衡与常见坑

第一个常见坑是“把扩写当多样性”。答案增强和问题重述都很有用，但它们主要扩的是表达密度，不一定扩了推理空间。论文结果已经说明，只靠这两类策略，收益会较早饱和。对新手最实用的判断标准是: 如果新样本的未知数、求解方向、校验目标都没变，它多半不算强多样性。

第二个坑是“反向题生成得像原题，其实不可解”。例如把多个中间变量都抹掉，只留下总价，结果新题存在多个解。FOBAR 不是机械换未知数，而是要保证新题仍然可解，最好还是单解。否则模型学到的不是逆向推理，而是含糊题面下的随机猜测。

第三个坑是“自我验证样本写成了答案泄漏”。比如题面直接说“答案是 110，请解释为什么”，这会让模型学会顺着声明写解释，而不是独立验证。更好的写法是给出完整陈述，让模型先重算，再判断真假。验证任务的关键不在“解释得像不像”，而在“是否真的重算”。

第四个坑是数据污染。若训练集里存在与测试集高度近似的重述样本，分数会上涨，但那不代表推理能力真的变强。所以需要用嵌入距离、规则去重和模板多样化共同控制。实践里经常见到一种假提升: 题面换了几个词，但数字、变量关系、句式骨架几乎没变，模型实际上只是记住了模板。

第五个坑是生成成本。四种策略叠加后，token 成本和人工抽检成本都会明显上升。对 7B 级模型做 SFT 时，真正的瓶颈常常不是训练，而是前面的清洗与验证。训练本身往往是线性的，难的是把错误样本挡在训练集外面。

第六个坑是格式漂移。比如同样是 `self_verify`，有的样本输出 `True/False`，有的输出“正确/错误”，有的先写长推理再给结论。格式如果不统一，模型会把“任务本身”和“输出风格差异”混在一起学，最终既不稳也不好评估。

下面这个表可以帮助判断投入方向:

| 策略组合 | 生成成本 | 多样性收益 | 常见风险 | 适合场景 |
| --- | --- | --- | --- | --- |
| 仅答案增强 | 低 | 低到中 | 很快饱和 | 预算很紧、先做基线 |
| 答案增强 + 重述 | 中 | 中 | 语义变了但推理没变 | 提升表述鲁棒性 |
| 加入 FOBAR | 中到高 | 高 | 新题可能不可解 | 需要逆向求解能力 |
| 加入 Self-Verification | 中 | 高 | 容易写成答案泄漏 | 需要稳定验算与纠错 |
| 全策略组合 | 高 | 最高 | 清洗成本大 | 追求整体上限 |

真实工程例子可以看财经建模问答。比如原题是“已知收入、成本、税率，求净利润”。只做答案增强，模型会更会写步骤；加入 FOBAR 后，它还能处理“已知净利润和收入，反推成本”；加入 Self-Verification 后，它还能检查分析师给出的利润结论是否与财务假设一致。对业务来说，后两者的价值通常比“把步骤写漂亮”更大。

工程上可以把四种策略的决策顺序压缩成一个简单判断表:

| 先问什么 | 如果答案是“是” | 优先策略 |
| --- | --- | --- |
| 现有模型是不是只会熟题顺推 | 是 | 先加 FOBAR |
| 模型是不是经常算对过程但给错结论 | 是 | 先加 Self-Verification |
| 模型是不是一换问法就掉点 | 是 | 再加问题重述 |
| 模型是不是经常少写步骤、跳步 | 是 | 再加答案增强 |

---

## 替代方案与适用边界

MetaMath 不是唯一方案。只要一种增强方法能稳定地产生“新的推理视角”，它就可能替代其中一部分策略。判断标准不是名字，而是它是否真的改变了模型的训练目标。

常见替代路线有三类。

第一类是 chain-of-thought paraphrasing，也就是“推理链重述”。白话讲是答案步骤本身也换表达，而不是只换题面。它能增加语言覆盖，但如果逻辑顺序不变，效果往往接近答案增强。也就是说，它更像“把同一条路换种说法再走一遍”，而不是“换一条路”。

第二类是 tool-augmented synthesis，也就是“工具辅助生成”。白话讲是用符号计算器、程序执行器或定理验证器先算出中间结果，再让模型围绕这些结果写题。这类方法正确率更高，但工程复杂度也更高。优点是样本更可靠，缺点是系统搭建成本高、覆盖题型也可能受工具能力限制。

第三类是 domain-specific inversion，只在某个垂直领域做反向构题。比如教育题库、财务报表、供应链优化，专门把常见顺推题转换成逆推题和校验题。这通常比全量四策略更省钱，因为它只在最有价值的变量上做反转。

适用边界也要说清楚:
- 如果任务主要是事实问答，不是多步推理，MetaMath 这套收益会很有限。
- 如果原始数据本身已经极其多样，额外增强的边际收益会下降。
- 如果教师模型不够强，生成样本噪声过高，增强可能比原始数据更差。
- 如果场景强依赖严格符号正确性，例如竞赛证明题，仅靠自然语言生成增强并不够。
- 如果场景输出格式高度刚性，例如必须产出可执行程序或定理证明脚本，MetaMath 只能作为辅助数据源，不能替代结构化监督。

面向零基础工程师，最实用的落地建议是: 不一定一开始就做全套。若预算有限，可以优先做 `FOBAR + Self-Verification`。原因很直接，这两者最容易真正改变任务分布，而不仅是换一种说法。前者扩展“倒着求”的能力，后者补上“算完检查”的能力，这两种变化通常比纯重述更接近真实业务需求。

| 方案 | 预期收益 | 成本 | 适用边界 |
| --- | --- | --- | --- |
| 只做重述 | 低到中 | 低 | 只想提升问法鲁棒性 |
| 只做 FOBAR | 中到高 | 中 | 逆向求解很多的任务 |
| FOBAR + Self-Verification | 高 | 中到高 | 教育、金融、运营分析 |
| 全策略 | 最高 | 高 | 追求通用数学推理上限 |
| 工具辅助增强 | 高且更稳 | 很高 | 高正确率、可接受复杂流水线 |

如果把替代方案和 MetaMath 放在一起比较，可以得到一个更直接的工程判断:

| 路线 | 最强项 | 最弱项 | 更适合什么团队 |
| --- | --- | --- | --- |
| MetaMath 四策略 | 低门槛扩展任务分布 | 依赖教师模型质量 | 先做 SFT、追求快迭代的团队 |
| 工具辅助增强 | 正确率高、可自动验证 | 系统复杂 | 有工程资源、重正确性的团队 |
| 垂直领域反向构题 | 性价比高 | 通用性弱 | 已知业务题型稳定的团队 |
| 纯重述/扩写 | 实现最简单 | 上限低 | 只做基线验证的团队 |

---

## 参考资料

| 资料 | 作用 |
| --- | --- |
| MetaMath ICLR 2024 论文 / 项目页 | 方法定义、四种增强策略、`dgain` 公式、7B 模型在 GSM8K 66.5% 与 MATH 19.8% 的结果 |
| MetaMathQA Hugging Face 数据集卡 | 确认 MetaMathQA 训练集规模为 395K，并可直接观察 `AnsAug`、`FOBAR`、`SV` 等样本格式 |
| TensorFlow Datasets 的 GSM8K 数据卡 | 说明 GSM8K 训练集规模为 7,473 条 |
| MATH 数据集论文与公开数据说明 | 说明 MATH 是竞赛风格数学数据集，训练集规模约 7,500 条 |
| MetaMath GitHub 仓库 | 查看公开训练脚本、模型结果与后续发布的衍生模型 |

把这些资料对应到文章里的结论，可以看到一个很清楚的链条:
- 论文给出方法与主实验结果
- 数据集卡证明“395K 规模”不是口头描述，而是实际公开数据
- GSM8K 与 MATH 的数据卡说明“约 1.5 万种子样本”这个起点是怎么来的
- 项目仓库则补上了工程实现入口，方便把论文结论落到可复现实验上
