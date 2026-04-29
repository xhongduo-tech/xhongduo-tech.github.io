## 核心结论

模型水印的鲁棒性评测，检验的不是“原文能不能被识别”，而是“文本经过真实流转后，水印还能不能被稳定识别”。这里的“鲁棒性”可以先理解成：系统遇到扰动时，性能不至于立刻崩掉的能力。

对大语言模型文本水印来说，真正重要的场景几乎都不是“用户原样复制模型输出”。更常见的是改写、摘要、翻译、删句、重新采样，甚至专门的去水印攻击。只要输出内容会被二次编辑，评测重点就必须从 clean text，也就是“未被改动的原始文本”，转到 attacked text，也就是“经过扰动后的文本”。

因此，鲁棒性评测至少要同时回答两个问题：

1. 攻击后还能检出多少，也就是检出率还能保留多少。
2. 没有水印的文本会不会被误判，也就是误报率是否仍可控。

一个只在原文上表现很好的水印系统，不等于工程上可用。只有在固定误报率约束下，经过常见扰动后仍能保持较高检出率，它才更接近“可落地的归属工具”。

下面这张表可以先把结论压缩成一个最小判断框架：

| 攻击类型 | 常见强度 | 典型影响 | 关注指标 | 工程判断 |
| --- | --- | --- | --- | --- |
| 轻度改写 | 同义替换、调语序 | 信号小幅衰减 | TPR 下降幅度 | 通常还能用 |
| 压缩摘要 | 删句、缩短篇幅 | 有效 token 变少 | TPR 与长度分桶 | 中风险 |
| 翻译再回译 | 跨语言重写 | 词级信号明显破坏 | TPR、AUC | 高风险 |
| 重采样 | 温度、top-p 改变 | 分布偏差被稀释 | 检测分数分布 | 中高风险 |
| 对抗攻击 | 专门去水印 | 有针对性削弱统计信号 | 最坏情况 TPR | 必测 |

若把攻击函数记成 $attack(\cdot)$，把检测器记成 $D(\cdot)$，一个最简鲁棒性定义可以写成：

$$
Robustness(a) = P(D(attack_a(x)) = 1 \mid x \text{ 含水印})
$$

这个式子强调的不是单个样本，而是“在某类攻击下的整体检出概率”。

---

## 问题定义与边界

本文讨论的是 LLM 文本水印，也就是大语言模型生成文本中的可检测信号，不讨论图片隐写、水印图层，或对训练数据打标签的溯源机制。这里的“水印”不是把某个字符串硬塞进输出，而是通过调整 token 采样分布，给整段文本引入统计上的偏好。

“token”可以先理解成模型生成时处理的最小文本单元，可能是一个字、一个词，或者词的一部分。很多文本水印方法的思路是：在每一步生成时，让某一类 token 被稍微偏好一些。单看一步几乎看不出来，但在足够长的文本上，整体偏差可以被统计检测器识别。

评测时要先分清三个目标，它们不是同一个问题：

| 对象 | 白话解释 | 评测目标 | 是否本文重点 |
| --- | --- | --- | --- |
| 可检出性 | 原文上能不能测出来 | clean TPR、AUC | 不是重点，只是起点 |
| 抗攻击性 | 被改动后还能不能测出来 | attacked TPR、鲁棒曲线 | 是 |
| 工程可用性 | 线上能不能稳定运行 | 误报率、延迟、长度要求 | 是 |

其中最基本的二分类指标是：

$$
TPR = \frac{TP}{TP + FN}, \qquad FPR = \frac{FP}{FP + TN}
$$

TPR 是 true positive rate，可以白话理解成“有水印文本里，有多少被成功抓到”；FPR 是 false positive rate，可以理解成“没水印文本里，有多少被错抓”。

边界也要说清楚。下面这些属于本文评测范围：

| 场景 | 是否纳入 |
| --- | --- |
| 同义改写、压缩、翻译、删句 | 是 |
| 解码参数变化，如温度、top-p | 是 |
| 文本长度变化对检测的影响 | 是 |
| 人工或模型发起的对抗式去水印 | 是 |
| 训练阶段如何嵌入水印 | 不是重点 |
| 平台日志存证、身份签名机制 | 作为替代方案比较，不做主线 |
| 图像、音频、多模态水印 | 不纳入 |

一个常见误区是把“鲁棒性”理解成“攻击后分数不下降”。这不对。攻击后分数几乎一定会下降，关键是下降后是否还在可判定区间。换句话说，鲁棒性评测不是要求信号不损失，而是要求损失后仍足够可检测。

---

## 核心机制与推导

大多数文本水印不是内容级标记，而是分布级标记。所谓“分布级”，就是它改的是生成概率，而不是明文插入某个标记词。检测器也不是找固定短语，而是检查整段文本里是否出现了统计上“不太自然的偏好”。

一种典型思路是 green list watermark。可以把 green list 先理解成：在当前上下文下，被算法临时标成“绿色候选”的一部分 token 集合。生成时，系统会稍微提高这些绿色 token 的概率。这样做的结果是，含水印文本里绿色 token 的命中数，会比自然生成更高。

若把文本记成 $x = (t_1, t_2, \dots, t_T)$，可以写一个通用打分函数：

$$
S(x) = \sum_{i=1}^{T} s(t_i, context_i)
$$

这里 $s(t_i, context_i)$ 是第 $i$ 个 token 在当前上下文下对应的局部水印得分。检测器最后做的事情很简单：

$$
D(x) = \mathbb{1}[S(x) > \tau]
$$

也就是当总分 $S(x)$ 大于阈值 $\tau$ 时，判定文本含水印。$\tau$ 不是随便拍脑袋定的，它要在目标误报率上标定，比如要求 FPR 不超过 $10^{-5}$ 或 $1\%$。

玩具例子可以最直观说明问题。

假设一段 200 token 的文本，在 clean 条件下检测分数是 $S=12$，阈值是 $\tau=8$，于是能通过检测。现在我们对文本做一次强改写：合并句子、替换表达、删除冗余。改写后核心语义还在，但很多原 token 被换掉，分数变成 $S=7$。此时不是水印算法“失效”，而是统计信号被扰动削弱到阈值以下，所以检测失败。

这个现象可以写成鲁棒性曲线：

$$
R(a) = P(S(attack_a(x)) > \tau)
$$

其中 $a$ 表示某类攻击及其强度。$R(a)$ 越高，说明水印在该攻击下越稳。工程上真正要画的，往往不是单个点，而是“攻击强度 - 检出率”曲线。例如改写比例从 10% 提到 40%，TPR 怎么掉；长度从 800 token 降到 80 token，分数分布怎么收缩。

为什么长度很关键？因为很多检测器依赖大数统计。样本越长，累计偏差信号越明显；样本越短，方差越大，检测器更难把“含水印”和“未含水印”分开。这也是为什么短摘要、标题、聊天短回复通常更难做可靠水印判定。

真实工程例子更能说明边界。假设一个内容平台接入了带水印的生成模型，平台希望在侵权投诉或内容溯源时判断“这段文章是否来自本站模型”。但真实链路不是“一次生成，原样发布”，而是：

1. 模型生成初稿。
2. 编辑工具自动改写一轮。
3. 用户把内容翻译成另一种语言。
4. CMS 系统为列表页生成摘要版。
5. 发布后又被第三方转载并二次编辑。

如果评测只在第 1 步原文上做，线上结果一定过于乐观。真正的鲁棒性评测应该按链路逐段测，或者把整条处理链当成复合攻击测。因为对业务而言，水印要么能穿过这条链路，要么就没有归属价值。

---

## 代码实现

工程实现的重点不是训练模型，而是把评测流程做成可重复实验。一个最小评测系统可以拆成三层：

1. 样本层：加载有水印和无水印文本。
2. 攻击层：对文本施加改写、截断、翻译、重采样等扰动。
3. 检测层：输出分数、套阈值、汇总 TPR/FPR/AUC。

实验配置通常至少要记录下面几项：

| attack_type | attack_strength | sample_size | threshold | 说明 |
| --- | --- | --- | --- | --- |
| truncate | 0.2 | 500 | 8.0 | 截断 20% |
| paraphrase | 0.3 | 500 | 8.0 | 中等改写 |
| translate_back | zh-en-zh | 500 | 8.0 | 回译 |
| resample | temp=1.2 | 500 | 8.0 | 高温重写 |

下面给一个可运行的最小 `python` 例子。它不是论文级检测器，而是一个“玩具检测框架”：把某些词看成“绿色词”，含水印文本会更偏向这些词；攻击函数通过删句、同义替换和截断来削弱信号。代码的目标是展示评测流程，而不是复现某篇具体论文。

```python
from dataclasses import dataclass
from typing import List, Dict
import re

GREEN_WORDS = {"system", "model", "agent", "policy", "token", "trace"}

@dataclass
class Sample:
    text: str
    watermarked: bool

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())

def detect_watermark_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    green_hits = sum(t in GREEN_WORDS for t in tokens)
    base_rate = 0.10
    observed_rate = green_hits / len(tokens)
    # 用“超出基线的比例”作为玩具分数
    return len(tokens) * max(0.0, observed_rate - base_rate)

def detect_watermark(text: str, tau: float = 2.0) -> bool:
    return detect_watermark_score(text) > tau

def apply_attack(text: str, attack_type: str) -> str:
    if attack_type == "clean":
        return text
    if attack_type == "truncate":
        tokens = tokenize(text)
        return " ".join(tokens[: max(1, len(tokens) // 2)])
    if attack_type == "paraphrase":
        repl = {
            "system": "platform",
            "model": "engine",
            "agent": "assistant",
            "policy": "rule",
            "token": "unit",
            "trace": "record",
        }
        tokens = tokenize(text)
        return " ".join(repl.get(t, t) for t in tokens)
    raise ValueError(f"unknown attack: {attack_type}")

def evaluate(samples: List[Sample], attack_type: str, tau: float = 2.0) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for sample in samples:
        attacked = apply_attack(sample.text, attack_type)
        pred = detect_watermark(attacked, tau=tau)
        if sample.watermarked and pred:
            tp += 1
        elif sample.watermarked and not pred:
            fn += 1
        elif (not sample.watermarked) and pred:
            fp += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"TPR": tpr, "FPR": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

samples = [
    Sample("system model agent policy token trace system model agent", True),
    Sample("model token trace policy agent system token trace model", True),
    Sample("human writing uses ordinary words without special bias", False),
    Sample("plain text from user discussion without watermark signal", False),
]

clean_metrics = evaluate(samples, "clean", tau=2.0)
truncate_metrics = evaluate(samples, "truncate", tau=2.0)
paraphrase_metrics = evaluate(samples, "paraphrase", tau=2.0)

assert clean_metrics["TPR"] >= truncate_metrics["TPR"]
assert clean_metrics["TPR"] >= paraphrase_metrics["TPR"]
assert clean_metrics["FPR"] <= 0.5

print(clean_metrics)
print(truncate_metrics)
print(paraphrase_metrics)
```

这段代码展示了三个关键点。

第一，评测必须先定义攻击函数 `apply_attack`。没有攻击集，就没有鲁棒性评测，只有“原文识别率测试”。

第二，检测与攻击要解耦。也就是 `detect_watermark` 不关心文本是怎么来的，它只接收文本并返回分数或布尔判断。这样你才能把同一检测器放到不同攻击条件下横向比较。

第三，结果不能只报一个平均分。至少要按攻击类型分别输出 TPR/FPR；更完整时还要加 AUC、长度分桶结果，以及不同阈值下的 ROC 或 PR 曲线。

如果把这套框架搬到真实项目里，`apply_attack` 通常会调用额外模型或规则流水线，`detect_watermark` 会实现具体论文里的打分统计，`evaluate` 则会把结果写入实验表、图表和报告系统。

---

## 工程权衡与常见坑

鲁棒性越强，往往越不是“白赚”的。很多水印方法靠提高信号强度换检测稳定性，但信号太强可能伤害文本自然度、任务效果或多样性。这里的“信号强度”可以白话理解成：系统在生成时对某类 token 偏好的力度。力度太弱，检不出来；力度太强，文本容易变形。

常见权衡可以压缩成下面这张表：

| 决策 | 好处 | 代价 | 适用情况 |
| --- | --- | --- | --- |
| 提高水印强度 | 检测更稳 | 可能损伤生成质量 | 高风险溯源场景 |
| 降低阈值 | 提升检出率 | 误报率上升 | 召回优先场景 |
| 只对长文本启用检测 | 结果更稳定 | 覆盖面变小 | 长文平台 |
| 做多攻击评测 | 结果更可信 | 成本和周期增加 | 上线前必须 |

常见坑也很固定。

| 常见坑 | 问题本质 | 规避方法 |
| --- | --- | --- |
| 只测原文 | 把“可检出”误当“鲁棒” | 必测改写、截断、翻译、回译 |
| 只看平均 TPR | 长尾失败被掩盖 | 看分位数、最坏情况、长度分桶 |
| 样本太短 | 方差太大，分不出分布 | 单独报告短文本边界 |
| 阈值固定不调 | FPR 在域迁移后漂移 | 按目标域重新标定阈值 |
| 攻击集太弱 | 结果虚高 | 加入模型改写和对抗式攻击 |
| 只测单语言 | 忽略跨语言破坏 | 测翻译链路和回译链路 |

一个很实际的坑是“训练域和评测域不一致”。例如你在英文新闻体文本上标定阈值，再拿去测中文社交平台短文本，分数分布可能完全变形。此时 clean TPR 下降，不一定是算法本身坏了，也可能是 tokenizer、长度分布、风格域都变了。

另一个坑是把“鲁棒性下降”解读成“水印没有意义”。这个结论太粗。更准确的说法应该是：在某个攻击强度、某个长度区间、某个误报率目标下，这套方法的有效性边界在哪里。工程上最需要的是边界，而不是抽象地说“行”或“不行”。

---

## 替代方案与适用边界

水印不是唯一溯源方案，也不应该被当成唯一方案。尤其在强改写、跨语言转述、短文本、多人多轮编辑场景下，文本水印的可靠性可能显著下降。

一个简单判断规则是：

$$
\text{if } attack\_strength > signal\_margin,\ \text{watermark becomes unreliable}
$$

这里的 `signal_margin` 可以白话理解成“水印分数离判定阈值还有多远的安全余量”。如果攻击造成的信号损失比这个余量还大，检测结果就容易翻转。

可以和几类替代方案对比：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 文本水印 | 不依赖外部日志，能离线检测 | 易受强改写和短文本影响 | 中长文本溯源 |
| 平台日志存证 | 证据链强，精度高 | 依赖平台侧保存与访问权限 | 平台内合规审计 |
| 数字签名/输出签名 | 篡改检测强 | 内容一改就失效 | 原样分发内容 |
| 内容指纹 | 可做相似检索 | 不是水印，难证明模型来源 | 查重、近重复检测 |
| 平台侧发布约束 | 可直接限制传播 | 只能覆盖自有平台 | 封闭生态治理 |

适用边界也很明确：

1. 短文本通常不适合做强结论，因为可累积统计信号太少。
2. 强改写与跨语言转述会明显削弱词级或 token 级水印。
3. 如果内容发布前天然会经历多轮人工编辑，单靠水印往往不够。
4. 如果业务对误报极其敏感，比如合规取证或法律争议，必须把水印与日志、签名或平台证据链组合使用。

所以更现实的工程策略不是“是否部署水印”，而是“把水印放在证据体系中的哪一层”。在很多系统里，水印适合做弱证据或预筛选：先用它缩小候选范围，再结合日志、账号行为、发布时间线索做最终判断。

---

## 参考资料

1. [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
2. [On the Reliability of Watermarks for Large Language Models](https://arxiv.org/abs/2306.04634)
3. [SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation](https://arxiv.org/abs/2310.03991)
4. [A Semantic Invariant Robust Watermark for Large Language Models](https://openreview.net/forum?id=6p8lpe4MNf)
5. [Who Wrote this Code? Watermarking for Code Generation](https://arxiv.org/abs/2402.10887)
6. [lm-watermarking: reference implementation for LLM watermarking experiments](https://github.com/jwkirchenbauer/lm-watermarking)
