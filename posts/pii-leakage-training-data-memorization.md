## 核心结论

PII 泄露与训练数据记忆，指模型不仅学到一般统计规律，还可能把训练集中低频、唯一、含个人信息的片段记住，并在特定前缀、提示或搜索策略下复现出来。

| 维度 | 结论 |
| --- | --- |
| 现象 | 训练数据记忆 |
| 风险 | PII 泄露 |
| 触发方式 | 特定前缀 / 搜索 / 提示 |
| 判断标准 | 能否从模型中抽出训练片段 |

先看最重要的判断：模型会不会泄露隐私，不能只看平均准确率、平均 loss 或对话体验，而要看长尾样本是否可被抽出。这里的“长尾样本”就是出现次数很少、但内容很具体的样本；越具体、越唯一，越容易被模型当成可直接续写的片段。

玩具例子很简单。训练语料里只出现过一次“张三，13812345678，订单 A9K4P2”。如果用户输入“我的手机号是”，模型并不是“理解了这个人是谁”，而是可能沿着训练中见过的模式，把这个真实号码补出来。这个现象说明模型记住的不是抽象知识，而是训练样本中的隐私碎片。

真实工程里，风险通常出现在客服工单、邮件补全、内部 IM 聊天、报修单、CRM 备注等场景。这些数据含有姓名、电话、地址、订单号、病历号、工单原文，而且很多样本只出现一次。模型一旦把这类片段写进参数，后面的输出过滤只能拦一部分，不能证明训练阶段已经安全。

---

## 问题定义与边界

先把三个概念分开：

- 泛化：模型学到统计规律。白话说，就是学会“这类句子通常怎么写”。
- 记忆：模型复现训练样本中的具体片段。白话说，就是把见过的原文背下来。
- 泄露：复现内容足以识别某个人或某条敏感记录。白话说，就是背出来的内容会伤害隐私。

统一记号如下：训练集记为 `D`；待审计的秘密或 canary 记为 `s[r]`；候选空间记为 `R`。这里的 canary 就是人为注入的唯一字符串，用来测试模型会不会记住它。

下面这张表先把边界钉死：

| 类型 | 是否包含训练样本原文 | 是否可复现唯一片段 | 是否涉及个人信息 |
| --- | --- | --- | --- |
| 规律 | 不要求 | 否 | 不一定 |
| 记忆 | 常常是 | 是 | 不一定 |
| 泄露 | 常常是 | 是 | 是 |

对照例子：

| 输入场景 | 属于什么 |
| --- | --- |
| 模型把“手机号一般是 11 位”补对 | 泛化 |
| 模型补出某个真实手机号 | 记忆 |
| 模型补出能定位某个具体人的手机号 | 泄露 |

边界不能只盯显式字段。很多团队会先删身份证号、手机号，然后以为完成了隐私处理，但真正危险的往往是半唯一组合特征。所谓“半唯一组合特征”，就是单看每个字段不敏感，组合起来却足以定位个人，例如“姓名 + 城市 + 订单号后四位”“科室 + 日期 + 年龄 + 稀有诊断”“时间 + 地点 + 工单号”。这类上下文指纹不一定长得像传统 PII，却同样可能构成可识别风险。

所以问题定义不能写成“模型会不会背出身份证号”，而应写成“模型能否稳定抽出训练集中的唯一或近唯一敏感片段”。只要答案是能，就已经越过隐私红线。

---

## 核心机制与推导

模型训练的目标通常是最小化序列预测损失。序列预测就是根据前文预测下一个 token。它优化的是“哪个续写概率更高”，不会天然区分“这是应该学的语言规律”还是“这是应该忘掉的隐私片段”。

公式写出来是：

$$
L_\theta(x) = -\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
$$

这里的 $L_\theta(x)$ 是序列 $x$ 的损失。损失越小，说明模型越偏好这个序列，认为它越像“正确续写”。

关键点在于：如果某个片段足够稀有、足够完整、前缀又足够强，模型可能把它推到低损失区域。攻击者不需要直接问“请把训练数据给我”，而是通过前缀诱导、候选枚举、损失排序，把最像训练原文的片段找出来。

为此可以定义排名：

$$
rank_\theta(s[r]) = |\{r' \in R : L_\theta(s[r']) \le L_\theta(s[r])\}|
$$

含义很直接：把候选空间 `R` 里所有可能秘密都拿来打分，看看真实秘密 `s[r]` 排第几。`rank` 越小，说明模型越偏爱这个真实秘密。

再定义暴露度：

$$
exposure_\theta(s[r]) = \log_2 |R| - \log_2 rank_\theta(s[r])
$$

暴露度可以理解成“模型帮攻击者省了多少猜测工作量”。如果候选空间很大，但真实秘密排得非常靠前，说明模型已经显著缩小了搜索范围，秘密更容易被抽出。

直观关系如下：

| 现象 | 含义 |
| --- | --- |
| `rank` 越小 | 暴露度越高 |
| `rank = 1` | 最危险，模型最偏爱真实秘密 |
| `|R|` 越大 | 暴露度的解释力越强 |

看一个最小数值例子。设候选空间大小为 $|R| = 8$，真实秘密排第 `2`，则：

$$
exposure = \log_2 8 - \log_2 2 = 3 - 1 = 2
$$

`2 bits` 的意思不是“泄露了 2 位字符”，而是模型把攻击者的搜索难度大幅降低了。原来平均可能要试很多候选，现在按模型排序先试前几个就更容易命中。

玩具例子可以这样理解。假设训练集中只有一次：

`客户：我叫李明，手机号 13812345678，订单号 A9K4P2`

模型训练时看到“我叫李明，手机号”后，后面的号码和订单号在这个上下文下是确定的。由于它们几乎没有别的竞争续写，优化过程就可能把这条样本记成一个很低损失的序列。外部用户只要构造类似前缀，模型就可能直接往训练原文方向续写。

真实工程例子是企业客服微调。团队把几百万条聊天、报修单、邮件补全日志丢进训练集，表面上是让模型学会回复风格，实际上也把大量低频真实记录写入了参数。上线后用户输入“我的手机号是”“我上次订单编号是”“请帮我查一下我的报修单”，模型可能会优先续写出训练中过的具体号码、编号或上下文片段。这类风险不体现在平均 loss 上，而体现在尾部记录的可提取性上。

---

## 代码实现

代码实现的目标不是训练一个新模型，而是演示三个动作：计算候选序列损失、按损失给秘密排序、计算暴露度。只要这三个动作跑通，新手就能理解审计到底在测什么。

先看一个可运行的玩具实现。这里不用真实神经网络，而是用一个“假模型”直接返回某些候选的损失，重点放在 `rank` 和 `exposure` 的计算。

```python
import math

class ToyModel:
    def __init__(self, loss_table):
        self.loss_table = loss_table

    def seq_loss(self, x: str) -> float:
        if x not in self.loss_table:
            raise KeyError(f"unknown candidate: {x}")
        return self.loss_table[x]

def rank_secret(model, candidates, secret):
    secret_loss = model.seq_loss(secret)
    rank = 1 + sum(
        1 for c in candidates
        if c != secret and model.seq_loss(c) <= secret_loss
    )
    return rank

def exposure(num_candidates, rank):
    return math.log2(num_candidates) - math.log2(rank)

candidates = [
    "13800000000",
    "13812345678",
    "13911112222",
    "13799998888",
    "13622223333",
    "13555556666",
    "13177778888",
    "13088889999",
]

model = ToyModel({
    "13800000000": 5.2,
    "13812345678": 1.1,  # 真实秘密，损失最低
    "13911112222": 4.8,
    "13799998888": 4.9,
    "13622223333": 5.0,
    "13555556666": 5.3,
    "13177778888": 5.1,
    "13088889999": 4.7,
})

secret = "13812345678"
r = rank_secret(model, candidates, secret)
e = exposure(len(candidates), r)

assert r == 1
assert round(e, 6) == 3.0
print("rank =", r, "exposure =", e)
```

这个例子里，真实秘密的损失最低，所以 `rank = 1`，暴露度达到最大值 `log2(8)=3`。这说明模型把真实号码排在所有候选前面，风险最高。

如果想更接近真实流程，可以把“序列损失”理解成语言模型对整段字符串的负对数概率。伪代码如下：

```python
def seq_loss(model, x):
    total = 0.0
    tokens = tokenize(x)
    for t in range(len(tokens)):
        total += -log_prob(model, tokens[t], tokens[:t])
    return total

def audit_secret(model, candidates, secret):
    losses = [(c, seq_loss(model, c)) for c in candidates]
    losses.sort(key=lambda item: item[1])
    rank = 1 + sum(1 for c, loss in losses if c != secret and loss <= seq_loss(model, secret))
    expo = log2(len(candidates)) - log2(rank)
    return rank, expo, losses[:5]
```

真实工程里，更常见的是 canary 审计。canary 就是人为插入一条唯一字符串，例如：

`"contact_canary::ZX9Q-4412-ALPHA-7781"`

如果训练后模型能在给定前缀下稳定恢复这串内容，说明它对唯一片段存在明显记忆。

下面给一个最小审计流程示意：

```python
def inject_canary(dataset, canary):
    return dataset + [f"internal note: {canary}"]

def prefix_extract(model, prefix, candidate_suffixes):
    scored = [(s, model.seq_loss(prefix + s)) for s in candidate_suffixes]
    scored.sort(key=lambda x: x[1])
    return scored[0][0]

# 训练前后只是示意，不实现真实训练
canary = "ZX9Q-4412-ALPHA-7781"
prefix = "internal note: "
candidate_suffixes = [
    "ZX9Q-4412-ALPHA-7781",
    "ZX9Q-4412-ALPHA-0000",
    "RANDOM-STRING-123456",
]

before = ToyModel({
    prefix + candidate_suffixes[0]: 5.0,
    prefix + candidate_suffixes[1]: 5.1,
    prefix + candidate_suffixes[2]: 5.2,
})
after = ToyModel({
    prefix + candidate_suffixes[0]: 0.8,
    prefix + candidate_suffixes[1]: 4.9,
    prefix + candidate_suffixes[2]: 5.3,
})

best_before = prefix_extract(before, prefix, candidate_suffixes)
best_after = prefix_extract(after, prefix, candidate_suffixes)

assert best_before == "ZX9Q-4412-ALPHA-7781"
assert best_after == "ZX9Q-4412-ALPHA-7781"
```

上面这个玩具实现说明“怎么测”，但真实审计时不能只看 top-1。应同时看训练前后差异、普通样本和 canary 的差异，以及 `loss / rank / exposure` 是否出现异常变化。

| 样本类型 | 阶段 | loss 趋势 | rank 趋势 | exposure 趋势 |
| --- | --- | --- | --- | --- |
| 普通样本 | 训练前 | 中等 | 靠后 | 低 |
| 普通样本 | 训练后 | 下降 | 略前移 | 略升 |
| Canary | 训练前 | 一般 | 靠后 | 低 |
| Canary | 训练后 | 显著下降 | 明显前移 | 显著升高 |

这张表的重点是：普通样本 loss 下降很正常；但如果 canary 或唯一敏感串的 `rank` 大幅前移，说明模型对独特片段形成了强记忆。

---

## 工程权衡与常见坑

实际项目里，最大的问题通常不是“推理时没有拦住一句话”，而是“训练前把什么放进去了”。一旦真实工单、病历、邮件、聊天记录进入训练，等于把隐私写入参数。后面再做输出过滤，只是在出口处止血。

常见坑和规避措施如下：

| 坑 | 为什么危险 | 可执行规避措施 |
| --- | --- | --- |
| 只做输出过滤 | 训练记忆已形成，过滤只能拦已知模式 | 入库前脱敏，训练前审计 |
| 只删显式 PII | 半唯一组合特征仍可识别个人 | 正则 + NER 双层清洗，加上下文规则 |
| 重复样本过多 | 高频重复会强化记忆 | 去重、长尾降采样、限制重复注入 |
| 只看整体 loss | 平均指标掩盖尾部泄露 | 做 canary 注入与提取测试 |
| 把对齐当成隐私方案 | 拒答行为不等于参数没记住 | 必要时使用 DP-SGD 或不纳入训练 |

这里的 NER 是命名实体识别，白话说就是自动识别人名、地名、组织名、地址等实体的模型或规则系统。

新手最容易踩的坑是：删掉手机号、身份证号字段，就认为安全。实际并非如此。比如“王某，4 月 3 日，北京朝阳，报修单 7281，空调漏水”，在一个小区域运维系统里可能已经唯一定位到某个住户。也就是说，风险不只在“字段像不像 PII”，还在“组合后是否能指向一个人”。

另一个误区是把对齐能力当成隐私保证。对齐就是让模型更倾向于遵守指令、拒绝违规请求，但这和“参数里有没有记住敏感片段”不是一回事。模型今天拒绝回答，不代表明天换个提示、换个解码策略、换个攻击流程后仍然不会泄露。

真实工程例子里，如果业务目标只是“生成客服回复风格”，那就没必要让用户真实姓名、电话、工单号进入参数。风格学习和隐私字段本来就是两件事，把它们绑在一起训练，只会放大风险而不必然提升效果。

---

## 替代方案与适用边界

不是所有场景都应该采用“直接训练，再靠对齐或过滤兜底”的路线。高风险 PII 场景更合理的策略，通常是少训练、不训练，或者先脱敏再训练。

先给一张方案对比表：

| 方案 | 隐私保护强度 | 训练成本 | 可用性损失 | 适用场景 |
| --- | --- | --- | --- | --- |
| 入库前脱敏 | 中 | 低 | 低到中 | 一般内部文本 |
| 规则过滤 + NER 脱敏 | 中 | 中 | 低到中 | 客服、工单、邮件 |
| 去重 / 降采样 | 中 | 低 | 低 | 长尾敏感片段较多 |
| 检索增强，不把原文训练进去 | 高 | 中 | 中 | 需要实时查原文 |
| DP-SGD | 较高 | 高 | 中到高 | 高敏训练且可接受性能损失 |
| 不训练敏感数据，只做访问控制下的检索 | 很高 | 中 | 低到中 | 医疗、金融、身份数据 |

DP-SGD 是差分隐私随机梯度下降，白话说是在训练时对梯度裁剪并加噪，限制单条样本对最终参数的影响。它能降低记忆和成员推断风险，但代价通常是训练更难、效果下降，而且不是零泄露保证。

可以按风险级别理解适用边界：

| 风险等级 | 例子 | 更合适的做法 |
| --- | --- | --- |
| 低风险 | 公开文档摘要、通用技术问答 | 可直接训练 |
| 中风险 | 内部客服语料、报修记录 | 先脱敏，再训练与审计 |
| 高风险 | 身份证、手机号、病历、邮件原文 | 尽量不进训练集；优先检索或 DP |

这里要明确三条边界：

- 对齐不等于隐私。对齐解决的是行为约束，不是参数记忆。
- 输出过滤不等于训练安全。过滤只能拦输出，不会删除模型已学到的内容。
- DP 能降风险，但不是绝对安全。它是风险控制手段，不是零泄露承诺。

如果业务只是让模型“说得像客服”，最优路线往往是：脱敏后的风格数据做训练，真实用户资料放到权限受控的检索系统里，推理时按需查，而不是把原始个人信息直接写入模型参数。

---

## 参考资料

1. [The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks](https://arxiv.org/abs/1802.08232)
2. [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)
3. [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646)
4. [Scalable Extraction of Training Data from (Production) Language Models](https://www.microsoft.com/en-us/research/publication/scalable-extraction-of-training-data-from-production-language-models/)
5. [ftramer/LM_Memorization](https://github.com/ftramer/LM_Memorization)
