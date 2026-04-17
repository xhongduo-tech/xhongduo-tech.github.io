## 核心结论

GSM8K 是一个面向小学到初中级数学文字题的基准，核心考察的是多步算术推理，不是公式背诵。所谓“多步推理”，白话说就是模型要先把题目里的量拆出来，再按顺序维护中间结果，最后得到一个数字。

它的官方评价口径很简单：只看最终答案是否完全一致，不要求推理过程逐步一致。准确率通常写成

$$
Accuracy=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat a_i=a_i)
$$

其中 $N$ 是评测样本数，$\hat a_i$ 是模型输出的最终答案，$a_i$ 是标准答案。对 Hugging Face 上的 `openai/gsm8k` 主配置，测试集是 1319 题，训练集是 7473 题，总计 8792 条；而 “8.5K” 是论文和社区对该数据集规模的习惯性简称。[OpenAI 论文](https://arxiv.org/abs/2110.14168)、[Hugging Face 数据卡](https://huggingface.co/datasets/openai/gsm8k)

截至 2026 年 3 月 27 日，LM Market Cap 的 GSM8K 排行页显示榜首为 Llama 3.1 405B，准确率 96.8%。这说明即使该基准已经接近饱和，exact-match GSM8K 仍然是对比不同提示策略、解码策略和开源模型稳定性的低成本基线。[LM Market Cap GSM8K](https://lmmarketcap.com/benchmarks/gsm8k)

玩具例子可以直接说明它怎么判分。题目是：“李老师有 7 个苹果，给 3 位学生每人 2 个后还剩几个？”正确推导是 $7-2\times3=1$。如果模型最终输出 `1`，即使中间解释写得很差，评分仍然算对；如果中间过程完全正确，但最后写成 `2`，评分仍然算错。

---

## 问题定义与边界

GSM8K 全称是 Grade School Math 8K。它由人工撰写的英文数学文字题组成，每题都配有自然语言解法和最终数字答案。题目通常需要 2 到 8 步运算，覆盖加减乘除、分数、小数、百分比、比例和简单代数。这里的“简单代数”，白话说就是最多把未知数当成一个待求数量，不会进入高等数学。[OpenAI 论文](https://arxiv.org/abs/2110.14168)、[OpenAI 介绍页](https://openai.com/index/solving-math-word-problems/)

| 维度 | 内容 |
| --- | --- |
| 任务形式 | 自然语言数学故事题 |
| 官方主配置规模 | 训练 7473 / 测试 1319 |
| 常见简称 | 8.5K |
| 常测能力 | 多步算术、数量提取、中间变量维护 |
| 标准评分 | 最终数值 exact match |
| 不覆盖的能力 | 长程证明、形式化定理证明、复杂几何 |

边界也要说清楚。GSM8K 测到的是“把文字题翻译成多步计算并稳定执行”的能力，不等于“通用数学能力”。一个模型在 GSM8K 上很高，不代表它能处理竞赛数学，也不代表它的推理过程一定可靠。

再看一个基础例子：“两个班共有 25 个学生，9 个去上体育课，其余每人发 2 本练习册，一共要多少本？”合理过程是先算剩余学生 $25-9=16$，再算练习册 $16\times2=32$。这类题的难点不是知识点，而是先减后乘的顺序不能乱。

---

## 核心机制与推导

GSM8K 之所以常被拿来比较提示策略，是因为它对“中间步骤有没有被稳定展开”非常敏感。

Chain-of-Thought，简称 CoT，白话说就是要求模型把脑内草稿显式写出来。它不是改模型参数，而是在提示里要求“分步思考”。经典论文表明，8-shot CoT 能显著提升 GSM8K 准确率。[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

Self-Consistency，白话说就是“不要只信一次回答，而是多采样几次，取多数票”。它的做法是采样多条推理链 $R_1,\dots,R_n$，抽取每条链的最终答案，然后选出现频次最高的答案：

$$
\hat a=\operatorname*{arg\,max}_{v}\sum_{j=1}^{n}\mathbf{1}(a_j=v)
$$

该方法在论文中对 GSM8K 报告了明显增益，提升幅度达到 17.9 个百分点。[Self-Consistency](https://arxiv.org/abs/2203.11171)

HoT，Hint of Thought，白话说就是把“分步想”再结构化一点，先拆成子问题，再把步骤写成更接近伪代码的形式。公开结果显示，HoT 在 GSM8K 零样本设置下相对标准 zero-shot CoT 有大幅提升，文中给出的数字是从 40.5% 提升到 67.8% 或 70.65% 左右，不同页面摘要有轻微差异，但方向一致。[OpenReview HoT](https://openreview.net/forum?id=aoOP62EuS9A)、[Emergent Mind HoT](https://www.emergentmind.com/papers/2305.11461)

| 方法 | 核心动作 | 优点 | 代价 |
| --- | --- | --- | --- |
| Direct Answer | 直接出答案 | 最便宜 | 易漏步、方差大 |
| CoT | 显式写步骤 | 易提升基础准确率 | token 成本上升 |
| HoT | 子问题 + 伪代码 | 结构更清晰、便于校验 | 提示更长 |
| Self-Consistency | 多次采样后投票 | 降低随机性 | 推理成本成倍增加 |
| Verifier | 独立打分选最好链 | 稳定性更强 | 系统最复杂 |

真实工程里常见做法不是只选一个策略，而是“CoT 负责展开，Self-Consistency 负责投票，Verifier 负责二次筛选”。OpenAI 在 GSM8K 论文里就强调了 verifier 思路：先生成很多候选解，再由判别器打分，选择最可信的一条。[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

---

## 代码实现

下面给一个最小可运行示例，模拟 GSM8K 常见的两件事：抽取最终答案，以及对多条推理结果做多数投票。

```python
import re
from collections import Counter

def extract_final_number(text: str) -> str:
    # 优先匹配 GSM8K 常见的 "#### 72" 结尾
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    # 否则退化为取最后一个数字
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        raise ValueError("no numeric answer found")
    return nums[-1]

def majority_vote(chains):
    answers = [extract_final_number(c) for c in chains]
    return Counter(answers).most_common(1)[0][0]

def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip() == gold.strip())

toy_problem = "李老师有7个苹果，给3位学生每人2个后还剩几个？"
chains = [
    "先算 2*3=6，再算 7-6=1。#### 1",
    "共有 3 个学生，每人 2 个，所以送出 6 个，剩下 1 个。#### 1",
    "先算 7-2=5，再算 5-3=2。#### 2",
]

pred = majority_vote(chains)
gold = "1"

assert pred == "1"
assert exact_match(pred, gold) == 1
assert exact_match("2", gold) == 0
```

这个例子表达的不是“怎么训练一个模型”，而是“怎么按 GSM8K 的判分口径把推理输出接起来”。真实评测管道通常是：

1. 给每道题构造 prompt。
2. 采样多条 CoT。
3. 从每条 CoT 提取最终数值。
4. 做多数投票，或者送入 verifier 重新排序。
5. 与标准答案做 exact match。

真实工程例子：如果你在做一个教育问答产品，想比较新提示词是否更稳，不需要先做微调。直接固定同一批 GSM8K 题目，分别跑 zero-shot、8-shot CoT、32 路 self-consistency，最后比较 exact-match 即可。这样能快速判断“问题是模型基础能力不够，还是提示与解码策略没调好”。

---

## 工程权衡与常见坑

最大的问题是，exact match 只看最终数字，不看过程。这会带来两个偏差。

第一，可能“过程错但答案对”。比如模型中间瞎编步骤，最后碰巧得到正确数字，分数照样记对。第二，可能“过程基本对但格式错”。如果输出 `72.`、`$72$`、`答案是 seventy-two`，没有做好归一化，工程实现可能误判为错。

| 问题 | 现象 | 开销 | 建议 |
| --- | --- | --- | --- |
| 只看 exact match | 过程质量不可见 | 低 | 额外记录推理链做抽检 |
| 单次 CoT | 结果波动大 | 低到中 | 用 self-consistency 降方差 |
| 多样本投票 | 成本随样本数线性上升 | 中 | 先试 16 或 32 路 |
| verifier 引入 | 系统复杂度高 | 高 | 用在高价值评测或回归测试 |
| 数据泄露 | 分数虚高 | 中 | 加去污染对照集 |

另一个常见坑是数据泄露。GSM8K 非常公开，很多模型可能在预训练或后训练阶段见过题目或近似改写。AntiLeak-Bench 相关综述明确指出，数学与代码类基准尤其要警惕 contamination，某些设定下 GSM8K 的准确率膨胀可以很明显。[AntiLeak-Bench](https://www.emergentmind.com/topics/antileak-bench)

所以，GSM8K 更适合做“同条件下的相对比较”，不适合单独当作“模型真的会数学”的充分证据。

---

## 替代方案与适用边界

如果你的目标是检查“模型是不是记住了题库”，就不能只看 GSM8K。应当增加污染更低或题型更广的对照集。

| 数据集/方法 | 覆盖范围 | 主要用途 | 泄露风险 |
| --- | --- | --- | --- |
| GSM8K | 小学到初中级文字题 | 快速比较提示与解码策略 | 中到高 |
| MGSM | 多语言版本算术题 | 看跨语言迁移 | 中 |
| MATH / MATH-500 | 更高难度数学题 | 看复杂数学推理 | 相对更低但更难 |
| AntiLeak-Bench 类框架 | 去污染评测流程 | 检查记忆干扰 | 设计目标就是降低泄露 |
| Verifier 流程 | 不是数据集，是方法 | 提高可靠性 | 不能替代去污染数据 |

如果模型在 GSM8K 上 96%，但换到更去污染的数据上明显下滑，优先怀疑记忆、模板化匹配或提示过拟合，而不是直接宣称“数学推理达到人类水平”。

因此，GSM8K 的最佳定位是：低成本、高复用、适合做版本回归和策略对比；不适合作为唯一的数学能力结论。

---

## 参考资料

- [Training Verifiers to Solve Math Word Problems, arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- [OpenAI: Solving math word problems](https://openai.com/index/solving-math-word-problems/)
- [Hugging Face 数据卡: openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models, arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- [Hint of Thought prompting: an explainable and zero-shot approach to reasoning tasks with LLMs, OpenReview](https://openreview.net/forum?id=aoOP62EuS9A)
- [LM Market Cap GSM8K Leaderboard，页面显示更新于 2026-03-27](https://lmmarketcap.com/benchmarks/gsm8k)
- [Emergent Mind: GSM8K Topic](https://api.emergentmind.com/topics/gsm8k)
- [Emergent Mind: GSM8K Verification Methods](https://www.emergentmind.com/topics/gsm8k-verification-96604d2f-ce0c-49bc-ad48-02c9eeea9bfe)
- [Emergent Mind: AntiLeak-Bench](https://www.emergentmind.com/topics/antileak-bench)
