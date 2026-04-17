## 核心结论

Minerva 不是“从零为数学设计的新模型”，而是把已经很强的通用大语言模型 PaLM 540B，继续放到数学语料上训练。这里的“续训”就是在已有语言能力上追加特定领域数据，让模型把数学符号、公式书写和推理模式学得更稳。论文给出的关键数字是：它使用约 38.5B token 的数学相关数据继续训练，这些数据主要来自 arXiv 和包含数学表达的网页内容。

这件事的价值不在于模型突然学会了“像计算器一样精确运算”，而在于两个更工程化的改进：

1. 训练数据保留了 LaTeX、MathJax、AsciiMath 等原始数学表达，模型不再只看到自然语言里的“分数、积分、三角函数”，而是直接看到 `\frac{a}{b}`、`\int`、`\sin` 这类符号。
2. 推理时不只生成一次答案，而是用 nucleus sampling 采样多条链式推理，再把最终答案做归一化和符号等价判断，最后用多数投票选结果。

因此，Minerva 的提升不是单点技巧，而是“数学数据续训 + 数学符号保留 + 多次采样 + 答案归一化 + SymPy 等价校验”这一整条链路共同起作用。对于 MATH 基准，540B 模型在 `maj1@64` 设置下达到 50.3%，说明多样化采样后的投票比单次输出更稳定。

一个最小例子就能看出归一化的重要性。模型输出：

`Final Answer: The final answer is $\boxed{1,000}$.`

如果直接按字符串比较，它和 `1000` 是不同文本；如果先去掉 `$`、`,` 和 `\boxed{}`，再把结果交给 SymPy 简化，就能把两者都映射到同一个数学对象 `1000`。这一步的本质是把“写法是否一样”转成“数学含义是否一样”。

---

## 问题定义与边界

问题可以表述为：如何让一个大语言模型稳定解决带公式的数学题，并输出可自动判分的标准答案。

这里有两个边界要先说清。

第一，Minerva 解决的是“数学推理建模”问题，不是“形式化证明系统”问题。形式化证明系统是指每一步推导都要满足严格机器校验规则，比如 Lean 或 Coq；而 Minerva 仍然是自然语言模型，输出的是带推理过程的文本和公式。它能显著提升竞赛题、教材题、数量推理题上的正确率，但不保证每一步推导都形式正确。

第二，它的目标不是只在训练里记住答案，而是学习数学文本的表达分布。表达分布可以理解为“这个领域里常见的写法、符号、解题顺序和答案格式”。所以训练数据必须尽量保留原始公式，而不是把网页中的数学脚本全部剥掉。如果把 `\frac{a}{b}` 变成普通文本 “a over b”，模型学到的就不再是数学原生表达。

数据处理边界可以概括为下表：

| 阶段 | 输入形态 | 关键处理 | 目标 |
|---|---|---|---|
| 原始网页 | HTML、MathJax、`<math>`、LaTeX 脚本 | 识别数学片段并保留 | 不丢失公式结构 |
| 训练文本 | 自然语言 + 数学 token | 将公式按原始形式送入模型 | 学会数学书写模式 |
| 推理输出 | 链式推理文本 + Final Answer | 抽取最终答案 | 分离“过程”和“判分对象” |
| 归一化 | `\boxed{}`、单位、标点、括号差异 | 正则清洗、标准化表达 | 消除表面格式差异 |
| 等价判断 | 字符串或表达式 | 用 SymPy 简化比较 | 判断数学含义是否一致 |

玩具例子很直接。若标准答案是 `2*pi`，模型输出 `\(2\pi\)`。从字符串看它们不同；从数学意义看它们相同。只做字符串比较会误判，只做弱规则替换也不够，因为 `\frac{3}{4}`、`0.75`、`3/4` 也都应视为同一答案。于是系统必须定义统一的答案表示。

真实工程例子出现在自动评测中。假设一个题库每天回归测试 5 万道数学题，若你仅用字符串精确匹配，那么大量“其实是对的”答案会被判错，误差会反向污染模型评估，进一步影响训练和部署决策。Minerva 的工作告诉我们：数学题评测不能把“文本相等”误当成“答案相等”。

---

## 核心机制与推导

核心机制分成训练侧和推理侧两部分。

训练侧的重点是保留数学 token。token 可以理解为模型看到的最小文本片段。普通语言模型把自然语言切成常见子词；Minerva 面对的是大量公式，因此必须让 `\sin`、`\theta`、`\frac`、上下标、括号结构等在训练数据里完整出现。这样模型学到的不只是“数学题会提到正弦函数”，而是“正弦函数在真实数学文本里如何书写、如何进入推导链”。

如果把公式看成一种特殊语言，那么续训过程近似是在优化下面的目标：

$$
\max_\theta \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

这里 $x_t$ 既可能是普通词，也可能是数学符号 token。区别在于，数学语料让条件概率分布更多地覆盖公式序列，而不是只覆盖自然语言叙述。

推理侧的重点是 `maj1@k`。它的流程不是“采样 k 次，然后选最长推理”，而是：

1. 用 nucleus sampling 采样 $k$ 条完整输出。
2. 从每条输出中提取 `Final Answer` 后的最终答案。
3. 对答案做归一化。
4. 用 SymPy 判断数学等价并映射到统一表达。
5. 对统一后的答案做多数投票。

nucleus sampling 可以理解为“只在累计概率达到阈值的高概率候选里随机采样”，比贪心解码更有多样性。论文中给出的典型设置是温度 $T=0.6$、top-p $=0.95$。它的目的不是制造随机噪声，而是让模型从多个合理推理路径里给出不同候选。

`maj1@k` 的关键不是“投票”，而是“先统一答案再投票”。设第 $i$ 次采样得到的原始答案为 $a_i$，归一化函数为 $N(\cdot)$，符号化简函数为 $S(\cdot)$，则参与统计的不是 $a_i$，而是：

$$
v_i = S(N(a_i))
$$

最终结果是：

$$
\hat{y} = \arg\max_{y} \sum_{i=1}^{k} \mathbf{1}[v_i = y]
$$

其中 $\mathbf{1}[\cdot]$ 是指示函数，意思是“若第 $i$ 个答案等于 $y$ 就计 1，否则计 0”。

玩具例子如下。某题采样 6 次后得到：

- `Final Answer: The answer is \( \frac{3}{4} \)`
- `Final Answer: 0.75`
- `Final Answer: \boxed{\frac34}`
- `Final Answer: 3/4`
- `Final Answer: 0.750`
- `Final Answer: 2/3`

若不归一化，前 5 条会被看成 5 个不同字符串；若做归一化并用 SymPy 简化，前 5 条都可映射到同一个对象，因为

$$
\mathrm{simplify}\left(\frac{3}{4} - 0.75\right) = 0
$$

于是投票结果会稳定落在 `3/4` 这组答案上，而不会被格式差异打散。

真实工程例子是 MATH 评测。对 540B 模型，每题采样 64 次后做 `maj1@64`，准确率达到 50.3%。这说明单条链式推理常常不是最可靠的，而“多条合理链 + 规范化投票”可以显著提高最终判对率。这个提升本质上是在利用模型的分布信息：正确答案可能不是单次最可能输出，但常常会在多次采样中反复出现。

---

## 代码实现

下面给一个最小可运行版本，演示答案抽取、归一化、SymPy 等价判断和多数投票。它不是 Minerva 论文代码的复刻，但机制与论文描述一致。

```python
import re
from collections import Counter
from sympy import simplify, sympify
from sympy.parsing.sympy_parser import parse_expr


def extract_final_answer(text: str) -> str:
    m = re.search(r"Final Answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return text.strip()
    return m.group(1).strip()


def strip_boxed(s: str) -> str:
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\boxed\s*([^\s]+)", r"\1", s)
    return s


def normalize_answer(s: str) -> str:
    s = extract_final_answer(s)
    s = s.replace("$", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace(",", "")
    s = strip_boxed(s)
    s = re.sub(r"^The final answer is\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^The answer is\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.$", "", s)
    s = s.strip()

    # 常见 LaTeX 归一化
    s = re.sub(r"\\frac\s*([0-9a-zA-Z]+)\s*([0-9a-zA-Z]+)", r"(\1)/(\2)", s)
    s = s.replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")")
    s = s.replace("\\pi", "pi")
    return s.strip()


def canonicalize(s: str) -> str:
    s = normalize_answer(s)
    expr = parse_expr(s, evaluate=True)
    return str(simplify(expr))


def equivalent(a: str, b: str) -> bool:
    ea = parse_expr(normalize_answer(a), evaluate=True)
    eb = parse_expr(normalize_answer(b), evaluate=True)
    return simplify(ea - eb) == 0


def majority_vote(chains):
    normalized = [canonicalize(x) for x in chains]
    return Counter(normalized).most_common(1)[0][0]


toy_outputs = [
    r"Final Answer: The final answer is $\boxed{1,000}$.",
    r"Final Answer: 1000",
    r"Final Answer: $1000$",
]

assert normalize_answer(toy_outputs[0]) == "1000"
assert equivalent(r"Final Answer: \(2\pi\)", r"Final Answer: 2*pi")
assert equivalent(r"Final Answer: \frac{3}{4}", r"Final Answer: 0.75")
assert majority_vote(toy_outputs) == "1000"
print("ok")
```

上面这段代码有三个要点。

第一，`extract_final_answer` 把整条链式推理和最终答案分开。因为投票对象应该是“最终答案”，不是整段 reasoning。否则两条逻辑相同但措辞不同的推理会被当成不同样本。

第二，`normalize_answer` 负责做近似于论文 Listing 1 思路的文本清洗，包括去掉 `$...$`、`\boxed{}`、千位分隔逗号、句号，以及把常见 LaTeX 写法转成更容易被 SymPy 解析的表达。真实系统里这一步会更长，因为还要处理单位、`\text{}`、不规范分数、空格、隐式乘法等问题。

第三，`canonicalize` 和 `equivalent` 把字符串变成符号表达式。SymPy 是 Python 的符号计算库，“符号计算”就是把表达式当作数学对象而不是普通字符串处理。这样 `0.75` 和 `3/4`、`2*pi` 和 `2\pi` 才能落到同一语义空间。

在真实工程中，推理模块通常长这样：

```python
answers = []
for _ in range(k):
    chain = sample_chain(prompt, temperature=0.6, top_p=0.95)
    final = extract_final_answer(chain)
    normalized = canonicalize(final)
    answers.append(normalized)

vote = Counter(answers).most_common(1)[0][0]
```

这里的 `sample_chain` 对应模型采样接口。若模型很大，`k=64` 或 `k=256` 的成本不低，因此这段逻辑往往不会对所有请求都启用，而是只对高价值评测或高不确定性样本启用。

---

## 工程权衡与常见坑

Minerva 方法有效，但工程代价很明确：准确率提升来自更多采样和更复杂后处理，而不是“免费收益”。

先看收益与成本的大致关系：

| 设置 | 典型收益 | 主要成本 | 适用场景 |
|---|---|---|---|
| 单次 greedy | 延迟最低 | 正确率低、容易卡在错误链路 | 在线低延迟 |
| 少量采样 `k=8~16` | 明显提升 | 成本线性增加 | 中等预算推理 |
| 中等采样 `k=32~64` | 接近论文主力区间 | 延迟和算力压力较高 | 离线评测、高价值题目 |
| 超大采样 `k=128~256` | 边际收益递减 | 成本很高 | 小模型补偿或研究实验 |

论文和相关材料表明，多数投票能带来大约 10 到 15 个点的提升，但这不是线性增长。`k` 增大后，收益会逐步饱和。原因很简单：如果模型分布里正确答案本来就很弱，继续采样只是在反复抽取错误；如果正确答案已经明显占优，再增加很多样本也只是重复已有结果。

最常见的坑有四类。

第一，只做字符串比较。这样会把 `1000`、`1,000`、`\boxed{1000}`、`$1000$` 看成不同答案，直接破坏投票结果。这类错误非常隐蔽，因为模型其实答对了，但评测系统说它错了。

第二，归一化规则写得过猛。比如简单删除所有 `\text{}` 内容，可能会把有意义的变量名或条件说明一并删掉。规则系统必须围绕“最终答案格式”设计，而不是粗暴清洗整段文本。

第三，SymPy 解析失败后没有兜底。真实输出里常有不闭合括号、隐式乘法、非法 LaTeX 片段。如果直接 `parse_expr` 抛异常，整道题的投票流程就断了。工程上必须加入 fallback，比如退回到轻量字符串标准化，或者只对可解析候选做等价归并。

第四，把高 `k` 采样直接用于在线服务。对 540B 级别模型，多次采样会带来极高延迟和推理成本。离线评测可以接受，面向用户的即时问答通常不行。

一个很典型的误判例子如下：

- 输出 A：`Final Answer: \(1000\)`
- 输出 B：`Final Answer: 1000`
- 输出 C：`Final Answer: \boxed{1,000}`

若不 normalize，投票频次可能是 `1, 1, 1`，系统无法聚合共识；加上归一化和 SymPy 后，三者都归到 `1000`，频次立即变成 `3`。这就是为什么论文提到不做归一化会损失约 1 个点准确率。这个数字看上去不大，但在成熟基准上，1 个点通常已经是显著差距。

---

## 替代方案与适用边界

第一种替代方案是 beam search。beam search 可以理解为“同时保留若干条最高分路径，再从中选最好的一条”。它的优点是更确定、更可复现；缺点是多样性不足。Minerva 依赖的是“正确答案在多条随机链中反复出现”，而 beam search 往往保留的是一组彼此很相似的高概率路径，因此不一定能像 nucleus sampling 那样带来投票收益。

第二种替代方案是只做 deterministic normalization，也就是不多次采样，只加强答案抽取和符号等价判断。这个方案便宜，适合作为基线，但它只能减少“判分误差”，不能修复“模型本身第一次就走错推理链”的问题。

第三种方案适合低资源模型。若模型只有 `<=8B` 参数，不适合默认 `k=64` 甚至 `k=256`，可以采用条件式流程：

```text
greedy
  -> 若答案格式清晰且置信度高，直接输出
  -> 若答案不稳定或题目是高难数学题，触发 16 次 nucleus sampling
  -> normalize
  -> SymPy check
  -> majority vote
```

这个流程的关键不是“永远采样很多次”，而是把采样预算集中给困难样本。困难样本可以用多种信号识别，比如答案提取失败、推理链自相矛盾、模型对候选 token 的置信度分散等。

玩具例子是一个简单分数题。对 `1/3 + 5/12`，greedy 已经可能稳定输出 `3/4`，没必要额外采样。真实工程例子是教育题库批改系统：基础代数题用单次推理即可，高中竞赛题或大学微积分题再触发多样本投票。这样能把整体成本控制在可接受范围内。

Minerva 方法的适用边界也要说清。它特别适合“答案可以符号化并自动校验”的任务，如代数、数论、几何计算、竞赛题和数量推理题。它不那么适合开放式证明题、需要图形直观推导的问题，或者答案本身不是数学表达式而是长篇解释的任务。因为这类任务里，“最终答案归一化”这一步并不总能定义得足够稳定。

---

## 参考资料

| 标题 | 出处 | 关键内容 |
|---|---|---|
| Solving Quantitative Reasoning Problems with Language Models | NeurIPS 2022 / Minerva 论文 | 介绍在 PaLM 基础上进行数学续训、采样推理和评测结果 |
| Minerva 相关公开版本与附录材料 | 论文附录与镜像资料 | 描述 38.5B 数学数据来源、数学网页抽取与答案归一化细节 |
| 论文中的推理与评测部分 | Section 2、表格结果、附录 B/D | 给出 nucleus sampling、`maj1@k`、SymPy 校验和 50.3% MATH 成绩 |

1. Lewkowycz, A. et al. *Solving Quantitative Reasoning Problems with Language Models*. NeurIPS 2022. 关键点：Minerva 基于 PaLM 540B 在数学语料上续训，并报告了 MATH 上 `maj1@64 = 50.3%` 的结果。  
2. 同论文 Section 2。关键点：说明推理阶段使用 nucleus sampling，保留链式推理，并通过答案提取与后处理进行评测。  
3. 同论文附录 B/D 及相关公开说明。关键点：描述数学网页与 arXiv 数据构建、LaTeX/MathJax/AsciiMath 保留策略，以及答案归一化与 SymPy 等价比较。
