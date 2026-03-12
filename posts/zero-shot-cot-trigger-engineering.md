## 核心结论

Zero-shot CoT（Zero-shot Chain-of-Thought，零样本思维链）指的是：**不提供示例，只在原问题后追加一句触发词，让模型先生成中间推理，再给最终答案**。这里“触发词”可以直白理解为一段开关语句，用来把模型从“直接猜答案”切到“按步骤展开”。

Kojima 等人在 2022 年论文中展示了一个非常强的结果：对同一个大模型，只把问题从 $P$ 改成 $P \Vert T$，其中 $T=$“Let’s think step by step.”，就能让多步推理任务的准确率显著上升。最常被引用的数字是：

| 任务 | 无触发词 | Zero-shot CoT |
|---|---:|---:|
| MultiArith | 17.7% | 78.7% |
| GSM8K | 10.4% | 40.7% |

这说明提示词不是“修饰文案”，而是在很多场景里直接决定模型走哪条求解路径。没有触发词时，模型更容易走“压缩式回答”，即直接输出一个看起来像答案的短文本；加入触发词后，模型更容易先生成子步骤，再汇总结论。

一个最小玩具例子就能看清楚：

问题：`150,000 人，60% 为成人，40% 成人有车，问有车人数？`

在后面追加 `Let’s solve this step by step.` 后，理想输出会变成：

1. 成人数 $= 150000 \times 0.6 = 90000$
2. 有车人数 $= 90000 \times 0.4 = 36000$
3. 最终答案：36000

也就是说，Zero-shot CoT 的核心价值不是“让模型更聪明”，而是**让模型把隐含的中间变量显式化**。对初级工程师来说，这一点很重要，因为你终于能看到模型到底是算错了，还是拆题错了。

---

## 问题定义与边界

严格定义下，Zero-shot CoT 可以写成：

$$
\text{Answer} = \operatorname{LM}(P \Vert T)
$$

其中：

- $P$ 是原始问题
- $T$ 是触发词
- $\Vert$ 表示把两段文本拼接起来
- $\operatorname{LM}$ 是语言模型，白话解释就是“根据上下文继续生成文本的系统”

如果没有触发词，形式就是：

$$
\text{Answer} = \operatorname{LM}(P)
$$

这两者只差一小段文字，但输出分布可能差很多。这里“输出分布”可以简单理解为：模型在下一步最可能写什么。

Zero-shot CoT 的适用边界很清楚：

| 任务类型 | 是否适合 Zero-shot CoT | 原因 |
|---|---|---|
| 多步数学题 | 适合 | 需要显式中间变量 |
| 逻辑推理题 | 适合 | 需要分解条件与顺序 |
| 简单事实问答 | 通常不适合 | 没有多步推理收益 |
| 强结构化抽取 | 通常不适合 | 规则比推理更重要 |
| 高实时要求接口 | 谨慎使用 | 输出 token 会明显增加 |

一个容易理解的边界例子：

问题：`计算 7 + 5`

即使追加 `Break down the problem into steps before answering.`，模型也不会获得本质收益，因为这本来就是一步题。  
但如果问题变成：`一个班有 48 人，男生比女生多 6 人，男女各多少？`，先拆步骤就很有意义，因为它涉及设变量、列式、求解三个动作。

触发词本身也有边界。它不是越长越好，也不是语义相近就一定效果相近。一些公开整理材料和后续工程实践表明，不同文案差异可能很大：

| 触发词 | 公开资料中常见结果 | 工程含义 |
|---|---:|---|
| Let’s think step by step. | 78.7% | 稳定基线 |
| Let’s think about this logically. | 约 74.5% 到 77.3% | 接近但略弱 |
| Let’s solve this problem by splitting it into steps. | 约 74.5% | 可用但不一定最优 |
| Let’s work together to solve this problem step by step. | 49.4% | 语义相近，但效果可能明显变差 |

结论很直接：**Zero-shot CoT 不是“随便写一句鼓励模型思考的话”**，而是一种对触发词高度敏感的推理开关。

---

## 核心机制与推导

从机制上看，Zero-shot CoT 可以理解为：在输入里插入一个很短的“解题框架”，让模型优先生成一串隐含步骤 $H_1, H_2, \dots, H_n$，再生成最终答案 $A$。

可以把过程写成：

$$
P(A \mid P \Vert T) = \sum_{H} P(A \mid H, P \Vert T)\, P(H \mid P \Vert T)
$$

这行式子的白话解释是：最终答案不是直接从问题里跳出来，而是经过一串中间步骤 $H$ 汇总出来。触发词 $T$ 的作用，不是直接告诉模型答案，而是提高“先写出合理步骤”的概率。

简化成流程图就是：

| 阶段 | 无触发词 | Zero-shot CoT |
|---|---|---|
| 输入 | 原问题 | 原问题 + 触发词 |
| 早期生成 | 倾向直接答 | 倾向先列子步骤 |
| 中间变量 | 常被省略 | 更容易显式输出 |
| 最终答案 | 可能跳步 | 更容易与步骤一致 |

继续看前面的玩具例子：

`150,000 人，60% 为成人，40% 成人有车，问有车人数？`

加上 `Let’s think step by step.` 后，模型更可能生成：

1. `150000 × 60% = 90000`
2. `90000 × 40% = 36000`
3. `答案是 36000`

这里的 `90000` 就是显式中间变量。没有 CoT 时，模型可能直接输出 `36000`，也可能误把两个百分比直接乘到总人数上后写错，或者在文本题里丢掉一个条件。  
有 CoT 时，至少你能检查每一步。

这也是为什么 Zero-shot CoT 在 MultiArith、GSM8K 这类题上效果明显。它们的共同特点不是“题目长”，而是**正确答案依赖多个中间状态**。如果模型不显式表示这些状态，错误通常会累积。

再看一个真实工程例子。  
假设你做一个金融分析 agent，需要回答：

`某公司 2025 年营收同比增长 12%，毛利率从 32% 升到 36%，若收入从 50 亿增到 56 亿，毛利润增长多少？`

如果直接回答，模型可能只给最终数值；如果加上 Zero-shot CoT，它更可能输出：

1. 2024 毛利润 $= 50 \times 32\% = 16$
2. 2025 毛利润 $= 56 \times 36\% = 20.16$
3. 增长额 $= 20.16 - 16 = 4.16$
4. 增长率 $= 4.16 / 16 = 26\%$

这种“可审计推理链”对工程系统很关键，因为日志里保留的不只是答案，还有计算路径。

---

## 代码实现

工程上实现 Zero-shot CoT 非常简单：**在 prompt 末尾追加触发词**。不需要微调，不需要训练集，不需要额外参数。

一个最小可运行的 Python 例子如下：

```python
import re

def build_zero_shot_cot_prompt(question: str, trigger: str = "Let's think step by step.") -> str:
    return f"{question.strip()}\n{trigger}"

def solve_toy_example(population: int, adult_ratio: float, car_ratio_among_adults: float) -> int:
    adults = int(population * adult_ratio)
    car_owners = int(adults * car_ratio_among_adults)
    return car_owners

def extract_last_int(text: str) -> int:
    nums = re.findall(r"\d[\d,]*", text)
    if not nums:
        raise ValueError("No integer found in text")
    return int(nums[-1].replace(",", ""))

question = "150,000 人，60% 为成人，40% 成人有车，问有车人数？"
prompt = build_zero_shot_cot_prompt(question)

mock_model_output = """
1) 成人数 = 150,000 × 60% = 90,000
2) 有车人数 = 90,000 × 40% = 36,000
最终答案：36,000
"""

assert "Let's think step by step." in prompt
assert solve_toy_example(150000, 0.6, 0.4) == 36000
assert extract_last_int(mock_model_output) == 36000
```

这段代码做了三件事：

| 组件 | 作用 |
|---|---|
| `build_zero_shot_cot_prompt` | 把问题和触发词拼起来 |
| `solve_toy_example` | 用确定性逻辑验证正确答案 |
| `extract_last_int` | 从模型输出里抽取最终数值 |

如果你用 API，结构通常也只是多一行触发词。为了避免 SDK 版本差异，下面用伪代码表达核心位置：

```python
question = "150,000 人，60% 为成人，40% 成人有车，问有车人数？"
trigger = "Let's think step by step."
prompt = f"{question}\n{trigger}"

response_text = llm_generate(prompt)

# 线上通常会同时保存完整推理文本与抽取后的最终答案
steps_text = response_text
final_answer = extract_last_int(response_text)

assert final_answer == 36000
```

实现时有两个细节经常被忽略。

第一，**触发词必须真的进模型上下文**。如果你的系统在服务端做了 prompt 截断，追加词可能被切掉，结果你以为在跑 Zero-shot CoT，实际上跑的是普通零样本。

第二，**解析最终答案要和步骤文本分开**。不少系统会要求模型最后一行输出 `Answer: ...`，否则日志很好看，但程序难以稳定抽取。

常见触发词的 token 成本通常不高，但它会引出更长的回答：

| 触发词 | 本身增加的 token | 典型影响 |
|---|---:|---|
| Let’s think step by step. | 很少 | 主要成本来自更长输出 |
| Think carefully before answering. | 很少 | 可能诱导解释，但未必稳定分步 |
| Break down the problem into steps before answering. | 略多 | 更明确，但文本更长 |

所以成本大头通常不是那一句触发词，而是后面生成的整段推理链。

---

## 工程权衡与常见坑

Zero-shot CoT 的工程优势很明显：提示短、接入快、统一性强。  
但它不是“免费午餐”。

最现实的权衡是 token、延迟和可控性。

| 维度 | Zero-shot CoT | 直接回答 |
|---|---|---|
| 准确率 | 多步任务通常更高 | 多步任务容易跳步 |
| 输出长度 | 更长 | 更短 |
| 延迟 | 更高 | 更低 |
| 审计性 | 更好 | 更差 |
| prompt 维护成本 | 低 | 低 |

很多实践会观察到 CoT 输出 token 比直接回答多出 2 到 10 倍。这意味着：

- 成本上升
- 流式输出时间变长
- 日志存储量变大
- 下游解析更复杂

常见坑主要有五类。

第一，**把触发词当同义改写问题**。  
“step by step”“think carefully”“work together”语义看起来很近，但模型不是按人工语义相似度工作的。它对特定词序、风格、训练中出现频率都敏感。公开资料里，“Let’s work together...” 的表现就明显弱于经典触发词。

第二，**默认所有任务都该加 CoT**。  
信息抽取、分类、简单事实问答，很多时候并不需要中间推理，强行加 CoT 只会增加延迟和噪声。

第三，**把中间步骤当成事实保证**。  
模型写出了步骤，不代表步骤就一定对。CoT 提高的是“更可能走对的路径”，不是“每一步都可证明正确”。高风险系统仍然需要规则校验、计算器、程序执行器或外部工具。

第四，**线上不记录推理链，只记录最终答案**。  
这样做等于放弃 Zero-shot CoT 最重要的调试价值。你本来多花了 token，就是为了看到推理过程；如果日志里没有，那成本花了，收益却没拿到。

第五，**忽略领域审计要求**。  
真实工程里，金融分析 agent 是一个典型例子。若系统必须回答“这个比率怎么来的”，Zero-shot CoT 的优势是：它用统一触发词就能把绝大多数计算路径展开，便于审计。Few-shot CoT 虽然往往更强，但示例一多，不同任务的 prompt 风格容易不一致，日志对比也更难做。

一个常见运维流程可以写成：

| 步骤 | 目标 |
|---|---|
| 监控 prompt 成本 | 看 Zero-shot CoT 是否超预算 |
| 比较多个触发词 | 找到任务集上的稳定基线 |
| 记录完整中间步骤 | 便于复盘错误来源 |
| 抽样人工复核 | 判断“步骤是否真正确” |
| 必要时引入工具校验 | 对高风险数值题做兜底 |

---

## 替代方案与适用边界

Zero-shot CoT 的直接替代方案是 Few-shot CoT。这里“Few-shot”可以直白理解为：**先给模型看几个带详细解题步骤的样例，再让它做新题**。

两者对比如下：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| Zero-shot CoT | 只加触发词 | 接入快，prompt 短，迁移方便 | 效果依赖模型与触发词 |
| Few-shot CoT | 给若干示例 + 推理链 | 通常更稳，任务对齐更强 | prompt 长，维护成本高 |
| 直接回答 | 不展示推理 | 快、便宜 | 多步题易跳步 |
| 工具增强推理 | 模型 + 计算器/代码执行 | 数值更可靠 | 系统更复杂 |

如果任务是跨很多领域的通用 agent，Zero-shot CoT 往往是第一步，因为它最轻。  
如果任务非常窄，比如只做小学数学题分类解答，Few-shot CoT 通常更容易把准确率再往上推。

可以把选择逻辑简化成一张表：

| 任务场景 | 推荐策略 |
|---|---|
| 多步推理，但预算敏感 | Zero-shot CoT |
| 多步推理，且有固定题型 | Few-shot CoT |
| 高风险数值计算 | CoT + 工具校验 |
| 单步问答或抽取 | 直接回答 |
| 超长上下文系统 | 优先短 prompt，谨慎加 Few-shot |

一个对比示例：

- `Zero-shot CoT + Let's think step by step`
  - 适合先快速验证模型是否具备该类推理能力
  - 不需要设计样例
- `Few-shot CoT + 1 个完整示例 + Think carefully`
  - 更适合你已经知道题型稳定，且愿意花上下文去换准确率
  - 但需要维护示例质量，示例过旧或偏题会直接污染输出

因此，Zero-shot CoT 的正确定位不是“最终形态”，而是**最低成本、最高杠杆的推理增强基线**。如果连这个基线都没有收益，继续堆复杂提示通常也不会有好结果。

---

## 参考资料

1. [Kojima et al., Large Language Models are Zero-Shot Reasoners, arXiv 2022](https://arxiv.org/abs/2205.11916)  
   贡献：提出 Zero-shot CoT，给出 MultiArith 从 17.7% 到 78.7%、GSM8K 从 10.4% 到 40.7% 的代表性结果。

2. [Emergent Mind 对 2205.11916 的论文摘要页](https://www.emergentmind.com/articles/2205.11916)  
   贡献：便于快速查看论文摘要与核心实验结论。

3. [Galileo, What Is Chain-of-Thought Prompting? A Guide to Improving LLM Reasoning, 2026-02-02](https://galileo.ai/blog/what-is-chain-of-thought-prompting-guide-improving-llm-reasoning)  
   贡献：从生产环境角度讨论 CoT 的收益、成本、延迟和可观测性。

4. [Prompt engineering lecture notes / secondary summaries on trigger phrase comparisons](https://blog.csdn.net/lythinking/article/details/135490462)  
   贡献：整理了不同触发词的效果差异，适合做工程上的敏感性提醒；属于二手资料，使用时应以原实验设定为准。

5. [Tetrate: Chain-of-Thought Prompting 入门说明](https://tetrate.io/learn/ai/chain-of-thought-prompting)  
   贡献：对“让模型先拆步骤再回答”的直观解释较清楚，适合新手建立概念。
