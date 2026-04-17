## 核心结论

IFEval 是一个专门评测“指令遵循”的基准，不评模型知道多少事实，而评模型能否把要求准确执行出来。这里的“指令遵循”可以先白话理解为：用户怎么规定输出，模型就要怎么做。它只收录可程序验证的约束，例如段落数、句子数、关键词、格式、顺序、禁用词、列表项个数等，因此结果可以自动判分、重复复现，也方便横向比较不同模型。

它最重要的设计，不是让题目更难，而是让“合格”这件事足够明确。比如提示要求“输出必须有 3 段、每段至少 3 句、`energy` 至少出现 4 次”，那么严格模式下只有三个条件同时满足才算通过。只要少一段、少一句，或者关键词次数不够，`prompt_level_strict` 都记为失败。

IFEval 常看四个轨道：`prompt_level_strict`、`prompt_level_loose`、`inst_level_strict`、`inst_level_loose`。其中“prompt 级”看整道题是否整体过关，“inst 级”看细粒度约束的满足比例；“strict”要求原始输出直接通过，“loose”会先做归一化再检查，减少格式噪声造成的误判。常用公式可以写成：

$$
prompt\_level\_strict = \frac{\text{strict 下合格的 prompt 数}}{\text{总 prompt 数}}
$$

$$
inst\_level\_loose = \frac{\text{loose 下通过的细粒度指令数}}{\text{总细粒度指令数}}
$$

如果你的目标是做上线前 QA，IFEval 很有价值；如果你的目标是评创意、知识深度或文风质量，IFEval 不能单独解决问题。

---

## 问题定义与边界

IFEval 的“问题定义”非常克制：它只接受能被脚本检查的指令。所谓“可程序验证”，白话说就是不用人拍脑袋判断，写个规则就能知道对不对。比如“请输出 2 段，每段 3 句，包含关键词 `energy`”是合格指令；但“写得有感染力一点”通常不适合作为 IFEval 里的核心约束，因为“感染力”很难稳定编码成规则。

这决定了 IFEval 的边界。它测的是“执行约束的能力”，不是“回答有没有洞察”。一个模型可能在 IFEval 上分数很高，但写开放式评论时很普通；也可能某个模型知识很强，却因为总爱多说一句废话，在严格约束下频繁丢分。

下面这张表可以把四个维度看清楚：

| 维度 | 检查对象 | 合格条件 | 适合回答的问题 |
| --- | --- | --- | --- |
| `prompt_level_strict` | 整个 prompt | 原始输出中全部约束都通过 | 这道题是否一次性完全做对 |
| `prompt_level_loose` | 整个 prompt | 归一化后全部约束都通过 | 失败是不是主要由格式噪声导致 |
| `inst_level_strict` | 单条细粒度指令 | 原始输出中该指令通过 | 模型最容易错在哪类约束 |
| `inst_level_loose` | 单条细粒度指令 | 归一化后该指令通过 | 去掉格式干扰后理解是否仍有缺陷 |

玩具例子可以这样看。提示写成：“请输出 2 段，每段 3 句，包含关键词 `energy`。”验证器只要统计段落数、句号数和关键词次数，就能给出结果。这种设计对新手很重要，因为它把“评测标准”从模糊印象变成了机器可执行的规则集合。

---

## 核心机制与推导

IFEval 的核心机制可以概括成一句话：先把自然语言要求形式化成约束，再把输出逐项验证。这里的“形式化”可以先理解为：把一句人话翻译成一组结构化规则。工程上通常会有类似 `config.yaml` 或数据集字段的描述，里面写明这道题有哪些检查项，例如：

- `format: markdown`
- `length: min 3 paragraphs`
- `keywords: ["energy"]`
- `keyword_min_count: 4`

验证时不是整体凭感觉判断，而是把每条约束交给对应的 validator。所谓 validator，白话说就是“只负责检查一件事的小函数”。比如段落检查器只统计段落，关键词检查器只数词频，格式检查器只认 Markdown 结构。

一个简化的流程是：

1. 读取 prompt 和它绑定的约束。
2. 取模型输出 `response`。
3. 在 strict 轨道上直接逐项检查。
4. 如果需要 loose 轨道，先做 normalization，再逐项检查。
5. 汇总为 prompt 级和 inst 级分数。

其中 normalization 是 loose 模式的关键。它的白话意思是“先把一些不重要的表面差异抹平”。例如去掉多余空行、统一列表前缀、清理首尾空白、把某些等价格式转成统一表示。这样做的目标不是放水，而是区分“模型没理解要求”和“模型理解了但格式抖了一下”。

可以把它写成非常直接的伪代码：

```python
for constraint in prompt.constraints:
    result = check(constraint, response)
    record(result)
```

如果展开一点，逻辑就是：

- strict：直接检查原始输出。
- loose：对输出做归一化后再检查。
- prompt 级：所有约束都通过才算过。
- inst 级：每条约束单独统计通过率。

这里有一个很重要的推导关系：多约束 prompt 的严格通过率，一定不会高于单条约束平均通过率。因为 prompt 级 strict 本质上是多个条件的“与”关系。若一题里有 $n$ 个约束，每个约束都必须通过，则整体通过条件是：

$$
Pass_{prompt} = \bigwedge_{i=1}^{n} Pass_i
$$

这也是为什么长指令、复合指令、冲突指令会让分数快速下降。约束越多，任何一个环节出错都会把整题打成失败。

---

## 代码实现

下面给一个可运行的 Python 简化版实现，演示 strict/loose 的分支，以及 prompt 级和 inst 级如何统计。这里不依赖真实 IFEval 数据集，但逻辑与其评测思想一致。

```python
import re

def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def split_paragraphs(text: str):
    return [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]

def sentence_count(paragraph: str) -> int:
    parts = re.split(r'[。！？.!?]+', paragraph)
    return len([x for x in parts if x.strip()])

def keyword_count(text: str, keyword: str) -> int:
    return len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))

def check_constraints(response: str, keyword: str = "energy"):
    paragraphs = split_paragraphs(response)
    checks = {
        "exactly_3_paragraphs": len(paragraphs) == 3,
        "each_paragraph_at_least_3_sentences": all(sentence_count(p) >= 3 for p in paragraphs) if paragraphs else False,
        "keyword_at_least_4_times": keyword_count(response, keyword) >= 4,
    }
    prompt_pass = all(checks.values())
    inst_pass_rate = sum(checks.values()) / len(checks)
    return checks, prompt_pass, inst_pass_rate

def check_prompt(response: str):
    strict_checks, strict_prompt_pass, strict_inst_rate = check_constraints(response)
    loose_response = normalize(response)
    loose_checks, loose_prompt_pass, loose_inst_rate = check_constraints(loose_response)
    return {
        "strict": {
            "checks": strict_checks,
            "prompt_pass": strict_prompt_pass,
            "inst_pass_rate": strict_inst_rate,
        },
        "loose": {
            "checks": loose_checks,
            "prompt_pass": loose_prompt_pass,
            "inst_pass_rate": loose_inst_rate,
        },
    }

good = """Energy is measurable. Energy is transferable. Energy is conserved.

We study energy in mechanics. We study energy in thermodynamics. Energy connects models.

Engineers track efficiency carefully. They compute losses explicitly. They optimize energy systems."""
bad = """Energy matters.

It appears in physics."""

result_good = check_prompt(good)
result_bad = check_prompt(bad)

assert result_good["strict"]["prompt_pass"] is True
assert result_good["strict"]["inst_pass_rate"] == 1.0
assert result_bad["strict"]["prompt_pass"] is False
assert result_bad["strict"]["inst_pass_rate"] < 1.0
```

这个例子里，strict 检查会直接验证原始输出是否满足“正好 3 段”“每段至少 3 句”“`energy` 至少 4 次”。如果你想接近真实工程流程，通常会先加载一组 prompt 数据，再批量跑模型输出。它和博客里的 `posts.json` 很像，本质上都是“索引 + 内容 + 元信息”的集合，只是 IFEval 的元信息是约束定义而不是文章标签。

真实工程例子是模型上线前的批量 QA。团队往往会准备一批约 500 条结构化 prompt，让候选模型统一跑一遍，然后导出四个指标。若某模型 `inst_level_loose` 高但 `prompt_level_strict` 低，通常说明它大致理解了要求，但容易在最终格式、额外空行、顺序细节上失手；若 strict 和 loose 都低，问题更可能出在指令理解本身。

---

## 工程权衡与常见坑

IFEval 的优点是可复现，但代价是它对“边界情况”很敏感。严格模式尤其如此。你要求模型输出 Markdown 列表，它多打一行空白；你要求 3 段，它写成 3 个带标题的块；你要求禁用某词，它却在代码块或引用里出现了一次。这些都可能让 strict 直接失败。

常见坑和对应策略可以整理成表：

| 常见坑 | 典型表现 | 风险 | 应对策略 |
| --- | --- | --- | --- |
| Markdown 标签差异 | `-`、`*`、`1.` 混用 | strict 误判格式错误 | 在 loose 中统一列表格式 |
| 额外空行 | 多出空段落 | 段落数统计偏大 | 预处理空行，做 normalization |
| 复合指令过多 | 关键词、字数、顺序同时要求 | 任一子项失败就整题失败 | 先拆分子任务，再观察聚合指标 |
| 顺序约束 | 要求先 A 后 B | 内容都有但顺序反了 | 单独加顺序 validator |
| 禁用词约束 | 引号、代码块里出现禁词 | 语义上无害但 strict fail | 明确定义是否忽略代码块/引用 |
| 长提示退化 | 后半段要求被遗忘 | prompt 级分数明显下降 | 对比短版 prompt 与完整版 prompt |

这里最容易误解的一点是：loose 不是“降低标准”，而是“减少假警报”。如果 strict 失败、loose 通过，通常说明模型并非完全没理解，而是输出表层格式和判定器的假设不一致。反过来，如果两者都失败，就更该怀疑模型在执行约束时真的掉链子了。

另一个工程权衡是多约束叠加。比如“必须包含关键词”加“总字数不超过 80”是典型冲突源。模型可能为了满足关键词次数而超字数，也可能为了压缩长度而丢关键词。此时如果 strict 直接判失败，你只知道“没过”，却不知道失败来自理解不足还是约束冲突。更稳妥的做法是先拆成两个 prompt 分别测，再看联合任务的退化幅度。

---

## 替代方案与适用边界

IFEval 不是万能评测，它只是把“是否按要求做”这件事测得很清楚。对于开放式写作、复杂推理解释、审美质量、创意表达，它的覆盖范围明显不够。这时更常见的替代方案是人工评审，或者让更强模型做裁判。

可以直接对比如下：

| 方案 | 优点 | 限制 | 适用场景 |
| --- | --- | --- | --- |
| IFEval | 自动化、可复现、可细分到约束级 | 不擅长评创意与主观质量 | 上线前结构化 QA、回归测试 |
| 人工评审 | 灵活，能判断上下文和质量 | 成本高，主观差异大，难复现 | 高价值样本、开放式任务 |
| LLM-as-a-Judge | 扩展性强，覆盖开放任务 | 评委模型自身有偏差 | 快速大规模质检、辅助人工 |

一个很现实的对比是：IFEval 能告诉你“某模型的 `prompt_level_strict` 是 0.62，`inst_level_loose` 是 0.88`”；人工评审则可能给出“自然度 7/10，帮助性 8/10”。前者适合做工程决策，因为它稳定、可比较；后者适合补足开放质量判断，因为它更灵活。

所以适用边界很明确：

- 如果你在做结构化输出、模板化客服、工具调用前的格式约束、上线回归测试，IFEval 很合适。
- 如果你在做写作助手、头脑风暴、复杂建议生成，仅靠 IFEval 不够，必须叠加主观评审或任务成功率指标。
- 如果你关心长指令退化、冲突约束下的稳定性，IFEval 反而特别有价值，因为这些问题正适合程序化暴露。

---

## 参考资料

1. EvalScope IFEval 基准页：给出了 IFEval 的定义、指标命名，以及 `prompt_level` / `inst_level`、`strict` / `loose` 的评测说明。  
   https://evalscope.readthedocs.io/en/latest/benchmarks/ifeval.html

2. Potato Showcase: IFEval Instruction Following：展示了基于结构化配置做指令约束验证的工程流程和样例，适合理解落地方式。  
   https://www.potatoannotator.com/showcase/ifeval-instruction-following

3. IFEval 相关论文与综述材料：用于理解“可程序验证指令”这套思路的研究背景，以及它与更广义指令遵循评测的关系。  
   https://arxiv.org/abs/2311.07911
