## 核心结论

`lm-evaluation-harness`，常简称 `lm-eval` 或 `LEH`，本质上是一套“把评测过程参数化”的框架。白话说，它不是只帮你跑一次 benchmark，而是把“任务定义、few-shot 样例、prompt 模板、模型调用方式、指标计算”固定成一条可复现流水线。这样同一个模型在不同时间、不同机器、不同人手里，才能尽量得到可比较的结果。

它的关键价值不在“能跑 MMLU/HellaSwag/ARC/TruthfulQA”，而在“用统一方式跑这些任务”。官方实现里，任务通常由 YAML 配置描述，再由 `TaskManager` 读取成 `TaskConfig`，最后结合 `ContextSampler` 组装 few-shot 上下文并送给模型打分。对多选题，最常见度量是 accuracy，也就是答对比例。

玩具例子可以先看 `hellaswag`。你把任务写成一个 YAML，至少声明 `task`、`dataset_path`、`output_type` 等字段，再执行 `lm-eval run --tasks hellaswag` 或兼容入口 `lm_eval --tasks hellaswag ...`，框架就会自动完成“读数据、构 prompt、调用模型、聚合指标”的全流程。新手不需要自己手写每道题的 prompt 拼接逻辑。

任务分数通常可以抽象成：

$$
score_{task}=\frac{\sum_{q\in Q} metric(q)}{|Q|}
$$

如果 `metric(q)` 是多选题是否答对，那么它取值就是 0 或 1；整组题的分数就是平均准确率。

| 维度 | LEH | 传统手写 benchmark 脚本 |
|---|---|---|
| 输入 | YAML + 任务名 + 模型参数 | Python 脚本 + 数据处理代码 |
| 配置表达 | 任务字段显式写出 | 常散落在代码常量里 |
| 输出 | 标准结果表与指标聚合 | 结果格式不统一 |
| few-shot 控制 | `num_fewshot`、`fewshot_split` 等字段 | 常需手写抽样逻辑 |
| 复现要点 | 版本库保存 YAML 和任务实现 | 很容易漏掉 prompt 与样例顺序 |

---

## 问题定义与边界

这类框架要解决的问题其实很具体：怎么让“同一模型在同一任务上的分数”具备比较意义。因为 LLM 评测里，模型之外还有很多变量，例如 prompt 写法、few-shot 样例条数、样例顺序、候选答案格式、指标聚合方式。只要其中一个变了，分数就可能变化。

`few-shot` 术语先说明一下。它的意思是在正式待测问题前，先给模型塞几道“题目+标准答案”的样例，让模型模仿格式和解题方式。白话说，就是先给它看几题示范，再让它做新题。LEH 的价值就在于把这部分也标准化。

本文边界限定在以下范围：

| 范围 | 包含 | 不包含 |
|---|---|---|
| 数据来源 | 公开 benchmark，通常来自 Hugging Face datasets | 需要人工在线标注的私有流程 |
| 任务定义 | YAML 配置、任务组、内置任务目录 | 完全临时的 notebook 试验 |
| few-shot 机制 | `ContextSampler` 从指定 split 抽样 | 人工手写、每次都变的示例 |
| 运行方式 | CLI 或 `TaskManager` + Python API | 自建整套评测后端 |
| 目标 | 可复现、可比较、可版本化 | 一次性探索、快速调 prompt |

一个新手场景是：你想在 `TruthfulQA` 上测试模型是否容易“说得像真话但其实不真”。如果任务已经内置，你通常只需要指定任务名与模型参数；如果要自定义变体，只需要 YAML 中声明 `task: truthfulqa`、`output_type: multiple_choice` 以及数据与模板字段，不需要每次重写整套 prompt 代码。

这里要特别强调边界：LEH 解决的是“标准化评测”，不是“所有评测”。如果你的数据是公司内部客服日志、答案规则经常改、还需要人工复核，那它就不是最合适的一站式方案。

---

## 核心机制与推导

`TaskManager` 可以理解为任务管理器，白话说就是“把任务名字解析成可执行评测对象”的那层。它会扫描内置任务目录，也可以通过 `include_path` 额外加载自定义 YAML。YAML 被解析成 `TaskConfig` 后，框架才知道这项任务的数据在哪、要怎么把样本变成 prompt、该取什么输出类型。

`TaskConfig` 是任务配置对象，白话说就是“任务说明书”。里面常见的字段包括：

- `task`：任务名
- `dataset_path` / `dataset_name`：数据集位置
- `output_type`：输出类型，如 `multiple_choice` 或 `generate_until`
- `doc_to_text`：把样本转成输入文本的模板
- `doc_to_target`：标准答案模板
- `doc_to_choice`：多选题的候选项
- `num_fewshot`：每题前面放几道示范题

`ContextSampler` 是上下文采样器，白话说就是“负责挑 few-shot 样例并塞进 prompt 的部件”。当 `num_fewshot=2` 时，它会从指定 split 里按既定规则抽两条样例，拼到当前待测题目前面。于是模型看到的不是孤立的一道题，而是：

1. 任务描述
2. 两道示范题与答案
3. 当前待测题

这就形成了“配置驱动 prompt”的链路：

`YAML -> TaskConfig -> ContextSampler 组装上下文 -> 模型推理 -> metric 聚合`

玩具例子可以这样理解。假设有 4 道多选题，`num_fewshot=2`，每道待测题前都插入两道固定示例。模型最终答对 3 题，则：

$$
score_{task}=\frac{1+1+1+0}{4}=0.75
$$

也就是 75%。如果你保持同样的 YAML、同样的 few-shot 样例来源、同样的模板和同样的模型版本，这个结果就具有复现性。反过来，只要把 few-shot 顺序偷偷改掉，历史对比就失效了。

---

## 代码实现

先看一个极简的任务配置。下面不是官方完整文件，而是足以帮助理解职责边界的简化版：

```yaml
task: arc_challenge
dataset_path: ai2_arc
dataset_name: ARC-Challenge
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
fewshot_split: train
num_fewshot: 5
doc_to_text: "{{question}}\nAnswer:"
doc_to_choice: "{{choices.text}}"
doc_to_target: "{{answerKey}}"
metric_list:
  - metric: acc
metadata:
  version: 1
```

这份 YAML 表达的是：任务是 `ARC-Challenge`，输出形式是多选，few-shot 从训练集取 5 条，指标看 accuracy。也就是说，评测逻辑不是散落在 Python if-else 中，而是尽量被声明式写出来。

简化后的运行流程大致如下：

```python
from dataclasses import dataclass

@dataclass
class TaskConfig:
    task: str
    output_type: str
    num_fewshot: int
    dataset_path: str

class TaskManager:
    def __init__(self):
        self.registry = {}

    def register(self, config: TaskConfig):
        self.registry[config.task] = config

    def get(self, task_name: str) -> TaskConfig:
        return self.registry[task_name]

class ContextSampler:
    def build_context(self, docs, current_doc, num_fewshot: int):
        fewshot = docs[:num_fewshot]
        prompt = ""
        for ex in fewshot:
            prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
        prompt += f"Q: {current_doc['q']}\nA:"
        return prompt

def score_task(results):
    # results 中的 1 表示答对，0 表示答错
    score = sum(results) / len(results)
    assert 0.0 <= score <= 1.0
    return score

assert score_task([1, 1, 1, 0]) == 0.75
```

这个 `python` 例子是可运行的，目的是说明 LEH 的三个核心职责：

| 组件 | 职责 | 你应该关心什么 |
|---|---|---|
| `TaskManager` | 发现、加载、注册任务 | 自定义任务有没有被扫描到 |
| `TaskConfig` | 描述数据、模板、指标、few-shot | YAML 是否把关键变量写全 |
| `ContextSampler` | 选择并拼接示例 | few-shot 来源、顺序、条数是否固定 |

真实工程例子是：团队发布一个新 checkpoint，需要和上周版本对比 `hellaswag,mmlu,arc_challenge,truthfulqa_mc`。如果用 LEH，做法通常是固定一份任务列表与参数模板，再把模型版本换掉即可。评测表中保留相同的任务定义、few-shot 设置和指标，结果才能进入持续对比面板。否则你只能得到一堆“看起来像分数，但没有可比性”的数字。

如果你要加载自定义任务，当前官方设计是通过 `TaskManager(include_path=...)` 或等价路径，把外部 YAML 纳入注册表，再交给 `simple_evaluate()` 或 CLI 使用。这一点很重要，因为它把“新增任务”从“改框架源码”变成了“增加配置与模板”。

---

## 工程权衡与常见坑

LEH 的核心优势是统一，但统一也意味着你要接受一些约束。最常见的工程问题不是“跑不起来”，而是“跑出来的结果其实不可比”。

| 常见坑 | 影响 | 规避策略 |
|---|---|---|
| 改了 `num_fewshot` | 分数变化，历史表失效 | 把 shot 数写入 YAML 并版本化 |
| 改了 prompt 模板 | 同任务结果不再同口径 | 固定 `doc_to_text/doc_to_target` |
| 改了 few-shot 样例顺序 | 分数波动且难排查 | 固定采样规则与种子 |
| 漏写 `tag` 或 `group` | 任务难被批量发现或汇总 | 任务配置中显式补齐组织字段 |
| 子任务聚合没写清 | “平均分”没有解释力 | 在 group YAML 统一 `aggregate_metric_list` |
| 只保存结果不保存配置 | 事后无法追责差异来源 | 结果、YAML、代码版本一起存档 |

组任务聚合是一个容易被忽略的点。比如你想把若干 `mmlu_*` 子任务合并成一个组，如果没有明确写聚合方式，那么表里的平均值可能无法解释，特别是在不同子任务样本量不同、指标不同的情况下。

一个合规的 group YAML 片段可以写成这样：

```yaml
group: my_reasoning_suite
task:
  - hellaswag
  - arc_challenge
aggregate_metric_list:
  - metric: acc
    weight_by_size: true
metadata:
  version: 1
```

这里 `aggregate_metric_list` 的含义是“定义组结果怎么汇总”。白话说，就是不要让“平均”这个词悬空。到底是简单平均、按样本数加权，还是别的聚合，都要写出来。

还有一个常见误区是把 LEH 当作“任何分数都天然公平”的保证器。它只能保证“在同一配置下复现”，不能自动保证“跨论文绝对公平”。如果两个团队用了不同 prompt 变体、不同子任务列表、不同去污染策略，即便都使用 LEH，也不能直接把数字强行横比。

---

## 替代方案与适用边界

替代方案最直接，就是自己写 Python 脚本：读数据、拼 prompt、调模型、算分。这个方式的优点是快，尤其适合快速原型。比如你今天只想粗略看一个模型在 MMLU 的表现，自己写几十行代码也能得到一个数字。

但问题在于，这个数字往往缺少“口径”。你可能忘了记录 few-shot 条数，可能每次随机抽不同示例，可能 prompt 改过三版却没存档。这样得到的不是标准 benchmark 结果，而是某次实验的局部观测。

| 方案 | 复现性 | 易用性 | 维护成本 | 可对比性 |
|---|---|---|---|---|
| LEH | 高 | 中 | 中 | 高 |
| DIY 脚本 | 低到中 | 高 | 高 | 低 |
| Notebook 临时实验 | 低 | 很高 | 很低 | 很低 |

初学者可以这样对比：

- 用 DIY 脚本测 MMLU：你得自己决定抽几条 few-shot、题目怎么排版、答案怎么映射、最后怎么算平均。
- 用 LEH 测 MMLU：你主要决定模型参数与任务列表，few-shot、模板、指标口径由任务配置承载。

所以 LEH 适合的场景是：公开 benchmark、需要复现实验、需要多人协作、需要长期看表格趋势。不太适合的场景是：只做一次探索、评测数据完全私有且字段经常变、重点不在 benchmark 而在人工业务指标。

换句话说，LEH 不是“最灵活”的评测方案，但它是“最适合把评测做成工程资产”的方案之一。

---

## 参考资料

1. EleutherAI 官方仓库 README  
   作用：说明项目定位、CLI 基本用法、内置 benchmark 范围与模型后端支持。  
   链接：https://github.com/EleutherAI/lm-evaluation-harness

2. 官方 `Task Guide`  
   作用：系统列出 `TaskConfig` 字段，包括 `task`、`tag`、`dataset_path`、`output_type`、`num_fewshot`、`metric_list` 等，是理解 YAML 任务定义的核心文档。  
   链接：https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md

3. 官方 Releases 说明中的 v0.4.0 变更  
   作用：解释为什么现在强调 `TaskManager` 与配置化任务创建，以及 `include_path` 加载自定义任务的推荐方式。  
   链接：https://github.com/EleutherAI/lm-evaluation-harness/releases

4. 官方任务目录 `lm_eval/tasks/`  
   作用：查看 `HellaSwag`、`MMLU`、`ARC`、`TruthfulQA` 等内置 benchmark 的具体组织方式与 YAML 实现。  
   链接：https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks

5. 官方文档中关于 filter、重复采样与配置继承的部分  
   作用：理解高级任务特性，如多次生成、后处理、聚合与 `include` 复用。  
   链接：https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
