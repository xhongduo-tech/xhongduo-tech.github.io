## 核心结论

CodeGemma 的关键做法不是“把代码模型做得更大”，而是把“代码补全”和“数学/自然语言推导”放进同一套继续预训练目标里。续训，白话说，就是在已有基座模型上继续喂新数据和新任务。公开模型卡说明，CodeGemma 2B/7B 是在 Gemma 基础上继续训练额外的约 500B 到 1000B token；公开分发说明和社区整理通常将 7B 的主配比概括为约 80% 代码、20% 自然语言与数学相关数据。这个配比的意义很直接：大头代码语料负责记住 API、缩进、调用模式，小头数学与文本语料负责维持符号推导、注释解释和步骤表达能力。

更重要的是，CodeGemma 没有只训练传统的从左到右生成，而是把大部分样本做成 FIM。FIM 是 fill-in-the-middle，白话说，就是让模型学会“在中间插空补全”。官方模型卡给出的设置是 80% 到 90% 的样本采用 FIM，并且 PSM 与 SPM 各占 50%。这意味着模型不是只会顺着往后写，而是能同时利用前缀和后缀来决定中间该填什么。

对初级工程师最有价值的结论有两条。

| 语料类别 | 作用 | 常见收益 |
| --- | --- | --- |
| 开源代码 | 记住函数体、库调用、项目结构 | 补全更像真实代码 |
| 数学与自然语言 | 记住公式表达、推导顺序、解释文字 | 注释、证明、题解更连贯 |

第一，代码和数学联合训练带来的不是“会算题”这么简单，而是让模型在写代码时更容易维持逻辑链。例如让它补 `def area(r):` 的函数体时，它既可能写出 `return math.pi * r**2`，也更可能补出解释半径与面积关系的注释。第二，FIM 训练让模型更适合 IDE 中间补全、重构插入、缺口修复这类真实工程场景，因为工程里经常不是“从空白开始写”，而是“在已有前后文之间补一段”。

---

## 问题定义与边界

这里的问题不是“CodeGemma 会不会写代码”，而是“为什么代码模型要额外混入数学与自然语言数据，并用 FIM 而不是纯左向训练”。

先看问题定义。代码生成任务里，模型常常要同时满足三类约束：语法约束、语义约束、推理约束。语法约束是代码能不能过解释器；语义约束是 API 和变量关系对不对；推理约束是公式、边界条件、注释说明是否自洽。只用代码训练，前两类通常够强，但第三类容易断。只用数学或文本训练，第三类可能增强，但代码细节会退化。CodeGemma 的联合训练，本质上是在同一组参数里平衡这三类约束。

再看 prompt 结构。L2R 是 left-to-right，白话说，就是只看前文顺着写后文。PSM 是 prefix-suffix-middle，先给前缀和后缀，再让模型补中间。SPM 是 suffix-prefix-middle，先给后缀再给前缀，最后仍然补中间。不同结构决定模型“先看到什么”。

| 结构 | 描述 | 最适用场景 |
| --- | --- | --- |
| L2R | 只给 prefix，按自然顺序继续生成 | 传统续写 |
| PSM | prefix + suffix + `<|fim_middle|>` | 已做 FIM 预训练的模型 |
| SPM | suffix + prefix + `<|fim_middle|>` | 兼容某些非 FIM 左向模型的变体用法 |

玩具例子可以说明边界。假设你在 IDE 里有下面这段代码，光标在 `return` 后面：

```python
def area(r):
    return 
print(area(2))
```

如果用 L2R，模型只知道前面是 `return`，很容易补出局部合理但与后文联系弱的内容。如果用 FIM，模型同时看到后面的 `print(area(2))` 和前面的函数定义，它更容易补出一个完整表达式，并保持函数返回值类型和后续调用一致。

边界也要说清楚。代码与数学联合训练提升的是“代码中带推理”的能力，不等于它变成了严格符号证明器。FIM 提升的是“中间补全”的建模能力，不等于所有编辑场景都优于普通自回归生成。特别是长链证明、形式验证、多文件依赖解析，仍然强依赖上下文质量和推理深度，而不只取决于训练配比。

---

## 核心机制与推导

CodeGemma 的核心机制有两层：数据混合机制和目标函数机制。

第一层是数据混合。代码数据提供高频模式，比如函数签名、循环、异常处理、测试结构。数学与自然语言数据提供低频但关键的“解释性结构”，比如“已知条件 -> 推导步骤 -> 结论”这种线性链条。对模型来说，这会影响条件概率分布。一个只看过代码的模型，在注释里更容易输出模板化解释；一个混入数学文本的模型，更容易把代码行为和公式关系对齐。

第二层是 FIM 目标。token 是分词后的最小建模单位，白话说，就是模型一次预测的最小文本块。FIM 不是改变模型结构，而是改变训练样本排列：把原文拆成前缀 $P$、中间段 $M$、后缀 $S$，输入里先给 $P$ 和 $S$，让模型预测 $M$。其损失仍然是标准交叉熵：

$$
\mathcal{L}_{\text{FIM}} = -\sum_{t=1}^{|M|} \log P_\theta(m_t \mid P, S, m_{<t})
$$

这条式子的含义很直接：在已知前缀、后缀和已经生成的中间部分时，最大化真实中间 token 的概率。也就是说，模型学习的不是“下一句怎么写”，而是“在前后文都固定时，中间最合理的内容是什么”。

一个更具体的玩具例子如下。

前缀：

```text
def solve(delta):
    return
```

后缀：

```text
print(math.sqrt(delta))
```

如果训练样本构造成：

```text
<|fim_prefix|>def solve(delta):
    return<|fim_suffix|>
print(math.sqrt(delta))<|fim_middle|>
```

那么模型必须学会：中间既要让 `return ...` 成立，又不能和后面的 `print(math.sqrt(delta))` 逻辑冲突。只要它知道平方根要求非负，并能关联 `delta` 的后续用途，就更可能补出与数学操作一致的表达。

真实工程例子更能体现协同效应。设想一个风控系统里有一段利率计算函数，工程师在函数体中间留空，希望模型补出“先年化、再折现、最后裁剪边界”的逻辑，并自动加一段说明公式来源的注释。这时代码数据负责让模型写出正确的 Python 结构，数学/文本数据负责让注释和公式关系不乱，FIM 负责让它利用前后的变量名、返回值和调用处约束去补完整中段。这三者缺一项，输出都容易退化。

---

## 代码实现

CodeGemma 的 FIM 控制主要依赖四个特殊 token：`<|fim_prefix|>`、`<|fim_suffix|>`、`<|fim_middle|>`、`<|file_separator|>`。前 3 个定义前缀、后缀和待补位置，最后一个通常用作多文件分隔或生成终止标记。

下面给出一个可运行的最小示例。它不调用真实模型，而是先把 FIM prompt 的构造与解析逻辑跑通，这样更适合初学者先理解协议本身。

```python
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FILE_SEPARATOR = "<|file_separator|>"

def build_fim_prompt(prefix: str, suffix: str) -> str:
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

def extract_completion(model_output: str) -> str:
    # 真实模型通常会回显 prompt，这里只取 middle 之后到 file_separator 之前的补全部分
    middle_pos = model_output.rfind(FIM_MIDDLE)
    assert middle_pos != -1, "missing <|fim_middle|>"
    tail = model_output[middle_pos + len(FIM_MIDDLE):]
    return tail.split(FILE_SEPARATOR, 1)[0]

prefix = "def area(r):\n    return "
suffix = "\nprint(area(2))"
prompt = build_fim_prompt(prefix, suffix)

fake_model_output = (
    prompt
    + "math.pi * r**2"
    + FILE_SEPARATOR
)

completion = extract_completion(fake_model_output)

assert prompt.startswith(FIM_PREFIX)
assert FIM_SUFFIX in prompt and prompt.endswith(FIM_MIDDLE)
assert completion == "math.pi * r**2"

print(completion)
```

如果换成真实推理，请求通常会像这样构造：

```python
prompt = (
    "<|fim_prefix|>"
    "def solve(delta):\n"
    "    return"
    "<|fim_suffix|>"
    "\nprint(math.sqrt(delta))"
    "<|fim_middle|>"
)
```

工程上要注意两点。第一，推理结果里模型往往会回显整个 prompt，因此不能直接把原始输出当作补全结果，要截取 `<|fim_middle|>` 之后、`<|file_separator|>` 之前的部分。第二，空格、换行、缩进都属于上下文的一部分，漏掉一个缩进，补全质量就可能明显下降。官方示例也明确强调，FIM token 与 prefix/suffix 之间不要额外插入空白。

---

## 工程权衡与常见坑

FIM 的主要收益是中间补全能力，但它的坑也刚好出在“边界”。

最常见的问题是 sub-token 断裂。sub-token，白话说，就是一个完整词被拆成两个或更多分词片段。传统 token 级 FIM 在 prefix、middle、suffix 交界处容易把单词截断，比如 `print` 被拆成前缀里的 `pr` 和待补里的 `int`。这样一来，模型在 `<|fim_middle|>` 后的第一个预测就会异常困难，因为它不是在预测一个自然起点，而是在预测一个“半截词”的后半段。ACL 2024 的 FIM-SE 论文就是针对这个问题：它通过行级格式和额外控制标记，避免推理时去预测这类半截 token。

| 任务 | FIM-SE 相对提升 |
| --- | --- |
| 单行 infill | +11.5% pass@1 |
| 多行 infill | +10.7% pass@1 |

第二个坑是“补全正确，但插入点错位”。例如模型知道应该补 `elif n == 1:`，但因为前缀最后没有保留换行或缩进，它把结果插进了错误的列。对代码模型来说，语义对了不代表工程可用，缩进错一格就是语法错误。

第三个坑是“解释链和代码链分离”。这在数学相关任务尤其明显。模型可能写出正确公式，却在注释里给出错误解释；或者注释正确，代码变量却没对齐。这恰好说明联合训练的价值，但也说明它不是完全解决方案。生产环境里仍然要靠单元测试、静态检查和少量规则模板兜底。

真实工程例子是智能 IDE 的公式补全。假设财务系统里有一段空白，需要补“计算月供并在注释中说明等额本息公式”。token 级 FIM 可能在边界处先吐出不完整子词，或者把注释和代码混写成半句，导致编辑器体验差。FIM-SE 这类行级方案更稳，因为它把补全目标限制为整行或多行单位，更符合人类在编辑器里的操作粒度。但代价也很明确：你需要额外的行级标记、不同的数据预处理和配套解码逻辑。

---

## 替代方案与适用边界

如果手里的模型没有做过 FIM 预训练，最直接的替代不是硬塞 FIM token，而是回到左向生成或使用近似的 SPM 风格重排。原因很简单：模型没学过这些控制 token，就不会稳定理解它们的语义。此时把后缀提前，有时能让模型“先知道目标长什么样”，再去顺着补前面缺口附近的内容，但这种方法本质上还是左向生成，不是真正的中间建模。

| 方案 | 适用条件 | 缺点 |
| --- | --- | --- |
| L2R | 从空白续写、模型未做 FIM 预训练 | 不擅长利用后缀约束 |
| PSM Token FIM | 模型已做 FIM 预训练，追求灵活补全 | 边界处可能出现 sub-token 问题 |
| 行级 FIM / FIM-SE | 关注整行一致性、IDE 编辑体验 | 需要额外格式和训练支持 |

什么时候该用哪一种，可以按任务粒度判断。

如果只是补一个函数尾巴，比如补 `return` 后面的表达式，普通 PSM 足够，灵活且实现简单。如果要补的是一整段含代码、注释和公式说明的内容，行级方案通常更稳，因为它更少出现“半个词起步”的边界噪声。如果是没有 FIM 训练的通用模型，那就不要假设它天然会处理中间插空，应优先把任务改写成传统续写、检索增强，或者显式分步生成。

这里也要给出适用边界。CodeGemma 的“代码与数学联合训练”最适合代码中带解释、公式、边界条件、测试样例的任务，比如算法题解、科学计算脚本、教育型代码生成、需要注释推导链的企业内部工具。它不一定适合严格形式证明、超长跨文件重构、需要完整依赖图理解的大型仓库级修改。这些任务往往需要检索、工具调用或更强的项目级上下文，而不是只靠训练配比和 FIM 结构。

---

## 参考资料

- [Google AI for Developers: CodeGemma model card](https://ai.google.dev/gemma/docs/codegemma/model_card)
- [Google AI for Developers: CodeGemma prompt structure](https://ai.google.dev/gemma/docs/codegemma/prompt-structure)
- [Google AI for Developers: AI assisted programming with CodeGemma and KerasNLP](https://ai.google.dev/gemma/docs/codegemma/code_assist_keras)
- [Hugging Face: google/codegemma-1.1-2b README](https://huggingface.co/google/codegemma-1.1-2b/blob/main/README.md)
- [ACL 2024: Empowering Character-level Text Infilling by Eliminating Sub-Tokens](https://aclanthology.org/2024.acl-long.179.pdf)
- [UC Berkeley EECS-2025-50: Advancing Large Language Models for Code Using Code-...](https://digicoll.lib.berkeley.edu/record/320662/files/EECS-2025-50.pdf)
