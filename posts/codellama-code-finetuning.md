## 核心结论

Code Llama 是在 Llama 2 基础上继续训练出来的代码模型家族，不是从零开始训练的新架构。它的核心策略可以概括为：先用大量代码 token 做领域持续预训练，再加入中间补全、长上下文微调和指令微调，让通用语言模型更适合代码生成、代码补全、代码解释和仓库级上下文理解。

领域持续预训练，白话说，就是模型已经学过通用语言之后，再拿某个领域的大量数据继续训练，让它更熟悉这个领域的表达规律。Code Llama 使用约 500B code tokens 做代码领域训练，同时保留代码相关自然语言和少量通用自然语言，避免模型只会输出代码片段，却读不懂需求、注释和报错信息。

中间补全，英文常叫 FIM，即 Fill-in-the-Middle，白话说就是“中间空了一段，让模型根据前后文补上”。它能补函数体，是因为训练时模型不只看左边上下文，还会把前缀和后缀一起纳入训练；所以在编辑器里缺一段代码时，它能利用函数签名、后续调用、返回值检查等前后信息生成中间实现。

长上下文微调，英文常叫 LCFT，即 Long Context Fine-Tuning，白话说就是让模型在训练阶段见过更长的输入序列，从而更适合处理多个文件片段、长函数和仓库级上下文。Code Llama 论文中的关键设置是把训练序列从 4096 tokens 提到 16384 tokens，并调整 RoPE 基周期。RoPE 是旋转位置编码，白话说就是模型用来判断 token 相对位置的一套数学规则。

| 阶段 | 作用 | 面向的问题 |
|---|---|---|
| Llama 2 | 通用语言基础模型 | 语言理解、通用生成 |
| 代码持续预训练 | 学习代码分布 | 代码生成、代码解释 |
| FIM | 学会根据前后文补中间 | IDE 补全、局部重写 |
| LCFT | 适配更长上下文 | 多文件片段、仓库理解 |
| 指令微调 | 学会按人类指令回答 | 问答、解释、调试建议 |

更准确的工程结论是：Code Llama 是开源代码模型中的强基线，尤其适合代码补全和代码生成任务；但不应简单写成“达到 GPT-4 级别性能”。公开资料更稳妥的说法是，它在多个开源代码基准上表现强，并不是在所有通用任务、复杂工程任务和真实生产场景中等同 GPT-4。

---

## 问题定义与边界

Code Llama 要解决的不是单纯聊天，而是代码场景里的生成、补全、编辑和理解。代码任务和普通问答的区别在于：代码有语法约束、类型约束、上下文依赖和工程约束。一个模型只会回答“这个函数是什么意思”还不够，它还要能根据已有代码风格补函数、根据调用关系推断参数、根据测试失败定位问题。

自回归目标，白话说，就是模型按从左到右的顺序预测下一个 token。基础语言模型和代码持续预训练大多仍然使用这个目标。它适合续写代码，比如根据注释继续写函数。但它天然只看左侧上下文，对“中间缺一段”的编辑场景不够直接。

FIM 解决的是局部编辑问题。比如 IDE 中已有：

```python
def normalize(xs):
    # cursor here

assert normalize([2, 4]) == [0.5, 1.0]
```

只看左边，模型知道函数名和参数；同时看右边，它还能知道返回值应该满足什么断言。这就是中间补全的意义。

LCFT 解决的是长上下文问题。比如真实仓库里，常量定义在 `config.py`，接口定义在 `types.py`，调用发生在 `service.py`。如果这些片段不能同时进入上下文窗口，模型就只能猜；如果能一起进入注意力范围，模型才有机会做跨文件推理。注意力，白话说，就是模型在生成当前 token 时决定重点参考哪些历史 token 的机制。

| 任务类型 | 需要的上下文 | 对应机制 | 是否依赖长上下文 |
|---|---|---|---|
| 续写注释后的函数 | 主要看左侧文本 | 自回归训练 | 不一定 |
| 补函数体中间缺口 | 前缀 + 后缀 | FIM | 不一定 |
| 根据测试补实现 | 函数签名 + 断言 + 错误信息 | FIM / 指令微调 | 视长度而定 |
| 跨文件重构 | 多个文件片段 | LCFT | 是 |
| 解释代码仓库设计 | 目录、接口、调用链 | LCFT / 指令微调 | 是 |

边界要分清：FIM 不是长上下文，LCFT 也不是中间补全。FIM 改变样本组织方式，让模型学会利用前后文；LCFT 改变训练长度和位置编码设置，让模型在更长序列中保持检索和关联能力。二者可以组合，但不能互相替代。

---

## 核心机制与推导

Code Llama 的第一步是代码领域持续预训练。它仍然使用标准自回归损失：

$$
L_{\text{AR}}=-\sum_t \log p_\theta(x_t\mid x_{<t})
$$

这里的 \(x_t\) 表示第 \(t\) 个 token，\(x_{<t}\) 表示它之前的所有 token，\(p_\theta\) 表示模型在参数 \(\theta\) 下给出的概率。白话解释：模型看到前面的内容后，要把下一个 token 的概率预测得尽量准。

为什么要混入自然语言？因为真实代码任务并不只有代码。需求描述、README、注释、错误日志、提交信息和用户指令都属于代码相关自然语言。如果只做 code-only 训练，模型可能更擅长生成语法正确的片段，但更不擅长理解“请把这个接口改成幂等”这种需求。

一个数值例子：如果有 1000 个训练 token，可以近似理解为 850 个代码 token、80 个代码相关自然语言 token、70 个通用自然语言 token。这个例子不是说每个 batch 都机械保持这个比例，而是说明 Code Llama 的训练目标不是“纯代码”，而是以代码为主、自然语言为辅。

FIM 的训练方式是把一个原始文档切成三段：前缀 \(p\)、中段 \(m\)、后缀 \(s\)。原始顺序是：

```text
[p, m, s]
```

训练时可以重排成：

```text
[p, s, m]
```

或者其他带特殊标记的变体，让模型学会“根据前缀和后缀生成中段”。其损失可以写成：

$$
L_{\text{FIM}}=-\sum_t \log p_\theta(T(x)_t\mid T(x)_{<t})
$$

其中 \(T(x)\) 表示被 FIM 规则重排后的训练样本。白话解释：不是换了预测目标，仍然是预测下一个 token；变化在于输入顺序被设计成了“先给前后文，再让模型生成中间”。

玩具例子如下。原始代码是：

```python
def add_one(x):
    return x + 1

print(add_one(2))
```

可以切成：

| 部分 | 内容 |
|---|---|
| prefix | `def add_one(x):` |
| middle | `return x + 1` |
| suffix | `print(add_one(2))` |

FIM 训练后，模型看到函数签名和后续调用，更容易补出中间实现，而不是只根据函数名猜测。

LCFT 的重点是训练长度。Code Llama 论文描述的长上下文微调把训练序列从 4096 提到 16384 tokens，并把 RoPE 基周期从 10000 调到 1000000。这里不能只理解成“把配置里的最大长度改大”。如果模型训练时没见过足够长的序列，仅在推理时拉长窗口，远处信息经常会被忽略，尤其是 key retrieval 这类需要从长文本中找准某个关键信息的任务。

真实工程例子：一个服务里，`A.py` 定义 `MAX_RETRY = 3`，`B.py` 定义 `RetryPolicy`，`C.py` 调用 `run_with_retry(policy)`。如果要让模型修改重试逻辑，它需要同时看到常量、类型和调用点。LCFT 的价值就是让这些片段能放进同一个长上下文里，并让模型在训练中学过如何处理这种长度。

---

## 代码实现

实现 Code Llama 类似策略时，先不要急着写训练命令。更重要的是训练数据怎么喂给模型：数据混合比例、FIM 样本构造、长序列打包、位置编码配置和 next-token prediction 目标。

新手版流程是：先准备代码文本，再按比例混入代码相关自然语言和通用自然语言；然后决定某些样本是否切开中间，构造成 FIM 样本；最后把样本打包成模型能处理的长序列，继续用“预测下一个 token”的方式训练。

下面是一个可运行的最小 Python 例子，演示数据配比和 FIM 重排。它不是完整训练代码，但能说明训练样本如何从原始代码变成模型输入。

```python
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MixConfig:
    code_ratio: float = 0.85
    code_nl_ratio: float = 0.08
    general_nl_ratio: float = 0.07


def allocate_tokens(total_tokens: int, config: MixConfig) -> dict:
    code = round(total_tokens * config.code_ratio)
    code_nl = round(total_tokens * config.code_nl_ratio)
    general_nl = total_tokens - code - code_nl
    return {
        "code": code,
        "code_related_nl": code_nl,
        "general_nl": general_nl,
    }


def split_for_fim(text: str, start: int, end: int) -> Tuple[str, str, str]:
    assert 0 <= start < end <= len(text)
    prefix = text[:start]
    middle = text[start:end]
    suffix = text[end:]
    return prefix, middle, suffix


def make_fim_sample(prefix: str, middle: str, suffix: str) -> str:
    return (
        "<PRE>" + prefix +
        "<SUF>" + suffix +
        "<MID>" + middle
    )


def pack_long_sequence(samples: List[str], max_seq_len: int) -> List[str]:
    packed = []
    current = ""
    for sample in samples:
        if len(current) + len(sample) > max_seq_len:
            if current:
                packed.append(current)
            current = sample
        else:
            current += sample
    if current:
        packed.append(current)
    return packed


config = MixConfig()
allocation = allocate_tokens(1000, config)
assert allocation == {
    "code": 850,
    "code_related_nl": 80,
    "general_nl": 70,
}

code = "def add_one(x):\n    return x + 1\n\nprint(add_one(2))\n"
prefix, middle, suffix = split_for_fim(code, 16, 33)
fim_sample = make_fim_sample(prefix, middle, suffix)

assert fim_sample.startswith("<PRE>def add_one(x):")
assert "<SUF>" in fim_sample
assert "<MID>" in fim_sample
assert "return x + 1" in fim_sample

packed = pack_long_sequence([fim_sample] * 3, max_seq_len=1000)
assert len(packed) == 1
```

一个配置片段可以写成这样：

```yaml
training:
  objective: next_token_prediction
  base_model: llama-2
  max_seq_len: 16384

data_mix:
  code: 0.85
  code_related_natural_language: 0.08
  general_natural_language: 0.07

fim:
  enabled: true
  format: prefix_suffix_middle
  special_tokens:
    prefix: "<PRE>"
    suffix: "<SUF>"
    middle: "<MID>"

position_encoding:
  type: rope
  base_theta: 1000000
```

这里的 `max_seq_len=16384` 对应长上下文训练长度；`base_theta` 对应 RoPE 的基周期设置。真实训练还需要 tokenizer、去重、许可证过滤、质量过滤、batch packing、分布式训练和评测集，但核心思想仍然是：训练目标没变，训练数据和上下文组织方式变了。

---

## 工程权衡与常见坑

Code Llama 的训练策略本质上是在几个目标之间取平衡：更懂代码、更会编辑、更能处理长上下文，同时尽量不丢掉自然语言理解能力。

第一个坑是过度 code-only。很多团队看到“代码模型”就只喂代码，这会让模型更贴近代码分布，但可能损害需求理解、注释生成、错误解释和文档问答。工程上更稳的做法是保留代码相关自然语言和少量通用自然语言。

第二个坑是把长上下文当成纯推理参数。只在推理时调大 RoPE scaling 或上下文窗口，不等于模型真的具备长上下文能力。模型在训练期没学过足够长的序列，就容易在远距离依赖上退化。表现出来就是：仓库片段一长，模型忘掉远处定义，或者引用了错误的类型、常量和函数签名。

第三个坑是把 100K context 当成训练长度。Code Llama 公开材料中应区分训练窗口和推理外推能力。论文里的 LCFT 训练长度是 16K，100K 更接近推理阶段的外推描述，不能写成“训练时用了 100K”。

第四个坑是忽略具体 checkpoint 的模型卡。checkpoint，白话说，就是一次训练后保存下来的模型权重文件。不同 checkpoint 的能力不同，有些偏 Python，有些偏指令问答，有些支持 FIM，有些未必适合 IDE 中间补全。部署前必须核对模型卡，而不是只看家族名称。

| 坑点 | 现象 | 规避方法 |
|---|---|---|
| 过度 code-only | 能写片段，但读不懂需求、注释和报错 | 保留 code-related NL 和少量通用 NL |
| 只做推理缩放 | 长输入可放入，但远处信息用不好 | 做 LCFT，并用长上下文任务评测 |
| 把 100K 当训练长度 | 对训练成本和能力边界判断错误 | 明确区分 16K 训练窗口和推理外推 |
| 忽略模型卡 | IDE 补全、中间填充或指令能力不符合预期 | 部署前核对 checkpoint 支持能力 |
| 只看 HumanEval | 线上仓库任务表现不稳定 | 增加真实仓库、单测修复、跨文件评测 |

真实部署例子：某团队把一个普通 code checkpoint 接进 IDE，以为它能补中间代码。结果用户在函数中间删除几行后，模型只会沿着左侧继续写，无法利用右侧断言和调用点。原因是该 checkpoint 没有显式支持 FIM，或者推理模板没有按 FIM 格式组织前后文。解决方法不是只调 temperature，而是换支持 FIM 的模型或修正输入格式。

---

## 替代方案与适用边界

Code Llama 的“持续预训练 + FIM + LCFT”适合做强代码基础模型，但不是所有代码场景都该走这条路。训练成本、数据规模、许可证合规、评测体系和推理成本都很高。对于很多企业内部场景，检索增强或轻量微调可能更合适。

LoRA，白话说，是一种低成本微调方法，只训练少量附加参数，不改动大部分原始模型权重。它适合企业内部风格适配，比如让模型更熟悉某个 SDK、代码规范或接口命名方式。但 LoRA 通常不能从根本上补齐基础模型缺失的代码能力，也不能单独解决长上下文训练不足的问题。

RAG，即 Retrieval-Augmented Generation，白话说，就是先从文档或代码库里检索相关片段，再把片段交给模型回答。它适合客服式代码问答、内部文档问答和 API 使用说明，不一定需要重新训练模型。但 RAG 的生成质量仍然受基础模型能力影响，而且检索片段错了，回答也会偏。

| 方案 | 成本 | 长上下文能力 | 编辑器补全能力 | 仓库理解能力 | 适用场景 |
|---|---:|---|---|---|---|
| 从头训练代码模型 | 极高 | 可设计 | 可设计 | 可设计 | 资源充足的大模型团队 |
| Code Llama 式持续预训练 + FIM + LCFT | 高 | 强，取决于训练 | 强，取决于 FIM | 较强 | 大规模代码补全、代码生成平台 |
| 通用模型 + LoRA | 低到中 | 依赖基座 | 有限 | 有限 | 企业风格适配、少量领域术语 |
| RAG / 检索增强 | 中 | 依赖检索窗口 | 不直接解决 | 较适合问答 | 内部文档、代码问答、API 查询 |
| 只做长上下文扩展 | 中 | 可能增强 | 不解决中间补全 | 部分增强 | 长文档阅读、日志分析 |
| 只做指令微调 | 中 | 不直接增强 | 不直接增强 | 依赖基座 | 问答格式、解释风格、对话行为 |

选择题式例子：

| 需求 | 更合适的方案 |
|---|---|
| 只需要回答“这个内部 SDK 怎么用” | 通用模型 + RAG |
| 只需要适配公司代码风格 | 通用代码模型 + LoRA |
| 要做 IDE 中大规模函数补全和局部重写 | FIM 代码模型 |
| 要处理跨文件调用链、长配置和多模块重构 | LCFT 后的长上下文代码模型 |
| 要从零打造开源代码基础模型 | 持续预训练 + FIM + LCFT + 指令微调 |

因此，不该把 Code Llama 策略理解为“所有代码 AI 的唯一答案”。如果只是客服式代码问答，优先考虑通用模型加检索；如果目标是编辑器补全、局部重写和仓库级代码生成，才更应该考虑类似 Code Llama 的训练路线。

---

## 参考资料

1. [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950)
2. [Introducing Code Llama, an AI Tool for Coding](https://about.fb.com/news/2023/08/code-llama-ai-for-coding/)
3. [Meta Code Llama GitHub Repository](https://github.com/meta-llama/codellama)
4. [Code Llama MODEL_CARD.md](https://github.com/meta-llama/codellama/blob/main/MODEL_CARD.md)
5. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

本文中关于 16K 训练窗口、RoPE 调整、FIM 和代码/NL 混合比例的描述，以论文和官方模型卡为准。
