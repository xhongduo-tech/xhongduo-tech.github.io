## 核心结论

FLAN-UL2 是把 **UL2 的统一去噪预训练** 和 **FLAN 的指令微调** 串起来的编码器-解码器模型：先让模型学会多种语言建模方式，再让模型学会按自然语言指令完成任务。

它可以写成一个两阶段目标：

$$
\text{FLAN-UL2}=\text{UL2 预训练}+\text{FLAN 指令微调}
$$

新手版本可以这样理解：第一阶段让模型做“补全文本训练”，例如补出句子里被遮住的词；第二阶段让模型做“按说明书回答问题训练”，例如看到“请总结这段文本”后输出摘要。前者让模型会语言，后者让模型会听指令。

| 阶段 | 训练目标 | 输入形式 | 输出形式 | 解决的问题 |
|---|---|---|---|---|
| UL2 预训练 | 统一去噪 | 被遮蔽或截断的文本 | 缺失文本或后续文本 | 建立通用语言能力 |
| FLAN 指令微调 | 按指令生成答案 | 任务说明 + 上下文 + 问题 | 自然语言答案 | 提升指令遵循与任务泛化 |

流程可以简化为：

```text
原始文本
  -> UL2 统一去噪预训练
  -> 通用 seq2seq 模型
  -> FLAN 指令数据微调
  -> FLAN-UL2
  -> 指令输入 -> 生成答案
```

FLAN-UL2 的重点不是只优化某一个任务，而是把通用生成、few-shot、问答、摘要、分类、推理等任务放进同一个 encoder-decoder 框架里统一处理。encoder-decoder 是一种“先读输入、再生成输出”的模型结构：encoder 负责理解输入，decoder 负责生成答案。

---

## 问题定义与边界

FLAN-UL2 解决的问题是：如何在一个模型里同时兼顾 **通用语言能力**、**任务泛化** 和 **指令遵循**。

UL2 是一种统一语言学习方法，白话说就是用多种“遮住文本再恢复”的方式训练模型，而不是只用一种预训练目标。FLAN 是指令微调方法，白话说就是把大量 NLP 任务改写成“人类可读的任务说明”，再让模型学习“看到说明后输出答案”。instruction tuning 指令微调，指的是用“任务描述 -> 标准答案”的数据继续训练模型。encoder-decoder 编码器-解码器，指的是模型先把输入整体编码，再一步步生成输出。

输入输出形式如下：

```text
指令 + 上下文 + 问题 -> FLAN-UL2 -> 生成答案
```

玩具例子：

```text
输入：
请用一句话解释法国首都是什么。

输出：
法国的首都是巴黎。
```

真实工程例子：

```text
输入：
任务：根据公司退款政策回答用户问题
上下文：退款通常在审核通过后 3 到 5 个工作日内原路返回。
问题：退款多久到账？
回答：

输出：
退款通常会在审核通过后 3 到 5 个工作日内到账。
```

这不是训练一个“只会背答案”的分类器。分类器通常输出固定标签，例如“可退款 / 不可退款”。FLAN-UL2 更适合生成式任务：读懂问题、结合上下文、生成自然语言答案。

| 维度 | 内容 |
|---|---|
| 要解决的问题 | 在统一模型中兼顾语言理解、生成、few-shot、指令遵循 |
| 不解决的问题 | 不保证低成本部署，不专门优化超低延迟聊天，不是纯 decoder-only 续写模型 |
| 适合场景 | 文档问答、企业 FAQ、工单助手、研究摘要、长输入理解 |
| 不适合场景 | 前端实时补全、极低延迟对话、大规模开放式续写、只需简单分类的任务 |

边界要明确：FLAN-UL2 不是纯 decoder-only 语言模型。decoder-only 是“只用生成器从左到右续写”的结构，常见于很多聊天模型和代码补全模型。FLAN-UL2 也不是靠少量样本就一定稳定泛化的轻量方案。它的优势来自大规模预训练和大规模指令微调，工程使用时仍然要考虑数据覆盖、上下文长度和推理成本。

---

## 核心机制与推导

UL2 阶段的核心是 corruption。corruption 指“故意破坏输入文本”，例如遮住一部分 token、截断后半句，要求模型恢复目标文本。token 是模型处理文本的基本单位，可以近似理解为“词、字或词片段”。

UL2 不只使用一种破坏方式，而是使用 Mixture-of-Denoisers。Mixture-of-Denoisers 是“多种去噪任务的混合”，即从多个训练模式中采样一种，让模型适应不同的补全方式。

设原始文本为 $x$，模式为 $m\in\{R,X,S\}$，腐化函数为 $c_m(\cdot)$，目标输出为 $y$。UL2 预训练目标可以写成：

$$
L_{\text{UL2}}=\mathbb{E}_{m\sim \pi}\,\mathbb{E}_{(x,y)}\big[-\log p_\theta(y\mid c_m(x), m)\big]
$$

这句话的意思是：从模式分布 $\pi$ 里抽一个模式 $m$，把原始文本 $x$ 破坏成 $c_m(x)$，再让模型以最大概率生成目标 $y$。$-\log p_\theta$ 是负对数似然，白话说就是“正确答案概率越低，损失越大”。

| Mode | 名称 | 典型做法 | 训练出的能力 |
|---|---|---|---|
| R | regular span corruption | 遮住普通比例的连续片段 | 常规文本理解与补全 |
| X | extreme span corruption | 遮住更大比例的片段 | 从少量线索恢复更多内容 |
| S | sequential / prefix-LM | 给前缀，预测后续 | 类似自回归续写能力 |

玩具例子：句子 `Paris is the capital of France.` 被处理成：

```text
输入：Paris is the capital of [mask].
目标：France
```

如果模型给 `France` 的概率是 $0.2$，这个 token 的损失就是：

$$
-\log(0.2)\approx 1.61
$$

概率越接近 1，损失越小；概率越接近 0，损失越大。

为什么要统一到 seq2seq？seq2seq 是“序列到序列”，也就是输入一段序列，输出另一段序列。问答、摘要、翻译、分类都可以改写成 seq2seq：

| 原任务 | seq2seq 改写 |
|---|---|
| 分类 | 输入文本，输出类别名 |
| 摘要 | 输入长文，输出摘要 |
| 问答 | 输入问题和上下文，输出答案 |
| 补全 | 输入被遮住的文本，输出缺失片段 |

FLAN 阶段的核心是把任务改写成统一指令格式。设指令输入为 $u$，参考答案为 $r$，指令数据集为 $D_{\text{inst}}$，训练目标是：

$$
L_{\text{FLAN}}=\mathbb{E}_{(u,r)\sim D_{\text{inst}}}\big[-\log p_\theta(r\mid u)\big]
$$

也就是让模型看到任务描述后生成标准答案。

指令微调放在第二阶段，是因为它依赖第一阶段学到的语言能力。没有足够预训练，模型连文本规律、事实关联、句法结构都不稳定，直接教它“按指令做任务”效果有限。UL2 先提供底层语言能力，FLAN 再把这种能力对齐到任务接口。

FLAN-UL2 还需要注意 mode token。mode token 是原 UL2 中用于提示模型当前使用哪种预训练模式的特殊标记。FLAN-UL2 的实际 checkpoint 经过继续训练和 FLAN 微调后，使用时通常不再要求用户手动提供这些 mode token。工程上应按指令模型使用它，而不是继续假设必须显式传入 R/X/S 模式标记。

---

## 代码实现

代码重点不是“调用一次模型”，而是把原始文本、FAQ、工单、政策文档改写成统一的训练和推理格式。

下面是一个可运行的最小例子，展示三件事：格式化指令、构造训练样本、用一个玩具生成器模拟推理。真实模型可替换为 Hugging Face 的 seq2seq 模型接口。

```python
def format_instruction(task, context, question):
    return f"任务：{task}\n上下文：{context}\n问题：{question}\n回答："


def build_training_example(task, context, question, answer):
    return {
        "input": format_instruction(task, context, question),
        "target": answer,
    }


class ToySeq2SeqModel:
    def generate(self, prompt):
        if "退款多久到账" in prompt and "3 到 5 个工作日" in prompt:
            return "通常在 3 到 5 个工作日内到账。"
        return "无法根据给定上下文确定。"


def generate(model, prompt, max_new_tokens=128):
    # max_new_tokens 在真实模型中用于限制生成长度，这里保留接口形状。
    return model.generate(prompt)[:max_new_tokens]


context = "公司退款政策：退款审核通过后，通常在 3 到 5 个工作日内原路返回。"
example = build_training_example(
    task="根据公司退款政策回答用户问题",
    context=context,
    question="退款多久到账？",
    answer="通常在 3 到 5 个工作日内到账。",
)

model = ToySeq2SeqModel()
prediction = generate(model, example["input"])

assert "任务：" in example["input"]
assert "上下文：" in example["input"]
assert "问题：" in example["input"]
assert prediction == example["target"]
```

真实工程中的数据构造可以按下面方式组织：

```python
def convert_faq_to_instruction_rows(faq_rows):
    rows = []
    for item in faq_rows:
        rows.append({
            "input": format_instruction(
                task="根据知识库回答用户问题",
                context=item["policy_text"],
                question=item["question"],
            ),
            "target": item["answer"],
        })
    return rows


def train_seq2seq(model, tokenizer, rows):
    for row in rows:
        inputs = tokenizer(row["input"], truncation=True, return_tensors="pt")
        labels = tokenizer(row["target"], truncation=True, return_tensors="pt")
        # 真实训练中这里会计算 loss，并执行 backward / optimizer.step。
        batch = {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}
        yield batch
```

训练流程图：

```text
原始文本 / FAQ / 工单 / 政策
  -> 清洗与切分
  -> UL2 预训练或复用已有 UL2 checkpoint
  -> 指令数据改写
  -> FLAN 微调
  -> 推理服务
  -> 指令 + 上下文 + 问题 -> 答案
```

真实接入 Hugging Face 时，推理函数通常长这样：

```python
def generate_with_hf(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

关键是保持训练和推理格式一致。训练时如果输入是 `任务 + 上下文 + 问题 + 回答：`，上线推理也应该保持同样结构。格式漂移会让模型无法稳定判断哪些内容是上下文，哪些内容是需要回答的问题。

---

## 工程权衡与常见坑

FLAN-UL2 的第一个常见错误，是把它当成 decoder-only 模型来用。decoder-only 模型通常更像“接着 prompt 往下写”。FLAN-UL2 更适合明确的 seq2seq 任务输入：给出任务、上下文、问题和期望输出格式。

错误做法：

```text
这里有一大段公司制度，你帮我看看退款问题怎么答……
```

更稳的做法：

```text
任务：根据公司退款政策回答问题
上下文：……
问题：退款多久到账？
输出格式：一句话回答，不要编造上下文之外的信息
回答：
```

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 把模型当 decoder-only 用 | 输出像自由续写，格式不稳定 | 使用 `指令 + 上下文 + 问题 + 输出约束` |
| 忽略 2048 receptive field | 长文被截断，答案缺依据 | 显式截断、分块、摘要或接入检索 |
| 继续沿用原 UL2 mode token 假设 | prompt 多出无用控制符，行为不可预期 | 确认 checkpoint 用法，按 FLAN 指令模型使用 |
| 微调任务覆盖太窄 | 只会某类模板，泛化变差 | 覆盖问答、摘要、抽取、分类、改写等任务 |
| 低估推理成本 | 延迟和显存超预算 | 量化、批处理、并行、缓存或换小模型 |
| 训练与推理格式不一致 | 线上效果明显低于验证集 | 固定 instruction schema 并做格式校验 |

receptive field 可以理解为模型一次能有效接收和处理的上下文窗口。FLAN-UL2 20B 公开模型常见上下文上限是 2048 token 级别，不能把几十页文档直接塞进去并期待模型完整理解。

部署建议清单：

```text
1. 截断策略：优先保留标题、问题相关段落、最近上下文。
2. 分块策略：长文按段落或语义块切分，每块单独召回。
3. 量化策略：用 int8 / int4 降低显存，但要验证质量损失。
4. RAG 接入策略：先检索相关文档片段，再交给模型生成答案。
5. 输出校验：对 JSON、标签、引用来源等格式做后处理验证。
```

真实工程例子：企业工单助手通常不应该把所有历史工单直接塞进 prompt。更合理的流程是先用检索系统找出相关政策、FAQ 和相似工单，再把少量高相关片段拼成上下文交给 FLAN-UL2。这样既控制输入长度，也降低模型编造答案的概率。

---

## 替代方案与适用边界

FLAN-UL2 适合“读输入后生成答案”的任务，但不是所有生成任务都优先选它。模型选择应从任务形态、延迟、成本、上下文长度和输出约束出发。

| 方案 | 适合任务 | 优点 | 局限 |
|---|---|---|---|
| FLAN-UL2 | 文档问答、摘要、抽取式生成、企业知识助手 | seq2seq 结构清晰，指令遵循强，适合统一任务接口 | 20B 级模型成本高，上下文长度有限，不适合极低延迟 |
| decoder-only LLM | 开放式续写、聊天、代码补全、流式生成 | 生态成熟，续写自然，长上下文版本多 | 输入输出边界需要 prompt 约束，抽取类任务可能更依赖提示设计 |
| RAG + 轻量生成模型 | 企业 FAQ、客服助手、内部知识库 | 成本较低，可更新知识库，答案来源可追踪 | 检索质量决定上限，复杂推理能力受小模型限制 |
| 专用分类 / 抽取模型 | 固定标签分类、实体抽取、规则明确任务 | 快、便宜、稳定 | 不适合开放式自然语言回答 |

新手版本可以直接按任务判断：

| 任务 | 是否优先考虑 FLAN-UL2 |
|---|---|
| 企业知识问答 | 适合 |
| 工单助手 | 适合 |
| 研究摘要 | 适合 |
| 文档问答 | 适合 |
| 超低延迟聊天 | 不一定 |
| 大规模开放式续写 | 不一定 |
| 前端实时补全 | 不一定 |
| 只输出固定类别 | 通常没必要 |

如果任务主要是开放式续写、代码补全、对话流式生成，decoder-only 模型通常更直接。如果任务核心是长文理解、抽取、按自然语言生成答案，FLAN-UL2 这类 encoder-decoder 指令模型通常更稳。

工程推断要保守：FLAN-UL2 的论文和模型卡能支撑“统一预训练 + 指令微调带来强泛化能力”这个结论，但不能自动推出“任何企业数据微调后都会稳定超过所有聊天模型”。真实上线仍然要用自己的数据集评测，包括准确率、幻觉率、延迟、成本和格式稳定性。

---

## 参考资料

| 来源 | 用途 | 能支撑哪一章 |
|---|---|---|
| UL2 论文 | 解释 Mixture-of-Denoisers、R/X/S 模式、统一 seq2seq 预训练 | 核心机制与推导 |
| FLAN 论文 | 解释指令微调为什么提升 zero-shot 与任务泛化 | 核心结论、问题定义 |
| Google Research UL2 博客 | 补充 UL2 20B 开源模型背景与工程定位 | 问题定义、工程权衡 |
| Google Research FLAN 博客 | 补充较低计算量下提升语言模型能力的研究背景 | 核心结论、替代方案 |
| Hugging Face 模型卡 | 补充 FLAN-UL2 checkpoint、使用方式和限制 | 代码实现、常见坑 |

1. [UL2: Unifying Language Learning Paradigms](https://research.google/pubs/ul2-unifying-language-learning-paradigms/)
2. [Finetuned Language Models Are Zero-Shot Learners](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/)
3. [Better Language Models Without Massive Compute](https://research.google/blog/better-language-models-without-massive-compute/)
4. [google/flan-ul2 Hugging Face Model Card](https://huggingface.co/google/flan-ul2)
5. [UL2 20B: An Open Source Unified Language Learner](https://research.google/blog/ul2-20b-an-open-source-unified-language-learner/)
