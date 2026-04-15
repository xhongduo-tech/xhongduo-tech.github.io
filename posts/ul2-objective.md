## 核心结论

UL2 更准确的全称是 **UL2 (Unifying Language Learning Paradigms)**。如果把它写成 “Ulv2”，读者容易误以为是在说某个单向语言模型变体。本文讨论的是 UL2 训练范式，不是“单向语言模型”这个泛称。

UL2 的一句话定义是：**把多种预训练目标统一到同一个模型中的多目标预训练框架**。它的重点不是发明一种全新的 Transformer 结构，而是让同一个 checkpoint 在训练时轮流学习不同任务形式，再通过 `mode token` 指定当前模式。

对初学者，最直观的理解是：同一篇文本可以被改造成不同题型。

- 有时只给左边，让模型继续往后写，这叫因果语言模型，白话就是“看前文续写后文”。
- 有时给前缀和一部分结构，让模型补完后续，这叫 prefix LM，白话就是“先看条件，再按顺序生成答案”。
- 有时把中间几段挖空，让模型恢复缺失内容，这叫 span corruption，白话就是“做填空题，而且一次填多段”。

差别不在“模型换了”，而在“训练任务切换了”。

| 目标类型 | 输入能看到什么 | 预测什么 | 强项 |
|---|---|---|---|
| span corruption | 左右两侧保留文本，缺失片段被占位符替代 | 被挖掉的若干 span | 理解、补全、鲁棒表示 |
| prefix LM | 完整前缀可见，后缀按自回归生成 | 前缀后的目标序列 | 条件生成、问答、摘要 |
| causal LM | 只看左侧历史 | 下一个 token | 长文本续写、流式生成 |

从统一视角看，UL2 的目标函数可以写成对模式 $m$ 的期望：
$$
\mathcal{L} = \mathbb{E}_{m \sim \pi}\,\mathbb{E}_{z \sim q_m(z|x)}\big[-\log p_\theta(y_m \mid c_m(x,z), t_m)\big]
$$
这里的意思很直接：先采样一个训练模式，再按这个模式构造输入和目标，最后让同一个模型去学。

---

## 问题定义与边界

传统预训练经常把“理解”和“生成”拆开处理。只做遮盖恢复的模型，擅长利用双向上下文，但长续写往往不是强项。只做左到右生成的模型，擅长续写，但面对“补中间一段”或“严格条件补全”时往往不自然。

玩具例子可以说明这个边界。假设句子是：

`数据库事务需要满足原子性、一致性、隔离性、持久性`

如果一个模型只做左到右训练，它很擅长看到“数据库事务需要满足”后继续写后面的词，但如果把“隔离性”挖掉，要求它基于左右两边恢复中间缺失，它并不是按这个题型训练出来的。反过来，只做遮盖恢复的模型面对“请继续写一段 500 字解释”时，也不一定稳定。

所以 UL2 解决的问题是：**能不能用统一预训练覆盖多种语言任务形式**。它不承诺“每个下游任务都比专用模型强”，它解决的是覆盖面和迁移性，而不是取消任务差异。

| 维度 | UL2 主要解决什么 | UL2 不直接解决什么 |
|---|---|---|
| 目标统一 | 把多种预训练目标放进同一框架 | 不保证所有任务都最优 |
| 模型能力 | 兼顾理解、补全、生成 | 不等于不需要指令微调 |
| 架构层面 | 可复用现有 Encoder-Decoder 思路 | 不是靠新 backbone 获胜 |
| 工程收益 | 减少多 checkpoint 并存 | 不减少训练配置复杂度 |

把它放到历史脉络里看更清楚。T5 强调 span corruption，UniLM 强调通过不同 attention mask 统一单向、双向和 seq2seq，UL2 则进一步把多种去噪和生成目标放进一套更明确的多模式训练流程。本文因此讨论的是“训练目标统一”，不是“某个单向结构的新名字”。

---

## 核心机制与推导

UL2 的关键抽象是两件事：**模式采样** 和 **模式条件化**。

模式采样，白话就是“这条样本这次用哪种题型训练”。模式条件化，白话就是“模型必须知道你现在让它做什么题”。

仍然用最小玩具例子，原始序列是：

`A B C D E F`

三种模式可以统一表示成：

1. `span corruption`
   - 输入：`A B <extra_id_0> E F`
   - 目标：`<extra_id_0> C D`
   - 含义：把中间一段挖空，再恢复。

2. `prefix LM`
   - 输入前缀：`A B C`
   - 目标：`D E F`
   - 含义：前缀作为条件，后缀按顺序生成。

3. `causal LM`
   - 训练过程：看到 `A` 预测 `B`，看到 `A B` 预测 `C`，依次继续。
   - 含义：永远只根据左侧历史预测下一个 token。

这三种方式的本质差异，不是参数矩阵换了，而是“输入哪些位置可见、哪些 token 要被预测”。

| 模式名称 | 输入可见范围 | 预测目标 | 典型 token 形式 | 适合场景 |
|---|---|---|---|---|
| span corruption | 被保留文本的双侧上下文 | 被删除的 span | `<extra_id_0>` 这类 sentinel token | 填空、去噪、鲁棒理解 |
| prefix LM | 完整前缀 + 受控后缀生成结构 | 后缀序列 | `[S2S]` 或类似 mode token | 摘要、翻译、问答 |
| causal LM | 当前位置左侧历史 | 下一个 token | `[NLG]` 或类似 mode token | 对话、续写、开放生成 |

如果画成流程，UL2 的训练链路可以压缩成：

原始文本  
→ 采样模式 $m$  
→ 按 $q_m(z|x)$ 构造上下文和目标  
→ 在输入前注入 `mode token` $t_m$  
→ 用同一个模型计算损失

这里 $q_m(z|x)$ 可以理解为“这个模式下如何改造样本”的规则分布，比如腐蚀率、span 长度、prefix 切分位置。于是统一公式
$$
\mathcal{L} = \mathbb{E}_{m \sim \pi}\,\mathbb{E}_{z \sim q_m(z|x)}\big[-\log p_\theta(y_m \mid c_m(x,z), t_m)\big]
$$
就很自然了：  
$\pi$ 决定不同模式抽到的概率；  
$c_m(x,z)$ 是构造后的输入上下文；  
$y_m$ 是该模式下要预测的目标；  
$t_m$ 告诉模型“现在处于什么模式”。

真实工程例子更容易看出它的价值。假设你在做一个客服模型平台：

- 工单分类，需要读完整上下文后判断标签，偏理解任务。
- 回复补全，需要给定用户问题和已有草稿，补出后半段，偏 prefix LM。
- 多轮对话，需要按 token 流式输出，偏 causal LM。

如果为每种任务各训一个模型，工程上会出现三套 checkpoint、三套部署策略、三套迁移方案。UL2 的思路是让同一个底座模型在预训练阶段就接触这些题型，再在推理阶段通过不同 `mode token` 触发不同行为。

---

## 代码实现

实现 UL2，核心通常不在“重写 Transformer”，而在“样本构造器、特殊 token 约定、loss 位置对齐”。

工程上常见做法是约定三类模式 token，例如：

- `[NLU]`：偏理解/去噪
- `[S2S]`：偏 seq2seq 条件生成
- `[NLG]`：偏左到右生成

下面给出一个可运行的最小 Python 示例，只演示“同一条文本按不同模式构造训练样本”的逻辑。它不是完整训练代码，但足够说明 UL2 的工程骨架。

```python
from dataclasses import dataclass

MODE_TOKENS = {
    "nlu": "[NLU]",
    "s2s": "[S2S]",
    "nlg": "[NLG]",
}

@dataclass
class Example:
    mode: str
    input_text: str
    target_text: str

def build_example(tokens, mode):
    assert len(tokens) >= 6

    if mode == "nlu":
        # span corruption: 挖掉中间一个 span
        input_tokens = tokens[:2] + ["<extra_id_0>"] + tokens[4:]
        target_tokens = ["<extra_id_0>"] + tokens[2:4]
        return Example(
            mode=mode,
            input_text=MODE_TOKENS[mode] + " " + " ".join(input_tokens),
            target_text=" ".join(target_tokens),
        )

    if mode == "s2s":
        # prefix LM: 前缀作为条件，预测后缀
        prefix = tokens[:3]
        suffix = tokens[3:]
        return Example(
            mode=mode,
            input_text=MODE_TOKENS[mode] + " " + " ".join(prefix),
            target_text=" ".join(suffix),
        )

    if mode == "nlg":
        # causal LM: 这里只展示训练样本形式
        prefix = tokens[:-1]
        next_token = tokens[-1]
        return Example(
            mode=mode,
            input_text=MODE_TOKENS[mode] + " " + " ".join(prefix),
            target_text=next_token,
        )

    raise ValueError(f"unknown mode: {mode}")

tokens = "A B C D E F".split()

ex_nlu = build_example(tokens, "nlu")
ex_s2s = build_example(tokens, "s2s")
ex_nlg = build_example(tokens, "nlg")

assert ex_nlu.input_text == "[NLU] A B <extra_id_0> E F"
assert ex_nlu.target_text == "<extra_id_0> C D"

assert ex_s2s.input_text == "[S2S] A B C"
assert ex_s2s.target_text == "D E F"

assert ex_nlg.input_text == "[NLG] A B C D E"
assert ex_nlg.target_text == "F"

print(ex_nlu)
print(ex_s2s)
print(ex_nlg)
```

把它扩成训练框架时，常见主循环就是：

```python
mode = sample_mode()                 # 采样模式
x_in, y = build_example(text, mode)  # 构造输入和目标
tokens = tokenize(x_in)
labels = tokenize(y)
loss = model(tokens=tokens, labels=labels)
```

数据管线可以概括为：

原始文本  
→ `sample_mode()`  
→ `build_example()`  
→ 在输入侧拼接 mode token  
→ tokenizer 编码  
→ 前向传播与 loss 计算

这里有一个实现细节经常被忽视：**训练和推理必须共享同一套 token 约定**。如果训练时模型学到 `[S2S]` 表示“条件补全”，推理时你改成别的名字，或者干脆不加，模型看到的条件分布就变了。

---

## 工程权衡与常见坑

UL2 的收益来自覆盖多种目标，但代价是数据构造和训练配置更复杂。很多失败案例不是模型思想有问题，而是实现细节不一致。

| 问题 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 训练时有 mode token，推理时忘了加 | 输出风格漂移，任务不受控 | 条件信号缺失 | 把 mode token 视为必填输入 |
| tokenizer 没注册特殊 token | token 被拆碎，模式学习无效 | 词表与数据约定不一致 | 先检查特殊 token 是否进入词表 |
| 各模式采样比例失衡 | 某类任务明显偏弱 | 训练分布偏斜 | 按目标任务重要性调 $\pi$ |
| span corruption 腐蚀率过高 | 生成质量下降，恢复不稳定 | 输入信息过少 | 不把高腐蚀率当万能配置 |
| 论文符号与工程 token 名不一致 | 复现偏差 | 实现约定混乱 | 统一一份模式映射表 |

最小检查清单可以直接列成四项：

- 训练和推理的 mode token 是否完全一致。
- 各模式采样比例是否和目标场景匹配。
- tokenizer 是否包含所有特殊 token。
- 数据构造是否真的符合论文中的任务定义，而不是“看起来差不多”。

一个典型坑是：训练阶段一直给 `[NLG]`，推理时只输入普通 prompt，希望模型自然进入生成模式。这在小规模实验里有时还能“看起来能跑”，但一旦任务复杂，行为就会明显漂移。原因不是模型突然失效，而是你删掉了它训练时依赖的条件变量。

---

## 替代方案与适用边界

UL2 不是唯一选择。是否采用它，取决于你要不要一个统一 checkpoint 覆盖多种任务形式。

| 方案 | 训练目标 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| T5 风格 | span corruption 为主 | 去噪与理解直接 | 长续写不是最自然 | 填空、摘要、结构化生成 |
| 纯 causal LM | 左到右 next-token | 生成链路简单，生态成熟 | 中间补全和双向理解较弱 | 对话、续写、代码生成 |
| UniLM 风格 | 用 mask 统一多任务 | 统一思路早、结构清晰 | 工程实现依赖 mask 设计 | 统一 NLU/NLG 实验 |
| UL2 | 多模式统一预训练 | 覆盖理解、补全、生成 | 配置复杂、复现门槛更高 | 通用助手、混合任务平台 |

可以用一个简单选择表判断：

- 只做对话续写或长文本生成，纯 causal LM 往往更直接。
- 只做填空、理解、摘要，T5 风格通常更省心。
- 既要理解，又要补全，还要生成，并且希望尽量共用一个底座模型，UL2 更合适。

真实工程里，客服助手、企业知识问答平台、混合任务推理服务，更容易从 UL2 受益。因为这些系统经常同时包含分类、改写、摘要、回复补全和开放生成。相反，如果你的产品目标非常单一，例如只做代码续写，那么把系统做成纯 causal LM 往往更简单，调参空间也更可控。

---

## 参考资料

下文中的统一符号写法为作者归纳，不是论文逐字公式。

原始论文：

- UL2 论文页: https://research.google/pubs/ul2-unifying-language-learning-paradigms/
- OpenReview 论文页: https://openreview.net/forum?id=6ruVLB727MC

模型卡：

- Hugging Face `google/ul2` 模型卡: https://huggingface.co/google/ul2

相关背景：

- T5 论文页: https://www.jmlr.org/papers/v21/20-074.html
- UniLM 论文页: https://www.microsoft.com/en-us/research/publication/unified-language-model-pre-training-for-natural-language-understanding-and-generation/
