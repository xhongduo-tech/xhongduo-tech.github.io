---
title: 数据集格式与工具链：Alpaca、ShareGPT 格式的互转与校验
date: 2026-08-07
---

# 数据集格式与工具链：Alpaca、ShareGPT 格式的互转与校验

<div class="epigraph">
<p>同一种原料，不同的菜系，端出的菜天差地别。</p>
<footer>—— 引意自烹饪常识</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第二章 ｜ 2026-08-07</p>
</div>

## 为什么从数据格式开始

数据工程这一篇，我们从「怎么造数据」讲到了「怎么筛、怎么配、怎么拼」。但还差最后一步落地：**这些数据以什么格式存在磁盘上？** 同一个数据集，可以是 Alpaca 格式，可以是 ShareGPT 格式，还可以是 OpenAI 的 messages 格式——它们记录的是同样的对话，却有着不同的结构。

格式问题看起来琐碎，实则每天都在制造事故：数据集从 HuggingFace 下载后字段名对不上、单轮数据被当成多轮解析、JSON 里一个转义符错误导致整批训练崩溃。本节把两大标准格式讲清，再把「互转」与「校验」这两道工序变成可执行的方法。<span class="marginnote">本主题偏工具与流程，核心不是数学公式而是「格式映射表」——因此本节以对比表替代公式解析，重点讲清字段如何对应、转换时什么会丢、校验时该查什么。</span>

## 1 两大标准格式：Alpaca 与 ShareGPT

### Alpaca 格式：单轮指令的标准容器

Alpaca 格式源自斯坦福 Alpaca 数据集，结构极简——一个 JSON 数组，每个元素是一条单轮样本：

```json
[
  {
    "instruction": "解释什么是质数",
    "input": "",
    "output": "质数是只能被 1 和自身整除的大于 1 的自然数。"
  },
  {
    "instruction": "把下面这句话翻译成英文",
    "input": "今天天气很好。",
    "output": "The weather is nice today."
  }
]
```

三个字段：**instruction**（指令）、**input**（可选输入，没有则为空字符串）、**output**（期望回答）。它只描述「一轮问答」，天然适合单轮指令数据。缺点也明显：**表达不了多轮对话**——没有「历史轮次」的位置。

### ShareGPT 格式：多轮对话的标准容器

ShareGPT 格式源自用户分享的 ChatGPT 对话，结构是「一个对话 id + 一系列消息」：

```json
{
  "id": "chatcmpl-7xY9abc",
  "conversations": [
    { "from": "human", "value": "解释什么是质数" },
    { "from": "gpt",   "value": "质数是只能被 1 和自身整除的大于 1 的自然数。" },
    { "from": "human", "value": "那 9 是质数吗？" },
    { "from": "gpt",   "value": "不是，9 可以被 3 整除。" }
  ]
}
```

关键字段：**from**（说话人，`human` 或 `gpt`）、**value**（消息内容），多条消息按顺序组成一轮完整对话。它天然支持多轮，还允许加 **system** 消息（部分实现里以 `system` 表达）。缺点是没有独立的「input」字段——输入被并入消息文本。

### 两种格式的对照

| 维度 | Alpaca 格式 | ShareGPT 格式 |
| --- | --- | --- |
| 基本单元 | 一条「指令-回答」样本 | 一段多轮对话 |
| 关键字段 | instruction / input / output | id / conversations[].from / value |
| 单轮表达 | 原生支持 | 也支持（一段一问一答） |
| 多轮表达 | 不支持 | 原生支持 |
| 系统提示词 | 无专门字段 | 部分实现支持 system 消息 |
| 典型来源 | 指令生成（Alpaca、Self-Instruct） | 真实对话（ShareGPT、Vicuna） |

## 2 格式互转：从 Alpaca 到 ShareGPT 的映射

转换的本质是**字段重排**。把一条 Alpaca 样本转成 ShareGPT：

$$
\text{Alpaca} = (i, x, o) \;\longmapsto\; \text{ShareGPT} = \big[\text{human}: i \oplus x,\;\; \text{gpt}: o\big]
$$

其中 $i$ 是指令、$x$ 是输入、$o$ 是输出，$\oplus$ 表示字符串拼接——**当 input 非空时，常见做法是把「指令 + 换行 + 输入」拼成一条 human 消息**。逐项拆解这条映射：

- 指令 $i$ → human 消息的开头；
- 输入 $x$ → 若为空串，直接忽略；若非空，接在指令后面（换行分隔）——因为 ShareGPT 没有 input 字段；
- 输出 $o$ → 紧跟的 gpt 消息；
- id → 可选，通常用哈希或序号生成，保证可追溯。

反向转换（ShareGPT → Alpaca）则只取**第一轮** human 与 gpt 消息，把 human 消息拆回「指令 + 输入」——但这一步**会丢信息**：多轮对话的后半段全部丢弃，且「哪个词算指令、哪个词算输入」无法精确还原。所以反向转换一般只用于「把多轮数据集粗筛成单轮」，细节损失要接受。<span class="marginnote">转换方向的不对称值得记住：<strong>Alpaca → ShareGPT 是无损的</strong>（单轮本就是多轮的特例），<strong>ShareGPT → Alpaca 是有损的</strong>（多轮被截成一轮）。工程上应尽量朝「信息更多」的方向转。</span>

## 3 工具链：从原始 JSON 到可训练样本

格式互转只是中间步骤，完整工具链是「原始 JSON → 标准格式 → 对话模板序列化 → token 化」四段：

1. **格式统一**：用脚本把数据集统一成 Alpaca 或 ShareGPT 之一。开源工具如 **LLaMA-Factory 的数据脚本**、**HF `datasets` 库**、以及社区常用的 `sharegpt2alpaca` / `alpaca2sharegpt` 脚本都能完成字段重排。
2. **模板序列化**：把标准格式的对话交给 **chat template**（上一节《对话模板》的主角），转成模型能读的 token 序列。这一步用的是 HuggingFace 的 Jinja 模板机制——每个模型自带一段 **`chat_template`** 字符串，定义如何把 **messages** 变成文本。
3. **token 化与截断**：对序列化后的文本做分词、加 attention mask、按 **`max_length`** 截断（或 packing，上一节已讲）。
4. **缓存落盘**：把 token 化结果存成 **Arrow**/**npy** 等二进制格式，避免每次训练重复分词。大型数据集这一步能省下数小时。

以 HuggingFace 生态为例，一段标准化的读取与转换长这样：

```python
from datasets import load_dataset

# 读取 Alpaca 格式，转成 ShareGPT 多轮格式（无损方向）
ds = load_dataset("json", data_files="alpaca.json", split="train")

def to_sharegpt(ex):
    return {
        "id": hashlib.md5(ex["instruction"].encode()).hexdigest(),
        "conversations": [
            {"from": "human",
             "value": ex["instruction"] + ("\n" + ex["input"] if ex["input"] else "")},
            {"from": "gpt", "value": ex["output"]},
        ],
    }

sharegpt_ds = ds.map(to_sharegpt, remove_columns=ds.column_names)
sharegpt_ds.to_json("sharegpt.jsonl", force_ascii=False)
```

**一个关键纪律**：**尽量在「标准格式」层做互转，而不是在「模板文本」层做**。有人直接把已套好模板的字符串硬切硬拼，等于放弃结构信息，任何后续修改（换模板、换模型）都要重来。

## 4 校验：数据进训练前必查的五件事

格式转完，数据还不能直接进训练——先过五道校验。每道都对应一类真实的翻车事故：

1. **Schema 校验**：字段是否齐全、类型是否正确（instruction 必须是字符串、conversations 必须是数组）。用 JSON Schema 或 **Pydantic 的 `BaseModel`** 检查。**崩溃事故最常见来源**——漏字段、字段类型混入 None。
2. **JSON 合法性**：转义符、引号、逗号是否规范。一条非法 JSON 会让**训练进程**在训练中途崩溃，且错误信息常常不指明是哪条。批量解析 + 定位报错行号，是这里的基本功。
3. **角色一致性**：ShareGPT 里 **`from`** 字段只能是 **`human` / `gpt` / `system`** 等合法值；出现拼写错误（如 `humnan` 混进 `human`）会导致模板序列化时角色丢失。**校验：枚举 from 的取值集合**。
4. **内容完整性**：回答为空、回答只是复述指令、指令或输出超长（超出上下文窗口）等。对应前几节的质量筛选，格式层只需做最基础的「非空、长度上限」检查。
5. **去重与泄漏复查**：格式互转后再次跑一遍 n-gram 去重与基准去污染（第一篇《数据质量筛选》），因为互转过程可能引入新的重复（例如同一条样本被多条转换脚本各产一份）。<span class="marginnote">五道校验里，<strong>Schema 与 JSON 合法性是「保训练不崩」的底线，角色一致性与内容完整性是「保训练不歪」的质量关，去重复查是「保评测可信」的保险</strong>——分层理解，才能知道每道校验该投入多少。</span>

## 5 常见错误与工程实践

数据格式环节的错误，大多能归到下面几条，每条都有对应的「药」：

- **错误一：多轮数据当单轮用**。把 ShareGPT 多轮数据直接按 Alpaca 三字段拆，只保留第一轮——后半段对话全部白费。药：先看数据来源，多轮数据就用多轮格式，别强行降维。
- **错误二：模板层硬切硬拼**。在已套模板的文本上做字符串操作来「改格式」。药：回到标准格式层操作，模板只作最后的序列化一步。
- **错误三：字段名不一致**。不同来源的数据集字段名五花八门（`instruction`/`prompt`/`query`、`output`/`response`/`answer`）。药：统一用 **HF `datasets` 的 `rename_columns` / `map`** 归一到约定字段，再进流程。
- **错误四：校验只在入库时做一次**。数据在后续清洗、互转、筛选后，字段可能又坏了。药：**每一道工序的输出都过一次轻量 schema 校验**，把「校验」做成管线的一部分，而不是一次性动作。
- **错误五：忽略 id 的可追溯性**。ShareGPT 样本不带 id，出问题时无法定位「哪条数据训坏了模型」。药：转换时强制生成 id，保留来源元数据。

一条实用的工程建议：**把「标准格式 + 校验脚本」固化成仓库里的公共模块**。任何新数据集进来，先转成标准格式、跑一遍校验、再进训练——这套固定流程能把数据问题挡在训练之外，而不是让训练崩溃来暴露问题。

## 6 小结

- **Alpaca 格式**（instruction / input / output）管单轮，**ShareGPT 格式**（id / conversations[].from / value）管多轮，各有适用场景。
- 格式互转是**字段重排**：Alpaca → ShareGPT 无损，ShareGPT → Alpaca 有损（多轮截成单轮）。
- 完整工具链：**原始 JSON → 标准格式 → 对话模板序列化 → token 化**；在标准格式层互转，模板只做最后一步。
- 五道校验：**Schema、JSON 合法性、角色一致性、内容完整性、去重复查**——分层对应「不崩、不歪、可信」。
- 常见错误：多轮当单轮、模板层硬切、字段名不一致、校验只做一次、缺 id 可追溯性。

到这一节为止，「数据工程」这一篇全部收尾。在下一篇，我们离开数据，进入训练的「体力活」：**全参数微调**——显存账本、混合精度、梯度检查点与 ZeRO/FSDP，把一条 7B 模型真正训练起来。
