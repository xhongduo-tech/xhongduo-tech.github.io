## 核心结论

对话模板（Chat Template）是把一组 `messages=[{role, content}, ...]` 转成模型训练时见过的唯一文本格式的规则。这里的“模板”可以先理解成“消息排版规范”：它决定 `system`、`user`、`assistant` 三种角色怎样插入分隔符、结束符和起始符。

核心结论有三条。

第一，不同模型的聊天格式并不通用。ChatML 常见写法是：

```text
<|im_start|>user
Hello<|im_end|>
```

Llama 3 常见写法是：

```text
<|start_header_id|>user<|end_header_id|>

Hello<|eot_id|>
```

Vicuna、Alpaca、Mistral Instruct 又会使用 `USER:` / `ASSISTANT:`、`[INST]...[/INST]` 或其他约定。模型不是“理解了聊天协议”本身，而是在训练中记住了某一种输入分布。

第二，`tokenizer.apply_chat_template` 的职责，是把结构化消息列表映射成该模型需要的字符串或 token 序列。可以把它写成一个简单关系式：

$$
\text{tokens} = \text{tokenizer.apply\_chat\_template}(\text{messages}, \text{tokenize=False}, \text{add\_generation\_prompt=mode})
$$

其中 `mode` 在训练时通常取 `False`，在推理时通常取 `True`。原因很直接：训练样本里往往已经包含了完整的 assistant 回复，不需要再额外追加“请从 assistant 开始生成”的提示；推理时最后一条常常是用户消息，需要补一个 assistant 起始标记，告诉模型接下来该由谁说话。

第三，训练模板与推理模板必须一致，否则性能会明显劣化。这里的“一致”不是“都像聊天格式”就行，而是要求角色标记、换行、BOS/EOS、结束 token 的组合方式都一致。模型训练时如果学到的是 `<s>[INST] ... [/INST]`，推理时却收到另一种 `USER:` 风格输入，它面对的是分布漂移，不是简单的语法差异。

下面先看一个最小玩具例子。只有一句用户消息 `"Hello"`，如果模型模板是 ChatML，那么：

```python
messages = [{"role": "user", "content": "Hello"}]
```

执行：

```python
tokenizer.apply_chat_template(messages, tokenize=False)
```

结果会是：

```text
<|im_start|>user
Hello<|im_end|>
```

这说明 role 分隔符不是你手写进去的，而是模板自动插入的。

| 模型族 | system 标记 | user 标记 | assistant 标记 | 轮次结束标记 |
|---|---|---|---|---|
| ChatML | `<|im_start|>system` | `<|im_start|>user` | `<|im_start|>assistant` | `<|im_end|>` |
| Llama 3 | `<|start_header_id|>system<|end_header_id|>` | `<|start_header_id|>user<|end_header_id|>` | `<|start_header_id|>assistant<|end_header_id|>` | `<|eot_id|>` |
| Vicuna 风格 | 常见为纯文本前缀或系统段 | `USER:` | `ASSISTANT:` | 依实现而定 |
| Mistral/Alpaca 一类 | 常见嵌入指令段 | `[INST]...[/INST]` 包裹用户指令 | assistant 直接接在后面 | 常配合 `</s>` |

---

## 问题定义与边界

问题定义很明确：给定一串有顺序的多轮消息，如何把它们稳定地转换成模型训练时使用的输入格式。这里的“稳定”意思是，同样的消息序列必须得到同样的字符串布局，否则训练和推理会发生格式偏移。

先定义几个基本术语。

`role` 是消息角色，也就是“这句话是谁说的”。常见取值是 `system`、`user`、`assistant`。

`special token` 是有特殊用途的 token，可以先理解成“不会被当普通自然语言处理的控制标记”。例如 BOS 表示序列开始，EOS 表示序列结束，PAD 表示补齐占位。

`chat_template` 是分角色拼接文本的模板规则。它通常保存在 `tokenizer.chat_template` 中，本质上是一段 Jinja 模板。

边界要分清两层。

第一层，哪些内容由聊天模板负责。通常包括角色名、角色分隔符、消息间换行、回合结束符，有时也包含 BOS/EOS。

第二层，哪些内容由 tokenizer 的额外 special token 机制负责。比如某些 tokenizer 在 `tokenizer(text, add_special_tokens=True)` 时会自动加 BOS/EOS；但有些 chat template 已经把这些 token 明确写进模板。如果两边都加，就可能重复。

下面这张表很关键。

| 组件 | 通常由谁负责 | 典型内容 | 常见错误 |
|---|---|---|---|
| 角色标记 | `chat_template` | `user`、`assistant` 的包裹格式 | 手动拼一套和模板不同的标记 |
| 回合结束符 | `chat_template` | `<|im_end|>`、`<|eot_id|>` | 忘记结束，导致模型把多轮混成一轮 |
| BOS/EOS | 可能在模板内，也可能由 tokenizer 额外添加 | `<s>`、`</s>` | 模板里已有，还在 tokenization 时再加一次 |
| PAD | tokenizer 配置 | `<pad>` | 用 EOS 乱充当 PAD，训练 mask 出错 |

一个低门槛例子是 Llama 3。你不需要手动再写 `<s>` 或 header token，只需要准备：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "解释一下 TCP 三次握手"}
]
```

然后交给模板即可。若该 tokenizer 已经定义好 Llama 3 的 chat template，那么输出字符串里会自动包含类似：

```text
<|start_header_id|>system<|end_header_id|>
...
<|eot_id|><|start_header_id|>user<|end_header_id|>
...
<|eot_id|>
```

你不应该再额外手工插 `<s>`，除非你确认该模型规范就是要求你自己拼，而且 tokenizer 没有内建模板。

训练与推理时 `add_generation_prompt` 的差异也属于边界问题。训练时一般 `False`，因为样本通常已经包含 assistant 的真实答案；推理时一般 `True`，因为最后一条往往是用户消息，需要模板补一个 assistant 起始段，让模型继续生成 assistant 内容，而不是继续生成 user 段或把结束符续写错位。

---

## 核心机制与推导

`tokenizer.chat_template` 可以理解为一段“遍历 messages 并渲染字符串”的模板程序。输入是结构化消息，输出是最终字符串；如果再设 `tokenize=True`，输出就会直接变成 token id。

抽象表达式可以写成：

$$
\text{output} = \text{template.render}(\text{role}, \text{content for each message})
$$

这不是神秘机制，本质上就是“按顺序遍历消息，给每一条套上模型要求的壳”。

用伪代码表示如下：

```text
function message_to_template(messages, add_generation_prompt):
    output = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        output += render_role_prefix(role)
        output += render_content(content)
        output += render_turn_suffix(role)

    if add_generation_prompt:
        output += render_assistant_prefix_only()

    return output
```

这里的 `render_role_prefix`、`render_turn_suffix` 具体长什么样，完全由模型模板决定。

看一个三轮玩具例子：

```python
messages = [
    {"role": "system", "content": "回答要简洁。"},
    {"role": "user", "content": "什么是哈希表？"},
    {"role": "assistant", "content": "哈希表是通过哈希函数定位数据的结构。"},
    {"role": "user", "content": "时间复杂度通常是多少？"}
]
```

若模板是 ChatML，渲染思路就是：

1. system 段包成 `<|im_start|>system ... <|im_end|>`
2. user 段包成 `<|im_start|>user ... <|im_end|>`
3. assistant 段包成 `<|im_start|>assistant ... <|im_end|>`
4. 最后一轮如果用于推理，且 `add_generation_prompt=True`，再补一个空的 `<|im_start|>assistant`

于是结果类似：

```text
<|im_start|>system
回答要简洁。<|im_end|>
<|im_start|>user
什么是哈希表？<|im_end|>
<|im_start|>assistant
哈希表是通过哈希函数定位数据的结构。<|im_end|>
<|im_start|>user
时间复杂度通常是多少？<|im_end|>
<|im_start|>assistant
```

最后一行没有内容，只有 assistant 起始标记，这就是 generation prompt 的作用：明确告诉模型“从这里开始，轮到 assistant 说话”。

`continue_final_message` 是另一个容易被忽略的参数。它可以先理解成“继续补写最后一条消息，而不是新开一条消息”。例如最后一条已经是 assistant 开头，但内容被截断，你想让模型接着写完整，而不是插入一个新的 assistant 起始标记，这时就需要类似的续写语义。

真实工程里，这个机制常见于两类场景。

第一类是监督微调（SFT）。数据集本来就是多轮对话，模板负责把结构化数据变成训练文本。这里的关键不是“能不能拼出来”，而是“拼出来的格式和底座模型原始 instruct 格式是否完全一致”。

第二类是在线推理服务。服务收到前端对话历史后，先转成 `messages`，再统一走 chat template，最终送入模型。这样前端不需要知道 ChatML、Llama 3、Vicuna 的具体差异，后端只维护一处模板逻辑。

---

## 代码实现

下面先给一个可运行的最小 Python 例子。它不依赖 Hugging Face，也能说明模板机制本身。

```python
def apply_chat_template_chatml(messages, add_generation_prompt=False):
    pieces = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        pieces.append("<|im_start|>assistant\n")
    return "".join(pieces)

# 玩具例子
messages = [{"role": "user", "content": "Hello"}]
rendered = apply_chat_template_chatml(messages, add_generation_prompt=False)
assert rendered == "<|im_start|>user\nHello<|im_end|>\n"

# 推理例子
messages2 = [
    {"role": "system", "content": "回答要简洁。"},
    {"role": "user", "content": "什么是队列？"}
]
prompt = apply_chat_template_chatml(messages2, add_generation_prompt=True)
assert prompt.endswith("<|im_start|>assistant\n")
assert "<|im_start|>system\n回答要简洁。<|im_end|>\n" in prompt
assert "<|im_start|>user\n什么是队列？<|im_end|>\n" in prompt
```

上面这个例子说明两件事。

1. 模板负责插入 role 分隔符。
2. `add_generation_prompt=True` 时，末尾会补 assistant 起始标记。

真实工程里，通常直接使用 tokenizer 内置模板。训练管道的核心写法如下：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

# Dataset.map(format_example) 之后，再统一分词
def tokenize_example(example):
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
        add_special_tokens=False,  # 模板若已含 special token，通常不要再重复加
    )
    return encoded
```

这里先 `tokenize=False`，再统一调用 `tokenizer(...)`，有两个工程好处。

第一，模板渲染与分词解耦，便于调试。你可以先打印纯字符串，确认分隔符是否正确，再看 token 长度。

第二，数据预处理更稳定。很多训练错误其实不是模型问题，而是文本模板不对；先拿到字符串再检查，定位成本更低。

推理管道则不同，关键差异是 `add_generation_prompt=True`：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "你是一个解释数据库概念的助手。"},
    {"role": "user", "content": "解释一下索引下推"}
]

prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer(
    prompt_text,
    return_tensors="pt",
    add_special_tokens=False,
)

# outputs = model.generate(**model_inputs, max_new_tokens=256)
```

真实工程例子是做一个内部问答机器人。前端传来历史消息，后端不直接拼接字符串，而是先标准化成：

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
]
```

然后统一走 `apply_chat_template`。这样模型从 Mistral 换到 Llama 3，只需要切换 tokenizer 和模板，不需要让前端、数据库、调用方一起重写格式逻辑。

---

## 工程权衡与常见坑

最大的问题不是“不会写模板”，而是“以为模板差一点没关系”。实际上，聊天模型对格式高度敏感，因为这些格式本身就是训练分布的一部分。

最常见的坑如下。

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 训练和推理模板不同 | 回复风格异常、忽略 system、接着 user 说 | 输入分布漂移 | 训练和推理共用同一 `chat_template` |
| 重复 BOS/EOS | 开头或结尾出现重复控制 token | 模板里已有，tokenizer 又自动加一次 | 检查 `add_special_tokens` 与模板内容 |
| 手动插 role token | 模型回复紊乱 | 手写格式与官方模板不一致 | 优先使用模型自带模板 |
| 忘记 generation prompt | 模型不进入 assistant 角色 | 末尾缺 assistant 起始标记 | 推理时设 `add_generation_prompt=True` |
| 续写场景误开新消息 | assistant 回复中断或重复前缀 | 本该续写最后一条，却新开一轮 | 使用 `continue_final_message` 语义 |
| PAD/EOS 混用不当 | 训练 loss 异常、mask 错位 | padding 策略与标签遮罩不一致 | 明确区分 `pad_token_id` 和 `eos_token_id` |

一个典型新手错误是双重 BOS。比如训练模板是：

```text
<s>[INST] 解释 TCP [/INST]
```

但推理端又手工写成：

```text
<s><s>[INST] 解释 TCP [/INST]
```

模型看到两个 BOS，并不会“自动忽略一个”。它只知道 token 序列变了。轻则风格变差，重则首轮输出直接异常。解决方法不是“模型自己能适应”，而是把 `<s>` 是否出现这件事固定到模板或 tokenizer 配置中的一处，不要两边都控制。

另一个常见问题是 system prompt 被误读。比如训练时模型看到的是：

```text
<system>你是代码助手</system><user>写一个二分查找</user>
```

推理时却改成：

```text
USER: 你是代码助手
USER: 写一个二分查找
ASSISTANT:
```

这时原本应该是系统层约束的话，被当成普通用户消息。模型不一定“彻底坏掉”，但服从性会下降，因为角色语义已经变了。

`continue_final_message` 在流式续写里尤其重要。假设上一次生成中断，最后一条 assistant 消息只生成了一半。你如果再次调用模板并开启新的 assistant prompt，模型可能会重复开头；而如果按“继续最后一条 assistant 消息”的方式组织输入，它更可能沿着已有回复往下写。这不是所有场景都需要，但在断点续写、工具调用回填、流式恢复里很有用。

---

## 替代方案与适用边界

并不是所有项目都必须依赖 Hugging Face 的 Jinja chat template。只要训练和推理保持一致，手写固定前缀也可以工作。但要清楚边界：越接近官方模板，风险越小；越自定义，兼容成本越高。

下面对比三种常见方案。

| 方案 | 适用场景 | 优点 | 风险 |
|---|---|---|---|
| Hugging Face `chat_template` | 使用官方 instruct/chat 模型 | 与模型原始格式一致，迁移方便 | 需要理解 tokenizer 配置 |
| 手写字符串拼接 | 简单实验、教学、极小项目 | 可控、直观 | 容易漏 token、和官方模板不一致 |
| 纯文本前缀 `USER:` / `ASSISTANT:` | 自训小模型或旧格式数据 | 实现最简单 | 对官方聊天模型常常不是最佳格式 |

例如 Vicuna/Alpaca 一类，很多教程会直接拼：

```text
### Instruction:
解释 B 树

### Response:
```

或者：

```text
[INST] 解释 B 树 [/INST]
```

这种方式不是错，而是它本来就是某类模型的训练格式。如果你整个训练集、验证集、推理服务都严格使用这一套，它就能工作。

但当你使用 Llama 3、Qwen Chat、Mistral Instruct 这类已有明确模板定义的模型时，最好切回模型提供的专用 template。原因不是“官方一定更高级”，而是这些模型在预训练后指令微调阶段就是按该格式学出来的。你越贴近原始格式，越少承担输入分布漂移风险。

可以把适用边界概括为三句话。

第一，如果模型仓库已经提供 `chat_template`，优先用它，不要自己猜。

第二，如果模型没有模板、你又只是做简单实验，手写拼接可以接受，但要固定规范并全链路一致。

第三，如果你在做生产系统，模板应该成为模型适配层的一部分，由后端统一维护，而不是散落在前端、数据脚本和推理代码里各写一份。

---

## 参考资料

- 官方文档
  - Hugging Face Transformers: Chat templates  
    https://huggingface.co/docs/transformers/en/chat_templating
  - Hugging Face Blog: Chat Templates  
    https://huggingface.co/blog/chat-templates

- 实战教程
  - PyTorch torchtune: Chat dataset/template tutorial  
    https://docs.pytorch.org/torchtune/stable/tutorials/chat.html

- 对比分析
  - Scaling Thoughts: 不同模型聊天格式对比（可作为格式速查材料）  
    https://www.scalingthoughts.com/
  - Unsloth: Chat template / dataset formatting 相关指南  
    https://docs.unsloth.ai/
