## 核心结论

代码 tokenizer 的差异，不在“会不会切词”，而在“把哪些字符当成值得保留的结构”。这里的结构，白话说，就是模型在输入里最先看到的边界，比如空格、缩进、文件名、括号、换行。

Codex 所代表的 OpenAI GPT BPE 路线，更强调细粒度可还原性：空格、标点、局部格式往往更容易被单独保留。CodeLlama 的 SentencePiece 路线，把空白以前缀 metaspace `▁` 绑定到后续片段上，目标是让“词和它前面的空格”一起进入模型。StarCoder 的 Hugging Face BPE 路线，则进一步强化“代码环境”这个上下文，把 `<filename>`、`<reponame>` 一类元信息直接做成专用 token，同时对高频空白模式更积极压缩。

最直接的后果是 token 数不同。玩具例子里，同样一行 `"    foo()"`，三者可能出现下面这种差异：

| 模型路线 | 空格策略 | metadata token | 典型词表规模 |
|---|---|---:|---:|
| Codex / GPT BPE | 更细粒度保留空格与符号 | 弱，更多依赖普通文本上下文 | 50K 级到 100K 级 |
| CodeLlama / SentencePiece | 用 `▁` 表示前导空白，空白与词绑定 | 有 infill 相关专用 token | 32K 级 |
| StarCoder / HF BPE | 对高频连续空白和代码模式更积极合并 | 强，含 `<filename>`、`<reponame>` 等 | 49,152 |

如果只看压缩率，StarCoder 和 CodeLlama 通常更省 token；如果看格式追踪、diff 对齐、精确还原，Codex 这类更细粒度的 GPT BPE 更稳。两者没有谁绝对更先进，本质是工程目标不同。

---

## 问题定义与边界

本文只讨论 tokenizer，也就是“把原始代码切成 token 序列的规则”。它不讨论模型权重、推理策略、代码语义理解、补全质量，也不把 tokenizer 的优劣直接等同于模型能力。

比较边界只有三类：

| 比较对象 | 关注点 | 不讨论 |
|---|---|---|
| Codex 所代表的 GPT BPE | 空格是否单独保留、符号拆分粒度 | 模型是否更会写代码 |
| CodeLlama 的 SentencePiece + byte fallback | metaspace、未知字符回退、空白配置 | 具体参数量差异 |
| StarCoder 的 HF BPE + 元 token | metadata token、多文件上下文、空白压缩 | 训练数据总量优劣 |

一个最小流程可以写成：

1. 输入代码文本
2. tokenizer 按规则切分
3. 得到 token stream
4. 模型按 token 统计长度、建立注意力、预测下一个 token

真正影响代码任务的点，主要集中在第 2 步。比如输入 `if (x>0) {`：

- 在 GPT BPE 路线里，空格和括号更容易作为独立边界被保留。
- 在 SentencePiece 路线里，`if` 前面的空白常通过 `▁if` 这样的形式编码。
- 在 StarCoder 路线里，除了普通代码字符，还可能把文件级上下文通过特殊 token 一起送入模型。

所以本文比较的是“同一段代码被如何切开”，而不是“谁的生成结果一定更好”。

---

## 核心机制与推导

判断 tokenizer 是否适合代码任务，至少要看两个量：压缩率与公平性。

压缩率可以用 fertility 衡量。这里的 fertility，白话说，就是“一个词平均被拆成多少个 token”。值越高，说明拆得越碎。

$$
\mathrm{fertility} = \frac{\text{总 token 数}}{\text{总词数}}
$$

如果一个 tokenizer 把 `foo_bar`、`    `、`->`、`</div>` 这些代码里常见模式都拆得很碎，fertility 就会上升。对代码模型来说，这意味着同样上下文窗口能装进的真实代码更少。

第二个量是多语言公平性，可以用 Parity 表示。白话说，它衡量“同样意思的一段内容，不同语言要不要付出更多 token 成本”。

$$
\mathrm{Parity}_{\ell} = \mathbb{E}\left[\frac{|t(S_{\ell})|}{|t(S_{en})|}\right]
$$

其中 $S_{\ell}$ 是语言 $\ell$ 的输入，$t(\cdot)$ 是 tokenizer 输出的 token 序列长度。若 Parity 接近 1，说明该语言与英语的 token 负担接近；若明显大于 1，说明该语言被拆得更碎。

玩具例子：`let 急件 = true;`

- 若词表小、对中文代码标识符支持弱，`急件` 可能被拆成多个 token，fertility 升高。
- 若词表扩展、常见 Unicode 标识符或字节回退机制更好，`急件` 的 token 数会下降，Parity 更接近 1。

这也是词表大小会反复被提起的原因。更大的词表，通常能覆盖更多常见代码片段、语言标识符、空白模式和 metadata 模板，从而降低 fertility、改善 Parity。但代价也明确：

$$
\text{Embedding 参数量} \approx V \times d
$$

$$
\text{输出 Softmax 计算量} \propto V
$$

其中 $V$ 是词表大小，$d$ 是隐层维度。词表从 32K 增到 100K，不只是“多记一点 token”，还会直接增加 embedding 表规模和输出层计算成本。

因此，32K、49K、100K 不是单纯的“大就是好”。它们分别代表不同取舍：

| 词表规模 | 优势 | 代价 |
|---|---|---|
| 32K 级 | 参数更省，训练稳定，通用性够用 | 小语种、罕见标识符更易被拆碎 |
| 49K 级 | 对代码常见模式覆盖更均衡 | 参数和推理成本上升 |
| 100K 级 | 更容易覆盖多语言、多符号、多格式模式 | embedding/softmax 更重，长尾 token 训练更难 |

---

## 代码实现

先看一个可运行的玩具实现。它不是三家官方 tokenizer 的源码，而是一个“机制模拟器”，用来展示三种路线在空白处理上的区别。

```python
import re

def encode_with_codex(text: str):
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            tokens.append(" ")
            i += 1
        elif ch in "(){}[],:;":
            tokens.append(ch)
            i += 1
        else:
            j = i
            while j < len(text) and text[j] not in " (){}[],:;":
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens

def encode_with_codellama(text: str):
    pieces = []
    for part in re.findall(r"\s+|[^\s]+", text):
        if part.isspace():
            continue
        prefix_space = "▁" if text[text.find(part) - 1:text.find(part)] == " " or text.startswith(part) is False else ""
        pieces.append(prefix_space + part)
    if text.startswith(" "):
        pieces[0] = "▁" + pieces[0].lstrip("▁")
    return pieces

def encode_with_starcoder(text: str):
    tokens = []
    for part in re.findall(r"\s+|[^\s]+", text):
        if part.isspace():
            tokens.append(f"<ws:{len(part)}>")
        else:
            tokens.extend(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[(){}[\],:;.]|.", part))
    return tokens

sample = "    foo()"
codex = encode_with_codex(sample)
codellama = encode_with_codellama(sample)
starcoder = encode_with_starcoder(sample)

assert codex[:4] == [" ", " ", " ", " "]
assert starcoder[0] == "<ws:4>"
assert any(piece.startswith("▁") for piece in codellama)

print(codex)
print(codellama)
print(starcoder)
```

这个例子表达的不是精确词表，而是三种典型策略：

- `encode_with_codex`：把空格当独立边界。
- `encode_with_codellama`：用 `▁` 表示“前面有空白”。
- `encode_with_starcoder`：把连续空白压成一个空白块 token。

真实实现里，Codex 所代表的 GPT BPE 路线依赖 `tiktoken`。它不是简单“逐字符切分”，而是先按正则模式分段，再在字节级做 BPE merge。结果上，空格和标点更容易被精确保留，因此格式边界清晰。这里要注意，OpenAI 的 GPT BPE 家族并不只有一个词表版本：早期代码系常见 `p50k_base`，后续通用编码又扩展到 `cl100k_base` 这类 100K 级方案。本文把它们视为同一路线，而不是同一个固定编码。

CodeLlama 使用 SentencePiece。SentencePiece 的核心思想是先把空白转成可见符号 `▁`，再对整个序列做子词切分。这样 ` foo` 可能变成 `▁foo`，模型天然知道这个片段前面有空格。它再叠加 byte fallback，也就是“遇到词表外字符时退回到字节级表示”，避免完全丢失未知字符。

StarCoder 则明显更面向真实代码仓库。真实工程例子是多文件补全：模型收到的输入不只是函数体，还可能有文件名、仓库名、issue 片段、notebook 单元。StarCoder 直接把 `<filename>`、`<reponame>` 一类模式做成专用 token，相当于告诉模型：“这不是普通文本，这是代码工作区的结构化元数据。”

一个简化的 metadata 流程可以写成：

```text
repo context
-> 插入 <reponame>
-> 插入 仓库名
-> 插入 <filename>
-> 插入 当前文件路径
-> 插入 代码正文
-> tokenizer 编码
```

这类设计的意义不是“少几个 token”那么简单，而是把代码环境的语义边界显式标出来。模型不需要从普通字符串里猜“这里像不像文件名”，因为 token 本身已经给了它提示。

---

## 工程权衡与常见坑

工程里最常见的误解，是把“token 更少”直接当成“效果更好”。这不成立。代码场景至少有三组相互冲突的目标：压缩率、可还原性、多语言公平性。

第一组权衡是空白保真度与 token 成本。

| 策略 | 潜在风险 | 规避方法 |
|---|---|---|
| 细粒度空格保留 | token 数上升，上下文更快用完 | 仅在格式敏感任务启用 |
| 连续空白压缩 | Python 缩进层级可能不够显式 | 保留常见缩进模式专用 piece |
| 空白与词绑定 | diff 对齐不直观 | 在评估中单独统计格式错误率 |

如果你的任务是代码格式检查、补丁生成、最小 diff 修复，那么 Codex 这类细粒度路线有实际优势。因为它更容易精确表示“少了 1 个空格”或“缩进错了 1 层”。相反，如果任务是长上下文仓库问答，空白压缩带来的 token 节省可能更值钱。

第二组权衡是 SentencePiece 的配置坑。SentencePiece 默认把空白当正常符号处理，但如果训练配置不当，多空格、换行、缩进可能被过度归并。对于 Python，这很危险，因为缩进不是装饰，而是语法的一部分。实践中通常要关注：

- `allow_whitespace_only_pieces=true`
- `split_by_whitespace=false`
- 必要时关闭额外空白清洗
- 对代码场景考虑 `byte_fallback=true`

否则看起来“只是 tokenizer 变紧凑了”，实际上可能把 `if:` 下方四个空格与八个空格的区别抹平。

第三组权衡是词表大小与公平性。低词表在英语代码仓库里未必明显吃亏，但一旦进入多语言代码环境，例如变量名含中文、日文、阿拉伯文，或者注释与文档字符串不是英语，Parity 就会恶化。模型会把非英语内容拆成更长的 token 序列，等价于让这些语言在同一窗口里“租金更贵”。

真实工程例子：一个 IDE 插件要在 monorepo 里做跨文件补全。

- 若用 StarCoder 风格 tokenizer，`<reponame>`、`<filename>` 先把结构定住，再编码正文，模型更容易区分“当前文件”和“引用文件”。
- 若用 CodeLlama 风格 tokenizer，但没有正确保留 whitespace-only pieces，Python 文件中的缩进样式可能在训练或推理前处理时受损。
- 若用 Codex 风格 tokenizer，跨文件 metadata 需要通过普通字符串拼进去，虽然可行，但显式结构提示较弱。

---

## 替代方案与适用边界

如果目标是统一代码风格、长上下文、多文件联动，StarCoder 路线通常更合适。它的强项不是“单个 token 更聪明”，而是 tokenizer 已经把仓库级结构显式编码了。边界是词表更大、embedding 成本更高，而且这类专用 token 对非仓库文本的收益有限。

如果目标是严格格式审查、代码修补、精确 diff，对空格和局部符号极其敏感，Codex 所代表的 GPT BPE 路线更稳。边界是 token 消耗较高，长文件时上下文压力更大。尤其在 Python、YAML、Makefile 这种格式即语义的场景，细粒度 token 往往不是浪费，而是必要信息。

如果目标是多语种代码提示、代码加自然语言混合输入、未知字符较多，CodeLlama 的 SentencePiece + byte fallback 是比较折中的方案。它既保留了较好的压缩率，也给长尾字符留了回退通道。边界是训练配置必须谨慎，尤其不能把空白处理当成默认安全。

可以把适用场景简化成下面这张表：

| tokenizer 路线 | 最适场景 | 边界条件 |
|---|---|---|
| Codex / GPT BPE | 格式校验、diff 修复、精细补丁生成 | token 更贵，长上下文不占优 |
| CodeLlama / SentencePiece | 多语种 prompt、代码+文本混合、长尾字符多 | whitespace 配置不当会伤到缩进信息 |
| StarCoder / HF BPE + metadata | IDE、多文件仓库补全、项目级上下文建模 | 词表更大，专用 token 对纯文本收益有限 |

还有一个经常被忽略的替代方向：不要只比较“谁的 tokenizer 更好”，而要比较“任务是否需要单一 tokenizer”。例如：

- 格式校验链路用细粒度 GPT BPE。
- 仓库检索与摘要链路用 StarCoder 风格 tokenizer。
- 多语种自然语言注释生成链路用 SentencePiece 路线。

也就是说，tokenizer 不一定是全系统统一组件，它完全可以按子任务拆分。

---

## 参考资料

- OpenAI `tiktoken` 项目：GPT BPE 路线与编码实现，说明 OpenAI tokenizer 的基本工作方式。https://github.com/openai/tiktoken
- OpenAI Help Center 关于 token 计数：说明 `tiktoken` 是 OpenAI 模型的官方计数工具。https://help.openai.com/en/articles/8984337-how-can-i-tell-how-many-tokens-a-string-will-have-before-i-try-to-embed-it
- Hugging Face Code Llama 文档：CodeLlamaTokenizer、infilling token、模型使用方式。https://huggingface.co/docs/transformers/en/model_doc/code_llama
- Meta CodeLlama 仓库：官方推理代码与 tokenizer 相关实现入口。https://github.com/meta-llama/codellama
- Google SentencePiece README：`▁` metaspace、语言无关子词切分的基本机制。https://github.com/google/sentencepiece
- SentencePiece 训练选项文档：`allow_whitespace_only_pieces`、`split_by_whitespace`、`byte_fallback` 等配置说明。https://chromium.googlesource.com/chromium/src/+/main/third_party/sentencepiece/src/doc/options.md
- SentencePiece Issue #684：多空格与换行处理的实际问题讨论。https://github.com/google/sentencepiece/issues/684
- StarCoder tokenizer 配置：49,152 词表规模与 `<filename>`、`<reponame>` 等 special tokens。https://huggingface.co/bigcode/starcoder-co-format/blob/main/tokenizer_config.json
- 关于 multilingual tokenizer 的 fertility / Parity 分析综述，可进一步看 Emergent Mind 的汇总入口。https://www.emergentmind.com/topics/multilingual-sentencepiece-bpe-tokenizer
