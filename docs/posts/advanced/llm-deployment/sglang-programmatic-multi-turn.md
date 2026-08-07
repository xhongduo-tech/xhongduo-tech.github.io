---
title: SGLang 程序式多轮对话原语
date: 2026-08-07
---

# SGLang 程序式多轮对话原语

<div class="epigraph">
<p>程序是思维的表达形式；把对话写进程序，对话就获得了逻辑。</p>
<footer>—— 奥列格·伊格拉西克，SGLang 核心作者（Oleg Igraciuk，风格化转述）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ SGLang 官方文档（Frontend Language） ｜ 2026-08-07</p>
</div>

## 为什么从程序式多轮对话开始

普通的 LLM API 把多轮对话表达成一个「消息列表」：`system`、`user`、`assistant` 依次排列，模型只能按固定顺序读一遍。但现实中的智能体任务远不止「一问一答」：需要先让模型生成工具调用、再注入工具返回、再让模型继续；需要把用户输入与历史拼接成模板；需要一次运行中并行采样多个分支再选优。<span class="marginnote">本专题《推理基础》讲过 prefill 与 decode 的两阶段计算特征，<strong>prompt 每长一点，prefill 就贵一点</strong>——程序式原语的价值正在于把「拼字符串」这件最容易被写错的事，交给前端语言去管理。</span>

**SGLang 的前端语言（frontend language）**把「对话」提升为一等公民：用 `sgl.gen`、`sgl.user`、`sgl.assistant`、`sgl.system` 等原语，把一次会话写成一段可读的 Python 程序，模型的输入、输出、分支、循环全都在程序里显式表达。它不只是语法糖——它直接对接 RadixAttention 的缓存共享、结构化生成与并行采样。本篇拆解这些原语的语义、背后的协议，以及它们如何改善部署质量。

## 1 从消息列表到程序

先看一个最朴素的多轮对话在传统 API 里的样子：每次请求都要把「系统提示词 + 历史所有轮次」完整拼成一段文本发给模型。代码里体现为一次次手工 `f-string` 拼接，极易出 bug：模板字符（`<|im_start|>` 之类）漏写、历史被重复包含、轮次顺序错乱。

SGLang 用装饰器 `@sgl.function` 把一次会话声明为一个函数：

```python
@sgl.function
def chat(s, user_input):
    s += sgl.system("你是一个乐于助人的助手")
    s += sgl.user(user_input)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))
```

调用 `chat.run(user_input="你好")` 时，前端语言按程序执行顺序把各片段串成完整的模型输入，并把 `sgl.gen` 的结果回填到 `s.answer`。<span class="marginnote">这里的 `+=` 不是字符串拼接，而是向<strong>结构化对话树</strong>追加节点。每个节点携带角色、内容、与生成相关的配置，最终一次性编译成模型可消费的 token 序列。</span>

**辨析｜易错点：`sgl.gen` 是「占位 + 回填」，不是「运行时的 I/O」。** 新手容易把 `sgl.gen` 理解成「此刻就去调用模型」。实际上整个函数先被解析成一张计算图（AST 级别的遍历），真正向模型发起请求发生在 `run()`/`run_batch()` 阶段。这意味着**同一段程序可以批量并行跑**——`run_batch` 把多份输入折叠进一个批次，复用 prefill 的共享前缀。

## 2 分支、循环与工具调用

多轮对话真正的复杂性来自控制流。SGLang 前端语言支持 Python 的 `if`、`for` 直接嵌在 `@sgl.function` 里，让对话结构跟着数据走：

```python
@sgl.function
def tool_agent(s, query):
    s += sgl.system("你可以调用工具。需要工具时，输出 JSON。")
    s += sgl.user(query)
    s += sgl.assistant(sgl.gen("first", stop=["<tool>", "</tool>"]))
    if "<tool>" in s["first"]:
        tool_call = parse_tool(s["first"])
        result = call_tool(tool_call)          # 外部工具执行
        s += sgl.user(f"工具返回：{result}")
        s += sgl.assistant(sgl.gen("final"))
    return s
```

关键点在于 `s["first"]`：它读取的是**已经生成的 token**，而不是 Python 变量。前端语言据此判断是否走分支——于是「模型决定要不要调用工具」就变成了「程序里的一个 `if`」。这类模式是 Agent 系统的骨架，SGLang 把它们做成原语，避免每个团队各自发明一套「工具调用协议」。<span class="marginnote">RadixAttention 的缓存会记录分支后追加的内容，同一用户下次再走同一条分支时直接命中缓存——<strong>程序式表达让「哪条路径被走过」可被索引</strong>，这是纯消息列表做不到的。</span>

## 3 流式、并行采样与结构化生成

除了表达对话结构，前端语言还直接暴露推理引擎的特性：

- **流式**：`sgl.gen` 支持 `stream=True`，逐 token 拿到生成结果，对交互式应用（打字机效果）至关重要。
- **并行采样**：`sgl.gen("answer", n=4)` 一次生成 4 个候选；`temperature` 等采样参数按 `n` 展开，可在 `run_batch` 时对同一 prompt 做多次采样，供后续打分选优。
- **结构化生成**：`sgl.gen(..., regex=...)` 把解码约束成一个正则表达式；`sgl.gen(..., json_schema=...)` 约束成 JSON Schema。两者都走本专题《结构化生成与有限状态机约束解码》讲的 FSM 解码路径，只是在前端以参数形式暴露。<span class="marginnote">这印证了本专题的一个主线：<strong>前端原语（易用性）与内核约束（正确性、吞吐）是同一套机制的上下两层</strong>。</span>

一个稍完整的例子是「多轮总结」：

```python
@sgl.function
def summarizer(s, docs, rounds=2):
    s += sgl.system("请分轮总结给定文档。")
    for i in range(rounds):
        s += sgl.user(f"第 {i+1} 轮：请总结。\n{docs if i == 0 else s['summary']}")
        s += sgl.assistant(sgl.gen("summary", temperature=0.3))
    return s
```

`for` 循环里，第二轮把第一轮的生成结果 `s["summary"]` 拼回输入——**对话历史被程序显式管理**，这正是「程序式多轮对话」这个名字的含义。

## 4 公式解析：程序展开的时间复杂度

程序式表达真正的部署收益，用「prompt 重算量」来衡量。设一轮对话的历史总长为 $H$，新增用户输入与生成长度分别为 $u$ 与 $g$。传统消息列表每轮完整重发，第 $k$ 轮的 prompt 长度是：

$$L_k = H_k = \sum_{i=1}^{k} (u_i + g_i)$$

三步拆解这条式子：

- **第一步，理解求和**：$u_i$、$g_i$ 是第 $i$ 轮的新输入与新增生成，历史逐轮累加。第 $k$ 轮的 prefill 要处理全部累计长度 $H_k$。
- **第二步，算总账**：$K$ 轮之后，总 prefill 量为 $\sum_{k=1}^{K} H_k = \sum_{i=1}^{K} (K-i+1)(u_i + g_i)$——**同一段历史被重复计算了 $K-i+1$ 次**，越早的内容被算得越多次。这是消息列表协议的根本浪费。
- **第三步，对比缓存命中**：RadixAttention 复用已缓存的前缀后，第 $k$ 轮实际只需 prefill 新增的 $u_k + g_k$，总 prefill 量降到 $\sum_{i=1}^{K}(u_i+g_i)$。**程序式表达本身不产生缓存，但它让「哪段是新增、哪段是复用」对引擎透明**，缓存复用率因此大幅提升。

在冷启动（无缓存）场景下两种方式等价；一旦有缓存，程序式 + 前缀复用把多轮成本从「历史反复重算」降到「每轮只算增量」。

## 5 小结

- **前端语言把对话写成程序**：`sgl.user`、`sgl.assistant`、`sgl.gen` 等原语表达角色与生成，`if`/`for` 表达分支与循环，工具调用成为程序控制流的一部分。
- **`sgl.gen` 是占位回填而非即时 I/O**：整个函数先编译成计算图，`run`/`run_batch` 才真正触发推理，天然支持批量并行。
- **流式、并行采样、正则/JSON Schema 约束**都以参数形式暴露，与后端 FSM 解码、缓存共享是同一套机制的两层。
- **部署收益是预填充量的降低**：消息列表协议让历史被重复计算 $K-i+1$ 次，程序式 + 前缀缓存把每轮成本压到「只算增量」。

在下一节，我们进入**第五篇 TensorRT-LLM**，从引擎的图优化讲起——看 TensorRT 如何把神经网络图重写成一堆高效的底层算子。
