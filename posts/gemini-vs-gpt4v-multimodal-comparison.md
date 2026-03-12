## 核心结论

如果把比较对象限定为 **Gemini 1.0 Ultra/Pro** 和 **GPT-4V** 这两代公开资料最完整的模型，那么结论可以先写清楚：

1. **Gemini 在“重感知”的多模态任务里整体更强**。这里的“重感知”是指答案高度依赖图片、表格、扫描文档、视频内容本身，而不只是依赖文字提示。Google 在 Gemini 1.0 技术报告里给出的数字是：MMMU 59.4 对 GPT-4V 56.8，DocVQA 90.9 对 88.4，TextVQA 82.3 对 78.0。
2. **GPT-4V 的核心优势不是“看图更强”，而是“继承 GPT-4 的语言推理与工程生态”**。白话说，它更像一个很强的文本模型接上了视觉入口，所以在复杂指令、代码、函数调用、工具编排上更容易进入已有工程体系。
3. **两者的架构取向不同**。Gemini 官方一直强调自己是 *natively multimodal*，意思是模型从设计阶段就把多种模态当成一等公民处理。GPT-4V 的官方资料没有完整公开内部结构，但从公开能力边界看，它是“GPT-4 加上视觉能力”的产品形态，这决定了它在视频、音频、长文档原生处理上的边界更窄。
4. **真实选型不能只看分数**。如果任务是“合同、表格、截图、图表、视频一起进来，再统一抽字段”，Gemini 往往更顺手；如果任务是“强文本推理、代码生成、函数调用、稳定集成”，GPT-4V 往往是更低风险的方案。

| 维度 | Gemini Ultra/Pro | GPT-4V | 更适合的任务 |
|---|---|---|---|
| 视觉理解 | 更强 | 强 | 图像问答、图表、OCR |
| 文档分析 | 更强 | 强 | PDF、表单、扫描件 |
| 视频/音频 | 原生路线更完整 | 官方公开能力以图像为主 | 长视频、多媒体分析 |
| 文本推理 | 强 | 很强 | 复杂指令、代码、系统设计 |
| 工程生态 | 在增强中 | 更成熟 | API 编排、工具调用 |

这里还有一个容易忽略的点：这篇对比讨论的是 **2023 年底到 2024 年公开资料对应的模型代际**，不是 2026 年最新通用模型横评。否则比较对象会变成 Gemini 2.x/3.x、GPT-4o 或更后的模型，结论会变。

---

## 问题定义与边界

“多模态能力”不是一句空话，至少要拆成三类问题：

| 类别 | 输入是什么 | 真正考验什么 |
|---|---|---|
| 视觉理解 | 单图、多图、图表、截图 | 看懂内容，不靠额外文字提示 |
| 文档分析 | PDF、扫描件、表格、票据 | OCR、版面理解、字段抽取 |
| 多模态推理 | 图像 + 文本，或视频 + 文本 | 不同信息源能否联合推导 |

术语先解释一下：

- **OCR**：把图片里的字读出来，白话说就是“先看见文字，再把文字变成机器能处理的文本”。
- **DocVQA**：文档视觉问答，白话说就是“给你一页文档，问你里面某个字段或关系是什么”。
- **MMMU**：多学科多模态理解基准，白话说就是“把大学级别题目和图片、图表、公式混在一起，看模型能不能一起做”。

这篇文章的边界也要先定住：

1. 比较对象是 **Gemini 1.0 Ultra/Pro 与 GPT-4V**。
2. 讨论重点是 **能力形态与工程后果**，不是 API 价格。
3. 不讨论 2025 年以后更新一代模型的重新洗牌。
4. 不把“聊天体验”混同于“多模态能力”。

一个玩具例子就能说明边界差异：

你给模型一张成绩单截图，问“数学是否高于物理”。  
这不是纯 OCR，因为它不只是识别“92”和“88”，还要把字段名和数值配对，再做比较。  
如果模型只会读字，不会做版面理解，它就会把相邻列、相邻行看混。

一个真实工程例子更明显：

运营团队要处理一份产品发布材料，输入里同时有：
- 30 页 PDF 手册
- 几张信息图
- 一段 20 分钟发布会视频
- 一段补充说明文字

这时真正的问题不是“谁更聪明”，而是“谁能少拆流水线”。  
Gemini 的路线更接近“尽量把这些内容放进同一次上下文里”；GPT-4V 官方公开能力则主要围绕图像，因此视频和音频通常要先拆帧、转写、再重组。

---

## 核心机制与推导

先看一个最基础的公式。Transformer 的注意力机制是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

白话解释：

- $Q$ 可以理解成“我现在想找什么”
- $K$ 可以理解成“我这里有哪些可匹配线索”
- $V$ 可以理解成“真正要取出来的信息”

多模态模型的关键，不是有没有这个公式，而是 **不同模态的数据在多早的阶段进入同一套注意力计算**。

### 1. Gemini 的路线：原生统一

Google 对 Gemini 的公开表述是 **natively multimodal**。  
白话说，不是先做完文本模型、再补一个视觉插件，而是一开始就把文本、图像、音频、视频放进统一建模框架里。

这会带来两个直接结果：

1. **跨模态关系更早建立**。  
   例如图表标题、坐标轴、柱状高度、问题文本，可以更早在同一轮推理里互相约束。
2. **长上下文更容易做成统一能力**。  
   Gemini 1.5 Pro 官方报告进一步展示了这一点：它能在超长上下文里同时处理文本、视频、音频，并在检索任务里保持很高召回。

### 2. GPT-4V 的路线：强语言模型加视觉入口

这里必须严谨一点：**OpenAI 没有完整公开 GPT-4V 的内部架构细节**。  
所以“视觉编码器 + 语言模型 + 融合层”更适合作为一种工程理解，而不是可以当作官方内部实现细节逐层复述的事实。

但从官方系统卡可以确认两件事：

1. GPT-4V 是 **GPT-4 with vision**，即 GPT-4 加入图像分析能力。
2. 官方公开能力边界主要是 **图像输入**，不是原生长视频和音频输入。

所以，工程上可以把它理解为：

- 文本推理能力主要继承 GPT-4
- 图像先经过视觉处理，再进入与语言相关的联合推理
- 这使它在“图片 + 指令”的场景很强，但在“视频/音频/长文档一起原生处理”的场景里，需要更多外围系统补齐

### 3. 为什么分数差一点点，工程后果会很大

设文档抽取准确率为 $a$，每天处理文档数为 $N$，则预期人工复核量近似为：

$$
R = N(1-a)
$$

如果 Gemini 的 DocVQA 准确率是 $90.9\%$，GPT-4V 是 $88.4\%$，那么复核量差值是：

$$
\Delta R = N(0.909 - 0.884) = 0.025N
$$

也就是每 1000 份文档少 25 份人工复核。  
这不是“学术分数领先 2.5 个点”而已，而是直接变成运维工作量。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，把“准确率差异如何变成人工复核成本”算清楚。

```python
def expected_manual_reviews(total_docs: int, accuracy: float) -> int:
    assert total_docs >= 0
    assert 0.0 <= accuracy <= 1.0
    return round(total_docs * (1 - accuracy))

def review_saving(total_docs: int, acc_a: float, acc_b: float) -> int:
    assert total_docs >= 0
    assert 0.0 <= acc_a <= 1.0
    assert 0.0 <= acc_b <= 1.0
    return expected_manual_reviews(total_docs, acc_b) - expected_manual_reviews(total_docs, acc_a)

gemini_docvqa = 0.909
gpt4v_docvqa = 0.884

saving = review_saving(1000, gemini_docvqa, gpt4v_docvqa)

assert expected_manual_reviews(1000, gemini_docvqa) == 91
assert expected_manual_reviews(1000, gpt4v_docvqa) == 116
assert saving == 25

print("每 1000 份文档，Gemini 相比 GPT-4V 约少", saving, "份人工复核")
```

这个例子很小，但它说明了一个工程事实：  
**基准分数不是展示用海报，它会落到队列长度、人工抽检量和 SLA 上。**

如果把两种系统的最小 pipeline 画成伪代码，差异更直观：

```python
def gemini_style_pipeline(texts, images, pdf_pages, video_segments):
    # 原生多模态路线：尽量把不同输入放进统一上下文
    inputs = []
    inputs.extend(texts)
    inputs.extend(images)
    inputs.extend(pdf_pages)
    inputs.extend(video_segments)
    return "统一推理后输出结构化结果"

def gpt4v_style_pipeline(text_prompt, images, pdf_file=None, video_file=None):
    # GPT-4V 路线：图像直接进模型，其他模态常要先转成图像/文本
    prepared_inputs = list(images)

    if pdf_file is not None:
        prepared_inputs.extend(["pdf_page_image_1", "pdf_page_image_2"])

    if video_file is not None:
        prepared_inputs.extend(["frame_001", "frame_045", "frame_090"])
        text_prompt += "\n请结合 frame 编号回答。"

    return "图像+文本联合推理后输出结果"
```

真实工程例子可以是“合同条款抽取”：

- **Gemini 风格**：上传整份合同 PDF，连同补充文本说明一起问“抽取付款周期、违约金、自动续约条款”。
- **GPT-4V 风格**：先 OCR 或逐页转图片，再把每页图片和问题一起送入模型，最后把多页答案做聚合。

前者更像“模型吞进去处理”，后者更像“系统先拆，再让模型逐步看”。

---

## 工程权衡与常见坑

### 1. 不要把“原生多模态”理解成“所有任务都更强”

这是最常见的误区。  
Gemini 在文档、图表、视频理解上更占优势，不等于它在所有任务上都绝对优于 GPT-4V。  
如果任务几乎全是文本推理、代码解释、函数调用，GPT-4V 往往更符合已有工程堆栈。

### 2. GPT-4V 的坑通常在预处理

如果你拿 GPT-4V 处理发票流或会议视频，常见问题不是模型答错，而是前面流程先丢了信息：

- OCR 把表格列对错了
- 抽帧没抽到关键帧
- 页码、帧号、文本转写不同步
- 同一字段在不同页出现，后处理合并失败

所以很多时候，问题不在“视觉不够强”，而在 **外层流水线过长**。

### 3. Gemini 的坑通常在上下文与成本控制

Gemini 的优势是可以把更多模态塞进统一上下文，但代价是：

- 输入管理更复杂
- 长上下文会推高延迟
- 多媒体内容太长时，费用和吞吐压力明显上升
- 如果什么都往里塞，反而会稀释关键信号

白话说，原生多模态不是“无脑全塞”，而是“少拆流水线，但要做输入裁剪”。

### 4. 基准分数不能直接映射业务指标

DocVQA 高，不代表你自己的发票系统就一定同比提升 2.5%。  
原因很简单：真实业务还有版式分布、图像质量、字段定义、容错规则、人工抽检阈值这些变量。

所以更稳的做法是：

1. 用公开 benchmark 判断能力上限。
2. 用自己的小样本集做 A/B 测试。
3. 关心端到端指标：准确率、复核率、延迟、失败重试率。

---

## 替代方案与适用边界

选型可以直接按任务类型来判断：

| 任务类型 | 更推荐 | 原因 |
|---|---|---|
| 纯文本推理 | GPT-4V | 语言能力和工程生态更稳 |
| 代码生成/审查 | GPT-4V | 继承 GPT-4 路线，指令跟随更成熟 |
| 图表、文档、扫描件 | Gemini | 感知侧更强，少做预处理 |
| 长视频、多媒体归档分析 | Gemini | 原生多模态路线更合适 |
| 现有 OpenAI 工作流扩展 | GPT-4V | 更容易接到既有系统 |
| 新建多模态数据管线 | Gemini | 更容易减少外围拼装步骤 |

一个简单判断标准：

- 如果你的系统主要在做“**看**”，例如看图、看表、看票据、看视频，优先考虑 Gemini。
- 如果你的系统主要在做“**写和调度**”，例如写代码、调工具、走函数调用、做复杂文本推理，优先考虑 GPT-4V。

还有一类替代方案值得提：  
**不用单模型硬吃所有模态，而是做分层流水线。**

例如：

1. 先用 OCR/ASR 专用模型抽文本
2. 再用表格解析器做结构化
3. 最后用强文本模型做归纳和决策

这种方案的优点是可控、可观测、便于排错；缺点是链路长、维护成本高。  
所以它更适合高合规、高审计需求的企业场景，不一定适合追求极简开发的团队。

---

## 参考资料

- Google DeepMind / Gemini Team, *Gemini: A Family of Highly Capable Multimodal Models*, 2023. https://arxiv.org/abs/2312.11805
- Google, *Our next-generation model: Gemini 1.5*, 2024. https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/
- Google Gemini Team, *Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context*, 2024. https://goo.gle/GeminiV1-5
- OpenAI, *GPT-4V(ision) System Card*, September 25, 2023. https://openai.com/index/gpt-4v-system-card/
- OpenAI, *GPT-4V(ision) technical work and authors*, 2023. https://openai.com/contributions/gpt-4v/
- MMMU Benchmark leaderboard / dataset card. https://huggingface.co/datasets/MMMU/MMMU
