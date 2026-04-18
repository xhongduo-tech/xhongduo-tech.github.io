## 核心结论

DocVQA 和 InfographicVQA 都是在评测“文档版面理解”，不是单纯 OCR。OCR 是 Optical Character Recognition，意思是把图片里的文字识别成文本；文档理解还要求模型知道文字、表格、图表、标题、脚注、版面位置之间的关系。

DocVQA 的核心任务是：给一张或多张文档图像，再给一个问题，让模型从文档中回答。它更接近“在合同、表单、报告、论文截图里找答案”。多数情况下，答案来自原文中的一个文本片段，也就是 span。span 是连续的一段文本，例如合同里原样出现的“违约金为合同总价的 10%”。

InfographicVQA 的核心任务是：给一张信息图、海报或可视化长图，再给一个问题，让模型结合文字、图标、图表、图例和简单数值关系回答。它更接近“读懂一张图讲了什么”。答案可能是原文里的文字，也可能是读图后得到的短结论或数值。

新手版对照：

| 任务 | 输入 | 主要能力 | 常见答案形式 |
|---|---|---|---|
| DocVQA | 文档图像 | 找文本证据 | 文本 span |
| InfographicVQA | 信息图/海报 | 读字 + 读图 + 读数 | 文本、数值、短结论 |

一个玩具例子：一张扫描表单里写着“姓名：张三，金额：1200 元”，问题是“金额是多少？”这更像 DocVQA。另一张信息图里有 2022 年和 2023 年两根柱子，问题是“2023 年比 2022 年增长了吗？”这更像 InfographicVQA，因为模型要读柱状图，而不是只找一行文字。

核心判断标准是：如果答案主要来自文档原文，优先按 DocVQA 思路处理；如果答案需要综合图表、图例、坐标、图标和文本，优先按 InfographicVQA 思路处理。

---

## 问题定义与边界

DocVQA 的输入是文档页图像，可以是扫描件、表单、报告、合同、论文截图、发票、说明书页面等。它的目标不是把整页都转成文本，而是回答指定问题。例如问题是“第 17 页的违约金是多少？”答案通常来自合同原文某个文本片段。模型需要先找到相关页，再找到相关区域，最后抽取正确文本。

InfographicVQA 的输入是信息图、海报、长图、可视化摘要或营销图。它的答案可能依赖标题、图例、坐标轴、图形大小、颜色含义和正文说明。例如问题是“2023 年增长率是多少？”模型可能需要先读标题确认指标，再读柱状图数值和图例，最后做简单计算或比较。

新手版：DocVQA 更像“找原文”，InfographicVQA 更像“读懂一张图讲了什么”。

| 维度 | DocVQA | InfographicVQA |
|---|---|---|
| 主要输入 | 文档页 | 信息图/海报 |
| 证据类型 | 文本为主 | 文本 + 图形 + 数值 |
| 典型答案 | 抽取式 | 抽取式 + 非抽取式 |
| 难点 | 小字、版面、跨栏 | 图表、图例、数值推理 |

这里的“抽取式”是指答案能从原文中直接截取出来。例如“付款期限为 30 天”出现在合同中，模型回答“30 天”。“非抽取式”是指答案不一定原样出现，需要推理得到。例如图中 2022 年是 100，2023 年是 120，问题问增长率，答案是 $20\%$，但图中可能没有直接写“20%”。

边界也要明确。DocVQA 不是普通文本问答，因为输入不是干净文本，而是图片。InfographicVQA 也不是普通图像分类，因为它要求细粒度读文字、读数值、读版面。两者共同要求模型处理视觉输入，但关注点不同：DocVQA 更重文本定位与抽取，InfographicVQA 更重视觉语义与数值关系。

---

## 核心机制与推导

统一记号如下：文档为 `D`，第 $i$ 页为 $P_i$，高分辨率图像为 $X_i^h$，低分辨率全局图为 $X_i^l$，OCR 或视觉 token 长度为 $N$，滑窗长度为 $w$，步长为 $s$。

token 是模型处理输入时的基本单元。文本模型里的 token 可能是一个词或子词；视觉模型里的 token 常来自图像 patch。patch 是把图像切成固定大小的小块，例如每个小块是 `16 x 16` 像素。

如果图像高为 $H$，宽为 $W$，patch 大小为 $p$，视觉 token 数可近似为：

$$
N \approx \lceil H/p \rceil \times \lceil W/p \rceil
$$

全注意力的计算代价近似为：

$$
O(N^2)
$$

注意力是 Transformer 用来让 token 互相读取信息的机制。$O(N^2)$ 的意思是 token 数翻倍，token 两两交互的配对数大约变成四倍。

数值例子：

| 分辨率 | token 估计 | 相对 token 数 | 相对注意力配对数 |
|---|---:|---:|---:|
| 1024 x 1024, p=16 | 4096 | 1x | 1x |
| 2048 x 2048, p=16 | 16384 | 4x | 16x |

这说明高分辨率不是免费提升。图片从 `1024 x 1024` 放大到 `2048 x 2048`，模型确实更容易看清小字、表格线和坐标轴刻度，但 token 数大约变成 4 倍，全注意力配对数大约变成 16 倍。显存、延迟和成本都会明显上升。

因此工程上常用分层处理：

| 阶段 | 输入 | 目标 |
|---|---|---|
| 全局粗读 | 低分辨率整页 $X_i^l$ | 判断页主题、版面、候选区域 |
| 候选定位 | 页级或区域级特征 | 找出可能含答案的位置 |
| 局部精读 | 高分辨率局部图 $X_i^h$ crop | 读取小字、表格、图表细节 |
| 答案生成 | 文本 + 视觉证据 | 输出答案、证据页、置信度 |

机制图可以写成：

```text
输入页图
  -> 低分辨率全局编码
  -> 候选页/候选区域定位
  -> 高分辨率局部读取
  -> 输出答案 + 证据
```

长文档还会遇到输入过长的问题。假设总 token 数为 $L$，滑窗长度为 $w$，步长为 $s$，窗口数量可近似为：

$$
K = \lceil (L - w)/s \rceil + 1
$$

滑窗是把长输入切成多个重叠片段。重叠区能提高召回，因为答案如果刚好跨越窗口边界，重叠可以避免被切断。但重叠越大，重复计算越多，多个窗口还可能返回冲突证据。

真实工程例子：法务系统要回答“合同第 17 页的违约金是多少”。稳妥做法通常不是把整份合同所有页面都用最高分辨率塞进模型，而是先用低分辨率或 OCR 检索定位到第 17 页及相邻页，再对相关区域做高分辨率切片，最后抽取金额并回查原图证据。这样能兼顾准确率和成本。

---

## 代码实现

一个可落地的最小 pipeline 包括五层：输入层、预处理层、检索层、推理层、输出层。

| 层级 | 作用 | 示例 |
|---|---|---|
| 输入层 | 加载 PDF 或图片 | PDF 转页图 |
| 预处理层 | 缩放、切片、OCR | 低分辨率图、高分辨率 crop |
| 检索层 | 页级排序、区域定位 | top-k 页、候选框 |
| 推理层 | span 提取、数值比较、文本归一化 | 找金额、算增长率 |
| 输出层 | 返回答案、证据页、置信度 | answer + evidence |

新手版流程：

1. 先把 PDF 或页面转成图片。
2. 低分辨率跑一遍，找出可能相关的页。
3. 对候选页做高分辨率切片。
4. 在切片里找答案，并把答案对应回原图位置。

伪代码如下：

```python
pages = load_document(doc)
coarse_scores = rank_pages_low_res(pages, question)

candidate_pages = select_topk(coarse_scores, k=3)
crops = []
for page in candidate_pages:
    crops.extend(sliding_window(page, window=w, stride=s))

evidence = find_answer(crops, question)
answer = normalize_and_parse(evidence)
return answer, evidence
```

下面是一个可运行的 Python 玩具实现。它不调用真实 OCR 或多模态模型，而是模拟“页级粗排 + 滑窗精读 + 答案归一化”的核心逻辑：

```python
import re
from math import ceil

def normalize_answer(text):
    text = text.strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d+)\s*%", r"\1 %", text)
    return text

def rank_pages_low_res(pages, question):
    q_terms = set(re.findall(r"\w+", question.lower()))
    scores = []
    for i, page in enumerate(pages):
        p_terms = set(re.findall(r"\w+", page.lower()))
        scores.append((i, len(q_terms & p_terms)))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def sliding_window_tokens(tokens, window, stride):
    if len(tokens) <= window:
        return [tokens]
    count = ceil((len(tokens) - window) / stride) + 1
    chunks = []
    for k in range(count):
        start = k * stride
        chunks.append(tokens[start:start + window])
    return chunks

def find_penalty_answer(pages, question, topk=1):
    ranked = rank_pages_low_res(pages, question)
    candidate_ids = [idx for idx, _ in ranked[:topk]]

    for page_id in candidate_ids:
        tokens = pages[page_id].split()
        for chunk in sliding_window_tokens(tokens, window=12, stride=6):
            text = " ".join(chunk)
            match = re.search(r"违约金[:：]?\s*([$]?\d[\d,]*\s*%?)", text)
            if match:
                return {
                    "answer": normalize_answer(match.group(1)),
                    "page": page_id + 1,
                    "evidence": text,
                }
    return None

pages = [
    "第一页 合同主体 付款方式 甲方 乙方",
    "第十七页 违约责任 违约金: $1,200 争议解决 管辖法院",
    "附件 表格 联系方式"
]

result = find_penalty_answer(pages, "第 17 页的违约金是多少")
assert result["answer"] == "1200"
assert result["page"] == 2
assert "违约金" in result["evidence"]
```

答案归一化很关键，因为模型、OCR 和标注答案可能格式不同：

| 原始输出 | 归一化后 |
|---|---|
| `17%` | `17 %` |
| `$1,200` | `1200` |
| `two hundred` | `200` |

DocVQA 的实现更适合抽取式 span 检测：先定位文本区域，再抽取原文片段。InfographicVQA 的实现通常要把 OCR 文本、视觉区域、图例、坐标轴和数值解析放在一起处理。例如问题是“2023 年增长率是多少”，模型不仅要识别“2023”，还要知道哪根柱子对应 2023、纵轴刻度是多少、是否需要和 2022 做比较。

---

## 工程权衡与常见坑

低分辨率适合粗定位，不适合直接终答。原因是小字、表格线、脚注、坐标轴刻度、图例颜色块都可能在缩放时丢失。新手版理解：低分辨率像隔着远处看合同，能知道这页大概是什么，但看不清第 17 页违约金。

高分辨率适合精读，但会显著增加 token、显存、延迟和费用。新手版理解：高分辨率像拿放大镜逐页扫，会很清楚，但太慢、太贵。更稳的工程策略是“低分辨率全局 + 局部高分辨率 + 分页或压缩”。

常见坑如下：

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 分辨率过低 | 小字丢失 | 低分辨率只做粗定位 |
| 分辨率过高 | token 爆炸 | 局部高分辨率 + 压缩 |
| 滑窗重叠过小 | 答案被切断 | 增大重叠 |
| 滑窗重叠过大 | 重复计算 | 控制 stride |
| OCR 错误 | 证据传播错误 | 文本归一化 + 数值校验 |
| 把两类任务混评 | 指标失真 | 分开评测抽取式和非抽取式 |

滑窗重叠的规则是：重叠区提升召回，但会带来重复证据。假设窗口长度 $w=1000$，步长 $s=500$，相邻窗口有 500 个 token 重叠，答案跨边界时更容易被保留。但如果 $s$ 太小，同一段证据会被重复处理很多次，系统延迟上升，还可能生成多个相似答案。

OCR 错误会传递到 span 提取和数值比较。例如 OCR 把 `$1,200` 识别成 `$l,200`，后面的金额解析就可能失败。工程上要做三件事：文本归一化、数值格式校验、证据回查。证据回查是指答案输出后再回到原图或 OCR 框，确认答案确实来自对应位置。

评测时还要避免把 DocVQA 和 InfographicVQA 的错误混在一起。DocVQA 答错常见原因是定位错页、漏读小字、跨栏顺序错。InfographicVQA 答错常见原因是图例颜色误解、坐标轴刻度误读、把标题和局部标签混淆。两类错误都叫“文档理解失败”，但修复方向不同。

---

## 替代方案与适用边界

DocVQA 可以用 OCR + 检索、布局模型、长上下文模型、多模态文档模型或分层检索 + 精读。选择取决于文档长度、版面复杂度、是否需要精确定位、是否允许较高推理成本。

InfographicVQA 对纯 OCR 更不友好。因为信息图里的答案往往不是“哪一行字”，而是图表和文字共同表达的结论。例如“这张信息图的增长率是多少”不能只靠 OCR，因为 OCR 只能读到文字，不能稳定理解柱高、折线趋势、颜色图例和视觉分组。

方案对比：

| 方案 | 适合任务 | 优点 | 局限 |
|---|---|---|---|
| OCR + 检索 | 简单 DocVQA | 便宜、可解释 | 遇到复杂版面不稳 |
| 长上下文模型 | 多页文档 | 便于跨页推理 | token 成本高 |
| 多模态文档模型 | DocVQA/InfographicVQA | 文本和视觉联合 | 依赖高质量输入 |
| 分层检索 + 精读 | 长文档/多页 | 工程上最稳 | 系统复杂度更高 |

边界可以按两个问题判断。

第一，答案是否主要由文本决定。如果问题是“合同里某个词在哪一段”“发票金额是多少”“报告作者是谁”，OCR + 检索或 DocVQA 风格方案通常够用。只要 OCR 质量好、版面顺序处理正确，系统可以做到便宜且可解释。

第二，答案是否依赖视觉结构。如果问题是“哪个类别增长最快”“2023 年增长率是多少”“图中红色区域代表什么趋势”，就需要 InfographicVQA 风格方案。模型必须把文字、图形、颜色、坐标和图例放在同一个语义空间里理解。

真实工程中，最稳的路线通常是混合方案：先用 OCR 和低分辨率视觉模型做召回，再用多模态模型精读候选区域，最后用规则或程序做数值校验。这样不会把全部压力放在单个模型上，也更容易定位错误来源。

---

## 参考资料

如果要复现任务定义，先看官方数据集页；如果要理解方法，再看论文页；如果要落地实现，再看模型文档。参考资料建议按“数据集 -> 方法 -> 工程库”顺序阅读。

| 类型 | 资源 |
|---|---|
| 数据集定义 | [DocVQA 官方数据集页](https://www.docvqa.org/datasets/docvqa)、[InfographicVQA 官方数据集页](https://www.docvqa.org/datasets/infographicvqa) |
| 任务论文 | [InfographicVQA 论文页](https://huggingface.co/papers/2104.12756) |
| 长文本机制 | [Longformer 文档](https://huggingface.co/docs/transformers/en/model_doc/longformer) |
| 统一多模态建模 | [LayoutLMv3 论文页](https://www.microsoft.com/en-us/research/publication/layoutlmv3-pre-training-for-document-ai-with-unified-text-and-image-masking/)、[Donut 文档](https://huggingface.co/docs/transformers/v4.23.0/model_doc/donut) |
| 多页文档 | [Hi-VT5 / MP-DocVQA 论文页](https://www.researchgate.net/publication/366212420_Hierarchical_multimodal_transformers_for_Multi-Page_DocVQA) |
| 新一代文档多模态 | [mPLUG-DocOwl2 论文页](https://huggingface.co/papers/2409.03420) |
