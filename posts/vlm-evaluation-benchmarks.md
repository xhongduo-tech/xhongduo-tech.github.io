## 核心结论

VLM，视觉语言模型，指同时处理图像和文本输入的模型。评测这类模型，不能只看一个总分，因为“看清图片”“理解领域知识”“跨模态推理”“读懂文字与文档”是不同能力。

当前较完整的一套全景评测，通常由以下几类基准拼接而成：

| Benchmark | 主要覆盖能力 | 典型任务形式 | 常见指标 | 适合回答的问题 |
|---|---|---|---|---|
| MMBench | 感知、知识、推理的细粒度拆分 | 多项选择题 | CircularEval 后的准确率 | 模型到底弱在看图、常识还是推理 |
| MMMU | 大学级多学科理解 | 图文混合问答/选择 | 准确率 | 模型能否做“跨学科考试题” |
| MathVista | 数学视觉推理 | 图形、公式、函数图、几何题 | 准确率 | 模型能否“看图算题” |
| OCRBench | OCR 与场景文字理解 | 图中文字识别、问答 | 准确率/子任务分 | 模型是否真的读懂文字 |
| ChartQA | 图表问答 | 柱状图、折线图、饼图问答 | 准确率 | 模型是否能读图表中的数值关系 |
| DocVQA | 文档理解 | 发票、表单、报告问答 | 准确率/ANLS 等 | 模型是否理解版面与字段关系 |

这套组合的价值不在于“把榜单拉长”，而在于建立一个能力向量。一个模型可以写成：

$$
detail\_vector = [Perception,\ Knowledge,\ Reasoning,\ Math,\ OCR,\ Doc,\ Chart]
$$

这比单一总分更有工程价值。因为真正上线时，失败往往来自某一维短板，而不是平均水平。

MMBench 里常见的思想可以概括为：

$$
CircularEval(Model) = \operatorname{mean}_{s \in \text{circular shifts}} score(Model, s)
$$

意思是把选择题选项做循环位移后重复评测，抵消“模型偏爱 A 或偏爱第一个选项”的顺序偏差。

玩具例子：同一个模型如果在 MMBench 感知维度拿到 82 分，在 MathVista 只有 46 分，在 OCRBench 有 87 分，那么结论不是“模型差”，而是“它能看清字，也能识别一般图像，但一旦需要数学图形推理就明显掉队”。

---

## 问题定义与边界

这篇文章讨论的是“多模态融合评测”，即模型要同时依赖视觉输入和文本理解来完成任务。边界上，下面两类不在核心讨论范围内：

| 不纳入本文核心范围 | 原因 |
|---|---|
| 纯文本大模型评测 | 没有视觉输入，不能检验跨模态能力 |
| 纯图像分类/检测评测 | 只测视觉感知，不测语言理解和回答能力 |

因此，问题可以定义为：

1. 横向全景：是否覆盖感知、OCR、文档、图表、数学、跨学科知识。
2. 纵向细化：是否能把错误进一步归因到细粒度能力，而不是只报一个均值。

MMBench 在纵向细化上最典型。它不是只说“模型做对了多少题”，而是把能力分层。可以抽象写成：

$$
L1 = \{Perception,\ Reasoning\}
$$

$$
L2 = \{coarse\ perception,\ fine\ perception,\ single\ reasoning,\ cross\ reasoning,\dots\}
$$

$$
L3 = \{OCR,\ spatial,\ attribute,\ relation,\ logic,\ commonsense,\dots\}
$$

白话解释：L1 是大类，L2 是中类，L3 是能直接定位问题的小类。这样你能知道模型是“看不清”还是“看清了但不会推”。

MMMU 的边界不同。它强调大学水平多学科题目，核心难点不是单纯视觉识别，而是把图像、学科知识、题目约束和选项一起整合。比如化学结构图、生物流程图、物理示意图、统计图表，这些都不是“看到物体名称”就能答对。

MathVista 的边界更窄但更深。它聚焦视觉数学，也就是既要识别图中元素，又要做定量推理。比如函数图、几何图、表格、示意图中的数值关系。

OCRBench、ChartQA、DocVQA 则属于专项集。专项集的意义是补洞。因为很多综合 benchmark 会掩盖基础短板，一个模型在综合题上答得不错，不代表它真的能稳定识别票据文字或读取复杂表格。

可以把 VLM 的工作想成一个智能助理：
先看清图里的字和结构，再理解题目，再做推理，再输出答案。任何一步弱，最终结果都会错。

---

## 核心机制与推导

MMBench 的关键机制是“细粒度拆分 + CircularEval”。

先看为什么需要 CircularEval。多项选择题天然有顺序偏差。有些模型偏向输出 `A`，有些模型更容易选择第一个或最后一个选项。如果不处理，分数会被选项排布污染。于是 MMBench 对同一题做多个“循环移位”版本，比如：

- 原始：A, B, C, D
- 位移 1 次：B, C, D, A
- 位移 2 次：C, D, A, B
- 位移 3 次：D, A, B, C

最终取平均：

$$
CircularEval(Model)=\frac{1}{K}\sum_{k=1}^{K} score(Model,\ shift_k)
$$

这里的 $K$ 是循环版本数。白话解释：把题目像转盘一样转几次，看模型是否真的理解内容，而不是记住某个字母位置。

MMMU、MathVista、OCRBench 这类基准，主指标通常更直接：

$$
acc=\frac{correct}{total}
$$

也就是正确题数除以总题数。公式简单，但解释时不能只看一个整体准确率，因为不同任务难度和分布差异很大。工程上更常见的做法是拆成子域向量，例如：

$$
detail\_vector=[acc_{perception},acc_{knowledge},acc_{reasoning},acc_{math},acc_{ocr}]
$$

这样可以比较两个模型的差异方向，而不是只比较“总分谁高”。

### 玩具例子

假设一个模型在 4 个 benchmark 上表现如下：

| 维度 | 分数 |
|---|---|
| MMBench-Perception | 0.84 |
| MMMU | 0.58 |
| MathVista | 0.43 |
| OCRBench | 0.89 |

这个结果的含义不是“模型平均 0.685”。更准确的解释是：

- 感知和文字识别不错。
- 跨学科理解一般。
- 数学视觉推理明显短板。

也就是说，若你的业务是票据识别，它可用；若你的业务是几何辅导，它未必可用。

### 真实工程例子

金融审批智能助手经常要同时处理三类输入：

| 输入对象 | 对应 benchmark 能力 | 失败风险 |
|---|---|---|
| 报表图表 | ChartQA | 读错趋势或数值 |
| 合同扫描件 | DocVQA + OCRBench | 提错字段、漏关键信息 |
| 风险解释材料 | MMMU/MMBench Reasoning | 推理链不完整 |

如果模型在 OCRBench 高、ChartQA 低，那么它可能能提取发票号码，但不能可靠解释收益率曲线的拐点。如果 MMBench 的 spatial reasoning 下降，可能意味着布局理解退化，进一步影响 DocVQA 字段定位。

所以综合评测的目的，不是做一个更大的排行榜，而是建立“上线前故障地图”。

---

## 代码实现

下面给一个最小可运行的 Python 示例。它做三件事：

1. 读取多个 benchmark 的结果。
2. 计算 MMBench 的 CircularEval。
3. 合成一个 `detail_vector` 供后续画雷达图或做回归分析。

```python
from statistics import mean

def circular_eval(shift_scores):
    assert len(shift_scores) > 0
    for x in shift_scores:
        assert 0.0 <= x <= 1.0
    return mean(shift_scores)

def accuracy(correct, total):
    assert total > 0
    assert 0 <= correct <= total
    return correct / total

def build_detail_vector(result):
    mmbench = circular_eval(result["mmbench_shift_scores"])
    mmmu = accuracy(result["mmmu_correct"], result["mmmu_total"])
    mathvista = accuracy(result["mathvista_correct"], result["mathvista_total"])
    ocrbench = accuracy(result["ocrbench_correct"], result["ocrbench_total"])
    chartqa = accuracy(result["chartqa_correct"], result["chartqa_total"])
    docvqa = accuracy(result["docvqa_correct"], result["docvqa_total"])

    detail_vector = {
        "perception_score": result["perception_score"],
        "knowledge_score": result["knowledge_score"],
        "reasoning_score": result["reasoning_score"],
        "mmbench_score": round(mmbench, 4),
        "mmmu_score": round(mmmu, 4),
        "math_score": round(mathvista, 4),
        "ocr_score": round(ocrbench, 4),
        "chart_score": round(chartqa, 4),
        "doc_score": round(docvqa, 4),
    }
    return detail_vector

sample = {
    "mmbench_shift_scores": [0.76, 0.78, 0.75, 0.77],
    "mmmu_correct": 58,
    "mmmu_total": 100,
    "mathvista_correct": 43,
    "mathvista_total": 100,
    "ocrbench_correct": 89,
    "ocrbench_total": 100,
    "chartqa_correct": 61,
    "chartqa_total": 100,
    "docvqa_correct": 72,
    "docvqa_total": 100,
    "perception_score": 0.81,
    "knowledge_score": 0.62,
    "reasoning_score": 0.57,
}

vector = build_detail_vector(sample)

assert vector["mmbench_score"] == 0.765
assert vector["math_score"] == 0.43
assert vector["ocr_score"] == 0.89
assert vector["reasoning_score"] == 0.57

print(vector)
```

字段含义可以先约定清楚：

| 字段 | 含义 | 来源 |
|---|---|---|
| `mmbench_shift_scores` | 同一道题组在不同选项位移下的分数 | MMBench |
| `perception_score` | 看清对象、文字、位置、属性的能力 | MMBench 细分类 |
| `knowledge_score` | 调用学科或常识知识的能力 | MMBench/MMMU |
| `reasoning_score` | 根据多模态信息做推理的能力 | MMBench/MMMU |
| `math_score` | 数学图形与符号推理能力 | MathVista |
| `ocr_score` | 图中文字识别与利用能力 | OCRBench |
| `chart_score` | 图表数值理解能力 | ChartQA |
| `doc_score` | 文档版面与字段问答能力 | DocVQA |

如果要把结果做成雷达图，核心不是画图代码，而是先保证向量定义稳定。否则今天的 `reasoning_score` 包含空间推理，明天又混入 OCR 子题，趋势图就失真了。

---

## 工程权衡与常见坑

最大的坑不是“哪个 benchmark 更火”，而是污染。污染，指模型训练时见过测试样本或极其接近的变体，导致评测分数虚高。

最常见的现象是：公开 benchmark 发布时间较早，训练数据大规模抓取互联网内容，题目、答案、截图、变体文本可能被吸入预训练或指令微调数据。结果是模型看起来很强，实际泛化并没有那么强。

可以用两类指标描述污染检测或抗污染效果：

$$
fidelity=\frac{\text{修复后保留的原始正常能力}}{\text{原始正常能力}}
$$

$$
resistance=\frac{\text{对污染样本的性能抑制或鲁棒提升}}{\text{期望抑制目标}}
$$

白话解释：`fidelity` 看“去污染后有没有把正常能力也一起伤到”，`resistance` 看“面对可疑样本时是否更不容易被诱导”。

常见手段如下：

| 手段 | 做法 | 目的 | 常见风险 |
|---|---|---|---|
| 语义扰动 | 改写题干、替换非关键表述 | 检测是否靠记忆模板作答 | 可能改变题目难度 |
| 视觉扰动 | 裁剪、缩放、换字体、轻微重排 | 检测是否记住像素模式 | 扰动过大可能破坏题意 |
| 选项重排 | 改变选项顺序 | 检测顺序偏置 | 对开放问答无效 |
| 重写样本 | 保留知识点但重做题面 | 检测真实泛化能力 | 成本高，需要人工审核 |
| 私有留出集 | 业务内部构造未公开测试集 | 减少公开集泄漏 | 覆盖面可能不足 |

这里有两个工程权衡。

第一，benchmark 拼得越多，评测成本越高。多模态题普遍推理开销大，尤其是长文档、复杂图表、数学题，单次评测的 token 和图像处理成本都不低。

第二，指标统一会损失细节。你把所有结果压成一个 `detail_vector`，比较方便做 dashboard；但每个 benchmark 的评分细则、题型分布和错误类型并不完全可比。所以仪表盘适合做趋势监控，不适合替代原始明细。

一个初学者容易踩的坑是：看到 OCRBench 高分，就默认 DocVQA 也会高。实际上两者不同。OCRBench 更像“能不能看清并读出字”，DocVQA 更强调“字在什么位置、属于哪个字段、和版面结构是什么关系”。

---

## 替代方案与适用边界

如果资源有限，没有必要一开始就跑完整全景集。更务实的做法是按业务目标裁剪。

| 方案 | 覆盖能力 | 优点 | 明显缺口 | 适用场景 |
|---|---|---|---|---|
| MMBench 子集 | 感知 + 推理细分 | 诊断能力强 | 数学/OCR 专项不够深 | 模型迭代早期体检 |
| MathVista + OCRBench | 数学 + 文字识别 | 成本相对可控 | 缺少跨学科和文档结构 | 教育、票据、截图助手 |
| ChartQA + DocVQA | 图表 + 文档 | 贴近办公/金融流程 | 基础感知诊断不足 | 企业文档智能体 |
| MMMU + MMBench | 广度 + 细粒度归因 | 适合综合比较 | 成本高，专项覆盖仍需补 | 研究型评测 |
| LLaVA Bench / COCO-QA 等补充集 | 通用视觉问答 | 快速得到概貌 | 细粒度不足 | 轻量基线验证 |

对于零基础到初级工程师，可以记一个简单判断：

- 想知道模型哪里弱，优先 MMBench。
- 想知道模型能不能做“大学题”，优先 MMMU。
- 想知道模型会不会看图做数学，优先 MathVista。
- 想知道模型会不会认字和读文档，优先 OCRBench、DocVQA。
- 想知道模型会不会读报表，补 ChartQA。

“只跑 MathVista + OCRBench 可以吗？”可以，但边界很明确。它更像一把只覆盖数学和文字理解的窄刀，无法告诉你模型在跨学科知识、长链推理、图表逻辑上的真实水平。

所以替代方案不是“错”，而是“你愿意放弃哪些可见性”。

---

## 参考资料

| 名称 | 类型 | 说明 | 用途 |
|---|---|---|---|
| MMBench | GitHub/论文项目 | 多模态综合评测，强调细粒度能力拆分与 CircularEval | 做能力归因与综合比较 |
| MMMU | 论文/官网 | 大学水平多学科多模态理解，约 11.5K 题 | 测试跨学科知识与深推理 |
| MathVista | 官网/论文 | 视觉数学推理数据集，覆盖函数图、几何图、符号等 | 测试“看图做数学” |
| OCRBench | 论文/评测说明 | OCR 与场景文字理解专项 benchmark | 测试认字、读字、用字 |
| ChartQA | 论文/项目 | 图表问答 benchmark | 测试图表数值与关系理解 |
| DocVQA | 比赛/论文 | 文档视觉问答 benchmark | 测试文档版面和字段抽取 |
| VLM 污染检测论文 | OpenReview 论文 | 讨论污染检测、扰动与抗污染指标 | 构建更可信的评测流程 |

1. MMBench: https://github.com/open-compass/MMBench  
2. MMMU: https://meta-metrics.github.io/  
3. MathVista: https://mathvista.github.io/  
4. OCRBench: https://link.springer.com/article/10.1007/s11432-024-4235-6  
5. ChartQA: https://theaiforger.com/benchmarks/chartqa  
6. DocVQA: https://www.docvqa.org/  
7. VLM 污染检测相关论文: https://openreview.net/forum?id=gk6OC3XIZW
