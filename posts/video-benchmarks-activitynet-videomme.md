## 核心结论

ActivityNet-QA 和 Video-MME 都评测“看视频回答问题”，但它们评测的不是同一种能力。

ActivityNet-QA 是面向长视频的开放式 VideoQA 基准。VideoQA 指“视频问答”：输入一段视频和一个问题，模型输出答案。它重点考察模型能否从长视频中找到相关片段，再结合动作、人物、位置和先后关系生成短答案。它的数据规模是 5,800 段视频、58,000 个 QA。

Video-MME 是面向短、中、长视频的综合视频理解基准。它重点考察模型在多领域、多时长、多模态条件下的理解能力。多模态指输入不只有画面，还可能包含字幕和音频。它的数据规模是 900 段视频、2,700 个 QA，题型是 4 选 1 多选题。

两者不能直接横比。ActivityNet-QA 更适合评估开放式短答案能力；Video-MME 更适合评估真实场景里的长视频理解能力，尤其是跨片段记忆、字幕/音频利用和分时长表现。

同样是“看视频回答问题”，ActivityNet-QA 更像：请说出 10 分钟视频中某个动作发生后的结果。Video-MME 更像：给你视频、字幕或音频，从 4 个选项里选出正确答案，并分别统计短视频、中视频、长视频上的表现。

| 维度 | ActivityNet-QA | Video-MME |
|---|---|---|
| 任务形式 | 视频 + 问题 -> 短答案 | 视频 + 问题 + 选项 -> 选项编号 |
| 视频长度 | 主要面向长视频 | 覆盖短、中、长视频 |
| 题型 | 开放式问答 | 4 选 1 多选题 |
| 评测指标 | Acc、WUPS | MCQ accuracy |
| 主要能力 | 片段定位、时序推理、空间推理、短答案生成 | 长上下文、多模态对齐、跨片段推理、多领域判断 |
| 是否利用字幕/音频 | 主要以视觉视频问答为核心 | 明确评测 frames、frames+subtitles、frames+audio 等设置 |

---

## 问题定义与边界

设视频为 $v$，问题为 $q$，真实答案为 $a$，模型输出为 $\hat a$。开放式 VideoQA 的基本形式是：

$$
v, q \rightarrow \hat a
$$

如果任务使用字幕和音频，可以写成：

$$
v, q, (\mathrm{subs}, \mathrm{audio}) \rightarrow \hat k
$$

其中 $\hat k$ 是模型预测的选项编号，例如 A、B、C、D。

ActivityNet-QA 的答案是开放式短文本。开放式答案指模型不是从固定选项里挑，而是自己生成一句或几个词。例如问题可以是“谁先进入房间？”“左边的人在做什么？”模型可能输出“a man”“cutting vegetables”“the woman”等短答案。

Video-MME 的答案是多选题。多选题指标准答案是一个选项编号，模型只要选中正确选项就算对。例如问题可以是“以下哪个选项最符合视频内容？”模型需要结合帧、字幕、音频判断 A/B/C/D 哪个正确。

| 对比项 | 开放式问答 | 多选题 |
|---|---|---|
| 输出形式 | 自由文本 | 固定选项编号 |
| 主要难点 | 答案生成、同义词匹配、短答案标准化 | 选项判别、干扰项排除、上下文对齐 |
| 评分方式 | 精确匹配或软匹配 | 是否选中正确选项 |
| 示例 | “左边的人在做什么？”->“跳舞” | “视频中人物在做什么？A 跑步 B 做饭 C 游泳 D 开车”-> B |
| 常见误差 | 语义对但文字不同 | 被相似干扰项误导 |

这里要明确边界：VideoQA 不是视频摘要、不是视频检索，也不是时间定位任务。

视频摘要是把整段视频压缩成一段文字；视频检索是从大量视频中找出与查询最相关的视频；时间定位是输出某个事件发生的开始和结束时间。ActivityNet-QA 和 Video-MME 可以涉及定位、摘要或检索能力，但它们最终评测的仍是“回答问题是否正确”。

---

## 核心机制与推导

ActivityNet-QA 的核心机制可以概括为：先找相关片段，再判断答案。

长视频的信息密度不均匀。一个问题通常只和其中几秒或几十秒有关。模型如果把整段视频平均看待，就容易被无关片段稀释。新手可以这样理解：ActivityNet-QA 不是单纯“看懂整段视频”，而是先从长视频里找到问题相关的一小段，再回答一句短话。

典型流程是：

```text
视频切帧 -> 帧/片段编码 -> 问题编码 -> 多模态融合 -> 答案生成 -> Acc/WUPS 评分
```

编码指把图片、文字或音频转换成模型可以计算的向量。融合指把视频向量和问题向量合在一起，让模型判断哪些视觉信息与问题相关。

ActivityNet-QA 通常需要处理三类线索：

| 线索类型 | 白话解释 | 示例问题 |
|---|---|---|
| 时序线索 | 事件先后关系 | “谁先进入房间？” |
| 空间线索 | 人或物在画面中的位置关系 | “左边的人在做什么？” |
| 动作线索 | 人或物正在发生的行为 | “女孩拿起杯子后做了什么？” |

它的精确匹配准确率可以写成：

$$
\mathrm{Acc}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat a_i=a_i]
$$

其中 $N$ 是问题总数，$\mathbf{1}[\cdot]$ 是指示函数；条件成立时取 1，否则取 0。

但开放式答案有一个问题：语义相近的答案可能字符串不同。例如 “bike” 和 “bicycle” 意思接近，但精确匹配会判错。因此 ActivityNet-QA 还使用 WUPS。WUPS 是词级软匹配指标，用 WordNet 等语义关系衡量预测答案和标准答案是否接近：

$$
\mathrm{WUPS}_{\tau}=\frac{1}{N}\sum_{i=1}^{N}\mathrm{softMatch}_{\tau}(\hat a_i,a_i)
$$

其中 $\tau$ 是相似度阈值，常见设置包括 0.0 和 0.9。阈值越高，匹配越严格。

Video-MME 的核心机制是：长上下文 + 多模态对齐 + 多选判别。

长上下文指模型需要处理更长的视频内容，并记住较远位置的信息。多模态对齐指画面、字幕和音频必须按时间对应起来。例如某一帧里人物张嘴说话，字幕里出现关键句，音频中有门铃声，这三种信息要被放在同一时间段附近理解。

新手可以这样理解：Video-MME 不是只看几帧就够。短视频可能靠对象、属性、动作识别就能答对；长视频的问题常常要求模型把几分钟前和几分钟后的事件连起来。尤其当关键信息在字幕或音频里时，只看画面会漏掉答案。

Video-MME 的多选准确率是：

$$
\mathrm{Acc}_{\mathrm{MCQ}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat k_i=k_i]
$$

其中 $\hat k_i$ 是模型预测的选项编号，$k_i$ 是正确选项编号。每题 4 选 1，随机猜测基线是 25%。

| 机制维度 | ActivityNet-QA | Video-MME |
|---|---|---|
| 第一目标 | 生成短答案 | 选出正确选项 |
| 视频处理重点 | 找到问题相关片段 | 覆盖短/中/长视频上下文 |
| 融合重点 | 视频片段与问题融合 | 帧、字幕、音频与问题融合 |
| 评测重点 | Acc、WUPS | 分时长、分领域、分模态的 MCQ accuracy |
| 易错来源 | 同义答案、片段定位失败 | 长程信息丢失、字幕/音频未对齐、干扰项误导 |

玩具例子：一个 30 秒视频里，前 10 秒有人切菜，中间 10 秒有人倒油，最后 10 秒有人炒菜。问题是“倒油之后发生了什么？”如果模型只看到前 8 秒，它会回答“切菜”，这是采样漏掉了关键片段，不一定是动作识别能力差。正确机制应该先覆盖完整时间轴，再根据“之后”找到倒油后面的片段，输出“炒菜”。

真实工程例子：会议录像问答中，用户问“产品经理在工程师演示之后提出了什么风险？”答案可能在 40 分钟会议的第 32 分钟。模型必须先定位“工程师演示”，再找到后续发言，并结合字幕识别具体风险。如果只抽少量均匀帧，画面上可能一直是会议室，关键信息完全在语音和字幕里。

---

## 代码实现

实现评测时要把流程拆开：数据读取、视频采样、模型推理、结果评测。不要把模型输出和评测逻辑写在一起，否则后面很难替换模型或指标。

推荐模块划分如下：

| 模块 | 输入 | 处理 | 输出 | 指标相关性 |
|---|---|---|---|---|
| loader | 标注文件、视频路径 | 读取样本和标准答案 | 样本对象 | 不直接算分 |
| sampler | 视频文件 | 按策略抽帧或切片 | frames/clips | 影响信息覆盖 |
| inferencer | frames、question、subs、audio | 调用模型 | answer 或 option | 产生预测 |
| scorer | prediction、ground truth | 计算 Acc/WUPS/MCQ Acc | 分数 | 负责评测 |

最小流程可以写成：

```python
frames = sample_video(video, strategy="uniform")
pred = model.answer(frames, question, subtitles=subs, audio=audio)
score = evaluate(pred, gt, metric="acc")
```

下面是一个可运行的 Python 玩具评测代码。它不依赖真实视频模型，只演示 ActivityNet-QA 风格的开放式准确率、简化 WUPS，以及 Video-MME 风格的多选准确率和 short/medium/long 分桶统计。

```python
from collections import defaultdict

def exact_acc(preds, gts):
    assert len(preds) == len(gts)
    return sum(p.strip().lower() == g.strip().lower() for p, g in zip(preds, gts)) / len(gts)

def soft_match(pred, gt):
    synonyms = {
        ("bike", "bicycle"),
        ("bicycle", "bike"),
        ("man", "person"),
        ("person", "man"),
    }
    p = pred.strip().lower()
    g = gt.strip().lower()
    return 1.0 if p == g or (p, g) in synonyms else 0.0

def wups_like(preds, gts):
    assert len(preds) == len(gts)
    return sum(soft_match(p, g) for p, g in zip(preds, gts)) / len(gts)

def mcq_acc(samples):
    assert samples
    return sum(s["pred"] == s["gt"] for s in samples) / len(samples)

def bucket_acc(samples):
    buckets = defaultdict(list)
    for s in samples:
        buckets[s["duration_bucket"]].append(s)
    return {bucket: mcq_acc(items) for bucket, items in buckets.items()}

activitynet_preds = ["bike", "cutting vegetables", "woman"]
activitynet_gts = ["bicycle", "cutting vegetables", "man"]

assert exact_acc(activitynet_preds, activitynet_gts) == 1 / 3
assert wups_like(activitynet_preds, activitynet_gts) == 2 / 3

videomme_samples = [
    {"id": 1, "duration_bucket": "short", "pred": "A", "gt": "A"},
    {"id": 2, "duration_bucket": "long", "pred": "B", "gt": "C"},
    {"id": 3, "duration_bucket": "long", "pred": "D", "gt": "D"},
]

assert mcq_acc(videomme_samples) == 2 / 3
assert bucket_acc(videomme_samples) == {"short": 1.0, "long": 0.5}
```

ActivityNet-QA 的工程流程更像读取一个索引文件，加载对应视频片段，抽取若干帧，把问题输入模型，生成短答案，再和标准答案计算 Acc/WUPS。

Video-MME 的工程流程更强调分组统计：加载视频帧、字幕和音频，对每道 4 选 1 题输出选项编号，最后统计整体准确率，并按 short、medium、long 拆分。分桶统计很重要，因为一个模型可能短视频很好，长视频明显下降；只看总分会掩盖这个问题。

---

## 工程权衡与常见坑

最大风险是把两个基准的分数当成同一类指标比较。ActivityNet-QA 的开放式短答案和 Video-MME 的多选题不是同一个难度体系。一个模型在多选题上高，不代表它能稳定生成开放式短答案；一个模型在开放式问答上低，也可能是答案同义词匹配不充分。

长视频评测的核心坑是采样和信息丢失。如果一个 30 分钟视频只抽 8 帧，很多关键事件根本没有被抽到。模型错的不一定是“理解差”，而是“没看到”。这在长视频、多事件、字幕信息密集的场景里尤其明显。

Video-MME 还要特别注意字幕和音频。根据 Video-MME 论文报告，Gemini 1.5 Pro 仅用帧时，短视频准确率约 82.3%，长视频约 67.5%；加入字幕后长视频可到约 76.3%。这说明长视频问题里，字幕能补足很多画面无法表达的信息。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 直接横比 ActivityNet-QA 和 Video-MME 分数 | 误判模型能力 | 分别报告 Acc/WUPS 和 MCQ accuracy |
| 长视频只抽少量均匀帧 | 关键事件漏采样 | 增加采样密度，或先分段再聚合 |
| 忽略字幕/音频 | 低估 Video-MME 上的真实能力 | 分别报告 frames、frames+subs、frames+audios |
| 只报总分 | 看不到长视频退化 | 至少拆 short/medium/long |
| 不区分任务类型 | 无法定位弱点 | 拆 perception 和 reasoning，或按题型统计 |
| 开放式答案只做严格字符串匹配 | 同义答案被误判 | 同时报 Acc 和 WUPS 类软匹配指标 |

可以把多模态设置写成：

$$
S_{\mathrm{frames}} \neq S_{\mathrm{frames+subs}} \neq S_{\mathrm{frames+audio}}
$$

这里 $S$ 表示分数。三者不同是正常现象，因为它们使用的信息源不同。工程报告里应明确模型到底看了哪些输入，不能只写“视频理解准确率”。

评测建议是至少拆两个维度：视频时长和任务类型。视频时长可以拆 short、medium、long；任务类型可以拆感知类和推理类。感知类问题偏向“画面里有什么”，推理类问题偏向“事件之间有什么关系”。

---

## 替代方案与适用边界

如果任务目标不是开放式问答，也不是多选视频理解，就不应直接拿 ActivityNet-QA 或 Video-MME 作为唯一评价标准。

例如做视频检索，用户输入“找出所有出现红色汽车的视频”，核心指标应该是召回率、精确率、排序质量，而不是问答准确率。做时间定位，用户关心的是事件开始和结束时间，应该使用 temporal grounding 或 action localization 指标。做视频摘要，用户关心摘要是否覆盖关键事件，而不是某道题是否答对。

| 场景 | 推荐基准或评测方向 | 原因 |
|---|---|---|
| 会议录像问答 | ActivityNet-QA + Video-MME 思路结合 | 既需要局部事实定位，也需要字幕和长上下文 |
| 课程录像理解 | Video-MME 更接近 | 讲解内容常在字幕/音频中，且跨分钟关联明显 |
| 商品短视频识别 | 对象检测或动作识别基准 | 核心是物体、属性、动作，不一定需要长程 QA |
| 安防视频事件定位 | 时间定位/动作检测基准 | 重点是事件发生时间段 |
| 短视频内容审核 | 分类、多标签识别基准 | 重点是类别判断和风险标签 |
| 训练数据质检 | 自定义 QA + 抽样人工复核 | 真实业务分布通常不同于公开基准 |

选型可以按任务目标判断：

```text
目标是生成短答案？
  -> 优先看开放式 VideoQA，如 ActivityNet-QA

目标是多选判别，并且要覆盖短/中/长视频？
  -> 优先看 Video-MME

目标是找视频或找片段？
  -> 用检索或时间定位评测

目标是识别单帧对象或短动作？
  -> 用对象检测、图像分类或动作识别基准
```

做会议录像问答时，ActivityNet-QA 更接近“谁先发言、某个动作何时发生、谁在左边”这类开放式短答案能力；Video-MME 更接近“要不要看字幕、音频，如何处理跨分钟的信息整合”这类上线能力。

做商品短视频识别时，问题可能只是“视频里是否出现某个品牌包装”“商品是否被拿起展示”。这类场景不一定需要长视频 QA，使用对象检测、动作识别或多标签分类会更直接。

最终边界可以总结为三句话：开放式短答案、多选判别、长上下文推理不能混用。ActivityNet-QA 适合看“能否生成答案”；Video-MME 适合看“能否在多模态长上下文里选对”。业务系统评测应根据目标任务重组指标，而不是只追一个公开榜单分数。

---

## 参考资料

1. [ActivityNet-QA: A Dataset for Understanding Complex Web Videos via Question Answering](https://arxiv.org/abs/1906.02467)
2. [ActivityNet-QA HTML Version](https://ar5iv.labs.arxiv.org/html/1906.02467)
3. [ActivityNet-QA GitHub Repository](https://github.com/MILVLG/activitynet-qa)
4. [Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://arxiv.org/abs/2405.21075)
5. [Video-MME HTML Version](https://ar5iv.labs.arxiv.org/html/2405.21075)
6. [Video-MME GitHub Repository](https://github.com/MME-Benchmarks/Video-MME)

本文中所有数字均来自原论文或其公开仓库；对工程场景、模块划分和选型建议的表述是本文基于这些资料的解释与推断。
