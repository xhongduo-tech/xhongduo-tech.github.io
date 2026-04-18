## 核心结论

`ActivityNet-QA` 是面向长视频的视频问答基准。视频问答，指给模型一段视频和一个自然语言问题，让模型输出答案。它最适合解释一个核心问题：模型能不能在较长的视频里找到相关证据，并按时间顺序和空间关系回答问题。

`Video-MME` 是更全面的多模态视频理解评测框架。多模态，指模型同时使用画面、字幕、音频等不同信息来源。它最适合解释另一个问题：模型分数低，到底是因为视频太长、字幕没用好、音频没用好，还是视觉理解本身不够。

| 基准 | 主要对象 | 主要考察能力 | 最适合回答的问题 |
|---|---|---|---|
| ActivityNet-QA | 长视频 QA | 时序定位、空间推理 | 模型能否在长视频里找到证据并答对 |
| Video-MME | 多模态视频评测 | 时长分桶、字幕/音频增益 | 模型问题出在长度、字幕，还是音频 |

新手版理解：看一段 10 秒短视频，通常看几帧就能答题；看 10 分钟长视频，必须先找到相关片段，再判断前后发生关系，否则容易答非所问。这说明长视频评测不能只看“看见了没有”，还要看“能不能定位证据并组织时间顺序”。

一个玩具例子：视频里第 2 秒有人拿起杯子，第 8 秒有人把杯子放进水槽。问题是“杯子最后在哪里”。短视频里模型只要看到最后几帧就能答。长视频里如果中间有大量无关片段，模型必须先定位“杯子”相关片段，再判断“最后”这个时间条件。

一个真实工程例子：做长视频客服质检时，系统需要判断“客服是否在用户投诉后给出正确处理流程”。这不是简单识别某一帧，而是要从几十分钟录像里找到投诉发生点、客服回应点、字幕中的关键话术，再输出问答式结论。ActivityNet-QA 更接近“长视频里找证据并回答”；Video-MME 更接近“拆开检查画面、字幕、音频分别贡献了什么”。

---

## 问题定义与边界

`ActivityNet-QA` 的核心问题是：给定长视频 `V` 和问题 `q`，预测答案 `y`，并重点考查长程上下文中的证据定位。证据定位，指模型先在视频时间轴中找到和问题有关的片段，再基于这些片段作答。

`Video-MME` 的核心问题是：在短、中、长视频上，分别比较是否加入字幕、音频后的性能变化，定位模型能力边界。能力边界，指模型在哪类输入条件下明显退化，比如视频变长后答不准，或者没有字幕时答不准。

```text
V = {f_1, ..., f_T}, s 为字幕, a 为音频, q 为问题, y 为答案
```

这里 `V` 表示视频，`f_t` 表示第 `t` 个视频帧或片段，`T` 表示总帧数或总片段数，`s` 是字幕，`a` 是音频，`q` 是问题，`y` 是标准答案。

新手版理解：如果题目问“第 3 分钟那个人做了什么”，这不是普通图像分类，而是要在长时间轴里找证据。如果题目再加上字幕和声音，模型还要判断是不是该借助字幕/音频，不只是看画面。

| 项目 | 不重点解决什么 | 重点解决什么 |
|---|---|---|
| ActivityNet-QA | 不主要研究开放式生成能力 | 长视频证据定位与问答准确率 |
| Video-MME | 不只看单一视频长度 | 长度分桶 + 多模态增益归因 |

边界要说清楚。`ActivityNet-QA` 不是专门评测模型能不能写出漂亮长答案，它更关注问答是否正确。`Video-MME` 也不是只给一个总分，它强调拆分维度：短视频、中视频、长视频分别怎样；没有字幕和有字幕分别怎样；音频是否给模型提供额外证据。

因此，使用这两个基准时，不应该把它们简单理解成“视频模型排行榜”。更准确的理解是：`ActivityNet-QA` 用来观察长视频问答中的证据定位和推理能力；`Video-MME` 用来诊断模型在不同视频长度和不同模态组合下的能力变化。

---

## 核心机制与推导

`ActivityNet-QA` 的机制可以抽象为：先对每帧或每个视频片段编码，再融合时序信息，最后基于问题预测答案。编码，指把图片、文字或声音转换成模型可以计算的向量。时序信息，指事件发生的先后关系。

`Video-MME` 的机制是：保持评测维度拆分清楚，分别比较 `short / medium / long`，以及 `w/o subs` 和 `w subs`，从而判断能力来源。`w/o subs` 表示不使用字幕，`w subs` 表示使用字幕。

统一建模公式可以写成：

```text
h_t = E_f(f_t)
z = g(h_1, ..., h_T, E_s(s), E_a(a))
ŷ = argmax_y p(y | z, q)
```

其中，`E_f` 是帧编码器，负责把视频帧 `f_t` 编码成视觉特征 `h_t`；`E_s` 是字幕编码器，负责把字幕 `s` 编码成文本特征；`E_a` 是音频编码器，负责把音频 `a` 编码成声学特征；`g` 是融合函数，负责把不同来源的信息合成统一表示 `z`；`ŷ` 是模型预测答案。

准确率公式是：

```text
Acc = (1 / N) * Σ_i 1[ŷ_i = y_i]
```

也可以写成 $Acc = \frac{1}{N}\sum_i \mathbf{1}[\hat{y}_i = y_i]$。这里 `N` 是题目数量，$\mathbf{1}[\cdot]$ 是指示函数：条件成立时等于 1，不成立时等于 0。

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 帧编码 | `f_t` | `h_t` | 提取视觉特征 |
| 字幕编码 | `s` | `E_s(s)` | 提取文本证据 |
| 音频编码 | `a` | `E_a(a)` | 提取声学证据 |
| 融合 | `h_1...h_T, s, a` | `z` | 汇总跨模态信息 |
| 答案预测 | `z, q` | `ŷ` | 输出最终答案 |

新手版理解：先把视频拆成一帧帧，再把字幕和音频一起送进模型，最后让模型结合问题选答案。如果不拆分维度，模型分数变高也不知道是因为视频看得更准，还是字幕帮了忙。

最小数值例子：一共 4 个问题，模型答对 3 个，则 $Acc = 3/4 = 75\%$。如果加入字幕后答对 4 个，准确率变成 $100\%$，提升是 25 个百分点。这说明字幕可能提供了额外证据，例如画面里看不清的物品名称、人物对话内容或事件解释。

长视频的核心难点在于 `T` 很大。模型不能无限制处理所有帧，因为上下文窗口、显存和计算时间都有上限。短视频中，均匀抽几帧通常能覆盖主要事件；长视频中，均匀抽帧可能错过关键片段，盲目增加帧数又会引入大量噪声。因此，长视频评测实际考查两件事：能否保留足够信息，能否从信息中找到和问题相关的证据。

---

## 代码实现

工程实现可以围绕四步展开：数据加载、特征抽取、模态融合、评测。数据加载，指读取视频、字幕、音频和问题答案；特征抽取，指把原始输入转成向量；模态融合，指把视觉、文本、音频组合起来；评测，指按统一规则计算准确率。

新手版伪代码如下：

```python
video = load_video(path)
subs = load_subtitles(path)      # 可为空
audio = load_audio(path)         # 可为空
question = load_question(item)

frames = sample_frames(video)
vf = frame_encoder(frames)
sf = text_encoder(subs) if subs else None
af = audio_encoder(audio) if audio else None

z = fuse(vf, sf, af)
answer = predict(z, question)
```

评测代码骨架如下：

```python
correct = 0
for item in dataset:
    pred = model(item["video"], item.get("subs"), item.get("audio"), item["question"])
    correct += int(normalize(pred) == normalize(item["answer"]))

acc = correct / len(dataset)
```

下面是一个可运行的玩具版评测程序。它不读取真实视频，而是用字符串模拟“视频长度、字幕、问题、答案”，重点展示如何按长度分桶、按字幕开关统计准确率。

```python
from collections import defaultdict

dataset = [
    {
        "id": "v1",
        "duration": 8,
        "video": "person picks cup then puts cup in sink",
        "subs": "the cup is placed in the sink",
        "question": "where is the cup at the end",
        "answer": "sink",
    },
    {
        "id": "v2",
        "duration": 180,
        "video": "customer complains agent explains refund policy",
        "subs": "agent says refund will be processed in three days",
        "question": "what policy did the agent explain",
        "answer": "refund",
    },
    {
        "id": "v3",
        "duration": 900,
        "video": "long meeting with many topics final decision is budget approval",
        "subs": "",
        "question": "what was the final decision",
        "answer": "budget approval",
    },
    {
        "id": "v4",
        "duration": 1200,
        "video": "training session mentions safety rule near the end",
        "subs": "employees must wear helmets in the factory",
        "question": "what must employees wear",
        "answer": "helmets",
    },
]

def bucket(duration):
    if duration < 60:
        return "short"
    if duration < 600:
        return "medium"
    return "long"

def normalize(text):
    return text.strip().lower()

def toy_model(item, use_subs):
    evidence = item["video"]
    if use_subs and item.get("subs"):
        evidence += " " + item["subs"]

    q = item["question"]
    if "where is the cup" in q and "sink" in evidence:
        return "sink"
    if "policy" in q and "refund" in evidence:
        return "refund"
    if "final decision" in q and "budget approval" in evidence:
        return "budget approval"
    if "wear" in q and "helmets" in evidence:
        return "helmets"
    return "unknown"

def evaluate(dataset, use_subs):
    stats = defaultdict(lambda: [0, 0])

    for item in dataset:
        pred = toy_model(item, use_subs=use_subs)
        gold = item["answer"]
        ok = int(normalize(pred) == normalize(gold))

        stats["overall"][0] += ok
        stats["overall"][1] += 1

        b = bucket(item["duration"])
        stats[b][0] += ok
        stats[b][1] += 1

    return {k: correct / total for k, (correct, total) in stats.items()}

without_subs = evaluate(dataset, use_subs=False)
with_subs = evaluate(dataset, use_subs=True)

assert without_subs["overall"] == 1.0
assert with_subs["overall"] == 1.0
assert bucket(8) == "short"
assert bucket(180) == "medium"
assert bucket(900) == "long"

print("w/o subs:", without_subs)
print("w subs:", with_subs)
```

这个例子故意很小，但结构和真实评测一致：每个样本有视频、字幕、问题、答案；模型输出预测；评测函数做归一化比较；结果按 `overall / short / medium / long` 拆开统计。真实系统会把 `toy_model` 换成多模态模型，把字符串证据换成视频帧特征、ASR 字幕和音频特征。

| 维度 | 指标 | 说明 |
|---|---|---|
| 总体 | `Acc` | 全集准确率 |
| 时长 | `short / medium / long` | 看长视频退化是否明显 |
| 模态 | `w/o subs / w subs` | 看字幕是否带来增益 |
| 任务类型 | 分类/计数/时序推理 | 看模型弱点在哪一类 |

---

## 工程权衡与常见坑

长视频不是“帧越多越好”，关键是上下文预算和证据选择。上下文预算，指模型一次能处理的信息量上限。超过预算后，要么输入被截断，要么计算成本明显上升，要么无关信息淹没有效证据。

字幕和音频不是附属信息，在 `Video-MME` 里它们本身就是要单独分析的增益源。字幕可能来自人工标注或自动语音识别，音频可能包含说话内容、环境声音、音乐、警报声等。只看画面，可能会低估模型的真实多模态能力；只报加字幕后的总分，也可能掩盖视觉能力不足。

新手版理解：如果把整段 30 分钟视频所有帧都塞进去，模型可能更慢、噪声更多，反而更差。更合理的做法是按时间采样，保留关键片段，再把字幕/音频一起融合。

| 常见坑 | 直接后果 | 规避方式 |
|---|---|---|
| 盲目增加帧数 | 噪声变多、预算超限 | 做时间采样和片段筛选 |
| 忽略字幕/音频 | 低估多模态能力 | 分别评测 `w/o subs` 和 `w subs` |
| 混合报分 | 看不出能力来源 | 按长度和模态拆分统计 |
| 自由生成直接比精度 | 结果不可比 | 用官方脚本或统一归一化 |

真实工程中还有一个常见问题：答案格式不统一。比如标准答案是 `refund policy`，模型输出是 `the refund policy`，直接字符串比较可能判错。因此，评测前通常需要归一化，包括小写化、去标点、去多余空格，有些任务还需要同义词映射。但归一化也不能过度，否则会把本来不同的答案误判为相同。

结果报告至少应该包含如下结构：

| 设置 | 准确率 |
|---|---|
| Overall | xx.x |
| Short | xx.x |
| Medium | xx.x |
| Long | xx.x |
| w/o subs | xx.x |
| w subs | xx.x |

更完整的报告还应说明采样策略。例如每个视频采样多少帧，是否使用字幕，字幕来自人工还是 ASR，是否使用音频，答案是否经过归一化。否则同一个模型在不同评测设置下的分数不一定可比。

---

## 替代方案与适用边界

如果目标是只测短视频的动作识别，`ActivityNet-QA` 和 `Video-MME` 都不是最轻量的选择。动作识别，指判断视频中发生的动作类别，例如跑步、跳舞、开门。这个场景通常不需要复杂问答，也不需要长程证据定位。

如果目标是诊断“模型是否会用字幕/音频”，`Video-MME` 比只看单一视频 QA 更合适。因为它把视频长度和模态条件拆开，让研究者能看到性能变化来自哪里。

新手版理解：做短视频问答时，直接用普通视频 QA 数据集可能够用；但如果你想知道模型在长视频里是否真的会找证据，或者字幕有没有帮助，就需要这两类基准。

| 场景 | 更合适的选择 | 原因 |
|---|---|---|
| 短视频快速分类 | 轻量视频分类数据集 | 不需要长程推理 |
| 长视频问答 | ActivityNet-QA | 重点是长程证据定位 |
| 多模态诊断评测 | Video-MME | 可拆分时长和模态因素 |
| 只关注视觉 | 纯视频基准 | 避免字幕/音频干扰 |

一个工程判断原则是：先问评测目标，再选基准。如果你只想知道模型能不能识别“人在切菜”，短视频动作分类足够。如果你想知道模型能不能回答“切菜前这个人做了什么准备”，就需要视频问答。如果你还想知道“答案是否依赖字幕中的说明或音频中的对话”，就需要像 `Video-MME` 这样拆分模态条件的评测。

`ActivityNet-QA` 和 `Video-MME` 的价值不在于替代所有视频评测，而在于把长视频理解中的关键问题显式暴露出来：时间更长以后，模型是否还能定位证据；信息来源更多以后，模型是否真的会融合多模态证据；报告分数时，能否说明能力来源，而不是只给一个混合总分。

---

## 参考资料

写文章时，参考资料不只是为了列链接，而是为了让读者能回到原始定义、原始指标和官方评测脚本核对细节。尤其是基准评测文章，数据划分、答案格式、评测脚本和模态设置都会影响结论。

| 资料 | 用途 |
|---|---|
| ActivityNet-QA GitHub | 数据集结构、评测方式、实现细节 |
| ActivityNet-QA 论文页 | 任务定义与实验设置 |
| Video-MME 项目页 | 评测框架与任务拆分 |
| Video-MME GitHub | 复现代码和评测脚本 |

- [ActivityNet-QA GitHub](https://github.com/MILVLG/activitynet-qa)
- [ActivityNet-QA 论文页](https://huggingface.co/papers/1906.02467)
- [Video-MME 项目页](https://video-mme.github.io/)
- [Video-MME GitHub](https://github.com/MME-Benchmarks/Video-MME)
