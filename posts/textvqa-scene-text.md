## 核心结论

TextVQA 是一个专门测“模型能不能在自然场景里读字并回答问题”的基准。它包含 28,408 张来自 OpenImages 的图片和 45,336 个问答对，问题设计要求答案必须依赖图中的可见文本，比如招牌、包装、价签、路牌，而不是只看物体轮廓就能猜出来。问题是英文，每题有 10 个独立标注答案，评测采用 VQA accuracy，也就是“只要有足够多的人类答案和模型答案一致，就给高分”。这一点决定了它不是纯 OCR 测试，也不是普通 VQA，而是“读字 + 看图 + 理解问题 + 推理作答”的联合测试。[TextVQA 数据集卡](https://huggingface.co/datasets/facebook/textvqa)

一个最容易理解的样本是数据集卡里的问题 `who is this copyrighted by?`。图中文字里出现 `Simon Clancy`，10 个标注答案里大多数也是这个名字。模型如果输出 `simon clancy`，就能得到满分或接近满分。这说明 TextVQA 的重点不是生成华丽句子，而是把图中读到的词和问题准确对齐。[TextVQA 数据集卡](https://huggingface.co/datasets/facebook/textvqa)

从工程角度看，TextVQA 的真正难点有两个。第一，OCR 是光学字符识别，也就是“把图片里的字变成文本”的过程，读错一个字符就会把后面的推理一起带偏。第二，答案空间是开放的，品牌名、人名、地址、编号都可能成为答案，不能只靠一个固定分类器。早期代表模型 LoRRA 用 “Look, Read, Reason, Answer” 四步把视觉特征、OCR 结果和问题融合；后续如 BOV++ 则进一步强调 OCR 与问答模块联动，减少级联误差。[LoRRA 论文](https://textvqa.org/assets/paper/TextVQA.pdf) [BOV++ 论文](https://www.sciencedirect.com/science/article/pii/S0031320323000389)

| 维度 | TextVQA 的定义 |
|---|---|
| 图像数量 | 28,408 |
| 问答数量 | 45,336 |
| 每题参考答案 | 10 个 |
| 输入 | 图像 + 问题 + OCR 文本 |
| 输出 | 一个开放式答案 |
| 核心能力 | 读场景文字并结合视觉语境推理 |
| 评测指标 | VQA accuracy |

---

## 问题定义与边界

先把任务说清楚。TextVQA 的输入至少包含三部分：图像、问题、图中可见文本。这里的“可见文本”通常不是现成给你的真值字符串，而是先由 OCR 模块从图片中检测并识别出来。输出是一个答案字符串，评测时拿它去和 10 个参考答案比较。[TextVQA 数据集卡](https://huggingface.co/datasets/facebook/textvqa)

这个任务的边界非常明确：

| 项目 | 内容 |
|---|---|
| 图像来源 | OpenImages 自然场景图像 |
| 语言 | 问题为英文 |
| 文本类型 | 招牌、包装、广告、标签、路牌等场景文字 |
| 不是啥 | 不是文档理解，不是扫描件问答，不是只靠视觉物体分类 |
| 依赖项 | OCR 质量直接影响上游输入 |
| 标注形式 | 每题 10 个众包答案，测试集答案不公开 |

数据集的构建流程也解释了它为什么难。研究者先用 OCR 自动筛出“可能含文字”的图片，再让人工确认图片里确实有文本，接着让标注者专门写“必须依赖文本才能回答”的问题，最后再找 10 个不同的人回答同一问题。也就是说，题目不是随便问，而是故意设计成“不读字就答不出来”。[TextVQA 数据集卡](https://huggingface.co/datasets/facebook/textvqa)

玩具例子可以这样理解：

- 图片里有一个饮料瓶，标签写着 `Orange Juice`
- 问题是：`what flavor is the drink?`

这题不能只看颜色猜“橙子味”。如果 OCR 没把 `Orange` 读出来，模型几乎没法稳健回答。TextVQA 要测的正是这种“文本是主证据”的场景。

真实工程例子更接近无障碍助手或巡检机器人。比如用户拿手机拍一家店，问“这家店叫什么”。系统不能只识别“这是一家店”，而要把门头文字读出来，再和问题对齐，最终输出店名。这和只做 OCR 不一样，因为不是把所有文字都抄下来，而是要找出“问题在问哪一段文字”。

---

## 核心机制与推导

TextVQA 论文提出的代表性框架叫 LoRRA，完整名字是 Look, Read, Reason & Answer。白话说就是四步：

1. Look：看图，提取视觉区域特征。
2. Read：读字，用 OCR 找到文本内容和位置。
3. Reason：把图像、问题、OCR token 一起做对齐和推理。
4. Answer：从固定词表里选答案，或者直接从 OCR token 里复制答案。[LoRRA 论文](https://textvqa.org/assets/paper/TextVQA.pdf)

它的关键不是“又加了一个 OCR 模块”，而是把 OCR token 当成与视觉区域并列的一类输入。这样模型既能知道某个词是什么，也能知道它出现在图中什么位置，是否和被问到的物体、区域、关系有关。

可以把流程压缩成这个示意：

| 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Look | 图像 | 视觉特征 | 看物体、场景、位置 |
| Read | 图像局部 | OCR token + bbox | 读出字及其坐标 |
| Reason | 问题 + 视觉 + OCR | 融合表示 | 判断问题在问哪个字、哪个区域 |
| Answer | 融合表示 | 词表答案或复制的 OCR token | 输出最终答案 |

为什么评测要用软打分，而不是完全匹配就 0/1？因为同一张图、同一个问题，不同标注者可能会写出大小写不同、单复数不同、简称不同的答案。VQA accuracy 用一个更宽松的共识分数来计分。常见写法是：

$$
\mathrm{Acc}(ans)=\min\left(\frac{\#\text{humans}(ans)}{3},1\right)
$$

这里 $\#\text{humans}(ans)$ 表示 10 个标注答案里，有多少人与模型答案一致。官方实现为了和人类评估保持一致，会把 10 个答案做 “10 选 9” 的平均，本质上仍然是在估计“有多少人类认可这个答案”。[VQA accuracy 实现](https://huggingface.co/spaces/Kamichanw/vqa_accuracy/blob/main/vqa_accuracy.py)

最小数值例子：

- 参考答案里有 8 个 `simon clancy`
- 模型预测 `simon clancy`

则

$$
\min(8/3,1)=1
$$

所以得满分。反过来，如果只有 2 个标注者写了同一个答案，那么分数就是 $2/3$，不会直接变成 0。这种设计适合自然语言答案存在轻微分歧的场景。

这里还要强调一个常被初学者忽略的点：TextVQA 不是“看到文本就行”，而是“看到文本后还要推理”。例如图片里同时有 `SALE 50%`, `OPEN`, `COFFEE` 三段字，问题是“这家店卖什么”，正确线索可能是 `COFFEE`，不是最显眼也不是面积最大的那段文本。

---

## 代码实现

一个面向工程的最小管线通常长这样：先 OCR，拿到文本与位置；再编码图像、问题和 OCR token；接着做多模态融合；最后输出固定词表答案或从 OCR token 中复制答案。

```python
from collections import Counter

def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())

def vqa_accuracy(pred: str, gts: list[str]) -> float:
    pred = normalize_answer(pred)
    gts = [normalize_answer(x) for x in gts]
    scores = []
    for i in range(len(gts)):
        other_gt = gts[:i] + gts[i + 1:]
        match_count = sum(1 for x in other_gt if x == pred)
        scores.append(min(match_count / 3.0, 1.0))
    return sum(scores) / len(scores)

def answer_by_copy(question: str, ocr_tokens: list[str]) -> str:
    q = normalize_answer(question)
    token_counts = Counter(normalize_answer(t) for t in ocr_tokens)

    # 一个玩具规则：如果问题像是在问“谁/店名/品牌”，优先返回最长且重复最多的 OCR token
    ranked = sorted(
        token_counts.items(),
        key=lambda x: (x[1], len(x[0])),
        reverse=True
    )
    return ranked[0][0] if ranked else ""

def textvqa_pipeline(image_stub: str, question: str, ocr_tokens: list[str]) -> str:
    # image_stub 只是占位，真实系统会把图像送入视觉编码器
    _ = image_stub
    return answer_by_copy(question, ocr_tokens)

# 玩具例子
gts = [
    "simon clancy", "simon ciancy", "simon clancy", "simon clancy",
    "the brand is bayard", "simon clancy", "simon clancy",
    "simon clancy", "simon clancy", "simon clancy"
]
pred = textvqa_pipeline("image.jpg", "who is this copyrighted by?", [
    "bayard", "simon clancy", "copyright"
])

score = vqa_accuracy(pred, gts)
assert pred == "simon clancy"
assert score > 0.9
print(pred, score)
```

这段代码能运行，但它只是“机制示意”，不是可用模型。它表达了三个关键工程事实：

1. 预测答案需要和 OCR token 有直接连接。
2. 评测不是简单的字符串全等 0/1。
3. 真实系统里答案头往往需要“分类 + 复制”混合能力。

如果把它扩展成真实模型，通常会是这样的伪代码：

```python
def textvqa_pipeline(image, question):
    ocr_tokens, ocr_boxes = ocr.read(image)
    img_feats = vision_encoder(image)
    q_feats = text_encoder(question)
    ocr_feats = ocr_encoder(ocr_tokens, ocr_boxes)
    fused = multimodal_attention(img_feats, q_feats, ocr_feats)
    answer = answer_head(fused, ocr_tokens)  # 词表分类 or copy
    return answer
```

真实工程例子是商超货架问答。用户拍一排商品，问题是“哪一瓶是无糖的”。系统先 OCR 读出 `Zero Sugar`、`Original`、`Diet` 等字样，再结合商品位置与问题中的属性词，把答案锁定到对应文本或对象。这里如果只做图像分类，很难稳健区分近似包装；如果只做 OCR 抄全文，又不知道哪段字才和问题相关。

---

## 工程权衡与常见坑

TextVQA 的第一大坑是 OCR 误差累积。BOV++ 论文明确指出，传统 “OCR + VQA” 级联方式里，如果 OCR 把 `PEPSI` 读成 `EPSI`，后续理解和答案生成都会一起偏掉。这不是普通噪声，因为文字本身常常就是答案核心。[BOV++ 论文](https://www.sciencedirect.com/science/article/pii/S0031320323000389)

第二大坑是开放词表。普通分类模型适合预测 `yes/no/red/2` 这类高频固定答案，但 TextVQA 经常要求输出品牌名、门店名、地址、人名、车型等长尾词。只靠固定 vocabulary，很多答案训练时几乎没见过。

第三大坑是跨模态同步。OCR token 有文本内容和位置信息，视觉分支有区域特征，问题分支有语义约束。如果三者对不齐，就会出现“字读对了，但找错对象”的问题。

| 常见坑 | 结果 | 常见缓解方案 |
|---|---|---|
| OCR 识别错误 | 后续推理建立在错误文本上 | 更强 OCR、视觉增强文本表示、OCR-VQA 联动优化 |
| 固定词表过小 | 长尾答案无法输出 | copy mechanism、生成式解码、候选词约束 |
| 文本与区域未对齐 | 读到对的字但答错对象 | 引入 bbox、区域注意力、跨模态 attention |
| 标准化不统一 | 大小写/空格差异导致掉分 | 评测前做归一化 |
| 只拼接 OCR 文本 | 缺少位置关系，问题难对齐 | 保留 token 坐标和视觉局部特征 |

对初级工程师最实用的一条建议是：不要把 OCR 结果简单拼成一句长文本再喂给 LLM，然后期待它自然解决一切。这样会丢掉最重要的空间信息。TextVQA 的很多问题本质上依赖“哪一块区域里的哪一段字”。

另一条建议是：尽量给答案头保留“复制 OCR token”的能力。因为很多正确答案不应该由模型“生成”，而应该由模型“定位并复制”。LoRRA 的价值就在这里，它允许答案来自固定词表，也允许直接来自图中读到的字符串。[LoRRA 论文](https://textvqa.org/assets/paper/TextVQA.pdf)

---

## 替代方案与适用边界

TextVQA 适合自然场景文本理解，但并不覆盖所有“图像里有字”的问题。如果你的任务更偏文档、表单、票据、试卷，那么文档问答基准通常更合适；如果任务仍是自然场景但更强调场景文本识别与定位，也会考虑 ST-VQA 这类数据集。TextVQA 数据集卡也提到，很多方法会联合 TextVQA 与 ST-VQA 训练，以提升泛化能力。[TextVQA 数据集卡](https://huggingface.co/datasets/facebook/textvqa)

可以用一个选型表来理解：

| 场景 | 更适合的基准 | 关注点 | 是否强依赖布局 |
|---|---|---|---|
| 自然街景、商店、包装、路牌 | TextVQA | 场景文字 + 问答推理 | 中等 |
| 自然场景文本问答、强调文字定位 | ST-VQA | 场景文本识别与回答 | 中等 |
| 文档、表单、发票、试卷 | DocVQA 类任务 | 页面结构、版面关系、字段抽取 | 高 |

如果你有 OCR 真值标注，并且愿意把 OCR 训练一起纳入系统，BOV++ 这类 end-to-end 方案值得考虑；它的核心思想是让问答模块反过来帮助修正阅读错误。相反，如果你只是做一个能快速上线的原型，LoRRA 这种“外部 OCR + 多模态推理 + copy 答案”的结构更简单、更容易替换模块。[BOV++ 论文](https://www.sciencedirect.com/science/article/pii/S0031320323000389)

具体到真实工程：

- 做街景辅助阅读、店铺识别、包装问答：优先把 TextVQA 当作能力检查项。
- 做工业巡检且已有高质量 OCR 标注：可以考虑更强的 OCR-VQA 联动训练。
- 做文档问答：不要硬套 TextVQA，因为它的难点和文档布局理解并不相同。

---

## 参考资料

- TextVQA 数据集卡：[https://huggingface.co/datasets/facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa)
- LoRRA 论文《Towards VQA Models That Can Read》：[https://textvqa.org/assets/paper/TextVQA.pdf](https://textvqa.org/assets/paper/TextVQA.pdf)
- VQA accuracy 参考实现：[https://huggingface.co/spaces/Kamichanw/vqa_accuracy/blob/main/vqa_accuracy.py](https://huggingface.co/spaces/Kamichanw/vqa_accuracy/blob/main/vqa_accuracy.py)
- BOV++ 论文《Beyond OCR + VQA: Towards end-to-end reading and reasoning for robust and accurate textvqa》：[https://www.sciencedirect.com/science/article/pii/S0031320323000389](https://www.sciencedirect.com/science/article/pii/S0031320323000389)
