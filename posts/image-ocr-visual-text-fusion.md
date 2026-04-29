## 核心结论

图像 OCR 与视觉文本融合，指的是把图中识别到的文字、文字所在位置、以及对应区域的视觉特征放到同一个模型里联合推理。白话说，模型不只要“看见这是一张图”，还要“读到图里的字，并理解这些字和版面的关系”。

它适合三类任务：文档理解、图表问答、场景文本问答。共同点是：关键信息不只存在于像素形状里，还存在于文字内容里。纯视觉模型能看出“这里有一块像标题的区域”，但未必知道标题写的是“发票号码”；纯 OCR 能读出字符串，但未必知道哪一段文字对应金额、哪一段对应商户名。融合模型的价值就在这里。

玩具例子：一张收据里有两行文字，`Subtotal 10.00` 和 `Total 12.00`。如果只做 OCR，系统拿到的是一串词；如果只看图像，系统看到的是几行深浅不同的文本区域；融合后，模型可以结合 `Total` 这个词、它右侧的 `12.00`、以及它位于底部汇总区这一布局特征，回答“总金额是 12.00”。

| 方法 | 能否读字 | 能否利用位置 | 典型短板 |
|---|---|---|---|
| 纯视觉模型 | 弱 | 中 | 看得见区域，看不准具体文本 |
| 纯 OCR 管线 | 强 | 弱到中 | 字能读出来，但语义关系和版式理解不足 |
| OCR + 视觉融合 | 强 | 强 | 依赖 OCR 质量，系统更复杂 |

结论可以压缩成一句话：这类模型真正解决的问题，不是“把字识别出来”，而是“让文字内容、空间位置、视觉上下文共同参与决策”。

---

## 问题定义与边界

这类问题的输入通常不是单一图像，而是多种信号的组合：

| 项目 | 内容 |
|---|---|
| 输入 | 图像 $I$、OCR token $x_i$、边界框 $b_i$、局部视觉特征 $c_i$、问题或指令 $q$ |
| 输出 | 答案文本、字段值、分类标签、结构化 JSON |
| 典型任务 | VQA、票据理解、表单抽取、图表问答、场景文本问答 |
| 不适合的问题 | 纯图像分类、完全无文字目标识别、只关心整体风格的任务 |

这里有两个边界要讲清楚。

第一，OCR 只是输入来源之一，不等于最终答案。比如问题是“这张发票的税号是多少”，系统不是把整张发票全文复述一遍，而是从 OCR 结果里找到与“税号”最相关的局部，再输出目标字段。

第二，融合模型不等于 OCR 模型。OCR 模型负责“把字读出来”；融合模型负责“把字和图像上下文联系起来”。如果收据上同时出现 `Total`、`Tax`、`Subtotal`，OCR 都能读到，但哪个数字才是最终要的答案，通常需要布局理解和上下文推理。

真实工程例子：报销系统抽取发票字段。输入是一张手机拍摄的发票图像，外加 OCR 输出的文字框；输出不是整页文本，而是结构化结果：
`{"invoice_no": "...", "seller": "...", "total_amount": "..."}`。这说明任务目标通常是“定向抽取”，不是“全文转写”。

---

## 核心机制与推导

核心思路是统一编码。编码，白话说，就是把原始输入转成模型能计算的向量。对第 $i$ 个 OCR token，可以写成：

$$
h_i = E_{txt}(x_i) + E_{box}(b_i) + E_{vis}(c_i)
$$

含义如下：

| 符号 | 含义 |
|---|---|
| $x_i$ | 第 $i$ 个 OCR token |
| $b_i$ | 该 token 的边界框坐标 |
| $c_i$ | token 对应区域的局部视觉特征 |
| $q$ | 问题文本 |
| $p_j$ | 图像 patch 特征 |
| $\alpha_i$ | 模型对第 $i$ 个 token 的注意力权重 |
| $\lambda$ | 生成分支与复制分支的混合系数 |

第一步是文本编码。`Nike`、`Total`、`12.00` 这些 token 会先变成文本向量，表示字面语义。

第二步是位置编码。位置编码，白话说，就是把“字在页面哪里”也变成向量。因为左上角和右下角的同一个词，在文档里可能代表完全不同的角色。

第三步是视觉编码。视觉特征不只是整图特征，还包括局部外观，例如 logo、字体粗细、表格线、底色区域。这样模型知道某段文字不仅写了什么，还“长什么样、在什么区域里”。

然后把问题、OCR token、图像 patch 一起喂给 Transformer：

$$
z = Transformer([q; h_1, ..., h_n; p_1, ..., p_m])
$$

Transformer 可以理解成一种全局关系建模器，它允许“问题中的词”和“图中各个文字块、图像块”相互看见对方。比如问题是“品牌名是什么”，模型会对可能的品牌词分配更高注意力：

$$
\alpha_i = softmax\left(\frac{(W_q z_q)(W_t h_i)^T}{\sqrt{d}}\right)
$$

再把高权重 token 聚合成融合表示：

$$
h_{fuse} = \sum_i \alpha_i h_i
$$

输出阶段常有两条路。`generate` 分支从固定词表生成答案，适合输出常见词；`copy` 分支直接从 OCR token 里复制，适合金额、编号、人名、店名这类开放词。混合形式可写成：

$$
p(a|I,q) = \lambda \cdot p_{vocab}(a|z) + (1-\lambda)\cdot \sum_{i:x_i=a}\alpha_i
$$

这条公式很重要。它说明答案概率由两部分组成：一部分来自词表生成，一部分来自 OCR token 复制。也正因为有复制分支，OCR 错误会被直接放大。

玩具例子：问题是“总金额是多少”，OCR 候选有 `12.00` 和 `15.00`，注意力分别是 $0.86$ 与 $0.14$，词表分支对 `12.00` 的概率是 $0.08$，设 $\lambda=0.25$，则：

$$
p(12.00)=0.25\times0.08+0.75\times0.86=0.665
$$

如果 OCR 把 `12.00` 误成 `17.00`，复制分支就会把错误 token 推成高概率答案。后面的推理模块越依赖 copy，错误越难纠正。

再补一个机制：word-patch alignment。它的作用是让文字 token 与对应图像区域对齐。白话说，模型不只知道“有个词叫 Total”，还知道“这个词就在这块浅灰色汇总栏里”。在票据、表单、图表这类布局敏感任务中，这个对齐通常很关键。

---

## 代码实现

工程实现一般拆成四层：OCR 读取、特征编码、跨模态融合、答案解码。下面给一个最小可运行示例，用简化版逻辑演示“复制分支会放大 OCR 误差”这一点。

```python
from math import exp

def softmax(xs):
    exps = [exp(x) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

def decode_with_copy_head(tokens, attn_logits, vocab_probs, lam=0.25):
    attn = softmax(attn_logits)
    scores = {}

    for token, weight in zip(tokens, attn):
        scores[token] = scores.get(token, 0.0) + (1 - lam) * weight

    for token, p in vocab_probs.items():
        scores[token] = scores.get(token, 0.0) + lam * p

    best = max(scores, key=scores.get)
    return best, scores

tokens = ["12.00", "15.00"]
attn_logits = [3.0, 1.0]
vocab_probs = {"12.00": 0.08, "15.00": 0.02}

answer, scores = decode_with_copy_head(tokens, attn_logits, vocab_probs, lam=0.25)
assert answer == "12.00"
assert scores["12.00"] > scores["15.00"]

wrong_tokens = ["17.00", "15.00"]  # OCR 把 12.00 误读成 17.00
wrong_answer, wrong_scores = decode_with_copy_head(wrong_tokens, attn_logits, vocab_probs, lam=0.25)
assert wrong_answer == "17.00"
assert wrong_scores["17.00"] > wrong_scores["15.00"]

print(answer, scores)
print(wrong_answer, wrong_scores)
```

这个例子没有真的跑视觉模型，但足够说明结构问题：当高注意力位置上的 OCR token 错了，复制分支会把错误答案直接推上去。

对应到完整系统，伪代码通常是：

```python
image = load_image(path)
ocr_tokens, boxes, scores = run_ocr(image)

ocr_tokens, boxes, scores = filter_and_sort(ocr_tokens, boxes, scores)
text_feat = text_encoder(ocr_tokens)
box_feat = box_encoder(boxes)
vis_feat = visual_encoder(image, boxes)

token_feat = text_feat + box_feat + vis_feat
question_feat = text_encoder(question)

fused = transformer([question_feat] + list(token_feat))
answer = decode_with_copy_head(fused, ocr_tokens)
```

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| OCR | 图像 | tokens, boxes, scores | 读字 |
| Text Encoder | tokens | 文本向量 | 表示文字语义 |
| Box Encoder | boxes | 位置向量 | 表示版面结构 |
| Visual Encoder | 图像或局部裁剪 | 视觉向量 | 表示外观信息 |
| Transformer | 多模态序列 | 融合表示 | 建模全局关系 |
| Copy / Generate Head | 融合表示 | 最终答案 | 输出字段或答案 |

真实工程例子：发票字段抽取系统通常不会直接信任 OCR 原始顺序，而是先按页、列、行、框中心点排序，再做低置信过滤、重复框合并、金额正则校验。原因很简单，模型输入质量本身就是最终效果的一部分。

---

## 工程权衡与常见坑

这类系统最大的风险往往不是网络结构，而是输入脏数据。准确率问题通常要拆成四段排查：OCR、布局、融合、解码。

| 问题 | 表现 | 规避方式 |
|---|---|---|
| OCR 错字、漏字 | 复制答案直接错误 | 领域 OCR、置信度过滤、词表校正 |
| 阅读顺序错 | 字段拼接混乱 | 2D 坐标、行列排序、版面规则 |
| 重复框、噪声 token | 注意力被干扰 | 去重、NMS、低分框剔除 |
| 过度依赖 OCR | OCR 错则整体崩 | 保留视觉校验分支 |
| 只看 exact match | 合理答案被误判 | 增加 ANLS、编辑距离指标 |

常见坑一：把 OCR 当真值。OCR 置信度低时，最好不要让低分 token 与高分 token 同权进入融合层。特别是金额、日期、编号这类字段，字符级错误会直接传导到下游审核、风控、对账。

常见坑二：忽视阅读顺序。OCR 引擎输出顺序未必等于人类阅读顺序。表格、多栏文档、旋转票据尤其容易出问题。顺序一乱，`税额` 可能和下一行金额拼在一起。

常见坑三：评估指标过粗。对于场景文本问答，只看完全匹配会把很多“几乎正确”的结果计成 0 分。工程上更关心字符级差异是否影响业务，比如 `2023-01-08` 和 `2023-01-06` 都是错，但前者只是局部字符错误，后者可能是严重业务错误，处理策略不同。

错误传播链可以概括成：OCR 识别错误 $\rightarrow$ token 进入融合层 $\rightarrow$ copy 分支放大错误 $\rightarrow$ 最终答案错误 $\rightarrow$ 下游系统继续放大损失。

---

## 替代方案与适用边界

OCR 融合模型不是唯一方案，它主要适合“文字很重要，而且需要定位”的任务。

| 方法 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| OCR + 融合模型 | 可解释、适合字段抽取 | 依赖 OCR 质量 | 发票、收据、表单、文本问答 |
| OCR-free 模型 | 端到端、流程简单 | 可解释性弱，长文本细节不一定稳 | 摘要、页面转写、生成任务 |
| 纯视觉模型 | 系统简单 | 不擅长精确读字 | 无文字或文字不重要任务 |
| 规则 + OCR | 可控、易审计 | 泛化差 | 结构固定业务流程 |

适用边界可以直接记三条。

第一，当目标是“看懂文字和版式”，优先选 OCR 融合模型。比如表单抽取、票据识别、图表问答。

第二，当目标是“生成整页内容”而不是逐字段定位，可以考虑 OCR-free 路线。OCR-free，白话说，就是不显式依赖外部 OCR，把整张图直接映射到输出序列。它流程更短，但在强约束字段抽取里未必更稳。

第三，当业务规则非常强、模板变化很小，规则 + OCR 仍然可能是更划算方案。比如固定版式快递单、固定格式银行回单，这时复杂多模态模型不一定带来更好的成本收益比。

---

## 参考资料

1. [Towards VQA Models That Can Read (TextVQA)](https://textvqa.org/assets/paper/TextVQA.pdf)
2. [Scene Text Visual Question Answering (ST-VQA)](https://openaccess.thecvf.com/content_ICCV_2019/html/Biten_Scene_Text_Visual_Question_Answering_ICCV_2019_paper.html)
3. [Iterative Answer Prediction With Pointer-Augmented Multimodal Transformers for TextVQA](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_Iterative_Answer_Prediction_With_Pointer-Augmented_Multimodal_Transformers_for_TextVQA_CVPR_2020_paper.html)
4. [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://www.microsoft.com/en-us/research/publication/layoutlmv3-pre-training-for-document-ai-with-unified-text-and-image-masking/)
5. [Donut: OCR-free Document Understanding Transformer](https://github.com/clovaai/donut)
