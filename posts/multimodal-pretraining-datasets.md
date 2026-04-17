## 核心结论

多模态预训练不是“把图片和文本都喂进去”这么简单，它至少包含四类不同职责的数据。

| 数据类型 | 代表数据集 | 主要模态 | 典型规模 | 主要作用 |
| --- | --- | --- | --- | --- |
| 图文对 | LAION-5B | 单张图片 + 单段文本 | 58.5 亿图文对 | 训练基础视觉-语言对齐 |
| 交错图文文档 | MMC4 | 长文本 + 多张插图 | 1.012 亿文档、约 5.71 亿图 | 训练图文交错 prompt 理解 |
| 强过滤交错文档 | OBELISC | 网页文本 + 网页图片 | 1.41 亿页面、115B token | 提高可读性、安全性与多样性 |
| 视频字幕/合成标注 | WebVid、Panda-70M、GPT-4V 标注 | 视频片段 + 字幕/描述 | 百万到千万级视频片段 | 补充时序语义与弱标注模态 |

结论可以压缩成一条训练流水线：先用 LAION-5B 这类大规模图文对教模型“图片和文字能对上”；再用 MMC4、OBELISC 这类交错图文文档教模型“图片会出现在长上下文里”；最后用 WebVid、Panda-70M 和 GPT-4V 合成标注补上视频、字幕和稀缺标注信号。

这里的 in-context 能力，白话解释就是：模型不只是看一张图回答一句话，而是能在一整段混合了图片、文字、示例和问题的上下文里继续推理。纯图文对擅长对齐，交错图文才更接近真实用户 prompt。

玩具例子是：你给模型一篇讲“咖啡制作流程”的 500 字说明文，并在中间插入“磨豆图”“萃取图”“奶泡图”，模型需要理解图片插入的位置和上下文关系。真实工程例子是：IDEFICS 一类模型会先吃大规模交错网页文档，再结合视频字幕做进一步训练，目标不是单点图像分类，而是 few-shot 图文问答和跨模态长上下文推理。

---

## 问题定义与边界

问题定义很明确：我们希望模型在同一段上下文中同时处理文字、图片，甚至进一步处理视频字幕，而不是把视觉任务和语言任务拆开单独学。

边界也必须明确，因为不是所有“公开网页数据”都能直接进入训练集。一个可用的多模态预训练数据源，至少要同时满足三件事：

| 维度 | 必要条件 | 为什么重要 |
| --- | --- | --- |
| 可用模态 | 图片、文本，必要时再加视频字幕 | 模型目标决定数据形态 |
| 可用规模 | 至少百万级，通常要上亿甚至十亿级样本 | 大模型参数量大，小数据不够支撑泛化 |
| 安全过滤 | 去重、NSFW 过滤、隐私与违法内容清理、举报响应 | 不做过滤会直接放大安全和偏见问题 |

MMC4 的意义就在这里。它不是简单收集“图 + 文”的对应关系，而是把网页中的长文本和多张图重新整理成一个文档序列。研究摘要给出的统计是：MMC4 子集包含 1.012 亿文档、5.71 亿张图、430 亿英文 token，平均每篇文档约 5.6 张图。这说明真实网页不是“一张图配一句话”，而是“长文章中穿插多张图”。如果训练时只看单张图文对，模型就很难学会处理这种交错输入。

因此本文的边界是：

| 本文讨论 | 本文不展开 |
| --- | --- |
| 公开网页图文对、交错图文、视频字幕、合成标注 | 私有商业语料采购细节 |
| 数据构建机制、过滤机制、训练用途 | 下游 benchmark 的逐项分数比较 |
| 面向预训练的数据工程 | 面向产品上线的全部合规流程 |

一句话说，本文关注的是“数据如何被整理成可训练的多模态语料”，不是“模型结构怎么设计”。

---

## 核心机制与推导

核心机制分两层。第一层是对齐，第二层是交错。

对齐，白话解释就是：判断哪张图最像哪段话说的内容。LAION-5B 主要解决这一层，它依赖类似 CLIP 的相似度过滤，把公开网页里的海量噪声图文对筛出一部分能用样本。

交错，白话解释就是：不是只找一张最匹配的图，而是要在整篇长文里决定“哪张图插在哪一句附近最合适”。MMC4 用的是线性指派思想，可以写成：

$$
\max_{\pi}\sum_i \text{sim}_{\text{CLIP}}(I_i, S_{\pi(i)})
$$

其中 $I_i$ 表示第 $i$ 张图片，$S_{\pi(i)}$ 表示被分配到的句子或文本片段，$\text{sim}_{\text{CLIP}}$ 是图文相似度。线性指派，白话解释就是：在不重复占用位置的前提下，为多张图找一组总分最高的插入位置。

玩具例子可以这样理解。假设一篇 500 字说明文被切成 6 个片段，每隔约 80 字一个片段；同时有 3 张图，分别是“磨豆机”“手冲壶”“咖啡成品”。做法不是把每张图都贴到文末，而是：

1. 先算每张图和每个片段的相似度。
2. 再求一个总分最大的分配方案。
3. 把图插到对应片段后面。
4. 最后检查是否有不安全内容、重复图、尺寸太小的图。

如果相似度矩阵是

$$
M=
\begin{bmatrix}
0.91 & 0.42 & 0.15 \\
0.30 & 0.88 & 0.27 \\
0.18 & 0.36 & 0.93
\end{bmatrix}
$$

那么最优分配显然是第一张图配第一段、第二张图配第二段、第三张图配第三段，总分 $0.91+0.88+0.93=2.72$。真实网页会更复杂，但机制相同。

OBELISC 在 MMC4 的基础上又加了一层过滤与排序。它不只看相似度，还会综合图片尺寸、重复度、NSFW 风险、文档可读性等信号，抽象成一个打分函数：

$$
\text{score}=w_1\cdot \text{sim}+w_2\cdot \text{filter}+w_3\cdot \text{diversity}
$$

这里的 diversity，白话解释就是：别让一篇文档里塞进很多内容几乎一样的图。filter 则可以包含 NSFW、分辨率、是否重复、文本困惑度等规则。困惑度 perplexity 可以粗略理解成“这段文本像不像正常语言”，太高通常意味着乱码、模板页或抓取失败。

可以把整个流程理解成一个文本版流程图：

`网页抓取 -> 文本分段 -> 图片收集 -> CLIP 相似度计算 -> 线性指派 -> 安全/尺寸/重复过滤 -> 文档级可读性过滤 -> 生成交错图文文档`

真实工程例子是 IDEFICS 这类模型的训练路线：先用 OBELISC 提供的大规模交错文档训练模型适应长上下文里的图片，再结合 WebVid、Panda-70M 一类视频字幕数据补上时序描述能力。视频字幕的价值在于，它给了“画面变化 + 时间顺序 + 文字说明”的联合信号，这和静态图片完全不同。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖真实 CLIP，只用手工构造的相似度矩阵和过滤信号，目的是把“打分、分配、过滤、交错输出”这个流程讲清楚。

```python
from itertools import permutations

def linear_assignment(score_matrix):
    n_images = len(score_matrix)
    n_slots = len(score_matrix[0])
    best_score = float("-inf")
    best_perm = None

    for perm in permutations(range(n_slots), n_images):
        score = sum(score_matrix[i][perm[i]] for i in range(n_images))
        if score > best_score:
            best_score = score
            best_perm = perm

    return list(enumerate(best_perm)), best_score

def apply_filters(pairs, nsfw_scores, duplicate_flags, min_safe_score=0.8):
    filtered = []
    for image_idx, slot_idx in pairs:
        if nsfw_scores[image_idx] > 0.2:
            continue
        if duplicate_flags[image_idx]:
            continue
        filtered.append((image_idx, slot_idx))
    return filtered

def merge_text_and_images(text_chunks, filtered_pairs, image_tokens):
    slot_to_image = {slot: image_tokens[img] for img, slot in filtered_pairs}
    output = []
    for slot, chunk in enumerate(text_chunks):
        output.append(chunk)
        if slot in slot_to_image:
            output.append(slot_to_image[slot])
    return "\n".join(output)

text_chunks = [
    "咖啡豆先磨成粉，粒径决定萃取速度。",
    "热水以稳定流速注入，保证粉层均匀浸润。",
    "萃取结束后观察液体颜色和流速变化。",
]

image_tokens = ["[IMG:grinder]", "[IMG:kettle]", "[IMG:coffee]"]

score_matrix = [
    [0.91, 0.20, 0.05],
    [0.15, 0.88, 0.25],
    [0.08, 0.31, 0.93],
]

pairs, best_score = linear_assignment(score_matrix)
filtered = apply_filters(
    pairs,
    nsfw_scores=[0.01, 0.03, 0.02],
    duplicate_flags=[False, False, False]
)
merged = merge_text_and_images(text_chunks, filtered, image_tokens)

assert round(best_score, 2) == 2.72
assert "[IMG:grinder]" in merged
assert "[IMG:kettle]" in merged
assert "[IMG:coffee]" in merged
assert merged.count("[IMG:") == 3

print(merged)
```

如果换成训练数据管道，结构通常更像下面这样：

```python
for doc in corpus:
    text = load_text(doc["text_path"])
    images = load_images(doc["image_paths"])
    gpt4v_labels = load_optional_labels(doc.get("synthetic_caption_paths", []))

    text_chunks = split_into_chunks(text, chunk_size=80)
    text_embeds = clip_encode_text(text_chunks)
    image_embeds = clip_encode_images(images)

    sim = cosine_similarity(image_embeds, text_embeds)
    best_pairs = linear_assignment(sim)

    filtered = apply_filters(
        best_pairs,
        nsfw_scores=score_nsfw(images),
        duplicate_flags=detect_duplicates(images),
        perplexity=score_text_readability(text),
        min_resolution=(256, 256),
    )

    interleaved = merge(text_chunks, filtered, gpt4v_labels=gpt4v_labels)
    write_training_sample(interleaved)
```

这里可以顺带给一个接近 LAION 风格的索引样板。真正工程里不会把图片字节塞进一条 JSON，而是保存 URL、对象存储路径、哈希和过滤标志：

| 字段 | 含义 |
| --- | --- |
| `image_uri` | 图片 URL 或存储路径 |
| `text` | 抓取到的描述、alt 文本或上下文文本 |
| `clip_score` | 图文相似度，用于初筛 |
| `nsfw_score` | 风险分数 |
| `phash` | 感知哈希，用于去重 |
| `source_url` | 来源网页，便于举报和回溯 |

真实工程例子里，MMC4/OBELISC 的处理不会逐文档单机运行，而是分布式流水线：先批量抽 embedding，再统一做匹配和过滤，最后写成训练框架能顺序读取的 shard 文件。

---

## 工程权衡与常见坑

最大的误区是把“规模大”误认为“可直接训练”。LAION-5B 的价值是规模，但它的问题也正来自规模。公开网页数据未经逐条人工审核，可能包含违法内容、隐私泄漏、偏见分布严重失衡、误标注和高重复样本。规模解决覆盖率，不解决质量。

| 风险 | 具体表现 | 常见规避措施 |
| --- | --- | --- |
| 违法/有害内容 | CSAM、极端暴力、仇恨内容 | 哈希黑名单、第三方举报通道、多轮重清洗 |
| 隐私泄漏 | 人脸、证件、私人照片、定位信息 | 来源回溯、敏感实体识别、删除请求响应 |
| 重复样本 | 同图多次抓取、模板站镜像 | pHash/近重复检索、diversity 权重 |
| 偏见失衡 | 年龄、性别、种族分布倾斜 | 重采样、分层审计、下游安全评测 |
| 文档不可读 | 网页碎片、乱码、广告模板 | perplexity 过滤、规则清洗、长度约束 |
| 交错错误 | 图插错段落导致上下文污染 | 线性指派 + 人工抽检 + 阈值过滤 |

一个未经过滤的 LAION 风格样本，可能长这样：文本是“young beautiful girl in school uniform”，图片却来自不明站点、人物年龄不清晰、同一图在多个镜像站重复出现。这种样本的问题不是“描述太短”，而是可能同时触发未成年人风险、性别偏见和重复采样。

加上 OBELISC 风格过滤后，系统会先做相似度和图文定位，再用 NSFW 分数、尺寸阈值、重复检测、文本可读性过滤把高风险和低质量样本淘掉。注意，这不是保证完全安全，而是把风险从“原始网页级别”压到“可控训练级别”。

另一个常见坑是把交错图文做成“视觉噪声”。如果一篇文档里图插得太密、位置不对、图片内容高度重复，few-shot 推理效果会退化，因为模型读到的是混乱上下文。交错数据的关键不是“有图”，而是“图出现得合理”。

GPT-4V 一类视觉模型参与合成标注时，也有两层风险。第一层是它会幻觉，白话解释就是看图后给出并不存在的细节；第二层是它会继承自身安全边界，导致某些图过度拒答或漏报。因此更合理的用法不是“全量自动标注直接训练”，而是“先生成候选描述，再做人审抽检与回流修正”。

---

## 替代方案与适用边界

不是所有多模态任务都必须上 MMC4 或 OBELISC。要先看目标是什么。

| 目标 | 更合适的数据 | 是否必须交错 |
| --- | --- | --- |
| 图像检索、图像标注 | LAION、CC3M、CC12M 一类图文对 | 否 |
| 图文问答、few-shot 视觉提示学习 | MMC4、OBELISC | 是，通常需要 |
| 视频字幕理解、时序事件描述 | WebVid、Panda-70M | 需要时间序列，不一定是图文交错 |
| 稀缺领域标注补齐 | GPT-4V 合成标注 + 人工校验 | 取决于下游任务 |

可以直接对比两个场景。

场景 A：目标是做 CLIP 风格图像检索。你关心的是“这张图和哪段文本最相似”。这时大规模图文对已经足够，训练重点是全局对齐，不必专门构造长文中插图的序列。

场景 B：目标是做 few-shot 图文问答。用户可能发来“先看这两张示例图，再看第三张图，按前两个例子的格式回答”。这时如果训练集从没出现过图文交错上下文，模型即使会看图，也未必会在长提示里稳定执行模式迁移。MMC4、OBELISC 的价值就在这里。

替代方案也有代价。你可以用商业闭源数据服务、自建爬虫和标注系统替代 LAION，但前提是你有采集权限、审查能力、存储预算和删除响应机制。你也可以只用小规模高质量人工数据，但通常只适合微调，不足以撑起大模型预训练。

所以适用边界很清楚：

1. 只做基础对齐，用图文对即可。
2. 要做交错 prompt 推理，必须引入交错文档。
3. 要做视频理解，需要字幕或 ASR 语料。
4. 缺少标注时，可以用 GPT-4V 生成弱标签，但不能跳过审核闭环。

---

## 参考资料

1. LAION-5B / Re-LAION 相关说明  
https://openreview.net/forum?id=M3Y74vmsMcY&isApp=1  
用于了解公开网页图文对的规模、清洗思路与争议。

2. MMC4 与 OBELISC 机制论文摘要页  
https://proceedings.neurips.cc/paper_files/paper/2023/hash/1c6bed78d3813886d3d72595dbecb80b-Abstract-Datasets_and_Benchmarks.html  
用于先看数据集目标、主要统计与交错图文构建思路。

3. MMC4 / OBELISC 论文 PDF  
https://proceedings.neurips.cc/paper_files/paper/2023/file/1c6bed78d3813886d3d72595dbecb80b-Paper-Datasets_and_Benchmarks.pdf  
用于看线性指派、过滤规则、规模细节与实验。

4. IDEFICS 相关发布材料  
https://openreview.net/forum?id=SKN2hflBIZ&noteId=dDhzqGrzJk  
用于理解交错文档如何进入真实大模型训练流程。

5. WebVid 数据说明  
https://www.tensorflow.org/datasets/catalog/webvid  
用于理解视频字幕数据的基本形式和读取方式。

6. 关于公开数据安全争议的报道  
https://www.theguardian.com/technology/2023/dec/20/ai-image-generators-child-sexual-abuse  
用于理解为何多轮过滤、举报与回溯机制是工程必需项。

建议阅读顺序是：先看 LAION 的规模与争议，再看 MMC4/OBELISC 的交错构建机制，最后看 WebVid 与 IDEFICS 的工程落地案例，这样最容易建立完整的数据流水线视角。
