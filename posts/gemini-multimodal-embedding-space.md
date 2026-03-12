## 核心结论

Gemini Embedding 2 的关键价值，不是“又多了一个 embedding 模型”，而是把文本、图像、音频、视频和 PDF 直接投到同一个向量空间。向量空间可以白话理解为“用一串数字表示内容语义的位置坐标”。只要空间足够统一，文本查询就能直接命中图片、视频片段或文档页，而不必为每种模态维护一套独立检索链。

这件事的难点不在“能不能输出向量”，而在“不同模态的向量是不是说同一种语义语言”。如果一张“猫趴在键盘上”的图片和一句“办公桌上的猫”在空间里仍然离得很远，那么统一索引只是表面统一，跨模态检索质量依然会差。工程上真正要盯的是三类指标：同义跨模态相似度、不同模态中心之间的 gap、以及不同层表示里模态特异性是否被过早抹平。

Google 在 2026-03-10 发布 Gemini Embedding 2，并给出几个明确边界：文本最多 8192 tokens，图像最多 6 张，视频最多 120 秒，PDF 最多 6 页，并支持混合输入；同时继续使用 Matryoshka Representation Learning。Matryoshka Representation Learning 可以白话理解为“把重要语义嵌在向量前缀里”，因此 3072 维向量可以裁成 1536 或 768 维继续用，只是精度会逐步下降。

| 模态 | 单次输入限制 | 工程含义 |
|---|---:|---|
| 文本 | 8192 tokens | 可直接嵌入较长段落或 chunk |
| 图像 | 6 张 PNG/JPEG | 适合商品图组、截图组、流程图组 |
| 视频 | 120 秒 MP4/MOV | 更适合短片段检索，不适合整场长会 |
| 音频 | 原生嵌入 | 可绕过先转录再检索的一部分链路 |
| 文档 | 6 页 PDF | 适合短报告、说明书片段，不适合整本手册 |

---

## 问题定义与边界

这里的问题不是“Gemini 会不会看图听音”，而是“统一 embedding 空间到底该怎么分析”。embedding 空间指模型输出向量后形成的几何空间；分析它，本质上是在分析语义是否真的对齐。

最小目标可以写成一句话：语义相同但模态不同的输入，应该在空间里尽量靠近；语义不同的输入，应该尽量分开。最常见的度量是余弦相似度和均方误差：

$$
\text{Cosine}(\mathbf{x}, \mathbf{y})=\frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|},\qquad
\text{MSE}(\mathbf{x}, \mathbf{y})=\frac{1}{d}\|\mathbf{x}-\mathbf{y}\|_2^2
$$

余弦相似度高，说明方向接近；方向可以白话理解为“语义大体一致”。MSE 低，说明坐标整体差距小。对跨模态检索来说，余弦通常更常用，因为检索更关心方向上的语义相近，而不是绝对数值大小。

“模态 gap”指不同模态整体分布之间的系统性偏移。一个简单定义是：

$$
\Delta_{\text{gap}}=\|\mu_{\text{text}}-\mu_{\text{image}}\|_2
$$

其中 $\mu_{\text{text}}$ 和 $\mu_{\text{image}}$ 分别是文本向量和图像向量的中心。若这个距离很大，说明即使两个内容主题相同，模型也可能先按模态分群，再按语义分群。这样会导致文本搜文本效果很好，但文本搜图片、搜视频明显变差。

边界也要说清楚。Gemini Embedding 2 统一的是“输出空间”，不是保证所有任务都同样强。它适合多模态检索、聚类、RAG 召回、媒体去重和跨模态相似搜索；它不自动解决细粒度时序对齐、因果推理、视频长程事件理解，也不保证在高风险领域中视觉证据一定压过语言先验。

玩具例子：有三条内容，文本“红色运动鞋”、商品图“红色跑鞋照片”、文本“蓝牙耳机”。理想情况是前两者距离近，第三条远。如果最终聚类先把“文本”聚成一堆，把“图像”聚成一堆，那说明空间首先学到的是模态，不是语义。

---

## 核心机制与推导

统一空间通常不是靠一个损失就能学出来，而是靠“共享语义”和“保留模态特征”两件事同时成立。Han 等人在 UniReps 的分析里指出，多模态模型不同层的表示作用不同：低层更偏模态特征，高层更偏任务语义。模态特征可以白话理解为“图像里的纹理、音频里的频谱、文本里的词形这些各自独有的信号”。

因此一个合理的目标不是把所有层都强行拉齐，而是让高层更对齐、低层保留足够专有信息。可以写成抽象形式：

$$
\mathcal{L}_{\text{total}}
=
\lambda_{\text{align}}\mathcal{L}_{\text{align}}
+
(1-\lambda_{\text{align}})\mathcal{L}_{\text{specific}}
$$

其中 $\mathcal{L}_{\text{align}}$ 负责把同义图文、图音、文档页与查询拉近；$\mathcal{L}_{\text{specific}}$ 负责保住各模态自身有效信号。若只优化对齐，图像可能失去局部视觉结构；若只保留模态特征，跨模态检索又会退化。

可以把它理解成“两段式几何”：

1. 低层先把原始模态压缩成各自稳定表征。
2. 高层再把这些表征映射到共享语义子空间。

Matryoshka Representation Learning 在这里的作用，是让高质量语义尽量集中在前缀维度中。前缀维度可以白话理解为“向量前半段先装最重要的信息”。这样同一个 3072 维向量，在召回阶段可只取前 768 维做粗搜，在重排阶段再用更高维做精排。它本质上是用维度换成本，而不是免费提升质量。

还要注意一个常被忽略的问题：统一空间不等于每个模态的分布天然均衡。文本样本通常量更大、标注更规范，所以训练后常出现“文本主导”。文本主导可以白话理解为“空间几何更像为文本优化出来的，其他模态被迫往文本靠”。结果是图像检索看似可用，但对图片里的细粒度视觉差异不敏感；音频则可能被压成偏语义摘要，而不是保留声学差异。

真实工程例子：做企业知识库时，上传一张设备照片、一段 20 秒故障录音、两页维修 PDF，再用一句“电机启动后有金属摩擦声”去检索。如果空间对齐良好，系统应该同时命中录音、PDF 中的故障描述页和对应设备图片；如果 gap 明显，结果往往只会优先返回文本页，而忽略音频与图像证据。

---

## 代码实现

下面先给一个可运行的玩具分析代码，不依赖外部 API。它模拟文本向量和图像向量，并计算跨模态相似度与模态 gap。这里的目的不是复现 Gemini，而是把“统一空间分析”变成能验证的最小实验。

```python
import math

def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))

def cosine(a, b):
    na = l2_norm(a)
    nb = l2_norm(b)
    assert na > 0 and nb > 0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def mean_vec(vectors):
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

def l2_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# 玩具例子：同义图文应该接近，异义图文应该更远
text_red_shoe = [0.90, 0.80, 0.10]
image_red_shoe = [0.88, 0.82, 0.12]
text_earphone = [0.10, 0.20, 0.95]
image_cat = [0.15, 0.10, 0.85]

pos_sim = cosine(text_red_shoe, image_red_shoe)
neg_sim = cosine(text_red_shoe, image_cat)

assert pos_sim > neg_sim
assert round(pos_sim, 3) > 0.99

text_vectors = [text_red_shoe, text_earphone]
image_vectors = [image_red_shoe, image_cat]

gap = l2_distance(mean_vec(text_vectors), mean_vec(image_vectors))

# 这个玩具数据里 gap 不应过大，否则说明先按模态分开了
assert gap < 0.2

print("positive cosine:", round(pos_sim, 4))
print("negative cosine:", round(neg_sim, 4))
print("modality gap:", round(gap, 4))
```

如果要接入真实服务，工程模式通常是“两层索引”或“一层统一索引”。Gemini Embedding 2 的优势在于后一种可以更自然：查询、图像、视频、PDF 都进同一个向量库。实际使用时应至少做三件事。

1. 区分 `RETRIEVAL_QUERY` 和 `RETRIEVAL_DOCUMENT` 之类的任务类型。
2. 统一做向量归一化，否则不同批次分布可能影响相似度阈值。
3. 对长视频、长 PDF 做切片，不要把超长内容当成单一向量。

示意代码如下，重点看流程，不依赖具体 SDK 细节完全一致：

```python
from google import genai

client = genai.Client()

response = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=[
        "电机启动后有金属摩擦声",
        {"mime_type": "audio/wav", "uri": "gs://demo/fault.wav"},
        {"mime_type": "application/pdf", "uri": "gs://demo/manual.pdf"},
    ],
    config={
        "task_type": "RETRIEVAL_QUERY",
        "output_dimension": 1536,
    },
)

embedding = response.embeddings[0]
```

真实工程里更稳妥的做法通常是：召回库内容使用 `RETRIEVAL_DOCUMENT` 嵌入；用户请求使用 `RETRIEVAL_QUERY` 嵌入；统一归一化后再做余弦检索。这样空间几何会更适合“短查询找长内容”。

---

## 工程权衡与常见坑

统一空间带来的是管道简化，但代价是你要更仔细地做分布监控。最常见的坑有三类。

| 常见坑 | 现象 | 原因 | 处理方式 |
|---|---|---|---|
| 模态主导 | 文本检索好，图像/音频召回差 | 训练或数据分布偏文本 | 分模态评测，补多模态正样本 |
| 中心漂移 | 不同模态各自成团 | 模态 gap 大 | 做中心距离监控，必要时分模态校准 |
| 切片错误 | 长视频、长 PDF 检索不准 | 单向量承载信息过多 | 时间切片、页级切片、层级召回 |

第一，别只看整体 recall。一个系统可能总体 recall@10 很高，但文本查文本贡献了大部分分数，文本查图像、文本查视频非常差。多模态系统必须拆开评测：text->image、text->video、image->pdf、audio->text 至少要分别算。

第二，统一索引不代表阈值统一。不同模态对的相似度分布常常不同，例如 text->text 的正样本余弦可能集中在 0.82 到 0.93，而 text->image 可能只有 0.62 到 0.78。若硬设一个统一阈值，跨模态结果会被误杀。

第三，预览模型有能力边界。Gemini Embedding 2 虽然支持混合模态输入，但不意味着“输入越多越好”。在检索场景里，过多无关模态会把查询语义拉偏。例如用户只是搜“发票抬头变更流程”，你额外塞入一张公司前台照片，通常没有帮助，反而增加噪声。

第四，别把 embedding 当概率。余弦高只表示相似，不表示事实为真。医疗、金融、合规等场景里，多模态 embedding 只能做召回和排序前置，不能单独作为结论依据。

---

## 替代方案与适用边界

如果任务只是图文搜索，经典方案仍然有效，比如 CLIP 一类图文对比模型，再配一个文本 embedding 模型。这类方案成熟、生态稳定，但问题也明确：你往往要维护多个空间，再靠 reranker 去桥接。

| 方案 | 适合场景 | 优势 | 局限 |
|---|---|---|---|
| CLIP + 文本 embedding | 图文搜索为主 | 成熟、资料多 | 多索引，扩到音频/PDF 更复杂 |
| 独立模态模型 + reranker | 存量系统改造 | 可渐进接入 | 链路长，调参多，延迟高 |
| Gemini Embedding 2 | 原生多模态检索 | 单空间、混合输入自然 | 仍在 Preview，需验证稳定性与成本 |

什么时候不该优先用统一多模态空间？

一是任务高度依赖细粒度视觉定位，例如医学影像病灶定位、工业缺陷像素级检测。这些任务需要的不只是“语义像不像”，而是“局部区域在哪里”。embedding 不是最合适的核心工具。

二是内容极长且结构复杂，例如 200 页手册、2 小时培训视频。此时更合理的方式通常是先分段、分页、分镜，再做层级索引，而不是把整份多媒体内容压成一个向量。

三是需要强可解释性。统一空间适合召回，不擅长解释“为什么这张图和这句话接近”。如果业务要求每个结果都能给出清晰证据链，就要增加页级、段级、帧级追踪，而不是只返回一个相似度分数。

---

## 参考资料

- Google 官方博客《Gemini Embedding 2: Our first natively multimodal embedding model》，2026-03-10。  
- Google AI for Developers 文档《Embeddings》。  
- Han et al., *UniReps: Towards Crisper Representations by Hierarchical Alignment of Multimodal LLMs*, PMLR 2024。  
- Google Cloud Vertex AI 文档《Get multimodal embeddings》。  
- 关于 Matryoshka Representation Learning 的相关公开资料与 Google embedding 模型说明。
