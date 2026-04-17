## 核心结论

CLIP 的零样本分类，本质上不是“模型临时猜新类别”，而是把图像和类别描述都投到同一个语义空间，再比较谁更近。语义空间可以理解成“按含义排布的坐标地图”：图像编码器把图像变成向量，文本编码器把 `a photo of a dog` 这类描述也变成向量，最后用相似度决定类别。

如果候选类集合是 $\{1,\dots,K\}$，图像向量记为 $x$，第 $j$ 个类别描述的文本向量记为 $z_j$，那么预测规则就是：

$$
\hat{y}=\arg\max_j z_j^\top x
$$

这条公式成立的前提，不是某个类别被监督训练过，而是 CLIP 在大规模图文对上学到了“图像内容”和“语言描述”之间的对齐关系。所以它能在部署时直接接收新类别的文字描述，而不必重新训练分类头。

对初学者最重要的一点是：零样本分类的能力，主要来自“共享表示”而不是“万能理解”。它适合语义清晰、文本容易描述、候选类可枚举的任务；对极细粒度视觉差异、专业领域图像、背景干扰很强的场景，效果会明显下降。

一个最直观的玩具例子是：把图像和“ 一张猫的照片 ”、“ 一张狗的照片 ”都投到同一张地图上，看图像点离哪一个文字点最近。CLIP 做的就是这件事，只不过地图维度很高，向量由神经网络学出来。

| 结论 | 含义 | 对工程的直接影响 |
|---|---|---|
| CLIP 用共享嵌入空间做匹配 | 不需要为每个新类重新训练分类器 | 新类上线速度快 |
| prompt 影响文本向量位置 | 同一类不同写法，得分会变 | 模板设计是性能变量 |
| 结果依赖预训练语义覆盖 | 域外概念、细粒度概念容易失效 | 需要验证边界而不是盲信 |
| 相似度不是因果理解 | 容易受共现背景、偏见影响 | 需要负提示、多模板、二阶段校验 |

---

## 问题定义与边界

零样本分类，指模型在没有见过目标类别标注样本的情况下，直接根据自然语言描述完成分类。这里的“零样本”不是完全没有学过任何知识，而是“没有学过这个任务里这些类的监督标签”。CLIP 在预训练阶段已经看过大量图文对，因此具备跨模态对齐能力；部署时只是不再依赖当前任务的标签数据。

更精确地说，给定一张图像和一组候选类别描述：

- 图像经过视觉编码器得到向量 $x$
- 每个类别文本经过文本编码器得到向量 $z_j$
- 经过归一化后，比较 $z_j^\top x$ 或 cosine 相似度
- 取最大者作为预测类别

可以把这个流程写成一个词图：

`prompt -> 文本向量 z_j`
`image -> 图像向量 x`
`x 与 z_j 做 cosine 相似度 -> 排序 -> top-1 / top-k 结果`

这里的 cosine 相似度，就是“只比较方向，不比较长度”的相似度；白话解释是：看两个向量是不是指向相近的语义方向，而不是谁数值更大。

它的边界也很明确。

第一，候选类必须能被语言清楚表达。像“猫”“狗”“救护车”这类概念，文字描述天然清晰；但如果类别差异来自非常细的局部纹理、工业缺陷、医学影像征象，单句 prompt 往往不够。

第二，候选类之间最好在语义空间中有可分性。比如“猫”和“飞机”差异大，零样本往往靠谱；“哈士奇”和“阿拉斯加犬”这种细粒度类别，往往需要更强局部视觉特征或少样本微调。

第三，分类结果受 prompt 质量强烈影响。同一个类别，写成 `a photo of a crane`，可能混淆“鹤”和“起重机”；写成 `a photo of a crane bird` 或 `a construction crane machine`，结果可能完全不同。

下面这个表格说明同一类别的 prompt 写法为什么会带来得分变化。数值是示意，重点是趋势。

| prompt 写法 | 预期效果 | 常见问题 |
|---|---|---|
| `a photo of a {class}` | 通用、稳定、适合作为基线 | 细节不足 |
| `a close-up photo of a {class}` | 对局部外观更敏感 | 可能损失场景信息 |
| `a photo of a {class} in the wild` | 对自然场景类更友好 | 对室内拍摄类不稳 |
| `a product photo of a {class}` | 电商图像常更准 | 对非商品场景偏置明显 |
| `a photo of a {class}, not background props` | 可减轻背景误导 | 负提示过强时会伤主语义 |

所以，CLIP 的零样本分类不是“给类别名就一定能识别”，而是“给出语言描述后，在已有语义空间中做最近邻匹配”。这决定了它更像一种高效检索与匹配机制，而不是对任务定义无条件鲁棒的专用分类器。

---

## 核心机制与推导

CLIP 的核心训练方式是对比学习。对比学习可以理解成“让正确图文对靠近，让不匹配图文对远离”的训练方法。它不是直接学一个固定标签集合，而是学一个共享空间中的相对位置关系。

设一个 batch 中有 $N$ 对图文样本，图像编码器输出 $v_i$，文本编码器输出 $t_i$。先做归一化：

$$
x_i=\frac{v_i}{\|v_i\|_2}, \quad z_i=\frac{t_i}{\|t_i\|_2}
$$

归一化的意义是把所有向量投到单位球面上，使点积等价于 cosine 相似度：

$$
x_i^\top z_j = \cos(\theta_{ij})
$$

这时，正样本对 $(x_i, z_i)$ 应该有更高相似度，负样本对 $(x_i, z_j), i\neq j$ 应该更低。典型的 InfoNCE 损失可以写成：

$$
\mathcal{L}_{img} = -\frac{1}{N}\sum_{i=1}^N
\log \frac{\exp(x_i^\top z_i / \tau)}
{\sum_{j=1}^N \exp(x_i^\top z_j / \tau)}
$$

$$
\mathcal{L}_{txt} = -\frac{1}{N}\sum_{i=1}^N
\log \frac{\exp(z_i^\top x_i / \tau)}
{\sum_{j=1}^N \exp(z_i^\top x_j / \tau)}
$$

总损失通常是两者平均。这里的 $\tau$ 是温度参数，可以理解成“拉开分数差距的缩放因子”；$\tau$ 越小，softmax 越尖锐，模型越强调正负样本之间的相对排序。

为什么训练完后可以做零样本分类？因为模型学到的不是“第 17 类是什么”，而是“图像内容和自然语言描述如何对齐”。只要新类别能写成文本，就可以通过文本编码器生成类别向量，再与图像向量比较。

看一个玩具数值例子。设图像向量已经归一化为：

$$
x=[0.6,0.8]
$$

两个候选类描述向量分别为：

$$
z_1=[0.707,0.707], \quad z_2=[0.2,0.98]
$$

则得分为：

$$
s_1=x^\top z_1=0.6\times0.707+0.8\times0.707\approx0.99
$$

$$
s_2=x^\top z_2=0.6\times0.2+0.8\times0.98\approx0.904
$$

因此选第一个类别。这个例子说明，CLIP 判断的是“语义方向更接近谁”，而不是“有没有见过这个类的标签”。

如果进一步转成概率，可以做 softmax：

$$
p(y=j\mid x)=\frac{\exp(s_j/\tau)}{\sum_k \exp(s_k/\tau)}
$$

但工程上要注意，这个“概率”更多是排序分数，不应直接当成严格校准后的真实置信度。

真实工程里，常见做法不是只写一个 prompt，而是给每个类准备多个模板，例如：

- `a photo of a {class}`
- `a close-up photo of a {class}`
- `a blurry photo of a {class}`
- `a low-light photo of a {class}`

然后对同一类别的多个文本向量做平均或投票。这样做的原因是：单个 prompt 会把类别绑死在某个表达方式上，而真实图像分布比一句模板更复杂。

---

## 代码实现

下面用一个可运行的 Python 玩具实现说明零样本分类的核心步骤。这里不依赖真实 CLIP 模型，只模拟“图像向量”和“文本向量”已经编码完成的情况，重点看归一化、相似度计算和多模板平均。

```python
import numpy as np

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.all(norm > 0), "zero vector is not allowed"
    return x / norm

def zero_shot_predict(image_vec, class_prompt_vecs):
    """
    image_vec: shape [D]
    class_prompt_vecs: dict[str, np.ndarray], each value shape [T, D]
    """
    x = l2_normalize(np.asarray(image_vec, dtype=float))

    class_names = []
    class_embeds = []

    for class_name, prompt_vecs in class_prompt_vecs.items():
        z = l2_normalize(np.asarray(prompt_vecs, dtype=float))
        z_mean = l2_normalize(z.mean(axis=0, keepdims=True))[0]
        class_names.append(class_name)
        class_embeds.append(z_mean)

    Z = np.vstack(class_embeds)              # [K, D]
    scores = Z @ x                           # cosine similarity after normalization
    pred_idx = int(np.argmax(scores))
    return class_names[pred_idx], scores

image_vec = np.array([0.6, 0.8])

class_prompt_vecs = {
    "class_1": np.array([
        [0.707, 0.707],
        [0.68, 0.73],
    ]),
    "class_2": np.array([
        [0.2, 0.98],
        [0.25, 0.96],
    ]),
}

pred, scores = zero_shot_predict(image_vec, class_prompt_vecs)
assert pred == "class_1"
assert scores[0] > scores[1]

print(pred, scores)
```

如果换成真实 CLIP，流程也类似，只是“向量如何得到”改成模型推理：

1. 加载图像编码器和文本编码器  
2. 为每个候选类生成若干 prompt  
3. 编码图像与文本  
4. 对向量归一化  
5. 做矩阵乘法得到相似度  
6. 对每类多模板求平均或最大值  
7. 输出 top-k 类别

伪代码如下：

```python
# pseudo code
image = preprocess(raw_image)
image_feat = clip.encode_image(image)             # [D]
image_feat = image_feat / ||image_feat||

texts = []
owner = []
for cls in candidate_classes:
    for template in templates:
        texts.append(template.format(cls))
        owner.append(cls)

text_feat = clip.encode_text(texts)               # [N, D]
text_feat = text_feat / ||text_feat||

scores = text_feat @ image_feat                   # [N]
grouped = aggregate_by_class(scores, owner)       # mean / max
pred = argmax(grouped)
```

一个真实工程例子是内容审核或社交媒体多标签归类。比如平台要快速上线“宠物”“枪支”“医疗器械”“广告图”“食物”几类识别，但没有时间先采集并标注一批训练集。此时可以直接构造类别描述与排除描述：

- `a photo of a pet animal`
- `a photo of a firearm weapon`
- `a photo of a medical device`
- `a commercial advertisement image`
- `a photo of food dish`

再把图像向量与这些描述做匹配，先得到一版零样本结果。若业务风险较高，再把高分样本送人工复核或二阶段模型。Springer 上的 CLIP-CMIC 思路，本质也是利用图文补全与 prompt 设计，把跨模态信息转成可比较的文本标签空间，适合冷启动场景。

下面这个表格说明 prompt 模板对得分的工程影响。

| 模板 | 类别：dog | 类别：wolf | 工程解释 |
|---|---:|---:|---|
| `a photo of a dog` | 0.81 | 0.63 | 通用基线 |
| `a close-up photo of a domestic dog` | 0.86 | 0.58 | 加入“domestic”后更利于与狼区分 |
| `an animal in the forest` | 0.54 | 0.72 | 场景词会把模型拉向狼 |
| `a pet dog, not wild animal` | 0.88 | 0.49 | 负提示抑制混淆类 |

---

## 工程权衡与常见坑

CLIP 零样本分类最常见的误解，是把它当成“无成本分类器”。实际上它把训练成本前置到了大规模预训练阶段，把部署时的成本转成了 prompt 设计、候选类定义和失败样本治理。

第一个坑是 prompt 硬化。所谓“硬化”，就是描述写得太死，只覆盖一种视觉形态。比如你要识别“杯子”，却只写 `a clean ceramic mug on a table`，那玻璃杯、保温杯、手持场景下的杯子都可能掉分。缓解方法不是无限加长 prompt，而是保留核心语义，再用多模板平均覆盖拍摄条件、视角和上下文。

第二个坑是背景共现偏差。模型有时会抓住“常和目标同时出现的东西”，而不是真正的目标本身。比如想识别“冲浪板”，图中如果有海浪、沙滩、泳衣，模型可能在缺少板体细节时仍给高分。这是因为 CLIP 学到的是互联网数据中的共现统计，不是严格的对象因果定义。

第三个坑是 CLIP-blind pairs。这个现象可以白话理解为：模型把“对人来说差很多，但语义上下文很接近”的图像看成邻居。例如外观差异明显的两类图片，因为都出现在相似场景、带有相似文本描述，最终在嵌入空间里被拉近。CVPR 的相关工作讨论的就是这类失败模式。它提醒我们，相似度高不等于真正理解了局部视觉差异。

第四个坑是概念关联偏差。比如图里有“人拿着球拍站在球场上”，如果 prompt 只问某个对象，模型可能会被整个场景带偏，把其他共现概念也一并补全。这在多概念图像里尤其常见。

下面是常见坑与缓解策略。

| 常见坑 | 典型表现 | 缓解策略 |
|---|---|---|
| prompt 太窄 | 换角度、换场景后掉分 | 多模板平均，覆盖场景和视角 |
| 背景干扰过强 | 靠背景猜中或猜错 | 在 prompt 中强调主体，加入排除项 |
| 细粒度类别混淆 | 相邻类分数很接近 | 二阶段精排，少样本微调 |
| 域外数据失效 | 医学、工业场景表现差 | 换领域模型或补充领域数据 |
| 分数不可校准 | 高分不等于高可靠 | 做阈值校准和人工抽检 |
| 多概念偏置 | 被共现物体带偏 | 多标签建模，增加负提示 |

实际工程里，一个很有用的办法是“多模板平均 + 负提示”。下面给一个简单代码片段：

```python
templates = [
    "a photo of a {cls}",
    "a close-up photo of a {cls}",
    "a product photo of a {cls}",
    "a photo of a {cls}, not background props",
]

def build_prompts(class_name):
    return [t.format(cls=class_name) for t in templates]

# 对每个 class_name 编码多个 prompt，再平均向量或平均分数
```

“not X” 这类负提示有时有效，但不能滥用。原因是 CLIP 并不是为复杂逻辑约束而专门训练的，负提示写得过多，反而会让文本向量偏离主概念，导致主类分数下降。实践上通常只排除最强干扰项，不做长句逻辑链。

---

## 替代方案与适用边界

如果任务目标是快速冷启动、候选类经常变化、标注预算很低，CLIP 零样本分类通常是优先选项。它的优势是部署快、类扩展快、工程实现简单。但这不代表它在所有任务里都是最优方案。

当任务需要更强的局部视觉理解时，可以结合自监督视觉特征，例如 DINO。自监督特征可以理解成“只靠图像本身结构学出来的视觉表示”，通常在局部纹理、形状差异上更稳。一个常见做法是用 CLIP 负责语义召回，用 DINO 或轻量分类器做近邻类别之间的二阶段精排。

当任务类目固定、而且误判成本较高时，少样本学习或轻量微调往往更可靠。比如质检场景里要区分正常焊点和多种细小缺陷，仅靠文本 prompt 很难描述足够充分，这时收集每类几十到几百张样本进行微调，收益通常很高。

一个实用的真实工程方案是：

1. 先用 CLIP 对几十个候选类做初选，保留 top-3  
2. 对 top-3 相邻类用小数据集训练一个精排模型  
3. 若最高分低于阈值，则输出“未知类”或送人工复核  

这个流程兼顾了冷启动速度和高风险场景下的可靠性。

下面对三种路线做一个对比。

| 方案 | 何时使用 | 样本需求 | 训练开销 | 优点 | 局限 |
|---|---|---:|---:|---|---|
| 零样本分类 | 冷启动、类频繁变化 | 0 | 无或极低 | 上线快，扩类方便 | 依赖 prompt 和预训练覆盖 |
| 少样本学习 | 类固定、样本少 | 每类少量 | 低到中 | 比零样本更稳 | 仍需采样和验证 |
| 全量微调 | 高精度、固定任务 | 较多 | 中到高 | 最适合专用任务 | 成本高、扩类慢 |

因此，CLIP 的合理定位不是“替代所有分类器”，而是“把开放类别问题先做成可上线系统”。当系统验证出高价值类目后，再决定是否进入少样本或全量微调阶段。这种“零样本召回 + 小样本精排”的混合路径，往往比直接追求一次到位更符合工程现实。

---

## 参考资料

| 来源 | 贡献 | 关键词 |
|---|---|---|
| Emergent Mind: Zero-Shot Accuracy in CLIP Models | 给出零样本分类公式、相似度定义与机制框架 | 零样本、共享嵌入、InfoNCE |
| Lightly: OpenAI CLIP Model Explained | 适合工程入门，解释图文共享空间与部署价值 | 工程背景、应用场景、语义地图 |
| Springer: CLIP-CMIC | 展示跨模态补全与 prompt 结合的零样本应用路径 | 跨模态分类、冷启动、多标签 |
| EngineersOfAI: CLIP and Contrastive Learning | 强调 prompt 设计与对比学习的工程含义 | prompt、多模板、对比学习 |
| CVPR 2024: Eyes Wide Shut? | 讨论 CLIP-blind pairs 等失败模式 | 失败模式、邻域错配、鲁棒性 |

- Emergent Mind 的价值在于把公式、推导和零样本分类定义讲清楚，适合作为机制主线。
- Lightly 的价值在于把 CLIP 解释成“统一语义空间”，适合新手理解为什么文字能直接变成分类器。
- Springer 的 CLIP-CMIC 提供了跨模态冷启动场景的工程视角，说明 prompt 不只是分类标签，还能参与信息补全。
- EngineersOfAI 更偏工程实践，能帮助理解 prompt 写法为什么会改变结果。
- CVPR 关于 CLIP-blind pairs 的工作提醒读者：高相似度不等于高可靠性，失败模式分析是部署前必须补的一课。
