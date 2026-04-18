## 核心结论

HallusionBench 是一个专门评测视觉语言模型是否会在图像证据上“看错、编造、前后不一致”的基准，而不是普通的视觉问答排行榜。视觉语言模型，指同时接收图像和文本、并输出文本答案的模型。

它的核心价值不是告诉你模型“总体答对多少题”，而是检查模型在成对问题、干扰图、反事实条件下是否仍然自洽。自洽，指模型在等价或受控变化的输入下，答案逻辑保持一致。

| 维度 | HallusionBench 测什么 | HallusionBench 不主要测什么 |
|---|---|---|
| 图像理解 | 图像证据是否被正确读取 | 一般物体识别的平均能力 |
| 推理稳定性 | easy / hard 条件下答案是否一致 | 单题碰巧答对 |
| 幻觉类型 | 语言幻觉、视觉幻觉、视觉错觉 | 长文本写作质量 |
| 输出形式 | yes/no 判断为主 | 开放式长回答 |
| 工程用途 | 构造多模态回归测试集 | 替代所有多模态评测 |

规模需要校正：公开论文与 CVPR 页面给出的 HallusionBench 是 346 张图、1129 道 yes/no 问题，不是 4K+ 问题。部分二次资料可能把派生样本、模型输出或其他集合混在一起统计，但做工程复现时应以官方论文、仓库和数据集页为准。

玩具例子：同一个问题“图中是否有红色汽车？”放在原图和干扰图里。如果原图没有红色汽车，干扰图也没有红色汽车，模型却一会儿答 yes、一会儿答 no，问题就不只是单题错误，而是模型没有稳定依赖图像证据。

---

## 问题定义与边界

幻觉，指模型输出的内容缺少输入证据支持，或者直接和输入证据冲突。在视觉语言模型中，输入证据通常包括图像 $I$ 和问题 $q$，模型输出答案 $\hat{y}$。如果图像里没有红色汽车，而模型回答“有红色汽车”，就是典型幻觉。

HallusionBench 重点区分两类错误：

| 类型 | 白话解释 | 典型表现 |
|---|---|---|
| language hallucination | 模型主要靠文字先验猜答案，没有真正使用图像 | 问到“香蕉是不是黄色”，即使图中没有香蕉也答 yes |
| visual hallucination / visual illusion | 模型看了图，但读错了图像细节或被视觉干扰误导 | 图表最高柱是瑞士，模型说中国最高 |

这里的 language hallucination 不是说模型语言能力差，而是说它把问题里的常识、统计偏见或训练记忆当成了图像证据。visual hallucination 更接近“看错图”：模型确实处理了图像，但把不存在的对象、错误的数量、错误的位置当成真实内容。

| 问题 | 图像证据 | 模型答案 | 幻觉判定 |
|---|---|---|---|
| 图中是否有红色汽车？ | 没有汽车 | 有 | 是，答案与图像冲突 |
| 图中最大柱子是否代表瑞士？ | 最大柱子代表瑞士 | 是 | 不是，答案被证据支持 |
| 猫通常有几条腿？ | 图像无关 | 四条 | 不宜算视觉理解正确 |
| 图中人穿的是蓝色衣服吗？ | 图像模糊，标注也不确定 | 否 | 可能是标注歧义，需人工复核 |

边界很重要。不是所有“模型输出不理想”都叫幻觉。格式错误、答非所问、拒答、解释太长，可能会影响自动评分，但不一定说明模型看到了不存在的东西。

| 情况 | 是否算幻觉 | 原因 |
|---|---|---|
| 图中没有狗，模型说有狗 | 是 | 与图像证据冲突 |
| 图中有狗，模型回答“yes, because ...” | 不一定 | 可能只是格式未归一 |
| 图像分辨率太低，人也无法判断 | 不宜直接算 | 标注边界不清 |
| 问题本身有歧义 | 不宜直接算 | 需要修正数据 |
| 模型拒绝回答 | 通常单独统计 | 这是可用性问题，不等同幻觉 |

真实工程例子：一个巡检系统需要判断截图里是否出现 “Error” 弹窗。如果模型在没有弹窗的截图中回答“有错误弹窗”，这会触发误报警；如果在有弹窗时回答“没有”，会漏报事故。HallusionBench 的思路适合把这类场景做成 yes/no 回归集：每次换模型、换提示词、换图像压缩策略，都重新检查证据一致性。

---

## 核心机制与推导

HallusionBench 的核心机制是控制变量。控制变量，指只改变一个关键条件，观察输出是否按预期变化。普通视觉问答只看一张图配一个问题；HallusionBench 更关注一组相关图像、相关问题、反事实问题之间的关系。

可以把视觉语言模型抽象为：

$$
\hat{y} = f(V_{enc}(I), L_{enc}(q))
$$

其中 $I$ 是图像，$q$ 是问题，$V_{enc}$ 是视觉编码器，$L_{enc}$ 是语言编码器，$\hat{y}$ 是模型预测答案。视觉编码器，指把图像转换成模型可处理向量的模块；语言编码器，指把文本转换成模型可处理向量的模块。

普通单题准确率是：

$$
Acc_q = \frac{1}{|Q|}\sum_i \mathbf{1}[\hat{y}_i = y_i]
$$

这里 $Q$ 是问题集合，$y_i$ 是标准答案，$\mathbf{1}[\cdot]$ 表示条件成立时取 1，否则取 0。

但单题准确率会高估模型。原因是 yes/no 题存在猜测空间，语言先验也可能让模型碰巧答对。例如问题是“这张图里香蕉是黄色的吗？”模型即使没看图，也可能凭常识回答 yes 并得分。

HallusionBench 因此引入更严格的图级和成对指标：

$$
Acc_{fig} = \frac{1}{|F|}\sum_f \mathbf{1}[\text{图 } f \text{ 上的所有题都答对}]
$$

$$
Acc_{pair} = \frac{1}{|P|}\sum_p \mathbf{1}[\hat{y}(q,I_{easy})=y_{easy} \land \hat{y}(q,I_{hard})=y_{hard}]
$$

| 指标 | 计算对象 | 严格程度 | 能发现的问题 |
|---|---|---|---|
| $Acc_q$ | 单个问题 | 低 | 基础答题错误 |
| $Acc_{fig}$ | 同一图上的所有问题 | 中 | 对局部细节不稳定 |
| $Acc_{pair}$ | easy / hard 成对问题 | 高 | 靠语言猜测、被干扰图误导 |

推导很直接。假设一个 pair 里有两道题，每题准确率都是 80%。如果两题错误独立，那么 pair 全对概率最多约为：

$$
0.8 \times 0.8 = 0.64
$$

这说明单题 80% 看起来不错，但要求同一组受控问题全部答对时，分数会明显下降。如果错误不是独立的，而是来自同一个视觉误读，pair 分数还会更低。

新手版例子：easy 图里目标物体明显，hard 图里有相似干扰物。模型 easy 答对、hard 答错，说明它不是稳定读取图像证据，而是依赖局部纹理、常识或问题里的关键词。

---

## 代码实现

工程上实现 HallusionBench 风格评测，通常分四步：加载数据、构造成对样本、调用模型、汇总指标。重点不是训练模型，而是把评测管线做成可复现、可追踪、可解释。

| 字段 | 含义 | 示例 |
|---|---|---|
| `q` | 问题文本 | 图中是否有红色汽车？ |
| `I` | 图像路径或图像对象 | `images/001.png` |
| `y` | 标准答案 | `no` |
| `pred` | 模型输出归一后的答案 | `no` |
| `pair_id` | 成对样本编号 | `car-color-001` |
| `figure_id` | 图像编号 | `fig-001` |

最小伪代码如下：

```text
读取评测数据
按 pair_id 分组成对样本
对每条样本调用模型：predict(image, question) -> yes/no
把模型输出归一成 yes 或 no
计算 Acc_q、Acc_fig、Acc_pair
输出按类别、子类别、错误类型聚合的结果
```

下面是一个可运行的 Python 玩具实现。它不调用真实模型，而是模拟评测指标的计算方式：

```python
from collections import defaultdict

samples = [
    {
        "id": "s1",
        "pair_id": "p1",
        "figure_id": "f1",
        "q": "Is there a red car in the image?",
        "y": "no",
        "pred": "no",
    },
    {
        "id": "s2",
        "pair_id": "p1",
        "figure_id": "f2",
        "q": "Is there a red car in the image?",
        "y": "no",
        "pred": "yes",
    },
    {
        "id": "s3",
        "pair_id": "p2",
        "figure_id": "f3",
        "q": "Is the tallest bar Switzerland?",
        "y": "yes",
        "pred": "yes",
    },
    {
        "id": "s4",
        "pair_id": "p2",
        "figure_id": "f3",
        "q": "Is the tallest bar China?",
        "y": "no",
        "pred": "no",
    },
]

def normalize_answer(text):
    text = text.strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    raise ValueError(f"cannot normalize answer: {text}")

def accuracy_by_question(rows):
    correct = sum(normalize_answer(r["pred"]) == normalize_answer(r["y"]) for r in rows)
    return correct / len(rows)

def grouped_all_correct(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    ok = 0
    for group_rows in groups.values():
        if all(normalize_answer(r["pred"]) == normalize_answer(r["y"]) for r in group_rows):
            ok += 1
    return ok / len(groups)

acc_q = accuracy_by_question(samples)
acc_fig = grouped_all_correct(samples, "figure_id")
acc_pair = grouped_all_correct(samples, "pair_id")

assert acc_q == 0.75
assert round(acc_fig, 4) == 2 / 3
assert acc_pair == 0.5
```

真实工程中，`predict(image, question)` 可以是一次 API 调用，也可以是本地模型推理。无论哪种方式，都要保存原始输出、归一后答案、图像版本、提示词版本和模型版本。否则当分数下降时，很难判断是模型退化、图片压缩变化，还是答案抽取规则变了。

---

## 工程权衡与常见坑

总准确率高，不代表模型没有幻觉。HallusionBench 的价值正在于拆开看：模型是在所有类型上都稳定，还是只在简单问题上分数高、在反事实和干扰条件下崩掉。

| 常见坑 | 问题 | 规避方式 |
|---|---|---|
| 只看总准确率 | 掩盖 hard 样本和 pair 样本失败 | 同时看 $Acc_q$、$Acc_{fig}$、$Acc_{pair}$ |
| 开放式输出直接判分 | “yes, because ...” 被误判为错误 | 先做答案归一，再判分 |
| 只调提示词不补数据 | 模型可能学会答题格式，但仍不看图 | 增加反事实图和负样本 |
| 混淆语言幻觉和视觉幻觉 | 无法定位问题来源 | 分别分析无图先验和有图误读 |
| 样本切分泄漏 | 调参时见过测试集模式 | 固定验证集和最终测试集 |
| 图像预处理不一致 | 缩放、裁剪导致证据消失 | 记录图像尺寸和预处理策略 |

新手版例子：模型输出 `yes, because I can see a red car`。如果标准答案是 `yes`，直接字符串比较会判错；如果只截取第一个词，则可以判对。但如果模型输出 `yes, but there is no car`，简单截取又会误判。所以答案抽取规则必须稳定，并且要抽样人工复核。

工程版例子：你在生产环境中评测“截图是否包含支付失败提示”。模型输出可能是“是的，页面显示 payment failed”。如果评测脚本只接受小写 `yes`，指标会被输出风格污染。更稳妥的做法是强制模型只输出 JSON，例如 `{"answer":"yes"}`，再用解析器读取字段。

规避规则可以分成两类：

| 现象 | 更可能的问题 | 优先改法 |
|---|---|---|
| 无图也能答对很多题 | 语言先验过强 | 增加反事实问题和负样本 |
| easy 对、hard 错 | 视觉细节不稳 | 提高分辨率、改视觉编码器、补困难样本 |
| 同图多题互相矛盾 | 推理链不一致 | 加一致性约束和成组评测 |
| 输出格式混乱 | 评测接口不稳 | 约束输出 schema |
| 某类图表持续错 | 数据覆盖不足 | 增加该子类训练或评测样本 |

---

## 替代方案与适用边界

HallusionBench 适合测“图像证据一致性”，不适合覆盖所有多模态能力。它的强项是 yes/no、反事实、干扰条件、成对一致性；弱项是开放式描述、长文档理解、多轮交互和复杂工具调用。

| 场景 | 更适合的评测 | 为什么 |
|---|---|---|
| 判断图中说法是否成立 | HallusionBench 风格评测 | 能检查证据一致性 |
| 给图片写自然语言描述 | Captioning 评测或人工评审 | 输出不是 yes/no |
| 长截图问答 | 文档视觉问答基准 | 需要 OCR 和版面理解 |
| 多轮图像对话 | 多轮多模态对话评测 | 需要上下文记忆 |
| 图表数值读取 | 图表理解基准 | 需要细粒度数值对齐 |
| 生产回归测试 | 自建 HallusionBench 风格集合 | 能贴合业务错误模式 |

如果你的任务是“给图片写一段描述”，HallusionBench 不能替代 captioning 评测。因为 captioning 的目标是覆盖主要对象、关系和场景，而 HallusionBench 主要检查某个判断是否被图像证据支持。

如果你的任务是“判断截图里是否有错误信息”，HallusionBench 的思路很适合做回归集。你可以为每类错误准备正样本、负样本、相似干扰样本，并要求模型只输出 yes/no。然后用 pair 指标检查模型是否真的读到了截图中的错误提示。

缓解方向通常有三类：

| 缓解方法 | 白话解释 | 适用情况 |
|---|---|---|
| 负样本构造 | 加入“图中不存在”的样本 | 模型总是过度回答 yes |
| 鲁棒指令微调 | 用更稳定的指令数据训练模型 | 输出格式和证据使用都不稳 |
| 视觉证据重加权 | 让模型更重视图像特征 | 语言先验压过图像证据 |
| 高分辨率输入 | 保留小字、图表、局部细节 | 截图、图表、文档场景 |
| 分步验证 | 先定位证据，再回答 yes/no | 需要可解释性和审计 |

适用边界可以简化为三句话：第一，问题最好能归一为明确 yes/no。第二，答案必须能被图像证据验证。第三，评测集要包含干扰条件，否则很难区分“真的看懂”和“碰巧猜对”。

---

## 参考资料

1. HallusionBench 官方 GitHub 仓库：https://github.com/tianyilab/HallusionBench  
2. CVPR 2024 论文页面：https://cvpr.thecvf.com/virtual/2024/poster/29422  
3. CVPR Open Access 论文页：https://openaccess.thecvf.com/content/CVPR2024/html/Guan_HallusionBench_An_Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_and_CVPR_2024_paper.html  
4. Hugging Face 数据集镜像 lmms-lab/HallusionBench：https://huggingface.co/datasets/lmms-lab/HallusionBench  
5. 相关缓解方向论文：Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning：https://arxiv.org/abs/2306.14565  
6. HallusionBench arXiv 论文：HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination & Visual Illusion in Large Vision-Language Models：https://arxiv.org/abs/2310.14566
