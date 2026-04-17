## 核心结论

多模态幻觉指视觉语言模型在看图回答或图像描述时，输出了与图像不一致的对象、属性、数量或关系。白话说，就是模型“看图说错话”，而且通常还说得很像真的。

这类错误大致分成两类：

| 类型 | 直接来源 | 典型触发 | 常见表现 | 更适合的检测方式 |
| --- | --- | --- | --- | --- |
| 视觉幻觉 | 视觉特征提取不充分 | 小目标、遮挡、错觉图、细粒度属性 | 把圆形区域说成“狗”，把同样大小的图形说成不一样大 | HallusionBench 的 VD/Hard、视觉推理题 |
| 语言幻觉 | 语言先验过强 | 训练中高频共现、常识模板、提示词诱导 | 把“猫在沙发上”套到任意室内图，把常见共现物体自动补全 | POPE 的 popular/adversarial、VS 对照题 |

关键判断标准不是“答案听起来像不像”，而是“答案是否被图像支持”。同一句话在语言模型里可能很自然，在多模态系统里却可能是错误的。

一个玩具例子很直观：

- 图里只有一个橙色圆和一个蓝色方块，模型却回答“有一只狗”。这更像视觉幻觉，因为视觉编码阶段已经把局部形状读错了。
- 图里是一张普通客厅照片，没有猫，但模型说“猫趴在沙发上”。这更像语言幻觉，因为“猫+沙发”在训练语料里共现太频繁。

工程上不能只看一个总分。POPE 更擅长量化“对象是否存在”这类语言偏差；HallusionBench 更擅长拆开“忽视视觉”与“误读视觉”两类错误；VCD、RLHF-V、HA-DPO 则分别对应推理时抑制偏差、训练时补强视觉对齐、偏好优化时拉开正确与错误答案的概率差。

---

## 问题定义与边界

先把边界说清楚。本文讨论的对象是 LVLM 或 MLLM。LVLM 是 large vision-language model，白话说就是“既看图又能生成文字的大模型”；MLLM 是 multimodal large language model，范围更广，除图文外也可包含音频、视频等。

多模态幻觉的严格定义可以写成：

$$
\hat{y}\not\subseteq \mathcal{E}(x)
$$

其中 $x$ 是输入图像，$\mathcal{E}(x)$ 表示图像中可被视觉证据支持的事实集合，$\hat{y}$ 是模型输出。只要输出中包含了图像证据不支持的对象、属性、数量或关系，就属于幻觉。

这里有两个容易混淆的边界：

| 区分项 | 语言幻觉 | 视觉幻觉 |
| --- | --- | --- |
| 本质 | 模型更信“记忆里的常见模式” | 模型更信“自己读到的错误视觉特征” |
| 是否必须有图像才能触发 | 不一定，弱视觉输入时更明显 | 通常必须依赖图像细节 |
| HallusionBench 关注点 | VS/Hard 下忽视图像修正 | VD/Hard 下误解图像本身 |
| POPE 对应样本策略 | popular、adversarial 更敏感 | 间接可测，但不是主强项 |

HallusionBench 的设计价值在于，它不是只问一个问题，而是给出控制组。所谓 VS，Visual Supplement，白话说是“即使不看图也容易凭常识回答的问题”，图像只负责修正常识；所谓 VD，Visual Dependent，白话说是“必须看图才能答对的问题”。

同一张图上可以同时出现两类错误：

- VS 问题：给一张看起来像“标准客厅”的图，问“沙发上有猫吗？”如果模型无视图像直接答“有”，这是语言幻觉。
- VD 问题：给一个经过编辑的几何图，问“右侧橙色圆是否和左侧一样大？”如果模型被视觉错觉误导答错，这是视觉幻觉。

POPE 的边界更窄但更干净。它把问题压缩成二分类：“图中有没有某个对象？”这样做的好处是评价稳定，不依赖生成长描述时的句法差异；缺点是它主要盯“对象存在性”，对属性、空间关系、视觉推理链路覆盖不够。

因此，POPE 不是 HallusionBench 的替代，而是对象层面的局部显微镜；HallusionBench 也不是 POPE 的替代，而是图像推理层面的诊断套件。

---

## 核心机制与推导

先看为什么会幻觉。简化后，模型在每一步生成 token 时，本质是在比较一组条件概率：

$$
p_\theta(y_t \mid x, q, y_{<t})
$$

这里 $x$ 是图像，$q$ 是文本提示，$y_{<t}$ 是前面已经生成的 token。问题在于，这个概率同时混入了视觉证据和语言先验。若视觉证据弱、语言先验强，某些 token 会被错误抬高。

### 1. VCD：用原图和扰动图做对比

VCD 是 Visual Contrastive Decoding。白话说，它不是只看“原图下哪个词概率高”，而是比较“原图”和“破坏视觉细节后的图”在同一个词上的概率差。

一个常见的简化写法是：

$$
s_t(w)=\log p_\theta(w\mid x,q,y_{<t})-\alpha \log p_\theta(w\mid \tilde{x},q,y_{<t})+\beta \cdot g_t(w)
$$

其中：

- $x$ 是原图
- $\tilde{x}$ 是扰动图，比如加噪、模糊、patch shuffle
- $\alpha$ 控制“减去多少扰动图分数”
- $\beta$ 控制额外的重加权项
- $g_t(w)$ 可以理解为可行性约束，避免把所有高频词都粗暴压掉

直觉是这样的：

- 如果一个 token 真依赖视觉，比如“traffic light”，那么原图清晰、扰动图模糊后，它的分数通常会下降明显。
- 如果一个 token 主要来自语言习惯，比如“sofa”后很容易跟“cat”，那么即使把图像扰乱，它的概率也未必下降，甚至还会因为视觉信号变弱而被语言先验顶上去。
- VCD 就把这类“对视觉不敏感、但对语言模板很敏感”的 token 压下去。

玩具例子：

- 原图里只有桌子，没有沙发。
- 当前候选 token 有 `table` 和 `sofa`。
- 原图 logits：`table=4.2`，`sofa=3.8`
- 扰动图 logits：`table=2.1`，`sofa=3.6`

这说明 `table` 真依赖视觉，而 `sofa` 更多来自语言惯性。对比后，`table` 应保留，`sofa` 应被压制。

可以把流程记成：

```text
原图 x -> logits_o
扰动图 x~ -> logits_d
logits_o - alpha * logits_d -> 视觉对比分数
再加 beta * 可行性约束 -> 最终采样分数
```

### 2. HA-DPO：把“非幻觉答案”作为偏好正样本

DPO 是 Direct Preference Optimization。白话说，它不显式训练奖励模型，而是直接让模型更偏向“好答案”而不是“坏答案”。

HA-DPO 把问题改写成：对同一张图和同一条指令，给模型一个非幻觉答案 $y^+$，和一个有幻觉答案 $y^-$，训练模型更偏好前者。常见损失写成：

$$
\mathcal{L}_{\rm DPO}=-\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y^+|x)}{\pi_{\rm ref}(y^+|x)}-\log\frac{\pi_\theta(y^-|x)}{\pi_{\rm ref}(y^-|x)}\right)\right)\right]
$$

这里：

- $\pi_\theta$ 是当前模型
- $\pi_{\rm ref}$ 是 reference policy，白话说是训练时拿来做约束的参考模型
- $y^+$ 是非幻觉答案
- $y^-$ 是幻觉答案
- $\beta$ 控制偏好拉开的强度

这个式子的核心含义不是“让正确答案概率绝对最大”，而是“让当前模型相对参考模型，更倾向正确答案而不是错误答案”。

一个新手能理解的例子：

- 问题：“图中桌子上有什么？”
- $y^+$：“桌子上有一本书和一个杯子。”
- $y^-$：“桌子上有一本书、一个杯子和一只猫。”

如果参考模型本来就很爱说“猫”，HA-DPO 的目标就是把“加了猫”的回答压下去，同时保留“书、杯子”这些被图像支持的内容。

### 3. 为什么 reference policy 对收敛很关键

DPO 隐含了 KL 约束。KL 是 Kullback-Leibler divergence，白话说是“新模型和参考模型分布差多远”。如果偏好数据太 off-policy，也就是正样本和参考模型天然分布相差太大，梯度会变弱，模型学不到。

这也是为什么 OPA-DPO 强调先把数据对齐到 on-policy，再做 DPO。否则会出现一个常见现象：人类改写后的答案非常好，但离原模型“会说的话”太远，训练时等于在强行拉模型跳过很大一段分布距离，结果反而不稳定。

---

## 代码实现

下面用一个最小可运行的 Python 例子，把 POPE 采样、HallusionBench 风格问题组织、VCD 打分、HA-DPO 损失串起来。这个例子是教学版，不依赖深度学习框架，但变量关系和真实工程一致。

```python
import math
from collections import Counter, defaultdict

# -------------------------
# 1) POPE 负样本采样
# -------------------------

images = [
    {"id": 1, "objects": ["person", "chair", "table"]},
    {"id": 2, "objects": ["dog", "sofa"]},
    {"id": 3, "objects": ["person", "surfboard", "sea"]},
]

# 全局频率
freq = Counter()
for item in images:
    freq.update(item["objects"])

# 共现频率
cooccur = defaultdict(Counter)
for item in images:
    objs = item["objects"]
    for a in objs:
        for b in objs:
            if a != b:
                cooccur[a][b] += 1

vocab = sorted(set(freq.keys()) | {"cat", "bottle", "knife", "apple"})

def random_sampling(image_objects, k=2):
    candidates = [o for o in vocab if o not in image_objects]
    return candidates[:k]

def popular_sampling(image_objects, k=2):
    candidates = [o for o, _ in freq.most_common() if o not in image_objects]
    return candidates[:k]

def adversarial_sampling(image_objects, k=2):
    score = Counter()
    for obj in image_objects:
        score.update(cooccur[obj])
    ranked = [o for o, _ in score.most_common() if o not in image_objects]
    # 若候选不够，用频率补齐
    for o, _ in freq.most_common():
        if o not in image_objects and o not in ranked:
            ranked.append(o)
    return ranked[:k]

img = images[0]
assert img["objects"] == ["person", "chair", "table"]
assert "dog" in random_sampling(img["objects"], k=4)
assert "sofa" in adversarial_sampling(["dog"], k=4)

# -------------------------
# 2) HallusionBench 风格组织
# -------------------------

questions = [
    {"type": "VS", "question": "沙发上有猫吗？", "gt": "no"},
    {"type": "VD", "question": "右侧圆是否比左侧更大？", "gt": "yes"},
]
assert {q["type"] for q in questions} == {"VS", "VD"}

# -------------------------
# 3) VCD 打分
# -------------------------

def softmax(logits):
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}

def vcd_score(logits_orig, logits_dist, alpha=1.0, beta=0.2):
    # beta 这里用一个简单的可行性项：偏向原图下概率不太低的 token
    p_orig = softmax(logits_orig)
    scores = {}
    for token in logits_orig:
        scores[token] = logits_orig[token] - alpha * logits_dist[token] + beta * math.log(p_orig[token] + 1e-8)
    return scores

orig = {"table": 4.2, "sofa": 3.8, "cat": 3.1}
dist = {"table": 2.1, "sofa": 3.6, "cat": 3.0}
scores = vcd_score(orig, dist, alpha=1.0, beta=0.2)

best_token = max(scores, key=scores.get)
assert best_token == "table"   # 桌子更依赖真实视觉证据
assert scores["sofa"] < scores["table"]

# -------------------------
# 4) HA-DPO 损失
# -------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def ha_dpo_loss(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, beta=0.5):
    margin = beta * ((logp_pos - logp_ref_pos) - (logp_neg - logp_ref_neg))
    return -math.log(sigmoid(margin) + 1e-8)

loss = ha_dpo_loss(
    logp_pos=-1.0,      # 当前模型对正确答案的 log prob
    logp_neg=-2.0,      # 当前模型对幻觉答案的 log prob
    logp_ref_pos=-1.2,  # 参考模型
    logp_ref_neg=-1.5,
    beta=0.5
)
assert loss > 0
assert round(loss, 4) > 0

print("toy pipeline ok")
```

变量说明如下：

| 变量 | 含义 |
| --- | --- |
| $\alpha$ | VCD 中扰动图对比分支权重 |
| $\beta$ | DPO 温度或 VCD 中附加重加权强度，具体含义取决于模块 |
| $y^+$ | 非幻觉答案 |
| $y^-$ | 幻觉答案 |
| $\pi_{\rm ref}$ | 参考策略，用于约束更新幅度 |
| logits\_orig | 原图下 token 分数 |
| logits\_dist | 扰动图下 token 分数 |

真实工程例子可以这样落地：

- 评测层：离线跑 POPE 的 random/popular/adversarial 三套问答，同时跑 HallusionBench 的 VS/VD。
- 推理层：在线服务对高风险场景开启 VCD，仅在描述生成或长答案模式启用。
- 训练层：收集人工纠错或 GPT-4V 修订结果，构造 $(y^+, y^-)$，做 RLHF-V 或 HA-DPO。
- 发布层：同时汇报 POPE F1、HallusionBench pair accuracy、业务自定义误报率。

---

## 工程权衡与常见坑

先看 POPE。它最大的价值不是“给一个分数”，而是用不同采样方式暴露不同偏差。POPE 原论文在 MSCOCO 上给出的 InstructBLIP 结果是：random 设置准确率 88.73、F1 89.29；popular 设置准确率 81.37、F1 83.45；adversarial 设置准确率 74.37、F1 78.45。性能从 random 到 adversarial 持续下降，说明模型会被高频共现和上下文共现牵着走。

| 采样策略 | 负样本来源 | 暴露能力 | 风险 |
| --- | --- | --- | --- |
| random | 随机不存在对象 | 基础体检 | 容易低估语言偏差 |
| popular | 全局高频对象 | 暴露高频先验 | 仍未针对图像上下文 |
| adversarial | 与真值对象高共现的对象 | 最容易戳出语言先验 | 分数更低，但更接近真实风险 |

HallusionBench 的坑不在“难”，而在“容易被误用”：

- 如果只看 easy question accuracy，会高估模型能力。
- 如果不分 VS 和 VD，就无法区分“忽视视觉”和“误读视觉”。
- 如果新领域图像和原 benchmark 差太远，分数下降未必全是幻觉，也可能是领域迁移失败。

DPO 系方法的坑更隐蔽：

- 离线构造的 $y^+$ 若改写过度，容易和 $\pi_{\rm ref}$ 严重失配，产生 off-policy 问题。
- reference policy 若选得太旧，KL 约束会把模型锁在旧分布附近。
- 只做语言偏好对，不做图像偏好对，会让模型学会“更会说”，但不一定“更会看”。

RLHF-V 提供了一个很实用的工程信号：其项目页公开说明，基于 Muffin 的 MLLM 只用 1.4K 条细粒度纠错数据，就把基础模型幻觉率降低了 34.8%，训练耗时约 1 小时，硬件为 8 张 A100。这个量级说明，在视觉客服、商品审核、图文质检这类任务里，先做小规模高质量纠错集，往往比盲目扩大全量 SFT 更划算。

一个真实工程例子：

- 视觉客服场景里，用户上传商品图问“盒子里还有充电头吗？”
- 基础模型常把“手机盒”默认补成“有充电头”，这属于语言幻觉。
- 若使用 POPE 风格对象探针做离线诊断，会发现 `charger` 在 popular/adversarial 下误报更高。
- 在线推理加 VCD，可压制“套餐模板”诱导。
- 训练上加 RLHF-V 或 HA-DPO，可用少量人工修正把“有/没有配件”的偏差拉回来。

常见注意事项可以直接记成下面几条：

- 只跑 random 采样不够，至少同时跑 popular 和 adversarial。
- HallusionBench 要拆分 VS/VD 和 easy/hard 四个切片看。
- DPO 训练前先检查 preferred response 与 reference policy 的可采样性。
- VCD 会增加推理成本，不适合所有请求默认开启。
- 医学、遥感、工业检测等领域不能直接照搬通用图像 benchmark。

---

## 替代方案与适用边界

可以把选择路径简化成一个决策树：

```text
如果主要担心“模型总爱补不存在的物体”
-> 先用 POPE

如果主要担心“模型在细节推理、编辑图、错觉图上读错”
-> 先用 HallusionBench

如果线上问题是“要立刻减少生成期幻觉，不能重训”
-> 用 VCD 一类解码方法

如果已经有纠错数据，准备做训练对齐
-> 用 RLHF-V / HA-DPO
-> 若担心 off-policy，优先看 OPA-DPO
```

各方案的适用边界如下：

| 方案 | 典型适用环境 | 优点 | 限制 |
| --- | --- | --- | --- |
| POPE | 对象存在性诊断、离线评测 | 稳定、便宜、易扩展到 yes/no | 对属性、关系、推理覆盖弱 |
| HallusionBench | 视觉细节、控制组分析、模型诊断 | 能拆语言幻觉和视觉幻觉 | 题量有限，领域迁移要谨慎 |
| VCD | 无法重训的线上推理服务 | 训练免费、即插即用 | 增加解码成本，需调 $\alpha$ |
| RLHF-V | 有小规模高质量人工纠错 | 数据效率高 | 标注成本仍然存在 |
| HA-DPO | 已有偏好对、要做对齐训练 | 直接优化偏好差 | 依赖 reference policy 选择 |
| OPA-DPO | 偏好对明显 off-policy 的场景 | 解决 KL 失配更稳 | 流程更复杂，需要额外对齐步骤 |

对新领域要有清醒判断。比如医学图像里，“肺纹理增多”“边界毛刺”不属于 POPE 的对象存在性强项，HallusionBench 的通用题也未必覆盖。这时更合理的做法是：

- 先保留 POPE 思想，但把对象词表改成领域实体
- 同时增加领域版 VD/VS 问题
- 收集 on-policy 纠错数据，再用 HA-DPO 或 OPA-DPO 训练

一句话概括适用边界：POPE 更像“对象偏差探针”，HallusionBench 更像“视觉推理体检”，VCD 和 DPO 系方法才是“真正改输出”的控制手段。

---

## 参考资料

| 资料 | 主要贡献 | 链接 |
| --- | --- | --- |
| Evaluating Object Hallucination in Large Vision-Language Models | 提出 POPE；定义 random、popular、adversarial 三种采样；给出 MSCOCO 上的系统评测 | [OpenReview](https://openreview.net/forum?id=xozJw0kZXF) |
| RUCAIBox/POPE | POPE 官方代码与数据构建方式；适合复现实验与扩展到自有数据 | [GitHub](https://github.com/RUCAIBox/POPE) |
| HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models | 提出 VS/VD 控制组，区分语言幻觉与视觉幻觉 | [GitHub](https://github.com/tianyi-lab/HallusionBench) |
| HallusionBench 论文页 | 说明 346 张图、1129 个问题，以及 question-pair accuracy 等核心指标 | [arXiv](https://arxiv.org/abs/2310.14566) |
| Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding | 提出 VCD；通过原图与扰动图对比分布，抑制语言先验 | [arXiv](https://arxiv.org/abs/2311.16922) |
| RLHF-V | 细粒度纠错式人类反馈；展示 1.4K 数据、8×A100、约 1 小时训练的工程可行性 | [项目页](https://rlhf-v.github.io/) |
| RLHF-V 官方仓库 | 训练、评测与数据说明；适合直接查实现流程 | [GitHub](https://github.com/RLHF-V/RLHF-V) |
| Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization | 提出 HA-DPO，用偏好优化降低多模态幻觉 | [arXiv](https://arxiv.org/abs/2311.16839) |
| OPA-DPO: Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key | 说明 off-policy 偏好对会受 KL 约束影响，提出先做 on-policy 对齐再做 DPO | [项目页](https://opa-dpo.github.io/) |
| Microsoft Research: OPA-DPO | 用工程化语言解释 why off-policy DPO 难学，以及 4.8k 数据的对齐流程 | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/mitigating-hallucinations-in-large-vision-language-models-via-dpo-on-policy-data-hold-the-key/) |
