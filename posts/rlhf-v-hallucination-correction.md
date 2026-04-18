## 核心结论

RLHF-V 是一种面向视觉语言模型的细粒度幻觉纠正方法。视觉语言模型，简称 VLM，指同时接收图像和文本并生成文本回答的模型。它的核心不是“再做一次普通 RLHF”，而是把偏好学习从整句级别推进到片段级别：人工标注者指出回答里具体哪个 token 或 span 是幻觉，再用这些位置构造更密集的训练信号。

普通偏好学习通常只知道：

> 回答 A 比回答 B 好。

RLHF-V 更接近：

> 回答 B 里“2 个螺丝”的“2”是错的，图中实际是 3 个螺丝。

新手版解释：不是让模型“整句话重写得更好”，而是让它知道“错的是哪个词”。

玩具例子：图片里有 3 个螺丝，问题是“图中有几个螺丝？”模型回答“有 2 个螺丝”。整句偏好只会把这句话整体判为差；RLHF-V 会把“2”标为幻觉片段，并把正确回答“3”作为偏好方向。

| 对比项 | 整句偏好学习 | RLHF-V 细粒度纠错 |
|---|---|---|
| 监督粒度 | 整个回答 winner/loser | token/span 级幻觉片段 |
| 标注成本 | 单条较低，但信号粗 | 单条较高，但信息密 |
| 训练信号密度 | 稀疏，只知道整体好坏 | 密集，知道错在何处 |
| 适用错误类型 | 整体质量、礼貌性、完整性 | 颜色、数量、位置、状态、对象属性 |
| 典型问题 | “这句不好”但不知道哪里不好 | “错词挨罚更重” |

最简地看，目标从整句对比变成了 span/token 加权对比：

$$
\text{sequence score}=\sum_t \log p(y_t),\quad
\text{fine-grained score}=\sum_t w_t \log p(y_t)
$$

其中 $w_t$ 是 token 权重。被标注为幻觉的片段权重更高，模型更新时会更关注这些位置。

---

## 问题定义与边界

视觉幻觉是指模型生成了与图像内容不一致的描述。白话解释：图里没有的东西，模型说有；图里是红色，模型说蓝色；图里有 3 个物体，模型说 2 个。

RLHF-V 主要解决的是图文对齐下的局部事实错误。这里的“局部”很重要：它不是重新定义所有对话质量，也不是解决所有推理失败，而是针对视觉输入中可核验的事实片段进行纠错。

例子：同一张图片里，按钮明明是红色，模型说“蓝色按钮”，这是典型视觉幻觉。若模型只是回答太啰嗦、语气不够自然，这不属于 RLHF-V 的主要优化目标。新手版解释：它管的是“看图说错了什么”，不是“说话风格好不好”。

符号定义如下：

| 符号 | 含义 |
|---|---|
| $x=(I,q)$ | 多模态输入，$I$ 是图像，$q$ 是问题 |
| $y^+$ | 偏好回答，通常是正确或更少幻觉的回答 |
| $y^-$ | 拒绝回答，通常包含幻觉片段 |
| $\pi_\theta$ | 当前正在训练的模型 |
| $\pi_{\text{ref}}$ | 参考模型，通常是训练前冻结的模型 |
| $\beta$ | DPO 温度系数，控制偏好差异的放大程度 |
| $w_t$ | 第 $t$ 个 token 的训练权重 |

边界可以这样划分：

| 任务类型 | 输入是否包含图像 | 监督信号 | 是否适合 RLHF-V |
|---|---:|---|---|
| 视觉幻觉纠错 | 是 | 图像事实、正确回答、幻觉 span | 适合 |
| 一般偏好对齐 | 可有可无 | 整体偏好、风格、帮助性 | 不一定适合 |
| 纯文本生成优化 | 否 | 文本质量、逻辑、格式 | 通常不适合 |

真实工程例子：工业质检助手读取零件图、仪表盘、截图或票据。模型常把“3 个螺丝”说成“2 个螺丝”，把“未连接”说成“已连接”，把“左侧告警灯亮起”说成“右侧告警灯亮起”。这些错误对业务结果有直接影响，也适合用细粒度标注纠正。

---

## 核心机制与推导

标准多模态 DPO 的思路是：给定同一个图文输入 $x=(I,q)$，让模型更偏向 $y^+$，更远离 $y^-$。DPO，Direct Preference Optimization，白话解释是“直接用偏好对训练模型，不额外训练奖励模型”。

标准多模态 DPO 目标可以写成：

$$
L_{\text{DPO}}=-\log \sigma\Big(\beta[(\log \pi_\theta(y^+|x)-\log \pi_{\text{ref}}(y^+|x))-(\log \pi_\theta(y^-|x)-\log \pi_{\text{ref}}(y^-|x))]\Big)
$$

其中 $\sigma$ 是 sigmoid 函数。这个公式衡量的是：当前模型相对参考模型，是否更支持好回答 $y^+$，更不支持坏回答 $y^-$。

问题在于，整句级 DPO 对局部幻觉不够精准。回答“有一个蓝色杯子”时，如果图中杯子实际是红色，只有“蓝色”是错的。整句训练会把“有”“一个”“杯子”等正确部分也卷进同一个偏好信号里。新手版解释：普通 DPO 是整句一起被罚，RLHF-V 希望“错词挨罚更重”。

工程上可以把 RLHF-V 的细粒度纠错抽象为加权 token logprob：

$$
g_\theta(y;x)=\frac{1}{W}\sum_t w_t \log \pi_\theta(y_t|x,y_{<t}),\quad
L_{\text{DDPO}}=-\log \sigma\Big(\beta[(g_\theta(y^+;x)-g_{\text{ref}}(y^+;x))-(g_\theta(y^-;x)-g_{\text{ref}}(y^-;x))]\Big)
$$

这里 $g_\theta(y;x)$ 是加权后的回答分数，$W=\sum_t w_t$ 是权重和。$w_t$ 对应纠错 span/token：普通 token 可以取 1，被标为幻觉或对应修正位置的 token 可以取更大的权重，例如 3 或 5。这里把论文中的 segment-level correction 表达成 token/span 加权，是对方法思路的工程化抽象，不是逐字复刻论文公式。

小数值例子如下，设 $\beta=1$，$d$ 表示括号中的偏好差值：

| 情况 | 偏好差值 $d$ | 损失 $-\log\sigma(d)$ | 含义 |
|---|---:|---:|---|
| 纠错前 | 0 | 0.693 | 模型分不清好坏回答 |
| 权重生效后 | 1 | 0.313 | 正确片段相对优势提升 |

这说明加大幻觉片段权重后，训练信号会集中到关键错误处，比如“蓝色”“2”“左侧”“已打开”这些词。

多模态场景还有一个特殊风险：模型可能学成“只看文本偏好”。MDPO 相关工作指出，多模态 DPO 可能弱化图像条件，导致模型不用认真看图也能区分偏好答案。因此训练和评估都要做去图像消融：把图像遮掉、换掉或置空，看模型性能是否明显下降。如果不下降，说明模型可能学到的是文本模式，而不是视觉对齐。

---

## 代码实现

实现 RLHF-V 式训练时，核心模块有三类：数据读取、纠错 span 转 token 权重、加权偏好损失计算。训练样本至少包含图片、问题、偏好回答、拒绝回答和纠错区间。

| 输入字段 | 中间表示 | 损失项 | 训练结果 |
|---|---|---|---|
| image, question | $x=(I,q)$ | 条件输入 | 模型必须依赖图像回答 |
| chosen | $y^+$ token 序列 | 正向偏好分数 | 提高正确回答概率 |
| rejected | $y^-$ token 序列 | 负向偏好分数 | 降低幻觉回答概率 |
| spans | token 权重 $w_t$ | 加权 DDPO | 错误片段更新更强 |

下面是一个可运行的最小 Python 例子。它不包含真实 VLM，只演示 DDPO 的数据结构和损失计算方式。

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def load_dataset():
    return [{
        "image": "screws.jpg",
        "question": "图中有几个螺丝？",
        "chosen": ["有", "3", "个", "螺丝"],
        "rejected": ["有", "2", "个", "螺丝"],
        "error_spans": [(1, 2)]  # rejected 中 token [1,2) 是幻觉片段：“2”
    }]

def build_token_weights(tokens, spans, base=1.0, error_weight=4.0):
    weights = [base] * len(tokens)
    for start, end in spans:
        for i in range(start, end):
            weights[i] = error_weight
    return weights

def compute_weighted_logprob(token_logprobs, weights):
    assert len(token_logprobs) == len(weights)
    total_weight = sum(weights)
    return sum(lp * w for lp, w in zip(token_logprobs, weights)) / total_weight

def ddpo_loss(g_chosen, g_rejected, g_ref_chosen, g_ref_rejected, beta=1.0):
    d = beta * ((g_chosen - g_ref_chosen) - (g_rejected - g_ref_rejected))
    return -math.log(sigmoid(d))

def train_step(sample):
    chosen_weights = build_token_weights(sample["chosen"], [])
    rejected_weights = build_token_weights(sample["rejected"], sample["error_spans"])

    # 假设这些 logprob 来自当前 VLM 和参考 VLM。
    theta_chosen_lp = [-0.2, -0.1, -0.2, -0.1]
    theta_rejected_lp = [-0.2, -1.2, -0.2, -0.1]
    ref_chosen_lp = [-0.3, -0.5, -0.2, -0.2]
    ref_rejected_lp = [-0.3, -0.6, -0.2, -0.2]

    g_chosen = compute_weighted_logprob(theta_chosen_lp, chosen_weights)
    g_rejected = compute_weighted_logprob(theta_rejected_lp, rejected_weights)
    g_ref_chosen = compute_weighted_logprob(ref_chosen_lp, chosen_weights)
    g_ref_rejected = compute_weighted_logprob(ref_rejected_lp, rejected_weights)

    loss = ddpo_loss(g_chosen, g_rejected, g_ref_chosen, g_ref_rejected)
    return loss

sample = load_dataset()[0]
weights = build_token_weights(sample["rejected"], sample["error_spans"])
assert weights == [1.0, 4.0, 1.0, 1.0]

loss = train_step(sample)
assert loss > 0
assert round(-math.log(sigmoid(0)), 3) == 0.693
assert round(-math.log(sigmoid(1)), 3) == 0.313
print(round(loss, 4))
```

真实训练时，`token_logprobs` 应来自视觉语言模型的前向计算。图片不能只是可选附件，而必须进入模型条件输入，否则训练会退化成文本 DPO。新手版解释：先告诉模型“哪里错了”，再让模型在这些位置上学得更快，同时确保它真的在看图。

---

## 工程权衡与常见坑

RLHF-V 的难点通常不在公式，而在数据。监督太稀、负样本太容易、标注边界不一致，都会让训练信号失真。

| 常见坑 | 后果 | 规避策略 |
|---|---|---|
| 只做整句 winner/loser | 信号太稀，不知道错在哪 | 保留 span/token 级纠错 |
| 负样本太容易 | 模型不用看图也能分胜负 | 构造 hard negative，错误只差颜色、数量、位置 |
| DPO 压低 chosen 概率 | 模型变保守，回答变短 | 加 reward anchor、KL 约束或监控 chosen logprob |
| 退化成语言偏好 | 图像条件被弱化 | 做去图像消融和换图测试 |
| 标注边界不一致 | 同类错误权重混乱 | 先定义幻觉片段标准，再做一致性抽检 |
| 只看总分 | 幻觉率可能不降反升 | 同时看 hallucination rate、CHAIR、人工复核 |

负样本构造尤其关键。例子：如果负样本只是把正确答案随机打乱，模型几乎不用看图就能知道它不好，训练会变成“文本形式辨别”，而不是“视觉纠错”。新手版解释：题目太容易时，模型学到的是“猜偏好”，不是“看图”。

评估也不能只看一个综合分。至少需要同时观察：

| 指标 | 作用 |
|---|---|
| hallucination rate | 直接衡量幻觉比例 |
| CHAIR 类指标 | 检查图像描述中对象幻觉 |
| 人工复核 | 发现自动指标漏掉的属性错误 |
| 去图像消融 | 判断模型是否真正依赖图像 |
| 分类型错误率 | 分别统计颜色、数量、位置、状态错误 |

工程上还要控制权重大小。$w_t$ 太小，和普通 DPO 差别不明显；$w_t$ 太大，模型可能过度修正少数片段，牺牲整体流畅性。更稳妥的做法是从较小倍率开始，例如 2 到 5，并按错误类型分别统计收益。

---

## 替代方案与适用边界

RLHF-V 不是万用对齐工具。它适合局部事实纠错，不一定适合大规模风格偏好、纯文本任务或开放式多步推理。

例子：如果目标是“让模型回答更有礼貌”，用 RLHF-V 没有必要；如果目标是“读仪表盘并指出错误数值”，RLHF-V 更合适。新手版解释：它是“纠错工具”，不是“万用对齐工具”。

| 方法 | 监督信号粒度 | 是否需要 span/token 纠错 | 是否依赖图像 | 适用场景 | 主要局限 |
|---|---|---:|---:|---|---|
| RLHF-V | 细粒度片段 | 是 | 是 | 局部视觉幻觉纠正 | 标注成本较高 |
| 标准 DPO | 整句偏好对 | 否 | 可选 | 一般偏好对齐 | 不知道具体错词 |
| MDPO | 多模态偏好约束 | 否或弱依赖 | 是 | 缓解多模态 DPO 图像弱化 | 实现和评估更复杂 |
| ORPO | 单阶段偏好优化 | 否 | 可选 | 简化 SFT 与偏好训练流程 | 不直接处理幻觉 span |
| IPO | 偏好目标变体 | 否 | 可选 | 稳定偏好优化 | 仍缺少细粒度视觉纠错 |

什么时候不用 RLHF-V：

| 场景 | 原因 |
|---|---|
| 没有稳定幻觉定义 | 标注者无法一致判断 span |
| 任务不依赖局部事实 | 细粒度纠错收益小 |
| 标注成本无法承受 | span 级标注比整句偏好更贵 |
| 主要目标是风格、语气、礼貌性 | 普通偏好对齐更直接 |
| 没有可靠图像评估集 | 无法判断是否真的降低视觉幻觉 |

对于零基础到初级工程师，判断是否使用 RLHF-V 可以抓住一个问题：错误能不能在回答里圈出来，并且能不能通过看图验证。如果能，比如颜色、数量、位置、对象状态，就适合；如果不能，比如“更自然”“更有帮助”“更像专家”，就不优先用 RLHF-V。

---

## 参考资料

| 类型 | 链接 | 用途 |
|---|---|---|
| 论文 | https://huggingface.co/papers/2312.00849 | RLHF-V 方法动机与核心定义 |
| 代码仓库 | https://github.com/RLHF-V/RLHF-V | 官方实现入口与训练流程参考 |
| 数据集卡 | https://huggingface.co/datasets/openbmb/RLHF-V-Dataset | 数据规模与 preference pairs 信息 |
| 相关方法 | https://huggingface.co/papers/2305.18290 | DPO 原始方法与偏好优化公式来源 |
| 相关方法 | https://arxiv.org/pdf/2406.11839 | MDPO 与多模态 DPO 图像条件弱化风险 |
