## 核心结论

SynthID 的核心不是“往文本里塞一个特殊词”，而是在生成过程中对 **logits** 做极小幅度、可重复的偏置。logits 可以理解为“模型给每个候选词打的原始分数，softmax 前的分值”。Google DeepMind 的做法是把一个由密钥驱动的水印信号，分散到很多步采样里，让单个词几乎看不出异常，但整段文本在统计上会呈现稳定偏移。

更具体地说，SynthID 由两个协同部分组成：

| 组件 | 做什么 | 是否需要访问原模型 |
|---|---|---|
| 插入器 | 在生成时按密钥微调 logits，让某些 token 更容易被采样 | 需要接在采样流程上 |
| 检测器 | 只看输出文本和同一密钥，计算统计得分判断是否带水印 | 不需要访问原模型 |

这件事成立的关键在于：水印不是“可见标记”，而是“统计签名”。统计签名的意思是，单个位置看不明显，但把整段文本所有位置的得分汇总后，会比未水印文本显著更高。一个典型形式是：

$$
{\rm Score}(x)=\frac{1}{mT}\sum_{t=1}^{T}\sum_{\ell=1}^{m} g_\ell(x_t, r_t)
$$

其中，$T$ 是文本长度，$m$ 是 tournament 的层数，$x_t$ 是第 $t$ 步最终采样到的 token，$r_t$ 是由密钥和上下文导出的伪随机状态，$g_\ell$ 是第 $\ell$ 层的水印打分函数。若文本来自带水印采样，平均得分通常会高于未水印文本的期望值。

新手版本可以这样理解：每个词候选本来就有模型分数，SynthID 再偷偷加一个“只有持钥匙的人才知道的附加分”。这个附加分加得很小，不足以让句子变形，但会让整段输出在秘密评分表上更像“被签过名的文本”。

一个最小对比表如下：

| 文本类型 | 平均得分趋势 | 是否容易被检测器识别 |
|---|---:|---|
| 未水印文本 | 接近随机期望 | 否 |
| 带水印文本 | 高于随机期望 | 是 |

---

## 问题定义与边界

SynthID 解决的问题可以精确定义为：

> 在不改基础模型训练、不显著损伤文本质量的前提下，为语言模型输出加入一个可验证的统计水印，使服务方之后只凭文本和密钥，就能判断这段内容是否由自己的水印采样流程生成。

这里有几个边界必须说清楚。

第一，它不是“证明文本一定由某个模型写的”。它证明的是：文本是否更像经过某个特定水印机制生成。也就是说，它识别的是“水印流程的痕迹”，不是“作者身份”的绝对证明。

第二，它通常不要求检测器访问模型内部。检测阶段的目标是脱离原始 LLM，只根据文本、密钥和检测逻辑进行判断。这一点很重要，因为真实工程里，检测往往发生在离线审计、平台巡检或跨系统归因场景中，不能指望每次都调原模型。

第三，它主要适用于自然语言生成场景，尤其是持续采样、篇幅较长、表达自由度较高的内容。对特别短的回答、强约束格式输出、逐字翻译、事实密集问答，水印信号会更弱。

可以把问题看成一句话：我想在每次生成的句子里悄悄夹带一个秘密签名，但这个签名不能被普通读者看出来，也不能明显降低回答质量。

插入器与检测器的输入输出可以整理成下表：

| 模块 | 输入 | 输出 |
|---|---|---|
| 插入器 | 当前上下文、模型 logits、私钥、步数 $t$ | 轻微偏置后的 logits 或采样结果 |
| 检测器 | 输出文本、私钥、检测配置 | 一个得分和真假判定 |

实际使用时还要接受这些边界条件：

- 无需修改基础模型权重，重点在采样层。
- 无需检测器访问模型内部，重点在可离线验证。
- 适合自然语言、多轮对话、长文本等有足够统计长度的输出。
- 需要兼容线上低延迟服务，因此偏置计算不能太重。
- 不承诺对强力改写、人工重写、翻译后文本仍保持同等检出率。
- 不等于版权证明，也不等于内容真实性证明。

这里有一个“玩具例子”。假设模型在某一步只考虑四个词：`mango`、`papaya`、`apple`、`banana`。模型原始上最想选 `apple`，但密钥驱动的水印函数会对这一轮稍微偏爱 `mango`。如果偏置非常小，模型大多数时候仍会保持句子自然；但很多轮之后，被偏爱的 token 在统计上会比随机期望出现得更频繁。检测器不需要知道模型为什么这样写，只需要复现“这一轮密钥本来偏爱谁”，然后统计最终选中的词是否长期偏向这些高分项。

真实工程例子是对话模型部署。平台把水印逻辑接到采样器后面，用户看见的仍是正常回答；但平台后续可以对投诉内容、滥用内容、外部流传内容做抽样检测，判断它是否大概率来自自家模型输出。

---

## 核心机制与推导

SynthID 的关键机制可以拆成两层：生成时如何嵌入，检测时如何判别。

先说生成。很多传统水印方法会直接把词表分成“绿名单”和“红名单”，然后偏向绿名单采样。这种方法简单，但有时会带来明显分布扭曲。SynthID 更进一步，引入 **tournament sampling**。tournament 可以理解为“分层比赛式筛选”：候选 token 不只经历一次偏置，而是在多层局部比较中逐步受到水印信号影响。

形式化地说，在第 $t$ 个生成步，系统按密钥和上下文导出一组伪随机信号：

$$
g_1(\cdot, r_t), g_2(\cdot, r_t), \ldots, g_m(\cdot, r_t)
$$

这组 $g$ 值并不是固定常数，而是依赖密钥、步数、上下文甚至滑动窗口状态。白话解释：同一个词在不同位置不一定总是高分，只有知道密钥的人才能复原“这一轮谁该高分”。

在 tournament 流程里，每一层 $\ell$ 都会对候选 token 的排序或局部胜出概率施加影响。结果是：被多层 $g_\ell$ 一致偏爱的 token，更容易进入最终采样。这样做的好处是把信号“摊薄”到多层决策里，而不是粗暴地一次加大偏置。

检测端则反过来。给定文本 $x=(x_1,\ldots,x_T)$，检测器按同一密钥重建每一步的 $g$ 值，并计算平均分：

$$
{\rm Score}(x)=\frac{1}{mT}\sum_{t=1}^{T}\sum_{\ell=1}^{m}g_\ell(x_t,r_t)
$$

如果 $x$ 是随机未水印文本，那么 $x_t$ 与秘密打分函数之间没有系统性相关性，得分应接近某个基线期望 $\mu_0$。如果 $x$ 是带水印采样生成，那么采样过程更频繁地选中了高 $g$ 值 token，于是

$$
\mathbb{E}[{\rm Score}(x)\mid \text{watermarked}] > \mathbb{E}[{\rm Score}(x)\mid \text{unwatermarked}] = \mu_0
$$

因此检测器只需要设定阈值 $\tau$，当 ${\rm Score}(x) > \tau$ 时判为带水印。

一个最小“玩具例子”可以直接算。令 $m=2,\ T=1$，只有一轮生成，候选词有 `mango` 和 `papaya`：

| token | $g_1$ | $g_2$ | 总和 |
|---|---:|---:|---:|
| mango | 1 | 0 | 1 |
| papaya | 0 | 1 | 1 |

上表太对称，不利于说明偏置。我们改成更接近“比赛”的版本：

| token | $g_1$ | $g_2$ | 平均得分 |
|---|---:|---:|---:|
| mango | 1 | 0 | 0.5 |
| papaya | 0 | 0 | 0.0 |

假设插入器让 `mango` 更容易在第 1 层胜出，最终选中 `mango`。那么文本得分就是：

$$
{\rm Score}(x)=\frac{1+0}{2\times 1}=0.5
$$

若未水印文本在同一密钥配置下的期望只有 $0.25$，那么 $0.5$ 就更可能落在“带水印”区间。这不是因为 `mango` 这个词本身特殊，而是因为它在当前密钥下恰好是“该轮被偏爱的 token”。

再看一个更接近真实工程的流程表：

| 步骤 | 输入 | 动作 | 输出 |
|---|---|---|---|
| 1 | 上下文、私钥、步数 $t$ | 生成本轮伪随机状态 $r_t$ | 本轮水印种子 |
| 2 | 候选 token 集合 | 计算每层 $g_\ell(token, r_t)$ | 每个 token 的分层水印分数 |
| 3 | 原始 logits | 将分层偏置注入 tournament / 采样流程 | 带水印采样分布 |
| 4 | 选中 token $x_t$ | 追加到输出序列 | 下一轮上下文 |
| 5 | 检测时重放 | 根据同样密钥复算所有 $g$ | 得到总分 ${\rm Score}(x)$ |

这里的关键推导不是复杂数学，而是统计分离。只要带水印文本和未水印文本的得分分布有足够间隔，检测器就能用阈值把两者区分开。文本越长，$T$ 越大，平均得分的波动通常越小，区分越稳定；文本越短，方差越大，误判风险越高。

---

## 代码实现

下面给一个最小可运行的 Python 示例。它不是 Google 论文中的完整实现，而是一个“能跑通机制”的教学版：用密钥驱动的伪随机函数给 token 加偏置，生成后再按同样规则做检测。

```python
import hashlib
import math
import random

VOCAB = ["mango", "papaya", "apple", "banana"]

def stable_rand01(key: str, context: str, step: int, layer: int, token: str) -> float:
    data = f"{key}|{context}|{step}|{layer}|{token}".encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    # 映射到 [0, 1)
    return int(digest[:8], 16) / 16**8

def g_value(key: str, context: str, step: int, layer: int, token: str) -> int:
    # 教学版 g: 以 0/1 表示该 token 在这一层是否被密钥偏爱
    return 1 if stable_rand01(key, context, step, layer, token) > 0.5 else 0

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]

def sample_from_probs(probs, rng):
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1

def watermark_generate(base_logits_per_step, key: str, m: int = 2, alpha: float = 0.8, seed: int = 0):
    rng = random.Random(seed)
    context_tokens = []
    output_tokens = []

    for step, base_logits in enumerate(base_logits_per_step):
        context = " ".join(context_tokens[-4:])  # 教学版滑动窗口上下文
        logits = list(base_logits)

        # 插入器：按共同密钥产生 g 值，并把它们转成很小的 logits 偏置
        for i, token in enumerate(VOCAB):
            bonus = sum(g_value(key, context, step, layer, token) for layer in range(m))
            logits[i] += alpha * bonus / m

        probs = softmax(logits)
        idx = sample_from_probs(probs, rng)
        token = VOCAB[idx]
        output_tokens.append(token)
        context_tokens.append(token)

    return output_tokens

def detect_score(tokens, key: str, m: int = 2):
    total = 0.0
    context_tokens = []

    for step, token in enumerate(tokens):
        context = " ".join(context_tokens[-4:])
        total += sum(g_value(key, context, step, layer, token) for layer in range(m))
        context_tokens.append(token)

    return total / (m * len(tokens))

def is_watermarked(tokens, key: str, threshold: float, m: int = 2):
    score = detect_score(tokens, key, m=m)
    return score > threshold, score

# 固定一个简单“模型”：每一步原始 logits 都偏向 apple，但不会绝对压制其它词
base_logits = [
    [0.3, 0.2, 1.1, 0.1],
    [0.4, 0.1, 1.0, 0.2],
    [0.2, 0.3, 1.2, 0.1],
    [0.3, 0.4, 0.9, 0.2],
    [0.2, 0.2, 1.0, 0.3],
    [0.1, 0.3, 1.1, 0.2],
]

key = "secret-demo-key"
tokens = watermark_generate(base_logits, key=key, m=3, alpha=1.0, seed=42)
flag, score = is_watermarked(tokens, key=key, threshold=0.45, m=3)

print("generated:", tokens)
print("score:", round(score, 4), "watermarked:", flag)

# 同一文本在错误密钥下，得分应显著下降或至少不稳定
wrong_flag, wrong_score = is_watermarked(tokens, key="wrong-key", threshold=0.45, m=3)

assert len(tokens) == len(base_logits)
assert 0.0 <= score <= 1.0
assert score != wrong_score
assert flag or score > wrong_score
```

这段代码对应了两个最核心动作。

第一，插入器在采样循环里生成 $g$ 值，再把 $g$ 变成 logits 的小偏置：

```python
bonus = sum(g_value(key, context, step, layer, token) for layer in range(m))
logits[i] += alpha * bonus / m
```

其中 `alpha` 控制水印强度。强一点更容易检测，弱一点更不影响质量。

第二，检测器不需要模型，只需要重放同样的密钥调度并累计分数：

```python
total += sum(g_value(key, context, step, layer, token) for layer in range(m))
return total / (m * len(tokens))
```

如果只是教学，你可以把它理解为“生成时偷偷给某些词加分，检测时看看最终被选中的词是不是总踩中加分点”。

一个真实工程例子会复杂得多。线上系统通常不会直接遍历整个词表做粗糙加分，而是把水印逻辑接进更精细的采样器中，配合滑动窗口种子、上下文去重、重复文本掩码、动态阈值等机制，减少对可读性的影响并提升检测稳健性。

---

## 工程权衡与常见坑

真正难的部分不是“能不能做出水印”，而是“在不伤质量的前提下做出足够稳的水印”。

先看几组核心权衡：

- 水印强度越高，检测越稳，但越可能扭曲文本分布。
- tournament 层数越多，信号越平滑，但计算开销更大。
- 阈值越严格，误报越低，但漏报会升高。
- 公开检测规则越多，第三方越容易验证，但攻击者也更容易反制。

Google 在公开材料里强调过偏向 **non-distortionary** 配置，即尽量不改变用户体感质量。这类配置常见做法包括多层 tournament、滑动窗口随机种子、重复上下文掩码等。白话解释：不是猛推一个词，而是把很多小偏置分散到各步决策里。

短文本是第一个常见坑。因为 $T$ 小，平均分的方差大，本来带水印的文本也可能刚好没踩中太多高分 token。结果就是检测不稳定。工程上通常会对极短文本直接标记为“不足以判定”，而不是硬下结论。

第二个坑是改写和翻译。SynthID 水印依赖 token 级统计结构。如果用户把一段文本彻底改写成另一种说法，甚至翻译成另一种语言，原先由密钥驱动的 token 选择偏好会被大量洗掉。这里要接受一个事实：它不是鲁棒到任意编辑的数字指纹。

第三个坑是事实密集型输出。比如 API 返回格式化答案、代码补全、固定术语密集内容。这时模型本来就几乎只能选少数几个词，插入器可操作空间变小，水印容量下降。

第四个坑是重复上下文。如果模型在模板化回复、列表枚举、代码片段中大量重复模式，固定密钥调度可能导致局部信号不均匀，因此实际系统会加上下文掩码，避免在重复区域重复施加强信号。

常见问题和处理方式可以整理为下表：

| 常见坑 | 为什么会出问题 | 常见规避措施 |
|---|---|---|
| 短文本 | $T$ 太小，统计波动大 | 设最小长度门槛，动态阈值，多轮聚合 |
| 翻译 | token 序列被重写 | 限定原语言检测，结合外部元数据 |
| 彻底改写 | 原 token 偏好被洗掉 | 只把水印当“来源线索”而非绝对证明 |
| 事实密集回复 | 候选空间太窄 | 提升层数 $m$，降低单次偏置，延长可检测文本 |
| 重复模板内容 | 局部信号失衡 | 上下文掩码，跳过重复片段 |
| 低延迟场景 | 每步计算预算有限 | 预计算 key schedule，限制层数，优化缓存 |

这里给一个“真实工程例子”。假设一个对话系统日均生成大量客服回复，平台希望后续识别站外传播的机器生成内容。若直接把偏置设得很强，检测是容易了，但用户可能会感觉措辞发僵。更可行的方法是：

- 用较多层的 tournament 把信号分散。
- 用滑动窗口密钥避免固定模板区域过强偏置。
- 对短回复只给“低置信度”标签。
- 对长对话合并多轮文本再检测。

这样做的结果不是“每条都百分之百检出”，而是在真实服务约束下，获得一个足够实用的来源判断系统。

---

## 替代方案与适用边界

SynthID 不是唯一的水印路线。至少还有两类常见替代方案：可见 token 水印，以及外部模型指纹。

**可见 token 水印** 的思路最直接，比如在文本中插入特定模式、固定短语、隐藏字符或特定格式习惯。优点是易解释，缺点也明显：容易被发现、容易被删除，而且常常破坏用户体验。

**外部模型指纹** 指的是不在生成时植入水印，而是训练一个外部分类器，从风格、词频、句法等内容特征上判断“像不像某模型生成”。优点是对未植入水印的历史内容也可能适用；缺点是它本质上是归因模型，不是密码学意义上的持钥验证，更容易受域迁移和对抗改写影响。

三者可以比较如下：

| 方案 | 核心思路 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| SynthID | 在采样时注入隐蔽统计信号 | 隐蔽、体验好、可持钥检测 | 短文本和改写后效果下降 | 高质量对话、托管生成服务 |
| 可见 token 水印 | 明示性标记或固定模式 | 简单、易审计 | 易发现、易删除、影响观感 | 需要明确告知用户的场景 |
| 外部模型指纹 | 从内容特征判断来源 | 不依赖生成时接入 | 误报更难控、易受域变化影响 | 历史数据分析、辅助归因 |

为什么还要讨论“公开 vs 半公开”检测器配置？因为这会直接影响误报和隐私控制。

- 公开检测器：规则更透明，第三方更容易独立验证；但攻击者也更容易研究如何规避。
- 半公开检测器：外部只知道部分接口或有限规则，真正密钥和细节由平台保留；这样更利于控制误报与反制攻击，但独立可审计性会下降。

如果场景是完全托管的在线服务，平台通常更倾向半公开或私有检测器，因为它关心的是大规模内部审计与滥用追踪。如果场景是跨机构协作、平台间联合治理，则可能需要更透明的验证接口，以便第三方接受检测结果。

要明确一个边界：SynthID 适合做“来源信号增强”，不适合做“内容真伪裁决”的唯一依据。它可以回答“这段内容是否更像来自某个带水印的生成流程”，但不能单独回答“这段话是否正确”“是否恶意”“是否应该处罚”。

---

## 参考资料

- 论文
- Nature 2024, *Scalable watermarking for identifying large language model outputs*：核心原理来源，重点看 logits 水印、tournament sampling、检测分数定义与实验。
- 论文 Supplementary Materials：补充实现细节、参数配置、分布推导与鲁棒性分析。

- 官方文档
- Google Responsible GenAI 的 SynthID 文档：解释 SynthID 的产品化接口、能力边界、适用限制，以及在文本场景中的部署方式。
- Google SynthID Detector 相关官方介绍：聚焦检测门户、实际部署与多模态扩展，适合理解工程落地视角。

- 补充阅读
- Google/DeepMind 面向开发者的说明材料与博客：适合对“为什么要做非扭曲配置”“为什么短文本更难检出”建立直观认识。

新手版本可以这样记：Nature 论文讲“原理为什么成立”，Google 官方文档讲“系统怎么接到真实服务里”。
