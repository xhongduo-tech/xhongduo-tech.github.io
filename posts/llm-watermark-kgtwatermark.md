## 核心结论

KGWatermark，通常简称 KGW，是一种**解码期水印**：不改模型参数，不重新训练，只在“下一个 token 怎么采样”这一步做轻微偏置。解码期的意思是，模型已经算出候选词概率后，再额外调整一次分布。

它的核心做法是：在每个生成位置，用“前文 + 私钥”生成一个伪随机种子，再把词表临时分成“绿表”和“红表”。伪随机的意思是“看起来随机，但只要种子相同就一定能重现”。采样时，系统对绿表 token 加一点分数，所以最终文本里绿词比例会高于自然文本。检测端只要拿同样的密钥和同样的前文，重建每一步的绿表，再统计绿词占比是否异常偏高，就能判断文本是否带水印。

给零基础读者的玩具例子：把它想成“每写一个词前都摇一次受密钥控制的骰子”。骰子决定本轮哪些词是“幸运词”，系统只是稍微偏向幸运词，不是强制只能选幸运词。人读起来通常感觉不到差异，但检测器能通过统计幸运词出现频率判断这段文本是否用了同一套规则。

当绿表大小设为词表的一半时，自然文本的绿词比例基线通常按 $p=0.5$ 建模。检测常用的统计量是

$$
z=\frac{G-n\cdot p}{\sqrt{n\cdot p(1-p)}}
$$

其中 $n$ 是参与检测的 token 数，$G$ 是其中落在绿表里的 token 数。如果 $z$ 足够大，就拒绝“这是自然文本”的原假设。

| 文本来源 | 绿词期望比例 | 检测直觉 |
| --- | ---: | --- |
| 自然文本或无水印文本 | 约 0.50 | 接近随机波动 |
| KGW 水印文本 | 常高于 0.50，例如 0.60 到 0.70 | 绿词偏多，可做统计拒绝 |

---

## 问题定义与边界

KGW 要解决的问题不是“证明某句话一定由某个模型写出”，而是更窄的目标：**在尽量不伤害生成质量的前提下，给模型输出植入一个统计上可检出的信号**。统计信号的意思是，单个词看不出来，但一段文本放在一起能看出分布偏差。

边界要说清楚：

| 输入类型 | 特征 | 检测可靠性 |
| --- | --- | --- |
| 高熵上下文 | 候选 token 很多，多个写法都合理 | 高 |
| 低熵上下文 | 几乎只有一个自然续写，例如固定短语、公式、专有名词 | 低 |
| 被轻度改写文本 | 有部分原 token 或短片段保留 | 中 |
| 被重写或多源拼接文本 | 原有局部模式被稀释 | 降低，但可累积证据 |

这里的**熵**可以先理解成“下一步到底有多少种合理写法”。高熵时，系统更容易悄悄把概率往绿词方向推；低熵时，模型本来就几乎只能选一个词，水印没什么可操作空间。

再给一个新手版本比喻：把“绿词比例”想成混合液里的某种染料浓度。自然文本浓度大约在 50%，带水印文本会更高。检测就是判断这段样本里浓度是否高到不能用随机波动解释。

所需 token 数和置信度也受信号强度影响。粗略上，$n$ 越大，证据越稳；但如果样本大多来自低熵区域，哪怕字很多，统计力也不一定强。

| 可用高熵 token 数 | 典型检测把握 |
| ---: | --- |
| 16 左右 | 在强信号设置下可形成很强证据 |
| 25 左右 | 常被视为短片段检测的实用下限 |
| 100+ | 对轻度改写、拼接更稳 |

---

## 核心机制与推导

KGW 的写入分两步：**按位置生成绿表**，再**对绿表加偏置**。

设当前位置原始 logits 为 $l_i$。logit 可以理解成“softmax 之前的未归一化分数”。先用前文 $c_{<t}$ 和密钥 $k$ 生成种子：

$$
s_t = H(c_{<t}\Vert k)
$$

再用伪随机数生成器根据 $s_t$ 把词表切成绿表 $G_t$ 与红表 $R_t$。常见设置是绿表占一半。随后把绿表 token 的 logit 增加一个常数 $\delta$：

$$
\tilde{l}_i = l_i + \delta \cdot \mathbf{1}[i \in G_t]
$$

其中 $\mathbf{1}[\cdot]$ 是指示函数，条件成立取 1，否则取 0。新的采样分布为：

$$
\tilde{p}_i=\frac{e^{\tilde{l}_i}}{\sum_j e^{\tilde{l}_j}}
$$

$\delta$ 可理解为“水印硬度”。它越大，绿词越容易被选中，检测越容易；但也越可能让文本变保守、不自然，甚至影响安全对齐行为。

检测端不需要模型参数，只需要密钥、同样的哈希和伪随机规则。对一段长度为 $n$ 的文本，重建每个位置的绿表，统计绿词个数 $G$。若绿表比例基线设为 $p=0.5$，则

$$
z=\frac{G-n\cdot p}{\sqrt{n\cdot p(1-p)}}
$$

如果 $z>T$，例如 $T=4$，就拒绝“自然文本”假设。

玩具例子：假设检测片段有 $n=25$ 个 token，其中 $G=18$ 个落在绿表。则

$$
z=\frac{18-12.5}{\sqrt{25\cdot 0.5\cdot 0.5}}=\frac{5.5}{2.5}=2.2
$$

这说明有偏置，但还不算特别强。若 $n=25,G=20$，则 $z=3.0$，证据更强。真正部署时不会只看一个例子，而会在更长片段或多个片段上累计证据。

真实工程例子：一家对外提供写作 API 的平台，在服务端对回答启用 KGW。内容平台收到举报样本时，不需要访问原模型，只需截取一段约 25 个以上连续 token，按密钥重建绿表并算 $z$。如果多个片段都偏高，可以把证据叠加，作为“该内容大概率来自此类受控输出”的判据。

---

## 代码实现

下面给一个可运行的简化 Python 版本。它不是完整 tokenizer 级实现，但足够说明“哈希生成绿表、对绿词加权、再做 z 检测”的主线。

```python
import hashlib
import math
import random

VOCAB = ["a", "b", "c", "d", "e", "f", "g", "h"]

def stable_seed(prefix_tokens, secret):
    text = "|".join(prefix_tokens) + "||" + secret
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def green_set(prefix_tokens, secret, gamma=0.5):
    rng = random.Random(stable_seed(prefix_tokens, secret))
    vocab = VOCAB[:]
    rng.shuffle(vocab)
    k = int(len(vocab) * gamma)
    return set(vocab[:k])

def watermark_sample(prefix_tokens, secret, logits, delta=0.8):
    greens = green_set(prefix_tokens, secret)
    adjusted = {}
    for tok, logit in logits.items():
        adjusted[tok] = logit + (delta if tok in greens else 0.0)
    return max(adjusted, key=adjusted.get), greens, adjusted

def detect(tokens, secret, gamma=0.5):
    G = 0
    n = 0
    prefix = []
    for tok in tokens:
        greens = green_set(prefix, secret, gamma=gamma)
        if tok in greens:
            G += 1
        n += 1
        prefix.append(tok)
    p = gamma
    z = (G - n * p) / math.sqrt(n * p * (1 - p))
    return G, n, z

# 玩具写入过程
secret = "demo-key"
prefix = []
logits = {tok: 0.0 for tok in VOCAB}
generated = []
for _ in range(20):
    tok, greens, _ = watermark_sample(prefix, secret, logits, delta=1.2)
    generated.append(tok)
    prefix.append(tok)

G, n, z = detect(generated, secret)
assert n == 20
assert G >= 10
assert z >= 0
print(generated, G, n, round(z, 3))
```

实现时的伪码可以压缩成两段：

```text
写入:
seed <- H(prefix || key)
green_set <- RNG(seed)
for token in vocab:
    if token in green_set:
        logit[token] += delta
sample(next_token)
```

```text
检测:
for each generated token:
    seed <- H(prefix || key)
    green_set <- RNG(seed)
    count whether token in green_set
compute z-score
if z > T: detect watermark
```

---

## 工程权衡与常见坑

最大权衡是 $\delta$。它直接决定“信号强度”和“文本自然度”之间的交换。

| $\delta$ 大小 | 检测性 | 流畅性 | 对齐风险 |
| --- | --- | --- | --- |
| 小 | 弱 | 高 | 低 |
| 中 | 平衡 | 较高 | 可控 |
| 大 | 强 | 下降 | 更容易出现 guard attenuation 或 guard amplification |

这里的 **alignment，对齐**，可以先理解成“模型是否同时保持有用、真实、安全”。2025 年的一项研究指出，水印会改变 token 分布，进而影响这些性质，表现为两种退化：
- guard attenuation：更愿意回答，但安全边界变松。
- guard amplification：更谨慎，但帮助性下降。

因此工程上常见做法不是“全程强打水印”，而是：
- 只在高熵位置启用或增强水印。
- 把 $\delta$ 设在中等范围。
- 配合 AR，也就是 Alignment Resampling。

AR 可以理解成“多采几次，再选最平衡的一次”。它不是改水印规则，而是改最终样本选择流程。

```text
用户请求
  -> 生成 2-4 个带水印候选
  -> 用外部 reward model 打分
  -> 选对齐分最高的候选
  -> 返回结果
```

常见坑还有三类：

| 坑 | 现象 | 处理方式 |
| --- | --- | --- |
| 低熵段落过多 | 检测 z 值上不去 | 只统计高熵片段，或拉长样本 |
| 人工改写、拼接多来源 | 局部证据被稀释 | 多片段累计证据，不依赖单段 |
| 密钥或前文不一致 | 检测完全失配 | 服务端固定规范，确保重放条件一致 |

一个容易被忽略的问题是：KGW 更适合“服务端完全控制生成和检测”的闭环场景。如果文本经过多轮摘要、翻译、OCR 或手工清洗，token 序列会被破坏，检测能力就会下降。它是统计溯源工具，不是不可擦除的数字指纹。

---

## 替代方案与适用边界

KGW 的优势是部署快。它不需要训练，不依赖修改模型权重，适合已有 API 直接上线。代价是它依赖密钥和一致的解码环境，本质上更像“私钥驱动的采样偏置”。

| 方法 | 是否需训练 | 对采样影响 | 检测难度 | 适用场景 |
| --- | --- | --- | --- | --- |
| KGWatermark | 否 | 中 | 低到中 | 闭源 API、快速部署 |
| 概率重写类水印 | 通常否 | 中到高 | 中 | 允许后处理改写 |
| 权重内生水印/蒸馏水印 | 是 | 低到中 | 中 | 开放模型、长期方案 |
| Embedding 或隐藏层扰动 | 通常是 | 依实现而定 | 高 | 研究性方案 |

系统流程可以概括为：

```text
API 输出
  -> KGW 写入（prefix + key -> green/red）
  -> 下游传播
  -> 检测端截取连续约 25+ token
  -> 重建 green set
  -> 计算 z
  -> 判断是否超阈值
```

适用边界可以直接记两条：
- 适合：你控制模型服务端、控制密钥、控制解码流程，且目标是做低成本统计检测。
- 不适合：你无法保证密钥保密、前文一致，或者文本会被严重改写、翻译、OCR 重排。

和其他方案相比，KGW 最像“每个位置换一次锁芯的水印器”。它的强项是在线、轻量、可复现；弱项是证据依赖 token 级连续性，抗强对抗编辑能力有限。

---

## 参考资料

| 文献 | 年份 | 核心贡献 |
| --- | ---: | --- |
| Kirchenbauer et al., *A Watermark for Large Language Models* | 2023 | 提出 KGW 机制、软水印写入和 z 检测 |
| Kirchenbauer et al., *On the Reliability of Watermarks for Large Language Models* | 2023 | 研究改写、拼接、混合来源下的可靠性 |
| Verma et al., *Watermarking Degrades Alignment in Language Models: Analysis and Mitigation* | 2025 | 分析水印对对齐的影响，并提出 AR 缓解 |

1. 2023，Kirchenbauer et al.，*A Watermark for Large Language Models*：机制介绍，包含软水印算法与检测统计量。链接：https://ar5iv.labs.arxiv.org/html/2301.10226
2. 2023，Kirchenbauer et al.，*On the Reliability of Watermarks for Large Language Models*：讨论人类改写、模型改写、长文混入后的可检测性。链接：https://huggingface.co/papers/2306.04634
3. 2025，Verma et al.，*Watermarking Degrades Alignment in Language Models: Analysis and Mitigation*：指出 guard attenuation / amplification，并提出 Alignment Resampling。链接：https://huggingface.co/papers/2506.04462
