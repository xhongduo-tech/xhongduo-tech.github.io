## 核心结论

模型水印的目标，是给 LLM 输出加上一种“人眼几乎看不见、机器可以统计检出”的痕迹。对文本水印来说，重点不是让读者看到某个隐藏符号，而是让部署方在事后能回答一个更具体的问题：

这段文本，是否大概率来自某个带水印的生成系统。

最经典的一类方法是绿色 token 水印。它的基本做法是：在每一步生成前，用密钥把当前可选词表拆成绿色集合 $G_t$ 和红色集合 $R_t$，然后只对绿色 token 的 logit 加一个小偏置 $\delta$。logit 可以理解为 softmax 之前的原始分数。这个偏置是软性的，不会强行规定“下一词必须是什么”，只是让模型在多个合理候选之间更偏向绿色 token。

检测端不需要访问模型参数，只要拿到同一把密钥，就能按相同规则重建每一步的绿/红划分，再统计文本中绿色 token 的比例。如果绿色比例显著高于无水印时的期望值，就判定这段文本更可能来自带水印的模型，而不是普通采样结果。

一个新手可理解的玩具例子如下。假设一段 500 token 的文本，在无水印时，绿色 token 的期望占比约为 50%，也就是平均大约 250 个；加入水印后，这个比例被推到 75%，也就是约 375 个。500 次采样里多出 125 个绿色 token，不太可能只是随机波动，因此检测器可以用 $Z$ 检验把它和“普通文本”区分开。

| 指标 | 人类写作/无水印文本 | 绿色 token 水印文本 |
| --- | --- | --- |
| 绿 token 期望占比 | 约 50% | 高于 50% |
| 可见性 | 人眼不可见 | 人眼仍不可见 |
| 检测方式 | 无稳定统计特征 | 统计绿 token 偏差 |
| 是否依赖密钥 | 不适用 | 依赖 |
| 是否抗改写 | 不适用 | 弱，改写后可能失效 |

如果只记一条流程，可以记下面这条：

生成上下文 $\rightarrow$ 用密钥划分绿/红 token $\rightarrow$ 给绿 token 加偏置 $\delta$ $\rightarrow$ 采样输出 $\rightarrow$ 统计绿色比例并做显著性检验。

这里的“可检测性”本质上是一种统计偏差可复现，不是肉眼可见标记，也不是在句子里埋入某个固定短语。

---

## 问题定义与边界

这里要解决的问题，不是“证明文本一定由 AI 写成”，而是“让模型提供方对自己生成的文本保留可验证痕迹”。这是一种来源归因问题，不是语义理解问题，也不是内容真伪判断问题。

换句话说，水印回答的是：

- 这段文本是否像某个带密钥的生成过程产物。
- 不是：这段文本是否由任何 AI 生成。
- 更不是：这段文本内容是否正确。

绿色 token 水印通常想同时满足四个条件：

| 约束 | 是否满足 | 说明 |
| --- | --- | --- |
| 对文本流畅度影响小 | 基本满足 | 偏置是软性的，不是强制替换 |
| 检测端不必拿到模型参数 | 满足 | 只需密钥和检测程序 |
| 外部观察者难以直接绕过 | 部分满足 | 不知道密钥时较难精确规避 |
| 经过改写、翻译后仍稳定存在 | 不满足 | token 级水印会随措辞变化而消失 |

先看一个直观例子。假设模型当前要表达“增长”，候选 token 里有 `increase`、`rise`、`grow`、`expand`、`jump` 等词。系统不会指定“必须选 increase”，而是先把这些候选随机分成两堆，再把绿色那一堆整体抬高一点。于是模型仍然在“合理词”里选词，只是统计上更偏绿。

这类方法的边界也非常明确。

第一，它依赖密钥保密。密钥如果泄露，攻击者就能重建每一步的绿色集合，进而有针对性地避开绿词，或者专门把已有文本改成“低绿比例”的版本。也就是说，绿色 token 水印不是公开标准下的稳健签名，而是“密钥保密前提下的检测机制”。

第二，它检测的是采样轨迹是否被系统性偏置，而不是句子语义是否携带签名。只要文本被改写、翻译、压缩、扩写，原来的 token 序列就会发生变化；token 一变，原来的绿色统计结构也就可能消失。

第三，它最适合黑箱 API 场景。黑箱的意思是外部只能调用接口，看不到模型参数，也控制不了采样细节。此时服务提供方可以在服务端对输出悄悄加水印，而平台侧或合规侧只需要密钥和检测器，不需要公开模型本体。

第四，它对短文本不友好。因为它依赖统计显著性，文本越短，随机波动越大。几十个 token 的一句话，即使带了水印，也未必能稳定检出。

可以把它和“数字签名”做一个边界对比：

| 机制 | 保护对象 | 是否允许文本被改写 | 检测方式 |
| --- | --- | --- | --- |
| 绿色 token 水印 | 生成轨迹 | 不允许大改写 | 统计检验 |
| 哈希/签名 | 原始字节串 | 几乎不允许改动 | 完整性校验 |

这也是为什么“来源归因”和“文件完整性”是两个不同问题。绿色 token 水印适合问“像不像这个系统生成的”，不适合问“有没有任何字符被改过”。

---

## 核心机制与推导

绿色 token 水印的核心发生在 logit 层。设当前步的候选 token 为 $x_t$，原始分数为 $\text{logit}(x_t)$，则加水印后的分数可以写成：

$$
\text{logit}_{wm}(x_t)=\text{logit}(x_t)+\delta \cdot \mathbf{1}_{x_t \in G_t}
$$

其中：

- $G_t$ 是第 $t$ 步的绿色集合。
- $\delta$ 是偏置强度。
- $\mathbf{1}_{x_t \in G_t}$ 是指示函数，表示当前 token 是否属于绿色集合。

这条式子的意思很简单：如果某个 token 恰好落在绿色集合里，它的分数会被额外加上 $\delta$；如果它在红色集合里，就不加。这里没有“必须选绿词”的硬约束，只有“绿词更容易被采样”的软偏置。

把这条式子放进 softmax 后，可以更清楚地看出影响。原始概率是：

$$
P(x_t=i \mid x_{<t})=\frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

加入水印后，概率变成：

$$
P_{wm}(x_t=i \mid x_{<t})=\frac{\exp(z_i+\delta \cdot \mathbf{1}_{i \in G_t})}{\sum_j \exp(z_j+\delta \cdot \mathbf{1}_{j \in G_t})}
$$

因此，绿色 token 的相对概率会整体上升。若两个候选词原本分数很接近，一个是绿的，一个是红的，那么绿词更容易胜出；若某个红词原本就远高于其他候选，那么小偏置也不一定能改变结果。这正是“质量影响较小”的原因：它是在合理候选之间轻微偏置，而不是粗暴重写输出。

为什么可以检测？因为在经典设置里，绿色集合通常约占词表的一半，即

$$
\gamma = \frac{|G_t|}{|V|} \approx 0.5
$$

如果没有水印，并且把每一步“是否落到绿色集合”近似看成独立伯努利试验，那么一段长度为 $T$ 的文本中，绿色 token 数 $s$ 可以近似写成：

$$
s \sim \text{Binomial}(T, \gamma)
$$

当 $\gamma=0.5$ 时，有：

$$
\mathbb{E}[s]=0.5T,\quad \text{Var}(s)=0.25T
$$

于是可以构造标准化统计量：

$$
Z = \frac{s - \gamma T}{\sqrt{T\gamma(1-\gamma)}}
$$

在最常见的 $\gamma=0.5$ 设置下，就是：

$$
Z = \frac{s - 0.5T}{\sqrt{0.25T}}
$$

如果 $Z$ 很大，说明“在无水印假设下看到这么多绿色 token”是小概率事件，于是就有理由拒绝原假设，判定文本更像带水印输出。

这个统计过程可以用“偏硬币”理解。无水印时，相当于每步抛一枚公平硬币，出现“绿面”的概率约是 0.5；有水印时，这枚硬币被悄悄做了手脚，绿面概率变成了 0.6、0.65、0.7，甚至更高。单次抛掷你看不出问题，但抛 200 次、500 次之后，绿面的数量会系统性偏多。

下面给一个具体数值例子。假设：

- 可计数 token 数为 $T=200$
- 绿色比例设定为 $\gamma=0.5$
- 实际观测到绿色 token 数为 $s=142$

则：

$$
Z = \frac{142-100}{\sqrt{50}} \approx 5.94
$$

$Z \approx 5.94$ 已经远高于常见阈值 4，对应的尾部概率非常小，因此很难解释为普通随机波动。这也是为什么“长文本上的偏差积累”能带来稳定检测力。

不过，这里还有两个现实修正必须说明。

第一，真实系统不会对所有 token 一视同仁。常见工程做法会过滤以下 token：

- 标点符号
- 过短或特殊子词
- 重复模式里的不稳定 token
- BOS/EOS 等控制 token

原因很简单：这些 token 的选择更多受 tokenizer 细节、格式、解码边界影响，统计噪声更大，纳入后会削弱检验稳定性。

第二，$s \sim \text{Binomial}(T,\gamma)$ 只是一个近似。真实文本中的 token 不是独立样本，绿色划分本身也依赖上下文哈希，因此严格来说并不满足简单独立同分布假设。但这个近似在工程上仍然有价值，因为它给出了可解释的检测分数、阈值和误报率近似。

真实工程例子可以这样想。某写作 API 服务商决定对公开生成内容默认加水印。当平台收到投诉，或需要溯源某段在站外传播的内容时，不必重新运行大模型，只需要：

1. 对文本做和生成端一致的 tokenizer 切分。
2. 用密钥和前缀重建每一步的绿色集合。
3. 统计绿色 token 数并计算 $Z$ 值。

这样就能给出“是否像本平台生成”的概率性证据。

---

## 代码实现

生成端和检测端都围绕同一件事：用密钥和上下文重建 $G_t$。常见做法是把“前缀 token + 密钥”做哈希，得到伪随机种子，再据此把词表打散并切一部分作为绿色集合。只要生成端和检测端的哈希规则、tokenizer、词表顺序一致，双方就能在每一步得到相同的 $G_t$。

先给出简化伪代码：

```text
生成端
for t in 1..T:
    seed = Hash(secret_key, prefix_tokens)
    G_t = RandomPartition(vocab, seed, gamma)
    for token in vocab:
        if token in G_t:
            logits[token] += delta
    next_token = Sample(softmax(logits))
    append(next_token)

检测端
green_count = 0
for t in 1..T:
    seed = Hash(secret_key, prefix_tokens_before_t)
    G_t = RandomPartition(vocab, seed, gamma)
    if observed_token_t in G_t:
        green_count += 1

Z = (green_count - gamma * T) / sqrt(T * gamma * (1 - gamma))
if Z > threshold:
    return "watermarked"
else:
    return "not detected"
```

下面给一个真正可运行的 Python 玩具实现。它不接真实 LLM，而是模拟“每一步先产生一组基础 logits，再给绿 token 加偏置，然后采样”。检测端用相同密钥复现绿色集合，最后算 $Z$ 值。代码已经补齐了哈希、划分、采样、生成和检测的完整流程。

```python
from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable, List, Sequence, Set


VOCAB = [
    "the",
    "model",
    "watermark",
    "token",
    "green",
    "red",
    "text",
    "detect",
    "sample",
    "probability",
    "attack",
    "rewrite",
]


def seeded_shuffle(items: Sequence[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    copied = list(items)
    rng.shuffle(copied)
    return copied


def make_greenlist(
    vocab: Sequence[str],
    secret_key: str,
    prefix_tokens: Sequence[str],
    gamma: float = 0.5,
) -> Set[str]:
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")

    payload = secret_key + "||" + " ".join(prefix_tokens)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)

    shuffled = seeded_shuffle(vocab, seed)
    cutoff = max(1, int(len(shuffled) * gamma))
    return set(shuffled[:cutoff])


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def sample_index(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def z_score(green_count: int, total: int, gamma: float = 0.5) -> float:
    if total <= 0:
        raise ValueError("total must be positive.")
    var = total * gamma * (1.0 - gamma)
    if var <= 0:
        raise ValueError("variance must be positive.")
    mean = gamma * total
    return (green_count - mean) / math.sqrt(var)


def generate_text(
    vocab: Sequence[str],
    total_steps: int,
    secret_key: str,
    gamma: float = 0.5,
    delta: float = 1.2,
    watermarked: bool = True,
    seed: int = 123,
) -> List[str]:
    rng = random.Random(seed)
    prefix = ["<bos>"]
    output = []

    for _ in range(total_steps):
        # 用随机数模拟模型本来的 logits
        base_logits = [rng.gauss(0.0, 1.0) for _ in vocab]

        green = make_greenlist(vocab, secret_key, prefix, gamma=gamma)
        logits = list(base_logits)

        if watermarked:
            for i, tok in enumerate(vocab):
                if tok in green:
                    logits[i] += delta

        probs = softmax(logits)
        next_idx = sample_index(probs, rng)
        next_token = vocab[next_idx]

        output.append(next_token)
        prefix.append(next_token)

    return output


def detect_watermark(
    tokens: Iterable[str],
    vocab: Sequence[str],
    secret_key: str,
    gamma: float = 0.5,
    threshold: float = 4.0,
) -> tuple[int, int, float, bool]:
    prefix = ["<bos>"]
    green_count = 0
    total = 0

    for tok in tokens:
        green = make_greenlist(vocab, secret_key, prefix, gamma=gamma)
        if tok in green:
            green_count += 1
        total += 1
        prefix.append(tok)

    score = z_score(green_count, total, gamma=gamma)
    return green_count, total, score, score > threshold


def main() -> None:
    secret_key = "secret-demo-key"

    plain_tokens = generate_text(
        vocab=VOCAB,
        total_steps=200,
        secret_key=secret_key,
        gamma=0.5,
        delta=1.2,
        watermarked=False,
        seed=123,
    )

    wm_tokens = generate_text(
        vocab=VOCAB,
        total_steps=200,
        secret_key=secret_key,
        gamma=0.5,
        delta=1.2,
        watermarked=True,
        seed=123,
    )

    plain_green, plain_total, plain_z, plain_flag = detect_watermark(
        plain_tokens, VOCAB, secret_key, gamma=0.5, threshold=4.0
    )
    wm_green, wm_total, wm_z, wm_flag = detect_watermark(
        wm_tokens, VOCAB, secret_key, gamma=0.5, threshold=4.0
    )

    assert plain_total == 200
    assert wm_total == 200
    assert wm_z > plain_z
    assert wm_flag is True
    assert plain_flag is False

    print("plain text:")
    print("green_count =", plain_green, "z =", round(plain_z, 3), "detected =", plain_flag)
    print("watermarked text:")
    print("green_count =", wm_green, "z =", round(wm_z, 3), "detected =", wm_flag)


if __name__ == "__main__":
    main()
```

如果用 `python3 demo.py` 运行这段代码，通常会看到类似结果：

- 无水印文本的 $Z$ 值在 0 附近浮动
- 带水印文本的 $Z$ 值显著更大
- 在默认阈值 4.0 下，只有带水印文本被判定为 `detected = True`

这个玩具版本和真实系统仍有差距，但它已经保留了最关键的工程结构：同一把密钥、同一套绿色集合重建逻辑、同一套统计检测公式。

真正接入模型时，还要再补三类工程约束：

1. 过滤不稳定 token。否则标点、控制 token、异常子词会污染统计。
2. 保证生成端和检测端使用完全一致的 tokenizer。只要分词边界不同，$G_t$ 就无法复现。
3. 固定哈希规则、词表顺序、$\gamma$ 和 $\delta$。任何一个细节不一致，都会让检测失败。

可以把这一段记成一句话：真正难的不是公式，而是“生成端和检测端必须一字不差地共享同一套离散化过程”。

---

## 工程权衡与常见坑

最大的权衡是：检测力、文本质量和抗攻击性之间没有免费午餐。

先看偏置强度 $\delta$。如果 $\delta$ 太小，文本自然度通常很好，但绿色 token 偏差不够明显，尤其在短文本上很难检出；如果 $\delta$ 太大，检测会更稳，可代价是模型可能开始系统性偏好某类词，导致措辞变窄、风格变硬，甚至影响回答质量。

这可以用一张表快速理解：

| 参数变化 | 对检测的影响 | 对文本质量的影响 |
| --- | --- | --- |
| $\delta$ 变小 | 更难检 | 更自然 |
| $\delta$ 变大 | 更易检 | 更可能拉偏措辞 |
| $\gamma$ 接近 0.5 | 检测分析更简单 | 常见默认选择 |
| 文本更长 | 更易显著 | 用户未必总能提供长文本 |

第二个坑是短文本。因为检测依赖的是统计显著性，而显著性随样本量增长才更稳定。设 $\gamma=0.5$，则标准差大约是 $\sqrt{0.25T}$。当 $T$ 很小时，这个标准差不小，随机波动完全可能把有水印文本“冲回”普通区间。

举个最简单的例子：

- 若 $T=20$，标准差约为 $\sqrt{5}\approx 2.24$
- 若 $T=500$，标准差约为 $\sqrt{125}\approx 11.18$

虽然第二个标准差绝对值更大，但均值附近的相对波动更稳定，因为水印造成的累积偏差也在同步增长。直观地说，短文本更像“掷了几次硬币”，长文本更像“掷了很多次硬币”。

第三个坑是重写攻击。攻击者不必知道完整密钥，也不一定要精确识别每一步的绿/红 token，只要他知道“系统在偏好某些词”，就可以通过改写、同义替换、句式重排、回译等方式扰乱原始 token 序列。因为绿色 token 水印嵌在 token 选择层，而不是语义层，所以“意思不变”并不意味着“水印保留”。

一个新手能理解的攻击例子如下。原文里经常出现 `increase`、`important`、`therefore` 这类更容易被采样到的词。攻击者不需要知道哪些一定是绿词，只需要用同义改写器批量替换成 `rise`、`key`、`thus`，再调整句式和标点，就可能把绿色比例打散。文本语义差不多，统计痕迹却变弱了。

ACL 2024 的颜色感知替换攻击（SCTS）就是这条思路的代表：一旦攻击者能通过探测推断哪些候选更可能偏绿，就能用更有针对性的替换方式削弱水印。

第四个坑是 tokenizer 不一致。生成端若使用某版本 tokenizer，把 `watermarking` 切成一个 token；检测端若换了版本，把它切成两个子词，那么后续所有前缀哈希都会错位，导致每一步的 $G_t$ 都不一致。这个问题看起来像“实现细节”，但在工程里往往比公式本身更致命。

第五个坑是公开接口探测。如果模型接口允许攻击者无限试探，他就能对相同上下文重复采样，统计哪些词被偏好，进而逐步恢复“颜色分布”的局部信息。即使拿不到完整密钥，也可能学到足以破坏检测的近似规律。

| 攻击类型 | 原理 | 防御手段 | 实施成本 |
| --- | --- | --- | --- |
| 同义替换 | 把绿词改成红词或中性词 | 多密钥轮换、限制外部探测 | 中 |
| 改写/翻译 | 改变 token 序列 | 混合语义水印或多证据归因 | 高 |
| 短文本规避 | 文本太短不显著 | 仅对长文本做强判定 | 低 |
| tokenizer 不一致 | 检测端无法复现前缀 | 固定模型与 tokenizer 版本 | 低 |
| 公开接口探测 | 通过频率差推断颜色偏好 | 限速、审计、混合策略 | 中 |

真实工程里，常见做法不是“只靠单一水印定生死”，而是把它放进完整证据链：

- 水印检测结果
- 请求日志
- API key 或账号身份
- 输出时间戳
- 原文存证或哈希
- 上下游产品链路信息

这样做的原因很直接：水印是统计证据，不是法律意义上的绝对证明。它的价值在于提高归因能力，而不是单独承担全部判定责任。

---

## 替代方案与适用边界

绿色 token 水印的优势，是简单、便宜、适合黑箱部署；它的弱点，是一旦文本被大幅改写，检测力会迅速下降。所以后来出现了更偏语义层的方法。

最有代表性的一类是语义水印。语义水印的思路不是在具体 token 上做偏置，而是在句子或片段的语义表示上做约束。SemStamp 就属于这个方向：它先把候选句子编码为语义向量，再用局部敏感哈希（LSH）等方法把语义空间切分成若干区域，只接受落入“目标区域”的候选句子。

直白地说，绿色 token 水印要求的是“你更常说某些词”；语义水印要求的是“你说出来的整句话更常落在某些语义格子里”。

这样做的好处是：同义改写后，句子表面 token 变了，但语义向量可能仍然停留在相近区域，所以水印保留性更强。代价也同样明显：生成时需要做句级编码、候选比较甚至拒绝采样，系统延迟、实现复杂度和算力成本都会上升。

另一类不是水印，而是完整性校验。例如把输出结果连同时间戳、模型版本、请求 ID 做签名或哈希。这适合“平台内闭环流转”的场景，比如内部文档、固定报告、合规存档。它解决的是“原文是否被改过”，不是“这段话是否来自某模型”。因为只要文本被人手改动几个字，原始签名就会失效。

三类方案的差异可以总结如下：

| 方案 | 检测对象 | 抗同义改写 | 黑箱部署 | 部署难度 |
| --- | --- | --- | --- | --- |
| 绿色 token 水印 | token 统计偏差 | 弱 | 强 | 低 |
| 语义水印（如 SemStamp） | 句子语义区域 | 较强 | 一般 | 高 |
| 哈希/签名完整性 | 原文是否被改动 | 很弱 | 强 | 中 |

适用边界可以直接按场景理解。

- 如果你是 API 服务商，目标是在不明显伤害质量的前提下，给较长文本加上低成本可检痕迹，绿色 token 水印最合适。
- 如果你预期文本会被大量改写、翻译、摘要后再传播，单纯 token 水印往往不够，应考虑语义水印或混合策略。
- 如果你的场景是内部文档流转，关注点是“这份文件是不是原件、有没有被动过”，那么签名和哈希比水印更直接。

还可以再加一个实践结论：这些方法解决的是不同层次的问题。

- 绿色 token 水印解决来源可检测性。
- 语义水印尝试提升改写后的存活率。
- 哈希/签名解决原文完整性。

因此，结论不是“哪种最好”，而是“哪种和目标更匹配”。一旦系统需求从“让平台能追溯自己输出”变成“即使被翻译和重写后仍能识别来源”，就已经超出了经典绿色 token 水印最强的工作区间。

---

## 参考资料

- John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. *A Watermark for Large Language Models*. ICML 2023 / PMLR 202.  
  重点：提出绿色 token 水印、logit 偏置、基于绿色计数的检测统计量，以及误报率和检测率的基础分析。  
  链接：https://proceedings.mlr.press/v202/kirchenbauer23a.html

- Qilong Wu, Varun Chandrasekaran. *Bypassing LLM Watermarks with Color-Aware Substitutions*. ACL 2024.  
  重点：提出颜色感知替换攻击，说明 token 级水印在改写和替换场景下容易被削弱。  
  链接：https://aclanthology.org/2024.acl-long.464/

- Abe Hou, Jingyu Zhang, Tianxing He, Yichen Wang, Yung-Sung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, Yulia Tsvetkov. *SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation*. NAACL 2024.  
  重点：把水印从 token 层提升到语义层，提升对同义改写和释义攻击的鲁棒性。  
  链接：https://aclanthology.org/2024.naacl-long.226/

- vLLM Watermark 文档，Maryland/KGW 算法说明。  
  重点：从工程实现角度说明如何构造绿色集合、如何在生成时加偏置、如何在检测时复现统计量。  
  链接：https://vermaapurv.com/vLLM-Watermark/algorithms/maryland.html

- 补充阅读方向：Kirchenbauer 系列后续讨论与实现文档。  
  重点：理解为什么检测统计量常写成 $Z$ 分数、为什么短文本检测困难，以及为什么 tokenizer 一致性在工程里是硬约束。  
  链接：https://github.com/jwkirchenbauer/lm-watermarking
