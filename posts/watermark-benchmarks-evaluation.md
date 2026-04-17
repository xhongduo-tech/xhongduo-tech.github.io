## 核心结论

水印评估不是只看“能不能检出”，而是看三件事能否同时成立：`Detectability`、`Quality Impact`、`Robustness`。`Detectability` 就是检测性，白话说是“你能否稳定看出这段文本带了水印”；`Quality Impact` 就是质量影响，白话说是“加了水印后文章有没有明显变差”；`Robustness` 就是鲁棒性，白话说是“别人稍微改写、翻译、删词以后，水印还在不在”。

`Mark My Words` 的价值在于把这三个维度放进同一个基准里，并且给出一个很实用的结论：某些分布偏移型水印方案，尤其是 Kirchenbauer 等人的 greenlist 方案，可以在较短文本内完成检测，同时几乎不牺牲生成质量，还能抵抗一部分简单篡改攻击。这说明“可检测”和“高质量”并不天然冲突，关键在于参数和评测协议是否合理。

`UWBench` 进一步补上了一个过去容易被忽略的问题：如果同一个 prompt 被反复采样，所谓“无偏水印”也可能逐渐暴露出分布漂移。它因此引入 `SPMG` 统计量，也就是“单个 prompt、多次生成”的偏移测度，重点评估 `unbiasedness`、`detectability`、`robustness` 三轴，而不只是看一次生成的 z-score。

截至 **2026-04-14**，公开可核验资料中我没有找到一个与“`WOMD` 是文本水印评估基准”一致的来源。当前公开搜索结果主要指向 `Waymo Open Motion Dataset` 及其衍生数据集，而不是 LLM 水印 benchmark。因此，若不补充论文或官方页面，严谨的分析边界应当先落在 `Mark My Words`、`UWBench` 以及早期代表性水印方案上。

---

## 问题定义与边界

要评估一个文本水印基准，先要明确“评估对象”是什么。这里讨论的是 **生成时嵌入型文本水印**，也就是模型在采样 token 时主动偏向一部分 token，使输出中留下可被密钥验证的统计信号。它不是后处理贴标签，也不是训练后分类器去猜“像不像 AI 写的”。

一个可部署的基准，通常至少回答三类问题：

| 维度 | 关心的问题 | 常见测法 |
|---|---|---|
| Detectability | 多长文本内能稳定检出，误报率多低 | z-score、p-value、TPR/FPR、所需 token 数 |
| Quality Impact | 加水印后文本是否变差 | 人工评审、LLM-as-a-judge、PPL、任务分数 |
| Robustness | 经编辑、翻译、重排后还能否检出 | paraphrase、translation、token edit 攻击 |

这里有一个边界必须说清：**短文本检测** 与 **长文本归因** 不是同一个问题。短文本场景通常更关心“100 token 左右能否做出可靠判断”；长文本场景更关心“对方经过多轮改写后还能否保留可验证信号”。前者偏 Size 和检测效率，后者偏 Robustness。

再看 “WOMD 与 Mark My Words 分析” 这个标题。`Mark My Words` 是明确存在的文本水印 benchmark；但 “WOMD” 若按公开资料检索，主流含义并不是文本水印评估。也就是说，当前能严谨展开的部分主要是：

1. `Mark My Words` 如何定义质量、检测长度与抗篡改。
2. `UWBench` 如何补充无偏性与重复采样漂移问题。
3. 为什么工程上不能只盯一个分数。

一个新手可理解的玩具例子是：你写了一个“检测盒子”，输入一段模型输出，先算 z-score；若 z-score 足够高，再看这段文本是否因为加水印而变得明显奇怪；最后再把它翻译回中文或做一次改写，看还能否检出。只有这三关都过，水印方案才有部署意义。

---

## 核心机制与推导

`Mark My Words` 的核心贡献，不是又发明了一种新水印，而是把已有方案放到统一实验框架里比较。它生成固定数量的输出样本，对每个方案同时测三件事：

1. 质量评分。
2. 检测所需最小长度，也就是 `Size`。
3. 遭受篡改后的剩余检测能力，也就是 `Tamper-Resistance`。

其中 `Size` 很重要。因为很多方案“理论上能检出”，但要几百上千 token 才显著，这在真实审核流里几乎不可用。`Mark My Words` 用一个固定显著性水平来定义检测门槛，并用“达到检测所需的中位 token 数”来比较方案效率。这个指标的白话解释就是：**平均需要读多长，检测器才敢下结论**。

早期以 Kirchenbauer 为代表的方案属于 `distribution-shift watermark`，即“分布偏移型水印”。白话说，就是对一部分由密钥决定的 token 集合加一点偏置，让模型更常选中它们。若无水印，绿名单 token 的出现近似符合基线分布；若有水印，它们的出现频率会系统性偏高，于是可以用统计检验检测。

一个常见简化推导是：设某段文本长度为 $n$，其中命中 greenlist 的 token 数为 $G$。在无水印假设下，greenlist 命中概率近似为 $\gamma$，则期望与方差近似为

$$
\mathbb{E}[G] = n\gamma,\quad \mathrm{Var}(G)=n\gamma(1-\gamma)
$$

于是可构造 z-score：

$$
z = \frac{G - n\gamma}{\sqrt{n\gamma(1-\gamma)}}
$$

如果 $z$ 足够大，就说明“greenlist 命中过多”不太可能只是随机波动，而更像是带密钥偏置后的结果。这就是很多检测器背后的统计直觉。

`UWBench` 则指出，仅靠这种单次检测统计还不够。原因是“无偏水印”声称不改变输出分布，但在多次重复查询同一个 prompt 时，微小偏移可能累积成可测的分布漂移。因此它引入 `SPMG`，即 Single-Prompt Multi-Generation。对白话来说，就是：**别只看一次生成，要看同一个 prompt 连续生成很多次后，整体分布有没有偏离原模型**。

它的核心量写成：

$$
\overline{Met}_i(P)=\frac{1}{m}\sum_{j=1}^m Met(g_j^{(p_i)}(P))
$$

这里 $\overline{Met}_i(P)$ 表示模型 $P$ 在第 $i$ 个 prompt 上多次生成后的平均指标值。再定义

$$
\Delta Met(P_M, P_T)=\frac{1}{n}\sum_{i=1}^n \left|\overline{Met}_i(P_M)-\overline{Met}_i(P_T)\right|
$$

其中 $P_M$ 是原模型，$P_T$ 是待评估的水印模型。为了扣掉模型本身重复采样就存在的噪声，再减去一个基线项：

$$
DetWmk(P_M, P_T)=\Delta Met(P_M, P_T)-\Delta Met(P_M, P_M')
$$

这里的 $P_M'$ 可以理解为原模型的独立重复采样副本。这个修正的意义是：如果两个“本质同分布”的模型也会因为采样波动产生差异，那么真正由水印带来的偏移应该是“总差异减去自然噪声”。

一个玩具例子如下。假设同一个 prompt“解释哈希表碰撞”分别由原模型和带水印模型各生成 5 次，然后用一个质量或风格指标打分：

| prompt | 原模型平均分 | 水印模型平均分 | 差值 |
|---|---:|---:|---:|
| 哈希表碰撞解释 | 0.81 | 0.84 | 0.03 |
| 二分查找解释 | 0.78 | 0.79 | 0.01 |
| TCP 三次握手解释 | 0.75 | 0.83 | 0.08 |

如果这些差值整体明显大于原模型自己重复采样的自然波动，就说明水印改变了输出分布。此时，即便单次文本质量看起来没问题，也要警惕所谓“无偏”只是单样本下近似成立。

真实工程例子可以放在内容合规链路里看。平台每天需要审核自动生成的客服回复，要求：
1. 100 到 200 token 内尽快判定是否来自内部模型。
2. 不能因为加水印让回复更僵硬。
3. 用户复制后做轻度改写，仍要尽量保留检测能力。

这时 `Mark My Words` 提供的是部署前的三维体检框架；`UWBench` 提供的是“反复采样后分布是否悄悄跑偏”的补充检查。两者关注点不同，但互补。

---

## 代码实现

下面给一个可运行的简化版 `python` 示例，用来模拟 `greenlist` 检测的核心统计流程。它不是完整水印器，只展示 benchmark 里最重要的一步：**统计命中次数，算 z-score，并验证阈值逻辑**。

```python
import math
import random

def z_score(green_hits: int, total_tokens: int, gamma: float) -> float:
    assert 0 < gamma < 1
    assert 0 <= green_hits <= total_tokens
    mean = total_tokens * gamma
    var = total_tokens * gamma * (1 - gamma)
    return (green_hits - mean) / math.sqrt(var)

def simulate_hits(total_tokens: int, gamma: float, bias: float = 0.0, seed: int = 0) -> int:
    """
    gamma: 无水印时命中 greenlist 的基线概率
    bias:  水印带来的额外偏置
    """
    rng = random.Random(seed)
    p = min(max(gamma + bias, 0.0), 1.0)
    hits = 0
    for _ in range(total_tokens):
        if rng.random() < p:
            hits += 1
    return hits

# 玩具例子：无水印 vs 有水印
gamma = 0.5
n = 100

hits_plain = simulate_hits(total_tokens=n, gamma=gamma, bias=0.0, seed=1)
hits_marked = simulate_hits(total_tokens=n, gamma=gamma, bias=0.12, seed=1)

z_plain = z_score(hits_plain, n, gamma)
z_marked = z_score(hits_marked, n, gamma)

# 阈值仅作演示，真实项目应按目标 FPR 校准
threshold = 2.05

assert z_plain < threshold
assert z_marked > threshold

print({
    "hits_plain": hits_plain,
    "z_plain": round(z_plain, 3),
    "hits_marked": hits_marked,
    "z_marked": round(z_marked, 3),
    "detected": z_marked > threshold
})
```

如果把它扩展到一个最小 benchmark，流程通常是：

1. 对一组 prompt 生成原模型输出与水印模型输出。
2. 对每条输出记录质量分、检测统计量、被攻击后的检测结果。
3. 汇总成 `Quality / Size / Robustness` 三列，比较方案。

可以把评测骨架抽象成下面这样：

```python
def evaluate_scheme(prompts, generate_plain, generate_marked, detect, judge_quality, attack):
    rows = []
    for prompt in prompts:
        plain = generate_plain(prompt)
        marked = generate_marked(prompt)

        quality_plain = judge_quality(prompt, plain)
        quality_marked = judge_quality(prompt, marked)

        detect_result = detect(marked)  # 例如返回 z-score、p-value、size

        attacked = attack(marked)       # 例如翻译、删词、局部重写
        attacked_detect = detect(attacked)

        rows.append({
            "prompt": prompt,
            "quality_plain": quality_plain,
            "quality_marked": quality_marked,
            "detect_score": detect_result["score"],
            "size": detect_result["size"],
            "robust_after_attack": attacked_detect["detected"],
        })

    assert len(rows) == len(prompts)
    return rows
```

这个骨架背后的工程含义很直接：**任何只输出一个“检测准确率”的 benchmark 都是不够的**。因为你还必须知道它是否伤害质量，以及是否经不起最常见的轻度编辑。

---

## 工程权衡与常见坑

最常见的误区，是把水印评估做成单指标竞赛。只追检测强度，会出现三类问题：

| 追求方向 | 直接收益 | 常见副作用 | 工程后果 |
|---|---|---|---|
| 更强 Detectability | 更短文本可检出 | 文本更生硬，分布偏移更大 | 用户体验下降，暴露模型特征 |
| 更低 Quality Impact | 文本更自然 | 信号变弱，检测长度上升 | 短文本审核失效 |
| 更强 Robustness | 更抗改写攻击 | 算法复杂、成本高、调参难 | 推理成本和系统复杂度上升 |

第一个坑是 **阈值偷懒**。很多团队直接沿用论文阈值，却不按自己业务的误报率目标重标定。结果是离线实验很好看，线上误报一堆。正确做法是固定目标 `FPR`，再校准阈值与最小检测长度。

第二个坑是 **只用 paraphrase 攻击评鲁棒性**。改写模型本身波动很大，不同改写器会把结果带偏。`UWBench` 的一个重要提醒是，token-level modification 这类更可控的攻击，在做跨方案对比时往往更稳定。

第三个坑是 **质量评估和检测评估脱节**。如果质量用一套 prompt，检测用另一套 prompt，最后得到的 trade-off 经常不可比。基准设计上应尽量共享任务集，并记录相同输出的多维指标。

第四个坑是 **忽略日志与审计链**。真实工程里，水印不是论文分数，而是审核系统的一部分。你至少要记录：
`prompt 类型`、`模型版本`、`阈值`、`z-score`、`检测长度`、`攻击后结果`、`人工复核结论`。  
否则后续无法解释误报，也无法知道是模型改了、提示词改了，还是攻击样本分布变了。

一个真实工程例子是企业内容风控。系统可以先做质量门禁，再做水印检测：

1. 先用 LLM judge 或任务分判断文本是否可用。
2. 再在前 100 到 200 token 内计算检测统计量。
3. 对高风险文本追加一次轻量篡改仿真，例如同义替换或局部删词。
4. 若原文和轻度篡改后都能通过阈值，再给出高置信结论。

这样做的核心不是“把 z-score 算出来”，而是把 `quality`、`size`、`robustness` 串成同一条自动化审核链。

---

## 替代方案与适用边界

`Mark My Words` 适合回答的问题是：**不同水印方案在质量、检测长度和抗篡改之间，谁更平衡**。如果你的目标是选型，它很有价值；如果你的目标是研究“无偏性是否真的成立”，它就不够。

`UWBench` 适合补上另一类边界：**一个号称不改分布的水印，在重复查询时是否仍会积累可观察偏移**。这对于 API 场景尤其重要，因为攻击者并不一定只拿一次输出。

不同业务场景，可以按下面的矩阵选基准重心：

| 场景 | 核心约束 | 更该看什么 | 更适合的评估视角 |
|---|---|---|---|
| 短文本审核 | 文本短、要求快 | Size、FPR、TPR | `Mark My Words` 风格 |
| 长文归因 | 文本长、会被编辑 | Robustness、攻击后残留信号 | `Mark My Words` + 攻击评测 |
| 高频 API 查询 | 同 prompt 多次采样 | Unbiasedness、分布漂移 | `UWBench` |
| 法规/审计敏感场景 | 需解释误报与证据链 | 多指标联合、阈值可审计 | 二者结合 |

如果只需要快速辨别一段 100 token 左右的客服短回复是否来自内部模型，优先看 `Size` 与目标 `FPR` 下的检测效率就够了。  
如果要处理会被转载、摘要、翻译的长文内容，就必须把 `Robustness` 拉到前面。  
如果面对的是能反复调用你模型的对手，`UWBench` 关注的重复采样漂移就不再是理论细节，而是实际风险。

至于 “WOMD”，当前公开证据不足，不能把它和 `Mark My Words` 并列当作文本水印 benchmark 分析。严谨做法不是硬补，而是等待具体出处后再纳入统一比较。

---

## 参考资料

- **Mark My Words 项目页**  
  用途：确认 benchmark 的三维指标定义，尤其是 `quality / size / tamper-resistance`。  
  https://wagner-group.github.io/projects/markmywords/index.html

- **Mark My Words 论文页（arXiv:2312.00273）**  
  用途：查看论文摘要、代码仓库入口与核心结论。  
  https://huggingface.co/papers/2312.00273

- **Analyzing and Evaluating Unbiased Language Model Watermark（OpenReview, ICLR 2026 Poster）**  
  用途：确认 `UWBench`、`SPMG`、`unbiasedness/detectability/robustness` 三轴协议。  
  https://openreview.net/forum?id=6T4LR1oRwA

- **Kirchenbauer et al., A Watermark for Large Language Models（PMLR 2023）**  
  用途：查看 greenlist / distribution-shift 水印的代表性原始方法。  
  https://proceedings.mlr.press/v202/kirchenbauer23a.html

- **截至 2026-04-14 关于 “WOMD” 的公开检索结果**  
  用途：说明当前常见指向是 `Waymo Open Motion Dataset` 或 `WOMD-Reasoning`，而非文本水印 benchmark，因此暂不能纳入同类对比。  
  例如：https://openreview.net/forum?id=lTBq5LOUKC
