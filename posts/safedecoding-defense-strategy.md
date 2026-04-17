## 核心结论

SafeDecoding 的核心价值，是**不改主模型参数，只在推理时改“怎么选下一个 token”**。这里的 token 可以先理解成“模型一次生成的最小文字单位”，可能是一个字、一个词，或者词的一部分。它的做法不是重新训练一个更安全的大模型，而是在解码阶段同时参考两个分布：

1. 原始模型的分布，负责保持回答能力与表达自然度。
2. 安全 expert 的分布，负责把回答往“拒绝危险请求”方向拉。

具体做法是：先分别取原模型与 expert 当前步的 top-$k$ 候选，再只保留二者交集中的 token，最后按权重混合概率。于是，一些原模型想说、但 expert 明显不认同的危险 token，会直接失去被采样的机会。

一个适合新手的直观理解是：把生成过程改成“双人投票”。原模型像“会答题的老师”，安全 expert 像“审稿人”。只有两者都接受的词，才进入候选池；然后再按权重决定最终更偏向“有帮助”还是“更安全”。

下面用一个玩具例子说明。假设当前第一个 token 的候选如下：

| token | 原模型概率 $P_{\text{orig}}$ | expert 概率 $P_{\text{expert}}$ | 是否在交集 | $\alpha=0.7$ 时未归一化分数 |
|---|---:|---:|---|---:|
| Sure | 0.60 | 0.00 | 否 | 0.00 |
| OK | 0.30 | 0.00 | 否 | 0.00 |
| Sorry | 0.10 | 0.70 | 是 | $0.7\times0.1+0.3\times0.7=0.28$ |
| Nope | 0.00 | 0.20 | 否 | 0.00 |

如果交集里只有 `Sorry`，那么 SafeDecoding 会几乎必然把首个 token 推向拒绝语气。对于越狱提示，这往往已经足够，因为回答一旦以拒绝开头，后续分布通常也会被带入安全轨道。

因此，SafeDecoding 可以概括为一句话：**它不是让模型“学会更安全”，而是让模型在生成最关键的前几个 token 时，更难踏进不安全分支。**

---

## 问题定义与边界

越狱攻击，指的是用户通过特殊提示词、角色扮演、编码伪装、多轮诱导等方式，让模型绕过原本的安全约束，输出本不应提供的内容。对初级工程师来说，可以把它理解成“用户想办法把模型从默认规则里哄出来”。

SafeDecoding 试图解决的问题非常明确：**在不重训主模型的前提下，降低模型在解码早期选出危险 token 的概率**。它主要防御的是 prompt 层面的攻击，也就是“用户如何措辞，影响模型第一步怎么开口”。

它的边界也同样明确：

| 边界项 | SafeDecoding 能做什么 | SafeDecoding 不能做什么 |
|---|---|---|
| 推理时防御 | 在生成阶段直接约束候选 token | 不能修复训练数据里的系统性偏差 |
| 作用范围 | 对前几个 token 的分布影响最大 | 不保证长文本后段永远安全 |
| 模型改造成本 | 无需重训主模型，可外挂 expert | 仍需要一个安全 expert |
| 攻击类型 | 对常见 jailbreak prompt 有效 | 对工具调用、外部执行链条需配合别的机制 |

数学上，设原模型当前步 top-$k$ 候选集合为 $S_o$，expert 当前步 top-$k$ 候选集合为 $S_e$，那么 SafeDecoding 的采样空间定义为：

$$
S = S_o \cap S_e
$$

白话解释：**只有原模型和安全 expert 都觉得“这词至少还行”的 token，才有资格进入最终抽样。**

这一步的意义非常大。很多危险回答的开头 token，比如 `Sure`、`Here`、`First`，在普通模型上概率可能很高，但如果安全 expert 不把它们放进 top-$k$，它们就会被直接排除。于是越界 token 不是“概率变小一点”，而是**候选资格被取消**。

这里还要强调一个常被忽略的边界：论文与工程实践通常只在**前几个 token**启用 SafeDecoding，例如前 2 个 token。原因是模型的开头决定语气和轨迹，一旦首句进入拒绝分支，后面大概率会延续安全风格；反过来，如果全程都强行做双模型约束，会明显牺牲自然度和多样性。

---

## 核心机制与推导

先把几个术语说清楚。

- **解码**：模型拿到当前上下文后，从概率分布里选下一个 token 的过程。
- **top-$k$**：只保留概率最高的 $k$ 个候选，其他全部忽略。
- **归一化**：把一组分数重新缩放成总和为 1 的概率。
- **expert**：安全微调后的辅助模型。白话说，就是“更保守、更会拒绝危险请求的版本”。

SafeDecoding 的每一步可以拆成四件事：

1. 原模型输出当前步的 logits 或概率分布。
2. 安全 expert 也输出同一步的 logits 或概率分布。
3. 各自取 top-$k$，求交集 $S=S_o\cap S_e$。
4. 对交集内 token 做加权混合并归一化，再采样。

若 token $t\in S$，其新分数定义为：

$$
\tilde{P}(t)=\alpha P_{\text{orig}}(t)+(1-\alpha)P_{\text{expert}}(t)
$$

其中 $\alpha\in[0,1]$ 是权重系数。它的工程含义非常直接：

- $\alpha$ 越大，越偏向原模型，回答更自然、更有帮助。
- $\alpha$ 越小，越偏向 expert，拒绝更坚决，但误拒风险更高。

因为 $\tilde{P}(t)$ 还不是标准概率，所以还需要归一化：

$$
P_{\text{safe}}(t)=\frac{\tilde{P}(t)}{\sum_{u\in S}\tilde{P}(u)}
\quad,\quad t\in S
$$

对 $S$ 外的 token，则有：

$$
P_{\text{safe}}(t)=0
\quad,\quad t\notin S
$$

这套机制里，交集筛选和线性混合分别解决两个问题：

| 机制 | 解决的问题 | 直观效果 |
|---|---|---|
| 交集 $S_o\cap S_e$ | 去掉 expert 明确不认同的 token | 直接剔除高风险开头 |
| 线性混合 | 在“安全”和“帮助性”之间可调 | 避免回答全都僵硬拒绝 |

玩具例子可以写得更完整一点。假设某个越狱提示要求模型输出恶意脚本，当前步候选如下：

- 原模型 top-3：`Sure` 0.60，`Here` 0.25，`Sorry` 0.15
- expert top-3：`Sorry` 0.70，`Cannot` 0.20，`I` 0.10

若 $k=3$，则交集只有 `Sorry`。此时无论原模型多想输出 `Sure`，SafeDecoding 都不会让它进入采样空间。这与传统“在完整分布上做一点惩罚”不同，它是更硬的约束。

真实工程例子则更接近下面这种部署方式：你有一个基于 Llama 或 Vicuna 的聊天服务，主模型负责通用对话；你再用 LoRA 在少量拒绝样本上训练出一个安全 expert。上线时，两者共享词表，推理前 2 个 token 走 SafeDecoding，后续 token 回到普通采样。这样做的原因是：

- 前 2 个 token 足够决定“拒绝”还是“顺从”。
- expert 只需要在关键节点介入，不必全程参与风格控制。
- 额外计算成本可控，因为只在极少数步数上做双模型前向与融合。

从推导角度看，SafeDecoding 不是在优化新的训练目标，而是在推理时定义了一个新的局部采样分布。它更像“在线重排”而不是“离线重训”。这也是它工程吸引力最大的地方。

---

## 代码实现

下面给出一个可运行的 Python 版本。它不依赖深度学习框架，只模拟“两个模型已经给出概率分布”的那一步。代码重点展示三件事：取 top-$k$、求交集、加权归一化。

```python
from math import isclose

def top_k(dist, k):
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    return dict(items[:k])

def safe_decode_step(orig_dist, expert_dist, alpha=0.7, k=3):
    """
    orig_dist/expert_dist: dict[token, prob]
    alpha: 原模型权重
    k: 各自保留的 top-k 候选
    return: (sampling_pool, safe_dist)
    """
    orig_top = top_k(orig_dist, k)
    expert_top = top_k(expert_dist, k)

    pool = set(orig_top) & set(expert_top)
    if not pool:
        # 工程上常见回退：若交集为空，可退回 expert top-1 或原模型安全模板
        fallback_token = max(expert_top.items(), key=lambda x: x[1])[0]
        return {fallback_token}, {fallback_token: 1.0}

    mixed = {}
    for token in pool:
        mixed[token] = alpha * orig_dist[token] + (1 - alpha) * expert_dist[token]

    total = sum(mixed.values())
    safe_dist = {token: score / total for token, score in mixed.items()}
    return pool, safe_dist


# 玩具例子
orig = {
    "Sure": 0.60,
    "OK": 0.30,
    "Sorry": 0.10,
    "I": 0.00,
}

expert = {
    "Sorry": 0.70,
    "Nope": 0.20,
    "I": 0.10,
    "Sure": 0.00,
}

pool, safe_dist = safe_decode_step(orig, expert, alpha=0.7, k=3)

assert pool == {"Sorry"}
assert isclose(safe_dist["Sorry"], 1.0)

# 一个 benign 例子：两者都认可“Python”
orig2 = {
    "Python": 0.50,
    "You": 0.30,
    "Sorry": 0.20,
}
expert2 = {
    "Python": 0.40,
    "Sorry": 0.35,
    "Use": 0.25,
}
pool2, safe_dist2 = safe_decode_step(orig2, expert2, alpha=0.7, k=3)

assert pool2 == {"Python", "Sorry"}
assert isclose(sum(safe_dist2.values()), 1.0)
assert safe_dist2["Python"] > safe_dist2["Sorry"]
```

这个实现虽然是玩具版本，但已经覆盖了工程里的关键控制项：

| 参数 | 作用 | 常见取值思路 |
|---|---|---|
| $k$ | 控制候选池大小 | 过小会僵硬，过大计算略增 |
| $\alpha$ | 控制安全与帮助性的平衡 | 常在 benign 集上调优 |
| 启用步数 | 控制只在前几个 token 生效 | 论文与实践常用前 2 步 |
| 词表一致性 | 保证 token 能做交集 | 常通过同基座模型 + LoRA 实现 |

如果换成真实推理伪代码，大致流程如下：

```python
for step in range(max_new_tokens):
    orig_logits = orig_model(context)
    if step < safe_steps:
        expert_logits = expert_model(context)
        token = safe_sample(orig_logits, expert_logits, alpha, k)
    else:
        token = normal_sample(orig_logits)

    context.append(token)
    if token == eos_token:
        break
```

这里有一个关键前提：**expert 最好与原模型共享词表与 tokenizer**。否则“同一个位置上的 token 概率”没有可比性，交集操作会失去意义。工程上最稳妥的办法，是在同一基座模型上训练一个 LoRA expert，这样词表天然一致。

---

## 工程权衡与常见坑

SafeDecoding 看起来简单，但真正落地时，问题几乎都出在参数与回退策略上。

第一类坑是 $k$ 太小。top-$k$ 太小意味着交集更容易变窄，甚至只剩下极少数拒绝 token。这样确实能挡住越狱，但 benign 场景也可能被误杀。比如用户正常问“怎么用 Python 读 CSV”，结果模型频繁回“抱歉，我不能……”，本质上不是模型不会答，而是采样空间被压得过于保守。

第二类坑是 $\alpha$ 太偏 expert。$\alpha$ 太低时，系统会变成“只要有一点风险就拒绝”，帮助性明显下降。对客服机器人、文档助手、企业问答来说，这种误拒是很昂贵的，因为它直接影响用户完成任务。

第三类坑是启用步数过长。SafeDecoding 的设计重点是“卡住开头”，不是“全程控制文风”。如果你在几十个 token 上都持续启用，它会把正常回答也拉向模板化拒绝，尤其在创作、总结、解释类任务上表现明显。

第四类坑是词表或 tokenizer 不一致。原模型和 expert 如果不是同一基座，哪怕字面上看起来都输出英文，其 token 切分方式也可能不同。这样一来，$S_o\cap S_e$ 在实现层面会异常稀疏，甚至根本不可用。

下面用表格汇总常见问题与规避方式：

| 常见坑 | 现象 | 根因 | 规避措施 |
|---|---|---|---|
| $k$ 过小 | 回答单一、频繁拒绝 | 交集太窄 | 增大 $k$，并在 benign 集评估帮助性 |
| $\alpha$ 过小 | 正常问题也被拒 | 过度依赖 expert | 在安全集和 benign 集上联合调参 |
| 启用步数过长 | 输出风格僵硬 | 安全约束覆盖过多 token | 只在前 1 到 2 步启用 |
| 交集为空 | 无法采样或退化 | 两模型候选差异过大 | 设计 fallback，如 expert top-1 |
| 词表不一致 | 实现复杂、交集异常 | tokenizer 不兼容 | 用同一基座模型和 LoRA |
| 只看离线攻击集 | 线上表现不稳定 | 分布偏差 | 加入真实用户 benign 流量回放测试 |

一个很典型的真实工程现象是：你把 $k$ 从 5 改成 2 后，红队攻击通过率明显下降，但 MT-Bench 这类通用能力评测也开始掉分。原因不是模型“变笨”，而是 SafeDecoding 把很多有用开头也一起关掉了。对工程团队来说，正确做法通常不是追求绝对安全，而是在目标业务上寻找可接受的误拒率。

还要补一条实践经验：**SafeDecoding 应该被当成推理时防御层，而不是唯一安全层。** 真实系统里，它通常需要和输入分类、系统提示约束、工具权限隔离、输出审核一起使用。否则即便文本回答被挡住，工具调用仍可能成为旁路。

---

## 替代方案与适用边界

SafeDecoding 不是唯一方案。它的优势在于不改主模型、上线快、可调，但它也不是全能防线。把它与几类常见方案放在一起看，边界会更清楚。

| 方案 | 是否改主模型 | 实时性 | 对多样性的影响 | 擅长场景 | 局限 |
|---|---|---|---|---|---|
| SafeDecoding | 否 | 中等，需双模型前向 | 低到中，取决于步数与参数 | 推理时拦截 jailbreak 开头 | 主要作用于前几个 token |
| 强系统提示 / prefix 防护 | 否 | 高 | 低 | 快速加规则、零训练部署 | 易被更强提示覆盖 |
| 拒绝分类器 | 否 | 高，可前后置 | 低 | 作为网关或后审查 | 只能“判”，不能细调生成过程 |
| 全量安全微调 | 是 | 高，单模型推理 | 中 | 长期产品化模型 | 训练与回归成本更高 |

对新手来说，可以把区别理解为：

- SafeDecoding：**生成时双人投票，只在最关键前几步做安全约束。**
- 系统提示：**提前写规矩，希望模型自己遵守。**
- 拒绝分类器：**先让模型答，再由另一个模块判断要不要拦。**
- 安全微调：**直接把模型训练成更难被带偏。**

真实工程里，SafeDecoding 很适合下面几类场景：

| 适合场景 | 原因 |
|---|---|
| 开源聊天模型在线服务 | 不必重训主模型，改推理层即可 |
| 需要快速试验安全增强 | 参数少，A/B 测试成本低 |
| 已有同基座 LoRA 安全 expert | 词表一致，集成最顺畅 |
| 首 token 决定风险走向的对话系统 | 早期约束收益最大 |

不太适合的场景也要明确：

| 不适合场景 | 原因 |
|---|---|
| 工具调用权限很高的 agent | 文本安全不等于动作安全 |
| 长链路多轮诱导 | 只约束开头，不覆盖全链路状态演化 |
| 原模型与 expert 基座完全不同 | 词表与分布难以稳定融合 |
| 极端低时延服务 | 双模型推理会增加首 token 延迟 |

因此，SafeDecoding 的准确定位应该是：**一种轻量、推理期、token 级的安全增强方法**。它最适合补强“模型开头容易被诱导”的弱点，但不能替代训练、权限控制和系统级防御。

---

## 参考资料

- Xu et al.，《SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding》，ACL 2024 / arXiv:2402.08983。用途：方法定义、公式、实验结果的主来源。
- Hugging Face Papers 聚合页《SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding》。用途：快速浏览论文摘要、定位原论文与社区讨论。
- ICLR 2024 Workshop 项目介绍《SafeDecoding》。用途：补充项目背景、演示信息与会议信息。
