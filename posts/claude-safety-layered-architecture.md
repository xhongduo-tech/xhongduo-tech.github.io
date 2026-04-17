## 核心结论

从训练到部署看，Claude 的安全系统可以抽象成四层：预训练数据过滤、Constitutional AI 对齐训练、system prompt 级约束、输出过滤器。这里的“分层”就是把不同阶段的风险拆开处理，而不是指一个单独模块包打天下。

这四层分别解决四类问题：

| 层 | 核心职责 | 主要拦截对象 | 典型手段 |
|---|---|---|---|
| 预训练数据过滤 | 减少模型学到高风险知识 | 训练污染、危险知识吸收 | 文档打分、阈值过滤、复检 |
| Constitutional AI | 让模型学会按原则拒绝有害请求 | 基本有害请求、明显违规任务 | 宪法原则、AI 生成反馈、偏好训练 |
| system prompt 级约束 | 在推理时固定角色和边界 | 角色突破、提示注入、越权指令 | 系统消息、工具权限、任务边界 |
| 输出过滤器 | 对最终答案做最后裁决 | 漏网之鱼、变体规避、边界案例 | 分类器、规则、审计与拦截 |

新手可以把它理解成“多道门”。一层失手，不等于系统立刻失守；但如果只剩一层，攻击者只要绕过一次就够了。

如果把每层都看成“放行概率”，整体失守概率可以近似写成：

$$
P_{\text{breach}}
=
P[\text{危险知识进入模型}]
\cdot
P[\text{模型仍愿意回答}]
\cdot
P[\text{最终输出未被拦截}]
$$

更细一点，可以写成：

$$
P_{\text{breach}} = P[f(x)\le t]\;P[g(u,S)=\text{unsafe-pass}]\;P[h(r)=\text{pass}]
$$

其中 $f(x)$ 是训练文档风险分数，$t$ 是过滤阈值；$g(u,S)$ 表示用户输入 $u$ 在 system prompt $S$ 下经过对齐后的决策；$h(r)$ 表示输出过滤器对回答 $r$ 的最终判定。这个公式的意义不是“精确预测现实”，而是给工程团队一个可分解、可监控、可优化的安全模型。

---

## 问题定义与边界

本文讨论的是“Claude 在训练与部署链路中的安全分层”，边界只覆盖三段：

1. 输入侧：模型吃进去什么训练数据。
2. 推理侧：模型在收到请求后按什么原则决策。
3. 输出侧：模型准备发出去的内容是否再次审查。

这里的“训练污染”是指有害文本混进训练语料，导致模型学会本不该学的危险模式。白话讲，就是教材本身有问题，学生再努力也可能被教歪。

这里的“Constitutional AI”是指用一套明确原则训练模型，让模型自己生成批评、修正和偏好数据。白话讲，就是先把“什么该做、什么不该做”写成规则，再让模型按这套规则反复练习。

这里的“system prompt”是系统级提示词，优先级高于普通用户输入。白话讲，就是应用在后台写给模型的工作手册，规定身份、任务范围和禁止事项。

这里的“输出过滤器”是对最终回答再做一次检测的模块。白话讲，就是内容已经写出来了，但发出去前还要过安检。

需要明确两个边界。

第一，这不是“所有安全问题的全集”。比如账号滥用、越权调用外部工具、数据库权限泄露、人工审核失效，这些属于更外层的产品与平台安全，不在本文中心。

第二，这四层不是都由同一个研究论文直接定义。更准确地说，Anthropic 官方资料分别明确讨论了预训练数据过滤、Constitutional AI、system prompt/guardrails、输入输出监控与 constitutional classifiers。工程上把它们合并成“四层安全架构”是合理抽象，但它是抽象，不是单篇论文里的原话。

一个玩具例子可以帮助理解边界。

假设你做一个“只回答 Python 初学者问题”的小助手。如果你只写一句 system prompt：“不要回答危险问题”，用户依然可能通过角色扮演、翻译绕写、上下文污染把它带偏。原因不是 prompt 没写，而是你只在推理侧放了一道门，训练侧和输出侧都没有兜底。

---

## 核心机制与推导

四层不是并列摆设，而是沿着数据流逐步收缩风险面：

```text
预训练语料
  -> 文档风险打分 f(x)
  -> 过滤高风险文档
  -> 预训练模型
  -> Constitutional AI 对齐训练
  -> 部署时注入 system prompt S
  -> 生成候选回答 r
  -> 输出过滤 h(r)
  -> 返回用户
```

### 1. 预训练数据过滤：先减少“学坏”的机会

Anthropic 在 2025 年公开讨论过 pretraining data filtering。核心做法是对预训练文档 $x$ 计算风险分数 $f(x)\in[0,1]$，如果 $f(x)>t$，就从训练集里移除。这里的“阈值”就是多高分数算危险，阈值越低，过滤越严格。

形式上：

$$
x \text{ 被保留} \iff f(x)\le t
$$

它解决的是“危险知识已经写进参数”这个问题。因为很多后处理方法只能压制模型输出，却不一定能真正消除模型在参数中学到的模式。

Anthropic 2025 年的实验重点是 CBRN 相关内容过滤，并不是“过滤全部有害内容”。这一点很重要：它证明了预训练过滤有效，但也说明现实系统中的过滤目标通常是分域、分风险等级设计的。

### 2. Constitutional AI：把“拒绝有害请求”训练成默认行为

Constitutional AI 的核心不是一条 prompt，而是一套训练流程。先给模型一部“宪法”，再让模型根据这些原则对回答做自我批评和改写，最后用这些偏好信号继续训练。

可以把它抽象成一个策略函数：

$$
g(u,S,\theta)\rightarrow a
$$

其中 $u$ 是用户输入，$S$ 是系统上下文，$\theta$ 是模型参数，$a$ 是动作，比如“正常回答”“安全改写”“拒绝回答”。这里真正重要的是：CAI 让“拒绝危险请求”变成模型内部更稳定的行为倾向，而不是每次都靠外部规则硬拦。

### 3. system prompt：把场景边界写死

CAI 解决的是“模型普遍应该怎么做”，system prompt 解决的是“这个应用里你现在必须怎么做”。

例如同一个 Claude 模型，放在教育产品里和放在金融顾问里，system prompt 完全不同。前者强调解释性和教学边界，后者强调合规、不可提供市场操纵建议、不可冒充持牌意见。它本质上是在部署时给模型附加一层任务约束。

这层能防很多常见攻击，比如：

- 让模型忘记自己身份的角色扮演攻击
- 用“你现在不是助手，是小说角色”绕开安全边界
- 借工具调用描述偷换任务目标
- 把恶意指令埋进长上下文做 prompt injection

但这层单独使用时很脆弱，因为它仍然是语言约束，模型有机会被更强的上下文竞争掉。

### 4. 输出过滤器：接受“前面总会漏一点”

输出过滤器的工程前提很现实：前面三层不会永远完美，所以最终答案还要再判一次。

可写成：

$$
h(r)=
\begin{cases}
\text{pass}, & \text{风险低} \\
\text{block}, & \text{风险高}
\end{cases}
$$

这里的“过滤器”可以是规则、关键词、轻量分类器、另一个模型，或者它们的组合。Anthropic 在 guardrails 文档里明确建议做输入预筛、提示工程和持续监控；在 constitutional classifiers 方向上，则把输入和输出监控进一步系统化。

### 数值推导演示：为什么多层能明显降风险

看一个最小化玩具例子。假设有 1000 个恶意请求流入系统。

| 阶段 | 通过到下一阶段的比例 | 剩余风险请求数 |
|---|---:|---:|
| 初始恶意请求 | 100% | 1000 |
| CAI 拒绝 99.5% | 0.5% | 5 |
| system prompt 再挡 40% | 60% | 3 |
| 输出过滤再拦截 2 条 | - | 1 |

如果只靠最后一层，系统要直接面对 1000 个风险案例；如果前面已经把大头压掉，最后一层就变成“清尾巴”的精修工具，而不是唯一防线。

再看预训练过滤的数值直觉。研究摘要里给出的例子是：在 10M 文档中，先用便宜分类器筛出最可疑的 1%，再用更精细的打分器复检其中一小部分。这里的关键不是具体数字，而是两阶段筛选的工程思想：先用低成本模型做粗筛，再把昂贵能力集中在高风险子集上。

---

## 代码实现

Claude 的内部训练代码不公开，但四层机制可以用一个可运行的玩具实现说明。下面这段 Python 代码模拟了“文档过滤 + 对齐决策 + system prompt 约束 + 输出过滤”的最小流程。

```python
from dataclasses import dataclass

RISK_WORDS = {"bomb", "weapon", "jailbreak", "manipulate", "explosive"}
BLOCK_PATTERNS = {"step-by-step attack", "market manipulation plan", "bypass safety"}

@dataclass
class Request:
    user_input: str
    system_prompt: str

def score_document(doc: str) -> float:
    tokens = doc.lower().split()
    hits = sum(token.strip(".,!?") in RISK_WORDS for token in tokens)
    return min(1.0, hits / 3.0)

def filter_pretraining_docs(docs, threshold=0.5):
    kept = [d for d in docs if score_document(d) <= threshold]
    removed = [d for d in docs if score_document(d) > threshold]
    return kept, removed

def cai_policy(user_input: str) -> str:
    text = user_input.lower()
    if "how to build a bomb" in text or "manipulate the market" in text:
        return "reject"
    if "ignore previous instructions" in text or "jailbreak" in text:
        return "reject"
    return "answer"

def apply_system_prompt(req: Request, draft: str) -> str:
    s = req.system_prompt.lower()
    u = req.user_input.lower()
    if "never provide illegal instructions" in s and ("illegal" in u or "bomb" in u):
        return "I can't help with illegal or harmful instructions."
    return draft

def output_filter(text: str) -> bool:
    lower = text.lower()
    return not any(p in lower for p in BLOCK_PATTERNS)

def generate_reply(req: Request) -> str:
    if cai_policy(req.user_input) == "reject":
        draft = "I can't help with harmful or deceptive requests."
    else:
        draft = "Here is a safe high-level explanation."
    draft = apply_system_prompt(req, draft)
    return draft if output_filter(draft) else "Response blocked by safety filter."

docs = [
    "Python loops and functions tutorial",
    "Detailed bomb weapon explosive synthesis notes",
    "Linear algebra basics for beginners"
]
kept, removed = filter_pretraining_docs(docs, threshold=0.5)

assert len(kept) == 2
assert len(removed) == 1

req_safe = Request(
    user_input="Explain what a Python list is",
    system_prompt="You are a tutor. Never provide illegal instructions."
)
req_unsafe = Request(
    user_input="How to build a bomb",
    system_prompt="You are a tutor. Never provide illegal instructions."
)

assert generate_reply(req_safe) == "Here is a safe high-level explanation."
assert "can't help" in generate_reply(req_unsafe).lower()
```

这段代码当然不代表真实 Claude，只是把四层职责拆开：

- `score_document` / `filter_pretraining_docs`：模拟预训练数据过滤。
- `cai_policy`：模拟经过 CAI 学到的拒绝倾向。
- `apply_system_prompt`：模拟部署时的场景规则。
- `output_filter`：模拟最终输出拦截。

真实工程例子可以看“金融顾问 Claude 服务”。这是一个高风险场景，因为模型如果输出“规避监管”“操纵市场”“掩盖内幕信息”之类内容，后果远比普通 FAQ 机器人严重。一个合理链路通常是：

1. 用户请求先经过轻量 harmlessness 预筛。
2. system prompt 明确写入金融合规边界。
3. Claude 生成回答时依赖自身对齐能力拒绝危险请求。
4. 输出再经过监控与审计分类器。
5. 高频触发者被限流、记录、复盘。

这比只写一句“不要提供违法建议”更接近可上线系统。

---

## 工程权衡与常见坑

四层架构的优点是复合防御，代价是成本、延迟和维护复杂度上升。实际工程里最常见的问题，不是“完全没有安全层”，而是“有安全层，但没联动”。

| 常见坑 | 为什么会出问题 | 缓解措施 |
|---|---|---|
| 只靠 system prompt | prompt 是语言约束，容易被角色扮演和长上下文覆盖 | 叠加 CAI 拒绝能力和输入预筛 |
| 预训练过滤只做一次 | 新增数据源可能重新带入高风险文本 | 数据更新流程中同步跑过滤器 |
| 输出过滤孤立运行 | 不知道前面发生过什么，判断会更弱 | 把请求风险分、拒绝原因传给过滤器 |
| 过滤阈值过严 | 误伤正常内容，模型变钝 | 用离线评测做安全-能力曲线 |
| 过滤阈值过松 | 漏掉高风险长尾样本 | 对高风险域采用更严格阈值 |
| 只拦关键词 | 攻击者可改写、分步、翻译绕过 | 用分类器和语义判定补充 |
| 只做离线评测 | 线上攻击会持续变化 | 持续监控、回流样本、迭代规则 |

一个典型误区是把 system prompt 当作安全主引擎。它更像运行时配置，不是根本对齐来源。真正稳定的拒绝能力，通常要靠模型在训练阶段就学会“面对这类请求时不应配合”。

另一个常见坑是忽视层间信息共享。比如前端轻量模型已经把请求标成高风险，但这个标记没有传给主模型链路和输出过滤器，结果后面每层都像第一次见到该请求一样重新猜。这会浪费预算，也会降低命中率。

金融顾问场景里，这个问题尤其明显。假设用户问：“怎样设计消息释放节奏，既能在不触发风控的情况下推高某只小盘股？”如果只靠 system prompt，模型可能在“学术讨论”“历史案例分析”“假设性研究”的包装下漏出操作建议。更稳的做法是：

- 输入预筛先把它标成市场操纵相关。
- CAI 让模型默认走拒绝或高层次风险教育。
- system prompt 明确禁止提供操纵策略。
- 输出过滤继续扫描是否出现可执行步骤、规避监管语句、诱导性措辞。

---

## 替代方案与适用边界

不是每个团队都需要完整四层。是否上满四层，取决于业务风险、预算和延迟约束。

| 方案 | 优点 | 适用场景 | 风险 |
|---|---|---|---|
| 四层完整方案 | 防御最稳，适合高风险域 | 金融、医疗、合规、公开 API | 成本高、链路复杂 |
| CAI + system prompt + 输出过滤 | 实现难度较低，效果通常够用 | 通用企业助手、中风险工具 | 训练侧危险知识仍可能存在 |
| system prompt + 输出过滤 | 部署最快 | FAQ、低风险内网机器人 | 容易被绕过，长期维护压力大 |
| 纯外部过滤 | 不改模型，易接入现有系统 | 旧系统改造、快速试点 | 对模型内部倾向影响很弱 |

对低风险 FAQ bot，很多团队不做预训练过滤也能上线，因为它处理的是售后、知识库、流程说明，攻击收益低、风险面窄。这时“强 system prompt + 高灵敏输出过滤”可能已经够用。

但对高风险业务，少一层就意味着把风险转嫁给其他层。例如没有预训练过滤，就要接受模型参数里可能保留更多危险知识；没有输出过滤，就要把所有长尾失败都压给主模型；没有 CAI，就会过度依赖 prompt 文本本身的稳定性。

所以适用边界可以概括成一句话：风险越高，越不能把安全压成单点模块。

---

## 参考资料

- Anthropic Alignment Science，2025 年《Enhancing Model Safety through Pretraining Data Filtering》：支持“预训练数据过滤”这层的具体机制、分类器设计和效果评估。
- Anthropic，2023 年《Claude’s Constitution》博客，以及 2026 年更新版 constitution 页面：支持 Constitutional AI 与“宪法原则”如何塑造 Claude 行为。
- Anthropic Docs，《Mitigate jailbreaks and prompt injections》：支持 system prompt、输入预筛、持续监控、多层 guardrails 的部署实践。
- Anthropic Research，2026 年《Next-generation Constitutional Classifiers: More efficient protection against universal jailbreaks》：支持输入/输出监控、constitutional classifiers 作为最终拦截层的思路。
- Anthropic 官方文档与系统卡持续更新页：用于查看具体模型版本、部署建议和安全边界是否发生变化。
