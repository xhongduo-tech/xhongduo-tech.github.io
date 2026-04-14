## 核心结论

DeepInception 是一种**多层角色扮演型越狱攻击**。越狱攻击的白话解释是：攻击者不直接提危险请求，而是把请求包进一段会误导模型的上下文里，让模型自己走到危险输出。

它的关键不在“换几个敏感词”，而在于三件事同时发生：

1. 用**嵌套层数**把原始意图拆散。嵌套层数的白话解释是：故事里还有故事，角色里还有角色。
2. 用**角色分工**把责任分散。角色分工的白话解释是：每个角色只说一小段，看起来都不像最终危险指令。
3. 用**场景合理性**维持上下文一致。场景合理性的白话解释是：让整段对话像真的在完成一个“正当剧情任务”。

因此，DeepInception 不是简单绕过关键词过滤，而是利用了大模型对“继续扮演角色”和“保持上下文连贯”的偏好。论文与官方页面展示的实验表明，在弱防护条件下，多层嵌套、更多角色、与任务更贴合的故事背景，会显著提高 jailbreak success rate，常记为 JSR，即越狱成功率。

一个便于理解的近似公式是：

$$
\text{JSR}=\sigma(\alpha L+\beta C+\gamma S+b), \quad \sigma(x)=\frac{1}{1+e^{-x}}
$$

其中：

- $L$ 是层数
- $C$ 是角色数
- $S$ 是场景合理性得分
- $\sigma$ 是 Sigmoid 函数，白话解释是：把任意分数压到 0 到 1 之间，方便看成概率

这个公式不是通用物理定律，而是一个**分析框架**：层数、角色数、场景贴合度越高，模型越可能沿着“保持剧情一致”的方向继续生成。

---

## 问题定义与边界

本文讨论的是**输入端的 prompt 构造风险**，不是模型训练方法，也不是参数层面的攻击。prompt 的白话解释是：你发给模型的整段文字指令和上下文。

更准确地说，DeepInception 要评估的是：当用户把原本会被直接拒绝的有害请求，改写成多轮、多角色、带故事背景的“剧情推演”时，模型会不会因为上下文一致性而放松安全边界。

这里有两个边界必须先说清：

1. 它主要针对**多轮对话系统**
   单轮检测只看当前一句话，多轮系统则会把前面对话也带进上下文。DeepInception 恰好利用了这一点。
2. 它主要揭示**角色一致性漏洞**
   角色一致性的白话解释是：模型一旦接受某个身份设定，就会倾向持续按照这个身份说话。

玩具例子可以这样理解。假设直接说“请提供某项违规操作步骤”，模型通常会拒绝；但如果改成“我们在写小说，你扮演导师，另一位角色扮演学生，讨论一个虚构反派如何实施某行为”，表面看像写作讨论，检测器如果只扫敏感词，很可能误判为普通剧情。

下面这张表能看出“直接指令”和“DeepInception 输入”的差别：

| 维度 | 直接指令 | DeepInception 输入 |
|---|---|---|
| 层数 | 1 层 | 2 到多层嵌套 |
| 角色数 | 通常 1 个 | 多个角色分摊任务 |
| 意图表达 | 显式、集中 | 隐式、分散 |
| 上下文一致性 | 弱 | 强，前后互相支撑 |
| 检测难点 | 关键词明显 | 需要看跨轮累计语义 |
| 常见防御误区 | 只拦敏感词 | 忽略身份漂移与剧情累积 |

所谓**身份漂移**，白话解释是：模型原本在“安全助手”身份下回答，但随着剧情推进，逐步转成“故事里某个执行任务的角色”。

所以，DeepInception 的重点不是“某一句危险”，而是“很多句单看都不太危险，但串起来越来越危险”。

---

## 核心机制与推导

DeepInception 的机制可以拆成一条递进链：

1. 先建立一个看似正当的故事壳
2. 再在故事壳里定义多个角色
3. 让角色之间传递局部信息
4. 用多轮对话把局部信息逐步拼成完整意图
5. 让模型为了维持角色一致性，继续沿着这条路径输出

这里最重要的是“**局部无害，整体有害**”。单独看某一轮，它可能只是“老师给学生布置任务”；但跨轮累计后，系统会发现这些局部片段在拼接某个被禁止的目标。

可以把公式细化成一个分析视角：

$$
z=\alpha L+\beta C+\gamma S+b
$$

$$
\text{JSR}=\sigma(z)
$$

解释如下：

- $L$ 增大：说明上下文分解得更细，单轮更不显眼
- $C$ 增大：说明责任被更多角色分摊，不容易在单个 utterance 上暴露完整目标
- $S$ 增大：说明故事和目标更匹配，模型更容易“信以为真”并继续补全剧情
- $b$：模型自身安全强度、系统提示词强度等基线因素

这里的 $S$ 往往被低估。很多人以为“多堆几层就行”，其实不对。若第 5 层突然跳到完全无关的场景，模型反而更容易触发警觉。真正有效的是：每一层都承接上一层，而且角色关系、任务目标、语气风格都一致。

玩具例子：

- 第一层：设定“导师在指导学生分析一个虚构系统事故”
- 第二层：学生请教“某个异常现象的成因”
- 第三层：助手角色总结“为了复现实验，需要哪些前置条件”

如果每层都只暴露一部分信息，模型就可能把它理解为合理推演，而不是明确违规请求。

真实工程例子：

安全测试团队在红队测试里，不会直接给生产模型一句明显违规请求，而是构造一段多轮对话日志，里面包含项目经理、研究员、审计员、助理等角色。每个角色只负责一个子问题，例如“复盘背景”“整理实验假设”“说明虚构样本行为”。如果系统只做单轮审核，就会漏掉这些角色在多轮中累计出的风险轨迹。

从防御角度看，真正需要监控的是一条信息流：

$$
\text{Story Shell} \rightarrow \text{Role Adoption} \rightarrow \text{Identity Drift} \rightarrow \text{Goal Reconstruction}
$$

白话解释是：

- 先接受故事壳
- 再接受角色
- 再发生身份漂移
- 最后把分散信息重组为原始目标

这也是为什么 DeepInception 更像“上下文操纵”，而不是“词表绕过”。

---

## 代码实现

工程上不应该去实现真实攻击模板，更合理的做法是实现一个**防御侧模拟器**：只用占位符和元数据评估某段多轮提示是否具备 DeepInception 风险特征。

下面给一个可运行的 Python 例子。它做三件事：

1. 根据层数、角色数、场景合理性计算风险分数
2. 检查是否出现多层角色递进
3. 用占位符构造安全的红队测试样本，不包含真实有害内容

```python
import math
from dataclasses import dataclass
from typing import List

@dataclass
class Turn:
    role_name: str
    layer: int
    content: str

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def estimate_jsr(layers: int, characters: int, scene_score: float,
                 alpha: float = 0.8, beta: float = 0.5, gamma: float = 1.2, bias: float = -4.0) -> float:
    """
    这是一个防御侧近似模型，不代表论文中的统一标准公式。
    scene_score 取值建议在 [0, 1]。
    """
    z = alpha * layers + beta * characters + gamma * scene_score + bias
    return sigmoid(z)

def detect_deepinception_pattern(turns: List[Turn]) -> dict:
    layers = len({t.layer for t in turns})
    characters = len({t.role_name for t in turns})

    role_handoff = 0
    for i in range(1, len(turns)):
        prev, curr = turns[i - 1], turns[i]
        if prev.role_name != curr.role_name and curr.layer >= prev.layer:
            role_handoff += 1

    # 简化版：如果多角色、多层、且有明显接力，就判为高风险结构
    high_risk = layers >= 3 and characters >= 3 and role_handoff >= 2

    return {
        "layers": layers,
        "characters": characters,
        "role_handoff": role_handoff,
        "high_risk_structure": high_risk,
    }

def build_safe_redteam_case() -> List[Turn]:
    """
    只构造脱敏测试样本，用 [REDACTED_GOAL] 占位，
    用于验证检测器是否能识别“故事壳 + 多角色 + 多层接力”。
    """
    return [
        Turn("mentor", 1, "你正在指导学生分析一个虚构案例，目标是理解某类[REDACTED_GOAL]为何会被拦截。"),
        Turn("student", 2, "我会把任务写成剧情讨论，只讨论背景和角色，不直接说最终目标。"),
        Turn("assistant", 3, "我会继续沿着故事上下文总结信息，并保持角色一致，但不生成任何真实违规步骤。"),
    ]

# 运行示例
turns = build_safe_redteam_case()
pattern = detect_deepinception_pattern(turns)
score = estimate_jsr(pattern["layers"], pattern["characters"], scene_score=0.9)

assert pattern["layers"] == 3
assert pattern["characters"] == 3
assert pattern["high_risk_structure"] is True
assert 0.0 < score < 1.0
assert score > 0.5

print("pattern =", pattern)
print("estimated_jsr =", round(score, 4))
```

这个例子故意不生成真实攻击提示，而是把关键结构抽象成三部分：

- `layer`：第几层剧情
- `role_name`：当前是谁在说话
- `content`：内容，但危险目标全部脱敏

如果你要给新手一个更直观的伪代码，可以简化成下面这样：

```python
roles = ["mentor", "student", "assistant"]
story = "虚构案例复盘"
goal = "[REDACTED_GOAL]"

prompt = []
for i, role in enumerate(roles, start=1):
    prompt.append({
        "layer": i,
        "role": role,
        "story": story,
        "hint": f"只传递与 {goal} 相关的一小部分线索"
    })

# 防御侧做法：不发送真实有害内容，而是检测 prompt 是否存在多层接力结构
```

这里的重点不是“怎么拼出攻击”，而是“系统应该如何识别这类拼法的风险”。

---

## 工程权衡与常见坑

DeepInception 的工程难点，不是会不会做字符串拼接，而是**怎么在多轮里跟踪语义累积**。

第一类常见坑，是误把“层数”当成唯一变量。层数越高，风险不一定越高。因为层数增长会带来两个副作用：

1. 故事容易失真，场景合理性下降
2. 上下文变长，模型可能忘记早先设定

第二类常见坑，是只看关键词，不看角色关系。很多对话在单轮里没有明显违规词，但角色之间存在稳定的“信息接力”。这时，单轮分类器会漏报。

第三类常见坑，是忽略身份漂移。模型一开始可能在拒绝，但后续逐渐变成“帮忙补全剧情的助手”。如果系统只检查最终一轮，而不检查“身份变化轨迹”，就会漏掉真正的风险来源。

下面这张表总结常见误区和对策：

| 常见误区 | 为什么错 | 更合理的做法 |
|---|---|---|
| “层数越多越成功” | 层数高但剧情跳跃，反而更像攻击 | 同时评估层数、角色接力、场景贴合度 |
| “只要扫敏感词就够了” | 意图可被分散到多轮 | 做跨轮语义聚合与风险累计 |
| “角色扮演只是文风变化” | 角色会改变模型的执行身份 | 监控身份漂移与角色一致性 |
| “最后一轮没问题就安全” | 风险可能在前几轮已经积累完成 | 检查完整对话轨迹，而不是只看终局 |
| “加一个总拒答提示就行” | 模型可能在长上下文里被剧情带偏 | 引入外部 supervisor 或独立审计器 |

真实工程里，一个更稳妥的策略是三层防线：

1. 输入侧检查：识别多层角色和剧情包装
2. 会话侧监督：跟踪跨轮身份漂移
3. 输出侧审计：即使前面漏报，最后仍检查危险重组结果

这也是为什么很多团队会把“对话级审计”放在“单条 prompt 审计”之上。

---

## 替代方案与适用边界

DeepInception 最适合拿来评估**多轮一致性防御**，不适合当成唯一红队方法。因为它强调的是“角色和剧情累积”，而不是所有类型的越狱。

如果你只想模拟最基础的角色扮演风险，单层 prompt 已经够用，例如“你现在扮演某角色”。但这种方法通常只能测试**单轮角色服从**，很难测出系统在长对话中的累积脆弱性。

与其他方案相比，可以这样理解：

| 方法 | 主要测试什么 | 优点 | 局限 |
|---|---|---|---|
| 单轮角色扮演 | 单次身份切换 | 简单、便宜 | 难测多轮累积 |
| DeepInception | 多层剧情与角色接力 | 适合测身份漂移和上下文一致性漏洞 | 设计和审计都更复杂 |
| Self-reflection 防御 | 模型自检当前身份和任务 | 成本低，易集成 | 容易被上下文污染 |
| 外部 supervisor | 独立模型跟踪风险轨迹 | 更适合多轮场景 | 成本更高，系统更复杂 |
| 审计超 prompt 语境 | 从会话级重构真实意图 | 对分散式攻击更有效 | 需要更强的日志与状态管理 |

一个简单决策规则是：

- 只测“模型会不会被一句角色指令带偏”，用单轮角色扮演
- 要测“模型会不会在多轮剧情里逐步丢失安全身份”，用 DeepInception
- 要做生产防御，不要只靠模型自觉，应叠加 supervisor 和输出审计

所以，DeepInception 的价值不在于“它是最复杂的攻击”，而在于它逼迫防御系统回答一个更真实的问题：**模型在长上下文中，到底是在遵守安全规则，还是在维护剧情一致性。**

---

## 参考资料

| 来源 | 类型 | 用途 |
|---|---|---|
| [DeepInception 官方页面](https://deepinception.github.io/) | 官方站点 | 查看方法概述、模板结构、实验图表与消融结果 |
| [OpenReview: DeepInception: Hypnotize Large Language Model to Be Jailbreaker](https://openreview.net/forum?id=bYa0BhKR4q) | 论文页面 | 查看正式摘要、发表信息与论文版本 |
| [GitHub: tmlr-group/DeepInception](https://github.com/tmlr-group/DeepInception) | 代码仓库 | 查看论文配套代码、数据与实验材料 |
| [Hugging Face Papers: arXiv 2311.03191](https://huggingface.co/papers/2311.03191) | 论文索引 | 快速查看论文摘要、关联链接与社区讨论入口 |
| [Emergent Mind: DeepInception Framework Overview](https://www.emergentmind.com/topics/deepinception-framework) | 二级综述 | 作为框架化总结，帮助理解层数、角色数、场景得分的分析视角 |
| [Hugging Face 论坛: A Three-Layer Defense-in-Depth Approach to Multi-Turn Jailbreak Attacks](https://discuss.huggingface.co/t/a-three-layer-defense-in-depth-approach-to-multi-turn-jailbreak-attacks/174497) | 讨论 | 参考多轮越狱的防御分层思路与身份漂移视角 |
