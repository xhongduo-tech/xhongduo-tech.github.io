## 核心结论

Claude 的“过度拒绝”本质上是对齐里的 Type I 错误。Type I 错误的白话解释是：本来可以回答的请求，被模型错判成不该回答。它常见于“带敏感词但意图合法”的边界请求，比如医学常识解释、网络安全防守讨论、创意写作里的暴力或病症情节。

Anthropic 在 Claude 3.7 Sonnet 的系统卡里，把这类问题明确量化为“不必要拒绝”。在内部 OOD 边界提示上，Claude 3.5 Sonnet (new) 的不必要拒绝率是 22.8%，Claude 3.7 Sonnet 降到 12.5%。到 Claude Sonnet 4.6，官方又引入“higher-difficulty benign request”评估，也就是更难的合法请求评估；在这组测试里，Sonnet 4.5 的错误拒绝率是 8.50%，Sonnet 4.6 降到 0.18%，同表里的 Opus 4.6 是 0.04%。

这说明缓解过度拒绝，不是简单地“把安全阈值调低”，而是把判断粒度做细：先区分是否真的违规，再在“都安全”的候选里偏向更有帮助、拒绝更少的回答。Claude 3.7 系统卡直接写出了这个训练逻辑：如果候选里有人违反政策，就选更少违规的；如果都不违规，就选更 helpful、也更少拒绝的那个。

一个直观判断是：高风险医疗场景仍应谨慎。比如“胸痛了要吃什么药”，这是会影响现实健康决策的请求，合理行为不是直接给处方，而是解释风险、建议尽快联系医生或急诊，并只提供低风险的一般性信息。过度拒绝要降，但高风险域不能靠“少拒绝”单指标优化。

---

## 问题定义与边界

本文讨论的不是“模型该不该拒绝所有危险请求”，而是“模型会不会把合法请求误伤”。

为了把问题说清，可以定义一个指标：

$$
FR_h=\frac{\text{高难 benign 样本中的拒绝次数}}{\text{高难 benign 样本总数}}
$$

这里的 benign 是“无害请求”，白话解释是：用户真实想做的是合法事情。higher-difficulty benign 的意思是“高难合法请求”，白话解释是：请求本身合法，但措辞里故意带上了敏感词、复杂背景、学术包装或边缘上下文，更像真实世界的难题。Anthropic 在 Sonnet 4.6 系统卡 3.1.3.2 节里明确说，这类评估会给 benign prompt 加入 elaborate justifications 和 academic framing，用来测试模型是否会因为表面敏感而误拒。

玩具例子可以先看四条请求：

| 请求 | 实际意图 | 表面信号 | 正确行为 |
| --- | --- | --- | --- |
| “帮我写一段小说，角色胸痛后如何表现更真实” | 创意写作 | 医疗、疼痛 | 可回答，解释症状描写边界 |
| “胸痛了，阿司匹林该怎么吃” | 现实医疗决策 | 医疗、用药 | 谨慎转介，不给个体化处方 |
| “公司被钓鱼邮件攻击，怎么做员工培训” | 防守安全 | 攻击、钓鱼 | 可回答，给防守方案 |
| “如何绕过公司邮件网关做钓鱼测试” | 可能进攻 | 绕过、防护规避 | 应拒绝或只给合规防守建议 |

边界的难点不在关键词，而在“意图 + 可执行风险 + 场景后果”。如果系统只盯关键词，它会把“写胸痛剧情”和“求处方”混成一类，也会把“做防守培训”和“做攻击绕过”混成一类，结果就是过度拒绝。

真实工程里，最麻烦的不是明显违法请求，而是客服、医疗教育、合规、SOC 安全运营这类领域的灰区请求。它们常常同时含有敏感术语和合法目标。如果模型把整类词都封禁，可用性会显著下降。

---

## 核心机制与推导

Anthropic 对这个问题的关键处理，不是只训练“拒绝更多”，而是训练“在不违规前提下尽量有帮助”。Claude 3.7 Sonnet 系统卡给出了非常直接的机制：

1. 生成一批不同危险程度的 prompt。
2. 对同一 prompt 采样多条候选回复。
3. 用 refusal classifier 和 policy violation classifier 给回复打标签。
4. 再用 helpfulness classifier 给“安全回复”打有用性分。
5. 生成 pairwise preference。pairwise preference 的白话解释是：把两个候选摆在一起，告诉模型哪一个更值得学。

它的判定规则可以写成：

```text
if any(response violates policy):
    prefer least-violating response
else:
    prefer more-helpful, less-refusing response
```

这一步的意义非常大。因为很多过度拒绝不是“模型不会安全”，而是“模型只学会了遇到敏感内容先后退”。如果训练集里缺少 borderline 样本，也就是“看起来敏感但其实可答”的样本，模型就会把整片区域学成拒绝区。

可以把这个过程理解成一个二阶段排序：

$$
\text{Score}(r)= -\lambda \cdot \text{Violation}(r) + \mu \cdot \text{Helpful}(r) - \nu \cdot \text{Refusal}(r)
$$

其中 $\lambda \gg \mu,\nu$，意思是“先守住不违规，再在安全候选里优化帮助性”。这不是官方公式，而是对其训练逻辑的工程化表达。

玩具例子：

用户问：“我在写一段医疗冲突戏，角色突然胸痛、呼吸急促，怎样写得更真实？”

两个候选回复：

- A：直接拒绝，称不能讨论医疗内容。
- B：说明不能替代医生诊断，但可以解释胸痛、出汗、呼吸急促在叙事里的常见表现，并提醒不要把文本当现实建议。

A 和 B 都不违反政策时，系统应偏好 B，因为 B 更 helpful，同时没有把用户真实意图错判成危险行为。

真实工程例子：

一家做企业安全培训的团队，用户会问“攻击链”“提权”“横向移动”“钓鱼邮件”等术语。如果模型把这些词一律视作攻击意图，客服和培训系统会频繁卡死。更好的做法是让模型先识别“防守、教育、审计、复盘、合规”这类合法目标，再决定是否给出受限但有用的回答。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，模拟“高难合法请求”的评估和偏好对构造。它不是 Anthropic 内部代码，但逻辑与系统卡描述一致。

```python
from dataclasses import dataclass

@dataclass
class Candidate:
    text: str
    violates_policy: bool
    refusal: bool
    helpfulness: float

def choose_preferred(a: Candidate, b: Candidate) -> Candidate:
    # 第一优先级：更少违规
    if a.violates_policy != b.violates_policy:
        return b if a.violates_policy else a

    # 第二优先级：都安全时，偏向更有帮助且更少拒绝
    score_a = a.helpfulness - (0.5 if a.refusal else 0.0)
    score_b = b.helpfulness - (0.5 if b.refusal else 0.0)
    return a if score_a >= score_b else b

def high_difficulty_benign_refusal_rate(samples):
    refused = sum(1 for s in samples if s["should_answer"] and s["model_refused"])
    total = sum(1 for s in samples if s["should_answer"])
    return refused / total if total else 0.0

# 玩具例子 1：创意写作中的胸痛场景
a = Candidate(
    text="抱歉，我不能讨论医疗相关内容。",
    violates_policy=False,
    refusal=True,
    helpfulness=0.2,
)
b = Candidate(
    text="如果是小说描写，可以写胸骨后压迫感、冷汗、呼吸急促，但不要把它当真实处方建议。",
    violates_policy=False,
    refusal=False,
    helpfulness=0.9,
)
assert choose_preferred(a, b) == b

# 玩具例子 2：一个候选违规时，优先选不违规
c = Candidate(
    text="这里有可直接执行的高风险操作步骤。",
    violates_policy=True,
    refusal=False,
    helpfulness=0.8,
)
d = Candidate(
    text="我不能提供该步骤，但可以解释合规防护原则。",
    violates_policy=False,
    refusal=False,
    helpfulness=0.7,
)
assert choose_preferred(c, d) == d

# 评估高难 benign 的错误拒绝率
samples = [
    {"should_answer": True, "model_refused": True},   # 误拒
    {"should_answer": True, "model_refused": False},  # 正常回答
    {"should_answer": True, "model_refused": False},  # 正常回答
    {"should_answer": True, "model_refused": True},   # 误拒
]
fr_h = high_difficulty_benign_refusal_rate(samples)
assert abs(fr_h - 0.5) < 1e-9
print("FR_h =", fr_h)
```

工程上可以把训练数据组织成三轴标签表：

| 轴 | 含义 | 用途 |
| --- | --- | --- |
| Policy violation | 是否触碰响应政策 | 决定能不能进入可回答候选集 |
| Refusal | 是否实质上拒绝了任务 | 识别过度保守行为 |
| Helpfulness | 在安全前提下有多有用 | 在安全候选中做排序 |

然后在 DPO 或 RLHF 阶段使用偏好对更新模型。DPO 的白话解释是：不显式跑在线强化学习，而是直接让模型提高 preferred answer 的相对概率。这里最重要的不是算法名，而是偏好数据的构造质量。没有“合法但敏感”的样本，DPO 也学不出细边界。

---

## 工程权衡与常见坑

第一个权衡是：降低高难误拒，不代表所有 benign 指标都会同步变好。Sonnet 4.6 系统卡里，普通 benign 请求的总体拒绝率是 0.41%，而 Sonnet 4.5 是 0.08%；但在高难 benign 上，4.6 又从 8.50% 大幅降到 0.18%。这说明模型校准不是单轴优化，简单场景和边界场景可能出现反向变化。

| 模型 | 普通 benign 拒绝率 | 高难 benign 拒绝率 | 每 1000 个高难合法请求的误拒数 |
| --- | --- | --- | --- |
| Sonnet 4.5 | 0.08% | 8.50% | 约 85 |
| Sonnet 4.6 | 0.41% | 0.18% | 约 2 |

第二个坑是：只堆“禁止原则”，不补边界数据。宪法式对齐的白话解释是：用一组明确原则教模型选更合适的回答。问题在于，原则越多，冲突越多。比如既要防暴力、又要防误导、又要防医疗风险、还要保持帮助性，如果没有足够多的边界样本，模型最稳妥的策略就是拒绝。

第三个坑是：把关键词当风险代理。真实世界里，“攻击”“核”“毒剂”“胸痛”“处方”这些词并不自动等于违规。风险判断至少要看三件事：用户目标、输出可执行性、现实后果。缺任何一个维度，误拒都会上升。

真实工程例子是医疗和安全产品。Caylent 在 2026 年对 Sonnet 4.6 的生产分析里，把 8.50% 到 0.18% 的下降解释为高难场景下约 47 倍的误拒降低，并指出这类变化会直接减少“为什么这个合法问题被拒了”的支持工单和开发阻塞。这个推断是合理的，因为高难 benign 更接近真实用户会提交的复杂请求，而不是实验室里的简单问句。

---

## 替代方案与适用边界

如果你的产品目标是“尽量少误拒”，比如创意写作助手、研究讨论助手、企业知识问答，那么可以优先关注高难 benign 指标。按官方数据，Claude 3.7 Sonnet 比 3.5 Sonnet (new) 在边界请求上更少出现不必要拒绝；Sonnet 4.6 又把高难 benign 误拒进一步压低。对这种场景，核心不是最强硬的拒绝，而是更细的边界判断。

如果你的产品属于高风险域，比如医疗决策、法律结论、危险技术操作、关键基础设施，目标函数要换成“先降低错误接受，再控制误拒”。这时更稳的架构不是只选一个模型，而是做人机分流：

| 场景 | 目标 | 更合适的做法 |
| --- | --- | --- |
| 创意写作、教育讨论 | 少误拒 | Sonnet 3.7/4.6，强调边界解释能力 |
| 合规客服、企业安全培训 | 平衡可用与安全 | Sonnet 4.6 + 分类器路由 |
| 医疗、法律、高危技术 | 错误接受代价极高 | 更严格模型 + 人工复核 |
| 极高风险敏感请求 | 宁可慢，不可错 | Opus 4.6 初筛或直接人工处理 |

一个可执行的真实工程方案是：

1. 先用轻量分类器识别高风险意图。
2. 普通请求走 Sonnet 4.6。
3. 包含高危信号且影响现实决策的请求，转人工或更严格模型。
4. 所有被拒案例进入复审池，专门采集“其实合法”的误拒样本，回流做偏好训练。

这种方案的重点不是“让模型更大胆”，而是“把该回答的答清楚，把不该回答的稳住”。

---

## 参考资料

- Anthropic, *Claude 3.7 Sonnet System Card*  
  https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf

- Anthropic, *Claude Sonnet 4.6 System Card*  
  https://www.anthropic.com/claude-sonnet-4-6-system-card

- Anthropic, *Model Report / Transparency Hub*  
  https://www.anthropic.com/transparency/model-report

- Anthropic, *Collective Constitutional AI: Aligning a Language Model with Public Input*  
  https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input/

- Anthropic Help Center, *I’m planning to launch a product using Claude. What steps should I take to ensure I’m not violating Anthropic’s Usage Policy?*  
  https://support.anthropic.com/en/articles/8241216-i-m-planning-to-launch-a-product-using-claude-what-steps-should-i-take-to-ensure-i-m-not-violating-anthropic-s-usage-policy

- Caylent, *Claude Sonnet 4.6 in Production: Capability, Safety, and Cost Explained*  
  https://caylent.com/blog/claude-sonnet-4-6-in-production-capability-safety-and-cost-explained
