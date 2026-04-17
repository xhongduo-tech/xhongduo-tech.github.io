## 核心结论

对话历史不该只靠固定窗口裁剪，而应当做“分层保留 + 增量摘要 + 衰减检索”。这里的“增量摘要”是指只对旧内容持续压缩，不反复重写整段历史；“衰减检索”是指给每条记忆一个随时间下降的可检索分数。一个常用形式是

$$
R=e^{-t/S}
$$

其中 $R$ 是可检索性，意思是“这条记忆现在有多容易被找回”；$t$ 是距离写入或上次强化后的时间；$S$ 是稳定度，意思是“这条记忆有多抗遗忘”。

核心做法不是“把旧内容删掉”，而是“把旧内容换一种更便宜的表示”。最近几轮保留原文细节，中间层保留关键句摘要，更久远的内容只保留主题标签、事实槽位或结构化结论。这样做的结果是：上下文预算主要花在当前任务上，但旧事实仍可被唤醒。

对新手最重要的结论只有三个。

| 结论 | 直接含义 | 工程效果 |
| --- | --- | --- |
| 近期信息保留原文 | 最近几轮最影响当前推理 | 降低答非所问 |
| 远期信息转为摘要 | 旧内容不再占满上下文 | 降低 token 成本 |
| 访问会强化稳定度 | 被反复引用的事实不容易丢 | 长任务中减少“断片” |

玩具例子：把最近 3 轮完整保留，第 4 轮到第 20 轮保留关键句摘要，第 20 轮以前只保留“用户偏好 / 已确认约束 / 已完成步骤”。这就像笔记里最近几页不折叠，旧章节只留提纲。系统并没有忘记旧内容，而是改成了低成本表示。

如果把用户给定实验设定作为目标指标，那么在 20 轮以上对话里，这种“遗忘曲线 + 摘要层”的记忆管理，相比固定窗口可取得约 12% 的任务完成率提升。更保守地看，公开论文也已经报告了超过 10% 的准确率增益，因此这个方向不是概念包装，而是有实验支撑的工程方案。

---

## 问题定义与边界

问题定义可以写得很具体：给定一个多轮对话序列，系统要决定每一轮历史在后续推理中应当以哪种粒度保留，以及何时从“原文”降级为“摘要”。

固定滑动窗口的问题很直接。它只回答“保留最后 N 轮”，却不回答“被删掉的信息是否仍然重要”。一旦对话超过 20 轮，前 10 轮可能恰好包含用户身份、任务约束、已经试错过的方案或关键定义。窗口把它们整段切掉，后面的推理就会像换了一个失忆模型。

这个问题的边界也要说清楚。本文讨论的是“多轮 Agent 对话记忆管理”，不是训练阶段的参数遗忘，也不是向量数据库的离线知识库建设。输入主要有四类：

| 输入量 | 含义 | 典型来源 |
| --- | --- | --- |
| 时间 $t$ | 距写入或上次访问经过了多少轮/分钟 | 会话时间戳 |
| 重要度 $I$ | 这条内容对目标有多关键 | 规则打分或 LLM 判别 |
| 访问频率 $f$ | 这条内容被检索、引用了几次 | 检索日志 |
| 冲突状态 $c$ | 是否已被更新信息推翻 | 事实校验模块 |

短期层和长期层通常不是同一种参数。

| 层级 | 保存内容 | 典型 $t$ 范围 | 稳定度 $S$ 特征 | 摘要频率 |
| --- | --- | --- | --- | --- |
| 短期层 | 最近原文、工具返回、局部推理 | 1 到 3 轮 | 小但直接可见 | 很低 |
| 语义层 | 关键句摘要、事实槽位、任务状态 | 4 到 20 轮 | 中等，可随访问增长 | 中等 |
| 长期层 | 用户画像、长期偏好、稳定规则 | 20 轮以上 | 初始高，衰减慢 | 低，但会定期重写 |

这里的“语义层”可以白话理解为“不是保存原话，而是保存原话表达的意思”。

真实工程例子：一个合规审查 Agent 连续处理 37 份跨境交易报告。最近报告的上下文必须保留原文，因为当前判断依赖细节措辞；而历史报告中的共同规则、已确认的判例映射、客户偏好写法，则进入语义层或长期层。到第 38 份报告时，系统不需要把前 37 份全文再塞进模型，只需拉起与当前规则最相关的摘要和长期事实。这样才能在成本受限时保持稳定引用。

---

## 核心机制与推导

先看最小数学形式：

$$
R=e^{-t/S}
$$

它表达的是单调衰减关系。$t$ 越大，$R$ 越小；$S$ 越大，衰减越慢。对工程实现来说，这个式子有三个直接好处。

第一，它连续可调，不是生硬的“保留或删除”二选一。  
第二，它允许“复习效应”，即访问后提高 $S$。  
第三，它很容易和重要度、频率结合成复合分数。

一个常见扩展是：

$$
S = S_0 \cdot (1+\alpha I+\beta \log(1+f))
$$

这里 $S_0$ 是初始稳定度，$I$ 是重要度，$f$ 是访问频率，$\alpha,\beta$ 是权重。白话解释：重要内容和被反复用到的内容更不容易被忘。

再定义一个记忆动作规则：

$$
\text{policy}(R)=
\begin{cases}
\text{保留原文}, & R>0.7 \\
\text{保留关键句摘要}, & 0.3<R\le 0.7 \\
\text{只保留主题标签或结构化槽位}, & R\le 0.3
\end{cases}
$$

这就把“遗忘”变成了“信息表示的切换”，而不是“信息直接消失”。

玩具例子可以直接算。设一条记忆初始稳定度 $S=8$，10 轮后再检索：

$$
R=e^{-10/8}\approx 0.2865
$$

这意味着它已经低于 0.3 阈值，不适合继续保留全文，而更适合保留摘要或事实标签。假设它此时又被命中一次，并把稳定度提升到 $S=14$，则

$$
R'=e^{-10/14}\approx 0.489
$$

同样是 10 轮前的内容，因为被再次访问，系统就有理由把它从“只留标签”提升到“保留关键句摘要”。这就是“旧事实重生”的工程含义。

把这个机制放到多层记忆中，可以理解为一个梯级系统：

1. 新内容先进入短期层，默认保留原文。
2. 当它变旧时，系统按 $R$ 决定是否摘要压缩。
3. 若后续任务再次依赖它，就提高 $S$，并把它提回更高可见层。
4. 若发现新事实与旧事实冲突，就不是简单增加一条，而是做冲突融合或标记失效。

“冲突融合”可以白话理解为：系统要判断“旧说法被新证据推翻了吗”。如果答案是是，那么旧记忆不该继续和新记忆并列生效，否则检索时会把过期内容再次拉回来。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现。它不依赖外部库，展示的是最小机制：写入记忆、计算遗忘分数、访问强化、低分摘要化。

```python
from math import exp, log

class MemoryItem:
    def __init__(self, text, timestamp, importance, frequency=0, base_stability=4.0):
        self.text = text
        self.timestamp = timestamp
        self.importance = importance
        self.frequency = frequency
        self.base_stability = base_stability
        self.summary = None
        self.archived = False

    def stability(self):
        # importance 越高、被访问越多，稳定度越高
        return self.base_stability * (1 + 0.8 * self.importance + 0.5 * log(1 + self.frequency))

    def retrievability(self, current_round):
        t = max(0, current_round - self.timestamp)
        s = self.stability()
        return exp(-t / s)

    def access(self):
        self.frequency += 1

    def maybe_summarize(self, current_round, threshold=0.3):
        r = self.retrievability(current_round)
        if r <= threshold and not self.summary:
            self.summary = self.text[:18] + "..." if len(self.text) > 18 else self.text
            self.archived = True
        return r

mem = MemoryItem(
    text="用户要求输出 CSV，字段顺序必须是 id,name,total，金额保留两位小数",
    timestamp=0,
    importance=0.9
)

r1 = mem.retrievability(current_round=2)
assert 0 < r1 <= 1

r2 = mem.maybe_summarize(current_round=10)
assert r2 < r1

mem.access()
r3 = mem.retrievability(current_round=10)
assert r3 > r2  # 访问后稳定度增加，可检索性回升

if r2 <= 0.3:
    assert mem.archived is True
    assert mem.summary is not None
```

这个例子里，`importance` 是“这条内容对当前任务有多关键”，`frequency` 是“它被重新取用过几次”。即使是最小实现，也已经体现出三个动作：衰减、强化、摘要。

如果换成前端 Agent 或博客站点中的 JavaScript 逻辑，通常会维护一个记忆对象：

```javascript
function calcRetrievability(currentRound, memory) {
  const t = Math.max(0, currentRound - memory.timestamp);
  const s = memory.stability * (1 + 0.6 * memory.importance + 0.4 * Math.log1p(memory.frequency));
  return Math.exp(-t / s);
}

function updateMemoryLayer(memory, currentRound) {
  const r = calcRetrievability(currentRound, memory);

  if (memory.conflicted) {
    memory.layer = "deprecated";
    return memory;
  }

  if (r > 0.7) {
    memory.layer = "short_term";
    memory.payload = memory.rawText;
  } else if (r > 0.3) {
    memory.layer = "semantic";
    memory.payload = memory.summary || summarize(memory.rawText);
  } else {
    memory.layer = "long_term";
    memory.payload = memory.tags || extractFacts(memory.rawText);
  }

  return memory;
}
```

摘要触发逻辑可以概括成一句话：当 `current_round - timestamp` 持续增大，且 `R` 低于阈值时，系统不再为这条记忆支付“原文展示成本”，而是只保留语义压缩结果。

真实工程例子：客服 Agent 在第 1 轮得知“用户不接受电话联系，只能邮件沟通”，第 18 轮又开始处理退款规则，第 31 轮继续追问物流编号。固定窗口可能已经把第 1 轮裁掉，但衰减式系统会把“仅邮件沟通”作为高重要度长期偏好保存在长期层，不需要每轮原文都在，却能在第 31 轮继续生效。

---

## 工程权衡与常见坑

这套方案的收益来自“更少 token 保留更多有效信息”，但代价是你需要做额外记忆管理。最常见的权衡是摘要频率、冲突处理和检索成本。

| 方案 | 上下文成本 | 长对话稳定性 | 实现复杂度 | 典型问题 |
| --- | --- | --- | --- | --- |
| 固定滑动窗口 | 低到中 | 差 | 低 | 旧关键信息被硬截断 |
| 全量历史拼接 | 很高 | 中 | 低 | 噪声堆积，成本失控 |
| 衰减 + 增量摘要 | 中 | 高 | 中到高 | 阈值和摘要质量要调 |
| 衰减 + 摘要 + 冲突融合 | 中 | 很高 | 高 | 需要额外一致性判断 |

第一个坑是摘要过早。很多系统一看到内容变旧就立刻压缩，结果把后续还要精确引用的细节提前丢掉。比如 API 调用参数、正则表达式、合同条款编号，这些都不适合只留一句“用户提到接口约束”。

第二个坑是摘要过勤。每一轮都重新摘要全部旧历史，会造成“摘要覆盖摘要”，信息越来越平，最后只剩下空泛结论。增量摘要的关键是只处理新增衰减区间，不重写整本账。

第三个坑是未处理冲突。比如第 5 轮说“交付日期是周三”，第 12 轮纠正为“改到周五”。如果系统只是追加新记忆而不降低旧记忆权重，第 20 轮检索时两条都可能被拉出，模型会自相矛盾。正确做法是给旧记忆打上 `deprecated`、`superseded_by` 或冲突标签，让它在排序时明显降权。

第四个坑是把“访问频率”理解成“出现次数”。真正该加权的是“被成功用于任务”的频率，而不是文本里被提到几次。否则高噪声内容会因为重复出现被错误强化。

第五个坑是只看时间不看任务类型。推理链、工具状态、用户偏好、世界事实的生命周期完全不同。工具调用报错通常是短命记忆，用户长期偏好通常是长命记忆，不能共用同一组阈值。

---

## 替代方案与适用边界

不是所有场景都需要遗忘曲线建模。

如果对话很短，例如 5 到 8 轮以内，直接保留全量历史往往更简单，也更稳。因为此时上下文预算还够，额外的摘要与衰减逻辑反而增加系统复杂度。

如果任务高度依赖精确措辞，例如法律条文比对、代码补丁审计、医学记录复核，那么低 $R$ 时也未必能只留摘要。更稳妥的策略是“原句 + 摘要双存”，即摘要用于检索召回，原句用于最终核验。

如果场景存在高频更新和事实冲突，例如项目管理 Agent、交易合规 Agent、故障排查 Agent，那么仅靠时间衰减还不够，需要引入类似 RMM 或 FadeMem 这类更强的机制：前者强调多粒度反思与在线检索修正，后者强调双层记忆、动态衰减与冲突融合。

可以用一个简单阈值表判断保留策略。

| 条件 | 保留形式 | 适用任务 |
| --- | --- | --- |
| $R > 0.7$ | 原文全文 | 当前轮推理、工具调用、精确引用 |
| $0.3 < R \le 0.7$ | 关键句摘要 + 事实槽位 | 中程任务追踪、用户偏好 |
| $R \le 0.3$ | 主题标签 / 结构化摘要 | 长期画像、历史主题 |
| 发生事实冲突 | 新事实原文 + 旧事实失效标记 | 动态知识更新 |
| 高风险领域 | 摘要召回 + 原文校验 | 法务、金融、医疗 |

一个形象但不失真的理解方式是：写长篇小说时，上一章剧情摘要足以帮助你继续写；但人物设定表和核心伏笔不能只剩模糊摘要，因为后文要精确回收。这就是为什么“是否只留摘要”不能只由时间决定，还要由任务风险和信息类型决定。

---

## 参考资料

| 资料 | 贡献 | 应用点 |
| --- | --- | --- |
| Tan et al., *In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents*, ACL 2025, DOI: 10.18653/v1/2025.acl-long.413, https://aclanthology.org/2025.acl-long.413/ | 提出 RMM，用多粒度摘要和回顾式反思改进长期对话检索；在 LongMemEval 上报告超过 10% 的准确率提升 | 支撑“摘要不是一次性压缩，而是可被后续检索反馈修正” |
| Wei et al., *FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory*, arXiv 2601.18642, https://arxiv.org/abs/2601.18642 | 提出双层记忆、动态衰减、冲突融合；报告在多项任务上改进检索与多跳推理，并减少约 45% 存储 | 支撑“遗忘不是删库，而是主动降权和结构化保留” |
| Ebbinghaus forgetting curve / retrievability formulation summary, https://en.wikipedia.org/wiki/Forgetting_curve | 给出 $R=e^{-t/S}$ 这类可检索性建模形式及其解释 | 支撑本文的最小数学模型 |
| 用户给定研究摘要中的社区实验与工程案例链接：https://adg.csdn.net/696f2af7437a6b4033698da1.html 、https://www.showapi.com/news/article/69789dd84ddd79ab670c8e57 、https://www.co-r-e.com/method/agent-memory-forgetting | 提供固定窗口在长对话中失效、工程案例和 FadeMem 的通俗解释 | 可作为工程直觉和案例补充，但应以论文原文为准 |
