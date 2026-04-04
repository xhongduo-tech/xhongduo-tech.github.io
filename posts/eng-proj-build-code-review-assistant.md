## 核心结论

构建代码审查助手，关键不在于“把一个更大的模型塞进 CI”，而在于把审查任务拆成两个问题：先补上下文，再做路由。上下文的意思是，模型不能只看当前 diff，还要看到与这次改动最相似的历史 review、相关 issue、已有代码约束；路由的意思是，不同类型的变更交给不同擅长的模型处理，而不是让一个通用模型包打天下。

已有实验给出了一组很有代表性的结果。在一篇 2026 年发表的研究中，Hybrid Expert System 在 `k=3` 的检索深度下，平均分达到 7.03，较 zero-shot 基线提升 13.2%，较单一最佳模型提升 7.7%；同时，Qwen2.5-Coder-32B 在 `k=3` 时的幻觉率为 7.86%，严重度过估为 11.43%。这说明收益并不只是“更多参数”，而是“更合适的任务分配”。

对初级工程师来说，可以把它理解成一个分工明确的流水线。一个 PR 既新增了“成员注册逻辑”，又更新了 `README`。系统先把 diff、issue、历史 review 取出来，检索最相近的 3 个样本；如果某些片段更像功能改动，就路由给更擅长代码逻辑的模型，例如 Qwen2.5；如果更像文档说明，就路由给更擅长自然语言表达的模型，例如 Mistral。最后再统一给出建议、严重度和理由，交给人工 reviewer 决策。

这类系统成立的原因可以写成一个很朴素的目标函数：我们想最大化建议质量，同时最小化误报和阻塞成本。若把建议质量记为 $Q$，幻觉率记为 $H$，严重度过估记为 $S$，一个简化目标可以写成
$$
\max \; U = Q - \lambda_1 H - \lambda_2 S
$$
其中 $\lambda_1,\lambda_2$ 表示团队对“误报”与“误拦截”的厌恶程度。文档团队通常更怕误拦截，支付系统团队通常更怕漏报，所以同一套模型在不同团队里未必最优。

| 方案 | Functional | Refactoring | Discussion | Documentation | Mean Score |
|---|---:|---:|---:|---:|---:|
| Zero-Shot Baseline | 6.06 | 6.76 | 7.02 | 5.00 | 6.21 |
| Single Best Model | 6.76 | 7.20 | 7.36 | 4.80 | 6.53 |
| Hybrid System | 6.76 | 7.20 | 7.36 | 6.80 | 7.03 |

这张表最值得看的不是均分，而是 `Documentation` 一列。单一最佳模型在总体上更强，但文档类得分反而更低；混合路由后，这一列从 4.80 提到 6.80，才真正说明“分语义路由”有工程价值。

---

## 问题定义与边界

代码审查助手不是“自动找 bug 的聊天机器人”，而是一套面向 Pull Request 的反馈系统。它的输入是代码变更、上下文和历史知识，输出是“是否值得提醒人类”的审查意见。这里的“审查意见”不是判决书，而是带证据、带严重度、可被人类采纳或忽略的候选建议。

这个问题的第一个边界是上下文短缺。所谓幻觉，白话说就是“模型说得像真的，但代码里根本没有这个问题”。在代码审查里，幻觉最常见的原因不是模型完全不会写代码，而是它没有仓库上下文，于是把通用经验误当成当前项目的真实约束。比如项目允许返回 `None` 代表“用户不存在”，模型如果没看到历史实现和约定，就可能错误建议“必须抛异常”。

第二个边界是严重度过估。这个词的白话解释是“把小问题说成大事故”。现代模型很容易把注释、命名、README、格式化这类变更，误判成逻辑缺陷。结果不是建议更多，而是流程更慢，因为每个无关紧要的文档 PR 都被打上“高优先级”标记，审核者被迫二次确认。

可以把两个关键误差定义为：
$$
H = \frac{\#\text{虚假建议}}{\#\text{总建议}}
,\qquad
S = \frac{\#\text{样式/文档变更被误标为功能缺陷}}{\#\text{样式/文档变更}}
$$

玩具例子很简单。一个 PR 只把函数注释从“返回用户”改成“返回已激活用户”，并同步更新 `README`。如果系统只看字面 diff，很可能给出“过滤条件改变可能影响兼容性”的高严重度建议。但如果它先检索到历史 review，发现这是文档同步提交，且相邻代码未改动，就应把该片段归类为 `Documentation`，交给更擅长文本质量判断的模型，减少假阳性。

真实工程例子更典型。假设一个金融 SaaS 系统在 nightly build 前自动审查所有待合并 PR。`payment.py` 中有一处退款状态流转修复，`README.md` 里还有一段对接文档更新。若全部交给同一模型，它可能对逻辑问题过于保守，也可能对文档问题过于苛刻。更稳妥的做法是把它们拆成语义片段分别处理：功能问题优先追求正确性，文档问题优先追求可读性和低误报。

所以，代码审查助手的边界不是“替代 reviewer”，而是“在足够低的幻觉率和足够低的严重度过估下，筛出值得人类看的问题”。一旦超过这个边界，它就会从效率工具变成噪音制造器。

---

## 核心机制与推导

主流可落地方案基本都符合一个两阶段结构。

第一阶段是检索增强。检索增强，白话说就是“先去旧知识里找相似案例，再让模型发言”。系统把 PR diff、相关代码片段或历史 review 编码成向量，存进向量库，例如 Qdrant；新 PR 到来时，把它也编码成向量，然后按余弦相似度检索最接近的历史邻居。余弦相似度衡量两个方向是否接近，公式是
$$
\operatorname{sim}(x,y)=\frac{x\cdot y}{\|x\|\|y\|}
$$
值越接近 1，说明两个改动在语义空间里越相似。

第二阶段是语义路由。语义路由，白话说就是“先判断这像哪一类问题，再送给对应专家”。研究中的做法不是直接用一个大分类器拍板，而是用检索出来的 top-k 邻居做多数投票。如果 3 个邻居里 2 个被标注为 `Documentation`，那当前片段就大概率属于文档类；如果 2 个是 `Functional`，就优先走逻辑专家模型。

这里可以写出一个简化判定：
$$
\hat c = \arg\max_{c \in \mathcal C}\sum_{i=1}^{k}\mathbf{1}[c_i=c]
$$
其中 $\hat c$ 是预测类别，$\mathcal C$ 是类别集合，$c_i$ 是第 $i$ 个邻居的标签。这个规则很朴素，但工程上很稳定，因为它不依赖额外训练一个在线分类器。

还可以定义一个上下文质量分数，帮助系统决定是否继续召回：
$$
C = \alpha \cdot \text{code\_context} + \beta \cdot \text{issue\_context} + \gamma \cdot \text{history\_context}
$$
如果 $C$ 太低，说明检索到的上下文不够像，继续让模型生成往往只会放大幻觉；这时应降级为“只给低置信度提示”或直接不评论。

玩具例子可以这样走。某个 `calc.py` 的 PR 把
```python
if count > 0:
```
改成
```python
if count >= 0:
```
系统检索到 3 条历史样本，其中 2 条都在讨论边界条件和空集合处理，于是把它路由到功能专家模型。模型看到相似 review 后，更容易给出“`count == 0` 是否会导致下游出现空值路径”的具体建议，而不是泛泛地说“请补单元测试”。

真实工程例子更完整。一个包含 `payment.py`、`api/refund.md` 和 issue 链接的 PR 到来后，事件驱动编排器收到 webhook，先抓取 diff 和元数据，再从 Qdrant 取回 top-3 历史片段。与 `payment.py` 相似的邻居多数属于 `Functional`，于是走 Qwen2.5-Coder；与 `refund.md` 相似的邻居多数属于 `Documentation`，于是走 Mistral。Prompt Builder 把“当前 diff + 相似历史 comment + 类别先验”组装成 few-shot prompt，最后输出按片段汇总的 review。这样做比“整个 PR 一把梭”更接近真实团队的认知流程。

这也是为什么 `k=3` 往往比 `k=0` 或 `k=5` 更平衡。`k=0` 没上下文，容易凭空脑补；`k=5` 上下文过多，小模型会出现注意力瓶颈，也就是“信息太多，反而抓不住重点”。研究里 Phi-3-Mini 的幻觉率就从 `k=0` 的 10.66% 升到 `k=5` 的 20.97%，这是一个很典型的信号。

---

## 代码实现

真正可运行的代码审查助手，通常由 6 个部件组成：Webhook 接入、上下文抓取、向量检索、Prompt Builder、专家模型路由、Judge 评分。研究中的实验环境使用了 `Qdrant v1.16.3` 作为向量库，`Ollama v0.12.7` 本地运行量化模型，再用 LLM-as-a-Judge 统一打分，这样可以保证线上和离线评测的可复现性。

下面先给一个最小化的可运行玩具实现。它不依赖真实模型，只展示“检索 + 投票路由 + 严重度控制”的骨架。

```python
from math import sqrt

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    return dot / (na * nb)

def topk_neighbors(query_vec, items, k=3):
    scored = []
    for item in items:
        sim = cosine(query_vec, item["vec"])
        scored.append((sim, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]

def majority_route(neighbors):
    votes = {}
    for n in neighbors:
        votes[n["category"]] = votes.get(n["category"], 0) + 1
    return max(votes.items(), key=lambda x: x[1])[0]

def select_expert(category):
    if category in {"Functional", "Refactoring"}:
        return "qwen2.5-coder"
    if category == "Documentation":
        return "mistral-instruct"
    return "general-reviewer"

def score_severity(category, changed_files):
    if category == "Documentation" and all(f.endswith((".md", ".rst")) for f in changed_files):
        return "low"
    if category == "Functional":
        return "high"
    return "medium"

history = [
    {"id": 1, "vec": [0.9, 0.1, 0.0], "category": "Functional"},
    {"id": 2, "vec": [0.85, 0.15, 0.0], "category": "Functional"},
    {"id": 3, "vec": [0.1, 0.9, 0.0], "category": "Documentation"},
    {"id": 4, "vec": [0.2, 0.8, 0.0], "category": "Documentation"},
]

query_payment = [0.88, 0.12, 0.0]
query_readme = [0.15, 0.85, 0.0]

n1 = topk_neighbors(query_payment, history, k=3)
c1 = majority_route(n1)
e1 = select_expert(c1)
s1 = score_severity(c1, ["payment.py"])

n2 = topk_neighbors(query_readme, history, k=3)
c2 = majority_route(n2)
e2 = select_expert(c2)
s2 = score_severity(c2, ["README.md"])

assert c1 == "Functional"
assert e1 == "qwen2.5-coder"
assert s1 == "high"

assert c2 == "Documentation"
assert e2 == "mistral-instruct"
assert s2 == "low"
```

这段代码刻意省略了 embedding 模型和真实 LLM 调用，但接口已经完整：先找邻居，再投票定类，再选专家，最后按类别约束严重度。你后续把 `vec` 换成真实 embedding、把 `select_expert` 接到 Ollama 或远程 API，就能升级成可部署版本。

生产系统中的 prompt 也不是随便拼字符串。一个靠谱的 Prompt Builder 至少要包含四类信息：当前代码片段、相似历史代码、相似历史 review、当前预测类别。研究附录里的 system prompt 也是这个思路，本质上是 few-shot 模板：让模型看到“类似代码当年是怎么被审的”，从而减少空想式建议。

真实工程里常见的数据流可以写成：

1. GitHub/GitLab webhook 触发。
2. 抽取 diff、文件类型、issue 链接、作者、目录信息。
3. 对片段做 embedding，去 Qdrant 检索 top-k。
4. 用邻居元数据做类别投票。
5. 依据类别路由到不同模型。
6. 产出 review comment、类别、严重度、证据。
7. 再用 Judge 模型或规则系统打分，低于阈值不发给人类。

Judge 的作用不是“再生成一次答案”，而是“用统一 rubric 评价候选建议值不值得发”。研究里给出的打分规则也很直白：10 分表示准确指出问题并给出正确修复，8 到 9 分表示技术上正确但表达略有问题，5 到 7 分表示只抓到主题但方案不完整，1 到 4 分表示偏离核心或出现幻觉，0 分表示事实错误或空输出。把这个 Judge 放在发评论前，能明显减少线上噪音。

---

## 工程权衡与常见坑

第一类坑是检索过深。很多人会自然地觉得“上下文越多越好”，但事实并不是这样。对小模型尤其如此，更多检索片段会带来注意力拥塞，让模型把相似但不相关的信息混进当前判断。研究里 Phi-3-Mini 在 `k=0` 时幻觉率为 10.66%，到了 `k=5` 升到 20.97%；这说明不是“召回更多”就一定“理解更准”。

第二类坑是把文档问题当功能故障。Mistral-Instruct-7B 在该研究中严重度过估长期高于 36%，`k=3` 时达到 43.81%。这不是说模型“差”，而是说它更擅长流畅表达，不等于更擅长区分故障级别。若你的系统没有单独的严重度约束层，文档 PR 会被大量无意义高优先级提醒淹没。

第三类坑是用单一阈值处理所有类别。功能类建议错一条，可能是线上事故；文档类建议错一条，通常只是 reviewer 多看一眼。所以发评论阈值不应统一。更合理的做法是按类别设不同阈值，例如 `Functional >= 8` 才发，`Documentation >= 6` 才发，并对文档类默认降低严重度上限。

第四类坑是只做生成，不做证据绑定。没有证据链的 review comment 很难被信任。系统至少要把“这条建议参考了哪几个历史片段、针对哪一行 diff、为什么归到该类别”一起记录下来，否则线上误报无法回溯。

第五类坑是数据不可复现。代码审查与 reviewer recommendation 领域长期都有这个问题：很多研究使用闭源 review 数据，训练集构成、source code availability、验证设置都不透明。部署时最危险的后果不是精度下降，而是行为漂移。你以为模型学到的是“你的团队如何审查代码”，实际学到的可能是“另一个组织的评论口味”。

第六类坑是把助手当自动合并器。代码审查助手适合做“排序、筛选、补充证据”，不适合单独决定 merge。尤其在支付、权限、数据删除、并发一致性这类高风险模块，它最多做第一道预审，不应成为最终裁决者。

---

## 替代方案与适用边界

如果你的目标不是“更准地给审查意见”，而是“生成多个可选修复方案”，那多专家检索路由并不是唯一选择。SMILER 代表的是另一条路线：用 CVAE，也就是条件变分自编码器，把随机噪声注入 prior 和 posterior 网络，让系统生成更有多样性的候选代码。白话说，它不是只给一个答案，而是努力给出几种不同修法。

这类方法适合开源社区或方案空间很大的场景。比如 reviewer 说“这个判空逻辑不够稳健”，SMILER 风格的系统可能同时给出“提前返回”“引入守卫子句”“拆分辅助函数”三种修改候选。它解决的是“建议太单一”，而不是“建议是否分类准确”。所以如果你团队更关心“给开发者几个可执行备选方案”，SMILER 方向有价值；如果更关心“不要把 README 改动误报成严重 bug”，那检索加路由更直接。

另一类替代方案是 reviewer recommendation，也就是先推荐“谁来审”，而不是先生成“审什么”。这条路线的核心是把 PR 路由给最合适的人类 reviewer。它适用于组织内部 reviewer 画像稳定、专家领域划分明确的团队。它和代码审查助手不是竞争关系，而是可叠加关系：先决定谁最适合看，再让助手给这个 reviewer 准备上下文和候选问题。

可以这样理解三类方案的边界：

| 方案 | 主要解决的问题 | 适合场景 | 主要风险 |
|---|---|---|---|
| 检索增强 + 多模型路由 | 提高建议准确率，降低幻觉与误拦截 | CI/CD 自动预审、混合代码与文档 PR | 检索噪音、路由错误 |
| SMILER / CVAE 多样化生成 | 提供多个可能修复方案 | 开源社区、方案空间大、强调候选多样性 | 候选多但未必更准 |
| Reviewer Recommendation | 找合适的人审查 | 组织内专家边界清晰、审查责任明确 | 数据不可复现、推荐偏置 |

对初级工程师最重要的判断标准只有一句：如果你的主要痛点是“自动建议经常胡说八道”，优先做检索和路由；如果痛点是“建议太单一，开发者看不到替代修法”，再考虑 SMILER 这类多样化生成；如果痛点是“PR 太多，不知道该找谁审”，优先做 reviewer recommendation。

---

## 参考资料

- Context-Aware Code Review Automation: A Retrieval-Augmented Approach. MDPI Applied Sciences, 2026. 重点参考了 Table 8、Table 9、Figure 1、Appendix A。https://www.mdpi.com/2076-3417/16/4/1875
- Structuring Meaningful Code Review Automation in Developer Community. Engineering Applications of Artificial Intelligence, 2024. 重点参考了 SMILER 的 CVAE 结构与多样性目标。https://www.sciencedirect.com/science/article/pii/S0952197623011545
- A review of code reviewer recommendation studies: Challenges and future directions. Science of Computer Programming, 2021. 重点参考了可复现性、数据集与验证设置问题。https://www.sciencedirect.com/science/article/pii/S0167642321000459
