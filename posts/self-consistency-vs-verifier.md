## 核心结论

Self-Consistency，简称 SC，可以理解为“让模型独立做多次题，再看哪个答案出现最多”。它的核心优势是不需要额外训练，直接利用已有大模型就能提升推理稳定性。对应公式是：

$$
a^*=\arg\max_a\sum_{i=1}^k \mathbb{1}[a_i=a]
$$

其中 $a_i$ 是第 $i$ 条推理链给出的最终答案，$\mathbb{1}[\cdot]$ 是指示函数，条件成立记为 1，否则记为 0。

Process Reward Model，简称 PRM，可以理解为“不给整条答案只打总分，而是给每一步推理单独打分”。它的优势不是采样多，而是能检查过程是否可靠。对一条候选推理链，常见的总分写法是：

$$
s(a)=\sum_t r_t
$$

其中 $r_t$ 表示第 $t$ 步的奖励，含义是“这一步是否让解题过程更接近正确结论”。

两者不是替代关系，而是互补关系。SC 擅长从多个候选中找“高频答案”，PRM 擅长识别“看起来像对、但中间推歪了”的过程。一个常见融合公式是：

$$
a^*=\arg\max_{a\in \text{SC-set}}\left(\frac{f(a)}{k}+\lambda s(a)\right)
$$

这里 $f(a)$ 是答案 $a$ 在 SC 采样中出现的次数，$\text{SC-set}$ 是 SC 收集到的候选答案集合，$\lambda$ 是权重，用来平衡“多数票强度”和“过程可靠性”。

对初级工程师最重要的结论只有一句：如果你没有额外标注和训练资源，先用 SC；如果你已经有验证器或 PRM，最有效的做法通常不是替换 SC，而是在 SC 之后做重排序。

---

## 问题定义与边界

本文讨论的是大模型在 Chain-of-Thought，简称 CoT，也就是“把中间推理步骤显式写出来”的场景下，如何从多条候选推理链中选最终答案。

问题不是“模型能不能生成答案”，而是“生成了多条答案后，该信谁”。

先给出三个术语的工作定义：

| 术语 | 白话解释 | 输入 | 输出 | 训练需求 |
|---|---|---|---|---|
| SC | 谁出现次数最多就优先信谁 | 多条完整推理链 | 一个最终答案 | 不需要 |
| PRM | 给每一步推理打分的验证器 | 一条推理链的步骤序列 | 过程总分 | 需要 |
| SC-set | SC 采样后得到的候选答案集合 | 多条推理结果 | 若干候选答案 | 不涉及 |
| $\lambda$ | 控制投票和打分谁更重要 | 频次与分数 | 融合权重 | 需要调参 |

边界也要说清楚。

第一，SC 的前提是你愿意付出多次采样成本。若只生成 1 条链，SC 不成立。它依赖“多样但相互独立”的候选路径。

第二，PRM 的前提是你有可用的过程监督。过程监督指“知道某一步是进步还是退步”的训练信号。没有这类数据时，PRM 很难可靠。

第三，SC 和 PRM 主要适用于答案可比对、推理步骤可拆分的任务，例如数学题、符号推理、多步问答、代码修复。对于开放式写作、创意生成、主观偏好任务，二者都没有同样稳定的意义。

可以把二者的典型弱点直接放在一个表里：

| 方法 | 主要优点 | 主要成本 | 典型弱点 |
|---|---|---|---|
| SC | 零训练、实现简单 | 推理成本随 $k$ 线性增长 | 多数答案可能一致错 |
| PRM | 能识别局部推理错误 | 训练和标注成本高 | 可能误杀正确但表达不同的链 |
| SC + PRM | 兼顾频次与过程 | 同时有采样和打分成本 | 权重不当会过度偏向某一侧 |

新手版可以这样理解：SC 是“先让模型做 5 次，谁答得最多就先占优”；PRM 是“把每次解题过程拆开检查”；融合就是“先收集候选，再比质量”。

---

## 核心机制与推导

SC 的逻辑并不神秘。它假设一件事：如果模型内部已经具备某个正确推理模式，那么在合理采样下，这个模式更容易重复出现，因此正确答案的频次通常更高。这里的“通常”很重要，它不是数学保证，而是经验统计规律。

玩具例子先看第一组：

$$
\{12,12,12,14,8\}
$$

若 $k=5$，那么频次函数为：

- $f(12)=3$
- $f(14)=1$
- $f(8)=1$

只做 SC 时，最终答案显然是 12。

再看 PRM。假设三条得到 12 的链，其过程分数分别为 0.86、0.83、0.80；得到 14 和 8 的两条链分数为 0.30、0.22。若按答案聚合平均分，则：

- $s(12)\approx 0.83$
- $s(14)=0.30$
- $s(8)=0.22$

这时融合后仍然是 12，因为它既高频，又高分。

第二组更能体现融合价值：

$$
\{12,12,14,14,14\}
$$

只做 SC，答案会选 14，因为它出现了 3 次。

但如果 PRM 检查发现，三条输出 14 的链都在某一步使用了错误公式，分数只有 0.25、0.28、0.31；反而两条输出 12 的链虽然少，但过程完整，分数分别是 0.88、0.84。那么答案级别的平均过程分数可以写成：

- $s(12)=0.86$
- $s(14)=0.28$

此时融合分数变成：

$$
\text{score}(a)=\frac{f(a)}{k}+\lambda s(a)
$$

代入 $k=5$：

- $\text{score}(12)=0.4+\lambda \cdot 0.86$
- $\text{score}(14)=0.6+\lambda \cdot 0.28$

若 $\lambda=0.5$，则：

- $\text{score}(12)=0.83$
- $\text{score}(14)=0.74$

最终答案从 14 变成 12。这个例子说明，PRM 的作用不是“推翻多数票”，而是在多数票不够可靠时提供纠错信号。

这也是公式里 $\lambda$ 的意义。它控制两个量纲不同的信号如何合并。

- 当 $\lambda \to 0$ 时，融合退化为纯 SC，系统只看频次。
- 当 $\lambda \to \infty$ 时，系统几乎只看 PRM 分数，多数票被忽略。
- 实际工程里通常选择一个较小但非零的 $\lambda$，让频次仍是主信号，过程分数只负责“修边”。

可以把流程压缩成一个文字流程图：

| 步骤 | 动作 | 目的 |
|---|---|---|
| 1 | 对同一问题采样 $k$ 条 CoT | 获得候选解空间 |
| 2 | 统计每个答案的频次 | 得到 SC-set 和多数票强度 |
| 3 | 用 PRM 给候选链逐步打分 | 识别局部错误和跳步 |
| 4 | 聚合到答案级别并重排序 | 输出最终答案 |

参数灵敏度也可以直观理解：

| $\lambda$ 区间 | 系统行为 |
|---|---|
| 很小 | 基本等于 SC，纠错能力弱 |
| 适中 | 既保留多数票，又能修正低质量高频答案 |
| 很大 | 变成“验证器主导”，容易忽略群体共识 |

因此，融合并不是“把两个模型简单加起来”，而是把“统计稳定性”和“过程可靠性”放到同一决策函数中。

---

## 代码实现

工程上建议拆成三个模块：`SC sampler` 负责采样，`PRM scorer` 负责打分，`reranker` 负责融合排序。这样做的原因很简单，后续你可以独立替换采样模型、验证器和融合函数。

下面给一个可运行的 Python 玩具实现，代码不依赖外部库：

```python
from collections import Counter, defaultdict

def vote(chains):
    # chains: [{"answer": int, "steps": [...], "prm_score": float}, ...]
    freq = Counter(chain["answer"] for chain in chains)
    best_answer, best_count = max(freq.items(), key=lambda x: x[1])
    return best_answer, best_count, freq

def aggregate_prm_by_answer(chains):
    grouped = defaultdict(list)
    for chain in chains:
        grouped[chain["answer"]].append(chain["prm_score"])
    # 这里用平均分，也可以改成最大值、中位数或 top-m 平均
    return {ans: sum(scores) / len(scores) for ans, scores in grouped.items()}

def rank_candidates(chains, lambda_weight=0.5):
    _, _, freq = vote(chains)
    prm_scores = aggregate_prm_by_answer(chains)
    k = len(chains)

    ranked = []
    for answer in freq:
        score = freq[answer] / k + lambda_weight * prm_scores[answer]
        ranked.append((answer, score, freq[answer], prm_scores[answer]))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# 玩具例子 1: 多数票和 PRM 一致支持 12
chains1 = [
    {"answer": 12, "steps": ["a", "b"], "prm_score": 0.86},
    {"answer": 12, "steps": ["a", "c"], "prm_score": 0.83},
    {"answer": 12, "steps": ["a", "d"], "prm_score": 0.80},
    {"answer": 14, "steps": ["x", "y"], "prm_score": 0.30},
    {"answer": 8,  "steps": ["m", "n"], "prm_score": 0.22},
]

ranked1 = rank_candidates(chains1, lambda_weight=0.5)
assert ranked1[0][0] == 12

# 玩具例子 2: 多数票支持 14，但 PRM 把 12 拉回第一
chains2 = [
    {"answer": 12, "steps": ["good1"], "prm_score": 0.88},
    {"answer": 12, "steps": ["good2"], "prm_score": 0.84},
    {"answer": 14, "steps": ["bad1"], "prm_score": 0.25},
    {"answer": 14, "steps": ["bad2"], "prm_score": 0.28},
    {"answer": 14, "steps": ["bad3"], "prm_score": 0.31},
]

ranked2 = rank_candidates(chains2, lambda_weight=0.5)
assert vote(chains2)[0] == 14
assert ranked2[0][0] == 12

print("All tests passed.")
```

如果把它扩展成真实系统，接口通常长这样：

```python
def sample_chains(question: str, k: int) -> list[dict]:
    """调用生成模型，返回 k 条候选推理链。"""

def score_with_prm(chains: list[dict]) -> list[dict]:
    """给每条链的每一步打分，并聚合成 prm_score。"""

def rank_candidates(chains: list[dict], lambda_weight: float) -> dict:
    """先统计频次，再结合 PRM 分数重排，输出最终答案。"""
```

真实工程例子可以参考 WizardMath 在 GSM8K 上的做法：先用 SFT 模型生成大量候选解，再做 SC 汇总频次，最后用 PRM 重排序。这个流程的关键不是“PRM 单独很强”，而是“PRM 在一个已经不错的候选集上继续清理局部错误”。

实现时有两个直接的性能建议：

| 优化点 | 做法 | 原因 |
|---|---|---|
| 批量 PRM 打分 | 一次送入多条候选链 | 减少验证器调用开销 |
| 缓存中间结果 | 缓存 token 化结果和重复候选 | 避免重复计算 |
| 先缩小候选集 | 只对高频答案对应的链打分 | 降低 PRM 成本 |

---

## 工程权衡与常见坑

第一类问题是成本。SC 的成本近似随采样次数 $k$ 线性增长。你把 $k$ 从 5 提到 50，通常延迟和费用也接近放大 10 倍。PRM 则是另一种成本，它不一定贵在推理，而是贵在训练数据。步级标注很难做，质量不稳定时，PRM 容易学到表面模式。

第二类问题是聚合策略。PRM 本来给的是“链分数”，最终你要的是“答案分数”。这中间要选聚合方法：

- 对同一答案的多条链取平均
- 取最高分
- 取 top-m 平均
- 同时考虑频次和方差

若处理不好，会出现一个常见坑：某个答案只有一条特别高分的链，却压过了另一个出现很多次、整体还不错的答案。对于初学者，默认用“频次 + 平均 PRM 分”通常最稳。

第三类问题是 tie-break，也就是平票处理。比如两个答案频次相同，这时必须明确规则，否则输出不稳定。常见规则有两种：

| 场景 | 建议 |
|---|---|
| 频次相同 | 看 PRM 平均分 |
| 频次和平均分都接近 | 优先选择步骤更短、格式更规范的链 |

第四类问题是 $\lambda$ 调得过大。若验证器有系统偏差，它会把“表达不标准但推理正确”的链压下去，出现 False Negative，也就是“把好样本误判成坏样本”。这在数学推理和代码推理中都很常见，因为正确过程可能有多种写法。

真实工程里，一个较稳的流程通常是：

1. 用小到中等的 $k$，例如 8、16、32，快速得到 SC-set。
2. 只对高频候选相关的链跑 PRM，而不是全量跑。
3. 用较小的 $\lambda$ 做重排序，先让 PRM 扮演“纠错器”，不要一开始就让它当“裁判长”。
4. 对显著低分链做过滤，但保留少量边界样本，避免误删。

WizardMath 的公开结果很有代表性。其在 GSM8K 的流程可以概括为：SFT 生成器先产出大量候选，再通过 SC 汇总，最后用 PRM reranker 重排，准确率从只用 SC 时的 92.3% 提升到 95.2%。这个例子说明，验证器最有价值的阶段往往不是“从零决定答案”，而是“在强候选集上做最后筛选”。

---

## 替代方案与适用边界

不是所有团队都需要训练 PRM。很多时候，预算、数据和上线周期决定了方法选择。

可以把常见方案放在一张表里：

| 方法 | 训练需求 | 推理成本 | 最适合 |
|---|---|---|---|
| 只做 SC | 无 | 中到高 | 没有标注资源、想快速提升 |
| SC + 简单 verifier | 低到中 | 高 | 有少量验证数据、想做轻量重排 |
| SC + PRM | 高 | 高 | 高准确率任务，且有过程监督 |
| Diverse prompts + vote | 无 | 中到高 | 想提升候选多样性 |
| Outcome Reward Model, ORM | 中 | 中 | 只关心最终答案对错，不关心细步骤 |

这里顺带解释一下 ORM。Outcome Reward Model，简称 ORM，可以理解为“只给最终结果打分的奖励模型”。它比 PRM 便宜，因为不需要步级监督，但缺点也明显：它不知道错误发生在哪一步，只知道最后像不像对。

因此，SC、ORM、PRM 的关系可以这样看：

- SC 负责“广撒网”，靠统计共识找候选。
- ORM 负责“看结果像不像好答案”。
- PRM 负责“看过程是不是站得住”。

对零基础到初级工程师，最实用的判断标准是：

| 团队条件 | 更合适的方案 |
|---|---|
| 没数据、想快上线 | SC-only |
| 有少量判分数据 | SC + 轻量 verifier 或 ORM |
| 有过程标签、追求最高准确率 | SC + PRM |
| 任务不是严格推理题 | 不一定要用 PRM，可能改用事实核查器更合适 |

还要强调一个边界：如果任务本身没有稳定、可分解的中间步骤，例如文案生成、开放问答、风格改写，那么 PRM 的“每步奖励”很难定义，这时强行上 PRM 往往收益有限。反过来，在数学题、代码补全修复、复杂工具调用决策中，PRM 才更容易发挥作用。

---

## 参考资料

1. Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023.  
结论：提出 Self-Consistency，用多条 CoT 路径的多数投票替代贪心单路径，在 GSM8K 等基准上带来显著提升，文中报告 GSM8K 可提升 17.9%。

2. Xie et al. *From Outcomes to Processes: Guiding Language Models to Use Grounded Reasoning*. ACL 2025.  
结论：系统比较结果级奖励与过程级奖励，说明 PRM 在细粒度推理判断上更强，但训练依赖更重，也强调偏好一致性与监督粒度的重要性。

3. Luo et al. *WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct*. ICLR 2025.  
结论：展示了生成模型、SC 和 PRM reranker 的组合流程。在 GSM8K 上，SC 基线约为 92.3%，加入 PRM 重排序后可到 95.2%，说明“候选集 + 验证器”有稳定协同收益。

4. 关键词建议  
可继续检索：`Self-Consistency`, `Process Reward Model`, `Outcome Reward Model`, `GSM8K`, `Verifier reranking`, `WizardMath PRM`.
