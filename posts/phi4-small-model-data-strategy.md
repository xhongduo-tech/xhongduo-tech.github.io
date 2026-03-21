## 核心结论

Phi-4 的核心判断很直接：对 14B 这一级别的小模型，决定上限的关键变量不只是参数量，而是训练数据是否足够“像教材”。这里的“教材式数据”是指结构清楚、步骤完整、答案可验证的数据，不是随手抓来的网页文本。Phi-4 把这个思路推到极致，用约 40% 合成教材式数据、30% 网页与网页重写、20% 代码、10% 定向采购学术与图书数据来训练 14B 模型，并在数学、科学问答、代码上做了强针对性优化。

这套策略最重要的证据，不是“整体感觉更聪明”，而是它在 STEM 基准上出现了明显越级。公开材料里，Phi-4 在 SimpleEval 的 GPQA 上为 56.1%，高于 GPT-4o-mini 的 40.9%；微软社区文章给出的 MATH 对比是 56.1% 对 52.2%，而 Hugging Face 模型卡里还能看到 Phi-4 在 MATH 行达到 80.4%。不同公开页面的表格排版和口径存在差异，但方向一致：Phi-4 在推理型任务上的表现，明显超出“小模型应该有的水平”。

可以把它理解成一个很朴素的教学实验。不是让学生盲刷海量网页，而是先给他“高质量题目 + 连贯解题过程 + 老师批改后的版本”，再专门训练他在关键步骤上做对选择。Phi-4 本质上就是把这种教学流程工程化了。

| 指标 | Phi-4 | GPT-4o-mini | 观察 |
| --- | ---: | ---: | --- |
| GPQA | 56.1 | 40.9 | 研究生级科学问答上优势明显 |
| MATH | 80.4 或 56.1 | 公开资料有 52.2、73.0 等不同口径 | 结论一致：Phi-4 在数学推理上更强 |
| HumanEval | 82.6 | 86.2 | 代码生成强，但不是所有项目都绝对领先 |
| SimpleQA | 3.0 | 9.9 | 事实型问答不是它的强项 |

上表要读懂一点：Phi-4 不是“全面碾压”，而是“把训练预算集中投到推理密集任务后，在目标任务上打出高分”。这正是它值得分析的地方。

---

## 问题定义与边界

Phi-4 解决的问题不是“做一个最全能的小模型”，而是“在可部署、可量化、低延迟的约束下，把 STEM 推理能力推到尽可能高”。这里的“可部署”很具体：14B 参数、16K 上下文、支持量化后在消费级显卡上运行，微软社区与第三方量化实践都强调了它面向单卡和低资源环境的取向。

因此，Phi-4 的目标函数不是通用聊天分数最大化，而更像下面这个形式：

$$
D = 0.4D_{syn} + 0.3D_{web} + 0.2D_{code} + 0.1D_{tar}
$$

其中：

- $D_{syn}$ 是合成教材式数据，也就是由教师模型和多轮工作流生成的“讲清楚过程”的样本。
- $D_{web}$ 是精选网页与网页重写数据，也就是保留事实覆盖并把散乱文本改写成更适合学习的形式。
- $D_{code}$ 是代码数据，用来补足程序执行、API 结构、调试模式。
- $D_{tar}$ 是定向采购数据，如学术资料、图书、问答库。

这个边界很重要。若你只看“合成数据很强”，很容易误读成“那就尽量全部合成”。Phi-4 的实验恰好说明相反：合成数据对数学、代码、理解任务很有效，但如果合成比例过高、网页与真实知识数据不足，事实问答会明显掉线。Graphcore 对技术报告的拆解里提到，偏重合成数据的模型在知识型测试上会有下降，而加入 web rewrites 后下降幅度明显收敛。

玩具例子可以这样理解。你要教一个初学者学一元二次方程：

| 训练方式 | 数据样子 | 学到的能力 |
| --- | --- | --- |
| 直接抓网页 | 定义、论坛争论、错误答案混在一起 | 知道很多词，但步骤不稳定 |
| 教材式合成 | 定义、例题、逐步推导、错因纠正 | 更容易学会“先展开再判别” |
| 纯教材式合成 | 解题过程很整齐，但世界知识少 | 做题强，常识问答弱 |

所以 Phi-4 的边界不是“合成数据万能”，而是“合成数据特别适合教推理，但必须被真实世界数据补平”。

---

## 核心机制与推导

Phi-4 的训练流程可以分成三层：预训练、中训练、后训练。

第一层是预训练。技术报告与模型卡显示，Phi-4 使用了约 9.8T token 的总体训练数据，其中核心创新不是模型结构大改，而是数据课表重写。Graphcore 对报告的总结提到，研究团队先构建了约 400B token 的高质量合成数据簇，再和筛选后的网页、代码、图书数据混合。这里的“知识蒸馏”不是只看最后答案，而是让教师模型把“如何得到答案”的路径写进样本里。白话说，蒸馏就是把大模型会做题的方式，压缩成小模型可学习的数据。

第二层是中训练。Phi-4 先在 4K 上下文训练，再把上下文扩到 16K。这个阶段的意义不是单纯“塞更多字”，而是让模型学会在更长的推理链里保留状态。长上下文如果只靠把短样本硬拼接，收益有限；Phi-4 会额外上调原本就足够长、结构也足够清楚的数据权重，比如长教材片段、长代码文件、长学术文本。

第三层是后训练。这里是 Phi-4 最有技术辨识度的地方：先 SFT，再两轮 DPO。SFT 是监督微调，白话说就是拿高质量问答直接教模型模仿；DPO 是直接偏好优化，白话说就是告诉模型“同一题这版更好，那版更差”，让模型把概率往更好的答案分布上推。

Phi-4 在第一轮 DPO 里引入了 Pivotal Token Search，简称 PTS。它的思想是：一道题最后做对，往往不是整段话每个 token 都同等重要，而是中途有少数几个“关键 token”改变了后续路径。于是可以定义：

$$
p_i = p(\text{success}\mid t_{\le i})
$$

$$
\Delta_i = p_i - p_{i-1}
$$

如果某个位置的 $\Delta_i$ 很大，说明这个 token 对最终成败影响大，它就是“关键 token”。比如一道代数题里，“先交叉相乘”与“先分别乘两边分母”都可能通向正确答案，但其中一个分支对当前小模型更稳。PTS 就是把这类分叉点找出来，再用 accept/reject 对构造偏好数据。

技术报告的一个细节也很重要：为了提升样本效率，PTS 重点筛选 $0.2 \le p(\text{success}) \le 0.8$ 的题。太简单的题没有学习价值，太难的题关键 token 也稀少。Graphcore 的总结还指出，PTS 这轮 DPO 把 SimpleQA 上的幻觉率从 38.7% 降到 17.4%。这不是让模型“知道更多事实”，而是让它更会在不确定时收手。

真实工程例子是本地 STEM 辅导器。你把教材章节、学生问题、几段 Python 演示代码一起放进 16K 上下文，让模型解释“为什么这里要先设未知数，再列方程”。这类任务需要的是过程稳定、步骤清楚、关键节点不乱跳。Phi-4 的训练方式，正好就是围绕这种能力设计的。

---

## 代码实现

下面给一个可运行的玩具实现，用来模拟 PTS 的核心流程。它不依赖大模型，只用一组手工设定的“前缀成功率”来演示如何找关键 token，并生成 DPO 训练对。

```python
from dataclasses import dataclass

@dataclass
class Pair:
    prefix: str
    accept: str
    reject: str
    gain: float

def pivotal_pairs(tokens, success_probs, low=0.2, high=0.8, min_gain=0.15):
    assert len(tokens) == len(success_probs)
    assert all(0.0 <= p <= 1.0 for p in success_probs)

    pairs = []
    prefix_tokens = []
    prev = 0.0

    for tok, p in zip(tokens, success_probs):
        prefix = " ".join(prefix_tokens)
        gain = p - prev

        # 只保留“难度适中”且概率跳变明显的位置
        if low <= p <= high and gain >= min_gain:
            pairs.append(
                Pair(
                    prefix=prefix,
                    accept=tok,
                    reject=f"ALT_{tok}",
                    gain=round(gain, 3),
                )
            )

        prefix_tokens.append(tok)
        prev = p

    return pairs

tokens = ["设", "x", "为", "未知数", "先交叉相乘", "整理", "求根"]
success_probs = [0.05, 0.08, 0.10, 0.18, 0.52, 0.71, 0.93]

pairs = pivotal_pairs(tokens, success_probs)

assert len(pairs) == 2
assert pairs[0].accept == "先交叉相乘"
assert pairs[1].accept == "整理"
assert pairs[0].gain == 0.34
print(pairs)
```

这个例子表达的不是“真实概率怎么估”，而是数据生成逻辑：

1. 对同一道题 rollout 多次，拿到不同解题路径。
2. 对每个前缀估计成功率 $p(\text{success} \mid prefix)$。
3. 找到成功率突增的位置，视为 pivotal token。
4. 把“更稳的续写”和“更差的续写”打包成偏好对，喂给 DPO。

如果把它映射回 Phi-4，可得到一个简化训练框架：

```python
for question in seed_questions:
    rollouts = generate_many_answers(question, n=8)
    for prefix in collect_prefixes(rollouts):
        p = estimate_success(prefix)
        if 0.2 <= p <= 0.8:
            accept, reject = branch_at_pivotal_token(prefix)
            save_dpo_pair(prefix, accept, reject)
```

工程里真正难的部分不在这 6 行，而在三个地方：一是成功率估计器怎么做，二是 accept/reject 如何保证只在关键点附近分叉，三是如何把教师模型偏差控制住。Phi-4 的答案是多 seed、self-revision、可执行验证和 judge-guided DPO 组合使用，而不是只信任单次教师输出。

---

## 工程权衡与常见坑

Phi-4 的方法强，但代价也很清楚：你把训练预算压到“推理路径质量”上，别的地方就要补课。

| 常见坑 | 现象 | 规避方式 |
| --- | --- | --- |
| 纯合成比例过高 | 数学变强，事实问答掉分 | 保留网页、代码、采购知识数据 |
| 教师偏差被复制 | 小模型学到教师的固定套路 | 多 seed 生成 + self-revision |
| 只看最终答案 | 模型会蒙对，不会稳定推导 | 用 PTS 标出关键步骤 |
| 长上下文只靠拼接 | 16K 能放内容，但不会用 | 提高原生长样本权重 |
| 过度追求拒答 | 幻觉少了，但有用回答也减少 | 在安全、拒答、完成度之间重新配比 |

最容易被新手忽略的是第一条。Graphcore 对报告的总结已经很明确：偏重合成训练会拉高 reasoning-heavy benchmark，但 general knowledge 会受损。Phi-4 在 SimpleQA 上并不亮眼，本身就说明它不是靠“知道更多世界事实”赢的，而是靠“在推理题上更会走对路径”赢的。

另一个坑是表格数字的误读。Phi-4 的公开资料分散在技术报告、Hugging Face 模型卡、社区博客和第三方解读里，部分页面的基准表格存在排版或口径差异。工程上应把趋势当结论，把单点数值当版本化指标，而不是拿某一个截图做绝对判断。

部署层面也有现实权衡。4-bit 量化版本可以把显存压到约 11GB 量级，但量化后吞吐、长上下文性能、注意力实现方式都会影响实际体验。换句话说，“能跑起来”和“跑得像论文里一样稳”不是一回事。

---

## 替代方案与适用边界

如果把小模型训练路线粗分，Phi-4 属于“高质量数据驱动的推理强化型”，Gemma 2 更接近“蒸馏驱动的通用能力复制型”。这里的“蒸馏驱动”是指用更大教师模型的输出概率分布或行为分布去训练更小学生模型。InfoQ 对 Gemma 2 的报道提到，Google 发布的是 2B、9B、27B 三个官方尺寸，其中 2B 和 9B 采用了知识蒸馏。常见二次转述里把 2B 写成 2.6B，这个说法不准确。

| 维度 | Phi-4 | Gemma 2 |
| --- | --- | --- |
| 官方主尺寸 | 14B | 2B / 9B / 27B |
| 主要策略 | 教材式合成数据 + SFT + 两轮 DPO | 知识蒸馏 + 架构优化 |
| 训练重点 | 数学、代码、科学问答、推理路径 | 通用语言能力与参数效率 |
| 长上下文策略 | 中训练从 4K 扩到 16K | 架构层面加入 GQA、局部/全局注意力组合 |
| 适合场景 | 本地 STEM tutor、代码助手、低延迟 reasoning | 通用聊天、泛知识、轻量部署 |

所以，什么时候选 Phi-4，什么时候不选？

选 Phi-4 的边界很明确：你关心的是“解释步骤是否稳定”“数学和代码是否靠谱”“单机部署是否便宜”。不选它的场景也同样明确：如果你更在意大范围事实覆盖、多轮社交式对话、跨语言通用表现，Phi-4 并不是天然最优。它的设计不是为了当“最均衡的小模型”，而是为了当“推理密集任务上的高性价比小模型”。

一句话概括新手能听懂的版本：Phi-4 更像“教科书训练出来的理科生”，Gemma 2 更像“把大模型说话方式压缩给小模型的通才”。前者做题更稳，后者分布更广。

---

## 参考资料

- Microsoft Phi-4 Hugging Face 模型卡：https://huggingface.co/microsoft/phi-4/blob/main/README.md
- Microsoft Phi-4 模型卡 PR 版本：https://huggingface.co/microsoft/phi-4/blob/refs%2Fpr%2F53/README.md
- Phi-4 技术报告解读，Graphcore Research Blog：https://graphcore-research.github.io/phi-4/
- Phi-4 技术报告综述，AZoAI：https://www.azoai.com/news/20250105/Phi-4-Microsoft-Researchs-14B-Parameter-Model-Advances-STEM-Reasoning.aspx
- Phi-4 论文索引页：https://huggingface.co/papers/2412.08905
- Microsoft Tech Community 对 Phi-4 的介绍：https://techcommunity.microsoft.com/blog/educatordeveloperblog/phi-4-small-language-models-that-pack-a-punch/4464167
- Phi-4 量化与推理速度实践：https://techcommunity.microsoft.com/blog/machinelearningblog/phi-4-quantization-and-inference-speedup/4360047
- Gemma 2 新闻概览，InfoQ：https://www.infoq.com/news/2024/07/google-gemma-2/
