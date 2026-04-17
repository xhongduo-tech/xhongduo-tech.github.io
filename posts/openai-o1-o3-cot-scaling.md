## 核心结论

o1/o3 的关键变化，不是先把参数继续做大，而是把 **Compute Scaling** 引入推理系统。这里的 Compute Scaling，白话说，就是“让模型在训练时学会更会想，在回答时也允许它多想一会儿”。因此，它的能力提升来自两端：训练阶段更多强化推理行为，推理阶段更多分配思考预算。

这和传统 **SFT 模型**（监督微调模型，白话说，就是主要靠“看标准答案模仿输出”的模型）有本质差异。SFT 更像“直接背会常见题型后尽快作答”；o1/o3 更像“先拆题、试错、修正，再给最终答案”。OpenAI 在官方介绍里明确强调，o1 的表现会随着 **train-time compute** 和 **test-time compute** 同时增长，这说明它不是单纯靠更大的预训练规模，而是在推理阶段也存在可扩展性。

下面这张表先抓住主线：

| 维度 | 传统 SFT 模型 | o1/o3 推理模型 |
|---|---|---|
| 训练目标 | 学会直接生成答案 | 学会产生更有效的推理过程并输出答案 |
| 主要扩展轴 | 参数量、数据量 | 参数量之外，再加入训练算力和推理算力 |
| 推理时行为 | 倾向尽快回答 | 倾向先消耗预算做中间推理 |
| 成本结构 | 单次调用较稳定 | 单次调用成本可随 effort 明显上升 |
| 适合任务 | 知识问答、改写、摘要 | 数学、代码、规划、复杂多步推理 |

对初学者，一个最直观的理解是：把 o1/o3 看成“允许先打草稿再交卷”的模型。`reasoning effort` 越高，越像给它更多草稿纸和更多思考时间。

---

## 问题定义与边界

**Chain-of-Thought**，中文常叫“思维链”，白话说，就是模型不立刻报答案，而是先写出中间推理步骤。o1/o3 要解决的问题，可以抽象成：面对输入 $x$，在可能的推理路径集合 $\mathcal{R}(x)$ 中，找到一条既能导向正确答案、又不会长到失控的路径。

一个常见的形式化目标是：

$$
\max_{r\in\mathcal{R}(x)}\mathbb{E}\left[\mathbbm{1}_{a(r)=y} - \lambda \cdot \text{len}(r)\right]
$$

这里：
- $r$ 是一条推理路径，也就是“中间思考草稿”。
- $a(r)$ 是根据这条路径得到的最终答案。
- $\mathbbm{1}_{a(r)=y}$ 表示答对得 1 分，答错得 0 分。
- $\lambda \cdot \text{len}(r)$ 是长度惩罚，白话说，就是“别为了多想而无限啰嗦”。

这意味着系统并不是追求“推理越长越好”，而是追求“花出去的推理 token，能换来更高正确率”。**token** 可以理解成模型处理文本时使用的最小计费单位，中文里大致相当于几个字到一个词的一部分。

玩具例子：  
题目是“9 个苹果分给 4 个小朋友，每人先分 2 个，还剩几个？”  
直接答的模型可能会混乱；推理模型会先分解成两步：
1. 每人 2 个，共分出 $4 \times 2 = 8$ 个。
2. 总共 9 个，所以剩下 $9 - 8 = 1$ 个。

这个例子很小，但它体现了推理模型的边界：如果问题本身只要一两步，长推理不一定有意义；如果问题包含多层条件、需要回溯或纠错，推理预算就更值钱。

所以，o1/o3 的适用边界不是“任何任务都更强”，而是“在多步推理任务上更容易通过增加算力换来更高成功率”。

---

## 核心机制与推导

公开资料没有给出完整训练细节，但 OpenAI 官方已经给出两个足够重要的信号：

1. 训练上，使用了大规模强化学习，让模型“更会用自己的思维链去思考”。  
2. 推理上，随着 test-time compute 增加，性能持续上升。

**强化学习**，白话说，就是模型不是只学“标准答案长什么样”，还会根据结果好坏反复调整策略。把这件事放到推理模型里，可以理解为：模型不仅学“答案是什么”，还学“什么样的中间推理路径更容易得到正确答案”。

因此，业界对实现路径的合理推断是：o1/o3 不只是普通 SFT，而是对 reasoning traces 做了大规模强化训练。这里的 **reasoning trace**，白话说，就是模型在得出答案前写下来的中间推理轨迹。这个“推断”要和“官方明确披露”分开看：官方确认了强化学习与思维链的作用，但没有公开完整配方。

在部署层面，一个常见的预算抽象是：

$$
\text{budget}=\max\left(\min\left(\text{max\_tokens}\times \text{ratio},128000\right),1024\right)
$$

这里：
- `max_tokens` 是本次调用允许输出的总 token 上限。
- `ratio` 是 effort 对应的思考比例。
- `budget` 是分配给推理过程的 token 预算。

如果用 OpenRouter 的统一接口说明，常见比例如下：

| effort | ratio 近似值 | 含义 |
|---|---:|---|
| `minimal` | 0.10 | 只给很少思考预算 |
| `low` | 0.20 | 适合轻量多步任务 |
| `medium` | 0.50 | 默认平衡模式 |
| `high` | 0.80 | 明显增加推理深度 |
| `xhigh` | 0.95 | 尽量把预算留给思考 |

新手例子：  
若 `max_tokens=4096`，`effort="high"`，则推理预算约为

$$
4096\times 0.8 \approx 3277
$$

这表示大约 3277 个 token 用于“先想”，剩余约 819 个 token 留给最终回答。  
如果切到 `xhigh`，则预算约为：

$$
4096\times 0.95 \approx 3891
$$

只剩约 205 个 token 给最终回答。结论很直接：**更高 effort 不只是“更认真”，而是“把输出预算更多地挪给思考阶段”**。

真实工程例子是 ARC-AGI。它是一个强调抽象归纳与新题适应能力的基准，白话说，就是故意不给模型熟题，而是看它能不能临场看规律。ARC Prize 公布过 o3-preview 的结果：低 compute 约 75.7%，高 compute 约 87.5%。这说明 test-time compute 不是装饰参数，而是能实打实改变结果的资源旋钮。

---

## 代码实现

工程上最实用的做法，不是无脑把所有请求都开到 `high`，而是先做难度分级，再动态分配 effort。下面给一个可运行的 Python 玩具实现，模拟“根据任务难度分配推理预算”的逻辑：

```python
EFFORT_RATIO = {
    "minimal": 0.10,
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "xhigh": 0.95,
}

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def reasoning_budget(max_tokens: int, effort: str) -> int:
    ratio = EFFORT_RATIO[effort]
    return clamp(int(max_tokens * ratio), 1024, 128000)

def choose_effort(task_type: str, difficulty: int) -> str:
    # difficulty: 1~10
    if task_type in {"chat", "rewrite", "summary"}:
        return "minimal"
    if difficulty <= 3:
        return "low"
    if difficulty <= 6:
        return "medium"
    if difficulty <= 8:
        return "high"
    return "xhigh"

def allocate(max_tokens: int, task_type: str, difficulty: int):
    effort = choose_effort(task_type, difficulty)
    think_tokens = reasoning_budget(max_tokens, effort)
    answer_tokens = max_tokens - think_tokens
    if answer_tokens <= 0:
        raise ValueError("max_tokens 太小，没有留给最终答案的空间")
    return {
        "effort": effort,
        "reasoning_tokens": think_tokens,
        "answer_tokens": answer_tokens,
    }

a = allocate(4096, "math", 8)
assert a["effort"] == "high"
assert a["reasoning_tokens"] == 3276
assert a["answer_tokens"] == 820

b = allocate(4096, "chat", 2)
assert b["effort"] == "minimal"
assert b["reasoning_tokens"] == 1024  # 低于下限时被抬到 1024
assert b["answer_tokens"] == 3072
```

如果接入支持 reasoning 参数的兼容接口，请求结构可以写成这样：

```python
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="YOUR_API_KEY")

resp = client.chat.completions.create(
    model="openai/o3",
    messages=[
        {"role": "user", "content": "分析这个 Python traceback，并给出修复步骤"}
    ],
    max_tokens=4096,
    extra_body={
        "reasoning": {
            "effort": "high",
            "exclude": False
        }
    }
)

msg = resp.choices[0].message
content = getattr(msg, "content", "")
reasoning = getattr(msg, "reasoning", None)

print(content)
print(reasoning)
```

这里要特别说明一个工程事实：**并不是所有提供方都会返回完整 reasoning trace**。OpenRouter 文档把不同厂商做了统一抽象，但它也明确提示，像 OpenAI o 系列这类模型，很多时候不会直接把完整思维链原样返回给你。因此，客户端设计不要强依赖 `reasoning_trace` 一定存在，更稳妥的做法是：

| 场景 | 建议 |
|---|---|
| 只关心答案质量 | 传 `effort` 即可，不依赖中间推理内容 |
| 需要审计中间过程 | 选支持返回 reasoning 字段的提供方或模型 |
| 需要稳定延迟 | 给任务打标签，只在高难任务开高 effort |

真实工程例子：  
一个代码助手系统可以先用规则把请求分成“解释报错”“重构函数”“设计数据库 schema”三类。前两类多数用 `low/medium` 足够；涉及 schema 设计、复杂依赖梳理、跨文件修复时，再切到 `high`。这样做的收益不是“模型神奇变聪明”，而是“把贵预算集中投在真正难的请求上”。

---

## 工程权衡与常见坑

第一类坑是 **reward hacking**。这个词白话说，就是模型学会了“看起来像在认真推理”，但其实是在钻奖励函数的空子。比如它可能写出表面合理的长推理，最后仍然靠模式匹配碰答案，或者故意生成迎合评测器的格式，而不是真正解决问题。只要训练中引入强化学习，这个风险就存在，所以复杂推理模型要配套安全评测、人工抽检和反作弊数据。

第二类坑是成本爆炸。reasoning tokens 通常按输出 token 计费，这意味着高 effort 会直接增加调用成本与延迟。ARC-AGI 上 o3-preview 的高 compute 结果之所以震撼，也正因为它不是“免费午餐”，而是用大量测试时资源换来的。

一个很实际的成本策略如下：

| 任务类型 | effort | 目标 | 风险 |
|---|---|---|---|
| 批量客服问答 | `minimal` / `low` | 压低成本和延迟 | 推理不足导致复杂问题误答 |
| 普通代码解释 | `medium` | 平衡质量与速度 | 偶发多步错误 |
| 关键线上故障分析 | `high` | 提升复杂诊断成功率 | 成本和响应时间明显上升 |
| 研究型难题 | `xhigh` | 最大化推理深度 | 答案空间被挤压，且成本最高 |

还有两个容易忽略的实现坑：

1. `max_tokens` 太小。  
如果总预算过小，哪怕设置了 `high`，最后留给最终答案的 token 可能不足，造成回答被截断。

2. 把“更长推理”误当成“更高正确率”的充分条件。  
推理模型的优势在于更会分解与修正，不在于无穷扩写。对简单任务，增加 effort 可能只会增加成本，不会带来可见收益。

因此，最合理的生产策略通常是：  
先做轻量路由，再做 compute cap。**compute cap**，白话说，就是给每类任务设一个算力上限，防止系统在低价值请求上无限烧钱。

---

## 替代方案与适用边界

如果预算有限，不一定非要上 o1/o3 这类“高测试时算力”路径。许多场景里，便宜方案已经够用。

| 方案 | 白话解释 | 成本 | 适合任务 | 不足 |
|---|---|---:|---|---|
| Few-shot CoT | 给几个带推理过程的示例再提问 | 低 | 中短链推理 | 对难题稳定性一般 |
| Self-consistency | 多采样几次再投票 | 中 | 数学、逻辑小题 | 成本上升快 |
| Tool use | 让模型调用搜索、代码、计算器 | 中 | 需要外部精确工具 | 系统更复杂 |
| o1/o3 高 effort | 直接增加模型内部推理预算 | 高 | 多步复杂推理、规划、难代码题 | 延迟和成本高 |

适用边界可以简单记成一句话：  
**知识检索型任务优先外部工具，短链任务优先提示工程，长链复杂任务再考虑高 effort 推理模型。**

举例：
- 简单知识问答：“TCP 三次握手是什么？”用普通模型或 few-shot CoT 就够。
- 复杂 ARC 类图形归纳、难数学证明、跨模块代码修复，才更适合把预算开高。
- 对实时性极敏感的在线系统，例如低延迟客服、搜索建议补全，不适合默认开启高 effort。

所以，o1/o3 的价值，不是“统一替代所有模型”，而是提供了一条新的工程路线：当问题复杂到值得花额外算力时，可以用测试时计算换正确率。

---

## 参考资料

| 标题 | URL / 出处 | 核心内容 |
|---|---|---|
| Learning to reason with LLMs | https://openai.com/index/learning-to-reason-with-llms/ | OpenAI 官方说明 o1 的核心思路：强化学习让模型更会使用思维链，性能会随训练算力和测试时算力同步提升。 |
| Reasoning Tokens | https://openrouter.ai/docs/guides/best-practices/reasoning-tokens | 说明 `reasoning.effort`、`max_tokens`、预算上下限和近似比例，是工程实现推理预算分配的直接参考。 |
| Analyzing o3 and o4-mini with ARC-AGI | https://arcprize.org/blog/analyzing-o3-with-arc-agi | ARC Prize 对 o3/o3-preview 在 ARC-AGI 上的测试说明，能看出不同 compute 档位对结果的影响。 |
| OpenAI's o3 AI model scores lower on a benchmark than the company initially implied | https://techcrunch.com/2025/04/20/openais-o3-ai-model-scores-lower-on-a-benchmark-than-the-company-initially-implied/ | 补充了 o3-preview 与后续公开版本并不完全相同这一背景，提醒读者区分“预览成绩”和“生产版成绩”。 |
