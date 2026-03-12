## 核心结论

Llemma 是一个面向数学推理的开源基础模型。基础模型指还没有针对某个具体产品做指令微调的通用权重。它不是从零训练，而是以 Code Llama 为起点，再用 Proof-Pile-2 这组 55B token 的数学数据继续预训练。继续预训练指在已有模型上追加下一阶段自监督训练，而不是换架构重来。

它的关键价值有三点。第一，训练成本比从零做一个数学大模型低很多，因为直接继承了 Code Llama 已有的代码能力。第二，数学论文、数学网页和数学代码三类数据一起上，使模型同时学到“定义与证明”“题目与解法”“可执行计算”三种模式。第三，论文结果表明，Llemma-7B 和 Llemma-34B 在 MATH、GSM8K 这类数学推理任务上明显强于同规模开源基础模型，并且无需额外微调就能做 Python 工具调用和形式化证明。

一个容易记住的玩具例子是：把一个已经会写很多程序的学生，送去集中读数学论文、刷带公式的网页、看 SymPy 和 Lean 代码。这个学生未必立刻变成“会聊天的老师”，但会更擅长把题目拆成步骤、把式子转成程序、再把程序结果带回推理链。

| 模型 | 初始化 | 数学续训 token | MATH 贪心 | MATH 多数投票 |
|---|---|---:|---:|---:|
| Code Llama 7B | Llama 2 + 代码预训练 | 0 | 4.5% | 未报告 |
| Llemma 7B | Code Llama 7B | 200B | 18.0% | 33.5% |
| Code Llama 34B | Llama 2 + 代码预训练 | 0 | 12.2% | 未报告 |
| Llemma 34B | Code Llama 34B | 50B | 25.0% | 43.1% |
| Minerva 540B | PaLM 系列 | 164B 适配 token 量口径不同 | 33.6% | 50.3% |

这里要严格区分条件。Llemma-34B 的 `maj@256` 为 43.1%，高于 Minerva-540B 的贪心 33.6%，但低于 Minerva-540B 自己的多数投票 50.3%。如果不说明“是否用了投票”，结论就会被说错。

---

## 问题定义与边界

Llemma 解决的问题不是“如何做一个全能聊天模型”，而是更窄的一件事：如何在不从零重训的前提下，把代码预训练得到的结构化建模能力迁移到数学推理、数学工具使用和形式化证明。

它的任务边界也很清楚：

| 维度 | Llemma 的设定 |
|---|---|
| 模型类型 | Llama 家族 decoder-only，自回归语言模型 |
| 训练方式 | continued pretraining，继续预训练 |
| 目标函数 | 标准 LM loss，不引入额外分类头 |
| 数据核心 | Proof-Pile-2 为主，少量通用数据做正则化 |
| 目标能力 | 数学推理、Python 工具、证明器、形式化数学 |
| 不直接追求 | 对话对齐、全领域常识、超长上下文产品化 |

训练混合分布可以写成：

$$
D = 0.95\times \text{Proof-Pile-2} + 0.02\times \text{Pile}_{\text{non-Arxiv}} + 0.03\times \text{RedPajama GitHub}
$$

这里的“正则化”可以白话理解成防止模型过度偏科。也就是 95% 时间学数学，另外 5% 保留一些一般文本和代码分布，让模型不要把语言能力收缩得太厉害。

Proof-Pile-2 的组成如下：

| 数据子集 | token 量 | 作用 |
|---|---:|---|
| ArXiv | 29B | 学术定义、证明风格、公式上下文 |
| OpenWebMath | 15B | 网页题解、问答、教育内容 |
| AlgebraicStack | 11B | 数学相关代码、符号计算、证明脚本 |

真实工程例子是数学竞赛平台或教学平台。平台往往不能部署 500B 级闭源模型，但又希望模型既能写解题步骤，又能在必要时转成 Python 验算。Llemma 的定位正好是“成本可控的数学基础模型”，而不是直接替代通用聊天模型。

---

## 核心机制与推导

Llemma 的核心并不神秘，几乎全是“朴素但选得对”的工程组合。

第一层机制是目标函数没变，仍然是标准自回归语言建模：

$$
\mathcal{L} = - \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

这意味着它没有靠复杂奖励模型或大量人工标注硬拽出数学能力，而是靠更适合的预训练分布，让“下一个 token 预测”本身学到更强的数学结构。

第二层机制是代码预训练的正迁移。正迁移可以白话理解成：旧任务里学到的能力，对新任务有帮助。Code Llama 先看过 500B token 代码，已经擅长处理缩进、变量绑定、括号配对、长程依赖和可执行步骤。数学推理里恰好也有这些结构特征，比如定义引用、符号替换、分步求解、把自然语言压缩成形式表达。因此它不是“代码和数学相似”这么简单，而是“二者都要求中间状态严格可追踪”。

第三层机制是 RoPE 调整。RoPE 是旋转位置编码，可以理解成模型记录“第几个 token”的方式。论文中 7B 模型把 base period 从 $\theta=10^6$ 收缩到 $\theta=10^4$。直观上，这是把一个原本偏向很长尺度的位置刻度，改成更适合 4096 token 训练窗口的刻度。否则模型在当前上下文长度上容易出现位置分辨率不合适的问题。

| 设置 | 7B 处理方式 | 含义 |
|---|---|---|
| Code Llama 原始 RoPE | $\theta=10^6$ | 为长上下文扩展预留 |
| Llemma 7B 续训前 | 改为 $\theta=10^4$ | 让 4K 训练更稳定，更适合后续再做长上下文微调 |
| Llemma 34B | 保持 $\theta=10^6$ | 论文说明因算力限制未验证收缩是否无副作用 |

这里可以用一个玩具例子理解。假设你在 0 到 100 米的尺子上读 20 厘米的零件，会比拿 0 到 100 公里的地图去量更稳定。RoPE 收缩本质上就是把“位置尺子”调回更适合当前训练窗口的量程。

模型路线可概括为：

`Code Llama 预训练 -> 用 Proof-Pile-2 继续训练 -> 获得数学推理 + 工具使用 + 形式化证明能力`

为什么它能在不少场景里打出超参数量直觉的效果？原因不是“数学比自然语言简单”，而是数据分布更对口。Minerva 很强，但闭源且更大；Llemma 的优势在于把参数预算更多集中在目标领域，同时继承了代码模型天然的结构化推理偏好。

---

## 代码实现

从工程角度看，Llemma 的实现路线非常克制：不换模型结构，不新增复杂模块，直接在 Code Llama checkpoint 上继续跑 LM loss。

下面是一个可运行的 Python 玩具实现，用来模拟论文里的数据混合比例。它不是完整训练脚本，但能准确表达训练输入是怎么采样的。

```python
import random
from collections import Counter

def sample_source(rng: random.Random) -> str:
    x = rng.random()
    if x < 0.95:
        return "proof-pile-2"
    if x < 0.97:
        return "pile-non-arxiv"
    return "redpajama-github"

def simulate(n: int, seed: int = 0):
    rng = random.Random(seed)
    cnt = Counter(sample_source(rng) for _ in range(n))
    return {k: v / n for k, v in cnt.items()}

dist = simulate(100000, seed=42)

assert 0.94 < dist["proof-pile-2"] < 0.96
assert 0.015 < dist["pile-non-arxiv"] < 0.025
assert 0.025 < dist["redpajama-github"] < 0.035

print(dist)
```

如果把它换成真实训练，核心逻辑大致如下：

```python
# pseudo code
model = load_codellama_checkpoint("7b-or-34b")
if model.size == "7b":
    model.rope_theta = 10_000

loader = MixedLoader(
    sources={
        "proof_pile_2": 0.95,
        "pile_non_arxiv": 0.02,
        "redpajama_github": 0.03,
    },
    seq_len=4096,
)

for batch in loader:
    logits = model(batch["input_ids"])
    loss = autoregressive_lm_loss(logits, batch["labels"])
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

训练超参的公开信息如下：

| 项目 | Llemma 7B | Llemma 34B |
|---|---:|---:|
| 初始化 | Code Llama 7B | Code Llama 34B |
| 续训 token | 200B | 50B |
| 全局 batch | 4M tokens | 4M tokens |
| 上下文长度 | 4096 | 4096 |
| 峰值学习率 | $1\times10^{-4}$ | $5\times10^{-5}$ |
| 训练步数 | 42000 | 12000 |
| RoPE | $\theta=10^4$ | $\theta=10^6$ |
| 训练资源 | 256 张 A100 40GB | 256 张 A100 40GB |

这说明它不是“靠神秘技巧提分”，而是靠数据配方、初始化选择和训练预算分配。

---

## 工程权衡与常见坑

Llemma 的工程价值很高，但坑也非常明确。

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| 忽略 RoPE 设置 | 长序列行为异常，续训与后续扩展不稳定 | 7B 按论文将 $\theta$ 从 $10^6$ 调到 $10^4$ |
| 只喂论文不喂代码 | 会写解释，不会把推理转成工具调用 | 保留 AlgebraicStack 这类数学代码数据 |
| 只看贪心结果 | 低估模型上限 | 对 MATH、GSM8K 使用多数投票 |
| 把投票结果和贪心结果混着比 | 结论失真 | 明确区分 `greedy`、`maj@k` |
| 以为它天然适合对话 | 输出可能偏学术、偏基础模型风格 | 需要后续指令微调或产品层包装 |

多数投票的作用尤其重要。多数投票指模型采样生成多个答案，再选出现次数最多的那个。对数学题，这相当于让模型“多做几遍，再看哪个结论最稳定”。论文里 Llemma-34B 在 MATH 上从贪心 25.0% 提升到 `maj@256` 的 43.1%，这不是小修小补，而是能力释放方式的变化。

但它也不是无限有效。若模型本身不会做某类题，多投几次只会重复错法；如果题目需要外部检索、图像理解或极长证明链，多数投票也补不上缺失能力。

另一个容易忽略的点是训练稳定性。论文里 7B 原计划训练 48000 步，但在 42000 步后遇到 NaN loss。NaN 就是数值溢出后变成“不是一个数”。这说明即便路线朴素，大规模续训依然会踩到优化或硬件稳定性问题。

---

## 替代方案与适用边界

如果目标是最高绝对分数，Llemma 不是终点。Minerva、后续更强的闭源数学模型、以及带搜索和程序执行的代理系统，都可能更强。但 Llemma 的优势是公开、可复现、可部署。

| 方案 | 优点 | 局限 |
|---|---|---|
| Llemma 7B/34B | 开源、数学专化、部署门槛相对低 | 不是对话优化模型，极限性能仍落后更大闭源系统 |
| Minerva | 数学基线强，论文影响大 | 闭源，复现与部署受限 |
| 通用基础模型 + 数学微调 | 灵活，可按自己数据定制 | 需要自己解决数据质量和工具接口 |
| 推理代理 + Python/CAS/检索 | 上限高，可验证结果 | 系统复杂度高，不是单模型问题 |

适用边界也要说清楚。Llemma 适合这些场景：教学平台、竞赛题批改辅助、本地部署的数学助手、把自然语言题目转成 Python 或证明器脚本的研究原型。它不适合这些场景：超长上下文文档推理、强对话对齐产品、依赖多模态输入的数学应用。

一句话概括选择标准：如果你要的是“开源、能本地跑、对数学更专注”的基础模型，Llemma 很合适；如果你要的是“现成 API、全场景最强、少管训练细节”，那就该看更大的闭源系统或更完整的代理框架。

---

## 参考资料

1. ICLR 2024 论文《Llemma: An Open Language Model for Mathematics》：核心来源，给出模型初始化、训练配方、评测结果与工具使用结论。
2. ICLR 2024 论文 PDF 附录：提供 Proof-Pile-2 构成、训练超参、RoPE 设置、AlgebraicStack 过滤细节。
3. EleutherAI `llemma_7b` / `llemma_34b` 模型卡：确认 7B 训练 200B token、34B 训练 50B token。
4. EleutherAI `proof-pile-2` 数据说明与相关公开介绍：用于核对 55B token 及 ArXiv、OpenWebMath、AlgebraicStack 的分布。
5. Minerva 论文《Solving Quantitative Reasoning Problems with Language Models》：对照理解 equi-parameter 比较和多数投票评测口径。
