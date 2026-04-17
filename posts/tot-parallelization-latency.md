## 核心结论

ToT，Tree of Thoughts，中文可理解为“思维树搜索”，本质上是把一次推理从单条链改成多条候选路径并行探索。对复杂任务，它通常比 CoT，Chain of Thought，即“单条思维链”，更稳，因为模型不必把全部希望押在第一条路径上。

但 ToT 的代价也直接：如果每层保留 $b$ 个候选、搜索深度为 $D$，最大节点数近似按 $O(b^D)$ 增长。常见设置里 $b=5$，这意味着即使只看 3 层，也可能需要处理几十到上百个状态。原始串行实现里，每个节点都要“生成一次 + 评估一次”，延迟会远高于 CoT。

并行化的核心不是“让树变小”，而是把原本串行等待的节点扩展、状态评分和候选调度压缩到更少的推理轮次里。实践上最有效的三类手段是：

| 策略 | 直观作用 | 主要收益 | 主要代价 |
| --- | --- | --- | --- |
| 批量节点扩展 | 一次送入多个候选状态 | 提高 GPU 利用率，减少轮次 | 需要 padding 和批量管理 |
| 异步评估 | 生成后立刻发起评分，不阻塞下一批准备 | 隐藏等待时间 | 调度复杂度上升 |
| 投机式搜索 | 先并行展开高潜力路径，再验证是否命中 | 降低平均路径等待 | 命中率低时会浪费算力 |

玩具例子可以这样理解：CoT 像一次只走一条岔路，错了再从头来；ToT 像同时让 5 个人各走一条，先看谁接近出口，再继续扩展那几条。它更稳，但如果 5 个人都要单独坐车，时间还是长；并行化的意义，是让这 5 个人拼车。

公开结果能说明两个事实。第一，ToT 的搜索收益是真实存在的。原始 ToT 论文在 Game of 24 上报告：GPT-4 的标准 CoT 成功率约 4%，ToT 达到 74%。第二，延迟优化也是真实存在的。像 LLMCompiler 这类并行执行框架，在相关多步推理/调用场景中报告了最高约 $3.7\times$ 的延迟加速，说明“多分支推理很慢”不是只能接受的宿命，而是可以通过系统设计压缩的工程问题。

---

## 问题定义与边界

ToT 解决的问题，不是“模型会不会输出答案”，而是“模型能不能在中间步骤犯错后仍有回退空间”。这里的 thought 可以理解为“一个中间推理状态”，也就是还没得到最终答案，但已经形成了一个可继续展开的局部思路。

形式化地看，一棵 ToT 搜索树包含三类操作：

1. 生成：从当前状态扩出若干候选 thought。
2. 评估：给候选打分，判断谁值得保留。
3. 选择：只保留 top-k 继续往下走。

若每层保留 $b$ 个状态，深度为 $D$，最大展开节点数可写为：

$$
N_{\max} = \sum_{i=0}^{D} b^i = \frac{b^{D+1}-1}{b-1}
$$

当 $b=5, D=4$ 时，理论上最多会接近 781 个节点。实际系统会剪枝，但延迟压力已经足够明显。

玩具例子是 24 点游戏。给定四个数，模型要组合成 24。CoT 只有一条链，第一步如果选错运算，后面几乎全错。ToT 每层保留 5 条候选，即使其中 3 条明显偏离，也还有 2 条可继续。这里的收益不是“每一步更聪明”，而是“允许早期错误存在，但不让它唯一化”。

边界也要说清楚。ToT 不是所有任务都值回成本：

| 任务类型 | 是否适合 ToT | 原因 |
| --- | --- | --- |
| 多步组合搜索 | 适合 | 早期选择错误会传染全局 |
| 有明确中间状态评分 | 适合 | 评估器容易做剪枝 |
| 单步问答 | 通常不适合 | 搜索空间太小，树开销不划算 |
| 极长上下文生成 | 需谨慎 | KV cache 膨胀明显，批量代价高 |

所以本文讨论的边界是：复杂推理、多步决策、每步可局部评估、且系统允许做批量推理优化的场景。离开这个边界，ToT 可能只是把简单问题做复杂。

---

## 核心机制与推导

ToT 慢，根因不是“树结构”三个字，而是大量小而散的请求把 GPU 用坏了。GPU 擅长大批量、规则化计算，不擅长频繁处理长度不同、阶段不同、优先级不同的小请求。

### 1. 批量节点扩展

每个候选状态长度不同，直接一起送模型会造成批次不整齐。工程上通常要先对齐序列或 KV cache。一个常见抽象写法是：

$$
\mathrm{KV}^{1\sim n}_{\mathrm{pad}}=
\mathtt{concat}\left(\mathbf{0}_{L-|\mathrm{KV}^{1\sim n}|},\mathrm{KV}^{1\sim n}\right)
$$

白话解释：把短序列前面补零，补到同样长度，再一起做一次前向。这样原本要执行 $n$ 次的小推理，可以合并成 1 次较大的批量推理。

这像把多个“思路”拼成一班车统一出发，而不是每个 thought 单独发一辆车。代价是 padding 会带来无效计算，但收益通常更大，因为 GPU 吞吐更高。

### 2. 异步评估

原始 ToT 常见写法是“先生成，再等评分，再选 top-k，再进入下一层”。这导致大量空等。异步评估的思路是：候选一生成出来，评分请求就立即排队；调度器同时准备下一批待扩展状态，不让 CPU/GPU 都停着等结果。

如果把一次层级搜索的时间写成：

$$
T_{\text{serial}} = T_{\text{gen}} + T_{\text{eval}} + T_{\text{select}}
$$

而异步后可重叠生成准备和部分评估，则有效时间更接近：

$$
T_{\text{async}} \approx \max(T_{\text{gen}}, T_{\text{eval}}) + T_{\text{select}}
$$

不是把计算量消灭，而是把等待重叠。

### 3. 投机式搜索

投机式搜索可以理解为“先押注几条最可能成功的分支”。它的前提不是正确率 100%，而是平均上命中收益大于额外开销。

若单个投机分支命中概率为 $p$，一次并行发出 $k$ 个样本，则至少命中一次的概率是：

$$
P(\text{hit}) = 1-(1-p)^k
$$

命中率越高，越值得提前展开；命中率越低，越像白跑。最近一些调度工作强调，投机请求会抬高 decode 并发数，过高时每步 decode 速度可能下降约 20%，prefill 成本还会接近线性增长。所以投机不是“越多越快”，而是要受负载约束。

有些系统会用动态并发近似式来控制批量规模，例如：

$$
|P|=\frac{O_{\text{max}}-O_{\text{init}}}{O_{\text{peak}}-O_{\text{init}}}
$$

它表达的是一个简单思想：根据当前负载离峰值还有多少空间，决定还能塞多少并行候选，而不是固定写死并发度。

真实工程例子是多工具调用代理。一个代理在“查资料、算数值、查天气、整理答案”四步之间，经常存在可并行的子任务。若仍按 ReAct 风格一步一步串行调用，LLM 会把工具等待时间全部暴露给用户。并行调度器把可独立的任务先发出，再等依赖汇总，可在保持质量的前提下显著降延迟。这和 ToT 的节点并行，本质上是同一件事：把树中的独立边提前重叠执行。

---

## 代码实现

下面用一个可运行的 Python 玩具实现演示 ToT 的并行批量扩展。这里不调用真实大模型，而是用简单规则模拟 `model.eval(batch)`，重点看批量准备、并发评估和 top-k 更新流程。

```python
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

@dataclass
class Path:
    expr: str
    value: int
    score: float = 0.0
    depth: int = 0
    history: list[str] = field(default_factory=list)

def expand(path: Path):
    # 玩具例子：尝试 +1, +2, *2 三种 thought
    ops = [
        (f"({path.expr}+1)", path.value + 1),
        (f"({path.expr}+2)", path.value + 2),
        (f"({path.expr}*2)", path.value * 2),
    ]
    return [
        Path(expr=e, value=v, depth=path.depth + 1, history=path.history + [e])
        for e, v in ops
    ]

def pad_batch(paths):
    # 用字符串长度模拟序列 padding
    max_len = max(len(p.expr) for p in paths)
    return [p.expr.rjust(max_len, "_") for p in paths]

def score_one(path: Path, target: int):
    # 分数越高表示越接近目标
    return -abs(target - path.value)

def eval_batch(paths, target: int):
    _ = pad_batch(paths)  # 真实系统里这里会准备 token/KV batch
    with ThreadPoolExecutor(max_workers=4) as ex:
        scores = list(ex.map(lambda p: score_one(p, target), paths))
    for p, s in zip(paths, scores):
        p.score = s
    return paths

def update_paths(candidates, width: int):
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:width]

def tot_search(start: int, target: int, width: int = 2, depth: int = 3):
    frontier = [Path(expr=str(start), value=start)]
    for _ in range(depth):
        candidates = []
        for path in frontier:
            candidates.extend(expand(path))
        candidates = eval_batch(candidates, target)
        frontier = update_paths(candidates, width)
    return frontier

result = tot_search(start=3, target=10, width=2, depth=3)
best = result[0]
assert best.value == 10
assert best.score == 0
print(best.expr, best.value, best.score)
```

这段代码对应的系统流程是：

1. `build_batch()`：把当前 frontier 的所有候选统一整理成一个 batch。
2. `model.eval(batch)`：并发评分，而不是逐个等待。
3. `update_paths()`：按分数保留 top-k，进入下一层。

如果要再加一层投机调度，可以把“高分路径”优先送入下一轮，并根据命中率动态调整并发度。伪代码如下：

```python
def schedule_speculative(paths, hit_rate, max_parallel):
    if hit_rate < 0.3:
        return paths[:1]
    if hit_rate < 0.6:
        return paths[: min(2, max_parallel)]
    return paths[:max_parallel]
```

新手理解时只要抓住一点：并行化不是改变搜索逻辑，而是改变“哪些请求一起送、哪些请求先送、哪些请求不值得送”。

---

## 工程权衡与常见坑

ToT 并行化最容易犯的错误，是把“更多并发”误当成“更低延迟”。实际上它是资源竞争问题。

| 常见坑 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 过度 speculative | 平均延迟反而升高 | 命中率低，额外请求挤占 decode | 按实时 hit rate 和负载限流 |
| 忽略长度对齐 | batch 很大但吞吐不高 | padding 过多，无效计算严重 | 按长度分桶后再批量 |
| 评分器过重 | 评估比生成还慢 | 每个节点又调用大模型 | 用轻量 value model 或规则过滤 |
| frontier 固定过宽 | 后期节点爆炸 | 所有层都按同样宽度搜索 | 早宽后窄，动态 top-k |
| 调度优先级冲突 | 新请求拖慢旧请求输出 | prefill 与 decode 抢算力 | 区分 prefill/decode 阶段调度 |

这里有一个很实际的工程事实：prefill 和 decode 的性能特征不同。prefill 像“先把整段输入读完”，更吃算力；decode 像“每次吐一个 token”，更容易受并发影响。投机请求一多，decode 会被拖慢，用户就看到输出一卡一卡。近期一些调度研究已经明确指出，投机并发高时 decode 每步速度会下降约 20%，而 prefill 成本近似随 batch 增长。这意味着搜索系统不能只看单任务最优，还要看整机吞吐和尾延迟。

所以 ToT 的工程优化目标不是单纯追求最小 step 数，而是平衡三件事：搜索质量、平均延迟、服务稳定性。一个常见做法是“前几层宽搜，后几层窄搜”，因为前期分支差异大，值得探索；后期只保留高分路径，避免把资源花在低价值节点上。

---

## 替代方案与适用边界

预算有限时，不必在所有任务上全量启用 ToT。更实用的选择是分层策略。

| 策略 | 成功率潜力 | 延迟 | 资源占用 | 适用场景 |
| --- | --- | --- | --- | --- |
| ToT | 高 | 高，但可并行优化 | 高 | 多步搜索、组合推理 |
| 有限 beam CoT | 中 | 中 | 中 | 候选不多、想保留少量分支 |
| MCTS | 中到高 | 高 | 高 | 需要长期回报估计的复杂规划 |
| 纯 CoT | 低到中 | 低 | 低 | 简单问答、快速响应 |

有限 beam CoT 可以理解为“窄版 ToT”。它仍保留多个候选，但通常没有显式状态评估器，适合预算紧、但又不想完全退回单链推理的场景。

MCTS，Monte Carlo Tree Search，中文常译“蒙特卡洛树搜索”，本质是通过模拟和回报估计来决定扩展哪条支路。它在博弈或长期规划里有优势，但对语言任务来说，回报设计和 rollout 成本往往更复杂。相比之下，ToT 更容易和 LLM 的“生成 + 自评”接口对接。

对初级工程师来说，最实用的经验是：

1. 默认先用 CoT。
2. 如果错误主要来自“早期路径选错”，升级到窄宽度 ToT。
3. 如果 ToT 太慢，先做批量扩展和异步评估。
4. 只有命中率可控时，再加投机式搜索。

也就是说，ToT 不是 CoT 的替代品，而是 CoT 在复杂推理场景下的升级档。预算紧张时，完全可以先串行 CoT，只有在多个候选分数接近、且单条路径不稳时，再升级到并行 ToT。这比一上来全树搜索更符合真实工程预算。

---

## 参考资料

1. Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. NeurIPS 2023. 说明：ToT 原始论文，给出 Game of 24 上 CoT 约 4%、ToT 约 74% 的代表性结果。  
   URL: https://arxiv.org/abs/2305.10601

2. Princeton NLP 官方代码仓库. *tree-of-thought-llm*. 说明：提供 ToT + BFS 的实验实现，仓库参数里可直接看到 `n_select_sample` 对应保留宽度。  
   URL: https://github.com/princeton-nlp/tree-of-thought-llm

3. Kim, Moon, Tabrizi, Lee, Mahoney, Keutzer, Gholami. *An LLM Compiler for Parallel Function Calling*. ICML 2024, PMLR 235. 说明：展示并行任务编排如何在多步推理/工具调用场景下降低延迟，报告最高约 $3.7\times$ latency speedup。  
   URL: https://proceedings.mlr.press/v235/kim24y.html

4. Wang, Wu, Lai, Zhang, Zhou. *SEED: Accelerating Reasoning Tree Construction via Scheduled Speculative Decoding*. COLING 2025. 说明：直接针对 reasoning tree 的构建延迟，使用 scheduled speculative decoding 管理 draft model 调度与显存占用。  
   URL: https://aclanthology.org/2025.coling-main.328/

5. *Reducing Latency of LLM Search Agent via Speculation-based Algorithm-System Co-Design*. 2025 预印本。说明：从系统角度分析 speculative 请求对 decode/prefill 的干扰，给出负载感知调度思路，并指出高并发 speculation 可能使 decode 每步速度下降约 20%。  
   URL: https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/66ef348c-c150-46d7-b2fc-c6f2afb217a5.pdf
