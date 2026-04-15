## 核心结论

Universal Transformer，简称 UT，本质上不是把 Transformer 继续往深处堆，而是把**同一组参数共享的 Transformer 计算块反复执行**。这里的“参数共享”可以先理解成：第 1 轮、第 2 轮、第 3 轮都用同一套权重，不是每一层各有一套独立参数。

它解决的核心问题是：普通 Transformer 对所有 token 都使用固定层数，算力分配是静态的；UT 允许不同位置根据自身难度决定要“想几轮”。这类机制通常叫**自适应计算**，白话讲就是“简单位置少算几步，复杂位置多算几步”。

普通 Transformer 可以理解成“统一做 6 轮处理后结束”；UT 更像“先做 1 轮，再判断哪些位置还没处理清楚，再继续做下一轮”。

| 维度 | 普通 Transformer | Universal Transformer |
|---|---|---|
| 深度 | 固定层数 | 共享层循环多次 |
| 参数 | 每层通常独立 | 各轮共享 |
| 计算分配 | 所有 token 相同 | 每个位置可不同 |
| 主要收益 | 结构简单、并行友好 | 自适应算力分配 |
| 主要代价 | 可能浪费算力 | 控制流更复杂 |

核心公式可以先记两组。第一组是共享迭代：

$$
A^{(t)} = SelfAttn(H^{(t-1)})
$$

$$
H^{(t)} = LayerNorm(H^{(t-1)} + Transition(A^{(t)}))
$$

第二组是停止机制，也叫 **halting mechanism**，可以先理解成“每轮都估计当前位置是否该停”：

$$
p_i^{(t)} = \sigma(w_h^\top h_i^{(t)} + b_h), \quad
s_i^{(t)} = s_i^{(t-1)} + p_i^{(t)}
$$

当 $s_i^{(t)} \ge \tau$，或者已经达到最大轮数 $T_{max}$，位置 $i$ 就停止继续迭代。

---

## 问题定义与边界

UT 试图解决的问题很具体：**固定深度模型无法按 token 难度分配计算量**。一句话里，“the”“is”这类语义稳定、依赖少的位置，往往很快就足够明确；而代词、关系词、需要跨句找证据的位置，通常需要更多轮上下文整合。

设输入长度为 $n$，隐藏维度为 $d$，初始表示记作：

$$
H^{(0)} \in \mathbb{R}^{n \times d}
$$

其中 $H^{(t)}$ 表示第 $t$ 轮之后的整句表示。本文只讨论**编码器视角**下的 UT 主机制，不展开解码器变体、特定任务头和论文中的全部实验细节。

下面这张表给出问题边界：

| 项目 | 内容 |
|---|---|
| 固定深度的局限 | 所有位置统一计算深度，简单 token 可能被过度计算，复杂 token 可能不够 |
| UT 想解决的点 | 用位置级动态停止，为不同 token 分配不同计算步数 |
| 它不是什么 | 不是单纯“更深层的 Transformer”，也不是所有任务都优于普通 Transformer |
| 本文边界 | 只讲共享迭代、位置级 halting、工程实现思路 |

玩具例子可以直接看停止概率。设阈值 $\tau=0.9$，最大步数 $T_{max}=3$。

- 位置 A 的停止概率依次为 `0.30, 0.40, 0.35`，累计是 `0.30 -> 0.70 -> 1.05`，所以它跑 3 轮。
- 位置 B 的停止概率依次为 `0.65, 0.30`，累计是 `0.65 -> 0.95`，所以它只跑 2 轮。

这说明同一句话里，不同 token 的“思考轮数”可以不同。UT 的边界也在这里：如果任务里所有位置难度都差不多，这种动态控制未必值得引入。

---

## 核心机制与推导

UT 的主循环可以概括成“自注意力更新 + 前馈变换 + 停止判断”。**自注意力**可以先理解成“每个位置都能读取整句其他位置的信息”。

第 $t$ 轮迭代的标准写法是：

$$
A^{(t)} = SelfAttn(H^{(t-1)})
$$

$$
H^{(t)} = LayerNorm(H^{(t-1)} + Transition(A^{(t)}))
$$

这里关键不是公式本身，而是这两步在每一轮都复用同一组参数。这样做带来两个结果：

1. 模型拥有类似递归过程的归纳偏置，也就是“反复精炼同一份表示”。
2. 深度不再完全由网络结构预先写死，而是部分由 halting 决定。

halting 的直觉是：每轮都让每个位置自己报一个“还需不需要继续”的分数。对位置 $i$，第 $t$ 轮有：

$$
p_i^{(t)} = \sigma(w_h^\top h_i^{(t)} + b_h)
$$

其中 $\sigma$ 是 sigmoid，把实数压到 $(0,1)$。把它累加起来：

$$
s_i^{(t)} = s_i^{(t-1)} + p_i^{(t)}
$$

当累计量满足 $s_i^{(t)} \ge \tau$ 时，该位置停止；如果一直没到阈值，也必须在 $T_{max}$ 强制停止。这个上限很重要，因为它是训练稳定性和部署可控性的兜底。

为了让停止过程可导，最后一轮通常不会机械地全加进去，而是用 **remainder**，也就是“差多少刚好到阈值”：

$$
remainder_i^{(t)} = 1 - s_i^{(t-1)}
$$

最后输出可以看成多轮状态的连续加权和，而不是硬切断。这一点继承了 ACT 思路，目的不是让公式更复杂，而是让梯度能稳定回传。

把整个流程写成时序，可以看成：

`输入表示 -> 共享 SelfAttn/Transition 第1轮 -> 计算停止概率 -> 未停止位置进入第2轮 -> ... -> 达到阈值或 T_max -> 输出`

如果用新手能直接理解的话说，一个 token 像在反复修改草稿。每改完一轮，就问一次“现在够清楚了吗”。够了就停，不够就继续。

真实工程例子是长文档问答或多跳推理。停用词、标点附近位置、明显实体往往很快稳定；而代词解析、跨段证据对齐、关系词附近位置更可能获得更多轮迭代。UT 的价值就在于把算力集中到这些真正难的位置上。

---

## 代码实现

实现 UT 的关键不是“把层数加深”，而是拆清楚四件事：共享变换、循环控制、位置级停止状态、最后一轮 remainder 加权。

下面给出一个可运行的最小 Python 版本。它不实现真实的自注意力，只保留 halting 的核心逻辑，用来验证位置级动态停止是怎么工作的。

```python
from math import isclose

def universal_transformer_halting(prob_steps, tau=0.9, t_max=5):
    """
    prob_steps: list[list[float]]
        prob_steps[t][i] 表示第 t 轮位置 i 的停止概率
    返回:
        steps_taken: 每个位置实际运行了几轮
        final_weights: 每个位置每轮的加权系数，最后一轮会使用 remainder
    """
    num_steps = min(len(prob_steps), t_max)
    n = len(prob_steps[0])
    halting_sum = [0.0] * n
    halted = [False] * n
    steps_taken = [0] * n
    final_weights = [[] for _ in range(n)]

    for t in range(num_steps):
        probs = prob_steps[t]
        for i, p in enumerate(probs):
            if halted[i]:
                final_weights[i].append(0.0)
                continue

            if halting_sum[i] + p >= tau:
                remainder = tau - halting_sum[i]
                final_weights[i].append(remainder)
                halting_sum[i] = tau
                halted[i] = True
                steps_taken[i] = t + 1
            else:
                halting_sum[i] += p
                final_weights[i].append(p)

        if all(halted):
            break

    for i in range(n):
        if steps_taken[i] == 0:
            steps_taken[i] = num_steps

    return steps_taken, final_weights

# 玩具例子
prob_steps = [
    [0.30, 0.65],
    [0.40, 0.30],
    [0.35, 0.10],
]
steps, weights = universal_transformer_halting(prob_steps, tau=0.9, t_max=3)

assert steps == [3, 2]
assert isclose(sum(weights[0]), 0.9, rel_tol=1e-9)
assert isclose(sum(weights[1]), 0.9, rel_tol=1e-9)
print("ok", steps, weights)
```

上面代码里有三个核心变量：

| 模块名 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `stop_head` | 从当前位置表示预测停止概率 | `h_i^(t)` | `p_i^(t)` |
| `halting_sum` | 记录累计停止质量 | `p_i^(t)` | `s_i^(t)` |
| `remainder` | 最后一轮补足到阈值 | `tau - s_i^(t-1)` | 最终连续权重 |

把它扩展成真实模型，结构通常类似下面的伪代码：

```python
H = H0
halting_sum = zeros(batch, seq_len)
active_mask = ones(batch, seq_len, dtype=bool)

for t in range(T_max):
    A = self_attn(H, attn_mask)
    H_new = layer_norm(H + transition(A) + time_step_embedding[t])

    p = sigmoid(stop_head(H_new))  # [batch, seq_len]
    # 只更新仍然活跃的位置
    # 计算 newly_halted、remainder、加权输出
    # batch 内不同 token 不同步结束，通常依赖 mask 处理

    H = H_new
    if all_positions_halted:
        break
```

工程上要注意两类 mask：

1. 注意力 mask：保证 padding 位置不会参与无效计算。
2. active mask：表示哪些 token 还没停止，只对这些位置更新 halting 状态和输出加权。

---

## 工程权衡与常见坑

UT 的收益是自适应计算，但代价是训练和部署都更复杂。对工程来说，真正要看的不是“有没有动态停止”，而是它是否换来了可观收益。

一个常用指标是平均 **ponder time**，可以先理解成“平均每个 token 实际跑了多少轮”。若样本总 token 数为 $N$，第 $i$ 个 token 运行了 $T_i$ 轮，则平均计算步数可写成：

$$
\bar{T} = \frac{1}{N}\sum_{i=1}^{N} T_i
$$

如果精度提升很小，但 $\bar{T}$ 和延迟明显上升，这种设计就未必划算。

| 指标 | 固定深度 Transformer | Universal Transformer |
|---|---|---|
| 准确率潜力 | 稳定 | 对难度不均匀任务可能更优 |
| 平均步数 | 固定 | 动态 |
| 推理延迟 | 更可预测 | 更难预测 |
| 实现复杂度 | 低 | 高 |
| 部署友好性 | 高 | 一般 |

常见坑与规避建议如下：

| 坑点 | 问题 | 规避建议 |
|---|---|---|
| 把 UT 误解成更深层 Transformer | 会忽略“共享参数 + 动态步数”本质 | 明确区分“堆叠深度”和“循环深度” |
| 忽略 time-step / position encoding | 模型难区分“现在是第几轮” | 给迭代步显式编码 |
| 没有 `T_max` 兜底 | 个别位置可能长时间不停止 | 始终设置最大步数 |
| 只看精度，不看平均步数 | 可能得到低效模型 | 同时监控 `ponder time`、延迟、吞吐 |
| 阈值设得过低或过高 | 太低会早停，太高又退化成固定深度 | 联合调 `tau` 与 `T_max` |

真实工程里最容易踩的点是 batch 内分支开销。因为不同 token 会在不同轮停止，所以虽然理论上节省了部分位置的计算，实际 GPU 上不一定线性省时。尤其在高吞吐在线服务中，动态控制流可能抵消理论收益。

---

## 替代方案与适用边界

UT 适合的不是“所有 NLP 任务”，而是**输入内部难度差异明显、并且愿意为自适应计算支付实现复杂度**的场景。

| 方案 | 特点 | 适用情况 |
|---|---|---|
| 固定深度 Transformer | 结构简单、并行友好 | 通用场景，尤其是低延迟部署 |
| Universal Transformer | 共享迭代、位置级动态深度 | 难度不均匀、需要按 token 分配算力 |
| ACT 类方法 | 更强调动态计算时间 | 研究或需要细粒度计算控制的场景 |
| 轻量化编码器 | 直接降模型成本 | 预算敏感、任务本身不复杂 |

再看推荐边界：

| 任务类型 | 预期收益 | 风险 | 是否推荐 |
|---|---|---|---|
| 长文本理解 | 中到高 | 实现复杂、延迟波动 | 可考虑 |
| 多跳推理 | 中到高 | 调参成本高 | 可考虑 |
| 简单分类 | 低 | 复杂度不值当 | 通常不推荐 |
| 严格低延迟在线服务 | 不确定 | 动态分支不友好 | 谨慎 |

一个简单选择准则是：

- 如果任务难度在 token 之间差异明显，而且你愿意监控平均步数、吞吐和阈值调度，可以考虑 UT。
- 如果任务本身简单、样本长度短、服务延迟要求高，固定层数 Transformer 往往更实用。
- 如果目标只是降低成本，而不是做更细的算力分配，轻量化模型通常更直接。

---

## 参考资料

1. Universal Transformers, arXiv:1807.03819  
2. Universal Transformers, ar5iv HTML 版本  
3. Adaptive Computation Time for Recurrent Neural Networks, arXiv:1603.08983  
4. Tensor2Tensor 官方仓库 README 与 `universal_transformer` 相关实现入口  

阅读顺序建议：

1. 先看 UT 原论文摘要和模型图，建立“共享参数 + 动态 halting”的整体认识。  
2. 再看 ar5iv HTML，补公式和附录里的停止机制细节。  
3. 如果对 remainder、ponder time 的来源不清楚，再看 ACT 论文。  
4. 最后看 Tensor2Tensor 的实现入口，理解工程层面的 mask、最大步数和训练细节。  

概念对照表：

| 术语 | 含义 |
|---|---|
| UT | 共享 Transformer 块并循环执行的模型 |
| ACT | 动态决定计算步数的方法族 |
| halting | 判断某个位置是否停止继续迭代 |
| remainder | 最后一轮用于补足阈值的连续权重 |
| ponder time | 平均每个位置实际使用的计算步数 |
