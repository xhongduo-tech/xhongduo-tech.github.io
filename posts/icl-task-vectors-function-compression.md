## 核心结论

上下文学习中的“任务向量”可以理解为一张内部生成的“规则卡片”。规则卡片的白话意思是：模型先把几个示例里共同的规律压缩成一个向量，后面回答新问题时主要依赖这个向量，而不是每一步都重新翻看全部示例。

形式化地说，给定示例集 $S$、查询 $x$、Transformer $T$，一个常见的机制分解是：

$$
T([S, x]) = f(x; A(S))
$$

其中，$A$ 是“任务提取器”，负责把示例集压缩成任务向量 $\theta(S)$；$f$ 是“规则执行器”，负责拿着查询 $x$ 和这个向量生成输出。Hendel 等人的工作说明，这种分解不只是解释上的方便，而是能通过中间层干预实验被验证的内部机制。

初学者版玩具例子如下。给模型三个样例：

- `2 -> 偶数`
- `7 -> 奇数`
- `10 -> 偶数`

模型后续看到 `13 -> ?` 时，并不一定逐条回看前面三行，更可能是在中间层形成“这是奇偶分类任务”的任务向量，再拿它去处理 `13`。

真实工程例子更直接。对于 bitwise AND 任务，若上下文里有示例 $(0,0)\to0,(1,1)\to1$，模型在中层可能形成“输出仅在两位都为 1 时为 1”的任务向量。后面询问 `1 AND 0` 时，如果把正确任务向量注入后层，模型更容易输出 `0`；如果注入错误向量，比如 XOR 风格的规则，精度会明显下降。

一个简图可以写成：

$$
S \xrightarrow{\text{前/中层 }A} \theta(S) \xrightarrow{\text{后层 }f} y,\quad x \to f(x;\theta)
$$

这也是本文最重要的结论：上下文学习并不只是“把示例塞进去”，而是“先压缩任务，再执行任务”。

---

## 问题定义与边界

这里的问题不是“模型能不能 few-shot”，而是“模型在内部怎样把 few-shot 示例变成一个可复用的任务描述”。

我们统一符号：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $S$ | 示例集 | 提示词里给模型看的若干输入输出对 |
| $x$ | 查询 | 当前真正要回答的新输入 |
| $A$ | 提取函数 | 从示例里总结规律的过程 |
| $\theta(S)$ | 任务向量 | 被压缩后的规则表示 |
| $f$ | 应用函数 | 拿着规则去处理查询的过程 |
| $y$ | 输出 | 模型最终答案 |

于是可以把 ICL 写成：

$$
\theta(S)=A(S),\qquad y=f(x;\theta(S))
$$

“任务向量存在”和“不存在”时，行为差异可以这样理解：

| 情况 | 模型是否需要反复依赖完整 $S$ | 推理结构 | 适合什么任务 |
|---|---|---|---|
| 存在较强任务向量 | 不必强依赖 | 先压缩，再执行 | 规则稳定、示例短、可归纳 |
| 任务向量弱或不存在 | 需要持续参考 | 边看示例边匹配 | 规则不稳定、示例噪声大、要求精确模仿 |

边界也要说清楚。

第一，这个分析主要针对有明确分层表示的 Transformer 结构，因为“在哪一层形成任务表示”本身依赖层级计算。第二，示例数必须落在 context window 内，否则连原始示例都放不进去。第三，示例间要有可总结的共性。如果示例本身互相矛盾，或者任务要求逐条检索原文而不是抽象规则，那么压缩成一个向量的效果会变差。

一个横向理解方式是：$A$ 像把一堆题目总结成一句“这是 AND 任务”，$f$ 再用这句话去解新题。它的价值在于避免后续层反复读取大段上下文。

---

## 核心机制与推导

核心机制可以分成两段。

第一段是任务编码。编码的白话意思是：模型前几层到中间层主要在看“这些例子共同在教我什么”。第二段是任务解码。解码的白话意思是：模型后续层在已有规则之下，把查询映射成输出。

从论文视角，任务向量不是随便取一个 hidden state 就算。困难在于两点：

1. 前层原本可以同时看示例 $S$ 和查询 $x$，这样得到的向量可能混入当前查询信息，不够“任务无关”。
2. 后层原本也可以直接看见示例 $S$，这样无法证明它真的只靠任务向量工作。

因此 Hendel 等人采用了“分离 $A$ 与 $f$”的做法。典型流程是：

1. 用示例集 $S$ 加一个 dummy query $x'$ 跑前 $L$ 层。
2. 取某个分隔位置或箭头位置的中间表示作为 $\theta(S)$。
3. 再单独输入真实查询 $x$，但在第 $L$ 层把该位置表示替换或注入为 $\theta(S)$。
4. 若性能仍接近原 ICL，就说明后层确实可近似视为 $f(x;\theta)$。

写成信号流就是：

$$
S \xrightarrow{A_{\le L}} \theta(S),\qquad
(x,\theta(S)) \xrightarrow{f_{>L}} y
$$

后续工作进一步观察到，任务可分性常在中间层最强。例如一些模型在第 13 到 15 层附近，任务解码性或可分性最高，然后后层再把这个表示“消费”为具体答案。这里的“消费”意思是：后层把抽象任务信息转成输出 token 分布，所以单纯线性读取任务类别的能力会下降。

正负干预实验是最关键的因果证据。因果证据的白话意思是：不是“相关”，而是“换这个向量，结果就跟着变”。

以 bitwise arithmetic 为例：

- 正干预：向后层注入正确任务向量，例如 AND 的向量，模型在 `1 AND 0` 上更容易输出 `0`。
- 负干预：注入错误任务向量，例如 OR 或随机向量，模型精度下降。

这说明任务向量确实携带了规则，而不只是一个“顺手留下的缓存”。

可以把它画成一个简图：

$$
\text{示例集 }S \to \text{中层表示} \to \theta
$$

$$
\text{查询 }x + \theta \to \text{后层计算} \to y
$$

更适合工程人员理解的伪码是：

1. 提取：在中层截获 separator token 的激活，记为 $\theta$
2. 保存：把 $\theta$ 缓存下来，作为该任务的紧凑表示
3. 注入：新查询推理时，把 $\theta$ 写回同一层或同一残差流位置

---

## 代码实现

下面给一个可运行的玩具实现。它不是完整 Transformer，而是把“示例压缩成任务向量，再用任务向量解题”的结构最小化实现出来。这里把布尔运算任务编码成向量，向量的每一维代表一种规则强度。

```python
from typing import List, Tuple

Example = Tuple[Tuple[int, int], int]

TASK_VECS = {
    "AND": (1.0, 0.0, 0.0),
    "OR":  (0.0, 1.0, 0.0),
    "XOR": (0.0, 0.0, 1.0),
}

def infer_task_vector(S: List[Example]):
    # A(S): 从示例拟合最匹配的规则卡片
    scores = {}
    for name in TASK_VECS:
        correct = 0
        for (a, b), y in S:
            pred = apply_with_name((a, b), name)
            correct += int(pred == y)
        scores[name] = correct
    best = max(scores, key=scores.get)
    return TASK_VECS[best], best

def apply_with_name(x: Tuple[int, int], name: str) -> int:
    a, b = x
    if name == "AND":
        return a & b
    if name == "OR":
        return a | b
    if name == "XOR":
        return a ^ b
    raise ValueError(name)

def apply_with_vector(x: Tuple[int, int], theta):
    # f(x; theta): 用任务向量执行规则
    idx = max(range(len(theta)), key=lambda i: theta[i])
    name = ["AND", "OR", "XOR"][idx]
    return apply_with_name(x, name)

def inject_vector_and_predict(x, theta):
    # 模拟“在后层注入任务向量”
    return apply_with_vector(x, theta)

# 玩具例子
S = [((0, 0), 0), ((1, 1), 1)]
theta, task_name = infer_task_vector(S)

assert task_name == "AND"
assert inject_vector_and_predict((1, 0), theta) == 0

# 负干预：注入错误规则卡片
bad_theta = TASK_VECS["XOR"]
assert inject_vector_and_predict((1, 0), bad_theta) == 1
```

这个玩具例子只保留了机制骨架：先提取任务向量，再用它处理新输入。真实模型里，$\theta$ 来自 hidden state 或 attention head 输出，而不是人工定义的三维向量。

如果在真实 Transformer 里实现，通常用 hook 截获中间层激活。思路如下：

```python
# 伪代码：真实工程通常基于 PyTorch hooks
cache = {}

def extract_hook(layer_id, token_pos):
    def fn(module, inp, out):
        cache["theta"] = out[:, token_pos, :].detach().clone()
    return fn

def inject_hook(layer_id, token_pos, alpha=1.0):
    def fn(module, inp, out):
        out = out.clone()
        out[:, token_pos, :] = (1 - alpha) * out[:, token_pos, :] + alpha * cache["theta"]
        return out
    return fn

# 步骤1：用 S + dummy query 跑到第 L 层，extract theta
# 步骤2：缓存 theta
# 步骤3：对真实查询 x 前向，在第 L 层 inject theta
# 步骤4：读最终输出
```

不同注入方式的差异如下：

| 注入方式 | 做法 | 优点 | 适用场景 |
|---|---|---|---|
| 直接替换 | 用 $\theta$ 覆盖某位置表示 | 因果性最清楚 | 机制验证、论文复现 |
| 残差相加 | $h \leftarrow h + \alpha\theta$ | 稳定、实现简单 | 线上实验、软干预 |
| 拼接后投影 | $[h;\theta]W$ | 表达力更强 | 有额外投影层时 |
| 条件层调制 | 用 $\theta$ 控制门控或缩放 | 可塑性高 | 专门设计的适配器结构 |

工程里还会做 low-rank projection。低秩投影的白话意思是：先把高维向量压到更小子空间，减少噪声和存储成本。另一个常见增强是 inner optimization，即从多个示例或多个 dummy query 提取多个候选状态向量，再做平均或迭代优化，让最终向量更稳定。

---

## 工程权衡与常见坑

任务向量方法很有吸引力，但落地时最常见的坑不是“提不出来”，而是“提出来不稳”。

第一个坑是示例数量饱和。饱和的白话意思是：继续加例子，信息增量已经很小，甚至新例子会把已有规律搅乱。对一些任务，模型在达到最大性能的约 95% 后，再塞更多示例并不会继续稳定提升；如果示例分布不均、格式不一致，反而会让任务向量变得更模糊。

第二个坑是 dummy query 噪声。噪声的白话意思是：你本想提取“任务规则”，结果混进了“这个查询长什么样”的无关信息。Hendel 的分离方法本身就依赖 dummy query，而后续工作指出，state/task/function vectors 的表现会明显受 dummy query 选择影响，因此需要优化或平均化。

| 常见坑 | 表现 | 监控指标 | 缓解策略 |
|---|---|---|---|
| 示例饱和 | 加更多样例后收益变小甚至抖动 | 准确率-样本数曲线、任务向量方差 | 控制示例数，优先去重与覆盖关键模式 |
| dummy query 噪声 | 换个占位查询，性能大幅变化 | 不同 dummy query 下的性能方差 | inner optimization、向量平均、固定模板 |

对新手，一个直观理解是：不是例子越多越好。过多例子会把“规则卡片”往不同方向拉，最后卡片上写的不是清晰规则，而是一团平均后的模糊提示。

inner optimization 的核心思想可以写成一个小过程。它把任务向量当作“测试时可继续打磨的状态”，通过多次候选提取或小步优化，减少偶然噪声的影响。简化地看，相当于：

$$
\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{L}_{\text{proxy}}(\theta)
$$

这里 $\mathcal{L}_{\text{proxy}}$ 不是重新训练大模型参数，而是一个代理损失，用来让当前状态向量更一致地解释示例。很多实现也不显式做梯度下降，而是做多向量平均、加权平均或 momentum 聚合，本质上都在追求更稳定的任务表示。

真实工程例子可以想成：做客服意图分类 few-shot 提示时，如果你把十几个风格差异很大的示例全部塞进提示，模型中层形成的“规则卡片”可能既包含标签规则，也混入大量措辞习惯，导致对新查询泛化变差。此时删掉冗余示例、统一模板、固定 dummy query，常常比继续堆例子更有效。

---

## 替代方案与适用边界

任务向量不是唯一方案。它适合“任务稳定、需要压缩、希望复用”的场景，但并不总是优于传统 ICL。

先看三种方法的对比：

| 方法 | 核心表示 | 推理时是否保留完整示例 | 更新频率 | 适用情形 |
|---|---|---|---|---|
| 传统 ICL | 完整上下文 | 是 | 每次都变 | 示例持续变化、需精确模仿上下文 |
| 任务向量 | 中层压缩向量 $\theta$ | 可不保留 | 任务变时更新 | 规则稳定、需节省上下文、做机制干预 |
| Function Vector | 功能方向 $\phi$ | 通常不保留 | 可组合、可迁移 | 希望跨上下文触发某类功能 |

传统 ICL 的优点是直接。模型每次都“再读一遍例子”，因此当你需要它模仿具体表述、依赖最新示例、或者任务本身没有稳定抽象规则时，这种方式反而更合适。

任务向量更像“先记住规则卡片，再反复使用”。它的优势在于压缩和复用，尤其当上下文预算紧张时很有价值。

Todd 等人的 Function Vectors 与任务向量相关，但不完全一样。function vector 可以看作模型内部某种“功能方向”，通常通过注意力头的因果中介分析提取，然后在中层注入以触发特定输入输出映射。若写成抽象形式，可以记为：

$$
y = g(x; \phi)
$$

其中 $\phi$ 是功能向量。和任务向量相比，function vector 更强调“功能电路”与“可组合方向”，而任务向量更强调“从示例集压缩出的任务状态”。

一个新手对照可以这样记：

- 传统 ICL：每次考试都把例题本带进去
- 任务向量：先把例题总结成规则卡片，再带卡片进去
- function vector：直接拿一个功能开关，打开后触发某类行为

适用边界也很明确：

- 如果任务长期稳定，任务向量更有价值。
- 如果上下文每次变化很大，传统 ICL 更稳妥。
- 如果你关心可组合的功能控制、跨场景迁移或机制解释，function vectors 值得优先考虑。

---

## 参考资料

1. Roee Hendel, Mor Geva, Amir Globerson. *In-Context Learning Creates Task Vectors*. EMNLP Findings 2023. 重点：提出 $T([S,x])=f(x;A(S))$ 视角，并通过中层提取与注入验证任务向量存在。链接：https://aclanthology.org/2023.findings-emnlp.624/

2. Seungwook Han, Jinyeop Song, Jeff Gore, Pulkit Agrawal. *Emergence and Effectiveness of Task Vectors in In-Context Learning: An Encoder Decoder Perspective*. 2024/2025 arXiv 版本。重点：分析任务编码与任务解码的分层形成，讨论中层任务可分性与前层微调收益。链接：https://arxiv.org/abs/2412.12276

3. Deep Paper 解读：*Decoding the Magic: How LLMs Build 'Task Vectors' for In-Context Learning*. 重点：用更直观方式解释中层任务向量、正负干预和层级分工。链接：https://deep-paper.org/en/paper/2412.12276/

4. Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron Wallace, David Bau. *Function Vectors in Large Language Models*. ICLR 2024. 重点：提出 function vector，强调中层可提取、可因果干预、可一定程度组合。链接：https://openreview.net/forum?id=AwyxtyMwaG

5. Dongfang Li, Zhenyu Liu, Xinshuo Hu, Zetian Sun, Baotian Hu, Min Zhang. *In-Context Learning State Vector with Inner and Momentum Optimization*. NeurIPS 2024. 重点：说明 task/function/state vectors 会受 dummy query 与示例噪声影响，并提出 inner optimization 提升稳定性。链接：https://papers.nips.cc/paper_files/paper/2024/hash/0ed52d7f6f641f228405d48a611e0684-Abstract-Conference.html

建议阅读顺序：

1. 先读 Hendel et al.，建立“任务向量 = 压缩规则表示”的基本框架。
2. 再读 Han et al. 或 Deep Paper，理解中层形成、任务可分性和前后层分工。
3. 最后读 Todd et al. 与 Li et al.，把视角扩展到 function vector 和 inner optimization。
