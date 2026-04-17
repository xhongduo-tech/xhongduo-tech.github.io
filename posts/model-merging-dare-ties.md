## 核心结论

模型合并的核心不是“把两个模型平均一下”，而是把每个微调模型相对基模型的改变量提取出来，再决定哪些改变量值得保留、哪些会互相打架。这里的“改变量”通常叫任务向量，白话解释就是“某个任务把基模型往哪个方向推了一步”。

最基础的写法是 Task Arithmetic：

$$
\tau_i = \theta_i - \theta_0,\qquad
\theta_{\text{merge}} = \theta_0 + \sum_i w_i \tau_i
$$

其中 $\theta_0$ 是基模型，$\theta_i$ 是第 $i$ 个微调模型，$w_i$ 是该模型在合并时的权重。它简单、便宜、好实现，但一旦多个模型在同一参数上给出相反方向，就容易互相抵消或放大噪声。

TIES 的价值在于先处理冲突，再做叠加。它的三步是 Trim、Elect、Disjoint Merge：

1. Trim：删掉幅度很小的改动。
2. Elect：对每个参数先决定“最终应该朝正还是朝负”。
3. Disjoint Merge：只合并那些方向和最终共识一致的改动。

DARE 的价值在于先稀疏化，再重缩放。它会随机丢掉大量 delta 参数，也就是任务向量中的很多元素，然后把保留下来的部分按 $1/(1-q)$ 放大，使整体期望不至于塌掉。直观上，它像是在“先减少冲突面，再把真正留下的信号补回来”。

工程上可以这样记：

| 方法 | 核心动作 | 是否处理符号冲突 | 是否显式稀疏 | 适合场景 |
| --- | --- | --- | --- | --- |
| Task Arithmetic | 直接加权叠加任务向量 | 否 | 否 | 两三个相近任务的小规模合并 |
| TIES | Trim + 符号共识 + 一致项合并 | 是 | 是，按幅度裁剪 | 多模型、冲突较明显 |
| DARE+TIES | 随机稀疏 + 重缩放 + TIES | 是 | 是，按随机掩码稀疏 | 冲突更多、模型数更多 |
| Passthrough / Franken | 直接拼接层或层段 | 不直接处理 | 不属于参数稀疏 | 探索性实验，不追求稳定首选 |

如果你只记住一句话：Task Arithmetic 是“直接相加”，TIES 是“先达成方向共识再相加”，DARE+TIES 是“先减冲突面，再做方向共识再相加”。

---

## 问题定义与边界

模型合并讨论的是“已有多个同架构、同基座或至少高度同源的模型，能否不再训练、只通过参数操作，把多种能力放进一个模型里”。这里的“同源”白话说就是它们最好来自同一个底座，否则参数位置虽然同名，语义也可能并不对齐。

问题边界要先说清：

| 维度 | 需要满足的条件 | 不满足时的风险 |
| --- | --- | --- |
| 架构 | 层数、隐藏维度、张量形状兼容 | 直接无法逐参数合并 |
| 基模型 | 最好来自同一 base model | 任务向量失去可比性 |
| tokenizer | 最好一致或明确指定来源 | 生成乱码、词表错位 |
| 任务关系 | 能力最好互补而不是对冲 | 合并后两边都变差 |
| 评测 | 必须对每项能力单独验证 | 看似成功，实际主能力退化 |

玩具例子先帮助建立直觉。假设一个基模型经过两次微调：

- 模型 A 强化了数学推理
- 模型 B 强化了代码补全

如果 A 在某个参数上建议“增大”，B 建议“减小”，这就是符号冲突。Task Arithmetic 会把它们硬加在一起；TIES 会先判断谁更重要；DARE 会进一步减少同时进入竞争的参数数量。

真实工程例子更接近 mergekit 的使用方式：你手里有同一个 7B 基座上训练出来的三个模型，分别偏向数学、代码、对话。目标不是得到三个模型的平均人格，而是尽量保留三个方向的长处，同时不让数学把对话风格打坏，也不让代码模型把普通问答变得生硬。这个时候，`density` 控制保留多少改动，`weight` 控制每个专家的发言权，`dare_ties` 控制冲突处理策略。

这里还有一个常见误区：模型合并不是免费的能力相加。它只是把已有参数变化重新组织，不能无中生有，也不能替代严格的多任务训练。

---

## 核心机制与推导

先看最小公式。对于第 $i$ 个模型的任务向量：

$$
\tau_i = \theta_i - \theta_0
$$

如果直接做 Task Arithmetic：

$$
\theta = \theta_0 + \sum_i w_i \tau_i
$$

问题出在两个地方：

1. 很多参数变化极小，属于噪声或冗余。
2. 多个模型可能在同一位置给出相反符号。

TIES 就是为了解这两个问题。

### 1. Trim

Trim 的意思是按绝对值保留每个任务向量中最重要的那部分参数，其余置零。设保留密度为 $\rho$，那么每个 $\tau_i$ 会变成稀疏向量 $\hat{\tau}_i$。白话说，就是“先别让小改动进场”。

### 2. Elect

Elect 要对每个参数位置选一个最终符号。常见思路是看多个模型在该位置上的符号及其幅度，得到一个多数或强度主导的方向。白话说，就是“先决定这颗螺丝总体该往左拧还是往右拧”。

### 3. Disjoint Merge

有了最终符号后，只保留与该符号一致的非零项，再按权重合并。也就是“方向一致的进来，方向相反的出去”。

看一个玩具例子。设基模型：

$$
\theta_0=[0,0]
$$

两个任务向量：

$$
\tau_A=[0.4,-0.2],\qquad \tau_B=[-0.1,0.3]
$$

如果 `density=0.5`，对每个任务向量都只保留绝对值更大的那个元素，则：

- A 保留 $0.4$，丢掉 $-0.2$
- B 保留 $0.3$，丢掉 $-0.1$

Trim 后得到：

$$
\hat{\tau}_A=[0.4,0],\qquad \hat{\tau}_B=[0,0.3]
$$

Elect 时：

- 第 1 个参数只有正号存活，选正
- 第 2 个参数只有正号存活，选正

Disjoint Merge 后，两个位置都没有冲突，于是直接加回：

$$
\theta=[0,0]+[0.4,0]+[0,0.3]=[0.4,0.3]
$$

DARE 再进一步。设随机丢弃概率为 $q$，掩码为 $Z_i \sim \text{Bernoulli}(q)$。保留的参数乘以 $1/(1-q)$ 做重缩放：

$$
\tilde{\tau}_i=\frac{(1-Z_i)\odot\tau_i}{1-q}
$$

它的直觉是：虽然很多元素被丢掉了，但保留下来的元素被适当放大后，整体期望仍接近原任务向量。比如 $q=0.5$ 时，平均只保留一半元素，但保留项乘以 2。这样既降低同位冲突，又不至于把整体能力直接砍半。

很多资料会把 DARE 写成 TIES 的插件，这是准确的。它不是新的“合并框架”，而是先对任务向量做随机稀疏和重缩放，然后再接 Task Arithmetic 或 TIES。于是有：

- `dare_linear`：DARE + Task Arithmetic
- `dare_ties`：DARE + TIES

---

## 代码实现

下面先给一个最小可运行的 Python 版本，用玩具向量模拟 `trim + elect + disjoint merge`。它不是生产实现，但足够帮助理解流程。

```python
import math

def trim(vec, density):
    k = max(1, math.ceil(len(vec) * density))
    idx = sorted(range(len(vec)), key=lambda i: abs(vec[i]), reverse=True)[:k]
    keep = set(idx)
    return [v if i in keep else 0.0 for i, v in enumerate(vec)]

def elect_sign(vectors):
    signs = []
    for values in zip(*vectors):
        score = sum(v for v in values if v != 0.0)
        if score > 0:
            signs.append(1.0)
        elif score < 0:
            signs.append(-1.0)
        else:
            signs.append(0.0)
    return signs

def disjoint_merge(base, vectors, weight=1.0):
    merged = base[:]
    signs = elect_sign(vectors)
    for i in range(len(base)):
        contribs = [v[i] for v in vectors if v[i] != 0.0 and (v[i] > 0) == (signs[i] > 0)]
        merged[i] += weight * sum(contribs)
    return merged

base = [0.0, 0.0]
tau_a = [0.4, -0.2]
tau_b = [-0.1, 0.3]

trimmed = [trim(tau_a, 0.5), trim(tau_b, 0.5)]
merged = disjoint_merge(base, trimmed)

assert trimmed == [[0.4, 0.0], [0.0, 0.3]]
assert merged == [0.4, 0.3]
print(merged)
```

真实工程里通常不会自己手写，而是直接用 mergekit。一个简化配置如下：

```yaml
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: mistralai/Mistral-7B-v0.1
  - model: org/math-model
    parameters:
      weight: 0.4
      density: 0.53
  - model: org/code-model
    parameters:
      weight: 0.3
      density: 0.53
  - model: org/chat-model
    parameters:
      weight: 0.3
      density: 0.53
parameters:
  normalize: true
  int8_mask: true
dtype: bfloat16
tokenizer_source: base
```

这个配置表达的是：

- 以 `base_model` 作为 $\theta_0$
- 其余模型先转成任务向量
- 对每个任务向量做 DARE 稀疏化
- 再进入 TIES 的符号共识流程
- 最后按 `weight` 叠加回基模型

如果要做 Franken-merging，也就是层级拼接，mergekit 常见写法是 `merge_method: passthrough` 或基于 `slices` 指定层段来源。它不是“同一参数位置的算术合并”，而是“这个层段来自模型 A，那个层段来自模型 B”。这类方法实验味更重，收益和风险都更不稳定。

真实工程例子可以是：保留前 8 层来自通用对话模型，中间若干层来自代码模型，最后几层继续保留对话模型，再对输出投影做轻量缩放。它有时能产生意外好的组合，但也很容易出现风格割裂、推理链不连续、特定 benchmark 大幅波动的问题。

---

## 工程权衡与常见坑

最常见的坑不是“命令跑不起来”，而是“合并成功了，但能力 quietly 退化了”。

第一类坑是 `density` 过高。很多新手直觉上会觉得“保留越多越安全”，实际上在多模型合并里往往相反。`density=0.9` 意味着几乎所有改动都进场，符号冲突会显著增加，TIES 的 Elect 也会更难做出干净决策。经验上通常从较低密度开始扫参更稳。

第二类坑是 DARE 只做 drop，不做 rescale。若丢掉 50% 甚至 90% 的 delta，却不做 $1/(1-q)$ 重缩放，能力退化几乎是必然的。因为你不是“去掉噪声”那么简单，而是在整体缩小任务向量。

第三类坑是 tokenizer 和 base model 不一致。模型合并成功只说明张量对上了，不说明语言空间对上了。真实表现可能是语气变怪、重复、乱码或格式异常。

第四类坑是只看综合分，不看单项回退。比如数学、代码、聊天三者合并后平均分上升，但数学大题能力下降 8 分。如果你的主要业务是数学问答，这次合并就是失败。

可以用一个简单 checklist 管理风险：

| 检查项 | 建议 | 失败后现象 |
| --- | --- | --- |
| density | 从 0.3 到 0.6 逐步试 | 冲突增多，性能不稳 |
| weight | 先均匀，再偏向主任务 | 某一能力被淹没 |
| DARE rescale | 必开 | 整体能力缩水 |
| tokenizer_source | 明确指定 `base` 或 `union` | 输出异常 |
| eval | 每个子任务单独测 | 平均分掩盖主能力退化 |
| rollback | 保留所有配置与评测结果 | 无法定位是哪次 merge 变坏 |

Passthrough 还要额外注意：层拼接不是连续优化过程，中间层一旦断裂，模型可能看起来“能说话”，但内部表征已经不稳定。所以它更适合研究和试验，不适合作为第一选择。

---

## 替代方案与适用边界

如果只合并两个相近任务、基座完全一致、冲突预期较低，Task Arithmetic 仍然值得先试。它便宜、快、解释简单，还是很多更复杂方法的起点。

如果是多个专家模型，尤其是数学、代码、通用问答这类差异较大的能力组合，TIES 通常比直接 Task Arithmetic 更稳，因为它明确处理了冗余项和符号冲突。

如果模型更多、冲突更明显，或者你观察到 TIES 仍有较大干扰，DARE+TIES 更值得优先尝试。它的关键收益不是“更花哨”，而是先随机缩小交战范围，再做符号协调。

Passthrough / Franken 更像结构实验，而不是标准答案。它适合这些场景：

- 你想探索某些层段是否承载特定能力
- 你愿意接受较高失败率
- 你有完善评测和回滚流程

对比可以总结为：

| 方案 | 优点 | 缺点 | 更适合谁 |
| --- | --- | --- | --- |
| Task Arithmetic | 简单、快、可解释 | 不处理冲突 | 两三个相近任务 |
| TIES | 冲突控制更强 | 需要调密度 | 多任务标准合并 |
| DARE+TIES | 更强稀疏化，常更稳 | 多一个随机因素 | 多模型、高冲突场景 |
| Passthrough | 能探索非常规组合 | 稳定性差，验证成本高 | 研究型实验 |

可以给一个决策规则：

- 合并 2 个 1B 级别、同基座、同方向任务，先试 Task Arithmetic。
- 合并 3 到 5 个 7B 级别专家模型，优先试 TIES。
- 合并更多专家模型，或已知冲突明显，优先试 DARE+TIES。
- 只有在你明确想做层级实验时，再试 Passthrough。

---

## 参考资料

- TIES-Merging 论文：<https://papers.nips.cc/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf>
- MergeKit 官方仓库与方法概览：<https://github.com/arcee-ai/mergekit>
- NVIDIA 关于 LLM 模型合并的技术综述：<https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/>
- Hugging Face PEFT 的 TIES / DARE 合并工具文档：<https://huggingface.co/docs/peft/en/package_reference/merge_utils>
- MergeKit 实践文章，含 `dare_ties` 配置示例：<https://www.theeasternarchivist.com/a-guide-to-mergekit-and-making-your-own-large-language-models/>
