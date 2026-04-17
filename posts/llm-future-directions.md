## 核心结论

未来几年，大模型研究的主线之一，不再只是“把静态参数做大”，而是让模型在推理时继续学习、在长上下文中维持稳定状态、并且在新增知识后尽量不遗忘旧能力。Test-Time Training，简称 TTT，指模型在推理阶段对当前输入做短暂梯度更新；World Model 指模型内部是否真的维护了一个可随动作推进的状态；组合泛化关注模型在没见过的新组合上能否按规则处理；长文本理解关注模型在超长输入里能否持续抓住关键证据；持续学习关注新知识写入后旧能力是否被冲掉。

这几条线索其实在问同一件事：LLM 现在更像“高容量的条件概率机器”，还是正在逼近“可在线更新、可维护内部状态、可稳定推演”的系统。TTT 和 TTT-E2E 给出的方向是，把推理本身变成一次局部学习过程，而不是一次纯前向计算。它的代表公式是：

$$
W_t = W_{t-1}-\eta\nabla_W \mathcal L(x_t;W_{t-1})
$$

这里 $W_t$ 可以理解为“当前时刻的快权重”，也就是临时可变的内部记忆。对新手可以这样理解：每条查询都做一次“现场调参”，模型在回答当前问题前，先用当前输入给自己做一次很短的自监督训练。

但另一面也要看清。组合泛化、RULER 一类长上下文基准、以及持续学习中的灾难性遗忘，都在说明今天的 LLM 还远没达到“稳定世界模型”的程度。它能在很多任务上看起来像理解了，但一旦任务要求精确顺序、长程状态跟踪、反事实推演或连续多轮知识更新，边界就会暴露。

把这几个方向连在一起看，结论更明确：

| 方向 | 它想解决什么 | 当前主要障碍 |
|---|---|---|
| TTT / TTT-E2E | 让模型对当前输入即时适配 | 推理成本、更新稳定性、工程复杂度 |
| World Models | 让模型维护可推演的内部状态 | 文本相关性不等于状态一致性 |
| 组合泛化 | 让模型按规则处理新组合 | 训练分布外结构约束容易失效 |
| 长文本理解 | 让模型在超长输入中持续对准证据 | 注意力漂移、聚合失败、多跳退化 |
| 持续学习 | 让模型持续写入新知识而不忘旧能力 | 梯度冲突、参数覆盖、回放成本 |

所以，未来方向并不是单点突破，而是三类能力同时改进：

1. 在线适配能力：模型能否在输入到来时快速修正自己。
2. 状态维护能力：模型能否长期保持一致、可追踪的内部表示。
3. 稳定写入能力：模型能否新增知识而不破坏已有知识。

---

## 问题定义与边界

这些方向容易混在一起，先把边界分清。否则会出现一种常见误判：把“模型回答得像懂了”当成“模型真的维护了状态”。

| 主题 | 定义 | 主要指标或形式 | 真实边界 |
|---|---|---|---|
| TTT / TTT-E2E | 推理时继续用当前输入做短暂更新 | 快权重更新、next-token 自监督损失 | 计算更重，且更新是否稳定是核心问题 |
| World Models | 是否存在可解释状态转移 $s_{t+1}=f(s_t,a_t)$ | 状态、动作、观测、反事实一致性 | 仅会续写文本不等于有世界模型 |
| 组合泛化 | 训练没见过的新组合还能按规则处理 | Ordered Coverage, novel coverage | 模型可能记住搭配，但不懂结构约束 |
| 长文本理解 | 超长输入中是否持续抓住关键证据 | 检索准确率、multi-hop 表现、drift | 标称 128K 不等于有效理解 128K |
| 持续学习 | 新知识写入后旧能力是否被保留 | $\Delta=Acc_{\text{before}}-Acc_{\text{after}}$ | 新旧任务梯度冲突会导致遗忘 |

先看组合泛化。一个最小例子是：指令要求输出“苹果-书”，如果模型输出“书-苹果”，那不是小错误，而是结构性错误。因为问题并不是“两个词都出现没”，而是“是否理解了顺序约束”。这类能力常用 Ordered Coverage 衡量：

$$
OC=\frac{\#\text{按要求顺序生成的概念}}{\#\text{要求概念数}}
$$

如果要求两个概念且顺序固定，只答对一个位置，那么 $OC=0.5$。这个指标的意义在于，它能把“词汇命中”与“结构满足”分开。很多模型在开放生成里能写出相关概念，但一旦要求精确顺序、层级或依赖关系，错误会成批出现。

再看 world model。LLM 的基础目标通常是：

$$
P(w_{t+1}\mid w_{\le t})
$$

它优化的是“下一个 token 的条件概率”。而 world model 关心的是状态如何演化：

$$
s_{t+1}=f(s_t,a_t), \quad o_t=g(s_t)
$$

这里 $s_t$ 是系统状态，白话说就是“环境当前真实情况”；$a_t$ 是动作；$o_t$ 是能观测到的信息。两者差异在于：

| 视角 | 关注对象 | 典型问题 |
|---|---|---|
| 语言建模 | 文本续写是否像训练分布 | 下一句话像不像人写的 |
| 世界建模 | 动作后状态是否一致 | 门已经打开后还能不能再次“打开” |
| 规划/控制 | 状态转移是否支持后续决策 | 前置条件不满足时动作是否应被拒绝 |

举一个新手容易理解的例子。假设环境里有一只空杯子、一壶水和一个冰箱：

1. 初始状态：杯子是空的，水壶在桌上，冰箱门关着。
2. 动作一：把水倒进杯子。
3. 动作二：把杯子放进冰箱。
4. 问题：此时“杯子是否还在桌上”？

如果模型只是做文本相似性匹配，它可能抓住“杯子”和“桌上”在前文共现过，就误答“在桌上”。但如果它内部维护了状态转移，就应该更新为：杯子进入冰箱，不再在桌上。world model 讨论的正是这种“动作改变状态”的一致性，而不是文本表面相关性。

长文本理解也有边界。很多模型能在“needle in a haystack”任务里找到单个事实，但这并不等于它能处理真实长文档。真实任务往往同时要求：

| 长文本子任务 | 难点 |
|---|---|
| Retrieval | 从很长输入中定位证据 |
| Tracing | 跟踪某个实体或变量的多次变化 |
| Aggregation | 把分散在多处的证据汇总 |
| Multi-hop | 先用证据 A 推出中间结论，再结合证据 B |
| Conflict resolution | 处理正文、附录、修订条款之间的冲突 |

RULER 的意义就在这里。它测的不只是“能不能找到一根针”，而是当输入变长、推理链变多、干扰项变强时，模型是否仍然稳定。

持续学习的边界则更直接。常见遗忘量定义为：

$$
\Delta=Acc_{\text{before}}-Acc_{\text{after}}
$$

如果模型学了新法规后，旧法规问答掉分明显，那么模型只是“写入了新参数”，并没有真正完成知识积累。对工程系统来说，这意味着上线新版本可能在你没注意到的旧场景上退化。

---

## 核心机制与推导

TTT 的核心机制是把“记忆”从显式缓存，部分转成参数更新。传统 Transformer 的长上下文主要依赖 KV cache，也就是把前文的键值对存下来，后续继续查。TTT 提供的是另一条路：让一部分内部参数在每个输入块后更新一次，等于把近期上下文压进参数。

最基本形式是：

$$
W_t = W_{t-1}-\eta\nabla_W \mathcal L(x_t;W_{t-1})
$$

这里 $\eta$ 是学习率，白话说就是每次更新迈多大一步；$\mathcal L$ 是当前输入上的自监督损失。玩具例子如下：若初始快权重 $W_0=1.0$，学习率 $\eta=0.5$，当前损失梯度为 $0.2$，则

$$
W_1=1.0-0.5\times 0.2=0.9
$$

下一次前向就用 $W_1=0.9$。这个例子虽然简单，但它准确表达了 TTT 的本质：模型不是只读上下文，而是在边读边改自己的内部参数。

如果把公式拆开，新手更容易看懂。设一个最小模型：

$$
\hat y = Wx
$$

目标是让预测接近伪标签 $y^\star$，损失取平方误差：

$$
\mathcal L = \frac{1}{2}(\hat y-y^\star)^2
$$

则对参数 $W$ 的梯度为：

$$
\nabla_W \mathcal L = (\hat y-y^\star)x
$$

代回更新式：

$$
W_t = W_{t-1}-\eta(\hat y-y^\star)x
$$

这说明三件事：

1. 预测偏差越大，更新越大。
2. 输入幅度越大，更新越大。
3. 学习率越大，适配越快，但也越容易不稳定。

所以 TTT 的难点从来不只是“能不能更新”，而是“更新后会不会把模型带偏”。如果每个窗口都在线更新，而输入本身又带噪声、带对抗样本或带格式异常，那么快权重可能迅速漂移。

TTT-E2E 把这个思想推进到长上下文建模。它把长文本分成窗口 $c_t$，并在推理时继续用 next-token loss 做更新：

$$
\theta_{t+1}=\theta_t-\alpha\nabla_{\theta_t}\mathcal L_{\text{next-token}}(c_t;\theta_t)
$$

这里 $\theta_t$ 是当前可适配参数，$\alpha$ 是更新步长。直白说，每个窗口不仅被“读到”，还被“学进去”。这样做的目标，是把原本需要越来越大 KV cache 才能保留的信息，压进一组持续更新的参数中。研究里的表述通常是“权重即上下文压缩”。

把 KV cache 与 TTT 对比，会更清楚：

| 机制 | 信息存在哪里 | 优点 | 代价 |
|---|---|---|---|
| KV cache | 显式缓存前文 token 表示 | 不改参数，行为稳定 | 长上下文时显存和带宽压力大 |
| TTT | 写入快权重或适配层 | 能跨窗口压缩上下文 | 需要反向传播，推理更慢 |
| 检索增强 | 存在外部数据库 | 可控、可更新 | 依赖召回质量，不等于内部理解 |

这和 world model 的问题可以并排看。若模型真有内部世界状态，它不该只在表面文本上相关，而应在动作和状态变化上连贯。例如在厨房任务中，若当前状态是“锅已热”，动作是“加油”，那么后续更合理的动作是“翻炒”。这要求系统内部有某种状态转移结构，而不是只靠“食谱文本里这些词经常一起出现”。

一个更形式化的 world model 视角是：

$$
s_t = \phi(o_{\le t}, a_{<t})
$$

也就是模型把过去观测与动作编码为当前内部状态。然后用这个状态预测后续观测：

$$
\hat o_{t+1}=g(s_t,a_t)
$$

如果系统具备稳定的状态表示，那么以下三种测试应该同时成立：

| 测试 | 含义 |
|---|---|
| 一致性测试 | 同一状态下同一动作应得到相近结果 |
| 反事实测试 | 换动作后状态变化应可解释 |
| 可回溯测试 | 能说明哪一步动作导致了当前状态 |

长文本理解中的注意力漂移，也可以抽象成状态稳定性问题。一个常见度量写成：

$$
\text{Drift}(t)=\text{JS}(A_t,A_{t-1})
$$

其中 JS 指 Jensen-Shannon divergence，白话说就是“两个注意力分布差了多少”；$A_t$ 表示第 $t$ 个时刻的注意力分布。如果 drift 持续升高，说明模型在长文中对焦点的保持越来越差。RULER 这类基准的重要结论不是“模型能不能捞出一根针”，而是“随着上下文变长、任务变成多跳追踪或聚合后，性能会不会系统性退化”。

持续学习的核心公式更直接：

$$
\Delta=Acc_{\text{before}}-Acc_{\text{after}}
$$

这里 $\Delta$ 表示遗忘量，白话说就是“学了新东西后旧任务掉了多少分”。如果一个模型在加入新法规后，旧法规问答明显变差，那么它并不是真正完成了知识积累，只是发生了参数覆盖。

一个真实工程例子可以把这些问题串起来。假设企业合同分析系统要处理 100K 字以上的新客户协议：首先，客户的字段命名和条款结构可能与训练分布不同，这时 TTT 思路可以让模型在推理时快速适配新格式；其次，合同跨页引用、附录定义、例外条件会造成长程依赖，RULER 暴露的问题会直接出现；再次，如果系统每周都要引入新法规，又不能把旧法规判断能力打掉，就进入持续学习问题。也就是说，这些研究方向不是彼此独立，而是在工程里同时出现。

---

## 代码实现

下面给一个可运行的最小 Python 版本，不依赖深度学习框架，只模拟 TTT 的“测试时更新”、组合泛化的 Ordered Coverage、长文本中的简单 drift，以及持续学习中的遗忘量。它不是工业实现，但机制是对的，直接保存为 `ttt_world_model_demo.py` 后可运行。

```python
from math import log2


def test_time_step(x_t, w_prev, eta=0.5):
    """
    最小 TTT 例子：
    模型 y = w * x
    伪标签设为 y* = 2 * x
    损失 L = 0.5 * (y - y*)^2

    返回：
    - update 前预测
    - update 后预测
    - 新权重
    - 梯度
    """
    pred_before = w_prev * x_t
    target = 2.0 * x_t
    grad = (pred_before - target) * x_t  # dL/dw
    w_new = w_prev - eta * grad
    pred_after = w_new * x_t
    return pred_before, pred_after, w_new, grad


def ordered_coverage(required, generated):
    if not required:
        raise ValueError("required must not be empty")
    hit = 0
    for i, token in enumerate(required):
        if i < len(generated) and generated[i] == token:
            hit += 1
    return hit / len(required)


def forgetting_delta(acc_before, acc_after):
    return acc_before - acc_after


def js_divergence(p, q, eps=1e-12):
    """
    Jensen-Shannon divergence for two discrete distributions.
    p 和 q 必须长度相同，且元素和为 1。
    """
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")

    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]

    def kl(a, b):
        total = 0.0
        for ai, bi in zip(a, b):
            ai = max(ai, eps)
            bi = max(bi, eps)
            total += ai * log2(ai / bi)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def normalize(xs):
    s = sum(xs)
    if s <= 0:
        raise ValueError("sum must be positive")
    return [x / s for x in xs]


def world_state_step(state, action):
    """
    一个极简 world model：
    state = {"cup": "table" or "fridge", "has_water": bool, "fridge_open": bool}
    """
    new_state = dict(state)

    if action == "open_fridge":
        new_state["fridge_open"] = True
    elif action == "close_fridge":
        new_state["fridge_open"] = False
    elif action == "pour_water":
        new_state["has_water"] = True
    elif action == "put_cup_in_fridge":
        if not new_state["fridge_open"]:
            raise ValueError("cannot put cup in fridge when fridge is closed")
        new_state["cup"] = "fridge"
    else:
        raise ValueError(f"unknown action: {action}")

    return new_state


def run_demo():
    print("== 1) TTT 现场更新 ==")
    w = 1.0
    for step, x in enumerate([1.0, 1.0, 1.0], start=1):
        pred_before, pred_after, w, grad = test_time_step(x_t=x, w_prev=w, eta=0.5)
        print(
            f"step={step}, pred_before={pred_before:.2f}, grad={grad:.2f}, "
            f"w_new={w:.2f}, pred_after={pred_after:.2f}"
        )

    print("\n== 2) 组合泛化 Ordered Coverage ==")
    required = ["apple", "book"]
    generated_ok = ["apple", "book"]
    generated_bad = ["book", "apple"]
    generated_half = ["apple", "pen"]
    print("OC(ok)   =", ordered_coverage(required, generated_ok))
    print("OC(bad)  =", ordered_coverage(required, generated_bad))
    print("OC(half) =", ordered_coverage(required, generated_half))

    print("\n== 3) 长文本 drift 的简化示意 ==")
    attn_t1 = normalize([7, 2, 1])
    attn_t2 = normalize([6, 3, 1])
    attn_t3 = normalize([2, 2, 6])
    print("drift(t2,t1) =", round(js_divergence(attn_t2, attn_t1), 4))
    print("drift(t3,t2) =", round(js_divergence(attn_t3, attn_t2), 4))

    print("\n== 4) 持续学习遗忘 ==")
    before = 0.92
    after = 0.88
    print("forgetting delta =", round(forgetting_delta(before, after), 4))

    print("\n== 5) 极简 world model ==")
    state = {"cup": "table", "has_water": False, "fridge_open": False}
    print("initial state:", state)

    for action in ["pour_water", "open_fridge", "put_cup_in_fridge", "close_fridge"]:
        state = world_state_step(state, action)
        print(f"after {action}:", state)

    assert state["cup"] == "fridge"
    assert state["has_water"] is True
    assert state["fridge_open"] is False


if __name__ == "__main__":
    run_demo()
```

运行后你会看到四类现象：

1. TTT 部分里，权重 `w` 会逐步向目标值靠近，说明“现场更新”确实改变了后续预测。
2. Ordered Coverage 会区分“概念出现了”和“顺序满足了”。
3. drift 会随着注意力焦点迁移而增大，说明长上下文里的“对焦点保持”是可测的。
4. world model 的例子里，动作会改变状态，后续判断依赖的是更新后的状态而不是表面共词。

如果把它翻译成真实推理循环，逻辑通常是：

1. 读取一个输入窗口。
2. 用当前快权重做前向。
3. 计算自监督损失，例如 next-token loss。
4. 对可适配参数求梯度。
5. 立刻更新快权重。
6. 用更新后的参数继续处理后续窗口或生成输出。

伪代码可以写成：

```python
def inference_with_ttt(chunks, theta):
    for chunk in chunks:
        logits = model(chunk, theta)
        loss = next_token_loss(logits, chunk)
        grads = autograd.grad(loss, theta_trainable_only)
        theta = update(theta, grads, lr=alpha)
    return theta
```

真实工程里通常不会直接更新全量参数，而是只更新小规模适配层，例如 LoRA。LoRA 可以理解为“只在低秩子空间里改参数”，白话说就是“只开一个小调节阀，不重写整台机器”。这样做的原因很直接：TTT 的难点不是公式，而是算力和内存。若对整模型做频繁更新，推理很快就会退化成慢训练。

另一个真实工程例子是“LLM + 显式世界模型”组合。做机器人或流程自动化时，前端 LLM 负责把自然语言解析成动作候选，后端状态机或因果模拟器负责验证“动作是否满足前置条件，执行后状态如何变化”。这样 LLM 不必独自承担全部因果推演，系统可验证性更强。

---

## 工程权衡与常见坑

TTT 看上去优雅，落地却不轻。第一个问题是二阶梯度支持。因为如果你要端到端训练“模型在测试时如何更新”，训练阶段往往会涉及“梯度的梯度”。这会直接碰到 kernel 支持、显存占用和带宽瓶颈。第二个问题是吞吐量。普通推理主要做前向，TTT 则把反向也带进来了，系统延迟模型完全变了。

| 常见坑 | 为什么发生 | 典型后果 | 规避策略 |
|---|---|---|---|
| TTT 内存带宽瓶颈 | 每步都有参数读写和梯度更新 | 吞吐急剧下降 | 大 chunk、滑动窗口并行、只更新小模块 |
| TTT-E2E 训练过慢 | 需要元学习初始化与高阶梯度 | 训练成本接近或超过主模型 | 自定义 kernel、低秩初始化、限制可训练参数 |
| 更新过度 | 学习率过大或窗口噪声太强 | 快权重漂移，生成不稳定 | 小步长、梯度裁剪、更新步数上限 |
| 把预测当因果 | 只看输出正确率，不测反事实 | 系统在新环境失效 | 加结构化 pre/post conditions 与环境验证 |
| 顺序控制被忽略 | 只看词是否出现，不看约束是否满足 | 代理执行顺序错乱 | 用 Ordered Coverage 一类指标单独验收 |
| 长文本只测检索 | 只做 needle 测试 | 真实 multi-hop 任务崩溃 | 增加 tracing、aggregation、drift 监控 |
| 持续学习直接微调 | 新旧任务梯度冲突 | 灾难性遗忘 | SSR、replay、LoRA、参数冻结 |

有一个很典型的误区：看到模型支持 128K 上下文，就默认它对 128K 输入有稳定理解。RULER 的价值就在这里。它说明“可输入长度”和“有效理解长度”不是一回事。很多模型在简单 needle 任务里表现很好，但任务一变成多跳追踪、聚合、带干扰问答，性能就开始掉。工程上如果不单独监控 drift，就会把“能塞进去”误当成“能理解”。

另一个常见坑出现在持续学习。很多团队会做增量微调，把最新知识集直接继续训练几轮。短期看新任务分数升了，但旧能力 silently 下降，直白说就是“悄悄遗忘”。当旧数据不能直接回放时，SSR 这类自合成回放就很有现实意义，因为它不依赖原始历史训练集完整可用。

把这些坑放到一个实际系统里看，会更容易判断是否该上复杂方案。以下是一个常见排查顺序：

| 现象 | 先怀疑什么 | 不要先做什么 |
|---|---|---|
| 新客户文档答不准 | 检索和格式适配是否失败 | 先全量微调整个模型 |
| 长文档跨页引用答错 | 证据追踪和聚合是否丢失 | 只看单点检索命中率 |
| 多步骤执行顺序错 | 顺序约束是否显式建模 | 把所有错误归因给提示词 |
| 新知识上线后旧任务退化 | 是否发生遗忘 | 继续叠加增量训练 |
| 机器人动作偶发不合法 | 状态机和前置条件是否缺失 | 假设 LLM 会自动学会因果 |

本质上，TTT、world model、组合泛化、长文本理解、持续学习五条线虽然研究对象不同，但工程失败模式高度相似：系统看起来能做，直到任务开始要求稳定性、一致性和可验证性。

---

## 替代方案与适用边界

TTT 不是唯一答案，而且很多场景不该直接上 TTT。判断标准很简单：你到底缺的是“在线适配”，还是“可控检索”，还是“显式状态约束”。

| 问题 | 可替代方案 | 适用边界 |
|---|---|---|
| 长上下文太慢 | Ring Attention、Squeezed Attention、局部检索 | 适合主要瓶颈在 KV cache 和注意力复杂度 |
| 输入分布轻度偏移 | retrieval + 轻量 rerank | 适合术语变化大但逻辑结构没变 |
| 因果要求强 | LLM interface + explicit simulator | 适合规划、安全、流程控制 |
| 顺序约束严格 | 状态机 / 规则校验器 | 适合操作序列和工作流执行 |
| 持续学习资源有限 | LoRA + SSR | 适合不能全量重训、又需保旧能力 |
| 长文本证据分散 | 局部摘要 + 证据链回填 | 适合合同、法规、日志分析 |

一个具体判断方式是这样的。若任务是企业知识库问答，主要问题是新客户文档格式变化、术语映射不一致，那么 retrieval 加轻量适配通常比 TTT 更稳，因为你需要的是“把对的证据找出来”，不是每次都在线改模型。相反，若任务是连续流式输入、上下文非常长、又不能无限扩 KV cache，那么 TTT-E2E 的思想就更有吸引力，因为它直接针对“把上下文写进权重”。

在 world model 争议上，也不必二选一。很多安全相关系统更合理的设计是：LLM 只做接口层，负责理解自然语言和生成候选动作；显式 world model 负责状态转移和结果验证。比如家庭自动化场景中，如果用户命令涉及“先关灯再开电视”，那么即使 LLM 生成顺序有误，后端状态机也可以拒绝非法步骤。这比假设 LLM 已经具备稳定组合泛化更现实。

持续学习同理。若知识更新频率高，但每次变更都不大，参数高效方法通常优于周期性全量重训。因为全量微调虽然容量大，但梯度干扰也最大；LoRA、冻结、回放等方法，本质是在控制“写入新知识时对旧参数空间的扰动”。

还可以把方案选择压缩成一个简单决策表：

| 如果你的主要问题是 | 优先方案 | 原因 |
|---|---|---|
| 找不到正确证据 | 检索增强 | 问题在召回，不在在线学习 |
| 找到证据但顺序总错 | 状态机/规则约束 | 问题在结构控制，不在参数容量 |
| 输入特别长且连续到来 | TTT / 压缩型长上下文方案 | 问题在上下文承载方式 |
| 动作执行必须可验证 | 显式 world model / simulator | 问题在状态转移一致性 |
| 新知识频繁上线 | PEFT + replay/SSR | 问题在知识写入与保留平衡 |

所以，TTT 不是“更先进就该优先上”的方案，而是“当记忆、适配和长上下文已经成为主瓶颈时”的方案。world model 也不是“证明模型像人类理解世界”的哲学标签，而是一个工程判断标准：系统是否真的维护了动作后的状态。

---

## 参考资料

| 标题 | 出处 | 用途 |
|---|---|---|
| [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://proceedings.mlr.press/v267/sun25h.html) | ICML 2025 / PMLR | TTT 层、快权重与线性复杂度序列建模 |
| [TTT: Reimagining LLM Memory by Training on the Fly](https://developer.nvidia.com/blog/reimagining-llm-memory-using-context-as-training-data-unlocks-models-that-learn-at-test-time/) | NVIDIA Technical Blog | TTT 工程直觉、长上下文与推理时更新的系统视角 |
| [TTT-E2E: End-to-End Test-Time Training for Long Context](https://www.emergentmind.com/papers/2512.23675) | 论文摘要索引 | TTT-E2E、sliding-window、权重压缩上下文 |
| [Language Agents Meet Causality: Bridging LLMs and Causal World Models](https://huggingface.co/papers/2410.19923) | arXiv 2024 paper page | World model 与 LLM 结合、因果推演接口设计 |
| [Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](https://aclanthology.org/2025.acl-long.1508/) | ACL 2025 | Ordered CommonGen、Ordered Coverage |
| [Evaluating Morphological Compositional Generalization in Large Language Models](https://aclanthology.org/2025.naacl-long.59/) | NAACL 2025 | 形态层面的组合泛化边界 |
| [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://huggingface.co/papers/2404.06654) | arXiv 2024 paper page | 长上下文有效长度、retrieval 之外的 tracing/aggregation |
| [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | GitHub | RULER 基准实现与任务说明 |
| [Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal](https://aclanthology.org/2024.acl-long.77/) | ACL 2024 | SSR、自合成回放、持续学习 |
| [A Unified Knowledge Management Framework for Continual Learning and Machine Unlearning in Large Language Models](https://www.mdpi.com/2078-2489/17/3/238) | Information 2026 | 持续学习与知识管理框架、LoRA 与参数保护 |
