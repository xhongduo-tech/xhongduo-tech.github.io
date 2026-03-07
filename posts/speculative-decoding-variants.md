## 核心结论

Speculative Decoding 的目标，不是改变语言模型“按顺序生成”的定义，而是把原本严格串行的解码过程，改写成“先并行提出多步候选，再由目标模型一次性验证并接收最长前缀”。这里的 token 可以先理解成模型输出的最小文本单位，可能是一个字、一个词片段，或者标点。

原始的 draft-target 方案思路很直接：先让一个更小、更快的草稿模型起草，再交给大模型验证。但它有四个持续存在的工程问题：要维护第二套模型权重，要保证 tokenizer 对齐，要处理训练和部署版本同步，还要承担“草稿质量一旦下降，接受率和加速比就一起下降”的风险。Medusa、EAGLE、Lookahead 的共同方向，就是尽量把“草稿生成”内嵌进主模型体系或推理流程里，减少独立草稿模型的维护成本。

三者的差异可以先压缩成一句话：

| 方法 | 草稿来源 | 是否需要额外训练 | 典型收益 |
|---|---|---:|---:|
| Medusa | 在主模型上增加多个未来 token 预测头 | 需要 | Medusa 论文报告 Medusa-1 超过 2.2x，Medusa-2 约 2.3x 到 2.8x |
| EAGLE | 用轻量模块预测未来隐藏态，再映射成 token | 需要 | EAGLE 论文在 LLaMA2-Chat 70B 上报告 2.7x 到 3.5x 延迟加速 |
| Lookahead / NGram | 直接复用历史 n-gram 或窗口中的候选轨迹 | 不需要 | NVIDIA 在 Qwen2.5-Coder 7B / 32B 上报告约 3.6x / 1.6x 吞吐提升 |

如果看统一基准而不是论文各自挑选的最好设置，结论会更保守。ACL 2024 的 Spec-Bench 在 Vicuna-7B、batch size = 1、贪心解码 $T=0$ 下给出的平均提速大致是：Medusa 约 $1.48\times$，EAGLE 约 $2.08\times$。这说明两件事。第一，论文里的最佳数字不能直接当成工程常态。第二，EAGLE 通常比 Medusa 更稳定地拿到高接受率，尤其是在通用任务上。

结论并不是“三选一里有绝对优胜者”，而是：

- 想避免独立小模型，同时能接受训练额外模块，优先看 EAGLE。
- 想在现有 LLM 上直接增加并行多步预测能力，优先看 Medusa。
- 不想训练任何额外参数，且任务里存在明显重复模式，优先试 Lookahead。

---

## 问题定义与边界

问题本身很清楚：自回归解码每一步只生成 1 个 token，这导致模型推理在很多场景下不是算力不够，而是串行依赖太强。GPU 擅长并行计算，但传统解码每次只能推进一步，很多时间消耗在重复读取参数、维护 KV Cache 和等待下一步输入，而不是做大规模并行计算。

Speculative Decoding 要解决的，就是这个“每次前进 1 步”的瓶颈。它的统一抽象可以写成：

`prompt -> 草稿生成 -> 目标模型验证 -> 接收最长前缀 / 回退 -> 下一轮`

这里有两个概念必须分开：

- 草稿生成：先给出未来若干步的候选 token 或候选隐藏态。
- 目标模型验证：不是简单比较字符串，而是让目标模型对这些候选做一次并行检查，决定哪些前缀可以被接受。

一个更具体的理解方式是：普通解码像“每次走一步再确认下一步”；speculative decoding 像“先把未来几步都画出来，再由主模型一次判断这条路能走到第几步”。

边界条件同样重要。很多线上收益不稳定，不是算法错，而是前提没满足。

| 边界项 | 为什么重要 | 错了会怎样 |
|---|---|---|
| tokenizer 一致性 | token 切分决定验证单位 | draft-target 接受率会明显下降，甚至直接负优化 |
| batch 大小 | speculative 更依赖 GPU 闲置算力 | 大 batch 下额外验证成本可能吃掉收益 |
| 草稿长度 / 窗口大小 | 决定单轮能前进多远 | 过大时候选变差，验证更重 |
| 任务模式 | 可预测性决定接受率 | 代码、模板化输出通常更适合；开放闲聊波动更大 |
| 采样温度 | 温度越高，不确定性越强 | 接受率通常下降，收益变小 |
| KV Cache 与调度实现 | 验证阶段也要占显存和带宽 | 配置不当时显存吃紧、调度复杂度上升 |

一个新手最容易理解的例子是 Lookahead / NGram。

假设当前前缀是 `The future is`，系统历史里已经见过这个 3-gram 前缀，后面常跟：

- `bright`
- `because`

那么系统可以先把 `bright because` 当成一段草稿提交给目标模型。若目标模型验证后认为这两个 token 都成立，那么本轮就不只输出 1 个 token，而是一次接收 2 个。

更贴近工程的例子是代码补全。比如用户刚输入：

- `def fibonacci(n):`
- `if n <= 1:`
- `return n`

这类局部结构在训练数据和仓库代码里高度重复。Lookahead 或 NGram 很容易从历史轨迹里匹配到后续片段，因此接受率往往比开放闲聊高得多。原因不是代码“更简单”，而是代码的局部续写更可预测。

可以把三类方法的适用边界先记成下面这张表：

| 维度 | Medusa | EAGLE | Lookahead |
|---|---|---|---|
| 是否改模型 | 是 | 是 | 否 |
| 是否要训练 | 是 | 是 | 否 |
| 是否依赖重复模式 | 中等 | 低到中等 | 高 |
| 通用性 | 高 | 高 | 中等 |
| 上线门槛 | 中等 | 中高 | 低 |
| 收益稳定性 | 中等 | 较高 | 很看任务 |

---

## 核心机制与推导

先写统一形式。设目标模型记作 $f_\theta$，输入提示为 $x$，生成序列为 $y_1, y_2, \dots$。普通自回归解码满足：

$$
y_t \sim p_\theta(\cdot \mid x, y_{<t})
$$

也就是第 $t$ 个 token 的分布依赖前面所有 token。这种依赖链导致推理天然串行。

Speculative 系列方法并不取消这个依赖，而是引入一个草稿机制 $g$，先提出长度为 $K$ 的候选：

$$
\hat{y}_{t:t+K-1} = g(x, y_{<t})
$$

然后让目标模型一次性验证这些候选，接收最长合法前缀。若接收长度为 $A$，则这一轮直接前进 $A$ 步，而不是只前进 1 步。

直观上，单轮收益取决于两个量：

$$
\text{收益} \approx \frac{\text{一次验证能接收的 token 数}}{\text{草稿与验证的额外开销}}
$$

所以工程上真正关键的不是“草稿能猜多远”，而是“平均能接收多长前缀，且验证成本是否足够低”。

### 1. Medusa：多个预测头直接猜未来 token

Medusa 的核心思路是：既然目标模型本身已经能很好地表示当前上下文，不如直接在主模型顶部增加多个 decoding heads，让它一次预测未来多个位置，而不是单独再训一个草稿模型。

可以把它理解成：

- 主干模型照常计算当前隐藏态；
- 第 1 个 head 预测 $t+1$；
- 第 2 个 head 预测 $t+2$；
- 第 3 个 head 预测 $t+3$；
- 依此类推。

若只让每个 head 给出 1 个 token，那它只是“一条线性的未来猜测”。Medusa 真正发挥作用的地方在于：每个 head 可以给出 top-k 候选，再把这些候选组织成一棵树，由目标模型统一验证。

形式上，若当前隐藏态为 $h_t$，第 $i$ 个 head 的预测可以写成：

$$
\hat{y}_{t+i} = \operatorname{Head}_i(h_t)
$$

如果每个 head 保留 top-$k$ 候选，就会形成一棵候选树。为了避免不同分支不正确地看到彼此的未来 token，Medusa 引入了 tree attention。它的作用可以直白地理解为：每条候选路径只能访问自己的祖先节点，不能偷看旁边分支的未来。

一个简化的树结构如下：

```text
t
├─ a   (t+1)
│  ├─ c (t+2)
│  └─ d (t+2)
└─ b   (t+1)
   ├─ e (t+2)
   └─ f (t+2)
```

目标模型验证后，接收其中与自身分布一致的最长前缀。Medusa 的优点是：

- 不需要维护第二个独立草稿模型；
- tokenizer、权重版本、部署链路更容易与主模型保持一致；
- 在已有训练流水线下比较容易集成。

代价也很明确：

- 需要为额外 heads 做训练或微调；
- 候选树越大，验证成本越高；
- 如果 heads 对未来 token 的预测不够准，接受率会下降。

对新手来说，一个常见误解是“Medusa 只是多输出几个 token”。不准确。它做的是“并行生成多个未来位置的候选，再由目标模型决定实际能接收多少”，不是跳过验证直接采纳。

### 2. EAGLE：先预测未来隐藏态，再解码成 token

EAGLE 的出发点比 Medusa 更进一步。它认为直接预测未来 token 太难，因为 token 是离散空间，局部差一个 token 就可能完全走到另一条分支；但未来隐藏态是连续特征空间，变化更平滑，更适合被轻量模块外推。

所以 EAGLE 不先猜 token，而是先猜未来特征。流程可以写成：

1. 从目标模型读取当前隐藏态；
2. 用一个轻量预测器自回归地外推出未来隐藏态；
3. 再通过 LM head 把这些隐藏态映射成 token 分布；
4. 由目标模型验证并接收最长前缀。

若当前层隐藏态为 $h_t$，EAGLE 的预测器可写成：

$$
\hat{h}_{t+1}, \hat{h}_{t+2}, \dots, \hat{h}_{t+K}
= g_\phi(h_{\le t})
$$

再通过共享或兼容的 LM head 映射成 token：

$$
p(\hat{y}_{t+i}\mid x, y_{<t}) = \operatorname{LMHead}(\hat{h}_{t+i})
$$

它与 Medusa 的根本区别是：

- Medusa：直接从当前状态猜未来 token。
- EAGLE：先猜未来隐藏态，再把隐藏态译回 token。

这带来两个结果。第一，特征预测通常更稳，因为连续空间更容易学习。第二，EAGLE 的接受率通常更高，所以在统一基准里提速更稳定。ACL 2024 Survey 里对它的总结也很直接：EAGLE 在多个 Spec-Bench 子任务上都保持较高速度提升。

对新手来说，可以把“隐藏态”先理解成模型在某一层对当前上下文的压缩表示。它不是最终输出，但它包含了“下一步大概要往哪里走”的内部信息。EAGLE 先预测这个内部表示，再从表示恢复 token，相当于先预测中间变量，而不是直接预测最终答案。

EAGLE 的工程代价主要在于：

- 需要训练轻量预测器；
- 需要保证预测器与主模型隐藏态接口兼容；
- 某些实现会重用主模型 KV Cache，这要求部署框架支持得足够好。

### 3. Lookahead：把未来窗口并行刷新

Lookahead 是三者里最容易部署、也最容易被误解的一类。它不训练额外参数，不增加模型头，而是改变“未来候选的组织方式”。

它借用了 Jacobi 迭代的视角。Jacobi 迭代的核心思想是：在一轮更新中，先用上一轮的旧值并行求出所有新值，再统一替换。把这个思路放到解码里，就得到：

$$
y_t^{(k+1)} = f\big(x, y_1^{(k)}, y_2^{(k)}, \dots, y_{t-1}^{(k)}\big)
$$

这里的 $k$ 表示第几轮刷新。普通自回归的顺序是“先确定 $y_1$，再用它算 $y_2$，再算 $y_3$”；Lookahead 的视角是“在固定窗口里，同时维护多个未来位置的候选，然后不断刷新这些候选”。

在 TensorRT-LLM 文档和 NVIDIA 博客的描述里，它由两个分支组成：

- lookahead branch：在固定窗口中并行生成 n-gram 候选；
- verification branch：挑出有希望的候选，交给目标模型验证。

三个核心参数是：

- $W$：window size，窗口有多宽，决定一次往前看多远。
- $N$：n-gram size，每个候选片段有多长。
- $G$：verification set size，每轮最多送多少条候选去验证。

一个直观的二维窗口可以画成：

```text
位置:   t+1   t+2   t+3
轮次1    a     b     c
轮次2    a'    b'    c'
轮次3    a''   b''   c''
```

如果某些轨迹在历史中反复出现，它们就会被放进 n-gram 池；验证分支再从这些候选里挑出最有希望命中的片段。

这也是为什么 Lookahead 特别吃“重复模式”：

- 代码补全里，函数定义、循环、异常处理、导包都高度重复；
- RAG 模板回答里，很多句式是固定的；
- 结构化生成里，JSON key、SQL 片段、配置块都有强重复性。

相反，在开放式创作或高温采样下，未来 token 的不确定性更大，历史轨迹的复用价值更低，Lookahead 的收益会明显收缩。

一个简单但实用的判断标准是：

- 任务越接近“局部模式复用”，Lookahead 越值得试。
- 任务越接近“每一步都高度发散”，Lookahead 越容易退化成额外开销。

为了帮助新手把三种机制放到同一个框架里看，可以再补一张对照表：

| 维度 | Medusa | EAGLE | Lookahead |
|---|---|---|---|
| 直接预测对象 | 未来 token | 未来隐藏态 | 历史 n-gram / 窗口候选 |
| 草稿结构 | 多头 + 候选树 | 轻量特征预测器 | n-gram 池 + 验证分支 |
| 验证方式 | 主模型验证树前缀 | 主模型验证草稿前缀 | 主模型验证候选段 |
| 主要瓶颈 | heads 训练质量 | 特征预测器训练与接口 | 命中率与窗口配置 |

---

## 代码实现

先给一个可以直接运行的纯 Python 玩具实现。它模拟的是最容易解释的 NGram / Lookahead 路线：先从历史池里提出草稿，再用“目标模型输出”做验证。

```python
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


class NGramPool:
    def __init__(self, prefix_len: int = 3):
        self.prefix_len = prefix_len
        self.pool: Dict[Tuple[str, ...], List[List[str]]] = defaultdict(list)

    def add_sequence(self, tokens: Sequence[str], draft_len: int = 3) -> None:
        if len(tokens) < self.prefix_len + 1:
            return
        for i in range(self.prefix_len, len(tokens)):
            key = tuple(tokens[i - self.prefix_len:i])
            draft = list(tokens[i:i + draft_len])
            if draft:
                self.pool[key].append(draft)

    def propose(self, prefix_tokens: Sequence[str]) -> List[str]:
        key = tuple(prefix_tokens[-self.prefix_len:])
        candidates = self.pool.get(key, [])
        if not candidates:
            return []
        # 这里用“最长候选”做一个简单策略；真实系统会更复杂
        return max(candidates, key=len)


def verify_draft(draft_tokens: Sequence[str], target_tokens: Sequence[str], max_accept: int) -> List[str]:
    accepted = []
    for i, tok in enumerate(draft_tokens[:max_accept]):
        if i < len(target_tokens) and tok == target_tokens[i]:
            accepted.append(tok)
        else:
            break
    return accepted


def speculative_ngram_step(
    prefix_tokens: Sequence[str],
    ngram_pool: NGramPool,
    target_tokens: Sequence[str],
    max_accept: int = 4,
) -> List[str]:
    draft = ngram_pool.propose(prefix_tokens)
    return verify_draft(draft, target_tokens, max_accept=max_accept)


if __name__ == "__main__":
    history = [
        "The future is bright because we build useful tools".split(),
        "The future is bright when models become easier to serve".split(),
    ]

    pool = NGramPool(prefix_len=3)
    for seq in history:
        pool.add_sequence(seq, draft_len=3)

    accepted = speculative_ngram_step(
        prefix_tokens="The future is".split(),
        ngram_pool=pool,
        target_tokens="bright because we build".split(),
        max_accept=3,
    )

    print("accepted:", accepted)
    assert accepted == ["bright", "because", "we"]
```

运行结果会输出：

```text
accepted: ['bright', 'because', 'we']
```

这个例子虽然简化，但它把 speculative decoding 的核心动作完整保留了：

1. 根据当前前缀拿草稿；
2. 用目标模型逐个验证；
3. 接收最长匹配前缀；
4. 一次前进多个 token。

如果想再往前一步，把整个“生成循环”也写出来，可以用下面这个更完整的版本：

```python
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


class NGramPool:
    def __init__(self, prefix_len: int = 3):
        self.prefix_len = prefix_len
        self.pool: Dict[Tuple[str, ...], List[List[str]]] = defaultdict(list)

    def add_sequence(self, tokens: Sequence[str], draft_len: int = 4) -> None:
        for i in range(self.prefix_len, len(tokens)):
            key = tuple(tokens[i - self.prefix_len:i])
            draft = list(tokens[i:i + draft_len])
            if draft:
                self.pool[key].append(draft)

    def propose(self, prefix: Sequence[str]) -> List[str]:
        if len(prefix) < self.prefix_len:
            return []
        key = tuple(prefix[-self.prefix_len:])
        candidates = self.pool.get(key, [])
        return max(candidates, key=len) if candidates else []


def accept_prefix(draft: Sequence[str], target_future: Sequence[str]) -> int:
    n = 0
    for d, t in zip(draft, target_future):
        if d != t:
            break
        n += 1
    return n


def generate_with_speculation(
    prompt_tokens: List[str],
    oracle_tokens: List[str],
    pool: NGramPool,
    fallback_one_token: bool = True,
) -> List[str]:
    output = list(prompt_tokens)
    cursor = 0

    while cursor < len(oracle_tokens):
        draft = pool.propose(output)
        accepted = accept_prefix(draft, oracle_tokens[cursor:])

        if accepted > 0:
            output.extend(oracle_tokens[cursor:cursor + accepted])
            cursor += accepted
        elif fallback_one_token:
            output.append(oracle_tokens[cursor])
            cursor += 1
        else:
            break

    return output


if __name__ == "__main__":
    pool = NGramPool(prefix_len=3)
    pool.add_sequence("def fibonacci ( n ) : if n <= 1 : return n".split(), draft_len=5)
    pool.add_sequence("def fibonacci ( n ) : return fibonacci ( n - 1 ) + fibonacci ( n - 2 )".split(), draft_len=5)

    prompt = "def fibonacci ( n ) :".split()
    oracle = "if n <= 1 : return n".split()

    result = generate_with_speculation(prompt, oracle, pool)
    print(" ".join(result))
```

这个脚本同样可以直接运行。它没有依赖真实大模型，但已经足够演示“草稿命中时一次前进多步，没命中时回退到普通自回归”的逻辑。

真正落到生产环境时，一般不会自己手写验证流程，而是直接使用推理框架提供的 speculative decoding API。以 NVIDIA 在 2025 年 2 月博客里展示的 Lookahead 配置为例，一个更接近可运行状态的 TensorRT-LLM 脚本可以写成：

```python
from tensorrt_llm.llmapi import (
    LLM,
    BuildConfig,
    KvCacheConfig,
    LookaheadDecodingConfig,
    SamplingParams,
)


def main():
    # max_draft_len 常按 (W + G - 1) * (N - 1) + (0 if N <= 1 else N - 2) 估算
    build_config = BuildConfig(
        max_batch_size=128,
        max_input_len=2048,
        max_seq_len=4096,
        max_num_tokens=16384,
        max_draft_len=111,
    )

    lookahead_config = LookaheadDecodingConfig(
        max_window_size=8,
        max_ngram_size=8,
        max_verification_set_size=8,
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

    llm = LLM(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        build_config=build_config,
        kv_cache_config=kv_cache_config,
        speculative_config=lookahead_config,
    )

    sampling_params = SamplingParams(lookahead_config=lookahead_config)

    outputs = llm.generate(
        "Write a Python LRU cache implementation with type hints.",
        sampling_params=sampling_params,
    )
    print(outputs)


if __name__ == "__main__":
    main()
```

如果只想使用不需要训练的 NGram 路线，配置通常更简单：

```python
from tensorrt_llm.llmapi import LLM, NGramDecodingConfig

speculative_config = NGramDecodingConfig(
    max_draft_len=3,
    max_matching_ngram_size=4,
    is_public_pool=True,
)

llm = LLM(
    model="/path/to/model",
    speculative_config=speculative_config,
    disable_overlap_scheduler=True,
)
```

对于 Medusa 和 EAGLE，NVIDIA TensorRT Model Optimizer 提供的入口是 `mtsp.convert()`。一个最小示例是：

```python
import modelopt.torch.speculative as mtsp
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

mode = "medusa"
config = {
    "medusa_num_heads": 2,
    "medusa_num_layers": 1,
}

mtsp.convert(model, [(mode, config)])
print(type(model))
```

如果切到 EAGLE，最小配置通常类似：

```python
import modelopt.torch.speculative as mtsp
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

mode = "eagle"
config = {
    "eagle_num_layers": 1,
}

mtsp.convert(model, [(mode, config)])
print(type(model))
```

但这里必须补一句工程上的关键说明：`convert()` 只是在模型里插入 Medusa heads 或 EAGLE 模块，后面还需要微调这些新模块，否则它们通常拿不到有意义的接受率。也就是说，真正可上线的路径是：

`基础模型 -> convert() -> 训练新增模块 -> 保存 / 恢复 -> 推理部署`

最后再把“代码补全为什么更适合 Lookahead”落实到数据上。NVIDIA 在 2025 年 2 月发布的 Qwen2.5-Coder 博客里给出的数字是：在 H100、batch size = 1 的条件下，Lookahead 让 Qwen2.5-Coder 7B Instruct 拿到约 $3.6\times$ 吞吐提升，32B Instruct 约 $1.6\times$。这类收益并不意味着 Lookahead 普遍优于 EAGLE 或 Medusa，而是说明在代码补全这种重复模式强、局部结构稳定的任务里，“零训练 + 历史模式复用”能非常有效。

---

## 工程权衡与常见坑

第一类坑是把参数调大当成普适优化。无论是 Lookahead 的 $W, N, G$，还是 Medusa 的 heads 数量、树宽，或者 EAGLE 的草稿长度，变大都不自动等于更快。因为 speculative decoding 不是白拿收益，它一定伴随额外草稿成本和额外验证成本。

一个非常实用的近似判断是：

$$
\text{若 } \mathbb{E}[\text{accept length}] \text{ 提升不够快，额外验证成本就会吞掉收益}
$$

所以参数 sweep 必须在目标硬件、目标 batch、目标任务上做，不能直接照搬论文或博客配置。

第二类坑是任务错配。Lookahead 依赖 n-gram 命中率，命中率低时，它剩下的就只有维护候选池、刷新窗口、送验证集的额外开销。Medusa 和 EAGLE 更通用，但它们也不是“训练完就稳定加速”。一旦线上 prompt 分布与训练数据差距过大，接受率也会掉。

第三类坑是 tokenizer 和 checkpoint 对齐。原始 draft-target 最怕 tokenizer 不一致，因为 token 切分不一样，验证时几乎天然冲突。Medusa 和 EAGLE 因为更贴近主模型体系，这个问题轻一些，但部署时仍然要核对：

- tokenizer 文件是否与权重对应；
- 训练和部署用的 vocab / special tokens 是否一致；
- 新模块是否是针对当前 checkpoint 训练出来的。

第四类坑是误把“吞吐提升”和“单请求延迟下降”当成同一个指标。Speculative decoding 论文和博客里有的报 latency，有的报 throughput，有的报 tokens/s。三者不能混读：

- latency：单次请求完成得多快；
- throughput：单位时间总共生成多少 token；
- speedup ratio：相对 baseline 的倍数。

Lookahead 在 NVIDIA 代码场景博客里强调的是吞吐提升；EAGLE 论文里更强调延迟加速；Spec-Bench 多数图表是在统一环境下比较 speedup ratio。写工程文档时如果不区分这几个量，结论会失真。

下面这张表可以作为排查清单：

| 参数/因素 | 调大后的潜在收益 | 主要风险 | 什么时候先调它 |
|---|---|---|---|
| `W` window size | 一次看得更远 | 验证开销上升，命中差时更慢 | Lookahead 命中已不错，但单轮前进还不够长 |
| `N` n-gram size | 单次可能接收更长片段 | 候选更稀疏，命中率下降 | 代码/模板场景重复度很高时 |
| `G` verification set size | 更多候选进入验证 | 主模型验证更重 | 候选质量高但被验证数限制时 |
| `max_draft_len` | 单轮潜在输出更多 token | 显存与调度复杂度增加 | 已确认接受率高时 |
| Medusa head 数 | 能预测更远未来 | heads 训练更难，树更大 | 现有 heads 明显不够用时 |
| EAGLE 草稿步数 | 平均前进距离上升 | 特征误差累积，接受率下降 | 中短期特征预测很稳时 |
| `is_public_pool` | 多请求共享模式命中 | 池污染，跨请求干扰 | 多请求场景 prompt 模式相似时 |
| `is_keep_all` | 候选更丰富 | 内存和检索开销上涨 | 命中不足但显存富余时 |

还有几个常见误区值得单独指出：

- “低 batch 一定有收益，高 batch 一定没收益。”
  
  更准确的说法是：speculative decoding 在低 batch 下通常更容易看到收益，因为基础自回归没把 GPU 吃满；batch 越大，额外验证的边际回报通常越低，但是否值得仍然取决于模型大小、序列长度和任务形态。

- “接受率高就一定快。”
  
  不一定。接受率高是必要条件，不是充分条件。如果验证阶段本身很重，或者调度实现不够高效，整体仍然可能只得到有限加速。

- “论文里 lossless，就等于线上一定完全无质量损失。”
  
  也不严谨。lossless 的前提是接受规则和采样流程严格满足目标分布恢复条件。工程里一旦加入近似接受、启发式剪枝、版本不一致、量化误差，实际表现就要重新评估。

可以把一次 speculative 上线前的最小检查流程写成：

1. 固定一个 baseline：同模型、同量化、同 prompt 集。
2. 分别测 latency、throughput、accept length。
3. 看 acceptance 是否稳定，而不是只看平均值。
4. 在真实业务 prompt 上测，而不是只在演示数据上测。
5. 观察显存、KV cache、调度冲突是否带来副作用。

---

## 替代方案与适用边界

如果把选择规则压缩成一个决策表，最实用的是下面这张：

| 方法 | 需不需要独立草稿模型 | 是否追求与原分布严格一致 | 更适合什么 prompt 模式 |
|---|---|---|---|
| Medusa | 不需要 | 可以做成 lossless，但依赖训练与接受实现 | 通用文本，已有训练能力，想直接扩展主模型 |
| EAGLE | 不需要独立小模型，但需要轻量特征预测器 | 强调保持目标分布一致 | 通用场景，尤其重视接受率与稳定性 |
| Lookahead | 不需要 | 取决于具体验证策略；最大优势是零训练接入 | 重复模式强的代码、模板、RAG 回答 |

如果换成“什么时候选谁”的语言，可以更直接一些：

- 需要较稳定的通用加速，同时愿意训练额外模块，优先 EAGLE。
  
  统一基准里它通常比 Medusa 更稳，Spec-Bench 在 Vicuna-7B、$T=0$ 的平均提速约为 $2.08\times$。

- 已经有现成的训练流程，且希望在主模型内部直接加入多步预测能力，可以选 Medusa。
  
  它的优点不是最好训练，而是结构直接、思路清楚、与主模型集成紧。

- 不想重新训练，只想先在部署层快速试探加速，并且业务文本高度重复，优先试 Lookahead。
  
  它非常适合代码补全、结构化生成、模板回答，但不适合把所有开放式生成都当成同样的收益场景。

还要补一句边界判断：真正决定方法选择的，不是哪篇论文的最高加速比更大，而是你所在系统能不能长期稳定地拿到足够高的接受率，同时不让验证成本、显存成本和调度复杂度反过来吞掉收益。

Medusa、EAGLE、Lookahead 之外，还有几条常见路线：

| 路线 | 简述 | 适用边界 |
|---|---|---|
| Draft-Target | 经典小模型起草、大模型验证 | 已有现成小模型，且 tokenizer 能严格对齐 |
| SpS | Self-speculative / skip 类路线 | 某些模型结构下能减少额外模块成本 |
| PLD / Prompt Lookup | 从 prompt 或历史中直接查找续写 | 输入输出重叠高时很强 |
| REST | 通过回收或重用解码信息减少串行开销 | 具体收益很依赖任务与实现 |

所以最终决策通常不是“论文名之间二选一”，而是下面这三个问题：

1. 你能不能接受额外训练与额外模块维护？
2. 你的任务是否存在足够强的局部重复模式？
3. 你的线上硬件与 batch 分布，是否给 speculative 留出了可利用的空闲算力？

如果三个问题的答案分别是：

- 能训练、要通用稳定：优先 EAGLE。
- 能训练、想直接增强主模型：优先 Medusa。
- 不想训练、任务高度重复：优先 Lookahead。

---

## 参考资料

- NVIDIA TensorRT-LLM Speculative Decoding 文档：<https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>
- NVIDIA TensorRT-LLM 高级文档 Speculative Sampling：<https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html>
- NVIDIA 技术博客《Optimizing Qwen2.5-Coder Throughput with NVIDIA TensorRT-LLM Lookahead Decoding》：<https://developer.nvidia.com/blog/optimizing-qwen2-5-coder-throughput-with-nvidia-tensorrt-llm-lookahead-decoding/>
- NVIDIA 技术博客《TensorRT-LLM Speculative Decoding Boosts Inference Throughput by up to 3.6x》：<https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/>
- Medusa 论文（ICML 2024 / PMLR）：<https://proceedings.mlr.press/v235/cai24b.html>
- EAGLE 论文（ICML 2024 / PMLR）：<https://proceedings.mlr.press/v235/li24bt.html>
- Lookahead Decoding 论文《Break the Sequential Dependency of LLM Inference Using Lookahead Decoding》：<https://arxiv.org/abs/2402.02057>
- ACL 2024 Survey《Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding》：<https://aclanthology.org/2024.findings-acl.456/>
- TensorRT Model Optimizer Speculative Decoding 指南：<https://nvidia.github.io/TensorRT-Model-Optimizer/guides/7_speculative_decoding.html>
