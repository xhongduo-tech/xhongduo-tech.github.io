## 核心结论

自回归生成的约束没有变：第 $t+1$ 个 token 的生成依赖前面已经确认的 token，所以语义上它仍然是顺序过程。能并行化的，不是“把顺序依赖删除”，而是把生成流程拆成几个角色：

- 谁先提出候选 token
- 谁负责验证这些候选 token
- 哪些请求可以一起送进 GPU 执行

因此，工程上的并行化目标不是“让模型一次性无条件生成整句”，而是把原本“主模型每轮只确认 1 个 token”的流程，改写成“主模型每轮平均确认多个 token”。

最常见的三类手段如下：

| 方案 | 并行的对象 | 是否需要额外草稿模型 | 主要收益 | 主要代价 |
| --- | --- | --- | --- | --- |
| 交错解码 / Speculative Decoding | 草稿生成与主模型验证 | 通常需要 | 降低主模型前向次数，改善吞吐与 TPOT | 额外模型、KVCache 管理更复杂 |
| 块级并行 / Blockwise Parallel Decoding | 一次预测一整块 token，再回退最长前缀 | 不一定 | 减少解码迭代轮数 | 评分/回退逻辑复杂，长上下文收益不稳定 |
| 批处理并行 / Batch Parallelism | 多个请求一起解码 | 不需要 | 摊薄调度与 kernel 启动开销，提高 GPU 利用率 | 单请求延迟不一定最优，排队与内存压力上升 |

对部署最关键的一条结论是：如果草稿阶段一次提出 $\gamma$ 个候选 token，且单个候选 token 的平均接受率为 $\alpha$，那么主模型每一轮验证平均能确认的 token 数为

$$
\tau = 1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma
= \frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$

这里的“接受率”可以直接理解为：草稿模型猜出来的 token，有多大概率会被主模型认可。$\tau$ 越大，主模型真正要执行的解码轮数越少。

一个最小数值例子：

- 若 $\alpha \approx 0.6$
- 草稿长度 $\gamma = 5$

则

$$
\tau = \frac{1-0.6^6}{1-0.6} \approx 2.38
$$

这意味着主模型平均一轮不再只确认 1 个 token，而是确认约 2.4 个 token。假设主模型原本需要 50 ms 才能稳定确认 1 个 token，那么只看主模型前向这一项，等效下来每个 token 的平均开销约为

$$
50 / 2.38 \approx 21 \text{ ms}
$$

当然，这只是理想化估算。实际系统还包含：

- 草稿模型本身的前向成本
- 调度器开销
- 采样开销
- KVCache 分配与回收
- GPU 同步与 kernel 启动开销

所以工程上通常看到的是 2x 到 5x 的吞吐提升，而不是简单按公式得到的理论上限。

---

## 问题定义与边界

问题不是“Transformer 能不能并行”，而是“解码阶段的关键路径能不能缩短”。

这两个阶段要严格区分：

| 阶段 | 是否天然并行 | 原因 |
| --- | --- | --- |
| 训练阶段 | 是 | 整个目标序列已知，所有位置都能并行计算 loss |
| 推理阶段 | 否 | 第 $t+1$ 个 token 的条件分布依赖已经确定的第 $t$ 个 token |

以最普通的 greedy decoding 为例，流程其实只有四步：

1. 对当前前缀做一次前向计算
2. 取出下一个 token
3. 把这个 token 拼回输入序列
4. 再做下一次前向计算

所以，自回归推理的瓶颈不是注意力矩阵本身能不能在单次前向里并行，而是“前向调用次数”很多，而且这些调用存在严格先后关系。

这会带来几个直接后果：

| 维度 | 串行解码 | speculative / 并行化解码 |
| --- | --- | --- |
| token 依赖 | 每次只推进 1 个 | 语义仍顺序，但执行上可先猜后验 |
| 主模型前向次数 | 约等于输出 token 数 | 可压缩到约 `输出 token 数 / τ` |
| KVCache 占用 | 只存已确认 token | 还要为草稿 token 预留空间 |
| 调度复杂度 | 低 | 高，需要 accept / reject / rewind |
| GPU 利用率 | 容易出现等待 | 更容易把 SM、带宽、CUDA graph 填满 |

这里的 KVCache 可以先用一句话理解：模型为了避免每次都重算历史 token 的注意力，会把历史 token 的 key/value 状态缓存在显存里。后续生成第 $t+1$ 个 token 时，会直接读取这些缓存。

在 speculative decoding 里，会出现一个新问题：草稿 token 还没被正式确认，但主模型已经要拿它们做一次“批量验证前向”。这意味着系统必须先给这些临时 token 分配 KV 页。如果后面发现一部分草稿 token 被拒绝，还要把相应的 KV 页回收掉。

所以实现边界非常明确：

- 你不能只说“算法上先猜 5 个 token”
- 你还必须回答“这 5 个 token 的 KV 空间从哪里来”
- 以及“如果其中 3 个被拒绝，哪些缓存需要 rewind，什么时候回收”

一个真实工程中的典型问题是：主模型和草稿模型共用同一张或同一组 GPU，打开 speculative 以后吞吐不升反降，甚至下一轮调度直接失败。原因往往不是算法本身有问题，而是 KVCache 的预算没做对。比如 `max_draft_len` 设得很大，但没有预留足够页数，结果草稿 token 抢占了后续请求的缓存空间，最终导致 OOM 或调度退化。

因此，本文讨论的边界是：

- 保持 autoregressive 语义不变
- 通过交错验证、块级预测、批处理调度缩短关键路径
- 不讨论彻底改成非自回归模型的方案

---

## 核心机制与推导

先看 speculative decoding 的最小循环。它的逻辑不是“草稿模型替代主模型”，而是“草稿模型先提案，主模型再验收”。

```text
当前前缀
  -> 草稿模型先猜 γ 个 token
  -> 主模型一次性验证这 γ 个 token
  -> 接受最长一致前缀
  -> 若中途不一致，则拒绝后续 token，并回退 KV
  -> 主模型补出下一个真实 token
  -> 进入下一轮
```

把这段流程翻译成更直白的话，就是：

- 草稿模型负责“便宜地多猜几个”
- 主模型负责“昂贵但可靠地核对”
- 接受的草稿 token 直接进入最终输出
- 不接受的草稿 token 必须丢弃，不能偷偷保留

### 玩具例子

假设当前前缀是：

`今天天气`

草稿模型一次猜 4 个 token：

`很 好 ， 适合`

主模型验证后发现：

- `很` 正确
- `好` 正确
- 第 3 个位置也许表面上还是标点，但在主模型分布下当前位置应该进入另一条生成轨迹
- 因为第 3 个 token 不再一致，第 4 个 token 的条件前缀也已经变了，所以不能继续沿用

于是这一轮的结果是：

| 项 | 结果 |
| --- | --- |
| 接受长度 | 2 |
| 拒绝长度 | 2 |
| KV 处理 | 前 2 个保留，后 2 个回退 |
| 主模型额外动作 | 补出一个真实 token，回到原始分布轨迹 |

这里“主模型补出一个真实 token”非常关键。它保证整个过程在概率分布上仍然等价于原始目标模型，而不是变成“草稿模型主导、主模型只是抽查”。

### 为什么 $\tau$ 是几何级数

假设每个草稿 token 独立地以概率 $\alpha$ 被接受。这当然不是严格真实世界，因为 token 接受并不独立，但作为工程近似，这个模型足够有用。

那么：

- 至少接受 1 个 token 的概率近似为 $\alpha$
- 至少接受 2 个 token 的概率近似为 $\alpha^2$
- 至少接受 3 个 token 的概率近似为 $\alpha^3$

一直到最多接受 $\gamma$ 个草稿 token。再加上“主模型补 1 个真实 token”的那一步，就得到平均确认长度：

$$
\tau = 1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma
$$

这是标准等比数列，化简后得到：

$$
\tau = \frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$

这个公式可以直接读出两个工程判断：

1. 如果 $\alpha$ 很低，继续增大 $\gamma$ 的收益会迅速衰减。
2. 如果 $\alpha$ 足够高，增加 $\gamma$ 才可能显著减少主模型前向次数。

下面给出几组常见数值：

| $\alpha$ | $\gamma$ | $\tau$ | 直观解释 |
| --- | --- | --- | --- |
| 0.2 | 5 | $\approx 1.25$ | 基本只是偶尔多确认一点，收益很弱 |
| 0.4 | 5 | $\approx 1.66$ | 有改善，但不一定覆盖额外调度成本 |
| 0.6 | 5 | $\approx 2.38$ | 开始进入值得部署和调优的区间 |
| 0.8 | 5 | $\approx 3.69$ | 主模型前向轮数显著下降 |

如果把主模型基线时间记为 $T_{\text{target}}$，草稿模型每轮时间记为 $T_{\text{draft}}$，忽略其他开销时，单 token 的粗略时间可写成：

$$
T_{\text{eff}} \approx \frac{T_{\text{target}} + T_{\text{draft}}}{\tau}
$$

更完整一点，可以把调度和缓存管理开销记为 $T_{\text{sched}}$：

$$
T_{\text{eff}} \approx \frac{T_{\text{target}} + T_{\text{draft}} + T_{\text{sched}}}{\tau}
$$

这个式子解释了为什么“猜更多 token”不一定更快。因为分子也在增加，只有当 $\tau$ 的增长快于额外成本时，整体收益才成立。

### 交错解码与块级并行的关系

两者的目标相同，都是减少关键路径上的解码轮数，但实现方式不同。

| 对比项 | 交错解码 / Speculative | 块级并行 / Blockwise |
| --- | --- | --- |
| 核心思想 | 先由草稿模型提案，再由主模型验证 | 直接并行预测一个块，再按评分回退 |
| 是否常用额外模型 | 是 | 不一定 |
| 回退依据 | 最长一致前缀 | 块内评分或置信度机制 |
| 工程重点 | 接受率、KV 回退、双模型调度 | 块预测质量、评分成本、回退策略 |

可以把它们理解成两条不同路线：

- speculative 是“先便宜猜，再昂贵验”
- blockwise 是“先成块预测，再回退到可信前缀”

### 真实工程例子

一个典型部署组合如下：

| 角色 | 示例配置 |
| --- | --- |
| 主模型 | 70B，`tensor_parallel_size=4` |
| 草稿模型 | 小型 draft model 或 EAGLE3，单卡 |
| 推理框架 | vLLM 或 TensorRT-LLM |
| 草稿长度 | 2 到 5 个 token 起步 |
| 核心指标 | 接受率、TPOT、吞吐、P99 延迟、显存水位 |

流程上通常是：

1. 草稿模型快速提出 2 到 5 个候选 token
2. 主模型跨多卡做一次验证前向
3. 调度器根据最长接受前缀决定保留哪些 KV
4. 对被拒绝部分执行 rewind
5. 进入下一轮

这样做的意义不是让 70B 主模型“少算参数”，而是让它“减少必须串行发生的前向轮数”。

---

## 代码实现

先给一个能直接运行的 Python 例子，把公式、直觉和一个最简的 speculative 模拟放在一起。下面代码不依赖任何第三方包，复制后即可运行。

```python
from __future__ import annotations


def acceptance_length(alpha: float, gamma: int) -> float:
    """平均每轮确认的 token 数 τ。"""
    assert 0.0 <= alpha < 1.0
    assert gamma >= 0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def simulate_round(draft_tokens: list[str], target_tokens: list[str]) -> tuple[int, list[str], list[str]]:
    """
    返回：
    1. accepted_len: 最长接受前缀长度
    2. accepted_tokens: 被接受的草稿 token
    3. rejected_tokens: 被拒绝的草稿 token
    """
    accepted_len = 0
    for d, t in zip(draft_tokens, target_tokens):
        if d == t:
            accepted_len += 1
        else:
            break

    accepted_tokens = draft_tokens[:accepted_len]
    rejected_tokens = draft_tokens[accepted_len:]
    return accepted_len, accepted_tokens, rejected_tokens


def main() -> None:
    # 公式示例
    alpha = 0.6
    gamma = 5
    tau = acceptance_length(alpha, gamma)
    assert round(tau, 2) == 2.38

    baseline_ms = 50.0
    effective_ms = baseline_ms / tau
    assert 20.0 < effective_ms < 22.0

    tau_low = acceptance_length(0.2, 5)
    assert tau_low < 1.3

    print("=== Formula Demo ===")
    print(f"alpha={alpha}, gamma={gamma}, tau={tau:.4f}")
    print(f"baseline_ms={baseline_ms:.1f}, effective_ms={effective_ms:.2f}")

    # 最长前缀接受模拟
    prefix = ["今天天气"]
    draft = ["很", "好", "，", "适合"]
    target = ["很", "好", "但", "风大"]

    accepted_len, accepted_tokens, rejected_tokens = simulate_round(draft, target)

    print("\n=== Prefix Accept Demo ===")
    print("prefix:", prefix)
    print("draft :", draft)
    print("target:", target)
    print("accepted_len:", accepted_len)
    print("accepted_tokens:", accepted_tokens)
    print("rejected_tokens:", rejected_tokens)

    # 结果解释
    if accepted_len < len(draft):
        true_next_token = target[accepted_len]
        print("target_model_next_true_token:", true_next_token)


if __name__ == "__main__":
    main()
```

这段程序至少说明了三件事：

| 代码片段 | 说明 |
| --- | --- |
| `acceptance_length()` | 用公式把接受率和草稿长度映射成平均确认长度 |
| `simulate_round()` | 演示“接受最长一致前缀”的核心逻辑 |
| `main()` | 给出数值例子和一个可打印的最小执行结果 |

如果运行它，输出会类似：

```text
=== Formula Demo ===
alpha=0.6, gamma=5, tau=2.3834
baseline_ms=50.0, effective_ms=20.98

=== Prefix Accept Demo ===
prefix: ['今天天气']
draft : ['很', '好', '，', '适合']
target: ['很', '好', '但', '风大']
accepted_len: 2
accepted_tokens: ['很', '好']
rejected_tokens: ['，', '适合']
target_model_next_true_token: 但
```

这个例子仍然不是完整推理系统，但它已经把 speculative decoding 的基本机制拆清楚了：收益不取决于“提出了多少候选”，而取决于“最终能确认多少候选”。

### vLLM 的核心配置

在 vLLM 中，speculative decoding 主要通过 `speculative_config` 暴露。下面给出一个结构完整、语义清晰的最小示例：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Explain speculative decoding in one paragraph."
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 5,
        "method": "eagle3",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

这段配置可以按角色来理解：

| 配置项 | 含义 |
| --- | --- |
| `model` | 主模型，负责最终验证 |
| `tensor_parallel_size=4` | 主模型分布在 4 张卡上执行 |
| `speculative_config.model` | 草稿模型，用于提出候选 token |
| `draft_tensor_parallel_size=1` | 草稿模型通常单卡运行 |
| `num_speculative_tokens=5` | 每轮最多先猜 5 个 token |
| `method="eagle3"` | 指定草稿方法实现 |

这里有两个对新手很重要的边界条件：

1. speculative decoding 不是把主模型替换成草稿模型。
2. `num_speculative_tokens` 不是越大越好，它必须和接受率、显存水位一起评估。

### TensorRT-LLM 的核心机制

TensorRT-LLM 的难点不只是“开关怎么写”，而是调度器如何为草稿 token 预留 KV 空间，以及拒绝后如何安全回收。

下面的伪代码刻意保留了 `py_draft_tokens` 和 `py_rewind_len` 这两个关键概念：

```python
class Request:
    def __init__(self, request_id: str, max_draft_len: int):
        self.request_id = request_id
        self.max_draft_len = max_draft_len
        self.py_draft_tokens: list[int] = []
        self.py_rewind_len: int = 0


def schedule_and_allocate_kv(requests: list[Request]) -> None:
    # 调度器读取 py_draft_tokens 的长度，为每个请求预留 KV 页
    for req in requests:
        reserved = len(req.py_draft_tokens)
        print(f"[schedule] req={req.request_id}, reserve_kv_for={reserved} draft tokens")


def prepare_draft_tokens(req: Request) -> list[int]:
    # 示例：真实系统里这里会调用 drafter
    return [101, 102, 103][: req.max_draft_len]


def longest_prefix_accepted(draft_tokens: list[int], target_outputs: list[int]) -> int:
    accepted = 0
    for d, t in zip(draft_tokens, target_outputs):
        if d == t:
            accepted += 1
        else:
            break
    return accepted


def run_target_model_verify(req: Request) -> list[int]:
    # 示例：假设前两个 token 被接受，第三个不接受
    return [101, 102, 999]


def main() -> None:
    requests = [Request("req-1", max_draft_len=3)]

    # 第一步：调度前先声明“本轮最多要多少草稿 token”
    for req in requests:
        req.py_draft_tokens = [0] * req.max_draft_len

    schedule_and_allocate_kv(requests)

    # 第二步：真正准备草稿 token
    for req in requests:
        req.py_draft_tokens = prepare_draft_tokens(req)

    # 第三步：目标模型验证
    for req in requests:
        target_outputs = run_target_model_verify(req)
        accepted = longest_prefix_accepted(req.py_draft_tokens, target_outputs)
        rewind = len(req.py_draft_tokens) - accepted
        req.py_rewind_len = rewind

        print(f"[verify] req={req.request_id}")
        print("  draft_tokens =", req.py_draft_tokens)
        print("  target_out   =", target_outputs)
        print("  accepted     =", accepted)
        print("  rewind_len   =", req.py_rewind_len)


if __name__ == "__main__":
    main()
```

这段伪代码表达的是机制，不是框架 API 的一比一映射。它强调两个实现事实：

| 字段 | 作用 | 为什么重要 |
| --- | --- | --- |
| `py_draft_tokens` | 告诉调度器本轮草稿 token 预算 | 决定 KV 预留是否足够 |
| `py_rewind_len` | 告诉系统需要回收多少草稿 token 的 KV | 决定拒绝后显存能否及时释放 |

如果只做“最长前缀接受”，却不做“拒绝后的 KV 回收”，系统在小样本测试里可能看起来没问题，但一旦 batch 拉高、上下文变长，很容易因为无效草稿 token 的缓存没有释放而把显存推满。

---

## 工程权衡与常见坑

speculative decoding 的难点主要不在论文推导，而在工程账是否划算。

先把最常见的问题放进一个表里：

| 问题描述 | 影响 | 建议设置 |
| --- | --- | --- |
| `alpha` 很低但 `gamma` 设很大 | 草稿与验证都在浪费算力，吞吐可能下降 | 先测接受率，再决定 `gamma`，常从 2 或 3 起 |
| `max_draft_len` 过大 | KVCache 预留页过多，易 OOM | 单卡先保守设 2 到 4，观察缓存水位 |
| speculative 与 pipeline parallel 混用 | 某些框架版本可能不兼容甚至 hang | speculative 场景下将 pipeline parallel 设为 1 |
| 草稿模型过大 | 占用显存，压缩 batch 空间 | 草稿模型要明显小于主模型 |
| 长上下文接受率下降 | 实际收益低于短上下文基准 | 按任务长度分开压测和配置 |
| 只看平均吞吐，不看 TPOT/P99 | 用户体感可能没有改善 | 同时看吞吐、TPOT、P95/P99 延迟 |

新手最容易误解的一点是：speculative 不是“白赚并行”。它本质上是一笔交易：

$$
\text{额外草稿计算} + \text{更复杂调度}
\quad \Longrightarrow \quad
\text{更少的主模型前向轮数}
$$

只有当右边节省下来的成本大于左边新增的成本时，收益才成立。

可以用下面这张判断表快速建立直觉：

| 场景 | 结果倾向 |
| --- | --- |
| 主模型很大、草稿模型很小、接受率高 | 很可能有收益 |
| 主模型不大、草稿模型接近主模型规模 | 收益容易被额外开销吃掉 |
| 请求多、GPU 容易空转 | 并行化更有价值 |
| 长上下文、任务分布发散、接受率低 | speculative 收益不稳定 |

一个实用的排查顺序如下：

1. 先把 `max_draft_len` 降到 2。
2. 观察开启 speculative 后的显存占用和 KVCache 水位。
3. 检查 reject 发生后，rewind 是否真的回收了对应 KV。
4. 再看接受率是否长期低于预期，例如低于 0.4。
5. 最后再比较吞吐、TPOT、P99，而不是只看平均 token/s。

这里的顺序有意义。因为很多“speculative 没收益”的案例，根因并不是论文失效，而是：

- 草稿长度设得过大
- KV 预算失控
- 回退实现不完整
- 任务分布与草稿模型不匹配

例如代码生成、长文总结、专业领域问答，它们的 token 分布往往比通用闲聊更尖锐或更发散，草稿模型接受率可能明显下降。此时即便短 benchmark 看起来漂亮，线上混合流量下也未必成立。

---

## 替代方案与适用边界

当 speculative decoding 不合适时，不代表没有别的并行化路径。并行化从来不是只有一条路，而是要看你的目标究竟是：

- 降低单请求延迟
- 提高整机吞吐
- 扩展模型规模
- 改善 GPU 利用率

下面把几个常见方向放在同一张表里：

| 方案 | 目标 | 依赖 | 适用上下文 | 不适用场景 |
| --- | --- | --- | --- | --- |
| Speculative Decoding | 压缩主模型前向轮数 | 草稿模型或 n-gram/EAGLE 机制 | 延迟敏感、主模型很大、能拿到高接受率 | 接受率低、显存很紧 |
| Blockwise Parallel Decoding | 减少解码迭代轮数 | 块预测与评分/回退逻辑 | 短上下文、可容忍更复杂实现 | 超长上下文、评分代价过高 |
| Batch Parallelism | 提高设备利用率 | 调度器、连续 batching、KV 共享 | 多用户 API、高并发服务 | 单请求极低延迟优先 |
| Tensor Parallel / Data Parallel | 扩展模型或吞吐 | 多 GPU 通信 | 大模型部署、稳定服务 | 小模型低并发场景 |

### 批处理并行的真实场景

如果你做的是多用户 API，而不是单用户本地聊天，那么最稳定、最容易落地的收益，很多时候来自 batch parallelism。

它的核心思想非常直接：不要让 GPU 为单个请求的单个 token 空转，而是让多个请求一起进入同一轮 decode。

一个典型服务对比如下：

| 模式 | 做法 | 常见问题 |
| --- | --- | --- |
| 单请求串行解码 | 每个请求单独 1-token-by-1-token 推进 | GPU 利用率低 |
| 连续 batching | 把接近时刻到达的请求拼到同一批次 | 调度更复杂，但吞吐明显改善 |
| batching + speculative | 每个请求先产生 draft token，再统一走验证 | 系统复杂度更高，但上限更高 |

这里要注意一个常见误区：系统整体吞吐提升，不一定意味着单个用户感觉“第一时间响应更快”。因为批处理可能引入排队时间。所以线上指标必须分开看：

| 指标 | 关注点 |
| --- | --- |
| Throughput | 整体每秒处理多少 token |
| TTFT | 首 token 时间是否变差 |
| TPOT | 每个输出 token 的平均时间 |
| P95 / P99 | 高峰和长尾是否恶化 |

### 什么时候不该上 speculative

以下场景通常要谨慎：

1. 单卡显存已经非常紧，基线 batch 都难以稳定运行。
2. 任务分布非常发散，草稿模型接受率长期偏低。
3. 你优先关心服务稳定性，而不是追求极限吞吐。
4. 当前框架版本对 quantization、LoRA、pipeline parallel、speculative 的组合支持还不成熟。
5. 线上流量长度差异很大，导致 KV 预算和批处理调度经常抖动。

换句话说，speculative decoding 不是一个“默认就该打开”的优化项。它更像是一个需要经过以下步骤的工程决策：

1. 测接受率
2. 测显存水位
3. 测吞吐与 TPOT
4. 测 P95 / P99
5. 再决定是否上线，以及上线时用什么 `gamma`

---

## 参考资料

- BentoML, *Speculative decoding | LLM Inference Handbook*  
  https://bentoml.com/llm/inference-optimization/speculative-decoding

- vLLM Docs, *Speculative Decoding*  
  https://docs.vllm.ai/en/v0.10.1/features/spec_decode.html

- NVIDIA TensorRT-LLM Docs, *Speculative Decoding*  
  https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html

- Mitchell Stern, Noam Shazeer, Jakob Uszkoreit, *Blockwise Parallel Decoding for Deep Autoregressive Models*, NeurIPS 2018  
  https://papers.nips.cc/paper/8212-blockwise-parallel-decoding-for-deep-autoregressive-models

- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023  
  https://proceedings.mlr.press/v202/leviathan23a.html

- Xia et al., *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding*  
  https://arxiv.org/abs/2401.07851
