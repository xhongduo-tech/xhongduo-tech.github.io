## 核心结论

推理框架性能对比的本质，不是看谁某一次跑得更快，而是在**同模型、同量化、同上下文、同负载**下，比较谁的**调度策略、KV 缓存管理和 runtime 路径**更适合目标场景。

这里先给三个直接结论。

| 框架 | 主要定位 | 强项 | 主要代价 |
|---|---|---|---|
| vLLM | 高并发在线推理 | 吞吐高、显存利用率高、OpenAI 兼容接口成熟 | 不一定是单请求极限低延迟最优 |
| TGI | 标准化服务化部署 | API、流式输出、监控、路由与平台接入顺手 | 官方当前已标记 maintenance mode，长期演进优先级弱于新路线 |
| TensorRT-LLM | NVIDIA 硬件特化优化 | 固定硬件上容易压低延迟、榨干 GPU 性能 | engine 构建复杂，版本与环境耦合更强 |

如果只想记一句话，可以记这句：

**vLLM 更像“把显存和批处理用到更满”的推理引擎，TGI 更像“服务接口和运维能力更完整”的推理服务，TensorRT-LLM 更像“为 NVIDIA 环境做硬件特化编译”的性能方案。**

“调度”这个词第一次出现时可以把它理解成：**系统决定哪些请求先算、哪些请求一起算、显存怎么分配的规则**。

---

## 问题定义与边界

先定义什么叫“公平比较”。如果下面任一项不一致，结论就容易失真。

| 维度 | 必须统一的内容 | 不统一会导致什么问题 |
|---|---|---|
| 模型 | 同一权重、同一 tokenizer | 不同模型本身就可能速度不同 |
| 精度/量化 | FP16、BF16、INT8、AWQ、GPTQ 等必须一致 | 量化差异会直接改变吞吐和显存 |
| 上下文 | prompt 长度、max tokens、总上下文窗口 | 长上下文会显著放大 KV cache 压力 |
| 负载形态 | batch、并发数、到达方式 | 单请求和高并发是两类问题 |
| 环境 | GPU 型号、显存、CUDA、驱动、通信库 | 底层算子路径可能完全不同 |
| 采样参数 | temperature、top_p、top_k、beam size | 生成策略会改变 decode 开销 |
| 预热 | warmup 次数、是否清缓存 | 冷启动和热态差异很大 |

很多 benchmark 没参考价值，不是因为数据造假，而是因为它回答的是另一个问题。比如：

- A 框架用 FP16，B 框架用 INT8
- A 跑 batch=1，B 跑并发 32
- A 用 128 token prompt，B 用 8k prompt

这时即使 B 的 `token/s` 更高，也不能推出“B 一定更强”。它只能说明：**在那组测试条件下，B 更适合那种负载。**

指标也不能只看一个。最少要拆成下面三类：

$$TTFT = T_{queue} + T_{prefill}$$

这里 `TTFT` 是 first token time，白话说就是**用户等到第一个字出来用了多久**。  
`T_queue` 是排队时间，`T_prefill` 是预填充时间，白话说就是**模型先把整段输入读一遍并建立上下文状态的时间**。

$$Throughput = \frac{\sum L_g}{T_{wall}}$$

这里 `Throughput` 是吞吐，白话说就是**单位总时间内一共生成了多少 token**。  
`L_g` 是每个请求生成的 token 数，`T_wall` 是总墙钟时间，也就是实际经过的真实时间。

---

## 核心机制与推导

理解三种框架差异，先看 KV cache。

“KV cache”第一次出现时可以把它理解成：**模型为了继续往后生成，需要把前文计算出的注意力记忆先存起来**。这部分通常占显存，而且会随着上下文和并发增加而快速膨胀。

常见近似公式是：

$$M_{kv} = 2 \cdot B \cdot (L_p + L_g) \cdot N \cdot H \cdot d \cdot s$$

变量含义如下。

| 符号 | 含义 |
|---|---|
| $B$ | 并发请求数 |
| $L_p$ | prompt token 数 |
| $L_g$ | 生成 token 数 |
| $N$ | 层数 |
| $H$ | 注意力头数 |
| $d$ | 每头维度 |
| $s$ | 单元素字节数 |

这个式子最重要的结论只有两个。

1. 当并发 $B$ 增大时，KV cache 近似线性增长。
2. 当上下文 $(L_p + L_g)$ 变长时，KV cache 也近似线性增长。

所以很多在线服务不是“算不动”，而是“先放不下”。显存先成为瓶颈，而不是算力先成为瓶颈。

### 玩具例子

设一个模型有：

- $N=32$
- $H=32$
- $d=128$
- $s=2$ 字节，近似对应 FP16/BF16 大小
- 单请求 $B=1$
- prompt 长度 $L_p=4096$
- 先不算生成长度，令 $L_g=0$

代入公式：

$$M_{kv}=2 \cdot 1 \cdot 4096 \cdot 32 \cdot 32 \cdot 128 \cdot 2 \approx 2\ \text{GiB}$$

只是一条 4k prompt，请求还没开始持续生成，就可能已经吃掉约 2 GiB KV cache。  
如果并发变成 8，请求条件不变，KV cache 直接接近 16 GiB。再加模型权重、激活、运行时开销，显存很快吃紧。

这就是为什么长上下文和高并发测试，往往比短 prompt、小 batch 更能拉开框架差异。

### 三个框架的机制重点

**vLLM**

`PagedAttention` 可以理解成：**把 KV cache 按块分页管理，而不是给每个请求预留一整段大连续空间**。这样做的核心收益不是魔法加速，而是减少碎片和过度预留，让更多请求能同时驻留在显存中。请求驻留更多，continuous batching 才更容易把 GPU 喂满。

**TGI**

TGI 的重点是 **continuous batching + 服务化能力**。continuous batching 可以理解成：**不是等一整批凑齐再一起算，而是把不断到来的请求动态插进正在运行的批次里**。这样吞吐通常更高，也更适合 API 服务。但它的价值不只是算得快，还在于流式输出、监控、路由、平台化接入这些服务侧能力。

**TensorRT-LLM**

TensorRT-LLM 的重点是 **先构建针对硬件和模型配置优化过的 engine，再用专门 runtime 执行**。这里的 “engine” 可以理解成：**已经为目标 GPU、精度、并行方式编译好的执行计划**。它通常能更充分利用 NVIDIA 硬件特性，但前提是你愿意接受构建、调参和环境锁定成本。

### 真实工程例子

假设你有一套 8 卡 A100 80G 的在线聊天服务，模型是 7B 或 8B，用户输入在 2k 到 8k 之间波动，晚高峰会突然放大并发。

这个场景里，核心问题通常不是某条请求的理论峰值速度，而是：

- 高峰时能不能多接住请求
- TTFT 会不会因为排队突然变差
- 长 prompt 会不会把显存顶满
- 框架升级是否会引入部署和兼容性成本

这时优先级通常变成：

- 要扛并发和显存效率，先看 vLLM
- 要服务化接口、监控和标准部署链路，TGI 上手更直观
- 要在固定 NVIDIA 集群里压极限延迟和吞吐，TensorRT-LLM 更值得投入

---

## 代码实现

比较框架时，代码的第一原则不是“快点跑起来”，而是“把输入条件钉死”。

下面这个 Python 例子不直接调用真实框架，而是先把**统一统计口径**建立出来。它可以运行，也能帮助你验证指标计算逻辑。

```python
from dataclasses import dataclass

@dataclass
class RequestResult:
    queue_s: float
    prefill_s: float
    decode_s: float
    generated_tokens: int

    @property
    def ttft_s(self) -> float:
        return self.queue_s + self.prefill_s

    @property
    def tpot_s(self) -> float:
        # time per output token
        return self.decode_s / self.generated_tokens

def throughput(results, wall_s: float) -> float:
    return sum(r.generated_tokens for r in results) / wall_s

# 玩具测试数据：假设 3 个请求在某框架下的结果
results = [
    RequestResult(queue_s=0.03, prefill_s=0.20, decode_s=1.28, generated_tokens=128),
    RequestResult(queue_s=0.05, prefill_s=0.22, decode_s=1.30, generated_tokens=128),
    RequestResult(queue_s=0.02, prefill_s=0.18, decode_s=1.26, generated_tokens=128),
]

wall_s = 1.90

avg_ttft = sum(r.ttft_s for r in results) / len(results)
avg_tpot = sum(r.tpot_s for r in results) / len(results)
tp = throughput(results, wall_s)

assert round(avg_ttft, 3) == round((0.23 + 0.27 + 0.20) / 3, 3)
assert round(avg_tpot, 4) == round(((1.28/128) + (1.30/128) + (1.26/128)) / 3, 4)
assert tp == 384 / 1.90

print("avg_ttft_s =", round(avg_ttft, 4))
print("avg_tpot_s =", round(avg_tpot, 6))
print("throughput_tok_per_s =", round(tp, 2))
```

如果把它扩成真实 benchmark，关键是三件事。

1. 固定 prompt 长度，而不是随便拿几条请求。
2. 固定输出长度和采样参数，而不是让不同框架生成不同长度。
3. 同时记录 `TTFT`、`TPOT`、吞吐、峰值显存，而不是只记 `token/s`。

一个统一请求模板可以长这样：

```python
request = {
    "prompt": fixed_prompt,
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": True,
}
```

真实测试结果建议至少整理成这张表。

| 框架 | TTFT(ms) | TPOT(ms/token) | 吞吐(tok/s) | 峰值显存(GB) | 启动或构建耗时 |
|---|---:|---:|---:|---:|---:|
| vLLM |  |  |  |  |  |
| TGI |  |  |  |  |  |
| TensorRT-LLM |  |  |  |  |  |

这里特别要强调一个初学者常犯错：  
**单请求压测脚本，不等于在线服务压测。**  
单请求测到的是更偏内核与路径效率；并发压测测到的才是调度、显存管理和服务化开销的综合结果。

---

## 工程权衡与常见坑

实际选型不能只看跑分，因为部署系统不是一张静态图，而是一组会变的约束。

### 常见坑

| 常见坑 | 为什么会误导 |
|---|---|
| 只换框架，不统一模型与 tokenizer | 你测到的是模型差异，不是框架差异 |
| 只看平均 `token/s` | 会把首字延迟和排队问题藏起来 |
| 忽略 warmup | 冷启动抖动会污染结果 |
| 用短 prompt 代表长上下文 | 长上下文才更容易暴露 KV cache 问题 |
| 忽略峰值显存 | 线上能不能稳定跑，先看放不放得下 |
| 不计入 engine build 成本 | TensorRT-LLM 的总成本会被低估 |
| 只测 batch=1 | 得不到高并发场景结论 |

### 指标怎么解读

- `TTFT` 适合看首字延迟，用户体感最直接。
- `TPOT` 适合看生成阶段效率，能反映 decode 阶段是否流畅。
- 吞吐适合看整体服务能力，尤其是高并发时的承载上限。

一个典型误区是：某框架在 `prompt=128, batch=1` 时非常快，于是被直接拿去做 8k 上下文、并发 16 的聊天服务。结果不是 TTFT 飙升，就是显存爆掉。这个结果不代表它“差”，只代表它的最佳区间和你的负载区间不重合。

另一个现实问题是部署成本。

- vLLM 的优势常常体现在较好的显存利用和高并发吞吐。
- TGI 的优势常常体现在服务接口和平台化接入。
- TensorRT-LLM 的优势常常体现在固定 NVIDIA 环境下的极限优化。

但如果你的团队没有足够强的 GPU 编译、内核、版本管理经验，TensorRT-LLM 的收益可能会被构建复杂度抵消。性能不是单点峰值，而是**目标部署条件下的综合结果**。

---

## 替代方案与适用边界

不是所有场景都必须在 vLLM、TGI、TensorRT-LLM 三者里选。

| 场景 | 更常见选择 |
|---|---|
| 在线聊天，高并发，长上下文 | vLLM |
| 平台 API 服务、监控、标准化接入 | TGI |
| 固定 NVIDIA 集群、追求极致优化 | TensorRT-LLM |
| 低频内部工具、QPS 很低 | 更简单的本地或轻量服务方案也够 |
| 本地开发、边缘设备、非 NVIDIA 环境 | 可能优先考虑 llama.cpp、MLX、SGLang 等其他方案 |

边界要说清楚：**框架选择不等于模型能力选择。**  
框架改变的是运行方式、调度策略、显存利用和服务能力，不会把一个 7B 模型“变成”70B 的知识水平。

对于零基础到初级工程师，最实用的选型思路是：

1. 先确认你的目标是低延迟、扛并发，还是方便部署。
2. 再确认你的硬件是不是固定在 NVIDIA 生态。
3. 最后用统一条件做小规模压测，不要直接相信别人的排行榜。

如果你的业务只是内部问答工具，每分钟只有几次调用，那么复杂框架带来的收益可能很小。  
如果你做的是高并发线上聊天服务，那么只看一次“最高 `token/s`”几乎没有意义，真正重要的是高峰时系统是否稳定、显存是否可控、TTFT 是否还能接受。

---

## 参考资料

1. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm)
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
3. [vLLM Documentation: Paged Attention](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
4. [Hugging Face Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference/main/index)
5. [Hugging Face Text Generation Inference Repository](https://github.com/huggingface/text-generation-inference)
6. [NVIDIA TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/index.html)
7. [NVIDIA TensorRT-LLM Repository](https://github.com/NVIDIA/TensorRT-LLM)
