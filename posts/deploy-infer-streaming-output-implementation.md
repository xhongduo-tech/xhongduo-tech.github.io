## 核心结论

流式输出的本质，不是“把字符串一点点发出去”，而是把一次完整推理拆成两个体验阶段：先尽快完成 `prefill` 并产出首个可见 Token，再让 `decode` 在后台持续推进，把后续 Token 逐步送到客户端。`prefill` 是“把整段输入先过一遍模型并建立上下文状态”的阶段，白话说，就是先把题目读完；`decode` 是“基于已有状态一个个往后写”的阶段，白话说，就是边想边写答案。

对用户来说，真正敏感的不是总耗时，而是三个指标：

| 指标 | 含义 | 用户感知 | 常见优化 |
| --- | --- | --- | --- |
| TTFT | Time to First Token，请求到首 Token 的时间 | 是否“立刻有反应” | Chunked Prefill、Prefix Cache、路由到热缓存副本 |
| TPOT | Time Per Output Token，后续每个 Token 的间隔 | 输出是否平滑 | 高效注意力、连续批处理、减少 decode 抢占 |
| TPS | Tokens Per Second，系统单位时间总输出 Token 数 | 系统吞吐能力 | Continuous Batching、PagedAttention、量化 |

流式输出为什么能明显改善体验？因为它把“首屏可见”从“整段回答完成后”提前到了“首个 Token 完成后”。数学上可以粗略写成：

$$
TTFT \approx T_{prefill} + T_{first\ decode}
$$

$$
TPOT \approx \frac{T_{decode\ tail}}{N_{tail}}
$$

$$
TPS \approx \frac{N_{all\ tokens}}{T_{total}}
$$

这三个指标经常互相拉扯。只盯吞吐做大批处理，可能让 TTFT 变差；只盯 TTFT 强行抢占新请求，可能让正在流式返回的请求变卡。工程上正确的目标不是“某一个指标绝对最优”，而是在 SLA 约束下同时控制 TTFT、TPOT 和 TPS。

一个玩具例子是聊天框。用户发出问题后，界面先出现“正在生成”，几十到几百毫秒后开始逐词刷新，这就是流式输出。一个真实工程例子是 Ray Serve + vLLM + AKS：AKS 管基础设施，Ray Serve 管请求路由、伸缩和流式响应，vLLM 管连续批处理、KV Cache 和 PagedAttention。三层分工清楚，系统才容易调优。

---

## 问题定义与边界

本文讨论的“流式输出”，边界是在线推理服务，不是训练，也不是离线批量生成。这里的目标是：模型还在继续计算时，服务端已经把已生成的 Token 通过 SSE 或 WebSocket 发给前端。

SSE 是 Server-Sent Events，白话说，就是服务端可以沿着一条 HTTP 连接不断往前端推文本事件。WebSocket 是双向长连接，白话说，就是前后端都能持续互发消息。对纯文本生成，SSE 通常更简单；对语音、协作编辑、工具调用状态同步，WebSocket 更灵活。

这个问题有两个硬边界。

第一是算力边界。请求到来速率记为 $\lambda$，系统稳定服务率记为 $\mu$。如果

$$
\lambda < \mu
$$

队列通常可稳定；如果

$$
\lambda \ge \mu
$$

等待队列会持续增长，P99 延迟会恶化，最终可能触发超时、取消、OOM。这里的 $\mu$ 不是一个固定常数，它会被模型大小、prompt 长度、输出长度、批处理策略和 GPU 内存一起影响。

第二是 KV Cache 边界。KV Cache 是注意力层为后续 Token 复用历史计算结果而保存的键值状态，白话说，就是为了“不用把前文每次重算一遍”而保留的中间结果。它会随上下文和输出长度增长而线性变大。长 prompt、大并发、长回复叠加时，瓶颈经常先出在 KV Cache，而不是算力本身。

因此，流式输出不是“前端改成逐字显示”这么简单。真正的工程问题是：在有限 GPU 内存和队列稳定性约束下，如何让新请求尽快见到第一个 Token，同时不把已经在流式发送的老请求卡住。

---

## 核心机制与推导

先看最小闭环。

1. 请求进入服务。
2. 模型执行 prefill，读完整个 prompt，建立 KV Cache。
3. 开始第一轮 decode，得到首个 Token。
4. 服务端立刻把首个 Token 发送给客户端。
5. 后续 decode 每迭代一次，就继续发送一个或多个 Token。
6. 直到生成结束，发送结束标记。

问题出在第 2 步和第 5 步的资源特征不同。`prefill` 通常更偏计算密集，`decode` 更偏内存带宽密集。如果把长 prompt 的 prefill 和大量正在进行的 decode 粗暴混在一起，新的大请求会堵住正在输出的小请求，用户看到的现象就是“本来在流，突然卡一下”。

所以现代推理引擎会做两类关键优化。

第一类是 `Chunked Prefill`。它把一次很长的 prefill 切成小块，与 decode 交替执行。白话说，就是不要让一个长题目一次把整张卡占满，而是分几口读，每读一段就让正在回答别人的请求继续吐几个词。这样做的直接收益是平滑 TPOT，并降低长 prompt 对全局流式体验的破坏。

第二类是 `PagedAttention`。它把 KV Cache 按固定大小分页管理，像操作系统分页一样按需分配，而不是给每个请求预留一大块连续显存。白话说，就是不再“先圈一整块大房间等着以后可能用”，而是“缺多少页给多少页”。这减少了碎片和过度预留，使同一块 GPU 能容纳更多并发请求。公开资料常把它描述为把 KV Cache 浪费从很高的碎片率压到很低水平，因此吞吐能显著提升，但具体提升倍数和 req/s 只能按模型、卡型、上下文长度分别测，不能拿单一数字直接外推。

玩具例子可以这样理解。假设有两个请求：

| 请求 | prompt 长度 | 输出长度 | 不做 Chunked Prefill | 做 Chunked Prefill |
| --- | --- | --- | --- | --- |
| A | 很长 | 中等 | A 的 prefill 独占 GPU，B 的流式输出被打断 | A 分块预填，B 还能继续流 |
| B | 很短 | 很长 | B 的 TTFT 和 TPOT 都受 A 影响 | B 更容易保持平滑输出 |

从推导上看，流式输出的核心并不是“网络边传边收”，而是调度把用户感知拆开了。总延迟仍然存在，但用户只先为 TTFT 付费，后面再按 TPOT 分期付款。一个简化的调度目标可以写成：

$$
\min \; \alpha \cdot TTFT_{p95} + \beta \cdot TPOT_{p95} - \gamma \cdot TPS
$$

其中 $\alpha,\beta,\gamma$ 是业务权重。聊天、代码补全通常更重视 TTFT 和 TPOT；批量摘要更重视 TPS。

长对话还有另一层机制问题。若上下文无限增长，KV Cache 也会无限增长。StreamingLLM 一类方法引入 `attention sink` 和滑动窗口。attention sink 指“始终保留的一小组早期 Token”，白话说，就是即使前文大部分丢了，也留住最能稳定注意力分布的锚点；滑动窗口则只保留最近的一段上下文。这类方法适合超长会话或无限流输入，但它已经不只是“把结果流式发出”，而是连模型内部上下文管理也改了，复杂度明显更高。

---

## 代码实现

下面先给一个能运行的 Python 玩具实现，模拟“prefill 后先返回首 Token，再持续流式输出”，同时用 `assert` 验证顺序与指标计算。

```python
import asyncio
import time

class FakeModel:
    def __init__(self, prefill_delay=0.05, decode_delay=0.02):
        self.prefill_delay = prefill_delay
        self.decode_delay = decode_delay

    async def stream_generate(self, prompt: str):
        # prefill: 先“读完整个 prompt”
        await asyncio.sleep(self.prefill_delay)

        # toy tokenizer
        tokens = ["流式", "输出", "可以", "先回首词", "再继续", "推送"]
        for token in tokens:
            await asyncio.sleep(self.decode_delay)
            yield token

async def collect():
    model = FakeModel()
    start = time.perf_counter()
    first_token_at = None
    out = []

    async for token in model.stream_generate("什么是流式输出"):
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        out.append(token)

    end = time.perf_counter()

    ttft = first_token_at - start
    total = end - start
    tpot = (end - first_token_at) / (len(out) - 1)
    tps = len(out) / total

    assert out[0] == "流式"
    assert "".join(out).startswith("流式输出")
    assert ttft > 0
    assert tpot > 0
    assert tps > 0
    return ttft, tpot, tps, out

if __name__ == "__main__":
    ttft, tpot, tps, out = asyncio.run(collect())
    print("TTFT:", round(ttft, 4))
    print("TPOT:", round(tpot, 4))
    print("TPS:", round(tps, 2))
    print("TEXT:", "".join(out))
```

上面的代码没有真的跑 LLM，但把流式接口的时序关系说清楚了：先有 prefill 延迟，再有首 Token，再有持续输出。

真实工程里，后端通常这样组织。FastAPI 负责 HTTP 接口，`StreamingResponse` 负责把生成器产出的事件逐条推给前端。关键点有三个。

1. 返回类型要是 `text/event-stream`，否则浏览器不会按 SSE 处理。
2. 生成器要边生成边 `yield`，不能先把整段内容拼完。
3. 客户端要处理结束标记、断线重连和中途取消。

一个简化实现如下：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def fake_llm(prompt: str):
    await asyncio.sleep(0.1)  # prefill
    for token in ["data", ":", " hello", " world"]:
        await asyncio.sleep(0.03)  # decode
        yield token

@app.get("/stream")
async def stream(prompt: str):
    async def event_gen():
        async for token in fake_llm(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

前端如果用浏览器原生 SSE，逻辑可以非常直接：建立 `EventSource`，每收到一条事件就把文本拼到页面上。真实项目里，再补上 markdown 增量渲染、光标动画、取消按钮即可。

一个真实工程例子是企业内聊天助手。入口在 API 网关之后，Ray Serve 负责副本伸缩、排队和流式响应，vLLM 负责连续批处理、Prefix Cache、Chunked Prefill、PagedAttention。这样做的原因不是“组件流行”，而是职责边界清晰：平台层处理资源，服务层处理路由，推理层处理 Token 级调度。

---

## 工程权衡与常见坑

流式输出最常见的误判，是把问题理解成“网络传输优化”。实际上，网络通常不是主瓶颈，调度和内存才是。

| 选项 | 优点 | 风险/坑 |
| --- | --- | --- |
| 高并发 prefill | 能让更多请求更快开工 | 长 prompt 容易挤压正在 decode 的流 |
| 粗粒度批处理 | TPS 高 | TTFT、TPOT 变差，流式感消失 |
| PagedAttention | 显存利用率高，并发更稳 | 只能缓解分配问题，不能消灭 KV 总量增长 |
| Prefix Cache | 共享系统提示词时很有效 | 路由不一致会让命中率很差 |
| Attention Sink + 滑窗 | 支持超长对话 | 实现复杂，语义一致性要单独验证 |

几个典型坑需要单独说。

第一，SSE 通了不等于流式做好了。很多服务端虽然用了 `StreamingResponse`，但模型层仍然同步阻塞，直到攒够一大段才吐一次。这种实现从协议看是流式，从体验看不是流式。

第二，长 prompt 会伤害别人。没有 Chunked Prefill 时，一个超长输入的 prefill 可能让整个 GPU 上所有 decode 请求都抖一下，P99 TPOT 会突然升高。

第三，取消和断连处理常被忽略。用户关掉页面或点击停止后，如果服务端还继续生成，GPU 时间就白白浪费。真实系统必须把前端取消信号传到推理层。

第四，增量渲染会破坏格式。Markdown 表格、代码块、LaTeX 在半截文本状态下经常不是合法结构。如果前端每个 Token 都全量重新解析，性能和闪烁都会有问题。常见做法是文本层即时显示，结构化渲染按句子、按块或按节流周期更新。

第五，长会话漂移。单纯做流式传输并不能解决上下文无限增长问题；如果不做窗口化、摘要化或 attention sink，KV Cache 会持续膨胀，最终要么 OOM，要么被迫截断，导致回答前后不一致。

---

## 替代方案与适用边界

不是所有场景都该做流式输出。若业务本身不依赖交互感，静态批量响应更简单。

| 方案 | 适用边界 | 注意事项 |
| --- | --- | --- |
| 同步批量返回 | 批量摘要、离线改写、文件生成 | 实现简单，但 TTFT 高 |
| SSE 流式返回 | 聊天、问答、代码补全 | 单向推送简单，浏览器支持好 |
| WebSocket 流式返回 | 语音、多模态、协同编辑 | 双向能力强，但状态管理更复杂 |
| StreamingLLM 类架构 | 超长上下文、无限流输入 | 需要专门设计上下文保留策略 |

如果你的场景是“生成一篇完整报告后下载”，用户并不盯着屏幕等首词，流式输出带来的收益有限，反而会增加前后端复杂度。如果你的场景是对话、Copilot、搜索问答，TTFT 直接决定用户是否觉得系统“卡”，流式输出通常值得做。

还有一条边界要明确：流式输出不等于低成本。它改善的是感知延迟，不是免费减少计算。总 Token 没变时，总算力消耗并不会因为“边算边发”自动下降。真正降低成本，要靠量化、Prefix Cache、连续批处理、投机解码等别的手段。

因此，选择方案的顺序通常是：

1. 先判断业务是否真的需要低 TTFT。
2. 需要的话，先上协议层流式能力。
3. 然后再补推理层调度优化，如 Chunked Prefill、PagedAttention、Prefix Cache。
4. 只有在长上下文成为主问题时，再考虑 StreamingLLM 一类上下文保持机制。

---

## 参考资料

1. CSDN，《大模型流式输出的7种核心方法》：<https://blog.csdn.net/2201_75798391/article/details/146385724>  
2. Microsoft Tech Community, *The LLM Inference Optimization Stack: A Prioritized Playbook for Enterprise Teams*：<https://techcommunity.microsoft.com/blog/appsonazureblog/the-llm-inference-optimization-stack-a-prioritized-playbook-for-enterprise-teams/4498818>  
3. Zylos Research, *LLM Inference Optimization and Quantization 2026*：<https://zylos.ai/research/2026-01-15-llm-inference-optimization>  
4. Emergent Mind, *StreamingLLM*：<https://www.emergentmind.com/topics/streamingllm>  
5. Emergent Mind, *StreamingLLM Framework*：<https://www.emergentmind.com/topics/streamingllm-framework>  
6. OpenReview, *A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints*：<https://openreview.net/forum?id=AWLJJRgvbA>
