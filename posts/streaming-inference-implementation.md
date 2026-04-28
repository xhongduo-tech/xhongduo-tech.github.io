## 核心结论

流式推理的核心不是“把一段完整结果拆成很多小段返回”，而是让用户更早看到第一个 token，并让后续 token 以稳定节奏持续出现。token 可以理解成模型生成时的最小文本单位，中文里常接近一个字、词或一个短片段。对用户来说，体验差异主要来自两个指标：

1. 首 token 时间，也就是从发起请求到看到第一个可见字符的时间。
2. token 间隔，也就是后续内容一段一段出现时，相邻两次可见更新之间的时间。

可以把它写成一个工程抽象：

$$
\text{流式推理} = \text{token 生成后尽快可见}
$$

很多系统“看起来在流式”，实际只是“分段返回”。比如服务端先攒 50 个 token 再统一发一次，用户仍然会看到明显停顿。真正有效的流式，一般不是缩短总生成时长，而是把等待从“全黑屏”变成“尽快开始、持续推进”。

下面这个表格先建立直觉：

| 方案 | 总耗时 | 首 token 时间 | token 间隔 | 用户体感 |
| --- | --- | --- | --- | --- |
| 非流式一次性返回 | 可能相近 | 很长 | 不适用 | 前面一直空白，最后一次性出现 |
| 伪流式分段返回 | 可能相近 | 中等 | 抖动大 | 过几秒跳一大段 |
| 真正流式返回 | 可能相近 | 短 | 稳定 | 很快开始，内容连续长出来 |

结论可以压缩成一句话：流式体验好不好，主要看首字出现得够不够快，以及后面是不是稳定连续。

---

## 问题定义与边界

流式推理解决的是“可见性延迟”问题。可见性延迟的意思是：模型内部可能已经开始生成，但用户还看不到。这个问题不只发生在模型层，而是发生在整条链路上：

`模型 -> 服务端 -> 代理 -> 浏览器`

只要其中任意一层缓存、排队、阻塞，流式就会退化。

先看一个玩具例子。模型每 30 毫秒能产出一个 token，本来应该持续输出。但服务端代码把 token 先放进列表，累计 1 秒才写一次 HTTP 响应，浏览器就会每秒看到一大段，而不是连续出现。这里模型并不慢，慢的是“暴露给用户”的过程。

这也是为什么“模型支持逐 token 生成”不等于“用户一定能看到流式”。边界要拆清楚：

| 问题位置 | 典型现象 | 本质问题 |
| --- | --- | --- |
| 模型计算层 | 首字很慢 | prefill 重，算第一个 token 前准备时间长 |
| 服务端转发层 | 明明在生成，但前端不动 | 没有及时写出或 flush |
| 代理层 | 每隔几秒冒一大段 | 反向代理缓存或缓冲输出 |
| 连接层 | 高并发下越来越慢 | 长连接占满，排队变长 |
| 前端层 | 数据到了但页面卡顿 | 渲染频率过高或 DOM 更新太碎 |

这里的 flush 可以白话理解成“把已经写进程序缓冲区的数据真正推出去”。`yield` 只是程序内部产出数据，`flush` 才决定网络另一端何时真的收到。

首 token 时间也不能只看模型。更准确的拆法是：

$$
T_{\text{first}} = T_{\text{conn}} + T_{\text{queue}} + T_{\text{prefill}} + T_{\text{flush}} + T_{\text{decode}}(1)
$$

含义分别是连接建立、排队等待、prefill、刷出网络、第一个 decode 步。prefill 可以理解成“模型先读完整个输入，把后续生成需要的上下文状态准备好”。

所以，流式推理的边界不是“服务端有没有循环发数据”，而是“第一个 token 和后续 token 是否能端到端稳定可见”。

---

## 核心机制与推导

大语言模型生成通常分成两段：prefill 和 decode。

prefill 是把整段输入一次性送进模型，得到后续生成所需的内部状态。它像“先把题目通读一遍并记到草稿纸上”。decode 是后续一步一步生成 token，每一步都依赖前一步结果。对用户体感最强的，是 prefill 结束后第一个 token 何时出现。

因此，首 token 时间公式写成：

$$
T_{\text{first}} = T_{\text{conn}} + T_{\text{queue}} + T_{\text{prefill}} + T_{\text{flush}} + T_{\text{decode}}(1)
$$

再看后续 token。第 $i$ 个 token 的可见间隔，不只由模型单步速度决定，还受调度和传输影响：

$$
\Delta t_{\text{token}}(i) \approx \max(T_{\text{step\_model}}(i), T_{\text{sched}}(i), T_{\text{flush}}(i))
$$

这里的调度可以理解成“系统决定什么时候轮到这个请求继续往前走”。如果 GPU 上多个请求共享执行机会，或者服务端事件循环被别的任务拖住，即使模型本身不慢，用户看到的间隔也会变长。

看一个数值玩具例子：

- $T_{\text{conn}} = 50\text{ms}$
- $T_{\text{queue}} = 100\text{ms}$
- $T_{\text{prefill}} = 900\text{ms}$
- $T_{\text{flush}} = 20\text{ms}$
- $T_{\text{decode}}(1) = 30\text{ms}$

那么：

$$
T_{\text{first}} = 50 + 100 + 900 + 20 + 30 = 1100\text{ms}
$$

这说明用户大约 1.1 秒后看到第一个字。若后续每个 token 平均 30ms，可粗略看作约 33 tok/s；如果并发上来后调度变成 80ms 一次，用户就会明显感到“卡一下再冒一点”。

再看队列。队列是“暂时等着被处理的数据或请求集合”。在高并发下，哪怕单个请求逻辑正确，系统仍可能因为积压而整体退化。一个常见抽象是：

$$
Q(t) = \max(0, Q(t-1) + \lambda_{in} - \mu_{out})
$$

其中 $\lambda_{in}$ 是输入速率，$\mu_{out}$ 是输出速率。当 $\lambda_{in} > \mu_{out}$ 时，队列会持续增长。增长带来的不是线性变慢，而是连锁放大：

1. 排队更长，首 token 更慢。
2. 每个连接活得更久，占更多内存和文件描述符。
3. 发送缓冲更多，flush 更慢。
4. 断连未回收时，GPU 和 KV cache 继续被浪费。

可以把各层职责总结成一张表：

| 层次 | 主要职责 | 影响指标 |
| --- | --- | --- |
| 连接层 | 建连、保活、断连感知 | 首 token 时间、连接稳定性 |
| 调度层 | 排队、公平性、并发限制 | 首 token 时间、token 间隔 |
| 模型层 | prefill、decode、KV cache | 首 token 时间、吞吐 |
| 传输层 | flush、缓冲、背压 | token 间隔、尾部延迟 |
| 前端层 | 增量渲染、取消、错误恢复 | 体感连续性、资源占用 |

真实工程例子通常发生在聊天系统。用户发送一条长问题后，模型端很快进入 decode，但 Nginx 或 CDN 默认开启响应缓冲，导致浏览器每几百毫秒甚至几秒才看到一批内容。此时监控里“模型 token/s”很好看，用户却觉得“不流畅”。问题不在模型，而在链路没打通。

---

## 代码实现

实现流式输出的关键动作不是“生成”，而是“生成后立即写出，并能在客户端断开时尽快停止”。下面用一个可运行的 Python 玩具实现说明核心机制。它不依赖真实网络，但保留了 `generate -> write -> flush -> cancel` 的结构。

```python
import time
from dataclasses import dataclass, field

@dataclass
class StreamSession:
    cancelled: bool = False
    sent: list[str] = field(default_factory=list)
    flush_count: int = 0

    def write(self, chunk: str) -> None:
        self.sent.append(chunk)

    def flush(self) -> None:
        self.flush_count += 1

def fake_model_generate(prompt: str):
    # 玩具模型：把 prompt 拆成字符再追加一个句号
    for ch in (prompt + "。"):
        time.sleep(0.001)  # 模拟 decode
        yield ch

def stream_infer(prompt: str, session: StreamSession, cancel_after: int | None = None) -> str:
    for idx, token in enumerate(fake_model_generate(prompt)):
        if session.cancelled:
            break
        session.write(token)
        session.flush()
        if cancel_after is not None and idx + 1 >= cancel_after:
            session.cancelled = True
    return "".join(session.sent)

# 正常完成
s1 = StreamSession()
out1 = stream_infer("流式推理", s1)
assert out1 == "流式推理。"
assert s1.flush_count == len(out1)

# 中途取消
s2 = StreamSession()
out2 = stream_infer("abcdef", s2, cancel_after=3)
assert out2 == "abc"
assert s2.cancelled is True
assert s2.flush_count == 3
```

如果把它映射到 Web 服务，可以写成 SSE 形式。SSE 是 Server-Sent Events，白话说就是“服务器持续往浏览器单向推送文本事件”的 HTTP 机制。它适合正文流式输出，因为浏览器支持直接、协议简单、代理兼容性通常比 WebSocket 更稳。

下面是简化伪代码：

```python
def sse_handler(request):
    set_header("Content-Type", "text/event-stream")
    set_header("Cache-Control", "no-cache")
    set_header("Connection", "keep-alive")

    session = create_generation_task(request.json)

    try:
        while True:
            if request.client_disconnected():
                session.cancel()
                break

            token = session.next_token(timeout=0.2)
            if token is None:
                send("event: ping\ndata: keepalive\n\n")
                flush()
                continue

            send(f"data: {token}\n\n")
            flush()

            if session.finished():
                send("event: done\ndata: [DONE]\n\n")
                flush()
                break
    finally:
        session.release()
```

一个请求生命周期通常长这样：

| 阶段 | 服务端动作 | 关键点 |
| --- | --- | --- |
| 开始 | 建立会话、校验参数 | 生成请求 ID，记录取消句柄 |
| prefill | 模型读入 prompt | 首 token 时间的主要组成部分 |
| stream tokens | 每拿到 token 就写出并 flush | 不要攒大块再发 |
| cancel/finish | 收到取消或自然结束 | 及时停止 decode |
| 回收资源 | 释放连接、队列项、KV cache | 避免“用户走了，GPU 还在算” |

在真实聊天产品里，常见设计是“正文走 SSE，控制走独立 POST”。比如用户点击“停止生成”，前端发一个 `/cancel` 请求，服务端根据请求 ID 中止对应任务。这样做的原因很直接：正文是单向高频文本流，控制是低频指令流，把两者分开，链路更稳定，也更容易调试。

---

## 工程权衡与常见坑

流式系统真正难的地方，不在 demo 能不能跑，而在高并发下还能不能稳。稳的含义包括：首 token 不劣化太快、连接不会无限堆积、断连能及时回收、发送侧不会因为慢客户端被拖死。

先看常见坑：

| 问题 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 只 `yield` 不 `flush` | 浏览器隔一段时间才更新 | 数据还在缓冲区 | 每个事件后显式 flush |
| 代理缓冲未关闭 | 看到一大段一大段跳出 | 中间层攒包转发 | 关闭响应缓冲，验证端到端可见性 |
| 无心跳 | 长连接随机断开 | LB/代理回收空闲连接 | 周期性发送 ping 事件 |
| 不做断连检测 | 用户走了还继续算 | 后端不知道客户端已断开 | 轮询连接状态并取消任务 |
| 无背压上限 | 内存、队列、连接暴涨 | 慢客户端拖住发送侧 | 限制每连接缓冲和总队列长度 |
| 纯单向场景硬上 WebSocket | 维护复杂度高 | 协议能力超出需求 | 优先 SSE，控制走独立接口 |

背压是“下游处理不过来时，把压力反馈给上游”的能力。它的白话解释是：别让快的一端无上限地往慢的一端塞数据。如果没有背压或至少没有队列上限，系统很容易进入这个状态：

$$
\lambda_{in} > \mu_{out} \Rightarrow Q(t) \uparrow \Rightarrow \text{延迟} \uparrow,\ \text{内存} \uparrow
$$

断连回收尤其关键。一个用户关掉页面后，如果服务端还继续让模型 decode 20 秒，这 20 秒的 GPU、KV cache、带宽都被白白占用。简化伪代码如下：

```python
def stream_loop(request, task):
    while not task.finished():
        if request.client_disconnected():
            task.cancel()
            break
        token = task.next_token()
        send(token)
        flush()
    task.release()
```

还有一个常见误区：把 token 切得越碎越好。实际并非如此。过于频繁的前端 DOM 更新会让浏览器渲染线程变忙，反而造成视觉抖动。工程上通常会在“尽快可见”和“渲染成本”之间找平衡，比如服务端按 token 发，前端每 20 到 50 毫秒合并一次文本更新。

---

## 替代方案与适用边界

不是所有场景都值得做流式。若回答很短、并发很低、用户不关心逐步出现的过程，一次性返回更简单，也更容易维护。流式是用复杂度换交互体验，不是默认正确答案。

常见传输方案可以这样比较：

| 方案 | 通信方向 | 优点 | 限制 | 适合场景 |
| --- | --- | --- | --- | --- |
| SSE | 服务端到客户端单向 | 简单、基于 HTTP、代理友好 | 双向能力弱 | 聊天正文、日志流、进度流 |
| WebSocket | 双向 | 双向交互强、状态持续 | 标准接口天然背压弱，运维更复杂 | 实时协同、频繁控制指令 |
| WebSocketStream | 双向流式 | 更接近 Streams 语义，支持背压思路 | 浏览器支持和生态不如前两者普遍 | 对背压和双向流都很敏感的实验性场景 |

可以把选择逻辑压成一张决策表：

| 问题 | 推荐 |
| --- | --- |
| 是否主要是单向输出 token？ | 是：优先 SSE |
| 是否需要频繁双向控制？ | 是：考虑 WebSocket |
| 是否必须处理细粒度背压？ | 是：评估 WebSocketStream 或服务端自建限流机制 |
| 是否只是停止、重试、切参？ | 多数情况：SSE + 独立 POST 控制足够 |

真实工程里，一个很常见且足够稳的方案是：

1. 正文流式输出用 SSE。
2. 停止生成、重试、切换温度等控制用独立 HTTP 接口。
3. 服务端保存请求 ID 到任务句柄的映射。
4. 每个连接都有心跳、断连检测、队列上限和超时回收。

边界判断只有一句：复杂度是否值得，取决于你是否真的需要双向交互，以及是否必须把背压做成一等公民。若只是“把回答一边生成一边显示”，SSE 往往已经够用，而且更容易穿透代理和负载均衡。

---

## 参考资料

下面这些资料分成“协议与标准”和“实践与实现”两类看会更高效。前者帮助确认浏览器和协议行为，后者帮助理解真实流式接口如何设计。文中公式均为工程抽象，不是协议标准定义。

| 名称 | 用途 | 适合章节 |
| --- | --- | --- |
| MDN Server-Sent Events | 理解 SSE 基本语义与浏览器行为 | 问题定义、代码实现 |
| MDN WebSocket | 理解 WebSocket 基础接口与限制 | 替代方案 |
| RFC 6455 | 确认 WebSocket 协议标准定义 | 替代方案、边界 |
| OpenAI Streaming API | 参考真实流式 API 的事件设计 | 代码实现、工程权衡 |
| Hugging Face TGI Streaming | 参考开源推理服务的流式方案 | 核心机制、工程权衡 |
| MDN WebSocketStream | 了解带流式背压思路的接口 | 替代方案 |

1. [MDN: Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
2. [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
3. [MDN: WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
4. [MDN: WebSocketStream](https://developer.mozilla.org/en-US/docs/Web/API/WebSocketStream)
5. [OpenAI API: Streaming API responses](https://platform.openai.com/docs/api-reference/streaming)
6. [Hugging Face Text Generation Inference: Streaming](https://huggingface.co/docs/text-generation-inference/main/conceptual/streaming)
7. [RFC 6455: The WebSocket Protocol](https://datatracker.ietf.org/doc/rfc6455)
