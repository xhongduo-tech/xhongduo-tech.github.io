## 核心结论

Backpressure，中文常译为“背压”，白话讲就是“下游吃不动时，上游必须主动慢下来”的控制机制。它解决的不是“系统怎么更快”，而是“系统怎么不因为局部变慢而整体失稳”。

在低延迟流式输出里，至少有三段速度可能不一致：

- 服务端生成速度 `P`
- 网络传输速度 `T`
- 客户端消费速度 `C`

真正能持续跑稳的速率不是三者里的最快值，而是最慢值：

$$
R = \min(P, T, C)
$$

如果上游仍按 `P` 持续推送，而端到端只能按 `R` 消化，那么积压队列会按下面的关系增长：

$$
Q(t + \Delta t) = Q(t) + (P - R) \times \Delta t
$$

这里 `Q(t)` 表示某一时刻的队列积压量，白话讲就是“还没被送完或处理完的数据”。只要 `P > R`，队列就会继续长。背压的目标就是把这个队列限制在一个可控上限附近，而不是让它无限堆积。

可以先记住一句结论：**背压是“限流 + 有界缓冲”机制，不是吞吐优化器。**

一个总览表如下：

| 符号 | 含义 | 白话解释 | 决策意义 |
|---|---|---|---|
| `P` | 上游生成速率 | 服务端每秒能产出多少数据 | 过快会制造积压 |
| `T` | 网络传输速率 | 链路每秒真正能送走多少数据 | 受带宽、拥塞、协议影响 |
| `C` | 客户端消费速率 | 浏览器、移动端、下游服务每秒能处理多少数据 | 常是最容易被忽略的瓶颈 |
| `R` | 实际吞吐 | 端到端真正稳定跑起来的速度 | `R = min(P, T, C)` |

玩具例子可以这样看：服务端像往瓶子里倒水，浏览器渲染像瓶口往外出水。瓶口变小以后，继续猛倒不会让出水更快，只会让瓶身中间的压力变大，最后变成缓冲区堆积、阻塞和尾延迟恶化。

---

## 问题定义与边界

本文讨论的是**低延迟流式输出**中的背压控制。低延迟，白话讲就是“用户希望边生成边看到结果，而不是等全部完成再一次性返回”。典型场景包括：

- LLM token 流式输出
- SSE（Server-Sent Events）推送
- WebSocket 消息流
- gRPC stream
- 浏览器 `ReadableStream` / Node.js stream

这里的核心问题不是“能不能发”，而是“当链路后半段变慢时，前半段是否还能自我约束”。

先看一个非常基础的例子。服务端 1 秒能生成 60 个 token，但手机端页面只能稳定渲染 15 个 token。即使网络足够快，端到端仍然只能按 15 token/s 稳定消费。多出来的 45 token/s 不会凭空消失，只会挤在中间缓冲区。

边界表如下：

| 场景 | 是否重点讨论 |
|---|---|
| SSE / WebSocket / gRPC stream | 是 |
| Web Streams / Node.js stream | 是 |
| 批处理导出 / 离线转码 | 否 |
| 日志归档 / 异步转存 | 否 |
| 可接受高延迟的消息队列 | 否 |

统一定义如下：

- 上游生成速率 `P`
- 网络传输速率 `T`
- 客户端消费速率 `C`
- 实际吞吐 `R = min(P, T, C)`

这篇文章不讨论两个边界外问题。

第一，**离线批处理**。如果目标是吞吐最大化而不是首字延迟最小化，那么批量化、队列削峰、异步转存通常更重要，背压只是辅助机制。

第二，**无损积压就是业务要求**。例如审计日志、账务流水、关键事件归档，这类系统宁可慢，也不能丢。它们仍然需要背压，但决策重点会从“交互延迟”转到“可靠落盘、持久队列、容量规划”。

所以本文的讨论边界很明确：**当用户正在等一个实时流式结果时，如何让输出既快又稳，并且不会在高并发下把内存和尾延迟拖爆。**

---

## 核心机制与推导

背压信号总是来自下游。白话讲，真正知道“自己快满了”的只能是接收方或更靠后的缓冲区。上游不能靠猜，必须靠显式信号或可观测状态来决定是否继续发送。

不同技术栈的信号长得不一样，但语义一致：

- Node.js `write()` 返回 `false`
- `drain` 事件到来
- Web Streams 的 `desiredSize <= 0`
- gRPC / HTTP2 的 flow control window 变小
- Reactive Streams 中下游显式 request 数量减少

它们本质上都在表达一句话：**先停，等我腾出空间。**

一个核心局部公式是：

$$
desiredSize = HWM - queuedBytes
$$

其中 `HWM` 是 high-water mark，中文常叫“高水位线”，白话讲就是“队列允许的安全上限”。`queuedBytes` 是当前已经堆进去但还没被送走的数据量。

当 `desiredSize > 0` 时，说明还可以继续写；当 `desiredSize <= 0` 时，说明缓冲区已经接近或达到上限，应该暂停推送。

状态变化可以整理成表：

| 状态 | 条件 | 动作 |
|---|---|---|
| 可继续写入 | `desiredSize > 0` | 继续发送 |
| 需要暂停 | `desiredSize <= 0` 或 `write() === false` | 停止推送 |
| 恢复写入 | `drain` / `pull` 到来 | 重新发送 |

现在做一个最小数值推导：

- `P = 1 KB/ms`
- `T = 0.5 KB/ms`
- `C = 0.2 KB/ms`

所以：

$$
R = \min(1, 0.5, 0.2) = 0.2\ \text{KB/ms}
$$

100 ms 内，如果不做背压，积压增长量约为：

$$
(1 - 0.2) \times 100 = 80\ \text{KB}
$$

若 `HWM = 64 KB`，则大约在 80 ms 左右触发背压。注意这里不是说“80 ms 一到系统就坏”，而是说“在这个量级上，缓冲区已经被明显顶满，继续无脑推送会开始进入失控区”。

这也是为什么低延迟流式接口很容易出现一个误判：**首字延迟看起来正常，但后半段越来越慢。**  
原因不是模型突然变慢，而是前面几段写得太快，中间缓冲区已经塞满，后续只能排队等待。用户看到的现象就是：

- 第一屏输出很快
- 中段开始抖动
- 尾段明显拖长
- 高并发时个别连接超时

真实工程例子是 LLM 网关。假设模型服务端能生成 60 tok/s，桌面浏览器大多还能跟上，但移动端弱机、复杂 DOM、长文本 diff、语法高亮、Markdown 增量渲染都可能把消费速度拉到 10 到 20 tok/s。此时如果服务端只盯着“模型生成很顺”，而不盯“客户端是否消费得动”，系统在单连接上也许还能扛，在高并发下就会变成：

- 每个连接各自积压一点
- 总 RSS 持续上涨
- GC 频率上升
- p95、p99 尾延迟被拉长

所以推导到工程结论就是：**背压不是可选优化，而是流式链路里的稳定性约束。**

---

## 代码实现

代码层面要同时满足两件事：

1. 生产端可暂停  
2. 发送端尊重背压信号

只做到第二点还不够。因为如果“网络发送停了”，但“上游生成还在继续”，那只是把压力从 socket 缓冲区转移到了应用内存队列。

先给一个可运行的 Python 玩具例子。这个例子不依赖具体协议，只模拟“生产速度大于消费速度时，队列如何在 HWM 处触发背压”。

```python
from collections import deque

def simulate_backpressure(p=10, c=4, hwm=20, steps=10):
    """
    p: 每个时间片生成的数据单位数
    c: 每个时间片消费的数据单位数
    hwm: high-water mark
    """
    q = deque()
    paused = False
    pause_count = 0

    for _ in range(steps):
        # 下游满了，暂停上游
        if len(q) >= hwm:
            paused = True
            pause_count += 1

        # 队列回落到安全区，恢复上游
        if paused and len(q) <= hwm // 2:
            paused = False

        if not paused:
            for _ in range(p):
                q.append(1)

        for _ in range(min(c, len(q))):
            q.popleft()

        # 关键断言：队列不会无界增长
        assert len(q) <= hwm + p

    return len(q), pause_count

final_q, pauses = simulate_backpressure()
assert pauses > 0
assert final_q >= 0
print("ok", final_q, pauses)
```

这个例子的重点不是精确模拟某个协议，而是说明一个原则：**一旦队列到达上限，就必须让上游停下来；恢复也要有条件，而不是立刻满速重启。**

下面看 Node.js 服务端最小实现。这里的 `write()` 是“往可写流里写数据”的 API，白话讲就是“把 chunk 往 socket 或响应流里塞”。如果返回 `false`，表示底层缓冲区压力已经较大，不能继续无脑写。

```js
import { once } from 'node:events';
import http from 'node:http';

async function* tokenGenerator() {
  const chunks = ['你', '好', '，', '这', '是', '流', '式', '输', '出'];
  for (const chunk of chunks) {
    await new Promise(r => setTimeout(r, 20));
    yield chunk;
  }
}

http.createServer(async (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  for await (const token of tokenGenerator()) {
    const ok = res.write(`data: ${token}\n\n`);
    if (!ok) {
      await once(res, 'drain');
    }
  }

  res.end();
}).listen(3000);
```

这个例子已经体现了发送端尊重背压，但它还不够完整。更完整的做法是让生成器本身也能被暂停。例如把“生成 token”设计成 pull 模式，只有在发送端确认还有空间时才继续取下一个 token。

Web Streams 的接口形式不同，但原则相同。`desiredSize` 表示“理想情况下还希望再接收多少数据”。小于等于 0，说明队列已经到高水位附近。

```js
function makeStream(chunks) {
  let i = 0;

  return new ReadableStream({
    pull(controller) {
      while (i < chunks.length && controller.desiredSize > 0) {
        controller.enqueue(chunks[i++]);
      }
      if (i >= chunks.length) {
        controller.close();
      }
    }
  });
}
```

代码到机制的对照表如下：

| API | 背压信号 | 推荐动作 |
|---|---|---|
| Node.js `write()` | `false` | 等 `drain` |
| Web Streams `desiredSize` | `<= 0` | 暂停 `enqueue` |
| gRPC stream | flow control window 变小 | 停止继续发送 |
| 应用内生成器 | 队列达到 `HWM` | 暂停生成或降速 |

真实工程里，推荐的结构通常是三层：

| 层级 | 责任 | 要点 |
|---|---|---|
| 生成层 | 产出 token / chunk | 必须可暂停、可取消 |
| 发送层 | 写入 socket / stream | 必须尊重 `write()` / `desiredSize` |
| 连接层 | 超时、断连、隔离 | 每个连接独立队列，避免互相拖累 |

如果只在“发送层”等 `drain`，但“生成层”依然继续产出并塞进内存数组，那么你只是把问题从协议缓冲区搬到了应用堆内存，系统仍然会在高并发下失稳。

---

## 工程权衡与常见坑

背压最核心的收益有三个：

- 内存有界
- 尾延迟可控
- 慢客户端不会无限放大系统风险

它的代价也很明确：实现复杂度上升，链路状态更多，调试难度变大。尤其在流式产品里，很多问题并不是“有没有背压”，而是“背压有没有贯穿生成、传输、消费三段链路”。

常见坑与修正如下：

| 常见坑 | 后果 | 修正方式 |
|---|---|---|
| 只控发送，不控生成 | 队列继续堆积 | 让生成器也可暂停 |
| 忽略 `write() === false` | 缓冲失控 | 严格等待 `drain` |
| 无界内存队列 | RSS 飙升 | 设置 `HWM` 和队列上限 |
| 慢快客户端混队列 | 头阻塞 | 按连接隔离 |
| 只看平均延迟 | 尾延迟恶化被掩盖 | 监控 p95/p99 |
| 没有超时/取消 | 慢连接拖垮系统 | `timeout` / `abort` / `circuit breaker` |

这里有一个很常见的误区：有人会说“先全量生成，再靠异步线程慢慢发给客户端，不就避免阻塞了吗？”  
这通常只是把实时链路改成了“先积压、后排队”。如果用户目标是低延迟交互，这么做等于主动放弃流式输出的意义。

再看一个真实工程坑。很多 LLM 前端为了做 Markdown 增量渲染，会在每个 token 到达时都触发：

- 文本拼接
- Markdown 重新解析
- 代码块高亮
- DOM 更新

这会让客户端消费速率 `C` 大幅下降。服务端如果不知道这一点，还按模型生成速度猛推，就会误以为“网络慢”或“模型尾部退化”，实际上根因是客户端消费能力下降导致的背压传导。

工程监控至少要覆盖：

```text
需要观察：队列深度、RSS、GC、p95/p99、写入阻塞次数、drain 等待时长
```

其中有两个指标特别重要。

第一，**写入阻塞次数**。它反映“下游忙不过来”的频率。  
第二，**`drain` 等待时长**。它反映“忙不过来”到底严重到什么程度。

如果只看平均响应时间，你会错过很多真实风险。因为平均值会被大量健康请求稀释，而流式接口真正影响体验的，往往是后 5% 或后 1% 的连接。

---

## 替代方案与适用边界

不是所有“慢”的问题都应该用强背压解决。背压适合的是“结果必须按顺序、尽量完整、低延迟地持续输出”的场景。如果业务允许损失、允许延迟、允许批量，那就应该考虑别的策略。

方案对比如下：

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 背压 | 实时流式输出 | 有界内存、稳态延迟 | 实现复杂度上升 |
| 丢弃/降采样 | 允许信息损失 | 简单直接 | 数据不完整 |
| 批量发送 | 可接受更高延迟 | 提升效率 | 首字延迟变差 |
| 队列削峰 | 峰值流量缓冲 | 易于解耦 | 可能积压更久 |

玩具例子很直观。直播弹幕太密时，可以做限速、合并、降采样，因为用户不需要看到每一条都严格无损。  
但如果是在线代码补全或问答 token 流，用户期待的是连续、低延迟、顺序正确的输出，此时更适合强背压，而不是简单丢弃。

再看一个真实工程边界。日志归档系统如果吞吐压力大，通常更适合：

- 批量写盘
- 消息队列缓冲
- 异步消费
- 压缩后落库

因为这类系统的关键目标是吞吐和可靠性，不是首字延迟。相反，面向用户的 AI 对话流式输出更看重“第一屏快”和“后半段别抖”，所以优先级正好相反。

适用边界可以总结为：

```text
如果目标是低延迟 + 稳定输出，优先背压。
如果目标是高吞吐 + 可延迟消费，可以考虑批量和异步化。
如果目标是允许损失，就应该考虑降采样而不是强背压。
```

最后再强调一次：背压不是万能药。它解决的是“速度不匹配时如何不失稳”，不是“为什么慢”。如果真正瓶颈在模型推理、前端渲染、网络抖动、协议开销，那么你仍然需要分别优化对应环节。

---

## 参考资料

参考资料总览：

| 主题 | 资料 |
|---|---|
| Node.js Streams 背压 | https://nodejs.org/learn/modules/backpressuring-in-streams |
| `write()` / `drain` | https://nodejs.org/download/release/v22.12.0/docs/api/stream.html |
| Web Streams API | https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Concepts |
| gRPC Flow Control | https://grpc.io/docs/guides/flow-control/ |
| Reactive Streams | https://www.reactive-streams.org/ |

这些资料分别对应“实现接口”和“协议层流控”，读的时候要把它们映射回同一个背压模型。

1. [Backpressuring in Streams](https://nodejs.org/learn/modules/backpressuring-in-streams)
2. [Node.js Stream API: writable.write() and drain](https://nodejs.org/download/release/v22.12.0/docs/api/stream.html)
3. [MDN Streams API Concepts](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Concepts)
4. [gRPC Flow Control](https://grpc.io/docs/guides/flow-control/)
5. [Reactive Streams](https://www.reactive-streams.org/)
