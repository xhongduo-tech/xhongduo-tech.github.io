## 核心结论

异步推理是“提交即忘记”的模型部署模式。所谓“提交即忘记”，意思不是系统真的忘了请求，而是客户端在提交后不再阻塞等待模型执行结束，只拿到一个 `request_id` 作为后续查询凭证，真正的计算由后台队列和 worker 在之后完成。

它成立的前提很直接：模型推理时间长、请求量高、任务之间允许排队、结果不要求在当前连接内立即返回。在这种条件下，异步模式把“用户等待时间”和“模型处理吞吐”拆开处理。用户先完成提交动作，系统再用队列、批处理、优先级调度、重试和 webhook 把结果送达。

一个新手能立刻理解的场景是：用户提交一段长文本摘要请求，系统马上返回 `request_id=abc123`，用户去喝杯咖啡，后台 worker 从队列取出任务执行，完成后把结果通过 webhook 发给业务系统，或者由前端稍后轮询状态页拿回结果。这里的重点不是“更快”，而是“不把用户卡在原地”。

简化流程可以写成：

`客户端 -> 请求队列 -> worker -> 结果存储/结果交付`

异步推理最大的价值有三点：

| 价值 | 解释 |
| --- | --- |
| 解耦延迟与吞吐 | 单个请求可以晚一点完成，但整体系统能吃下更多请求 |
| 支持长任务 | 图像生成、长文总结、批量 embedding、复杂多步推理更适合排队执行 |
| 支持调度策略 | 可以做 FIFO、优先级队列、租户隔离、限流、超时丢弃 |

---

## 问题定义与边界

同步推理和异步推理的区别，不在“有没有并发”，而在“客户端是否必须在线等结果”。

同步阻塞模式下，客户端发起请求后，HTTP 连接一直挂着，直到模型返回结果或超时。异步模式下，客户端提交后立即拿到 `request_id`，后续通过轮询或 webhook 取结果。`request_id` 可以理解成“取件码”，它把“提交请求”和“领取结果”拆成两个动作。

玩具例子：一个“打印照片”服务要 30 秒生成高清图。同步模式下，用户必须盯着页面等 30 秒；异步模式下，用户拍照后回到桌面继续做别的事，稍后打开“我的任务”查看状态，或者等系统通知“已完成”。

这两种模式的边界可以直接对比：

| 维度 | 同步推理 | 异步推理 |
| --- | --- | --- |
| 客户端等待 | 必须等待模型完成 | 只等到拿到 `request_id` |
| 连接占用 | 长连接占用明显 | 提交后连接快速释放 |
| 适用任务 | 短任务、低延迟交互 | 长任务、高吞吐、需要排队 |
| 用户体验 | 即时结果 | 延迟结果，但不阻塞 |
| 调度能力 | 较弱 | 强，可做优先级和重试 |
| 失败处理 | 以请求超时为主 | 要处理排队过期、重试、回调失败 |

异步模式不是“任何场景都更好”。它有明确边界：

1. 任务允许排队，不能要求当前页面立刻展示最终结果。
2. 系统必须保存任务状态，至少要有 `queued / running / succeeded / failed / expired` 这类状态。
3. 系统必须支持“轮询”或 “webhook” 至少一种结果交付方式。
4. 系统必须控制排队上限，例如最长排队时间、最大重试次数、任务过期策略。
5. 任务结果不能无限期有效，否则容易出现“结果变旧”。结果变旧指的是输入环境已经变化，晚到的结果虽然算对了旧问题，却不再适合当前业务状态。

真实工程里，很多平台会给排队设置硬边界。例如输入队列最长等待时间可能是数小时到数十小时。这个边界不是装饰，而是防止“积压请求把未来的容量也吃掉”。

---

## 核心机制与推导

异步推理真正提高吞吐，不只是因为“先排队”，而是因为排队之后调度器可以做批处理。批处理就是“把多条请求合成一批一起送进 GPU”，这样可以摊薄固定开销。摊薄固定开销，白话讲就是原本每次调用模型都要付一次“开机费”，合批之后多条请求共付一次。

### 1. 延迟和吞吐的基本关系

设：

- $W$ 是收集窗口，表示调度器愿意多等多久来凑批次
- $B$ 是该窗口内最终形成的批次大小
- $T(B)$ 是 GPU 处理一个大小为 $B$ 的批次所需时间

则常见近似关系是：

$$
L \approx \frac{W}{2} + T(B)
$$

这里 $L$ 是平均延迟。为什么是 $\frac{W}{2}$？因为请求不是都在窗口起点到达，平均来看，大致会在窗口里等半个窗口长度。

吞吐量近似为：

$$
\text{Throughput} \approx \frac{B}{W + T(B)}
$$

它表达的是：每经过一个“收集 + 执行”周期，系统处理了 $B$ 条请求。

### 2. 数值例子

假设调度器设置收集窗口 $W = 10ms$，并且测得：

- 单条请求单独跑要 $5ms$
- 如果凑到 $32$ 条一起跑，$T(32)=20ms$

那么平均延迟约为：

$$
L \approx 10/2 + 20 = 25ms
$$

吞吐量约为：

$$
\frac{32}{10ms + 20ms} = \frac{32}{30ms} \approx 1066.7 \text{ req/s}
$$

如果单条单跑，每条约 $5ms$，吞吐量约是：

$$
\frac{1}{5ms} = 200 \text{ req/s}
$$

结论很清楚：批处理显著提高吞吐，但额外引入了收集窗口带来的排队延迟。

### 3. 为什么动批有效

动批，即动态批处理，意思是调度器不是永远固定批次大小，而是在一个很短的窗口内，比如 1 到 50ms，观察当前队列，把能合并的请求尽量装进同一批。它像一个“短时间收集器”：先等几毫秒，尽量多收一点，再一起送 GPU。

新手版理解可以用“10ms 收集时间”：

- 第 1 条请求到了，不立刻喂 GPU
- 调度器再等最多 10ms，看有没有更多请求
- 10ms 内陆续来了很多请求，就合成一批
- 10ms 到了就发车，不再无限等下去

ASCII 图示如下：

```text
时间轴:  |---- W=10ms ----| 执行 T(B)=20ms
请求到达:  x   x x    x  x x x
批次形成:  [       B=8       ] -> GPU
```

这里的关键权衡是：

- 窗口太小：批次不够大，GPU 利用率低
- 窗口太大：吞吐高了，但延迟和方差上升
- 方差上升指的是同类请求的完成时间差距变大，用户更难预测什么时候拿到结果

### 4. 玩具例子和真实工程例子

玩具例子：一个班级交作业，如果老师每收到一份就立刻批改，相当于同步单条处理；如果老师每 10 分钟收一摞一起批，相当于异步批处理。后者对老师更高效，但最早交的人会额外等待。

真实工程例子：图像生成平台在晚高峰收到大量“生成海报”请求。单条图片生成可能要 10 到 60 秒，业务又允许稍后拿结果。这时同步接口会占满连接、撑爆超时重试、把前端卡死；异步接口则把请求写入持久队列，由多个 GPU worker 拉取任务，按模型类型分桶、按租户做优先级，再通过 webhook 回调业务系统。这样做的重点不是“单张图更快”，而是“高峰期系统仍然可控”。

---

## 代码实现

一个最小可用的异步推理系统，至少包含四个组件：

1. 提交接口：接收请求，生成 `request_id`，写入队列。
2. 状态存储：记录任务状态、结果、错误、过期时间。
3. worker：从队列拉取任务，执行模型，写回结果。
4. 结果交付：轮询接口或 webhook。

### 1. 前端/客户端流程

下面是新手版 JavaScript 伪代码，展示“提交后轮询”的基本过程：

```javascript
async function submitJob(payload) {
  const resp = await fetch("/api/infer/async", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  return data.request_id;
}

async function pollStatus(requestId) {
  const resp = await fetch(`/api/infer/result/${requestId}`);
  return resp.json();
}

async function run() {
  const requestId = await submitJob({prompt: "写一段产品介绍"});
  const timer = setInterval(async () => {
    const result = await pollStatus(requestId);
    if (result.status === "succeeded") {
      console.log("done:", result.output);
      clearInterval(timer);
    } else if (result.status === "failed" || result.status === "expired") {
      console.error("job ended:", result.status);
      clearInterval(timer);
    }
  }, 2000);
}
```

### 2. 后端队列与 worker

下面是 Python 玩具实现，能直接运行，演示核心状态流转、过期判断和幂等 webhook 重试。幂等，白话讲就是“同一件事重复做多次，结果仍然只算一次”。

```python
import time
from collections import deque

class AsyncInferenceService:
    def __init__(self, max_time_in_queue=5.0):
        self.queue = deque()
        self.jobs = {}
        self.delivered = set()
        self.max_time_in_queue = max_time_in_queue

    def submit(self, payload, request_id):
        now = time.time()
        self.jobs[request_id] = {
            "status": "queued",
            "payload": payload,
            "created_at": now,
            "output": None,
            "attempts": 0,
        }
        self.queue.append(request_id)
        return request_id

    def worker_once(self):
        if not self.queue:
            return None
        request_id = self.queue.popleft()
        job = self.jobs[request_id]
        now = time.time()

        if now - job["created_at"] > self.max_time_in_queue:
            job["status"] = "expired"
            return "expired"

        job["status"] = "running"
        prompt = job["payload"]["prompt"]
        job["output"] = prompt.upper()
        job["status"] = "succeeded"
        return "succeeded"

    def deliver_webhook(self, request_id):
        job = self.jobs[request_id]
        if job["status"] != "succeeded":
            return False

        delivery_key = f"webhook:{request_id}"
        if delivery_key in self.delivered:
            return True

        job["attempts"] += 1
        self.delivered.add(delivery_key)
        return True

svc = AsyncInferenceService(max_time_in_queue=10.0)
rid = svc.submit({"prompt": "hello async"}, request_id="req-1")
assert rid == "req-1"
assert svc.jobs[rid]["status"] == "queued"

status = svc.worker_once()
assert status == "succeeded"
assert svc.jobs[rid]["output"] == "HELLO ASYNC"

first = svc.deliver_webhook("req-1")
second = svc.deliver_webhook("req-1")
assert first is True
assert second is True
assert svc.jobs["req-1"]["attempts"] == 1
```

这个例子刻意保持简单，但保留了几个关键点：

- `submit` 只负责接收并入队，不同步执行模型
- `worker_once` 先检查是否过期，再执行模型
- `deliver_webhook` 用 `request_id` 做幂等键，防止重复投递造成重复记账、重复发货或重复写库

### 3. 一个更接近工程的状态机

```text
queued -> running -> succeeded
queued -> expired
running -> failed -> retrying -> queued
succeeded -> webhook_delivering -> delivered
```

真实工程例子可以这样设计：

- `/api/infer/async`：写入 Kafka、Redis Stream 或数据库任务表
- GPU worker：按模型类型消费队列，做动态批处理
- 结果表：存 `request_id`、状态、输出位置、错误码、过期时间
- webhook 服务：失败重试，指数退避，超过阈值后转人工或死信队列
- `/api/infer/result/:id`：给前端轮询

### 4. 必要控制项

| 控制项 | 为什么必须有 |
| --- | --- |
| `max_time_in_queue` | 防止请求永远排队，结果失去业务价值 |
| 限流 | 防止某个租户瞬间打爆系统 |
| 幂等键 | 防止重复投递导致重复副作用 |
| 重试上限 | 防止坏任务无限循环 |
| 任务过期 | 防止旧结果覆盖新状态 |

---

## 工程权衡与常见坑

异步推理的主要风险，不是“代码难写”，而是“系统更难失控地坏掉”。同步接口的问题通常会立刻显现为超时；异步系统的问题则可能表现为队列堆积、结果延迟漂移、回调丢失、缓存变旧，定位更难。

下面是常见风险与缓解方式：

| 问题 | 触发条件 | 缓解手段 |
| --- | --- | --- |
| 队列无限堆积 | 流量突增、容量不足、下游卡住 | 设置 `max_time_in_queue`、限流、扩容、熔断 |
| webhook 丢失 | 下游服务 5xx、网络抖动 | 幂等重试、死信队列、签名校验 |
| 结果变旧 | 排队过长、输入关联状态变化 | 任务过期、版本号校验、状态二次确认 |
| 延迟方差过大 | 动批窗口过大、优先级争用 | 缩小窗口、拆分队列、按 SLA 分级 |
| GPU 利用率低 | 批次太小、模型切换频繁 | 分桶调度、模型预热、合适的动批策略 |
| 重试风暴 | worker 或 webhook 故障恢复时集中重试 | 指数退避、随机抖动、全局重试预算 |

新手最容易忽略的是 webhook 幂等。比如订单生成完成后，系统调用业务方 `/notify`。如果业务方第一次收到了，但回包超时，平台会认为失败而重试。若业务方没有按 `request_id` 去重，就可能重复发优惠券、重复入账。正确做法是：每次 webhook 都带 `request_id` 和签名，下游以 `request_id` 作为幂等键。

另一个常见坑是“异步就等于更稳定”。这不准确。异步只是把压力从在线请求阶段转移到了队列和后台执行阶段。如果没有容量治理，积压照样会把系统拖垮，而且坏得更慢、更隐蔽。

工程上通常还要回答三个问题：

1. 这个任务最长可以排多久？
2. 这个结果多久后就没有业务价值？
3. 用户能接受“完成率高但慢”，还是“慢了就直接失败”？

这三个问题不先定义，队列参数就无从设定。

---

## 替代方案与适用边界

同步和异步不是谁替代谁，而是处理不同目标。

同步推理适合：

- 聊天问答首 token 很敏感
- 搜索重排、推荐打分这类链路内调用
- 页面加载必须立即展示结果

异步推理适合：

- 图像生成、视频生成、长文改写
- 大批量 embedding、离线分析
- 多租户高吞吐任务池
- 允许排队、允许稍后通知的业务

可以用一个条件判断表概括：

| 条件 | 更适合的方式 |
| --- | --- |
| 单次任务 < 1 秒，用户正在等页面 | 同步 |
| 单次任务数秒到数分钟，可稍后取结果 | 异步 |
| 高峰时请求突刺明显 | 异步或同步+排队保护 |
| 用户分层明显，有 VIP 优先需求 | 异步优先级队列 |
| 既想快返回，又要兜住超时 | Hybrid |

Hybrid 是常见演进方案：先尝试同步执行，若超过阈值，比如 800ms 还没完成，就自动转入异步队列，并返回 `request_id`。这样短任务走快路径，长任务走稳路径。

新手版例子：

- “快速聊天”更适合同步，因为用户在等屏幕立即出字。
- “批量生成 500 张商品海报”更适合异步，因为核心目标是总吞吐和可控完成率，不是某一张海报必须立即返回。

还存在一种“批次同步”模式：客户端依然在线等，但服务端在内部用极短窗口做动态批处理。它适用于低延迟但流量很高的场景，比如在线分类、重排模型。它的边界是窗口必须非常小，否则用户感知延迟会上升。

---

## 参考资料

1. Baseten Async Inference 文档  
   作用：用于定义“提交即忘记”、`request_id`、输入队列、webhook 回调、最大排队时间等机制。  
   链接：https://docs.baseten.co/inference/async

2. SystemOverflow 关于 Dynamic Batching 的说明  
   作用：用于解释收集窗口 $W$、批次大小 $B$、延迟与吞吐之间的关系，以及为什么 GPU 批处理能摊薄固定开销。  
   链接：https://www.systemoverflow.com/learn/ml-model-serving/serving-infrastructure/dynamic-batching-throughput-vs-latency-tradeoffs-in-request-scheduling

3. 阿里云 PAI 异步推理相关文档  
   作用：可用于补充云上异步推理的任务式调用、排队执行和工程配置方式。写作时适合作为“平台实现形态”的参考，而不是基础定义来源。

4. AWS 系统调优建议与服务化实践资料  
   作用：适合补充限流、扩缩容、预热、重试风暴治理这类系统层面建议。

5. 关于 staleness / variance 的讨论资料  
   作用：用于解释“结果变旧”和“延迟方差变大”为什么是异步系统的真实代价，而不是纯理论问题。

一个直接可用的引用方式示例：如果后续要写“动态批处理为什么会增加平均延迟”，优先引用 SystemOverflow 对窗口收集和吞吐计算的说明，因为它直接对应本文“核心机制与推导”部分。
