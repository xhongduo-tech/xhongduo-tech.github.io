## 核心结论

分布式爬虫的核心不是“多开几个进程”，而是把**请求状态**和**去重状态**从单机内存抽离成共享服务。这里的“共享服务”可以先理解成所有爬虫节点共同访问的一套排队与记账系统。对 Scrapy-Redis 来说，这套系统通常就是 Redis：它既保存全局请求队列，也保存全局去重集合。

一个可落地的基础架构通常由四部分组成：Scrapy 工作节点、Redis 请求队列、Redis 去重结构、反爬控制层。工作节点负责发请求和解析页面；请求队列决定“下一个抓什么”；去重结构保证同一个 URL 不被多次抓取；反爬控制层负责限制速率、轮换代理、切换 User-Agent 和管理 Cookie。

给零基础读者先一个玩具例子。把 Redis 想成一个统一“排队机”，URL 是排队小票，多个 Scrapy 进程像多个窗口同时叫号。某个窗口取走一个 URL 后开始抓取；若一个新 URL 以前已经见过，Bloom Filter 会先把它拦下来。Bloom Filter 可以先理解成“低内存的重复检查器”，它不是百分之百精确，但能用极小内存挡住绝大多数重复请求。

这套架构能解决三个实际问题：

| 问题 | 单机爬虫常见表现 | 分布式方案的处理方式 |
|---|---|---|
| 请求分发 | 多进程之间各抓各的，可能重复 | Redis 统一排队，节点共享消费 |
| URL 去重 | 每台机器各自记忆，跨节点重复 | Bloom Filter 或集合做全局去重 |
| 节点故障 | 某节点挂掉后任务丢失 | 未完成请求仍在队列，其他节点接手 |

真实工程例子是商品采集。假设你要抓取 300 万个商品详情页，来源包括分类页、搜索页、推荐页。若每台机器独立跑，重复 URL 会很多，失败请求也难重试。改成 Redis 共享队列后，新增机器只要接入同一个队列即可扩容；某节点宕机，其未完成的请求会被其他节点继续消费；再叠加代理池和速率限制，系统才能在可控封禁风险下稳定运行。

---

## 问题定义与边界

这类系统的目标可以精确表述为：在多节点环境中，实现**公平、高效、可恢复**的 URL 分发与去重，并在目标站点反爬限制下保持可接受吞吐。这里的“公平”是指节点不会长期抢不到任务；“可恢复”是指单个节点失败不会导致全局状态丢失。

边界先明确，否则架构会被说得过宽。

| 维度 | 本文假设 | 不负责解决的问题 |
|---|---|---|
| 全局状态源 | Redis 是唯一共享状态源 | 多数据中心强一致同步 |
| 去重机制 | Bloom Filter 为主，可接受少量误报 | 严格零误报的精细去重 |
| 调度策略 | FIFO/LIFO/优先级三类 | 复杂图搜索最优路径问题 |
| 反爬策略 | 延迟、代理、UA、Cookie 管理 | 绕过验证码、JS 混淆逆向 |
| 故障恢复 | 节点重启、请求重试、任务接管 | 分布式事务与 exactly-once |

为什么要强调“唯一全局状态源”？因为如果一部分请求在 Redis，一部分请求还留在各节点内存里，你就已经失去全局可观测性。系统会出现“看似没任务，实际上有节点手里攥着任务”的情况，这会让扩容、重试、恢复都变得不可靠。

再看新手容易误解的去重问题。很多人会问：既然是去重，为什么不用 Python `set`？答案是跨节点共享和内存成本。`set` 在单机上很好用，但分布式场景里它既难共享，也会占用较多内存。Bloom Filter 的价值就在这里：它用位数组和多个哈希函数表示“某个元素大概率已出现过”。

误报率定义要明确。Bloom Filter 的“误报”是：某个 URL 实际上没插入过，但过滤器判断它“可能已经存在”。它不会漏掉已存在元素，但会偶尔把新 URL 误判为重复。常用公式是：

$$
P \approx \left(1-e^{-kn/m}\right)^k
$$

其中，$P$ 是误报率，$n$ 是插入元素数，$m$ 是总 bit 数，$k$ 是哈希函数个数。

这意味着边界很清楚：如果你的业务不能接受“少抓少量页面”，Bloom Filter 就不是最终方案；如果你的业务更重视吞吐和内存，而能容忍极低比例的漏抓，它就很适合。

再给一个玩具例子。你有 10 万个 URL，设定误报率为 0.1%。这相当于允许极少量 URL 被系统误判为“已经见过”。对新闻聚合、列表页增量抓取，这通常能接受；但对司法文书、财务凭证这种高完整性采集，就可能不够。

---

## 核心机制与推导

先看去重层。Redis Bloom Filter 的常用推导有两个：

$$
k = \left\lceil \frac{-\ln(error\_rate)}{\ln 2} \right\rceil
$$

$$
bits/item = \frac{-\ln(error\_rate)}{(\ln 2)^2}
$$

这里的 `error_rate` 就是目标误报率。第一式给出最优哈希函数个数，第二式给出每个元素平均需要多少 bit。

代入一个具体数值：`error_rate = 0.001`，也就是 0.1%。

$$
bits/item = \frac{-\ln(0.001)}{(\ln 2)^2} \approx 14.378
$$

若容量是 100000 条 URL，则总 bit 数为：

$$
m = 100000 \times 14.378 = 1{,}437{,}800\ bits
$$

换算成字节约为：

$$
1{,}437{,}800 / 8 \approx 179{,}725\ bytes \approx 0.17MB
$$

最优哈希函数个数约为：

$$
k = \left\lceil \frac{-\ln(0.001)}{\ln 2} \right\rceil = 10
$$

这就是 Bloom Filter 在工程上常被采用的原因：去重空间成本随误报率平滑可控，而不是像哈希集合那样线性放大到更高常数级别。

再看调度层。Scrapy-Redis 通过 `SCHEDULER_QUEUE_CLASS` 切换底层队列行为。这里的“调度器”可以理解成“决定下一张票怎么发”的组件。

| 队列类型 | 对应类 | 访问顺序 | 常见搜索风格 |
|---|---|---|---|
| 优先级队列 | `SpiderPriorityQueue` | 先取高优先级 | 重点页面先抓 |
| FIFO 队列 | `SpiderQueue` | 先进先出 | 接近 BFS，广度优先 |
| LIFO 栈 | `SpiderStack` | 后进先出 | 接近 DFS，深度优先 |

为什么说是“接近” BFS/DFS？因为真正的遍历顺序不仅由队列结构决定，还取决于新链接产生顺序、优先级设置、并发数和重试插入位置。但从工程上看，这三种队列已经足够控制绝大部分抓取风格。

新手玩具例子可以这样理解：

- `FIFO`：像食堂排队，先来先拿，适合一层层往外扩。
- `LIFO`：像压文件栈，最后放进去的先拿，适合一路往深处钻。
- `Priority`：像急诊分诊，先处理更重要的请求，比如详情页高于列表页。

真实工程例子是电商站点采集。分类页、搜索结果页、详情页混在一起时，通常不会简单用 BFS 或 DFS，而是把详情页优先级设高，因为详情页直接产出结构化数据；列表页优先级略低，只负责发现新 URL。这样做能在抓取窗口受限时优先拿到高价值页面。

系统吞吐也可以粗算。若有 3 个节点，每个节点并发 16，请求平均响应时间 0.1 秒，则理想上限是：

$$
throughput \approx \frac{3 \times 16}{0.1} = 480\ req/s
$$

这个值只是理想值。真实场景还会受目标站 QPS 限制、代理质量、DNS、网络抖动、解析耗时影响。所以分布式架构不是无限提速，而是在约束下更稳定地逼近可用吞吐。

---

## 代码实现

最小配置要先把调度器和去重器切到 Redis 版本，否则你只是“多节点跑了多个单机爬虫”。

```python
# settings.py
SCHEDULER = "scrapy_redis.scheduler.Scheduler"
DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"

REDIS_URL = "redis://127.0.0.1:6379/0"

# 持久化队列，爬虫关闭后任务仍保留
SCHEDULER_PERSIST = True

# 三选一：
# 1. 优先级队列
SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderPriorityQueue"
# 2. FIFO
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderQueue"
# 3. LIFO
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderStack"

# 反爬基础参数
DOWNLOAD_DELAY = 1.0
RANDOMIZE_DOWNLOAD_DELAY = True
CONCURRENT_REQUESTS = 16
COOKIES_ENABLED = False
RETRY_ENABLED = True
RETRY_TIMES = 3

DOWNLOADER_MIDDLEWARES = {
    "myproject.middlewares.RandomUserAgentMiddleware": 400,
    "myproject.middlewares.ProxyRotationMiddleware": 410,
    "scrapy.downloadermiddlewares.retry.RetryMiddleware": 550,
}
```

若你使用 RedisBloom 模块，也可以在初始化时显式创建过滤器，例如容量 10 万、误报率 0.1%。下面是一个可运行的 Python 玩具实现，用来验证参数推导逻辑：

```python
import math

def bloom_params(capacity: int, error_rate: float):
    bits_per_item = -math.log(error_rate) / (math.log(2) ** 2)
    total_bits = capacity * bits_per_item
    hashes = math.ceil(-math.log(error_rate) / math.log(2))
    total_bytes = total_bits / 8
    return {
        "bits_per_item": bits_per_item,
        "total_bits": total_bits,
        "hashes": hashes,
        "total_bytes": total_bytes,
    }

params = bloom_params(100000, 0.001)

assert round(params["bits_per_item"], 3) == 14.378
assert round(params["total_bits"]) == 1437759
assert params["hashes"] == 10
assert 170000 < params["total_bytes"] < 190000

print(params)
```

请求处理流程可以简化成下面这样：

```text
新 URL -> 先查 Bloom Filter
      -> 已存在：丢弃
      -> 不存在：写入 Bloom Filter -> 入 Redis Queue
Redis Queue -> 任一 Spider 节点弹出请求
Spider 抓取 -> 成功：交给 Pipeline 落库
           -> 失败：Retry Middleware 重新入队
```

一个更贴近真实工程的伪代码如下：

```python
def schedule(url, priority=0):
    if bloom_filter.exists(url):
        return "duplicate"
    bloom_filter.add(url)
    redis_queue.push(url, priority=priority)
    return "queued"

def consume():
    req = redis_queue.pop()
    if not req:
        return "idle"

    try:
        resp = downloader.fetch(req)
        item = spider.parse(resp)
        pipeline.save(item)
        return "ok"
    except TemporaryError:
        if req.retry_times < 3:
            req.retry_times += 1
            redis_queue.push(req.url, priority=req.priority)
        return "retry"
```

真实工程例子可以看招聘站或商品站。列表页发现详情页 URL 后，先过 Bloom Filter，再按优先级入 Redis。多个节点同时从同一队列取任务。若某个节点因为代理失效导致请求失败，重试请求重新回到队列，别的节点可能用另一条代理拿到页面。这就是“故障恢复”在爬虫系统里的实际含义：不是节点永不失败，而是失败不会直接变成任务丢失。

说明图可以直接抽象为：

```text
          +----------------------+
          |      Redis Queue     |
          |  FIFO/LIFO/Priority  |
          +----------+-----------+
                     |
     +---------------+----------------+
     |               |                |
+----v----+     +----v----+      +----v----+
| Spider A|     | Spider B|      | Spider C|
+----+----+     +----+----+      +----+----+
     |               |                |
     +---------------+----------------+
                     |
          +----------v-----------+
          |   Bloom Filter       |
          |  duplicate checking  |
          +----------------------+
```

---

## 工程权衡与常见坑

第一类坑是“只做分布式，不做反爬控制”。多节点共享队列后，请求速度往往更快，但更快不等于更好。目标站感知到的是单位时间内来自你系统的访问压力，而不是你用了几台机器。如果不设 `DOWNLOAD_DELAY`、不轮换代理、不切换 UA，很容易直接拿到 429 或 403。

第二类坑是代理池规模与质量不足。代理池可以理解成“替你出门的人群”。如果这个人群太小，系统虽然有多个爬虫节点，但出口 IP 仍然集中，目标站会把你当成同一个访问者连续敲门。真实工程里，代理池至少要跟并发规模匹配，还要考虑子网和 ASN 分散度。ASN 可以先理解成“网络运营归属”，过多代理集中在同一网络归属下，也可能暴露机器流量特征。

第三类坑是 Bloom Filter 误报率设置失衡。误报率太低，内存成本会上升；误报率太高，又会误杀太多新 URL。对 10 万量级和 100 万量级 URL，参数不能照搬。很多线上漏抓问题不是代码 bug，而是去重参数太激进。

第四类坑是调度策略和业务目标错配。抓站点地图时用 BFS 更稳，因为你希望均匀扩展覆盖面；抓评论翻页或链式详情页时，DFS 往往更快进入深层页面；而带业务价值权重时，优先级队列更合适。策略错了，系统看起来还在正常工作，但产出效率会很差。

常见坑和对策可以汇总如下：

| 常见坑 | 现象 | 根因 | 对策 |
|---|---|---|---|
| 未设置下载延迟 | 429/403 激增 | 请求过密 | 设置 `DOWNLOAD_DELAY` 与随机抖动 |
| 代理池太小 | 某些 IP 很快封禁 | 出口过于集中 | 扩大池子，按子网/ASN 分层 |
| Bloom 误报过高 | 新 URL 被误丢弃 | `error_rate` 设太大 | 按容量重算参数 |
| 队列策略错误 | 抓取顺序混乱 | BFS/DFS/优先级不匹配业务 | 按页面价值切换队列 |
| Cookie 管理混乱 | 会话污染、账号串用 | 多节点共享状态不清晰 | 按账号隔离 Cookie Jar |
| 重试无上限 | 失败请求堆积 | 永久失败被反复重投 | 限制重试次数并记录失败原因 |

代理失效冷却是一个很常见但容易被忽略的设计。所谓“冷却”就是代理失败后暂时下线一段时间，不要立刻继续用。玩具例子可以理解成：某个代理刚被目标站警告，这时继续用它大概率还是失败，不如让它“离线休息”几分钟。

```python
def mark_proxy_result(proxy, success, now):
    if success:
        proxy.fail_count = 0
        proxy.next_available_at = now
    else:
        proxy.fail_count += 1
        cooldown = min(300, 2 ** proxy.fail_count)
        proxy.next_available_at = now + cooldown

def get_proxy(pool, now):
    candidates = [p for p in pool if p.next_available_at <= now]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.fail_count)
```

真实工程例子里，某商品站详情页平均响应 800ms，但同一代理连续 20 次请求后封禁率显著上升。这时仅靠 `DOWNLOAD_DELAY` 不够，必须把并发与代理池共同调参：例如每个代理并发上限 1 到 2，请求间隔 1 到 3 秒，失败代理冷却 60 到 300 秒，按成功率淘汰劣质代理。否则系统会进入“越失败越重试，越重试越封禁”的恶性循环。

---

## 替代方案与适用边界

Scrapy-Redis 不是唯一方案，它的优势是轻量、接入快、和 Scrapy 生态贴合。缺点也明显：Redis 更像高性能缓存和轻量队列，不是强一致消息系统。如果你的需求升级到复杂任务编排、精细消费确认、回溯重放，Kafka 或 RabbitMQ 往往更合适。

对比可以直接看下面这张表：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Scrapy-Redis | 部署简单、接入快、生态成熟 | 强一致能力弱、复杂任务流支持有限 | 中小规模采集、快速扩容 |
| Kafka + 自定义调度 | 吞吐高、消费语义更强、回放能力好 | 开发与运维复杂 | 大规模采集平台、事件流处理 |
| RabbitMQ + 自定义调度 | 路由灵活、确认机制清晰 | 吞吐通常不如 Kafka | 任务精细编排、消费确认要求高 |

新手可以先用统一取号窗口理解。Redis 队列像一个共享柜台，够快、够简单；Kafka 更像大型物流分拨中心，规则更强，但搭建和维护成本高得多。若只是几百并发、几百万 URL、团队又希望快速上线，Scrapy-Redis 往往是更合理的起点。

适用边界可以概括为：

- 适合 URL 总量可预估的采集任务。
- 适合并发规模约 100 到 1000 的中等体量系统。
- 适合容忍极低比例去重误报的场景。
- 适合希望快速横向扩容、无需复杂部署的团队。
- 不适合要求事务一致、严格不丢不重的任务系统。
- 不适合必须零误报、零漏抓的高完整性采集。

如果业务后续扩大，通常不是一开始就替换掉 Scrapy，而是逐步把调度、代理、任务编排、监控拆出去。换句话说，Scrapy-Redis 更像一个性价比很高的第一代分布式爬虫架构，而不是所有阶段的终局架构。

---

## 参考资料

- `topic.alibabacloud.com`：Scrapy-Redis 分布式爬虫介绍，涵盖 Redis 共享请求队列、去重集合、队列切换等基础机制。
- `redis.io`：Redis Bloom Filter 文档，给出误报率、容量、哈希函数个数与空间占用的公式和参数说明。
- `scrapy-cluster.readthedocs.io`：Scrapy Cluster 关于 crawler、queue、失败重试和分布式任务流的说明，可用来理解故障恢复与任务再投递。
- `docs.scrapy.org`：Scrapy 官方实践文档，涉及下载延迟、并发控制、中间件、Cookie、User-Agent 等工程配置建议。
