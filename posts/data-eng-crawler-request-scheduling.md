## 核心结论

爬虫请求调度器的核心，不是“谁先入队谁先抓”，而是把“URL 价值”和“站点礼貌约束”放进同一个决策过程。最实用的统一指标是“域名就绪时间”。域名就绪时间指某个域名下一次允许发请求的最早时刻，白话说，就是这个站点的“下次可出发时间”。

如果一个 URL 优先级很高，但它所属域名还没到可请求时间，那么调度器不能直接发。相反，它应该先去找其他已经到期、且同样有价值的域名继续工作。这样才能同时满足三件事：

1. 高价值 URL 尽量先抓。
2. 单域名请求间隔不被打爆。
3. 多个域名之间轮询，整体吞吐不至于因为某一个站点的 `crawl-delay` 降到很低。

最常见的统一公式是：

$$
next\_ready = last\_fetch\_time + \max(crawl\_delay,\ adaptive\_backoff)
$$

其中，`crawl-delay` 是站点通过 `robots.txt` 明示或你主动设定的最小间隔，`adaptive_backoff` 是自适应退避，白话说，就是当站点变慢、报错、返回 429 时，系统临时把等待时间拉长。

可以把调度器想成多车道收费站。每个域名是一条车道，每条车道前面都有一个倒计时。调度器每次只从“倒计时已经结束”的车道里，挑出最值得先过的一辆车。这样既不会堵死某一条车道，也不会插队撞规则。

| 因素 | 作用对象 | 决定什么 | 典型后果 |
|---|---|---|---|
| URL 优先级 | 单个 URL | 先抓谁 | 重要页面更早进入索引 |
| `crawl-delay` | 单个域名 | 多久才能再抓一次 | 防止对同站过度打压 |
| 域名并发限制 | 单个域名 | 同时能有几个请求在飞 | 防止连接数过多被封 |
| 自适应退避 | 单个域名 | 出错后额外等多久 | 遇到限流时自动降速 |

---

## 问题定义与边界

问题定义可以写得很直接：在不违反站点访问规则的前提下，以尽量高的效率抓取尽量高价值的 URL。

这里的“高价值”通常来自业务目标，比如：

- 商品页比列表页优先级更高。
- 更新频繁的页面比静态页面优先级更高。
- 已知能产出结构化数据的页面比未知页面更高。

这里的“访问规则”至少包括四类边界：

| 限制或需求 | 对应策略 |
|---|---|
| `robots.txt` 允许/禁止路径 | 首次遇到新域名先抓规则并缓存 |
| `crawl-delay` | 计算域名 `next_ready` |
| 域名级并发限制 | 为每个域名维护 in-flight 计数 |
| 断点续传 | 把 frontier 和域名状态持久化 |

frontier 指待抓取集合，白话说，就是“未来还要抓的 URL 池子”。

一个新手最容易忽略的边界是：调度器不是只看 URL，而是要同时看 URL、域名状态、robots 缓存、失败历史。否则看起来“优先级正确”，实际上会不断撞限流。

看一个最小场景。假设默认 `crawl-delay = 1s`，且单域并发限制为 1：

- `a.com` 有 100 个高优先级 URL
- `b.com` 有 5 个中优先级 URL
- `c.com` 有 5 个中优先级 URL

如果你只按优先级抓，结果会变成：不断打 `a.com`，很快触发封禁或 429。
如果你按域名就绪时间调度，结果会变成：`a.com` 每秒最多 1 次，同时空档时间抓 `b.com` 和 `c.com`。这样总吞吐更高，也更稳。

所以边界不是“单次挑哪个 URL”，而是“在当前时刻，哪些域名具备合法发送资格”。只有先过这个边界，优先级排序才有意义。

---

## 核心机制与推导

一个可落地的请求调度器，通常会拆成三层状态：

1. URL 层：待抓取 URL 的优先级。
2. 域名层：这个域名什么时候允许再发。
3. 规则层：这个域名哪些路径允许抓、最小间隔是多少。

最常见的结构是两个堆加一个缓存。

- 优先级队列：按 URL 价值排序。
- 域名最小堆：按 `next_ready` 排序。
- `robots.txt` 缓存：缓存允许规则、禁止规则、`crawl-delay`、过期时间。

这里“堆”是堆数据结构，白话说，就是一种能快速找出“最小值”或“最大值”的队列。

核心推导从域名约束开始。对每个域名 $d$，维护：

$$
next\_ready(d)=last\_fetch\_time(d)+\max(crawl\_delay(d),\ adaptive\_backoff(d))
$$

如果还有域名并发上限 $k_d$，那就再加一个条件：

$$
inflight(d) < k_d
$$

于是，一个请求可被调度，当且仅当同时满足：

$$
now \ge next\_ready(d)
$$

并且

$$
inflight(d) < k_d
$$

并且 URL 路径满足 `robots.txt` 规则。

这套逻辑的关键价值是：优先级不再直接决定“立刻发”，而是决定“在合法域名集合中先发谁”。

一个简化时序可以写成：

1. 从域名最小堆看最早到期的域名。
2. 如果它还没到 `next_ready`，全局等待到该时间或等待其他事件。
3. 如果它已经到期，从该域名自己的待抓队列里取最高优先级 URL。
4. 检查路径是否被 `robots.txt` 允许。
5. 发请求，增加 `inflight`。
6. 请求结束后，更新 `last_fetch_time`、错误计数、自适应退避，并重新计算 `next_ready`。
7. 若该域名还有待抓 URL，则重新压回域名堆。

为什么不直接用一个全局优先级堆？因为那样每次弹出一个高优 URL，都还要检查其域名是否到期。若没到期，再塞回去，会导致大量无效弹出和重排。URL 越多，浪费越明显。

因此更稳定的做法是“域名先合法，再域内选高优”。也就是：

- 全局不直接调度 URL。
- 全局先调度“哪个域名现在可以工作”。
- 域名内部再调度“哪个 URL 值得先抓”。

这个设计天然适合“多域轮询”。玩具例子如下：

- `news.com` 的 `next_ready = 10:00:01`
- `shop.com` 的 `next_ready = 10:00:00`
- `forum.com` 的 `next_ready = 10:00:03`

当前时间是 `10:00:00.5`。这时只能调度 `shop.com`。即使 `news.com/article-1` 的优先级最高，也必须等 `10:00:01`。如果 `shop.com` 当前没有可抓 URL，就继续看下一个到期域名，而不是违规抢 `news.com`。

这个机制表面上像“慢”，实际上更快。原因是调度器不会把时间浪费在反复命中同一个受限域名上，而是持续从全局可用资源里榨吞吐。

---

## 代码实现

下面给出一个单机可运行的简化实现。它不发真实网络请求，但完整演示了三个核心点：

1. 域名级优先级队列。
2. `next_ready` 调度。
3. 失败后自适应退避。

```python
from dataclasses import dataclass, field
from urllib.parse import urlparse
import heapq


@dataclass(order=True)
class DomainState:
    next_ready: float
    domain: str = field(compare=False)
    crawl_delay: float = field(default=1.0, compare=False)
    adaptive_backoff: float = field(default=0.0, compare=False)
    inflight: int = field(default=0, compare=False)
    max_inflight: int = field(default=1, compare=False)
    last_fetch_time: float = field(default=-10**9, compare=False)


class Scheduler:
    def __init__(self):
        self.domain_heap = []  # min-heap by next_ready
        self.domain_states = {}
        self.domain_queues = {}  # domain -> max-heap by priority

    def add_request(self, url: str, priority: int, crawl_delay: float = 1.0):
        domain = urlparse(url).netloc
        if domain not in self.domain_states:
            state = DomainState(next_ready=0.0, domain=domain, crawl_delay=crawl_delay)
            self.domain_states[domain] = state
            self.domain_queues[domain] = []
            heapq.heappush(self.domain_heap, state)
        heapq.heappush(self.domain_queues[domain], (-priority, url))

    def pop_ready(self, now: float):
        while self.domain_heap:
            state = heapq.heappop(self.domain_heap)
            if state.next_ready > now:
                heapq.heappush(self.domain_heap, state)
                return None
            if state.inflight >= state.max_inflight:
                heapq.heappush(self.domain_heap, state)
                return None
            queue = self.domain_queues[state.domain]
            if not queue:
                continue
            priority, url = heapq.heappop(queue)
            state.inflight += 1
            return state.domain, url, -priority
        return None

    def finish(self, domain: str, now: float, ok: bool):
        state = self.domain_states[domain]
        state.inflight -= 1
        state.last_fetch_time = now
        if ok:
            state.adaptive_backoff = 0.0
        else:
            state.adaptive_backoff = max(1.0, state.adaptive_backoff * 2 or 1.0)
        state.next_ready = now + max(state.crawl_delay, state.adaptive_backoff)
        heapq.heappush(self.domain_heap, state)


s = Scheduler()
s.add_request("https://a.com/p1", priority=100, crawl_delay=1.0)
s.add_request("https://a.com/p2", priority=90, crawl_delay=1.0)
s.add_request("https://b.com/p1", priority=80, crawl_delay=1.0)

job1 = s.pop_ready(now=0.0)
assert job1[0] == "a.com"
assert job1[1] == "https://a.com/p1"

s.finish("a.com", now=0.0, ok=True)

job2 = s.pop_ready(now=0.0)
assert job2[0] == "b.com"
assert job2[1] == "https://b.com/p1"

s.finish("b.com", now=0.0, ok=False)

job3 = s.pop_ready(now=0.5)
assert job3 is None  # a.com 和 b.com 都没到 next_ready

job4 = s.pop_ready(now=1.0)
assert job4[0] == "a.com"
assert job4[1] == "https://a.com/p2"
```

这段代码用了“域名一个队列”的方式，而不是“全局 URL 一个堆”。原因很实际：单机下这样更容易保证域名轮询，也更容易把 `crawl-delay` 与并发状态放进同一个状态对象。

核心数据结构职责如下：

| 数据结构 | 键 | 职责 |
|---|---|---|
| 域名最小堆 | `next_ready` | 找出最早可请求的域名 |
| 域内优先队列 | `priority` | 在同一域名内部先抓高价值 URL |
| `robots.txt` 缓存 | `domain` | 缓存允许规则、禁止规则、`crawl-delay` |
| 状态存储 | `domain` / `url` | 保存断点续传所需的 frontier 和域名状态 |

`robots.txt` 模块的工作流通常是：

1. 首次遇到新域名，先请求 `https://domain/robots.txt`。
2. 解析出允许路径、禁止路径、`crawl-delay`、缓存时间。
3. 如果抓取失败，不直接放弃，而是短时间退避后重试。
4. 周期刷新缓存，避免长期使用旧规则。

伪代码可以写成：

```text
loop:
  domain = pop_min_next_ready_domain()
  if domain.next_ready > now:
      sleep_until(domain.next_ready)
      continue

  if robots_cache.missing(domain):
      fetch_and_parse_robots(domain)
      recompute_domain_state(domain)
      push_back(domain)
      continue

  url = pop_highest_priority_url(domain)
  if not robots_cache.allowed(domain, url.path):
      mark_skipped(url)
      push_back(domain)
      continue

  send_request(url)
  on_complete:
      update_backoff(domain, response)
      domain.next_ready = now + max(crawl_delay, adaptive_backoff)
      push_back(domain)
```

真实工程里，如果是多进程或多机，域名堆可以迁移到 Redis Sorted Set。Sorted Set 是有序集合，白话说，就是按分数排序的集合，适合把 `next_ready` 作为 score。这样多个 worker 都能围绕同一份域名状态协作。

一个真实工程例子是电商聚合或商品情报系统。假设系统要抓数千个商家域名，目标是尽快更新价格变更页面。此时常见做法是：

- Redis Sorted Set 存“已到期可调度域名”。
- Redis Hash 存域名状态、错误次数、最后抓取时间。
- 每个域名对应一个 URL 优先队列。
- worker 通过分布式锁抢占一个域名调度权，避免多个节点同时打同一站点。

这样机器重启后，只要 Redis 还在，frontier 和域名状态都能恢复，断点续传成本很低。

---

## 工程权衡与常见坑

调度器的难点不在“能不能发请求”，而在“长期运行是否稳定”。下面这些坑最常见。

| 坑或风险 | 后果 | 规避手段 |
|---|---|---|
| 忽略 `robots.txt` | 被封禁、法律和合规风险 | 首次遇域名先抓规则并缓存 |
| 把 `crawl-delay` 写死成全局值 | 对部分站点过快，对部分站点过慢 | 按域名维护独立延迟 |
| 不做自适应退避 | 遇到 429/503 时持续撞墙 | 指数退避，错误恢复后逐步回落 |
| 只存 URL 不存域名状态 | 重启后间隔控制失真 | 同时持久化 `last_fetch_time`、`next_ready`、错误计数 |
| 只做单 URL 优先级，不做域名轮询 | 高优域名长期霸占带宽 | 先域名可用，再域内选高优 |
| 分布式环境不加锁 | 多 worker 同时抓同域 | 用 Redis 锁或 ZooKeeper 协调 |

一个典型误区是：看见 429 只重试 URL，不调整域名状态。这样会导致同域后续请求继续正常发出，相当于调度器没有吸收错误反馈。正确做法是把 429 解释为“域名级节流信号”，直接抬高该域名的 `adaptive_backoff`。

另一个误区是：frontier 持久化了，但域名 `next_ready` 没持久化。结果是系统重启后，所有待抓 URL 立刻重新发，形成冷启动洪峰。这个问题在线上比“丢几条 URL”更危险，因为它会瞬间打穿目标站点限制。

真实工程里，Semantics3 这类近似分布式爬虫的经验很典型：URL 状态和优先级放在 Redis 中，调度层做跨 worker 分发，重启后依靠 Redis 快速恢复。它不一定提供严格全局顺序，但在吞吐、恢复速度和实现复杂度之间取得了实用平衡。

还有一个容易低估的问题是 robots 缓存刷新。`robots.txt` 不是永远不变。如果缓存永不过期，站点新增禁止路径后，你的系统可能继续违规抓取。工程上通常会给 robots 缓存设置 TTL，TTL 指生存时间，白话说，就是“多久后必须重新拉一遍规则”。

---

## 替代方案与适用边界

并不是所有爬虫都需要完整的域名就绪堆。是否上这套设计，取决于规模、礼貌要求和分布式需求。

| 方案 | 一致性 | 扩展性 | 实现成本 | 适用边界 |
|---|---|---|---|---|
| 简化版优先队列 + `sleep` | 低 | 低 | 低 | 单域、低并发、小脚本 |
| 单机域名就绪堆 | 中 | 中 | 中 | 多域、需要礼貌抓取、单机或单进程 |
| Redis Sorted Set 调度 | 中高 | 高 | 中高 | 多 worker、需要断点续传 |
| ZooKeeper/Curator 延迟队列 | 高 | 中高 | 高 | 强协调、节点多、调度一致性要求高 |

简化版方案可以非常简单：

1. 所有 URL 放一个优先队列。
2. 取一个 URL。
3. 请求后 `sleep(crawl_delay)`。
4. 继续下一个。

这种方案只有在以下边界里成立：

- 基本只有一个域名。
- 请求量不大。
- 对吞吐要求很低。
- 不需要复杂恢复。

一旦进入多域场景，这种方案的问题就暴露了。因为它会把一个域名的等待时间传播给整个系统，导致其他本可立即抓取的域名也被迫一起等。

如果系统已经是分布式，且你不想自己处理抢锁、顺序竞争、节点协调，可以考虑 ZooKeeper/Curator 提供的分布式延迟队列或优先级队列。它的优势是协调能力强，缺点是系统复杂度、维护成本、故障诊断门槛都更高。

所以边界可以直接这样判断：

- 你只是写一个一次性数据采集脚本，用简单队列。
- 你要长期跑、多域轮询、遵守礼貌策略，用域名就绪堆。
- 你有多个 worker、要断点续传、要共享调度状态，用 Redis。
- 你要求强协调、严格分布式一致性，再考虑 ZooKeeper/Curator。

本质上，不同方案是在三件事之间取平衡：礼貌性、吞吐、实现复杂度。没有哪种设计在所有场景都最优。

---

## 参考资料

1. System Design Sandbox, *Design Web Crawler*  
   重点：请求调度、域名级轮询、`robots.txt` 处理、`crawl-delay` 与礼貌抓取设计。  
   https://www.systemdesignsandbox.com/learn/design-web-crawler

2. Semantics3 Engineering, *How We Built Our 60-Node Almost-Distributed Web Crawler*  
   重点：Redis 支撑的近似分布式调度、状态恢复、跨 worker 任务分发。  
   https://medium.com/engineering-semantics3/how-we-built-our-60-node-almost-distributed-web-crawler-3b086e3e9ef4

3. Firecrawl Glossary, *What Is Polite Crawling?*  
   重点：礼貌抓取、`robots.txt`、退避、速率控制的工程实践。  
   https://www.firecrawl.dev/glossary/web-crawling-apis/what-is-polite-crawling

4. Apache Curator Documentation  
   重点：分布式队列、延迟队列、依赖 ZooKeeper 的协调能力。  
   https://curator.apache.org/
