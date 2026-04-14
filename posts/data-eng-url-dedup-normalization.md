## 核心结论

URL 去重的第一步不是“查数据库”，而是“先把 URL 规范化，再做指纹判断”。规范化指把文本写法不同、但语义上指向同一资源的 URL 统一成同一个标准字符串。只有这一步做对，后面的哈希、Redis Set、Bloom Filter 才有意义。

在爬虫里，去重通常发生在 frontier 层。frontier 可以理解为“待抓取 URL 的调度入口”。它的目标不是判断两页内容是否相同，而是判断“这个请求是否已经见过，是否值得再次入队”。因此，`HTTP://example.com/?b=2&a=1&utm_source=x#frag` 和 `http://example.com/?a=1&b=2` 应该在进入队列前就被识别为同一个目标。

Bloom Filter 是一种“用少量内存做近似成员判断”的数据结构。它非常省内存，但会有误报，也就是“没见过的 URL 被误判成见过”。Redis Set 是精确集合，不会误报，但内存成本更高。工程上常见做法不是二选一，而是分层：先用 Bloom Filter 挡住大部分重复，再用 Redis Set 或持久化哈希集做精确确认。

一个新手最容易忽略的点是：URL 规范化规则本身就是业务规则。是否忽略协议、是否保留大小写、是否删除某些参数，不是“语法题”，而是“你想把哪些 URL 视为同一资源”的系统设计选择。

| 原始 URL | 规范化后 | 是否视为同一目标 |
|---|---|---|
| `HTTP://example.com/?a=1&b=2&utm_source=foo#frag` | `http://example.com/?a=1&b=2` | 是 |
| `http://example.com:80/path` | `http://example.com/path` | 是 |
| `https://example.com/path/` | `https://example.com/path` | 常见实现视为是 |
| `https://example.com/?b=2&a=1` | `https://example.com/?a=1&b=2` | 是 |
| `https://example.com/item?id=1` | `https://example.com/item?id=1` | 与 `id=2` 不是同一目标 |

---

## 问题定义与边界

URL 去重要解决的问题是：避免同一个资源因为 URL 文本写法差异被重复下载、重复入队、重复存储。这里的“重复”是请求层重复，不是内容层重复。

请求层重复的意思是：两个 URL 在抓取系统里应当被视为同一个访问目标。内容层重复的意思是：两个不同 URL 下载回来后内容完全相同，或者高度相似。前者解决的是调度效率，后者解决的是内容索引和存储冗余，它们不是同一个模块。

例如下面两个 URL：

- `HTTP://docs.example.com:80/api?b=2&a=1#top`
- `http://docs.example.com/api?a=1&b=2`

如果你的抓取系统把它们当成两个任务，就会重复发请求、占用带宽、污染统计结果。规范化后，这两个 URL 都会映射到同一个标准串，因此只会入队一次。

这里要明确边界：

| 问题 | 负责层级 | 处理对象 | 典型方法 |
|---|---|---|---|
| URL 规范化去重 | Frontier / Fetch Queue | URL 字符串 | 规范化 + 哈希 + Set/Bloom |
| 内容去重 | 存储/索引层 | HTML、正文、二进制内容 | SimHash、MinHash、正文摘要 |
| 增量抓取判断 | 调度 + 抓取策略 | 资源是否变化 | ETag、Last-Modified、内容指纹 |

一个“玩具例子”最能说明边界。假设你抓一个博客站：

- `/post?id=123&utm_source=twitter`
- `/post?id=123&utm_source=wechat`

这两个 URL 的正文大概率一样，所以 URL 层应当去重。
但下面两个 URL：

- `/category/python?page=1`
- `/category/python?page=2`

它们的结构相似，甚至正文有大量重复导航区块，但 URL 层不能去重，因为它们对应不同分页资源。

所以，URL 去重的判断标准不是“长得像不像”，而是“是否应该被系统当成同一个请求目标”。

---

## 核心机制与推导

URL 规范化通常按组件处理。组件就是 URL 的几个部分：scheme、host、port、path、query、fragment。它们分别表示协议、主机、端口、路径、查询参数、片段锚点。

常见规则如下：

1. scheme 转小写，例如 `HTTP` 变 `http`
2. host 转小写，例如 `Example.COM` 变 `example.com`
3. 删除默认端口，`http:80`、`https:443` 不再保留
4. 删除 fragment，`#section1` 只影响浏览器定位，不影响服务端资源
5. 统一 path 末尾斜杠策略，例如 `/path/` 归一成 `/path`
6. 查询参数按 key/value 排序，消除顺序差异
7. 删除追踪参数，例如 `utm_source`、`utm_campaign`、`gclid`
8. 需要时保留重复参数，例如 `a=1&a=2` 不能简单丢掉一个

做完规范化后，再把结果字符串做哈希。哈希可以理解为“把任意长字符串压成固定长度指纹”。如果规范化方向一致，那么多个文本变体就会产生同一个 fingerprint。

一个新手视角的玩具例子：

- 原始：`HTTP://example.com/?a=1&b=2&utm_source=foo#frag`
- 处理后：`http://example.com/?a=1&b=2`
- 再做 `sha256`，得到统一指纹

这样一来，无论 URL 变体以什么顺序出现，只要归一结果相同，就会命中相同去重记录。

接下来是 Bloom Filter 的推导。Bloom Filter 用一个长度为 $m$ 的位数组保存集合状态，用 $k$ 个哈希函数映射元素位置，插入 $n$ 个元素后，其误报率近似为：

$$
p=\left(1-e^{-kn/m}\right)^k
$$

其中：

- $n$ 是预计存储元素数
- $m$ 是位数组长度，单位是 bit
- $k$ 是哈希函数个数
- $p$ 是误报率

最优哈希函数个数是：

$$
k=\frac{m}{n}\ln 2
$$

给定目标误报率 $p$，可反推所需位数：

$$
m=-\frac{n\ln p}{(\ln 2)^2}
$$

这几个公式的工程意义很直接：你先估计一轮任务会有多少唯一 URL，再决定愿意接受多大的误报率，就能推算 Bloom Filter 该分配多少内存。

以 $n=1{,}000{,}000$、目标误报率 $p=1\%$ 为例：

$$
m=-\frac{1{,}000{,}000 \cdot \ln 0.01}{(\ln 2)^2}\approx 9.59 \times 10^6 \text{ bits}
$$

换算成字节约为：

$$
\frac{9.59 \times 10^6}{8}\approx 1.2 \text{ MB}
$$

此时最优哈希函数个数约为：

$$
k=\frac{m}{n}\ln 2 \approx 6.64
$$

因此工程上常取 $k=7$。这就是常见说法“100 万 URL、1% 误报，大约 1.2MB”的来源。

真实工程例子更能看出价值。假设一个新闻聚合爬虫每天发现 5000 万个候选 URL，如果每个都直接去查数据库是否已访问，数据库会被变成热点判重服务，QPS 很快失控。更合理的做法是：

- 先规范化
- 先查本地或 Redis Bloom Filter
- Bloom 说“没见过”时，再写入精确集合并入队
- Bloom 说“见过”时，可再查 Redis Set 做二次确认

这样大多数重复 URL 会在内存层被截断，精确存储只处理少量边界情况。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它展示两件事：

1. 如何做 URL 规范化与指纹生成
2. 如何用“近似集合 + 精确集合”的两级去重思路模拟爬虫入队

```python
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import hashlib
import math

TRACKING_PREFIXES = ("utm_",)
TRACKING_KEYS = {"gclid", "fbclid"}

def normalize_url(url: str) -> str:
    parts = urlsplit(url)

    scheme = parts.scheme.lower() or "http"
    host = (parts.hostname or "").lower()

    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = host
    elif port is None:
        netloc = host
    else:
        netloc = f"{host}:{port}"

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query_items = []
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        kl = k.lower()
        if kl in TRACKING_KEYS or any(kl.startswith(prefix) for prefix in TRACKING_PREFIXES):
            continue
        query_items.append((k, v))

    query_items.sort(key=lambda x: (x[0], x[1]))
    query = urlencode(query_items, doseq=True)

    # fragment 被直接丢弃
    normalized = urlunsplit((scheme, netloc, path, query, ""))
    return normalized

def fingerprint(url: str) -> str:
    normalized = normalize_url(url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

class SimpleBloom:
    def __init__(self, bit_size: int, hash_count: int):
        self.bit_size = bit_size
        self.hash_count = hash_count
        self.bits = bytearray((bit_size + 7) // 8)

    def _hashes(self, item: str):
        base = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(base[:16], "big")
        h2 = int.from_bytes(base[16:], "big")
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.bit_size

    def add(self, item: str):
        for pos in self._hashes(item):
            self.bits[pos // 8] |= 1 << (pos % 8)

    def __contains__(self, item: str):
        for pos in self._hashes(item):
            if not (self.bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True

class Deduper:
    def __init__(self):
        self.bloom = SimpleBloom(bit_size=1_000_003, hash_count=7)
        self.exact = set()  # 真实工程里可替换为 Redis Set / RocksDB / SQLite

    def should_enqueue(self, url: str) -> bool:
        fp = fingerprint(url)

        # Bloom 未命中，直接认为是新 URL
        if fp not in self.bloom:
            self.bloom.add(fp)
            self.exact.add(fp)
            return True

        # Bloom 命中后，去精确集合确认，避免误报导致漏抓
        if fp in self.exact:
            return False

        self.exact.add(fp)
        return True

u1 = "HTTP://example.com/?a=1&b=2&utm_source=foo#frag"
u2 = "http://example.com/?b=2&a=1"

assert normalize_url(u1) == "http://example.com/?a=1&b=2"
assert normalize_url(u2) == "http://example.com/?a=1&b=2"
assert fingerprint(u1) == fingerprint(u2)

d = Deduper()
assert d.should_enqueue(u1) is True
assert d.should_enqueue(u2) is False

n = 1_000_000
p = 0.01
m = -n * math.log(p) / (math.log(2) ** 2)
k = (m / n) * math.log(2)

assert int(m / 8 / 1024 / 1024 * 10) >= 11  # 约 1.2MB
assert round(k) == 7
```

上面的 `exact = set()` 只是演示。真实系统里，常见替换方案有：

- Redis Set：适合分布式共享状态
- SQLite / RocksDB：适合单机持久化
- 数据库唯一索引：适合吞吐不高、但需要强一致判重的场景

更接近生产的入队流程可以写成这样：

```python
def enqueue_if_new(raw_url, bloom, exact_store, queue):
    normalized = normalize_url(raw_url)
    fp = fingerprint(normalized)

    if fp not in bloom:
        bloom.add(fp)
        exact_store.add(fp)
        queue.push(raw_url)
        return True

    if exact_store.contains(fp):
        return False

    exact_store.add(fp)
    queue.push(raw_url)
    return True
```

注意这里队列里推入的是原始 URL 还是规范化 URL，要提前定规则。多数系统会同时保留两者：

- 原始 URL：方便排查来源、回溯日志
- 规范化 URL：方便统一去重与统计

---

## 工程权衡与常见坑

URL 去重最常见的错误，不是公式算错，而是系统分层做错。

第一类坑是“每个 URL 都查数据库”。这在小项目里看起来最直接，但规模一上来，数据库立刻变成瓶颈。假设系统每秒发现 8 万个链接候选，哪怕只做一次主键查询，也是在把数据库当高频缓存用，这通常不是它擅长的事。

第二类坑是“只用 Bloom Filter”。Bloom Filter 省内存，但它有误报。误报的后果是：某个从未抓过的 URL，被误判成“已经抓过”，结果直接漏抓。如果业务允许少量漏抓，比如广告链接发现、低价值页面采集，只用 Bloom 可能可接受；如果业务要求高召回，例如商品变价监控、合规审计抓取，只用 Bloom 往往不够。

第三类坑是“规范化规则过度激进”。例如把 `http` 和 `https` 强行视为同一资源，或者把所有查询参数都删掉。这会把本来不同的资源错误合并。比如商品详情页 `?sku=1` 和 `?sku=2` 显然不能被合并成一个 URL。

第四类坑是“去重状态不持久化”。Bloom、内存 Set、本地 LRU 都可能随着进程退出而消失。重启后如果没有恢复机制，系统会把旧 URL 当成新 URL 再抓一遍，既浪费资源，又破坏增量抓取语义。

| 常见坑 | 后果 | 规避策略 |
|---|---|---|
| 每次都查数据库 | 数据库高 QPS、延迟抖动 | 前置 Bloom/Redis，减少精确查询 |
| 只用 Bloom Filter | 误报导致漏抓 | Bloom + Redis Set 二级确认 |
| 规范化过度 | 不同资源被合并 | 参数白名单/黑名单按站点配置 |
| 状态不落盘 | 重启后重复抓取 | Redis RDB/AOF、SQLite/RocksDB、WAL |
| 多机各自判重 | 集群间重复抓取 | 共享 Redis / 分片一致哈希 |

真实工程例子：一个增量新闻爬虫每天跑 24 次，每次只希望抓变化页面。系统可以把“已见过 URL 的 fingerprint”保存在 Redis，同时保存 `ETag` 或 `Last-Modified`。下次调度时先做 URL 去重，再用条件请求判断资源是否变化。这样，URL 层避免重复访问，内容层避免无效下载，两层职责清楚，系统成本也更可控。

---

## 替代方案与适用边界

不同去重方案没有绝对优劣，只有适用边界。

| 方案 | 是否误报 | 内存成本 | 持久化能力 | 适用场景 |
|---|---|---|---|---|
| 只用 Redis Set | 否 | 高 | 强，依赖 Redis 持久化 | 中小规模、高精度 |
| 只用 Bloom Filter | 是 | 很低 | 取决于实现 | 超大规模、可接受少量漏抓 |
| Bloom + Redis Set | Bloom 有误报，但整体可精确 | 中 | 强 | 大多数工程折中方案 |
| 数据库唯一索引 | 否 | 外部存储承担 | 强 | 吞吐较低、审计要求高 |
| 内容指纹 + 增量策略 | URL 层之外的补充 | 中到高 | 强 | 页面内容重复多、重抓成本高 |

“只用 Bloom Filter”和“Bloom + Set”的差异，新手可以这样理解：

- 只用 Bloom：速度快、省内存，但命中就是“算了，不抓了”
- Bloom + Set：先用 Bloom 过滤大头，命中后再问精确集合一次，防止误杀

如果业务还要求“增量抓取”，URL 去重也不是终点。增量抓取指“只重新抓取可能变化的资源”。常见做法是记录上次抓取时的 `ETag` 或 `Last-Modified`，下次带上条件请求头：

- `If-None-Match: <etag>`
- `If-Modified-Since: <time>`

如果服务端返回 `304 Not Modified`，说明资源没变，就不必重新下载正文。这里 URL 去重解决的是“同一个请求不要重复入队”，而增量抓取解决的是“同一个 URL 以后要不要重抓”。两者经常一起使用，但不是一个问题。

适用边界可以简化为三条：

1. 你最在意准确性，规模不大：优先 Redis Set 或数据库唯一索引
2. 你最在意内存和吞吐，允许少量漏抓：可以只用 Bloom
3. 你既在意规模，又不能接受明显漏抓：Bloom + 精确集合 + 持久化恢复

---

## 参考资料

| 资料 | 作者/平台 | 主要贡献点 |
|---|---|---|
| https://www.systemoverflow.com/learn/resilience-patterns/web-crawler-design/deduplication-strategies-url-normalization-and-content-fingerprinting | System Overflow | 说明 URL 规范化与内容指纹在爬虫中的分层作用 |
| https://oneuptime.com/blog/post/2026-03-31-redis-bloom-filter-url-deduplication/view | OneUptime | 给出 URL 规范化与 Redis Bloom 判重的工程示例 |
| https://www.sciencedirect.com/topics/computer-science/bloom-filter | ScienceDirect | Bloom Filter 基本定义与误报率公式来源 |
| https://singhajit.com/data-structures/bloom-filter/ | Singhajit | 提供百万级元素、1% 误报时的直观数值示例 |
| https://www.whiteboardscale.com/topics/web-crawler/concepts/url-dedup-bloom-filter | WhiteboardScale | 说明大规模爬虫下 Bloom Filter 的空间收益 |
| https://www.datainterview.com/blog/system-design-design-web-crawler | DataInterview | 讨论数据库瓶颈、Redis 组合与去重系统设计 |
| https://www.firecrawl.dev/glossary/web-crawling-apis/incremental-crawling | Firecrawl | 解释增量抓取、状态复用与变化检测语义 |
