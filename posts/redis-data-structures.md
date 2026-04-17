## 核心结论

Redis 不是“只有几种数据类型”的键值库，而是“用少数基础结构组合出多种对象”的内存数据库。基础结构可以概括为六类：SDS、链表/quicklist、字典、跳跃表、整数集合、压缩列表。这里的“基础结构”就是底层存储骨架，决定了时间复杂度、内存占用和持久化时的数据组织方式。

它们的分工很明确：

| 基础结构 | 主要服务对象 | 典型操作复杂度 | 设计目标 |
|---|---|---:|---|
| SDS | String | 取长度 $O(1)$ | 替代 C 字符串，支持二进制安全与预分配 |
| 链表 / quicklist | List | 头尾插入 $O(1)$ | 保持顺序，同时控制节点内存碎片 |
| 字典 | Hash、Set 的大对象编码 | 查找/插入平均 $O(1)$ | 高并发下快速读写 |
| 跳跃表 | ZSet | 查找/插入平均 $O(\log n)$ | 有序、排名、范围扫描 |
| 整数集合 intset | Set 的小整数场景 | 查找 $O(\log n)$，插入 $O(n)$ | 小集合节省内存 |
| 压缩列表 ziplist | 小 List/Hash/ZSet 的旧版紧凑编码 | 顺序访问快，插入中间偏贵 | 用连续内存换空间效率 |

核心判断只有一句：Redis 用不同编码在“省内存”和“低延迟”之间动态切换，小对象尽量压缩，大对象转为通用结构，持久化则由 RDB 和 AOF 分别负责“快照恢复”和“写命令重放”。

---

## 问题定义与边界

本文讨论的是 Redis 内部对象编码，而不是命令语法。重点问题是：为什么 `SET`、`HSET`、`LPUSH`、`ZADD` 这些看起来完全不同的命令，最终都能在内存里保持很低延迟？

边界要先说清楚：

1. 这里的“压缩列表 ziplist”主要对应 Redis 6.2 及更早版本。
2. Redis 7 之后，很多小对象编码已经逐步改为 `listpack`，但设计思想没有变，仍然是“紧凑顺序存储”。
3. 列表对象今天通常看到的是 `quicklist`，它不是单纯双向链表，而是“链表节点里再放紧凑片段”的折中结构。
4. 持久化不是独立于数据结构之外的话题。RDB 和 AOF 的成本，会直接影响你敢不敢把 Redis 当主存储。

一个新手最容易理解的玩具例子是小列表：

- 如果只存 `["Alice", "Bob", "Charlie"]`，用连续内存存三项更省。
- 如果是百万级消息队列，频繁头尾进出、偶尔压缩、还要避免巨大连续内存搬移，就更适合 `quicklist`。

一个真实工程例子是排行榜：

- 排行榜要求按分数排序，同时支持 `TOP 100`、查询某人的排名、查询分数区间。
- 如果用普通数组，插入一个新分数往往要移动后面的元素，代价接近 $O(n)$。
- Redis 选跳跃表，是因为它能把插入和范围查询稳定在平均 $O(\log n)$。

常见阈值配置体现了这个边界思想：

| 对象 | 小对象紧凑编码阈值示例 | 超过后转为 |
|---|---|---|
| Hash | `hash-max-ziplist-entries`、`hash-max-ziplist-value` | hashtable |
| ZSet | `zset-max-ziplist-entries`、`zset-max-ziplist-value` | skiplist + dict |
| Set | `set-max-intset-entries` | hashtable |
| List | `list-max-ziplist-size`、`list-compress-depth` | quicklist 内部节点调整 |

可以把这些阈值理解成一条经验规则：当 entry 数量和单个 value 字节数都较小时，紧凑编码更划算；一旦越界，就转向更通用的结构。

---

## 核心机制与推导

### 1. SDS：让字符串长度变成 $O(1)$

SDS 是简单动态字符串，白话说就是“把字符串长度和剩余空间直接记在头部”。因此不需要像 C 字符串那样每次遍历到 `\0` 才知道长度。

$$
\text{strlen(SDS)} = O(1)
$$

因为长度直接读取 `len` 字段，而不是扫描整个字节数组。

这带来三个结果：

- 读长度快。
- 支持二进制安全，内容里可以有 `\0`。
- 追加时能利用 `free` 预留空间，减少频繁 `malloc`。

### 2. quicklist：用分段顺序存储替代“大链表”或“大数组”

早期 Redis 列表可以理解为链表，后来主流实现是 `quicklist`。它的思路是：外层是链，内层是 ziplist 或 listpack 片段。这样比“每个元素一个链表节点”更省指针开销，也比“所有元素一整块连续内存”更容易做局部插入和裁剪。

玩具例子：

- 一个列表只有 20 个短字符串，放一个紧凑片段最省。
- 一个列表有 100 万条消息，不会真的变成 100 万个独立小对象，而是被分成多个片段，每段控制大小。

### 3. 字典：双哈希表 + 渐进 rehash

字典是 Redis 的核心。白话说，它就是哈希表，但不是一次性扩容，而是“边处理请求边搬家”。

其关键结构可抽象为：

- `ht[0]`：旧表
- `ht[1]`：新表
- `rehashidx`：当前搬迁进度

当需要扩容时，不是一次把全部桶迁完，而是：

$$
\text{每次普通读写} \Rightarrow \text{顺带迁移少量 bucket}
$$

查找逻辑因此变成：

- 如果没在 rehash，只查 `ht[0]`
- 如果正在 rehash，先查 `ht[0]`，再查 `ht[1]`

这保证了单次操作不会因为扩容卡住太久。平均查找仍然接近 $O(1)$，但最坏情况要考虑冲突链增长。

### 4. 跳跃表：多层索引换取有序访问

跳跃表可以理解为“给有序链表加多层捷径”。底层包含全部节点，上层只抽样保留部分节点。查找时先在高层快速跳，再逐步下降。

如果层高按概率生成，平均复杂度为：

$$
\text{search} = O(\log n), \quad \text{insert} = O(\log n)
$$

Redis 的有序集合不是只靠跳跃表。它通常同时维护：

- `dict`：按成员查分数快
- `skiplist`：按分数排序、范围查询快

这就是为什么 `ZADD`、`ZRANGE`、`ZRANK` 都能兼顾。

一个直观路径例子：

- 当前要找分数 64。
- 先在最高层从 10 跳到 40，再跳到 60。
- 发现下一个 80 超过目标，于是下降一层。
- 在更低层从 60 到 64，完成定位。

### 5. intset：小整数集合的紧凑数组

整数集合是“有序整数数组 + 自动升级位宽”。白话说，如果集合里都是整数，而且数量不大，不用哈希表更省。

它的两个关键点：

- 查找可用二分，复杂度 $O(\log n)$
- 插入要保持有序，可能搬移后续元素，复杂度 $O(n)$

如果从 `int16` 范围突然插入一个大整数，底层会升级到 `int32` 或 `int64`，然后整体重排。

### 6. ziplist：连续内存省空间，但中间修改昂贵

ziplist 是一段连续内存，entry 紧挨着排。每个 entry 通常包含前一项长度、当前项编码/长度和内容。它非常省空间，因为没有大量独立分配和指针。

但代价也明确：

- 顺序访问很合适
- 中间插入和删除可能触发大量内存搬移
- 某些场景还会引起连锁更新

所以 ziplist 适合“小、短、改动不剧烈”的对象，不适合超大热数据结构。

---

## 代码实现

下面用 Python 写三个最小实现，分别模拟 SDS、渐进 rehash 的查找逻辑和跳跃表层高分布。它们不是 Redis 源码，但能运行并验证核心机制。

```python
from dataclasses import dataclass
import random
import bisect
from collections import defaultdict

@dataclass
class SDS:
    buf: str
    free: int = 0

    @property
    def len(self) -> int:
        return len(self.buf)

    def append(self, s: str) -> None:
        if self.free < len(s):
            # 简化版预分配：至少多留与追加长度相同的空余
            self.free += len(s)
        self.buf += s
        self.free -= len(s)

s = SDS("redis", free=3)
assert s.len == 5
s.append("!")
assert s.buf == "redis!"
assert s.len == 6

class IncrementalDict:
    def __init__(self):
        self.ht0 = {}
        self.ht1 = None
        self.rehashing = False

    def start_rehash(self):
        self.ht1 = {}
        self.rehashing = True

    def move_one_key(self):
        if not self.rehashing or not self.ht0:
            return
        k = next(iter(self.ht0))
        self.ht1[k] = self.ht0.pop(k)
        if not self.ht0:
            self.ht0 = self.ht1
            self.ht1 = None
            self.rehashing = False

    def set(self, k, v):
        target = self.ht1 if self.rehashing else self.ht0
        target[k] = v
        self.move_one_key()

    def get(self, k):
        if k in self.ht0:
            self.move_one_key()
            return self.ht0[k]
        if self.rehashing and k in self.ht1:
            self.move_one_key()
            return self.ht1[k]
        self.move_one_key()
        raise KeyError(k)

d = IncrementalDict()
for i in range(5):
    d.set(f"k{i}", i)
d.start_rehash()
assert d.get("k1") == 1
d.set("k5", 5)
assert d.get("k5") == 5

def random_level(p=0.5, max_level=16):
    level = 1
    while random.random() < p and level < max_level:
        level += 1
    return level

levels = [random_level() for _ in range(5000)]
avg_level = sum(levels) / len(levels)
assert 1.5 < avg_level < 3.5  # p=0.5 时平均层数接近 2

nums = [10, 20, 40, 64, 80]
idx = bisect.bisect_left(nums, 64)
assert idx == 3 and nums[idx] == 64
```

SDS 头部结构示意可以写成：

```c
struct sdshdr {
    size_t len;
    size_t free;
    char buf[];
};
```

跳跃表节点的简化结构通常写成：

```c
typedef struct zskiplistNode {
    double score;
    struct zskiplistNode *backward;
    struct zskiplistLevel {
        struct zskiplistNode *forward;
        unsigned int span;
    } level[];
} zskiplistNode;
```

ziplist entry 的概念示意：

```text
<prevlen><encoding/entrylen><content>
```

关键配置和作用如下：

| 配置项 | 作用 | 实践建议 |
|---|---|---|
| `hash-max-ziplist-entries` | Hash 保持紧凑编码的最大 field 数 | 小对象多时可适当提高，先压测 CPU |
| `hash-max-ziplist-value` | Hash 中单 value 的最大字节数 | value 偏短时收益明显 |
| `zset-max-ziplist-entries` | 小型 ZSet 的紧凑编码阈值 | 排行榜一般很快越界 |
| `zset-max-ziplist-value` | ZSet member 最大字节数阈值 | member 太长会放大搬移成本 |
| `set-max-intset-entries` | Set 使用 intset 的最大元素数 | 仅适合全整数集合 |
| `list-max-ziplist-size` | quicklist 每个紧凑片段的大小控制 | 不宜盲目调大，避免单节点太重 |

---

## 工程权衡与常见坑

Redis 的优势不是“永远最快”，而是“在正确编码下足够快且足够省”。因此工程上最重要的是识别切换点。

常见坑如下：

| 坑 / 风险 | 触发条件 | 规避建议 |
|---|---|---|
| 调整 `*-max-ziplist-*` 后旧 key 不会自动重编码 | 已存在对象已经以旧编码落地 | 通过重建 key、重载数据或 `DUMP/RESTORE` 触发重建 |
| 误把 ziplist 适合所有小对象 | 高频中间插入、更新剧烈 | 热写场景优先考虑通用结构 |
| 误以为 RDB 不影响线上延迟 | 大实例 `fork()` 成本高 | 控制实例体量，避免超大单实例 |
| 误以为 AOF 最安全且无代价 | `BGREWRITEAOF` 期间额外 I/O 和内存压力 | 预留磁盘带宽与内存水位 |
| 排行榜只看 `ZADD` 不看范围查询 | 用户需要分页、前后邻居、按分数区间查 | 直接用 ZSet，不要自己拼数组 |

持久化的权衡可以用一个很直接的式子表示：

$$
\text{RDB 数据丢失窗口} \approx \text{两次快照之间的时间}
$$

如果你每 5 分钟快照一次，异常宕机时理论上可能丢最近几分钟写入。

而 AOF 在默认每秒刷盘策略下，通常可以理解为：

$$
\text{AOF 理论丢失窗口} \approx 1 \text{ 秒}
$$

但 AOF 文件更大，重写期间还会产生额外增量写入缓存和磁盘压力。

真实工程例子：

- 业务要做实时游戏排行榜，日活 200 万，榜单常驻 50 万用户。
- 这类场景直接选 ZSet，因为你需要 `ZADD`、查前 100、查某用户排名、查分数区间。
- 如果硬用 Hash 存用户到分数，再在应用层排序，每次读榜单都会把复杂度和网络开销转移到业务服务，最终更慢。

---

## 替代方案与适用边界

不是所有数据都该塞进 Redis。判断标准有两条：是否要求极低延迟，是否能接受内存成本。

| 场景规模 | 推荐结构 | 适用边界 |
|---|---|---|
| 很小的短字符串集合 | SDS / intset / ziplist 风格紧凑编码 | 极致省内存，更新不重 |
| 中等规模列表队列 | quicklist | 顺序读写多，头尾操作频繁 |
| 大量键值映射 | dict | 查找与更新为主 |
| 排行榜、延迟分布、有序窗口 | skiplist + dict | 需要排序、排名、范围查询 |
| 超大数据、强事务、复杂查询 | 外部数据库 | Redis 不适合替代全部主存储 |

替代方案的核心不是“哪个更先进”，而是“哪个成本结构更对”：

- 小型排行榜，成员很少、读多写少，可以先吃紧凑编码收益。
- 大型排行榜，一旦超出阈值，Redis 会自动切到更通用编码，用户不需要手动改命令。
- 如果榜单要做复杂多字段过滤、历史回溯、跨维度分析，数据库或搜索引擎比 Redis 更合适。

对版本边界还要补一句：

- 文章里把 ziplist 当作理解入口是合理的，因为它最能体现 Redis 的空间优化思想。
- 但在 Redis 7+ 实际排障时，应优先通过 `OBJECT ENCODING` 看真实编码，很多小对象已经表现为 `listpack` 而不是 `ziplist`。

---

## 参考资料

| 来源 | 覆盖内容 | 建议阅读顺序 |
|---|---|---|
| [Redis 官方 `OBJECT ENCODING`](https://redis.io/docs/latest/commands/object-encoding/) | 各对象在不同版本下的编码方式，含 `ziplist`、`listpack`、`quicklist`、`intset`、`skiplist` | 1 |
| [Redis 官方 Data Types](https://redis.io/docs/latest/develop/data-types/) | Redis 提供哪些对象类型，以及应用场景入口 | 2 |
| [Redis 官方 Persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/) | RDB、AOF、混合持久化的优缺点与数据丢失窗口 | 3 |
| [Redis 官方 Ziplist Glossary](https://redis.io/glossary/redis-ziplist/) | ziplist 的 entry 组成和顺序存储思路 | 4 |
| [代码酷：Redis 内部数据结构](https://www.echo.cool/docs/middleware/redis/redis-premium-theme/redis-internal-data-structure/) | SDS、字典、跳跃表、ziplist 的中文综述，适合建立整体图景 | 5 |
| [Stack Overflow: hash ziplist 配置变更后是否自动重编码](https://stackoverflow.com/questions/25715165/are-redis-hashes-kept-in-ziplist-after-changing-hash-max-ziplist-entries) | “调阈值后旧对象不会自动解码”的常见坑 | 6 |

如果要继续深入，推荐先看官方 `OBJECT ENCODING`，因为排障时最先需要确认的不是“理论上 Redis 怎么设计”，而是“你这台实例上的 key 现在到底是什么编码”。
