## 核心结论

Bloom Filter 是一种**概率型集合**，白话说，它不是把元素本身存下来，而是用一段位数组和多个哈希结果，近似回答“这个元素是不是出现过”。它的结论只有两类：

| 查询结果 | 含义 | 是否绝对正确 |
|---|---|---|
| 肯定不在 | 只要命中的任一位是 0，就能断定没插入过 | 是 |
| 可能在 | 命中的位全是 1，说明可能插入过，也可能只是碰撞 | 否 |

它最重要的工程价值不是“完全正确”，而是**在固定内存下提供极快的去重和存在性判断**。这类能力特别适合大规模 URL 去重、事件去重、缓存穿透保护、日志去重等场景。

假阳性率的近似公式是：

$$
p=(1-e^{-kn/m})^k
$$

其中：

- $m$ 是位数组长度
- $n$ 是已插入元素数
- $k$ 是哈希函数个数

在给定 $m,n$ 时，常用最优哈希数近似为：

$$
k\approx \frac{m}{n}\ln 2
$$

这意味着 Bloom Filter 不是“随便调几个参数就能用”的黑盒，而是一个可以**按容量和误判率反推内存成本**的数据结构。

一个新手可理解的玩具例子：假设只有 10 个格子和 3 个哈希函数。插入 URL 时，相当于有 3 支箭头打到 10 个格子中的 3 个位置，并把这些位置点亮。以后再查一个 URL，如果 3 个位置里只要有一个没亮，就能确定没见过；如果 3 个都亮了，只能说“可能见过”。

---

## 问题定义与边界

Bloom Filter 解决的问题不是“精确存储集合”，而是：

**在内存有限时，高速判断一个元素是否很可能已经出现过。**

这里的“内存有限”很关键。比如你要给 1 亿个 URL 做去重，如果直接存字符串，内存会远大于字符串长度本身，因为还要加上哈希表桶、对象头、指针、负载因子等额外开销。Bloom Filter 通过只存 bit，把问题从“保存值”变成“保存痕迹”。

它的边界也非常明确：

| 特性 | 标准 Bloom Filter |
|---|---|
| 查询未出现元素 | 一定能判对 |
| 查询已出现元素 | 可能因碰撞误判为“可能在” |
| 假阳性 | 允许 |
| 假阴性 | 不允许 |
| 删除元素 | 不支持 |
| 容量增长 | 固定大小下会导致误判率上升 |

这里的**假阳性**，白话说是“系统说可能出现过，其实没出现过”；**假阴性**，白话说是“系统说没出现过，其实出现过”。标准 Bloom Filter 的核心承诺是：**不产生假阴性**。

查一个 URL 的过程可以画成下面这样：

```text
URL --> h1 --> [2]
    --> h2 --> [5]   bit array: [0 1 1 0 0 1 0 0 0 1]
    --> h3 --> [9]

若 [2]/[5]/[9] 中有任一位为 0 => 肯定未见过
若三位全为 1 => 可能见过
```

这也决定了它的典型使用方式：Bloom Filter 常作为**前置拦截层**。如果业务能容忍误杀少量请求，可以直接根据 Bloom 结果跳过；如果业务不能容忍误判，就要在“可能在”之后再查真实集合或数据库确认。

---

## 核心机制与推导

Bloom Filter 的机制可以分成两步。

第一步是插入。对元素 $x$ 计算 $k$ 个哈希：

$$
h_1(x), h_2(x), ..., h_k(x)
$$

把它们映射到 $[0,m-1]$，并把对应 bit 置为 1。

第二步是查询。再次计算同样的 $k$ 个位置：

- 只要有一个位置是 0，说明这个元素从未插入
- 如果全部是 1，说明它可能插入过

### 假阳性率如何推导

先看某一位在一次哈希下**不被命中**的概率：

$$
1-\frac{1}{m}
$$

一个元素有 $k$ 次哈希，所以某一位在插入一个元素后仍为 0 的概率是：

$$
\left(1-\frac{1}{m}\right)^k
$$

插入 $n$ 个元素后，这一位仍为 0 的概率是：

$$
\left(1-\frac{1}{m}\right)^{kn}
$$

当 $m$ 较大时，用指数近似：

$$
\left(1-\frac{1}{m}\right)^{kn}\approx e^{-kn/m}
$$

所以某一位为 1 的概率约为：

$$
1-e^{-kn/m}
$$

查询一个**实际不在集合中的元素**时，它命中的 $k$ 个位置如果恰好全是 1，就会出现假阳性，因此：

$$
p=(1-e^{-kn/m})^k
$$

这条公式非常重要，因为它直接告诉你三个事实：

1. $n$ 增大时，$p$ 上升
2. $m$ 增大时，$p$ 下降
3. $k$ 不是越大越好，过大反而让更多 bit 被点亮

对 $p$ 关于 $k$ 求最优，可得到常用近似：

$$
k\approx \frac{m}{n}\ln 2
$$

进一步代入可得最优位数设计公式：

$$
m\approx -\frac{n\ln p}{(\ln 2)^2}
$$

### 数值玩具例子

设 $m=10,n=2,k=3$，则：

$$
p=(1-e^{-3\times2/10})^3=(1-e^{-0.6})^3
$$

因为 $e^{-0.6}\approx 0.549$，所以：

$$
p\approx (1-0.549)^3 = 0.451^3 \approx 0.092
$$

也就是约 9.2% 的假阳性率。这个数不低，正好说明一个现实：**位数组太小，哪怕元素不多，误判也会明显上升。**

### Scalable Bloom Filter 为什么能扩容

标准 Bloom Filter 的问题是位数组大小预先固定。若实际 $n$ 超出预期，填充率会持续升高，误判率快速恶化。

**Scalable Bloom Filter** 的思路是：不是把一个滤器硬撑到报废，而是维护一个**子滤器列表**。当前子滤器接近饱和时，新建一个更大的子滤器，并给它分配更严格的目标假阳率。查询时从所有子滤器中检查，任一命中都返回“可能在”。

```text
写入流程:
新元素 -> 当前子滤器
        -> 填充率超阈值?
           是 -> 新建更大子滤器 -> 后续写入切到新滤器
           否 -> 继续写入
```

这样做的核心收益是：**总误判率仍可被上界控制，同时避免一次性按最坏容量分配巨量内存。**

---

## 代码实现

下面先给出一个可运行的 Python 版本。为了避免依赖第三方库，哈希函数使用 `hashlib.sha256` 做不同种子的派生。位数组用 `bytearray` 模拟，`1 bit` 对应一个位置。

```python
import math
import hashlib


class BloomFilter:
    def __init__(self, capacity: int, error_rate: float):
        assert capacity > 0
        assert 0 < error_rate < 1

        self.capacity = capacity
        self.error_rate = error_rate
        self.m = math.ceil(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.k = max(1, round((self.m / capacity) * math.log(2)))
        self.bits = bytearray(self.m)
        self.count = 0

    def _indexes(self, item: str):
        raw = item.encode("utf-8")
        for i in range(self.k):
            digest = hashlib.sha256(i.to_bytes(2, "big") + raw).digest()
            yield int.from_bytes(digest[:8], "big") % self.m

    def insert(self, item: str):
        for idx in self._indexes(item):
            self.bits[idx] = 1
        self.count += 1

    def query(self, item: str) -> bool:
        return all(self.bits[idx] == 1 for idx in self._indexes(item))

    def fill_ratio(self) -> float:
        return sum(self.bits) / self.m


class CountingBloomFilter(BloomFilter):
    def __init__(self, capacity: int, error_rate: float):
        super().__init__(capacity, error_rate)
        self.bits = [0] * self.m  # 改成计数数组

    def insert(self, item: str):
        for idx in self._indexes(item):
            self.bits[idx] += 1
        self.count += 1

    def remove(self, item: str):
        assert self.query(item), "item definitely not present"
        for idx in self._indexes(item):
            assert self.bits[idx] > 0
            self.bits[idx] -= 1
        self.count -= 1

    def query(self, item: str) -> bool:
        return all(self.bits[idx] > 0 for idx in self._indexes(item))


class ScalableBloomFilter:
    def __init__(self, init_capacity=100, error_rate=0.01, growth=2.0, fill_threshold=0.5):
        self.error_rate = error_rate
        self.growth = growth
        self.fill_threshold = fill_threshold
        self.filters = [BloomFilter(init_capacity, error_rate / 2)]

    def scale_if_needed(self):
        current = self.filters[-1]
        if current.fill_ratio() >= self.fill_threshold:
            new_capacity = math.ceil(current.capacity * self.growth)
            # 新子滤器给更小误判预算，控制总误判率
            new_error = max(self.error_rate / (2 ** (len(self.filters) + 1)), 1e-9)
            self.filters.append(BloomFilter(new_capacity, new_error))

    def insert(self, item: str):
        self.scale_if_needed()
        self.filters[-1].insert(item)

    def query(self, item: str) -> bool:
        return any(f.query(item) for f in self.filters)


bf = BloomFilter(capacity=1000, error_rate=0.01)
assert bf.query("https://a.com") is False
bf.insert("https://a.com")
assert bf.query("https://a.com") is True

cbf = CountingBloomFilter(capacity=1000, error_rate=0.01)
cbf.insert("u1")
assert cbf.query("u1") is True
cbf.remove("u1")
assert cbf.query("u1") is False

sbf = ScalableBloomFilter(init_capacity=2, error_rate=0.01, fill_threshold=0.1)
for i in range(20):
    sbf.insert(f"url-{i}")
assert sbf.query("url-3") is True
assert len(sbf.filters) > 1
```

核心 API 可以概括成下面的伪代码：

```text
insert(x):
  for idx in hash_k(x):
    bit_array[idx] = 1

query(x):
  for idx in hash_k(x):
    if bit_array[idx] == 0:
      return False
  return True

scale_if_needed():
  if current_filter.fill_ratio > threshold:
    create_new_larger_filter()
```

不同实现的结构差异如下：

| 实现 | 底层结构 | 是否支持删除 | 扩容方式 |
|---|---|---|---|
| 标准 Bloom | 位数组 | 否 | 重建 |
| Scalable Bloom | 多个子滤器列表 | 否 | 追加新子滤器 |
| Counting Bloom | 计数数组 | 是 | 可重建或分层 |

### 真实工程例子：大规模 URL 去重

爬虫系统里，一个 URL 常要经过规范化、查重、入队、抓取四步。典型做法是：

1. 先把 URL 规范化，消掉大小写、尾部 `/`、无关 query 参数顺序差异
2. 用 Bloom Filter 查询
3. 若“肯定不在”，直接入队并写入 Bloom
4. 若“可能在”，按业务选择跳过，或再查精确集合确认

当 URL 量级到 1 亿甚至 10 亿时，Bloom Filter 的优势非常直接：它不保存 URL 本身，只保存哈希命中的 bit，因此常能把去重内存从“几十 GB 级哈希集合”压到“几 GB 内”。

---

## 工程权衡与常见坑

Bloom Filter 在工程里最常见的问题，不是“不会写”，而是**写完以后没监控**。

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| 填充率过高 | 假阳性率快速上升 | 监控填充率，超阈值重建或切到 Scalable |
| 误把 Bloom 当精确去重 | 新数据被误杀 | 在关键链路后接真实集合二次确认 |
| 需要删除却用标准 Bloom | 无法安全删除 | 换 Counting Bloom 或 Cuckoo Filter |
| 热点集中在单一业务域 | 某一类数据先把滤器打满 | 按域名、租户、时间窗口拆分滤器 |
| URL 未规范化 | 同一页面多种写法重复抓取 | 统一 canonical 规则再做哈希 |

真实工程里可以用一个简单决策流程：

```text
监控填充率/误判率
    -> 正常: 继续使用
    -> 升高: 新建子滤器或整体重建
    -> 仍不满足 SLA: 改用精确集合或 Cuckoo/Counting
```

对爬虫去重来说，一个很实际的问题是：**假阳性到底能不能接受？**  
答案取决于代价函数。

- 如果误杀一个 URL，只会少抓一篇长尾页面，业务能接受，那么 Bloom 很合适
- 如果误杀一个 URL，会漏掉订单、支付、审计数据，那就不能只靠 Bloom

因此 Bloom Filter 常用于**高吞吐、可容忍少量漏处理候选项**的系统，而不是强一致主账本系统。

---

## 替代方案与适用边界

Bloom Filter 不是唯一选择。最常见的两个替代是 Counting Bloom Filter 和 Cuckoo Filter。

**Counting Bloom Filter**：把 bit 改成小计数器。白话说，不是只记“这个位置亮没亮”，而是记“这个位置被打了几次”，因此可以删除。代价是空间明显增加，且计数溢出、并发更新都更复杂。

**Cuckoo Filter**：基于 Cuckoo Hashing 的近似集合结构。白话说，它存的是短指纹，并允许元素在几个候选桶之间搬家，因此支持删除，低误判率时通常比标准 Bloom 更省空间。但插入可能失败，满载时要搬迁甚至重建。

横向对比如下：

| 方案 | 空间占用 | 支持删除 | 插入失败 | 假阳率特性 |
|---|---|---|---|---|
| Bloom Filter | 低 | 否 | 否 | 可配置，容量超限后恶化明显 |
| Counting Bloom | 高于 Bloom | 是 | 否 | 与 Bloom 类似，但空间代价更高 |
| Cuckoo Filter | 低到中，低误判时常优于 Bloom | 是 | 可能 | 低误判场景表现好 |

适用边界可以直接记成三句话：

- **持续追加、几乎不删除、极度看重内存**：优先 Bloom 或 Scalable Bloom
- **需要删除，但仍接受概率误判**：优先 Counting Bloom
- **需要删除，且希望低误判率下更高空间效率**：优先 Cuckoo Filter

一个具体情景对比：

| 场景 | 更合适的方案 | 原因 |
|---|---|---|
| 爬虫 URL 去重，持续新增 | Bloom + Scalable | 内存低，查询快，扩容自然 |
| 黑名单项频繁过期删除 | Counting Bloom 或 Cuckoo | 标准 Bloom 不能删 |
| 极低误判率且要求可删 | Cuckoo Filter | 低误判时通常更划算 |
| 金融主数据去重 | 精确集合/数据库索引 | 不能接受假阳性 |

---

## 参考资料

- 论文：Burton H. Bloom, *Space/Time Trade-offs in Hash Coding with Allowable Errors*, 1970。Bloom Filter 原始论文，定义了允许错误的集合查询思路。  
  链接：https://dl.acm.org/doi/10.1145/362686.362692

- 概念综述：Wikipedia, *Bloom filter*。用于核对标准定义、假阳性公式、最优 $k$ 与位数公式。  
  链接：https://en.wikipedia.org/wiki/Bloom_filter

- 论文：Paulo Sérgio Almeida, Carlos Baquero, Nuno Preguiça, David Hutchison, *Scalable Bloom Filters*, Information Processing Letters, 2007。用于说明通过多个子滤器控制总体假阳率的扩容方法。  
  链接：https://doi.org/10.1016/j.ipl.2006.10.007

- 概念综述：Wikipedia, *Counting Bloom filter*。用于核对计数型 Bloom 的“计数数组支持删除”机制。  
  链接：https://en.wikipedia.org/wiki/Counting_Bloom_filter

- 论文/文章：Bin Fan, David G. Andersen, Michael Kaminsky, *Cuckoo Filter: Better Than Bloom*，USENIX ;login:, 2013。用于比较删除支持、低假阳率下的空间效率与插入行为。  
  链接：https://www.usenix.org/publications/login/august-2013-volume-38-number-4/cuckoo-filter-better-bloom

- 实践文章：OneUptime, *How to Use Redis Bloom Filters for URL Deduplication in Crawlers*, 2026-03-31。用于 URL 去重的工程示例、RedisBloom 命令和内存收益直觉。  
  链接：https://oneuptime.com/blog/post/2026-03-31-redis-bloom-filter-url-deduplication/view
