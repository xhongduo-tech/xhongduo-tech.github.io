## 核心结论

B+ 树和 LSM 树解决的是同一个问题：在磁盘上高效保存大量有序键值数据，但两者优化方向不同。

B+ 树的核心优势是“稳定读性能”。它把索引组织成多层页结构，页可以理解为一次磁盘 I/O 读取的基本块。查找时从根节点走到叶子节点，路径长度通常很短；叶子节点按键顺序链接，因此范围查询特别高效。数据库里常见的聚簇索引，意思是“数据行本身按索引顺序存放”，通常基于 B+ 树实现；非聚簇索引，意思是“索引里存的是指向数据行的位置”，也常基于 B+ 树。

LSM 树的核心优势是“把随机写变成顺序写”。它先把写入落到 WAL 和 MemTable。WAL 是预写日志，意思是“先记一份可恢复日志，防止宕机丢数据”；MemTable 是内存中的有序表，意思是“先在内存里排好序再批量刷盘”。当 MemTable 满了，再一次性刷成不可变的 SSTable 文件。SSTable 是排序字符串表，意思是“磁盘上只追加、不原地修改的有序文件”。后台再通过 Compaction 合并这些文件。Compaction 是压缩整理过程，意思是“把多个旧文件重写成更少的新文件，顺便清理旧版本和删除标记”。

可以先记住一个粗略判断：

| 维度 | B+ 树 | LSM 树 |
|---|---|---|
| 随机点查 | 稳定、通常较强 | 依赖缓存/Bloom Filter/层数 |
| 范围查询 | 很强 | 可做，但跨多个 SST 扫描更复杂 |
| 写入吞吐 | 中等，随机写较多 | 高，顺序写友好 |
| 更新/删除 | 原地修改，逻辑直观 | 追加新版本和 tombstone，后续再清理 |
| 典型场景 | OLTP、二级索引、范围读多 | 日志、时序、KV、高写入系统 |

如果把数据规模记为 $N$，B+ 树扇出记为 $f$，树高近似为：

$$
h \approx \log_f N
$$

因此单次读取的页访问成本常近似为树高 $h$。

LSM 树的读放大可以粗略理解为：

$$
\text{Read Amplification} \approx 1 + \text{MemTable 数} + \text{需要探测的 SST 数}
$$

写放大则来自多轮 compaction，把同一条逻辑数据反复重写到更低层。

---

## 问题定义与边界

本文讨论的是“磁盘型数据库或存储引擎如何组织索引与数据文件”，不是单纯讨论内存数据结构。零基础读者容易混淆的一点是：在内存里，一棵普通平衡树和跳表都很好用；但在磁盘上，核心约束变成了 I/O 次数、顺序写能力、页大小、后台整理成本。

判断 B+ 树还是 LSM 树，至少要先看四个边界：

| 边界维度 | 问题 | 对选择的影响 |
|---|---|---|
| 读写比例 | 读多还是写多 | 读多偏 B+ 树，写多偏 LSM |
| 访问模式 | 点查多还是范围扫多 | 范围扫多偏 B+ 树 |
| 延迟目标 | 要稳定低延迟还是接受后台抖动 | 稳定低延迟偏 B+ 树 |
| 存储介质 | 随机写贵不贵 | 随机写昂贵时 LSM 更占优 |

玩具例子：有一张 1000 行的学生表，主键是 `id`。  
如果你经常查 `id between 200 and 260`，B+ 树聚簇索引会很自然，因为叶子节点本来就是按 `id` 排列的，扫起来几乎是连续读取。  
如果你是一个传感器系统，每秒写入很多温度记录，先落 WAL、写 MemTable，等积累到一定量再 flush 成 SST，这就是更典型的 LSM 思路。

真实工程例子：一个日志采集系统每秒写入几十万条事件，按 `service_id + timestamp` 作为键。绝大多数请求是“追加写入”和“最近 5 分钟某服务的日志筛选”。这类系统常更偏向 LSM，因为写入密度太高，若每条写都直接更新磁盘页，随机写放大和页分裂成本都很难压低。

还要明确一个常见误区：  
“LSM 一定比 B+ 树快”是错的。  
更准确的说法是：LSM 通常在高写入吞吐下更占优，但它把复杂度转移到了读取、后台 compaction、空间管理和尾延迟控制上。

---

## 核心机制与推导

先看 B+ 树。

B+ 树可以理解成“磁盘版多级目录”。根节点记录若干键范围，中间节点继续缩小范围，叶子节点保存最终键以及数据行或行指针。因为一个磁盘页能容纳很多键，所以扇出 $f$ 往往很大，几十到几百都常见。这意味着即使有几千万条记录，树高也可能只有 3 到 4 层。

例如，假设一张表有 1000 条记录，单页可以放 100 个键，那么扇出近似 $f=100$：

$$
h \approx \log_{100}(1000) \approx 1.5
$$

实际实现中需要向上取整，所以大约 2 到 3 层即可完成定位。  
这就是 B+ 树点查稳定的原因：路径短，页访问数量可控。

如果是聚簇索引，叶子节点直接保存整行或主数据页地址，数据天然有序，因此查 `id in [200, 260]` 时，从起始叶子定位后顺着叶链往后扫即可。  
如果是非聚簇索引，叶子节点保存的是主键或行地址，先查索引，再“回表”取整行数据。回表的意思是“从二级索引命中后，再去主索引或数据页取完整记录”。

B+ 树的写入成本主要来自三部分：

1. WAL 记录。
2. 定位目标页。
3. 页更新、页分裂、父节点更新。

热点写入时，如果某个范围不断插入，页会频繁分裂，造成碎片和缓存竞争。这就是为什么 B+ 树写多场景容易出现抖动。

再看 LSM 树。

LSM 树把写入流程拆成前台轻量写和后台重整理两部分：

$$
\text{Write Path} = \text{WAL append} + \text{MemTable insert} + \text{Flush} + \text{Compaction}
$$

前台写入时不急着改磁盘上的旧记录，而是：

1. 先顺序追加到 WAL。
2. 再写入内存中的有序结构 MemTable。
3. MemTable 满后刷成一个新的 SSTable。
4. 后台不断把多个 SSTable 合并。

可以用一个流程表示：

`请求写入 -> WAL -> MemTable -> Flush 成 L0 SST -> Compaction 到 L1/L2/...`

这带来两个直接后果。

第一，写入更适合磁盘。  
因为磁盘尤其是机械盘和某些 SSD 场景下，顺序写通常比频繁随机改页更友好。

第二，读取会变复杂。  
一次点查可能要查：

1. 当前 MemTable
2. 不可变 MemTable
3. Level 0 的多个 SST
4. 更低层级的 SST

所以 LSM 树常借助 Bloom Filter。Bloom Filter 是布隆过滤器，意思是“用很小空间快速判断某个键大概率不存在”。它不能保证“存在”，但能高效过滤“大量肯定不存在的 SST”，从而减少磁盘探测。

一个常见的粗略推导是：  
如果 LSM 有 4 个主要层级，每级压缩倍数约为 10，那么一条记录可能在 compaction 中被多次重写。工程上常把写放大近似看成与层级数和合并策略相关。若采用偏保守的 leveled compaction，写放大常明显高于 size-tiered compaction。可以做一个直观理解：

$$
\text{WA}_{LSM} \propto L \times T
$$

其中 $L$ 是层级数，$T$ 是每层目标大小比或压缩倍数。  
例如 $L=4$、$T=10$，可把它粗略看成“每单位逻辑写入会引发数十倍的物理重写量级”，常见直观表达就是接近 $W \times 40$ 的量级估计。这个值不是精确公式，但能帮助理解：LSM 不是免费写入，它只是把“随机写成本”换成了“后台顺序重写成本”。

空间放大也来自同一机制。  
同一个 key 的旧版本、删除标记 tombstone，以及尚未完成 compaction 的重复数据都会短期共存。

---

## 代码实现

下面先用一个玩具代码模拟“B+ 树范围查询”和“LSM 写入后 flush 成 SST”的最小逻辑。代码不是完整数据库实现，但足够说明路径差异。

```python
from bisect import bisect_left, bisect_right, insort

# 玩具 B+ 树叶层：直接用有序数组模拟叶子节点
class ToyClusteredBPlusTree:
    def __init__(self):
        self.rows = []  # (id, value) 按 id 有序

    def insert(self, key, value):
        insort(self.rows, (key, value))

    def range_query(self, left, right):
        keys = [k for k, _ in self.rows]
        i = bisect_left(keys, left)
        j = bisect_right(keys, right)
        return self.rows[i:j]

tree = ToyClusteredBPlusTree()
for i in range(1, 1001):
    tree.insert(i, f"row-{i}")

result = tree.range_query(200, 260)
assert len(result) == 61
assert result[0] == (200, "row-200")
assert result[-1] == (260, "row-260")


# 玩具 LSM：MemTable 满了就 flush 成一个 SST
class ToyLSM:
    def __init__(self, mem_limit=128):
        self.mem_limit = mem_limit
        self.wal = []
        self.memtable = {}
        self.sstables = []  # 每个 SST 用排好序的 dict 列表表示

    def put(self, key, value):
        self.wal.append((key, value))     # 先写 WAL
        self.memtable[key] = value        # 再写 MemTable
        if len(self.memtable) >= self.mem_limit:
            self.flush()

    def flush(self):
        if not self.memtable:
            return
        sst = dict(sorted(self.memtable.items()))
        self.sstables.insert(0, sst)      # 新 SST 放最前
        self.memtable.clear()

    def get(self, key):
        if key in self.memtable:
            return self.memtable[key]
        for sst in self.sstables:
            if key in sst:
                return sst[key]
        return None

lsm = ToyLSM(mem_limit=128)
for i in range(1, 1001):
    lsm.put(i, f"event-{i}")
lsm.flush()

assert lsm.get(1) == "event-1"
assert lsm.get(1000) == "event-1000"
assert len(lsm.sstables) >= 1
```

这个玩具例子对应的含义是：

| 路径 | B+ 树 | LSM |
|---|---|---|
| 写入 | 定位页并修改，必要时分裂 | WAL 追加 + MemTable，满了再 flush |
| 点查 | 根到叶一次路径 | 先查内存，再查多个 SST |
| 范围查 | 叶链顺扫 | 需跨多个有序文件归并扫描 |

再给出更接近实现思路的伪代码。

B+ 树查找与叶分裂：

```text
function bplus_find(root, key):
    node = root
    while node is not leaf:
        node = choose_child(node, key)
    return binary_search(node.keys, key)

function bplus_insert(leaf, key, value):
    insert_into_leaf_in_order(leaf, key, value)
    if leaf.is_full():
        new_leaf = split_leaf(leaf)
        promote_separator_to_parent(leaf, new_leaf)
```

LSM 写入与 flush：

```text
function lsm_put(key, value):
    wal.append(key, value)
    memtable.insert(key, value)
    if memtable.size >= threshold:
        freeze(memtable)
        flush_to_sstable(memtable)
        schedule_compaction()
```

真实工程例子：  
以 TiKV/RocksDB 类引擎为例，前台写入通常先进入 WAL 和 MemTable，后台线程负责 flush 与 compaction。这样即使上层是分布式事务系统，底层局部写路径仍然是“先顺序写，再后台重整”。

---

## 工程权衡与常见坑

真正决定系统表现的，往往不是“哪种树更先进”，而是放大效应是否被控制住。

先看常见问题和对策：

| 问题 | 更常见于 | 后果 | 常见缓解手段 |
|---|---|---|---|
| 页分裂频繁 | B+ 树 | 写延迟抖动、碎片增加 | 调整填充率、批量写、页分裂策略 |
| 回表过多 | B+ 树 | 随机读增多 | 覆盖索引、减少宽表列回表 |
| compaction 跟不上 | LSM | 写阻塞、磁盘占用飙升 | 增加后台线程、调 compaction 策略 |
| 读放大高 | LSM | 点查尾延迟上升 | Bloom Filter、Block Cache、leveled compaction |
| tombstone 堆积 | LSM | 范围读变慢、空间膨胀 | 及时 compaction、TTL 策略 |
| 小文件过多 | LSM | metadata 压力、查找成本上升 | 控制 flush 频率、限制 L0 文件数 |

日志系统是一个典型真实工程例子。  
假设某业务在促销时日志量突然上涨 10 倍，LSM 引擎前台写入最开始还能扛住，因为写 MemTable 很快；但如果后台 compaction 吞吐跟不上，L0 文件会快速堆积。接下来会发生两件事：

1. 读请求需要检查更多 SST。
2. 系统为了避免失控，可能对写入做 stall，也就是主动阻塞。

这时 Bloom Filter 的价值就会体现出来。  
如果某个 key 根本不在大多数 SST 中，Bloom Filter 可以快速排除这些文件，避免每个文件都打开数据块查一遍。但 Bloom Filter 只能缓解“无效探测”，不能消除 compaction 落后的根因。

B+ 树的问题则更偏“局部热点”。  
例如订单表按自增主键写入时，右侧叶子页会成为热点页。所有新写都集中打到同一页附近，容易导致 latch 竞争、页分裂和缓存局部抖动。这里常见的工程手段包括：

1. 合理设置页填充率，给未来插入预留空间。
2. 将批量插入合并，减少每条记录独立分裂。
3. 对二级索引做覆盖设计，降低回表压力。

要注意，B+ 树并不是“写一定差”。  
如果工作负载是中等写入、强事务、范围查询频繁、索引设计稳定，那么 InnoDB 这类 B+ 树引擎依然非常强，因为它把读写路径、锁语义、事务恢复和扫描能力整合得很成熟。

---

## 替代方案与适用边界

如果只讨论主流数据库引擎，B+ 树和 LSM 是最常见的两条路线，但并不是全部。

适用边界可以直接记成下面这张表：

| 场景 | 更优选择 | 触发条件 | 说明 |
|---|---|---|---|
| 交易型业务库 | B+ 树 | 点查和范围查都多，更新较频繁 | 如 MySQL InnoDB |
| 日志/指标/时序写入 | LSM | 持续高写入，允许后台整理 | 如 RocksDB、TiKV、Cassandra |
| 纯缓存 | 都不是首选 | 数据主要在内存 | 更像 Redis 这类内存结构 |
| 分析型列存 | 都不是核心结构 | 大批量扫描和聚合 | 往往转向列式存储 |

MySQL InnoDB 是 B+ 树路线的代表。  
它的聚簇索引让主键范围扫描非常自然，二级索引再回到主键取整行，适合大量事务型业务。

TiKV 更偏 LSM 路线。  
它底层依托 RocksDB 类存储思路，适合高并发写入和分布式扩展。在日志、指标、事件流这类“写多读少或写多范围读”的场景里，LSM 通常比 B+ 树更容易把吞吐做高。

是否存在替代结构？有，但适用边界更窄。  
例如：

1. 哈希索引适合等值查询，不擅长范围查询。
2. 跳表常用于内存层结构，比如 MemTable 实现。
3. 列式存储适合分析，不适合高频点更新。

所以实际工程决策通常不是“理论上谁更优”，而是“你的瓶颈究竟在随机读、随机写、范围扫，还是后台整理能力”。

---

## 参考资料

1. TiKV, *B-Tree vs LSM-Tree*  
   链接：https://tikv.org/deep-dive/key-value-engine/b-tree-vs-lsm/  
   贡献视角：从存储引擎实现角度系统比较两者的读放大、写放大、空间放大，并解释了 LSM 的 flush 与 compaction 路径。文中的核心结论可以概括为：LSM 用更复杂的后台整理换取更高写吞吐。

2. System Overflow, *When Should You Choose LSM Trees Over B+ Trees*  
   链接：https://www.systemoverflow.com/learn/database-design/indexing/when-should-you-choose-lsm-trees-over-b-trees  
   贡献视角：强调实际选型边界，特别是日志、时序、KV 等写密集负载下，LSM 的顺序写优势与 compaction 代价。

3. SolarWinds, *Clustered vs Non-Clustered Index*  
   链接：https://www.solarwinds.com/database-optimization/clustered-vs-non-clustered-index/  
   贡献视角：清晰说明聚簇索引和非聚簇索引的物理差别，有助于理解为什么 B+ 树在范围查询和回表成本上表现不同。

两条关键结论可以直接引用为工程判断依据：

1. B+ 树在聚簇组织下天然适合范围扫描，因为叶子节点有序且可顺序遍历，这一点在事务型数据库中极其重要。来源：SolarWinds 对聚簇/非聚簇索引的解释。
2. LSM 树通过 MemTable、SSTable 和 compaction 将随机写转换为顺序写，更适合高写入吞吐系统，但代价是读放大、写放大和空间放大管理更复杂。来源：TiKV 与 System Overflow 的对比分析。
