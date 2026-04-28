## 核心结论

Neo4j 的事务模型可以概括为：**ACID + `read committed`（读已提交）+ 自动写锁 + 集群内按 Raft 顺序提交**。这里的 `read committed`，白话解释是“读请求只能看到已经提交的数据，看不到别人还没提交的修改”。它和很多人熟悉的“全库快照式 MVCC”不是一回事。

对初学者最重要的判断有两个。

第一，**Neo4j 不是默认把整个查询固定在一个全局静态快照上执行**。图遍历期间，如果别的事务提交了新边或删除了边，你后续读到的内容可能变化。事务保证的是数据库基本正确性，不是“查询期间世界冻结”。

第二，**Neo4j 的写路径明显偏锁模型**。锁，白话解释是“先把目标对象占住，避免别人同时改同一份数据”。可以把它抽象成：

$$
read_T(x) \to version \in committed\_state\_at\_read\_time
$$

$$
write_T(x) \to acquire\ L(x) \to append\ log \to commit
$$

新手版可以直接记成一句话：你先看到图里已经提交的边，别人还没提交的修改你看不到；你写数据时会先加锁，再提交；在集群里，提交顺序还要经过 Raft 排队。

工程上更准确的说法是：`T1` 在遍历 3 跳路径时，`T2` 插入新边并提交，`T1` 是否看到这条边，取决于它在那个读点看到的已提交状态，而不是某个固定快照。

| 机制 | Neo4j 默认语义 | 作用 |
|---|---|---|
| ACID | 支持 | 保证原子性、一致性、隔离性、持久性 |
| `read committed` | 支持 | 只能读到已提交数据 |
| MVCC snapshot | 不是默认全库快照 | 不应假设长查询结果静态不变 |
| lock-based write | 支持 | 写前加锁，避免并发覆盖 |
| Raft commit order | 集群内支持 | 多副本提交顺序一致 |

---

## 问题定义与边界

本文只讨论 **Neo4j 的事务语义**，不把结论泛化到所有图数据库。图数据库，白话解释是“把点和边当作一等公民来存储和查询的数据库”。本文关心的问题是：Neo4j 如何在图场景下实现 ACID，它和典型 MVCC 关系型数据库的事务感受为什么不同。

统一记号如下：

- `T`：事务
- `S(T)`：事务 `T` 开始时可见状态
- `W(T)`：事务 `T` 的写集，也就是它打算修改的对象集合
- `L(x)`：对象 `x` 上的锁

这里要先把边界说清楚。**事务一致**不等于**查询过程中世界冻结**。如果把 Neo4j 当成默认快照隔离数据库，就会误判长查询结果。比如社交图推荐里，一个 3 跳遍历正在运行，后台同时不断发生关注和取关，结果可能变化。这在默认隔离边界内是正常现象，不是数据库出错。

| 范围 | 内容 |
|---|---|
| 讨论 | 读已提交、写锁、死锁、Raft、bookmark |
| 不讨论 | 图算法优化、索引内部实现、分布式多主写 |

一个玩具例子先固定直觉。初始图是：

`1 - 2 - 3 - 4`

`T1` 从节点 `1` 做 3 跳遍历。与此同时，`T2` 插入新边 `2 - 5` 并提交。很多初学者会默认以为 `T1` 从头到尾一定只看到最初那张图，但在 Neo4j 默认模型里，这个假设并不稳。`T1` 在后续读步骤里看到的是**读时刻已经提交的状态**，而不是一定绑定在 `S(T1)` 的全库静态副本上。

---

## 核心机制与推导

可以把 Neo4j 的事务行为拆成五步：**读、写、提交、死锁检测、集群排序**。

| 阶段 | 抽象动作 | 关键保证 |
|---|---|---|
| 读 | `read_T(x)` | 只读已提交状态 |
| 写 | `acquire L(x)` 后修改 | 并发写受锁约束 |
| 提交 | 追加日志并提交 | 成功后持久化并对外可见 |
| 死锁检测 | 检查等待图是否有环 | 有环就回滚一个事务 |
| 集群排序 | leader 统一提交顺序 | 副本间顺序一致 |

### 1. 读怎么做

在默认语义下，读不是“把数据库冻结成一张全局照片”，而是：

$$
read_T(x) \to version \in committed\_state\_at\_read\_time
$$

这里的 `version` 可以直白理解成“这个对象当前对事务可见的已提交状态”。它强调的是 **committed at read time**，不是 **snapshot at tx begin**。

这也是为什么长遍历要格外小心。图查询不是简单扫一行数据，而是一步步跟着边往外跳。每跳一步，都可能遇到别的事务已经提交的新状态。

### 2. 写怎么做

写路径更接近“先锁后写”的模型：

$$
write_T(x) \to acquire\ L(x) \to append\ log \to commit
$$

写锁的作用很直接：防止两个事务同时改同一个节点、关系或属性，最后彼此覆盖。对新手而言，可以先记住一句：**读多数时候不阻塞，但写必须先拿到对象上的排他控制权**。

### 3. 为什么会死锁

死锁，白话解释是“两个或多个事务互相等对方释放资源，谁都走不下去”。形式化地看，系统会维护等待图：

$$
G_w = (T, E)
$$

如果等待图里存在环，就判定死锁。例如：

`T1 -> T2 -> T3 -> T1`

表示 `T1` 等 `T2`，`T2` 等 `T3`，`T3` 又等 `T1`，形成闭环。

最经典的工程例子是：

- `T1` 先锁 `A`，再请求锁 `B`
- `T2` 先锁 `B`，再请求锁 `A`

这时等待图变成：

`T1 -> T2 -> T1`

系统检测到环后，会回滚代价更小的事务，让另一个继续。这也是为什么应用层必须把某些事务设计成**可重试**。

### 4. 集群里为什么要排序

单机事务只要本地提交即可，集群事务还要回答一个额外问题：**多个副本看到的提交顺序是否一致**。Neo4j 4.x 的 causal clustering 用 Raft 来保证同一数据库副本组里的提交顺序。

Raft，白话解释是“由一个 leader 负责给操作排队并复制给多数副本的共识协议”。它不等于“所有副本瞬时同时完成”，但它保证**顺序一致**。这点非常重要，因为图数据的边和点强相关，顺序错乱会直接破坏事务语义。

### 5. 多跳查询为什么容易让人误判

图数据库最有代表性的操作是遍历。例如 3 跳邻居查询：从起点出发，连续沿边走 3 次，找所有可达节点。问题在于，这种查询天然是多步读，不像主键查询那样只有一次点查。

玩具例子：

- 初始图：`1 - 2 - 3 - 4`
- `T1`：从 `1` 做 3 跳遍历
- `T2`：中途插入 `2 - 5` 并提交

如果你预期 `T1` 一定只返回原始路径，那是在按“固定快照”思考；如果你预期 `T1` 后续访问到某些新可见边，那更接近 Neo4j 默认读已提交的现实边界。

真实工程例子是社交图推荐。一次推荐可能遍历：

`User -> FOLLOWS -> User -> FOLLOWS -> User`

后台又持续有关注、取关写入。如果你要求一次长查询绝对静态，那默认事务模型未必满足；如果你接受“查询期间基于已提交状态前进，但不承诺全局静态快照”，那它就是合理的。

---

## 代码实现

应用层最关键的不是“怎么写一条 Cypher”，而是**怎么组织事务、怎么重试、怎么传播 bookmark**。bookmark，白话解释是“告诉后续请求：至少要读到某次已确认提交之后的状态”。

先看最小伪代码：

```text
begin transaction
  read graph data
  if need update:
    lock target nodes
    write changes
commit
on deadlock:
  retry transaction
```

工程版通常会变成带重试的模式：

```text
for attempt in 1..max_retry:
  tx = beginTx()
  try:
    result = tx.run(query)
    tx.commit()
    return result
  catch DeadlockDetected:
    tx.rollback()
    backoff()
raise error
```

下面用 Python 写一个可运行的玩具实现，模拟“读已提交、写锁、死锁重试、bookmark 传播”的核心思路，不依赖 Neo4j 驱动，但行为模型对应真实工程设计。

```python
from dataclasses import dataclass, field
from typing import Dict, Set, Optional
import time

class DeadlockDetected(Exception):
    pass

@dataclass
class GraphStore:
    adjacency: Dict[int, Set[int]] = field(default_factory=dict)
    lock_owner: Dict[int, str] = field(default_factory=dict)
    commit_index: int = 0

    def read_neighbors(self, node: int):
        return sorted(self.adjacency.get(node, set()))

    def acquire_lock(self, tx_id: str, node: int):
        owner = self.lock_owner.get(node)
        if owner is None:
            self.lock_owner[node] = tx_id
            return
        if owner != tx_id:
            raise DeadlockDetected(f"node {node} locked by {owner}")

    def release_locks(self, tx_id: str):
        for node, owner in list(self.lock_owner.items()):
            if owner == tx_id:
                del self.lock_owner[node]

    def add_edge(self, a: int, b: int):
        self.adjacency.setdefault(a, set()).add(b)
        self.adjacency.setdefault(b, set()).add(a)

    def commit(self):
        self.commit_index += 1
        return f"bookmark:{self.commit_index}"

def read_tx(store: GraphStore, node: int):
    # 读事务：只读取当前已提交状态
    return store.read_neighbors(node)

def write_tx_add_edge(store: GraphStore, tx_id: str, a: int, b: int):
    # 写事务：先加锁，再写，再提交
    for node in sorted([a, b]):  # 统一加锁顺序可降低死锁概率
        store.acquire_lock(tx_id, node)
    store.add_edge(a, b)
    bookmark = store.commit()
    store.release_locks(tx_id)
    return bookmark

def retrying_write(store: GraphStore, a: int, b: int, max_retry: int = 3):
    for attempt in range(1, max_retry + 1):
        tx_id = f"tx-{attempt}"
        try:
            return write_tx_add_edge(store, tx_id, a, b)
        except DeadlockDetected:
            store.release_locks(tx_id)
            time.sleep(0.01 * attempt)
    raise DeadlockDetected("retry exhausted")

def read_with_bookmark(store: GraphStore, required_bookmark: Optional[str]):
    # 真实系统里会路由到满足 bookmark 的副本
    if required_bookmark is not None:
        required_index = int(required_bookmark.split(":")[1])
        assert store.commit_index >= required_index
    return store.commit_index

store = GraphStore()
store.add_edge(1, 2)
store.add_edge(2, 3)
assert read_tx(store, 2) == [1, 3]

bm = retrying_write(store, 3, 4)
assert bm == "bookmark:1"
assert read_tx(store, 3) == [2, 4]

current_index = read_with_bookmark(store, bm)
assert current_index >= 1
```

上面代码表达了四个工程事实：

| 操作 | 事务行为 | 失败模式 | 处理方式 |
|---|---|---|---|
| 读查询 | 读已提交 | 非重复读、结果漂移 | 缩短事务，接受变化或改方案 |
| 并发写 | 写锁保护 | 死锁、锁等待 | 统一加锁顺序，失败后重试 |
| 提交 | 产出新 bookmark | 提交失败 | 回滚并上抛错误 |
| 集群读写 | 通过 bookmark 保证因果读 | 读到旧副本 | 传播 bookmark |

真实工程里，如果你用 Neo4j 官方驱动，代码结构通常是：

1. 读事务放在短生命周期函数里。
2. 写事务显式捕获 `DeadlockDetected` 或其他 transient error。
3. 成功写入后把 bookmark 带到下一次请求。
4. 不把一个超长推荐查询和一堆更新混在同一个大事务里。

---

## 工程权衡与常见坑

Neo4j 的事务设计在图业务里很实用，但它的代价也很明确：**长事务、大范围遍历、并发写热点子图** 会迅速放大冲突。

先给两个简单规则：

$$
事务持续时间越长，锁冲突概率越高
$$

$$
冲突点越集中，死锁概率越高
$$

长事务不是更安全，只是更容易占着锁不放，让别的写请求排队，甚至触发死锁。尤其在图里，一个热门枢纽节点可能被大量边更新，冲突会比普通表更新更集中。

真实工程例子：3 跳好友推荐如果执行很久，后台又在频繁改边，就可能出现结果漂移。常见解法不是“把事务开得更久”，而是：

- 把在线事务缩短
- 把遍历拆批
- 对极少数关键起点控制并发
- 把复杂分析转成异步或离线

| 坑 | 表现 | 原因 | 规避 |
|---|---|---|---|
| 误以为默认快照隔离 | 查询结果和预期不一致 | 把 `read committed` 当成 snapshot | 明确默认隔离边界 |
| 长事务遍历 | 延迟升高、结果漂移 | 查询期间可见状态变化 | 短事务、分批处理 |
| 并发写同一子图 | 锁等待、吞吐下降 | 热点节点冲突集中 | 统一加锁顺序，削峰 |
| 死锁后不重试 | 请求直接失败 | 死锁是并发下正常现象 | 捕获 transient error 自动重试 |
| 忽略 bookmark | 写后读不到最新结果 | 读落到较旧副本 | 显式传播 bookmark |

一个常见误区是“ACID 已经保证一致性，所以推荐结果必须稳定”。这其实把**数据库一致性**和**业务结果静态性**混在了一起。ACID 保证的是提交和隔离规则内的正确，不是帮你自动提供长查询的理想化快照视图。

---

## 替代方案与适用边界

Neo4j 的默认事务方案不是唯一正确方案，而是在“在线图查询 + 一般事务写入”这个区间里做了平衡。如果你要的是**强一致、可重复读、长时间静态视图**，那它未必是最合适的默认选择。

新手可以先记一句：如果你需要“从查询开始到查询结束看到的世界完全不变”，Neo4j 默认事务模型不一定满足这个目标。

工程上常见替代思路有四类。

| 方案 | 一致性 | 并发写能力 | 长查询表现 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|---|
| Neo4j 默认事务 | 中等，偏在线一致 | 中等 | 一般 | 低 | 在线图查询、关系更新 |
| 更强快照语义数据库 | 更强 | 视实现而定 | 更稳定 | 中到高 | 强事务报表、可重复读要求高 |
| 读写拆分 | 写路径强，读路径可调 | 较好 | 可优化 | 中 | 在线写入多、读一致性分层 |
| 离线图计算 | 最终一致 | 高 | 很好 | 高 | 推荐、风控、批量分析 |

真实工程里，常见组合是：

- 强一致写入仍放在线事务里
- 复杂多跳分析异步化
- 推荐结果预计算或增量刷新
- 跨副本读写链路用 bookmark 保证因果一致

这类拆分的本质是：**把必须立刻正确的部分留给事务，把不必同步完成的部分交给异步系统**。对图业务尤其重要，因为复杂遍历天然比 OLTP 点查更容易和并发写冲突。

---

## 参考资料

下表列出本文结论对应的主要来源与用途。

| 来源 | 覆盖内容 | 适合引用的位置 |
|---|---|---|
| Neo4j Operations Manual: Database internals and transactional behavior | 事务内部行为、存储与提交语义 | 核心结论、机制部分 |
| Neo4j Operations Manual: Concurrent data access | 并发访问、锁、死锁 | 死锁与工程坑 |
| Neo4j clustering architecture | 集群角色与复制架构 | 集群事务顺序 |
| Leadership, routing, and load balancing | leader、路由、副本访问 | bookmark 与读写路径 |
| Query API bookmarks | causal consistency、bookmark 传播 | 代码实现、工程实践 |
| Raft 论文 | 共识与日志顺序 | 集群排序原理 |

本文依赖的关键事实清单：

- Neo4j 提供 ACID 事务。
- 默认语义应理解为读已提交，而不是通用意义上的全库快照隔离。
- 写路径依赖锁控制并发冲突。
- 死锁可通过等待图判定，并以回滚一个事务解除。
- 集群内提交顺序通过 Raft 类机制统一。
- 跨副本读写要借助 bookmark 保证因果一致性。

1. [Neo4j Operations Manual: Database internals](https://neo4j.com/docs/operations-manual/current/database-internals/)
2. [Neo4j Operations Manual: Concurrent data access](https://neo4j.com/docs/operations-manual/current/database-internals/concurrent-data-access/)
3. [Neo4j Operations Manual: Introduction: Neo4j clustering architecture](https://neo4j.com/docs/operations-manual/current/clustering/introduction/)
4. [Neo4j Operations Manual: Leadership, routing, and load balancing](https://neo4j.com/docs/operations-manual/current/clustering/setup/routing/)
5. [Neo4j Query API: Coordinate transactions and enforce causal consistency](https://neo4j.com/docs/query-api/current/bookmarks/)
6. [Raft: In Search of an Understandable Consensus Algorithm](https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro)
