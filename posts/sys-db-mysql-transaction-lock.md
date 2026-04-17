## 核心结论

关键点 1：MySQL 事务并不等于“所有读都加锁”，InnoDB 主要靠 MVCC 处理普通查询的一致性。MVCC 可以理解为“同一行同时保留多个历史版本，读操作按快照挑一个可见版本”。

关键点 2：真正决定并发冲突强度的，是“当前读是否要锁住哪些索引区间”。当前读指会读取最新版本并可能加锁的读，比如 `SELECT ... FOR UPDATE`。

一个新手最容易混淆的事实是：在 `REPEATABLE READ` 下，普通 `SELECT` 往往是快照读，不加锁；但 `UPDATE`、`DELETE`、`SELECT ... FOR UPDATE` 这类当前读，会依赖记录锁、间隙锁和 next-key lock 来防止并发写入破坏结果。

玩具例子：

- 事务 A 开始，执行 `SELECT balance FROM account WHERE id = 1;`，读到余额 `100`
- 事务 B 把这行改成 `120` 并提交
- 事务 A 再次执行同样的普通 `SELECT`
- 在 `REPEATABLE READ` 下，事务 A 仍然可能看到 `100`，因为它读的是自己的快照，而不是最新值

这说明：可重复读先靠“旧版本可见性”，不是先靠“把别人都锁住”。

必要元素：隔离级别与典型行为对比

| 隔离级别 | 普通 SELECT | 当前读范围保护 | 幻读风险 | 并发性 |
|---|---|---|---|---|
| READ COMMITTED | 每次生成新快照 | 通常弱于 RR | 更高 | 更高 |
| REPEATABLE READ | 同一事务内快照稳定 | 默认更依赖 next-key | 更低 | 中等 |
| SERIALIZABLE | 读也更容易参与锁冲突 | 最强 | 最低 | 最低 |

简化地说，可以把一致性目标写成：

$$\text{事务隔离效果} \approx \text{MVCC快照能力} + \text{锁覆盖范围}$$

---

## 问题定义与边界

关键点 1：事务要解决的不只是“读到脏数据”，还要解决“同一个条件查出来的行集合突然变了”。

关键点 2：幻读不是“同一行的值变了”，而是“满足条件的新行冒出来了”。这类问题通常发生在范围查询上。

先区分两个概念：

- 快照读：读某个时间点可见的旧版本，通常不加锁
- 当前读：读最新版本，并在需要时给记录或区间加锁

为什么要防幻读？看一个范围查询：

```sql
SELECT * FROM orders WHERE amount BETWEEN 100 AND 200 FOR UPDATE;
```

如果这时另一个事务插入一行 `amount = 150`，那么当前事务后续再次按同样条件读取时，结果集会多一行。这不是旧行被修改，而是“新行进入了查询范围”，这就是幻读。

玩具例子：

表里当前有索引值 `100, 180, 250`。

- 事务 A：`SELECT * FROM orders WHERE amount BETWEEN 100 AND 200 FOR UPDATE;`
- 事务 B：`INSERT INTO orders(amount) VALUES(150);`

如果不锁住 `100` 到 `200` 之间的索引间隙，事务 B 就能插入 `150`，事务 A 的范围结果就不稳定。

必要元素：可见性与锁类型对比

| 场景 | 读到什么 | 是否加锁 | 主要目标 |
|---|---|---|---|
| 普通 SELECT | 快照版本 | 通常否 | 一致读 |
| SELECT ... FOR UPDATE | 最新版本 | 是 | 防并发写 |
| UPDATE / DELETE | 最新版本 | 是 | 修改前先占住目标 |
| 范围当前读 | 最新版本 + 区间保护 | 是 | 防幻读 |

边界也要说清楚：锁不是按 SQL 文本锁，而是按“索引访问路径”锁。没有合适索引时，MySQL 可能扫描更大范围，锁也会跟着扩大。这是很多线上阻塞的根因。

---

## 核心机制与推导

关键点 1：MVCC 的底层基础是 undo log 和 read view。undo log 可以理解为“行被修改前的旧版本链”。

关键点 2：next-key lock 本质上是“记录锁 + 间隙锁”，既锁住现有行，也锁住行与行之间可插入的位置。

### 1. MVCC 如何让普通 SELECT 不加锁

当一行被更新时，InnoDB 不会只保留新值，还会通过 undo log 保存旧版本。事务创建读视图后，会按可见性规则决定读新版本还是旧版本。

可以把它抽象成：

$$\text{快照读结果} = \text{当前行版本链中对读视图可见的第一个版本}$$

所以，普通 `SELECT` 不必阻塞写事务，也不必被写事务阻塞。这是 InnoDB 在读多写少场景还能保持吞吐的关键。

### 2. next-key 为什么能防幻读

定义上：

$$\text{next-key lock} = \text{record lock} + \text{gap lock}$$

- 记录锁：锁住某条已有索引记录
- 间隙锁：锁住两条索引记录之间的空档，不让别人插入
- next-key lock：把这两者合起来

最小数值例子：索引值集合 `{10, 11, 13, 20}`。

如果事务执行一个会锁范围的当前读，并扫描到 `11` 和 `13`，那么它可能锁住这些区间：

- `(10, 11]`
- `(11, 13]`

这里 `(11, 13]` 的含义是：`13` 这条记录本身被锁住，同时 `11` 和 `13` 之间的插入位置也被锁住。因此别人不能插入 `12`。

还会有一个重要术语：supremum 伪记录。它可以理解为“索引末尾的虚拟上界”。如果扫描碰到最后一段，锁可能延伸到末尾上界，用来防止在最大值之后插入满足条件的新记录。

必要元素：简化锁逻辑伪代码

```text
for each index record matched by current read:
    lock current record
    lock gap before current record

if range reaches index end:
    lock gap before supremum
```

这段伪代码不是 MySQL 源码，而是帮助理解 next-key 的组成。

---

## 代码实现

关键点 1：要显式拿锁，最常见的是 `SELECT ... FOR UPDATE`。它会触发当前读。

关键点 2：是否锁得很大，关键看“条件是否命中索引、扫描了多少索引项、隔离级别是什么”。

先看一个真实工程里常见的 SQL 场景。假设有表：

```sql
CREATE TABLE coupon (
  id BIGINT PRIMARY KEY,
  user_id BIGINT NOT NULL,
  status TINYINT NOT NULL,
  KEY idx_user_status (user_id, status)
);
```

事务 A：

```sql
START TRANSACTION;
SELECT * FROM coupon
WHERE user_id = 42 AND status = 0
FOR UPDATE;
-- 这里做领券校验
```

事务 B：

```sql
START TRANSACTION;
INSERT INTO coupon(id, user_id, status)
VALUES (1001, 42, 0);
```

如果事务 A 在 `REPEATABLE READ` 下通过索引范围扫描锁住了相关区间，事务 B 的插入可能等待。新手常见误解是“只查了几行，为什么插入也被卡住”，真正原因不是查了几行，而是锁住了哪些索引区间。

下面给一个可运行的 Python 玩具程序，用区间模型模拟 next-key 的阻塞效果：

```python
def next_key_intervals(keys):
    keys = sorted(keys)
    intervals = []
    prev = None
    for k in keys:
        intervals.append((prev, k))  # 表示 (prev, k]
        prev = k
    intervals.append((prev, "supremum"))
    return intervals

def can_insert(locked_intervals, value):
    for left, right in locked_intervals:
        left_ok = left is None or value > left
        right_ok = right == "supremum" or value <= right
        if left_ok and right_ok:
            return False
    return True

keys = [10, 11, 13, 20]
intervals = next_key_intervals(keys)

# 模拟锁住 (10,11] 和 (11,13]
locked = [(10, 11), (11, 13)]

assert can_insert(locked, 12) is False
assert can_insert(locked, 11) is False
assert can_insert(locked, 14) is True
assert intervals[-1] == (20, "supremum")
```

这个程序不等价于数据库实现，但足够表达核心机制：一旦区间被锁，插入点就被占住了。

再给一个更接近线上排障的双事务步骤：

```sql
-- 事务 A
START TRANSACTION;
SELECT * FROM order_seq
WHERE biz_type = 'pay' AND seq_no >= 100 AND seq_no < 200
FOR UPDATE;

-- 事务 B
START TRANSACTION;
INSERT INTO order_seq(biz_type, seq_no) VALUES('pay', 150);
-- 等待 A 提交或回滚
```

如果 `biz_type, seq_no` 上有索引，A 锁住的是索引范围；如果没有合适索引，A 可能扫描更多记录，锁冲突会更广。

---

## 工程权衡与常见坑

关键点 1：隔离级别越高，不代表一定“更好”，而是意味着更强的一致性和更高的并发代价。

关键点 2：慢事务、死锁、插入堆积，很多时候不是数据库“性能差”，而是锁范围设计错了。

真实工程例子：

某库存服务在 `REPEATABLE READ` 下执行：

```sql
SELECT * FROM stock
WHERE sku_id = 9001 AND warehouse_id BETWEEN 1 AND 20
FOR UPDATE;
```

高峰期另一个接口不断插入新仓库存记录。结果是：

- 插入线程大量等待
- 事务持有时间变长
- 应用连接池耗尽
- 最后出现超时和死锁重试风暴

根因不是单条 SQL 慢，而是范围当前读触发 gap/next-key 锁，导致“新插入”被拦住。

必要元素：常见问题表

| 问题 | 常见原因 | 监控信号 | 缓解措施 |
|---|---|---|---|
| 插入被莫名阻塞 | 范围当前读触发 gap 锁 | Lock wait 增长 | 缩小范围、改索引、评估 RC |
| 死锁频发 | 多事务抢锁顺序不一致 | Deadlock 日志增加 | 统一按主键顺序访问 |
| 慢事务堆积 | 事务里夹杂远程调用 | 活跃事务时间变长 | 缩短事务、先算后写 |
| 锁范围过大 | 条件未命中索引 | 行扫描数高 | 补联合索引 |
| 自增热点 | 高并发插入争用 | 吞吐抖动 | 减少热点、分段或分库 |

常见坑主要有四个：

第一，误以为 `SELECT` 永远无锁。普通 `SELECT` 通常无锁，但 `FOR UPDATE` 不是。

第二，只看 SQL 条件，不看索引。数据库锁的是索引访问区间，不是你肉眼看到的“逻辑条件”。

第三，把事务写得太长。事务里如果还做 RPC、缓存访问、复杂计算，锁持有时间就被放大。

第四，忽视死锁的正常性。死锁不是异常中的异常，而是高并发下的自然结果，关键是让死锁少、重试轻、顺序一致。

---

## 替代方案与适用边界

关键点 1：不是所有业务都需要 `REPEATABLE READ` 的强范围保护。很多业务用 `READ COMMITTED` 就够了。

关键点 2：比“调隔离级别”更优先的，往往是“缩小扫描范围”和“统一抢锁顺序”。

先给出一个简表：

| 方案 | 优点 | 代价 | 适用场景 |
|---|---|---|---|
| 保持 RR + next-key | 幻读防护强 | 插入冲突更多 | 金额核对、强一致扣减 |
| 切到 RC | 并发更高 | 范围保护变弱 | 读多写多、可接受重试 |
| 补精确索引 | 锁范围小 | 需要索引维护 | 高频热点查询 |
| 主键顺序抢锁 | 死锁少 | 代码需统一规范 | 批量更新、多行修改 |

玩具例子：

原来事务按用户条件扫范围：

```sql
SELECT * FROM task
WHERE status = 0 AND created_at < NOW()
FOR UPDATE;
```

如果改成先通过精确索引取一批主键，再按主键顺序更新，锁范围会明显变小：

```text
1. 先查待处理任务 id 列表
2. 按 id 从小到大排序
3. 逐步更新或批量更新
```

统一顺序的目的，是让多个事务尽量按同一顺序拿锁，减少“你等我、我等你”的环。

简化伪代码：

```text
ids = select candidate ids
sort(ids)
for id in ids:
    update ... where id = ?
```

适用边界也必须明确：

- 如果业务要求“同一条件集合在事务期间绝不变化”，RR 的范围锁更合适
- 如果业务允许“看见后来插入的数据”，RC 往往能换来更高并发
- 如果冲突集中在少量热点键，优化索引和访问顺序通常比单纯切隔离级别更有效
- 如果是长事务报表类场景，优先考虑快照读，不要误用 `FOR UPDATE`

---

## 参考资料

| 来源 | 链接 | 用途 |
|---|---|---|
| MySQL 8.x Reference Manual: InnoDB Locking | https://dev.mysql.com/doc/refman/8.3/en/innodb-locking.html | 记录锁、间隙锁、next-key lock 的官方定义 |
| MySQL Reference Manual: Transaction Isolation Levels | https://dev.mysql.com/doc/refman/8.3/en/innodb-transaction-isolation-levels.html | 隔离级别与一致性读行为 |
| MySQL 历史文档：Consistent Read / Next-Key Locking | https://documentation.help/MySQL-5.0/ch14s02.html | 早期文档里对一致性读与 next-key 的说明 |
| DeepWiki: InnoDB Transaction and Locking System | https://deepwiki.com/mysql/mysql-server/2.1.1-innodb-transaction-and-locking-system | 工程视角理解锁系统实现结构 |
| DEV Community: PostgreSQL MVCC vs MySQL Key-Next Locking | https://dev.to/deko39/postgresql-mvcc-vs-mysql-key-next-locking-how-transaction-isolation-affects-concurrency-3a37 | 用对比方式理解 MySQL 范围锁对并发的影响 |
