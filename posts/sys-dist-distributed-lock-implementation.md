## 核心结论

分布式锁的定义很直接：它是一个由外部协调服务维护的“全局互斥开关”，保证多个进程、多个机器在竞争同一份资源时，任意时刻只有一个活跃持有者。这里的“活跃持有者”意思是：不仅拿到锁，而且锁还没有过期、没有被别人接管。

它真正难的地方，不是“某次加锁成功”，而是下面五件事能否同时成立：

| 能力 | 白话解释 | 为什么关键 |
| --- | --- | --- |
| 互斥 | 同一时刻只能有一个人进去 | 避免并发写坏数据 |
| 可重入性 | 同一持有者再次进入时不要把自己挡住 | 避免业务递归或嵌套调用卡死 |
| 租约 | 锁不是永久的，有过期时间 | 防止持有者崩溃后永不释放 |
| 故障释放 | 持有者挂了，别人最终还能继续工作 | 保证系统可恢复 |
| 误删防护 | 只能删掉“自己的锁” | 防止把别人的锁误释放 |

对零基础读者，先记一个结论：Redis 用 `SET key value NX PX ttl` 可以很低成本地做分布式锁，适合短临界区；ZooKeeper 或 etcd 依赖临时节点或 lease，强一致性更强，但写延迟更高。锁不是万能协调工具，长事务、跨系统大流程、需要人工介入的审批链，通常不适合靠分布式锁硬控。

一个最小玩具例子是“多人抢同一文件处理权”。客户端先写入 `lock:file`，并附带 30 秒 TTL。别人看到这个键存在，就先不处理。持有者定期续租；如果它宕机，TTL 到期后其他客户端再抢。这个模型能工作，但前提是你必须处理续租失败、网络抖动、误删和旧客户端苏醒后继续执行的问题。

续租和超时释放的关系可以先用一个近似公式理解：

$$
effective\_hold \approx \min(TTL,\ renew\_interval - network\_delay)
$$

它表达的不是严格协议证明，而是工程直觉：名义 TTL 并不等于你真实还能安全持有多久，网络延迟、调度延迟、GC 暂停都会吃掉安全边界。

| 场景 | 行为 | 风险 |
| --- | --- | --- |
| 持锁者持续续租 | 锁一直保留给当前客户端 | 续租线程卡住会导致“以为还持有” |
| 锁超时自动释放 | 其他客户端可以继续抢锁 | 原持有者醒来后可能误操作旧资源 |

---

## 问题定义与边界

问题定义可以压缩成一句话：在多进程、多机器环境下，如何让同一资源在同一时刻只被一个执行单元修改。这里的“资源”可以是库存记录、定时任务、订单状态机、某个文件、某个分区的消费权。

边界比定义更重要。因为单机锁只考虑线程调度，而分布式锁要考虑网络和机器故障。你需要明确以下边界：

| 失败场景 | 白话解释 | 边界要求 |
| --- | --- | --- |
| 持有者宕机 | 机器直接挂掉，不会主动解锁 | 必须有 TTL 或 lease 自动释放 |
| 网络延迟 | 消息到得慢 | TTL 要覆盖合理抖动 |
| GC 暂停 | 进程没死，但几秒不能工作 | 续租要有安全裕量 |
| 网络分区 | 客户端和锁服务短暂失联 | 失联后不能默认自己仍持有 |
| 主从切换 | 新主数据可能落后 | 不能只依赖异步复制状态 |
| 误删 | 客户端删掉别人的锁 | 解锁必须校验 token |

一个最简单的竞争场景是：节点 A 和节点 B 同时执行

```text
SET resource myid NX PX 30000
```

`NX` 的意思是“只在不存在时设置”，也就是只有第一个成功的人能写进去；`PX 30000` 的意思是“毫秒级过期时间 30 秒”。这样就建立了最基础的互斥。

但请注意，这只解决了“抢到”这一步，没有解决“抢到后系统暂停 40 秒怎么办”“解锁时是否删错”“Redis 主从切换后新主是否认识这把锁”这些问题。所以讨论分布式锁时，不能只看加锁 API 是否原子，还要看完整生命周期。

这里给一个真实工程例子。假设有一个订单超时取消任务，部署了 10 个实例，每分钟都会扫描“待取消订单”。如果没有锁，10 个实例会重复取消、重复发消息、重复退款。此时可以用“任务级锁”保证某一分钟的扫描只有一个实例执行。但如果这个扫描本身要跑 2 分钟，而你的 TTL 只给了 30 秒，那么锁会中途失效，第二个实例会进来重复处理。这说明锁的边界必须和业务耗时匹配。

---

## 核心机制与推导

### 1. Redis 的基本做法

Redis 方案的核心是单条原子命令：

```text
SET foo abc123 NX PX 30000
```

这里 `abc123` 不是随便写的字符串，而是“持有者令牌”，也就是一段随机 token，用来标识“这把锁是谁加的”。它的白话解释是：锁名字相同不够，必须知道锁主人是谁，否则无法安全释放。

Redis 的正确释放不是直接 `DEL foo`，而是先比对 token，再删除。通常写成 Lua 脚本：

```lua
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
```

原因很简单。假设 A 拿锁 30 秒后发生长时间 GC，锁超时了；B 随后拿到同名锁。此时 A 恢复执行，如果它直接 `DEL foo`，删掉的就是 B 的锁。这就是“误删”。

### 2. 续租为什么需要数学边界

如果业务执行时间可能超过 TTL，就需要续租。续租就是在锁还属于自己的前提下，延长过期时间。工程上常用关系是：

$$
renew\_interval < TTL - network\_margin
$$

其中 `network_margin` 是网络延迟、调度抖动、GC 暂停预留出的安全边界。比如 TTL 是 30 秒，续租间隔可以是 10 秒，但不能是 29 秒。因为只要一次抖动超过 1 秒，锁就会先过期，再被别人拿走。

更进一步，可以把“是否仍安全持有”近似理解为：

$$
safe\_window = TTL - pause\_time - network\_delay - clock\_drift
$$

只要 `safe_window <= 0`，你就不能再假设自己一定还是持有者。这里的 `clock_drift` 是时钟漂移，白话解释是不同机器的时间不完全一致。在 Redis 多实例或跨机房场景里，这个因素不能完全忽略。

### 3. ZooKeeper / etcd 的思路

ZooKeeper 和 etcd 的思路不是“设置一个带 TTL 的普通键”，而是创建临时节点或 lease。临时节点的白话解释是：客户端会话还在，节点就在；会话断了，节点自动回收。

例如在 ZooKeeper 中，客户端尝试创建 `/locks/foo` 的临时节点。创建成功就是拿到锁；创建失败说明别人已持有。其他客户端可以监听这个节点，一旦它被删除，就立刻收到通知并重试。etcd 则用 lease 绑定键，lease 到期或连接断开，键自动删除。

它们的优势是更强的一致性：写入通常要经过多数派确认。代价是写路径更长，延迟更高，吞吐更低。

### 4. 玩具例子与真实工程例子

玩具例子：两个爬虫实例都想写同一个输出文件。A 成功执行 `SET lock:file tokenA NX PX 30000`，B 失败，只能等待。A 每 10 秒续租一次；如果 A 宕机，30 秒后 B 再尝试，成功后继续写。

真实工程例子：电商平台做“优惠券库存扣减”。如果只是数据库库存减一，通常更推荐数据库原子更新或乐观锁，而不是外部锁。只有当扣减逻辑跨缓存、消息队列、审计日志多个组件，且需要把“同一优惠券批次的复杂更新”包成短临界区时，分布式锁才有明确价值。即便如此，也仍要配合幂等和库存约束，不能只靠锁兜底。

---

## 代码实现

下面用 Python 写一个“可运行的本地玩具实现”，它不是 Redis 客户端，而是用内存模拟“带 token、TTL、续租、校验释放”的核心行为。目的是把协议要点讲清楚。

```python
import time
import uuid


class InMemoryLeaseLock:
    def __init__(self):
        self.store = {}

    def _now_ms(self):
        return int(time.time() * 1000)

    def _cleanup_if_expired(self, key):
        item = self.store.get(key)
        if not item:
            return
        if item["expire_at"] <= self._now_ms():
            del self.store[key]

    def try_lock(self, key, ttl_ms):
        self._cleanup_if_expired(key)
        if key in self.store:
            return None
        token = str(uuid.uuid4())
        self.store[key] = {
            "token": token,
            "expire_at": self._now_ms() + ttl_ms,
        }
        return token

    def renew(self, key, token, ttl_ms):
        self._cleanup_if_expired(key)
        item = self.store.get(key)
        if not item:
            return False
        if item["token"] != token:
            return False
        item["expire_at"] = self._now_ms() + ttl_ms
        return True

    def release(self, key, token):
        self._cleanup_if_expired(key)
        item = self.store.get(key)
        if not item:
            return False
        if item["token"] != token:
            return False
        del self.store[key]
        return True


lock = InMemoryLeaseLock()

token_a = lock.try_lock("order:42", ttl_ms=200)
assert token_a is not None

token_b = lock.try_lock("order:42", ttl_ms=200)
assert token_b is None  # 互斥：第二个持有者不能进入

assert lock.renew("order:42", token_a, ttl_ms=200) is True
assert lock.release("order:42", token_a) is True

token_c = lock.try_lock("order:42", ttl_ms=200)
assert token_c is not None

# 旧 token 不能误删新锁
assert lock.release("order:42", token_a) is False
assert lock.release("order:42", token_c) is True
```

如果换成 Redis，核心逻辑通常如下：

```python
LOCK_SCRIPT_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""

LOCK_SCRIPT_RENEW = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('pexpire', KEYS[1], ARGV[2])
else
  return 0
end
"""
```

加锁命令对应：

```text
SET resource_name random_token NX PX 30000
```

这三部分分别解决：

| 函数 | 输入 | 成功条件 | 失败处理 |
| --- | --- | --- | --- |
| `try_lock` | `key, ttl` | 键不存在，原子写入 token 和 TTL | 返回失败，按退避策略重试 |
| `renew` | `key, token, ttl` | 当前 token 仍匹配 | 立即停止业务或进入自检 |
| `release` | `key, token` | 当前 token 仍匹配 | 返回失败，不得强删 |

真实工程里还要再补三层：

1. 重试策略。不要所有实例固定每 100ms 重试，否则会形成锁风暴。
2. 超时感知。等待锁的总时间应该受业务 SLA 限制。
3. 业务侧二次校验。即使拿到锁，真正写数据前最好仍有版本号、状态机或幂等键兜底。

一个常见伪代码如下：

```text
tryLock():
  token = random()
  ok = SET key token NX PX ttl
  if ok:
    start renew loop
    return token
  else:
    sleep with jitter
    retry or fail fast

renew():
  if compare token success:
    extend ttl
  else:
    stop working immediately

release():
  stop renew loop
  run lua compare-and-del
```

---

## 工程权衡与常见坑

Redis 锁容易上手，但工程坑很多，主要集中在“一致性没有你想得那么强”。

| 坑 | 现象 | 规避措施 |
| --- | --- | --- |
| TTL 太短 | GC 或网络抖动后锁提前过期 | TTL 至少覆盖业务 P99 耗时和安全边界 |
| TTL 太长 | 持有者挂了后别人等太久 | 配合续租，不要靠超长 TTL 硬撑 |
| 直接 `DEL` | 误删别人新获得的锁 | 必须 compare-and-del |
| 主从异步复制 | 主挂后新主不知道旧锁 | 不把单主从 Redis 当强一致锁服务 |
| 续租线程和业务线程分离失控 | 业务已卡死但续租还在继续 | 续租与业务生命周期绑定 |
| 只靠锁不做幂等 | 锁失效后出现重复执行 | 业务写路径仍要有幂等约束 |

最容易误解的坑，是“我已经加了锁，为什么还要 fencing token”。fencing token 可以翻译成“单调递增的操作序号”。白话说法是：每次成功拿锁，都给你发一个越来越大的号码；下游资源只接受号码更大的请求。这样就算旧客户端因为 GC 暂停晚醒，也会因为号码过旧被拒绝。

例如：

```text
lock grant #101 -> client A
lock expires
lock grant #102 -> client B
client A wakes up and tries to write
storage rejects #101 because #102 already appeared
```

这比“只判断当前有没有锁”更强，因为它把“旧持有者误操作”挡在资源入口。

GC 暂停例子很典型。假设 A 拿到 30 秒锁，执行到第 20 秒发生 15 秒 Stop-The-World。第 30 秒时锁已过期，B 成功拿锁。第 35 秒 A 醒来，如果它没有在每次关键写入前确认自己仍是合法持有者，或者没有 fencing token，下游就可能被 A 和 B 同时写坏。

真实工程里，锁最适合“短、窄、可回滚”的临界区，例如：

- 同一任务分片只能由一个 worker 拉取
- 同一批次对账只能启动一个执行器
- 同一个用户的低频配置变更串行化

它不适合：

- 持续几分钟甚至几小时的长流程
- 跨多个外部系统的人机混合流程
- 一旦重复执行就会产生不可逆副作用，但下游又没有幂等保护的流程

---

## 替代方案与适用边界

分布式锁只是协调手段之一，不是唯一手段。选择方案时，核心不是“谁最流行”，而是谁的故障模型和你的业务更匹配。

| 方案 | 一致性 | 性能 | 适用场景 | 不适合 |
| --- | --- | --- | --- | --- |
| Redis `SET NX PX` | 中等，依赖部署方式 | 高 | 短临界区、低延迟协调 | 强一致要求极高的场景 |
| ZooKeeper 临时节点 | 高 | 中 | 需要 watcher、强一致控制 | 高频低延迟写入 |
| etcd lease | 高 | 中 | 云原生控制面、选主、配置协调 | 超高吞吐业务热路径 |
| 数据库乐观锁 | 取决于数据库事务 | 中低 | 单库内状态更新 | 跨系统资源协调 |
| 唯一约束 + 幂等键 | 高 | 中 | 防重复提交、一次性任务 | 需要长期持有资源 |
| 消息队列串行化 | 中高 | 中高 | 按 key 顺序消费 | 需要即时抢占式互斥 |

Redis 适合什么边界？适合临界区很短、锁失败可以重试、偶发锁漂移可以由业务幂等兜住的情况。比如“同一篇文章的静态页只允许一个实例重建”。

ZooKeeper 或 etcd 适合什么边界？适合“锁本身就是系统控制面一部分”，例如主节点选举、分片归属、调度器 leader 选举、配置变更串行化。这类场景比起极致低延迟，更看重顺序性和一致性。

数据库乐观锁是另一种常见替代。它不是“先占资源再执行”，而是“执行时带版本号提交，版本不对就失败重试”。对白话理解就是：不阻止别人开始做，但提交时只允许一个版本成功。对于库存扣减、余额变更、状态迁移，这通常比外部锁更直接。

再给一个真实工程例子。假设你要防止“同一订单重复退款”。很多团队第一反应是给 `refund:order_id` 加 Redis 锁。更稳的做法往往是：

- 数据库里用唯一退款单号或状态机约束
- 对外部支付网关使用幂等请求号
- 必要时再加短时分布式锁减少冲突

也就是：锁可以减并发冲突，但真正的正确性仍应落在数据约束和幂等上。

---

## 参考资料

1. SystemOverflow, *What are Distributed Locks and Why Do They Need Leases?*  
   重点：解释为什么分布式锁必须引入 lease，也就是带过期时间的租约模型；适合建立基础概念。

2. Architecture Weekly, *Distributed Locking: A Practical Guide*  
   重点：从工程角度对比 Redis、ZooKeeper、etcd 的实现路径与适用场景，强调强一致与延迟的取舍。

3. Medium, *Implementing Distributed Locks Correctly*  
   重点：说明 Redis `SET NX PX`、随机 token、Lua compare-and-del 等关键细节，适合理解最小正确实现。

4. Zeeklog, *Distributed Locks with Redis*  
   重点：讨论 Redis 在主从切换、TTL 设置、续租和 fencing token 方面的常见坑。
