## 核心结论

CAP 定理讨论的是分布式系统在**网络分区**发生时的硬约束。网络分区可以直白理解为“节点都还活着，但彼此暂时通信失败”。这时系统不可能同时保证：

- 一致性 `C`：所有节点读到同一份最新数据
- 可用性 `A`：每个未故障节点都必须返回结果，不能一直卡住
- 分区容错 `P`：网络断裂后系统仍继续工作

所以 CAP 不是“平时三选二”，而是：

$$
\text{发生网络分区} \Rightarrow \text{只能在 C 和 A 之间取舍}
$$

更准确地说：

$$
P \text{ 通常是分布式部署的默认前提} \Rightarrow \text{分区时只能选 CP 或 AP}
$$

BASE 理论不是推翻 CAP，而是在接受 AP 倾向的前提下，给出一套工程化表述：

- `Basically Available`：基本可用，系统优先继续响应
- `Soft state`：软状态，数据短时间内允许处于未收敛状态
- `Eventually consistent`：最终一致，经过异步同步后数据会收敛到一致

结论落到工程上，不是简单问“系统选 CP 还是 AP”，而是问“**哪类操作必须拒绝，哪类操作可以接受旧值或延迟一致**”。资金扣减、库存锁定、额度控制更偏 CP；日志、推荐、计数器近实时聚合、缓存失效传播更适合 BASE。

---

## 问题定义与边界

先把三个术语定准，否则后面容易混淆。

| 维度 | 严格定义 | 白话解释 | 常见误解 |
|---|---|---|---|
| 一致性 C | 所有副本对同一数据项的读结果一致，通常要求读到最近一次成功写入 | 任何节点看同一条数据，都像在看同一本最新账本 | 误以为“最终会一样”也算 C |
| 可用性 A | 每个非故障节点都必须在有限时间内返回非错误响应 | 只要机器还活着，请求就不能一直挂起 | 误以为“偶尔拒绝请求”也算高可用 |
| 分区容错 P | 节点间出现消息丢失、延迟、链路中断时，系统仍能继续运行 | 机房断链了，系统不能整体瘫痪 | 误以为 P 是可选功能 |

这里的“一致性”不是数据库 ACID 里的事务一致性，而是**副本一致性**。副本一致性可以直白理解为“多份数据拷贝是否对外呈现同一个值”。

一个最小玩具例子：

- 东区节点 `E`
- 西区节点 `W`
- 用户先在东区把余额从 `100` 改成 `80`
- 紧接着西区有人查询余额

如果东西区链路正常，西区先收到同步，再返回 `80`，这时系统既一致也可用。

如果此时链路断了，系统必须做决定：

- 选 CP：西区拒绝读或拒绝写，直到确认最新值
- 选 AP：西区继续返回结果，但可能还是旧值 `100`

所以 CAP 的触发条件很明确：**不是只要有分布式就立刻损失一半能力，而是在分区出现时必须做选择**。

这也是边界所在。很多新人把 CAP 理解成“分布式系统永远不可能同时高一致和高可用”，这不准确。正常无分区时，系统完全可以做到“当前既一致又可用”。真正不可兼得的是“**分区发生的那一刻**”。

---

## 核心机制与推导

CAP 的推导可以压缩成一条逻辑链：

1. 系统有多个副本
2. 副本之间靠网络复制数据
3. 网络可能分区
4. 分区后，某些副本拿不到最新写入
5. 此时如果还强行要求所有请求都成功返回，就可能返回旧值
6. 如果还强行要求所有读都必须最新，就必须让部分节点拒绝请求

于是得到：

$$
\text{Partition} \Rightarrow
\begin{cases}
\text{保一致} \Rightarrow \text{牺牲部分可用性}\\
\text{保可用} \Rightarrow \text{接受暂时不一致}
\end{cases}
$$

### CP 的机制

CP 可以理解为“先对齐账本，再对外答复”。

常见做法是：

- 写请求必须拿到多数派确认才算成功
- 如果当前节点失去多数派，就拒绝写
- 某些严格读也必须从主节点或多数派读取

白话解释：多数派就是“超过半数节点同意”。它的作用是保证不会同时出现两个都自称最新的版本。

例如 3 节点集群 `A/B/C`，多数派是 2。若 `C` 被隔离：

- `A+B` 还能组成多数派，继续提供写服务
- `C` 因为只有自己，不能确认全局最新状态，只能拒绝写

这样牺牲的是 `C` 所在分区的可用性，换来的是全局不会出现两份互相冲突的“最新值”。

### AP / BASE 的机制

AP 不要求每次读写都立刻全局对齐，而是先保证服务活着，再让后台同步追平。

BASE 的三个词可以连起来理解：

- `Basically Available`：先别让系统整体停掉
- `Soft state`：允许节点暂时各自持有不同版本
- `Eventually consistent`：只要没有新的更新持续打进来，经过同步后最终会收敛

设某键在东区版本号为 `v=5`，西区因分区停留在 `v=4`。若西区继续对外读写，那么短时间可能出现：

$$
state_{east} \neq state_{west}
$$

但网络恢复后，通过版本比较、重放日志、冲突合并，目标是达到：

$$
\lim_{t \to \infty} state_{east}(t) = state_{west}(t)
$$

这就是最终一致。

### 玩具例子：两个机房的配置中心

假设有一个配置项 `discount_rate`：

- 东区最新值是 `0.85`
- 西区在分区前还是 `0.90`

如果是 CP 设计：

- 西区读取时发现自己无法确认全局最新值
- 直接报错或降级为“配置暂不可读”

如果是 AP/BASE 设计：

- 西区继续返回 `0.90`
- 同时标记“本地副本已过期”
- 网络恢复后再异步同步成 `0.85`

这个例子说明：配置中心如果控制支付风控参数，偏 CP；如果只是首页展示样式开关，偏 AP 更合理。

### 真实工程例子：电商系统

一个电商平台同时有这些接口：

| 接口 | 更合适的倾向 | 原因 |
|---|---|---|
| 扣减库存 | CP | 超卖比短暂失败更危险 |
| 账户扣款 | CP | 金额错误通常不可接受 |
| 评论列表读取 | AP/BASE | 短时间旧数据影响较小 |
| 推荐流刷新 | AP/BASE | 更看重持续响应 |
| 缓存失效广播 | AP/BASE | 允许短时间读到旧缓存 |

这里最重要的工程结论是：**CAP 的决策粒度应该是接口级、操作级，而不是整套系统级**。同一个系统内部完全可以同时存在 CP 路径和 AP 路径。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，展示“分区时 CP 拒绝写，AP 接受写并在恢复后收敛”的核心逻辑。这个例子不是生产级协议，只是把分支决策映射成代码。

```python
from dataclasses import dataclass, field

@dataclass
class Replica:
    name: str
    value: int = 0
    version: int = 0
    pending: list = field(default_factory=list)

    def apply(self, value: int):
        self.version += 1
        self.value = value

class Cluster:
    def __init__(self):
        self.east = Replica("east")
        self.west = Replica("west")
        self.partitioned = False

    def set_partition(self, flag: bool):
        self.partitioned = flag

    def write_cp(self, region: str, value: int):
        # CP: 分区时必须能同步到另一侧，否则拒绝写
        target = self.east if region == "east" else self.west
        other = self.west if region == "east" else self.east

        if self.partitioned:
            raise RuntimeError(f"CP reject write on {region}: partition detected")

        target.apply(value)
        other.value = target.value
        other.version = target.version

    def write_ap(self, region: str, value: int):
        # AP/BASE: 分区时先写本地，并把操作记入待同步队列
        target = self.east if region == "east" else self.west
        other = self.west if region == "east" else self.east

        target.apply(value)
        if self.partitioned:
            target.pending.append((target.version, value))
        else:
            other.value = target.value
            other.version = target.version

    def heal_and_merge_last_write_wins(self):
        # 恢复后用版本号做一个最简单的 LWW 合并
        all_ops = []
        for replica in (self.east, self.west):
            all_ops.extend(replica.pending)
            replica.pending.clear()

        latest = max(
            [(self.east.version, self.east.value), (self.west.version, self.west.value)] + all_ops,
            key=lambda x: x[0]
        )
        version, value = latest
        self.east.version = version
        self.west.version = version
        self.east.value = value
        self.west.value = value
        self.partitioned = False


# CP 场景：分区时拒绝写
c1 = Cluster()
c1.write_cp("east", 10)
assert c1.east.value == 10 and c1.west.value == 10

c1.set_partition(True)
try:
    c1.write_cp("west", 20)
    assert False, "CP mode should reject write during partition"
except RuntimeError:
    pass

# AP/BASE 场景：分区时继续写，恢复后收敛
c2 = Cluster()
c2.write_ap("east", 10)
assert c2.east.value == 10 and c2.west.value == 10

c2.set_partition(True)
c2.write_ap("west", 20)   # 西区继续服务
assert c2.west.value == 20
assert c2.east.value == 10  # 东区暂时还是旧值

c2.heal_and_merge_last_write_wins()
assert c2.east.value == 20 and c2.west.value == 20
```

这段代码刻意保留了几个核心点：

- `partitioned=True` 代表网络分区
- `write_cp` 在分区时直接拒绝
- `write_ap` 在分区时只写本地，并记录 `pending`
- `heal_and_merge_last_write_wins` 表示恢复后的异步收敛

如果把它映射到真实工程，通常会变成下面的伪逻辑：

```python
if partition_detected():
    if operation.requires_strict_consistency:
        reject_request("SERVICE_UNAVAILABLE_OR_NOT_LEADER")
    else:
        accept_locally()
        append_replication_log()
        enqueue_async_sync_task()
else:
    write_to_quorum()
    replicate_to_followers()
```

真实工程例子：库存服务通常会把“扣减库存”设计成多数派提交或单主写入，因为它不能接受同一件商品被两个分区同时卖出。评论流、点赞数、推荐结果则可能采用异步复制，因为用户更在意页面能不能刷出来，而不是所有机房同一毫秒完全一致。

---

## 工程权衡与常见坑

CAP 和 BASE 进入工程后，核心不是背定义，而是识别“错误的代价”。

| 业务类型 | 更常见选择 | 如果选错的后果 | 常见缓解手段 |
|---|---|---|
| 资金扣减 | CP | 账错、重复扣款、对账困难 | 幂等键、事务日志、对账补偿 |
| 库存/配额 | CP | 超卖、超发额度 | 预留库存、排队、限流 |
| 评论/点赞展示 | AP/BASE | 短时间显示旧值 | 异步回刷、用户提示 |
| 推荐/Feed | AP/BASE | 数据短暂滞后 | 后台重算、版本覆盖 |
| 缓存失效传播 | AP/BASE | 读到旧缓存 | TTL、双删、版本号 |

### 常见坑 1：把 CAP 理解成系统只能贴一个标签

现实里，一个电商站点不会整体叫“CP 系统”或“AP 系统”。更准确的说法是：

- 订单状态流转里的某些写路径偏 CP
- 首页推荐、埋点、搜索提示词偏 AP/BASE
- 同一个服务内部，不同接口也可能不同策略

### 常见坑 2：以为 AP 就等于“随便不一致”

AP 不是放弃治理，而是把治理手段从“同步阻塞”换成“异步收敛”。常见补充机制包括：

- 版本号
- 幂等键
- 去重日志
- 冲突合并策略
- 补偿任务
- 死信队列

没有这些，AP 只会退化成“数据烂掉”。

### 常见坑 3：忽略读路径的一致性要求

很多团队只盯写请求，忘了读请求也有一致性等级。例如：

- 账户余额页读取，通常不能读旧值太久
- 评论列表读取，读到几秒前的数据问题不大

所以设计时要明确：

- 哪些读必须读主
- 哪些读可以读副本
- 哪些接口可以带“可能延迟”的产品说明

### 常见坑 4：只在文档里谈 CAP，不做故障演练

纸面上“我们选 CP”没有意义，必须验证以下场景：

- 主从链路中断时，请求是否真的被拒绝
- 跨机房延迟飙升时，是否误判为分区
- 恢复后是否会出现脏写覆盖
- AP 路径的补偿队列是否会堆积失控

很多事故不是因为不懂 CAP，而是因为**实现和宣称的策略不一致**。

---

## 替代方案与适用边界

BASE 是 CAP 约束下的一种工程立场，但不是唯一答案。更常见的是“混合策略”。

### 方案一：关键写 CP，非关键读 AP

这是最常见的折中方式。

例子：社交平台用户资料更新。

- 用户名、实名认证资料写入偏 CP，因为错误覆盖代价高
- 朋友圈时间线、点赞计数偏 BASE，因为更重视页面持续可用

### 方案二：核心状态 CP，派生数据 AP

“派生数据”可以直白理解为“由主数据计算出来的数据副本”。例如：

- 订单主状态：CP
- 订单统计报表：AP/BASE
- 库存主账本：CP
- 商品热度榜单：AP/BASE

这样做的好处是把高一致成本集中在小而关键的数据面上，而不是让所有链路都走最重协议。

### 方案三：单主强一致，多副本最终一致

很多系统并不追求“全球任意点都能写”，而是：

- 核心写入只进主区域
- 其他区域主要承担读流量
- 通过异步复制把数据扩散出去

这不是逃离 CAP，而是通过限制写拓扑来降低冲突面。

### 适用边界判断

可以用一个简单决策表：

| 判断问题 | 倾向 |
|---|---|
| 错一次是否会造成资金、法律、审计风险？ | CP |
| 用户能否容忍几秒到几分钟旧数据？ | AP/BASE |
| 是否可以通过补偿、重算、去重修复？ | 更适合 AP/BASE |
| 请求失败是否比旧数据更糟？ | 更适合 AP/BASE |
| 冲突一旦发生是否很难自动合并？ | CP |

一句更实用的话是：**先看业务能承受哪种错，再选一致性模型**。  
因为 CAP 讨论的是“分区时怎么错”，而工程设计讨论的是“哪种错更便宜、更可恢复”。

---

## 参考资料

- IBM, *What Is the CAP Theorem?*  
- System Design Sandbox, *The CAP Theorem*  
- GeeksforGeeks, *CAP Theorem vs. BASE Consistency Model - Distributed System*
