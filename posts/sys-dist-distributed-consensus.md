## 核心结论

分布式一致性协议解决的问题，不是“所有机器内容完全一样”这么宽泛，而是更具体的目标：在节点宕机、消息延迟、网络分区存在时，仍然让一组副本对一串操作形成同一个提交顺序，并且这个顺序对外表现得像“只在一台可靠机器上执行过一次”。

它的容错核心是多数派。多数派的白话解释是：不是所有节点都同意才算成功，而是“超过一半”同意就生效。若集群规模为 $N=2f+1$，则最多容忍 $f$ 个 crash 故障；一次写入至少要被 $f+1$ 个节点持久化，后续的新领导者才一定能看到这条记录。公式写成：

$$
N = 2f + 1,\quad quorum = f + 1
$$

Paxos、Raft、ZAB 的共同骨架都是“选出可写入的主节点，再把日志复制到多数派，最后按统一顺序提交”。差异在于：

| 协议 | 核心目标 | 领导模型 | 提交依据 | 典型场景 |
| --- | --- | --- | --- | --- |
| Paxos | 理论完备的一致性证明 | 可演化为稳定 leader | 多数派接受提案 | Chubby、Spanner 一类元数据系统 |
| Raft | 把一致性过程拆得更容易实现 | 明确单 Leader | 当前任期日志获多数复制后提交 | etcd、Consul、控制平面 |
| ZAB | 为 ZooKeeper 提供顺序广播和恢复 | 主备广播模型 | Leader 广播事务并由多数确认 | ZooKeeper |

如果只记一个判断标准，可以记这句：一致性协议的安全性来自“多数派交集”，可用性则取决于“当前还能不能凑出多数派”。

---

## 问题定义与边界

分布式一致性的核心问题是：多个节点共同维护一份状态时，怎样在真实的不稳定环境下，让所有节点最终认同同一条操作历史。这里的“操作历史”可以理解为一份日志，谁都不能随意改写前面已经提交的顺序。

这个问题有明确边界。

第一，它通常讨论的是 crash fault，也就是节点停机、超时、断网、重启，但不会故意作恶。白话解释是：机器可能坏掉，但默认它不会伪造投票、不会恶意签两份相反结果。如果节点可能作恶，就进入拜占庭容错，不再是 Paxos、Raft、ZAB 的主战场。

第二，它关心的是一致顺序，不是高吞吐本身。吞吐、延迟、地理部署都重要，但它们是工程代价，不是协议首先保证的对象。

第三，它解决的是复制状态机问题。复制状态机的白话解释是：每台机器都按同样顺序执行同样命令，只要顺序一致，最终状态就一致。

一个最小玩具例子是 5 个队友投票定计划。只要 3 个人同意，计划就算正式通过；2 个人自己形成不了多数，就算他们内部达成一致，也不能代表整个团队。这对应分布式系统里的多数派写入。因为任意两个“3 票通过”的集合必然至少重叠 1 个人，所以新决定不会完全绕开旧决定，这就是多数派交集。

再看网络分区这个经典场景。5 节点集群被分成 3+2 两部分时：

| 分区 | 节点数 | 是否有多数派 | 能否选主 | 能否提交新写入 |
| --- | --- | --- | --- | --- |
| A 分区 | 3 | 是 | 可以 | 可以 |
| B 分区 | 2 | 否 | 不应该成功 | 不可以 |

这里体现了“安全优先于部分可用”。那 2 台机器即使还活着，也必须停止提交写入，否则就会出现 split brain。split brain 的白话解释是：系统同时出现两个都自称合法的主节点，各自接受写入，最后产生冲突历史。

所以一致性协议不是承诺“任何时候都能写”，而是承诺“只要继续写，就不会写乱”。这也是 CAP 里常见的取舍：发生分区时，要么暂停部分写请求保安全，要么继续服务但可能破坏全局顺序。Paxos、Raft、ZAB 在 crash fault 模型下都选择前者。

---

## 核心机制与推导

### 1. 多数派为什么能保证安全

多数派之所以有效，不是因为“人多力量大”，而是因为任意两个多数派一定相交。

设系统有 $N=2f+1$ 个节点，多数派大小是 $f+1$。若存在两个互不相交的多数派，那么它们总节点数至少是：

$$
(f+1) + (f+1) = 2f+2 > 2f+1 = N
$$

这与总节点数矛盾，所以两个多数派不可能完全不相交。于是，一条已经被多数派持久化的日志，未来任何合法领导者在重新竞选和恢复时，都必然能从交集节点那里“继承”到它。这就是“已提交日志不会丢”的数学基础。

### 2. Paxos、Raft、ZAB 的共同骨架

虽然三者术语不同，但都在处理三件事：

| 阶段 | Paxos | Raft | ZAB |
| --- | --- | --- | --- |
| 领导协商 | Prepare / Promise | RequestVote | discovery |
| 数据对齐 | Accept 前带历史约束 | AppendEntries 一致性检查 | sync |
| 正式广播 | Accept / Chosen | 日志复制并 commit | broadcast |

Paxos 里的 proposal number 是提案编号，白话解释是“这次提议的版本号”；Raft 里的 term 是任期，白话解释是“某一轮选举和领导期的编号”；ZAB 里的 epoch/ZXID 负责标识领导纪元和事务顺序。它们本质上都在回答同一个问题：当前谁的话算数，以及新日志能不能接在旧日志后面。

### 3. Paxos：先保证不会选错，再讨论效率

Paxos 的单次共识通常分两步：

1. Prepare：提案者拿一个更大的 proposal number，询问多数节点“我如果用这个编号提案，你们还会接受更小编号的吗？”
2. Accept：若多数节点承诺不再接受更小编号，提案者再发正式值，请它们接受。

关键点在于：如果某些节点已经接受过旧值，新的提案者必须继承其中编号最高的那个值，不能任意换内容。这样即使出现竞争提案，最终被多数派“选中”的值也不会冲突。

Paxos 理论严谨，但对初学者不友好，因为“哪个值必须继承”“单次共识和多次共识的关系”理解门槛高。工程里常用 Multi-Paxos，通过稳定 leader 把多轮 Prepare 省掉，后续日志像顺序复制。

### 4. Raft：把复杂性拆成可实现的规则

Raft 把过程拆成三块：选主、日志复制、安全提交。

会议类玩具例子可以这样理解：一个 leader 负责提议议程，跟随者收到后先记在本子上；只有当半数以上的人都记下，这个议程才算正式通过；旧候选人一旦看到更高 term，就必须退下，因为会议已经进入新一轮。

Raft 的关键安全规则有三个：

| 规则 | 含义 | 解决的问题 |
| --- | --- | --- |
| Election Safety | 一个任期最多一个 leader | 避免双主 |
| Log Matching | 若两条日志在同一 index 和 term 相同，则此前所有日志都相同 | 避免分叉历史混用 |
| Leader Completeness | 已提交日志一定出现在后续 leader 中 | 防止提交结果丢失 |

Raft 的复制流程是这样的：leader 收到客户端命令后先写本地日志，再通过 `AppendEntries` 发给 followers。每个 follower 会检查 `prevLogIndex` 和 `prevLogTerm`，只有前缀一致才接受新日志；否则拒绝，leader 回退重发，直到双方找到共同前缀。这样旧 leader 残留的未提交日志会被覆盖，但已提交日志不会被覆盖。

“只有当前 term 的日志才能由 leader 直接推进 commitIndex”是个常被忽略的细节。它的作用是防止一个新 leader 仅凭旧任期日志的多数副本，就错误地宣布这些旧日志已提交。正确做法是：新 leader 至少先提交一条当前任期日志，再借此把前面连续的旧日志一并视作已提交。

### 5. ZAB：围绕 ZooKeeper 的广播语义设计

ZAB 不是一个“泛用共识名字”，而是 ZooKeeper 的原子广播协议。它强调三阶段：

1. discovery：选出新的 leader，并确定谁的数据最新。
2. sync：leader 把缺失事务同步给 followers。
3. broadcast：后续新事务按统一顺序广播。

ZAB 特别适合 ZooKeeper 这种“写入必须全局有序，读侧依赖 watch 和会话语义”的系统。它依赖 ZXID 来标记事务顺序。ZXID 的白话解释是“全局单调递增的事务编号”。新的 leader 会先确保多数派收敛到同一个前缀，再开始广播新事务。

### 6. 真实工程例子

真实工程里，etcd 用 Raft 维护键值元数据。比如 Kubernetes 更新一个 Pod 对象，本质上是把“修改对象版本”的命令写进 etcd 的复制日志。leader 成功把这条命令复制到多数派并提交后，watcher 才会看到稳定事件。这里共识协议不是为了“大数据存储”，而是为了保证“控制平面的元数据顺序绝对一致”。

另一个例子是 ZooKeeper。配置变更、分布式锁节点创建、服务发现元数据更新，都依赖 ZAB 提供顺序广播。客户端看到的不是“某台机器刚好写成功”，而是“整个法定多数已经承认这个事务存在”。

---

## 代码实现

下面用一个极简的 Raft 提交模型说明核心字段和提交判定。它不是完整协议实现，但可以运行，并验证“只有当前任期日志在多数复制后才能直接提交”。

先看状态字段：

| 字段 | 含义 |
| --- | --- |
| `currentTerm` | 当前任期编号 |
| `votedFor` | 当前任期投给谁 |
| `log` | 日志数组，元素通常包含 `index/term/command` |
| `commitIndex` | 已确认提交的最大日志下标 |
| `lastApplied` | 已应用到状态机的最大日志下标 |

下面的玩具代码只模拟 leader 侧的提交推进逻辑。

```python
from dataclasses import dataclass

@dataclass
class Entry:
    index: int
    term: int
    command: str

class LeaderState:
    def __init__(self, current_term: int, cluster_size: int, log: list[Entry]):
        self.currentTerm = current_term
        self.clusterSize = cluster_size
        self.log = log
        self.commitIndex = 0
        # matchIndex[i] 表示第 i 个节点已复制到的最大日志下标，0 号节点视为 leader 自己
        self.matchIndex = [0] * cluster_size
        if log:
            self.matchIndex[0] = log[-1].index

    def majority(self) -> int:
        return self.clusterSize // 2 + 1

    def ack_from_follower(self, follower_id: int, replicated_index: int):
        self.matchIndex[follower_id] = replicated_index
        self.try_advance_commit()

    def try_advance_commit(self):
        last_index = self.log[-1].index if self.log else 0
        for candidate in range(last_index, self.commitIndex, -1):
            replicated = sum(1 for x in self.matchIndex if x >= candidate)
            if replicated >= self.majority():
                entry = self.log[candidate - 1]  # index 从 1 开始
                # 关键约束：只有当前 term 的日志才能被 leader 直接提交
                if entry.term == self.currentTerm:
                    self.commitIndex = candidate
                    return

# 三节点集群，leader 当前 term=3
log = [
    Entry(index=1, term=1, command="set x=1"),
    Entry(index=2, term=2, command="set y=2"),
    Entry(index=3, term=3, command="set z=3"),
]

leader = LeaderState(current_term=3, cluster_size=3, log=log)

# leader 自己已经有全部日志
assert leader.commitIndex == 0

# 一个 follower 只复制到 index=2，尚不足以提交 index=3
leader.ack_from_follower(follower_id=1, replicated_index=2)
assert leader.commitIndex == 0

# 第二个 follower 复制到 index=3，index=3 获得多数派且属于当前 term，可提交
leader.ack_from_follower(follower_id=2, replicated_index=3)
assert leader.commitIndex == 3

# 反例：如果最后一条日志不属于当前 term，则不能直接推进
old_term_log = [
    Entry(index=1, term=1, command="a"),
    Entry(index=2, term=2, command="b"),
]
leader2 = LeaderState(current_term=3, cluster_size=3, log=old_term_log)
leader2.ack_from_follower(1, 2)
leader2.ack_from_follower(2, 2)
assert leader2.commitIndex == 0

print("raft commit toy example passed")
```

这个例子故意省略了选举和回退细节，但保留了两个重要事实：

1. 多数复制是提交前提。
2. 提交推进要看日志任期，不是“多数到了就无条件提交”。

如果写成更接近工程实现的伪代码，leader 接收客户端命令通常是：

```text
onClientCommand(cmd):
  append (currentTerm, nextIndex, cmd) to local log
  persist log
  for each follower:
    send AppendEntries(
      term=currentTerm,
      prevLogIndex,
      prevLogTerm,
      entries,
      leaderCommit=commitIndex
    )

onAppendEntriesReply(follower, success, matchIndex, followerTerm):
  if followerTerm > currentTerm:
    stepDownToFollower(followerTerm)
    return

  if success:
    update follower matchIndex
    find largest N such that:
      N > commitIndex
      majority(matchIndex >= N)
      log[N].term == currentTerm
    commitIndex = N
    apply entries up to commitIndex
  else:
    decrement nextIndex[follower]
    retry
```

真实工程实现里，通常还要补上这些模块：

| 模块 | 责任 | 常见实现点 |
| --- | --- | --- |
| RPC 层 | 传输投票和复制请求 | `RequestVote`、`AppendEntries`、超时重试 |
| 持久化层 | 保证重启后不丢元数据 | `currentTerm`、`votedFor`、日志刷盘 |
| 状态机层 | 将已提交日志应用成业务状态 | `lastApplied` 递增执行 |
| 恢复层 | 领导变更后重新对齐前缀 | 冲突回退、快照安装 |
| 监控层 | 发现慢盘和选举抖动 | fsync、复制延迟、term 抖动 |

---

## 工程权衡与常见坑

一致性协议最常见的误解是“协议选对了，性能自然够用”。实际正相反，协议只给安全边界，性能主要由复制路径决定。

单 leader 模型下，写入延迟通常受三段路径支配：

$$
T_{write} \approx T_{leader\_fsync} + T_{quorum\_network} + T_{follower\_fsync}
$$

这里 `fsync` 的白话解释是“把内存里的写入真正落到磁盘，避免机器突然断电后数据丢失”。在很多系统里，慢的不是算法本身，而是磁盘落盘和跨机网络往返。

一个常见真实工程例子是：同一区域内部署 3 副本，且使用 NVMe，本地网络 RTT 很低时，写延迟可能在 2 到 6 ms。若扩展成 5 副本并跨可用区部署，为了等到“慢多数派”，写延迟常见会抬升到 5 到 15 ms；如果进一步跨地域，RTT 量级再上去，延迟还会明显增加。这里不是协议“坏了”，而是 quorum 必须等待更远的副本完成确认。

可以把主要瓶颈归纳成表：

| 瓶颈 | 现象 | 直接后果 | 常见缓解 |
| --- | --- | --- | --- |
| leader fsync 慢 | 单点磁盘忙、写 stall | 提交抖动、选举超时 | 更快磁盘、批量写、独立 WAL |
| 网络 RTT 高 | 跨区确认慢 | p50/p99 写延迟上升 | 拓扑收敛、就近多数派 |
| follower 落后 | 复制追赶慢 | 新 leader 切换时间长 | 快照、流水线复制 |
| leader CPU 热点 | 单点解析/编码压力大 | 吞吐受限 | 多分片 Raft 组 |
| 频繁选举 | term 持续增长 | 请求抖动、短暂不可写 | 调整超时、隔离慢盘 |

几个常见坑值得单独指出。

第一，误把“日志写到 leader 内存”当成提交。没有多数派持久化前，客户端不应收到成功。否则 leader 一宕机，这次写入就会蒸发。

第二，忽略旧 leader 的退位条件。只要看到更高 term，节点必须立刻转为 follower。否则容易出现短时间双主，尤其在网络抖动时。

第三，把“读请求”也全部强制走多数派，导致不必要的性能损失。工程里常用 leader lease、ReadIndex、只读租约等手段平衡读一致性与延迟，但前提是你清楚自己需要的是线性一致读还是最终一致读。

第四，日志无限增长。真实系统必须做快照和日志截断，否则恢复时间、磁盘占用、复制追赶都会失控。

第五，忽略存储层抖动带来的假故障。某些团队把超时只归因于网络，实际上 leader 本地 RocksDB stall 或 WAL fsync 尖峰也会触发 follower 误判，进而反复选举。

---

## 替代方案与适用边界

如果场景是内部可信机器、目标是元数据强一致，Raft 往往是最实用的起点。它更容易让团队在代码、排障、文档上保持一致理解。etcd、Consul 这类控制平面都属于这个路线。

如果系统追求更强的理论抽象，或者已有成熟 Paxos 系实现，Paxos 仍然成立。它不是“过时协议”，只是学习和实现门槛更高。工程里很多系统实际运行的是 Multi-Paxos，而不是教科书里那个单值版 Paxos。

如果你的系统就是 ZooKeeper 语义，尤其依赖顺序广播、会话、watch 通知，那么 ZAB 是更贴合的专用方案。它不是通用替代 Raft 的热门选择，而是与 ZooKeeper 的模型深度绑定。

再往外一层，是故障模型的边界。Paxos、Raft、ZAB 默认处理的是 crash fault。若节点可能作恶，就要转向 BFT 协议，例如 PBFT、Tendermint、HotStuff。BFT 的白话解释是：不仅机器会挂，还可能主动发送错误消息，所以协议要能在“有人撒谎”时仍达成一致。

两类模型可以直接对比：

| 协议类别 | 容错模型 | 节点规模要求 | 消息复杂度 | 适用场景 |
| --- | --- | --- | --- | --- |
| Paxos / Raft / ZAB | crash fault | $2f+1$ | 相对较低 | 数据中心内部控制面、元数据服务 |
| PBFT / Tendermint / HotStuff | Byzantine fault | $3f+1$ | 更高，常见接近 $O(n^2)$ | 不完全可信节点、联盟链、公链基础设施 |

一个直观判断方法是：如果所有节点都在同一家公司、同一运维体系内，通常先考虑 crash fault 协议；如果节点来自多个互不完全信任的主体，就要认真评估 BFT。

还要强调一个边界：一致性协议并不适合承载所有业务数据。大体量、高吞吐、允许分区局部写入的业务数据，通常放在分片存储、日志系统或数据库复制层；一致性协议更适合“少量但关键”的控制信息，比如配置、元数据、锁、选主状态、集群成员变更。

---

## 参考资料

1. DesignGurus: Paxos vs Raft vs ZAB，对三者的目标、领导模型和典型场景做了并列比较。  
2. AlgoMaster: 共识算法章节，适合回顾多数派、$N=2f+1$ 与故障容忍的基础推导。  
3. 阿里云开发者社区：Raft、ZAB 相关文章，适合补 ZooKeeper 广播模型与恢复流程。  
4. SystemOverflow: 一致性协议的性能、延迟、跨地域权衡，适合工程部署前估算瓶颈。  
5. 社区 Q&A / 博客资料：用于补充 3+2 分区、任期回退、日志覆盖等直观例子，但应以协议论文或主流实现文档为准。

DAQ 检索建议：
- D: DesignGurus，先看横向比较，建立整体框架。
- A: AlgoMaster，再看多数派、安全性和公式推导。
- Q: Q&A / 社区资料，用于补充具体案例和实现细节，再回到官方论文核实。
