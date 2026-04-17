## 核心结论

分布式事务的目标，是把多个服务、多个数据库上的操作，协调成一次“对外看起来要么都成功、要么都失败”的业务动作。这里的“事务”不再只靠单库锁和日志完成，而是靠协议协调多个节点共同决策。

最重要的判断不是“哪种协议最先进”，而是这三个问题：

| 问题 | 说明 | 直接影响 |
|---|---|---|
| 数据能不能补偿 | 补偿，白话说就是“出错后有没有办法反向撤销” | 能补偿时优先考虑 Saga |
| 资源能不能预留 | 预留，白话说就是“先占住额度/库存，稍后再确认” | 能预留时适合 TCC |
| 能容忍多长不一致窗口 | 不一致窗口，白话说就是“短时间内各系统状态不同步能不能接受” | 不能容忍时才考虑更强一致方案 |

四类常见方案可以先记成一张表：

| 方案 | 一致性强度 | 可用性影响 | 核心代价 | 典型场景 |
|---|---|---|---|---|
| 2PC | 强一致较强 | 较低，容易阻塞 | 锁持有时间长，协调者故障敏感 | 同机房、传统数据库协调 |
| 3PC | 理论上比 2PC 更少阻塞 | 仍受网络分区影响 | 假设条件强，实现复杂 | 可信网络中的学术/特定系统 |
| TCC | 业务可控的强一致 | 中等 | 业务要实现 Try/Confirm/Cancel | 支付冻结、额度预占 |
| Saga | 最终一致 | 高 | 需要补偿逻辑与状态追踪 | 跨服务下单、跨区域流程 |

玩具例子是“买一件 100 元商品”：订单服务要同时让余额服务扣 100 元、库存服务减 1 件。真实工程里，这通常还会再加优惠券、积分、物流、发票等服务，链路越长，分布式事务越不适合靠长时间加锁硬扛。

简短结论：强一致方案靠“统一提交”，高可用方案靠“显式补偿”，选型关键在补偿能力、资源预留能力和时延预算。

---

## 问题定义与边界

单机数据库事务解决的是一个数据库实例内部的一致性问题。分布式事务解决的是多个独立节点之间的一致性问题。这些节点可能各自有数据库、各自有锁、各自独立部署，彼此之间只能通过网络发消息。

边界先说清楚：

1. 如果所有数据都在同一个数据库里，优先用单库事务，不要上分布式事务。
2. 如果只是“消息最终会送达”，但业务能接受稍后修正，很多场景用可靠消息加本地事务就够了。
3. 只有当一个业务动作必须跨多个资源管理器统一决策时，才进入分布式事务问题域。

最小失败例子如下。协调者同时向余额服务和库存服务发 `PREPARE`：

- 余额服务锁住 100 元，返回“我可以提交”
- 库存服务锁住 1 件商品，返回“我可以提交”
- 协调者正准备发最终 `COMMIT`，结果宕机

这时两个参与者都处于 `PREPARED` 状态。`PREPARED` 可以理解成“我已经把资源占住，随时等最终命令”。问题是：它们不知道最终应该提交还是回滚，于是锁不能随便放。新手版本可以理解成：调度员让两个仓库都把东西先留出来，但调度员失联了，仓库既不敢发货，也不敢把货重新上架。

2PC 的核心消息顺序可以写成：

$$
\text{PREPARE} \rightarrow \text{VOTE} \rightarrow \text{COMMIT/ABORT}
$$

其中最麻烦的一点是：

$$
\text{participant in PREPARED state} \Rightarrow \text{cannot safely release lock before decision}
$$

也就是，只要参与者已经投票说“我准备好了”，在不知道最终决策前，安全做法就是继续占着资源。

常见失败边界如下：

| 失败类型 | 发生位置 | 影响 | 典型后果 |
|---|---|---|---|
| 网络抖动 | 协调者与参与者之间 | 消息超时、重复投递 | 决策延迟、重试风暴 |
| 协调者宕机 | 统一决策节点 | 参与者无法得知最终结果 | 2PC 阻塞 |
| 参与者宕机 | 某个业务服务 | 局部状态丢失或恢复慢 | 无法 prepare 或无法 confirm |
| 网络分区 | 节点被分成多个孤岛 | 各节点观察到的世界不同 | 3PC 也可能分裂决策 |
| 重复请求 | 上游重试或消息重复 | 同一事务被多次执行 | 需要幂等控制 |

这里的“幂等”首次出现，意思是“同一个请求做一次和做多次，结果应该一样”。如果没有幂等，重试机制本身就会制造数据错误。

---

## 核心机制与推导

### 1. 2PC：两阶段提交

2PC 由协调者和参与者组成。协调者负责收集意见并发布最终决策。

状态可以简化成：

| 角色 | 阶段 | 动作 | 失败时的安全行为 |
|---|---|---|---|
| 协调者 | INIT | 创建事务 | 直接终止 |
| 协调者 | PREPARE | 询问所有参与者是否可提交 | 未收到全票前可 abort |
| 协调者 | COMMIT/ABORT | 广播最终决策 | 需依赖日志恢复重发 |
| 参与者 | INIT | 接收请求 | 可直接拒绝 |
| 参与者 | PREPARED | 锁资源并写日志 | 等待最终决策，不能随便释放 |
| 参与者 | COMMITTED/ABORTED | 完成提交或回滚 | 幂等处理重复消息 |

推导逻辑很直接：

1. 如果协调者不先问各方是否准备好，就可能在一部分节点已提交、另一部分节点失败时出现分裂。
2. 所以必须先 `PREPARE`，让每个参与者单独检查本地条件并持久化“我已准备好”。
3. 一旦所有人都准备好，协调者才能发 `COMMIT`。
4. 问题也正出在这里：为了保证后续可提交，参与者必须先锁住资源，因此阻塞成本被拉长。

玩具例子：

- 账户 A 扣 100 元
- 库存减 1
- 两边都返回 `VOTE_COMMIT`
- 协调者掉线
- 100 元和 1 件库存短时间内都被锁住

这就是 2PC 的本质代价：它用锁和统一决策换一致性。

### 2. 3PC：三阶段提交

3PC 在 2PC 基础上插入一个 `PRE-COMMIT` 阶段，想减少参与者长时间不确定的情况。白话解释是：协调者不是直接说“准备好了吗”，而是先问一次、再说一次“大家即将提交”，最后才正式提交。

它试图达到两个目标：

1. 让参与者更清楚当前离最终提交还有多远。
2. 在协调者失联时，参与者能基于超时和阶段信息做出更激进的推断。

但它没有根治网络分区问题。因为分区时，不同节点看到的超时与消息顺序可能不同，仍可能出现一边决定提交、一边决定回滚。所以 3PC 不是“彻底无阻塞”，而是“在更强网络假设下减少阻塞”。

### 3. TCC：Try-Confirm-Cancel

TCC 是业务层协议，不是数据库内核协议。它要求业务自己把一个操作拆成三步：

- Try：尝试执行并预留资源
- Confirm：正式确认
- Cancel：撤销 Try 的副作用

“预留资源”可以理解成“先冻结，但不真正消耗”。比如支付系统里冻结余额、冻结红包额度，就是典型的 TCC 思路。

真实工程例子：支付扣款。

- `Try`：冻结用户余额 100 元，冻结优惠券 1 张
- `Confirm`：真正扣减余额并核销优惠券
- `Cancel`：释放冻结余额，恢复优惠券可用状态

TCC 的关键不是协议名字，而是业务建模是否成立：

$$
\text{Try}(r) \Rightarrow \text{reserve}(r)
$$

$$
\text{Confirm}(r) \Rightarrow \text{consume reserved } r
$$

$$
\text{Cancel}(r) \Rightarrow \text{release reserved } r
$$

如果业务没有“预留”这个中间状态，比如某些外部第三方接口只支持直接扣款，不支持冻结，那么 TCC 就很难做干净。

### 4. Saga：本地事务加补偿

Saga 的思路是放弃跨节点的长事务锁，改成一连串本地事务，每一步先各自提交；如果后面某一步失败，再执行之前步骤对应的补偿动作。

设一条 Saga 为：

$$
T_1, T_2, T_3, \dots, T_n
$$

其中每个 $T_i$ 都是一个本地事务，对应补偿操作为 $C_i$。如果在 $T_k$ 失败，则回滚路径是：

$$
C_{k-1}, C_{k-2}, \dots, C_1
$$

注意不是所有补偿都等于“物理删除”或“直接回滚数据库”。补偿的白话解释是“用另一个动作，把业务状态纠正回来”。例如：

- 创建订单的补偿不是删库，而是把订单改成“已取消”
- 扣库存的补偿不是时间倒流，而是加回可售库存
- 发优惠券的补偿不是删记录，而是新增一条作废或冲正记录

真实工程例子：跨区域电商下单。

1. `T1` 订单服务创建订单
2. `T2` 库存服务冻结库存
3. `T3` 支付服务发起扣款
4. 若 `T3` 失败，则执行 `C2` 解冻库存，再执行 `C1` 取消订单

Saga 的优点是高可用、低锁占用；缺点是存在最终一致窗口，中间状态可能短暂暴露。

---

## 代码实现

下面先用一个可运行的 Python 玩具程序模拟 Saga 补偿链。它不是生产实现，但能说明“前向事务成功一部分，失败后逆序补偿”的核心机制。

```python
class SagaExecutor:
    def __init__(self):
        self.compensations = []
        self.events = []

    def run_step(self, action, compensation):
        action(self.events)
        self.compensations.append(compensation)

    def rollback(self):
        for compensation in reversed(self.compensations):
            compensation(self.events)

def create_order(events):
    events.append("order_created")

def cancel_order(events):
    events.append("order_cancelled")

def reserve_inventory(events):
    events.append("inventory_reserved")

def release_inventory(events):
    events.append("inventory_released")

def charge_payment_fail(events):
    raise RuntimeError("payment failed")

executor = SagaExecutor()

try:
    executor.run_step(create_order, cancel_order)
    executor.run_step(reserve_inventory, release_inventory)
    charge_payment_fail(executor.events)
except RuntimeError:
    executor.rollback()

assert executor.events == [
    "order_created",
    "inventory_reserved",
    "inventory_released",
    "order_cancelled",
]
```

上面这个玩具例子里，订单创建和库存预留都成功了，但支付失败，于是补偿按逆序执行：先释放库存，再取消订单。

下面给出 2PC 风格的协调者伪代码，重点看消息流，不看语言细节：

```python
class Participant:
    def __init__(self, name):
        self.name = name
        self.state = "INIT"

    def prepare(self):
        self.state = "PREPARED"
        return "VOTE_COMMIT"

    def commit(self):
        if self.state == "PREPARED":
            self.state = "COMMITTED"

    def abort(self):
        self.state = "ABORTED"

participants = [Participant("balance"), Participant("inventory")]

votes = [p.prepare() for p in participants]
if all(v == "VOTE_COMMIT" for v in votes):
    for p in participants:
        p.commit()
else:
    for p in participants:
        p.abort()

assert [p.state for p in participants] == ["COMMITTED", "COMMITTED"]
```

真实工程里，TCC 服务接口通常长这样：

```python
class PaymentTCCService:
    def __init__(self):
        self.frozen = {}
        self.paid = set()

    def try_freeze(self, tx_id, user_id, amount):
        if tx_id in self.paid:
            return
        self.frozen[tx_id] = (user_id, amount)

    def confirm(self, tx_id):
        if tx_id in self.paid:
            return
        assert tx_id in self.frozen
        self.paid.add(tx_id)
        del self.frozen[tx_id]

    def cancel(self, tx_id):
        self.frozen.pop(tx_id, None)

svc = PaymentTCCService()
svc.try_freeze("tx-1", "u1", 100)
svc.confirm("tx-1")
assert "tx-1" in svc.paid
assert "tx-1" not in svc.frozen
```

接口设计重点如下：

| 方法 | 输入 | 预期副作用 | 幂等要求 |
|---|---|---|---|
| `try_freeze` | `tx_id, user_id, amount` | 冻结资源，不正式扣减 | 重复调用不能重复冻结 |
| `confirm` | `tx_id` | 将冻结转为真实扣减 | 重复调用不能重复扣款 |
| `cancel` | `tx_id` | 释放冻结资源 | 空回滚要安全 |
| `compensateReserveInventory` | `order_id` | 释放已预留库存 | 多次执行结果一致 |

工程实现时还会再加两类表：

1. 事务表：记录全局事务号、当前阶段、超时信息
2. 幂等表：记录某个 `tx_id` 是否已执行某个动作

没有这两类状态记录，重试和恢复几乎不可控。

---

## 工程权衡与常见坑

分布式事务的难点不在“实现一次成功路径”，而在“失败路径能不能稳定收敛”。

常见故障可以先看表：

| 故障 | 根因 | 影响 | 缓解手段 |
|---|---|---|---|
| 2PC 长时间阻塞 | `PREPARED` 后协调者失联 | 锁住余额、库存，吞吐下降 | 协调者高可用、日志恢复、人工介入 |
| 重复扣款/重复减库存 | 重试但无幂等 | 数据被多次修改 | 全局事务号、去重表、状态机约束 |
| TCC 空回滚 | `Cancel` 先于 `Try` 到达或 `Try` 未落库 | 回滚报错或漏释放 | 允许空回滚，接口天然幂等 |
| TCC 悬挂 | `Try` 超时后又迟到执行 | 已取消事务再次冻结资源 | 在 `Try` 校验事务状态，拒绝过期请求 |
| Saga 补偿延迟 | 消息积压或补偿任务故障 | 短时间账实不一致 | 补偿重试、死信队列、监控告警 |
| 补偿失败 | 补偿逻辑本身依赖外部系统 | 状态长期不一致 | 补偿可重入、人工兜底流程 |

几个典型坑需要单独展开。

第一，2PC 的阻塞不是实现不好，而是机制天然存在。只要参与者已经进入 `PREPARED`，就必须守住资源直到知道全局决策。所以 2PC 不适合高延迟网络、跨地域链路、超长业务流程。

第二，TCC 最大的坑是“业务接口设计成本”。数据库帮不了你设计 `Try/Confirm/Cancel`。比如冻结余额还算自然，但“发短信”“调用第三方物流揽收”这种动作天然不可逆，TCC 就不适合。

第三，Saga 的核心风险不是失败，而是“不一致窗口”。比如订单已创建、库存已冻结，但支付因网关故障 30 秒后才失败；这 30 秒内，系统里就已经存在一个中间状态。业务必须明确：这种状态能不能让用户看到，能不能让下游继续消费。

第四，补偿不等于回滚。数据库回滚依赖日志把数据恢复到原点；Saga 补偿是在“世界已经向前走了一步”后，再追加一个动作纠偏。这也是为什么很多补偿操作必须保留审计日志，而不是直接删除记录。

---

## 替代方案与适用边界

没有一种方案能同时拿到最低延迟、最高可用、最强一致和最低业务复杂度。选型时应按场景裁剪。

| 方案 | 适用场景 | 优先考虑的系统特性 | 不适合的边界 |
|---|---|---|---|
| 2PC | 少量核心资源、同机房、参与者支持 XA 类协议 | 事务隔离和提交一致性 | 跨地域、高并发长链路 |
| 3PC | 网络稳定、节点失效模型可控的特定环境 | 减少部分阻塞 | 网络分区明显的互联网环境 |
| TCC | 余额冻结、额度占用、库存锁定 | 较强一致、可显式预留资源 | 无法预留或不可逆业务 |
| Saga | 电商下单、履约编排、跨区域流程 | 高可用、最终一致 | 无法接受中间不一致状态 |

可以按两个真实业务快速判断。

### 场景一：跨区电商下单

步骤是订单、库存、支付、优惠券、积分。链路长、跨服务多、跨区域网络不可避免，且短时间的中间状态通常可接受。这里优先考虑 Saga。

流程可以写成：

$$
T_1(\text{create order}) \rightarrow T_2(\text{reserve stock}) \rightarrow T_3(\text{charge payment})
$$

若 $T_3$ 失败，则：

$$
C_2(\text{release stock}) \rightarrow C_1(\text{cancel order})
$$

新手解释：每一步先做完再进入下一步，后面失败了，就把前面已经做过的事情按相反顺序撤销。

### 场景二：实时资金冻结

支付场景里，用户点击“确认付款”后，系统通常不能容忍“订单成功但钱状态不清楚”。如果账户系统支持冻结余额、红包系统支持冻结券额度，那么 TCC 比 Saga 更合适。

- `Try`：冻结余额、冻结红包
- `Confirm`：正式扣款、正式核销
- `Cancel`：解冻余额、恢复红包

这里的关键不是“最终一致就行”，而是“在确认前先把资源锁成可控状态”。

至于 3PC，需要特别强调边界。它常被描述成比 2PC 更先进，但在真实互联网环境里，网络分区、时钟不准、延迟抖动都很常见，3PC 的前提并不稳固。工程上更多看到的是 TCC、Saga、可靠消息加本地事务，而不是把 3PC 当主流生产方案。

一个实用的选择原则是：

1. 先问能否收敛到单库事务。
2. 再问能否改成可靠消息加最终一致。
3. 必须跨服务时，若能补偿，优先 Saga。
4. 若不能接受中间不一致，但能预留资源，选 TCC。
5. 只有参与方都支持标准分布式提交，且网络环境受控时，才认真考虑 2PC/XA 一类方案。

---

## 参考资料

- Shekhar Gulati, *Two-phase commit protocol*  
  https://shekhargulati.com/2018/09/05/two-phase-commit-protocol/  
  用于说明 2PC 的两阶段流程、阻塞问题和“协调者失联导致资源锁定”的最小例子。

- Oracle, *Two-Phase Commit Mechanism*  
  https://docs.oracle.com/html/E25494_01/ds_txns003.htm  
  用于说明 2PC 的正式机制、参与者在 prepared 状态下的约束，以及数据库层面的提交语义。

- Microsoft Learn, *Saga Design Pattern*  
  https://learn.microsoft.com/en-us/azure/architecture/reference-architectures/saga/saga  
  用于说明 Saga 的本地事务加补偿模型，以及适用于高可用、最终一致业务流程的边界。

- 阿里云开发者社区, *详解TCC分布式事务模型原理与应用实践*  
  https://developer.aliyun.com/article/1529008  
  用于说明 TCC 的 Try、Confirm、Cancel 三阶段设计，以及支付、库存等预留型业务的实现要点。

- Saket Kumar, *3-Phase Commit and the Myth of Non-Blocking Coordination*  
  https://saket2785.medium.com/3-phase-commit-and-the-myth-of-non-blocking-coordination-1bbd59c70214  
  用于说明 3PC 虽试图降低阻塞，但在网络分区等现实条件下仍存在一致性风险。
