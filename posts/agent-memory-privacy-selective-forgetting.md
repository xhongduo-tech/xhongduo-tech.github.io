## 核心结论

Agent 的长期记忆，可以先理解成一个独立于大模型参数之外的持久化记忆层：系统把过去的对话片段、工单摘要、用户偏好、操作历史等内容长期存起来，后续再通过检索拿回来参与回答。它确实能提升多轮推理和个性化能力，但也会把敏感信息一起长期保存。这里真正的风险点不是“原文文件还在不在”，而是“检索系统还能不能把旧信息重新找回来”。

结论先写清楚：

1. 只删除原始文本，不等于完成遗忘。因为 embedding，白话说就是“把文本压成一个数字向量”，以及它所在的近似索引，仍可能泄露旧内容。
2. 真正可用的方案不是单点删除，而是“存储层删除 + 查询层阻断 + 索引层清理 + 审计层留痕”一起做。
3. 在工程上，常见做法是先对待遗忘向量做差分隐私扰动，再执行近似 unlearning，最后通过 compaction 或 rebuild 逼近彻底遗忘。
4. 差分隐私，白话说就是“故意加入受控噪声，让别人很难从结果反推出原始敏感数据”。常见扰动形式是：
$$
n\sim\mathrm{Laplace}(0,\Delta/\varepsilon), \quad v' = v + n
$$
其中 $\varepsilon$ 越小，隐私保护越强，但检索失真越大。本文按题设采用 $\varepsilon=1.0$ 讨论。
5. 如果业务需要满足 GDPR 第 17 条这类“被遗忘权”，仅靠软删除通常不够。完全遗忘往往需要索引重建，其计算开销通常约为增量插入的 5 到 10 倍。

一个新手能立刻理解的玩具例子是：你把客服对话做成 embedding 存进向量库。后来用户要求删除一句敏感发言。你删掉了数据库里的原文，但没动向量索引。结果用户再次提问时，系统还是能把那句相近内容检索出来。这时“文本已删”只是表面成立，“系统仍可回忆”才是实际风险。

再补一句最容易被忽略的话：对 Agent 来说，删除不是一个数据库动作，而是一条跨系统链路。只要链路中有一层还保留了可检索痕迹，用户数据就没有真正“被忘掉”。

---

## 问题定义与边界

先定义边界。本文讨论的不是“大模型参数里学到的所有知识”，而是 Agent 自己维护的长期记忆系统，包括会话片段、用户画像、操作历史、工单摘要、偏好设置等持久化内容。只要这些内容被做成 embedding 并进入向量索引，它们就进入了可检索风险面。

隐私风险的核心不是“库里是否还有原句”，而是“系统是否还能通过相似度检索、邻居图遍历、缓存命中或派生摘要，重新拼出用户信息”。这也是为什么删除请求必须定义到以下层级：

| 数据 / 操作 | 风险 | 合规要求 |
|-------------|------|----------|
| 原始会话文本 | 直接暴露敏感内容 | 立即可删、可审计 |
| 会话 embedding | ghost vector 复现旧信息 | 立即失效、支持后续清理 |
| 向量索引邻接关系 | 删除后仍可能命中近邻 | compaction 或 rebuild |
| 派生摘要 / 用户画像 | 从旧数据二次生成 | 级联删除 |
| 查询缓存 / rerank 缓存 | 短期继续返回旧答案 | 失效处理 |
| 审计日志 | 删除过程不可追踪 | 保留删除证明，不保留敏感正文 |

这里的 ghost vector，白话说就是“原始记录删了，但系统里还留着能把它影子找回来的向量痕迹”。它常见于两类场景。

第一类是只删源数据，不删 embedding。  
第二类是 embedding 标记删除了，但 ANN 索引，白话说就是“为了快速找近邻而构建的近似搜索结构”，并没有立刻重排或重建。

为了让边界更清楚，可以把一次删除请求想成下面这条链：

用户数据  
→ 原文存储  
→ embedding  
→ 向量索引  
→ 检索结果缓存  
→ rerank 输入  
→ 摘要 / 画像  
→ 最终回答

如果只删最左边的一层，而右边的几层还在，系统照样可能把旧信息说回来。

真实工程例子更典型：一个客服 RAG 系统把用户工单、地址、退款原因、身份证后四位等内容都做成向量，存进索引后供客服助手召回。用户提出删除请求后，源表删了，但客服助手仍在后续问答里召回“曾经住在浦东新区”“最近一次退款因银行卡变更失败”这样的片段。这种系统在审计上会被认定为删除不完整，因为检索接口仍能重现个人数据。

边界也要说清：本文讨论的是“长期记忆中的选择性遗忘”，不覆盖如下问题：

| 范畴 | 是否在本文范围内 | 原因 |
|------|------------------|------|
| 模型预训练语料中的个人信息 | 否 | 需要模型级数据治理 |
| 浏览器本地缓存 | 部分相关 | 属于端侧清理问题 |
| 单次会话上下文窗口 | 否 | 不属于长期持久化记忆 |
| 外部对象存储中的附件原件 | 部分相关 | 需与对象生命周期联动 |
| 向量索引与检索服务 | 是 | 本文核心问题 |

新手常见误区也顺手列清楚：

| 误区 | 实际情况 |
|------|----------|
| “删了数据库那一行就够了” | 不够，向量、缓存、摘要都可能残留 |
| “embedding 只是数字，不算敏感信息” | 不对，embedding 可能泄露原文语义甚至被反演 |
| “软删除已经让用户看不到了，所以合规了” | 不一定，很多制度关注的是是否真正停止处理和可重建性 |
| “向量库删除和关系库删除差不多” | 不一样，ANN 索引有图结构、tombstone、异步清理等问题 |

---

## 核心机制与推导

机制要拆成两件事：一件是“先让旧向量不再精确可用”，另一件是“再让系统状态逼近没见过这条数据的样子”。

### 1. 差分隐私扰动

差分隐私噪声可以直接施加在向量上。形式是：

$$
n\sim\mathrm{Laplace}(0,\Delta/\varepsilon)
$$

其中 $\Delta$ 是敏感度，白话说就是“单条数据变化会把结果推多远”；$\varepsilon$ 是隐私预算，白话说就是“愿意用多少准确率去换隐私”。在本文设定里可用 $\Delta=1,\varepsilon=1.0$，则 Laplace 噪声尺度为 1。

扰动后的向量为：

$$
v' = v + n
$$

这一步的意义不是“数学上绝对删掉”，而是先阻断精确回溯，让旧向量即使还短暂存在，也更难被稳定命中。

如果把检索看成“比较查询向量和候选向量的夹角”，一个常见相似度就是余弦相似度：

$$
\mathrm{cos}(q,v)=\frac{q\cdot v}{\|q\|\|v\|}
$$

加入噪声后，系统实际比的是 $\mathrm{cos}(q,v')$。当 $v'$ 偏离原始方向足够多时，旧记录就更难排到前面。

玩具例子：

原始 embedding 为
$$
v=[0.6,0.8]
$$

若采样到噪声
$$
n=[-0.3,0.7]
$$

则扰动后
$$
v'=[0.3,1.5]
$$

现在假设查询向量是
$$
q=[0.59,0.81]
$$

则原始相似度约为
$$
\mathrm{cos}(q,v)\approx 0.9998
$$

扰动后相似度约为
$$
\mathrm{cos}(q,v')\approx 0.915
$$

这个数仍然不低，说明一件重要事实：差分隐私扰动不是“删干净”，而是“先降低稳定命中概率”。如果你的系统 top-k 召回阈值较高、邻近候选很多，这种下降已经足以把目标从前几名挤出去；如果系统本来就很稀疏，它可能仍会命中。这也是为什么差分隐私必须与查询阻断、索引清理配合使用。

对新手来说，可以把这一步理解成两句话：

1. 它先把“精确指纹”打模糊。
2. 它不能替代后续的结构清理。

### 2. 近似 unlearning

unlearning，白话说就是“让系统把某条训练或记忆数据当成没见过”。在长期记忆场景里，它不一定是重新训练整模型，也可以是对局部参数、检索权重、摘要缓存、用户画像进行校正。

理想目标是让修正后的系统参数 $\theta'$ 尽量接近“不包含待删数据集 $D$ 时的参数” $\theta_{-D}$：

$$
\min \|\theta' - \theta_{-D}\|
$$

问题在于，$\theta_{-D}$ 通常拿不到，因为你不会为每一条删除请求都把系统从头重训一次。所以工程上用的是近似方法，比如影响函数、局部梯度更新、派生摘要回滚、缓存失效、索引局部重排。这些做法不保证严格等价，但能逐步逼近“未见该数据”的状态。

在 Agent 记忆里，近似 unlearning 通常落在四类对象上：

| 对象 | 近似 unlearning 动作 |
|------|----------------------|
| 向量条目 | 加噪、标删、脱离召回路径 |
| 摘要 | 重新生成不含待删记录的新摘要 |
| 用户画像 | 回滚由该记忆推导出的属性 |
| rerank / cache | 失效或清空相关缓存 |

可以把它理解成“删除影响传播”。一条待删记录不只是一行文本，它还可能影响很多派生状态。unlearning 的任务就是把这些派生影响追回来。

### 3. 为什么向量删除后仍可能残留信息

很多新手以为“删记录 ID”就结束了，实际不对。以 HNSW 这类近似索引为例，它不是单纯的一张表，而是一个近邻图结构。某个节点即使被 tombstone，白话说就是“打删除标记但不立刻物理清除”，图里相关边和搜索路径仍可能受它影响。查询时系统可能绕过它，也可能在遍历中仍受旧结构干扰。长期累积后，会出现两类后果：

1. 隐私后果：旧信息通过邻近节点和缓存继续浮现。
2. 性能后果：空节点和脏图增加遍历成本，recall 和延迟一起恶化。

可以把关系数据库删除和 ANN 删除对比着看：

| 维度 | 关系数据库删一行 | ANN 索引删一个向量 |
|------|------------------|--------------------|
| 主体 | 行记录 | 图节点 + 邻接关系 |
| 影响范围 | 当前行 | 当前节点、邻边、搜索路径 |
| 是否立刻物理清除 | 取决于引擎，但语义明确 | 往往先 tombstone，再清理 |
| 查询残留风险 | 较低 | 较高 |
| 后续维护动作 | vacuum / compact | compaction / rebuild |

所以完整流程通常是：

记忆插入  
→ embedding 生成  
→ 敏感内容分级  
→ 对高敏内容加入差分隐私噪声  
→ 向量索引更新  
→ 删除请求到来后标记失效  
→ 查询层立即过滤  
→ 异步执行近似 unlearning  
→ 周期性 compaction / rebuild  
→ 写入审计证明

这个流程的核心不是“等最终重建”，而是分层止血。查询层必须先阻断返回，索引层再补做清理。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是生产级向量数据库，但能展示四个关键动作：插入时可选加噪、删除时立刻失效、异步重建前查询侧先过滤、以及派生摘要的级联清理。

```python
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vector = List[float]


def laplace_noise(scale: float) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive")
    # inverse CDF sampling
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def add_noise(embedding: Vector, delta: float = 1.0, epsilon: float = 1.0) -> Vector:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    scale = delta / epsilon
    return [x + laplace_noise(scale) for x in embedding]


def cosine(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def verify_user(user_id: str) -> bool:
    # 玩具鉴权：真实系统里要做签名校验、主体归属校验和权限校验
    return user_id.startswith("user_")


@dataclass
class MemoryItem:
    owner: str
    raw_text: str
    embedding: Vector
    deleted: bool = False
    needs_rebuild: bool = False
    sensitive: bool = False


@dataclass
class MemoryIndex:
    items: Dict[str, MemoryItem] = field(default_factory=dict)
    summaries: Dict[str, str] = field(default_factory=dict)
    audit_log: List[Tuple[str, str, str]] = field(default_factory=list)

    def insert(
        self,
        memory_id: str,
        owner: str,
        raw_text: str,
        emb: Vector,
        use_dp: bool = False,
        delta: float = 1.0,
        epsilon: float = 1.0,
    ) -> None:
        stored = add_noise(emb, delta=delta, epsilon=epsilon) if use_dp else list(emb)
        self.items[memory_id] = MemoryItem(
            owner=owner,
            raw_text=raw_text,
            embedding=stored,
            deleted=False,
            needs_rebuild=False,
            sensitive=use_dp,
        )

    def rebuild_summary(self, owner: str) -> None:
        visible_texts = [
            item.raw_text
            for item in self.items.values()
            if item.owner == owner and not item.deleted
        ]
        self.summaries[owner] = " | ".join(visible_texts) if visible_texts else ""

    def search(
        self,
        query: Vector,
        owner: Optional[str] = None,
        top_k: int = 3,
        min_score: float = -1.0,
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for memory_id, item in self.items.items():
            if item.deleted:
                continue
            if owner is not None and item.owner != owner:
                continue
            score = cosine(query, item.embedding)
            if score >= min_score:
                scored.append((memory_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def forget_vector(self, memory_id: str, user_id: str, epsilon: float = 1.0) -> None:
        if not verify_user(user_id):
            raise PermissionError("invalid user")
        if memory_id not in self.items:
            raise KeyError(memory_id)

        item = self.items[memory_id]
        owner = item.owner

        # 1. 先扰动，降低短时泄露价值
        item.embedding = add_noise(item.embedding, delta=1.0, epsilon=epsilon)

        # 2. 查询面立刻不可见
        item.deleted = True

        # 3. 标记索引后续需要清理
        item.needs_rebuild = True

        # 4. 清理派生摘要
        self.rebuild_summary(owner)

        # 5. 记录可审计但不含原文的日志
        self.audit_log.append(("forget", memory_id, user_id))

    def rebuild(self) -> None:
        # 玩具版 rebuild：真实系统里应重写 shard / segment / graph 结构
        self.items = {
            memory_id: item
            for memory_id, item in self.items.items()
            if not item.deleted
        }
        for item in self.items.values():
            item.needs_rebuild = False


def main() -> None:
    random.seed(7)

    index = MemoryIndex()

    index.insert(
        "m1",
        owner="alice",
        raw_text="用户地址是浦东新区，银行卡尾号 4321",
        emb=[0.60, 0.80],
        use_dp=False,
    )
    index.insert(
        "m2",
        owner="alice",
        raw_text="退款原因是商品尺码不合适",
        emb=[0.10, 0.20],
        use_dp=False,
    )

    index.rebuild_summary("alice")
    assert "浦东新区" in index.summaries["alice"]

    before = index.search([0.59, 0.81], owner="alice", top_k=1)
    assert before[0][0] == "m1"

    index.forget_vector("m1", "user_42", epsilon=1.0)

    after = index.search([0.59, 0.81], owner="alice", top_k=2)
    assert after[0][0] == "m2"

    assert index.items["m1"].deleted is True
    assert index.items["m1"].needs_rebuild is True
    assert "浦东新区" not in index.summaries["alice"]
    assert index.audit_log[-1] == ("forget", "m1", "user_42")

    index.rebuild()

    assert "m1" not in index.items
    print("demo passed")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，执行后输出：

```text
demo passed
```

上面代码故意把流程拆得很直接：

1. `insert` 阶段可以对高敏内容开启 `use_dp=True`。
2. `forget_vector` 不是只做一件事，而是同时做扰动、查询失效、标记重建、重建派生摘要、写审计日志。
3. `search` 明确跳过 `deleted=True` 的记录，保证删除请求一到，查询面立刻不再返回。
4. `rebuild` 负责把逻辑删除推进到结构清理，模拟“最终逼近彻底遗忘”的动作。

如果要映射到真实工程，可以按下面的伪代码理解：

```python
def forget_memory(memory_id, user_id):
    assert verify_user(user_id)
    mark_deleted(memory_id)              # 查询面立刻不可见
    rewrite_with_noise(memory_id)        # 降低短时泄露价值
    invalidate_cache(memory_id)          # 摘要、rerank、结果缓存一起清
    cascade_delete_derivatives(memory_id) # 用户画像、摘要、特征表联动
    schedule_unlearning(memory_id)       # 异步修正派生状态
    schedule_compaction(shard_of(memory_id))
    log_deletion(memory_id, user_id)
```

真实工程例子：客服 RAG 系统中，一条工单记忆可能同时存在于原始消息表、摘要表、向量索引、热门问题缓存、客户画像表。删除流程不能只删一张表，而要以 `memory_id` 或 `subject_id` 为主键，把所有派生路径一起追踪。否则系统虽然“数据库删了”，助手却还能通过摘要或画像把信息说回来。

再补一个生产化时必须加的检查表：

| 检查项 | 为什么必须做 |
|--------|--------------|
| 请求主体校验 | 防止用户删除别人的记忆 |
| 级联删除映射表 | 防止删了原文却漏删摘要 |
| 查询侧硬过滤 | 防止异步清理前继续召回 |
| 缓存失效 | 防止删后几分钟内仍返回旧答案 |
| 分片级 compaction / rebuild 队列 | 防止大量删除把系统拖垮 |
| 审计日志脱敏 | 既要留痕，又不能把敏感正文留在日志里 |

---

## 工程权衡与常见坑

工程上最难的不是“知道该删”，而是“删到什么程度算完成、代价能否承受”。

先看常见坑：

| 问题 | 原因 | 缓解方式 |
|------|------|----------|
| ghost vector | 仅删文本，不删 embedding 或索引结构 | 噪声扰动 + compaction |
| 软删除后仍命中 | 查询层没做过滤，缓存未失效 | 服务端过滤 + 缓存清理 |
| recall 下降 | 噪声过大或 tombstone 太多 | 调整 $\varepsilon$ 与阈值，定期重建 |
| rebuild storm | 多 shard 同时 rebuild | 限流、排期、分区重建 |
| 审计失败 | 没有删除凭证和链路记录 | 保留不可逆审计日志 |
| 越权删除 | 未校验请求身份 | 服务端鉴权 + 删除授权模型 |

最大的权衡是隐私预算和检索质量。

当 $\varepsilon$ 很小，噪声更大，隐私更强，但相似度检索会更不稳定。对于“高敏、低频、删除优先”的数据，比如身份证片段、地址、银行卡后缀，可以接受更大的失真。对于“高频、业务关键”的知识型记忆，比如产品手册摘要，就不适合过强噪声，否则召回质量会明显下降。

可以把这种权衡粗略写成：

$$
\text{总体成本} \approx \text{隐私风险残留} + \lambda \cdot \text{检索质量损失} + \mu \cdot \text{重建资源消耗}
$$

这里不是严格定理，只是工程视角下的决策框架。$\lambda$ 和 $\mu$ 表示你的业务有多看重效果和成本。客服、医疗、金融、法务等高敏场景，通常会把隐私风险权重调得更高。

另一个现实权衡是 rebuild 成本。索引重建通常要重新扫描和组织大量向量，CPU、内存、IO 都会被拉高。很多系统里，完全 rebuild 的成本大约是增量插入的 5 到 10 倍，这意味着你不能对每一次删除都全库重建。更现实的方案是：

1. 删除请求到来时，查询层立刻屏蔽。
2. 对目标向量做扰动，降低短期泄露风险。
3. 按 shard、时间窗或删除数量阈值触发局部 compaction。
4. 低峰期再做全量 rebuild。

这里还要提醒一个新手很容易忽视的问题：近似索引不是关系数据库。关系库里“按主键删一行”是确定动作，向量索引里“删一个节点”可能仍在图结构和近邻分布上留下影响。因此所谓“已删除”至少要拆成三个等级：

| 删除等级 | 含义 | 适用场景 |
|----------|------|----------|
| 逻辑删除 | 查询不再返回 | 需要快速响应请求 |
| 结构清理 | 索引不再保留脏节点影响 | 常规合规与性能治理 |
| 完全重建 | 尽量逼近未出现过该数据的结构状态 | 严格审计或高敏数据 |

如果系统没有这三个等级的定义，团队很容易在审计时陷入争议：产品认为“用户看不到了就算删完”，安全团队则会追问“索引里还在不在”“邻居关系是否仍受影响”“缓存和摘要是否清理”。

还有两个实操坑很常见。

第一，删的是“记忆”，不是“用户”。  
同一个用户可能有几十条记忆，删除请求有时是删单条事件，有时是删整个人的所有历史。系统设计时必须区分 `memory_id`、`session_id`、`subject_id` 三种粒度，否则一删就容易删多，或者该删的没删全。

第二，日志本身会二次泄露。  
很多团队把删除流水写得很细，结果把原始文本、旧摘要、旧 embedding 哈希一并记进日志。这样虽然“删了业务表”，却又在日志系统里重新存了一份敏感痕迹。正确做法是只记录删除动作、对象 ID、时间、审批人和结果，不保留敏感正文。

---

## 替代方案与适用边界

不是所有场景都必须全量 rebuild。要按敏感度、审计要求、资源预算来选方案。

| 方案 | 核心思路 | 优点 | 缺点 | 适用边界 |
|------|----------|------|------|----------|
| 差分隐私 + 近似 unlearning | 先扰动，再校正，再周期清理 | 响应快，平衡隐私与成本 | 不是绝对彻底遗忘 | 中高敏场景，删除请求较频繁 |
| TTL / tag 过滤 | 向量保留，但查询时只看未过期标签 | 实现简单，成本低 | 物理数据仍在，审计说服力弱 | 低敏场景，强调“不再返回” |
| 全量 rebuild | 重建索引，移除 ghost vector | 审计最强，结构最干净 | 成本高，可能引发抖动 | 高敏、强合规、低删除频率 |

TTL，白话说就是“给数据设置过期时间，到期后不再参与查询”。它很适合“业务上不要求立即物理清除，只要求结果不可见”的系统。比如短期会话偏好、推荐历史、临时任务上下文。

下面给一个简单的标签过滤可运行示例：

```python
from datetime import datetime, timedelta, timezone

records = {
    "m1": {
        "tag": "active",
        "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
    },
    "m2": {
        "tag": "deleted",
        "expires_at": datetime.now(timezone.utc),
    },
}

def visible(record, now=None):
    now = now or datetime.now(timezone.utc)
    if record["tag"] == "deleted":
        return False
    return now < record["expires_at"]

assert visible(records["m1"]) is True
assert visible(records["m2"]) is False
```

这种方案的优点是快，但边界也很明确：它只解决“查询不返回”，不解决“底层物理数据仍存在”。如果你面对的是 GDPR 删除、内部审计、事故复盘或高敏个人信息，这种方案通常只能作为第一道止血措施，不能作为最终证明。

全量 rebuild 则相反。它成本最高，但在严格场景下仍是最后保障。因为只要底层索引结构保留旧痕迹，安全和合规团队就很难给出强保证。因此，实际落地时更常见的是分层策略：

1. 低敏记忆用 TTL / tag。
2. 中敏记忆用差分隐私 + 近似 unlearning + 周期 compaction。
3. 高敏记忆触发优先级更高的 shard rebuild，必要时做全量重建。

可以进一步按数据类型粗分：

| 数据类型 | 推荐遗忘强度 |
|----------|--------------|
| 临时偏好、短期会话提示 | TTL / tag 即可 |
| 工单摘要、内部操作历史 | 逻辑删除 + 派生清理 + 周期 compaction |
| 地址、证件号、支付标识、医疗信息 | 查询阻断 + 级联清理 + 优先 rebuild |
| 法务要求明确删除的高敏记录 | 以可审计的彻底清理为目标，必要时全量或分片重建 |

这也是“选择性遗忘”的真正含义：不是所有数据都用同一种忘法，而是按风险等级选择不同的遗忘强度。

---

## 参考资料

| 资料 | 简介 | 关键贡献 |
|------|------|----------|
| [Dwork, Roth: *The Algorithmic Foundations of Differential Privacy*](https://www.microsoft.com/en-us/research/publication/algorithmic-foundations-differential-privacy/) | 差分隐私经典综述 | 给出差分隐私的正式定义、敏感度与噪声机制基础 |
| [Malkov, Yashunin: *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*](https://www.emergentmind.com/papers/1603.09320) | HNSW 原始论文 | 说明图式 ANN 索引为什么会有结构残留与维护成本 |
| [Morris et al.: *Text Embeddings Reveal (Almost) As Much As Text*](https://aclanthology.org/2023.emnlp-main.765/) | embedding 反演代表工作 | 说明 embedding 本身可能泄露大量原文信息，不能当成天然脱敏 |
| [Nguyen et al.: *A Survey of Machine Unlearning*](https://arxiv.org/abs/2209.02299) | 机器遗忘综述 | 给出 unlearning 的问题定义、方法分类和评估框架 |
| [GDPR Article 17: Right to Erasure](https://gdprinfo.eu/en-article-17) | 欧盟“被遗忘权”条文 | 说明为什么“停止返回”与“完成删除”在合规上不是一回事 |
| [Weaviate Docs: Vector Index / Tombstones](https://docs.weaviate.io/weaviate/config-refs/schema/vector-index) | 工程化向量索引文档 | 明确 tombstone 清理、周期回收和资源开销问题 |
| [FreshDiskANN](https://www.microsoft.com/en-us/research/publication/freshdiskann-a-fast-and-accurate-graph-based-ann-index-for-streaming-similarity-search/) | 动态 ANN 工程论文 | 支持“动态更新与重建成本很高，工程上需要分层维护”的判断 |

把这些资料连起来看，结论其实很稳定：Agent 长期记忆的隐私删除从来不是“删掉原文”这么简单，而是要证明系统已经失去重新找回那条信息的能力。对工程团队来说，这意味着删除语义必须覆盖文本、向量、索引结构、派生状态和审计链路，而不是只覆盖数据库里的某一行。
