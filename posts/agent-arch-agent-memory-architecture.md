## 核心结论

Agent 的记忆架构，本质上是在有限上下文窗口内，给智能体建立一套“该记什么、记多久、谁能读、什么时候写回”的规则。它不是单独一个数据库，也不是简单把历史对话全塞回提示词，而是把**短期记忆**和**长期记忆**分层管理，再配合调度器或多 Agent 协作策略，让系统在低延迟下拿到足够相关的上下文。

对零基础读者，先用一句白话解释两个术语：

- **短期记忆**：就是当前任务现场的工作草稿，放的是“这一步马上要用”的内容。
- **长期记忆**：就是跨会话保留的经验档案，放的是“以后大概率还会用”的结论、偏好和历史事实。

一个直观类比是笔记系统。短期记忆像你今天桌上的草稿纸，写满当前题目的中间步骤；长期记忆像文件柜里的总结文档，存放已经确认的定义、结论和经验；调度器像项目经理，决定哪些笔记要共享给其他人，哪些只留在自己桌上，避免所有人都反复抄同一段内容。

| 项 | 短期记忆 | 长期记忆 |
|---|---|---|
| 作用 | 当前会话上下文 | 跨会话偏好、经验、事实 |
| 典型容量 | 5k-40k Token | 40k-200k Token 或外部存储 |
| 更新频率 | 每步都可能更新 | 事件触发或策略触发 |
| 访问方式 | 直接拼接到上下文 | 检索、摘要、召回 |
| 共享策略 | 通常高度可见 | 可隔离、可摘要共享 |
| 失败代价 | 当前回答跑偏 | 长期持续重复犯错 |

真正有效的记忆架构有两个核心收益：

1. 它让 Agent 不必在每一步都重新“认识世界”。
2. 它让系统不必把全部历史原样带上，而是只带最相关的部分。

---

## 问题定义与边界

从形式化角度看，Agent 记忆架构要解决的问题是：在环境状态不可完全直接观测时，如何用有限历史维护一个足够好的内部状态表示，用来支持后续决策。

这里的 **POMDP**，白话说就是“智能体看不见完整世界，只能看到局部线索，再根据线索猜当前真实状态”。因此，记忆不是简单存文本，而是在维护一个对当前状态的估计，也就是**信念状态**：

$$
b_t(s)=P(s_t=s \mid a_{0:t-1}, o_{1:t})
$$

这表示：在时刻 $t$，系统根据过去做过的动作 $a_{0:t-1}$ 和收到的观测 $o_{1:t}$，认为当前真实状态是 $s$ 的概率。

有了这个分布，动作选择才有依据：

$$
a_t=\arg\max_a \mathbb{E}_{s\sim b_t}[Q(s,a)]
$$

白话解释是：系统不要求百分之百知道当前状态，而是在“当前最可能的状态分布”上，选择期望收益最高的动作。

所以，记忆架构的目标不是“记得越多越好”，而是：

- 让 $b_t$ 更新得足够快
- 让决策所需信息尽量相关
- 避开上下文窗口和成本限制
- 在多 Agent 下维持可接受的一致性

边界也很明确。Agent 记忆架构通常不直接解决以下问题，而是与它们强耦合：

| 边界问题 | 为什么相关 | 但不等于记忆本身 |
|---|---|---|
| Token 预算 | 能放进上下文的内容有限 | 预算控制还涉及模型定价与调度 |
| 多 Agent 一致性 | 多个 Agent 可能读到不同版本 | 一致性需要协议、锁或版本管理 |
| 权限隔离 | 有些信息不能全局共享 | 这属于访问控制设计 |
| Artifact 互通 | 外部文档需要跨 Agent 共享 | 这属于存储与引用机制 |

一个最小多 Agent 例子可以说明边界。设总上下文窗口上限是 200k Token，但为了降低费用和避免截断，系统只允许本轮实际注入 85k：

- Lead Agent 保留 40k 的全局摘要
- 3 个子 Agent 各自持有 10k 工作窗口
- Artifact 临时引入 5k 新内容

总量为：

$$
40k + 10k \times 3 + 5k = 75k
$$

如果再预留 10k 给系统提示词、工具返回和安全冗余，总预算仍控制在 85k 内。这里的关键不是“窗口有 200k 就用满 200k”，而是**把昂贵上下文留给当前决策真正需要的信息**。

---

## 核心机制与推导

记忆更新的本质，是每来一次新观测，就用它修正上一时刻的信念状态。抽象地看，它分两步：

1. 用状态转移模型，预测系统可能来到哪些状态。
2. 用新观测，修正这些状态的概率。

写成常见形式是：

$$
b_t(s') \propto P(o_t \mid s') \sum_s P(s' \mid s, a_{t-1}) b_{t-1}(s)
$$

白话解释：

- $b_{t-1}(s)$：上一轮你对世界状态的判断
- $P(s' \mid s, a_{t-1})$：你执行动作后，世界可能怎么变化
- $P(o_t \mid s')$：如果世界真处于 $s'$，这次观测出现的可能性有多大

新的记忆，不是把旧文本机械追加，而是把“过去判断”与“新证据”合并，得到新的内部状态。

这也是为什么纯聊天历史不等于记忆架构。聊天历史只是原始材料；真正的记忆架构会做筛选、压缩、打分、同步和遗忘。

可以用一个打牌的玩具例子理解：

- 每轮你看到别人出了一张牌，这就是 **observation**，白话说就是“新线索”。
- 你根据之前出牌历史，推测对方手里可能还有什么，这就是 **belief**，白话说就是“你脑子里的当前判断”。
- 你再决定这轮出哪张牌，这就是 **action**，白话说就是“下一步动作”。

如果四个玩家都各记各的细节，会有重复工作。于是引入一个 Leader，负责记录“前几轮谁已经暴露出什么风格”的摘要，其他玩家只保留本轮局部推理。这样可以减少重复记忆成本。

多 Agent 系统里，层级记忆通常按下面路径工作：

观察 → belief 更新 → action 选择 → 短期记忆写入 → 长期记忆筛选/摘要 → 共享或隔离同步

其中最容易被忽略的是“同步路径”。如果没有明确同步规则，会出现两类典型故障：

- 子 Agent 各自形成局部真相，但没人汇总，全局决策失真
- 所有人都写全局记忆，导致冲突、覆盖和噪声累积

因此常见做法是：

- Lead Agent 持有全局摘要，负责跨任务连续性
- 子 Agent 持有局部工作视图，负责高频、细粒度推理
- Artifact 或外部存储保存中间成果，避免直接塞满上下文

真实工程里，这比“统一共享一个长提示词”更稳定，因为不同层级承担不同职责：局部窗口追求精度，全局摘要追求压缩率。

---

## 代码实现

工程上，记忆系统至少要回答四个问题：

1. 这条信息属于哪个 scope，也就是作用域，白话说就是“它归谁管、谁能看”。
2. 这条信息重要吗。
3. 这条信息新吗。
4. 这条信息和当前问题语义相近吗。

下面先给一个可运行的 Python 玩具实现。它不依赖向量库，只用简单权重说明检索思路，但结构与真实系统一致。

```python
from dataclasses import dataclass, field

@dataclass
class Entry:
    text: str
    importance: float
    freshness: float
    semantic: float
    version: str = "v1.0"
    shared: bool = True

    def score(self):
        return 0.5 * self.importance + 0.3 * self.freshness + 0.2 * self.semantic

class Memory:
    def __init__(self):
        self.store = {}

    def write(self, scope, entry: Entry):
        self.store.setdefault(scope, []).append(entry)

    def read(self, scope, topk=3, include_private=False):
        entries = list(self.store.get(scope, []))
        if include_private:
            entries += self.store.get(f"{scope}:private", [])
        entries = sorted(entries, key=lambda x: x.score(), reverse=True)
        return entries[:topk]

    def decay(self, factor=0.9):
        for entries in self.store.values():
            for e in entries:
                e.freshness *= factor

mem = Memory()
mem.write("lead", Entry("用户偏好中文回答", importance=0.9, freshness=0.8, semantic=0.6))
mem.write("lead", Entry("旧版本接口已废弃", importance=0.7, freshness=0.4, semantic=0.9))
mem.write("lead:private", Entry("仅领导可见的风险备注", importance=0.8, freshness=0.9, semantic=0.3, shared=False))

public_items = mem.read("lead")
all_items = mem.read("lead", include_private=True)

assert len(public_items) == 2
assert len(all_items) == 3
assert public_items[0].text == "用户偏好中文回答"

old_freshness = public_items[0].freshness
mem.decay(0.5)
assert mem.read("lead")[0].freshness <= old_freshness
```

这个玩具例子体现了三件事：

- `scope` 控制可见范围
- `importance/freshness/semantic` 共同决定召回顺序
- `decay` 用来实现遗忘，避免旧信息长期霸占上下文

如果换成更接近生产环境的接口，JavaScript 风格大致如下：

```js
class Memory {
  constructor(scopeTree, index, artifactStore) {
    this.scopeTree = scopeTree;
    this.index = index;
    this.artifactStore = artifactStore;
  }

  read(scope, query) {
    return this.index.query({
      scope,
      query,
      weight: ["importance", "freshness", "semantic"],
      topK: 8
    });
  }

  write(scope, entry, share = true) {
    const targetScope = share ? scope : `${scope}:private`;
    this.index.add(targetScope, {
      ...entry,
      version: entry.version || "v1.0",
      ts: Date.now()
    });
  }

  pushArtifact(scope, entry) {
    this.artifactStore.push({ scope, entry });
  }

  pullArtifacts(scope) {
    return this.artifactStore.pull(scope);
  }
}
```

这里的 **Artifact**，白话说就是“放在外部仓库里的中间产物”，例如：

- 子 Agent 抽取的网页摘要
- 代码扫描结果
- 用户上传文件的结构化解析
- 任务中间计划

真实工程例子可以这样看。一个研究型多 Agent 系统里：

- Lead Agent 负责和用户持续对话，只保留任务摘要、用户偏好、当前目标
- Research Agent 负责搜集资料，把长网页内容写入 Artifact
- Coding Agent 负责代码修改，只在本地工作窗口保留 API 细节和错误日志
- Lead Agent 不直接读完所有原始资料，而是拉取摘要和高权重结论

这样做的收益非常直接：

- 上下文不会被原始材料淹没
- 子 Agent 的局部噪声不会污染全局记忆
- 需要追溯时，可以回 Artifact，而不是把所有历史反复塞进提示词

---

## 工程权衡与常见坑

记忆架构最难的部分，不是“怎么存”，而是“怎么防错”。下面是常见问题与规避方式。

- 共享记忆没有版本号。结果是 Agent A 读到旧结论，Agent B 已经写入新结论，最终系统在两个版本之间来回震荡。
- 长短期记忆都直接注入上下文。结果是相关信息占比下降，模型把注意力浪费在过时内容上。
- 所有 Agent 默认共享全部内容。结果是局部推理草稿污染全局，甚至暴露本不该共享的中间状态。
- 没有遗忘机制。结果是“历史上重要过一次”的内容长期霸占检索结果。
- 只有语义相似度，没有重要性和新鲜度。结果是系统总能召回“看起来像”，但不一定“现在该用”的信息。
- 多 Agent 没有显式同步协议。结果是重复工作，或者互相覆盖状态。

一个常见规避方案，是给共享记忆加上版本标签，例如 `v3.1`，同时对每条记忆维护时间衰减函数：

$$
score = \alpha \cdot importance + \beta \cdot semantic + \gamma \cdot decay(age)
$$

其中可以取：

$$
decay(age)=e^{-\lambda \cdot age}
$$

白话解释是：时间越久，分数自动下降，除非它本身足够重要。

“问题 -> 规避”可以整理成表：

| 问题 | 典型后果 | 规避方式 |
|---|---|---|
| 共享内容过期 | 读到 stale 信息，决策失真 | 版本号、更新时间、写入人标记 |
| 记忆无限累积 | Token 爆炸、检索噪声增加 | 摘要压缩、归档、衰减 |
| 权限边界不清 | 隐私泄露、局部草稿污染全局 | scope 树、私有视图、角色隔离 |
| 检索只看相似度 | 召回“像”的内容，不是“该用”的内容 | importance + freshness + semantic 加权 |
| 多 Agent 并发写入 | 覆盖、冲突、重复劳动 | Lead 汇总、Artifact 中转、写入协议 |

还有一个很容易忽略的坑：**把摘要当作事实原文**。摘要适合决策压缩，不适合高精度追责。只要任务涉及精确代码、合同条款、数值表格，就必须能回源读取原文，否则摘要误差会在多轮传递中放大。

---

## 替代方案与适用边界

没有一种记忆架构适合所有系统。常见方案大致有三类。

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 完全隔离 | 高隐私、高独立性任务 | 简单、冲突少、权限清晰 | 重复工作多，难形成全局连续性 |
| 完全共享 | 统一知识库、强协作任务 | 信息同步快，整体视图一致 | 噪声大，容易冲突，权限难控 |
| Lead + Artifact | Token 紧张但需要协作 | 全局摘要稳定，原始材料外置，性价比高 | 系统更复杂，需要同步协议 |

可以再具体解释它们的适用边界。

**完全隔离**适合什么场景？  
例如金融合规审查，不同 Agent 分别处理不同客户材料，彼此不能读取对方上下文。这时宁可牺牲协作效率，也要确保边界清楚。

**完全共享**适合什么场景？  
例如一个小型内部知识助手，所有 Agent 都服务同一个文档库，目标是统一答案而不是角色隔离。这时共享带来的收益大于风险。

**Lead + Artifact**适合什么场景？  
这是当前更实用的折中方案。尤其在多角色协作、上下文成本敏感、任务链较长时，它比“全共享大上下文”更稳。Lead 负责压缩与协调，Artifact 保留可追溯原文，子 Agent 负责局部推理。

如果角色层级非常明确，还可以引入 **scope 树**。白话说，就是把记忆按组织结构分层，例如：

- `org`
- `org/project`
- `org/project/lead`
- `org/project/agent-a:private`

这样做的优点是共享边界天然清楚；缺点是设计和维护成本更高，尤其在角色动态创建和销毁时，需要额外治理。

所以选型时不要问“哪种最先进”，而要问三件事：

1. 任务是否需要跨会话连续性。
2. 多个 Agent 是否需要共享同一事实源。
3. 错误成本更怕哪种，是漏信息，还是错共享。

---

## 参考资料

1. Stephen J. Bigelow 等，*What Is AI Agent Memory? Types, Tradeoffs and Implementation*，TechTarget，2025-10-03。核心贡献：从企业应用角度解释 Agent 记忆的类型、实现方式与工程权衡。  
2. Zylos Research，*AI Agent Memory Architectures for Multi-Agent Systems*，2026-03-09。核心贡献：讨论多 Agent 下共享、隔离、层级记忆与 token 管理策略。  
3. Zhongming Yu, Jishen Zhao，*Multi-Agent Memory from a Computer Architecture Perspective: Visions and Challenges Ahead*，SIGARCH，2026-01-20。核心贡献：从系统架构视角讨论多 Agent 记忆的一致性、扩展性与资源分配问题。
