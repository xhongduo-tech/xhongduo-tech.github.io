## 核心结论

HNSW 是分层可导航小世界图，白话说就是“先在稀疏的大路网上快速靠近目标，再在局部密集小路里细找”。它之所以成为 Agent 记忆检索最常用的 ANN 索引，不是因为理论最漂亮，而是因为在 `10K` 到 `100M` 量级上，通常能用可接受的内存换到很高的召回率和很低的查询延迟。

对 Agent 来说，记忆检索的目标不是“绝对最近邻”，而是“在延迟预算内，尽量别漏掉关键记忆”。因此参数调整的顺序应该固定：

1. 先确定是否真的需要 HNSW。
2. 再固定 `M` 和 `efConstruction` 作为建索引参数。
3. 最后优先调 `efSearch` 作为运行时参数。

原因很简单：`efSearch` 不需要重建索引，改起来最便宜；`M` 和 `efConstruction` 一旦设低，后面想补质量通常只能重建。

在 Agent 记忆场景里，可以先记住三条经验线：

| 数据规模 | 优先方案 | 推荐起步参数 | 目标 |
|---|---|---|---|
| `< 10K` | Flat 精确检索 | 无 | 直接拿 100% recall |
| `10K ~ 1M` | HNSW | `M=16`，`efConstruction=100`，`efSearch=64` | 先稳住 95% 左右 recall |
| `1M ~ 100M` | HNSW 或 HNSW+量化 | `M=32~48`，`efConstruction=200~256`，`efSearch=128+` | 在内存可控下冲 97%~99% recall |

如果只看一个最重要结论，那就是：`M` 决定图有多密，`efConstruction` 决定图建得有多认真，`efSearch` 决定查询时愿意多花多少步数。调参时应先把 `efSearch` 拉到能接受的上限，再判断是否值得提高 `M`。

---

## 问题定义与边界

Agent 记忆检索的“记忆”本质上是向量化后的历史片段，例如对话摘要、工具调用结果、用户偏好、长文档切片。向量检索的任务是：给定一个查询向量，从历史向量中找出最相关的前 `k` 条。这里的核心指标通常不是单一 QPS，而是以下三者同时成立：

| 指标 | 含义 | 对 Agent 的影响 |
|---|---|---|
| Recall@k | 前 `k` 个真实近邻中找回了多少 | recall 低会让回答忽然“失忆” |
| P95 延迟 | 95% 请求的查询耗时 | 延迟高会直接拖慢响应 |
| 内存占用 | 索引和向量总 RAM | 内存不够会导致扩容或换索引 |

这里要先划清边界。HNSW 解决的是“近似最近邻检索”，不是“向量质量差”问题。若 embedding 模型本身不适合你的语料，或者 chunk 切得太碎、太长、太噪声，HNSW 调得再好，也只能更快地找到“不太对”的结果。

一个玩具例子很能说明这个边界。

假设只有 8 条记忆：

- “用户喜欢 Go 语言”
- “用户讨厌周会上写长文档”
- “用户上次问过 Redis 过期策略”
- “用户的项目部署在 AWS”
- “用户更关心成本不是极致性能”
- “用户最近在学 RAG”
- “用户有 PostgreSQL 经验”
- “用户正在排查向量检索召回下降”

如果总数据量只有 8 条，最优方案根本不是 HNSW，而是 Flat。因为精确扫描 8 个向量的成本几乎为零，额外建图只是徒增复杂度。这就是第一个边界：**小数据集不要为了“高级”而上 HNSW。**

真实工程里，情况完全不同。比如一个生产 Agent 系统维护：

- 30 天对话摘要
- 工具执行日志
- 用户偏好记忆
- 文档切片记忆
- 多轮任务中间状态

总量很容易到 `200K` 甚至 `5M`。这时全量扫描每次都做 $O(N)$ 距离计算，成本会随着 $N$ 线性增长；HNSW 的价值才开始体现，因为它试图把搜索成本压到近似 $O(\log N)$ 的轨道上。

因此常见选型边界可以直接记成：

| 向量规模 | 推荐索引 | 原因 |
|---|---|---|
| `< 10K` | Flat | 简单、稳定、100% recall |
| `10K ~ 1M` | HNSW | 查询快，召回高，工程成熟 |
| `1M ~ 100M` | HNSW + 量化 / IVF | 内存开始成为硬约束 |
| `> 100M` | IVF/PQ、DiskANN 等 | 需要更强的磁盘友好或压缩能力 |

---

## 核心机制与推导

HNSW 可以理解成“多层图 + 贪心搜索”。

- 图：每个向量是一个节点，节点之间连边。
- 分层：上层节点少但覆盖广，下层节点多但更精细。
- 贪心搜索：从高层快速靠近目标，再逐层下钻。

三个参数各司其职：

| 参数 | 含义 | 白话解释 | 主要影响 |
|---|---|---|---|
| `M` | 每个节点的最大邻居数 | 每个点愿意保留多少条路 | 内存、索引质量、构建时间 |
| `efConstruction` | 建图时的候选池大小 | 建索引时看多大范围再决定连边 | 构建时间、索引质量 |
| `efSearch` | 查询时的候选池大小 | 查询时愿意多走多少步 | 延迟、CPU、召回率 |

一个足够实用的经验是：

- `efSearch >= k`
- `efConstruction` 通常至少不小于 `2 * M`
- `M` 调高通常会明显增内存，`efSearch` 调高通常主要增查询成本

内存估算可以先用一个粗公式：

$$
\text{Total Memory} \approx \text{Vector Memory} + \text{Graph Memory} + \text{Overhead}
$$

其中：

$$
\text{Vector Memory} \approx N \times d \times 4 \text{B}
$$

如果向量用 `float32`，每一维 4 字节；$N$ 是向量数，$d$ 是维度。

图内存常用近似：

$$
\text{Graph Memory} \approx N \times M \times 2 \times 8 \text{B}
$$

这里的 `2` 可以理解为双向边和层级附加结构带来的近似倍数，`8B` 是单条邻接记录的粗略字节成本。再额外留出 `10%~20%` 元数据和实现细节开销，通常就够做容量规划。

看一个具体数值。假设有 `50K` 条向量、维度 `128`、使用 `float32`：

$$
\text{Vector Memory} = 50000 \times 128 \times 4 \approx 25.6\text{MB}
$$

若 `M=32`：

$$
\text{Graph Memory} = 50000 \times 32 \times 2 \times 8 \approx 25.6\text{MB}
$$

总计大约：

$$
25.6 + 25.6 + \text{Overhead} \approx 55\text{MB} \sim 60\text{MB}
$$

若把 `M` 提到 `64`，图内存近似翻倍到 `51.2MB`，总占用会明显上升。这就是为什么 `M=64` 往往只适合“内存比延迟更宽松、且确实要逼近 0.99 recall”的场景。

公开实测里，`50K x 128` 这一规模上，参数变化通常呈现类似趋势：

| 配置 | Recall 走势 | 索引体积走势 | 适用判断 |
|---|---|---|---|
| `M=4` | 很低，常见只到低召回 | 最小 | 不适合作为 Agent 生产配置 |
| `M=16` | 可作为起点，配高 `efSearch` 可逼近高召回 | 中等 | 最常见默认档 |
| `M=32` | 明显更稳 | 偏高 | 中大规模生产更常用 |
| `M=64` | 极高召回 | 很高 | 追求极致质量时使用 |

关键推导其实只有一句：**`M` 决定上限，`efSearch` 决定你离这个上限能跑多近。** 如果 `M` 太小，图本身就不够好，后面再把 `efSearch` 加到很高，也只是“在坏图里更努力地搜索”。

---

## 代码实现

先给一个可运行的 Python 玩具脚本，用来估算不同参数下的内存。它不能替代真实 benchmark，但适合做第一轮容量规划。

```python
def estimate_hnsw_memory(num_vectors: int, dim: int, m: int, overhead_ratio: float = 0.15):
    vector_bytes = num_vectors * dim * 4
    graph_bytes = num_vectors * m * 2 * 8
    total_bytes = int((vector_bytes + graph_bytes) * (1 + overhead_ratio))
    return {
        "vector_mb": vector_bytes / (1024 * 1024),
        "graph_mb": graph_bytes / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
    }

small = estimate_hnsw_memory(num_vectors=50_000, dim=128, m=16)
large = estimate_hnsw_memory(num_vectors=50_000, dim=128, m=64)

assert small["graph_mb"] < large["graph_mb"]
assert large["graph_mb"] >= small["graph_mb"] * 3.9
assert small["total_mb"] > small["vector_mb"]

def recommend_hnsw_params(num_vectors: int):
    if num_vectors < 10_000:
        return {"index": "flat", "m": None, "ef_construction": None, "ef_search": None}
    if num_vectors < 1_000_000:
        return {"index": "hnsw", "m": 16, "ef_construction": 100, "ef_search": 64}
    if num_vectors < 10_000_000:
        return {"index": "hnsw", "m": 32, "ef_construction": 200, "ef_search": 128}
    return {"index": "hnsw_or_diskann", "m": 48, "ef_construction": 256, "ef_search": 200}

assert recommend_hnsw_params(8_000)["index"] == "flat"
assert recommend_hnsw_params(200_000)["m"] == 16
assert recommend_hnsw_params(5_000_000)["ef_search"] == 128
```

在生产里，更常见的是直接用 `pgvector`。下面是一组最小可复制片段。

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    content text NOT NULL,
    embedding vector(1536) NOT NULL
);

CREATE INDEX documents_embedding_hnsw_idx
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 100);
```

查询时不必重建索引，直接调运行时参数：

```sql
SET hnsw.ef_search = 64;

SELECT id, content
FROM documents
ORDER BY embedding <=> '[0.01, 0.02, ...]'::vector
LIMIT 10;
```

如果你的 Agent 要检索 `top 100` 记忆，就不要把 `ef_search` 设成 `50`。因为候选池比目标结果还小，搜索空间先天被截断，漏召回几乎是必然的。至少应满足：

$$
efSearch \ge k
$$

更稳妥的工程起点通常是：

$$
efSearch \approx 2k \text{ 到 } 4k
$$

真实工程例子如下。假设一个客服 Agent 维护 `200K` 条记忆：

- 用户历史工单摘要 `80K`
- FAQ 与知识片段 `70K`
- 工具调用结果缓存 `30K`
- 长任务过程记忆 `20K`

实践上可以这样起步：

| 阶段 | 参数 | 目的 |
|---|---|---|
| 初始上线 | `M=16, efConstruction=100, efSearch=64` | 控制内存，先建立基线 |
| 发现 recall 不稳 | `efSearch=96/128` | 不重建索引，先补召回 |
| 仍不够 | 重建为 `M=32, efConstruction=200` | 提高图质量 |
| 内存吃紧 | 保留 `M=16`，考虑量化或冷热分层 | 不盲目加边 |

这个流程比“一上来就把所有参数拉满”更稳，因为它把成本最大的动作，也就是重建索引，放在最后。

---

## 工程权衡与常见坑

HNSW 调参不是“越大越好”，而是三个预算的交换：内存、构建时间、查询延迟。

可以把常见配置分成三档：

| 档位 | 参数风格 | 内存 | 构建速度 | 查询延迟 | Recall |
|---|---|---|---|---|---|
| Fast prototype | `M=16, efConstruction=64~100, efSearch=32~64` | 低到中 | 快 | 低 | 中到较高 |
| Production high recall | `M=16~32, efConstruction=100~200, efSearch=64~128` | 中 | 中 | 中 | 高 |
| Max quality | `M=48~64, efConstruction=200~400, efSearch=128~256+` | 高 | 慢 | 较高 | 很高 |

常见坑主要有五类。

第一，`efSearch < k`。  
这是最直接的配置错误。你要返回 100 条，候选池只有 50，系统连“可能正确的 100 条”都没有机会看完。

第二，余弦相似度没做归一化。  
归一化就是把向量长度缩到 1，白话说是“只比较方向，不比较绝对长度”。如果数据和查询向量不在同一归一化策略下，`cosine` 的结果会失真，很多人误以为是 HNSW 召回下降，实际上是向量预处理不一致。

第三，想靠修改 `efConstruction` 修复已建好的索引。  
做不到。`efConstruction` 是建图参数，索引建完以后再改配置，不会自动让已有图变密。想吃到收益，必须重建。

第四，把 recall 问题和 rerank 问题混在一起。  
召回阶段的任务是“别漏掉对的候选”，重排阶段的任务是“把候选排对顺序”。如果前面候选没召回，后面的 reranker 再强也救不回来。

第五，忽视数据分布变化。  
Agent 记忆不是静态语料。随着时间推移，最近记忆、工具结果、摘要记忆的比例会变化，向量分布也会漂移。看到 recall 下滑时，先检查数据和 embedding 分布，再决定是否重建，不要条件反射地把 `M` 一路拉高。

一个非常实际的判断规则是：

- 若延迟还能接受，但 recall 不够：先加 `efSearch`
- 若 `efSearch` 已很高仍不够：再增 `M`
- 若 `M` 增大后内存爆炸：考虑量化、分片、冷热分层
- 若数据量其实很小：回退到 Flat，别硬上 HNSW

---

## 替代方案与适用边界

HNSW 很强，但不是所有规模、所有成本模型下的终点。

最简单的替代方案是 Flat。它的优点是结果最准、行为最稳定、没有复杂参数；缺点是查询成本线性增长。因此在单 Agent 记忆池只有 `8K` 片段时，Flat 往往就是最合理选择。

当数据继续膨胀，HNSW 的问题会变成“内存太贵”。这时常见路线有两条：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| HNSW | 高召回、低延迟、生态成熟 | 吃内存 | `10K ~ 100M` 主流场景 |
| HNSW + Quantization | 内存更省 | 会损失一部分精度 | 中大规模、内存紧张 |
| IVF/PQ | 更省内存，易分桶 | 调参更复杂，召回更依赖数据分布 | 超大规模或成本敏感 |
| DiskANN | 磁盘友好，大规模性价比好 | 部署复杂度更高 | 上亿到更大规模 |

真实工程里，一个常见分界点是 `5M` 到 `100M`。如果 Agent 记忆已经扩到数百万条，并且宿主环境不能给足 RAM，那么“继续把 HNSW 的 `M` 拉高”通常不是最优动作，更合理的是：

1. 先做冷热分层，只把热记忆放高性能索引。
2. 对冷数据做量化，减少向量存储成本。
3. 若总规模继续上升，再考虑 IVF/PQ 或 DiskANN。

例如当记忆池扩到 `250M` 条，且预算有限时，继续维持高 `M` 的纯内存 HNSW 往往不现实。此时转向量化配合 DiskANN 之类更磁盘友好的结构，通常比死扛 HNSW 更符合成本约束。换句话说，**HNSW 的边界不是“不能搜”，而是“还能不能以合理成本搜”。**

---

## 参考资料

- OneUptime, *How to Create HNSW Index*: https://oneuptime.com/blog/post/2026-01-30-vector-db-hnsw-index/view
- QueryPlane, *pgvector HNSW Tuning Guide*: https://queryplane.com/docs/blog/pgvector-hnsw-tuning-guide
- Antigravity, *Vector Index Tuning*: https://antigravity.codes/agent-skills/ai-tools/vector-index-tuning
- Apache Doris, *HNSW Vector Search Documentation*: https://doris.apache.org/docs/dev/ai/vector-search/hnsw/
