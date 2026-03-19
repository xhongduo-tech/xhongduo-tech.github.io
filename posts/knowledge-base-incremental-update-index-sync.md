## 核心结论

外部知识库要想支持持续更新，核心不是“把向量库建得更快”，而是“只处理真的变了的文档”。增量更新的标准流程是：先做文档哈希比对，再只对变更文档重新切块、重新生成 embedding，最后删除旧向量并插入新向量，同时更新哈希与版本记录。

这里的哈希，白话说就是“文档内容的指纹”；embedding，白话说就是“把文本变成模型可比较的向量”；索引同步，白话说就是“保证库里可检索到的向量和当前文档内容一致”。

如果知识库规模到百万文档，全量重建一次要约 2 小时，那么每天几十次小改动时，全量方案几乎不可用。增量方案的成本近似与变更文档数成正比，而不是与总文档数成正比。对单文档更新，常见成本是“哈希检测 <1ms + 重新 embedding 约 5s + 向量插入约 0.1s”，整体仍在秒级。

| 方案 | 处理范围 | 单次更新时间 | 触发频率适配 | 复杂度 |
|---|---:|---:|---|---|
| 全量重建 | 全部文档 | 百万文档约 2 小时 | 低频更新 | 低 |
| 增量更新 | 仅变更文档 | 单文档常见为数秒 | 高频更新 | 中 |
| 增量 + 在线索引 | 仅变更文档 | 插入可到亚秒级 | 高频且要求实时可查 | 中高 |

---

## 问题定义与边界

问题定义很具体：Agent 使用外部知识库做检索增强时，文档内容会变化，系统必须让“当前文档内容”和“当前向量索引”保持一致，否则检索结果会过时、重复或指向已删除内容。

记当前文档集合为 $D$，每篇文档 $d$ 都有新哈希 $H_{\text{new}}(d)$ 和上次已入库哈希 $H_{\text{stored}}(d)$。变更集合定义为：

$$
\Delta=\{d \mid H_{\text{new}}(d)\neq H_{\text{stored}}(d)\}
$$

这表示：只有指纹变了的文档，才进入后续处理。

边界也要说清楚。本文只讨论“单文档内容变化可被明确检测”的场景，例如 Markdown、PDF 抽取文本、FAQ 页面快照。不讨论以下复杂问题：

- 多数据源合并后才形成一篇知识条目
- 文档结构变化导致 chunk 边界整体重排
- 需要跨系统事务一致性的强同步
- 需要按权限做多租户隔离的复杂查询

玩具例子：知识库里有 1000 篇文档，今天只改了 3 篇。如果仍然全量重建，997 篇没变的文档也会重复切块、重复 embedding，纯属浪费。增量更新只处理那 3 篇，成本直接缩小到原来的千分之几量级。

真实工程例子：一个客服 Agent 每天接收用户反馈修正文档 50 次。如果每次全量重建要 2 小时，系统一天根本跑不完；如果每次只处理变更文档，并在线写入 Milvus，新知识可以在秒级进入检索结果。

---

## 核心机制与推导

增量更新通常是一个五步流水线：

1. 扫描文档并计算哈希
2. 找出变更集合 $\Delta$
3. 对每个 $d\in\Delta$ 执行清洗、切块、embedding
4. 删除旧向量或标记 tombstone
5. 插入新向量，并更新哈希与版本记录

这里 tombstone，白话说就是“先标记这条数据作废，但不立刻物理删除”。它常用于在线系统，因为立即重写大索引代价高。

embedding 过程可写成：

$$
v_d=f_{\text{embed}}(d)
$$

更准确地说，真实系统不是为整篇文档只生成一个向量，而是先把文档切成多个 chunk，再对每个 chunk 生成向量集合 $\{v_{d,1}, v_{d,2}, \dots\}$。这样做的原因是长文档直接压成一个向量，检索粒度太粗。

伪代码如下：

```text
for each document d:
    new_hash = hash(d)
    if new_hash != stored_hash[d]:
        old_ids = get_chunk_ids(d)
        soft_delete(old_ids)
        chunks = split(clean(d))
        vectors = embed(chunks)
        insert(vectors, metadata={doc_id, version+1})
        stored_hash[d] = new_hash
```

为什么这个流程快？因为总耗时变成：

$$
T_{\text{incremental}} \approx T_{\text{hash-scan}} + \sum_{d\in\Delta}(T_{\text{chunk}}+T_{\text{embed}}+T_{\text{upsert}})
$$

而不是：

$$
T_{\text{full}} \approx \sum_{d\in D}(T_{\text{chunk}}+T_{\text{embed}}+T_{\text{index-build}})
$$

只要 $|\Delta| \ll |D|$，两者差距就会非常大。对单文档更新，哈希扫描几乎可以忽略，主要开销在重新 embedding；Milvus 一类在线向量库支持动态插入，因此新向量可以很快进入可检索状态。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖真实 Milvus，但把核心控制流完整保留了：哈希检测、旧向量软删除、新向量插入、版本更新。soft delete，白话说就是“查询时把旧数据过滤掉”。

```python
import hashlib
from dataclasses import dataclass, field

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def split_chunks(text: str, chunk_size: int = 12):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed(chunk: str):
    # 玩具 embedding：把字符长度和字符码求和拼成向量
    return [len(chunk), sum(ord(c) for c in chunk) % 997]

@dataclass
class InMemoryVectorStore:
    rows: list = field(default_factory=list)

    def soft_delete_by_doc(self, doc_id: str):
        for row in self.rows:
            if row["doc_id"] == doc_id:
                row["deleted"] = True

    def insert_chunks(self, doc_id: str, version: int, chunks):
        for idx, chunk in enumerate(chunks):
            self.rows.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}:{version}:{idx}",
                "version": version,
                "text": chunk,
                "vector": embed(chunk),
                "deleted": False,
            })

    def search_visible(self, doc_id: str):
        return [r for r in self.rows if r["doc_id"] == doc_id and not r["deleted"]]

class IncrementalIndexer:
    def __init__(self):
        self.hash_state = {}   # doc_id -> hash
        self.version_state = {}  # doc_id -> version
        self.store = InMemoryVectorStore()

    def update_document(self, doc_id: str, text: str):
        new_hash = sha256_text(text)
        old_hash = self.hash_state.get(doc_id)

        if new_hash == old_hash:
            return "skip"

        old_version = self.version_state.get(doc_id, 0)
        new_version = old_version + 1

        self.store.soft_delete_by_doc(doc_id)
        chunks = split_chunks(text)
        self.store.insert_chunks(doc_id, new_version, chunks)

        self.hash_state[doc_id] = new_hash
        self.version_state[doc_id] = new_version
        return "updated"

idx = IncrementalIndexer()

doc_v1 = "Milvus supports online vector insertion."
doc_v2 = "Milvus supports online vector insertion and soft delete."

assert idx.update_document("doc-1", doc_v1) == "updated"
visible_v1 = idx.store.search_visible("doc-1")
assert len(visible_v1) >= 1
assert all(row["version"] == 1 for row in visible_v1)

assert idx.update_document("doc-1", doc_v1) == "skip"

assert idx.update_document("doc-1", doc_v2) == "updated"
visible_v2 = idx.store.search_visible("doc-1")
assert len(visible_v2) >= 1
assert all(row["version"] == 2 for row in visible_v2)

deleted_rows = [r for r in idx.store.rows if r["doc_id"] == "doc-1" and r["deleted"]]
assert len(deleted_rows) >= 1
```

真实工程里，把上面的 `InMemoryVectorStore` 换成 Milvus 调用即可。典型实现要点有三项：

- 文档元数据表保存 `doc_id -> hash -> version -> updated_at`
- 向量元数据保存 `doc_id、chunk_id、version、deleted`
- 查询时增加过滤条件，只返回 `deleted=false` 的最新记录

如果接 Milvus，工程逻辑通常不是“先重建整个索引”，而是“删除旧 chunk 对应向量，再插入新 chunk 向量”。在线索引能让新增内容较快可查，但你仍要自己保证版本控制是对的，否则向量库不会替你判断哪条是旧数据。

---

## 工程权衡与常见坑

增量更新省的是计算和重建时间，换来的是状态管理复杂度。系统里一旦多了“哈希表、版本号、软删除标记、异步插入队列”，错误通常不是出在 embedding 模型，而是出在同步协议。

最常见的坑如下：

| 坑点 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 重复向量 | 同一文档搜索出两份相似结果 | 插入新向量前没删旧向量 | 先软删除旧版本，再插入新版本 |
| 无限重建 | 每次扫描都触发 re-embed | 新哈希没回写状态表 | 更新成功后原子写入 hash/version |
| 查到已删除内容 | 文档已下线但仍被召回 | 查询没过滤 tombstone | 检索条件显式加 `deleted=false` |
| 版本错乱 | 新旧 chunk 混在一起 | 一个文档部分更新成功、部分失败 | 以文档版本为单位提交，失败可回滚 |
| chunk 漂移 | 小改动导致大量 chunk 全变 | 固定切块边界设计差 | 用稳定切块策略，减少无效重算 |

一个新手容易忽略的问题是“旧向量删除顺序”。如果你先插入新向量、后删旧向量，在高并发查询窗口里，用户可能短时间看到双份结果。如果业务对一致性要求更高，应把写入流程做成带版本门控的原子切换。

另一个坑是“哈希算的不是最终可检索文本”。例如原始 HTML 里广告时间戳每天都变，但正文没变；如果你对原始页面直接算哈希，就会每天误判为文档变更。更稳妥的做法是：先清洗出业务正文，再对清洗后的规范化文本算哈希。

---

## 替代方案与适用边界

不是所有项目都值得上增量同步。小系统先用全量重建，往往更稳。

| 条件 | 全量重建更合适 | 增量更新更合适 |
|---|---|---|
| 文档规模 | 几百到几千篇 | 十万到百万篇以上 |
| 更新频率 | 每天 1 次或更低 | 每小时多次、持续更新 |
| 可接受延迟 | 小时级 | 分钟级到秒级 |
| 工程人力 | 少，优先简单 | 有能力维护状态与索引协议 |
| 一致性要求 | 可容忍夜间批处理 | 希望新知识快速可查 |

对比示例：

- 小团队内部知识库只有 200 篇文档，每天凌晨 2 点跑一次全量重建，逻辑简单、运维负担低，这就是合理方案。
- 大机构的售后知识库有几百万 chunk，白天持续接收人工修订、产品公告和工单总结，这时必须用“哈希检测 + 差量 embedding + 在线向量插入”的增量链路，否则更新时间会压垮系统。

还有一个常见替代方案是“定时微批处理”。它不是每来一条改动就立刻更新，而是每 5 分钟聚合一次变更再批量写入。这样能降低写放大，适合对实时性要求没那么高、但又不想全量重建的系统。

---

## 参考资料

- ShShell，《Incremental Updates and Re-Indexing》，2026-01-05。支持点：给出哈希检测、增量 ingest、版本记录、soft delete 的基本流程，适合用来定义本文的问题边界与最小实现。
- Milvus Blog，《How does vector search handle real-time updates?》，访问日期 2026-03-19。支持点：解释向量搜索系统如何做动态更新、缓冲写入、后台合并与 tombstone，支撑“在线索引可持续插入”的机制描述。
- Zilliz on Medium，《Clearing Up Misconceptions about Data Insertion Speed in Milvus》，2024-06-21。支持点：给出 embedding 大约 5 秒、插入约 0.1 秒的实验量级，说明瓶颈通常不在向量库写入，而在向量生成。
- Milvus/Zilliz 相关 Quick Reference 与产品文档，访问日期 2026-03-19。支持点：补充动态索引、近实时可见性与插入后查询一致性的工程背景，适合对接实际系统时查 API 细节。
