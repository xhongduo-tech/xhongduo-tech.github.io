## 核心结论

实时推荐架构的核心不是“把模型搬到线上”，而是把**离线特征**和**实时特征**按同一套定义、同一套时间约束、同一套服务接口组合起来，在 `<100ms` 的推理预算内返回可直接喂给模型的特征向量。

离线特征，白话说，就是按小时或按天从历史数据里批量算出的“相对完整但不够新”的信息，例如用户最近 7 天点击率、商品近 30 天转化率。实时特征，白话说，就是请求发生时刚产生或刚更新的“最新但通常更贵”的信息，例如用户刚刚浏览的类目、当前库存、最近 5 分钟点击次数。

真正决定线上效果的，不是是否用了 Redis、Flink 或某个特征平台，而是两件事：

1. 训练和推理是否使用**一致的特征定义**。
2. 训练样本是否满足**点位时间正确**，也就是只能看到标签时刻之前本来就存在的数据，不能偷看未来。

如果这两件事做错，模型离线评估再好，上线也会出现训练/推理偏差。很多团队看到的现象是：离线 AUC 很高，线上 CTR 却掉得很明显，本质通常不是模型退化，而是特征在两条链路上“长得不一样”。

可以把服务时刻 $T$ 的输入写成：

$$
f_{serving}(T)=concat(f_{offline}(T-\Delta),\ f_{online}(T))
$$

其中 $f_{offline}(T-\Delta)$ 表示在标签时间之前构造、满足点位时间约束的离线特征，$f_{online}(T)$ 表示当前请求时刻的实时特征。实时推荐架构的目标，就是稳定地构造这个向量，并在低延迟预算内交给模型。

---

## 问题定义与边界

推荐系统里的“实时”不是单指模型推理快，而是指从“用户发起请求”到“返回推荐结果”的整条链路足够快，通常要求几十毫秒到一两百毫秒。若页面要求流畅，很多核心链路会把目标压到 `50ms-100ms`。

这时系统面临两个同时成立的约束：

| 维度 | 离线特征 | 实时特征 |
| --- | --- | --- |
| 计算频率 | 小时级或天级批处理 | 秒级流处理或请求时刷新 |
| 数据新鲜度 | 较旧但覆盖完整 | 最新但计算和读取更贵 |
| 常见存储 | 数据仓库、Parquet、湖仓 | Redis、DynamoDB 等低延迟 KV |
| 主要用途 | 训练、回溯、评估 | 在线推理、会话级决策 |
| 风险重点 | 点位时间错误、未来信息泄露 | 延迟超标、缓存失效、热点击穿 |

问题边界也要说清楚。实时推荐架构主要解决的是“**特征到模型输入**”这段问题，不直接等于召回、排序、重排、AB 实验、向量检索的全部架构。它关注的是：

- 如何定义特征。
- 如何在训练时正确构造历史特征。
- 如何在推理时快速取到当前特征。
- 如何保证两边一致。

一个简化的延迟预算公式是：

$$
latency_{network}+latency_{feature}+latency_{model} \le SLA_{total}
$$

如果总 SLA 是 `100ms`，模型推理占 `20ms`，网络与业务编排占 `30ms`，那么特征服务常常只能拿到 `10ms-15ms` 的 p99 预算，剩下的时间还要给序列化、重试保护和尾延迟冗余。

玩具例子可以说明这个约束。假设一个小型推荐页只需要 6 个特征：

- 用户最近 7 天点击数
- 用户最近 7 天购买数
- 商品最近 1 天曝光数
- 商品最近 1 天点击数
- 当前会话是否刚浏览过该类目
- 当前商品是否有库存

前 4 个更适合离线或准实时预计算，后 2 个必须在请求时获取。如果系统把这 6 个特征拆成 6 次远程读取，哪怕每次只要 `3ms`，累计也会明显侵蚀预算；如果用统一服务一次 `multi-get` 批量取回，整体延迟才可能稳定。

真实工程里，这个问题会放大得多。比如电商首页排序一次请求要读取 50 个以上用户、商品、上下文特征，QPS 到 `20k` 时，相当于每秒上百万次特征读。如果没有统一特征服务，业务代码里会散落着数据库查询、缓存拼接、兜底逻辑、字段兼容分支，最终难以维护，也难以验证训练和推理是否一致。

---

## 核心机制与推导

实时推荐架构通常拆成四层：**特征定义层、离线构造层、在线物化层、推理读取层**。

特征定义层，白话说，就是“先把字段和计算规则说清楚”。例如：

- `user_ctr_7d = 用户过去7天点击数 / 过去7天曝光数`
- `item_cvr_30d = 商品过去30天购买数 / 点击数`
- `session_last_category = 当前会话最近一次浏览类目`

这一步最关键。因为一旦定义分裂，离线训练和在线推理就会各算各的。一个团队最容易犯的错是：训练脚本里用 SQL 算 `ctr_7d`，线上服务里由 Java 或 Python 重新实现一次，字段过滤条件、时间窗口、缺失值填充只要有一点不同，就会产生偏差。

接着是离线构造层。这里强调**点位时间**。点位时间，白话说，就是“在历史某个时刻回头看，当时你理论上能看到哪些信息”。如果标签是“用户在 `2026-03-01 10:00` 是否点击了推荐结果”，那么用于构造样本的特征只能使用 `10:00` 之前已经存在的数据。

形式化地说，若标签时间为 $T$，训练时可见的特征必须满足：

$$
feature\_time \le T
$$

更严格地，若数据存在采集延迟或落库延迟，还应满足：

$$
availability\_time \le T
$$

这比“事件发生时间早于 T”更严格，因为有些日志虽然事件时间早，但在真实系统里直到更晚才可见。训练时如果错误地用了这些“未来才到达”的特征，就出现**特征穿越**。特征穿越，白话说，就是训练样本偷偷看到了线上当时不可能拿到的信息。

再看在线物化层。物化，白话说，就是“提前把算好的结果落到可快速读取的存储里”。例如：

- 批处理每天把用户长期统计特征写入在线 KV。
- 流处理每秒更新近实时计数。
- 请求到来时，再补充上下文特征。

此时服务时刻的最终向量可写成：

$$
f_{serving}(T)=concat(f_{offline}(T-\Delta), f_{online}(T))
$$

这里的 $\Delta$ 不是固定常数，而是离线特征相对当前请求的滞后量。它可能是 1 小时，也可能是 1 天。关键不在于绝对实时，而在于系统明确知道“哪些特征是滞后的、哪些特征是当前的”，并且训练时按同样的语义构造样本。

玩具例子如下。假设要预测“用户此刻是否会点击商品 A”：

- `user_clicks_7d = 20`
- `item_ctr_1d = 0.08`
- `current_session_same_category_views = 3`

如果训练集里的第三个特征是“本次点击之后该会话总共浏览了几次同类商品”，那就是错误特征，因为它包含了未来行为。线上请求时根本取不到这个值。离线看起来相关性很强，上线后却失效。

真实工程例子更典型。电商或内容平台常见做法是：

- Hadoop/Spark 生成长周期用户和商品画像。
- Kafka/Flink 维护分钟级点击、曝光、库存、价格变化。
- Redis 保存在线可直接读取的实体特征。
- 排序服务在一次请求内批量读取用户特征、候选商品特征和上下文特征。
- 模型服务接收拼接后的向量，输出排序分数。

这一架构的本质不是组件名字，而是“**同一份特征定义，分别生成离线训练数据和在线服务数据**”。只要定义统一，离线训练和在线推理才有可比性。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，模拟统一注册、点位时间读取和在线 `multi-get` 回退。代码不依赖第三方库，重点是机制而不是框架。

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FeatureRecord:
    value: float
    event_time: int       # 事件时间
    available_time: int   # 真正可用时间


class FeatureRegistry:
    def __init__(self):
        self.feature_order = [
            "user_ctr_7d",
            "item_ctr_1d",
            "session_same_category_views",
        ]

    def keys_for(self, user_id: str, item_id: str) -> Dict[str, str]:
        return {
            "user_ctr_7d": f"user:{user_id}:ctr_7d",
            "item_ctr_1d": f"item:{item_id}:ctr_1d",
            "session_same_category_views": f"session:{user_id}:same_category_views",
        }


class OfflineStore:
    def __init__(self, history: Dict[str, List[FeatureRecord]]):
        self.history = history

    def read_point_in_time(self, key: str, as_of: int) -> float:
        candidates = [
            r for r in self.history.get(key, [])
            if r.event_time <= as_of and r.available_time <= as_of
        ]
        if not candidates:
            return 0.0
        return sorted(candidates, key=lambda x: x.event_time)[-1].value


class OnlineCache:
    def __init__(self, current: Dict[str, float]):
        self.current = current

    def multi_get(self, keys: List[str]) -> Dict[str, float]:
        return {k: self.current[k] for k in keys if k in self.current}


def fetch_features(registry, offline_store, online_cache, user_id, item_id, label_time, request_time):
    key_map = registry.keys_for(user_id, item_id)

    # 离线特征：必须按标签时间点位读取
    offline_values = {
        "user_ctr_7d": offline_store.read_point_in_time(key_map["user_ctr_7d"], label_time),
        "item_ctr_1d": offline_store.read_point_in_time(key_map["item_ctr_1d"], label_time),
    }

    # 实时特征：按请求时刻从在线缓存批量读取
    online_keys = [key_map["session_same_category_views"]]
    online_raw = online_cache.multi_get(online_keys)
    online_values = {
        "session_same_category_views": online_raw.get(key_map["session_same_category_views"], 0.0)
    }

    feature_vector = [
        offline_values["user_ctr_7d"],
        offline_values["item_ctr_1d"],
        online_values["session_same_category_views"],
    ]

    assert len(feature_vector) == len(registry.feature_order)
    assert request_time >= label_time
    return feature_vector


registry = FeatureRegistry()

offline_store = OfflineStore({
    "user:u1:ctr_7d": [
        FeatureRecord(value=0.10, event_time=100, available_time=100),
        FeatureRecord(value=0.20, event_time=200, available_time=200),
    ],
    "item:i1:ctr_1d": [
        FeatureRecord(value=0.05, event_time=100, available_time=100),
        FeatureRecord(value=0.08, event_time=200, available_time=200),
        # 这条虽然事件时间早，但要到 260 才可用；label_time=250 时不能读取
        FeatureRecord(value=0.30, event_time=240, available_time=260),
    ],
})

online_cache = OnlineCache({
    "session:u1:same_category_views": 3.0
})

vec = fetch_features(
    registry=registry,
    offline_store=offline_store,
    online_cache=online_cache,
    user_id="u1",
    item_id="i1",
    label_time=250,
    request_time=300,
)

assert vec == [0.20, 0.08, 3.0]
print(vec)
```

这段代码体现了四个工程要点：

1. `FeatureRegistry` 决定特征顺序和 key 规则，避免训练和推理各自拼字段。
2. `read_point_in_time` 同时检查 `event_time` 和 `available_time`，避免未来信息泄露。
3. `multi_get` 一次批量取在线特征，减少网络往返。
4. 离线特征和实时特征在服务层统一拼接，而不是让上层业务自己组装。

如果把它映射到真实工程，常见结构如下：

| 组件 | 作用 | 典型实现 |
| --- | --- | --- |
| Feature Registry | 管理特征定义、版本、实体主键 | 内部 DSL、Feast、配置中心 |
| Offline Pipeline | 回放历史事件，构造训练样本 | Spark、Hive、Flink Batch |
| Online Materialization | 将在线可读特征写入低延迟存储 | Flink、CDC、Kafka Consumer |
| Online Store | 提供毫秒级读取 | Redis、DynamoDB、Cassandra |
| Serving Layer | 请求时 `multi-get` 并拼向量 | Ranking Service / Feature Service |

真实工程例子可以想成这样：某电商排序服务接到首页请求后，先从召回层拿到 200 个候选商品，再调用特征服务：

- 用户级特征读 1 次。
- 商品级特征按 200 个候选商品做批量读取。
- 会话级特征从本地缓存或请求上下文直接取。
- 特征服务统一返回 `[candidate_count, feature_dim]` 的张量。
- 模型服务直接推理，不再自己查缓存。

这样做的价值不是“代码更优雅”，而是把线上耗时和一致性问题集中收敛到一个地方治理。

---

## 工程权衡与常见坑

第一类坑是**特征穿越**。这是影响最大的错误。训练时如果使用了未来行为、未来库存、未来曝光统计，模型会学到线上不可能获得的模式。离线指标通常会虚高，上线后明显回落。规避方式不是靠人工自觉，而是靠制度化约束：

| 风险 | 影响 | 规避方式 |
| --- | --- | --- |
| 特征穿越 | 离线效果虚高，线上指标下滑 | 点位时间校验、availability time 校验、统一定义 |
| 训练/推理逻辑分叉 | 线上输入分布与训练不一致 | 共享 registry、共享变换逻辑、版本化 |
| 高并发缓存击穿 | p99 延迟飙升，服务雪崩 | 热点保护、TTL、批量读取、本地缓存 |
| 在线存储成本过高 | 资源浪费，扩容困难 | 只物化高价值热特征、冷热分层 |
| 特征顺序错误 | 模型输入错位，结果不可解释 | 固定 schema、强校验、向量签名 |

第二类坑是“以为所有特征都该实时化”。这通常不成立。实时特征越多，链路越复杂，成本越高，稳定性越差。很多长期稳定的画像特征完全可以小时级或天级刷新。真正必须实时的，通常只有与当前请求强相关的少数特征，例如：

- 当前会话行为
- 实时库存与价格
- 分钟级热度
- 风控状态
- 实验开关上下文

第三类坑是在线存储设计不当。在线 KV 的成本通常远高于离线仓库，所以不能把所有离线字段完整镜像到 Redis。更现实的做法是：

- 热特征进入在线存储。
- 冷特征保留在离线仓库或较慢的服务中。
- 对超高频热点做本地缓存或二级缓存。
- 对长尾特征设置合理 TTL 和淘汰策略。

第四类坑是把特征服务做成“透明转发层”。如果它只负责读缓存，不负责 schema、版本、缺失值、默认值、日志回放，那它很快会变成新的耦合点。一个合格的特征服务至少要提供：

- 统一的特征名和版本管理
- 按实体批量读取
- 缺失值填充规则
- 训练回放所需的点位时间读取能力
- 特征日志记录，用于排查线上样本问题

真实工程里，最难排查的问题往往不是“服务挂了”，而是“服务没挂但指标悄悄变差”。这时如果没有特征日志，很难确认是模型参数变了、特征缺失率升高了，还是某个字段被错误回填了默认值。

---

## 替代方案与适用边界

不存在一套对所有团队都最优的实时推荐架构。方案应按团队规模、特征复杂度、实时性要求和工程成熟度选择。

| 方案 | 适用边界 | 优点 | 劣势 |
| --- | --- | --- | --- |
| 统一 DSL / Feature Store | 特征多、团队多、版本治理要求高 | 一份定义生成离线与在线逻辑，偏差更可控 | 上手成本高，平台建设重 |
| Spark/Hive → Redis 批量物化 | 中小团队、实时性要求中等 | 实现快，系统简单 | 实时性受批处理窗口限制 |
| 流处理 + Online KV | 强实时业务，如广告、交易、实时风控 | 新鲜度高，适合会话级决策 | 运维复杂，成本高 |
| Cache-only 快速方案 | 早期验证或低流量业务 | 开发速度快 | 很容易出现训练/推理不一致 |

对初学者最实用的判断标准不是“哪个名词更先进”，而是问三个问题：

1. 这个特征必须实时吗，还是小时级刷新就够？
2. 训练时能否严格复原线上可见信息？
3. 线上读取能否在预算内稳定完成？

如果团队还小，最常见的务实路径是：

- 先把特征定义收敛到统一 registry。
- 先用批处理把高价值特征物化到 Redis。
- 只把极少数会话级特征做成实时。
- 等指标和稳定性都证明有价值，再上更复杂的流式特征平台。

如果团队已经有大量模型、多业务线共享特征，或需要严格治理训练/推理一致性，那么直接建设统一特征平台更合理。此时重点不是“把更多特征实时化”，而是“把定义、时间语义、版本管理和服务接口系统化”。

换句话说，实时推荐架构不是越实时越好，而是在**准确性、成本、复杂度、延迟**之间找到可长期维护的平衡点。

---

## 参考资料

- Tacnode, “What Is an Online Feature Store? Architecture & Use Cases”, 2025-12-18  
  https://tacnode.io/post/what-is-an-online-feature-store-definition-architecture-use-cases?utm_source=openai

- NVIDIA Developer Blog, “Offline to Online Feature Storage for Real-Time Recommendation Systems with NVIDIA Merlin”, 2023-03-01  
  https://developer.nvidia.com/blog/offline-to-online-feature-storage-for-real-time-recommendation-systems-with-nvidia-merlin/?utm_source=openai

- SystemOverflow, “Online vs Offline Features: Core Distinction”  
  https://www.systemoverflow.com/learn/ml-feature-stores/online-vs-offline-features/online-vs-offline-features-core-distinction?utm_source=openai

- SystemOverflow, “Training-Serving Skew: The Silent Accuracy Killer”  
  https://www.systemoverflow.com/learn/ml-feature-stores/feature-sharing-discovery/training-serving-skew-the-silent-accuracy-killer?utm_source=openai

- SystemOverflow, “Training Serving Skew and Distribution Drift”  
  https://www.systemoverflow.com/learn/ml-feature-stores/feature-store-architecture/training-serving-skew-and-distribution-drift?utm_source=openai

- SystemOverflow, “Training-Serving Skew Root Causes and Mitigation”  
  https://www.systemoverflow.com/learn/ml-feature-stores/online-vs-offline-features/training-serving-skew-root-causes-and-mitigation?utm_source=openai
