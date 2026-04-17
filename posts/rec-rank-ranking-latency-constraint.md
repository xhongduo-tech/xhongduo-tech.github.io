## 核心结论

延迟约束下的排序建模，不是单独追求“模型越大越准”或“参数越少越快”，而是在效果、特征成本、推理资源三者之间找一个可上线的 Pareto 最优点。Pareto 最优，意思是你很难继续改善一个目标而不恶化另一个目标。在线系统真正关心的是：在满足 P99 SLA 的前提下，业务指标不能明显退化，总体算力和存储成本也要可控。

推荐系统里的排序，白话说就是“给候选内容排先后顺序”。一条常见的稳定路线是：先用粗排把大候选集快速缩小，再让精排把高价值候选排得更准。比如先从 5000 个候选里用轻量模型筛到 200 个，再把精排 80ms 的深度交叉计算只用在这 200 个上，最终总延迟 92ms，仍然落在 120ms 的 SLA 内。这类设计的价值不只在“省模型算力”，还在“少拉无用特征”。

| 目标维度 | 典型瓶颈 | 常见策略 | 直接收益 | 常见代价 |
| --- | --- | --- | --- | --- |
| 效果 | 模型表达能力不足 | 精排深模型、交叉特征、蒸馏 | 提升 CTR/CVR/NDCG | 推理更重 |
| 延迟 | 特征拉取慢、推理慢、后处理堵塞 | 级联、量化、剪枝、算子优化 | 降低 P99 | 可能损失精度 |
| 成本 | 在线特征存储和计算贵 | 特征裁剪、小模型替代、共享特征 | 降低机器和存储成本 | 需要额外离线验证 |

工程上，最重要的判断不是“这个模型参数少了多少”，而是“端到端 P99 有没有下降”。因为线上总延迟往往满足：
$$
L_{total} = L_{feature} + L_{infer} + L_{post}
$$
也就是总延迟等于特征拉取、模型推理、后处理三段之和。只压模型参数量，未必能压住整个链路的尾部延迟。

---

## 问题定义与边界

排序模型的延迟约束问题，可以定义成一个带业务下限的多目标优化问题。多目标优化，白话说就是要同时看多个指标，而不是只优化一个数。给定 P99 延迟预算、机器资源预算和最小业务效果阈值，系统要从一组候选方案里选出最合适的上线点。

一个简单表达是：

$$
\max \; Quality(model, features)
$$

同时满足：

$$
L_{feature} + L_{infer} + L_{post} \leq SLA_{p99}
$$

$$
Cost_{feature} + Cost_{compute} \leq Budget
$$

$$
Metric_{online} \geq Threshold
$$

这里的边界有三类。

第一类是特征边界。特征，就是模型输入信号，比如用户画像、商品统计、交互历史。不是所有特征都能实时拿到，也不是所有特征都值得在线拉取。部分特征可能需要跨机访问，部分特征可能需要复杂聚合，这些都会直接抬高 $L_{feature}$。

第二类是资源边界。比如 CPU 机器、GPU 推理卡、内存上限、缓存命中率、网络带宽。这些条件决定你能不能跑复杂交叉网络，也决定量化和批处理是否有意义。

第三类是业务边界。排序不是研究比赛，线上要保底指标，比如点击率、转化率、GMV、停留时长。如果某个模型虽然更快，但核心指标跌破红线，它就不属于可行解。

玩具例子可以这样理解。假设你有 5000 个候选商品，全部走深度精排要 300ms，超出 SLA。于是你先用轻量粗排在 12ms 内把 5000 个缩到 200 个，再对这 200 个跑 80ms 精排，后处理 5ms，总计 97ms。虽然粗排不如精排精细，但它把大部分明显不优的候选提前排除，给精排留出了预算。

这说明延迟约束下的排序，不是“把最强模型硬塞进线上”，而是“定义可行空间后再选点”。

---

## 核心机制与推导

核心机制有三个：级联、分段优化、Pareto 选点。

级联，白话说就是“先快筛，再细判”。粗排负责识别大范围候选中的明显好坏，精排负责在少量高价值候选中做更细的判断。它的本质不是把模型拆两层，而是把计算预算按候选规模分配。候选多时，只能用轻模型；候选少时，才值得投入重模型。

仍然看 5000 → 200 的例子。设粗排单条打分成本是 $c_c$，精排单条打分成本是 $c_f$，且 $c_f \gg c_c$。全量精排成本接近：
$$
5000 \cdot c_f
$$
级联后成本变成：
$$
5000 \cdot c_c + 200 \cdot c_f
$$
只要粗排足够轻，且不会把真正优质候选过早淘汰，这种结构就比“全量精排”更合理。

分段优化，是把总延迟拆开分别看。很多团队一开始只盯模型耗时，但线上瓶颈常常在别处。比如：

| 延迟阶段 | 典型问题 | 优化方法 | 是否一定和参数量相关 |
| --- | --- | --- | --- |
| 特征拉取 $L_{feature}$ | 跨机 I/O、多路 join、缓存未命中 | 预取、缓存、本地化、裁剪特征 | 否 |
| 模型推理 $L_{infer}$ | 网络太深、算子不友好、并行度低 | 蒸馏、量化、剪枝、编译优化 | 部分相关 |
| 后处理 $L_{post}$ | 去重、规则融合、重排串行 | 流式化、并行 merge、提前截断 | 否 |

这就是为什么“模型参数量下降”不必然等于“端到端 P99 下降”。如果你的主要瓶颈是远程特征服务，那么把模型从 80MB 压到 40MB，端到端收益可能非常有限。

Pareto 选点，是在多个方案之间找折中。假设有三种方案：

| 方案 | 效果提升 | P99 延迟 | 单请求成本 | 结论 |
| --- | --- | --- | --- | --- |
| A: 单层轻模型 | +1.5% | 45ms | 低 | 快，但效果上限低 |
| B: 粗排 + 精排 | +4.2% | 92ms | 中 | 常见上线点 |
| C: 全量重交叉网络 | +4.8% | 165ms | 高 | 超出 SLA，不可上线 |

如果 SLA 是 120ms，方案 C 虽然效果最好，但不可行。方案 B 在可行空间内效果最好，所以它更接近 Pareto 前沿上的上线点。

真实工程例子通常更复杂。比如搜索或推荐链路中的粗排层，会把“粗排和精排的差距”看成一个蒸馏问题。蒸馏，白话说就是“让小模型学大模型的输出习惯”。精排模型离线提供更稳定的软标签，粗排模型在线学习这些标签，从而在 10 到 20ms 的实时预算内，尽量逼近精排判断。再叠加量化和特征共享，最终实现“候选规模大时先保速度，候选规模小后再保精度”。

---

## 代码实现

下面先给一个可运行的 Python 玩具例子，模拟三段延迟和级联收益。这里不依赖真实模型，只计算候选规模、阶段耗时和 SLA 判断。

```python
from dataclasses import dataclass

@dataclass
class StageLatency:
    feature_ms: float
    infer_ms: float
    post_ms: float

    @property
    def total_ms(self) -> float:
        return self.feature_ms + self.infer_ms + self.post_ms

def cascade_latency(
    num_candidates: int,
    topk: int,
    coarse_feature_ms: float,
    coarse_infer_ms: float,
    fine_feature_ms: float,
    fine_infer_ms: float,
    post_ms: float,
) -> StageLatency:
    coarse_total = coarse_feature_ms + coarse_infer_ms
    fine_total = fine_feature_ms + fine_infer_ms
    # 玩具模型：粗排全量跑一次，精排只对 TopK 跑一次
    return StageLatency(
        feature_ms=coarse_feature_ms + fine_feature_ms,
        infer_ms=coarse_infer_ms + fine_infer_ms,
        post_ms=post_ms,
    )

def full_ranking_latency(
    feature_ms: float,
    infer_ms: float,
    post_ms: float,
) -> StageLatency:
    return StageLatency(feature_ms=feature_ms, infer_ms=infer_ms, post_ms=post_ms)

cascade = cascade_latency(
    num_candidates=5000,
    topk=200,
    coarse_feature_ms=4.0,
    coarse_infer_ms=8.0,
    fine_feature_ms=3.0,
    fine_infer_ms=80.0,
    post_ms=5.0,
)

full = full_ranking_latency(
    feature_ms=25.0,
    infer_ms=140.0,
    post_ms=8.0,
)

assert cascade.total_ms == 100.0
assert full.total_ms == 173.0
assert cascade.total_ms < 120.0  # 满足 SLA
assert full.total_ms > cascade.total_ms
```

上面的例子表达两个事实。第一，粗排和精排是预算分层，不是简单堆模型。第二，真正需要打点的是分段耗时，而不是只看总耗时。

下面是更贴近工程的 TypeScript 伪代码，展示如何在线上链路里采集 `L_feature`、`L_infer`、`L_post`。

```ts
type Candidate = { id: string; score?: number };
type RankResult = { items: Candidate[]; metrics: Record<string, number> };

async function fetchCoarseFeatures(req: any, cands: Candidate[]) {
  performance.mark("feature_coarse_start");
  const feats = await coarseFeatureService.batchGet(req, cands);
  performance.mark("feature_coarse_end");
  performance.measure("L_feature_coarse", "feature_coarse_start", "feature_coarse_end");
  return feats;
}

async function coarseRank(req: any, cands: Candidate[]): Promise<Candidate[]> {
  const feats = await fetchCoarseFeatures(req, cands);

  performance.mark("infer_coarse_start");
  const scored = await coarseModel.predict(feats); // 可替换为蒸馏后小模型
  performance.mark("infer_coarse_end");
  performance.measure("L_infer_coarse", "infer_coarse_start", "infer_coarse_end");

  return topK(scored, 200);
}

async function fineRank(req: any, cands: Candidate[]): Promise<Candidate[]> {
  performance.mark("feature_fine_start");
  const feats = await fineFeatureService.batchGet(req, cands);
  performance.mark("feature_fine_end");
  performance.measure("L_feature_fine", "feature_fine_start", "feature_fine_end");

  performance.mark("infer_fine_start");
  const scored = await fineModel.predict(feats, {
    quantized: true,
    pruned: true,
  });
  performance.mark("infer_fine_end");
  performance.measure("L_infer_fine", "infer_fine_start", "infer_fine_end");

  return scored;
}

async function rerankAndFilter(cands: Candidate[]): Promise<Candidate[]> {
  performance.mark("post_start");
  const deduped = dedup(cands);
  const diversified = diversify(deduped);
  const finalItems = diversified.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  performance.mark("post_end");
  performance.measure("L_post", "post_start", "post_end");
  return finalItems;
}

export async function rank(req: any, candidates: Candidate[]): Promise<RankResult> {
  const coarseTopK = await coarseRank(req, candidates);
  const fineScored = await fineRank(req, coarseTopK);
  const finalItems = await rerankAndFilter(fineScored);

  const metrics = readPerfMetrics([
    "L_feature_coarse",
    "L_infer_coarse",
    "L_feature_fine",
    "L_infer_fine",
    "L_post",
  ]);

  metrics["L_feature"] = metrics["L_feature_coarse"] + metrics["L_feature_fine"];
  metrics["L_infer"] = metrics["L_infer_coarse"] + metrics["L_infer_fine"];
  metrics["L_total"] = metrics["L_feature"] + metrics["L_infer"] + metrics["L_post"];

  logRankMetrics(req.requestId, metrics);
  return { items: finalItems, metrics };
}
```

如果还要把蒸馏和压缩纳入上线流程，通常不会在请求路径里“临时蒸馏”，而是在训练和部署阶段完成：

| 模块 | 训练或部署动作 | 目标 |
| --- | --- | --- |
| 粗排模型 | 学习精排软标签 | 用轻模型逼近重模型排序趋势 |
| 精排模型 | 量化、剪枝、算子融合 | 压缩推理时延 |
| 特征层 | 裁剪低收益特征 | 降低特征拉取成本 |
| 监控层 | 记录 P50/P95/P99 和分段时延 | 找到真正瓶颈 |

---

## 工程权衡与常见坑

第一个常见坑，是只看模型，不看特征。特征成本常被低估，但很多线上系统最贵的不是矩阵乘法，而是远程拉数、join、反序列化和缓存失效。粗排如果也拉全量特征，就像每次出门都把整个房子搬走，理论上信息最全，实际上成本根本不可接受。正确做法通常是建立最小可行特征集，也就是先找出“足够支撑粗筛”的最少特征集合。

第二个常见坑，是把压缩当成无代价操作。量化，白话说就是把高精度数值换成低位宽表示；剪枝，白话说就是删掉不重要的连接或通道。它们都可能改变模型分数分布。排序模型对分数相对次序很敏感，特别是在 Top-K 边界附近，轻微误差就可能放大成列表变化。所以压缩后不能只测离线 AUC，还要看线上 topN 一致性、分桶效果和分阶段耗时。

第三个常见坑，是只盯平均延迟。线上 SLA 大多看 P99，因为尾部请求最容易打穿体验和机器容量。平均值好看，不代表高峰期稳定。一次远程特征服务抖动、一次线程池争抢、一次 GC 抖动，都可能把 P99 拉高很多。

| 问题 | 后果 | 规避方式 |
| --- | --- | --- |
| 拉全量特征做粗排 | 特征拉取慢，存储和网络成本高 | 建最小特征集，优先本地或缓存特征 |
| 盲目量化剪枝 | 排序分布漂移，线上效果掉点 | 小流量灰度，对齐评估指标 |
| 只压参数量 | 端到端 P99 变化小 | 分段打点，先找主瓶颈 |
| 只看均值不看 P99 | 高峰期 SLA 失守 | 同时监控 P50/P95/P99 |
| 后处理串行过长 | 模型已经够快但总时延仍高 | 去重、多样性、规则融合做流水线 |

监控上至少要分三层：

| 监控维度 | 用途 |
| --- | --- |
| 平均延迟 | 看整体资源利用情况 |
| P99 延迟 | 看 SLA 和尾部稳定性 |
| 阶段分解延迟 | 判断问题在特征、推理还是后处理 |

一个经验判断是：如果你不能回答“当前 P99 主要是哪个阶段贡献的”，那就还没进入可优化状态。

---

## 替代方案与适用边界

级联加蒸馏、压缩，是最常见也最稳妥的主方案，但不是唯一方案。是否采用它，取决于延迟级别、吞吐目标和硬件条件。

第一类替代方案是异步批处理。批处理，白话说就是把多个请求拼成一批一起算，提升硬件利用率。它适合高吞吐、可接受轻微排队等待的场景，比如信息流大盘流量。但如果业务对单请求时延极其敏感，批处理排队可能直接侵蚀 P99 预算。

第二类替代方案是特征共享分层。也就是粗排和精排共享一部分特征编码结果，避免重复拉数和重复编码。它适合特征处理本身很重、模型结构相对稳定的团队。

第三类替代方案是流式后处理。流式化，白话说就是结果一出来就开始下一步，而不是等全部结束再统一处理。比如精排返回分数后，去重、业务规则过滤、多样性控制可以边到边做，这在 10 到 20ms 级硬件预算里尤其重要，因为后处理占比会被放大。

| 方案 | 主要优点 | 适用场景 | 不适用场景 |
| --- | --- | --- | --- |
| 级联粗排 + 精排 | 效果和时延平衡好 | 候选规模大、线上排序常规场景 | 候选很少时收益有限 |
| 知识蒸馏 | 小模型学大模型，部署稳定 | 粗排要逼近精排质量 | 教师模型不稳定时 |
| 量化/剪枝 | 压推理开销 | 推理是主瓶颈、硬件支持低位宽 | 主要瓶颈在 I/O 时 |
| 异步批处理 | 吞吐高，硬件利用率高 | 高并发、可容忍小排队 | 极低时延交互场景 |
| 流式后处理 | 压缩尾部时间 | 后处理规则复杂 | 后处理本来很轻时收益小 |
| 硬件定制/编译优化 | 上限高 | 大规模稳定业务 | 业务变化快、迭代频繁 |

因此，适用边界可以总结成一句话：如果你的瓶颈在“候选规模大 + 推理预算有限”，优先级联；如果瓶颈在“模型太重”，优先蒸馏和压缩；如果瓶颈在“后处理或系统调度”，优先流水线和工程重构。不要用单一武器处理所有问题。

---

## 参考资料

1. 阿里云开发者社区，《大模型推理优化：延迟、吞吐量与成本的平衡点》；用途：支撑“延迟、吞吐、成本三向权衡”和 Pareto 视角。
2. 美团技术团队，《美团搜索粗排优化的探索与实践》；用途：支撑粗排-精排级联、蒸馏、小模型上线的真实工程例子。
3. CSDN 问答，《端到端推理链路分解与 P99 追踪》；用途：支撑 `L_total = L_feature + L_infer + L_post` 的链路拆解思路。
4. CSDN 博客，《特征工程成本分析与最小可行特征集》；用途：支撑特征成本、全量拉取风险与特征裁剪策略。
5. 阿里云开发者社区，《2025 年结构化剪枝、量化与知识蒸馏进展》；用途：支撑量化、剪枝、蒸馏在部署侧的常见手段。
6. 一般推荐系统工程实践资料，如 coarse ranking、learning to rank、serving optimization 相关论文与技术博客；用途：补充排序级联、在线推理和系统监控方法。
