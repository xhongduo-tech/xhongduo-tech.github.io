## 核心结论

会话推荐是基于用户当前会话行为预测下一步物品的推荐任务。它的核心难点不在于“能不能推荐”，而在于“能不能在用户每次点击、滑动、加购之后，立刻更新推荐结果”。

结论链路如下：

`用户点击 -> 状态增量更新 -> 在线特征读取 -> 候选打分 -> 返回推荐`

形式化地看，当前会话状态记为 $h_t$，当前点击物品记为 $x_t$，候选物品 $i$ 的向量记为 $e_i$。实时会话推荐通常可以抽象成两步：

$$
h_t = f(h_{t-1}, x_t)
$$

$$
\hat y_{t+1}(i) = g(h_t, e_i)
$$

其中，$h_t$ 是“当前会话兴趣摘要”，用一个向量表示用户在本次会话里的短期兴趣；$\hat y_{t+1}(i)$ 是候选物品 $i$ 成为下一步点击目标的预测分数。

新手版例子：用户刚点完一个商品，系统不能等“重新训练一次模型”再推荐下一屏。正确做法是用这次点击更新当前会话状态，然后马上对候选商品重新打分。

真实工程例子：电商首页的“下一屏推荐”。用户连续点击 3 个手机配件后，服务端更新本次会话向量，从在线特征库读取候选商品的类目、品牌、文本向量等特征，在 50-100 ms 内返回下一屏推荐。类目、品牌、文本向量属于静态特征，提前离线缓存；当前会话状态属于动态状态，在线实时更新。

---

## 问题定义与边界

会话推荐的输入是当前会话序列：

$$
S_t=(x_1,\dots,x_t)
$$

目标是预测下一步物品：

$$
x_{t+1}
$$

这里的“会话”指一次连续访问过程，例如用户打开 App 后在几分钟内连续浏览、点击、搜索、加购。会话推荐默认主要依赖当前会话内的行为序列，而不是长期用户画像。

术语说明：用户画像是长期积累的用户特征，例如年龄、性别、会员等级、历史购买偏好、长期兴趣标签。会话推荐强调短期上下文，用户画像推荐强调长期偏好。

新手版边界：如果系统只看你刚点过的 3 个商品来推荐下一屏，这属于会话推荐；如果系统还长期依赖你的历史购买、年龄、会员等级，那就已经超出纯会话推荐的定义边界。

| 维度 | 会话推荐 | 用户画像推荐 |
|---|---|---|
| 输入 | 当前会话序列 $S_t$ | 长期历史、人口属性、会员信息 |
| 目标 | 预测下一步物品 $x_{t+1}$ | 预测长期偏好或转化概率 |
| 更新频率 | 每次点击后更新 | 周期性或离线更新 |
| 延迟要求 | 通常是毫秒级 | 可以相对更宽松 |
| 典型场景 | 下一屏推荐、直播间推荐、信息流连续刷新 | 首页长期偏好、会员营销、人群定向 |

本文讨论的实时性挑战只覆盖在线链路：状态维护、特征读取、候选打分、排序、缓存、蒸馏和延迟监控。离线指标如 Recall@K、NDCG@K 很重要，但它们不是本文的中心。一个模型离线指标高，不代表它能在高并发线上服务中稳定满足 p95/p99 延迟要求。

---

## 核心机制与推导

实时会话推荐的核心机制是增量更新。增量更新是指只根据上一个状态 $h_{t-1}$ 和当前新行为 $x_t$ 计算新状态 $h_t$，而不是每次都从 $x_1$ 到 $x_t$ 重算整段历史。

$$
h_t=f(h_{t-1},x_t)
$$

候选打分可以写成：

$$
\hat y_{t+1}(i)=g(h_t,e_i)
$$

最简单的 $g$ 可以是点积：

$$
g(h_t,e_i)=h_t \cdot e_i
$$

玩具例子：设当前会话状态 $h_t=[0.2,0.8]$。候选 A 的向量是 $e_A=[0.3,0.7]$，候选 B 的向量是 $e_B=[0.9,0.1]$。

A 的分数：

$$
0.2 \times 0.3 + 0.8 \times 0.7 = 0.62
$$

B 的分数：

$$
0.2 \times 0.9 + 0.8 \times 0.1 = 0.26
$$

所以 A 排在 B 前面。若用户又点击了一个更偏向第二维兴趣的商品，状态更新为 $h_t=[0.1,0.9]$，系统只需要用新状态重新计算候选分数，不需要重新训练模型，也不需要完整重算整段会话历史。

实时系统通常把静态部分离线化，把动态部分在线增量化。

| 类型 | 示例 | 处理方式 | 原因 |
|---|---|---|---|
| 静态特征 | 类目、品牌、价格段 | 离线计算后写入在线存储 | 变化慢，适合缓存 |
| 语义特征 | 标题向量、图片向量、文本向量 | 离线模型批量生成 | 计算成本高 |
| 动态状态 | 当前会话向量 $h_t$ | 在线增量更新 | 每次点击都会变化 |
| 实时上下文 | 当前页面、时间、设备、入口 | 在线读取或拼接 | 与请求强相关 |

推导链路可以压缩为：

`历史点击 -> 状态更新 -> 候选打分 -> 排序输出`

当模型较大时，另一个常见手段是模型蒸馏。蒸馏是指用一个大模型作为 teacher，让一个小模型 student 学习 teacher 的输出分布，从而降低线上推理延迟。典型损失函数为：

$$
L=\alpha L_{CE}+(1-\alpha)T^2 KL(\sigma(z^T/T)\|\sigma(z^S/T))
$$

其中，$L_{CE}$ 是交叉熵损失，用来拟合真实标签；$KL$ 是 KL 散度，用来衡量两个概率分布的差异；$z^T$ 是 teacher 的 logits，$z^S$ 是 student 的 logits；$T$ 是温度参数，用来软化概率分布；$\alpha$ 控制真实标签和 teacher 知识之间的权重。

工程含义很直接：teacher 可以离线慢慢算，student 必须在线快速跑。如果 teacher 推理一次要 80 ms，而线上预算只有 20 ms，就需要把 teacher 的排序能力压缩到更小的 student 模型里。

---

## 代码实现

在线链路的最小闭环包括四件事：状态更新、特征读取、打分排序、缓存回填。重点不是训练一个复杂模型，而是把请求路径拆清楚。

伪代码如下：

```python
def recommend(session_state, clicked_item, candidates):
    session_state = update_state(session_state, clicked_item)  # 增量更新
    item_feats = fetch_online_features(candidates)             # 在线读取
    scores = score(session_state, item_feats)                  # 只算当前候选
    return rank(scores)
```

一个可运行的最小 Python 例子如下。这里用二维向量表示会话状态和物品特征，用点积表示打分函数。

```python
from math import isclose

ITEM_FEATURES = {
    "A": [0.3, 0.7],
    "B": [0.9, 0.1],
    "C": [0.1, 0.9],
}

def normalize(v):
    s = sum(v)
    if s == 0:
        return v
    return [x / s for x in v]

def update_state(session_state, clicked_item):
    clicked_vec = ITEM_FEATURES[clicked_item]
    # 简化版增量更新：新状态 = 旧状态 * 0.7 + 新点击向量 * 0.3
    new_state = [
        0.7 * session_state[i] + 0.3 * clicked_vec[i]
        for i in range(len(session_state))
    ]
    return normalize(new_state)

def fetch_online_features(candidates):
    return {item: ITEM_FEATURES[item] for item in candidates}

def score(session_state, item_feats):
    return {
        item: sum(a * b for a, b in zip(session_state, feat))
        for item, feat in item_feats.items()
    }

def rank(scores):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def recommend(session_state, clicked_item, candidates):
    new_state = update_state(session_state, clicked_item)
    item_feats = fetch_online_features(candidates)
    scores = score(new_state, item_feats)
    return new_state, rank(scores)

state = [0.2, 0.8]
new_state, ranked = recommend(state, "C", ["A", "B", "C"])

assert ranked[0][0] == "C"
assert ranked[-1][0] == "B"
assert isclose(sum(new_state), 1.0)

print(new_state)
print(ranked)
```

这段代码对应的工程结构如下：

| 函数 | 职责 | 在线链路中的位置 |
|---|---|---|
| `update_state()` | 根据新点击更新会话状态 | 请求刚进入时 |
| `fetch_online_features()` | 读取候选物品特征 | 特征服务阶段 |
| `score()` | 对候选进行打分 | 模型推理阶段 |
| `rank()` | 按分数排序 | 返回前处理 |
| `cache_refresh()` | 刷新特征或会话缓存 | 异步或旁路任务 |

训练侧可以补一个 teacher-student 蒸馏流程：

```python
for batch in train_loader:
    teacher_logits = teacher(batch.features)   # 离线大模型
    student_logits = student(batch.features)   # 线上小模型

    hard_loss = cross_entropy(student_logits, batch.labels)
    soft_loss = kl_divergence(
        softmax(teacher_logits / T),
        softmax(student_logits / T),
    )

    loss = alpha * hard_loss + (1 - alpha) * T * T * soft_loss
    loss.backward()
    optimizer.step()
```

关键边界是：离线构建和在线查询必须分开。训练期可以处理全量样本、复杂特征和大模型；在线服务只应该做当前请求必要的状态更新、特征读取和候选打分。

---

## 工程权衡与常见坑

会话推荐的实时性不是单点优化，而是延迟、精度、特征新鲜度之间的取舍。延迟是请求从进入系统到返回结果的耗时；特征新鲜度是在线特征反映真实业务状态的及时程度。

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 每次全量重算会话历史 | 会话越长延迟越高 | 会话 state 增量更新 |
| 特征不新鲜 | 推荐过时商品或错误候选 | 使用 `materialize` / `materialize-incremental` 同步在线存储 |
| 蒸馏后精度掉太多 | 线上点击率下降 | teacher 离线校准 + A/B 对照 |
| 训练和线上口径不一致 | 离线好、线上差 | 统一 feature definition 和 point-in-time 逻辑 |
| 只看平均延迟 | 长尾请求拖垮体验 | 增加 p95/p99 延迟监控 |
| 缓存没有失效策略 | 读到旧状态或旧特征 | 设置 TTL、版本号和回源逻辑 |

新手版例子：如果线上的类目特征还是昨天的，模型可能会推荐“看起来分数高但已经过时”的商品。比如商品已经下架、库存不足、价格变化，但在线特征没有更新。即使离线指标不错，线上体验也会变差。

真实工程中必须同时看这些指标：

| 指标 | 含义 | 用途 |
|---|---|---|
| 平均延迟 | 请求平均耗时 | 判断整体成本 |
| p95/p99 延迟 | 最慢 5% / 1% 请求耗时 | 判断长尾稳定性 |
| CTR | 点击率 | 判断推荐吸引力 |
| CVR | 转化率 | 判断业务收益 |
| Recall@K / NDCG@K | 离线排序指标 | 判断模型基础能力 |
| 特征新鲜度 | 特征从产生到可被读取的延迟 | 判断在线特征是否可靠 |
| 缓存命中率 | 请求命中缓存的比例 | 判断缓存设计是否有效 |

一个常见工程决策是：不要让所有候选都进入复杂模型。先用轻量召回拿到几百个候选，再用小模型排序前几十个。这样可以把模型计算量控制在稳定范围内。

---

## 替代方案与适用边界

会话状态增量推荐不是唯一方案。它适合强交互、连续反馈、用户兴趣快速变化的场景。如果业务对实时性要求没那么高，或者会话很短，复杂状态模型可能不是最优选择。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 纯离线推荐 | 实现简单，线上延迟低 | 无法快速响应当前点击 | 长周期偏好推荐、低频刷新页面 |
| 会话状态增量推荐 | 能快速反映当前兴趣 | 状态维护和缓存复杂 | 信息流、直播间、下一屏推荐 |
| Teacher/Student 蒸馏后在线推荐 | 兼顾精度和延迟 | 训练链路更复杂 | 大模型效果好但线上预算紧 |
| 轻量召回 + 小排序模型 | 工程稳定，成本可控 | 表达能力有限 | 电商首页、内容推荐、候选规模较大场景 |

新手版例子：电商首页如果只是做“下一屏推荐”，轻量召回 + 小排序模型可能就够了。先根据最近点击类目召回一批商品，再用小模型排序。如果是直播间、短视频信息流、强交互搜索推荐，用户兴趣变化更快，就更需要会话状态增量更新。

选择方案时可以按以下条件判断：

1. 延迟预算是否严格：如果线上预算只有几十毫秒，大模型直接在线推理通常不合适。
2. 会话长度是否足够：如果大多数用户只点一次，会话状态模型的收益可能有限。
3. 特征是否可实时更新：如果关键特征无法在线更新，实时模型会被旧特征限制。
4. 精度损失是否可接受：蒸馏和轻量模型会降低表达能力，需要 A/B 验证。
5. 系统复杂度是否可维护：状态服务、特征服务、缓存刷新、监控报警都需要工程投入。

本文中的公式和蒸馏机制主要来自推荐系统和知识蒸馏文献；延迟、缓存、特征新鲜度、p95/p99 监控属于工程实践归纳。它们不是数学定理，而是线上推荐系统长期运行后形成的稳定经验。

---

## 参考资料

| 阅读顺序 | 资料 | 解决的问题 | 对应章节 |
|---|---|---|---|
| 1 | 会话推荐综述 | 为什么要维护会话状态 | 问题定义与边界、核心机制与推导 |
| 2 | 增量学习论文 | 如何理解在线更新 | 核心机制与推导 |
| 3 | 蒸馏论文 | 如何把大模型压到小模型 | 核心机制与推导、代码实现 |
| 4 | Feast 特征读取文档 | 在线如何取回特征 | 代码实现、工程权衡 |
| 5 | Feast 在线存储文档 | 特征如何进入在线存储 | 工程权衡与常见坑 |

1. [Session-aware recommendation: A surprising quest for the state-of-the-art](https://doi.org/10.1016/j.ins.2021.05.048)
2. [Incremental Learning for Personalized Recommender Systems](https://doi.org/10.48550/arXiv.2108.13299)
3. [Distilling the Knowledge in a Neural Network](https://doi.org/10.48550/arXiv.1503.02531)
4. [Feast: Feature retrieval](https://docs.feast.dev/getting-started/concepts/feature-retrieval)
5. [Feast: Online store](https://docs.feast.dev/getting-started/components/online-store)
