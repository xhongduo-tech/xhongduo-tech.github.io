## 核心结论

冷启动召回的任务，是在几乎没有用户行为历史时，仍然给系统产出一批“可以拿去排序或直接展示”的候选集合。这里的“召回”可以理解为先粗筛一批可能相关的内容，而不是立刻给出最终排序；“冷启动”则是用户、物品或会话刚进入系统，历史数据极少甚至为零。

冷启动阶段不能依赖经典协同过滤。协同过滤的意思是“根据相似用户或相似物品的交互关系推断兴趣”，但它要求先有点击、收藏、购买、评分等历史。没有这些数据时，召回只能更多依赖三类信息：

| 输入信号 | 候选来源 | 探索保障机制 |
|---|---|---|
| 用户注册信息、地域、设备、时间段 | 相似人群池、热门类目池、规则池 | 固定探索比例、分桶插入 |
| 新会话上下文、搜索词、浏览路径 | 相似 session 池、上下文规则召回 | $\epsilon$-greedy、流量留白 |
| 物品标题、文本、图像、类目 embedding | 内容相似物品、类目邻域、新品池 | Thompson Sampling、动态先验 |

核心结论只有三点。

第一，冷启动召回不是“等数据够了再推荐”，而是先用内容特征、上下文信号和规则先验快速造出可消费候选，尽快让系统进入可学习状态。

第二，只做 exploitation 不做 exploration 会让新物品长期零曝光。这里的 exploitation 可以直白理解为“总挑当前看起来最好的那个”，exploration 则是“故意留一部分机会去试新东西”。冷启动必须内置探索，否则新内容永远学不到反馈。

第三，工程上最常见的有效组合，不是单一模型，而是“内容召回 + 人群/上下文映射 + 规则兜底 + 探索机制”。这套组合的目标不是一开始就最优，而是先稳定产出候选，再逐步把交互数据积累起来。

玩具例子：一个新用户刚注册，系统只知道他在上海、使用 iPhone、注册时选择了“数码”和“运动”。此时可以把他映射到“华东地区 iPhone 用户 + 数码偏好人群”，先召回该人群最近点击率高的内容，再用 $\epsilon=0.1$ 给新品池保留 10% 的曝光机会。

真实工程例子：电商首页对新用户或匿名会话，通常会并行拉取地域热销、类目热门、搜索词相关、新品池、品牌池几条召回链路，再由简单策略控制探索比例，避免整页全是老爆款。

---

## 问题定义与边界

冷启动召回指的是：在缺少可靠交互历史的前提下，系统仍需要输出候选集合供后续排序或直接展示。

这个定义有三个典型场景。

| 场景 | 可用信号 | 受限原因 |
|---|---|---|
| 新用户冷启动 | 注册信息、年龄段、地域、设备、首跳页面、时间段 | 没有点击和购买历史，无法做用户协同过滤 |
| 新物品冷启动 | 标题、摘要、类目、图像、品牌、价格、上架时间 | 没有曝光和点击，无法估计真实 CTR |
| 新会话冷启动 | 当前搜索词、来源页、浏览路径、设备、地域 | 用户可能匿名，长期画像不可用 |

边界也要说清楚。

第一，本文讨论的是召回，不是精排。召回的目标是“尽量不漏掉可用候选”，允许结果较粗；精排才负责更细的个性化打分。

第二，本文默认系统能拿到静态特征和上下文特征，但拿不到充分交互历史。因此主要手段是内容相似、规则映射、人群映射、探索分发，而不是矩阵分解或深度协同过滤。

第三，冷启动不是永恒状态。很多系统里，用户只要产生几次点击，会话只要积累几步路径，物品只要拿到几十到几百次曝光，系统就会从“冷启动召回”逐步切到“混合召回”甚至“行为主导召回”。

一个容易混淆的点是：新用户冷启动和新物品冷启动不是同一个问题。新用户冷启动的关键在“理解这个人可能喜欢什么”；新物品冷启动的关键在“让这个东西先被看见并收集反馈”。前者更依赖人群映射和上下文，后者更依赖内容特征和探索机制。

例如一个新上架耳机，没有任何点击记录，系统不能说“买过这个的人还买了什么”，因为还没人买过。它只能利用标题 embedding、类目、品牌、价格带，把它挂到“蓝牙耳机”“通勤数码”“百元配件”等候选池里，再给它保留一定曝光概率。

---

## 核心机制与推导

冷启动召回的核心机制可以拆成两步：先构造候选池，再决定如何分发曝光。

### 1. 候选池构造

候选池的意思是“先把可能相关的内容收集起来”。常见来源有四类：

1. 内容相似召回：基于文本、图像、类目 embedding 找相近物品。embedding 可以理解为把文本或图片编码成向量，便于计算相似度。
2. 人群映射召回：根据注册信息、地域、设备，把用户映射到相似用户群。
3. 上下文召回：根据当前会话的搜索词、来源页、时间段，匹配相似 session。
4. 规则兜底召回：热门榜、新品榜、类目保底池、运营白名单。

### 2. 探索分发机制

如果只按当前估计分数从高到低拿结果，新物品会长期吃亏，因为它没有历史，估计值往往保守。于是需要探索机制。

#### $\epsilon$-greedy

$\epsilon$-greedy 的含义很直接：以 $1-\epsilon$ 的概率选当前最优候选，以 $\epsilon$ 的概率随机探索。

对于动作集合 $A$，策略可以写成：

$$
\pi(a|s)=
\begin{cases}
1-\epsilon+\frac{\epsilon}{|A|}, & a=\arg\max_{a'}Q(s,a') \\
\frac{\epsilon}{|A|}, & \text{otherwise}
\end{cases}
$$

这里 $Q(s,a)$ 可以理解为“在当前状态 $s$ 下，把候选 $a$ 展示出去的预估收益”。

玩具例子：有两个候选，估计 CTR 分别为 0.4 和 0.3，$\epsilon=0.1$。  
那么最优候选被选中的概率是：

$$
0.9 + 0.1/2 = 0.95
$$

次优候选被选中的概率是：

$$
0.1/2 = 0.05
$$

这 5% 很关键。它的作用不是立刻提高当前收益，而是保证次优或新品不会永远没机会。

#### Thompson Sampling

Thompson Sampling 的思路是：不要只存一个点估计，而是给每个候选维护一个“成功率分布”。分布可以理解为“我们对这个候选真实 CTR 有多大把握”。

在点击建模里，常见做法是把点击看成 Bernoulli 事件，用 Beta 分布做后验：

$$
\theta_i \sim \text{Beta}(\alpha_i,\beta_i)
$$

每轮从每个候选的后验分布采样一次，取采样值最大的候选展示。点击则更新 $\alpha_i$，未点击更新 $\beta_i$。

它比 $\epsilon$-greedy 更细的一点在于：不确定性越大，越容易被探索。新品虽然没有历史，但因为方差大，仍可能被采样到较高值。

真实工程例子：电商新品刚上架，系统可以给每个新品设置一个动态先验，比如让新品的 $\text{Beta}(\alpha,\beta)$ 与同类目老品的平均水平同步，而不是一律从极低先验开始。这样做能避免新品在强势类目里完全跑不出来。

---

## 代码实现

下面给一个可运行的简化示例。它不依赖线上系统，但把“内容分数 + 人群映射 + 探索”这条主线表达清楚。

```python
import random

def epsilon_greedy_select(candidates, epsilon=0.1):
    """
    candidates: list of dict
      {
        "id": str,
        "score": float,      # 当前估计收益
        "is_new": bool       # 是否新品/冷物品
      }
    """
    assert 0.0 <= epsilon <= 1.0
    assert len(candidates) > 0

    if random.random() < epsilon:
        return random.choice(candidates)

    best = max(candidates, key=lambda x: x["score"])
    return best

def build_candidate_pool(user_profile, context, items):
    pool = []

    for item in items:
        score = 0.0

        # 内容特征
        if item["category"] in user_profile["preferred_categories"]:
            score += 0.4

        # 上下文特征
        if item["region"] == context["region"]:
            score += 0.2
        if context["device"] in item["target_devices"]:
            score += 0.1

        # 规则先验
        if item["is_hot"]:
            score += 0.2
        if item["is_new"]:
            score += 0.1  # 新品保底先验，不然容易彻底没机会

        pool.append({
            "id": item["id"],
            "score": score,
            "is_new": item["is_new"]
        })

    return pool

user_profile = {
    "preferred_categories": ["sports", "digital"]
}

context = {
    "region": "shanghai",
    "device": "iphone"
}

items = [
    {"id": "A", "category": "digital", "region": "shanghai", "target_devices": ["iphone"], "is_hot": True,  "is_new": False},
    {"id": "B", "category": "sports",  "region": "beijing",  "target_devices": ["android"], "is_hot": False, "is_new": True},
    {"id": "C", "category": "books",   "region": "shanghai", "target_devices": ["iphone"], "is_hot": False, "is_new": True},
]

pool = build_candidate_pool(user_profile, context, items)
assert len(pool) == 3
assert max(pool, key=lambda x: x["score"])["id"] == "A"

picked = epsilon_greedy_select(pool, epsilon=0.1)
assert picked["id"] in {"A", "B", "C"}
```

这段代码表达了三个工程要点。

第一，候选池不是从一个地方来，而是多个信号并行拼起来。  
第二，新品即使分数低，也应有保底先验。  
第三，最终分发阶段要留探索口，不然“冷启动”会变成“永远冷”。

如果把它翻译成更接近线上服务的伪代码，大致是：

```text
signals = gather(user_profile, session_context, device, region)
content_candidates = recall_by_content(signals)
crowd_candidates = recall_by_crowd_mapping(signals)
rule_candidates = recall_by_rules(signals)
new_item_candidates = recall_new_items(signals)

pool = merge_dedup(content_candidates, crowd_candidates, rule_candidates, new_item_candidates)
scored_pool = score_with_priors(pool, signals)

result = []
for slot in slots:
    item = apply_epsilon_greedy_or_ts(scored_pool, epsilon, posterior)
    result.append(item)
    scored_pool.remove(item)

return topk(result)
```

关键变量解释如下。

| 变量 | 含义 | 工程作用 |
|---|---|---|
| `epsilon` | 探索概率 | 控制给新候选多少试错流量 |
| `pool` | 候选池 | 汇总多路召回结果 |
| `posterior` | 后验分布 | 保存 Thompson Sampling 的不确定性信息 |
| `signals` | 上下文与用户静态信号 | 冷启动阶段的主要输入 |

---

## 工程权衡与常见坑

冷启动召回难的地方，不是公式本身，而是工程上要在质量、探索、延迟、可解释性之间取平衡。

| 维度 | 做强后的收益 | 代价 | 常见规避措施 |
|---|---|---|---|
| 可解释性 | 方便排查和运营理解 | 规则可能僵化 | 规则只做兜底，主链路保留内容相似 |
| 探索强度 | 新品更快获得反馈 | 短期 CTR 可能下降 | 分位置、分流量、分人群控制探索 |
| 实时性 | 能快速响应上下文变化 | 计算和缓存更复杂 | 召回离线预计算，分发在线轻量化 |
| 系统复杂度 | 召回覆盖更全 | 维护成本上升 | 明确主召回和兜底召回职责 |

常见坑 1：只做 exploitation。  
后果是老物品越来越强，新物品越来越弱，系统进入反馈闭环。解决方法不是“给新品单独开页面”，而是在主召回链路中就保留探索概率，或者给新品单独候选池并混入主结果。

常见坑 2：只看内容，不看上下文。  
同一个用户在晚间、通勤、节假日、不同设备上的需求可能不同。上下文就是“当前场景信息”，比如时间、地点、设备、来源页。冷启动时长期画像弱，上下文反而更重要。

常见坑 3：探索全局固定，不按位置区分。  
首页首屏第 1 位和列表第 20 位的商业价值不一样。探索预算应该分槽位控制，而不是全站统一一个 $\epsilon$。

常见坑 4：新品先验太低。  
如果 Thompson Sampling 的先验对新品过于保守，新品即使被纳入候选池，也很难真正拿到曝光。工程上常见做法是按类目、价格带、品牌层级给新品设动态先验。

真实工程例子：某电商在新会话进入首页时抓取来源渠道、搜索词、设备、时间、地理位置，并匹配相似 session 的高点击商品作为候选；随后在每个版位上按不同 $\epsilon$ 执行探索，第 1 屏探索比例较低，后续楼层略高。这样既不明显伤害主转化，又能持续给新品收集反馈。

---

## 替代方案与适用边界

“内容 + 规则 + 探索”不是唯一方案，但在冷启动阶段通常最稳。

| 方案 | 适用场景 | 不适用场景 |
|---|---|---|
| 内容召回 + 规则兜底 | 新物品多、文本图像质量高 | 内容特征贫弱、类目标注差 |
| 人群映射 + 上下文召回 | 新用户、新会话 | 用户信息缺失严重 |
| $\epsilon$-greedy | 实现简单、易控流量 | 候选很多且质量差异大时较粗糙 |
| Thompson Sampling | 候选数有限，需利用不确定性探索 | 超大候选集、分布维护成本高 |
| 协同过滤/深度召回 | 已有丰富交互历史 | 冷启动初期数据不足 |

如果系统已经积累了足够行为历史，那么主召回通常会逐步切换到协同过滤、双塔召回、图召回等更强的行为驱动方法。冷启动策略不会消失，但会退居为补充链路，例如：

1. 用协同过滤召回大头候选。
2. 在顶部或固定比例槽位中插入探索候选。
3. 对新物品维持保底曝光和反馈收集。

如果业务极度重视可解释性，比如运营活动页、教育内容分发、金融资讯推荐，那么带权规则和流量分桶有时比 Thompson Sampling 更容易落地，因为它更容易审计和复盘。

如果业务极度追求短期转化，探索预算会被压得很低，这时要明确接受一个事实：新品成长速度会变慢，系统会更偏向存量优势内容。这个边界不是算法错误，而是业务目标选择。

---

## 参考资料

| 标题 | 作用 | 网址/描述 |
|---|---|---|
| Cold-Start Item Recommendations | 说明冷启动物品推荐依赖内容特征与侧信息 | https://www.emergentmind.com/topics/cold-start-item-recommendations |
| Mastering Cold Start Challenges | 说明用户冷启动可利用注册信息、上下文、人群映射 | https://www.shaped.ai/blog/mastering-cold-start-challenges |
| Reinforcement Learning: Core Concepts | 提供 $\epsilon$-greedy 基本公式与探索解释 | https://www.next.gr/ai/ai-in-education/reinforcement-learning-core-concepts |
| Exploration vs Exploitation Strategies | 说明 Thompson Sampling 的基本流程与适用场景 | https://next.gr/ai/reinforcement-learning/exploration-vs-exploitation-strategies |
| Core Bandit Algorithms: Epsilon-Greedy, UCB, and Thompson Sampling | 提供 bandit 视角下的玩具例子与比较 | https://www.systemoverflow.com/learn/ml-recommendation-systems/diversity-exploration/core-bandit-algorithms-epsilon-greedy-ucb-and-thompson-sampling |
| Dynamic Prior Thompson Sampling for Cold-Start Exploration | 说明动态先验如何提升新品探索概率 | https://www.themoonlight.io/review/dynamic-prior-thompson-sampling-for-cold-start-exploration-in-recommender-systems |
| Session context data framework for e-commerce recommendation | 说明新会话上下文如何参与冷启动推荐 | ScienceDirect 上关于 session context 的电商推荐论文 |
