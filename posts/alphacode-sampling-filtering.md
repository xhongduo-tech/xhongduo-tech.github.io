## 核心结论

AlphaCode 的关键不是“单次生成更聪明”，而是“用海量采样把搜索空间铺开，再用执行结果把候选压缩到可提交规模”。这里的采样，白话说就是同一题反复生成很多份不同代码；过滤，白话说就是先拿公开样例把明显错的代码删掉；聚类，白话说就是把“行为几乎一样”的代码归成一组，避免 10 次提交都浪费在同一路解法上。

原始 AlphaCode 在 Codeforces 的 10 场模拟真实比赛中，平均排名进入前 54.3%。它使用约 41B 参数的 encoder-decoder 模型做生成，但真正决定上限的，是“百万级生成 + 样例过滤 + 行为聚类 + 最多 10 次提交”这条流水线，而不是单一模型分数。

一个最小直观例子是：某题先生成 1,000,000 份代码，样例过滤后只剩约 50,000 份，再按运行行为聚成约 200 组，最后只从最大的 10 个簇里各选 1 份提交。对新手来说，这相当于先写出海量草稿，再删掉编不通和样例不过的，最后只交 10 份“思路彼此不同”的答案。

| 阶段 | 输入数量 | 输出数量 | 作用 |
| --- | ---: | ---: | --- |
| 大规模采样 | 1,000,000 | 1,000,000 | 扩大搜索覆盖面 |
| 样例过滤 | 1,000,000 | 约 50,000 | 删掉编译失败或样例不符 |
| 行为聚类 | 约 50,000 | 约 200 个簇 | 去掉语义重复解 |
| 最终提交 | 约 200 个簇 | ≤ 10 | 满足比赛提交限制 |

---

## 问题定义与边界

问题定义很具体：给定一整道竞赛题描述，系统要在有限时间内自动生成程序，并在最多 10 次提交内尽量命中正确解。这里的“正确”不是看代码像不像答案，而是能不能通过隐藏测试。

边界也必须说清楚。

| 边界 | 描述 |
| --- | --- |
| 提交次数 | Codeforces 一题可多次提交，但 AlphaCode 的评估按“最多 10 次候选”约束来做 |
| 正确性标准 | 最终以在线评测隐藏测试是否通过为准 |
| 样例测试作用 | 只能做早筛，不能证明程序一定正确 |
| 资源约束 | 百万级采样、编译、执行都非常贵 |
| 指标差异 | `top 54.3%` 是比赛排名；`34.2%` 是 CodeContests 离线 solve rate，不能直接混算 |

这意味着 AlphaCode 解决的不是普通补全任务。普通补全更像“把 API 调用补完整”，而竞赛编程更像“先读懂题，再选算法，再精确实现”。前者主要考局部续写，后者本质上是一个极大搜索空间中的程序搜索问题。

真实工程例子是 Codeforces 四小时比赛。系统面对多题、隐藏测试、罚时和提交上限，不能像人类一样边写边调很多轮，所以必须提前把 10 次提交预算尽量留给“互相独立”的强候选，而不是 10 份细节不同但本质同错的代码。

---

## 核心机制与推导

这条流水线可以先写成一个压缩公式：

$$
N_{out}=\min\left(10,\left|\mathrm{clusters}\big(\mathrm{Filter}(\mathrm{Samples}(N_{in}))\big)\right|\right)
$$

其中：
- `Samples` 是大规模采样。
- `Filter` 是样例执行过滤。
- `clusters` 是按行为签名分组。

如果样例过滤平均保留率记为 $p_f$，那么过滤后的候选数近似是：

$$
N_f \approx p_f \cdot N_{in}
$$

在 AlphaCode 2 技术报告对同类流程的描述里，公开样例过滤平均会删掉约 95% 的样本，也就是 $p_f \approx 0.05$。当 $N_{in}=10^6$ 时，保留下来的量级就是：

$$
N_f \approx 0.05 \times 10^6 = 5 \times 10^4
$$

这就是为什么 AlphaCode 的重点不是“把 10 个答案排好”，而是“先把 100 万个答案里面的有效尾部挖出来”。

行为聚类的核心思想也很直接。它不看代码文本，而看程序在一组测试输入上的输出模式。若两段代码在这些输入上的输出一致，它们大概率属于同一类解法，至少在当前可观察行为上等价。于是，系统更愿意提交 10 个大簇的代表，而不是 10 份非常相似的代码。

流程可以压成一行：

`题目 -> 采样 1M -> 样例执行过滤 -> 生成额外测试 -> 按输出签名聚类 -> 每簇选最好样本 -> 提交前 10 个`

这里还有一个训练目标上的关键改造：GOLD。GOLD 的直觉是，训练时不必把所有参考解都同等学会，而要更偏向那些模型本来就更可能走到、且能带来成功提交的轨迹。对白话解释就是，目标从“尽量记住所有标准答案”转成“尽量提高找到任意一个正确答案的概率”。

在 AlphaCode 的后续说明中，GOLD-δ 的重要性权重被稳定化为：

$$
\tilde w = \max(\pi_\theta(\cdot)^\alpha,\beta), \quad \alpha=0.5,\ \beta=0.05
$$

这里的 $\pi_\theta(\cdot)$ 可以理解为模型当前对某段输出的置信度。这个裁剪权重的作用是两头控制：高置信样本被放大，极低置信样本不会把梯度拉得太离谱。它更贴近 `10@1K` 这类指标，因为竞赛任务关心的是“1,000 个里能不能挑出 10 个里至少 1 个正确”，而不是平均给所有参考解分配概率。

玩具例子可以把这套机制看得更清楚。假设一道题存在三类常见思路：
- 簇 A：双指针，正确，20,000 份
- 簇 B：暴力枚举，样例能过但会超时，15,000 份
- 簇 C：边界条件错一位，10,000 份

如果只按模型分数排序，前 10 份可能几乎全来自簇 A 或簇 B；但按聚类后选代表，至少能保证提交覆盖多个思路，降低“10 次都撞同一种错法”的风险。

---

## 代码实现

下面的代码不是 AlphaCode 原始实现，而是一个可运行的极简玩具版，用来说明“过滤 + 行为聚类 + 选代表”的基本结构。

```python
from collections import defaultdict

def passes_examples(code):
    # 玩具规则：包含 solve 且不包含 bug，视为样例通过
    return "solve" in code and "bug" not in code

def behavior_signature(outputs):
    # 行为签名：同一组输出视为同一簇
    return tuple(outputs)

def cluster_and_select(candidates, max_submit=10):
    clusters = defaultdict(list)
    for item in candidates:
        sig = behavior_signature(item["outputs"])
        clusters[sig].append(item)

    ranked_clusters = sorted(
        clusters.values(),
        key=lambda group: len(group),
        reverse=True,
    )

    selected = []
    for group in ranked_clusters[:max_submit]:
        best = max(group, key=lambda x: x["score"])
        selected.append(best)
    return selected

samples = [
    {"code": "solve_a()", "outputs": [1, 2, 3], "score": 0.80},
    {"code": "solve_a_v2()", "outputs": [1, 2, 3], "score": 0.85},
    {"code": "solve_b()", "outputs": [1, 2, 4], "score": 0.70},
    {"code": "solve_b_bug()", "outputs": [1, 2, 4], "score": 0.90},
    {"code": "bug_only()", "outputs": [9, 9, 9], "score": 0.99},
]

filtered = [x for x in samples if passes_examples(x["code"])]
selected = cluster_and_select(filtered, max_submit=2)

assert len(filtered) == 3
assert len(selected) == 2
assert selected[0]["code"] == "solve_a_v2()"
assert {tuple(x["outputs"]) for x in selected} == {(1, 2, 3), (1, 2, 4)}
```

这个玩具例子体现了两个点。第一，`solve_b_bug()` 分数更高，但因为代码名里有 `bug`，在过滤阶段就被删掉。第二，`solve_a()` 和 `solve_a_v2()` 行为相同，被归入同一簇，最后只保留分数更高的代表。

把它映射到真实工程实现，通常会拆成四个模块：

| 模块 | 输入 | 输出 | 工程职责 |
| --- | --- | --- | --- |
| 采样器 | 题面、语言、采样温度 | 批量候选代码 | 并行生成海量样本 |
| 过滤器 | 候选代码、公开样例 | 通过样例的代码 | 编译、执行、超时控制 |
| 聚类器 | 过滤后代码、附加测试 | 候选簇 | 依据运行行为去重 |
| 选择器 | 候选簇、评分模型 | 最终 10 份提交 | 每簇挑最优代表 |

真实系统的伪代码更接近这样：

```text
while collected_samples < 1_000_000:
    batch = model.sample(problem, language, temperature)
    for code in batch:
        if compile_ok(code) and run_examples_ok(code):
            passed.append(code)

generated_tests = test_generator(problem)
for code in passed:
    sig = execute_on_tests(code, generated_tests)
    clusters[sig].append(code)

for cluster in top_k_by_size(clusters, k=10):
    submit(best_by_score(cluster))
```

---

## 工程权衡与常见坑

最大代价不是模型前向，而是整条执行流水线的总成本。生成 100 万份程序后，还要处理编译、沙箱执行、超时、内存限制、输出比对和失败重试。对工程系统来说，这是一条很重的分布式批处理链路。

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 采样成本过高 | GPU/TPU 推理与执行资源爆炸 | 分题并行、批量采样、冷热分层调度 |
| 样例覆盖不足 | 错误程序误保留 | 生成附加测试、加大行为测试多样性 |
| 聚类签名过弱 | 错误代码扎堆进大簇 | 使用更多输入、加入超时/异常特征 |
| 评分模型偏差 | 每簇选错代表 | 结合静态特征与执行特征重排 |
| 语言分布失衡 | 候选多样性下降 | 多语言采样、随机标签和难度条件 |

一个常见误区是把“过样例”当成“基本正确”。这是错的。竞赛题的公开样例只覆盖题意展示，不覆盖边界条件。即使过滤后剩下的是 $0.05 \times N_{in}$，这批代码里仍可能大多数是错的，只是错得更隐蔽。

第二个常见坑是误把“文本去重”当“语义去重”。两份代码长得不一样，可能行为完全相同；两份代码长得很像，也可能一个会超时、一个不会。AlphaCode 选择行为聚类，就是因为竞赛评测最终看行为，不看代码表面。

第三个坑是把 AlphaCode 理解成“模型直接会做竞赛题”。更准确的说法是：模型负责高熵搜索，执行系统负责强约束筛选。没有后者，前者的大部分样本价值很低。

---

## 替代方案与适用边界

AlphaCode 的方案适合“隐藏测试强、一次命中价值高、允许重执行筛选”的任务，但它不是唯一方案。

| 策略 | 适合场景 | 局限 |
| --- | --- | --- |
| AlphaCode 式大采样+过滤+聚类 | 难题、多解空间、提交预算小 | 成本高，依赖执行基础设施 |
| 纯评分排序 | 资源紧、需要快速出少量候选 | 容易重复提交相似解 |
| 搜索式自改进 | 可反复验证局部修补的问题 | 时间长，调度复杂 |
| 生成+单元测试修复 | 企业代码、测试现成 | 对开放式竞赛题泛化有限 |

如果资源很紧，可以在过滤后只保留 `top-100` 再按分数提交。这种方法实现简单，但在难题上容易把 10 次机会全押在同一类错误思路上。反过来，如果有足够执行预算，可以在聚类后对每个簇代表再做一轮局部搜索或修补，进一步提高命中率。

适用边界也很明确。AlphaCode 风格方法最适合“答案可以通过程序执行严格验证”的问题，比如竞赛编程、测试驱动代码生成、约束明确的小型算法题。它不适合需求模糊、正确性难自动判定、或者一次生成必须非常便宜的场景。

---

## 参考资料

1. Li, Y. et al. “Competition-Level Code Generation with AlphaCode.” arXiv:2203.07814, 2022. https://arxiv.org/abs/2203.07814  
2. AlphaCode Team. “AlphaCode 2 Technical Report.” Google DeepMind, 2023. https://deepmind.google/AlphaCode2_Tech_Report.pdf  
3. Pang, R. Y. and He, H. “Text Generation by Learning from Demonstrations.” ICLR 2021. https://openreview.net/forum?id=RovX-uQ1Hua  
4. Pang, R. Y. “Learning from Rewards in Text Generation.” NYU dissertation, 2024. https://cs.nyu.edu/media/publications/pang-dissertation-20240708.pdf
