## 核心结论

事件关系抽取，白话说，就是判断两个事件之间是不是存在“谁导致谁”“谁先谁后”“满足什么条件才发生”的连接。它不是只看单句里的连接词，而是要在整篇文档里把事件串成图，再判断边的类型和方向。

对初级工程师最重要的结论有三点：

1. 显式线索只能解决一部分问题。像“因为”“导致”“随后”这类连接词很有用，但覆盖有限，跨句隐式关系很容易漏掉。
2. 因果关系和时序关系应该分开建模，再融合。因为“先发生”不等于“导致”，但很多因果判断又必须依赖时间顺序。
3. 文档级图模型比单句分类更稳。R-GCN 这类关系图卷积网络可以让“因果边、时序边、共指边、同句边”之间传播信息，适合处理新闻、灾害通报、事故调查这类长文本。

一个直接的经验值是：在 EventStoryLine 数据集上，文档级双通道模型论文中给出的 `BERT+SCITE` 基线 F1 为 `54.72`，说明只靠局部文本线索，性能很快会碰到上限。

| 方法 | 主要线索 | 能覆盖跨句隐式关系吗 | 优点 | 短板 |
| --- | --- | --- | --- | --- |
| 显式连接词规则 | 因为、导致、随后、如果 | 很弱 | 快、便宜、可解释 | 漏召回严重 |
| 句级分类模型 | 句内上下文 | 一般 | 实现简单 | 难处理文档级依赖 |
| 文档级图模型 | 事件节点 + 多类边 | 强 | 能整合因果与时序 | 成本高、实现复杂 |

---

## 问题定义与边界

设文档中抽到的事件集合为 $E=\{E_1,E_2,\dots,E_n\}$。这里“事件”指文本里一个可定位的动作、状态变化或发生事实，白话说，就是“真的发生了什么”。我们的目标是给事件对 $(E_i,E_j)$ 判定关系：

$$
R(E_i,E_j)\in\{\text{Causal},\text{Temporal},\text{Conditional},\text{None}\}
$$

更细一点，可以把它写成二分类或多分类问题。例如：

- 因果：$E_i \rightarrow E_j$ 表示 $E_i$ 是原因，$E_j$ 是结果
- 时序：$E_i \prec E_j$ 表示 $E_i$ 先于 $E_j$
- 条件：$E_i \Rightarrow E_j$ 表示只有在 $E_i$ 成立时，$E_j$ 才可能发生

边界要先说清楚，否则模型会学乱：

| 情况 | 是否连边 | 说明 |
| --- | --- | --- |
| 明确连接词“暴雨导致洪水” | 通常连因果边 | 显式证据强 |
| “先爆炸，后起火” | 连时序边，因果未必成立 | 先后不自动推出因果 |
| “若持续降雨，将停课” | 连条件边 | 这是条件触发，不是已发生因果 |
| 否定句“未发生泄漏，因此无污染” | 要谨慎 | 事件极性改变，不能直接套模板 |
| 反事实“如果当时撤离，就不会受伤” | 通常不当作已发生因果 | 它描述的是假设世界 |

玩具例子：

“暴雨即将来临。地质局发布预警。洪水出现。”

这里至少有三个事件：`暴雨`、`发布预警`、`洪水出现`。常见判断是：

- `暴雨 -> 洪水出现`：可能是因果
- `发布预警 ≺ 洪水出现`：通常是时序
- `暴雨 -> 发布预警`：很多场景下可判为因果，但依赖领域定义，因为预警是机构决策，不是自然结果

这说明事件关系抽取不是“看见两个事件就连线”，而是要先定义任务边界。

---

## 核心机制与推导

一个实用框架是“双通道 + 文档级图”。

- 因果通道：专门学习“原因-结果”模式
- 时序通道：专门学习“先后顺序”模式

“通道”可以理解成两套并行的判断路径，像两名不同分工的分析员：一个专看因果，一个专看时间。

先看为什么要拆开。句子“先爆炸，后起火”给出很强的时序信号，但真实因果可能是“爆炸导致起火”，也可能二者都由“线路短路”引发。只用一个分类器，模型容易把“先后”误学成“因果”。双通道的目的，就是把这两种信息先分开，再在更高层融合。

图模型里，每个事件是一个节点，不同关系是不同类型的边。R-GCN 的更新可以写成：

$$
h_i^{(l)}=\sigma\left(
W_0^{(l)}h_i^{(l-1)}+
\sum_{r\in \mathcal{R}}\sum_{j\in \mathcal{N}_i^r}
\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l-1)}
\right)
$$

含义很直接：

- $h_i^{(l)}$：第 $l$ 层后事件 $E_i$ 的表示，白话说，是模型当前对这个事件的“理解向量”
- $\mathcal{R}$：边类型集合，比如因果边、时序边、共指边、同句边
- $\mathcal{N}_i^r$：通过关系 $r$ 连到 $E_i$ 的邻居
- $W_r^{(l)}$：每类边各自的参数矩阵
- $W_0^{(l)}$：自环参数，保留节点自身信息

玩具例子可以画成下面这样：

```text
因果通道:
暴雨  ----causal----> 洪水

时序通道:
暴雨  ----before----> 次日景区关闭
预警  ----before----> 洪水
```

如果只看“暴雨”和“次日景区关闭”，文本里可能没有“因为”这种词。但经过图传播后，模型能利用：

- `暴雨 -> 洪水` 的因果边
- `洪水 ≺ 景区关闭` 的时序或邻接信息
- 文档中这些事件共处同一主题

于是隐式关系更容易被补全。

真实工程例子是灾害监测。通报里常见写法是：

“受持续强降雨影响，多地河流水位上涨。次日，景区关闭，部分列车停运。”

这里 `强降雨 -> 水位上涨` 常常显式；`强降雨 -> 景区关闭` 却常常跨句、跨段、隐式。工程上如果只靠连接词规则，后者基本抓不到；如果先抽时序，再把时序结果喂给因果图，召回会明显更稳。

---

## 代码实现

一个最小可运行实现，通常分四步：

| 组件 | 作用 |
| --- | --- |
| Tokenizer / Encoder | 把文本编码成向量 |
| Candidate Finder | 用连接词、窗口、句法先筛候选事件对 |
| Dual-channel Graph | 构建因果图和时序图 |
| Classifier | 分别输出因果概率与时序概率 |

下面这个 `python` 例子不是生产模型，而是一个能跑通思路的玩具版本：先用显式词筛候选，再根据简单规则输出因果与时序概率，最后做约束推理。

```python
from dataclasses import dataclass

@dataclass
class Event:
    name: str
    sent_id: int

CONNECTIVES_CAUSAL = {"因为", "导致", "造成", "引发"}
CONNECTIVES_TEMPORAL = {"随后", "之后", "次日", "先", "后"}

def find_candidates(text, events):
    pairs = []
    for i in range(len(events)):
        for j in range(len(events)):
            if i == j:
                continue
            e1, e2 = events[i], events[j]
            if abs(e1.sent_id - e2.sent_id) <= 1:
                pairs.append((e1, e2))
    return pairs

def score_pair(text, e1, e2):
    causal = 0.1
    temporal = 0.1

    if e1.name in text and e2.name in text:
        if any(w in text for w in CONNECTIVES_CAUSAL):
            causal += 0.6
        if any(w in text for w in CONNECTIVES_TEMPORAL):
            temporal += 0.6
        if e1.sent_id < e2.sent_id:
            temporal += 0.2

    causal = min(causal, 0.99)
    temporal = min(temporal, 0.99)
    return causal, temporal

def decode_relation(causal_prob, temporal_prob, threshold=0.7):
    result = []
    if causal_prob >= threshold:
        result.append("Causal")
    if temporal_prob >= threshold:
        result.append("Temporal")
    return result or ["None"]

text = "因为暴雨，洪水泛滥。次日，景区关闭。"
events = [
    Event("暴雨", 0),
    Event("洪水泛滥", 0),
    Event("景区关闭", 1),
]

pairs = find_candidates(text, events)
results = {(a.name, b.name): decode_relation(*score_pair(text, a, b)) for a, b in pairs}

assert ("暴雨", "洪水泛滥") in results
assert "Causal" in results[("暴雨", "洪水泛滥")]
assert "Temporal" in results[("暴雨", "景区关闭")]
assert "Causal" not in results[("景区关闭", "暴雨")]
```

真正的工程版会把上面的 `score_pair` 换成：

1. 用 BERT 编码事件提及和上下文
2. 构造两张邻接矩阵 `A_causal`、`A_temporal`
3. 在图上跑 R-GCN 或 GAT
4. 用两个分类头分别输出 $p_c$ 和 $p_t$
5. 推理时加入非自反、唯一性、共指一致性约束

伪代码可以写成：

```text
events = extract_events(document)
candidates = explicit_filter(document, events) + window_expand(events)
X = bert_encode(document, events)

A_causal = build_graph(candidates, type="causal")
A_temporal = build_graph(candidates, type="temporal")

H_c = rgcn(X, A_causal)
H_t = rgcn(X, A_temporal)
H = fuse(H_c, H_t)

p_causal = causal_head(H)
p_temporal = temporal_head(H)

relations = constrained_decode(p_causal, p_temporal)
```

---

## 工程权衡与常见坑

最常见的误区，是把事件关系抽取理解成“句子分类的加长版”。它其实更接近“受约束的图推理”。

| 坑 | 现象 | 规避策略 |
| --- | --- | --- |
| 只依赖显式连接词 | 召回低，跨句关系大量漏掉 | 显式词只做候选过滤，不做最终判断 |
| 不区分因果与时序 | 把“先发生”误判成“导致” | 双通道建模，再融合 |
| 不加图约束 | 出现自反、互相矛盾、共指不一致 | 加非自反、唯一性、共指一致性约束 |
| 边权静态 | 文档换领域后不稳 | 用时序信号或高斯核动态调边权 |
| 只看单次提及 | “暴雨”“此次暴雨”“7月暴雨”判定不一致 | 做事件聚合或共指链接 |

一个典型坑是同一事件的不同提及。比如文档里同时出现“2024年7月暴雨”和“此次暴雨”，如果模型把前者判成 `导致洪水`，后者又判成 `不导致洪水`，那就是图层面不一致。EiGC 这类方法专门加入了三类约束：唯一性、非自反、共指一致性，本质上是在推理阶段把“不合理解”剪掉。

另一个坑是静态边权。很多文本里，时间越接近，直接因果的概率越大；时间隔得很远，通常更可能是间接关联。因此一些双通道方法会用高斯核根据时序信号调节因果边权，而不是让所有边同等传播。

---

## 替代方案与适用边界

不是所有场景都要上图神经网络。方法选择应看文本长度、关系复杂度、标注预算。

| 方案 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| 模板/规则 | 标题、短句、公告 | 快、便宜、上线简单 | 泛化差 |
| 句级分类 | 单句内显式关系多 | 训练成本较低 | 跨句弱 |
| 双塔/交叉编码器 | 候选事件对已给定 | 判别能力强 | 缺少全局一致性 |
| 文档级图模型 | 长文、跨句、关系交织 | 能建全局结构 | 开发和标注成本高 |
| 规则 + 弱监督 | 小语料、低资源领域 | 启动快 | 上限有限 |

新手可以这样理解使用边界：

- 新闻标题“暴雨导致航班延误”：模板法往往够用。
- 事故复盘长文里“先报警、后停机、最终起火”，且多个事件跨段落出现：更适合文档级图方法。
- 如果标注数据很少，先用规则做候选生成，再用弱监督或小模型微调，通常比直接训练大图模型更稳。

因此，事件关系抽取没有单一最优解。短文本、明示关系多的场景，传统模板依然有价值；长文档、隐式关系多、要求全局一致性的场景，图方法才真正发挥优势。

---

## 参考资料

| 资料 | 主要贡献 | 适合怎么读 |
| --- | --- | --- |
| Yang, Han, Poon. *A survey on extraction of causal relations from natural language text* | 系统梳理显式/隐式、句内/跨句因果抽取方法 | 先建立任务全景，再看具体模型 |
| *Document-Level Causal Event Extraction Enhanced by Temporal Relations Using Dual-Channel Neural Network* | 双通道思路：ETC 处理时序，ECC 处理因果，并用时序增强因果图；论文中 EventStoryLine 上 `BERT+SCITE` 基线 F1 为 `54.72` | 适合作为“从连接词走向文档级图”的入门论文 |
| *EiGC: An Event-Induced Graph with Constraints for Event Causality Identification* | 用事件诱导图、R-GCN 和约束推理处理事件因果识别 | 适合理解“为什么必须加全局约束” |

参考链接：

- https://link.springer.com/article/10.1007/s10115-022-01665-w
- https://www.mdpi.com/2079-9292/14/5/992/xml
- https://www.mdpi.com/2079-9292/13/23/4608
