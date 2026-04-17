## 核心结论

医疗知识图谱（Medical Knowledge Graph, Medical KG）是把“症状、疾病、检查、药物、不良反应、禁忌证”等医学对象，整理成“实体-关系-实体”的图结构。它和普通文档的区别不在于信息多少，而在于**关系被显式写出来**：谁提示谁、谁依赖谁、谁和谁冲突，都能在图上直接表达。

它的工程价值主要有三点。

1. 它能把诊断过程显式化。输入“发热、咳嗽、白细胞升高”，系统不是直接吐一个病名，而是沿着“症状 $\rightarrow$ 疾病 $\rightarrow$ 检查/药物”的路径给出判断依据。
2. 它能同时支持用药安全推理。药物节点之间可以接入药物相互作用（DDI, Drug-Drug Interaction，白话说就是“两种药一起用会不会出问题”）关系，用同一张图处理“该用什么药”和“这些药能不能一起用”。
3. 它比纯文本检索更可解释。临床系统最怕“黑盒答案”。知识图谱把答案拆成路径、子图、证据边，便于医生复核，也便于工程团队排错、审计和版本追踪。

一个最小图示可以写成：

| 输入 | 推理路径 | 输出 |
|---|---|---|
| 发热 + 咳嗽 | 症状 $\rightarrow$ 肺炎 | 候选诊断：肺炎 |
| 肺炎 + 胸片浸润影 | 症状/检查 $\rightarrow$ 肺炎 | 诊断置信度提高 |
| 肺炎 | 疾病 $\rightarrow$ 抗感染药物 | 候选治疗方向 |
| 阿奇霉素 + 华法林 | 药物 $\rightarrow$ DDI | 提示潜在相互作用风险 |

玩具例子里，患者只有“发热 + 咳嗽”，系统可能从症状节点走到“感冒、支气管炎、肺炎”三个疾病节点；如果再补充“胸片异常”或“CRP 升高”，肺炎路径得分会继续上升。这就是图谱推理的基本模式：**证据越完整，相关路径越密集，候选结论排序越靠前**。

真实工程里，图谱不会只停在“查路径”。它通常还会结合历史病例、临床指南、药典、实验室检查参考范围和药物相互作用库，对候选路径打分，再把高分子图返回给前端作为解释证据。换句话说，图谱不是单独替代检索、规则或模型，而是充当**医学知识的结构化骨架**。

---

## 问题定义与边界

先定义问题。医疗知识图谱要解决的不是“自动替代医生”，而是“把医疗知识结构化，并让诊断与用药推理更稳定、更可解释”。

最常见的实体（Entity，白话说就是图里的“节点”）包括：

| 实体类型 | 含义 | 例子 |
|---|---|---|
| 症状 | 患者主诉或体征 | 发热、咳嗽、胸痛 |
| 疾病 | 诊断对象 | 肺炎、哮喘、糖尿病 |
| 检查 | 化验或影像项目 | 血常规、CT、CRP |
| 指标 | 检查结果项 | 白细胞、血氧饱和度 |
| 药物 | 治疗药品 | 阿莫西林、华法林 |
| 不良反应 | 药物相关风险 | 出血、肝损伤 |
| 禁忌证 | 不适合使用某药的条件 | 妊娠、严重肝损害 |
| 人群特征 | 会影响风险或疗效的背景因素 | 老年、儿童、肾功能不全 |

常见关系（Relation，白话说就是“节点之间的连线含义”）包括：

| 关系 | 含义 | 例子 |
|---|---|---|
| `has_symptom` | 疾病有哪些症状 | 肺炎 `has_symptom` 发热 |
| `suggests_disease` | 某症状提示某疾病 | 咳嗽 `suggests_disease` 肺炎 |
| `requires_test` | 疾病建议做什么检查 | 肺炎 `requires_test` 胸片 |
| `has_finding` | 某检查结果支持某疾病 | 胸片浸润影 `has_finding` 肺炎 |
| `treated_by` | 疾病可由什么药治疗 | 肺炎 `treated_by` 阿莫西林 |
| `interacts_with` | 药物之间存在相互作用 | 华法林 `interacts_with` 阿奇霉素 |
| `contraindicated_for` | 某药不适用于某类患者 | 华法林 `contraindicated_for` 活动性出血 |
| `causes_adverse_event` | 药物可能引发某不良反应 | 华法林 `causes_adverse_event` 出血 |

如果把它写成一个最小诊疗闭环，图上通常至少要覆盖下面四类链路：

| 链路 | 作用 | 例子 |
|---|---|---|
| 症状 $\rightarrow$ 疾病 | 诊断召回 | 发热、咳嗽 $\rightarrow$ 肺炎 |
| 疾病 $\rightarrow$ 检查 | 补证据 | 肺炎 $\rightarrow$ 胸片、CRP |
| 疾病 $\rightarrow$ 药物 | 治疗推荐 | 肺炎 $\rightarrow$ 抗感染药 |
| 药物 $\rightarrow$ 风险 | 安全审查 | 阿奇霉素 $\rightarrow$ 华法林相互作用 |

边界也必须说清楚，否则项目会很快失控。

第一，图谱不是事实本身，而是“被整理后的医学知识”。如果知识来源过期、标注错误或不同指南相互冲突，图谱会把错误结构化，后果比普通文档更严重。

第二，图谱能覆盖的是“已知关系”，不擅长直接发现全新的病理机制。它更像结构化索引和推理框架，而不是无中生有的科学发现器。

第三，医疗数据有强隐私约束。患者病历、处方、检验结果常常不能直接离开医院系统，因此很多团队只能做脱敏、聚合统计或联邦同步。白话说，模型想吃到最好的数据，但合规不允许它随便拿。

第四，专业标注成本很高。通用文本分类可以靠大规模弱标注，医疗图谱不行。一个“症状是否提示疾病”的边，往往需要医生、药师或医学知识工程师共同确认。

第五，图谱输出的本质是“辅助建议”，不是法律意义上的诊断结论。系统可以提示“肺炎概率上升”“建议复查胸片”“阿奇霉素与华法林存在联用风险”，但不能把这类输出包装成“自动确诊”或“自动开药”。

所以工程上要先定边界：是做门诊分诊辅助，还是做住院用药安全；是做呼吸专科子图，还是全院通用图；是只覆盖常见病，还是接入罕见病知识。范围不收敛，图谱质量通常会崩。

---

## 核心机制与推导

知识图谱的基本存储单位是三元组：

$$
(h, r, t)
$$

其中 $h$ 是头实体，$r$ 是关系，$t$ 是尾实体。比如：

$$
(\text{肺炎}, \text{has\_symptom}, \text{发热})
$$

对诊断问题，输入通常是一组症状和检查结果。系统要做的是在图上寻找从输入节点到疾病节点的高分路径。一个简化思路是：

$$
\text{Score}(d)=\sum_{p \in P_d} w(p)
$$

这里 $P_d$ 是从当前证据集合走到疾病 $d$ 的所有候选路径，$w(p)$ 是每条路径的权重。白话说，不是只看“有没有边”，而是看“有哪些路径、每条路径可信度多高、路径之间能不能互相补强”。

如果把“基础患病概率”和“检查支持强度”也纳入，一个更接近工程实现的写法是：

$$
\text{Score}(d)=\alpha \cdot \text{Prior}(d)+\beta \sum_{s \in S} w(s,d)+\gamma \sum_{e \in E} w(e,d)
$$

其中：

- $S$ 是症状集合，如“发热、咳嗽、呼吸急促”
- $E$ 是检查证据集合，如“胸片浸润影、CRP 升高”
- $\text{Prior}(d)$ 是疾病先验，可以来自科室分布、季节流行情况或历史统计
- $w(s,d)$、$w(e,d)$ 分别表示症状和检查对疾病的支持强度
- $\alpha,\beta,\gamma$ 是工程上可调的权重

例如患者有“发热、咳嗽、胸痛、胸片浸润影”，可以得到：

- 发热 $\rightarrow$ 肺炎
- 咳嗽 $\rightarrow$ 肺炎
- 胸痛 $\rightarrow$ 胸膜炎
- 胸片浸润影 $\rightarrow$ 肺炎

如果“胸片浸润影”这条边权重很高，那么肺炎的总得分会明显高于普通感冒。这里可以把“胸片浸润影”理解为**高特异度证据**：不是所有呼吸道疾病都会出现它，所以它对区分疾病更有价值。

为了让新手更容易理解，可以把整件事看成“加权投票”：

| 证据 | 支持肺炎 | 支持感冒 | 说明 |
|---|---:|---:|---|
| 发热 | 0.6 | 0.5 | 两边都支持，但区分力一般 |
| 咳嗽 | 0.5 | 0.4 | 两边都支持，区分力仍不强 |
| 胸片浸润影 | 0.9 | 0.0 | 对肺炎更有区分力 |
| CRP 升高 | 0.4 | 0.1 | 炎症证据，但不是绝对特异 |

于是一个简化的结果可能是：

$$
\text{Score}(\text{肺炎}) = 0.6+0.5+0.9+0.4=2.4
$$

$$
\text{Score}(\text{感冒}) = 0.5+0.4+0.0+0.1=1.0
$$

这时系统不是说“100% 就是肺炎”，而是说“在当前证据下，肺炎路径得分更高，优先级更靠前”。

在知识图谱补全或关系预测任务里，常用 MAP（Mean Average Precision，平均精度均值，白话说就是“整体排序质量的平均分”）评估路径推理效果：

$$
MAP = \frac{1}{|Q_r|}\sum_{q \in Q_r} AP(q)
$$

其中 $Q_r$ 表示待评估关系集合，$AP(q)$ 表示某个关系查询的平均精度。这个指标的意义是：如果系统把真正相关的疾病、药物或关系排得更靠前，MAP 就更高。BMC Medical Informatics and Decision Making 2021 的开放获取论文就采用 MAP 评估医疗知识图谱补全效果，并报告其方法相对既有路径推理基线有平均提升。

为什么只靠“图结构”还不够？因为医学里很多关系带有语义细节。比如“咳嗽”对“感冒”和“肺炎”都可能成立，但“持续高热 + 胸片浸润影 + CRP 升高”对应肺炎的语义更强。所以很多系统会把路径文本、实体定义、临床描述编码成向量，再和图结构一起建模。常见做法是用 BERT 这类预训练模型把实体描述转成嵌入（Embedding，白话说就是“把词或句子压成可计算的数字向量”），再与图路径特征联合计算分数。

药物推理比诊断多一层复杂度。它不仅要回答“治什么”，还要回答“能不能一起用”。于是图上除了“疾病 $\rightarrow$ 药物”边，还要增加“药物 $\rightarrow$ 药物相互作用”边，以及药物与靶点、通路、不良反应、禁忌证之间的关系。对一组药物组合 $D=\{d_1,d_2,\dots,d_n\}$，系统常常抽取相关知识子图，再预测风险分数：

$$
Risk(D) = f(G_D)
$$

其中 $G_D$ 是这组药物对应的局部子图，$f$ 可以是图神经网络、关系表示学习模型或基于规则的打分函数。白话说，系统不是只看两种药名字是否撞上数据库，而是看它们在整个医学知识网络里的邻居关系、机制链条和历史证据。

如果再把患者背景因素考虑进去，可以写成：

$$
Risk(D, x)=f(G_D, x)
$$

其中 $x$ 代表患者特征，如年龄、肝肾功能、妊娠状态、既往出血史。因为同一对药物在不同人群中的风险并不相同。

玩具例子：

- 药物 A：抗凝药
- 药物 B：某抗生素
- 已知边：A `interacts_with` B
- 已知边：A `causes_adverse_event` 出血
- 患者特征：老年 + 肝功能异常

这时子图会提示：A 与 B 联用可能放大出血风险，而患者又属于高风险人群，因此系统应把该组合排进高危列表。

真实工程例子是 DDI 预测系统。Communications Medicine 2024 的 KnowDDI 论文把 DDI 图和外部生物医学知识图合并，再为每个目标药对学习一个“知识子图（knowledge subgraph）”，保留有解释价值的路径，并用这些路径解释预测结果。它的关键点不是只“查已知药对”，而是借助邻域知识和相似关系，对未知或稀疏记录的药物组合做更稳的推断。

所以从机制上看，医疗 KG 推理至少有三层：

| 层次 | 做什么 | 典型方法 |
|---|---|---|
| 结构层 | 建实体和关系 | 三元组、图数据库、规则 |
| 表示层 | 把节点和路径变成可计算特征 | Embedding、BERT、图表示学习 |
| 决策层 | 生成排序、风险和解释 | 路径打分、子图学习、规则校验 |

---

## 代码实现

下面用一个极小但可运行的例子，演示“症状推疾病，再扩展到药物，并检查药物相互作用”的流程。这个例子不是医疗建议，只是帮助理解知识图谱推理的最小闭环。

```python
from collections import defaultdict
from itertools import combinations


# 图谱边：head, relation, tail, weight
TRIPLES = [
    ("发热", "suggests_disease", "肺炎", 0.6),
    ("咳嗽", "suggests_disease", "肺炎", 0.5),
    ("呼吸急促", "suggests_disease", "肺炎", 0.7),
    ("胸片浸润影", "suggests_disease", "肺炎", 0.9),
    ("CRP升高", "suggests_disease", "肺炎", 0.4),

    ("发热", "suggests_disease", "感冒", 0.5),
    ("咳嗽", "suggests_disease", "感冒", 0.4),
    ("鼻塞", "suggests_disease", "感冒", 0.6),

    ("肺炎", "treated_by", "阿奇霉素", 0.8),
    ("肺炎", "treated_by", "阿莫西林", 0.7),
    ("感冒", "treated_by", "对乙酰氨基酚", 0.6),

    ("阿奇霉素", "interacts_with", "华法林", 0.95),
    ("阿奇霉素", "causes_adverse_event", "QT间期延长", 0.50),
    ("华法林", "causes_adverse_event", "出血", 0.90),
]

# 同义词归一化：解决“发烧/高热/体温39度”落不到同一节点的问题
ALIASES = {
    "发烧": "发热",
    "高热": "发热",
    "胸片异常": "胸片浸润影",
    "c反应蛋白升高": "CRP升高",
}

# 双向索引，方便按 head -> tail 或 tail <- head 查询
graph_out = defaultdict(list)
graph_in = defaultdict(list)

for head, relation, tail, weight in TRIPLES:
    graph_out[(head, relation)].append((tail, weight))
    graph_in[(tail, relation)].append((head, weight))

# DDI 在很多场景下要看成无向风险关系，这里补成双向
for head, relation, tail, weight in list(TRIPLES):
    if relation == "interacts_with":
        graph_out[(tail, relation)].append((head, weight))
        graph_in[(head, relation)].append((tail, weight))


def normalize_terms(terms):
    return [ALIASES.get(term, term) for term in terms]


def infer_diseases(observations):
    scores = defaultdict(float)
    evidence = defaultdict(list)

    for obs in normalize_terms(observations):
        for disease, weight in graph_out.get((obs, "suggests_disease"), []):
            scores[disease] += weight
            evidence[disease].append((obs, weight))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked, evidence


def recommend_drugs(disease, topn=2):
    drugs = sorted(
        graph_out.get((disease, "treated_by"), []),
        key=lambda item: item[1],
        reverse=True,
    )
    return drugs[:topn]


def check_ddi(drugs):
    risks = []
    for drug_a, drug_b in combinations(drugs, 2):
        for other, weight in graph_out.get((drug_a, "interacts_with"), []):
            if other == drug_b:
                adverse_events = [
                    event for event, _ in graph_out.get((drug_a, "causes_adverse_event"), [])
                ] + [
                    event for event, _ in graph_out.get((drug_b, "causes_adverse_event"), [])
                ]
                risks.append(
                    {
                        "pair": (drug_a, drug_b),
                        "weight": weight,
                        "events": sorted(set(adverse_events)),
                    }
                )
    return risks


def explain_disease(disease, evidence):
    parts = [f"{obs}({weight:.2f})" for obs, weight in evidence.get(disease, [])]
    return f"{disease}: " + " + ".join(parts)


def main():
    # 玩具患者：已有长期用药华法林，新出现呼吸道症状
    observations = ["发烧", "咳嗽", "胸片异常", "CRP升高"]
    existing_drugs = ["华法林"]

    diseases, evidence = infer_diseases(observations)
    assert diseases, "没有召回任何疾病，请检查输入或图谱覆盖范围。"

    best_disease, best_score = diseases[0]
    assert best_disease == "肺炎"

    candidate_drugs = [name for name, _ in recommend_drugs(best_disease, topn=2)]
    final_drugs = candidate_drugs + existing_drugs
    ddi_risks = check_ddi(final_drugs)

    assert "阿奇霉素" in candidate_drugs
    assert any(risk["pair"] == ("阿奇霉素", "华法林") or risk["pair"] == ("华法林", "阿奇霉素")
               for risk in ddi_risks)

    print("标准化输入:", normalize_terms(observations))
    print("候选疾病:", diseases)
    print("最佳疾病解释:", explain_disease(best_disease, evidence))
    print("候选药物:", candidate_drugs)
    print("合并既往用药后:", final_drugs)
    print("DDI 风险:", ddi_risks)
    print(f"结论: 当前证据下 {best_disease} 得分最高，为 {best_score:.2f}")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出逻辑分成四步：

1. 先做术语归一化，把“发烧”归到标准实体“发热”。
2. 再根据症状和检查结果累加疾病分数，得到候选诊断排序。
3. 选出高分疾病后展开候选药物。
4. 最后把候选药物和患者已有用药合并，检查是否存在 DDI 风险。

如果手动代入上面的输入，结果大致会是：

| 步骤 | 中间结果 | 含义 |
|---|---|---|
| 术语归一化 | 发烧 $\rightarrow$ 发热，胸片异常 $\rightarrow$ 胸片浸润影 | 保证落到统一节点 |
| 诊断打分 | 肺炎得分 $0.6+0.5+0.9+0.4=2.4$ | 肺炎排第一 |
| 治疗扩展 | 阿奇霉素、阿莫西林 | 从疾病节点走到药物节点 |
| 风险审查 | 阿奇霉素 + 华法林命中 DDI | 触发联用风险提醒 |

这个玩具例子的价值不在于医学完整性，而在于把一个完整工程链路拆清楚：

- 输入不是“文本直接生成答案”，而是先做实体标准化。
- 诊断不是单点判断，而是多条证据路径共同加权。
- 治疗推荐不是诊断结束后的附属功能，而是图谱中的另一跳。
- 用药安全不是额外外挂，而是同一张图里继续推理。

如果把它扩展成真实工程模块，数据结构通常会像这样：

| 模块 | 输入 | 输出 | 说明 |
|---|---|---|---|
| 图谱加载 | 三元组、词表、规则库 | 内存图/索引 | 负责读取知识 |
| 症状归一化 | 用户输入文本 | 标准症状实体 | 解决“发烧/高热”同义词问题 |
| 诊断召回 | 症状集合 | 候选疾病列表 | 先做广召回 |
| 路径重排 | 候选疾病 + 路径特征 | 排序后的疾病 | 融合语义与规则 |
| 治疗扩展 | 疾病 | 候选药物/检查 | 基于图谱邻接关系 |
| 安全审查 | 药物列表 | DDI/禁忌提示 | 检查联用风险 |
| 解释输出 | 路径、证据边 | 前端可读说明 | 给医生或用户查看 |

真实工程例子可以想象成门诊辅助系统。

患者主诉“发热 3 天，咳嗽，呼吸急促”，系统先做实体抽取，识别出标准症状节点；然后在呼吸系统子图中召回肺炎、支气管炎、流感；接着结合“血氧低、胸片浸润影”把肺炎排到前面；最后扩展到“建议检查”和“候选药物”，并检查如果患者已有华法林处方，是否与新增药物发生风险冲突。前端不是只显示“可能是肺炎”，而是显示“发热、咳嗽、胸片浸润影三条证据共同支持肺炎；拟用阿奇霉素时需注意与华法林联用风险”。

这一步很关键，因为医疗系统最重要的不是“给答案”，而是“给可审查的答案”。

---

## 工程权衡与常见坑

医疗 KG 项目最常见的问题，不是模型不够深，而是数据和流程不够稳。

先看几个主要权衡：

| 维度 | 做法 A | 做法 B | 典型取舍 |
|---|---|---|---|
| 诊断能力 | 规则优先 | 学习优先 | 规则稳但覆盖差，学习强但更依赖数据 |
| 数据来源 | 公开知识库 | 医院真实数据 | 公开数据易启动，真实数据更有价值但合规更难 |
| 图谱规模 | 全科通用 | 专科垂直 | 全科复杂度高，专科更容易做深 |
| DDI 建模 | 查表匹配 | 子图学习 | 查表简单，子图学习能覆盖未知组合 |
| 更新方式 | 批量更新 | 增量同步 | 批量稳定，增量更实时但更复杂 |
| 解释形式 | 文本摘要 | 路径/子图证据 | 文本更易读，子图更可审计 |
| 部署位置 | 中心化服务 | 院内本地部署 | 中心化维护方便，院内部署更易合规 |

常见坑主要有五类。

第一，实体归一化失败。  
同一个意思在真实输入里可能写成“发烧、发热、高热、体温 39 度”。如果没有标准化，图上会产生很多碎节点，召回率会立刻下降。新手最容易忽略这一点，以为图谱问题出在模型，实际上问题常常出在入口词表。

第二，图谱边“看起来很多”，但真正可用于推理的边很稀疏。  
特别是 DDI 数据，经常只覆盖常见药对。到了临床组合用药场景，缺边非常严重。这就是为什么很多系统会做相似性传播或 KG 补全。白话说，已知拼图太少，只能用已知相似关系去补缺口。

第三，解释路径不等于因果证明。  
图上存在“症状 $\rightarrow$ 疾病”的边，不代表患者就一定是这个病。很多关系本质上是统计相关或指南建议，不是严格因果。前端文案必须写成“候选依据”而不是“自动确诊”。

第四，知识版本漂移。  
医学指南、药品说明书、禁忌证会更新。如果图谱版本落后，而系统还在给实时建议，风险很高。工程上必须给边打上来源、版本和生效时间。

第五，隐私治理后补通常来不及。  
很多团队一开始先拉病历数据做效果，后面才想脱敏和审计。医疗项目这样做基本会踩线。正确顺序是先定数据最小化原则、脱敏策略、访问审计和回溯机制，再开模型流程。

把这些问题再压缩成一个实用避坑清单：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 同义词未合并 | 召回下降 | 建标准术语映射表 |
| 关系来源不明 | 无法审计 | 每条边记录来源、版本、时间 |
| DDI 边过少 | 风险漏报 | 接外部 KG，做补全或相似传播 |
| 用训练集指标替代临床验证 | 结果不可信 | 做专家评审、离线回放和病例复核 |
| 输出像“最终诊断” | 合规风险 | 强制标注为辅助建议 |
| 图谱过大但无人维护 | 很快失效 | 先做专科子图，再逐步扩展 |
| 忽略患者个体差异 | 风险评估失真 | 把年龄、妊娠、肝肾功能并入风险层 |

如果只记一个判断标准，可以记这个：**医疗 KG 失败时，通常不是因为“图不够大”，而是因为“词不统一、边不可信、版本不可追踪、输出不可审计”**。

---

## 替代方案与适用边界

知识图谱不是唯一方案。至少还有两类常见替代路线。

第一类是纯文本检索。  
做法是把指南、论文、药典、病例说明书切片后检索，再让模型总结。优点是启动快，不需要先建图；缺点是结构不稳定，答案容易随检索片段变化，解释也常停留在“我看到了哪几段文本”。

第二类是 Hybrid，也就是 KG + LLM。  
KG 负责提供稳定结构和显式证据，LLM 负责把路径解释成自然语言、做术语归一化、补足模糊表达。这个方向在工程上很实用，因为它兼顾可解释性和交互体验。

三种路线可以这样比较：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 纯知识图谱 | 结构稳定、可解释、便于审计 | 建设成本高、覆盖依赖标注 | 临床辅助决策、药物安全审查 |
| 纯文本检索/RAG | 上线快、数据准备简单 | 结构弱、一致性差 | 原型验证、知识问答 |
| KG + LLM | 解释自然、兼顾结构与交互 | 系统更复杂，评估更难 | 面向医生或患者的智能助手 |

什么时候优先上 KG？

- 需要审计路径，必须回答“为什么推荐这个结论”。
- 涉及药物联用、禁忌证、检查流程等强结构化知识。
- 需要长期维护领域知识，而不是一次性问答。
- 输出需要经过专家复核，不能只给自然语言总结。

什么时候不必先上 KG？

- 只是做内容检索或医学科普问答。
- 数据源还非常散，没有稳定术语体系。
- 团队没有医学标注和知识工程能力，短期目标只是验证需求。
- 业务暂时不要求审计和版本追踪。

更现实的做法通常是分阶段：

1. 先用检索系统验证真实需求，确定高频问题和高风险场景。
2. 再把高频问答和高风险流程抽成图谱，先做专科或单病种子图。
3. 最后让 LLM 做语言层交互，KG 做事实层约束和证据输出。

这样可以避免一开始就建设“大而全”的医疗图谱，结果半年后没人维护。Clinical Therapeutics 2024 的两篇综述也基本指向同一个工程结论：知识图谱在药物警戒和 DDI 场景有潜力，但真正落地时，必须先明确用例、数据来源、构图方法和可解释输出形式，而不是只追求模型复杂度。

---

## 参考资料

| 资料 | 要点 | 用途 |
|---|---|---|
| Hauben M, Rafi M. *Knowledge Graphs in Pharmacovigilance: A Step-By-Step Guide*. Clinical Therapeutics, 2024. DOI: `10.1016/j.clinthera.2024.03.006` | 用通俗步骤解释 KG 在药物警戒中的落地流程：先定用例，再选数据、构图、嵌入和输出 | 适合新手建立工程全景 |
| Hauben M, Rafi M, Abdelaziz I, Hassanzadeh O. *Knowledge Graphs in Pharmacovigilance: A Scoping Review*. Clinical Therapeutics, 2024. DOI: `10.1016/j.clinthera.2024.06.003` | 综述了 47 篇相关研究，指出 KG 已用于单药不良反应和 DDI 预测，但与传统方法的系统对比仍不足 | 用于把握医疗/药物安全 KG 的应用边界 |
| Lan Y, He S, Liu K, et al. *Path-based knowledge reasoning with textual semantic information for medical knowledge graph completion*. BMC Medical Informatics and Decision Making, 2021. DOI: `10.1186/s12911-021-01622-7` | 把 BERT 文本语义和路径推理结合，使用 MAP 评估医疗 KG 补全效果，并强调路径可解释性 | 用于理解“症状到疾病”的路径推理与评估 |
| Wang Y, Yang Z, Yao Q. *Accurate and interpretable drug-drug interaction prediction enabled by knowledge subgraph learning*. Communications Medicine, 2024. DOI: `10.1038/s43856-024-00486-y` | 提出 KnowDDI，把 DDI 图和外部生物医学知识图连接起来，为每个药对学习可解释知识子图 | 用于理解真实工程中的药物推理与 DDI 预测 |
| PubMed: KnowDDI 论文索引页 | 提供论文元数据、摘要和定位入口 | 用于快速确认原始研究信息 |
| PubMed: Clinical Therapeutics 两篇 2024 综述页 | 提供 PMID、DOI、摘要和作者信息 | 用于核对文献元数据 |

参考链接：

- Clinical Therapeutics 2024 Step-By-Step Guide: https://pubmed.ncbi.nlm.nih.gov/38670887/
- Clinical Therapeutics 2024 Scoping Review: https://pubmed.ncbi.nlm.nih.gov/38981792/
- BMC 2021 全文开放获取页: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01622-7
- Communications Medicine 2024 KnowDDI: https://www.nature.com/articles/s43856-024-00486-y
- KnowDDI 代码仓库: https://github.com/LARS-research/KnowDDI
