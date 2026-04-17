## 核心结论

Visual CoT 的核心不是“让模型多说几步”，而是把推理顺序改成“先找图，再说理”。CoT，Chain of Thought，指把中间推理步骤显式写出来；Visual CoT 则进一步要求：每一步文本推理，都要先绑定到图像中的局部证据，再继续往下推。

这件事的直接价值有两点。

1. 可解释性更强。答案不是凭全图印象直接生成，而是能回溯到具体区域。
2. 复杂场景更稳。模型不容易只靠数据分布里的“常见答案”去猜，而是被迫检查局部证据。

把它写成一个简化公式，可以表示为：

$$
A = \mathrm{Decode}\Big(\mathrm{TextCoT}\big(\{h_i\}_{i=1}^T,\{r_i\}_{i=1}^T,Q\big)\Big)
$$

其中：

- $r_i$ 表示第 $i$ 步对应的视觉区域，可能是 bbox、mask 或 patch 集合。
- $h_i$ 表示第 $i$ 步的中间推理状态。
- $Q$ 是问题。
- $A$ 是最终答案。

如果进一步强调“先定位再推理”，可以写成两段式：

$$
r_i \sim p(r_i \mid I, Q, h_{i-1})
$$

$$
h_i = f_\theta(h_{i-1}, \mathrm{Feat}(I, r_i), Q)
$$

意思很直接：先根据图像 $I$、问题 $Q$ 和上一步状态 $h_{i-1}$ 选出下一块该看的区域 $r_i$，再用这块区域的特征更新推理状态。

这里的 grounding，直白说就是“让文字结论能落回图上的具体位置”。没有 grounding，模型可能答对了，但并不知道自己到底看了哪里，也无法证明答案依赖了正确证据。

一个最容易理解的例子是“图片中的故障灯亮了吗？”普通多模态模型可能直接回答“亮了”；Visual CoT 会把过程拆开：

1. 先定位仪表盘区域。
2. 再定位告警灯区域。
3. 检查告警灯颜色和亮起状态。
4. 最后把这些视觉证据转成文字结论：“亮了，且属于告警状态”。

简化流程如下：

```text
Question
  ↓
Visual Step 1 -> Region 1 -> "这是仪表盘"
  ↓
Visual Step 2 -> Region 2 -> "这是红色故障灯，且处于亮起状态"
  ↓
Text Step 3 -> "因此故障灯亮了"
  ↓
Answer
```

这里最关键的约束不是“步骤多”，而是“文字结论必须由前面的视觉步骤支撑”。如果第 2 步没有找到告警灯区域，后面的“故障灯亮了”就不应被当成高可信答案。

---

## 问题定义与边界

Visual CoT 解决的核心问题，是多模态推理中“看哪里”和“怎么解释”经常脱节。传统文本 CoT 擅长把答案写成步骤，但在图像任务里，这些步骤未必真的依赖图中局部证据。模型有可能只看了一眼全图，再根据训练语料生成一段“像解释的解释”。

因此，Visual CoT 的问题定义可以写成：

给定图像 $I$ 和问题 $Q$，模型不仅要输出答案 $A$，还要输出一串视觉步骤 $\{r_i\}_{i=1}^T$ 与对应的文本推理状态 $\{h_i\}_{i=1}^T$，并且要求后续文本推理显式依赖这些区域证据。

形式化地说：

$$
\mathcal{Y} = \{A, (r_1,h_1), (r_2,h_2), \dots, (r_T,h_T)\}
$$

并要求满足依赖关系：

$$
h_i \not\!\perp r_i \mid (Q, h_{i-1})
$$

这行式子的直白含义是：第 $i$ 步文本状态 $h_i$ 不能在条件上独立于对应区域 $r_i$。也就是说，文本步骤必须真实使用该区域，而不是事后补一句“我看了这里”。

它适合的任务主要是“答案依赖局部视觉证据，且最好能解释”的问题。

| 任务类型 | 是否适合 Visual CoT | 原因 |
|---|---|---|
| 视觉问答（VQA） | 适合 | 问题通常能拆成若干局部证据步骤 |
| 医学影像诊断 | 很适合 | 需要病灶定位与诊断理由对应 |
| 图表/文档理解 | 适合 | 需要按区域读取标题、数值、标注 |
| 多目标交通场景分析 | 适合 | 需要同时定位信号灯、车辆、停止线等 |
| 纯文本问答 | 不适合 | 没有视觉区域可 grounding |
| 纯分类且无需解释 | 价值有限 | 直接分类更便宜 |
| 无明确局部证据的审美判断 | 边际收益小 | 很难定义可靠 bbox 链 |

bbox，bounding box，直白说就是“用坐标圈出一个矩形框”，通常写作 $(x_1, y_1, x_2, y_2)$。如果对象边界不规则，也可能用 mask；如果视觉编码器本身按 patch 工作，也可能直接返回 patch 索引集合。

新手最容易理解的边界是医学影像。问题是“这张胸片是否提示肺部病变？”如果系统不能先指出疑似病灶区域，再解释“该区域存在异常高密度影，因此支持某种诊断”，那它给出的就只是语言层面的解释，不是可核查证据链。

常规 CoT 与 Visual CoT 的差异可以压缩成下面这张表：

| 模型阶段 | 常规 CoT 的视觉输入 | Visual CoT 的视觉输入 | 结果 |
|---|---|---|---|
| 输入阶段 | 全图一次性输入 | 全图 + 分步局部区域 | 后者更适合局部证据任务 |
| 中间推理 | 主要是文本步骤 | 区域步骤 + 文本步骤 | 后者更可追溯 |
| 输出阶段 | 文本答案 | 文本答案 + 证据链 | 后者更可审计 |
| 错误定位 | 难判断错在“看”还是“想” | 可拆成定位错或推理错 | 后者更利于调试 |

判断是否应该上 Visual CoT，可以先问三个问题：

| 判断问题 | 如果回答是“是” | 含义 |
|---|---|---|
| 答案是否依赖局部区域？ | 更适合 | 说明需要分步找证据 |
| 错误是否需要追责或复核？ | 更适合 | 说明需要可审计链条 |
| 是否能定义中间视觉步骤？ | 更适合 | 说明链条可以落地，而不是空想 |

---

## 核心机制与推导

Visual CoT 的关键机制，是把“区域选择”纳入推理循环，而不是把检测器当作一个可有可无的前处理模块。

先定义几个对象：

- `VisualFeatures`：图像编码后的局部特征。直白说，就是图像切成很多小块后，每一块对应的机器表示。
- $h_{i-1}$：上一步推理状态。可以理解成“模型当前已经确认了什么，还缺什么”。
- $r_i$：第 $i$ 步要看的区域。
- $v_i = \mathrm{Pool}(VisualFeatures, r_i)$：从区域 $r_i$ 中抽取出的区域特征。
- $h_i$：融合了问题、历史状态和当前区域特征后的新推理状态。

一个最常见的简化过程是：

### 1. 按当前状态选择下一块区域

$$
s_i = \mathrm{Attention}(h_{i-1}, VisualFeatures)
$$

$$
r_i = \mathrm{Locate}(s_i)
$$

第一步先得到各局部位置与当前推理状态的相关性分数 $s_i$，第二步把这些分数变成显式区域 $r_i$。如果模型输出的是 bbox，那么 `Locate` 就是一个框回归头；如果输出的是 patch 集合，那么 `Locate` 可以是 top-k patch 选择。

### 2. 从该区域抽取视觉证据

$$
v_i = \mathrm{Pool}(VisualFeatures, r_i)
$$

这一步的意思是：不要让文本模块继续依赖全图，而是只看当前步骤真正选中的区域证据。这里的 `Pool` 可以是 ROI Align、mask pooling，也可以是简单的 patch 平均。

### 3. 用区域证据更新推理状态

$$
h_i = \mathrm{TextDecoder}(Q, h_{i-1}, v_i)
$$

也可以写成更强调多模态融合的形式：

$$
h_i = f_\theta([Q;h_{i-1};v_i])
$$

其中 $[\,]$ 表示拼接或融合。直白理解就是：模型先看问题，再看上一步已经确定的内容，最后结合当前区域证据，得到下一步结论。

### 4. 在最后一步输出答案

$$
A = \mathrm{Decode}(h_T)
$$

如果任务还要求显式解释，可以把解释链一起输出：

$$
E = \{(r_i, \text{reason}_i)\}_{i=1}^{T}
$$

于是最终系统输出变成：

$$
\mathrm{Output} = (A, E)
$$

这类模型训练时，通常也不只优化答案损失。一个常见思路是联合训练答案、区域和链条一致性：

$$
\mathcal{L} = \lambda_a \mathcal{L}_{ans}
+ \lambda_r \mathcal{L}_{region}
+ \lambda_c \mathcal{L}_{consistency}
$$

其中：

- $\mathcal{L}_{ans}$：答案监督。
- $\mathcal{L}_{region}$：区域定位监督，比如 bbox 回归或 mask 损失。
- $\mathcal{L}_{consistency}$：链条一致性约束，防止文本推理脱离已选区域。

如果没有最后这类一致性约束，模型就很容易出现“区域是一个故事，答案是另一个故事”的问题。

把整条逻辑写成序列，更容易看清：

```text
Question + Image
  ↓
Initialize reasoning state h0
  ↓
r1 = Locate(h0, VisualFeatures)
  ↓
h1 = Update(Q, h0, Feat(r1))
  ↓
r2 = Locate(h1, VisualFeatures)
  ↓
h2 = Update(Q, h1, Feat(r2))
  ↓
...
  ↓
A = Decode(hT)
```

交通场景的玩具例子很适合说明这个循环：

1. 第 1 步：定位信号灯区域，判断当前是红灯还是绿灯。
2. 第 2 步：定位停止线区域。
3. 第 3 步：定位车辆前轮区域，检查是否越过停止线。
4. 最终结论：若红灯且前轮越线，则构成闯红灯；否则不构成。

这类任务里，任何一步找错都会污染后续结论。例如第 1 步没有正确定位信号灯，那么“当前是红灯”就不应进入高置信推理链。

真实工程里，医学影像是更典型的落地场景。比如系统要回答“这张片子是否提示某种病变？”一个更可信的链条是：

1. 先定位疑似病灶区域。
2. 再读取病灶的形态或密度特征。
3. 最后把局部证据映射成医学文本结论。

这样做的实际好处，不只是“看起来更解释”，而是调试时可以分解故障来源：

- 定位错了。
- 定位对了，但区域特征抽取错了。
- 区域特征没错，但文本推理错了。

这比“全图输入后直接答错”更容易排查。

---

## 代码实现

工程上，一个最小可用的 Visual CoT 原型通常包含三块：

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| Region Extractor | 给出候选区域或直接定位目标区域 | 图像、问题、当前状态 | bbox / region feature |
| Multimodal Fusion | 融合区域证据与已有推理状态 | 上一步状态、region feature、问题 | 新推理状态 |
| Text Decoder | 生成中间解释与最终答案 | 推理状态 | 文本步骤 / 最终答案 |

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只模拟“先选区域，再更新推理状态，再给答案”的控制流。重点不是模型精度，而是把 Visual CoT 的执行顺序写清楚。

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Region:
    name: str
    bbox: BBox
    attrs: Dict[str, object]


@dataclass
class Evidence:
    step: int
    region: str
    bbox: BBox
    attrs: Dict[str, object]
    note: str


@dataclass
class State:
    question: str
    evidence: List[Evidence] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    confidence: float = 0.0


class VisualCoTAgent:
    def __init__(self, image_regions: List[Region]) -> None:
        self.region_map = {region.name: region for region in image_regions}

    def locate(self, target_name: str) -> Region:
        if target_name not in self.region_map:
            raise ValueError(f"region not found: {target_name}")
        return self.region_map[target_name]

    def attend(self, state: State, region: Region, step: int, note: str) -> State:
        state.evidence.append(
            Evidence(
                step=step,
                region=region.name,
                bbox=region.bbox,
                attrs=dict(region.attrs),
                note=note,
            )
        )
        return state

    def decode_step(self, state: State) -> State:
        names = {e.region for e in state.evidence}

        if "dashboard" in names and not any("已定位仪表盘" in r for r in state.reasoning):
            state.reasoning.append("已定位仪表盘区域，后续判断限定在仪表盘局部证据内。")

        if "warning_light" in names:
            warning = next(e for e in state.evidence if e.region == "warning_light")
            is_red = bool(warning.attrs.get("is_red", False))
            is_on = bool(warning.attrs.get("is_on", False))

            if is_red and is_on:
                state.reasoning.append("已定位到红色告警灯，且该区域处于点亮状态。")
                state.answer = "故障灯亮了，且属于告警状态。"
                state.confidence = 0.95
            elif is_red and not is_on:
                state.reasoning.append("已定位到红色告警灯，但当前未点亮。")
                state.answer = "故障灯未亮。"
                state.confidence = 0.88
            else:
                state.reasoning.append("已定位到相关灯位，但它不是当前问题对应的红色告警灯。")
                state.answer = "没有足够证据证明故障灯亮起。"
                state.confidence = 0.60
        else:
            state.reasoning.append("缺少告警灯区域证据，不能直接下结论。")
            state.answer = "证据不足。"
            state.confidence = 0.20

        return state

    def run(self, question: str, visual_plan: List[Tuple[str, str]]) -> State:
        state = State(question=question)

        for step_id, (region_name, note) in enumerate(visual_plan, start=1):
            region = self.locate(region_name)
            self.attend(state, region, step=step_id, note=note)

        return self.decode_step(state)


def main() -> None:
    image_regions = [
        Region("dashboard", (10, 10, 120, 90), {"type": "panel"}),
        Region("warning_light", (70, 30, 85, 45), {"is_red": True, "is_on": True}),
    ]

    plan = [
        ("dashboard", "先确认问题涉及的观察对象是仪表盘。"),
        ("warning_light", "再检查仪表盘上的红色告警灯是否亮起。"),
    ]

    agent = VisualCoTAgent(image_regions=image_regions)
    state = agent.run(question="图片中的故障灯亮了吗？", visual_plan=plan)

    assert state.answer == "故障灯亮了，且属于告警状态。"
    assert len(state.evidence) == 2
    assert state.evidence[1].bbox == (70, 30, 85, 45)
    assert state.confidence > 0.9

    print("Answer:", state.answer)
    print("Confidence:", state.confidence)
    print("Reasoning:")
    for item in state.reasoning:
        print("-", item)
    print("Evidence:")
    for ev in state.evidence:
        print(f"- step={ev.step}, region={ev.region}, bbox={ev.bbox}, note={ev.note}")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出会类似：

```text
Answer: 故障灯亮了，且属于告警状态。
Confidence: 0.95
Reasoning:
- 已定位仪表盘区域，后续判断限定在仪表盘局部证据内。
- 已定位到红色告警灯，且该区域处于点亮状态。
Evidence:
- step=1, region=dashboard, bbox=(10, 10, 120, 90), note=先确认问题涉及的观察对象是仪表盘。
- step=2, region=warning_light, bbox=(70, 30, 85, 45), note=再检查仪表盘上的红色告警灯是否亮起。
```

这个实现刻意保留了 Visual CoT 最重要的工程约束：

1. 先执行视觉计划，再执行文本判断。
2. 文本结论只能读取已收集的区域证据。
3. 如果缺证据，就输出“证据不足”，而不是硬编答案。

这就是 faithful reasoning 的基本要求。faithful 直白说就是“推理过程真的依赖了证据”，而不是答案先出来，再补一段看似合理的解释。

如果把它写成更接近真实模型的伪代码，通常是这样：

```python
state = init_state(question)

for step in range(max_steps):
    region = region_extractor(image, question, state)
    region_feat = roi_pool(visual_features, region)
    state = multimodal_fusion(state, region_feat, question)

answer = text_decoder(state)
```

这几个函数在真实系统中的常见对应关系如下：

| 伪代码函数 | 真实实现可能是什么 |
|---|---|
| `region_extractor` | 检测器、grounding 模型、cross-attention 定位头 |
| `roi_pool` | ROI Align、mask pooling、patch gather |
| `multimodal_fusion` | cross-attention、Q-Former、MLP + Transformer |
| `text_decoder` | LLM 自回归解码头或分类头 |

如果要把这个玩具实现进一步扩展到真实原型，通常要补三件事：

| 扩展点 | 为什么需要 |
|---|---|
| 候选区域打分 | 真实场景不会提前知道该看哪里 |
| 区域置信度 | 避免低质量定位直接污染文本推理 |
| 步骤终止条件 | 防止模型无意义地一直找下一个区域 |

---

## 工程权衡与常见坑

Visual CoT 最常见的问题，不是“答错”，而是“看错了还能说得很像对”。这在医疗、安全、工业质检里尤其危险，因为系统会给出一种形式完整、但证据错位的解释。

常见坑可以直接展开看：

| 坑名称 | 典型表现 | 为什么危险 | 常见缓解手段 |
|---|---|---|---|
| 区域错位 | bbox 没框到真正目标，文本却继续推理 | 错误证据会污染后续全部步骤 | 专家框校验、置信度阈值、候选框重排 |
| 事后解释 | 先出答案，再补“我看了这里” | 推理链不可信，不可审计 | 强制“先视觉后文本”的训练和解码顺序 |
| 全图捷径 | 模型偷偷靠全图统计特征答题 | 局部链条沦为装饰 | 遮挡测试、region ablation、随机打乱区域顺序 |
| 步骤漂移 | 前一步还在看信号灯，下一步突然跳去天空 | 链条断裂，解释失真 | 模板约束、状态机约束、步骤监督 |
| 标注噪声 | 自动 bbox 偏移严重 | 会把噪声学成“正确证据” | 混合专家标注、自训练过滤、困难样本复核 |
| 推理过长 | 步骤多、延迟高、错误累积 | 实时性差，链条越长越脆弱 | 限制最大步数，先粗后细，两阶段检索 |
| 证据冲突 | 不同步骤给出矛盾区域结论 | 最终答案不稳定 | 加一致性校验和冲突仲裁模块 |

新手可以把这些坑理解成一句话：模型不只会“瞎说”，还会“瞎看之后说得很像有道理”。

一个很实用的工程思路，是把错误拆成两类来监控：

| 错误类型 | 现象 | 处理方式 |
|---|---|---|
| 定位错误 | 没找到正确区域或框偏了 | 优先改检测器、grounding 头、标注质量 |
| 推理错误 | 区域对了，但结论错了 | 优先改融合模块、提示模板、解码策略 |

这比只看“最终答对率”更有价值，因为最终答对率会掩盖很多失败模式。

另一个关键问题是 fidelity。这里可以把 fidelity 理解成“解释和真实证据的一致程度”。一个常见但危险的误区是只检查答案是否正确，而不检查链条是否可信。实际系统里，更应该同时看三类指标：

| 指标 | 看什么 | 只看它为什么不够 |
|---|---|---|
| Answer Accuracy | 最终答案对不对 | 可能答对但理由错 |
| Grounding Quality | 框或 mask 准不准 | 可能框对了但推理错 |
| Chain Fidelity | 文本步骤是否真实依赖区域 | 这是防止“事后编理由”的关键 |

从成本角度看，Visual CoT 天然比“全图 + 问题直接回答”更慢。额外开销主要来自：

1. 需要做显式定位。
2. 需要多轮区域特征提取。
3. 需要保存和消费中间状态。
4. 需要更复杂的监督数据。

所以在端侧部署、实时监控、低算力设备上，完整多步链条未必划算。工程里更常见的是两阶段近似：

```text
Stage 1: 轻量检测器先给 3~5 个候选区域
Stage 2: 语言模块只对最相关的 1~2 个区域做推理
```

这样做牺牲了一部分完整链条，但能把延迟和成本压下来。

---

## 替代方案与适用边界

不是所有多模态任务都值得上 Visual CoT。它最适合“局部证据决定答案，且答案需要解释”的场景。如果任务更看重吞吐、成本或实现简单性，替代方案往往更合适。

可以把几类典型方案并列比较：

| Approach | 典型适用 | 解释性 | 资源开销 | 局限 |
|---|---|---|---|---|
| Visual CoT | 医疗 QA、安全审计、复杂 VQA | 高 | 高 | 标注和推理成本高 |
| Direct Fusion | 通用图文问答、快速原型 | 低到中 | 低到中 | 很难审计模型到底看了哪里 |
| Implicit Attention | 分类、检索、粗粒度理解 | 中 | 中 | 有注意力但不显式输出证据链 |
| Detector + Rule | 固定流程工业检测 | 高 | 中 | 灵活性差，迁移困难 |
| Retriever + VLM | 文档、图表、多页场景 | 中到高 | 中到高 | 依赖检索质量，不一定有细粒度步骤 |

几种方案的直白差别可以这样理解。

Direct Fusion 是：

```text
全图 + 问题 -> 直接输出答案
```

实现最简单，适合快速验证需求。但它通常回答得快，不一定回答得可查。

Visual CoT 是：

```text
全图 + 问题
  -> 先找区域
  -> 再写中间依据
  -> 最后输出答案
```

实现更重，但它把“答案从哪里来”这件事显式化了。

Implicit Attention 处在中间地带。模型内部可能确实关注了某些区域，但不会把这些区域步骤吐出来。对于一般消费级应用，它经常已经够用；但在高风险业务里，这类“内部看过、外部说不清”的解释通常不够。

Detector + Rule 则适合高度固定的工业场景。例如：

- 先检测表计指针。
- 再读取刻度位置。
- 最后按规则判断是否超阈值。

这种方案解释性也很强，但前提是任务结构非常稳定。一旦场景开放化、问题多样化，它就会失去灵活性。

所以适用边界可以明确成下面这张表：

| 场景特征 | 更推荐的方案 |
|---|---|
| 必须可解释、可复核、可追责 | Visual CoT |
| 只要快速可用、成本敏感 | Direct Fusion |
| 需要一定解释，但不要求显式证据链 | Implicit Attention |
| 场景固定、规则明确、类别有限 | Detector + Rule |
| 图像数量多、页面长、先找页再找框 | Retriever + VLM 或分层 Visual CoT |

最后可以用一个简单判断规则收束：

- 如果任务答案依赖几个关键局部区域，而且错了之后必须解释为什么错，优先 Visual CoT。
- 如果任务只是“这是不是猫”“这张图大概讲什么”，通常没必要上 Visual CoT。
- 如果既没有可靠局部标注，也无法清楚定义中间视觉步骤，不要强行套 Visual CoT。这样通常只会抬高系统复杂度，却拿不到真实收益。

---

## 参考资料

下面的资料可以按“先总览、再实现、再特定领域”的顺序阅读。

| 资料 | 作用 | 阅读重点 |
|---|---|---|
| Visual CoT 论文：Hao Shao 等，*Visual CoT: Unleashing Chain-of-Thought Reasoning in Multi-Modal Language Models* | 这是概念入口 | 看作者如何把 bbox 级中间步骤纳入推理链，以及为什么仅有文本 CoT 不够 |
| Visual CoT 项目仓库 | 这是工程入口 | 看数据组织、训练流程、标注格式、推理可视化方式 |
| V2T-CoT：Yuan Wang 等，*V2T-CoT: From Vision to Text Chain-of-Thought for Medical Reasoning and Diagnosis* | 这是垂直领域案例 | 看医学场景里如何把病灶定位和文本诊断绑定 |
| S-Chain：*Structured Visual Chain-of-Thought for Medicine* | 这是结构化监督案例 | 看结构化视觉链如何提升 grounding fidelity 与鲁棒性 |

链接如下：

- Visual CoT 论文：https://www.catalyzex.com/paper/visual-cot-unleashing-chain-of-thought
- Visual CoT 项目仓库：https://github.com/deepcs233/Visual-CoT
- V2T-CoT：https://papers.miccai.org/miccai-2025/0993-Paper1207.html
- S-Chain：https://s-chain.github.io/

如果你是第一次读这一方向，建议按下面的顺序理解，而不是一上来就看模型细节：

1. 先弄清楚 Visual CoT 解决的不是“让模型写长答案”，而是“让中间推理绑定局部证据”。
2. 再区分三个对象：区域 $r_i$、区域特征 $v_i$、文本状态 $h_i$。这三者混在一起，新手最容易看晕。
3. 最后再看训练与评测。真正困难的地方，通常不是“怎么把框画出来”，而是“如何证明文本步骤确实用了这个框”。

从工程视角看，阅读资料时最值得关注的不是论文 headline，而是下面四个具体问题：

| 问题 | 为什么重要 |
|---|---|
| 中间区域是人工标注、自动生成，还是混合得到？ | 它决定了链条质量上限 |
| 文本步骤是否被约束为显式依赖区域？ | 它决定了解释是否 faithful |
| 评测是否同时报告答案与 grounding 指标？ | 只看答案很容易误判系统质量 |
| 推理链是否支持失败分析？ | 这是它在生产环境里的实际价值 |
