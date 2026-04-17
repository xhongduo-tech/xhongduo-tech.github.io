## 核心结论

大海捞针测试（Needle In A Haystack，简称 NIAH）测的不是“模型懂不懂知识”，而是**模型能不能在超长输入里把一个小而精确的事实找出来**。这里的“长上下文”通常指一次性送入模型的 16K、64K、128K 甚至更长的 token 序列；“needle” 是被插入的一条唯一事实；“haystack” 是围绕它的大量无关文本。

把它理解成一次封闭环境检索最准确。你给模型一本很长的电子文档，在某一页插入一句“内部审计窗口为 14 天”，然后只问它这句话是什么，或者它大致在哪里。模型答对，只能说明它在长文本里完成了**定位 + 读取 + 复述**；这不等于它真正理解了审计制度，也不等于它在复杂推理上同样强。

工程上最重要的三个结论不变：

| 结论 | 含义 | 工程影响 |
|---|---|---|
| 上下文越长，检索越难 | 干扰项增多，相关线索更容易被稀释 | “支持 128K” 不等于 “128K 任意位置都稳定可用” |
| needle 越靠中间，越容易丢 | 典型现象是 `lost-in-the-middle` | 关键事实不要无提示地埋在中段 |
| RAG 往往比直接硬喂长上下文更稳 | RAG 先检索候选片段，再生成答案 | 法规、合同、知识库问答通常优先做检索层 |

如果只记一个工程判断，可以记下面这句话：**NIAH 是长上下文检索召回测试，不是知识测试，也不是综合推理测试。**

常见的经验公式可以写成：

$$
\text{success} \approx \text{baseline} - \beta \cdot d_{\text{edge}} \cdot g(L)
$$

其中：

- $\text{baseline}$：在理想条件下的基线成功率
- $d_{\text{edge}}$：needle 到最近边界的距离
- $L$：上下文长度
- $\beta$：模型对位置偏差的敏感度
- $g(L)$：长度惩罚函数，表示上下文越长，定位通常越难

这个公式不是严格定律，而是一个工程近似。它表达的是：**同样的事实，埋得越深、文档越长，模型越容易漏掉。**

再看一个新手版例子。假设你把一本 1,000 页教材整本交给模型，在第 437 页插一张纸条：

```text
NEEDLE_FACT: 火星的两颗天然卫星是福波斯和戴摩斯。
```

然后提问：

```text
文中被插入的唯一事实是什么？请原样回答。
```

如果模型答对，只能推出一件事：它在长文档里找到了那句事实。它没有因为“知道天文学”而天然通过测试；真正被测的是**长输入条件下的精确提取能力**。

---

## 问题定义与边界

NIAH 的问题定义可以压缩成一句话：**给定一段长文本和一个只在文中出现一次的目标事实，模型能否把该事实准确找出并复述。**

这个定义看起来简单，但边界必须说清楚，否则很容易把不同任务混在一起。

第一，NIAH 主要测**定位能力**，不是综合推理能力。定位能力就是“知道该看哪里”。如果 needle 是一句明确事实，例如“项目代号是 `orion-17`”，模型只需要找到并复述；如果题目要求跨多段文本整合条件、比较冲突版本、做因果推断，那已经超出纯 NIAH 了。

第二，needle 的数量要受控。单针测试最适合测位置偏差，因为变量最少。多针测试会额外混入排序、去重、冲突消解、是否漏召回等问题。对初学者来说，先把它理解成“只藏一张纸条”最合适；否则你很难判断模型到底是没看到、看到了但答错了，还是看到了多个候选却选错了。

第三，干扰密度必须定义。所谓干扰密度，是指单位长度里有多少看起来像答案、但并不是真答案的信息。比如文档里反复出现日期、版本号、端口号、产品代号，那么“找出唯一正确数字”会明显更难。干扰密度不一致时，两个实验的准确率不能直接比较。

下面这个 ASCII 图示可以帮助建立直觉：

```text
[开头]
  段落1：背景描述
  段落2：背景描述
  ...
  段落218：背景描述
  >>> NEEDLE: "内部审计窗口为 14 天" <<<
  段落219：背景描述
  ...
[结尾]
```

一个更严格的定义方式，是把 needle 的位置写成相对深度：

$$
d_{\text{rel}}=\frac{\text{needle 起始 token 位置}}{\text{总 token 数}}
$$

其中：

- $d_{\text{rel}} \approx 0$ 表示靠近开头
- $d_{\text{rel}} \approx 0.5$ 表示位于中段
- $d_{\text{rel}} \approx 1$ 表示靠近结尾

这样定义的好处是，不同长度的文档可以直接比较。第 5,000 个 token 在 8K 文档和 128K 文档中的意义完全不同，但相对深度 50% 的含义是一致的。

下面这张表列出评测时最常见的控制项：

| 变量 | 示例取值 | 是否纳入当前评测 | 对结果的影响 |
|---|---|---|---|
| 上下文长度 | 16K / 64K / 128K | 是 | 越长通常越难 |
| needle 距边界位置 | 近边 / 中段 / 深中段 | 是 | 中段常是错误高发区 |
| needle 数量 | 1 / 5 / 10 | 视实验而定 | 多针会混入排序和去重 |
| 干扰密度 | 低 / 中 / 高 | 是 | 相似信息越多越难 |
| 提问形式 | 复述事实 / 指出位置 / 二者都要 | 是 | 同时要求位置时更难 |
| 是否允许外部检索 | 否 / 是 | 通常否 | 允许外检就不再是纯 NIAH |

新手最容易误解的地方有两个。

第一，NIAH 不是开放世界问答。模型不能去“想一个可能正确的答案”，它只能在给定文本里找。  
第二，NIAH 也不是摘要任务。模型说“文档里提到了审计窗口”并不够，通常要把具体事实完整复述出来。

因此，NIAH 的边界可以概括为：**它测的是长输入条件下的检索召回能力，尤其是位置敏感性，而不是泛化到一切长文本能力。**

---

## 核心机制与推导

为什么 needle 越靠中间越容易丢？最常见的经验解释是：在超长上下文中，模型对开头和结尾更容易形成强锚点，而中段同时失去“开头优先”和“结尾新近”两种优势，所以更容易被稀释。

这类现象通常表现为一条 U 型曲线：靠近开头准确率较高，靠近结尾也较高，中间最低。可以写成一个简化形式：

$$
\text{Acc}(L,d)=A_0-\beta \cdot d \cdot g(L)
$$

其中：

- $A_0$：理想基线准确率
- $d$：needle 到最近边界的距离
- $\beta$：位置敏感系数
- $g(L)$：长度放大函数

为了避免单位混乱，工程上更常把距离写成相对距离：

$$
d_{\text{edge}}=\frac{\min(i, L-i)}{L}
$$

这里：

- $i$：needle 在上下文中的位置
- $L$：上下文总长度
- $\min(i, L-i)$：needle 到最近边界的距离

于是公式可改写为：

$$
\text{Acc}(L,d)=A_0-\beta \cdot d_{\text{edge}} \cdot g(L)
$$

常见的 $g(L)$ 近似写法有两种：

$$
g(L)=\log(1+L)
$$

或者

$$
g(L)=\frac{L}{L_0}
$$

前者表示“长度越长，惩罚继续增长，但增速放缓”；后者表示“长度惩罚近似线性增长”。真实模型未必严格符合任何一种，但这两个形式足够帮助工程分析。

看一个更容易算的玩具例子。假设：

- $A_0=0.95$
- $\beta=0.20$
- $L=128K$
- needle 放在正中间，所以 $d_{\text{edge}} \approx 0.5$
- 取 $g(L)=\log_{10}(L/1000)$

那么：

$$
g(128K)=\log_{10}(128)\approx 2.11
$$

带入得：

$$
\text{Acc}\approx 0.95 - 0.20 \times 0.5 \times 2.11
= 0.95 - 0.211
\approx 0.739
$$

这个结果的解释是：在相同模型下，如果文档非常长，且 needle 埋在中段，那么准确率可能从接近 95% 掉到约 74%。这不是精确预测，而是说明**长度惩罚和位置惩罚会叠加**。

再看一张工程上更直观的示意表：

| 上下文长度 | 近边位置准确率 | 中段位置准确率 | 现象 |
|---|---:|---:|---|
| 16K | 95% | 89% | 基本可用，差距可见 |
| 64K | 92% | 76% | 中段开始明显下滑 |
| 128K | 89% | 61% | U 型差异显著 |
| 256K | 84% | 47% | 中段召回进一步恶化 |

这张表不是某个单一模型的官方成绩，而是对公开研究和工程实践的概括。关键不是某个百分比本身，而是趋势：**上下文增长和中段埋针会共同放大召回下降。**

新手可以把“lost-in-the-middle”理解成下面这个现象：

1. 开头的信息容易被当成全局设定保留下来。
2. 结尾的信息离回答阶段最近，容易被再次利用。
3. 中间信息既不占开头优势，也不占结尾优势，所以最容易弱化。

真实工程里，这种弱化会造成三类常见错误：

| 错误类型 | 表现 | 根因 |
|---|---|---|
| 漏召回 | 直接说“文中未提到” | 没有定位到 needle |
| 近邻替代 | 把附近相似条款当答案 | 看到了相关区域，但没有精确取回 |
| 幻觉补全 | 给出看似合理但文中不存在的答案 | 检索失败后生成器仍继续作答 |

例如，你把一整本 100K token 的合规手册喂给模型，问“罚则条款里关于数据保留期限的例外是什么”。如果相关句子埋在中间章节，模型可能答成相邻条款、只给模糊概括，或者直接编造一个看似合理的例外条件。这时问题通常不是“它不懂合规”，而是**它没有把证据句准确捞出来**。

---

## 代码实现

最小实现只需要四步：

1. 构造 haystack，也就是长文档背景文本
2. 生成 needle，也就是唯一事实
3. 插入到指定位置
4. 让模型回答，再做严格校验

下面给出一个**可直接运行**的 Python 示例。它只依赖标准库，不调用真实 LLM，但完整演示了数据构造、插针、提问、判分、按位置统计结果的全过程。你可以把它保存为 `niah_demo.py` 直接运行。

```python
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List


SEED = 42
random.seed(SEED)


@dataclass
class NeedleCase:
    context_length: int
    depth_ratio: float
    haystack: str
    needle_fact: str
    insert_char_pos: int
    question: str
    expected_answer: str


def build_haystack(paragraphs: int = 220) -> str:
    blocks = []
    for i in range(paragraphs):
        blocks.append(
            " ".join(
                [
                    f"段落{i}：这是背景材料。",
                    f"系统版本 v{i % 9}.{(i * 3) % 10}。",
                    f"记录年份 20{(10 + i) % 100:02d}。",
                    f"区域编号 R-{i % 17:02d}。",
                    "这一段不包含目标事实。"
                ]
            )
        )
    return "\n".join(blocks)


def insert_needle(haystack: str, needle: str, ratio: float) -> NeedleCase:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0.0, 1.0]")

    pos = min(int(len(haystack) * ratio), len(haystack))
    merged = haystack[:pos] + "\n" + needle + "\n" + haystack[pos:]

    return NeedleCase(
        context_length=len(merged),
        depth_ratio=ratio,
        haystack=merged,
        needle_fact=needle,
        insert_char_pos=pos,
        question="文档中被插入的唯一事实是什么？请逐字精确回答。",
        expected_answer=needle,
    )


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("：", ":")
    return text


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def bucket_depth(ratio: float) -> str:
    if ratio < 0.2:
        return "near-start"
    if ratio < 0.4:
        return "early-middle"
    if ratio < 0.6:
        return "middle"
    if ratio < 0.8:
        return "late-middle"
    return "near-end"


def mock_biased_model(case: NeedleCase) -> str:
    """
    一个带位置偏差的假模型：
    - 开头、结尾更容易答对
    - 中间更容易答错
    - 上下文越长，中间区域再额外降一点成功率
    """
    depth = case.depth_ratio
    length = case.context_length

    if depth <= 0.15 or depth >= 0.85:
        success_prob = 0.95
    elif depth <= 0.30 or depth >= 0.70:
        success_prob = 0.78
    else:
        success_prob = 0.52

    if length > 15000 and 0.40 <= depth <= 0.60:
        success_prob -= 0.12

    success_prob = max(0.0, min(1.0, success_prob))

    if random.random() < success_prob:
        match = re.search(r"NEEDLE_FACT:.*", case.haystack)
        return match.group(0) if match else ""

    distractors = [
        "NEEDLE_FACT: 内部审计窗口为 7 天。",
        "NEEDLE_FACT: 默认保留期限为 30 天。",
        "未找到。",
    ]
    return random.choice(distractors)


def evaluate_positions(depths: List[float], trials_per_depth: int = 40) -> List[Dict]:
    rows = []

    for depth in depths:
        correct = 0
        total = 0

        for _ in range(trials_per_depth):
            haystack = build_haystack()
            needle = "NEEDLE_FACT: 内部审计窗口为 14 天。"
            case = insert_needle(haystack, needle, ratio=depth)

            pred = mock_biased_model(case)
            hit = exact_match(pred, case.expected_answer)

            total += 1
            correct += int(hit)

        acc = correct / total
        se = math.sqrt(acc * (1 - acc) / total)
        rows.append(
            {
                "depth_ratio": depth,
                "depth_bucket": bucket_depth(depth),
                "correct": correct,
                "total": total,
                "accuracy": round(acc, 4),
                "stderr": round(se, 4),
            }
        )

    return rows


def print_report(rows: List[Dict]) -> None:
    print("| depth | bucket | correct | total | accuracy | stderr |")
    print("|---:|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['depth_ratio']:.2f} | {row['depth_bucket']} | "
            f"{row['correct']} | {row['total']} | "
            f"{row['accuracy']:.4f} | {row['stderr']:.4f} |"
        )


if __name__ == "__main__":
    # 单样本演示
    haystack = build_haystack()
    needle = "NEEDLE_FACT: 内部审计窗口为 14 天。"
    case = insert_needle(haystack, needle, ratio=0.52)

    pred = mock_biased_model(case)

    print("Question:", case.question)
    print("Prediction:", pred)
    print("Expected:", case.expected_answer)
    print("Exact Match:", exact_match(pred, case.expected_answer))
    print()

    # 扫位置，观察 U 型趋势
    depths = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
    rows = evaluate_positions(depths, trials_per_depth=40)
    print_report(rows)
```

这段代码验证了四件事：

| 步骤 | 代码位置 | 作用 |
|---|---|---|
| 构造长文档 | `build_haystack` | 生成背景噪声 |
| 插入唯一事实 | `insert_needle` | 控制 needle 的深度位置 |
| 模拟模型作答 | `mock_biased_model` | 故意制造“中段更差”的位置偏差 |
| 严格判分 | `exact_match` | 防止“差不多对”被误算成正确 |

如果你运行它，通常会看到靠近开头和结尾的位置准确率更高，而 0.50 附近的 `middle` 桶更低，这就是 U 型趋势的最小演示。

如果要接入真实模型，核心替换只有一处：把 `mock_biased_model(case)` 换成真实 API 调用。最小伪代码可以写成：

```python
prompt = f"""
请阅读下面全文，只根据文档回答，不要使用外部知识。
如果无法确定，请明确回答“未找到”，不要猜测。

{case.haystack}

问题：{case.question}
"""

# response = call_llm(prompt, temperature=0)
# score = exact_match(response, case.expected_answer)
```

做真实评测时，建议固定下面这些设置：

| 项目 | 建议做法 | 原因 |
|---|---|---|
| 温度 | `0` 或接近 `0` | 降低随机性，避免污染指标 |
| needle 格式 | 唯一、不可猜、可精确匹配 | 让结果真正反映检索能力 |
| 位置采样 | 覆盖头部、中段、尾部 | 否则看不到 U 型差异 |
| 判分方式 | exact match + 规范化 | 防止主观判分 |
| 重复次数 | 每档至少 20 到 50 次 | 降低抽样波动 |

如果你希望显式测试“位置偏差”，不要只随机插入，而要系统扫位置：

- 近开头：`ratio = 0.05`
- 早中段：`ratio = 0.30`
- 正中段：`ratio = 0.50`
- 晚中段：`ratio = 0.70`
- 近结尾：`ratio = 0.95`

这样得到的结果更容易画成热图或折线图，也更容易判断系统从哪个长度、哪个位置开始退化。

---

## 工程权衡与常见坑

工程里最常见的误判，是把“模型支持 128K 输入”误当成“128K 任意位置都能稳定找对”。前者是**容量声明**，后者是**可用性声明**。这两个命题不能混为一谈。

下面这些坑在真实系统里最常见。

| 常见坑 | 表现 | 原因 | 缓解措施 |
|---|---|---|---|
| 位置偏差 | 中段事实看漏 | 中段锚点弱 | 给章节提示，或先做检索缩短上下文 |
| 上下文过长 | 回答退化成模糊摘要 | 精确信息被稀释 | 切段后只送相关片段 |
| 干扰过强 | 把相似条款当成答案 | 相邻内容竞争 | 降噪、重排、增加唯一标识 |
| needle 太短 | 单个数字、单个词容易淹没 | 显著性不足 | 给事实加结构化前缀 |
| 判分太松 | “差不多对”被算正确 | 验收口径错误 | 用 exact match 或规则归一化 |
| prompt 污染 | 结果好看但不可比 | 额外给了位置线索 | 把提示策略单独作为变量记录 |

先看一个新手版例子。你把下面这句插入文档：

```text
NEEDLE_FACT: 默认上报端口为 4317。
```

如果题目问的是“默认端口是多少”，模型答出 `4317` 才算通过。它答“文中提到一个默认端口”不算；它答 `8080` 也不能因为“看起来像端口号”就算部分正确。NIAH 测的是**精确取回**，不是模糊理解。

再看一个真实工程例子。假设你做法规问答，把一整本 100K token 的合规手册喂给模型，问题是“跨境传输审批的例外条件是什么”。如果答案恰好埋在中间章，模型常见有三种失败方式：

1. 答成附近相似条款。
2. 只给一个笼统概括，没有精确例外条件。
3. 直接编造一个文中不存在、但表面上合理的例外。

这时最危险的不是“答不出来”，而是**检索失败后仍然继续生成**。因为用户看到的是一段流畅自然的答案，很难立刻意识到它没有证据支撑。

“位置提示”是一个成本很低、但经常有效的补救手段。所谓 edge-aware，就是显式告诉模型目标大致位于文档的前部、后部或某个章节附近。例如：

```python
prompt = f"""
请阅读文档并回答问题。
提示：目标事实出现在文档靠近结尾的位置，请优先检查后部段落。
如果找不到，请明确回答“未找到”，不要猜测。

{long_document}

问题：被插入的唯一事实是什么？
"""
```

这种提示不能消除结构性的中段偏差，但常常能减少盲目搜索导致的误答。要注意的是，它属于**策略增强**，不是模型原生能力的提升，所以评测时必须单独记录。

工程上更稳妥的做法，是把 NIAH 当成**诊断工具**，而不是当成宣传口径。它回答的是：

- 文档长到什么程度开始掉点？
- 中段衰减有多严重？
- 哪类事实最容易漏？
- 仅靠长上下文是否已经够用？

这些答案比“模型支持 128K”更接近系统真实可用性。

---

## 替代方案与适用边界

如果 NIAH 暴露出明显的位置偏差，替代方案通常有三类。

第一类是 **RAG + LLM**。RAG 的意思是“先检索，再生成”。它先用检索器把相关片段找出来，再把更短、更集中的候选上下文交给模型回答。对于法规、合同、知识库、FAQ、技术手册这类可切段、可索引的语料，RAG 往往比“全文硬喂”更稳。

第二类是 **分段摘要或局部窗口 prompting**。它把长文档拆成多个窗口，每次只处理一部分，再做阶段性汇总。这种方法适合结构清晰、章节边界明显的问题，例如会议纪要、长报告、操作手册。

第三类是 **纯长上下文直喂**。它不是不能用，而是更适合下面这些情况：needle 靠近边界、文本长度不极端、问题允许一定容错、系统更看重链路简单而不是极致稳定。

一个新手版流程图如下：

```text
原始长文档
   |
   v
Retriever 检索相关 5 段
   |
   v
LLM 阅读这 5 段
   |
   v
Answer + Evidence
```

这比“把 100K 全塞进去再问”更符合大多数生产系统对鲁棒性和可解释性的要求。

下面给出对比表：

| 方案 | 适用长度/场景 | 优点 | 缺点 |
|---|---|---|---|
| 纯 LC-LLM 直喂 | 中等长度、needle 靠边、问题简单 | 实现最简单，链路最短 | 中段信息容易丢，成本高 |
| RAG + LLM | 法规、合规、知识库、企业文档 | 长度增长时更稳，可返回证据 | 依赖检索质量和切段策略 |
| 分段摘要 | 长报告、长手册、章节清晰文档 | 易解释，可逐层压缩 | 精确细节可能丢失 |
| Windowed Prompting | 可切块的长文本分析 | 可显式控制阅读范围 | 实现复杂，需要调度策略 |
| 结构化索引 + 生成 | 表格、日志、配置库 | 定位精度高，适合精确字段 | 前处理成本高 |

适用边界也要说清楚。

| 场景 | 更适合的方案 | 原因 |
|---|---|---|
| 事实靠近开头或结尾 | 纯长上下文可能够用 | 边界位置天然更有优势 |
| 事实深埋中段且文档极长 | 优先考虑 RAG | 中段掉针风险高 |
| 问题要求先全局理解再局部定位 | 检索 + 摘要 + 生成组合 | 单一手段往往不稳 |
| 必须返回证据链 | RAG / span extraction | 更容易给出出处和引用 |
| 容错率极低 | 检索 + 验证 + 严格判分 | 需要显式失败兜底 |

换句话说，NIAH 告诉你的不是“长上下文没用”，而是：**长上下文不是免费的。窗口越大，不代表任意位置的信息都同样容易被取回。**

---

## 参考资料

| 参考 | 年份 | 要点摘要 |
|---|---:|---|
| [Lost in the Middle: How Language Models Use Long Contexts](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long) | 2024 | 系统展示了长上下文中的位置偏差，指出相关信息位于开头或结尾时更容易被模型利用，中段更脆弱 |
| [Multilingual Needle in a Haystack: Investigating Long-Context Behavior of Multilingual Large Language Models](https://arxiv.org/abs/2408.10151) | 2024 | 把 NIAH 扩展到多语言场景，观察到语言类型和插入位置都会影响检索效果，中段仍是弱点 |
| [Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal Large Language Models](https://www.microsoft.com/en-us/research/publication/multimodal-needle-in-a-haystack-benchmarking-long-context-capability-of-multimodal-large-language-models/) | 2024 | 把“捞针”扩展到图文混合长上下文，说明多模态模型同样存在长上下文定位和负样本幻觉问题 |
| [U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack](https://www.aimodels.fyi/papers/arxiv/u-niah-unified-rag-llm-evaluation-long) | 2025 | 用统一框架比较长上下文 LLM 与 RAG，强调单针、多针、长针等不同设置下两类方案的差异 |
| [Anthropic Claude 3.5 Sonnet Model Card Addendum](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/ModelCardClaude3Addendum.pdf) | 2025 | 给出企业模型在 NIAH 类测试上的召回曲线，说明商业模型也会专门用这类评测验证长上下文读取能力 |
| [MMNeedle 项目页](https://mmneedle.github.io/) | 2025 | 提供多模态 NIAH 基准的任务定义、数据和榜单，便于工程读者快速理解多模态版本的测试形式 |
