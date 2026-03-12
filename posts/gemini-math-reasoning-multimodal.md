## 核心结论

Gemini 在“多模态数学推理”上的价值，不是单纯把图片转成文字，再让语言模型算题，而是把图像、公式、坐标、图表和问题放进同一条推理链里处理。白话说，它不是先“看完图再忘掉图”，而是尽量边看边推。

公开技术报告能确认两件事。第一，Gemini 1.5 Pro 在需要“看图 + 数学推理”的公开基准上表现明显更强：MathVista 为 63.9%，ChartQA 为 87.2%，InfographicVQA 为 81.0%。这说明它对图表、文档、数学图形的联合理解已经超过只靠 OCR 或只靠语言补全的老方法。第二，Gemini 1.5 是原生多模态模型，能把文本、图像、音频、视频交错放进同一上下文中处理；但 Google 没有公开完整的内部视觉对齐细节，所以外部工程复现时，不能把“某种具体 bbox 推理流水线”误写成 Gemini 已公开确认的内部实现。

最稳妥的技术判断是：Gemini 的优势来自统一多模态建模，而不是某一个单独模块。对初级工程师来说，可以把它理解为“图、字、公式都被编码成模型可继续推理的表示”，然后模型在这些表示上生成链式推理。典型玩具例子是平行四边形：图中给出底边 $b=20$、面积 $A=100$，模型先识别符号关系，再推出
$$
A=b\times h,\quad h=\frac{A}{b}=\frac{100}{20}=5
$$
这里真正重要的不是最后算出 5，而是模型要先看懂图里哪个数字对应底边，哪个数字对应面积。

| 基准 | 任务类型 | Gemini 1.5 Pro |
|---|---|---:|
| MathVista | 视觉情境中的数学推理 | 63.9% |
| ChartQA | 图表问答与数值推导 | 87.2% |
| InfographicVQA | 信息图阅读与问答 | 81.0% |

---

## 问题定义与边界

“多模态数学推理”指模型不只读自然语言题干，还要同时理解图片里的几何图形、坐标轴、函数图、表格、公式排版和视觉布局，再据此完成数学推导。白话说，题目不再是纯文字，而是“课本截图 + 图 + 问题”的组合。

MathVista 是这个问题的代表 benchmark。它包含 6,141 个样本，来源于 28 个已有数据集和 3 个新构造数据集：IQTest、FunctionQA、PaperQA。这里的边界很清楚：

| 子任务 | 主要输入 | 模型必须完成什么 |
|---|---|---|
| IQTest | 图形规律题、拼图、视觉逻辑图 | 找视觉模式，再做逻辑推断 |
| FunctionQA | 函数图像、坐标、曲线 | 读图取值、找交点、分析趋势 |
| PaperQA | 论文图表、科学插图 | 理解专业图，再做科学或数学判断 |

例如 FunctionQA 可能问“图中两条曲线的交点横坐标是多少”。这类题不能只靠语言模型猜，也不能只靠 OCR 把图中文字抄下来。模型至少要完成三步：读出坐标轴含义、定位曲线关系、把图像事实转成可计算的符号表达。

这也是边界所在。多模态数学推理不等于“任何数学题都能靠看图解决”。如果图像极度模糊、手写公式严重扭曲、坐标轴缺失刻度，模型即使推理能力强，也会卡在输入解析阶段。工程上要把问题拆成两层：

1. 感知层是否把图中的符号、线段、数值读对。
2. 推理层是否在读对之后做出了正确演算。

很多失败其实不是“不会推理”，而是“第一眼看错了”。

---

## 核心机制与推导

从公开报告可以确认，Gemini 1.5 属于原生多模态模型，可以在统一上下文中接收图像与文本，并在视觉推理基准上直接输出答案。这里“原生多模态”的白话解释是：不是外挂几个独立工具串起来，而是在同一个模型系统里处理不同模态。

但外部文章容易写错的一点是：Google 并没有在公开报告里完整披露“视觉 token 如何与符号 token 对齐”的专有细节，更没有逐步公布一个“Gemini 内部 bbox 校验流水线”。因此，更准确的写法是：

- 已公开确认：Gemini 1.5 能直接处理图像与文本的联合输入，并在 MathVista、ChartQA 等任务上达到较高准确率。
- 合理工程推断：如果你想在自己的系统里复现类似能力，通常要引入“接地推理”机制，也就是让每一步推理都尽量能回指到图中的证据区域。

“接地”这个词的意思是，推理步骤不能只写一句话，还要能说明这句话对应图里的哪一部分。可形式化写成：
$$
s_t = (r_t, g_t)
$$
其中 $r_t$ 是第 $t$ 步的文字或符号推理，$g_t$ 是与这一步绑定的视觉证据，比如坐标区域、公式区域或图例区域。白话说，每一步都尽量做到“这句话是看着哪里得出的”。

一个简化的推理链可以写成：

1. 输入题图与问题。
2. 视觉编码器提取局部区域特征。
3. 文本解码阶段生成候选推理步骤。
4. 每一步都尝试回到图中寻找对应证据。
5. 若证据与步骤冲突，则回退重读。
6. 在所有关键中间量都稳定后输出答案。

玩具例子最容易理解。给一张平行四边形图，图上只标出底边 20、面积 100，并问高度。系统的正确中间表示应接近：

- 识别到“20”绑定底边。
- 识别到“100”绑定面积。
- 调用面积公式 $A=b\times h$。
- 推出 $h=A/b=5$。

这和纯 OCR 的区别在于，纯 OCR 只能抄出“20, 100”，却不知道哪个数字属于哪条边、哪个数字是面积量。

真实工程例子是仪表盘分析。比如工厂监控界面里有一张产线良率折线图，问题是“本周平均良率是否低于上周 5% 以上”。系统必须先读出图例、横轴时间、纵轴比例，再聚合若干点值，最后比较两个周均值。这里的难点不是四则运算，而是图表语义绑定。

---

## 代码实现

下面给一个可运行的极简版 Python 示例，模拟“先接地图中证据，再做数学推理”的流程。它不是 Gemini 的内部实现，而是一个适合工程复现和调试的外部思路。

```python
from dataclasses import dataclass

@dataclass
class VisualToken:
    text: str
    role: str
    bbox: tuple  # (x1, y1, x2, y2)

def verify_token(token: VisualToken, expected_role: str) -> bool:
    return token.role == expected_role and len(token.bbox) == 4

def solve_parallelogram(tokens):
    base = None
    area = None

    for token in tokens:
        if verify_token(token, "base"):
            base = float(token.text)
        elif verify_token(token, "area"):
            area = float(token.text)

    assert base is not None, "missing base token"
    assert area is not None, "missing area token"
    assert base != 0, "base must be non-zero"

    height = area / base
    return height

toy_tokens = [
    VisualToken(text="20", role="base", bbox=(10, 80, 40, 100)),
    VisualToken(text="100", role="area", bbox=(45, 20, 90, 45)),
]

result = solve_parallelogram(toy_tokens)
assert result == 5.0
print(result)
```

这个例子里，`VisualToken` 可以理解为“从图中读出的带位置的符号”。白话说，它不是裸数字，而是“数字 + 它在图里的身份 + 它在图里的位置”。

如果把这个思路扩展到多步推理，解题 loop 可以写成：

```python
def grounded_reasoning_loop(reasoning_steps, grounding_module, verify_with_ocr, cot_decoder):
    cot_tokens = []

    for step in reasoning_steps:
        bbox = grounding_module(step["targets"])
        if verify_with_ocr(bbox, step["text"]):
            cot_tokens.append(step["text"])
        else:
            cot_tokens.append("RETRY: 请重读第 1 步定位的 bbox")

    return cot_decoder(cot_tokens)
```

工程上建议每一轮都输出三类 debug 信息：

| 调试项 | 作用 | 为什么重要 |
|---|---|---|
| `bbox` | 记录模型看的区域 | 方便回查“到底看了哪里” |
| `recognized_text` | 记录 OCR/符号识别结果 | 检查是否把 $+$ 看成了 $-$ |
| `reasoning_step` | 记录当前推理语句 | 检查推理是否与视觉证据一致 |

如果你在做真实系统，前端可把“第 3 步引用的图块”高亮给人工复核。这样一旦答案错了，不用整条链重跑，先看是读图错还是算错。

---

## 工程权衡与常见坑

第一类坑是视觉误读。数学图像里的小符号特别脆弱，尤其是负号、根号、角度标记、上下标和细线。把 $x+15$ 误读成 $x-15$，后面再完整的方程求解也会全部失效。误差传播可以概括成：
$$
\text{输入图像} \rightarrow \text{视觉识别} \rightarrow \text{中间符号} \rightarrow \text{链式推理} \rightarrow \text{最终答案}
$$
越早的错误，越贵。

第二类坑是过早压缩。很多系统一开始就把图像压成一句 caption，比如“一个坐标图上有两条曲线”。这对问答不够，因为数值、位置、关系都被抹平了。数学任务需要保留局部结构，尤其是坐标值、图例颜色、箭头方向、表头单位。

第三类坑是把 benchmark 分数直接等同于真实工程可用性。ChartQA 87.2% 很高，但真实报表往往比 benchmark 更脏：字体不统一、图例遮挡、截图压缩、颜色失真、模板杂乱。生产环境里，你要追求的不只是平均准确率，而是“出错时能不能定位错在哪里”。

第四类坑是链式推理不可审计。模型如果只吐出一个最终答案，用户很难知道它是看懂图了，还是碰巧猜对。对数学类任务，最好保留中间量。例如：
- 读到的数值是什么
- 数值来自图中哪里
- 用了哪条公式
- 最终单位是什么

真实工程例子：做经营看板问答时，用户问“Q4 华东区利润率是否高于 Q3 目标线”。如果系统把图例“目标线”识别成“实际线”，结论会完全反转。解决办法不是单纯换更强模型，而是强制日志化：图例 bbox、区域 OCR、对应语义标签、最终引用链路全部打出来。

---

## 替代方案与适用边界

不是所有视觉数学任务都适合直接 end-to-end 交给 Gemini。图像质量、任务类型、错误代价不同，方案也应不同。

| 方案 | 适合场景 | 优点 | 边界 |
|---|---|---|---|
| Gemini 直读图像 + 推理 | 图文混合、上下文长、问题多变 | 集成度高，交互自然 | 错误来源不易拆解 |
| 专用 OCR/公式识别 + LLM 推理 | 公式密集、排版规则强 | 可控、易调试 | 管线更长，维护更重 |
| 图表结构化解析 + 规则计算 | BI、报表、工业监控 | 稳定、可审计 | 对开放问题不灵活 |
| 人工复核混合流 | 高风险场景 | 降低误判成本 | 吞吐受限 |

切换策略可以很实用：

1. 如果图像清晰、问题开放、需要多轮对话，优先直接用 Gemini。
2. 如果是公式截图、扫描件、表格密集页，先做专用 OCR/公式识别，再把结构化结果喂给模型。
3. 如果是固定模板报表，优先结构化解析，不要把简单数值逻辑全交给生成式模型。
4. 如果错误代价高，比如医疗、财务、工业控制，必须保留人工校验或规则兜底。

这里要特别强调一个准确性边界：MathVista 的早期公开结果显示，当时最强模型在这个 benchmark 上仍明显低于人类。Gemini 1.5 把分数推进到 63.9%，说明进步很大，但还远没有到“任何数学图题都可靠”的程度。它更适合做高能力助手，而不是无条件终裁器。

---

## 参考资料

1. Gemini 1.5 技术报告：Google, *Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context*  
https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf

2. MathVista 项目页：Pan Lu et al., *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts*  
https://mathvista.github.io/

3. Microsoft Research 论文页：*MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts*  
https://www.microsoft.com/en-us/research/publication/mathvista-evaluating-mathematical-reasoning-of-foundation-models-in-visual-contexts/

4. ChartQA 论文：Ahmed Masry et al., *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning*  
https://aclanthology.org/2022.acl-long.375/
