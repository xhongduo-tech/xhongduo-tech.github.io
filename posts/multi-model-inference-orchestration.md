## 核心结论

多模型推理编排，不是把多个模型堆在一起同时跑，而是把一次请求拆成“谁先处理、处理到哪一步、什么时候升级、要不要验证”这四个决策。最直白的理解是：简单问题走小模型，复杂问题升级大模型，关键结果再做验证。它不是“多个模型同时输出后投票”，而是先决定谁来做、做到哪一步、要不要升级。

它成立的原因很简单：不同模型在成本、延迟、质量上通常存在稳定差异。小模型便宜、快，但上限低；大模型质量高，但贵、慢。编排的目标不是追求单次请求最强，而是在整体流量上，用更低的平均成本和更短的平均延迟，拿到接近强模型的整体质量。

最有效的生产形态通常不是单一技巧，而是三件事一起工作：

1. 路由：先判断请求属于哪一类、难度多高。
2. 级联：如果当前模型把握不够，就升级到更强模型。
3. 验证：对高风险输出再做正确性或安全性检查。

核心级联规则可以写成：

\[
\text{若 }\hat q_i(x)\ge \tau_i,\ \text{则接受 }M_i;\ \text{否则升级到 }M_{i+1}
\]

这里，$\hat q_i(x)$ 是“当前模型对输入 $x$ 的质量估计”，白话讲就是“它觉得自己这次答得靠不靠谱”；$\tau_i$ 是放行阈值，白话讲就是“业务愿意接受的最低把握线”。

下面这张总览表先把整体思路定住：

| 输入类型 | 路由策略 | 使用模型 | 是否升级 | 是否验证 | 预期收益 |
|---|---|---|---|---|---|
| FAQ、模板问答 | 规则匹配 + 高置信度放行 | 小模型 | 通常否 | 低风险可不验 | 极低延迟、低成本 |
| 普通检索问答 | 检索命中率 + 长度特征 | 中模型 | 可能 | 建议验引用 | 平衡成本与质量 |
| 长上下文总结 | 上下文长度 + 压缩比估计 | 大模型 | 常见 | 关键摘要需验证 | 降低漏信息风险 |
| 代码生成 | 任务难度 + 语法校验 | 中/大模型 | 常见 | 必须跑测试 | 提高可执行率 |
| 合规审核 | 风险分级 + 规则命中 | 小模型初筛，大模型复核 | 高频 | 必须验证 | 控制误判和漏判 |
| 高风险决策 | 默认高等级处理 | 大模型或人工 | 不强调逐级升级 | 必须验证 | 保守但稳定 |

---

## 问题定义与边界

多模型推理编排，讨论的是“如何分配推理资源”，不是“如何训练新模型”。对象可以是分类、检索问答、摘要、代码生成、合规审核，也可以是更细的流水线步骤，比如先抽取实体，再做生成，再做审查。

为了统一讨论，先定义符号：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $M_i$ | 第 $i$ 个模型 | 第几档能力的模型 |
| $c_i$ | 使用 $M_i$ 的单次成本 | 一次调用要花多少钱或多少算力 |
| $q_i$ | $M_i$ 的真实平均质量 | 它长期来看到底有多准 |
| $\hat q_i(x)$ | 对样本 $x$ 的质量估计 | 这次请求上它大概答得怎样 |
| $\tau_i$ | 第 $i$ 层接受阈值 | 分数过线就放行 |
| $L_i$ | 第 $i$ 个模型延迟 | 响应要多久 |

它适用的前提有三个。

第一，模型之间必须有可利用的差异。比如 7B 模型和 70B 模型在成本、延迟、复杂任务质量上差得明显，编排才有空间。如果两个模型性能接近、价格也接近，强行编排只会增加系统复杂度。

第二，任务必须具备可分流性。可分流性，白话讲就是“有一部分请求明显更简单，能提前识别出来”。客服 FAQ、常规模板写作、短文本分类都很适合；但如果每个请求都同样复杂，路由器很难发挥作用。

第三，必须有稳定的路由信号。这个信号可以来自规则、轻量分类器、检索分数、模型自评置信度、历史失败模式。没有信号，路由器就只能靠猜，收益会很弱。

一个新手容易理解的边界例子是客服系统：

- “退货时间是多少”这种 FAQ，直接走小模型。
- “请总结这份 80 页合同中的违约责任”这种长上下文问题，升级大模型。
- “这段回复是否包含法律承诺风险”这种高风险输出，再加验证或人工兜底。

但也有明显不适合的场景。比如每个请求都必须按同一高标准处理，且延迟不敏感、流量也不大，那么直接调用强模型通常更简单、更稳定。

下面用表格收一下适用边界：

| 任务类型 | 可分流性 | 风险等级 | 是否适合编排 |
|---|---|---|---|
| FAQ 问答 | 高 | 低 | 适合 |
| 检索增强问答 | 中高 | 中 | 适合 |
| 代码补全 | 中 | 中高 | 适合，但需验证 |
| 合规审核 | 中 | 高 | 适合，但必须兜底 |
| 医疗/法律最终结论 | 低 | 极高 | 谨慎，常需强模型或人工 |
| 小流量、高一致性服务 | 低 | 中高 | 通常不值得 |
| 模型差异极小的任务 | 低 | 任意 | 不适合 |

---

## 核心机制与推导

多模型编排可以拆成三层：路由器、级联器、验证器。

路由器负责估计难度。难度不是抽象概念，而是“当前便宜模型大概率能不能答对”。它可以用输入长度、领域标签、检索覆盖率、历史失败率、敏感词命中等特征来估计。比如上下文越长、跨文档引用越多、需要结构化输出越严格，通常越难。

级联器负责决定接受还是升级。它的最基本规则就是：

\[
\text{若 }\hat q_i(x)\ge \tau_i,\ \text{则接受 }M_i;\ \text{否则升级到 }M_{i+1}
\]

阈值 $\tau_i$ 越高，说明系统越保守。保守的好处是质量更稳，坏处是升级率更高，成本和尾延迟也更高。尾延迟，白话讲就是慢请求那一小撮的延迟，通常用 P95、P99 表示，比平均值更能反映用户体感。

验证器负责兜底。它不是再“做一遍推理”，而是检查输出是否满足某种约束。比如分类任务看标签是否合法，RAG 问答看引用是否来自检索文档，代码生成看单元测试是否通过，合规回复看是否命中禁用模式。生成式任务里，验证器往往比路由器更关键，因为生成错误不是“没结果”，而是“看起来像对的错结果”。

从成本角度，编排的核心收益来自期望值，而不是每个请求都更快。期望成本写成：

\[
\mathbb E[C]=\sum_i \Pr(\text{最终用到 }M_i)\,c_i
\]

期望质量写成：

\[
\mathbb E[Q]=\sum_i \Pr(\text{最终由 }M_i\text{ 输出})\,q_i
\]

如果再关心延迟，也可以写成：

\[
\mathbb E[L]=\sum_i \Pr(\text{最终用到 }M_i)\,L_i
\]

玩具例子先看最小版本。设小模型 $M_s$ 延迟 2ms，准确率 92%；大模型 $M_b$ 延迟 20ms，准确率 98%。路由器把 70% 请求留在小模型，30% 请求升级到大模型，则：

\[
\mathbb E[L]=0.7\times 2 + 0.3\times 20 = 7.4\text{ms}
\]

\[
\mathbb E[Q]=0.7\times 92\% + 0.3\times 98\% = 93.8\%
\]

这个结果说明两件事。第一，整体延迟从 20ms 降到 7.4ms，强模型没有必要处理所有样本。第二，整体准确率不是 98%，因为简单样本用了便宜模型，但只要路由做得准，损失可以很可控。多模型编排的价值正是在这里：把强模型留给少数难例。

真实工程例子更典型。假设企业内部知识库问答系统每天有 100 万次请求：

- 60% 是固定 FAQ。
- 25% 是普通文档问答。
- 10% 是长上下文总结。
- 5% 是高风险合规相关问题。

可以这样编排：

1. 规则路由先截住 FAQ，直接命中模板或小模型。
2. 普通问答走中模型，并要求返回引用片段。
3. 长上下文和低置信度请求升级大模型。
4. 合规相关输出再过一层验证器，不通过则人工兜底。

这不是学术上的漂亮结构，而是生产上最常见的做法，因为请求分布本来就不均匀。大部分流量很便宜，少部分流量才值得花重成本。

生成式场景里还有一种特殊编排：speculative decoding。它的流程不是“路由请求”，而是“路由 token 生成过程”：

`drafter 生成草稿 -> target 批量验证 -> 接受前缀 -> 继续生成`

这里的 drafter 是小草稿模型，白话讲就是“先帮大模型猜几步”；target 是最终目标模型，白话讲就是“真正有决定权的强模型”。如果草稿与目标模型兼容度高，就能减少强模型的串行前向次数，从而加速生成。但它依赖 tokenizer 和模型家族兼容，否则接受率会很差。

---

## 代码实现

工程上不要把所有逻辑塞进一个 `infer()`。最少应拆成四个模块：请求特征提取、路由决策、模型调用与升级、结果验证与回退。这样做的原因不是“代码更优雅”，而是为了可观测、可回滚、可限流。没有这三样，线上一旦误路由或拥塞，问题会定位不清。

先给一个最小伪代码，表达主流程：

```python
def infer(x):
    features = extract_features(x)
    route = router.predict(features)

    for model in route.cascade:
        y, q = model.predict_with_confidence(x)
        if q >= model.threshold:
            if verifier.enabled(model):
                ok = verifier.check(x, y)
                if ok:
                    return y
            else:
                return y

    return fallback_model.predict(x)
```

再给一个可运行的 Python 玩具实现。它不依赖外部库，但体现了“先估计难度，再决定升级，再做验证”的核心逻辑。

```python
from dataclasses import dataclass

@dataclass
class Request:
    text: str
    risk: str = "low"

@dataclass
class ModelConfig:
    name: str
    threshold: float
    max_latency_ms: int
    verify_enabled: bool
    fallback: str | None = None

def extract_features(req: Request) -> dict:
    length = len(req.text)
    has_code = "def " in req.text or "class " in req.text
    return {
        "length": length,
        "has_code": has_code,
        "risk": req.risk,
    }

def predict_with_confidence(model_name: str, req: Request, features: dict):
    # 玩具规则：短文本更适合小模型，长文本或代码问题更适合升级
    if model_name == "small":
        if features["length"] < 40 and not features["has_code"] and req.risk == "low":
            return "small_answer", 0.93
        return "small_answer_uncertain", 0.61

    if model_name == "large":
        return "large_answer", 0.97

    raise ValueError("unknown model")

def verify(req: Request, answer: str) -> bool:
    # 玩具验证器：高风险请求不能直接接受 uncertain 结果
    if req.risk == "high" and "uncertain" in answer:
        return False
    return True

def route(features: dict):
    # 简单路由：代码、高风险、超长文本直接进入 small -> large 级联
    return ["small", "large"]

def infer(req: Request, configs: dict) -> str:
    features = extract_features(req)
    cascade = route(features)

    for model_name in cascade:
        answer, conf = predict_with_confidence(model_name, req, features)
        cfg = configs[model_name]

        if conf >= cfg.threshold:
            if cfg.verify_enabled:
                if verify(req, answer):
                    return answer
            else:
                return answer

    fallback_name = configs[cascade[-1]].fallback
    if fallback_name:
        answer, _ = predict_with_confidence(fallback_name, req, features)
        return answer
    raise RuntimeError("no valid answer")

configs = {
    "small": ModelConfig(
        name="small",
        threshold=0.90,
        max_latency_ms=20,
        verify_enabled=False,
        fallback=None,
    ),
    "large": ModelConfig(
        name="large",
        threshold=0.95,
        max_latency_ms=200,
        verify_enabled=True,
        fallback="large",
    ),
}

easy_req = Request("退货政策是什么？", risk="low")
hard_req = Request("请解释下面代码为什么会有竞态条件：def update(x): ...", risk="high")

assert infer(easy_req, configs) == "small_answer"
assert infer(hard_req, configs) == "large_answer"
```

对应的配置表应显式存在，而不是散落在代码分支里：

| model_name | threshold | max_latency_ms | fallback | verify_enabled |
|---|---|---:|---|---|
| small | 0.90 | 20 | 无 | 否 |
| medium | 0.94 | 80 | large | 条件开启 |
| large | 0.97 | 200 | human_review | 是 |

一个更接近生产的流程图可以抽象成：

`preprocess -> route -> infer -> verify -> fallback/log`

真实工程里还要补三类能力。

第一是可观测。每次请求至少记录：路由结果、各层置信度、是否升级、验证是否通过、最终模型、总延迟。否则你只能看到“成本高了”或“质量差了”，但不知道坏在哪一层。

第二是可回滚。阈值、验证开关、路由规则都应配置化，必要时能一键切回“全部强模型”或“禁用升级”。编排系统的问题往往不是单个 bug，而是分布漂移，回滚速度很重要。

第三是可限流。大模型通常是稀缺资源，如果前置路由一波抖动把大量流量同时升级，下游就会被打爆。所以需要并发上限、队列长度控制、超时与拒绝策略。

---

## 工程权衡与常见坑

多模型编排最常见的失败，不是模型不够强，而是系统错误地相信了一个不可靠的信号。其中最典型的是“置信度未校准”。校准，白话讲就是“95% 置信度到底是不是真的约等于 95% 会答对”。很多小模型会“自信地错”：分数很高，但答案是错的。如果业务直接把高分当放行依据，错误会在最便宜的一层被放大。

新手例子很直接。小模型对一条法律问答给出 0.95 置信度，路由器于是直接放行，结果回答引用了过期条例。问题不在“大模型没用上”，而在“小模型的 0.95 根本不代表 95% 可靠”。修正方式通常是：拿独立校准集做温度缩放，再按业务线分别回归阈值，而不是共用一个全局阈值。

第二个坑是只优化平均值，不看尾延迟、失败率和风险。平均延迟下降，不等于用户体验就更好。如果 5% 的请求升级三次才完成，P99 延迟会非常难看。对在线系统，平均值只是成本视角，尾部指标才是稳定性视角。

第三个坑是级联链过深。理论上你可以放很多层：小模型、中模型、大模型、超大模型、人工兜底。但层数越多，决策开销、排队、重试、日志复杂度都会上升。生产里更常见的好方案是两层或三层，高收益、低复杂度。

第四个坑是没有背压。背压，白话讲就是“当前面太快、后面吃不下时，要有办法把流量压住”。没有背压，前面的小模型会快速给大量请求打上“需要升级”的标记，下游大模型队列瞬间堆满，最后系统整体变慢。

第五个坑是 tokenizer 不一致。尤其在 speculative decoding 中，如果 drafter 和 target 的分词方式不同，草稿 token 很难被目标模型高比例接受，理论加速就会被抵消。

第六个坑是优化目标和业务 KPI 脱节。比如系统只追求降低平均 token 成本，却忽略拒答率、人工介入率和高风险错误率。这样做出的阈值，在离线评测上看起来漂亮，在线上却可能不可用。

下面这张表把常见坑、后果和规避方式收拢：

| 坑位 | 后果 | 规避策略 |
|---|---|---|
| 置信度未校准 | 小模型高分错答被直接放行 | 独立校准集、温度缩放、分业务阈值 |
| 只优化平均值 | 平均成本好看，但 P95/P99 很差 | 同时约束尾延迟、失败率、最低质量 |
| 级联链过深 | 路径复杂、开销上升、收益递减 | 控制在 2 到 3 层，优先做高收益分流 |
| 无背压 | 下游大模型拥塞，整体雪崩 | 并发上限、队列长度、超时拒绝 |
| tokenizer 不一致 | speculative decoding 接受率低 | 同 tokenizer、同家族模型先测接受率 |
| 目标与 KPI 脱节 | 离线指标好，线上业务差 | 把成本、质量、风险、拒答率统一建模 |

---

## 替代方案与适用边界

多模型编排不是默认最优，它只是若干方案中的一种。判断要不要上编排，关键不是“这个架构听起来高级”，而是它是否比替代方案更值。

如果业务只关心最高质量，且延迟不敏感，直接用强模型往往更简单。系统更短，故障点更少，调参也少。反过来，如果请求量大、任务分布很偏、低风险请求很多，多模型编排通常更省钱。

下面把常见替代方案放在一起比较：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 单模型直出 | 架构最简单，一致性高 | 成本高，平均延迟高 | 低流量、高一致性要求 |
| 规则路由 | 可解释、低成本 | 覆盖有限，脆弱 | FAQ、模板化任务 |
| 多模型级联 | 兼顾成本、质量、延迟 | 调参与监控复杂 | 高流量、任务异质性强 |
| 投票式多模型 | 对部分任务更稳 | 成本高，经常更慢 | 高价值低流量判定任务 |
| speculative decoding | 可加速生成 | 依赖模型兼容性 | 大模型生成服务 |
| 人工兜底 | 风险最低 | 成本最高，扩展性差 | 高风险最终决策 |

这些方案之间不是互斥关系。真实系统经常是组合拳。比如“规则路由 + 多模型级联 + 人工兜底”，或者“检索增强后单模型推理 + 高风险验证”。要注意，检索增强本身不是多模型编排，它主要解决“知识来源”问题；多模型编排解决的是“算力怎么分配”问题。两者可以叠加，但不应混为一谈。

适用边界可以总结成一句更工程化的话：当任务异质性强、风险分层明显、流量足够大、模型差异明显时，编排值得做；当所有请求都差不多、流量很小、或者必须强一致处理时，编排收益可能不足以覆盖复杂度。

---

## 参考资料

如果只看三篇，优先看“模型级联”“speculative decoding”“推理服务编排”三类材料。前两类帮助理解机制，后一类帮助理解怎么把机制落到服务系统里。还要区分：论文结论通常是在特定数据集和评测设置下成立，生产系统还要额外考虑延迟尾部、缓存命中、热更新、并发和故障恢复。

| 名称 | 类型 | 为什么值得看 | 对应章节 |
|---|---|---|---|
| NVIDIA Triton: Ensemble Models | 推理服务框架文档 | 看服务侧如何把预处理、路由、模型、后处理串起来 | 代码实现 |
| KServe: ModelMesh Installation Guide | 多模型服务系统文档 | 看大量模型动态装载、缓存与服务治理 | 问题定义与边界 |
| RouteLLM GitHub | 工程项目 | 看路由式 LLM 编排的开源实现思路 | 核心机制与推导 |
| Efficient Inference With Model Cascades | 研究论文 | 看级联推理的正式问题定义与收益分析 | 核心机制与推导 |
| Faster Cascades via Speculative Decoding | 研究论文 | 看生成式场景中的级联加速思路 | 替代方案与适用边界 |
| TensorRT-LLM: Speculative Decoding | 工程文档 | 看 speculative decoding 的工程实现约束 | 工程权衡与常见坑 |

1. [NVIDIA Triton: Ensemble Models](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html)
2. [KServe: ModelMesh Installation Guide](https://kserve.github.io/website/docs/admin-guide/modelmesh)
3. [RouteLLM GitHub](https://github.com/lm-sys/RouteLLM)
4. [Efficient Inference With Model Cascades](https://openreview.net/forum?id=obB415rg8q)
5. [Faster Cascades via Speculative Decoding](https://research.google/pubs/faster-cascades-via-speculative-decoding/)
6. [TensorRT-LLM: Speculative Decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md)
