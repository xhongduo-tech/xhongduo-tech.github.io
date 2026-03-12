## 核心结论

Gemini 的 Thinking 模式，本质上不是“让模型更会说”，而是“给模型更多测试时推理资源”。测试时推理，白话说，就是模型在真正回答前，先花一段额外 token 做拆解、尝试、校验，再给结论。  
对 Gemini 2.0 Flash Thinking 来说，这条路线最早以“可见思考过程”的产品形态被大众感知；到 2025 年后的 Gemini 2.5 与 Gemini 3 系列，这件事在 API 里被正式做成了可控参数：`thinkingBudget` 或 `thinkingLevel`。

需要先纠正一个常见误解：Gemini 并不是稳定返回“完整内部推理原文”。当前官方文档区分了 raw thoughts 和 thought summaries。`includeThoughts: true` 返回的是思考摘要，不保证每次都返回，也不是原始全部推理链。这个设计和 OpenAI o1 的路线差异，不是“一个完全透明、一个完全黑箱”这么简单，而是两边都在避免直接暴露原始链路，只是 Gemini 提供了更细的预算控制，以及在部分产品界面里更强的可观察性。

从工程角度看，Thinking 模式值不值得开，不取决于“任务重不重要”，而取决于“任务是否需要多步推理”。数学证明、代码修复、跨文档一致性检查、工具调用前规划，通常受益明显；FAQ、摘要改写、模板回复、客服问答，通常不值得为 thinking token 付费。

一个最实用的判断公式是：

$$
\text{总输出计费 token} = \text{可见输出 token} + \text{thinking token}
$$

所以，Thinking 模式提升准确率的同时，也直接增加延迟和成本。预算控制不是可选优化，而是上线前必须做的基础治理。

---

## 问题定义与边界

先定义术语。

“显式推理链”，白话说，就是你能在结果外部观察到模型推理过程的一部分，而不是只看到答案。  
“thinking token”，白话说，就是模型在正式回答前用于内部思考的 token。  
“thinking budget”，白话说，就是你允许模型最多花多少 token 去思考。  
“thinking level”，白话说，就是不用精确填 token 数，而是用低、中、高这类档位控制推理强度。

Gemini 当前有两套控制面：

| 模型代际 | 主要控制参数 | 说明 |
|---|---|---|
| Gemini 2.5 系列 | `thinkingBudget` | 用 token 上限直接控预算 |
| Gemini 3 系列 | `thinkingLevel` | 用档位控制推理强度 |
| 兼容说明 | 不建议混用 | 官方建议 Gemini 3 用 `thinkingLevel`，2.5 用 `thinkingBudget` |

截至 2026 年 3 月 12 日，Google 官方文档给出的 2.5 系列预算边界是：

| 模型 | 最小值 | 最大值 | 关闭方式 |
|---|---:|---:|---|
| Gemini 2.5 Flash | 1 | 24576 | `thinkingBudget=0` |
| Gemini 2.5 Pro | 128 | 32768 | 不能关闭 |
| Gemini 2.5 Flash-Lite | 512 | 24576 | `thinkingBudget=0` |

Gemini 3 则改成 `thinkingLevel`。官方页面给出的典型枚举是 `minimal / low / medium / high`。这意味着你的请求校验器必须先按模型路由，再决定允许哪个字段进入请求体；否则轻则性能退化，重则直接报错。

这里的边界要讲清楚：

1. Thinking 模式不是训练阶段再学习一遍，而是推理阶段多花算力。
2. `includeThoughts` 不是“强制返回完整推理链”，而是“尽量返回摘要级思考内容”。
3. `thinkingBudget=0` 只对支持关闭思考的模型有效，不是所有模型都能关。
4. Flash-Lite 这类低成本模型，适合大规模轻推理或默认不依赖重思考的场景。
5. Thinking 模式与 fine-tuning 并非任意叠加。Vertex 文档明确提到，开启 thinking 时不支持 fine-tuning。

一个玩具例子最容易理解这个边界。

问题是“为什么 $12 \times 12 = 144$”。  
如果 `thinkingBudget=0`，模型通常会直接答“因为 12 乘以 12 等于 144”。  
如果给一个较高预算，比如 8000，模型更可能先做三步：

1. 把 $12 \times 12$ 拆成 $12 \times (10 + 2)$。
2. 计算 $12 \times 10 = 120$，$12 \times 2 = 24$。
3. 校验 $120 + 24 = 144$。

这里真正有价值的，不是答案本身，而是“拆解和校验被显式纳入生成过程”。

---

## 核心机制与推导

Thinking 模式的核心机制可以抽象成三层。

第一层是任务分解。  
模型先判断这是不是一个需要多步推理的问题。比如证明题、约束满足、跨文件代码定位、复杂工具调用，一般会触发更多思考。

第二层是候选路径生成。  
模型不会只沿一条路径硬算到底，而是会生成若干中间假设，做局部验证，再淘汰不一致路径。白话说，就是“先打草稿，再删错稿”。

第三层是压缩输出。  
真正返回给用户的往往不是原始草稿，而是最终答案，外加可能出现的一段思考摘要。也就是说，思考过程参与了生成，但不等于全过程都被返回。

这解释了为什么 `usage_metadata` 比界面展示更重要。因为计费与监控看的是实际消耗，而不是你肉眼看到多少“思考文字”。

工程上可以把一次响应近似拆成：

$$
T_{\text{out}} = T_{\text{visible}} + T_{\text{thought}}
$$

其中：

- $T_{\text{visible}}$ 是用户看到的输出 token
- $T_{\text{thought}}$ 是模型内部思考 token

如果按单价 $p_{\text{out}}$ 对输出计费，那么输出侧费用近似为：

$$
C_{\text{out}} = p_{\text{out}} \cdot (T_{\text{visible}} + T_{\text{thought}})
$$

这也是为什么很多团队上线后会觉得“回答也没多长，为什么账单涨这么快”。问题不在可见输出，而在思考 token。

再看“显式推理链”和 OpenAI o1 的技术路线差异。

| 维度 | Gemini Thinking | OpenAI o1 |
|---|---|---|
| 推理是否存在 | 是 | 是 |
| 是否有预算旋钮 | 有，2.5 用 budget，3 用 level | 有 effort 类控制，但理念不同 |
| 是否稳定返回原始链路 | 否 | 否 |
| 是否支持摘要级暴露 | 是，`includeThoughts` | 是，官方强调摘要而非原始链路 |
| 产品感知 | 2.0 Flash Thinking 曾强调“显示思考” | 更强调隐藏原始 CoT 与安全监控 |

这里最容易误判的一点是：Gemini 2.0 Flash Thinking 的“显示思考”，在产品叙事上很强，但从 API 机制上看，后续正式文档仍然把可返回内容定义成 summary，而不是 raw chain。这说明 Google 的路线也没有走向“完整公开内部推理原文”，而是选择了“可观测但不完全暴露”的中间方案。

为什么这种方案对数学和编程任务更有效？因为这类任务的错误往往不是知识缺失，而是中间步骤出错。  
例如一道数学题，从最终答案看只有对或错；但从中间过程看，可能出现：

- 定义用错
- 边界漏掉
- 符号翻转
- 枚举不全
- 自检没做

Thinking 模式的作用，就是给模型更多机会在输出前发现这些中间错误。因此在 AIME、GPQA、代码修复这类任务上，增加测试时推理资源通常会带来明显收益。

一个真实工程例子是分布式系统审查。  
假设你让模型分析一段 trace，判断“订单服务和库存服务之间是否存在竞态条件”。这不是一句话问答，而是多步验证任务：

1. 识别跨服务调用顺序。
2. 判断写入是否有因果关系。
3. 检查重试与幂等键。
4. 识别是否存在“先读旧值再写新值”的窗口。
5. 给出证据链。

低预算下，模型可能只给出模糊判断；高预算下，它更可能做“事件排序 + 条件验证 + 反例排除”。Thinking 模式的收益，本质上来自这里。

---

## 代码实现

先给一个可运行的 Python 玩具实现。它不调用真实 API，而是把请求校验和成本估算逻辑先写清楚。这样你在接 SDK 前，就能先把工程边界固定住。

```python
from dataclasses import dataclass
from typing import Optional, Literal

ThinkingLevel = Literal["minimal", "low", "medium", "high"]

@dataclass
class ThinkingConfig:
    model: str
    thinking_budget: Optional[int] = None
    thinking_level: Optional[ThinkingLevel] = None
    include_thoughts: bool = False

def validate_config(cfg: ThinkingConfig) -> None:
    is_gemini3 = cfg.model.startswith("gemini-3")
    is_gemini25 = cfg.model.startswith("gemini-2.5")

    if is_gemini3 and cfg.thinking_budget is not None:
        raise ValueError("Gemini 3 应优先使用 thinkingLevel，不应传 thinkingBudget")
    if is_gemini25 and cfg.thinking_level is not None:
        raise ValueError("Gemini 2.5 应使用 thinkingBudget，不应传 thinkingLevel")

    if cfg.thinking_budget is not None:
        if cfg.model == "gemini-2.5-flash":
            assert cfg.thinking_budget in ([0, -1] + list(range(1, 24577)))
        elif cfg.model == "gemini-2.5-pro":
            assert cfg.thinking_budget in ([-1] + list(range(128, 32769)))
        elif cfg.model == "gemini-2.5-flash-lite":
            assert cfg.thinking_budget in ([0, -1] + list(range(512, 24577)))
        else:
            raise ValueError("未知 2.5 模型")

    if cfg.thinking_level is not None:
        assert cfg.thinking_level in ("minimal", "low", "medium", "high")

def estimate_output_cost_usd(output_price_per_million: float,
                             visible_tokens: int,
                             thought_tokens: int) -> float:
    total = visible_tokens + thought_tokens
    return output_price_per_million * total / 1_000_000

toy = ThinkingConfig(
    model="gemini-2.5-flash",
    thinking_budget=8000,
    include_thoughts=True
)
validate_config(toy)

fast = estimate_output_cost_usd(2.50, visible_tokens=200, thought_tokens=0)
deep = estimate_output_cost_usd(2.50, visible_tokens=200, thought_tokens=1800)

assert round(fast, 6) == 0.0005
assert round(deep, 6) == 0.005
assert deep > fast
```

这个例子表达了两件事：

1. 配置层必须先做模型级校验。
2. 成本估算必须把 thought tokens 算进去。

如果你要接官方 Python SDK，结构大致是这样：

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="证明 12×12=144，并给出分步解释。",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=8000,
            include_thoughts=True,
        )
    ),
)

print(response.text)
```

对应的真实工程例子，可以把 budget 封成策略函数，而不是在业务代码里到处写死：

```python
def choose_thinking_budget(task_type: str) -> int:
    policy = {
        "faq": 0,
        "summarization": 0,
        "code_review": 2048,
        "bug_debug": 4096,
        "math_proof": 8192,
        "cross_service_audit": 12288,
    }
    return policy.get(task_type, 1024)

assert choose_thinking_budget("faq") == 0
assert choose_thinking_budget("math_proof") == 8192
```

前端或 Node.js 侧也类似，关键不是 SDK 语法，而是把 `thinkingConfig` 纳入统一请求层，避免业务方随手把预算拉满。

---

## 工程权衡与常见坑

最重要的权衡只有三个：准确率、延迟、成本。

### 1. 不要把“能思考”误当成“应该思考”

很多任务从输入到输出本来就是单跳映射，比如：

- 改写一句文案
- 提取订单号
- 给错误码查说明
- 翻译短句
- 生成模板邮件

这些场景 Thinking 模式往往只会增加延迟和费用，收益很小。  
真正适合开思考的是那些“中间步骤决定正确性”的任务。

### 2. 不要只监控可见输出 token

这是最常见的成本坑。  
截至 2026 年 3 月 12 日，Google 官方 Gemini Developer API 定价页写得很明确：输出价格包含 thinking tokens。以官方当前标准价举例：

| 模型 | 输出价格（每 1M token） | 是否包含 thinking tokens |
|---|---:|---|
| Gemini 2.5 Flash | $2.50 | 是 |
| Gemini 2.5 Flash-Lite | $0.40 | 是 |
| Gemini 2.5 Pro | $10.00 | 是 |
| Gemini 3 Flash Preview | $3.00 | 是 |

所以如果你只看“回答文本长度”，成本估算会系统性偏低。  
正确做法是从响应里的 `usage_metadata` 读取总消耗，并单独记录 thought token 相关字段。

### 3. `includeThoughts` 适合调试，不适合当审计真相

它返回的是 thought summaries，还是 best-effort。也就是说：

- 可能返回
- 可能不返回
- 返回了也不等于完整原始推理
- 更不等于“模型一定按这段摘要那样思考过”

所以它适合做开发调试、提示词评审、教学展示，不适合当严格审计日志。

### 4. 预算过高会掩盖提示词问题

不少团队遇到“高预算效果更好”，就默认继续加预算。  
但很多时候，真正的问题是提示词没把验证标准写清楚。例如让模型“检查认证漏洞”，却没有明确要求：

- 先找身份边界
- 再找权限边界
- 再找 token 生命周期
- 再列出失败路径

这时加预算只是让模型更久地在模糊目标上思考，不一定更准。

### 5. 多轮工具调用时，要理解 reasoning state 的保存机制

Google 文档后来引入了 thought signatures。它的白话解释是“推理存档指纹”。模型调用工具时会暂停内部推理，signature 用来在下一轮恢复状态。  
如果你自己手写多轮函数调用，而不是完全依赖官方 SDK 自动处理，就不能随意丢掉这些字段，否则 reasoning 可能断档。

### 6. 真实工程里必须做任务分级

一个典型生产策略是：

| 任务类型 | 预算策略 | 目标 |
|---|---|---|
| FAQ / 客服 | 0 或 minimal | 最低延迟 |
| 结构化提取 | 0 或很低 | 稳定吞吐 |
| 常规代码解释 | 1024-2048 | 平衡成本 |
| Debug / 审查 / 数学 | 4096-12288 | 提升正确率 |
| 高风险复杂分析 | 按上限封顶并人工复核 | 控风险，不盲信模型 |

这比“默认全开 thinking”更可控。

---

## 替代方案与适用边界

Thinking 模式不是唯一解，它只是“用更多推理资源换更高正确率”的一条路线。至少还有四类替代方案。

### 1. 直接关闭思考

如果任务只是快速回复、改写、摘要、轻提取，最简单的替代方案就是 `thinkingBudget=0`，或者在 Gemini 3 上用最低思考等级。  
好处是延迟最低、成本最低，坏处是复杂任务正确率下降。

### 2. 换到 Flash-Lite

Flash-Lite 的定位就是低成本高吞吐。  
如果你的系统日调用量大，且大多数请求并不需要复杂多步推理，用 Flash-Lite 往往比在 Flash 上硬开 thinking 更合适。

### 3. 把“思考”外置到工作流，而不是内置到模型

这是很多工程团队后期会做的升级。  
不要让模型在一个请求里自己想完全部步骤，而是把任务拆成显式流程：

1. 检索证据
2. 结构化抽取
3. 规则校验
4. 生成结论

这样做的好处是可审计、可缓存、可回放。坏处是系统复杂度更高。

### 4. 使用其他 reasoning 模型，但接受不同透明度设计

OpenAI o1 的路线证明了：高推理能力不依赖公开原始链路。  
如果你的核心目标是最终正确率，而不是教学可解释性，那么只比较“能不能看到思考摘要”是不够的，应该同时比较：

- 延迟
- 成本
- 工具调用稳定性
- 长上下文表现
- 代码基准表现
- 多轮状态保持

可以做一个简化路由表：

| 模型类型 | 思考控制 | 适合任务 | 不适合任务 |
|---|---|---|---|
| Gemini 2.5 Flash | `thinkingBudget` | 数学、代码、复杂问答、审查 | 超低延迟批量回复 |
| Gemini 2.5 Flash-Lite | 可低思考或关闭 | 大规模轻推理、客服、分类 | 高难推理证明 |
| Gemini 2.5 Pro | 高质量复杂推理 | 高复杂代码、科学/数学分析 | 成本敏感的大规模流量 |
| Gemini 3 Flash/Pro | `thinkingLevel` | 需要档位控制和新一代功能 | 需要精确 token 级预算控制的旧系统 |

最终边界可以压缩成一句话：  
如果错误主要来自“中间步骤推理失败”，就考虑 Thinking；如果错误主要来自“知识没命中、数据没取到、规则没写清”，先别急着加 Thinking。

---

## 参考资料

- Google AI for Developers, Gemini API Thinking: https://ai.google.dev/gemini-api/docs/thinking
- Google Cloud Vertex AI Thinking 文档: https://cloud.google.com/vertex-ai/generative-ai/docs/thinking
- Google AI for Developers Pricing: https://ai.google.dev/pricing
- Google 官方博客，Gemini 2.0 Flash Thinking Experimental 更新，2025-03-13: https://blog.google/products-and-platforms/products/gemini/new-gemini-app-features-march-2025/
- Google 官方博客，Gemini 2.0 模型更新，2025-02-05: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-updates-february-2025/
- Google 官方博客，Gemini 2.5 thinking 模型更新，2025-03-25: https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/
- Google AI for Developers, Thought Signatures: https://ai.google.dev/gemini-api/docs/thought-signatures
- OpenAI, Learning to reason with LLMs, 2024-09-12: https://openai.com/index/learning-to-reason-with-llms/
- OpenAI API 模型页，o1: https://platform.openai.com/docs/models/o1
