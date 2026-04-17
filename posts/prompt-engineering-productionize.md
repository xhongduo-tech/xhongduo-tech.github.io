## 核心结论

提示工程工程化，指的是把 Prompt 从“代码里的长字符串”升级成“可管理的生产配置”。最核心的做法有三件事：

第一，把 Prompt 拆成模板、变量、模型参数三部分。模板就是带占位符的固定文本，变量就是运行时填进去的数据，模型参数就是 `temperature`、`max_tokens` 这类调用配置。这样可以用一个公式描述：

$$
Prompt = Render(Template, Vars)
$$

也就是：先有模板，再用变量渲染，最后得到真正发给模型的 Prompt。

第二，把 Prompt 纳入版本控制。版本控制就是“记录每次改了什么、谁改的、何时改的，并且能回退”。Git 适合做这件事，因为 Prompt 的本质仍然是文本。每次变更不仅要提交文本差异，还要记录元数据，例如目标任务、作者、实验标签、预期指标。

第三，把测试和 A/B 实验接到 Prompt 版本后面。A/B 测试就是把流量分给两个版本，看哪个效果更好。没有测试的 Prompt 调整，本质上是在生产环境盲改；没有指标闭环的优化，本质上只是主观感觉。工程上应形成 `Version -> Tests -> Metrics` 的闭环：有版本，先测，再灰度，再放量。

一个最小玩具例子很直观：

- 模板：`请计算 {{price}}*{{discount}} 的总价`
- 变量：`{price: 100, discount: 0.15}`
- 渲染结果：`请计算 100*0.15 的总价`

这个例子看似简单，但它已经说明了工程化的根：文本和数据分离后，才有自动测试、复用、灰度和回滚的空间。

---

## 问题定义与边界

先定义问题。这里讨论的不是“怎么写出更聪明的 Prompt”，而是“怎么让 Prompt 在团队和生产环境里稳定可控”。边界很重要，因为很多团队把提示工程理解成“一个人不断试文案”，这只适合原型，不适合线上系统。

在生产环境里，Prompt 至少承担四类责任：

| 问题 | 风险 | 解决方式 |
| --- | --- | --- |
| 无版本 | 出问题时不知道改动来源，无法快速回退 | `Git + Prompt Registry + metadata` |
| 无测试 | 变量漏填、格式变更、输出跑偏直接进生产 | `promptfoo eval + CI` |
| 无指标 | 只知道“感觉变好了”，不知道准确率和成本是否下降 | 指标采集与告警 |
| 无发布机制 | 新 Prompt 一次性全量上线，风险集中爆发 | 灰度发布与 A/B 实验 |

这里的 Prompt Registry 可以理解成“提示词注册表”，白话说就是一个专门存 Prompt 的仓库或服务。它和 Git 不冲突。Git 解决源码追踪和审计，Registry 解决运行时取哪个版本、给哪个标签、让非工程师也能参与编辑的问题。

问题边界也要说明清楚：

- 如果你只是本地试验、单人开发、没有线上用户，那么纯 Git 文件管理可能已经够用。
- 如果你有多环境发布、多人协作、合规审计、线上灰度，就不能只靠文档和口头同步。
- 如果 Prompt 输出影响用户决策、账单、客服答复、结构化数据写入数据库，那么测试和回滚能力不是加分项，而是基本能力。

一个典型反例是：团队把 Prompt 直接写死在 Python 代码里，工程师临时改一句话后发版。第二天效果下降，团队只能猜是模型波动、数据变化还是 Prompt 改坏了。这个问题不是“Prompt 不够好”，而是“Prompt 没进入工程系统”。

---

## 核心机制与推导

Prompt 工程化可以拆成一个稳定的机制链条：

$$
\text{Template} + \text{Vars} + \text{LLMConfig} \rightarrow \text{Rendered Prompt} \rightarrow \text{Evaluation} \rightarrow \text{Release}
$$

其中：

- Template：模板，白话说就是可复用的固定骨架。
- Vars：变量，白话说就是每次请求不同的数据。
- LLMConfig：模型配置，白话说就是控制模型输出风格和成本的参数。
- Evaluation：评测，白话说就是自动判断结果是否达标。
- Release：发布，白话说就是决定哪个版本真正接收线上流量。

为什么一定要拆？因为如果模板、变量、配置混在一起，就无法分别验证。比如一个输出变差，可能是模板措辞改了，也可能是 `temperature` 从 `0.2` 变成 `0.8`，还可能是变量名拼错导致上下文缺失。只有分层记录，问题才能定位。

### 1. 模板管理

Jinja2 这类模板引擎适合做 Prompt 模板管理。它的核心价值不是“语法优雅”，而是把结构和数据分离。术语上这叫“模板化”，白话说就是先写一个带坑位的标准文本，再把数据填进去。

玩具例子：

- 模板：`请计算 {{ price }} * {{ discount }} 的总价，只输出数字。`
- 输入：`price=100, discount=0.15`
- 渲染后：`请计算 100 * 0.15 的总价，只输出数字。`

这一步的好处有两个：

- 同一个模板可以复用到很多请求。
- 可以单独测试“变量是否都填上了”。

### 2. 版本控制

Prompt 版本控制不只是“存历史”，而是让每次改动都带上下文。建议最少记录：

- `slug`：唯一标识
- `version`：版本号
- `author`：谁改的
- `purpose`：这次为什么改
- `model`：配套模型
- `metrics`：期望改善的指标
- `label`：比如 `dev`、`staging`、`prod`

可以把它理解成一条规则：

$$
\text{可回滚性} = \text{版本标识} + \text{变更说明} + \text{可复现实验}
$$

只有文本 diff，没有测试和指标，不算真正可回滚。因为你能回到旧文本，不代表你知道为什么那个版本更好。

### 3. 自动评测

Prompt 的测试和普通函数测试不完全一样。普通函数往往要求确定性输入输出；大模型输出有概率性，所以评测应分层：

| 测试类型 | 检查对象 | 例子 |
| --- | --- | --- |
| 渲染测试 | 模板变量是否完整替换 | 是否还残留 `{{price}}` |
| 结构测试 | 输出是否符合格式 | 是否为合法 JSON |
| 语义测试 | 输出是否满足任务目标 | 是否包含正确字段、关键结论 |
| 回归测试 | 新版本是否比旧版本更差 | 历史样本通过率是否下降 |

`promptfoo` 这类工具的价值就在这里。它不是替你写 Prompt，而是把 Prompt 评测变成可执行脚本。

### 4. A/B 发布

真实工程里，Prompt 不是改完就全量上线，而是按比例放流量，比如 5%、20%、50%、100%。A/B 的本质不是“试试看”，而是控制变量。通常要求：

- 模型版本固定
- 数据切片可追踪
- 指标一致采集
- 回滚按钮明确

真实工程例子：一个 SaaS 客服系统有“工单摘要 Prompt”。团队把旧版本 `v12` 作为 `prod`，新版本 `v13` 作为 `candidate`。先给 `v13` 分 5% 流量，观察三个指标：

- 摘要正确率
- 平均 token 成本
- 用户人工修订率

如果 `v13` 正确率提升 3%，但 token 成本增加 40%，那么是否放量要看业务目标。如果这是高价值工单场景，可能值得；如果是低价订阅产品，成本上升可能不可接受。Prompt 工程化的关键不是“只看效果”，而是把效果、成本、稳定性一起算。

---

## 代码实现

下面给一个最小可运行实现。它不依赖真实 Registry 服务，但结构已经和生产系统一致：模板存储、版本信息、渲染、测试。

```python
from dataclasses import dataclass
from jinja2 import Template

@dataclass
class PromptVersion:
    slug: str
    version: str
    template: str
    model: str
    temperature: float
    label: str

class InMemoryPromptRegistry:
    def __init__(self):
        self._store = {}

    def register(self, prompt: PromptVersion):
        self._store[(prompt.slug, prompt.label)] = prompt

    def get(self, slug: str, label: str = "prod") -> PromptVersion:
        prompt = self._store.get((slug, label))
        if prompt is None:
            raise KeyError(f"Prompt not found: slug={slug}, label={label}")
        return prompt

def render_prompt(template_str: str, variables: dict) -> str:
    return Template(template_str).render(**variables)

def run_basic_assertions(rendered: str):
    # 基础回归检查：变量占位符不应残留
    assert "{{" not in rendered and "}}" not in rendered
    # 这个玩具例子要求包含计算表达式
    assert "100" in rendered
    assert "0.15" in rendered

registry = InMemoryPromptRegistry()
registry.register(
    PromptVersion(
        slug="invoice-calc",
        version="v1.2.0",
        template="请计算 {{ price }} * {{ discount }} 的总价，只输出数字。",
        model="gpt-4.1-mini",
        temperature=0.0,
        label="prod",
    )
)

prompt = registry.get("invoice-calc", label="prod")
rendered = render_prompt(prompt.template, {"price": 100, "discount": 0.15})
run_basic_assertions(rendered)

assert rendered == "请计算 100 * 0.15 的总价，只输出数字。"
print(rendered)
```

如果接入真实生产系统，结构一般是下面这样：

```python
from jinja2 import Template

def build_prompt(template_text: str, variables: dict) -> str:
    return Template(template_text).render(**variables)

def call_llm(client, model: str, prompt: str, temperature: float):
    return client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        metadata={
            "prompt_slug": "invoice-calc",
            "prompt_version": "v1.2.0",
            "experiment": "discount-format-ab",
        },
    )

template_text = "请计算 {{ price }} * {{ discount }} 的总价，只输出数字。"
variables = {"price": 100, "discount": 0.15}
rendered_prompt = build_prompt(template_text, variables)

# response = call_llm(client, "gpt-4.1-mini", rendered_prompt, 0.0)
```

再看一个 `promptfoo` 配置示意，用来做 CI 评测：

```yaml
description: invoice-calc regression test
prompts:
  - "请计算 {{price}} * {{discount}} 的总价，只输出数字。"
providers:
  - id: openai:gpt-4.1-mini
tests:
  - vars:
      price: 100
      discount: 0.15
    assert:
      - type: contains
        value: "15"
```

CI 里通常会这样执行：

```bash
npx promptfoo eval -c promptfooconfig.yaml --fail-on-error
npx promptfoo eval -c promptfooconfig.yaml --output report.html
```

这里的 `--fail-on-error` 含义很直接：一旦评测失败，流水线直接失败，阻止坏版本合并。`report.html` 则用于给工程师或产品同学查看测试报告。

真实工程例子可以再复杂一点。比如“工单摘要 Prompt”系统：

- Registry 中存两版模板：`ticket-summary:v12` 和 `ticket-summary:v13`
- 运行时按标签取：`prod -> v12`，`candidate -> v13`
- 路由层把 5% 流量打到 `candidate`
- 模型调用时把 `prompt_version`、`experiment_id` 写入 metadata
- 日志系统按版本聚合准确率、修订率、token 成本
- 如果 `v13` 通过阈值，就把 `prod` 标签切到 `v13`

这时 Prompt 不再是一段散落文本，而是一条完整的可发布工件。

---

## 工程权衡与常见坑

工程化不是没有成本，它的代价是流程变重，但收益是稳定性显著提高。常见坑主要有下面几类：

| 坑 | 影响 | 缓解机制 |
| --- | --- | --- |
| Prompt 写死在代码里 | 文本修改必须随应用发版，回滚慢 | 抽到模板文件或 Registry |
| 只看主观效果，不做自动测试 | 小改动可能破坏格式或变量完整性 | `promptfoo + CI` |
| 版本只有文本，没有指标 | 知道改了什么，不知道是否更好 | 绑定 metrics 与实验标签 |
| 一次性全量上线 | 错误影响全部用户 | 灰度发布和 A/B |
| 不记录模型参数 | 同样 Prompt 在不同参数下结果不可比 | 模板版本和模型配置一起存档 |
| 不留输入样本集 | 回归时无法复现历史问题 | 固定评测集和线上抽样集 |

最常见的坑是把 Prompt 当成“纯文案问题”。实际上线上 Prompt 至少同时受四个变量影响：

$$
Output = f(Template, Vars, Model, Params)
$$

也就是说，输出是模板、变量、模型和参数共同作用的结果。你只改模板，却不固定模型和参数，实验结论就不干净。

第二个坑是测试过于理想化。很多团队只用一两个样例验证，结果线上遇到长文本、多语言、缺字段输入就崩。应至少准备三类样本：

- 正常样本：代表最常见请求
- 边界样本：空字段、超长文本、异常格式
- 历史失败样本：以前出过问题的数据

第三个坑是过度追求 Prompt 调优，忽略系统设计。有些问题不是换一句 Prompt 能解决，而是检索上下文质量差、输入结构不稳定、模型选型不合适。Prompt 工程化的目标是把问题暴露清楚，而不是把所有问题都压到 Prompt 上。

一个常见新手场景是：改了折扣计算 Prompt 后，忘了更新测试，PR 被 CI 拦截，报告显示 `contains '15'` 失败。这个拦截看起来增加了开发步骤，但它避免了“金额算错”直接进生产。对于财务、客服、审批类系统，这种阻断是必要的。

---

## 替代方案与适用边界

不是所有团队都需要一上来就用完整的 Prompt Registry。选型应看协作复杂度、上线风险和团队角色。

| 方案 | 门槛 | 非工程师编辑 | 自动 A/B | CI 集成 |
| --- | --- | --- | --- | --- |
| Notion/Google Docs 手工管理 | 最低 | 支持 | 不支持 | 不支持 |
| Git + 本地模板文件 | 中 | 基本不支持 | 需自己开发 | 支持 |
| Git + metadata + promptfoo | 中高 | 弱支持 | 需部分自研 | 强支持 |
| Prompt Registry 平台 | 较高 | 强支持 | 通常支持 | 通常支持 |

如果是单人项目或早期原型，直接在 `prompts/` 目录下放模板文件，再用 Git 管理，已经比把 Prompt 写死在业务代码里好很多。例如：

- `prompts/marketing/sales.md`
- `prompts/support/ticket_summary.md`

每次提交时写清楚 commit message，例如：`add output json schema for ticket summary`。同时在 `promptfooconfig.yaml` 补对应测试，这就是一个可落地的最小工程化起点。

什么时候该上 Registry？

- 需要产品、运营、内容同学参与编辑时
- 需要运行时按标签切换版本时
- 需要灰度和 A/B 实验时
- 需要审计和发布记录时

什么时候不该过度设计？

- Prompt 只用于内部脚本
- 没有稳定流量
- 任务还没跑通，需求每天变
- 团队连基本样本集都没整理出来

结论是：先把 Prompt 从代码里抽出来，再把 Git、测试、指标逐层补上；当协作和发布复杂度继续上升，再引入 Registry。顺序不要反，否则容易平台先行、流程空转。

---

## 参考资料

- AWS Well-Architected Generative AI Lens, GENOPS03-BP01: https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/genops03-bp01.html
- PromptLayer Prompt Registry 文档: https://docs.promptlayer.com/features/prompt-registry
- PromptLayer Prompt Registry Overview: https://docs.promptlayer.com/features/prompt-registry/overview
- PromptLayer Prompt Management: https://www.promptlayer.com/platform/prompt-management
- ManagePrompts Prompt Version Control Guide: https://manageprompts.com/blog/prompt-version-control-guide
- Promptfoo Getting Started: https://www.promptfoo.dev/docs/getting-started/
- Promptfoo CI/CD Integrations: https://www.promptfoo.dev/docs/integrations/ci-cd/
- KumoHQ Prompt Engineering Best Practices: https://www.kumohq.co/blog/prompt-engineering-best-practices
