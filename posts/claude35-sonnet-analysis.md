## 核心结论

Claude 3.5 Sonnet 真正重要的不是“更会聊天”，而是“更适合放进 agent 框架里做多步任务”。这里的 agent，可以先理解为“模型加一层工具循环”，也就是模型不只回答，还会反复观察环境、采取动作、读取结果、再继续决策。

从公开数据看，Anthropic 在 **2025 年 1 月 6 日**公布：升级版 Claude 3.5 Sonnet 在 SWE-bench Verified 上达到 **49.0%**；Anthropic 在 **2024 年 10 月 22 日**发布升级版时，也给出了从 **33.4% 提升到 49.0%** 的对比。同一基准上，OpenAI 在 **2024 年 8 月 6 日**公布 GPT-4o 为 **33.2%**。这说明至少在“给定代码库、要修改、要测试、要多步迭代”的 agentic 编码任务上，升级版 Claude 3.5 Sonnet 的公开成绩明显更强。  
数学上可以把差距写成：

$$
\Delta_{\text{SWE}} = 49.0\% - 33.2\% = 15.8\%
$$

如果只看 Anthropic 在 2024 年 10 月 22 日新闻稿里引用的旧版 Sonnet 33.4%，那么相对提升为：

$$
49.0\% - 33.4\% = 15.6\%
$$

Computer Use 则把这种“多步决策”从代码终端扩展到图形界面。它的基本模式不是一次性生成整套脚本，而是：

$$
Observation_k \rightarrow Reasoning_k \rightarrow Action_{k+1} \rightarrow Observation_{k+1}
$$

也就是“看一眼屏幕，做一步操作，再看新屏幕”。这类循环的价值在于，它把 GUI 自动化从硬编码脚本，变成了可根据现场画面动态修正的策略。

但要先把一个常见误解说清楚：今天 Anthropic 文档里的 **Extended Thinking**，是对 **Claude 3.7 和 Claude 4 系列**公开支持的 API 能力，不是 2024 年那版 Claude 3.5 Sonnet 的正式公开特性。因此，讨论“Claude 3.5 Sonnet 的混合推理”时，准确说法应是：**3.5 Sonnet 已经表现出更强的 agentic 推理能力，但公开可配置的 `thinking` 块接口，是后续版本文档化、产品化的能力。**

---

## 问题定义与边界

这篇文章讨论的问题，不是“Claude 3.5 Sonnet 聪不聪明”，而是它为什么在两类任务上特别值得分析：

| 任务类型 | 任务本质 | 关键能力 |
|---|---|---|
| Agentic 编码 | 读仓库、改代码、跑测试、修回归 | 长链路推理 + 工具调用 |
| Computer Use | 看屏幕、点按钮、输文字、反复确认 | 视觉理解 + 动作规划 |

这里的“混合推理”，白话解释就是：**模型不是只靠一次文本生成完成任务，而是把内部判断和外部执行穿插起来**。任务越长、环境越不确定，这种能力越重要。

边界也要明确。

第一，SWE-bench Verified 衡量的是“模型加 scaffold”的整体效果，不是纯模型裸跑分数。这里的 scaffold，可以理解成“围绕模型搭的执行框架”，负责提示词、工具调用、日志回写、错误恢复。Anthropic 自己就在 SWE-bench 文章里强调，agent 的外壳会显著影响成绩。

第二，Computer Use 在 **2024 年 10 月 22 日**上线时，Anthropic 明确说它还“experimental”且“error-prone”。当时官方特别指出，**scrolling、dragging、zooming** 都是挑战项。到了后续文档，工具接口已经列出 drag 支持，但这并不意味着 3.5 Sonnet 上线初期的能力边界消失了。做历史分析时，必须按发布日期理解能力。

第三，Extended Thinking 不能被简单地回填到 Claude 3.5 Sonnet 身上。今天文档中 `thinking: { type: "enabled", budget_tokens: N }` 的接口，适用于 3.7/4 系列。对 3.5 Sonnet，我们可以做机制推断，但不能把后来的公开 API 当成当时已发布事实。

一个玩具例子：让模型“打开浏览器，搜索 Python 官网，点击下载页，再回到首页”。  
这不是一道知识题，而是一串状态转移题。每一步都依赖新屏幕是否真的出现目标按钮，所以它天然适合 observation-action loop。

一个真实工程例子：在 CI 失败后，让 agent 打开测试报告页面，定位失败用例，回到本地仓库修改代码，再重新运行测试。  
这类任务要同时处理代码、终端输出和网页界面，单次生成脚本很脆弱，循环式 agent 更稳。

---

## 核心机制与推导

先看代码能力为什么会提升。

SWE-bench Verified 的任务本质是：给定仓库和 issue，模型要像工程师一样完成“理解问题、定位文件、修改实现、执行验证”。这不是单轮问答，而是多轮闭环。所以性能提升通常来自两层叠加：

1. 底层模型更会做长链路技术判断。
2. 外层 agent scaffold 更会调度工具与回写上下文。

Anthropic 在 SWE-bench 文章里把典型日志抽象成 `THOUGHT / ACTION / OBSERVATION`。这说明它的强项不是只会“想”，而是把“想”和“做”串起来。

可以把一次 agentic 编码任务写成以下递推：

$$
s_{k+1} = T(s_k, a_k, o_k)
$$

其中：

- $s_k$：第 $k$ 轮的任务状态，比如“已经读过哪些文件、测试过哪些分支”
- $o_k$：第 $k$ 轮观察到的新信息，比如报错栈、测试输出、屏幕截图
- $a_k$：第 $k$ 轮动作，比如编辑文件、执行测试、点击按钮
- $T$：状态更新函数，也就是 agent 框架如何把观察与动作整合进上下文

如果模型只会单轮回答，那么它近似只完成一次 $a_0$。  
如果模型具备稳定的 agent 循环，它才能连续完成 $a_0, a_1, a_2, ...$，直到收敛。

再看 Computer Use 的技术点。

Anthropic 官方文档把它定义为：提供截图、鼠标、键盘控制的桌面交互工具。关键不是“能点击”，而是“点击前先看屏幕，点击后再重新看”。这比传统 UI 自动化脚本更接近闭环控制系统。传统脚本常见问题是：按钮位置变了，脚本直接失效；而视觉闭环可以重新识别当前状态。

可以把 Computer Use 近似成一个带反馈的控制回路：

$$
a_{k+1} = \pi(x_k, h_k)
$$

其中：

- $x_k$：当前截图和界面状态
- $h_k$：历史动作与历史观察
- $\pi$：策略函数，也就是模型当前学到的决策规则

为什么这比纯脚本更强？因为 $\pi$ 不是写死坐标，而是根据当前页面状态重新决策。

但“Extended Thinking”这一层，需要分历史阶段理解。

今天文档里的可配置 thinking 机制，形式上是：

$$
\text{visible output} = \text{thinking blocks} + \text{text blocks}
$$

并且预算约束大致可以写成：

$$
\text{thinking\_budget} < \text{max\_tokens}
$$

在 3.7/4 系列里，这表示你能显式给模型更多推理 token。  
而对 Claude 3.5 Sonnet，更合理的推断是：**它已经在内部受益于更强的测试时推理与 scaffold 协作，但这种推理当时没有以今天的 `thinking` API 形式向开发者公开暴露。**

所以“Claude 3.5 Sonnet 的混合推理”可以分两层理解：

| 层次 | 是否对开发者显式可配 | 在 3.5 Sonnet 分析中的地位 |
|---|---|---|
| 内部测试时推理 | 否 | 可以合理推断存在 |
| 公开 `thinking` API | 否（3.5 时期） | 不能当作已发布事实 |
| scaffold 中的观察-动作循环 | 是 | 3.5 Sonnet 能力提升的核心组成 |

---

## 代码实现

下面先用一个可运行的 Python 玩具例子，模拟“截图 -> 判断 -> 点击”的极简 agent 循环。这里的 `screen` 不是图片，而是用字符串代替界面状态，便于零基础读者看懂。

```python
from dataclasses import dataclass

@dataclass
class State:
    screen: str
    steps: int = 0

def policy(screen: str) -> str:
    if "搜索框" in screen:
        return "type:Python 官网"
    if "搜索结果" in screen:
        return "click:python.org"
    if "首页" in screen and "Download" in screen:
        return "click:Download"
    return "finish"

def transition(state: State, action: str) -> State:
    if action == "type:Python 官网":
        return State(screen="搜索结果: python.org", steps=state.steps + 1)
    if action == "click:python.org":
        return State(screen="首页: Welcome to Python.org | Download", steps=state.steps + 1)
    if action == "click:Download":
        return State(screen="下载页: Python Releases", steps=state.steps + 1)
    return State(screen=state.screen, steps=state.steps + 1)

state = State(screen="浏览器首页: 搜索框")
history = []

for _ in range(5):
    action = policy(state.screen)
    history.append((state.screen, action))
    if action == "finish":
        break
    state = transition(state, action)

assert "下载页" in state.screen
assert state.steps == 3
assert history[0][1] == "type:Python 官网"
```

这个例子虽然简单，但已经包含了真实机制的骨架：

1. 先观察当前状态。
2. 按当前状态选动作。
3. 执行动作后进入新状态。
4. 重复直到结束。

如果换成真实 Anthropic Computer Use，请求结构会接近“模型输出动作，你的执行器负责真正点击，再把截图回传”。伪代码如下：

```python
def agent_loop(model, env, user_task, max_rounds=20):
    messages = [{"role": "user", "content": user_task}]
    for _ in range(max_rounds):
        screenshot = env.capture_screen()
        messages.append({"role": "user", "content": f"observation={screenshot}"})

        response = model(messages)  # 返回 click/type/scroll/wait 等动作
        action = response["action"]

        if action["type"] == "finish":
            return response["final_text"]

        env.execute(action)
        messages.append({"role": "assistant", "content": str(action)})

    raise RuntimeError("agent did not converge")
```

真实工程例子可以更贴近前端或测试团队：

- 用户任务：打开预发布环境。
- Agent 第 1 轮：截图，发现登录页，输入测试账号。
- 第 2 轮：截图，发现首页加载完成，点击“订单管理”。
- 第 3 轮：截图，发现筛选器，输入订单号。
- 第 4 轮：截图，发现状态异常，记录并生成缺陷摘要。

这类流程的难点不是某个动作本身，而是每一步都必须依赖上一步的真实结果。也正因为如此，模型的“推理”和“执行”不能分家。

如果把后来 3.7/4 的 extended thinking API 作为参考模板，它的预算控制可以抽象成：

```python
def allocate_tokens(max_tokens: int, thinking_budget: int) -> int:
    assert max_tokens > 0
    assert 0 < thinking_budget < max_tokens
    visible_answer_budget = max_tokens - thinking_budget
    assert visible_answer_budget > 0
    return visible_answer_budget

usable = allocate_tokens(16000, 10000)
assert usable == 6000
```

这段代码不是在说“3.5 Sonnet 当时已有这个 API”，而是在帮助读者理解后来 Anthropic 如何把“推理深度”正式参数化。

---

## 工程权衡与常见坑

最核心的工程权衡有三项：准确率、延迟、成本。

| 方案 | 准确率潜力 | 延迟 | 成本 | 适合场景 |
|---|---|---|---|---|
| 单轮文本生成 | 低到中 | 低 | 低 | 简单问答、短代码片段 |
| 代码工具循环 | 中到高 | 中 | 中 | 修 bug、跑测试、读日志 |
| GUI Computer Use 循环 | 中到高 | 高 | 高 | 无 API 的桌面或网页流程 |
| Extended Thinking + 工具 | 更高上限 | 更高 | 更高 | 高价值复杂任务 |

常见坑也很具体。

第一，把 benchmark 分数误读成“模型本体智商”。  
SWE-bench Verified 是 agent 系统成绩。模型强很重要，但 scaffold、工具权限、重试策略同样重要。

第二，把后续文档能力误写回旧版本。  
`thinking`、interleaved thinking、更新版 computer tool，很多是 2025 年之后文档逐步明确的。分析 2024 年 10 月的 Claude 3.5 Sonnet，必须按当时资料写。

第三，高估 GUI 自动化稳定性。  
屏幕上多一个弹窗、按钮轻微改版、网络慢半秒，都可能让下一步判断偏掉。这里的“脆弱”，白话解释就是：**环境一变，连续决策就可能连锁出错**。

第四，忽视安全边界。  
Anthropic 和 OpenAI 都在文档中强调，computer use 不适合高风险、全权限、强认证环境。最稳妥的做法仍然是：

- 跑在隔离 VM 或容器里
- 不给敏感账号
- 关键动作要求人工确认
- 保留完整执行日志

第五，把“会点网页”误认为“能稳定完成生产流程”。  
公开 benchmark 说明这类模型已经有明显进步，但离人类工程师的稳定性仍有距离。OpenAI 在 2025 年 1 月发布的 CUA 研究里给出 OSWorld **38.1%**；Anthropic 在 2024 年 10 月给出 Claude 3.5 Sonnet 在 OSWorld screenshot-only **14.9%**，允许更多步骤时 **22.0%**。这两组数字不能直接横比产品强弱，因为设置不同，但能共同说明一件事：**GUI agent 还处在“可用但不够稳”的阶段。**

---

## 替代方案与适用边界

如果任务只是“问一个知识点”或“补一小段函数”，Claude 3.5 Sonnet 的 agent 框架优势并不会完全发挥出来。因为这时主要矛盾不是多步执行，而是单轮生成质量。

更实用的选择矩阵如下：

| 任务复杂度 | 是否需要 GUI 控制 | 推荐思路 |
|---|---|---|
| 低 | 否 | 普通文本模型即可 |
| 中 | 否 | 代码模型 + bash / test 工具 |
| 中到高 | 是 | Claude 3.5 Sonnet 式 observation-action loop |
| 很高 | 是，且需更长推理 | 新一代 thinking 模型或专门 agent 框架 |

和 GPT-4o 的对比也应该分两层看。

在 **agentic 编码** 上，公开 SWE-bench Verified 数据里，升级版 Claude 3.5 Sonnet 的 49.0% 明显高于 GPT-4o 的 33.2%。如果你的核心任务是“读仓库、改代码、跑测试、修回归”，Claude 3.5 Sonnet 这一路线在当时公开数据里更占优。

在 **computer use** 上，OpenAI 后续把 GPT-4o 视觉能力和强化后的推理结合成 CUA，用在 Operator 与 API 的 `computer-use-preview`。这说明 OpenAI 的路线是“以 4o 视觉为基础，额外训练专门 computer-using agent”；Anthropic 的路线则更早把 computer use 直接挂到 Claude 3.5 Sonnet 这一主力模型上。两者共同点是都采用截图反馈回路，不同点在于产品形态和训练打磨重点。

所以结论不是“谁全方面碾压谁”，而是：

- 要做 **代码仓库里的多步修复**，Claude 3.5 Sonnet 的公开 benchmark 结论更强。
- 要做 **通用 GUI 代理**，两家都在走 observation-action loop，但都仍需人工监控。
- 要做 **显式可控的长推理**，今天应看 Claude 3.7/4 的 thinking API，而不是把它直接套回 3.5 Sonnet。

---

## 参考资料

- Anthropic, *Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku*, 2024-10-22: https://www.anthropic.com/news/3-5-models-and-computer-use
- Anthropic, *Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet*, 2025-01-06: https://www.anthropic.com/research/swe-bench-sonnet
- Anthropic Docs, *Computer use tool*: https://docs.anthropic.com/en/docs/build-with-claude/computer-use
- Anthropic Docs, *Building with extended thinking*: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
- OpenAI, *Introducing SWE-bench Verified*, 2024-08-06: https://openai.com/index/introducing-swe-bench-verified/
- OpenAI, *GPT-4o System Card*: https://openai.com/index/gpt-4o-system-card/
- OpenAI, *Computer-Using Agent*, 2025-01-23: https://openai.com/index/computer-using-agent/
- OpenAI Docs, *Computer use*: https://platform.openai.com/docs/guides/tools-computer-use
