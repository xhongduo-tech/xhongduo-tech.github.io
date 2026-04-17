## 核心结论

ReAct 的核心不是“会思考”，而是把**观察、推理、执行**串成闭环。放到多模态场景里，它的扩展方式也很直接：把原本只有文本的 `Observation` 扩成“截图、可访问树、Set-of-Marks 标注、搜索结果、历史动作”等联合输入，再让模型继续按 `obs_t -> thought_t -> action_t` 的节奏迭代。

这里的“多模态”可以先用一句白话理解：**模型不只读字，还要看屏幕、认控件、决定点哪里**。这使它能处理网页导航、表单填写、视觉搜索、桌面 UI 操作这类任务。

真正决定效果上限的，通常不是 planning，而是 **grounding**。grounding 的白话解释是：**把“点搜索框”这种语言描述，准确映射成页面上的具体元素或像素坐标**。如果这一步不准，再好的推理也会在执行阶段失真。SeeAct 明确指出，最佳 grounding 策略相对 oracle grounding 仍有约 20 到 25 个百分点的差距，这说明多模态 ReAct 的瓶颈常常出在“想对了，但点错了”。

一个最容易理解的玩具例子是“点击 iPhone 分类”。系统先给模型当前截图和元素树，模型推理出“顶部导航里应该有 iPhone 标签”，再把这个动作交给视觉 grounding 模块去找对应区域，输出 `click(x, y)`，点击后重新截屏，再进入下一轮观察。整个过程仍然是 ReAct，只是 Observation 和 Action 都变“带视觉”了。

VisualWebArena 的基准结果也说明了这一点。文本 only 的模型在视觉网页任务上成功率极低，而引入视觉输入后成功率明显上升。下面给一个有代表性的对比：

| 设定 | 输入 | 模型 | VisualWebArena 总体成功率 |
|---|---|---:|---:|
| Text-only | Accessibility Tree | LLaMA-2-70B | 1.10% |
| Text-only | Accessibility Tree | GPT-4 | 7.25% |
| Caption-augmented | Acc. Tree + Caption | GPT-4 | 12.75% |
| Multimodal | 截图 + 文本上下文 | GPT-4V | 15.05% |
| Multimodal + SoM | 截图 + SoM + 文本上下文 | GPT-4V | 16.37% |

结论可以压缩成两句：

1. 多模态 ReAct 把视觉、DOM、搜索结果、历史轨迹统一进同一个推理-行动循环，因此比文本 only 更适合界面任务。  
2. 视觉 grounding 不是附属模块，而是执行正确率的决定因素；SoM、坐标对齐、专门的 grounding 模型会直接影响最终任务成功率。

---

## 问题定义与边界

这类系统要解决的问题，不是“让大模型理解图片”这么宽泛，而是更具体的：

**给定一个用户任务，让代理在只能看见界面的前提下，逐步决定下一步操作，并把操作真正落到屏幕上的目标元素。**

这里有两个边界必须先说清。

第一，代理通常拿不到“可直接调用的真按钮对象”。它看到的往往只是以下几类 Observation：

| Observation 类型 | 白话解释 | 典型用途 |
|---|---|---|
| Screenshot | 当前屏幕截图 | 看布局、颜色、图标、位置关系 |
| Accessibility Tree / DOM 摘要 | 页面结构化文字描述 | 读标签、表单名、按钮文本 |
| SoM（Set-of-Marks） | 给可交互元素加编号或标记 | 降低“找不到目标”的难度 |
| 历史动作轨迹 | 前几步做了什么 | 防止重复点错 |
| 外部搜索结果 | 搜索引擎或站内检索返回内容 | 支持检索型任务 |

第二，这里的 Action 也不是无限开放的。工程里常见的是：

- `click(element or x,y)`
- `type(text, target)`
- `select(option, target)`
- `scroll(direction or distance)`
- `search(query)`
- `answer(text)`

也就是说，系统的边界更像是：**模型负责决定“做什么”，执行器负责把它变成浏览器、桌面或移动端的真实事件**。如果没有可用的截图、可访问树、mask 或坐标标注，很多动作就无法稳定落地。

可以把整个流程抽象成下面这个简化图：

$$
\text{Observation}_t \rightarrow \text{LLM Reasoning}_t \rightarrow \text{Action}_t \rightarrow \text{Grounding}_t \rightarrow \text{Observation}_{t+1}
$$

新手最容易误解的一点是：多模态 ReAct 不是“直接让模型看图然后自己点”。更准确的说法是：

1. 模型先基于图像和文本理解当前状态。  
2. 模型产出一个动作意图，例如“点击顶部导航中的 iPhone”。  
3. grounding 模块把这个意图映射到页面里的具体框或坐标。  
4. 控制器执行点击，生成新的观察。  

所以它本质上是一个**决策层和执行层分离**的系统。

---

## 核心机制与推导

标准 ReAct 的形式是：

$$
obs_t \rightarrow thought_t \rightarrow action_t \rightarrow obs_{t+1}
$$

其中：

- `obs_t` 是第 $t$ 步看到的信息。
- `thought_t` 是第 $t$ 步的中间推理，也就是子目标。
- `action_t` 是第 $t$ 步实际输出的操作。

在多模态场景里，这个式子只需要做一次扩展：

$$
obs_t^{multi} = \{I_t, D_t, M_t, H_t\}
$$

其中：

- $I_t$ 表示截图图像。
- $D_t$ 表示 DOM 或 accessibility tree。
- $M_t$ 表示 SoM、bounding box、mask 等视觉标记。
- $H_t$ 表示历史动作和环境反馈。

于是代理的每一步可以写成：

$$
(I_t, D_t, M_t, H_t) \rightarrow thought_t \rightarrow a_t^{intent} \rightarrow g(a_t^{intent}, I_t, M_t) \rightarrow a_t^{exec} \rightarrow obs_{t+1}
$$

这里多出来的 $g(\cdot)$ 就是 grounding 模块。它的作用是把“意图动作”变成“可执行动作”。

例如模型先生成：

- `thought_t`: 当前在 Apple 首页，目标是进入 iPhone 分类页。
- `a_t^{intent}`: 点击顶部导航中的 “iPhone”。

这还不够执行。系统还要继续做：

- `g(...)`: 在截图或 SoM 标注中定位 “iPhone” 对应区域。
- `a_t^{exec}`: `click(842, 116)` 或 `click(mark_id=7)`。

这一步为什么单独拿出来讲？因为“动作决策正确”并不等于“动作执行正确”。

### 玩具例子

任务：在一个商品站点首页点击“搜索框”。

当前 Observation：

- 截图里顶部有 logo、搜索框、登录按钮。
- accessibility tree 里有 `textbox name="Search products"`。
- SoM 把搜索框标成 `#3`。

模型可能这样推：

1. 页面已经在首页，不需要先导航。  
2. 用户下一步大概率是输入关键词。  
3. 当前最关键子目标是聚焦搜索框。  

于是输出动作意图：

```text
Action: click the search box at the top center
```

grounding 模块再把它落成：

```text
ActionExec: click(mark=3)
```

如果没有 SoM，只能从截图里自己找搜索框，模型更容易把登录框、订阅框或站内其他输入框看成目标。

### 真实工程例子

VisualWebArena 里的真实网页任务常常需要同时利用结构信息和视觉信息。比如任务是“进入 iPhone 分类并比较商品”。如果只给文本树，模型知道页面上可能有一个 `link: iPhone`，但它不知道：

- 这个元素当前是否可见；
- 是否被弹窗遮挡；
- 页面是否存在多个同名元素；
- 当前视口里哪个才是顶部导航的目标入口。

如果提供截图和 SoM，模型就能把“语言上的 iPhone”与“屏幕上的 iPhone”对齐。这个对齐过程就是视觉 grounding 的价值所在。

从工程角度看，决策误差通常来自三层：

| 误差层 | 具体问题 | 后果 |
|---|---|---|
| 感知误差 | 没看见弹窗、遮挡、禁用状态 | 模型以为按钮可点，实际无效 |
| 推理误差 | 子目标判断错 | 点击了不该点击的路径 |
| grounding 误差 | 语言意图没映射到正确坐标 | 思路对，但执行错 |

所以多模态 ReAct 的收益，不只是“多看一张图”，而是把这三层误差拆开，并允许分别优化。

---

## 代码实现

下面给一个最小闭环实现。它不是完整浏览器代理，但能把多模态 ReAct 的工程骨架表达清楚。

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Observation:
    screenshot_path: str
    accessibility_tree: str
    som_marks: Dict[str, tuple]
    history: List[str]

@dataclass
class ActionIntent:
    action_type: str
    target_desc: str
    text: Optional[str] = None

@dataclass
class ExecutableAction:
    action_type: str
    coords: Optional[tuple] = None
    text: Optional[str] = None

def react_planner(obs: Observation, task: str) -> ActionIntent:
    tree = obs.accessibility_tree.lower()
    if "iphone" in task.lower() and "iphone" in tree:
        return ActionIntent(action_type="click", target_desc="iphone nav item")
    if "search" in tree:
        return ActionIntent(action_type="click", target_desc="search box")
    return ActionIntent(action_type="scroll", target_desc="scroll down")

def grounding(intent: ActionIntent, obs: Observation) -> ExecutableAction:
    normalized = intent.target_desc.lower()
    for mark, coords in obs.som_marks.items():
        if mark.lower() in normalized or normalized in mark.lower():
            return ExecutableAction(action_type=intent.action_type, coords=coords, text=intent.text)

    # 简化策略：按关键词做回退匹配
    aliases = {
        "iphone nav item": "iphone",
        "search box": "search",
    }
    keyword = aliases.get(normalized, normalized)
    for mark, coords in obs.som_marks.items():
        if keyword in mark.lower():
            return ExecutableAction(action_type=intent.action_type, coords=coords, text=intent.text)

    return ExecutableAction(action_type="scroll", coords=None, text=None)

def execute(action: ExecutableAction) -> str:
    if action.action_type == "click" and action.coords:
        return f"clicked at {action.coords}"
    if action.action_type == "type" and action.coords:
        return f"typed '{action.text}' at {action.coords}"
    return "scrolled"

def step(task: str, obs: Observation) -> str:
    intent = react_planner(obs, task)
    exec_action = grounding(intent, obs)
    return execute(exec_action)

obs = Observation(
    screenshot_path="apple_home.png",
    accessibility_tree="navigation link Mac link iPhone link iPad textbox Search",
    som_marks={
        "iphone": (842, 116),
        "search": (1090, 118),
    },
    history=[]
)

result = step("Go to iPhone category", obs)
assert result == "clicked at (842, 116)"
print(result)
```

这个代码块里有四个关键点：

1. `Observation` 同时容纳截图路径、结构树、SoM 和历史轨迹。  
2. `react_planner` 只负责生成动作意图，不直接输出像素坐标。  
3. `grounding` 负责把描述映射为实际坐标。  
4. `execute` 执行后，系统应重新采集 Observation，进入下一轮。

如果把它写成更接近实际工程的伪代码，流程通常是这样：

```python
while not done and step_id < max_steps:
    obs = collect_observation(
        screenshot=capture_screenshot(),
        a11y_tree=get_accessibility_tree(),
        som=detect_set_of_marks(),
        history=trajectory,
    )

    thought, action_intent = llm.react(task, obs)

    action_exec = grounding_model.resolve(
        screenshot=obs["screenshot"],
        som=obs["som"],
        action=action_intent,
    )

    env_feedback = ui_controller.execute(action_exec)
    trajectory.append((obs, thought, action_intent, action_exec, env_feedback))
```

一个常见的配置表如下：

| 配置项 | 含义 | 常见取值 |
|---|---|---|
| `observation.screenshot` | 当前截图 | `png/jpg` |
| `observation.a11y_tree` | 可访问树 | 截断后的文本 |
| `observation.som` | 标注元素或 mask | `id -> bbox/point` |
| `planner.model` | 决策模型 | GPT-4o、Qwen2-VL 等 |
| `grounder.model` | grounding 模型 | UGround、规则匹配、候选重排 |
| `action.type` | 动作类型 | `click/type/select/scroll/search` |
| `action.target` | 语言级目标 | “顶部导航中的 iPhone” |
| `action.exec` | 执行级目标 | `mark_id=7` 或 `(x, y)` |
| `guard.max_steps` | 最大步数 | 10 到 30 |
| `guard.verify` | 执行后校验 | URL 变化、页面文本变化、元素消失 |

这里有一个工程上很重要但容易被忽略的点：**Action 最好分成“意图层”和“执行层”两个对象**。如果一开始就让大模型直接输出最终坐标，你很难诊断错误到底来自推理还是来自定位。

---

## 工程权衡与常见坑

多模态 ReAct 在论文里常被写得很顺，但真实工程里通常卡在下面几类问题。

| 常见问题 | 现象 | 根因 | 规避措施 |
|---|---|---|---|
| 描述模糊 | “点菜单”但页面有多个菜单 | 语言目标不唯一 | 要求模型输出位置约束，如“顶部右上角菜单” |
| 坐标漂移 | 点击后偏到旁边元素 | 分辨率变化、缩放变化、滚动后坐标失效 | 统一 viewport，点击前二次确认 bbox |
| 视觉遮挡 | 弹窗盖住原目标 | 仅依赖旧截图 | 每步重新截图，优先处理遮挡层 |
| 文本与图像不一致 | a11y tree 有元素但屏幕没显示 | DOM 存在但不可见 | 用截图可见性校验 |
| SoM 编号失效 | 标号和当前界面不同步 | 页面刷新后重排 | 每次动作后重建 SoM |
| 错误传播 | 第一步点错，后面越走越偏 | 历史状态污染 | 每步做结果验证，不通过就回滚或重规划 |

最关键的权衡有三个。

第一，**Observation 越全，不等于越好**。  
截图、树、SoM、OCR、历史轨迹全塞给模型，会提升信息量，但也会推高 token 成本和注意力分散风险。很多任务里，最有价值的是“结构树 + 当前截图 + 少量历史摘要”，而不是完整原始日志。

第二，**planning 和 grounding 要解耦**。  
SeeAct 的经验说明，大模型很擅长生成“下一步应该做什么”的自然语言动作，但不擅长稳定地把它转成精确可执行元素。把 grounding 交给专门模块，通常比让单个大模型端到端输出坐标更稳。

第三，**视觉 only 与结构增强是两种不同路线**。  
像 SeeAct-V、UGround 这类方向强调纯视觉观察和像素级操作，优势是统一桌面、移动端、网页三类界面；但如果你的场景稳定且能拿到高质量 DOM/a11y tree，那么“文本结构 + 视觉校验”的混合路线通常更省成本。

一个真实工程坑是“描述正确但执行失败”。例如模型说“点击菜单中的 Pricing”，推理本身可能没错，但：

- 当前页面有桌面端和移动端两个 Pricing；
- 其中一个在汉堡菜单里，需要先展开；
- 另一个虽然可见，但被 cookie banner 挡住。

如果没有执行后验证，系统只会把失败归因给“模型不聪明”，而实际上是 grounding 和状态同步出了问题。

---

## 替代方案与适用边界

不是所有任务都值得上多模态 ReAct。选型时先看环境，再看动作粒度。

| 场景 | 推荐策略 | 适用原因 | 限制 |
|---|---|---|---|
| 命令行、日志分析、API 编排 | 文本 only ReAct | 输入天然是结构化文本 | 没有视觉上下文 |
| 规则稳定的内部后台 | DOM 工具调用 + 轻量规划 | 元素可直接定位，效率高 | 迁移到别的站点差 |
| 公开网页、多站点任务 | 多模态 ReAct + SoM/grounding | 页面样式多变，视觉信息关键 | 成本高，执行链更复杂 |
| 桌面/移动端 GUI | 视觉 ReAct + 像素 grounding | 没有统一 DOM 接口 | 对坐标和分辨率敏感 |
| 高风险表单操作 | 多模态 ReAct + 强校验 + 人工确认 | 能感知界面状态 | 仍可能误点，不适合无监督执行 |

可以把几个替代方案概括为下面三类。

### 1. 文本 only ReAct

如果任务环境本来就是文本，例如 shell、SQL 控制台、日志检索、文档问答，那么传统 ReAct 足够好。这里“看图”不会带来实质收益，反而增加复杂度。

### 2. 工具调用式 Agent

如果系统能直接暴露稳定 API，比如 `click_element_by_id("nav-iphone")`、`submit_form(fields)`，那最优方案通常不是视觉代理，而是工具代理。因为你绕开了 grounding 这一高误差环节。

### 3. 纯视觉 Agent

如果 DOM 不可靠、界面跨平台、元素语义弱，例如桌面软件、远程桌面、移动端 App，那么纯视觉路线更合理。SeeAct-V 和 UGround 这类方法的价值就在这里：它们把“网页”“桌面”“手机”统一成“看屏幕并在像素上操作”的问题。

因此，多模态 ReAct 的适用边界可以用一句话概括：

**当任务成功与否依赖屏幕上的空间关系、可见状态和视觉定位时，才有必要把 ReAct 扩展到多模态。**

---

## 参考资料

- ReAct 官方项目：说明“推理 + 行动”循环的基础范式，以及 `obs -> thought -> action` 的核心形式。  
  https://react-lm.github.io/

- MM-ReAct 项目页：展示 ReAct 在多模态任务中的扩展思路，把视觉理解、外部工具和语言推理串到统一流程中。  
  https://multimodal-react.github.io/

- VisualWebArena 项目页：给出真实视觉网页任务基准，并展示文本 only、caption 增强、多模态、SoM 等不同输入设定下的成功率差异。  
  https://jykoh.com/vwa

- VisualWebArena ACL 2024 论文：提供更完整的实验表，包含 LLaMA-2-70B 1.10%、GPT-4 7.25%、GPT-4V + SoM 16.37% 等结果。  
  https://aclanthology.org/2024.acl-long.50.pdf

- SeeAct 项目页：提出“动作生成 + 动作 grounding”的两阶段网页代理框架，并明确指出 grounding 相对 oracle 仍有约 20-25 个百分点差距。  
  https://osu-nlp-group.github.io/SeeAct/

- UGround 项目页：给出面向 GUI 的通用视觉 grounding 模型，强调从自然语言动作描述到像素坐标的映射能力，并展示在 web/mobile/desktop 上的效果。  
  https://osu-nlp-group.github.io/UGround/

- UGround 模型仓库与结果表：补充 ScreenSpot 等数据，方便理解专门 grounding 模型为什么比通用大模型更适合做像素级定位。  
  https://github.com/OSU-NLP-Group/UGround
