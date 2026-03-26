## 核心结论

Computer Use 是一种让多模态大模型直接“看屏幕并动手”的 Agent 范式。这里的多模态大模型，指既能处理文字也能处理图像的模型；在这个场景里，它读入用户指令和当前截图，再输出点击、输入、滚动等 GUI 动作。GUI 是图形界面，也就是按钮、输入框、菜单、窗口这些人平时直接操作的界面。

它的关键价值不在于“会点鼠标”本身，而在于闭合了感知到行动的回路：模型不再只会解释界面，而是能在未改造的现有软件中完成任务。对工程系统来说，这意味着它可以接入桌面应用、老旧内网系统、远程虚拟机、没有公开 API 的后台页面，而不必先把这些系统重写成适合 LLM 调用的工具接口。

但它现在还远不到“稳定替代人工”的阶段。OSWorld 论文给出的基线里，人类成功率约为 72.36%，当时最优模型只有 12.24%。这说明 GUI Agent 已经可用来做演示、测试、半自动执行与低风险任务，却还不能被当成无监督的高可靠执行器。

一个最直观的闭环例子是“保存网页为 PDF”：

| 步骤 | 状态 | Agent 行动 | 反馈 |
| --- | --- | --- | --- |
| 1 | 接收任务 | 读取指令“将当前网页保存为 PDF” | 需要先看当前页面 |
| 2 | 感知界面 | 请求截图，识别浏览器工具栏与菜单位置 | 找到“打印”或菜单入口 |
| 3 | 执行动作 | 点击菜单，选择“打印” | 新截图出现打印弹窗 |
| 4 | 继续执行 | 点击“另存为 PDF”或“Save as PDF” | 新截图出现保存位置选择 |
| 5 | 验证结果 | 输入文件名，点击保存 | 新截图显示已保存或下载完成 |

简化流程图可以写成：

```text
Plan -> Action -> Critic
指令+截图 -> 输出动作 -> 执行后再截图检查
```

---

## 问题定义与边界

严格定义上，Computer Use 是：模型根据自然语言指令 $X_{\rm instr}$、当前屏幕截图 $I^t$ 和历史交互信息，预测下一步 GUI 动作。动作通常包含动作类型与参数，比如点击坐标、输入文本、滚动方向与距离。

可以把它写成：

$$
a^t = f(X_{\rm instr}, I^t, H^{t-1})
$$

其中：

- $a^t$ 是第 $t$ 步动作
- $X_{\rm instr}$ 是用户任务描述
- $I^t$ 是当前截图
- $H^{t-1}$ 是历史，包含过去动作和过去截图

这里的 history，可以白话理解为“模型的工作记事本”。如果没有它，模型很容易忘记自己刚才点过哪里、输入过什么、现在做到第几步。

玩具例子可以用“关闭浏览器当前标签页”理解：

1. 用户说“帮我关闭当前标签页”
2. 系统截屏
3. 模型在截图右上角寻找标签页的 `X`
4. 输出 `click(x, y)`
5. 执行后重新截屏
6. 如果标签页消失，任务完成；如果没消失，说明坐标可能偏了，要重试

它的边界也很明确：

| 适用边界 | 说明 |
| --- | --- |
| 可视界面 | 必须能看到按钮、菜单、输入框等视觉元素 |
| 连续截图 | 依赖一步一步截图反馈，不是直接读取程序内部状态 |
| 弱结构化环境 | 适合没 API、没 DOM、没 accessibility tree 的系统 |
| 不适合纯后端任务 | 如果任务只有接口调用，没有屏幕可看，Computer Use 反而多此一举 |
| 不适合高精度高风险操作 | 如财务转账、生产删库、医疗录入等，必须有人类复核 |

因此它不是“API 自动化的升级版”，而是“无接口时的人类式操作替代方案”。只要系统可以被人眼看见并被鼠标键盘操作，它理论上就可以尝试；但如果系统本身已经暴露稳定 API，那么直接用 API 往往更便宜、更快、更稳。

---

## 核心机制与推导

大多数 GUI Agent 都可以抽象成 ReAct 风格循环。ReAct 的白话解释是“边想边做再检查”，不是先把整条长计划一次性写死。

一次循环可以写成：

$$
\text{Plan}_t: (X_{\rm instr}, I^t, H^{t-1}) \to \hat{a}^t
$$

$$
\text{Action}_t: \hat{a}^t \to e^t
$$

$$
\text{Critic}_t: (I^{t+1}, e^t, H^{t-1}) \to r^t,\ H^t
$$

其中：

- Planner 是规划器，负责决定“下一步做什么”
- Actor 是执行器，负责把动作落到系统里
- Critic 是检查器，负责看执行后截图是否符合预期
- $r^t$ 是反馈，可以理解成“这一步成功、失败还是需要重试”

用“保存网页为 PDF”这个任务展开，一次真实推导类似这样：

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| Plan | 指令 + 当前浏览器截图 + 历史 | `key("ctrl+p")` 或 `click(x,y)` |
| Action | 动作指令 | 浏览器弹出打印窗口 |
| Critic | 新截图 + 已执行动作 | 判断是否出现打印窗口；若未出现则重规划 |

这类系统为什么强调 GUI Grounding？Grounding 可以白话理解为“把语言和屏幕上的具体位置对上号”。模型不仅要知道“下载按钮”是什么意思，还要知道它在这张图的哪里。ShowUI、UI-TARS 这类开源工作，重点都在提升这种从截图到像素动作的对齐能力。

一个常见误区是把 GUI Agent 理解成“纯视觉 OCR + 鼠标宏”。实际上它至少包含三层能力：

| 层次 | 作用 | 失败表现 |
| --- | --- | --- |
| 感知 | 识别按钮、文本、状态 | 看错按钮、漏掉弹窗 |
| 推理 | 决定下一步动作 | 点错顺序、忘记前提 |
| 控制 | 生成准确动作参数 | 坐标偏移、滚动过量 |

真实工程例子比玩具例子复杂得多。以论文《The Dawn of GUI Agent》中的 Claude 3.5 Computer Use 框架为代表，系统会在浏览器、文件管理器等环境里连续执行多步任务。这里难点不是单步点击，而是长链路：打开网站、定位菜单、下载文件、切换窗口、确认保存、处理弹窗。只要任一步 grounding 偏移，后面整条轨迹就会漂移。

---

## 代码实现

工程实现通常不是“让模型直接接管电脑”，而是搭一个外层 agent loop。Anthropic 官方文档也明确说明：模型不会自己执行点击和截图，执行器必须由你的应用实现。

下面是一个可运行的最小 Python 示例。它不是直接调用真实模型，而是把 Computer Use 的闭环抽象出来，便于理解 Planner、Executor、Critic 各自做什么。

```python
from dataclasses import dataclass

@dataclass
class Action:
    kind: str
    payload: dict

def planner(instruction: str, screenshot: dict, history: list[Action]) -> Action:
    # 玩具规则：如果用户要关闭标签页，且截图显示 close_visible=True，就点击关闭按钮
    if instruction == "close current tab" and screenshot.get("close_visible"):
        return Action("click", {"target": "tab_close_button", "x": 980, "y": 18})
    # 如果按钮不可见，就先滚动或报错；这里简化为 no_op
    return Action("no_op", {})

def executor(screenshot: dict, action: Action) -> dict:
    # 执行动作后返回“新截图”
    new_screen = dict(screenshot)
    if action.kind == "click" and action.payload.get("target") == "tab_close_button":
        new_screen["tab_closed"] = True
        new_screen["close_visible"] = False
    return new_screen

def critic(before: dict, after: dict, action: Action) -> bool:
    # 检查动作是否真的改变了界面
    if action.kind == "click" and action.payload.get("target") == "tab_close_button":
        return before.get("close_visible") and after.get("tab_closed")
    return False

instruction = "close current tab"
screen_t = {"close_visible": True, "tab_closed": False}
history: list[Action] = []

action = planner(instruction, screen_t, history)
history.append(action)
screen_t1 = executor(screen_t, action)
ok = critic(screen_t, screen_t1, action)

assert action.kind == "click"
assert ok is True
assert screen_t1["tab_closed"] is True
```

如果换成真实 API 形态，通常是下面这种结构：

```python
instruction = "open downloads"
screenshot = capture()
action = claude.plan(instruction, screenshot, history)
execute(action)
feedback = capture()
result = claude.critic(feedback, history)
```

每一步的角色很明确：

| 动作类型 | 输入 | 输出 |
| --- | --- | --- |
| `screenshot` | 显示器编号、分辨率 | 当前屏幕图像 |
| `left_click` | 坐标 `(x, y)` | 点击执行结果 |
| `type` | 文本字符串 | 输入后的新界面 |
| `key` | 快捷键，如 `ctrl+s` | 系统快捷动作结果 |
| `scroll` | 方向与距离 | 新的可见区域 |

真实工程里还要处理坐标缩放。因为截图常被缩小后再送入模型，模型返回的是缩放空间里的坐标，而执行器操作的是原始屏幕坐标。如果不做比例还原，就会出现“看起来点对了，实际总是点偏”的现象。

---

## 工程权衡与常见坑

第一个核心权衡是 grounding 精度和泛化能力。你想让模型适配任意 GUI，就不能只靠某个网站的 DOM 规则；但一旦退回纯截图，像素定位误差又会立刻放大。

第二个权衡是灵活性和确定性。自由指令越强，模型越能处理陌生界面；但动作空间越开放，测试和回放就越难做。

常见坑如下：

| 坑点 | 现象 | 缓解方式 |
| --- | --- | --- |
| 坐标误差 | 点到按钮旁边而不是按钮上 | 做截图缩放映射，加入点击后验证 |
| 弹窗漏检 | 模型继续执行旧计划 | 每步执行后强制重截图，不盲目连续点 |
| history 缺失 | 重复点击、重复输入 | 显式保存动作轨迹与阶段状态 |
| 长任务漂移 | 前面一步错，后面全错 | 每个里程碑设置局部成功条件 |
| 滚动不稳定 | 滚过头或没滚到位 | 小步滚动，边滚边验证 |
| 提示注入 | 网页内容诱导模型偏离任务 | 放到隔离容器，只给低权限账号 |
| 延迟高 | 一步一截图导致慢 | 限定任务类型，用于后台自动化而非实时交互 |

一个很典型的新手场景是“把文件 A 移动到文件夹 B”。如果没有 history，模型可能发生三类错误：

1. 已经选中过 A，但下一轮忘了，又重新选一次
2. 已经打开了目标文件夹，但下一轮仍在找源目录
3. 拖拽失败后没有回到稳定状态，继续在错误界面上规划

所以 GUI Agent 的 memory 不是可选优化，而是多步任务的基本组件。UI-TARS 一类工作强调 unified action modeling、reflection thinking 和 reflective online traces，本质上都在解决这件事：让模型不仅会“当下点击”，还会从连续轨迹里纠错。

---

## 替代方案与适用边界

Computer Use 最容易被拿来和 DOM-based Agent 比较。DOM 是网页内部结构树，可以理解为浏览器里“按钮、输入框、文本节点”的程序化表示。Selenium、Playwright 这类自动化工具优先操作 DOM，所以稳定性通常远高于像素点击。

两者对比如下：

| 维度 | Computer Use | DOM/API 自动化 |
| --- | --- | --- |
| 输入 | 截图 + 指令 | 页面结构/API 文档 |
| 动作 | 鼠标键盘级 | 结构化调用 |
| 泛化对象 | 任意可视 GUI，包括桌面应用 | 主要是可编程网页或公开接口 |
| 稳定性 | 较低，受布局变化影响大 | 较高，元素选择器更稳定 |
| 开发成本 | 早期接入快，后期调试难 | 初期要接系统接口，但长期更稳 |
| 可解释性 | 依赖轨迹回放 | 调用链更清晰 |
| 适用任务 | 封闭系统、遗留系统、跨应用操作 | 规则明确、接口稳定、批量执行 |

可以用两个具体任务对比：

- “自动点击某电商后台已知按钮并导出报表”：优先 Selenium/Playwright，因为页面结构明确，脚本稳定，成本更低。
- “控制任意桌面应用，把截图工具保存的图片拖进聊天软件再发送”：更接近 Computer Use，因为跨多个应用、没有统一 DOM，也不一定有 API。

因此选择标准不是“哪种更先进”，而是“系统是否已经暴露可靠结构”。如果能走 API，就先走 API；如果只能像人一样看屏幕操作，才用 Computer Use。很多成熟系统最后会采用混合架构：能调用 API 的部分走 API，必须落到界面的部分才交给 GUI Agent。

---

## 参考资料

| 资料 | 用途 |
| --- | --- |
| [Anthropic Computer Use Tool Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) | 官方定义、动作类型、agent loop、实现约束 |
| [Developing a computer use model](https://www.anthropic.com/research/developing-computer-use) | Anthropic 对训练思路、安全边界和能力限制的说明 |
| [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://os-world.github.io/) | 人类与模型成功率基线、评测场景 |
| [The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use](https://arxiv.org/abs/2411.10323) | Claude 3.5 Computer Use 的案例分析与 out-of-the-box 框架 |
| [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326) | 原生 GUI Agent、统一动作建模、反思式在线轨迹 |
| [ShowUI: One Vision-Language-Action Model for GUI Visual Agent](https://arxiv.org/abs/2411.17465) | GUI grounding、视觉 token 选择、开源轻量模型 |
