## 核心结论

浏览器 Agent 的动作空间，可以理解为“策略模型被允许调用的操作集合”。它不是越大越好，关键是粒度是否合适。对多数网页任务，保留 `goto`、`click`、`fill`、`select_option`、`press`、`scroll`、`upload_file`、`go_back`、`hover`、`drag_and_drop` 这类约 10 到 20 个高阶原语，通常比直接暴露几十种坐标、鼠标和键盘细节更稳定。

原因很直接：模型的每一步都在做“从候选动作里选一个”的决策。候选越多，错误分叉越多，推理预算越容易浪费在低价值动作上。对于固定步数 $T$ 的任务，搜索空间近似随 $|\mathcal{A}|^T$ 增长，其中 $|\mathcal{A}|$ 是动作数量。动作空间从 50 个缩到 15 个，不只是“少了 35 个命令”，而是把多步任务里的组合爆炸一起压下去了。

这也是为什么很多真实浏览器 Agent 框架都优先采用高阶动作。BrowserGym 的动作集既支持基于元素 ID 的 `click(bid)`、`fill(bid, text)`、`select_option(bid, options)`，也支持坐标和底层键盘操作，但工程上更常把前者暴露给策略模型，把后者留在执行器内部。Go-Browse、WebArena 一类工作流也反复说明同一件事：模型更擅长理解“点哪个元素、填什么字段”，不擅长稳定地产生“鼠标从 $(x_1,y_1)$ 移到 $(x_2,y_2)$ 再按下”的微操作序列。

先把结论压缩成一张表：

| 层级 | 描述 | 典型原语 | 对模型的负担 | 典型失败点 |
|------|------|----------|--------------|------------|
| 高阶 | 直接表达网页语义意图 | `goto(url)` `click(elem)` `fill(elem,text)` `select_option(elem,option)` `upload_file(elem)` | 低，模型主要做目标匹配 | 选错元素、文本填错、时机判断错 |
| 低阶 | 直接操纵鼠标、坐标、按键 | `mouse_move(x,y)` `mouse_down()` `mouse_up()` `keyboard_press(key)` | 高，模型还要处理几何和时序细节 | 坐标漂移、焦点丢失、点击被遮挡、按键顺序错 |

玩具例子很容易说明差异。目标是“提交登录表单”。

| 设计方式 | 模型需要决定的内容 | 典型输出 |
|------|------|------|
| 高阶动作 | 哪个输入框、填什么值、何时提交 | `click(username)` → `fill(username,"alice")` → `fill(password,"***")` → `click(submit)` |
| 低阶动作 | 坐标、鼠标路径、焦点、键入顺序、提交时机 | `move(412,286)` → `down()` → `up()` → `type(a)` ... |

后者并没有增加业务能力，只是增加了失败点。任务成功率放到真实流水线里看，很多团队会观察到一个经验现象：动作原语精简到 15 到 20 个核心动作后，成功率常会出现一个可见抬升，量级常在几个百分点到约 8 个百分点之间。这个数字不是常数，但方向通常稳定，因为问题本质是决策接口被压缩了。

---

## 问题定义与边界

“浏览器 Agent 的动作空间设计”讨论的不是模型参数，而是接口设计。接口设计错了，模型再强，也会把推理预算耗在错误层级。

这里先划边界。本文只讨论四类动作：

| 类别 | 示例原语 | 语义说明 | 新手可直接理解的说法 |
|------|----------|----------|----------------------|
| 交互 | `click(elem)` `fill(elem,text)` `select_option(elem,option)` | 直接作用于页面元素 | 点按钮、填输入框、选下拉框 |
| 导航 | `goto(url)` `go_back()` `go_forward()` | 控制页面和历史 | 打开页面、返回上一页、前进 |
| 辅助操作 | `scroll(dx,dy)` `hover(elem)` `upload_file(elem)` | 支持视口、提示、文件上传 | 滚动页面、悬停查看提示、上传文件 |
| 控制流 | `noop()` `send_msg_to_user(text)` `report_infeasible(reason)` | 处理停顿、沟通和失败退出 | 暂停、向用户说明情况、报告做不到 |

这里的“高阶”不是指算法复杂，而是“把多个底层动作封装成一个带语义的命令”。例如 `fill(elem, text)`，白话解释就是“找到输入框并把文本填进去”，而不是要求模型自己完成点击、聚焦、全选、删除、键入、确认的一整串步骤。

相对地，“低阶”动作是连续控制或近连续控制，例如 `mouse_move(x,y)`、`keyboard_down(key)`。它们的优点是表达能力强，理论上几乎任何桌面动作都能拼出来；缺点是动作空间一旦变大，模型就会被迫承担执行细节，而不是只承担任务规划。

因此本文的边界结论是：

1. 默认面向网页任务，而不是任意桌面 GUI。
2. 默认优先离散、高阶、可验证的动作。
3. 坐标、像素和按键流尽量保留在执行层，不直接暴露给策略层。
4. 如果任务依赖绘图、复杂拖拽、游戏类微操，本文结论要打折。

新手最容易犯的错误，是把“能力覆盖率”理解成“动作数量越多越全”。实际上，很多动作完全可以由执行器完成编译。模型说“点击搜索框”，执行器再去解析元素、滚动到可见区域、计算中心点、执行点击。分层以后，模型负责意图，执行器负责精度。

再看一个常见误解：

| 误解 | 实际情况 |
|------|----------|
| “给更多动作，模型更自由，能力更强” | 更多动作通常先带来更多歧义和错误分叉 |
| “高阶动作太抽象，不够灵活” | 对绝大多数网页任务，语义动作已经足够覆盖 |
| “底层动作更通用，所以更好” | 通用不等于适合作为策略接口 |
| “动作空间只是工程细节” | 它直接决定策略搜索难度和故障恢复方式 |

---

## 核心机制与推导

动作空间为什么会拖垮浏览器 Agent，核心是组合爆炸。

设动作集合为 $\mathcal{A}$，任务最多允许 $T$ 步。最粗略地看，策略需要在每一步从 $|\mathcal{A}|$ 个候选里选一个，所以可能路径数近似为：

$$
N \approx |\mathcal{A}|^T
$$

这不是说系统真的会枚举整棵树，而是说错误分叉会按这个量级增加。动作一多，模型即使局部判断还行，整体成功率也会被多步串联放大地吞掉。

先看最简单的玩具推导。假设每一步只有一个动作是真正正确的，随机命中概率就是：

$$
P_{\text{hit}} = \frac{1}{|\mathcal{A}|}
$$

如果动作数是 50，那么单步命中率是：

$$
P_{\text{hit}}(50)=\frac{1}{50}=2\%
$$

如果缩到 15，单步命中率变成：

$$
P_{\text{hit}}(15)=\frac{1}{15}\approx 6.7\%
$$

看起来只是提升了约 4.7 个百分点，但如果任务要求连续 4 步都选对，概率分别是：

$$
P_{50}=\left(\frac{1}{50}\right)^4,\qquad P_{15}=\left(\frac{1}{15}\right)^4
$$

两者比值约为：

$$
\frac{P_{15}}{P_{50}} = \left(\frac{50}{15}\right)^4 \approx 123
$$

这说明多步任务里，“少一点动作”不是线性收益，而常常是连锁收益。

当然，真实模型不是随机选动作，因此上面的公式不能拿来直接预测线上成功率。但它抓住了本质：动作空间越大，模型越难把概率质量集中到真正有用的候选上。可以把它理解成一次多选题考试：

| 场景 | 题目没变 | 选项数量 | 结果 |
|------|----------|----------|------|
| 小动作空间 | “下一步做什么” | 15 个选项 | 模型更容易把注意力放到关键元素 |
| 大动作空间 | “下一步做什么” | 50 个选项 | 模型先消耗预算排除大量低价值选项 |

再看真实工程例子。假设客服 Agent 要在工单系统里查找资产：

1. 打开目标页面。
2. 点击筛选器。
3. 填写 `Asset tag`。
4. 回车确认。
5. 点开结果。

如果给模型 60 种动作，其中很多只是不同鼠标按键、滚轮方向、组合键，它就会花更多推理预算区分“能不能右键”“要不要先 `mouse_down` 再 `mouse_up`”“是否需要先 hover 再 click”。这些判断对业务价值很低。相反，如果只给它 15 个语义明确的动作，模型更可能把注意力放到“当前页面哪个元素与资产筛选最相关”。

这也是 BrowserGym 一类系统偏爱基于元素标识符 `bid` 的原因。`bid` 可以白话理解成“页面里可交互元素的编号”。模型输出的是“对哪个编号做什么”，执行器再去做定位和具体操作。于是“语义搜索”和“物理执行”被拆开，问题难度明显下降。

从工程角度看，动作空间裁剪的收益主要来自三个方向：

| 机制 | 发生了什么 | 直接收益 |
|------|------------|----------|
| 候选数下降 | 每步需要比较的动作更少 | 推理更稳定 |
| 语义更清晰 | 每个动作的副作用更可预期 | 错误更容易诊断 |
| 执行器接管细节 | 滚动、等待、聚焦、重试被封装 | 模型无需处理时序噪声 |

所以“动作设计”本质上不是列一个 API 清单，而是在决定：哪些复杂性让模型承担，哪些复杂性让系统承担。对网页任务，合理答案通常是让模型承担语义选择，让执行器承担物理细节。

---

## 代码实现

实现上，最稳的做法不是把所有浏览器 API 都直接给模型，而是做一层动作编译器：模型只输出有限高阶命令，系统把高阶命令翻译成真正的浏览器调用。

下面是一个可直接运行的 Python 示例。它不依赖真实浏览器，但把三个关键点写清楚了：

1. 高阶动作集合如何定义。
2. 模型动作如何被编译到执行器。
3. 动作裁剪后，多步成功概率如何变化。

```python
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Action:
    name: str
    target: Optional[str] = None
    value: Optional[str] = None


@dataclass(frozen=True)
class ActionSpace:
    name: str
    actions: Tuple[str, ...]

    def contains(self, action_name: str) -> bool:
        return action_name in self.actions

    def size(self) -> int:
        return len(self.actions)


class MockPage:
    """
    一个极简网页状态机：
    - goto("/login") 打开登录页
    - click("username"), fill("username", ...)
    - click("password"), fill("password", ...)
    - click("submit") 提交
    满足用户名和密码后，页面状态变为 success
    """

    def __init__(self) -> None:
        self.url = "about:blank"
        self.focused: Optional[str] = None
        self.fields: Dict[str, str] = {"username": "", "password": ""}
        self.logged_in = False
        self.history: List[str] = [self.url]

    def goto(self, url: str) -> str:
        self.url = url
        self.history.append(url)
        self.focused = None
        return f"navigated to {url}"

    def go_back(self) -> str:
        if len(self.history) <= 1:
            return "no history to go back"
        self.history.pop()
        self.url = self.history[-1]
        self.focused = None
        return f"went back to {self.url}"

    def click(self, target: str) -> str:
        if target in self.fields or target == "submit":
            self.focused = target
            return f"clicked {target}"
        return f"unknown target: {target}"

    def fill(self, target: str, value: str) -> str:
        if target not in self.fields:
            return f"cannot fill {target}"
        self.fields[target] = value
        self.focused = target
        return f"filled {target}"

    def press(self, key: str) -> str:
        if key != "Enter":
            return f"ignored key {key}"
        if self.url == "/login" and self.fields["username"] and self.fields["password"]:
            self.logged_in = True
            return "login success"
        return "login failed"

    def status(self) -> str:
        return (
            f"url={self.url}, focused={self.focused}, "
            f"username={self.fields['username']!r}, "
            f"password={'***' if self.fields['password'] else ''}, "
            f"logged_in={self.logged_in}"
        )


class ActionCompiler:
    """
    执行器只接受有限高阶动作。
    底层细节，例如滚动到可见区、等待元素可点击、重试点击，在真实系统中应放在这里。
    """

    def __init__(self, action_space: ActionSpace, page: MockPage) -> None:
        self.action_space = action_space
        self.page = page

    def execute(self, action: Action) -> str:
        if not self.action_space.contains(action.name):
            raise ValueError(f"action {action.name!r} is not allowed in {self.action_space.name}")

        if action.name == "goto":
            assert action.value is not None
            return self.page.goto(action.value)

        if action.name == "go_back":
            return self.page.go_back()

        if action.name == "click":
            assert action.target is not None
            return self.page.click(action.target)

        if action.name == "fill":
            assert action.target is not None and action.value is not None
            return self.page.fill(action.target, action.value)

        if action.name == "press":
            assert action.value is not None
            return self.page.press(action.value)

        if action.name in {"scroll", "hover", "select_option", "upload_file", "drag_and_drop", "noop"}:
            return f"simulated {action.name}"

        if action.name in {"send_msg_to_user", "report_infeasible"}:
            return f"{action.name}: {action.value}"

        raise NotImplementedError(f"handler for {action.name!r} is missing")


def step_hit_prob(action_space_size: int) -> float:
    if action_space_size <= 0:
        raise ValueError("action_space_size must be > 0")
    return 1.0 / action_space_size


def trajectory_hit_prob(action_space_size: int, steps: int) -> float:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return prod(step_hit_prob(action_space_size) for _ in range(steps))


def main() -> None:
    high_level = ActionSpace(
        name="high_level_core",
        actions=(
            "goto",
            "click",
            "fill",
            "select_option",
            "press",
            "scroll",
            "upload_file",
            "go_back",
            "hover",
            "drag_and_drop",
            "noop",
            "send_msg_to_user",
            "report_infeasible",
        ),
    )

    bloated = ActionSpace(
        name="bloated_low_level",
        actions=tuple(f"action_{i}" for i in range(50)),
    )

    assert high_level.contains("click")
    assert not high_level.contains("mouse_move")

    page = MockPage()
    executor = ActionCompiler(high_level, page)

    plan = [
        Action(name="goto", value="/login"),
        Action(name="click", target="username"),
        Action(name="fill", target="username", value="alice"),
        Action(name="click", target="password"),
        Action(name="fill", target="password", value="secret"),
        Action(name="press", value="Enter"),
    ]

    print("== Execute plan ==")
    for step_id, action in enumerate(plan, start=1):
        result = executor.execute(action)
        print(f"{step_id}. {action} -> {result}")
        print("   ", page.status())

    assert page.logged_in is True

    print("\n== Probability comparison ==")
    p15 = step_hit_prob(15)
    p50 = step_hit_prob(50)
    p15_4 = trajectory_hit_prob(15, 4)
    p50_4 = trajectory_hit_prob(50, 4)

    print(f"single-step hit with 15 actions: {p15:.4f}")
    print(f"single-step hit with 50 actions: {p50:.4f}")
    print(f"4-step hit with 15 actions:     {p15_4:.8f}")
    print(f"4-step hit with 50 actions:     {p50_4:.8f}")
    print(f"ratio (15 vs 50):               {p15_4 / p50_4:.2f}x")

    assert round(p15, 4) == 0.0667
    assert round(p50, 4) == 0.0200
    assert p15_4 > p50_4
    assert high_level.size() == 13
    assert bloated.size() == 50


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出会分成两部分：

| 输出部分 | 含义 |
|------|------|
| `Execute plan` | 演示高阶动作如何驱动一个简化登录流程 |
| `Probability comparison` | 演示动作空间缩小后，多步命中概率如何变化 |

如果要接到真实浏览器里，常见结构通常是四层：

1. 观察层提取可交互元素，生成元素 ID。
2. 策略层只输出 `click("a46")`、`fill("a2164","Asset tag")` 这类命令。
3. 执行层负责等待、滚动到可见区域、重试、异常处理。
4. 校验层判断页面是否变化，失败时触发回退或重试。

真实工程中常见的动作序列大致长这样：

```python
actions = [
    "goto('/assets')",
    "click('filter_button')",
    "fill('asset_tag_input', 'AT-1024')",
    "press('Enter')",
    "click('result_row_0')",
]
```

这种写法的价值，不在于字符串本身，而在于它把“决策接口”控制在少量稳定原语里。模型负责选动作，系统负责把动作执行正确。

如果进一步工程化，执行器通常还会补上这些逻辑：

| 执行器能力 | 为什么需要 |
|------|------------|
| 自动等待元素可见 | 避免页面未渲染完就点击 |
| 滚动到可见区域 | 避免元素存在但不在视口内 |
| 点击失败重试 | 处理遮挡、动画、布局抖动 |
| DOM/URL/截图校验 | 判断动作是否真的生效 |
| 失败回退 | 避免陷入死循环 |

这也是“高阶动作 + 强执行器”组合通常优于“低阶动作 + 弱执行器”的原因。

---

## 工程权衡与常见坑

动作空间设计不是只看成功率，还要看维护成本、可解释性和故障恢复。

| 常见坑 | 后果 | 规避策略 |
|------|------|----------|
| 原语过多，超过 50 个 | 决策分叉大，模型像在“猜工具” | 先压到 15 到 20 个核心动作 |
| 直接暴露坐标 | 页面一变就漂移，模型重复点错 | 用元素 ID 或语义引用，坐标留在执行器 |
| 点击后不等待 | 页面未加载完成就继续下一步 | 每步后做 DOM/URL/截图变化检查 |
| 没有失败终止条件 | 陷入 `click-loop` 或 `scroll-loop` | 加断路器，重复动作超阈值后回退 |
| 一个动作语义太宽 | `type` 既可能输入又可能提交，副作用不清晰 | 把 `fill` 与 `press` 分开定义 |
| 原语命名不稳定 | 同类动作在不同项目里含义漂移 | 保持动作语义单一、参数结构统一 |

“断路器”这个术语，白话解释就是“当系统发现自己在重复失败时，强制停下来换策略”。例如某个 Agent 连续两次执行同一 `click`，而 URL、DOM 结构、截图哈希都没有明显变化，这通常意味着点击无效、元素被遮挡或判断错了。此时继续点第三次，收益接近零，应该回退、刷新观察，或者直接报告当前动作不可行。

一个常见的断路器条件可以写成：

$$
\text{trip} =
\mathbf{1}\left[
(\text{same\_action\_count} \ge k)
\land
(\Delta \text{URL} = 0)
\land
(\Delta \text{DOM} < \epsilon)
\right]
$$

其中：

| 符号 | 含义 |
|------|------|
| $\text{same\_action\_count}$ | 同一动作连续重复次数 |
| $k$ | 触发阈值，例如 2 或 3 |
| $\Delta \text{URL}$ | 动作前后 URL 是否变化 |
| $\Delta \text{DOM}$ | DOM 变化幅度，可用节点差异或哈希差异近似 |
| $\epsilon$ | 很小的阈值，表示“几乎没变化” |

工程上另一个常见误区，是把高阶动作做得过度复杂。例如定义 `complete_checkout_and_pay()` 这种超大动作。它虽然“更高阶”，但副作用太多、可复用性差，也不利于失败定位。合理的高阶原语通常满足三个条件：

1. 语义单一。
2. 结果可验证。
3. 能跨站点复用。

所以 `fill`、`click`、`select_option` 通常是好原语；“完成整个注册流程”通常不是。

可以用下面这张表判断一个动作是否设计过头：

| 动作 | 是否推荐 | 原因 |
|------|----------|------|
| `click(elem)` | 推荐 | 副作用明确，结果容易验证 |
| `fill(elem,text)` | 推荐 | 参数简单，站点间可复用 |
| `select_option(elem,option)` | 推荐 | 明确对应下拉框语义 |
| `submit_login_form_with_retry()` | 不推荐 | 把规划、执行、重试混在一起 |
| `do_everything_needed()` | 不推荐 | 无法测试，也无法定位失败 |

对新手来说，最实用的一条经验是：如果一个动作的名字里已经包含了完整业务流程，它大概率不是好原语；如果一个动作只表达一个网页意图，它通常更健康。

---

## 替代方案与适用边界

如果任务本身就需要大动作空间，不能只靠“删动作”解决。这时可以用替代方案，但代价要说清楚。

| 方案 | 优点 | 代价/适用边界 |
|------|------|--------------|
| Beam Search / 候选保留 | 同时保留多条动作路径，降低一次选错就失败 | 推理更贵，适合短任务 |
| Action Re-ranking / 动作重排 | 先粗筛，再精排，减少一次性全量选择 | 需要额外打分模型 |
| World Model + Action Selection | 先模拟候选动作后果，再决定执行哪个 | 训练复杂，数据需求高 |
| 高阶动作 + 执行器回退 | 实现简单，最适合通用网页任务 | 对极细粒度操作覆盖有限 |

这些方案解决的是“动作已经不少了，如何降低选择难度”，不是“为什么一开始要把动作设计得这么多”。

可以把它们理解成三种不同层级的补救：

| 层级 | 方法 | 本质 |
|------|------|------|
| 接口层 | 先裁剪动作空间 | 减少问题规模 |
| 推理层 | 重排候选、保留多条路径 | 提高选择质量 |
| 训练层 | 世界模型、价值模型 | 提前估计动作后果 |

ICLR 2025 的 WMA 一类工作说明了一条现实路线：当动作候选很多时，不一定直接让策略模型从所有动作里硬选，而是先构造少量候选，再做 action selection，也就是“动作选择器”。白话解释就是“先圈出几个像样答案，再从里面挑最优”。在 WebArena 类任务中，这种方式通常能在较短动作预算下提升成功率，同时比树搜索更节省时间和 API 成本。

但这些替代方案并不推翻本文主结论。它们解决的是“动作已经不少了怎么办”，不是“为什么一开始就把动作做成 80 个”。默认策略仍然应该是：先把动作空间压到核心集合，再决定是否需要候选重排或世界模型。

适用边界也要说明。下面这些场景里，低阶动作会变得更重要：

1. 远程桌面、Citrix、游戏界面，没有稳定 DOM 或可访问性树。
2. 复杂拖拽、画布编辑、地图缩放，元素语义难以离散化。
3. 视觉元素频繁变化，元素 ID 很难稳定抽取。

这些场景里，系统可能必须回到坐标、像素和轨迹层。但即使如此，也通常会先做一层离散化，例如 SoM、候选框编号、视觉 grounding，把连续控制改写成“点第几个候选框”，而不是直接让模型输出原始像素轨迹。

最后给一个实用判断标准：

| 任务类型 | 默认动作设计 |
|------|--------------|
| 标准网页表单、检索、后台操作 | 高阶语义动作优先 |
| 有少量复杂控件的网页 | 高阶动作为主，少量补充拖拽/悬停 |
| 纯视觉 GUI、远程桌面 | 半离散化动作或低阶动作 |
| 游戏、绘图、实时交互 | 低阶动作不可避免，但需要额外约束 |

因此，本文结论不是“低阶动作永远没用”，而是“在标准浏览器任务里，低阶动作不应成为默认策略接口”。

---

## 参考资料

- BrowserGym Action Space 文档：<https://browsergym.readthedocs.io/latest/core/action_space.html>
- The BrowserGym Ecosystem for Web Agent Research：<https://openreview.net/pdf?id=5298fKGmv3>
- WebArena: A Realistic Web Environment for Building Autonomous Agents：<https://arxiv.org/abs/2307.13854>
- Arun Baby, UI Automation Agents：<https://arunbaby.com/ai-agents/0023-ui-automation-agents/>
- Dulac-Arnold et al., Reinforcement Learning in Large Discrete Action Spaces：<https://www.researchgate.net/publication/288059770_Reinforcement_Learning_in_Large_Discrete_Action_Spaces>
- Go-Browse: Training Web Agents with Structured Exploration：<https://arxiv.org/pdf/2506.03533>
- ICLR 2025 WMA paper：<https://proceedings.iclr.cc/paper_files/paper/2025/file/a00548031e4647b13042c97c922fadf1-Paper-Conference.pdf>
