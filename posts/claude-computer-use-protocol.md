## 核心结论

Claude Computer Use 的屏幕交互协议，本质上不是“让模型直接控制电脑”，而是把 GUI 自动化拆成三个受控工具：`computer`、`text_editor`、`bash`。`computer` 负责屏幕级操作，可以理解成“先看截图，再按坐标点击、输入、滚动”；`text_editor` 负责直接查看和修改文本文件；`bash` 负责执行终端命令。三者组合后，系统形成一个稳定闭环：截图、理解、定位、执行、验证。

这个协议最关键的设计，不在“能不能点”，而在“坐标是不是稳定”。Claude 看到的往往不是宿主机原始屏幕，而是满足尺寸约束后的缩放图，所以它返回的坐标 $(x', y')$ 通常属于缩放后的截图，而不是宿主机真实屏幕。宿主侧必须按缩放因子 $s$ 做反向映射，才能把模型给出的点转换成真实点击位置。

工程上有两个硬约束必须优先处理。第一，分辨率不要贪高，官方文档持续强调要控制屏幕尺寸，并明确给出最长边和总像素限制；在实践里，`1280×800` 或 `1024×768` 往往更稳。第二，不要连续抓太多截图，因为截图会直接进入多模态输入，成本几乎随截图次数线性增长。换句话说，Computer Use 能解决“必须看界面才能做”的问题，但它既不无限便宜，也不无限精确。

一个最小例子是：模型先请求截图，在界面中找到“Deploy”按钮，返回 `[x, y]`；宿主程序完成点击；然后让 `text_editor` 往日志文件写入“已触发部署”；最后用 `bash` 执行 `./deploy.sh`。这说明它不是单一点击协议，而是 GUI、文件、命令三类执行面统一编排的协议。

---

## 问题定义与边界

问题定义可以写得很直接：目标不是训练一个“会用电脑的人”，而是在受限的虚拟显示器或容器桌面里，让模型通过标准化工具完成 GUI 操作，并把结果继续传递给文件编辑和命令执行链路。

这里的“协议”指的是一组固定交互规则：

1. 宿主提供当前屏幕状态，通常是截图。
2. 模型基于截图和上下文，决定调用哪个工具。
3. 如果调用 `computer`，模型返回动作类型和坐标或文本。
4. 宿主执行动作，再把执行结果和新状态返回。
5. 循环继续，直到任务结束。

它不是远程桌面协议，也不是浏览器脚本接口，而是面向“屏幕像素 + 动作执行”的通用控制层。它的优势是通用，代价是精度和成本都受屏幕表示方式影响。

它的边界主要有三类。

| 限制项 | 说明 | 工程意义 |
|---|---|---|
| 最大边长 `1568` | 输入图像最长边不能无限增大 | 防止截图过大，降低坐标映射复杂度 |
| 总像素约 `1.15M` | 图像总像素受上限约束 | 控制图像成本和视觉理解稳定性 |
| 建议较低分辨率运行 | 官方示例常用 `1024×768`，实践常收敛到 `1280×800` 附近 | 在清晰度、定位精度、成本之间折中 |
| 必须受限运行环境 | 官方文档明确建议运行在 VM/容器/沙箱内 | 避免误操作真实宿主系统 |
| GUI 只是执行面之一 | 文档明确支持与 `bash`、文本编辑工具联动 | 不要把所有任务都塞给屏幕点击 |

缩放约束通常可以写成：

$$
s = \min\left(1.0,\ \frac{1568}{\max(W,H)},\ \sqrt{\frac{1.15\times10^6}{W\cdot H}}\right)
$$

其中，$W,H$ 是原始屏幕宽高，$s$ 是缩放因子。这个公式的意思是：如果屏幕过大，就同时检查“最长边限制”和“总像素限制”，取更严格的那个；如果屏幕本来就够小，则不放大，直接取 `1.0`。

这个公式可以拆开理解：

| 约束 | 数学形式 | 直观含义 |
|---|---|---|
| 最长边限制 | $\frac{1568}{\max(W,H)}$ | 任何一边都不能大到超过上限 |
| 总像素限制 | $\sqrt{\frac{1.15\times10^6}{W\cdot H}}$ | 整张图像不能大到超出预算 |
| 不放大原则 | `1.0` | 小图不因为协议而被人为放大 |

为什么不能直接给模型一个 4K 屏幕？因为 4K 会同时带来三个问题。

| 问题 | 结果 |
|---|---|
| 元素在缩放图里变小 | 按钮、复选框、标签更难识别 |
| 反向映射误差被放大 | 截图里偏 2 像素，真实屏幕可能偏更多 |
| 图片成本上升 | 每一步观察都更贵 |

对新手来说，一个简单判断标准是：如果同一个按钮在 `1024×768` 或 `1280×800` 已经能清楚识别，就没有必要为了“更清晰”把分辨率继续拉高。Computer Use 不是做视觉渲染，而是做任务执行。能稳定执行，比名义上的高清更重要。

所以，这个协议适用的不是“任意桌面自动化”，而是“在受控分辨率、受控权限、可重复截图的环境中做有边界的图形交互”。

---

## 核心机制与推导

整个机制可以压缩成一句话：Claude 看的是缩放后的图，你点的是原始屏幕，所以中间必须有一层坐标反算。

完整流程如下：

`抓图 → 必要时缩放 → Claude 输出动作 → 坐标反算 → 执行动作 → 重新抓图 → 重复`

这里最重要的是区分两个坐标系。

| 坐标系 | 含义 | 谁在使用 |
|---|---|---|
| 原始坐标 `(x, y)` | 宿主机真实屏幕像素位置 | 自动化执行器 |
| 缩放坐标 `(x', y')` | 模型基于缩放截图理解后的像素位置 | Claude |

两者关系是：

$$
x' = s\cdot x,\quad y' = s\cdot y
$$

宿主真正执行点击时，应反推：

$$
x = \frac{x'}{s},\quad y = \frac{y'}{s}
$$

如果需要整数像素点，通常还要再做取整：

$$
x_{\text{real}} = \operatorname{round}\left(\frac{x'}{s}\right),\quad
y_{\text{real}} = \operatorname{round}\left(\frac{y'}{s}\right)
$$

再做一次边界裁剪，避免点击越界：

$$
x_{\text{safe}} = \min(\max(x_{\text{real}}, 0), W-1)
$$

$$
y_{\text{safe}} = \min(\max(y_{\text{real}}, 0), H-1)
$$

看一个完整数值例子。假设原始屏幕是 `1920×1080`。

先算最长边约束：

$$
\frac{1568}{1920}\approx 0.8167
$$

再算总像素约束：

$$
\sqrt{\frac{1.15\times10^6}{1920\times1080}}
= \sqrt{\frac{1.15\times10^6}{2073600}}
\approx \sqrt{0.5546}
\approx 0.7447
$$

因此：

$$
s = \min(1.0, 0.8167, 0.7447)=0.7447
$$

这说明真正生效的是总像素限制，而不是最长边限制。

如果 Claude 在缩放图上返回 `[400, 400]`，那么真实点击点应为：

$$
x = 400/0.7447 \approx 537,\quad y = 400/0.7447 \approx 537
$$

如果返回的是截图中心点，推导也一样。假设缩放后图片尺寸为：

$$
W' = W\cdot s,\quad H' = H\cdot s
$$

当 $W=1920,H=1080,s=0.7447$ 时，大约有：

$$
W' \approx 1430,\quad H' \approx 804
$$

如果模型点了缩放图中心附近的 `(715, 402)`，反算后仍接近原图中心 `(960, 540)`。这说明反向映射不是“补偿误差”，而是协议本身的一部分。没有这一步，点击就天然偏移。

为什么每次动作后都要重新截图？因为 GUI 是有状态的。一个按钮被点击后，界面可能出现以下变化：

| 动作后变化 | 不重新截图会怎样 |
|---|---|
| 弹窗出现 | 模型还以为原按钮仍可点击 |
| 页面跳转 | 坐标对应区域已经变成别的控件 |
| 按钮禁用 | 模型会重复执行无效点击 |
| 列表刷新 | 旧目标元素的相对位置已变化 |

所以 Computer Use 与传统“录制坐标脚本”的根本差别在于：它不是一次性录好坐标后盲点，而是每一步都重新观察、重新决策。

再看一个工程例子。假设内部发布平台必须人工点击“Deploy”才能触发灰度发布，完整闭环可以这样设计：

1. `computer` 请求控制台截图。
2. Claude 在图中识别“Deploy”按钮，返回缩放图坐标。
3. 宿主按缩放因子反算真实坐标并点击。
4. 页面出现“任务已创建”提示后，再截图回传。
5. `text_editor` 追加日志：`已触发部署，job_id=...`。
6. `bash` 执行 `./deploy.sh --track job_id` 进入后续自动化阶段。

这个例子说明：Computer Use 的价值，不是替代全部自动化，而是把“必须经过界面确认”的那一小段流程，安全嵌入更长的工程流水线中。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它不依赖真实截图，也不依赖第三方库，只模拟三件事：

1. 计算截图缩放因子。
2. 把模型返回的缩放图坐标反算成真实点击点。
3. 用一个最小代理循环把 `computer`、`text_editor`、`bash` 串起来。

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MAX_EDGE = 1568
MAX_PIXELS = 1_150_000


@dataclass(frozen=True)
class Screen:
    width: int
    height: int


@dataclass(frozen=True)
class Screenshot:
    original: Screen
    scale: float
    scaled_width: int
    scaled_height: int


@dataclass(frozen=True)
class ToolCall:
    tool: Literal["computer", "text_editor", "bash"]
    action: str
    payload: dict


class ProtocolError(ValueError):
    pass


def calc_scale(screen: Screen) -> float:
    if screen.width <= 0 or screen.height <= 0:
        raise ProtocolError("screen width and height must be positive")

    edge_limit = MAX_EDGE / max(screen.width, screen.height)
    pixel_limit = math.sqrt(MAX_PIXELS / (screen.width * screen.height))
    return min(1.0, edge_limit, pixel_limit)


def build_screenshot(screen: Screen) -> Screenshot:
    scale = calc_scale(screen)
    return Screenshot(
        original=screen,
        scale=scale,
        scaled_width=max(1, round(screen.width * scale)),
        scaled_height=max(1, round(screen.height * scale)),
    )


def to_real_coords(x_prime: float, y_prime: float, scale: float) -> tuple[int, int]:
    if scale <= 0:
        raise ProtocolError("scale must be greater than zero")
    return round(x_prime / scale), round(y_prime / scale)


def clamp_point(x: int, y: int, screen: Screen) -> tuple[int, int]:
    safe_x = min(max(x, 0), screen.width - 1)
    safe_y = min(max(y, 0), screen.height - 1)
    return safe_x, safe_y


def execute_computer_click(screenshot: Screenshot, x_prime: int, y_prime: int) -> tuple[int, int]:
    real_x, real_y = to_real_coords(x_prime, y_prime, screenshot.scale)
    return clamp_point(real_x, real_y, screenshot.original)


def run_text_editor(path: str, text: str) -> str:
    file_path = Path(path)
    old_text = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    file_path.write_text(old_text + text, encoding="utf-8")
    return f"wrote {len(text)} chars to {file_path}"


def run_bash(cmd: str) -> str:
    # 玩具实现：真实系统中这里应调用受限 shell 或任务执行器
    return f"execute: {cmd}"


def mock_model_plan(step: int) -> ToolCall:
    """
    用固定规则模拟 Claude 的多步决策：
    1. 先点击 Deploy
    2. 再写日志
    3. 最后执行 deploy.sh
    """
    if step == 0:
        return ToolCall(
            tool="computer",
            action="left_click",
            payload={"x": 400, "y": 400},
        )
    if step == 1:
        return ToolCall(
            tool="text_editor",
            action="append",
            payload={"path": "deploy.log", "text": "已触发部署\n"},
        )
    if step == 2:
        return ToolCall(
            tool="bash",
            action="run",
            payload={"cmd": "./deploy.sh"},
        )
    raise StopIteration("workflow finished")


def run_agent_loop(screen: Screen) -> list[str]:
    events: list[str] = []

    step = 0
    while True:
        screenshot = build_screenshot(screen)
        events.append(
            f"screenshot: {screenshot.scaled_width}x{screenshot.scaled_height}, scale={screenshot.scale:.4f}"
        )

        try:
            tool_call = mock_model_plan(step)
        except StopIteration:
            events.append("done")
            break

        if tool_call.tool == "computer":
            click_x, click_y = execute_computer_click(
                screenshot,
                tool_call.payload["x"],
                tool_call.payload["y"],
            )
            events.append(f"computer.{tool_call.action}: ({click_x}, {click_y})")

        elif tool_call.tool == "text_editor":
            result = run_text_editor(
                tool_call.payload["path"],
                tool_call.payload["text"],
            )
            events.append(f"text_editor.{tool_call.action}: {result}")

        elif tool_call.tool == "bash":
            result = run_bash(tool_call.payload["cmd"])
            events.append(f"bash.{tool_call.action}: {result}")

        else:
            raise ProtocolError(f"unsupported tool: {tool_call.tool}")

        step += 1

    return events


def _self_check() -> None:
    screen = Screen(1920, 1080)
    shot = build_screenshot(screen)

    assert round(shot.scale, 4) == round(
        min(1.0, 1568 / 1920, math.sqrt(1_150_000 / (1920 * 1080))), 4
    )

    real_x, real_y = execute_computer_click(shot, 400, 400)
    assert 530 <= real_x <= 545
    assert 530 <= real_y <= 545

    events = run_agent_loop(screen)
    assert events[0].startswith("screenshot:")
    assert any("computer.left_click" in item for item in events)
    assert any("text_editor.append" in item for item in events)
    assert any("bash.run" in item for item in events)
    assert events[-1] == "done"


if __name__ == "__main__":
    _self_check()
    for line in run_agent_loop(Screen(1920, 1080)):
        print(line)
```

这段代码能直接运行，执行后会输出一个最小闭环。它没有真实接入 Anthropic API，但协议层逻辑是完整的：先“截图”，再“模型决定工具”，然后执行动作并进入下一轮。

这段代码里有几个新手最容易写错的点。

| 容易写错的位置 | 错误写法 | 正确写法 | 原因 |
|---|---|---|---|
| 缩放方向 | `real = scaled * s` | `real = scaled / s` | 模型坐标来自缩小后的图，回到原屏幕必须除以缩放因子 |
| 缩放因子来源 | 每个函数各算一遍 | 整个轮次统一使用同一个 `scale` | 否则截图和点击依据不同，协议失真 |
| 取整时机 | 先取整再反算 | 反算后再取整 | 避免误差提前放大 |
| 边界处理 | 直接点击 | 先 `clamp` 到屏幕范围 | 防止底层自动化库越界报错 |
| 工具职责 | 全都走 `computer` | 文件改动走 `text_editor`，命令走 `bash` | 这样更快、更稳、更便宜 |

如果要把玩具代码扩成真实服务，主循环通常会变成下面这样：

1. `capture_screenshot()`
2. 如果截图过大，先按协议缩放
3. 把截图和上下文发给 Claude
4. 读取模型返回的 `computer`、`text_editor`、`bash` 工具调用
5. 对 `computer` 动作做坐标反算并执行
6. 把工具执行结果作为 `tool_result` 回传
7. 如果界面状态变化，再抓新图进入下一轮

也就是说，真正的关键不是“调用 API”，而是确保每一轮看到的图、返回的坐标、实际执行的动作，都属于同一个协议上下文。

---

## 工程权衡与常见坑

工程上最贵的通常不是点击，而是截图。因为截图会进入多模态输入，成本通常和图片数量近似线性相关。可以先用一个简化估算式理解：

$$
\text{total\_tokens} \approx n_{\text{shots}} \times \text{tokens\_per\_shot} + \text{tool\_context} + \text{model\_output}
$$

如果按经验量级估算，每张截图约 `1500 token`，那么 10 次独立截图大约就是：

$$
10 \times 1500 = 15000
$$

这还不包含工具说明、历史消息和模型自己的输出。也就是说，一个“点按钮 → 等加载 → 再点确认 → 再看结果”的长链路，很快就会比纯 `bash` 自动化贵得多。

看一个更容易理解的流程成本例子：

| 步骤 | 是否需要新截图 | 原因 |
|---|---|---|
| 定位“Deploy”按钮 | 是 | 必须先观察当前界面 |
| 点击后等待弹窗 | 是 | 界面状态已变化 |
| 确认弹窗内容 | 是 | 需要重新识别文本和按钮 |
| 写日志 | 否 | 可直接用 `text_editor` |
| 运行 `./deploy.sh` | 否 | 可直接用 `bash` |

这意味着“点按钮 + 写日志 + 跑命令”并不一定等于 3 次截图。合理的工程设计，往往只让 GUI 处理 GUI 必须完成的那几步，剩下全部交还给文件或命令工具。

常见坑可以归纳如下：

| 坑 | 直接后果 | 根因 | 规避方式 |
|---|---|---|---|
| 连续截图过多 | token 成本快速上升 | 每个小动作都重新观察 | 合并无状态动作，减少观察轮次 |
| 使用高分辨率屏幕 | 误点概率上升 | 元素在缩放后更小 | 控制在 `1280×800` 或更低 |
| 忽略缩放映射 | 点到错误位置 | 模型坐标和真实坐标不在同一空间 | 显式维护 `scale` 和逆变换 |
| 动作后不重新截图 | 流程漂移 | 仍按旧界面做决策 | 每次界面变化后重新观察 |
| 把所有工作都交给 `computer` | 慢、贵、脆弱 | 错把 GUI 当唯一执行面 | 文件走 `text_editor`，命令走 `bash` |
| 运行环境权限过大 | 风险放大 | 模型可触达过多系统资源 | 容器化、白名单、只读挂载、隔离网络 |
| 忽略等待状态 | 连续误操作 | 页面尚未加载完成 | 显式等待或检查新截图中的状态信号 |

这里最容易被忽视的是“等待”。在传统脚本里，很多人习惯 `sleep(1)` 后继续执行，但 Computer Use 更适合“基于画面状态等待”，例如等到弹窗标题出现、按钮可见、加载动画消失，再进入下一步。因为 GUI 的真实问题不是“等几秒”，而是“界面是否已经变成可操作状态”。

另一个常见误区是把 Computer Use 当作“万能 RPA”。这会导致设计失衡。只要某一步已经可以被 CLI、API 或直接改文件稳定替代，就不应该继续让模型看屏幕。GUI 自动化最适合处理“只有界面里才有”的状态，例如：

| 适合交给 GUI 的步骤 | 原因 |
|---|---|
| 点人工确认按钮 | 状态只存在于界面里 |
| 检查某个控件是否禁用 | 后端不一定暴露这个状态 |
| 验证页面是否出现成功提示 | 文本或样式只在前端可见 |
| 处理只存在于网页里的门禁流程 | 没有公开 API |

相反，下面这些步骤通常应尽早切回 `bash` 或 API：

| 更适合 CLI/API 的步骤 | 原因 |
|---|---|
| 拉代码、跑脚本、查进程 | 终端接口更稳定 |
| 写配置、追加日志、替换文本 | 文件操作更直接 |
| 查询任务状态 | API 更可重复、可监控 |
| 高频重复执行的流程 | GUI 太慢且成本高 |

一句话说，GUI 应该是补洞工具，不是主通道。

---

## 替代方案与适用边界

Computer Use 不是默认选项，而是最后补上的一层 GUI 能力。判断是否该用它，先看有没有更直接的执行面。

| 方案 | 适用条件 | 优点 | 限制 |
|---|---|---|---|
| 纯 CLI / `bash` | 系统本来就有命令接口 | 最稳、最快、最便宜 | 不能处理必须人工点击的界面 |
| API 直连 | 后端暴露了稳定接口 | 可测试、可监控、可重试 | 对只存在于前端的状态无能为力 |
| `text_editor` + `bash` | 任务核心是改文件和跑命令 | 适合工程自动化 | 不解决图形界面确认问题 |
| 浏览器脚本 / DOM 自动化 | 页面结构稳定、元素可定位 | 精度高于像素点击 | 只适用于浏览器环境 |
| Computer Use | 必须看界面、点控件、读屏幕状态 | 通用性强，可覆盖非结构化界面 | 成本高、速度慢、精度受限 |

一个直观对比是“完全靠 `bash` 部署”和“需要 UI 验证再部署”。

前者适合标准化工程系统：脚本发命令、轮询状态、写日志，全程不看屏幕，成本最低，稳定性最高。

后者适合带人工门禁的系统：例如某个发布平台要求必须在网页中勾选“我已确认风险”，然后点击“Deploy”。这一步没有公开 API，前端结构又不稳定，写浏览器脚本也不划算，那么 Computer Use 就有价值，因为它能通过截图理解和坐标点击完成最后一跳。

可以把选择逻辑整理成一个简单决策表：

| 问题 | 如果答案是“是” | 建议 |
|---|---|---|
| 有没有稳定 API？ | 是 | 直接用 API，不用 Computer Use |
| 有没有稳定 CLI？ | 是 | 优先 `bash` |
| 是否只需改文本文件？ | 是 | 优先 `text_editor` |
| 状态是否只存在于 GUI？ | 是 | 才考虑 `computer` |
| 是否要求高频、低延迟、像素级精度？ | 是 | 不适合 Computer Use |
| 是否运行在受限沙箱？ | 否 | 先补齐隔离和权限控制 |

它的适用边界也要说清楚。适合的场景包括：GUI 验证、内部运维面板操作、CI/CD 中少量人工门禁步骤、回填日志与脚本执行需要纳入同一控制回路的场景。不适合的场景包括：高频实时交易、精细绘图、长时间连续桌面操控、高 DPI 精度要求极高的操作，以及任何可以被纯 API 稳定替代的流程。

一句话总结选择原则：能用 API 就别看屏幕，能用 `bash` 就别点按钮，只有当“界面本身是任务的一部分”时，才引入 Computer Use。

---

## 参考资料

下表按“定义协议边界”“确认尺寸与坐标规则”“补充工程经验”三个层次组织。

| 来源 | 内容概述 | 用途 |
|---|---|---|
| Anthropic Docs: Computer Use | 说明 `computer` 工具如何截图、点击、输入、滚动，以及如何与其他工具组合 | 用于界定 GUI 工具职责和 agent loop 结构 |
| Anthropic Docs: Text Editor Tool | 说明文本编辑工具的定位、版本和适用方式 | 用于明确 `text_editor` 不等于屏幕点击，而是文件级操作 |
| Anthropic Docs: Bash Tool | 说明持久 shell 会话、命令执行与自动化用途 | 用于明确 `bash` 是命令执行面，不应被 GUI 替代 |
| Claude Platform Docs: Computer Use Tool | 给出尺寸限制、缩放示例、坐标反向映射说明 | 用于本文缩放公式、坐标系推导与实现细节 |
| Anthropic 官方参考实现与安全说明 | 强调沙箱、容器、权限限制、提示注入风险 | 用于界定运行边界和工程安全要求 |
| 工程案例文章 | 提供截图成本量级、分辨率经验与实际工作流示例 | 用于补充成本估算和应用场景分析 |

1. Anthropic Docs: Computer Use  
   https://docs.anthropic.com/en/docs/build-with-claude/computer-use

2. Claude Platform Docs: Computer Use Tool  
   https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool

3. Anthropic Docs: Text Editor Tool  
   https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool

4. Anthropic Docs: Bash Tool  
   https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/bash-tool

5. Anthropic Computer Use Reference Implementation  
   https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo

6. Athenic Blog: Anthropic Computer Use / Claude Desktop Control  
   https://getathenic.com/blog/anthropic-computer-use-claude-desktop-control

7. Digital Applied Blog: Anthropic Computer Use API Guide  
   https://www.digitalapplied.com/blog/anthropic-computer-use-api-guide
