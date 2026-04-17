## 核心结论

网页元素定位的本质，不是“在屏幕上找一个点”，而是把用户意图稳定地映射为浏览器中一个可执行对象。这个对象最终必须能被点击、输入、选择或读取。只要这一步不稳定，后续动作链路再复杂也没有意义。

在浏览器 Agent 场景中，`Set-of-Mark`，简称 `SoM`，可以理解为一种“候选元素离散化标注”方法。系统先从 DOM 和可访问性信息中提取当前页面里可能被操作的元素，再把这些元素的边界框叠加到截图上，并给每个候选分配唯一编号。模型看到的就不再是“右上角那个蓝色按钮”这种模糊描述，而是“点击[5]”“在[3]输入邮箱”这种可以记录、回放和审计的动作指令。

它的工程价值不在“编号这件事本身”，而在于它把视觉理解层和执行层明确拆开了。模型负责在有限候选中做判断，控制器负责把编号还原回 DOM 句柄、选择器、边界框或具体浏览器动作。这样做的结果是，模型不需要直接生成 XPath 或像素坐标，执行器也不需要重新猜测用户到底想操作哪一个控件。

| 方法 | 依赖内容 | 输出形式 | 抗页面变动能力 | 可调试性 |
| --- | --- | --- | --- | --- |
| CSS 选择器 | `id`、`class`、DOM 结构 | `#login-btn` | 中到低 | 中 |
| XPath | DOM 路径、属性、层级关系 | `/html/body/...` | 低 | 低 |
| 坐标点击 | 屏幕像素位置 | `(x, y)` | 很低 | 很低 |
| SoM 标注 | DOM 元数据 + 截图 + 编号映射 | `点击[5]` | 高 | 很高 |

这里有三个直接结论。

第一，SoM 把定位问题从“预测连续坐标”改成了“在有限候选中选择编号”。连续坐标要求模型自己处理缩放、滚动、截图分辨率、遮挡和视口变化；离散编号只要求它在当前候选集合中做分类判断，稳定性通常更高。

第二，SoM 天然适合审计。日志里如果记录的是“模型在 2026-03-08 14:31:05 选择了 `[7]`”，工程上就可以回放当时的截图、编号图、候选元数据和执行结果，分析错误到底发生在候选提取、模型决策还是动作执行环节。

第三，SoM 不是要替代 DOM 选择器，而是把选择器放到控制器后端。模型前面只看编号和上下文，系统后面再用编号恢复真实执行对象。这种分层通常比直接要求模型生成 CSS 或 XPath 更稳，因为执行细节不暴露给模型，模型也不需要掌握页面内部实现。

一个足够实用的结论是：如果目标是构建“可解释、可调试、可回放”的浏览器 Agent，SoM 通常优于纯坐标点击，也通常比直接让模型输出 CSS/XPath 更适合作为主路径。

---

## 问题定义与边界

问题可以形式化为：

给定一条自然语言指令，例如“打开设置”“填写邮箱并提交”，系统需要在当前网页状态下识别对应的目标元素，并将其转换成一个可执行动作。

这个定义里有两个边界必须先明确。

第一，定位目标不是任意 DOM 节点，而是“当前可操作”的候选集合。可操作通常包括按钮、输入框、下拉框、复选框、单选框、链接，以及带交互语义的自定义组件。所谓“语义”，就是该元素在页面里承担的角色信息，常来自标签名、文本内容、`role`、`aria-label`、可见性、启用状态以及绑定的交互行为。

第二，网页元素定位不是纯视觉问题，也不是纯 DOM 查询问题。浏览器页面至少同时存在三层信息：

| 信息层 | 例子 | 优势 | 局限 |
| --- | --- | --- | --- |
| DOM 结构层 | `button`、`input`、`href`、`aria-label` | 易检索、易执行、信息明确 | 不等于用户真实看到的区域 |
| 渲染结果层 | 截图、颜色、布局、遮挡、层叠关系 | 接近用户视角 | 不直接携带可执行句柄 |
| 任务语义层 | “提交订单”“关闭弹窗”“切换到高级设置” | 决定目标是谁 | 需要和前两层对齐 |

所以真正的问题不是“页面上有没有按钮”，而是“任务语义如何对齐到当前视觉区域，再对齐到可执行 DOM 对象”。如果这三层没有对齐，常见错误会非常直接：

| 失败类型 | 现象 | 根因 |
| --- | --- | --- |
| 语义错配 | 把“提交”点成“保存草稿” | 文本相近，但任务约束不足 |
| 视觉错配 | 识别到了被遮挡按钮 | 截图和可操作状态不一致 |
| 执行错配 | 模型选对区域，执行点错元素 | 编号与 DOM 映射失效 |
| 候选污染 | 候选太多，模型选到噪声节点 | 候选提取和去重不足 |

评价定位系统时，通常会用 `Precision`、`Recall` 和 `F1`。它们分别衡量“候选准不准”“漏得多不多”和“综合平衡”。

$$
Precision=\frac{|P\cap G|}{|P|},\quad
Recall=\frac{|P\cap G|}{|G|},\quad
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}
$$

其中，`P` 是系统给出的候选集合，`G` 是真实相关目标集合。

这个定义要注意一点：它评估的是“候选提取得好不好”，不是最终动作成功率。动作成功率还会额外受到模型选择、执行时序、页面变化和容错策略影响。因此工程上通常会同时看两组指标：

| 指标层级 | 指标 | 含义 |
| --- | --- | --- |
| 候选层 | Precision / Recall / F1 | 候选集合是否覆盖正确元素、噪声是否过多 |
| 决策层 | Top-1 / Top-k 命中率 | 模型选中的编号是否正确 |
| 执行层 | Action success rate | 点击、输入、选择是否真正完成 |
| 任务层 | Task completion rate | 整个任务是否最终完成 |

玩具例子：一个登录页的真实相关元素有 4 个，分别是邮箱框、密码框、登录按钮、忘记密码链接。系统提取了 6 个候选，其中只有 3 个真的相关。那么：

- `Precision = 3/6 = 0.5`
- `Recall = 3/4 = 0.75`
- `F1 ≈ 0.60`

这说明系统不是“完全没找到”，而是“候选集合噪声较大”。SoM 的作用之一，就是把这些候选整理成模型更容易消费的离散空间；但如果候选集本身很差，SoM 不会自动把坏候选变成好决策。

---

## 核心机制与推导

SoM 的工作流通常可以拆成四步：

| 步骤 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| 候选提取 | DOM、可见文本、ARIA、布局信息 | 候选元素列表 | 定义“谁有资格被操作” |
| 标注生成 | 候选列表 + 页面截图 | 编号截图 + 映射表 | 定义“模型看到谁” |
| 模型决策 | 任务描述 + 编号截图 + 候选上下文 | `点击[5]` | 定义“模型选谁” |
| 动作执行 | 编号到元素的映射 | `click/type/select/read` | 定义“系统真正操作谁” |

核心转折点是第二步：把连续页面空间变成离散编号空间。

设当前页面提取到了 `n` 个候选元素。传统坐标点击要求模型在二维平面输出一个点 `(x, y)`，本质上是回归问题：

$$
f_{coord}(instruction, screenshot)\rightarrow (x,y)
$$

而 SoM 把问题改写成编号选择：

$$
f_{som}(instruction, screenshot, C)\rightarrow k,\quad k\in\{1,2,\dots,n\}
$$

其中 `C` 是候选集合。前者要求模型直接生成绝对空间位置，后者要求模型在有限集合中做分类判断。分类一般比回归更容易稳定，原因不是“模型更聪明”，而是搜索空间被大幅压缩了。

控制器维护的映射可以写成：

$$
M: label \rightarrow (bbox, dom\_handle, selector, role, text)
$$

当模型输出 `点击[5]` 时，系统并不重新理解“登录按钮在哪”，而只是做查表和执行：

$$
target = M(5)
$$

$$
action = Execute(target)
$$

如果要再细一点，整个链路可以抽象成：

$$
instruction \xrightarrow[\text{候选提取}]{page} C
\xrightarrow[\text{叠加编号}]{screenshot} \hat{I}
\xrightarrow[\text{模型决策}]{} k
\xrightarrow[\text{查表}]{} M(k)
\xrightarrow[\text{浏览器控制}]{} result
$$

这就是 SoM 的核心机制：模型负责“选标签”，控制器负责“落地动作”，中间通过稳定映射解耦。这样一来，前端实现细节不需要暴露给模型，模型也不需要直接输出容易失效的执行代码。

一个新手最容易理解的例子是登录页：

- `[3]` 邮箱输入框
- `[4]` 密码输入框
- `[5]` 登录按钮

用户说“登录这个网站”，模型看到带编号的截图后，输出：

1. 在 `[3]` 输入邮箱
2. 在 `[4]` 输入密码
3. 点击 `[5]`

如果第二天前端改版，把登录按钮的 `id` 从 `login-btn` 改成 `submit-primary`，那么基于旧选择器的脚本可能立刻失效；但只要 SoM 重新从当前页面提取候选并重新编号，只要那个按钮依然是当前视觉上明确的“登录按钮”，模型仍然可以继续输出对应编号。

真实工程里这个过程会更复杂。以报销系统为例：

- 左侧菜单、顶部通知、正文表单、底部固定操作栏同时存在
- “提交”按钮可能被禁用、滚动后悬浮、或被确认弹窗覆盖
- 有些控件是自定义组件，没有稳定 `id`
- 不同部门模板略有差异，但业务语义相同

在这种情况下：

- 纯 XPath 往往脆弱，因为层级一改就断
- 纯坐标更脆弱，因为滚动、分辨率、缩放一变就偏
- SoM 则倾向于在每个页面状态上重新构建当前候选，再让模型引用当前页的 `[12]`、`[18]`、`[23]`

这里要特别强调一个容易被忽略的事实：SoM 的“稳定”不是指“编号全局固定”，而是指“编号在当前页面快照和映射表里可追踪、可回放、可执行”。它稳定的是协议，不是数字本身。

当然，SoM 也不是零成本方案。它对候选质量非常敏感。如果同一个按钮被提取成两个几乎重叠的框，模型就会看到两个极其相似的标签；如果候选里混入大量不可点击节点，模型的选择难度也会上升。因此工程上通常要在候选层先做整理：

| 处理项 | 目的 | 常见规则 |
| --- | --- | --- |
| 可见性过滤 | 去掉用户当前看不到的元素 | `display != none`、有尺寸、在视口内 |
| 可交互过滤 | 去掉无动作意义的节点 | `button`、`input`、链接、有 `role` |
| 去重 | 合并重复候选 | DOM 唯一性、文本+角色、IoU |
| 大小过滤 | 去掉异常小或异常大的框 | 最小面积、最大面积比例 |
| 语义增强 | 给模型更多可判别信息 | 文本、角色、邻近标签、状态 |

否则，SoM 只是在把“定位问题”转移成“候选生成问题”。

---

## 代码实现

下面给出一个最小可运行的 Python 示例。它不依赖浏览器截图，但完整演示了 SoM 的核心数据结构、去重、编号映射和动作执行。代码可以直接运行：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Element:
    label: int
    role: str
    text: str
    bbox: BBox
    selector: str
    visible: bool = True
    enabled: bool = True


def area(box: BBox) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def is_interactive(elem: Element) -> bool:
    interactive_roles = {"button", "textbox", "link", "checkbox", "combobox"}
    return elem.visible and elem.enabled and elem.role in interactive_roles and area(elem.bbox) > 0


def dedupe(elements: Iterable[Element], threshold: float = 0.80) -> List[Element]:
    kept: List[Element] = []

    for elem in elements:
        if not is_interactive(elem):
            continue

        duplicate_of_existing = False
        for old in kept:
            same_role = elem.role == old.role
            same_text = elem.text.strip().lower() == old.text.strip().lower()
            highly_overlapped = iou(elem.bbox, old.bbox) >= threshold

            if same_role and same_text and highly_overlapped:
                duplicate_of_existing = True
                break

        if not duplicate_of_existing:
            kept.append(elem)

    return kept


def build_element_map(elements: Iterable[Element]) -> Dict[int, Element]:
    mapping: Dict[int, Element] = {}
    for elem in elements:
        if elem.label in mapping:
            raise ValueError(f"duplicated label: {elem.label}")
        mapping[elem.label] = elem
    return mapping


def execute(
    element_map: Dict[int, Element],
    action: str,
    label: int,
    value: Optional[str] = None,
) -> str:
    if label not in element_map:
        raise KeyError(f"unknown label: {label}")

    target = element_map[label]

    if action == "click":
        return f"click({target.selector})"

    if action == "type":
        if value is None:
            raise ValueError("type action requires value")
        return f"type({target.selector}, value={value!r})"

    if action == "read":
        return f"read_text({target.selector})"

    raise ValueError(f"unknown action: {action}")


def main() -> None:
    raw_elements = [
        Element(3, "textbox", "Email", (10, 10, 210, 50), "#email"),
        Element(4, "textbox", "Password", (10, 60, 210, 100), "#password"),
        Element(5, "button", "Login", (10, 110, 120, 150), "#login-btn"),
        Element(6, "button", "Login", (12, 112, 122, 152), "#login-btn"),
        Element(7, "link", "Forgot Password", (10, 160, 160, 190), "a.forgot"),
        Element(8, "generic", "Decorative Wrapper", (0, 0, 400, 300), "div.wrapper"),
    ]

    candidates = dedupe(raw_elements)
    element_map = build_element_map(candidates)

    print("Candidates after dedupe:")
    for elem in candidates:
        print(f"[{elem.label}] role={elem.role:<8} text={elem.text!r} selector={elem.selector}")

    print()
    print(execute(element_map, "type", 3, "alice@example.com"))
    print(execute(element_map, "type", 4, "secret-password"))
    print(execute(element_map, "click", 5))
    print(execute(element_map, "read", 7))


if __name__ == "__main__":
    main()
```

这段代码运行后的典型输出如下：

```text
Candidates after dedupe:
[3] role=textbox  text='Email' selector=#email
[4] role=textbox  text='Password' selector=#password
[5] role=button   text='Login' selector=#login-btn
[7] role=link     text='Forgot Password' selector=a.forgot

type(#email, value='alice@example.com')
type(#password, value='secret-password')
click(#login-btn)
read_text(a.forgot)
```

这个玩具实现体现了四个关键点：

| 点 | 说明 |
| --- | --- |
| `label` 是模型接口 | 模型不需要知道 CSS 选择器，只需要输出编号 |
| `element_map` 是控制器接口 | 执行层通过 `label -> element` 恢复真实对象 |
| 去重是基本能力 | 否则同一个按钮可能出现多个标签 |
| 过滤先于决策 | 候选质量差，后面的模型再强也会被拖累 |

如果换成真实浏览器工程，常见流程一般是：

1. 用 Playwright、Selenium 或 CDP 获取当前页面状态。
2. 过滤出可见且可交互的元素。
3. 读取每个元素的文本、角色、`aria-label`、边界框和启用状态。
4. 在截图上叠加编号，同时保留 `label -> element` 映射表。
5. 把任务指令、截图和候选摘要送入多模态模型。
6. 从模型输出中解析 `[k]`。
7. 用 `element_map[k]` 执行点击、输入、选择或读取。
8. 如果失败，进入分层 fallback。

下面给一个更接近真实工程的 Playwright 伪代码。它不是完整产品代码，但结构可以直接迁移到项目里：

```python
from playwright.sync_api import sync_playwright

JS_GET_CANDIDATES = """
() => {
  const nodes = Array.from(document.querySelectorAll('*'));
  const candidates = [];

  for (const el of nodes) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    const role = (el.getAttribute('role') || el.tagName).toLowerCase();
    const text = (el.innerText || el.getAttribute('aria-label') || '').trim();

    const interactiveTags = new Set(['button', 'input', 'select', 'textarea', 'a']);
    const interactiveRoles = new Set(['button', 'link', 'textbox', 'checkbox', 'combobox']);

    const isInteractive =
      interactiveTags.has(el.tagName.toLowerCase()) || interactiveRoles.has(role);

    const visible =
      rect.width > 0 &&
      rect.height > 0 &&
      style.visibility !== 'hidden' &&
      style.display !== 'none';

    if (!isInteractive || !visible) continue;

    candidates.push({
      role,
      text,
      bbox: {
        x1: Math.round(rect.left),
        y1: Math.round(rect.top),
        x2: Math.round(rect.right),
        y2: Math.round(rect.bottom),
      },
      selectorHint: el.id ? `#${el.id}` : el.tagName.toLowerCase(),
    });
  }

  return candidates;
}
"""

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("https://example.com/login")
    page.wait_for_load_state("networkidle")

    candidates = page.evaluate(JS_GET_CANDIDATES)
    screenshot_path = "page.png"
    page.screenshot(path=screenshot_path, full_page=False)

    # 真实系统会在这里做：
    # 1. 去重
    # 2. 叠加编号
    # 3. 发送给多模态模型
    # 4. 根据模型输出的 label 执行动作

    browser.close()
```

这个例子说明，SoM 并不是“放弃 DOM、只看图片”，而是把 DOM 和图片组合起来：DOM 负责提供可执行对象，图片负责提供用户视角下的空间和视觉上下文，编号负责把两者接起来。

---

## 工程权衡与常见坑

四种主流定位方式各有明显边界：

| 方法 | 优点 | 常见失效点 | 适合做什么 |
| --- | --- | --- | --- |
| CSS 选择器 | 执行快、实现简单、成本低 | `class`/`id` 变化、层级调整 | 稳定页面上的固定控件 |
| XPath | 表达力强、适合复杂关系查询 | 路径脆弱、调试困难 | 遗留系统、结构明确的页面 |
| 坐标点击 | 不依赖 DOM、最后可兜底 | 缩放、滚动、遮挡、分辨率变化 | 临时自动化、极端兜底 |
| SoM | 可解释、可回放、结构变化下更稳 | 候选质量差时误导模型 | 浏览器 Agent 主路径 |

真正落地时，最常见的坑至少有五类。

第一，候选过多。  
如果系统给页面上几百个节点全部编号，多模态模型会被噪声淹没。经验上应优先保留可交互元素，再按任务需要补充少量关键文本节点，例如表单标签、提示文案、弹窗标题。

第二，候选重复。  
一个自定义按钮可能同时被外层 `div`、中层 `span` 和内层 `button` 捕获。如果这三个框都上屏，模型看到的是三个几乎相同的目标。去重不是优化项，而是主流程的一部分。

第三，截图与 DOM 不一致。  
页面动画、懒加载、异步弹窗、滚动恢复、骨架屏替换都会让“截图时的视觉状态”和“提取候选时的 DOM 状态”发生偏差。解决方式不是靠模型猜，而是冻结时序：等待页面稳定，在同一时刻采样 DOM、布局信息和截图。

第四，fallback 设计不合理。  
有些系统一旦 SoM 没选中，就直接切到坐标点击。短期看像是提高了成功率，长期看会让问题更难排查，因为系统失去了“为什么点这里”的解释链。更合理的顺序通常是：

1. SoM 编号命中，直接执行。
2. 编号失效，但保留了原 DOM 句柄，优先直接执行句柄。
3. 句柄失效，尝试文本、角色、属性进行重定位。
4. 最后才进入坐标点击兜底。

第五，编号本身没有语义。  
`[12]` 只是索引，不携带“它是提交按钮”这种解释。因此给模型的输入通常不应只有编号截图，还要补充候选摘要，例如文本、角色、邻近说明、启用状态。否则模型只能靠视觉猜测，稳定性会下降。

下面这张表可以帮助判断问题更可能出在什么位置：

| 现象 | 更可能的根因 | 优先检查项 |
| --- | --- | --- |
| 选中了相邻按钮 | 候选过密或编号遮挡内容 | 标注样式、候选粒度 |
| 选中了不可点击元素 | 候选过滤不足 | 可见性、启用状态、角色判断 |
| 执行时报元素不存在 | 映射失效或页面刷新 | 执行时序、页面跳转、DOM 重建 |
| 同页重复失败 | 候选提取策略有系统性偏差 | 去重逻辑、面积阈值、交互规则 |
| 某些机器稳定、另一些机器不稳定 | 环境差异影响坐标或布局 | DPI、缩放、字体、视口尺寸 |

还有一个非常现实的故障模式是高缩放屏幕。假设开发机使用 100% 缩放，而用户机器使用 200% 缩放，同时接了额外显示器。纯坐标脚本在开发机上点击 `(1240, 760)` 正常，在用户机器上可能已经落到相邻控件甚至另一块屏幕区域。SoM 的价值就在这里：它依赖当前页面重新提取出的候选和重新渲染后的截图，而不是依赖一组脱离上下文的绝对像素坐标。

---

## 替代方案与适用边界

SoM 不是唯一方案，也不是所有场景下都最优。更准确的做法是把它放回整个自动化系统的设计空间里比较。

第一类替代方案是纯 DOM 方案。  
这类方法完全依赖 CSS 选择器、XPath、文本匹配或可访问性树。它的优势是快、便宜、实现直接，不需要截图也不需要视觉模型。在结构稳定、模板固定、页面变化少的内部系统里，这通常仍然是性价比最高的方案。

第二类替代方案是纯视觉坐标方案。  
模型直接根据截图输出点击位置，或者输出目标区域框。这类方案链路最短，前期实现简单，但审计和复现最弱。模型一旦点错，往往很难区分是视觉理解错了，还是执行时坐标漂移了。

第三类替代方案是文本候选列表方案。  
系统先提取候选元素，但不在截图上画编号，而是把候选写成结构化文本列表，例如“1. 按钮：登录；2. 输入框：邮箱；3. 链接：忘记密码”。这样比纯视觉便宜，也保留了 DOM 执行能力，但会损失布局、颜色、邻近关系、遮挡状态等视觉信息。

可以把几种方案放在一张表里比较：

| 方案 | 是否依赖视觉 | 是否保留 DOM 执行能力 | 可解释性 | 延迟/成本 | 适用边界 |
| --- | --- | --- | --- | --- | --- |
| 纯 DOM 选择器 | 否 | 是 | 中 | 低 | 页面结构稳定、任务固定 |
| 纯坐标点击 | 是 | 否 | 低 | 中 | 临时自动化、最终兜底 |
| 文本候选列表 | 弱依赖 | 是 | 中到高 | 中低 | 页面简单、视觉关系不重要 |
| SoM | 是 | 是 | 高 | 中到高 | 多步骤 Agent、审计要求高 |

SoM 最适合的边界通常是：

- 页面变化频繁，但仍能稳定提取候选元素。
- 任务是多步骤的，需要追踪每一步“模型为什么这样选”。
- 需要把多模态理解和浏览器执行明确解耦。
- 系统要和 DOM 选择器、文本匹配、坐标点击一起组成分层 fallback 链路。

SoM 不一定最适合的边界通常是：

- 页面完全固定，直接使用选择器更便宜、更快。
- 对延迟极端敏感，无法承担截图和视觉推理开销。
- 候选提取质量本身很差，导致编号图噪声过大。
- 页面高度图形化，真实可操作区域在 DOM 中没有稳定映射，例如某些 Canvas 或远程流式桌面场景。

因此，更准确的结论不是“以后都用 SoM”，而是“把 SoM 作为浏览器 Agent 的主定位层，并为执行层保留 DOM 和坐标兜底”。这是一种架构分层选择，不是单点技术崇拜。

---

## 参考资料

下面这些资料可以按用途分组阅读，而不是混在一起看。

### 1. SoM 与多模态网页定位

- Microsoft SoM / Set-of-Mark 介绍：https://som-gpt4v.github.io/
- Multimodal-Mind2Web 论文页面：https://openreview.net/forum?id=kxnoqaisCT
- ICLR 2025 论文 PDF：https://proceedings.iclr.cc/paper_files/paper/2025/file/4ca0e369689dadb25a5345ba9755ad6f-Paper-Conference.pdf

### 2. 工程落地与 Grounding

- OpenAdapt Grounding 文档：https://docs.openadapt.ai/packages/grounding/
- OpenAdapt 架构演进说明：https://docs.openadapt.ai/architecture-evolution/
- RunNCap 项目说明：https://yanwangweb.com/projects/runncap

### 3. 传统定位方法对比

- XPath 与 CSS Selector 对比：https://www.geeksforgeeks.org/software-testing/xpath-vs-css-selector-in-java/

### 4. 坐标与设备差异问题

- PyAutoGUI 多屏缩放问题讨论：https://github.com/asweigart/pyautogui/issues/413

### 5. 候选框去重与检测相关背景

- 对象检测中候选框去重相关研究：https://link.springer.com/article/10.1007/s00521-022-07469-x

如果按学习顺序建议，可以这样读：

1. 先看 SoM 官方页面，理解“编号截图 + 编号动作”的基本协议。
2. 再看 Multimodal-Mind2Web，理解网页 grounding 在学术任务里的定义方式。
3. 然后看 OpenAdapt 的 grounding 文档，理解工程实现如何组织候选、标注和执行。
4. 最后再回头看 XPath/CSS、坐标缩放、多框去重这些传统问题，它们会更容易放到正确位置上理解。
