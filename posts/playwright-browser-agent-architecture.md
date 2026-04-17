## 核心结论

Playwright 驱动的浏览器 Agent，核心不是“让大模型直接看网页”，而是先把网页压缩成**可操作的状态表示**，再把执行限制在一个小而明确的动作集内。这里最常见的状态表示是 Accessibility Tree，也可以简称 A11y Tree。白话说，它就是“浏览器整理给辅助技术的一份界面语义说明书”，重点保留元素的角色、名称、状态和层级关系，而不是把整页源码原样塞给模型。

这种架构的价值主要有两个。

第一，降低 token 成本。原始 DOM 是“网页源码树”，包含大量样式节点、容器节点、脚本注入节点和布局噪声。对“我要点击哪个按钮”这类任务来说，这些内容大多不是必要信息。Accessibility Tree 更接近“用户实际能理解和操作的界面”，通常只保留按钮、输入框、链接、标题、列表项等高价值节点。于是状态输入的体积，往往能从几千到几万个 DOM 节点，压缩到几百个甚至几十个语义节点。在很多表单、后台和内容站场景里，token 消耗可以下降 90% 以上。

第二，保证动作闭环。浏览器 Agent 不能只“理解页面”，还必须确认“动作是否真的生效”。因此可靠架构不是一次性读完整页面再连续执行十几步，而是按 `snapshot -> action -> resnapshot -> verify` 的闭环推进。白话说，就是“先看、再动、再核对”。如果没有这个闭环，Agent 很容易点击旧引用、往失焦输入框里打字，或者在页面已经跳转后还拿上一轮节点继续操作。

这个思路可以先用一个近似公式表示：

$$
TokenCost \approx \sum_{i=1}^{n}(node_i \times tokenPerNode_i)
$$

如果进一步把每个节点的平均 token 近似成常数 $\bar t$，就得到：

$$
TokenCost \approx NodeCount \times \bar t
$$

因此，对浏览器 Agent 来说，**减少状态节点数**通常比“优化提示词措辞”更直接有效。

如果把多轮交互也算进去，总成本更接近：

$$
TotalCost \approx \sum_{t=1}^{T} \left(|S_t| \cdot \bar t + |H_t|\right)
$$

其中：

- $S_t$ 是第 $t$ 轮送给模型的页面状态
- $|S_t|$ 是该状态包含的节点量
- $H_t$ 是该轮还保留的历史上下文

这说明一件很重要的工程事实：浏览器 Agent 的成本不是一次性成本，而是**每轮状态成本的累积**。所以要尽量把每一轮状态都压到“刚好够做下一步”。

下面这个表格展示三种常见状态表示的差别：

| 状态表示 | 主要内容 | 优点 | 主要问题 | 适合场景 |
| --- | --- | --- | --- | --- |
| 原始 DOM | 全量 HTML、属性、层级、脚本相关结构 | 信息最全 | 噪声大、token 高、交互语义弱 | 调试、精细解析、规则抽取 |
| 截图 | 像素级页面图像 | 接近人眼观察 | OCR 成本高、定位不稳定、滚动后坐标易漂移 | 视觉优先页面、canvas、游戏 UI |
| Accessibility Tree | role、name、state、层级关系 | 语义清晰、token 低、动作对齐 | 对无语义页面能力有限 | 大多数表单、后台、内容站 |

所以，Playwright Agent 的主流架构不是“拿全网页给模型”，而是“拿语义树给模型，再用有限动作执行并复核”。

---

## 问题定义与边界

这里要解决的问题很具体：如何让 Agent 在真实浏览器里持续、低成本、较稳定地完成任务。

这个问题有三个基本约束。

第一，页面状态会变化。弹窗会出现，按钮会禁用，点击后会跳转，异步请求会补出新节点，局部组件还可能在不刷新整页的情况下重渲染。也就是说，Agent 面对的不是一段静态文本，而是一个不断演化的状态机。状态机，白话说就是“系统会随着动作进入不同状态的一套规则”。

第二，模型上下文有限。即使上下文窗口在变大，浏览器自动化依然是多轮交互问题。你不仅要传当前页面状态，还可能要保留任务目标、历史动作、错误原因、恢复策略。如果每一轮都把完整 DOM、整页截图 OCR、历史日志全部送进去，成本会非常快地失控。

第三，动作必须可验证。网页自动化失败，很多时候不是因为模型不会推理，而是因为它操作的对象已经不是“当前页面上那个元素”。因此状态表示必须和动作接口对齐。模型看到的节点，必须能够映射成可以执行、可以复核、可以在下一轮重新确认的目标。

新手最容易忽略的一点是：浏览器 Agent 真正需要的，不是“页面的一切信息”，而是“完成当前任务所需的最小充分信息”。

比如在 GitHub 仓库页点击 `Star`，模型不需要知道页脚所有链接，也不需要知道推荐仓库卡片的完整结构，更不需要知道导航栏每个 SVG 图标的样式。它真正需要的是下面这类信息：

| 任务问题 | 最小充分信息 |
| --- | --- |
| 页面是否可继续执行 | 当前是否出现登录态、错误弹窗、加载中状态 |
| 目标按钮在哪里 | 是否存在 `role=button` 且名称接近 `Star` 的节点 |
| 这个按钮能否点击 | 当前是否可见、可用、未禁用 |
| 动作后是否成功 | 按钮是否变成 `Starred`，或 `pressed/selected` 状态是否变化 |

以“在 GitHub 页面点击 Star”为例，流程边界通常是这样的：

1. 先导航到仓库页。
2. 获取当前快照，确认页面上是否出现 `Sign in` 或 `Star`。
3. 如果未登录，点击 `Sign in`，登录后重新快照。
4. 再次查找 `Star` 按钮。
5. 点击后再次快照，确认状态从未加星变成已加星，或按钮状态发生变化。

这里的关键不是“记住 Star 在右上角”，而是“每一步都重新确认它是否存在、名称是否变化、引用是否已经过期”。

下面这个表格给出问题边界：

| 方案 | 模型看到什么 | 动作如何对齐 | 典型失效点 | 对新手的直白解释 |
| --- | --- | --- | --- | --- |
| 传统 DOM 抓取 | 全量源码结构 | 需额外推断选择器 | 元素太多、噪声太大 | 像把整栋楼的施工图交给你找门把手 |
| MCP 风格抽象 | 语义化树 + 明确动作 | 直接按 role、name、引用操作 | 树可能延迟，引用可能失效 | 像给你一张“当前可操作控件清单” |
| 纯截图 | 图片 + OCR 文字 | 依赖坐标或视觉框 | 文本识别误差、滚动后坐标漂移 | 像看照片后凭感觉去点位置 |

Accessibility Tree 也有明确边界。

它可能漏掉纯视觉元素，比如完全由 `canvas` 绘制的控件。它可能保留过多无效节点，比如广告 iframe、隐藏抽屉、重复导航。它还可能在异步更新后短暂滞后，导致上一轮拿到的节点引用在下一轮已经失效。

因此，一条很重要的工程规则是：

**只要发生导航、弹窗切换、表单提交、局部刷新、展开折叠菜单、登录态切换，就应重新执行 `browser_snapshot` 或等价的状态提取。**

这不是“谨慎一点”的建议，而是浏览器 Agent 能否稳定运行的基本同步纪律。

---

## 核心机制与推导

这个架构可以拆成两个部分：状态表示和动作闭环。

状态表示通常用 Accessibility Tree。它保留的字段不一定完全一致，但核心含义通常包括：

- `role`：元素角色。白话说，就是“它是什么类型的控件”，例如 `button`、`link`、`textbox`。
- `name`：可读名称。白话说，就是“用户一般会怎么叫它”，例如 `Sign in`、`Search`、`Add to cart`。
- `state`：状态信息。白话说，就是“它现在是不是可用、是否被选中、是否展开”。
- `children`：层级关系。白话说，就是“谁包着谁，谁属于谁”。

动作空间则故意设计得很小。常见动作只有：

- `navigate`
- `click`
- `type`
- `press`
- `scroll`
- `select`
- `hover`

动作空间，白话说，就是“Agent 被允许使用的操作集合”。

为什么动作集要小？因为浏览器自动化不是开放世界推理，而是受限控制问题。动作越多，模型自由度越大，但错误面也越大；动作越少，系统越容易做验证、恢复和审计。

下面是状态字段与动作之间的对应关系：

| 状态字段 | 为什么重要 | 典型动作 |
| --- | --- | --- |
| `role=button` | 表明它是可触发操作的控件 | `click` |
| `role=textbox` | 表明它是输入目标 | `type`、`press` |
| `name="Search"` | 用于区分同类元素 | `click`、`type` |
| `disabled=true` | 表明当前不可操作 | 阻止执行或等待 |
| `expanded=true/false` | 用于判断折叠菜单状态 | `click` 后复核 |
| `selected/pressed` | 用于判断切换类按钮是否生效 | `click` 后复核 |

整个系统的核心闭环如下：

1. 获取快照。
2. 从快照中筛选高价值节点。
3. 选择一个动作执行。
4. 重新获取快照。
5. 比较新旧状态，判断动作是否成功。
6. 若失败，重试、等待、回退或改计划。

可以写成简化流程：

```text
snapshot(S_t)
-> choose action(a_t)
-> execute(a_t)
-> snapshot(S_t+1)
-> verify(S_t, a_t, S_t+1)
-> continue or recover
```

这个闭环比“先规划十步再连续执行”更稳，因为浏览器环境是高噪声、强时序依赖的。一个弹窗、一次重定向、一个按钮文案变化，都可能让长链规划在中途失效。

### Token 成本推导

假设原始 DOM 有 $N_{dom}$ 个节点，每个节点平均展开后带来 $\bar t_{dom}$ 个 token；Accessibility Tree 有 $N_{a11y}$ 个节点，每个节点平均 $\bar t_{a11y}$ 个 token。那么：

$$
TokenCost_{dom} \approx N_{dom} \cdot \bar t_{dom}
$$

$$
TokenCost_{a11y} \approx N_{a11y} \cdot \bar t_{a11y}
$$

压缩比约为：

$$
CompressionRatio = 1 - \frac{TokenCost_{a11y}}{TokenCost_{dom}}
$$

如果原始 DOM 是 10,000 个节点，Accessibility Tree 是 600 个节点，即使每个 A11y 节点的描述更丰富，只要平均 token 没有放大十几倍，总成本仍会显著下降。

再进一步，很多系统不会把完整 Accessibility Tree 全部交给模型，而是做 Predicate Snapshot。Predicate，白话说，就是“筛选条件”。它的意思不是改变页面，而是用规则先做一轮任务化过滤。

常见筛选条件包括：

- 只保留可点击元素
- 只保留可输入元素
- 只保留当前视口附近元素
- 只保留名称非空元素
- 只保留与任务关键词相近的元素
- 只保留主内容区域元素，过滤页头页脚和广告区

这样，节点数可以从几百再压到几十。

下面给一个玩具量化例子。

假设一个页面有 1,200 个 DOM 节点，其中真正和任务相关的只有：

- 1 个搜索输入框
- 1 个搜索按钮
- 8 个结果链接
- 2 个分页按钮

如果把整个 DOM 都给模型，模型要在 1,200 个节点里找到 12 个关键节点；如果改成 A11y Tree，再加“只保留 `link/button/textbox`”的筛选，最后可能只剩 20 到 40 个节点。对模型来说，这就不是“在仓库里翻零件”，而是“直接看已经摆上工作台的零件盒”。

真实工程里，压缩往往更明显。大型站点完整语义树可能有数千到上万元素，而经过“仅保留高价值交互节点”的 Predicate Snapshot 后，可能只剩数十到数百个元素。这并不等于“重要信息被删除”，而是“与当前动作无关的信息暂时不进入模型上下文”。

下面这个表格概括三层表示：

| 表示层 | 节点规模 | 信息密度 | 任务相关性 | 常见用途 |
| --- | --- | --- | --- | --- |
| 原 DOM | 1,000 到 10,000+ | 低 | 低 | 全量解析、调试、规则抓取 |
| 完整 Accessibility Tree | 100 到 1,000+ | 中高 | 中高 | 通用 Agent 输入 |
| Predicate Snapshot | 20 到 200 | 很高 | 很高 | 单轮决策、低成本执行 |

很多系统还会显式为节点打分。一个简单的排序函数可以写成：

$$
Score(node) = w_1 \cdot clickable + w_2 \cdot keywordMatch + w_3 \cdot inViewport + w_4 \cdot freshness + w_5 \cdot visible
$$

然后只保留前 $k$ 个高分节点送给模型。

这里每一项都很好理解：

- `clickable`：是否能直接触发操作
- `keywordMatch`：名字是否和任务目标接近
- `inViewport`：是否在当前视口附近
- `freshness`：是否是最近刚出现的节点
- `visible`：是否对用户真实可见

因此，Playwright Agent 的关键设计不是“怎么把网页看得更全”，而是“怎么把下一步动作需要的状态压到刚刚够用”。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现。它不依赖真实浏览器，只模拟 `snapshot -> 查找目标 -> 执行动作 -> 重新快照验证` 的闭环。这个玩具例子的目的不是模拟完整浏览器，而是把架构里的几个关键约束拆开看清楚。

```python
from dataclasses import dataclass, replace
from typing import Iterable, Optional


@dataclass(frozen=True)
class Node:
    ref: str
    role: str
    name: str
    enabled: bool = True
    pressed: bool = False


def snapshot_initial() -> list[Node]:
    return [
        Node(ref="h1-repo", role="heading", name="octo/repo"),
        Node(ref="btn-star", role="button", name="Star", enabled=True, pressed=False),
        Node(ref="lnk-issues", role="link", name="Issues"),
    ]


def find_actionable(nodes: Iterable[Node], *, role: str, name: str) -> Optional[Node]:
    for node in nodes:
        if node.role == role and node.name == name and node.enabled:
            return node
    return None


def click(nodes: list[Node], *, ref: str) -> list[Node]:
    next_nodes: list[Node] = []
    for node in nodes:
        if node.ref == ref and node.role == "button" and node.name == "Star":
            # 点击后状态变了，同时 ref 也可能变化，旧引用不能继续复用
            next_nodes.append(
                Node(
                    ref="btn-starred",
                    role="button",
                    name="Starred",
                    enabled=True,
                    pressed=True,
                )
            )
        else:
            next_nodes.append(node)
    return next_nodes


def verify_star_transition(before: list[Node], after: list[Node]) -> bool:
    before_star = find_actionable(before, role="button", name="Star")
    after_starred = find_actionable(after, role="button", name="Starred")
    return before_star is not None and after_starred is not None and after_starred.pressed


def main() -> None:
    state_1 = snapshot_initial()

    star_button = find_actionable(state_1, role="button", name="Star")
    if star_button is None:
        raise RuntimeError("Star button not found")

    state_2 = click(state_1, ref=star_button.ref)

    if not verify_star_transition(state_1, state_2):
        raise RuntimeError("Click was issued, but state verification failed")

    print("OK: state changed from Star to Starred")


if __name__ == "__main__":
    main()
```

这段代码表达了三个工程事实：

1. 定位最好基于 `role + name`，而不是只靠一段模糊文本。
2. 动作执行后必须重新读取状态，而不是假设按钮一定已经变成目标状态。
3. 动作前后的节点引用可能变化，所以旧 `ref` 不应在新状态里继续复用。

如果你想看一个更接近真实工程、并且可以直接运行的 Playwright Python 示例，下面这段代码更有参考价值。它不依赖外部网站，而是在本地构造一个最小页面，演示三件事：

- 用 `aria_snapshot()` 观察页面语义树
- 用 `get_by_role()` 按角色和名称定位
- 点击后重新快照并验证状态变化

运行前需要先安装：

```bash
pip install playwright
playwright install chromium
```

示例代码：

```python
from playwright.sync_api import sync_playwright


HTML = """
<!DOCTYPE html>
<html lang="en">
  <body>
    <main>
      <h1>octo/repo</h1>
      <button id="star-btn" aria-pressed="false">Star</button>
      <script>
        const btn = document.getElementById("star-btn");
        btn.addEventListener("click", () => {
          btn.setAttribute("aria-pressed", "true");
          btn.textContent = "Starred";
        });
      </script>
    </main>
  </body>
</html>
"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(HTML)

        before = page.locator("body").aria_snapshot()
        print("=== BEFORE ===")
        print(before)

        star_button = page.get_by_role("button", name="Star")
        star_button.click()

        after = page.locator("body").aria_snapshot()
        print("=== AFTER ===")
        print(after)

        starred_button = page.get_by_role("button", name="Starred", pressed=True)
        assert starred_button.is_visible()

        browser.close()


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出的是 Playwright 提供的 ARIA Snapshot。你会看到点击前后，按钮的可访问名称和状态都发生了变化。这个例子虽然简单，但已经完整体现了浏览器 Agent 的主干流程：

| 步骤 | Playwright 操作 | 架构含义 |
| --- | --- | --- |
| 读状态 | `page.locator("body").aria_snapshot()` | 把当前页面压成语义树 |
| 找目标 | `page.get_by_role("button", name="Star")` | 用语义字段定位 |
| 执行动作 | `click()` | 只允许有限动作 |
| 再读状态 | 再次 `aria_snapshot()` | 不信任旧状态 |
| 验证成功 | `get_by_role("button", name="Starred", pressed=True)` | 根据新状态复核 |

如果换成真实浏览器 Agent，调用序列通常类似这样：

1. `browser_navigate(url)`
2. `browser_snapshot()` 或等价的 `ariaTree()` / `aria_snapshot()`
3. 解析节点，找到目标
4. `browser_click(...)` / `browser_type(...)` / `browser_scroll(...)`
5. 再次 `browser_snapshot()`
6. 验证动作是否生效
7. 继续下一步或进入恢复流程

下面给一个偏真实工程的伪代码，场景仍然是“进入 GitHub 仓库，必要时登录，再确认 Star 状态”。这里特意保留伪代码形式，因为真实 MCP 接口、鉴权方式、人工接管策略会因系统而异。

```python
def run_agent(browser, repo_url):
    browser.navigate(repo_url)

    snap = browser.snapshot()
    sign_in = snap.find(role="link", name="Sign in")
    star = snap.find(role="button", name="Star")

    if sign_in is not None and star is None:
        browser.click(ref=sign_in.ref)

        # 登录页通常不应把真实密码交给模型，适合人工接管
        browser.wait_for_user("Complete sign-in in the opened browser window")

        snap = browser.snapshot()

    star = snap.find(role="button", name="Star")
    if star is None:
        raise RuntimeError("Star button not found in current snapshot")

    browser.click(ref=star.ref)

    snap = browser.snapshot()
    starred = snap.find(role="button", name="Starred")

    if starred is None:
        raise RuntimeError("Star action did not take effect")
```

这里有几个实现细节，比代码表面更重要。

第一，解析 `ariaTree` 时应优先使用语义字段，而不是默认退回 CSS 选择器。因为语义字段直接对应用户界面含义，更适合给模型做推理。选择器可以作为兜底，但不应成为主路径。

第二，执行动作时尽量使用当前快照里的稳定目标，而不是跨轮缓存旧引用。原因很简单：只要页面重渲染，旧引用就可能不再属于当前文档。

第三，操作后必须强制快照。不要把“click 调用成功返回”当成“页面目标已经达成”。浏览器层面的点击成功，只说明事件发出去了，不说明按钮一定切换、请求一定返回、页面一定完成跳转。

可以把一个稳定版本的执行器抽象成下面这种结构：

```text
for step in plan:
    snapshot = get_snapshot()
    target = resolve_target(snapshot, step)
    if not target:
        repair_or_replan()

    execute(step, target)

    snapshot2 = get_snapshot()
    if not verify(step, snapshot, snapshot2):
        recover()
```

再进一步，如果任务是“搜索商品并加入购物车”，实际流程往往会细化成：

1. 导航到首页
2. 快照中找到搜索框
3. 输入商品关键词
4. 点击搜索
5. 等结果页稳定
6. 重新快照，只保留商品卡片、筛选按钮、价格标签
7. 选择目标商品
8. 打开详情页后再次快照
9. 点击 `Add to cart`
10. 再次快照确认购物车计数变化

这里每一步都围绕同一原则：**动作必须和最新状态绑定，验证必须基于新快照完成。**

---

## 工程权衡与常见坑

Accessibility Tree 不是银弹。它的优势是语义对齐和 token 节省，代价是你必须接受一套更严格的同步纪律和更明确的回退策略。

下面是常见坑和规避方式：

| 常见坑 | 表现 | 根因 | 规避策略 |
| --- | --- | --- | --- |
| 动作后树未及时更新 | 明明点了按钮，快照里仍是旧状态 | 异步渲染、动画、接口未返回 | 关键动作后重新 snapshot，必要时等待稳定信号 |
| 导航后引用失效 | `ref` 还在缓存里，但点击报错 | 页面已切换，旧节点不属于新文档 | 每次导航、刷新、提交后都重新取引用 |
| 大站无效节点太多 | 模型总盯着广告、页脚、隐藏层 | 树虽语义化，但仍有噪声 | 增加过滤和排名，只保留高价值节点 |
| 名称不稳定 | 同一按钮有时叫 `Submit`，有时叫 `Save` | 国际化、动态文案、A/B 测试 | role 优先，name 做模糊匹配并保留候选 |
| 纯视觉控件无法识别 | 树里找不到实际可点元素 | 页面缺语义标签，或使用 canvas 绘制 | 回退到截图、OCR、坐标或规则策略 |
| 过度依赖长链规划 | 连续执行数步后开始偏航 | 中间状态变化未被重新观测 | 使用短回路执行，缩短盲走长度 |

新手很容易踩的第一个坑是：以为“树已经比 DOM 干净很多了，所以把整棵树都给模型就够了”。这在简单页面上问题不大，但在大型门户、电商、内容平台上，Accessibility Tree 仍可能包含大量导航项、推荐项、广告区、页脚链接、隐藏抽屉、重复入口。如果不做过滤，模型依然会在大量“语义上合法但任务上无关”的节点中分心。

比如一个新闻站首页可能有 300 个 `link` 节点，但和当前任务真正相关的可能只有：

- 搜索框
- 登录入口
- 当前文章标题
- 下一页按钮

这时比较合理的做法不是“让模型自己在 300 个链接里选”，而是先增加一层任务化排名。可以按下面维度给节点打分：

- 是否可点击或可输入
- 是否在当前视口附近
- 是否名称非空
- 是否与任务关键词相似
- 是否最近刚出现
- 是否属于主内容区域而不是页脚、广告区或侧边栏

这种排名本质上是在做“结构化注意力分配”。不是让模型更聪明，而是让模型先少看无关对象。

第二个常见问题是“隐藏节点仍在树里”。有些广告 iframe、折叠区域、过渡动画中的节点，可能仍以某种形式出现在语义树中。如果不结合可见性、可交互性和布局位置做判断，Agent 可能会点击一个“理论存在、用户实际上碰不到”的元素。

所以很多系统在 `browser_snapshot` 之后还会再做一层清洗：

```text
raw aria tree
-> remove invisible / disabled / duplicated nodes
-> rank by action value
-> keep top-k
-> send to model
```

这一步不是“可选优化”，而是大型页面上稳定性的关键组成部分。

第三个容易被低估的问题是“验证条件写得太弱”。例如点击 `Submit` 后，只检查“页面没报错”是不够的。更稳的验证应该绑定任务结果，例如：

| 动作 | 弱验证 | 强验证 |
| --- | --- | --- |
| 点击 `Star` | 点击未报错 | 按钮变成 `Starred` 或 `pressed=true` |
| 提交登录表单 | 页面跳了一下 | 用户头像出现、`Sign in` 消失 |
| 加入购物车 | 按钮点击成功 | 购物车数字增加、成功提示出现 |
| 展开菜单 | 节点没报错 | `expanded=false -> true` |

强验证的价值在于：系统能分清“动作发出去了”和“任务真的完成了”是两回事。

---

## 替代方案与适用边界

Playwright + Accessibility Tree 是很强的默认方案，但不是唯一方案，也不是所有页面都适用。

最常见的三类策略如下：

| 策略 | 状态来源 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- | --- |
| 截图 + OCR / 视觉框 | 页面图像 | 接近人类视觉，可覆盖无语义页面 | token 高、定位漂移、依赖视觉模型 | 游戏 UI、canvas、大量图形控件 |
| 原始 DOM 抓取 | HTML、属性、脚本结构 | 信息最全，便于精细选择器和字段抽取 | 噪声大、语义弱、模型负担重 | 数据抽取、规则自动化 |
| A11y Tree + 有限动作 | role、name、state | token 低、动作对齐、适合多轮决策 | 遇到无语义页面会掉能力 | 表单、控制台、后台系统、常规网站 |

可以把它们理解成三种不同的感知方式：

- 截图方案像“用眼睛看页面”
- DOM 方案像“读页面内部源码”
- Accessibility Tree 方案像“读页面给辅助技术准备的交互提纲”

大多数工程场景里，第三种最平衡，因为它同时满足三件事：

1. 模型能理解页面在说什么
2. 系统能把理解结果映射成动作
3. 动作完成后还能重新验证状态

但下面几类页面要准备 fallback：

1. 游戏或强视觉交互页面。按钮画在 canvas 上，树里几乎没有语义信息。
2. 极端自定义组件库。外观存在，但语义标签缺失，`role/name` 不完整。
3. 需要像素级判断的任务。比如识别图表走势、拖动滑块、图像验证码。
4. 反自动化很强的页面。节点频繁重排、文本故意随机化、可访问属性不稳定。

这时更合理的策略通常不是“完全放弃 A11y Tree”，而是做分层回退：

1. 先尝试 A11y Tree。
2. 树里找不到高置信目标时，退回截图或 OCR。
3. 若视觉也不稳定，再使用规则选择器、专用脚本或人工介入。
4. 成功进入下一个语义明确页面后，再切回 A11y Tree 主路径。

下面这个表格可以帮助快速判断主路径应该选哪种：

| 页面特征 | 首选方案 | 原因 |
| --- | --- | --- |
| 标准表单、后台系统、常规网站 | A11y Tree | 语义清楚，动作容易对齐 |
| 大量自定义图形控件 | 截图 + 视觉 | 树里缺少可操作语义 |
| 批量抽取字段、规则固定 | DOM | 结构信息最全，便于规则化 |
| 混合型复杂站点 | A11y Tree 为主，截图 / DOM 为辅 | 能兼顾成本、稳定性和补盲能力 |

一个直白的判断标准是：

- 如果页面本身“对屏幕阅读器友好”，A11y Tree 往往也对 Agent 友好。
- 如果页面主要靠视觉呈现但缺少语义标注，Agent 就要更多借助截图和坐标。
- 如果任务目标是“批量抽字段而不是交互闭环”，原始 DOM 反而可能更划算。

所以，Playwright 浏览器 Agent 的最佳实践，不是押注单一感知方式，而是以 Accessibility Tree 为主干，用截图和 DOM 作为补充通道。

---

## 参考资料

- [Playwright MCP: Browser Automation](https://playwright.dev/agents/playwright-mcp-browser-automation)  
  Playwright 官方文档，说明 MCP 模式下浏览器自动化的基本流程，包含“导航、快照、点击、复核”的典型示例。
- [Playwright Python: Aria snapshots](https://playwright.dev/python/docs/aria-snapshots)  
  Playwright 官方文档，说明 `aria_snapshot()` 的输出形式，以及可访问树如何被序列化成 YAML。
- [Playwright Python: Locators](https://playwright.dev/python/docs/locators)  
  Playwright 官方文档，说明为什么优先使用 `get_by_role()`、`get_by_label()` 这类贴近用户感知的定位方式。
- [Playwright Python API: Locator.aria_snapshot](https://playwright.dev/python/docs/api/class-locator#locator-aria-snapshot)  
  Playwright 官方 API 文档，说明 `locator.aria_snapshot()` 的具体接口和返回值。
- [WAI-ARIA 1.2](https://www.w3.org/TR/wai-aria-1.2/)  
  W3C 标准文档，定义 role、state、property 等语义来源，是理解 Accessibility Tree 的基础规范。
- [MDN: ARIA](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA)  
  MDN 对 ARIA 的基础解释，适合补足“角色、可访问名称、状态”这些概念的背景知识。
- [Playwright Agents: Traces and Screenshots](https://playwright.dev/agents/traces-and-screenshots)  
  Playwright 官方文档，说明如何保存 trace 和截图，适合排查“动作发出去了但状态没更新”的问题。
- [Stagehand Agent System](https://deepwiki.com/browserbase/stagehand/5-agent-system)  
  对基于语义树的 Agent 系统做工程化拆解，可作为状态表示、动作选择和恢复逻辑的补充阅读。
- [Predicate Snapshot and OpenClaw Skill](https://predicatesystems.ai/blog/predicate-snapshot-openclaw-skill)  
  说明 Predicate Snapshot 的筛选思路，适合理解“为什么不是整棵树都送给模型”。
