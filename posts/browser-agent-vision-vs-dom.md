## 核心结论

浏览器 Agent 的“感知”不是一个抽象前置步骤，而是决定任务上限的核心模块。这里的感知，指模型在动作决策前能看到什么信息。只看截图的纯视觉方案，输入是像素；只读 DOM 的纯结构方案，输入是 HTML、Accessibility Tree 或元素属性；混合方案同时接入两类信号，再根据场景切换、交叉验证或做失败回退。

在同一任务集、同一执行器下，成功率差异往往可以直接解释为观测质量差异。常被引用的一组结果是：纯视觉基线约 14.41%，加入 DOM 搜索的 WebArena 文本路线可到 19.2%，更广义的混合浏览方案在 arXiv 版本中报告 35.8%；如果再把浏览器 UI 感知和后端 API 一起纳入，Narada Operator 在 2025 年 11 月 7 日的公开文章中报告 WebArena 64.16%。这些数值不说明“模型只要更大就行”，而说明“观测通道决定模型有没有机会做对”。

新手可以先记一个直观结论：只看截图，像人在远处看网页；只读 DOM，像直接翻网页结构树；混合模式，像一边看页面一边看开发者工具。前者更接近人类视觉，后者更接近程序定位，混合模式的价值来自互补，不来自简单叠加模型参数。

| 感知方案 | 典型输入 | 代表性成功率 | 强项 | 典型误差 |
|---|---|---:|---|---|
| 纯视觉 | 截图、像素、OCR 文本 | 14.41% 左右 | 动态渲染、Canvas、视觉布局、图标识别 | 遮挡误判、读不到隐藏结构、坐标点击漂移 |
| 纯 DOM | HTML、Accessibility Tree、元素属性 | 19.2% 左右 | 表单定位、表格抽取、结构化检索、重复流程 | 看不到真实渲染状态、难处理视觉歧义、可见性误判 |
| 混合模式 | 截图 + DOM/树搜索，或浏览 + API | 35.8% 到 64.16% | 冗余校验、失败回退、复杂网页流程、跨系统联动 | 同步复杂、成本更高、状态一致性难保证 |

一个玩具例子就能说明差异。网页上有两个“提交”按钮，一个是顶部灰色禁用按钮，一个是底部蓝色可点击按钮。纯视觉看到颜色，容易点对；纯 DOM 如果只按文本匹配，可能先找到第一个“提交”；混合模式会先发现“两个文本相同但状态不同”，再用可见性、`disabled`、命中区域或 Accessibility name 过滤，成功率自然更高。

还可以把这个差异写成更工程化的一句话：

$$
\text{可行动作质量} \le \text{观测质量} \times \text{决策质量}
$$

决策模型再强，如果观测里没有“这个按钮当前不可点”“这个文本被遮罩挡住”“这个元素在 Shadow DOM 里”这类关键信息，动作质量也上不去。

---

## 问题定义与边界

本文讨论的是浏览器 Agent 的观测层，而不是整个 Agent 的推理能力。浏览器 Agent 指能在网页中执行点击、输入、滚动、跳转的一类自动体。观测层指它在每一步动作前拿到的状态表示。执行器保持不变时，比较不同观测层，才能把结论收敛到“视觉感知和 DOM 感知谁更适合什么任务”。

衡量指标通常写成：

$$
SR=\frac{\text{完成的任务数}}{\text{总任务数}}
$$

其中 $SR$ 是任务成功率。分母一致很重要，因为只有在同一批任务、同一套评测规则、同一动作预算下，成功率提高才可以解释为感知质量提升，而不是任务难度下降。

如果把任务再拆细一点，可以得到更适合工程排查的分解式：

$$
SR \approx P(\text{看见目标}) \cdot P(\text{正确定位}\mid \text{看见}) \cdot P(\text{成功执行}\mid \text{已定位})
$$

这不是严格证明，而是一个很好用的诊断框架。  
如果失败主要发生在“看见目标”阶段，优先补感知。  
如果失败主要发生在“成功执行”阶段，优先修执行器、等待条件和回退逻辑。

边界也要说清楚。

第一，纯视觉不是“完全不知道网页结构”，而是主要依赖截图、OCR 和视觉问答能力，不把 DOM 当主输入。第二，纯 DOM 不是“完全不看页面”，而是把 HTML、无障碍树、节点属性作为主观测，不依赖截图理解布局。第三，混合模式也分层次：狭义是截图+DOM，广义还包括浏览器网络日志、接口调用结果、工具返回值等外部观测。文中 35.8% 这类更高结果，通常已经不只是“截图+DOM”这么窄；而 64.16% 这类结果更明确属于“浏览器 UI + API”的环境级混合。

新手可以用一个最简单的类比理解边界：

| 类比对象 | 对应方案 | 本质 |
|---|---|---|
| 截屏后人工找按钮 | 纯视觉 | 看的是结果图像 |
| 打开开发者工具搜元素 | 纯 DOM | 看的是结构和属性 |
| 一边看页面一边看结构树 | 混合模式 | 同时利用渲染和结构 |

真实工程里还有一个常被忽略的边界：很多页面并不把“可交互状态”完整暴露在单一通道里。例如一个按钮可能视觉上存在，但 DOM 里挂在 Shadow DOM 内部；也可能 DOM 中存在，但被 `opacity:0`、`pointer-events:none`、遮罩层或滚动容器裁切影响，用户根本点不到。只比较“能不能找到节点”是不够的，还要比较“能不能基于当前页面真实状态完成动作”。

再补两个容易混淆的术语。

| 术语 | 含义 | 为什么重要 |
|---|---|---|
| DOM | 浏览器内部的文档对象模型树 | 适合做节点检索、属性过滤、表单读写 |
| Accessibility Tree | 为辅助技术暴露的语义树 | 常比原始 DOM 更接近“用户可操作对象” |
| Screenshot | 浏览器当前视口的渲染结果 | 适合判断颜色、遮挡、布局、图标、Canvas |
| Viewport | 当前可见区域 | 决定截图能看到什么，也影响坐标点击 |
| Hit Test | 点击命中测试 | 决定一次点击最终打到哪个元素 |

这些术语如果不区分，就容易把“找到了节点”和“真实能操作”混成一回事。

---

## 核心机制与推导

纯视觉方案的优势在于它直接读取渲染结果。渲染结果就是浏览器最终画出来的页面外观，包含颜色、位置、遮挡关系、图标、截图中的文本和图形控件。对于富交互页面，这类信息往往比原始 DOM 更接近用户真实看到的内容。例如图表、Canvas 按钮、拖拽面板、样式驱动的可见性变化，视觉方案更容易对齐。

但纯视觉的代价也直接。它的定位通常依赖“看图后描述目标区域”，再映射回坐标或可点击区域。这一步很容易受分辨率、滚动位置、弹窗遮挡、响应式布局影响。一个字读错、一个坐标偏几像素，动作就失败。视觉路线的典型误差不是“完全没看见”，而是“看见了，但坐错地方”。

纯 DOM 方案走的是另一条路。DOM 可以理解为网页在浏览器内部的一棵结构树。Accessibility Tree 是为屏幕阅读器等辅助工具准备的语义树，往往比原始 DOM 更贴近“哪个元素可被用户操作”。这类方案的好处是元素有稳定标识：标签名、属性、层级关系、可访问名称、输入框类型、表格行列信息都可以精确抽取。所以在表单填写、订单列表抽取、按钮定位这类任务上，DOM 往往更可靠。

问题在于 DOM 不等于当前画面。一个节点存在于树中，不代表它可见、可点、在当前视口内，甚至不代表它被用户理解为主按钮。于是 DOM 方案容易出现“结构正确、交互错误”的问题。最典型的症状是：

| 现象 | DOM 看起来怎样 | 用户实际上怎样 |
|---|---|---|
| 按钮被遮罩盖住 | 节点存在，可查询 | 点不到 |
| 选项被折叠在下拉框里 | 节点存在，可枚举 | 没展开前不可点 |
| 文本被 CSS 截断 | DOM 里是完整文本 | 页面上只显示省略号 |
| 内容在 iframe / Shadow DOM | 主树里不明显 | 用户能看到，但检索路径变复杂 |

混合模式的核心不是把两份信息都喂给模型就结束，而是建立一种互补流程。最简单的抽象可以写成：

$$
SR_{hybrid}\approx \alpha \cdot SR_{\text{visual}} + (1-\alpha)\cdot SR_{\text{DOM}} + \Delta
$$

其中 $\alpha$ 表示当前任务对视觉信号的依赖程度，$\Delta$ 表示互补增益。这里的互补增益不是数学上自动出现的项，而是来自两个机制：

1. 视觉确认“页面真实长什么样”。
2. DOM 确认“应该精确操作哪个节点”。

如果没有这两个机制，只把两种输入简单拼接，$\Delta$ 可能接近 0，甚至因为噪声增多而变成负值。

更实用的写法是把互补增益展开为几个条件项：

$$
\Delta \approx G_{\text{disambiguation}} + G_{\text{fallback}} + G_{\text{state-check}}
$$

其中：

- $G_{\text{disambiguation}}$：同名元素、重复按钮、相似图标之间的消歧收益。
- $G_{\text{fallback}}$：主通道失败后，副通道接管带来的回退收益。
- $G_{\text{state-check}}$：动作前后做状态核验，避免“点了但页面没变”的收益。

看一个玩具例子。页面上有一个商品筛选器，显示“价格从低到高”。视觉模型能看出当前排序箭头朝上，但不一定知道对应节点是哪一个；DOM 模型能找到 `select[name=sort]`，却不一定知道当前页面是否已展开下拉框。混合模式可以先从视觉判断“当前 UI 状态”，再从 DOM 精确定位选项节点，最后执行点击或选择。这个流程比“纯看图乱点”或“纯按结构盲找”都稳。

再看一个真实工程例子。某些企业后台把“导出 Excel”做成表格右上角一个下载图标，没有文字标签；点击后还会异步弹出二次确认框。纯 DOM 可能只能看到一个无语义的 `button` 或 `svg` 容器，难判断它是不是导出；纯视觉能看到下载图标和弹窗，但可能抓不准确认框里的主按钮。混合模式通常这样做：

1. 先根据视觉识别“这是下载图标”。
2. 再用 DOM 找到邻近区域内唯一可点击按钮。
3. 点击后等待 DOM 或网络状态变化。
4. 再用视觉确认弹窗真的出现。
5. 最后用 DOM 定位“确认导出”。

这就是冗余感知的工程价值。它不是把系统做复杂，而是把错误从“无声失败”改成“可检测、可回退的失败”。

---

## 代码实现

实现上最重要的不是模型多复杂，而是把“观测”和“执行”彻底拆开。执行器只负责统一动作接口，比如 `click`、`type`、`scroll`；观测器分别产出视觉状态和 DOM 状态。这样纯视觉、纯 DOM、混合模式只是换观测模块，不需要重写底层浏览器控制代码。

下面先给一个可运行的 Python 玩具实现。它不控制真实浏览器，只模拟三种感知策略如何在同一任务集上做决策，并演示 fallback 逻辑。代码可以直接运行。

```python
from dataclasses import dataclass
from typing import List


@dataclass
class PageState:
    name: str
    visual_visible: bool
    dom_present: bool
    dom_clickable: bool
    target: str


def observe_visual(page: PageState) -> dict:
    return {
        "target_seen": page.visual_visible,
        "guess": page.target if page.visual_visible else None,
    }


def observe_dom(page: PageState) -> dict:
    return {
        "node_found": page.dom_present,
        "clickable": page.dom_clickable,
        "selector": f"[data-name='{page.target}']" if page.dom_present else None,
    }


def decide_visual(obs: dict) -> bool:
    # 纯视觉：只要能在截图里确认目标存在，就允许坐标点击
    return bool(obs["target_seen"])


def decide_dom(obs: dict) -> bool:
    # 纯 DOM：必须同时满足“找到节点”与“节点可点”
    return bool(obs["node_found"] and obs["clickable"])


def decide_hybrid(page: PageState, alpha: float = 0.5) -> bool:
    """
    alpha 越高，越偏向先用视觉确认页面状态。
    这不是概率模型，只是演示“先确认，再精点，再回退”的控制流。
    """
    visual_obs = observe_visual(page)

    # 路径 1：视觉确认页面确实有目标，再走 DOM 精确点击
    if alpha >= 0.5 and decide_visual(visual_obs):
        dom_obs = observe_dom(page)
        if decide_dom(dom_obs):
            return True

    # 路径 2：视觉不确定时，直接尝试 DOM 路线
    dom_obs = observe_dom(page)
    if decide_dom(dom_obs):
        return True

    # 路径 3：DOM 不可用，但视觉明确看到了目标，允许坐标点击兜底
    return decide_visual(visual_obs)


def success_rate(results: List[bool]) -> float:
    return sum(results) / len(results)


def main() -> None:
    pages = [
        PageState("both_ok", True, True, True, "submit"),
        PageState("visual_only", True, False, False, "chart"),
        PageState("dom_only", False, True, True, "export"),
        PageState("ghost", False, True, False, "ghost"),
        PageState("visible_but_disabled", True, True, False, "disabled_submit"),
    ]

    visual_results = [decide_visual(observe_visual(page)) for page in pages]
    dom_results = [decide_dom(observe_dom(page)) for page in pages]
    hybrid_results = [decide_hybrid(page, alpha=0.7) for page in pages]

    assert visual_results == [True, True, False, False, True]
    assert dom_results == [True, False, True, False, False]
    assert hybrid_results == [True, True, True, False, True]

    print("visual results =", visual_results)
    print("dom results    =", dom_results)
    print("hybrid results =", hybrid_results)
    print(f"visual SR = {success_rate(visual_results):.2%}")
    print(f"dom SR    = {success_rate(dom_results):.2%}")
    print(f"hybrid SR = {success_rate(hybrid_results):.2%}")


if __name__ == "__main__":
    main()
```

这段代码表达了四个工程原则。

第一，观测模块独立。`observe_visual()` 只负责从视觉侧返回“看到了什么”，`observe_dom()` 只负责返回“结构上能否定位和点击”。

第二，混合模式不是平均投票，而是有顺序的控制流。先视觉确认，再 DOM 精点；DOM 失败时，再考虑视觉兜底。

第三，评估统一。三种方案都在同一组 `PageState` 上运行，因此结果可比。

第四，回退逻辑是显式写出来的，而不是隐含在 prompt 里。真正稳定的 Agent 系统，不能把所有恢复能力都寄托给一次自然语言推理。

如果换成更接近真实浏览器的最小可运行示意，结构通常会是下面这样。这里仍然是“框架代码”，但接口是工程上常见的拆法。

```python
from dataclasses import dataclass


@dataclass
class VisualObs:
    target_confidence: float
    bbox: tuple[int, int, int, int] | None


@dataclass
class DOMObs:
    selector_confidence: float
    selector: str | None
    clickable: bool


class Executor:
    def click_selector(self, selector: str) -> None:
        print(f"[executor] click selector: {selector}")

    def click_bbox(self, bbox: tuple[int, int, int, int]) -> None:
        print(f"[executor] click bbox: {bbox}")

    def scroll_down(self) -> None:
        print("[executor] scroll down")


class HybridPolicy:
    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha

    def act(self, visual_obs: VisualObs, dom_obs: DOMObs) -> tuple[str, str | tuple[int, int, int, int]]:
        if (
            self.alpha >= 0.5
            and visual_obs.target_confidence > 0.8
            and dom_obs.selector_confidence > 0.7
            and dom_obs.clickable
            and dom_obs.selector is not None
        ):
            return ("click_selector", dom_obs.selector)

        if dom_obs.selector_confidence > 0.85 and dom_obs.clickable and dom_obs.selector is not None:
            return ("click_selector", dom_obs.selector)

        if visual_obs.target_confidence > 0.9 and visual_obs.bbox is not None:
            return ("click_bbox", visual_obs.bbox)

        return ("scroll_down", "down")


def main() -> None:
    executor = Executor()
    policy = HybridPolicy(alpha=0.6)

    visual_obs = VisualObs(target_confidence=0.92, bbox=(120, 300, 210, 340))
    dom_obs = DOMObs(selector_confidence=0.88, selector="button[data-name='submit']", clickable=True)

    action, target = policy.act(visual_obs, dom_obs)

    if action == "click_selector":
        executor.click_selector(target)  # type: ignore[arg-type]
    elif action == "click_bbox":
        executor.click_bbox(target)  # type: ignore[arg-type]
    else:
        executor.scroll_down()


if __name__ == "__main__":
    main()
```

如果再向前走一步，真实浏览器实现通常会补上三个模块：

| 模块 | 作用 | 为什么不能省 |
|---|---|---|
| 稳定性等待器 | 等待 DOM、网络、动画或布局稳定 | 否则同一动作前后的观测会错位 |
| 状态核验器 | 点击后确认页面真的发生了预期变化 | 否则会把“点击无效”误判为“任务完成” |
| 失败恢复器 | 超时后重试、换通道、退回上一步 | 否则单次误判会终止整条任务链 |

真实系统里，`alpha` 往往不是固定常数，而是由任务类型动态调整。例如表格抽取任务偏 DOM，视觉问答或图形控件任务偏视觉。更成熟的做法不是直接学习一个全局权重，而是按步骤判断“当前需要的是语义结构、可见状态，还是空间布局”。

---

## 工程权衡与常见坑

工程里最常见的误区，是把“元素存在”误当成“元素可用”。这会同时坑到视觉和 DOM 两侧。

| 常见坑 | 现象 | 哪种方案更容易踩坑 | 推荐规避策略 |
|---|---|---|---|
| `opacity:0` 或遮罩覆盖 | DOM 有节点，但用户点不到 | 纯 DOM | 点击前校验可见性、层级和命中测试 |
| `aria-hidden` 或无障碍树缺失 | 视觉看得到，AX Tree 不完整 | 纯 DOM | 同时保留原始 DOM、AX Tree 和截图 |
| Shadow DOM | 结构被封装，普通检索找不到 | 纯 DOM | 使用浏览器原生遍历接口或宿主映射 |
| Canvas 控件 | 页面上有按钮，但 DOM 没语义 | 纯 DOM | 依赖视觉检测或注入额外交互标注 |
| 异步渲染 | 刚加载时没有内容，随后才出现 | 两者都会 | 动作后等待稳定条件，而不是固定 `sleep` |
| 响应式布局 | 同一元素在移动端和桌面端位置不同 | 纯视觉 | 绑定视口信息与坐标归一化 |
| 长页面滚动 | 截图只覆盖当前视口 | 纯视觉 | 多视口拼接或边滚动边观测 |
| 文本重复 | 多个“提交”“确定”同名按钮 | 两者都会 | 联合使用邻域、状态和可见性约束 |
| iframe 嵌套 | 主页面能看见，检索上下文错了 | 纯 DOM | 维护 frame 上下文，操作前切换作用域 |
| 动画过渡 | 截图与 DOM 都正确，但时机不对 | 两者都会 | 观测与执行绑定同一稳定时刻 |

ST-WebAgentBench 一类数据集专门放大这些坑。它把“视觉能看见但 AX Tree 看不见”的情况，以及“DOM 有内容但视觉上读不到”的情况做成专门任务。例如 `aria-hidden`、Canvas、CSS 重排更偏向视觉优势；`opacity:0`、极低对比度、被遮罩遮挡、Shadow DOM/iframe 等更偏向 DOM 优势。这提示的是一个很实际的结论：如果训练和评估阶段不做多模态对齐，模型会在某类网页上系统性失效，而不是偶然失误。

真实工程里还有两个代价常被低估。

第一是同步成本。你必须保证“截图对应的页面状态”和“DOM 快照对应的页面状态”来自同一时刻，否则视觉看到的是弹窗前，DOM 读到的是弹窗后，模型会得到互相冲突的观测。

第二是资源成本。视觉输入耗 token、耗图像推理；DOM 输入耗解析、耗截断策略；混合模式同时做两套，时延和费用都会上升。

可以把这件事写成一个很朴素的成本函数：

$$
C_{\text{hybrid}} \approx C_{\text{vision}} + C_{\text{dom}} + C_{\text{sync}} + C_{\text{recovery}}
$$

其中前两项是基础观测成本，后两项是很多系统上线后才暴露出来的隐性成本。  
`C_sync` 是双通道对齐开销。  
`C_recovery` 是失败检测、重试和回退带来的额外动作。

所以工程上更合理的做法通常不是“全程双通道最高配”，而是“默认单通道，遇到不确定性时升级为混合”。例如结构化录单任务默认走 DOM；检测到 Canvas、图标按钮、复杂弹窗、异常遮挡或低置信度匹配时，再补一张截图做视觉确认。

新手如果要排查一个 Agent 为什么总失败，可以按下面顺序看：

1. 它到底没看到目标，还是看到了但点错了。
2. 它拿到的是 DOM，还是 AX Tree，还是截图。
3. 点击前有没有做可见性和命中测试。
4. 点击后有没有等页面稳定并核验状态变化。
5. 不确定时有没有切换感知通道。

多数线上失败，不是“模型智力不够”，而是这五步里少了一步。

---

## 替代方案与适用边界

纯视觉、纯 DOM、混合模式并不存在绝对优劣，只有任务匹配度不同。

| 方案 | 更适合的任务 | 典型成功率 | 资源成本 | 适用边界 |
|---|---|---:|---:|---|
| 纯视觉 | 图形界面、布局理解、Canvas、强视觉依赖流程 | 14.41% 左右 | 中到高 | 页面真实外观比结构更重要时 |
| 纯 DOM | 表单填写、表格抽取、后台管理、重复结构页面 | 19.2% 左右 | 低到中 | 元素语义清晰、结构稳定时 |
| 混合浏览 | 复杂网站、多步流程、视觉与结构都不稳定 | 35.8% 左右 | 高 | 追求鲁棒性高于成本时 |
| 浏览+API 混合 | 企业自动化、跨系统操作、长链路任务 | 64.16% | 很高 | 可调用后端能力、要求强可靠性时 |

纯视觉的边界很明确。它适合回答“这个页面长什么样”“哪个图标像下载按钮”“图表上哪个区域被选中”这类问题，但不适合长期依赖文本重复、隐藏节点、结构筛选的任务。只要页面上存在大量同名按钮、隐藏字段、表格列映射，纯视觉就会迅速变脆。

纯 DOM 的边界也很明确。它非常适合结构化任务，尤其是表格抽取、表单填写、后台录入。因为 DOM 天然有层次、属性和值，抽取准确率高，且动作可重复。但遇到自绘控件、富前端框架、强视觉歧义页面时，DOM 只能告诉你“结构里有什么”，不能告诉你“用户真正看到了什么”。

混合模式适合生产环境，但也只在两个前提下值得上。第一，任务失败成本高；第二，系统能承受更高时延与更复杂的状态管理。更进一步的“浏览器 UI + API”方案，本质上已经不是单纯页面感知，而是“任务环境感知”。这在复杂流程自动化里很有效，因为很多操作并不需要真的点网页，只要先确认状态，再直接调接口，通常更稳、更快。

这里要补一个版本说明。`Beyond Browsing: API-Based Web Agents` 的 arXiv 版本常被引用为 35.8%，而 ACL Anthology 上的 2025 版摘要写到 38.9%。两者不是任务定义突然变化，更可能是论文版本更新、实验配置收敛或最终稿结果修订。写作时如果引用数值，最好写清楚“引用的是哪个版本”。

如果给零基础读者一个选择建议，可以这样记：

1. 先做表单和后台操作，优先学 DOM 感知。
2. 需要处理图形界面、Canvas、复杂布局，再加视觉感知。
3. 真正上线做关键流程自动化，再考虑混合模式和 API 联动。

如果再压缩成一句工程判断：

| 你面对的问题 | 默认优先方案 |
|---|---|
| “这个字段在哪一列、这一行的值是什么” | 纯 DOM |
| “这个图标是不是下载、这个区域是不是被选中” | 纯视觉 |
| “这个按钮到底能不能点、点完后页面到底变没变” | 混合模式 |
| “这个动作有没有必要真的点网页” | 浏览+API 混合 |

---

## 参考资料

- WebArena 项目与论文：给出统一的浏览器代理评测环境、任务设计和基线，适合理解“同任务集下比较不同观测通道”的意义。初始论文与公开站点常被用作 14% 左右浏览基线的来源。  
  https://webarena.dev/  
  https://arxiv.org/abs/2307.13854

- Tree Search for Language Model Agents：在 WebArena 上报告把基线 GPT-4o 路线从 15.0% 提升到 19.2%，说明结构搜索和更显式的状态探索能显著增强 DOM/文本路线。适合理解 19.2% 这类结果背后的“搜索+结构”思路。  
  https://arxiv.org/abs/2407.01476  
  https://jykoh.com/search-agents

- Beyond Browsing: API-Based Web Agents：说明“混合”不一定只指截图+DOM，也可以是浏览器 UI + API 的联合执行。需要注意版本差异：arXiv 版本常被引用为 35.8%，ACL 2025 页面摘要写为 38.9%。写作时要标明引用版本。  
  https://arxiv.org/abs/2410.16464  
  https://aclanthology.org/2025.findings-acl.577/

- Narada Operator 介绍：2025 年 11 月 7 日的公开文章报告 WebArena 64.16%，强调 UI 感知和 API 联动在复杂自动化中的价值。它更接近工程产品成绩，不等同于只比较“截图 vs DOM”的学术基线。  
  https://narada.ai/blog/narada-ai-web-agent-operator

- ST-WebAgentBench：专门覆盖隐藏元素、对抗性前端设计、异步渲染、`aria-hidden`、Canvas、Shadow DOM/iframe 等坑，适合理解为什么单一通道在真实网页中会系统性失效。  
  https://huggingface.co/datasets/ST-WebAgentBench/st-webagentbench

- WebArena 中文解读文章：适合快速建立背景，理解 4 个自托管网站、长序列任务和观测空间设计。  
  https://jishuzhan.net/article/2005834506310828033

- 如果需要把 DOM、无障碍树与真实浏览状态的关系再补得更扎实，可以继续看浏览器标准和开发者文档，尤其是 Shadow DOM、可访问树、点击命中测试、iframe 上下文切换这几类主题。它们不是“额外知识”，而是浏览器 Agent 感知设计里的基础约束。
