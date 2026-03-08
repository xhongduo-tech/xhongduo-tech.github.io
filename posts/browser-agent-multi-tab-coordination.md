## 核心结论

浏览器 Agent 的多标签页协同控制，关键不是“能不能同时打开很多页面”，而是“能不能在有限 token 预算内维持正确状态”。

先把 token 说清楚。这里的 token，不是浏览器资源，也不是网络流量，而是模型读取输入时消耗的上下文计量单位。截图会占 token，DOM 树会占 token，可交互元素列表、工具 schema、历史步骤摘要，同样会占 token。对浏览器 Agent 来说，页面越多、切换越频繁、输入越冗余，模型就越容易把预算浪费在“重新理解旧页面”上，而不是推进任务本身。

多标签任务一旦进入竞品对比、跨站表单填写、邮箱验证码确认这类流程，真正的瓶颈通常不是浏览器渲染速度，也不是点击速度，而是上下文反复重建的成本。模型不是天然“记得所有标签页”。每次切回旧页面，如果没有可复用的状态，它就要重新读取截图、重新解析 DOM、重新判断当前页面是什么、能做什么、下一步该做什么。

一次标签切换的感知成本，可以先用一个简单公式估算：

$$
T_{\text{switch}} = S_{\text{shot}} + D_{\text{dom}}
$$

其中：

- $S_{\text{shot}}$ 表示截图输入成本
- $D_{\text{dom}}$ 表示 DOM 或可交互元素提取成本

如果以常见网页为例：

| 输入类型 | 常见范围 |
| --- | --- |
| 1080p 页面截图 | 1200 到 1800 token |
| DOM 或可交互元素摘要 | 300 到 500 token |

那么一次完整的“切过去并重新看懂页面”，大致就是：

$$
T_{\text{switch}} \approx 1500 \sim 2300
$$

这只是一次切换的成本。假设一个三标签任务要经历多次来回切换，总代价会很快累计。若有 $k$ 次完整重看，则大致有：

$$
T_{\text{rebuild}} \approx k \cdot T_{\text{switch}}
$$

例如：

- 切换 3 次，可能消耗约 4500 到 6900 token
- 切换 5 次，可能消耗约 7500 到 11500 token

这还没有算上系统提示词、工具说明、任务历史、用户目标、输出计划等其他上下文。

因此，多标签 Agent 至少要同时管理三件事：

| 对象 | 要解决的问题 | 如果没管好会怎样 |
| --- | --- | --- |
| 焦点顺序 | 现在真正操作哪个标签 | 容易切错页、读错页、写错页 |
| 状态缓存 | 切回旧标签时是否需要完整重看 | token 反复浪费在旧页面上 |
| 令牌预算 | 什么时候允许高保真观察 | 上下文窗口被无关页面占满 |

工程上最有效的模式，通常不是“所有标签都实时保活”，而是：

$$
\text{1 个前台高保真标签} + \text{若干后台摘要标签} + \text{明确失效规则}
$$

这里的“高保真”指保留较完整的最新状态，比如截图、可交互元素树、关键 DOM 片段；“后台摘要”指只保留压缩后的结构化信息，比如页面用途、提取出的字段、下次可能要做的动作。

一个新手更容易理解的玩具例子是：Agent 同时打开“竞品 A”“竞品 B”“内部录入表单”三个标签页。

流程可能是：

1. 去 A 页读取价格和套餐结构。
2. 切到 B 页读取同类字段。
3. 切回 A 页复核一个功能差异。
4. 切到内部表单页填写对比结果。

如果每次切换都完整截图并重新抽取 DOM，假设每次只花 1500 token，那么 3 次切换就要 4500 token；如果页面更复杂，接近 2000 token 一次，那么 3 次切换就是 6000 token。相反，如果切走时为每个标签保留结构化摘要，切回时只补 delta，也就是只补充变化部分，那么成本通常会明显下降。

所以，浏览器 Agent 的多标签协同，本质上不是“同时开很多页”的问题，而是“如何避免模型在重复看旧页面上白白烧掉上下文”。

---

## 问题定义与边界

先定义问题。

多标签页协同控制，是指 Agent 在一个任务中需要跨多个浏览器标签页完成读信息、比信息、写信息，并且这些标签之间存在顺序依赖。所谓顺序依赖，就是后一步必须基于前一步得到的状态或结果，不能随意交换顺序。

例如：

- 先从页面 A 读取价格
- 再从页面 B 读取价格
- 最后把比较结果写入页面 C 的表单

这里的 A、B、C 不是独立任务，而是同一个任务链条上的不同状态来源或执行位置。

这类问题通常出现在三种典型场景中：

| 场景 | 标签页角色 | 为什么会复杂 |
| --- | --- | --- |
| 信息对比 | 多个竞品页 + 一个汇总页 | 要反复回看相同字段，容易重复感知同一页面 |
| 跨站表单填写 | 数据来源页 + 目标系统表单页 | 一个标签负责读，一个标签负责写，读写切换频繁 |
| 多步验证 | 登录页 + MFA 页 + 邮箱或短信页 | 页面状态会快速变化，旧快照很快失效 |

为了避免讨论范围过大，还需要明确几个边界。

第一，本文讨论的是“基于页面感知的浏览器 Agent”。

也就是说，模型需要通过截图、DOM、可交互元素树、表单结构、视觉提示等信息来理解页面当前状态，然后决定下一步动作。它不是传统脚本自动化那种“提前写死选择器和步骤”的方案。传统脚本当然也能做多标签控制，但那主要是脚本调度问题，不是上下文预算问题。

第二，本文不讨论“标签页越多越强”这种误解。

并不是所有多标签任务都值得做成“并行活跃”。所谓并行活跃，是指多个标签都持续保持高保真状态，随时可供模型继续推理。这件事很贵，因为每个活跃标签都可能需要：

- 最新截图
- 最新 DOM
- 最新可交互元素列表
- 最新页面摘要
- 最新动作历史

很多任务其实只需要：

- 1 个当前高优先级标签
- 若干冻结摘要标签
- 少数短周期刷新的动态标签

这里“冻结摘要”并不是丢失页面，而是把页面压缩成一份足够支持后续决策的中间状态。

第三，真实工程中的页面不是静止的。

这是新手最容易忽略的一点。页面状态不只会因为 Agent 的点击而变化，也会因为外部事件变化。比如：

- 邮件验证码几秒后才到
- 提交表单后页面自动跳转
- 列表页轮询刷新
- 登录态过期后页面自动回到登录页
- 弹窗在定时器触发后出现或消失

如果没有失效机制，Agent 就会把旧页面摘要当作新状态继续使用。这样做看起来“省 token”，实际是在制造错误决策。

因此，多标签问题真正的定义不是：

“如何让 Agent 同时开多个页？”

而是：

“在多个相互依赖、可能变化的页面之间，如何以尽可能低的上下文成本保持足够正确的任务状态？”

这个定义决定了后面的设计重点：不是盲目保留所有页面，而是判断哪些状态必须实时看，哪些状态可以压缩记忆，哪些状态已经不该再信。

---

## 核心机制与推导

多标签协同控制，可以拆成四个核心机制：焦点栈、状态快照、分层缓存、失效规则。

先看焦点栈。

焦点栈不是浏览器本身的 tab 顺序，而是 Agent 的工作顺序。它回答的问题是：当前最可能需要继续操作的是哪个标签，最近刚用过的是哪些标签，哪些标签已经可以降级或关闭。

例如，一个任务的真实工作顺序可能是：

$$
A \rightarrow B \rightarrow A \rightarrow C
$$

那么 Agent 的焦点优先级应该接近这个顺序，而不是简单沿用浏览器界面左到右的标签排列。这样做的原因很直接：最近刚操作过的标签，更可能马上再次访问，优先缓存它们收益最高。

再看状态快照。

状态快照，是某个标签在某一时刻的压缩表示。它不是完整页面副本，而是足以支撑后续判断的一组关键字段。最少通常要有下面这些内容：

| 字段 | 含义 | 作用 |
| --- | --- | --- |
| `url` | 当前页面地址 | 判断是否跳页或重定向 |
| `title` | 页面标题或路由名 | 快速识别页面身份 |
| `view_hash` | 关键视图指纹 | 判断页面是否仍是同一状态 |
| `summary` | 结构化摘要 | 切回时减少整页重看成本 |
| `dirty` | 是否可能过期 | 决定是否必须重新感知 |
| `ttl` | 可复用时长或步数 | 控制摘要能保留多久 |

这里的 `view_hash` 可以理解为一个“轻量身份指纹”。它不要求逐像素精确，也不要求完全代表整页，只要能帮助系统快速判断“这个页面还是不是刚才那页的有效状态”即可。

然后是分层缓存。

这是很多实现会出问题的地方。缓存不能只有一层，至少要区分两层：

| 缓存层 | 保存什么 | 生命周期 | 典型失效原因 |
| --- | --- | --- | --- |
| workflow cache | 任务语义结果 | 跟随任务阶段 | 任务进入下一阶段、业务字段被覆盖 |
| tool cache | 原子感知结果 | 通常更短 | 页面跳转、DOM 变化、截图过时 |

可以把它理解成两个问题：

- workflow cache 回答“我们已经知道什么”
- tool cache 回答“我们上次怎么看到的”

例如：

- “竞品 A 的价格是 299 元/月”属于 workflow cache
- “上次抽出来的按钮树包含 `提交申请` 按钮”属于 tool cache

这两者不能混用。前者更接近业务事实，后者更接近观测痕迹。一个页面跳转后，按钮树可能立刻失效，但已经提取出的价格结论未必失效。

最后是失效规则。

失效规则决定缓存什么时候还能信，什么时候不能信。至少要同时处理两类边界：

| 失效类型 | 含义 | 典型触发方式 |
| --- | --- | --- |
| TTL 失效 | 时间或步数过长 | 太久没访问、步骤推进过多 |
| 事件失效 | 页面发生关键动作 | 点击提交、跳转、轮询刷新、外部验证码到达 |

可以把它们简化成一句话：

- TTL 解决“太久没看”
- dirty 解决“刚发生了变化”

在这个框架下，可以做一个简化推导。

如果任务里有 $n$ 个标签页，且每个标签第一次激活都要做完整观察，那么初始成本大致是：

$$
T_{\text{init}} = \sum_{i=1}^{n}(S_{\text{shot},i} + D_{\text{dom},i})
$$

如果后续还有 $m$ 次回切，并且缓存命中率为 $h$，命中时只需做增量观察，平均增量成本记为 $D_{\Delta}$，完整重看成本记为 $D_{\text{full}}$，那么总成本可以近似写成：

$$
T_{\text{total}} = T_{\text{init}} + (1-h)mD_{\text{full}} + hmD_{\Delta}
$$

这个公式的意义不在于精确到每一笔 token，而在于说明两个线性关系：

1. 切换次数 $m$ 越大，成本线性上升。
2. 缓存命中率 $h$ 越高，完整重看的比例线性下降。

进一步看边际收益：

$$
\frac{\partial T_{\text{total}}}{\partial h} = m(D_{\Delta} - D_{\text{full}})
$$

因为通常有：

$$
D_{\Delta} \ll D_{\text{full}}
$$

所以提升命中率几乎总是划算的。

用一个更具体的玩具例子说明。

假设有三个标签：

| 标签 | 内容 | 初次观察成本 |
| --- | --- | --- |
| A | 竞品 A 定价页 | 1600 token |
| B | 竞品 B 定价页 | 1500 token |
| C | 内部表单页 | 1700 token |

初次打开三页，总成本：

$$
T_{\text{init}} = 1600 + 1500 + 1700 = 4800
$$

之后流程是：

$$
A \rightarrow B \rightarrow A \rightarrow C
$$

如果完全没有缓存，后续三次切换都按 1500 token 估算，那么追加成本约为：

$$
3 \times 1500 = 4500
$$

总成本接近：

$$
4800 + 4500 = 9300
$$

如果采用摘要缓存，假设：

- 切回 A 只需补 300 token
- 切到 C 因为要操作表单，补 600 token

那么后续成本大致只有：

$$
300 + 600 = 900
$$

总成本变成：

$$
4800 + 900 = 5700
$$

区别不在于模型“更聪明”，而在于系统不再要求模型把旧页面重新完整看一遍。

再看真实工程里的多步验证例子。

一个典型流程是：

1. 在登录页输入账号密码
2. 跳到 MFA 页等待输入验证码
3. 打开邮箱页读取一次性验证码
4. 回到 MFA 页提交
5. 跳回业务页继续操作

这里至少有三个标签：

- 登录页
- MFA 页
- 邮箱页

不同页面的状态特性不同：

| 页面 | 状态特性 | 合理策略 |
| --- | --- | --- |
| 登录页 | 提交后价值快速下降 | 提交成功后降级为低优先级摘要 |
| MFA 页 | 当前核心操作页 | 保持前台高保真状态 |
| 邮箱页 | 会异步变化 | 设短 TTL，按窗口刷新 |

如果不区分这三种状态，Agent 很容易在错误的页面上浪费预算，或者在已经失效的摘要上继续做判断。

所以，四个机制合在一起的核心逻辑是：

- 焦点栈解决“优先看谁”
- 状态快照解决“记住什么”
- 分层缓存解决“分别存什么”
- 失效规则解决“什么时候不能再信”

---

## 代码实现

下面给出一个可以直接运行的 Python 示例，用来演示多标签调度层的核心思路。这个示例不是完整浏览器自动化框架，而是一个最小可运行版本的“多标签状态管理器”。它负责决定：切换标签时，到底是复用已有摘要，还是重新做完整感知。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


SHOT_TOKENS_1080P = 1400
DOM_TOKENS = 400
DELTA_TOKENS = 250


@dataclass
class Snapshot:
    url: str
    title: str
    view_hash: str
    summary: str
    ttl_steps: int
    dirty: bool = False
    last_seen_step: int = 0

    def is_valid(self, current_step: int) -> bool:
        age = current_step - self.last_seen_step
        return (not self.dirty) and age <= self.ttl_steps


@dataclass
class TabState:
    tab_id: str
    url: str
    title: str
    focus_rank: int = 0
    snapshot: Optional[Snapshot] = None


@dataclass
class TabManager:
    tabs: Dict[str, TabState] = field(default_factory=dict)
    focus_stack: List[str] = field(default_factory=list)
    workflow_cache: Dict[str, str] = field(default_factory=dict)
    tool_cache: Dict[str, Dict[str, str]] = field(default_factory=dict)
    step: int = 0
    token_spent: int = 0

    def add_tab(self, tab_id: str, url: str, title: str) -> None:
        if tab_id in self.tabs:
            raise ValueError(f"tab already exists: {tab_id}")
        self.tabs[tab_id] = TabState(tab_id=tab_id, url=url, title=title)

    def mark_dirty(self, tab_id: str) -> None:
        tab = self._get_tab(tab_id)
        if tab.snapshot is not None:
            tab.snapshot.dirty = True

    def close_tab(self, tab_id: str) -> None:
        if tab_id not in self.tabs:
            return
        self.tabs.pop(tab_id)
        self.focus_stack = [x for x in self.focus_stack if x != tab_id]
        self.tool_cache.pop(tab_id, None)

    def switch_tab(self, tab_id: str) -> str:
        self.step += 1
        tab = self._get_tab(tab_id)

        if tab.snapshot and tab.snapshot.is_valid(self.step):
            self.token_spent += DELTA_TOKENS
            tab.snapshot.last_seen_step = self.step
            self._focus(tab_id)
            return "cache_hit_delta"

        self.token_spent += SHOT_TOKENS_1080P + DOM_TOKENS
        tab.snapshot = Snapshot(
            url=tab.url,
            title=tab.title,
            view_hash=self._make_view_hash(tab),
            summary=self._build_summary(tab),
            ttl_steps=self._default_ttl(tab_id),
            dirty=False,
            last_seen_step=self.step,
        )
        self.tool_cache[tab_id] = {
            "last_observation": "full_refresh",
            "view_hash": tab.snapshot.view_hash,
        }
        self._focus(tab_id)
        return "full_refresh"

    def write_workflow_fact(self, key: str, value: str) -> None:
        self.workflow_cache[key] = value

    def read_workflow_fact(self, key: str) -> Optional[str]:
        return self.workflow_cache.get(key)

    def recent_focus(self) -> List[str]:
        return list(self.focus_stack)

    def tab_snapshot_summary(self, tab_id: str) -> Optional[str]:
        tab = self._get_tab(tab_id)
        if tab.snapshot is None:
            return None
        return tab.snapshot.summary

    def _focus(self, tab_id: str) -> None:
        self.focus_stack = [x for x in self.focus_stack if x != tab_id]
        self.focus_stack.insert(0, tab_id)
        for idx, focused_tab_id in enumerate(self.focus_stack, start=1):
            self.tabs[focused_tab_id].focus_rank = idx

    def _default_ttl(self, tab_id: str) -> int:
        # 动态页给更短 TTL，静态页给更长 TTL
        if tab_id == "MAIL":
            return 1
        return 2

    def _build_summary(self, tab: TabState) -> str:
        return f"{tab.title} | {tab.url} | current step={self.step}"

    def _make_view_hash(self, tab: TabState) -> str:
        return f"hash:{tab.tab_id}:{tab.url}:{self.step}"

    def _get_tab(self, tab_id: str) -> TabState:
        if tab_id not in self.tabs:
            raise KeyError(f"unknown tab: {tab_id}")
        return self.tabs[tab_id]


def demo() -> None:
    manager = TabManager()
    manager.add_tab("A", "https://competitor-a.example", "Competitor A Pricing")
    manager.add_tab("B", "https://competitor-b.example", "Competitor B Pricing")
    manager.add_tab("FORM", "https://form.example", "Internal Intake Form")
    manager.add_tab("MAIL", "https://mail.example", "Inbox")

    assert manager.switch_tab("A") == "full_refresh"
    manager.write_workflow_fact("price_a", "299/month")

    assert manager.switch_tab("B") == "full_refresh"
    manager.write_workflow_fact("price_b", "279/month")

    assert manager.switch_tab("A") == "cache_hit_delta"
    assert manager.read_workflow_fact("price_a") == "299/month"

    manager.mark_dirty("FORM")
    assert manager.switch_tab("FORM") == "full_refresh"

    manager.mark_dirty("MAIL")
    assert manager.switch_tab("MAIL") == "full_refresh"

    expected = (
        (1400 + 400) +   # A full
        (1400 + 400) +   # B full
        250 +            # A delta
        (1400 + 400) +   # FORM full
        (1400 + 400)     # MAIL full
    )
    assert manager.token_spent == expected
    assert manager.recent_focus()[0] == "MAIL"
    assert manager.read_workflow_fact("price_b") == "279/month"

    print("recent_focus =", manager.recent_focus())
    print("token_spent =", manager.token_spent)
    print("form_summary =", manager.tab_snapshot_summary("FORM"))


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行。保存为 `tab_manager_demo.py` 后执行：

```bash
python3 tab_manager_demo.py
```

预期输出类似：

```text
recent_focus = ['MAIL', 'FORM', 'A', 'B']
token_spent = 7450
form_summary = Internal Intake Form | https://form.example | current step=4
```

这段代码体现了几个关键点。

第一，`switch_tab()` 不是盲目重看页面，而是先判断缓存是否还能用。

如果快照存在，且没有过期，也没有被标记为 `dirty`，那么系统只做一次增量确认，成本是 `DELTA_TOKENS`。这个 delta 不等于“完全不看页面”，而是“只做足够确认当前状态的最小观察”。

第二，快照不是永久可信的。

`Snapshot.is_valid()` 同时检查两件事：

- 是否被 `dirty` 标记
- 是否超过 `ttl_steps`

这分别对应前面说的两类失效边界：

- 事件边界
- 时间或步数边界

第三，workflow cache 和 tool cache 被分开了。

| 缓存 | 示例内容 | 为什么单独存 |
| --- | --- | --- |
| `workflow_cache` | `price_a = 299/month` | 这是任务事实，不应随页面小变动而丢失 |
| `tool_cache` | `last_observation = full_refresh` | 这是观测痕迹，页面变化后可能立即失效 |

对新手来说，这个区分非常重要。很多实现之所以越做越乱，就是把“任务已经知道的事实”和“上次怎么看到的页面”混在同一个对象里，结果页面一变，连业务结果也跟着丢了。

第四，`focus_stack` 记录的是最近使用顺序，而不是固定 tab 顺序。

`_focus()` 每次把当前标签放到列表最前面，再重算所有标签的 `focus_rank`。这样系统就可以很方便地做进一步优化，例如：

- 只保留前 1 到 2 个标签的高保真状态
- 对长时间未访问的标签自动降级
- 优先关闭低优先级且已完成任务的标签

如果把这套示例映射到真实系统，常见组件关系通常如下：

| 组件 | 作用 | 典型内容 |
| --- | --- | --- |
| `TabState` | 记录标签当前状态 | `tab_id`、URL、标题、快照、焦点顺序 |
| `Snapshot` | 保存可复用页面摘要 | `view_hash`、`summary`、TTL、dirty |
| `workflow_cache` | 保存任务阶段结论 | 已提取价格、已获取验证码、已完成字段 |
| `tool_cache` | 保存底层观测结果 | 某次截图摘要、DOM 过滤结果、元素树指纹 |
| `focus_stack` | 记录优先级 | 最近访问顺序、待降级顺序 |

再看一个更贴近工程的流程例子：跨站采购申请。

假设 Agent 要做下面几件事：

1. 在供应商网站查看价格
2. 在内部系统填写采购申请单
3. 到邮箱里取登录验证码
4. 再回内部系统提交

合理实现通常不是让三个标签持续上传整页上下文，而是按下面方式做：

1. 供应商页提取完价格后，把结构化字段写入 workflow cache。
2. 内部表单页保持为前台高优先级页，保留最新可交互元素树。
3. 邮箱页设置短 TTL，只在预期验证码到达的窗口内刷新。
4. 提交采购单后，供应商页如果不再需要，就关闭或降级。

这样做的价值不只是“节约一些 token”，更重要的是降低错误率。页面越多，模型越容易被旧状态、无关状态、相似状态干扰。多标签调度层的职责，就是限制模型真正需要看到的页面范围。

对新手来说，可以把这一层理解成一句话：

不是让模型“记住所有页面”，而是替模型决定“哪些页面现在值得重新看”。

---

## 工程权衡与常见坑

多标签控制里，最大的权衡通常是“并行加载”与“串行操作”。

并行加载的优点是节省等待时间。比如同时打开多个竞品页，让网络请求一起发出；等模型真正需要读内容时，这些页可能已经加载完成。

串行操作的优点是状态清晰。一次只把一个标签当作当前工作面，容易命中缓存，也更不容易误操作。

大多数系统适合的折中方案是：

$$
\text{并行预加载} + \text{串行决策与操作}
$$

这里要区分两个概念：

| 概念 | 含义 | 成本 |
| --- | --- | --- |
| 预加载 | 页面先打开、资源先加载 | 主要是浏览器和网络成本 |
| 高保真活跃 | 页面状态持续提供给模型 | 主要是 token 和上下文成本 |

预加载并不等于高保真活跃。很多新手会把这两件事混在一起，结果一边想节省等待时间，一边又把所有页的完整状态都喂给模型，最后两边都没优化好。

第二个权衡是“截图优先”还是“结构化输入优先”。

两者各有优势：

| 输入方式 | 优点 | 缺点 | 更适合什么 |
| --- | --- | --- | --- |
| 截图 | 保留完整视觉布局 | token 高，信息冗余大 | 验证码弹窗、视觉定位、复杂富文本页 |
| DOM/可交互树 | 结构清晰、token 更省 | 对视觉细节不敏感 | 表单、表格、标准化管理后台 |
| 裁剪截图 + 局部 DOM | 平衡视觉和成本 | 需要额外调度逻辑 | 混合型页面 |

工程上更稳妥的做法通常是：

1. 先用结构化输入做主观察。
2. 如果结构信息不足，再补截图。
3. 尽量补局部，不要默认整页重看。

这相当于把“高成本全量观察”变成“低成本默认 + 按需升级”。

下面直接列常见坑。

| 坑 | 结果 | 修正方式 |
| --- | --- | --- |
| 每次切换都完整截图和抽 DOM | token 快速爆炸 | 先查缓存，优先补 delta |
| 缓存只有一层 | 写操作后全局污染，状态难追踪 | 区分 workflow cache 和 tool cache |
| 没有失效规则 | 继续使用陈旧页面摘要 | 同时用 TTL 和 dirty 事件 |
| 并行高保真标签太多 | 焦点混乱，容易误操作 | 限制活跃标签数量 |
| 不主动关页 | 上下文长期残留无关状态 | 阶段完成后关闭或降级 |
| 只记录 URL 不记录视图状态 | 同 URL 不同页面状态被混淆 | 增加 `view_hash` 或关键字段摘要 |
| 任务共享同一缓存空间 | 一个任务污染另一个任务 | 为任务隔离命名空间 |
| 写入结果不落到外部状态层 | 一切依赖页面回看 | 关键结论同步写入 workflow cache |

这里重点展开两个最容易踩的坑。

第一个是 MFA 场景中的陈旧邮箱摘要。

流程通常是：

1. Agent 打开邮箱页，第一次没看到验证码。
2. 几秒后验证码到了。
3. 系统还在使用第一次生成的邮箱摘要。
4. 模型继续得出“未收到验证码”的结论。

这不是模型推理差，而是缓存边界错了。邮箱页这种异步变化标签，本质上不适合长 TTL。正确策略通常是：

- 短 TTL
- 有预期时间窗口
- 必要时主动轮询刷新
- 每次重新进入验证码确认步骤前做一次增量观察

第二个是多 Agent 共享浏览器。

共享浏览器意味着多个任务可能同时改写：

- 当前焦点
- 页面路由
- 已打开标签
- 登录态
- 表单内容

如果没有每个任务自己的焦点栈和缓存命名空间，一个 Agent 刚摘要完的页面，可能已经被另一个 Agent 点击跳转了。前者还以为自己拿到的是有效状态，后者又会把错误进一步扩散。

这个问题通常不是靠“更大模型”解决，而是靠系统隔离解决。常见做法是：

- 会话隔离
- 缓存隔离
- 标签空间隔离
- 必要时每任务独占浏览器上下文

从工程视角看，多标签系统最容易失败的地方，不是不会开新标签，而是没有定义“什么时候该忘记旧状态”。

---

## 替代方案与适用边界

并不是所有任务都需要复杂的多标签缓存系统。

如果任务满足下面几个条件，直接用较高保真的浏览器上下文驱动，通常是可接受的：

- 总步骤较短
- 页面数量不多
- 状态变化不复杂
- 视觉理解比结构抽取更重要

例如，5 到 8 步的登录验证任务，或者需要多个 Agent 共享同一浏览器语义状态的研究型场景，保持较完整的可见上下文，反而能减少系统复杂度。因为这时引入复杂缓存层的收益未必高，维护成本反而会更明显。

但一旦任务变长，尤其是混合多个外部系统时，浏览器上下文就不应该继续充当唯一事实来源。

例如下面这类任务：

- 浏览器里做采购申请
- 文件系统里读取附件
- 代码仓库里查配置
- CI 日志里看失败原因
- 邮箱里取验证码或确认邮件

这种场景下，如果还让浏览器页面承担全部记忆功能，系统会越来越脆弱。因为浏览器页面适合“观察当前界面”，不适合“长期承载整个任务状态”。

更合适的替代方案通常有三类：

| 方案 | 适用情况 | 局限 |
| --- | --- | --- |
| 纯浏览器高保真模式 | 短流程、强视觉依赖、多 Agent 协作实验 | token 成本高，陈旧状态难控制 |
| 浏览器摘要 + 外部状态存储 | 中长流程、跨站读写、预算敏感 | 需要设计缓存、失效与同步逻辑 |
| 传统脚本自动化 + 少量模型决策 | 页面稳定、规则明确、重复任务多 | 泛化差，未知页面上脆弱 |

对零基础到初级工程师来说，判断标准可以先压缩成三个问题：

1. 页面是否高度动态？
2. 步骤是否很多？
3. 是否必须让模型持续看到完整页面？

如果这三个问题中有两个回答是“是”，就不应该让所有标签都长期保持高保真上下文。更稳妥的做法是：

- 浏览器层负责采集和执行
- 外部状态层负责记忆和预算控制
- 模型只在需要重新观察时拿到增量输入

可以把这个判断进一步写成一个简化决策表：

| 条件 | 更建议的方案 |
| --- | --- |
| 2 到 3 个标签，少于 8 步，强视觉依赖 | 纯浏览器高保真模式 |
| 3 到 6 个标签，10 到 30 步，频繁跨站读写 | 浏览器摘要 + 外部状态存储 |
| 页面稳定、规则清晰、长期重复执行 | 脚本自动化为主，模型只处理异常分支 |

再给一个决策化的真实例子。

假设任务是“竞品调研 + 内部表单提交”：

- 供应商网站 4 个标签页
- 内部系统 1 个表单页
- 总流程超过 20 步
- 中间还要做字段比对和归纳

这个任务更适合的架构是：

1. 并行预加载供应商页。
2. 用结构化方式抽取价格、功能、套餐信息。
3. 把结果写入外部状态层。
4. 关闭或降级低价值标签。
5. 串行填写内部表单。

不适合的做法，则是让 5 个标签一直完整驻留在主上下文中，要求模型每一步都“记得全部页面”。

反过来，如果只是 6 步左右的登录与二次验证，保持 2 到 3 个标签的完整状态通常就够了。因为在这种短流程里，复杂缓存系统带来的实现成本，未必能换来足够收益。

所以，多标签控制不是能力越强越复杂，而是应该和任务长度、页面动态性、视觉依赖程度匹配。系统设计的目标，不是最大化页面数量，而是最小化无效重看。

---

## 参考资料

- Browser Use 多标签与状态管理机制解析：<https://enricopiovano.com/blog/browser-use-ai-browser-automation-deep-dive?utm_source=openai>
- 截图、可访问性树与结构化查询的 token 成本讨论：<https://medium.com/%40kaptnqu/i-built-a-new-software-primitive-in-8-5-hours-it-replaces-the-eyes-of-every-ai-agent-on-earth-e085bd223033?utm_source=openai>
- Agent 工作流中的分层缓存与失效策略：<https://www.mdpi.com/2504-4990/8/2/30?utm_source=openai>
- UI Agent 中 DOM 过滤与提取成本说明：<https://arunbaby.com/ai-agents/0023-ui-automation-agents/?utm_source=openai>
- Playwright MCP 与 CLI 的 token 开销对比：<https://scrolltest.medium.com/playwright-mcp-burns-114k-tokens-per-test-the-new-cli-uses-27k-heres-when-to-use-each-65dabeaac7a0?utm_source=openai>
- MCP schema 负担与上下文成本讨论：<https://www.buildmvpfast.com/blog/mcp-hidden-cost-cli-agent-infrastructure-2026?utm_source=openai>
