## 核心结论

浏览器 Agent 的表单自动填写，本质上不是“看见页面就把字打进去”，而是先把页面转换成一组**字段类型 + 操作规则 + 页面状态**，再按顺序执行。字段类型指输入框、下拉框、单选框、复选框、文件框这类控件类别；操作规则指“文本框输入字符串、下拉框选择 option、复选框切换状态、文件框设置文件路径”这类动作映射；页面状态指哪些字段当前可见、哪些字段会在前一步动作后出现、哪些字段正在校验或暂时不可提交。

对新手可以直接这样理解：Agent 先回答三个问题。

1. 这是什么框。
2. 应该往里填什么。
3. 现在能不能填，填完之后页面会不会变。

如果页面写着“选择有驾照后才显示驾照号”，它不能一开始就假设“驾照号”一定存在，而要先完成“是否有驾照”这个决策，再等待新字段出现，再继续填写。也就是说，表单自动填写不是单轮识别，而是一个持续更新的决策过程。

复杂表单的难点，不在单个字段本身，而在三个问题同时出现时的组合爆炸：字段会动态出现，控件可能不是原生 HTML，提交前还可能有验证码或异步校验。问题一叠加，成功率就会明显下降。

| 场景 | 典型特征 | 自动填写难点 | 结果 |
|---|---|---|---|
| 简单表单 | 固定字段、原生控件 | 类型识别即可 | 成功率高 |
| 动态表单 | 条件字段、多步骤 | 要监听 DOM 变化并重规划 | 成功率明显下降 |
| 真实注册表单 | 动态字段 + 验证码 + 自定义组件 | 识别、交互、验证三重耦合 | 成功率进一步下降 |

公开研究与工程观察都指向同一结论：浏览器 Agent 在真实网页任务上还远没到“稳定代替人”的程度。WebArena 一类真实网页任务的代表性成绩大约在 60% 左右，JSAI 2025 针对真实表单的覆盖率结果只有 37.12%。你给出的 50 个真实注册表单完整填写成功率约 62%，和这个区间是相容的。失败主要集中在验证码、滑块、人造下拉框、日期选择器、异步校验与多步骤流程。

Chrome Autofill 给出的工程启发也很明确：**稳定结构上的自动填充非常有效，复杂路径上的全自动执行仍然脆弱。** Chrome 官方披露，使用 Autofill 的用户相比纯手输，表单放弃率平均低 75%。这说明工程上最可靠的方向不是“单次全自动莽过去”，而是把流程拆成多阶段、优先吃掉稳定字段、对高风险节点单独处理。

---

## 问题定义与边界

先区分两个概念。

**Autofill**：浏览器根据历史资料，把“姓名、地址、邮箱、卡号”这类标准字段自动填进去。它的前提是字段语义相对稳定，浏览器能较高置信度地判断“这是地址栏”“这是邮箱框”。

**浏览器 Agent**：不仅填字段，还要理解当前任务目标、决定下一步点击什么、处理跳转、等待页面变化、必要时调用外部工具。它不是简单的字段补全器，而是一个在网页环境里执行任务的控制器。

所以，这篇文章讨论的不是传统自动补全，而是更难的任务：给定目标，例如“完成注册”“提交贷款申请”“填写求职表”，Agent 如何在真实网页中完成整条表单链路。

边界也必须提前说清楚，否则讨论会失真。

1. 不讨论绕过站点安全策略的非法手段。
2. 不把 CAPTCHA 识别当作必然可解问题，它通常就是自动化的硬边界。
3. 不假设所有页面都使用标准 HTML 控件。很多站点会把下拉框、日期框、上传框包装成自定义组件。
4. 不假设页面静态不变。现代前端会在输入后即时增删字段、切换校验规则、刷新按钮状态。
5. 不把“页面上看起来有值”误当成“任务已经成功”。真实系统往往还要求前端状态和后端校验同时通过。

可以把整体成功率粗略写成：

$$
P(\text{success}) \approx P(\text{字段识别正确}) \times P(\text{动作执行正确}) \times P(\text{动态路径覆盖}) \times (1 - P(\text{验证码/风控阻断}))
$$

这条式子表达的是乘法关系：任何一项明显偏低，整体成功率都会被拉下来。比如字段识别率 95%、动作成功率 90%、动态路径覆盖率 80%、验证码阻断概率 20%，整体成功率大约只有：

$$
0.95 \times 0.90 \times 0.80 \times (1 - 0.20) = 0.5472
$$

也就是 54.72%。这正好解释了为什么很多“单点能力不错”的系统，最终端到端成功率仍然不高。

如果进一步展开到字段层面，可以写成：

$$
P(\text{success}) \approx \left(\frac{\sum_{i=1}^{n}\mathbf{1}(\text{field}_i\ \text{识别并填写成功})}{n}\right)\times (1-r_{captcha}) \times (1-r_{custom})
$$

其中：

- $n$ 是当前任务涉及的字段数量。
- $\mathbf{1}(\cdot)$ 是指示函数，某个字段识别并填写成功时取 1，否则取 0。
- $r_{captcha}$ 是验证码或风控导致的失败率。
- $r_{custom}$ 是自定义组件导致的失败率。

对新手来说，可以把表单理解成一张会变化的状态图，而不是固定问卷。Agent 不是“拿着答案逐项抄写”，而是“每做一步，就重新确认下一步是否发生变化”。

下面这张表可以把边界和难点压缩清楚。

| 维度 | 传统 Autofill | 浏览器 Agent | 为什么会难 |
|---|---|---|---|
| 目标 | 补全单个字段 | 完成整条任务链路 | 要处理跳转、等待、条件分支 |
| 控件假设 | 标准字段为主 | 必须面对自定义组件 | 页面实现方式不统一 |
| 页面状态 | 默认静态 | 默认动态 | 输入后 DOM 可能重排 |
| 失败来源 | 字段识别不准 | 识别、交互、验证、风控叠加 | 端到端失败点更多 |
| 成功标准 | 字段被填上 | 最终提交成功 | 需要完整闭环 |

---

## 核心机制与推导

核心流水线可以压缩成四步：

1. 识别字段
2. 选择动作
3. 监听页面变化
4. 处理验证与异常

这四步不是一次性执行，而是不断循环执行，直到所有必填字段完成或流程被阻断。

### 1. 字段识别：先判断“这是什么”

一个可靠的 Agent 不会只看标签文字，它通常会综合多个信号，因为真实页面经常缺 label、乱命名、把语义藏在别的属性里。

| 信号 | 用途 | 例子 |
|---|---|---|
| DOM 标签 | 判断基础控件类型 | `input`、`select`、`textarea` |
| `type/name/id` | 判断语义 | `email`、`password`、`dob` |
| 标签文本 | 推断字段含义 | “手机号”“所在城市” |
| ARIA 属性 | 补足可访问性语义 | `role="combobox"` |
| 邻近文本 | 识别无 label 表单 | 占位符、帮助文案 |
| 校验特征 | 判断字段约束 | `required`、`pattern`、错误文案 |
| 可交互状态 | 避免对隐藏/禁用字段操作 | `disabled`、`display:none` |

这里最容易被新手忽略的一点是：**字段识别不等于字段命名匹配。**  
例如一个“公司邮箱”输入框，真实页面上可能没有 `name="company_email"`，而只有：

```html
<input type="text" placeholder="Work email">
```

这时 Agent 需要结合 `placeholder`、旁边的“Business Email”文本、以及后续出现的企业资料字段，才能判断它是“公司邮箱”，而不是个人邮箱。

### 2. 动作映射：再决定“该怎么做”

识别出字段类型之后，Agent 才能选择对应动作。最常见的映射如下。

| field type | 常用动作 | 典型失败点 |
|---|---|---|
| text/email/password | 聚焦后输入，必要时触发 `input/change/blur` | React/Vue 受控输入框不认直接赋值 |
| select | 打开后按 value/label 选项 | 自定义下拉不是真 `select` |
| radio | 找同组项并点击目标值 | 标签文字和实际 value 不一致 |
| checkbox | 判断当前状态后勾选/取消 | 误触发联动字段 |
| file | 设置文件路径或桥接上传接口 | 浏览器安全限制、拖拽上传 |
| date | 按站点要求格式输入或调用日期面板 | 本地化格式差异 |
| otp/code | 等待外部验证码并填入 | 数据不在当前页面，需要外部工具 |

这一步的关键不是“能不能把值写进去”，而是“有没有按页面预期的方式写进去”。  
例如某些 React 受控输入框，如果只改 DOM 的 `value`，界面上虽然看着有字，但组件内部状态并没有更新，提交时仍会报空值。

### 3. 监听页面变化：把页面当作状态机

一个“玩具例子”最能说明动态路径问题。

假设页面有两题：

- 是否有驾照：`是 / 否`
- 驾照号：只有选“是”才显示

正确顺序不是“先搜索驾照号输入框”，而是：

1. 识别到单选题“是否有驾照”
2. 根据用户资料选择“是”
3. 等待 DOM 更新
4. 发现新出现的“驾照号”字段
5. 再执行文本输入

这就是条件规则驱动的路径展开。工程上常写成“当/则”规则：

- 当字段 A 的值变成 `yes`
- 则重新扫描受影响区域
- 将新增必填字段加入待办队列

真实工程里，这种路径依赖比玩具例子复杂得多。以“海外 SaaS 注册 + 公司资料补充”为例：

| 步骤 | 页面动作 | 可能触发的变化 |
|---|---|---|
| 1 | 输入邮箱、密码 | 邮箱域名可能决定后续分支 |
| 2 | 选择个人或企业 | 企业分支新增公司名、员工数、行业 |
| 3 | 选择行业 | 某些行业新增合规声明 |
| 4 | 上传资料 | 出现文件校验、格式校验 |
| 5 | 点击提交 | 触发 reCAPTCHA、邮箱验证码或短信验证 |

这里失败往往不是“不会输入邮箱”，而是路径管理失效：Agent 填完前两步后，没有意识到后续新增了三个必填字段，或者把自定义下拉误判成普通文本框。

因此，真正可靠的机制不是单轮解析，而是“解析 -> 操作 -> 重新观察 -> 再规划”的闭环。可以抽象成：

$$
S_{t+1} = f(S_t, a_t, \Delta DOM_t)
$$

其中：

- $S_t$ 是时刻 $t$ 的表单状态；
- $a_t$ 是当前动作；
- $\Delta DOM_t$ 是这一步动作后页面结构变化。

只要 $\Delta DOM_t \neq 0$，就必须重新规划后续字段队列。

进一步写成队列更新形式，会更接近工程实现：

$$
Q_{t+1} = \big(Q_t - \{f_t\}\big) \cup \text{NewVisibleFields}(\Delta DOM_t)
$$

其中 $Q_t$ 表示时刻 $t$ 尚未完成的字段队列，$f_t$ 表示当前刚处理完的字段。这个式子的意思很直接：每做完一个字段，就要把它从待办中移除，再把新暴露出来的字段补进队列。

### 4. 处理验证与异常：最后确认“任务真的完成了”

表单系统常见的验证分三层：

| 验证层 | 检查对象 | 常见例子 |
|---|---|---|
| 前端即时校验 | 输入格式是否正确 | 邮箱格式、密码强度、手机号长度 |
| 异步业务校验 | 服务器是否接受当前值 | 用户名是否已存在、邀请码是否有效 |
| 提交级校验 | 整体表单能否提交 | 必填项缺失、风控命中、验证码失败 |

因此，“看到页面里有值”不代表任务已经完成。至少要满足三层状态一致：

1. DOM 里的值变了。
2. 前端框架状态变了。
3. 后端校验认可了。

只满足第 1 层，经常会出现肉眼看着填上了，提交时却报“必填项为空”的情况。

---

## 代码实现

下面用一个**可运行的 Python 示例**演示核心思想：字段类型识别、动作映射、条件字段展开、校验、失败重试。它不是浏览器驱动代码，但它表达的是 Agent 的决策骨架，而且可以直接运行。

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FormField:
    name: str
    field_type: str
    required: bool = True
    visible: bool = True
    options: Optional[List[str]] = None
    depends_on: Optional[str] = None
    depends_value: Optional[str] = None


@dataclass
class FormState:
    fields: List[FormField]
    values: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def visible_fields(self) -> List[FormField]:
        result: List[FormField] = []
        for field_obj in self.fields:
            if not field_obj.visible:
                continue
            if field_obj.depends_on is None:
                result.append(field_obj)
                continue
            if self.values.get(field_obj.depends_on) == field_obj.depends_value:
                result.append(field_obj)
        return result

    def required_visible_field_names(self) -> List[str]:
        return [
            field_obj.name
            for field_obj in self.visible_fields()
            if field_obj.required
        ]


def detect_action(field_obj: FormField) -> str:
    mapping = {
        "text": "send_keys",
        "email": "send_keys",
        "password": "send_keys",
        "select": "select_option",
        "radio": "click_radio",
        "checkbox": "toggle_checkbox",
        "file": "set_file",
        "date": "set_date",
    }
    if field_obj.field_type not in mapping:
        raise ValueError(f"unsupported field type: {field_obj.field_type}")
    return mapping[field_obj.field_type]


def validate_value(field_obj: FormField, value: str) -> Tuple[bool, str]:
    if field_obj.field_type == "email" and "@" not in value:
        return False, "invalid email"
    if field_obj.field_type in {"select", "radio"}:
        if not field_obj.options:
            return False, "missing options"
        if value not in field_obj.options:
            return False, f"invalid option: {value}"
    if field_obj.field_type == "file" and value == "":
        return False, "empty file path"
    return True, ""


def apply_action(state: FormState, field_obj: FormField, value: str) -> None:
    action = detect_action(field_obj)
    ok, err = validate_value(field_obj, value)
    if not ok:
        state.errors[field_obj.name] = err
        raise ValueError(f"{field_obj.name}: {err}")

    # 在真实浏览器里，这里会触发 input/change/blur 等事件
    state.values[field_obj.name] = value
    state.errors.pop(field_obj.name, None)

    allowed_actions = {
        "send_keys",
        "select_option",
        "click_radio",
        "toggle_checkbox",
        "set_file",
        "set_date",
    }
    if action not in allowed_actions:
        raise RuntimeError(f"unexpected action: {action}")


def fill_form(state: FormState, payload: Dict[str, str], max_retry: int = 2) -> Dict[str, str]:
    attempt = 0
    while attempt <= max_retry:
        attempt += 1
        progress_made = False

        for field_obj in state.visible_fields():
            if field_obj.name in state.values:
                continue

            if field_obj.required and field_obj.name not in payload:
                continue

            if field_obj.name in payload:
                apply_action(state, field_obj, payload[field_obj.name])
                progress_made = True

        missing_required = [
            name for name in state.required_visible_field_names()
            if name not in state.values
        ]

        if not missing_required:
            return state.values

        if not progress_made:
            break

    missing_required = [
        name for name in state.required_visible_field_names()
        if name not in state.values
    ]
    raise RuntimeError(f"form fill failed, missing required fields: {missing_required}")


def main() -> None:
    fields = [
        FormField(name="email", field_type="email"),
        FormField(
            name="has_license",
            field_type="radio",
            options=["yes", "no"],
        ),
        FormField(
            name="license_no",
            field_type="text",
            depends_on="has_license",
            depends_value="yes",
        ),
        FormField(
            name="country",
            field_type="select",
            options=["CN", "US", "SG"],
        ),
        FormField(name="resume", field_type="file", required=False),
    ]

    payload = {
        "email": "alice@example.com",
        "has_license": "yes",
        "license_no": "A1234567",
        "country": "CN",
    }

    state = FormState(fields=fields)
    result = fill_form(state, payload)

    assert result["email"] == "alice@example.com"
    assert result["has_license"] == "yes"
    assert result["license_no"] == "A1234567"
    assert result["country"] == "CN"
    assert "resume" not in result

    print("fill success")
    print(result)


if __name__ == "__main__":
    main()
```

这段代码有三个要点。

1. `visible_fields()` 不是返回全部字段，而是返回**当前状态下可见的字段**。这对应浏览器里“条件字段出现后再处理”。
2. `detect_action()` 把字段类型映射成动作，这对应真实网页里的点击、输入、选择、上传。
3. `fill_form()` 不是只扫一遍，而是在每一轮基于最新状态重新判断还有哪些必填字段没有完成。

如果把它翻译成浏览器里的实际执行逻辑，骨架通常类似下面这样：

```text
扫描当前可见字段
  -> 识别 fieldType
  -> 从用户资料中取目标值
  -> 执行动作
  -> 等待 input/change/blur/网络响应
  -> 监听 DOM 变化
  -> 如果出现新字段，更新待办队列
  -> 如果发现验证码或自定义组件，切换到专门处理器
  -> 重复直到所有必填字段完成或任务阻断
```

再往工程实现推进一步，通常至少要预留两个 hook。

1. `custom_component_resolver`  
作用是告诉 Agent：这个不是原生 `select`，而是 Ant Design、MUI、Headless UI 之类的自定义组件，该怎么点开、怎么选值。

2. `challenge_handler`  
作用是统一处理验证码、短信码、邮件码、图片选择、人机确认等高风险步骤。没有这个钩子，所谓“全自动”通常只能跑到提交前一步。

如果要把“玩具代码”扩展到真实浏览器自动化，一般还需要补三类能力。

| 能力 | 为什么需要 | 常用实现思路 |
|---|---|---|
| 事件仿真 | 让前端框架状态同步更新 | 触发 `focus/input/change/blur` |
| DOM 监听 | 捕获条件字段和按钮状态变化 | `MutationObserver` + 局部重扫 |
| 提交判定 | 区分“填了值”和“任务成功” | 观察错误提示、成功跳转、网络响应 |

---

## 工程权衡与常见坑

实际项目里，问题几乎都出在“页面不像标准 HTML 教科书那样工作”。

| 问题 | 影响 | 缓解方式 |
|---|---|---|
| 验证码 / 滑块 / 风控挑战 | 直接阻断提交 | 独立挑战处理器、人工接管、失败回退 |
| 自定义下拉/日期组件 | 误判 field type | 组件白名单 + 专用选择器策略 |
| 条件字段未重扫 | 漏填必填项 | `MutationObserver` + 局部 DOM diff |
| 受控输入框 | 值写进 DOM 但业务层没更新 | 模拟真实键盘事件并触发 blur/change |
| 异步校验 | 还没校验完就点提交 | 等待校验状态稳定、观察错误提示 |
| 多步骤表单 | 上一步信息影响下一步结构 | 分步提交，每步后重新建模 |
| 文件上传 | 浏览器安全限制 | 走原生 file input 或页面提供的上传 API |

最常见的误区，是把“识别到字段”误当成“填写成功”。真实网页里，至少有三层状态要一致：

1. DOM 里的值变了。
2. 前端框架状态变了。
3. 后端校验认可了。

只满足第 1 层，经常会出现肉眼看着填上了，提交时却报“必填项为空”。

另一个高频坑，是没有把“页面变化”作为一等公民。对新手可以这样理解：表单不是纸质表，而像会自己长出新问题的活页面。你刚填完“公司类型”，页面突然多出“税号”和“注册地址”；如果 Agent 不重新看一遍，就等于漏做题。

下面这个“失败分层”表，对排错很有用。

| 失败层 | 表现 | 根因 | 修复方向 |
|---|---|---|---|
| 识别失败 | 把日期框当文本框 | 语义信号不足或规则缺失 | 增加字段分类规则 |
| 交互失败 | 页面有值但提交不认 | 事件没触发、组件是受控的 | 改用真实事件流 |
| 路径失败 | 后续字段没填到 | 条件字段出现后未重扫 | 增加 DOM 监听与队列更新 |
| 校验失败 | 提交前被红字拦下 | 格式错误、异步校验未通过 | 加入等待和错误恢复 |
| 挑战失败 | 验证码、滑块阻断 | 风控是独立对抗面 | 人工接管或独立处理 |

验证码要单独强调。Baymard 的用户研究显示，CAPTCHA 首次输入失败率约 8.66%，区分大小写时失败率更高。这个数据本来是人类用户的数据，不是 Agent 数据；但它恰恰说明验证码天然制造摩擦。对 Agent 来说，验证码不是普通字段，而是一个独立的对抗面。很多系统完整填写率被压低，不是因为文本框不会填，而是因为最后一道挑战让整个任务归零。

再补一个容易被忽略的工程权衡：**不要用统一策略硬套所有站点。**  
如果一个系统同时覆盖招聘网站、政府系统、企业 SaaS 注册页、电商结账页，那么它面对的控件库、字段命名、验证逻辑、风控级别会完全不同。真正可维护的做法通常是：

1. 先有一套通用字段识别和动作映射。
2. 再给高频站点加站点级或组件级适配器。
3. 把验证码、OTP、文件上传这类高风险节点拆成独立模块。

---

## 替代方案与适用边界

不是所有表单都值得上 Agent。选择方案时，核心不是“哪个技术更先进”，而是“哪个方案在当前风险和成本约束下最可靠”。

| 方案 | 适合场景 | 优点 | 局限 |
|---|---|---|---|
| Chrome Autofill / 传统 Autofill | 地址、姓名、支付、固定注册字段 | 稳定、快、用户体验好 | 几乎不处理复杂逻辑 |
| 浏览器 Agent | 多步骤、跨页面、条件字段、需要决策 | 能联动上下文与工具 | 稳定性受验证码和自定义组件影响大 |
| 人工 + 辅助工具 | 高价值、低容错、安全要求高 | 准确率最高 | 成本高、扩展性差 |

可以用一个简单判断：

- 表单字段固定、语义标准、无复杂联动，用 Autofill。
- 表单跨多个步骤、字段会动态出现、还要读取上下文资料，用 Agent。
- 页面风控强、验证码频繁、一次失败代价高，用人工或半自动接管。

一个直观例子：

- “收货地址新增页面”只有姓名、电话、地址、邮编，这类页面用 Chrome Autofill 最合适。
- “企业服务注册 + 身份认证 + 上传营业执照 + 条件声明 + 邮箱验证”这种链路，用 Agent 才有意义。
- “银行开户、政府申报、高价值 B2B 采购准入”这类流程，即使用 Agent，也最好设计成人工确认式，而不是盲目追求全自动。

工程上真正可落地的策略通常不是三选一，而是分层：

1. 先用 Autofill 吃掉标准字段。
2. 再用 Agent 处理动态路径和跨页面跳转。
3. 在验证码、OTP、风控节点切人工或专门服务。

这比“让一个 Agent 从头包到尾”更现实，也更接近今天的成功率上限。

如果把适用边界再压缩成一句话，就是：

$$
\text{Agent 的价值} \uparrow \quad \text{当且仅当} \quad \text{页面决策复杂度} > \text{字段填写复杂度}
$$

意思是：当难点主要来自“要不要跳转、要不要等待、后续会不会变”时，Agent 才真正有优势；如果页面只是几个稳定字段，Agent 往往是在用更复杂的系统解决一个简单问题。

---

## 参考资料

- [Chrome for Developers: Autofill in action: real-world insights](https://developer.chrome.com/blog/autofill-insights-2024)  
  贡献：给出 Chrome Autofill 在真实地址/支付表单中的效果，核心数据是使用 Autofill 的用户平均放弃率低 75%。  
  建议阅读方式：重点看表单放弃率、填写速度和错误率三个指标，不必把它误读成“复杂表单也适合全自动”。

- [RoboTask: Set Form Field Value](https://robotask.com/help/set-form-field-value.htm)  
  贡献：清楚列出表单字段类型与动作映射，适合说明 `text/select/radio/checkbox/file` 的操作模型。  
  建议阅读方式：把它当作“动作字典”，用于理解字段类型到执行动作的基础映射，不要把它当成真实网页 Agent 的完整方案。

- [JSAI 2025 Paper 3K4-IS-2a-03](https://www.jstage.jst.go.jp/article/pjsai/JSAI2025/0/JSAI2025_3K4IS2a03/_pdf/-char/en)  
  贡献：给出简单表单、动态表单、真实表单三类任务的覆盖率对比，其中 GPT-4o 在真实表单上的覆盖率为 37.12%，说明真实场景显著更难。  
  建议阅读方式：重点看任务分层和实验设置，理解为什么“简单表单表现不错”不等于“真实业务可稳定上线”。

- [Baymard: CAPTCHAs Have an 8% Failure Rate, and 29% if Case Sensitive](https://baymard.com/blog/captchas-in-checkout)  
  贡献：说明 CAPTCHA 对人类用户本身就有显著摩擦，因此对自动化系统更是强阻断项。  
  建议阅读方式：重点看它提供的失败率和用户摩擦结论，把 CAPTCHA 视为流程阻断因子，而不是普通表单字段。

- [Medium: AI Agents: The Hype, The Reality, and Why 95% of Projects Still Fail](https://medium.com/%40hashim200222/ai-agents-the-hype-the-reality-and-why-95-of-projects-still-fail-351590958e21)  
  贡献：汇总 WebArena 等网页代理基准的量级数据，用来定位“网页 Agent 整体还不稳定”的行业背景。文中提到 WebArena 中期代表成绩约 61.7%。  
  建议阅读方式：只把它当作行业背景材料，具体技术判断仍应优先参考论文、官方工程博客和可复现实验。
