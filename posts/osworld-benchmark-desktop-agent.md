## 核心结论

OSWorld 是一个跨平台桌面 Agent 基准。这里的“基准”，可以直接理解成一套统一出题、统一环境、统一判分的测试集。它把 Ubuntu、Windows、macOS 上的真实电脑操作任务放进同一套评测里，让不同 Agent 能被直接比较，而不是各自做一段好看的 demo。

它最重要的价值，不是展示“模型会不会点鼠标”，而是把桌面自动化能力压缩成一个清晰指标：

$$
SuccessRate = \frac{\text{成功任务数}}{\text{总任务数}}
$$

在 OSWorld 原始论文设定里，总任务数是 369：

$$
SuccessRate = \frac{\text{成功任务数}}{369}
$$

这 369 个任务覆盖浏览器、办公软件、文件系统、系统设置、多应用协作等常见电脑使用场景。论文还明确包含 30 个故意不可解任务，用来检查 Agent 是否能识别“这题本来就做不成”，而不是无条件硬做。OSWorld 官网在 2025 年 7 月 28 日升级到 OSWorld-Verified 后，仍沿用 369 个任务这一总规模，同时说明其中 8 个 Google Drive 相关任务在某些网络条件下可按官方规则排除，形成 361 题评测版本。

对初学者，可以把它想成 369 道电脑操作题。Agent 不是回答“我知道怎么做”，而是必须真的去看屏幕、点按钮、输入文字、切换应用、检查结果。每题最终只有两种判定：成功，或失败。把做成的题数除以总题数，就是成功率。

从公开资料看，早期通用桌面 Agent 与人类差距非常大。OSWorld 论文与官网给出的原始基线里，人类成功率约为 72.36%，当时最强模型约为 12.24%。后续公开报道中，Anthropic 在推出 computer use 时提到 Claude 3.5 Sonnet 在 OSWorld 上约为 14.9%；一些企业与媒体报道继续给出更高分数，例如 OSWorld-Verified 上约 72.5% 的说法，以及在更受控企业流程里约 94% 的结果。这些后者属于“后续版本或企业公开报道”，不能和论文原始基线直接混为一谈，但它们共同说明了一件事：桌面 Agent 的瓶颈不只是“看见按钮”，更在于长流程规划、跨应用联动和错误恢复是否稳定。

一个很典型的工程例子是保险、财务、政务协同流程。工作人员经常需要在政府门户、邮件附件、内部 ERP 三处来回切换：下载附件，抽取字段，登录内部系统，填写监管网页，再核对回写。这里真正难的不是单击某个按钮，而是中间任一步出错后，系统还能不能回到正确状态继续做。企业流程里分数往往更高，通常不是因为任务更难，而是因为流程更固定、界面更稳定、模板更重复、可加更多工程保护。benchmark 中大量失败，常见原因不是“完全不会点”，而是“点错一次后收不回来”。

| 对象 | 单次成功率 | 长流程表现 | 恢复能力 | 典型问题 |
|---|---:|---|---|---|
| 人类 | 72.36% 左右 | 稳定 | 强 | 主要受时间、疲劳、注意力影响 |
| GPT-4V 早期基线 | 12.24% 左右 | 明显下降 | 弱 | GUI grounding 弱、软件语义理解不足 |
| Claude 3.5 Sonnet | 14.9% 左右 | 有提升但仍明显落后 | 中等偏弱 | 长程断链、回退困难、状态跟踪不稳 |
| 企业定制 computer-use Agent | 特定流程可显著更高 | 取决于模板化程度 | 可通过工程补强 | 泛化范围有限、迁移到新流程成本高 |

一句话概括：OSWorld 测的不是“能不能自动点一次”，而是“能不能像一个稳定的电脑用户一样，把跨应用任务做完，并在出错后恢复”。

---

## 问题定义与边界

OSWorld 的边界非常明确：任务必须通过真实 UI 完成。这里的“真实 UI”指的不是直接调 API，也不是直接改数据库，而是像人一样使用电脑界面。比如打开浏览器、进入文件夹、复制一段内容、填写表单、切换窗口、修改系统设置、把一个应用里的结果写进另一个应用。

所以它测的是“电脑使用能力”，不是“纯文本问答能力”。一个在聊天里解释得很清楚的模型，到了桌面上仍然可能失败，因为桌面任务要求它同时具备下面四种能力：

| 维度 | 含义 | 为什么难 |
|---|---|---|
| GUI grounding | 把语言目标对应到屏幕上的具体元素 | 同样叫“保存”的按钮，在不同软件里位置、图标、层级都不同 |
| Operational knowledge | 知道软件通常怎样操作 | 不理解菜单结构和交互逻辑，就会乱点 |
| Workflow reasoning | 能把大任务拆成多步可执行流程 | 多应用任务漏掉一步，后续状态就全部偏掉 |
| Recovery | 出错后知道怎么恢复 | 点错之后，界面状态已经变化，不能假装一切没发生 |

其中，GUI grounding 可以白话理解成“把话里的目标，准确对准屏幕上的那个东西”。例如用户说“把附件另存为桌面”，Agent 至少要完成四次映射：

| 用户目标 | 屏幕上需要找到什么 | 常见失败点 |
|---|---|---|
| 找到附件 | 邮件列表、附件图标、下载按钮 | 没找到附件区域，误把正文当附件 |
| 打开保存菜单 | 右键菜单、更多操作、下载按钮 | 菜单层级理解错误 |
| 选择桌面路径 | 文件选择器、路径栏、收藏目录 | 路径树定位错误 |
| 确认保存完成 | 文件名、目录、提示信息 | 没验证文件是否真的落盘 |

OSWorld 的最基础指标仍然是成功率：

$$
SuccessRate = \frac{\text{完成任务数}}{369}
$$

例如，一个 Agent 完成了 45 个任务，那么：

$$
SuccessRate = \frac{45}{369} \approx 0.12195 \approx 12.2\%
$$

这正好接近论文里早期强模型的量级。

新手版理解方式很简单：把 OSWorld 想成 369 道电脑操作题。Agent 不能只说“我会”，而是必须真的做出来。做对一题记 1 分，做错或放弃记 0 分，最后总分除以 369。

从流程角度，可以把任务边界画成下面这样：

```text
用户指令
   |
   v
判断任务是否可解
   |------ 不可解 -> 正确拒绝 / 报告失败
   |
   v
读取当前屏幕与系统状态
   |
   v
规划下一步动作
   |
   v
执行点击 / 输入 / 拖拽 / 切换应用
   |
   v
检查是否达到当前子目标
   |------ 否 -> 重试 / 回滚 / 改计划
   |
   v
任务完成
```

这里故意不可解任务非常关键。真实世界里，并不是每个命令都能完成。按钮可能不存在，权限可能不够，网页可能改版，账号可能没有登录，网络也可能中断。一个不会判断“这题做不成”的 Agent，在生产环境里往往比一个保守一点的 Agent 更危险，因为它会继续操作，甚至在错误状态上制造新的错误。

因此，OSWorld 的真正问题定义可以写成下面这个更完整的形式：

$$
\text{DesktopAgent} = f(\text{观察}, \text{历史}, \text{目标}, \text{环境约束})
$$

其中：

- 观察：当前截图、可访问性树、OCR 结果、窗口信息
- 历史：前几步动作、上一步是否报错、之前提取出的字段
- 目标：用户最终要完成什么
- 环境约束：权限、网络、操作系统、应用版本、时间预算

也就是说，桌面 Agent 不是一次性的图像识别器，而是一个持续决策系统。

---

## 核心机制与推导

OSWorld 表面上只给一个 SuccessRate，但真正决定桌面 Agent 能不能用的，往往是多步任务中的过程稳定性。很多模型不是“第一步不会”，而是“第六步开始偏航”。前面几步看起来都对，最后仍然交不了卷。

### 1. 为什么长流程会把小错误放大

先看一个最简单的玩具推导。

假设一个任务需要连续完成 5 步，每一步都必须成功，单步成功概率是 $p$。如果把各步近似看成独立事件，那么整条任务链的成功概率大约是：

$$
P(\text{task success}) = p^5
$$

如果单步成功率是 0.9，那么整任务成功率是：

$$
0.9^5 \approx 0.59049
$$

如果单步成功率掉到 0.8，那么：

$$
0.8^5 = 0.32768
$$

只少了 0.1 的单步能力，整任务成功率却从约 59% 掉到约 33%。这就是桌面 Agent 常见的体验：前几步看起来挺像那么回事，但一到长流程就明显失稳。

更一般地，若任务长度为 $L$，则有：

$$
P(\text{task success}) \approx p^L
$$

这意味着：

- 当 $p$ 固定时，任务越长，成功率越低
- 当 $L$ 固定时，单步能力稍微提升，整任务成功率会明显上升
- 因此长流程评测比短任务更能拉开模型差距

### 2. 为什么“恢复能力”比“首步能力”更关键

上面的公式默认每一步要么成功，要么直接失败到底。但真实系统通常允许重试、回滚、改计划，所以更接近下面这种过程：

$$
P(\text{最终完成}) = P(\text{直接成功}) + P(\text{出错后恢复并成功})
$$

如果把“直接成功”记为 $p$，“出错但成功恢复”的概率记为 $r$，一个简化写法可以是：

$$
P(\text{step survives}) = p + (1-p)r
$$

那么一个长度为 $L$ 的任务，粗略上可以写成：

$$
P(\text{task success}) \approx \bigl[p + (1-p)r\bigr]^L
$$

这个式子虽然是简化模型，但它揭示了工程上非常重要的一点：提升恢复概率 $r$，有时比继续硬抬单步点击精度更划算。

举个数值例子。

如果单步直接成功率 $p = 0.8$，恢复成功率 $r = 0$，那么有效单步存活率仍是 0.8。  
若任务有 5 步：

$$
0.8^5 = 0.32768
$$

如果通过 checkpoint、状态机、可解性判断把恢复成功率提高到 $r = 0.5$，那么有效单步存活率变成：

$$
0.8 + (1-0.8)\times 0.5 = 0.9
$$

此时 5 步任务的成功率变成：

$$
0.9^5 \approx 0.59049
$$

也就是说，不改模型视觉能力，只增强恢复链路，整任务成功率都可能大幅改善。

### 3. 为什么需要多步得分而不只看最终成败

SuccessRate 适合做主榜单，但它有一个明显问题：只看最终成败，很难区分下面两类失败：

- 几乎做完，只差最后一步提交
- 一开始就走错，后面全是错误动作

所以工程上经常会补充一个“多步得分”或“子目标完成度”指标。一个常见的简化形式是：

$$
\text{MultiStepScore} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\text{task}_i \text{ 在预算步数内完成})
$$

如果想更细一点，也可以按子目标完成比例评分：

$$
\text{MultiStepScore} = \frac{1}{N}\sum_{i=1}^{N}\frac{\text{完成的子目标数}}{\text{总子目标数}}
$$

前者更像最终交卷分，后者更像过程得分。两者都重要：

| 指标 | 反映什么 | 适合回答的问题 |
|---|---|---|
| SuccessRate | 最终能否交卷 | 这个 Agent 能不能“做成”任务 |
| MultiStepScore | 中途走到了哪一步 | 它是“几乎成功”还是“完全跑偏” |
| 平均步数 / 平均重试数 | 效率与稳定性 | 它是不是靠大量试错才做成 |
| 不可解识别率 | 安全边界 | 它会不会在做不成时继续乱操作 |

### 4. 一个更贴近现实的任务链条

新手可以把桌面 Agent 理解成下面四个连续环节：

| 环节 | 它在做什么 | 失败会怎样 |
|---|---|---|
| 观察 | 看当前屏幕和环境状态 | 看错了，后面都偏 |
| 规划 | 决定下一步要干什么 | 想错了，会走弯路 |
| 执行 | 真正点击、输入、切换 | 手抖式失败，状态改变 |
| 校验与恢复 | 检查是否达到子目标，必要时回退 | 没有这一层，就会在错误状态上继续累错 |

下面是一段更完整的多步任务伪代码：

```text
for task in tasks:
    state = env.reset(task)
    checkpoints = []
    success = False

    for step in range(max_steps):
        obs = agent.observe(state)
        plan = agent.plan(obs, task.goal)
        action = agent.act(plan, obs)
        next_state, done, error = env.execute(action)

        if step % checkpoint_interval == 0:
            checkpoints.append(next_state.snapshot())

        if error:
            next_state = recover_from_checkpoint(checkpoints)
            continue

        if done and evaluator.pass(task, next_state):
            success = True
            break

        state = next_state
```

这段伪代码里，真正决定系统上限的，往往不是 `act` 本身，而是下面三件事：

1. 观察是否包含足够上下文，而不是只看一张局部截图。
2. 规划是否明确记录“当前做到第几段”，而不是每一步都重新猜。
3. 恢复是否能回到可信状态，而不是在错误页面上继续瞎试。

真实工程里，这种问题更明显。比如一个保险理赔录入流程，Agent 先下载附件，再提取保单号，再登录内部 ERP，再登录监管门户，再填写多个页面。只要中间一次字段映射错了，后续页面就会一起被污染。此时失败不是单点失败，而是整条状态链被污染。

所以，OSWorld 的核心机制可以浓缩成一句话：它不是在测“会不会操作某个按钮”，而是在测“面对真实电脑环境时，能否把观察、规划、执行、恢复四个环节连成一条稳定闭环”。

---

## 代码实现

实现一个最小可运行评测器，不需要真的驱动桌面，也可以先把“任务计分”“重试机制”“多步得分”这些基础逻辑做对。下面这段 Python 代码可以直接运行，演示的是一个简化版 OSWorld 风格评测框架：

- 用 `success_rate` 统计最终成功率
- 用 `multistep_score` 统计子目标完成比例
- 用 `run_task` 模拟“失败若干次后成功”或“预算耗尽仍失败”
- 用 `evaluate` 汇总任务结果并打印报告

```python
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TaskSpec:
    name: str
    fail_before_success: int
    max_retries: int
    subgoals_total: int


@dataclass
class TaskResult:
    name: str
    solved: bool
    retries_used: int
    attempts: int
    subgoals_done: int
    subgoals_total: int
    note: str


def success_rate(results: Iterable[TaskResult]) -> float:
    results = list(results)
    if not results:
        return 0.0
    solved = sum(1 for r in results if r.solved)
    return solved / len(results)


def multistep_score(results: Iterable[TaskResult]) -> float:
    results = list(results)
    if not results:
        return 0.0

    total = 0.0
    for r in results:
        if r.subgoals_total <= 0:
            raise ValueError(f"{r.name}: subgoals_total must be > 0")
        total += r.subgoals_done / r.subgoals_total
    return total / len(results)


def run_task(spec: TaskSpec) -> TaskResult:
    """
    模拟一个任务：
    - 前 fail_before_success 次尝试都失败
    - 若在 max_retries 允许范围内等到成功，则任务 solved=True
    - 否则预算耗尽，任务失败
    """
    max_attempts = spec.max_retries + 1  # 首次执行 + 最多重试次数

    for attempt in range(1, max_attempts + 1):
        current_retry = attempt - 1

        if current_retry >= spec.fail_before_success:
            return TaskResult(
                name=spec.name,
                solved=True,
                retries_used=current_retry,
                attempts=attempt,
                subgoals_done=spec.subgoals_total,
                subgoals_total=spec.subgoals_total,
                note="recovered and completed" if current_retry > 0 else "completed directly",
            )

    # 预算耗尽仍未成功：用一个简化规则模拟“部分子目标已完成”
    partial_done = max(0, spec.subgoals_total - 1)

    return TaskResult(
        name=spec.name,
        solved=False,
        retries_used=spec.max_retries,
        attempts=max_attempts,
        subgoals_done=partial_done,
        subgoals_total=spec.subgoals_total,
        note="retry budget exhausted",
    )


def evaluate(task_specs: Iterable[TaskSpec]) -> List[TaskResult]:
    return [run_task(spec) for spec in task_specs]


def print_report(results: Iterable[TaskResult]) -> None:
    results = list(results)

    print("Task report")
    print("-" * 72)
    for r in results:
        print(
            f"{r.name:28} solved={str(r.solved):5} "
            f"attempts={r.attempts:<2} retries={r.retries_used:<2} "
            f"subgoals={r.subgoals_done}/{r.subgoals_total} note={r.note}"
        )

    sr = success_rate(results)
    ms = multistep_score(results)

    print("-" * 72)
    print(f"success_rate    = {sr:.4f}")
    print(f"multistep_score = {ms:.4f}")


if __name__ == "__main__":
    tasks = [
        TaskSpec(
            name="open-browser-and-login",
            fail_before_success=0,
            max_retries=2,
            subgoals_total=3,
        ),
        TaskSpec(
            name="download-file-and-rename",
            fail_before_success=1,
            max_retries=2,
            subgoals_total=3,
        ),
        TaskSpec(
            name="fill-form-across-apps",
            fail_before_success=3,
            max_retries=2,
            subgoals_total=3,
        ),
    ]

    results = evaluate(tasks)

    assert len(results) == 3
    assert results[0].solved is True
    assert results[1].retries_used == 1
    assert results[2].solved is False

    sr = success_rate(results)
    ms = multistep_score(results)

    assert round(sr, 4) == 0.6667
    assert round(ms, 4) == 0.8889

    print_report(results)
```

运行后会得到这样的结果：

```text
Task report
------------------------------------------------------------------------
open-browser-and-login      solved=True  attempts=1  retries=0  subgoals=3/3 note=completed directly
download-file-and-rename    solved=True  attempts=2  retries=1  subgoals=3/3 note=recovered and completed
fill-form-across-apps       solved=False attempts=3  retries=2  subgoals=2/3 note=retry budget exhausted
------------------------------------------------------------------------
success_rate    = 0.6667
multistep_score = 0.8889
```

这段代码虽然没有真的去控制桌面，但已经把评测里几个最核心的概念理顺了：

| 组件 | 作用 | 在真实桌面 Agent 中对应什么 |
|---|---|---|
| `TaskSpec` | 描述任务预算与难度 | 任务配置、环境初始化脚本、评测元数据 |
| `TaskResult` | 记录最终结果 | 成功标记、日志、截图、动作轨迹、子目标进度 |
| `success_rate` | 统计最终交卷率 | 榜单主指标 |
| `multistep_score` | 统计过程完成度 | 辅助分析指标 |
| `max_retries` | 控制恢复预算 | 重试次数、回滚次数、人机接管阈值 |

如果把它映射到真实桌面 Agent，最小工程骨架通常是下面这样：

```text
加载任务索引
    ->
初始化虚拟机 / 桌面环境
    ->
循环执行任务
    ->
采集屏幕 + A11y tree + OCR + 历史轨迹
    ->
模型规划动作
    ->
执行鼠标 / 键盘操作
    ->
写入日志与截图
    ->
评估脚本判定成功 / 失败
    ->
汇总成功率、多步得分、错误类型
```

这里有两个新手最容易忽略的点。

第一，真实系统里“观察”通常不止截图。很多团队会同时使用：

| 观察来源 | 提供什么信息 | 优点 | 局限 |
|---|---|---|---|
| Screenshot | 真实视觉界面 | 直观、接近人类 | 文本小、遮挡多、定位难 |
| A11y tree | 控件结构、角色、文本 | 对按钮、输入框更稳定 | 某些应用支持差 |
| OCR | 从图像补文字 | 对截图中文本有帮助 | 易受分辨率和字体影响 |
| Window metadata | 窗口标题、应用名 | 便于判断当前在哪个应用 | 信息粒度有限 |

第二，checkpoint 模块尤其重要。checkpoint 就是“中途存档”。在桌面环境里，存档不一定是整个虚拟机快照，也可以是更轻量的状态组合，例如：

- 当前窗口标题
- 当前活动应用
- 最近一张截图
- 当前文件路径
- 已提取出的结构化字段
- 当前子目标编号
- 最近一次成功动作

目标不是百分之百恢复整个世界，而是恢复到“足以继续推理”的可信状态。

如果把这一层补上，一个更接近生产系统的伪代码会是这样：

```text
for task in tasks:
    state = env.reset(task)
    memory = []
    checkpoints = []

    for step in range(max_steps):
        obs = collect_observation(state)
        subgoal = planner.next_subgoal(task, obs, memory)
        action = policy.act(obs, subgoal, memory)
        next_state, status = env.execute(action)

        log(step, obs, subgoal, action, status)

        if reached(subgoal, next_state):
            memory.append(mark_done(subgoal))
            checkpoints.append(snapshot(next_state, memory))

        elif status.is_error:
            next_state = rollback(checkpoints)
            memory.append(record_failure(action, status))

        if evaluator.pass_(task, next_state):
            break

        state = next_state
```

新手可以把这套逻辑记成一句话：不要只写“让 Agent 一直点”，而要写“让 Agent 观察、规划、执行、验证、出错就回退”。

---

## 工程权衡与常见坑

桌面 Agent 的难点，不是把鼠标点下去，而是点错以后怎么办。只要任务链稍微变长，系统就会从“看起来能用”迅速暴露出大量工程问题。

最常见的问题是长程任务规划退化。所谓退化，就是随着步骤变长，原本还算合理的策略逐渐偏航。5 步以内可能不明显，但一旦跨应用、跨窗口、跨文件，错误会层层累积。

新手可以把它理解成做一张长表单。前两页都填对了，不代表第三页不会因为一个字段名理解错而全盘出错。桌面 Agent 本身就容易“局部合理、全局偏离”，所以状态记录和回退不是可选项，而是主能力。

下面是高频工程坑：

| 常见坑 | 表现 | 根因 | 应对策略 |
|---|---|---|---|
| 规划退化 | 前几步对，后几步乱 | 长上下文跟踪失败 | 把任务拆成子目标，显式记录“现在做到哪一步” |
| 误点控件 | 点错按钮、菜单、标签页 | GUI grounding 不稳 | 联合使用截图、A11y tree、OCR、操作历史 |
| 错误不可恢复 | 一步错导致整链失败 | 没有 checkpoint 或回滚点太少 | 固定间隔存档，关键节点强制校验 |
| 无限重试 | 同一错误反复发生 | 没有失败判别与退出条件 | 为每类错误设重试上限和降级路径 |
| 幻觉式继续执行 | 页面没有目标控件却硬做 | 可解性判断弱 | 加“找不到目标就停”的规则与人工接管 |
| 环境漂移 | 弹窗、更新提示、网络波动打断流程 | 桌面环境不稳定 | 用虚拟机快照、冻结版本、清理后台通知 |
| 日志不可审计 | 出错后无法定位原因 | 只存自然语言总结，没有结构化轨迹 | 同时保存动作、截图、控件信息、错误码 |
| 安全边界不清 | 误提交、误删除、误外发 | 没有高风险操作确认机制 | 提交类、支付类、权限类动作必须人工确认 |

对新手最有帮助的判断标准不是“模型分数高不高”，而是“系统出了错以后，是否知道自己错了，并且能不能安全停下”。在生产环境里，恢复能力往往比榜单分数更影响可用性。因为真实环境不会像 benchmark 一样总是干净开局：

- 登录态可能过期
- 窗口可能被遮挡
- 浏览器可能弹安全提示
- 下载文件可能改名
- 网络可能中断后重连
- 页面元素可能加载很慢

一个只在“完美初始状态”下工作的 Agent，几乎不能上线。

很多团队会在这里做三类补强。

### 1. 用有限状态机约束长任务

把长任务改写成“有限状态机”通常很有效。白话讲，就是不要让 Agent 每一步都自由发挥，而是把流程拆成几个可验证阶段：

```text
状态 S1: 下载附件
状态 S2: 提取关键信息
状态 S3: 登录内部系统
状态 S4: 填写外部门户
状态 S5: 提交前核对
```

每个状态只允许少数几类动作，并且必须有“进入条件”和“完成条件”。这样做的代价是灵活性下降，但稳定性显著提高。

### 2. 用结构化日志替代“自然语言自述”

很多 Agent 看起来有“思考过程”，但真正排障时几乎没用。因为日志里只有“我现在尝试点击保存按钮”，而没有记录：

- 点击的是哪个控件
- 坐标是多少
- 当时窗口标题是什么
- 上一步是否报错
- 当前处于哪个子目标
- 回滚点是否存在

工程上更有价值的日志通常长这样：

| 字段 | 示例 | 用途 |
|---|---|---|
| `task_id` | `fill-form-across-apps-023` | 定位任务 |
| `step_id` | `17` | 定位具体步骤 |
| `subgoal` | `open_download_folder` | 确认阶段 |
| `window_title` | `Downloads - File Manager` | 判断当前应用 |
| `action_type` | `double_click` | 审计动作 |
| `target_hint` | `invoice_march.pdf` | 复盘 grounding 是否错误 |
| `status` | `error:not_found` | 统计错误类型 |
| `checkpoint_id` | `ckpt_4` | 便于回滚与重放 |

### 3. 高风险动作必须有人类确认

桌面 Agent 很适合做“重体力”工作，但不适合无监督完成所有高风险动作。尤其是下面这些操作：

| 操作类型 | 风险 | 建议 |
|---|---|---|
| 最终提交报表 | 一旦提交难以撤回 | 提交前人工确认 |
| 付款、转账、下单 | 财务风险高 | 必须人工审批 |
| 删除文件、批量覆盖 | 容易造成数据损失 | 增加二次确认和恢复机制 |
| 权限变更 | 安全风险高 | 不建议纯自动执行 |

最后要强调一点：成功率低，不一定说明模型“不会操作电脑”；更常见的解释是，模型缺乏稳定的错误闭环。一次失误后，它既不知道错在哪里，也不知道该退回哪一步。桌面 Agent 真正稀缺的，不是“敢做”，而是“做错后还能收住”。

---

## 替代方案与适用边界

不是所有桌面自动化都应该上 OSWorld 风格 Agent。很多任务根本不需要那么重的系统。

如果任务短、规则固定、平台单一，传统 RPA 或脚本方案通常更便宜。RPA 可以理解成“按事先写好的路线机械重复”。例如只在 Windows 内部系统里录入固定字段，页面结构几乎不变，这时用固定控件定位、宏命令、规则引擎往往更稳，也更容易审计。

但如果任务跨平台、跨应用、页面经常变化、还要理解自然语言指令，那么规则脚本会迅速失效。这就是 OSWorld 这类基准存在的原因：它测的是“通用电脑使用能力”，而不是“某个页面的自动点按钮能力”。

| 方案 | 优点 | 缺点 | 适合任务 |
|---|---|---|---|
| OSWorld 风格通用 Agent | 泛化强，可跨应用跨平台 | 成本高，恢复难，安全要求高 | 开放式电脑操作、复杂工作流 |
| 单平台 RPA | 稳定、可审计、部署成熟 | 跨应用泛化差，页面一变就脆 | 固定流程、固定 UI、规则明确 |
| API / 数据库集成 | 最稳、速度快、可维护性好 | 前提是系统有接口 | 标准化系统对接、批量数据同步 |
| 人工操作 | 最灵活，异常处理最强 | 成本高、速度慢、不可规模化 | 高风险、低频、异常多的任务 |

实际选型时，先判断任务属于哪一类：

| 场景 | 建议 |
|---|---|
| 短文本录入、固定报表、单系统重复操作 | 优先考虑 API、脚本或单平台 RPA |
| 跨浏览器、跨操作系统、多应用协作 | 更适合通用桌面 Agent |
| 财务报表提交、合同签署、权限变更 | 必须加人工监督，不建议纯自动 |
| 老旧门户、无 API 的内部 ERP | 通用 computer-use Agent 价值最高 |
| 页面频繁改版但规则仍稳定 | 可先尝试半结构化自动化，再决定是否上通用 Agent |

新手可以用下面这个判断法：

- 能调 API，就不要点 UI。
- 能用脚本稳定完成，就不要上通用 Agent。
- 只有当任务确实跨系统、跨应用、无 API、界面还在变时，OSWorld 风格 Agent 才真正划算。

这一点也解释了为什么企业宣传里的高分常常高于 benchmark。企业场景并不一定更难，它往往只是更可控：

| 维度 | OSWorld 基准 | 企业固定流程 |
|---|---|---|
| 任务分布 | 开放、跨平台、跨应用 | 较固定、模板化 |
| 初始状态 | 更丰富、更不统一 | 常可标准化 |
| 页面变化 | 多样 | 相对可控 |
| 错误恢复 | 要求强泛化 | 可用规则工程补强 |
| 指标目标 | 测“通用能力” | 测“业务可用性” |

因此，现实中最稳妥的方案通常不是“完全自动化”，而是“Agent 负责重体力，人类负责最终确认”。尤其在高价值流程里，更合理的角色分工是：

- Agent 负责搜集信息、搬运数据、填写草稿、做格式转换
- 人类负责审核、确认、提交、承担责任

这比追求“全自动无人值守”更符合当前桌面 Agent 的真实能力边界。

---

## 参考资料

1. [OSWorld 官网](https://os-world.github.io/)  
重点：官方任务定义、369 个任务、原始 benchmark 描述，以及 2025 年 7 月 28 日升级为 OSWorld-Verified 的说明。官网还注明 8 个 Google Drive 任务在某些网络条件下可排除，形成 361 题评测版本。

2. [OSWorld 论文（arXiv:2404.07972）](https://arxiv.org/abs/2404.07972)  
重点：基准设计、30 个不可解任务、人类约 72.36%、当时最强模型约 12.24% 的原始基线，以及失败模式分析。原始论文是理解 OSWorld 定义边界的第一来源。

3. [Anthropic computer use 文档](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)  
重点：computer use 工具的工作方式、截图与鼠标键盘动作接口、推荐分辨率、虚拟机隔离与安全注意事项。它不是 OSWorld 论文来源，但适合理解“computer use 系统在工程上怎样落地”。

4. [WIRED 对 Anthropic computer-use agent 的报道](https://www.wired.com/story/anthropic-ai-agent/)  
重点：给出 Claude 3.5 Sonnet 在 OSWorld 上约 14.9% 的公开报道，并强调瓶颈不只是识别界面，而是长程规划与错误恢复。该来源属于媒体报道，不等同于论文原始实验。

5. [TechBuddies 关于 Sonnet 4.6 与企业 computer use 的报道](https://www.techbuddies.io/2026/02/18/anthropics-claude-sonnet-4-6-opus-level-intelligence-at-one-fifth-the-cost-reshapes-enterprise-ai-economics/)  
重点：整理了 OSWorld-Verified 72.5% 与保险场景 94% 的公开说法，适合用来理解 benchmark 与企业流程之间的差别。需要注意，这类数据属于企业或媒体公开报道，不应与 OSWorld 论文原始基线直接并列比较。

6. [EmergentMind 对 OSWorld 的综述](https://www.emergentmind.com/topics/osworld-benchmark)  
重点：把 GUI grounding、workflow reasoning、长流程退化等概念放到同一语境里，适合补充阅读。它更适合作为二级解释材料，而不是一手实验来源。
