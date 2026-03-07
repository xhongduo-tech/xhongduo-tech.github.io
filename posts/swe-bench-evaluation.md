## 核心结论

SWE-bench 可以理解为“让模型像工程师一样修 GitHub 真实问题”的评测。它给模型的不是一道脱离上下文的函数题，而是一个真实 issue、一份修复前的代码仓库快照，以及一个最终要在隔离环境里验证的测试标准。模型需要输出可应用的补丁，评测系统再把补丁打到仓库上执行测试。只有目标问题对应测试从失败变为通过，同时原有功能不回归，这个实例才算真正解决。

它难，不是因为题目偏门，而是因为任务形态接近真实开发。模型通常要先理解 issue 在说什么，再决定读哪些文件、定位哪条调用链、改动哪些模块、是否要扩展上下文、是否要通过测试反馈修正假设。很多实例的失败原因并不直接写在 issue 描述里，而是埋在仓库结构、历史接口约束、边界条件和隐藏测试里。

官方原始版本覆盖 12 个 Python 开源仓库、2,294 个实例。Lite 版本抽取 300 个代表性实例，目标是降低运行成本、提高迭代速度，同时尽量保留任务分布。早期基线很低，说明“只会生成代码”远远不够；后续 agent 系统分数持续上升，则说明工具调用、仓库检索、测试反馈和多轮修复闭环确实能提升真实修复能力。

一个直接判断是：如果一个系统在 SWE-bench 上提升，通常不只是“写代码能力”更强，而是“读仓库、定位根因、生成最小补丁、调用工具验证、避免回归”的整套工程能力更强。

| 版本/系统 | 任务规模 | 公开结果 | 解读 |
| --- | ---: | ---: | --- |
| SWE-bench Full | 2,294 | 初始 RAG 1.96% | 只靠简单检索加生成，几乎无法稳定修复真实仓库问题 |
| SWE-bench Full | 2,294 | SWE-agent 12.47% | 具备搜索、编辑、执行等 agent 闭环后，能力明显提升 |
| SWE-bench Lite | 300 | OpenHands CodeAct 2.1 41.7% | Lite 更适合高频迭代，但高分仍不等于“可稳定自动修库” |

这里的 resolve rate 可以写成：

$$
\text{Resolve Rate}=\frac{\sum_{i=1}^{N}\mathbf{1}(\text{instance}_i\ \text{resolved})}{N}
$$

其中 $N$ 是实例总数，$\mathbf{1}(\cdot)$ 是指示函数，条件满足记为 1，不满足记为 0。

但“resolved”不是“补丁能编译”就够，而是至少同时满足两类测试要求：

$$
\text{Resolved}=\mathbf{1}(\text{Patch Applies})
\cdot
\mathbf{1}(\text{Fail-to-Pass All Pass})
\cdot
\mathbf{1}(\text{Regression All Pass})
$$

这也是 SWE-bench 与普通代码题的根本区别。它测的不是“能不能写出一段像样的代码”，而是“能不能在真实代码库中交付一个可验证、可上线的修复”。

---

## 问题定义与边界

先把几个核心术语定义清楚。

`issue`：GitHub 上的问题描述，通常是 bug 报告、行为不一致说明，或者一个明确的修复请求。  
`patch`：补丁，也就是对仓库中文件的一组改动，通常以 diff 的形式表达。  
`fail-to-pass tests`：修复前失败、修复后应该通过的测试，用来证明“问题真的被修好”。  
`regression tests`：修复前后都应该通过的测试，用来证明“修复没有把其他能力改坏”。  
`repo snapshot`：仓库在某个特定提交点上的快照，评测要求所有系统都从同一个起点开始修。

一个 SWE-bench 实例最核心的输入通常有三部分：

| 输入对象 | 作用 | 模型是否可见 |
| --- | --- | --- |
| issue 描述 | 告诉系统要解决什么问题 | 可见 |
| 修复前仓库快照 | 提供全部代码、目录结构和历史约束 | 可见 |
| 测试集合 | 判断补丁是否正确 | 通常只在评测时执行，细节不完全可见 |

输出则只有一个核心对象：补丁。  
评测系统不关心模型中间说了多少分析，也不关心解释写得是否漂亮。最终只看补丁是否能成功应用，以及应用后测试是否满足判定标准。

完整流程可以概括为：

1. 读取 issue 和仓库快照。
2. 生成一个或多个候选补丁。
3. 将补丁应用到修复前仓库。
4. 在隔离环境中安装依赖并运行测试。
5. 若目标测试全部转绿且无回归，则该实例记为 `resolved`。

为什么要使用 Docker 或等价的隔离环境？因为评测要尽量固定系统依赖、解释器版本、命令执行方式和外部环境。如果不隔离，两个系统即使输出完全相同的补丁，也可能因为本地环境不同得到不同结果。隔离环境的作用不是“增加难度”，而是“保证结果可复现、可比较”。

这里还要明确几个边界。

第一，SWE-bench 主要覆盖真实 Python 仓库任务。  
这意味着它非常适合衡量“在真实后端或库项目里修 bug”的能力，但不能直接代表前端视觉调整、产品需求拆解、跨语言迁移、多人协作流程等完整软件工程能力。

第二，它评测的是“给定 issue 的仓库修复能力”，不是从零构建功能。  
系统通常是在已有代码库上做局部修复、兼容和验证，而不是设计一个全新系统。

第三，它默认“测试是裁判”，但测试本身也可能不完美。  
有些实例里，功能上正确的修复可能因为测试覆盖方式、边界设定或样例假设而被拒绝；这也是后续围绕 Verified、Live、Pro 等变体持续讨论的原因。

一个最小玩具例子可以帮助初学者理解边界。假设 issue 写的是：

> `sqrt_if_perfect_square(-0.0)` 返回了错误的字符串表示。

这句话表面上像是“改一个函数”。但进入真实仓库后，你可能会发现问题分散在三个位置：

| 位置 | 可能问题 | 为什么单改一处不够 |
| --- | --- | --- |
| 数值归一化 | `-0.0` 被错误视为负数 | 会直接影响分支选择 |
| 格式化层 | 输出保留了错误符号位 | 即使计算正确，最终字符串仍然错误 |
| 兼容接口 | 旧 API 期望某种返回类型 | 强行改返回值可能破坏其他调用方 |

这就是 SWE-bench 真正考的能力边界：不是“会不会写 `if`”，而是“能不能在真实代码上下文里做最小、正确、兼容的工程改动”。

---

## 核心机制与推导

SWE-bench 的核心机制可以压缩成一条闭环：

| 步骤 | 系统动作 | 目的 |
| --- | --- | --- |
| 1 | 读取 issue 和仓库 | 建立问题假设 |
| 2 | 搜索相关文件、测试、调用链 | 缩小定位范围 |
| 3 | 生成候选补丁 | 提出可执行修复 |
| 4 | 应用补丁 | 验证改动能否落地 |
| 5 | 在隔离环境中运行测试 | 检查是否真的修好且无回归 |
| 6 | 根据结果决定成功或继续迭代 | 形成工程闭环 |

这个机制之所以有效，是因为它同时堵住了两类常见的“投机解法”。

第一类投机是“只让某个断言暂时消失”。  
例如，模型可能删除一段检查、返回一个硬编码值，甚至通过更宽松的异常处理把错误吞掉。这样做有时能让单个目标行为看上去恢复正常，但回归测试往往会失败，因为项目其他路径仍然依赖原有接口语义。

第二类投机是“生成一段看起来像 patch 的文本”。  
SWE-bench 不按语言流畅度给分，也不按 diff 长短给分。补丁如果打不上、依赖装不起来、测试起不来、结果不稳定，最后都是 0 分。

因此，SWE-bench 的成功条件更接近工程上的合取判定，而不是开放式主观评分：

$$
\text{resolved}_i =
\begin{cases}
1, & \text{if } A_i \land F_i \land R_i \\
0, & \text{otherwise}
\end{cases}
$$

其中：

- $A_i$ 表示第 $i$ 个实例的补丁可以成功应用；
- $F_i$ 表示该实例所有 fail-to-pass 测试全部通过；
- $R_i$ 表示该实例所有 regression 测试全部通过。

对全数据集求平均，就得到 resolve rate：

$$
\text{RR} = \frac{1}{N}\sum_{i=1}^{N}\text{resolved}_i
$$

如果某个系统在 300 个 Lite 实例中解决了 125 个，那么：

$$
\text{RR}=\frac{125}{300}=41.67\%
$$

这类计算很简单，但它隐含了一个重要性质：SWE-bench 是实例级二值记分，不是测试级部分得分。  
也就是说，只要有一个关键 regression 测试失败，这个实例通常仍然记为未解决。

下面用一个最小数值例子说明。

假设某实例有 2 个 fail-to-pass 测试和 3 个 regression 测试：

| 测试类型 | 数量 | 通过数量 | 该实例是否 resolved |
| --- | ---: | ---: | --- |
| fail-to-pass | 2 | 2 | 还不能下结论 |
| regression | 3 | 2 | 否 |

这时很多新手会误以为“总共 5 个测过了 4 个，应该算 80% 成功”。  
SWE-bench 不是这样计分。它更像上线前的工程验收：只要修复引入回归，这个补丁就不能算合格修复。

为什么这一点重要？因为真实仓库修复的难点，恰恰不在“让一个输入变对”，而在“让这个输入变对，同时不破坏整套系统的不变量”。

进一步看，一个强系统在 SWE-bench 上通常需要经过下面的推理链：

1. 从 issue 中提取显式症状。
2. 从目录结构和调用链推断可疑模块。
3. 通过阅读测试名、函数签名和错误栈形成根因假设。
4. 生成最小补丁，尽量少改与问题无关的部分。
5. 用测试反馈验证假设是否成立。
6. 若失败，再基于失败报告调整定位或补丁范围。

这也是为什么 agent 系统通常优于单轮一次性生成。后者往往只能“猜一次答案”；前者至少具备“观察 -> 假设 -> 执行 -> 反馈 -> 再修正”的基本工程循环。

可以把这个过程抽象成下面的策略：

$$
\text{Next Action} = \pi(\text{issue}, \text{repo context}, \text{tool outputs}, \text{test feedback})
$$

这里的 $\pi$ 表示策略。一个优秀的 agent，不只是代码生成器，它还是“下一步该看什么、改什么、验什么”的决策器。

---

## 代码实现

下面给出一个可运行的极简 Python 版本，用来模拟 SWE-bench 的评测闭环。它不是官方 harness，也不涉及真实 Docker、真实仓库和真实 patch 解析，但它完整表达了三个核心条件：

1. 补丁必须能应用。
2. 目标问题对应测试必须全部转绿。
3. 原有能力不能回归。

```python
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvalResult:
    patch_applied: bool
    fail_to_pass_passed: int
    fail_to_pass_total: int
    regression_passed: int
    regression_total: int

    @property
    def resolved(self) -> bool:
        return (
            self.patch_applied
            and self.fail_to_pass_passed == self.fail_to_pass_total
            and self.regression_passed == self.regression_total
        )


def apply_patch(repo_state: Dict[str, bool], patch: str) -> bool:
    # 用极简规则模拟“补丁能否落地”。
    # 如果补丁里包含 BREAK_PATCH，表示 diff 无法正确应用。
    if "BREAK_PATCH" in patch:
        return False

    # 这两个开关分别模拟“修到根因”和“保留兼容性”。
    repo_state["handle_negative_zero"] = "handle_negative_zero" in patch
    repo_state["preserve_old_api"] = "preserve_old_api" in patch
    return True


def run_fail_to_pass_tests(repo_state: Dict[str, bool]) -> List[bool]:
    # 两个目标测试都依赖根因是否被修正。
    fixed = repo_state["handle_negative_zero"]
    return [fixed, fixed]


def run_regression_tests(repo_state: Dict[str, bool]) -> List[bool]:
    # 三个回归测试中，后两个依赖兼容接口是否保留。
    return [
        True,
        repo_state["preserve_old_api"],
        repo_state["preserve_old_api"],
    ]


def evaluate_patch(issue: str, patch: str) -> EvalResult:
    repo_state = {
        "handle_negative_zero": False,
        "preserve_old_api": False,
    }

    patch_applied = apply_patch(repo_state, patch)

    if not patch_applied:
        return EvalResult(
            patch_applied=False,
            fail_to_pass_passed=0,
            fail_to_pass_total=2,
            regression_passed=0,
            regression_total=3,
        )

    fail_to_pass_results = run_fail_to_pass_tests(repo_state)
    regression_results = run_regression_tests(repo_state)

    return EvalResult(
        patch_applied=True,
        fail_to_pass_passed=sum(fail_to_pass_results),
        fail_to_pass_total=len(fail_to_pass_results),
        regression_passed=sum(regression_results),
        regression_total=len(regression_results),
    )


def print_report(name: str, result: EvalResult) -> None:
    print(f"== {name} ==")
    print(f"patch_applied      : {result.patch_applied}")
    print(
        "fail_to_pass       : "
        f"{result.fail_to_pass_passed}/{result.fail_to_pass_total}"
    )
    print(
        "regression         : "
        f"{result.regression_passed}/{result.regression_total}"
    )
    print(f"resolved           : {result.resolved}")
    print()


if __name__ == "__main__":
    issue = "sqrt_if_perfect_square(-0.0) returns the wrong representation"

    bad_patch = "handle_negative_zero"
    good_patch = "handle_negative_zero\npreserve_old_api"
    broken_patch = "BREAK_PATCH\nhandle_negative_zero\npreserve_old_api"

    r1 = evaluate_patch(issue, bad_patch)
    r2 = evaluate_patch(issue, good_patch)
    r3 = evaluate_patch(issue, broken_patch)

    print_report("bad_patch", r1)
    print_report("good_patch", r2)
    print_report("broken_patch", r3)

    assert r1.patch_applied is True
    assert r1.fail_to_pass_passed == 2
    assert r1.regression_passed == 1
    assert r1.resolved is False

    assert r2.patch_applied is True
    assert r2.fail_to_pass_passed == 2
    assert r2.regression_passed == 3
    assert r2.resolved is True

    assert r3.patch_applied is False
    assert r3.resolved is False
```

这段代码可以直接运行。它表达的是一个最小但完整的判定流程。

先看 `bad_patch`。  
它确实修到了根因，所以两个 fail-to-pass 测试都通过；但它没有保留旧接口兼容性，因此 regression 只有 1/3 通过，最终 `resolved=False`。

再看 `good_patch`。  
它同时满足“修到根因”和“保留旧接口”，于是目标测试和回归测试全部通过，实例才真正被记为解决。

最后看 `broken_patch`。  
哪怕补丁文本里包含了正确意图，只要补丁本身无法应用，评测直接失败。这一点和真实 SWE-bench 很像：思路正确但 diff 落不了地，工程上仍然等于没交付。

如果把这个玩具版本映射到真实工程，一个 agent 的工作流更接近下面这样：

```python
def agent_loop(issue_text, repo_snapshot, tools):
    suspects = tools.search_repo(issue_text, repo_snapshot)
    files = tools.read_files(suspects)
    hypothesis = tools.infer_root_cause(issue_text, files)
    patch = tools.generate_patch(hypothesis)

    applied = tools.apply_patch(repo_snapshot, patch)
    if not applied:
        return {"status": "unresolved", "reason": "patch_not_applied"}

    report = tools.run_tests_in_docker(repo_snapshot)
    if report.fail_to_pass_all_green and report.regression_all_green:
        return {"status": "resolved", "patch": patch}

    return {"status": "unresolved", "report": report}
```

这段伪代码里，真正关键的不是 `generate_patch` 一行，而是前后两端：

| 环节 | 为什么关键 |
| --- | --- |
| `search_repo` | 决定模型是否能在大仓库里先缩小搜索空间 |
| `read_files` | 决定模型看到的是根因相关代码，还是无关噪声 |
| `infer_root_cause` | 决定改动是治标还是治本 |
| `apply_patch` | 决定输出是否真能落到仓库 |
| `run_tests_in_docker` | 决定系统能否从反馈中判断补丁质量 |

因此，SWE-bench 不是单纯比较“谁的补丁文本更像人写的”，而是在比较一整套修复系统的工程闭环是否成立。

对初学者来说，一个很实用的理解方式是把 SWE-bench 看成“受约束的自动 debug 系统”：

- 输入不是题面，而是 issue 加仓库。
- 目标不是写新函数，而是修复现有系统。
- 裁判不是人工主观评分，而是测试。
- 成功标准不是部分对，而是实例级通过。

---

## 工程权衡与常见坑

SWE-bench 的价值很高，但它不是完美评测。真正拿它做系统优化时，至少要理解下面几类工程权衡。

| 常见坑 | 本质问题 | 典型后果 | 实用规避思路 |
| --- | --- | --- | --- |
| issue 描述含糊 | 用户报告的是症状，不是根因 | 修了表层，未修到底层逻辑 | 先看调用链、异常栈、相关测试命名 |
| 仓库过大 | 搜索空间高，注意力容易浪费 | 读错模块，补丁偏题 | 先缩目录，再扩函数，再扩依赖关系 |
| 测试不可见 | 真实约束部分隐藏 | 只按 issue 猜，漏掉兼容条件 | 通过接口签名、文档、旧行为推断隐含约束 |
| 改动过大 | 一次性重构太多 | 回归测试大面积失败 | 优先最小补丁，只动问题必要路径 |
| 多轮编辑漂移 | 每轮假设不一致 | 越改越乱，前后 patch 互相抵消 | 每轮明确记录当前根因假设 |
| 只追求目标测试 | 忽视系统其他路径 | fail-to-pass 变绿，但 regression 失败 | 每次修改都问“谁会被这个改动影响” |
| 工具使用失控 | 搜索、读文件、测试顺序混乱 | 上下文爆炸，成本升高 | 先定位，后读取，再修改，最后验证 |

一个容易被忽略的问题是：SWE-bench 分数高，不自动等于“生产环境可直接替代工程师”。  
原因至少有三类。

第一，评测实例是有限分布。  
即使 Full 覆盖 2,294 个任务，它仍然只是现实软件工程空间中的一个样本，不可能覆盖所有语言、框架、组织流程和发布约束。

第二，测试是代理指标，不是真实用户体验本身。  
如果测试写得不完整，系统有可能“通过测试但仍不理想”；反过来，如果测试假设过于僵硬，功能上合理的修复也可能被拒绝。

第三，公开数据可能带来污染问题。  
当模型或系统直接或间接见过相关 issue、提交记录、训练样本或榜单调参经验时，结果就未必只反映“泛化修复能力”。

这也是为什么围绕 SWE-bench Verified 的讨论后来转向两个核心批评：

1. 部分测试设计可能拒绝功能上正确的解。
2. 公开样本与生态传播可能带来数据污染。

因此，看 SWE-bench 成绩时，至少要同时问三个问题：

1. 是哪个 split？Full、Lite、Verified、Live、Pro 不能混着比。
2. 系统配置是什么？模型、agent 框架、工具权限、重试轮数、采样策略都会影响结果。
3. 目标是什么？是追求高频迭代、论文对比、还是更接近真实时间线的评测？

如果你自己做 agent，下面这组工程原则通常更有用：

| 原则 | 含义 | 为什么有效 |
| --- | --- | --- |
| 先定位再生成 | 先缩小相关代码范围，再写补丁 | 降低无关上下文干扰 |
| 先最小修复再扩展 | 先解决主症状，再处理兼容边界 | 降低引入回归的概率 |
| 用测试做反馈，不把测试当答案 | 测试报告用于修正假设，而不是机械迎合 | 更接近真实工程调试 |
| 保留假设链 | 明确“为什么改这里” | 便于多轮修复保持一致性 |

---

## 替代方案与适用边界

SWE-bench 现在已经不是一个单一数据集，而是一组面向不同目标的评测变体。对初学者最重要的不是把所有名字背下来，而是知道“不同 split 在解决什么问题”。

| 变体 | 规模/特点 | 适用场景 | 主要边界 |
| --- | --- | --- | --- |
| Full | 2,294 个真实任务，覆盖最完整 | 论文主结果、系统全面评估 | 成本高、运行慢、调试周期长 |
| Lite | 300 个代表性任务 | 日常回归、快速比较不同 agent 配置 | 覆盖面小于 Full |
| Verified | 500 个人工筛选任务 | 早期用于更稳定地评估“可解样本” | 后续被指出存在测试与污染局限 |
| Live | 持续加入更新鲜任务 | 更接近真实时间线，降低记忆污染影响 | 基础设施更复杂，对比成本更高 |
| Pro | 面向更高质量前沿评测 | 前沿模型和强 agent 的竞争评估 | 生态和公开材料仍在快速变化 |

如果你的目标是做 agent 系统开发，比较实用的路径通常是：

1. 本地调参阶段优先使用 Lite，因为它便宜、快，适合频繁回归。
2. 系统接近稳定后再跑 Full，用来检查提升是否只是 Lite 特例。
3. 如果你更关心“新鲜数据上的泛化”，就要关注 Live、Pro 一类更新的评测。

如果你的目标是“读榜单”，则要防止三种误读。

第一，不同 split 的百分比不能直接横向比较。  
例如，Lite 上的 40% 并不自动等于 Full 上的 40%，因为任务规模、样本分布和筛选标准不同。

第二，不同系统配置下的分数不能脱离设置去理解。  
同一个底座模型，在单轮生成、带检索、带 shell 工具、带多轮回滚策略下，结果可能差出很多。

第三，SWE-bench 只能代表“真实工程能力中的一个高价值切片”。  
它尤其擅长衡量：

| 更适合衡量的能力 | 原因 |
| --- | --- |
| 仓库内定位问题 | 输入天然包含真实代码上下文 |
| 基于测试反馈迭代修复 | 判定机制与工程闭环一致 |
| 做最小兼容补丁 | regression 测试会约束激进改动 |
| 工具调用与多轮策略设计 | 强系统通常依赖搜索、执行、验证能力 |

但它不直接衡量下面这些能力：

| 不直接衡量的能力 | 原因 |
| --- | --- |
| 前端视觉设计和交互体验 | 这类任务通常没有统一测试裁判 |
| 多语言跨仓迁移 | 数据集主体仍以 Python 仓库为主 |
| 长周期团队协作 | 缺少多人协同、需求变更、代码评审流程 |
| 产品需求抽象和优先级决策 | 评测目标更像“修复已定义问题” |

换句话说，SWE-bench 很强，但它测的是“模型能否在真实代码库中完成可验证修复”这件事，而不是全部软件工程能力。

---

## 参考资料

| 资源 | 链接 | 核心贡献 | 建议阅读方式 |
| --- | --- | --- | --- |
| SWE-bench 官方原始介绍 | https://www.swebench.com/original.html | 介绍原始数据集设定、实例来源和 Full 基线 | 先读任务定义，再看基线数字，不要只看榜单 |
| SWE-bench Lite 官方页面 | https://www.swebench.com/lite.html | 说明 Lite 的 300 个代表性任务及其定位 | 如果你要低成本回归，优先看这一页 |
| SWE-bench 官方文档总览 | https://www.swebench.com/SWE-bench/ | 汇总数据集、安装、评测入口和不同 split | 用来建立全局地图，区分 Full/Lite/Verified/Live/Pro |
| SWE-bench Harness 文档 | https://www.swebench.com/SWE-bench/reference/harness/ | 解释补丁应用、环境构建和测试执行机制 | 阅读时重点看隔离环境和执行流程 |
| SWE-bench Verified 官方页面 | https://www.swebench.com/verified.html | 说明 Verified 的人工筛选思路与目标 | 用来理解为什么后来会讨论“更稳定样本” |
| OpenHands CodeAct 2.1 | https://openhands.dev/blog/openhands-codeact-21-an-open-state-of-the-art-software-development-agent | 给出 Lite 上较强 agent 的公开结果和系统设计线索 | 不只看分数，也看工具链和 rollout 设置 |
| OpenAI: Introducing SWE-bench Verified | https://openai.com/index/introducing-swe-bench-verified/ | 介绍 Verified 的背景和人工审核价值 | 适合补充“为什么需要更干净的可解样本” |
| OpenAI: Why SWE-bench Verified no longer measures frontier coding capabilities | https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ | 讨论测试拒绝正确解、数据污染等局限 | 用来理解为什么不能把单一分数当作最终答案 |
| SWE-bench 论文页面 | https://openreview.net/forum?id=VTF8yNQM66 | 提供原始论文、方法、数据构造和学术引用入口 | 需要系统理解时，从论文入手最完整 |
