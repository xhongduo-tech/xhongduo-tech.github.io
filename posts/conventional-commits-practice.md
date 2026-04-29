## 核心结论

`Conventional Commits` 是一套提交信息规范，但更准确地说，它是一种**可解析协议**。可解析协议的意思是：不仅人能看懂，工具也能稳定读懂。它把“这次改动是什么类型、影响范围多大、是否有破坏性”编码进 commit message，直接服务于自动发版、版本号计算、changelog 生成和发布审计。

它解决的不是“历史记录好不好看”这种表层问题，而是工程流程中的确定性问题。团队如果希望从 Git 历史里自动推导版本变化，就必须让提交信息具备统一结构。最常见的映射是：

| 提交类型 | 含义 | 常见版本影响 |
|---|---|---|
| `fix` | 修复缺陷 | `patch` |
| `feat` | 新增功能 | `minor` |
| `BREAKING CHANGE` 或 `!` | 破坏性变更 | `major` |
| `docs` / `style` / `chore` 等 | 文档、格式、杂项 | 通常不发版 |

这里的 `patch`、`minor`、`major` 对应语义化版本 `SemVer`。语义化版本就是把版本号拆成 `主版本.次版本.修订号`，例如 `1.4.2`。

玩具例子很直接：

- `feat(search): add fuzzy match`
- `fix(auth): handle expired token`
- `docs: update readme`

如果当前版本是 `1.4.2`，这三条里最高级别是 `feat`，最终版本通常升到 `1.5.0`，而不是 `1.4.3`。规则不是“看最后一条 commit”，而是“看本次发布范围内最高等级的变更”。

---

## 问题定义与边界

先定义问题。没有提交规范时，团队常见的 commit message 是：

- `update`
- `fix bug`
- `change api`
- `wip`

这种写法的问题不在于“不专业”，而在于**信息不可判定**。不可判定的意思是：你既无法稳定知道这是新功能还是 bug 修复，也无法让工具自动决定版本号该怎么变。结果就是每次发版都要人工阅读历史、人工整理说明、人工判断风险，流程慢，而且容易错。

`Conventional Commits` 的边界也要说清楚。它只负责表达**变更意图**和**影响级别**，不负责承载全部工程信息。比如测试报告、设计背景、回滚方案、上线窗口，这些更适合放在 PR 描述、Issue、变更单或发布单里，不应该全塞进 commit message。

适用范围可以直接看表：

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 自动发版 | 非常适合 | 机器可直接推版本号 |
| 多人协作仓库 | 非常适合 | 统一语义，降低沟通成本 |
| 有 changelog 要求 | 非常适合 | 能按类型自动生成 |
| 纯个人实验仓库 | 可选 | 自动化收益较低 |
| 完全手工发布的小脚本仓库 | 价值较低 | 流程简单，规范成本可能高于收益 |

真实工程里，是否值得上这套规范，不是看“团队喜不喜欢整齐”，而是看发布流程是否需要自动化。如果一个仓库已经接入 CI、自动测试、自动打 tag、自动出 release note，那么提交规范基本不是可选项，而是自动化链路的输入约束。

---

## 核心机制与推导

`Conventional Commits` 常见格式是：

```text
<type>(<scope>): <description>

<body>

<footer>
```

这里的 `type` 是变更类型，`scope` 是作用域，也就是这次改动主要影响哪个模块；`description` 是简短摘要；`body` 用来补充为什么改；`footer` 常放破坏性说明、Issue 编号或回退信息。

结构可以概括成下面这张表：

| 字段 | 作用 |
|---|---|
| `header` | 声明类型、作用域、摘要 |
| `body` | 解释改动原因和关键实现 |
| `footer` | 破坏性变更、关联 Issue、兼容性说明 |

工具之所以能自动发版，是因为它不是在“理解自然语言”，而是在读取结构化信号。最核心的推导可以写成：

$$
R(fix)=patch,\quad R(feat)=minor,\quad R(BREAKING\ CHANGE)=major,\quad R(other)=none
$$

如果一次发布范围内有多个 commit，那么最终发布级别不是求和，而是取最高优先级：

$$
release=\max_i R(commit_i), \quad major > minor > patch > none
$$

玩具例子：

当前版本是 `1.4.2`，最近一次 tag 之后有三条提交：

1. `docs: update readme`
2. `fix(auth): handle expired token`
3. `feat(search): add fuzzy match`

映射后分别是：

- `docs -> none`
- `fix -> patch`
- `feat -> minor`

因此最终结果是 `minor`，新版本为 `1.5.0`。

再看一个边界更清晰的例子：

- `feat(api)!: change response schema`

这里的 `!` 是破坏性变更标记。破坏性变更的意思是：旧调用方如果不改代码，升级后可能直接出错。比如字段名变了、返回结构变了、接口删了。即使它表面上也是 `feat`，版本仍应按 `major` 处理。

真实工程例子更能说明机制。假设一个支付服务仓库有如下流程：

- 开发者提交代码，本地 `commit-msg` hook 校验格式
- PR 合并前，CI 再扫一遍本分支所有 commit
- `main` 分支合并后，发布作业读取“最近 tag 之后”的 commit 列表
- 发布工具计算最高版本级别，生成 changelog，打 tag，发包

这里的输入和输出非常明确：

| 环节 | 输入 | 输出 |
|---|---|---|
| 本地提交 | 单条 commit message | 是否允许提交 |
| PR 校验 | 分支内所有 commits | 是否允许合并 |
| 发布任务 | 最近 tag 后的 commits | 新版本号、release notes、tag |
| 回滚判断 | 发布记录 + commit 语义 | 是否可快速定位风险变更 |

所以提交规范不能孤立讨论。它必须放进完整工程流程里看：输入是什么，谁消费它，失败时如何阻断，发布后如何回溯。

---

## 代码实现

实践里通常做两层校验。

第一层是本地 hook。`hook` 就是 Git 在特定时机自动执行的脚本。这里用 `commit-msg` hook，在用户输入提交信息后立刻校验格式。第二层是 CI 校验，防止有人用 `--no-verify` 跳过本地检查。

常见最小配置如下。

```js
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
};
```

```sh
# .husky/commit-msg
npx --no -- commitlint --edit "$1"
```

```yaml
# CI job
- name: Validate commits
  run: npx commitlint --from origin/main --to HEAD
```

如果还要自动发版，发布流程通常是：

```text
读取最近 tag
扫描 tag 之后的 commits
计算最高版本级别
生成 changelog
打 tag
发布制品
```

下面给一个可运行的 Python 玩具实现。它不依赖真实 Git，只演示“如何从提交列表计算版本号”。这段代码的目标不是替代现成工具，而是把机制讲透。

```python
import re

PRIORITY = {
    "none": 0,
    "patch": 1,
    "minor": 2,
    "major": 3,
}

def classify_commit(message: str) -> str:
    msg = message.strip()

    if "BREAKING CHANGE:" in msg:
        return "major"

    header = msg.splitlines()[0]
    if re.match(r"^[a-z]+(\([^)]+\))?!:", header):
        return "major"
    if re.match(r"^feat(\([^)]+\))?:", header):
        return "minor"
    if re.match(r"^fix(\([^)]+\))?:", header):
        return "patch"
    return "none"

def bump_version(version: str, release_type: str) -> str:
    major, minor, patch = map(int, version.split("."))
    if release_type == "major":
        return f"{major + 1}.0.0"
    if release_type == "minor":
        return f"{major}.{minor + 1}.0"
    if release_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return version

def next_version(current_version: str, commits: list[str]) -> str:
    highest = "none"
    for commit in commits:
        level = classify_commit(commit)
        if PRIORITY[level] > PRIORITY[highest]:
            highest = level
    return bump_version(current_version, highest)

commits = [
    "docs: update readme",
    "fix(auth): handle expired token",
    "feat(search): add fuzzy match",
]
assert classify_commit("fix(api): handle empty response") == "patch"
assert classify_commit("feat(ui): add dark mode") == "minor"
assert classify_commit("feat(api)!: change response schema") == "major"
assert next_version("1.4.2", commits) == "1.5.0"
assert next_version("1.4.2", ["docs: update docs"]) == "1.4.2"
assert next_version("1.4.2", ["fix: correct typo"]) == "1.4.3"
assert next_version("1.4.2", ["feat!: remove legacy config"]) == "2.0.0"

print("all tests passed")
```

这段代码体现了三个关键点：

1. 版本判断依赖提交语义，不依赖代码 diff 大小。
2. 最终结果取最高级别，不是把多个类型“累加”。
3. 破坏性变更优先级最高，必须显式标记。

真实工程中，通常不建议自己从零实现完整解析器，因为已有工具已经覆盖了大量边界情况，比如 merge commit、revert commit、不同分支范围计算、发布渠道区分等。更合理的做法是：用 `commitlint` 负责校验格式，用 `semantic-release` 负责消费规范并自动发版。

---

## 工程权衡与常见坑

第一类坑是**只有本地校验，没有 CI 复验**。这相当于把门禁只装在个人电脑上，任何人都可以通过 `--no-verify` 绕过。结果是仓库历史里仍会混入不合规提交，发布流程要么失败，要么误判版本。

第二类坑是**一个 commit 混多个意图**。比如一次提交同时做了格式化、重构、修 bug、加功能，然后写成：

- `feat: update project`

这种写法的问题不是“标题不精确”这么简单，而是它破坏了回滚和发版粒度。你以后想只回滚 bug 修复时，会把重构和新功能一起回掉；想只发补丁时，又被 `feat` 抬成次版本。

第三类坑是**漏写破坏性变更**。例如接口返回从：

```json
{"userName": "alice"}
```

改成：

```json
{"name": "alice"}
```

如果这条提交只写 `feat(api): simplify response`，发布工具可能只升 `minor`，但下游调用方升级后会直接报错。正确写法应该是：

- `feat(api)!: change response schema`

或者在 footer 中明确写：

```text
BREAKING CHANGE: rename userName to name in response payload
```

第四类坑是**把规范当成格式检查，不看语义正确性**。例如某人为了过校验，把所有提交都写成 `fix:`。这样形式合规，但语义错误。长期看，自动发版会被系统性低估，changelog 也会失真。

常见坑与规避方式可以汇总成表：

| 常见坑 | 风险 | 规避方式 |
|---|---|---|
| 只做本地校验 | 可被绕过 | 本地 hook + CI 双检 |
| 一个 commit 混多个目的 | 版本误判、回滚粗糙 | 单 commit 单意图 |
| 漏写 breaking 说明 | 风险被低估 | 用 `!` 或 `BREAKING CHANGE` 明示 |
| 为了过校验乱写类型 | changelog 和版本失真 | 评审时检查“类型和改动是否一致” |
| 合并策略过于随意 | 发布范围难追踪 | 统一 squash / merge 策略并定义口径 |

真实工程里还应该补上评测口径。只说“接了 commitlint”是不够的，至少要能观察这些指标：

| 指标 | 含义 |
|---|---|
| commit 合规率 | 提交格式通过率 |
| 无效提交拦截率 | 被 hook 或 CI 挡下的比例 |
| 版本误判率 | 发布级别和实际影响不一致的比例 |
| 发布回滚次数 | 反映语义标注是否可靠 |

如果这些指标长期没有改善，那么说明团队只是“上了工具”，没有真正把提交规范接入工程治理。

---

## 替代方案与适用边界

`Conventional Commits` 不是唯一方案。它最适合“希望把提交历史作为自动化输入”的团队，而不是所有仓库都必须强行采用。

常见替代方案有四类：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| Conventional Commits | 可解析、可自动发版 | 需要团队约束 | 中大型协作项目 |
| 自然语言 commit | 灵活、上手快 | 不可稳定自动化 | 小项目、临时仓库 |
| Gitmoji | 视觉直观 | 语义不稳定，版本难映射 | 轻量协作 |
| 手工维护 changelog 和版本号 | 流程可定制 | 成本高，易漏 | 发布频率低 |

对零基础到初级工程师来说，最容易产生误解的点是：是不是只要写了 `feat` 和 `fix`，流程就自动高级了。不是。提交规范只有和下面这些环节接起来，价值才完整：

- 本地提交拦截
- PR 校验
- 自动版本推导
- changelog 生成
- 发布阻断条件
- 回滚策略

如果一个项目根本没有自动发版，也没有 changelog 生成需求，那么统一写成“标题清楚、主体解释原因”的普通 Git 提交，也可以满足需要。反过来，如果项目已经是多人协作、频繁上线、要做版本回滚和发布审计，那么不采用结构化提交，后面迟早要补工程债。

一个实用判断标准是：**当提交信息需要被机器消费时，就该把它当协议设计；当提交信息只给人看时，简单规范可能已经够用。**

---

## 参考资料

1. [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
2. [commitlint - Commit conventions](https://commitlint.js.org/concepts/commit-conventions.html)
3. [commitlint - Local setup](https://commitlint.js.org/guides/local-setup)
4. [semantic-release Documentation](https://semantic-release.gitbook.io/semantic-release)
5. [Git - SubmittingPatches](https://git-scm.com/docs/SubmittingPatches)
