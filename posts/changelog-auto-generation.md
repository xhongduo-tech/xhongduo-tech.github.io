## 核心结论

`CHANGELOG` 自动生成，不是把 `git log` 原样打印出来，而是把一个发布窗口内的变更记录，转换成面向使用者和发布者的版本说明。

白话说，提交记录像开发者的工作流水账，发布说明像给用户看的版本公告。自动化系统做的是“翻译”和“压缩”，不是“转储”。

它的核心定义可以写成：

$$
\text{CHANGELOG Generation} = \text{Git History} + \text{PR Metadata} \rightarrow \text{Versioned Release Notes}
$$

这类系统能不能稳定工作，关键不在“用了什么工具”，而在输入是否规范。提交信息是否可解析、PR 是否带标签、版本 tag 是否连续、破坏性变更是否显式标记，这些决定了输出是否可信。

下面这张表先把最容易混淆的两个对象区分开：

| 对比项 | `git log` | `release notes / changelog` |
|---|---|---|
| 面向对象 | 开发者 | 用户、测试、运维、发布负责人 |
| 信息形态 | 原始流水 | 归类后的摘要 |
| 噪声处理 | 几乎没有 | 会过滤或降权 |
| 是否强调影响面 | 不一定 | 必须强调 |
| 是否按类别组织 | 通常没有 | 通常有，如 Added / Fixed / Breaking |

反面例子很常见。如果直接输出 `git log`，你会看到 `merge branch`、`wip`、`fix typo`、`refactor`、`chore` 这类记录。开发者能大致看懂，但用户无法快速判断“这次升级到底改了什么，会不会有兼容性风险”。

所以，真正可用的自动生成系统，不是一个“日志导出器”，而是一个放进发布流程里的规则引擎。

---

## 问题定义与边界

先把问题说窄，否则很容易把它谈成“AI 理解代码语义”这种更大的命题。

`CHANGELOG` 自动生成解决的是“发布说明生成”问题，不是“自动理解整个仓库”的问题。它通常只关心一个明确的发布区间，而不是从项目创建第一天开始全盘扫描。

发布范围通常写成：

$$
R = [t_{prev}, t_{curr})
$$

这里的 $t_{prev}$ 是上一个发布 tag，$t_{curr}$ 是当前发布 tag 或待发布提交点。对应的输入集合可以写成：

$$
C = \text{git log}(R) \cup \text{merged PRs}(R)
$$

白话说，这次发布只看“上一个版本到当前版本之间发生了什么”。

这个边界非常重要。很多发布事故不是生成器不会写，而是“区间切错了”。例如一个 hotfix 从老分支回补，如果你不是按 tag 切片，而是按某个分支的 `HEAD` 去猜范围，就可能把不属于本次版本的修复混进来，导致 changelog 记错版本。

可以把边界列清楚：

| 类别 | 包含什么 | 不包含什么 |
|---|---|---|
| 版本范围 | `last_tag..HEAD` 或 `last_release_tag..new_tag` | 整个仓库全部历史 |
| 数据输入 | commit、merged PR、标签、compare link | 未合并分支、草稿提交 |
| 用户可见性 | 新功能、修复、破坏性变更、性能变化 | 临时调试、格式化、内部噪声 |
| 发布对象 | 一个明确版本 | 模糊的“最近改动” |

这里还有一个常见误解：自动生成并不能凭空补足语义。如果团队提交信息很乱，PR 没有标签，breaking change 也没有写明，那自动化不会变聪明，它只会更快地产生错误结果。

一个玩具例子能说明这个边界。

假设某个版本区间里有 4 条提交：

- `feat: add CSV export`
- `fix: handle empty input`
- `chore: bump deps`
- `tmp: test release`

如果目标是用户可读的发布说明，通常只应该稳定产出前两条，第三条可能按策略决定是否显示，第四条应当被过滤。也就是说，输入集合不等于输出集合，系统中间必须有规则层。

---

## 核心机制与推导

这类系统的机制可以概括成三步：归类、压缩、生成。

第一步是归类。归类就是判断每条变更属于什么类别。术语“归类函数”可以写成：

$$
f: C \rightarrow \{\text{feat}, \text{fix}, \text{perf}, \text{docs}, \text{refactor}, \text{breaking}, \text{noise}\}
$$

白话说，系统先给每条输入打标签，判断它是新增、修复、性能优化、文档更新、重构、破坏性变更，还是噪声。

第二步是压缩。压缩不是丢信息，而是把多条原始记录收束成可读结构。例如 8 条 `fix` 可能会被整理成 3 条更高层的修复摘要，或者至少按模块聚合。

第三步是生成。最终输出函数可以写成：

$$
L = w(f(C), labels, compare\_link, contributors)
$$

其中 $w$ 表示模板渲染过程。也就是把归类后的结果、PR 标签、版本比较链接、贡献者信息拼成一份发布说明。

一个最小数值例子：

某次发布里有 6 个 commit：

- 2 个 `feat`
- 2 个 `fix`
- 1 个 `docs`
- 1 个 `feat!`

这里 `feat!` 的 `!` 表示破坏性变更，也就是升级后旧用法可能失效。

那么合理输出至少应该包含：

- `Added = 2`
- `Fixed = 2`
- `Breaking = 1`

如果系统解析出 5 条有效提交，那么覆盖率是：

$$
coverage = \frac{parsed(C)}{|C|} = \frac{5}{6} \approx 83.3\%
$$

但覆盖率高不代表能上线。假设漏掉的刚好是那条 `feat!`，那么：

$$
breaking\_recall = \frac{detected\_breaking}{actual\_breaking} = \frac{0}{1} = 0
$$

这时即使 `coverage` 还不错，发布也应阻断。因为破坏性变更漏检比少写一条文档更新严重得多。

实际工程里，至少要看下面几个指标：

| 指标 | 公式 | 说明 |
|---|---|---|
| `coverage` | `parsed(C) / |C|` | 有多少输入被成功解析 |
| `breaking_recall` | `detected_breaking / actual_breaking` | 破坏性变更识别是否漏掉 |
| `precision` | `correct_entries / generated_entries` | 生成出来的条目是否准确 |
| `user_visible_coverage` | `surfaced_user_visible_changes / actual_user_visible_changes` | 用户真正关心的变更是否被覆盖 |

这里最容易被忽略的是“用户可见变更覆盖率”。因为有些仓库提交写得很规范，但很多提交只是内部重构。你如果把这些都写进 changelog，形式上很完整，实际上对用户没有帮助。

真实工程里，质量通常来自规则层，而不是模型层。最常见的规则包括：

- 使用 Conventional Commits，也就是约定式提交规范
- 通过 `!` 或 `BREAKING CHANGE:` 标记破坏性变更
- PR label 作为提交语义的兜底信息
- 只按 tag 切分发布范围
- 对 `docs`、`chore`、`ci` 这类条目做过滤或降权

换句话说，生成质量主要由“输入规约 + 分类规则 + 模板约束”决定，而不是靠最后那一步“把文本写得漂亮”。

---

## 代码实现

实现上最典型的路径，是在 GitHub Actions 里监听 tag 推送，触发一次发布流程。术语“CI”指持续集成系统，也就是自动执行检查和发布脚本的平台。

一个完整流程通常是：

| 步骤 | 输入 | 输出 |
|---|---|---|
| 1. tag push | 新版本 tag | 触发发布任务 |
| 2. 收集数据 | `last_tag..HEAD`、PR 元数据 | 原始变更集合 |
| 3. 分类与判级 | commits、labels、breaking footer | 版本级别和分类结果 |
| 4. 生成 release notes | 分类结果、模板 | 发布说明草稿 |
| 5. 写入 `CHANGELOG.md` | 新版本说明 | 仓库内 changelog 更新 |
| 6. 发布 draft | changelog、版本号 | GitHub Release 草稿 |
| 7. 人工审核 | 草稿内容 | 正式发布或回滚 |

如果用 `semantic-release` 这类工具链，常见职责分工如下：

| 插件 | 作用 |
|---|---|
| `@semantic-release/commit-analyzer` | 根据提交信息判断版本升级级别 |
| `@semantic-release/release-notes-generator` | 生成 release notes |
| `@semantic-release/changelog` | 写入 `CHANGELOG.md` |
| `@semantic-release/github` | 发布到 GitHub Releases |

下面给一个最小可运行的 Python 玩具实现。它不依赖 Git 命令，只演示“归类 + 计数 + 阻断条件”这三个核心动作。

```python
from collections import Counter

COMMITS = [
    "feat: add CSV export",
    "fix: handle empty input",
    "docs: update README",
    "feat!: remove legacy auth API",
    "chore: bump deps",
    "fix: retry on timeout",
]

def classify_commit(msg: str) -> str:
    if "BREAKING CHANGE:" in msg or msg.startswith("feat!:") or msg.startswith("fix!:"):
        return "breaking"
    if msg.startswith("feat:"):
        return "feat"
    if msg.startswith("fix:"):
        return "fix"
    if msg.startswith("perf:"):
        return "perf"
    if msg.startswith("docs:"):
        return "docs"
    if msg.startswith("refactor:"):
        return "refactor"
    return "noise"

def summarize(commits):
    classes = [classify_commit(c) for c in commits]
    counter = Counter(classes)
    parsed = sum(v for k, v in counter.items() if k != "noise")
    total = len(commits)

    coverage = parsed / total if total else 1.0
    breaking_count = counter["breaking"]

    notes = {
        "Added": counter["feat"],
        "Fixed": counter["fix"],
        "Breaking": breaking_count,
        "Docs": counter["docs"],
        "coverage": round(coverage, 4),
    }
    return notes

result = summarize(COMMITS)

assert result["Added"] == 1
assert result["Fixed"] == 2
assert result["Breaking"] == 1
assert abs(result["coverage"] - (5 / 6)) < 1e-4

def release_gate(notes):
    # 示例门槛：覆盖率至少 80%，且 breaking 变更必须单独显式出现
    assert notes["coverage"] >= 0.8
    if notes["Breaking"] > 0:
        assert "Breaking" in notes and notes["Breaking"] >= 1

release_gate(result)
print(result)
```

这个例子故意保留了一个 `noise` 提交，所以覆盖率是 $5/6$。它说明两个事实：

1. 自动生成的第一步不是写文案，而是先做机器可判断的结构化分类。
2. 发布门槛必须写成可以执行的规则，否则“人工感觉没问题”很难稳定复制。

再看一个更接近真实工程的例子。

假设团队使用 GitHub Flow，所有改动都通过 PR 合并。发布时在 `v2.3.0` 上打 tag。CI 做的事情不是直接读整个 `main` 分支，而是：

- 找到上一个 tag，比如 `v2.2.4`
- 收集 `v2.2.4..v2.3.0` 的提交和对应 PR
- 用提交规范和 PR label 做双重归类
- 生成 `release notes`
- 写入 `CHANGELOG.md`
- 先发一个 draft release
- 由发布负责人检查 compare range、breaking section、安全修复条目
- 审核通过后再公开

这里“draft release”就是草稿版本。它的价值很高，因为 changelog 自动化最怕的是“看起来能跑，但内容悄悄错了”。

---

## 工程权衡与常见坑

最大的权衡，是自动化效率和发布准确性之间的平衡。

自动化越强，对输入规范的要求越高。没有稳定规范时，自动化不是节省人力，而是把错误批量化。

下面这张表列常见坑和规避方式：

| 现象 | 成因 | 规避方式 |
|---|---|---|
| changelog 里全是噪声 | 直接转储 `git log` | 过滤 `chore`、`ci`、临时提交 |
| 新功能漏进修复分类 | 提交信息不规范 | 强制 Conventional Commits |
| breaking change 没显示 | 只在正文提及，未写 `!` 或 `BREAKING CHANGE:` | 用 `commitlint` 校验 |
| hotfix 被记到错误版本 | 发布区间不是按 tag 切 | 只用 tag 做切片 |
| squash merge 后语义丢失 | PR 标题和 commit 语义不一致 | 统一 squash 标题规范，PR label 兜底 |
| 自动生成和手工修改冲突 | 仓库里存在两个事实源 | 定义唯一权威来源，自动生成后只做审核 |

这里有几条规则几乎应该视为硬约束：

| 规则 | 原因 |
|---|---|
| `!` 或 `BREAKING CHANGE:` 必须可检测 | 破坏性变更不能靠人工猜 |
| 发布只按 tag 切片 | 否则区间不可复现 |
| PR label 可作为兜底 | 提交信息质量不稳定时还能补语义 |
| release notes 只生成一次权威文本 | 避免人工版和自动版双写 |

还要补一个经常被忽视的点：上线门槛和可回滚条件。

如果 changelog 生成已经是正式发布的一部分，那么它就不只是“文档任务”，而是发布流水线的一环。它应该有阻断条件，例如：

- `breaking_recall` 必须是 `100%`
- 安全修复条目必须人工确认
- compare range 必须能复现
- 版本升级级别必须和 breaking 语义一致
- 草稿审核未通过时，不得正式发布

可回滚条件也要明确。比如：

- 生成内容为空，但本次版本实际有用户可见变更
- compare link 指向错误区间
- 漏掉 breaking change
- 把 backport 错记到主线版本
- 发布后发现 `CHANGELOG.md` 与 GitHub Release 文案不一致

这些都不应该靠“出了问题再看”。它们应该在流程设计时就被定义成告警或阻断。

---

## 替代方案与适用边界

自动生成不是唯一方案，也不是所有项目都值得上完整自动化。

如果项目很小、发布不频繁、贡献者只有一两个人，手工维护 changelog 往往更简单。相反，如果项目多人协作、每周都发版、还需要 GitHub Releases、版本比对和回滚审计，那么自动生成的价值会迅速上升。

可以把常见方案并列比较：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 纯人工 changelog | 灵活、语义准确 | 成本高、容易漏 | 小型个人项目 |
| Conventional Commits + 自动生成 | 规则清晰、可持续 | 依赖提交规范 | 中大型工程团队 |
| PR label 驱动生成 | 适合 PR 流程稳定的团队 | 依赖评审时打标签 | GitHub 团队协作 |
| 语义模型辅助生成 | 文案可读性高 | 可解释性和稳定性弱 | 人工审核充分的辅助场景 |

还可以做一个适用性判断表：

| 判断项 | 是 | 否 |
|---|---|---|
| 是否有稳定 tag | 适合自动生成 | 先建立发布边界 |
| 是否有规范提交 | 直接上规则驱动 | 先治理输入 |
| 是否有 PR 流程 | 可利用 label 和 reviewer 信息 | 更多依赖 commit 本身 |
| 是否需要 GitHub Releases | 自动化收益高 | 收益可能有限 |
| 是否有人审核发布说明 | 适合放入正式发布链路 | 不建议全自动直接公开 |

实践上可以给出一个很实用的决策顺序：

1. 先确认有没有稳定 tag。
2. 再确认提交和 PR 语义是否足够规范。
3. 然后决定是以 commit 为主，还是以 PR label 为主。
4. 最后再决定要不要引入更复杂的模板或模型辅助生成。

也就是说，语义弱、提交乱的仓库，正确顺序不是“先上 changelog 生成器”，而是“先把发布流程和输入规约建起来，再谈自动化”。

---

## 参考资料

1. [GitHub Docs: Automatically generated release notes](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes)
2. [GitHub Docs: About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
3. [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
4. [semantic-release Plugin List](https://semantic-release.gitbook.io/semantic-release/extending/plugins-list)
5. [@semantic-release/release-notes-generator](https://github.com/semantic-release/release-notes-generator)
6. [Git Pretty Formats](https://git-scm.com/docs/pretty-formats.html)
