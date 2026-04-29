## 核心结论

Git 分支策略的核心区别，不在于你把分支叫 `develop`、`main` 还是 `release`，而在于你把“集成风险”放在哪里处理。集成风险的白话解释是：多人同时改代码后，最终把改动拼回一条主线时会不会冲突、会不会漏修、会不会难回滚。

先看一个“修一条线上 bug”的开场对比。

如果团队同时维护旧版本 `1.8` 和新版本 `1.9`，线上 bug 往往要先在已发布版本上修，再把修复同步到后续版本，这种流程天然更像 Git Flow。因为它默认你会长期保留发布线、补丁线和开发线，专门处理“多版本并行”的复杂度。

如果团队做的是纯 SaaS，代码合并后当天甚至几小时内就能上线，那么更像 GitHub Flow 或 Trunk-Based。因为这类团队的重点不是“维护很多旧版本”，而是“让主干尽快集成、尽快验证、尽快发布”。

可以先记一句判断结论：

$$
\text{选择分支策略} \approx \text{选择把发布风险、集成风险、回滚成本放在哪个阶段承担}
$$

结论总表如下：

| 策略 | 分支长度 | 发布方式 | 适用团队 |
|---|---|---|---|
| Git Flow | 较长，常有 `feature` / `release` / `hotfix` | 版本化发布，可并行维护多个版本 | 客户端、SDK、嵌入式、需要长期维护稳定版本的团队 |
| GitHub Flow | 较短，通常围绕 PR 存活 | 合并到 `main` 后快速发布 | 中小型 Web/SaaS 团队，发布频繁 |
| Trunk-Based | 极短，接近当天合并 | 主干优先，持续集成，随时可发布 | CI 成熟、自动化高、强调主干可发布的大团队或高效小团队 |

对零基础读者，一个够用的判断方式是：如果你要“同时开旧车和新车”，更容易走向 Git Flow；如果你一直开同一辆车，但每天都在保养和升级，更容易走向 GitHub Flow 或 Trunk-Based。

---

## 问题定义与边界

本文讨论的对象，不是“Git 能不能建分支”，而是“团队如何组织开发、集成、发布和回滚”。分支只是表面工具，真正决定策略的是发布模型。

要判断策略，先看四个边界条件：

| 问题边界 | 是 | 否 |
|---|---|---|
| 是否多版本并行 | 更偏 Git Flow | 更偏 GitHub Flow / Trunk-Based |
| 是否强制主干可发布 | 更偏 Trunk-Based / GitHub Flow | 可接受 Git Flow 的中间状态 |
| 是否需要发布冻结 | 更偏 Git Flow | 更偏持续交付模型 |
| 是否有热修复回灌 | 必须设计版本同步流程 | 流程可简化 |

这里的“主干可发布”白话解释是：`main` 上任何一个通过门禁的提交，理论上都能拿去上线，不需要再临时拼很多补丁。

同样是 5 人团队，分支策略也可能完全不同。

玩具例子：5 人做一个练习型博客网站，只部署一套线上环境，没有历史版本维护需求。这个团队最怕的是分支挂太久，最后一合并全冲突。此时 GitHub Flow 往往够用：每个功能一个短分支，提 PR，审完就合并。

真实工程例子：5 人做桌面客户端或支付 SDK。企业客户可能还在用 `1.8`，新客户开始试 `1.9`，主线已经在做 `2.0`。线上 bug 一旦出现，修复不能只进 `main`，还要回补到仍在服务的发布线。这个场景里，“如何同步修复”比“如何快速合并功能”更重要，所以更接近 Git Flow。

因此，分支策略不是团队人数函数，而是发布约束函数。人数相同，系统生命周期和交付方式不同，结论可以完全相反。

---

## 核心机制与推导

理解三种策略，最有用的不是背定义，而是记住一个经验公式：

$$
D = v \times L
$$

其中：

- $D$ 表示分叉量或合并压力。白话讲，就是“你这条分支和主线已经差多远了”。
- $v$ 表示主干变化速率。白话讲，就是主线每天变动有多快。
- $L$ 表示分支存活时间。白话讲，就是这条分支离开主线后独立活了多久。

这个公式不是 Git 官方公式，而是工程上的经验抽象。它抓住了一个关键事实：主干变化越快，分支活得越久，最后合并时需要理解和协调的上下文就越多。

看一个最小数值例子。

假设主干每天有 20 个有效提交：

- 5 天长分支：$D \approx 20 \times 5 = 100$
- 1 天短分支：$D \approx 20 \times 1 = 20$

这不表示一定有 100 处文本冲突，而是表示你合并时要面对的大致上下文规模更大。分支越久没同步，最后一次合并越像“把两本不同版本的书拼回一本”。

三种策略，本质上就是对 $L$ 的不同管理方式。

| 策略 | 对 `L` 的态度 | 集成时机 | 冲突风险 |
|---|---|---|---|
| Git Flow | 允许较长 `L` | 在 `develop`、`release`、`hotfix` 阶段分层集成 | 单次合并和回灌风险更高 |
| GitHub Flow | 主动压短 `L` | 通过短 PR 尽快并回 `main` | 风险较低，但依赖评审与 CI |
| Trunk-Based | 尽量让 `L \to 0` | 频繁直接回主干或极短分支 | 单次冲突最小，但对工程纪律要求最高 |

Git Flow 为什么成立？因为它承认有些团队必须维护多个稳定版本，于是接受更长的 `L`，再通过 `release` 和 `hotfix` 把风险拆成“可管理的多段流程”。代价是回灌复杂，漏同步的概率上升。

GitHub Flow 为什么成立？因为它不想把大量风险堆到最后，所以要求分支尽量短，把评审、测试、部署前置到每次 PR。它不是不要分支，而是只接受“足够短的分支”。

Trunk-Based 为什么成立？因为它进一步认为，很多冲突不是“代码难”，而是“分开太久”。所以它尽量让每个人快速回主干，再用 feature flag 和 branch by abstraction 控制未完成功能。feature flag 的白话解释是：代码已经合并，但功能默认关闭，不对用户生效。branch by abstraction 的白话解释是：先加一层稳定接口，让新旧实现并存，再逐步切换。

所以三者不是“谁更先进”，而是谁更匹配你的约束：你到底是要管理多版本，还是要管理高频集成。

---

## 代码实现

这里不写业务逻辑，而是把分支策略翻成最小可执行流程。先给一个辅助脚本，模拟 $D = v \times L$ 的直觉。

```python
def divergence_pressure(commits_per_day: int, branch_days: float) -> float:
    assert commits_per_day >= 0
    assert branch_days >= 0
    return commits_per_day * branch_days

# 玩具例子：主干每天 20 个提交
long_branch = divergence_pressure(20, 5)
short_branch = divergence_pressure(20, 1)

assert long_branch == 100
assert short_branch == 20
assert long_branch > short_branch

def choose_strategy(has_multi_release: bool, must_keep_trunk_releasable: bool, ci_mature: bool) -> str:
    if has_multi_release:
        return "Git Flow"
    if must_keep_trunk_releasable and ci_mature:
        return "Trunk-Based"
    return "GitHub Flow"

assert choose_strategy(True, False, False) == "Git Flow"
assert choose_strategy(False, True, True) == "Trunk-Based"
assert choose_strategy(False, False, True) == "GitHub Flow"

print("ok")
```

上面不是严格数学模型，而是把核心判断显式化：多版本并行优先考虑 Git Flow；主干必须可发布且 CI 成熟，才适合更激进的 Trunk-Based。

下面看三种策略的最小 Git 命令流。

```bash
# GitHub Flow
git checkout main
git pull origin main
git checkout -b feature/login
git commit -am "add login"
git push origin feature/login
# open PR -> review -> CI pass -> merge to main
# deploy from main
```

GitHub Flow 的重点是：功能分支短，PR 小，合并快。它默认 `main` 是事实上的集成中心。

```bash
# Git Flow
git checkout develop
git pull origin develop
git checkout -b feature/login
git commit -am "add login"
git push origin feature/login
# merge feature/login -> develop

git checkout -b release/1.9 develop
# test, freeze, fix release issues
git checkout main
git merge release/1.9
git tag v1.9.0

git checkout develop
git merge main
# make sure release fixes are back-merged into develop

git checkout -b hotfix/1.9.1 main
git commit -am "fix prod bug"
git checkout main
git merge hotfix/1.9.1
git tag v1.9.1
git checkout develop
git merge main
```

Git Flow 的关键不在分支名字，而在“发布冻结”和“热修复回灌”。回灌的白话解释是：你在发布线或线上修过的问题，必须同步回开发线，否则以后还会重新出现。

```bash
# Trunk-Based
git checkout main
git pull origin main
git checkout -b short/login-flag
git commit -am "add login behind feature flag"
git push origin short/login-flag
# open PR quickly -> CI pass -> merge to main the same day
# feature remains disabled until fully ready
```

Trunk-Based 的重点是“尽快回主干”。如果一个功能做两周，那不是保留两周分支，而是拆成很多可单独合并的小步，并用 flag 隔离未完成部分。

真实工程例子可以这样理解：

一个 SaaS 订单系统要接新支付渠道。若采用 GitHub Flow，团队会把“新增接口”“校验参数”“后台配置页”“灰度开关”拆成多个短 PR。若采用 Trunk-Based，则会进一步要求这些改动尽量当天合并到 `main`，即便功能还不能对全部用户开放，也先用 flag 关住。

---

## 工程权衡与常见坑

真正的失败，通常不是“选错术语”，而是流程约束没有落地。你说自己用 Trunk-Based，但分支一挂就是 10 天；你说自己用 Git Flow，但 hotfix 从不回灌；这时问题已经不是模型，而是执行失效。

| 类别 | 常见坑 | 后果 | 规避 |
|---|---|---|---|
| Git Flow | `release` 分支滞留太久 | 发布前积累大量差异，回灌容易漏 | 限制发布冻结时间，强制回灌清单 |
| GitHub Flow | PR 越做越大，几天不合并 | 评审困难，冲突上升，变成伪长期分支 | 设定 PR 大小和分支寿命上限 |
| Trunk-Based | 没有 CI、flag、自动化测试就追求快合并 | 主干频繁坏，所有人被阻塞 | 先建门禁，再谈主干优先 |
| 通用坑 | 把策略当命名规范，不当发布约束 | 口头上有流程，实际不可执行 | 明确主干状态、回滚路径、回灌规则 |

几个必须直接执行的规则：

1. 规定分支寿命。
   建议 GitHub Flow 的普通功能分支控制在 1 到 2 天内，Trunk-Based 更短。超过这个时长，默认要拆分。

2. 规定回灌规则。
   只要修复发生在 `release`、`hotfix` 或生产补丁线上，就必须回到后续开发线。否则同一个 bug 会“修了又来”。

3. 规定 CI 门槛。
   CI 是持续集成流水线，白话讲就是“自动替你跑测试、检查构建、验证最基本质量”。没有 CI，主干优先只是口号。

4. 规定 feature flag 生命周期。
   flag 不是永久垃圾桶。功能上线稳定后，要清理旧 flag，否则代码会留下大量双路径逻辑。

一个典型失败模式是：团队自称 Git Flow，但 `develop` 长期不稳定，`release` 挂三周，hotfix 只修 `main` 不回 `develop`。结果是下一次发版时，旧 bug 再次出现。这不是 Git Flow 的问题，而是“接受了长分支，却没承担长分支所需的回灌纪律”。

另一个典型失败模式是：团队说自己用 GitHub Flow，但每个 PR 都包含十几个需求点，评审靠口头，合并前不跑测试。这样做表面轻量，实际只是把 Git Flow 的复杂度藏进了 PR 里。

---

## 替代方案与适用边界

分支策略不是互斥教条，很多团队实际采用的是混合形态。

最常见的混合做法是：“GitHub Flow 外壳 + Trunk-Based 约束”。也就是仍然使用短 PR 分支和代码评审，但要求分支极短、主干始终可发布、未完成功能必须走 flag。这种方式在现代 SaaS 团队里很常见，因为它兼顾了协作可见性和主干优先。

另一种常见做法是：“Git Flow 的发布管理 + 更短功能分支”。也就是保留 `release` / `hotfix` 处理多版本维护，但不再鼓励长期功能分支，尽量把 feature 分支缩短。这适合仍需多版本支持、但不想承担过高开发分叉成本的团队。

团队类型对照如下：

| 团队类型 | 推荐策略 | 原因 |
|---|---|---|
| 纯 SaaS、每天发版、CI 成熟 | GitHub Flow 或 Trunk-Based | 主目标是快速集成和持续发布 |
| 大型 Web 平台、多人并行、自动化很强 | Trunk-Based | 可用强门禁换取最低分叉成本 |
| 桌面客户端、移动端、嵌入式 | Git Flow 或混合方案 | 常见版本冻结、灰度慢、需要维护旧版本 |
| SDK / 库团队 | Git Flow 或带发布分支的混合方案 | 下游用户升级慢，需要维护多个稳定版本 |
| 小团队、流程基础一般 | GitHub Flow | 实现成本低，比 Git Flow 更容易执行 |

可以把“是否长期保留发布分支”视为一道分水岭。

如果你必须长期保留发布分支，因为客户环境升级慢、版本支持周期长，那么 Git Flow 或其变体更自然。

如果你不需要长期保留发布分支，而且可以把每次改动都快速验证、快速上线，那么 GitHub Flow 已经足够；如果 CI、测试、灰度、flag 都很成熟，再进一步收紧到 Trunk-Based，收益才会明显。

所以最后的选型逻辑不是“哪个最流行”，而是：

- 多版本并行维护强不强
- 主干是否必须始终可发布
- 自动化门禁是否足够成熟
- 团队是否有能力把大需求拆成小步集成

---

## 参考资料

1. [GitHub flow - GitHub Docs](https://docs.github.com/en/enterprise-server%403.19/get-started/using-github/github-flow)
2. [About pull requests - GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
3. [A successful Git branching model - Vincent Driessen](https://nvie.com/posts/a-successful-git-branching-model/)
4. [Trunk Based Development: Introduction](https://trunkbaseddevelopment.com/)
5. [Trunk Based Development: Branch by Abstraction](https://trunkbaseddevelopment.com/branch-by-abstraction/)
