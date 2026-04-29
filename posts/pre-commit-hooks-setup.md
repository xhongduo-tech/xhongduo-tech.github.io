## 核心结论

`pre-commit hook` 是 Git 在执行 `git commit`、真正生成提交对象之前运行的本地门禁。门禁的意思是：它先检查，再决定这次提交能不能继续。只要有任意一个 hook 以非 `0` 退出，这次提交就会被阻断。

用一个统一公式表示就是：

$$
提交成功 \iff \forall h \in H(S),\ e(h)=0
$$

这里：

- `S` 是这次提交的暂存文件集合，也就是你已经 `git add` 的那部分文件
- `H(S)` 是会作用到这批暂存文件上的 hook 子集，不是仓库里所有可能的检查
- `e(h)` 是 hook 的退出码。退出码可以理解成“检查结果编号”：`0` 表示成功，非 `0` 表示失败

新手可以把它理解成“提交前的本地门口检查”。代码要进仓库，先过门；门没过，就进不去。

它的价值不是替代 CI。CI 是持续集成流水线，通常在远端统一执行，负责更重、更全、更难绕过的检查。`pre-commit hook` 更适合把低成本、高频、机械性的错误前移到本机，比如尾随空格、文件末尾换行、格式化、明显语法错误、简单敏感信息泄漏。这样 reviewer 和 CI 不需要反复处理这些低级问题。

一个最小例子：

- 暂存了 `a.py`、`b.js`、`c.md`
- 配置了两个 hook：`trailing-whitespace` 和 `black`
- 如果 `b.js` 有尾随空格，那么 `trailing-whitespace` 返回 `1`
- 结果：这次 `git commit` 直接失败
- 修复后重新提交，如果两个 hook 都返回 `0`，提交才通过

本地 hook 和 CI 的分工可以先看这个表：

| 机制 | 运行位置 | 反馈速度 | 覆盖范围 | 是否易绕过 | 适合放什么规则 |
|---|---|---:|---|---|---|
| 本地 hook | 开发者机器 | 快 | 通常是暂存文件 | 是 | 格式、简单 lint、敏感词快筛 |
| CI | 远端流水线 | 中到慢 | 通常全仓库或全流程 | 相对难 | 单测、集成测试、全量扫描 |
| server-side hook | 远端 Git 服务器 | 中 | 推送或接收阶段 | 较难 | 强制组织级策略 |
| IDE 保存时检查 | 编辑器本地 | 非常快 | 当前编辑文件 | 很容易 | 格式化、即时提示 |

---

## 问题定义与边界

本文讨论的是 `git commit` 阶段的本地 hook，重点是提交前的那道门，而不是所有 Git hook，也不是远端仓库的 server-side hook，更不是 CI pipeline。

边界先划清楚：

- 它发生在本地
- 它默认服务于“这次准备提交的内容”
- 它不天然等于“整个仓库都被检查过”
- 它不天然等于“团队规则已经被强制执行”

最容易让新手误解的一点是：hook 默认面对的是暂存区，而不是工作区全量状态。暂存区可以理解成“这次提交的候选快照”。你写了很多代码，但只有 `git add` 进去的那部分，才属于这次提交的输入。

玩具例子：

- `a.py` 已暂存
- `b.js` 已暂存
- `c.md` 已暂存
- `c.md` 还有一些未暂存改动
- `b.js` 的已暂存内容里有尾随空格

如果 hook 是 `trailing-whitespace`，那么它会拦住这次提交，因为它检查到了 `b.js` 的已暂存问题。至于 `c.md` 那些还没进暂存区的坏内容，默认并不算这次提交的一部分，未必会被这次 hook 看到。

这也是为什么很多团队会误以为“本地 hook 已经守住了质量”，但实际上它只守住了“某一层质量”，而不是全部质量。

下面这个表把边界区别展开：

| 对象/机制 | 典型检查范围 | 是否默认参与本次 `commit` | 是否容易遗漏 | 能否被绕过 |
|---|---|---|---|---|
| staged files | 已暂存文件 | 是 | 低 | 可绕过 |
| unstaged files | 未暂存文件 | 否 | 高 | 天然不参与 |
| CI 全量检查 | 仓库、测试、构建流程 | 否，提交后才跑 | 低 | 相对难 |
| server-side hook | 远端接收推送时的数据 | 否，本地提交时不跑 | 低 | 较难 |

因此，`pre-commit hook` 的问题定义应当非常准确：它是“本地、提交前、默认围绕暂存内容”的快速质量门禁。只要把这个边界说清，后面的机制、能力和局限就都顺了。

---

## 核心机制与推导

机制可以拆成四步：

1. Git 先拿到这次提交的暂存文件集合 `S`
2. 根据 hook 配置和文件匹配规则，选出会参与执行的检查集合 `H(S)`
3. 逐个执行这些 hook，并收集退出码
4. 只要有一个退出码非 `0`，提交就中断

这个机制成立的根本原因不是“某个框架规定了这样做”，而是 Git hook 本身就是一个基于进程退出状态的门禁协议。协议的意思是“双方约定好如何沟通结果”。这里的约定很简单：脚本成功就返回 `0`，失败就返回非 `0`。

所以：

$$
提交成功 \iff \forall h \in H(S),\ e(h)=0
$$

反过来写也成立：

$$
\exists h \in H(S),\ e(h)\neq 0 \Rightarrow 提交失败
$$

这里最重要的细节是 `H(S)`。它不是“仓库启用的全部 hook”，而是“和这次暂存文件相关、实际会被执行的 hook 子集”。例如：

- `black` 通常只关心 Python 文件
- `trailing-whitespace` 可以关心很多文本文件
- `eslint` 可能只关心 JS/TS
- 某些 hook 只对特定目录生效

于是，暂存 3 个文件、配置 2 个 hook 的例子就可以精确推导：

- `S = {a.py, b.js, c.md}`
- hook 1：`trailing-whitespace`
- hook 2：`black`

如果 `a.py`、`b.js`、`c.md` 都进入 `trailing-whitespace` 的匹配范围，但只有 `b.js` 含尾随空格，那么：

- `e(trailing-whitespace) = 1`
- `e(black) = 0`，假设 `a.py` 格式没问题

因为存在一个 hook 失败，所以提交失败。修复 `b.js` 后再执行：

- `e(trailing-whitespace) = 0`
- `e(black) = 0`

这时所有参与执行的 hook 都成功，提交才通过。

把它画成流程就是：

`暂存区 S → 选择 H(S) → 逐个执行 hook → 汇总退出码 → 全为 0 则允许提交，否则阻断提交`

这里顺便指出一个常见误区：很多人把“hook 能执行某些检查”误解成“hook 就等于这些检查工具”。其实不是。hook 更像调度层，负责在正确时机调用工具。真正的格式化、lint、扫描，往往由 Black、ESLint、Ruff、detect-secrets 这些工具完成。

真实工程里，这个分层非常关键。一个前端 + Python 单仓库，通常会这样设计：

- 本地 hook 跑 `prettier --check`
- 本地 hook 跑 `eslint`
- 本地 hook 跑 `ruff`
- 本地 hook 跑 `detect-secrets`
- CI 再跑单元测试、集成测试、构建和全量扫描

原因很直接：本地门禁负责快筛，远端流水线负责终审。

---

## 代码实现

实现上有两条常见路线：

- 原生 Git hook 脚本
- `pre-commit` 框架

原生脚本的特点是轻量、直接、无额外抽象。`pre-commit` 框架的特点是配置统一、跨语言工具接入方便、团队复制成本更低。个人项目里，原生脚本足够；团队项目里，`pre-commit` 通常更稳。

先看一个最小 `pre-commit` 配置：

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer

  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
```

这段配置表达了三件事：

- 检查什么：尾随空格、文件末尾换行、Python 格式
- 用谁检查：`pre-commit-hooks` 和 `black`
- 用哪个版本：`rev` 固定版本，避免团队机器上规则漂移

如果你想用脚本显式触发这些检查，可以有一个最小封装：

```bash
#!/bin/sh
pre-commit run --files "$@"
status=$?
[ $status -eq 0 ] || exit $status
```

它的逻辑很直接：

- 对传入文件运行 `pre-commit`
- 读取退出码
- 如果不是 `0`，原样失败退出

为了让新手看懂“读取暂存文件、运行检查、根据退出码决定是否阻断提交”的核心逻辑，下面给一个可运行的 Python 玩具实现。它不是替代 Git，而是把机制翻译成一段最小代码。

```python
from dataclasses import dataclass

@dataclass
class HookResult:
    name: str
    exit_code: int

def trailing_whitespace_check(files):
    for path, content in files.items():
        for line in content.splitlines():
            if line.endswith(" "):
                return HookResult("trailing-whitespace", 1)
    return HookResult("trailing-whitespace", 0)

def black_check(files):
    py_files = {k: v for k, v in files.items() if k.endswith(".py")}
    # 玩具实现：这里只模拟“能通过”
    return HookResult("black", 0 if py_files is not None else 0)

def commit_allowed(staged_files):
    hooks = [
        trailing_whitespace_check,
        black_check,
    ]
    results = [hook(staged_files) for hook in hooks]
    return all(r.exit_code == 0 for r in results), results

# 失败案例：b.js 有尾随空格
staged = {
    "a.py": "print('ok')\n",
    "b.js": "const x = 1; \n",
    "c.md": "hello\n",
}
ok, results = commit_allowed(staged)
assert ok is False
assert any(r.name == "trailing-whitespace" and r.exit_code == 1 for r in results)

# 修复后通过
staged["b.js"] = "const x = 1;\n"
ok, results = commit_allowed(staged)
assert ok is True
assert all(r.exit_code == 0 for r in results)
```

这段代码对应的就是前面的公式：

- `staged_files` 对应 `S`
- `hooks` 经过筛选后形成 `H(S)` 的近似实现
- `exit_code` 对应 `e(h)`
- `all(...)` 对应 “所有 hook 都必须返回 `0`”

如果用原生 Git hook，一个最小的 `.git/hooks/pre-commit` 脚本往往会做三件事：

| 动作 | 作用 | 对应风险 |
|---|---|---|
| 读取暂存文件列表 | 限定本次提交范围 | 只覆盖 staged 内容 |
| 运行检查工具 | 执行格式化、lint、扫描 | 工具太慢会影响体验 |
| 返回退出码 | 通知 Git 是否放行 | 非 `0` 直接阻断提交 |

因此，代码实现的本质并不复杂。复杂的是工程化：如何统一版本、控制性能、避免不同机器结果不一致。

---

## 工程权衡与常见坑

`pre-commit hook` 最容易被高估的地方，是“它能管很多事”；最容易被低估的地方，是“它很容易失效”。工程上真正要关注的是权衡，不是功能清单。

先看常见问题表：

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 只查暂存文件 | 未暂存坏改动漏检 | 新规则上线时跑 `pre-commit run --all-files`，CI 再做全量检查 |
| 可被绕过 | 本地门禁形同虚设 | 关键规则在 CI 重跑，不把唯一保障放本地 |
| 太慢 | 开发者频繁跳过 | 本地只放快检查，重任务放 CI |
| 权限/可执行位错误 | hook 根本没运行 | 安装脚本里显式设置执行位并验证 |
| 规则漂移 | 不同人、不同分支结果不一致 | 固定版本，配置入库，纳入 onboarding |

先说覆盖问题。本地 hook 默认只看 staged files，这意味着它天然不是全量审计系统。比如一个人只 `git add` 了部分文件，本地检查只会围绕那部分执行。对于格式问题，这通常够用；对跨文件一致性、全仓库依赖关系、集成行为，则明显不够。

再说绕过问题。以下方式都能削弱或绕开本地门禁：

- `git commit --no-verify`
- `SKIP=hook_id git commit`
- `core.hooksPath=/dev/null` 或指向别处

这里的 `core.hooksPath` 是 Git 配置项，作用是指定 hook 脚本目录。换句话说，门在哪里，本身是可配置的。如果组织把所有质量保障都压在本地 hook 上，本质上是在把最终控制权交给每个开发者本机设置。

性能是第三个核心权衡。如果每次提交都要跑几十秒甚至几分钟，开发者的真实行为不会是“更自律”，而是“更频繁地绕过”。所以本地 hook 要尽量满足两个条件：

- 快
- 稳

快，意味着优先放格式、轻量 lint、简单扫描。稳，意味着不要依赖脆弱环境，不要一会儿能跑一会儿不能跑，不要把网络、容器、长时间测试都塞进提交前这一跳。

真实工程例子：

一个前端 + Python 单仓库，如果在本地 hook 里放这些规则，通常是合理的：

- `prettier --check`
- `eslint`
- `ruff`
- `detect-secrets`

如果你再把这些也塞进去：

- 全量单元测试
- 数据库集成测试
- 浏览器端到端测试
- 镜像构建

那大概率就过重了。因为提交前反馈要的是秒级到十几秒级，不是分钟级。

还有一个很常见的坑是“规则漂移”。比如：

- 有的人装了 hook，有的人没装
- 有的人用旧版本 `black`
- 有的人分支里配置不同
- 有的人本地 Python/Node 环境不一致

结果就是：同一份代码，在不同机器上门禁结果不同。这会直接破坏规则公信力。解决方式不是靠口头约定，而是把配置、版本、安装流程、CI 复验一起标准化。

---

## 替代方案与适用边界

`pre-commit hook` 适合做本地快速门禁，但不适合做最终质量裁决。这里的“最终质量裁决”指的是：决定代码是否可以被主干接受，是否满足团队统一标准，是否能通过构建、测试和集成验证。

因此更合理的做法是分层防线：

- 本地 hook：快筛
- CI：终审
- server-side hook：补充强制约束
- IDE 保存时检查：更早反馈

对比表如下：

| 机制 | 速度 | 覆盖面 | 是否可绕过 | 适合规则 |
|---|---:|---|---|---|
| 本地 hook | 快 | 暂存文件为主 | 是 | 格式、轻量 lint、简单敏感信息扫描 |
| CI | 中到慢 | 全仓库、全流程 | 相对难 | 单测、构建、集成测试、全量扫描 |
| server-side hook | 中 | 推送或接收内容 | 较难 | 强制拒绝不合规推送 |
| IDE 保存时检查 | 非常快 | 当前编辑文件 | 很容易 | 自动格式化、即时语法提示 |

适用边界可以概括为三句话。

第一，本地 hook 适合“便宜错误”。便宜错误是指检查成本低、反馈收益高、失败后修复简单的问题，比如尾随空格、文件末尾换行、代码格式、明显 lint 违规、简单密钥模式匹配。

第二，本地 hook 不适合承担“唯一可信裁决”。因为它在本地执行、可被绕过、默认只覆盖 staged 内容，所以关键规则必须在 CI 重跑。否则你得到的不是质量体系，而是“希望大家都自觉”。

第三，复杂系统要靠组合，而不是单点。真实工程里，前端 + Python 单仓库最常见的健康分工是：

- 本地 hook 跑 `prettier --check`、`eslint`、`ruff`、`detect-secrets`
- CI 跑单元测试、集成测试、构建、全量扫描
- 必要时用 server-side hook 阻止明显不合规推送
- IDE 保存时自动格式化，进一步减少低级错误进入提交阶段

如果团队规模小、仓库简单，甚至只用 IDE 格式化 + CI 也能工作。但一旦多人协作、review 成本上升，`pre-commit hook` 作为本地门禁的收益会迅速变大，因为它能把很多无意义往返提前消化掉。

---

## 参考资料

1. [Git 官方文档：githooks](https://git-scm.com/docs/githooks/2.46.0.html)
2. [Git 官方文档：git-config 中的 core.hooksPath](https://git-scm.com/docs/git-config/2.50.0.html)
3. [pre-commit 官方文档](https://pre-commit.com/)
4. [pre-commit 源码仓库：pre-commit/pre-commit](https://github.com/pre-commit/pre-commit)
5. [pre-commit-hooks 源码仓库](https://github.com/pre-commit/pre-commit-hooks)
