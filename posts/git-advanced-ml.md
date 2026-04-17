## 核心结论

`git rebase --interactive` 是“交互式变基”：白话说，它让你把一串旧提交重新按顺序播放，并在播放前决定哪些保留、哪些合并、哪些改写。它最适合整理个人功能分支的历史，不适合改写已经被多人共享的公共分支。

`git bisect` 是“二分定位”：白话说，它不是一条条试提交，而是每次挑中间那个提交测试，把排查范围直接砍半，所以定位回归 bug 的速度通常远快于人工翻日志。若待查范围有 $N$ 个提交，需要的测试轮数近似为：

$$
s=\lceil \log_2 N \rceil
$$

例如 $N=100$ 时，$\log_2 100 \approx 6.64$，所以最多约 7 轮就能定位到引入问题的提交。

在大模型仓库里，Git 只擅长管理代码和小文本，不擅长直接管理大权重和大数据。常见分工是：

| 工具 | 解决的问题 | 适合存什么 | 核心特点 |
|---|---|---|---|
| Git | 代码版本控制 | 源码、配置、脚本、文档 | 快、分支强、协作成熟 |
| Git LFS | 大文件指针化 | 模型权重、检查点、二进制包 | Git 中只存指针 |
| DVC | 数据与模型版本 | 数据集、特征产物、训练输出 | Git 管元数据，远端存内容 |
| monorepo | 统一仓库组织 | 训练代码、推理服务、共享库 | 统一依赖与目录边界 |

一个新手最常见的安全用法是：在自己的 `feature` 分支执行 `git rebase -i main`，把“调试打印”“修 typo”之类碎提交 `drop` 或 `squash` 掉，整理完后仅对自己的远程私有分支执行 `git push --force-with-lease`。`--force-with-lease` 的意思是“带保护的强推”：白话说，只有远端还是你预期的状态时才覆盖，能避免把别人的新提交顶掉。

`rebase`、`merge`、`bisect` 经常被一起提到，但它们不是同一类操作：

| 命令 | 目标 | 是否改写历史 | 典型场景 |
|---|---|---|---|
| `git rebase -i` | 整理提交历史 | 是 | 合并零散提交、删调试提交 |
| `git merge` | 合并两条分支 | 否 | 公共分支协作、保留真实分叉历史 |
| `git bisect` | 定位引入 bug 的提交 | 否 | 回归排查 |

---

## 问题定义与边界

这篇文章讨论的是一个具体问题：当项目从单纯代码仓库，变成“代码 + 模型权重 + 数据集 + 训练流程”的大模型工程仓库后，如何同时解决三件事：

1. 历史不能太脏，否则主干上的提交难以阅读和回滚。
2. 回归必须能快速定位，否则一次训练链路变更会卡住整个团队。
3. 大文件和数据不能直接塞进 Git，否则 clone、fetch、CI 都会变慢甚至失效。

边界也要先说清楚。

第一，交互式 rebase 主要用于“本地未共享历史”或“你负责的私有分支”。如果你在公共分支上改写历史，其他协作者的提交基线会失效，他们下次 `git pull` 往往会遇到分叉、冲突，甚至需要强制重置。

第二，`git bisect` 依赖“可判定测试”。可判定测试的意思是：给定某个提交，你能明确说“好”还是“坏”。如果测试结果今天过、明天挂，或者依赖在线服务随机波动，二分的结论就不可靠。

第三，Git LFS 和 DVC 都不是“装了就自动工作”。它们要求团队统一安装、统一远端配置、统一目录约定。否则你会看到仓库里只有指针文件，真实内容却拉不下来。

一个最小的 bisect 新手流程如下。假设你知道 `HEAD` 已经坏了，而标签 `v1.0` 还是好的：

```bash
git bisect start
git bisect bad HEAD
git bisect good v1.0
# Git 自动切到中间某个提交
# 你运行测试后，继续标记：
git bisect good
# 或
git bisect bad
# 反复几轮后，Git 会输出 first bad commit
git bisect reset
```

也可以写成一条命令启动：

```bash
git bisect start HEAD v1.0
```

这里 `HEAD` 是“当前最新提交”，`v1.0` 是“已知正确基线”。白话说，bisect 需要你先给它一个坏的终点和一个好的起点，它才知道从哪一段历史里查。

玩具例子：你有 8 个提交，`c1` 到 `c8`，其中 `c8` 已坏、`c1` 已好。人工顺序检查最坏要看 8 次，bisect 只要看中间点 `c4`、再看 `c6`、再看 `c5`，3 次就能确定问题是否从 `c5` 开始出现。这就是“每轮减半”的直观含义。

真实工程例子：一个大模型仓库中，`/apps/training` 放训练入口，`/libs/features` 放特征处理，`/configs` 放实验配置，`/services/infer` 放推理服务。某次上线后发现验证集 F1 从 0.82 掉到 0.76。此时不能只看训练脚本，因为回归可能来自特征库、配置改动、依赖升级或错误权重。你需要同时有：干净的 Git 历史、可执行的回归测试、和可追踪的数据/模型版本。

---

## 核心机制与推导

交互式 rebase 的底层思路不是“在原地编辑旧提交”，而是“从共同基线之后，把每个提交变成补丁，再按新顺序重放”。这就是为什么 rebase 后提交哈希会变化：因为新的提交对象已经不是原来的那一批了。

一个典型的 `git rebase -i main` 编辑界面大致如下：

```text
pick a1b2c3 feat: add trainer cli
pick d4e5f6 debug: print batch shape
pick 1a2b3c fix: correct tokenizer path
pick 4d5e6f chore: rename variable
```

你可以把它改成：

```text
pick a1b2c3 feat: add trainer cli
drop d4e5f6 debug: print batch shape
squash 1a2b3c fix: correct tokenizer path
squash 4d5e6f chore: rename variable
```

这些指令的含义可以直接记成表：

| 指令 | 作用 | 适合场景 |
|---|---|---|
| `pick` | 保留提交 | 这个提交本身完整且有意义 |
| `reword` | 保留但改提交信息 | 内容对，说明不清 |
| `squash` | 合并到前一个提交 | 一次功能被拆成多个碎提交 |
| `drop` | 丢弃提交 | 调试代码、临时试验、无效修改 |

`git bisect` 的机制更像算法题。假设在“好提交”和“坏提交”之间有 $N$ 个候选提交，每轮选中间一个点测试。若中间点是好，说明问题在后半段；若中间点是坏，说明问题在前半段。每轮把搜索空间缩小到原来的一半，因此轮数近似是：

$$
s = \lceil \log_2 N \rceil
$$

当 $N=100$ 时，区间收缩过程可以近似理解为：

| 轮次 | 剩余候选提交数上界 |
|---|---|
| 0 | 100 |
| 1 | 50 |
| 2 | 25 |
| 3 | 13 |
| 4 | 7 |
| 5 | 4 |
| 6 | 2 |
| 7 | 1 |

所以 100 个提交不用看 100 次，通常 7 次左右就够。

下面用一个可运行的 Python 玩具程序模拟这个推导。假设第 73 个提交开始出错，算法每次测试中间点：

```python
import math

def first_bad_commit(n, first_bad):
    left, right = 1, n
    steps = 0

    def is_bad(commit_id):
        return commit_id >= first_bad

    while left < right:
        steps += 1
        mid = (left + right) // 2
        if is_bad(mid):
            right = mid
        else:
            left = mid + 1
    return left, steps

bad, steps = first_bad_commit(100, 73)
assert bad == 73
assert steps <= math.ceil(math.log2(100))
print(bad, steps)
```

这个例子不是 Git 命令本身，但它把 bisect 的数学本质直接算出来了。

在真实 Git 使用里，你还会看到这样的 bisect 记录：

```bash
git bisect start
git bisect bad HEAD
git bisect good v1.0
git bisect good e12345
git bisect bad f23456
git bisect good a34567
git bisect visualize
git bisect log
```

`git bisect visualize` 是“把当前候选范围可视化”：白话说，它会调用日志或图形工具，让你看当前还剩哪些可能有问题的提交。`git bisect log` 会把你每次标记 good/bad 的过程记下来，便于复现和分享。

---

## 代码实现

先看交互式 rebase 的最小流程：

```bash
git checkout feature/train-refactor
git rebase -i main
# 编辑 pick/squash/drop 后保存退出
# 若发生冲突：
git add .
git rebase --continue
# 若想放弃：
git rebase --abort
# 仅在自己的远程分支上使用
git push --force-with-lease
```

如果你只是想把功能分支整理成一到两个逻辑清晰的提交，这套流程已经足够。核心原则只有一条：整理历史发生在合并前，不发生在共享主干上。

然后看大文件管理。Git LFS 的最小配置通常是：

```bash
git lfs install
git lfs track "weights/*.ckpt"
git add .gitattributes
git add weights/model.ckpt
git commit -m "track checkpoint with lfs"
```

对应的 `.gitattributes` 会出现类似内容：

```gitattributes
weights/*.ckpt filter=lfs diff=lfs merge=lfs -text
```

这表示 `weights/*.ckpt` 不再直接作为普通 Git blob 存进仓库，而是由 LFS 存储真实文件，Git 里只保留指针信息。

如果数据集和训练产物也要严格版本化，DVC 更合适。先看最小命令：

```bash
dvc init
dvc add data/dataset.csv
git add data/.gitignore data/dataset.csv.dvc
git commit -m "track dataset with dvc"

dvc remote add -d storage s3://my-ml-bucket/dvc-store
dvc push
```

`data/dataset.csv.dvc` 是“DVC 元数据文件”：白话说，它不存真实数据，只记录数据文件的哈希和路径。真实内容存到 `.dvc/cache`，再由 `dvc push` 同步到远端对象存储。

一个最小的 `.dvc` 文件可能像这样：

```yaml
outs:
  - md5: 4a7d1ed414474e4033ac29ccb8653d9b
    path: data/dataset.csv
```

如果你还要把训练流程也定义成可复现管线，可以写 `dvc.yaml`：

```yaml
stages:
  prepare:
    cmd: python scripts/prepare.py
    deps:
      - raw/train.csv
      - scripts/prepare.py
    outs:
      - data/processed

  train:
    cmd: python scripts/train.py --config configs/base.yaml
    deps:
      - data/processed
      - scripts/train.py
      - configs/base.yaml
    outs:
      - artifacts/model.ckpt
      - artifacts/metrics.json
```

这类配置的价值在于：代码、依赖、输入、输出都被显式记录。团队成员执行 `git pull` 后，再执行 `dvc pull`，就能拉到与当前代码版本匹配的数据和模型产物。

在 monorepo 里，一个适合训练工程的目录约定可以是：

```text
/apps
  /training
  /inference
/libs
  /features
  /metrics
/configs
/data
/models
```

真实工程例子：你维护一个推荐模型仓库。训练入口在 `/apps/training`，共享特征处理在 `/libs/features`，模型配置在 `/configs/ranker`。权重文件 `weights/ranker-v3.ckpt` 用 Git LFS 跟踪，训练样本快照 `data/ranker_2026_03.parquet` 用 DVC 管理。某次线上指标回退后，先用 `git bisect run ./bisect-test.sh` 自动找出坏提交，再用 `git rebase -i main` 整理修复分支，把“尝试 A”“尝试 B”“修日志”压成一条真正有意义的修复提交。

---

## 工程权衡与常见坑

这些工具都有效，但都有明确成本。

最危险的坑是“在公共历史上做 rebase”。后果不是单纯冲突，而是协作者本地历史和远端历史不再同一条链。常见表现是：别人 `pull` 后看到重复提交、奇怪冲突，或者不得不 `fetch` 后重新基于新历史整理自己的分支。

第二个常见坑是“bisect 依赖不确定测试”。如果你的测试要访问不稳定服务、依赖随机种子、或者不同机器环境不一致，那么同一个提交可能一会儿判好、一会儿判坏。此时二分结果没有工程意义。解决方法通常是固定随机种子、锁依赖版本、减少外部依赖，必要时写自动脚本统一判定。

第三个常见坑是“LFS/DVC 只在一台机器配置好”。你本地能跑，不代表同事和 CI 能跑。LFS 需要统一安装并正确认证，DVC 需要统一 remote。少一个步骤，别人拉到的就只是指针或元数据。

下面是高频问题汇总：

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| 公共分支上 `rebase -i` | 他人拉取后历史分叉 | 只在私有分支整理历史 |
| 直接 `push --force` | 覆盖别人的远端提交 | 用 `--force-with-lease` |
| bisect 测试不稳定 | 同一提交结果反复变化 | 固定环境、种子、依赖 |
| LFS 未安装 | 拉到 pointer 文件而非真实权重 | 团队统一执行 `git lfs install` |
| DVC 未设 remote | `dvc push` 或 `dvc pull` 失败 | 项目初始化时写明 remote |
| monorepo 无目录边界 | 改动范围失控，CI 全量触发 | 按应用、共享库、配置分层 |

还要补一个认知误区：monorepo 不是“把所有内容丢进一个仓库”。它真正要求的是边界清晰、触发规则清晰、依赖关系清晰。否则仓库虽然只有一个，但团队心智负担会更重。

---

## 替代方案与适用边界

如果你的团队明确禁止改写历史，那么不要硬用交互式 rebase。更稳妥的替代是 `git merge --squash`。它的意思是“把一整个分支的改动压成一次合并结果”，但不会去改写原有公共提交链。缺点是分支内部的细粒度历史会消失。

如果 bisect 需要人工跑很久，或者每次测试步骤都一样，优先使用自动化：

```bash
git bisect start HEAD v1.0
git bisect run ./bisect-test.sh
git bisect reset
```

这里的 `bisect-test.sh` 只需要做一件事：退出码为 `0` 表示“好”，非 `0` 表示“坏”。这样 Git 就能自动完成整个二分。

在大文件与数据管理上，也不是只有 DVC 一个选项：

| 方案 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| DVC | 需要版本化数据、模型、pipeline | 复现链路完整 | 需要额外学习与 remote 配置 |
| `git-annex` | 大量文件、分布式存储 | 文件管理灵活 | 团队采用率较低 |
| 手动云存储 | 数据版本简单、流程轻 | 上手快 | 版本关联弱，复现容易漂移 |

如果仓库规模继续扩大，monorepo 也不是唯一答案。多仓库加子模块也能工作，但要接受更高的跨仓同步成本。简单说：

- monorepo 适合共享库多、改动联动强、需要统一 CI 的团队。
- multi-repo 适合边界稳定、发布节奏不同、团队相对独立的系统。

所以适用边界可以归纳成一句话：如果你的核心矛盾是“历史脏、回归难查、数据和模型不可复现”，那么 `rebase + bisect + LFS/DVC + 清晰 monorepo` 是一套互相补位的工具链；如果你的核心矛盾其实是“流程太复杂、团队纪律不足”，那先补规范，再补工具。

---

## 参考资料

| 文档 | 关键内容 | 访问路径 |
|---|---|---|
| Atlassian Git Rebase | 解释 rebase 与 merge 的差异，强调历史改写边界 | https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase/ |
| GitHub Docs: About Git Rebase | 说明 rebase 的基本行为与使用场景 | https://docs.github.com/ |
| Git 官方文档: git bisect | 说明 `start/good/bad/run/reset` 等标准流程 | https://git-scm.com/docs/git-bisect |
| Tower Guide: Using git bisect | 用示例解释二分定位回归的实际步骤 | https://www.git-tower.com/learn/git/faq/git-bisect |
| GitLab LFS 文档 | 说明 LFS 的指针机制与团队配置要求 | https://docs.gitlab.com/topics/git/lfs/ |
| DVC 官方指南 | 说明 `dvc add`、remote、pipeline、cache 的工作方式 | https://dvc.org/doc |
| Monorepo 结构实践文章 | 讨论统一仓库下的目录边界与组织方式 | https://mindfulchase.com/deep-dives/monorepo-fundamentals-deep-dives-into-unified-codebases/structuring-your-monorepo-best-practices-for-directory-and-code-organization.html |
