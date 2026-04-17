## 核心结论

GitHub Actions 可以把机器学习项目里的“代码变更后要做什么”写成一套可重复执行的流程。最重要的结构只有三层：`workflow`、`job`、`step`。`workflow` 是整条流水线，白话说就是“这次 CI/CD 要跑哪些阶段”；`job` 是一个资源单元，白话说就是“在哪台机器上完整跑一段任务”；`step` 是最小动作，白话说就是“执行一个具体命令或动作”。

对 ML 项目，最实用的做法不是把完整大训练塞进 CI，而是把“可快速暴露错误的小规模训练和评测”接到 `pull_request` 和 `push` 上。这样，代码一进 PR，就会自动检查三类问题：训练脚本能不能启动、关键指标是否明显退化、模型产物是否能被正确保存和传递。

一个最小可用思路是：

| 层级 | 角色 | 说明 |
| --- | --- | --- |
| workflow | 流程入口 | 定义触发器 `on:`、权限、并列执行的 jobs |
| job | 资源粒度 | 绑定 `runs-on`，同一个 job 内的 steps 在同一 runner 上执行 |
| step | 原子动作 | 拉代码、装依赖、训练、评测、上传 artifact |

如果项目需要 GPU，就把训练 job 放到 `self-hosted` runner。`self-hosted runner` 指你自己维护的执行节点，白话说就是“不是 GitHub 提供的公用机器，而是你自己的服务器”。再配合 `matrix strategy`，也就是“把多个配置自动展开成多份并行任务”，就能在 PR 阶段同时验证多个模型或参数组合。

---

## 问题定义与边界

这篇文章讨论的不是“如何做大规模分布式训练平台”，而是更具体的问题：

1. 代码提交到 GitHub 后，如何自动触发一轮轻量训练与评测。
2. 训练出来的权重、日志、指标文件，如何作为 Artifact 保存。
3. PR、主干分支、自托管 GPU 节点之间，如何建立一条既可用又相对安全的自动化链路。

这里的 `artifact` 是“流水线运行过程中产出的文件包”，白话说就是“这次运行生成的可下载结果”，例如 `metrics.json`、`model.pt`、`train.log`。

边界也要先说清楚：

| 维度 | 本文范围 | 不展开的内容 |
| --- | --- | --- |
| 训练规模 | 冒烟训练、样本训练、短时评测 | 多天大规模训练 |
| 执行环境 | GitHub Actions + 自托管或托管 runner | Kubeflow、Airflow 全平台编排 |
| 产物管理 | Actions Artifact | 完整模型仓库、权重版本平台 |
| 安全假设 | 基本隔离、PR 限制、权限收敛 | 企业级零信任体系 |

一个玩具例子是：你有个文本分类脚本，平时完整训练要 6 小时。CI 里不需要跑完整数据集，只需要抽 1000 条样本，训练 1 个 epoch，再检查程序是否报错、loss 是否下降、验证集准确率是否高于一个极低阈值，比如 0.55。这个流程的目标不是拿到最好模型，而是尽早发现“这次改动把训练弄坏了”。

一个真实工程例子是：团队维护两个模型分支 `bert` 和 `roberta`，每次 PR 都需要验证：
- CPU 环境下的数据预处理和单元测试能通过。
- GPU 环境下最小训练能跑通。
- 每个模型都能产出独立权重文件并上传。
- 指标文件可以被下游 job 汇总，决定是否允许合并。

触发器通常这样设置：

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main]
```

这段配置的含义很直接：PR 创建、更新、重新打开时跑；主干 `main` 被推送时也跑。这样可以把“开发中的检查”和“合并后的主干验证”统一到一份工作流里。

---

## 核心机制与推导

GitHub Actions 的执行逻辑可以理解成一条链：

$$
\text{event} \rightarrow \text{workflow} \rightarrow \text{job} \rightarrow \text{step}
$$

也就是“事件触发流程，流程启动若干任务，任务内部执行若干步骤”。

对 ML 来说，最关键的是两个机制：`runs-on` 和 `matrix`。

`runs-on` 决定 job 去哪里跑。比如：

```yaml
runs-on: [self-hosted, linux, x64, gpu]
```

这不是“任选一个标签”，而是“必须匹配所有标签”。所以只有同时带有 `self-hosted`、`linux`、`x64`、`gpu` 的 runner 才会接任务。这个设计很适合区分不同资源池，比如：
- `ubuntu-latest` 跑代码风格和单元测试。
- `[self-hosted, gpu]` 跑训练。
- `[self-hosted, gpu, a100]` 跑更贵的专项评测。

`matrix` 的作用是把一个 job 模板扩展成多份。假设：

```yaml
strategy:
  matrix:
    model: [bert, roberta]
    dataset: [mini]
```

那最终 job 数量为：

$$
job\_count = \prod_{k=1}^{n} |options_k|
$$

这里 $options_k$ 表示第 $k$ 个维度的候选值集合。上面这个例子里一共有两个维度，数量分别是 2 和 1，所以：

$$
job\_count = 2 \times 1 = 2
$$

也就是说会自动生成 2 个并行 job。每个 job 拿到不同的 `matrix.model` 值，然后执行自己的训练命令。

再看一个更贴近工程的推导。假设你要同时验证：
- 2 个模型：`bert`、`roberta`
- 2 个数据切片：`mini`、`debug`
- 2 个 Python 版本：`3.10`、`3.11`

那么总 job 数是：

$$
2 \times 2 \times 2 = 8
$$

如果每个 job 平均 6 分钟，理想并行下总墙钟时间接近 6 分钟；如果只有 2 台 GPU runner，实际排队后会更长。所以 matrix 不是越大越好，它本质上是在“覆盖率”和“资源成本”之间做交换。

玩具例子可以这样理解。你写了一个 `train.py`，支持两个模型名。CI 中写：

```yaml
- run: python train.py --model=${{ matrix.model }} --dataset=mini --epochs=1
```

当 `matrix.model` 分别取 `bert` 和 `roberta` 时，同一段模板就生成了两份独立训练任务。你不用复制两遍 job，只需要写一次模板。

真实工程里，常见做法是把流程拆成三个 job：
1. `lint_and_unit`：GitHub-hosted runner 上跑格式、静态检查、单元测试。
2. `smoke_train`：GPU runner 上跑 1 个 epoch 的小训练。
3. `eval_and_gate`：下载训练产物，做阈值判断，决定是否允许进入主干。

其中 `needs` 关键字可以让后一个 job 等前一个 job 成功后再执行。这就形成了“先便宜检查，后昂贵训练”的分层结构。

---

## 代码实现

先看一份适合入门的完整工作流。它包含 PR 触发、矩阵并行、GPU runner、artifact 上传和结果汇总。

```yaml
name: ml-ci

on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  smoke-train:
    strategy:
      fail-fast: false
      matrix:
        model: [bert, roberta]
    runs-on: [self-hosted, linux, x64, gpu]
    timeout-minutes: 30

    steps:
      - name: Checkout
        uses: actions/checkout@v5

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: pip install -r requirements.txt

      - name: Run smoke train
        run: |
          mkdir -p outputs weights
          python train.py \
            --model=${{ matrix.model }} \
            --dataset=mini \
            --epochs=1 \
            --save-path=weights/${{ matrix.model }}.pt \
            --metrics-path=outputs/${{ matrix.model }}.json

      - name: Upload weights
        uses: actions/upload-artifact@v4
        with:
          name: weights-${{ matrix.model }}
          path: weights/${{ matrix.model }}.pt
          if-no-files-found: error

      - name: Upload metrics
        uses: actions/upload-artifact@v4
        with:
          name: metrics-${{ matrix.model }}
          path: outputs/${{ matrix.model }}.json
          if-no-files-found: error
```

这里有几个关键点。

第一，`fail-fast: false` 表示矩阵里的某一个 job 失败时，不要立刻取消其他 job。对 ML 检查很有用，因为你通常想看到“到底是哪个模型、哪个配置坏了”，而不是只看到第一处失败。

第二，artifact 名必须唯一。因为 `upload-artifact@v4` 不允许多个上传动作复用同名 artifact。所以要把 `${{ matrix.model }}` 拼到名字里。

第三，训练脚本要显式产出机器可读文件，而不是只打印日志。最常见的是：
- `weights/*.pt` 保存权重。
- `outputs/*.json` 保存指标。
- 必要时再加 `train.log`。

下面给一个可运行的 Python 玩具实现，模拟“检查矩阵展开数量”和“指标是否达到阈值”。它不是 GitHub Actions 代码，但能帮助初学者理解 CI 中的判断逻辑。

```python
from itertools import product

def expand_matrix(matrix):
    keys = list(matrix.keys())
    values = [matrix[k] for k in keys]
    jobs = []
    for combo in product(*values):
        jobs.append(dict(zip(keys, combo)))
    return jobs

def should_pass(metrics, min_accuracy=0.60, max_loss=1.20):
    return metrics["accuracy"] >= min_accuracy and metrics["loss"] <= max_loss

matrix = {
    "model": ["bert", "roberta"],
    "dataset": ["mini"],
    "python": ["3.10", "3.11"],
}

jobs = expand_matrix(matrix)
assert len(jobs) == 4
assert {"model": "bert", "dataset": "mini", "python": "3.10"} in jobs

good_metrics = {"accuracy": 0.71, "loss": 0.83}
bad_metrics = {"accuracy": 0.52, "loss": 1.45}

assert should_pass(good_metrics) is True
assert should_pass(bad_metrics) is False

print("matrix and metric gate checks passed")
```

真实工程里，`eval_and_gate` job 往往会下载这些 `metrics-*.json`，然后统一判断。例如：
- 任意一个模型准确率跌破阈值，则 PR 失败。
- 主干分支上的 nightly 训练允许更长时长和更高门槛。
- PR 只做冒烟，主干才做更完整评测。

这时可以再写一个 job：

```yaml
  gate:
    needs: smoke-train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: collected
      - run: python ci/check_metrics.py --dir collected --min-accuracy 0.60
```

这个设计体现了一个原则：训练留在 GPU 节点，汇总和阈值判断尽量放回便宜、稳定的 GitHub-hosted runner。这样 GPU 只做“非做不可”的事。

---

## 工程权衡与常见坑

CI/CD for ML 最容易踩坑的地方，不是 YAML 语法，而是边界条件。

| 坑 | 影响 | 规避策略 |
| --- | --- | --- |
| 把完整训练放进 PR | 成本高、排队严重、反馈慢 | PR 只做冒烟训练，完整训练放主干或定时任务 |
| artifact 同名上传 | job 失败或互相覆盖逻辑混乱 | 名称拼接模型名、数据集名、提交号 |
| fork PR 直接消费产物 | 可能把未审查内容带入下游流程 | 对 PR 来源和事件类型做限制，避免危险链路 |
| 自托管 runner 网络未放行 | 上传/下载 artifact 失败 | 预先检查 GitHub 所需域名和出口策略 |
| 一个 runner 承载太多任务 | GPU 被排队占满，CI 失去意义 | 区分轻量 hosted job 和重型 GPU job |
| 阈值定得过高 | 开发期频繁误报失败 | PR 阶段用宽松阈值，主干阶段再收紧 |

先说最重要的工程权衡：PR 不是训练平台，而是变更守门口。它的目标是“快速发现明显错误”，不是“产出最终可用模型”。所以在 PR 上做短时、小数据、低成本的训练最合理。

再说 artifact。很多初学者会把它理解成“模型仓库”，这不准确。Artifact 更像“这次运行的临时产物”。它适合：
- 让后续 job 下载本次结果。
- 让开发者查看这次运行的输出。
- 暂存权重、日志、曲线文件。

但它通常不适合作为长期模型版本管理系统。长期权重更适合专门的对象存储、模型仓库或发布系统。

安全上，和 PR 相关的工作流要格外谨慎。原因很简单：PR 可能来自 fork，而 fork 的代码你还没有审查。如果你让后续高权限 job 直接信任这些运行产物，就可能把不可信内容带入更敏感的环境。对零基础读者，最稳妥的原则是：PR 流水线尽量只读仓库内容，不给额外写权限，不把其产物直接喂给高权限发布流程。

真实工程例子里，一个常见设计是：
- `pull_request`：只跑轻量训练和评测，不发布。
- `push` 到 `main`：在受控环境里跑更正式的训练或回归评测。
- `workflow_dispatch`：手动触发专项实验。
- `schedule`：每天凌晨跑一次较完整的 benchmark。

这样分层后，CI 才会稳定。否则所有事情都堆到 PR，会得到一个又慢又贵、还经常误报的系统。

---

## 替代方案与适用边界

GitHub Actions 不是唯一方案，但它很适合作为中小团队的起点，因为代码、流程、触发器、结果查看都在同一个平台里。

常见选择有三类：

| 方案 | 优点 | 适用边界 |
| --- | --- | --- |
| GitHub-hosted runner | 零运维，上手最快 | 适合 CPU 检查、轻量测试，不适合 GPU 训练 |
| Self-hosted GPU runner | 资源完全自控，可接私有网络和数据 | 适合已有服务器和运维能力的团队 |
| Managed GPU runner | 少管机器，保留 Actions 体验 | 适合需要 GPU 但不想自建节点的团队 |

如果你的项目还很早期，最简单的做法是：
- 先用 GitHub-hosted runner 跑格式检查、单元测试、数据脚本测试。
- 再增加一个自托管 GPU 节点，只负责最小训练。
- 最后再决定是否引入 matrix、nightly benchmark、模型发布。

什么时候 GitHub Actions 不再适合当主平台？通常有几个信号：
- 训练任务经常超过数小时，PR 无法承受等待时间。
- 需要复杂资源调度，如多机多卡、弹性伸缩、队列优先级。
- 需要长期实验管理、参数追踪、模型血缘管理。
- 需要更细粒度的数据权限与计算隔离。

这时你可能会把 GitHub Actions 保留为“入口层”，只负责：
- 代码检查
- 配置校验
- 提交训练请求
- 拉取最终结果摘要

而真正的大训练放到专门平台，比如内部调度系统、云训练服务或工作流编排平台。也就是说，GitHub Actions 很适合做“提交即验证”，不一定适合做“全生命周期训练中台”。

---

## 参考资料

- GitHub Docs: About workflows  
  https://docs.github.com/en/actions/using-workflows/about-workflows
- GitHub Docs: Workflow syntax for GitHub Actions  
  https://docs.github.com/actions/automating-your-workflow-with-github-actions/workflow-syntax-for-github-actions
- GitHub Docs: Choose the runner for a job  
  https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job
- GitHub Docs: Using a matrix for your jobs  
  https://docs.github.com/enterprise-cloud/latest/actions/examples/using-concurrency-expressions-and-a-test-matrix
- actions/upload-artifact README  
  https://github.com/actions/upload-artifact
- GitHub Well-Architected: Actions security recommendations  
  https://wellarchitected.github.com/library/application-security/recommendations/actions-security/
- Graphite: GitHub Actions on pull requests  
  https://graphite.com/guides/github-actions-on-pull-requests
- IREE: GitHub Actions practices  
  https://iree.dev/developers/general/github-actions/
- Machine.dev: GitHub Actions GPU runners  
  https://machine.dev/github-actions/
