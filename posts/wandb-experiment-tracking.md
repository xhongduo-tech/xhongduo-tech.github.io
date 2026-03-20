## 核心结论

W&B（Weights & Biases）可以看成一个“实验记录中台”，意思是把训练配置、指标曲线、数据版本、模型版本、分析报告放进同一套系统里管理。对初学者最直接的价值不是“更高级”，而是“少丢信息、能复现、能比较”。

它的核心使用模式可以压缩成三件事：

1. 用 `wandb.init(...)` 创建一次 Run。Run 就是“一次完整实验”的记录单元。
2. 用 `run.log(...)` 持续记录训练过程中的指标，比如 `loss`、`accuracy`、学习率。
3. 用 Artifact 管理输入和输出版本，比如训练集版本、模型检查点版本。

如果再加上 Sweep，W&B 就不只是“记日志”，而是“自动搜索超参数”。Sweep 可以按照设定的目标指标，例如最小化 `validation_loss`，自动尝试不同参数组合；当方法设为 `bayes` 时，它会根据历史实验结果，优先探索更可能变好的区域。

下表可以先把 W&B 的几个主要模块分清：

| 模块 | 作用 | 最适合解决的问题 |
|---|---|---|
| Run | 记录一次实验的配置、指标、系统信息 | 训练过程难复盘 |
| Artifacts | 管理数据集、模型、文件版本 | 数据和模型版本混乱 |
| Sweeps | 自动搜索超参数 | 人工试参效率低 |
| Reports | 汇总图表、结论和说明 | 团队分享和汇报成本高 |

一个最小新手用法通常是下面这种结构：在 `with wandb.init(...) as run` 中启动实验，在训练循环里 `run.log({"train_loss": ..., "val_acc": ...})`，训练结束后把模型文件记成 Artifact。这样一个 Run 里同时具备“参数、过程、结果”三类信息，后续对比才有意义。

---

## 问题定义与边界

机器学习实验的常见问题不是“不会训练”，而是“训练完以后说不清到底发生了什么”。这里有三个典型痛点：

| 痛点 | 具体表现 | 后果 |
|---|---|---|
| 指标分散 | `loss` 写在终端，超参记在笔记，模型文件在另一个目录 | 无法稳定复现 |
| 版本混乱 | 不知道模型对应哪一版数据 | 结果不可追溯 |
| 对比困难 | 多次实验只能手工截图或复制数值 | 难以做系统优化 |

玩具例子很简单。假设你训练一个二分类模型，试了三组学习率：`0.1`、`0.01`、`0.001`。如果只靠命令行输出，最后你可能只记得“第二组看起来最好”。但你未必记得：

- 它用的是哪个 batch size
- 它在第几轮开始过拟合
- 它对应的模型文件是不是后来被覆盖了

W&B 的目标不是替代训练框架，而是替代“零散记录”。训练流程可以抽象成下面这条链路：

训练代码 → 创建 Run → 记录指标与配置 → 保存数据/模型 Artifact → 汇总到 Project 与 Report

这里的边界也要说清楚。W&B 主要适用于这些场景：

- 有多次实验需要比较
- 需要共享结果给同事或导师
- 需要追踪模型和数据版本
- 可以接受依赖外部平台或其部署方式

它不一定适合这些场景：

- 完全离线、严格隔离的内网环境
- 只有极少量实验，且只需本地临时记录
- 团队已经有成熟的自建实验管理系统

换句话说，W&B 解决的是“实验管理问题”，不是“模型效果问题”。它不会直接让模型更准，但会显著降低“你不知道为什么不准”的成本。

---

## 核心机制与推导

### 1. Run：把一次实验变成一个可追踪对象

Run 可以理解成“实验容器”。你在 `wandb.init` 中写入项目名、实验名、配置参数后，W&B 会为这次训练建立唯一记录。后续所有日志、图表、文件都挂在这个 Run 上。

- `config`：实验配置，也就是超参数和关键开关
- `run.log(...)`：按步骤写入标量指标
- `run.watch(...)`：跟踪模型参数和梯度分布
- `run.log_artifact(...)`：记录输入输出文件版本

`run.log` 和 `run.watch` 容易混淆，职责其实不同：

| 接口 | 记录对象 | 常见内容 | 用途 |
|---|---|---|---|
| `run.log` | 你主动提交的指标 | `loss`、`accuracy`、`lr` | 画训练曲线、做对比 |
| `run.watch` | 模型内部统计 | 参数分布、梯度直方图 | 诊断训练是否异常 |
| `log_artifact` | 文件与版本 | 数据集、模型权重、预测结果 | 保证可追溯 |

这里要注意一个基本事实：`watch` 不是 `log` 的替代品。`watch` 更像“仪器检测”，`log` 才是“业务指标上报”。如果只开 `watch` 不记核心指标，仪表板信息仍然不完整。

### 2. Artifact：让输入和输出都有版本

Artifact 可以理解成“有版本号的文件集合”。最常见的两类 Artifact 是：

- 输入 Artifact：训练数据集、预处理后的特征文件
- 输出 Artifact：模型检查点、预测结果、评估报告

这样做的价值在于建立依赖链。理想情况下，一个模型 Run 应当能回答两件事：

1. 它基于哪一版数据训练出来
2. 它产出了哪一版模型文件

这就形成一个实验图谱：数据版本指向训练 Run，训练 Run 再指向模型版本。以后看到某个线上模型效果异常，可以反向查到它是用哪一批数据、哪组参数生成的。

### 3. Sweep：用历史结果指导下一次试参

超参数搜索的核心问题是：下一次该试哪里。最朴素的方法是网格搜索，也就是把候选点一个个枚举；随机搜索则是在范围内随机抽样。Bayesian Sweep 不同，它会利用历史观测结果推断“哪里更值得试”。

假设目标是最小化验证集损失 $f(x)$，其中 $x$ 表示一组超参数，例如学习率与权重衰减。贝叶斯优化通常会维护一个代理模型，也就是“用来近似真实目标函数的统计模型”，再通过采集函数决定下一个采样点。

常见的 Expected Improvement（EI，期望改进）可写成：

$$
EI(x) = \mathbb{E}[\max(f^\* - f(x), 0)]
$$

其中 $f^\*$ 是当前已知的最优损失。直观上，如果某个位置既“有机会比当前最好更优”，又“不确定性足够大”，它的 EI 往往更高，系统就更愿意去试它。

玩具例子如下。你已经有三次实验结果：

| learning_rate | validation_loss |
|---|---|
| 0.1 | 0.62 |
| 0.01 | 0.38 |
| 0.001 | 0.44 |

此时最优点在 `0.01` 附近。贝叶斯方法通常不会继续把大量预算浪费在 `0.1` 附近，而更可能在 `0.003`、`0.02`、`0.008` 一类区域继续试探，因为这些区域更有希望进一步降低损失。

真实工程例子中，情况会复杂得多。比如你训练一个文本分类模型，要同时搜索：

- `learning_rate`
- `batch_size`
- `warmup_ratio`
- `weight_decay`

手工尝试往往只能试几组“凭经验的组合”。Sweep 的意义在于把“经验”变成“系统搜索”，并把所有试验结果统一落到同一个 Project 中。

---

## 代码实现

下面先给一个不依赖 W&B 的玩具 Python 代码，用来说明“记录实验”到底记录什么。它可以直接运行，并带有 `assert`。

```python
from math import isfinite

runs = []

def fake_train(learning_rate, batch_size):
    # 构造一个玩具损失函数：lr 过大或过小都会变差
    loss = (learning_rate - 0.01) ** 2 * 100 + (64 - batch_size) ** 2 / 10000 + 0.35
    acc = 1.0 - min(loss / 2, 0.9)
    return {"val_loss": loss, "val_acc": acc}

for lr in [0.1, 0.01, 0.001]:
    for bs in [32, 64]:
        result = fake_train(lr, bs)
        runs.append({
            "config": {"learning_rate": lr, "batch_size": bs},
            "metrics": result
        })

best_run = min(runs, key=lambda x: x["metrics"]["val_loss"])

assert isfinite(best_run["metrics"]["val_loss"])
assert best_run["config"]["learning_rate"] == 0.01
assert best_run["config"]["batch_size"] == 64
print(best_run)
```

这段代码本质上做了三件事：

1. 保存配置
2. 保存结果
3. 找到最优实验

W&B 做的事情，就是把这三件事标准化、可视化，并支持团队共享。

下面是一个更接近真实训练脚本的最小 W&B 示例。术语说明：检查点（checkpoint）就是训练过程中保存的一份模型权重快照，便于恢复或回滚。

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

config = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 3
}

with wandb.init(project="demo-classifier", config=config) as run:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)

    # 记录参数与梯度分布，log_freq 不宜过高
    run.watch(model, log="all", log_freq=100)

    for epoch in range(run.config.epochs):
        train_loss = 1.0 / (epoch + 1)
        val_acc = 0.7 + 0.1 * epoch

        run.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

    torch.save(model.state_dict(), "model.pt")

    artifact = wandb.Artifact("demo-model", type="model")
    artifact.add_file("model.pt")
    run.log_artifact(artifact)
```

如果你是新手，先记住这个模式就够了：

- 配置放进 `config`
- 指标用 `run.log`
- 文件用 `Artifact`
- 需要内部统计时再加 `run.watch`

接着看 Sweep 配置。下面这个配置表示：目标是最小化 `validation_loss`，搜索方法用贝叶斯优化，参数空间里学习率是连续范围，batch size 是离散枚举。

```python
import wandb

sweep_configuration = {
    "method": "bayes",
    "metric": {
        "name": "validation_loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "min": 0.0001,
            "max": 0.1
        },
        "batch_size": {
            "values": [32, 64]
        }
    }
}

def train():
    with wandb.init() as run:
        config = run.config

        # 用玩具函数模拟训练结果
        validation_loss = (config.learning_rate - 0.01) ** 2 * 100 + (64 - config.batch_size) ** 2 / 10000 + 0.35
        validation_accuracy = 1.0 - min(validation_loss / 2, 0.9)

        run.log({
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        })

        assert validation_loss > 0
        assert config.batch_size in [32, 64]

sweep_id = wandb.sweep(sweep_configuration, project="demo-sweep")
wandb.agent(sweep_id, function=train, count=6)
```

参数字段可以先按下表理解：

| 字段 | 含义 | 典型示例 |
|---|---|---|
| `min` / `max` | 连续取值范围 | 学习率、权重衰减 |
| `values` | 离散候选列表 | batch size、优化器类型 |
| `distribution` | 采样分布 | `uniform`、`log_uniform` |
| `metric.name` | 优化目标名称 | `validation_loss` |
| `metric.goal` | 优化方向 | `minimize` 或 `maximize` |

真实工程例子中，一个常见实践是：

- 每次训练 Run 固定记录 `train_loss`、`val_loss`、`val_f1`
- 每个数据集版本都做成单独 Artifact
- 每个表现最好的模型都保存为 `best-model` 类型 Artifact
- 每周把关键 Run 拉进一个 Report，写明结论、失败尝试和后续动作

这样团队成员不需要翻日志目录，只需要打开 Project 和 Report 就能知道当前进展。

---

## 工程权衡与常见坑

W&B 并不是“开了就完事”，它的收益和成本一起存在。常见坑主要集中在日志频率、文件体积和权限设置。

| 坑位 | 现象 | 解决措施 |
|---|---|---|
| `watch` 频率过高 | 训练变慢，页面同步延迟 | 把 `log_freq` 控制在 100 或更低频 |
| `run.log` 太频繁 | I/O 压力大，上传拥堵 | 每 N 步聚合一次再上报 |
| Artifact 太大 | 上传时间长，占空间 | 只保存必要文件，避免重复原始数据 |
| Project 可见性错误 | 不该公开的实验被看见 | 提前检查 Visibility 与访问权限 |
| 指标命名混乱 | 图表难比较 | 统一命名，如 `train_loss`、`val_loss` |
| 数据版本未关联 | 无法追溯模型来源 | 每次训练显式记录输入 Artifact |

最典型的新手错误有两个。

第一个错误是把 `run.watch` 当成万能日志工具。实际上下面这句只负责参数和梯度统计：

```python
run.watch(model, log="all", log_freq=100)  # 用于监控内部状态，不替代核心指标日志
```

如果你不写 `run.log({"train_loss": ...})`，仪表板上仍然不会有完整的训练曲线。

第二个错误是“每一步都上传一切”。例如每个 step 同时上传十几个指标、直方图、图像、模型文件，这很容易把训练过程拖慢。更合理的做法通常是：

- 标量指标每 10 到 100 步记录一次
- 图像或样本预测每个 epoch 记录一次
- 模型文件只在关键节点保存，例如最佳验证集结果时

权限问题也常被忽略。多人共享 Project 时，Visibility 就是“项目可见范围”的设置。如果项目要发给团队外部人员看，应该明确区分只读分享、内部协作和公开展示，不要默认沿用历史设置。

---

## 替代方案与适用边界

W&B 不是唯一方案，只是“标准化程度较高”的方案。是否值得引入，取决于实验数量、协作需求和环境限制。

| 方案 | 协作能力 | 版本管理 | 可视化便利性 | 适用场景 |
|---|---|---|---|---|
| 本地日志 / CSV / Excel | 弱 | 很弱 | 弱 | 单人、少量实验、临时分析 |
| W&B | 强 | 强 | 强 | 多实验、多人协作、需要汇报 |
| 自建系统 | 可强可弱 | 可强可弱 | 取决于实现 | 强合规、强定制、私有化要求高 |

对初学者来说，可以这样判断：

- 如果你只是在本地练一个小模型，跑 3 到 5 次实验，本地 `csv` 足够。
- 如果你已经开始对比不同模型、不同数据版本，或者需要把结果发给别人看，W&B 的收益会迅速上升。
- 如果你的环境完全不能连接外部服务，或者组织要求所有数据和元数据都留在私有网络，那么应优先考虑自建或私有部署路线。

一个很实际的对比是：

- 本地记录：实验结束后，你要自己整理图、复制参数、截图发消息
- W&B：训练过程自动生成曲线，对比多个 Run 后可以直接做 Report 并分享链接

因此，W&B 的最佳适用边界不是“所有机器学习项目”，而是“需要持续实验管理、结果复盘和团队共享的项目”。

---

## 参考资料

- W&B Tracking 文档：介绍 `wandb.init`、`run.log`、`run.watch` 等基础记录方式  
  https://docs.wandb.ai/guides/track/log/
- W&B Artifacts 教程：说明数据集、模型等文件如何版本化并与 Run 关联  
  https://docs.wandb.ai/models/tutorials/artifacts
- W&B Sweeps 配置文档：说明 `method: "bayes"`、`metric`、`parameters` 的写法  
  https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
- W&B Reports 文档：说明如何汇总 Run、图表与文字说明并共享  
  https://docs.wandb.ai/guides/reports
- W&B 权限与 Artifact 访问说明：说明可见性、共享和访问边界  
  https://docs.wandb.ai/support/access_artifacts/
