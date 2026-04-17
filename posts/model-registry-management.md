## 核心结论

模型注册与管理，本质上是在“训练完成之后、部署之前”给模型建立一套统一的身份系统。身份系统至少包含两部分：

$$
\text{模型注册} = \text{版本号} + \text{元数据}
$$

版本号就是这次产物是第几版；元数据就是和模型一起保存的说明信息，比如训练数据、代码提交、指标、责任人、依赖环境。白话说，它不是只把 `model.pkl` 丢进文件夹，而是把“这个文件从哪来、表现如何、谁批准上线”一起存进去。

模型管理再往前走一步：它把“哪个版本能部署、哪个版本要回滚、哪个版本只能测试”变成可执行流程。对生产系统来说，真正重要的不是“我们有模型”，而是“我们能准确定位当前线上跑的是哪个版本，并且能回到上一个安全版本”。

一个新手版理解方式是：每次训练结束，先把产物存档并自动编号 `V1、V2、V3...`，再写清楚数据集、评估指标和审批状态，最后只有被批准的版本才能绑定到线上别名，比如 `latest`、`champion` 或 `production`。

| 版本 | 状态 | 审批人 | 上线时间 |
|---|---|---|---|
| V3 | Rejected | 李工 | 未上线 |
| V4 | Pending | 王工 | 未上线 |
| V5 | Approved | 周工 | 2026-03-20 14:30 |
| V6 | Canary | 自动规则 + 周工 | 2026-03-25 10:00 |

这张表的意义不是展示信息，而是回答四个工程问题：现在跑谁、谁批准的、什么时候上的、出故障回退到谁。

---

## 问题定义与边界

问题定义很直接：团队缺少统一模型仓库时，模型文件会散落在对象存储、训练机目录、聊天记录和脚本参数里。最后部署阶段经常出现三类混乱：

1. 不知道线上版本对应哪次训练。
2. 不知道哪个版本已经评估通过。
3. 不知道模型为什么变了，也不知道该回滚到哪一个。

这就是模型管理缺失。它不是算法问题，而是交付问题。

边界也要讲清楚。本文讨论的是“已经训练完成、准备进入共享与部署流程”的模型产物管理，不覆盖下面两类对象：

1. 实验阶段的大量临时文件，比如中间 checkpoint、调试日志、失败试验的随手导出 artefact。
2. 数据版本系统本身，比如原始样本、特征表、标签集的完整治理。

也就是说，本文只处理“可被团队复用并可能进入生产”的模型版本。

玩具例子可以用一个很小的团队来理解。假设三个人每天各训练一次垃圾邮件分类模型，目录结构是：

- `model_final.pkl`
- `model_final_v2.pkl`
- `new_model.pkl`
- `use_this_one.pkl`

到周五时，没有人敢确认线上该部署哪个文件。这不是因为模型太复杂，而是因为缺少注册表。只要改成“先登记到 registry，再审批上线”，问题就立即收敛：同名模型进入同一个 Model Group，版本自动递增，评估结果和标签跟版本绑定。

真实工程例子更常见。一个推荐系统团队每天夜间重训 CTR 模型，训练流水线会产出：

- 模型文件
- 特征统计
- 训练参数
- AUC、LogLoss、校准误差
- 使用的数据快照 ID
- Git commit hash

如果这些内容不进入统一注册表，那么线上服务只知道“现在有个模型在跑”，而平台团队无法回答“它对应哪批数据、是否经过审批、是否比上一个版本更好”。这会直接阻断审计、回滚和自动化部署。

---

## 核心机制与推导

模型注册系统通常围绕一个核心对象工作：Model Group 或 Registered Model。白话说，它相当于“某一类模型的总目录”，例如 `ctr-ranker`、`fraud-detector`、`price-forecast`。同一个目录下的每次注册，形成一个新版本。

版本号的最基本规律是：

$$
V_n = V_{n-1} + 1
$$

它的含义不是“性能更好”，而是“在这个模型组里，它是一次新的、唯一的注册事件”。版本号解决的是唯一标识，不是质量判断。质量判断来自评估和审批。

假设同一个 Model Group 连续注册 5 次，那么版本就是 1 到 5。第 6 次注册时，即使这版只是为了做灰度实验，只要它进入同一个组，仍然应当拿到版本 6。然后你可以给它打 `canary=true` 之类的标签，表示它的用途，而不是篡改版本语义。

模型仓库真正有工程价值，是因为它把“版本”和“流程状态”连起来。一个简化的状态机可以写成：

- `Pending`: 已注册，但还没批准部署
- `Approved`: 允许进入部署或绑定生产别名
- `Rejected`: 明确不允许进入生产

SageMaker 官方文档给出的默认审批流里，常见状态是 `PendingManualApproval -> Approved/Rejected`。这其实就是一个有限状态机。白话说，有限状态机就是“系统只能在有限几个合法状态之间跳转”的规则表。

| 当前状态 | 触发事件 | 下一状态 | 工程动作 |
|---|---|---|---|
| PendingManualApproval | 人工审批通过 | Approved | 触发 CI/CD 或允许部署 |
| PendingManualApproval | 人工审批拒绝 | Rejected | 停止发布 |
| Rejected | 重新审核通过 | Approved | 恢复部署资格 |
| Approved | 风险回退或复核拒绝 | Rejected | 切回最近一个已批准版本 |

这里要区分两个概念：

1. `version` 是不可变编号。
2. `alias` 或 `stage/tag` 是可变引用。

例如 `V6` 永远是 `V6`，但 `latest`、`champion`、`production` 可以从 `V5` 改指向 `V6`。这样线上调用不需要硬编码版本号，只需要读取一个稳定别名。MLflow 当前推荐更多使用 alias 和 tag 做部署编排，而不是过度依赖旧式 stage。

再看一个最小推导。假设你只允许一个“生产指针”：

$$
\text{production\_alias} \rightarrow V_k,\quad V_k \in \{\text{Approved versions}\}
$$

那么生产版本切换的安全前提就是：

$$
V_k \text{ 被批准 } \land \text{评估结果满足门槛}
$$

这里的门槛可能是：

$$
\text{AUC}_{new} \ge \text{AUC}_{old} - 0.002
$$

以及

$$
\text{P99 latency}_{new} \le 1.1 \times \text{P99 latency}_{old}
$$

这说明模型管理不是只存文件，而是把“能否接管流量”变成规则。

---

## 代码实现

先用一个可运行的玩具例子说明“版本递增 + 审批门禁”如何工作。这个例子不依赖任何平台 SDK，只模拟注册表最小行为。

```python
class ToyRegistry:
    def __init__(self):
        self.versions = []
        self.aliases = {}

    def register(self, metrics, tags=None):
        version = len(self.versions) + 1
        record = {
            "version": version,
            "status": "Pending",
            "metrics": metrics,
            "tags": tags or {},
        }
        self.versions.append(record)
        return version

    def approve(self, version):
        for item in self.versions:
            if item["version"] == version:
                item["status"] = "Approved"
                self.aliases["latest"] = version
                return
        raise ValueError("version not found")

    def reject(self, version):
        for item in self.versions:
            if item["version"] == version:
                item["status"] = "Rejected"
                return
        raise ValueError("version not found")

    def get_by_alias(self, alias):
        version = self.aliases[alias]
        return next(v for v in self.versions if v["version"] == version)


registry = ToyRegistry()

v1 = registry.register(metrics={"auc": 0.81}, tags={"dataset": "ds_20260301"})
v2 = registry.register(metrics={"auc": 0.84}, tags={"dataset": "ds_20260302", "canary": "true"})

assert v1 == 1
assert v2 == 2

registry.approve(v2)
latest = registry.get_by_alias("latest")

assert latest["version"] == 2
assert latest["status"] == "Approved"
assert latest["metrics"]["auc"] == 0.84
```

这个例子说明三件事：

1. 注册动作只负责生成唯一版本。
2. 审批动作决定版本是否具备上线资格。
3. 别名 `latest` 指向的是“当前被选中的版本”，而不是“数字最大的版本”。

如果你使用 MLflow，注册动作通常是把某次 run 产物登记为模型版本。官方常见写法如下：

```python
import mlflow
from mlflow import MlflowClient

model_name = "MyModel"
run_id = "<run_id>"
model_uri = f"runs:/{run_id}/model"

# 注册模型
result = mlflow.register_model(model_uri=model_uri, name=model_name)

client = MlflowClient()

# 写入版本级元数据
client.set_model_version_tag(model_name, result.version, "dataset", "orders_2026_03_25")
client.set_model_version_tag(model_name, result.version, "git_commit", "a1b2c3d")
client.set_model_version_tag(model_name, result.version, "validation_status", "approved")

# 用 alias 指向当前推荐部署版本
client.set_registered_model_alias(model_name, "champion", result.version)

approved_uri = f"models:/{model_name}@champion"
assert approved_uri == f"models:/MyModel@champion"
```

这里有一个当前工程上很关键的细节：MLflow 官方最新文档仍保留 `transition_model_version_stage(...)` 示例，但也明确说明 model stages 已废弃并将在未来主版本移除。因此，新系统更适合把“审批状态”放到 tag、description、外部审批系统或 alias 绑定逻辑中，而不是把 stage 当长期设计中心。

如果你维护的是旧系统，仍可能看到这样的代码：

```python
from mlflow import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=3,
    stage="Production"
)
```

它能工作，但不应作为新架构的首选。

再看一个真实工程例子。假设你在做风控模型发布，流水线在训练结束后要把模型推入注册表，并要求人工审核后再切生产。SageMaker 的审批调用更贴近“显式状态变更”这个思路：

```python
import boto3

sm_client = boto3.client("sagemaker")

model_package_arn = "arn:aws:sagemaker:region:account:model-package/my-group/6"

response = sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved"
)

assert response["ModelPackageArn"] == model_package_arn
```

官方文档明确指出，当状态从 `PendingManualApproval` 变成 `Approved` 时，可以触发后续 CI/CD。也就是说，模型仓库不是文档柜，而是发布系统的控制点。

下面把注册接口里常见输入做一个压缩总结：

| 输入参数 | 作用 | 典型内容 |
|---|---|---|
| `model_uri` / 模型路径 | 指向实际模型产物 | `runs:/<run_id>/model`、对象存储路径 |
| `name` / model group | 定义模型族 | `fraud-detector` |
| `version` | 唯一识别某次注册 | `6` |
| `tag` / metadata | 存治理信息 | 数据集、commit、owner、任务类型 |
| `approval/status` | 控制是否允许部署 | `Pending`、`Approved`、`Rejected` |
| `alias` | 提供稳定引用 | `champion`、`latest`、`canary` |

---

## 工程权衡与常见坑

最常见的错误不是“不会注册模型”，而是“只注册文件，不注册上下文”。

第一个坑是缺元数据。没有记录数据集版本、训练代码版本、依赖环境、负责人时，模型仓库会退化成一个高级文件夹。你能看到 `V12`，但不知道它和 `V11` 的差异是什么，也无法在事故后做责任追踪。

第二个坑是缺评估结果。审批人不是训练脚本，他必须看到证据。至少要把核心离线指标、对比基线、数据时间窗口、关键分桶结果一起归档。否则审批流会卡死，因为没人能证明“这版确实更适合生产”。

第三个坑是跳过审批状态更新。有些团队做法是：模型虽然注册了，但部署脚本直接拿“最新版本”上线，不读审批字段。这样 registry 只是摆设。真正的自动化链路应该是“读取 Approved 版本”或“读取绑定到 `champion` alias 的版本”，而不是默认拿数字最大的版本。

第四个坑是把版本号当业务语义。比如把 `V8` 解释成“8号线上模型”。版本号只是注册次序，不能替代环境概念。环境应通过 alias、tag、部署记录或发布系统表达。

第五个坑是忽略 lineage。lineage 可以理解为血缘关系，也就是“这个模型是由哪次 run、哪份数据、哪段代码生成的”。没有 lineage，排查时只能靠人工回忆。MLflow、Azure ML 这类系统的价值之一，就是把 run、模型、环境、部署串起来。

| 常见坑 | 后果 | 规避措施 |
|---|---|---|
| 缺 metadata | 无法搜索、对比、审计 | 强制记录数据集、commit、owner、环境 |
| 缺评估结果 | 审批无依据 | 评估报告与模型版本一起归档 |
| 跳过审批流 | 自动部署失控 | 部署只消费 `Approved` 或 alias |
| 用“最新版本”直接上线 | 把未验证模型推到生产 | 区分“最新注册”和“最新批准” |
| 无 lineage | 无法复现和追责 | 将 run、数据、代码版本全部关联 |

新手版例子很典型：没有记录评估指标时，审批人看到两个模型 `V9` 和 `V10`，只知道它们都“训练成功了”，却无法判断哪个可以去生产。结果往往是审批长期挂起，或者靠聊天记录拍脑袋上线。这两种都不合格。

---

## 替代方案与适用边界

不是所有团队一开始都需要完整模型仓库。选择要看协作复杂度和上线风险。

如果团队还处于研究阶段，成员少、模型不直接承载生产流量，那么“对象存储 + 规范命名 + README + 评估报告”可以是短期方案。它足够便宜，也容易理解。但它的上限很明确：无法天然支持审批、回滚、自动部署和跨团队审计。

如果团队已经进入稳定发布阶段，尤其是需要多人协作、版本回滚、审批责任和流水线联动，那么应尽快使用正式 registry。MLflow 适合自建或平台中立场景；Azure ML 和 SageMaker 更适合已经深度绑定对应云平台的团队。

一个常见迁移路径是：科研组先用文件夹管理模型，只保存“模型文件 + 简短说明”；等模型真正要对外提供服务时，再迁移到 MLflow Registry 或云厂商 Registry，把版本、元数据、审批和部署记录补齐。这是合理路径，但前提是你承认前期方案只是过渡，不是长期治理方案。

| 方案 | 自动记录程度 | 审批能力 | Traceability 可追溯性 | 适用边界 |
|---|---|---|---|---|
| 文件夹 + README | 低 | 基本没有 | 低 | 个人研究、小团队原型 |
| 自建 MLflow Registry | 中到高 | 可通过 alias/tag/外部流程实现 | 高 | 中小团队、跨云环境 |
| Azure ML Registry | 高 | 常与平台治理和发布流程结合 | 高 | Azure 生态、需要资产治理 |
| SageMaker Model Registry | 高 | 内建审批状态更明确 | 高 | AWS 生态、强 CI/CD 集成 |

这里还要补一个准确性边界：不同平台对“审批”支持方式并不完全一样。SageMaker 明确暴露 `PendingManualApproval/Approved/Rejected` 这类模型审批状态；MLflow 当前更强调版本、alias、tags 和 lineage；Azure ML v2 更强调模型资产版本化、标签和生命周期管理，审批通常会和 DevOps 流程、部署门禁或组织治理策略结合使用，而不是所有场景都暴露一个统一的内建审批枚举。设计时不要把某个平台的术语硬套到另一个平台。

---

## 参考资料

| 来源 | 要点 | 适用场景 |
|---|---|---|
| MLflow Model Registry 官方文档: https://mlflow.org/docs/latest/ml/model-registry | 说明 Registered Model、Model Version、alias、tags、lineage 的核心概念 | 自建或平台中立的模型注册体系 |
| MLflow Model Registry Workflow: https://mlflow.org/docs/latest/ml/model-registry/workflow/ | 展示注册、alias 管理，以及已废弃的 stage 工作流 | 需要理解 MLflow 新旧发布方式差异 |
| Amazon SageMaker Model Registry 文档: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-models.html | 说明 Model Group、版本递增、版本管理和对比 | AWS 上的模型治理与部署 |
| Amazon SageMaker 审批状态文档: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-approve.html | 说明 `PendingManualApproval`、`Approved`、`Rejected` 及其与 CI/CD 的关系 | 需要显式审批门禁的上线流程 |
| Azure ML 注册模型文档: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2 | 说明模型作为资产进行注册、版本化、打 tag、归档 | Azure ML v2 模型资产管理 |
| Azure ML 架构与注册概念: https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture?view=azureml-api-1 | 说明同名模型注册时版本递增与 tags 搜索 | 理解 Azure 中模型版本和元数据 |
| Azure ML MLOps 文档: https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-1 | 说明模型管理、lineage、治理与自动化部署的关系 | 需要把模型注册接入 MLOps 全流程 |
