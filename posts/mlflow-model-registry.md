## 核心结论

MLflow 的核心价值不是“记日志”，而是把模型从实验到上线的链路标准化。最重要的链条是：

$$
\text{Experiment} \rightarrow \text{Run} \rightarrow \{\text{Params}, \text{Metrics}_t, \text{Artifacts}_{files}\}
$$

其中：

- `Experiment` 是一次任务集合。白话说，它像“同一个课题的文件夹”。
- `Run` 是一次具体训练。白话说，它像“你按下训练按钮后产生的一次完整记录”。
- `Params` 是超参数。
- `Metrics_t` 是带时间步或训练步的指标序列。
- `Artifacts` 是文件产物，比如模型权重、图表、配置文件。

只做 Tracking，你只能回答“这次训练跑了什么、指标是多少”；再加上 Model Registry，你才能回答“当前线上到底用哪个模型版本、它是何时从测试转到生产的、能否回滚”。

对零基础到初级工程师，最直接的理解是：先把一次训练完整记录下来，再把训练出的模型注册成一个可版本化的资产，最后用 `Staging` 和 `Production` 这样的标准阶段控制发布。这样“实验结果最好”与“最终谁上线”之间不再靠口头沟通，而是靠系统记录。

下面这张表可以先建立整体心智模型：

| 对象 | 作用 | 典型内容 | 是否直接参与部署 |
|---|---|---|---|
| Experiment | 组织一组 Run | 任务名、说明 | 否 |
| Run | 记录一次训练过程 | 参数、指标、artifact | 间接 |
| Artifact | 保存文件产物 | 模型文件、图、日志 | 是 |
| Registered Model | 模型逻辑名称 | 如 `FraudNet` | 是 |
| Model Version | 某次注册后的具体版本 | `FraudNet` v3 | 是 |
| Stage | 模型阶段状态 | `Staging`、`Production`、`Archived` | 是 |

玩具例子：你做一个二分类小模型，调了 5 次学习率。Tracking 解决“第 3 次的 AUC 是不是最高”；Registry 解决“第 3 次是否已经批准上线”。

真实工程例子：风控团队每天重训欺诈检测模型。训练脚本把每次结果写入 MLflow，CI 根据验证集指标把某个版本推进到 `Staging`，人工验收后切到 `Production`，Kubernetes 的推理服务固定加载 `models:/FraudNet/Production`。这样线上服务只关心“生产阶段的模型是谁”，而不关心底层文件路径。

---

## 问题定义与边界

MLflow 主要解决两类工程问题。

第一类问题是“找不到最佳训练结果”。很多团队最初只在本地保存 `model.pt`、`train.log`、`config.yaml`，几周后就分不清哪个模型对应哪个参数组合。结果不是模型训练不出来，而是训练出来后无法复现。

第二类问题是“模型上线路径不透明”。训练同学说“我把最好的模型发你了”，平台同学说“我部署的是昨天那个包”，最后线上模型来源不清楚，回滚也靠人工记忆。

MLflow 的边界也很明确。它不是完整的训练平台，不负责数据标注、特征存储、在线流量切分这些全链路能力。它主要负责三件事：

1. 记录实验元数据
2. 以统一格式打包模型
3. 管理模型版本与生命周期状态

一个容易踩坑的边界是后端存储能力。

| 模式 | 能力 | 后端要求 | 适合场景 |
|---|---|---|---|
| Tracking-only | 记录 Run、参数、指标、artifact | 本地文件或远程存储都可 | 个人实验、早期 PoC |
| Tracking + Registry | 额外支持模型注册、版本、阶段切换 | 需要数据库后端，通常还要对象存储 | 多人协作、上线审计 |
| 仅本地 `mlruns/` | 可查看实验 | 通常无 Registry 能力 | 学习与原型验证 |

白话说，如果你只用本地 `mlruns/` 文件夹，通常只能把它当“实验记录本”，不能把它当“生产级模型注册中心”。

Run 与模型版本的关系可以写成：

$$
\text{Run}_i \xrightarrow{\text{log\_model}} \text{Model Version}_j
$$

也就是说，模型版本不是凭空产生的，它通常对应某个具体 Run 的产物。于是你可以从线上模型版本反查训练参数、代码版本、评估指标。

玩具例子：`Run123` 记录了 `lr=1e-3`、`batch_size=32`、`auc=0.92`。你把这个 Run 的模型注册成 `FraudNet` 的第 3 个版本，那么 `FraudNet v3` 就能追溯回 `Run123`。

真实工程例子：推荐系统团队把 Tracking Server 连到 PostgreSQL，把模型文件存到 S3。这样做的原因不是“技术更高级”，而是线上模型必须可审计：谁注册的、何时切换到生产、是否有审批记录，都要可查。

---

## 核心机制与推导

MLflow 的核心机制可以拆成四步。

### 1. 用 Experiment 组织任务

同一个业务问题下，所有训练尝试放在一个 Experiment 里。比如“欺诈检测”“CTR 预估”“图像分类”分别建不同 Experiment。

### 2. 用 Run 记录一次完整训练

每次训练开一个 Run，在 Run 里记录：

- 参数：如学习率、层数、随机种子
- 指标：如 loss、AUC、F1
- 文件：如模型权重、混淆矩阵、特征列表

如果一个指标会随训练步变化，它实际上是一个序列：

$$
\text{Metrics}_{auc} = \{(step_1, 0.81), (step_2, 0.86), (step_3, 0.89)\}
$$

这就是为什么 `log_metric` 可以带 `step` 参数。白话说，你不是只记“最终分数”，而是记“分数如何随训练过程变化”。

### 3. 用 `log_model` 把模型打包成标准产物

`mlflow.<flavor>.log_model(...)` 中的 `flavor` 指模型框架适配器。白话说，它告诉 MLflow “这到底是 PyTorch 模型、sklearn 模型，还是通用 Python 函数模型”。

例如：

```python
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    registered_model_name="FraudNet"
)
```

这一步会产生两层结果：

1. 当前 Run 下生成一个模型 artifact
2. 如果指定了 `registered_model_name`，还会把它注册到 Registry，生成新版本

因此，逻辑路径会从：

- `runs:/<run_id>/model`

扩展为：

- `models:/FraudNet/3`
- `models:/FraudNet/Production`

前者按 Run 取模型，后者按注册中心版本或阶段取模型。

### 4. 用 Stage 管理发布状态

`Stage` 是模型处于哪个发布阶段。白话说，它表示“这个版本现在该拿来做什么”。

常见阶段：

| Stage | 含义 | 常见动作 |
|---|---|---|
| `Staging` | 待验证 | 集成测试、灰度验证 |
| `Production` | 正式生产 | 在线服务加载 |
| `Archived` | 归档 | 不再对外服务，但保留追溯 |

典型脚本化操作是：

```python
client.transition_model_version_stage(
    name="FraudNet",
    version=3,
    stage="Staging"
)
```

于是可以形成一条完整闭环：

$$
\text{Train Code} \rightarrow \text{Run Record} \rightarrow \text{Model Version} \rightarrow \text{Stage Transition} \rightarrow \text{Deployment}
$$

玩具例子：你训练一个小型二分类器，`step=5` 时 `auc=0.92`，然后将模型注册成 `FraudNet v3`，再把它切到 `Staging`。这就表示“这个版本已经不只是一个训练文件，而是一个进入发布流程的候选模型”。

真实工程例子：在 Kubernetes 上，推理容器不再直接挂载某个固定文件，而是启动时加载 `models:/FraudNet/Production`。这样发布动作变成“切阶段”，而不是“手工改路径”。

常用 API 可以先记住下面这张表：

| API | 作用 | 典型输入 | 输出/效果 |
|---|---|---|---|
| `mlflow.log_param` | 记录参数 | `("lr", 1e-3)` | Run 下新增参数 |
| `mlflow.log_metric` | 记录指标 | `("auc", 0.92, step=5)` | Run 下新增指标点 |
| `mlflow.log_artifact` | 上传文件 | 本地文件路径 | Run 下新增 artifact |
| `mlflow.pytorch.log_model` | 记录并打包 PyTorch 模型 | 模型对象、artifact 路径 | 生成模型 artifact |
| `MlflowClient.transition_model_version_stage` | 切换模型阶段 | 名称、版本、阶段 | 更新 Registry 状态 |

---

## 代码实现

下面给一个可运行的玩具实现。它不依赖真实 MLflow 服务，先用简化代码把对象关系跑通，再给出真实 MLflow 写法。这样零基础读者可以先理解机制，再接触完整工程接口。

```python
from dataclasses import dataclass, field

@dataclass
class RunRecord:
    params: dict = field(default_factory=dict)
    metrics: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value, step):
        self.metrics.append((key, step, float(value)))

    def log_artifact(self, name, payload):
        self.artifacts[name] = payload

@dataclass
class Registry:
    models: dict = field(default_factory=dict)   # name -> versions
    stages: dict = field(default_factory=dict)   # (name, stage) -> version

    def register_model(self, name, run_record):
        versions = self.models.setdefault(name, [])
        versions.append(run_record)
        return len(versions)

    def transition_stage(self, name, version, stage):
        assert stage in {"Staging", "Production", "Archived"}
        assert 1 <= version <= len(self.models.get(name, []))
        self.stages[(name, stage)] = version

    def load_by_stage(self, name, stage):
        version = self.stages[(name, stage)]
        return self.models[name][version - 1]

run = RunRecord()
run.log_param("lr", 1e-3)
run.log_param("batch_size", 32)
run.log_metric("auc", 0.91, step=1)
run.log_metric("auc", 0.92, step=5)
run.log_artifact("model", {"weights": [0.1, 0.2, 0.3]})

registry = Registry()
version = registry.register_model("FraudNet", run)
registry.transition_stage("FraudNet", version, "Staging")
loaded = registry.load_by_stage("FraudNet", "Staging")

assert version == 1
assert loaded.params["lr"] == 1e-3
assert loaded.metrics[-1] == ("auc", 5, 0.92)
assert "model" in loaded.artifacts
```

上面这段代码体现了三件事：

1. Run 负责承载训练记录
2. Registry 负责把 Run 产物变成版本
3. Stage 负责决定当前应加载哪个版本

真实工程里，对应的 MLflow 训练与注册代码通常如下：

```python
import mlflow
import mlflow.pytorch
import torch
from mlflow.models import infer_signature

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = TinyNet()
x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
y = model(x).detach().numpy()
signature = infer_signature(x.numpy(), y)

mlflow.set_experiment("fraud-detect")

with mlflow.start_run() as run:
    mlflow.log_param("lr", 1e-3)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("loss", 0.42, step=1)
    mlflow.log_metric("auc", 0.92, step=5)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="FraudNet",
    )

    print("run_id =", run.info.run_id)
```

推理端可以按阶段加载：

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/FraudNet/Production")
```

如果要把模型服务化并做容器部署，常用 CLI 如下：

| 命令 | 作用 |
|---|---|
| `mlflow ui` | 本地查看实验界面 |
| `mlflow server` | 启动远程 Tracking Server |
| `mlflow models build-docker -m runs:/<run_id>/model --enable-mlserver` | 基于模型构建 Docker 镜像 |
| `mlflow models serve -m models:/FraudNet/Production` | 本地启动推理服务 |

真实工程例子可以这样理解：训练作业跑在 GPU 节点，结束后把 PyTorch 模型通过 `mlflow.pytorch.log_model` 打包；CI 再执行 `build-docker` 产出镜像；镜像发布到 Kubernetes，KServe 或 Seldon 拉起推理服务。此时部署系统管理的是“镜像”和“服务实例”，MLflow 管理的是“这个服务应该指向哪个模型版本”。

---

## 工程权衡与常见坑

MLflow 的优势是可追溯、可回滚、可协作；代价是你必须接受额外的基础设施和流程约束。

第一类权衡是后端复杂度。个人开发时，本地文件足够简单；团队上线时，数据库和对象存储几乎是必需的。原因不是功能炫技，而是 Registry、本地容灾、多人并发访问都要求更稳定的后端。

第二类权衡是流程标准化。标准化会降低灵活度，但能减少线上事故。比如阶段名称应该统一，而不是今天叫 `beta`、明天叫 `preprod`。

常见坑如下：

| 坑 | 现象 | 规避方法 |
|---|---|---|
| 只用本地 file backend 就想用 Registry | 界面里只能看到 Run，看不到完整注册流程 | 生产环境使用数据库后端，并配合对象存储保存 artifact |
| 不记录 `signature` | 推理时输入 schema 不稳定，服务端类型推断可能出错 | 在 `log_model` 时显式传 `infer_signature` |
| 阶段命名混乱 | 发布流程靠约定，脚本无法稳定工作 | 固定使用标准阶段名并统一权限流程 |
| 只保存模型文件，不保存参数与指标 | 模型能跑，但无法解释它为何上线 | 保证每次 Run 至少记录关键超参数、核心评估指标、训练代码版本 |
| 直接用 Run URI 上线 | 线上路径绑死到某次实验产物，难以统一切换 | 稳定环境优先使用 `models:/<name>/Production` |

`signature` 的本质是输入输出模式约束。可以简单理解成：

$$
\text{signature} = (\text{inputs schema}, \text{outputs schema})
$$

白话说，它是在告诉部署系统：“这个模型要吃什么形状和类型的数据，会吐出什么格式的结果。”

玩具例子：你本地测试时输入是二维浮点数组 `[[0.1, 0.2]]`，线上却传了字符串列表，没签名时错误可能要到运行期才暴露；有签名时可以更早发现。

真实工程例子：风控模型上线后，A/B 两个服务都读取 `Production`。如果你把新版本先切到 `Staging` 做回归校验，再提升到 `Production`，回滚就只需要把旧版本重新切回生产阶段，而不是重新打包和分发模型文件。

---

## 替代方案与适用边界

不是所有团队一开始都要上完整 Registry。

如果你当前只有一个人做实验，模型也不频繁上线，那么先把 Tracking 做好，比立刻引入复杂的发布状态管理更重要。因为最先出现的问题通常不是“阶段切换不规范”，而是“上周跑出来的最好模型找不到了”。

可以按场景做判断：

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| Tracking only | 个人实验、PoC、小团队 | 简单、学习成本低 | 无正式版本治理 |
| Tracking + Registry | 多团队协作、生产发布、审计要求高 | 可追溯、可回滚、可脚本化 | 后端与流程更复杂 |
| 其他平台 Registry | 企业已深度绑定云平台 | 与现有 CI/CD、权限体系整合更深 | 平台耦合更强 |

何时引入 Registry，可以用下面这个判断流程：

| 问题 | 如果答案是“是” | 建议 |
|---|---|---|
| 是否多人协作训练与上线？ | 容易口头交接失真 | 引入 Registry |
| 是否需要明确“当前生产模型是谁”？ | 需要版本和阶段统一入口 | 引入 Registry |
| 是否有审计、回滚、审批要求？ | 需要可追溯链路 | 引入 Registry |
| 只是做算法探索吗？ | 暂无正式发布流程 | 先做 Tracking |

玩具例子：你在本地做手写数字识别，只关心哪组参数效果最好，那么 `mlflow ui` 就已经够用。

真实工程例子：公司已经有 SageMaker 或 GitHub Actions 驱动的模型发布体系，那么 MLflow 也不一定非要接管全部生命周期。它可以只负责实验记录与模型 artifact 标准化，再把产物交给现有平台注册和部署。

因此，适用边界很清楚：

- 先解决 reproducibility，再解决 governance
- 先保证实验可追溯，再升级到生产可发布
- 先统一模型打包格式，再统一上线流程

---

## 参考资料

1. MLflow Tracking 官方文档  
   https://mlflow.org/docs/1.23.1/tracking.html  
   解决什么问题：解释 `Experiment`、`Run`、参数、指标、artifact 的基本层次。

2. MLflow PyTorch / `log_model` 官方文档  
   https://mlflow.org/docs/latest/ml/deep-learning/pytorch/  
   解决什么问题：说明如何把 PyTorch 模型打包进 MLflow，并补齐 `signature` 等部署关键信息。

3. MLflow Model Registry 官方文档  
   https://www.mlflow.org/docs/2.5.0/model-registry.html  
   解决什么问题：说明模型版本、阶段流转、注册中心的生命周期管理。

4. MLflow 部署到 Kubernetes 官方文档  
   https://mlflow.org/docs/latest/ml/deployment/deploy-model-to-kubernetes/  
   解决什么问题：说明如何将模型构建为容器并接入 Kubernetes 推理部署链路。
