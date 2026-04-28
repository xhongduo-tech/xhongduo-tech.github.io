## 核心结论

Vertex AI 模型部署的核心不是“把模型文件传上去”，而是把模型组织成一套可被在线调用的推理服务。最重要的结构是：

`Model + Endpoint + DeployedModel = 可在线调用的推理服务`

这里先把术语说清楚。

| 术语 | 白话解释 | 在部署里的角色 |
|---|---|---|
| `Model` | 已登记到 Vertex AI 的模型资产，可以理解为“可被部署的模型描述” | 表示“要部署什么” |
| `Endpoint` | 对外暴露推理能力的访问入口，可以理解为“接请求的门面” | 表示“请求打到哪里” |
| `DeployedModel` | 挂载到某个 Endpoint 上的一份具体部署实例 | 表示“用什么容器、多少机器、怎么服务” |
| `Custom Container` | 自定义容器，即你自己提供推理镜像 | 表示“推理服务内部怎么跑” |
| `Regional Deployment` | 区域化部署，即资源固定在某个云区域 | 表示“服务实际在哪里” |

对初级工程师来说，最容易忽略的事实有三个。

第一，自定义容器提高了自由度，但把启动脚本、依赖安装、模型加载、健康检查接口这些责任重新交回给你。

第二，专用节点规格与最小副本数决定吞吐基线。也就是说，系统在“流量还没暴涨之前”最低能扛住多少请求，主要由这两个量决定。

第三，调用方、数据和 Endpoint 不在同一区域时，问题不只是“稍微慢一点”，还会直接增加尾延迟与跨区网络成本。

玩具例子可以这样理解：`Model` 像商品，`Endpoint` 像柜台，`DeployedModel` 是“放到柜台里实际售卖的那一份商品”，并且附带了包装方式、仓储位置和供货能力。真正决定用户能不能顺利买到商品的，不是仓库里有没有货，而是柜台能不能稳定出货。

---

## 问题定义与边界

本文要解决的问题是：如何把一个模型稳定部署成 Vertex AI 的在线推理服务。

这里的“在线推理服务”有明确边界。它指的是请求来了之后，服务能在较短时间内返回预测结果，并且具备基本的可达性、可扩缩性和可观测性。它不是训练教程，也不是离线批处理作业说明。

边界先用表格定死：

| 范围 | 包含 | 不包含 |
|---|---|---|
| 本文重点 | 在线预测、容器启动、健康检查、吞吐、副本、区域成本 | 模型训练、特征工程、离线批处理训练 |
| 部署方式 | 自定义容器、预构建容器 | 纯本地推理脚本 |
| 网络视角 | 同区域访问、跨区 egress | 纯内网算法细节 |

为什么要强调边界？因为很多新手会把“模型能在本地跑通”误认为“模型已经可部署”。这两者差别很大。

本地推理脚本只需要你自己调用函数。在线部署则要回答更多问题：

1. 容器启动后谁来监听 HTTP 端口？
2. 健康检查打过来时谁来响应？
3. 模型文件从哪里加载，失败怎么办？
4. 请求格式不规范时返回什么错误？
5. 流量从 10 QPS 突然涨到 100 QPS 时谁来扛？

所以，“上传模型”只是开始，不是完成。

一个新手常见误区是把部署理解成“存储问题”，好像只要把 `model.pkl` 放进 GCS 就够了。实际上 Vertex AI 更关心的是“服务问题”：请求如何进入、进程是否存活、容器是否按约定返回结果、资源是否足以承接流量。

真实工程例子：一个风控服务每天接收线上授信请求，要求百毫秒级返回。这个场景的核心就不是“模型文件存没存好”，而是“服务在白天高峰时是否稳定、是否能快速扩容、是否跨区导致请求变慢”。因此，部署设计天然比模型文件管理更重要。

---

## 核心机制与推导

Vertex AI 在线部署的主链路可以写成：

`请求 -> Endpoint -> DeployedModel -> 容器 predict route -> 返回结果`

如果使用自定义容器，关键机制是：容器启动后必须在指定端口提供 HTTP 服务，并实现平台期望的健康检查与预测接口。平台不会替你补全这层逻辑。

### 1. 自定义容器为什么是核心

自定义容器的白话解释是：你自己决定推理进程怎么启动、依赖怎么装、模型怎么加载、接口怎么暴露。好处是灵活，代价是责任回收。

比如你可以自由选择 FastAPI、Flask、TorchServe、Triton，甚至自己写一个最小 HTTP server。但无论选什么，本质都一样：Vertex AI 只负责把请求转发给你的容器，不负责理解你内部的业务逻辑。

### 2. 吞吐基线怎么估算

为了让问题可计算，先定义几个符号：

| 符号 | 含义 | 作用 |
|---|---|---|
| `q_node` | 单节点稳定吞吐 | 估算单副本能力 |
| `R_min` | 最小副本数 | 决定基础吞吐 |
| `λ` | 目标峰值 QPS | 决定所需副本数 |
| `V` | 跨区传输量 | 估算网络成本 |
| `p_cross_region` | 跨区单价 | 估算 egress 支出 |

一个常用近似是：

$$
B = R_{min} \times q_{node}
$$

这里的 $B$ 是基础吞吐能力，也就是在不依赖额外扩容时的稳态承接上限。

如果目标峰值是 $\lambda$，需要的副本数可以近似写成：

$$
R_{req} = \left\lceil \frac{\lambda}{q_{node}} \right\rceil
$$

这个公式不保证精确，但足以做一线容量规划。因为真正的系统还会受请求大小、模型加载方式、CPU/GPU 利用率、序列化开销和网络抖动影响。

### 3. 玩具例子

假设单节点稳定处理 `25 QPS`，最小副本数设置为 `2`。那么基础吞吐大约是：

$$
B = 2 \times 25 = 50
$$

如果业务峰值到 `90 QPS`，理论所需副本数至少是：

$$
R_{req} = \lceil 90 / 25 \rceil = 4
$$

这说明什么？说明 `minReplicaCount=2` 只能保住低峰稳态，峰值阶段如果扩容不及时，请求会排队，P95 或 P99 延迟会明显上升。

这也是为什么“最小副本数”不是纯成本参数，它本质上是延迟稳定性的下限控制参数。

### 4. 跨区成本怎么估算

跨区调用的成本近似可以写成：

$$
C_{net} = V \times p_{cross\_region}
$$

其中，$V$ 是跨区传输量，$p_{cross\_region}$ 是跨区单价。

例如每月跨区传输 `200 GiB`，假设单价为 `0.05 USD/GiB`，那么网络费用约为：

$$
C_{net} = 200 \times 0.05 = 10 \text{ USD}
$$

这还只是显性成本。隐性成本是 RTT 增加后，尾延迟可能放大，重试也会更多。

### 5. 真实工程例子

设一个金融风控服务把 XGBoost 模型部署到 `us-central1`。容器启动时从 `AIP_STORAGE_URI` 拉取模型，`/predict` 负责接收请求并返回风险分数。业务白天稳定在 `60 QPS`，活动期间峰值 `150 QPS`，单节点压测稳定吞吐 `30 QPS`。

那么：

- 若 `minReplicaCount=1`，基础吞吐只有 `30 QPS`，白天常态都不稳。
- 若 `minReplicaCount=3`，基础吞吐约 `90 QPS`，能覆盖日常流量。
- 峰值 `150 QPS` 至少需要 `ceil(150/30)=5` 个副本。
- 如果调用服务也放在 `us-central1`，就能避免额外跨区 RTT 和网络费。

这里能看出一个典型工程规律：部署不是“先跑起来再说”，而是要先把稳态流量、峰值流量和区域布局算清楚。

---

## 代码实现

下面给出一个最小可运行思路。目标不是覆盖所有配置，而是把自定义容器的三个职责讲清楚：启动服务、加载模型、暴露预测接口。

### 1. Python 里的推理逻辑玩具实现

先用纯 Python 写一个可运行的最小模型，方便理解请求与响应结构：

```python
from math import exp

class SimpleRiskModel:
    def predict_proba(self, rows):
        probs = []
        for income, debt_ratio in rows:
            # 一个玩具打分函数：收入越高风险越低，负债率越高风险越高
            z = -0.00005 * income + 4.0 * debt_ratio - 1.2
            p = 1 / (1 + exp(-z))
            probs.append(round(p, 6))
        return probs

def predict(instances):
    model = SimpleRiskModel()
    rows = [(x["income"], x["debt_ratio"]) for x in instances]
    scores = model.predict_proba(rows)
    return {"predictions": [{"risk_score": s} for s in scores]}

payload = {
    "instances": [
        {"income": 20000, "debt_ratio": 0.7},
        {"income": 80000, "debt_ratio": 0.2},
    ]
}

result = predict(payload["instances"])
assert "predictions" in result
assert len(result["predictions"]) == 2
assert result["predictions"][0]["risk_score"] > result["predictions"][1]["risk_score"]
print(result)
```

这段代码的重点不是算法，而是接口结构：输入是 `instances`，输出是 `predictions`。白话解释是，外部请求进来后，服务把一批样本转换成模型需要的格式，再把结果封装回 JSON。

### 2. FastAPI 容器入口

```python
import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
model = None

class PredictRequest(BaseModel):
    instances: list[dict]

@app.on_event("startup")
def load_artifact():
    global model
    model_path = os.getenv("MODEL_PATH", "/app/model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="model not loaded")

    rows = []
    for item in req.instances:
        rows.append([item["income"], item["debt_ratio"]])

    scores = model.predict_proba(rows)[:, 1].tolist()
    return {"predictions": [{"risk_score": s} for s in scores]}
```

这段代码对应三件事：

1. 进程启动时加载模型。
2. `/health` 用于健康检查。
3. `/predict` 暴露预测接口。

真正部署时，服务必须监听 `0.0.0.0:$PORT`。白话解释是，不监听这个地址和端口，平台就进不来。

### 3. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY model.pkl .

ENV PORT=8080
ENV MODEL_PATH=/app/model.pkl

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
```

这个 Dockerfile 只做四件事：选基础镜像、装依赖、复制代码与模型、启动 HTTP 服务。对新手来说，这已经足够形成最小闭环。

### 4. 部署命令思路

```bash
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --display-name=risk-model-v1 \
  --machine-type=n1-standard-4 \
  --min-replica-count=3 \
  --max-replica-count=8
```

这条命令的重点不是参数背诵，而是理解资源绑定关系：你不是只在“部署模型”，你是在把某个 `Model` 以某种资源规格和副本策略挂到某个 `Endpoint` 上。

### 5. 真实工程例子

假设一个推荐服务需要部署自定义预处理逻辑：请求进来先做特征补齐，再调用模型，再把置信度和候选结果一起返回。这时预构建容器往往不够灵活，自定义容器更合理。容器里通常会包含：

- 启动时从 GCS 拉模型和词表
- `/health` 返回进程是否就绪
- `/predict` 完成特征转换与推理
- 日志输出请求耗时和错误信息

这类场景正是 Vertex AI 自定义容器的主要价值所在。

---

## 工程权衡与常见坑

### 1. `minReplicaCount` 不是越低越省

`minReplicaCount` 决定保底吞吐和冷启动风险。设成 1 的确可能更便宜，但代价是高峰来临时缓冲极弱。

一个常见误判是：平均流量不高，所以副本设 1 就行。问题在于线上系统受的是峰值，不是平均值。只要短时间并发上来，请求就会排队。

比如白天稳定 `40 QPS`、晚上 `5 QPS` 的服务，如果单节点只稳住 `25 QPS`，那你白天至少要让基线覆盖主要流量，而不是指望每次都临时扩容成功。

### 2. `maxReplicaCount` 太低会直接碰天花板

`maxReplicaCount` 是峰值承接上限。它设太低时，自动扩容再聪明也没用，因为系统被人为封顶了。

### 3. 常见坑位表

| 常见问题 | 结果 | 规避方式 |
|---|---|---|
| 没监听 `0.0.0.0:PORT` | 容器不可达 | 按环境变量端口启动 |
| 进程启动后退出 | liveness check 失败 | 保持主进程常驻 |
| 请求/响应 JSON 不符合要求 | 调用失败 | 按 Vertex 约定对齐 |
| `minReplicaCount` 太低 | 冷启动、排队 | 提高保底副本 |
| `maxReplicaCount` 太低 | 高峰丢流量 | 提前压测并放宽上限 |
| 跨区调用 | 延迟和费用上升 | 应用、数据、Endpoint 同区域 |

### 4. 跨区访问常被低估

很多团队先把模型部署起来，后面才发现业务服务在另一区域。结果是服务“能用”，但总觉得慢，而且月账单也开始出现不必要的网络项。

这里的工程原则很直接：如果没有强约束，应用、模型和主要数据源优先同区域部署。

### 5. 自定义容器不是默认最优

自定义容器的好处是自由度高，但这意味着你要对镜像体积、依赖冲突、模型加载时间、服务稳定性负责。只要你的模型能被 Vertex AI 的预构建容器顺利承载，通常应优先评估预构建方案，因为它能显著减少运维面。

---

## 替代方案与适用边界

选择部署方式时，关键不是“哪种最强”，而是“哪种最匹配需求”。

| 方案 | 适合场景 | 优点 | 代价 |
|---|---|---|---|
| 自定义容器 | 框架特殊、启动逻辑复杂 | 自由度高 | 运维责任更大 |
| 预构建容器 | 常见框架模型 | 简单省事 | 灵活性较低 |
| 在线 Endpoint | 低延迟实时请求 | 对外服务稳定 | 需要副本与区域规划 |
| 批处理/离线方案 | 不要求实时响应 | 成本更低 | 不能即时返回 |

### 1. 什么时候用自定义容器

当你的服务需要特殊依赖、复杂预处理、非标准推理框架，或者需要精细控制启动逻辑时，自定义容器通常是合理选择。

### 2. 什么时候更适合预构建容器

如果模型属于常见框架、输入输出格式标准、业务不要求复杂服务逻辑，那么预构建容器更省事。它减少了你维护 HTTP 服务和运行时环境的负担。

### 3. 什么时候不该用在线 Endpoint

如果任务是每天夜里批量跑一次评分结果，或者是离线向量计算、离线报表生成，那就不应为了“看起来现代”而硬上在线 Endpoint。在线服务的核心价值是低延迟实时响应，不是通用计算托管。

换句话说，需求若不是“请求来了马上回”，那在线部署往往不是成本最优。

---

## 参考资料

1. [Use a custom container for inference](https://docs.cloud.google.com/vertex-ai/docs/predictions/use-custom-container)
2. [Custom container requirements for inference](https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements)
3. [DedicatedResources API reference](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/DedicatedResources)
4. [Deploy a model to an endpoint](https://docs.cloud.google.com/vertex-ai/docs/general/deployment)
5. [Vertex AI private services access endpoints](https://docs.cloud.google.com/vertex-ai/docs/predictions/using-private-endpoints)
6. [Google Cloud network pricing](https://cloud.google.com/vpc/network-pricing#service-extensions)
