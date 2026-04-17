## 核心结论

Helm 是 Kubernetes 上的“标准化打包与发布工具”。它把原本分散的 Deployment、Service、ConfigMap、HPA 等 YAML 文件，组织成一套可复用、可参数化、可追踪历史的发布单元。对 ML 平台来说，这个抽象很重要，因为模型服务、推理网关、向量数据库、任务调度器都不是“一次性部署”，而是需要在开发、测试、预发、生产之间重复安装、升级、回滚。

先给出三个核心概念：

| 概念 | 作用 | 典型输入 |
|------|------|----------|
| Chart | 封装 Kubernetes 模板和默认值的发布包 | `Chart.yaml`、`templates/`、`values.yaml` |
| Release | Chart 在某个集群和命名空间中的一次实例化结果 | `helm install myapp ./mychart` |
| Repository | 托管 Chart 压缩包和索引的仓库 | `index.yaml` + `*.tgz` HTTP 服务 |

给新手的比喻可以这样理解：Chart 是“带默认参数的厨师食谱”，Release 是“某个厨房按食谱实际做出的那份菜”，Repository 是“云端食谱库”。`helm install` 的动作，就是去仓库拿食谱，再根据本地参数下单，最后在 Kubernetes 集群里真正做出这份菜。

对 ML 平台组件来说，Helm 的价值不是“少写几份 YAML”，而是把“部署方式”变成可以复制、审计、版本化的工程资产。只要团队把 vLLM、Prometheus、Ray、MinIO 这类组件都封装成 Chart，后续环境复制、参数切换、灰度升级、故障回滚都会更稳定。

---

## 问题定义与边界

问题定义可以写成一句公式：

$$
可复用部署 = Chart 模板 + 参数输入 + Release 管理
$$

这里的“可复用部署”不是指“某一次 YAML 能跑起来”，而是指同一个组件在多个环境中都能按统一方法部署，并且知道“这次装了什么、改过什么、能不能退回上一版”。

如果不使用 Helm，团队常见做法是手动维护多份 YAML：

- 开发环境一份 `deployment-dev.yaml`
- 生产环境一份 `deployment-prod.yaml`
- GPU 机型一份
- CPU 机型一份
- 又加一个服务发现配置一份

这样做短期看起来直接，长期会出现三个问题：

- 配置漂移：不同环境的差异散落在多份文件里，谁覆盖了谁不清楚。
- 升级不可追踪：改过镜像 tag、资源限制、环境变量后，缺少明确发布历史。
- 回滚成本高：线上异常时，只能手工找旧 YAML，重新 apply，风险高。

Helm 解决的是“标准化部署与版本管理”问题，边界也要说清楚：

- 只讨论 Kubernetes 场景。
- 只讨论 Helm 的 Chart、Values、Template、Release 机制。
- 不展开 Helmfile、Kustomize、Argo CD 的细节实现。
- 不讨论应用内部逻辑，只讨论部署层的参数化和发布流程。

一个玩具例子可以帮助理解。假设你要部署一个最简单的 `hello-api` 服务，它只需要一个 Deployment 和一个 Service。手写 YAML 没问题。但一旦你要支持：

- 副本数可变
- 镜像 tag 可变
- 端口可变
- 是否开启 Ingress 可变

那么“配置项”就已经超过了单文件硬编码的舒适区。Helm 的本质，就是把“固定结构”和“可变参数”拆开管理。

---

## 核心机制与推导

Helm 最重要的机制不是模板语法，而是 `.Values` 的来源和覆盖规则。`.Values` 可以理解成“模板在渲染时能读到的参数对象”。

其优先级可以写成：

```text
最终 .Values = Merge(chart:values.yaml, parent:values.yaml, -f values files in order, --set overrides)
```

从低到高通常是：

1. Chart 自带的 `values.yaml`
2. 父 Chart 的 `values.yaml`，主要影响子 Chart
3. 命令行 `-f` 传入的值文件，支持多个，后者覆盖前者
4. `--set`、`--set-string`，优先级最高

这里有两个初学者必须知道的细节。

第一，Helm 对对象通常做深度合并。深度合并的意思是：如果 `image.repository` 和 `image.tag` 分散在不同文件中，最后可以合成一个完整对象，而不是整块丢掉。

第二，数组通常不是合并，而是整体替换。整体替换的意思是：如果前一个文件里定义了两个 tolerations，后一个文件只写了一个，那么最终通常只剩后一个文件那一个，不会自动拼起来。

看一个最小例子。假设 `values.yaml` 内容是：

```yaml
replicaCount: 1
image:
  repository: myapp
  tag: v1
resources:
  limits:
    cpu: "1"
```

`prod.yaml` 内容是：

```yaml
replicaCount: 2
resources:
  limits:
    cpu: "4"
```

执行：

```bash
helm install myapp ./mychart -f values.yaml -f prod.yaml --set replicaCount=3 --set image.tag=v2.1
```

最终关键结果是：

- `replicaCount = 3`，因为 `--set` 优先级最高
- `image.tag = v2.1`，也来自 `--set`
- `image.repository = myapp`，保留自默认值
- `resources.limits.cpu = 4`，来自 `prod.yaml`

这套规则决定了 Helm 的工程设计方法：默认配置放 Chart，自定义环境放值文件，临时覆盖才用 `--set`。

再看 Helm 的执行流程。一次 `helm install` 或 `helm upgrade`，大致经过四步：

1. 拉取或读取 Chart
2. 合并 values
3. 用 Go template 渲染 `templates/` 下的 YAML
4. 把渲染结果提交给 Kubernetes API

其中 Go template 可以理解成“在 YAML 里嵌入变量和条件逻辑的模板语言”。例如：

```yaml
spec:
  replicas: {{ .Values.replicaCount }}
```

这表示渲染时从 `.Values` 里取 `replicaCount` 填进去。

如果再进一步抽象，Helm 其实在做这样一件事：

$$
渲染后的 Kubernetes 资源 = 模板结构 + 运行时参数
$$

也就是说，Chart 负责定义“资源长什么样”，Values 负责定义“这次部署具体要什么数值”，Release 负责保存“这次实际装出来的版本历史”。

---

## 代码实现

先看一个玩具例子。我们要把一个最简单的 Web 服务做成 Helm Chart，目录通常像这样：

```text
mychart/
  Chart.yaml
  values.yaml
  values.schema.json
  templates/
    deployment.yaml
    service.yaml
```

其中 `values.yaml` 是默认参数：

```yaml
replicaCount: 1

image:
  repository: nginx
  tag: "1.27"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
```

`templates/deployment.yaml` 可以这样写：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: app
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - containerPort: {{ .Values.service.port }}
```

这里有两个术语要解释：

- `.Release.Name`：Release 名称，也就是这次安装实例的名字。
- `.Values.xxx`：模板渲染时读取的参数路径。

对应的 `templates/service.yaml`：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}
spec:
  type: {{ .Values.service.type }}
  selector:
    app: {{ .Release.Name }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: {{ .Values.service.port }}
```

部署命令通常写成：

```bash
helm upgrade --install --namespace default hello-web . -f values.yaml
```

`upgrade --install` 的意思是：如果 Release 不存在就安装，存在就升级。这个写法适合 CI/CD，因为它是幂等的。幂等的意思是同一条命令重复执行，结果保持一致，而不是越跑越乱。

再看一个真实工程例子：把 vLLM 推理服务封装成 Helm Chart。vLLM 是一个大模型推理引擎，常用于部署 OpenAI 兼容接口或高吞吐推理服务。它的部署通常不只是一个容器，还会涉及：

- GPU 资源申请
- 模型存储挂载
- Service 暴露
- HPA 自动扩缩容
- Secret 注入对象存储或模型仓库凭据

一个典型命令类似：

```bash
helm upgrade --install --namespace=ns-vllm test-vllm . \
  -f values.yaml \
  --set secrets.s3bucketname=my-bucket
```

对应模板里，Deployment 可能这样读取参数：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: vllm
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          args:
            - "--model={{ .Values.model.name }}"
            - "--tensor-parallel-size={{ .Values.model.tensorParallelSize }}"
          resources:
            limits:
              nvidia.com/gpu: {{ .Values.resources.limits.gpu }}
          env:
            - name: S3_BUCKET
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.secrets.name }}
                  key: s3bucketname
```

如果再加上 `values.schema.json`，就可以在安装前校验关键字段。例如要求 `replicaCount` 必须是整数，`image.repository` 必须存在，`resources.limits.gpu` 不能漏填。它的作用相当于“给 values 文件加类型约束”，能把很多错误提前到渲染前暴露。

下面给一个可运行的 Python 小程序，模拟 Helm 值覆盖的核心思想：对象深度合并，标量后者覆盖，数组整体替换。

```python
from copy import deepcopy

def merge_values(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        result = deepcopy(base)
        for k, v in override.items():
            if k in result:
                result[k] = merge_values(result[k], v)
            else:
                result[k] = deepcopy(v)
        return result
    # Helm 常见行为里，数组通常由后者整体替换
    return deepcopy(override)

chart_values = {
    "replicaCount": 1,
    "image": {"repository": "myapp", "tag": "v1"},
    "resources": {"limits": {"cpu": "1"}},
    "tolerations": [{"key": "gpu", "effect": "NoSchedule"}],
}

prod_values = {
    "replicaCount": 2,
    "resources": {"limits": {"cpu": "4"}},
    "tolerations": [{"key": "spot", "effect": "NoSchedule"}],
}

set_values = {
    "replicaCount": 3,
    "image": {"tag": "v2.1"},
}

final_values = merge_values(chart_values, prod_values)
final_values = merge_values(final_values, set_values)

assert final_values["replicaCount"] == 3
assert final_values["image"]["repository"] == "myapp"
assert final_values["image"]["tag"] == "v2.1"
assert final_values["resources"]["limits"]["cpu"] == "4"
assert final_values["tolerations"] == [{"key": "spot", "effect": "NoSchedule"}]

print("merged ok:", final_values)
```

这段代码不是 Helm 源码，只是把 Helm 值覆盖的关键直觉用最小程序表达出来，方便新手建立模型。

---

## 工程权衡与常见坑

Helm 好用，但前提是参数设计清楚。真正让团队吃亏的，往往不是模板语法，而是 values 结构混乱。

建议的多环境组织方式是：

- `values.yaml` 放通用默认值
- `values-dev.yaml` 放开发环境差异
- `values-prod.yaml` 放生产环境差异

安装时按“从通用到特定”的顺序传入：

```bash
helm upgrade --install myapp ./mychart -f values.yaml -f values-prod.yaml
```

这样做的原因是覆盖顺序可读。看到命令的人能直接知道：先吃默认值，再吃生产差异。

常见坑可以总结如下：

| 坑 | 原因 | 规避 |
|----|------|------|
| `-f` 顺序错误 | Helm 按顺序覆盖，后者生效 | 固定为“默认到环境”的顺序 |
| `--set` 层级写错 | 点路径表示嵌套字段 | 维护统一的 values 结构文档 |
| 数组被覆写 | 数组通常整体替换，不做拼接 | 在环境文件中显式重写完整数组 |
| 默认值过少 | 每次部署都靠命令行补参数 | 把稳定字段沉淀到 `values.yaml` |
| 默认值过多 | 把环境差异也塞进默认值 | 只保留跨环境通用参数 |
| 模板逻辑过重 | 在模板里写太多条件分支 | 把复杂判断前移到 values 设计 |
| 缺少 schema 校验 | 漏字段要等部署时报错 | 使用 `values.schema.json` 预校验 |

这里给一个典型误区。很多团队喜欢把所有内容都写进 `--set`：

```bash
helm upgrade --install myapp . \
  --set image.repository=repo/app \
  --set image.tag=v2 \
  --set resources.limits.cpu=4 \
  --set resources.limits.memory=8Gi
```

短期看方便，长期问题很大：

- 命令不可读
- 历史难复现
- 层级路径容易拼错
- shell 转义复杂

`--set` 更适合临时覆盖少量字段，例如镜像 tag、单个 Secret 名称、一次性调试参数，而不是承载整套环境配置。

对 ML 场景还有一个特殊坑：GPU、模型路径、对象存储、节点亲和性经常是数组或复杂对象。只要这些字段被多个值文件同时修改，就容易出现“以为追加，实际替换”的问题。处理方法不是赌模板聪明，而是让环境文件显式完整。

---

## 替代方案与适用边界

Helm 不是唯一方案，它的优势在“模板化 + 参数化 + Release 历史”。如果你的需求不在这个组合上，替代方案可能更合适。

| 方案 | 优势 | 何时选 |
|------|------|--------|
| Helm Chart | 参数化、版本化、可回滚 | 多环境、多组件、标准化部署 |
| Plain YAML | 直接、无额外抽象 | 单次、小规模、临时任务 |
| Kustomize | Overlay 和 patch 更灵活 | 需要大量资源拼接和差异补丁 |
| GitOps + Helm | 声明式同步和审计更强 | 多团队协作、集群持续同步 |

Plain YAML 适合什么场景？比如 ML 团队做一次内部小测试，只起一个临时推理服务，生命周期很短，也没有多环境复用需求。这时直接 `kubectl apply -f` 更简单，没有必要先引入 Chart 结构。

Kustomize 适合什么场景？如果你的基础 YAML 已经比较稳定，主要工作是做 patch 和 overlay，比如不同环境替换节点选择器、标签、探针策略，那么 Kustomize 的资源拼接会更直接。

Helm 更适合什么场景？当你要管理的是“产品化组件”，比如 vLLM 服务、内部模型网关、日志采集器、监控组件，且需要：

- 参数输入清晰
- 版本迭代可追踪
- 发布能回滚
- 不同团队能重复安装

这时 Helm 的收益明显高于维护散乱 YAML。

再进一步，如果多个团队共享一个或多个 Release，仅靠 Helm 命令还不够。因为 Helm 负责“渲染和发布”，但不负责“持续对齐 Git 状态和集群状态”。这种时候通常会引入 Argo CD 或 Flux。它们的关系是：

- Helm：负责把模板 + 参数变成 Kubernetes 资源
- GitOps 工具：负责持续同步、漂移检测、权限审计

所以边界也很清楚：Helm 不是 GitOps 替代品，但它常常是 GitOps 体系里的模板层。

---

## 参考资料

- Helm 官方 Glossary：理解 Chart、Release、Repository 的定义与关系
- Helm 官方 Chart Template Guide：学习 Go template、`.Values`、`include`、函数管道
- Helm 官方 Values Files：理解 values 覆盖优先级、多文件叠加、命令行覆盖
- vLLM 官方 Helm 部署文档：查看真实推理服务的 Chart 组织方式与安装命令
- Helm 官方 Charts 文档：了解 Chart 目录结构、`Chart.yaml`、依赖与打包方式
- Helm 官方 Schema Files 相关说明：了解 `values.schema.json` 的参数校验能力
