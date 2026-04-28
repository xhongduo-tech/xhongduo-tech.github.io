## 核心结论

模型版本管理与回滚，不是“替换一个模型文件”，而是把发布、灰度、验证、回切做成一套可审计流程。这里的“可审计”可以先理解为：你能明确回答“谁在什么时间，把哪个版本，以什么流量比例，上到了哪条生产链路”。

它的核心目标也不是避免一切故障，而是把故障限制在小流量、可观测、可快速回退的范围内。对线上系统来说，真正危险的不是“新版本偶尔出错”，而是“新版本直接全量接流量，出错后还不能立刻切回”。

新手先记住最小工作流：先让 5% 流量走新模型 `v_c`，其余 95% 仍走稳定模型 `v_s`。如果新模型的错误率、延迟、输出一致性、显存占用都通过门槛，再逐步放到 20%、50%、100%；一旦任一指标异常，立即切回 `v_s`。这就是版本管理与回滚最重要的操作语义。

核心定义可以先统一如下：

| 符号 | 含义 |
|---|---|
| `v_s` | 稳定版本，已经被验证过、当前主要承接生产流量的版本 |
| `v_c` | 候选版本，准备上线但尚未全量的版本 |
| `p` | 灰度流量比例，指发往候选版本的请求占比 |
| `h` | 健康门禁，先白话理解为“服务当前是否允许接流量” |
| `d` | 输出差异，指候选版本与稳定版本在同类输入上的结果差别 |
| `m` | 显存占用，指模型运行时实际占用的 GPU 显存 |
| `M` | 显存上限，指系统允许的资源上界 |

---

## 问题定义与边界

版本管理解决的是模型上线过程中的风险控制、审计追踪和快速回切，不解决训练本身好不好、数据集质量高不高、特征工程是否正确。换句话说，它管的是“怎么安全地把模型送进生产”，不是“怎么把模型训出来”。

一个最简单的边界场景是：推荐模型从 `v17` 升级到 `v18`。如果只是把服务器上的 `model.bin` 覆盖掉，虽然表面上完成了升级，但你会立刻失去三类能力：

1. 无法准确记录谁上线了什么版本。
2. 无法按流量比例分阶段验证。
3. 出问题时无法快速、确定地切回旧版本。

版本管理要求 `v17` 仍然可服务，`v18` 先灰度，失败后立即回退到 `v17`。这里“灰度”可以先理解为：不是所有用户一起用新版本，而是只有一小部分流量先用。

手工替换与版本管理的差别很直接：

| 事情 | 手工替换文件 | 版本管理与回滚 |
|---|---|---|
| 是否有版本号 | 否 | 是 |
| 是否可灰度 | 否 | 是 |
| 是否可审计 | 弱 | 强 |
| 是否可快速回退 | 弱 | 强 |
| 是否能限制风险 | 弱 | 强 |

本文边界也要说清楚。这里只讨论推理模型的生产发布与回滚，不展开训练实验管理、特征版本管理、数据版本管理。后面会提到它们之间的关系，但不把重点放在训练平台设计上。

适用边界如下：

- 适用：在线推理、A/B 发布、canary 灰度、自动回滚
- 不展开：训练过程 checkpoint 管理、实验平台对比、数据集版本治理

这里顺手补一句容易混淆的关系：训练 checkpoint 是训练过程里的中间保存点，解决“训练能否续跑”；模型版本管理是生产治理问题，解决“上线是否可控”。两者相关，但不是同一层。

---

## 核心机制与推导

要判断一个候选版本能不能放量，不能只看一个健康检查。健康检查只说明“服务是不是活着”，不说明“结果是不是还对”“资源是不是会爆”。因此，最少要拆成三个门槛：

1. 健康状态 `h`
2. 输出差异 `d`
3. 资源上限 `m <= M`

判定规则可以写成：

$$
promote(v_c) = 1 \iff h = 1 \ 且 \ d \le \delta \ 且 \ m \le M
$$

$$
rollback(v_s) = 1 \iff h = 0 \ 或 \ d > \delta \ 或 \ m > M
$$

其中，$\delta$ 是你预先设定的差异门槛。它的直白含义是：新旧版本可以有差别，但这个差别不能超过业务允许范围。

输出差异可抽象为：

$$
d = \frac{1}{n}\sum_{i=1}^{n}\Delta(f_{v_c}(x_i), f_{v_s}(x_i))
$$

这里的 `Δ` 是“两个输出之间怎么比较”的函数，白话解释就是：不同任务，比较方式不同。

| 任务类型 | `Δ` 的含义 |
|---|---|
| 分类 | 标签不一致率 |
| 回归 | 归一化误差 |
| 生成 | 人工或自动评测差 |
| 检索/排序 | Top-K 差异、NDCG 变化、命中率变化 |

为什么一定要分阶段控制流量比例 `p`？原因很简单：`p` 越小，问题暴露得越早，影响范围越可控。假设总流量是 10,000 req/min：

- 当 `p = 0.05` 时，新模型只接 500 req/min
- 当 `p = 0.50` 时，新模型接 5,000 req/min
- 当 `p = 1.00` 时，新模型接全部 10,000 req/min

如果 `v_c` 的错误率突然升高，或者显存从 10 GiB 增到 14 GiB，而机器上限 `M = 12 GiB`，那么在 5% 灰度阶段发现问题，和在 100% 全量阶段发现问题，后果完全不是一个量级。

一个玩具例子可以直接算。设稳定版本 `v_s` 在一组分类样本上输出为 `[1, 0, 1, 1, 0]`，候选版本 `v_c` 输出为 `[1, 1, 1, 0, 0]``。如果把 `Δ` 定义为“标签是否一致”，那么不一致的位置有 2 个，所以：

$$
d = \frac{2}{5} = 0.4
$$

如果你的门槛是 `δ = 0.1`，那么这个候选版本即使接口返回正常，也不能放量，因为输出偏差过大。

一个真实工程例子更有代表性。推荐系统把 CTR 预估模型从 `v17` 升级到 `v18`。`v18` 在线功能完全正常，健康检查也通过，但因为训练样本窗口改了，导致长尾商品得分系统性偏低，线上点击率静默下降 2%。这种问题不会在“接口通不通”里暴露，只会在输出差异对比、业务指标监控和小流量灰度里被发现。所以“健康通过 ≠ 可放量”是生产系统里必须反复强调的原则。

---

## 代码实现

代码层面至少要体现五个动作：模型注册、别名切换、灰度路由、健康检查、自动回滚。只写一个 `load_model()` 不叫版本管理，那只是把模型读进内存。

下面先给一个可运行的玩具实现。它不依赖外部库，但把核心控制流程都保留下来了。

```python
from dataclasses import dataclass, field

@dataclass
class ModelVersion:
    name: str
    version: str
    artifact_path: str
    healthy: int = 1
    diff: float = 0.0
    mem_gib: float = 0.0

@dataclass
class Registry:
    versions: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)

    def register(self, name: str, version: str, artifact_path: str, **metrics):
        key = (name, version)
        self.versions[key] = ModelVersion(name, version, artifact_path, **metrics)

    def alias(self, alias_name: str, version: str):
        self.aliases[alias_name] = version

    def resolve(self, name: str, alias_name: str) -> ModelVersion:
        version = self.aliases[alias_name]
        return self.versions[(name, version)]

def route_request(user_id: str, p: float = 0.05) -> str:
    bucket = sum(ord(c) for c in user_id) % 100
    if bucket < int(p * 100):
        return "candidate"
    return "stable"

def should_promote(health: int, diff: float, delta: float, mem: float, M: float) -> bool:
    return health == 1 and diff <= delta and mem <= M

def should_rollback(health: int, diff: float, delta: float, mem: float, M: float) -> bool:
    return health == 0 or diff > delta or mem > M

registry = Registry()
registry.register(name="reco-model", version="v17", artifact_path="s3://models/reco/v17", healthy=1, diff=0.0, mem_gib=9.5)
registry.register(name="reco-model", version="v18", artifact_path="s3://models/reco/v18", healthy=1, diff=0.03, mem_gib=10.8)

registry.alias("production", "v17")
registry.alias("candidate", "v18")
registry.alias("stable", "v17")

stable = registry.resolve("reco-model", "stable")
candidate = registry.resolve("reco-model", "candidate")

delta = 0.05
M = 12.0

assert route_request("user_a", p=0.05) in {"stable", "candidate"}
assert should_promote(candidate.healthy, candidate.diff, delta, candidate.mem_gib, M) is True
assert should_rollback(candidate.healthy, candidate.diff, delta, candidate.mem_gib, M) is False

# 模拟候选版本显存超限
candidate.mem_gib = 14.0
assert should_promote(candidate.healthy, candidate.diff, delta, candidate.mem_gib, M) is False
assert should_rollback(candidate.healthy, candidate.diff, delta, candidate.mem_gib, M) is True

if should_rollback(candidate.healthy, candidate.diff, delta, candidate.mem_gib, M):
    registry.alias("production", "v17")

assert registry.aliases["production"] == "v17"
```

上面这段代码对应的工程语义很明确：

```python
registry.register(name="reco-model", version="v18", artifact_path="s3://...")
registry.alias("production", version="v17")
registry.alias("candidate", version="v18")
```

“注册”解决的是产物记录问题，“alias”解决的是可切换问题。`production` 不是模型文件本身，而是一个稳定引用。这样回滚时切的是引用，不是现场改文件。

灰度路由的最小逻辑可以写成：

```python
def route_request(user_id, p=0.05):
    if hash(user_id) % 100 < p * 100:
        return "candidate"
    return "stable"
```

这里用 `user_id` 做稳定分桶，是为了让同一个用户尽量落到同一版本，避免今天看到 `v17`、下一秒又落到 `v18`，造成体验抖动。

回滚条件则必须被明确编码：

```python
if health == 0 or diff > delta or mem > M:
    switch_alias("production", "v17")
```

完整的真实工程闭环通常是：

1. 注册 `v17` / `v18`
2. `production -> v17`，`candidate -> v18`
3. 5% 灰度到 `v18`
4. 监控 `5xx`、`p95 latency`、业务指标、OOM
5. 达标则逐步放量，不达标则回切

相关组件职责可以拆开看：

| 组件 | 作用 |
|---|---|
| 模型注册表 | 记录版本、标签、注释、产物位置 |
| Alias | 指向当前生产版本 |
| 路由层 | 控制灰度流量比例 |
| Readiness Probe | 控制是否允许接流量 |
| Monitoring | 提供延迟、错误率、业务指标 |
| Rollback 逻辑 | 失败时切回稳定版本 |

真实工程例子里，推荐系统或风控系统常见做法是：模型产物进入注册表，Serving 层通过 alias 拉取指定版本，Kubernetes readiness probe 只在模型加载成功后放流量，Prometheus 监控错误率和延迟，离线或影子流量对比模块计算 `d`，一旦 `h = 0` 或 `d > δ` 或 `m > M`，自动把 `production` alias 指回上一个稳定版本。

---

## 工程权衡与常见坑

版本管理不是越复杂越好。核心权衡是“控制粒度”和“操作成本”之间的平衡。灰度越细、门禁越多、回滚越自动化，风险越低，但系统也越复杂，维护成本越高。

对中小团队，最危险的往往不是“能力不够高级”，而是根本没有最小治理闭环。比如没有版本号、没有 alias、没有统一回滚入口、没有上线记录。这样一旦故障出现，排障时间会比修 bug 本身更长。

常见坑如下：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 只替换文件，不记录版本 | 不知道线上到底是谁 | 必须有版本号、alias、审计记录 |
| 只看健康检查，不看输出一致性 | 静默精度退化上线 | 增加输出差异门槛 `d` |
| 灰度一次放太大 | 风险瞬间放大 | 分阶段放量，设置暂停点 |
| 只验证功能，不测容量 | OOM、吞吐下降、冷启动失败 | 发布前做压测和资源校验 |
| 没有回滚演练 | 真出问题时回不去 | 定期演练 rollback |

一个典型事故是：`v18` 的健康检查是通过的，接口也能返回结果，但输出和 `v17` 相比静默退化了 2%。如果团队只看 readiness，就会误以为“服务没挂，可以全量”；如果同时比较输出差异 `d`，就会在灰度阶段发现问题并回滚。

所以要反复强调这句约束：

$$
健康通过 \ne 可放量
$$

必须满足：

$$
h = 1 \ 且 \ d \le \delta \ 且 \ m \le M
$$

还有一个常被忽略的坑是资源错配。很多模型在离线评测时指标很好，但线上批大小、并发数、KV cache 或 tokenizer 行为不同，显存曲线会变。结果是单请求能跑，多请求就 OOM。因此 `m` 不是部署前填一个静态值就够了，而是要在接近真实并发下测出来。

---

## 替代方案与适用边界

不是所有团队都必须上一整套复杂平台。方案应该按场景选，不是按“技术先进程度”选。

如果团队刚起步，最小可行方案是模型注册表 + alias 切换 + 手工监控回滚。它已经能解决“版本号、可追踪、可回切”这三个最基础问题。等到发布频率变高、业务风险变高，再逐步引入自动分析与自动回滚。

常见方案对比如下：

| 方案 | 优点 | 适用场景 | 局限 |
|---|---|---|---|
| 模型注册表 + alias | 简单、可审计 | 中小团队、低复杂度场景 | 自动化较弱 |
| KServe Canary | 天然灰度流量控制 | 模型服务在 K8s 上 | 依赖 KServe 体系 |
| Argo Rollouts + Analysis | 指标驱动自动推进/回滚 | 指标体系成熟、发布频繁 | 配置复杂度更高 |
| Kubernetes Deployment rollback | 原生、通用 | 容器化服务 | 对模型语义支持弱 |

适用边界可以直接记成下面几条：

- 低风险业务：优先简化流程，重点放在审计和可回滚
- 高风险业务：优先自动化门禁和指标分析
- 强资源约束场景：必须加入显存、吞吐、冷启动约束
- 强合规场景：必须保留历史 revision 和回滚记录

还要说明一层现实边界：`kubectl rollout undo` 能回滚容器版本，但它只理解 Deployment revision，不天然理解“模型输出差异是否可接受”。所以它适合当基础设施层回滚手段，但不能替代模型语义层的验证。

---

## 参考资料

下面这张表对应文章里的主要机制点：

| 资料 | 支撑内容 |
|---|---|
| MLflow Model Registry | 版本号、alias、tag、注释、审计 |
| KServe Canary Rollout Strategy | `canaryTrafficPercent`、last good revision、失败回滚 |
| Kubernetes Probes | readiness / liveness / startup 的职责 |
| Kubernetes Deployment / Rollout | revision 历史、`kubectl rollout undo/status` |
| Argo Rollouts Canary + Analysis | 指标驱动自动推进或回滚 |

1. [MLflow Model Registry](https://www.mlflow.org/docs/3.0.1/model-registry)
2. [KServe Canary Rollout Strategy](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary)
3. [Kubernetes Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
4. [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
5. [kubectl rollout status](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_rollout/kubectl_rollout_status/)
6. [kubectl rollout undo](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_rollout/kubectl_rollout_undo/)
7. [Argo Rollouts Canary Strategy](https://argo-rollouts.readthedocs.io/en/stable/features/canary/)
8. [Argo Rollouts Analysis](https://argo-rollouts.readthedocs.io/en/stable/features/analysis/)
