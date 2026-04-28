## 核心结论

模型热更新的核心不是“把旧模型替换成新模型”，而是“让新旧模型短时间并存，再安全切换流量”。这里的“热更新”可以先用一句白话理解：服务不停、连接不断、请求不中断，但后台已经把模型版本换掉了。

在线推理服务里，真正难的不是把权重文件读进来，而是同时满足三个条件：

$$
ready_n = loaded_n \land warmup_n \land probe_n
$$

其中，`loaded_n` 表示新模型已经加载到内存或显存；`warmup_n` 表示新模型已经跑过预热请求；`probe_n` 表示健康检查通过。只有这三者同时成立，新模型才应该接流量。

流量切换通常不是一步到位，而是灰度切流。灰度的白话解释是：先给少量请求试运行，再逐步扩大范围。它可以写成：

$$
r_{new} = \alpha,\quad r_{old} = 1-\alpha,\quad \alpha: 0 \rightarrow 1
$$

也就是新模型接收 $\alpha$ 比例的流量，旧模型接收剩余流量，$\alpha$ 从 0 慢慢增加到 1。

一个最容易被忽略的结论是：热更新的主要风险不是“逻辑写错”，而是“资源峰值失控”。因为切换期间新旧模型会同时存在，峰值资源通常近似为：

$$
M_{peak} \approx M_{old} + M_{new} + M_{ws} + M_{cache}
$$

`M_ws` 是推理时的临时工作区，白话讲就是算子执行时额外要占的那部分内存；`M_cache` 是缓存和运行时附加开销。很多线上故障不是新模型精度差，而是切换那一刻显存双占，直接触发 OOM。

| 状态 | 含义 | 是否可接流量 |
|---|---|---|
| `loaded_n` | 新权重已载入 | 否 |
| `warmup_n` | 预热完成 | 否 |
| `probe_n` | readiness/health 通过 | 条件之一 |
| `ready_n` | 三者同时满足 | 是 |

玩具例子可以这样看：一个小型文本分类服务当前跑着 `v1`，新版本 `v2` 要上线。正确做法不是把 `v1` 从内存里删掉，再把 `v2` 塞进去，而是让 `v2` 先在备用槽位里完成加载，跑几条典型样本，确认接口、延迟、输出形状都正常，再先给它 5% 请求。如果出错，流量立刻回退到 `v1`，用户不会感觉到服务“掉了一次”。

---

## 问题定义与边界

模型热更新解决的是“在线推理服务如何在不停服条件下切换模型版本”这个问题。这里的“在线推理”是指用户请求到来时，系统立即返回预测结果，比如搜索排序、推荐、OCR、问答接口。它不讨论训练过程，也不讨论离线批处理任务如何换模型。

这个问题有三个典型约束。

第一，不能切断现有连接。用户已经发出的请求不能因为模型升级而直接失败。

第二，不能把风险一次性扩散到全量流量。即使新模型有问题，也要先影响少量请求，便于快速回滚。

第三，不能把“进程活着”误判为“模型可服务”。进程没崩，只说明服务容器还在；模型是否真正能推理，要看权重、预热、依赖资源和健康状态。

因此，下面这些事情不属于本文所说的热更新：

| 场景 | 是否适合热更新 | 原因 |
|---|---|---|
| 在线推荐服务 | 适合 | 必须不停服，且流量可分批切换 |
| 批处理离线任务 | 一般不需要 | 任务批次之间可直接换版本 |
| 训练中的参数同步 | 不属于 | 这是训练系统问题，不是线上发布问题 |
| 单次脚本推理 | 通常不需要 | 直接重启进程成本更低 |
| 大模型结构频繁变化 | 视情况而定 | 资源峰值和兼容成本可能过高 |

这里要特别区分两个常见误解。

一类误解是把热更新等同于“进程内原地替换权重”。这通常不安全。因为原模型还在处理已有请求时，新模型又要分配内存，最容易出现短时双占和状态不一致。

另一类误解是把热更新等同于“模型在线学习”。在线学习是边接收数据边更新参数，目标是让模型持续适应数据分布；热更新是把一个已经训练完成的新版本替换旧版本，目标是稳定发布。

玩具例子：一个电商商品分类接口平均每秒 30 个请求，旧模型是 `v1`。运营要求中午更新为 `v2`，但页面不能出现“维护中”。这时就适合热更新。反过来，如果是每天夜里跑一次离线特征打分任务，中间允许任务切批，通常直接切作业版本就够了，没有必要引入灰度和双缓冲复杂度。

真实工程例子：Kubernetes 上的推荐服务通常是多副本、长时间运行、SLA 明确的系统。SLA 可以白话理解为服务可用性承诺，例如 99.9% 可用。对于这类服务，简单重启实例会造成请求抖动和容量波动，因此更常见的做法是拉起新 revision，做 readiness 检查，再逐步分流。

---

## 核心机制与推导

热更新可以抽象成一个很简单的状态机：

```text
load new model -> warmup -> readiness pass -> canary 5% -> 50% -> 100% -> unload old model
```

这条链路里每一步都不能省。

### 1. 加载不是上线

加载的含义是：模型文件、权重、依赖图、算子库、Tokenizer 或前后处理资源已经放到可运行环境中。它解决的是“能否创建出一个模型实例”，不解决“这个实例能否稳定接流量”。

很多初学者在这里会犯错：看到 `load_model()` 返回成功，就认为发布完成。实际上这时模型可能还没有分配完全部显存，也没有建立运行时缓存，更没有跑过典型请求。

### 2. 预热是在为第一批真实请求消除冷启动

“预热”可以白话理解成正式营业前先试跑几单。模型服务第一次执行时，常常会触发图优化、Kernel 选择、显存页分配、JIT 编译或缓存填充。如果把这些开销留给第一批线上用户，请求延迟会出现尖峰。

所以 `warmup_n` 的作用，是把这些一次性成本提前支付掉。预热样本最好来自真实流量分布，而不是随便造一个空输入。否则你虽然“预热过”，但没有命中真正重的路径，真实请求一来还是抖。

### 3. 健康检查是在定义“什么叫可接流量”

`probe_n` 通常由 readiness probe 或等价机制给出。readiness 的白话解释是“这个实例现在适不适合分配请求”。它不关心你是不是活着，而关心你能不能稳定服务。

所以 readiness 最好检查这些内容：

| 检查项 | 作用 |
|---|---|
| 权重已加载 | 避免空模型接流量 |
| 预热完成 | 避免首批请求抖动 |
| 关键依赖可用 | 避免 tokenizer、特征服务、缓存未就绪 |
| 基础推理成功 | 避免加载成功但实际推理失败 |

### 4. 灰度切流是在把风险做成可控变量

设新模型流量比例为 $\alpha$，那么旧模型比例自然就是 $1-\alpha$。这不是数学上的装饰，而是一个非常实用的发布控制杆。你可以把发布过程理解为不断调整 $\alpha$ 的过程。

当 $\alpha = 0.05$ 时，新模型只接 5% 请求，风险暴露有限；当 $\alpha = 0.5$ 时，新旧模型几乎平分流量，能更快观察总体行为；当 $\alpha = 1$ 时，说明旧版本已经可以准备下线。

### 5. 资源峰值决定方案能不能落地

双缓冲的本质是“旧版本还没下，新版本已经上来”。这带来的直接代价是峰值资源上涨：

$$
M_{peak} \approx M_{old} + M_{new} + M_{ws} + M_{cache}
$$

做一个玩具计算。假设一张 GPU 有 48 GiB 显存，旧模型占 18 GiB，新模型占 20 GiB，运行时工作区占 5 GiB，额外缓存占 2 GiB，那么：

$$
M_{peak} \approx 18 + 20 + 5 + 2 = 45\ \text{GiB}
$$

45 GiB 小于 48 GiB，这次热更新理论上可行。若新模型变成 24 GiB，则峰值升到 49 GiB，已经超过显存上限，发布时就可能 OOM。这里的 OOM 是 out of memory，也就是内存或显存不足导致分配失败。

真实工程里还要更保守，因为碎片化、框架内部缓存、batch 波动都会让实际峰值高于纸面计算值。经验上，如果理论峰值已经贴近上限，就不该继续赌热更新，而应改成双集群、蓝绿或离线窗口升级。

---

## 代码实现

下面给一个最小可理解的实现骨架。它不依赖具体框架，重点是把“活动版本”“备用版本”“预热状态”“切流比例”显式建模出来。

```python
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ModelSlot:
    name: str
    loaded: bool = False
    warmed: bool = False
    healthy: bool = False

    @property
    def ready(self) -> bool:
        return self.loaded and self.warmed and self.healthy

    def predict(self, x: int) -> int:
        if not self.ready:
            raise RuntimeError(f"{self.name} is not ready")
        return x * 2


class ModelManager:
    def __init__(self, active_model: ModelSlot):
        assert active_model.ready
        self.active_model = active_model
        self.standby_model: Optional[ModelSlot] = None
        self.route_ratio = 0.0

    def load_standby(self, slot: ModelSlot, warmup_fn: Callable[[ModelSlot], None],
                     readiness_fn: Callable[[ModelSlot], bool]) -> None:
        slot.loaded = True
        warmup_fn(slot)
        slot.healthy = readiness_fn(slot)
        assert slot.ready, "standby model must be ready before receiving traffic"
        self.standby_model = slot

    def choose_model(self, request_id: int) -> ModelSlot:
        if self.standby_model and self.standby_model.ready:
            bucket = request_id % 100
            if bucket < int(self.route_ratio * 100):
                return self.standby_model
        return self.active_model

    def switch_traffic(self, ratio: float) -> None:
        assert 0.0 <= ratio <= 1.0
        if ratio > 0:
            assert self.standby_model and self.standby_model.ready
        self.route_ratio = ratio

    def promote(self) -> None:
        assert self.route_ratio == 1.0
        assert self.standby_model and self.standby_model.ready
        self.active_model = self.standby_model
        self.standby_model = None
        self.route_ratio = 0.0


def warmup(slot: ModelSlot) -> None:
    # 用一次试跑模拟缓存建立和算子预热
    _ = 1 + 1
    slot.warmed = True


def readiness_check(slot: ModelSlot) -> bool:
    try:
        # 真实系统里这里会检查模型推理、依赖资源和关键路径
        return slot.loaded and slot.warmed
    except Exception:
        return False


active = ModelSlot(name="v1", loaded=True, warmed=True, healthy=True)
manager = ModelManager(active)

standby = ModelSlot(name="v2")
manager.load_standby(standby, warmup, readiness_check)
manager.switch_traffic(0.05)

chosen = manager.choose_model(request_id=3)
assert chosen.name == "v2"   # 3 % 100 < 5

chosen = manager.choose_model(request_id=77)
assert chosen.name == "v1"   # 77 % 100 >= 5

manager.switch_traffic(1.0)
manager.promote()
assert manager.active_model.name == "v2"
assert manager.standby_model is None
```

这段代码刻意保留了几个工程上必须明确的点。

第一，`ready` 不是单个布尔变量随手写死，而是由 `loaded && warmed && healthy` 推导出来。这样你不会因为少做一步就误放流量。

第二，`route_ratio` 是显式控制项，不是“准备好了就全量切”。发布系统需要把切流能力做成独立开关。

第三，`promote()` 只在 `ratio == 1.0` 时执行，这意味着发布和提拔主版本是两个不同动作。这样失败回滚更清楚。

如果要贴近 Kubernetes 或 KServe，通常会再加上探针配置。下面是简化示意：

```yaml
readinessProbe:
  exec:
    command: ["sh", "-c", "check_model_ready"]
livenessProbe:
  exec:
    command: ["sh", "-c", "check_process_alive"]
```

这里 `liveness` 的白话解释是“进程是不是卡死或失活”，它主要决定要不要重启容器；`readiness` 决定这个实例是否接收流量。两者不能混用。

真实工程例子：推荐服务的 `v2` revision 启动后，先从对象存储拉权重，完成 GPU 映射；随后跑一批真实 query 特征做 warmup；接着通过 readiness；网关把 5% 用户请求打到 `v2`；监控系统观察 10 分钟的 p95 延迟、错误率和显存水位；若稳定再把比例提到 50%，最终到 100%，最后下线 `v1`。

实现步骤可以概括为：

| 步骤 | 动作 | 目的 |
|---|---|---|
| 1 | 拉起新 revision | 准备新模型实例 |
| 2 | 加载权重和依赖 | 建立可运行环境 |
| 3 | 执行 warmup | 降低冷启动抖动 |
| 4 | readiness 通过 | 确认可接流量 |
| 5 | 灰度切流 | 控制风险扩散 |
| 6 | 观察指标 | 验证稳定性 |
| 7 | 下线旧版本 | 回收资源 |

---

## 工程权衡与常见坑

热更新的优点很明确，但它绝不是“免费午餐”。你要用更多资源和更复杂的控制逻辑，去换取不停服和可回滚。

最核心的权衡是资源双占。旧模型还在服务，新模型又要加载和预热，显存、内存、文件句柄、缓存都可能短时间翻倍。对于 CPU 模型，问题通常是内存和缓存；对于 GPU 模型，问题最尖锐的是显存和 workspace。

常见坑可以直接列成对照表：

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 直接进程内替换模型 | 显存双占、短时 OOM、请求中断 | 双缓冲，先新后旧 |
| warmup 数据不真实 | 首批请求延迟抖动大 | 用真实分布采样做预热 |
| readiness 只看进程存活 | 模型未就绪就放流 | 将加载与预热纳入 readiness |
| liveness 和 readiness 混用 | 不必要重启，甚至重启风暴 | 分开职责，分别检查 |
| 切流过快 | 异常迅速扩散到全量 | 分阶段灰度并设回滚阈值 |
| 只看平均延迟 | 长尾问题被掩盖 | 重点观察 p95/p99、错误率、超时率 |
| 忽略旧会话粘性 | 会话状态错乱 | 为长会话保留版本粘性 |
| 先删旧版本再验新版本 | 无法快速回滚 | 旧版本留到新版本稳定后再下线 |

这里有一个非常实际的坑：有些服务用了 KV cache、embedding cache 或编译缓存，新模型上线时这些缓存并不兼容。结果是服务看起来“ready”，但第一波真实请求命中了缓存缺失或格式不一致，延迟飙升。解决办法通常是把缓存兼容性也纳入发布检查，或者对新版本独立建缓存命名空间。

另一个常见坑是把灰度做成随机切流，却忘了会话粘性。会话粘性可以白话理解为“同一个用户或同一段会话尽量固定落到同一版本”。如果一个多轮对话请求一会儿落到 `v1`，一会儿落到 `v2`，就可能出现上下文格式不一致、缓存错配或者回复风格突变。

真实工程里，回滚条件也必须前置定义，而不是异常出现后临时拍脑袋。比如可以规定：

- 新版本 5% 灰度阶段，5 分钟内错误率上升超过 0.5 个百分点则回滚。
- p95 延迟较旧版本上升超过 20% 且持续 3 个采样窗口则暂停升量。
- 显存使用率超过 90% 且伴随分配失败日志则立即回滚。

没有这些阈值，灰度就只是“慢一点全量”，不是受控发布。

---

## 替代方案与适用边界

热更新不是唯一方案。判断标准很简单：如果双缓冲的峰值成本或控制复杂度已经超过收益，就不该硬上。

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 热更新 | 不停服、可灰度、可快速回滚 | 双占资源、实现复杂 | 资源足够，且要求连续在线 |
| 蓝绿发布 | 环境隔离强，回滚快 | 成本高，要维护两套环境 | 预算充足、追求强隔离 |
| 滚动更新 | 平台支持好，实现简单 | 单实例或大状态服务不友好 | 多副本无状态服务 |
| 离线窗口升级 | 操作简单、最稳妥 | 有停机时间 | 可接受短暂停服 |
| 双集群切流 | 风险隔离最好 | 基础设施成本高 | 核心业务、大模型重发布 |

什么时候适合热更新？

第一，业务要求持续在线，不能明显停服。

第二，流量可以逐步切换，且有足够监控能力观察新版本。

第三，资源预算允许短时间双占。

什么时候不适合热更新？

第一，模型太大，按
$$
M_{old} + M_{new} + M_{ws} + M_{cache}
$$
算下来已经超出可用显存或接近上限。

第二，新旧模型的输入输出协议变化很大，导致同一条调用链要同时兼容两套逻辑，系统复杂度陡增。

第三，系统根本没有成熟的探针、监控、回滚机制。没有这些基础能力，热更新只是把风险从停机风险换成线上事故风险。

玩具例子：如果一台机器上旧模型 22 GiB，新模型 24 GiB，workspace 4 GiB，总峰值已经 50 GiB，而机器只有 48 GiB 显存，那就不应该再讨论“怎么优化热更新脚本”，而应该直接换策略，比如蓝绿发布或双集群切流。

真实工程例子：超大语言模型推理服务常常因为权重体积、KV cache 和张量并行限制，不适合在单机单卡上做进程内热更新。更常见的做法是提前拉起一整组新实例，完成预热后通过网关、Service Mesh 或 DNS 做分流，再逐步摘除旧组实例。这本质上更接近蓝绿或双集群，而不是单进程热替换。

一句话判断边界：如果峰值资源、接口兼容性和回滚能力三者里有一项明显不满足，优先考虑替代方案，而不是勉强做热更新。

---

## 参考资料

1. [TensorFlow SavedModel Warmup](https://www.tensorflow.org/tfx/serving/saved_model_warmup)
2. [TensorFlow Serving Configuration](https://www.tensorflow.org/tfx/serving/serving_config)
3. [TensorFlow Serving Advanced Configuration and Manager](https://www.tensorflow.org/tfx/serving/serving_advanced)
4. [Kubernetes Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
5. [KServe Canary Rollout Strategy](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary)
