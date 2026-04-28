## 核心结论

边缘模型更新策略，指的是把新模型发布到手机、摄像头、门店盒子、车载设备这类边缘节点时，用一套可控流程完成“下发、观测、切换、回滚”，而不是把文件传过去就算完成。

它成立的原因很直接：边缘环境不是统一机房，而是大量在线状态不稳定、硬件能力不同、网络质量不同的分散设备。云端服务可以直接替换线上实例，边缘设备如果全量同时升级，一旦新模型触发崩溃、延迟升高或内存打爆，问题会同步扩散，而且回收成本高。真正需要优化的对象不是“更新速度”本身，而是在资源受限条件下，把失败影响面控制在最小范围内。

一个最小判断原则是：新版本只有在核心指标稳定时，才允许继续放量。所谓放量，就是把新版本从少量设备逐步扩大到更多设备。比如 5% 设备试跑后，若延迟下降但崩溃率明显上升，结论仍然是不能扩量，因为稳定性优先于局部指标优化。

下表可以先把“云端全量发布”和“边缘分批更新”的差异框住：

| 维度 | 云端全量发布 | 边缘分批更新 |
|---|---|---|
| 发布对象 | 少量集中式服务实例 | 大量分散设备 |
| 网络条件 | 机房内稳定 | 经常断连、弱网 |
| 硬件环境 | 相对统一 | 高度异构 |
| 失败影响 | 可快速替换实例 | 可能长期滞留在故障版本 |
| 回滚方式 | 重新切流即可 | 需要远程回切并等设备生效 |
| 推荐策略 | 自动扩容+集中切换 | 灰度、分层、可回滚发布 |

玩具例子：有 100 台门禁设备，新模型识别更快，但只在 8GB 内存设备上稳定。如果你把它直接发到所有设备，2GB 内存设备可能频繁 OOM。这里 OOM 指“内存不够导致进程被系统杀掉”。正确做法不是“统一升级”，而是先筛选设备能力，再分批验证。

真实工程例子：零售门店摄像头做本地目标检测。云端训练出新模型后，不能直接给全国门店一起上线。更稳的流程是：先给网络稳定、硬件高配、业务风险低的门店推送；观测误报率、漏报率、推理延迟、进程崩溃率；指标达标后再扩大范围。否则，哪怕模型离线精度更高，线上业务仍可能更差。

---

## 问题定义与边界

本文讨论的“更新策略”，不是如何训练出更高精度模型，而是如何让一个新模型安全进入线上边缘系统。它解决的是发布控制问题，不是训练优化问题。

可以把这个问题抽象成一个输入输出关系：

| 项目 | 内容 |
|---|---|
| 输入 | 新模型版本、设备列表、当前运行版本、设备资源约束、监控指标、回滚条件 |
| 输出 | 分批更新计划、设备准入结果、切换决策、回滚决策 |

这里的“准入”是术语，意思是“某一批设备是否有资格进入下一步发布”。它本质上是一个门槛判断，不是概率猜测。

更新策略通常同时追求两个目标：

| 目标 | 含义 | 常见冲突 |
|---|---|---|
| 服务稳定性 | 不让线上崩、不卡、不过度误报 | 越保守，模型上线越慢 |
| 模型新鲜度 | 尽快让设备使用新模型 | 越激进，线上风险越大 |

这说明它不是单目标优化。若只追求新鲜度，就会倾向快速全量；若只追求稳定性，就可能长期停在旧模型，业务收益无法落地。工程设计必须在两者之间做权衡。

适用边界如下：

| 适用场景 | 说明 |
|---|---|
| 边缘部署 | 模型运行在端侧而不是集中式云端 |
| 灰度发布 | 需要先小范围验证再扩大 |
| 异步联邦更新 | 设备回传更新时间不一致 |
| 设备分层策略 | 不同硬件跑不同模型或不同节奏 |

不适用边界如下：

| 不适用场景 | 原因 |
|---|---|
| 纯离线训练 | 没有线上发布与回滚问题 |
| 单机模型调参 | 不涉及大规模设备管理 |
| 无回滚要求的批处理任务 | 失败影响面和恢复方式完全不同 |

这里还要明确一个常见误区：模型更新策略不保证“新模型一定更准”。它只保证“即使新模型有问题，系统也能尽量小范围试错并快速撤回”。这就是它和模型训练、模型评测的职责边界。

玩具例子：你在自己电脑上替换一个本地分类模型，只要程序能跑就行，这不属于本文重点。因为它没有大规模设备、异构硬件和远程回滚要求。

真实工程例子：同样是摄像头检测，A 门店是高配 NPU 设备，B 门店是老旧 ARM CPU 设备。即使同一个模型在离线验证集都通过了，上线策略也不能相同，因为硬件承载能力不同，风险模型不同。

---

## 核心机制与推导

边缘模型更新可以抽象为五步：放量、准入、观测、切换、回滚。

1. 放量：先只让少量设备获得新版本。
2. 准入：只把符合条件的设备纳入这一轮。
3. 观测：持续看延迟、错误率、崩溃率、资源占用。
4. 切换：确认指标达标后，让请求真正使用新模型。
5. 回滚：一旦越过阈值，停止扩量并回切旧版本。

这个过程之所以需要拆开，是因为“模型文件到设备”与“请求真正使用它”是两个不同事件。前者是分发，后者是生效。把两者混成一步，会让错误没有缓冲区。

### 1. 灰度放量

设总设备数为 $N$，第 $t$ 轮放量比例为 $p_t$，则本轮计划更新设备数为：

$$
B_t = p_t \cdot N
$$

其中：

| 符号 | 含义 |
|---|---|
| $N$ | 总设备数 |
| $p_t$ | 第 $t$ 轮放量比例 |
| $B_t$ | 第 $t$ 轮更新设备数 |

如果 $N=1000$，$p_t=0.05$，那么 $B_t=50$，即先更新 50 台。这个做法的本质是控制风险暴露面。风险暴露面，白话讲，就是“出事时会被波及的设备数量”。

### 2. 准入条件

仅仅把新模型发到设备上不够，还要判断这一批设备是否允许继续扩量。可以写成一个布尔条件：

$$
M_t = [L_t \le L_{max},\ E_t \le E_{max},\ C_t \le C_{max}]
$$

这里的布尔条件意思是“每个约束都要同时成立”。其中：

| 符号 | 含义 |
|---|---|
| $L_t$ | 当前批次延迟 |
| $E_t$ | 当前批次错误率 |
| $C_t$ | 当前批次崩溃率 |
| $L_{max}$ | 最大允许延迟 |
| $E_{max}$ | 最大允许错误率 |
| $C_{max}$ | 最大允许崩溃率 |

最小数值例子：1000 台设备中先放 50 台。新模型把 p95 延迟从 80ms 降到 76ms，但崩溃率从 0.1% 升到 0.8%。若阈值是 0.5%，则虽然延迟更好，结论仍然是禁止扩量并回滚。因为上线决策不是只看单一收益，而是看约束是否全部满足。

这里的 p95 指“95% 请求都不超过这个延迟”，白话就是“尾部慢请求的大致水平”。

### 3. 异步更新的陈旧度处理

如果边缘设备不仅做推理，还会本地训练或本地微调，就会出现“回传更新并不同步”的问题。某些设备晚了很多轮才上传旧参数，如果不处理，旧信息会污染新模型。

可用一个陈旧度权重：

$$
w_i = e^{-\lambda \tau_i}
$$

其中：

| 符号 | 含义 |
|---|---|
| $\tau_i$ | 第 $i$ 个更新的陈旧度 |
| $\lambda$ | 衰减系数，控制旧更新降权速度 |
| $w_i$ | 第 $i$ 个更新的权重 |

再做加权聚合：

$$
\theta_{t+1} = \frac{\sum_i w_i \theta_i}{\sum_i w_i}
$$

$\theta_i$ 表示设备回传的模型参数或增量，白话讲就是“每台设备带回来的更新结果”。

若 $\tau=[1,4,10]$，$\lambda=0.5$，则权重大致为：

$$
[e^{-0.5}, e^{-2}, e^{-5}] \approx [0.61, 0.14, 0.007]
$$

可以看到最旧的更新几乎不该主导合并结果。这不是说旧更新毫无价值，而是它的参考权应明显更低。

### 4. 控制流程

用步骤表表示更直观：

| 步骤 | 动作 | 目标 |
|---|---|---|
| 1 | 选择灰度设备 | 控制首批风险 |
| 2 | 下发新模型 | 让候选设备具备新版本 |
| 3 | 观测指标 | 判断是否稳定 |
| 4 | 满足阈值则扩量 | 逐步扩大收益面 |
| 5 | 不满足则回滚 | 快速止损 |

玩具例子：100 台传感器设备中，先选 5 台网络稳定设备安装新模型，只做影子推理。影子推理就是“算出结果但不拿来驱动真实业务”。如果结果稳定，再让这 5 台切到真实请求，再扩大到 20 台、50 台、100 台。

真实工程例子：门店摄像头升级时，不同门店环境差异很大。逆光门店、货架密集门店、夜间噪声大的门店，都会放大模型问题。因此灰度设备不应随机乱抽，而应覆盖关键风险分层，否则首批样本看起来稳定，扩量后仍可能出大问题。

---

## 代码实现

工程上通常至少拆成三个模块：设备分组、发布控制、指标判定。一个核心原则是：不要直接覆盖当前模型文件，而要保留版本目录，通过原子切换切换到新版本。原子切换指“外界要么看到旧版本，要么看到新版本，不会看到一半写好的中间状态”。

先看一个配置表：

| 配置项 | 作用 | 例子 |
|---|---|---|
| `max_batch_ratio` | 单轮最大放量比例 | `0.2` |
| `latency_threshold` | 延迟阈值 | `90` ms |
| `error_threshold` | 错误率阈值 | `0.02` |
| `crash_threshold` | 崩溃率阈值 | `0.005` |
| `max_staleness` | 最大允许陈旧度 | `5` |

目录结构可以设计成：

```text
models/
  v1/
  v2/
current -> v1
```

新模型先完整下载到 `models/v2/`，验签、校验哈希、预热加载完成后，再把 `current` 从 `v1` 原子切到 `v2`。这样即使下载失败或目录不完整，也不会破坏当前服务。

下面给一个可运行的 Python 玩具实现，重点是控制逻辑，不是具体网络通信：

```python
from dataclasses import dataclass
from math import exp
from typing import List


@dataclass
class Device:
    device_id: str
    ram_gb: int
    soc_tier: int
    online: bool
    current_version: str


@dataclass
class Metrics:
    p95_latency_ms: float
    error_rate: float
    crash_rate: float


@dataclass
class Config:
    max_batch_ratio: float
    latency_threshold: float
    error_threshold: float
    crash_threshold: float
    max_staleness: int


def select_eligible_devices(devices: List[Device], min_ram_gb: int, min_soc_tier: int) -> List[Device]:
    return [
        d for d in devices
        if d.online and d.ram_gb >= min_ram_gb and d.soc_tier >= min_soc_tier
    ]


def pick_batch(devices: List[Device], ratio: float) -> List[Device]:
    batch_size = max(1, int(len(devices) * ratio))
    return devices[:batch_size]


def metrics_ok(metrics: Metrics, config: Config) -> bool:
    return (
        metrics.p95_latency_ms <= config.latency_threshold
        and metrics.error_rate <= config.error_threshold
        and metrics.crash_rate <= config.crash_threshold
    )


def staleness_weight(tau: int, decay_lambda: float) -> float:
    return exp(-decay_lambda * tau)


def should_accept_update(tau: int, config: Config) -> bool:
    return tau <= config.max_staleness


def rollout_once(devices: List[Device], config: Config, metrics: Metrics) -> str:
    eligible = select_eligible_devices(devices, min_ram_gb=4, min_soc_tier=2)
    batch = pick_batch(eligible, config.max_batch_ratio)

    if not batch:
        return "no_eligible_devices"

    # 实际系统里这里应执行：下载到新版本目录、校验、预热、再切换软链接
    if metrics_ok(metrics, config):
        for d in batch:
            d.current_version = "v2"
        return "expanded"
    else:
        # 实际系统里这里应回切 current -> v1
        for d in batch:
            d.current_version = "v1"
        return "rolled_back"


devices = [
    Device("a", ram_gb=8, soc_tier=3, online=True, current_version="v1"),
    Device("b", ram_gb=6, soc_tier=2, online=True, current_version="v1"),
    Device("c", ram_gb=2, soc_tier=1, online=True, current_version="v1"),
    Device("d", ram_gb=8, soc_tier=3, online=False, current_version="v1"),
]

config = Config(
    max_batch_ratio=0.5,
    latency_threshold=90,
    error_threshold=0.02,
    crash_threshold=0.005,
    max_staleness=5,
)

good_metrics = Metrics(p95_latency_ms=76, error_rate=0.01, crash_rate=0.003)
bad_metrics = Metrics(p95_latency_ms=76, error_rate=0.01, crash_rate=0.008)

assert rollout_once(devices, config, good_metrics) == "expanded"
assert devices[0].current_version == "v2"

devices[0].current_version = "v1"
devices[1].current_version = "v1"

assert rollout_once(devices, config, bad_metrics) == "rolled_back"
assert devices[0].current_version == "v1"

weights = [round(staleness_weight(t, 0.5), 3) for t in [1, 4, 10]]
assert weights == [0.607, 0.135, 0.007]
assert should_accept_update(3, config) is True
assert should_accept_update(7, config) is False
```

这个示例只保留了最关键的策略骨架：

1. 先按设备资源筛选。
2. 再按比例选批次。
3. 指标达标才扩量。
4. 指标不达标则回滚。
5. 异步更新按陈旧度降权或拒收。

真实工程例子可以再往前走一步：如果设备侧是视频推理服务，还要等在途请求完成后再切换版本。在途请求指“已经进入处理流程、但还没处理完的请求”。否则同一段视频前半段可能用旧模型，后半段用新模型，结果不一致。

伪代码可以写成：

```python
for batch in rollout_batches:
    stage_model(batch, version="v2")      # 下载到新目录，不切流
    warmup_model(batch, version="v2")     # 预热模型
    wait_for_inflight_done(batch)         # 等在途请求完成
    switch_current_symlink(batch, "v2")   # 原子切换
    metrics = collect_metrics(batch)

    if metrics_ok(metrics):
        continue
    else:
        switch_current_symlink(batch, "v1")
        stop_rollout()
        break
```

---

## 工程权衡与常见坑

边缘更新最容易出问题的地方，往往不是模型本身，而是发布链路。发布链路就是“模型从云端到设备、再从设备目录变成真实服务版本”的整条路径。

常见坑与规避方式如下：

| 常见坑 | 为什么危险 | 规避方式 |
|---|---|---|
| 上传即切流 | 文件刚到就影响线上 | 先下载、校验、预热，再显式切流 |
| 直接覆盖文件 | 失败时旧版本也被破坏 | 用版本目录和原子切换 |
| 忽略 in-flight 请求 | 同一请求跨版本执行 | 等旧请求排空，或做状态迁移 |
| 所有设备同速更新 | 弱设备容易集中失败 | 按 RAM、SoC、网络质量分层 |
| 不处理 stale update | 旧更新污染聚合结果 | 设陈旧度权重和最大陈旧阈值 |
| 轮询热更新读半成品目录 | 可能加载未完整下载模型 | 使用显式发布标记或原子目录切换 |

这里有几个典型权衡。

第一，分层越细，安全性越高，但运营复杂度越高。比如按 RAM、芯片代次、系统版本、门店场景四维分层会更稳，但策略配置会迅速变多。

第二，观测窗口越长，结论越稳，但上线越慢。观测窗口指“切换后等待指标稳定的时间段”。如果只看 5 分钟，可能漏掉夜间流量高峰或某些长尾场景；如果看 24 小时，发布效率又过低。

第三，回滚越快，恢复越稳，但实现成本更高。真正可靠的回滚通常要求旧版本仍常驻本地、切换动作原子化、监控告警足够快，这些都增加磁盘、内存和控制系统复杂度。

回滚逻辑可以用一个简化状态机理解：

| 状态 | 进入条件 | 下一步 |
|---|---|---|
| `staged` | 新模型已下载未生效 | 等校验/预热 |
| `canary` | 小流量试运行 | 看指标是否达标 |
| `expanded` | 指标达标 | 继续扩量 |
| `rollback` | 任一关键阈值超限 | 回切旧版并停止发布 |

回滚伪代码：

```python
state = "canary"

if crash_rate > crash_threshold or error_rate > error_threshold:
    state = "rollback"
    switch_current_symlink(devices=batch, version="v1")
    mark_version_blocked("v2")
```

玩具例子：你把 `current` 目录直接覆盖成新模型，下载到一半程序重载，读到了半个文件，服务立刻报错。问题不在模型，而在发布动作不是原子的。

真实工程例子：门店摄像头在白天样本上表现正常，但夜间玻璃反光导致误报激增。如果系统只看平均指标，不看门店分层或场景分层，就可能误判为“整体可扩量”。所以监控不能只有全局平均值，还要按设备层级、地区、场景切片分析。

---

## 替代方案与适用边界

边缘更新没有单一标准答案，不同场景需要在稳定性、实时性、实现复杂度之间取舍。

常见方案对比如下：

| 方案 | 是否切主流量 | 是否易回滚 | 是否适合边缘设备 | 主要代价 |
|---|---|---|---|---|
| `shadow` | 否 | 很易 | 适合 | 额外计算开销，不能直接产生产线收益 |
| `rolling update` | 是 | 中等 | 较适合 | 需要持续监控和批次控制 |
| `blue-green` | 是 | 易 | 部分适合 | 需要双份资源，边缘端常常承受不起 |
| `staged rollout` | 是 | 易 | 很适合 | 上线速度较慢，控制逻辑更复杂 |

这几种方案的区别，不在名字，而在“是否真正接管业务流量”和“回退成本是多少”。

`shadow` 适合高风险场景。它只让新模型旁路跑，不接管真实决策。优点是安全，缺点是需要额外算力，而且你只能知道“如果切过去会怎样”，不能真正拿到业务收益。

`rolling update` 是逐批切换，适合大多数在线边缘系统。它平衡了速度和风险，但要求发布控制、监控、回滚链路足够成熟。

`blue-green` 是两套完整环境直接切换。云端很常见，边缘端不总现实，因为很多设备磁盘、内存、算力不足，无法长期维持两套完整推理环境。

`staged rollout` 强调按比例分阶段推进，本质上是边缘系统最常用的工程化形态。它和 rolling 的差异更多在操作层面：staged 更关注“先 1%、再 5%、再 20%”这类明确阶段控制。

适用边界可以总结为：

| 适合使用严格更新策略的条件 | 说明 |
|---|---|
| 设备异构 | 不同设备能力差异大 |
| 需要灰度 | 不能接受全量试错 |
| 需要回滚 | 线上故障必须可恢复 |
| 资源受限 | 每次失败都代价高 |

| 不适合或很难落地的条件 | 说明 |
|---|---|
| 完全离线 | 无远程下发与监控能力 |
| 无在线监控 | 无法判断是否该扩量 |
| 无法远程管理 | 出问题后不能及时回滚 |

玩具例子：一个离线跑批任务每天只在本地生成报表，失败了第二天人工重跑即可，这种任务没必要引入复杂灰度发布。

真实工程例子：自动售货机上的视觉识别设备，弱网、弱算力、现场无人值守。这里比起追求最快更新，更适合采用保守 staged rollout，加显式回滚和长观测窗口。因为任何一次失败都可能在设备现场长期滞留。

---

## 参考资料

1. [Google Play for on-device AI](https://developer.android.com/google/play/on-device-ai) 用于理解移动端与边缘端模型分发、分阶段发布与设备侧交付约束。
2. [NVIDIA Triton Inference Server: Model Management](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html) 对应正文里的模型版本管理、显式加载卸载与避免直接覆盖文件。
3. [NVIDIA Triton Inference Server: Version Policy](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_220/user-guide/docs/model_configuration.html) 对应多版本共存、版本选择与受控切换。
4. [Kubernetes Rolling Updates](https://kubernetes.io/docs/tutorials/kubernetes-basics/update/update-intro/) 对应正文中“先小范围再扩量”的滚动发布思想；文中公式是统一抽象，不是 Kubernetes 文档原式。
5. [FedSA: A Staleness-Aware Asynchronous Federated Learning Algorithm with Non-IID Data for Edge Computing](https://www.sciencedirect.com/science/article/pii/S0167739X21000649) 对应异步联邦中的陈旧更新降权思路；正文中的 $w_i=e^{-\lambda\tau_i}$ 是统一抽象写法，不是该文逐字原式。
6. [Apple Core ML: Personalizing a Model with On-Device Updates](https://developer.apple.com/documentation/CoreML/personalizing-a-model-with-on-device-updates) 用于理解端侧更新、设备本地学习和 on-device 模型生命周期管理。
