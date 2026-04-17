## 核心结论

Ray 可以理解为一个“统一的 Python 分布式运行时”，白话说，就是你还写 Python，但任务可以分散到多核、多机、甚至多 GPU 上执行。它的核心不是某个单点库，而是一组围绕同一调度内核构建的能力：

| 组件 | 负责什么 | 典型输入输出 | 适合场景 |
|---|---|---|---|
| Ray Core | 任务调度、状态化服务、对象引用 | `task`、`actor`、`ObjectRef` | 并行计算、异步 DAG、服务骨架 |
| Ray Data | 流式数据处理与分片 | Dataset、batch、pipeline | 预处理、ETL、训练前喂数 |
| Ray Train | 分布式训练封装 | `train_func`、`ScalingConfig` | PyTorch DDP、多 GPU 训练 |
| Ray Tune | 超参数搜索与实验调度 | trial、search space | 自动调参、早停、实验对比 |
| Ray Serve | 在线部署与请求路由 | deployment、replica、request | 模型服务、推理路由、在线 API |

最重要的结论有两个。

第一，Ray Core 的 `task/actor` 模型是整个栈的基础。`task` 是远程任务，白话说，就是一个无状态函数调用；`Actor` 是远程常驻对象，白话说，就是带内存和状态的服务实例。只要把函数或类用 `@ray.remote` 包起来，再调用 `.remote()`，Ray 就会在后台调度 worker 执行。

第二，Ray 的价值不只是“把 Python 并行起来”，而是把数据处理、训练、调参、部署放进同一套资源调度体系里。这样做的直接收益是：CPU 预处理、GPU 训练、在线推理可以共享集群和对象存储，不必在多个系统之间反复搬数据。

一个新手最容易上手的例子是：把现有函数改成远程任务。

```python
import ray

ray.init()

@ray.remote
def square(x):
    return x * x

ref = square.remote(8)   # 立即返回 ObjectRef
result = ray.get(ref)    # 真正取回结果
assert result == 64
```

这里的 `ObjectRef` 是“远程结果的引用”，白话说，它像一个提货单，先拿到编号，真正用结果时再去取货。这个设计让 Ray 能自动构造异步依赖图，而不是每一步都同步阻塞。

---

## 问题定义与边界

Ray 解决的不是“任何分布式问题”，而是 Python 工作流里最常见的一类问题：任务很多、数据很多、资源异构，而且 CPU 与 GPU 的节奏不一致。

一个典型矛盾是这样的：

1. CPU 负责解码、清洗、增强、切分数据。
2. GPU 负责训练或推理。
3. 如果必须“前一步全做完，后一步再开始”，GPU 很容易空转。
4. 如果每个环节都由不同系统管理，数据复制、序列化和故障恢复会变复杂。

Ray Data 试图解决的就是这个“供数速度跟不上算力消耗”的问题。它的流式执行可以理解为“批次沿流水线滑动前进”，不是等所有数据预处理完再统一训练，而是预处理一点，训练一点，继续往前推。

下面这个表可以把资源流向看清楚：

| 阶段 | 主要资源 | Ray 里的角色 | 可能的瓶颈 | 典型通信点 |
|---|---|---|---|---|
| 数据读取 | CPU、磁盘、网络 | Ray Data source/read | IO 吞吐不足 | 对象存储写入 |
| 预处理/增强 | CPU、内存 | `map_batches` 等算子 | CPU 算子太慢 | batch 传入下游 |
| 训练 | GPU、显存、网络 | Ray Train worker | GPU 等数据、梯度同步慢 | DDP all-reduce |
| 在线推理 | GPU/CPU、网络 | Ray Serve replica | 请求突发、慢副本堆积 | 路由器到副本 |

玩具例子可以先看“图片分类流水线”的缩小版。假设你有 1000 张图片：

- CPU 做 resize、归一化、数据增强。
- GPU 做前向和反向传播。
- 如果 CPU 每秒只准备 20 个 batch，但 GPU 每秒能吃 40 个 batch，那么 GPU 有一半时间在等。
- 如果改为 Ray Data 的流式 batch，CPU 不断把处理好的小批次送到下游，GPU 就能持续消费，而不是整批等待。

真实工程里，这种问题更明显。比如一个图像分类平台既要离线训练，又要在线推理：

- 白天在线流量高，Serve 需要更多副本处理请求。
- 夜间训练任务多，Train 需要更多 GPU。
- 数据增强与特征抽取又主要吃 CPU。

Ray 的边界在于：它擅长统一调度 Python 计算和 ML 工作流，但并不是数据库、消息队列或通用大数据仓库的替代品。你仍然需要外部存储保存原始数据，需要监控系统观测集群健康，也可能需要对象存储或湖仓承担长期数据管理。

另一个必须说清的边界是容错语义。很多新手会默认“分布式框架一定自动恢复一切”，这在 Ray 中并不成立：

- Serve 有默认路由策略，但不等于业务级幂等和状态恢复自动完成。
- Actor 默认不重启，进程崩了，状态就可能直接丢失。
- Train 可以重试 worker，但你的数据写入、checkpoint 和外部副作用仍要自己设计。

---

## 核心机制与推导

Ray Core 的底层抽象可以概括成一句话：任务形成 DAG，状态绑定到 Actor，对象通过引用连接依赖。DAG 是“有向无环图”，白话说，就是任务之间谁先谁后的依赖关系图。

### 1. Task/Actor 如何组成异步执行图

当你写：

```python
a = f.remote()
b = g.remote(a)
```

这不是“先取回 `a` 再传给 `g`”，而是把 `a` 的 `ObjectRef` 作为依赖交给 Ray。调度器知道：`g` 必须等 `f` 产出结果，但调用方不需要卡住等待。

对新手而言，这一点非常重要，因为它意味着你不用手写线程池、进程池、队列和锁，就能得到一个异步 DAG。

### 2. Ray Data 为什么能提高 GPU 利用率

Ray Data 的核心不是“又一个 DataFrame”，而是把数据处理拆成批次并交给运行时持续推进。设：

- CPU 预处理平均速度为 $v_c$ 个 batch/秒
- GPU 消费平均速度为 $v_g$ 个 batch/秒
- 队列长度为 $Q$

如果采用同步分阶段执行，总耗时近似是：
$$
T_{\text{sync}} = T_{\text{cpu}} + T_{\text{gpu}}
$$

如果采用流水化执行，总耗时更接近：
$$
T_{\text{pipe}} \approx \max(T_{\text{cpu}}, T_{\text{gpu}}) + T_{\text{fill/drain}}
$$

其中 `fill/drain` 是流水线开始和结束时的填充、清空成本。只要 CPU 和 GPU 可以并行推进，整体时间通常更接近两者中的较大者，而不是简单相加。

### 3. Ray Train 如何映射到 DDP

DDP 是 Distributed Data Parallel，白话说，就是每张 GPU 各跑一份模型，分别处理不同数据，再把梯度同步起来。Ray Train 的作用是把这些 worker 的启动、资源申请、环境布置和分片数据接入统一封装。

在 4 节点 × 4 GPU 的集群里，共 16 张 GPU。如果你写：

```python
ScalingConfig(num_workers=16, use_gpu=True)
```

Ray Train 会启动 16 个训练 worker。每个 worker 运行相同的 `train_func`，每个 worker 看到的是自己那一份数据分片，然后通过 DDP 做梯度同步。

一个最小数值推导如下：

- 全局 batch size = 256
- worker 数 = 16
- 每个 worker 本地 batch size = 16

则有：
$$
256 = 16 \times 16
$$

每次迭代中，16 个 worker 分别算出本地梯度，再通过 all-reduce 汇总。对使用者而言，重点不是自己手搓进程组，而是确认 `num_workers`、每卡 batch size、显存占用与吞吐匹配。

### 4. Ray Serve 的默认路由为什么是 O(1)

Serve 在多副本下要回答一个问题：新请求发给谁？默认的一个关键思路是 “Power of Two Choices”。它不是扫描所有副本，而是随机采样两个副本 $j,k$，比较队列长度 $q_j,q_k$，选择更空闲的那个：

$$
r=\arg\min_{i\in\{j,k\}} q_i
$$

这里的意思很直接：只看两个候选，就能在常数时间内做出足够好的均衡决策。

玩具例子：

- 副本 A 当前排队 8 个请求
- 副本 B 当前排队 3 个请求
- 随机抽到的是 A 和 B
- 则选择 B，因为 $3 < 8$

真实工程例子是 LLM 服务。假设某些副本因为上下文较长、KV cache 压力较大而暂时变慢，如果路由器总是平均轮询，慢副本会继续积压；而 Two Choices 至少能快速避开明显更拥堵的副本。对于前缀相似请求较多的 LLM 服务，还可以在更高级策略中结合 prefix cache 亲和性，把相似前缀路由到同一副本，提高缓存命中率。

---

## 代码实现

下面给出一个“能跑、能看懂、能映射到真实 Ray 抽象”的最小实现。第一段是纯 Python 的玩具模拟，用来解释 `ObjectRef`、任务和路由思想；第二段是 Ray 风格代码骨架，用来对应真实工程写法。

```python
from dataclasses import dataclass

@dataclass
class ObjectRef:
    value: object

def remote_func(fn):
    def wrapper(*args, **kwargs):
        real_args = [a.value if isinstance(a, ObjectRef) else a for a in args]
        real_kwargs = {
            k: (v.value if isinstance(v, ObjectRef) else v)
            for k, v in kwargs.items()
        }
        return ObjectRef(fn(*real_args, **real_kwargs))
    wrapper.remote = wrapper
    return wrapper

def choose_replica(q1, q2):
    return 0 if q1 <= q2 else 1

@remote_func
def preprocess(batch):
    return [x / 255.0 for x in batch]

@remote_func
def train_step(batch):
    return sum(batch) / len(batch)

class ServingActor:
    def __init__(self):
        self.counter = 0

    def predict(self, batch):
        self.counter += len(batch)
        return [x * 2 for x in batch]

raw = [0, 128, 255]
ref_batch = preprocess.remote(raw)
loss_ref = train_step.remote(ref_batch)

actor = ServingActor()
pred = actor.predict([1.0, 2.0])

assert ref_batch.value == [0.0, 128/255.0, 1.0]
assert abs(loss_ref.value - sum(ref_batch.value) / 3) < 1e-12
assert pred == [2.0, 4.0]
assert actor.counter == 2
assert choose_replica(8, 3) == 1
assert choose_replica(2, 2) == 0
```

这段代码不是 Ray 本体，但它把核心概念缩小到了肉眼可验证的级别：

- `ObjectRef`：远程结果引用。
- `preprocess.remote(raw)`：模拟异步任务提交。
- `train_step.remote(ref_batch)`：模拟“把上游结果引用交给下游任务”。
- `ServingActor`：模拟有状态服务对象。
- `choose_replica`：模拟 Serve 路由决策。

对应到真实 Ray，代码骨架通常是这样：

```python
import os
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
ray.init()

@ray.remote
def preprocess(batch):
    return batch / 255.0

@ray.remote
class ServingActor:
    def __init__(self, model):
        self.model = model
        self.counter = 0

    def predict(self, batch):
        self.counter += len(batch)
        return self.model(batch)

def train_func():
    # 运行在每个 GPU worker 上
    # 这里通常会：
    # 1. 构建模型
    # 2. 从 Ray Data 读取分片 batch
    # 3. 前向、反向、优化器更新
    # 4. 保存 checkpoint
    pass

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True)
)

result = trainer.fit()
```

这段代码里每一行都能映射到一个具体抽象：

| 代码元素 | 抽象 | 作用 |
|---|---|---|
| `ray.init()` | 运行时入口 | 连接本地或集群 Ray |
| `@ray.remote` 函数 | Task | 无状态并行执行单元 |
| `@ray.remote` 类 | Actor | 带状态的远程服务 |
| `.remote()` | 异步提交 | 立即返回引用，不阻塞 |
| `TorchTrainer` | 训练编排器 | 启动多 worker 并接入 DDP |
| `ScalingConfig` | 资源声明 | 指定 worker 数量和 GPU 配额 |

真实工程例子可以这样串起来：

1. 用 Ray Data 从对象存储读取图片数据。
2. 在 CPU 上做解码、resize、标准化。
3. 把处理好的分片 batch 交给 `TorchTrainer` 的 worker。
4. 训练完成后，把模型挂到 Serve deployment。
5. 在线请求再进入多个推理副本，由 Serve 路由。

这样数据流、训练流、服务流都在同一套运行时里，而不是“预处理一套系统、训练一套系统、服务又一套系统”。

---

## 工程权衡与常见坑

Ray 的工程价值很高，但坑也非常具体，尤其是新手常常把“能跑通 demo”和“能长期稳定运行”混为一谈。下面这个表最值得先记住：

| 常见坑 | 现象 | 根因 | 规避方式 |
|---|---|---|---|
| Train + Tune 配合不稳定 | 调参试验行为不一致 | 走了旧训练轨道 | 设置 `RAY_TRAIN_V2_ENABLED=1` |
| Actor 崩溃后状态丢失 | 服务偶发挂掉后计数器/缓存没了 | Actor 默认不重启 | `.options(max_restarts=2, max_task_retries=...)` |
| 过早 `ray.get()` | 并发度下降，像串行执行 | 把异步 DAG 人为同步化 | 尽量晚取结果，先提交更多任务 |
| batch 太小 | GPU 利用率低，调度开销占比高 | 任务过碎 | 增大 batch 或合并算子 |
| batch 太大 | OOM、长尾严重 | 单任务资源占用过高 | 做 profiling，按显存和吞吐回推 |
| 把 Actor 当数据库 | 状态难恢复，扩缩容麻烦 | Actor 不是持久化系统 | 关键状态写外部存储 |
| 忽略数据倾斜 | 某些 worker 特别慢 | 分片不均衡 | 重分片、采样、调整输入分布 |

有两个问题尤其值得展开。

### 1. 训练轨道切换

Ray Train 和 Ray Tune 的集成不是“只要 import 就自动最佳实践”。如果你需要稳定做分布式训练 + 调参，应该显式启用新训练轨道，也就是 `Train V2`。最直接的做法是在脚本启动前设置：

```bash
export RAY_TRAIN_V2_ENABLED=1
```

或者在 Python 顶部设置环境变量。这一步的意义不是语法，而是让调参与训练走同一条较新的执行路径，减少旧路径里的行为差异。

### 2. Actor 容错不是默认开启

很多服务会把模型缓存、用户会话、计数器放在 Actor 里。如果这个 Actor 挂了，而你没有设置重启策略，那就是直接丢状态。可以写成：

```python
MyActor.options(max_restarts=2, max_task_retries=3).remote()
```

这并不等于“完全高可用”，但至少把短暂崩溃从人工介入恢复，变成自动拉起恢复。真正关键的状态，仍然建议落到外部存储。

再给一个真实工程判断标准：如果你的 Serve 副本里缓存的是“可丢失的推理缓存”，Actor 重启通常可以接受；如果缓存的是“订单、会话、扣费结果”，那就不该只存在 Actor 内存里。

---

## 替代方案与适用边界

Ray 不是所有团队的起点。很多时候，更简单的系统更合适。

| 方案 | 适用场景 | 复杂度 | Stateful 服务/LLM |
|---|---|---|---|
| Ray | 需要统一数据、训练、调参、部署 | 中到高 | 强，Actor 和 Serve 适合 |
| Dask | 以 Python 数据处理、并行为主 | 中 | 弱，不以状态化服务见长 |
| Spark | 大规模批处理、SQL、湖仓生态 | 中到高 | 弱，在线服务不是强项 |

如果你只是做 ETL，也就是抽取、转换、加载数据的批处理流程，Dask 往往更轻。它对“把表算完、把结果落盘”这类任务很直接。新手如果只是想把一批 CSV 清洗成 Parquet，不一定需要一开始就上 Ray。

如果你主要做超大规模离线数据处理、SQL 分析、湖仓接入，Spark 的生态更成熟。它的优势在于数据工程体系，而不是 Python 原生任务和在线服务统一。

Ray 适合升维的时机通常有三个：

1. 你已经不只做 ETL，还要接 GPU 训练。
2. 你需要状态化服务，例如模型副本、会话缓存、在线路由。
3. 你希望调参与训练、训练与部署处在同一集群资源视图下。

可以这样理解：

- “只做数据处理”时，Dask/Spark 往往够用。
- “数据处理 + 训练 + 在线推理”都要打通时，Ray 的统一运行时价值才真正体现出来。
- “服务要带状态”时，Actor/Serve 的模型比单纯批处理框架更合适。

一个现实判断边界是：如果团队暂时没有 GPU、没有在线服务、没有复杂状态，只是周期性批处理，那么 Ray 可能偏重；但如果你已经进入 ML 工作流阶段，尤其需要从单机自然扩到多机多 GPU，Ray 会比“拼装多个孤立系统”更省工程摩擦。

---

## 参考资料

- Ray 官方概览：<https://docs.ray.io/en/master/ray-overview/index.html>
- Ray Core Actors 文档：<https://docs.ray.io/en/latest/ray-core/actors.html>
- Ray Data 文档：<https://docs.ray.io/en/latest/data/data.html>
- Ray Serve 路由策略：<https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html>
- PyTorch 官方教程，使用 Ray 做分布式训练：<https://docs.pytorch.org/tutorials/beginner/distributed_training_with_ray_tutorial.html>
- Ray Train 超参数优化与 Train V2 说明：<https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html>
