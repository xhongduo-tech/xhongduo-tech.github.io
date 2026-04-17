## 核心结论

Tensor Parallel，简称 TP，就是把同一层的张量切到多张 GPU 上并行计算。它带来的一个直接后果是：随机操作不能再只看“这一步要不要随机”，而要看“这一步的随机性是否必须跨 TP rank 保持一致”。

结论分两条：

1. 某些算子必须在同一个 TP group 内共享随机种子。最典型的是 residual 分支汇合前后、会参与后续聚合或要求数学等价的 dropout。原因很直接：同一份逻辑张量如果在不同 GPU 上被不同 mask 丢弃，后面的梯度聚合就不再等价于单卡训练。
2. 另一些算子必须在 TP 内部使用不同随机种子。最典型的是 model-parallel 子模块内部、每个 rank 只持有局部权重分片时的 dropout。原因也直接：这些局部计算本来就代表不同子空间，如果所有 rank 用完全相同的 mask，会减少正则化多样性。

Megatron-LM 的做法不是“一个全局 seed 走天下”，而是维护两个 CUDA RNG 状态：

- 默认状态：给跨 TP 需要同步随机性的算子使用
- `model-parallel` 状态：给 TP 内部需要差异化随机性的算子使用

核心等价关系可以写成：

$$
m = \text{DropoutMask}(\text{seed}), \quad y = f(m \odot x)
$$

如果所有 TP rank 使用同一个 `seed`，那么它们得到同构的随机行为，即 mask 的生成规则一致。对于必须同步的算子，这样才能保持：

$$
\text{AllReduce}(f(\text{dropout}(x;\,\text{shared seed})))
\equiv
f(\text{dropout}(x;\,\text{shared seed}))
$$

这里“同构”用白话说，就是不同 GPU 虽然各自执行，但随机决策像是按同一本随机脚本在走。

---

## 问题定义与边界

问题不是“dropout 要不要同步”，而是“哪个 dropout 要同步，哪个不该同步”。边界由算子所在的位置决定，而不是由算子名字决定。

可以先看一个三层玩具 block：

1. 输入经过 TP 线性层，输出被切到不同 rank
2. 每个 rank 做自己那一片的局部计算
3. 残差汇合、通信、进入下一层

这时随机操作通常分成两类：

| 算子位置 | 张量语义 | TP 内 seed 规则 | 原因 |
|---|---|---:|---|
| residual 路径前后的 dropout | 逻辑上代表同一个激活 | 相同 | 要保证后续聚合与单卡等价 |
| TP 线性层内部局部 dropout | 每个 rank 只处理自己分片 | 不同 | 保持局部正则化多样性 |
| attention/MLP 内部、分片后局部激活 dropout | 局部子空间 | 不同 | 各 rank 本来就不是同一整张量 |
| 某些需要重计算的随机算子 | 取决于是否跨 rank 对齐 | 由实现决定 | 前向和反向必须可重放 |

可以用一个简化流程图理解：

| 阶段 | 是否代表“同一逻辑张量” | 是否会与其他 rank 聚合保持等价 | 种子策略 |
|---|---|---|---|
| 分片前或残差主干上 | 是 | 是 | 共享 seed |
| 分片后的 rank 私有子模块 | 否 | 否 | 每 rank 不同 seed |

最容易犯错的地方，是把“同一种算子”误认为“同一种 seed 规则”。例如都是 dropout，但 residual 前的 dropout 和 MLP 分片内部的 dropout，种子要求可能相反。

如果边界搞错，会出现两类后果：

- 该同步的没同步：不同 rank 对同一逻辑激活使用不同 mask，梯度聚合失去等价性
- 该不同的反而同步：所有 rank 做出几乎一样的随机删减，正则化变弱，收敛和泛化都可能变差

---

## 核心机制与推导

Megatron 的种子初始化不是简单 `seed = base_seed`，而是先把并行维度编码进去。常见形式是：

$$
\text{seed} = \text{base} + 100 \times pp\_rank + 10 \times dp\_rank
$$

这里：

- PP 是 Pipeline Parallel，流水并行，就是不同 GPU 负责不同层段
- DP 是 Data Parallel，数据并行，就是不同 GPU 处理不同样本批次
- TP 是 Tensor Parallel，张量并行，就是同一层拆到多卡

然后再构造 TP 专用种子：

$$
\text{tensor\_seed} = \text{seed} + 2718 + tp\_rank
$$

给出题目里的数值例子：

- `base = 2024`
- `pp_rank = 1`
- `dp_rank = 2`
- `tp_rank = 3`

先算基础 seed：

$$
\text{seed} = 2024 + 100 \times 1 + 10 \times 2 = 2144
$$

再算 TP 专用 seed：

$$
\text{tensor\_seed} = 2144 + 2718 + 3 = 4865
$$

含义是：

- `2144` 这条 RNG 状态，用在跨 TP 需要同步的随机算子
- `4865` 这条 RNG 状态，用在 TP 内部需要差异化的随机算子

Megatron 不是每次算子执行前都重新 `manual_seed`，而是用 `CudaRNGStatesTracker` 保存多个命名状态。`tracker` 可以理解成“随机数上下文管理器”，白话说就是它帮你记住多份随机进度条，并在进入不同算子时切换到对应那一份。

伪代码如下：

```python
tracker.add("default", seed=2144)          # 跨 TP 同步
tracker.add("model-parallel", seed=4865)   # TP 内部独立

with tracker.fork("default"):
    y = residual_dropout(x)

with tracker.fork("model-parallel"):
    z = tp_internal_dropout(local_x)
```

这里的 `fork` 不是复制整个模型，而是临时切到某个 RNG 状态，退出时再恢复。这样做有两个作用：

1. 前向和反向重计算时，可以复现完全一致的随机序列
2. 不同类别算子不会互相污染随机状态

为什么共享 seed 能保持等价？看一个最小推导。设 dropout mask 为 $m$，输入为 $x$，输出为：

$$
\text{dropout}(x) = \frac{m \odot x}{1-p}
$$

如果 rank 0 和 rank 1 对同一逻辑激活使用不同 mask，即 $m_0 \neq m_1$，那么聚合后的结果是：

$$
\frac{m_0 \odot x}{1-p} + \frac{m_1 \odot x}{1-p}
$$

它不再对应“单次 dropout 后的同一个张量”，而是两个不同随机试验的叠加。对某些需要严格等价的路径，这就是错误来源。

---

## 代码实现

下面用一个可运行的 Python 玩具实现说明“共享 seed”和“不同 seed”的区别。它不是 CUDA 版，但逻辑和 Megatron 的 tracker 一致。

```python
import random
from contextlib import contextmanager

class RNGTracker:
    def __init__(self):
        self.states = {}
        self.active_name = None

    def add(self, name, seed):
        rng = random.Random(seed)
        self.states[name] = rng.getstate()

    @contextmanager
    def fork(self, name):
        if name not in self.states:
            raise KeyError(f"unknown rng state: {name}")
        current = random.getstate()
        random.setstate(self.states[name])
        try:
            yield
        finally:
            self.states[name] = random.getstate()
            random.setstate(current)

def dropout_mask(n, p=0.5):
    return [0 if random.random() < p else 1 for _ in range(n)]

def apply_dropout(x, p=0.5):
    mask = dropout_mask(len(x), p)
    scale = 1.0 / (1.0 - p)
    return [xi * mi * scale for xi, mi in zip(x, mask)], mask

base = 2024
pp_rank = 1
dp_rank = 2
tp_rank = 3

seed = base + 100 * pp_rank + 10 * dp_rank
tensor_seed = seed + 2718 + tp_rank

assert seed == 2144
assert tensor_seed == 4865

tracker_rank0 = RNGTracker()
tracker_rank1 = RNGTracker()

for tracker in (tracker_rank0, tracker_rank1):
    tracker.add("default", seed)           # 共享
tracker_rank0.add("model-parallel", tensor_seed + 0)
tracker_rank1.add("model-parallel", tensor_seed + 1)

x = [1.0, 2.0, 3.0, 4.0]

with tracker_rank0.fork("default"):
    y0, m0 = apply_dropout(x, p=0.5)
with tracker_rank1.fork("default"):
    y1, m1 = apply_dropout(x, p=0.5)

assert m0 == m1  # residual dropout 需要相同 mask

with tracker_rank0.fork("model-parallel"):
    z0, mz0 = apply_dropout(x, p=0.5)
with tracker_rank1.fork("model-parallel"):
    z1, mz1 = apply_dropout(x, p=0.5)

assert mz0 != mz1  # TP 内部 dropout 允许不同 mask
```

上面这段代码对应两个场景：

- `default`：残差路径或其他跨 TP 需要同步的随机操作
- `model-parallel`：TP 子模块内部的局部随机操作

如果把它翻成更接近 Megatron 的伪代码，通常是这样：

```python
def forward(x, rng_tracker):
    with rng_tracker.fork("default"):
        x = residual_dropout(x)

    x_local = tp_linear_shard(x)

    with rng_tracker.fork("model-parallel"):
        x_local = internal_dropout(x_local)

    x = all_reduce_or_gather(x_local)
    return x
```

工程上还有一个关键约束：执行 `model_parallel_cuda_manual_seed(...)` 之后，不要再随手调用 `torch.cuda.manual_seed(...)` 覆写默认 CUDA RNG 状态。否则 tracker 记录的“默认状态”和你实际运行时的状态会脱节，结果就是看起来 seed 设置了，实际 mask 还是错的。

真实工程例子可以看 Transformer block：

1. 输入经过 LayerNorm
2. 进入 TP 切分的 attention 或 MLP
3. 局部计算结束后做通信
4. residual 加回主干

此时：

- block 内部局部 dropout 可以走 `model-parallel` RNG
- residual 主干上的 dropout 必须走默认同步 RNG

这不是“写法偏好”，而是数值一致性要求。

---

## 工程权衡与常见坑

TP 随机性同步的代价不是算力，而是状态管理复杂度。你必须显式知道每个随机算子属于哪一类，并保证前向、反向、重计算的切换完全一致。

常见坑可以直接列出来：

| 错误做法 | 结果 | 规避方式 |
|---|---|---|
| `model_parallel_cuda_manual_seed` 后又调用 `torch.cuda.manual_seed` | 默认 RNG 被覆写，残差 dropout 不再同步 | 统一通过 tracker 管理，不要绕过它 |
| 所有 dropout 都共用同一个 seed | TP 内部随机性不足，正则化变弱 | 区分 `default` 与 `model-parallel` |
| 所有 dropout 都按 rank 加偏移 | residual 路径 mask 不一致，AllReduce 后不等价 | 只对 TP 内局部算子使用 rank 专属 seed |
| 前向用了 tracker，反向重计算没用 | 重新计算出的 mask 不一致 | checkpoint/recompute 路径也必须走同一 tracker |
| 不做任何可视化或哈希检查 | 问题只在验证集下降时暴露，定位困难 | 采样打印 RNG state hash 和 mask checksum |

一个典型线上 bug 是：

- 初始化时正确调用了 `model_parallel_cuda_manual_seed`
- 后面某个模块作者又写了 `torch.cuda.manual_seed(step + token_id)`
- 结果 residual dropout 在 TP rank 间不再一致

这种 bug 的症状通常不是立刻崩溃，而是：

- loss 抖动变大
- 相同配置复现不了
- 验证集指标明显下降
- 某些 batch 上出现异常大梯度

排查时可以做两类检查。

第一类，检查同组 rank 的“共享算子”mask 是否一致。  
第二类，检查 TP 局部算子的 mask 是否意外完全相同。

如果要快速打点，一个实用方法是打印 RNG state 的摘要值，或者直接打印某个 dropout mask 的哈希。例如对前几个 token、前几个 hidden 位置做 checksum，比全量打印更便宜，也足够发现不同步问题。

这里的权衡很明确：

- 管得越严格，可复现性越强，调试越容易
- 管得越松，代码看起来简单，但一旦出错很难定位

在大模型训练里，后者通常不可接受。

---

## 替代方案与适用边界

不是所有并行训练都需要这套 tracker 机制。关键看你是否同时存在“有些随机操作必须相同，有些又必须不同”的局部约束。

可以按场景判断：

| 场景 | 是否需要多 RNG 状态 tracker | 原因 |
|---|---|---|
| 只有单卡 | 不需要 | 一个 CUDA RNG 就够 |
| 只有 Data Parallel | 通常不需要 | 各副本本来独立，单一 seed 策略足够 |
| TP + DP | 需要 | 同时存在跨 TP 同步与 TP 内差异 |
| TP + DP + PP | 强烈需要 | 并行维度更多，单一全局 seed 无法表达局部约束 |

简单说：

- 只做 DP 时，可以直接 `torch.cuda.manual_seed_all(...)`
- 一旦进入 TP，单一全局 seed 基本就不够用了
- 当 TP、DP、PP 混合时，tracker 不是“更优雅”，而是“唯一能稳定表达约束的方案”

也可以把它理解成一个决策规则：

1. 这个随机算子对应的是不是同一逻辑张量？
2. 它的输出后面是否要求跨 rank 数学等价？
3. 如果答案是“是”，就用共享 RNG
4. 如果答案是“否”，且它位于 TP 局部子模块内部，就用 rank 专属 RNG

替代方案当然存在，比如手工给每个算子硬编码 seed 偏移。但这种方法有三个问题：

- 算子一多，维护成本迅速失控
- 前向和反向重计算很容易对不上
- 新增并行维度后规则会重新爆炸

所以在真实大模型工程里，手工 seed 偏移更像调试手段，不是长期方案。Megatron 这类 tracker 方案的价值，正是在于把“随机性同步策略”从零散约定变成一套可维护机制。

---

## 参考资料

1. NVIDIA Megatron Core `tensor_parallel.random` API  
   支持章节：`核心机制与推导`、`代码实现`、`工程权衡与常见坑`。  
   作用：说明 `CudaRNGStatesTracker`、`model_parallel_cuda_manual_seed` 和多 RNG 状态管理接口的官方语义。

2. Megatron-LM 论文《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》  
   支持章节：`核心结论`、`问题定义与边界`。  
   作用：给出模型并行训练中保持数值等价的基本原则，可据此理解为什么某些 dropout 必须同步。

3. Slapo 随机数管理实现与文档  
   支持章节：`核心机制与推导`、`代码实现`。  
   作用：提供 `seed = base + 100*pp + 10*dp` 与 `tensor_seed = seed + 2718 + tp` 这一类具体公式和实现参考。

4. Ultra-Scale Playbook on GPU Clusters  
   支持章节：`工程权衡与常见坑`、`替代方案与适用边界`。  
   作用：从大规模训练实践角度强调 TP 中同步 dropout seed 的必要性，以及错误同步对可复现性和收敛的影响。

5. NVIDIA Megatron Core Developer Guide  
   支持章节：`代码实现`、`工程权衡与常见坑`。  
   作用：解释为什么不能在初始化后随意覆写 CUDA 默认 RNG 状态，以及 checkpoint/recompute 路径如何保持一致。
