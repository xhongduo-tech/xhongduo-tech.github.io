## 核心结论

流水线并行（Pipeline Parallelism, PP）中的“权重同步”问题，核心不在于“有没有缓存权重”，而在于：

**同一个微批次在穿过整条流水线时，看到的是不是同一代参数。**

这一点必须拆成两个层面看。

第一层是 **stage 内一致性**。同一个微批次在某个 stage 上做 forward 时用了哪一版参数，之后回到这个 stage 做 backward 时也必须用同一版。否则该 stage 内部的链式法则就不成立，算出来的梯度不再对应同一张计算图。PipeDream 的 `weight stashing` 解决的正是这个问题。

第二层是 **跨 stage 一致性**。即使每个 stage 都各自做了 stashing，同一个微批次在 stage 1 看到的参数版本，也可能和它后来到 stage 2、stage 3 时看到的版本不同。这叫跨 stage 的版本错位，或者 vertical inconsistency。它不会立即让程序报错，但会改变训练语义：梯度不再来自同一个全局参数快照。

因此，常见三类流水线策略必须分开理解：

| 策略 | 前后向是否同版 | 跨 stage 是否同版 | 需要缓存的权重版本数 | 吞吐 | 收敛语义 |
|---|---|---:|---:|---:|---|
| PipeDream | 是 | 不一定 | 多份 | 高 | 异步、延迟梯度 |
| PipeDream-2BW | 是 | 延迟统一为 1 步 | 2 份 | 高 | 固定 1 步延迟 |
| PipeDream-Flush | 是 | 是 | 通常更少 | 中等 | 接近同步 SGD |

结论可以压缩成三条：

1. `weight stashing` 是局部一致性机制，只保证“某个 stage 内”前后向一致。
2. PipeDream-2BW 通过双缓冲把全流水线约束成统一的 1 步延迟，因此只需要 2 个版本。
3. PipeDream-Flush 通过周期性清空流水线恢复同步语义，代价是 bubble（流水线空泡，即设备短时空转）更多，吞吐下降。

---

## 问题定义与边界

本文只讨论 **训练阶段** 的流水线并行，不讨论推理阶段。讨论重点也不是带宽压缩、激活重计算、通信 overlap，而是：

**当不同 stage 可能持有不同版本参数时，这会怎样影响梯度正确性、优化语义和训练稳定性。**

先固定几个术语。

| 术语 | 定义 | 直观理解 |
|---|---|---|
| 微批次（microbatch） | 把一个大 batch 切成更小的训练单元 | 让不同样本分批进入流水线 |
| stage | 把模型按层切开的一个分段 | 一段连续层放到一组设备上 |
| in-flight microbatch | 已进入流水线、但还没完成前后向的微批次 | 还“挂”在流水线里的样本 |
| 版本错位 | 同一个逻辑训练步中，不同 stage 用了不同代参数 | 样本沿流水线看到的不是同一套权重 |
| stashing | 把某微批次 forward 时用到的权重版本保存起来，供 backward 回放 | 给 backward 找回当时的参数上下文 |
| flush | 一轮微批次执行完后先排空流水线，再统一更新 | 用停顿换同步语义 |

下面用一个 3-stage 玩具时间线说明问题。设参数版本依次为 $W^{(0)}, W^{(1)}, W^{(2)}$。

| 时间片 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| t1 | F(m1, $W^{(0)}$) | - | - |
| t2 | F(m2, $W^{(0)}$) | F(m1, $W^{(0)}$) | - |
| t3 | F(m3, $W^{(1)}$) | F(m2, $W^{(0)}$) | F(m1, $W^{(0)}$) |
| t4 | B(m1, stash=$W^{(0)}$) | F(m3, $W^{(1)}$) 或 $W^{(0)}$ | F(m2, $W^{(0)}$) |

这个时间线里有两类不一致。

### 1. stage 内不一致

如果 `m1` 在 stage 1 做 forward 时用的是 $W^{(0)}$，但它回到 stage 1 做 backward 时 stage 1 已经更新到了 $W^{(1)}$，那么 backward 实际上是在另一张计算图上求导。  
这会直接破坏该 stage 的梯度语义。

所以对每个 stage，都必须保证：

$$
W_{i,\text{fwd}}(m)=W_{i,\text{bwd}}(m)
$$

这里 $i$ 表示第 $i$ 个 stage，$m$ 表示某个微批次。

### 2. 跨 stage 不一致

即使上式成立，也仍可能出现：

$$
W_1(m)\neq W_2(m)\neq \cdots \neq W_p(m)
$$

也就是同一个微批次在 stage 1、2、3 上看到的不是同一代参数。  
这不会让局部 backward 失效，但会让整个模型的梯度变成“拼接快照”的梯度。

因此真正的问题不是“要不要缓存权重”，而是：

$$
\text{同一微批次 } m \text{ 的全模型梯度，是否来自同一个参数上下文 } W
$$

如果答案是“是”，训练语义接近同步 SGD。  
如果答案是“不是”，训练本质上就已经进入了延迟梯度或异步优化范式。

这也是为什么很多初学者会误解：  
他们看到 `stash` 后，以为“既然 backward 用回了正确权重，那不就同步了吗”。实际上不是。`stash` 只修复了 stage 内部的一致性，没有自动修复整条流水线上的全局一致性。

---

## 核心机制与推导

假设模型被切成 $p$ 个 stage，整体参数写成：

$$
W=(W_1, W_2, \dots, W_p)
$$

标准同步训练下，一次更新的理想形式是：

$$
W^{(t+1)} = W^{(t)} - \eta \nabla f\left(W^{(t)}\right)
$$

其中：

- $t$ 是逻辑训练步
- $\eta$ 是学习率
- $f(W)$ 是当前 batch 的损失函数

这条公式隐含了一个很强的前提：

**计算梯度时，整模型都使用同一时刻的参数快照 $W^{(t)}$。**

而在流水线并行中，这个前提往往不成立。更接近真实情况的写法是：

$$
W^{(t+1)} = W^{(t)} - \eta \nabla f\left(W_1^{(t-\tau_1)}, W_2^{(t-\tau_2)}, \dots, W_p^{(t-\tau_p)}\right)
$$

其中 $\tau_i$ 表示第 $i$ 个 stage 的参数延迟。  
如果各 stage 的 $\tau_i$ 不相同，说明同一次梯度是由不同时间点的局部参数拼起来的。

这时区分三种机制就很重要。

### 1. PipeDream：局部一致，整体错位

PipeDream 的关键设计是 `weight stashing`。  
对每个 stage，它保证：

$$
\text{Forward}_i(m) \text{ 与 } \text{Backward}_i(m) \text{ 使用同一个 } W_i^{(k)}
$$

这样单个 stage 内部的链式法则是成立的。  
如果某个 stage 的局部函数记作 $h_i(\cdot; W_i)$，那么该 stage 的局部梯度仍然是在一致参数上下文中得到的：

$$
\frac{\partial h_i}{\partial W_i}\Bigg|_{W_i=W_i^{(k)}}
$$

但全模型层面，情况并不是：

$$
\nabla f(W^{(k)})
$$

而更像是：

$$
\nabla f\left(W_1^{(k_1)}, W_2^{(k_2)}, \dots, W_p^{(k_p)}\right), \quad k_1,k_2,\dots,k_p \text{ 不一定相同}
$$

这就是为什么 PipeDream 在理论语义上更接近 **异步优化**。  
它保住了每个 stage 的局部正确性，但放宽了全局同步性。

对新手来说，最容易记住的说法是：

- `stash` 解决“同一个 stage 里别串版本”
- 它不解决“同一个样本跨 stage 别串版本”

### 2. PipeDream-2BW：统一成固定 1 步延迟

PipeDream-2BW 的关键不是“消灭延迟”，而是 **把原本不规则的多级延迟，约束成统一的 1 步延迟**。

它的目标更新语义更接近：

$$
W^{(t+1)} = W^{(t)} - \eta \nabla f\left(W^{(t-1)}\right)
$$

这里所有 stage 都明确用“上一代”参数完成当前这轮 in-flight 微批次，而“下一代”参数只供下一轮新进入的微批次使用。

于是每个 stage 只需要两份参数：

- `weight_old`：当前在流水线内部仍可能被旧微批次引用
- `weight_new`：已经根据上一轮梯度更新好，但暂不立即替换所有 in-flight 计算

这就是双缓冲（double buffering）名字的来源。

它带来的直接收益有两点。

第一，版本管理从“每个 stage 维护多份历史 stash”收缩成“只维护 old/new 两代”。  
第二，跨 stage 的延迟差异从 $\tau_1,\tau_2,\dots,\tau_p$ 这种不规则形式，收敛成统一常数 1。

注意这仍然不是同步 SGD。  
因为它优化的不是 $\nabla f(W^{(t)})$，而是 $\nabla f(W^{(t-1)})$。  
只是这种偏差变得稳定、可分析、可工程化了。

### 3. PipeDream-Flush：直接恢复同步语义

Flush 的思路最直接：

1. 让一轮微批次全部进入并完成前后向
2. 在流水线排空后统一做参数更新
3. 更新完成后再开始下一轮

这时所有微批次都基于同一版参数进行本轮计算，所以重新回到：

$$
W^{(t+1)} = W^{(t)} - \eta \nabla f\left(W^{(t)}\right)
$$

因此 Flush 的训练语义最接近同步 SGD，也最容易和非流水线训练结果对齐。

代价也清楚：

- 排空期间不能继续无缝塞新微批次
- bubble 增大
- 设备利用率下降
- stage 越多，排空代价越明显

### 4. 版本差异如何变成梯度噪声

可以把版本错位理解成一种额外优化噪声。设理想同步梯度为：

$$
g^\star = \nabla f(W)
$$

实际流水线得到的是：

$$
\tilde g = \nabla f(W + \Delta W)
$$

其中 $\Delta W$ 表示由于延迟导致的参数偏移。  
若梯度满足 Lipschitz 条件：

$$
\|\nabla f(W+\Delta W)-\nabla f(W)\| \le L \|\Delta W\|
$$

则有：

$$
\|\tilde g - g^\star\| \le L\|\Delta W\|
$$

如果进一步把 $\Delta W$ 近似成延迟步数乘以单步参数漂移：

$$
\|\Delta W\| \approx \tau \cdot \|\delta w\|
$$

则梯度偏差上界可写成：

$$
\|\tilde g - g^\star\| \le L \tau \|\delta w\|
$$

这条式子非常实用，因为它告诉我们三个直接结论：

| 因子 | 变大时会怎样 | 工程含义 |
|---|---|---|
| $L$ 大 | 梯度对参数变化更敏感 | 模型处在陡峭区域时更怕延迟 |
| $\tau$ 大 | 版本错位更严重 | 深流水线、慢 stage、过多 in-flight 都会放大问题 |
| $\|\delta w\|$ 大 | 单步更新漂移更大 | 高学习率、激进优化器更危险 |

代入一个简单数值例子。若：

- 延迟步数 $\tau = 2$
- 单步权重变化量 $\|\delta w\| = 0.01$
- Lipschitz 常数近似 $L = 5$

则有：

$$
\|\tilde g - g^\star\| \le 5 \times 2 \times 0.01 = 0.1
$$

如果把这部分误差视作附加噪声，其平方量级约为：

$$
0.1^2 = 0.01
$$

这个数值本身不是结论，结论是：

**即使 2BW 已经把延迟固定到 1 步，它仍然会引入与同步训练不同的梯度噪声。**

### 5. 一个完整工程尺度的直观例子

假设一个 24 层 Transformer 被切成 4 个 stage，每个 stage 6 层；每轮放入 16 个微批次，采用 1F1B 调度。

那么：

- 在原始 PipeDream 中，靠前 stage 更早接收新微批次，但这些微批次的 backward 要很久之后才回来，所以前部 stage 通常需要保留更多历史版本。
- 在 2BW 中，调度器强制所有 in-flight 微批次只读 `old`，下一波再切到 `new`，因此参数管理明显简化。
- 在 Flush 中，16 个微批次全部完成后再做一次统一更新，所以这 16 个微批次共享同一代参数，训练语义最干净。

这一例子足以解释为什么三者的差别不是“缓存实现细节不同”，而是 **优化语义不同**。

---

## 代码实现

下面给出一个可以直接运行的最小 Python 示例。它不依赖深度学习框架，只模拟参数版本、微批次流动和更新语义，用来展示三件事：

1. `stash` 如何保证某个 stage 内 forward/backward 同版
2. `2BW` 如何只维护 `old/new` 两份权重
3. `flush` 如何在轮次边界恢复同步语义

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Stage:
    name: str
    weight: float
    stash: Dict[int, float] = field(default_factory=dict)

    def forward(self, microbatch_id: int, x: float) -> float:
        # 记录该微批次在本 stage 前向时实际看到的权重版本
        self.stash[microbatch_id] = self.weight
        return x * self.weight

    def backward(self, microbatch_id: int, grad_out: float, x: float) -> Tuple[float, float]:
        # backward 必须回放 forward 当时使用的权重
        used_weight = self.stash.pop(microbatch_id)
        grad_w = grad_out * x
        grad_x = grad_out * used_weight
        return grad_w, grad_x


def demo_weight_stashing() -> None:
    print("=== Demo 1: stage 内一致性（weight stashing）===")
    stage = Stage(name="s1", weight=2.0)

    x = 3.0
    y = stage.forward(microbatch_id=1, x=x)   # 使用 weight=2.0
    assert y == 6.0

    # 假设在 backward 回来之前，stage 当前权重已经更新
    stage.weight = 5.0

    grad_out = 1.0
    grad_w, grad_x = stage.backward(microbatch_id=1, grad_out=grad_out, x=x)

    # grad_x 必须基于 forward 时的旧权重 2.0，而不是当前权重 5.0
    assert grad_w == 3.0
    assert grad_x == 2.0
    print("forward 使用 2.0, backward 仍回放 2.0，stage 内一致性成立")


@dataclass
class TwoBWStage:
    name: str
    weight_old: float
    weight_new: float
    use_new_for_incoming: bool = False
    stash: Dict[int, float] = field(default_factory=dict)

    def current_read_weight(self) -> float:
        return self.weight_new if self.use_new_for_incoming else self.weight_old

    def forward(self, microbatch_id: int, x: float) -> float:
        w = self.current_read_weight()
        self.stash[microbatch_id] = w
        return x * w

    def backward(self, microbatch_id: int, grad_out: float, x: float) -> Tuple[float, float]:
        w = self.stash.pop(microbatch_id)
        grad_w = grad_out * x
        grad_x = grad_out * w
        return grad_w, grad_x

    def apply_update(self, lr: float, grad_w: float) -> None:
        # 只更新 new；old 仍保留给在途微批次使用
        self.weight_new = self.weight_old - lr * grad_w

    def commit_new_wave(self) -> None:
        # 下一波微批次统一切到新版本
        self.weight_old = self.weight_new
        self.use_new_for_incoming = False


def demo_2bw() -> None:
    print("\n=== Demo 2: 2BW 双缓冲 ===")
    stage = TwoBWStage(name="s1", weight_old=1.0, weight_new=1.0)

    # 第一波微批次读取 old
    y1 = stage.forward(microbatch_id=1, x=2.0)
    grad_w1, _ = stage.backward(microbatch_id=1, grad_out=1.0, x=2.0)
    stage.apply_update(lr=0.1, grad_w=grad_w1)

    assert abs(y1 - 2.0) < 1e-9
    assert abs(stage.weight_old - 1.0) < 1e-9
    assert abs(stage.weight_new - 0.8) < 1e-9

    # 切到下一波
    stage.commit_new_wave()

    y2 = stage.forward(microbatch_id=2, x=2.0)
    grad_w2, _ = stage.backward(microbatch_id=2, grad_out=1.0, x=2.0)

    assert abs(y2 - 1.6) < 1e-9
    assert abs(grad_w2 - 2.0) < 1e-9
    print("old/new 两份权重足够表达固定 1 步延迟")


def flush_round(weight: float, microbatches: List[float], lr: float = 0.1) -> float:
    # 一轮内所有微批次都共享同一个 weight
    grads = []
    for x in microbatches:
        # toy loss: y = w*x, loss = y，故 dloss/dw = x
        grads.append(x)

    mean_grad = sum(grads) / len(grads)
    return weight - lr * mean_grad


def demo_flush() -> None:
    print("\n=== Demo 3: Flush 恢复同步语义 ===")
    weight = 1.0
    microbatches = [1.0, 2.0, 3.0, 4.0]

    new_weight = flush_round(weight, microbatches, lr=0.1)
    expected = 1.0 - 0.1 * ((1.0 + 2.0 + 3.0 + 4.0) / 4.0)

    assert abs(new_weight - expected) < 1e-9
    print(f"本轮所有微批次统一使用 weight={weight:.1f}，轮末一次更新到 {new_weight:.3f}")


def main() -> None:
    demo_weight_stashing()
    demo_2bw()
    demo_flush()


if __name__ == "__main__":
    main()
```

运行输出应类似于：

```text
=== Demo 1: stage 内一致性（weight stashing）===
forward 使用 2.0, backward 仍回放 2.0，stage 内一致性成立

=== Demo 2: 2BW 双缓冲 ===
old/new 两份权重足够表达固定 1 步延迟

=== Demo 3: Flush 恢复同步语义 ===
本轮所有微批次统一使用 weight=1.0，轮末一次更新到 0.750
```

上面这段代码是“机制最小化模型”，故意省略了真实训练中的激活缓存、跨 stage 通信、梯度累积和优化器状态。  
它的作用不是模拟真实吞吐，而是把三个概念拆开：

| 代码对象 | 对应真实语义 |
|---|---|
| `stash[microbatch_id]` | 记录某微批次在该 stage 的 forward 权重版本 |
| `weight_old` / `weight_new` | 2BW 的双缓冲参数 |
| `flush_round` | 一轮结束统一更新的同步语义 |

### 玩具例子：三阶段时间线

继续看一个对新手更友好的时间线。设有 3 个 stage，微批次 `m5` 进入时发生如下状态：

| 时刻 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| 某时刻 | 正在接收 `m5` | 还在处理更早的 `m3` | 正在对 `m1` 做 backward |

此时可能出现：

- stage 1 已切到 $W_1^{(1)}$
- stage 2 还保留 $W_2^{(0)}$
- stage 3 正在回放更早版本的 stash

那么 `m5` 的经历可能是：

1. 在 stage 1 forward 时使用 $W_1^{(1)}$
2. 传播到 stage 2 时，stage 2 仍对它应用 $W_2^{(0)}$
3. 很久以后回到 stage 1 backward 时，stage 1 再通过 stash 回放 $W_1^{(1)}$

这里有两件事同时成立：

- `m5` 在 stage 1 内部是自洽的，因为 forward/backward 都是 $W_1^{(1)}$
- 但 `m5` 穿过整个模型时并没有看到统一参数集，因为它在 stage 2 看到的是旧版 $W_2^{(0)}$

这正是“stage 内一致，但跨 stage 不一致”的精确定义。

### 真实工程例子：Transformer 四段流水

以大规模 Transformer 训练为例，假设模型被切成 4 段，每段部署在不同 GPU 组上，每轮输入 32 个微批次。

三类方案的工程表现大致如下：

| 方案 | 参数版本管理 | 调度复杂度 | 训练语义 | 典型场景 |
|---|---|---:|---|---|
| PipeDream | 多份历史版本 | 高 | 异步延迟梯度 | 强调吞吐、可接受更复杂调度 |
| 2BW | 两份版本 | 中 | 固定 1 步延迟 | 工程上常见折中 |
| Flush | 一轮一更新 | 低 | 近同步 SGD | 基线验证、训练初期求稳 |

实践里，很多团队最终把 2BW 作为默认实现，不是因为它“最理论正确”，而是因为它在以下三点上平衡得最好：

- 参数管理稳定
- 调度器实现难度可控
- 吞吐与收敛之间折中合理

---

## 工程权衡与常见坑

工程上最常见的错误，不是流水线写不出来，而是把几个不同层次的问题混为一谈：  
有些问题属于计算图正确性，有些问题属于优化延迟，有些问题属于资源利用率。它们的表现都可能是“loss 不稳定”，但根因完全不同。

先看一张总表。

| 问题 | 直接后果 | 常见表现 | 处理方式 |
|---|---|---|---|
| 没有 stashing | 同一 stage 前后向跳版本 | loss 异常抖动，梯度不可信 | 必须缓存前向版本 |
| 有 stashing 但无统一延迟 | 跨 stage 梯度不是真梯度 | 能训练但收敛变慢 | 用 2BW 或 flush |
| 学习率照搬同步训练 | 延迟噪声被放大 | 后期震荡、偶发发散 | 降低 lr，必要时增大 warmup |
| 微批次数过小 | bubble 比例高 | GPU 利用率差 | 增加 microbatch 数 |
| 微批次数过大 | 激活与版本缓存膨胀 | OOM、调度拥塞 | 重新平衡 microbatch 与 stage 切分 |
| stage 切分不均衡 | 有效延迟被慢 stage 拉长 | 尾部堆积，吞吐下降 | 重做层切分或异构并行映射 |

下面把几个坑拆开说。

### 1. 忽略 stashing 是原则性错误

这一点不是“效果可能差一点”，而是 **数学对象变了**。  
Backward 必须对应它自己的 forward 图。如果 forward 用的是 $W^{(0)}$，backward 却在 $W^{(1)}$ 上求导，那么该梯度并不对应原来那条样本路径。

换句话说，没有 stashing 时，问题甚至不只是“异步”，而是“局部梯度定义已经错了”。

### 2. 2BW 不是免费同步

2BW 常被误解成“只差一点点的同步”。更准确的说法是：

**它把不可控延迟变成可控延迟，但没有消除延迟。**

这点对带状态的优化器尤其重要。  
以 Adam 为例，更新不仅依赖当前梯度，还依赖一阶、二阶矩估计：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

如果 $g_t$ 本身来自延迟参数 $W^{(t-1)}$，那么参数、梯度和优化器状态之间会形成额外错位。  
这也是为什么同样一套超参数，在同步训练里稳定，在 2BW 下却可能需要：

- 更低学习率
- 更长 warmup
- 更强梯度裁剪
- 更频繁的 flush 边界

### 3. Flush 不只是吞吐下降

Flush 的代价不只是平均利用率变低，它还会改变设备利用时间分布。  
在排空阶段，前部 stage 往往更早结束并开始空转，后部 stage 则还在处理尾部 backward。stage 越多，这种“阶梯式空闲”越明显。

如果流水线深度是 $p$，每轮微批次数是 $m$，一个常见近似下，bubble 比例可写成：

$$
\text{bubble fraction} \approx \frac{p-1}{m+p-1}
$$

这条近似有两个直接启发：

- 固定 stage 数时，增大微批次数可降低 bubble 占比
- 固定微批次数时，stage 越多，flush 代价越高

所以 Flush 并不是“肯定慢很多”，而是要结合 $p$ 与 $m$ 一起看。  
在浅流水线、较大微批次数下，Flush 的代价可能是可接受的。

### 4. stage 切分不均衡会放大所有问题

如果某个 stage 明显更慢，问题不只是吞吐下降。更重要的是：

- 前面的 stage 会积压输出
- 后面的 stage 会等待输入
- in-flight 微批次滞留时间增加
- 有效延迟 $\tau$ 变大

也就是说，stage 切分不均衡会同时恶化：

- bubble
- 显存占用
- 参数版本滞留
- 梯度延迟噪声

很多看起来像“权重同步出错”的问题，根因其实是 stage 切分本身不合理。

### 5. 初学者最容易混淆的三个层面

| 层面 | 问题本质 | 典型修复 |
|---|---|---|
| 计算图层面 | backward 是否回到正确 forward 图 | stashing |
| 优化语义层面 | 整体梯度是否对应统一参数快照 | 2BW / flush |
| 系统效率层面 | GPU 是否被流水线充分填满 | 调整微批次、切分、调度 |

如果不把这三层分开，排查训练异常时就会反复误判。

---

## 替代方案与适用边界

把现有方案按训练语义分成三类，会更容易判断适用边界。

| 类别 | 代表方法 | 核心思想 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|---|
| 同步型 | GPipe, Flush | 一轮结束后统一更新 | 梯度语义最干净 | bubble 大 | 稳定性优先、基线验证 |
| 异步型 | PipeDream, 2BW | 允许延迟梯度 | 吞吐高 | 版本错位带噪声 | 吞吐优先、可接受延迟 |
| 预测型 | PipeMare, SpecTrain | 预测旧权重或未来权重 | 可减少显式缓存 | 实现复杂，假设更强 | 研究场景、强内存约束 |

### 1. 同步型：GPipe / Flush

同步型方案最容易理解：  
同一轮微批次全部基于同一版参数完成前后向，轮末统一更新。

优点是语义简单：

- 和普通数据并行训练最接近
- 容易验证正确性
- 便于对比 baseline

缺点是系统效率不一定最好：

- 流水线深时 bubble 明显
- 每轮边界存在排空成本

对初学者来说，GPipe 或 Flush 通常是最好的第一基线。  
因为你先把“训练语义正确”这件事固定住，再去追吞吐，排查会容易很多。

### 2. 异步型：PipeDream / 2BW

异步型方案承认一个事实：  
为了吞吐，允许梯度相对于当前参数有延迟。

两者差别在于延迟是否规则。

| 方法 | 延迟形态 | 工程特征 |
|---|---|---|
| PipeDream | 各 stage 延迟可不同 | 版本缓存更多，调度更复杂 |
| 2BW | 全流水线固定 1 步延迟 | 双缓冲实现更稳定 |

工程上，2BW 是更常见的折中点。  
原因不是它绝对最好，而是它把问题收缩到了一个比较可控的范围：

- 延迟固定
- 权重两份即可
- 训练调参经验更容易迁移

### 3. 预测型：PipeMare / SpecTrain 一类

预测型方法尝试进一步减少历史版本依赖。它的思路是：

- 不显式保存太多旧参数
- 而是对旧权重或未来权重进行近似估计

一个常见近似写法是：

$$
\nabla f(W^{(t-\tau)}) \approx \nabla f(W^{(t)}) + H \cdot (W^{(t-\tau)} - W^{(t)})
$$

其中 $H$ 是 Hessian 的局部近似。  
这相当于用局部二阶信息估计延迟梯度与当前梯度之间的偏差。

这类方法的难点在于：

- 曲率近似本身不稳定
- 对学习率与局部光滑性更敏感
- 实现与调试复杂度显著升高

因此它通常不是默认工程方案，更适合研究性尝试，或者显存特别紧张、团队又能承担复杂调试成本的场景。

### 4. 一个实用选择准则

实际项目里，可以用下面这个判断顺序：

1. 如果训练刚起步，经常发散，先用 `flush` 建立稳定基线。
2. 如果训练已经稳定，瓶颈主要在吞吐，优先试 `2BW`。
3. 如果显存极紧、历史权重版本成本过高，再评估预测型方法。
4. 如果问题表现像“理论上能收敛但工程上难维护”，通常 PipeDream 原始多版本实现不是首选。

### 5. 一个更直接的决策表

| 你的首要目标 | 优先方案 | 原因 |
|---|---|---|
| 先确保结果对 | Flush / GPipe | 语义最清晰 |
| 先把吞吐做上去 | 2BW | 延迟可控，工程折中最好 |
| 极限节省版本缓存 | 预测型 | 但复杂度最高 |
| 做论文复现 | 对齐原方法 | 因为不同语义不能直接比较 |

---

## 参考资料

- Narayanan et al. *PipeDream: Generalized Pipeline Parallelism for DNN Training*. 介绍 1F1B 调度、weight stashing 与异步流水线训练语义。https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf
- Deepak Narayanan. *Memory-Efficient Pipeline-Parallel DNN Training*（博士论文）. 系统说明 PipeDream、PipeDream-2BW、PipeDream-Flush 的更新公式、双缓冲设计与调度分析。https://cs.stanford.edu/~deepakn/assets/papers/thesis.pdf
- Narayanan et al. *Memory-Efficient Pipeline-Parallel DNN Training*. 给出 PipeDream-2BW 的正式算法、内存分析与实验结果。https://people.eecs.berkeley.edu/~matei/papers/2021/icml_pipedream_2bw.pdf
- Harlap et al. *PipeMare: Asynchronous Pipeline Parallel DNN Training*. 讨论如何通过参数预测减轻流水线版本缓存问题。适合理解“预测型”路线。https://arxiv.org/abs/2106.10414
- Chen et al. *SpecTrain: A Speculative and Error-Bounded Approach to Pipeline Parallelism*. 代表性的预测/估计类流水线方法，可对照理解显式 stash 与近似修正的差别。https://arxiv.org/abs/1905.10936
- Selam G. *The Frontier of Pipeline Parallelism*. 对 delay discrepancy、梯度噪声和学习率约束做了直观总结，适合作为阅读论文前的辅助材料。https://selamjie.medium.com/the-frontier-of-pipeline-parallelism-an-overview-d4264cc9f877
- PipeDream / 2BW / Flush 的中文综述与实现解读，可辅助理解时间线、调度细节与工程取舍。https://www.mo4tech.com/deep-learning-pipeline-parallel-pipedream6-1f1b-strategy.html
