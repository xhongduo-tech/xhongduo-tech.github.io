## 核心结论

Prefetching 是“预取”：在当前 batch 正在被 GPU 计算时，提前准备后续 batch。它的目标不是让单次数据读取本身变快，而是把数据加载、CPU 预处理、batch 拼接、H2D 拷贝与 GPU 计算重叠起来。H2D 是 Host to Device，指数据从主机内存拷贝到 GPU 显存。

训练单步时间可以近似写成：

$$
T_{step} \approx \max(T_{compute}, T_{data})
$$

其中：

$$
T_{data}=T_{load\_cpu}+T_{collate}+T_{h2d}
$$

`T_compute` 是 GPU 对当前 batch 做前向、反向、优化器更新的时间。`T_data` 是下一个 batch 从读取、处理、拼接到送入 GPU 的总时间。理想状态下，只要 $T_{data} \le T_{compute}$，数据路径就能被计算路径遮住，GPU 基本不用等数据。

玩具例子：训练一个小 CNN。GPU 计算当前 batch 需要 `120ms`，CPU 准备下一个 batch 加 H2D 拷贝需要 `30ms`，那么单步时间接近 `120ms`。如果数据准备变成 `150ms`，GPU 算完后还要等数据，单步时间接近 `150ms`。

| 情况 | `T_compute` | `T_data` | 近似 `T_step` | 结论 |
|---|---:|---:|---:|---|
| 数据被完全隐藏 | 120ms | 30ms | 120ms | GPU 不明显等数据 |
| 数据接近临界 | 120ms | 110ms | 120ms | 仍可接受，但余量小 |
| 数据成为瓶颈 | 120ms | 150ms | 150ms | 输入管线拖慢训练 |

---

## 问题定义与边界

Prefetching 解决的是训练中的“输入侧等待问题”。输入侧指模型计算之前的所有数据路径，包括从磁盘或网络读取样本、CPU 解码、数据增强、`collate_fn` 拼 batch、把 batch 从 CPU 拷到 GPU。

它不等同于模型并行、数据并行、梯度累积或增大 batch size。模型并行解决模型太大或计算分布问题；梯度累积解决显存不足时模拟大 batch 的问题；prefetching 解决的是 GPU 算得出来，但下一批数据没准备好的问题。

一个新手可以把训练理解成“GPU 做题，CPU 发题”。GPU 做当前题时，CPU 提前把下一题打印好、整理好、放到桌上。只要下一题准备得比 GPU 做题更快，GPU 就不会停下来等。

| 符号 | 含义 | 常见来源 |
|---|---|---|
| `T_compute` | GPU 计算当前 batch 的时间 | 前向、反向、优化器步骤 |
| `T_load_cpu` | CPU 侧读取和预处理样本的时间 | 磁盘读取、解码、数据增强 |
| `T_collate` | 把多个样本合成 batch 的时间 | `collate_fn`、padding、stack |
| `T_h2d` | CPU 到 GPU 的拷贝时间 | `.to("cuda")`、`.cuda()` |

在 PyTorch `DataLoader` 中，`num_workers` 控制加载数据的 worker 进程数，`prefetch_factor` 控制每个 worker 预先加载多少个 batch。自动 batch 模式下，在途 batch 的理论上限近似为：

$$
N_{inflight}=num\_workers \times prefetch\_factor
$$

`num_workers=0` 时，数据加载主要发生在主进程中，基本没有真正的多进程预取；此时显式设置 `prefetch_factor` 也不符合 PyTorch 的参数约束。

---

## 核心机制与推导

PyTorch 的 `DataLoader` 在 `num_workers > 0` 时会启动多个 worker 进程。每个 worker 从 dataset 中取样本，执行必要的数据读取和变换，然后把结果交给主进程。主进程再按顺序或配置要求取出 batch，必要时把 batch 放入 pinned memory。

Pinned memory 是页锁定内存，白话说就是操作系统不会把这块 CPU 内存换出到磁盘，因此 GPU 可以更稳定、更高效地从这里发起拷贝。它是异步 H2D 拷贝的重要条件之一，但不是全部条件。

设 `num_workers=4`，`prefetch_factor=2`，则最多大约有：

$$
N_{inflight}=4 \times 2=8
$$

个 batch 正在被提前准备或已经在队列中等待消费。注意，`prefetch_factor` 影响“提前排队多少个 batch”，不直接缩短某个 batch 的 CPU 解码时间，也不缩短模型计算时间。

| `T_compute` | `T_data` | `T_data / T_compute` | 近似 `T_step` | 判断 |
|---:|---:|---:|---:|---|
| 120ms | 30ms | 0.25 | 120ms | 数据路径被隐藏 |
| 120ms | 90ms | 0.75 | 120ms | 预取有效 |
| 120ms | 120ms | 1.00 | 120ms | 临界状态 |
| 120ms | 150ms | 1.25 | 150ms | 输入侧瓶颈 |
| 120ms | 300ms | 2.50 | 300ms | 单靠预取不够 |

真实工程例子：图像分类训练中，每个样本是 JPEG 图片。CPU 需要解码、随机裁剪、颜色扰动、归一化，然后 `DataLoader` 把多个样本拼成 batch。若模型是较重的 ResNet 或 ViT，反向传播本身较久，数据准备和 H2D 通常有机会被覆盖。若模型很小但图像巨大、增强很重，GPU 很快算完，输入侧就会暴露成瓶颈。

---

## 代码实现

实际使用时，通常同时配置 `num_workers`、`prefetch_factor`、`pin_memory=True`，并在送入 GPU 时使用 `non_blocking=True`。`non_blocking=True` 的白话解释是：发起拷贝后，主机线程不强制等拷贝完成才继续往下走；它是否真的有效，还取决于源数据是否在 pinned memory、设备和后续同步点。

先看一个可运行的纯 Python 玩具模拟，用来验证重叠模型：

```python
def step_time_ms(t_compute, t_data):
    return max(t_compute, t_data)

def inflight_batches(num_workers, prefetch_factor):
    return num_workers * prefetch_factor

assert step_time_ms(120, 30) == 120
assert step_time_ms(120, 150) == 150
assert inflight_batches(4, 2) == 8

hidden = step_time_ms(120, 30)
bottleneck = step_time_ms(120, 150)

print({"hidden_ms": hidden, "bottleneck_ms": bottleneck})
```

PyTorch 训练循环通常写成下面这样：

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
)

for batch, target in loader:
    batch = batch.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
```

| 参数或写法 | 作用 | 是否影响预取 | 是否影响异步 H2D |
|---|---|---:|---:|
| `num_workers=4` | 启动 4 个 worker 进程加载数据 | 是 | 否 |
| `prefetch_factor=2` | 每个 worker 预先准备 2 个 batch | 是 | 否 |
| `pin_memory=True` | 尽量把 batch 放到页锁定内存 | 间接 | 是 |
| `non_blocking=True` | 发起非阻塞设备拷贝 | 否 | 是 |
| `collate_fn` | 控制样本如何拼成 batch | 可能 | 否 |

---

## 工程权衡与常见坑

`num_workers` 和 `prefetch_factor` 不是越大越好。它们会增加主机内存占用，因为更多 batch 会同时留在队列中；也会增加进程调度、序列化、反序列化和 CPU 缓存压力。数据增强很重时，增加 worker 可能有效；样本对象很大、Python 处理很多时，增加 worker 也可能把瓶颈转移到内存带宽或进程通信。

常见误区是把 `pin_memory=True` 当成“已经异步”。更准确的说法是：`pin_memory=True` 让 batch 更适合作为异步 H2D 的源内存；真正送入 GPU 时仍要使用 `non_blocking=True`，并且后续代码不能立刻用同步操作把流水线打断。

| 常见坑 | 表现 | 修复手段 |
|---|---|---|
| `num_workers=0` | GPU 经常等下一个 batch | 增加 `num_workers`，从 2、4、8 逐步测 |
| `pin_memory=False` | H2D 拷贝难以重叠 | CUDA 训练时尝试开启 `pin_memory=True` |
| 缺少 `non_blocking=True` | 拷贝更容易阻塞主机线程 | `.to(device, non_blocking=True)` |
| 自定义 batch 不支持 pin | `pin_memory=True` 效果不明显 | 给自定义 batch 类型实现 `pin_memory()` |
| worker 过多 | 内存上涨、CPU 抖动、吞吐下降 | 降低 worker 或 prefetch 深度 |
| 增强过重 | GPU 利用率仍然不稳 | 简化增强、离线预处理、缓存样本 |

瓶颈定位时要把 `T_data` 拆开看：

$$
T_{data}=T_{load\_cpu}+T_{collate}+T_{h2d}
$$

真实训练日志中，输入管线问题通常重点看等待 batch、CPU 处理和 H2D 时间。D2H 是 Device to Host，指 GPU 到 CPU 的拷贝；如果 profiler 里 D2H 很高，常见原因反而是日志、指标计算、`.item()`、调试输出或把 GPU tensor 拷回 CPU，不是 DataLoader 预取本身。判断输入瓶颈时，应优先确认 H2D 和取下一个 batch 的等待时间。

---

## 替代方案与适用边界

Prefetching 适合“计算较重、输入次之”的训练。也就是数据准备能在一个 step 内追上计算，或只略慢一点。它不适合“输入远重于计算”的任务；当 $T_{data} \gg T_{compute}$ 时，即使提前排了很多 batch，长期吞吐仍然近似受数据路径支配：

$$
T_{step} \approx T_{data}
$$

如果训练样本来自大图像文件，每次都要 CPU 解码、随机裁剪、复杂增强和归一化，那么继续把 `num_workers` 从 16 加到 32 未必有效。更直接的办法是把部分预处理离线化，或者把常用数据缓存到本地 SSD、内存缓存、LMDB、WebDataset shard 等更适合顺序读取的格式中。

| 方案 | 解决的问题 | 代价 | 适用边界 |
|---|---|---|---|
| `DataLoader` prefetch | 遮住可并行的数据准备时间 | 占用更多内存和 worker | `T_data` 不明显大于 `T_compute` |
| 离线预处理 | 减少训练时 CPU 解码和变换 | 增加预处理流程和存储 | 变换稳定、可提前固化 |
| 样本缓存 | 降低重复读取成本 | 占用内存或高速存储 | 数据集可重复访问 |
| 减少增强复杂度 | 直接降低 `T_load_cpu` | 可能影响泛化 | 增强成本高于收益 |
| 更快存储介质 | 降低 I/O 等待 | 增加硬件成本 | 瓶颈确实在读取层 |
| GPU 侧增强 | 把部分处理移到 GPU | 增加显存和实现复杂度 | CPU 增强成为主要瓶颈 |

工程上应先测量，再调参。一个稳妥顺序是：先记录单步时间和 GPU 利用率；再从 `num_workers=0`、`2`、`4`、`8` 做小范围扫描；然后打开 `pin_memory=True` 和 `non_blocking=True`；最后再考虑离线预处理、缓存或数据格式改造。

---

## 参考资料

| 资料类型 | 可回答的问题 | 适合引用的位置 |
|---|---|---|
| 官方文档 | 参数语义、默认值、约束 | `DataLoader` 配置说明 |
| 源码 | worker 队列、预取调度细节 | 核心机制分析 |
| 官方教程 | pinned memory 与 non-blocking 的使用边界 | H2D 优化与常见坑 |
| API 文档 | `.cuda()` / `.to()` 参数行为 | 代码实现说明 |

1. [PyTorch DataLoader 官方文档](https://docs.pytorch.org/docs/stable/data.html)
2. [PyTorch dataloader.py 源码](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py)
3. [PyTorch pinned memory / non_blocking 教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html?highlight=dataloader)
4. [PyTorch Tensor.cuda 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cuda.html)
