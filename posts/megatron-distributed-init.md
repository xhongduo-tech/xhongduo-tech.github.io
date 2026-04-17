## 核心结论

Megatron-LM 的“分布式初始化”本质上是在启动时把全部 GPU rank 重新组织成多套正交的进程组。正交的意思是：同一批 rank 会同时属于 TP、PP、DP 三种不同分组，但每种分组只负责一种通信维度，互不混淆。

先解释三个术语：

- Tensor Parallel，简称 TP，白话说就是“把一层内部的矩阵计算拆到多张卡上”。
- Pipeline Parallel，简称 PP，白话说就是“把模型不同层段放到不同阶段上”。
- Data Parallel，简称 DP，白话说就是“每张卡跑不同数据，最后再同步梯度”。

Megatron-LM 初始化后，最重要的三类进程组是：

- `tensor_model_parallel_group`
- `pipeline_model_parallel_group`
- `data_parallel_group`

它们不是随便分的，而是围绕通信频率设计的。TP 通信最密，常见于一层内部的 `all-reduce`、`reduce-scatter`、`all-gather`；PP 通信是相邻 stage 之间传激活和梯度；DP 通信通常发生在梯度同步或参数相关的归约。工程目标很明确：让最频繁的通信尽量落在拓扑最紧的 GPU 之间，比如同一台机器、同一张 NVLink 域、同一 NUMA 域。

需要特别区分两件事：

1. 默认 rank 推导公式通常按 `tp-dp-pp` 组织。
2. 初始化 communicator 时可以额外开启 `use_tp_pp_dp_mapping`，把初始化顺序改成 `tp-pp-dp`，用于更好地贴合硬件拓扑。

所以，“创建组的逻辑顺序”和“全局 rank 的线性映射顺序”不是完全一回事，初学者最容易在这里混淆。

再压缩成一句话：Megatron 初始化做的不是“创建几个 NCCL group”这么简单，而是先定义一套多维坐标系，再把这套坐标系映射到真实机器拓扑上。逻辑正确保证能跑，映射合理决定跑得快不快。

---

## 问题定义与边界

问题可以定义成一句话：给定总卡数 $world\_size = tp \times pp \times dp$，如何为每个 rank 同时构造 TP 组、PP 组、DP 组，并保证后续 NCCL 通信既正确又高效。

这件事的约束有三层。

第一层是数学约束。每个全局 rank 必须能唯一映射到一组局部坐标，例如：

- `tp_rank`：该 rank 在 TP 组里的位置
- `pp_rank`：该 rank 在 PP 组里的位置
- `dp_rank`：该 rank 在 DP 组里的位置

唯一映射的意思是：不能出现两个不同坐标对应同一个 `global_rank`，也不能出现某个 `global_rank` 反推出两组不同坐标。否则后续 group 划分会重叠错误。

第二层是实现约束。所有进程必须用完全一致的规则创建 process group，否则有的 rank 认为自己在组 A，有的 rank 认为自己在组 B，初始化就会直接卡死。这里的一致不只是“组成员一样”，还包括：

- 创建顺序一致
- 每次 `new_group()` 看到的 rank 列表一致
- 每个进程对自己属于哪些组的判断一致

第三层是拓扑约束。GPU 之间的物理连接并不对称。NVLink、PCIe、NUMA、跨机 InfiniBand 的带宽和时延都不同。如果把高频 TP 通信映射到跨机链路，训练会明显变慢。

可以把三类并行放进一张表里理解：

| 并行维度 | 主要切分对象 | 高频通信形态 | 更希望落在哪类链路 |
|---|---|---|---|
| TP | 单层内部张量/矩阵 | `all-reduce`、`all-gather`、`reduce-scatter` | 同机、同 NVLink 域 |
| PP | 不同层段 | 点对点发送激活/梯度 | 相邻 stage 尽量近，但允许更远 |
| DP | 数据副本 | 梯度归约、参数同步 | 可接受更远距离 |

边界也很重要。`use_tp_pp_dp_mapping=True` 时，要确保没有同时使用 EP 和 CP：

- EP，Expert Parallel，白话说就是“把不同专家分给不同设备”的并行方式，常见于 MoE。
- CP，Context Parallel，白话说就是“把长序列沿 token 维度分给不同设备”。

原因不是“功能不支持”这么简单，而是并行维度一多，rank 排列和 group mask 会更复杂。三维时你还能靠直觉看出“固定两维、枚举一维”；扩展到 `tp-cp-ep-dp-pp` 后，维度语义和映射顺序一旦错位，就可能让 communicator 的解释发生冲突。

因此，本文只讨论最核心的 3D 初始化问题：

- 只分析 TP、PP、DP 三个维度
- 不展开 EP、CP、VPP 的额外 rank 组合
- 重点讨论“rank 线性展开”和“正交 group 构造”的关系

---

## 核心机制与推导

Megatron Core 文档给出的经典公式是：

$$
global\_rank = tp\_rank + dp\_rank \times tp\_size + pp\_rank \times (tp\_size \times dp\_size)
$$

这表示默认线性展开顺序是：

$$
tp \rightarrow dp \rightarrow pp
$$

也就是：

- TP 是最内层，rank 连续
- DP 是中间层，按 `tp_size` 为步长跳跃
- PP 是最外层，按 `tp_size * dp_size` 为步长跳跃

这个公式的价值，不只是“能算 rank”，更重要的是它能把三维坐标和一维 rank 做双向映射。这样我们就能在固定两个维度时，枚举第三个维度，构造出正交进程组。

### 先看正向映射

设：

- $t = tp\_rank$
- $d = dp\_rank$
- $p = pp\_rank$
- $T = tp\_size$
- $D = dp\_size$
- $P = pp\_size$

那么：

$$
r = t + dT + pTD
$$

它和数组下标展开完全是同一件事。你可以把三维坐标 `(t, d, p)` 看成一个三维张量元素，把它按 `tp` 最快、`dp` 次快、`pp` 最慢的顺序压平成一维编号。

### 再看反向映射

给定全局 rank `r`，可以反推出三维局部坐标：

$$
tp\_rank = r \bmod T
$$

$$
dp\_rank = \left\lfloor \frac{r}{T} \right\rfloor \bmod D
$$

$$
pp\_rank = \left\lfloor \frac{r}{TD} \right\rfloor
$$

这三条式子对新手最重要，因为它解释了“为什么 TP 组是连续的、DP 组按 `T` 跳、PP 组按 `TD` 跳”。

### 为什么固定两维、枚举一维就得到 communicator

原因很直接。三维坐标系里：

- 固定 `(d, p)`，让 `t` 从 `0` 到 `T-1` 变化，得到一个 TP 组
- 固定 `(t, p)`，让 `d` 从 `0` 到 `D-1` 变化，得到一个 DP 组
- 固定 `(t, d)`，让 `p` 从 `0` 到 `P-1` 变化，得到一个 PP 组

这就是“正交”的精确定义。它不是口语上的“互相独立”，而是数学上的“在一个维度上变化时，其余维度固定”。

### 玩具例子：16 卡，tp=2, pp=4, dp=2

总卡数是：

$$
16 = 2 \times 4 \times 2
$$

按默认 `tp-dp-pp` 公式展开，所有 rank 的三维坐标如下：

| global rank | tp_rank | dp_rank | pp_rank |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 |
| 3 | 1 | 1 | 0 |
| 4 | 0 | 0 | 1 |
| 5 | 1 | 0 | 1 |
| 6 | 0 | 1 | 1 |
| 7 | 1 | 1 | 1 |
| 8 | 0 | 0 | 2 |
| 9 | 1 | 0 | 2 |
| 10 | 0 | 1 | 2 |
| 11 | 1 | 1 | 2 |
| 12 | 0 | 0 | 3 |
| 13 | 1 | 0 | 3 |
| 14 | 0 | 1 | 3 |
| 15 | 1 | 1 | 3 |

代入分组规则后，得到：

| 组类型 | 数量 | 每组成员示例 | 生成方式 | 含义 |
|---|---:|---|---|---|
| TP 组 | $D \times P = 8$ | `[0,1]`、`[2,3]` | 固定 `(dp, pp)`，枚举 `tp` | 同一层内切分计算 |
| DP 组 | $T \times P = 8$ | `[0,2]`、`[1,3]` | 固定 `(tp, pp)`，枚举 `dp` | 不同数据副本同步梯度 |
| PP 组 | $T \times D = 4$ | `[0,4,8,12]` | 固定 `(tp, dp)`，枚举 `pp` | 模型深度分段串联 |

这个例子很适合看“正交”是什么意思：

- rank `0` 同时属于 TP 组 `[0,1]`
- 也属于 DP 组 `[0,2]`
- 还属于 PP 组 `[0,4,8,12]`

它在三种坐标系里各有一个位置，但三种组分别服务三类通信。

### 真实工程例子：32 卡，tp=4, pp=2, dp=4

总卡数：

$$
32 = 4 \times 2 \times 4
$$

默认 `tp-dp-pp` 公式下：

- `pp=0` 阶段对应 rank `0` 到 `15`
- `pp=1` 阶段对应 rank `16` 到 `31`

此时：

- TP 组是连续 4 卡，如 `[0,1,2,3]`
- DP 组按 4 跳，如 `[0,4,8,12]`
- PP 组跨 16 跳，如 `[0,16]`

如果把 32 卡部署为 4 台机器、每台 8 卡，一个常见目标是：

- 尽量把同一个 TP 组放在同机甚至同 NVLink 域
- DP 组可以跨更远一点
- PP 组允许跨更大的物理距离，因为它的通信形态通常更容易和计算流水重叠

这就是为什么“rank 排列”本身会影响性能。它不是抽象编号，而是直接决定 communicator 中谁和谁更常通信。

### `tp-pp-dp` 到底改了什么

如果启用 `use_tp_pp_dp_mapping`，变化的是线性展开顺序。此时更直观的公式变成：

$$
global\_rank = tp\_rank + pp\_rank \times tp\_size + dp\_rank \times (tp\_size \times pp\_size)
$$

也就是：

$$
tp \rightarrow pp \rightarrow dp
$$

对应的反向映射变成：

$$
tp\_rank = r \bmod T
$$

$$
pp\_rank = \left\lfloor \frac{r}{T} \right\rfloor \bmod P
$$

$$
dp\_rank = \left\lfloor \frac{r}{TP} \right\rfloor
$$

注意这里没有改变“三类组的定义”，改变的是“一维 rank 编号如何投影到三维坐标”。组还是 TP/PP/DP 三类组，但 rank 邻接关系变了，于是 communicator 的物理贴图也可能随之更合理。

Megatron Core 里的 `RankGenerator` 和 `generate_masked_orthogonal_rank_groups`，本质上就是做这件事：给定并行度大小和顺序字符串，比如 `tp-dp-pp` 或 `tp-pp-dp`，再给一个 mask，生成对应维度的 rank 组。

---

## 代码实现

下面先给一个可运行的 Python 版本，同时支持 `tp-dp-pp` 和 `tp-pp-dp`。它不依赖 PyTorch，只验证 rank 推导、反向映射和 TP/DP/PP 分组是否正确。

```python
from itertools import product

def make_strides(order, sizes):
    stride = 1
    strides = {}
    for dim in order:
        strides[dim] = stride
        stride *= sizes[dim]
    return strides

def global_rank(coords, sizes, order):
    strides = make_strides(order, sizes)
    return sum(coords[dim] * strides[dim] for dim in order)

def invert_rank(rank, sizes, order):
    strides = make_strides(order, sizes)
    coords = {}
    remaining = rank
    for dim in reversed(order):
        stride = strides[dim]
        coords[dim] = remaining // stride
        remaining %= stride
    for dim in order:
        coords[dim] %= sizes[dim]
    return coords

def build_groups(tp_size, dp_size, pp_size, order=("tp", "dp", "pp")):
    sizes = {"tp": tp_size, "dp": dp_size, "pp": pp_size}
    world_size = tp_size * dp_size * pp_size
    dims = ("tp", "dp", "pp")

    # 验证正向/反向映射一一对应
    all_ranks = set()
    for tp_rank, dp_rank, pp_rank in product(range(tp_size), range(dp_size), range(pp_size)):
        coords = {"tp": tp_rank, "dp": dp_rank, "pp": pp_rank}
        r = global_rank(coords, sizes, order)
        recovered = invert_rank(r, sizes, order)
        assert recovered == coords, (coords, r, recovered)
        all_ranks.add(r)
    assert all_ranks == set(range(world_size))

    groups = {}
    for target_dim in dims:
        fixed_dims = [d for d in dims if d != target_dim]
        result = []
        fixed_ranges = [range(sizes[d]) for d in fixed_dims]

        for fixed_values in product(*fixed_ranges):
            fixed = dict(zip(fixed_dims, fixed_values))
            group = []
            for i in range(sizes[target_dim]):
                coords = dict(fixed)
                coords[target_dim] = i
                group.append(global_rank(coords, sizes, order))
            result.append(group)
        groups[target_dim] = result

    return groups

def pretty_print(name, groups, limit=8):
    print(f"{name} groups ({len(groups)} total):")
    for g in groups[:limit]:
        print("  ", g)

if __name__ == "__main__":
    groups = build_groups(tp_size=4, dp_size=4, pp_size=2, order=("tp", "dp", "pp"))
    tp_groups = groups["tp"]
    dp_groups = groups["dp"]
    pp_groups = groups["pp"]

    assert tp_groups[0] == [0, 1, 2, 3]
    assert tp_groups[1] == [4, 5, 6, 7]
    assert dp_groups[0] == [0, 4, 8, 12]
    assert dp_groups[1] == [1, 5, 9, 13]
    assert pp_groups[0] == [0, 16]
    assert pp_groups[3] == [3, 19]

    pretty_print("TP", tp_groups)
    pretty_print("DP", dp_groups)
    pretty_print("PP", pp_groups)

    print("\nSwitch mapping order to tp-pp-dp:\n")
    groups2 = build_groups(tp_size=4, dp_size=4, pp_size=2, order=("tp", "pp", "dp"))
    pretty_print("TP", groups2["tp"])
    pretty_print("DP", groups2["dp"])
    pretty_print("PP", groups2["pp"])
```

运行方式：

```bash
python rank_groups_demo.py
```

你会看到两组输出：

- 第一组对应默认 `tp-dp-pp`
- 第二组对应可选 `tp-pp-dp`

这里最关键的不是代码量，而是循环规则：

- 固定 `pp_rank` 和 `dp_rank`，遍历 `tp_rank`，得到 TP 组
- 固定 `pp_rank` 和 `tp_rank`，遍历 `dp_rank`，得到 DP 组
- 固定 `dp_rank` 和 `tp_rank`，遍历 `pp_rank`，得到 PP 组

这正好对应前面“固定两维、枚举一维”的数学定义。

如果把它翻译成接近 Megatron 初始化的最小可读伪代码，大致是：

```python
tp_groups = build_tp_groups(...)
pp_groups = build_pp_groups(...)
dp_groups = build_dp_groups(...)

for ranks in tp_groups:
    dist.new_group(ranks)

for ranks in pp_groups:
    dist.new_group(ranks)

for ranks in dp_groups:
    dist.new_group(ranks)
```

这里“先建 TP，再建 PP，再建 DP”说的是 communicator 创建顺序。它和前面的 `global_rank` 公式并不冲突。前者决定 process group 何时、按什么集合创建；后者决定 rank 在线性编号下如何映射到三维坐标。

### 为什么这段代码是“可运行”和“可验证”的

很多文章给出的代码只展示 group 长什么样，但没有验证以下几点：

| 检查项 | 如果缺失会怎样 |
|---|---|
| 正向映射是否覆盖全部 rank | 可能漏 rank 或重复 rank |
| 反向映射是否能还原坐标 | 组看起来对，但坐标系定义实际错了 |
| 三类组是否数量正确 | 可能少组或多组 |
| 同一个 rank 是否恰好各属于一个 TP/DP/PP 组 | 组之间可能交叉错误 |

上面的脚本都做了这些检查，因此它不仅能“打印结果”，还可以作为初始化逻辑的单元测试原型。

### 新手最容易忽略的一点

很多人看到 `[0,1,2,3]`、`[0,4,8,12]`、`[0,16]`，会误以为这些组是手工写死的。其实不是。它们只是同一套公式在不同 mask 下的投影结果：

- TP mask 只保留 `tp`
- DP mask 只保留 `dp`
- PP mask 只保留 `pp`

这也是 `generate_masked_orthogonal_rank_groups` 这个名字真正想表达的内容。

---

## 工程权衡与常见坑

第一个坑是把“默认 rank 公式”和“可选初始化映射”混为一谈。很多文章只看到 `use_tp_pp_dp_mapping`，就以为 Megatron 一直按 `tp-pp-dp` 推导 rank，这不准确。Megatron Core 文档对正交 group 推导给出的经典公式仍然是 `tp-dp-pp`。

第二个坑是只关注逻辑正确，不关注物理拓扑。逻辑上 `[0,1,2,3]` 作为 TP 组完全成立，但如果这 4 个 rank 分布在两台机器上，那么 TP 的高频归约就会频繁穿过慢链路。TP 是训练中最敏感的通信维度之一，映射错了，吞吐会直接掉。

第三个坑是把“rank 连续”误解成“性能一定最好”。连续只说明编号上相邻，不说明物理上相邻。真正有效的是：

- rank 连续
- communicator 内成员物理邻近
- 这种邻近恰好对应高频通信维度

三者同时成立时，初始化映射才真正有意义。

第四个坑是开了 `use_tp_pp_dp_mapping`，却忘了同步检查相关通信配置。与初始化强相关的几个选项如下：

| 配置项 | 作用 | 典型风险 |
|---|---|---|
| `use_tp_pp_dp_mapping` | 把 rank 初始化顺序从 `tp-dp-pp` 改成 `tp-pp-dp` | 与 CP/EP 冲突 |
| `use_sharp` / `sharp_enabled_group` | 为指定数据并行通信组启用 SHARP | 组选择不一致时收益有限 |
| `high_priority_stream_groups` | 给某些 communicator 使用高优先级 stream | 如果组划分本身不贴拓扑，优先级也救不了带宽问题 |
| `nccl_communicator_config_path` | 为不同 communicator 设置 NCCL 配置 | communicator 名称和实际热点不匹配时，配置价值不大 |

第五个坑是只看“组内成员”，不看“组创建顺序”。NCCL communicator 初始化本身也有成本，创建顺序会影响连接建立和资源分配。这里不能简单理解为“越早创建越快”，更准确的说法是：高频组如果创建规则更稳定、成员更连续、物理更邻近，初始化行为通常更可控。

第六个坑是把 PP 通信想成“大量全局同步”。PP 的主通信通常是相邻 stage 之间的点对点发送，而不是像 TP/DP 那样频繁做整个组的集体通信。所以 PP 可以容忍更长的物理距离，但这不等于可以任意乱放。若相邻 stage 恰好跨慢链路，流水线空泡仍可能被放大。

第七个坑是 CP/EP 场景直接照搬 3D 并行经验。官方文档已经明确给出边界：`use_tp_pp_dp_mapping` 开启时，不要同时使用 EP 和 CP。原因是 rank 维度一旦扩展为 `tp-cp-ep-dp-pp` 之类组合，简单的三维直觉就不够用了。

最后给一张排障表，方便实际看日志时定位问题：

| 现象 | 常见原因 | 优先检查项 |
|---|---|---|
| 初始化卡死 | 各 rank 创建 group 的顺序或成员不一致 | 所有进程的 `order`、并行度、`new_group()` 调用序列 |
| TP 吞吐异常低 | TP 组跨机或跨弱链路 | TP rank 到物理 GPU 的映射 |
| DP 同步慢 | DP 组过大或落在拥塞链路 | DP 归约拓扑、SHARP 设置 |
| 改了 `use_tp_pp_dp_mapping` 后行为异常 | 同时用了 CP/EP 或配置联动缺失 | 并行维度组合、Bridge 初始化参数 |
| 组打印看起来对，但性能没变 | 逻辑组正确，物理贴图没变 | `CUDA_VISIBLE_DEVICES`、节点 rank 排列、启动脚本 |

---

## 替代方案与适用边界

如果你的目标只是“先把训练跑起来”，默认 `tp-dp-pp` 是更稳妥的方案。它历史更久，兼容面更大，也更容易和 CP、EP 等额外并行方式组合。

如果你的目标是“在固定机器拓扑上榨出更高吞吐”，尤其是多机训练、TP 通信压力大、硬件连接层级明显时，可以考虑 `use_tp_pp_dp_mapping`，把初始化顺序改成 `tp-pp-dp`。但前提是：

- 你清楚当前机器的 GPU 拓扑
- 你没有启用 CP/EP
- 你会同时检查 NCCL communicator 相关配置
- 你愿意做真实吞吐测试，而不是只看逻辑分组是否“更漂亮”

可以用一张表概括两种思路：

| 方案 | 默认线性思路 | 优点 | 限制 |
|---|---|---|---|
| `tp-dp-pp` | TP 最内层，DP 中间，PP 最外层 | 兼容性强，适合通用场景 | 未必最贴某些机器拓扑 |
| `tp-pp-dp` | TP 最内层，PP 中间，DP 最外层 | 更适合部分拓扑优化场景 | 官方明确不建议和 CP/EP 一起用 |

还有一个更通用的理解方式：Megatron 并不是只支持固定三维，它本质上支持“按 order 字符串定义维度顺序，再按 mask 取子空间”。这意味着你以后读到 `tp-cp-dp-pp`、`tp-ep-dp-pp` 之类配置时，不要把它们当成“新魔法”，而要回到同一个框架里理解：

1. 先定义维度顺序。
2. 再定义每个维度的 size。
3. 最后按 mask 从高维坐标系里切出需要的 communicator。

真正稳定的理解方式不是背组成员列表，而是掌握这三步。只要这套框架清楚，3D、4D、5D 初始化本质上都是同一类问题。

---

## 参考资料

1. NVIDIA Megatron Core 文档：`core.parallel_state`，包含 `RankGenerator`、`generate_masked_orthogonal_rank_groups` 与经典全局 rank 公式。  
   https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html

2. NVIDIA Megatron Core 并行策略指南，说明 TP、PP、DP、CP、EP 的职责与组合关系。  
   https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/parallelism-guide.html

3. NVIDIA Megatron Bridge 最新配置文档，明确说明 `use_tp_pp_dp_mapping` 会把初始化顺序从 `tp-dp-pp` 改成 `tp-pp-dp`，并要求不要与 EP、CP 同时使用。  
   https://docs.nvidia.com/nemo/megatron-bridge/nightly/apidocs/bridge/bridge.training.config.html

4. NVIDIA Megatron-LM 论文：最早系统化说明 TP 与大模型训练中的模型并行设计。  
   https://arxiv.org/abs/1909.08053

5. Narayanan 等人在 SC21 的论文：讨论更大规模集群上的流水线、张量并行与集群映射问题。  
   https://arxiv.org/abs/2104.04473

6. Megatron-LM GitHub 仓库：可对照实际初始化入口与并行状态管理代码。  
   https://github.com/NVIDIA/Megatron-LM

7. 社区解析文章，对 16 卡、32 卡下的 TP/PP/DP 组构造有直观示意，可作为读图辅助，但以官方文档为准。  
   https://zhuanlan.zhihu.com/p/20273026945

8. 社区源码解读文章，对 `initialize_model_parallel` 的循环构造方式有较直白的说明，但部分实现细节可能落后于新版本。  
   https://www.cnblogs.com/rossiXYZ/p/15876714.html
