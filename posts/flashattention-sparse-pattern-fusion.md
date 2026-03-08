## 核心结论

FlashAttention 与稀疏模式融合的关键，不是把注意力矩阵保存成传统意义上的稀疏张量，而是先把序列切成固定大小的块，再让 kernel 只遍历那些“确实需要计算”的块。这里的块可以理解为一整片连续 token 组成的计算单元；在 Hopper 上，常见块大小是 $128\times128$，而 PyTorch 于 2026 年 3 月 5 日发布的 FlexAttention + FlashAttention-4（FA4）博客中说明，Blackwell 上当前 block-sparse 路径的最小粒度已受 `q_stage=2` 影响，调优重点转向 $256\times128$。

这件事之所以重要，在于它没有放弃 FlashAttention 的 IO-aware 路线。所谓 IO-aware，不是先问“乘法能不能更少”，而是先问“显存、共享内存、寄存器之间的数据搬运能不能更少”。在现代 GPU 上，很多长序列注意力的瓶颈并不是 FLOPs 不够，而是读写和同步把吞吐拖住了。块稀疏的价值，就是把“跳过无效区域”这件事放到 kernel 调度层完成，于是无效块不仅不参与 GEMM，也不触发对应的 K/V 加载、softmax 更新和写回。

FlexAttention 把这套能力进一步工程化。你在 Python 里写的是 `mask_mod` 或 `score_mod`，表达“哪些位置能看见”“哪些分数要修改”；编译链负责把这些规则降到真正可执行的底层代码。PyTorch 官方 2026 年 3 月 5 日的博客给出的结论是：在 Hopper 和 Blackwell 上，FlexAttention 现在已经能接入 FA4 后端；对 sliding window、document mask、ALiBi 等模式，Flash 后端相对原 Triton 后端能拿到更高吞吐，但也带来块大小、重编译、梯度支持和确定性方面的新边界。

| 对比项 | 完整注意力 | block-sparse FlashAttention |
|---|---:|---:|
| 计算范围 | 所有 $T_r\times T_c$ 块 | 仅遍历 mask 非零块 |
| 理论 FLOPs | $O(N^2 d)$ | $O(\rho N^2 d)$ |
| K/V 访存 | 每个块都可能被加载 | 仅活跃块触发加载 |
| softmax 方式 | 在线分块 softmax | 仍是在线分块 softmax |
| 是否显式存完整注意力矩阵 | 不需要 | 同样不需要 |
| 适合场景 | 短序列、全局依赖强 | 长序列、窗口、因果、文档分段、固定稀疏 |

---

## 问题定义与边界

标准自注意力最贵的部分来自 $QK^\top$。若序列长度为 $N$，每个头的维度为 $d$，那么完整注意力的主计算量近似为：
$$
\mathrm{Cost}_{dense}\approx O(N^2 d)
$$
新手最容易忽略的一点是，这里的平方项不是“数学上不好看”，而是会直接变成工程成本。例如，长度从 2k 增加到 8k，相当于长度变成原来的 4 倍，但注意力主计算会接近变成原来的 16 倍；从 8k 再拉到 32k，又是再乘 16。长上下文模型一旦进入 32k、64k、128k 区间，注意力的代价就不再是“稍慢一些”，而是决定系统是否还能跑得动。

但真实模型往往并不要求每个 token 都看见所有 token。最常见的几类模式如下：

| 模式 | 约束含义 | 直观解释 |
|---|---|---|
| 因果 attention | 只能看当前位置及其左侧 | 不能偷看未来 |
| 滑动窗口 attention | 只能看附近一段 token | 主要关注局部上下文 |
| 文档分段 mask | 只能看同一文档片段内 token | packed 训练时避免跨样本串扰 |
| 局部 + 全局 token | 大部分 token 看局部，少数 token 看全局 | 常见于长文档建模 |
| 固定图结构 mask | 可见关系由先验结构定义 | 树、表格、代码块等结构化输入 |

这里需要先把边界说清楚，否则容易把不同路线混在一起：

1. 这里讨论的是“注意力计算阶段的块级稀疏”，不是改造 Transformer 主干，也不是换成别的架构。
2. 这里讨论的是“按块跳过完全无效的区域”，不是对每个单独元素做不规则稀疏。
3. 这类方法最适合“模式能提前描述”的场景，例如因果、窗口、文档边界、固定全局 token；如果 mask 严重依赖输入内容且每步变化很大，块稀疏的收益会被元数据组织和调度开销侵蚀。

一个足够直观的玩具例子是 $4\times4$ 块网格。设序列长度 $N=512$，块大小 $B=128$，则 query 和 key/value 两个方向都各有 4 个块。若采用“每个 query 块只能看自己和左边两个块”的左侧窗口模式，则可见关系如下：

| Query 块 | 可见 KV 块 |
|---|---|
| 0 | 0 |
| 1 | 0, 1 |
| 2 | 0, 1, 2 |
| 3 | 1, 2, 3 |

原本总共有 $4\times4=16$ 个块，但真正需要遍历的只有 9 个。对白话理解来说，重点不是“算完再抹掉右上角”，而是“右上角那些块根本不发起计算任务”。

如果把这个例子画成块密度，得到：
$$
\rho=\frac{9}{16}=0.5625
$$
这意味着块级主计算已经从完整网格的 16 个 tile，缩减到 9 个 tile。真正落到 GPU 上，节省的不只是矩阵乘法次数，也包括对应块的 K/V 读入、softmax 累积与中间状态维护。

---

## 核心机制与推导

设：
$$
Q\in\mathbb{R}^{N\times d},\quad K\in\mathbb{R}^{M\times d},\quad V\in\mathbb{R}^{M\times d}
$$
按块大小 $B$ 切分后，query 方向有
$$
T_r=\left\lceil \frac{N}{B}\right\rceil
$$
个块，key/value 方向有
$$
T_c=\left\lceil \frac{M}{B}\right\rceil
$$
个块。随后定义块级二值 mask：
$$
M^{(blk)}\in\{0,1\}^{T_r\times T_c}
$$
其中 $M^{(blk)}_{i,j}=1$ 表示第 $i$ 个 query 块需要访问第 $j$ 个 KV 块，$0$ 表示整个块都可跳过。

对某个 query 块 $Q_i$，输出可以写成：
$$
O_i=\sum_{j:M^{(blk)}_{i,j}=1}\mathrm{softmax}_{\mathcal{A}_i}\left(\frac{Q_iK_j^\top}{\sqrt d}\right)V_j
$$
这里 $\mathcal{A}_i=\{j\mid M^{(blk)}_{i,j}=1\}$ 表示第 $i$ 行允许访问的块集合。

这个写法容易让初学者误解成“softmax 被拆坏了”。实际上没有。因为 FlashAttention 本来就不是先把完整注意力矩阵写出来再统一做 softmax，而是按块流式遍历，并在线维护每一行 softmax 所需的两个统计量：

$$
m_i^{(t)}=\max\left(m_i^{(t-1)},\max S_i^{(t)}\right)
$$

$$
\ell_i^{(t)}=\ell_i^{(t-1)}e^{m_i^{(t-1)}-m_i^{(t)}}+\sum_{x\in S_i^{(t)}}e^{x-m_i^{(t)}}
$$

其中 $S_i^{(t)}$ 表示第 $t$ 次读入时当前块给第 $i$ 行带来的局部分数集合，$m_i$ 是当前最大值，$\ell_i$ 是归一化因子。在线 softmax 的意义是：即便分数分多批到来，只要最大值和归一化项按上式更新，最终结果和“一次性对整行做 softmax”在数学上等价。因此，那些本来就应该被 mask 掉的块，直接不纳入遍历集合即可，不会破坏正确性。

若总块数为 $T_rT_c$，其中活跃块数为 $K_{active}$，定义块密度：
$$
\rho=\frac{K_{active}}{T_rT_c}
$$
则主计算复杂度可近似写成：
$$
O(\rho N^2 d)
$$
这里的 $\rho$ 不是 token 级稀疏度，而是块级稀疏度。两者差别很关键。因为只要某个块中仍存在有效元素，这个块通常就要整体保留，硬件看到的是连续 tile，不是任意散落的单点。

继续用前面的例子，$N=512,B=128$，总块数 16，活跃块数 9，因此：
$$
\rho=\frac{9}{16}
$$
若进一步把窗口缩窄到“每个块只看自己和左边一块”，则活跃块数会变成：
$$
1+2+2+2=7,\quad \rho=\frac{7}{16}=0.4375
$$
这时理论 GEMM 数量已经比完整块网格少了一半以上。

真实工程里，FA4 的价值还不只在“少算”，而在“少算时仍尽量保持 tensor core 和异步流水线处于高利用率状态”。PyTorch 2026 年 3 月 5 日的官方博客明确写到，FlexAttention 已能把 `score_mod` / `mask_mod` 生成到 CuTeDSL，再接入 FA4 的异步 pipeline；博客同时说明，在 Blackwell 上由于两块 ping-pong 调度和 `q_stage=2`，block-sparse 的最小跳过粒度会变成 256 行 query。这一条告诉我们的不是“公式变了”，而是“同样的稀疏思想到了新硬件上，最优块形状会跟着底层 pipeline 设计改变”。

---

## 代码实现

下面先给出一个完全可运行、只依赖 Python 标准库的块 mask 生成器。它的作用不是跑高性能注意力，而是把“块级可见关系”这个核心机制先验证清楚。

```python
import math
from typing import List


def build_left_window_block_mask(
    seq_len: int,
    block_size: int,
    left_blocks: int,
) -> List[List[int]]:
    """
    构造一个块级因果 + 左窗口 mask。
    含义：第 qi 个 query 块只能看 [qi-left_blocks, qi] 范围内的 KV 块。
    """
    assert seq_len > 0
    assert block_size > 0
    assert left_blocks >= 0

    num_blocks = math.ceil(seq_len / block_size)
    mask = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]

    for qi in range(num_blocks):
        start = max(0, qi - left_blocks)
        end = qi  # 因果约束：不能看右边
        for kj in range(start, end + 1):
            mask[qi][kj] = 1
    return mask


def count_active_blocks(mask: List[List[int]]) -> int:
    return sum(sum(row) for row in mask)


def pretty_print(mask: List[List[int]]) -> None:
    for row in mask:
        print(" ".join(str(x) for x in row))


if __name__ == "__main__":
    mask = build_left_window_block_mask(seq_len=512, block_size=128, left_blocks=2)

    assert len(mask) == 4
    assert len(mask[0]) == 4
    assert count_active_blocks(mask) == 9

    assert mask[0] == [1, 0, 0, 0]
    assert mask[1] == [1, 1, 0, 0]
    assert mask[2] == [1, 1, 1, 0]
    assert mask[3] == [0, 1, 1, 1]

    pretty_print(mask)
```

这段代码运行后会打印：

```text
1 0 0 0
1 1 0 0
1 1 1 0
0 1 1 1
```

如果你是第一次接触 block-sparse attention，可以把这张 0/1 网格理解成“调度表”。值为 1 的地方，表示那个 query 块和 KV 块之间会触发一次 tile 级计算；值为 0 的地方，表示整块直接跳过。

下面再补一个更贴近公式的纯 Python 版本，用它验证块密度和理论复杂度估计：

```python
def block_density(mask: List[List[int]]) -> float:
    rows = len(mask)
    cols = len(mask[0]) if rows > 0 else 0
    total = rows * cols
    active = count_active_blocks(mask)
    return active / total if total > 0 else 0.0


if __name__ == "__main__":
    mask = build_left_window_block_mask(seq_len=512, block_size=128, left_blocks=2)
    rho = block_density(mask)

    assert abs(rho - 9 / 16) < 1e-12
    print(f"active_blocks={count_active_blocks(mask)}")
    print(f"density={rho:.4f}")
```

输出结果应为：

```text
active_blocks=9
density=0.5625
```

如果你已经在用 PyTorch 2.5 之后的 FlexAttention，写法会从“手工构造 0/1 网格”转向“先定义 token 级规则，再交给 `create_block_mask` 生成块元数据”。下面这段代码补成了更完整、能直接落地的版本：

```python
import torch
from functools import partial
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)

WINDOW = 256
DEVICE = "cuda"
DTYPE = torch.bfloat16

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def sliding_window_mask(b, h, q_idx, kv_idx):
    # 同时限制为“最多看左边 WINDOW 个 token”
    return (q_idx - kv_idx) <= WINDOW

mask_mod = and_masks(causal_mask, sliding_window_mask)

block_mask = create_block_mask(
    mask_mod,
    B=1,
    H=8,
    Q_LEN=8192,
    KV_LEN=8192,
    device=DEVICE,
    BLOCK_SIZE=128,  # Hopper 常见设置；Blackwell 上当前 FA4 block-sparse 粒度更大
)

flex_flash = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
    dynamic=False,
)

q = torch.randn(1, 8, 8192, 128, device=DEVICE, dtype=DTYPE)
k = torch.randn(1, 8, 8192, 128, device=DEVICE, dtype=DTYPE)
v = torch.randn(1, 8, 8192, 128, device=DEVICE, dtype=DTYPE)

out = flex_flash(q, k, v, block_mask=block_mask)

assert out.shape == (1, 8, 8192, 128)
assert out.dtype == DTYPE
```

这段代码里最值得新手分清的是三层概念：

| 层次 | 你写的东西 | 它表达什么 |
|---|---|---|
| 语义层 | `mask_mod` | 哪些 token 对允许互相看见 |
| 执行元数据层 | `BlockMask` | 哪些块需要遍历、每行遍历哪些列块 |
| kernel 层 | `flex_attention(..., kernel_options={"BACKEND": "FLASH"})` | 用哪个后端真正执行 |

也就是说，FlexAttention 的接口让你描述“规则”，而不是自己手写“遍历循环”。这正是它对研究和工程都友好的地方。

再给一个最小 CPU/非 Flash 验证思路，帮助初学者确认规则没有写错：可以先用 `create_mask` 生成完整布尔 mask，在小尺寸上和手工实现对比，再切到 `create_block_mask` + `flex_attention`。原因很简单，块稀疏真正难的不是数学，而是容易把掩码方向、边界条件、因果方向写反。先在小例子上验证语义，再进高性能后端，排错成本最低。

另外，PyTorch 官方博客在 2026 年 3 月 5 日给出了一条很关键的端到端验证：在 Llama 3 70B、64 张 H100、序列长度 8192、训练 1000 步的实验中，Flash 后端与参考实现最终都收敛到约 3.7 的 loss。这说明 block-sparse + FlexAttention 并不是只在微基准里成立，而是已经进入真实训练验证场景。

---

## 工程权衡与常见坑

第一类坑是重编译。PyTorch 官方博客在 2026 年 3 月 5 日明确写到：动态 tensor shape 可以运行时解析，但 `score_mod` 或 `mask_mod` 中捕获的 scalar 会被 baked into compiled kernel。换句话说，如果你写了一个会频繁变化的 `soft_cap`，每个新取值都可能触发一次新编译。对初学者来说，这种问题最隐蔽，因为表面上你只是“改了个超参数”，实际上底层在重造 kernel。

第二类坑是 captured buffer 的梯度支持。若你在 `score_mod` 中读取一个 `requires_grad=True` 的 bias 张量，当前 Flash 后端不支持这种 captured buffer 的反向梯度，官方建议回退 Triton。这里不是公式不成立，而是当前后端实现边界还没覆盖这类情况。

第三类坑是 block-sparse backward 的确定性。PyTorch 官方博客在 2026 年 3 月 5 日说明：启用 block-sparsity 时，Flash 后端的 backward 还不是 deterministic；只有 score-mod-only 工作负载保持 deterministic。若你的实验强依赖严格复现，例如要逐 bit 比对、跑确定性回归或科研复现实验，这就是必须提前接受的边界。

第四类坑是块大小选择。很多新手会本能地认为“块越小越精细，越小越省算”。这在硬件上通常不成立。块太小会抬高调度、同步、索引访问和流水线填充开销，导致 tensor core 吃不满。PyTorch 官方博客写得很清楚：截至 2026 年 3 月 5 日，FA4 的 block-sparse 路径在 Hopper 重点围绕 $128\times128$ 调优，在 Blackwell 上则受 `q_stage=2` 影响，当前最小稀疏粒度是 $256\times128$。

第五类坑是把 token 级稀疏误当成块级收益。比如某个 $128\times128$ 的块里，只有很少一部分元素有效，但只要不是整块全零，它通常还是要被整体计算。于是“元素级看起来很稀疏”不等于“块级执行一定很省”。这也是为什么块设计、窗口边界和序列分桶方式会直接影响真实加速比。

第六类坑是 `create_block_mask` 本身也有成本。PyTorch 文档和早期 FlexAttention 博客都强调过，`create_block_mask` 不是零开销操作。若同一种规则会在多个层、多个 step 中复用，通常应该缓存这份 block metadata，而不是每次 attention 调用都重新生成。

| 问题 | 根因 | 典型症状 | 更稳妥的做法 |
|---|---|---|---|
| 动态 scalar 频繁变动 | scalar 被编进 kernel | 吞吐抖动、首次调用慢 | 固定常量、按桶离散化、减少变更频率 |
| 可训练 bias 放进 `score_mod` | Flash 后端暂不支持 captured buffer grad | 反向报错或无法走预期路径 | 回退 Triton 后端 |
| block-sparse backward 非确定性 | 当前 Flash 实现仍在完善 | 同配置多次运行结果细微不一致 | 可复现实验改用 Triton 或 dense |
| 块太小 | 调度/访存/同步开销过高 | 理论省算，实际不加速 | 优先采用官方推荐块大小 |
| mask 每次都重建 | 元数据生成本身耗时 | 前处理开销变大 | 对可复用 mask 做缓存 |
| 稀疏模式过度动态 | 元数据组织和调度成本上升 | 端到端收益不稳定 | 先离散化规则，再决定是否 block-sparse |

一个常见误用是：把训练中会变化的温度参数、阈值参数直接捕获进 `mask_mod`，试图做“动态更智能的稀疏控制”。结果往往是两头受损。一头是重编译，一头是规则过于动态后 block metadata 难以复用，最终理论稀疏度很好看，真实吞吐却不升反降。工程上更稳的做法通常是：让 mask 结构保持离散、低频变化，把连续可训练部分放在更外层控制逻辑，必要时退回 Triton 路线换取更宽松的语义支持。

---

## 替代方案与适用边界

如果稀疏结构是静态的、规则的、容易按块表达，block-sparse FlashAttention 通常是很好的选择。典型例子包括因果 mask、滑动窗口、文档 packing 后的分段 mask、固定全局 token 模式。这类问题的共同点是：稀疏关系可以在执行前就用比较紧凑的块元数据表达出来，kernel 不需要为了“临时判断谁该算”付出太大额外代价。

如果稀疏结构高度依赖输入内容，并且每个 batch、每层、每个头都可能发生明显变化，就要更谨慎。FlexAttention 当然可以表达复杂规则，但“能表达”不等于“端到端一定更快”。当数据依赖过强时，mask 生成、块索引组织、调度负载不均衡都可能吞掉理论收益。

还有一条硬边界来自硬件。PyTorch 2026 年 3 月 5 日的博客明确指出，新的 Flash 后端收益主要体现在 Hopper 与 Blackwell 上；同时博客也写明，Triton 后端仍支持更广范围硬件，并且未来还会继续改进。因此，如果你的部署环境混杂、设备较旧、或者必须把同一套代码跑在更广 GPU 范围内，Triton 常常是更稳的折中。

当块密度已经很低时，block-sparse 也不一定自动成为最优方案。原因有两个。第一，块内可能仍然保留了很多“形式上有效、实际利用率不高”的元素。第二，当有效块太少时，块索引、调度和负载均衡本身可能开始变得显著。此时工程上常见的思路，不是盲目继续缩块，而是把块稀疏与别的压缩方法组合，例如分页 KV cache、候选块预筛选、分层路由或低秩近似。

| 方案 | 能否表达自定义模式 | 性能上限 | 梯度/确定性边界 | 硬件覆盖 |
|---|---|---|---|---|
| Dense FlashAttention | 中 | 高 | 边界较少 | 较广 |
| FlexAttention + Flash | 高 | 在新 GPU 上更高 | 受 captured buffer grad、determinism 限制 | 更偏 Hopper/Blackwell |
| FlexAttention + Triton | 很高 | 通常低于 Flash 峰值 | 语义支持更宽 | 更广 |
| 显式稀疏张量方案 | 中到高 | 往往受 kernel 质量影响 | 依实现而定 | 常不如专用 attention kernel 友好 |

一个更实用的选型顺序是：

1. 先判断稀疏规则能否稳定按块表达。
2. 再判断目标硬件是否主要是 Hopper 或 Blackwell。
3. 再判断训练是否依赖 captured buffer grad 或 deterministic backward。
4. 最后再看稀疏结构是否会高频动态变化。

如果答案接近“能、是、不依赖、变化低”，优先考虑 FlexAttention + Flash。  
如果答案接近“能、硬件一般、依赖梯度或确定性、变化低”，FlexAttention + Triton 更稳。  
如果答案接近“不能稳定按块表达、变化很高”，那就不要强行套 block-sparse，先重新设计稀疏策略。

---

## 参考资料

1. PyTorch Blog. *FlexAttention + FlashAttention-4: Fast and Flexible*. Published March 5, 2026.  
   价值：确认 FlexAttention 已接入 FA4 后端；给出 Hopper/Blackwell 性能对比、CuTeDSL 集成、Llama 3 70B 训练验证，以及动态 scalar、captured buffer grad、deterministic backward、块大小约束等限制。  
   https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/

2. PyTorch Documentation. *torch.nn.attention.flex_attention*. Stable docs, accessed March 8, 2026.  
   价值：给出 `flex_attention`、`create_block_mask`、`BlockMask`、`and_masks`、`score_mod`/`mask_mod` 签名与 `BLOCK_SIZE`、`BACKEND="FLASH"` 等 API 边界。  
   https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html

3. PyTorch Blog. *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*. Published 2024.  
   价值：补充 FlexAttention 的原始设计动机、BlockMask 基本思想、`create_block_mask` 成本、滑动窗口与文档 mask 的实现直觉。  
   https://pytorch.org/blog/flexattention/

4. Emergent Mind. *Block-Sparse FlashAttention (BSFA)*. Updated December 14, 2025.  
   价值：提供 block-sparse FlashAttention 的背景综述、复杂度表达、块跳过机制与相关扩展路线。该来源更适合做综述补充，不应替代官方文档作为 API 依据。  
   https://www.emergentmind.com/topics/block-sparse-flashattention-bsfa
