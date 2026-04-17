## 核心结论

DeepSeek-V3 的 FP8 混合精度训练，核心不是“把全模型一键改成 8 位”，而是把**最耗算力的 GEMM**（矩阵乘法，白话说就是神经网络里最重的乘加计算）尽量放到 FP8 上执行，同时把**容易失真的敏感路径**保留在 BF16 或 FP32。公开资料显示，它在 671B 参数规模的 MoE 模型上，把 FP8 从“小范围试验”推进到了“大规模主干计算可用”的级别。

这里有三个关键结论。

第一，V3 采用的不是粗粒度的 tensor-wise scaling，而是**tile-wise scaling**。tile 是矩阵中的一个小块，白话说就是“不是整张大表共用一把尺子，而是每个小方块各用一把尺子”。这样做的原因很直接：同一个大矩阵里，不同区域的数值范围往往差很多，统一缩放会浪费动态范围，细粒度缩放更容易把 FP8 的可表示范围用满。

第二，FP8 并不意味着训练过程中所有东西都用 8 位。V3 的思路是：权重块、激活块进入 Tensor Core 前先量化成 FP8/E4M3，主干 GEMM 用 FP8 乘法完成，但**累加器**和若干关键算子仍保留更高精度，尤其会周期性把部分和 promote 到 FP32。promote 的意思是“把中间结果提到更高精度继续算”，白话说就是“轻量搬运，关键记账仍用精确账本”。

第三，FP8 的价值不只是“能跑”，而是**在大模型训练中真的省资源**。按公开披露口径，FP8 训练带来约 40% 的显存节省，并支撑 2048 张 H800、约 2 个月、总成本约 557.6 万美元的训练方案。这个数字不代表任何团队都能复制，但它说明一件事：低精度训练已经从“实验室技巧”变成了“超大规模工程手段”。

| 维度 | FP8 主干计算 | BF16 主干计算 |
|---|---|---|
| 单元素存储 | 1 字节 | 2 字节 |
| 主干 GEMM 带宽压力 | 更低 | 更高 |
| Tensor Core 吞吐潜力 | 更高 | 较高 |
| 量化/反量化复杂度 | 更高 | 更低 |
| 精度管理难度 | 更高 | 更稳 |
| 适合场景 | 超大规模、强互联集群 | 通用训练、实现更简单 |

---

## 问题定义与边界

问题可以定义为：**如何让一个 671B 级别的 MoE 模型，把最重的训练计算迁移到 FP8，同时不让误差扩散到模型发散或明显掉点。**

这里的 MoE，意思是 Mixture of Experts，白话说就是“不是每次都让所有参数都工作，而是让少数专家模块被激活”。它天然有更高的参数规模和更复杂的通信，因此对显存、带宽、吞吐都更敏感。DeepSeek-V3 的 FP8 训练，本质上是在回答两个工程问题：

1. 哪些模块可以降到 FP8？
2. 哪些模块必须留在 BF16/FP32？

边界不能画错。低精度最适合的是**规则、大规模、重复性高的矩阵乘法**；最不适合的是**对数值抖动极敏感的归一化、路由和输出路径**。

| 模块 | 是否适合 FP8 | 原因 |
|---|---|---|
| 线性层 GEMM | 适合 | 计算量最大，且可通过分块缩放控制误差 |
| 注意力中的投影矩阵乘法 | 适合 | 结构规整，硬件支持成熟 |
| MLP/Expert 内部大矩阵乘法 | 适合 | 是主干吞吐热点 |
| LayerNorm | 不适合 | 均值和方差对白噪声误差敏感 |
| Embedding | 通常不作为主 FP8 路径 | 访问模式特殊，误差会直接传入后续层 |
| MoE Gate | 不适合 | 路由分数一旦失真，会直接选错专家 |
| 最终输出头 | 不适合或谨慎使用 | 小误差可能被放大到 logits |

可以把它理解成“只在粗重搬运环节用轻工具，在精细校准环节继续用精密工具”。如果把 LayerNorm、MoE gate 也强行压到 FP8，模型不一定立刻报错，但训练曲线很容易出现震荡、损失下降变慢，严重时直接发散。

还有一个边界经常被忽略：**FP8 的风险不只在表示，更多在累加。** FP8 乘法可以快，但如果部分和长期停留在低精度累加器里，误差会不断堆积。因此 V3 的关键不是“纯 FP8 到底有多纯”，而是“在哪里及时退出 FP8，进入 FP32 记账”。

---

## 核心机制与推导

FP8 训练的第一步不是乘法，而是**缩放**。缩放就是先给一个 tile 单独算比例尺，把原始值映射到 FP8 可表达的区间里。

如果使用 E4M3 格式，公开资料里常见的有效动态上限可近似看作 448。对一个 tile 内的数据 $x$，先求最大绝对值：

$$
m = \max_i |x_i|
$$

再定义这个 tile 的缩放因子：

$$
s = \frac{m}{448}
$$

然后做量化：

$$
q_i = \operatorname{round}\left(\operatorname{clip}\left(\frac{x_i}{s}, -448, 448\right)\right)
$$

反量化时再乘回去：

$$
x_i \approx q_i \cdot s
$$

这套公式背后的逻辑很简单：让 tile 中绝对值最大的元素正好贴近 FP8 的上限，其余元素按比例缩进去。这样既减少溢出，也尽量避免大量数值被压扁到 0 附近。

玩具例子可以直接看两个数。假设一个 tile 中最大绝对值是 200，那么：

$$
s = 200 / 448 \approx 0.446
$$

若 tile 里有两个值 $[100,-200]$，量化后约为：

$$
[100/0.446,\ -200/0.446] \approx [224,\ -448]
$$

这说明原本看起来“超出 FP8 能力”的 100 和 200，只要先做局部缩放，就能被映射到 FP8 的表示范围内。它不是凭空提高精度，而是通过“分块单独拉伸坐标轴”减少浪费。

流程可以概括为：

| 步骤 | 做什么 | 目的 |
|---|---|---|
| tile max | 扫描每个 tile 的最大绝对值 | 获得当前块的动态范围 |
| scale | 计算 $s=m/448$ | 生成该块专属比例尺 |
| quantize | 用 $x/s$ 量化到 FP8 | 让数据进入 Tensor Core 友好格式 |
| FP8 GEMM | 执行 FP8×FP8 的矩阵乘 | 获得高吞吐和低带宽占用 |
| promote | 周期性把部分和转到 FP32 | 阻断累加误差持续扩散 |

真正难点在最后一步。矩阵乘法不是一次乘完，而是沿着 $K$ 维不断做乘加：

$$
C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}
$$

如果这个求和全过程都停留在低精度累加路径里，舍入误差会随着 $K$ 增大而放大。公开描述里，V3 采用“每 128 次乘加进行 promote”的策略，本质是在误差还没积累到危险区之前，把局部结果搬到 FP32 累加器里继续累计。这不是数学上唯一可行的方法，但它很符合硬件现实：**乘法吃 FP8 吞吐，求和吃 FP32 稳定性。**

真实工程例子可以这样理解。假设一个 Expert 的前馈层是超大矩阵乘法，输入激活分成若干个 128×128 tile，权重也按块存储。每个输入块和权重块都带一个 scale 元数据。Tensor Core 执行块乘时，只处理压缩后的 FP8 值；每累计一段，就把部分和提升到 FP32 寄存器，再与后续块结果合并。LayerNorm 和 gate 不进入这条路径，仍走高精度实现。于是系统实现的是“低精度主干 + 高精度关键点”的组合，而不是“全局统一低精度”。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，展示 tile 级 scale、量化、反量化，以及“分段累加再提升”的思路。它不是 CUDA 内核，但机制与工程实现是一致的。

```python
import math

FP8_MAX = 448.0

def calc_scale(tile, fp8_max=FP8_MAX):
    max_abs = max(abs(x) for row in tile for x in row)
    # 防止全 0 tile 导致除 0
    return max(max_abs / fp8_max, 1e-12)

def quantize_tile(tile, scale, fp8_max=FP8_MAX):
    q = []
    for row in tile:
        q_row = []
        for x in row:
            v = round(x / scale)
            v = max(-fp8_max, min(fp8_max, v))
            q_row.append(int(v))
        q.append(q_row)
    return q

def dequantize_tile(qtile, scale):
    return [[x * scale for x in row] for row in qtile]

def matmul_fp8_mixed(A, B, chunk_k=2):
    """
    A: m x k
    B: k x n
    先对 A、B 分别做整体演示量化，再模拟分段累加 promote 到 FP32。
    """
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k == k2

    sA = calc_scale(A)
    sB = calc_scale(B)
    qA = quantize_tile(A, sA)
    qB = quantize_tile(B, sB)

    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for kk in range(0, k, chunk_k):
        partial = [[0.0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                acc_low_precision_path = 0
                for t in range(kk, min(kk + chunk_k, k)):
                    acc_low_precision_path += qA[i][t] * qB[t][j]
                # promote: 这里把局部和转成 FP32 语义下的浮点
                partial[i][j] = float(acc_low_precision_path) * sA * sB
        for i in range(m):
            for j in range(n):
                C[i][j] += partial[i][j]
    return C

def matmul_fp32(A, B):
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k == k2
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = sum(A[i][t] * B[t][j] for t in range(k))
    return C

A = [
    [1.0, 2.0, -3.0, 4.0],
    [0.5, -1.0, 2.5, 3.0],
]
B = [
    [2.0, -1.0],
    [0.0, 3.0],
    [1.5, 2.0],
    [-2.0, 1.0],
]

c_mixed = matmul_fp8_mixed(A, B, chunk_k=2)
c_ref = matmul_fp32(A, B)

# 玩具例子里允许有量化误差，但误差应可控
for i in range(len(c_ref)):
    for j in range(len(c_ref[0])):
        assert abs(c_mixed[i][j] - c_ref[i][j]) < 0.2

# 检查量化-反量化基本成立
s = calc_scale([[100.0, -200.0]])
q = quantize_tile([[100.0, -200.0]], s)
dq = dequantize_tile(q, s)
assert abs(dq[0][0] - 100.0) < 1.0
assert abs(dq[0][1] + 200.0) < 1.0
```

如果把这段代码翻译成更贴近工程的伪流程，就是：

```python
for act_tile, weight_tile in tiled_stream:
    s_act = max_abs(act_tile) / 448
    s_w   = max_abs(weight_tile) / 448

    q_act = fp8_quantize(act_tile, s_act)
    q_w   = fp8_quantize(weight_tile, s_w)

    partial = tensor_core_fp8_gemm(q_act, q_w)

    fp32_accumulator += promote_to_fp32(partial, s_act, s_w)
```

真实工程里还要再加三层东西。

第一，scale 元数据要和 tile 一起流动，否则反量化时无法恢复量纲。

第二，LayerNorm、MoE gate、输出头等模块不能直接套这套流程，而要走 BF16/FP32 内核。

第三，tile 划分通常要和硬件友好尺寸对齐，比如 128×128 或类似 block 大小，因为它要服务 Tensor Core 的吞吐组织，而不是只服务数学表达。

---

## 工程权衡与常见坑

FP8 训练的难点不在“能不能写出量化公式”，而在“能不能把误差预算守住”。

最常见的错误，是把量化看成一个静态预处理。实际上训练中的激活分布在不断变化，尤其是 MoE 场景下，不同专家接到的数据分布可能明显不同。如果复用旧 scale，或者把 scale 做成太粗的统计量，量化误差会迅速增大。

| 常见坑 | 典型症状 | 规避策略 |
|---|---|---|
| 复用历史 scale | loss 抖动、局部层输出异常饱和 | 每个 tile 在线扫描 max abs |
| scale 粒度过粗 | 大量值接近 0 或频繁截断 | 改为 tile/block 级缩放 |
| 忘记定期 promote | 长序列或大 K 维时误差迅速积累 | 固定步数或固定 K 段 promote 到 FP32 |
| LayerNorm 误上 FP8 | 训练不稳定，收敛明显变差 | 保持 BF16/FP32 |
| MoE gate 误上 FP8 | 专家路由错位，负载失衡 | gate 保持高精度 |
| 只看显存不看带宽 | 多机吞吐上不去 | 联合评估 NVLink/InfiniBand 通信能力 |

这里最值得强调的是：**scale 必须“在线”算。** 在线的意思是“当前 tile 当前算”，不是“过去十步平均一下”。因为量化本质是用一把尺子重新刻度，而训练中的数据分布不是静止的。尺子一旦旧了，所有后续值都会被错量。

第二个坑是误以为“只要 FP8 乘法没问题，剩下就没事”。实际不是。GEMM 的乘法部分能低精度，不代表求和路径也能长期低精度。尤其当 $K$ 很大时，误差会像滚雪球一样积累。V3 的做法之所以关键，就在于它没有把“低精度”理解成“每一个中间状态都低精度”，而是把高低精度拆分到不同职责上。

第三个坑发生在系统层面。FP8 能节省显存，但 MoE 同时会引入更复杂的专家通信。如果集群互联差，省下来的显存未必能转化成有效吞吐。也就是说，**FP8 不是孤立优化，它需要和并行策略、网络拓扑、内核实现一起成立。**

---

## 替代方案与适用边界

如果不采用 FP8+MoE，大致还有三类路线。

| 方案 | 显存占用 | 带宽要求 | 精度风险 | 适用边界 |
|---|---|---|---|---|
| 全 BF16 训练 | 高 | 高 | 低 | 最稳妥，适合中大型集群 |
| FP8 混合精度训练 | 较低 | 很高 | 中 | 适合强互联、大规模训练 |
| 更激进的 INT4/更低位训练 | 最低 | 高 | 高 | 研究或特定优化场景 |

全 BF16 的优点是实现简单、训练稳定、工具链成熟。缺点也很明显：显存和带宽成本高，在超大模型上不够经济。

INT4 或更低位训练从存储角度更诱人，但训练阶段比推理阶段难得多。原因是训练不仅关心表示，还关心梯度传播、优化器状态、动态范围变化。很多团队能做 INT4 推理，不代表能稳定做 INT4 训练。

因此 FP8 的位置很明确：它不是“最省”的极限方案，而是当前硬件和算法共同支持下，**吞吐、显存、可训练性三者之间较平衡**的方案。

它也有清晰边界。以下场景更适合 FP8+MoE：

1. 有高带宽互联，如 NVLink 和 InfiniBand。
2. 有支持 FP8 Tensor Core 的新架构 GPU。
3. 愿意为 scale 管理、promote 策略、混合精度白名单投入工程复杂度。
4. 模型足够大，节省的显存和带宽能转化成真实收益。

以下场景则未必合适：

1. 只有少量普通 GPU，互联弱，通信容易成为主瓶颈。
2. 训练框架对 FP8 支持不完整，无法稳定管理 scale 和高精度回退。
3. 团队没有足够监控手段，一旦 loss 异常很难判断是模型问题还是数值问题。

对零基础到初级工程师来说，可以把结论记成一句话：**FP8 训练不是简单地“把精度调低”，而是用更细的缩放、更严格的边界控制，换取主干算力和显存效率。** 如果没有对应硬件、网络和内核支持，直接上 FP8 往往不是捷径，而是风险放大器。

---

## 参考资料

1. DeepSeek-V3 Low-Precision Training, EmergentMind 论文索引：2412.19437  
2. DeepSeek-V3 硬件软件协同与 FP8 实践, EmergentMind 论文索引：2505.09343  
3. DeepSeek 官方 FAQ 与公开训练成本说明，以及相关公开解读资料  
4. NVIDIA 关于 FP8、Tensor Core 与混合精度训练的公开技术资料  
5. E4M3 / FP8 数值格式与低精度训练相关实现文章、社区技术解读
