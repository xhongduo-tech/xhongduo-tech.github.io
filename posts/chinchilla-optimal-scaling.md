## 核心结论

Chinchilla 定律讨论的是**在固定训练算力下，参数量和训练数据该怎么分配才最省算力**。结论可以直接写成三条：

1. 最优参数量 $N^*$ 与最优训练 token 数 $D^*$ 都近似按 $\sqrt{C}$ 增长，其中 $C$ 是总训练计算量（也就是总 FLOPs，白话讲就是训练期间做了多少次浮点运算）：
   $$
   N^* \propto C^{0.5}, \qquad D^* \propto C^{0.5}
   $$
2. 因为二者同阶增长，所以最优点上它们的比例近似不变：
   $$
   \frac{D^*}{N^*} \approx \text{const}
   $$
   对 Chinchilla 论文覆盖的实验区间，这个常数常被工程上总结为 **约 20 token/parameter**。
3. 这意味着同样的训练预算下，很多场景里“**更小的模型，训练更久**”优于“**更大的模型，训练更短**”。经典例子是：70B 参数、1.4T token 的 Chinchilla，比 280B 参数、300B token 的 Gopher 在同等训练算力下表现更好，同时推理成本更低。

这里的“20 token/parameter”不要理解成自然常数。它是一个**经验比例**，意思是每个参数大致需要看到约 20 个训练 token，模型容量和数据规模才比较匹配。工程上常把它当成第一版预算公式：

$$
D \approx 20N
$$

例如：

| 参数量 $N$ | 经验最优 token 数 $D$ | 直观解释 |
|---|---:|---|
| 1B | 20B | 10 亿参数通常需要约 200 亿 token |
| 7B | 140B | 常见开源基座量级 |
| 70B | 1.4T | Chinchilla 代表点 |

一个面向新手的玩具例子是：如果你计划训练 1B 参数模型，只喂 5B token，那么模型大概率是**欠训练**，白话讲就是参数很多，但每个参数看到的数据太少，学不满。如果改成 20B token，通常更接近算力最优点。

---

## 问题定义与边界

Chinchilla 处理的是一个很具体的问题：**给定总训练算力，怎样选参数量 $N$ 和训练数据量 $D$，使最终 loss 最低**。

这里的 loss 指训练或验证时的语言建模损失，白话讲就是“模型预测下一个 token 时平均有多不准”。论文里常用一个参数化形式来描述它：

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
$$

这几个符号可以这样理解：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $E$ | 不可约误差 | 就算模型和数据继续变大，也很难再降的底噪 |
| $A/N^\alpha$ | 模型项 | 参数更多时，模型容量不足带来的误差会下降 |
| $B/D^\beta$ | 数据项 | 数据更多时，训练信号不足带来的误差会下降 |
| $\alpha,\beta$ | 幂律指数 | 参数和数据各自扩大时，loss 下降的速度 |

同时有一个算力约束。对 dense Transformer 的预训练，常见近似是：

$$
C \approx 6ND
$$

意思是训练总 FLOPs 与“参数量 × token 数”近似成正比，常数 6 来自前向、反向和实现细节的粗略折算。它不是数学真理，但足够适合做预算估算。

所以问题变成：

$$
\min_{N,D} L(N,D)\quad \text{s.t.}\quad C\approx 6ND
$$

这一定义有三个边界条件必须说清。

第一，**参数怎么计数**。Kaplan 2020 与 Chinchilla 2022 的一个重要差异，是是否把 embedding 等参数完整计入总参数量。白话讲，如果你把一部分参数“漏算”，就会把最优配比看偏，容易误以为“大模型更划算”。

第二，**结论适用的是预训练主导场景**。这里优化的是预训练阶段的 compute-optimal point，不是所有业务目标下的全局最优点。如果上线后推理量极大，推理成本会反过来改变最优选择。

第三，**20 倍比例是区间规律，不是所有设定下都精确等于 20**。数据质量、去重策略、tokenizer、优化器、训练稳定性都会让常数项变化，但不会轻易推翻“参数和数据应近似线性配比”的主结论。

把边界直观看成预算表，会更容易理解：

| 参数量 $N$ | 建议 token 数 $D \approx 20N$ | 训练算力 $C \approx 6ND$ |
|---|---:|---:|
| 1B | 20B | $1.2\times10^{20}$ FLOPs |
| 7B | 140B | $5.88\times10^{21}$ FLOPs |
| 70B | 1.4T | $5.88\times10^{23}$ FLOPs |

上表里 70B/1.4T 对应的就是 Chinchilla 代表点。它告诉我们：如果你的预算大概在 $6\times10^{23}$ FLOPs 量级，那么与其做一个 200B 以上但只训练几千亿 token 的模型，不如把参数压到 70B 左右，再把 token 拉到 1.4T 左右。

---

## 核心机制与推导

核心机制其实就是一句话：**参数不足和数据不足都会造成损失，最优点出现在两种瓶颈被平衡的时候**。

从
$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
$$
和
$$
C=6ND
$$
出发，可以先把 $D=C/(6N)$ 代入 loss：

$$
L(N)=E+\frac{A}{N^\alpha}+B\left(\frac{6N}{C}\right)^\beta
$$

这时 loss 变成只关于 $N$ 的函数。对 $N$ 求导并令导数为 0，可得最优条件：

$$
-\alpha A N^{-(\alpha+1)}+\beta B\left(\frac{6}{C}\right)^\beta N^{\beta-1}=0
$$

整理后得到：

$$
N^{\alpha+\beta} \propto C^\beta
$$

因此：

$$
N^*(C)\propto C^{\frac{\beta}{\alpha+\beta}}
$$

再由 $D=C/(6N)$ 得：

$$
D^*(C)\propto C^{\frac{\alpha}{\alpha+\beta}}
$$

如果定义

$$
a=\frac{\beta}{\alpha+\beta}, \qquad b=\frac{\alpha}{\alpha+\beta}
$$

那么就有：

$$
N^*(C)=G\left(\frac{C}{6}\right)^a,\qquad D^*(C)=G^{-1}\left(\frac{C}{6}\right)^b
$$

这里的 $G$ 是由常数项组成的比例因子，可写成：

$$
G=\left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}
$$

它决定的是“常数倍”，而 $a,b$ 决定的是“增长斜率”。

这一步已经能解释为什么 Chinchilla 会得到接近 $0.5/0.5$ 的结论。因为实验拟合中，$\alpha$ 和 $\beta$ 的量级接近，意味着“多加参数”与“多加数据”带来的边际收益相近。于是：

$$
a\approx b\approx 0.5
$$

也就是说，随着总算力增长，最优参数和最优数据都应按平方根增长，而不是一边倒地偏向模型尺寸。

一个玩具例子可以直接代数代进去。假设：

$$
\alpha=0.34,\qquad \beta=0.28
$$

则

$$
a=\frac{0.28}{0.34+0.28}\approx 0.452,\qquad
b=\frac{0.34}{0.34+0.28}\approx 0.548
$$

这说明即便两个指数不完全相等，最优规律仍然很接近“各占一半”：参数随算力约按 $C^{0.45}$ 增长，数据按 $C^{0.55}$ 增长，二者仍然是近似线性配比，不会出现 Kaplan 早期那种明显偏向“多堆参数、少喂数据”的结论。

为什么会有三种实证方法却得到一致结果？因为它们都在逼近同一个最优曲面。

| 方法 | 做法 | 会看到什么 |
|---|---|---|
| 固定 $N$，扫描 $D$ | 对每个参数量找最优 token 数 | 会发现最优 $D$ 随 $N$ 近似线性增长 |
| 固定 FLOPs 包络线 | 在同一算力上比较不同 $(N,D)$ 组合 | 最优点落在“小一些的模型、更长训练”的区域 |
| 直接拟合参数化 loss | 用大量实验拟合 $L(N,D)$ | 求导后直接得到 $a,b$ 的公式 |

可以把 iso-FLOP 曲线理解成“同一预算下所有可能的模型配置”。在这条曲线上，增大 $N$ 就必须减小 $D$；增大 $D$ 就必须减小 $N$。Chinchilla 发现，最佳点不是曲线最右边的“大模型”，而是中间更平衡的位置。

真实工程例子就是 Gopher 与 Chinchilla 的对比。Gopher 约 280B 参数、300B token；Chinchilla 约 70B 参数、1.4T token。二者训练算力大致同级，但 Chinchilla 在多项 benchmark 上更好。这个结果的重要性不在“70B 恰好神奇”，而在于它验证了一个更一般的命题：**如果数据明显少于参数所需，额外增大模型只是扩大未训练充分的容量**。

---

## 代码实现

下面给一个可运行的 Python 小脚本，把公式直接变成预算工具。它做三件事：

1. 根据 $\alpha,\beta,A,B$ 和总算力 $C$ 估算最优 $N,D$
2. 检查 $D/N$ 是否接近经验比例
3. 用 70B/1.4T 这个点反推算力量级，方便对照

```python
def chinchilla_optimal(alpha, beta, A, B, C):
    """
    Return (N_opt, D_opt) under C ~= 6ND
    N: parameter count
    D: training tokens
    C: training FLOPs
    """
    G = ((alpha * A) / (beta * B)) ** (1.0 / (alpha + beta))
    n_opt = G * ((C / 6.0) ** (beta / (alpha + beta)))
    d_opt = (1.0 / G) * ((C / 6.0) ** (alpha / (alpha + beta)))
    return n_opt, d_opt


def flops_from_nd(N, D):
    return 6.0 * N * D


# Toy setup: choose A/B so that D/N ~= 20 near alpha ~= beta
alpha = 0.34
beta = 0.28
A = 406.0
B = 1.0

# Real engineering-sized budget: roughly the 70B / 1.4T scale
C = flops_from_nd(70e9, 1.4e12)

n_opt, d_opt = chinchilla_optimal(alpha, beta, A, B, C)
tokens_per_param = d_opt / n_opt

print(f"N_opt = {n_opt:.3e} params")
print(f"D_opt = {d_opt:.3e} tokens")
print(f"tokens/param = {tokens_per_param:.2f}")
print(f"total FLOPs = {C:.3e}")

# Basic sanity checks
assert C > 0
assert n_opt > 0 and d_opt > 0
assert abs(flops_from_nd(n_opt, d_opt) - C) / C < 1e-10
assert 10 < tokens_per_param < 40

# Quick beginner examples using the 20x heuristic
for N in [1e9, 7e9, 70e9]:
    D = 20 * N
    print(f"N={N:.0e}, suggested D={D:.0e}, FLOPs={flops_from_nd(N, D):.3e}")
```

这段代码里最重要的不是常数的绝对值，而是结构：

- `alpha, beta` 控制参数与数据的边际收益斜率
- `A, B` 控制最优比例常数，也就是为什么有时是 15x、有时更接近 20x 或 25x
- `C ~= 6ND` 负责把“参数和数据”的选择绑定到同一预算里
- `tokens/param` 是最直接的工程检查量

如果你只是做粗预算，完全可以不用拟合 $A,B,\alpha,\beta$，直接上工程近似：

```python
def heuristic_budget_from_params(N):
    D = 20 * N
    C = 6 * N * D
    return D, C

D_70B, C_70B = heuristic_budget_from_params(70e9)
print(D_70B, C_70B)
assert D_70B == 1.4e12
```

这个玩具版适合做第一轮方案筛选。比如你手里大概能承受 $6\times10^{23}$ FLOPs 的训练预算，那么：

- 若做 70B，按 20x 经验，需要约 1.4T token
- 若你手头只有 300B token，那么 70B 可能已经偏大
- 更合理的方向可能是把模型压到 15B 左右，再把数据尽量喂满

真实工程里，预算函数通常还会继续加三类约束：

| 额外约束 | 为什么要加 | 对最优点的影响 |
|---|---|---|
| 最大显存/并行度 | 模型放不下或训练效率太低 | 限制 $N$ 上界 |
| 数据可得性 | 高质量 token 不足 | 限制 $D$ 上界 |
| 推理成本目标 | 上线调用量很大 | 倾向更小的 $N$ |

所以代码实现的正确用法不是“机械相信 20 倍”，而是先用它给出基线，再把工程约束叠上去。

---

## 工程权衡与常见坑

第一类权衡是**训练成本和推理成本不是一回事**。Chinchilla 最初优化的是训练算力，但它的一个额外好处是：更小的模型通常也更省推理 FLOPs。对需要长期在线服务的系统，这很关键。因为一旦进入高并发推理阶段，生命周期成本常常不是预训练，而是每天都在烧的钱。

Gopher 与 Chinchilla 的对比就是典型工程案例。两者训练预算接近，但 Chinchilla 只用 70B 参数。结果是：

- 训练效果更好，因为数据更充分
- 推理更便宜，因为每次前向通过的参数更少
- 部署更灵活，因为显存和延迟压力更小

第二类常见坑是**沿用 Kaplan 的旧直觉**。Kaplan 结论常被简化成“固定算力下优先加大模型”。这在小规模实验、特定参数统计口径下能观察到，但如果直接把它搬到更大规模预训练，容易得到明显欠训练的模型。白话讲，就是模型壳子很大，但内容没有喂够。

第三类坑是**参数计数口径不一致**。如果有人说“这个模型是 65B”，但计算最优比例时只统计了部分参数，另一个人按全参数统计，那么两人的 token/parameter 会看起来差很多，最后会误判到底是数据太少，还是模型太大。

第四类坑是**把 20x 当成不变真理**。真正稳定的是“$D$ 与 $N$ 近似线性配比”，不是“必须恰好 20”。以下因素都会改变常数：

- 训练数据去重强度
- 数据质量分布
- tokenizer 粒度
- 优化器与 batch 设计
- 训练稳定性约束
- 是否使用更强正则化或 curriculum

实际做项目时，建议用一张检查清单：

| 检查项 | 典型问题 | 建议动作 |
|---|---|---|
| 参数是否全量统计 | 漏算 embedding、输出头 | 统一口径后再算 $D/N$ |
| token 是否是真实去重后数量 | 重复数据虚高 | 以有效 token 为准 |
| FLOPs 估算是否一致 | 只算主干不算其他开销 | 至少统一用 $6ND$ 粗估 |
| 当前 $D/N$ 是否过低 | 容易欠训练 | 优先加数据，谨慎加参数 |
| 推理是否是主要成本 | 训练后上线极贵 | 倾向更小模型和更长训练 |

一个很常见的错误决策是：团队有 300B token 数据，想训练 70B 模型，因为“参数更大看起来更先进”。按 Chinchilla 经验，70B 更匹配约 1.4T token；如果只有 300B token，这个配置很可能没有训练够。与其硬上，不如做更小模型，或者先解决数据扩充与质量问题。

---

## 替代方案与适用边界

Chinchilla 不是“唯一正确方案”，它是**在 dense 预训练、固定训练预算、以最终 loss 为主要目标时的强基线**。一旦目标函数变了，最优点也会变。

第一种替代思路是 **Kaplan 风格配比**。它更偏向“大模型、相对少数据”。在小规模实验、旧的参数统计口径，或者数据扩展确实做不到的时候，它仍然可能是现实中的折中方案。它的问题不是“完全错”，而是容易在更大规模下把模型推向欠训练区。

第二种替代思路是 **Beyond Chinchilla-Optimal**。它考虑的不只是训练 loss，还把推理延迟、推理总量、服务成本一起纳入目标。此时最优解可能继续往“更小模型、更多训练 token”的方向移动。原因很直接：如果每天要处理海量请求，推理成本会长期压过训练成本，参数越小越省钱。

一个具体例子是低延迟服务。假设某业务非常看重单次响应延迟，那么它可能宁愿选择 10B 参数、1.7T token 的方案。这个比例已经远高于 20x：

$$
\frac{1.7T}{10B}=170
$$

这在纯训练最优角度看不是标准 Chinchilla 点，但在“总拥有成本最小”的目标下可能更优，因为它显著降低了在线推理开销。

第三种替代思路是 **MoE（Mixture of Experts，专家混合）**。它的白话解释是：总参数很多，但每个 token 只激活其中一小部分参数。这样可以把“训练容量”和“推理活跃参数”部分解耦。对 MoE 来说，简单的 dense 模型 20x 经验不能直接照搬，因为你要区分：

- 总参数
- 激活参数
- 每 token 实际消耗的 FLOPs

第四种替代思路是 **数据受限场景**。如果高质量语料就是不够，那么你不能靠公式“要求”出更多 token。此时更现实的路线是：

- 降低参数量，保证已有数据能训满
- 提高数据质量，而不是只堆数量
- 使用合成数据或蒸馏，但要单独评估有效性

因此，判断是否偏离 20x 时，可以按这个边界表看：

| 场景 | 是否应严格贴近 20x | 原因 |
|---|---|---|
| dense 基座预训练，目标是最低 loss | 是，优先作为基线 | 最符合 Chinchilla 假设 |
| 数据严重不足 | 否，可能被迫偏离 | 上界先由数据供给决定 |
| 推理成本极高 | 否，常偏向更小模型 | 生命周期成本改变目标 |
| MoE/稀疏模型 | 否，需重算有效参数 | dense 公式不能直接套 |
| 小规模研究复现 | 不一定 | 指数和常数更容易漂移 |

工程上最稳妥的态度不是背诵“20 倍法则”，而是先问三个问题：

1. 我优化的是训练 loss，还是总拥有成本？
2. 我统计的是全参数，还是部分参数？
3. 我的有效 token 真的足够支撑这个参数量吗？

如果这三个问题没答清，任何“最优配比”都可能只是表面精确。

---

## 参考资料

1. Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022.  
2. Emergent Mind, *Compute-Optimal Dataset Sizes* 主题综述。  
3. Pearce & Song, *Reconciling Kaplan and Chinchilla Scaling Laws*, 2024.  
4. Chinchilla 与 Gopher 配比和 benchmark 的工程复盘资料。  
5. 关于神经网络 scaling law 的综述性资料与公式整理。
