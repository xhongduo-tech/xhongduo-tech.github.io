## 核心结论

Chinchilla 最优解只回答一个问题：在固定训练 FLOPs 下，怎样把训练 loss 做到最低。它没有回答另一个更接近生产的问题：模型上线后要跑多少推理 token，整条生命周期的总成本是多少。

Sardana 等人在 2024 年 ICML 论文《Beyond Chinchilla-Optimal》中把这个缺口补上了。核心目标从“最小化训练成本”改成“在达到目标 loss $\ell$ 的前提下，最小化训练 + 推理总 FLOPs”：

$$
(N^\*, D_{tr}^\*)=\arg\min_{L(N,D_{tr})=\ell}\left(6ND_{tr}+2NT_{inf}\right)
$$

这里：

- $N$ 是参数量，也就是模型大小；白话说，就是模型里要存多少权重。
- $D_{tr}$ 是训练 token 数；白话说，就是预训练时喂进去多少文本。
- $T_{inf}$ 是模型生命周期内预计要生成或处理的推理 token 总量；白话说，就是上线后用户总共会“消耗”它多少文本吞吐。

结论非常直接：

| 目标 | 参数量 | 训练 Token | 生命周期 FLOPs |
|---|---:|---:|---:|
| Chinchilla 最优 | 70B | 4.26T | $15.8\times10^{24}$ |
| 推理感知最优 | 23.5B | 24.4T | $8.14\times10^{24}$ |

这是 ICLR 2025 论文《The Journey Matters》附录复现的同一组对比：目标训练 loss 为 $1.89$、预期推理量为 $100T$ token。结论不是“训得更少”，而是“模型更小，但训练更久”。总生命周期 FLOPs 约下降一半。

这也是为什么 LLaMA 系列明显偏离 Chinchilla 的约 $20$ token/参数。典型例子是 Llama 2 7B 约 $284$ token/参数，Llama 3 8B 约 $1875$ token/参数。它们不是“没按最优训练”，而是在按另一种更接近部署现实的最优训练。

---

## 问题定义与边界

先把问题说清楚。我们讨论的不是“如何在同样训练预算下拿到最低 loss”，而是：

> 已知你希望模型达到某个目标质量 $\ell$，并且大致能预估上线后会跑多少推理 token，应该选多大的模型、训多少 token，才能让总算力成本最低。

经典 Chinchilla 的损失模型通常写成：

$$
L(N,D_{tr})=E + A N^{-\alpha} + B D_{tr}^{-\beta}
$$

含义是：

- $E$ 是不可消除误差，可以理解为“这条任务曲线的地板”。
- $A N^{-\alpha}$ 是参数不足带来的损失项。模型越小，这一项越大。
- $B D_{tr}^{-\beta}$ 是数据不足带来的损失项。训练 token 越少，这一项越大。

Chinchilla 只最小化训练 FLOPs，大致是 $6ND_{tr}$。而推理感知版本把推理 FLOPs 加进来：

$$
C_{life}=6ND_{tr}+2NT_{inf}
$$

为什么是这个形式：

- 训练每个 token 需要前向、反向和梯度更新，经验上可近似为 $6N$ FLOPs/token。
- 推理每个 token 只做前向，经验上可近似为 $2N$ FLOPs/token。

所以，训练成本随 $ND_{tr}$ 增长，推理成本随 $NT_{inf}$ 增长。只要 $T_{inf}$ 很大，参数量 $N$ 就会成为强约束。

这件事的边界也要讲清楚：

| 场景 | 是否应优先考虑推理感知最优 |
|---|---|
| 面向大量用户的聊天、搜索、Copilot、Agent 服务 | 是 |
| 内部低频使用模型，生命周期短 | 未必 |
| 学术实验，只关注训练阶段指标 | 未必 |
| 离线批处理、推理量远小于训练量 | Chinchilla 仍有意义 |

真实工程里，Meta 的路线已经说明问题。Llama 3 官方说明 8B 和 70B 模型都在最高约 15T token 上继续收益；这远高于 Chinchilla 约 20 token/参数的训练点。换句话说，业界已经在主动放弃“训练最优”，换取“部署总成本更优”。

---

## 核心机制与推导

机制可以压缩成一句话：

> 推理 token 总量越大，最优模型越应该小；模型变小后损失会变差，只能靠更多训练 token 把质量补回来。

从约束式开始：

$$
L(N,D_{tr})=\ell
$$

代入损失函数，可以把 $D_{tr}$ 写成 $N$ 的函数：

$$
D_{tr}(N)=\left(\frac{B}{\ell - E - A N^{-\alpha}}\right)^{1/\beta}
$$

只要右边有定义，也就是 $\ell > E + A N^{-\alpha}$，说明“这个模型大小还能通过多喂数据达到目标 loss”。于是总成本变成单变量问题：

$$
C(N)=6N\cdot D_{tr}(N)+2NT_{inf}
$$

现在看两个方向的力：

| 项 | 作用 | 直观解释 |
|---|---|---|
| $2NT_{inf}$ | 压低 $N$ | 每次推理都要反复支付，推理量越大越怕大模型 |
| $A N^{-\alpha}$ | 抬高 $N$ | 模型太小会损失表达能力 |
| 通过增大 $D_{tr}$ 补偿 | 允许更小的 $N$ | 多训练数据是一次性成本，可换更便宜的长期推理 |

这就是“过训练”的本质。过训练不是把模型白白多训几轮，而是故意选一个比 Chinchilla 更小的模型，然后用远高于 Chinchilla 比例的数据把 loss 拉回目标线。

### 玩具例子

假设你要做一个客服模型，目标是固定 loss 不变。现在有两种方案：

| 方案 | 模型参数 | 训练 token | 单 token 推理成本 |
|---|---:|---:|---:|
| 大模型 | 60B | 5T | 高 |
| 小模型过训练 | 20B | 20T | 低 |

如果你的产品一生只跑 1T token，那么第二种方案不一定划算，因为多出来的训练太贵。

但如果你的产品一生要跑 100T token，那么每个推理 token 都便宜 3 倍，累计节省会远超额外训练成本。推理感知 scaling law 讨论的就是这个拐点。

### 真实工程例子

设想一个日请求量很高的企业问答系统，全年累计处理 $10^{12}$ token，且要持续多年。你按 Chinchilla 思路训练出 70B 模型，训练阶段也许“很优”，但上线后问题会立刻出现：

- 显存压力高，副本数上不去。
- 首 token 延迟更难压。
- 同样吞吐下需要更多 GPU。
- 每多服务一天，都在重复支付大模型推理成本。

如果换成更小的 20B 级模型，并通过更多高质量 token 预训练、蒸馏、量化把效果补回来，训练阶段确实更重，但部署期会轻很多。LLaMA 系列选择长训练、小模型、强调推理效率，本质就是在顺着这条逻辑走。

---

## 代码实现

下面给一个可以直接运行的 Python 版本。它不复现论文拟合常数，而是演示“如何在固定目标 loss 下搜索总 FLOPs 最小的 $(N,D_{tr})$”。代码里输入了题目要求的 $\ell=1.89,\ T_{inf}=100T$，但系数是玩具系数，因此输出只用于理解流程，不用于复现实验表格。

```python
import math

def loss_fn(N, D, E, A, alpha, B, beta):
    return E + A * (N ** (-alpha)) + B * (D ** (-beta))

def required_tokens_for_target_loss(N, target_loss, E, A, alpha, B, beta):
    gap = target_loss - E - A * (N ** (-alpha))
    if gap <= 0:
        return None
    return (B / gap) ** (1.0 / beta)

def lifetime_flops(N, D_tr, T_inf):
    # 训练近似 6ND，推理近似 2NT_inf
    return 6.0 * N * D_tr + 2.0 * N * T_inf

def solve_inference_aware(
    target_loss,
    T_inf,
    E, A, alpha, B, beta,
    N_min,
    N_max,
    num_points=2000,
):
    best = None
    for i in range(num_points):
        # 在对数空间均匀扫描 N，更适合跨数量级搜索
        ratio = i / (num_points - 1)
        N = N_min * ((N_max / N_min) ** ratio)
        D_tr = required_tokens_for_target_loss(N, target_loss, E, A, alpha, B, beta)
        if D_tr is None:
            continue
        cost = lifetime_flops(N, D_tr, T_inf)
        item = {"N": N, "D_tr": D_tr, "cost": cost}
        if best is None or item["cost"] < best["cost"]:
            best = item
    return best

def pretty_tokens(x):
    if x >= 1e12:
        return f"{x / 1e12:.2f}T"
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    return f"{x:.2f}"

# 玩具系数：只用于展示求解流程
E = 1.50
A = 120.0
alpha = 0.34
B = 900.0
beta = 0.28

target_loss = 1.89
T_inf = 100e12  # 100T

best = solve_inference_aware(
    target_loss=target_loss,
    T_inf=T_inf,
    E=E, A=A, alpha=alpha, B=B, beta=beta,
    N_min=1e8,     # 0.1B
    N_max=2e11,    # 200B
    num_points=3000,
)

assert best is not None
assert best["N"] > 0
assert best["D_tr"] > 0
assert abs(loss_fn(best["N"], best["D_tr"], E, A, alpha, B, beta) - target_loss) < 1e-9

# 再验证：当 T_inf 增大时，最优 N 应该倾向更小
best_small_inf = solve_inference_aware(
    target_loss, 1e12, E, A, alpha, B, beta, 1e8, 2e11
)
best_large_inf = solve_inference_aware(
    target_loss, 100e12, E, A, alpha, B, beta, 1e8, 2e11
)
assert best_large_inf["N"] < best_small_inf["N"]

print("Best N:", pretty_tokens(best["N"]))
print("Best D_tr:", pretty_tokens(best["D_tr"]))
print("Best lifetime FLOPs:", f"{best['cost']:.3e}")
```

如果你要把它改成更接近论文的工程版本，流程通常是：

| 步骤 | 做什么 |
|---|---|
| 1 | 用已有训练实验拟合 $E,A,\alpha,B,\beta$ |
| 2 | 根据业务预估 $T_{inf}$ |
| 3 | 设定目标 loss $\ell$ 或目标 benchmark 水平 |
| 4 | 搜索满足 $L(N,D)=\ell$ 的所有可行点 |
| 5 | 选总成本 $6ND+2NT_{inf}$ 最小的点 |

伪代码可以写成：

```text
for N in candidate_model_sizes:
    D = solve L(N, D) = target_loss
    if feasible:
        total = 6*N*D + 2*N*T_inf
choose argmin(total)
```

这套方法对新手最重要的启发是：先估算产品会跑多少推理 token，再决定模型大小，而不是先拍脑袋定一个“看起来先进”的参数规模。

---

## 工程权衡与常见坑

推理感知最优不是一句“用小模型”就结束了，真正落地时有很多工程权衡。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只看训练 FLOPs | 训练看起来省，部署反而爆成本 | 先估算 $T_{inf}$，用 $6ND+2NT_{inf}$ 重算 |
| 把 Chinchilla 比例当硬规则 | 模型偏大，推理延迟难压 | 把 token/参数当可调旋钮，而不是教条 |
| 忽视长尾推理量 | 预算只覆盖首发阶段 | 估生命周期总 token，不只看上线首月 |
| 过训练但数据质量差 | token 变多，收益不成比例 | 先做数据去重、清洗、课程化采样 |
| 只缩小模型，不做量化/蒸馏 | 推理成本仍偏高 | 小模型 + 量化 + 蒸馏一起上 |
| 以训练 loss 直接替代产品效果 | 线上指标不稳定 | 加上任务指标、延迟、吞吐联合评估 |

一个典型误区是：团队看到 70B 模型 benchmark 更好，就直接按大模型路线推进。结果一上线才发现首 token 延迟、并发数、显存占用都不达标，最后又临时做蒸馏和量化，时间成本更高。

如果产品预计每天处理 $10^{12}$ token 级别的请求，按 Chinchilla 训练一个 70B 大模型，长期推理成本会非常难看。反过来，选择类似论文中 23.5B + 更长训练的路线，再叠加 4-bit/8-bit 量化、KV cache 优化、批处理和 speculative decoding，常常更接近真实最优。

要注意另一个边界：推理感知 scaling law 优化的是“总 FLOPs”，不是“唯一正确答案”。现实里还有很多论文公式没完全覆盖的变量，比如：

- 不同硬件上训练和推理的利用率不同。
- Prefill 和 decode 的成本结构不同。
- 量化会改变实际每 token 成本。
- 小模型在某些复杂任务上可能出现能力断层，不能只看 loss。

所以这条 law 更像“系统级一阶近似”，非常有用，但不能代替完整容量规划。

---

## 替代方案与适用边界

推理感知最优不是唯一思路，至少还有两类常见替代方案。

### 1. 稀疏化或 MoE

MoE 是混合专家模型，白话说，就是总参数很多，但每次只激活其中一小部分。这样可以把“模型容量”和“单次推理成本”部分解耦。

ICLR 2025 的《The Journey Matters》进一步说明：如果把“训练期间平均激活参数量”纳入 scaling law，稀疏预训练可以在相近训练质量下，得到更小的最终推理模型。它和推理感知最优不是冲突关系，而是可叠加关系。

### 2. 继续使用 Chinchilla

如果你的场景满足下面条件，Chinchilla 仍然合理：

- 推理生命周期很短。
- 主要目标是尽快完成一次训练实验。
- 模型主要离线跑，不需要大规模在线服务。
- $T_{inf}$ 相对训练 token 不大，推理项 $2NT_{inf}$ 不是主导项。

可以用一个简单决策表判断：

| 场景 | 建议策略 |
|---|---|
| $T_{inf}$ 很小，模型几乎不上线 | Chinchilla 优先 |
| $T_{inf}$ 与训练 token 同量级 | 两种方案都评估 |
| $T_{inf}$ 远大于训练 token | 推理感知最优优先 |
| 对推理吞吐、显存、延迟极敏感 | 推理感知 + 量化/蒸馏/MoE |
| 有足够高质量数据，但部署预算紧 | 小模型过训练 |

一个实用判断是：如果你怀疑自己未来会服务海量请求，就不要把参数量只当“能力杠杆”，还要把它当“长期账单倍率”。在这种条件下，小模型过训练通常比大模型欠训练更接近工程最优。

---

## 参考资料

| 来源 | 时间 | 核心贡献 |
|---|---|---|
| Sardana, Portes, Doubov, Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*. ICML 2024. https://proceedings.mlr.press/v235/sardana24a.html | 2024-07 | 首次把推理成本显式加入 scaling law，提出固定目标 loss 下最小化 $6ND_{tr}+2NT_{inf}$ 的框架。 |
| Graphcore Research Blog. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*. https://graphcore-research.github.io/beyond-chinchilla/ | 2024-01-29 | 用更工程化的语言解释为什么高推理需求下应“训更小、训更久”，并指出 Llama 式 overtraining 的部署动机。 |
| Jonas Vetterle. *A brief history of LLM Scaling Laws and what to expect in 2025*. https://www.jonvet.com/blog/llm-scaling-in-2025 | 2024-12 | 归纳了 Chinchilla 之后的 overtraining 趋势，给出 Llama 2、Llama 3 等 token/参数比例的直观比较。 |
| Jin et al. *The Journey Matters: Average Parameter Count over Pre-training Unifies Sparse and Dense Scaling Laws*. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/4b96695d9885f038110b8b16ef50e882-Paper-Conference.pdf | 2025-04 | 说明平均参数量可统一稀疏与稠密预训练 scaling law，并在附录复现 Chinchilla 与 Beyond Chinchilla 在 $\ell=1.89,\ T_{inf}=100T$ 下的数值对比。 |
| Meta. *Introducing Meta Llama 3*. https://about.fb.com/news/2024/04/meta-llama-3/ | 2024-04-18 | 官方给出 Llama 3 使用超过 15T token 预训练，并强调小模型在推理效率上的优势。 |
| FourWeekMBA. *The Chinchilla Correction: How "Train Longer, Not Bigger" Changed Everything*. https://fourweekmba.com/the-chinchilla-correction-how-train-longer-not-bigger-changed-everything/ | 2026-02-22 | 面向商业和工程读者总结“训练更久而不是更大”的产业含义，便于理解 Llama 路线为何偏离 Chinchilla。 |
