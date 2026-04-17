## 核心结论

模型合并指的是：**直接在参数空间操作多个模型的权重**，把多个微调模型的能力组合到一个模型里，而不是重新拿训练数据再训一遍。参数空间可以先理解成“模型所有参数构成的高维坐标空间”。

最基础的 baseline 是线性平均：

$$
W_{\text{merge}}=\frac{1}{N}\sum_{i=1}^N W_i
$$

它实现简单，但常常效果不稳定。原因不是“平均”这个动作本身错了，而是不同微调模型在参数空间里的更新方向可能互相抵消，或者把某些层的范数拉偏。范数可以先理解成“向量长度”，它大致反映了参数更新的整体幅度。

更实用的三类方法分别解决不同问题：

1. **Task Arithmetic**：先算每个任务相对 base 的“位移”，再把这些位移加回去。
2. **SLERP**：不是走直线，而是在球面上沿弧线插值，尽量保持参数向量的长度结构。
3. **TIES / TIES-Merging**：先剪掉不重要的微小变化，再处理符号冲突，减少不同任务互相打架。

一个适合新手的玩具例子是二维向量。把 base 看成坐标系中的一个点，任务微调就是从这个点往某个方向走一小步。Task Arithmetic 直接把几段位移相加；SLERP 则在球面上走圆弧；TIES 先删掉“抖动”再加。它们都在做“参数合成”，但保护的性质不同。

下表可以先建立全局判断：

| 目标 | Task Arithmetic | SLERP | TIES | Slerp-Opt |
|---|---|---|---|---|
| 保持范数 | 弱 | 强 | 中 | 强 |
| 处理方向/符号冲突 | 弱 | 弱 | 强 | 中 |
| 是否需要 base 模型 | 是 | 常见场景下是 | 是 | 常见场景下是 |
| 资源消耗 | 低 | 低 | 中 | 中到高 |
| 可解释性 | 强 | 中 | 中 | 低到中 |
| 多专家合并稳定性 | 中 | 低到中 | 高 | 高 |
| 典型用途 | 多任务增量叠加 | 两模型平滑插值 | 冲突明显的多任务合并 | 分层、动态权重优化 |

结论可以压缩成一句话：**任务方向一致时优先用 Task Arithmetic，想保持范数时用 SLERP，任务冲突明显时优先用 TIES，多层 LoRA 专家精细融合时再考虑 Slerp-Opt。**

---

## 问题定义与边界

设一个基础模型为 $W_{\text{base}}$，多个微调后的专家模型为 $W_1,W_2,\dots,W_k$。模型合并的目标不是让新模型“平均长得像这些模型”，而是让它**同时保留 base 的通用能力，并复用多个任务专家的增量能力**。

这里的“增量能力”指的是：微调后参数相对 base 发生的变化，通常记作任务向量：

$$
\tau_j = W_j - W_{\text{base}}
$$

白话讲，$\tau_j$ 就是“这个任务把 base 朝哪个方向推了多远”。

问题边界很重要：

| 边界条件 | 含义 |
|---|---|
| 同架构 | 只能合并参数形状完全对应的模型 |
| 同 base 更稳 | 多个专家最好来自同一个 base |
| 无需训练数据 | 只看权重差，不依赖原始训练样本 |
| 仍需验证集 | 合并后必须调 $\lambda$、阈值、层权重 |
| 不是万能压缩 | 合并不等于自动得到更强模型 |
| 可能丢 base 能力 | 合并后常见副作用是通用能力下降 |

可以把它理解成一条路网问题。base 是起点，两个专家模型是从同一个起点走向不同方向的终点。模型合并不是重新修路，而是只看“每条路径的位移向量”，尝试构造一条兼顾多个终点优势的新路径。

一个简化符号图可以这样读：

- 起点：$W_{\text{base}}$
- 专家 A：$W_1 = W_{\text{base}} + \tau_1$
- 专家 B：$W_2 = W_{\text{base}} + \tau_2$
- 合并目标：$W_{\text{merge}} = W_{\text{base}} + f(\tau_1,\tau_2,\dots)$

因此，核心难点不是“如何相加”，而是：

1. 哪些更新是真正有用的任务信号。
2. 哪些更新只是噪声或局部抖动。
3. 不同任务在同一参数位置是否发生方向冲突。
4. 合并后是否破坏了基础模型原本稳定的尺度结构。

---

## 核心机制与推导

### 1. Task Arithmetic

Task Arithmetic 最直接。它把每个微调模型转换成任务向量，再做线性组合：

$$
\tau_j = W_j - W_{\text{base}}
$$

$$
W_{\text{merge}} = W_{\text{base}} + \lambda \sum_{j=1}^{m}\tau_j
$$

其中 $\lambda$ 是缩放系数，用来控制“把任务能力加回去多少”。如果 $\lambda=1$，表示全量加回；如果 $\lambda<1$，表示保守合并。

玩具例子如下：

- $W_{\text{base}}=[1,0]$
- $W_1=[1.3,0.1]$
- $W_2=[0.9,-0.2]$

则：

$$
\tau_1 = [1.3,0.1]-[1,0]=[0.3,0.1]
$$

$$
\tau_2 = [0.9,-0.2]-[1,0]=[-0.1,-0.2]
$$

取 $\lambda=0.5$：

$$
W_{\text{merge}}=[1,0]+0.5([0.3,0.1]+[-0.1,-0.2])=[1.1,-0.05]
$$

这个式子好理解，但问题也直接暴露出来：如果两个任务在同一维上方向相反，就会抵消；如果某个任务更新幅度过大，也会把整体拖偏。

### 2. SLERP

SLERP 是 spherical linear interpolation，中文常译为球面线性插值。白话解释：**不是在两点之间走直线，而是在单位球面上走最短弧线**。它的主要价值是保持插值过程的几何结构，特别是范数更稳定。

先归一化：

$$
v_i = \frac{W_i}{\|W_i\|}
$$

设两个向量夹角为 $\theta$，则：

$$
\theta = \arccos\left(\frac{v_1 \cdot v_2}{\|v_1\|\|v_2\|}\right)
$$

SLERP 公式为：

$$
W_{\text{slerp}}=
\frac{\sin((1-t)\theta)}{\sin\theta}W_{\text{base}}+
\frac{\sin(t\theta)}{\sin\theta}W_{\text{expert}}
$$

其中 $t\in[0,1]$。当 $t=0$ 时得到 base，当 $t=1$ 时得到 expert。

它适合两模型平滑过渡，尤其在“直接线性插值会让长度结构失真”的场景更有价值。缺点也明确：它本质上更偏向“两者之间找平衡”，不天然解决多任务冲突。

### 3. TIES / TIES-Merging

TIES 的核心不是新的几何路径，而是**合并前先处理冲突**。通常包含两步：

1. **Trim**：剪枝，删掉变化幅度很小的参数更新。
2. **Elect Sign**：在同一位置上确定主要符号方向。
3. **Merge**：只保留共识方向，再加回 base。

这里的“符号冲突”可以理解为：某个参数位置上，一个任务想增大，另一个任务想减小。如果直接加，结果可能接近 0，但这个 0 不是“合理折中”，而是“冲突后的相互抵消”。

一个最小例子：

- 位置 $i$ 上，$\tau_1^{(i)}=+0.08$
- 位置 $i$ 上，$\tau_2^{(i)}=-0.07$
- 位置 $i$ 上，$\tau_3^{(i)}=+0.09$

如果直接相加，结果是 $+0.10$，但如果还有一堆接近 0 的噪声更新，就会把真正有共识的方向稀释。TIES 会先剪掉小幅波动，再按多数或幅度共识选符号，使这个位置最终保留“主要往正方向推”的结论。

### 4. Slerp-Opt 的工程意义

Slerp-Opt 可以理解为“把 SLERP 从全模型级别推进到更细粒度的层级权重优化”。在真实工程里，LoRA 专家模型常常不是每层都同样重要。某些层更像在处理语义路由，某些层更像在处理风格或格式控制。

因此，Slerp-Opt 的思路不是统一用一个 $t$，而是按层、按模块甚至按 LoRA adapter 的重要性分配不同权重。这样能减少“全局一个比例把所有层都混在一起”的粗糙性。

---

## 代码实现

下面的代码用 `numpy` 演示 Task Arithmetic、SLERP 和一个简化版 TIES 预处理。它是可运行的玩具实现，不依赖深度学习框架，但足够说明参数空间操作的核心步骤。

```python
import numpy as np

def merge_task_arithmetic(base, experts, lam=1.0):
    base = np.array(base, dtype=float)
    experts = [np.array(x, dtype=float) for x in experts]
    task_vectors = [w - base for w in experts]
    merged = base + lam * np.sum(task_vectors, axis=0)
    return merged, task_vectors

def slerp(base, expert, t=0.5, eps=1e-8):
    base = np.array(base, dtype=float)
    expert = np.array(expert, dtype=float)

    nb = np.linalg.norm(base)
    ne = np.linalg.norm(expert)
    assert nb > eps and ne > eps

    vb = base / nb
    ve = expert / ne

    dot = np.clip(np.dot(vb, ve), -1.0, 1.0)
    theta = np.arccos(dot)

    if abs(theta) < eps:
        # 方向几乎一致时退化为线性插值
        out = (1 - t) * base + t * expert
        return out

    # 保持球面路径，避免直接线性插值造成的范数结构偏移
    coeff1 = np.sin((1 - t) * theta) / np.sin(theta)
    coeff2 = np.sin(t * theta) / np.sin(theta)

    # 这里用单位方向插值，再恢复平均尺度
    direction = coeff1 * vb + coeff2 * ve
    scale = (1 - t) * nb + t * ne
    return direction * scale

def ties_preprocess(task_vectors, threshold=0.05):
    tv = np.array(task_vectors, dtype=float)

    # 1. Trim: 小于阈值的更新视为噪声
    pruned = np.where(np.abs(tv) >= threshold, tv, 0.0)

    # 2. Sign consensus: 统计每个位置的主导方向
    signs = np.sign(pruned)
    sign_score = np.sum(signs, axis=0)
    consensus = np.sign(sign_score)

    # 3. 只保留与共识符号一致的更新
    aligned = np.where(np.sign(pruned) == consensus, pruned, 0.0)
    merged_task = np.sum(aligned, axis=0)
    return merged_task, pruned, consensus

# 玩具例子
base = np.array([1.0, 0.0])
w1 = np.array([1.3, 0.1])
w2 = np.array([0.9, -0.2])

merged_ta, task_vectors = merge_task_arithmetic(base, [w1, w2], lam=0.5)
assert np.allclose(merged_ta, np.array([1.1, -0.05]))

mid_slerp = slerp(base, w1, t=0.5)
assert mid_slerp.shape == base.shape
assert np.linalg.norm(mid_slerp) > 0

merged_task_ties, pruned, consensus = ties_preprocess(task_vectors, threshold=0.05)
merged_ties = base + 0.5 * merged_task_ties
assert merged_ties.shape == base.shape

print("Task Arithmetic:", merged_ta, "norm=", np.linalg.norm(merged_ta))
print("SLERP(base, w1, 0.5):", mid_slerp, "norm=", np.linalg.norm(mid_slerp))
print("TIES merged:", merged_ties, "norm=", np.linalg.norm(merged_ties))
```

这段代码对应的目的很明确：

| 代码块 | 作用 |
|---|---|
| `merge_task_arithmetic` | 计算任务向量并线性加回 base |
| `slerp` | 在两组参数间做球面插值，尽量保持几何结构 |
| `ties_preprocess` | 先剪枝，再做符号一致性过滤 |

如果放到真实工程里，参数不再是一个小向量，而是很多层的张量字典，例如：

- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.self_attn.k_proj.weight`
- `...`

此时的流程通常是：

1. 读取 `base state_dict`
2. 读取多个专家 `state_dict`
3. 对每个参数名分别计算任务向量
4. 逐层执行 Task Arithmetic、SLERP 或 TIES
5. 写回新的 `state_dict`
6. 在验证集上评估

真实工程例子可以用多 LoRA 专家来理解。假设你有一个 base LLM，再有三个 LoRA：

- 数学推理 LoRA
- SQL 生成 LoRA
- 安全拒答 LoRA

如果直接把 LoRA 增量线性相加，常见问题是数学 LoRA 抬高了某些层的输出幅度，安全 LoRA 又在同一层压制某些模式，最后在推理题和普通问答上都退化。更稳的做法是：

1. 先把每个 LoRA 转为相对 base 的增量。
2. 对冲突明显的层做 TIES。
3. 对关键层用分层权重，接近 Slerp-Opt 的思路。
4. 单独检查 base 能力是否被破坏，例如常识问答、格式遵循、拒答边界。

---

## 工程权衡与常见坑

最常见的误区是把模型合并理解成“权重平均一下就行”。实际上，参数空间里的每一维都可能承担不同功能，不同任务对同一维的更新方向可能完全相反。

下表是工程里最常见的问题与规避方式：

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| 直接平均后性能下降 | 多任务都没明显提升，base 能力也掉 | 先做 Task Arithmetic，不直接平均全量权重 |
| 符号冲突 | 某些指标剧烈退化 | 用 TIES 的 sign consensus |
| 小幅噪声过多 | 合并后不稳定、结果漂移大 | 先阈值剪枝，只保留显著更新 |
| 范数不稳定 | 输出分布异常、生成风格突变 | 用 SLERP 或分层缩放 |
| $\lambda$ 过大 | 专家能力增强但基础能力崩 | 在验证集上扫 $\lambda$ 曲线 |
| 多层统一权重过粗 | 某些任务提升，另一些明显退化 | 分层加权，关键层单独调参 |
| 不同 base 强行合并 | 结果不可控 | 尽量只合并同源 base 的专家模型 |

一个很实用的新手实验是：固定同一组合并方案，只改 $\lambda$。

- 当 $\lambda=0.5$ 时，增量较保守，base 能力通常保得更好。
- 当 $\lambda=1.0$ 时，任务能力更强，但冲突和偏移会被放大。
- 当 $\lambda>1.0$ 时，常见现象是局部任务更强，但全局稳定性快速恶化。

因此调参建议可以直接写成操作清单：

1. 先跑 `Task Arithmetic + λ∈{0.2,0.4,0.6,0.8,1.0}`。
2. 记录任务指标和 base 指标，不只看单任务提升。
3. 如果出现明显冲突，再加 TIES 阈值扫描，例如 `threshold∈{1e-4,5e-4,1e-3}`。
4. 如果输出尺度异常，再尝试 SLERP 或对高敏感层单独缩放。
5. LoRA 场景优先按层验证，不要假设所有层适合相同权重。

一个真实工程上的判断标准是：**合并是否“值得”不看最高分，而看 Pareto 改善**。也就是在多个任务一起看时，是否能在不显著伤害 base 的前提下提升整体能力。如果数学题涨了 4 分，但通用问答掉了 8 分，这通常不是可上线的合并。

---

## 替代方案与适用边界

模型合并不是唯一方案。它的优势是无需原始数据、成本低、部署快；但如果任务差异很大，或者你能访问训练数据，重新做联合训练、继续指令微调，往往上限更高。

下面给出常见方案对比：

| 方案 | 适用边界 | 代价 |
|---|---|---|
| 线性平均 | 同类任务、差异小、只求快速试验 | 最低，但最不稳 |
| Task Arithmetic | 同源 base、任务向量方向较一致 | 低 |
| SLERP | 两模型插值、对范数稳定敏感 | 低 |
| TIES | 多任务冲突明显、希望稳健合并 | 中 |
| Slerp-Opt | 多层 LoRA 专家、愿意做细粒度调参 | 中到高 |
| LoRA 运行时路由 | 不想真正合并，只想按请求切换专家 | 推理链路更复杂 |
| 继续联合微调 | 有训练数据、追求更高上限 | 最高 |

一个新手可执行的决策树可以这样记：

1. 如果只是两个很相近的专家，先试线性平均或 Task Arithmetic。
2. 如果发现输出尺度异常或风格漂移，改试 SLERP。
3. 如果是三个以上专家，且指标互相拉扯，优先上 TIES。
4. 如果你合并的是多个 LoRA，且不同层的重要性明显不同，再考虑 Slerp-Opt。
5. 如果所有方法都需要大量手调，而且你手里有数据，直接继续微调通常更省总成本。

需要特别强调一个边界：**Task Arithmetic 有一个隐含前提，即任务向量携带的是“可加的能力”而不是“互斥的偏置”**。例如“数学格式强化”和“医学问答”可能还可以部分共存；但“极度口语化风格”和“严格法律文书风格”在同一位置上可能就是互斥更新。此时单纯相加通常不如路由式专家系统。

---

## 参考资料

1. Ilharco et al., *Editing Models with Task Arithmetic*, 2023。提出任务向量框架，用 $W_{\text{ft}}-W_{\text{base}}$ 表示任务增量，是参数空间合并的基础参考。
2. Yadav et al., *TIES-Merging: Resolving Interference When Merging Models*, 2023。核心贡献是剪枝与符号一致性处理，用于缓解多任务冲突。
3. Shoemake, *Animating Rotation with Quaternion Curves*, 1985。SLERP 的经典来源，虽然最早用于旋转插值，但其球面插值思想被广泛借到参数空间讨论中。
4. 关于模型合并与参数空间操作的综述文章，2025 年前后已有多篇总结，关注点通常包括线性平均、任务向量、几何插值和冲突消解。
5. Jiang et al., 2025，与 Slerp-Opt 相关工作。核心思路是把球面插值推广到更细粒度的层级或模块级权重优化，特别适合 LoRA 专家合并场景。
6. 工程实践中可进一步阅读 LoRA merge、adapter composition、model soups 等方向，它们和本文的共同点都是“尽量不重新训练，通过参数或模块组合复用已有能力”。
