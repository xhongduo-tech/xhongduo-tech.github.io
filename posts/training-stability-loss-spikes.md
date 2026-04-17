## 核心结论

训练稳定性不是“把学习率调小”这么简单。大规模预训练里常见的 loss spike，本质上往往是一次局部更新过大：某些参数的真实梯度突然变大，但 Adam 的二阶矩估计 $v_t$ 还停留在旧尺度，导致预条件器，也就是“按历史梯度大小自动缩放步长的机制”，暂时失真，更新被放大。若此时预条件后的曲率超过稳定阈值，训练就会从平滑下降切到震荡，严重时直接发散。

把结论写成公式，Adam 的核心更新是
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$
其中：

| 符号 | 含义 | 新手直观理解 |
|---|---|---|
| $g_t$ | 当前梯度 | 这一步“往哪改、改多大” |
| $m_t$ | 一阶矩估计 | 最近几步的平均方向 |
| $v_t$ | 二阶矩估计 | 最近几步的梯度尺度 |
| $\eta$ | 学习率 | 全局油门大小 |
| $\epsilon$ | 数值稳定项 | 防止分母过小 |

当 $\hat v_t$ 对当前梯度反应过慢时，单步有效步长
$$
\eta_{\text{eff},i}\approx \frac{\eta}{\sqrt{\hat v_{t,i}}+\epsilon}
$$
会在某些坐标上突然变大。此时可以把不稳定理解为：预条件后的 Hessian 在梯度方向上的曲率超过了稳定阈值 $2/\eta$，于是原本应该下降的一步，反而把 loss 推高。近期关于 Adam spike 的分析工作也是沿这个方向解释机制的。

对新手可以这样理解。Adam 像一辆带自动限速器的车，$v_t$ 是“最近路况有多危险”的估计。平时路很平，限速器允许你开快；突然遇到一段急弯，也就是罕见样本带来大梯度，但限速器慢半拍，车先冲出去，于是损失曲线像火箭一样窜上去。

实际工程里，稳定训练通常不是单招，而是四件事配合：合理的 gradient clipping，也就是“梯度过大时只保留方向、压缩幅度”；必要时调低 $\beta_2$ 或单独处理稀疏 embedding；用 $\mu$P 让小模型调出的超参能迁移到大模型；在输出端加 Z-loss 抑制 logit 尺度失控。它们解决的是不同层面的不稳定。

| 策略 | 直接作用点 | 对 loss spike 的影响 | 主要代价 |
|---|---|---|---|
| 降低 $\beta_2$ | 让 $v_t$ 更快跟上当前梯度 | 减少二阶矩迟滞导致的大步长 | 噪声更大，损失更抖 |
| rare embedding 特殊 lr/优化 | 单独抑制稀疏参数异常更新 | 对推荐系统、检索模型很有效 | 实现更复杂，参数组更多 |
| $\mu$P / $\mu$Transfer | 让超参跨模型规模迁移 | 避免扩容后落入新不稳定区 | 需要按规则改初始化与 lr |
| Z-loss | 压制 softmax logit 尺度 | 降低输出端数值爆炸概率 | 只控尺度，不治表示几何问题 |

---

## 问题定义与边界

Loss spike 指训练损失在短时间内出现突发性跳升，常伴随梯度范数、更新范数、logit 幅度同步上升。梯度范数就是“当前一步整体梯度有多大”的数；更新范数是“优化器真正把参数改了多少”的数。两者不一定同步，所以只看 loss 曲线往往不够。

一个实用的观察框架是把异常分成三层：

| 层次 | 监控量 | 常见信号 | 对应问题 |
|---|---|---|---|
| 梯度层 | grad norm、per-layer grad norm | 某一步梯度突然放大 | 数据异常、难样本、长尾 ID |
| 更新层 | update norm、参数增量 | 梯度不大但更新异常大 | Adam 状态失配、$\epsilon$ 过小 |
| 输出层 | logits mean/std/max | logits 尺度飙升 | softmax 数值不稳、输出头漂移 |

这类问题不是所有模型都会明显遇到。它更常见于三类场景：

1. 大规模模型，特别是学习率推得比较高时。
2. 稀疏参数占比高的模型，例如推荐系统中的用户 ID、物品 ID embedding。
3. 扩容时沿用旧超参，但参数化方式没有对齐，导致大模型实际更新尺度变了。

典型触发链条是：

1. 某个罕见样本或罕见 ID 出现在 batch 中。
2. 对应 embedding 行很久没更新，当前梯度突然很大。
3. Adam 的 $\hat v_t$ 因为指数滑动平均而迟滞，仍然偏小。
4. 更新项 $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ 被放大。
5. 局部 logit 或中间激活快速变大，loss 突然抬升。

新手版本的玩具例子可以这样看。假设某个“一天只登录一次”的物品 ID 很久没出现在训练里，这次突然在一个高权重样本里出现。这个 embedding 过去几千步几乎没动，Adam 以为它附近很平稳，于是给出较大的有效步长。结果这一步把该 ID 对应的分数推得过头，接下来 softmax 输出失真，loss 立刻抬头。

梯度裁剪的基本判定式是
$$
\text{if }\|g\|_2>t,\qquad g\leftarrow t\frac{g}{\|g\|_2}
$$
它的意思是：如果梯度总长度超过阈值 $t$，就保持方向不变，把长度缩到 $t$。这能防止一次异常 batch 直接把模型推飞。

再把“梯度大”和“更新大”区分清楚：

$$
\Delta\theta_t=-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

即使 $\|g_t\|$ 没有异常，如果 $\hat v_t$ 小得不合理，$\|\Delta\theta_t\|$ 也可能异常大。所以排查 spike 时，必须同时看 grad norm 和 update norm。

但要注意边界。Loss spike 不是所有训练抖动的总称。数据分布切换、AMP 溢出、坏 checkpoint 恢复、标签污染，也会造成损失跳变。本文讨论的重点是“优化器状态与当前梯度尺度失配”这一类 spike，尤其是 Adam 二阶矩迟滞、稀疏 embedding 主导更新、以及扩容后超参失真这三种情况。

---

## 核心机制与推导

先看 Adam 为什么会慢半拍。$v_t$ 是平方梯度的指数滑动平均，$\beta_2$ 越大，历史记忆越长，响应越慢。若某参数长时间梯度很小，突然出现大梯度 $g_t$，则
$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$
当 $\beta_2=0.999$ 时，当前步只占 $0.1\%$ 权重。也就是说，即便 $g_t^2$ 已经暴涨，$v_t$ 仍会被旧值拖住。此时分母 $\sqrt{\hat v_t}$ 偏小，导致有效步长偏大。

把这个过程写得更具体一点。假设某个坐标过去长期满足 $g^2\approx 10^{-6}$，因此 $v_{t-1}\approx 10^{-6}$。现在突然来了一个大梯度 $g_t=10$，于是 $g_t^2=100$。若 $\beta_2=0.999$，则
$$
v_t\approx 0.999\times 10^{-6}+0.001\times 100\approx 0.100001
$$
分母约为
$$
\sqrt{v_t}\approx 0.316
$$
而当前梯度量级是 $10$。这意味着“当前危险程度”已经很高，但预条件器还没有完全反映这件事，步长仍可能偏大。实际训练里若再叠加动量、残差连接、输出层放大，这一步就可能跨过稳定边界。

为什么这会触发 spike？用二阶近似看一步损失变化：
$$
L(\theta_{t+1})-L(\theta_t)\approx \nabla L(\theta_t)^\top \Delta\theta_t+\frac12\Delta\theta_t^\top H_t\Delta\theta_t
$$
对普通梯度下降，$\Delta\theta_t=-\eta \nabla L(\theta_t)$，可写成
$$
L(\theta_{t+1})-L(\theta_t)\approx -\eta\|\nabla L(\theta_t)\|^2+\frac12\eta^2\nabla L(\theta_t)^\top H_t\nabla L(\theta_t)
$$
当梯度方向上的曲率足够大时，二次项会压过一次下降项，loss 就会上升。对 Adam 这类预条件优化器，更准确的理解是：真正决定稳定性的不是原始 Hessian，而是“预条件后的 Hessian”。近期关于 Adam spike 的分析指出，当预条件 Hessian 在梯度方向上的曲率超过 $2/\eta$ 时，就会进入 spike 区域。

这里有一个重要细节：不是所有大梯度都危险，危险的是“当前梯度已经大了，但分母还按旧世界计算”。这正是 rare embedding 特别容易出事的原因。embedding 是“把离散 ID 映射成连续向量的参数表”。很多行长期不更新，一更新就可能很猛。

梯度裁剪为什么有效？因为它不改方向，只缩步幅。若原始梯度是 $g$，裁剪后是
$$
g_{\text{clip}}=\min\left(1,\frac{t}{\|g\|}\right)g
$$
更新方向保留，异常 batch 的破坏力被限制。玩具例子里，如果某一步 $\|g\|=150$，阈值 $t=1$，那么整步缩成原来的 $1/150$。这不是在“修复优化器”，而是在给最坏情况上保险。

但全局裁剪和模块级裁剪不是一回事。设总梯度由 embedding 和 dense 两部分组成：
$$
g=\begin{bmatrix}g_{\text{emb}}\\ g_{\text{dense}}\end{bmatrix},\qquad
\|g\|_2^2=\|g_{\text{emb}}\|_2^2+\|g_{\text{dense}}\|_2^2
$$
如果 spike 来自 embedding，且 $\|g_{\text{emb}}\|_2\gg \|g_{\text{dense}}\|_2$，那么全局裁剪会把 dense 一起压小。结果是“病灶在 embedding，dense 却跟着停工”。这也是推荐系统中模块级裁剪更有价值的原因。

Z-loss 解决的是另一类稳定性问题。softmax 的分母是
$$
Z=\sum_k e^{o_k}
$$
其中 $o_k$ 是 logits，也就是“分类头输出的未归一化分数”。当 logits 整体漂移或尺度过大时，$\exp(o_k)$ 会非常敏感。Z-loss 先对 logits 做标准化：
$$
\mu=\frac1D\sum_k o_k,\qquad
\sigma^2=\frac1D\sum_k o_k^2-\mu^2,\qquad
z_c=\frac{o_c-\mu}{\sigma}
$$
再定义
$$
L_Z=\frac1a\log\left(1+\exp(a(b-z_c))\right)
$$
这相当于在目标类得分的 Z-score 上加 softplus 惩罚。它的作用不是直接修复 Adam，而是抑制输出层 logit 的整体尺度失控，让 softmax 分母不那么容易炸。

$\mu$P 的位置又不同。$\mu$P，Maximal Update Parametrization，直译是“最大更新参数化”，核心思想是：当模型宽度变大时，初始化尺度和各类参数的学习率必须按统一规则缩放，才能让不同规模模型在训练中保持相近的更新行为。白话说，$\mu$P 追求的是“小模型试出来的最高安全速度，大模型还能照着开”。

如果没有 $\mu$P，常见情况是：小模型上稳定的 lr，扩到大模型后不再稳定，因为不同层、不同形状参数的更新幅度随宽度变化了。$\mu$P 的价值不是神奇地消灭 spike，而是减少“每次扩容都重调一轮”的不确定性，从源头降低落入不稳定超参区域的概率。

真实工程例子是推荐系统。一个 batch 中出现大量稀有用户 ID 和长尾物品 ID，embedding 表的少量行产生巨大的梯度；全局裁剪一开，整个 dense tower 也被一起缩小；下一步 dense 层相当于“白学一轮”，而 embedding 又已经把 logits 拉偏，于是 loss spike 后接若干步震荡。这时只做全局裁剪通常不够，模块级监控与裁剪更关键。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，展示四件事：

1. Adam 二阶矩迟滞。
2. 全局裁剪会连带压缩 healthy module。
3. 模块级裁剪能更精确地限制异常来源。
4. 如何记录 grad norm、update norm 和 logit 尺度。

代码不依赖深度学习框架，直接用标准库即可运行。

```python
import math
from dataclasses import dataclass


@dataclass
class AdamState:
    m: float = 0.0
    v: float = 0.0
    step: int = 0


def adam_step(param, grad, state, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    state.step += 1
    state.m = beta1 * state.m + (1 - beta1) * grad
    state.v = beta2 * state.v + (1 - beta2) * (grad ** 2)

    m_hat = state.m / (1 - beta1 ** state.step)
    v_hat = state.v / (1 - beta2 ** state.step)

    update = -lr * m_hat / (math.sqrt(v_hat) + eps)
    param = param + update
    return param, update, m_hat, v_hat


def l2_norm(values):
    return math.sqrt(sum(v * v for v in values))


def clip_by_global_norm(grads, threshold):
    norm = l2_norm(grads.values())
    if norm <= threshold:
        return dict(grads), norm, 1.0
    scale = threshold / norm
    clipped = {k: v * scale for k, v in grads.items()}
    return clipped, norm, scale


def clip_by_module_norm(grads, thresholds):
    clipped = {}
    norms = {}
    scales = {}
    for name, grad in grads.items():
        norm = abs(grad)
        th = thresholds.get(name, float("inf"))
        scale = 1.0 if norm <= th else th / norm
        clipped[name] = grad * scale
        norms[name] = norm
        scales[name] = scale
    return clipped, norms, scales


def softmax_logits(score_a, score_b):
    m = max(score_a, score_b)
    ea = math.exp(score_a - m)
    eb = math.exp(score_b - m)
    z = ea + eb
    return ea / z, eb / z


def cross_entropy_for_positive(logit_pos, logit_neg):
    p_pos, _ = softmax_logits(logit_pos, logit_neg)
    return -math.log(max(p_pos, 1e-12))


def simulate_one_step(grads, clip_mode):
    params = {"embedding": 0.0, "dense": 0.0}
    states = {"embedding": AdamState(v=1e-8), "dense": AdamState(v=0.04)}

    if clip_mode == "global":
        used_grads, global_norm, scale = clip_by_global_norm(grads, threshold=1.0)
        clip_info = {
            "global_norm": global_norm,
            "global_scale": scale,
        }
    elif clip_mode == "module":
        used_grads, module_norms, module_scales = clip_by_module_norm(
            grads, {"embedding": 1.0, "dense": 5.0}
        )
        clip_info = {
            "module_norms": module_norms,
            "module_scales": module_scales,
        }
    else:
        used_grads = dict(grads)
        clip_info = {}

    updates = {}
    v_hats = {}
    for name in params:
        params[name], updates[name], _, v_hats[name] = adam_step(
            params[name], used_grads[name], states[name]
        )

    # 一个简化的二分类打分：正类 logit = embedding + dense，负类 logit = 0
    logit_pos = params["embedding"] + params["dense"]
    logit_neg = 0.0
    loss = cross_entropy_for_positive(logit_pos, logit_neg)

    return {
        "used_grads": used_grads,
        "updates": updates,
        "update_norm": l2_norm(updates.values()),
        "v_hats": v_hats,
        "logit_pos": logit_pos,
        "loss": loss,
        "clip_info": clip_info,
    }


if __name__ == "__main__":
    # 稀疏 embedding 行突然出现异常大梯度，dense 层梯度正常
    grads = {"embedding": 150.0, "dense": 0.8}

    no_clip = simulate_one_step(grads, clip_mode="none")
    global_clip = simulate_one_step(grads, clip_mode="global")
    module_clip = simulate_one_step(grads, clip_mode="module")

    print("=== raw grads ===")
    print(grads)

    print("\n=== no clipping ===")
    print(no_clip)

    print("\n=== global clipping ===")
    print(global_clip)

    print("\n=== module clipping ===")
    print(module_clip)

    # 可运行断言：全局裁剪会把 dense 一起压缩，模块级裁剪基本保留 dense
    assert abs(global_clip["used_grads"]["dense"]) < abs(grads["dense"])
    assert math.isclose(module_clip["used_grads"]["dense"], grads["dense"], rel_tol=1e-12)

    # embedding 梯度被模块级裁剪到 1.0
    assert math.isclose(module_clip["used_grads"]["embedding"], 1.0, rel_tol=1e-12)

    # 无裁剪时，embedding update 绝对值通常最大
    assert abs(no_clip["updates"]["embedding"]) > abs(no_clip["updates"]["dense"])
```

这段代码可以直接运行，重点看三组输出：

| 观察项 | `none` | `global` | `module` |
|---|---|---|---|
| `used_grads["embedding"]` | 很大 | 被压小 | 被压到 embedding 阈值 |
| `used_grads["dense"]` | 正常 | 也被压小 | 基本不变 |
| `update_norm` | 往往最大 | 下降 | 更可控 |
| `logit_pos` / `loss` | 最易异常 | 有时过于保守 | 通常更平衡 |

如果你是新手，最该盯的不是某一个绝对数值，而是“谁在放大”。例如：

1. `grad norm` 很大，说明是输入或样本把梯度推高了。
2. `grad norm` 不大但 `update norm` 很大，说明更像优化器状态问题。
3. `logit_pos` 突然偏离历史区间，说明输出层已经被拉歪。

在真实训练代码里，建议至少做四件监控：

| 监控项 | 为什么要看 | 典型异常信号 |
|---|---|---|
| global grad norm | 判断是否整体爆炸 | 突然放大 3 到 10 倍 |
| per-module grad norm | 定位是 embedding 还是 dense 出问题 | embedding 远高于其历史分位数 |
| update norm | 区分“梯度大”还是“优化器放大了更新” | grad 正常但 update 异常大 |
| logits mean/std/max | 判断输出层是否在漂移 | std、max 同时上升 |

工程实现的伪代码可以写成：

```python
for batch in loader:
    logits = model(batch)
    ce_loss = criterion(logits, batch["target"])
    loss = ce_loss

    if use_z_loss:
        loss = loss + z_loss(logits, batch["target"], a=1.0, b=0.0) * z_loss_weight

    loss.backward()

    log_global_grad_norm(model)
    log_module_grad_norm("embedding", model.embedding.parameters())
    log_module_grad_norm("dense", model.dense.parameters())
    log_update_norm(optimizer)
    log_logits_stats(logits)

    clip_module(model.embedding.parameters(), max_norm=1.0)
    clip_module(model.dense.parameters(), max_norm=5.0)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

如果是推荐系统或检索系统，还可以给稀疏 embedding 单独参数组：

```python
optimizer = AdamW([
    {"params": dense_params, "lr": base_lr, "betas": (0.9, 0.98), "eps": 1e-8},
    {"params": embedding_params, "lr": embed_lr, "betas": (0.9, 0.95), "eps": 1e-6},
])
```

这里不是说 embedding 一定要更大学习率或更小学习率，而是它常常需要独立对待。关键原则是：稀疏参数和 dense 参数的更新统计分布不同，不应默认共享完全相同的优化器配置。

---

## 工程权衡与常见坑

第一类坑是 $\beta_2$ 过大。$\beta_2=0.9999$ 会让 $v_t$ 极其平滑，平时看起来很稳，遇到罕见大梯度时却反应太慢。第二类坑是 $\beta_2$ 过小，例如直接降到 0.9，虽然响应快了，但全模型会更抖，损失方差增大，甚至把正常层也带进噪声区。

一个简单经验是：

| $\beta_2$ 区间 | 常见感觉 | 风险 |
|---|---|---|
| 0.9999 左右 | 曲线很平 | rare update 时可能滞后严重 |
| 0.999 左右 | 较常见默认值 | 多数场景可用，但仍可能慢半拍 |
| 0.98 到 0.995 | 响应更快 | 噪声上升，要结合 batch size 看 |
| 0.95 及以下 | 很灵敏 | 训练常明显变抖 |

第二类坑是只做全局裁剪。它看起来简单，但在 embedding-heavy 模型里常被少数稀疏梯度“霸占”。结果是 embedding 出事时，dense tower 也被迫缩小更新，相当于健康模块替病灶买单。模块级裁剪通常更合理。

第三类坑是把 Z-loss 当成总开关。Z-loss 主要抑制 logit 尺度问题，对输出端数值稳定有帮助，但它不解决输入分布突变、embedding 几何偏移、优化器状态恢复错误等根因。它更像护栏，不是发动机维修。

第四类坑是扩容时沿用默认参数化。一个 100M 模型上稳定的学习率，到了 10B 模型可能完全不是同一件事。若没有 $\mu$P 或等价的尺度对齐，所谓“同样的 lr”在不同规模上对应的真实更新幅度可能差很多。

第五类坑是忽略 $\epsilon$。在一些参数上，若 $\hat v_t$ 很小，$\epsilon$ 会直接影响分母的下界。$\epsilon$ 过小会让某些坐标的有效步长异常大；$\epsilon$ 过大又会削弱自适应预条件的意义。很多团队排查 spike 时只看 lr 和 $\beta_2$，但不检查 $\epsilon$，这是遗漏。

第六类坑是 checkpoint 恢复不完整。若参数恢复了，但 optimizer moments 没恢复，模型会以“旧参数 + 新优化器状态”的错配组合继续跑。表面看像随机 spike，本质上是恢复过程出了错。

| 常见坑 | 现象 | 建议 |
|---|---|---|
| $\beta_2$ 太大 | rare update 后 spike 更尖 | 适度下调，结合监控 update norm |
| 裁剪太严 | loss 不 spike，但长期不降 | 看触发频率，别让每步都在裁 |
| 只做全局裁剪 | dense 层学习变慢 | 优先模块级裁剪 |
| 只加 Z-loss | logit 好一点，但根因仍在 | 与优化器和参数化一起看 |
| 扩容不重审参数化 | 大模型超参全面失效 | 用 $\mu$P/μTransfer 做迁移 |
| $\epsilon$ 不合适 | 某些参数更新异常尖 | 和 $\beta_2$ 一起排查 |
| checkpoint 状态缺失 | 恢复后立刻抖动 | 确认 moments、scaler、step 全恢复 |

真实工程里，一个常见推荐是：先定位 spike 来源，再下药。若是 embedding 主导，就先做 per-module norm 监控和模块级裁剪；若是 update norm 高于 grad norm 异常明显，就检查 Adam 状态、$\beta_2$、$\epsilon$ 与 checkpoint 恢复；若是 logits 尾部明显发散，再考虑 Z-loss 或输出 embedding centering。

---

## 替代方案与适用边界

梯度裁剪不是唯一方案。对稀疏模型，更细粒度的 per-table clipping、per-row clipping 往往比全局裁剪更合适。它的直觉是：只限制真正危险的表或行，不拖累全局学习。

另一类替代是换优化器思路。例如 embedding 用 Adagrad 风格，dense 部分用 AdamW。这么做的原因是 Adagrad 会更快累积稀疏参数的历史平方梯度，对长尾特征常更稳。但代价是优化器逻辑更复杂，状态更重，调参面更大。

对输出端，如果发现 Z-loss 只能缓解但不能根治，近期还可以考虑 output embedding centering。它针对的是输出 embedding 几何偏移，而不只是 logit 尺度。简单说，centering 相当于把输出空间的整体偏移减掉，让 logits 分布更居中。

$\mu$P 也有适用边界。它最适合“宽度扩展规律明确”的架构，例如标准 Transformer、MLP、ResNet 变体。若模型包含大量非均匀抽样、动态路由、极端异构子模块，超参跨规模迁移不一定还能像标准设置那样顺滑。

这里可以把几类方案按“解决哪一层问题”来理解：

| 方案 | 主要解决层次 | 适用性 | 限制 |
|---|---|---|---|
| per-module clipping | 更新层 | embedding-heavy 模型 | 需要更多监控与阈值设计 |
| per-row/per-table clipping | 梯度层 | 推荐系统、稀疏 ID 场景 | 实现成本高 |
| Adagrad for embeddings | 优化器层 | 长尾稀疏特征稳定 | 统一优化器被拆开 |
| Z-loss | 输出层 | 输出 logit 容易爆的分类/LM | 只抑制尺度 |
| output embedding centering | 输出几何层 | 输出层几何偏移明显 | 研究较新，工程经验较少 |
| $\mu$P / μTransfer | 参数化层 | 需要反复扩容的训练体系 | 需遵守参数化规则 |

如果只给一个初学者版本建议，可以是这套顺序：

1. 先加 per-module grad 监控和 update norm 监控，不要只看总 loss。
2. 再上模块级 clipping，优先保护 dense 主干不被异常 embedding 连坐。
3. 若 spike 仍明显，检查 $\beta_2$、$\epsilon$ 与 rare embedding 参数组。
4. 如果团队要频繁从小模型放大到大模型，把 $\mu$P 引入超参迁移流程。
5. 若 LM 输出端 logits 仍失控，再考虑 Z-loss 或 centering。

---

## 参考资料

| 来源 | 类型 | 日期 | 关键贡献 | 适用章节 |
|---|---|---:|---|---|
| [Adaptive Preconditioners Trigger Loss Spikes in Adam](https://www.catalyzex.com/paper/adaptive-preconditioners-trigger-loss-spikes) | 论文索引 | 2025-06-05 | 给出 Adam 预条件器导致 spike 的机制；指出预条件 Hessian 的梯度方向曲率越过 $2/\eta$ 时会进入不稳定区 | 核心结论、机制推导 |
| [Stable and Low-Precision Training for Large-Scale Vision-Language Models](https://nips.cc/virtual/2023/poster/70245) | NeurIPS 2023 论文页面 | 2023 | 观察到 AdamW 二阶矩低估后 1 到 8 步出现 spike，并提出更稳的优化思路 | 机制推导、替代方案 |
| [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://www.microsoft.com/en-us/research/publication/tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer/) | Microsoft Research 论文页 | 2022-03 | 给出 $\mu$P 与 μTransfer，说明超参可跨模型规模零样本迁移 | 核心结论、替代方案 |
| [µTransfer: A technique for hyperparameter tuning of enormous neural networks](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/) | 官方博客 | 2022-03-08 | 用工程语言解释 $\mu$P 为什么能保持最优学习率跨宽度稳定 | 核心机制、工程权衡 |
| [The Z-loss: a shift and scale invariant classification loss belonging to the Spherical Family](https://www.catalyzex.com/paper/the-z-loss-a-shift-and-scale-invariant) | 论文索引 | 2016-04-29 | 给出 Z-loss 原始定义，说明其 shift/scale invariant 性质 | 核心机制 |
| [Z-Loss: Scale & Shift Invariant Loss](https://www.emergentmind.com/topics/z-loss) | 综述 | 2026-02-17 | 整理 Z-loss 公式、性质与适用边界，适合工程概览 | 核心机制、替代方案 |
| [Output Embedding Centering for Stable LLM Pretraining](https://www.emergentmind.com/papers/2601.02031) | 论文索引 | 2026-01-05 | 指出 z-loss 主要缓解症状，提出 centering/μ-loss 作为替代 | 工程权衡、替代方案 |
| [Loss spikes in training: causes, detection, and mitigations](https://medium.com/better-ml/loss-spikes-in-training-causes-detection-and-mitigations-ed66e591b1a1) | 工程博客 | 2026-01-19 | 给出推荐系统中 rare embedding、全局裁剪、监控指标的工程案例 | 问题定义、代码实现、常见坑 |
