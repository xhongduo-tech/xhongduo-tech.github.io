## 核心结论

“涌现能力”通常指一种表面现象：模型在某个规模之前几乎做不到某项任务，到了某个规模后分数突然抬升，看起来像能力从无到有地出现了。更细地看，这种“突然出现”在多数情况下并不是能力真正断裂式生成，而是**底层能力连续改进，被离散指标放大成了阶跃**。

这里要先区分两类指标：

| 指标类型 | 常见例子 | 看什么 | 典型问题 |
|---|---|---|---|
| 离散指标 | Accuracy、Exact Match、Pass/Fail | 只看最终是否答对 | 会把“差一点对”和“完全不会”都记成 0 |
| 连续指标 | Brier Score、Cross-Entropy、Log Loss | 看概率分布离真实答案有多近 | 更敏感，但解释门槛更高 |

这正是“可预测性”争论的核心。Wei 等人在 2022 年把“emergent abilities”系统化描述为一种可能无法从小模型外推的大模型现象；Schaeffer 等人在 NeurIPS 2023 则指出，很多所谓“涌现”会在换成连续指标后变成平滑曲线；Ruan 等人在 NeurIPS 2024 进一步说明，若把不同模型投影到低维能力空间，许多复杂 benchmark 的走势可以仅用小模型观测值做合理预测。

最直观的玩具例子是四选一题。假设正确答案是 A，对应 one-hot 标签为 $(1,0,0,0)$。下面三个模型都还不算“完全可靠”，但它们离正确答案的距离不同：

| 模型输出概率 | 是否答对 | Brier Score |
|---|---:|---:|
| $(0,1,0,0)$ | 0 | 2.00 |
| $(0.25,0.25,0.25,0.25)$ | 1 或 0，取决于并列决策规则 | 0.75 |
| $(0.40,0.20,0.20,0.20)$ | 1 | 0.48 |

Brier Score 定义为：

$$
\text{Brier}=\sum_{i=1}^{K}(p_i-o_i)^2
$$

其中 $p_i$ 是模型给第 $i$ 个选项的概率，$o_i$ 是真实 one-hot 标签。这个指标直接比较“预测分布”和“真实分布”的距离，所以即使两个模型在 Accuracy 上同为 0，它也能分清谁是“完全压错方向”，谁是“已经开始把概率往正确答案集中”。

对新手来说，可以这样理解：  
Accuracy 问的是“最后有没有过线”；Brier 问的是“离过线还有多远”。如果只看前者，小模型的很多细微进步会被压成一串 0，看上去就像什么都不会。

但这个结论不能说成“所有涌现都是错觉”。少数任务仍然显示出难以平滑外推的跃升。例如 BIG-Bench 中的 `navigate` 任务，在 PaLM 到 U-PaLM 的对比里出现过明显抬升。这类现象可能不仅仅是指标问题，还可能涉及内部表示或求解策略发生了结构变化。更稳妥的结论是：

**多数所谓涌现是可预测的，少数涌现可能对应真实的结构跃迁。**

---

## 问题定义与边界

要讨论“涌现能力是否可预测”，必须先把问题拆成两个层面，否则很容易把不同现象混在一起。

| 层面 | 核心问题 | 典型现象 | 判断办法 |
|---|---|---|---|
| 测量层面 | 指标是否把连续进步压成阶跃 | 准确率长期接近 0，之后突然抬升 | 换成连续指标后是否变平滑 |
| 表示层面 | 模型内部是否真的形成了新结构 | 某些任务在多个尺度上都难以平滑外推 | 换指标后仍非平滑，且跨模型族复现 |

这里“表示”可以简单理解为：模型在参数空间里到底有没有学到一种新的内部组织方式，使它能够稳定解决某类问题。若只是评估太粗，换指标通常就能看到连续进步；若真的是表示发生重组，那么即使改用更细的指标，曲线也可能仍然不平滑。

一个常见误区是把“高规模时才第一次看见高分”直接等同于“高规模时能力才第一次产生”。这并不成立。原因很简单：很多任务的最终评分是一个**阈值函数**，而不是一个线性函数。

以长度为 $L$ 的序列任务为例，如果模型每个 token 都要预测正确，整体全对概率会被局部误差指数放大。一个常见近似写法是：

$$
\Pr(\text{all correct}) \approx \exp(-L \cdot L_{\text{CE}}(N))
$$

其中：

- $L$ 是输出序列长度
- $L_{\text{CE}}(N)$ 是规模为 $N$ 的模型在该任务上的平均 token 级交叉熵损失

这个式子不是严格恒等式，而是一个机制说明：  
**当任务要求“整段全对”时，哪怕每个 token 的误差只下降一点点，最终的全对概率也可能在某个区间里看起来突然上升。**

这类机制在工程里非常常见。比如一个结构化抽取任务要求 12 个字段全部命中才能算对，那么单字段正确率从 0.86 提升到 0.92，最终“整条记录全对率”可能就会产生很陡的变化。团队如果只看最终通过率，很容易误判为“能力突然出现”。

因此，“可预测性”的正确问题不是：

> 这个任务会不会突然从 0 变成 1？

而应该是：

> 在更细粒度的能力度量下，这个任务的底层趋势是否已经在小模型上可见？

Ruan 等人在 NeurIPS 2024 的 observational scaling work 基本给出的就是这个答案：  
对很多任务，答案是“可以”；对少数具有特殊结构要求的任务，答案仍然是“风险较高，不能盲目平滑外推”。

---

## 核心机制与推导

先看最底层的量：语言模型的预训练损失。经典 scaling law 工作观察到，在相当大的范围内，损失会随模型规模呈近似幂律下降：

$$
L(N) \approx aN^{\alpha}+b,\quad \alpha<0
$$

有些文章会把它写成等价形式：

$$
L(N) \approx \left(\frac{N}{c}\right)^\alpha + b
$$

含义相同：模型越大，平均预测误差越低，但收益递减，曲线通常是平滑下降而不是突然折断。

### 从平滑损失到阶跃准确率

问题在于，很多 benchmark 并不直接评分这个平滑损失，而是先经过一层强阈值映射。例如多选题只看 argmax 是否选中正确项，数学题只看最终答案字符串是否完全一致，代码任务只看单测是否全过。

于是会出现下面的链条：

$$
\text{规模增大}
\rightarrow
\text{token 级概率微调}
\rightarrow
\text{局部误差连续下降}
\rightarrow
\text{最终离散指标在阈值附近陡升}
$$

这就是“涌现幻象”最常见的来源。

可以把几个常见量并列看：

| 量 | 数学性质 | 对小改进是否敏感 | 容易看到什么 |
|---|---|---|---|
| 交叉熵损失 $L_{\text{CE}}$ | 连续 | 高 | 平滑下降 |
| Brier Score | 连续 | 高 | 置信度逐步向正确答案集中 |
| Accuracy / Exact Match | 离散 | 低 | 长期 0，之后突然跨阈值 |

### 为什么 Brier 更适合观察“有没有在学”

假设四选一题的正确答案是 A，对应真实标签 $(1,0,0,0)$。考虑三个模型：

- 模型 1：$(0.25,0.25,0.25,0.25)$
- 模型 2：$(0.40,0.20,0.20,0.20)$
- 模型 3：$(0.70,0.10,0.10,0.10)$

它们的 Brier Score 分别是：

$$
\begin{aligned}
\text{Brier}_1 &= (0.25-1)^2 + 3 \times (0.25-0)^2 = 0.75 \\
\text{Brier}_2 &= (0.40-1)^2 + 3 \times (0.20-0)^2 = 0.48 \\
\text{Brier}_3 &= (0.70-1)^2 + 3 \times (0.10-0)^2 = 0.12
\end{aligned}
$$

这组数字表达了一个非常重要的事实：

- 模型 1 只是平均分配概率，还没有明显偏向正确答案
- 模型 2 已经开始把更多概率压到正确答案上
- 模型 3 对正确答案的概率集中更强

如果只看 Accuracy，这三者可能是 0、1、1，也可能因为并列处理规则变成 1、1、1。无论哪一种，都掩盖了模型在“靠近正确答案”这件事上的连续进步。Brier 则把这条进步轨迹保留下来了。

### 为什么长序列任务更容易“看起来突然会了”

假设每个位置都以独立近似处理，单 token 正确概率为 $q(N)$，那么长度为 $L$ 的序列全对概率近似为：

$$
\text{EM}(N)\approx q(N)^L
$$

若 $q(N)$ 随规模平滑上升，例如从 0.85 升到 0.93，那么：

| 单 token 正确率 $q$ | 长度 $L=5$ 时全对率 | 长度 $L=20$ 时全对率 |
|---|---:|---:|
| 0.85 | 0.444 | 0.039 |
| 0.89 | 0.559 | 0.097 |
| 0.93 | 0.696 | 0.234 |

对短序列，这看起来是平滑提升；对长序列，前期分数可能长期接近 0，后期才突然抬头。注意，这种“突然抬头”不一定意味着模型内部发生了某种神秘跳变，只是因为最终指标把很多局部误差连乘了。

### 为什么 observational scaling 更稳

Ruan 等人的思路不是盯住单个 benchmark 分数，而是先把多个任务的表现压缩到一个低维能力空间，再研究能力随训练计算量或参数规模的变化。这里“低维能力空间”的直观意思是：

- 很多 benchmark 并不是完全独立的
- 它们往往共享一些底层能力轴
- 例如语言理解、知识检索、数学推理、代码生成等可以视作若干潜在维度

在这个表示下，模型家族之间的主要区别可以理解为：

- 谁把训练计算量更高效地转成这些能力
- 哪些任务主要由共享能力决定
- 哪些任务对特殊结构或策略特别敏感

因此，多数任务可以外推，是因为它们主要受共享能力控制；少数任务难外推，是因为它们更依赖某种特殊结构突然形成，例如路径追踪、复杂计数、组合搜索或长程状态维护。

---

## 代码实现

下面给一个最小但可运行的 Python 例子，演示三件事：

1. 如何计算 Brier Score  
2. 为什么 Accuracy 会掩盖“正在变好”的过程  
3. 为什么一个平滑底层趋势会投影成“像涌现一样”的 S 型曲线

```python
import math
from typing import Iterable, List


def normalize(probs: Iterable[float]) -> List[float]:
    probs = list(probs)
    s = sum(probs)
    if s <= 0:
        raise ValueError("probabilities must sum to a positive number")
    return [p / s for p in probs]


def brier_score(probs: Iterable[float], correct_idx: int) -> float:
    probs = normalize(probs)
    if not 0 <= correct_idx < len(probs):
        raise IndexError("correct_idx out of range")
    target = [0.0] * len(probs)
    target[correct_idx] = 1.0
    return sum((p - t) ** 2 for p, t in zip(probs, target))


def top1_accuracy(probs: Iterable[float], correct_idx: int) -> float:
    probs = list(probs)
    pred_idx = max(range(len(probs)), key=lambda i: probs[i])
    return 1.0 if pred_idx == correct_idx else 0.0


def exact_match_from_token_acc(token_acc: float, length: int) -> float:
    if not (0.0 <= token_acc <= 1.0):
        raise ValueError("token_acc must be between 0 and 1")
    if length <= 0:
        raise ValueError("length must be positive")
    return token_acc ** length


def sigmoid(x: float, a: float = 1.0, b: float = 0.0) -> float:
    return 1.0 / (1.0 + math.exp(-(a * x + b)))


def demo():
    correct_idx = 0  # 正确答案是 A

    models = {
        "wrong_direction": [0.0, 1.0, 0.0, 0.0],
        "uniform": [0.25, 0.25, 0.25, 0.25],
        "better": [0.40, 0.20, 0.20, 0.20],
        "confident": [0.70, 0.10, 0.10, 0.10],
    }

    print("=== Multi-choice example ===")
    for name, probs in models.items():
        acc = top1_accuracy(probs, correct_idx)
        brier = brier_score(probs, correct_idx)
        print(f"{name:15s} acc={acc:.0f}  brier={brier:.4f}  probs={probs}")

    # 验证 Brier 连续下降
    assert brier_score(models["wrong_direction"], 0) > brier_score(models["uniform"], 0)
    assert brier_score(models["uniform"], 0) > brier_score(models["better"], 0)
    assert brier_score(models["better"], 0) > brier_score(models["confident"], 0)

    print("\n=== Sequence-level amplification ===")
    for token_acc in [0.85, 0.89, 0.93]:
        em_5 = exact_match_from_token_acc(token_acc, 5)
        em_20 = exact_match_from_token_acc(token_acc, 20)
        print(f"token_acc={token_acc:.2f}  EM@5={em_5:.4f}  EM@20={em_20:.4f}")

    print("\n=== Smooth latent trend -> emergent-looking curve ===")
    log_flops = [20, 21, 22, 23, 24, 25]
    latent_capability = [sigmoid(x, a=1.4, b=-32.0) for x in log_flops]
    for x, y in zip(log_flops, latent_capability):
        print(f"log_flops={x}  latent_score={y:.4f}")

    # 单调递增检查
    for i in range(len(latent_capability) - 1):
        assert latent_capability[i] < latent_capability[i + 1]


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行，输出会体现两点：

- `wrong_direction -> uniform -> better -> confident` 的 Brier Score 单调下降，说明模型在持续接近正确答案
- 即使底层 `latent_score` 是平滑增长，映射到最终离散指标后也可能表现成一段很陡的上升

如果做真实工程分析，原始数据表至少应包含下面这些列：

| 字段 | 含义 | 为什么需要 |
|---|---|---|
| `model` | 模型名或检查点名 | 区分不同模型点 |
| `family` | 模型族 | 控制架构差异 |
| `params` | 参数量 | 观察规模效应 |
| `train_flops` | 训练 FLOPs | 比单看参数更稳 |
| `task` | 任务名 | 支持分任务分析 |
| `sample_id` | 样本编号 | 便于重算统计量 |
| `prob_correct` | 正确答案概率 | 计算连续指标 |
| `brier` | Brier Score | 观察概率对齐趋势 |
| `cross_entropy` | 交叉熵 | 观察 token 级平滑变化 |
| `accuracy` | Top-1 是否命中 | 最终业务指标 |
| `exact_match` | 是否全对 | 长序列任务特别重要 |

一个示意表如下：

| model | family | log_flops | task | prob_correct | brier | accuracy |
|---|---|---:|---|---:|---:|---:|
| small-1 | A | 22.1 | gsm8k-like | 0.18 | 0.67 | 0 |
| small-2 | A | 22.8 | gsm8k-like | 0.24 | 0.58 | 0 |
| mid-1 | A | 23.6 | gsm8k-like | 0.41 | 0.35 | 0 |
| large-1 | A | 24.4 | gsm8k-like | 0.73 | 0.11 | 1 |

推荐分析流程是：

$$
\text{数据采集}
\rightarrow
\text{连续指标计算}
\rightarrow
\text{按 } \log(\text{FLOPs}) \text{ 或 } \log(N) \text{ 拟合}
\rightarrow
\text{最后再检查离散指标}
$$

原因很直接：  
连续指标负责发现趋势，离散指标负责回答“最终有没有达标”。这两个问题都重要，但不是同一个问题。

---

## 工程权衡与常见坑

实际工作里，很多“涌现讨论”最后争的不是理论，而是评估设计是否合格。

| 常见坑 | 会造成什么误判 | 更稳妥的做法 |
|---|---|---|
| 只看 Exact Match | 把渐进改进误读为 0 到 1 跳跃 | 同时记录 Brier、CE、正确项概率 |
| 样本太少 | 曲线抖动被误当成“突然跃升” | 增加样本量并报告置信区间 |
| 只取 2 到 3 个模型点 | 外推极不稳定 | 补中间尺度模型 |
| 混入不同架构或训练配方 | 把 recipe 改动误当成 scaling | 按模型族分组拟合 |
| 只拟合单任务离散分数 | 噪声大、解释弱 | 先看能力空间或联合拟合 |
| 只报均值不报方差 | 无法区分稳定趋势和偶然波动 | 报标准误、bootstrap CI |
| 忽略 prompt/解码设置 | 把推理策略变化误当能力涌现 | 固定提示模板与采样策略 |

### 为什么样本量不足会制造假涌现

设某模型在某任务上的真实准确率是 8%。如果你只测 20 个样本，很可能一次评估里恰好全错，于是记成 0%；换个更大模型又恰好对了 4 个，于是你看到 20%。表面上像“从不会到会了”，实际只是小样本波动非常大。

因此，在报告所谓“能力首次出现”时，至少要同步报告：

- 评估样本量
- 多次重复评估的均值与方差
- 置信区间或 bootstrap 区间
- 是否更换过 prompt、采样温度、解码策略

没有这些信息，“涌现”往往只是一个图形印象。

### 为什么只看 0 分会误判

一个典型误用是：团队发现某任务在 7B、13B、34B 上 Accuracy 都是 0，于是得出结论“这个能力要到 70B 才出现”。这在统计上站不住脚。因为同时看连续指标时，完全可能出现：

| 模型 | Accuracy | Brier | 正确项平均概率 |
|---|---:|---:|---:|
| 7B | 0 | 0.82 | 0.11 |
| 13B | 0 | 0.71 | 0.18 |
| 34B | 0 | 0.60 | 0.27 |
| 70B | 1 | 0.21 | 0.63 |

这种情况下，前三个模型并不是“没有能力”，而是“能力还没跨过业务阈值”。如果把这个过程描述成“70B 前完全不会”，那是在错误地把评估阈值当成了能力边界。

### 为什么也不能把一切都解释成指标幻象

另一种极端是：看到 Schaeffer 等人的结论后，就把所有跳跃都说成“只是指标错觉”。这同样不严谨。`navigate` 一类任务提醒我们，某些现象在换成更细指标后仍然不容易被平滑解释，尤其当任务要求：

- 空间状态连续更新
- 多步路径跟踪
- 复杂计数或组合约束维护
- 长程规划或搜索式推理

这些任务更可能依赖特定内部策略或表示结构是否形成。工程上更合理的态度是：

**默认多数任务可平滑建模，但对少数结构敏感任务保留“非平滑风险”假设。**

---

## 替代方案与适用边界

Brier Score 很有用，但它不是唯一选择。不同阶段关心的问题不同，对应的指标也应该不同。

| 指标 | 连续性 | 可解释性 | 适合场景 | 局限 |
|---|---|---|---|---|
| Accuracy / Exact Match | 低 | 高 | 最终验收、线上 SLA、用户体验 | 阈值效应强 |
| Brier Score | 高 | 中 | 多分类概率质量、早期趋势判断 | 需要拿到概率输出 |
| Cross-Entropy / Log Loss | 高 | 中 | 训练分析、token 级趋势 | 数值直觉不如 Accuracy |
| ECE / Calibration Error | 中 | 中 | 关心置信度是否可靠 | 不直接反映任务完成度 |
| Pass@k | 中 | 高 | 代码生成、多候选搜索 | 依赖采样与预算设置 |
| 低维能力空间分数 | 中到高 | 低 | 多任务联合预测、observational scaling | 解释成本高 |

### 什么时候必须看离散指标

如果产品目标本身就是“答对没有”，离散指标仍然是最终标准。例如：

- 客服问答是否命中正确政策
- 结构化抽取是否字段全部正确
- SQL 生成是否执行结果完全一致
- 代码生成是否单测通过

这些场景里，用户不关心模型“差一点对”。所以最终验收时，Accuracy、EM、Pass@k 不能被替代。

### 什么时候应该优先看连续指标

如果你关心的是：

- 是否值得继续扩规模
- 某条能力曲线是否已经起势
- 小模型结果能否预测大模型
- 哪个训练 recipe 更有潜力

那么连续指标往往更有信息量。因为这时你要回答的是“趋势如何”，而不是“现在是否达标”。

一个实用的分层方案是：

1. 早期研究阶段：优先看 CE、Brier、正确项概率，判断趋势是否平滑。  
2. 中期缩放评估：按 $\log(\text{FLOPs})$ 或 $\log(N)$ 拟合，并检查拟合残差与 $R^2$。  
3. 最终交付阶段：回到 Accuracy、Exact Match、Pass@k 等业务指标。  

这里的 $R^2$ 可以理解为“拟合解释了多少方差”。它高不代表理论正确，但低通常意味着外推风险高。尤其当单任务离散分数的 $R^2$ 很差时，直接拿它预测更大模型是否会“涌现”，风险通常很大。

### 哪些任务要默认更保守

如果任务高度依赖下列结构，应默认存在非平滑风险：

- 搜索
- 长程规划
- 空间路径跟踪
- 复杂计数
- 多步状态维护
- 强组合约束推理

这类任务不是不能用连续指标，而是不能把连续指标的外推结果当成确定事实。更合适的做法是把预测写成区间，并明确注明高不确定性。

---

## 参考资料

| 论文 | 关键结论 | 可复现资源出处 | 链接 |
|---|---|---|---|
| Wei et al., 2022, *Emergent Abilities of Large Language Models* | 系统提出“涌现能力”现象，定义其“突然性”和“不可由小模型直接外推”的特征 | arXiv / TMLR 公开版本 | [arXiv](https://arxiv.org/abs/2206.07682) |
| Schaeffer et al., NeurIPS 2023, *Are Emergent Abilities of Large Language Models a Mirage?* | 许多涌现来自离散指标放大；换成连续指标后，不少曲线会变平滑 | NeurIPS 2023 论文页 / OpenReview | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html) |
| Ruan et al., NeurIPS 2024, *Observational Scaling Laws and the Predictability of Language Model Performance* | 不必自己训练一整套缩放模型，仅用公开模型观测值，也能对许多高尺度能力做合理预测 | NeurIPS 2024 论文页 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1cded4f97cf5f01a284c574110b7e3b9-Abstract-Conference.html) |
| Tay et al., EMNLP 2023, *Transcending Scaling Laws with 0.1% Extra Compute* | U-PaLM 说明训练目标和 recipe 的小改动也能明显改写曲线，少数任务出现显著跃升 | ACL Anthology | [ACL](https://aclanthology.org/2023.emnlp-main.91/) |
| Kaplan et al., 2020, *Scaling Laws for Neural Language Models* | 给出经典损失缩放规律，是理解“底层平滑改进”最重要的基础文献之一 | OpenAI / arXiv | [OpenAI](https://openai.com/index/scaling-laws-for-neural-language-models/) |
| Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models* | Chinchilla 表明参数、数据、计算的配比会显著影响 scaling 曲线，不能只盯参数量 | arXiv / DeepMind 公开版本 | [arXiv](https://arxiv.org/abs/2203.15556) |
