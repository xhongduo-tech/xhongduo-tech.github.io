## 核心结论

模型蒸馏推理加速 = 行为迁移 + 结构瘦身 + 后端适配。

这里的“行为迁移”是指：让小模型学习大模型输出概率、错误偏好和部分中间表示，而不是只记最终标准答案。“后端适配”是指：学生模型的结构要和推理框架、算子实现、显存管理方式匹配，否则纸面上的参数缩小，不一定能换来真实吞吐提升。

新手版可以先记一句话：蒸馏不是把参数表直接砍短，而是让“学生模型”学会“老师模型”做判断时的分布和习惯，所以学生在更少层数、更低显存下，仍能保住主要能力。

| 对象 | 典型形态 | 推理后端关注点 | 主要收益来源 |
| --- | --- | --- | --- |
| teacher | 大模型，层数多，容量高 | 通常用于离线打标或蒸馏训练 | 提供高质量行为分布 |
| student | 小模型，层数少，结构更紧凑 | 是否原生支持 attention、并行方式、KV cache | 更低计算量与显存占用 |
| 推理后端 | vLLM、TGI、TensorRT-LLM 等 | kernel 是否高效、KV cache 管理是否成熟 | 把结构优势转成真实延迟和吞吐收益 |
| 收益来源 | 不是单一环节 | 模型缩小 + 层数减少 + 后端匹配 | 延迟下降、吞吐上升、成本降低 |

玩具例子：teacher 是 12 层模型，student 是 6 层模型。如果隐藏维度不变，单 token 前向时，注意力和前馈网络大致都少跑一半层数。再叠加更小的 KV cache，长序列场景下显存压力也会下降。真正加速不只来自“参数更少”，还来自“每一步生成时要搬运和缓存的状态更少”。

真实工程例子：在线问答系统里，团队常用一个 14B 或 32B 的 teacher 生成蒸馏监督，再训练 7B 甚至更小的 student。若 student 仍是推理后端友好的 decoder-only 结构，并且头数、维度、attention 形式都在后端高效支持范围内，那么线上能同时拿到更低首 token 延迟和更高并发吞吐。

---

## 问题定义与边界

这篇文章讨论的问题，不是“怎么把模型缩小”，而是“怎么在质量尽量不掉的前提下，把在线推理成本和响应时间降下来”。

几个基本术语先说明：

| 术语 | 白话解释 | 这里的作用 |
| --- | --- | --- |
| teacher | 大老师模型 | 负责提供更细的参考答案 |
| student | 学生模型 | 负责上线推理，目标是更快更省 |
| logits | softmax 之前的原始分数 | 比最终类别更保留相对偏好信息 |
| KV cache | 自回归生成时缓存历史 key/value 的内存块 | 直接影响长上下文显存和速度 |

边界很重要。蒸馏适合“已有强 teacher、任务分布明确、能收集代表性样本”的场景；不适合“没有 teacher、需求每天变、输入分布漂移很大”的场景。因为 student 学到的是 teacher 在已见分布上的行为压缩，而不是无条件得到更强泛化。

| 适合场景 | 不适合场景 | 需要补充的数据 |
| --- | --- | --- |
| 有成熟 teacher 的分类、检索、问答、摘要 | 没有可靠 teacher 的全新任务 | 真实线上样本 |
| 业务输入分布相对稳定 | 分布频繁变化、概念持续漂移 | 难例与失败样本 |
| 关心推理成本、显存、吞吐 | 只想最快上线且不想训练 | 长上下文与工具调用轨迹 |
| 可以做离线蒸馏集构建 | 无法拿到代表性数据 | OOD 样本与边界样本 |

新手版常见误区是：只要 student 在验证集平均准确率接近 teacher，就算蒸馏成功。这个判断过于粗。若业务真正关心的是长上下文、多轮工具调用、尾延迟，那么“平均答对率不错”并不等于“上线表现安全”。

---

## 核心机制与推导

蒸馏最常见的做法，是同时优化三类信息：软标签、硬标签和中间层表示。

“软标签”是指 teacher 给出的完整概率分布，不只是最终答案。“硬标签”是真实标注答案。“中间层表示”是网络中间位置的特征向量，代表模型在内部怎样组织信息。

公式通常写成：

$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

$$
L = \alpha \cdot T^2 \cdot KL(p_t^{(T)} \parallel p_s^{(T)})
+ (1 - \alpha) \cdot CE(y, p_s^{(1)})
+ \beta \cdot \sum_l \|h_t^l - h_s^{m(l)}\|_2^2
$$

这里温度 $T > 1$ 的作用是把分布“拉平”。拉平后，teacher 不只告诉你“A 是对的”，还告诉你“B 比 C 更像正确答案”。这对模糊样本很重要。

必须看的数值例子如下。设：

- teacher：$p_t = [0.70, 0.20, 0.10]$
- student：$p_s = [0.55, 0.30, 0.15]$

两者最大概率都是 A，但分布并没对齐。teacher 认为 A 明显更可信，而 student 把过多概率给了 B 和 C。此时 $KL(p_t \parallel p_s)$ 仍大于 0，说明 student 还有蒸馏空间。它不只要学“答案是 A”，还要学“B 比 C 更接近，但都不该过高”。

| 损失项 | 含义 | 对应作用 |
| --- | --- | --- |
| KL 项 | 对齐 teacher 和 student 的高温分布 | 学到类间关系、错误偏好、置信度结构 |
| CE 项 | 对齐真实标签 | 防止 student 只模仿 teacher 的偏差 |
| 中间层对齐项 | 对齐隐藏层表示 | 让 student 学到更稳定的内部特征组织 |

推导直觉可以这样理解。推理成本大致由两部分主导：每层计算量和每层缓存量。若 teacher 有 $L_t$ 层，student 有 $L_s$ 层，且其他维度近似不变，那么生成阶段的主要计算和 KV cache 都近似随层数按比例缩小，比例约为 $L_s / L_t$。所以从 12 层到 6 层，很多成本可以近似减半。蒸馏的价值是：尽量让这次“减半”不直接变成“质量断崖”。

---

## 代码实现

实现上可以拆成三步：准备蒸馏数据、同时前向 teacher 和 student、计算复合损失并只更新 student。训练时就要考虑部署目标，因为 student 的结构一旦定死，后面上线很难靠后处理补回架构不匹配的问题。

训练流程可以概括为：

`数据 -> teacher/student 前向 -> 蒸馏损失 -> 更新 student -> 部署`

下面给一个可运行的最小 Python 例子，只演示高温 softmax、KL 项和总损失的计算逻辑。它不是深度学习框架训练代码，但能把蒸馏核心算清楚。

```python
import math

def softmax(logits, T=1.0):
    scaled = [x / T for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def kl_divergence(p, q):
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q):
        total += pi * math.log((pi + eps) / (qi + eps))
    return total

def cross_entropy(one_hot_index, probs):
    eps = 1e-12
    return -math.log(probs[one_hot_index] + eps)

teacher_logits = [2.8, 1.5, 0.7]
student_logits = [2.2, 1.6, 0.9]
y = 0
T = 2.0
alpha = 0.7

pt = softmax(teacher_logits, T)
ps = softmax(student_logits, T)
ps_hard = softmax(student_logits, 1.0)

loss_kd = (T * T) * kl_divergence(pt, ps)
loss_ce = cross_entropy(y, ps_hard)
loss = alpha * loss_kd + (1 - alpha) * loss_ce

assert len(pt) == 3 and abs(sum(pt) - 1.0) < 1e-9
assert len(ps) == 3 and abs(sum(ps) - 1.0) < 1e-9
assert loss_kd > 0
assert loss_ce > 0
assert loss > 0

print("teacher(T=2):", [round(x, 4) for x in pt])
print("student(T=2):", [round(x, 4) for x in ps])
print("student(T=1):", [round(x, 4) for x in ps_hard])
print("loss_kd:", round(loss_kd, 6))
print("loss_ce:", round(loss_ce, 6))
print("total_loss:", round(loss, 6))
```

如果换成深度学习训练框架，主循环通常就是：

```python
teacher.eval()
for batch in dataloader:
    x, y = batch
    with torch.no_grad():
        z_t, h_t = teacher(x)
    z_s, h_s = student(x)

    loss_kd = kl_div(softmax(z_t / T), softmax(z_s / T))
    loss_ce = ce_loss(z_s, y)
    loss_hid = hidden_loss(h_t, h_s)

    loss = alpha * T*T * loss_kd + (1 - alpha) * loss_ce + beta * loss_hid
    loss.backward()
    optimizer.step()
```

如果目标后端是 vLLM 一类面向高吞吐的服务框架，student 结构选择最好提前约束：

| 结构选择点 | 对部署的影响 | 建议 |
| --- | --- | --- |
| 层数更少 | 直接降低每 token 计算和缓存 | 是蒸馏优先收益项 |
| attention 形式标准化 | 避免 fallback kernel | 优先选后端原生支持配置 |
| 头数与维度规整 | 更容易吃满并行和 kernel | 避免奇怪维度组合 |
| decoder-only 等主流结构 | 工具链更成熟 | 优先于冷门自定义结构 |

---

## 工程权衡与常见坑

蒸馏最容易出现的问题，不是“完全没加速”，而是“平均指标看着好，真实业务掉得很难看”。

| 问题 | 表现 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 只看 logits，不看长链路样本 | 长上下文、多轮任务掉崖 | teacher 行为只在短样本上被学习 | 把长上下文、工具轨迹纳入蒸馏集 |
| 只看参数量，不看后端兼容性 | 理论变小，线上不快 | kernel fallback、attention 实现低效 | 先按后端支持列表选 student 结构 |
| 只看平均值，不看尾部质量 | p95/p99 很差，用户体感差 | 难例和边界样本被平均数掩盖 | 单独评测尾延迟、难例、OOD |

新手版可以把第二类坑理解成：车身更轻了，但轮胎规格和路不匹配，最后还是跑不快。模型参数少，不等于系统吞吐一定高。

一组更靠谱的评测指标至少应包含：

| 指标 | 为什么要看 |
| --- | --- |
| 平均指标 | 看总体质量是否明显下滑 |
| p95/p99 延迟 | 看尾部请求是否失控 |
| long-context | 看 KV cache 压力下是否还能稳定 |
| tool-calling | 看动作序列和调用格式是否保住 |
| OOD | 看分布外样本是否出现异常退化 |

真实工程里最危险的场景，是 teacher 本来擅长复杂工具调用，但蒸馏集只保留了“最终答案文本”，没有保留中间工具轨迹、失败重试样本和边界案例。这样 student 会学到“结果像”，却学不到“过程像”。上线后表现就是：简单题还行，复杂链路突然断层。

---

## 替代方案与适用边界

蒸馏不是唯一的加速方案。很多时候，量化、剪枝、稀疏化、缓存优化更直接。

“量化”是把权重和激活从高精度数值换成更低比特表示，比如 FP16 到 INT8 或 4-bit。“剪枝”是删掉一部分权重、头或层。“稀疏化”是主动制造大量零值，让计算跳过无效位置。

| 方法 | 适用目标 | 主要代价 |
| --- | --- | --- |
| 蒸馏 | 尽量保住 teacher 行为与质量 | 需要 teacher、数据和训练成本 |
| 量化 | 最快降低显存与带宽开销 | 可能有精度损失与算子兼容问题 |
| 剪枝 | 进一步压缩模型结构 | 训练和恢复质量较复杂 |
| 稀疏化 | 追求特定硬件上的高效计算 | 对硬件和框架支持依赖强 |

如果你只是想“尽快省显存”，4-bit 量化往往比重训一个 student 更快落地；如果你想“保住 teacher 的答题风格、概率排序和错误偏好”，蒸馏通常更合适。

一句边界总结：当目标是“质量尽量不掉、推理成本明显下降”，优先考虑蒸馏；当目标是“最快上线、最少训练成本”，先看量化、缓存优化和现成后端调优。

---

## 参考资料

1. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
2. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
3. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
4. [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
5. [vLLM Paged Attention Design](https://docs.vllm.ai/en/v0.13.0/design/paged_attention/)
