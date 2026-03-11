## 核心结论

公开信息里，**已经被明确确认**的是：Amazon Bedrock 支持把 **Claude 3.5 Sonnet 作为教师模型**，把 **Claude 3 Haiku 作为学生模型**做自动化蒸馏；结果目标不是让 Haiku 变成另一个大模型，而是让它在**同样的推理成本和速度档位**上，尽量继承更大模型在特定任务上的判断方式与输出质量。

“知识蒸馏”里的“蒸馏”，白话说就是：**让小模型学会大模型的信心分布，而不只是背标准答案**。新手版可以这样理解：把 Sonnet 或更强的模型当老师，老师不只说“选 A”，还会给出“我对 A、B、C 分别有多大把握”；Haiku 学的是这组把握值，所以它学到的不只是“对错”，还有“哪些错误其实接近正确”。这部分额外信息常被叫作“暗知识”，也就是藏在概率分布里的相对判断。

如果把 Claude 家族放在一起看，可以把它理解成一个典型的多档位产品结构：

| 模型档位 | 角色直觉 | 成本/速度 | 能力侧重点 | 蒸馏关系 |
| --- | --- | --- | --- | --- |
| Haiku | 轻量学生 | 最低成本、最高吞吐 | 高频、低延迟服务 | 公开确认可由 Sonnet 蒸馏 |
| Sonnet | 中档教师/主力模型 | 平衡 | 综合推理、通用生产任务 | 可作为 Haiku 教师 |
| Opus | 更大规模上限模型 | 更高成本、更慢 | 复杂推理、难任务上限 | 公开资料未确认直接用于 Bedrock 蒸馏 |

这里要严格区分“直观推测”和“官方确认”：**官方确认的是 Sonnet 到 Haiku 的蒸馏流程**；“Opus 也可能在更大训练链路里提供知识”是合理推测，但不是当前公开文档里的明确结论。

---

## 问题定义与边界

这篇文章讨论的问题很具体：**如何让 Claude 3 Haiku 在不提高线上推理成本的前提下，尽量接近更大 Claude 模型在某些任务上的表现**。

边界也要先说清楚：

| 边界项 | 本文范围 |
| --- | --- |
| 部署平台 | Amazon Bedrock |
| 模型家族 | Claude 3 / 3.5 家族中的 Haiku、Sonnet |
| 目标能力 | RAG、数据分析、重复型高频任务的质量提升 |
| 不讨论内容 | 其他厂商模型、离线自训练、从零预训练 |

新手版可以把它想成：Haiku 是跑得快的学生，Sonnet 是理解更深的老师。蒸馏不是给学生换更大的脑子，而是把老师最有价值的判断模式压缩过去，让学生在原本的硬件预算内答得更像老师。

因此，蒸馏解决的不是“模型能不能无限变强”，而是一个工程问题：当业务要求**低延迟、低单次成本、高并发**时，怎样让便宜模型少损失关键能力。

---

## 核心机制与推导

“softmax”是把一组原始分数转成概率分布的方法。模型最后输出的原始分数叫“logits”，白话说就是“每个候选答案的内部打分”。蒸馏先不看最终文字，而是先看这些分数对应的概率。

教师和学生在温度 $T$ 下的分布分别是：

$$
p_t^T=\mathrm{softmax}(z_t/T), \qquad
p_s^T=\mathrm{softmax}(z_s/T)
$$

这里的“温度”不是物理温度，而是**控制分布尖锐程度的缩放系数**。当 $T>1$ 时，分布会更平滑，次优选项的概率会被放大，学生因此能看到更多“差一点就对”的信息。

最常见的蒸馏目标写成：

$$
L=\alpha L_{\mathrm{CE}}+(1-\alpha)\,T^2\mathrm{KL}(p_t^T \parallel p_s^T)
$$

其中：

- $L_{\mathrm{CE}}$ 是交叉熵，白话说就是“逼学生对真实答案负责”
- $\mathrm{KL}$ 是 KL 散度，白话说就是“逼学生模仿老师整组概率分布”
- $\alpha$ 控制“学标准答案”和“学老师风格”的权重
- 前面的 $T^2$ 用来补偿高温下梯度缩小，否则温度一升，蒸馏项容易变得太弱

看一个玩具例子。设教师 logits 为 $[3,1,0]$：

- 当 $T=1$ 时，softmax 约为 $[0.844, 0.114, 0.042]$
- 当 $T=2$ 时，softmax 约为 $[0.629, 0.231, 0.140]$

差别在于：第二、第三个选项不再几乎被压成 0。学生会学到“老师虽然选第一个，但第二个也有一定合理性”。这就是暗知识迁移。

可以把流程压缩成一张表：

| 步骤 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| 1 | 教师 logits $z_t$ | $p_t^T$ | 生成软标签 |
| 2 | 学生 logits $z_s$ | $p_s^T$ | 生成待对齐分布 |
| 3 | 真实标签 + 两组分布 | CE + KL | 同时学正确性和教师偏好 |
| 4 | 反向传播 | 学生参数更新 | 完成蒸馏 |

真实工程例子则是 Bedrock 的自动化流程：先根据示例任务生成合成 prompt-response 数据，再用 Sonnet 的输出监督 Haiku，最后把蒸馏后的 Haiku 部署成可直接推理的模型。这里迁移的重点不是“复制参数”，而是“复制行为分布”。

---

## 代码实现

下面是一个最小可运行的 Python 例子，用来演示温度缩放、KL 蒸馏项和 $T^2$ 补偿。它不是完整训练框架，但足够说明核心计算。

```python
import math

def softmax(logits, T=1.0):
    scaled = [x / T for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def cross_entropy(student_probs, true_index):
    return -math.log(student_probs[true_index])

def kl_divergence(p_teacher, p_student):
    eps = 1e-12
    total = 0.0
    for pt, ps in zip(p_teacher, p_student):
        pt = max(pt, eps)
        ps = max(ps, eps)
        total += pt * math.log(pt / ps)
    return total

teacher_logits = [3.0, 1.0, 0.0]
student_logits = [2.2, 1.4, 0.3]
true_index = 0
T = 2.0
alpha = 0.5

teacher_soft = softmax(teacher_logits, T)
student_soft = softmax(student_logits, T)
student_hard = softmax(student_logits, 1.0)

ce = cross_entropy(student_hard, true_index)
kl = kl_divergence(teacher_soft, student_soft)
loss = alpha * ce + (1 - alpha) * (T ** 2) * kl

assert round(sum(teacher_soft), 6) == 1.0
assert teacher_soft[1] > softmax(teacher_logits, 1.0)[1]
assert loss > 0

print("teacher_soft =", teacher_soft)
print("student_soft =", student_soft)
print("loss =", loss)
```

如果把它映射到 LLM 蒸馏训练 loop，可以理解成：

1. 输入一批 prompt
2. 用教师模型生成 logits 或软标签
3. 学生模型前向计算自己的 logits
4. 计算 $L_{\mathrm{CE}} + T^2 \cdot \mathrm{KL}$
5. 只更新学生，不更新教师

简化数据流如下：

| 阶段 | 数据 |
| --- | --- |
| 提示构造 | 业务 prompt、示例问答 |
| 教师生成 | Sonnet 输出软标签或响应 |
| 样本整理 | prompt + teacher target + hard label |
| 学生训练 | Haiku mini-batch 反向传播 |
| 验证上线 | 延迟、成本、幻觉率、任务准确率 |

---

## 工程权衡与常见坑

蒸馏不是“加一个 KL 就结束”。真正难的是让迁移发生，而不是让 loss 看起来下降。

| 常见坑 | 现象 | 规避策略 |
| --- | --- | --- |
| $T$ 太低 | 分布太尖，只学到“谁是第一名” | 从中等温度开始做网格搜索 |
| $T$ 太高 | 分布过平，训练方向变模糊 | 用验证集监控稳定性和收敛速度 |
| 漏乘 $T^2$ | 高温下蒸馏梯度太小 | 显式在 KL 项前补偿 |
| $\alpha$ 太大 | 退化成普通微调 | 保留足够蒸馏权重 |
| $\alpha$ 太小 | 学老师风格但偏离真实标签 | 结合真实标注校正 |
| 数据只靠硬标签 | 能力迁移有限 | 让教师参与合成数据生成 |
| 验证只看准确率 | 线上效果不稳 | 同时看幻觉率、格式稳定性、延迟 |

一个实用经验是：**蒸馏数据质量通常比公式细节更重要**。如果教师给出的样本覆盖面差、格式不一致、任务边界不清楚，学生只会学到噪声。

验证流程也不能只做离线打分。至少要检查三类指标：

| 验证项 | 目的 |
| --- | --- |
| 任务准确率 | 看是否真的接近教师 |
| 幻觉率/拒答稳定性 | 看安全边界是否被破坏 |
| 延迟与成本 | 确认蒸馏后仍满足部署目标 |

---

## 替代方案与适用边界

蒸馏不是唯一方案。最常见的替代方案是**直接微调 Haiku**。微调的意思是继续用任务数据训练原模型，但只用硬标签时，它更像“背答案”，不擅长复制大模型细腻的判断结构。

| 方案 | 优点 | 缺点 | 适合场景 |
| --- | --- | --- | --- |
| 纯微调 | 流程简单 | 难复制教师暗知识 | 标签清晰、任务单一 |
| 蒸馏 | 能迁移教师分布信息 | 流程更复杂、依赖教师 | 高频调用、低延迟服务 |
| 蒸馏 + 人类标签 | 质量更稳 | 成本更高 | 关键业务流程 |
| 直接用大模型 | 最省训练工作 | 推理贵、吞吐低 | 低频高价值任务 |

新手版可以这样区分：  
直接微调像让学生背教材；蒸馏像把老师做题时的信心分布和思路痕迹一起交给学生。

因此，蒸馏最适合：

- RAG 问答
- 数据分析助手
- 大批量分类、抽取、改写
- 对延迟和单次成本敏感的在线服务

不太适合：

- 全新任务、教师本身也不稳定的场景
- 需要持续开放式探索推理的任务
- 希望小模型完全替代大模型上限能力的场景

一句话说，蒸馏能让小模型更像大模型，但**不会免费创造出大模型本来没有的能力，也不会突破学生模型自身容量上限**。

---

## 参考资料

- Anthropic, *Claude 3.5 Haiku on AWS Trainium2 and model distillation in Amazon Bedrock*  
  https://claude.com/blog/trainium2-and-distillation
- Adaline Labs, *LLM Distillation Explained*  
  https://labs.adaline.ai/p/llm-distillation-explained
- Geoffrey Hinton, Oriol Vinyals, Jeff Dean, *Distilling the Knowledge in a Neural Network*  
  https://www.cs.toronto.edu/~hinton/absps/distillation.pdf
- Wikipedia, *Knowledge distillation*  
  https://en.wikipedia.org/wiki/Knowledge_distillation
- 维基百科，《知识蒸馏》  
  https://zh.wikipedia.org/wiki/%E7%9F%A5%E8%AD%98%E8%92%B8%E9%A4%BE
