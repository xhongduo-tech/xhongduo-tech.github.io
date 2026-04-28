## 核心结论

课程学习（Curriculum Learning）不是简单地把样本按难度排个队，而是在训练过程中主动控制采样分布。这里的“采样分布”就是模型每一轮更常遇到哪类任务。它的核心目标是让模型持续处在“已经有一定成功率，但还没有学会”的区间。

用统一记号表示，就是让训练分布 $p_t(z)$ 随时间 $t$ 演化：前期偏向简单任务，中期增加过渡任务，后期再覆盖复杂任务。这里的 $z$ 可以表示一道题、一个任务配置、一个机器人起始状态，或者一组环境参数。

$$
p_t(z): \text{from easy} \rightarrow \text{hard as } t \text{ increases}
$$

它主要解决两类问题：

1. 训练早期反馈太稀疏，模型几乎学不到东西。
2. 任务本身太难，或者仿真和真实环境差异太大，直接训练不稳定。

一个玩具例子最容易理解。先做 10 道一眼能算出的加法，再做带进位加法，最后做混合运算。目的不是“偷看答案”，而是先建立稳定正确的局部能力，再把这个能力迁移到更复杂的问题。

一个真实工程例子是机器人抓取。训练初期只给无遮挡、固定光照、固定相机角度的物体；之后再加入遮挡、杂乱背景、材质变化和相机扰动。这样模型不会在一开始就被复杂场景压垮。

下表给出四类常见方法的总览。

| 方法 | 目标 | 输入 | 适用场景 | 主要风险 |
|---|---|---|---|---|
| 课程学习 | 从易到难组织训练 | 难度定义 $d(z)$ | 难度可排序的监督学习或强化学习 | 排序不准导致误导 |
| 自动课程学习 | 根据当前能力自动选任务 | 能力指标 $c_t$、任务策略 $\pi_\phi$ | 任务空间大、人工排课困难 | 分布学偏，泛化变差 |
| Domain Randomization | 提升跨环境泛化 | 环境参数分布 $P(\xi)$ | sim-to-real、视觉鲁棒性 | 随机化过宽，学成弱策略 |
| Reverse Curriculum | 解决稀疏奖励探索 | 目标状态 $g$、可回溯起点 | 机器人控制、导航、操作任务 | 目标附近状态难构造 |

---

## 问题定义与边界

先统一记号。

| 记号 | 含义 | 白话解释 |
|---|---|---|
| $z$ | 任务、起始状态或环境参数 | 一次训练样本到底长什么样 |
| $c_t$ | 第 $t$ 轮模型能力 | 当前学到了什么水平 |
| $d(z)$ | 难度函数 | 这个任务有多难 |
| $p_t(z)$ | 第 $t$ 轮采样分布 | 当前更常抽到什么任务 |
| $\xi$ | 环境随机变量 | 光照、摩擦、质量、相机角度这类可变因素 |
| $g$ | 目标状态 | 想达到的最终状态 |

课程学习解决的是“训练过程如何组织”，不是“模型结构如何设计”。它改善的是优化路径，也就是模型沿着什么顺序接触数据；它并不保证最终上限一定更高。如果模型容量不足、标签质量差、奖励定义错误，只靠课程学习通常救不回来。

这件事的边界要说清楚：课程学习更适用于可分级、可度量、可动态采样的问题。换句话说，你至少要能回答三个问题：

1. 哪些任务相对简单，哪些更难。
2. 模型当前处在什么能力水平。
3. 训练过程中能不能改变采样策略。

新手版例子：不是题目越难越好，而是题目难度要和当前水平匹配。如果一开始全是竞赛题，正确反馈极少；如果永远只做最简单题，能力不会增长。

工程版例子：图像分类可以按清晰度、遮挡程度、类别相似度组织课程，但如果标签本身噪声很大，排序不能修复错误标注。此时主要矛盾不在课程，而在数据质量。

课程学习和普通数据打乱也不同。普通打乱假设样本顺序不重要，只要总体分布固定就行；课程学习则显式承认顺序重要，并主动让训练分布随时间变化。也就是：

- 普通打乱：训练期间总体分布基本不变。
- 课程学习：训练期间总体分布是被控制着变化的。

---

## 核心机制与推导

课程学习的基本思想是让采样分布跟着能力走。若 $d(z)$ 是任务难度，$c_t$ 是第 $t$ 轮能力，那么一个直观目标是让模型主要看到满足“略高于当前能力”的任务。可以写成一种非严格但实用的思路：

$$
p_t(z) \propto \exp\left(-\frac{|d(z)-\alpha c_t|}{\tau}\right)
$$

其中 $\alpha$ 控制目标难度随能力提升的速度，$\tau$ 控制分布有多尖锐。意思很直接：当前能力附近的任务采样概率更高。

最小玩具例子如下。设三类任务的难度分别为：

- $d(A)=0.2$
- $d(B)=0.5$
- $d(C)=0.8$

如果早期能力 $c_0=0.3$，可以设置：

$$
p_0(A,B,C)=(0.7,0.25,0.05)
$$

当能力提升到 $c_1=0.7$，则转为：

$$
p_1(A,B,C)=(0.2,0.5,0.3)
$$

这里真正变化的不是模型，而是训练时它“更容易遇到什么”。

自动课程学习进一步把“谁先学什么”交给策略来决定。其核心形式是：

$$
z_t \sim \pi_\phi(z \mid c_t)
$$

这里 $\pi_\phi$ 是任务策略，也就是一个“出题器”。它根据当前能力 $c_t$ 选择下一个任务。这个出题器可以优化不同目标，比如最大化学习进展、维持目标成功率区间，或者平衡探索和巩固。

Domain Randomization 的控制对象不是任务难度，而是环境变化。这里的“随机化”不是胡乱加噪声，而是在合理物理范围内随机化训练环境，让策略学会对变化不敏感。目标通常写为：

$$
\min_\theta \mathbb{E}_{\xi \sim P(\xi)} \left[ L(f_\theta(o(\xi)), y) \right]
$$

其中 $\xi$ 可以是摩擦、质量、纹理、光照、相机位姿；$o(\xi)$ 是在参数 $\xi$ 下观察到的输入；$f_\theta$ 是模型；$L$ 是损失函数。其本质是把“真实世界的不确定性”提前注入训练分布。

Reverse Curriculum 针对的是稀疏奖励问题。所谓“稀疏奖励”，就是大多数尝试都得不到有效正反馈，模型很难知道自己是否接近成功。解决办法不是让模型从任意地方乱试，而是从目标附近开始学。因为在目标附近，成功更容易发生，学习信号更密集。

它的思路可以概括为：

1. 固定目标状态 $g$。
2. 从 $g$ 附近构造一批起始状态。
3. 训练到这些起点的成功率稳定。
4. 再把起点逐步向外扩展。

可以把它理解成“倒着铺路”。不是一开始就从最远端走到终点，而是先学最后几步，再学倒数十步，最后覆盖整个路径。

一个真实工程例子是机械臂放置任务。若奖励只在“物体被放进指定槽位”时给出，那么随机起点几乎学不到。Reverse Curriculum 会让机械臂先从“距离槽位很近”的位置开始，等成功率提高后，再让起点逐步远离槽位。

下面这张表对比四类机制。

| 方法 | 控制对象 | 控制方式 | 关键风险 |
|---|---|---|---|
| 课程学习 | 任务难度 | 人工或规则调度 $p_t(z)$ | 难度定义偏差 |
| 自动课程学习 | 任务选择策略 | 学习 $\pi_\phi(z \mid c_t)$ | 任务分布塌缩 |
| Domain Randomization | 环境参数 $\xi$ | 从 $P(\xi)$ 采样 | 范围失真 |
| Reverse Curriculum | 起始状态 | 从目标附近向外扩展 | 起点不可回溯 |

“方法选择图”可以直接写成规则：

| 条件 | 优先方法 |
|---|---|
| 任务天然可排序，难度可定义 | 排序式课程学习 |
| 任务空间巨大，人工排课成本高 | 自动课程学习 |
| 主要问题是 sim-to-real 或环境不确定性 | Domain Randomization |
| 主要问题是稀疏奖励，且目标附近可构造起点 | Reverse Curriculum |

---

## 代码实现

代码层面最关键的问题不是“概念上懂不懂”，而是训练循环里怎么接入课程策略。下面用一个最小可运行的 Python 例子展示三件事：

1. 静态课程如何按阶段更新采样权重。
2. 自动课程如何根据能力选择任务。
3. 环境随机化和反向课程如何以接口形式接入训练。

```python
from dataclasses import dataclass
import random

@dataclass
class Task:
    name: str
    difficulty: float

tasks = [
    Task("easy_add", 0.2),
    Task("carry_add", 0.5),
    Task("mixed_ops", 0.8),
]

def stage_weights(epoch: int):
    # 静态课程：前期偏简单，中期偏中等，后期覆盖困难
    if epoch < 3:
        return [0.7, 0.25, 0.05]
    if epoch < 6:
        return [0.3, 0.5, 0.2]
    return [0.15, 0.35, 0.5]

def sample_task_by_weights(tasks, weights, n=1000):
    counts = {t.name: 0 for t in tasks}
    for _ in range(n):
        t = random.choices(tasks, weights=weights, k=1)[0]
        counts[t.name] += 1
    return counts

def auto_curriculum_sample(tasks, capability_t: float):
    # 自动课程：优先抽取“略高于当前能力”的任务
    scores = []
    for t in tasks:
        score = 1.0 / (abs(t.difficulty - capability_t) + 0.05)
        scores.append(score)
    return random.choices(tasks, weights=scores, k=1)[0]

def randomize_env():
    # Domain Randomization：随机化环境参数
    return {
        "friction": random.uniform(0.4, 1.2),
        "lighting": random.uniform(0.5, 1.5),
        "camera_yaw_deg": random.uniform(-12.0, 12.0),
    }

def reverse_curriculum_starts(goal=0.0, radius=0.1, n=5):
    # Reverse Curriculum：从目标附近采样起点
    return [goal + random.uniform(-radius, radius) for _ in range(n)]

# 验证静态课程确实从易到难迁移
early = sample_task_by_weights(tasks, stage_weights(0), n=2000)
late = sample_task_by_weights(tasks, stage_weights(8), n=2000)
assert early["easy_add"] > early["mixed_ops"]
assert late["mixed_ops"] > late["easy_add"]

# 验证自动课程更偏向当前能力附近任务
picked = [auto_curriculum_sample(tasks, capability_t=0.55).name for _ in range(500)]
assert picked.count("carry_add") > picked.count("easy_add")

# 验证环境随机化范围合法
env = randomize_env()
assert 0.4 <= env["friction"] <= 1.2
assert 0.5 <= env["lighting"] <= 1.5
assert -12.0 <= env["camera_yaw_deg"] <= 12.0

# 验证反向课程起点围绕目标
starts = reverse_curriculum_starts(goal=1.0, radius=0.2, n=50)
assert all(0.8 <= s <= 1.2 for s in starts)
```

按难度分桶的最简单做法如下：

| difficulty 区间 | 阶段 |
|---|---|
| 0.0 - 0.3 | early stage |
| 0.3 - 0.6 | middle stage |
| 0.6 - 1.0 | late stage |

训练主循环可以写成下面的伪代码：

```python
for epoch in range(num_epochs):
    capability_t = evaluate_success_rate(model)

    # 1. 更新课程分布 p_t(z)
    sampler.update(capability_t)

    # 2. 更新环境参数范围
    env_cfg = randomization_schedule(epoch)

    # 3. 如果是稀疏奖励任务，更新反向课程起点集合
    start_states = reverse_curriculum.expand_if_ready(capability_t)

    for step in range(steps_per_epoch):
        z_t = task_policy.sample(condition=capability_t)   # 自动课程
        batch = dataset.sample(z_t, sampler=sampler)
        obs = env.reset(config=env_cfg, starts=start_states)
        loss = train_step(model, batch, obs)
```

真实工程里，课程策略通常挂在两种位置：

1. `DataLoader` 级别：修改样本权重或桶采样概率。
2. `Env.reset()` 级别：修改环境参数范围和起始状态分布。

一个机器人抓取任务的随机化配置片段可以写成：

```text
friction:      [0.4, 1.2]
lighting:      [0.5, 1.5]
camera_pose_x: [-0.03, 0.03]
camera_pose_y: [-0.03, 0.03]
camera_yaw:    [-12deg, 12deg]
object_texture: random
background:    random
```

Reverse Curriculum 的扩展过程则可理解为：

```text
goal g
-> sample starts within radius 0.05
-> success stable
-> expand radius to 0.10
-> success stable
-> expand radius to 0.20
-> until full workspace covered
```

这里的工程要点不是把所有复杂度一次性打开，而是每个阶段只引入“下一个主要挑战”。

---

## 工程权衡与常见坑

课程学习失败，很多时候不是因为理念错，而是因为难度定义、推进节奏、采样偏置出了问题。

经验上最危险的两个问题是：

1. 推进过快，前一阶段能力还没稳住就升档。
2. 随机化过宽，模型被迫学成对所有情况都不够好的平均化策略。

新手版类比可以直接说：如果还不会走，就让他直接跑马拉松，多半只会形成错误动作；但如果永远只在平地慢走，也学不会上坡、转弯和变速。

工程版例子：机器人抓取里，如果一开始就加入强遮挡、强反光、强相机扰动，策略常见结果不是“更强”，而是收敛到保守动作，最后什么都抓不稳。

下面是常见坑位表。

| 坑位 | 后果 | 规避方式 |
|---|---|---|
| 难度排序错误 | 模型卡在假难题上 | 用成功率、loss、learning progress 校验难度定义 |
| 推进过快 | 前序能力不稳，后期反复震荡 | 每阶段设置成功率门槛再进入下一阶段 |
| Domain Randomization 过宽 | 学成弱策略，收敛变慢 | 随机范围覆盖真实分布，不要脱离物理常识 |
| Reverse Curriculum 目标不可回溯 | 无法稳定生成起点，课程中断 | 先验证目标附近状态能否通过可行动作构造 |
| 自动课程分布偏置 | 过拟合局部任务，泛化差 | 保留一部分均匀采样或难例采样做校正 |

两条实用规则很重要。

第一条是成功率门槛规则。比如某阶段任务成功率连续若干轮超过 80%，再进入下一阶段。这个门槛不是理论定值，但必须有，否则课程推进会凭感觉。

第二条是分布校正规则。即使当前主要用课程分布 $p_t(z)$，也要保留一部分均匀采样或难例采样。因为如果训练分布过度偏移，模型可能在主课程上看起来很好，但一旦脱离课程轨道就明显退化。

---

## 替代方案与适用边界

课程学习不是唯一解。它和难例挖掘、重加权、数据增强、领域自适应都有交集，但焦点不同。

- 课程学习关注“训练顺序和阶段性分布”。
- 难例挖掘关注“哪些样本最值得反复训练”。
- Domain Randomization 关注“环境变化下的鲁棒性”。
- Reverse Curriculum 关注“稀疏奖励下如何获得可学习起点”。

如果只是背单词，间隔重复往往比课程学习更直接；如果是多步骤解题训练，课程学习通常更合适。因为前者核心问题是长期记忆保持，后者核心问题是能力分层构建。

真实工程里也一样。图像分类若边界样本很关键，难例挖掘经常比显式课程更有效；如果目标是 sim-to-real，Domain Randomization 往往更直接；如果是稀疏奖励控制任务，Reverse Curriculum 通常更有针对性。

方法选择表如下。

| 方法 | 适用条件 | 不适合的情况 |
|---|---|---|
| 课程学习 | 任务可排序，难度可定义 | 难度无法稳定估计 |
| 难例挖掘 | 边界样本决定性能 | 早期模型还没有基本能力 |
| 均匀采样 | 数据规模适中、分布稳定 | 稀疏奖励或极难任务 |
| Domain Randomization | 环境不确定性主导泛化 | 随机参数空间无法合理定义 |
| Reverse Curriculum | 目标附近可构造起点 | 目标状态不可回溯或不可安全生成 |

边界可以压缩成三句话：

1. 任务可排序时，课程学习更有效。
2. 目标状态附近可回溯时，Reverse Curriculum 更合适。
3. 环境不确定性主导泛化时，Domain Randomization 更合适。

再给一张“场景 -> 方法”映射表。

| 场景 | 更合适的方法 |
|---|---|
| 小学算术从无进位到混合运算 | 课程学习 |
| 大规模题库自动出题 | 自动课程学习 |
| 仿真机器人迁移到真实工厂 | Domain Randomization |
| 稀疏奖励机械臂放置 | Reverse Curriculum |
| 图像分类边界样本少 | 难例挖掘 |
| 数据质量一般但无明显难度层级 | 均匀采样或基础增强 |

因此，课程学习最值得用在“学习路径本身是瓶颈”的地方；如果主要瓶颈是数据错误、模型容量不足、奖励定义失败，那就该先修这些更底层的问题。

---

## 参考资料

下表按主题整理，便于把它当成延伸阅读清单。

| 论文名 | 作者 | 年份 | 解决的问题 | 链接 |
|---|---|---:|---|---|
| Curriculum Learning | Bengio et al. | 2009 | 系统提出从易到难组织训练样本 | https://dl.acm.org/doi/10.1145/1553374.1553380 |
| Automatic Curriculum Learning For Deep RL: A Short Survey | Portelas et al. | 2020 | 总结深度强化学习中的自动课程方法 | https://www.ijcai.org/Proceedings/2020/671 |
| Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World | Tobin et al. | 2017 | 用环境随机化缓解 sim-to-real 差异 | https://arxiv.org/abs/1703.06907 |
| Reverse Curriculum Generation for Reinforcement Learning | Florensa et al. | 2017 | 从目标附近反向生成起点，解决稀疏奖励探索 | https://arxiv.org/abs/1707.05300 |
| Reverse Curriculum Generation for Reinforcement Learning 页面 | CMU RI | 2017 | 作为作者机构页面补充核对信息 | https://publications.ri.cmu.edu/reverse-curriculum-generation-for-reinforcement-learning |

1. [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) 这篇文献定义了经典课程学习问题，即如何通过从易到难改善优化过程。  
2. [Automatic Curriculum Learning For Deep RL: A Short Survey](https://www.ijcai.org/Proceedings/2020/671) 这篇综述梳理了自动课程学习在深度强化学习中的主要路线。  
3. [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) 这篇工作说明了为什么随机化环境参数能帮助策略迁移到真实世界。  
4. [Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/abs/1707.05300) 这篇工作展示了如何从目标附近构造起点来解决稀疏奖励探索。  
5. [CMU Reverse Curriculum 页面](https://publications.ri.cmu.edu/reverse-curriculum-generation-for-reinforcement-learning) 这个页面适合快速核对作者、摘要与机构信息。
