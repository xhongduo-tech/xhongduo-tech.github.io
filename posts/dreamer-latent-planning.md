## 核心结论

Dreamer 的核心路线可以压成一句话：**先学世界模型，再在潜空间想象训练 actor-critic**。

这里的“世界模型”是指一个能预测未来状态、奖励和结束信号的内部环境模型；“潜空间”是指把原始图像、传感器等高维输入压缩成更紧凑的隐状态后，再在这个隐状态里做推演。Dreamer 不把主要训练预算花在真实环境反复试错上，而是先用少量真实数据学出一个可用的内部模拟器，再在模拟器里批量“脑内演练”。

如果把传统强化学习理解成“在真环境里一遍遍试动作”，Dreamer 更像“先学一个脑内沙盘，再在沙盘里练习”。它的直接收益是样本效率高，尤其适合真实交互昂贵、存在风险、奖励又不密集的任务。

下表先给出直观对比：

| 维度 | 传统 model-free RL | Dreamer |
|---|---|---|
| 学习对象 | 直接学策略/价值 | 先学世界模型，再学策略/价值 |
| 训练位置 | 主要在真实环境 | 真实环境 + 潜空间想象 |
| 样本效率 | 往往较低 | 往往更高 |
| 额外复杂度 | 相对低 | 需要训练动态模型 |
| 主要风险 | 真实试错成本高 | 模型误差累积 |

可以把主流程记成一条链：`观测 x_t -> 编码器 -> RSSM -> imagined rollout -> actor/critic`。DreamerV2 把这条路线扩展到 Atari，并在 200M 帧尺度达到人类水平；DreamerV3 进一步把训练稳定性和泛化能力做强，扩展到 Minecraft 等更复杂环境。

---

## 问题定义与边界

Dreamer 解决的问题不是“任何任务都能更强”，而是：**在有限真实交互预算下，如何尽量高效地学会控制策略**。

强化学习的基本输入是时间序列交互数据：观测 $x_t$、动作 $a_t$、奖励 $r_t$。目标是在长期累计回报意义下找到更好的策略。难点在于，真实环境往往贵、慢、危险，而且很多环境还是部分可观测的。所谓“部分可观测”，白话说就是当前一帧信息不够，你必须结合历史才能判断真实状态。

统一记号如下：

| 记号 | 含义 | 白话解释 |
|---|---|---|
| $x_t$ | 观测 | 相机图像、传感器读数等原始输入 |
| $a_t$ | 动作 | 智能体在时刻 $t$ 的控制输出 |
| $h_t$ | 确定性状态 | 记忆历史的隐藏向量 |
| $z_t$ | 随机潜变量 | 表示当前不确定信息的隐变量 |
| $s_t=(h_t,z_t)$ | 隐状态 | Dreamer 真正用来推演未来的内部状态 |

再看它的适用边界：

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 机器人控制 | 适合 | 真机试错贵，样本效率重要 |
| 视觉控制 | 适合 | 高维图像可先压到潜空间 |
| 部分可观测环境 | 适合 | RSSM 有显式时序记忆 |
| 简单低维密集奖励任务 | 未必划算 | 模型开销可能大于收益 |
| 极高精度长期规划任务 | 需谨慎 | 模型误差会沿时间累积 |

一个真实工程例子是机器人抓取。真实机械臂每试一次都消耗时间，还可能撞坏物体或夹具。Dreamer 的目标不是完全不做真机实验，而是把大量训练搬到潜空间里完成，用较少真实交互不断刷新模型。

---

## 核心机制与推导

Dreamer 的机制分三步：**学状态、做想象、用想象更新策略**。

第一步是用 RSSM 学潜变量动态模型。RSSM 全称 Recurrent State-Space Model，可以理解为“带记忆的状态空间模型”，即一部分状态负责记历史，一部分状态负责表达当前不确定性。其典型形式是：

$$
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
$$

$$
z_t \sim q_\phi(z_t \mid h_t, e_t), \qquad p_\phi(z_t \mid h_t)
$$

其中 $e_t$ 是编码后的观测特征。后验 $q_\phi$ 结合了当前观测，所以更接近真实；先验 $p_\phi$ 只根据历史和动作预测未来，所以它是想象阶段真正要用的模型。

世界模型训练时通常同时做三件事：

1. 重建观测，确认隐状态保留了足够信息。
2. 预测奖励，确认隐状态包含决策相关信息。
3. 用 KL 散度约束后验和先验，避免两套状态系统彼此脱节。

常见损失写成：

$$
L_{wm} = L_{rec} + L_{rew} + \beta \, KL\big(q_\phi(z_t\mid h_t,e_t)\,\|\,p_\phi(z_t\mid h_t)\big)
$$

这里的 KL 散度可以理解为“两个分布差多远”的量。它的作用不是让表示绝对准确，而是让“看见观测时得到的状态”和“只靠模型预测得到的状态”尽量对齐。因为后面做 imagination 时，模型拿不到真实未来观测，只能靠先验往前滚。

第二步是在潜空间 imagine rollout，也就是想象轨迹展开。给定某个起始隐状态 $\hat s_t$，策略产生动作，世界模型在潜空间里推下一个状态：

$$
s_{t+1} = g_\theta(s_t, a_t)
$$

同时预测奖励 $\hat r_{t+1}$ 和价值。这样就得到一条“内部模拟的未来轨迹”。Dreamer 不是在像素空间生成整帧视频再训练策略，而是在潜空间里直接做滚动，成本更低，也更稳定。

第三步是用想象轨迹训练 actor 和 critic。actor 的目标通常写成：

$$
J(\psi)=\mathbb{E}\Big[\sum_{k=1}^{H}\gamma^{k-1}\hat r_{t+k}+\gamma^H \hat V(\hat s_{t+H})\Big]
$$

这里 $\gamma$ 是折扣因子，白话说就是“越远的收益权重越低”；$\hat V(\hat s_{t+H})$ 是末端 bootstrap 价值，也就是“后面没继续展开的那一段，用 critic 估一个尾值补上”。

玩具例子最容易看清目标。设 $\gamma=0.9$，模型预测两步奖励分别为 $1$、$2$，末端价值为 $5$，那么：

$$
G_t = 1 + 0.9\times 2 + 0.9^2\times 5 = 6.85
$$

策略更新的方向，就是让它产生的动作把这个想象回报抬高。

Dreamer 能成立，关键在“想象过程可微”。可微的意思是损失变化能反向传到前面的参数。连续潜变量常用重参数化技巧：

$$
z = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

这样采样看似随机，但梯度还能通过 $\mu,\sigma$ 回传。DreamerV2 用离散 latent 时，常配 straight-through 近似，即前向按离散采样走，反向把它近似成可传梯度的连续操作。它不是严格数学可导，但工程上足够有效。

真实工程例子里，这意味着机器人并不是直接在真实世界里试出“抓这里会不会成功”，而是先把相机输入压成隐状态，再在隐状态里模拟未来十几步抓取结果，最后用这些想象结果更新策略。

---

## 代码实现

实现上通常拆成四层：编码器、RSSM、解码/奖励头、actor-critic。训练顺序一般是先更新世界模型，再从某些起始隐状态出发做 imagined rollout，最后更新 actor 和 critic。

训练循环可以概括成：

| 步骤 | 做什么 | 目的 |
|---|---|---|
| 采样真实数据 | 从 replay 取序列 | 提供真实监督信号 |
| 更新世界模型 | 观察、重建、奖励预测、KL 对齐 | 学会内部动态 |
| 生成 imagined trajectories | 从隐状态展开未来 | 构造低成本训练轨迹 |
| 更新 actor/critic | 最大化想象回报、拟合价值 | 学控制策略 |

关键接口通常长这样：

| 接口 | 作用 |
|---|---|
| `observe()` | 用真实序列推后验状态与先验状态 |
| `imagine()` | 从起始状态在潜空间滚动未来 |
| `policy()` | 根据隐状态输出动作分布 |
| `value()` | 估计隐状态下的未来回报 |

一个最小伪代码如下：

```python
for batch in replay:
    post, prior = world_model.observe(batch.obs, batch.act, batch.rew)
    loss_wm = recon_loss(post, batch.obs) + reward_loss(post, batch.rew) + kl_loss(post, prior)
    update(world_model, loss_wm)

    imagined = world_model.imagine(actor, post.last_state(), horizon=15)
    loss_actor = actor_loss(imagined)
    loss_critic = critic_loss(imagined)
    update(actor, loss_actor)
    update(critic, loss_critic)
```

下面给一个可运行的 Python 玩具代码，只演示“想象回报如何计算”，不依赖深度学习框架：

```python
def imagined_return(rewards, terminal_value, gamma):
    total = 0.0
    for k, r in enumerate(rewards):
        total += (gamma ** k) * r
    total += (gamma ** len(rewards)) * terminal_value
    return total

g = imagined_return([1.0, 2.0], terminal_value=5.0, gamma=0.9)
assert abs(g - 6.85) < 1e-9

def choose_better_plan():
    plan_a = imagined_return([1.0, 2.0], 5.0, 0.9)
    plan_b = imagined_return([0.5, 3.0], 4.0, 0.9)
    return "A" if plan_a > plan_b else "B"

assert choose_better_plan() == "A"
```

这个例子表达的本质是：策略不一定非要等真实结果出来再更新，它可以先比较多个“脑内计划”的预测回报，再朝预测更优的方向改。

---

## 工程权衡与常见坑

Dreamer 最核心的工程风险是**模型误差累积**。如果第 1 步就把障碍物位置估错一点，第 20 步的想象轨迹可能已经完全偏离真实环境。此时 actor 学到的不是“真实世界里的好动作”，而是“模型幻觉里的好动作”。

常见坑与对策如下：

| 风险 | 现象 | 常见对策 |
|---|---|---|
| 模型误差累积 | 想象越长越假 | 控制 horizon，持续刷新 replay |
| 重建压过任务信号 | 会还原像素，不会决策 | KL balancing、latent bottleneck |
| 奖励尺度不稳 | actor/critic 震荡 | 归一化、symlog、two-hot |
| 离散 latent 梯度近似 | 训练有偏差 | 接受近似，增强正则与数据覆盖 |
| 起始状态分布偏移 | 想象脱离真实访问区域 | 从真实后验状态启动想象 |

这里的“latent bottleneck”可以白话理解成“故意把信息通道收窄”，逼模型只保留对控制有用的信息，而不是把所有视觉细节都背下来。DreamerV3 之所以更稳，重点就在于把这些训练细节系统化，例如奖励和值目标的尺度处理。

排查不稳定训练时，优先看这几项：

| 检查项 | 典型症状 |
|---|---|
| KL 是否塌缩或爆炸 | 状态表示失真，重建或奖励突然变差 |
| imagined horizon 是否过长 | 训练初期回报高但真实评估差 |
| reward head 是否收敛 | actor 学不到明确优化方向 |
| replay 是否过旧 | 世界模型跟不上当前策略分布 |
| value target 是否尺度失衡 | critic loss 持续大幅振荡 |

---

## 替代方案与适用边界

Dreamer 不是唯一可行路线。它和 model-free RL、显式规划型 MBRL、MuZero 类方法的差异很明确。

| 方法 | 优点 | 代价 | 适用边界 |
|---|---|---|---|
| Dreamer | 样本效率高，适合高维观测 | 训练链路复杂 | 昂贵交互、视觉控制、部分可观测 |
| Model-free RL | 实现直接，少一层模型误差 | 样本消耗大 | 简单环境、便宜试错 |
| 显式 planning MBRL | 规划解释性强 | 在线规划成本高 | 低维动力学、可精确建模 |
| MuZero 类方法 | 模型与价值深度耦合 | 通常依赖搜索 | 棋类、结构化决策问题 |

再看场景边界：

| 场景 | Dreamer 是否占优 |
|---|---|
| 昂贵机器人实验 | 通常更占优 |
| 图像输入控制任务 | 通常更占优 |
| 简单小游戏 | 未必 |
| 极端长时延奖励 | 需要更强稳定性设计 |
| 模型极难学准的环境 | 优势可能下降 |

结论可以压成三句。

第一，环境越贵、越慢、越难反复试错，Dreamer 越有价值。  
第二，环境越简单、奖励越密集、真实交互越便宜，Dreamer 的额外模型成本越可能不划算。  
第三，Dreamer 的上限不只取决于策略优化，还取决于世界模型有没有学到“对控制真的有用”的潜表示。

---

## 参考资料

1. [Dream to Control: Learning Behaviors by Latent Imagination](https://research.google/pubs/dream-to-control-learning-behaviors-by-latent-imagination/)
2. [Introducing Dreamer: Scalable Reinforcement Learning Using World Models](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/)
3. [Mastering Atari with Discrete World Models](https://research.google/pubs/mastering-atari-with-discrete-world-models/)
4. [Mastering Atari with Discrete World Models Blog](https://research.google/blog/mastering-atari-with-discrete-world-models/)
5. [Mastering Diverse Domains through World Models](https://www.nature.com/articles/s41586-025-08744-2)
6. [Dreamer Official Code](https://github.com/google-research/dreamer)
7. [DreamerV2 Official Code](https://github.com/danijar/dreamerv2)
8. [DreamerV3 Official Code](https://github.com/danijar/dreamerv3)
