## 核心结论

CQL（Conservative Q-Learning，保守 Q 学习）是一种面向离线强化学习的 critic 正则化方法。critic 可以理解为“打分器”，它负责估计在状态 $s$ 下执行动作 $a$ 的长期回报，也就是 $Q(s,a)$。CQL 的核心不是发明一套全新的 Bellman 更新，而是在标准 Q 学习损失外，额外加入一个“压低可疑动作分数”的惩罚项。

离线 RL 可以直白地理解为：只能用历史日志训练，训练过程中不能回到环境里继续试错。问题恰好出在这里。普通 Q 学习会利用 Bellman 备份不断向未来传播价值，但一旦网络把数据集外动作估高，后续备份就会放大这个错误，最终学出“日志里没见过、分数却很高”的动作。CQL 的作用，就是优先把这类 OOD 动作（out-of-distribution，分布外动作，指训练数据里很少出现或根本没出现的动作）压低。

它的代表性目标可以写成：

$$
L(Q)=\tfrac12\,\mathbb{E}_{D}\!\left[(Q-\hat B^\pi Q)^2\right]
+\alpha\Big(\mathbb{E}_{s\sim D,a\sim\mu(\cdot|s)}[Q(s,a)]
-\mathbb{E}_{s\sim D,a\sim\hat\pi_\beta(\cdot|s)}[Q(s,a)]\Big)
$$

其中第一项是 Bellman 误差，第二项是保守正则。$\alpha$ 控制保守强度。直观上，这一项会让“某种参考分布 $\mu$ 下的动作平均分”减去“数据行为策略下动作平均分”变小，于是数据动作相对更高，数据外动作相对更低。

一句话讲清楚为什么需要 CQL：训练日志里只有安全动作 A，普通离线 Q 学习却可能把没见过的动作 B 估得更高，CQL 就是把 B 的假高分先压回去。

下表可以先把三类常见方法区分清楚：

| 方法 | 学什么 | 对 OOD 动作的态度 | 典型结果 |
| --- | --- | --- | --- |
| 普通离线 Q 学习 | 直接做 Bellman 回归 | 可能误估过高 | 策略容易被假高分带偏 |
| CQL | Bellman 回归 + Q 正则化 | 主动压低 | 更稳，倾向保守提升 |
| 行为克隆 BC | 直接模仿动作 | 不做价值外推 | 最稳，但上限受示范限制 |

---

## 问题定义与边界

先把问题边界收紧。离线强化学习的数据集记为 $D$，它由历史轨迹组成，例如 $(s,a,r,s')$。这些轨迹通常来自某个行为策略 $\hat\pi_\beta$，也就是“当年记录日志时实际在执行的策略”。目标是学出一个新策略 $\pi$，希望它比历史行为更好，但训练时不能重新与环境交互。

相关符号如下：

| 符号 | 含义 |
| --- | --- |
| $D$ | 离线数据集，历史日志样本集合 |
| $\hat\pi_\beta$ | 经验行为策略，即生成数据的策略近似 |
| $\pi$ | 目标策略，即最终想部署的策略 |
| $B^\pi$ | 真实 Bellman 算子，用真实环境转移定义 |
| $\hat B^\pi$ | 经验 Bellman 算子，用数据样本近似 |

离线 RL 的核心困难叫分布偏移。白话说：训练时能看到的动作分布，和目标策略未来真正会选择的动作分布，往往不是一回事。只要目标策略开始偏向数据里少见的动作，Q 网络就在“自己没见过的区域”做外推。函数逼近一旦外推，误差通常不是温和增加，而是可能直接反号，低值动作被估成高值动作。

仓储机器人是一个典型真实工程例子。假设机器人只能用过去三个月的抓取日志学习，日志里的动作大多是低速、稳定、保守的电机控制序列。数据里几乎没有“急转弯 + 高加速度 + 极限伸臂”这种组合。普通离线 Q 学习可能因为网络插值外推，把这类组合估成高收益，因为它只看到局部特征，看不出组合动作会导致掉货或撞架。CQL 则会把这些没有数据支持的动作先压低，使策略主要沿着日志中真实出现过的区域优化。

这里必须明确边界。CQL 解决的是“Q 过高估计 OOD 动作”的问题，不是通用安全框架。它能改善离线策略提升的稳定性，但不能保证数据完全没覆盖的区域一定安全。论文中的理论下界依赖 support 条件。support 可以白话理解为“目标策略和参考分布关心的动作，数据里至少要有足够支持”，否则下界结论无从成立。

边界可以压缩成一张表：

| 维度 | CQL 能解决什么 | CQL 不能保证什么 |
| --- | --- | --- |
| 价值估计 | 压低 OOD 动作的假高 Q | 未覆盖区域上的真实精确值 |
| 策略学习 | 减少离线策略被外推误导 | 在极窄数据上仍稳定超越专家 |
| 安全性 | 降低“看起来很优”的风险动作被选中概率 | 对未知环境区域给出硬安全证明 |

---

## 核心机制与推导

CQL 的机制先于公式。普通 Bellman 更新会问：当前 $Q(s,a)$ 是否接近“即时奖励 + 下一步最优值”。CQL 在这个问题之外再加一个问题：你是不是把一些数据外动作估得太高了。如果是，就给这些动作额外罚分。

论文里一个更基础的形式可以写成：

$$
\hat Q_{k+1}\leftarrow \arg\min_Q
\alpha\,\mathbb{E}_{s\sim D,a\sim\mu(\cdot|s)}[Q(s,a)]
+\tfrac12\,\mathbb{E}_{D}\!\left[(Q-\hat B^\pi \hat Q_k)^2\right]
$$

这里的 $\mu$ 是参考分布，可以理解为“我们想保守对待的一批动作来自哪里”。如果让 $\mu$ 覆盖更广，它就更倾向于整体压低动作值。更常用的实现形式叫 CQL(H)：

$$
\alpha\,\mathbb{E}_{s\sim D}\!\left[\log\sum_a \exp Q(s,a)
-\mathbb{E}_{a\sim\hat\pi_\beta(\cdot|s)}Q(s,a)\right]
$$

$\log\sum_a \exp Q(s,a)$ 是一个 soft maximum，可以白话理解为“对所有动作高分区域的软上界聚合”。它会特别关注高 Q 动作，因为高值经过指数放大会占更大权重。减去数据动作的平均 Q 后，优化器就会倾向于保留数据动作的相对分数，同时压制那些没有数据支持却异常高的动作。

可以把推导逻辑理解成三层。

第一层是 point-wise pessimism，也就是点值保守。它追求某些动作上的 $Q$ 不要高于真实值。这个目标严格，但在复杂函数逼近下不一定最实用。

第二层是 policy value lower bound，也就是策略价值下界。真正关心的不是每个动作点值都比真值小，而是目标策略整体执行后的价值 $V^\pi$ 不被高估。CQL 的理论更强调这一层，这比“每个点都压得很低”更符合策略优化需求。

第三层是 gap-expanding。gap 就是间隔，指数据内动作和 OOD 动作之间的 Q 差距。CQL 的关键收益之一不是简单把所有动作都降下来，而是扩大“可信动作”和“可疑动作”的分数差，让策略更容易选回数据支持区域。

看一个玩具例子。假设只有一个状态 $s$，两个动作：

- 数据动作 $a_{\text{data}}$，真实回报为 $10$
- OOD 动作 $a_{\text{ood}}$，真实回报为 $2$

普通离线 Q 学习可能学到：

- $Q(s,a_{\text{data}})=8.5$
- $Q(s,a_{\text{ood}})=11$

这时贪心策略会错误选择 $a_{\text{ood}}$。加入 CQL 后，可能变成：

- $Q(s,a_{\text{data}})=8.5$
- $Q(s,a_{\text{ood}})=3$

现在策略会重新选择 $a_{\text{data}}$。同时 $8.5 \le 10$，说明对真正有数据支持的优动作，估计依然是保守的下界。

如果把流程画成文字图，就是：

| 阶段 | 发生什么 |
| --- | --- |
| Bellman backup | 用奖励和下一步价值更新当前 Q |
| softmax 压制 | 用 $\log\sum\exp$ 聚焦高分动作并施加惩罚 |
| gap 扩大 | 数据动作相对保留，OOD 动作相对下降 |

理论结论可以简化成两句。第一，在满足 support 等条件、并且正则足够强时，学到的 $Q$ 或 $V$ 可以成为真实值的下界。第二，当参考分布和目标策略匹配，即 $\mu=\pi$ 时，下界直接对应目标策略价值，而不是无关的动作集合均值。

---

## 代码实现

实现上，CQL 通常不是独立代码栈，而是往现有 actor-critic 或 value-based 框架里加一项 critic loss。最常见的是在 SAC 上加 CQL 正则；在离散动作场景，也可以叠加到 QR-DQN 这类方法上。

最小伪代码如下：

```text
初始化 Q 网络，若是 actor-critic 再初始化 policy
for each update:
    从离线数据集采样 batch
    计算 Bellman target
    计算 Bellman error
    计算 CQL 正则项，压低 OOD 动作 Q
    合并成 critic loss，更新 Q
    如果有 actor，再用当前 Q 更新 policy
```

几个核心开关最重要：

| 超参 | 作用 | 常见现象 |
| --- | --- | --- |
| `min_q_weight` / `alpha` | 控制保守强度 | 太小压不住，太大变得过悲观 |
| `lagrange_thresh` | 用拉格朗日方式自动调保守度 | 训练更自适应，但调试更复杂 |
| `min_q_version` | 选择 CQL 正则近似版本 | 不同实现数值特性不同 |

不同任务形态下的落地点也不完全一样：

| 场景 | 常见骨架 | 说明 |
| --- | --- | --- |
| SAC 版连续控制 | SAC + CQL(H) critic loss | D4RL MuJoCo 最常见 |
| QR-DQN 版离散动作 | 分布式 Q 学习 + CQL 正则 | 适合 Atari 等离散动作环境 |
| 大规模离线日志 | 双 Q + 目标网络 + 采样近似 | 关注稳定性与吞吐 |

下面给一个可以运行的 Python 玩具实现。它不依赖环境交互，只展示“普通贪心”和“保守贪心”的差异，并用 `assert` 验证 CQL 风格惩罚确实会压低 OOD 动作。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def cql_penalty(q_values, data_action, alpha=1.0):
    # CQL(H) 风格的单状态正则:
    # alpha * (logsumexp(Q) - Q(data_action))
    m = max(q_values)
    lse = m + math.log(sum(math.exp(q - m) for q in q_values))
    return alpha * (lse - q_values[data_action])

def one_step_cql_shrink(q_values, data_action, lr=0.5, alpha=1.0):
    # 对正则项做一小步梯度下降
    probs = softmax(q_values)
    new_q = q_values[:]
    for i in range(len(q_values)):
        grad = alpha * (probs[i] - (1.0 if i == data_action else 0.0))
        new_q[i] -= lr * grad
    return new_q

# 动作0是数据内动作，动作1是OOD动作
q = [8.5, 11.0]
penalty_before = cql_penalty(q, data_action=0, alpha=2.0)
q_new = one_step_cql_shrink(q, data_action=0, lr=1.0, alpha=2.0)
penalty_after = cql_penalty(q_new, data_action=0, alpha=2.0)

assert q[1] > q[0]          # 普通贪心会选错 OOD 动作
assert q_new[1] < q[1]      # CQL 惩罚压低了 OOD 动作
assert penalty_after < penalty_before
print("before:", q)
print("after :", q_new)
```

如果把论文里的配置翻成新手可跑的样子，连续动作场景里常会见到类似命令：

```bash
python examples/cql_mujoco_new.py \
  --env=hopper-medium-v0 \
  --policy_lr=1e-4 \
  --lagrange_thresh=-1.0 \
  --min_q_weight=5.0 \
  --min_q_version=3
```

实现上还有三个要点。第一，`CQL(H)` 往往比更原始的形式更常用，因为数值更稳定。第二，连续动作不能真的枚举所有动作，所以 `logsumexp` 通常要靠动作采样近似。第三，官方实现的核心思想就是“少改 critic，多加正则”，这也是它能成为强基准的重要原因。

---

## 工程权衡与常见坑

CQL 最常见的工程问题不是“不会跑”，而是“跑偏了但表面上还在收敛”。第一类问题是保守度过弱。$\alpha$ 太小时，OOD 动作的高估计仍会通过 bootstrapping 传播，最终策略继续乱选分布外动作。第二类问题是保守度过强。$\alpha$ 太大时，Q 会被整体压得过低，策略几乎不敢离开行为策略，效果看起来越来越像行为克隆。

连续动作场景下还有采样噪声问题。因为 $\log\sum\exp$ 很难精确算，通常只能从当前策略、均匀分布或行为分布附近采样一批动作近似。如果采样数太少，正则项就不是稳定约束，而是高方差噪声，训练会明显抖动。

下面是一张常用排障表：

| 现象 | 更可能的原因 | 常见处理 |
| --- | --- | --- |
| Q 全部偏低，动作都像不值得做 | `alpha` 过大 | 降低 `min_q_weight`，检查拉格朗日阈值 |
| 策略仍常选 OOD 动作 | `alpha` 过小或采样过少 | 增大保守权重，增加动作采样数 |
| critic 抖动很大 | `logsumexp` 近似方差高 | 提高采样质量，配合双 Q 和目标网络 |
| 最终效果像 BC | 约束过强 | 放松保守项，检查行为数据是否本身很窄 |

工程上建议固定看三个诊断指标：

1. 看 in-distribution 与 OOD 的 Q gap 是否被拉开。
2. 看 critic 是否“塌成一片”，也就是大量动作值挤在同一低区间。
3. 看行为策略估计和动作采样是否稳定，否则正则项本身不可信。

还要提醒一个常见误解。论文说“下界”，很多人就把它等价成“安全”。这不成立。下界针对的是理论假设下的价值估计关系，不是现实系统里的机械安全、电流安全、约束满足证明。对于机器人、推荐系统、自动驾驶这类真实系统，CQL 只是降低了“因为外推失真导致的错误乐观”，不是替代规则约束或风险控制模块。

如果把方法特性横向看，CQL 的位置大致如下：

| 维度 | CQL 表现 |
| --- | --- |
| 保守性 | 中到高，可调 |
| 稳定性 | 通常强于普通离线 Q 学习 |
| 表达能力 | 保留价值提升能力，不等于纯模仿 |
| 训练成本 | 高于 BC，略高于普通 SAC/Q-learning |

---

## 替代方案与适用边界

CQL 不是唯一的离线 RL 路线。离线方法大体可以分成三类：一类是像 CQL 这样直接改价值函数；一类是 BEAR、BRAC 这种显式约束 policy 不要偏离行为分布太远；还有一类是 BC 这种根本不做价值提升、只做动作模仿的方法。SPIBB 则更偏向保守 bootstrapping，可以理解为“只在有数据支持的地方放心更新”。

把它们放在一张表里更清楚：

| 方法 | 是否需要行为模型 | 是否显式约束 policy | 是否压低 OOD Q | 适合的数据形态 |
| --- | --- | --- | --- | --- |
| CQL | 通常不强依赖 | 不一定显式约束 | 是 | 混合来源、质量不一、OOD 风险高 |
| BEAR | 需要估计行为分布 | 是 | 间接 | 希望策略严格贴近数据分布 |
| BRAC | 需要行为策略或其近似 | 是 | 间接 | 明确要做行为约束 |
| SPIBB | 依赖支持计数或保守规则 | 部分是 | 部分是 | 离散或支持集可识别场景 |
| BC | 不需要价值学习 | 是，且最强 | 否 | 纯专家、小而干净的数据集 |

选择规则可以压成三句：

- 数据混合、日志很杂、OOD 风险高：优先试 CQL。
- 数据单一、近似纯专家轨迹、环境也不复杂：先试 BC，通常更稳更省。
- 明确要求策略不能偏离行为分布太远：优先考虑 BEAR、BRAC 这类 policy constraint 方法。

再给一个真实工程判断。假设你做工业调度，日志来自多个版本策略、人工接管和异常回退，数据分布混杂，这时直接做普通离线 Q 学习非常容易高估边缘动作，CQL 通常更合适。反过来，如果你只有一批高质量手工示范，例如固定装配流程里的机械臂轨迹，而且目标只是复现专家表现，那么 BC 往往比 CQL 更省心，因为你并不需要额外承担价值学习的方差。

最终边界要说得非常明确：CQL 解决的是“价值估计保守化”，不是对环境未知区域的硬安全证明。它是离线 RL 里非常强、非常实用的一把锤子，但不是所有离线问题都该用它敲。

---

## 参考资料

1. [Conservative Q-Learning for Offline Reinforcement Learning, NeurIPS 2020 Abstract](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
2. [CQL Paper PDF](https://papers.nips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf)
3. [CQL Supplemental PDF](https://proceedings.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Supplemental.pdf)
4. [Official CQL Repository](https://github.com/aviralkumar2907/CQL)
5. [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://sites.google.com/view/d4rl-anonymous/)
