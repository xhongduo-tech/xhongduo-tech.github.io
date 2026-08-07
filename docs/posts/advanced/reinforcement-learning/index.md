---
pageClass: plain-doc
---

# 强化学习

以 Sutton & Barto《强化学习（第2版）》为主线，逐节拆解从多臂老虎机到策略梯度的经典理论与算法，并延伸到深度强化学习、基于模型的 RL、离线 RL、多智能体 RL 与 RLHF 等前沿专题。

## 主题规划

<ProgressGrid cat="advanced/reinforcement-learning" />


### 第一篇 引言（第1章）
- [x] [强化学习：一种试错与延迟奖励驱动的学习范式](./rl-paradigm)
- [x] [强化学习的要素：策略、奖励信号、价值函数与环境模型](./rl-elements)
- [x] [强化学习与监督学习、无监督学习的异同](./rl-vs-supervised-unsupervised)

### 第二篇 多臂老虎机（第2章）
- [x] [k臂老虎机问题与动作价值方法](./k-armed-bandits)
- [x] [ε-贪心策略与增量式实现](./epsilon-greedy-incremental)
- [x] [乐观初始值与置信度上界（UCB）动作选择](./optimistic-initial-values-ucb)
- [x] [梯度老虎机算法](./gradient-bandits)
- [x] [关联搜索与上下文老虎机（Contextual Bandits）](./contextual-bandits)

### 第三篇 有限马尔可夫决策过程（第3章）
- [x] [智能体-环境交互接口与马尔可夫性质](./agent-environment-interface)
- [x] [目标、奖励、回报与折扣](./goals-rewards-returns-discounting)
- [x] [统一回合式任务与持续性任务的记号](./episodic-continuing-notation)
- [x] [价值函数与贝尔曼方程](./value-functions-bellman-equations)
- [x] [最优价值函数、贝尔曼最优方程与最优策略](./optimal-value-functions-bellman-optimality)

### 第四篇 动态规划（第4章）
- [x] [策略评估（预测）](./policy-evaluation)
- [x] [策略改进与策略迭代](./policy-improvement-policy-iteration)
- [x] [值迭代](./value-iteration)
- [x] [异步动态规划与广义策略迭代（GPI）](./asynchronous-dp-generalized-policy-iteration)

### 第五篇 蒙特卡洛方法（第5章）
- [x] [蒙特卡洛预测与动作价值估计](./monte-carlo-prediction-action-values)
- [x] [蒙特卡洛控制与探索起点](./monte-carlo-control-exploring-starts)
- [x] [基于重要性采样的离策略预测](./off-policy-prediction-importance-sampling)
- [x] [离策略蒙特卡洛控制](./off-policy-monte-carlo-control)
- [x] [折扣感知与每决策重要性采样](./discounting-aware-per-decision-importance-sampling)

### 第六篇 时序差分学习（第6章）
- [x] [TD预测：TD(0)与一步TD误差](./td-prediction-td0)
- [x] [TD方法与蒙特卡洛、动态规划的优势对比](./td-vs-mc-dp)
- [x] [Sarsa：同策略TD控制](./sarsa-on-policy-td-control)
- [x] [Q-learning：离策略TD控制](./q-learning-off-policy-td-control)
- [x] [期望Sarsa](./expected-sarsa)
- [x] [最大化偏差与Double Learning](./maximization-bias-double-learning)

### 第七篇 n步自举（第7章）
- [x] [n步TD预测](./n-step-td-prediction)
- [x] [n步Sarsa与离策略n步学习](./n-step-sarsa-off-policy)
- [x] [带控制变量的每决策方法](./per-decision-methods-control-variates)
- [x] [n步树回溯算法与统一视角：n步Q(σ)](./n-step-tree-backup-q-sigma)

### 第八篇 表格型规划与学习（第8章）
- [x] [模型与规划](./models-planning)
- [x] [Dyna：集成规划、行动与学习](./dyna-integrated-architecture)
- [x] [当模型出错时与优先扫描](./model-mistakes-prioritized-sweeping)
- [x] [轨迹采样与实时动态规划](./trajectory-sampling-rtdp)
- [x] [决策时规划：启发式搜索、Rollout与蒙特卡洛树搜索（MCTS）](./decision-time-planning-mcts)

### 第九篇 函数逼近预测（第9章）
- [x] [价值函数逼近与预测目标](./value-function-approximation)
- [x] [随机梯度与半梯度方法](./stochastic-gradient-semi-gradient)
- [x] [线性方法：特征构造、多项式基与傅里叶基](./linear-methods-features-polynomial-fourier)
- [x] [Tile Coding（瓦片编码）与粗编码](./tile-coding-coarse-coding)
- [x] [径向基函数与人工神经网络](./rbf-artificial-neural-networks)
- [x] [最小二乘时序差分（LSTD）与基于记忆的函数逼近](./lstd-memory-based-function-approximation)

### 第十篇 函数逼近控制（第10章）
- [x] [回合式半梯度控制与半梯度n步Sarsa](./episodic-semi-gradient-n-step-sarsa)
- [x] [平均奖励：持续性任务的新问题设定](./average-reward-setting)
- [x] [差分半梯度n步Sarsa](./differential-semi-gradient-n-step-sarsa)

### 第十一篇 离策略函数逼近（第11章）
- [x] [半梯度方法在离策略下的发散示例](./divergence-example-off-policy)
- [x] [致命三要素（The Deadly Triad）](./deadly-triad)
- [x] [贝尔曼误差的梯度下降与不可学习性](./bellman-error-gradient-descent)
- [x] [梯度TD方法与强调TD方法（Emphatic TD）](./gradient-td-emphatic-td)

### 第十二篇 资格迹（第12章）
- [x] [λ-回报与TD(λ)](./lambda-return-td-lambda)
- [x] [在线λ-回报算法与真在线TD(λ)](./online-lambda-return-true-online-td-lambda)
- [x] [Sarsa(λ)与荷兰迹](./sarsa-lambda-dutch-traces)
- [x] [变量λ与变量γ](./variable-lambda-gamma)
- [x] [带控制变量的离策略资格迹](./off-policy-traces-control-variates)
- [x] [Watkins的Q(λ)与树回溯TB(λ)](./watkins-q-lambda-tree-backup)

### 第十三篇 策略梯度方法（第13章）
- [x] [策略近似及其优势](./policy-approximation-advantages)
- [x] [策略梯度定理](./policy-gradient-theorem)
- [x] [REINFORCE：蒙特卡洛策略梯度](./reinforce-monte-carlo-policy-gradient)
- [x] [带基线的REINFORCE与Actor-Critic方法](./reinforce-baseline-actor-critic)
- [x] [连续动作空间的策略参数化方法](./continuous-action-policy-parameterization)

### 第十四篇 深度强化学习专题
- [x] [DQN：深度Q网络、经验回放与目标网络](./dqn-replay-target-network)
- [x] [Double DQN与Dueling Network](./double-dqn-dueling-network)
- [x] [优先经验回放（Prioritized Experience Replay）与Rainbow](./prioritized-experience-replay-rainbow)
- [x] [分布式价值函数：C51与QR-DQN](./distributional-value-c51-qr-dqn)
- [x] [优势估计与GAE（广义优势估计）](./advantage-estimation-gae)
- [x] [A3C/A2C：（异步）优势Actor-Critic](./a3c-a2c)
- [x] [TRPO：信赖域策略优化](./trpo)
- [x] [PPO：近端策略优化](./ppo)
- [x] [DDPG：深度确定性策略梯度](./ddpg)
- [x] [TD3：双延迟深度确定性策略梯度](./td3)
- [x] [SAC：软Actor-Critic](./sac)
- [x] [IMPALA：大规模分布式Actor-Learner架构](./impala)

### 第十五篇 基于模型的强化学习
- [x] [基于模型的RL总览：从Dyna到学习世界模型](./model-based-rl-overview)
- [x] [蒙特卡洛树搜索（MCTS）深入](./mcts-in-depth)
- [x] [AlphaGo/AlphaZero：自我对弈与树搜索的结合](./alphago-alphazero)
- [x] [MuZero：无需规则的学习模型规划](./muzero)

### 第十六篇 离线强化学习
- [x] [离线RL的问题设定：分布偏移与外推误差](./offline-rl-problem-setup)
- [x] [BCQ：批约束Q学习](./bcq)
- [x] [CQL：保守Q学习](./cql)
- [x] [IQL：隐式Q学习](./iql)

### 第十七篇 多智能体强化学习
- [x] [多智能体博弈基础：纳什均衡与随机博弈](./multi-agent-games-nash-stochastic-games)
- [x] [合作式多智能体：VDN与QMIX](./vdn-qmix)
- [x] [集中训练分布执行（CTDE）：MADDPG](./madppg)

### 第十八篇 逆强化学习与层次强化学习
- [x] [逆强化学习：从专家行为推断奖励函数](./inverse-reinforcement-learning)
- [x] [最大熵逆强化学习与引导代价学习](./maxent-irl-guided-cost-learning)
- [x] [选项框架（Options Framework）与半马尔可夫决策过程](./options-framework-semi-mdp)
- [x] [层次强化学习：封建式RL与FeUdal Networks](./feudal-rl-hierarchical)

### 第十九篇 强化学习与大型语言模型
- [x] [RLHF的RL视角：从人类偏好到奖励模型](./rlhf-reward-modeling)
- [x] [PPO在LLM对齐中的实践：KL约束与训练稳定性](./ppo-llm-alignment)
- [x] [DPO及后RLHF时代的对齐算法：隐式奖励与直接偏好优化](./dpo-direct-preference-optimization)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
