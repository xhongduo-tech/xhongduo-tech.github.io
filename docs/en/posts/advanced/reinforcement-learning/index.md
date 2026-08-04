---
pageClass: plain-doc
---

# Reinforcement Learning

Following Sutton & Barto's *Reinforcement Learning: An Introduction* (2nd edition) as the main thread, this series breaks down, chapter by chapter, the classic theory and algorithms from multi-armed bandits to policy gradients, and extends into cutting-edge topics such as deep reinforcement learning, model-based RL, offline RL, multi-agent RL, and RLHF.

## Topic Planning

<ProgressGrid cat="advanced/reinforcement-learning" />


### Part 1 Introduction (Chapter 1)
- [ ] Reinforcement learning: a learning paradigm driven by trial and error and delayed rewards
- [ ] Elements of reinforcement learning: policy, reward signal, value function, and model of the environment
- [ ] Similarities and differences among reinforcement learning, supervised learning, and unsupervised learning

### Part 2 Multi-Armed Bandits (Chapter 2)
- [ ] The k-armed bandit problem and action-value methods
- [ ] ε-greedy policy and incremental implementation
- [ ] Optimistic initial values and upper confidence bound (UCB) action selection
- [ ] Gradient bandit algorithms
- [ ] Associative search and contextual bandits

### Part 3 Finite Markov Decision Processes (Chapter 3)
- [ ] The agent–environment interface and the Markov property
- [ ] Goals, rewards, returns, and discounting
- [ ] Unifying the notation for episodic and continuing tasks
- [ ] Value functions and the Bellman equations
- [ ] Optimal value functions, Bellman optimality equations, and optimal policies

### Part 4 Dynamic Programming (Chapter 4)
- [ ] Policy evaluation (prediction)
- [ ] Policy improvement and policy iteration
- [ ] Value iteration
- [ ] Asynchronous dynamic programming and generalized policy iteration (GPI)

### Part 5 Monte Carlo Methods (Chapter 5)
- [ ] Monte Carlo prediction and estimation of action values
- [ ] Monte Carlo control and exploring starts
- [ ] Off-policy prediction via importance sampling
- [ ] Off-policy Monte Carlo control
- [ ] Discounting-aware and per-decision importance sampling

### Part 6 Temporal-Difference Learning (Chapter 6)
- [ ] TD prediction: TD(0) and the one-step TD error
- [ ] Advantages of TD methods over Monte Carlo and dynamic programming
- [ ] Sarsa: on-policy TD control
- [ ] Q-learning: off-policy TD control
- [ ] Expected Sarsa
- [ ] Maximization bias and Double Learning

### Part 7 n-step Bootstrapping (Chapter 7)
- [ ] n-step TD prediction
- [ ] n-step Sarsa and off-policy n-step learning
- [ ] Per-decision methods with control variates
- [ ] n-step tree-backup algorithm and the unifying view: n-step Q(σ)

### Part 8 Planning and Learning with Tabular Methods (Chapter 8)
- [ ] Models and planning
- [ ] Dyna: integrating planning, acting, and learning
- [ ] When the model is wrong and prioritized sweeping
- [ ] Trajectory sampling and real-time dynamic programming
- [ ] Decision-time planning: heuristic search, rollout, and Monte Carlo tree search (MCTS)

### Part 9 Function Approximation for Prediction (Chapter 9)
- [ ] Value-function approximation and prediction objectives
- [ ] Stochastic-gradient and semi-gradient methods
- [ ] Linear methods: feature construction, polynomial bases, and Fourier bases
- [ ] Tile coding and coarse coding
- [ ] Radial basis functions and artificial neural networks
- [ ] Least-squares temporal difference (LSTD) and memory-based function approximation

### Part 10 On-policy Control with Approximation (Chapter 10)
- [ ] Episodic semi-gradient control and semi-gradient n-step Sarsa
- [ ] Average reward: a new problem setting for continuing tasks
- [ ] Differential semi-gradient n-step Sarsa

### Part 11 Off-policy Methods with Approximation (Chapter 11)
- [ ] A divergence example for semi-gradient methods under off-policy
- [ ] The Deadly Triad
- [ ] Gradient descent on the Bellman error and its unlearnability
- [ ] Gradient-TD methods and emphatic TD methods

### Part 12 Eligibility Traces (Chapter 12)
- [ ] The λ-return and TD(λ)
- [ ] The online λ-return algorithm and true online TD(λ)
- [ ] Sarsa(λ) and Dutch traces
- [ ] Variable λ and variable γ
- [ ] Off-policy eligibility traces with control variates
- [ ] Watkins's Q(λ) and tree-backup TB(λ)

### Part 13 Policy Gradient Methods (Chapter 13)
- [ ] Policy approximation and its advantages
- [ ] The policy gradient theorem
- [ ] REINFORCE: Monte Carlo policy gradient
- [ ] REINFORCE with baseline and Actor-Critic methods
- [ ] Policy parameterization for continuous action spaces

### Part 14 Deep Reinforcement Learning Topics
- [ ] DQN: deep Q-networks, experience replay, and target networks
- [ ] Double DQN and Dueling Networks
- [ ] Prioritized Experience Replay and Rainbow
- [ ] Distributional value functions: C51 and QR-DQN
- [ ] Advantage estimation and GAE (generalized advantage estimation)
- [ ] A3C/A2C: (asynchronous) advantage Actor-Critic
- [ ] TRPO: trust region policy optimization
- [ ] PPO: proximal policy optimization
- [ ] DDPG: deep deterministic policy gradient
- [ ] TD3: twin delayed deep deterministic policy gradient
- [ ] SAC: soft Actor-Critic
- [ ] IMPALA: scalable distributed Actor-Learner architecture

### Part 15 Model-Based Reinforcement Learning
- [ ] An overview of model-based RL: from Dyna to learning world models
- [ ] Monte Carlo tree search (MCTS) in depth
- [ ] AlphaGo/AlphaZero: combining self-play with tree search
- [ ] MuZero: planning with learned models and no rules

### Part 16 Offline Reinforcement Learning
- [ ] The offline RL problem setting: distribution shift and extrapolation error
- [ ] BCQ: batch-constrained Q-learning
- [ ] CQL: conservative Q-learning
- [ ] IQL: implicit Q-learning

### Part 17 Multi-Agent Reinforcement Learning
- [ ] Foundations of multi-agent games: Nash equilibria and stochastic games
- [ ] Cooperative multi-agent: VDN and QMIX
- [ ] Centralized training with decentralized execution (CTDE): MADDPG

### Part 18 Inverse Reinforcement Learning and Hierarchical Reinforcement Learning
- [ ] Inverse reinforcement learning: inferring reward functions from expert behavior
- [ ] Maximum-entropy inverse reinforcement learning and guided cost learning
- [ ] The options framework and semi-Markov decision processes
- [ ] Hierarchical reinforcement learning: feudal RL and FeUdal Networks

### Part 19 Reinforcement Learning and Large Language Models
- [ ] An RL perspective on RLHF: from human preferences to reward models
- [ ] PPO in practice for LLM alignment: KL constraints and training stability
- [ ] DPO and post-RLHF alignment algorithms: implicit rewards and direct preference optimization

> After finishing a write-up: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [标题](./xxx)`.
