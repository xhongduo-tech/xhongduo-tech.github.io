---
pageClass: plain-doc
---

# 具身智能

具身智能（Embodied AI）研究让智能体通过物理身体与环境交互来感知、决策与行动，是连接人工智能与机器人学的前沿方向。本篇主题规划覆盖从机器人学基础、感知与 SLAM、运动规划与控制，到模仿学习、机器人强化学习、视觉-语言-动作模型（VLA）、世界模型与具身大模型的完整知识体系。

## 主题规划

<ProgressGrid cat="advanced/embodied-ai" />


### 第一篇 机器人学基础（对标 Craig《机器人学导论》、Lynch & Park《Modern Robotics》）

#### 第 1 章 空间描述与坐标变换
- [x] [位置、姿态与坐标系：旋转矩阵的性质与运算](./rotation-matrices)
- [x] [齐次变换矩阵与复合变换](./homogeneous-transformations)
- [x] [欧拉角、等效轴角与四元数表示](./euler-angles-quaternions)
- [x] [李群 SO(3)/SE(3)、李代数与指数映射：旋量（Twist）与螺旋运动](./lie-groups-twists)

#### 第 2 章 正运动学与逆运动学
- [x] [连杆参数与 D-H 约定](./dh-parameters)
- [x] [正运动学：连乘齐次变换与 POE（指数积）公式](./forward-kinematics-poe)
- [x] [逆运动学的解析解法：Pieper 准则与常见六轴构型](./inverse-kinematics-analytical)
- [x] [逆运动学的数值解法：牛顿迭代、阻尼最小二乘与冗余机械臂的零空间运动](./inverse-kinematics-numerical)

#### 第 3 章 雅可比矩阵与静力学
- [x] [速度传播与几何雅可比、解析雅可比](./jacobian-geometric-analytic)
- [x] [运动旋量与空间雅可比（基于 POE）](./spatial-jacobian-poe)
- [x] [奇异性分析与可操作度椭球](./singularity-manipulability)
- [x] [力雅可比与静力平衡：力-速度对偶性](./force-jacobian-statics)

#### 第 4 章 刚体动力学
- [x] [牛顿-欧拉递推动力学算法（RNEA）](./newton-euler-rnea)
- [x] [拉格朗日动力学与运动方程的结构性质：质量矩阵、科氏力与重力项](./lagrangian-dynamics-structure)
- [x] [浮动基座动力学与质心动力学（Centroidal Dynamics）](./floating-base-centroidal-dynamics)

### 第二篇 传感器与感知

#### 第 5 章 机器人传感器
- [x] [相机模型与标定：针孔模型、畸变与张正友标定法](./camera-model-calibration)
- [x] [深度相机：结构光、ToF 与双目立体视觉](./depth-cameras)
- [x] [激光雷达：测距原理、点云表示与多线扫描](./lidar-point-cloud)
- [x] [触觉传感与力/力矩传感器：从压阻式到视触觉（GelSight）](./tactile-force-sensing)

#### 第 6 章 感知与状态估计基础
- [x] [点云处理：滤波、ICP 配准与法向量估计](./point-cloud-processing)
- [x] [卡尔曼滤波、扩展卡尔曼滤波与粒子滤波](./kalman-filtering)
- [x] [2D/3D 物体检测与 6D 位姿估计](./object-detection-6dof-pose)

### 第三篇 SLAM（对标 Thrun《Probabilistic Robotics》、高翔《视觉 SLAM 十四讲》）

#### 第 7 章 SLAM 基础
- [x] [SLAM 问题建模：状态估计与图优化视角](./slam-problem-formulation)
- [x] [EKF-SLAM 与稀疏性分析](./ekf-slam-sparsity)
- [x] [基于图优化的 SLAM：位姿图、非线性最小二乘与 g2o/GTSAM 实践](./graph-based-slam)
- [x] [回环检测：词袋模型与描述子匹配](./loop-closure-detection)

#### 第 8 章 视觉与激光 SLAM 系统
- [x] [特征法视觉里程计：从 ORB-SLAM 到 ORB-SLAM3](./orbslam-visual-odometry)
- [x] [激光 SLAM：LOAM、LeGO-LOAM 与 FAST-LIO](./lidar-slam-loam)
- [x] [视觉-惯性里程计（VIO）：MSCKF 与 VINS-Mono](./vio-msckf-vins)
- [x] [神经隐式 SLAM 与 3D 高斯泼溅（3D Gaussian Splatting）建图](./nerf-gaussian-slam)

### 第四篇 运动规划（对标 LaValle《Planning Algorithms》）

#### 第 9 章 构型空间与基于采样的规划
- [x] [构型空间（C-Space）与障碍物](./configuration-space)
- [x] [概率路线图（PRM）与快速扩展随机树（RRT、RRT-Connect）](./prm-rrt-planning)
- [x] [渐近最优规划：RRT* 与 Informed RRT*](./asymptotically-optimal-planning)
- [x] [机械臂运动规划中的碰撞检测与运动学约束](./collision-detection-kinematic-constraints)

#### 第 10 章 轨迹优化
- [x] [多项式轨迹插值与时间参数化](./polynomial-trajectory-parametrization)
- [x] [轨迹优化问题建模：CHOMP、STOMP 与 TrajOpt](./trajectory-optimization-chomp)
- [x] [微分平坦与 Minimum Snap 轨迹](./minimum-snap-differential-flatness)

### 第五篇 控制

#### 第 11 章 经典控制与力控
- [x] [关节空间 PID 控制与前馈补偿](./joint-space-pid)
- [x] [计算力矩控制与逆动力学控制](./computed-torque-control)
- [x] [阻抗控制与导纳控制](./impedance-admittance-control)
- [x] [力/位混合控制与操作空间控制（Operational Space Control）](./operational-space-control)

#### 第 12 章 模型预测控制与全身控制
- [x] [线性模型预测控制（MPC）原理与实时迭代](./linear-mpc)
- [x] [基于倒立摆与 ZMP 的行走控制](./zmp-walking-control)
- [x] [质心 MPC 与四足机器人运动控制](./centroidal-mpc-quadruped)
- [x] [全身控制（Whole-Body Control）：任务优先级与 QP 求解](./whole-body-control)

### 第六篇 模仿学习

#### 第 13 章 模仿学习方法
- [x] [行为克隆（Behavior Cloning）：问题建模与协变量偏移](./behavior-cloning)
- [x] [DAgger：数据集聚合与交互式模仿学习](./dagger-interactive-imitation)
- [x] [扩散策略（Diffusion Policy）](./diffusion-policy)
- [x] [ACT（Action Chunking with Transformers）与动作分块](./act-action-chunking)
- [x] [遥操作数据采集系统：ALOHA 与 Mobile ALOHA](./aloha-teleoperation)

### 第七篇 机器人强化学习与 Sim2Real

#### 第 14 章 机器人强化学习
- [x] [机器人 RL 的问题建模：观测、动作空间与奖励设计](./robot-rl-problem-formulation)
- [x] [深度强化学习算法在机器人上的应用：PPO、SAC 与 TD3](./deep-rl-algorithms-robots)
- [x] [离线强化学习：CQL、IQL 与 Decision Transformer](./offline-rl-cql-iql)
- [x] [课程学习（Curriculum Learning）与分层强化学习](./curriculum-hierarchical-rl)

#### 第 15 章 仿真到现实迁移（Sim2Real）
- [x] [域随机化（Domain Randomization）](./domain-randomization)
- [x] [系统辨识与动力学参数校准](./system-identification)
- [x] [域自适应与对抗式迁移](./domain-adaptation-transfer)
- [x] [四足机器人运动控制的 Sim2Real 实践：从 ANYmal 到 Unitree](./sim2real-quadruped)

### 第八篇 操作与移动

#### 第 16 章 灵巧操作（Dexterous Manipulation）
- [x] [抓取基础：力封闭、抓取质量指标与 GraspNet 类抓取生成](./grasping-force-closure)
- [x] [多指灵巧手建模与控制：Shadow Hand 与 Allegro Hand](./dexterous-hands)
- [x] [基于强化学习的灵巧操作：OpenAI 魔方手案例分析](./rl-dexterous-manipulation-cube)
- [x] [触觉反馈与接触丰富的操作策略](./tactile-contact-rich-manipulation)

#### 第 17 章 移动操作（Mobile Manipulation）
- [x] [移动底盘与机械臂的协同控制：全身运动规划与导航-操作耦合](./mobile-manipulation-integration)
- [x] [视觉伺服（Visual Servoing）：IBVS 与 PBVS](./visual-servoing)
- [x] [开放场景物体搜寻与语义导航](./open-vocabulary-semantic-navigation)

### 第九篇 视觉-语言-动作模型（VLA）与具身大模型

#### 第 18 章 视觉-语言-动作模型
- [x] [RT-1：大规模真实机器人数据上的机器人 Transformer](./rt1-robotic-transformer)
- [x] [RT-2：视觉-语言模型的动作化迁移](./rt2-vla)
- [x] [OpenVLA：开源 VLA 模型的架构与微调](./openvla)
- [x] [Octo：通用机器人策略与多机器人数据混合](./octo-robot-policy)
- [x] [π0：基于流匹配（Flow Matching）的 VLA 架构](./pi0-flow-matching)
- [x] [机器人数据集：Open X-Embodiment 与 DROID](./open-x-embodiment-droid)

#### 第 19 章 世界模型与视频预测
- [x] [世界模型基础：从 Dreamer 到生成式驾驶世界模型](./world-models-dreamer)
- [x] [基于视频预测的机器人策略学习：UniPi 与后续工作](./unipi-video-prediction-policies)
- [x] [交互式世界模型与模型预测规划的结合](./interactive-world-models-mpc)

#### 第 20 章 具身大模型与分层架构
- [x] [具身智能中的分层架构：高层规划与低层控制](./hierarchical-architecture-embodied)
- [x] [SayCan：语言模型可供性（Affordance）落地](./saycan)
- [x] [Code as Policies：用代码生成机器人策略](./code-as-policies)
- [x] [VoxPoser 与基于 3D 价值图的任务规划](./voxposer)
- [x] [具身多模态大模型：从 PaLM-E 到 Gemini Robotics](./palm-e-gemini-robotics)

### 第十篇 仿真平台、人形机器人与评测

#### 第 21 章 仿真平台
- [x] [MuJoCo：接触动力学建模与 Python API 实践](./mujoco)
- [x] [Isaac Gym 与 Isaac Sim：GPU 并行仿真与大规模 RL 训练](./isaac-gym-sim)
- [x] [Habitat 与室内具身导航仿真](./habitat-navigation-sim)
- [x] [操作学习仿真基准：ManiSkill 与 RoboSuite](./maniskill-robosuite)

#### 第 22 章 人形机器人（Humanoid Robots）
- [x] [人形机器人的硬件构型与驱动技术](./humanoid-hardware)
- [x] [双足行走控制：从 ZMP 到基于学习的全身控制](./bipedal-walking-control)
- [x] [代表性平台解析：Atlas、Optimus、Figure 与宇树 H1](./humanoid-platforms-atlas-optimus)

#### 第 23 章 评测基准
- [x] [操作基准：RLBench、LIBERO 与 CALVIN](./manipulation-benchmarks)
- [x] [导航与交互基准：ALFRED、Habitat Challenge 与 BEHAVIOR](./navigation-benchmarks)
- [x] [真实世界评测协议与泛化性测试](./real-world-eval)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
