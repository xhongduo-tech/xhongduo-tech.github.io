# 具身智能

具身智能（Embodied AI）研究让智能体通过物理身体与环境交互来感知、决策与行动，是连接人工智能与机器人学的前沿方向。本篇主题规划覆盖从机器人学基础、感知与 SLAM、运动规划与控制，到模仿学习、机器人强化学习、视觉-语言-动作模型（VLA）、世界模型与具身大模型的完整知识体系。

## 主题规划

### 第一篇 机器人学基础（对标 Craig《机器人学导论》、Lynch & Park《Modern Robotics》）

#### 第 1 章 空间描述与坐标变换
- [ ] 位置、姿态与坐标系：旋转矩阵的性质与运算
- [ ] 齐次变换矩阵与复合变换
- [ ] 欧拉角、等效轴角与四元数表示
- [ ] 李群 SO(3)/SE(3)、李代数与指数映射：旋量（Twist）与螺旋运动

#### 第 2 章 正运动学与逆运动学
- [ ] 连杆参数与 D-H 约定
- [ ] 正运动学：连乘齐次变换与 POE（指数积）公式
- [ ] 逆运动学的解析解法：Pieper 准则与常见六轴构型
- [ ] 逆运动学的数值解法：牛顿迭代、阻尼最小二乘与冗余机械臂的零空间运动

#### 第 3 章 雅可比矩阵与静力学
- [ ] 速度传播与几何雅可比、解析雅可比
- [ ] 运动旋量与空间雅可比（基于 POE）
- [ ] 奇异性分析与可操作度椭球
- [ ] 力雅可比与静力平衡：力-速度对偶性

#### 第 4 章 刚体动力学
- [ ] 牛顿-欧拉递推动力学算法（RNEA）
- [ ] 拉格朗日动力学与运动方程的结构性质：质量矩阵、科氏力与重力项
- [ ] 浮动基座动力学与质心动力学（Centroidal Dynamics）

### 第二篇 传感器与感知

#### 第 5 章 机器人传感器
- [ ] 相机模型与标定：针孔模型、畸变与张正友标定法
- [ ] 深度相机：结构光、ToF 与双目立体视觉
- [ ] 激光雷达：测距原理、点云表示与多线扫描
- [ ] 触觉传感与力/力矩传感器：从压阻式到视触觉（GelSight）

#### 第 6 章 感知与状态估计基础
- [ ] 点云处理：滤波、ICP 配准与法向量估计
- [ ] 卡尔曼滤波、扩展卡尔曼滤波与粒子滤波
- [ ] 2D/3D 物体检测与 6D 位姿估计

### 第三篇 SLAM（对标 Thrun《Probabilistic Robotics》、高翔《视觉 SLAM 十四讲》）

#### 第 7 章 SLAM 基础
- [ ] SLAM 问题建模：状态估计与图优化视角
- [ ] EKF-SLAM 与稀疏性分析
- [ ] 基于图优化的 SLAM：位姿图、非线性最小二乘与 g2o/GTSAM 实践
- [ ] 回环检测：词袋模型与描述子匹配

#### 第 8 章 视觉与激光 SLAM 系统
- [ ] 特征法视觉里程计：从 ORB-SLAM 到 ORB-SLAM3
- [ ] 激光 SLAM：LOAM、LeGO-LOAM 与 FAST-LIO
- [ ] 视觉-惯性里程计（VIO）：MSCKF 与 VINS-Mono
- [ ] 神经隐式 SLAM 与 3D 高斯泼溅（3D Gaussian Splatting）建图

### 第四篇 运动规划（对标 LaValle《Planning Algorithms》）

#### 第 9 章 构型空间与基于采样的规划
- [ ] 构型空间（C-Space）与障碍物
- [ ] 概率路线图（PRM）与快速扩展随机树（RRT、RRT-Connect）
- [ ] 渐近最优规划：RRT* 与 Informed RRT*
- [ ] 机械臂运动规划中的碰撞检测与运动学约束

#### 第 10 章 轨迹优化
- [ ] 多项式轨迹插值与时间参数化
- [ ] 轨迹优化问题建模：CHOMP、STOMP 与 TrajOpt
- [ ] 微分平坦与 Minimum Snap 轨迹

### 第五篇 控制

#### 第 11 章 经典控制与力控
- [ ] 关节空间 PID 控制与前馈补偿
- [ ] 计算力矩控制与逆动力学控制
- [ ] 阻抗控制与导纳控制
- [ ] 力/位混合控制与操作空间控制（Operational Space Control）

#### 第 12 章 模型预测控制与全身控制
- [ ] 线性模型预测控制（MPC）原理与实时迭代
- [ ] 基于倒立摆与 ZMP 的行走控制
- [ ] 质心 MPC 与四足机器人运动控制
- [ ] 全身控制（Whole-Body Control）：任务优先级与 QP 求解

### 第六篇 模仿学习

#### 第 13 章 模仿学习方法
- [ ] 行为克隆（Behavior Cloning）：问题建模与协变量偏移
- [ ] DAgger：数据集聚合与交互式模仿学习
- [ ] 扩散策略（Diffusion Policy）
- [ ] ACT（Action Chunking with Transformers）与动作分块
- [ ] 遥操作数据采集系统：ALOHA 与 Mobile ALOHA

### 第七篇 机器人强化学习与 Sim2Real

#### 第 14 章 机器人强化学习
- [ ] 机器人 RL 的问题建模：观测、动作空间与奖励设计
- [ ] 深度强化学习算法在机器人上的应用：PPO、SAC 与 TD3
- [ ] 离线强化学习：CQL、IQL 与 Decision Transformer
- [ ] 课程学习（Curriculum Learning）与分层强化学习

#### 第 15 章 仿真到现实迁移（Sim2Real）
- [ ] 域随机化（Domain Randomization）
- [ ] 系统辨识与动力学参数校准
- [ ] 域自适应与对抗式迁移
- [ ] 四足机器人运动控制的 Sim2Real 实践：从 ANYmal 到 Unitree

### 第八篇 操作与移动

#### 第 16 章 灵巧操作（Dexterous Manipulation）
- [ ] 抓取基础：力封闭、抓取质量指标与 GraspNet 类抓取生成
- [ ] 多指灵巧手建模与控制：Shadow Hand 与 Allegro Hand
- [ ] 基于强化学习的灵巧操作：OpenAI 魔方手案例分析
- [ ] 触觉反馈与接触丰富的操作策略

#### 第 17 章 移动操作（Mobile Manipulation）
- [ ] 移动底盘与机械臂的协同控制：全身运动规划与导航-操作耦合
- [ ] 视觉伺服（Visual Servoing）：IBVS 与 PBVS
- [ ] 开放场景物体搜寻与语义导航

### 第九篇 视觉-语言-动作模型（VLA）与具身大模型

#### 第 18 章 视觉-语言-动作模型
- [ ] RT-1：大规模真实机器人数据上的机器人 Transformer
- [ ] RT-2：视觉-语言模型的动作化迁移
- [ ] OpenVLA：开源 VLA 模型的架构与微调
- [ ] Octo：通用机器人策略与多机器人数据混合
- [ ] π0：基于流匹配（Flow Matching）的 VLA 架构
- [ ] 机器人数据集：Open X-Embodiment 与 DROID

#### 第 19 章 世界模型与视频预测
- [ ] 世界模型基础：从 Dreamer 到生成式驾驶世界模型
- [ ] 基于视频预测的机器人策略学习：UniPi 与后续工作
- [ ] 交互式世界模型与模型预测规划的结合

#### 第 20 章 具身大模型与分层架构
- [ ] 具身智能中的分层架构：高层规划与低层控制
- [ ] SayCan：语言模型可供性（Affordance）落地
- [ ] Code as Policies：用代码生成机器人策略
- [ ] VoxPoser 与基于 3D 价值图的任务规划
- [ ] 具身多模态大模型：从 PaLM-E 到 Gemini Robotics

### 第十篇 仿真平台、人形机器人与评测

#### 第 21 章 仿真平台
- [ ] MuJoCo：接触动力学建模与 Python API 实践
- [ ] Isaac Gym 与 Isaac Sim：GPU 并行仿真与大规模 RL 训练
- [ ] Habitat 与室内具身导航仿真
- [ ] 操作学习仿真基准：ManiSkill 与 RoboSuite

#### 第 22 章 人形机器人（Humanoid Robots）
- [ ] 人形机器人的硬件构型与驱动技术
- [ ] 双足行走控制：从 ZMP 到基于学习的全身控制
- [ ] 代表性平台解析：Atlas、Optimus、Figure 与宇树 H1

#### 第 23 章 评测基准
- [ ] 操作基准：RLBench、LIBERO 与 CALVIN
- [ ] 导航与交互基准：ALFRED、Habitat Challenge 与 BEHAVIOR
- [ ] 真实世界评测协议与泛化性测试

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
