---
pageClass: plain-doc
---

# 自动驾驶

本篇覆盖自动驾驶技术全景：从感知、定位、预测、规划、控制的经典模块化栈，到端到端与大模型驱动的新范式，再到仿真、数据闭环与量产工程化。

## 主题规划

<ProgressGrid cat="advanced/autonomous-driving" />


### 自动驾驶概述

- [x] [自动驾驶的发展历程与产业格局](./history-and-industry)
- [x] [SAE 自动驾驶分级（L0–L5）详解](./sae-autonomy-levels)
- [x] [自动驾驶传感器套件：相机、LiDAR、毫米波雷达、超声波与 IMU](./sensor-suite)
- [x] [模块化架构 vs 端到端架构](./modular-vs-end-to-end)
- [x] [自动驾驶系统的安全-case 与 ODD（设计运行域）](./odd-and-safety-case)

### 感知

- [x] [基于相机的 2D 目标检测在自动驾驶中的应用](./camera-2d-object-detection)
- [x] [单目与双目 3D 目标检测](./mono-stereo-3d-detection)
- [x] [LiDAR 点云处理：地面分割与聚类](./lidar-point-cloud-processing)
- [x] [基于点云的 3D 目标检测（PointPillars、VoxelNet、CenterPoint）](./point-cloud-3d-detection)
- [x] [毫米波雷达感知与 4D 成像雷达](./mmwave-radar-perception)
- [x] [BEV 感知的由来与核心思想](./bev-perception-origins)
- [x] [BEVFormer：基于 Transformer 的多相机 BEV 感知](./bevformer)
- [x] [LSS 与 BEVDepth：显式深度估计的 BEV 方案](./lss-bevdepth)
- [x] [Occupancy 占用网络：从检测框到体素化场景表达](./occupancy-network)
- [x] [多传感器融合：前融合、特征级融合与后融合](./multi-sensor-fusion)
- [x] [时序融合与 StreamPETR 式流式感知](./streaming-perception)
- [x] [车道线与可行驶区域检测](./lane-drivable-area-detection)
- [x] [交通信号灯与交通标志识别](./traffic-light-sign-recognition)
- [x] [在线矢量地图构建（HDMapNet、MapTR）](./online-hd-map-construction)

### 高精地图与定位

- [x] [高精地图的要素、格式与制作流程](./hd-map-elements-formats)
- [x] [GNSS/RTK 定位原理与误差来源](./gnss-rtk-localization)
- [x] [惯性导航与组合导航（GNSS+IMU 紧耦合）](./inertial-navigation-gnss-imu)
- [x] [LiDAR SLAM：LOAM 系列与回环检测](./lidar-slam-loam-loop-closure)
- [x] [视觉 SLAM 与视觉惯性里程计（VIO）在定位中的应用](./visual-slam-vio-localization)
- [x] [基于高精地图的点云匹配定位（NDT、ICP）](./point-cloud-matching-ndt-icp)
- [x] [多传感器融合定位与定位完整性监测](./multi-sensor-fusion-localization-integrity)

### 预测

- [x] [轨迹预测问题建模与评价指标（ADE/FDE）](./trajectory-prediction-modeling-metrics)
- [x] [基于物理与意图的预测方法](./physics-intent-based-prediction)
- [x] [基于学习的轨迹预测（Social LSTM、Social GAN）](./learning-based-trajectory-prediction)
- [x] [场景编码与矢量化预测（VectorNet、LaneGCN）](./scene-encoding-vectorized-prediction)
- [x] [意图识别与多模态轨迹预测](./intent-recognition-multimodal-trajectory)
- [x] [基于 Transformer 的预测（MTR、Wayformer）](./transformer-based-trajectory-prediction)
- [x] [预测与规划的联合建模](./joint-prediction-planning)

### 规划

- [x] [规划问题分层：路由、行为、运动规划](./planning-hierarchy-routing-behavior-motion)
- [x] [行为规划：有限状态机与决策树](./behavior-planning-fsm-decision-tree)
- [x] [基于 MDP 与 POMDP 的决策](./decision-making-mdp-pomdp)
- [x] [行为规划的博弈论方法](./game-theoretic-behavior-planning)
- [x] [全局路径规划：Dijkstra、A* 与混合 A*](./global-path-planning-dijkstra-astar-hybrid)
- [x] [采样类规划：RRT 与 RRT*](./sampling-based-planning-rrt)
- [x] [Frenet 坐标系下的路径规划](./frenet-frame-path-planning)
- [x] [轨迹优化：二次规划与凸优化方法](./trajectory-optimization-qp-convex)
- [x] [EM Planner 式的路径-速度分解迭代优化](./em-planner-path-velocity-decomposition)
- [x] [时空联合规划](./spatial-temporal-joint-planning)
- [x] [基于学习的规划：模仿学习规划器](./learning-based-planning-imitation)
- [x] [基于学习的规划：强化学习与树搜索](./learning-based-planning-rl-tree-search)

### 控制

- [x] [车辆运动学与动力学模型](./vehicle-kinematics-dynamics-models)
- [x] [横向控制：纯跟踪与 Stanley 算法](./lateral-control-pure-pursuit-stanley)
- [x] [横向控制：LQR 与 MPC](./lateral-control-lqr-mpc)
- [x] [纵向控制：PID 与速度规划跟踪](./longitudinal-control-pid)
- [x] [横纵向耦合控制与执行器约束](./coupled-control-actuator-constraints)

### 端到端自动驾驶

- [x] [端到端自动驾驶的演进：从行为克隆到模块化端到端](./end-to-end-evolution-bc-to-modular)
- [x] [UniAD：规划导向的统一多任务端到端架构](./uniad)
- [x] [VAD：矢量化场景表达的端到端规划](./vad-vectorized-planning)
- [x] [基于扩散模型的端到端规划](./diffusion-based-end-to-end-planning)
- [x] [世界模型基础：学习环境动力学](./world-model-foundations)
- [x] [世界模型驱动的规划（GAIA-1、DriveDreamer 思路）](./world-model-driven-planning)
- [x] [端到端系统的可解释性与安全兜底](./end-to-end-interpretability-safety)

### 仿真与数据闭环

- [x] [自动驾驶仿真的意义与仿真可信度](./simulation-significance-fidelity)
- [x] [主流仿真器：CARLA、LGSVL 与 NVIDIA DRIVE Sim](./mainstream-simulators-carla-lgsvl-drive-sim)
- [x] [场景库建设与基于场景的分类（具体/逻辑/抽象场景）](./scenario-library-construction-classification)
- [x] [传感器仿真与神经渲染（NeRF/3DGS 重建仿真）](./sensor-simulation-neural-rendering)
- [x] [数据闭环：数据采集、筛选、标注与回流](./data-closed-loop)
- [x] [影子模式（Shadow Mode）与长尾场景挖掘](./shadow-mode-long-tail-mining)
- [x] [自动化标注与 4D 标注](./automated-labeling-4d-annotation)

### 车端计算与工程化

- [x] [车载计算平台演进：从 MCU 到中央计算](./vehicle-computing-platform-evolution)
- [x] [主流自动驾驶芯片对比（英伟达 Orin/Thor、地平线征程、特斯拉 HW）](./autonomous-driving-chips-comparison)
- [x] [车载中间件：ROS 2、AUTOSAR 与 DDS](./vehicle-middleware-ros2-autosar-dds)
- [x] [功能安全 ISO 26262 与预期功能安全 SOTIF](./functional-safety-iso26262-sotif)
- [x] [模型车端部署：量化、剪枝与 TensorRT 推理优化](./model-onboard-deployment-quantization)
- [x] [时间同步与传感器标定的工程实践](./time-sync-sensor-calibration)

### Robotaxi 与量产方案

- [x] [Robotaxi 商业模式与运营难点](./robotaxi-business-model-operations)
- [x] [Waymo 的技术路线解析](./waymo-technical-route)
- [x] [Tesla FSD：纯视觉路线与 HW4 方案分析](./tesla-fsd-vision-hw4)
- [x] [华为 ADS 方案分析](./huawei-ads)
- [x] [小鹏 XNGP 与无图城市领航方案分析](./xpeng-xngp-mapless-navigation)
- [x] [高精地图派 vs 无图/轻图派之争](./hd-map-vs-mapless)
- [x] [L2++ 城市 NOA 量产落地的工程挑战](./l2pp-city-noa-engineering-challenges)

### VLA 与自动驾驶大模型

- [x] [视觉-语言模型（VLM）在自动驾驶中的应用](./vlm-in-autonomous-driving)
- [x] [VLA（视觉-语言-动作）模型范式解析](./vla-paradigm)
- [x] [端到端大模型与「两段式」端到端方案](./end-to-end-llm-two-stage)
- [x] [大模型的场景理解与可解释决策](./llm-scene-understanding-interpretable)
- [x] [自动驾驶基础模型：数据规模、训练范式与 Scaling Law](./autonomous-driving-foundation-models)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
