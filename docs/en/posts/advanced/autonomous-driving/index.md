---
pageClass: plain-doc
---

# Autonomous Driving

This post surveys the full landscape of autonomous driving technology: from the classic modular stack of perception, localization, prediction, planning and control, to new paradigms driven by end-to-end and large-model approaches, and on to simulation, data loops and production engineering.

## Topic Planning

<ProgressGrid cat="advanced/autonomous-driving" />


### Autonomous Driving Overview

- [ ] History and industry landscape of autonomous driving
- [ ] SAE autonomy levels (L0–L5) explained in detail
- [ ] Autonomous driving sensor suite: camera, LiDAR, millimeter-wave radar, ultrasonic and IMU
- [ ] Modular architecture vs end-to-end architecture
- [ ] Safety cases and ODD (Operational Design Domain) for autonomous driving systems

### Perception

- [ ] Camera-based 2D object detection in autonomous driving
- [ ] Monocular and stereo 3D object detection
- [ ] LiDAR point cloud processing: ground segmentation and clustering
- [ ] Point cloud-based 3D object detection (PointPillars, VoxelNet, CenterPoint)
- [ ] Millimeter-wave radar perception and 4D imaging radar
- [ ] The origins and core ideas of BEV perception
- [ ] BEVFormer: Transformer-based multi-camera BEV perception
- [ ] LSS and BEVDepth: BEV approaches with explicit depth estimation
- [ ] Occupancy networks: from bounding boxes to voxelized scene representations
- [ ] Multi-sensor fusion: early fusion, feature-level fusion and late fusion
- [ ] Temporal fusion and stream-based perception in the style of StreamPETR
- [ ] Lane line and drivable area detection
- [ ] Traffic light and traffic sign recognition
- [ ] Online vector map construction (HDMapNet, MapTR)

### High-Definition Maps and Localization

- [ ] Elements, formats and production pipeline of HD maps
- [ ] GNSS/RTK localization principles and error sources
- [ ] Inertial navigation and integrated navigation (tightly coupled GNSS+IMU)
- [ ] LiDAR SLAM: the LOAM family and loop closure detection
- [ ] Visual SLAM and visual-inertial odometry (VIO) in localization
- [ ] Point cloud matching localization against HD maps (NDT, ICP)
- [ ] Multi-sensor fusion localization and localization integrity monitoring

### Prediction

- [ ] Trajectory prediction problem formulation and evaluation metrics (ADE/FDE)
- [ ] Physics-based and intent-based prediction methods
- [ ] Learning-based trajectory prediction (Social LSTM, Social GAN)
- [ ] Scene encoding and vectorized prediction (VectorNet, LaneGCN)
- [ ] Intent recognition and multimodal trajectory prediction
- [ ] Transformer-based prediction (MTR, Wayformer)
- [ ] Joint modeling of prediction and planning

### Planning

- [ ] Hierarchical planning problems: routing, behavior and motion planning
- [ ] Behavior planning: finite state machines and decision trees
- [ ] Decision making based on MDP and POMDP
- [ ] Game-theoretic approaches to behavior planning
- [ ] Global path planning: Dijkstra, A* and Hybrid A*
- [ ] Sampling-based planning: RRT and RRT*
- [ ] Path planning in the Frenet frame
- [ ] Trajectory optimization: quadratic programming and convex optimization methods
- [ ] EM Planner-style iterative path–speed decomposition optimization
- [ ] Spatio-temporal joint planning
- [ ] Learning-based planning: imitation learning planners
- [ ] Learning-based planning: reinforcement learning and tree search

### Control

- [ ] Vehicle kinematics and dynamics models
- [ ] Lateral control: pure pursuit and the Stanley algorithm
- [ ] Lateral control: LQR and MPC
- [ ] Longitudinal control: PID and speed planning tracking
- [ ] Coupled lateral–longitudinal control and actuator constraints

### End-to-End Autonomous Driving

- [ ] Evolution of end-to-end autonomous driving: from behavior cloning to modular end-to-end
- [ ] UniAD: a planning-oriented unified multi-task end-to-end architecture
- [ ] VAD: end-to-end planning with vectorized scene representations
- [ ] Diffusion model-based end-to-end planning
- [ ] World model foundations: learning environment dynamics
- [ ] World model-driven planning (GAIA-1, DriveDreamer approaches)
- [ ] Interpretability and safety fallbacks in end-to-end systems

### Simulation and Data Loop

- [ ] The significance of autonomous driving simulation and simulation fidelity
- [ ] Mainstream simulators: CARLA, LGSVL and NVIDIA DRIVE Sim
- [ ] Scenario library construction and scenario-based classification (concrete/logical/abstract scenarios)
- [ ] Sensor simulation and neural rendering (NeRF/3DGS reconstruction-based simulation)
- [ ] Data loop: data collection, mining, labeling and feedback
- [ ] Shadow Mode and long-tail scenario mining
- [ ] Automated labeling and 4D labeling

### On-Vehicle Computing and Engineering

- [ ] Evolution of on-vehicle computing platforms: from MCU to centralized computing
- [ ] Comparison of mainstream autonomous driving chips (NVIDIA Orin/Thor, Horizon Journey, Tesla HW)
- [ ] On-vehicle middleware: ROS 2, AUTOSAR and DDS
- [ ] Functional safety ISO 26262 and Safety of the Intended Functionality (SOTIF)
- [ ] On-vehicle model deployment: quantization, pruning and TensorRT inference optimization
- [ ] Engineering practice of time synchronization and sensor calibration

### Robotaxi and Production Solutions

- [ ] Robotaxi business models and operational challenges
- [ ] Analysis of Waymo's technical approach
- [ ] Tesla FSD: vision-only approach and analysis of the HW4 platform
- [ ] Analysis of Huawei's ADS solution
- [ ] Analysis of XPeng XNGP and mapless urban navigation solutions
- [ ] The HD-map faction vs the mapless/light-map faction debate
- [ ] Engineering challenges of mass-producing L2++ urban NOA

### VLA and Autonomous Driving Foundation Models

- [ ] Applications of vision-language models (VLM) in autonomous driving
- [ ] Analysis of the VLA (vision-language-action) model paradigm
- [ ] End-to-end foundation models and "two-stage" end-to-end solutions
- [ ] Scene understanding and interpretable decision making with foundation models
- [ ] Autonomous driving foundation models: data scale, training paradigms and scaling laws

> After the writing is complete: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
