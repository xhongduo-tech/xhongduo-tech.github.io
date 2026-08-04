---
pageClass: plain-doc
---

# Embodied AI

Embodied AI studies how agents perceive, decide, and act by interacting with the environment through a physical body, a frontier bridging artificial intelligence and robotics. This topic plan covers a complete body of knowledge spanning robotics fundamentals, perception and SLAM, motion planning and control, imitation learning, robot reinforcement learning, vision-language-action models (VLA), world models, and embodied foundation models.

## Topic Plan

<ProgressGrid cat="advanced/embodied-ai" />


### Part 1 Robotics Fundamentals (benchmarked against Craig's *Introduction to Robotics* and Lynch & Park's *Modern Robotics*)

#### Chapter 1 Spatial Descriptions and Coordinate Transformations
- [ ] Position, orientation, and coordinate frames: properties and operations of rotation matrices
- [ ] Homogeneous transformation matrices and composite transformations
- [ ] Euler angles, equivalent axis-angle, and quaternion representations
- [ ] Lie groups SO(3)/SE(3), Lie algebras, and the exponential map: twists and screw motion

#### Chapter 2 Forward and Inverse Kinematics
- [ ] Link parameters and the D-H convention
- [ ] Forward kinematics: chained homogeneous transformations and the product of exponentials (POE) formula
- [ ] Analytical solutions for inverse kinematics: the Pieper criterion and common six-axis configurations
- [ ] Numerical solutions for inverse kinematics: Newton iteration, damped least squares, and null-space motion of redundant manipulators

#### Chapter 3 Jacobians and Statics
- [ ] Velocity propagation, the geometric Jacobian, and the analytic Jacobian
- [ ] Spatial velocities and the spatial Jacobian (POE-based)
- [ ] Singularity analysis and the manipulability ellipsoid
- [ ] Force Jacobian and static equilibrium: force-velocity duality

#### Chapter 4 Rigid-Body Dynamics
- [ ] Recursive Newton-Euler algorithm (RNEA)
- [ ] Lagrangian dynamics and the structural properties of the equations of motion: mass matrix, Coriolis, and gravity terms
- [ ] Floating-base dynamics and centroidal dynamics

### Part 2 Sensors and Perception

#### Chapter 5 Robot Sensors
- [ ] Camera models and calibration: the pinhole model, distortion, and Zhang Zhengyou's calibration method
- [ ] Depth cameras: structured light, ToF, and stereo vision
- [ ] LiDAR: ranging principles, point-cloud representation, and multi-line scanning
- [ ] Tactile sensing and force/torque sensors: from piezoresistive to vision-based tactile sensing (GelSight)

#### Chapter 6 Perception and State Estimation Basics
- [ ] Point-cloud processing: filtering, ICP registration, and normal estimation
- [ ] Kalman filter, extended Kalman filter, and particle filter
- [ ] 2D/3D object detection and 6D pose estimation

### Part 3 SLAM (benchmarked against Thrun's *Probabilistic Robotics* and Gao Xiang's *Visual SLAM: Fourteen Lectures*)

#### Chapter 7 SLAM Fundamentals
- [ ] Modeling the SLAM problem: state estimation and graph-optimization perspectives
- [ ] EKF-SLAM and sparsity analysis
- [ ] Graph-based SLAM: pose graphs, nonlinear least squares, and g2o/GTSAM practice
- [ ] Loop closure: bag-of-words models and descriptor matching

#### Chapter 8 Visual and LiDAR SLAM Systems
- [ ] Feature-based visual odometry: from ORB-SLAM to ORB-SLAM3
- [ ] LiDAR SLAM: LOAM, LeGO-LOAM, and FAST-LIO
- [ ] Visual-inertial odometry (VIO): MSCKF and VINS-Mono
- [ ] Neural implicit SLAM and 3D Gaussian Splatting mapping

### Part 4 Motion Planning (benchmarked against LaValle's *Planning Algorithms*)

#### Chapter 9 Configuration Space and Sampling-Based Planning
- [ ] Configuration space (C-Space) and obstacles
- [ ] Probabilistic roadmaps (PRM) and rapidly exploring random trees (RRT, RRT-Connect)
- [ ] Asymptotically optimal planning: RRT* and Informed RRT*
- [ ] Collision detection and kinematic constraints in manipulator motion planning

#### Chapter 10 Trajectory Optimization
- [ ] Polynomial trajectory interpolation and time parameterization
- [ ] Trajectory optimization problem modeling: CHOMP, STOMP, and TrajOpt
- [ ] Differential flatness and minimum-snap trajectories

### Part 5 Control

#### Chapter 11 Classical Control and Force Control
- [ ] Joint-space PID control and feedforward compensation
- [ ] Computed-torque control and inverse-dynamics control
- [ ] Impedance control and admittance control
- [ ] Hybrid force/position control and operational space control

#### Chapter 12 Model Predictive Control and Whole-Body Control
- [ ] Linear model predictive control (MPC): principles and real-time iteration
- [ ] Walking control based on the inverted pendulum and ZMP
- [ ] Centroidal MPC and quadruped locomotion control
- [ ] Whole-body control: task prioritization and QP solving

### Part 6 Imitation Learning

#### Chapter 13 Imitation Learning Methods
- [ ] Behavior cloning: problem modeling and covariate shift
- [ ] DAgger: dataset aggregation and interactive imitation learning
- [ ] Diffusion Policy
- [ ] ACT (Action Chunking with Transformers) and action chunking
- [ ] Teleoperation data-collection systems: ALOHA and Mobile ALOHA

### Part 7 Robot Reinforcement Learning and Sim2Real

#### Chapter 14 Robot Reinforcement Learning
- [ ] Problem modeling for robot RL: observations, action spaces, and reward design
- [ ] Deep RL algorithms applied to robots: PPO, SAC, and TD3
- [ ] Offline reinforcement learning: CQL, IQL, and Decision Transformer
- [ ] Curriculum learning and hierarchical reinforcement learning

#### Chapter 15 Sim2Real Transfer
- [ ] Domain randomization
- [ ] System identification and dynamic-parameter calibration
- [ ] Domain adaptation and adversarial transfer
- [ ] Sim2Real practice for quadruped locomotion control: from ANYmal to Unitree

### Part 8 Manipulation and Mobility

#### Chapter 16 Dexterous Manipulation
- [ ] Grasping fundamentals: force closure, grasp-quality metrics, and GraspNet-style grasp generation
- [ ] Modeling and control of multi-fingered dexterous hands: Shadow Hand and Allegro Hand
- [ ] RL-based dexterous manipulation: the OpenAI Rubik's cube hand case study
- [ ] Tactile feedback and contact-rich manipulation policies

#### Chapter 17 Mobile Manipulation
- [ ] Coordinated control of mobile bases and manipulators: whole-body motion planning and navigation-manipulation coupling
- [ ] Visual servoing: IBVS and PBVS
- [ ] Open-vocabulary object search and semantic navigation

### Part 9 Vision-Language-Action Models (VLA) and Embodied Foundation Models

#### Chapter 18 Vision-Language-Action Models
- [ ] RT-1: a robotic transformer on large-scale real-world robot data
- [ ] RT-2: transferring vision-language models to action
- [ ] OpenVLA: architecture and fine-tuning of an open-source VLA model
- [ ] Octo: generalist robot policies and cross-robot data mixing
- [ ] π0: a VLA architecture based on flow matching
- [ ] Robot datasets: Open X-Embodiment and DROID

#### Chapter 19 World Models and Video Prediction
- [ ] World model fundamentals: from Dreamer to generative driving world models
- [ ] Robot policy learning from video prediction: UniPi and follow-up work
- [ ] Combining interactive world models with model-predictive planning

#### Chapter 20 Embodied Foundation Models and Hierarchical Architectures
- [ ] Hierarchical architectures in embodied AI: high-level planning and low-level control
- [ ] SayCan: grounding language-model affordances
- [ ] Code as Policies: generating robot policies with code
- [ ] VoxPoser and task planning with 3D value maps
- [ ] Embodied multimodal foundation models: from PaLM-E to Gemini Robotics

### Part 10 Simulation Platforms, Humanoid Robots, and Benchmarks

#### Chapter 21 Simulation Platforms
- [ ] MuJoCo: contact dynamics modeling and the Python API in practice
- [ ] Isaac Gym and Isaac Sim: GPU-parallel simulation and large-scale RL training
- [ ] Habitat and indoor embodied navigation simulation
- [ ] Manipulation learning simulation benchmarks: ManiSkill and RoboSuite

#### Chapter 22 Humanoid Robots
- [ ] Hardware configurations and actuation technologies for humanoid robots
- [ ] Bipedal walking control: from ZMP to learning-based whole-body control
- [ ] Analysis of representative platforms: Atlas, Optimus, Figure, and Unitree H1

#### Chapter 23 Evaluation Benchmarks
- [ ] Manipulation benchmarks: RLBench, LIBERO, and CALVIN
- [ ] Navigation and interaction benchmarks: ALFRED, Habitat Challenge, and BEHAVIOR
- [ ] Real-world evaluation protocols and generalization testing

> After writing is done: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [title](./xxx)`.
