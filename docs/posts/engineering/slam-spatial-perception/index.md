---
pageClass: plain-doc
---

# SLAM 与空间感知（定位建图/多传感器融合）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Barfoot, "State Estimation for Robotics" (2017)
- Cadena et al., "Past, Present, and Future of SLAM" (IEEE T-RO, 2016)
- 高翔等, 《视觉 SLAM 十四讲》(2019)

## 主题规划

<ProgressGrid cat="engineering/slam-spatial-perception" />

### 第1篇

- [x] [状态估计问题（贝叶斯滤波→因子图的统一视角）](./state-estimation-bayesian-filter-factor-graph)
- [x] [前端里程计（特征法 ORB/直接法 LK、光流与匹配）](./frontend-odometry-feature-and-direct)
- [x] [视觉惯性融合 VIO（IMU 预积分、紧耦合优化）](./visual-inertial-odometry-vio)
- [x] [后端优化（Bundle Adjustment、位姿图优化、g2o/Ceres）](./backend-optimization-bundle-adjustment)
- [x] [回环检测（词袋模型、深度学习描述子）](./loop-closure-detection-bag-of-words)
- [x] [激光 SLAM（LOAM 谱系、点云配准 ICP/NDT）](./lidar-slam-loam-icp-ndt)
- [x] [地图表示（占据栅格/八叉树/符号距离场 TSDF）](./map-representation-occupancy-octree-tsdf)
- [x] [语义 SLAM（动态物体剔除、场景理解辅助定位）](./semantic-slam-dynamic-object)

### 第2篇

- [x] [神经 SLAM（NeRF/GS 建图、可微渲染的回环）](./neural-slam-nerf-gaussian-splatting)
- [x] [多机协同（分布式 SLAM、地图融合）](./multi-agent-collaborative-slam)
- [x] [鲁棒性与退化（隧道/走廊退化、多模态冗余设计）](./robustness-degeneracy-multimodal)
- [x] [工程落地（自动驾驶 HD Map、机器人导航栈、AR 锚定）](./slam-engineering-deployment)
