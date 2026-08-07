---
pageClass: plain-doc
---

# 计算机视觉

对标《计算机视觉：算法与应用》（Szeliski）与 CS231n 课程体系，从成像几何与底层图像处理一路写到深度学习方法与视觉基础模型，覆盖经典 CV 与现代 CV 的完整脉络。

## 主题规划

<ProgressGrid cat="advanced/computer-vision" />


### 图像形成与成像几何

- [x] [光的物理性质与光谱：可见光、色度学基础](./light-and-color)
- [x] [针孔相机模型与透视投影](./pinhole-camera-model)
- [x] [镜头与景深：光圈、焦距、弥散圆](./lens-and-depth-of-field)
- [x] [数字图像的采样与量化：奈奎斯特采样定理与混叠](./sampling-and-quantization)
- [x] [颜色空间：RGB、HSV、Lab 与 Bayer 阵列](./color-spaces)
- [x] [2D 与 3D 几何变换：平移、旋转、相似、仿射与投影变换](./geometric-transformations)
- [x] [齐次坐标与变换矩阵的组合](./homogeneous-coordinates)

### 图像处理基础

- [x] [点运算：灰度变换、直方图均衡化与直方图匹配](./point-operations)
- [x] [线性滤波：卷积与互相关、可分离滤波器](./linear-filtering)
- [x] [平滑滤波：盒式滤波、高斯滤波与边界处理](./smoothing-filters)
- [x] [非线性滤波：中值滤波、双边滤波与导向滤波](./nonlinear-filters)
- [x] [图像金字塔：高斯金字塔与拉普拉斯金字塔](./image-pyramids)
- [x] [傅里叶变换与频域滤波](./frequency-domain-filtering)
- [x] [边缘检测：Sobel、Prewitt、LoG 与 Canny 算子](./edge-detection)
- [x] [角点检测：Harris 角点与 Shi-Tomasi 角点](./corner-detection-harris)
- [x] [图像形态学：腐蚀、膨胀、开运算与闭运算](./morphological-operations)
- [x] [图像插值与几何变形：最近邻、双线性、双三次](./image-interpolation-warping)

### 特征描述与匹配

- [x] [尺度空间理论与 DoG 关键点检测](./scale-space-dog)
- [x] [SIFT：方向分配与 128 维描述子的构建](./sift)
- [x] [SURF：积分图像与 Hessian 快速近似](./surf)
- [x] [ORB：旋转 BRIEF 与实时特征提取](./orb)
- [x] [特征匹配策略：最近邻、比值测试与交叉验证](./feature-matching-strategies)
- [x] [基于 RANSAC 的误匹配剔除](./ransac-outlier-rejection)
- [x] [图像拼接：单应性估计与多波段融合](./image-stitching)

### 相机模型与标定

- [x] [相机内参：焦距、主点、倾斜与径向/切向畸变](./camera-intrinsics)
- [x] [相机外参与世界-相机坐标变换](./camera-extrinsics)
- [x] [张正友标定法：单应性约束与内参求解](./zhang-camera-calibration)
- [x] [镜头畸变校正与去畸变实现](./lens-distortion-correction)
- [x] [对极几何：本质矩阵与基础矩阵](./epipolar-geometry)
- [x] [八点法与五点法求解基础矩阵](./eight-point-five-point)
- [x] [三角测量：由双视图恢复三维点](./triangulation)

### 双目立体视觉

- [x] [双目成像原理与视差-深度关系](./stereo-imaging-disparity)
- [x] [极线校正（Rectification）与行对准](./rectification)
- [x] [局部立体匹配：块匹配与代价聚合](./local-stereo-matching)
- [x] [全局立体匹配：图割与置信度传播](./global-stereo-matching)
- [x] [半全局匹配（SGM/SGBM）算法](./sgm-stereo)
- [x] [视差图后处理与深度图计算](./disparity-postprocessing)

### 运动与跟踪

- [x] [光流约束方程与孔径问题](./optical-flow-constraint-aperture)
- [x] [Lucas-Kanade 稀疏光流与金字塔实现](./lucas-kanade-optical-flow)
- [x] [Horn-Schunck 稠密光流](./horn-schunck-optical-flow)
- [x] [均值漂移（Mean Shift）与 CamShift 跟踪](./mean-shift-camshift-tracking)
- [x] [Kalman 滤波：预测-更新框架与运动建模](./kalman-filter-tracking)
- [x] [扩展 Kalman 滤波与粒子滤波](./ekf-particle-filter)
- [x] [多目标跟踪：SORT 与 Deep SORT](./sort-deepsort)

### 图像分类与卷积神经网络

- [x] [从图像分类问题谈起：数据驱动方法与 KNN、线性分类器](./image-classification-data-driven)
- [x] [卷积、池化与感受野：CNN 的基本构件](./cnn-building-blocks)
- [x] [LeNet 与 AlexNet：深度学习的开端](./lenet-alexnet)
- [x] [VGG 与 GoogLeNet：更深网络的探索](./vgg-googlenet)
- [x] [ResNet：残差连接与退化问题的解决](./resnet)
- [x] [EfficientNet 与轻量化网络：MobileNet、ShuffleNet](./efficientnet-mobilenet)
- [x] [迁移学习与数据增强实践](./transfer-learning-data-augmentation)

### 目标检测

- [x] [检测任务定义：边界框、IoU 与 mAP 指标](./detection-task-iou-map)
- [x] [R-CNN：候选区域与两阶段检测的开端](./rcnn)
- [x] [Fast R-CNN 与 Faster R-CNN：RoI Pooling 与 RPN](./fast-faster-rcnn)
- [x] [YOLO 系列：单阶段检测的演进（v1 到 v8+）](./yolo-series)
- [x] [SSD 与多尺度特征图检测](./ssd-multiscale)
- [x] [FPN：特征金字塔网络](./fpn)
- [x] [Anchor-Free 检测：FCOS 与 CenterNet](./anchor-free-fcos-centernet)
- [x] [DETR：基于 Transformer 的端到端检测](./detr)

### 语义与实例分割

- [x] [分割任务谱系：语义、实例与全景分割](./segmentation-task-spectrum)
- [x] [FCN：全卷积网络与跳跃连接](./fcn)
- [x] [U-Net：编码器-解码器结构与医学图像分割](./unet)
- [x] [空洞卷积与 DeepLab 系列：ASPP 模块](./deeplab-aspp)
- [x] [Mask R-CNN：RoIAlign 与实例分割](./mask-rcnn)
- [x] [全景分割与现代统一分割框架](./panoptic-segmentation)

### 人脸与人体分析

- [x] [人脸检测：从 Haar 级联到深度学习检测器](./face-detection)
- [x] [人脸对齐与关键点定位](./face-alignment)
- [x] [人脸识别：DeepFace、FaceNet 与度量学习](./face-recognition)
- [x] [人脸属性分析与活体检测](./face-attributes-liveness)
- [x] [人体姿态估计：自顶向下与自底向上范式](./pose-estimation-topdown-bottomup)
- [x] [OpenPose 与 HRNet：多人 2D 姿态估计](./openpose-hrnet)
- [x] [3D 人体重建：SMPL 参数化人体模型](./smpl-body-model)

### 三维视觉

- [x] [多视图几何与光束法平差（Bundle Adjustment）](./bundle-adjustment)
- [x] [运动恢复结构（SfM）与 COLMAP 实践](./sfm-colmap)
- [x] [多视图立体（MVS）稠密重建](./mvs-dense-reconstruction)
- [x] [点云表示与处理：滤波、配准（ICP）与特征](./point-cloud-processing-icp)
- [x] [深度学习的三维表示：体素、点云网络（PointNet/PointNet++）](./pointnet-voxel)
- [x] [NeRF：神经辐射场的原理与渲染](./nerf)
- [x] [NeRF 的加速与改进：Instant-NGP、Mip-NeRF](./nerf-acceleration)
- [x] [3D 高斯泼溅（3D Gaussian Splatting）：显式辐射场与实时渲染](./3d-gaussian-splatting)

### 视频理解

- [x] [视频表示：帧采样与时序建模基础](./video-representation-temporal-modeling)
- [x] [双流网络：空间流与光流时间流](./two-stream-networks)
- [x] [3D 卷积网络：C3D 与 I3D](./c3d-i3d)
- [x] [SlowFast 与视频 Transformer：TimeSformer、ViViT](./slowfast-video-transformer)
- [x] [视频目标检测与时空动作定位](./video-object-detection-action-localization)
- [x] [视频分割与目标跟踪的统一建模](./video-segmentation-tracking)

### 视觉 Transformer

- [x] [自注意力机制回顾：从 NLP 到视觉](./self-attention-vision)
- [x] [ViT：图像分块与纯 Transformer 分类](./vit)
- [x] [DeiT：数据高效的视觉 Transformer 与蒸馏](./deit)
- [x] [Swin Transformer：移位窗口与层级结构](./swin-transformer)
- [x] [MAE：掩码自编码器与视觉自监督预训练](./mae)

### 视觉基础模型

- [x] [CLIP：对比式视觉-语言预训练](./clip)
- [x] [CLIP 的零样本分类与下游应用](./clip-zero-shot)
- [x] [SAM（Segment Anything）：可提示分割与数据引擎](./sam)
- [x] [视觉基础模型的微调与提示（Prompting）范式](./vfm-finetuning-prompting)
- [x] [DINO/DINOv2：自监督视觉特征学习](./dino-dinov2)

### 图像生成

- [x] [生成模型概览：VAE 与 GAN 在视觉中的角色](./vae-gan-overview)
- [x] [扩散模型原理：前向加噪与反向去噪](./diffusion-principles)
- [x] [DDPM 与 DDIM：采样过程与加速](./ddpm-ddim)
- [x] [潜在扩散模型（LDM）与 Stable Diffusion 架构](./ldm-stable-diffusion)
- [x] [扩散模型在 CV 中的应用：修复、超分辨率与图像编辑](./diffusion-applications)

### 多模态视觉语言模型

- [x] [视觉-语言任务谱系：图文检索、视觉问答、图像描述](./vision-language-task-spectrum)
- [x] [视觉语言预训练：BLIP、BLIP-2 与 Q-Former](./blip-blip2)
- [x] [LLaVA 系列：视觉指令微调](./llava)
- [x] [多模态大模型的架构：视觉编码器与 LLM 的桥接](./multimodal-llm-architecture)
- [x] [多模态模型的评测与幻觉问题](./multimodal-evaluation-hallucination)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
