---
pageClass: plain-doc
---

# CV · 目标检测与图像分割

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Richard Szeliski, "Computer Vision: Algorithms and Applications" (2nd, 2022)
- David A. Forsyth & Jean Ponce, "Computer Vision: A Modern Approach" (3rd, 2012)
- Jian Yang et al., "Deep Learning for Computer Vision" (2021)

## 主题规划

<ProgressGrid cat="advanced/detection-segmentation" />

### 第1篇

- [x] [滑动窗口与候选区域 (Szeliski §6)](./sliding-window-region-proposal)
- [x] [HOG 特征与 DPM (Forsyth & Ponce §16)](./hog-dpm)
- [x] [R-CNN 系列 (Girshick et al., R-CNN 2014; Szeliski §6)](./anchor-free-centernet-fcos)
- [x] [YOLO 单阶段检测 (Yang et al., DLCV §6)](./yolo-one-stage)
- [x] [FCN 语义分割 (Long et al., FCN 2015; Szeliski §6)](./bev-3d-detection)
- [x] [U-Net 编码解码结构 (Yang et al., DLCV §8)](./classical-segmentation-graphcut-slic)
- [x] [Mask R-CNN 实例分割 (He et al., Mask R-CNN 2017)](./detection-metrics-focal-loss)
- [x] [全景分割 Panoptic (Yang et al., DLCV §9)](./detr-transformer-detection)

### 第2篇

- [x] [无锚框检测器（CenterNet/FCOS） (Duan et al., CenterNet 2019; Tian et al., FCOS 2019)](./fcn-semantic-segmentation)
- [x] [DETR 与 Transformer 检测 (Carion et al., DETR 2020)](./hog-dpm)
- [x] [3D 检测与 BEV 感知 (Lang et al., PointPillars 2019; Huang et al., BEVFormer 2022)](./bev-3d-detection)
- [x] [检测评估指标与损失（mAP/IoU/Focal Loss） (Lin et al., Focal Loss 2017)](./mask-rcnn-instance-segmentation)
- [x] [传统图像分割基础（图割/超像素/均值漂移） (Boykov & Jolly 2001; Achanta et al., SLIC 2012; Comaniciu & Meer 2002)](./panoptic-segmentation)
