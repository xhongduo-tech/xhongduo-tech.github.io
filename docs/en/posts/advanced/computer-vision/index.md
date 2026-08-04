---
pageClass: plain-doc
---

# Computer Vision

Following the curriculum of *Computer Vision: Algorithms and Applications* (Szeliski) and the CS231n course, this series spans the full arc from imaging geometry and low-level image processing to deep learning methods and vision foundation models, covering the complete lineage of classical and modern CV.

## Topic Plan

<ProgressGrid cat="advanced/computer-vision" />

### Image Formation and Imaging Geometry

- [ ] Physical properties of light and the spectrum: visible light, basics of colorimetry
- [ ] Pinhole camera model and perspective projection
- [ ] Lenses and depth of field: aperture, focal length, circle of confusion
- [ ] Sampling and quantization of digital images: Nyquist sampling theorem and aliasing
- [ ] Color spaces: RGB, HSV, Lab and the Bayer array
- [ ] 2D and 3D geometric transformations: translation, rotation, similarity, affine and projective transformations
- [ ] Homogeneous coordinates and composition of transformation matrices

### Image Processing Fundamentals

- [ ] Point operations: intensity transformations, histogram equalization and histogram matching
- [ ] Linear filtering: convolution and cross-correlation, separable filters
- [ ] Smoothing filters: box filter, Gaussian filter and boundary handling
- [ ] Nonlinear filters: median filter, bilateral filter and guided filter
- [ ] Image pyramids: Gaussian pyramid and Laplacian pyramid
- [ ] Fourier transform and frequency-domain filtering
- [ ] Edge detection: Sobel, Prewitt, LoG and Canny operators
- [ ] Corner detection: Harris corners and Shi-Tomasi corners
- [ ] Image morphology: erosion, dilation, opening and closing
- [ ] Image interpolation and geometric warping: nearest neighbor, bilinear, bicubic

### Feature Description and Matching

- [ ] Scale-space theory and DoG keypoint detection
- [ ] SIFT: orientation assignment and construction of the 128-dimensional descriptor
- [ ] SURF: integral images and fast Hessian approximation
- [ ] ORB: rotation-aware BRIEF and real-time feature extraction
- [ ] Feature matching strategies: nearest neighbor, ratio test and cross-check
- [ ] RANSAC-based outlier rejection for mismatches
- [ ] Image stitching: homography estimation and multi-band blending

### Camera Models and Calibration

- [ ] Camera intrinsics: focal length, principal point, skew and radial/tangential distortion
- [ ] Camera extrinsics and world-to-camera coordinate transformations
- [ ] Zhang's calibration method: homography constraints and intrinsic estimation
- [ ] Lens distortion correction and undistortion implementation
- [ ] Epipolar geometry: essential matrix and fundamental matrix
- [ ] Eight-point and five-point algorithms for estimating the fundamental matrix
- [ ] Triangulation: recovering 3D points from two views

### Binocular Stereo Vision

- [ ] Binocular imaging principles and the disparity-depth relationship
- [ ] Rectification and row alignment
- [ ] Local stereo matching: block matching and cost aggregation
- [ ] Global stereo matching: graph cuts and belief propagation
- [ ] Semi-global matching (SGM/SGBM) algorithm
- [ ] Disparity map post-processing and depth map computation

### Motion and Tracking

- [ ] Optical flow constraint equation and the aperture problem
- [ ] Lucas-Kanade sparse optical flow and the pyramidal implementation
- [ ] Horn-Schunck dense optical flow
- [ ] Mean Shift and CamShift tracking
- [ ] Kalman filter: predict-update framework and motion modeling
- [ ] Extended Kalman filter and particle filter
- [ ] Multi-object tracking: SORT and Deep SORT

### Image Classification and Convolutional Neural Networks

- [ ] Starting from the image classification problem: data-driven approach, KNN and linear classifiers
- [ ] Convolution, pooling and receptive fields: the basic building blocks of CNNs
- [ ] LeNet and AlexNet: the dawn of deep learning
- [ ] VGG and GoogLeNet: exploring deeper networks
- [ ] ResNet: residual connections and solving the degradation problem
- [ ] EfficientNet and lightweight networks: MobileNet, ShuffleNet
- [ ] Transfer learning and data augmentation in practice

### Object Detection

- [ ] Defining the detection task: bounding boxes, IoU and the mAP metric
- [ ] R-CNN: region proposals and the beginning of two-stage detection
- [ ] Fast R-CNN and Faster R-CNN: RoI Pooling and RPN
- [ ] YOLO series: the evolution of single-stage detection (v1 to v8+)
- [ ] SSD and multi-scale feature map detection
- [ ] FPN: Feature Pyramid Network
- [ ] Anchor-free detection: FCOS and CenterNet
- [ ] DETR: Transformer-based end-to-end detection

### Semantic and Instance Segmentation

- [ ] The segmentation task spectrum: semantic, instance and panoptic segmentation
- [ ] FCN: fully convolutional networks and skip connections
- [ ] U-Net: encoder-decoder architecture and medical image segmentation
- [ ] Dilated convolutions and the DeepLab series: the ASPP module
- [ ] Mask R-CNN: RoIAlign and instance segmentation
- [ ] Panoptic segmentation and modern unified segmentation frameworks

### Face and Human Body Analysis

- [ ] Face detection: from Haar cascades to deep learning detectors
- [ ] Face alignment and facial landmark localization
- [ ] Face recognition: DeepFace, FaceNet and metric learning
- [ ] Face attribute analysis and liveness detection
- [ ] Human pose estimation: top-down and bottom-up paradigms
- [ ] OpenPose and HRNet: multi-person 2D pose estimation
- [ ] 3D human body reconstruction: the SMPL parametric body model

### 3D Vision

- [ ] Multi-view geometry and bundle adjustment
- [ ] Structure from Motion (SfM) and COLMAP in practice
- [ ] Multi-view stereo (MVS) dense reconstruction
- [ ] Point cloud representation and processing: filtering, registration (ICP) and features
- [ ] Deep learning 3D representations: voxels, point cloud networks (PointNet/PointNet++)
- [ ] NeRF: principles and rendering of neural radiance fields
- [ ] NeRF acceleration and improvements: Instant-NGP, Mip-NeRF
- [ ] 3D Gaussian Splatting: explicit radiance fields and real-time rendering

### Video Understanding

- [ ] Video representation: frame sampling and the basics of temporal modeling
- [ ] Two-stream networks: spatial stream and optical-flow temporal stream
- [ ] 3D convolutional networks: C3D and I3D
- [ ] SlowFast and video Transformers: TimeSformer, ViViT
- [ ] Video object detection and spatio-temporal action localization
- [ ] Unified modeling of video segmentation and object tracking

### Vision Transformers

- [ ] Review of self-attention: from NLP to vision
- [ ] ViT: image patching and pure Transformer classification
- [ ] DeiT: data-efficient image Transformers and distillation
- [ ] Swin Transformer: shifted windows and hierarchical structure
- [ ] MAE: masked autoencoders and visual self-supervised pretraining

### Vision Foundation Models

- [ ] CLIP: contrastive vision-language pretraining
- [ ] CLIP's zero-shot classification and downstream applications
- [ ] SAM (Segment Anything): promptable segmentation and the data engine
- [ ] Fine-tuning and prompting paradigms for vision foundation models
- [ ] DINO/DINOv2: self-supervised visual feature learning

### Image Generation

- [ ] Overview of generative models: the role of VAE and GAN in vision
- [ ] Diffusion model principles: forward noising and reverse denoising
- [ ] DDPM and DDIM: sampling process and acceleration
- [ ] Latent diffusion models (LDM) and the Stable Diffusion architecture
- [ ] Diffusion models in CV applications: inpainting, super-resolution and image editing

### Multimodal Vision-Language Models

- [ ] The vision-language task spectrum: image-text retrieval, visual question answering, image captioning
- [ ] Vision-language pretraining: BLIP, BLIP-2 and Q-Former
- [ ] The LLaVA series: visual instruction tuning
- [ ] Architectures of multimodal foundation models: bridging the vision encoder and LLM
- [ ] Evaluation of multimodal models and the hallucination problem

> When a post is finished: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
