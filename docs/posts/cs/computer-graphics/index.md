---
pageClass: plain-doc
---

# 计算机图形学

学完计算机图形学 = 写完以下全部博文。选题体系对标 GAMES101（闫令琪）与《Fundamentals of Computer Graphics》(虎书) 的章节结构，覆盖从数学基础、光栅化、几何、光线追踪到动画模拟与神经渲染的完整内容。

## 主题规划

<ProgressGrid cat="cs/computer-graphics" />


### 第一篇 图形学概述

- [x] [什么是计算机图形学：研究内容、与计算机视觉/图像处理的关系](./what-is-computer-graphics)
- [x] [图形学的应用版图：游戏、影视、可视化、CAD 与虚拟现实](./applications-of-computer-graphics)
- [x] [渲染管线总览：从三维场景到二维图像的整体流程](./rendering-pipeline-overview)
- [x] [颜色与图像基础：像素、位图与色彩空间初步](./color-and-image-basics)

### 第二篇 线性代数回顾

- [x] [向量：点积及其在光照中的应用](./vectors-dot-product-lighting)
- [x] [向量：叉积与左右手坐标系](./vectors-cross-product-handedness)
- [x] [向量：法向量与正交基](./vectors-normals-orthonormal-basis)
- [x] [矩阵：矩阵乘法与线性变换的表示](./matrix-multiplication-linear-transformations)
- [x] [矩阵：逆矩阵、转置与正交矩阵](./matrix-inverse-transpose-orthogonal)
- [x] [行列式与变换的朝向判定](./determinant-transformation-orientation)

### 第三篇 变换

- [x] [2D 基础变换：缩放、切变与旋转](./2d-basic-transformations-scale-shear-rotation)
- [x] [齐次坐标：用矩阵统一表示平移](./homogeneous-coordinates-translation)
- [x] [2D 复合变换与变换顺序](./2d-composite-transformations-order)
- [x] [3D 变换：绕任意轴旋转与罗德里格斯公式](./3d-rotation-arbitrary-axis-rodrigues)
- [x] [模型变换（Model Transformation）](./model-transformation)
- [x] [视图变换（View Transformation）：相机标架](./view-transformation-camera-frame)
- [x] [正交投影变换](./orthographic-projection)
- [x] [透视投影变换：视锥体挤压到标准立方体](./perspective-projection)

### 第四篇 光栅化

- [x] [屏幕空间与视口变换](./screen-space-viewport-transformation)
- [x] [三角形：图形学中的基本图元](./triangles-basic-primitive)
- [x] [采样与走样（Aliasing）：频域分析](./sampling-aliasing-frequency)
- [x] [三角形遍历：包围盒与叉积判定法](./triangle-rasterization-bounding-box-cross-product)
- [x] [反走样：先滤波后采样](./antialiasing-filter-then-sample)
- [x] [MSAA 及其他实用反走样方案](./msaa-and-practical-antialiasing)
- [x] [可见性与深度缓冲（Z-Buffer）算法](./z-buffer-visibility)

### 第五篇 着色

- [x] [着色与明暗：着色频率（Flat / Gouraud / Phong）](./shading-frequency-flat-gouraud-phong)
- [x] [Blinn-Phong 反射模型：漫反射项](./blinn-phong-diffuse)
- [x] [Blinn-Phong 反射模型：高光项与环境项](./blinn-phong-specular-ambient)
- [x] [图形管线（Graphics Pipeline）：从顶点着色到片元着色](./graphics-pipeline)
- [x] [纹理映射：重心坐标插值](./texture-mapping-barycentric-interpolation)
- [x] [纹理采样问题：走样与 Mipmap](./texture-sampling-aliasing-mipmap)
- [x] [各向异性过滤与 EWA 过滤](./anisotropic-filtering-ewa)
- [x] [纹理的应用：环境光贴图与球面/立方体贴图](./texture-applications-environment-mapping-cubemap)
- [x] [凹凸贴图（Bump Mapping）](./bump-mapping)
- [x] [法线贴图（Normal Mapping）](./normal-mapping)
- [x] [位移贴图（Displacement Mapping）与三维程序纹理](./displacement-mapping-procedural-texture)

### 第六篇 几何表示

- [x] [几何表示方法总览：隐式与显式](./geometry-representation-overview-implicit-explicit)
- [x] [隐式表示：代数曲面与 CSG](./implicit-algebraic-surfaces-csg)
- [x] [隐式表示：距离函数与水平集](./implicit-distance-functions-level-sets)
- [x] [隐式表示：分形与自相似](./implicit-fractals-self-similarity)
- [x] [显式表示：点云与多边形网格](./explicit-point-clouds-polygon-meshes)
- [x] [贝塞尔曲线：de Casteljau 算法](./bezier-curves-de-casteljau)
- [x] [贝塞尔曲线的性质与分段构造](./bezier-curves-properties-piecewise)
- [x] [B 样条曲线与 NURBS](./bspline-curves-nurbs)
- [x] [贝塞尔曲面](./bezier-surfaces)
- [x] [网格处理：细分、简化与正则化总览](./mesh-processing-overview)
- [x] [网格细分：Loop 细分](./loop-subdivision)
- [x] [网格细分：Catmull-Clark 细分](./catmull-clark-subdivision)
- [x] [网格简化：边坍缩与二次误差度量（QEM）](./mesh-simplification-edge-collapse-qem)
- [x] [网格正则化（Mesh Regularization）](./mesh-regularization)

### 第七篇 光线追踪

- [x] [为什么需要光线追踪：光栅化的局限](./why-ray-tracing-limits-of-rasterization)
- [x] [Whitted-Style 光线追踪：递归光线求交](./whitted-style-ray-tracing)
- [x] [光线生成：从像素到视线方程](./ray-generation-pixel-ray)
- [x] [光线与隐式曲面求交](./ray-implicit-surface-intersection)
- [x] [光线与三角形求交：Möller–Trumbore 算法](./ray-triangle-intersection-moeller-trumbore)
- [x] [加速结构：轴对齐包围盒（AABB）与均匀网格](./acceleration-structures-aabb-uniform-grid)
- [x] [空间划分：KD-Tree 与 Octree](./space-partitioning-kdtree-octree)
- [x] [物体划分：包围盒层次结构（BVH）](./bounding-volume-hierarchy-bvh)
- [x] [BVH 的构建与遍历：SAH 划分](./bvh-construction-sah)
- [x] [辐射度量学（一）：辐射通量、辐射强度与辐照度](./radiometry-part1-flux-intensity-irradiance)
- [x] [辐射度量学（二）：辐射亮度（Radiance）](./radiometry-part2-radiance)
- [x] [BRDF 与反射方程](./brdf-and-reflection-equation)
- [x] [渲染方程及其推导](./rendering-equation-derivation)
- [x] [蒙特卡洛积分：估计定积分](./monte-carlo-integration)
- [x] [路径追踪：从渲染方程到采样算法](./path-tracing-algorithm)
- [x] [路径追踪的递归终止：俄罗斯轮盘赌](./russian-roulette-termination)
- [x] [直接光照采样：对光源的重要性采样](./light-source-importance-sampling)
- [x] [采样理论：重要性采样与多重重要性采样](./importance-sampling-mis)

### 第八篇 材质与外观

- [x] [材质即 BRDF：漫反射、镜面与折射材质](./material-as-brdf)
- [x] [完美镜面反射与折射：斯涅尔定律与菲涅尔项](./snell-law-fresnel)
- [x] [微表面模型：Cook-Torrance BRDF](./microfacet-cook-torrance)
- [x] [微表面模型的各项：法线分布、几何遮蔽与 Fresnel](./microfacet-components-ndf-geometry-fresnel)
- [x] [各向异性材质与纤维外观](./anisotropic-materials-fibers)

### 第九篇 动画与模拟

- [x] [动画基础：关键帧与插值](./animation-keyframes-interpolation)
- [x] [骨骼动画：正向运动学与反向运动学](./skeletal-animation-fk-ik)
- [x] [蒙皮（Skinning）与混合变形（Blend Shape）](./skinning-blend-shapes)
- [x] [模拟基础：单粒子与显式欧拉积分](./simulation-euler-integration)
- [x] [数值积分的稳定性：半隐式欧拉、RK4 与隐式方法](./numerical-integration-stability)
- [x] [质点弹簧系统：布料模拟](./mass-spring-cloth-simulation)
- [x] [刚体模拟与碰撞检测](./rigid-body-simulation-collision-detection)
- [x] [流体模拟初步：基于网格（欧拉）方法](./fluid-simulation-eulerian)
- [x] [流体模拟初步：基于粒子（拉格朗日）方法与 SPH](./fluid-simulation-lagrangian-sph)

### 第十篇 实时渲染

- [x] [实时渲染管线与 GPU 架构概述](./realtime-rendering-pipeline-gpu-architecture)
- [x] [阴影映射（Shadow Mapping）：原理与走样](./shadow-mapping)
- [x] [阴影映射改进：PCF、CSM 与 VSM](./shadow-mapping-improvements-pcf-csm-vsm)
- [x] [环境光遮蔽：SSAO 与 HBAO](./ssao-hbao)
- [x] [延迟渲染（Deferred Rendering）与前向渲染的比较](./deferred-vs-forward-rendering)
- [x] [色调映射（Tone Mapping）与 HDR 管线](./tone-mapping-hdr)
- [x] [实时全局光照：RSM、LPV 与 VXGI 概览](./realtime-global-illumination-rsm-lpv-vxgi)
- [x] [基于图像的光照（IBL）与预滤波环境贴图](./ibl-prefiltered-environment-maps)

### 第十一篇 现代专题

- [x] [GPU 渲染管线深入：可编程着色器与 GLSL/HLSL](./programmable-shaders-glsl-hlsl)
- [x] [GPU 并行计算：CUDA/Compute Shader 与图形加速](./gpu-parallel-computing-cuda-compute-shader)
- [x] [可微渲染（Differentiable Rendering）：原理与框架](./differentiable-rendering)
- [x] [逆向渲染：从图像重建几何、材质与光照](./inverse-rendering)
- [x] [神经辐射场（NeRF）：体渲染与位置编码](./nerf-neural-radiance-fields)
- [x] [NeRF 的加速与改进：Instant-NGP、Mip-NeRF](./nerf-accelerations-instant-ngp-mip-nerf)
- [x] [3D 高斯泼溅（3D Gaussian Splatting）：实时新视角合成](./3d-gaussian-splatting)
- [x] [神经渲染与传统管线的融合：趋势与展望](./neural-rendering-fusion-trends)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
