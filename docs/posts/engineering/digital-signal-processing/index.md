---
pageClass: plain-doc
---

# 数字信号处理

语音、音频、图像与通信处理共同上游的数学方法：DFT/FFT、z 变换、数字滤波器设计与小波分析。掌握这些底层工具，才能理解各应用分支的算法为何如此设计。

## 对标教材

- Oppenheim & Schafer, "Discrete-Time Signal Processing" (Pearson, 3rd ed.)
- 程佩青《数字信号处理教程》（清华大学出版社）
- Mallat, "A Wavelet Tour of Signal Processing" (Academic Press, 3rd ed.)

## 主题规划

<ProgressGrid cat="engineering/digital-signal-processing" />

### 第1篇

- [x] [离散时间信号与系统](./discrete-time-signals-and-systems)
- [x] [离散时间系统的频域分析](./frequency-domain-analysis-of-dt-systems)
- [x] [z 变换](./z-transform)
- [x] [连续时间信号的采样与重建](./sampling-and-reconstruction)
- [x] [LTI 系统的变换域分析](./transform-domain-analysis-of-lti-systems)

### 第2篇

- [x] [离散傅里叶变换 DFT](./discrete-fourier-transform)
- [x] [DFT 的圆周卷积与线性卷积](./dft-circular-and-linear-convolution)
- [x] [快速傅里叶变换 FFT](./fast-fourier-transform)
- [x] [利用 DFT 的频谱分析与窗函数](./spectrum-analysis-and-windowing)

### 第3篇

- [x] [离散时间系统的结构](./discrete-time-system-structures)
- [x] [IIR 数字滤波器设计](./iir-filter-design)
- [x] [FIR 数字滤波器设计](./fir-filter-design)
- [x] [有限字长效应](./finite-wordlength-effects)

### 第4篇

- [x] [参数信号建模](./parametric-signal-modeling)
- [x] [离散希尔伯特变换](./discrete-hilbert-transform)
- [x] [倒谱分析与同态解卷积](./cepstrum-and-homomorphic-deconvolution)
- [x] [稀疏表示导论](./sparse-representations-introduction)
- [x] [框架与离散基](./frames-and-discrete-bases)
- [x] [小波基](./wavelet-bases)
- [x] [小波包与局部余弦基](./wavelet-packets-and-local-cosine-bases)
- [x] [冗余字典稀疏表示与压缩感知](./redundant-dictionaries-and-compressed-sensing)

### 第5篇

- [ ] 离散时间信号与系统（序列、线性时不变系统与卷积）
- [ ] z 变换与系统分析（收敛域、传递函数与稳定性）
- [ ] 信号的采样与重构（奈奎斯特采样定理与混叠）
- [ ] 离散傅里叶变换（DFT/DFS 性质与频域采样）
- [ ] 快速傅里叶变换（基-2 FFT 算法与运算量优化）
- [ ] 数字滤波器结构（直接型、级联型与格型网络）
- [ ] IIR 滤波器设计（双线性变换、冲激不变法）
- [ ] FIR 滤波器设计（窗函数法、等波纹逼近）
- [ ] 有限字长效应（量化噪声与系数量化敏感度）
- [ ] 多速率信号处理（抽取、内插与滤波器组）
- [ ] 谱估计与应用（功率谱估计、自适应滤波与小波变换）
