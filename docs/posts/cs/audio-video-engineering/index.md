---
pageClass: plain-doc
---

# 音视频编解码与流媒体工程（H.266/AV1/RTC）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Richardson, "The H.264 Advanced Video Compression Standard" (2010)
- Sze, Budagavi & Sullivan (eds.), "High Efficiency Video Coding (HEVC)" (2014)
- Wiegand & Sullivan, "The Picture Tel. Codec 到 AV1 的技术谱系" 综述体系

## 主题规划

<ProgressGrid cat="cs/audio-video-engineering" />

### 第1篇

- [x] [感知编码基础（人眼/人耳特性、冗余类型）](./perceptual-coding-basics)
- [x] [变换与量化（DCT/整数变换、率失真理论）](./transform-and-quantization)
- [x] [帧内/帧间预测（运动估计、块划分演进）](./intra-inter-prediction)
- [x] [视频编码标准史（MPEG-2→H.264→HEVC→H.266/VVC）](./video-coding-standard-history)
- [x] [开源编码器（x264/x265/SVT-AV1 的实现取舍）](./open-source-encoders)
- [x] [AV1 与开放生态（专利池规避、硬件解码普及）](./av1-open-ecosystem)
- [x] [音频编码（AAC/Opus、心理声学模型）](./audio-coding)
- [x] [封装与传输协议（MP4/fMP4、HLS/DASH 自适应码率）](./muxing-transport-protocols)

### 第2篇

- [x] [实时通信 RTC（WebRTC 架构、拥塞控制 GCC、弱网对抗）](./realtime-rtc-webrtc)
- [x] [直播系统（CDN 分发、秒开/低延迟 LL-HLS）](./live-streaming-cdn)
- [x] [画质增强（超分/插帧、AI 编码端到端方案）](./quality-enhancement-super-resolution)
- [x] [沉浸媒体（VR 视频/空间音频、点云压缩 V-PCC）](./immersive-media-vr-vpcc)