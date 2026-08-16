---
title: 封装与传输协议（MP4/fMP4、HLS/DASH 自适应码率）
date: 2026-08-07
---

# 封装与传输协议（MP4/fMP4、HLS/DASH 自适应码率）

<div class="epigraph">
<p>编码负责让视频变小，容器负责让世界看懂它。</p>
<footer>—— 佚名，流媒体工程界流传的说法</footer>
</div>

<div class="article-byline">
<p>第三级 · 音视频编解码与流媒体工程 ｜ Richardson《The H.264…》第 8 章（传输与封装） ｜ ISO/IEC 14496-12（BMFF）、23009-1（DASH） ｜ 2026-08-07</p>
</div>

## 为什么从封装与传输协议开始

编码器产出的是「裸码流」，而浏览器、播放器、CDN 需要的是一个**有序、带时间戳、可随机访问、可切片的文件**。**封装（muxing）**就是给码流穿上容器外衣，**传输协议**决定这些外衣如何在网络上分发。H.264 码流 + MP4 容器 + HLS/DASH 切片，这一组合撑起了今天的 VOD 与直播。没有容器与协议，再好的编码也是孤岛。

## 1 MP4：一切容器的现代母语

**MP4（ISO Base Media File Format，ISO/IEC 14496-12）** 用「盒子（box）」递归组织数据，核心三件套：

- **`ftyp`**：文件类型；
- **`moov`**：元数据——轨道（track）、采样表（sample table）、时间戳、编码参数；
- **`mdat`**：媒体数据本体（已压缩的音视频样本）。

传统 MP4 的 `moov` 在文件头部，**必须下载完 moov 才能解码**——对网络流不友好。流媒体的解法是**分片 MP4（fMP4）**：把时间轴切成**片段（fragment）**，每个片段由 `moof`（片段元数据）+ `mdat`（片段数据）构成。<span class="marginnote">fMP4 之所以是 HLS/DASH 的公共底层，是因为它能**边下载边解码**，且片段与 MPEG-2 TS 相比更轻、更易与 HTTP 缓存协同。</span>

## 2 HLS 与 DASH：HTTP 之上的自适应码率

**自适应码率（ABR）** 的核心思想：同一内容编码成多条**码率阶梯（ladder）**，客户端按带宽实时切换。两条主流协议：

- **HLS（HTTP Live Streaming）**：Apple 2017 标准化（RFC 8216）。媒体切成 TS 或 fMP4 分片，`m3u8` 播放列表描述所有分片与各档码率；客户端读列表、选档、逐片拉取。
- **MPEG-DASH（ISO/IEC 23009-1）**：国际化版本，用 **MPD（Media Presentation Description）** 描述分片；与 HLS 语法不同但理念一致。

两者的现代形态统一到 **CMAF（Common Media Application Format）**——同一份 fMP4 分片可同时供 HLS 与 DASH 使用，省去重复封装。

## 3 ABR 决策：客户端如何选码率

客户端不知未来带宽，只能观察**下载吞吐量**与**播放缓冲深度**，做出切换决策。经典模型是 **BOLA（buffer-based）** 与**吞吐量启发式**的结合：

- **吞吐量法**：用最近几个分片的下载速率平滑估计带宽，选不超带宽的最高档；
- **缓冲法**：缓冲低于阈值就降档避险，高于阈值才敢升档。

切换有代价——画质抖动（quality oscillation）会显著恶化主观体验，因此工业实现都会加**滞后（hysteresis）** 与**冷却期**，避免在相邻档位间来回横跳。<span class="marginnote">「秒开」的秘密一半在这里：播放器预取 playlist 与首个分片，同时用**渐进式渲染**让已下载部分先播放。HLS/DASH 的「秒开」体验 = 分片时长 + 预取策略 + 渲染引擎三方配合。</span>

## 4 公式解析：视频码率的经验估算

规划码率阶梯时，工程上用一个经验公式把画质目标翻译成码率：

$$R \approx W \times H \times \text{fps} \times \text{bpp}$$

- **$R$**：视频码率（bps）。
- **$W \times H$**：分辨率像素数。
- **$\text{fps}$**：帧率。
- **$\text{bpp}$**：每像素每帧比特数（bits per pixel），依内容复杂度取 0.05–0.15。

三步拆解：

- **第一步，读出量纲**：像素总数 × 帧率 = 每秒像素数，再乘 bpp 得每秒比特。
- **第二步，理解 bpp 的语义**：bpp 是「内容越复杂越高」的经验系数——实景高纹理 0.1–0.15，动画可压到 0.05。
- **第三步，落回阶梯**：1080p30 用 bpp=0.1 估算 $R \approx 1920\times1080\times30\times0.1 \approx 6.2$ Mbps。码率阶梯就从 6 Mbps（高清）一直降到 250 kbps（低清），各档间隔约 1.5–2 倍。**一句话：分辨率与帧率决定「要多少比特」，内容复杂度决定「值多少比特」。**

## 5 核心对比表：主流传输/封装方案

| 方案 | 容器 | 传输 | 延迟 | 自适应 | 场景 |
| --- | --- | --- | --- | --- | --- |
| MP4 文件 | ftyp/moov/mdat | 整文件 HTTP | 高 | 无 | VOD 下载 |
| HLS | TS/fMP4 | 分片 HTTP(S) | 6–30 s（LL-HLS 至 ~1 s） | 有 | Apple 系、直播 |
| DASH | fMP4 | 分片 HTTP(S) | 同上 | 有 | Android、通用 |
| CMAF | fMP4 | 分片 HTTP(S) | 同 LL | 有 | HLS+DASH 通用 |

## 6 小结

- **MP4/fMP4** 用 box 组织码流，fMP4 的分片设计让它成为流媒体通用底层。
- **HLS 与 DASH** 用播放列表/MPD + 分片实现**自适应码率**。
- **ABR 决策** = 吞吐量估计 + 缓冲深度 + 滞后防抖，决定用户画质与卡顿体验。
- 码率阶梯用 $R \approx W \times H \times \text{fps} \times \text{bpp}$ 估算，各档约 1.5–2 倍间隔。
- 封装与传输是编码器与播放器的「协议之桥」——至此，从感知到封装的 VOD 链路已闭环。

在下一节，我们把延迟从秒级拉到毫秒级：**实时通信 RTC**——WebRTC 架构、GCC 拥塞控制与弱网对抗。
