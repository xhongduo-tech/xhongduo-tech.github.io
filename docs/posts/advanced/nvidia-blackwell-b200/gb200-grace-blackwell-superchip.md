---
title: GB200 Grace Blackwell 超级芯片
date: 2026-08-07
---

# GB200 Grace Blackwell 超级芯片

<div class="epigraph">
<p>把 CPU 和 GPU 揉进同一个封装，不是把它们放得更近，而是把「数据搬运」这件事从软件里删掉。</p>
<footer>—— 黄仁勋（Jensen Huang）谈 Grace Blackwell 设计理念</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA Blackwell/B200 ｜ NVIDIA Blackwell 白皮书 §4 ｜ 2026-08-07</p>
</div>

## 为什么从超级芯片开始

上一节我们看到 Blackwell 用「双裸片」突破了芯片级上限；这一节把镜头再拉远一格：**B200 几乎不会独自工作，它总是与一颗 Grace CPU 组成「超级芯片」一起交付**。GB200 超级芯片 = 1 颗 Grace CPU + 2 颗 B200 GPU。理解它的关键是「分工」与「黏合」：CPU 与 GPU 各擅长什么，用什么带宽把它们黏成一个可编程的整体。这与我们熟知的「GPU 服务器里插一块 PCIe 加速卡」是完全不同的设计哲学。

## 1 Grace：为 AI 训练重造的一颗 CPU

Grace 是 NVIDIA 自研的 **Arm 架构服务器 CPU**，72 个 Arm Neoverse V2 核心，面向数据中心。它不是为了取代 x86 的通用生态，而是围绕一个任务设计：**当好 GPU 的「数据管家」**。

在 LLM 训练与推理里，CPU 承担的工作远比「发指令」多：

- **数据装载与预处理**：训练数据要先解压、解码、切 batch、做增广，再喂给 GPU。这是一条流水线，CPU 慢则 GPU 饿。
- **图与嵌入（embedding）查找**：推荐系统、图神经网络里有海量的稀疏查表，天生适合 CPU 大内存。
- **MoE 路由决策**：混合专家模型的路由（router）每 token 都要算一次，通常放在 CPU 或 GPU 上的轻量算子，但它的输入统计常由 CPU 聚合。
- **检查点（checkpoint）与协调**：周期性落盘、初始化、故障恢复，都需要 CPU 的通用能力。

**Grace 的核心规格：480GB LPDDR5X 内存，约 1 TB/s 带宽。**<span class="marginnote">对比一下：普通 x86 服务器的 DDR5 内存带宽约 0.3–0.5 TB/s；Grace 用低功耗的 LPDDR5X 堆出 1 TB/s，正是为了「喂 GPU 喂得动」。容量与带宽都朝「数据管家」这个定位倾斜。</span>

## 2 互连拓扑：NVLink-C2C 把三种存储黏成一体

GB200 超级芯片内部有三类存储：Grace 的 LPDDR5X（480GB）、每颗 B200 的 HBM3e（192GB）。把三类存储黏成一个可编程整体的，是 **NVLink-C2C（Chip-to-Chip）** 互连。

在 GB200 内：

| 连接 | 带宽 | 用途 |
| --- | --- | --- |
| Grace CPU ↔ B200 GPU ① | NVLink-C2C，900 GB/s | CPU 读写 GPU 显存、GPU 读写 CPU 内存 |
| Grace CPU ↔ B200 GPU ② | NVLink-C2C，900 GB/s | 同上 |
| B200 GPU ① ↔ B200 GPU ② | NVLink 5，1.8 TB/s | 两张 GPU 之间高速交换 |
| CPU 内存总带宽 | LPDDR5X，~1 TB/s | 数据预处理、稀疏查表 |
| 每颗 GPU 显存带宽 | HBM3e，8 TB/s | 矩阵运算主体 |

两个技术点值得停下来：

**第一，NVLink-C2C 是「缓存一致」的**。CPU 与 GPU 看到的不是「需要手动拷贝的两块显存」，而是**同一份地址空间**。CPU 写一块数据，GPU 直接读，不需要 `cudaMemcpy` 这类的显式搬运。<span class="marginnote">对比传统 PCIe 加速卡：数据要先拷贝进 GPU 显存，算完再拷回，拷贝本身占掉大量延迟与带宽。NVLink-C2C 把「搬运」从程序员手里删掉——这正是 epigraph 里那句话的落地。</span>

**第二，两台 Grace 的 LPDDR5X 合计 960GB，加上 384GB HBM3e**（2×192），一个超级芯片就拥有了超过 1.3TB 的「可编程存储池」。对很多推理场景，「模型 + KV 缓存 + 工具数据」全部驻留在一个超级芯片内，跨节点通信因此大幅减少。<span class="marginnote">这一点与《大模型部署》里的「模型放置」决策直接相关：KV 缓存有多大、要占多少显存，决定了服务能同时支撑多少并发请求。</span>

## 3 公式解析：900 GB/s 意味着什么

NVLink-C2C 的 900 GB/s 不是抽象数字，把它与 PCIe 一比就有感觉。**这一代数据中心 GPU 之间的标准通路——PCIe Gen5 x16 的带宽是 128 GB/s（双向）**：

$$
\frac{B_{\text{NVLink-C2C}}}{B_{\text{PCIe 5.0 x16}}} = \frac{900\ \text{GB/s}}{128\ \text{GB/s}} \approx 7
$$