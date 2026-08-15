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

逐项拆解：

- **$900\ \text{GB/s}$**：NVLink-C2C 在 GB200 内 CPU-GPU 的双向带宽。注意这是「双 CPU-GPU 链路里每一条」的带宽，两条合计 1.8 TB/s。
- **$128\ \text{GB/s}$**：PCIe Gen5 ×16 的双向带宽，是当前主流数据中心 GPU 卡的标准插槽带宽。
- **$\approx 7$**：7 倍差距——但更要紧的是「语义差距」：PCIe 卡是「两块内存 + 手动拷贝」，NVLink-C2C 是「同一份内存、硬件保证一致」。

**关键结论：NVLink-C2C 的 7 倍带宽只是表象，真正的质变是「缓存一致」——CPU 与 GPU 不再各自持有一份数据，而是共享同一个地址空间。** 对 LLM 训练，「数据不用拷来拷去」把每步里被拷贝吃掉的延迟与带宽全部还给了计算。<span class="marginnote">再对照一下 NVLink-C2C 与 NVLink 5：前者是 CPU-GPU 的「一致性互连」，强调共享内存；后者是 GPU-GPU 的「高速数据面」，强调带宽。GB200 里两者并存、各司其职——这也是「超级芯片」与「双 GPU 卡」的本质区别：<strong>它引入了 CPU，而 CPU 的参与靠一致性互连才不拖后腿。</strong></span>

## 4 超级芯片的软件形态：统一寻址下的「一台电脑」

硬件黏好了，软件必须跟上。GB200 超级芯片对程序员的呈现方式是 **CUDA 统一寻址（Unified Memory 的硬件加速版）**：

- 程序员写 `cudaMallocManaged` 或直接操作统一指针，CPU 与 GPU 共享数据，由硬件与驱动负责一致性；
- 传统的 `cudaMemcpy`（显式拷贝）在超级芯片内被大幅简化——很多场景不再需要；
- CUDA 的 `host`/`device` 边界变得模糊，一套代码里 CPU 与 GPU 可以自由地「看同一份数据」。

这套「统一内存」不是新概念——2017 年 Pascal 就引入了 Unified Memory。但 GB200 把它推向「物理上就近、带宽上够用」的新水平：**CPU 内存（LPDDR5X）与 GPU 显存（HBM3e）之间 1.8 TB/s 的 NVLink-C2C，让「统一寻址」第一次真正不拖性能后腿。**<span class="marginnote">统一内存的经典痛点是「页迁移」：数据在 CPU 侧时 GPU 访问要搬页，迁移本身很慢。GB200 的做法是「不搬页」——CPU 内存与 GPU 显存之间的一致性由 C2C 硬件维护，数据就地访问。这与《操作系统》的「分布式共享内存（DSM）」思想一脉相承，只是这次做进了硬件。</span>

**给读者的落点**：超级芯片把「CPU 服务器 + GPU 卡」的两件套，变成了「一颗带 CPU 的 GPU」。对开发者的影响是：**写代码时不再需要把「数据在 CPU 还是 GPU」当成头等大事**——这也是 GB200 敢叫「超级芯片」的原因。

## 5 超级芯片 vs 传统 PCIe 服务器：一张对照表

把 GB200 与最常见的「x86 服务器插 GPU 卡」放在一起，差异一目了然：

| 维度 | GB200 超级芯片 | x86 服务器 + PCIe GPU |
| --- | --- | --- |
| CPU | Grace（Arm，72 核） | x86（Intel/AMD） |
| CPU-GPU 互连 | NVLink-C2C，900 GB/s | PCIe Gen5，128 GB/s |
| 一致性 | 硬件缓存一致 | 显式拷贝（cudaMemcpy） |
| CPU 内存 | 480GB LPDDR5X，~1 TB/s | DDR5，0.3–0.5 TB/s |
| 数据搬运 | 基本免拷贝 | 每次训练步都拷 |
| 可编程性 | 统一寻址 | 显式管理两块内存 |

**核心差异不是「谁算得快」，而是「数据搬得省不省」。** 训练与推理里 CPU-GPU 的数据往返是常事（数据装载、预处理、MoE 路由、KV 缓存管理），PCIe 方案每一步都在为拷贝买单；超级芯片把这些「搬运税」直接免了。<span class="marginnote">这也解释了为什么 NVIDIA 为 GB200 专门造了 Grace 这颗 CPU：<strong>它要把「喂数据」这环做到极致，而不是替代 x86 的全部生态。</strong>对大部分通用工作负载 x86 仍是主角，但在「Grace + GPU」这个组合里，Grace 的任务非常纯粹——当最称职的数据管家。</span>

## 6 超级芯片意味着什么：一场产品逻辑的重构

Grace Blackwell 不只是「把 CPU 和 GPU 焊在一起」，它改变的是**「一台服务器里谁说了算」**。

**传统服务器**：CPU 是主人，GPU 是加速卡。GPU 要数据，得通过 CPU、PCIe、驱动——一切以 CPU 为中心组织。
**超级芯片**：GPU 是主角，CPU 是配角。Grace 的存在是为了「让 B200 吃饱」，连 CPU 的内存带宽与容量都是按 GPU 的需求设计的。**系统组织的中心从 CPU 移到了 GPU。**<span class="marginnote">这是一次微妙但深刻的转变：<strong>当计算重心从「通用 CPU」移到「专用 GPU」，连「服务器该长什么样」都要重写。</strong>NVL72 整柜、液冷、48V 供电，全都是这个新中心的自然推论。</span>

**对软件的影响**：CUDA 的 `host`/`device` 模型原本把 CPU 当「大脑」、GPU 当「计算器官」；超级芯片让两者更平等——CPU 可以当「数据管家」，GPU 可以当「唯一主角」。程序员写代码的思维方式也随之变化：**先想「计算放哪」，再想「数据怎么流」。**

**给读者的落点**：超级芯片是 NVIDIA「系统级竞争」的缩影——不只在芯片里竞争，还在「CPU + GPU + 内存 + 互连 + 软件」的整套组合里竞争。理解这一点，才能看懂为什么 NVIDIA 愿意投入造一颗「只为喂 GPU 而生的 CPU」。

## 7 小结

- **GB200 超级芯片 = 1 颗 Grace CPU + 2 颗 B200 GPU**，Grace 是 Arm Neoverse V2 的 72 核服务器 CPU。
- Grace 的定位是**「数据管家」**：480GB LPDDR5X、~1 TB/s 带宽，为「喂饱 GPU」而生。
- **NVLink-C2C（900 GB/s）** 提供缓存一致性互连，把 CPU 内存、GPU 显存黏成统一可编程存储池（合计超 1.3 TB）。
- 900 GB/s 是 PCIe Gen5 ×16（128 GB/s）的 **7 倍**，但质变在「免拷贝」的语义，而非带宽本身。
- 与 PCIe 服务器相比，超级芯片**免掉了 CPU-GPU 之间的搬运税**，是「机柜级 GPU」的第一级拼图。
- 超级芯片是一次**产品逻辑重构**：系统的组织中心从「通用 CPU」移到了「专用 GPU」，连服务器的形态都要重写。
- 对标教材提示：Grace CPU 规格、NVLink-C2C 一致性互连的细节，见 NVIDIA Blackwell 白皮书 §4 与 GB200 超级芯片官方文档。
- 与《操作系统》的呼应：统一寻址的「页迁移」痛点在超级芯片里被「一致性互连免拷贝」取代——CPU 与 GPU 共享同一地址空间，数据就地访问，是分布式共享内存思想做进了硬件。
- **辨析｜易错点**：NVLink-C2C（CPU-GPU）与 NVLink 5（GPU-GPU）不是一回事——前者强调缓存一致与共享内存，后者强调带宽与低延迟；GB200 里两者并存、各司其职。
- 一张表读懂「超级芯片的三类存储」：**Grace 的 LPDDR5X（480GB，喂数据）、每颗 B200 的 HBM3e（192GB×2，算数据）、NVLink-C2C 黏合成的统一池（>1.3TB，程序可见）**——三者分工明确、合成一个可编程整体。
- 记忆锚点：**GB200 = 1 CPU + 2 GPU + 1.3TB 统一存储**——超级芯片把「CPU 服务器 + GPU 卡」的两件套，变成「一颗带 CPU 的 GPU」。

在下一节，我们把超级芯片放进机柜——看 **DGX GB200 NVL72 整柜系统**如何用 36 颗超级芯片加 18 颗 NVSwitch，织成一颗 72 卡的「机柜级 GPU」。