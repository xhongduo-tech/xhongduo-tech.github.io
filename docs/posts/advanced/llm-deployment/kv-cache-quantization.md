---
title: KV Cache 量化的收益与精度损失
date: 2026-08-07
---

# KV Cache 量化的收益与精度损失

<div class="epigraph">
<p>缓存里存着记忆，记忆的压缩要格外小心。</p>
<footer>—— 部署实践感悟（借自 KVCache 量化研究）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ KV Cache 量化综述与论文（KIVI 等） ｜ 2026-08-07</p>
</div>

## 为什么从 KV Cache 量化开始

本专题《KV Cache 显存占用估算》算过：上下文越长，KV Cache 越吃显存。权重量化（INT4/FP8）已经把权重压到极限，但**KV Cache 还在随上下文线性膨胀**——一个 128k 上下文的部署，KV Cache 体积可以数倍于权重。长上下文时代，KV Cache 成了新的显存黑洞与访存瓶颈。<span class="marginnote">KV Cache 量化与权重量化是<strong>互补的两块拼图</strong>：权重省的是「固定的静态存储」，KV Cache 省的是「随请求增长的动态存储」。两者都做了，长上下文才跑得动。</span>

本篇讲 KV Cache 量化的收益、为什么它对精度「意外地宽容」、K/V 分开处理的原理，以及量化 KV Cache 的精度损失到底从哪来。

## 1 KV Cache 的量级：量化能省多少

KV Cache 的体积公式（每层、每头、每个 token）：$2 \times L \times H \times d_h \times \text{bytes}$（K、V 各一份）。对一个 70B 模型（约 80 层、8k 隐藏维），**每个 token 的 KV Cache 约 1 MB**（FP16 下）。<span class="marginnote">这个数字来自本专题《KV Cache 显存占用估算》：<strong>FP16 下每 token ≈ 2 × 层数 × 隐藏维 × 2 字节</strong>。70B 级模型约 1 MB/token，128k 上下文就是约 128 GB——比权重还大。</span>

量化到 INT8：每 token 0.5 MB，128k 上下文降到 64 GB；量化到 INT4：每 token 0.25 MB，降到 32 GB。**KV Cache 量化是长上下文部署的「使能技术」**——不做它，128k 上下文在单卡上基本不可能。

但收益不止显存：**访存也减半/减四分之三**。decode 时每个 token 都要读取全部历史 KV Cache 做注意力（见 FlashDecoding 篇），KV Cache 量化直接降低这个最大访存项的字节数，decode 吞吐随之提升。

## 2 为什么 KV Cache 对量化「宽容」

直觉上，KV Cache 参与注意力打分，精度损失应该直接伤害生成质量。但大量实验显示，KV Cache 量化到 INT8 精度损失极小，INT4 也往往可接受。原因有三：

- **注意力打分是「相对比较」**：softmax 对 $QK^T$ 做归一化，**共同的缩放误差会抵消**。KV Cache 量化引入的系统性偏差，很大一部分在 softmax 里被「平均掉」。
- **K 与 V 的容错不同**：K 影响 softmax 的分数（对误差敏感），V 影响加权求和的结果（相对宽容）。**分开量化（K 高精度、V 低精度）比统一量化省得多**。
- **局部性**：注意力是「按位置加权」，近期 token 的权重高、远期 token 权重低。KV Cache 量化的误差更多影响远期低权重 token，被加权和「稀释」。

**关键洞察：不是 KV Cache 不在乎精度，而是「量化的误差结构」恰好落在注意力不敏感的方向上。** 这让 KV Cache 量化成为「低成本高回报」的操作。

## 3 工程实现：K 与 V 分开、per-channel 与 per-token

主流 KV Cache 量化（如 KIVI、KVQuant）的工程要点：

- **K 与 V 分开量化**：K 用 per-channel（或 per-head）scale，V 用 per-token scale。K 对误差敏感但通道分布稳定，V 宽容但逐 token 分布变化大——**各自用最合适的粒度**。<span class="marginnote">存储 scale 有额外开销：per-channel 的 scale 只要每通道一个（便宜），per-token 的 scale 要每个 token 一个（贵）。工程上常在<strong>「scale 的存储开销」与「量化的步长精度」</strong>之间权衡。</span>
- **在线/离线量化**：离线（权重固定、统计一次）对 K 友好；V 随 token 变化，有时需要在线统计。多数引擎用离线校准 + 少量在线修正。
- **与页式内存结合**：量化后的 KV Cache 依然按页（block）分配，页大小、对齐方式要配合新位宽调整——vLLM 的 KV 量化、TensorRT-LLM 的 INT8/FP8 KV 路径都改了块布局。

**辨析｜易错点：KV Cache 量化的收益在「长上下文」才体现。** 短上下文（几百 token）下 KV Cache 体积小，量化省下的显存与访存占比低，反而引入不必要的精度风险与 kernel 复杂度。**KV Cache 量化是「长上下文专用工具」，别在短上下文场景盲目启用**。

## 4 公式解析：量化 KV 后注意力的误差传播

设原始注意力分数为 $s_{ij} = q_i^T k_j$，KV Cache 量化后 $k_j \to \hat{k}_j = k_j + \epsilon_j$（$\epsilon_j$ 为量化误差）。量化后的分数：

$$\hat{s}_{ij} = q_i^T (k_j + \epsilon_j) = s_{ij} + q_i^T \epsilon_j$$

- **第一步，读误差的放大因子**：误差项 $q_i^T \epsilon_j$ 由 query 与误差的内积决定。$q_i$ 的范数若较大（未归一化时），误差会被放大；若 $q_i$ 与 $\epsilon_j$ 方向正交，误差项为 0。**这正是 per-channel 量化 K 的理由：让 $\epsilon_j$ 的各分量与典型 query 的匹配尽量小**。
- **第二步，读 softmax 的缓冲**：softmax 输出 $p_j = e^{\hat{s}_{ij}} / \sum_l e^{\hat{s}_{il}}$。误差 $\delta_j$ 使 $p_j$ 偏移，但**归一化分母保证所有概率和为 1**——系统性偏移被重新分配而非累积。
- **第三步，读最终输出**：输出 $o_i = \sum_j p_j v_j$。V 的量化误差 $\eta_j$ 直接进入加权和，但由于 $p_j$ 集中在少数 token，**高权重位置的误差被放大、低权重位置的误差被稀释**。总体精度损失近似：

$$\Delta o \approx \sum_j p_j \, \eta_j + \sum_j \frac{\partial p}{\partial s} \, (q^T \epsilon)$$

第一项（V 的误差）与第二项（K 的误差）都由「注意力集中度」调节——**注意力越集中，V 的误差越重要、K 的误差越次要**。这解释了为什么对长上下文（注意力通常更分散）KV 量化更安全。

## 5 小结

- **KV Cache 是长上下文的新瓶颈**：128k 上下文的 KV Cache 可数倍于权重，量化它是长上下文部署的使能技术。
- **收益双份**：显存减半/减四分之三，decode 访存同步下降、吞吐提升。
- **KV Cache 对量化「宽容」**：softmax 归一化抵消系统误差、注意力加权稀释远端误差、K/V 容错不同。
- **工程要点**：K 用 per-channel、V 用 per-token，scale 存储与步长精度权衡，与页式内存布局配合。
- **适用场景**：长上下文收益显著；短上下文收益低、风险与复杂度不划算。

在下一节，我们把量化篇收尾——**量化模型的精度评测方法**，看怎么科学判断「这个量化能不能上线」。
