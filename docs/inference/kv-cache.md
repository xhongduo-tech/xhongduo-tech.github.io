---
title: KV Cache与注意力执行
description: 从多层因果不变量证明增量计算等价，再用数值反例、分页状态和固定源码检查复用是否安全。
kind: topic
status: planned
---

# KV Cache 与注意力执行

::: info 深度专题计划 · 待研究
本页给出待完成练习的推导、reference 规格和验收条件。数值例子是解析预期；核验源码入口不代表本人已经读通、实现或完成实验。尚无个人 CPU 练习记录或硬件性能结果。
:::

## 研究问题：复用之后，为什么还是同一个计算？

KV Cache 复用过去已经算好的状态；分页通过块表改变这些状态的存放方式；前缀共享与回收再决定它们何时能被其他请求读取、何时能被覆盖。这三步都需要守住同一件事：**当前 query 读到的，仍然是该模型对当前有效历史应有的 K、V。**

本页沿这条依赖展开：因果性使旧状态不变 → 增量计算与整段计算等价 → 地址映射保持逻辑状态 → 共享和回收保持所有权 → 用源码与反例检查每个条件。

先修只需矩阵乘法、softmax、因果 mask、张量 shape 和位置编号。首次练习从下面的单头小矩阵开始；[基础地图](/foundations/)负责补齐概念，不要求同时学习整个推理系统。

## 一、复用成立的前提，比“用了因果 mask”更多

### 固定的输入和计算规则

先把研究对象限制为具有因果 self-attention 的 decoder，比较同一段**预先固定的 token ID 序列**。两条计算路径须满足：

| 条件 | 为什么会影响旧 KV |
| --- | --- |
| 相同 token ID、有效前缀与请求边界 | 原始文字相同，不保证 tokenizer、模板和特殊 token 处理后仍相同；不同请求不能串用历史 |
| 相同模型参数、adapter 及相关配置 | K、V 是参数化计算的结果；更换权重或 adapter 后，旧缓存不能继续被当作同一模型的状态 |
| 相同逻辑位置与位置编码规则 | 新 token 延续原有位置；padding、块号和物理槽位不能冒充模型位置；位置编码的频率、缩放规则也须一致 |
| 因果可见范围与 token 局部模块不变 | 当前层的位置只依赖允许的历史；通常逐 token 的归一化、残差和 FFN 不引入未来依赖 |
| 固定推理模式与随机行为 | 关闭 dropout 等随机层；只设同一个 seed，未必让整段与逐步路径使用相同随机掩码 |
| 相同缓存语义、mask 与精度约定 | 漏掉当前 token、重复追加、误读 padding 或额外量化，都改变了待比较的计算 |

如果某模块的行为依赖其他请求、整个序列长度或动态更新的外部状态，还要单独证明它在追加 token 后不改变旧结果。不能仅凭 attention 是因果的，就替整个模型作出保证。

因果 attention 与逐位置模块的结构依据见 [Transformer 原论文 §3.1–3.3](https://arxiv.org/html/1706.03762v7#S3)。缓存的逐层结构、位置和 mask 接口可对照 [Transformers v4.57.1 文档](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)；具体接口的 cache position 与模型 position ID 可能承担不同职责，应分别核对，不能一律视为同一个数组。

### 多层因果不变量

记 $h_i^{(\ell;n)}$ 为“输入前 $n$ 个 token 时，第 $\ell$ 层在位置 $i$ 的表示”。在上述前提下，待证明的不变量是：

$$
h_i^{(\ell;n)}=h_i^{(\ell;i)}
\qquad(n\ge i).
$$

含义是：**在位置 $i$ 之后追加 token，不改变位置 $i$ 在任何一层的表示。**

证明按层进行：

1. **输入层。** 相同 token 和位置得到相同输入表示。
2. **假设前一层成立。** 第 $\ell$ 层位置 $i$ 只读取前一层的 $1,\ldots,i$，这些表示按假设都未改变。
3. **当前层。** Q/K/V 投影、位置变换、允许的 attention 行及 token 局部模块输入相同，故输出相同。
4. 对全部层重复，即得到不变量；每层由旧位置表示产生的 K、V 也随之保持不变。

这里说明了为什么要**逐层缓存**：第二层的旧 K、V 来自第一层的旧表示，而不只是原始 embedding。单层 attention 的小例子只能检查其中一环，不能替代多层论证。

## 二、从不变量推出整段与增量计算等价

在某一层，省略 head 下标，位置 $t$ 的输出为：

$$
o_t=\operatorname{softmax}
\left(\frac{q_tK_{1:t}^{\mathsf T}}{\sqrt{d_h}}\right)V_{1:t}.
$$

整段路径同时算出所有位置；增量路径只计算新位置的 $q_t,k_t,v_t$，再构造逻辑视图：

$$
K_{1:t}=[K_{1:t-1};k_t],\qquad
V_{1:t}=[V_{1:t-1};v_t].
$$

旧 K、V 由上一节的不变量保证相同，新 K、V 与 query 由同样的输入和参数产生，因此此层的新输出相同。把新位置的表示继续送入下一层，同理得到最后的 hidden state 和 logits 相同。

这份推理同时解释了两个容易漏掉的细节：

- 历史 Q 只用于计算其自身位置当时的输出；未来位置读取的是历史 K、V，因此这条路径不要求保留所有旧 Q。
- 缓存避免重复计算旧位置，当前 query 仍需读取有效历史；不能由此推断长上下文 attention 的成本变成常数。

### 有历史时，矩形 mask 怎样对齐

已有 $T$ 个缓存位置，本次一次处理 $m$ 个新位置。对新 chunk 的第 $a$ 行，允许读取全部历史和当前 chunk 中截至该行的位置。用从 0 开始的索引写为：

$$
M_{a,j}=
\begin{cases}
0,&j\le T+a,\\
-\infty,&j>T+a,
\end{cases}
\quad
0\le a<m,\quad 0\le j<T+m.
$$

这里 $M$ 是加到 attention score 的二维因果偏置；还要叠加有效长度和 padding 约束。它与模型接口可能接收的二维 padding mask、广播后的四维 mask 不应混为一个 shape。

例如 $T=1,m=1$，新 query 应看见两项，允许范围是 $[1,1]$。若机械地构造 $1\times2$ 的左上角三角形，会得到 $[1,0]$，把当前 token 排除。多 token chunk 则还必须屏蔽 chunk 内的未来位置。

### 数学等价与数值验收分别处理

上面的等号是在相同算子的精确算术模型下成立。真实整段与增量路径可能选用不同矩阵内核、归约顺序或精度，不能默认逐 bit 一致。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)也明确区分数学等价与浮点执行结果。

CPU float64 教学 reference 可预先采用逐元素检查：

$$
|a-b|\le \mathrm{atol}+\mathrm{rtol}|b|.
$$

例如先设 atol、rtol 均为 $10^{-12}$，同时检查有限值；这是本页小规模 reference 的验收约定，不是所有 GPU／低精度模型的通用阈值。若失败，先定位首个偏差层、位置和张量，不能只放宽容差。

比较时固定输入 token，先核对逐层 KV、hidden state 与 logits。随机采样发生在此后；logits 在容差内接近，并不自动保证所有生成文本逐字相同，尤其在概率接近或 argmax 临界处。

## 三、一个可以手算，也应该故意算错的 tiny 例子

### 正确计算的解析预期

取单层、单头、$d_h=1$，输入为 $x_1=1,x_2=2$，三个投影均为恒等映射；不加位置编码、残差、归一化或 FFN。它只是 attention 单元，不是完整语言模型。

此时第二个 query 为 2，对两个 key 的 score 为 $[2,4]$，于是：

$$
o_2=\frac{e^2\cdot1+e^4\cdot2}{e^2+e^4}
=\frac{1+2e^2}{1+e^2}
\approx1.880797077978.
$$

第一步缓存 $(k_1,v_1)=(1,1)$，第二步追加 $(2,2)$，再计算新 query，应得到同一个值。以下数值由式子解析得出，**不是本人已经完成练习的记录**。

| 路径或故障 | 读到的内容 | 第二位置的解析输出 |
| --- | --- | --- |
| 整段因果 attention | K、V 都是 $[1,2]$ | 约 1.880797077978 |
| 正确增量追加 | 历史 $[1]$ 加当前 $[2]$ | 同上 |
| 漏掉历史 | 只保留当前 $[2]$ | 2 |
| mask 排除当前 token | 只允许读历史 $[1]$ | 1 |
| K 正确、V 次序错位 | K 为 $[1,2]$，V 为 $[2,1]$ | 约 1.119202922022 |

同一个输入还能否证“没有因果约束也能一直复用旧表示”：若第一个 query 看见未来 token，第一位置输出变成

$$
o_1^{\mathrm{unmasked}}
=\frac{e^1\cdot1+e^2\cdot2}{e^1+e^2}
\approx1.731058578630,
$$

而只输入第一个 token 时为 1。旧位置已经改变；在多层模型中，依赖该旧表示的更高层 K、V 也可能改变。此时失去的是原证明的前提，不能归结为缓存实现的小数误差。

### reference 要检查哪些东西

计划由学习者实现三条最小路径，先不引入真实权重：

| 路径 | 输入与输出约定 | 核对对象 |
| --- | --- | --- |
| 整段 reference | 固定完整输入与因果 mask，返回每个 query 的输出 | 用显式小矩阵建立基准 |
| 连续缓存 | 每次只传新 token，返回新增输出与更新后的缓存 | 每步 KV、长度和输出 |
| 分页逻辑视图 | 块表、物理存储、有效长度 → 按逻辑位置读取的 KV | 先逐位置比 KV，再比输出 |

一个重要的反向检查是：**把 K、V 成对做相同置换，第二个 query 的加权和可能完全不变。** 此例中同时把两者变为 $[2,1]$，仍得到约 1.880797077978。因此只比较最后一个输出，可能漏掉逻辑顺序错误；还要检查每个逻辑位置的 KV，并检查不同可见范围的 query 行。

这份无位置编码的 tiny 例子也无法抓住“位置重置为 0”的错误。要检查位置前提，须另加非平凡位置编码的固定 fixture；要声明多层实现通过，则另加至少两层、可追踪中间状态的 reference。不能把尚未覆盖的性质写成已验证。

## 四、为什么分页仍能保持同一个逻辑状态

### 先看需要管理多少状态

对各层 KV head 数和维度相同、K/V 同精度的普通 MHA／GQA，未共享时的有效数据量近似为：

$$
M_{\mathrm{KV}}\approx
2L H_{\mathrm{KV}}d_h
\left(\sum_{r=1}^{B}T_r\right)\frac{b}{8}
\quad\mathrm{bytes}.
$$

它按 KV head 数计，不是 query head 数；不包含块预留、元数据、量化参数、临时区或设备副本。MLA、混合层、滑动窗口以及后续架构需按实际缓存张量重新算；[V3 的 MLA](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)不能直接套这个式子。

分页要处理的是动态增长、分配和复用。它通过块表保持逻辑历史，配套内核按该映射访问数据；[PagedAttention 论文 §4](https://arxiv.org/html/2309.06180v1#S4)给出机制。FlashAttention 则处理 attention 执行的 IO 组织，两者可以协作，但不是同一个状态管理问题。

### 逻辑位置怎样变成物理槽号

以下只研究一个 cache group、无 context-parallel 分片的简化布局；块索引从 0 开始。设块长为 $S$，请求 $r$ 的块表为 $\mathrm{BT}_r$，则逻辑 token $j$ 的 token 槽号为：

$$
\mathrm{slot}(r,j)
=S\cdot\mathrm{BT}_r\!\left[\left\lfloor j/S\right\rfloor\right]
+(j\bmod S).
$$

这是 token 槽号，不是完整字节地址；层、head 和具体张量 layout 另有维度。

将 tiny 序列再追加第三个 token，取块长 $S=2$，请求 R 的逻辑长度为 3，块表为 $[7,3]$：

| 逻辑 token 索引 | 逻辑块／块内偏移 | 物理块 | token 槽号 |
| --- | --- | --- | --- |
| 0 | 0／0 | 7 | 14 |
| 1 | 0／1 | 7 | 15 |
| 2 | 1／0 | 3 | 6 |

块 3 的偏移 1 已有存储容量，但尚无有效 token。**分配容量不等于可读历史。** reference 应只读槽 14、15、6；给无效槽填入明显不同的有限哨兵值，可检查有效长度错误是否会污染结果。

这段地址公式可直接对照固定源码 [BlockTable.compute_slot_mapping 的非 DCP 分支](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/worker/block_table.py#L106)。它不意味着 attention 内核必须先物化一个连续 gather 张量；逻辑视图是用于说明和检查读取语义的 reference。

### 必须持续成立的状态不变量

| 不变量 | 可以怎样检查 |
| --- | --- |
| 有效位置都映射到正确、已就绪的 KV | 对每个层、head、逻辑位置，与连续 reference 比较；预留或待写位置不能提前当作有效数据 |
| 同一 token 的 K/V 配对一致 | 给不同 token 不同数值，故意只改 K 或 V 的映射，确认检查会失败 |
| 长度与 mask 共同限定可见范围 | 改变尾部无效槽内容，合法输出不应变化；chunk 内未来位置仍不可见 |
| 共享的已完成前缀不被请求私自覆盖 | 两请求共享前缀后分别追加，另一请求原有逻辑 KV 不应变化 |
| 读者仍在使用的块不被重新分配 | 在完成、取消、共享和重用边界记录所有者与引用变化 |
| 重用物理块时，旧身份先失效 | 新请求写入后，旧前缀键不能仍命中已换内容的块 |

这些是教学模型的稳定状态约束；真实系统可能预先登记待计算块，但必须由执行依赖保证写先于读。仅看到块号或哈希已经登记，不能推出对应设备数据已经完成。

## 五、共享与回收：释放请求不等于立刻擦除数据

### 为什么 prefix key 必须带上历史

设两个请求都出现同一个块内 token 序列 C、D，但前面分别是 A、B 与 X、Y。更高层表示依赖各自允许的历史；仅凭 C、D 相同，不能推出这些位置的所有层 KV 相同。

所以前缀共享的安全条件需要回到第一节：有效 token 前缀、位置和计算配置一致。权重通常在引擎命名空间内固定，adapter 等额外身份需纳入实现的键或隔离规则；原始字符串相同不足以验收。

固定源码 [hash_block_tokens](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_utils.py#L547)组合 parent hash、当前块 token ID 与 extra keys；[该版设计文档](https://docs.vllm.ai/en/v0.11.0/design/prefix_caching.html)说明 LoRA、多模态标识和隔离盐等情形。更换参数后，旧缓存必须通过失效或不同命名空间排除；该版也提供 [reset_prefix_cache](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L316)。

### 沿一个块看完整生命周期

沿用块长 2、R 的块表 $[7,3]$。假设 Q 的长输入以 R 的前两个 token 开始，获得已完成的完整前缀块 7，后续 token 另行计算。以下忽略保留 null block 和设备异步读写，只检查请求所有权：

| 稳定状态／事件 | 块 7 的引用数 | 块 3 的引用数 | 应当发生什么 |
| --- | --- | --- | --- |
| 仅 R 持有 $[7,3]$ | 1 | 1 | 两块都不能交给无关请求覆盖 |
| Q 也持有完整前缀块 7 | 2 | 1 | 共享数据不变；后续追加使用各自可写位置 |
| R 完成或取消 | 1 | 0 | 块 7 仍受 Q 保护；块 3 才成为可重用候选 |
| Q 也释放该前缀 | 0 | 0 | 块 7 可以保留缓存身份，等待再次命中或淘汰 |
| 块 7 被用于无关新内容 | 新所有者计数 | — | 先移除旧缓存身份，再允许作为新块使用 |

在这个教学模型里，引用数等于持有该块的请求数，每个请求对同一块只持有一次。**引用数为 0 表示没有这些请求持有，不表示内容已清零，也不表示 prefix hash 必须立即删除。**

这正好对应 [BlockPool.touch](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py#L322)、[free_blocks](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py#L338)和[重新分配／淘汰路径](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py#L257)：未被使用的缓存块可进入候选队列，命中时重新保护，真正改作他用时才失去旧身份。

本页采用该版 V1 前缀缓存只共享完整块的简化路径。若系统允许共享仍会被修改的尾块，须有 copy-on-write 或等价的写隔离；这是 [PagedAttention 论文 §4.4](https://arxiv.org/html/2309.06180v1#S4.SS4)讨论的另一种路径，不由本页所列 V1 函数自动证明。

真实异步系统还需要保护尚未完成的读写或传输。引用计数不能单独证明执行顺序安全；例如该版 [scheduler 的请求释放路径](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L1147)存在 connector 要求延迟释放的分支。CPU 所有权模拟通过，仍不代表 GPU 生命周期或竞态已经验证。

## 六、把每个不变量对应到真实源码

这里有意固定 **vLLM v0.11.0，commit b8b302cde434df8c9289a2b465406b47ebab1c2d**，只作为可复查的教学快照，不称其为当前最新版本。先读普通 decoder、完整 attention、非 cascade、无分布式分片的路径，再看额外分支。

| 阅读顺序与入口 | 要带着什么问题读 | 对应前文的检查 |
| --- | --- | --- |
| [KVCacheManager.get_computed_blocks](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L154) | 返回哪些完整块？命中长度如何受限制？ | 不能把有 KV 等同于已有所需 logits；此版为获得 logits 会限制末 token 命中 |
| [allocate_slots](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L193) | 怎样区分已计算、要计算和 lookahead 容量？何时 touch 命中块？ | 容量、有效长度和前缀保护是不同状态 |
| [BlockPool 的分配与引用路径](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py#L257) | 哪些块可选，旧 hash 在何处清除，何时增减引用？ | 完成／取消／共享／重用的事件表 |
| [BlockTable.compute_slot_mapping](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/worker/block_table.py#L76) | 请求索引、位置和块号如何生成槽号？ | 手算的 14、15、6；不要把多请求 batch 的行号当请求身份 |
| [FlashAttentionImpl.forward 中的缓存写入](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/attention/backends/flash_attn.py#L489) | 新 K/V 通过哪个 slot mapping 写入？ | 当前 token 的位置和 K/V 配对 |
| [同一函数的普通 attention 调用](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/attention/backends/flash_attn.py#L516) | block table、seq lengths、causal 参数如何传给内核？ | 逻辑历史、有效范围与 mask 是否同时到位 |

这是一条数据流阅读路线，表中相邻行不保证在源码中互相直接调用；协调器、runner 和模型执行还承接中间步骤。首次的验收是能指出字段在哪里产生、传给谁、由谁消费。以上路径已核验存在；其余调用链、各模型／内核分支仍待学习者逐项确认。

## 七、对照、消融与否证条件

先做正确性，再谈运行成本。所有对照使用同一固定 token 输入，禁止让采样历史提前分叉后仍比较不同上下文的输出。

| 待检查的主张 | 只改变什么 | 什么结果会否证当前实现或暴露测试不足 |
| --- | --- | --- |
| 连续缓存复现整段计算 | 全量重算 ↔ 增量追加 | 在约定容差外出现首个 KV／hidden／输出偏差；应定位首个层和位置 |
| 分页只改变存放方式 | 搬动物理块，并一致更新块表 | 逻辑 KV 不同即失败；输出相同不足以掩盖映射差异 |
| mask 包含当前、排除未来与无效槽 | 注入错位 mask 或修改哨兵槽 | 合法路径不应受无效槽影响；已知故障仍未被检出，说明 fixture 太弱 |
| 位置一致是等价前提 | 在有位置编码的 fixture 中重置新位置 | 对照应能发现位置或状态差异；无位置编码的例子不能验收此项 |
| 前缀命中保持同一计算身份 | 仅更换 adapter／参数命名空间或历史前缀 | 旧身份仍被当作有效命中即失败；不要求所有输入的数值必然变化 |
| 共享块在最后持有者释放前安全 | R 释放，Q 继续读；随后再重用 | 提前覆盖、负引用、重复回收、旧键命中新内容或跨请求污染均失败 |

这些故意破坏前提的负对照用来证明测试能抓到错误；它们不推翻“在前提成立时数学等价”的结论。即使全部通过，也只覆盖已列性质，不能外推为所有模型、所有 batch 和所有并发时序正确。

### 后续硬件实验：单独检验收益

若有兼容 GPU，再选一个小模型与固定引擎，先核对输出与缓存配置。只改变 prefix caching 开关或命中比例中的一项，分别测冷、暖缓存；记录实际命中 token、排队、prefill 时间、TTFT、吞吐与峰值内存。

“命中率提高，所以服务必然更快”也是待否证主张：若相同负载下重算确实减少，但排队、查找／管理或其他成本抵消收益，就只能结论为减少了重复计算，不能宣称端到端加速。块长实验另固定请求长度分布和缓存预算，不与命中策略一起改变。

没有硬件时，以数学解释、CPU reference 和状态日志作为本轮范围，性能项继续标记待实测；不需要编造 GPU 数字。

## 八、本页怎样才算完成一次学习

计划留下五份可检查的材料，而不是只打勾“学过 KV Cache”：

1. **一页论证。** 用自己的话说明多层因果不变量、增量等价前提和位置／mask 边界，解释为什么不缓存所有历史 Q。
2. **一份 reference 记录。** 同一 tiny 输入下的整段、连续、分页逻辑 KV 和输出；包含已知故障，逐项注明哪些性质尚未覆盖。
3. **一份状态日志。** 手算并核对槽号、两请求共享、取消／完成、最终释放与物理重用；说明 ref 为 0 与 hash 仍有效如何同时成立。
4. **一份源码对照。** 固定 commit，标出命中长度、槽号、引用与有效长度字段的生产和消费位置；区分确认过的代码与推测。
5. **一次迁移与排错。** 在已实现范围内改变 chunk 长度、块长或共享请求数中的一项，先预测结果；再定位一个未预先告知位置的映射、长度或身份错误，解释它破坏哪个不变量并验证修复。允许 AI 协作，判断与解释需要本人完成。

练习代码与这些解释由学习者实际完成后，才记录限定范围内的相应证据；未确认的模型分支、内核和异步竞态继续待深入。本文的规格、解析结果和资料核验不计为个人已完成学习或实验。

## 一手资料与版本

本轮核验日期：**2026-11-17**。以下是教学快照与机制依据，不代表推荐按这些旧版本部署当前服务。

- [Attention Is All You Need，arXiv:1706.03762v7](https://arxiv.org/html/1706.03762v7)：因果 decoder、逐位置模块和 attention 公式。
- [Transformers v4.57.1：Caching](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)：逐层 KV、增量输入、mask 与 cache position 的接口约束。
- [PyTorch 2.15：Numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)：数学等价、批量计算与浮点结果的边界，不作为本页运行环境声明。
- [PagedAttention，arXiv:2309.06180v1](https://arxiv.org/html/2309.06180v1)：逻辑／物理块、共享与写隔离；论文的 2023 年设计与后来的 V1 实现分开看。
- [vLLM v0.11.0 prefix caching 设计](https://docs.vllm.ai/en/v0.11.0/design/prefix_caching.html)、[固定源码树](https://github.com/vllm-project/vllm/tree/b8b302cde434df8c9289a2b465406b47ebab1c2d)：本页地址映射、引用和回收路径的依据，具体函数见源码表。
- [FlashAttention，arXiv:2205.14135v2](https://arxiv.org/abs/2205.14135v2)：用于区分 attention IO 优化与分页状态管理。
