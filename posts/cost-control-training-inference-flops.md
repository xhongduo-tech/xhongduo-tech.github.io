## 核心结论

训练与推理的 FLOPs 预算，本质上不是“模型大不大”的问题，而是“有限算力怎么分账”的问题。这里的 FLOPs 是 floating-point operations，白话说就是“芯片需要做多少次数学计算”。训练侧关心总账：一共要花多少计算、多久能训完；推理侧关心流水账：每秒要接多少请求、在服务等级协议（SLA，白话说就是“响应时间和可用性的承诺”）下能不能稳住。

对 dense decoder-only Transformer，训练总 FLOPs 常用经验式是：

$$
C_{train} \approx 6ND
$$

其中 $N$ 是参数量，$D$ 是训练 token 数。这个式子不追求电路级精度，但足够指导预算级决策。比如一个 $8\text{B}$ 模型训练 $3\times 10^{11}$ 个 token，估算就是：

$$
C_{train} \approx 6 \times 8\times 10^9 \times 3\times 10^{11}
= 1.44\times 10^{22}
$$

这笔账在上线前基本已经决定。上线后又是另一套账：每个请求都要消耗 prefill 和 decode 的计算，吞吐一高，持续成本会迅速变成主瓶颈。

可以把它理解成两本账。训练像一次性买车，先把整车成本付掉；推理像持续加油，车一旦开始跑，每公里都要花钱。很多系统失败，并不是模型不够先进，而是最开始就把这两本账混成了一本。

| 维度 | 训练预算 | 推理预算 |
|---|---|---|
| 核心问题 | 总 FLOPs、训练时长 | 吞吐、延迟、稳定性 |
| 主要变量 | 参数量 $N$、token 数 $D$ | QPS、prompt 长度、输出长度 |
| 典型错误 | 只看模型规模 | 只看平均吞吐 |
| 结果形态 | 一次性大额总账 | 持续性运营成本 |

---

## 问题定义与边界

先明确边界。训练 FLOPs 统计的是前向传播、反向传播和参数更新的总计算量。前向传播是“模型做一次预测”，反向传播是“根据误差回头改参数”，参数更新是“真正把权重改掉”。训练阶段最关心的是总账和工期：要多少卡、跑多少天、总电费和机会成本是多少。

推理 FLOPs 统计的是单请求或单位时间内的计算量。推理阶段最关心的是服务能力：每秒能接多少请求、p95 和 p99 尾延迟能不能守住。尾延迟指“慢的那部分请求到底有多慢”，线上系统真正爆炸时，通常先坏在尾部，不是均值。

几个常见术语也要先分开：

| 符号/术语 | 含义 | 白话解释 |
|---|---|---|
| $N$ | 参数量 | 模型里可学习数字的总数 |
| $D$ | 训练 token 数 | 训练时喂给模型的文本量 |
| prefill | 提示词阶段计算 | 先把整段输入读完并建立上下文 |
| decode | 逐 token 生成 | 每多生成 1 个 token 就再算一轮 |
| KV cache | 键值缓存 | 把历史中间结果存起来，避免重复算 |
| $\lambda$ | 请求速率 | 每秒来了多少请求 |
| $\eta$ | 安全余量 | 给抖动、故障、峰值预留的空间 |

玩具例子很直观。训练一个小模型时，你关注“这次作业什么时候跑完”；上线一个问答机器人时，你关注“100 个用户同时问问题会不会卡住”。同一个模型，离线训练和在线推理的预算口径完全不同。

真实工程例子更明显。批量训练一个基础模型时，决定总成本的大头通常是 $N$ 和 $D$；但上线客服机器人后，真正拉高账单的往往不是参数量本身，而是长 prompt、长输出、峰值并发和 SLA 余量。也就是说，训练成本更像资本开支，推理成本更像运营开支。

---

## 核心机制与推导

为什么训练常用 $6ND$？因为对 dense decoder-only Transformer，可以把每个 token 的训练成本近似看成和参数量成正比，前向大约 $2N$，反向再加上梯度相关计算，整体常被压缩成约 $6N$ 每 token。于是总训练成本就是每 token 成本乘以 token 总数，得到：

$$
C_{train} \approx 6ND
$$

这不是精确到 kernel 的公式，但对“预算应该落在哪个量级”非常有用。它直接告诉你：训练总账不是只由模型大小决定，而是由“模型大小乘训练数据量”决定。只看参数量、不看 token 数，是最常见的第一类错账。

推理不能只用一个“每请求平均 FLOPs”糊过去，必须拆成两段：

$$
C_{req} \approx C_{prefill}(L_p) + L_o \cdot C_{decode}
$$

其中 $L_p$ 是 prompt 长度，$L_o$ 是输出长度。prefill 是把整段输入吃进去，decode 是逐 token 生成。两者增长方式不同。短 prompt、长输出的任务，decode 占大头；长上下文总结、RAG 拼接文档这类任务，prefill 往往先把预算吃掉。

如果把请求速率记作 $\lambda$，单请求成本记作 $C_{req}$，GPU 可提供的有效 FLOPs/s 记作 $\eta F_{gpu}$，则可持续运行的基本约束是：

$$
\lambda C_{req} \le \eta F_{gpu}
$$

这里的 $\eta$ 不能取 1。真实系统存在批次波动、KV cache 膨胀、调度碎片、故障切换和尾延迟目标，所以必须预留安全余量。工程上把 $\eta$ 取成 0.4 到 0.8 往往比“理论跑满”更可信。

看一个数值例子。假设 $8\text{B}$ 模型，平均输出 200 token，每个生成 token 的 decode 成本粗略按 $2N$ 量级估：

$$
C_{decode,total} \approx 200 \times 2 \times 8\times 10^9
= 3.2\times 10^{12}
$$

如果每秒有 1000 个请求，仅 decode 部分就接近：

$$
3.2\times 10^{15}\ \text{FLOPs/s}
$$

也就是 PFLOPs/s 量级。这还没算 prefill。这个数量级足够说明：训练是大额一次性支出，但推理会在高并发下迅速变成持续主成本。

| 变量 | 含义 | 对成本的影响 |
|---|---|---|
| $N$ | 参数量 | 训练和推理都会近似线性上升 |
| $D$ | 训练 token 数 | 只直接影响训练总账 |
| $L_p$ | prompt 长度 | 主要抬高 prefill 成本 |
| $L_o$ | 输出长度 | 主要抬高 decode 成本 |
| $\lambda$ | 请求速率 | 直接抬高每秒总需求 |
| $\eta$ | 安全余量 | 余量越大，理论容量越保守 |

---

## 代码实现

下面给一个最小可用预算计算器。它不是性能模拟器，但能把训练总账、单请求推理账和吞吐约束拆开算清楚。

```python
def train_flops(num_params, num_tokens):
    # 经验公式：dense decoder-only Transformer 训练约为 6ND
    return 6 * num_params * num_tokens


def request_flops(prompt_len, output_len, num_params,
                  prefill_per_token_factor=2.0,
                  decode_per_token_factor=2.0):
    # 这里用“每 token 约等于 factor * N”做预算级近似
    prefill = prompt_len * prefill_per_token_factor * num_params
    decode = output_len * decode_per_token_factor * num_params
    return prefill + decode, prefill, decode


def required_flops_per_sec(qps, req_flops):
    return qps * req_flops


def sustainable_qps(gpu_flops_per_sec, req_flops, utilization=0.6, safety_margin=0.8):
    # 有效供给 = 峰值算力 * 利用率 * 安全余量
    return gpu_flops_per_sec * utilization * safety_margin / req_flops


def capacity_margin(gpu_flops_per_sec, qps, req_flops, utilization=0.6, safety_margin=0.8):
    supply = gpu_flops_per_sec * utilization * safety_margin
    demand = required_flops_per_sec(qps, req_flops)
    return supply - demand


# 玩具例子：1B 模型，prompt 100，输出 50，100 req/s
N = 1_000_000_000
req_total, prefill, decode = request_flops(prompt_len=100, output_len=50, num_params=N)
assert req_total == prefill + decode
assert train_flops(N, 1_000_000) == 6 * N * 1_000_000

# 真实工程例子：8B 模型，3e11 训练 token
train_total = train_flops(8_000_000_000, 300_000_000_000)
assert train_total == 1.44e22

# 假设集群有效供给 2e15 FLOPs/s，算可持续 QPS
req_total_8b, _, _ = request_flops(prompt_len=800, output_len=200, num_params=8_000_000_000)
qps_limit = sustainable_qps(2e15, req_total_8b, utilization=0.55, safety_margin=0.75)
assert qps_limit > 0

print("训练总 FLOPs:", train_total)
print("单请求 FLOPs:", req_total_8b)
print("估算可持续 QPS:", qps_limit)
```

这段代码故意把训练和推理拆成不同函数，因为两者回答的不是同一个问题。训练函数回答“总共要烧多少算力”；推理函数回答“一个请求多少钱、每秒能扛多少单”。

| 输入项 | 单位 | 说明 |
|---|---|---|
| `num_params` | 参数 | 模型规模 |
| `num_tokens` | token | 训练数据量 |
| `prompt_len` | token | 输入上下文长度 |
| `output_len` | token | 输出长度 |
| `gpu_flops_per_sec` | FLOPs/s | 集群峰值算力 |
| `utilization` | 比例 | 实际可用利用率 |
| `safety_margin` | 比例 | 给 SLA 与波动留余量 |

---

## 工程权衡与常见坑

第一类坑是只看参数量，不看 token 数。一个 8B 模型训练 50B token 和训练 300B token，账完全不是一个量级。第二类坑是只看平均吞吐，不看尾延迟。系统平均每秒能接 1000 个请求，不等于高峰和长 prompt 场景也能稳住 1000 个请求。

第三类坑是混淆 prefill 和 decode。很多团队只盯着“输出 1 个 token 的速度”，却忘了长 prompt 会先把 prefill 做满。做长文总结、RAG、代码仓库问答时，真正的大头可能在 prefill，不在 decode。

第四类坑是只看 FLOPs，不看显存和带宽。显存是“能装下多少中间状态”，带宽是“数据搬运有多快”。系统很可能不是算不动，而是 KV cache 撑爆显存，或者多卡通信把吞吐拖死。第五类坑是不留 $\eta$。预算里没有安全余量，线上就会拿 SLA 还债。

| 常见坑 | 表现 | 后果 | 规避办法 |
|---|---|---|---|
| 只看参数量，不看 token 数 | 训练预算明显偏低 | 工期和成本失控 | 用 $6ND$ 先建总账 |
| 只看平均吞吐，不看尾延迟 | 均值很好看，峰值崩 | SLA 违约 | 以 p95/p99 反推容量 |
| 混淆 prefill 和 decode | 误判长上下文成本 | 上线后账单暴涨 | 单独统计两段成本 |
| 只看 FLOPs，不看显存和带宽 | 理论算力够，实测跑不满 | 扩卡也不见效 | 联合看 KV cache、通信、带宽 |
| 不留安全余量 $\eta$ | 压线部署 | 峰值波动直接击穿 | 预留 20% 到 60% 缓冲 |

真实工程例子：同样是 8B 模型，短问答场景平均 prompt 100、输出 80，decode 往往更关键；企业知识库总结场景 prompt 可能到 8k 甚至 32k，prefill 会把账单结构彻底改写。很多“模型没变但成本翻倍”的事故，本质上不是模型退化，而是业务请求分布变了。

上线前至少检查三件事：请求长度分布而不是均值、峰值并发而不是日均流量、显存/KV cache 上限而不是只看理论 FLOPs。

---

## 替代方案与适用边界

FLOPs 预算适合做第一层容量规划，因为它简单、统一、能快速排除明显不合理的方案。但它不是万能指标。当系统进入 memory-bound、communication-bound 或 tail-latency-bound 状态时，只算 FLOPs 就会过度乐观。memory-bound 的意思是“瓶颈在内存读写，不在纯计算”；communication-bound 的意思是“瓶颈在多卡之间搬数据”；tail-latency-bound 的意思是“瓶颈在少数慢请求把 SLA 拉爆”。

| 方法 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| FLOPs 预算 | 早期容量规划 | 快、统一、便于比较方案 | 忽略显存与通信瓶颈 |
| Roofline / 带宽模型 | 怀疑带宽受限时 | 能识别算力瓶颈还是搬运瓶颈 | 建模更复杂 |
| 经验吞吐 benchmark | 上线前压测 | 最贴近真实系统 | 依赖具体硬件与实现 |
| token budget 预算 | API/产品计费 | 直连业务指标 | 对底层瓶颈解释弱 |
| SLA 驱动预算 | 强时延约束服务 | 直接面向用户体验 | 容易忽略长期算力成本 |

玩具例子：两台机器账面 FLOPs 一样，但一台显存小、带宽差，长上下文任务就是跑不起来。这时问题不是“算得慢”，而是“搬得慢、装不下”。

真实工程例子：MoE 模型不是所有参数每次都激活，dense FLOPs 估算会失真；长上下文服务更容易被 KV cache 和内存带宽限制；多模态模型还会引入图像编码和跨模态模块，预算结构进一步偏离纯文本 dense 模型。所以经验规则是：方案初筛用 FLOPs，接近上线时必须升级到压测、带宽和内存联合建模。

---

## 参考资料

1. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
2. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
3. [Techniques for Training Large Neural Networks](https://arxiv.org/abs/2203.03466)
4. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
5. [vLLM Documentation](https://docs.vllm.ai/)
6. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
