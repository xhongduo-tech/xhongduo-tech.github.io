## 核心结论

LLM 推理显存可以先拆成一句话：`推理显存 = 常驻项 + 线性增长项`。常驻项主要是模型权重，线性增长项主要是 KV Cache。激活值是前向计算中的中间结果，白话说就是“这一层刚算出来、下一层马上要用的数据”，在常见自回归推理里通常不是主瓶颈。

权重像一批搬进机房后基本不再变化的设备，加载一次就长期占着显存；KV Cache 是历史上下文的缓存，白话说就是“模型为了不重复读旧内容而保存的记忆”，每多一个 token 就继续增长。于是结论很直接：

$$
M_{weight}=P \times b_w
$$

$$
M_{KV}=2 \times L \times H_{kv} \times d_h \times S \times b_{kv}
$$

其中前面的 `2` 表示同时存 `K` 和 `V` 两份。

| 项目 | 是否常驻 | 是否随上下文长度 $S$ 增长 | 是否随 batch 增长 | 常见优化 |
| --- | --- | --- | --- | --- |
| 模型权重 | 是 | 否 | 否 | W4/AWQ/GPTQ/FP8 |
| KV Cache | 否 | 是，线性增长 | 是，近似线性增长 | GQA、KV 量化、PagedAttention |
| 激活值 | 部分暂存 | 弱相关 | 相关 | 一般不是推理主因 |

一个必须先纠正的常见误解是：70B 模型的 4K KV Cache 不是几十 GB 级别。以 Llama 3.1 70B 常见公开结构 `L=80, H_kv=8, d_h=128, b_kv=2, S=4096` 估算，KV Cache 约为 `1.34 GB`，而不是 `80 GB`。`80 GB` 更接近超长上下文，例如 128K，再叠加并发后的量级。

---

## 问题定义与边界

本文只讨论推理阶段，不讨论训练阶段。训练会额外保存梯度、优化器状态和更大的激活值；推理则主要关心“模型能不能装下”和“上下文拉长后会不会爆显存”。

统一记号如下：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $P$ | 参数量 | 模型里一共有多少个可学习参数 |
| $b_w$ | 权重每参数字节数 | 一个权重占几个字节，如 FP16 是 2 |
| $L$ | 层数 | Transformer 堆了多少层 |
| $H_{kv}$ | KV heads 数 | 真正存进缓存的头数，不一定等于 Q heads |
| $d_h$ | 每个 head 的维度 | 每个注意力头内部向量长度 |
| $S$ | 上下文长度 | 当前已处理 token 数 |
| $b_{kv}$ | KV 每元素字节数 | KV Cache 每个数值占几个字节 |

边界要分清，因为不同场景的瓶颈不同：

| 维度 | 场景 | 显存主因 |
| --- | --- | --- |
| 推理 vs 训练 | 推理 | 权重 + KV Cache |
| 推理 vs 训练 | 训练 | 权重 + 激活 + 梯度 + 优化器状态 |
| 短上下文 vs 长上下文 | 短上下文单轮 | 权重更可能先成为瓶颈 |
| 短上下文 vs 长上下文 | 长上下文/多轮/RAG | KV Cache 更可能先成为瓶颈 |
| 单请求 vs 高并发 | 单请求 | 先看权重能否装下 |
| 单请求 vs 高并发 | 高并发 | KV 总量和碎片管理更关键 |

玩具例子：单次 4K 输入看起来不长，但如果你同时服务 64 个请求，KV Cache 会近似按 `64 × 4K` 的总 token 规模上涨。也就是说，真正压垮显存的经常不是“单个请求太长”，而是“长上下文和并发叠在一起”。

---

## 核心机制与推导

先看权重。参数量是 $P$，每个参数如果用 FP16 存储，就是 2 bytes，所以：

$$
M_{weight}=P \times b_w
$$

70B 模型在 FP16 下约为：

$$
70 \times 10^9 \times 2 \approx 140\,GB
$$

这就是“为什么很多 70B 模型单卡根本装不下”的直接原因。

再看 KV Cache。对单层、单 token 来说，每个 KV head 需要存一份 key 和一份 value，因此显存是：

$$
M_{1\ layer,\ 1\ token}=2 \times H_{kv} \times d_h \times b_{kv}
$$

扩展到 $L$ 层、$S$ 个 token：

$$
M_{KV}=2 \times L \times H_{kv} \times d_h \times S \times b_{kv}
$$

这里最重要的不是背公式，而是看出线性关系：层数翻倍，KV 翻倍；上下文长度翻倍，KV 翻倍；并发翻倍，KV 总量也近似翻倍。

玩具例子：假设一个小模型只有 `L=2, H_kv=2, d_h=4, S=8, b_kv=2`，那么

$$
M_{KV}=2 \times 2 \times 2 \times 4 \times 8 \times 2 = 512\ bytes
$$

这个数很小，但增长规律和大模型完全一样。

真实工程例子：Llama 3.1 70B 公开可见的是 `num_hidden_layers=80`、`num_attention_heads=64`、`num_key_value_heads=8`、`max_position_embeddings=131072`。因此 4K 上下文下：

$$
M_{KV}=2 \times 80 \times 8 \times 128 \times 4096 \times 2 \approx 1.34\,GB
$$

如果把上下文拉到 128K：

$$
M_{KV}(128K)\approx 1.34 \times 32 \approx 42.9\,GB
$$

如果再有多个并发请求，总量会继续叠加，这时 KV Cache 才真正进入“几十 GB 很正常”的区间。

GQA 的作用也就清楚了。GQA 是 Grouped-Query Attention，白话说就是“很多查询头共享更少的 KV 头”。它减少的是 $H_{kv}$，所以直接减少 KV Cache。若 `64 Q heads + 8 KV heads`，则 KV 规模相对标准多头注意力约降到 `1/8`。

| 结构 | Q heads | KV heads | KV Cache 相对规模 |
| --- | --- | --- | --- |
| MHA | 64 | 64 | 1 |
| GQA | 64 | 8 | 1/8 |
| MQA | 64 | 1 | 1/64 |

---

## 代码实现

先把估算逻辑写成最小可运行代码。这里用 `GB = 1024^3 bytes`。

```python
def estimate_memory(P, bw, L, H_kv, d_h, S, b_kv, batch_size=1):
    m_weight = P * bw
    m_kv_per_req = 2 * L * H_kv * d_h * S * b_kv
    m_kv_total = m_kv_per_req * batch_size
    return m_weight, m_kv_per_req, m_kv_total

def to_gb(x):
    return x / (1024 ** 3)

# Llama 3.1 70B 近似估算
P = 70_000_000_000
bw = 2          # FP16
L = 80
H_kv = 8
d_h = 128
S = 4096
b_kv = 2        # FP16/BF16 cache

m_weight, m_kv_4k, _ = estimate_memory(P, bw, L, H_kv, d_h, S, b_kv)
assert round(to_gb(m_weight), 1) > 130
assert 1.2 < to_gb(m_kv_4k) < 1.5

# W4 权重量化，按 0.5 byte / param 近似
m_weight_w4 = P * 0.5
assert 30 < to_gb(m_weight_w4) < 40

# 128K 上下文、8 并发
_, m_kv_128k_per_req, m_kv_128k_total = estimate_memory(P, bw, L, H_kv, d_h, 131072, 2, batch_size=8)
assert to_gb(m_kv_128k_per_req) > 40
assert to_gb(m_kv_128k_total) > 300

print("FP16 权重 GB:", round(to_gb(m_weight), 2))
print("4K KV GB:", round(to_gb(m_kv_4k), 2))
print("W4 权重 GB:", round(to_gb(m_weight_w4), 2))
print("128K 单请求 KV GB:", round(to_gb(m_kv_128k_per_req), 2))
print("128K, 8并发 KV 总 GB:", round(to_gb(m_kv_128k_total), 2))
```

示例结果可以整理成表：

| 模型/配置 | $P$ | $L$ | $H_{kv}$ | $d_h$ | $S$ | 权重显存 | KV 显存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 70B FP16 | 70B | 80 | 8 | 128 | 4K | 约 140GB | 约 1.34GB |
| 70B W4 | 70B | 80 | 8 | 128 | 4K | 约 35GB | 约 1.34GB |
| 70B W4 + KV INT8 | 70B | 80 | 8 | 128 | 128K | 约 35GB | 约 21.5GB/请求 |

真实工程里，配置思路一般不是“只开一个优化”，而是组合拳：

```bash
# 伪配置示意，不同框架参数名会不同
serve \
  --model meta-llama/Llama-3.1-70B \
  --weight-quant awq \
  --kv-cache-dtype fp8_or_int8 \
  --max-model-len 32768 \
  --enable-paged-attention
```

含义分别是：先压权重，让模型能装下；再限制 `max_model_len`，避免业务默认开到 128K；再压 KV Cache，支撑长上下文和并发；最后用分页管理减少碎片和浪费。

---

## 工程权衡与常见坑

显存优化不是“找一个最猛的量化开关”就结束，而是权重、KV、调度和内存管理一起看。

| 误区 | 后果 | 正确做法 |
| --- | --- | --- |
| 把 Q heads 当成 KV heads | KV 估算放大数倍 | 公式里明确写 $H_{kv}$ |
| 只算 4K，不按目标上下文重算 | 线上一开长上下文就爆 | 按真实 `max_model_len` 和并发估算 |
| 只做权重量化 | 模型能加载，但长对话仍 OOM | 同时评估 KV 优化 |
| 把 PagedAttention 当压缩算法 | 理论 KV 总量没变 | 认识到它主要解决碎片和分配效率 |
| 只看峰值显存，不测质量回归 | 线上答案质量下降 | 长文摘要、RAG、多轮对话都要回归 |

一个常见失败案例是：70B 模型用 W4 后权重从 `140GB` 降到 `35GB`，部署者以为问题解决了；结果业务一开 32K 或 128K 上下文，再叠加并发，KV Cache 立刻成为新瓶颈。也就是说，权重量化解决的是“常驻项太大”，不是“线性增长项失控”。

上线前至少要做这几类压测：短上下文、长上下文、高并发、重复多轮对话、多轮 RAG。因为

$$
M_{KV} \propto S \times batch
$$

只测单请求短文本，几乎一定低估线上压力。

---

## 替代方案与适用边界

不同方案解决的是不同问题，不能混成一个模糊的“显存优化工具箱”。

| 方案 | 主要作用 | 适合场景 | 主要边界 |
| --- | --- | --- | --- |
| 权重量化 | 降低常驻权重显存 | 模型装不下 | 不能解决长上下文 KV 增长 |
| KV 量化 | 降低线性增长项 | 长上下文、高并发 | 需要做质量回归 |
| PagedAttention | 提高 KV 分配与回收效率 | 在线服务、高吞吐 | 不改变理论 KV 总量 |
| GQA | 从结构上减少 $H_{kv}$ | 模型本身支持 GQA | 不是部署时随手能加的开关 |

可以用一个很短的决策树判断：

1. 模型根本装不下：先做权重量化。
2. 一开长上下文就爆：优先看 KV 量化、GQA、`max_model_len`。
3. 并发一高吞吐就掉：看 PagedAttention 和 KV 内存管理。
4. 对质量非常敏感：先试 FP8 或更保守的低风险方案，再评估更激进的低比特量化。

单卡离线推理通常先考虑 `W4/AWQ/GPTQ`；长上下文在线服务通常是“权重量化 + KV 量化 + PagedAttention”一起上；如果模型天然支持 GQA，那么它相当于从结构上先把 KV 的底座做小。

---

## 参考资料

| 资料 | 对应章节 |
| --- | --- |
| Llama 3.1 70B model card | 问题边界、真实工程例子 |
| GQA 论文 | 核心机制与推导 |
| AWQ 论文 | 代码实现、替代方案 |
| KIVI 论文 | 工程权衡、替代方案 |
| PagedAttention 论文 | 工程权衡、替代方案 |

1. [Llama 3.1 70B Model Card](https://huggingface.co/meta-llama/Llama-3.1-70B)
2. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://huggingface.co/papers/2305.13245)
3. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://huggingface.co/papers/2306.00978)
4. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://huggingface.co/papers/2402.02750)
5. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://huggingface.co/papers/2309.06180)
