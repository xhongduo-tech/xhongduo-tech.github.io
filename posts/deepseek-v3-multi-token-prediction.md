## 核心结论

DeepSeek-V3 的 Multi-Token Prediction，简称 MTP，意思是“一个位置不只学会预测下一个 token，还顺次预测后面更多 token”。它不是把多个未来 token 并行乱猜，而是保留因果链，按顺序往后推。公开实现和技术解读都指向同一个结构：在主模型输出之上再接 $D$ 个顺序模块，DeepSeek-V3 的公开工程生态通常按 `D=2` 来使用，也就是主模型预测 $t_{i+1}$，两个 MTP 模块继续预测 $t_{i+2}$ 和 $t_{i+3}$。

这件事的价值有两层。第一层是训练：传统 next-token prediction 每个位置只有一个监督目标，MTP 让同一个位置贡献多个监督信号，训练信号更密，模型更容易学到“提前规划”的表示。第二层是推理：训练完成后，额外 MTP 头可以直接丢弃，不影响标准生成；但如果推理引擎支持 speculative decoding，就能把这些“提前写好的草稿”拿来做验证复用。在 SGLang 的 DeepSeek-V3 文档里，官方给出的 H200 TP8 数据是 batch=1 约 1.8x，batch=32 约 1.5x。

| 场景 | 直接结果 |
| --- | --- |
| 只做传统 next-token 训练 | 每个位置 1 个监督目标 |
| 加入 MTP 训练 | 每个位置变成主目标 + 多个辅助目标 |
| 推理时丢弃 MTP 头 | 行为回到普通自回归模型 |
| 推理时启用 EAGLE 验证草稿 | 可能获得明显吞吐提升 |

---

## 问题定义与边界

先定义问题。token 是文本切分后的最小离散单元，可以粗看成词片。自回归语言模型的基本任务是：给定前文 $t_1,\dots,t_i$，预测下一个 token $t_{i+1}$。这个目标简单、稳定，但监督很稀疏，因为每个位置只学一步。

MTP 解决的不是“模型不会生成”，而是“训练信号太单薄”。它把问题改写成：

给定同一个位置的隐藏状态，除了预测 $t_{i+1}$，还继续预测 $t_{i+2}, t_{i+3}, \dots, t_{i+D}$。

边界也要讲清楚：

| 模块 | 训练时目标 | 推理默认是否保留 |
| --- | --- | --- |
| 主模型 | 预测下一个 token | 保留 |
| MTP-1 | 预测后第 2 个 token | 默认不保留 |
| MTP-2 | 预测后第 3 个 token | 默认不保留 |

这说明 MTP 首先是训练目标增强机制，不是默认常驻的推理结构。很多初学者会把它理解成“DeepSeek-V3 一次生成三个 token”。这个说法不准确。更准确的说法是：训练时它学会提前规划多个未来 token；推理时默认仍然逐 token 决策，只是在支持 speculative decoding 的引擎里，MTP 输出可以作为草稿候选复用。

玩具例子很直观。假设当前已经看到：

`t1=today, t2=is, t3=a`

标准训练只要求模型从当前位置预测 `t4=good`。  
MTP 则进一步要求：

- 主模型：预测 `good`
- MTP-1：预测 `day`
- MTP-2：预测 `.`

于是，同一个上下文不再只回答“下一个是什么”，还要回答“再下一个是什么”和“再下下个是什么”。

---

## 核心机制与推导

DeepSeek 风格的 MTP 关键不在“多几个 loss”，而在“顺序保持因果链”。因果链的白话解释是：后面的预测要建立在前面预测深度形成的表示之上，而不是几个互不相干的并行分类头。

NVIDIA Megatron Core 对 DeepSeek-V3 风格 MTP 的描述比较清楚：第 $k$ 个 MTP 模块由共享 embedding、投影矩阵、一个 Transformer block、共享输出头组成。这里的共享 embedding，意思是额外模块复用主模型的词向量表，不单独再养一套输入表示参数。共享输出头，意思是最后映射回词表分布时也复用主模型的输出投影。

如果记主模型在位置 $i$ 的表示为 $\mathbf h_i^{0}$，那么第 $k$ 层 MTP 会把上一步表示和第 $(i+k)$ 个 token 的 embedding 结合，再送入一个 Transformer block：

$$
\mathbf h_i^{\prime k}=M_k[\mathrm{RMSNorm}(\mathbf h_i^{k-1});\mathrm{RMSNorm}(\mathrm{Emb}(t_{i+k}))]
$$

$$
\mathbf h_i^{k}=\mathrm{TRM}_k(\mathbf h_i^{\prime k})
$$

最后得到第 $k$ 个未来 token 的预测分布。总损失通常写成：

$$
L_{\text{total}} = L_{\text{main}} + \lambda \cdot \frac{1}{D}\sum_{k=1}^{D}L_{\text{MTP}}^{(k)}
$$

其中 $\lambda$ 是缩放系数，白话讲就是“辅助任务别抢主任务的权重”。Megatron Bridge 文档里给的默认值是 `0.1`，常见调节范围是 `0.05-0.2`。

数值上看更容易理解。设：

- 主损失 $L_{\text{main}} = 2.0$
- 第一个 MTP 损失 $L^{(1)}_{\text{MTP}} = 3.0$
- 第二个 MTP 损失 $L^{(2)}_{\text{MTP}} = 3.5$
- $\lambda = 0.1$

那么：

$$
L_{\text{total}} = 2.0 + 0.1 \cdot \frac{3.0+3.5}{2}=2.325
$$

重点不是把辅助损失做得和主损失一样大，而是用较小权重把未来 token 的监督信号加进来。这样训练梯度更密，但主目标仍然是 next-token prediction。

真实工程例子可以看 SGLang。DeepSeek-V3 的 MTP 训练结果，在推理端被接到 EAGLE 验证流程里。草稿 token 如果高概率被主模型接受，就能减少逐 token 的同步等待，从而提升 TPS。这里“接受率”就是草稿通过验证的比例。接受率越高，加速越明显；接受率低，验证开销会吃掉收益。

---

## 代码实现

下面先用一个可运行的 Python 玩具程序，把损失计算和草稿接受收益算清楚。

```python
def mtp_total_loss(main_loss, mtp_losses, scale):
    avg_mtp = sum(mtp_losses) / len(mtp_losses)
    return main_loss + scale * avg_mtp

def expected_verified_tokens(main_tokens, draft_tokens, acceptance_rate):
    # 一个极简估算：主路径先产出 main_tokens，
    # 草稿路径期望复用 acceptance_rate * draft_tokens
    return main_tokens + acceptance_rate * draft_tokens

loss = mtp_total_loss(2.0, [3.0, 3.5], 0.1)
assert abs(loss - 2.325) < 1e-9

tokens = expected_verified_tokens(main_tokens=1, draft_tokens=2, acceptance_rate=0.9)
assert abs(tokens - 2.8) < 1e-9

low_accept = expected_verified_tokens(main_tokens=1, draft_tokens=2, acceptance_rate=0.6)
assert low_accept < tokens
print(loss, tokens, low_accept)
```

这个例子没有实现 Transformer，只是把两个工程事实算出来了：

1. MTP loss 是辅助项，不应压过主损失。
2. 草稿接受率下降时，理论收益会明显变差。

如果你要在真实服务里复现 DeepSeek-V3 的 MTP 推理收益，SGLang 的最小配置是：

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --trust-remote-code \
  --tp 8
```

这里几个参数的含义如下：

| 参数 | 作用 |
| --- | --- |
| `--speculative-algorithm EAGLE` | 打开基于验证的草稿机制 |
| `--speculative-num-steps` | 草稿自回归展开深度 |
| `--speculative-eagle-topk` | 每步扩展时保留的候选分支数 |
| `--speculative-num-draft-tokens` | 最终提交验证的草稿 token 数 |

对初学者最重要的理解是：这不是“另起一个小模型做 draft”，而是复用 DeepSeek-V3 已经训练出的多 token 预测能力。部署上更简单，但收益仍然取决于接受率和 batch 设置。

---

## 工程权衡与常见坑

第一个坑是把 MTP 当成“免费提升”。不是。它会增加训练开销和显存开销。Megatron 文档明确提到，MTP 层的训练时间大致接近普通 Transformer 层，`mtp_num_layers` 增加时，内存和训练时间会近似按比例上涨。

第二个坑是 $\lambda$ 设太大。辅助损失过强时，会出现主损失被盖住的现象。训练日志上常见表现是主 `lm loss` 下降变慢，甚至 MTP loss 在拉着主任务跑。经验上可以这样判断：

| 现象 | 更可能的问题 |
| --- | --- |
| MTP loss 明显不降 | $\lambda$ 太小，辅助监督太弱 |
| 主 loss 被拖慢 | $\lambda$ 太大，辅助目标喧宾夺主 |
| 显存突然顶满 | `mtp_num_layers` 太多 |
| 推理没加速 | 草稿接受率低，或根本没启 speculative |

第三个坑是把训练边界和推理边界混在一起。训练时 MTP 需要未来 token 的真实 embedding 参与顺序构造；推理时没有真实未来 token，因此必须通过草稿生成加验证的方式使用。也就是说，训练中的“teacher-forced MTP”和推理中的“speculative verification”不是同一件事，只是前者为后者提供了可复用的能力基础。

第四个坑出现在大 batch。SGLang 文档明确写了，大 batch 大于 48 时，需要额外调整 `--max-running-requests` 和 `--cuda-graph-bs`。否则你会遇到两种表面症状：不是速度没上去，就是调度不稳定。这个问题和模型好坏无关，纯粹是服务端调度参数没对齐。

---

## 替代方案与适用边界

MTP 不是唯一的加速方案。更常见的替代路线是外部 draft model，也就是单独准备一个更小、更快的模型先写草稿，再由主模型验证。SGLang 的通用 speculative decoding 文档主要讲的就是这一路线，比如 EAGLE-2、EAGLE-3。

两类方案可以这样比较：

| 方案 | 特点 | 更适合什么情况 |
| --- | --- | --- |
| 内置 MTP / NextN | 草稿能力来自主模型训练目标，无需额外 draft 模型 | 已有 DeepSeek-V3 权重，想降低部署复杂度 |
| 外部 draft model | 需要两套模型协同，但可独立替换草稿模型 | 已有成熟主模型，希望单独优化推理链路 |

适用边界也很明确。

如果你的目标是“提高预训练信号密度”，MTP 很合适，因为它本质是训练目标增强。  
如果你的目标是“马上给任何模型提速”，MTP 不一定合适，因为前提是模型在训练阶段就为多 token 草稿做过准备。  
如果你的业务场景极度强调确定性和稳定尾延迟，大 batch 下甚至可能主动关闭 speculative decoding，只保留普通 next-token 生成。因为一旦验证失败频繁，复杂调度本身也会成为成本。

一句话总结边界：MTP 最适合“在训练阶段就为未来 token 规划能力买单，并在推理阶段按条件回收收益”的模型体系，不适合把它当成脱离训练上下文的通用加速插件。

---

## 参考资料

- DeepSeek-V3 官方仓库与技术报告：说明 DeepSeek-V3 引入 MTP 训练目标，并给出主模型权重与 MTP 模块权重划分  
  https://github.com/deepseek-ai/DeepSeek-V3

- NVIDIA Megatron Bridge: Multi-Token Prediction  
  说明 `mtp_num_layers`、`mtp_loss_scaling_factor`、总损失组合方式与调参建议  
  https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/multi-token-prediction.html

- NVIDIA Megatron Core: Multi-Token Prediction  
  说明 DeepSeek-V3 风格 MTP 采用顺序模块、共享 embedding、共享 output head  
  https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/multi_token_prediction.html

- SGLang: DeepSeek V3/V3.1/R1 Usage  
  说明 DeepSeek-V3 在 SGLang 中启用 EAGLE 后的最小配置、默认配置、以及 batch=1 和 batch=32 的加速数据  
  https://docs.sglang.io/basic_usage/deepseek_v3.html

- SGLang: Speculative Decoding  
  说明通用 EAGLE speculative decoding 的工作方式与外部 draft model 路线  
  https://docs.sglang.io/advanced_features/speculative_decoding.html

- VITALab 对 DeepSeek-V3 的技术解读  
  提到 MTP 顺序保持因果链，并给出第二个 token 草稿接受率约 85% 到 90% 的分析信息  
  https://vitalab.github.io/article/2025/02/11/DeepSeekV3.html
