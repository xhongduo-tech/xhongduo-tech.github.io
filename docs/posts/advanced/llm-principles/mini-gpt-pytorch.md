---
title: 动手实现：用 PyTorch 从零写一个 mini-GPT
date: 2026-08-07
---

# 动手实现：用 PyTorch 从零写一个 mini-GPT

<div class="epigraph">
<p>能写出来的，才是真正理解的。</p>
<footer>—— 学习箴言（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ 《Attention Is All You Need》 / Karpathy《nanoGPT》 ｜ 2026-08-07</p>
</div>

## 为什么动手写一遍

前面十一篇把 GPT 的每个组件都拆开了——嵌入、注意力、掩码、FFN、归一化、输出头。这一篇把它们**组装成一个能跑、能训练、能生成的 mini-GPT**。代码约 200 行，覆盖全部核心组件；写完你就拥有了「从 token 到文本」的完整闭环。**这一篇是第三篇《GPT 架构逐层解析》的收官之作**——前面的知识在这里全部落地。<span class="marginnote">Karpathy 的 nanoGPT 是这篇的蓝本（约 300 行）。这里我们用更「教学」的写法：每个组件一个类、注释讲清每个张量的形状。目标是「读懂每一行、跑通每一步」——当你真正写出 attention 的 reshape 与 mask 时，之前所有「为什么」都有了答案。</span>

## 1 骨架：配置与整体类

先定义配置与模型骨架：

```python
class GPTConfig:
    block_size: int = 1024      # 上下文长度
    vocab_size: int = 50257     # GPT-2 词表大小
    n_layer: int = 12           # Transformer 层数
    n_head: int = 12            # 注意力头数
    n_embd: int = 768           # 嵌入维度（宽度 d）


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # 可学习位置嵌入
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**注意**：GPT-2 用的是「可学习位置嵌入」（`nn.Embedding` 查表）+ **Pre-LN**——与我们前面讲的 LLaMA 系（RoPE + RMSNorm）不同。这里跟随 GPT-2/nanoGPT 的经典写法，便于对照论文。<span class="marginnote">模型用的是「权重共享」：`lm_head.weight = tok_emb.weight`（输入嵌入与输出头共享同一张权重）。nanoGPT 里通过直接赋值实现。这省了 $V \cdot d$ 参数——我们在《词嵌入层》里讲过它的语义一致性。</span>

## 2 核心组件：Block、CausalSelfAttention、MLP

三个核心类：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # 一次投影出 QKV
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)      # [B,T,3C] → 3×[B,T,C]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (C // self.n_head) ** 0.5
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                     # [B, H, T, d_k] 加权求和
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并头
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Pre-LN：注意力分支先归一化
        x = x + self.mlp(self.ln_2(x))    # Pre-LN：MLP 分支先归一化
        return x
```

**逐行要点**：

- `c_attn` 一次投影出 QKV（`[B, T, 3C]`），再 `split` 成三份——**用一个大线性层替代三个小线性层**，GPU 更高效。
- `view` 拆头：`[B, T, C]` → `[B, H, T, d_k]`——**这是多头注意力的张量落地**。
- `tril` 掩码：把上三角填成 `-inf`——**因果掩码**（softmax 前）。
- `softmax` 加权求和。`transpose → view` 合并头。

## 3 前向与因果掩码

因果掩码是**注册的 buffer**（不参与梯度）：

```python
mask = torch.tril(torch.ones(config.block_size, config.block_size))
self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
```

`tril`（下三角）在 `(i, j)` 处为 1 当且仅当 $j \le i$——正好是「只看过去」。`masked_fill` 把「看未来」（$j > i$）的位置填成 `-inf`，softmax 后权重为 0。

**整个模型的前向**（含损失）：

```python
def forward(self, idx, targets=None):
    B, T = idx.size()
    pos = torch.arange(T, device=idx.device)
    x = self.tok_emb(idx) + self.pos_emb(pos)      # 词嵌入 + 位置嵌入
    x = self.blocks(x)                              # 逐层过 Block
    x = self.ln_f(x)                                # 最终 LayerNorm
    logits = self.lm_head(x)                        # [B, T, V] 输出头
    if targets is None:
        return logits, None
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
```

- `view(-1, vocab_size)` 把 `logits` 摊平成 `[B*T, V]`，与 targets 摊平——**每个位置都算损失**（CLM 的「每位置监督」）。

## 4 公式解析：多头注意力的形状账本

把注意力的张量形状走一遍（$B$=batch, $T$=序列, $C$=宽度, $H$=头数, $d_k=C/H$）：

$$
\underbrace{[B, T, C]}_{\text{输入}} \xrightarrow{c\_attn} [B, T, 3C] \xrightarrow{split} 3 \times [B, T, C] \xrightarrow{view/transpose} [B, H, T, d_k]
$$

$$
\underbrace{[B, H, T, d_k]}_{Q} \cdot \underbrace{[B, H, d_k, T]}_{K^{\top}} \to [B, H, T, T] \xrightarrow{\text{mask}+\text{softmax}} \xrightarrow{\cdot V} [B, H, T, d_k]
$$

对这套式子做三步拆解：

- **第一步，读懂「一次投影」**：`c_attn` 用 $3C \times C$ 的权重一次算出 QKV——比「三个独立线性层」省内存带宽，且 PyTorch 更高效。
- **第二步，读懂「拆头」**：`view` 把最后的 $C$ 拆成 $H \times d_k$，`transpose` 把「头」挪到第二维——**多头并行**（各头独立计算）。$d_k = C/H$，$H \cdot d_k = C$——**形状守恒**。
- **第三步，读懂「合并头」**：`transpose → view`（`c_proj`）把头合并回宽度——**多头输出拼接**。这就是「$h$ 个头各自算、最后拼起来」的实现。

**辨析｜易错点：** mask 必须在 **softmax 之前**（填 `-inf` 后 softmax 权重为 0），如果先 softmax 再 mask 就错了。另外 `contiguous()` 不能省——`transpose` 后张量是非连续内存的，`view` 要求连续，必须 `contiguous()` 再 `view`。**这两个「顺序」与「连续性」是新手最容易踩的坑**。

## 5 训练与生成

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.config.block_size:]       # 只看最后 block_size 个 token
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature           # 只取最后一个位置的分数
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样
        idx = torch.cat((idx, idx_next), dim=1)           # 采样结果追加回序列
    return idx
```

- 生成用 `torch.no_grad()`——推理不需要梯度。
- 每步只看最后 `block_size` 个 token（上下文截断）。
- `multinomial` 按概率采样——**不是贪心**，保留了多样性。

**训练**：普通 PyTorch 训练循环——`forward` → `backward` → `optimizer.step()`，对每个 batch 算 loss、反向、更新。在 nanoGPT 的 toy 数据（莎士比亚）上跑 5000 步，就能生成「有模有样」的莎士比亚风格文本——**你亲手训练的「大模型」**。

## 6 小结

- 一个 mini-GPT = **词嵌入 + 位置嵌入 + N 层（Pre-LN 注意力 + MLP）+ 最终 LN + 输出头**。
- 关键实现细节：**一次投影 QKV、view/transpose 拆头、tril 掩码、contiguous 合并头**。
- 前向返回 `logits`（推理用）与 `loss`（训练用）。
- 生成是「**逐 token 采样 + 追加**」的自回归循环。
- 形状守恒贯穿始终：主干保持 `[B, T, C]`，只有输出头到 `[B, T, V]`。

到这里，第三篇《GPT 架构逐层解析》全部完成。下一篇进入**主流开源架构对比**——从 LLaMA 的「三件套」讲起。
