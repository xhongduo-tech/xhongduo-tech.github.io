---
title: RoPE 代码精读：HuggingFace 实现逐行解析
date: 2026-08-07
---

# RoPE 代码精读：HuggingFace 实现逐行解析

<div class="epigraph">
<p>代码是唯一不撒谎的文档。</p>
<footer>—— 编程谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ HuggingFace Transformers 源码（modeling_llama.py） ｜ 2026-08-07</p>
</div>

## 为什么逐行读代码

上一节我们从数学上理解了 RoPE。但数学到代码之间还有一道「翻译」：如何把「每对旋转」「乘以 $e^{im\theta}$」写成 GPU 友好的张量运算？HuggingFace 的实现是社区公认的参考实现——LLaMA 系的 `LlamaRotaryEmbedding` 与 `apply_rotary_pos_emb` 被几乎所有开源仓库借鉴。<span class="marginnote">读代码的目的不是背 API，而是验证三条主线：频率矩阵怎么预计算、旋转怎么用「reshape 成两半」实现、前向里 cache 怎么管理位置。读懂了这几行，你对 RoPE 的理解就从「知道公式」升级为「能改代码」。</span>

## 1 频率矩阵：`inv_freq` 的预计算

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
```

三步拆解：

**`torch.arange(0, dim, 2)`**：生成 $0, 2, 4, \ldots$——只取偶数下标，对应「每对分量的第一个」。`dim` 是 `head_dim`（如 64），不是 `hidden_size`。
**`inv_freq`**：计算 $10000^{2i/d}$，再取倒数得 $\theta_i = 10000^{-2i/d}$——与公式完全一致。
**`register_buffer`**：把 `inv_freq` 注册为**非持久化 buffer**（不参与反向、不入权重文件），随设备移动。

这里的关键设计：`inv_freq` 是「预计算的频率表」，长度 `dim // 2`，一次算好存起来，之后每次前向复用。<span class="marginnote">注意 `dim` 是每个头的维度（head_dim），所以频率表按头内维度生成。为什么用头维度而不是整个 hidden？因为 RoPE 作用在 Q/K 的每个头上，头与头共享同一套频率——这也让旋转矩阵的实现与头数解耦。</span>

## 2 缓存位置角：`freqs`

```python
    @torch.no_grad()
    def forward(self, x, position_ids):
        # 外积：把「每个位置 × 每个频率」的角度全部算出来
        inv_freq_expanded = self.inv_freq[None, None, :, None].float()
        inv_freq_expanded = inv_freq_expanded.expand(
            position_ids.shape[0], 1, -1, 1)              # [bs, 1, dim//2, 1]
        position_ids_expanded = position_ids[:, None, None, :].float()  # [bs, 1, 1, seq_len]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [bs, seq_len, dim//2]

        # 复制一份拼起来：前半是 freqs，后半是同样的值（两半拼接技巧）
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

三步拆解：

**`freqs`**：这是**外积**——形状 `seq_len` × `dim // 2` 得到 `[bs, seq_len, dim//2]`，即「每个位置 × 每个频率」的角度矩阵 `freqs`。<span class="marginnote">用矩阵乘法实现外积是 PyTorch 的惯用技巧：`inv_freq` 与 `position_ids` 广播后就是所有配对。这里 `inv_freq` 展开成 `[bs, 1, dim//2, 1]`，`position_ids` 展开成 `[bs, 1, 1, seq_len]`，matmul 产生 `[bs, 1, dim//2, seq_len]`，转置后即 `freqs`——与「直接外积」的结果一致。</span>
- **`torch.cat((freqs, freqs), dim=-1)`**：把同一份角度复制一份拼起来，得到 `emb`——前半是 `freqs`，后半是同样的值。**为什么复制？** 这是「两半拼接」技巧的伏笔，下一步 `rotate_half` 要用。
- **`emb.cos()` / `emb.sin()`**：一次性算出所有位置的余弦与正弦缓存，`cos` 和 `sin` 都是逐元素对 `emb` 取三角函数得到的缓存（形状与 `emb` 相同）。

## 3 旋转的核心：`rotate_half` 与 `apply_rotary_pos_emb`

```python
def rotate_half(x):
    """后一半取负接前一半——等价于 90 度旋转。"""
    x1 = x[..., : x.shape[-1] // 2]      # 前半
    x2 = x[..., x.shape[-1] // 2 :]      # 后半
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

这里有一个巧妙的**等价变形**。回忆二维旋转公式：

$$
R(\phi) \begin{pmatrix} x_0 \\ x_1 \end{pmatrix} = \begin{pmatrix} x_0\cos\phi - x_1\sin\phi \\ x_0\sin\phi + x_1\cos\phi \end{pmatrix}
$$

HuggingFace 实现把「旋转」重新组织成：

$$
\text{Rot}(x) = x \cdot \cos + \text{rotate\_half}(x) \cdot \sin
$$

其中 `rotate_half(x)` 输出「后半取负接前半」的向量。<span class="marginnote">为什么能这么写？因为「拼接两份角度 + 前后各半」的技巧：把 `x` 的前半 $(x_0, x_2, \ldots)$ 与后半 $(x_1, x_3, \ldots)$ 看成每对的「实部」与「虚部」，则 $x\cos\phi + \text{rotate\_half}(x)\sin\phi$ 精确还原了每对的旋转——前半得到 $x_0\cos\phi - x_1\sin\phi$（旋转后的实部），后半得到 $x_0\sin\phi + x_1\cos\phi$（旋转后的虚部）。</span>

- **`rotate_half`** 不涉及任何三角函数，只是**拼接与取负**——$O(d)$ 的纯张量操作。
- **`q * cos`** 与 **`rotate_half(q) * sin`** 是逐元素乘——一次融合，GPU 上极快。

## 4 公式解析：旋转的「拼接实现」与数学等价

设 $x = [a_0, b_0, a_1, b_1, \ldots]$，其中第 $i$ 对是 $(a_i, b_i)$。HuggingFace 把向量视为「前半 $A = [a_0, a_1, \ldots]$，后半 $B = [b_0, b_1, \ldots]$」，于是：

$$
\text{Rot}(x) = [A\cos\phi - B\sin\phi,\; A\sin\phi + B\cos\phi]
$$

对这条式子做三步拆解：

- **第一步，对比标准旋转**：标准旋转对每对 $(a_i, b_i)$ 输出 $(a_i\cos\phi_i - b_i\sin\phi_i, a_i\sin\phi_i + b_i\cos\phi_i)$。上式逐元素看，前半第 $i$ 个元素正是标准旋转的第一个分量，后半正是第二个分量——**两半拼接与逐对旋转完全等价**。
- **第二步，读懂 `rotate_half`**：`rotate_half(x) = [-B, A]`（后半取负接前半）。代入：$\text{Rot}(x) = [A\cos\phi - B\sin\phi,\; A\sin\phi + B\cos\phi]$——正是上式。**`rotate_half` 就是「90 度旋转」的向量化实现。**
- **第三步，读出为什么快**：整个旋转 = 两次逐元素乘 + 一次拼接 + 一次取负，**没有任何三角函数调用**（三角函数只在缓存生成时算过一次）。这是把「每对旋转」重写为「两半的线性组合」带来的计算红利。

**辨析｜易错点：** 注意 `rotate_half` 的拼接顺序是 `(-x2, x1)`，不是 `(x1, -x2)`。这个顺序直接决定旋转方向（顺时针还是逆时针），也决定「相对位置」的符号约定。改错符号，模型的位置关系就全部反了——实现里最隐蔽的坑之一。

## 5 完整前向：Q/K 投影后立即旋转

在 `LlamaAttention.forward` 里：

```python
# ① 投影 → 拆头：Q/K/V 线性投影后按注意力头重排
q = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
k = self.k_proj(hidden_states).view(bsz, k_len, num_heads, head_dim).transpose(1, 2)

# ② 取位置角 → 旋转：位置 id 生成 cos/sin，立即旋转 Q/K
cos, sin = self.rotary_emb(q, position_ids)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

流程清晰：**投影 → 拆头 → 取位置角 → 旋转**。旋转后的 Q/K 再做 `q @ k.transpose(-2, -1)`（QKᵀ）得到注意力分数——此时分数的相对位置信息已由旋转注入。

**推理细节**：推理时 `position_ids` 是新 token 的位置，`self.rotary_emb` 只算新位置的角，KV Cache 里旧的 K/V 已经旋转过，直接复用——**旋转只作用在新 token 上，不重算历史**。这是 RoPE 与 KV Cache 天然兼容的关键。

## 6 小结

- **`inv_freq`**：预计算频率表 $\theta_i = 10000^{-2i/d}$，非持久化 buffer。
- **`freqs`**：用外积（`position_ids` × `inv_freq`）算角度，`cos` / `sin` 生成缓存。
- **`rotate_half`**：`(-x2, x1)` 的拼接取负 = 90 度旋转的向量化。
- **`apply_rotary_pos_emb`**：`x·cos + rotate_half(x)·sin`——与逐对旋转数学等价，且零三角函数调用。
- 前向位置：投影 → 拆头 → 旋转；推理只旋转新 token，与 KV Cache 天然兼容。

在下一节，我们看一个「零学习参数」的相对位置方案——**ALiBi**：用固定的线性偏置，让模型外推到训练长度之外的序列。
