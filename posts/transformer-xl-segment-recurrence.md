## 核心结论

Transformer-XL 可以用一句话定义：**段级递归 + 相对位置编码的自回归 Transformer**。

它解决的不是“单层注意力看不远”，而是“序列被切成固定段后，上下文会断掉”。普通 Transformer 处理第 $t$ 段时，通常只能看这一段内部的 token；上一段即使刚刚算过，也不会以结构化方式继续参与当前段注意力。Transformer-XL 的做法是把上一段各层的隐藏状态缓存成 `memory`，在处理下一段时直接拼进去复用。

可以先用一句白话理解“隐藏状态”：它不是原始文本，而是模型读完一个位置后形成的内部表示，相当于“已经编码过的上下文摘要”。Transformer-XL 缓存的就是这种摘要，而不是把原文再喂一遍。

总流程可以概括成：

`上一段 hidden states -> memory -> 当前段 attention -> 新 memory`

和普通 Transformer 的核心差异如下：

| 维度 | 普通 Transformer | Transformer-XL |
|---|---|---|
| 上下文范围 | 固定窗口 | 当前段 + 历史 memory |
| 历史复用 | 通常不复用 | 复用上一段隐藏状态 |
| 位置表示 | 常见做法是绝对位置 | 相对位置，更适合跨段 |
| 长程依赖 | 窗口外信息直接丢失 | 通过 memory 接续 |
| 推理效率 | 历史部分反复计算 | 历史表示可缓存 |

玩具例子先看最小版本。假设每段长度是 2，`mem_len=2`。上一段已经得到隐藏状态 `[1, 2]`，当前段是 `[3, 4]`。普通 Transformer 在当前段里只能看 `[3, 4]`；Transformer-XL 可以让当前段注意力看到 `[1, 2, 3, 4]`。这就是“不是每次都从零开始看文本，而是把上一页的关键信息继续拿来用”。

---

## 问题定义与边界

问题定义很具体：**当语言建模、代码补全、日志生成这类任务必须按固定长度分段处理时，如何让模型仍然保留跨段依赖，并且不破坏位置语义**。

这里的“分段”指把长序列切成多个连续片段逐段送入模型。原因通常不是算法偏好，而是显存和计算限制。普通 Transformer 的自注意力复杂度随序列长度增长很快，长度不能无限拉长，所以工程上必须截断。

普通做法会引出两个直接问题：

| 问题 | 普通 Transformer 的表现 | Transformer-XL 的处理 |
|---|---|---|
| 长程依赖 | 窗口外丢失 | 用 `memory` 接续 |
| 跨段一致性 | 段和段之间位置容易断裂 | 用相对位置保持一致 |
| 计算复用 | 历史部分重复算 | 复用历史 hidden states |

真实工程例子可以看代码补全。前 200 行定义了 `config`, `tokenizer`, `cache_dir`，后 200 行才真正调用。若模型只能看到最后一个窗口，它可能知道“这里需要一个路径变量”，却不知道具体变量名是什么。Transformer-XL 至少有机会通过前几段缓存的隐藏状态继续感知这些定义。

但它的边界也要说清楚：

- 它不是无限长上下文。`memory` 是截断的，只保留最近若干步。
- 它不是无限 memory。`mem_len` 增大会增加显存、延迟和噪声。
- 它不是双向编码器。它仍然是自回归模型，只能看当前位置之前的内容。
- 它不是缓存 logits。`logits` 是最终输出分数；Transformer-XL 缓存的是各层 `hidden states`。
- 它也不是所有长序列任务的统一最优解。若任务重点是“在超长文档里定位证据”，仅靠它通常不够。

一句白话总结边界：Transformer-XL 擅长“接着读、接着写”，不擅长把几万 token 当数据库一样精确检索。

---

## 核心机制与推导

Transformer-XL 的第一根支柱是**段级递归**。这里的“递归”不是编程语言里函数自己调自己，而是“上一段的表示进入下一段计算”。

对第 $\tau+1$ 段，在第 $n$ 层构造扩展上下文：

$$
\tilde h^{n-1}_{\tau+1} = [SG(h^{n-1}_\tau) \circ h^{n-1}_{\tau+1}]
$$

这里：

- $h^{n-1}_\tau$ 是上一段在第 $n-1$ 层的隐藏状态
- $h^{n-1}_{\tau+1}$ 是当前段在第 $n-1$ 层的隐藏状态
- $\circ$ 表示按时间维拼接
- `SG` 表示 stop-gradient，也就是“前向可见，反向不继续追到很久以前”

为什么要 `SG`？因为如果历史段也持续参与梯度回传，计算图会无限增长，训练会变得非常重，而且不稳定。Transformer-XL 的设计是：**历史可以提供信息，但不要求每一步梯度都穿透整个历史链条**。

第二根支柱是**相对位置编码**。这里的“相对位置”可以白话理解为“两个 token 之间隔了多远”，而不是“它在整篇序列中编号是多少”。

论文中的注意力打分可以写成：

$$
A_{i,j}=q_i^\top k_j + q_i^\top W_{k,R}R_{i-j} + u^\top k_j + v^\top W_{k,R}R_{i-j}
$$

这四项分别在做不同事情：

| 项 | 含义 | 作用 |
|---|---|---|
| $q_i^\top k_j$ | 内容相似度 | 看语义是否相关 |
| $q_i^\top W_{k,R}R_{i-j}$ | query 与相对位置交互 | 让距离影响注意力 |
| $u^\top k_j$ | 全局内容偏置 | 稳定内容选择 |
| $v^\top W_{k,R}R_{i-j}$ | 全局位置偏置 | 稳定距离判断 |

为什么 memory 和相对位置必须一起用？因为如果只把上一段 hidden states 拼进来，却继续使用绝对位置编码，那么模型会面临一个混乱问题：当前段里的“第 1 个 token”和上一段里的“第 1 个 token”绝对编号重复，跨段后位置含义不一致。相对位置把问题改写成“我和你相隔几步”，这样段号变化不会破坏时序关系。

可以把推导顺序理解成：

`memory 拼接 -> 生成 q/k/v -> 注入相对位置信息 -> 计算 score -> softmax -> 得到输出`

再看一个玩具例子。当前 token 是 `4`，它前面依次能看到 `3、2、1`，那么模型关心的是距离 `1/2/3`，而不是“`3` 在第几段、第几个绝对位置”。这就是“不是记住门牌号，而是记住前后距离”。

---

## 代码实现

实现层面最关键的对象通常叫 `mems`。它可以理解成“每一层各自保存的一段历史缓存”。注意不是只有顶层缓存，而是多层都有自己的 `memory`。

下面给出一个可运行的最小 Python 示例，只演示 memory 更新逻辑，不实现完整注意力。目的不是复现论文性能，而是把“拼接、截断、detach”这三个工程动作说清楚。

```python
from dataclasses import dataclass
from typing import List

@dataclass
class TensorLike:
    values: List[int]

    def detach(self):
        # 这里用返回拷贝模拟“停止梯度但保留数值”
        return TensorLike(self.values[:])

def concat_mem_and_cur(mem: TensorLike, cur: TensorLike) -> TensorLike:
    return TensorLike(mem.detach().values + cur.values)

def update_mems(old_mem: TensorLike, cur: TensorLike, mem_len: int) -> TensorLike:
    cat = old_mem.values + cur.values
    return TensorLike(cat[-mem_len:] if mem_len > 0 else [])

# 玩具例子
old_mem = TensorLike([1, 2])
cur = TensorLike([3, 4])

visible_context = concat_mem_and_cur(old_mem, cur)
new_mem = update_mems(old_mem, cur, mem_len=2)

assert visible_context.values == [1, 2, 3, 4]
assert new_mem.values == [3, 4]

# 再来一段
cur2 = TensorLike([5, 6])
visible_context2 = concat_mem_and_cur(new_mem, cur2)
new_mem2 = update_mems(new_mem, cur2, mem_len=2)

assert visible_context2.values == [3, 4, 5, 6]
assert new_mem2.values == [5, 6]
print("memory update works")
```

上面这个例子说明了两件事：

1. 当前段可见上下文会变长，因为历史缓存参与了前向。
2. 新 memory 不会无限增长，只保留最近 `mem_len` 个状态。

如果把它翻译成更接近模型代码的伪代码，大致是：

```python
# 1) 取上一段记忆
mems = get_mems()

# 2) 拼接记忆与当前段
h_cat = concat(detach(mems), h_cur)

# 3) 做相对位置注意力
out = attention_with_relative_position(h_cat)

# 4) 更新新的记忆，只保留最近 mem_len
new_mems = update_mems(mems, h_cur, mem_len)
```

张量流转关系通常如下：

| 变量 | 形状含义 | 作用 |
|---|---|---|
| `mems[l]` | 第 `l` 层历史状态 | 跨段缓存 |
| `h_cur` | 当前段 hidden states | 当前前向输入 |
| `h_cat` | 历史 + 当前拼接 | 注意力上下文 |
| `new_mems` | 截断后的新缓存 | 传给下一段 |

阅读实际实现时，建议按这个清单查：

- `mems` 是如何传进每一层的
- `update_mems` 如何裁剪到固定长度
- 相对位置项是在 attention score 的哪一步加入
- `SG` 在代码里通常对应 `detach` 或 `no_grad` 语义

真实工程例子是长日志生成。系统按固定 `tgt_len` 一段一段处理日志文本，保留最近 `mem_len` 的各层表示。这样当前生成位置还能感知前面已经出现过的 request id、service name、错误链路模式，而不是每段都重新猜一遍。

---

## 工程权衡与常见坑

Transformer-XL 的价值非常明确，但工程上绝不是“加个缓存就行”。

第一个权衡是 `mem_len`。它越大，可见历史越长，但代价也越高：

| 选择 | 好处 | 代价 |
|---|---|---|
| 增大 `mem_len` | 更长历史 | 更高显存与计算 |
| 使用相对位置 | 跨段更稳定 | 实现更复杂 |
| 只做缓存不改位置 | 开发简单 | 跨段结果容易错 |

为什么 `mem_len` 不能一味加大？因为 attention 看到的上下文更多，不代表有效信息一定更多。太旧的状态可能已经和当前任务无关，反而会分散注意力。白话说，不是便签越多越好，便签太多会让真正有用的提示被淹没。

第二个常见坑是**只缓存，不改位置编码**。这会导致跨段位置混乱。模型能“看到”旧状态，却不知道这些状态离当前 token 到底多远，注意力打分会缺少稳定的距离语义。

第三个坑是忘记 `detach`。如果 memory 既参与前向又保留完整反向链路，训练图会不断拉长，显存暴涨，训练速度下降，还可能出现梯度不稳定。

第四个坑是训练和推理分段方式不一致。比如训练时每段长度 128、memory 64，推理时却改成完全不同的切分和缓存策略，模型可能无法稳定复用历史，因为它学到的节奏和线上输入分布不同。

第五个坑是把 memory 误解为“全文可见”。不是。memory 只是过去某一截隐藏状态的压缩表示，而且长度有限。若任务需要精确回看很久以前的原句、字段、编号，它未必可靠。

工程上可用一个简化判断：

- 需要跨段连续生成，memory 很有价值
- 需要精确长距离事实回查，memory 往往不够
- 需要严格控制延迟，`mem_len` 必须做 profiling，而不是拍脑袋

---

## 替代方案与适用边界

Transformer-XL 适合的场景是：**任务天然按时间连续展开，而且上一段对下一段持续有帮助**。例如长文续写、代码补全、日志生成、持续对话。

如果任务目标变了，选型也会变：

| 方案 | 优势 | 适用场景 | 局限 |
|---|---|---|---|
| Transformer-XL | 跨段记忆、连续生成自然 | 续写、补全、生成 | 不是无限上下文 |
| Sliding Window Transformer | 机制简单 | 固定窗口任务 | 远距离依赖弱 |
| Longformer 类 | 面向长文档注意力 | 文档理解 | 更偏理解，不是同一路线 |
| 检索增强 | 能显式取回外部信息 | 知识密集任务 | 系统复杂度高 |

可以把选型规则压缩成三条：

- 需要连续生成：优先考虑 Transformer-XL 一类缓存式架构。
- 需要精确长文定位：优先考虑检索或稀疏注意力结构。
- 需要双向理解：优先考虑编码器类或专门的长文理解模型。

最后给一个真实边界判断。若你在做小说续写，前文角色关系、语气、未完成事件都需要延续，Transformer-XL 很自然。若你在做“从十万字技术手册中找某个参数定义”，它通常不如检索增强系统，因为后者能把相关原文直接取回，而不是指望 hidden states 长期记住所有事实。

---

## 参考资料

1. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (arXiv 摘要)](https://arxiv.org/abs/1901.02860)
2. [Transformer-XL 论文 PDF](https://arxiv.org/pdf/1901.02860.pdf)
3. [官方仓库 README](https://github.com/kimiyoung/transformer-xl)
4. [官方 PyTorch 实现 `mem_transformer.py`](https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py)
5. [官方仓库中的 PyTorch 目录](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch)
