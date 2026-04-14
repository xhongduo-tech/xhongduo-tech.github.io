## 核心结论

RWKV 是一种把 Transformer 训练方式和 RNN 推理方式拼接到同一套参数里的语言模型架构。白话说，训练时它像 Transformer 一样能把整段序列并行送进 GPU；生成时它像 RNN 一样只维护一份状态，来一个 token 更新一次，不需要像标准 Transformer 那样持续扩张 KV cache。

它的关键不是“把注意力删掉”这么简单，而是把“时间依赖”和“通道变换”拆成两个交替模块：

- `time-mixing`：处理当前 token 与历史状态怎么融合
- `channel-mixing`：处理每个位置内部，各个特征维之间怎么非线性变换

这带来一个很重要的工程性质：训练和推理可以用两种等价视角实现。训练阶段把整段对话当成并行序列计算；推理阶段只在每个新 token 上更新隐藏态，显存从依赖上下文长度的 $O(N)$ 降到近似常数级 $O(1)$（更准确地说，是与上下文长度无关，只与层数和隐藏维度有关）。

| 架构 | 训练时序列处理 | 推理时是否保存整段历史 | 推理显存随上下文增长 | 长上下文部署特点 |
|---|---|---|---|---|
| 标准 Transformer | 并行 | 是，依赖 KV cache | 增长 | 上下文越长越吃显存 |
| RWKV | 并行 | 否，只保留状态 | 基本不增长 | 更适合流式生成和长上下文 |
| 传统 RNN/LSTM | 串行 | 否，只保留状态 | 基本不增长 | 推理省显存，但训练难并行 |

一个最直观的理解方式是：Transformer 把“历史所有 token”显式放在缓存里；RWKV 把“历史影响”压缩进状态里。

---

## 问题定义与边界

RWKV 要解决的问题很明确：大型语言模型训练需要并行，推理又希望省显存，这两件事在传统架构里通常冲突。

- Transformer 的优势是训练并行，适合现代 GPU。
- RNN 的优势是推理状态固定，适合流式和低显存部署。
- RWKV 的目标是同时拿到这两种优势。

这里的“并行”指训练时同一批 token 可以同时算；“常数显存”指推理时显存不随上下文长度线性增加。它不是说模型完全没有状态，而是说状态大小固定。

RWKV 的边界也要说清楚：

1. 它不是标准 self-attention 的直接等价替身。
2. 不同版本的 RWKV 细节有差异，尤其 v4、v5、v6、v7 在 time-mix 和状态递推上不断演化。
3. 面向初学者时，通常先用简化公式理解“当前输入 + 上一步状态 + 门控 + 衰减”，再看论文中的并行化推导。

一个简化的 time-mix 写法是：

$$
x_t^{(r)}=\alpha_r \odot x_t + (1-\alpha_r)\odot x_{t-1}
$$

$$
x_t^{(k)}=\alpha_k \odot x_t + (1-\alpha_k)\odot x_{t-1}
$$

$$
x_t^{(v)}=\alpha_v \odot x_t + (1-\alpha_v)\odot x_{t-1}
$$

然后分别生成门控、键和值：

$$
r_t=\sigma(W_r x_t^{(r)}),\quad
k_t=W_k x_t^{(k)},\quad
v_t=W_v x_t^{(v)}
$$

其中“门控”就是一个 0 到 1 之间的开关，决定当前位置应该放行多少信息。

如果把最简单的混合理解成 $\alpha=\beta=0.5$，那就相当于“当前 token 和前一 token 各占一半”。这正是 RWKV 能在推理时只靠状态推进的直觉基础：新 token 不是重新看全历史，而是接着上一时刻的压缩结果继续算。

---

## 核心机制与推导

先看最核心的两块。

### 1. Time-Mixing：把时间依赖改写成可递推形式

“递推”就是下一步只依赖上一步，不需要回头重看全部历史。RWKV 的 time-mixing 用的是“当前输入 + 历史残留”的结构。直觉上，它想模拟“我既看当前 token，也带一点过去的信息进来”。

用一个简化版玩具公式表示：

$$
r_t=\sigma\left(W_r(\alpha x_t+(1-\alpha)x_{t-1})\right)
$$

$$
k_t=\phi\left(W_k(\beta x_t+(1-\beta)x_{t-1})\right)
$$

$$
v_t=W_v(\gamma x_t+(1-\gamma)x_{t-1})
$$

这里：

- `receptance`：接收门，白话说是“这次该不该把信息放进来”
- `key`：特征索引，白话说是“这条信息是什么类型”
- `value`：信息内容，白话说是“真正要传递的数值”
- $\phi$：常见可取 `ReLU`、`square(ReLU)` 等，版本不同实现不同

再往前一步，RWKV 真正重要的是它会维护某种带时间衰减的累计量。把历史记忆写成一个递推状态 $s_t$，可抽象成：

$$
s_t = e^{-w}\odot s_{t-1} + k_t \odot v_t
$$

$$
o_t = r_t \odot \frac{s_t}{z_t+\varepsilon}
$$

其中 $e^{-w}$ 是可学习衰减，白话说就是“旧信息每过一步要保留多少”；$z_t$ 是归一化项，目的是防止数值无限变大。论文里的并行化推导，本质上就是把这种递推写成前缀扫描友好的形式，这样训练时就能并行。

### 2. Channel-Mixing：把通道维的非线性变换单独拿出来

“通道”就是隐藏向量里的每一维特征。channel-mixing 本质上类似 Transformer 里的 FFN，只不过它也常带一点 token shift。

“Token Shift” 的白话解释是：把上一 token 的一部分特征挪过来，和当前 token 拼着用。这样模型不用完整注意力，也能获得非常近邻的局部时序信息。

一个简化表达是：

$$
\tilde{x}_t=\eta \odot x_t + (1-\eta)\odot x_{t-1}
$$

$$
h_t=\mathrm{ReLU}(W_1\tilde{x}_t)^2
$$

$$
c_t=\sigma(W_g\tilde{x}_t)\odot W_2 h_t
$$

这里 $c_t$ 就是 channel-mixing 的输出。它解决的是“当前位置内部的特征组合”，不是“跨位置聚合历史”。

### 3. 玩具例子：2 维向量看门控怎么工作

设：

$$
x_{t-1}=[0.2,0.1],\quad x_t=[0.4,0.3],\quad \alpha=0.5,\quad W_r=I
$$

则 time-mix 后的输入是：

$$
0.5x_t+0.5x_{t-1}=[0.3,0.2]
$$

经过 sigmoid：

$$
r_t=\sigma([0.3,0.2])\approx[0.574,0.550]
$$

这表示第一维大约放行 57.4%，第二维放行 55.0%。结论很直接：RWKV 的门控不是“全收”或“全拒”，而是按维度细粒度地决定历史和当前信息如何进入输出。

### 4. 真实工程例子：长对话聊天系统

假设你在本地做一个长期记忆聊天机器人：

- 用 Transformer 时，你要保留所有历史 token 的 KV cache。
- 对话越长，显存越吃紧，速度通常也会掉。
- 用 RWKV 时，你只需要把每层状态继续传下去。

所以训练时可以像普通大模型那样并行吃 batch，部署时却更像一个流式状态机。这就是它最有工程价值的地方。

---

## 代码实现

下面给一个可运行的极简 Python 版本。它不是论文原始实现，而是帮助理解“推理时只保存状态”的骨架。

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class TinyRWKVCell:
    def __init__(self, alpha=0.5, decay=0.8):
        self.alpha = alpha
        self.decay = decay
        self.prev_x = 0.0
        self.state = 0.0

    def step(self, x_t):
        # time-mix: 当前输入和上一输入做线性混合
        x_mix = self.alpha * x_t + (1 - self.alpha) * self.prev_x

        # receptance gate: 决定当前信息放行多少
        r_t = sigmoid(x_mix)

        # 简化的 key / value
        k_t = max(0.0, x_mix)      # ReLU
        v_t = math.log1p(math.exp(x_t))  # softplus

        # 递推状态：旧状态衰减 + 当前贡献
        self.state = self.decay * self.state + k_t * v_t

        # 输出
        out = r_t * self.state

        # 更新上一步输入
        self.prev_x = x_t
        return out

# 玩具序列
cell = TinyRWKVCell(alpha=0.5, decay=0.8)
seq = [0.2, 0.4, -0.1, 0.3]
outs = [cell.step(x) for x in seq]

# 基本正确性检查
assert len(outs) == 4
assert outs[0] >= 0.0
assert outs[1] != outs[0]

# 关键性质：继续推理只需要 cell 内部状态，不需要完整历史
saved_prev_x = cell.prev_x
saved_state = cell.state

next_out = cell.step(0.5)

restored = TinyRWKVCell(alpha=0.5, decay=0.8)
restored.prev_x = saved_prev_x
restored.state = saved_state
next_out_restored = restored.step(0.5)

assert abs(next_out - next_out_restored) < 1e-12
print("OK", outs, next_out)
```

这个例子体现了三件事：

1. 每次只处理一个新输入。
2. 历史被压缩在 `prev_x` 和 `state` 里。
3. 只要状态连续，后续输出就可复现。

如果写成更接近工程伪代码的形式，大致是：

```python
state = prev_state
xmix = time_mix(x_t, state.prev_x)
r = sigmoid(W_r @ xmix)
k = relu(W_k @ xmix)
v = softplus(W_v @ xmix)
state.mem = decay * state.mem + k * v
out = r * normalize(state.mem)
out = channel_mix(out, state.prev_x)
state.prev_x = x_t
```

训练时为什么还能并行？因为 RWKV 会把这种递推形式改写成适合批量计算的表达，让整段序列能在 GPU 上一起算；但推理时你仍然可以退回逐 token 更新状态的 RNN 视角。

---

## 工程权衡与常见坑

RWKV 的优点很清楚，但它不是“所有场景都更好”。

### 优势

- 不依赖 KV cache，长上下文推理更稳定
- 更适合流式、在线、边缘设备部署
- 推理显存和上下文长度解耦
- 训练仍保留并行化能力

### 代价

- 状态递推对数值稳定性要求高
- 不同版本实现差异较大，迁移时不能只看名字
- 历史被压缩进固定状态后，精确回忆远处细节不一定总比显式缓存更强
- 内核优化、生态支持通常不如主流 Transformer 完整

常见坑如下：

| 常见坑 | 现象 | 原因 | 对策 |
|---|---|---|---|
| time-mix 参数不稳 | 训练损失抖动、发散 | 历史与当前混合比例失衡 | 用成熟初始化，沿用官方实现 |
| 衰减项数值爆炸/下溢 | 输出出现 `inf` 或几乎全 0 | 指数衰减直接算不稳定 | 用 log-space 或稳定化技巧 |
| 缺少规范化层 | 深层训练不稳 | 状态累计后尺度漂移 | 配合 RMSNorm / LayerNorm |
| 推理时重置状态 | 模型“忘记前文” | state 没续上 | 序列切换时显式保存和恢复 state |
| 并行训练和逐步推理结果不一致 | 线上线下表现偏移 | 实现细节不一致 | 保证训练版和推理版公式严格对齐 |

一个真实工程坑很典型：你做聊天服务时，如果每轮请求都只把“最后一句用户输入”丢进模型，却没把上一轮状态带上，那么 RWKV 会像被清空记忆一样重新开始。正确做法是把每个会话对应的 state 持久化，至少包括前一输入相关缓存和递推记忆量。

另一个坑是过度宣传“无限上下文”。更准确的说法应该是：RWKV 的推理内存不限制上下文长度，但模型是否还能精确利用很远的历史，取决于训练、衰减机制和任务本身。内存可扩展，不等于信息可无损保存。

---

## 替代方案与适用边界

如果你只是做标准离线训练和常规长度推理，Transformer 仍然是默认选择。因为生态最好，工具最多，推理框架最成熟。

但如果你更看重下面这些场景，RWKV 会更有吸引力：

- 本地低显存部署
- 长对话、长文档流式处理
- 需要会话状态持久化
- 生成时不希望 KV cache 持续膨胀

下面给一个横向对比：

| 方案 | 训练并行性 | 推理显存随上下文增长 | 长距离显式记忆 | 工程生态 | 适用场景 |
|---|---|---|---|---|---|
| Transformer | 强 | 是 | 强，KV cache 显式保留 | 最成熟 | 通用大模型训练与推理 |
| RWKV | 强 | 否 | 中等到强，依赖状态表达 | 仍在发展 | 流式生成、边缘部署、长上下文 |
| Linear Transformer | 强 | 通常较低 | 依具体核函数而定 | 中等 | 追求低复杂度的变体研究 |
| 传统 RNN/LSTM | 弱 | 否 | 容易遗忘长依赖 | 成熟但不适合超大 LLM | 小模型、时序任务 |

一个真实工程例子是百万 token 级日志流处理。标准 Transformer 即使有 FlashAttention，仍然绕不开 KV cache 在生成阶段持续增长的问题；RWKV 的状态大小固定，更像一个长期运行的流处理器。但如果你的任务要求“精确找回 5 万 token 前某个具体变量名”，显式缓存的 Transformer 可能更直接。

所以结论不是“RWKV 替代 Transformer”，而是“RWKV 在训练并行 + 推理常数状态这个交叉点上很有价值”。

---

## 参考资料

1. Peng et al.，《[RWKV: Reinventing RNNs for the Transformer Era](https://aclanthology.org/2023.findings-emnlp.936/)》，Findings of EMNLP 2023。核心论文，给出 RWKV 的架构定义、并行训练形式与实验结果。
2. BlinkDL，《[RWKV-LM GitHub Repository](https://github.com/BlinkDL/RWKV-LM)》，持续更新。官方代码与工程说明，适合核对训练、推理、初始化和版本差异。
3. RWKV Community，《[RWKV Language Model Wiki](https://wiki.rwkv.com/)》，截至 2026-04 可访问。社区维护入口，适合先建立整体认知，再顺着实现看细节。
4. Vife AI，《[RWKV Explained: The Linear RNN Revolutionizing AI Architecture](https://vife.ai/blog/rwkv-explained-linear-rnn-architecture)》，2026-01-08。偏入门的解释材料，适合先理解“为什么训练像 Transformer、推理像 RNN”。
5. Ankush Choudhary，《[RWKV v6 Time-Mix: Receptance Weighted Key Value RNN 2025](https://johal.in/rwkv-v6-time-mix-receptance-weighted-key-value-rnn-2025/)》，2025-12-01。偏机制梳理，适合辅助理解 time-mix、门控和衰减的直觉，但细节仍应以论文与官方实现为准。
