## 核心结论

LoRA 推理合并，指的是把训练阶段学到的低秩增量 $\Delta W=\frac{\alpha}{r}BA$ 直接加回基座权重 $W_0$，得到一个新的推理权重 $W'$。这样做不会改变模型学到的任务能力，只会改变推理时的执行路径。

新手可以先这样理解：训练时只学一小块补丁，推理时要么每次临时贴补丁，要么提前把补丁贴进原墙面。合并就是提前贴好，线上直接把它当成一堵完整的墙。

合并的收益主要是工程层面的。未合并时，线上要同时维护“基座权重 + adapter 权重”，前向还要多一段低秩修正计算；合并后，线上只剩普通线性层或普通 Transformer 权重路径，延迟通常更稳，吞吐通常更高，调度逻辑也更简单。代价是失去快速切换 adapter 的灵活性。

| 对比项 | 未合并 | 合并后 |
| --- | --- | --- |
| 推理路径 | 基座前向 + LoRA 修正 | 只跑合并后的基座前向 |
| 线上灵活性 | 高，可按请求切换 adapter | 低，权重基本固定 |
| 热路径开销 | 更高 | 更低 |
| 适合场景 | 多租户动态切换 | 固定版本、稳定流量 |

---

## 问题定义与边界

LoRA 是一种参数高效微调方法。参数高效微调，意思是不用改动整套大模型参数，只训练少量新增参数。它解决的问题是：在不重训整个大模型的前提下，怎样把任务相关的改动加到模型输出上。

这里先统一几个术语：

| 术语 | 含义 | 白话解释 |
| --- | --- | --- |
| 基座模型 | 原始预训练模型 | 没做任务定制的“大底座” |
| LoRA | 低秩适配方法 | 用两个小矩阵近似一大块权重改动 |
| adapter | 适配器参数 | 挂在基座模型旁边的小补丁 |
| merge | 合并 | 把补丁写回原权重 |
| unmerge | 反合并 | 把写回去的补丁再拆出来 |
| 离线合并 | 上线前先合并 | 发布时生成独立权重文件 |
| 在线加载 | 推理时动态挂载 | 请求来了再选 adapter |

本文讨论的是“推理合并”。边界很明确：

1. 不讨论 LoRA 的训练过程本身。
2. 不讨论多个 LoRA 在训练阶段如何共同优化。
3. 重点讨论线上是否把 adapter 折叠进基座权重，以及这样做带来的性能与灵活性取舍。

玩具例子很简单：同一个基座模型服务两个部门，法务部门挂法务 LoRA，客服部门挂客服 LoRA。如果法务版本已经固定两个月不会变，就可以把法务 LoRA 离线合并成一份独立模型上线；客服如果还要频繁换版本，就继续保留在线 adapter。

---

## 核心机制与推导

设基座线性层权重为 $W_0 \in \mathbb{R}^{d_{out}\times d_{in}}$，输入为 $x \in \mathbb{R}^{d_{in}}$。LoRA 不直接学习完整的 $\Delta W$，而是学习两个更小的矩阵：

$$
A \in \mathbb{R}^{r\times d_{in}}, \quad B \in \mathbb{R}^{d_{out}\times r}
$$

其中 $r$ 是秩，秩可以理解成“补丁容量”的大小。于是前向写成：

$$
h = W_0x + \Delta Wx = W_0x + \frac{\alpha}{r}B(Ax)
$$

这里的 $\alpha$ 是缩放系数，用来控制补丁强度。因为矩阵乘法满足结合律，所以可以直接定义：

$$
W' = W_0 + \frac{\alpha}{r}BA
$$

于是有：

$$
h = W'x
$$

这就是合并前后等价性的来源。未合并时是“两段式计算”，合并后变成“一段式计算”，数学结果一致，执行路径不同。

下面给一个最小数值例子。设：

$$
W_0=\begin{bmatrix}1&0\\0&1\end{bmatrix},\;
A=\begin{bmatrix}1&2\end{bmatrix},\;
B=\begin{bmatrix}3\\4\end{bmatrix},\;
r=1,\;
\alpha=1
$$

则：

$$
\Delta W = BA =
\begin{bmatrix}
3 & 6 \\
4 & 8
\end{bmatrix}
$$

所以：

$$
W' = W_0 + \Delta W =
\begin{bmatrix}
4 & 6 \\
4 & 9
\end{bmatrix}
$$

若输入

$$
x=\begin{bmatrix}2\\1\end{bmatrix}
$$

未合并时：

$$
W_0x + \Delta Wx =
\begin{bmatrix}2\\1\end{bmatrix} +
\begin{bmatrix}12\\16\end{bmatrix}
=
\begin{bmatrix}14\\17\end{bmatrix}
$$

合并后：

$$
W'x =
\begin{bmatrix}
4 & 6 \\
4 & 9
\end{bmatrix}
\begin{bmatrix}2\\1\end{bmatrix}
=
\begin{bmatrix}14\\17\end{bmatrix}
$$

两者完全一致。

| 项目 | 未合并前向 | 合并后前向 | 是否等价 | 额外计算 |
| --- | --- | --- | --- | --- |
| 单 adapter | $W_0x + \frac{\alpha}{r}B(Ax)$ | $\left(W_0 + \frac{\alpha}{r}BA\right)x$ | 是 | 未合并多一段低秩路径 |
| 多 adapter 动态切换 | 每次选不同 $A,B$ | 每个版本需单独生成 $W'$ | 版本内等价 | 合并后失去动态切换 |

---

## 代码实现

在 PEFT 这类实现里，LoRA 常见有两条路径。

第一条是动态叠加路径。前向时先算基座层输出，再额外算一段 `lora_B(lora_A(x)) * scaling`，最后把结果加回去。`scaling` 就是上面公式里的 $\frac{\alpha}{r}$ 或其变体。

第二条是合并路径。先算出 `delta_weight`，再把它加到 `base_layer.weight` 上。之后推理时，这一层已经和普通线性层没有本质区别。

可运行的玩具代码如下，它直接验证“合并前后结果一致”：

```python
import numpy as np

W0 = np.array([[1.0, 0.0],
               [0.0, 1.0]])
A = np.array([[1.0, 2.0]])      # shape: (r, d_in), r = 1
B = np.array([[3.0],
              [4.0]])           # shape: (d_out, r)
x = np.array([[2.0],
              [1.0]])
alpha = 1.0
r = 1.0
scaling = alpha / r

delta_W = scaling * (B @ A)
h_unmerged = W0 @ x + delta_W @ x

W_merged = W0 + delta_W
h_merged = W_merged @ x

assert np.allclose(h_unmerged, h_merged)
assert np.allclose(h_merged, np.array([[14.0], [17.0]]))

print("unmerged:", h_unmerged.ravel())
print("merged:", h_merged.ravel())
```

真实工程里的典型流程更像下面这样：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "your-base-model"
lora_path = "your-lora-adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_path)

merged = model.merge_and_unload()   # 重点：接住返回值
inputs = tokenizer("解释 LoRA merge 的作用", return_tensors="pt")
out = merged.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

伪代码流程可以压成三步：

```text
加载基座模型
    -> 加载 LoRA adapter
        -> 选择：
           A. 直接在线推理：base + adapter 动态叠加
           B. merge_and_unload：生成合并后模型，再用普通模型推理
```

实现时最容易踩的点不是公式，而是状态管理。

如果你要保留切换能力，`merge_adapter()` / `unmerge_adapter()` 更合适。它们的含义是“临时把 adapter 融进前向路径，但仍保留 adapter 状态，之后还能拆回去”。而 `merge_and_unload()` 的目标是产出一个可独立使用的基座模型，适合部署独立权重文件。

---

## 工程权衡与常见坑

真实部署里，LoRA 推理合并本质上是在“灵活性”和“性能”之间做交换。

真实工程例子：一个多租户文本生成服务，租户 A 是法务问答，租户 B 是客服问答。如果 A、B 都固定版本且流量稳定，通常会分别产出两份合并后的模型，部署成两个独立服务实例。这样热路径最短，延迟和吞吐都更稳。反过来，如果同一个服务实例需要按请求动态选择十几个 adapter，就不能简单全量合并，否则每个版本都要维护一份独立大权重，成本会迅速变高。

| 风险点 | 现象 | 影响 | 规避方式 |
| --- | --- | --- | --- |
| `merge_and_unload()` 不是原地修改 | 忘记接返回值 | 仍在用旧模型对象 | `merged = model.merge_and_unload()` |
| 合并后不能直接再切换适配器 | adapter 状态被卸载 | 失去热切换能力 | 需要切换时改用 `merge_adapter()` / `unmerge_adapter()` |
| batch 内 adapter 过杂会降吞吐 | 同批请求需要不同 LoRA | 调度复杂、算子利用率下降 | 做租户分流或 adapter 分桶 |
| `max_lora_rank` 配置影响显存 | rank 上限设太大 | 平白增加显存预留 | 按真实 rank 分布设置 |
| `safe_merge` 用于检查 NaN | 低精度或异常权重时可能出错 | 合并后结果不稳定 | 离线合并时开启校验 |
| DoRA 等变体未必完全兼容普通合并路径 | 变体逻辑不止是 $BA$ | 行为可能与普通 LoRA 不同 | 先看框架文档与实现说明 |

一个常见误区是把“LoRA 理论上无额外推理延迟”和“工程上完全零开销”混为一谈。理论上，LoRA 通过低秩结构减少训练参数；工程上，如果 adapter 是在线动态挂载的，仍然会带来额外权重管理、请求路径判断、batch 分组和低秩分支计算开销。合并的价值，恰恰在于把这些开销挪到离线阶段。

---

## 替代方案与适用边界

不是所有场景都应该做合并。是否合并，核心看两件事：版本是否稳定，线上是否需要频繁切换任务。

如果场景是“法务 LoRA 固定上线，未来一周都不会换”，离线合并通常是更优解。因为服务目标是低延迟、稳吞吐、少调度逻辑。

如果场景是“同一请求可能走客服、售后、退款、质检四种 LoRA 之一”，在线 adapter 更合理。因为服务目标是灵活切换，而不是把每个任务都做成一份独立大模型。

| 场景 | 推荐方案 | 原因 | 代价 |
| --- | --- | --- | --- |
| 固定版本、独立部署 | 离线合并 | 热路径最短，延迟低 | 失去快速切换 |
| 单租户稳定流量 | 离线合并 | 运维简单，吞吐稳 | 每次换版本都要重新产物化 |
| 多租户按请求切换 | 在线 adapter | 灵活，便于统一服务入口 | 显存与调度复杂度更高 |
| 灰度测试、频繁回滚 | 在线 adapter 或可逆合并 | 便于快速切换版本 | 性能不如完全离线合并 |
| 多 LoRA 混合实验 | 保留 adapter | 方便对比与实验 | 不适合追求极致时延 |

因此，LoRA 推理合并不是“默认更先进”的方案，而是“在版本固定时更像生产化”的方案。你优先优化的是线上执行效率，还是优先保留任务切换自由度，决定了是否该 merge。

---

## 参考资料

1. [LoRA: Low-Rank Adaptation of Large Language Models - Microsoft Research](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/?locale=zh-cn)
2. [Hugging Face PEFT LoRA Developer Guide](https://huggingface.co/docs/peft/en/developer_guides/lora)
3. [Hugging Face PEFT Package Reference: Tuners](https://huggingface.co/docs/peft/package_reference/tuners)
4. [Hugging Face PEFT Package Reference: PeftModel](https://huggingface.co/docs/peft/package_reference/peft_model)
5. [PEFT Source: peft/src/peft/tuners/lora/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)
6. [vLLM Documentation: LoRA Adapters](https://docs.vllm.ai/features/lora.html)
