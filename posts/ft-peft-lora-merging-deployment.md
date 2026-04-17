## 核心结论

LoRA 是一种参数高效微调方法，白话说，就是不改大模型主体参数，只额外训练一小组低秩矩阵。部署时有两种主流形态。

第一种是在线叠加：保留 base model 和 LoRA adapter 两条路径，推理时计算
$$
h = Wx + \frac{\alpha}{r}BAx
$$
优点是可以在同一个底座模型上热切换多个 adapter，适合多任务服务、灰度测试、快速回滚。缺点是推理路径里仍然存在 LoRA 分支，工程上通常会多一层调度和额外计算。

第二种是离线合并：在上线前把 LoRA 增量直接写回原权重，得到
$$
W' = W + \frac{\alpha}{r}BA
$$
之后推理只需要计算 $W'x$。这意味着模型前向路径重新退化成普通线性层，不再需要额外的 LoRA 分支。对低延迟部署，这通常是默认选择。

结论可以压缩成一句话：如果你需要“一个底座 + 多个技能随时切换”，选在线叠加；如果你需要“固定能力 + 最低推理复杂度”，选离线合并。

| 方案 | 推理路径 | 延迟 | 灵活性 | 典型场景 |
|---|---|---:|---:|---|
| 在线叠加 | `base + adapter` 同时参与 | 较高 | 高 | 多租户、多任务热切换 |
| 离线合并 | 只保留 `merged weight` | 较低 | 低 | 固定能力生产服务 |
| 多 LoRA 组合后再合并 | 先组合 adapter，再写回 base | 取决于最终是否合并 | 中 | 多技能统一部署 |

---

## 问题定义与边界

“LoRA 的合并与部署”本质上不是训练问题，而是推理形态选择问题。你要解决的是三个目标之间的冲突：延迟、内存、灵活性。

这里先定义几个边界。

Base model 是底座模型，白话说，就是没加当前任务能力之前的原始模型。Adapter 是适配器，白话说，就是那组额外训练出来的小矩阵参数。Merge 是合并，白话说，就是把 adapter 的影响直接加回原始权重，让模型以后像普通模型一样跑。

部署决策通常看下面四件事。

| 决策问题 | 更偏向在线叠加 | 更偏向离线合并 |
|---|---|---|
| 是否要频繁切换任务 | 是 | 否 |
| 是否极度在意单次请求延迟 | 否 | 是 |
| 是否需要快速回滚/灰度不同 adapter | 是 | 否 |
| 是否准备把模型再做量化和导出 | 不一定 | 通常是 |

一个常见边界是量化模型。量化，白话说，就是用更少 bit 表示权重来省显存。很多新手会问：我已经有一个 4-bit 的底座模型，能不能直接把 fp16 的 LoRA 权重加上去？

工程上不应该默认这么做。原因不是公式失效，而是实现载体变了。量化后的线性层可能不是普通浮点权重张量，直接相加可能不支持，或者支持但数值路径不可靠。更稳妥的顺序通常是：

1. 加载高精度 base model，例如 fp16 或 bf16。
2. 加载 LoRA adapter。
3. 先 merge。
4. 再对 merged model 做量化。
5. 用量化后的最终产物部署。

如果你已经在做多 adapter 服务，还要再加一个边界：多个 LoRA 可以做线性组合，但前提是它们的目标模块、rank、缩放规则、底座版本要兼容，而且任务不能强冲突。否则最后得到的不是“多技能”，而是“互相抵消的参数汤”。

玩具例子可以先这样理解：你有一个“通用中文模型”，又训练了一个“SQL 生成 LoRA”和一个“合同摘要 LoRA”。如果线上要根据请求即时切换任务，用在线叠加最直接；如果线上其实只需要一个固定模型同时覆盖这两种能力，那就更偏向先组合再合并，最终只部署一个模型权重文件。

---

## 核心机制与推导

LoRA 的核心是把一个大矩阵更新 $\Delta W$ 拆成两个小矩阵乘积：
$$
\Delta W = \frac{\alpha}{r}BA
$$
其中：

- $A \in \mathbb{R}^{r \times d_{in}}$
- $B \in \mathbb{R}^{d_{out} \times r}$
- $r$ 是 rank，白话说，就是中间瓶颈的宽度
- $\alpha$ 是 scaling，白话说，就是把低秩更新按比例放大或缩小

原始线性层输出是
$$
h = Wx
$$
加上 LoRA 后变成
$$
h = Wx + \frac{\alpha}{r}BAx
$$
把右边整理一下：
$$
h = \left(W + \frac{\alpha}{r}BA\right)x
$$
于是定义
$$
W' = W + \frac{\alpha}{r}BA
$$
就得到
$$
h = W'x
$$
这就是“可以合并”的数学原因。只要该层是线性可加的，LoRA 分支就可以提前折叠进原权重。

最小数值例子：

设某一层是标量版玩具模型，原始权重 $W=2.0$，LoRA 学到的 $BA=0.4$，并且 $\alpha=16, r=4$，则
$$
\Delta W=\frac{16}{4}\times 0.4 = 1.6
$$
所以合并后
$$
W' = 2.0 + 1.6 = 3.6
$$
如果输入 $x=5$，那么：

- 在线叠加：$2.0 \times 5 + 1.6 \times 5 = 18$
- 离线合并：$3.6 \times 5 = 18$

两者数学等价。

这也是为什么微软 `loralib` 的设计里，`model.eval()` 可以触发 merge，而 `model.train()` 又可以 unmerge。训练时需要把 LoRA 参数保持显式结构，梯度才能只流向 adapter；评估时则可以把它们写回去，拿到没有额外分支的前向路径。

需要注意一个推导边界：这里的“等价”成立在权重和算子语义一致时。若中途引入量化、混合精度、特定 fused kernel，理论公式仍成立，但数值误差会影响“完全一致”的程度。所以工程上正确说法不是“永远零差异”，而是“在相同高精度表示下前向等价；在量化部署中还要额外验证误差”。

真实工程例子：

一个法务助手服务，需要同时支持“合同摘要”和“法条问答”。团队在同一个 7B base model 上分别训练了两个 LoRA。测试环境保留两个 adapter 在线切换，便于 A/B 和回滚。生产环境发现 99 分位延迟过高，于是改成先验证两个 adapter 的兼容性，再按权重做组合，最后 merge 成一个统一模型并重新量化。这样线上只保留一个权重产物，调用链也从“加载底座 + 激活 adapter”简化成“直接推理”。

---

## 代码实现

下面先用一个纯 Python 的玩具实现说明“在线叠加”和“离线合并”为什么等价。代码可直接运行。

```python
import numpy as np

def lora_delta(A, B, alpha, r):
    return (alpha / r) * (B @ A)

# base weight: dout=2, din=3
W = np.array([
    [1.0, 2.0, 0.0],
    [0.5, -1.0, 3.0],
], dtype=np.float64)

# LoRA: B in R^{2x1}, A in R^{1x3}, rank r=1
A = np.array([[2.0, -1.0, 0.5]], dtype=np.float64)
B = np.array([[0.1], [0.2]], dtype=np.float64)
alpha = 4.0
r = 1.0

x = np.array([1.5, -2.0, 0.5], dtype=np.float64)

delta = lora_delta(A, B, alpha, r)
W_merged = W + delta

y_online = W @ x + delta @ x
y_merged = W_merged @ x

assert np.allclose(y_online, y_merged), "在线叠加与离线合并应当等价"

# 再验证上文的标量玩具例子
W_scalar = 2.0
BA_scalar = 0.4
alpha = 16.0
r = 4.0
delta_scalar = (alpha / r) * BA_scalar
W_prime = W_scalar + delta_scalar

assert abs(delta_scalar - 1.6) < 1e-9
assert abs(W_prime - 3.6) < 1e-9
assert abs(W_scalar * 5 + delta_scalar * 5 - W_prime * 5) < 1e-9

print("all assertions passed")
```

在 PyTorch/PEFT 工程里，实际流程更接近下面这样：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "your-base-model",
    torch_dtype="float16",
)

model = PeftModel.from_pretrained(base, "your-lora-adapter")

# 生产场景：直接合并并卸载 adapter 结构
merged_model = model.merge_and_unload()

merged_model.save_pretrained("merged-artifact")
```

如果你使用的是微软 `loralib` 风格实现，则常见逻辑是：

- `model.eval()` 时把 $\Delta W$ 加到 `weight`
- `model.train()` 时再减回去
- `merge_weights=False` 时保持在线叠加，不自动合并

这两套接口长得不同，但底层思想一致：是否把 $\frac{\alpha}{r}BA$ 物化到原权重里。

工程建议是把部署流水线固定成一条可重复脚本：

1. 校验 base checkpoint 版本。
2. 校验 LoRA target modules、rank、dtype。
3. 以高精度加载 base。
4. merge 或 merge multiple adapters。
5. 做一轮回归推理。
6. 再导出量化版本。
7. 保存 tokenizer、config、generation config 与模型一同发布。

---

## 工程权衡与常见坑

最常见的坑不是公式错，而是工程对象不一致。

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| dtype 不一致 | merge 后报错或输出异常 | base、adapter、量化权重格式不同 | 统一到 fp16/bf16 后 merge |
| `alpha/r` 处理错 | 效果明显变弱或过强 | 忘了 scaling 或重复 scaling | 明确检查 `lora_alpha` 与 `r` |
| target modules 不一致 | 加载 adapter 失败或部分失效 | 训练和部署命中的层不同 | 固定模块名并做启动时校验 |
| 多 LoRA 强冲突 | 组合后性能下降 | 不同任务更新方向互相覆盖 | 先离线评测再决定是否组合 |
| 量化顺序错误 | merge 不支持或误差放大 | 量化层不是普通浮点矩阵 | 优先“高精 merge，再量化” |
| 以为 merge 后一定更快 | 端到端收益不明显 | 真瓶颈可能在 KV cache、采样、IO | 用 profiler 看真实热点 |

这里有两个经常被误解的点。

第一，离线合并并不自动保证“整体请求一定更快”。它只能保证 LoRA 分支不再单独参与前向。若你的瓶颈在 tokenizer、网络、KV cache 或长序列采样，收益会被摊薄。

第二，多 LoRA 线性组合不是免费午餐。线性组合，白话说，就是把多个 adapter 的增量按系数加权，例如
$$
\Delta W = \lambda_1 \Delta W_1 + \lambda_2 \Delta W_2
$$
这要求几个 adapter 至少在结构上兼容。即便结构兼容，语义上也可能冲突。例如“写 SQL”偏向严格模式化输出，“合同摘要”偏向长文本压缩，两者并不一定天然能混成一个更好的 adapter。

真实工程里，一个更稳的流程通常是：先分别测单 adapter 效果，再测组合效果，最后再决定是否 merge 成统一 artifact。不要一开始就把“能相加”误解成“相加后一定更强”。

---

## 替代方案与适用边界

除了“在线叠加”和“离线合并”，还有第三类思路：多 adapter 融合后再部署。这类方法的核心不是只把一个 LoRA 写回 base，而是先把多个 LoRA 做合成，再决定是否最终 merge 到 base 中。

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 在线叠加 | 最灵活，可热切换 | 延迟和管理复杂度更高 | 测试、多租户、动态任务 |
| 离线合并 | 部署最简单，路径最短 | 不方便再切换 adapter | 固定任务、低延迟生产 |
| 多 LoRA 融合 | 可把多技能压成一个产物 | 需要额外验证冲突 | 多技能统一交付 |

像 PEFT 已支持多种 adapter merging 方法，例如线性加权、SVD、TIES、DARE 等。这里要把边界说清楚：这些方法不是“LoRA 基本公式的一部分”，而是“多 adapter 合成策略”。它们解决的是“多个任务更新如何组合”，不是“单个 LoRA 是否能写回 base”。

因此推荐的判断标准很简单。

如果你还在调参、做评测、频繁换任务，用在线叠加。

如果任务已经稳定，线上追求低复杂度和低维护成本，用离线合并。

如果你想把多个技能压成一个部署产物，但又不想重新全量训练，可以尝试多 adapter 融合；不过前提是你愿意为兼容性验证和回归测试付出额外成本。

---

## 参考资料

- LoRA 原始论文：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*，arXiv 2106.09685  
  https://arxiv.org/abs/2106.09685
- Microsoft `loralib` 官方仓库，说明 `model.eval()` 会触发合并、`model.train()` 会撤销合并  
  https://github.com/microsoft/LoRA
- Hugging Face PEFT LoRA 文档，`merge_and_unload()`、`merge_adapter()`、`unmerge_adapter()` 说明  
  https://huggingface.co/docs/peft/en/developer_guides/lora
- Hugging Face PEFT 合并多 LoRA 方法介绍，包含 linear、SVD、TIES、DARE 等  
  https://huggingface.co/blog/peft_merging
- Hugging Face Transformers/PEFT adapter 管理文档，包含多 adapter 加载与切换  
  https://huggingface.co/docs/transformers/main/peft
- Hugging Face PEFT 量化文档，说明量化模型与 merge/unmerge 的支持边界  
  https://huggingface.co/docs/peft/en/developer_guides/quantization
