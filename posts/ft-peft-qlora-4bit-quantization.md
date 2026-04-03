## 核心结论

QLoRA 的关键不是“把整个大模型都用 4-bit 训练”，而是“把底座模型权重以 4-bit 形式存储并冻结，只训练少量高精度 LoRA 适配器”。这里的底座模型，指预训练好的原始大模型；冻结，指训练时不更新这些原始权重。这样做的结果是：显存主要花在前向计算、少量激活值和 LoRA 参数上，而不是花在 65B 级别参数的梯度与优化器状态上。

对零基础读者，可以先抓住一句话：QLoRA 相当于把一个超大的模型压缩保存成 4-bit 版本，训练时不给它“动骨架”，只外挂一个很小的可训练模块 LoRA 去适配新任务。显存下降的核心来自“冻结大头，只训练小头”。

传统全量微调与 QLoRA 的差异可以先看表：

| 方案 | 底座权重存储 | 训练时更新对象 | 优化器状态规模 | 显存压力 |
|---|---:|---|---|---|
| 全量微调 | FP16/BF16 | 全部参数 | 与全部参数同量级 | 最高 |
| LoRA + FP16 底座 | FP16/BF16 | LoRA | 只覆盖 LoRA | 中等 |
| QLoRA | 4-bit NF4 | LoRA | 只覆盖 LoRA | 最低 |

如果把模型总参数记为 $\#Full$，LoRA 可训练参数记为 $\#LoRA$，偏置等额外训练项记为 $\#Bias$，那么训练参数规模从

$$
\#Train_{\text{full}} = \#Full
$$

变成

$$
\#Train_{\text{QLoRA}} = \#LoRA + \#Bias,\quad \#Train_{\text{QLoRA}} \ll \#Full
$$

这就是 QLoRA 能在单张 48GB 卡上处理 65B 级模型的根本原因。

---

## 问题定义与边界

QLoRA 要解决的问题很直接：当显卡显存只有几十 GB 时，如何对超大语言模型做有效微调。这里的微调，指把通用预训练模型继续训练成某个任务或领域更强的版本。全量微调不现实，因为显存不只要放权重，还要放梯度、优化器状态和中间激活。参数一旦到几十亿甚至几百亿，代价会成倍增长。

QLoRA 的边界也要说清楚，否则很容易误解成“任何场景都该上 4-bit”。

| 边界项 | QLoRA 的做法 | 不做的事 |
|---|---|---|
| 底座权重 | 4-bit 量化后冻结 | 不回传梯度到这些权重 |
| 训练对象 | LoRA 适配器 | 不更新全部线性层原始参数 |
| 前向计算 | 按需解量化到高精度计算 | 不直接用 4-bit 做完整高精度训练替代 |
| 依赖实现 | 常见是 bitsandbytes + PEFT | 不是纯手写 PyTorch 默认功能 |

新手最容易理解的一句话是：只有外挂的 LoRA 会动，底座权重像一张定格的底图。每次前向时，系统把这张“压缩底图”临时还原到计算精度参与矩阵乘法，但训练后的修改只写进 LoRA，不写回底座。

这个边界带来两个直接后果。

第一，QLoRA 省显存，但不会让你获得“和全量微调一样的参数自由度”。它本质仍是参数高效微调，英文叫 PEFT，意思是只更新很少一部分参数。

第二，它依赖量化假设。NF4 之所以有效，是因为很多预训练权重近似服从以 0 为中心的正态分布。若权重分布明显偏离这一假设，量化误差就可能放大。

玩具例子可以这样理解。假设一个小线性层有 100 万个参数。全量微调时，这 100 万个参数都要保存高精度权重、梯度和优化器状态。QLoRA 时，这 100 万个参数可以压成 4-bit 冻结不动，只额外加一个秩为 $r$ 的低秩更新：

$$
W' = W + \Delta W,\quad \Delta W = BA
$$

其中 $A,B$ 就是 LoRA 的两个小矩阵。低秩，意思是这个更新矩阵不是任意形状地变化，而是受限在一个较小的子空间里，所以参数量远小于直接更新整个 $W$。

---

## 核心机制与推导

QLoRA 的 4-bit 量化通常使用 NF4。NF4 是 NormalFloat 4 的缩写，可以把它理解成“专门为近似正态分布权重设计的 4-bit 码本”。码本，指预先定义好的离散取值表。4-bit 一共只能表示 $2^4 = 16$ 个离散值，所以系统不会直接存原始浮点数，而是存一个 0 到 15 的索引码。

量化过程可以抽象成四步：

| 步骤 | 作用 | 结果 |
|---|---|---|
| 1. 分组 | 把权重按 block 分组 | 每组单独统计 |
| 2. 规范化 | 用组内尺度把权重缩到可量化范围 | 降低不同组的数值差异 |
| 3. NF4 查表编码 | 每个权重映射到 16 个离散码之一 | 得到 4-bit 索引 $q$ |
| 4. 反量化 | 前向时按尺度恢复近似值 | 得到 $\hat{w}$ |

它的核心公式可以写成：

$$
\hat{w} = s \cdot c[q] + b
$$

其中：

- $q$ 是 4-bit 量化码；
- $c[q]$ 是码本中的第 $q$ 个离散中心值；
- $s$ 是组级尺度，意思是这一组权重的缩放系数；
- $b$ 是偏置项，有些实现可视为 0 或并入其他统计量。

这套机制为什么成立？因为很多大模型权重集中在 0 附近，极端大值较少。若仍用均匀刻度，0 附近会浪费分辨率；NF4 把有限的 16 个码值更合理地分配在正态分布常见区域，因此同样 4-bit 下误差更小。

题目给出的数值例子正好能说明这一点。假设某个权重 $w_0 = 0.15$，量化后落到 NF4 的第 7 号码，且 $c[7] = 0.18$，组级尺度 $s = 0.8$。那么恢复值是：

$$
\hat{w} = 0.8 \times 0.18 = 0.144
$$

误差为：

$$
|w_0 - \hat{w}| = |0.15 - 0.144| = 0.006
$$

这个误差通常可以接受，因为底座权重不再被训练修正，真正的任务适配交给 LoRA 来补。

再往前一步，QLoRA 还用了 double quantization。它不是把权重再量化一次，而是把量化过程里用于恢复权重的常数，比如尺度 $s$，继续量化保存。这样做的原因很简单：如果每组权重都要额外保存高精度尺度，那总体开销并不小。double quantization 通过对这些尺度再次压缩，进一步减少显存。

继续上面的例子，若 $s=0.8$ 不直接用高精度浮点保存，而是再映射成一个 8-bit 值，比如编码成 205，在前向时再近似恢复为 0.8，那么整体元数据开销会继续下降。很多资料会把它概括为“平均每个参数继续省掉一部分 bit”。

然后是 LoRA。LoRA 的本质是给线性层加一个低秩增量：

$$
y = Wx + BAx \cdot \gamma_r
$$

其中 $\gamma_r = \alpha / r$。$r$ 是秩，也就是低秩更新的宽度；$\alpha$ 是缩放超参数。QLoRA 中，$W$ 是量化并冻结的底座权重，$A,B$ 则通常保留在 FP16 或 BF16。也就是说，真正学习的是高精度小矩阵，不是 4-bit 大矩阵。

最后一个关键部件是 paged optimizer。paged 的直译是“分页”。在这里可以理解成：当优化器状态占用显存过高时，系统借助统一内存机制，把一部分状态临时换到 CPU 侧，避免训练峰值一下冲爆 GPU 显存。它不是让训练免费变快，而是把原本会 OOM 的峰值压平到可以运行的范围。

真实工程例子是 65B 级模型指令微调。若底座用 FP16，光模型权重就非常重，再加梯度和 Adam 状态几乎不可能在单卡完成。换成 QLoRA 后，底座变成 4-bit 冻结，训练只维护 LoRA 参数和对应优化器状态，再配合 paged optimizer，单张 48GB 卡就能跑起来。这也是 QLoRA 在工程上真正有价值的地方。

---

## 代码实现

下面先给一个不依赖大模型的玩具 Python 例子，用最简方式模拟“冻结底座，只训练 LoRA”的核心逻辑。可运行，且包含 `assert`。

```python
import numpy as np

# 冻结的底座权重，模拟量化后解量化得到的近似权重
W_base = np.array([
    [0.20, -0.10],
    [0.05,  0.30],
], dtype=np.float32)

# LoRA 低秩参数，实际训练时只更新这两个矩阵
A = np.array([[0.10, -0.20]], dtype=np.float32)   # shape: (r=1, in=2)
B = np.array([[0.30], [0.40]], dtype=np.float32)  # shape: (out=2, r=1)

alpha = 1.0
r = 1
scale = alpha / r

x = np.array([[2.0], [1.0]], dtype=np.float32)

# 前向：y = (W + B @ A * scale) x
delta_W = B @ A * scale
y = (W_base + delta_W) @ x

assert W_base.shape == (2, 2)
assert delta_W.shape == (2, 2)
assert y.shape == (2, 1)

# 模拟一次“只更新 LoRA”的训练后，底座保持不变
W_before = W_base.copy()
A = A + 0.01
B = B - 0.02

assert np.allclose(W_base, W_before)  # 底座没被更新
print("y =", y.ravel())
```

如果进入真实工程，常见写法是 `transformers + bitsandbytes + peft`。关键配置有三件事：

| 参数 | 典型值 | 作用 |
|---|---|---|
| `load_in_4bit` | `True` | 以 4-bit 加载底座 |
| `bnb_4bit_quant_type` | `"nf4"` | 使用 NF4 量化类型 |
| `bnb_4bit_use_double_quant` | `True` | 开启 double quantization |
| `bnb_4bit_compute_dtype` | `torch.bfloat16` | 前向/反向计算精度 |
| `r` | `8/16/32/64` | LoRA 秩 |
| `lora_alpha` | 常与 `r` 同量级 | LoRA 缩放系数 |

示例代码如下：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import PagedAdamW32bit

model_name = "meta-llama/Llama-2-7b-hf"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)

# 防止误更新底座参数
model.requires_grad_(False)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

lora_model = get_peft_model(model, lora_config)

trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
optimizer = PagedAdamW32bit(trainable_params, lr=2e-4)

# 简单校验：底座冻结，至少有一部分 LoRA 参数可训练
num_trainable = sum(p.numel() for p in trainable_params)
num_frozen = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)

assert num_trainable > 0
assert num_frozen > num_trainable
```

这里最关键的不是 API 名字，而是训练对象必须是 `trainable_params` 这类 LoRA 参数集合，而不是整个 `model.parameters()`。一旦把底座参数送进优化器，就失去 QLoRA 的意义了。

---

## 工程权衡与常见坑

QLoRA 很强，但它不是“白送显存”。它是用量化误差、实现复杂度和一定的吞吐损失，换取可训练性。

常见坑可以直接列出来：

| 坑 | 现象 | 对策 |
|---|---|---|
| 忘记冻结底座 | 显存瞬间暴涨，直接 OOM | 明确 `model.requires_grad_(False)`，并检查优化器参数 |
| 没开 double quant | 元数据开销偏高 | `bnb_4bit_use_double_quant=True` |
| 没用 paged optimizer | 峰值显存超预算 | 改用 `PagedAdamW` 或 Trainer 的 paged 配置 |
| 把 FP4 当 NF4 直接替换 | 训练可跑，但效果可能下滑 | 训练优先用 NF4 |
| target modules 选错 | 几乎不学习或效果差 | 先确认模型线性层命名 |
| 计算 dtype 太低 | 数值不稳定 | 优先 `bf16`，其次 `fp16` |

新手最容易踩的坑就是第一条。只要忘了冻结底座，训练图里就会为海量参数保存梯度，优化器也会为它们分配状态。对 65B 模型，这不是“小失误”，而是直接从“单卡可训练”退回“根本跑不动”。

再强调一次，QLoRA 不是让量化权重自己学习。梯度经过这些量化层时，作用是让 LoRA 学会如何补偿量化底座在当前任务上的不足，而不是去修补那份 4-bit 权重本身。

真实工程里还有一个隐蔽问题：显存峰值往往不出现在模型刚加载时，而出现在长序列 batch、梯度累计或 optimizer step 那一下。所以很多人看到“模型已经成功加载到 41GB”就以为能训练，结果一反向或一更新就 OOM。paged optimizer 的价值正是在这里，它处理的是峰值，而不是静态占用。

可直接复用的安全写法如下：

```python
import torch
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 训练前必须确认：
# 1. 底座冻结
# 2. double quant 已开启
# 3. 优化器只接收 LoRA 参数
assert quant_config.load_in_4bit is True
assert quant_config.bnb_4bit_quant_type == "nf4"
assert quant_config.bnb_4bit_use_double_quant is True
```

---

## 替代方案与适用边界

如果目标是“在有限显存下做大模型微调”，QLoRA 不是唯一方案。关键是看你优先级是什么：显存、精度、速度，还是实现简单。

| 方案 | 显存需求 | 精度风险 | 训练成本 | 适用场景 |
|---|---|---|---|---|
| 全量微调 | 最高 | 最低 | 最高 | 资源充足、追求最大自由度 |
| LoRA + FP16 底座 | 高 | 低 | 中等 | 有多卡或较大显存 |
| QLoRA + NF4 | 低 | 低到中 | 低 | 单卡或受限显存训练 |
| QLoRA + FP4 | 更低或更快 | 中到较高 | 低 | 更看重资源压缩 |
| 训练后再做 GPTQ | 训练期不省 | 推理期省 | 额外量化流程 | 部署优化 |

一个实用判断标准是：

- 显存不到 48GB，模型又在 30B、65B 这个量级，优先考虑 QLoRA。
- 如果你手里有足够多卡，且任务对极限精度特别敏感，可以考虑 FP16/BF16 的 LoRA 或全量微调。
- 如果训练已经完成，只是想把部署成本降下来，那么 GPTQ、AWQ 这类推理量化更相关。它们主要优化推理，不负责训练阶段的显存压力。

对初学者，最容易记住的对比是：QLoRA vs FP16 LoRA，本质是“用更多量化技巧换更低显存”。前者通常更省显存，后者通常更直观、兼容面更宽。选择哪一个，不取决于概念是否先进，而取决于你手里的卡够不够大，以及任务是否接受一点量化近似。

---

## 参考资料

| 类别 | 资料 | 作用 |
|---|---|---|
| 论文 | Dettmers 等，QLoRA: Efficient Finetuning of Quantized LLMs，arXiv:2305.14314 | QLoRA 的原始方法、实验和结论 |
| 官方文档 | Hugging Face Transformers bitsandbytes 量化文档 | `BitsAndBytesConfig`、NF4、nested quantization 配置说明 |
| 官方文档 | bitsandbytes optimizer 文档 | `PagedAdamW`、分页优化器行为说明 |
| 实践文档 | FinLoRA 文档中的 QLoRA 章节 | 用公式解释 QLoRA 前向与 LoRA 连接方式 |
| 博客解析 | sanowl 的 QLoRA 机制解析 | 用更直观的方式理解 NF4、double quantization |
| 部署延伸 | GPTQ 相关资料 | 训练后推理量化的补充路线 |

- QLoRA 论文：https://arxiv.org/abs/2305.14314
- Hugging Face 论文页：https://huggingface.co/papers/2305.14314
- Transformers bitsandbytes 文档：https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- bitsandbytes optimizer 文档：https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw
- FinLoRA QLoRA 文档：https://finlora.readthedocs.io/en/latest/lora_methods/qlora.html
- sanowl QLoRA 解析：https://sanowl.github.io/qlora.html
