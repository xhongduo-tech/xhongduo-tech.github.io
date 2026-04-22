## 核心结论

Layer-wise 学习率是一种微调策略：把模型参数按层分组，给不同层设置不同学习率。学习率是优化器每一步修改参数的步长，步长越大，参数变化越明显。

新手版解释是：预训练模型的底层更像通用特征提取器，顶层更像任务适配器，所以微调时不应该一视同仁。底层用小学习率，尽量保留已有知识；顶层用大学习率，更快适应新任务。

常用形式是：

$$
\eta_d = \eta_{top} \cdot \rho^d,\quad 0 < \rho < 1
$$

其中 $d=0$ 表示最上层，$d=L-1$ 表示最底层。参数更新为：

$$
\theta_d \leftarrow \theta_d - \eta_d \cdot \nabla_{\theta_d}J(\theta)
$$

| 层位置 | 角色 | 学习率策略 |
|---|---|---|
| 顶层 / 分类头 | 直接适配下游任务 | 较大 lr |
| 中间层 | 连接通用表示和任务表示 | 中等 lr |
| 底层 / embedding | 保留通用语言或视觉特征 | 较小 lr |

在 BERT fine-tuning 中，常见做法是让底层学习率约为顶层的 $0.1$ 到 $0.5$ 倍。这个比例不是定律，而是经验范围：数据越少、任务越接近预训练任务，底层越应该保守；数据越多、任务差异越大，底层可以更积极更新。

---

## 问题定义与边界

Layer-wise 学习率主要解决的是预训练模型微调阶段的优化问题，不是从零训练模型时的通用必选项。微调是指拿已经训练好的模型，在新任务数据上继续训练。它的目标通常不是重新学习所有能力，而是在保留原有表示能力的基础上适配新任务。

它要缓解两个问题：

| 问题 | 白话解释 | Layer-wise lr 的作用 |
|---|---|---|
| 灾难性遗忘 | 模型在新数据上训练后，把预训练阶段学到的通用能力破坏了 | 降低底层更新幅度 |
| 微调不稳定 | 同一配置下不同随机种子效果波动较大 | 让参数更新更有层次 |
| 小样本过拟合 | 数据很少时，模型很快记住训练集 | 限制底层大幅变化 |

需要区分几个相近概念：

| 名称 | 解决什么 | 是否按层设置 lr |
|---|---|---|
| Layer-wise lr | 不同层用不同学习率 | 是 |
| Freeze | 冻结部分参数，不更新 | 不是，冻结层 lr 等价于 0 |
| Warmup | 训练初期把学习率从小逐渐升高 | 不是，通常作用于全局 lr |
| Weight decay | 限制参数变大，起正则化作用 | 不是，控制的是权重惩罚 |
| Scheduler | 随训练步数调整学习率 | 不一定按层 |

真实工程例子：用 BERT 做文本分类。如果只训练最后的分类头，统一学习率通常够用，因为主体模型没有更新。如果要微调整个 BERT，底层 embedding 和前几层 Transformer 若用过大学习率，可能破坏原本的语言表示；这时分层学习率更有价值。

适用边界如下：

| 场景 | 是否适合 |
|---|---|
| BERT、ViT、ResNet 等预训练模型全量微调 | 适合 |
| 小样本分类、检索、领域迁移 | 适合 |
| 只训练 LoRA、adapter 或分类头 | 收益有限 |
| 从零训练一个小模型 | 通常不是优先项 |
| 模型层级关系不清晰 | 分组成本较高 |

---

## 核心机制与推导

先明确层号方向。这里定义 $d=0$ 是最上层，最靠近输出；$d=L-1$ 是最底层，最靠近输入。若总共有 $L$ 层，学习率按以下方式递减：

$$
\eta_d = \eta_{top} \cdot \rho^d
$$

因为 $0 < \rho < 1$，所以 $d$ 越大，$\rho^d$ 越小，学习率越小。也就是说，越靠近底层，更新越保守。

玩具例子：假设有 4 层，顶层学习率 $\eta_{top}=2e-5$，衰减系数 $\rho=0.5$。

| d | 层位置 | 学习率 |
|---:|---|---:|
| 0 | 顶层 | 2e-5 |
| 1 | 次顶层 | 1e-5 |
| 2 | 次底层 | 5e-6 |
| 3 | 底层 | 2.5e-6 |

最底层是顶层的：

$$
\frac{2.5e-6}{2e-5}=0.125=\frac{1}{8}
$$

这就是“差异化更新”的具体含义：不是所有参数一起变快或变慢，而是不同层用不同步长更新。

为什么这个机制合理？浅层通常学习更通用的表示。语言模型的底层更偏词形、局部语法、基础上下文；视觉模型的底层更偏边缘、纹理、局部形状。这些能力对很多任务都通用，改动过大容易损害迁移能力。深层更靠近输出，包含更多任务相关表示，应该更快适配当前数据分布。

但实际参数变化不只由学习率决定，还受梯度大小影响。真实更新量是：

$$
\Delta\theta_d = -\eta_d \cdot \nabla_{\theta_d}J(\theta)
$$

所以分层学习率控制的是“允许该层走多大步”，不是保证每层参数变化一定按固定比例发生。

---

## 代码实现

实现关键点是：必须把每组参数的实际 `lr` 写入 optimizer 的 `param_groups`。只保存 `lr_scale` 没用，优化器不会自动理解它。

下面是一个可运行的 Python 玩具实现，用简单参数对象模拟按层分组，并用 `assert` 检查层号方向和学习率是否正确。

```python
from dataclasses import dataclass

@dataclass
class Param:
    name: str
    numel: int

def build_layerwise_param_groups(named_params, top_lr=2e-5, decay=0.5):
    """
    约定 layer.0 是最底层，layer.3 是最上层。
    输出时转换为 d=0 顶层、d 越大越底层。
    """
    max_layer = 3
    groups = []

    for name, param in named_params:
        if name.startswith("head."):
            d = 0
            lr = top_lr
            group_name = "head"
        elif name.startswith("layer."):
            layer_id = int(name.split(".")[1])
            d = max_layer - layer_id
            lr = top_lr * (decay ** d)
            group_name = f"layer_{layer_id}"
        else:
            d = max_layer + 1
            lr = top_lr * (decay ** d)
            group_name = "embedding"

        groups.append({
            "name": group_name,
            "params": [param],
            "lr": lr,
            "numel": param.numel,
        })

    return groups

params = [
    ("embedding.weight", Param("embedding.weight", 1000)),
    ("layer.0.weight", Param("layer.0.weight", 200)),
    ("layer.1.weight", Param("layer.1.weight", 200)),
    ("layer.2.weight", Param("layer.2.weight", 200)),
    ("layer.3.weight", Param("layer.3.weight", 200)),
    ("head.weight", Param("head.weight", 20)),
]

groups = build_layerwise_param_groups(params, top_lr=2e-5, decay=0.5)
lr_by_name = {g["params"][0].name: g["lr"] for g in groups}

assert lr_by_name["head.weight"] == 2e-5
assert lr_by_name["layer.3.weight"] == 2e-5
assert lr_by_name["layer.2.weight"] == 1e-5
assert lr_by_name["layer.1.weight"] == 5e-6
assert lr_by_name["layer.0.weight"] == 2.5e-6
assert lr_by_name["embedding.weight"] < lr_by_name["layer.0.weight"]

for g in groups:
    print(g["name"], g["numel"], g["lr"])
```

真实 PyTorch 中结构类似：

```python
# optimizer = torch.optim.AdamW(param_groups)
# param_groups = [
#   {"params": head_params, "lr": 2e-5, "weight_decay": 0.01},
#   {"params": upper_block_params, "lr": 1e-5, "weight_decay": 0.01},
#   {"params": lower_block_params, "lr": 5e-6, "weight_decay": 0.01},
# ]
```

工程上常见分组如下：

| 参数组 | 示例 | lr |
|---|---|---:|
| head | 分类头、投影头 | 2e-5 |
| upper blocks | BERT 后几层、ViT 后几层 | 1e-5 到 2e-5 |
| middle blocks | 中间 Transformer blocks | 5e-6 到 1e-5 |
| lower blocks / embedding | embedding、前几层 | 1e-6 到 5e-6 |

检查逻辑必须做三件事：打印层名、打印参数量、打印实际 `lr`。如果发现 embedding 的 `lr` 比 head 大，基本就是层号方向写反了。

---

## 工程权衡与常见坑

Layer-wise 学习率不是越小越稳。底层学习率压得过低，模型主体几乎不动，最后只有分类头在适配任务。训练集指标可能上升，但验证集收益不明显，因为底层表示没有真正适配新领域。

失败场景：用通用 BERT 微调医学文本分类。分类头学习率是 `2e-5`，底层被压到 `1e-8`。训练后分类头能记住少量标签模式，但底层词汇和上下文表示没有适配医学术语，验证集提升很小。这不是模型能力不够，而是底层更新被限制得太死。

常见错误清单：

| 错误 | 后果 | 修正 |
|---|---|---|
| 层号方向写反 | 底层 lr 最大，预训练表示被破坏 | 明确 d=0 是顶层还是底层 |
| 只保存 `lr_scale` | optimizer 实际仍用统一 lr | 检查 `param_group["lr"]` |
| 混淆 lr 和 weight decay | 调参结论失真 | 分别记录两类超参 |
| 没有统一 lr 基线 | 无法判断是否真的有效 | 先跑 baseline |
| decay 过小 | 底层几乎冻结，欠拟合 | 尝试 0.7、0.8、0.9 |
| decay 过大 | 各层 lr 差异太小 | 与统一 lr 效果接近 |

排查表：

| 现象 | 可能原因 | 修正方式 |
|---|---|---|
| 训练初期 loss 剧烈震荡 | 顶层 lr 过大或 warmup 不足 | 降低 top lr，加 warmup |
| 验证集无提升 | 底层 lr 太小或任务数据不足 | 增大 decay，比较冻结方案 |
| 小样本很快过拟合 | 顶层 lr 过大，正则不足 | 降低 head lr，加 weight decay |
| 不同随机种子差异大 | 微调不稳定 | 增加 warmup，降低全局 lr |

推荐实验流程是：先跑统一学习率基线，例如 `2e-5`；再跑 layer-wise lr，例如 `top_lr=2e-5, decay=0.8`；最后只改一个变量做对比。不要同时改 batch size、epoch、warmup、weight decay，否则无法判断收益来自哪里。

---

## 替代方案与适用边界

Layer-wise 学习率不是唯一方案。它适合“模型主体需要更新，但又不能大幅破坏预训练表示”的场景。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 统一 lr | 简单、容易复现 | 对所有层一视同仁 | 任务简单、数据充足 |
| 冻结底层 | 稳定、省显存 | 表示适配能力弱 | 数据很少，只训分类头 |
| 逐步解冻 | 训练更平滑 | 流程更复杂 | 小样本迁移 |
| Warmup + linear decay | 提升训练稳定性 | 不解决层间差异 | 大多数 Transformer 微调 |
| Layer-wise lr | 兼顾保留和适配 | 需要正确分层 | BERT、ViT、ResNet 全量微调 |
| AutoLR | 自动调层间 lr | 实现和复现实验成本更高 | 需要自动化调参的研究或复杂工程 |

文本分类微调例子：如果是情感分类，数据和通用语料差异不大，统一 lr 加 warmup 可能已经够用。如果是法律、医学、金融文本，小样本且术语密集，分层 lr 更有意义，因为模型既要保留语言能力，又要调整领域表示。

检索或视觉微调例子：图像检索任务常需要让 embedding 空间适配新的相似度标准。底层边缘、纹理特征仍然通用，顶层语义表示需要明显调整，因此 layer-wise decay 更常见。MAE 的微调脚本中就提供了 `layer_decay` 参数，常见值如 `0.65` 或 `0.75`。

关于 UDO：如果这里指“Universal Deep Learning”提供自动层间学习率分配算法，需要谨慎表述。当前更容易核验的公开方案是 AutoLR，它是自动化层间学习率调参方法，并包含 layer-wise pruning 思路；不要把 UDO 写成已经有公开论文和源码支持的确定结论，除非能提供可核验来源。

---

## 参考资料

1. [Universal Language Model Fine-tuning for Text Classification](https://s10251.pcdn.co/pdf/2018-howard-ulm.pdf)：ULMFiT 提出 discriminative fine-tuning 和 gradual unfreezing，是分层微调思想的重要实践来源。
2. [TensorFlow 官方 BERT 微调指南](https://www.tensorflow.org/official_models/fine_tuning_bert)：展示 BERT 微调中全局学习率、warmup 和 decay 的工程基线。
3. [MAE 官方微调说明](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md)：提供 `--layer_decay` 参数，是视觉预训练模型微调的工程实例。
4. [Measuring the Instability of Fine-Tuning](https://arxiv.org/pdf/2302.07778)：讨论微调不稳定性，并涉及 layer-wise learning rate decay 的稳定性收益。
5. [AutoLR: Layer-wise Pruning and Auto-tuning of Learning Rates in Fine-tuning of Deep Networks](https://ojs.aaai.org/index.php/AAAI/article/view/16350)：提出自动化层间学习率调参和层级剪枝方法，可作为 AutoLR 方向参考。
