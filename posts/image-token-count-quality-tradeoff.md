## 核心结论

图像 token 数量是视觉 Transformer 中最直接的成本控制变量。这里的 token 可以先理解成“模型看到的一块图像单元”。如果输入分辨率为 $H \times W$，patch 大小为 $p \times p$，忽略 `CLS` 等附加 token 后，图像 token 数量近似为：

$$
N = (H/p)\cdot(W/p)
$$

这个公式直接决定两件事：

1. patch 越小，token 越多，模型保留的空间细节越多；
2. 分辨率越高，token 也越多，计算量、显存占用和推理延迟都会上升。

真正昂贵的部分通常在 self-attention。它的直观含义是：每个 token 都要和其他所有 token 建立一次关系，因此其主要计算代价可近似写成：

$$
\text{FLOPs} \propto N^2D
$$

其中 $D$ 是隐藏维度。关键不是常数项，而是平方项 $N^2$。这意味着 token 从 196 增加到 256，看起来只多了 60 个，但 attention 相关计算会从 $196^2=38416$ 增加到 $256^2=65536$；如果进一步升到 576，则变成 $576^2=331776$，已经不是线性增加，而是数量级变化。

因此，图像 token 数量与质量的权衡，本质不是“越多越好”，而是“在任务所需的细粒度信息和系统能够承担的成本之间做匹配”。粗粒度分类、边缘设备实时推理、服务器离线高精度分析，这三类场景的最优 token 配置通常并不相同。

---

## 问题定义与边界

本文讨论的是视觉 Transformer 及其多模态变体中的图像 token 数量问题，不讨论卷积网络内部的特征图压缩，也不讨论视频中的时间维 token。

核心问题可以写成一句话：

> 在模型主干基本固定的前提下，图像应该被切成多少个 token，才能在给定预算内保留足够的视觉信息？

这里的“质量”不是单一指标，而是多个能力的组合结果：

| 维度 | 含义 | token 更多时通常的变化 |
| --- | --- | --- |
| 细粒度识别 | 是否更容易看清小目标、边缘、纹理、字符 | 更强 |
| 空间定位 | 是否更容易区分“哪个局部区域包含关键信息” | 更强 |
| 上下文负担 | 序列更长后，后续层处理是否更重 | 更高 |
| 训练与推理成本 | 显存、FLOPs、延迟、吞吐压力 | 更高 |
| 工程稳定性 | 是否更容易 OOM、超时、批量吞吐下降 | 更差 |

先记住几组常见配置：

| 配置 | 网格 | Token 数 | 说明 |
| --- | --- | --- | --- |
| `224×224 / patch16` | `14×14` | 196 | 经典 ViT-B/16 起点 |
| `224×224 / patch14` | `16×16` | 256 | patch 更细，局部细节更充分 |
| `384×384 / patch32` | `12×12` | 144 | 分辨率更高，但 patch 较粗 |
| `384×384 / patch16` | `24×24` | 576 | 高分辨率高 token，成本明显增加 |
| `224×224 / patch1` | `224×224` | 50,176 | 近似像素级 token，工程上极贵 |

有两个边界需要先说明。

第一，token 变多不代表任务一定变好。如果任务只需要回答“是猫还是狗”，很多高频细节并不能稳定转化为准确率提升。更细的 patch 只是在提供更多信息，不是在保证更多有效信息。

第二，token 变多后，模型参数不会自动适配。尤其是把 `224×224 / patch16` 的预训练模型切换到 `384×384 / patch16` 时，位置编码必须进行 2D 插值。否则模型虽然可以前向运行，但新增位置缺少可用表示，效果通常会异常。

先看一个最小例子：

- `224×224`，`patch=16`，得到 $14\times14=196$ 个 token；
- 同样输入，`patch=14`，得到 $16\times16=256$ 个 token；
- 改成 `384×384` 且 `patch=16`，得到 $24\times24=576$ 个 token。

这三个数字 196、256、576，已经概括了很多真实模型配置中的主要差异。

为了帮助新手建立直觉，可以把它理解为“看图的网格密度”：

| 方式 | 直观理解 | 结果 |
| --- | --- | --- |
| 大 patch、少 token | 用大格子粗看整张图 | 全局快，但细节容易丢 |
| 小 patch、多 token | 用小格子细看整张图 | 细节强，但成本迅速上升 |

---

## 核心机制与推导

难点不在公式本身，而在于把“token 数变化”准确映射到“代价变化”和“信息保留变化”。

### 1. token 数量如何产生

视觉 Transformer 的第一步通常是 patchify。即把输入图像切成固定大小的小方块，每个方块映射为一个向量，最终整张图被转成一个 token 序列。

公式为：

$$
N=(H/p)\cdot(W/p)
$$

如果输入是正方形，即 $H=W=S$，则可以简化为：

$$
N=(S/p)^2
$$

因此只要 patch 边长减半，token 数通常就会增加到原来的 4 倍。

例如：

| 输入尺寸 | patch | 网格 | token 数 |
| --- | --- | --- | --- |
| `224×224` | 32 | `7×7` | 49 |
| `224×224` | 16 | `14×14` | 196 |
| `224×224` | 8 | `28×28` | 784 |

这个表说明一个基本事实：patch 缩小一半，不是“多一点 token”，而是会把序列长度成倍拉长。

### 2. 为什么成本增长比 token 增长更快

Transformer 的 attention 会构造 token 两两之间的关系矩阵，因此其核心张量规模是 $N\times N$。直观地说，196 个 token 不是做 196 次比较，而是做 $196\times196$ 规模的全连接比较。

单层 attention 的主要代价常近似写成：

$$
\text{Attention FLOPs} \propto N^2D
$$

如果考虑多头注意力，常见写法仍可抽象成同样结论：对固定模型宽度和层数，序列长度 $N$ 是最危险的增长项。

进一步看相对成本更直观。设 196 token 为基线，则：

$$
\text{相对成本} \approx \left(\frac{N}{196}\right)^2
$$

对应数值如下：

| Token 数 $N$ | Attention 规模 $N^2$ | 相对 196 token 成本 |
| --- | --- | --- |
| 64 | 4,096 | 0.11x |
| 144 | 20,736 | 0.54x |
| 196 | 38,416 | 1.00x |
| 256 | 65,536 | 1.71x |
| 576 | 331,776 | 8.64x |
| 784 | 614,656 | 16.00x |

这张表说明三件事：

1. 从 196 到 256，不是“小改动”，而是接近 1.7 倍的 attention 负担；
2. 从 196 到 576，attention 相关计算已经接近 8.6 倍；
3. 一旦进入 700+ token 区间，哪怕模型参数没变，系统侧也会明显吃紧。

补充一点：完整 Transformer 块并不只有 attention，还有 MLP、LayerNorm、残差连接等部分。若做粗略估算，可写成：

$$
\text{Block FLOPs} \approx c_1N^2D + c_2ND^2
$$

其中 $c_1,c_2$ 为常数。这个式子说明，当 $N$ 较小时，$ND^2$ 也可能占很大比例；但随着序列变长，$N^2D$ 很快会成为主导项。这也是为什么高分辨率输入下，attention 往往最先成为瓶颈。

### 3. 为什么更多 token 能提升细粒度能力

patch 越小，每个 token 覆盖的像素区域越少，局部边缘、角点、细纹理、字符结构就越不容易被平均掉。可以把它理解为：模型用更密的采样网格在看图。

这类收益在以下任务中通常更明显：

| 任务 | 原因 |
| --- | --- |
| 小目标分类 | 目标本身只占少量像素，粗 patch 容易直接淹没目标 |
| OCR | 字符笔画密度高，patch 太大时会丢失关键形状 |
| 文档理解 | 表格线、字号差异、局部布局都依赖细网格 |
| 缺陷检测 | 裂纹、脏点、孔洞等异常通常是局部微小模式 |
| 视觉问答 | 问题经常对应图像中的局部区域而非全局语义 |

但收益不是无限增长，原因至少有三类：

1. 标签是粗粒度的。若任务只预测一个大类标签，过多细节未必有用。
2. 序列更长会带来更多噪声和优化压力。更多 token 不只是更多信息，也可能是更多冗余。
3. 主干容量有限。如果模型宽度、深度或训练数据量不足，新增 token 可能无法被充分利用。

一个常见误区是把“分辨率更高”直接理解成“效果一定更好”。更准确的说法应是：更高分辨率和更小 patch 只是在提供更多原始局部信息，能否转化为性能提升，取决于任务是否需要这些信息，以及模型是否有能力消化它们。

### 4. 动态 token 分配为什么有意义

固定 token 策略默认所有图片都值得同样多的计算，这在真实数据中通常不成立。

- 一张背景干净、主体居中的商品图，可能 144 token 已经足够；
- 一张包含印章、小字、表格、编号的文档图，可能 576 token 仍然紧张；
- 一张监控截图中只有少量区域包含目标，均匀细切会把大量 token 浪费在无效背景上。

因此研究和工程里都会出现两类思路：

| 路线 | 思路 | 核心目标 |
| --- | --- | --- |
| 混合分辨率 tokenization | 重要区域切细，次要区域切粗 | 在固定预算下保留关键局部 |
| 动态 token 分配 / 剪枝 | 推理过程中决定哪些 token 保留 | 减少无信息 token 的后续计算 |

它们解决的是同一个问题：不要把预算平均浪费在整张图上，而要把 token 集中分配给高价值区域。

一个现实例子是多模态文档问答。页面中的标题、表头、金额区域、页脚说明，信息密度非常不均匀。如果整页统一用粗 patch，模型可能看不到小字；如果整页统一用高分辨率 token，序列长度又会显著膨胀，挤占文本上下文并增加推理成本。工程上更常见的做法是先用低分辨率获得全局布局，再对高价值区域做局部放大或细分 patch。这就是动态 token 分配在真实系统中的直接对应。

---

## 代码实现

下面给出两个可运行的 Python 片段。第一个用于计算常见配置的 token 数、网格尺寸和 attention 相对成本；第二个用于演示视觉 Transformer 中常见的 2D 位置编码插值。两个例子都可以直接保存为 `.py` 文件运行。

### 1. 计算 token 数与相对成本

```python
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    height: int
    width: int
    patch: int

    def grid_size(self) -> tuple[int, int]:
        if self.height % self.patch != 0 or self.width % self.patch != 0:
            raise ValueError(
                f"Input ({self.height}, {self.width}) is not divisible by patch {self.patch}"
            )
        return self.height // self.patch, self.width // self.patch

    def num_tokens(self) -> int:
        gh, gw = self.grid_size()
        return gh * gw


def relative_attention_cost(base_tokens: int, new_tokens: int) -> float:
    if base_tokens <= 0 or new_tokens <= 0:
        raise ValueError("token count must be positive")
    return (new_tokens ** 2) / (base_tokens ** 2)


def describe_config(name: str, cfg: ViTConfig, base_tokens: int) -> dict:
    gh, gw = cfg.grid_size()
    n = cfg.num_tokens()
    return {
        "name": name,
        "image": f"{cfg.height}x{cfg.width}",
        "patch": cfg.patch,
        "grid": f"{gh}x{gw}",
        "tokens": n,
        "rel_cost": round(relative_attention_cost(base_tokens, n), 2),
    }


def main() -> None:
    configs = {
        "224/16": ViTConfig(224, 224, 16),
        "224/14": ViTConfig(224, 224, 14),
        "384/32": ViTConfig(384, 384, 32),
        "384/16": ViTConfig(384, 384, 16),
        "224/8": ViTConfig(224, 224, 8),
    }

    base_tokens = configs["224/16"].num_tokens()
    rows = [describe_config(name, cfg, base_tokens) for name, cfg in configs.items()]

    print(f"{'name':<8} {'image':<10} {'patch':<6} {'grid':<8} {'tokens':<8} {'rel_cost':<8}")
    for row in rows:
        print(
            f"{row['name']:<8} {row['image']:<10} {row['patch']:<6} "
            f"{row['grid']:<8} {row['tokens']:<8} {row['rel_cost']:<8}"
        )

    assert configs["224/16"].num_tokens() == 196
    assert configs["224/14"].num_tokens() == 256
    assert configs["384/32"].num_tokens() == 144
    assert configs["384/16"].num_tokens() == 576
    assert math.isclose(relative_attention_cost(196, 256), 1.7067888379841732)
    assert math.isclose(relative_attention_cost(196, 576), 8.636401915868387)


if __name__ == "__main__":
    main()
```

运行后会得到类似输出：

```text
name     image      patch  grid     tokens   rel_cost
224/16   224x224    16     14x14    196      1.0
224/14   224x224    14     16x16    256      1.71
384/32   384x384    32     12x12    144      0.54
384/16   384x384    16     24x24    576      8.64
224/8    224x224    8      28x28    784      16.0
```

这个例子有三个用途：

1. 新手可以直接看到 144、196、256、576 是怎样从输入尺寸和 patch 大小算出来的；
2. 工程上可以先做粗预算，判断某个配置是否值得进入实验；
3. 读表时能立刻建立“token 增长不是线性成本增长”的直觉。

### 2. 位置编码插值的最小实现

位置编码的作用是告诉模型“当前 token 在图像中的什么空间位置”。当 patch 网格从 `14×14` 变成 `24×24` 时，原先训练好的位置编码不能直接照搬，必须先恢复成二维网格再插值到新网格。

下面是一个最小可运行实现：

```python
import math
import torch
import torch.nn.functional as F


def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    new_patch_tokens: int,
    num_prefix_tokens: int = 1,
    mode: str = "bicubic",
) -> torch.Tensor:
    """
    pos_embed shape: [1, old_patch_tokens + num_prefix_tokens, dim]
    return shape:    [1, new_patch_tokens + num_prefix_tokens, dim]
    """
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError("pos_embed must have shape [1, seq_len, dim]")

    old_patch_tokens = pos_embed.shape[1] - num_prefix_tokens
    old_size = int(math.sqrt(old_patch_tokens))
    new_size = int(math.sqrt(new_patch_tokens))

    if old_size * old_size != old_patch_tokens:
        raise ValueError("old patch token count is not a square number")
    if new_size * new_size != new_patch_tokens:
        raise ValueError("new patch token count is not a square number")

    if old_patch_tokens == new_patch_tokens:
        return pos_embed

    prefix = pos_embed[:, :num_prefix_tokens]          # [1, prefix, dim]
    patch_embed = pos_embed[:, num_prefix_tokens:]     # [1, old_patch_tokens, dim]

    # [1, old_patch_tokens, dim] -> [1, dim, old_h, old_w]
    patch_embed = patch_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)

    patch_embed = F.interpolate(
        patch_embed.float(),
        size=(new_size, new_size),
        mode=mode,
        align_corners=False,
    )

    # [1, dim, new_h, new_w] -> [1, new_patch_tokens, dim]
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_patch_tokens, -1)

    return torch.cat([prefix, patch_embed], dim=1)


def main() -> None:
    torch.manual_seed(0)

    # 经典场景：1 个 CLS token + 14x14 patch token = 197
    old = torch.randn(1, 197, 768)
    new = interpolate_pos_embed(old, new_patch_tokens=576, num_prefix_tokens=1)

    print("old shape:", tuple(old.shape))
    print("new shape:", tuple(new.shape))

    assert old.shape == (1, 197, 768)
    assert new.shape == (1, 577, 768)
    assert torch.allclose(new[:, :1], old[:, :1])  # prefix token 不变


if __name__ == "__main__":
    main()
```

这个实现对应最常见的 fine-tune 迁移路径：

| 训练阶段 | 输入配置 | patch token | 总序列长度（含 `CLS`） |
| --- | --- | --- | --- |
| 预训练 | `224×224 / patch16` | 196 | 197 |
| 微调 | `384×384 / patch16` | 576 | 577 |

新手需要特别注意三点：

1. 插值的是 patch 位置编码，不是图像像素本身；
2. `CLS` 等 prefix token 通常不参与二维插值，应直接保留；
3. 只有当 patch token 数是平方数时，才能自然恢复为二维网格。

如果不做这一步，模型虽然多数情况下还能前向运行，但效果往往会异常。这不是小误差，而是配置迁移中的高频错误。

---

## 工程权衡与常见坑

工程上的难点不是会写公式，而是知道应该把预算花在哪里，以及知道哪些地方最容易踩坑。

### 1. 常见配置并不是越大越好

在普通图像分类任务中，`144` 到 `196` token 往往已经处于较好的性价比区间。`256` token 常用于更细粒度的场景，`576` token 则更适合离线分析、高精度文档理解或局部信息极密集的任务。

可以先用下面这张表做一阶判断：

| Token 区间 | 常见场景 | 优点 | 主要风险 |
| --- | --- | --- | --- |
| 49-64 | 极低延迟、轻量边缘设备 | 成本低、吞吐高 | 局部细节明显不足 |
| 144-196 | 通用分类、常规视觉编码 | 精度与成本较平衡 | 对小目标和小字不够强 |
| 256 | 细粒度分类、OCR、局部问答 | 细节更充分 | 显存和延迟明显上升 |
| 576 及以上 | 离线分析、文档解析、高精度任务 | 细粒度能力更强 | attention 成本快速膨胀 |

一个简单经验是：

- 要先做稳妥基线，用 196 token；
- 要压缩延迟，先尝试 144 token 或动态预算；
- 要增强局部细节，优先尝试 256 token；
- 只有任务确实依赖大量局部信息时，再考虑 576 及以上。

### 2. 位置编码不插值，是最常见坑

很多人第一次把输入从 `224` 改到 `384` 时，只修改了图像 resize，没有修改位置编码。模型可能不会立即报错，但性能会明显不稳定。

原因不是“更高分辨率破坏了模型”，而是原先的位置编码只覆盖 `14×14` 网格，新增的空间位置没有合理初始化。对 Transformer 来说，这相当于让模型在陌生坐标系里工作。

正确做法是：

1. 分离 `CLS` 等 prefix token 与 patch token；
2. 将 patch token 的位置编码从 1D 序列恢复成 2D 网格；
3. 对二维网格做双三次插值；
4. 再展平成新序列并拼回 prefix token。

这不是可选优化，而是分辨率迁移中的基础步骤。

### 3. 只看 token 数，不看任务信息密度，也是坑

同样是 576 token，在不同任务中的价值差异会很大：

| 任务 | 576 token 的典型评价 |
| --- | --- |
| 自然图像大类分类 | 可能偏浪费 |
| 商品细粒度识别 | 可能有收益 |
| 表格解析 | 往往合理甚至仍偏少 |
| OCR 小字识别 | 通常有明显帮助 |

问题不在 token 数本身，而在信息密度。可以把信息密度理解为“单位面积中有多少对结果有用的细节”。一张背景纯净的人像图和一页密集表格，虽然像素尺寸相同，但对 token 的需求完全不同。

### 4. 极高 token 数可能继续提升精度，但系统代价常先失控

patch 越细，模型理论上能观察到越多局部结构，因此精度有时还会继续上升。但工程上不能只看任务指标，还必须同时检查以下约束：

| 约束项 | 实际问题 |
| --- | --- |
| 显存 | 单卡是否能容纳目标 batch size |
| 吞吐 | 每秒处理样本数是否满足业务量 |
| 延迟 | 单请求 p95 / p99 是否满足 SLA |
| 稳定性 | 是否频繁 OOM 或触发降批量 |
| 成本 | GPU 小时成本是否可接受 |

例如在线商品审核系统每天可能处理百万级图片。即使 576 token 相对 196 token 能带来几个点的收益，也未必能上线，因为整体 GPU 成本和响应延迟可能显著增加。相反，在离线质检、法务证据分析或高价值文档抽取中，更高 token 数往往是可接受的。

### 5. 动态策略的坑在实现复杂度，而不只在论文指标

动态 token 分配、token 剪枝、级联预算这些方法在论文里通常能减少理论 FLOPs，但上线时要额外检查：

| 风险点 | 说明 |
| --- | --- |
| 动态 shape | 部署框架可能难以充分优化 |
| 批处理效率 | 不同样本长度不同，容易拖慢批量吞吐 |
| masking 正确性 | 剪枝后 attention mask 很容易写错 |
| wall-clock 偏差 | 理论 FLOPs 降低不代表真实延迟同比下降 |
| 置信度校准 | 级联系统若校准差，容易把难样本提前放走 |

因此“论文更省算力”不自动等于“线上更快、更稳、更便宜”。

---

## 替代方案与适用边界

固定 patch、固定分辨率只是基础方案。真正实用的替代路线，核心都在于一句话：不要把同样多的 token 平均分配给所有区域。

### 1. 混合分辨率 tokenization

这类方法的代表思路，是根据显著性或空间结构把重要区域切得更细，把不重要区域切得更粗。四叉树式划分就是典型实现方式。

它解决的问题很直接：同一张图里，不同区域的信息密度本来就不同，没有必要强制使用统一 patch 粒度。

优点：

- 更接近真实图像的信息分布；
- 在相同 token 预算下，更容易保住关键细节；
- 对已有主干模型通常比较友好，不一定需要完全重写架构。

适用边界：

- 图像存在明显主次区域；
- 任务依赖局部细节；
- 系统预算不允许全图统一高分辨率。

典型例子是文档图像、遥感图像和医疗图像，这些输入往往都具有“局部很重要、大片区域很普通”的结构特点。

### 2. 动态 token 剪枝

这类方法的思路是在中间层判断哪些 token 价值较低，然后逐步移除或弱化这些 token，避免它们继续参与后续计算。

它的关键点不在“输入时就少切”，而在“先保留，再筛掉不重要的 token”。这使它比静态裁剪更灵活，因为模型可以根据当前样本的内容动态决定保留哪些区域。

优点：

- 前几层先完整感知，再逐步降低成本；
- 难样本和易样本可以走不同计算路径；
- 在部分任务上，比一开始就减少 token 更稳妥。

适用边界：

- 推理成本敏感；
- 部署链路允许动态 masking 或分支控制；
- 团队能接受更复杂的调试和验证流程。

### 3. 级联式动态预算

这类方法通常先用低 token 配置做一次便宜推理，如果模型置信度足够高，就直接输出；只有不确定样本才进入更贵的高分辨率或高 token 路径。

它特别适合线上服务，因为真实请求分布通常不均匀，简单样本往往占多数。把高成本路径只留给难样本，平均成本通常会明显下降。

这种方法的核心前提是：置信度要足够可信，否则系统会把本该进入高成本路径的难样本误判成简单样本。

下面给出一个对比表：

| 方法 | Token 策略 | 适用场景 | 主要代价 |
| --- | --- | --- | --- |
| 固定 patch + 固定分辨率 | 所有图统一 token 数 | 基线系统、实现最简单 | 冗余 token 较多 |
| 混合分辨率 tokenization | 关键区域细切，其他区域粗切 | 空间信息分布不均的图像 | 路由与预处理复杂 |
| 动态 token 剪枝 | 中间层逐步删掉低价值 token | 推理成本敏感系统 | 部署与验证复杂 |
| 级联式动态预算 | 先便宜推理，再按需升级 | 难度分布不均的线上服务 | 需要置信度校准 |

如果只给一个实际决策建议，可以用下面的经验法则：

- 先建立基线，用 196 token；
- 延迟压力大，先试 144 token 或级联系统；
- 局部细节重要，先试 256 token；
- 只有在任务确实依赖高密度局部信息，且资源预算明确允许时，再考虑 576 及以上；
- 若业务输入的信息密度高度不均匀，优先考虑动态预算或混合分辨率，而不是简单把全图 token 全部抬高。

---

## 参考资料

为了让参考资料更容易使用，下面按“用途”分组。阅读顺序建议是：先看基础原理，再看工程插值，再看动态策略。

| 类型 | 资料 | 作用 |
| --- | --- | --- |
| 基础入门 | How To Compute The Token Consumption Of Vision Transformers (ML Digest): https://ml-digest.com/computing-vision-transformer-tokens/ | 快速建立 token 计算直觉 |
| 基础入门 | ViT patch-size ablation + interpolation warning (TildAlice): https://tildalice.io/vit-vision-transformer-paper-review/ | 理解 patch size 变化与位置编码插值 |
| 理论扩展 | Scaling Laws in Patchification (PMLR 2025): https://proceedings.mlr.press/v267/wang25ed.html | 理解 patch 缩小与性能扩展趋势 |
| 混合分辨率 | Vision Transformers With Mixed-Resolution Tokenization / Quadformer (CVPRW 2023): https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Ronen_Vision_Transformers_With_Mixed-Resolution_Tokenization_CVPRW_2023_paper.pdf | 了解重要区域细切的代表方案 |
| 动态预算 | Dynamic Vision Transformer (NeurIPS 2021): https://proceedings.neurips.cc/paper/2021/file/64517d8435994992e682b3e4aa0a0661-Paper.pdf | 了解按样本动态调整计算预算 |
| 动态剪枝 | DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification (NeurIPS 2021): https://papers.neurips.cc/paper_files/paper/2021/file/747d3443e319a22747fbb873e8b2f9f2-Paper.pdf | 了解中间层 token 稀疏化 |

如果只想抓住本文的主线，可以按下面顺序理解：

1. 用 $N=(H/p)\cdot(W/p)$ 算清 token 数；
2. 用 $\text{FLOPs} \propto N^2D$ 判断注意力成本是否可接受；
3. 用任务的信息密度决定是否真的需要更多 token；
4. 若固定 token 太贵，优先考虑动态分配，而不是盲目降低精度目标。
