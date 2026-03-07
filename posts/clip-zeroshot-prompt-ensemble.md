## 核心结论

CLIP 的 zero-shot 分类，本质上不是“直接拿类别名去匹配图片”，而是“把类别名写成一句自然语言，再让文本编码器把这句话变成分类器权重”。这里的 `prompt` 可以理解成“给类别补上下文的句子模板”。

Prompt Ensemble 的核心作用，是用多个模板而不是单个模板来描述同一个类别，然后对这些文本向量求平均并归一化。这样做的收益不是增加模型参数，而是降低单条模板的措辞偏差，让类向量在语义空间里更稳定。对零样本分类来说，这种稳定性通常直接转化为更高的 top-1 精度和更好的跨分布鲁棒性。

新手可以先抓住一句话：CLIP 在测试时没有重新训练一个线性分类头，而是把“文本描述后的类别向量”直接当成分类头来用。也就是说，分类器参数不是从监督标签里学出来的，而是由文本编码器即时生成的。

假设要识别 `cat`。如果只写 `cat` 或 `a photo of a cat`，模型会被一种固定上下文绑定；如果同时写：

- `a photo of a cat`
- `a blurry photo of a cat`
- `a black and white photo of a cat`
- `a drawing of a cat`

再把这些句子的向量平均，得到的“猫”类向量会更接近各种真实输入，而不只接近某一种拍摄条件或表达方式。

下面是单模板和 Prompt Ensemble 的差异概览：

| 方案 | 类向量来源 | 对措辞是否敏感 | 泛化稳定性 | 离线成本 | 推理时成本 |
| --- | --- | --- | --- | --- | --- |
| 单模板 | 1 条 prompt | 高 | 一般 | 低 | 低 |
| Prompt Ensemble | 多条 prompt 平均 | 低 | 更高 | 中 | 低 |
| 可学习 Prompt | 训练得到上下文 token | 低到中 | 依赖训练分布 | 高 | 低 |

经验上，多模板平均通常比单模板更稳定，尤其是在 ImageNet 这类类别多、视觉变化大的任务里，常见收益是几个百分点的 top-1 提升。这个提升不是来自更复杂的在线推理，而是来自更可靠的文本侧分类器构造。

再补一个容易混淆的点。Prompt Ensemble 并不是“让多个 prompt 分别投票后多数表决”，而是先在文本特征空间里求平均，把多个 prompt 折叠成一个类别中心。最终线上仍然只有一个 `cat` 向量、一个 `dog` 向量、一个 `car` 向量。

---

## 问题定义与边界

Zero-shot 分类，指模型在训练时没有用当前任务的类别标签训练一个专门分类头，但在测试时仍然能根据类别名称完成分类。这里的“零样本”不是没有见过图片，而是“没有针对这组目标类别做专门监督训练”。

对 CLIP 而言，问题可以定义为：

1. 输入一张图像 $x$
2. 用图像编码器得到图像向量
3. 对每个候选类别 $y$，把类别名放进 prompt 模板里
4. 用文本编码器得到类别文本向量
5. 比较图像向量和各类别向量的相似度
6. 通过 softmax 输出类别概率

如果把它说得更白话一点，ImageNet 的 zero-shot 分类器可以看成一本“类别词典”：每个类别不是一行训练出来的参数，而是一组由句子生成的向量；这些向量平均后，就成了该类别的分类器权重。

本文讨论的 Prompt Ensemble 有明确边界：

- 只讨论测试阶段的模板集合平均
- 不涉及反向传播更新 CLIP 参数
- 不涉及少样本监督下的分类头训练
- 不把 prompt tuning、adapter、LoRA 这类训练式方法混在一起

也就是说，这里说的 Ensemble 是“离线构造更好的文本类向量”，不是“在线学习更好的模型参数”。

最小流程可以写成下面的步骤列表：

1. 输入图像
2. 用 CLIP 图像编码器提取图像向量
3. 对每个类别提前生成多条 prompt
4. 用文本编码器得到多个文本向量并平均
5. 将平均后的类向量缓存
6. 对运行时图像向量和所有类向量做余弦相似度
7. 对相似度乘以温度系数后做 softmax

这个问题设定非常适合服务端部署。原因很简单：文本类向量可以预计算，运行时只需要一次图像编码和一次矩阵乘法。

为了避免概念混淆，下面给出几个术语的最小定义：

| 术语 | 最小定义 | 在本文里的角色 |
| --- | --- | --- |
| `prompt` | 把类别名放进自然语言句子的模板 | 文本侧输入 |
| 文本编码器 | 把 prompt 变成向量的网络 | 生成类向量 |
| 图像编码器 | 把图片变成向量的网络 | 生成查询向量 |
| 类向量 | 某个类别最终用于匹配的文本表示 | 相当于分类器权重 |
| `logit scale` / 温度 | 控制 softmax 锐度的缩放系数 | 放大相似度差异 |

一个具体例子更直观。假设候选类别只有 `cat`、`dog`、`car` 三类，那么 zero-shot 分类并不是训练一个 `3 x D` 的线性层，而是先构造：

- `a photo of a cat`
- `a photo of a dog`
- `a photo of a car`

把这三句话编码成三个 $D$ 维向量，堆起来就得到一个文本分类器矩阵。图像向量和这个矩阵做点积，就得到三个类别的分数。

---

## 核心机制与推导

CLIP 有两个编码器：

- 图像编码器：把图片映射成向量
- 文本编码器：把句子映射成向量

这两个向量会被投到同一个语义空间里。所谓“同一个语义空间”，就是图像向量和文本向量的维度一致、尺度一致、可直接比较方向相似性。

设第 $t$ 个模板为 $\text{prompt}_t(\text{label})$，文本编码器记为 $f_{\text{txt}}$。那么第 $t$ 条 prompt 的向量是：

$$
h_t = f_{\text{txt}}(\text{prompt}_t(\text{label}))
$$

如果一共有 $T$ 个模板，则类别 `label` 的最终类向量定义为：

$$
e_{\text{label}} = \text{Normalize}\left(\frac{1}{T}\sum_{t=1}^{T} h_t\right)
$$

这里的 `Normalize` 指 $L_2$ 归一化，即：

$$
\text{Normalize}(z) = \frac{z}{\|z\|_2}
$$

归一化的作用是只保留方向，不让向量长度影响相似度。因为 CLIP 做分类时更关心“方向是否一致”，而不是“向量有多长”。

对输入图像 $x$，图像向量写成：

$$
v_x = \text{Normalize}(f_{\text{img}}(x))
$$

分类 logit 可写成：

$$
\ell_y(x) = \tau \cdot v_x^\top e_y
$$

由于 $v_x$ 和 $e_y$ 都已归一化，所以点积就等于余弦相似度：

$$
v_x^\top e_y = \cos(v_x, e_y)
$$

因此分类概率为：

$$
P(y|x) = \text{softmax}_y\left(\tau \cdot \cos(v_x, e_y)\right)
$$

其中：

- $\cos(\cdot,\cdot)$ 是余弦相似度，表示两个方向有多接近
- $\tau$ 是温度或 `logit scale`，表示“把相似度差异放大多少”

如果写成矩阵形式，工程上会更清楚。设一共有 $C$ 个类别，每个类别最终有一个 $D$ 维类向量，把它们堆成矩阵：

$$
W =
\begin{bmatrix}
e_1^\top \\
e_2^\top \\
\vdots \\
e_C^\top
\end{bmatrix}
\in \mathbb{R}^{C \times D}
$$

则图像 $x$ 的分类分数向量为：

$$
\ell(x) = \tau \cdot W v_x
$$

这正是一个线性分类器的形式。区别只在于：普通分类器的 $W$ 是监督训练学出来的；CLIP zero-shot 的 $W$ 是文本编码器生成出来的。

### 玩具例子

假设对 `cat` 有三个模板，编码后得到：

$$
h_1 = [0.6, 0.8], \quad
h_2 = [0.0, 1.0], \quad
h_3 = [0.2, 0.98]
$$

先求平均：

$$
\bar{h} = \frac{h_1+h_2+h_3}{3}
= \left[\frac{0.6+0.0+0.2}{3}, \frac{0.8+1.0+0.98}{3}\right]
= [0.2667, 0.9267]
$$

再归一化，得到：

$$
e_{\text{cat}} = \frac{\bar{h}}{\|\bar{h}\|_2}
\approx [0.2765, 0.9610]
$$

如果某张图片的图像向量是：

$$
v_x = [0.0, 1.0]
$$

那么两者余弦相似度约为：

$$
\cos(v_x, e_{\text{cat}})
= 0.0 \times 0.2765 + 1.0 \times 0.9610
\approx 0.9610
$$

若另一个类别 `dog` 的类向量是：

$$
e_{\text{dog}} \approx [0.9852, 0.1715]
$$

则：

$$
\cos(v_x, e_{\text{dog}}) \approx 0.1715
$$

这时 `cat` 的分数显著高于 `dog`。如果再设 $\tau = 10$，则两个类别的 logits 约为：

$$
\ell_{\text{cat}} \approx 9.610,\quad
\ell_{\text{dog}} \approx 1.715
$$

对应的 softmax 概率为：

$$
P(\text{cat}|x)
=
\frac{e^{9.610}}{e^{9.610}+e^{1.715}}
\approx 0.9996
$$

下面用表格把这个数值过程写清楚：

| 对象 | 向量/数值 | 说明 |
| --- | --- | --- |
| $h_1$ | $[0.6, 0.8]$ | 模板 1 的文本向量 |
| $h_2$ | $[0.0, 1.0]$ | 模板 2 的文本向量 |
| $h_3$ | $[0.2, 0.98]$ | 模板 3 的文本向量 |
| 平均向量 $\bar{h}$ | $[0.2667, 0.9267]$ | 多模板平均 |
| $e_{\text{cat}}$ | $\approx [0.2765, 0.9610]$ | 归一化后的类向量 |
| $v_x$ | $[0.0, 1.0]$ | 图像向量 |
| $\cos(v_x, e_{\text{cat}})$ | $\approx 0.9610$ | 与 `cat` 高相似 |
| $\cos(v_x, e_{\text{dog}})$ | $\approx 0.1715$ | 与 `dog` 低相似 |

这个玩具例子说明了一件关键事实：Prompt Ensemble 不是在做投票，而是在构造一个更稳的语义中心。平均后的类向量更像“这个类别在多种描述下的共同方向”。

### 为什么 `a photo of a` 常常有效

`a photo of a {label}` 这类上下文词通常有效，主要有两个原因：

1. 它把单个词变成自然语言句子，让文本编码器处在更接近预训练分布的输入形式上。
2. 它显式声明视觉语境，告诉模型这里描述的是“图像中的对象”，而不是词典释义或抽象概念。

如果只输入 `crane`，模型可能混淆“起重机”和“鹤”；如果写 `a photo of a crane`，歧义会减少，但仍可能偏向某一类视觉风格。再加入 `a close-up photo of a crane`、`a blurry photo of a crane`、`a drawing of a crane`，类别向量对上下文扰动会更鲁棒。

### 先归一化还是后平均

实现里常见两种写法：

1. 每条 prompt 的文本向量先归一化，再求平均，再整体归一化
2. 先对原始文本向量求平均，再整体归一化

CLIP 及很多复现代码更常见的是第一种：

$$
e_{\text{label}}
=
\text{Normalize}\left(
\frac{1}{T}\sum_{t=1}^{T}\text{Normalize}(h_t)
\right)
$$

原因是不同 prompt 产生的原始向量范数可能不同。如果不先逐条归一化，某些范数较大的 prompt 会在平均时占更大权重。对 zero-shot 分类，通常希望每条模板先等权贡献，再形成类别中心。

### 真实工程例子

以 ImageNet 式服务端分类为例，系统要支持 1000 个类别。若每个类别使用 80 个模板，则离线阶段需要做：

$$
1000 \times 80 = 80000
$$

次文本编码。这个成本不低，但它只发生一次。完成后，每个类别只保留一个平均后的向量，推理时仍然是：

- 1 次图像编码
- 1 次与 1000 个类向量的相似度计算
- 1 次 softmax

所以工程上常见做法不是“运行时逐条 prompt 算分”，而是“离线把 prompt ensemble 折叠成分类器矩阵”。

---

## 代码实现

下面先给出一个可运行的 Python 玩具实现。它不依赖真实 CLIP，只用 `numpy` 模拟“多模板平均后做余弦分类”的流程，目的是把数学对象和工程流程对齐。

```python
import numpy as np

def l2_normalize(x, axis=-1, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def build_class_vector(prompt_vectors):
    """
    prompt_vectors: [T, D]
    先逐条归一化，再平均，再整体归一化
    """
    prompt_vectors = l2_normalize(prompt_vectors, axis=1)
    class_vector = prompt_vectors.mean(axis=0)
    class_vector = l2_normalize(class_vector, axis=0)
    return class_vector

def classify(image_vector, class_vectors, tau=10.0):
    """
    image_vector: [D]
    class_vectors: [C, D]
    """
    image_vector = l2_normalize(image_vector, axis=0)
    class_vectors = l2_normalize(class_vectors, axis=1)
    logits = tau * (class_vectors @ image_vector)
    probs = softmax(logits, axis=0)
    pred = int(np.argmax(probs))
    return pred, logits, probs

cat_prompts = np.array([
    [0.6, 0.8],
    [0.0, 1.0],
    [0.2, 0.98],
], dtype=np.float64)

dog_prompts = np.array([
    [1.0, 0.0],
    [0.9, 0.2],
    [0.85, 0.1],
], dtype=np.float64)

e_cat = build_class_vector(cat_prompts)
e_dog = build_class_vector(dog_prompts)

class_vectors = np.stack([e_cat, e_dog], axis=0)
image_vector = np.array([0.0, 1.0], dtype=np.float64)

pred, logits, probs = classify(image_vector, class_vectors, tau=10.0)

labels = ["cat", "dog"]

assert pred == 0
assert probs[0] > probs[1]
assert round(float(probs[0]), 4) > 0.99

print("e_cat =", np.round(e_cat, 4))
print("e_dog =", np.round(e_dog, 4))
print("logits =", np.round(logits, 4))
print("probs =", np.round(probs, 4))
print("predict =", labels[pred])
```

这段代码可以直接运行，预期输出形式类似：

```text
e_cat = [0.2765 0.961 ]
e_dog = [0.9852 0.1715]
logits = [9.6104 1.7147]
probs = [9.996e-01 4.000e-04]
predict = cat
```

这个玩具实现对应的工程含义如下：

| 代码对象 | 数学对象 | 工程含义 |
| --- | --- | --- |
| `cat_prompts` | $\{h_t\}_{t=1}^T$ | 某个类别的多条文本特征 |
| `build_class_vector` | $e_{\text{label}}$ | 折叠后的类向量 |
| `class_vectors` | $W \in \mathbb{R}^{C \times D}$ | 零样本文本分类器矩阵 |
| `image_vector` | $v_x$ | 图像编码器输出 |
| `class_vectors @ image_vector` | $Wv_x$ | 所有类别分数 |

如果换成真实 CLIP，代码结构通常是下面这样。这个版本仍然是完整的可运行结构，但需要你已经安装相应库并下载好权重。

```python
import torch
import open_clip
from PIL import Image

PROMPTS = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a close-up photo of a {}",
    "a black and white photo of a {}",
    "a drawing of a {}",
    "a bad photo of a {}",
]

def build_zero_shot_weights(labels, model, tokenizer, device):
    weights = []
    with torch.no_grad():
        for label in labels:
            texts = [template.format(label) for template in PROMPTS]
            token_ids = tokenizer(texts).to(device)          # [T, L]
            text_features = model.encode_text(token_ids)     # [T, D]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            class_feature = text_features.mean(dim=0)        # [D]
            class_feature = class_feature / class_feature.norm()
            weights.append(class_feature)
    return torch.stack(weights, dim=0)                       # [C, D]

def predict(image_path, weights, labels, model, preprocess, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image)            # [1, D]
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (image_feature @ weights.T)   # [1, C]
        probs = logits.softmax(dim=-1)[0]
        pred = int(probs.argmax().item())
    return labels[pred], probs.cpu()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    labels = ["cat", "dog", "car"]
    weights = build_zero_shot_weights(labels, model, tokenizer, device)
    name, probs = predict("test.jpg", weights, labels, model, preprocess, device)

    print("predict =", name)
    print("probs =", probs.numpy().round(4))
```

这里有几个实现细节必须说清。

第一，文本侧最好做缓存。因为类别列表和模板列表通常是固定的，所以 `weights` 可以在服务启动前生成并持久化到磁盘。运行时不应该重复编码文本。

第二，推理阶段要矩阵化。不要对每个类别写 Python 循环逐个算余弦，而是把所有类向量堆成矩阵，一次做批量点积。原因很直接：文本类向量是静态的，图像向量是动态的，矩阵乘法最适合这种“一个查询对很多类”的模式。

第三，训练式 prompt 方法和 Prompt Ensemble 的代码结构完全不同：

- Prompt Ensemble：没有训练，只有离线文本编码和在线匹配
- CoOp/CoCoOp：需要训练可学习上下文 token，再导出类向量或在推理态动态生成 prompt

这两类方法在工程上不应混写，否则很容易把“无训练零样本方案”和“少样本微调方案”混成一套。

第四，类名本身也可能需要标准化。比如 ImageNet 的类名里常出现同义词、逗号分隔别名、复数形式。真实系统里常见做法是先把标签映射成人类可读的标准短语，再套模板，而不是直接把原始数据库主键丢给文本编码器。

下面是缓存设计的最小建议：

| 缓存项 | 内容 | 是否运行时重复计算 |
| --- | --- | --- |
| 模板列表 | prompt 字符串集合 | 否 |
| tokenized prompts | 分词后的 token | 否 |
| 每类平均向量 | 最终类向量 | 否 |
| 类别矩阵 | 所有类向量堆叠后的矩阵 | 否 |
| 图像特征 | 输入图像向量 | 是 |

---

## 工程权衡与常见坑

Prompt Ensemble 的优点是简单、稳健、无需微调，但它不是没有成本。最大成本在离线文本编码阶段，而不是在线服务阶段。

### 成本与收益

| 方案 | 是否需训练 | 离线成本 | 在线成本 | 对 unseen 类泛化 |
| --- | --- | --- | --- | --- |
| Prompt Ensemble | 否 | 中 | 低 | 强 |
| CoOp | 是 | 高 | 低 | 往往变弱 |
| CoCoOp | 是 | 高 | 低到中 | 比 CoOp 好，但依赖训练设置 |

对大多数“无标注或少标注”的零样本部署任务，Prompt Ensemble 往往是第一选择。它的本质是启发式优化，但这个启发式和 CLIP 的预训练方式高度一致，所以常常有效。

### 常见坑 1：随意加形容词会引入偏置

例如把模板改成 `a cute {label}`。`cute` 这个词本身带有审美偏好，会让类向量偏向“可爱风格”的图像分布。对 `cat` 可能问题不大，对 `truck`、`crab`、`drill` 这类类别就可能造成无意义偏移。

更安全的做法是使用描述成像条件、视角、质量、媒介的模板，例如：

- `a photo of a {}`
- `a cropped photo of a {}`
- `a blurry photo of a {}`
- `a rendering of a {}`
- `a black and white photo of a {}`

这些模板提供的是上下文覆盖，而不是额外类别属性。

### 常见坑 2：模板数量不是越多越好

如果模板集质量差，增加数量只会把噪声平均进去。真正有价值的是“语义覆盖面”，不是“字符串数量”。5 到 20 条高质量模板，往往比 50 条随意改写的模板更有效。

可以把模板设计目标写成一个简单检查表：

| 检查项 | 问题 | 建议 |
| --- | --- | --- |
| 视觉语境 | 是否明确这是图像描述 | 优先保留 `photo / image / picture` 一类上下文 |
| 视角覆盖 | 是否覆盖近景、裁剪、模糊等常见变化 | 少量加入质量和视角模板 |
| 媒介覆盖 | 是否包含照片、绘图、渲染等类型 | 按任务分布选择 |
| 属性泄漏 | 是否引入不属于类别定义的主观词 | 删除 `cute / beautiful / ugly` 等词 |
| 冗余程度 | 多个模板是否只是表面改写 | 去掉等价重复模板 |

### 常见坑 3：标签词本身有歧义

例如 `crane`、`seal`、`bat` 都可能有多重含义。只写标签词，模型可能把文本理解成另一种概念。更稳的做法有两种：

1. 把类名写成更完整的短语，例如 `a photo of a construction crane`
2. 在标签映射层先把数据库标签改成人类可读的无歧义名称

这不是模板技巧，而是标签语义清洗。很多 zero-shot 系统效果差，问题并不在模型，而在标签定义本身就不干净。

### 常见坑 4：平均方式不一致

有些实现直接平均原始文本向量，有些实现先逐条归一化再平均。如果你在不同实验里混用这两种写法，结果可能不稳定。更稳妥的做法是固定一套流程：

1. 每条 prompt 编码
2. 每条特征做 $L_2$ 归一化
3. 按类别求平均
4. 对平均结果再做一次 $L_2$ 归一化

只有这样，模板之间才更接近“等权”。

### 常见坑 5：CoOp 容易学到 base 类偏好

CoOp 是 Context Optimization，意思是“把 prompt 里的上下文 token 设成可学习参数”。白话解释就是“不手写 `a photo of a`，而是让模型自己学前缀词向量”。

它在 base classes 上经常有效，因为它能适配训练集分布；但问题是，这些可学习上下文可能只对见过的类别有利。一旦测试类别是 unseen classes，泛化能力会下降。

### 常见坑 6：CoCoOp 不是免费修复

CoCoOp 在 CoOp 基础上引入图像条件 token，也就是根据输入图像动态生成一部分上下文。白话解释就是“prompt 不再对所有图像固定，而是按图像内容轻微调整”。

这确实能缓解 CoOp 的泛化问题，但会带来新的工程要求：

- 需要训练数据
- 需要额外训练模块
- 需要控制正则化，防止条件 token 过拟合
- 推理图比纯 Ensemble 更复杂

所以 CoCoOp 不是 Prompt Ensemble 的直接替代，而是另一类训练式方案。

下面是工程决策表：

| 方法 | 泛化 | 训练开销 | 对模板依赖 | 正则要求 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| Prompt Ensemble | 高 | 无 | 高 | 无 | 零标注部署、快速上线 |
| CoOp | 中到低 | 高 | 低 | 中 | 有标注、关心 base 类效果 |
| CoCoOp | 中到高 | 高 | 低 | 高 | 有标注、希望缓解 unseen 问题 |

如果必须做线上服务，还要加一个性能层面的判断：

| 环节 | Prompt Ensemble | CoCoOp |
| --- | --- | --- |
| 启动预热 | 编码全部文本并缓存 | 加载训练模块与权重 |
| 单图推理 | 图像编码 + 矩阵乘法 | 图像编码 + 条件 prompt 生成 + 矩阵乘法 |
| 可解释性 | 能直接查看模板集合 | 动态 prompt 更难排查 |
| 故障定位 | 模板或标签问题更容易定位 | 训练与推理链路都可能出错 |

---

## 替代方案与适用边界

Prompt Ensemble 的最优使用场景，是类别集合已知、可以离线预计算、又不希望引入训练过程的系统。典型例子就是服务端图片分类、内容审核标签预筛、商品图粗粒度归类。

如果场景满足下面几个条件，Prompt Ensemble 通常很合适：

- 没有或不想维护标注数据
- 希望快速接入 CLIP 零样本能力
- 类别集合相对稳定
- 可以接受一次性离线文本编码
- 更看重 unseen 类的稳健性

如果已经有一定标注数据，并且任务分布与线上目标比较接近，训练式 prompt 方法才有讨论价值。尤其是当你只关心少量固定类别、并愿意接受微调成本时，CoOp 或 CoCoOp 才可能带来比启发式模板更高的上限。

可以用一个简单规则来判断：

- 没数据：先用 Prompt Ensemble
- 有少量数据但类别会变：优先保守地继续用 Ensemble
- 有足够数据且类别固定：再考虑 CoOp/CoCoOp

最终对比如下：

| 方案 | 是否需微调 | 对 unseen 类表现 | 实验依赖 |
| --- | --- | --- | --- |
| Prompt Ensemble | 否 | 通常最好 | 依赖模板设计与缓存 |
| CoOp | 是 | 常弱于 Ensemble | 依赖 base 类标注与训练配置 |
| CoCoOp | 是 | 通常优于 CoOp，但未必超过 Ensemble | 依赖标注、条件网络、正则化 |

一句话总结边界：Prompt Ensemble 解决的是“如何更稳地把类别名变成分类器”，不是“如何让模型学会新视觉知识”。如果任务需要学习领域特有概念，仅靠模板改写通常不够，还是要引入微调或额外监督。

再补两个容易忽略的边界。

第一，Prompt Ensemble 并不能自动解决细粒度区分问题。比如 `husky` 和 `malamute`，即使模板很多，若预训练数据本身对这两个概念区分不够稳定，类向量仍然可能高度接近。

第二，Prompt Ensemble 也不能替代检测、分割、多标签建模。它最适合的是“整张图对应一个主标签”或“先做粗筛再下游精判”的链路。如果一张图里同时有多类对象、且你需要位置或实例级输出，就应切换到检测或分割模型。

可以把适用边界收敛成下面这张表：

| 任务类型 | Prompt Ensemble 是否合适 | 原因 |
| --- | --- | --- |
| 整图单标签分类 | 合适 | 类向量匹配最直接 |
| 粗粒度召回/预筛 | 合适 | 零标注、上线快 |
| 多标签分类 | 视阈值设计而定 | 可做，但需要独立分数校准 |
| 目标检测 | 不合适 | 缺少区域定位能力 |
| 实例分割 | 不合适 | 缺少像素级监督与输出 |
| 领域专有分类 | 通常不够 | 需要额外监督或微调 |

---

## 参考资料

1. Alec Radford, Jong Wook Kim, Chris Hallacy, et al. *Learning Transferable Visual Models From Natural Language Supervision*，2021。原始 CLIP 论文，核心看 zero-shot 分类构造、模板设计和 ImageNet 评估部分。  
   链接：https://arxiv.org/abs/2103.00020

2. OpenAI CLIP 源码仓库。适合对照 zero-shot 推理代码，重点看文本模板、文本特征归一化和类别权重构造方式。  
   链接：https://github.com/openai/CLIP

3. Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu. *Learning to Prompt for Vision-Language Models*，2021。CoOp 原始论文，重点看 hand-crafted prompt 与 learnable context 的对比，以及 base / novel classes 设置。  
   链接：https://arxiv.org/abs/2109.01134

4. Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu. *Conditional Prompt Learning for Vision-Language Models*，2022。CoCoOp 原始论文，重点看 CoOp 在 unseen classes 上的泛化问题，以及条件 token 的改进。  
   链接：https://arxiv.org/abs/2203.05557

5. OpenCLIP 仓库中的 zero-shot 与模板实现。适合工程落地时参考不同模型族的推理接口、模板复用和缓存方式。  
   链接：https://github.com/mlfoundations/open_clip

6. ImageNet prompt template 的开源实现与讨论资料。建议阅读方式是先看模板集合本身，再对照实验观察“模板覆盖面”如何影响 zero-shot 结果。  
   链接：https://github.com/openai/CLIP/blob/main/data/prompts.md
