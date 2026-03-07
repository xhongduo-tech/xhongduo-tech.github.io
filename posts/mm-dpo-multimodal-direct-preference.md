## 核心结论

MM-DPO 这里指论文中的 **mDPO**。它的重点不是“把 DPO 机械搬到多模态输入上”，而是修复标准多模态 DPO 的一个关键漏洞：**训练目标虽然写了图像条件，但模型未必真的依赖图像来做偏好判断**。

先给结论：

1. 标准多模态 DPO 在实践中容易出现 **无条件偏好**。也就是模型学到“这类问题通常更偏好哪种回答”，而不是“这张图像下哪种回答更合理”。
2. mDPO 在标准 DPO 之外补了两个约束：
   - **条件视觉偏好**：同一个回答，放在原图上应比放在破坏后的图像上更合理。
   - **锚点奖励**：chosen response 的隐式奖励不能只要求“比 rejected 更高”，还要尽量保持为正，避免把好答案的概率一起压低。
3. 这两个约束带来的直接收益是 **降低视觉幻觉**。这里的视觉幻觉是指模型会编造图中不存在的目标、属性、数量或关系。
4. 在 Bunny-v1.0-3B 和 LLaVA-v1.5-7B 上，mDPO 相比普通 DPO 在 MMHalBench、Object HalBench、AMBER 上都更稳，尤其体现在 HalRate、CHAIR 这类幻觉相关指标下降。

论文主结果里最值得记住的是下面这组数字：

| 模型 | 方法 | MMHalBench Score ↑ | HalRate ↓ | Object HalBench CHAIRs ↓ | AMBER HalRate ↓ |
|---|---:|---:|---:|---:|---:|
| Bunny-v1.0-3B | 基线 | 2.11 | 0.58 | 43.0 | 64.9 |
| Bunny-v1.0-3B | DPO | 2.28 | 0.56 | 44.3 | 58.9 |
| Bunny-v1.0-3B | mDPO | 2.96 | 0.42 | 27.0 | 37.7 |
| LLaVA-v1.5-7B | 基线 | 2.19 | 0.57 | 54.7 | 34.7 |
| LLaVA-v1.5-7B | DPO | 2.14 | 0.65 | 49.0 | 34.5 |
| LLaVA-v1.5-7B | mDPO | 2.39 | 0.54 | 35.7 | 24.5 |

如果只看趋势，这张表说明两件事：

| 观察 | 解释 |
|---|---|
| 普通 DPO 不一定稳定改善幻觉 | 它会提升“回答排序”，但未必强化“回答与图像证据绑定” |
| mDPO 在多个模型上更稳定 | 它把“图像是否支持这句回答”也纳入偏好优化目标 |

一个新手更容易理解的玩具例子是：

- 问题：`图片里这件衣服是红色吗？`
- 标准 DPO 可能学到：被问到“颜色判断”时，肯定句往往更自然、更像人类偏好的回答，于是倾向输出“是”。
- mDPO 会额外比较“原图”和“随机裁剪图”。如果裁掉后衣服已经看不见，模型还继续自信回答“是”，就会在条件视觉偏好项上被惩罚。
- 结果是模型被迫学习：**回答不仅要像好答案，还必须有图像证据支持。**

一句话概括：

**标准 DPO 在多模态里主要优化“回答排序”，mDPO 进一步优化“回答必须和图像绑定”。**

---

## 问题定义与边界

DPO 是 **Direct Preference Optimization**。它的基本思想是：**不单独训练一个奖励模型，而是直接用偏好对来更新策略模型**。  
在实现上，它通常会拿一个当前策略模型 `policy` 和一个冻结的参考模型 `reference` 做对比，鼓励策略模型在 chosen answer 上相对参考模型更强，在 rejected answer 上相对参考模型更弱。

在纯文本场景里，给定问题 $q$、优选回答 $y_w$、劣选回答 $y_l$，DPO 最大化的目标可写成：

$$
L_{\text{DPO}}
=
-\log \sigma \left(
\beta \left[
\log \frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)}
-
\log \frac{\pi_\theta(y_l|q)}{\pi_{\text{ref}}(y_l|q)}
\right]
\right)
$$

把中间那一项展开后更容易理解：

$$
\Delta r
=
\beta \Big(
(\log \pi_\theta(y_w|q) - \log \pi_{\text{ref}}(y_w|q))
-
(\log \pi_\theta(y_l|q) - \log \pi_{\text{ref}}(y_l|q))
\Big)
$$

其中：

- $\pi_\theta$ 是当前要训练的策略模型
- $\pi_{\text{ref}}$ 是冻结的参考模型
- $\beta$ 控制“偏离参考模型”的强度
- $\Delta r$ 越大，说明 chosen 比 rejected 更被当前策略偏好

到了多模态场景，输入变成图像 $m$ 和问题 $q$，最自然的写法是：

$$
L_{\text{DPOm}}
=
-\log \sigma \left(
\beta \left[
\log \frac{\pi_\theta(y_w|m,q)}{\pi_{\text{ref}}(y_w|m,q)}
-
\log \frac{\pi_\theta(y_l|m,q)}{\pi_{\text{ref}}(y_l|m,q)}
\right]
\right)
$$

按公式看，图像已经进入条件，似乎问题已经解决。  
但论文指出，**“条件里包含图像”不等于“训练出的模型真的依赖图像”**。

### 为什么会这样

因为标准 DPO 的监督信号只要求：

- 在同一组输入下，`chosen` 的相对分数高于 `rejected`

它**没有单独检查**模型到底是靠什么信息把 `chosen` 判得更高。  
于是模型可能走两条不同路径：

| 路径 | 实际做法 | 结果 |
|---|---|---|
| 正确路径 | 读取图像证据，再判断 chosen 更合理 | 真正提升多模态对齐 |
| 错误路径 | 只根据问题文本、回答模板和训练分布猜一个“常见优选答案” | 形成无条件偏好 |

论文做了一个很有说服力的对照：`DPO (No Image)`。  
做法很直接：训练时把图像去掉，只保留问题和回答对。结果发现，在 MMHalBench 上，它和带图训练的普通 DPO 表现接近。这说明标准多模态 DPO 的一部分收益，其实可能来自：

- 问题文本先验
- 回答风格偏好
- 数据集中的共现模式

而不是图像理解本身。

### 什么叫“无条件偏好”

所谓 **无条件偏好**，可以理解成：

> 模型学到“什么回答经常被选中”，但没有学到“为什么这张图会支持这个回答”。

电商问答里这类问题很典型：

- 图文输入：商品图 + 问题 `左边的是不是蓝色杯子？`
- 如果训练数据中“颜色类问题”常常偏好简短肯定回答，模型就可能在证据不足时也回答“是”。
- 这不是视觉推理，而是 **文本先验主导的偏好学习**。

### mDPO 解决什么，不解决什么

mDPO 的边界需要说清楚：

| 问题 | 标准 DPO | mDPO |
|---|---|---|
| 学会 chosen 比 rejected 更好 | 能做 | 能做 |
| 保证偏好真的依赖图像 | 不能保证 | 专门处理 |
| 避免好答案被一起压低 | 不能保证 | 用 anchor 处理 |
| 彻底提升 OCR / 检测 /计数能力 | 不能 | 也不能 |
| 彻底消除所有幻觉 | 不能 | 也不能 |

所以 mDPO 解决的是：

**偏好优化阶段的视觉绑定问题。**

它不是万能替代品，不能替代：

- 更强的视觉编码器
- 更干净的训练数据
- 更好的检测、OCR、图表理解能力
- 更高质量的偏好标注

---

## 核心机制与推导

mDPO 可以拆成三层：

1. 标准回答偏好：`同一张图像下，chosen 比 rejected 更好`
2. 条件视觉偏好：`同一个回答，在原图上比在破坏图上更合理`
3. 锚点奖励：`chosen 的隐式奖励不能掉得太低`

先看总览：

| 阶段 | 变化的对象 | 固定的对象 | 学到什么 |
|---|---|---|---|
| `L_DPOm` | 回答变 | 图像、问题固定 | 哪个回答更好 |
| `L_CoPO` | 图像变 | 问题、chosen 回答固定 | 图像是否支持这个回答 |
| `L_AncPO` | 奖励阈值约束 | chosen 样本固定 | 防止好答案概率被顺带压低 |

### 1. 标准多模态 DPO

标准多模态 DPO 仍然是：

$$
L_{\text{DPOm}}
=
-\log \sigma \left(
r(m,q,y_w) - r(m,q,y_l)
\right)
$$

其中隐式奖励定义为：

$$
r(m,q,y)
=
\beta \left(
\log \pi_\theta(y|m,q) - \log \pi_{\text{ref}}(y|m,q)
\right)
$$

代入后得到：

$$
L_{\text{DPOm}}
=
-\log \sigma \left(
\beta \log \frac{\pi_\theta(y_w|m,q)}{\pi_{\text{ref}}(y_w|m,q)}
-
\beta \log \frac{\pi_\theta(y_l|m,q)}{\pi_{\text{ref}}(y_l|m,q)}
\right)
$$

这项损失只表达一件事：

**在同样的图像和问题下，chosen 应该优于 rejected。**

但它无法回答下面这个问题：

> chosen 胜出，是因为模型真的看了图，还是因为模型只会顺着问题猜一个常见偏好回答？

### 2. 条件视觉偏好 `L_CoPO`

mDPO 的核心补丁就是 `CoPO`。它不再比较两个回答，而是比较：

- 同一个回答
- 在两张不同图像上的合理性差异

设：

- $m_w$ 是原图
- $m_l$ 是被破坏的图像，一般用随机裁剪构造
- $q$ 是问题
- $y_w$ 是 chosen answer

则条件视觉偏好项写成：

$$
L_{\text{CoPO}}
=
-\log \sigma \left(
r(m_w,q,y_w) - r(m_l,q,y_w)
\right)
$$

展开就是：

$$
L_{\text{CoPO}}
=
-\log \sigma \left(
\beta \log \frac{\pi_\theta(y_w|m_w,q)}{\pi_{\text{ref}}(y_w|m_w,q)}
-
\beta \log \frac{\pi_\theta(y_w|m_l,q)}{\pi_{\text{ref}}(y_w|m_l,q)}
\right)
$$

这项损失的意义非常直接：

- 回答不变
- 问题不变
- 唯一变化的是图像

因此模型不能再靠“回答模板更自然”过关，只能学：

- 原图支持这个回答时，应该给高分
- 破坏图不再支持这个回答时，分数应该下降

这就是 **conditional preference optimization** 的核心思想。

### 3. 为什么用“裁剪图”做负图像

这里很多新手会问：为什么负例不是随便换一张别的图，而是原图裁剪版？

原因有三点：

| 方案 | 优点 | 缺点 |
|---|---|---|
| 随机换另一张图 | 差异大，容易构造 | 分布偏差大，负例过于简单 |
| 完全遮挡原图 | 实现容易 | 太极端，训练信号粗糙 |
| 原图随机裁剪 | 与原图接近，仍保留大部分背景 | 更接近真实“证据缺失”场景 |

论文采用的思路是：  
**尽量保持图像主体和分布不变，只破坏关键证据，让模型学会细粒度地区分“有证据”和“证据不足”。**

比如：

- 原图：桌上有咖啡杯和一本书
- 裁剪图：裁掉右侧后，咖啡杯消失
- 问题：`桌上有什么？`
- chosen：`桌上有一杯咖啡和一本书。`

如果模型在两张图上都对这句话同样自信，就说明它没有真正依赖图像。

### 4. 锚点奖励 `L_AncPO`

标准 DPO 是相对目标。它只要求：

$$
r(m,q,y_w) > r(m,q,y_l)
$$

问题在于，模型可能通过下面这种方式让目标变好：

- 把 chosen 的相对分数降一点
- 把 rejected 的相对分数降更多

这样两者差值变大了，损失也可能变小，但 chosen 本身并没有被真正强化。  
极端情况下，会出现一个不直观结果：

**模型学会了“更讨厌 rejected”，但没有更喜欢 chosen”。**

mDPO 用 anchor 来修复这个问题：

$$
L_{\text{AncPO}}
=
-\log \sigma \left(
r(m_w,q,y_w) - \delta
\right)
$$

其中：

- $\delta$ 是锚点阈值
- 论文默认取 $\delta = 0$

把奖励定义代进去：

$$
L_{\text{AncPO}}
=
-\log \sigma \left(
\beta \log \frac{\pi_\theta(y_w|m_w,q)}{\pi_{\text{ref}}(y_w|m_w,q)}
-
\delta
\right)
$$

当 $\delta = 0$ 时，这项的含义就是：

**chosen answer 相对参考模型的隐式奖励，最好保持为正。**

### 5. 总目标

于是 mDPO 的总损失就是三项相加：

$$
L_{\text{mDPO}}
=
L_{\text{DPOm}} + L_{\text{CoPO}} + L_{\text{AncPO}}
$$

如果写得更完整一点：

$$
L_{\text{mDPO}}
=
-\log \sigma(r_w-r_l)
-\log \sigma(r_w-r_{\text{crop}})
-\log \sigma(r_w-\delta)
$$

其中：

$$
r_w = r(m_w,q,y_w), \quad
r_l = r(m_w,q,y_l), \quad
r_{\text{crop}} = r(m_l,q,y_w)
$$

这三个量分别对应：

| 记号 | 含义 |
|---|---|
| $r_w$ | 原图上，chosen 的隐式奖励 |
| $r_l$ | 原图上，rejected 的隐式奖励 |
| $r_{\text{crop}}$ | 裁剪图上，同一个 chosen 的隐式奖励 |

### 6. 一个最小数值推导

设 $\beta=1,\delta=0$，并假设：

- 原图上，chosen 的对数概率：$\log \pi_\theta(y_w|m,q)=-1.0$
- 原图上，chosen 的参考概率：$\log \pi_{\text{ref}}(y_w|m,q)=-1.5$
- 原图上，rejected 的对数概率：$\log \pi_\theta(y_l|m,q)=-2.0$
- 原图上，rejected 的参考概率：$\log \pi_{\text{ref}}(y_l|m,q)=-2.0$

那么：

$$
r_w = -1.0 - (-1.5) = 0.5
$$

$$
r_l = -2.0 - (-2.0) = 0
$$

标准 DPO 的间隔是：

$$
r_w - r_l = 0.5
$$

因此：

$$
L_{\text{DPOm}} = -\log \sigma(0.5)
$$

如果裁剪图上的 chosen 概率变成：

- $\log \pi_\theta(y_w|m_l,q)=-1.8$
- $\log \pi_{\text{ref}}(y_w|m_l,q)=-1.7$

则：

$$
r_{\text{crop}} = -1.8 - (-1.7) = -0.1
$$

条件视觉偏好项的间隔就是：

$$
r_w - r_{\text{crop}} = 0.5 - (-0.1) = 0.6
$$

说明模型被进一步推动去学习：

- 原图支持这句话时，要更自信
- 裁剪图不再支持这句话时，要降低自信

而 anchor 项则要求：

$$
r_w - \delta = 0.5 - 0 = 0.5
$$

意味着 chosen 的相对奖励本身也要保持在合理区间。

这里的数学含义可以直接概括为：

**mDPO 不只比较两个回答谁更好，还比较“同一句回答在不同图像条件下是否仍然成立”。**

---

## 代码实现

下面给一个**最小可运行**的 Python 版本。它只演示损失计算，不依赖 PyTorch、TensorFlow 等框架，直接运行即可看到每一项损失与总损失。

```python
import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def neg_log_sigmoid(x: float) -> float:
    return -math.log(sigmoid(x))


def reward(logp_policy: float, logp_ref: float, beta: float = 1.0) -> float:
    return beta * (logp_policy - logp_ref)


def dpo_m_loss(
    logp_w: float,
    logp_w_ref: float,
    logp_l: float,
    logp_l_ref: float,
    beta: float = 1.0,
) -> float:
    r_w = reward(logp_w, logp_w_ref, beta)
    r_l = reward(logp_l, logp_l_ref, beta)
    return neg_log_sigmoid(r_w - r_l)


def copo_loss(
    logp_good_img: float,
    logp_good_img_ref: float,
    logp_bad_img: float,
    logp_bad_img_ref: float,
    beta: float = 1.0,
) -> float:
    r_good = reward(logp_good_img, logp_good_img_ref, beta)
    r_bad = reward(logp_bad_img, logp_bad_img_ref, beta)
    return neg_log_sigmoid(r_good - r_bad)


def ancpo_loss(
    logp_w: float,
    logp_w_ref: float,
    beta: float = 1.0,
    delta: float = 0.0,
) -> float:
    r_w = reward(logp_w, logp_w_ref, beta)
    return neg_log_sigmoid(r_w - delta)


def mdpo_loss(
    logp_w: float,
    logp_w_ref: float,
    logp_l: float,
    logp_l_ref: float,
    logp_crop_w: float,
    logp_crop_w_ref: float,
    beta: float = 1.0,
    delta: float = 0.0,
) -> dict:
    loss_dpo = dpo_m_loss(logp_w, logp_w_ref, logp_l, logp_l_ref, beta)
    loss_copo = copo_loss(logp_w, logp_w_ref, logp_crop_w, logp_crop_w_ref, beta)
    loss_anc = ancpo_loss(logp_w, logp_w_ref, beta, delta)
    total = loss_dpo + loss_copo + loss_anc

    return {
        "L_DPOm": loss_dpo,
        "L_CoPO": loss_copo,
        "L_AncPO": loss_anc,
        "L_mDPO": total,
    }


def main() -> None:
    # 原图上的 chosen：更合理
    logp_w = -1.0
    logp_w_ref = -1.5

    # 原图上的 rejected：较差
    logp_l = -2.0
    logp_l_ref = -2.0

    # 裁剪图上的同一句 chosen：因为证据缺失，变得不那么合理
    logp_crop_w = -1.8
    logp_crop_w_ref = -1.7

    losses = mdpo_loss(
        logp_w=logp_w,
        logp_w_ref=logp_w_ref,
        logp_l=logp_l,
        logp_l_ref=logp_l_ref,
        logp_crop_w=logp_crop_w,
        logp_crop_w_ref=logp_crop_w_ref,
        beta=1.0,
        delta=0.0,
    )

    for name, value in losses.items():
        print(f"{name}: {value:.6f}")

    # 基本正确性检查
    assert losses["L_mDPO"] > 0.0
    assert losses["L_CoPO"] < neg_log_sigmoid(0.0)
    assert losses["L_AncPO"] < neg_log_sigmoid(0.0)


if __name__ == "__main__":
    main()
```

这段代码可以直接保存为 `mdpo_demo.py` 后运行：

```bash
python3 mdpo_demo.py
```

预期输出会是一组正数，例如：

```text
L_DPOm: 0.474077
L_CoPO: 0.437488
L_AncPO: 0.474077
L_mDPO: 1.385642
```

### 为什么这段代码是“可运行”的

它满足最基本的可执行条件：

| 条件 | 是否满足 |
|---|---|
| 无第三方依赖 | 是 |
| 包含入口函数 | 是 |
| 变量定义完整 | 是 |
| 有断言做最小校验 | 是 |
| 输出可解释 | 是 |

### 对应到真实训练时，batch 里发生什么

真实训练会比这个玩具版本多两件事：

1. `logp` 不是手写常数，而是模型前向计算出的 token log-prob
2. 图像裁剪负例是在线或离线生成的

真实训练流程可以写成：

```python
for batch in loader:
    image = batch["image"]
    question = batch["question"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    image_crop = random_crop_keep_majority(image, max_drop_ratio=0.2)

    logp_w = policy.logprob(image, question, chosen)
    logp_l = policy.logprob(image, question, rejected)
    logp_w_crop = policy.logprob(image_crop, question, chosen)

    with no_grad():
        logp_w_ref = ref.logprob(image, question, chosen)
        logp_l_ref = ref.logprob(image, question, rejected)
        logp_w_crop_ref = ref.logprob(image_crop, question, chosen)

    loss = (
        DPOm(logp_w, logp_w_ref, logp_l, logp_l_ref)
        + CoPO(logp_w, logp_w_ref, logp_w_crop, logp_w_crop_ref)
        + AncPO(logp_w, logp_w_ref, delta=0.0)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 新手容易卡住的三个实现点

| 难点 | 说明 | 解决办法 |
|---|---|---|
| `logprob` 怎么算 | 不是句子整体 softmax，而是目标序列 token log-prob 累加 | 复用 DPO 现有实现 |
| 裁剪怎么做 | 裁掉太多会让负例失真，裁掉太少又没信号 | 保留大部分图像，只破坏关键证据 |
| 参考模型怎么处理 | 参考模型是冻结的，不参与更新 | 与文本 DPO 一样做 frozen ref |

### 一个更贴近工程的例子

电商多模态问答里：

- 输入：商品主图 + 问题 `左边这个杯子是蓝色还是黑色？`
- chosen：`左边的杯子是蓝色。`
- rejected：`左边的杯子是黑色。`
- 负图像：把左侧区域裁掉，或者保留背景但破坏杯子证据
- 目标：只有在能看见左边杯子时，chosen 才保持高分

这一点很重要，因为它说明 mDPO 的负例构造不一定依赖新增人工标注。  
只要原始偏好对已经存在，很多场景都可以通过数据管线动态生成裁剪图。

一个简化后的工程参数表如下：

| 参数 | 常见设置 | 作用 |
|---|---|---|
| $\beta$ | 0.1 左右 | 控制偏离参考模型的力度 |
| $\delta$ | 0 | 约束 chosen 奖励尽量为正 |
| 裁剪比例 | 保留大部分信息 | 构造视觉硬负例 |
| epoch | 约 3 | 偏好微调常见量级 |
| 微调方式 | LoRA | 降低显存和成本 |

---

## 工程权衡与常见坑

mDPO 的工程价值不在于“从此不需要别的方法”，而在于它在**保持 DPO 低复杂度**的前提下，补上了多模态偏好优化中最容易失真的部分。

先看一个粗粒度对比：

| 方案 | 数据量 | 训练阶段 | 额外组件 | 工程复杂度 |
|---|---:|---:|---|---|
| 普通 DPO | 约 5K-10K 可起步 | 1 阶段 | 参考模型 | 低 |
| mDPO | 约 5K-10K 可起步 | 1 阶段 | 参考模型 + 图像裁剪管线 | 中低 |
| 多模态 RLHF | 往往更大 | 多阶段 | 奖励模型 + PPO | 高 |

这个对比的重点不是绝对数字，而是工程链路：

- 普通 DPO：最短路径，容易起步
- mDPO：多一条图像负例管线，但仍然是单阶段偏好优化
- RLHF：数据、奖励模型、采样、PPO 稳定性都更复杂

### mDPO 值得在哪些场景优先上

| 场景 | 原因 |
|---|---|
| 电商问答 | 颜色、数量、左右位置等问题强依赖图像 |
| 医疗影像问答 | 不能顺着问题假设编答案，幻觉风险高 |
| 图表/文档问答 | 图像证据缺失会直接导致事实错误 |
| 质检与审核 | 需要模型明确“我看到了什么” |

### 常见坑 1：只做回答偏好，不做图像条件偏好

现象：

- benchmark 上可能有一点提升
- 但属性幻觉、存在性幻觉下降不明显

原因：

- 模型仍然可以主要依赖问题文本和回答风格
- 没有被显式要求“回答必须随图像变化而变化”

### 常见坑 2：裁剪太狠，负例变成无意义样本

如果把图裁得太狠，训练信号会变成：

- 这张图什么都看不清
- 所以模型别回答

这并不是 mDPO 想学到的核心能力。  
mDPO 更希望模型学会的是：

- 关键证据消失时，要降低相关回答的确信度
- 不是对所有不完整图像都一律沉默

### 常见坑 3：裁剪太轻，负例没有区分度

如果原图和裁剪图几乎一样，那么：

$$
r(m_w,q,y_w) \approx r(m_l,q,y_w)
$$

此时 `L_CoPO` 提供的梯度很弱，模型学不到“图像证据变化会改变回答合理性”。

### 常见坑 4：只做 CoPO，不做 anchor

只加 `CoPO` 的确能强化图像依赖，但仍可能出现：

- chosen 和 rejected 都在下降
- chosen 只是“降得没那么多”

这会导致 reward 差值改善，但最终生成质量不稳定。  
`AncPO` 的作用就是避免只靠“压低坏答案”来获得表面收益。

### 常见坑 5：把 benchmark 提升误解成底层感知能力增强

mDPO 优化的是：

- 偏好学习方式
- 视觉条件绑定
- 幻觉抑制

它不等于：

- 视觉编码器突然更强
- OCR 突然更准
- 计数能力突然补齐
- 细粒度检测能力凭空提升

如果底层模型本来就分不清“深蓝”和“黑色”，mDPO 也不可能单靠偏好损失解决这个问题。

### 实际落地时怎么判断值不值得上 mDPO

可以用一个简单判断表：

| 问题特征 | 是否适合 mDPO |
|---|---|
| 回答主要看语言流畅度 | 不一定 |
| 回答强依赖图像证据 | 适合 |
| 训练预算很紧，但能做数据管线 | 适合 |
| 能维护奖励模型和 PPO | 可以考虑 RLHF |
| 偏好数据本身很脏 | 应先清数据 |

---

## 替代方案与适用边界

mDPO 不是唯一可选项。是否用它，取决于你的预算、目标和数据形态。

### 1. 直接用普通多模态 DPO

适用条件：

- 数据量不大
- 目标是快速做一轮偏好对齐
- 任务对图像真实性要求没那么高

优点：

- 实现最简单
- 训练链路短
- 起步成本低

缺点：

- 无法显式约束“回答必须依赖图像”
- 对幻觉问题的改善可能不稳定

更直白地说，普通 DPO 更像：

**先把回答风格、偏好顺序调顺。**

如果任务是开放式问答、泛聊天、多轮助手对齐，它可能已经够用。  
但如果任务是颜色判断、目标存在性判断、位置关系识别，普通 DPO 往往不够稳。

### 2. 上 RLHF

适用条件：

- 预算更充足
- 能维护奖励模型
- 需要更复杂的行为控制

优点：

- 奖励表达能力更强
- 可以编码更复杂的偏好模式
- 策略优化空间更大

缺点：

- 管线长
- 调参更难
- 稳定性与成本压力更高

可以把 RLHF 理解成：

**你不只是要模型“排序对”，还要模型学一整套更复杂的行为策略。**

如果团队没有维护奖励模型和 PPO 的资源，mDPO 往往是更现实的方案。

### 3. 引入更细粒度的偏好数据，例如 VisionPrefer

VisionPrefer 的意义，不是直接替代 mDPO，而是说明：

**多模态偏好本身可能是多维的，不是一条单分数轴。**

它把视觉偏好拆成多个方向：

| 维度 | 白话解释 | 更适合优化什么 |
|---|---|---|
| Prompt-following | 是否按提示执行 | 指令跟随 |
| Fidelity | 内容和结构是否真实 | 视觉真实性 |
| Aesthetic | 是否美观 | 生成质量 |
| Harmlessness | 是否安全 | 安全对齐 |

这对理解 mDPO 很有帮助。  
mDPO 主要关注的是其中偏 **fidelity / evidence-grounding** 的部分，也就是：

- 回答是否真的被图像支持
- 模型是否减少无依据输出

如果你预算有限，但又非常在意“看图是否靠谱”，那就应该优先优化与视觉真实性最相关的偏好维度，而不是一开始就追求全套 RLHF。

### 4. 一个实用选择表

| 目标 | 推荐方案 | 原因 |
|---|---|---|
| 快速低成本对齐 | 普通 DPO | 实现最短，起步快 |
| 重点降低视觉幻觉 | mDPO | 显式约束图像条件 |
| 需要全局复杂行为塑形 | RLHF | 奖励表达能力更强 |
| 希望纳入更细视觉偏好 | DPO / mDPO + 更细粒度偏好数据 | 数据监督更丰富 |

所以更务实的路线通常是：

1. 先用普通 DPO 验证偏好数据是否有效
2. 再用 mDPO 修复“只看文本”的问题
3. 最后在预算足够时，再考虑 RLHF 或多维奖励建模

这条路线的好处是：  
**每一步新增的工程复杂度都有限，但收益来源是清楚的。**

---

## 参考资料

1. Fei Wang et al. *mDPO: Conditional Preference Optimization for Multimodal Large Language Models*. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.460.pdf  
2. Shengzhi Li et al. *Multi-modal Preference Alignment Remedies Degradation of Visual Instruction Tuning on Language Models*. ACL 2024. https://aclanthology.org/2024.acl-long.765.pdf  
3. VisionPrefer dataset report. https://www.emergentmind.com/topics/visionprefer-dataset  
4. Rafael Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. https://arxiv.org/abs/2305.18290  
5. Yuzhe Yang et al. *MMHalBench: A Fine-grained Benchmark for Evaluating Hallucination in Multimodal Large Language Models*. https://arxiv.org/abs/2404.00926  
6. AMBER benchmark/project materials. https://github.com/junyangwang0410/AMBER  
7. LLaVA project page. https://llava-vl.github.io/  
8. Bunny project repository. https://github.com/BAAI-DCAI/Bunny
