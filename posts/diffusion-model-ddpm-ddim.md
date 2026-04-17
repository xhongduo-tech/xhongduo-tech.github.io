## 核心结论

DDPM（Denoising Diffusion Probabilistic Model，去噪扩散概率模型）做的事可以压缩成一句话：先用一个固定规则把干净图像一步步加成噪声，再训练一个网络学会“这一步加了什么噪声”，最后把过程倒着走回来。

它的前向扩散过程是一个高斯马尔可夫链。马尔可夫链的白话解释是：下一步只依赖当前步，不直接依赖更早历史。每一步都按同一个结构加一点高斯噪声：

$$
q(x_t \mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)
$$

通常记 $\beta_t=1-\alpha_t$，表示第 $t$ 步注入的噪声强度。于是上式也可写成：

$$
q(x_t \mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

把多步连起来，可以直接写成闭式：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

其中：

$$
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s=\prod_{s=1}^{t}(1-\beta_s)
$$

这条式子很关键。它说明训练时不需要真的一步步加噪，只要随机抽一个时间步 $t$，直接从 $x_0$ 采样出 $x_t$，再让网络预测噪声 $\epsilon$ 或原图 $x_0$ 即可。

DDIM（Denoising Diffusion Implicit Model）不重新训练模型，而是在同一个 denoiser 上重写采样轨迹。它把原本依赖随机项的马尔可夫逆过程，改成非马尔可夫、可确定的路径，因此能把 1000 步左右的采样压到 50 步甚至更少，质量仍然可用。

新手版理解可以只记一句：DDPM 是“一层层擦噪声”，DDIM 是“沿着同一套去噪知识直接走捷径”。

为了避免术语堆在一起，先把本文最重要的三个对象分开：

| 对象 | 含义 | 在训练/采样中的角色 |
|---|---|---|
| $x_0$ | 干净样本，例如原始图像 | 训练监督目标的来源 |
| $x_t$ | 第 $t$ 步的带噪样本 | 模型的实际输入 |
| $\epsilon$ | 加到样本上的高斯噪声 | 最常见的预测目标 |

---

## 问题定义与边界

扩散模型要解决的问题是：如何从随机噪声中采样出一张符合数据分布的干净图像。这里的“数据分布”可以白话理解成“真实图片通常长什么样”。

训练阶段的输入不是纯噪声，而是随机时间步上的带噪样本 $x_t$。模型知道当前时间步 $t$，目标是恢复噪声或恢复原图。最常见的损失是噪声预测目标：

$$
\mathcal L_{\text{simple}}=\mathbb E_{x_0,\epsilon,t}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]
$$

这里的 $\epsilon_\theta(x_t,t)$ 表示“网络看到第 $t$ 步的带噪样本后，对噪声的预测”。

为什么预测噪声是合理的？因为一旦知道了 $x_t$ 和噪声 $\epsilon$，就能反推出原图近似值：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

如果模型预测的是精确噪声，那么 $\hat x_0$ 就会回到真实 $x_0$。这也是“预测噪声”和“恢复原图”本质等价的原因。

采样阶段的边界主要有两个。

第一，步数与质量的权衡。DDPM 原始形式通常需要很多步，因为每一步只改一点点；步数直接砍掉，误差会迅速积累。

第二，随机性与可复现性的权衡。DDPM 的逆过程通常含随机项，所以同一个提示词和种子之外的细节会波动；DDIM 可以走确定性轨迹，更容易复现和做插值，但多样性会下降。

下面这个表格先把三条路径的差别压缩出来：

| 路径 | 依赖结构 | 典型步数 | 是否含随机项 | 适合场景 |
|---|---|---:|---|---|
| 前向扩散 | 马尔可夫 | 训练中任意抽样 | 是 | 构造带噪训练样本 |
| DDPM 反向采样 | 马尔可夫 | 250-1000 | 是 | 追求稳健质量 |
| DDIM 反向采样 | 非马尔可夫 | 20-100 | 可设为否 | 快速生成、可复现 |

玩具例子可以先看一维数值，不看图像。设 $x_0=1$，表示“干净信号”；训练时随机取某个时间步，把它加成 $x_t$；模型只需回答“这里面混进了多少噪声”。

真实工程例子是 Stable Diffusion 一类文生图系统。它们训练出的 U-Net 去噪器可以配不同 scheduler（调度器，白话解释是“决定每一步怎么走的规则”）。切到 `DDIMScheduler` 后，常见做法就是把采样步数从几百步降到 50 步左右。

如果你是第一次接触这类模型，可以先把问题拆成下面三问：

| 问题 | 直接回答 |
|---|---|
| 训练时模型在看什么？ | 看带噪样本 $x_t$ 和时间步 $t$ |
| 模型在输出什么？ | 最常见是输出噪声 $\epsilon$ |
| 采样时为什么能从纯噪声变回图像？ | 因为模型学会了“每一步该去掉多少噪声” |

---

## 核心机制与推导

### 1. 为什么前向过程能闭式采样

从

$$
q(x_t \mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)
$$

出发，可以把第 $t$ 步写成：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,I)
$$

继续展开前两步：

$$
x_t=\sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-1}\right)+\sqrt{1-\alpha_t}\epsilon_t
$$

整理后可见，$x_t$ 是“原图乘上一串系数”再加上“多个高斯噪声线性组合”。由于高斯分布对线性组合封闭，多个噪声项最终仍然能合并成一个标准高斯噪声，于是得到闭式：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

其中 $\bar\alpha_t=\prod_{s=1}^t \alpha_s$。这说明 $x_t$ 本质上是“原图信号”和“高斯噪声”的线性组合。时间越晚，$\bar\alpha_t$ 越小，原图权重越低，噪声权重越高。

这也是扩散模型训练高效的根源：随机采一个 $t$ 就能直接构造监督信号，不必真的跑完整条前向链。

更具体一点，假设某一步有：

$$
\bar\alpha_t=0.81
$$

那么：

$$
x_t=0.9x_0+\sqrt{0.19}\epsilon
$$

这说明此时样本仍然保留较多原图信息；而如果 $\bar\alpha_t$ 已经很小，比如 $0.01$，那么样本几乎已经接近纯噪声。

### 2. 为什么预测噪声和预测原图是等价路线

网络如果输出 $\epsilon_\theta(x_t,t)$，就能还原 $\hat x_0$；反过来如果网络直接输出 $\hat x_{0,\theta}(x_t,t)$，也能推回噪声。两者只是参数化方式不同，本质都在学习“当前这个带噪样本更像哪张干净图”。

术语“参数化”可以白话理解成：同一个目标，用不同输出格式表达。

三种常见参数化方式可以放在一起看：

| 参数化 | 模型输出 | 优点 | 常见场景 |
|---|---|---|---|
| $\epsilon$-prediction | 预测噪声 $\epsilon$ | 实现简单，最常见 | DDPM 基础讲解、早期实现 |
| $x_0$-prediction | 预测干净样本 $x_0$ | 更直接对应重建目标 | 某些图像恢复任务 |
| $v$-prediction | 预测速度变量 $v$ | 数值稳定性常更好 | 较新的 latent diffusion 实现 |

其中 $x_0$ 与 $\epsilon$ 的互相转换最常用：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

$$
\hat \epsilon=\frac{x_t-\sqrt{\bar\alpha_t}\hat x_0}{\sqrt{1-\bar\alpha_t}}
$$

所以不要把“预测噪声”和“预测原图”理解成两套完全不同的模型。多数时候，它们只是同一去噪问题的不同坐标系。

### 3. score function 和分数匹配是什么关系

score function 写成：

$$
s_t(x)=\nabla_x \log p_t(x)
$$

它表示“朝哪个方向移动，样本会更像高概率真实数据”。梯度的白话解释是：函数增长最快的方向。

如果把 $\log p_t(x)$ 理解成“这个样本在第 $t$ 步有多像真实数据”的打分，那么 score 就是在说：把样本往哪里推，会更像真实数据。

分数匹配（score matching）训练的就是这个方向场。去噪扩散模型中的噪声预测目标，与学习 score 存在严格联系。在高斯扰动条件下，预测噪声等价于估计某种归一化后的 score。因此很多文献会把 DDPM 和 score-based model 放在同一框架里理解。

对高斯扰动分布 $q(x_t\mid x_0)$，可以写出：

$$
\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
$$

又因为

$$
x_t-\sqrt{\bar\alpha_t}x_0=\sqrt{1-\bar\alpha_t}\epsilon
$$

所以有：

$$
\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

这条式子说明：如果模型能预测噪声 $\epsilon$，它也就等价地掌握了一个与 score 成比例的方向信息。

这又连到朗之万动力学（Langevin dynamics，白话解释是“按概率梯度往高密度区域走，同时加一点随机扰动”）：

$$
dx=s_t(x)\,dt+\sqrt{2}\,dw_t
$$

其中 $dw_t$ 是布朗运动增量。直观上，score 告诉你“哪里更像真实图像”，随机项避免你卡死在单一路径。扩散模型的逆采样，本质上也是在构造一条从噪声回到数据分布的轨迹。

如果把它翻成不那么抽象的话，可以理解成：

| 数学对象 | 直白含义 |
|---|---|
| $\log p_t(x)$ | 当前位置“像真实数据”的程度 |
| $\nabla_x \log p_t(x)$ | 往哪边走会更像真实数据 |
| 随机扰动项 | 避免只走一条死板路径 |

### 4. DDIM 为什么能少走很多步

DDPM 的逆过程来自对真实后验的高斯近似，通常带随机项；DDIM 的核心观察是：只要训练目标不变，可以构造一族共享同一边际分布的非马尔可夫前向过程，并由此得到不同的逆采样轨迹。

先看 DDPM 常见的一步逆采样形式：

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t), \sigma_t^2 I)
$$

其中均值项通常依赖模型对噪声的预测，方差项则带来随机性。

DDIM 则把采样重写成另一条轨迹。在确定性版本下，DDIM 常写成：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta(x_t,t)
$$

把 $\hat x_0$ 代入，可得：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\left(\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}\right)+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta(x_t,t)
$$

更一般地，如果引入控制随机性的参数 $\eta$，DDIM 一步可写成：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t)+\sigma_t z
$$

其中：

$$
\sigma_t=\eta\cdot
\sqrt{
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}
}
\cdot
\sqrt{
1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}
}
,\quad z\sim\mathcal N(0,I)
$$

这时：

- $\eta=0$：完全确定性，得到常说的 deterministic DDIM
- $\eta>0$：重新注入一部分随机性，增加多样性

它和 DDPM 的关键区别不是“更会去噪”，而是“沿另一条轨迹去噪”。这条轨迹可以跨更大的时间步跳转，因此 1000 步训练得到的模型，推理时可以只选其中 50 个离散时间点。

看一个玩具例子。设：

- $x_0=1$
- $\bar\alpha_1=0.9$
- $\epsilon=0.1$

则前向得到：

$$
x_1=\sqrt{0.9}\cdot 1+\sqrt{0.1}\cdot 0.1\approx 0.9803
$$

如果模型预测 $\epsilon_\theta=0.12$，则反推原图：

$$
\hat x_0=\frac{0.9803-\sqrt{0.1}\cdot 0.12}{\sqrt{0.9}}\approx 0.9933
$$

数值不完全回到 1，因为模型预测有误差。这就是采样误差传播的最小形式。DDIM 的价值在于：即使只取较少步数，也能让误差按一条更平滑、可控的轨迹传播，而不是粗暴删掉 DDPM 中的大量中间步。

真实工程里，这意味着同一个 checkpoint 不改训练，只改 scheduler，就能把推理从“慢但稳”切到“快且基本可用”。

最后把 DDPM 和 DDIM 的差别压成一句更准确的话：

| 问题 | DDPM | DDIM |
|---|---|---|
| 模型是否重训 | 否 | 否 |
| 去噪网络是否同一个 | 是 | 是 |
| 变化发生在哪里 | 逆采样公式 | 逆采样轨迹 |
| 为什么能加速 | 不能直接大幅跳步 | 可以选稀疏时间点跳步 |

---

## 代码实现

训练最小原型不复杂。流程只有三步：

1. 采样干净样本 $x_0$
2. 随机采样时间步 $t$ 和噪声 $\epsilon$，构造 $x_t$
3. 让模型预测 $\epsilon$，最小化均方误差

下面是一段可运行的 Python 玩具代码，验证前向采样、反推关系，以及一个最小 DDIM 更新。代码只依赖 Python 标准库，直接保存为 `toy_diffusion.py` 后执行即可。

```python
import math
import random


def q_sample(x0, alpha_bar_t, eps):
    return math.sqrt(alpha_bar_t) * x0 + math.sqrt(1.0 - alpha_bar_t) * eps


def predict_x0_from_eps(x_t, alpha_bar_t, eps_pred):
    return (x_t - math.sqrt(1.0 - alpha_bar_t) * eps_pred) / math.sqrt(alpha_bar_t)


def ddim_step(x_t, alpha_bar_t, alpha_bar_prev, eps_pred, eta=0.0, noise=None):
    """
    1D DDIM update.
    eta=0 时是确定性 DDIM；eta>0 时加入随机项。
    """
    if not (0.0 < alpha_bar_t <= 1.0 and 0.0 < alpha_bar_prev <= 1.0):
        raise ValueError("alpha_bar_t and alpha_bar_prev must be in (0, 1].")
    if alpha_bar_prev < alpha_bar_t:
        raise ValueError("reverse step expects alpha_bar_prev >= alpha_bar_t.")

    x0_hat = predict_x0_from_eps(x_t, alpha_bar_t, eps_pred)

    sigma_t = eta * math.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * math.sqrt(
        1.0 - alpha_bar_t / alpha_bar_prev
    )
    direction_coeff = math.sqrt(max(1.0 - alpha_bar_prev - sigma_t ** 2, 0.0))
    z = 0.0 if noise is None else noise

    x_prev = math.sqrt(alpha_bar_prev) * x0_hat + direction_coeff * eps_pred + sigma_t * z
    return x_prev, x0_hat, sigma_t


def main():
    # 例 1：前向采样与精确反推
    x0 = 1.0
    alpha_bar_t = 0.9
    eps = 0.1

    x_t = q_sample(x0, alpha_bar_t, eps)
    x0_hat = predict_x0_from_eps(x_t, alpha_bar_t, eps)

    assert abs(x0_hat - x0) < 1e-12

    # 例 2：模型预测存在误差时，x0_hat 会偏离
    eps_pred_bad = 0.12
    x0_hat_bad = predict_x0_from_eps(x_t, alpha_bar_t, eps_pred_bad)
    assert x0_hat_bad < x0

    # 例 3：最小 DDIM 更新
    # 反向时 alpha_bar_prev 通常大于 alpha_bar_t，因为更靠近干净样本
    alpha_bar_prev = 0.96
    x_prev_det, x0_hat_det, sigma_det = ddim_step(
        x_t=x_t,
        alpha_bar_t=alpha_bar_t,
        alpha_bar_prev=alpha_bar_prev,
        eps_pred=eps_pred_bad,
        eta=0.0,
    )

    rng = random.Random(42)
    x_prev_sto, x0_hat_sto, sigma_sto = ddim_step(
        x_t=x_t,
        alpha_bar_t=alpha_bar_t,
        alpha_bar_prev=alpha_bar_prev,
        eps_pred=eps_pred_bad,
        eta=0.5,
        noise=rng.gauss(0.0, 1.0),
    )

    print("x_t =", round(x_t, 6))
    print("x0_hat_bad =", round(x0_hat_bad, 6))
    print("deterministic DDIM -> x_prev =", round(x_prev_det, 6), "sigma =", round(sigma_det, 6))
    print("stochastic DDIM    -> x_prev =", round(x_prev_sto, 6), "sigma =", round(sigma_sto, 6))
    print("same x0_hat?", abs(x0_hat_det - x0_hat_sto) < 1e-12)


if __name__ == "__main__":
    main()
```

这段代码对应三个核心事实：

| 代码片段 | 验证的结论 |
|---|---|
| `q_sample` | 前向过程能直接闭式采样 |
| `predict_x0_from_eps` | 预测噪声可反推原图 |
| `ddim_step` | DDIM 只是改了采样更新规则 |

训练伪代码可以写得更接近真实框架：

```python
# x0: [B, C, H, W]
# alphas_bar[t]: 预先定义好的累计噪声计划
# model: 输入 x_t 和时间步 t，输出 eps_pred

t = randint(1, T, size=(B,))
eps = randn_like(x0)

a_bar = alphas_bar[t].view(B, 1, 1, 1)
x_t = (a_bar.sqrt() * x0) + ((1.0 - a_bar).sqrt() * eps)

eps_pred = model(x_t, t)
loss = ((eps - eps_pred) ** 2).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

如果你是新手，容易困惑的点有两个。

第一，为什么训练时不用真的从 $x_0 \to x_1 \to x_2 \to \cdots \to x_t$ 全部走一遍？因为闭式公式已经允许你直接采样任意一步的 $x_t$。

第二，为什么输入里要有时间步 $t$？因为噪声强度随着步数变化，模型必须知道“当前噪到了什么程度”，否则同一个 $x_t$ 无法判断该去多少噪。

下面补一个可运行的 PyTorch 最小训练样例。它不依赖图像数据集，只用一维高斯混合分布做演示，但训练逻辑与真实扩散训练一致。安装 `torch` 后即可运行。

```python
import math
import torch
import torch.nn as nn
import torch.optim as optim


T = 100
BETAS = torch.linspace(1e-4, 2e-2, T)
ALPHAS = 1.0 - BETAS
ALPHAS_BAR = torch.cumprod(ALPHAS, dim=0)


class TinyDenoiser(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, t):
        # 把时间步归一化到 [0, 1]
        t_embed = t.float().unsqueeze(1) / (T - 1)
        inp = torch.cat([x_t, t_embed], dim=1)
        return self.net(inp)


def sample_x0(batch_size):
    # 两个峰的 1D toy data
    centers = torch.where(torch.rand(batch_size) > 0.5, -2.0, 2.0)
    return (centers + 0.3 * torch.randn(batch_size)).unsqueeze(1)


def q_sample(x0, t, eps):
    a_bar = ALPHAS_BAR[t].unsqueeze(1)
    return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * eps


def main():
    torch.manual_seed(0)
    model = TinyDenoiser()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(2000):
        x0 = sample_x0(batch_size=256)
        t = torch.randint(low=0, high=T, size=(x0.size(0),))
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps)

        eps_pred = model(x_t, t)
        loss = ((eps_pred - eps) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 400 == 0:
            print(f"step={step:4d} loss={loss.item():.6f}")

    # 简单验证：拿一个样本看反推的 x0_hat
    x0 = sample_x0(batch_size=4)
    t = torch.tensor([80, 80, 80, 80])
    eps = torch.randn_like(x0)
    x_t = q_sample(x0, t, eps)
    with torch.no_grad():
        eps_pred = model(x_t, t)
        a_bar = ALPHAS_BAR[t].unsqueeze(1)
        x0_hat = (x_t - (1.0 - a_bar).sqrt() * eps_pred) / a_bar.sqrt()

    print("true x0 :", x0.squeeze().tolist())
    print("pred x0 :", x0_hat.squeeze().tolist())


if __name__ == "__main__":
    main()
```

真实工程例子通常不是自己手写 scheduler，而是直接用 Diffusers：

```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt="a clean technical illustration of a diffusion model pipeline",
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=generator,
).images[0]

image.save("ddim_50_steps.png")
```

这段代码的关键点不是 API 名字，而是两件事：

- 模型权重没变，只替换了 scheduler
- `num_inference_steps=50` 直接控制采样步数

如果想理解 DDIM 在推理时做了什么，可以把它压成一句伪代码：

```python
# 从 x_T ~ N(0, I) 开始
for t in reversed(sampled_timesteps):
    eps_pred = model(x_t, t)
    x0_hat = (x_t - sqrt(1 - a_bar_t) * eps_pred) / sqrt(a_bar_t)
    x_t = sqrt(a_bar_prev) * x0_hat + sqrt(1 - a_bar_prev) * eps_pred
```

如果要写得更接近带 $\eta$ 的正式实现，则是：

```python
for t in reversed(sampled_timesteps):
    eps_pred = model(x_t, t)
    x0_hat = (x_t - sqrt(1 - a_bar_t) * eps_pred) / sqrt(a_bar_t)

    sigma_t = eta * sqrt((1 - a_bar_prev) / (1 - a_bar_t)) * sqrt(1 - a_bar_t / a_bar_prev)
    noise = randn_like(x_t) if eta > 0 else 0.0

    x_t = (
        sqrt(a_bar_prev) * x0_hat
        + sqrt(1 - a_bar_prev - sigma_t**2) * eps_pred
        + sigma_t * noise
    )
```

---

## 工程权衡与常见坑

最大误区是：以为“DDPM 训练 1000 步，所以推理时直接改成 50 步就行”。这通常会直接掉质量，因为原来的逆过程假设了密集时间离散，硬砍步数等于放大每一步的近似误差。

实践里要区分四种情况：

| 方案 | 速度 | 质量稳定性 | 多样性 | 常见问题 |
|---|---|---|---|---|
| 1000 步 DDPM | 慢 | 高 | 高 | 推理成本大 |
| 50 步直接硬砍 DDPM | 快 | 低 | 不稳定 | 伪影、结构崩坏 |
| 50 步 DDIM deterministic | 很快 | 高于硬砍 | 较低 | 结果偏“收敛到固定风格” |
| 50 步 DDIM + $\eta$ | 快 | 中高 | 中高 | 参数不好调时会发散 |

这里的 $\eta$ 可以白话理解成“往确定性路径里重新掺一点随机噪声”。完全确定的 DDIM 适合做复现、图像编辑和插值；如果你希望同一个提示词能采样出更多不同细节，就需要加回少量随机项。

一个新手可验证的小实验是：固定 prompt 和 seed，分别跑 `eta=0` 与 `eta>0`。你会观察到：

- `eta=0` 时输出更稳定，几乎每次都一样
- `eta>0` 时纹理和布局变化更大，但偶尔会出现多余噪点或结构飘移

真实工程中还有几个常见坑。

第一，训练参数化和 scheduler 假设要对齐。你训练的是 `epsilon prediction`、`x0 prediction` 还是 `v prediction`，会影响推理公式；错配时，图像可能不报错但质量明显变差。

第二，步数减少后，classifier-free guidance 往往更敏感。指导强度太大，少步采样时容易过曝、边缘发硬、细节重复。

第三，低步数下的时间步采样策略很重要。不是简单均匀抽 50 个点就总是最优，很多实现会使用特定 spacing 规则。

第四，确定性采样不等于“绝对更好”。它只是更可控。如果你的业务目标是批量生成更多候选图，纯 deterministic 反而可能限制探索空间。

再补三类新手经常踩到、但文档里不一定直接强调的坑。

| 坑点 | 现象 | 直接原因 | 处理方式 |
|---|---|---|---|
| scheduler 与训练配置不一致 | 图像能出，但风格怪、细节差 | 预测目标或时间步定义不匹配 | 检查 `prediction_type`、beta schedule、timestep spacing |
| 步数太少仍用高 guidance | 颜色过曝、边缘发硬 | 每一步条件引导被放大 | 先降低 guidance，再加步数 |
| 以为 `eta=0` 永远最佳 | 结果稳定但变化太少 | 完全确定性抑制多样性 | 根据任务目标选择 `eta` |

如果要做实验对比，建议固定下面四个量，只改一个变量：

1. 模型权重
2. prompt
3. 随机种子
4. 输出分辨率

否则你看到的差异很可能不是 scheduler 带来的。

---

## 替代方案与适用边界

如果你的目标是最高质量，而不是最低延迟，原始 DDPM 或更完整的 score-based sampler 仍然有价值。它们走得更细，随机性更充分，对复杂分布的覆盖通常更稳。

如果你的目标是快速生成、固定 seed 可复现、做图像反演或潜空间插值，DDIM 更合适。因为它的轨迹更稳定，输入条件稍变时输出也更可控。

但 DDIM 不是终点。工程里常见的替代 scheduler 还包括 PLMS、Euler、DPM-Solver、UniPC 等。它们共同目标都是：在更少步数下逼近原始逆过程，同时尽量少损失质量。

可以把它们的适用边界压成下面这张表：

| 方法 | 优势 | 劣势 | 适用场景 |
|---|---|---|---|
| DDPM | 理论直观、质量稳 | 慢 | 离线高质量生成 |
| DDIM | 快、可复现、易插值 | 多样性下降 | 交互式生成、编辑 |
| Score + Langevin | 理论统一、随机性强 | 采样慢、实现复杂 | 研究与理论分析 |
| DPM-Solver 类 | 少步质量高 | 实现依赖较新框架 | 生产推理优化 |

如果把这些方法按“原理重心”再分一层，可以更容易建立整体地图：

| 方法族 | 核心思想 | 你需要先懂什么 |
|---|---|---|
| DDPM | 学会逐步去噪 | 前向加噪、噪声预测 |
| DDIM | 改采样路径而非改模型 | DDPM 逆过程、时间步跳跃 |
| Score-based | 直接学概率密度梯度 | score、分数匹配、SDE |
| Solver 类 | 把采样看成数值积分问题 | ODE/SDE 离散化、误差控制 |

如果在 Diffusers 中切换到别的 scheduler，通常只需几行：

```python
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
```

这也是扩散模型工程化的一个关键现实：多数情况下，训练不是最先动的，先动的是 scheduler。

最后给出一个边界判断：

- 想讲清原理，先学 DDPM
- 想理解 score 与概率梯度，补分数匹配和 Langevin
- 想把同一个模型跑得更快，先看 DDIM
- 想在生产里进一步压步数，再比较 DPM-Solver 一类高阶方法

把选择逻辑压成一句更实用的话：

| 你的目标 | 优先看什么 |
|---|---|
| 先把基本机制学明白 | DDPM |
| 想理解“为什么预测噪声也算在学分布” | score matching |
| 想在不重训的前提下降低推理延迟 | DDIM |
| 想做生产级少步推理优化 | DPM-Solver / UniPC |

---

## 参考资料

| 出处 | 聚焦点 | 用途/推荐阅读阶段 |
|---|---|---|
| Ho et al., *Denoising Diffusion Probabilistic Models* | DDPM 原始建模与训练目标 | 第一性原理论文，适合建立正式定义 |
| Song et al., *Denoising Diffusion Implicit Models* | DDIM 非马尔可夫、快速采样 | 理解为何同一模型能少步推理 |
| Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* | 分数匹配与 score-based generative modeling | 理解 DDPM 与 score 路线的统一视角 |
| Yang Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* | score、SDE、逆向采样统一框架 | 进阶理解扩散与随机微分方程 |
| Hugging Face Diffusers `DDIMScheduler` 文档 | 工程接口与参数 | 实际落地配置，适合直接上手 |
| Lilian Weng, *What are Diffusion Models?* | 对 DDPM、score、采样的系统讲解 | 入门到中级，适合建立全景图 |
| OpenAI / community blog posts on diffusion samplers | scheduler 直觉与经验配置 | 工程实践补充阅读 |
| Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* | 采样设计空间、噪声调度、工程经验 | 生产优化与方法比较 |

如果只按阅读顺序给一个最小路线，可以这样排：

1. 先读 Ho et al.，搞清楚前向加噪、闭式采样、噪声预测目标。
2. 再读 Song et al. 的 DDIM，理解“同一模型、不同采样轨迹”。
3. 然后补 score matching 与 SDE 文章，把扩散模型放进更统一的概率建模框架。
4. 最后看 Diffusers 文档，把公式映射到实际代码参数。
