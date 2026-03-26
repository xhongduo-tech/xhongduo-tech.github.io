## 核心结论

Flow Matching 的核心不是“逐步加噪再逐步去噪”，而是直接学习一个连续时间速度场 $v_\theta(x,t)$。速度场可以理解为“样本在当前时刻应该朝哪个方向、以多大速度移动”的函数。只要这个函数学对了，模型就能把初始噪声沿着一条可积分的路径推到真实图像分布。

对图像生成来说，最常用的训练目标是条件流匹配（Conditional Flow Matching, CFM）。它先人为规定一条从噪声到数据的解析路径，再让网络回归这条路径的导数。在线性路径下：

$$
x_t=(1-t)x_0+t x_1
$$

其中 $x_0$ 是噪声样本，$x_1$ 是真实图像样本，$t\in[0,1]$ 是时间。对应的目标速度恒为：

$$
\frac{d x_t}{dt}=x_1-x_0
$$

所以训练损失写成：

$$
L(\theta)=\mathbb{E}_{t,x_0,x_1}\|v_\theta(x_t,t)-(x_1-x_0)\|^2
$$

这比 DDPM 的 score matching 更直接。score matching 可以理解为“学习噪声污染后分布的梯度方向”，而 velocity matching 是“直接学习样本怎么移动”。

更进一步，Rectified Flow（整流流）把“路径尽量直”作为目标。直线轨迹的直接收益是 ODE 采样更省步数。ReFlow 则是一个再训练过程：先用当前模型生成一批“噪声到生成样本”的新配对，再用这些新配对继续训练，让轨迹越来越直。工程上，Flux 和 SD3 这一类系统采用整流流后，常见采样步数可以降到 4 到 8 步，而传统 DDPM 常见是几十到上千步。

一个最小玩具例子就能看出这个思路。令 $x_0=[0,0]$，$x_1=[1,1]$，若 $t=0.5$，则：

$$
x_t=[0.5,0.5],\quad v^\*(x_t,t)=x_1-x_0=[1,1]
$$

这说明网络在这个样本上只需要输出常数向量 $[1,1]$，就知道如何把点继续推向目标。

---

## 问题定义与边界

问题定义很明确：给定一个简单的基分布，通常是高斯噪声分布 $p_0$，构造一个连续、可微、可数值积分的运输过程，把样本变成数据分布 $p_1$ 中的图像。这里的“运输”可以理解为“把一个随机点逐渐搬运到真实图像流形附近”。

这类方法的边界主要有三层。

第一层，Flow Matching 解决的是“如何训练一个连续时间生成轨迹”，不是“如何自动找到最优语义结构”。文本条件、图像编码器、Transformer 骨干、VAE 编解码器，这些仍然决定了最终画质和语义对齐能力。整流流只负责生成过程的动力学。

第二层，线性插值路径虽然简单，但不等于真实最优路径。训练时我们规定：

$$
x_t=(1-t)x_0+t x_1
$$

这是人为设计的监督信号，不是数据自己长出来的路径。它的优点是目标明确、导数简单；缺点是在高维空间里，不同配对可能在某个时刻落到相同或相近的 $x_t$，导致网络必须对多个方向求平均。

第三层，是否真的能低步数采样，取决于轨迹是否足够直。很多初学者会误以为“用了 Flow Matching 就一定只要几步”。这不准确。只做初始 CFM 训练，轨迹仍可能弯曲，ODE 还是要走很多小步。Rectified Flow 和 ReFlow 的价值就在这里。

下面这张表先把 DDPM 和 Rectified Flow 的边界差异摆清楚：

| 维度 | DDPM / DDIM | Flow Matching / Rectified Flow |
|---|---|---|
| 基本对象 | score，分布梯度方向 | velocity，样本运动速度 |
| 训练方式 | 逐噪声等级监督 | 连续时间路径导数监督 |
| 采样过程 | 反向扩散，常带随机性 | ODE 积分，通常确定性 |
| 轨迹形状 | 常见为弯曲轨迹 | 目标是尽量直 |
| 典型步数 | 50 到 1000 | 4 到 20，整流后可更低 |
| 主要优势 | 训练成熟、生态完整 | 采样快、便于蒸馏 |
| 主要风险 | 推理慢 | 路径不直时收益下降 |

一个直观反例可以说明“平均速度”问题。假设在某个时间 $t$，两组配对都映射到了相近的 $x_t$：

- 配对 A：噪声点应往右上移动
- 配对 B：噪声点应往左上移动

如果网络只能看到当前点和时间，而无法区分这两个配对来源，它输出的最优均方误差解就会是“向上”的平均方向。结果是原本两条直线目标，被网络学成一条弯曲的折中轨迹。这就是高维交叉耦合带来的问题。

---

## 核心机制与推导

先看最基本的 CFM 训练流程。

1. 从高斯分布采样噪声 $x_0$。
2. 从数据集中采样真实样本 $x_1$。
3. 随机采样时间 $t\sim U[0,1]$。
4. 用线性插值得到中间点 $x_t=(1-t)x_0+t x_1$。
5. 令网络预测 $v_\theta(x_t,t)$。
6. 用目标速度 $x_1-x_0$ 做均方误差监督。

损失函数就是：

$$
L(\theta)=\mathbb{E}_{t,x_0,x_1}\left[\|v_\theta(x_t,t)-(x_1-x_0)\|^2\right]
$$

为什么目标速度是常数？因为线性路径对时间求导后就是：

$$
\frac{d}{dt}\big((1-t)x_0+t x_1\big)=x_1-x_0
$$

所以网络实际上在学一件非常具体的事：给你某个时刻的中间状态 $x_t$ 和时间 $t$，请你输出这条路径此刻的速度。

如果训练完成，从纯噪声 $x(0)=x_0$ 出发，解下面这个 ODE 即可生成：

$$
\frac{d x(t)}{dt}=v_\theta(x(t),t)
$$

Euler 离散化最简单：

$$
x_{t+\Delta t}=x_t+\Delta t\cdot v_\theta(x_t,t)
$$

玩具例子可以把这个过程看得很清楚。设：

- $x_0=[0,0]$
- $x_1=[1,1]$
- $t=0.5$

则：

$$
x_t=[0.5,0.5],\quad v^\*=x_1-x_0=[1,1]
$$

如果网络每次都输出 $[1,1]$，Euler 一步更新就会沿着对角线走向终点。这就是“拉直路径”的最小版本。

Rectified Flow 比普通 Flow Matching 多做了一件事：它不满足于“有一条能走通的路”，而是希望“这条路尽量接近直线”。原因很实际。ODE 数值积分的误差和曲率强相关，轨迹越弯，越需要更多步才能逼近真实终点；轨迹越直，少量步数就够。

ReFlow 的核心思想是迭代重建配对关系。第 $k$ 轮时，设当前模型诱导出的噪声到样本配对分布为 $\pi_k$，再训练损失可写成：

$$
\mathcal{L}^{(k)}_{\text{reflow}}
=
\mathbb{E}_{(x_0,z)\sim\pi_k,\ t}
\left[
\| (z-x_0)-v_\theta(x_t,t)\|^2
\right]
$$

其中：

$$
x_t=(1-t)x_0+t z
$$

这里的 $z$ 可以理解为当前模型从噪声生成出来的样本。与初始训练不同，ReFlow 不再用“真实数据随机配对噪声”，而是用“模型自己实际走出来的配对”。这样做的直觉是：既然模型已经偏好某些运输关系，那就把这些关系固定下来继续训练，减少不同路径在中间态的交叉，轨迹会越来越稳定、越来越直。

真实工程例子是 SD3 和 Flux 这类大模型。它们并不是简单把 DDPM 的损失替换掉，而是把整套生成动力学改成基于流的参数化，再配合强骨干网络和大规模条件建模。最终收益不是“训练更便宜”，而是“在相近质量下推理步数显著降低”，这对线上服务成本影响很大。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，用最小二维例子验证线性插值、目标速度和 Euler 采样。它不是训练神经网络，而是把“真速度”直接写出来，目的是先把数学关系跑通。

```python
import numpy as np

def interpolate(x0, x1, t):
    return (1 - t) * x0 + t * x1

def target_velocity(x0, x1):
    return x1 - x0

def euler_sample(x_init, velocity_fn, steps=4):
    x = x_init.astype(float).copy()
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        x = x + dt * velocity_fn(x, t)
    return x

# 玩具例子
x0 = np.array([0.0, 0.0])
x1 = np.array([1.0, 1.0])
t = 0.5

xt = interpolate(x0, x1, t)
v = target_velocity(x0, x1)

assert np.allclose(xt, np.array([0.5, 0.5]))
assert np.allclose(v, np.array([1.0, 1.0]))

# 若速度场恒为 [1, 1]，4 步 Euler 后应到达终点
x_final = euler_sample(
    x0,
    velocity_fn=lambda x, t: np.array([1.0, 1.0]),
    steps=4,
)
assert np.allclose(x_final, x1)

print("toy flow matching check passed")
```

真正训练时，速度场由神经网络预测。下面是一个新手可读的 PyTorch 风格伪代码，展示 `RectifiedFlow` 的核心接口：

```python
import torch
import torch.nn.functional as F

class RectifiedFlow:
    def __init__(self, model):
        self.model = model  # 输入 x_t, t, cond，输出 velocity

    def interpolate(self, x_noise, x_data, t):
        t = t.view(-1, 1, 1, 1)
        return (1.0 - t) * x_noise + t * x_data

    def target_velocity(self, x_noise, x_data):
        return x_data - x_noise

    def loss(self, x_data, cond=None):
        x_noise = torch.randn_like(x_data)
        t = torch.rand(x_data.shape[0], device=x_data.device)
        x_t = self.interpolate(x_noise, x_data, t)
        v_target = self.target_velocity(x_noise, x_data)
        v_pred = self.model(x_t, t, cond)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, shape, cond=None, steps=8, device="cuda"):
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i / steps, device=device)
            v = self.model(x, t, cond)
            x = x + dt * v
        return x
```

ReFlow 训练循环的关键不是改损失形式，而是改配对来源：

```python
# stage 1: 初始 flow matching
for batch in dataloader:
    x_data, cond = batch
    loss = flow.loss(x_data, cond)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# stage 2: ReFlow
generated_pairs = []
for _ in range(num_pairs):
    z = torch.randn(latent_shape, device=device)
    x_hat = flow.sample(z.shape, cond=None, steps=8, device=device)
    generated_pairs.append((z.cpu(), x_hat.cpu()))

for epoch in range(reflow_epochs):
    for z, x_hat in generated_pairs:
        z = z.to(device)
        x_hat = x_hat.to(device)
        t = torch.rand(z.shape[0], device=device)
        x_t = flow.interpolate(z, x_hat, t)
        v_target = x_hat - z
        v_pred = flow.model(x_t, t, cond=None)
        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

真实工程里通常还会加上：

| 模块 | 作用 |
|---|---|
| VAE latent 空间训练 | 降低像素空间训练成本 |
| 文本编码器 | 提供 prompt 条件 |
| Transformer/MMDiT 骨干 | 建模跨模态条件关系 |
| CFG 或其他条件引导 | 提升文本跟随能力 |
| 蒸馏 student | 进一步压缩到 1 步或少步 |

所以“Rectified Flow”不是一个单独可交付产品，而是现代文生图系统中负责生成动力学的一层设计。

---

## 工程权衡与常见坑

第一类权衡是训练成本和推理成本的交换。Rectified Flow 常见结论是“推理更快”，但这个快不是白来的。为了把轨迹拉直，你往往需要 ReFlow、蒸馏、重新生成训练配对，这些都增加离线训练开销。如果你的业务是离线批量生成，推理速度未必是首要矛盾，传统扩散仍可能更划算。

第二类权衡是路径直线化和模型表达力的交换。轨迹越直，数值积分越容易；但如果约束过强，也可能牺牲复杂分布下的灵活性。因此工程上不是“越直越好”，而是“在目标步数下足够直”。

第三类权衡是采样步数和质量稳定性。很多系统在 4 到 8 步效果很好，但继续压到 1 步时，通常需要专门蒸馏。原因是 teacher 的连续轨迹信息被压成单次映射后，细节恢复难度明显上升。

常见坑可以直接列成表：

| 常见坑 | 现象 | 原因 | 处理方式 |
|---|---|---|---|
| 只做初始 CFM，不做 ReFlow | 8 步结果发糊、结构漂移 | 轨迹仍弯曲 | 增加 1 到 3 轮 ReFlow |
| 随机配对交叉严重 | 中间态速度方向不稳定 | 多条路径在同一区域冲突 | 用模型生成配对再训练 |
| 过早做一步蒸馏 | 细节丢失、风格塌陷 | teacher 轨迹还不够直 | 先保证少步 teacher 稳定 |
| ODE 步数压得过低 | 颜色和几何失真 | 数值误差过大 | 回退到 4 到 8 步 |
| 把速度和噪声预测混用 | 训练不收敛或指标异常 | 参数化定义不一致 | 明确统一 velocity 目标 |

一个新手常见误解是：“既然目标速度是 $x_1-x_0$，那不就是学一个常数向量吗？”这只在玩具例子里成立。真实图像是高维张量，且网络只看到当前状态 $x_t$、时间 $t$ 和条件信息，不直接知道配对的全信息。它必须从局部状态中推断当前应朝哪种全局结构演化，这才是难点。

另一个真实工程坑是评估方式错误。DDPM 常用固定采样器和固定步数评估，而 Rectified Flow 的优劣很大程度上体现在“步数下降后质量是否还能保持”。如果你只比较高步数质量，可能看不出它的主要收益。实际应该至少比较 4、8、16、32 步下的 FID、CLIP 对齐和人工偏好。

---

## 替代方案与适用边界

如果你更关心训练稳定性、复用现有生态、快速搭建 baseline，DDPM/DDIM 仍然是合理选择。它们的原理、采样器、蒸馏方法、控制模块都更成熟。代价是推理慢，尤其在需要大分辨率和高并发时，成本明显偏高。

如果你更关心低步数生成、高吞吐在线服务、后续一步蒸馏潜力，Flow Matching 尤其是 Rectified Flow 更合适。它的核心收益不是“更容易训练”，而是“更适合做高效生成系统”。

可以用一张表快速选型：

| 方案 | 轨迹类型 | 训练目标 | 最低常见采样步数 | 适用边界 |
|---|---|---|---|---|
| DDPM | 弯曲随机轨迹 | score / noise prediction | 数十到上千步 | 经典基线、训练体系成熟 |
| DDIM | 确定性近似轨迹 | 仍基于扩散参数化 | 十几到几十步 | 比 DDPM 快，但仍不算极低步 |
| 普通 Flow Matching | 解析路径，可为线性 | velocity regression | 约 10 到 20 步 | 想用连续流框架，但未做整流 |
| Rectified Flow | 尽量直的确定性轨迹 | velocity regression + path straightening | 4 到 8 步 | 追求高质量低步数生成 |
| ReFlow + 蒸馏 | 近一步映射 | teacher-student 压缩 | 1 到 4 步 | 实时生成、服务成本敏感 |

真实工程例子可以对比两代系统思路。

旧版 Stable Diffusion 的主干是 latent U-Net，训练目标建立在扩散噪声预测或等价参数化上，采样一般依赖 DDPM/DDIM 一类过程。新版 SD3 和 Flux 则转向基于流匹配的动力学建模，并结合更强的 Transformer 类骨干。架构升级的重点不只是“模型更大”，而是“生成过程从反向扩散转为可整流的连续流”，这样才能把推理步数真正压低。

因此结论很明确：如果场景不要求极限速度，传统扩散依然有竞争力；如果目标是高质量、低延迟、可进一步蒸馏的图像生成系统，Rectified Flow 是更合理的方向，但前提是你愿意承担额外的路径整流成本。

---

## 参考资料

1. Gaurav Gandhi, *Flow Matching: Architecture & How It Works*  
   https://nonlinear.technology/blog/flow-matching-architecture-how-it-works/?utm_source=openai

2. EmergentMind, *Flow Matching for Diffusion Training*  
   https://www.emergentmind.com/topics/flow-matching-for-diffusion-training?utm_source=openai

3. EmergentMind, *Rectified Flow Matching*  
   https://www.emergentmind.com/topics/rectified-flow-matching?utm_source=openai

4. SOTAAZ, *Rectified Flow: Straightening Paths Toward 1-Step Generation*  
   https://blog.sotaaz.com/post/rectified-flow-en?utm_source=openai

5. SOTAAZ, *Stable Diffusion 3 & FLUX Architecture*  
   https://blog.sotaaz.com/post/sd3-flux-architecture-en?utm_source=openai
