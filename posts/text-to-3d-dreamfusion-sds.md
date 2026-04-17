## 核心结论

DreamFusion 的关键创新，不是重新训练一个会直接输出 3D 的大模型，而是把已经学会“文本到图像”的 2D 扩散模型，当成一个外部评价器，用它来指导 3D 表示的优化。这里的“扩散模型”可以先理解成一种会判断“当前图像是否像目标文本描述”的概率模型；“3D 表示”可以先理解成一组可被渲染成图片的参数。

它的核心闭环很简单：给定文本提示词 $y$，维护一个待优化的 3D 参数 $\theta$，从随机相机位姿 $c$ 渲染出图像

$$
x = g(\theta, c)
$$

再对图像加噪：

$$
x_t = \alpha_t x + \sigma_t \epsilon,\quad \epsilon \sim \mathcal N(0, I)
$$

然后把 $x_t$、文本 $y$ 和时间步 $t$ 送入扩散模型，让扩散模型给出噪声预测 $\epsilon_\phi(x_t, y, t)$。如果预测结果与真实加入的噪声 $\epsilon$ 不一致，就说明当前渲染图像还不够符合文本先验，于是把这个偏差反向传回 3D 参数 $\theta$。

一句话概括：DreamFusion 是“用 2D 扩散模型反向雕刻 3D”。

| 术语 | 白话解释 | 在 DreamFusion 里的作用 |
|---|---|---|
| $\theta$ | 3D 物体当前的“内部参数” | 被优化的对象 |
| $c$ | 相机位置和朝向 | 决定从哪个视角看物体 |
| $g(\theta,c)$ | 可微渲染器，即“能传梯度的拍照器” | 把 3D 参数变成 2D 图像 |
| $\epsilon_\phi$ | 扩散模型预测的噪声 | 提供文本一致性的优化信号 |
| SDS | Score Distillation Sampling，得分蒸馏采样 | 把 2D 先验转成 3D 梯度 |

玩具例子可以这样理解：文本是“a red chair”。系统先随便猜一个 3D 体积，从某个角度渲染出来，可能像一团红色噪声。扩散模型会给出“这张图不像红椅子”的梯度，于是 3D 参数被更新。重复很多轮后，不同视角下都开始接近“红椅子”的图像分布。

---

## 问题定义与边界

DreamFusion 解决的问题是：

给定文本 $y$，找到一个 3D 表示 $\theta$，使得它在多视角渲染后，生成的图像都与文本语义一致。形式上可以写成一种理想目标：

$$
\max_\theta\; p\big(y \mid g(\theta, c)\big)
$$

但实际训练不会直接最大化这个概率，而是使用 SDS 提供的梯度近似。

这里要先划清边界。DreamFusion 追求的是“语义正确的 3D 原型”，不是“可直接交付的生产级 3D 资产”。“原型”可以理解成用于验证概念、讨论外形和大致结构的模型；“生产级资产”则要求拓扑干净、法线稳定、UV 完整、贴图可控、适合绑定或物理仿真。

真实工程里，这个边界非常重要。比如提示词是“a ceramic teapot with blue floral patterns”，DreamFusion 往往能生成一个从常见视角看起来像陶瓷茶壶的 3D 原型，但它不保证：
1. 壶嘴与壶身连接处拓扑正确。
2. 纹样在背面仍然清晰一致。
3. 网格适合直接导入游戏引擎。
4. 材质参数符合 PBR 工作流。

| 能做 | 不能直接保证 |
|---|---|
| 文本驱动的单物体 3D 原型 | 生产级拓扑 |
| 多视角语义大致一致 | 精确机械结构 |
| 无 3D 数据下的概念生成 | 直接可用的 UV/贴图 |
| 研究和快速原型验证 | 一步到位资产交付 |

因此，判断 DreamFusion 是否合适，标准不是“它能不能生成 3D”，而是“你要的是原型，还是最终资产”。

---

## 核心机制与推导

DreamFusion 的核心机制由五步组成：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 采样相机 | 随机 $c$ | 一个视角 | 强制模型兼顾多视角 |
| 可微渲染 | $\theta, c$ | 图像 $x$ | 把 3D 表示转成 2D |
| 加噪 | $x, \epsilon, t$ | $x_t$ | 对齐扩散模型训练分布 |
| 噪声预测 | $x_t, y, t$ | $\epsilon_\phi$ | 获得文本条件下的方向信号 |
| 梯度回传 | $\epsilon_\phi-\epsilon$ | $\nabla_\theta$ | 更新 3D 参数 |

先看扩散模型一侧。扩散模型训练时学的是“给一张被加噪的图，预测其中噪声”。在常见参数化下，它预测的是 $\epsilon_\phi(x_t, y, t)$，对应的 score，也就是“概率密度上升方向”，可近似写成：

$$
s_\phi(x_t, y, t) \approx -\frac{\epsilon_\phi(x_t, y, t)}{\sigma_t}
$$

DreamFusion 的关键想法是：既然 score 表示“往更像真实图像分布的方向走”，那就可以把这个方向蒸馏到 3D 参数上。于是得到 SDS 梯度形式：

$$
\nabla_\theta L_{\text{SDS}} \approx \mathbb E_{t,\epsilon}\left[w(t)\cdot\big(\epsilon_\phi(x_t,y,t)-\epsilon\big)\cdot \frac{\partial x}{\partial \theta}\right]
$$

这条式子要这样读：

- $\epsilon$ 是你真的加进去的噪声。
- $\epsilon_\phi$ 是扩散模型认为这张图里“应该解释为噪声”的部分。
- 如果两者差距很大，说明当前渲染图像不处在文本条件下的高概率区域。
- $\frac{\partial x}{\partial \theta}$ 负责把图像空间的修改意见，传回 3D 参数空间。
- $w(t)$ 是时间步权重，用来控制不同噪声强度对训练的影响。

这里有一个常被误解的点：SDS 更像“定义了一个可用的梯度场”，而不是一个总能写成漂亮闭式的普通监督损失。工程上你真正依赖的是这条梯度更新规则，而不是某个容易解释的标量 loss 数值。

玩具数值例子最容易看懂。设当前只有一个标量像素：

$$
x = 0.50,\ \alpha_t = 0.8,\ \sigma_t = 0.6,\ \epsilon=0.20
$$

则加噪后：

$$
x_t = 0.8\times 0.50 + 0.6\times 0.20 = 0.52
$$

如果扩散模型给出噪声预测 $\epsilon_\phi=0.50$，又设 $w(t)=2$，那么像素空间的梯度项近似为：

$$
2\times(0.50-0.20)=0.60
$$

如果学习率 $\eta=0.1$，并且这里把 $\frac{\partial x}{\partial \theta}$ 简化成 1，那么一次更新约为 $-0.06$。这表示参数会朝着“让下一次渲染更符合文本先验”的方向移动。

为什么 DreamFusion 常用 NeRF 一类表示？因为 NeRF 本质上是一个连续场函数，可以把空间位置和视线方向映射到密度与颜色；“连续场”可以先理解成“不是固定顶点网格，而是任意空间点都能查询属性的函数表示”。这种表示天然支持体渲染和梯度回传，适合在随机视角下被 2D 扩散模型持续监督。

---

## 代码实现

实现上通常分三层：

| 模块 | 典型实现 | 说明 |
|---|---|---|
| 3D 表示 | NeRF / implicit field | 用连续函数表示体密度和颜色 |
| 渲染器 | differentiable volume renderer | 既生成图像，也传回梯度 |
| 文本先验 | 预训练扩散模型 | 提供 SDS 噪声预测 |
| 优化器 | Adam | 更新 $\theta$ |

训练循环的主线如下：采样相机、渲染图像、加噪、预测噪声、构造 SDS 梯度、更新 3D 参数。下面给一个可运行的最小 Python 例子。它不是完整 DreamFusion，而是把 SDS 的单步更新压缩成一个可验证的数值玩具模型。

```python
def sds_update(theta, alpha_t, sigma_t, eps, eps_pred, w_t, lr):
    # 这里把 x = theta，且 dx/dtheta = 1，作为最简玩具设定
    x = theta
    x_t = alpha_t * x + sigma_t * eps
    grad = w_t * (eps_pred - eps)
    new_theta = theta - lr * grad
    return x_t, grad, new_theta


theta = 0.50
alpha_t = 0.8
sigma_t = 0.6
eps = 0.20
eps_pred = 0.50
w_t = 2.0
lr = 0.1

x_t, grad, new_theta = sds_update(theta, alpha_t, sigma_t, eps, eps_pred, w_t, lr)

assert abs(x_t - 0.52) < 1e-9
assert abs(grad - 0.60) < 1e-9
assert abs(new_theta - 0.44) < 1e-9

print(x_t, grad, new_theta)
```

完整训练伪代码可以写成：

```python
for step in range(num_steps):
    c = sample_camera()
    x = renderer(theta, c)

    t = sample_timestep()
    eps = sample_noise_like(x)
    x_t = alpha[t] * x + sigma[t] * eps

    eps_pred = diffusion_model(x_t, text_prompt, t)
    grad_signal = w[t] * (eps_pred - eps)

    loss = (grad_signal.detach() * x).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

上面这个 `loss` 写法体现了工程里的一个常见技巧：虽然 SDS 本质上是梯度规则，但在自动求导框架里，常通过“构造一个拥有目标梯度的代理 loss”来触发反向传播。

真实工程例子：如果团队要做“电商家居概念图到 3D 草模”，流程通常不是 DreamFusion 直接出最终资产，而是：
1. 文本生成一个茶几、餐椅或花瓶的 3D 原型。
2. 从 NeRF 提取 mesh。
3. 美术或技术美术清理网格、重拓扑、补洞。
4. 再进入 UV、材质烘焙和引擎导入流程。

这里 DreamFusion 负责“语义原型生成”，后处理负责“资产生产化”。

| 配置项 | 作用 |
|---|---|
| `CFG` | 文本约束强度，过大会导致过饱和与假细节 |
| `timestep schedule` | 决定训练时主要使用哪些噪声级别 |
| `camera range` | 控制视角覆盖，影响多视角一致性 |
| `density regularization` | 抑制漂浮物和雾状体积 |
| `background model` | 控制背景颜色或环境，减少错误归因 |

---

## 工程权衡与常见坑

DreamFusion 最难的部分不是“把文本变成 3D”，而是“让生成结果既像文本，又在 3D 上站得住”。

第一个常见坑是 Janus 问题。Janus 可以先理解成“模型在不同视角下长出多个互相冲突的正面”。根源是 2D 扩散模型只懂图像分布，不真正理解单一 3D 实体的全局几何约束。如果相机采样偏窄，模型可能学会“每个常见视角都画一个看起来像正面的人脸”，而不是学习一个统一几何体。

第二个坑是纹理污染几何。因为 SDS 只关心渲染图像是否像目标文本，它可能用错误方式讨好扩散模型，例如通过高频纹理假装几何细节。工程上常见处理是先做几何阶段，用较弱材质或 Lambertian 假设稳定形体，再做纹理阶段。

第三个坑是密度漂浮。NeRF 类表示容易出现漂浮小块、雾状体积和空中碎片，因为体渲染允许很多“从单个视角看合理、从 3D 看脏乱”的解。于是需要额外的密度稀疏正则、熵正则或 occupancy 约束。

第四个坑是 guidance 过强。CFG 可以先理解成“把文本条件的声音放大多少倍”。过大的 CFG 确实能让文本相关性变强，但也会把噪声预测推向极端，导致假高光、过锐化、色块脏污和不稳定表面。

| 常见坑 | 表现 | 原因 | 常见缓解手段 |
|---|---|---|---|
| CFG 过大 | 过饱和、假高光、脏纹理 | 文本引导过强 | 降低 guidance scale |
| 视角不足 | Janus、多脸、多前视 | 多视角约束不够 | 扩大相机采样范围 |
| 密度初始化差 | 漂浮物、雾体积 | 体表示自由度太高 | 稀疏正则与更稳初始化 |
| 纹理过早介入 | 几何被花纹污染 | 2D 纹理补偿 3D 缺陷 | 分阶段训练 |
| SDS 不稳 | 局部塌缩、细节乱跳 | 梯度方差大 | 调整时间步采样和正则 |

如果只从研究演示看 DreamFusion，很容易误以为它已经接近“文本直出高质量 3D 资产”。真实工程里更常见的做法是把它放在原型链路前端：先快速得到有讨论价值的形体，再交由传统 3D 管线修复。

---

## 替代方案与适用边界

DreamFusion 的优势很明确：不需要 3D 训练数据，只要一个强大的 2D 扩散模型和可微渲染器，就能把文本先验迁移到 3D。它适合“没有现成 3D 数据，但想快速探索概念”的场景。

它的局限也同样明确：SDS 只是在 2D 图像分布下优化 3D 参数，因此先天存在 3D 约束不足的问题。后续方法的改进，主要集中在两个方向。

第一类改进是优化过程更稳。ProlificDreamer 的代表思路是把“单个 3D 点估计”扩展成“分布层面的生成与优化”，核心目标是减少 SDS 的高方差和模式塌缩问题。直观说，DreamFusion 更像盯着一个当前解反复修；ProlificDreamer 试图维护更丰富的候选分布，让搜索空间不那么容易陷入坏局部最优。

第二类改进是表示和正则更强。包括更好的 density initialization、更合理的多阶段训练、mesh-friendly 表示、法线或表面约束等。这些方法不一定改变 SDS 本身，但会显著影响最终质量。

| 方案 | 优点 | 局限 | 适合场景 |
|---|---|---|---|
| DreamFusion | 不需要 3D 数据，概念清晰 | 易不稳，3D 约束弱 | 文本到 3D 原型 |
| ProlificDreamer | 质量和稳定性通常更好 | 仍不是生产级资产 | 更高质量研究原型 |
| 传统人工建模 | 可控性最强 | 成本高、速度慢 | 正式生产资产 |
| 检索/拼装式方法 | 结构可控、可复用 | 创造性受限 | 工业零件、规则场景 |

适用边界可以直接记成一句话：

如果目标是“快速讨论外形”，DreamFusion 及其改进方法很有价值；如果目标是“严格可控、可交付、可维护的生产资产”，它们通常只能作为前置原型工具，而不是最终管线。

---

## 参考资料

| 资料 | 用途 |
|---|---|
| DreamFusion 官方页: https://dreamfusion3d.github.io/ | 方法总览与结果展示 |
| DreamFusion 论文: https://arxiv.org/abs/2209.14988 | SDS 与文本到 3D 的理论来源 |
| ProlificDreamer 官方页: https://ml.cs.tsinghua.edu.cn/prolificdreamer/ | 后续改进思路总览 |
| ProlificDreamer 代码: https://github.com/thu-ml/prolificdreamer | 工程实现与训练细节 |
| 各项目 README 与配置说明 | 三阶段训练、`scale`、`n_particles`、正则项等超参解释 |
