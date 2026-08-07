---
title: 随机梯度与半梯度方法
date: 2026-08-07
---

# 随机梯度与半梯度方法

<div class="epigraph">
<p>往山下走一步，再走一步——但若山脚在移动，就得想清楚你追的是什么。</p>
<footer>—— 改编自理查德 · 萨顿（Richard S. Sutton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第9章 §9.3 ｜ 2026-08-07</p>
</div>

## 为什么「梯度下降」在 RL 里要打折扣

上一课确立了目标：最小化均方价值误差 $\overline{\text{VE}}(\mathbf{w})$。监督学习顺手就是梯度下降——但强化学习有两个拦路虎：**真值 $v_\pi$ 未知**，只能拿采样目标顶替；**目标自身含参数**，对它求梯度会把问题搅浑。**随机梯度下降（stochastic gradient descent，SGD）** 处理第一个难题（用样本目标估计真梯度方向）；**半梯度（semi-gradient）** 处理第二个难题（把目标当常数，只对「当前项」求导）。这一课是函数逼近预测方法的理论心脏：**半梯度不收敛到 $\overline{\text{VE}}$ 的最小点，却稳定收敛到一个「TD 不动点」**——这个「次优但稳定」的取舍，定义了 RL 逼近方法的全部气质。<span class="marginnote">「半」字的分寸：完整梯度需要对「目标」也求导，但目标里的 $\hat v(S_{t+1},\mathbf w)$ 随 $\mathbf w$ 变化，把它的梯度也算进去会让更新不稳定（第11章讲离策略发散时还会再遇）。半梯度只保留「当前项」的梯度——丢了收敛到全局最小，换来稳定。</span>

## 1 随机梯度下降：用样本估计梯度

对 $\overline{\text{VE}}$ 的梯度，逐状态展开：

$$
\nabla \overline{\text{VE}}(\mathbf{w}) \;=\; \sum_s \mu(s)\,\big(\hat{v}(s,\mathbf{w}) - v_\pi(s)\big)\,\nabla \hat{v}(s,\mathbf{w})
$$

真值 $v_\pi$ 未知，但若我们能对每个状态拿到一个**无偏的回报样本** $U_t$（如 MC 回报 $G_t$），那么 $\big(\hat{v}(S_t,\mathbf{w}) - U_t\big)\nabla\hat{v}(S_t,\mathbf{w})$ 就是该梯度的无偏估计——它的期望等于上面的梯度（差一个符号与权重常数）。于是 **SGD 更新**：

$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t - \tfrac{1}{2}\alpha\,\nabla\big[U_t - \hat{v}(S_t,\mathbf{w}_t)\big]^2 \;=\; \mathbf{w}_t + \alpha\big[U_t - \hat{v}(S_t,\mathbf{w}_t)\big]\nabla \hat{v}(S_t,\mathbf{w}_t)
$$

**只要目标 $U_t$ 是 $v_\pi(S_t)$ 的无偏估计、且步长满足 Robbins–Monro 条件，SGD 就收敛到 $\overline{\text{VE}}$ 的（局部）最小值**——这是函数逼近预测里少数「干净」的结论之一。<span class="marginnote">关键条件在「无偏」二字：MC 回报 $G_t$ 无偏（方差大）；TD 目标 $R+\gamma\hat v(S_{t+1},\mathbf w)$ 有偏（因为用了不准确的 $\hat v$）。无偏 → 标准 SGD 收敛；有偏 → 必须降级为半梯度。</span>

## 2 半梯度 TD(0)：把目标当常数

如果目标 $U_t$ 里含参数（TD 目标、n步目标都是如此），还拿它对 $\mathbf{w}$ 求导，就会把「目标随参数移动」也计进梯度——通常导致发散或振荡。**半梯度方法**干脆规定：**对 $U_t$ 不求导**。半梯度 TD(0) 的更新是：

$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t + \alpha\big[\underbrace{R_{t+1} + \gamma\,\hat{v}(S_{t+1},\mathbf{w}_t)}_{\text{目标：视为常数}} - \hat{v}(S_t,\mathbf{w}_t)\big]\nabla \hat{v}(S_t,\mathbf{w}_t)
$$

对比完整梯度，缺的是「对 $\gamma\hat{v}(S_{t+1},\mathbf{w}_t)$ 关于 $\mathbf{w}$ 的导数项」。它的行为特征：

- **不收敛到 $\overline{\text{VE}}$ 最小值**，而是收敛到**TD 不动点（TD fixed point）**——一个由「当前目标自洽」决定的折中解。
- 但在**线性参数化**下，半梯度 TD(0) 的收敛性是**可靠**的（步长衰减时收敛到不动点），且收敛解通常误差不大。
- **表格型里半梯度 TD(0) 退化为标准 TD(0)**——所以你在第6章学的一切都没有白费，只是被包进了更大的框架。<span class="marginnote">半梯度的「妥协」换来两个实际好处：方差小（目标是自举的）、可在线更新。教材的观点很务实：在函数逼近里，TD 的半梯度不动点「虽不是全局最优，却是稳定可得的最优」——工程上先要稳，再谈优。</span>

## 3 n步半梯度：把 n 步回报当目标

同样的半梯度思想直接搬到 n步。目标换成 **n步回报** $G_{t:t+n}$（视为常数）：

$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t + \alpha\big[G_{t:t+n} - \hat{v}(S_t,\mathbf{w}_t)\big]\nabla \hat{v}(S_t,\mathbf{w}_t)
$$

这里的 $G_{t:t+n}$ 含 $\gamma^n \hat{v}(S_{t+n},\mathbf{w})$——仍然有参数，仍然不求导（半梯度）。n步半梯度在**线性情形**下也是稳定收敛的，且 n 越大、收敛解越接近 $\overline{\text{VE}}$ 最小值（因为自举成分被推迟），代价是方差增大——**第7章的 U 形权衡在函数逼近里原样重演**。<span class="marginnote">第7章学的「n 是偏差-方差旋钮」在这里没有过时，只是旋钮的作用对象从「表格里的值」变成「参数化的值」。这也暗示了第12章 TD(λ)：用 λ 把所有 n 的回报加权平均，一次性收获「n 的多样性」。</span>

## 4 公式解析：一条更新里的「梯度」与「非梯度」

$$
\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\underbrace{\big[\,R_{t+1} + \gamma\,\hat{v}(S_{t+1},\mathbf{w}_t) - \hat{v}(S_t,\mathbf{w}_t)\,\big]}_{\text{TD 误差 } \delta_t}\underbrace{\nabla \hat{v}(S_t,\mathbf{w}_t)}_{\text{当前项的梯度}}
$$

- **第一步，认误差**：$\delta_t = R_{t+1} + \gamma\hat{v}(S_{t+1},\mathbf{w}_t) - \hat{v}(S_t,\mathbf{w}_t)$——和第6章的 TD 误差一模一样的量，只是 $\hat{v}$ 现在带参数。它告诉更新「往哪个方向调」。
- **第二步，认梯度**：$\nabla\hat{v}(S_t,\mathbf{w}_t)$ 是当前状态的梯度，指示「$\mathbf{w}$ 朝哪个方向调，能最快改变 $\hat{v}(S_t)$」。两项相乘：误差大 → 步子大；梯度陡 → 调得猛。
- **第三步，认缺失**：$\gamma\hat{v}(S_{t+1},\mathbf{w}_t)$ 里的 $\mathbf{w}_t$ **没有被求导**——这就是「半」。若把它也求导，梯度里会多出 $\gamma\nabla\hat v(S_{t+1})$ 项，正是这一项在离策略+自举+函数逼近下引发发散（第11章）。<span class="marginnote">一个记忆锚点：完整梯度是「追着当前样本的最小化」，半梯度是「追着一个移动目标的稳定追随」。前者可能追空，后者总追得着——虽然追到的不是全局最优。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为「半梯度只是 SGD 的一个实现细节」。它们的目标根本不同：SGD 最小化 $\overline{\text{VE}}$（需无偏目标），半梯度收敛到 TD 不动点（容忍有偏目标）。**把 MC 目标换成 TD 目标，不只是「换了个目标函数」，而是整个收敛点都变了**——这也是教材坚持把二者分开讲的深层原因。

**另一个易错点**：混淆「步长 $\alpha$ 是否要衰减」。半梯度 TD(0) 在线性情形，**即使 $\alpha$ 固定也能收敛到不动点附近的界内**（不会无界漂移）；但若要收敛到精确不动点，仍需 $\sum\alpha=\infty,\sum\alpha^2<\infty$。工程上常见「固定小 $\alpha$」，学到的不是严格收敛点而是「准稳态」——知道这点，看论文曲线就不会被「没衰减也稳了」误导。

**第三个易错点**：把非线性参数化（神经网络）当线性一样分析。**半梯度 TD 的收敛性证明只对线性情形成立**；神经网络（非线性）下连「收敛到不动点」都未必保证——第14篇深度 RL 的种种工程技巧，大半是为了在非线性下「假装它是线性」地稳住训练。

## 6 小结

- **SGD**：用无偏样本目标 $U_t$ 估计 $\overline{\text{VE}}$ 梯度；无偏目标 → 收敛到（局部）最小。
- **半梯度**：对含参数的目标不求导；有偏但稳定，收敛到**TD 不动点**而非全局最小。
- 半梯度 TD(0)：$\mathbf{w} \leftarrow \mathbf{w} + \alpha\delta_t\nabla\hat v(S_t,\mathbf{w})$；表格型下退化为标准 TD(0)。
- **n步半梯度**：目标换 n步回报，n 仍是偏差-方差旋钮。
- 线性参数化下收敛性可靠；非线性下只是工程性稳定，理论保证薄弱。

在下一节，我们看向函数逼近的「主力形态」——**线性方法**：$\hat v = \mathbf{w}^\top\mathbf{x}(s)$，以及如何用多项式基、傅里叶基等特征构造把原始状态喂进线性回归。
