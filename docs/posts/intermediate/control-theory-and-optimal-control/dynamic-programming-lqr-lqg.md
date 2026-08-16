---
title: 动态规划与 LQR/LQG（HJB 方程、卡尔曼滤波）
date: 2026-08-07
---

# 动态规划与 LQR/LQG（HJB 方程、卡尔曼滤波）

<div class="epigraph">
<p>An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.（一个最优策略具有这样的性质：无论初始状态与初始决策如何，剩余的决策都必须相对第一个决策产生的状态构成最优策略。）</p>
<footer>—— 理查德 · 贝尔曼（Richard Bellman，*Dynamic Programming*，1957）</footer>
</div>

<div class="article-byline">
<p>第二级 · 控制论与最优控制 ｜ Kirk《最优控制理论导论》Ch. 5 · Ogata《现代控制工程》Ch. 10 ｜ 2026-08-07</p>
</div>

## 为什么最优控制需要「递推」这条腿

第 9 篇的变分法给出必要条件、第 10 篇的极大值原理处理约束，但它们都要解**两点边值问题**——状态从初值出发、协态从终值反推，求解常常要打靶迭代。贝尔曼 1950 年代给出的**动态规划（dynamic programming）**是完全不同的思路：**从终点倒着往前递推**，每一步只做一个「局部最优」，却保证「全局最优」。<span class="marginnote">贝尔曼的「最优性原理」一句话：<strong>整体最优的后半段，也必须是其后半段状态下的最优</strong>。这听起来像废话，却是动态规划的基石——它把「一次选一整条轨线」拆成「逐时刻选一步」，是「倒推」的思想源头。大模型里的强化学习（RL）、最优控制里的模型预测控制（MPC），都是这棵树的枝叶。</span>

这一节把两条腿都接上：**动态规划/HJB**（给出最优性条件与最优值函数）与 **LQR/LQG**（把动态规划落到线性二次型这个「能解析求解」的温床上）。卡尔曼滤波作为 LQG 的「观测器」半身，也在这一站登场——它是 20 世纪控制与信号处理领域影响最深远的成果之一。

## 1 动态规划：从终点倒着递推

先看**离散时间**版本，思想最清楚。设离散系统 $x_{k+1} = f(x_k, u_k)$，成本 $J = \sum_{k=0}^{N-1} L(x_k, u_k) + \phi(x_N)$。定义**最优值函数（cost-to-go）**：

$$
V_k(x) = \min_{u_k, \ldots, u_{N-1}}\; \sum_{i=k}^{N-1} L(x_i, u_i) + \phi(x_N), \qquad x_k = x.
$$

由最优性原理，$V_k$ 满足**递推关系**：

$$
V_k(x) = \min_{u_k}\;\Big[\, L(x, u_k) + V_{k+1}\big(f(x, u_k)\big) \Big], \qquad V_N(x) = \phi(x).
$$

**从 $V_N$ 出发，逐时刻倒推，最后一步得到 $V_0$ 与最优策略**。<span class="marginnote">注意这里的关键：递推公式里没有「未来的 $u$」只有「当下的一步 $u_k$」——<strong>未来的成本已经被 $V_{k+1}$ 打包好了</strong>。这就是「每一步只管当下、却保证全局最优」的秘密：当下的选择 + 对未来成本的最优估计。</span>

**代价是维数灾**：$V_k$ 是状态的函数，要在状态空间的每个格点上求值。连续状态空间的动态规划在「精确意义」上不可行——于是需要连续化的 HJB，以及只对「二次型」能闭式求解的 LQR。**动态规划给了思想，LQR 给了可算性。**

## 2 HJB 方程：连续时间的动态规划

把离散递推取极限（$N \to \infty$，$\Delta t \to 0$），连续时间系统 $\dot{x} = f(x,u,t)$ 的最优值函数 $V(x,t)$ 满足 **Hamilton-Jacobi-Bellman 方程（HJB）**：

$$
-\frac{\partial V}{\partial t} = \min_{u \in \Omega}\left[ L(x, u, t) + \nabla_x V \cdot f(x, u, t) \right], \qquad V(x, t_f) = \phi(x(t_f)).
$$

定义 Hamilton 函数 $H(x, u, \nabla_x V, t) = L + \nabla_x V\cdot f$，HJB 就是「**$V$ 沿最优轨线的变化 = $-L$**」，即最优值函数的下降速度等于瞬时成本。注意到 $V$ 沿时间的变化正好是协态的连续类比——**HJB 与极大值原理在此合流**：协态 $\lambda = \nabla_x V$。<span class="marginnote">HJB 是偏微分方程（PDE），一般解不出闭式；但它有两大价值：<strong>其一，给出最优性的充分条件（找到一个光滑解即全局最优）；其二，在 LQR 这种二次型问题上退化成代数方程（Riccati）</strong>。从 PDE 到代数方程，是二次型的奇迹。</span>

## 3 LQR：动态规划在二次型温床上的闭式解

**线性二次型调节器（Linear Quadratic Regulator, LQR）**：线性系统 + 二次成本

$$
\dot{x} = Ax + Bu, \qquad J = \int_0^\infty \left( x^TQx + u^TRu \right)\mathrm{d}t,
$$

$Q \succeq 0$ 惩罚状态偏差，$R \succ 0$ 惩罚控制代价。<span class="marginnote">为什么二次型？因为<strong>二次型在「平方」下与「正态噪声」配对，且求导后保持线性</strong>——最优解从「函数的函数」退回「矩阵方程」，整类问题被统一处理。$Q$、$R$ 的相对大小是设计旋钮：$Q$ 大则快而猛，$R$ 大则稳而省。</span>

猜测最优值函数是二次型 $V(x) = x^TPx$（$P \succ 0$ 待定），代入 HJB 并极小化 $u$，得到最优反馈

$$
u^* = -R^{-1}B^TPx \equiv -Kx,
$$

其中 $P$ 满足**代数 Riccati 方程（ARE）**：

$$
A^TP + PA - PBR^{-1}B^TP + Q = 0.
$$

**LQR 的三大美德**：其一，稳定反馈 $u = -Kx$ 是**状态反馈**，与第 7 篇极点配置同一套语言；其二，闭环系统自动稳定（最优值函数 $x^TPx$ 就是 Lyapunov 函数，第 8 篇的连接在此兑现）；其三，$P$ 由凸代数方程唯一正定解给出——**可计算、可保证、可调参**。<span class="marginnote">把 LQR 与极点配置对比：<strong>极点配置只「指定极点位置」，LQR 却隐式地优化了「整个闭环」</strong>——Riccati 解给出的反馈是「在 $Q$、$R$ 意义下最好的」。前者是工程处方，后者是最优决策。LQR 闭环的极点与 $Q/R$ 之间存在解析联系，这也是它常作为「自动选极点」工具的原因。</span>

## 4 LQG 与卡尔曼滤波：最优控制 + 最优估计

LQR 假设状态全部可测；现实里状态有噪声且测不全。**LQG（Linear Quadratic Gaussian）**是 LQR + 噪声的完整组合：系统

$$
\dot{x} = Ax + Bu + w, \qquad y = Cx + v,
$$

$w$、$v$ 是高斯白噪声（协方差 $W, V$）。**分离原理**保证：LQG 最优控制器 = 最优状态反馈（LQR 的 $K$）+ 最优状态估计（**卡尔曼滤波器**）的拼接，二者可以独立设计。

**卡尔曼滤波器（Kalman filter, 1960）**是最小方差意义下的最优状态估计器：

$$
\dot{\hat{x}} = A\hat{x} + Bu + L_k(y - C\hat{x}), \qquad
L_k = \Sigma C^T V^{-1},
$$

其中估计误差协方差 $\Sigma$ 满足**滤波 Riccati 方程**：

$$
A\Sigma + \Sigma A^T - \Sigma C^TV^{-1}C\Sigma + W = 0.
$$

**卡尔曼滤波器与 Luenberger 观测器结构相同，但增益 $L_k$ 由噪声统计决定**——它不只是「够用」，而是「统计最优」。<span class="marginnote">卡尔曼滤波的伟大在于把「估计问题」也变成了「Riccati 问题」：<strong>控制与估计在数学上对偶</strong>（$A \leftrightarrow A^T$，$B \leftrightarrow C^T$，$Q \leftrightarrow W$，$R \leftrightarrow V$）。1960 年卡尔曼在同一年给出能控能观判据、滤波理论、与 LQR 的雏形——现代控制理论的「创世纪」。</span>

LQG 的工程化身无处不在：GPS 惯性导航（Apollo 登月导航、民航惯导）、股票定价、机器人状态估计、直到今天大模型里的各种「状态估计 + 策略优化」。**凡是「传感器噪声 + 动态模型」的场景，卡尔曼滤波都是默认的第一选择。**

## 5 公式解析：从 HJB 到代数 Riccati 方程

把「HJB 是 PDE、LQR 却得到代数方程」这条神奇推导走一遍：

$$
0 = x^T\big(A^TP + PA - PBR^{-1}B^TP + Q\big)x.
$$

- **第一步，猜二次型值函数**：$V(x) = x^TPx$（$P \succ 0$）。对稳态 LQR，$\partial V/\partial t = 0$，HJB 退化为

$$
0 = \min_u \left[ x^TQx + u^TRu + 2x^TP(Ax + Bu) \right].
$$

- **第二步，对 $u$ 求极小**：$\nabla_u[\cdots] = 2Ru + 2B^TPx = 0$，得 $u^* = -R^{-1}B^TPx$。这一步把「对函数的优化」变成「解线性方程」——二次型的威力在此爆发。
- **第三步，代回 HJB**：把 $u^*$ 代回，整理得 $x^T(A^TP + PA - PBR^{-1}B^TP + Q)x = 0$ 对一切 $x$ 成立。
- **第四步，读结论**：括号内必须为零，即 ARE。**从 PDE 到代数方程，靠的是「二次型猜解 + 对 $u$ 显式求极小」两步**——二次型结构让「猜解」成为「必然解」，这就是 LQR 能闭式求解的深层原因。

## 6 小结

- **最优性原理**：整体最优的后半段也是局部最优；动态规划从终点倒推，逐步求最优值函数。
- **HJB 方程**：连续时间动态规划的偏微分方程，是充分条件；协态与 $\nabla_x V$ 在此合流。
- **LQR**：二次成本下最优值函数是二次型，最优反馈 $u = -Kx$，$K = R^{-1}B^TP$，$P$ 由代数 Riccati 方程确定。
- **分离原理 + LQG**：最优控制（LQR）与最优估计（卡尔曼滤波）可独立设计后拼接。
- **卡尔曼滤波**：最小方差最优估计器，增益由滤波 Riccati 方程与噪声统计决定；与控制问题数学对偶。
- LQR/LQG 的全链路「可算、可保证、可调参」，是动态规划思想在工程上的最佳落点；也是后续 MPC、H∞ 的参照系。

在下一节，LQR/LQG 已经达到「给定模型即最优」，但模型本身有误差怎么办？面对不确定性，控制理论还有最后一层武装——**现代控制前沿（鲁棒控制 H∞、模型预测控制）**。
