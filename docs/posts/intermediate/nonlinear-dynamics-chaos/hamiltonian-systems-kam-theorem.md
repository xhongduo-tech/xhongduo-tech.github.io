---
title: Hamilton 系统与 KAM 定理
date: 2026-08-11
---

# Hamilton 系统与 KAM 定理

<div class="epigraph">
<p>我的方法本质上是工作与思考的方法；正因如此，它们才匿名地四处蔓延。</p>
<footer>—— 埃米 · 诺特（Emmy Noether）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 非线性动力学与混沌 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Hamilton 系统开始

前面七篇处理的系统大多是**耗散**的——能量被摩擦、被阻尼、被扩散吃掉，相空间体积收缩，一切归于吸引子。但自然界还有另一大类系统：**保守系统**。钟摆若无摩擦、行星绕日、弹簧振子，它们不消耗能量，也不「忘记初值」。这一类的标准数学框架，就是 **Hamilton 系统（Hamiltonian system）**。<span class="marginnote">Hamilton 力学是牛顿力学的优雅改写，也是通往量子力学与统计物理的桥梁：《第一级 · 量子力学》用 Hamilton 量写薛定谔方程，《统计物理》用 Hamilton 量写配分函数。而在动力学这里，Hamilton 系统的地位是「保守世界的宪法」——它规定了没有耗散时，相空间里究竟能发生什么。</span>

Hamilton 系统看似与混沌水火不容（能量守恒，怎么还会敏感依赖？），但事实恰恰相反：**保守系统同样可以混沌，只是混沌的形式不同**——没有吸引子，而是「混沌海」（chaotic sea）里漂着几座秩序的孤岛。而这一切的定量叙述，就是本专题的收官定理——**KAM 定理**。<span class="marginnote">从「从极限到大模型」的主线看，这一篇是「数学 → 物理 → 机器学习」的收官枢纽：Hamilton 几何支撑着辛流形（现代 AI 里 Riemann 流优化、Hamiltonian 蒙特卡洛、Nesterov 加速都可以在辛结构下理解），KAM 定理则是「微扰论何时可靠」的终极回答。</span>

## 1 Hamilton 方程：用能量重写动力学

设系统有 $n$ 个**广义坐标** $q_i$ 与 $n$ 个**广义动量** $p_i$（$i=1,\dots,n$），能量函数 $H(q,p)$ 称为 **Hamilton 量**。Hamilton 方程是 $2n$ 个一阶方程：

$$\dot{q}_i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}, \qquad i = 1, \dots, n.$$

这个结构之美在于：**知道能量 $H$，就知道了全部动力学**——$H$ 对 $p$ 的导数给位置的变化，$H$ 对 $q$ 的导数（带负号）给动量的变化。<span class="marginnote">$q$ 与 $p$ 合在一起构成 $2n$ 维<strong>相空间</strong>。$n=1$ 的单摆、$n=2$ 的平面行星运动，都在二维、四维相空间里运行。注意这里「相空间维数」是连续的 $2n$，与《分形几何与奇怪吸引子》里分数维的「维数」是两回事——那是测度维数，这里是自由度维数。</span>

从 Hamilton 方程立刻得到两个守恒律：

$$\frac{\mathrm{d}H}{\mathrm{d}t} = \sum_i \left( \frac{\partial H}{\partial q_i}\dot{q}_i + \frac{\partial H}{\partial p_i}\dot{p}_i \right) = \sum_i \left( \frac{\partial H}{\partial q_i}\frac{\partial H}{\partial p_i} - \frac{\partial H}{\partial p_i}\frac{\partial H}{\partial q_i} \right) = 0,$$

即 **Hamilton 量守恒（能量守恒）**；由诺特定理（Noether's theorem），每一个连续对称性对应一个守恒量——时间平移对称性对应能量守恒。<span class="marginnote">诺特定理是本专题与《第一级 · 理论力学》以及数学里对称性理论的最重要接点：能量 = 时间平移的守恒量，角动量 = 空间转动的守恒量，动量 = 空间平移的守恒量。「守恒」不是偶然，而是「对称性」的必然结果——这条线索贯穿现代物理。</span>

## 2 相空间体积守恒：与耗散的分道扬镳

**Liouville 定理（Liouville's theorem）**：Hamilton 流的相空间**体积守恒**——相空间任意区域随时间演化时体积不变，只是形状被扭曲。

证明只需一步：Hamilton 流的散度恒为零，

$$\nabla \cdot \mathbf{f} = \sum_i \left( \frac{\partial \dot{q}_i}{\partial q_i} + \frac{\partial \dot{p}_i}{\partial p_i} \right) = \sum_i \left( \frac{\partial^2 H}{\partial q_i \partial p_i} - \frac{\partial^2 H}{\partial p_i \partial q_i} \right) = 0,$$

而散度为零（不可压缩流）意味着体积保持。<span class="marginnote">这与 Lorenz 系统的散度 $-(\sigma+1+\beta)<0$ 形成鲜明对比：耗散系统体积收缩、有吸引子；Hamilton 系统体积守恒、没有吸引子。<strong>「有没有吸引子」是保守与耗散世界的分水岭</strong>——保守混沌永远不可能把初始条件「俘获」到一个低维集合上。</span>

**辨析｜易错点：** 体积守恒 ≠ 形状不变。Liouville 说的是「体」不变，「形」可以拉得很长很细——两个邻近轨道照样可以指数分离（Lyapunov 指数为正），只是它们分离的「方向」必有另一方向以同样速率靠近，以便总体积守恒。所以**保守混沌与耗散混沌的判据不同**：保守系统看「$\lambda$ 关于 $0$ 对称成对出现」，耗散系统看「$\sum \lambda < 0$」。

## 3 可积系统与不变环面：秩序的原型

在保守系统中，规则运动的原型是**可积系统（integrable system）**。一个 $n$ 自由度的 Hamilton 系统若存在 $n$ 个**相互对合**的独立守恒量，称为可积的；由 **Liouville–Arnold 定理**，可积系统的有界相空间由 $n$ 维的**不变环面（invariant tori）** 层层填满，运动在这些环面上是**准周期**的（多个不可公度频率的叠加），由**作用–角变量**（action–angle variables）$(I, \theta)$ 描述：

$$\dot{I} = 0, \qquad \dot{\theta} = \omega(I).$$

在作用–角变量下，方程被「解耦」成直线运动——这就是可积的意义：**换一组坐标，动力学变得平凡**。<span class="marginnote">典型例子：自由单摆、开普勒行星运动、谐振子。混沌的正反对立面正是「可积」：可积意味着无穷多守恒量，混沌意味着几乎只有能量一个守恒量。太阳系的「可积近况」——行星轨道几乎不变——正是 KAM 定理要解释的奇迹。</span>

作用量 $I$ 在经典力学里也是量子化的对象：玻尔–索末菲量子化条件 $\oint p\,\mathrm{d}q = nh$ 量化的是作用量。这里再次印证：**Hamilton 力学是量子力学的经典骨架**。

## 4 微扰破坏与 KAM 定理：环面的存亡

现实系统几乎总带微扰：土星会扰动木星、摩擦会轻微地破坏理想守恒。经典微扰论的问题是**小除数（small divisor）**：当微扰频率与固有频率谐振时，分母趋于零，级数发散——预言「环面被摧毁」；但太阳系明明稳定了数十亿年，理论与观测矛盾。

**KAM 定理（Kolmogorov–Arnold–Moser theorem）** 解决了这个矛盾。其核心结论（定性版）：**对「足够好」的可积 Hamilton 系统施加足够小的光滑微扰，大多数（在测度意义上）不变环面并不会消失，只是被轻微形变**——条件是其频率向量 $\omega$ 满足「够无理」的 Diophantine 条件

$$|\omega \cdot k| \ge \frac{\gamma}{|k|^{\tau}}, \qquad \forall\, k \in \mathbb{Z}^n \setminus \{0\},$$

即频率与任何整数向量都「保持距离」；反之，频率「够有理」（共振）的环面才会破碎成混沌海。<span class="marginnote">KAM 定理是 1954 年 Kolmogorov 提出、1960 年代 Arnold 与 Moser 完善证明的。它标志着「微扰论」从形式技巧升格为严格定理，是 20 世纪分析力学最深刻的成就之一。Diophantine 条件把「无理程度」数学化：无理数「够无理」（如黄金比，其连分数近似被 $1/|k|^2$ 控制）对应的环面能幸存。</span>

微扰后的相图成为：**大部分环面形变幸存，少数共振环面破碎成细密的「分形薄带」，薄带之间渗出混沌海**——秩序与混沌以 Cantor 集式的精细结构共存。这就是**保守混沌**的图景：没有吸引子，只有「混沌海 + 岛上的环面」。

## 5 公式解析：Hamilton 方程 $\dot{p} = -\partial H/\partial q$

以单摆为例，把这条方程讲透。单摆（质量 $m$、摆长 $l$、重力加速度 $g$）的 Hamilton 量

$$H(q, p) = \frac{p^2}{2ml^2} - mgl\cos q,$$

其中 $q$ 是摆角、$p = ml^2\dot{q}$ 是角动量。三步拆解：

- **第一步，动能的 Hamilton 形式**：动能 $\frac{1}{2}ml^2\dot{q}^2$ 用 $p$ 代写为 $p^2/(2ml^2)$。Hamilton 方程的第一条 $\dot{q} = \partial H/\partial p = p/(ml^2)$ 正是角速度的定义，自洽闭合。
- **第二步，第二条方程 $\dot{p} = -\partial H/\partial q = -mgl\sin q$**：这正是牛顿方程 $ml^2\ddot{q} = -mgl\sin q$ 的动量形式。**Hamilton 方程把一条二阶方程拆成两条一阶方程**——这是所有「向量场化」的标准动作，与《二维线性系统与相平面》里把 $\ddot{x}$ 写成 $\dot{x}=y, \dot{y}=\dots$ 的做法完全一致。
- **第三步，守恒律回检**：$H$ 不显含 $q$（旋转对称）的场合 $\partial H/\partial q = 0$，则 $\dot{p} = 0$——动量守恒。这是诺特定理最朴素的体现：**对称性直接读出守恒量，Hamilton 形式让「看见守恒」变成纯粹的求导**。

小角度 $\sin q \approx q$ 时方程退化和谐振子（可积），大摆角进入非线性；若加上周期驱动的微扰（参数共振），单摆相空间就出现 KAM 环面破碎与混沌海——从最优雅的模型里直接孵化出混沌。

## 6 小结

- **Hamilton 系统** $\dot{q}=\partial_p H,\ \dot{p}=-\partial_q H$ 是保守系统的宪法；能量守恒，且有诺特定理给出「对称性 ⇒ 守恒量」。
- **Liouville 定理**：Hamilton 流相空间体积守恒（散度为零）——保守系统**没有吸引子**，与耗散系统分道扬镳。
- **可积系统**：$n$ 个守恒量 ⇒ 相空间由**不变环面**填满，运动准周期；作用–角变量让动力学平凡化。
- **KAM 定理**：足够小的微扰只毁掉「够有理」的共振环面，**大多数「够无理」的环面形变幸存**——守恒混沌以「混沌海 + 环面孤岛」的形式出现。
- 保守混沌没有吸引子，Lyapunov 指数成对反号；它是混沌理论在「非耗散」舞台上的最终收束。

至此，我们从一维流出发，走过不动点、分岔、相平面、极限环、混沌吸引子、倍周期级联与分形，最后在 Hamilton 系统与 KAM 定理处合拢。**「非线性动力学与混沌」全专题完结**——下一站，你可以带着这套语言进入《分形几何》的严格构造、进入《统计物理》的遍历理论，或回到《第一级 · 数值计算》去思考：混沌系统的数值模拟，究竟什么才是可靠的？
