---
title: 欧拉角、等效轴角与四元数：姿态的紧凑表示
date: 2026-08-07
---

# 欧拉角、等效轴角与四元数：姿态的紧凑表示

<div class="epigraph">
<p>i² = j² = k² = ijk = −1</p>
<footer>—— 威廉 · 哈密顿（William Rowan Hamilton），
1843 年 10 月 16 日刻于都柏林布鲁姆桥</footer>
</div>

<div class="article-byline">
<p>第四级 · 具身智能 ｜ Craig《机器人学导论》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从紧凑表示开始

旋转矩阵有 9 个数，
却只有 3 个自由度——它太「胖」了。
胖带来的麻烦有三重：一是**冗余**，
存 9 个数却只用得上 3 个；
二是**难插值**，
想在两个姿态之间平滑过渡，
对 9 个分量逐个做线性插值，
得到的中间矩阵很可能不再是旋转矩阵（正交性和行列式都会跑掉）；
三是**难优化**，
在机器人强化学习、姿态估计里把 $R$ 当优化变量，
9 个分量要受 6 条约束拉扯，
收敛又慢又脆弱。

具身智能的日常充满了这三种场景。
无人机、四足机器人、机械臂的控制器都要在几十到上千赫兹的循环里读写姿态；
视觉里程计每帧都要比较两个相机姿态；
模仿学习与 VLA 模型输出的动作里，
末端的朝向也常以紧凑形式编码。
<span class="marginnote">RT-1 等机器人 Transformer 把末端的旋转动作编码成 8 维离散化（6 个桶 + 1 维连续 + 1 位标志）；
π0 里则用单位四元数直接作为动作维度。
姿态怎么表示，
直接决定了动作空间的设计。
</span>于是机器人学界换用三种更「瘦」的表示：**欧拉角（Euler angles）**、**等效轴角（axis-angle）** 与 **四元数（quaternion）**。

这一节回答一个问题：同一个姿态，
用哪几个数、怎么记、怎么算，
才不会出错？

## 1 欧拉角：三次基本旋转的复合

欧拉角的思想非常朴素：**任意姿态都可以分解为绕坐标轴的、有顺序的三次基本旋转**。
姿态的 3 个自由度，
正好对应 3 个角。

最常见的有两组约定。

- **ZYZ 约定**（Craig 教材采用）：先绕 $Z$ 转 $\phi$，
再绕新的 $Y$ 转 $\theta$，
最后绕新的 $Z$ 转 $\psi$。
注意这里每次都是**相对当前坐标系**旋转，
所以按上一节的规则要**右乘**：

$$
R = R_z(\phi)\; R_y(\theta)\; R_z(\psi)
$$

- **ZYX 约定**（航空航天、ROS 常用）：三个角叫 **yaw（偏航）／pitch（俯仰）／roll（横滚）**，
先绕 $Z$ 转 yaw、再绕新 $Y$ 转 pitch、最后绕新 $X$ 转 roll：

$$
R = R_z(\text{yaw})\; R_y(\text{pitch})\; R_x(\text{roll})
$$

**重点：欧拉角永远要带上「约定」两个字。** 三个角是多少，
取决于三个问题——绕哪三根轴、旋转的先后顺序、旋转相对固定系还是当前系。
只说「俯仰 30°、横滚 20°」而不说顺序，
姿态就是未定义的。
这是欧拉角使用中最容易踩的坑。

**辨析｜易错点：** 很多工程事故源于约定不统一。
同一个角度三元组，
按 ZYZ 与按 ZYX 算出的矩阵完全不同；
ROS 与 MATLAB 的 roll-pitch-yaw 甚至对应不同的旋转顺序实现。**读代码先看约定，写代码先写注释注明约定。**

## 2 公式解析：从旋转矩阵反解欧拉角，以及万向锁

有了正问题 $R = R_z(\phi)R_y(\theta)R_z(\psi)$，
自然要问反问题：**给定 $R$，怎么反解出 $\phi, \theta, \psi$？** 把乘积写开。

**第一步，展开矩阵。** 记 $c\theta = \cos\theta$、$s\theta = \sin\theta$，
逐项相乘：

$$
R =
\begin{bmatrix}
c\phi c\theta c\psi - s\phi s\psi & -c\phi c\theta s\psi - s\phi c\psi & c\phi s\theta \\
s\phi c\theta c\psi + c\phi s\psi & -s\phi c\theta s\psi + c\phi c\psi & s\phi s\theta \\
-s\theta c\psi & s\theta s\psi & c\theta
\end{bmatrix}
$$

**第二步，挑出关键元素。** 观察第 3 行第 3 列是 $c\theta$，
第 1 行第 3 列是 $c\phi s\theta$，
第 2 行第 3 列是 $s\phi s\theta$，
第 3 行第 1、2 列分别是 $-s\theta c\psi$ 与 $s\theta s\psi$。
于是三个角可以从这些元素反解：

$$
\theta = \operatorname{atan2}\!\Big(\sqrt{r_{13}^2 + r_{23}^2},\; r_{33}\Big)
$$

$$
\phi = \operatorname{atan2}(r_{23}, r_{13}), \qquad
\psi = \operatorname{atan2}(r_{32}, -r_{31})
$$

（取 $\sin\theta > 0$ 的分支；
另一支对应 $\theta$ 取补角。
）

**第三步，观察退化情形。** 当 $\theta = 0$ 时，
$\sin\theta = 0$，
于是 $r_{13} = r_{23} = 0$、$r_{31} = r_{32} = 0$。
这时 $\phi$ 与 $\psi$ 的公式全部变成 $\operatorname{atan2}(0,0)$——**无定义**。
回头看你发现，
此时

$$
R = R_z(\phi)\; I\; R_z(\psi) = R_z(\phi + \psi)
$$

**两个外部旋转合并成了一个，三个角只剩两个自由度。**

**辨析｜易错点：** 这就是著名的**万向锁（gimbal lock）**。
它不是矩阵或数学的缺陷，
而是**欧拉角这个参数化的固有奇点**：在 $\theta = 0$（ZYZ）或 pitch $= \pm 90°$（ZYX）附近，
姿态对 $\phi$ 和 $\psi$ 的变化几乎不再敏感，
控制器会突然「分不清」该转哪根轴。
历史上阿波罗登月舱的惯性导航单元、早期的机械臂姿态控制器都吃过万向锁的亏——**解决方案不是修补欧拉角，而是换一种没有奇点的表示**。
这就是等效轴角与四元数登场的原因。
<span class="marginnote">工程上有两招应对万向锁：一是在奇点附近切换到另一组约定（复杂、容易引入抖动）；
二是干脆不用欧拉角做内部表示，
只在与人交互时把它当作「读数的外壳」。
今天的航姿参考系统、SLAM 后端、VR 手柄，
内部几乎全是四元数。
</span>

## 3 等效轴角与罗德里格斯公式

欧拉定理（1775）说：**任何三维旋转都可以表示为绕某一固定轴的单一旋转**。
于是姿态可以用「一根轴 + 一个角」来记——4 个参数（轴 3 个分量 + 角 1 个），
比 9 个少，
但仍然多一个约束（轴是单位向量）。

设旋转轴为单位向量 $\hat{k}$，
转角为 $\theta$，
记 $[\hat{k}]_\times$ 为 $\hat{k}$ 的**叉乘矩阵（skew-symmetric matrix）**：

$$
[\hat{k}]_\times =
\begin{bmatrix}
0 & -k_z & k_y \\
k_z & 0 & -k_x \\
-k_y & k_x & 0
\end{bmatrix},
\qquad
[\hat{k}]_\times v = \hat{k} \times v
$$

**罗德里格斯公式（Rodrigues' formula）** 给出旋转矩阵：

$$
R(\hat{k}, \theta) = I + \sin\theta\, [\hat{k}]_\times + (1 - \cos\theta)\, [\hat{k}]_\times^2
$$

**公式解析：罗德里格斯公式为什么成立。** 拆成三步理解：

- **第一步，把向量分解。** 任意向量 $v$ 沿轴 $\hat{k}$ 的分量为 $v_\parallel = \hat{k}(\hat{k} \cdot v)$，
垂直分量为 $v_\perp = v - v_\parallel$。
绕 $\hat{k}$ 旋转时 $v_\parallel$ 不动，
只有 $v_\perp$ 在垂直于轴的平面里转一个 $\theta$。
- **第二步，识别叉乘的几何作用。** 叉乘矩阵 $[\hat{k}]_\times v_\perp = \hat{k} \times v_\perp$：它给出 $v_\perp$ 在平面里转 $90°$ 的方向。
于是旋转后的垂直分量是 $\cos\theta\, v_\perp + \sin\theta\, (\hat{k} \times v_\perp)$。
- **第三步，整理成矩阵形式。** 把所有项合起来，
$v_\perp = v - \hat{k}(\hat{k}\cdot v)$ 且 $[\hat{k}]_\times^2 v = \hat{k}(\hat{k}\cdot v) - v = -v_\perp$（即 $[\hat{k}]_\times^2$ 是「把垂直分量取反」的算子），
代回并提取公共的 $v$，
就得到上面的公式。
<span class="marginnote">叉乘矩阵平方的几何意义值得记住：$[\hat{k}]_\times^2$ 把向量投影到与轴垂直的平面并取反，
$[\hat{k}]_\times$ 把向量在垂直平面内转 90°。
一个旋转矩阵，
就是 $I$、$[\hat{k}]_\times$、$[\hat{k}]_\times^2$ 三者的线性组合——下一节讲李代数时，
这个组合会以「矩阵指数」的面貌再次出现。
</span>

验证两个边界：$\theta = 0$ 时 $R = I$；
$\theta = \pi$ 时 $R = I + 2[\hat{k}]_\times^2$，
绕任意轴转半圈的结果是对称矩阵——符合「转 180° 后沿轴的分量不变、垂直分量取反」。

等效轴角仍有一个小毛病：$\hat{k}$ 与 $-\hat{k}$、$\theta$ 与 $2\pi-\theta$ 表示同一个旋转，
且 $\theta = 0$ 时轴无定义（任何轴都不转）。
它没有万向锁，
但在数值与插值上仍不理想。

## 4 四元数：哈密顿的礼物

四元数把姿态用 **4 个参数**表示，
其中 3 个是「虚」的，
1 个是实的：

$$
q = w + xi + yj + zk = (w, x, y, z), \qquad i^2 = j^2 = k^2 = ijk = -1
$$

三个虚单位 $i, j, k$ 不满足交换律，
但满足 $ij = k$、$jk = i$、$ki = j$。**绕单位轴 $\hat{k}$ 转 $\theta$ 的四元数是**

$$
q = \Big(\cos\frac{\theta}{2},\ \ \hat{k}\sin\frac{\theta}{2}\Big)
$$

注意角度出现的是 $\theta/2$——这是四元数表示旋转时最容易忽略的系数。
旋转矩阵与四元数的换算（$\|q\| = 1$）：

$$
R =
\begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}
$$

**重点：$q$ 与 $-q$ 表示同一个旋转。** 因为 $q \to -q$ 相当于 $(\cos\frac{\theta}{2},\hat{k}\sin\frac{\theta}{2}) \to (-\cos\frac{\theta}{2}, -\hat{k}\sin\frac{\theta}{2})$，
代入上面的矩阵后每一项都乘以两个负号，
$R$ 不变。
四元数对旋转矩阵是**二对一**的映射——这称为**双覆盖（double cover）**。

**辨析｜易错点：** 四元数不是「4 个任意的数」。
表示旋转必须满足**单位约束** $\|q\| = w^2 + x^2 + y^2 + z^2 = 1$。
任何把四元数当 4 维向量处理的算法（插值、优化、加噪声）都可能把长度推出 1，
得到非法的旋转——所以每步之后都要归一化。
这也是强化学习、滤波里把四元数放进网络前必须谨慎的原因。
<span class="marginnote">单位约束让四元数生活在三维单位球面 $S^3$ 上，
这个球面处处光滑、没有奇点，
正是它取代欧拉角的原因。
而「$q$ 与 $-q$ 相同」意味着球面上对径点要粘合，
这个「一半的球面」记作 $S^3/\{\pm 1\}$，
恰好与 $SO(3)$ 同胚——群论与几何在这里交汇。
</span>

## 5 公式解析：四元数乘法与旋转合成

四元数最大的优点是可以**像矩阵一样相乘来合成旋转**。
设旋转 $q_1$ 后再旋转 $q_2$，
等效于：

$$
q = q_2 \otimes q_1
$$

（注意顺序：先作用在右边的 $q_1$，
再作用左边的 $q_2$，
与矩阵左乘的顺序一致。
）

**四元数乘法（Grassmann 积）** 的定义，
把 $q_a = (w_a, \mathbf{v}_a)$ 拆成标量部分 $w$ 与矢量部分 $\mathbf{v} = (x,y,z)$：

$$
q_a \otimes q_b = (w_a w_b - \mathbf{v}_a \cdot \mathbf{v}_b,\ \ w_a \mathbf{v}_b + w_b \mathbf{v}_a + \mathbf{v}_a \times \mathbf{v}_b)
$$

逐步拆解这个公式：

**第一步，标量部分 $w_a w_b - \mathbf{v}_a \cdot \mathbf{v}_b$。** 两个数相乘，
实部来自「实数乘实数」减去「矢量点积」。
矢量点积带负号，
正是 $i^2 = -1$ 在代数里的体现——三个虚单位的平方都产生 $-1$，
所以虚部两两相乘贡献到实部时要取负。

**第二步，矢量部分 $w_a \mathbf{v}_b + w_b \mathbf{v}_a + \mathbf{v}_a \times \mathbf{v}_b$。** 第一项是 $q_a$ 的实部乘 $q_b$ 的虚部，
第二项反之；
第三项是叉积——它来自 $ij = k$ 这类非交换乘积。
叉积方向依赖顺序，
所以四元数乘法**不可交换**，
与旋转合成的物理事实一致。

**第三步，验证旋转合成与半角。** 把两个旋转 $q_1$、$q_2$ 代入上式，
可以证明合成后总角度满足半角公式：

$$
\cos\frac{\theta}{2} = \cos\frac{\theta_1}{2}\cos\frac{\theta_2}{2} - \sin\frac{\theta_1}{2}\sin\frac{\theta_2}{2}\cos\alpha
$$

其中 $\alpha$ 是两根转轴的夹角。
这正是球面三角里的余弦定理——**四元数乘法把「旋转的合成」翻译成了「球面上的几何」**，
这也是它被广泛用于姿态插值（slerp）的原因。
<span class="marginnote">四元数插值 slerp（spherical linear interpolation）沿 $S^3$ 上的大圆弧平滑过渡，
得到的是测地线——两点之间旋转的「最短路径」；
而对矩阵或欧拉角的线性插值会走出乱七八糟的中间姿态。
VLA 模型、动画引擎、轨迹平滑里到处是它。
</span>

## 6 用代码把四元数算一遍

SciPy 提供现成的换算，
这里手写乘法并验证「$q$ 与 $-q$ 同旋转」。

```python
import numpy as np

def quat_mul(q, r):
    """四元数乘法 q ⊗ r（q, r = (w, x, y, z)）"""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_to_R(q):
    """四元数 → 旋转矩阵（w, x, y, z）"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])

# 绕 z 轴转 0.6 rad、绕 x 轴转 0.8 rad 的两个单位四元数
qz = np.array([np.cos(0.3), 0, 0, np.sin(0.3)])
qx = np.array([np.cos(0.4), np.sin(0.4), 0, 0])

print(np.allclose(quat_to_R(quat_mul(qz, qx)),
                  quat_to_R(qz) @ quat_to_R(qx)))   # 乘法同构于矩阵乘法
print(np.allclose(quat_to_R(qz), quat_to_R(-qz)))   # True：q 与 -q 同旋转
print(np.allclose(quat_mul(qz, qx), quat_mul(qx, qz)))  # False：不可交换
```

代码印证：四元数乘法顺序不可交换（$q_z \otimes q_x \neq q_x \otimes q_z$）、$q$ 与 $-q$ 对应同一旋转矩阵、单位约束是合法性的生命线。

## 7 小结

- **欧拉角**：三次基本旋转的复合，
最少 3 个参数；
必须注明**轴序与固定系/当前系约定**；
存在**万向锁**奇点（ZYZ 中 $\theta=0$、ZYX 中 pitch $=\pm 90°$）。
- **等效轴角**：欧拉定理保证「任一旋转 = 绕某轴转一角」；**罗德里格斯公式** $R = I + \sin\theta[\hat{k}]_\times + (1-\cos\theta)[\hat{k}]_\times^2$ 给出矩阵。
- **四元数**：$q = (\cos\frac{\theta}{2},\ \hat{k}\sin\frac{\theta}{2})$，
4 参数、无奇点；**$q$ 与 $-q$ 表示同一旋转（双覆盖）**；
必须保持单位约束。
- **合成**：四元数乘法 $q_2 \otimes q_1$（顺序敏感）等价于矩阵乘法，
适合插值（slerp）与连续优化。
- **选型**：与人交互读欧拉角，
内部计算与插值用四元数，
求导与李群运算用旋转矩阵。

在下一节，
我们将从「四元数生活在球面 $S^3$ 上」这个观察出发，
把姿态表示升级成李群 $SO(3)$ / $SE(3)$ 与李代数的语言：旋转矩阵在时间里的速度——反对称矩阵——如何通过指数映射回到旋转矩阵，
以及机器人学的「旋量」如何把旋转与平移统一成一条螺旋运动。



