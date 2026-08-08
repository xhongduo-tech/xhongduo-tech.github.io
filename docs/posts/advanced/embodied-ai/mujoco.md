---
title: MuJoCo：接触动力学建模与 Python API 实践
date: 2026-08-07
---

# MuJoCo：接触动力学建模与 Python API 实践

<div class="epigraph">
<p>工欲善其事，
必先利其器。
</p>
<footer>—— 《论语·卫灵公》</footer>
</div>

<div class="article-byline">
<p>第四级 · 具身智能 ｜ Todorov, Erez & Tassa, MuJoCo（2012）；
MuJoCo 官方文档 ｜ 2026-08-07
</p></div>

## 为什么从 MuJoCo 开始

机器人学习（RL、模仿、Sim2Real）的命脉是**仿真器**——而在接触丰富的操作/行走仿真里，**MuJoCo** 是事实标准：DeepMind 在 2022 年将其开源，
ANYmal、OpenAI 魔方手、以及大量操作/运动学习都用它。
它的核心优势是**接触动力学的稳定与快速**——用凸优化处理接触，
不「穿透」、不「爆炸」<span class="marginnote">MuJoCo 的全名是 Multi-Joint dynamics with Contact（多关节动力学与接触）。
作者 Todorov 的学术背景是最优控制，
他把「接触」建模成一个<strong>凸优化问题</strong>——这让接触求解稳定、可微、快，
成为接触丰富任务仿真的首选。
</span>。

## 1 为什么接触仿真难

接触动力学是仿真器最容易翻车的地方：

**接触是「软硬切换」**：不接触时自由运动，
接触时突然被约束——**不连续**；
**摩擦锥非线性**：切向摩擦力受法向力约束（第 16 章）——**非线性约束**；
**穿透问题**：数值积分会把物体「陷进」地面，
要么爆炸要么粘住。

朴素仿真的接触又慢又不稳——**一个接触点处理不好，整个仿真就「炸」**。
MuJoCo 用「凸优化」一举解决稳定与速度。

## 2 MuJoCo 的接触模型：凸优化

MuJoCo 把「这一时刻世界该怎么动」写成一个**凸二次优化**：

1. 假设所有接触点处有一个「软接触（soft contact）」——允许微小穿透（不是硬约束），
穿透产生弹性力；
2. 每个接触点的力必须满足**摩擦锥**；
3. 求解「满足动力学 + 接触约束 + 摩擦锥」的**最小二乘/凸优化问题**——找到「最合理的加速度与接触力」。

**结果**：接触求解是一个**凸 QP**——有全局解、不振荡、速度快。
加上**多线程与专门优化**，
MuJoCo 能在几毫秒内跑完一个接触丰富的仿真步。
<span class="marginnote">「软接触」是关键设计：<strong>硬接触（不允许穿透）需要迭代求解且容易抖动；
软接触允许微小穿透、用弹性力回推，
求解变成凸优化，
稳且快</strong>。
这牺牲了「接触的绝对精确」，
换来了「大规模 RL 训练需要的稳定与速度」——对学习任务，
后者远比前者重要。
</span>

## 3 MJCF：用 XML 描述世界

MuJoCo 用 **MJCF（XML）** 描述模型，
核心元素：

- `<body>`：刚体（位置、质量、惯量、几何）；
- `<joint>`：几何体（盒、球、胶囊、网格）——用于碰撞与渲染；
- `<actuator>`：几何体（盒、球、胶囊、网格）——用于碰撞与渲染；
- `<actuator>`：执行器（电机、力矩、位置）——输出控制；
- `<sensor>`：传感器（关节角、速度、接触力）——读取观测。

**MJCF 与 URDF 的区别**：MJCF 是 MuJoCo 原生（面向仿真，
简洁、稳定）；
URDF 是 ROS 标准（面向机器人描述，
要转换）。**MuJoCo 支持 URDF 导入**，
但「原生 MJCF 更顺手」是共识。
一个四足/机械臂模型通常几行到几十行 XML。

## 4 Python API 实践

MuJoCo 的 Python 接口（`mujoco` 包，
v2.2+）很简单：

```python
import mujoco
import numpy as np

xml = """
<mujoco model="pendulum">
  `<worldbody>`
    `<body>`
      <joint name="hinge1" type="hinge" axis="0 0 1"/>
      <geom name="link" type="capsule" size="0.05 0.2"/>
    </body>
  </worldbody>
  `<actuator>`
    <motor name="m1" joint="hinge1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)   # 静态描述（结构、参数）
data  = mujoco.MjData(model)                  # 动态状态（每步可变）

data.ctrl[0] = 1.0                            # 电机出力
mujoco.mj_step(model, data)                   # 推进一个仿真步
print(data.qpos, data.qvel)                   # 关节角与角速度
```

**关键概念**：

- **MjModel**：模型的「静态描述」（结构、参数，
只读）；
- **MjData**：仿真的「动态状态」（每步可变）；
- **mj_step**：推进一个仿真步（**控制频率**决定步长，
通常 500 Hz 仿真 / 20–50 Hz 控制）；
- **渲染**：`mujoco.Renderer` 离屏渲染图像（供视觉 RL / 模仿学习用）。

**与 Gymnasium 集成**：`gymnasium.envs.mujoco` 的 MuJoCo 环境（HalfCheetah、Ant 等经典环境）与 **MuJoCo Playground**（官方模型库：Franka、Unitree 等）——**开箱即用的机器人环境**。

## 5 公式解析：接触动力学的凸优化

$$
\min_{\ddot q,\;\lambda}\; \big\| M\ddot q + C\dot q + g - \tau - J^T\lambda \big\|^2 + \epsilon\|\lambda\|^2, \qquad \text{s.t.}\; \lambda \in \text{摩擦锥}
$$

- **第一步，看动力学平衡**：$M\ddot q + C\dot q + g = \tau + J^T\lambda$——接触力 $\lambda$ 通过雅可比 $J^T$ 进动力学。
MuJoCo 求解的是「找一个 $\ddot q$ 和 $\lambda$ 让这个方程尽可能成立」。
- **第二步，看约束**：$\lambda$ 必须在**摩擦锥**内（法向非负 + 切向受摩擦限制）——**接触力不能拉地、不能超摩擦极限**。
- **第三步，看凸性**：目标是二次的（最小二乘）、约束是凸锥——**整个问题是凸 QP，有唯一解、可快速求解**
  这就是「稳定」的数学来源。
- **第四步，看 $\epsilon\|\lambda\|^2$（正则项）**：加一个小阻尼防止接触力「零空间漂移」——**让解唯一且平滑**，
避免接触力振荡。

## 6 辨析｜MuJoCo vs PyBullet vs Isaac；MJCF vs URDF

**辨析｜易错点一：MuJoCo、PyBullet、Isaac 是三种取向。** MuJoCo：接触稳定、快速、可微（MJX），**RL 与接触仿真首选**；
PyBullet：**URDF 原生、易用**，
但接触不如 MuJoCo 稳，
适合 ROS 生态与快速验证；
Isaac（Gym/Sim）：**GPU 并行**，
上万环境并行 RL 的王者，
但更重。
<span class="marginnote">选型判断：<strong>接触丰富、要稳定 → MuJoCo；
ROS/URDF 生态、快速验证 → PyBullet；
大规模并行 RL（四足、灵巧手）→ Isaac</strong>。
三者常混用——研究里 MuJoCo 打底、Isaac 扩规模。
</span>

**辨析｜易错点二：MJCF 与 URDF 不是「两种格式」那么简单。** MJCF 为仿真优化（接触、执行器、传感器配置齐全）；
URDF 为 ROS 描述优化（少仿真细节）。**用 MuJoCo 最好写/转换到 MJCF**——直接吃 URDF 会有接触、执行器配置缺失。
官方 **MuJoCo Menagerie** / 社区工具能自动转换，
但要检查。

**再辨析｜仿真步长是「精度 vs 速度」的旋钮**。
步长小（0.1 ms）接触准但慢；
步长大（2 ms）快但可能「漏掉快速接触」或抖。**RL 训练常在 500 Hz–1 kHz 仿真步长**——足够稳又不至于太慢；
要「绝对准」的接触（精密装配）需要更小步长或专用接触仿真。

## 7 小结

- **MuJoCo** = 接触动力学仿真器，
凸优化接触求解——稳定、快、可微，
RL/接触任务事实标准。
- **接触建模**：软接触 + 摩擦锥约束 + 凸 QP——「不炸、不穿透、快」。
- **MJCF**：body/joint/geom/actuator/sensor 的 XML 描述；
支持 URDF 导入。
- **Python API**：MjModel（静态）+ MjData（动态）+ mj_step，
配 Gymnasium/Menagerie 开箱即用。
- MuJoCo（稳）、PyBullet（ROS 生态）、Isaac（GPU 并行）——按任务选，
可混用。

在下一节，
我们看「GPU 并行仿真」的王者：**Isaac Gym 与 Isaac Sim——GPU 并行仿真与大规模 RL 训练**。



