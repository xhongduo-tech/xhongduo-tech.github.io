---
pageClass: plain-doc
---

# 量子计算

学完量子计算，就是写完 Nielsen & Chuang《量子计算与量子信息》与 IBM Qiskit 课程体系对应的全部博文。本篇从量子力学与线性代数基础出发，覆盖量子线路、量子算法、量子纠错、量子硬件与 NISQ 时代的变分算法，直至量子编程实践与前沿展望。

## 主题规划

<ProgressGrid cat="advanced/quantum-computing" />


### 第零篇 量子计算概述

- [x] [量子计算的起源：Feynman 的提议与量子模拟](./origin-feynman)
- [x] [量子计算的发展简史：从 Deutsch 到 NISQ](./history-deutsch-to-nisq)
- [x] [量子计算为什么可能更快：量子并行性与纠缠](./why-faster-parallelism-entanglement)
- [x] [量子计算的计算模型：线路模型、绝热计算与测量量子计算](./computational-models)
- [x] [量子计算的误区澄清：指数加速不是免费的](./myths-exponential-speedup)
- [x] [量子信息的基本任务：计算、通信、密码与模拟](./basic-tasks-quantum-information)

### 第一篇 线性代数与量子力学基础

- [x] [复数域上的向量空间与内积](./complex-vector-spaces-inner-product)
- [x] [Dirac 记号（bra-ket）：右矢、左矢与内外积](./dirac-notation-bra-ket)
- [x] [线性算符与矩阵表示](./linear-operators-matrix-representation)
- [x] [厄米算符与幺正算符](./hermitian-unitary-operators)
- [x] [本征值、本征向量与谱分解](./eigenvalues-spectral-decomposition)
- [x] [张量积（tensor product）：多体系统的状态空间](./tensor-product-multipartite)
- [x] [量子力学的基本假设：状态、演化、测量](./postulates-quantum-mechanics)
- [x] [投影测量与广义测量（POVM）](./projection-measurement-povm)
- [x] [密度算符：混合态与部分迹](./density-operator-mixed-states)
- [x] [Schmidt 分解与纯化](./schmidt-decomposition-purification)

### 第二篇 量子比特与布洛赫球

- [x] [量子比特：|0⟩、|1⟩ 与叠加态](./qubit-superposition)
- [x] [布洛赫球（Bloch sphere）表示](./bloch-sphere-representation)
- [x] [单比特量子态的测量与基的选择](./single-qubit-measurement-basis)
- [x] [多量子比特系统与纠缠态](./multiqubit-systems-entanglement)
- [x] [不可克隆定理（no-cloning theorem）](./no-cloning-theorem)
- [ ] 量子隐形传态（quantum teleportation）
- [ ] 超密编码（superdense coding）

### 第三篇 量子门与量子线路

- [ ] 量子线路模型与线路图约定
- [ ] 单比特门：X、Y、Z 与 Pauli 门
- [ ] Hadamard 门与相位门（S、T 门）
- [ ] 旋转门：Rx、Ry、Rz 与任意单比特门分解
- [ ] 受控门：CNOT、CZ 与受控-U
- [ ] Toffoli 门与 Fredkin 门
- [ ] 量子测量的线路表示与延迟测量原理
- [ ] 通用量子门集：CNOT + 单比特门
- [ ] Solovay-Kitaev 定理与门的近似
- [ ] 线路的深度、宽度与复杂度

### 第四篇 量子纠缠与贝尔不等式

- [ ] 纠缠的定义：可分态与纠缠态
- [ ] 贝尔态（Bell states）与贝尔测量
- [ ] EPR 佯谬与隐变量理论
- [ ] CHSH 不等式及其量子违背
- [ ] 纠缠的度量：并发度（concurrence）与纠缠熵
- [ ] GHZ 态与 W 态：多体纠缠的两种类型

### 第五篇 量子算法基础

- [ ] 量子查询复杂度与黑盒模型
- [ ] Deutsch 算法与 Deutsch-Jozsa 算法
- [ ] Bernstein-Vazirani 算法
- [ ] Simon 算法
- [ ] 量子傅里叶变换（QFT）及其实现
- [ ] 相位估计（phase estimation）算法
- [ ] 量子振幅放大与振幅估计

### 第六篇 Shor 算法

- [ ] 大数分解与 RSA 密码的关系
- [ ] 大数分解归约为周期寻找问题
- [ ] 模幂运算的量子线路实现
- [ ] Shor 算法的完整流程分析
- [ ] Shor 算法的复杂度与经典算法对比
- [ ] 离散对数的量子算法及其对椭圆曲线密码的威胁

### 第七篇 Grover 算法与搜索

- [ ] 无结构搜索问题与经典下界
- [ ] Grover 迭代：Oracle 与扩散算子
- [ ] Grover 算法的几何解释：旋转与振幅放大
- [ ] 多次解与部分解的搜索
- [ ] Grover 算法的复杂度分析：O(√N) 的最优性
- [ ] Grover 算法的应用：碰撞查找、计数与最小值查找

### 第八篇 量子纠错

- [ ] 经典纠错与量子纠错的困难
- [ ] 三比特比特翻转码（bit-flip code）
- [ ] 三比特相位翻转码（phase-flip code）
- [ ] Shor 九比特码
- [ ] 量子纠错的条件与差错离散化
- [ ] 稳定子（stabilizer）形式体系
- [ ] CSS 码与 Steane 码
- [ ] 表面码（surface code）初步与容错阈值
- [ ] 容错量子计算与阈值定理

### 第九篇 量子硬件

- [ ] 量子比特的物理实现条件：DiVincenzo 判据
- [ ] 超导量子比特：Transmon 与磁通量子比特
- [ ] 离子阱（trapped ion）量子计算
- [ ] 光量子计算与线性光学方案
- [ ] 中性原子与里德伯原子阵列
- [ ] 其他路线：半导体自旋、拓扑量子计算
- [ ] 退相干、噪声与量子门保真度

### 第十篇 NISQ 与变分量子算法

- [ ] NISQ 时代的定义与限制
- [ ] 变分量子算法的框架：参数化线路与经典优化
- [ ] 变分量子本征求解器（VQE）
- [ ] 量子近似优化算法（QAOA）
- [ ] 成本函数景观与贫瘠高原（barren plateau）
- [ ] 量子退火与绝热量子计算

### 第十一篇 量子机器学习初步

- [ ] 量子机器学习的问题设定与数据编码
- [ ] 振幅编码、角度编码与基编码
- [ ] 量子核方法（quantum kernel）
- [ ] 量子神经网络（QNN）与参数化量子线路
- [ ] HHL 算法与量子线性代数
- [ ] 量子机器学习的加速争议与去量化

### 第十二篇 量子编程实践（Qiskit）

- [ ] Qiskit 环境搭建与第一个量子线路
- [ ] 在 Qiskit 中实现布洛赫球上的单比特门
- [ ] 用 Qiskit 构造贝尔态并验证纠缠
- [ ] 用 Qiskit 实现 Deutsch-Jozsa 算法
- [ ] 用 Qiskit 实现 QFT 与相位估计
- [ ] 用 Qiskit 实现 Grover 搜索
- [ ] 用 Qiskit 实现 VQE 与 QAOA 实例
- [ ] 模拟器与真实后端：IBM Quantum 云端实验
- [ ] 量子线路的编译、转译与噪声模拟

### 第十三篇 量子计算的现状与展望

- [ ] 量子优势（quantum supremacy）：随机线路采样与玻色采样
- [ ] 后量子密码（post-quantum cryptography）的应对
- [ ] 量子互联网与量子中继
- [ ] 量子计算在化学、材料与优化中的应用前景
- [ ] 量子计算的开放问题与学习路线图

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
