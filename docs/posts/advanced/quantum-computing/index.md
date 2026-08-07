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
- [x] [量子隐形传态（quantum teleportation）](./quantum-teleportation)
- [x] [超密编码（superdense coding）](./superdense-coding)

### 第三篇 量子门与量子线路

- [x] [量子线路模型与线路图约定](./quantum-circuit-model)
- [x] [单比特门：X、Y、Z 与 Pauli 门](./single-qubit-gates-pauli)
- [x] [Hadamard 门与相位门（S、T 门）](./hadamard-phase-gates)
- [x] [旋转门：Rx、Ry、Rz 与任意单比特门分解](./rotation-gates-single-qubit-decomposition)
- [x] [受控门：CNOT、CZ 与受控-U](./controlled-gates-cnot-cz)
- [x] [Toffoli 门与 Fredkin 门](./toffoli-fredkin-gates)
- [x] [量子测量的线路表示与延迟测量原理](./quantum-measurement-circuit-delayed-measurement)
- [x] [通用量子门集：CNOT + 单比特门](./universal-gate-set-cnot-single-qubit)
- [x] [Solovay-Kitaev 定理与门的近似](./solovay-kitaev-theorem)
- [x] [线路的深度、宽度与复杂度](./circuit-depth-width-complexity)

### 第四篇 量子纠缠与贝尔不等式

- [x] [纠缠的定义：可分态与纠缠态](./entanglement-separable-entangled-states)
- [x] [贝尔态（Bell states）与贝尔测量](./bell-states-bell-measurement)
- [x] [EPR 佯谬与隐变量理论](./epr-paradox-hidden-variables)
- [x] [CHSH 不等式及其量子违背](./chsh-inequality)
- [x] [纠缠的度量：并发度（concurrence）与纠缠熵](./entanglement-measures-concurrence-entropy)
- [x] [GHZ 态与 W 态：多体纠缠的两种类型](./ghz-w-states-multipartite-entanglement)

### 第五篇 量子算法基础

- [x] [量子查询复杂度与黑盒模型](./quantum-query-complexity-black-box)
- [x] [Deutsch 算法与 Deutsch-Jozsa 算法](./deutsch-jozsa-algorithm)
- [x] [Bernstein-Vazirani 算法](./bernstein-vazirani)
- [x] [Simon 算法](./simon-algorithm)
- [x] [量子傅里叶变换（QFT）及其实现](./quantum-fourier-transform)
- [x] [相位估计（phase estimation）算法](./phase-estimation)
- [x] [量子振幅放大与振幅估计](./amplitude-amplification-estimation)

### 第六篇 Shor 算法

- [x] [大数分解与 RSA 密码的关系](./factoring-rsa-cryptography)
- [x] [大数分解归约为周期寻找问题](./factoring-to-period-finding)
- [x] [模幂运算的量子线路实现](./modular-exponentiation-circuit)
- [x] [Shor 算法的完整流程分析](./shor-algorithm-full-analysis)
- [x] [Shor 算法的复杂度与经典算法对比](./shor-complexity-classical-comparison)
- [x] [离散对数的量子算法及其对椭圆曲线密码的威胁](./discrete-logarithm-elliptic-curve)

### 第七篇 Grover 算法与搜索

- [x] [无结构搜索问题与经典下界](./unstructured-search-classical-lower-bound)
- [x] [Grover 迭代：Oracle 与扩散算子](./grover-iteration-oracle-diffusion)
- [x] [Grover 算法的几何解释：旋转与振幅放大](./grover-geometric-rotation-amplification)
- [x] [多次解与部分解的搜索](./grover-multiple-solutions)
- [x] [Grover 算法的复杂度分析：O(√N) 的最优性](./grover-complexity-optimality)
- [x] [Grover 算法的应用：碰撞查找、计数与最小值查找](./grover-applications-collision-counting-minimum)

### 第八篇 量子纠错

- [x] [经典纠错与量子纠错的困难](./classical-quantum-error-correction-challenges)
- [x] [三比特比特翻转码（bit-flip code）](./bit-flip-code)
- [x] [三比特相位翻转码（phase-flip code）](./phase-flip-code)
- [x] [Shor 九比特码](./shor-nine-qubit-code)
- [x] [量子纠错的条件与差错离散化](./quantum-error-correction-conditions-discretization)
- [x] [稳定子（stabilizer）形式体系](./stabilizer-formalism)
- [x] [CSS 码与 Steane 码](./css-steane-codes)
- [x] [表面码（surface code）初步与容错阈值](./surface-code-fault-tolerance-threshold)
- [x] [容错量子计算与阈值定理](./fault-tolerant-quantum-computation-threshold-theorem)

### 第九篇 量子硬件

- [x] [量子比特的物理实现条件：DiVincenzo 判据](./divincenzo-criteria)
- [x] [超导量子比特：Transmon 与磁通量子比特](./superconducting-qubits-transmon-flux)
- [x] [离子阱（trapped ion）量子计算](./trapped-ion-quantum-computing)
- [x] [光量子计算与线性光学方案](./optical-quantum-computing-linear-optics)
- [x] [中性原子与里德伯原子阵列](./neutral-atoms-rydberg-arrays)
- [x] [其他路线：半导体自旋、拓扑量子计算](./other-approaches-spin-topological)
- [x] [退相干、噪声与量子门保真度](./decoherence-noise-gate-fidelity)

### 第十篇 NISQ 与变分量子算法

- [x] [NISQ 时代的定义与限制](./nisq-definition-limitations)
- [x] [变分量子算法的框架：参数化线路与经典优化](./variational-quantum-algorithm-framework)
- [x] [变分量子本征求解器（VQE）](./vqe-variational-quantum-eigensolver)
- [x] [量子近似优化算法（QAOA）](./qaoa-quantum-approximate-optimization)
- [x] [成本函数景观与贫瘠高原（barren plateau）](./barren-plateaus-cost-landscape)
- [x] [量子退火与绝热量子计算](./quantum-annealing-adiabatic)

### 第十一篇 量子机器学习初步

- [x] [量子机器学习的问题设定与数据编码](./qml-problem-setting-data-encoding)
- [x] [振幅编码、角度编码与基编码](./amplitude-angle-basis-encoding)
- [x] [量子核方法（quantum kernel）](./quantum-kernel-methods)
- [x] [量子神经网络（QNN）与参数化量子线路](./quantum-neural-networks-parameterized-circuits)
- [x] [HHL 算法与量子线性代数](./hhl-quantum-linear-algebra)
- [x] [量子机器学习的加速争议与去量化](./qml-speedup-controversy-dequantization)

### 第十二篇 量子编程实践（Qiskit）

- [x] [Qiskit 环境搭建与第一个量子线路](./qiskit-setup-first-circuit)
- [x] [在 Qiskit 中实现布洛赫球上的单比特门](./qiskit-single-qubit-gates-bloch)
- [x] [用 Qiskit 构造贝尔态并验证纠缠](./qiskit-bell-states-entanglement)
- [x] [用 Qiskit 实现 Deutsch-Jozsa 算法](./qiskit-deutsch-jozsa)
- [x] [用 Qiskit 实现 QFT 与相位估计](./qiskit-qft-phase-estimation)
- [x] [用 Qiskit 实现 Grover 搜索](./qiskit-grover)
- [x] [用 Qiskit 实现 VQE 与 QAOA 实例](./qiskit-vqe-qaoa)
- [x] [模拟器与真实后端：IBM Quantum 云端实验](./qiskit-simulators-real-backends)
- [x] [量子线路的编译、转译与噪声模拟](./qiskit-compilation-transpilation-noise)

### 第十三篇 量子计算的现状与展望

- [x] [量子优势（quantum supremacy）：随机线路采样与玻色采样](./quantum-supremacy-random-circuit-sampling-boson-sampling)
- [x] [后量子密码（post-quantum cryptography）的应对](./post-quantum-cryptography)
- [x] [量子互联网与量子中继](./quantum-internet-repeaters)
- [x] [量子计算在化学、材料与优化中的应用前景](./quantum-applications-chemistry-materials-optimization)
- [x] [量子计算的开放问题与学习路线图](./open-problems-learning-roadmap)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
