---
title: 模拟器与真实后端：IBM Quantum 云端实验
date: 2026-08-07
---

# 模拟器与真实后端：IBM Quantum 云端实验

<div class="epigraph">
<p>模拟器告诉你「理论上该得到什么」，真机告诉你「现实给了你什么」。</p>
<footer>—— IBM Quantum 团队（代拟）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit-ibm-runtime 文档 ｜ 2026-08-07</p>
</div>

## 为什么从模拟器与真机的对比开始

前面所有实验都在本地模拟器上跑——理想、无噪、可复现。但量子计算的价值在真机。本节讲两条线：**模拟器**（`AerSimulator` 的各种模式）与**真实后端**（IBM Quantum 云上的超导处理器），以及连接两者的关键步骤——**认证、运行、对比理想与噪声**。<span class="marginnote">模拟器分三类：`statevector`（精确态矢量）、`matrix_product_state`（张量网络，省内存）、`qasm`（采样）。真机通过 `qiskit-ibm-runtime` 连接，IBM Quantum 提供免费公开后端（如 `ibm_brisbane`）。「模拟器验证逻辑 → 真机验证物理」是标准研发流程。</span>学完本节，你就能把「纸面算法」真正跑在「云端量子硬件」上。

## 1 模拟器的三种模式

```python
from qiskit_aer import AerSimulator

sv_sim = AerSimulator(method='statevector')    # 精确态矢量，适合小线路
mps_sim = AerSimulator(method='matrix_product_state')  # 张量网络，省内存
qasm_sim = AerSimulator(method='qasm')         # 采样模式，模拟真机行为
```

- **statevector**：精确计算终态矢量，再采样——最准、内存随比特数指数。
- **matrix_product_state**：用 MPS 近似，对「低纠缠」线路省内存，可跑更多比特。
- **qasm**：模拟「测量采样 + 可选噪声」，最接近真机统计。<span class="marginnote">选型经验：<strong>小线路（≤25 比特）用 statevector，大而浅的线路用 MPS，需要「真实统计感」的用 qasm + 噪声模型</strong>。AerSimulator 还能注入噪声模型（`from qiskit_aer.noise import NoiseModel`），模拟真实硬件的门错误与退相干——这是「真机预演」的关键工具。</span>

## 2 连接真实后端

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# 1. 认证（首次需要 API token）
service = QiskitRuntimeService(channel='ibm_quantum', token='你的token')

# 2. 选择后端
backend = service.least_busy(operational=True, simulator=False)

# 3. 运行（Sampler：采样任务的现代接口）
sampler = Sampler(mode=backend)
job = sampler.run([(qc, None, 1024)])   # (线路, 可观测, shots)
result = job.result()
counts = result[0].data.meas.get_counts()
```

- `QiskitRuntimeService` 认证并列出可用后端；`least_busy` 选最空闲的真机。
- `SamplerV2` 是 Qiskit 1.0 的现代采样接口，`(qc, None, shots)` 指定线路与采样数。<span class="marginnote">真实后端的「排队」是常态——免费用户的任务可能等几分钟到几小时。`Sampler` 会自动做 `transpile`（把逻辑门映射到硬件的原生门与连接），这是「逻辑线路 → 物理线路」的自动化（第十二篇最后一节详述）。IBM Quantum 还提供「模拟后端」（如 `ibmq_qasm_simulator`），可以无 token 先体验流程。</span>

## 3 公式解析：真机结果与理想结果的偏差来源

同一线路在模拟器与真机上跑的 counts 差异，来自三类误差：

$$
P_{\rm real}(x) = P_{\rm ideal}(x) \cdot F_{\rm gate} + \text{噪声项}
$$

- **第一步，门误差**：每个门以 $\sim 10^{-3}$–$10^{-2}$ 的错误率偏离理想——深度越大偏差越大（$d\bar\epsilon<1$ 铁律的实践）。
- **第二步，测量误差**：读出时 $\lvert1\rangle$ 可能被误判为 $\lvert0\rangle$ 等——真机 counts 有「读出偏置」。
- **第三步，退相干**：线路运行期间比特忘记自己的态——长线路更明显。<span class="marginnote">工程应对：<strong>先用噪声模拟器预演</strong>（把真机的校准数据 `backend.properties()` 喂给 `AerSimulator`），预测「真机会输出什么」；再跑真机，与预测对比。真机结果应落在「噪声模拟」的统计范围内——这是「真机实验合理性的体检」。</span>

## 4 实践：贝尔态真机 vs 模拟

跑 $\lvert\Phi^+\rangle$（H + CNOT + 测量），对比三种环境：

```python
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])

# 理想模拟
ideal = AerSimulator().run(transpile(qc, AerSimulator()), shots=1024).result().get_counts()
# 噪声模拟（用真机校准）
from qiskit_aer.noise import NoiseModel
nm = NoiseModel.from_backend(backend)
noisy = AerSimulator(noise_model=nm).run(transpile(qc, AerSimulator()), shots=1024).result().get_counts()
# 真机
real = sampler.run([(qc, None, 1024)]).result()[0].data.meas.get_counts()
```

- **理想**：`{'00': 512, '11': 512}`（完美纠缠态）。
- **噪声模拟**：`'01'`、`'10'` 出现少量（门 + 测量误差的预演）。
- **真机**：与噪声模拟相近，但可能略差（模型未覆盖的噪声）。<span class="marginnote">把三个 counts 并排看，你会直观理解「噪声长什么样」：<strong>理想是两条竖线，真机是四条竖线（两条短的在错误位置）</strong>。这个「理想 → 噪声预演 → 真机」三段对比，是量子实验的基本功，也是评估「算法在真机上的保真度」的标准流程。</span>

**辨析｜易错点：** 真机的「比特编号」与「连接图」与逻辑线路不同——`transpile` 会把你的逻辑比特映射到物理比特，映射结果可用 `qc_t.draw()` 查看。若你的线路需要「非最近邻」连接（如远程 CNOT），转译会插入 SWAP，深度与噪声大增——**为硬件拓扑设计线路**是量子编译的实战课（下一节）。

## 5 真机实验的「正确姿势」

- **先模拟再真机**：任何线路先在 statevector/噪声模拟上验证逻辑，再上真机——省排队、省 token、少踩坑。
- **shots 要够**：噪声下需要更多 shots 压统计误差（真机 counts 的噪声除了门误差还有 shot 噪声）。
- **用校准数据**：跑之前看 `backend.qubit_properties()` 与 `backend.coupling_map`，了解比特质量与拓扑。
- **误差缓解**：对期望值类任务，用「读出校正」（`measure_error_mitigation`）压测量误差。<span class="marginnote">真机实验是「量子工程的试炼场」：你会亲身体会为什么 NISQ 时代（第十篇）说「浅线路是唯一活路」——在真机上跑深线路，counts 会迅速「糊」成均匀噪声。这种「理论 vs 现实的落差感」，只有跑过真机才真正建立。</span>

## 6 小结

- **模拟器三模式**：statevector（精确）、MPS（省内存）、qasm（采样）+ 噪声模型（真机预演）。
- **连真机**：`QiskitRuntimeService` 认证 → `least_busy` 选后端 → `SamplerV2` 运行。
- **偏差来源**：门误差 + 测量误差 + 退相干——「理想 → 噪声预演 → 真机」三段对比。
- **拓扑意识**：`transpile` 映射逻辑→物理比特，非最近邻连接会插 SWAP 增噪。
- **正确姿势**：先模拟、shots 足够、用校准数据、上误差缓解。

在下一节，我们深入编译管线——**量子线路的编译、转译与噪声模拟**（第十二篇最后一篇）。
