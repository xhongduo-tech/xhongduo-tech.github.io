---
pageClass: plain-doc
---

# AI for Science

用深度学习重塑科学研究的范式：从蛋白质结构到天气气候，从分子药物到定理证明。本篇覆盖 AI4Science 的核心方向，对标各领域的经典论文、课程与专著体系。

## 主题规划

<ProgressGrid cat="advanced/ai4science" />


### 第一篇 AI4Science 概述与范式

- [x] [AI for Science 的兴起：从实验、理论、计算到数据驱动的第四范式](./fourth-paradigm)
- [x] [科学机器学习（Scientific Machine Learning, SciML）的问题分类与研究版图](./sciml-taxonomy)
- [x] [科学数据的特点：多尺度、稀疏、噪声与物理约束](./scientific-data-characteristics)
- [x] [对称性与等变性：群论视角下的神经网络设计原则](./symmetry-equivariance-groups)
- [x] [可微分编程与科学计算：自动微分在物理模拟中的角色](./differentiable-programming-autodiff)
- [x] [AI 科学家的工作流：假设生成、实验设计、数据分析与自动化闭环](./ai-scientist-workflow)

### 第二篇 蛋白质结构预测

- [x] [蛋白质折叠问题：从 Anfinsen 原理到 Levinthal 悖论](./protein-folding-anfinsen-levinthal)
- [x] [多序列比对（MSA）与共进化信息的提取](./msa-coevolution)
- [x] [AlphaFold2 架构解析：Evoformer 与三角注意力更新](./alphafold2-evoformer)
- [ ] AlphaFold2 的结构模块：不变点注意力（IPA）与端到端可微优化
- [ ] 置信度评估：pLDDT 与 PAE 的含义与解读
- [ ] ESMFold：用语言模型绕开 MSA 的快速结构预测
- [ ] AlphaFold3：扩散模型统一蛋白质、核酸与配体复合物预测
- [ ] 蛋白质设计：ProteinMPNN、RFdiffusion 与逆向折叠

### 第三篇 分子与药物发现

- [ ] 分子表征：SMILES、分子图与 3D 构象几何
- [ ] 图神经网络用于分子性质预测：消息传递机制与预训练策略
- [ ] 分子指纹与描述符：从 Morgan 指纹到神经指纹
- [ ] 分子生成模型：VAE、GAN 与基于扩散的分子生成
- [ ] 3D 分子生成：等变扩散模型（EDM）与构象生成
- [ ] 虚拟筛选：分子对接、结合亲和力预测与打分函数
- [ ] ADMET 性质预测与成药性评估
- [ ] 先导化合物优化：基于强化学习的分子设计
- [ ] 靶点识别与药物重定位

### 第四篇 AI 与数学

- [ ] 形式化数学与 Lean 定理证明器入门
- [ ] 自动定理证明：从启发式搜索到语言模型引导的证明生成
- [ ] 大模型定理证明实践：GPT-f、LeanDojo 与 DeepSeek-Prover
- [ ] FunSearch：用大模型在函数空间中搜索新发现
- [ ] AI 辅助数学猜想：Pattern Boost 与直觉的机器化
- [ ] 神经符号推理：符号计算与深度学习的融合
- [ ] AlphaGeometry 与几何定理证明

### 第五篇 AI 与物理模拟

- [ ] 物理信息神经网络（PINN）：将控制方程写入损失函数
- [ ] PINN 的训练难点：频谱偏差、损失加权与因果训练
- [ ] 神经算子（Neural Operator）：学习函数空间之间的映射
- [ ] 傅里叶神经算子（FNO）架构详解
- [ ] DeepONet 与通用算子逼近定理
- [ ] 网格无关性与多分辨率学习
- [ ] 用神经网络加速流体模拟：从 Navier-Stokes 到湍流建模
- [ ] 数据驱动的降阶模型与代理模型（Surrogate Model）

### 第六篇 天气与气候预测

- [ ] 数值天气预报（NWP）的传统技术路线及其瓶颈
- [ ] FourCastNet：基于傅里叶神经算子的全球天气预报
- [ ] GraphCast：图神经网络与多尺度网格上的中期预报
- [ ] 盘古气象大模型：3D 地球特定 Transformer（3DEST）
- [ ] 集合预报与概率天气预报
- [ ] Nowcasting：短临降水预报与雷达回波外推（DGMR）
- [ ] 气候模拟与 AI 降尺度（Downscaling）
- [ ] 数据同化与 AI 的结合

### 第七篇 材料发现

- [ ] 材料信息学：从材料基因组计划到数据驱动发现
- [ ] 晶体结构与周期性的数学表示
- [ ] 晶体图神经网络：CGCNN 与等变图网络（M3GNet、CHGNet）
- [ ] GNoME：大规模图网络主动学习发现稳定晶体
- [ ] 形成能、带隙与弹性性质预测
- [ ] 原子间势函数：机器学习力场（MACE、NequIP）
- [ ] 逆向材料设计与生成模型（MatterGen、CDVAE）

### 第八篇 计算生物学

- [ ] 单细胞 RNA 测序数据分析流程：降维、聚类与细胞注释
- [ ] 单细胞基础模型：scGPT、Geneformer 与 scFoundation
- [ ] 基因组学语言模型：DNABERT、Nucleotide Transformer
- [ ] 基因调控网络推断与染色质可及性预测（Enformer）
- [ ] 变异效应预测与致病性评估（AlphaMissense）
- [ ] 空间转录组学与多模态整合
- [ ] AI 驱动的药物-基因关联与精准医疗

### 第九篇 AI 与化学

- [ ] 化学反应的表示：反应 SMILES 与反应图
- [ ] 反应产物预测：基于模板与无模板的序列到序列方法
- [ ] 逆合成分析：单步逆合成模型与多步路线规划
- [ ] 反应条件推荐与产率预测
- [ ] 量子化学性质计算与机器学习加速（DFT 代理模型）
- [ ] 自动化化学实验室：机器人化学家与闭环实验

### 第十篇 科学基础模型与科学 Agent

- [ ] 科学基础模型总览：跨学科预训练模型的设计原则
- [ ] 多模态科学数据的对齐：文本、结构、序列与信号
- [ ] 科学知识增强的大语言模型：文献挖掘与知识图谱
- [ ] 科学 Agent：自主规划、工具调用与实验执行
- [ ] AI 科研助手的评测基准：ScienceQA、DiscoveryBench 等
- [ ] 人机协同的科学发现循环

### 第十一篇 微分方程与科学机器学习

- [ ] 常微分方程的神经网络求解与神经常微分方程（Neural ODE）
- [ ] 偏微分方程求解范式对比：PINN、神经算子与传统数值方法
- [ ] 正问题与反问题：参数辨识与源项反演
- [ ] 多物理场耦合问题的学习求解
- [ ] 守恒律与哈密顿/拉格朗日神经网络
- [ ] 通用近似能力之外的可靠性：误差估计与不确定性量化
- [ ] 科学机器学习的基准测试与开放问题

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
