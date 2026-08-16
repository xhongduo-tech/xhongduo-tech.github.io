---
pageClass: plain-doc
---

# 端侧 AI 与小模型（量化蒸馏/NPU 部署/SLM）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- MIT Technology Review, "10 Breakthrough Technologies 2025" — Small Language Models (2025)
- Han, Mao & Dally, "Deep Compression" 论文体系 (ICLR 2016)
- Apple, "Apple Intelligence 端侧架构" 及 Qualcomm AI 白皮书 (2024)

## 主题规划

<ProgressGrid cat="advanced/on-device-ai" />

### 第1篇

- [x] [端侧 AI 的驱动力（隐私/延迟/成本、云端协同的边界）](./on-device-ai-drivers)
- [x] [小语言模型 SLM（Phi/Gemma Mini 的数据质量优先路线）](./small-language-models-slm)
- [x] [知识蒸馏（Logits/特征/注意力蒸馏、从大模型到小模型）](./knowledge-distillation)
- [x] [量化技术（PTQ/QAT、INT8→INT4→二值化的精度保卫战）](./model-quantization)
- [x] [剪枝与稀疏化（结构化剪枝、N:M 稀疏的硬件亲和性）](./pruning-and-sparsity)
- [x] [高效架构（MobileNet/Mamba 变体、混合精度设计）](./efficient-model-architectures)
- [x] [NPU/DSP 硬件（移动 SoC 的 AI 算力、算子支持矩阵）](./npu-dsp-hardware)
- [x] [推理框架（ONNX/CoreML/MLC-LLM、端侧推理引擎）](./inference-frameworks)

### 第2篇

- [x] [端侧大模型部署（手机跑 7B 的内存/功耗工程）](./on-device-llm-deployment)
- [x] [端云协同（投机采样的云端版、隐私计算卸载）](./device-cloud-collaboration)
- [x] [个人化（端侧微调 LoRA、设备端 RAG）](./on-device-personalization)
- [x] [应用场景（AI 手机/AI PC/可穿戴、离线翻译与影像）](./application-scenarios)
