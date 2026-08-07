---
pageClass: plain-doc
---

# 深度学习

对标 Goodfellow《深度学习》（花书）与李沐《动手学深度学习》的章节体系，覆盖从数学基础到工程实践的完整学习路径。学完一本经典教材 = 写完其全部章节对应的博文。

## 主题规划

<ProgressGrid cat="advanced/deep-learning" />


### 第一篇 数学基础回顾

- [x] [线性代数：标量、向量、矩阵与张量](./linear-algebra-basics)
- [x] [矩阵分解：特征分解与奇异值分解](./matrix-decomposition)
- [x] [范数、迹运算与伪逆](./norms-trace-pseudoinverse)
- [x] [概率论基础：随机变量与概率分布](./probability-basics)
- [x] [常见分布：伯努利、高斯、指数族分布](./common-distributions)
- [x] [条件概率、期望、方差与协方差](./conditional-probability-expectation-covariance)
- [x] [最大似然估计与贝叶斯统计](./maximum-likelihood-bayesian-statistics)
- [x] [信息论：熵、交叉熵与 KL 散度](./information-theory-entropy-kl)
- [x] [数值计算：上溢、下溢与病态条件](./numerical-computation-underflow-overflow)
- [x] [梯度与雅可比矩阵、黑塞矩阵](./gradient-jacobian-hessian)
- [x] [微积分与自动微分基础](./calculus-autograd-basics)

### 第二篇 深度学习基础

- [x] [机器学习基础：容量、过拟合与欠拟合](./ml-basics-capacity-overfitting)
- [x] [没有免费午餐定理与正则化思想](./no-free-lunch-regularization)
- [x] [超参数、验证集与交叉验证](./hyperparameters-validation-cross-validation)
- [x] [线性回归与从零实现](./linear-regression-scratch)
- [x] [Softmax 回归与分类问题](./softmax-regression-classification)
- [x] [感知机与多层感知机](./perceptron-multilayer-perceptron)
- [x] [深度前馈网络：隐藏单元与通用近似定理](./feedforward-networks-universal-approximation)
- [x] [激活函数：Sigmoid、Tanh、ReLU 及其变体](./activation-functions)
- [x] [损失函数设计：最大似然视角下的输出单元](./loss-functions-output-units)
- [x] [反向传播算法：链式法则与计算图](./backpropagation-chain-rule)
- [x] [前向传播与反向传播的符号推导](./forward-backward-symbolic-derivation)
- [x] [数值梯度检验与高效实现](./gradient-checking-efficient-implementation)

### 第三篇 正则化与优化

- [x] [正则化：L2 参数惩罚与权重衰减](./l2-regularization-weight-decay)
- [x] [正则化：L1 正则化与稀疏表示](./l1-regularization-sparsity)
- [x] [数据集增强与噪声鲁棒性](./data-augmentation-noise-robustness)
- [x] [提前终止与多任务学习](./early-stopping-multitask-learning)
- [x] [Dropout 与 DropConnect](./dropout-dropconnect)
- [x] [批量归一化与层归一化](./batchnorm-layernorm)
- [x] [参数绑定与参数共享](./parameter-sharing-tying)
- [x] [对抗训练与流形正切分类器](./adversarial-training-manifold-tangent-classifier)
- [x] [优化问题：病态条件、局部极小值与鞍点](./optimization-challenges-conditioning-saddle)
- [x] [随机梯度下降与小批量策略](./stochastic-gradient-descent-minibatch)
- [x] [动量法与 Nesterov 加速梯度](./momentum-nesterov)
- [x] [学习率策略：衰减、预热与余弦退火](./learning-rate-schedules)
- [x] [自适应学习率：AdaGrad 与 RMSProp](./adagrad-rmsprop)
- [x] [Adam 及其变体（AdamW、AMSGrad）](./adam-adamw-amsgrad)
- [x] [二阶优化近似：牛顿法与拟牛顿法](./second-order-optimization-newton-quasi)
- [x] [参数初始化：Xavier 与 Kaiming 初始化](./parameter-initialization-xavier-kaiming)
- [x] [梯度消失、梯度爆炸与梯度裁剪](./vanishing-exploding-gradients-clipping)

### 第四篇 卷积神经网络

- [x] [卷积运算与互相关](./convolution-cross-correlation)
- [x] [卷积的三大动机：稀疏交互、参数共享、等变表示](./convolution-motivations)
- [x] [填充、步幅与感受野](./padding-stride-receptive-field)
- [x] [池化：最大池化与平均池化](./pooling-max-average)
- [x] [多输入多输出通道与 1×1 卷积](./multi-channel-convolution-1x1)
- [x] [LeNet：首个成功的卷积网络](./lenet)
- [x] [AlexNet：深度卷积网络的开端](./alexnet)
- [x] [VGG：使用块的网络与深度探索](./vgg)
- [x] [NiN：网络中的网络与全局平均池化](./nin-network-in-network)
- [x] [GoogLeNet：Inception 并行结构](./googlenet-inception)
- [x] [ResNet：残差连接与恒等映射](./resnet)
- [x] [DenseNet：稠密连接的特征复用](./densenet)
- [x] [深度可分离卷积与 MobileNet 轻量化设计](./depthwise-separable-mobilenet)
- [x] [卷积网络的可视化与特征解释](./cnn-visualization-interpretation)

### 第五篇 序列建模

- [x] [序列数据与语言模型基础](./sequence-data-language-models)
- [x] [循环神经网络的结构与前向计算](./rnn-forward-computation)
- [x] [通过时间的反向传播（BPTT）](./bptt)
- [x] [序列的截断与梯度计算策略](./sequence-truncation-gradient-strategy)
- [x] [LSTM：门控记忆与长依赖](./lstm)
- [x] [GRU：门控循环单元](./gru)
- [x] [深层循环神经网络与双向循环神经网络](./deep-bidirectional-rnn)
- [x] [序列模型的采样与文本生成](./sequence-sampling-text-generation)
- [x] [序列到序列学习（Seq2Seq）与编码器-解码器架构](./seq2seq-encoder-decoder)
- [x] [束搜索（Beam Search）](./beam-search)
- [x] [注意力机制：Bahdanau 与 Luong 注意力](./attention-bahdanau-luong)
- [x] [注意力汇聚：Nadaraya-Watson 核回归视角](./nadaraya-watson-attention)
- [x] [自注意力与缩放点积注意力](./self-attention-scaled-dot-product)

### 第六篇 Transformer 与预训练

- [x] [Transformer 整体架构解析](./transformer-architecture)
- [x] [多头注意力机制](./multi-head-attention)
- [x] [位置编码：正弦编码与可学习编码](./positional-encoding)
- [x] [位置编码进阶：RoPE 与 ALiBi](./rope-alibi)
- [x] [Transformer 的编码器、解码器与交叉注意力](./encoder-decoder-cross-attention)
- [x] [前馈网络、残差连接与归一化位置（Pre-LN/Post-LN）](./ffn-residual-preln-postln)
- [x] [BERT：掩码语言模型与预训练表征](./bert)
- [x] [GPT 系列：自回归语言建模](./gpt-series)
- [x] [预训练任务设计：MLM、NSP、因果语言建模](./pretraining-tasks-mlm-nsp-causal)
- [x] [下游任务微调与提示学习](./fine-tuning-prompting)
- [x] [大语言模型的缩放定律与涌现能力](./scaling-laws-emergence)

### 第七篇 表示学习与生成模型

- [x] [表示学习：分布式表示与嵌入](./representation-learning-embeddings)
- [x] [词嵌入：Word2Vec 与 GloVe](./word2vec-glove)
- [x] [自编码器：欠完备、稀疏与去噪自编码器](./autoencoders)
- [x] [变分自编码器（VAE）与重参数化技巧](./vae-reparameterization)
- [x] [生成对抗网络（GAN）：博弈与训练动态](./gan)
- [x] [GAN 的改进：WGAN、条件 GAN 与模式崩溃](./gan-improvements-wgan-cgan)
- [x] [扩散模型：前向加噪与反向去噪](./diffusion-models)
- [x] [DDPM 与 DDIM：采样加速](./ddim-sampling-acceleration)
- [x] [流模型与归一化流（Normalizing Flow）](./normalizing-flows)
- [x] [自监督学习：预训练任务的构造](./self-supervised-learning)
- [x] [对比学习：SimCLR 与 MoCo](./contrastive-learning-simclr-moco)
- [x] [掩码自编码器（MAE）与掩码图像建模](./mae-masked-image-modeling)

### 第八篇 视觉与多模态扩展

- [x] [Vision Transformer（ViT）：图像分块嵌入](./vision-transformer)
- [x] [Swin Transformer：移位窗口与层级结构](./swin-transformer)
- [x] [目标检测基础：边界框、锚框与非极大值抑制](./object-detection-basics)
- [x] [R-CNN 系列与单阶段检测（SSD、YOLO）](./rcnn-yolo-detection)
- [x] [语义分割与全卷积网络、U-Net](./semantic-segmentation-fcn-unet)
- [x] [多模态学习：CLIP 与图文对齐](./clip-multimodal-alignment)
- [x] [神经风格迁移与图像生成的经典应用](./neural-style-transfer)

### 第九篇 实践方法论与工程

- [x] [性能指标与默认基准模型](./performance-metrics-baselines)
- [x] [数据预处理：归一化、标准化与增强策略](./data-preprocessing-normalization-augmentation)
- [x] [调参方法论：网格搜索、随机搜索与贝叶斯优化](./hyperparameter-search)
- [x] [调试策略：从单样本过拟合到全流程检查](./debugging-strategies)
- [x] [混合精度训练与梯度缩放](./mixed-precision-training)
- [x] [分布式训练：数据并行与模型并行](./distributed-training-data-model-parallelism)
- [x] [多 GPU 训练与梯度同步](./multi-gpu-gradient-sync)
- [x] [PyTorch 工程实践：数据加载与 Dataset/DataLoader](./pytorch-dataloader)
- [x] [PyTorch 工程实践：模型定义、训练循环与检查点](./pytorch-training-loop-checkpoint)
- [x] [模型部署：ONNX 导出与推理优化](./model-deployment-onnx)
- [x] [实验管理：日志、可复现性与随机性控制](./experiment-management)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
