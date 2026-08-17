---
pageClass: plain-doc
---

# 信息论

对标 Cover & Thomas《Elements of Information Theory》的核心章节，从熵与互信息出发，覆盖信源编码、信道容量、率失真理论与科尔莫戈罗夫复杂度，并延伸至信息论与统计、机器学习的联系。

## 主题规划

<ProgressGrid cat="intermediate/information-theory" />


### 第一篇 熵、相对熵与互信息

- [x] [自信息（Self-information）与熵的定义](./self-information-and-entropy)
- [x] [熵的性质：非负性、对称性与可加性](./entropy-properties)
- [x] [联合熵（Joint Entropy）与条件熵（Conditional Entropy）](./joint-and-conditional-entropy)
- [x] [熵的链式法则（Chain Rule for Entropy）](./chain-rule-for-entropy)
- [x] [相对熵（KL 散度，Relative Entropy）的定义与性质](./relative-entropy)
- [x] [互信息（Mutual Information）的定义与多种等价表达](./mutual-information)
- [x] [条件互信息与互信息的链式法则](./conditional-mutual-information-and-chain-rule)
- [x] [熵、条件熵与互信息之间的 Venn 图关系](./information-venn-diagram)
- [x] [相对熵的性质：非负性与吉布斯不等式](./relative-entropy-nonnegativity-gibbs-inequality)
- [x] [相对熵的凸性与链式法则](./relative-entropy-convexity-chain-rule)

### 第二篇 基本不等式与数据处理

- [x] [对数和不等式（Log Sum Inequality）](./log-sum-inequality)
- [x] [Jensen 不等式及其在信息论中的应用](./jensen-inequality-information-theory)
- [x] [数据处理不等式（Data Processing Inequality）](./data-processing-inequality)
- [x] [马尔可夫链上的互信息单调性](./markov-chain-mutual-information-monotonicity)
- [x] [充分统计量（Sufficient Statistic）与数据处理不等式](./sufficient-statistic-data-processing)
- [x] [Fano 不等式及其误差概率含义](./fano-inequality)
- [x] [均匀分布与最大熵](./uniform-distribution-maximum-entropy)
- [x] [条件作用使熵减小（Conditioning Reduces Entropy）](./conditioning-reduces-entropy)

### 第三篇 渐近均分性（AEP）

- [x] [弱大数定律回顾与依概率收敛](./weak-law-large-numbers)
- [x] [渐近均分性定理（Asymptotic Equipartition Property）](./asymptotic-equipartition-property)
- [x] [典型集（Typical Set）的定义与性质](./typical-set)
- [x] [高概率集与典型集的关系](./high-probability-set-typical-set)
- [x] [AEP 与数据压缩的联系](./aep-data-compression)
- [x] [联合典型性与联合典型序列](./jointly-typical-sequences)
- [x] [联合典型序列的数量估计](./count-jointly-typical-sequences)

### 第四篇 信源编码与数据压缩

- [x] [信源编码问题与码的分类（唯一可译码、即时码、前缀码）](./source-coding-codes-classification)
- [x] [Kraft 不等式（Kraft's Inequality）](./kraft-inequality)
- [x] [最优码长度与码长的下界](./optimal-code-length-lower-bound)
- [x] [香农码（Shannon Code）及其构造](./shannon-code)
- [x] [霍夫曼编码（Huffman Coding）的构造与最优性证明](./huffman-coding)
- [x] [算术编码（Arithmetic Coding）的原理与实现](./arithmetic-coding)
- [x] [Lempel-Ziv 通用编码（LZ77/LZ78）初步](./lz-coding)
- [x] [信源编码定理：平均码长的熵界](./source-coding-theorem)

### 第五篇 信道容量

- [x] [离散无记忆信道（DMC）模型](./dmc-model)
- [x] [信道容量的定义与直观含义](./channel-capacity-definition)
- [x] [无噪二元信道与有噪信道的例子](./noiseless-binary-channel)
- [x] [二进制对称信道（Binary Symmetric Channel）的容量](./binary-symmetric-channel-capacity)
- [x] [二进制删除信道（Binary Erasure Channel）的容量](./binary-erasure-channel-capacity)
- [x] [对称信道及其容量的简化计算](./symmetric-channel-capacity)
- [x] [信道容量的性质与求解思路](./channel-capacity-properties)

### 第六篇 信道编码定理初步

- [x] [联合典型序列与译码方法](./jointly-typical-decoding)
- [x] [随机编码思想与典型序列译码](./random-coding-typical-decoding)
- [x] [信道编码定理：可达性的证明](./channel-coding-theorem-direct)
- [x] [信道编码定理逆定理（Fano 不等式的应用）](./channel-coding-theorem-converse)
- [x] [信源信道分离定理（Source-Channel Separation）](./source-channel-separation)
- [x] [反馈信道与反馈不能增大容量的结论](./feedback-channel-capacity)

### 第七篇 微分熵

- [x] [微分熵（Differential Entropy）的定义](./differential-entropy-definition)
- [x] [均匀分布、指数分布与高斯分布的微分熵](./differential-entropy-common-distributions)
- [x] [微分熵与离散熵的区别：可为负值的原因](./differential-entropy-negative)
- [x] [联合微分熵与条件微分熵](./joint-conditional-differential-entropy)
- [x] [相对熵与互信息的连续形式](./relative-entropy-mutual-information-continuous)
- [x] [微分熵的性质：变换下的变化规律](./differential-entropy-transform)
- [x] [最大微分熵定理：高斯分布的熵最大](./maximum-differential-entropy-gaussian)

### 第八篇 高斯信道

- [x] [高斯信道模型与功率约束](./gaussian-channel-model)
- [x] [高斯信道容量的推导](./gaussian-channel-capacity)
- [x] [并联高斯信道与注水定理（Water-filling）](./parallel-gaussian-channels-waterfilling)
- [x] [有色噪声信道与注水公式](./colored-noise-channel)
- [x] [带限高斯信道与香农公式（Shannon's Capacity Formula）](./bandlimited-gaussian-channel-shannon-formula)

### 第九篇 率失真理论

- [x] [失真度量与量化问题](./distortion-measure-quantization)
- [x] [率失真函数 R(D) 的定义](./rate-distortion-function-definition)
- [x] [率失真函数的性质：单调性、凸性与连续性](./rate-distortion-properties)
- [x] [二元信源的率失真函数](./binary-source-rate-distortion)
- [x] [高斯信源的率失真函数](./gaussian-source-rate-distortion)
- [x] [高斯信源的逆注水（Reverse Water-filling）](./reverse-water-filling)
- [x] [率失真定理：可达性与逆定理](./rate-distortion-theorem)
- [x] [有约束情况下的信源信道分离](./source-channel-separation-constraints)

### 第十篇 科尔莫戈罗夫复杂度初步

- [x] [算法信息论与描述长度](./algorithmic-information-theory)
- [x] [科尔莫戈罗夫复杂度（Kolmogorov Complexity）的定义](./kolmogorov-complexity-definition)
- [x] [科尔莫戈罗夫复杂度的性质与不可计算性](./kolmogorov-complexity-properties)
- [x] [科尔莫戈罗夫复杂度与熵的联系](./kolmogorov-complexity-entropy)
- [x] [通用概率与 Solomonoff 先验](./universal-probability-solomonoff-prior)
- [x] [最小描述长度原理（MDL，Minimum Description Length）](./minimum-description-length)

### 第十一篇 信息论与统计、机器学习

- [x] [最大熵原理（Maximum Entropy Principle）](./maximum-entropy-principle)
- [x] [最大熵模型的推导：矩约束下的分布](./maximum-entropy-model-derivation)
- [x] [最大熵模型与逻辑回归的关系](./maximum-entropy-logistic-regression)
- [x] [Fisher 信息（Fisher Information）的定义与性质](./fisher-information)
- [x] [Cramér-Rao 不等式与信息不等式](./cramer-rao-inequality)
- [x] [熵幂不等式（Entropy Power Inequality）初步](./entropy-power-inequality)
- [x] [交叉熵（Cross-Entropy）与 KL 散度的联系](./cross-entropy-kl-divergence)
- [x] [交叉熵损失函数：从信息论看分类任务](./cross-entropy-loss-classification)

### 第1篇

- [x] [熵与信息度量（熵、联合熵、条件熵）](./entropy-and-information-measures)
- [x] [互信息与相对熵（KL 散度、数据处理不等式）](./mutual-information-and-relative-entropy)
- [x] [渐近等分割性（典型序列、信源编码定理）](./asymptotic-equipartition-property-and-source-coding)
- [x] [数据压缩（Kraft 不等式、Huffman 编码、算术编码）](./data-compression-kraft-huffman-arithmetic)
- [x] [信道容量（定义、对称信道计算）](./channel-capacity-definition-symmetric-channels)
- [x] [信道编码定理（随机编码、联合典型译码）](./channel-coding-theorem-random-coding-typical-decoding)
- [x] [高斯信道（功率约束、注水法）](./gaussian-channel-power-constraint-water-filling)
- [x] [率失真理论（有损压缩的理论极限）](./rate-distortion-theory-lossy-compression)
- [ ] 网络信息论初步（多址信道、广播信道）
- [ ] 信息论与统计学习（Fisher 信息、最大熵原理、MDL）
