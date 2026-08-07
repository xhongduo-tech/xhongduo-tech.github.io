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
- [ ] 条件互信息与互信息的链式法则
- [ ] 熵、条件熵与互信息之间的 Venn 图关系
- [ ] 相对熵的性质：非负性与吉布斯不等式
- [ ] 相对熵的凸性与链式法则

### 第二篇 基本不等式与数据处理

- [ ] 对数和不等式（Log Sum Inequality）
- [ ] Jensen 不等式及其在信息论中的应用
- [ ] 数据处理不等式（Data Processing Inequality）
- [ ] 马尔可夫链上的互信息单调性
- [ ] 充分统计量（Sufficient Statistic）与数据处理不等式
- [ ] Fano 不等式及其误差概率含义
- [ ] 均匀分布与最大熵
- [ ] 条件作用使熵减小（Conditioning Reduces Entropy）

### 第三篇 渐近均分性（AEP）

- [ ] 弱大数定律回顾与依概率收敛
- [ ] 渐近均分性定理（Asymptotic Equipartition Property）
- [ ] 典型集（Typical Set）的定义与性质
- [ ] 高概率集与典型集的关系
- [ ] AEP 与数据压缩的联系
- [ ] 联合典型性与联合典型序列
- [ ] 联合典型序列的数量估计

### 第四篇 信源编码与数据压缩

- [ ] 信源编码问题与码的分类（唯一可译码、即时码、前缀码）
- [ ] Kraft 不等式（Kraft's Inequality）
- [ ] 最优码长度与码长的下界
- [ ] 香农码（Shannon Code）及其构造
- [ ] 霍夫曼编码（Huffman Coding）的构造与最优性证明
- [ ] 算术编码（Arithmetic Coding）的原理与实现
- [ ] Lempel-Ziv 通用编码（LZ77/LZ78）初步
- [ ] 信源编码定理：平均码长的熵界

### 第五篇 信道容量

- [ ] 离散无记忆信道（DMC）模型
- [ ] 信道容量的定义与直观含义
- [ ] 无噪二元信道与有噪信道的例子
- [ ] 二进制对称信道（Binary Symmetric Channel）的容量
- [ ] 二进制删除信道（Binary Erasure Channel）的容量
- [ ] 对称信道及其容量的简化计算
- [ ] 信道容量的性质与求解思路

### 第六篇 信道编码定理初步

- [ ] 联合典型序列与译码方法
- [ ] 随机编码思想与典型序列译码
- [ ] 信道编码定理：可达性的证明
- [ ] 信道编码定理逆定理（Fano 不等式的应用）
- [ ] 信源信道分离定理（Source-Channel Separation）
- [ ] 反馈信道与反馈不能增大容量的结论

### 第七篇 微分熵

- [ ] 微分熵（Differential Entropy）的定义
- [ ] 均匀分布、指数分布与高斯分布的微分熵
- [ ] 微分熵与离散熵的区别：可为负值的原因
- [ ] 联合微分熵与条件微分熵
- [ ] 相对熵与互信息的连续形式
- [ ] 微分熵的性质：变换下的变化规律
- [ ] 最大微分熵定理：高斯分布的熵最大

### 第八篇 高斯信道

- [ ] 高斯信道模型与功率约束
- [ ] 高斯信道容量的推导
- [ ] 并联高斯信道与注水定理（Water-filling）
- [ ] 有色噪声信道与注水公式
- [ ] 带限高斯信道与香农公式（Shannon's Capacity Formula）

### 第九篇 率失真理论

- [ ] 失真度量与量化问题
- [ ] 率失真函数 R(D) 的定义
- [ ] 率失真函数的性质：单调性、凸性与连续性
- [ ] 二元信源的率失真函数
- [ ] 高斯信源的率失真函数
- [ ] 高斯信源的逆注水（Reverse Water-filling）
- [ ] 率失真定理：可达性与逆定理
- [ ] 有约束情况下的信源信道分离

### 第十篇 科尔莫戈罗夫复杂度初步

- [ ] 算法信息论与描述长度
- [ ] 科尔莫戈罗夫复杂度（Kolmogorov Complexity）的定义
- [ ] 科尔莫戈罗夫复杂度的性质与不可计算性
- [ ] 科尔莫戈罗夫复杂度与熵的联系
- [ ] 通用概率与 Solomonoff 先验
- [ ] 最小描述长度原理（MDL，Minimum Description Length）

### 第十一篇 信息论与统计、机器学习

- [ ] 最大熵原理（Maximum Entropy Principle）
- [ ] 最大熵模型的推导：矩约束下的分布
- [ ] 最大熵模型与逻辑回归的关系
- [ ] Fisher 信息（Fisher Information）的定义与性质
- [ ] Cramér-Rao 不等式与信息不等式
- [ ] 熵幂不等式（Entropy Power Inequality）初步
- [ ] 交叉熵（Cross-Entropy）与 KL 散度的联系
- [ ] 交叉熵损失函数：从信息论看分类任务

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
