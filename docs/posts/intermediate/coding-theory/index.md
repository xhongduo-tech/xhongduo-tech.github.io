---
pageClass: plain-doc
---

# 编码理论（纠错编码）

编码理论（纠错编码）研究在噪声信道上可靠传输信息的数学方法：通过向信息中添加受控冗余，使传输过程中产生的错误可以被检测与纠正。它以有限域、线性代数与代数几何为工具，贯通从 Hamming 码、BCH 与 Reed-Solomon 码到代数几何码的完整谱系，是数字通信、存储系统与深空探测背后不可或缺的应用代数分支。

## 对标教材

- J. H. van Lint, "Introduction to Coding Theory" (3rd ed., Springer GTM 86, 1999)
- Ron Roth, "Introduction to Coding Theory" (Cambridge University Press, 2006)
- F. J. MacWilliams and N. J. A. Sloane, "The Theory of Error-Correcting Codes" (North-Holland, 1977)

## 主题规划

<ProgressGrid cat="intermediate/coding-theory" />

### 第1篇 基础：信道、有限域与线性码

- [x] [编码理论的通信模型与纠错原理](./communication-model)
- [x] [有限域的构造、结构与计算](./finite-fields)
- [x] [线性码：生成矩阵、校验矩阵与最小距离](./linear-codes)
- [x] [线性码解码：伴随式、标准阵列与 Hamming 码](./syndrome-decoding-and-hamming-codes)
- [x] [对偶码与 MacWilliams 恒等式](./dual-codes-macwilliams)
- [x] [码参数界：Hamming、Singleton 与 Gilbert-Varshamov 界](./code-bounds)

### 第2篇 循环码、BCH 码与 Reed-Solomon 码

- [x] [循环码：多项式表示、生成多项式与校验多项式](./cyclic-codes)
- [x] [循环码的幂等元与 Mattson-Solomon 多项式](./idempotents-and-mattson-solomon)
- [x] [BCH 码：根集、BCH 界与纠错能力](./bch-codes)
- [x] [Reed-Solomon 码与广义 RS 码的编码](./reed-solomon-codes)
- [x] [RS/BCH 码的伴随式与关键方程解码](./syndrome-and-key-equation-decoding)
- [x] [Berlekamp-Massey 算法与欧几里得解码](./berlekamp-massey-and-euclidean-decoding)
- [x] [MDS 码、扩展与 MDS 猜想](./mds-codes)

### 第3篇 组合构造、特殊码与先进理论

- [x] [完美码、Golay 码与 Lloyd 定理](./perfect-codes-golay-lloyd)
- [x] [Reed-Muller 码与有限几何码](./reed-muller-codes)
- [x] [Hadamard 码、Kerdock 码与非线性码](./hadamard-kerdock-nonlinear-codes)
- [x] [Z₄ 上的码：Galois 环与格理论联系](./z4-codes-galois-rings)
- [x] [代数几何（Goppa）码与 Riemann-Roch 定理](./algebraic-geometry-goppa-codes)
- [x] [列表解码：Sudan 与 Guruswami-Sudan 算法](./list-decoding-sudan-guruswami)
- [x] [级联码、Justesen 码与渐近好码](./concatenated-justesen-asymptotic-codes)

### 第4篇 卷积码、图码与现代应用

- [x] [卷积码：生成多项式、网格图与 Viterbi 解码](./convolutional-codes-viterbi)
- [x] [网格码、有限状态机与软输出解码](./trellis-codes-soft-output)
- [x] [图码：扩展图码、Ramanujan 图与迭代解码](./graph-codes-expander-ramanujan)
- [x] [LDPC 码与消息传递解码](./ldpc-message-passing)
- [x] [纠错编码的应用：深空通信、数字存储与无线系统](./applications-deep-space-storage-wireless)

### 第5篇

- [ ] 通信信道与编码基本概念（码率、最小距离）
- [ ] 线性分组码（生成矩阵、校验矩阵、伴随式译码）
- [ ] 码的界（汉明界、Singleton 界、完美码）
- [ ] 循环码（多项式表示、生成多项式）
- [ ] BCH 码与 Reed-Solomon 码（代数译码）
- [ ] 卷积码（状态图、维特比译码）
- [ ] 编码的代数方法（有限域、代数几何码简介）
- [ ] 现代迭代译码（LDPC 码、Turbo 码、极化码）
- [ ] 编码与密码学（McEliece 体制）
- [ ] 应用前沿（存储系统、深空通信、网络编码）
