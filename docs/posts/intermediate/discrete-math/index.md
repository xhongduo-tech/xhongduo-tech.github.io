---
pageClass: plain-doc
---

# 离散数学

学完一个学科，就是写完该学科经典教材对应的全部博文。本分类对标 Kenneth H. Rosen《离散数学及其应用》，覆盖从逻辑基础到计算模型的完整知识体系，每篇博文对应教材的一节。

## 主题规划

<ProgressGrid cat="intermediate/discrete-math" />


### 第一篇 逻辑与证明

- [x] [命题逻辑：命题、联结词与真值表](./propositional-logic)
- [x] [命题等价式：逻辑等价与德摩根律](./logical-equivalence)
- [x] [谓词与量词：全称量词、存在量词与论域](./predicates-quantifiers)
- [x] [嵌套量词：量词的顺序与否定](./nested-quantifiers)
- [x] [推理规则：有效论证与常见谬误](./rules-of-inference)
- [x] [命题逻辑的推理规则：假言推理、拒取式与归结](./inference-forms-resolution)
- [x] [证明导论：定理、公理与猜想](./proof-introduction)
- [x] [直接证明、反证法与归谬法](./direct-proof-contradiction)
- [x] [证明方法综述：分情形证明、存在性证明与唯一性证明](./proof-methods-survey)

### 第二篇 集合

- [x] [集合的基本概念：属于、子集与空集](./set-basics)
- [x] [集合运算：并、交、补、差与对称差](./set-operations)
- [x] [集合恒等式与文氏图](./set-identities-venn-diagrams)
- [x] [幂集与笛卡尔积](./power-set-cartesian-product)
- [x] [集合的划分与带计算机表示的集合运算](./set-partitions-computer-representation)

### 第三篇 函数

- [x] [函数的定义：定义域、值域与像](./functions-definition-domain-codomain)
- [x] [单射、满射与双射（一对一、映上与一一对应）](./injective-surjective-bijective)
- [x] [逆函数与函数复合](./inverse-functions-composition)
- [x] [取整函数：下取整（floor）与上取整（ceiling）](./floor-ceiling)
- [x] [基数与可数性：可数集与康托尔对角线法](./cardinality-countability)

### 第四篇 序列与求和

- [x] [序列：等差序列、等比序列与递推定义的序列](./sequences-arithmetic-geometric-recurrence)
- [x] [求和记号与常用求和公式](./summation-notation)
- [x] [双重求和与伸缩求和（telescoping sum）](./double-summation-telescoping)

### 第五篇 算法与复杂度

- [x] [算法的概念与伪代码描述](./algorithms-concept-pseudocode)
- [x] [搜索算法：线性搜索与二分搜索](./searching-linear-binary)
- [x] [排序算法：冒泡排序与插入排序](./sorting-bubble-insertion)
- [x] [贪心算法及其正确性论证](./greedy-algorithms-correctness)
- [x] [算法增长：大 O 记号](./big-o-notation)
- [x] [大 Ω 与大 Θ 记号](./big-omega-theta-notation)
- [x] [算法复杂度：时间复杂度与空间复杂度的分析](./algorithm-complexity-analysis)

### 第六篇 数论与密码学

- [x] [整除性：整除、因子与带余除法](./divisibility-factors-division-algorithm)
- [x] [模算术：同余及其运算性质](./modular-arithmetic-congruence)
- [x] [整数的表示：二进制、十六进制与进制转换](./integer-representation-base-conversion)
- [x] [素数：定义、分布与素性判定](./primes-distribution-primality-testing)
- [x] [最大公约数与欧几里得算法](./gcd-euclidean-algorithm)
- [x] [算术基本定理与唯一分解](./fundamental-theorem-of-arithmetic)
- [x] [扩展欧几里得算法与模逆元](./extended-euclidean-algorithm-modular-inverse)
- [x] [线性同余方程与中国剩余定理](./linear-congruences-chinese-remainder-theorem)
- [x] [费马小定理与欧拉定理](./fermats-little-theorem-eulers-theorem)
- [x] [密码学导论：古典密码与凯撒密码](./cryptography-classical-ciphers-caesar)
- [x] [公钥密码学与 RSA 加密算法](./public-key-cryptography-rsa)

### 第七篇 归纳与递归

- [x] [数学归纳法：原理与基本证明](./mathematical-induction)
- [x] [强归纳法与良序原理](./strong-induction-well-ordering)
- [x] [递归定义：递归定义的函数与序列](./recursive-definitions-sequences)
- [x] [递归定义的集合与结构归纳法](./recursively-defined-sets-structural-induction)
- [x] [递归算法：设计与正确性证明](./recursive-algorithms-correctness)
- [x] [程序正确性：前置断言、后置断言与循环不变量](./program-correctness-loop-invariants)

### 第八篇 计数

- [x] [基本计数原理：乘法原理与加法原理](./counting-product-sum-rules)
- [x] [减法原理（补集计数）与除法原理](./subtraction-division-counting-principles)
- [x] [鸽笼原理及其推广](./pigeonhole-principle)
- [x] [排列：无重复与有重复的排列](./permutations)
- [x] [组合：二项式系数与组合数](./combinations-binomial-coefficients)
- [x] [二项式定理及其推论](./binomial-theorem)
- [x] [帕斯卡恒等式与范德蒙德恒等式](./pascals-identity-vandermondes-identity)
- [x] [有重复的排列与组合：重集计数](./permutations-combinations-with-repetition)
- [x] [排列与组合中的物体分配问题](./distributing-objects)
- [x] [容斥原理：两个与三个集合的情形](./inclusion-exclusion-two-three-sets)
- [x] [容斥原理的一般形式与错排问题](./inclusion-exclusion-general-derangements)

### 第九篇 高级计数技术

- [x] [递推关系：建模与应用](./recurrence-relations-modeling)
- [x] [线性齐次递推关系：常系数情形与特征方程](./linear-homogeneous-recurrences)
- [x] [特征方程有重根的情形](./repeated-roots-characteristic-equation)
- [x] [线性非齐次递推关系的求解](./linear-nonhomogeneous-recurrences)
- [x] [分治递推关系与主定理](./divide-conquer-recurrences-master-theorem)
- [x] [生成函数：定义与基本运算](./generating-functions-basics)
- [x] [用生成函数求解递推关系](./generating-functions-solving-recurrences)
- [x] [用生成函数证明恒等式](./generating-functions-proving-identities)
- [x] [广义二项式定理与计数应用](./generalized-binomial-theorem-counting)

### 第十篇 关系

- [x] [二元关系：定义与 n 元关系](./binary-relations-n-ary)
- [x] [关系的性质：自反、对称、反对称与传递](./relation-properties-reflexive-symmetric-antisymmetric-transitive)
- [x] [关系的运算：复合与逆关系](./relation-composition-inverse)
- [x] [关系的表示：矩阵与有向图](./relation-representation-matrix-digraph)
- [x] [关系的闭包：自反闭包、对称闭包与传递闭包](./relation-closures)
- [x] [沃舍尔算法（Warshall's algorithm）](./warshalls-algorithm)
- [x] [等价关系与划分](./equivalence-relations-partitions)
- [x] [偏序关系与哈斯图](./partial-order-hasse-diagrams)
- [x] [极大元、极小元、上界与下界](./maximal-minimal-elements-bounds)
- [x] [格（lattice）与拓扑排序](./lattices-topological-sorting)

### 第十一篇 图

- [x] [图的基本概念：图、有向图与多重图](./graphs-basics-directed-multigraphs)
- [x] [图的术语：度、度序列与握手定理](./graph-terminology-degrees-handshake-lemma)
- [x] [特殊图：完全图、圈图、轮图与二分图](./special-graphs-complete-cycles-wheels-bipartite)
- [x] [图的运算与子图](./graph-operations-subgraphs)
- [x] [图的表示：邻接矩阵与关联矩阵](./graph-representation-adjacency-incidence-matrices)
- [x] [图的同构判定](./graph-isomorphism)
- [x] [连通性：通路、回路与连通分量](./connectivity-paths-cycles-components)
- [x] [欧拉通路与欧拉回路](./euler-paths-circuits)
- [x] [哈密顿通路与哈密顿回路](./hamiltonian-paths-circuits)
- [x] [最短通路问题：迪克斯特拉算法（Dijkstra's algorithm）](./dijkstras-shortest-path)
- [x] [旅行商问题](./traveling-salesman-problem)
- [x] [平面图：欧拉公式与库拉托夫斯基定理](./planar-graphs-eulers-formula-kuratowskis-theorem)
- [x] [图着色：四色定理与色数](./graph-coloring-four-color-chromatic-number)

### 第十二篇 树

- [x] [树的基本概念：根树、有序根树与树的性质](./trees-basics-rooted-ordered)
- [x] [m 叉树与树的计数性质](./m-ary-trees-counting)
- [x] [树的应用：二叉搜索树与决策树](./tree-applications-binary-search-decision)
- [x] [前缀码与赫夫曼编码（Huffman coding）](./prefix-codes-huffman-coding)
- [x] [树的遍历：前序、中序与后序遍历](./tree-traversal-preorder-inorder-postorder)
- [x] [中缀、前缀与后缀记法](./infix-prefix-postfix-notation)
- [x] [生成树：深度优先搜索与广度优先搜索](./spanning-trees-dfs-bfs)
- [x] [回溯法及其应用](./backtracking)
- [x] [最小生成树：普里姆算法（Prim's algorithm）与克鲁斯卡尔算法（Kruskal's algorithm）](./minimum-spanning-trees-prim-kruskal)

### 第十三篇 布尔代数

- [x] [布尔函数：布尔运算与布尔表达式](./boolean-functions)
- [x] [布尔代数的恒等式与对偶原理](./boolean-algebra-identities-duality)
- [x] [布尔函数的表示：积之和展开（析取范式）](./boolean-function-representation-sum-of-products)
- [x] [函数完备性：与非门与或非门](./functional-completeness-nand-nor)
- [x] [逻辑门电路：组合电路的设计](./logic-gates-combinational-circuits)
- [x] [卡诺图（Karnaugh map）化简](./karnaugh-maps)
- [x] [奎因-麦克拉斯基方法（Quine–McCluskey method）](./quine-mccluskey-method)

### 第十四篇 计算模型

- [x] [语言与文法：短语结构文法与推导](./languages-grammars-derivations)
- [x] [文法的类型：乔姆斯基谱系](./grammar-types-chomsky-hierarchy)
- [x] [巴科斯-诺尔范式（BNF）](./backus-naur-form)
- [x] [有限状态机：带输出的有限状态机（米利机与摩尔机）](./finite-state-machines-output-mealy-moore)
- [x] [不带输出的有限状态机与语言识别](./finite-state-machines-language-recognition)
- [x] [有限状态自动机与正则语言](./finite-state-automata-regular-languages)
- [x] [非确定性有限状态自动机与 Kleene 定理](./nondeterministic-finite-state-automata-kleene-theorem)
- [x] [图灵机：定义与计算](./turing-machines)
- [x] [可计算性与停机问题](./computability-halting-problem)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [ ] 逻辑与证明（命题逻辑、谓词逻辑、证明方法）
- [ ] 集合与关系（等价关系、偏序关系）
- [ ] 函数与基数（可数集、鸽巢原理）
- [ ] 组合计数（排列组合、容斥原理）
- [ ] 递推关系与生成函数（线性递推、母函数方法）
- [ ] 数论初步与密码应用（同余、RSA）
- [ ] 图论基础（路径、连通性、欧拉回路与哈密顿回路）
- [ ] 树（生成树、二叉树、遍历）
- [ ] 代数结构（群、环、格与布尔代数）
- [ ] 形式语言与自动机初步（正则语言、有限自动机）
