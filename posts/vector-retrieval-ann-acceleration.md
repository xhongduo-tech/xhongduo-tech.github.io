## 1) 一句话核心定义

ANN（Approximate Nearest Neighbor）加速，是在向量库里不用全量扫描，而是先用 KD-Tree、HNSW 或 PQ/OPQ 生成候选集，再做有限重排；目标是用可接受的 Recall@k 换更低的延迟和内存占用。

---

## 2) 面向新手的直观解释

向量检索的任务，是在一堆向量里找和查询向量最接近的几个。全量比较最准确，但数据一到百万级，计算量和内存占用都会上来。

ANN 的思路是先“缩小范围”再“精细比较”。KD-Tree 擅长低维切分，HNSW 擅长图上跳转，PQ/OPQ 擅长把向量压缩后再搜。

---

## 3) 关键公式/机制

设查询向量为 $q \in \mathbb R^d$，候选库为 $X=\{x_i\}_{i=1}^n$，精确最近邻是：

$$
x^*=\arg\min_{x_i\in X}\operatorname{dist}(q,x_i)
$$

ANN 则先构造候选集 $C(q)\subset X$，再在子集上求近邻：

$$
\hat x=\arg\min_{x\in C(q)}\operatorname{dist}(q,x)
$$

Recall@k 直接看前 $k$ 个结果里找回了多少真近邻：

$$
\mathrm{Recall@}k=\frac{|T_k\cap \hat T_k|}{k}
$$

三类主流机制可以概括为：

| 方法 | 核心机制 | 典型约束 |
| --- | --- | --- |
| KD-Tree | 按坐标轴递归切分空间，剪枝掉不可能的分支 | 低维更有效，高维会退化 |
| HNSW | 分层小世界图 + 贪心跳转 + `ef_search` 候选扩展 | 读多写少，内存占用较高 |
| PQ/OPQ | 把向量分段量化成码字，查询时做近似距离累加 | 省内存，但要接受量化误差 |

PQ 的编码可以写成：

$$
x\approx [e^{(1)}_{c_1},e^{(2)}_{c_2},\dots,e^{(m)}_{c_m}]
$$

OPQ 先做旋转再量化：

$$
x' = R x
$$

其中 $R$ 是正交旋转矩阵，作用是把信息分布“拌匀”，减小分段量化误差。

---

## 4) 一个最小数值例子

假设有 $1{,}000{,}000$ 条 $768$ 维向量，float32 原始存储约为：

$$
1{,}000{,}000 \times 768 \times 4 \approx 3.07\ \mathrm{GB}
$$

如果改成 PQ 编码，按 64 字节/条估算，向量本身只剩：

$$
1{,}000{,}000 \times 64 \approx 64\ \mathrm{MB}
$$

如果真实 top-10 里你找回 9 条，Recall@10 = 0.9；压缩或近似后只找回 8 条，Recall@10 = 0.8。这里少掉的 0.1，往往就是延迟和内存换来的代价。

---

## 5) 一个真实工程场景

企业知识库或推荐召回常见做法是：`query -> embedding -> metadata filter -> ANN top-k -> reranker -> LLM/排序器`。

当文档块到百万级，热数据可以放 HNSW，冷数据放 IVF-PQ 或 PQ/OPQ；前者保高召回，后者保内存。真正上线时，常常不是单选一种索引，而是按数据热度和延迟预算做混合部署。

---

## 6) 常见坑与规避

| 坑 | 表现 | 规避 |
| --- | --- | --- |
| KD-Tree 硬上高维 embedding | 剪枝失效，速度接近暴搜 | 维度很高时优先考虑 HNSW 或 PQ |
| 只看延迟，不看 Recall@k | 接口快，但结果不准 | 同时监控 Recall@k、P95、内存 |
| PQ 训练样本太少 | 码本不稳，召回波动大 | 用代表性训练集，定期重训 |
| HNSW 频繁删改 | 图质量下降，维护成本高 | 批量更新、逻辑删除、定期重建 |
| 只拿 PQ top-k 直接上线 | 量化误差直接传给结果 | 先 overfetch，再用原始向量重排 |

---

## 7) 参考来源

1. [Malkov, Yashunin: *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*](https://pubmed.ncbi.nlm.nih.gov/30602420/)  
2. [scikit-learn 文档：Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)  
3. [scikit-learn 文档：KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)  
4. [FAISS 文档：ProductQuantizer](https://faiss.ai/cpp_api/struct/structfaiss_1_1ProductQuantizer.html)  
5. [FAISS 文档：OPQMatrix](https://faiss.ai/cpp_api/struct/structfaiss_1_1OPQMatrix.html)  

