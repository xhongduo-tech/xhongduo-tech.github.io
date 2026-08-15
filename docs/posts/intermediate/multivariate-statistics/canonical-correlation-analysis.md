---
title: 典型相关分析
date: 2026-08-07
---

# 典型相关分析

<div class="epigraph">
<p>两组变量之间的整体关系，不该靠 p 乘 q 个散点图来拼凑，而该靠少数几对「最有代表性的方向」来讲。</p>
<footer>—— 哈罗德·霍特林（Harold Hotelling）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Anderson《An Introduction to Multivariate Statistical Analysis》Ch.12 · Johnson & Wichern Ch.10 ｜ 2026-08-07</p>
</div>

## 为什么需要两组变量的相关

到目前为止我们研究过「一个变量与一个变量」的相关、「一组变量内部」的结构（PCA/FA）。但很多问题涉及**两组变量之间的关系**：一组生理指标与一组心理量表、一组投入变量与一组产出变量、学生的几门课成绩与几项能力测试。最朴素的做法是把两组变量两两算相关系数——$p \times q$ 个数字，看得人眼花，还看不出「整体关系」的强度。**典型相关分析（canonical correlation analysis, CCA）**用一句话解决：**在两组变量里各找一个线性组合，使它们的相关系数最大**——于是整组关系被压缩成少数几对「最能互相说明」的方向。<span class="marginnote">CCA 由霍特林 1936 年提出，与 PCA、判别分析同属「谱方法家族」：都是把一个最优化问题化成特征值问题。可以说 PCA 处理一组变量的内部结构，CCA 处理两组变量之间的结构。</span>

## 1 第一对典型变量：最大化相关系数

设第一组变量 $\mathbf{X}$（$p$ 维）、第二组 $\mathbf{Y}$（$q$ 维），联合协方差矩阵分块为

$$
\operatorname{Cov}\begin{pmatrix}\mathbf{X} \\ \mathbf{Y}\end{pmatrix} =
\begin{pmatrix}\boldsymbol{\Sigma}_{XX} & \boldsymbol{\Sigma}_{XY} \\ \boldsymbol{\Sigma}_{YX} & \boldsymbol{\Sigma}_{YY}\end{pmatrix}
$$

要找单位向量的线性组合 $U = \mathbf{a}'\mathbf{X}$、$V = \mathbf{b}'\mathbf{Y}$ 使相关系数最大：

$$
\rho = \operatorname{Corr}(U, V) = \frac{\mathbf{a}'\boldsymbol{\Sigma}_{XY}\mathbf{b}}{\sqrt{\mathbf{a}'\boldsymbol{\Sigma}_{XX}\mathbf{a}} \sqrt{\mathbf{b}'\boldsymbol{\Sigma}_{YY}\mathbf{b}}} \ \to \ \max
$$

这比 PCA 的目标复杂：分子是两组之间的交叉协方差，分母各自归一化。但解的结构惊人地干净——$U_1 = \mathbf{a}_1'\mathbf{X}$、$V_1 = \mathbf{b}_1'\mathbf{Y}$ 称为**第一对典型变量（canonical variables）**，它们的相关系数 $\rho_1$ 叫**第一典型相关系数**。<span class="marginnote">解读的直觉：$U_1$ 是「第一组变量里最能被第二组解释的那个综合指标」，$V_1$ 同理；$\rho_1$ 是这两套综合指标之间的最大可能相关。若 $\rho_1 \approx 0.9$，说明两组变量之间存在很强的线性关系。</span>

## 2 解的形式：两个特征方程

把优化问题求导置零，得到一对互锁的特征方程。若假设两组合并后协方差矩阵正定，令

$$
\mathbf{A} = \boldsymbol{\Sigma}_{XX}^{-1}\boldsymbol{\Sigma}_{XY}\boldsymbol{\Sigma}_{YY}^{-1}\boldsymbol{\Sigma}_{YX}, \qquad
\mathbf{B} = \boldsymbol{\Sigma}_{YY}^{-1}\boldsymbol{\Sigma}_{YX}\boldsymbol{\Sigma}_{XX}^{-1}\boldsymbol{\Sigma}_{XY}
$$

则 $\mathbf{A}$ 与 $\mathbf{B}$ 的非零特征值相同（记 $\rho_1^2 \geq \rho_2^2 \geq \cdots \geq \rho_m^2$），其中 $m = \min(p, q)$；$\mathbf{a}_k$ 是 $\mathbf{A}$ 的第 $k$ 大特征值对应的特征向量，$\mathbf{b}_k$ 是 $\mathbf{B}$ 的对应特征向量。**第 $k$ 个典型相关系数就是 $\sqrt{\lambda_k}$，即 $\mathbf{A}$（或 $\mathbf{B}$）第 $k$ 大特征值的平方根**。<span class="marginnote">$\mathbf{A} = \boldsymbol{\Sigma}_{XX}^{-1}\boldsymbol{\Sigma}_{XY}\boldsymbol{\Sigma}_{YY}^{-1}\boldsymbol{\Sigma}_{YX}$ 这个「之字形」矩阵值得记住：它把「组内精化（$\boldsymbol{\Sigma}_{XX}^{-1}$、$\boldsymbol{\Sigma}_{YY}^{-1}$）与组间联系（$\boldsymbol{\Sigma}_{XY}$、$\boldsymbol{\Sigma}_{YX}$）」缝在一起。Fisher 判别里的 $\mathbf{W}^{-1}\mathbf{B}$ 是它的近亲。</span>

特征值 $\lambda_k$ 的平方根就是相关系数，且**平方后 $0 \leq \lambda_k \leq 1$**——这是典型相关的另一层便利：$\rho_k^2$ 直接读作「第 $k$ 对典型变量共享的方差比例」。

### 与回归、判别、PCA 的对照

| 方法 | 研究对象 | 优化的量 |
| --- | --- | --- |
| PCA | 一组变量内部 | 投影方差最大 |
| Fisher 判别 | 分组 vs 变量 | 组间/组内方差比 |
| 多元回归 | 一组变量预测另一个（组） | 预测误差最小 |
| CCA | 两组变量之间 | 投影相关最大 |

这张表揭示了谱方法家族的统一性：**它们都在回答「往哪个方向投影」，区别只是目标函数里放的是哪一散布矩阵**。记住这张表，后面读到任何「最大化某比值的投影」方法都不会陌生。

## 3 后续典型变量与样本情形

第一对抓住最强的相关后，可以继续找**第二对**：$U_2 = \mathbf{a}_2'\mathbf{X}$、$V_2 = \mathbf{b}_2'\mathbf{Y}$，要求与 $U_1$、$V_1$ 都不相关（组内正交），且相关系数最大。解出来就是第二个特征值对应的特征向量。**典型变量成对出现、依次递减**，一共最多 $\min(p,q)$ 对，像 PCA 一样给出一串递减的「主相关」<span class="marginnote">正交约束的几何意义：第一对抓走了两组关系里最粗的一条，第二对在「剩下的」方向里再找最强的——就像先量椭圆长轴、再量短轴。典型相关排序与 PCA 的「轴按方差排序」是同一套叙事。</span>

样本情形把总体协方差换成样本协方差：$\boldsymbol{\Sigma}_{XX} \to \mathbf{S}_{XX}$ 等，得到样本典型相关系数 $\hat{\rho}_k$。检验「第 $k$ 对及其后是否显著」用 Bartlett 近似：$H_0$:「前 $k-1$ 对之后没有更多相关」时，

$$
-\Bigl(n - \frac{p+q+1}{2}\Bigr) \sum_{j=k}^{m} \ln(1 - \hat{\rho}_j^2) \ \sim \ \chi^2_{(p-k+1)(q-k+1)}
$$

这给出「该保留几对典型变量」的推断依据——与 PCA 选主成分个数不同，CCA 有正式的显著性检验。

## 4 载荷与冗余指数：怎么解读结果

算完典型相关系数只是开始，真正的难点是**解释**。三个工具依次上场：

**典型载荷（canonical loadings）**：$U_1$ 与第一组各变量的相关系数向量 $\operatorname{Corr}(\mathbf{X}, U_1)$、$V_1$ 与第二组各变量的相关系数 $\operatorname{Corr}(\mathbf{Y}, V_1)$。它回答「$U_1$ 主要由哪些变量撑起来」——与 PCA 载荷的用法完全相同。<span class="marginnote">注意别把「系数 $\mathbf{a}$」和「载荷 $\operatorname{Corr}(\mathbf{X},U_1)$」搞混：系数受组内共线性影响、可以互相抵消，载荷是单个变量与典型变量的边际相关、更直观。解释时优先看载荷。</span>

**冗余指数（redundancy index）**：典型相关只告诉我们「两对综合变量有多相关」，但没回答「$U_1$ 解释了多少第一组变量的方差、又经由 $V_1$ 传递了多少」。冗余指数分两步：先看 $\rho_1^2$（$U_1$ 与 $V_1$ 共享方差），再看 $V_1$ 对第一组变量的平均解释比例。两者的乘积度量「第二组通过第一对典型变量能解释第一组方差的比例」——这是把「相关性」翻译成「可解释方差」的关键指标。

**留多少对？** 统计上用第 3 节的 Bartlett 检验；实践上习惯看**累计冗余指数**与碎石图式的 $\hat{\rho}_k^2$ 序列。通常前一两对就吸收了绝大部分关系，后面的对只是细枝末节。<span class="marginnote">一个典型的应用是消费者研究：第一组是产品属性评分，第二组是购买意愿量表；第一典型变量可能把「功能评分」与「购买意向」捆在一起，$\rho_1 = 0.85$ 就说明「感知功能与购买意愿强相关」——一次分析顶 $p \times q$ 张散点图。</span>

## 5 实战流程与易错点

完整跑一次 CCA 的标准流程：

1. **预处理**：两组变量各自标准化（量纲不同时尤其必要），检查缺失与离群。
2. **估计**：算分块样本协方差矩阵，解 $\mathbf{S}_{XX}^{-1}\mathbf{S}_{XY}\mathbf{S}_{YY}^{-1}\mathbf{S}_{YX}$ 的特征分解。
3. **定对数**：用 Bartlett 检验与 $\hat{\rho}_k^2$ 序列决定保留几对。
4. **解释**：看典型载荷给每对起名字，算冗余指数量化「解释了多少方差」。
5. **报告**：典型相关系数、载荷、冗余指数三者齐备才算讲完一个故事。

几个反复出现的坑：

**样本量要够**：估计需要 $\mathbf{S}$ 正定，粗略要求 $n > p + q$。$n$ 接近 $p+q$ 时特征值剧烈波动、结果不可信——这是高维专题里「$p \approx n$ 危机」在 CCA 里的具体表现。
**组内共线性会捣乱**：两组变量内部高度相关时，$\boldsymbol{\Sigma}_{XX}^{-1}$、$\boldsymbol{\Sigma}_{YY}^{-1}$ 不稳定，典型系数跳跃。先用 PCA 压缩组内结构，再做 CCA，是常见对策。
**别过度解读小相关**：大样本下即使典型相关系数只有 0.3 也可能显著（Bartlett 检验通过），但冗余指数可能低得可怜——**显著 ≠ 有实际意义**，冗余指数才是「值不值得解释」的裁判。<span class="marginnote">现代变体值得一提：<strong>稀疏 CCA</strong> 在系数上施加稀疏约束，让典型变量只由少数变量构成、可解释性大增；<strong>深度 CCA</strong> 用神经网络学非线性投影。它们的骨架仍是这里的两步——最大化相关、特征分解。</span>

## 6 公式解析：为什么最优系数来自特征方程

把「最大化相关系数」一步步翻译成特征方程，是理解 CCA 的关键一步：

- **第一步，消去分母**：目标 $\rho = \mathbf{a}'\boldsymbol{\Sigma}_{XY}\mathbf{b}/\sqrt{\mathbf{a}'\boldsymbol{\Sigma}_{XX}\mathbf{a}\sqrt{\mathbf{b}'\boldsymbol{\Sigma}_{YY}\mathbf{b}}}$。注意到 $\rho$ 在 $\mathbf{a} \to c\mathbf{a}$、$\mathbf{b} \to d\mathbf{b}$ 下不变，可加约束 $\mathbf{a}'\boldsymbol{\Sigma}_{XX}\mathbf{a} = 1$、$\mathbf{b}'\boldsymbol{\Sigma}_{YY}\mathbf{b} = 1$。
- **第二步，拉格朗日**：$\mathcal{L} = \mathbf{a}'\boldsymbol{\Sigma}_{XY}\mathbf{b} - \frac{\lambda}{2}(\mathbf{a}'\boldsymbol{\Sigma}_{XX}\mathbf{a} - 1) - \frac{\mu}{2}(\mathbf{b}'\boldsymbol{\Sigma}_{YY}\mathbf{b} - 1)$。
- **第三步，对 $\mathbf{a}$、$\mathbf{b}$ 分别求导置零**：得 $\boldsymbol{\Sigma}_{XY}\mathbf{b} = \lambda\boldsymbol{\Sigma}_{XX}\mathbf{a}$ 与 $\boldsymbol{\Sigma}_{YX}\mathbf{a} = \mu\boldsymbol{\Sigma}_{YY}\mathbf{b}$。两式联立消元，把 $\mathbf{b}$ 用 $\mathbf{a}$ 表出再代入，就得到 $\mathbf{A}\mathbf{a} = \lambda^2\mathbf{a}$——**最大化问题化为广义特征值问题**。
- **第四步，读解**：$\lambda$ 取正值时就是典型相关系数 $\rho_1$；$\mathbf{a}_1$、$\mathbf{b}_1$ 即第一对典型变量的系数。后续对依次取下一个特征向量。

**核心结论：CCA 的最优系数 = 矩阵 $\boldsymbol{\Sigma}_{XX}^{-1}\boldsymbol{\Sigma}_{XY}\boldsymbol{\Sigma}_{YY}^{-1}\boldsymbol{\Sigma}_{YX}$ 的特征向量，典型相关系数 = 特征值的平方根**。与 Fisher 判别、PCA 同一套数学，只是目标矩阵不同。

## 7 小结

- **典型相关分析**研究两组变量的整体关系：各找一个线性组合使相关系数最大。
- 最优系数来自特征方程 $\mathbf{A}\mathbf{a} = \rho^2\mathbf{a}$，$\mathbf{A} = \boldsymbol{\Sigma}_{XX}^{-1}\boldsymbol{\Sigma}_{XY}\boldsymbol{\Sigma}_{YY}^{-1}\boldsymbol{\Sigma}_{YX}$；典型相关系数 = 特征值平方根。
- 典型变量**成对出现、依次递减**，最多 $\min(p,q)$ 对；第二对起要求与前面的对正交。
- 样本情形用 $\mathbf{S}$ 代替 $\boldsymbol{\Sigma}$；保留几对可用 Bartlett $\chi^2$ 检验，统计量在零假设下近似服从 $\chi^2_{(p-k+1)(q-k+1)}$ 分布。
- 解读结果三件套：**典型载荷**命名典型变量、**冗余指数**把相关翻译成可解释方差、Bartlett 检验定对数。
- 现代变体：**稀疏 CCA** 与**深度 CCA** 在高维与非线性场景延续「最大化相关」的骨架。
- 实战底线：**样本量要超过 $p+q$**（保证 $\mathbf{S}$ 正定），组内共线性严重时先 PCA 压缩再跑 CCA。

在下一节，我们将换一条路回答「变量之间的关系」：不再假设数据活在欧氏空间，而是直接从距离或相异度出发重建低维坐标——这就是**对应分析与多维标度**。