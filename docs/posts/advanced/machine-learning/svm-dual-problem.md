---
title: 对偶问题
date: 2026-08-07
---

# 对偶问题

<div class="epigraph">
<p>实数域上连接两个真理的最短路径，往往要穿过复数域。</p>
<footer>—— 雅克 · 阿达马（Jacques Hadamard）</footer>
</div>

<div class="article-byline">
<p>第四级 · 机器学习 ｜ 周志华《机器学习》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从对偶问题开始

上一节我们把 SVM 写成了带不等式约束的优化问题：最小化 $\frac{1}{2}\|\boldsymbol{w}\|^2$，约束 $y_i(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}_i + b) \geq 1$。问题本身很干净，但直接解它有两个麻烦：约束是**不等式**，处理起来笨重；而且最终解要是能写成样本内积的形式，才能为核方法铺路。

**拉格朗日对偶（Lagrange duality）**就是那扇门。它把「带约束的原问题」改写成「无约束的拉格朗日函数」，再转成**对偶问题**。对 SVM 而言，对偶问题不仅更好解，还让 **KKT 条件**把「哪些样本是支持向量」这一信息显式地挖了出来。<span class="marginnote">对偶理论是凸优化的核心工具之一，完整体系在第二级《最优化理论》里展开；这里只需要「构造拉格朗日函数 → 求偏导置零 → 代入消元」这一条标准流水线。</span>

## 1 构造拉格朗日函数对原问题的每个约束引入一个拉格朗日乘子 $\alpha_i \geq 0$，构造：`$$`L(\boldsymbol{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\boldsymbol{w}\|^2 + \sum_{i=1}^{m} \alpha_i \left(1 - y_i\left(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}_i + b\right)\right)$$
其中 $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_m)$。**直觉**：乘子 $\alpha_i$ 是「第 $i$ 个约束的分量」——约束违反得越厉害（$1 - y_i f(x_i) > 0$），罚得越重；约束满足（$\leq 0$ 项被乘子吸收）则不罚。

**对偶问题的由来**：原问题是 $\min_{\boldsymbol{w},b}\max_{\boldsymbol{\alpha}\geq 0} L$。对偶问题把 $\min$ 与 $\max$ 交换成 $\max_{\boldsymbol{\alpha}\geq 0}\min_{\boldsymbol{w},b} L$。由于 SVM 的原问题是凸的、且满足强对偶条件（Slater 条件），**两者最优值相等**——这是对偶成立的保证。<span class="marginnote">「min-max 换序」不是免费午餐：只有满足强对偶的凸问题才成立。教材在此直接使用结论，因为 SVM 恰好满足条件；更一般的对偶间隙讨论留给最优化理论。</span>

## 2 求偏导置零，消去 $\boldsymbol{w}$ 与 $b$让 $L$ 对 $\boldsymbol{w}$ 与 $b$ 分别求偏导并置零：`$$`\boldsymbol{w} = \sum_{i=1}^{m} \alpha_i y_i \boldsymbol{x}_i, \qquad 0 = \sum_{i=1}^{m} \alpha_i y_i$$
把这两个式子代回 $L$，消去 $\boldsymbol{w}$ 与 $b$，得到**对偶问题**：

$$\max_{\boldsymbol{\alpha}} \; \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m}\sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x}_j$$

$$\text{s.t.} \quad \sum_{i=1}^{m} \alpha_i y_i = 0, \qquad \alpha_i \geq 0$$

**关键观察：对偶问题里样本只以内积 $\boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x}_j$ 的形式出现**——这是 SVM 通向核函数的那扇门：只要能把内积替换成核函数 $k(\boldsymbol{x}_i, \boldsymbol{x}_j)$，SVM 就能在不显式做高维映射的情况下处理非线性问题。<span class="marginnote">「原问题看权重、对偶问题看样本内积」——对偶形式让 SVM 的核心运算只依赖样本两两之间的相似度（内积），这正是下一节《核函数》的全部前提。这个「只用内积」的性质，也让 SVM 可以不用存 $\boldsymbol{w}$ 的显式坐标。</span>

## 3 KKT 条件：支持向量显形对偶问题解出 $\boldsymbol{\alpha}$ 后，原问题的解由**KKT 条件**给出。对 SVM 而言，最关键的一条是**互补松弛（complementary slackness）**：`$$`\alpha_i \left(y_i\left(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}_i + b\right) - 1\right) = 0$$
由此推出决定性结论：

- 若 $\alpha_i > 0$，则 $y_i(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}_i + b) = 1$——样本**恰好落在间隔边界上**，是**支持向量**；
- 若 $y_i(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}_i + b) > 1$（样本离边界更远），则必须 $\alpha_i = 0$——它对解没有贡献。

于是最终模型为

$$f(\boldsymbol{x}) = \boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b = \sum_{i=1}^{m} \alpha_i y_i \boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x} + b$$

求和只对 $\alpha_i \neq 0$ 的**支持向量**进行。<span class="marginnote">这就是「解只由支持向量决定」的数学形态：多数 $\alpha_i$ 是 0，只有边界上的样本带着非零权重。训练完成后，非支持向量可以全部丢弃——模型既稀疏又省内存。</span>

## 4 公式解析：KKT 条件如何「认出」支持向量对偶问题的最优解 $\boldsymbol{\alpha}^*$ 与原始问题的最优解 $(\boldsymbol{w}^*, b^*)$ 通过互补松弛联结。逐步看：- **第一步，写互补松弛**：$\alpha_i^* \left(y_i\left(\boldsymbol{w}^{*\mathrm{T}}\boldsymbol{x}_i + b^*\right) - 1\right) = 0$。- **第二步，分类讨论**：两个因子相乘为零，必有一个为零。若 $\alpha_i^* = 0$，样本不参与求和；若 $\alpha_i^* > 0$，则括号内必须为 0，即 $y_i(\boldsymbol{w}^{*\mathrm{T}}\boldsymbol{x}_i + b^*) = 1$——样本精确坐在间隔边界上。
**第三步，读出结论**：$\alpha_i^* > 0$ 的样本集合，恰是支持向量集合。**对偶问题自动完成了「哪些样本重要」的挑选，不需要人指定。**
**第四步，如何求 $b^*$**：任取一个支持向量（$\alpha_s > 0$），代入 $y_s(\boldsymbol{w}^{*\mathrm{T}}\boldsymbol{x}_s + b^*) = 1$，解出 $b^*$；工程上常用所有支持向量的平均以保证数值稳定。

**直觉一句话**：KKT 互补松弛把「间隔边界上的样本」翻译成了「带非零乘子的样本」——优化问题自己在最后一步说出了哪些样本定义了模型。

## 5 为什么对偶值得做- **解法更高效**：对偶问题是二次规划，且现代实现（如 SMO 算法）通过**每次只更新少数 $\alpha_i$** 来逼近最优解，特别适合大规模数据。- **核化通道**：内积 $\boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x}_j$ 可替换为核函数，SVM 由此获得非线性能力（下一节）。- **稀疏性**：大量 $\alpha_i = 0$，预测时只算支持向量，模型紧凑。- **理论衔接**：对偶形式与第2章的结构风险、第12章的泛化界联系更直接。

**辨析｜易错点：** 对偶问题与原问题的最优解**相等**，不是「对偶更弱」。有些资料误以为对偶是近似——那是非凸问题的情况。对 SVM 这个凸问题，强对偶成立，解对偶即解原问题；差别只在**求解路径**，不在**解本身**。<span class="marginnote">教材采用「先对偶、后 SMO」的路径还有一个实际原因：直接解带不等式约束的原问题需要通用凸优化工具，而对偶问题的结构允许专门设计的快速算法。工程专题里你会发现，现代大规模 SVM 实现几乎清一色走对偶这条路。</span>


用两个样本把 KKT 条件走一遍：设 $x_1=(1,0), y_1=+1$；$x_2=(0,1), y_2=-1$。- **写对偶**：最大化 $\alpha_1+\alpha_2 - \frac12(\alpha_1^2\|x_1\|^2 + 2\alpha_1\alpha_2 y_1y_2 x_1^\top x_2 + \alpha_2^2\|x_2\|^2)$，约束 $\alpha_1-\alpha_2=0$、$\alpha_i\ge0$；- **代入**：$x_1^\top x_2=0$（正交），$\alpha_1=\alpha_2=\alpha$，目标变为 $2\alpha - \alpha^2$；- **解出**：$\alpha=1$（最大化 $2\alpha-\alpha^2$），两个样本都是支持向量（$\alpha_i>0$）；- **还原**：$w=\alpha_1 y_1 x_1 + \alpha_2 y_2 x_2 = (1,-1)$，$b$ 由任一支持向量代入解得——**对偶解出的 $\alpha$ 完美还原了原问题的 $w$**。这个例子展示了「对偶 = 原问题」：KKT 互补松弛自动把「间隔边界上的样本」标成支持向量。**思考题**：如果 $x_1^\top x_2 \ne 0$（不正交），对偶问题会更复杂在哪一步？
## 拓展：KKT 条件的一次数值演练

用两个样本把 KKT 条件走一遍：设 $x_1=(1,0), y_1=+1$；$x_2=(0,1), y_2=-1$。

- **写对偶**：最大化 $\alpha_1+\alpha_2 - \frac12(\alpha_1^2\|x_1\|^2 + 2\alpha_1\alpha_2 y_1y_2 x_1^\top x_2 + \alpha_2^2\|x_2\|^2)$，约束 $\alpha_1-\alpha_2=0$、$\alpha_i\ge0$；
- **代入**：$x_1^\top x_2=0$（正交），$\alpha_1=\alpha_2=\alpha$，目标变为 $2\alpha - \alpha^2$；
- **解出**：$\alpha=1$（最大化 $2\alpha-\alpha^2$），两个样本都是支持向量（$\alpha_i>0$）；
- **还原**：$w=\alpha_1 y_1 x_1 + \alpha_2 y_2 x_2 = (1,-1)$，$b$ 由任一支持向量代入解得——**对偶解出的 $\alpha$ 完美还原了原问题的 $w$**。

这个例子展示了「对偶 = 原问题」：KKT 互补松弛自动把「间隔边界上的样本」标成支持向量。

**思考题**：如果 $x_1^\top x_2 \ne 0$（不正交），对偶问题会更复杂在哪一步？

## 知识速查：对偶问题

**本节关键词**
- 拉格朗日对偶
- KKT 条件
- 互补松弛
- 对偶问题
- 强对偶
- SMO
- 内积形式
- 支持向量

**三条常见误区**
1. 以为对偶是近似——强对偶下两者相等；
2. 忽略 KKT 互补松弛识别支持向量的作用；
3. 把对偶当作纯数学技巧——它带来核化与高效求解。

**核心结论**
1. 对偶把带约束原问题变无约束拉格朗日；
2. 对偶解中样本只以内积出现——核的入口；
3. KKT 互补松弛自动识别支持向量；
4. 凸问题强对偶成立，解对偶即解原问题。

**与全书/后续的连接**
- 第2级最优化理论的凸对偶；
- 第6章核函数接在对偶之后；
- 专题一 GBDT 用梯度替代对偶。

**常见面试题**
1. 问：为什么说对偶不是近似？ 答：强对偶下对偶与原问题最优值相等。
2. 问：α_i > 0 的样本意味着什么？ 答：它落在间隔边界上，是支持向量。

**一句话记忆**
对偶把约束优化变成只含内积的问题，KKT 让支持向量显形。

## 6 小结- 构造**拉格朗日函数**，把带约束原问题转成对偶问题：$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x}_j$。- 对偶解中样本**只以内积形式出现**，这是核函数的入口。- **KKT 互补松弛** $\alpha_i(y_i f(\boldsymbol{x}_i)-1)=0$ 自动识别支持向量：$\alpha_i > 0$ 的样本即间隔边界上的样本。- 最终模型 $f(\boldsymbol{x}) = \sum_i \alpha_i y_i \boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x} + b$ 只对支持向量求和，**稀疏且省内存**。
- 凸问题满足强对偶，解对偶即解原问题；对偶的工程价值是 SMO 等高效算法。

## 本节路线图

- **第1节**：构造拉格朗日函数对原问题的每个约束引入一个拉格朗日乘子 $\alpha_i \geq 0$，构造：
- **第2节**：求偏导置零，消去 $\boldsymbol{w}$ 与 $b$让 $L$ 对 $\boldsymbol{w}$ 与 $b$ 分别求偏导并置零：
- **第3节**：KKT 条件：支持向量显形对偶问题解出 $\boldsymbol{\alpha}$ 后，原问题的解由**KKT 条件**给出。对 SVM 而言，最关键的一条是**互补松弛（complementary slackness）**：
- **第4节**：公式解析：KKT 条件如何「认出」支持向量对偶问题的最优解 $\boldsymbol{\alpha}^*$ 与原始问题的最优解 $(\boldsymbol{w}^*, b^*)$ 通过互补松弛联结。逐步看：
- **第5节**：为什么对偶值得做- **解法更高效**：对偶问题是二次规划，且现代实现（如 SMO 算法）通过**每次只更新少数 $\alpha_i$** 来逼近最优解，特别适合大规模数据。- **核化通道**：内积 $\boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x}_j$ 可替换为核函数，SVM 由此获得非线性能力（下一节）。
- **小结**：要点复盘与下一课衔接

## 复习自查清单

读完后，试着不翻书复述以下各点：

- [ ] 最终模型 $f(\boldsymbol{x}) = \sum_i \alpha_i y_i \boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x} + b$ 只对支持向量求和，**稀疏且省内存**。
- [ ] 凸问题满足强对偶，解对偶即解原问题；对偶的工程价值是 SMO 等高效算法。
- [ ] **第1节**：构造拉格朗日函数对原问题的每个约束引入一个拉格朗日乘子 $\alpha_i \geq 0$，构造：
- [ ] **第2节**：求偏导置零，消去 $\boldsymbol{w}$ 与 $b$让 $L$ 对 $\boldsymbol{w}$ 与 $b$ 分别求偏导并置零：
- [ ] **第3节**：KKT 条件：支持向量显形对偶问题解出 $\boldsymbol{\alpha}$ 后，原问题的解由**KKT 条件**给出。对 SVM 而言，最关键的一条是**互补松弛（complementary slackness）**：
- [ ] **第4节**：公式解析：KKT 条件如何「认出」支持向量对偶问题的最优解 $\boldsymbol{\alpha}^*$ 与原始问题的最优解 $(\boldsymbol{w}^*, b^*)$ 通过互补松弛联结。逐步看：
- [ ] **第5节**：为什么对偶值得做- **解法更高效**：对偶问题是二次规划，且现代实现（如 SMO 算法）通过**每次只更新少数 $\alpha_i$** 来逼近最优解，特别适合大规模数据。- **核化通道**：内积 $\boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x}_j$ 可替换为核函数，SVM 由此获得非线性能力（下一节）。
- [ ] **小结**：要点复盘与下一课衔接

在下一节，对偶留下的那扇门被推开：**核函数**——让 SVM 在不显式映射的情况下处理非线性数据。
