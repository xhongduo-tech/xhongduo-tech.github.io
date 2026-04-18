## 核心结论

经验贝叶斯（Empirical Bayes, EB）是一种两步推断方法：先用一批数据估计先验或超参数，再把这个估计结果代入后验推断。它不是“先验来自个人经验”，而是“先验从同一批问题的数据结构中估计出来”。

用分层模型写就是：

$$
y_i\mid \theta_i \sim p(y_i\mid \theta_i),\quad \theta_i\mid \phi \sim p(\theta_i\mid \phi)
$$

其中，$y_i$ 是第 $i$ 组观测，$\theta_i$ 是第 $i$ 组真实参数，$\phi$ 是控制所有组共享分布的超参数。经验贝叶斯先估计：

$$
\hat\phi=\arg\max_\phi \prod_i \int p(y_i\mid \theta_i)p(\theta_i\mid \phi)\,d\theta_i
$$

再计算：

$$
p(\theta_i\mid y_i,\hat\phi)
$$

核心收益是收缩（shrinkage）：把噪声大的单组估计往总体结构上拉一点。收缩不是把所有组变成一样，而是在“单组证据”和“总体规律”之间加权。

新手版例子：有很多广告组，每组曝光和转化都很少。直接用“转化数 / 曝光数”估计每组转化率，会让 1 次曝光 1 次转化的广告组显示为 100%。经验贝叶斯会先从所有广告组学出一个整体转化率分布，再把小样本广告组的估计往整体均值收一点。

| 维度 | 普通贝叶斯 | 经验贝叶斯 |
|---|---|---|
| 先验来源 | 人工指定或外部知识 | 从数据估计 |
| 超参数处理 | 通常给定或建模后积分 | 先估计成 $\hat\phi$ 再代入 |
| 优点 | 不确定性表达更完整 | 计算更便宜，适合大规模 |
| 风险 | 先验选择影响结果 | 会低估超参数不确定性 |

| 方法 | 单组估计 | 总体收缩估计 |
|---|---|---|
| 信息来源 | 只看本组数据 | 同时看本组和所有组 |
| 小样本表现 | 方差大，极端值多 | 更稳定 |
| 大样本表现 | 通常可靠 | 收缩变弱，接近单组估计 |
| 典型场景 | 单个广告组转化率 | 大量广告组、医院、城市、学校 |

---

## 问题定义与边界

经验贝叶斯解决的问题是：有很多组数据，每组样本少，但组与组之间存在共享结构，目标是给每组参数做更稳定的估计。

这里的“组”可以是广告组、医院、城市、学校、实验分层、人群标签。每组有自己的真实参数 $\theta_i$，但这些 $\theta_i$ 不是完全无关的，而是来自某个共同分布。这个共同分布由超参数 $\phi$ 控制。

定义框：

> 经验贝叶斯 = 数据驱动的先验估计 + 后验推断。

新手版例子：比较 3 家医院的治疗效果。每家医院病例少时，直接比较治愈率很不稳定。一家医院 5 个病例治好 5 个，不等于它真实水平一定高于 1000 个病例治好 850 个的医院。经验贝叶斯允许三家医院共享一个总体效果分布，再对每家医院做单独估计。

但边界必须清楚。EB 不是 full Bayes。full Bayes 是全贝叶斯推断，意思是也给超参数 $\phi$ 建先验，并在推断时积分它的不确定性。EB 则把 $\phi$ 估成一个点值 $\hat\phi$，后续把它当作已知。这让计算变快，但区间估计通常偏窄。

反例版：如果只有 2 个组，而且两组分布差异很大，比如一个是成人医院，一个是儿童医院，疾病结构完全不同，那么从这 2 个组学一个共享先验就很脆弱。此时收缩可能不是降低噪声，而是把两类本来不同的对象错误地拉到一起。

| 场景 | 是否适合 EB | 原因 |
|---|---:|---|
| 多组数据 | 适合 | 可以从组间结构估计先验 |
| 稀疏观测 | 适合 | 单组估计方差大，收缩有价值 |
| 共享分布明显 | 适合 | 收缩方向有统计依据 |
| 组数太少 | 不适合 | 超参数估计不稳 |
| 先验族明显错设 | 不适合 | 会往错误结构收缩 |
| 强选择偏差 | 不适合直接用 | 先筛选再估计会过于乐观 |

---

## 核心机制与推导

经验贝叶斯的机制可以按四步理解。

第一步，写出分层模型。分层模型是指参数也来自一个更高层的分布。不是只假设 $y_i$ 由 $\theta_i$ 生成，还假设 $\theta_i$ 来自共同分布 $p(\theta_i\mid\phi)$。

第二步，边际化掉 $\theta_i$。边际化是把暂时不关心的未知量积分掉，只留下观测数据对超参数的支持：

$$
p(y_i\mid\phi)=\int p(y_i\mid\theta_i)p(\theta_i\mid\phi)d\theta_i
$$

第三步，用所有组的边际似然估计 $\phi$：

$$
\hat\phi=\arg\max_\phi \prod_i p(y_i\mid\phi)
$$

第四步，把 $\hat\phi$ 代入每组后验：

$$
p(\theta_i\mid y_i,\hat\phi)
$$

这就是“先学总体，再估单组”。

James-Stein 估计说明了 EB 的核心不是玄学修正，而是数据驱动的收缩强度。设有 $p$ 个正态均值问题，观测向量 $x$ 满足每个坐标都有噪声，方差为 $\sigma^2$。经典 James-Stein 估计为：

$$
\hat\theta^{JS}(x)=\left(1-\frac{(p-2)\sigma^2}{\|x\|^2}\right)x,\quad p\ge 3
$$

其中 $\|x\|^2$ 是向量长度平方。公式里的括号就是收缩系数。$\|x\|^2$ 越大，观测离零越远，收缩越弱；噪声 $\sigma^2$ 越大，收缩越强。

玩具例子：先想象每个坐标都是一门成绩，单看一门会受当天发挥影响。如果知道所有成绩来自一个共同水平分布，就能把极端值往整体中心拉回一点。

数值验证：

$$
x=(3,1,0),\quad \sigma^2=1,\quad p=3,\quad \|x\|^2=10
$$

收缩系数是：

$$
1-\frac{3-2}{10}=0.9
$$

所以：

$$
\hat\theta^{JS}=(2.7,0.9,0)
$$

MLE 是最大似然估计，意思是只选择最能解释当前观测的参数。这里 MLE 就是 $(3,1,0)$。James-Stein 把它往零点收缩了一点。经典结论是：当 $p\ge 3$ 时，在平方损失下，James-Stein 可以在整体风险上优于逐坐标 MLE；当 $p\le 2$ 时，不出现同样的普适优势。

| 方法 | 使用信息 | 是否收缩 | 和 EB 的关系 |
|---|---|---:|---|
| MLE | 只看当前观测 | 否 | EB 的对照基线 |
| 普通贝叶斯 | 观测 + 指定先验 | 是 | 先验通常外部给定 |
| 经验贝叶斯 | 观测 + 数据估计的先验 | 是 | 先从数据学超参数 |
| James-Stein | 多维观测的整体长度 | 是 | 可从 EB 视角理解收缩 |

复杂度边界也很实际。正态-正态共轭模型常能闭式计算，遍历一次组汇总即可，通常是 $O(n)$。一般层级模型需要边际似然优化或近似推断，常见复杂度可写成 $O(Tn)$，其中 $n$ 是组数，$T$ 是迭代步数。

---

## 代码实现

下面实现一个最小正态-正态经验贝叶斯流程。正态-正态共轭是指观测分布和先验分布都是正态，后验也仍然是正态，因此有闭式公式。

设每组观测均值为 $\bar y_i$，组内观测方差已知为 $\sigma_i^2$，总体先验为：

$$
\theta_i \sim N(\mu_0,\tau^2)
$$

后验均值为：

$$
E[\theta_i\mid \bar y_i,\hat\mu_0,\hat\tau^2]
=
w_i\bar y_i+(1-w_i)\hat\mu_0
$$

其中：

$$
w_i=\frac{\hat\tau^2}{\hat\tau^2+\sigma_i^2}
$$

$w_i$ 是本组数据权重。组内噪声越大，$w_i$ 越小，估计越靠近总体均值。

```python
import numpy as np

def empirical_bayes_normal(group_means, group_variances):
    """
    正态-正态经验贝叶斯：
    1. 用所有组均值估计总体均值
    2. 用组间方差扣除平均组内噪声，估计 tau^2
    3. 计算每组后验均值
    """
    y = np.asarray(group_means, dtype=float)
    sigma2 = np.asarray(group_variances, dtype=float)

    mu0_hat = float(np.mean(y))

    # method-of-moments: Var(y_i) ~= tau^2 + mean(sigma_i^2)
    raw_between = float(np.var(y, ddof=1))
    tau2_hat = max(0.0, raw_between - float(np.mean(sigma2)))

    if tau2_hat == 0.0:
        posterior_mean = np.full_like(y, mu0_hat)
        weights = np.zeros_like(y)
    else:
        weights = tau2_hat / (tau2_hat + sigma2)
        posterior_mean = weights * y + (1.0 - weights) * mu0_hat

    return {
        "mu0_hat": mu0_hat,
        "tau2_hat": tau2_hat,
        "weights": weights,
        "posterior_mean": posterior_mean,
    }

# 玩具数据：5 个广告组的转化率估计，方差越大表示该组样本越少
group_means = [0.20, 0.01, 0.08, 0.13, 0.50]
group_variances = [0.002, 0.004, 0.003, 0.002, 0.030]

result = empirical_bayes_normal(group_means, group_variances)

assert result["posterior_mean"].shape == (5,)
assert result["tau2_hat"] >= 0
assert result["posterior_mean"][-1] < group_means[-1]  # 0.50 被往总体均值收缩
assert result["posterior_mean"][1] > group_means[1]   # 0.01 被往总体均值收缩

print("mu0_hat =", round(result["mu0_hat"], 4))
print("tau2_hat =", round(result["tau2_hat"], 4))
print("weights =", np.round(result["weights"], 3))
print("posterior_mean =", np.round(result["posterior_mean"], 4))
```

这个代码没有实现所有 EB 方法，但体现了关键顺序：

```python
# 1. collect group summaries
# 2. estimate hyperparameters
# 3. compute shrinkage
# 4. return posterior estimates
```

| 名称 | 含义 | 在代码中的变量 |
|---|---|---|
| 输入 | 每组观测均值 | `group_means` |
| 输入 | 每组估计方差 | `group_variances` |
| 中间量 | 总体均值估计 | `mu0_hat` |
| 中间量 | 组间真实方差估计 | `tau2_hat` |
| 输出 | 每组收缩权重 | `weights` |
| 输出 | 后验均值点估计 | `posterior_mean` |

真实工程例子：大规模 A/B 分层分析中，每个城市、渠道、设备类型都有一个实验效果估计。长尾分层样本很少，直接报告单层 lift 会出现大量极端值。EB 可以先估计所有分层效果的总体分布，再对每个分层效果做收缩，降低“偶然极端分层”进入决策的概率。

---

## 工程权衡与常见坑

经验贝叶斯的效果依赖一个前提：组之间确实共享某种结构。如果共享结构不存在，EB 的收缩就不是稳定估计，而是系统性偏误。

第一个坑是把 EB 当 full Bayes。EB 把 $\hat\phi$ 当作已知，所以后验区间只反映 $\theta_i$ 的不确定性，往往没有充分反映 $\phi$ 的不确定性。点估计变稳和区间估计变准不是一回事。点估计更稳通常指均方误差降低；区间更准则要求覆盖率可靠。

第二个坑是组数太少。组数少时，超参数估计容易被异常组带偏。比如只有 3 家医院，其中 1 家是特殊专科医院，它会显著影响总体分布，导致其他医院被错误收缩。

第三个坑是先验族错设。先验族是你假设 $\theta_i$ 来自哪类分布，比如正态、Beta、Gamma。如果真实分布是明显多峰的，却用单峰正态，EB 会把不同类型的组拉向同一个中心。

第四个坑是选择偏差。选择偏差是指样本进入分析之前已经被某种规则筛过，导致观测不再代表总体。新手版例子：如果你先挑出“效果最好”的 10 个广告组再做 EB，结果通常会过于乐观，因为这些组是因为噪声和真实效果共同作用才被选出来的。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 把 EB 当 full Bayes | 区间偏窄，风险低估 | 明确报告 $\hat\phi$ 是点估计 |
| 组数太少 | 先验不稳，收缩方向漂移 | 做敏感性分析或改用 full Bayes |
| 先验族错设 | 往错误中心收缩 | 检查残差、分组画图、尝试混合先验 |
| 选择偏差 | 结果过于乐观 | 在筛选前建模，或做选择偏差校正 |
| 忽略异方差 | 小样本组权重过高 | 使用每组方差或样本量进入权重 |

工程上常见做法是：先用 EB 作为低成本主估计，再用 bootstrap 或 full Bayes 检查关键结论。bootstrap 是重复重采样估计不确定性的方法，适合检验排序、筛选和报告区间是否稳定。

---

## 替代方案与适用边界

经验贝叶斯不是所有层级问题的最终答案。它适合“大量组、小样本、共享分布”的场景，尤其适合需要上线的批量估计系统。它的优势是快、稳定、容易解释；代价是对超参数不确定性处理不足。

边界结论：

> EB 适合：大量组 + 单组小样本 + 共享分布明显。  
> full Bayes 更适合：需要完整不确定性量化 + 组数不多 + 模型风险较高。

新手版例子：如果你只有少量城市，而且每个城市差异很大，full Bayes 往往更合适。因为它不会把超参数当成确定值，而是保留“总体分布到底是什么”的不确定性。

工程版例子：如果你要做大规模 A/B 分层分析，每天对上万个分层输出效果估计，EB 常常比 full Bayes 更快、更容易上线。你可以用 EB 做日报和监控，用 full Bayes 对少数高风险决策做复核。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 经验贝叶斯 | 快，稳定，适合批量估计 | 区间可能偏窄 | 大量广告组、城市、医院、学校 |
| full Bayes | 不确定性表达完整 | 计算更重，建模成本高 | 医疗决策、高风险实验、组数较少 |
| 单独分组估计 | 简单直观 | 小样本方差大 | 每组样本都很充足 |
| bootstrap 校正 | 实现灵活，可评估稳定性 | 计算量较大，需设计重采样方式 | 排名、筛选、置信区间复核 |
| 稳健方法 | 抗异常值 | 可能牺牲效率 | 异常组多、分布重尾 |

经验贝叶斯的定位可以更直接地说：它是低成本收缩估计方案，不是完整概率建模的终点。当业务目标是稳定排序、减少极端噪声、批量输出点估计时，EB 很合适；当业务目标是严肃的风险评估、覆盖率可靠的区间、模型不确定性传播时，应考虑 full Bayes 或更完整的分层推断。

---

## 参考资料

| 阅读目的 | 资料 | 用途 |
|---|---|---|
| 经典起点 | Robbins (1956) | 提出经验贝叶斯统计思想的早期经典来源 |
| 收缩估计 | James & Stein (1961) | 理解 James-Stein 估计和平方损失下的收缩现象 |
| 现代 EB 视角 | Efron (2011) | 解释 Tweedie 公式、选择偏差和现代 EB 问题 |
| 建模策略 | Efron (2014) | 比较经验贝叶斯中的不同建模策略 |
| 近似贝叶斯推断 | Kass & Steffey (1989) | 理解条件独立层级模型中的近似推断 |
| 元分析应用 | Raudenbush & Bryk (1985) | 了解 EB 在教育和元分析中的应用 |

1. Robbins, H. *An Empirical Bayes Approach to Statistics* (1956). https://digicoll.lib.berkeley.edu/record/112828?ln=en  
   用途：理解经验贝叶斯的起点，即用总体数据估计先验结构。

2. James, W. & Stein, C. *Estimation with Quadratic Loss* (1961). https://cir.nii.ac.jp/crid/1573105975105161344  
   用途：理解多维正态均值估计中的收缩优势，以及 $p\ge 3$ 的经典边界。

3. Efron, B. *Tweedie’s Formula and Selection Bias* (2011). https://pmc.ncbi.nlm.nih.gov/articles/PMC3325056/  
   用途：理解现代经验贝叶斯如何处理选择偏差和大规模估计问题。

4. Efron, B. *Two Modeling Strategies for Empirical Bayes Estimation* (2014). https://pmc.ncbi.nlm.nih.gov/articles/PMC4196219/  
   用途：理解经验贝叶斯中的建模路线，尤其是直接建模和间接建模的差异。

5. Kass, R. E. & Steffey, D. *Approximate Bayesian Inference in Conditionally Independent Hierarchical Models* (1989). https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478825  
   用途：理解层级模型中近似贝叶斯推断和 EB 的关系。

6. Raudenbush, S. W. & Bryk, A. S. *Empirical Bayes Meta-Analysis* (1985). https://journals.sagepub.com/doi/10.3102/10769986010002075  
   用途：理解 EB 在多研究、多学校、多中心效果估计中的应用。

如果只看一篇，先看 Robbins 1956 了解 EB 的起点；如果想理解收缩思想，再看 James-Stein；如果想看现代视角，再看 Efron 2011 和 2014。
