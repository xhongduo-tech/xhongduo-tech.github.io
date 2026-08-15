---
title: 统计软件实现（R/SAS）
date: 2026-08-07
---

# 统计软件实现（R/SAS）

<div class="epigraph">
<p>统计软件不会替你思考，但它会让诚实的思考变得可行。</p>
<footer>—— 统计分析实践者言</footer>
</div>

<div class="article-byline">
<p>第五级 · 田间试验与生物统计（农业试验设计与统计分析） ｜ 盖钧镒《试验统计方法》附录 ｜ 2026-08-07</p>
</div>

## 为什么专门讲软件

前面 13 讲把方法讲透了，但真实的分析很少手算：几十个小区、几百行数据，手算方差分析要整整一天，还容易算错。统计软件的价值不在「代替思考」，而在<strong>把「方法正确」变成「执行可靠」</strong>——同样的方差分析，手算与软件算结果一致，但软件能处理大数据、能自动化、能出图、能复现。<span class="marginnote">本章选用 R 与 SAS 两条线：<strong>R 免费、开源、可编程</strong>，是当今农业科研与学术发表的主流；<strong>SAS 在官方统计机构、农业推广体系与许多国家试验站仍是标准</strong>。学会一个的语法，再对照着学另一个并不难，因为背后的统计模型是同一套。</span>这一讲以「随机区组设计 + 方差分析」为主线，演示两种软件的实现，并给出全套通用套路。

## 1 数据的组织：长表还是宽表

统计软件的输入是**结构化数据**。田间数据通常整理成两种形态：

**宽表（wide format）**：每行一个小区，处理与区组是独立的列。适合人看，也适合多数统计过程。

| 区组 | 处理 | 产量 |
| --- | --- | --- |
| I | A | 18.2 |
| I | B | 25.6 |
| II | A | 20.1 |
| II | B | 26.8 |

**长表（long format）**：一个观测一行、变量一列，与宽表在「区组 × 处理」单元较多时等价。R 的 tidyverse 生态偏爱长表，因为分面、分组绘图方便。

关键原则：**每个观测（小区）一行，每个变量（区组、处理、产量）一列**。区组与处理是**因子（factor）**，产量是**数值（numeric）**——因子与分析必须区分开，否则软件会误把「处理 1、2、3」当数值。

## 2 R 实现：读入、整理与描述统计

R 是开源统计语言，配合 RStudio 使用。先读入数据并做描述统计：

```r
library(readr)
dat <- read_csv("field_trial.csv")  # 列：block, treatment, yield
dat$block     \lt - factor(dat$block)
dat$treatment \lt - factor(dat$treatment)

# 按处理分组求均值与标准差
library(dplyr)
dat |>
  group_by(treatment) |>
  summarise(mean = mean(yield), sd = sd(yield), n = n())
```

要点：

**读入**：`read_csv` 读 CSV，中文数据注意文件编码。
**因子化**：`factor()` 把区组与处理转成因子，方差分析才能正确识别组别。
<strong>管道</strong>：`|>` 是 R 的管道符，把数据依次送入后续操作，代码可读性大幅提升。<span class="marginnote"><strong>编码坑</strong>：Windows 下导出的 CSV 常是 GBK 编码，`read_csv` 可能读乱。解决：`read_csv("file.csv", locale = locale(encoding = "GBK"))`，或先用记事本另存为 UTF-8。</span>

描述统计之后，先画箱线图看数据分布与离群点，再决定是否继续分析。

## 3 R 实现：方差分析与多重比较

随机区组设计的方差分析在 R 里用 `aov` 或 `lm` 完成：

```r
# 方差分析：产量 ~ 处理 + 区组
fit <- aov(yield ~ treatment + block, data = dat)
summary(fit)

# 多重比较：Tukey HSD
library(agricolae)
hsd <- HSD.test(fit, "treatment", group = TRUE)
print(hsd$groups)

# LSD 多重比较
lsd \lt - LSD.test(fit, "treatment", p.adj = "none")
print(lsd$groups)
```

**模型公式**：`yield ~ treatment + block` 读作「产量由处理与区组解释」。注意**区组也要写进公式**——它从误差里扣走区组平方和，这正是随机区组设计的要求。
**Tukey HSD**：控制整体错误率，`group = TRUE` 输出分组字母标记（同字母表示不显著）。
<strong>agricolae 包</strong>：农业试验专用包，`HSD.test`、`LSD.test`、`duncan.test` 一应俱全。<span class="marginnote">R 里做方差分析最易犯的错是<strong>漏写区组</strong>：`aov(yield ~ treatment)` 把区组差异全算进误差，得到错误的 F 值与 p 值。随机区组、拉丁方、裂区的模型公式都必须包含设计结构项。</span>

若数据不平衡（各处理重复数不等），`aov` 的平方和分解可能不唯一，宜改用 `lm` + `Anova(fit, type = 3)`（car 包）做 III 型平方和。

## 4 SAS 实现：方差分析的经典流程

SAS 用过程（procedure）驱动分析。随机区组设计的方差分析如下：

```sas
data field;
  input block $ treatment $ yield;
  datalines;
I  A 18.2
I  B 25.6
II A 20.1
II B 26.8
;
run;

proc glm data = field;
  class block treatment;
  model yield = block treatment;
  means treatment / lsd tukey lines;
run;
```

**data 步**：建数据集，`$` 表示字符型变量，`datalines` 后跟原始数据。
**proc glm**：一般线性模型过程，是农业统计的主力。
**class 语句**：声明哪些变量是分类变量（因子）——**漏写 `class` 是最常见的错误**，软件会把处理当连续变量跑回归。
- **model 语句**：`yield = block treatment` 与 R 的公式对应。
- <strong>means 语句</strong>：`lsd` 与 `tukey` 指定多重比较方法，`lines` 输出字母标记。<span class="marginnote">SAS 与 R 的术语对照：`class` 语句 ≈ R 的 `factor()`；`model y = x` ≈ R 的公式 `y ~ x`；`proc` ≈ R 的包/函数。<strong>模型结构两边完全一样，只是语法外壳不同</strong>——理解统计模型，软件只是换着皮囊。</span>

裂区设计的 SAS 模型要写清主区误差项：`model y = block A block*A B A*B; test h = A e = block*A;`——`test` 语句指定主区因子的检验分母，对应第 3 讲的「两套误差」。

## 5 公式解析：软件输出怎么读

以 R 的 `summary(fit)` 输出为例，读懂方差分析表的列：

```text
            Df Sum Sq Mean Sq F value  Pr(>F)
treatment    2  220.7   110.3   20.53 0.00039
block        3   45.0    15.0    2.79 0.10200
Residuals    9   48.4     5.4
```

**Df**：自由度——处理 2、区组 3、残差 9，与手算一致（见第 9 讲）。
**Sum Sq / Mean Sq**：平方和与均方。
**F value**：处理 $F = 110.3/5.4 = 20.53$。
- <strong>Pr(>F)</strong>：p 值。处理 $p = 0.00039 < 0.05$，<strong>显著</strong>；区组 $p = 0.102$，不显著——区组不显著没关系，它本就是用来控制误差的，不是考察对象。<span class="marginnote"><strong>读表要点</strong>：只看「处理」那行的 p 值决定结论；「区组」行显著与否都不影响处理结论——区组的作用已在模型里扣除了。别因「区组不显著」怀疑设计，那是两回事。</span>

把软件输出的每一列与手算公式对上，是学习软件最有效的路径——**软件替你算数，你负责读懂数。**

## 6 常见软件工作流的完整套路

把全专题的方法装进一条可复用的流水线：

**数据准备**：宽表录入 → 转长表（R 用 `pivot_longer`）→ 因子化 → 数据审查（箱线图、异常值）。
**描述统计**：分组均值、标准差、CV，画图。
**前提检验**：正态性（Shapiro-Wilk 或 QQ 图）、方差齐性（Levene 检验）。
**主分析**：按设计选模型——CRD 用 `y ~ trt`，RCBD 用 `y ~ trt + block`，拉丁方用 `y ~ trt + row + col`，裂区按两套误差写 `test`。
**多重比较**：显著后做 LSD/Tukey/Duncan，输出字母分组。
**回归与相关**：`cor()`、`lm(y ~ x)`、`summary()`、通径分析用 `lavaan` 或 `agricolae::path.analysis`。
<strong>报告</strong>：把表、图、置信区间整理成规范结果，注明软件与版本，保证可复现。<span class="marginnote"><strong>可复现性</strong>是软件时代的科研底线：保存脚本、数据与版本号（R 用 `sessionInfo()`，SAS 用 `proc version;`），让任何人重跑都能得到同样结果。分析脚本与试验报告同样值得归档。</span>

## 7 易错辨析

**辨析｜易错点：**分类变量没声明成因子就进模型。R 里不 `factor()`、SAS 里不写 `class`，软件都会把「处理 1、2、3」当成数值回归，结果毫无意义。**凡是设计中的分组变量，第一件事就是声明成因子。**

另一高频错误是**盲目信任默认输出**：软件默认可能做的是 I 型平方和，数据不平衡时结论不稳；默认图可能掩盖离群点。用软件的默认值之前，先确认它默认的是什么——**软件是工具，不是裁判**。

## 8 小结

- 数据组织铁律：**每小区一行、每变量一列**；分组变量必须因子化。
- R 实现：`aov(yield ~ treatment + block)` + `HSD.test`；模型公式必须包含设计结构项。
- SAS 实现：`proc glm` + `class` + `model` + `means / lsd tukey lines`；漏写 `class` 是头号错误。
- 读软件输出：把 **Df、Sum Sq、Mean Sq、F、Pr(>F)** 各列与手算公式一一对上。
- 完整套路：**整理 → 描述 → 前提检验 → 主分析 → 多重比较 → 回归 → 可复现报告**。

到这里，田间试验与生物统计的十四讲走完了：从设计的三大原理，到随机区组、拉丁方、裂区、正交五种方案；从描述统计、概率分布，到方差分析、卡方检验、相关回归与通径分析；再到抽样调查与软件实现。下一站，你可以把这些工具带回田间——**设计一个试验、收集一份数据、写出一份可信的结论**，这正是「从极限到大模型」里，用统计的确定性去对抗自然的随机性的第一课。
