---
title: R / 统计软件实践
date: 2026-08-07
---

# R / 统计软件实践

<div class="epigraph">
<p>软件是统计学的仓库：方法写进教科书，也写进代码。</p>
<footer>—— 罗伯特·金特尔曼（Robert Gentleman）</footer>
</div>

<div class="article-byline">
<p>第二级 · 生物统计学与实验设计 ｜ 杜荣骞《生物统计学》第 17 章 · Zar 附录 ｜ 2026-08-07</p>
</div>

## 为什么从 R 开始

前十四篇建立了生物统计的完整方法谱系——从描述统计到生存分析，从实验设计到功效计算。但一套方法若只活在纸面上，就无法真正服务你的数据。**计算工具**是这套方法的落地形式。

为什么选 R？它是免费开源的、由统计学家自己维护的、覆盖本专题全部方法的通用语言。从 `t.test()` 到 `survfit()`，从 `aov()` 到 `glm()`，前十四篇的每一个方法在 R 里都有对应的、经过同行评审的函数。<span class="marginnote">R 由 Ross Ihaka 与 Robert Gentleman 于 1993 年在奥克兰大学创建，脱胎于 S 语言。它的生态（CRAN 上超过两万个包）让「方法发表 → 方法进包」几乎成为当代统计学的惯例。</span>

## 1 数据导入与整理

R 里数据的基本容器是**数据框（data frame）**——表格，行是个体、列是变量。数据导入的三个常见来源：

**CSV 文件**：`read.csv()`，注意 `stringsAsFactors = FALSE`。
**Excel**：`readxl` 包的 `read_excel()`。
**手工录入**：`data.frame()` 直接构造。

**数据整理（data wrangling）**是分析前最耗时的一步，`dplyr` 包的五个动词覆盖九成需求：`filter()` 选行、`select()` 选列、`mutate()` 造新变量、`group_by()` 分组、`summarise()` 汇总。<span class="marginnote">`dplyr` 是「整洁数据」（tidy data）哲学的实现：每行一个观测、每列一个变量、每格一个值。把数据整理成整洁格式，后续所有分析函数都能无缝衔接——这套哲学由 Hadley Wickham 系统化，如今是 R 社区的事实标准。</span>

```r
library(dplyr)
df <- read.csv("mouse.csv") |>
  filter(sex == "F") |>                     # 只保留雌鼠
  mutate(bmi = weight / (length/100)^2) |>  # 计算新变量
  group_by(treatment) |>
  summarise(mean_weight = mean(weight), sd_weight = sd(weight))
```

## 2 描述统计与图形

描述统计在 R 里可以一行算完，但**先画图**永远是纪律：

```r
# 图形诊断
hist(df$weight)                       # 直方图：看分布
boxplot(weight ~ treatment, data = df) # 箱线图：多组比较
plot(x, y)                            # 散点图：看关联
qqnorm(df$weight)                     # 正态概率图

# 描述统计
mean(df$weight); sd(df$weight)        # 均值与标准差
summary(df)                           # 五数概括 + 均值
```

`ggplot2` 是 R 的图形语法实现：一切图形由「数据 + 几何对象 + 映射」组合而成。<span class="marginnote">ggplot2 的「图形语法」由 Leland Wilkinson 提出：一张图 = 数据 + 坐标系 + 几何标记（点/线/箱） + 美学映射（颜色/形状/大小）。学会这套语法后，画任何图都只是「换几何对象」。</span>

```r
library(ggplot2)
ggplot(df, aes(x = treatment, y = weight, fill = treatment)) +
  geom_boxplot() +
  labs(title = "不同处理下的体重分布", x = "处理", y = "体重 (g)")
```

## 3 本专题方法的 R 实现

前十四篇的方法，在 R 里一一对应：

```r
# t 检验与非参数检验（第 5 篇）
t.test(weight ~ treatment, data = df)                    # Welch t（默认）
t.test(before, after, paired = TRUE)                     # 配对 t
wilcox.test(weight ~ treatment, data = df)               # Mann–Whitney U

# 方差分析（第 6 篇）
res <- aov(weight ~ treatment, data = df)
summary(res)                                             # 方差分析表
TukeyHSD(res)                                            # 事后多重比较

# 相关与回归（第 9 篇）
cor.test(x, y)                                           # 相关系数 + 检验
fit <- lm(weight ~ length, data = df)
summary(fit)                                             # 回归系数、R²、p
plot(fit)                                                # 残差诊断

# 卡方与列联表（第 10 篇）
chisq.test(table(df$smoke, df$disease))                  # 卡方独立性检验
fisher.test(table(df$smoke, df$disease))                 # 小样本精确检验

# 广义线性模型（第 11 篇）
glm(disease ~ age + smoke, data = df, family = binomial) # logistic 回归
glm(count ~ exposure, data = df, family = poisson)       # Poisson 回归

# 生存分析（第 13 篇）
library(survival)
km <- survfit(Surv(time, status) ~ treatment, data = df) # KM 曲线
survdiff(Surv(time, status) ~ treatment, data = df)      # 对数秩检验
coxph(Surv(time, status) ~ treatment + age, data = df)   # Cox 比例风险

# 功效分析（第 14 篇）
library(pwr)
pwr.t.test(d = 0.6, sig.level = 0.05, power = 0.8, type = "two.sample")
```

## 4 随机化与可复现研究

R 的 `set.seed()` 让随机操作可复现：同一种子 → 同一结果。做模拟、抽样、随机分组前先设种子：

```r
set.seed(20260807)                             # 固定随机种子
df <- df[sample(nrow(df)), ]                   # 打乱行序（随机化）
df$group <- sample(c("A", "B"), nrow(df), replace = TRUE)  # 随机分组
```

**可复现的完整实践**：用 R Markdown / Quarto 把「数据清理 → 分析 → 图表 → 报告」写成一份文档，读者可一键重跑。这比「论文 + 孤立脚本」先进一代——它把可复现性从「附带脚本」提升到「文档即结果」。<span class="marginnote">R Markdown 里代码与叙述并存，同一份文档既是代码也是论文。学术期刊与基金评审正越来越多地要求提供可复现的完整分析，这是统计实践的行业新标准。</span>

## 5 常见陷阱

**读入即错**：CSV 里的中文编码、空值、列名自动加句点，`read.csv()` 之后先 `str()` / `head()` 检查再往下走。
**忘看缺失值**：R 的 `mean()` 遇到 `NA` 返回 `NA`——先 `na.omit()` 或写 `na.rm = TRUE`。<span class="marginnote">`na.rm = TRUE` 是 R 新手第一个大坑：`mean(x, na.rm = TRUE)` 才算得出，缺了它整行返回 `NA`，后面的管线全部静默失败。</span>
**factor 陷阱**：`read.csv()` 旧默认把字符串变 factor（`stringsAsFactors = TRUE`），忘写 `stringsAsFactors = FALSE` 会导致分组顺序错乱。
**绘图先于检验**：先 `hist()` / `qqnorm()` / `boxplot()` 看数据形态，再决定用参数还是非参数检验——把《t 检验与非参数检验》的决策表变成肌肉记忆。

## 6 小结

- R 的**数据框**是分析起点：`read.csv()` 导入、`dplyr` 五个动词（`filter` / `select` / `mutate` / `group_by` / `summarise`）整理。
- 本专题每个方法都有对应函数：`t.test`、`aov`、`lm`、`chisq.test`、`glm`、`survfit`、`coxph`、`pwr.t.test`。
- **先画图再算数**是 R 社区第一纪律；`ggplot2` 用「数据 + 几何对象 + 美学映射」画一切图。
- **可复现**：`set.seed()` 固定随机性，R Markdown / Quarto 让分析一键重跑。

到这里，生物统计学的主干方法——从描述统计、推断检验、方差分析、回归与卡方、GLM、多元与生存分析，到实验设计与功效——已经全部落地为可以运行的工具。十五篇的每一行公式，最终都通向 `t.test()` 之类的一行代码；而更重要的是，你现在知道**每一行代码背后在检验什么假设、问什么科学问题**。在更深的层级里，你将把这些方法组合成完整的分析管线，并逐步接触贝叶斯统计、混合效应模型与因果推断——那是「从极限到大模型」这座知识大厦的更高楼层。