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
fisher.test(table(df$smoke, df$