---
title: 晶粒尺寸、微应变与 X 射线线形分析
date: 2026-08-07
---

# 晶粒尺寸、微应变与 X 射线线形分析

<div class="epigraph">
<p>峰的宽度，是晶体在纳米尺度写下的自传。</p>
<footer>—— 传统晶体学格言</footer>
</div>

<div class="article-byline">
<p>第二级 · 晶体学与衍射分析（X 射线晶体学） ｜ Massa 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从线形分析继续

Rietveld 精修把峰形当「工具」——拟合峰形是为了得到准确的强度与位置。但峰形本身还藏着另一层宝藏：**峰为什么变宽？** 理想晶体的衍射峰极其锐利，真实材料的峰却总是展宽的。这个「多余展宽」的根源，是晶粒太小与晶格微应变——它们都是纳米尺度的结构信息。

**线形分析（line profile analysis, LPA）** 就是从粉末峰的展宽里，反推晶粒尺寸与微应变。它是纳米材料研究最常用的 X 射线手段之一：催化剂活性相尺寸、电池电极应变、金属冷加工硬化、水泥水化产物尺寸……全都靠它。<span class="marginnote">峰宽的两种根源要分清楚：<strong>仪器展宽（准直、波长色散）是「放大镜糊」，样品展宽（尺寸、应变）才是「本尊</strong>」。线形分析的第一件事，就是用标准样品（无展宽的 LaB₆）测出仪器展宽，把它从实测峰形里「解卷积」剥掉，剩下的才是样品信息。</span>

本节讲 Scherrer 公式、微应变展宽、以及 Williamson–Hall 与更现代的 Fourier/Whole Powder Pattern 方法。

## 1 晶粒尺寸展宽：Scherrer 公式

当晶粒小到纳米尺度，衍射峰显著变宽。经典 **Scherrer 公式**：

$$D = \frac{K\lambda}{\beta \cos\theta}$$

拆解这条式子：

- **第一步，$D$ 是晶粒尺寸**：沿垂直于反射面方向的平均晶粒（相干散射畴）尺寸。
- **第二步，$\beta$ 是展宽**：样品贡献的峰宽（FWHM 或积分宽度，扣除仪器展宽后），以弧度为单位。
- **第三步，$K$ 是形状因子**：约 0.9（球形晶粒）；$\lambda$ 是波长，$\theta$ 是 Bragg 角。<span class="marginnote">Scherrer 公式的直觉：<strong>晶粒越小，相干散射的区域越短，衍射峰越宽</strong>——正如光栅刻线越少、衍射极大越宽。它测的是「相干散射畴」，不是电镜里的颗粒边界：一个电镜颗粒可能由多个相干畴组成，Scherrer 给的是畴尺寸。</span>

**适用范围**：$D$ 约 1–100 nm（$D \lt  \lambda/(2\beta)$ 数量级）。晶粒 > 100 nm 时展宽小到被仪器展宽淹没，Scherrer 失效；< 1 nm 时接近无定形。Scherrer 只测尺寸，不含应变信息——这引向下一节。

## 2 微应变展宽：晶格的「颤抖」

**微应变（microstrain）** 是晶格常数在晶粒内的微小、局部的变化（由位错、缺陷、掺杂、热应力引起）。应变让晶面间距 $d$ 有一个分布，Bragg 角随之有分布 → 峰展宽。

应变的展宽规律（Stokes–Wilson 关系）：

$$\beta_{\varepsilon} = 4\varepsilon \tan\theta$$

其中 $\varepsilon = \Delta d/d$ 是均方应变。注意它与 Scherrer 的关键区别：**尺寸展宽 ∝ $1/\cos\theta$，应变展宽 ∝ $\tan\theta$**——两者对 $\theta$ 的依赖不同，这正是把它们分开的钥匙。<span class="marginnote">记忆诀窍：<strong>尺寸展宽随 $\theta$ 平缓变化（$\sec\theta$），应变展宽随 $\theta$ 剧增（$\tan\theta$）</strong>。所以低角峰宽主要反映尺寸、高角峰宽主要反映应变——测多组峰宽就能同时解出两者。</span>

## 3 Williamson–Hall 法：一张图分开两种展宽

把两种展宽**相加**（洛伦兹近似），总展宽为：

$$\beta_{\text{sample}} = \beta_D + \beta_{\varepsilon} = \frac{K\lambda}{D\cos\theta} + 4\varepsilon\tan\theta$$

**Williamson–Hall 作图**：两边乘 $\cos\theta$：

$$\beta_{\text{sample}}\cos\theta = \frac{K\lambda}{D} + 4\varepsilon\sin\theta$$

拆解这条式子：

- **第一步，线性化**：以 $\beta\cos\theta$ 为 $y$、$4\sin\theta$ 为 $x$ 作图，得一条直线。
- **第二步，截距给尺寸**：$y$ 轴截距 $= K\lambda/D$ → 解出 $D$。
- **第三步，斜率给应变**：斜率 $= \varepsilon$ → 应变。<span class="marginnote">Williamson–Hall 图的局限：它假设两种展宽都来自洛伦兹形，且尺寸/应变各向同性——真实材料常「非线性」（曲线），此时 W-H 只给平均值。更严格的方法是 Fourier 分析（Warren–Averbach）或 Whole Powder Pattern Modelling（WPPM），后者把位错密度等物理量直接作为参数。</span>

**Warren–Averbach 法**是更严谨的 Fourier 方法：把峰形做傅里叶变换，Fourier 系数的初始斜率给尺寸、曲率给应变——能分离不同反射方向的尺寸与应变分布，但需要高质量高分辨数据。

## 4 公式解析：Voigt 分解与展宽分离

现代线形分析常用 **Voigt 分解**：把实测峰形分解为高斯（洛伦兹贡献的应变）与洛伦兹（尺寸贡献）分量，分别求展宽。峰的积分宽度 $\beta$ 与分量关系：

$$\beta = \beta_{\text{Lorentz}} + \beta_{\text{Gauss}} \quad (\text{近似相加})$$

实际分离用 Voigt 拟合：峰形 = 高斯与洛伦兹的卷积，拟合出各自宽度 $\beta_G$、$\beta_L$，再代入：

$$D = \frac{K\lambda}{\beta_L\cos\theta}, \qquad \varepsilon = \frac{\beta_G}{4\tan\theta}$$

拆解这条式子：

- **第一步，洛伦兹分量给尺寸**：尺寸展宽呈洛伦兹形（立方晶粒），$\beta_L$ 代入 Scherrer 得 $D$。
- **第二步，高斯分量给应变**：应变展宽近似高斯，$\beta_G$ 代入 Stokes–Wilson 得 $\varepsilon$。
- **第三步，各自对 $\theta$ 外推**：把不同反射的 $D$、$\varepsilon$ 分别按 $\theta$ 外推（低角给尺寸、高角给应变），取稳定平均值。<span class="marginnote">工程经验：<strong>线形分析对数据要求高——需要高分辨、峰形干净的图谱（同步辐射或高质量实验室数据）</strong>。标准样品定仪器展宽是前提；数据差时强行解卷积只会得到假尺寸。</span>

## 5 结晶度与非晶含量

峰宽还能给出**结晶度（crystallinity）**——样品中结晶相所占的质量/体积分数。原理：

结晶相贡献尖锐峰，非晶相贡献弥散背景（宽丘）。
结晶度 $\approx$ 结晶峰面积 /（结晶峰面积 + 非晶背景面积）。

方法：**Ruland 法**（比较总散射与结晶散射）较严格；简单面积比法常见但受制样影响大。Rietveld 定量中若样品含非晶，也可以加「非晶峰形模型」或内标测定非晶含量。<span class="marginnote">结晶度在聚合物、药物、淀粉等「半结晶」材料里是核心指标：<strong>结晶度影响强度、溶解性、保质期</strong>。但「结晶度」的数值依赖方法定义——面积比、Ruland、Rietveld 给出的数可能不同，报告时要说明方法。</span>

## 6 线形分析的应用

线形分析是材料科学最「接地气」的 X 射线工具：

**催化剂**：活性金属纳米颗粒尺寸（Scherrer）——颗粒越小、活性越高。
**电池**：电极材料充放电时的晶格应变与颗粒碎裂（微应变变化）。
- **金属**：冷加工硬化、退火回复——位错密度与微应变演变。
- **水泥**：水化产物（C–S–H）的纳米尺度结构与结晶度。
- **地质**：矿物颗粒尺寸、变质岩的应变历史。
- **纳米材料**：合成产物的晶粒尺寸与尺寸分布（WPPM）。<span class="marginnote">一个交叉提醒：<strong>Scherrer 尺寸与电镜（TEM）尺寸常常不同</strong>——Scherrer 是「相干散射畴」，TEM 是「颗粒形貌」。畴 < 颗粒是常态（畴是颗粒内的有序区）。解释数据时别把两者画等号。</span>

## 7 辨析｜易错点：线形分析的五个坑

**辨析｜易错点：** 线形分析五个高频失误：

**忘了扣仪器展宽**：直接把实测峰宽代入 Scherrer，得到的尺寸偏小。先测标准样品（LaB₆），从实测宽里扣除仪器宽。
**$K$ 因子与峰宽定义不匹配**：Scherrer 的 $K$ 取决于「用 FWHM 还是积分宽度」与晶粒形状。FWHM 用 $K≈0.9$、积分宽度用 $K≈1$；混用会差 10%。<span class="marginnote">细节决定成败：<strong>报 Scherrer 尺寸时必须声明用的哪个峰（$hkl$）与哪个 $K$</strong>。不同反射方向的尺寸可能不同（晶粒各向异性），只报一个数不完整。</span>
- **把应变当尺寸**：只用一个峰算 Scherrer，忽略高角峰应变展宽——尺寸会虚高。用 W-H 图或多个 $\theta$ 的峰交叉检查。
- **数据分辨率不足**：峰形采样点不够、仪器太宽，解卷积不可靠。高分辨数据 + 好峰形拟合是前提。
- **择优取向干扰强度、峰形污染**：制样不当让峰形畸变，尺寸/应变结果失真。

## 8 小结

- **晶粒尺寸展宽** ∝ $1/\cos\theta$（Scherrer $D = K\lambda/(\beta\cos\theta)$）；**微应变展宽** ∝ $\tan\theta$（$\beta_\varepsilon = 4\varepsilon\tan\theta$）。
- **Williamson–Hall 图**：$\beta\cos\theta$ vs $4\sin\theta$ 直线，截距给 $D$、斜率给 $\varepsilon$。
- **Voigt 分解**：洛伦兹分量给尺寸、高斯分量给应变，是现代线形分析的标配。
- **Warren–Averbach** 与 **WPPM** 是更严格的 Fourier/物理模型方法。
- **结晶度** = 结晶峰面积 / 总散射面积；Ruland 法与 Rietveld 非晶模型可定量。
- 线形分析贯穿催化、电池、金属、水泥、纳米材料研究；Scherrer 尺寸 ≠ TEM 颗粒尺寸。

在下一节，我们走出 X 射线，进入电子探针的世界：**电子衍射与透射电子显微术**——一束电子，如何看见单个原子。
