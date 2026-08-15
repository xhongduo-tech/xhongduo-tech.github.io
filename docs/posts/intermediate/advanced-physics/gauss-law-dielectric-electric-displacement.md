---
title: 有电介质时的高斯定理与电位移矢量
date: 2026-08-07
---

# 有电介质时的高斯定理与电位移矢量

<div class="epigraph">
<p>电位移矢量把「自由电荷」从「束缚电荷」的纠缠中解放出来——高斯定理从此只看自由电荷的眼色。</p>
<footer>—— 静电学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 程守洙《普通物理学》第十一章 §11-5 ｜ 2026-08-07</p>
</div>

## 为什么从电位移矢量开始

上一节我们看到电介质极化产生束缚电荷，这让「真空中的高斯定理」在介质中变得麻烦：闭合曲面内的电荷既有自由电荷又有束缚电荷，而束缚电荷往往未知。这一节引入**电位移矢量（electric displacement）** $\boldsymbol{D}$——一个把束缚电荷的效应「吸收」进定义的辅助场量。有了它，有介质时的高斯定理写成「$\boldsymbol{D}$ 的通量 = 自由电荷」，形式与真空一致、却只含自由电荷。这是处理介质中电场问题的标准工具，也是第二十一章麦克斯韦方程组中 $\boldsymbol{D}$ 的首次亮相。

## 1 电位移矢量

**电位移矢量（electric displacement）** 定义：

$$\boldsymbol{D} = \varepsilon_0\boldsymbol{E} + \boldsymbol{P}$$

其中 $\boldsymbol{E}$ 是介质中的总电场，$\boldsymbol{P}$ 是极化强度。对各向同性线性介质（$\boldsymbol{P} = \varepsilon_0\chi_e\boldsymbol{E}$）：

$$\boldsymbol{D} = \varepsilon_0\boldsymbol{E} + \varepsilon_0\chi_e\boldsymbol{E} = \varepsilon_0(1+\chi_e)\boldsymbol{E} = \varepsilon\boldsymbol{E}$$

其中 $\varepsilon = \varepsilon_0\varepsilon_r$ 是**介电常数（电容率）**。<span class="marginnote">$\boldsymbol{D}$ 是「辅助场量」，不是全新的物理场——它的意义在于简化介质中的计算。$\boldsymbol{D}$ 的单位是 $\text{C/m}^2$（电荷面密度的量纲），这个名字「电位移」来自历史（位移电流的提出），不必纠结字面。</span>

**重点：$\boldsymbol{D}$ 的源是自由电荷，$\boldsymbol{E}$ 的源是全部电荷（自由 + 束缚）。** 这正是引入 $\boldsymbol{D}$ 的目的：把「不知道的束缚电荷」藏进 $\boldsymbol{D}$ 的定义里，让方程只面对「知道的自由电荷」。

**数值算例（电位移的量级）**：平行板电容器极板自由电荷面密度 $\sigma_0 = 8.85\times10^{-6}$ C/m²（对应真空中 $E_0 = \sigma_0/\varepsilon_0 = 10^6$ V/m，很强的场）。填满 $\varepsilon_r = 5$ 的介质后：$D = \sigma_0 = 8.85\times10^{-6}$ C/m²（与真空相同），$E = D/(\varepsilon_0\varepsilon_r) = 10^6/5 = 2\times10^5$ V/m——场强降为原来的五分之一，而 $D$ 完全不变。**$D$ 的「不变性」正是它比 $E$ 方便的地方：$D$ 只认自由电荷，不随介质改变。**

## 2 有介质时的高斯定理

**有介质时的高斯定理**：通过任意闭合曲面的电位移通量，等于曲面内**自由电荷**的代数和：

$$\oint_S \boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$$

对比真空高斯定理 $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{全部}}/\varepsilon_0$——两者的差别正是「束缚电荷被吸收进 $\boldsymbol{D}$」。

**重点：有介质时的高斯定理与真空形式完全同构，只是把 $\varepsilon_0\boldsymbol{E}$ 换成 $\boldsymbol{D}$、电荷换成自由电荷。** 所有上一节的方法（选高斯面、对称性、反推场量）原封不动搬过来，只需最后用 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ 换回 $\boldsymbol{E}$。<span class="marginnote">解题套路：① 对称性选高斯面 → ② 用 $\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$ 求 $\boldsymbol{D}$ → ③ 用 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ 求 $\boldsymbol{E}$。三步走，束缚电荷从头到尾不需要显式求解。</span>

**辨析｜易错点：$\boldsymbol{D}$ 与 $\boldsymbol{E}$ 的边界条件不同。** 法向分量：$\boldsymbol{D}$ 的法向分量差由自由电荷面密度决定（无自由电荷时连续），$\boldsymbol{E}$ 的法向分量差还与束缚电荷面密度有关。切向分量：$\boldsymbol{E}$ 的切向分量连续（静电场无旋），$\boldsymbol{D}$ 的切向分量在介质分界面一般不连续。解题时——求 $\boldsymbol{D}$ 用法向条件，求 $\boldsymbol{E}$ 记得切向连续——两者互补，缺一不可。

**常见电介质的相对介电常数**：

| 介质 | $\varepsilon_r$ |
| --- | --- |
| 真空 | 1 |
| 空气（常压） | 约 1.0006 |
| 石蜡 | 约 2.1 |
| 玻璃 | 约 5–10 |
| 水（20 ℃） | 约 80 |
| 二氧化钛 | 约 100 |
| 钛酸钡 | 约 1000–10000 |

**辨析｜易错点：水的 $\varepsilon_r \approx 80$ 特别大**——因为水分子有永久电偶极矩（极性分子），极化靠取向极化，机制强；非极性分子（石蜡、苯）只有电子极化与原子极化，$\varepsilon_r$ 小（约 2）。「介电常数大」对应「容易极化」，水的强极化正是它作为优秀溶剂的原因之一。

## 3 公式解析：介质中的平行板

平行板电容器极板自由电荷面密度 $\sigma_0$，板间填满相对介电常数 $\varepsilon_r$ 的介质。求介质中的 $\boldsymbol{D}$、$\boldsymbol{E}$ 与束缚电荷面密度。

$$
D = \sigma_0, \qquad E = \frac{D}{\varepsilon_0\varepsilon_r} = \frac{\sigma_0}{\varepsilon_0\varepsilon_r}
$$

- **第一步，选高斯面求 $D$**：取跨介质的圆柱盒高斯面（一底在极板内），$\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = D\cdot\Delta S = \sigma_0\Delta S$，得 $D = \sigma_0$——**电位移在数值上等于自由电荷面密度**。
- **第二步，换回 $E$**：$E = D/(\varepsilon_0\varepsilon_r) = \sigma_0/(\varepsilon_0\varepsilon_r)$——介质中场强削弱 $\varepsilon_r$ 倍。
- **第三步，对比真空**：真空时 $E_0 = \sigma_0/\varepsilon_0$，故 $E = E_0/\varepsilon_r$，与上节结论一致。
- **第四步，求束缚电荷（可选）**：$\sigma' = \sigma_0(1 - 1/\varepsilon_r)$——用 $\boldsymbol{D}$ 方法时这一步甚至可以不求。

**辨析｜易错点：**$\boldsymbol{D}$ 与 $\boldsymbol{E}$ 的边界行为不同：$\boldsymbol{D}$ 的法向分量由自由电荷决定（连续或跳变按自由电荷面密度），$\boldsymbol{E}$ 的法向分量还与束缚电荷有关。解题时「先 $\boldsymbol{D}$ 后 $\boldsymbol{E}$」的顺序不能反——直接用 $\boldsymbol{E}$ 的高斯定理会撞上未知的束缚电荷。

## 4 静电场的能量密度

有了 $\boldsymbol{D}$ 与 $\boldsymbol{E}$，静电场的能量可写成更一般的形式。真空中能量密度：

$$w_e = \frac{1}{2}\varepsilon_0 E^2$$

介质中：

$$w_e = \frac{1}{2}\boldsymbol{D}\cdot\boldsymbol{E} = \frac{1}{2}\varepsilon E^2$$

电容器储能 $W = \int w_e\,\mathrm{d}V$ 与 $W = \frac{1}{2}CU^2$ 一致。<span class="marginnote">「能量储存在电场里」是法拉第场思想的关键推论：不是储存在电荷上，而是储存在电荷激发的场中。能量密度 $\frac{1}{2}\varepsilon E^2$ 与运动学能量、弹簧势能的形式同构，再次体现「$\frac{1}{2}$×系数×（广义量）²」的普适结构。</span>

**重点：静电场能量密度 $w_e = \frac{1}{2}\varepsilon E^2$，正比于场强平方。** 填介质（$\varepsilon$ 增大）后同样场强下储能增大——这是高介电常数材料能储存更多能量的原因。下节《静电场的能量》会专门讨论电容储能的细节。

**数值算例（介质电容器的储能）**：平行板电容器极板面积 $S = 100$ cm²、间距 $d = 1$ mm，真空电容 $C_0 = \varepsilon_0S/d \approx 8.85$ pF。充至 $U = 100$ V 储能 $W_0 = \frac{1}{2}C_0U^2 \approx 4.4\times10^{-8}$ J。若板间填满 $\varepsilon_r = 4$ 的介质（保持电压不变），$C = \varepsilon_rC_0 = 35.4$ pF，储能 $W = \frac{1}{2}CU^2 \approx 1.77\times10^{-7}$ J——**储能变为 4 倍**，能量密度 $\frac{1}{2}\varepsilon E^2$ 因 $\varepsilon$ 增大而增大。这就是高介电常数材料（如钽电容的介质，$\varepsilon_r$ 可达数十）能储存更多能量的原因。

## 5 电位移矢量与真空中高斯定理的对照

| 比较项 | 真空高斯定理 | 介质中高斯定理 |
| --- | --- | --- |
| 场量 | $\boldsymbol{E}$ | $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ |
| 曲面内电荷 | 全部电荷 | 自由电荷 |
| 应用条件 | 真空 | 任意介质（线性/非线性均可定义） |
| 关系 | $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = \sum q/\varepsilon_0$ | $\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$ |

<span class="marginnote">两条高斯定理殊途同归：真空中 $\boldsymbol{D} = \varepsilon_0\boldsymbol{E}$，介质中 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$。$\boldsymbol{D}$ 的语言在第二十一章麦克斯韦方程组的微分形式（$\nabla\cdot\boldsymbol{D} = \rho_f$）中延续，是电磁场基本方程的核心场量之一。</span>

## 6 电位移与第二十一章的连接

**辨析｜易错点：$\boldsymbol{D}$ 是「辅助量」还是「物理场」？** 在静电场中 $\boldsymbol{D}$ 更多是计算工具；但在时变电磁场中，$\boldsymbol{D}$ 的时变部分 $\partial\boldsymbol{D}/\partial t$ 就是**位移电流密度**（第十四章），是麦克斯韦方程组的核心成员。$\boldsymbol{D}$ 由此从「静电场辅助量」升级为「电磁场基本场量」。

**与第二十一章的衔接**：麦克斯韦方程组的微分形式中，$\nabla\cdot\boldsymbol{D} = \rho_f$（高斯定理的微分版）与 $\nabla\times\boldsymbol{H} = \boldsymbol{J}_f + \partial\boldsymbol{D}/\partial t$（安培-麦克斯韦定律）都以 $\boldsymbol{D}$ 登场——介质中电磁学的整个体系建立在 $\boldsymbol{D}$ 与 $\boldsymbol{H}$ 这对「辅助场量」之上。<span class="marginnote">「D、H vs E、B」：电动力学的标准做法是用 $\boldsymbol{E}$、$\boldsymbol{B}$ 描述「真正的场」（进入洛伦兹力公式），用 $\boldsymbol{D}$、$\boldsymbol{H}$ 描述「介质中的场」（进入介质中的麦克斯韦方程）。搞清这两组的角色分工，是学习第二十一章电动力学的关键一步，也是「从极限到大模型」电磁学主线通向四大力学的接口。</span>

## 7 小结

- **电位移矢量**：$\boldsymbol{D} = \varepsilon_0\boldsymbol{E} + \boldsymbol{P} = \varepsilon\boldsymbol{E}$；源是自由电荷。
- **有介质时的高斯定理**：$\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$。
- 解题三步：选高斯面求 $\boldsymbol{D}$ → 用 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ 求 $\boldsymbol{E}$ → 必要时求束缚电荷。
- 平行板介质中：$D = \sigma_0$，$E = \sigma_0/(\varepsilon_0\varepsilon_r)$。
- **边界条件**：$\boldsymbol{D}$ 法向由自由电荷定，$\boldsymbol{E}$ 切向连续——两者互补。
- 能量密度：$w_e = \frac{1}{2}\boldsymbol{D}\cdot\boldsymbol{E} = \frac{1}{2}\varepsilon E^2$。
- $\boldsymbol{D}$ 是麦克斯韦方程组的基本场量（$\partial\boldsymbol{D}/\partial t$ = 位移电流），通向第二十一章。

在下一节，我们深入研究静电场的能量——**静电场的能量**，从电容器储能到能量密度。
