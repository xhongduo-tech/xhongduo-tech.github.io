---
title: 导体中的电磁波与趋肤效应
date: 2026-08-07
---

# 导体中的电磁波与趋肤效应

<div class="epigraph">
<p>电磁波进了金属，就只剩下一层皮。</p>
<footer>—— 奥利弗 · 亥维赛（Oliver Heaviside）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第四章 §4.4 ｜ 2026-08-07</p>
</div>

## 为什么导体会「吃掉」电磁波

前几节的介质都是绝缘体（无自由电流），电磁波能长驱直入。导体里有大量自由电子，电磁波一旦进入，电场就驱动自由电子形成传导电流，电流的焦耳热把波的能量耗散掉——所以电磁波在导体中**指数衰减**，只存在很薄的一层。这一节引入**复介电常数**的语言，把「导体中的波动方程」统一到亥姆霍兹方程框架里，并得到工程上极其重要的**趋肤深度**。<span class="marginnote">金属「不透明」的本质不是挡光，而是吸收：光进入金属后在一个波长以内就被电子电流耗散殆尽。这解释了为什么金属有光泽（反射率极高，表面立刻把波「弹」回去）、为什么金属不能做成天线以外的无线传输介质。</span>

## 1 复介电常数与导体的波动方程

导体中，自由电荷满足欧姆定律 $\mathbf{J} = \sigma\mathbf{E}$。时谐场的安培-麦克斯韦方程为

$$\nabla\times\widetilde{\mathbf{H}} = \sigma\widetilde{\mathbf{E}} - i\omega\varepsilon\widetilde{\mathbf{E}} = -i\omega\left(\varepsilon + i\frac{\sigma}{\omega}\right)\widetilde{\mathbf{E}}$$

括号里的量定义了**复介电常数（complex permittivity）**：

$$\varepsilon_c = \varepsilon + i\frac{\sigma}{\omega}$$

于是导体中的波动方程与介质中的形式完全相同，只是 $\varepsilon \to \varepsilon_c$。定义**复波数**

$$\tilde{k} = \omega\sqrt{\mu\varepsilon_c} = \omega\sqrt{\mu\varepsilon\left(1 + i\frac{\sigma}{\omega\varepsilon}\right)}$$

导体中 $\sigma$ 很大，$\sigma/(\omega\varepsilon) \gg 1$（良导体条件），此时

$$\tilde{k} \approx \sqrt{i\mu\sigma\omega} = (1+i)\sqrt{\frac{\mu\sigma\omega}{2}}$$

记 $k = \sqrt{\mu\sigma\omega/2}$，则 $\tilde{k} = (1+i)k$。<span class="marginnote">把 $\varepsilon$ 换成 $\varepsilon_c$ 是「统一语言」的胜利：绝缘体、有损介质、导体全都装进同一个亥姆霍兹方程，差别只是 $\varepsilon_c$ 的实部与虚部之比。良导体条件 $\sigma \gg \omega\varepsilon$ 表明「传导电流主导」；绝缘体中 $\sigma \to 0$，回到无损传播。</span>

## 2 趋肤效应与趋肤深度

**趋肤效应（skin effect）**：电磁波进入导体后，电场、磁场与电流密度都按指数衰减：

$$\widetilde{\mathbf{E}}(z) = \mathbf{E}_0 e^{-z/\delta} e^{ikz}$$

其中**趋肤深度（skin depth）**

$$\delta = \frac{1}{\operatorname{Im}\tilde{k}} = \sqrt{\frac{2}{\mu\sigma\omega}}$$

它表示场衰减到表面值的 $1/e$（约 37%）所深入的厚度。场在导体中既衰减又振荡（$e^{ikz}$ 振荡项），两者由同一个 $k$ 决定——衰减与振荡是复波数的实部与虚部。<span class="marginnote">趋肤深度的物理图像：感应涡流在表面最强，往内指数减弱。良导体的 $\delta$ 很小——铜在 50 Hz 工频下 $\delta \approx 9.4\ \mathrm{mm}$，在 1 GHz 微波下只有约 $2\ \mu\mathrm{m}$。频率越高，电流越贴表。</span>

**典型数值**：铜（$\sigma = 5.8\times10^7\ \mathrm{S/m}$）的趋肤深度：50 Hz 约 $9.4\ \mathrm{mm}$、1 kHz 约 $2.1\ \mathrm{mm}$、1 MHz 约 $66\ \mu\mathrm{m}$、1 GHz 约 $2.1\ \mu\mathrm{m}$。正比于 $1/\sqrt{\omega}$。

**辨析｜易错点：** 趋肤深度依赖频率——「高频电流集中在表面」不代表「直流也有趋肤效应」。直流时 $\omega = 0$，$\delta \to \infty$，电流均匀分布。趋肤效应的实用后果：**高频导线的有效截面积远小于几何截面积**，电阻随频率增大而增大（交流电阻 > 直流电阻）。

## 3 公式解析：为什么良导体的波数是 $(1+i)\sqrt{\mu\sigma\omega/2}$

这一步是整个趋肤理论的枢纽，拆开：

- **第一步，良导体近似**：$\sigma/(\omega\varepsilon) \gg 1$ 时，复介电常数中虚部主导：$\varepsilon_c \approx i\sigma/\omega$。于是 $\tilde{k} = \omega\sqrt{\mu\varepsilon_c} \approx \omega\sqrt{\mu\sigma i/\omega} = \sqrt{i\mu\sigma\omega}$。
- **第二步，开方**：$i$ 的平方根是 $(1+i)/\sqrt{2}$（因为 $[(1+i)/\sqrt2]^2 = (1+2i+i^2)/2 = i$）。故 $\tilde{k} = (1+i)\sqrt{\mu\sigma\omega/2}$。实部与虚部相等——衰减常数与相位常数都等于 $k = \sqrt{\mu\sigma\omega/2}$。<span class="marginnote">「衰减与相位共用同一个常数」是良导体的特征：在导体中，波传播一个波长就衰减 $2\pi$ 倍，几乎没有任何穿透深度意义上的「波」存在。这也解释了为什么金属内部的电磁场在工程上总被忽略——它衰减得太快。</span>
- **第三步，读出趋肤深度**：场 $\propto e^{-\operatorname{Im}\tilde{k}z} = e^{-kz}$，所以 $\delta = 1/k = \sqrt{2/(\mu\sigma\omega)}$。**结论：$\delta$ 反比于 $\sqrt{\omega}$、$\sqrt{\sigma}$、$\sqrt{\mu}$**——频率越高、电导率越大、磁导率越大，电流越集中表面。<span class="marginnote">磁导率的影响有趣：铁磁材料 $\mu$ 大，趋肤深度反而小——所以变压器铁芯（高 $\mu$ 硅钢）更要叠片来切碎涡流路径，每片厚度远小于 $\delta$，把涡流损耗压到最低。</span>

## 4 导体表面的边界条件与表面阻抗

导体内部场衰减极快，工程上常用**边界条件近似**：把导体内部当作「场为零」的理想导体（PEC）来处理，电磁波在导体表面几乎全反射。真实的良导体反射率接近 100% 但不等于 100%——剩余的透射就是趋肤效应与欧姆损耗。

用**表面阻抗（surface impedance）**$Z_s$ 统一描述导体表面对波的响应：

$$Z_s = \frac{E_t}{H_t} = (1+i)\sqrt{\frac{\mu\omega}{2\sigma}} = (1+i)\frac{1}{\sigma\delta}$$

它的实部与虚部相等（良导体），实部是**表面电阻**（每单位长度的欧姆损耗），虚部是表面电抗。这个量是微波工程、天线、传输线损耗分析的核心参数。<span class="marginnote">表面阻抗的妙处：把「趋肤效应导致的高频损耗」全部打包进一个复数参数，电路工程师把它当「频率相关的电阻」用。$R_s = 1/(\sigma\delta)$ 随 $\sqrt{\omega}$ 增长，所以高频导线的损耗怎么都压不下去——这就是为什么射频电路用镀银、镀金的道理（银的电导率最高）。</span>

## 5 趋肤效应的工程对策

- **空心导线 / 多股线（Litz wire）**：高频电流集中在表面，实心导体的中心部分是「死重」——改用空管或多股细线并绕，等效增加表面积，降低交流电阻。
- **叠片铁芯**：变压器、电机铁芯用互相绝缘的薄硅钢片叠成，每片厚度（约 0.3–0.5 mm）远小于该频率下的 $\delta$，把涡流路径切碎，涡流损耗与片厚的平方成正比地下降。
- **屏蔽（skin-depth 屏蔽）**：金属屏蔽层对高频电磁波的良好屏蔽效果来自趋肤效应——波被表面反射并吸收，穿透厚度远小于屏蔽层厚度。
- **感应加热与淬火**：利用涡流在表层集中的特性，对金属表面进行快速加热淬火，芯部仍保持韧性的工艺。<span class="marginnote">「趋肤效应是祸是福取决于场景」：对输电是祸（交流损耗大），对屏蔽与感应加热是福（能量被锁在表面）。同一个物理，工程上可以顺着用、也可以对着用。</span>

**辨析｜易错点：** 理想导体（$\sigma \to \infty$）中 $\delta \to 0$，场完全不能进入，边界条件简化为「表面 $\mathbf{E}_t = 0$、$\mathbf{H}_t = \mathbf{K}$（面电流）」。但**没有真正的理想导体**——超导体只是 $\sigma$ 极大（且更本质地是 $\mathbf{B}$ 被迈斯纳效应排斥），也不是理想导体。把「良导体」当「理想导体」只在损耗可忽略时成立。

## 6 趋肤效应的完整计算：同轴电缆的交流电阻

把趋肤深度用到实际的工程计算里，才能体会它「不是理论玩具」。

**问题**：同轴电缆内导体半径 $a$、电导率 $\sigma$、长 $l$，工作在频率 $\omega$。求高频时的交流电阻（相对直流的增大）。

**第一步，直流电阻。** $R_{\text{dc}} = \dfrac{l}{\sigma\pi a^2}$——电流均匀分布，面积 $\pi a^2$。

**第二步，高频时的有效面积。** 当 $\delta \ll a$（趋肤深度远小于半径，高频典型成立）时，电流集中在半径 $a$ 附近的薄壳内，有效导电面积约为 $2\pi a\delta$。交流电阻：

$$R_{\text{ac}} \approx \frac{l}{\sigma\cdot 2\pi a\delta} = \frac{l}{2\pi a\sigma}\sqrt{\frac{\mu\sigma\omega}{2}}$$

**第三步，对比**：$\dfrac{R_{\text{ac}}}{R_{\text{dc}}} = \dfrac{\pi a^2}{2\pi a\delta} = \dfrac{a}{2\delta}$。**当 $a \gg \delta$ 时，交流电阻比直流大 $a/2\delta$ 倍**。对 1 GHz 铜导线（$\delta \approx 2\ \mu\mathrm{m}$），若 $a = 1\ \mathrm{mm}$，交流电阻是直流的 250 倍！**这就是为什么射频传输线要用镀银（提高 $\sigma$）、用空心或绞合结构（提高有效面积）**。

**从电阻看能量：** 交流电阻的本质是「趋肤层内焦耳热」。表面阻抗 $Z_s = (1+i)/(\sigma\delta)$ 的实部 $R_s = 1/(\sigma\delta)$ 直接给出每单位表面的损耗功率密度 $p = \frac{1}{2}R_s|K|^2$（$K$ 为面电流密度）。**「损耗 = 面电流平方 × 表面电阻」是微波工程里最常用的功耗公式**，屏蔽效能、天线效率、腔体 Q 值的计算都从这里出发。

**辨析｜易错点：** ① $\delta \ll a$ 是薄壳近似的前提，低频时 $\delta$ 可与 $a$ 同量级，公式失效，需用完整的贝塞尔函数解。② 趋肤深度公式 $\delta = \sqrt{2/(\mu\sigma\omega)}$ 中 $\mu$ 是导体的绝对磁导率——铁磁导体 $\mu$ 大，$\delta$ 小，交流电阻更大。③ 「交流电阻」不是欧姆定律的「电阻」，它是**等效损耗电阻**，包含焦耳热损耗但不包含辐射损耗——把它当成真实电阻直接套 $P = I^2R$ 会在辐射场里出错。

**电磁屏蔽的趋肤深度逻辑**：金属外壳对高频电磁波的屏蔽效果，来自「波在金属表层被反射 + 趋肤层内吸收」两道防线。屏蔽效能（SE）正比于「反射损耗 + 吸收损耗」，其中吸收损耗 $A \approx 8.69\dfrac{t}{\delta}\ \mathrm{dB}$（$t$ 为屏蔽层厚度）。**要屏蔽 1 GHz 的信号，铜只需要约 $6\ \mu\mathrm{m}$ 的厚度就给出 20 dB 的衰减**——这就是为什么电子设备只要薄薄一层金属壳就能过电磁兼容（EMC）测试。趋肤深度直接决定屏蔽层的最小厚度。

**法拉第笼的「频率限制」**：经典法拉第笼能屏蔽静电场与低频场（电场被导体表面自由电荷重新分布完全抵消），但**频率越高、趋肤深度越小，屏蔽越强**；反过来，极低频磁场（如工频 50 Hz）的趋肤深度很大（铜约 9 mm），需要更厚的屏蔽或高磁导率材料。**「法拉第笼屏蔽一切」是误区——它屏蔽的是「电场」和「高频电磁波」，对静磁场与极低频磁场效果有限。**

**导体中的色散与群速度**：导体中的波数 $\tilde{k} = (1+i)/\delta$ 是纯复数（实部虚部相等），意味着**导体中不存在真正的「传播波」，只有衰减振荡**。相速度 $v_p = \omega/\operatorname{Re}\tilde{k} = \omega\delta$ 与频率的平方根成正比，群速度 $v_g = \mathrm{d}\omega/\mathrm{d}k = 2\omega\delta$——**导体是强色散介质**，任何波包进入导体都会迅速展宽并衰减。

**表面波与等离子体**：金属中自由电子气的集体振荡形成**等离子体振荡**。当电磁波频率高于金属的等离子体频率 $\omega_p = \sqrt{ne^2/(\varepsilon_0 m)}$ 时，金属变得「透明」（波可以传播）；低于 $\omega_p$ 时波被反射——**这就是金属「亮」与「不透明」的统一解释**。紫外光（高于 $\omega_p$）能穿过薄金属，可见光被反射，铝箔在紫外下透光，就是等离子体频率在起作用。

**辨析｜易错点：** ① 「导体中波数纯虚数」的结论依赖良导体近似 $\sigma \gg \omega\varepsilon$；对半导体、损耗介质，$\tilde{k}$ 的实虚部都非零，波「边传播边衰减」。② 等离子体频率讨论的是「自由电子气」，与德鲁德模型相连；高频下金属的介电常数 $\varepsilon(\omega) = \varepsilon_0(1 - \omega_p^2/\omega^2)$ 可以是负的——负介电常数是许多超材料设计的物理基础。

## 7 小结

- 导体的效应并入**复介电常数** $\varepsilon_c = \varepsilon + i\sigma/\omega$，波动方程形式不变。
- 良导体中 $\tilde{k} = (1+i)\sqrt{\mu\sigma\omega/2}$——衰减与相位常数相等。
- **趋肤深度** $\delta = \sqrt{2/(\mu\sigma\omega)}$，正比 $1/\sqrt{\omega}$；高频电流集中于表面。
- **表面阻抗** $Z_s = (1+i)/(\sigma\delta)$，描述高频损耗。
- 工程对策：空心导线、Litz 线、叠片铁芯、金属屏蔽、感应加热。

在下一节，我们终于回答「波从哪里来」：变化的电流如何在空间中激发出电磁波——**电磁波的辐射（偶极辐射）**。
