---
title: 近自由电子近似
date: 2026-08-07
---

# 近自由电子近似

<div class="epigraph">
<p>晶格周期势对自由电子的作用，是把动量空间切成了布里渊区。</p>
<footer>—— 鲁道夫 · 派尔斯（Rudolf Peierls）</footer>
</div>

<div class="article-byline">
<p>第四级 · 凝聚态物理 ｜ 黄昆《固体物理学》第4章、Ashcroft &amp; Mermin Ch.9 ｜ 2026-08-07</p>
</div>

## 为什么从近自由电子近似开始

自由电子模型解释了金属，却无法回答一个根本问题：**为什么铜导电而硅不导电？
** 答案藏在「周期势」里。
晶体中电子感受到的势 $V(\mathbf{r})$ 不是常数——它随晶格周期起伏。
**近自由电子近似（NFE）** 把周期势当作微扰加在自由电子气上，是连接「自由电子」与「真实能带」的第一座桥。<span class="marginnote">为什么值得专门学 NFE？
因为它是<strong>能隙概念最清晰的诞生地</strong>。
周期势在布里渊区边界把自由电子抛物线劈开成两条，中间出现<strong>禁带</strong>——这个「劈开」就是绝缘体与半导体的全部奥秘。
NFE 用最少的数学讲清了这件事。</span>

本节先立布洛赫定理这个总纲，再在弱周期势极限下做微扰，看能隙如何出现。

## 1 布洛赫定理

**布洛赫定理（Bloch theorem）**：周期势 $V(\mathbf{r} + \mathbf{R}_n) = V(\mathbf{r})$ 中，电子波函数可以写成：

$$\psi_{\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}}\, u_{\mathbf{k}}(\mathbf{r})$$

其中 $u_{\mathbf{k}}(\mathbf{r})$ 与晶格同周期：$u_{\mathbf{k}}(\mathbf{r} + \mathbf{R}_n) = u_{\mathbf{k}}(\mathbf{r})$。

**波函数 = 平面波 × 周期包络**。这个结构直接继承自晶格的平移对称性：沿格矢平移一个格点，波函数只改变相位 $e^{i\mathbf{k}\cdot\mathbf{R}_n}$。<span class="marginnote">布洛赫定理是凝聚态物理的「动量守恒律」：周期系统里，波矢 $\mathbf{k}$ 是<strong>好量子数</strong>——每个能带态都能用 $\mathbf{k}$ 标记。它与「倒格子与布里渊区」一节完全衔接：$\mathbf{k}$ 以倒格矢 $\mathbf{G}$ 为周期，$\psi_{\mathbf{k}+\mathbf{G}}$ 与 $\psi_{\mathbf{k}}$ 只是同一状态的另一种写法。</span>

布洛赫定理的两个直接推论：

- **能带**：$E_n(\mathbf{k})$ 是 $\mathbf{k}$ 的连续函数，量子数 $n$ 标记能带编号；
- **晶体动量**：$\hbar\mathbf{k}$ 是「准动量」，电子受外力时 $\mathbf{k}$ 按 $\hbar\frac{d\mathbf{k}}{dt} = \mathbf{F}$ 演化，但在布里渊区边界会被「折回」。

## 2 公式解析：弱周期势的微扰

NFE 把哈密顿量写成自由项 + 微扰：

$$H = \frac{-\hbar^2}{2m}\nabla^2 + V(\mathbf{r}), \qquad V(\mathbf{r}) = \sum_{\mathbf{G}} V_{\mathbf{G}}\, e^{i\mathbf{G}\cdot\mathbf{r}}$$

周期势展成倒格矢 $\mathbf{G}$ 的傅里叶级数，系数 $V_{\mathbf{G}}$ 是势的傅里叶分量。

**第一步，零阶波函数**：取布洛赫波的平面波部分 $\psi_{\mathbf{k}}^0 = \frac{1}{\sqrt{V}}e^{i\mathbf{k}\cdot\mathbf{r}}$，能量 $E^0 = \hbar^2 k^2/2m$。

**第二步，矩阵元**：微扰 $V$ 在平面波基矢 $\langle \mathbf{k}'|V|\mathbf{k}\rangle$ 中，只有当 $\mathbf{k}' - \mathbf{k} = \mathbf{G}$ 时非零：

$$\langle \mathbf{k} + \mathbf{G}|V|\mathbf{k}\rangle = V_{\mathbf{G}}$$

**这是核心机制**：周期势只在「相差一个倒格矢」的态之间耦合。对大多数 $\mathbf{k}$，$V_{\mathbf{G}}$ 的能量尺度远小于动能差，微扰只是把抛物线**微微推高**（二阶效应），能带几乎还是自由电子的样子。

**第三步，简并处：能隙**。但当动能接近简并时，即：

$$\frac{\hbar^2|\mathbf{k}+\mathbf{G}|^2}{2m} \approx \frac{\hbar^2 k^2}{2m}$$

微扰不再小。
这就是**布里渊区边界** $\mathbf{k} \cdot \mathbf{G} = -|\mathbf{G}|^2/2$。
两个态近乎简并，需用简并微扰论——2×2 矩阵对角化给出：

$$E_\pm = \bar{E} \pm \sqrt{\left(\frac{\hbar^2}{2m}\frac{|\mathbf{k}+\mathbf{G}|^2 - k^2}{2}\right)^2 + |V_{\mathbf{G}}|^2}$$

在精确边界处动能相等，能量劈裂为：

$$E_+ - E_- = 2|V_{\mathbf{G}}|$$

**能隙等于周期势傅里叶分量模长的两倍**。这个简单公式是整个能带理论的第一块基石。<span class="marginnote">物理图像：在布里渊区边界，入射波 $e^{ikx}$ 与布拉格反射波 $e^{i(k-G)x}$ 发生<strong>驻波混合</strong>——像弦上的驻波，一列波峰落在势谷（能量降低），一列波峰落在势峰（能量升高），两列驻波能量不同，于是劈开成两条能带，中间是 $2|V_G|$ 的禁带。<strong>能隙 = 电子在周期势里的布拉格衍射</strong>，与 X 射线衍射同出一源。</span>

## 3 能隙处的驻波图像

在边界 $\mathbf{k} = \mathbf{G}/2$ 处，两个简并态组合成驻波：

$$\psi_+ \propto e^{i\mathbf{G}\cdot\mathbf{r}/2} + e^{-i\mathbf{G}\cdot\mathbf{r}/2} \propto \cos\frac{\mathbf{G}\cdot\mathbf{r}}{2}, \qquad \psi_- \propto \sin\frac{\mathbf{G}\cdot\mathbf{r}}{2}$$

- $\psi_+$：电荷密度集中在**势能低**的位置（离子实附近），能量降低；
- $\psi_-$：电荷密度集中在**势能高**的位置（离子实之间），能量升高。

**两个驻波把电子密度重新分配，一个讨好势、一个对抗势**——劈开成能隙的上下两支。<span class="marginnote">这是「能隙」最直观的解释：<strong>不是电子被「禁止」在某个能量存在，而是波函数的边界条件（布拉格反射）强制把两个简并态劈开</strong>。上下两支分别对应「在势谷安家」与「在势峰受苦」的电子。</span>

## 4 一维的完整能带图

一维晶格（周期 $a$），布里渊区 $[-\pi/a, \pi/a]$。
NFE 给出：

大部分 $\mathbf{k}$：$E \approx \hbar^2 k^2/2m$，自由电子抛物线；
边界 $k = \pm\pi/a$：抛物线被劈开，出现能隙 $2|V_G|$；
能带以 $2\pi/a$ 为周期重复。

**重点：能隙在布里渊区边界处打开**，边界正是 $\mathbf{k}$ 满足布拉格条件 $\mathbf{k}\cdot\mathbf{G} = |\mathbf{G}|^2/2$ 的地方。所以「**能隙在哪」完全由倒格子几何决定**，与势的强度无关（势只决定能隙**大小** $2|V_G|$）。

**辨析｜易错点：**NFE 假设 $|V_{\mathbf{G}}| \ll$ 动能，对**碱金属**（弱周期势）成立；但对**过渡金属 d 电子、共价半导体**，周期势根本不弱，NFE 失效，需要紧束缚近似（下一节）。**NFE 不是万能方法，它是「周期势弱」极限下的正确方法**——教材用它讲清能隙机制，工程上则要按材料选择方法。<span class="marginnote">判断该用 NFE 还是紧束缚：<strong>电子「接近自由」用 NFE（能带接近抛物线，只在边界开口）</strong>；<strong>电子「贴近原子」用紧束缚（能带窄，源于原子轨道跃迁）</strong>。同一种材料，s、p 传导电子可能接近自由，d、f 电子则高度局域——一铜身上两种图像并存，这也是凝聚态计算的一大主题。</span>

## 5 NFE 的成就与边界

**成就**：

- 解释了**能隙的存在**与**金属/绝缘体之别**的微观起源；
- 解释了费米面的折叠与畸变（上一节）；
- 解释了为什么碱金属的能带接近抛物线——周期势弱，微扰很小。

**边界**：

- 强周期势、窄能带材料失效；
- 无法描述局域 d/f 电子、磁性、强关联；
- 对绝缘体只能给出「能隙大小」，不能给出能带细节。

## 6 小结

- **布洛赫定理**：$\psi_{\mathbf{k}} = e^{i\mathbf{k}\cdot\mathbf{r}}u_{\mathbf{k}}$，$\mathbf{k}$ 是好量子数，能带 $E_n(\mathbf{k})$ 存在。
- **周期势只耦合相差一个倒格矢 $\mathbf{G}$ 的态**：矩阵元 $V_{\mathbf{G}}$ 是开关。
- **简并微扰**在布里渊区边界打开能隙，能隙 $2|V_{\mathbf{G}}|$ 等于周期势傅里叶分量模长的两倍。
- 能隙的驻波图像：电荷密度重新分布，一降一升。
- NFE 适用于**弱周期势**（碱金属）；强局域电子需紧束缚近似。

在下一节，我们从另一个极端出发——假设电子「紧贴原子」——写**紧束缚近似**，它天然适合半导体与过渡金属。
