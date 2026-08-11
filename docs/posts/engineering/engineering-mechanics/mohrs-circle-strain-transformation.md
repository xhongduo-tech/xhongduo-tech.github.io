---
title: Mohr 应力圆与应变转换：二维应力状态的几何
date: 2026-08-11
---

# Mohr 应力圆与应变转换：二维应力状态的几何

<div class="epigraph">
<p>数学是打开科学大门的钥匙。</p>
<footer>—— 弗朗西斯·培根（Francis Bacon）</footer>
</div>

<div class="article-byline">
<p>第六级 · 工程技术 · 工程力学（理论/材料/结构力学） ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要一个「圆」

上一节我们反复把问题「收敛」到同一件事：**给定一点的应力状态，怎么求任意方向截面上的应力、主应力与最大剪应力**。这一节把这件事彻底解决，而且用一种漂亮的方式——**Mohr 应力圆**。<span class="marginnote">奥托·莫尔（Otto Mohr，1835—1918）在 1882 年提出：平面应力状态的变换公式本质上描述的是一个圆。把代数的变换公式换成几何的一张圆图，工程界从此可以「画图求解」应力变换——这是力学史上「以图代算」的经典一课。</span>

同一点在不同方向的截面上，正应力与剪应力取不同的值。这听起来不可思议，但事实如此：**应力状态属于「点」而非「面」**——同一个应力状态，换个截面看，数字就变了。Mohr 圆把「这个点的全部截面信息」装进一张圆，圆心是平均正应力，半径是「最大剪应力」。

## 1 平面应力状态的变换公式

设某点处于**平面应力状态（plane stress）**，已知 $\sigma_x$、$\sigma_y$、$\tau_{xy}$。取一个法线与 $x$ 轴成 $\theta$ 角的斜截面（逆时针为正），该截面上的正应力与剪应力为：

$$\sigma_{x'} = \frac{\sigma_x + \sigma_y}{2} + \frac{\sigma_x - \sigma_y}{2}\cos 2\theta + \tau_{xy}\sin 2\theta$$

$$\tau_{x'y'} = -\frac{\sigma_x - \sigma_y}{2}\sin 2\theta + \tau_{xy}\cos 2\theta$$

这一对公式就是应力变换的全部代数内容。<span class="marginnote">推导的关键是「平衡」：把斜截面切出的三角形微元列出力的平衡，利用几何关系把 $\sigma_{x'}$、$\tau_{x'y'}$ 用 $\sigma_x$、$\sigma_y$、$\tau_{xy}$、$\theta$ 表出。三角函数里的 $2\theta$ 是个奇特但本质的特征：应力是张量，转 $180^\circ$ 才回到自身，所以变换里处处出现 $2\theta$。</span>

**主应力（principal stress）**：使斜截面上剪应力为零的方向所对应的正应力，就是主应力 $\sigma_1$、$\sigma_2$（$\sigma_1 \ge \sigma_2$）。主应力所在的截面称为**主平面**。主应力与主平面方位由下式给出：

$$\sigma_{1,2} = \frac{\sigma_x + \sigma_y}{2} \pm \sqrt{\left(\frac{\sigma_x - \sigma_y}{2}\right)^2 + \tau_{xy}^2}$$

$$\tan 2\theta_p = \frac{2\tau_{xy}}{\sigma_x - \sigma_y}$$

对应的**最大平面内剪应力（max in-plane shear stress）**为

$$\tau_{\max} = \sqrt{\left(\frac{\sigma_x - \sigma_y}{2}\right)^2 + \tau_{xy}^2} = \frac{\sigma_1 - \sigma_2}{2}$$

主应力方向面上剪应力为零，最大剪应力面与主平面成 $45^\circ$——这一条在工程里极常用（断裂面、剪切面常沿 45° 出现）。

## 2 Mohr 圆：把变换公式画成圆

观察变换公式：令横轴为 $\sigma$、纵轴为 $\tau$，圆心在 $(\sigma_{\text{avg}}, 0)$ 处，其中 $\sigma_{\text{avg}} = (\sigma_x + \sigma_y)/2$；半径 $R = \sqrt{(\frac{\sigma_x - \sigma_y}{2})^2 + \tau_{xy}^2}$。变换公式恰好是圆的参数方程——**一切平面应力状态都对应一个圆**。

![平面应力状态的 Mohr 圆](/images/engineering-mechanics/mohrs-circle-1.svg)

在 Mohr 圆上读信息，只需几个「对应规则」：

- **圆上每一点代表一个方向的截面**：从点 $(\sigma_x, \tau_{xy})$ 出发，在圆上转过 $2\theta$ 角，就到达法线转过 $\theta$ 的那个截面。
- **主应力**是圆与 $\sigma$ 轴的交点：$\sigma_1 = \sigma_{\text{avg}} + R$，$\sigma_2 = \sigma_{\text{avg}} - R$。
- **最大剪应力**等于半径 $R$。
- **转过的圆心角是截面角的 2 倍**——这是用圆的最大原因：几何上的「倍角」让作图直观。<span class="marginnote">「圆心角 $2\theta$」是 Mohr 圆最重要的约定，也是初学者最容易出错的点：应力状态旋转 $\theta$，圆上对应旋转 $2\theta$。想找转 $45^\circ$ 的截面，就在圆上转 $90^\circ$。反复默念「转角 ×2」能省去大半作图错误。</span>

**解析｜易错点：** 画圆前先定**符号约定**：通常约定剪应力「使微元顺时针转」为正，画在纵轴上方。若约定相反，圆上的点会对纵轴对称翻转。更隐蔽的坑是**旋转方向**：$\theta$ 逆时针为正时，圆上要按 $2\theta$ 反向旋转。先把符号约定写在纸上，再动笔画图。

## 3 三向应力与绝对最大剪应力

真实构件有时处于**三向应力状态（triaxial）**，如厚壁压力容器内壁、滚珠轴承接触区。此时有三个主应力 $\sigma_1 \ge \sigma_2 \ge \sigma_3$，可画出**三个 Mohr 圆**。工程上最重要的结论是：

**绝对最大剪应力（absolute maximum shear stress）**为

$$\tau_{\text{abs max}} = \frac{\sigma_1 - \sigma_3}{2}$$

它由**最大与最小主应力之差**决定，中间主应力 $\sigma_2$ 不参与。这一条在《失效理论》里是 Tresca 判据的直接基础。<span class="marginnote">「中间主应力不参与最大剪应力」是三向应力状态里最反直觉、也最有用的结论：设计时只要抓住 $\sigma_1$ 与 $\sigma_3$ 两个极端值。压力容器内壁处 $\sigma_3 = -p$（内压），这就是壁厚设计公式里内压直接进入的原因。</span>

## 4 应变转换与应变花

应力状态可以转，应变状态同样可以。二维应变状态 $\varepsilon_x$、$\varepsilon_y$、$\gamma_{xy}$ 的变换公式与应力变换公式**同构**（把 $\sigma$ 换 $\varepsilon$、$\tau$ 换 $\gamma/2$）：

$$\varepsilon_{x'} = \frac{\varepsilon_x + \varepsilon_y}{2} + \frac{\varepsilon_x - \varepsilon_y}{2}\cos 2\theta + \frac{\gamma_{xy}}{2}\sin 2\theta$$

所以应变也有自己的「Mohr 圆」，且主应变方向与主应力方向一致（各向同性线弹性材料）。<span class="marginnote">「应力变换 vs 应变变换」同构，本质是胡克定律的线性性：应力张量与应变张量成线性映射，旋转（坐标变换）与线性映射可交换——这正是《线性代数》里「张量」的语言。到研究应变时，请回看一遍线性代数里的基变换，会发现这里是同一件事的力学化身。</span>

工程实测应变常用**应变花（strain rosette）**：在同一位置按 $45^\circ$ 或 $60^\circ$ 方向粘贴三片应变片，测得 $\varepsilon_{0^\circ}$、$\varepsilon_{45^\circ}$、$\varepsilon_{90^\circ}$，解出 $\varepsilon_x$、$\varepsilon_y$、$\gamma_{xy}$，再经广义胡克定律换算出该点的应力状态。**材料的真实应力，往往是这么「量出来的」而不是「算出来的」**——设计给的是名义值，实测给的是真实值，二者互相校核是工程规范的标准动作。

## 5 公式解析：主应力公式

主应力公式 $\sigma_{1,2} = \sigma_{\text{avg}} \pm R$ 是这一节的核心，拆解它的四步：

**第一步，先求平均正应力** $\sigma_{\text{avg}} = (\sigma_x + \sigma_y)/2$：这是「旋转不变量」——无论截面怎么转，正应力的平均值不变，它就是 Mohr 圆的**圆心**。<span class="marginnote">「平均正应力不变」在物理上是「静水压力部分」：改变截面只会在平均应力附近重新分配正/剪应力，而均值是应力状态内在的量。主应力 $\sigma_1 = \sigma_{\text{avg}} + R$ 也可以读成「平均应力 + 偏离量」。</span>
- **第二步，求偏离半径** $R = \sqrt{(\frac{\sigma_x - \sigma_y}{2})^2 + \tau_{xy}^2}$：它度量「该截面应力状态与平均状态的偏离」——$R$ 越大，最大剪应力越大。纯剪切时（$\sigma_x = -\sigma_y$，$\sigma_{\text{avg}} = 0$），$R = \tau_{xy}$，主应力为 $\pm \tau$，恰是「纯剪应力等价于等值拉压」这一著名结论。
- **第三步，相加取主应力**：$\sigma_1 = \sigma_{\text{avg}} + R$、$\sigma_2 = \sigma_{\text{avg}} - R$。圆与 $\sigma$ 轴的两个交点就是主应力。
- **第四步，换成物理直觉**：把公式倒着用——已知主应力，最大剪应力就是半径 $\tau_{\max} = (\sigma_1 - \sigma_2)/2$，主平面与最大剪应力面相差 $45^\circ$。这两条口诀能直接回答绝大多数工程问题。

## 6 小结

- 同一点**不同截面应力不同**：应力状态属于点，用变换公式或 **Mohr 圆**描述全部截面。
- **Mohr 圆**：圆心 $\sigma_{\text{avg}}$，半径 $R$；圆上点对应截面，圆心角是截面角的 **2 倍**。
- **主应力** $\sigma_{1,2} = \sigma_{\text{avg}} \pm R$（剪应力为零的截面上）；**最大剪应力** $\tau_{\max} = R = (\sigma_1 - \sigma_2)/2$，与主平面成 $45^\circ$。
- **三向应力**：绝对最大剪应力 $\tau_{\text{abs max}} = (\sigma_1 - \sigma_3)/2$，中间主应力不参与。
- **应变转换与应力转换同构**；工程用**应变花**实测应变反推应力。

在下一节，我们手里终于有了「任意应力状态的主应力」这把钥匙。现在可以回答那个最尖锐的问题：**材料到底在什么条件下失效**——这就是《失效理论（von Mises/Tresca/Mohr）》的内容。
