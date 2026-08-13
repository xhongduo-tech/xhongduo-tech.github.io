---
title: Goemans–Williamson 最大割算法
date: 2026-08-07
---

# Goemans–Williamson 最大割算法

<div class="epigraph">
<p>0.878567——一个从三角不等式的缝隙里挤出来的常数，它让最大割的近似从 1/2 跃升到 0.878，并开启了一个「半定规划 + 随机超平面」的时代。</p>
<footer>—— 米歇尔 · 戈曼斯（Michel Goemans）与大卫 · 威廉森</footer>
</div>

<div class="article-byline">
<p>第三级 · 近似算法 ｜ Williamson & Shmoys, *The Design of Approximation Algorithms\*, Ch.6 ｜ 2026-08-07</p>
</div>

## 为什么从 Goemans–Williamson 开始

最大割（max-cut）在本专题第一篇就出过场：
随机抛硬币给出 2-近似（即 0.5 因子），局部搜索也停在 0.5。
**1995 年，Goemans 与 Williamson 用一个半定规划 + 随机超平面把 0.5 一举推到 0.878**——从此「最大割 ≈ 0.878」成为教科书标准。这不是一次普通的改进：它证明了「向量几何」能捕捉「成对冲突」的全部微妙结构，也把近似算法的技术重心从「线性」拨向「半定」。

这个算法值得单独立传，因为它浓缩了 SDP 近似方法的全部元素：
**松弛（向量规划）→ 舍入（随机超平面）→ 三角恒等式（0.878 常数）**。
读懂它，你就读懂了后续一切 SDP 算法的模板。
而它的紧性（在 Unique Games 猜想下 0.878 是最优）更让它成为现代近似复杂性的基石。

## 1 向量规划与随机超平面：算法

**最大割**：图 $G=(V,E)$，把顶点分到 $\{+1,-1\}$，最大化跨割边数。

**第一步，松弛成向量规划。** 给每个顶点 $i$ 赋一个单位向量 $v_i \in \mathbb{R}^n$，目标是：

$$
\max \sum_{(i,j)\in E} \frac{1 - \langle v_i, v_j\rangle}{2} \quad \text{s.t.} \quad \|v_i\| = 1
$$

当向量限制为 $v_i \in \{+1,-1\}$ 时，$v_i = -v_j$ 表示异侧（贡献 1）、$v_i = v_j$ 表示同侧（贡献 0）——**这就是整数最大割的精确重写**。放宽成任意单位向量，SDP 可多项式求解。

**第二步，随机超平面舍入。** 取随机向量 $r \sim N(0, I_n)$（各坐标独立标准正态），顶点 $i$ 归 +1 当且仅当 $\langle v_i, r\rangle \ge 0$。<span class="marginnote">为什么用高斯向量做超平面？因为 $v_i\cdot r$ 的符号在球面上是「均匀随机方向」——超平面的法向在球面均匀分布，任何一对向量的劈开概率只取决于它们的夹角。高斯向量的方向均匀性正是这个性质，且实现只需生成 $n$ 个独立正态数。</span>

**第三步，期望。** 设 $v_i, v_j$ 夹角为 $\theta_{ij} = \arccos\langle v_i,v_j\rangle$，随机超平面劈开它们的概率为 $\theta_{ij}/\pi$，于是

$$
\mathbb{E}[\text{跨割边数}] \ =\ \sum_{(i,j)\in E} \frac{\theta_{ij}}{\pi}
$$

## 2 公式解析：0.878 从哪来

算法的全部希望都在「把 $\frac{\theta}{\pi}$ 与 SDP 得分 $\frac{1-\cos\theta}{2}$ 挂钩」。定义一个随夹角变化的比值：

$$
g(\theta) \ =\ \frac{\theta / \pi}{(1-\cos\theta)/2} \ =\ \frac{2\theta}{\pi(1-\cos\theta)}, \qquad \theta \in [0,\pi]
$$

**Goemans–Williamson 引理：** $g(\theta) \ge \alpha^*$ 对所有 $\theta \in [0,\pi]$ 成立，其中

$$
\alpha^* \ =\ \min_{\theta \in [0,\pi]} g(\theta) \ \approx\ 0.878567
$$

**证明骨架（三步）：**
- **第一步（求驻点）**：$g$ 在 $(0,\pi)$ 内取最小。令 $g'(\theta) = 0$，即

$$
\frac{d}{d\theta}\frac{2\theta}{\pi(1-\cos\theta)} = 0 \ \Longrightarrow\ (1-\cos\theta) + \theta\sin\theta \cdot (\cdots) = 0
$$

化简得驻点满足 $\theta^* \sin\theta^* = 1 - \cos\theta^*$（即 $\theta^* = \tan(\theta^*/2)$ 的变形），数值解 $\theta^* \approx 2.331122$。
- **第二步（二阶条件）**：检查 $g''(\theta^*) > 0$，确认是极小值；端点 $g(0) \to 1$（洛必达）、$g(\pi) = 1$，内部最小值更小。
- **第三步（代入）**：$g(\theta^*) = \frac{2\theta^*}{\pi(1-\cos\theta^*)} \approx \frac{2 \times 2.331122}{\pi \times (1-\cos 2.331122)} \approx 0.878567$。<span class="marginnote">这个常数不是某个「好看」的数，而是一个超越方程的解——$g$ 在 $\theta^* \approx 133.6°$ 处取到约 0.8786。有意思的是，$g(0)=1$（夹角 0 时比值无损）、$g(\pi)=1$（完全反向也无损），最坏情形发生在中间某处——一对「半生不熟」的向量最难舍入。这就是近似算法的美：常数不是设计的，是<strong>结构逼出来的</strong>。</span>

- **第四步（应用）**：对每条边 $(i,j)$，$\frac{\theta_{ij}}{\pi} \ge \alpha^* \cdot \frac{1-\cos\theta_{ij}}{2}$。累加并对比 SDP 最优值 $\mathrm{SDP}^*$：

$$
\mathbb{E}[\text{跨割边数}] \ \ge\ \alpha^* \sum_{(i,j)\in E} \frac{1-\cos\theta_{ij}}{2} \ \ge\ \alpha^*\, \mathrm{SDP}^* \ \ge\ \alpha^*\, \mathrm{OPT}
$$

即 **$\alpha^*$-近似**（最大化问题，比值 ≥ 0.878）。

**重点：** 0.878 的推导只有「一个三角不等式 + 一个驻点计算」。它的一切精巧都在松弛与舍入的**匹配**上：向量规划的得分函数 $\frac{1-\cos\theta}{2}$ 与随机超平面的劈开概率 $\frac{\theta}{\pi}$ 的**比值**恰好被 $\alpha^*$ 夹住。**设计 SDP 松弛时就在为舍入「量身定做」得分函数**——这是 SDP 方法最重要的设计原则。

## 3 改进与变体：同一算法的家族

Goemans–Williamson 之后，同一个「向量规划 + 随机超平面」框架衍生出一批变体：

- **MAX-2SAT**：GW 的 0.878 直接适用（每个 2-子句类似一条「边」），还有专门针对 2-SAT 的改进（Feige–Goemans 0.931）。
- **最大有向割（MAX-DICUT）**：把顶点放两边、定向边从一边到另一边，GW 框架给出 0.874 左右。
- **约束满足问题（CSP）**：Raghavendra 的突破性工作证明，**在 UGC 下，某个「标准 SDP 松弛 + 随机舍入」对一切 CSP 都是最优的**——GW 是它在 MaxCut 上的特例。<span class="marginnote">Raghavendra 的定理是「SDP 时代」的收官之笔：它说明 GW 不是灵光一现，而是「用 SDP 松弛 CSP」这一整套思路的必然产物。在 Unique Games 猜想成立的假设下，最大割、MAX-2SAT、以及每一个 CSP，其可近似比都被「配对的 SDP 松弛 + 随机舍入」精确给出。GW 的 0.878 因此从「一个巧妙的算法」升格为「一个普遍原理的例子」。</span>

**重点：** 变体虽多，骨架不变：**设计得分函数（松弛）↔ 设计舍入概率（几何）↔ 夹出常数（三角）**。学一个 GW，等于学了整整一族 SDP 算法。

## 4 紧性：0.878 还能更好吗

答案取决于你信不信 **Unique Games 猜想（UGC）**：

**无 UGC 的已知下界**（Håstad 2001）：最大割不可近似到 0.941 以内（即任何 $> 0.941$ 的近似都蕴含 P = NP）。0.878 与 0.941 之间留着一道缝隙。
**UGC 下**（Khot–Kindler–Mossel–O'Donnell 2007）：0.878 是最优——任何超越 GW 的近似都蕴含 UGC 为假。

于是 GW 的 0.878 处于一个微妙位置：
**它可能不是绝对最优（0.941 那端），但在 UGC 下是终极最优**。
这个「一个算法常数同时是上界与条件最优」的状态，让最大割成为近似复杂性理论的橱窗展品。<span class="marginnote">把最大割的处境与 MAX-3SAT 对比：MAX-3SAT 的 7/8 是「PCP 焊死的绝对紧」；最大割的 0.878 是「UGC 条件下的紧」。两者分别代表近似下界的两种来源——PCP 给出无条件下界，UGC 给出条件性下界。UGC 如果成立，无数 SDP 算法的常数都会「转正」为最优；如果不成立，则要重新寻找下界来源。这是近二十年计算复杂性最核心的开放张力。</span>

**辨析｜易错点：** 0.878 是**期望**意义上的近似比，且算法有正概率输出比 0.878 更差的割。要通过去随机化（条件期望法）获得确定性保证，或重复运行取最好来放大概率。**别把「期望 ≥ 0.878·OPT」误读成「每次输出都 ≥ 0.878·OPT」**——这是所有随机化近似算法的共同注意点。

**延伸：** 0.878 证明里的三角不等式 $g(\theta) \ge \alpha^*$ 是整个 GW 算法的灵魂。
后续改进（如 Feige–Goemans 对 MAX-2SAT 的 0.931）正是把 $g$ 换成更聪明的得分函数，
让比值更贴近下界。
这条「改得分函数、重算三角不等式」的路线，是 SDP 方法后续二十年的主旋律。

## 5 小结

- **GW 算法**：向量规划松弛（$\frac{1-\cos\theta}{2}$ 得分）+ 随机超平面舍入（$\frac{\theta}{\pi}$ 概率）。
- **0.878 常数**：$g(\theta) = \frac{2\theta}{\pi(1-\cos\theta)}$ 的最小值 ≈ 0.878567，驻点 $\theta^* \approx 2.331$