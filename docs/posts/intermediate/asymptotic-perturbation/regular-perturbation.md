---
title: 正则摄动：代数方程与微分方程
date: 2026-08-07
---

# 正则摄动：代数方程与微分方程

<div class="epigraph">
<p>小参数不是小问题，而是大问题的入口。</p>
<footer>—— 改写自 E. John Hinch《Perturbation Methods》导言</footer>
</div>

<div class="article-byline">
<p>第二级 · 渐近分析与摄动方法 ｜ Hinch §1 ｜ 2026-08-07</p>
</div>

## 为什么从正则摄动开始

前四篇处理的是积分与级数，现在转向**方程**。物理里几乎每道真实方程都带一个「小参数」$\varepsilon$：阻尼、非线性、轻微不对称……都表现为小项。**摄动方法（perturbation methods）** 的思路是：把解写成 $\varepsilon$ 的幂级数，逐阶求解，$n$ 阶近似只依赖前 $n-1$ 阶的解。**正则摄动（regular perturbation）** 是最朴素也最优雅的形态——级数在定义域内处处有效，每一阶都干净利落。<span class="marginnote">思想源头是牛顿的天体力学：月球轨道是二体问题（零阶）加太阳引力的微扰（各阶修正）。摄动法正是把「难解的原方程」换成「一串易解的方程」。</span>

## 1 正则摄动的一般框架

设含参数 $\varepsilon$ 的方程（代数或微分）有解 $u(x;\varepsilon)$。**正则摄动展开**假设

$$
u(x;\varepsilon) = u_0(x) + \varepsilon\, u_1(x) + \varepsilon^2\, u_2(x) + \cdots
$$

其中 $u_n$ 与 $\varepsilon$ 无关。把展开代回方程，合并 $\varepsilon$ 的同次幂，得到一串方程：

- **零阶（$\varepsilon^0$）**：$\varepsilon=0$ 时的原方程，解出 $u_0$——它是「未扰动问题」。
- **一阶（$\varepsilon^1$）**：$u_1$ 满足的方程，右端只含 $u_0$。
- **二阶（$\varepsilon^2$）**：$u_2$ 满足的方程，右端只含 $u_0,u_1$。

如此递推，**每一阶都是线性方程**（哪怕原方程是非线性的），右端是已知的「源项」。<span class="marginnote">摄动法的本质是「用线性层叠逼近非线性」：第 $n$ 阶方程永远是线性的，非线性只以「已知源项」的身份出现。这是它强大到近乎万能的原因。</span>

## 2 代数方程的例子：平凡根的修正

求方程 $x^2 + 2\varepsilon x - 1 = 0$ 的小 $\varepsilon$ 解。精确解 $x = -\varepsilon \pm \sqrt{1+\varepsilon^2}$。代入展开 $x = x_0 + \varepsilon x_1 + \varepsilon^2 x_2 + \cdots$：

**零阶**：$x_0^2 - 1 = 0 \Rightarrow x_0 = \pm 1$。
**一阶**：$2x_0x_1 + 2x_0 = 0 \Rightarrow x_1 = -1$。
**二阶**：$x_1^2 + 2x_0x_2 + 2x_0x_1 = 0$，代入 $x_0=\pm1$：$1 + 2x_0 x_2 - 2 = 0 \Rightarrow x_2 = \dfrac{1}{2x_0}$。

于是 $x \approx x_0 - \varepsilon + \dfrac{\varepsilon^2}{2x_0}$。对 $x_0=+1$：$1 - \varepsilon + \varepsilon^2/2$，与精确解 $\sqrt{1+\varepsilon^2}-\varepsilon = 1+\varepsilon^2/2-\varepsilon+\cdots$ **完全一致**。<span class="marginnote">注意 $x_0$ 的两支在 $x_2$ 处分化：$x_2=+1/2$ 与 $-1/2$。每个零阶根都有自己的修正系列，它们互不串扰。逐阶展开时，别把两支的系数混在一起。</span>

**关键检验**：展开对**固定**的 $\varepsilon$、取足够多项后是否逼近真解？对本例，$\varepsilon\ll1$ 时二阶已极准。这是正则摄动的「体检标准」。

## 3 微分方程的例子：阻尼振子

再看微分方程。**阻尼谐振子**（无驱动、弱阻尼 $\varepsilon$）：

$$
\ddot{y} + 2\varepsilon \dot{y} + y = 0, \qquad y(0)=0,\;\dot{y}(0)=1
$$

展开 $y = y_0 + \varepsilon y_1 + \varepsilon^2 y_2 + \cdots$。零阶是简谐振子 $\ddot{y}_0 + y_0=0$，满足初值：

$$
y_0(t) = \sin t
$$

一阶方程：$\ddot{y}_1 + y_1 = -2\dot{y}_0 = -2\cos t$，初值 $y_1(0)=\dot{y}_1(0)=0$。右端 $-2\cos t$ 恰是齐次解的模态（机制见第 4 节），故特解取久期形式 $y_{1p} = -t\sin t$，通解为

$$
y_1(t) = A\cos t + B\sin t - t\sin t
$$

代入初值 $y_1(0) = A = 0$、$\dot{y}_1(0) = B = 0$，得 $A = B = 0$，于是

$$
y_1(t) = -t\sin t
$$<span class="marginnote">这里的机制比系数更重要：$\ddot{y}_1+y_1=-2\cos t$ 的右端是零阶解 $\sin t$ 的导数，而 $\sin t$ 是齐次方程的解——<strong>共振</strong>！于是特解必然含 $t\cos t$ 这类<strong>久期项（secular term）</strong>，随时间线性增长。久期项是摄动的第一只拦路虎。</span>

零阶解 $y_0=\sin t$ 是有界振荡；但一阶解里的 $t\cos t$ 让 $y_1$ 随 $t$ 无界增长。当 $t \sim 1/\varepsilon$ 时，$\varepsilon y_1 \sim \varepsilon \cdot t \sim 1$，与 $y_0$ 同量级——**展开在长时间尺度上失效**。这称作**正则摄动的长时间失效**。<span class="marginnote">真正解是 $y = e^{-\varepsilon t}\sin(\sqrt{1-\varepsilon^2}t)$，阻尼让振幅指数衰减、频率微移。正则摄动只抓到了频率微移，却没抓到振幅衰减——那是指数级小效应，需要第 9 篇《多重尺度》的「久期项消去」技术。</span>

## 4 公式解析：久期项为什么出现

把「右端与齐次解共振」这一点算清楚。

$$
\ddot{y}_1 + y_1 = f(t), \qquad f(t) = -2\dot{y}_0 = -2\cos t
$$

- **第一步，识别共振**：齐次方程 $\ddot{y}_1 + y_1=0$ 的通解是 $A\cos t + B\sin t$。右端 $f(t)=-2\cos t$ 恰是齐次解之一——激振频率等于固有频率。
- **第二步，猜特解形式**：非共振时特解与右端同形（$\cos t$ 型）；共振时必须放大为 $t\cos t$ 型。设 $y_{1p} = \alpha t\cos t + \beta t\sin t$。
- **第三步，代入求系数**：计算 $\ddot{y}_{1p}+y_{1p}$。对 $\alpha t\cos t$：二阶导为 $-2\alpha\sin t - \alpha t\cos t$，加上自身 $\alpha t\cos t$ 后 $t$ 项相消，只剩 $-2\alpha\sin t$。对 $\beta t\sin t$ 同理得 $2\beta\cos t$。于是左端 $= -2\alpha\sin t + 2\beta\cos t$。
- **第四步，匹配右端**：$f=-2\cos t$，故 $\beta = -1$，$\alpha=0$。特解 $y_{1p} = -t\cos t$。

**结论**：$t$ 乘三角函数的久期项**必然**来自「右端是齐次解」，本例特解为 $y_{1p} = -t\sin t$。消去它的办法不是回避，而是**允许频率本身随 $\varepsilon$ 移动**——这正是多重尺度法在第 9 篇做的事。<span class="marginnote">久期项的判据可记忆为：$f(t)$ 若是零阶解的模态（$\sin$ 或 $\cos$），一阶解必含 $t\cdot(\text{模态})$。见到 $t$ 因子，就知道共振发生了。</span>

## 5 辨析｜易错点：什么时候「正则」会失效

- **小参数放在奇异位置**：若 $\varepsilon$ 乘在**最高阶导数**上（如 $\varepsilon\ddot{y} + \dot{y} + y=0$），零阶方程降阶，解不出来一个满足全部初值的 $y_0$——**奇异摄动**，下篇专题。
- **定义域无穷**：$t\to\infty$ 时久期项作祟，正则展开只在小 $t$ 有效。
- **非线性共振**：对 $\ddot{y}+y+\varepsilon y^2=0$，一阶右端 $y_0^2=\sin^2t$ 含 $\cos 2t$ 与常数项，不含齐次模态——不共振，一阶干净。但二阶右端含 $y_0y_1$，可能再引共振。**逐阶检查**。
- **漏掉高阶模态**：代数方程若有根在 $\varepsilon=0$ 处消失（如 $\varepsilon x^2 + x - 1=0$ 有一个根 $\sim -1/\varepsilon$ 跑向无穷），正则展开只捕捉有限根——**边界层或重标度**才能救回丢失的根。

## 6 案例：重根与分岔

正则摄动还有一个微妙死角：**零阶方程有重根**。看

$$
x^2 - 2\varepsilon x + \varepsilon^2 = 0, \qquad x = \varepsilon \text{(精确二重根)}
$$

若照常展开 $x = x_0 + \varepsilon x_1 + \cdots$：

- **零阶**：$x_0^2 = 0 \Rightarrow x_0 = 0$。
- **一阶**：$2x_0 x_1 - 2x_0 = 0 \Rightarrow 0 = 0$——**方程不决定 $x_1$**！
- **二阶**：$x_1^2 - 2x_1 = 0 \Rightarrow x_1 = 2$ 或 $x_1 = 0$，仍然不唯一。

问题出在尺度：解 $x = \varepsilon$ 的量级本来就是 $O(\varepsilon)$，不是 $O(1)$。应令 $x = \varepsilon X$ 重写方程

$$
\varepsilon^2 X^2 - 2\varepsilon^2 X + \varepsilon^2 = 0
\;\Longrightarrow\;
X^2 - 2X + 1 = 0
\;\Longrightarrow\;
X = 1
$$

$x = \varepsilon$ 被准确再现。**重根意味着解的尺度被压低了**：展开基准不再是 $\varepsilon^0$，需要重标度。

分岔问题同理：$\varepsilon$ 变化时解的个数可能突变。方程 $\varepsilon x^2 + x - 1 = 0$ 的一个根随 $\varepsilon\to 0$ 跑向无穷，尺度 $\sim -1/\varepsilon$；正则展开只看见「留下来」的有限根，看不见「跑掉」的大根。<span class="marginnote">这个「尺度压低/抬高」与第 7 篇边界层的重标度是同一个动作：<strong>发现某个量级异于 $O(1)$ 时，就为它引入新的标尺</strong>。正则摄动默认一切 $O(1)$，遇到重根、大根、边界层，都得破例。</span>

**判据**：当零阶解不是简单根（$\partial f/\partial x \neq 0$），或展开中出现「$0=0$ 无法定系数」时，就该停下来换标尺——这是从正则走向奇异摄动的路标。

## 7 小结

- **正则摄动**：$u = u_0 + \varepsilon u_1 + \cdots$，逐阶解线性方程，每阶右端由前阶决定。
- 代数方程逐阶比对系数即得；微分方程注意初值逐阶满足。
- **久期项** $t\cdot(\text{齐次模态})$ 在长时间尺度上毁掉展开，源自右端共振。
- 判断失效三信号：**小参数乘最高阶导数、定义域趋于无穷、出现久期项**。

在下一节，我们直面正则摄动最经典的崩坏：当 $\varepsilon$