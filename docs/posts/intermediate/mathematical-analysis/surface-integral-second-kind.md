---
title: 第二型曲面积分：曲面的侧、定义与计算
date: 2026-08-07
---

# 第二型曲面积分：曲面的侧、定义与计算

<div class="epigraph">
<p>穿过曲面的流体总量——它依赖曲面「朝向哪一边」。第二型曲面积分携带定向，是「向量场穿出曲面的通量」的精确表述。</p>
<footer>—— 高斯（Carl Friedrich Gauss），通量理论（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§22.2 ｜ 2026-08-07</p>
</div>

## 为什么「曲面的侧」是新概念

第一型曲面积分沿曲面累积标量，方向无关。但「流体穿过曲面的总量」不同：**同样一块曲面，法向朝上穿出的流量与朝下不同**——积分必须知道「曲面朝向哪边」。这就是第二型曲面积分（对坐标的积分），它需要**定向曲面**（选好法向）。

物理直觉：$\iint_S\vec F\cdot d\vec S$ 是「向量场 $\vec F$ 穿出曲面的通量」——**每小块曲面的「穿出量」是「场在法向的投影 × 面积」**。这个「通量」是高斯公式（§22.3）的核心对象。<span class="marginnote">「通量」的直觉：想象一张渔网（曲面）放在水流（向量场）里，单位时间穿过的水量就是通量。穿出方向取曲面的法向——选上法向还是下法向，通量差个正负号。<strong>「$\iint_S\vec F\cdot d\vec S$」中 $d\vec S=\vec n\,dS$（法向 × 面积微元）是「带方向的面积」</strong>——这就是「第二型」与「第一型」的本质区别：一个积分的是「标量 × 面积」，另一个是「向量点积向量 × 面积」。</span>

## 1 曲面的侧与定向

**曲面的侧（side）**：光滑曲面 $S$ 有**两侧**——由单位法向量 $\vec n$ 区分：$\vec n$ 指向的一侧与 $- \vec n$ 指向的一侧。

**可定向曲面**：能选出一个「连续变化的法向」的曲面。**双侧曲面**（平面、球面、柱面）可定向；**单侧曲面**（莫比乌斯带）不可定向——它的法向沿闭路走一圈后翻转，无法一致定向。

**定向曲面**：$S$ 带上指定的法向 $\vec n$，记 $S^+$（或 $S^-$）表示指定了某一侧。

> **辨析｜易错点：**第二型曲面积分的**方向性**：$S^-$（反向定向）时 $\iint_{S^-}\vec F\cdot d\vec S=-\iint_{S^+}\vec F\cdot d\vec S$——因为 $d\vec S$ 反号。**这是与第一型的根本区别**（第一型无方向）。另一个易错点：**莫比乌斯带不可定向**——它只有一侧，无法定义「穿出」方向，第二型积分无意义。**「先确认曲面可定向、再选定向」**是第二型积分的第一步。

## 2 第二型曲面积分的定义

**定义**：设 $S$ 是定向光滑曲面，$\vec n=(\cos\alpha,\cos\beta,\cos\gamma)$ 是 $S$ 的单位法向量，$\vec F=(P,Q,R)$ 在 $S$ 上连续。**第二型曲面积分**：

$$\iint_S\vec F\cdot d\vec S=\iint_S\vec F\cdot\vec n\,dS=\iint_S(P\cos\alpha+Q\cos\beta+R\cos\gamma)dS.$$

**坐标形式**：$d\vec S=\vec n\,dS=(dy\,dz,dz\,dx,dx\,dy)$（法向分量 × 面积），故

$$\iint_S\vec F\cdot d\vec S=\iint_S P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy.$$

**三个坐标项**：$P\,dy\,dz$ 是「$P$ 沿 $yz$ 方向投影面积」的通量贡献，$Q\,dz\,dx$、$R\,dx\,dy$ 同理。

**与第一型的关系**：**第二型 = 第一型的「法向投影」**——$\iint_S\vec F\cdot d\vec S=\iint_S(\vec F\cdot\vec n)dS$，把「场在法向的分量」对面积积分。这与 §20.2 第二型曲线积分 = 第一型的「切向投影」完全平行。

> **辨析｜易错点：**第二型曲面积分的**记号** $P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy$ 中，每项的「面积微元」是「两坐标微元的乘积」——$dy\,dz$ 是「在 $yz$ 平面的投影面积」。（$P,Q,R$ 分别是场在 $x,y,z$ 方向的分量，配「不含该坐标的面积微元」。）**「$P$ 配 $dy\,dz$、$Q$ 配 $dz\,dx$、$R$ 配 $dx\,dy$」**是写坐标形式的记忆规则。

## 3 计算公式

**定理：设 $S$ 由 $z=z(x,y)$（$(x,y)\in D$）给出，定向为「上侧」（法向与 $z$ 轴正向夹角为锐角），则**

$$\iint_S R(x,y,z)\,dx\,dy=\iint_D R(x,y,z(x,y))\,dx\,dy.$$

**一般地（上侧）**：

$$\iint_S P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy=\iint_D\left(-Pz_x-Qz_y+R\right)dx\,dy.$$

**公式解析：为什么出现 $-Pz_x-Qz_y+R$**

**第一步，法向投影**。上侧法向 $\vec n=\frac{(-z_x,-z_y,1)}{\sqrt{1+z_x^2+z_y^2}}$（§17.1 切平面法向）。$\vec F\cdot\vec n=\frac{-Pz_x-Qz_y+R}{\sqrt{1+z_x^2+z_y^2}}$；

**第二步，与 $dS$ 相乘**。$dS=\sqrt{1+z_x^2+z_y^2}dx\,dy$，两者相乘——**根号因子消掉**：

$$\iint_S\vec F\cdot d\vec S=\iint_D(\vec F\cdot\vec n)dS=\iint_D(-Pz_x-Qz_y+R)dx\,dy;$$

**第三步，符号规则**。**上侧（法向朝上）取 $+$**，下侧（法向朝下）取 $-$——定向决定整体符号。

**示范**：$\vec F=(0,0,z)$ 沿抛物面 $z=x^2+y^2$ 在 $x^2+y^2\le1$ 的上侧。$P=Q=0,\ R=z$，$z_x=2x,\ z_y=2y$：

$$\iint_S z\,dx\,dy=\iint_{x^2+y^2\le1}(x^2+y^2)dx\,dy=\int_0^{2\pi}\int_0^1r^2\cdot r\,dr\,d\theta=\frac{\pi}{2}.$$

**「$\vec F=(0,0,z)$ 沿上侧的 $z$ 分量积分」= 二重积分直接算**——$P,Q$ 项为零时公式最简。

**参数曲面**：$S:\ \vec r(u,v)$，定向由 $\vec r_u\times\vec r_v$ 决定，$\iint_S\vec F\cdot d\vec S=\iint_D\vec F(\vec r(u,v))\cdot(\vec r_u\times\vec r_v)\,du\,dv$——**「$d\vec S=\vec r_u\times\vec r_v\,du\,dv$」是参数曲面的定向面积微元**。

## 4 通量的物理意义

**通量（flux）**：$\iint_S\vec F\cdot d\vec S$ 是「向量场 $\vec F$ 穿出曲面 $S$ 的通量」——**单位时间穿过曲面的「流」的总量**。

- **流体力学**：速度场 $\vec v$ 穿过曲面的通量 = 体积流量；
- **电磁学**：电位移 $\vec D$ 的通量 = 电通量，磁感应 $\vec B$ 的通量 = 磁通量（高斯定律、磁高斯定律，§22.3）；
- **热传导**：热流 $\vec q=-k\nabla T$ 的通量 = 热量流失。

**示范**：$\vec F=(x,y,z)$ 沿单位球面**外侧**的通量。$S$ 外侧法向 $\vec n=(x,y,z)$（球面法向 = 径向），$\vec F\cdot\vec n=x^2+y^2+z^2=1$：

$$\iint_S\vec F\cdot d\vec S=\iint_S1\,dS=4\pi.$$

**「$\vec F=\vec r$（径向场）穿出单位球面的通量 $=4\pi$ = 球面面积」**——这个结果在高斯公式（§22.3）里将扮演验证角色。<span class="marginnote">「径向场 $\vec F=\vec r$ 穿出球面的通量 $=4\pi$」是高斯定理的经典开场白：直接算（§22.2）与用高斯公式（§22.3，$\text{div}\,\vec F=3$，$\iiint3\,dV=3\cdot\frac43\pi=4\pi$）结果一致。高斯定律「$\oint\vec E\cdot d\vec S=\frac{Q}{\varepsilon_0}$」正是「通量 = 内部源总量」的物理形态——第二级《电动力学》里，高斯定律是最基础的积分定律之一。</span>

## 5 第一型与第二型曲面积分对比

| | 第一型 $\iint f\,dS$ | 第二型 $\iint\vec F\cdot d\vec S$ |
| --- | --- | --- |
| 对象 | 标量场 | 向量场 |
| 微元 | 面积 $dS$ | 定向面积 $d\vec S=\vec n\,dS$ |
| 方向 | 无关 | 有关（反定向反号） |
| 物理 | 曲面质量 | 通量 |
| 联系 | — | $=\iint(\vec F\cdot\vec n)dS$ |

**第一型管「总量」，第二型管「通量」**——通过「法向投影」相连，正如曲线积分通过「切向投影」相连。

## 6 计算示范：第二型曲面积分的完整演练

**示范一（平面上的通量）**：$\vec F=(x,y,z)$ 沿平面 $x+y+z=1$ 在第一卦限部分的上侧。平面 $z=1-x-y$，$z_x=z_y=-1$，由公式 $\iint(-Pz_x-Qz_y+R)dx\,dy=\iint(x+y+z)dx\,dy$（$P=x,Q=y,R=z$），在投影域 $x\ge0,y\ge0,x+y\le1$ 上 $z=1-x-y$，故被积函数 $=x+y+1-x-y=1$，积分 $=$ 三角形面积 $=\frac12$。**「$\vec F\cdot\vec n$ 恰为常数」时，通量 $=$ 常数 × 投影面积**。

**示范二（球面的通量）**：$\vec F=(0,0,z)$ 沿单位球面**外侧**。外侧法向 $\vec n=(x,y,z)$（径向），$\vec F\cdot\vec n=z^2$，故通量 $=\iint_S z^2\,dS$。用球坐标参数化 $\vec r(\theta,\varphi)=(\sin\theta\cos\varphi,\sin\theta\sin\varphi,\cos\theta)$，$dS=\sin\theta\,d\theta\,d\varphi$：

$$\iint_Sz^2dS=\int_0^{2\pi}\int_0^\pi\cos^2\theta\sin\theta\,d\theta\,d\varphi=2\pi\cdot\frac23=\frac{4\pi}{3}.$$

**「$\vec F=(0,0,z)$ 穿球面通量 $=\frac{4\pi}{3}$」**——用参数曲面的 $d\vec S=\vec r_\theta\times\vec r_\varphi\,d\theta\,d\varphi$ 计算，是球面第二型积分的标准套路。

**示范三（定向符号的核对）**：示范一若取下侧，积分 $=-\frac12$。**「反向定向整体反号」**用一正一负两个示范核对，确认 $d\vec S=\vec n\,dS$ 中的法向选对。<span class="marginnote">「上侧取 $+$、下侧取 $-$」的符号规则在 $z=z(x,y)$ 情形最直观；换成「前侧/后侧」「左侧/右侧」（法向沿 $x$ 或 $y$ 轴）时，公式里的 $(-Pz_x-Qz_y+R)$ 会换成不同的组合（$P$ 项主导）。工程里选「外法向」为正向是约定俗成——闭曲面的外侧总是标准定向。<strong>「先定法向、再定符号、最后积分」</strong>是三步铁律。</span>

**示范四（通量的物理核对）**：速度场 $\vec v=(0,0,1)$ 穿过单位圆盘 $x^2+y^2\le1$（上侧，$z=0$ 平面）。$\vec F\cdot\vec n=1$，通量 $=$ 圆盘面积 $=\pi$。**「单位速度场穿单位面积 = 面积」**——通量就是「流量」，单位场下退化为面积，这是定义自洽性的直观检验。

## 7 小结

- **曲面的侧**：法向分两侧；双侧可定向、莫比乌斯带单侧不可定向。
- **第二型曲面积分**：$\iint_S\vec F\cdot d\vec S=\iint_S\vec F\cdot\vec n\,dS=\iint P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy$。
- **计算**：$z=z(x,y)$ 上侧时 $\iint(-Pz_x-Qz_y+R)dx\,dy$；根号因子与法向分量对消。
- **与第一型联系**：第二型 = 第一型的法向投影——「通量 = 法向分量 × 面积」。
- **通量物理**：$\iint\vec F\cdot d\vec S$ 是「穿出量」；$\vec F=\vec r$ 穿球面 = $4\pi$（高斯公式的伏笔）。

在下一节，我们迎来场论第一定理：**高斯公式——三重积分与曲面积分的联系**。「散度的体积分 = 通量的面积分」，这是三维的格林公式。
