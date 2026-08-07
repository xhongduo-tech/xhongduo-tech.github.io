---
title: 拉普拉斯变换的性质：线性、微分、积分
date: 2026-08-08
---

# 拉普拉斯变换的性质：线性、微分、积分

<div class="epigraph">
<p>微分性质里藏着初始条件：$y'(t)\leftrightarrow sY(s)-y(0)$——这正是拉普拉斯变换解初值问题的全部秘密。</p>
<footer>—— 拉普拉斯变换的性质</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数与积分变换》§8.3 ｜ 2026-08-08</p>
</div>

## 为什么微分性质是「解方程的开关」

拉普拉斯变换的核心价值在微分性质：**对 $y'(t)$ 取变换得到 $sY(s)-y(0)$，初始条件 $y(0)$ 直接出现在公式里。** 于是「解带初值的微分方程」变成「解代数方程 + 逆变换」——初值自动带入，无需事后定常数。这一节建立线性、微分（一阶与 $n$ 阶）、积分三条性质。它们与傅里叶的对应性质同构，但多了「初始条件项」——这正是拉普拉斯比傅里叶更适合初值问题的原因。<span class="marginnote">对比傅里叶：$f'(t)\leftrightarrow i\omega F(\omega)$ 没有初始条件项，因为傅里叶在 $(-\infty,\infty)$ 上积分、默认「稳态」。拉普拉斯在 $[0,\infty)$ 上积分，端点 $t=0$ 的信息（初始条件）天然进入公式。<strong>「从 $0$ 开始积分」带来「初始条件项」——因果性换来的解方程能力。</strong></span>

## 1 线性性质

**性质：** $\mathcal{L}[af(t)+bg(t)] = aF(s)+bG(s)$。

**直接来自积分线性。** 线性让「信号拆成表内函数之和」→「变换拆成表内变换之和」成为可能——查表法的基础。

**例：** $\mathcal{L}[3+2e^{-t}]=\frac3s+\frac2{s+1}$。秒出。

## 2 一阶微分性质

**核心性质：**

$$\mathcal{L}[f'(t)] = sF(s) - f(0)$$

**其中 $f(0)=\lim_{t\to0^+}f(t)$（初值）。**

**证明：** 分部积分 $\int_0^{\infty}f'(t)e^{-st}dt$，取 $u=e^{-st}$，$dv=f'(t)dt$：

$$\mathcal{L}[f'] = \left[f(t)e^{-st}\right]_0^{\infty} + s\int_0^{\infty}f(t)e^{-st}dt = -f(0) + sF(s)$$

**（$\mathrm{Re}s$ 充分大时 $f(t)e^{-st}\big|_{\infty}=0$。）**

**高阶推广（反复应用）：**

$$\mathcal{L}[f''(t)] = s^2F(s) - sf(0) - f'(0)$$

$$\mathcal{L}[f^{(n)}(t)] = s^nF(s) - s^{n-1}f(0) - s^{n-2}f'(0) - \cdots - f^{(n-1)}(0)$$

**重点：$n$ 阶导数 = $s^n$ 乘变换，减去一串初始条件项。** 初始条件 $f(0),f'(0),\dots,f^{(n-1)}(0)$ 全部显式进入公式——这就是「初值问题在变换里自动携带初值」的含义。<span class="marginnote">记忆法：「$s$ 的降幂乘以初值」——$f''$ 的变换是 $s^2F$ 减去 $sf(0)$（$s$ 一次）再减 $f'(0)$（$s$ 零次）。$s$ 的指数随初值阶数递减：$f^{(k)}(0)$ 前乘 $s^{n-1-k}$。</span>

## 3 积分性质

**性质：**

$$\mathcal{L}\left[\int_0^{t}f(\tau)\,d\tau\right] = \frac{F(s)}{s}$$

**证明：** 令 $g(t)=\int_0^t f(\tau)d\tau$，则 $g'(t)=f(t)$、$g(0)=0$。对 $g'(t)=f(t)$ 取变换，用微分性质：$sG(s)-g(0)=F(s)$，故 $G(s)=\frac{F(s)}s$。

**重点：积分在频域 = 除以 $s$**（与微分乘 $s$ 互逆）。注意积分下限是 $0$（从 $t=0$ 积到 $t$），与傅里叶的 $\int_{-\infty}^t$ 不同——因为因果信号在 $t<0$ 为零。

**例：** $\mathcal{L}\left[\int_0^t\tau\,d\tau\right]$：$\mathcal{L}[t]=\frac1{s^2}$，除以 $s$ 得 $\frac1{s^3}$。验证：$\int_0^t\tau d\tau=\frac{t^2}2$，$\mathcal{L}[\frac{t^2}2]=\frac12\cdot\frac2{s^3}=\frac1{s^3}$ ✓。

**辨析｜易错点：积分下限是 $0$ 还是 $-\infty$。** 拉普拉斯性质用 $\int_0^t$（因果起点），不是 $\int_{-\infty}^t$。若题目给 $\int_{-\infty}^t$ 的积分，先把 $f$ 在 $t<0$ 的部分按因果性处理（通常为零），再套公式。<span class="marginnote">「积分除 $s$」与「微分乘 $s$」的互逆在解积分方程时特别有用：含 $\int_0^t$ 的积分方程取变换后变成代数方程，未知函数被解出。第八章「积分方程」一节的正是一套路。</span>

## 4 公式解析：$f''(t)$ 变换的「两次分部」

把二阶微分性质拆成可操作的推导，理解「初始条件为什么成串出现」：

$$\mathcal{L}[f''(t)] = s^2F(s) - sf(0) - f'(0)$$

- **第一步，对 $f''$ 用一次微分性质。** 令 $h=f'$，则 $\mathcal{L}[h']=s\mathcal{L}[h]-h(0)=s\mathcal{L}[f']-f'(0)$。
- **第二步，代入 $\mathcal{L}[f']$。** $\mathcal{L}[f']=sF(s)-f(0)$，于是
$$\mathcal{L}[f'']=s[sF(s)-f(0)]-f'(0)=s^2F(s)-sf(0)-f'(0)$$
- **第三步，看出模式。** 每「多一阶导数」就多一次微分性质：$s$ 的幂升一阶，初始条件串往下多一项。**$n$ 阶是 $n-1$ 次套用一阶公式的产物。**

**直觉：** 微分性质像「结算时检查初值」——每求一次导，就把「当时的初值」扣掉一次。**解方程时，初始条件作为「已知数」进入代数方程，这正是「初值问题直接可解」的原因。**

## 5 与傅里叶性质对照

| 性质 | 傅里叶 | 拉普拉斯 |
| --- | --- | --- |
| 线性 | $\mathcal{F}[af+bg]=aF+bG$ | $\mathcal{L}[af+bg]=aF+bG$ |
| 一阶微分 | $f'\leftrightarrow i\omega F$ | $f'\leftrightarrow sF-f(0)$ |
| $n$ 阶微分 | $f^{(n)}\leftrightarrow(i\omega)^nF$ | $f^{(n)}\leftrightarrow s^nF-\sum s^{n-1-k}f^{(k)}(0)$ |
| 积分 | $\int_{-\infty}^t f\leftrightarrow\frac F{i\omega}$ | $\int_0^t f\leftrightarrow\frac Fs$ |
| 卷积 | $f*g\leftrightarrow FG$ | $f*g\leftrightarrow FG$ |

**重点：拉普拉斯性质的骨架与傅里叶完全一致**（微分乘 $s$、积分除 $s$、卷积乘），**唯一的差别是微分性质多了初始条件项、积分下限从 $-\infty$ 变成 $0$**。学会一套，两章通用。<span class="marginnote">「骨架相同、细节不同」是学习积分变换的最佳策略：先掌握傅里叶性质（第七章），拉普拉斯性质的记忆成本就极低——只需额外记住「初始条件项」与「因果积分下限」。两章对照着学，事半功倍。</span>

## 6 补充：微分性质的应用实例

微分性质「$f'\leftrightarrow sF-f(0)$」是最常用的单条性质，用它做三道题感受威力。

**例 1（用微分性质求变换）：** 求 $\mathcal{L}[\cos\omega t]$。已知 $\mathcal{L}[\sin\omega t]=\frac{\omega}{s^2+\omega^2}$，且 $(\sin\omega t)'=\omega\cos\omega t$。取变换：

$$\mathcal{L}[\omega\cos\omega t]=s\cdot\frac{\omega}{s^2+\omega^2}-\sin0=s\frac{\omega}{s^2+\omega^2}$$

故 $\mathcal{L}[\cos\omega t]=\frac{s}{s^2+\omega^2}$。**用微分性质从正弦「变」出余弦，不必重新积分。**

**例 2（初值定理）：** 已知 $F(s)$，可直接读出 $f(0^+)$：

$$f(0^+)=\lim_{s\to\infty}sF(s)$$

**验证：** $F(s)=\frac{s}{(s+1)^2+4}$，$\lim_{s\to\infty}sF(s)=\lim\frac{s^2}{(s+1)^2+4}=1$。而 $f(t)=e^{-t}\cos2t$，$f(0)=1$ ✓。**初值定理让「读初值」不需要逆变换。**

**例 3（终值定理）：** 若 $\lim_{t\to\infty}f(t)$ 存在，则

$$f(\infty)=\lim_{s\to0}sF(s)$$

**验证：** $F(s)=\frac1{s(s+1)}$，$\lim_{s\to0}sF(s)=\lim\frac1{s+1}=1$。而 $f(t)=1-e^{-t}$，$f(\infty)=1$ ✓。**终值定理让「稳态值」唾手可得——控制理论里用它算系统稳态误差。**

**重点：初值/终值定理是微分性质的「极限版」**——$\lim_{s\to\infty}sF(s)=f(0)$ 来自 $f'\leftrightarrow sF-f(0)$ 中「$sF\to f(0)$」的观察。**工程里先取变换、再用终值定理算稳态，省去完整逆变换。**

**辨析｜易错点：终值定理要求 $sF(s)$ 在 $\mathrm{Re}s\ge0$ 解析。** 若 $F$ 在虚轴或右半平面有极点（如 $\sin$ 型），终值不存在，定理失效。**「$f(\infty)$ 存在」是使用前提，振荡信号的 $f(\infty)$ 不存在。**

## 7 小结

- **线性**：$\mathcal{L}[af+bg]=aF+bG$。
- **一阶微分**：$f'\leftrightarrow sF(s)-f(0)$；**$n$ 阶**：$f^{(n)}\leftrightarrow s^nF-\sum_{k=0}^{n-1}s^{n-1-k}f^{(k)}(0)$。
- **积分**：$\int_0^t f(\tau)d\tau\leftrightarrow\frac{F(s)}s$（因果下限 $0$）。
- **初始条件进场**：微分性质的初值项让「带初值的微分方程」直接可解。
- **与傅里叶对照**：骨架一致（乘/除 $s$、卷积乘），差异只在初值项与积分下限。

在下一节，我们补齐拉普拉斯性质的另一半：**位移、延迟、周期函数**。$e^{at}$ 平移 $s$、$u(t-a)$ 延迟乘 $e^{-as}$、周期函数公式化——这些性质让「组合信号」的变换也能查表即得。
