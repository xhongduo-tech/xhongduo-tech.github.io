---
title: 变换、过渡与动画
date: 2026-08-07
---

# 变换、过渡与动画

<div class="epigraph">
<p>动效不是装饰，而是界面在告诉用户：我听见了，我正因此而改变。</p>
<footer>—— 唐 · 诺曼（Don Norman），《设计心理学》</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs CSS ｜ 2026-08-07</p>
</div>

## 为什么从变换、过渡与动画开始

到目前为止，页面都是**静态**的：内容摆好，样式固定。但界面需要「反应」——按钮按下要反馈、弹窗出现要过渡、加载要有等待感。CSS 的三件套把「变化」做成声明式能力：

- **变换（transform）**：让元素在 2D/3D 空间里移动、缩放、旋转——改变的是**几何**。
- **过渡（transition）**：当属性值变化时，让变化**平滑过渡**而不是瞬跳——改变的是**过程**。
- **动画（animation）**：用 `@keyframes` 定义**多阶段、可循环**的运动序列——改变的是**剧本**。

三者的关系是递进的：变换定义「怎么动」，过渡定义「动得多顺」，动画定义「动的完整剧本」。动效对**可访问性**也有要求——这正是上一节 `prefers-reduced-motion` 的实际应用场景。<span class="marginnote">动效的心理学基础：人眼对突然的位置/大小变化极其敏感（这是视觉注意的原始机制），但对平滑过渡能自然追踪。所以「瞬跳」会让用户困惑「发生了什么」，平滑过渡则建立「因果关系」的直觉。好的动效 = 让状态变化可理解。</span>

## 1 变换：transform 的几何语法

**`transform`** 改变元素的几何形态，多个变换函数写在一行、从右到左复合：

```css
.card {
  transform: translate(20px, 10px) scale(1.05) rotate(2deg);
}
```

常用函数：

| 函数 | 作用 | 示例 |
| --- | --- | --- |
| `translate(tx, ty)` | 平移 | `translateX(50%)` 相对自身宽度移动 |
| `scale(sx, sy)` | 缩放 | `scale(1.2)` 放大 20% |
| `rotate(deg)` | 旋转 | `rotate(45deg)` 顺时针 45° |
| `skew(xdeg)` | 斜切 | `skew(10deg)` 倾斜变形 |
| `translate(-50%, -50%)` | 居中技巧 | 配合绝对定位实现精确居中 |

**`transform-origin`** 指定变换的原点（默认中心）。`rotate` 时原点决定「绕哪里转」——绕左上角转与绕中心转，结果完全不同。

**辨析｜易错点：** 变换的复合顺序**不可交换**。`translate(10px) rotate(45deg)` 先平移再绕（平移后的）原点旋转；`rotate(45deg) translate(10px)` 先旋转再沿（旋转后的）方向平移——两者结果不同。想保持直觉，就固定「先 translate 后 rotate」的习惯写法。<span class="marginnote">transform 的三大性能优势：它不触发布局重排（reflow）、只合成（composite）、能在 GPU 上跑——因此动画首选 transform/opacity 而不是改 `top`/`left`/`width`。这直接呼应第28篇《Web 性能优化》的「合成层」概念。</span>

## 2 过渡：transition 让变化变平滑

**`transition`** 把一个属性的变化变成「动画」，由四个子属性控制：

```css
.btn {
  background-color: #c0392b;
  transition: background-color 0.3s ease 0s;
}
.btn:hover {
  background-color: #8e241a;
}
```

顺序是 `transition: property duration timing-function delay`：

**`property`**：过渡哪个属性（`all` 全部，但性能和语义都不如显式列出）。
**`duration`**：时长，`0.3s`。
- **`timing-function`**：速度曲线——`ease`（缓入缓出，默认）、`linear`（匀速）、`ease-in`（加速）、`ease-out`（减速）、`cubic-bezier(...)`（自定义贝塞尔曲线）。
- **`delay`**：延迟开始，`0s` 无延迟。

**过渡的触发条件**：只有属性值**真的发生变化**（hover、类切换、JS 修改）时才触发。初始加载时不会自动过渡——若想要「进页面就播放一次效果」，那该用动画。

**timing-function 的直觉**：`ease` 是最自然的「开始快、收尾慢」；`ease-out` 适合「元素飞入」（入场减速更优雅）；`ease-in` 适合「元素飞离」（离场加速更利落）。`cubic-bezier(0.68, -0.55, 0.265, 1.55)` 是著名的「回弹」曲线——超过目标再弹回，物理感的来源。<span class="marginnote">`cubic-bezier` 是三次贝塞尔曲线 `cubic-bezier(x1, y1, x2, y2)`，控制点在 [0,1] 区间内为常规缓动，y 值越界（如负值或 >1）就产生「回弹/过冲」效果。DevTools 里可视化拖拽就能调出想要的曲线。</span>

## 3 动画：@keyframes 的完整剧本

**动画（animation）** 由两部分组成：`@keyframes` 定义**关键帧**，`animation` 属性应用它。

```css
@keyframes fade-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

.banner {
  animation: fade-in 0.8s ease-out 0.2s both;
}
```

`animation` 简写依次是 `animation: name duration timing-function delay fill-mode`（外加可选的 `iteration-count`、`direction`）。关键属性：

**`iteration-count`**：播放次数，`infinite` 无限循环。
**`direction`**：`normal` / `reverse` / `alternate`（正反交替）/ `alternate-reverse`。
- **`fill-mode`**：播放前后的状态——`both` 最常用：开始前停在 `from`、结束后停在 `to`，避免「动画前闪一下初始态」。
- **`keyframes` 可用百分比**多阶段：`0% {…} 50% {…} 100% {…}`，把一段动画拆成任意段。

**辨析｜易错点：** `transition` 需要「外部触发」（hover、类变），`animation` **自驱动**——加载即播、可循环、可多阶段。用动画做「入场效果」「循环加载」；用过渡做「交互反馈」。两者都能动，但触发机制完全不同，选错就得到「不播」或「过度实现」的尴尬。<span class="marginnote">加载动画的经典做法：一个旋转的圆环 `animation: spin 1s linear infinite` + `@keyframes spin { to { transform: rotate(360deg); } }`。只动 transform 保证 60fps——动画性能的黄金法则就是「只动画 transform 与 opacity」。</span>

## 4 公式解析：贝塞尔曲线如何决定速度

timing-function 的数学内核是**三次贝塞尔曲线**，它把「时间进度」映射成「进度值」：

$$
B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t)t^2 P_2 + t^3 P_3
$$