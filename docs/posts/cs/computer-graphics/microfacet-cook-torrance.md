---
title: 微表面模型：Cook-Torrance BRDF
date: 2026-08-08
---

# 微表面模型：Cook-Torrance BRDF

<div class="epigraph">
<p>表面不是一面镜子，而是一百万面朝向各异的镜子。</p>
<footer>—— 图形学课堂常谈</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机图形学 ｜ GAMES101 & 虎书（Fundamentals of Computer Graphics）§18.5 ｜ 2026-08-08</p>
</div>

## 为什么需要微表面模型

Blinn-Phong 用「指数 $p$」调高光的宽度，经验而粗糙。真实表面的高光形状有物理根源：**表面由无数微小的理想镜面（微面）组成，微面朝向各异**。宏观上看到的「模糊反射」，是这些微面各反射各的方向、统计平均的结果。**微表面模型（microfacet model）** 用这个物理图像推导 BRDF，是 PBR 高光的事实标准。<span class="marginnote">微表面模型的直觉：<strong>一块「哑光金属」放大看，是由百万个朝东朝西的小镜面拼成的</strong>——每个小镜面按反射定律反射，但朝向不同，反射方向散开。宏观 BRDF 的形状（峰多宽、多高）完全由「微面朝向的统计分布」决定——这就是法线分布函数。</span>

**Cook-Torrance BRDF**（Cook & Torrance 1982）是微表面模型的经典实现，也是今天所有 PBR 渲染器（游戏、影视、实时）的默认高光模型。

## 1 Cook-Torrance BRDF 的结构

Cook-Torrance 把材质 BRDF 拆成**漫反射 + 高光**两项：

$$
f_r = f_{\text{diffuse}} + f_{\text{specular}} = \frac{\rho_d}{\pi} + \frac{DFG}{4\,(\vec{n}\cdot\vec{l})\,(\vec{n}\cdot\vec{v})}
$$

高光项 $f_{\text{specular}}$ 是三个物理因子的乘积除以归一化分母：

$$
f_{\text{specular}} = \frac{D(\vec{h})\; G(\vec{l}, \vec{v}, \vec{h})\; F(\vec{v}, \vec{h})}{4\,(\vec{n}\cdot\vec{l})\,(\vec{n}\cdot\vec{v})}
$$

- **$D$（法线分布函数，NDF）**：微面朝向的分布——决定高光的「宽度与形状」。
- **$G$（几何遮蔽项）**：微面互相遮挡——决定高光的「边缘暗化」（暗角）。
- **$F$（菲涅尔项）**：微面的反射率——决定高光的「强度与颜色」。
- 分母：几何归一化（投影因子）。

**「DFG 三件套」** 是 Cook-Torrance 的全部——每个因子回答一个物理问题：微面朝哪、互相挡不挡、反射多少。<span class="marginnote">Cook-Torrance 的每一项都有清晰的物理含义：$D$ 管「<strong>有多少微面恰好对准半程向量</strong>」（对准了才反射到视线）、$G$ 管「<strong>这些微面有没有被邻居挡住</strong>」（挡住就贡献不了）、$F$ 管「<strong>反射的那部分光有多少能量</strong>」。三个因子相乘，就是「对准 × 没被挡 × 反射率」的物理正确组合。</span>

## 2 法线分布函数 D：高光的形状

**法线分布函数（Normal Distribution Function）** $D(\vec{h})$ 描述「朝向为 $\vec{h}$ 的微面占多少比例」——只有朝向恰好等于半程向量 $\vec{h}$ 的微面才能把光反射到视线。常用的是 **GGX（Trowbridge-Reitz）**：

$$
D_{\text{GGX}}(\vec{h}) = \frac{\alpha^2}{\pi\left((\vec{n}\cdot\vec{h})^2(\alpha^2 - 1) + 1\right)^2}
$$

$\alpha$ 是**粗糙度**参数（$\alpha = \text{roughness}^2$）。$\alpha$ 小（光滑）→ $D$ 在 $\vec{n}$ 附近尖锐 → 高光窄亮；$\alpha$ 大（粗糙）→ $D$ 平缓 → 高光宽暗。<span class="marginnote">GGX 相对 Blinn-Phong 的优势：<strong>GGX 有「长尾」（高光衰减慢）</strong>——真实金属的高光在高光点周围有一个微亮的拖尾（因微面里少数「极度倾斜」的贡献），GGX 的长尾精确复现了它，而 Blinn-Phong 的指数衰减太快、高光「太干净」。这也是 PBR 高光比老模型更真实的原因之一。</span>

## 3 几何遮蔽项 G：微面互挡

**几何遮蔽项（Geometry term）** $G(\vec{l}, \vec{v}, \vec{h})$ 处理两个效应：

- **遮蔽（shadowing）**：出射方向的微面被别的微面挡住。
- **掩蔽（masking）**：入射方向的微面被别的微面挡住。

$G$ 的值在 $[0, 1]$，粗糙表面 $G$ 小（微面乱、互相挡得多）。常用 **Smith 模型**：

$$
G(\vec{l}, \vec{v}, \vec{h}) = G_1(\vec{l})\, G_1(\vec{v}), \qquad G_1(x) = \frac{2(\vec{n}\cdot\vec{x})}{(\vec{n}\cdot\vec{x}) + \sqrt{\alpha^2 + (1-\alpha^2)(\vec{n}\cdot\vec{x})^2}}
$$

**辨析｜易错点：** 几何项是 DFG 三件套里最容易被新手漏掉的——漏掉 $G$ 的材质在掠射角（光线擦着表面）时**过亮**，因为被忽略的微面遮挡本该让高光暗下来。$G$ 的「掠射暗化」是真实材质的重要特征（金属边缘的暗角）。<span class="marginnote">$G$ 的「掠射暗化」是 PBR 的真实感来源之一：<strong>光线擦着粗糙表面时，微面互相遮挡严重，反射光显著变暗</strong>——这形成了金属圆球边缘的暗环。忽略 $G$ 的粗糙材质会在边缘「发光」，一眼假。理解 $G$ = 理解 PBR 的「边缘行为」。</span>

## 4 公式解析：Cook-Torrance 为什么长这样

把「微表面物理」翻译成 DFG 公式，一步步看每个因子怎么进来：

$$
f_{\text{specular}} = \frac{D(\vec{h})\, G(\vec{l},\vec{v},\vec{h})\, F(\vec{v},\vec{h})}{4(\vec{n}\cdot\vec{l})(\vec{n}\cdot\vec{v})}
$$

- **第一步，$D$ 进分子**：只有朝向 $\vec{h}$（半程向量）的微面能反射光到视线——$D$ 给出「这类微面占多少」。但 $D$ 是「单位面积微面的朝向分布」，需要积分归一（立体角域）。
- **第二步，$F$ 进分子**：这些微面的反射率 $F$——菲涅尔项（上一节），垂直入射取 $F_0$、掠射趋 1。
- **第三步，$G$ 进分子**：微面被遮挡的折扣——只有「没被挡」的微面真正贡献。
- **第四步，分母归一**：$4(\vec{n}\cdot\vec{l})(\vec{n}\cdot\vec{v})$ 是「立体角域 ↔ 面积域」与「辐射亮度 ↔ 辐照度」的雅可比转换——让 $f_r$ 成为合法的 BRDF（单位 sr⁻¹）。

## 5 Cook-Torrance 与 Blinn-Phong 的对比

| | Blinn-Phong | Cook-Torrance |
| --- | --- | --- |
| 高光形状 | 指数 $(\vec{n}\cdot\vec{h})^p$ | 物理 $D$（GGX 长尾） |
| 菲涅尔 | 无（$k_s$ 常数） | $F$（掠射变强） |
| 几何遮蔽 | 无 | $G$（掠射暗化） |
| 能量守恒 | 不保证 | 有 $G$ 与分母保证 |
| 参数含义 | 经验 $p$ | 物理粗糙度 $\alpha$ |

Blinn-Phong 是 Cook-Torrance 的「特例退化」——取 $D$ 为 Blinn-Phong 形状、忽略 $G$ 与 $F$，就能还原。PBR 的升级就是「把经验项换成物理项」：**形状用 $D$、强度用 $F$、边缘用 $G$**。<span class="marginnote">「PBR = Blinn-Phong 的物理化」是最直接的理解路径：<strong>Blinn-Phong 的高光指数 $p$ 换成 GGX 的 $D$（形状），常数 $k_s$ 换成 $F$（随角度变），补上 $G$（边缘暗化）</strong>——三个物理项替代三个经验参数。这也是为什么学会 Blinn-Phong 再学 PBR 几乎无缝：不是学新东西，是把旧的每一项「升级成对的」。</span>

## 6 Cook-Torrance 的完整材质模型

把漫反射与高光按能量分配组合成最终 BRDF：

$$
f_r = \underbrace{(1 - F(\vec{v},\vec{h}))\, \frac{\rho_d}{\pi}}_{\text{漫反射（菲涅尔折扣）}} + \underbrace{\frac{D\, G\, F}{4(\vec{n}\cdot\vec{l})(\vec{n}\cdot\vec{v})}}_{\text{高光}}
$$

漫反射项乘 $(1-F)$——菲涅尔反射掉的光不能进漫反射（能量守恒）。金属（$F_0$ 高）漫反射几乎为零，电介质（$F_0$ 低）漫反射为主。这就是「金属度 slider」的物理。<span class="marginnote">完整 PBR 材质 = Cook-Torrance BRDF + 纹理参数（base color、roughness、metallic、normal、AO）——<strong>现代引擎（UE、Unity、Disney）的材质系统全部围绕这个公式搭建</strong>。从 Cook-Torrance 出发，材质参数的每个旋钮都有 DFG 的对应：roughness 调 $D$、metallic 调 $F_0$ 与漫反射、AO 是后处理。理解 DFG，就理解了整个 PBR 材质体系。</span>

## 7 Cook-Torrance 的「参数 ↔ 视觉」速查

把 Cook-Torrance 的调参直觉压成速查，方便「调材质时一眼找到」：

- **roughness（粗糙度）**：调 $D$ 的峰宽——大 → 高光宽暗（哑光）、小 → 高光窄亮（光滑）。
- **metallic（金属度）**：调 $F_0$ 与漫反射比例——1 → 无漫反射、彩色高光（金属）；0 → 漫反射为主、灰白高光（电介质）。
- **specular / F0**：垂直入射反射率——金属用彩色、电介质用 0.04。
- **normal map**：扰动 $D$ 的输入法线——给高光「形状」。
- **AO**：乘在环境光（间接）部分——「角落暗」。

**一句话记忆**：「粗糙管形状、金属管颜色、法线管细节、AO 管角落」——这四句覆盖了 PBR 材质的大部分调参。

**常见「调崩」的场景与对策**：

- 高光「糊成一片」→ roughness 太大或 normal map 弱。
- 高光「全白塑料感」→ metallic = 0 且 F0 用 0.04（电介质），要金属感调 metallic。
- 材质「发黑」→ AO 乘错了层（乘到最终颜色）或环境光太弱。
- 边缘「发光」→ $G$ 缺失或 roughness 太小。

**「会调 Cook-Torrance = 会调 PBR 材质」**——这张速查把「DFG 数学」翻译成「旋钮语言」，让你从「试参数」变成「诊材质」。

## 8 小结

- **微表面模型**：表面由无数微镜面组成，宏观 BRDF = 微面的统计平均。
- **Cook-Torrance** $f_r = f_{\text{diffuse}} + \frac{DFG}{4(\vec{n}\cdot\vec{l})(\vec{n}\cdot\vec{v})}$——PBR 高光标准。
- **$D$（法线分布）** 管高光形状：GGX 有长尾，比 Blinn-Phong 更真实。
- **$G$（几何遮蔽）** 管掠射暗化，漏掉会「边缘发光」。
- **$F$（菲涅尔）** 管反射强度与颜色，与漫反射按 $(1-F)$ 分配能量。
- Blinn-Phong 是 Cook-Torrance 的特例退化；PBR = 把经验项物理化。

在下一节，我们把 DFG 三个因子逐一解剖——**微表面模型的各项：法线分布、几何遮蔽与 Fresnel**。
